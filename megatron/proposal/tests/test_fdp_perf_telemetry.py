import json
from pathlib import Path


def _reset(monkeypatch):
    from megatron.proposal.telemetry import fdp_perf

    monkeypatch.setattr(fdp_perf, "_ENABLED", True)
    monkeypatch.setattr(fdp_perf, "_RECORDS", [])
    monkeypatch.setattr(fdp_perf, "_INIT_RUNTIME", None)
    monkeypatch.setattr(fdp_perf, "_START_RUNTIME", None)
    monkeypatch.setattr(fdp_perf, "_END_RUNTIME", None)
    return fdp_perf


def test_measured_call_records_wall_and_thread_cpu(monkeypatch):
    perf = _reset(monkeypatch)
    wall = iter((100, 175))
    cpu = iter((20, 45))
    monkeypatch.setattr(perf, "monotonic_raw_ns", lambda: next(wall))
    monkeypatch.setattr(perf.time, "thread_time_ns", lambda: next(cpu))

    assert perf.measured_call("phase", 7, lambda: 9, marker="x") == 9
    record = perf._RECORDS[0]
    assert record["phase"] == "phase"
    assert record["step"] == 7
    assert record["wall_ns"] == 75
    assert record["thread_cpu_ns"] == 25
    assert record["marker"] == "x"


def test_disabled_path_does_not_read_clocks_or_allocate_records(monkeypatch):
    from megatron.proposal.telemetry import fdp_perf

    monkeypatch.setattr(fdp_perf, "_ENABLED", False)
    monkeypatch.setattr(fdp_perf, "_RECORDS", [])
    monkeypatch.setattr(
        fdp_perf,
        "monotonic_raw_ns",
        lambda: (_ for _ in ()).throw(AssertionError("clock read")),
    )

    assert fdp_perf.measured_call("off", 1, lambda: 17) == 17
    assert fdp_perf._RECORDS == []


def test_mark_writer_done_keeps_metadata_commit(monkeypatch):
    perf = _reset(monkeypatch)
    wall = iter((1000, 1400))
    cpu = iter((10, 30))
    monkeypatch.setattr(perf, "monotonic_raw_ns", lambda: next(wall))
    monkeypatch.setattr(perf.time, "thread_time_ns", lambda: next(cpu))

    from megatron.proposal.storage import proposal_io

    monkeypatch.setattr(
        proposal_io,
        "fdp_perf_runtime_stats",
        lambda: {
            "last_metadata_commit": {
                "valid": True,
                "iter": 11,
                "start_mono_raw_ns": 1100,
                "end_mono_raw_ns": 1350,
            }
        },
    )
    assert perf.measured_mark_writer_done(
        step=11,
        writer_kind="expert",
        site="test",
        fn=lambda: True,
    )
    record = perf._RECORDS[0]
    assert record["wall_ns"] == 400
    assert record["did_metadata_flush"] is True
    assert record["metadata_commit"]["end_mono_raw_ns"] == 1350


def test_dump_is_rank_distinct_and_atomic(monkeypatch, tmp_path: Path):
    perf = _reset(monkeypatch)
    monkeypatch.setenv("PROPOSAL_FDP_RANK", "3")
    monkeypatch.setattr(perf, "capture_runtime_end", lambda: None)
    perf._RECORDS.append({"phase": "x", "step": 2})

    path = perf.dump(tmp_path, "run.")
    assert path == tmp_path / "run.fdp_perf_rank00003.json"
    payload = json.loads(path.read_text())
    assert payload["schema"] == "proposal-fdp-perf-v1"
    assert payload["rank"] == 3
    assert payload["clock"] == "CLOCK_MONOTONIC_RAW"
    assert payload["records"] == [{"phase": "x", "step": 2}]
    assert not list(tmp_path.glob("*.tmp.*"))
