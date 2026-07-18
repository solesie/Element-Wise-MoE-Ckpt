import json
from pathlib import Path

import pytest

from megatron.proposal.telemetry.io_backlog import (
    WriteObject,
    _reset_io_backlog_trace_for_tests,
    close_io_backlog_trace,
    io_backlog_enabled,
    record_write_batch,
)


@pytest.fixture(autouse=True)
def _isolated_tracer():
    _reset_io_backlog_trace_for_tests()
    yield
    _reset_io_backlog_trace_for_tests()


def _trace(tmp_path: Path, rank: int = 0) -> Path:
    paths = list(tmp_path.glob(f"waf_rank{rank:05d}_pid*_*.jsonl"))
    assert len(paths) == 1
    return paths[0]


def _records(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disabled_trace_does_not_consume_writes(monkeypatch, tmp_path):
    monkeypatch.delenv("PROPOSAL_IO_BACKLOG_TRACE_DIR", raising=False)

    def must_not_iterate():
        raise AssertionError("disabled tracing consumed its input")
        yield WriteObject("lfsy", "append")

    assert not io_backlog_enabled()
    assert not record_write_batch(must_not_iterate(), step=1)
    assert list(tmp_path.iterdir()) == []


def test_compact_write_schema_and_global_expert_mapping(monkeypatch, tmp_path):
    monkeypatch.setenv("PROPOSAL_IO_BACKLOG_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("PROPOSAL_FDP_PP_RANK", "0")
    monkeypatch.setenv("PROPOSAL_FDP_PP_SIZE", "2")
    monkeypatch.setenv("PROPOSAL_FDP_EP_RANK", "2")
    monkeypatch.setenv("PROPOSAL_FDP_EP_SIZE", "4")
    monkeypatch.setenv("PROPOSAL_FDP_EXPERTS_PER_LAYER", "128")
    monkeypatch.setenv("PROPOSAL_CP_TARGET", "test_dram")
    monkeypatch.setenv("PROPOSAL_IO_BACKEND", "file")

    assert record_write_batch(
        (
            WriteObject(
                "lfsy", "append", layer=5, local_expert=1,
                io_bytes=8192, offset=65536,
            ),
            WriteObject(
                "lfcl", "replace", layer=5, local_expert=2,
                io_bytes=1048576,
            ),
        ),
        step=51,
        timestamp=987654321,
    )
    close_io_backlog_trace()

    records = _records(_trace(tmp_path, rank=7))
    metadata, batch, footer = records
    assert metadata["schema"] == "proposal-waf-write-v1"
    assert metadata["granularity"] == "logical write object, never individual SQE"
    assert metadata["write_fields"] == [
        "kind", "op", "layer", "global_expert", "io_bytes", "offset",
    ]
    assert metadata["global_expert_base"] == 64
    assert metadata["max_io_bytes"] == 256 * 1024
    assert metadata["queue_max_batches"] == 256
    assert metadata["source_cp_target"] == "test_dram"
    assert metadata["source_io_backend"] == "file"
    assert batch == {
        "schema": "proposal-waf-write-v1",
        "record": "write_batch",
        "mono_ns": 987654321,
        "batch_seq": 0,
        "step": 51,
        "writes": [
            ["y", "a", 5, 65, 8192, 65536],
            ["c", "r", 5, 66, 1048576, 0],
        ],
    }
    assert footer["status"] == "ok"
    assert footer["batches"] == 1
    assert footer["writes"] == 2


def test_close_drains_queue_and_reinit_preserves_trace(monkeypatch, tmp_path):
    monkeypatch.setenv("PROPOSAL_IO_BACKLOG_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "2")
    for step in range(100):
        assert record_write_batch(
            [WriteObject("lfsy", "append", io_bytes=4096)], step=step,
        )
    close_io_backlog_trace()
    first = set(tmp_path.glob("waf_rank00002_pid*_*.jsonl"))
    assert len(first) == 1
    assert _records(next(iter(first)))[-1]["writes"] == 100

    assert record_write_batch(
        [WriteObject("nefc", "replace", io_bytes=65536)], step=101,
    )
    close_io_backlog_trace()
    traces = set(tmp_path.glob("waf_rank00002_pid*_*.jsonl"))
    assert len(traces) == 2
    assert first < traces
