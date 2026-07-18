from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

_EVALUATOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "megatron"
    / "proposal"
    / "tests"
    / "vram_audit_eval.py"
)
_SPEC = importlib.util.spec_from_file_location("proposal_vram_audit_eval", _EVALUATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = audit
_SPEC.loader.exec_module(audit)


_CONFIG = {
    "NUM_LAYERS": "8",
    "HIDDEN_SIZE": "512",
    "NUM_GPUS": "2",
    "PP_SIZE": "1",
    "EP_PARALLEL_SIZE": "2",
    "ZERO_STAGE": "1",
}


def _record(kind: str, label: str, step: int, **extra):
    record = {
        "audit_schema": 2,
        "audit_run_kind": kind,
        "audit_config": dict(_CONFIG),
        "allocator_backend": "native",
        "allocator_config": "expandable_segments:True",
        "torch_version": "test",
        "cuda_version": "test",
        "rank": 0,
        "local_rank": 0,
        "label": label,
        "iter": step,
    }
    record.update(extra)
    return record


def _iter_end(kind: str, step: int):
    proposal = kind == "proposal"
    return _record(
        kind,
        "iter_end",
        step,
        alloc_mb=1400.0 if proposal else 1000.0,
        peak_mb=1500.0 if proposal else 1100.0,
        reserved_mb=1800.0 if proposal else 1400.0,
        inactive_split_bytes=128 * 1024 * 1024,
        tracked_total_mb=400.0 if proposal else 0.0,
        num_device_alloc=5,
        num_device_free=2,
        num_alloc_retries=0,
        num_ooms=0,
    )


def _steady_trace(step: int):
    stats = {
        "allocated_bytes.all.current": 1400 * 1000 * 1000,
        "reserved_bytes.all.current": 1800 * 1000 * 1000,
        "inactive_split_bytes.all.current": 128 * 1024 * 1024,
        "allocation.all.allocated": 100,
        "allocation.all.freed": 90,
        "num_device_alloc": 5,
        "num_device_free": 2,
        "num_alloc_retries": 0,
        "num_ooms": 0,
    }
    return _record(
        "proposal",
        "allocator_trace_steady",
        step,
        allocator_trace_schema=2,
        region="steady",
        coverage_complete=True,
        history_truncated=False,
        history_max_entries=1000,
        snapshot_marker_count=2,
        trace_events=2,
        skipped_actions=["free_completed"],
        memory_stats_missing_keys=[],
        reconciliation={
            "alloc_events_match": True,
            "free_events_match": True,
            "device_alloc_events_match": True,
            "device_free_events_match": True,
        },
        action_counts={"snapshot": 2},
        action_bytes={},
        proposal_action_counts={},
        proposal_action_bytes={},
        start_stats=dict(stats),
        end_stats=dict(stats),
        stats_delta={
            "allocated_bytes.all.current": 0,
            "reserved_bytes.all.current": 0,
            "inactive_split_bytes.all.current": 0,
            "allocation.all.allocated": 0,
            "allocation.all.freed": 0,
            "num_device_alloc": 0,
            "num_device_free": 0,
            "num_alloc_retries": 0,
            "num_ooms": 0,
        },
    )


def _h_refresh_trace(step: int):
    start_stats = {
        "allocated_bytes.all.current": 1400 * 1000 * 1000,
        "reserved_bytes.all.current": 1800 * 1000 * 1000,
        "inactive_split_bytes.all.current": 128 * 1024 * 1024,
        "allocation.all.allocated": 100,
        "allocation.all.freed": 90,
        "num_device_alloc": 5,
        "num_device_free": 2,
        "num_alloc_retries": 0,
        "num_ooms": 0,
    }
    end_stats = dict(start_stats)
    end_stats["allocation.all.allocated"] += 200000
    end_stats["allocation.all.freed"] += 200000
    return _record(
        "proposal",
        "allocator_trace_h_refresh",
        step,
        allocator_trace_schema=2,
        region="h_refresh",
        coverage_complete=True,
        history_truncated=False,
        history_max_entries=1000,
        snapshot_marker_count=2,
        trace_events=2,
        skipped_actions=["alloc", "free_requested", "free_completed"],
        memory_stats_missing_keys=[],
        reconciliation={
            "device_alloc_events_match": True,
            "device_free_events_match": True,
        },
        action_counts={"snapshot": 2},
        action_bytes={},
        proposal_action_counts={},
        proposal_action_bytes={},
        start_stats=start_stats,
        end_stats=end_stats,
        stats_delta={
            "allocated_bytes.all.current": 0,
            "reserved_bytes.all.current": 0,
            "inactive_split_bytes.all.current": 0,
            "allocation.all.allocated": 200000,
            "allocation.all.freed": 200000,
            "num_device_alloc": 0,
            "num_device_free": 0,
            "num_alloc_retries": 0,
            "num_ooms": 0,
        },
    )


def _logs():
    proposal = [_iter_end("proposal", step) for step in range(1, 131)]
    baseline = [_iter_end("baseline", step) for step in range(1, 131)]
    steady_steps = [111, 112, 113]
    anchor_steps = [20, 40, 60, 80]
    proposal.extend(_steady_trace(step) for step in steady_steps)
    for step in anchor_steps:
        proposal.extend(
            [
                _record("proposal", "pre_anchor", step, peak_mb=1500.0),
                _record("proposal", "inside_anchor_fd_peak", step, peak_mb=1550.0),
                _record("proposal", "post_anchor", step, peak_mb=1550.0),
                _h_refresh_trace(step),
            ]
        )
    proposal.append(
        _record(
            "proposal",
            "alloc_trace_summary",
            130,
            allocator_trace_schema=2,
            allocator_history_enabled=True,
            expected_steady_iters=steady_steps,
            traced_steady_iters=steady_steps,
            traced_h_refresh_iters=anchor_steps,
            total_regions=7,
        )
    )
    return proposal, baseline


def _results(proposal, baseline):
    return audit.evaluate_audits(proposal, baseline)[0]


def _named(results, name):
    return next(result for result in results if result.name == name)


def test_complete_audit_passes():
    proposal, baseline = _logs()
    results = _results(proposal, baseline)
    assert results
    assert all(result.ok for result in results), results


def test_fewer_than_100_regular_iterations_fails_closed():
    proposal, baseline = _logs()
    proposal = [r for r in proposal if r["label"] != "iter_end" or r["iter"] <= 100]
    baseline = [r for r in baseline if r["label"] != "iter_end" or r["iter"] <= 100]
    result = _named(_results(proposal, baseline), "steady coverage")
    assert not result.ok


@pytest.mark.parametrize(
    ("field", "value"),
    [("coverage_complete", False), ("history_truncated", True)],
)
def test_incomplete_exact_steady_trace_fails(field, value):
    proposal, baseline = _logs()
    trace = next(r for r in proposal if r["label"] == "allocator_trace_steady")
    trace[field] = value
    result = _named(_results(proposal, baseline), "R2/steady exact")
    assert not result.ok


def test_steady_proposal_allocation_fails():
    proposal, baseline = _logs()
    trace = next(r for r in proposal if r["label"] == "allocator_trace_steady")
    trace["proposal_action_counts"]["alloc"] = 1
    trace["proposal_action_bytes"]["alloc"] = 4096
    result = _named(_results(proposal, baseline), "R2/steady exact")
    assert not result.ok
    assert "proposal alloc" in result.detail


def test_three_total_h_refreshes_are_not_three_warm_refreshes():
    proposal, baseline = _logs()
    proposal = [
        r
        for r in proposal
        if not (r["iter"] == 80 and r["label"] in audit._ANCHOR_LABELS)
    ]
    summary = next(r for r in proposal if r["label"] == "alloc_trace_summary")
    summary["traced_h_refresh_iters"] = [20, 40, 60]
    result = _named(_results(proposal, baseline), "H-refresh reuse")
    assert not result.ok
    assert "only 2 warm" in result.detail


@pytest.mark.parametrize(
    ("mutation", "needle"),
    [
        (("action_counts", "segment_map", 1), "segment_map"),
        (("stats_delta", "reserved_bytes.all.current", 2 * 1024 * 1024), "reserved"),
    ],
)
def test_warm_h_refresh_segment_or_retained_growth_fails(mutation, needle):
    proposal, baseline = _logs()
    traces = [r for r in proposal if r["label"] == "allocator_trace_h_refresh"]
    outer, inner, value = mutation
    traces[1][outer][inner] = value
    if outer == "stats_delta":
        traces[1]["end_stats"][inner] += value
    result = _named(_results(proposal, baseline), "H-refresh reuse")
    assert not result.ok
    assert needle in result.detail


def test_baseline_relative_reserved_slope_fails():
    proposal, baseline = _logs()
    for record in proposal:
        if record["label"] == "iter_end":
            record["reserved_mb"] += float(record["iter"])
    result = _named(_results(proposal, baseline), "R4/long")
    assert not result.ok
    assert "reserved baseline-relative slope" in result.detail


def test_malformed_json_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"rank": 0}\nnot-json\n', encoding="utf-8")
    with pytest.raises(audit.AuditDataError, match="malformed JSON"):
        audit.load_audit(str(path))


def test_mismatched_run_metadata_is_rejected():
    proposal, baseline = _logs()
    baseline[0]["audit_config"]["PP_SIZE"] = "2"
    with pytest.raises(audit.AuditDataError, match="metadata"):
        audit.evaluate_audits(proposal, baseline)
