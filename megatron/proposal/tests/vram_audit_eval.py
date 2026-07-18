"""Fail-closed evaluator for the proposal VRAM/allocation audit.

This evaluator deliberately uses two different measurements:

* long proposal/baseline runs detect retained allocator growth; and
* short, exact PyTorch allocator-history regions attribute hot-path churn.

Oracle regions are not part of this audit.  The first H refresh is treated as
the allocator warm-up refresh; every later refresh must reuse the warmed
allocator state without obtaining or releasing a CUDA segment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


AUDIT_SCHEMA = 2
STEADY_MIN_ITER = 11
MIN_REGULAR_ITERS = 100
MIN_EXACT_STEADY_REGIONS = 3
MIN_WARM_H_REFRESH_REGIONS = 3
R1_TOLERANCE_BYTES = 50_000_000
R3_TOLERANCE_BYTES = 100_000_000
# One 2 MiB allocator block over the full audit window is the largest retained
# growth treated as measurement granularity rather than a trend.
LONG_RETAINED_TOLERANCE_BYTES = 2 * 1024 * 1024

_ANCHOR_LABELS = {
    "pre_anchor",
    "inside_anchor_fd_peak",
    "post_anchor",
    "allocator_trace_h_refresh",
}
_TRACE_RECONCILIATION_KEYS = {
    "alloc_events_match",
    "free_events_match",
    "device_alloc_events_match",
    "device_free_events_match",
}
_H_REFRESH_RECONCILIATION_KEYS = {
    "device_alloc_events_match",
    "device_free_events_match",
}
_PROPOSAL_ACTIONS = {
    "alloc",
    "free_requested",
    "free_completed",
    "segment_alloc",
    "segment_free",
    "segment_map",
    "segment_unmap",
}
_SEGMENT_ACTIONS = {
    "segment_alloc",
    "segment_free",
    "segment_map",
    "segment_unmap",
}
_RETAINED_STATS = {
    "allocated": "allocated_bytes.all.current",
    "reserved": "reserved_bytes.all.current",
    "inactive_split": "inactive_split_bytes.all.current",
}
_TRACE_STATS_KEYS = {
    "allocated_bytes.all.current",
    "reserved_bytes.all.current",
    "inactive_split_bytes.all.current",
    "allocation.all.allocated",
    "allocation.all.freed",
    "num_device_alloc",
    "num_device_free",
    "num_alloc_retries",
    "num_ooms",
}


class AuditDataError(ValueError):
    """The audit cannot be evaluated without guessing missing information."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def load_audit(path: str) -> List[Dict[str, Any]]:
    """Load one JSONL audit, rejecting absent, empty, or malformed input."""

    if not os.path.isfile(path):
        raise AuditDataError(f"audit log does not exist: {path}")
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditDataError(
                    f"{path}:{line_number}: malformed JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise AuditDataError(
                    f"{path}:{line_number}: record must be a JSON object"
                )
            records.append(record)
    if not records:
        raise AuditDataError(f"audit log is empty: {path}")
    return records


def _int(record: Mapping[str, Any], key: str, context: str) -> int:
    if key not in record or isinstance(record[key], bool):
        raise AuditDataError(f"{context}: missing integer field {key!r}")
    try:
        value = int(record[key])
    except (TypeError, ValueError) as exc:
        raise AuditDataError(f"{context}: invalid integer field {key!r}") from exc
    return value


def _float(record: Mapping[str, Any], key: str, context: str) -> float:
    if key not in record or isinstance(record[key], bool):
        raise AuditDataError(f"{context}: missing numeric field {key!r}")
    try:
        value = float(record[key])
    except (TypeError, ValueError) as exc:
        raise AuditDataError(f"{context}: invalid numeric field {key!r}") from exc
    if not math.isfinite(value):
        raise AuditDataError(f"{context}: non-finite numeric field {key!r}")
    return value


def _dict(record: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise AuditDataError(f"{context}: missing object field {key!r}")
    return value


def _rank_groups(records: Sequence[Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for index, record in enumerate(records):
        rank = _int(record, "rank", f"record {index}")
        groups.setdefault(rank, []).append(dict(record))
    return groups


def _records_with_label(
    records: Sequence[Mapping[str, Any]], label: str
) -> List[Dict[str, Any]]:
    return [dict(record) for record in records if record.get("label") == label]


def _unique_by_iter(
    records: Sequence[Mapping[str, Any]], label: str, context: str
) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    for record in _records_with_label(records, label):
        step = _int(record, "iter", f"{context}/{label}")
        if step in result:
            raise AuditDataError(
                f"{context}: duplicate {label!r} record for iteration {step}"
            )
        result[step] = record
    return result


def _validate_log_metadata(
    records: Sequence[Mapping[str, Any]], expected_kind: str, context: str
) -> Tuple[Mapping[str, Any], str, str, str, str]:
    """Validate schema and invariant metadata on every record."""

    invariant: Tuple[Mapping[str, Any], str, str, str, str] | None = None
    for index, record in enumerate(records):
        record_context = f"{context}/record {index}"
        if _int(record, "audit_schema", record_context) != AUDIT_SCHEMA:
            raise AuditDataError(
                f"{record_context}: audit_schema must be {AUDIT_SCHEMA}"
            )
        kind = str(record.get("audit_run_kind", ""))
        if kind != expected_kind:
            raise AuditDataError(
                f"{record_context}: audit_run_kind={kind!r}, expected {expected_kind!r}"
            )
        config = _dict(record, "audit_config", record_context)
        backend = str(record.get("allocator_backend", ""))
        allocator_config = str(record.get("allocator_config", ""))
        torch_version = str(record.get("torch_version", ""))
        cuda_version = str(record.get("cuda_version", ""))
        if not backend or backend.lower() == "unknown":
            raise AuditDataError(f"{record_context}: allocator backend is unknown")
        current = (config, backend, allocator_config, torch_version, cuda_version)
        if invariant is None:
            invariant = current
        elif current != invariant:
            raise AuditDataError(f"{record_context}: run metadata changed within one log")
    assert invariant is not None
    return invariant


def _excluded_steps(records: Sequence[Mapping[str, Any]]) -> Tuple[set[int], set[int]]:
    anchor_steps: set[int] = set()
    oracle_steps: set[int] = set()
    for record in records:
        label = str(record.get("label", ""))
        if label in _ANCHOR_LABELS:
            anchor_steps.add(_int(record, "iter", label))
        if "oracle" in label.lower():
            oracle_steps.add(_int(record, "iter", label))
    return anchor_steps, oracle_steps


def _regular_iter_ends(
    records: Sequence[Mapping[str, Any]], context: str
) -> Dict[int, Dict[str, Any]]:
    iter_ends = _unique_by_iter(records, "iter_end", context)
    anchors, oracle = _excluded_steps(records)
    return {
        step: record
        for step, record in iter_ends.items()
        if step >= STEADY_MIN_ITER and step not in anchors and step not in oracle
    }


def _median(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise AuditDataError("cannot take median of an empty series")
    return float(statistics.median(materialized))


def _linear_slope(xs: Sequence[int], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise AuditDataError("linear slope needs at least two aligned samples")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise AuditDataError("linear slope iteration span is zero")
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def _record_metric_bytes(record: Mapping[str, Any], metric: str, context: str) -> float:
    if metric == "allocated":
        return _float(record, "alloc_mb", context) * 1_000_000.0
    if metric == "reserved":
        return _float(record, "reserved_mb", context) * 1_000_000.0
    if metric == "inactive_split":
        return float(_int(record, "inactive_split_bytes", context))
    if metric == "waste":
        return (
            _float(record, "reserved_mb", context)
            - _float(record, "alloc_mb", context)
        ) * 1_000_000.0
    raise AssertionError(metric)


def _trace_coverage_errors(record: Mapping[str, Any], region: str) -> List[str]:
    errors: List[str] = []
    context = f"{region}@{record.get('iter', '?')}"
    if _int(record, "allocator_trace_schema", context) != AUDIT_SCHEMA:
        errors.append(f"{context}: allocator_trace_schema != {AUDIT_SCHEMA}")
    if record.get("region") != region:
        errors.append(f"{context}: region={record.get('region')!r}")
    if record.get("coverage_complete") is not True:
        errors.append(f"{context}: coverage_complete is not true")
    if record.get("history_truncated") is not False:
        errors.append(f"{context}: history is truncated or unspecified")
    if record.get("allocator_trace_error"):
        errors.append(f"{context}: allocator_trace_error={record['allocator_trace_error']}")
    if _int(record, "snapshot_marker_count", context) != 2:
        errors.append(f"{context}: snapshot marker count is not 2")
    missing = record.get("memory_stats_missing_keys")
    if not isinstance(missing, list) or missing:
        errors.append(f"{context}: memory stats keys missing: {missing!r}")
    skipped = record.get("skipped_actions")
    expected_skipped = (
        {"free_completed"}
        if region == "steady"
        else {"alloc", "free_requested", "free_completed"}
    )
    if not isinstance(skipped, list) or set(skipped) != expected_skipped:
        errors.append(
            f"{context}: skipped_actions={skipped!r}, expected {sorted(expected_skipped)}"
        )
    reconciliation = _dict(record, "reconciliation", context)
    required_reconciliation = (
        _TRACE_RECONCILIATION_KEYS
        if region == "steady"
        else _H_REFRESH_RECONCILIATION_KEYS
    )
    missing_reconciliation = required_reconciliation - set(reconciliation)
    if missing_reconciliation:
        errors.append(
            f"{context}: reconciliation keys missing: {sorted(missing_reconciliation)}"
        )
    for key in sorted(required_reconciliation):
        if reconciliation.get(key) is not True:
            errors.append(f"{context}: reconciliation {key} is not true")
    if region == "h_refresh":
        unexpected = {"alloc_events_match", "free_events_match"} & set(reconciliation)
        if unexpected:
            errors.append(
                f"{context}: skipped alloc/free reconciliation was unexpectedly emitted"
            )
    trace_events = _int(record, "trace_events", context)
    history_limit = _int(record, "history_max_entries", context)
    if history_limit <= trace_events:
        errors.append(f"{context}: trace reached its history capacity")
    actions = _dict(record, "action_counts", context)
    _dict(record, "action_bytes", context)
    try:
        counted_events = sum(int(value) for value in actions.values())
    except (TypeError, ValueError) as exc:
        raise AuditDataError(f"{context}: invalid action count") from exc
    if counted_events != trace_events:
        errors.append(
            f"{context}: action count sum={counted_events}, trace_events={trace_events}"
        )
    if int(actions.get("snapshot", 0)) != 2:
        errors.append(f"{context}: action_counts snapshot is not 2")
    _dict(record, "proposal_action_counts", context)
    _dict(record, "proposal_action_bytes", context)
    start_stats = _dict(record, "start_stats", context)
    end_stats = _dict(record, "end_stats", context)
    stats_delta = _dict(record, "stats_delta", context)
    for key in sorted(_TRACE_STATS_KEYS):
        if key not in start_stats or key not in end_stats or key not in stats_delta:
            errors.append(f"{context}: incomplete start/end/delta stats for {key}")
            continue
        expected_delta = int(end_stats[key]) - int(start_stats[key])
        if int(stats_delta[key]) != expected_delta:
            errors.append(
                f"{context}: stats_delta {key}={stats_delta[key]}, "
                f"expected {expected_delta}"
            )
    device_alloc_actions = int(actions.get("segment_alloc", 0)) + int(
        actions.get("segment_map", 0)
    )
    device_free_actions = int(actions.get("segment_free", 0)) + int(
        actions.get("segment_unmap", 0)
    )
    if device_alloc_actions != int(stats_delta.get("num_device_alloc", -1)):
        errors.append(f"{context}: device allocation action/stat counts disagree")
    if device_free_actions != int(stats_delta.get("num_device_free", -1)):
        errors.append(f"{context}: device free action/stat counts disagree")
    if region == "steady":
        if int(actions.get("alloc", 0)) != int(
            stats_delta.get("allocation.all.allocated", -1)
        ):
            errors.append(f"{context}: allocation action/stat counts disagree")
        if int(actions.get("free_requested", 0)) != int(
            stats_delta.get("allocation.all.freed", -1)
        ):
            errors.append(f"{context}: free action/stat counts disagree")
    return errors


def _eval_r1(
    proposal: Mapping[int, Mapping[str, Any]],
    baseline: Mapping[int, Mapping[str, Any]],
) -> CheckResult:
    steps = sorted(set(proposal) & set(baseline))
    actual = _median(
        (_float(proposal[s], "alloc_mb", f"proposal iter {s}")
         - _float(baseline[s], "alloc_mb", f"baseline iter {s}"))
        * 1_000_000.0
        for s in steps
    )
    expected = _median(
        _float(proposal[s], "tracked_total_mb", f"proposal iter {s}")
        * 1_000_000.0
        for s in steps
    )
    if expected <= 0:
        return CheckResult("R1", False, "tracked proposal allocation is absent or zero")
    difference = actual - expected
    ok = abs(difference) <= R1_TOLERANCE_BYTES
    return CheckResult(
        "R1",
        ok,
        f"steady allocated delta={actual / 1e6:.2f} MB, "
        f"tracked={expected / 1e6:.2f} MB, diff={difference / 1e6:+.2f} MB "
        f"(limit +/-{R1_TOLERANCE_BYTES / 1e6:.0f} MB)",
    )


def _eval_exact_steady(
    records: Sequence[Mapping[str, Any]], regular_steps: set[int]
) -> CheckResult:
    traces = _unique_by_iter(records, "allocator_trace_steady", "proposal")
    errors: List[str] = []
    if len(traces) < MIN_EXACT_STEADY_REGIONS:
        errors.append(
            f"only {len(traces)} exact steady regions; need {MIN_EXACT_STEADY_REGIONS}"
        )
    for step, trace in sorted(traces.items()):
        if step not in regular_steps:
            errors.append(f"steady trace iter {step} is not a regular steady iteration")
        errors.extend(_trace_coverage_errors(trace, "steady"))
        counts = _dict(trace, "proposal_action_counts", f"steady@{step}")
        action_bytes = _dict(trace, "proposal_action_bytes", f"steady@{step}")
        global_counts = _dict(trace, "action_counts", f"steady@{step}")
        global_bytes = _dict(trace, "action_bytes", f"steady@{step}")
        for action in sorted(_PROPOSAL_ACTIONS):
            if int(global_counts.get(action, 0)) != 0:
                errors.append(
                    f"steady@{step}: global {action} count={global_counts.get(action)}"
                )
            if int(global_bytes.get(action, 0)) != 0:
                errors.append(
                    f"steady@{step}: global {action} bytes={global_bytes.get(action)}"
                )
            if int(counts.get(action, 0)) != 0:
                errors.append(
                    f"steady@{step}: proposal {action} count={counts.get(action)}"
                )
            if int(action_bytes.get(action, 0)) != 0:
                errors.append(
                    f"steady@{step}: proposal {action} bytes={action_bytes.get(action)}"
                )

    summaries = _records_with_label(records, "alloc_trace_summary")
    if len(summaries) != 1:
        errors.append(f"need exactly one alloc_trace_summary, found {len(summaries)}")
    else:
        summary = summaries[0]
        if summary.get("allocator_history_enabled") is not True:
            errors.append("allocator history was not enabled")
        expected = summary.get("expected_steady_iters")
        traced = summary.get("traced_steady_iters")
        if not isinstance(expected, list) or len(set(expected)) < MIN_EXACT_STEADY_REGIONS:
            errors.append("alloc_trace_summary has fewer than three expected steady iters")
        if not isinstance(traced, list) or sorted(set(traced)) != sorted(traces):
            errors.append("alloc_trace_summary steady iteration list disagrees with traces")
        elif isinstance(expected, list) and not set(expected).issubset(set(traced)):
            errors.append("one or more configured steady trace iterations were not captured")

    detail = (
        f"{len(traces)} exact regions, zero proposal-attributed alloc/free/segment actions"
        if not errors
        else "; ".join(errors[:12])
    )
    return CheckResult("R2/steady exact", not errors, detail)


def _eval_h_refresh(records: Sequence[Mapping[str, Any]]) -> CheckResult:
    traces = _unique_by_iter(records, "allocator_trace_h_refresh", "proposal")
    errors: List[str] = []
    ordered = sorted(traces.items())
    warm = ordered[1:]
    if len(warm) < MIN_WARM_H_REFRESH_REGIONS:
        errors.append(
            f"only {len(warm)} warm H-refresh regions after excluding first; "
            f"need {MIN_WARM_H_REFRESH_REGIONS}"
        )
    for _step, trace in ordered:
        errors.extend(_trace_coverage_errors(trace, "h_refresh"))

    trace_steps = set(traces)
    for label in ("pre_anchor", "inside_anchor_fd_peak", "post_anchor"):
        label_steps = set(_unique_by_iter(records, label, "proposal"))
        if label_steps != trace_steps:
            errors.append(
                f"{label} steps {sorted(label_steps)} do not match traced H refreshes "
                f"{sorted(trace_steps)}"
            )

    for step, trace in warm:
        actions = _dict(trace, "action_counts", f"h_refresh@{step}")
        for action in sorted(_SEGMENT_ACTIONS):
            if int(actions.get(action, 0)) != 0:
                errors.append(
                    f"h_refresh@{step}: global {action} count={actions.get(action)}"
                )
        delta = _dict(trace, "stats_delta", f"h_refresh@{step}")
        for name, key in _RETAINED_STATS.items():
            if key not in delta:
                errors.append(f"h_refresh@{step}: missing stats_delta {key}")
                continue
            growth = int(delta[key])
            if growth > 0:
                errors.append(
                    f"h_refresh@{step}: retained {name} grew {growth / 2**20:.3f} MiB"
                )
        for key in ("num_device_alloc", "num_device_free", "num_alloc_retries", "num_ooms"):
            if key not in delta:
                errors.append(f"h_refresh@{step}: missing stats_delta {key}")
            elif int(delta[key]) != 0:
                errors.append(f"h_refresh@{step}: {key} delta={delta[key]}")

    if ordered:
        summaries = _records_with_label(records, "alloc_trace_summary")
        if len(summaries) == 1:
            traced = summaries[0].get("traced_h_refresh_iters")
            if not isinstance(traced, list) or sorted(set(traced)) != sorted(traces):
                errors.append("alloc_trace_summary H-refresh list disagrees with traces")

    detail = (
        f"cold={ordered[0][0] if ordered else 'none'}, warm={[s for s, _ in warm]}; "
        "no warm segment activity or retained allocator growth"
        if not errors
        else "; ".join(errors[:12])
    )
    return CheckResult("H-refresh reuse", not errors, detail)


def _eval_r3(
    records: Sequence[Mapping[str, Any]], regular: Mapping[int, Mapping[str, Any]]
) -> CheckResult:
    traces = _unique_by_iter(records, "allocator_trace_h_refresh", "proposal")
    warm_steps = sorted(traces)[1:]
    peaks = _unique_by_iter(records, "inside_anchor_fd_peak", "proposal")
    missing = [step for step in warm_steps if step not in peaks]
    if missing:
        return CheckResult("R3", False, f"missing inside-anchor peak records: {missing}")
    training_peak = _median(
        _float(record, "peak_mb", f"regular iter {step}") * 1_000_000.0
        for step, record in regular.items()
    )
    anchor_peak = max(
        (_float(peaks[step], "peak_mb", f"anchor iter {step}") * 1_000_000.0
         for step in warm_steps),
        default=0.0,
    )
    delta = anchor_peak - training_peak
    ok = bool(warm_steps) and delta <= R3_TOLERANCE_BYTES
    return CheckResult(
        "R3",
        ok,
        f"warm anchor peak delta={delta / 1e6:+.2f} MB "
        f"(limit {R3_TOLERANCE_BYTES / 1e6:.0f} MB)",
    )


def _eval_long_fragmentation(
    proposal: Mapping[int, Mapping[str, Any]],
    baseline: Mapping[int, Mapping[str, Any]],
) -> CheckResult:
    steps = sorted(proposal)
    errors: List[str] = []
    missing = [step for step in steps if step not in baseline]
    if missing:
        return CheckResult(
            "R4/long",
            False,
            f"baseline missing {len(missing)} aligned regular iters; first={missing[:5]}",
        )
    span = steps[-1] - steps[0]
    if span <= 0:
        return CheckResult("R4/long", False, "regular iteration span is zero")
    slope_limit = LONG_RETAINED_TOLERANCE_BYTES / span
    reports: List[str] = []
    edge_count = min(10, max(1, len(steps) // 4))
    for metric in ("allocated", "reserved", "inactive_split", "waste"):
        differential = [
            _record_metric_bytes(proposal[step], metric, f"proposal iter {step}")
            - _record_metric_bytes(baseline[step], metric, f"baseline iter {step}")
            for step in steps
        ]
        slope = _linear_slope(steps, differential)
        growth = _median(differential[-edge_count:]) - _median(differential[:edge_count])
        reports.append(
            f"{metric}: slope={slope / 2**20:+.4f} MiB/iter, "
            f"edge-growth={growth / 2**20:+.3f} MiB"
        )
        if slope > slope_limit:
            errors.append(
                f"{metric} baseline-relative slope {slope / 2**20:+.4f} MiB/iter "
                f"> {slope_limit / 2**20:.4f}"
            )
        if growth > LONG_RETAINED_TOLERANCE_BYTES:
            errors.append(
                f"{metric} retained edge growth {growth / 2**20:+.3f} MiB > 2 MiB"
            )

    first, last = steps[0], steps[-1]
    for key in ("num_device_alloc", "num_device_free"):
        proposal_delta = _int(proposal[last], key, f"proposal iter {last}") - _int(
            proposal[first], key, f"proposal iter {first}"
        )
        baseline_delta = _int(baseline[last], key, f"baseline iter {last}") - _int(
            baseline[first], key, f"baseline iter {first}"
        )
        excess = proposal_delta - baseline_delta
        if excess > 0:
            errors.append(f"{key} baseline-relative cumulative excess={excess}")
    for key in ("num_alloc_retries", "num_ooms"):
        delta = _int(proposal[last], key, f"proposal iter {last}") - _int(
            proposal[first], key, f"proposal iter {first}"
        )
        if delta != 0:
            errors.append(f"proposal {key} increased by {delta}")

    return CheckResult(
        "R4/long",
        not errors,
        "; ".join(errors[:10] if errors else reports),
    )


def evaluate_audits(
    proposal_records: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]],
) -> Dict[int, List[CheckResult]]:
    """Evaluate complete proposal and baseline logs, grouped by rank."""

    proposal_meta = _validate_log_metadata(proposal_records, "proposal", "proposal")
    baseline_meta = _validate_log_metadata(baseline_records, "baseline", "baseline")
    if proposal_meta != baseline_meta:
        raise AuditDataError(
            "proposal and baseline model/allocator/Torch/CUDA metadata do not match"
        )

    proposal_ranks = _rank_groups(proposal_records)
    baseline_ranks = _rank_groups(baseline_records)
    if set(proposal_ranks) != set(baseline_ranks):
        raise AuditDataError(
            f"rank sets differ: proposal={sorted(proposal_ranks)}, "
            f"baseline={sorted(baseline_ranks)}"
        )

    output: Dict[int, List[CheckResult]] = {}
    for rank in sorted(proposal_ranks):
        proposal = proposal_ranks[rank]
        baseline = baseline_ranks[rank]
        proposal_regular = _regular_iter_ends(proposal, f"proposal rank {rank}")
        baseline_regular = _regular_iter_ends(baseline, f"baseline rank {rank}")
        count_ok = (
            len(proposal_regular) >= MIN_REGULAR_ITERS
            and len(baseline_regular) >= MIN_REGULAR_ITERS
        )
        results = [
            CheckResult(
                "steady coverage",
                count_ok,
                f"proposal={len(proposal_regular)}, baseline={len(baseline_regular)} "
                f"regular iters (need >= {MIN_REGULAR_ITERS} each)",
            )
        ]
        if not count_ok:
            output[rank] = results
            continue
        results.extend(
            [
                _eval_r1(proposal_regular, baseline_regular),
                _eval_exact_steady(proposal, set(proposal_regular)),
                _eval_h_refresh(proposal),
                _eval_r3(proposal, proposal_regular),
                _eval_long_fragmentation(proposal_regular, baseline_regular),
            ]
        )
        output[rank] = results
    return output


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal", help="proposal VRAM audit JSONL")
    parser.add_argument(
        "--baseline", required=True, help="matching proposal-off baseline audit JSONL"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        proposal = load_audit(args.proposal)
        baseline = load_audit(args.baseline)
        results = evaluate_audits(proposal, baseline)
    except AuditDataError as exc:
        print(f"VRAM AUDIT: FAIL (invalid/incomplete evidence) — {exc}", file=sys.stderr)
        return 2

    all_ok = True
    print(
        f"=== VRAM AUDIT ({len(proposal)} proposal records, "
        f"{len(baseline)} baseline records) ==="
    )
    for rank, rank_results in results.items():
        print(f"--- rank {rank} ---")
        for result in rank_results:
            print(f"{result.name}: {'PASS' if result.ok else 'FAIL'} — {result.detail}")
            all_ok = all_ok and result.ok
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
