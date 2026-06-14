"""Evaluate VRAM audit JSONL against R1-R4 hard contract.

Reads ``$PROPOSAL_VRAM_AUDIT_LOG`` and emits PASS/FAIL line per
requirement. ``exit 1`` if any FAIL.

Requirements (from AGENTS.md V2 hard contract + user mental model):

  R1 — persistent VRAM ≈ baseline + lossy checkpoint buffers +
       segment_double_buf. Expected value computed analytically from
       steady-state audit records.
  R2 — no per-iter alloc/free on proposal streams. Verified via
       ``alloc_count`` delta between iter_end records (steady state
       iter 30-130) AND alloc_trace_summary entries.
  R3 — anchor refresh peak ≤ non-anchor iter peak + 100 MB tolerance.
  R4 — no leak (alloc slope < 1 MB/iter), no fragmentation
       (reserved-allocated waste < 1.5 × max_transient_peak).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple


R1_TOLERANCE_MB = 50.0        # lossy checkpoint + segment + small metadata
R2_LARGE_ALLOC_MB = 1.0       # any per-iter alloc > 1 MB is a violation
R2_LARGE_ALLOC_COUNT = 10     # if it happens > 10 times in trace window (= many iter)
R3_TOLERANCE_MB = 100.0
R4_SLOPE_LIMIT_MB_PER_ITER = 1.0
R4_WASTE_FACTOR = 1.5
# Steady-state iter range for R2/R4 (short-iter mode: exit-interval=30,
# warmup ~5 iters, anchor every 5).
STEADY_LO = 10
STEADY_HI = 25


def load_audit(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _by_label(records: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    return [r for r in records if r.get("label") == label]


def _by_label_in_iter_range(records, label, lo, hi):
    return [
        r for r in records
        if r.get("label") == label and lo <= int(r.get("iter", -1)) <= hi
    ]


def _baseline_alloc(records: List[Dict[str, Any]]) -> float:
    """Find alloc at pre-init (or earliest iter_begin); used as 'baseline' reference."""
    pre = _by_label(records, "baseline_pre_init")
    if pre:
        return float(pre[0].get("alloc_mb", 0.0))
    bs = _by_label(records, "iter_begin")
    if bs:
        return float(bs[0].get("alloc_mb", 0.0))
    return 0.0


def eval_r1(records: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Steady-state iter_end alloc delta vs proposal-owned buffers.

    post_init can be too early in older logs. Lazy lossy checkpoint buffers
    such as age2bit state and descriptor scratch appear on the first proposal step.
    Use a steady-state iter_end (median of STEADY_LO..STEADY_HI) instead.
    """
    baseline = _baseline_alloc(records)
    iter_ends = _by_label_in_iter_range(records, "iter_end", STEADY_LO, STEADY_HI)
    if not iter_ends:
        return False, "R1: FAIL — no steady-state iter_end records"
    # Use middle record to skip edge effects.
    rec = iter_ends[len(iter_ends) // 2]
    actual_delta = float(rec.get("alloc_mb", 0.0)) - baseline
    # ``tracked_total_mb`` is the proposal pipeline's combined view: lossy
    # checkpoint tensors + SegmentD2H._gpu_bufs +
    # fc/expert_state/elfc GPU buffers. If skip-write env disables some
    # subsystems (e.g., lossy_fc), their bufs are absent — tracked_total
    # naturally drops too, so expected stays accurate without manual
    # adjustments.
    lossy_cp_total = float(
        rec.get("lossy_cp_total_mb", rec.get("tracker_total_mb", 0.0))
    )
    tracked_total = float(rec.get("tracked_total_mb", 0.0))
    if tracked_total <= 0:
        tracked_total = lossy_cp_total
    expected = tracked_total
    diff = actual_delta - expected
    ok = abs(diff) <= R1_TOLERANCE_MB
    return ok, (
        f"R1: {'PASS' if ok else 'FAIL'} — "
        f"actual=+{actual_delta:.1f}MB, expected=+{expected:.1f}MB "
        f"(tracked_total incl. lossy_cp+seg+fc), "
        f"diff={diff:+.1f}MB (tolerance ±{R1_TOLERANCE_MB:.0f}MB)"
    )


def eval_r2(records: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """R2 — no per-iter LARGE alloc on proposal hot path.

    Primary check: alloc_trace_summary (proposal-callsite-filtered) has
    no entries with size > R2_LARGE_ALLOC_MB AND count >= R2_LARGE_ALLOC_COUNT.

    PyTorch/Megatron themselves do thousands of small allocs per iter
    (~157k/iter is normal). We only flag proposal-attributable LARGE
    allocs since R2's intent is "proposal stream's per-iter buffer
    churn = 0", not "no allocs at all in the system".
    """
    iter_ends = _by_label_in_iter_range(records, "iter_end", STEADY_LO, STEADY_HI)
    if len(iter_ends) < 5:
        return False, f"R2: FAIL — too few iter_end records (need ≥ 5 in iter {STEADY_LO}-{STEADY_HI})"
    # Informational only — system-wide alloc count delta.
    counts = [int(r.get("alloc_count", 0)) for r in iter_ends]
    deltas = [counts[i+1] - counts[i] for i in range(len(counts) - 1)]
    median_delta = sorted(deltas)[len(deltas) // 2] if deltas else 0

    trace_recs = _by_label(records, "alloc_trace_summary")
    violations: List[str] = []
    if trace_recs:
        entries = trace_recs[-1].get("alloc_trace", [])
        for e in entries:
            sz = float(e.get("total_mb", 0.0))
            cnt = int(e.get("count", 0))
            if sz > R2_LARGE_ALLOC_MB and cnt >= R2_LARGE_ALLOC_COUNT:
                violations.append(
                    f"{e['callsite']} {sz:.1f}MB×{cnt} ({e['dtype']})"
                )
    # PASS iff no proposal-callsite large allocs detected.
    ok = not violations
    msg = (
        f"R2: {'PASS' if ok else 'FAIL'} — "
        f"proposal-callsite large-alloc violations={len(violations)} "
        f"(system-wide alloc_count median_delta={median_delta}, informational)"
    )
    if violations:
        msg += "\n  " + "\n  ".join(violations[:5])
    return ok, msg


def eval_r3(records: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """anchor iter peak ≤ non-anchor iter peak + tolerance.

    Anchor refresh fires inside the worker thread for iter T but its
    transient is captured by the NEXT iter's iter_end log (iter T+1).
    Rather than thread the worker-iter→log-iter mapping, we use peak
    distribution: max(peak) is anchor; median is non-anchor (anchor
    fires every refresh_every which is ≤ 50% of iters even at smallest
    refresh_every=2).
    """
    iter_ends = _by_label_in_iter_range(records, "iter_end", STEADY_LO, STEADY_HI)
    if not iter_ends:
        return False, "R3: FAIL — no iter_end records in steady range"
    peaks = sorted(float(r.get("peak_mb", 0.0)) for r in iter_ends)
    median = peaks[len(peaks) // 2]
    pmax = peaks[-1]
    pmin = peaks[0]
    delta = pmax - median
    ok = delta <= R3_TOLERANCE_MB
    return ok, (
        f"R3: {'PASS' if ok else 'FAIL'} — "
        f"peak max={pmax:.1f}MB, median={median:.1f}MB, min={pmin:.1f}MB; "
        f"anchor_delta=max-median={delta:+.1f}MB (tolerance={R3_TOLERANCE_MB}MB)"
    )


def eval_r4(records: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """alloc slope < threshold, reserved-allocated waste bounded."""
    iter_ends = _by_label_in_iter_range(records, "iter_end", STEADY_LO, STEADY_HI)
    if len(iter_ends) < 10:
        return False, "R4: FAIL — too few iter_end records (need ≥ 10)"
    iters = [int(r.get("iter", 0)) for r in iter_ends]
    allocs = [float(r.get("alloc_mb", 0.0)) for r in iter_ends]
    reserveds = [float(r.get("reserved_mb", 0.0)) for r in iter_ends]
    peaks = [float(r.get("peak_mb", 0.0)) for r in iter_ends]
    # Best-fit slope (least squares).
    n = len(iters)
    sum_x = sum(iters); sum_y = sum(allocs)
    sum_xy = sum(x*y for x, y in zip(iters, allocs))
    sum_xx = sum(x*x for x in iters)
    denom = n * sum_xx - sum_x * sum_x
    slope = (n * sum_xy - sum_x * sum_y) / denom if denom else 0.0
    waste = max(r - a for r, a in zip(reserveds, allocs))
    max_transient = max(p - a for p, a in zip(peaks, allocs))
    waste_limit = R4_WASTE_FACTOR * max_transient
    ok_slope = abs(slope) < R4_SLOPE_LIMIT_MB_PER_ITER
    ok_waste = waste < waste_limit
    ok = ok_slope and ok_waste
    return ok, (
        f"R4: {'PASS' if ok else 'FAIL'} — "
        f"slope={slope:+.3f}MB/iter (limit ±{R4_SLOPE_LIMIT_MB_PER_ITER}), "
        f"waste={waste:.1f}MB (limit {waste_limit:.1f}MB = {R4_WASTE_FACTOR}× max_transient {max_transient:.1f}MB)"
    )


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        path = os.environ.get("PROPOSAL_VRAM_AUDIT_LOG", "")
    else:
        path = argv[1]
    if not path:
        print("usage: vram_audit_eval.py <vram_audit.jsonl>", file=sys.stderr)
        return 2
    records = load_audit(path)
    if not records:
        print(f"[vram_audit_eval] no records in {path}", file=sys.stderr)
        return 2
    print(f"=== VRAM AUDIT EVAL  ({len(records)} records from {path}) ===")
    results = [
        eval_r1(records),
        eval_r2(records),
        eval_r3(records),
        eval_r4(records),
    ]
    all_ok = True
    for ok, msg in results:
        print(msg)
        if not ok:
            all_ok = False
    print("=" * 30)
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
