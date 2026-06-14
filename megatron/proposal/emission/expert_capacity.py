"""Expert AOS capacity allocation helpers.

The pinned AOS slab is fixed-size. These helpers only decide how many record
slots each expert receives inside that slab; they never allocate memory.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def blob_bytes_for_records(
    n_records: int,
    *,
    prefix_bytes: int,
    record_bytes: int,
    sector_bytes: int,
) -> int:
    n = max(0, int(n_records))
    m = max(1, int(sector_bytes))
    raw = int(prefix_bytes) + n * int(record_bytes)
    return ((raw + m - 1) // m) * m


def fit_record_capacities_to_bytes(
    requested: Sequence[int],
    *,
    max_total_bytes: int,
    prefix_bytes: int,
    record_bytes: int,
    sector_bytes: int,
) -> List[int]:
    """Scale requested per-expert capacities to fit the fixed slab bytes."""

    req = [max(0, int(x)) for x in requested]
    if not req:
        return []

    req_arr = np.asarray(req, dtype=np.int64)
    prefix_i = int(prefix_bytes)
    record_i = int(record_bytes)
    sector_i = max(1, int(sector_bytes))

    def total_bytes(caps: np.ndarray) -> int:
        raw = prefix_i + caps.astype(np.int64, copy=False) * record_i
        padded = ((raw + sector_i - 1) // sector_i) * sector_i
        return int(padded.sum())

    if total_bytes(req_arr) <= int(max_total_bytes):
        return req
    zero_arr = np.zeros_like(req_arr)
    if total_bytes(zero_arr) > int(max_total_bytes):
        raise RuntimeError(
            "pinned AOS slab is too small even for per-expert prefixes"
        )

    lo = 0.0
    hi = 1.0
    best = zero_arr
    for _ in range(24):
        mid = (lo + hi) * 0.5
        caps = (req_arr.astype(np.float64) * mid).astype(np.int64)
        if total_bytes(caps) <= int(max_total_bytes):
            best = caps
            lo = mid
        else:
            hi = mid
    return [int(x) for x in best.tolist()]


def redistribute_expert_record_capacities(
    *,
    selected_by_expert: Sequence[int],
    overflow_by_expert: Sequence[int],
    max_records_by_expert: Sequence[int],
    target_record_capacity: int,
    max_total_bytes: int,
    prefix_bytes: int,
    record_bytes: int,
    sector_bytes: int,
    previous_capacities: Optional[Sequence[int]] = None,
) -> List[int]:
    """Redistribute a fixed AOS slab across experts from direct feedback.

    The slab allocation itself is fixed. This helper only changes each
    expert's contiguous slice capacity. Existing capacities are frozen only
    when every expert had zero overflow; any expert-local overflow means cold
    expert spare capacity must be allowed to move to hot experts.
    """

    selected = [max(0, int(x)) for x in selected_by_expert]
    overflow = [max(0, int(x)) for x in overflow_by_expert]
    max_records = [max(0, int(x)) for x in max_records_by_expert]
    if len(selected) != len(overflow) or len(selected) != len(max_records):
        raise ValueError(
            "selected_by_expert, overflow_by_expert, and max_records_by_expert "
            "must have the same length"
        )
    n = len(selected)
    if n == 0:
        return []

    if previous_capacities is not None and len(previous_capacities) == n:
        prev = [
            min(max_records[i], max(0, int(previous_capacities[i])))
            for i in range(n)
        ]
        if all(x == 0 for x in overflow):
            return fit_record_capacities_to_bytes(
                prev,
                max_total_bytes=int(max_total_bytes),
                prefix_bytes=int(prefix_bytes),
                record_bytes=int(record_bytes),
                sector_bytes=int(sector_bytes),
            )
    target = max(0, min(int(target_record_capacity), int(sum(max_records))))
    if target <= 0:
        return [0] * n

    weights = np.asarray(selected, dtype=np.float64)
    if float(weights.sum()) <= 0.0:
        weights = np.asarray(max_records, dtype=np.float64)
    max_arr = np.asarray(max_records, dtype=np.int64)
    caps = _capped_proportional_records(weights, max_arr, int(target))
    return fit_record_capacities_to_bytes(
        caps,
        max_total_bytes=int(max_total_bytes),
        prefix_bytes=int(prefix_bytes),
        record_bytes=int(record_bytes),
        sector_bytes=int(sector_bytes),
    )


def _capped_proportional_records(
    weights: np.ndarray,
    max_records: np.ndarray,
    target_record_capacity: int,
) -> List[int]:
    n = int(max_records.size)
    if n == 0:
        return []
    target = max(0, min(int(target_record_capacity), int(max_records.sum())))
    if target <= 0:
        return [0] * n

    weights = np.maximum(weights.astype(np.float64, copy=False), 0.0)
    caps = np.zeros(n, dtype=np.int64)
    active = max_records > 0

    while int(caps.sum()) < target and bool(active.any()):
        remaining_target = target - int(caps.sum())
        room = max_records - caps
        active_weight = np.where(active, weights, 0.0)
        if float(active_weight.sum()) <= 0.0:
            active_weight = np.where(active, room.astype(np.float64), 0.0)
        if float(active_weight.sum()) <= 0.0:
            break

        raw = remaining_target * active_weight / float(active_weight.sum())
        capped = (raw >= room) & active
        if bool(capped.any()):
            caps[capped] = max_records[capped]
            active[capped] = False
            continue

        add = np.minimum(np.floor(raw).astype(np.int64), room)
        caps += add
        break

    spare = target - int(caps.sum())
    if spare > 0:
        room = max_records - caps
        active_weight = np.where(room > 0, weights, 0.0)
        if float(active_weight.sum()) <= 0.0:
            active_weight = room.astype(np.float64)
        raw = spare * active_weight / max(float(active_weight.sum()), 1.0)
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        for idx in order:
            if spare <= 0:
                break
            if room[idx] <= 0:
                continue
            caps[idx] += 1
            spare -= 1
    return [int(x) for x in caps.tolist()]


__all__ = [
    "blob_bytes_for_records",
    "fit_record_capacities_to_bytes",
    "redistribute_expert_record_capacities",
]
