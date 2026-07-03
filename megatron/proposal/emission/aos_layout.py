"""AOS layout header and pinned writer metadata."""

from __future__ import annotations

import struct
from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


ExpertId = Tuple[int, int]
AOS_LAYOUT_MAGIC = 0x414F534C41594F54  # "AOSLAYOT"
AOS_LAYOUT_VERSION = 1
AOS_LAYOUT_HEADER_WORDS_FIXED = 4
PREFIX_BYTES = 8
RECORD_BYTES = 16
SECTOR_BYTES = 4096


def aos_layout_header_bytes(n_experts: int) -> int:
    # Header payload: fixed fields, per-expert AOS offsets, per-expert record
    # capacities.  The header is only the in-memory pinned slab contract between
    # the GPU layout kernel and the CPU writer; per-expert disk blobs are
    # unchanged.
    words = AOS_LAYOUT_HEADER_WORDS_FIXED + 2 * max(0, int(n_experts))
    raw = words * 8
    return ((raw + SECTOR_BYTES - 1) // SECTOR_BYTES) * SECTOR_BYTES


@dataclass(frozen=True)
class AosLayoutRow:
    expert_id: ExpertId
    aos_byte_off: int
    blob_padded_bytes: int
    record_capacity: int


def _empty_pinned(n: int, dtype: torch.dtype) -> torch.Tensor:
    try:
        return torch.empty(int(n), dtype=dtype, pin_memory=True)
    except RuntimeError:
        return torch.empty(int(n), dtype=dtype)


class PinnedAosLayoutMirror:
    """Pinned writer metadata for one AOS generation."""

    def __init__(self, max_experts: int) -> None:
        if int(max_experts) < 0:
            raise ValueError("max_experts must be non-negative")
        self.max_experts = int(max_experts)
        n = max(1, self.max_experts)
        self.layer = _empty_pinned(n, torch.int32)
        self.expert = _empty_pinned(n, torch.int32)
        self.aos_byte_off = _empty_pinned(n, torch.int64)
        self.blob_padded_bytes = _empty_pinned(n, torch.int64)
        self.record_capacity = _empty_pinned(n, torch.int64)
        self.active_count = 0

    def update_from_header(
        self,
        *,
        pinned_aos: torch.Tensor,
        prepared: Sequence[object],
        max_records_by_expert: dict[ExpertId, int],
    ) -> None:
        n = int(len(prepared))
        if n > self.max_experts:
            raise RuntimeError(
                f"AOS layout mirror capacity exceeded: {n} > {self.max_experts}"
            )
        if pinned_aos.device.type != "cpu" or pinned_aos.dtype != torch.uint8:
            raise RuntimeError("pinned_aos must be a CPU uint8 tensor")
        if not pinned_aos.is_contiguous():
            raise RuntimeError("pinned_aos must be contiguous")
        header_nbytes = aos_layout_header_bytes(n)
        if int(pinned_aos.numel()) < header_nbytes:
            raise RuntimeError("pinned AOS slab is smaller than its layout header")
        view = pinned_aos.narrow(0, 0, header_nbytes).view(torch.int64)
        magic = int(view[0].item())
        version = int(view[1].item())
        n_header = int(view[2].item())
        header_bytes = int(view[3].item())
        if magic != AOS_LAYOUT_MAGIC or version != AOS_LAYOUT_VERSION:
            raise RuntimeError(
                f"invalid AOS layout header: magic=0x{magic:x} version={version}"
            )
        if n_header != n:
            raise RuntimeError(
                f"AOS layout header expert count mismatch: {n_header} vs {n}"
            )
        if header_bytes != header_nbytes:
            raise RuntimeError(
                f"AOS layout header size mismatch: {header_bytes} vs {header_nbytes}"
            )
        for i, item in enumerate(prepared):
            ent = item[0]
            expert_id = ent.expert_id
            off = int(view[AOS_LAYOUT_HEADER_WORDS_FIXED + i].item())
            cap = int(view[AOS_LAYOUT_HEADER_WORDS_FIXED + n + i].item())
            cap = min(
                max(0, cap),
                max(0, int(max_records_by_expert.get(expert_id, 0))),
            )
            self.layer[i] = int(expert_id[0])
            self.expert[i] = int(expert_id[1])
            self.aos_byte_off[i] = off
            raw = int(PREFIX_BYTES) + cap * int(RECORD_BYTES)
            self.blob_padded_bytes[i] = (
                0 if off < 0 else ((raw + SECTOR_BYTES - 1) // SECTOR_BYTES) * SECTOR_BYTES
            )
            self.record_capacity[i] = cap
        self.active_count = n

    def rows(self) -> Iterable[AosLayoutRow]:
        for i in range(int(self.active_count)):
            yield AosLayoutRow(
                expert_id=(int(self.layer[i].item()), int(self.expert[i].item())),
                aos_byte_off=int(self.aos_byte_off[i].item()),
                blob_padded_bytes=int(self.blob_padded_bytes[i].item()),
                record_capacity=int(self.record_capacity[i].item()),
            )


def read_lfsy_prefix_count(
    pinned_aos: torch.Tensor,
    *,
    aos_byte_off: int,
    record_capacity: int,
    prefix_bytes: int = 8,
) -> int:
    """Read GPU-stamped ``count`` from a pinned AOS blob prefix."""

    if pinned_aos.device.type != "cpu" or pinned_aos.dtype != torch.uint8:
        raise RuntimeError("pinned_aos must be a CPU uint8 tensor")
    if not pinned_aos.is_contiguous():
        raise RuntimeError("pinned_aos must be contiguous")
    off = int(aos_byte_off)
    if off < 0 or off + int(prefix_bytes) > int(pinned_aos.numel()):
        raise RuntimeError(
            f"AOS prefix offset out of range: off={off}, "
            f"pinned={int(pinned_aos.numel())}"
        )
    view = pinned_aos.numpy()
    count, _step = struct.unpack_from("<II", view, off)
    return min(max(0, int(count)), max(0, int(record_capacity)))


if _HAS_TRITON:
    @triton.jit
    def _build_aos_layout_kernel(
        max_records_ptr,
        cursor_ptr,
        offset_ptr,
        capacity_ptr,
        header_ptr,
        initialized_ptr,
        sample_blocks_current_ptr,
        probe_count_mismatch_streak_ptr,
        sample_grow_hold_ptr,
        ci_shortfall_streak_ptr,
        probe_diagnostics_ptr,
        probe_expected_selected_count_ptr,
        capacity_demand_hint_ptr,
        N_EXPERTS: tl.constexpr,
        TARGET_RECORDS: tl.constexpr,
        MAX_TOTAL_BYTES: tl.constexpr,
        SEGMENT_BYTES: tl.constexpr,
        N_SEGMENTS: tl.constexpr,
        BLOCK_SEGMENTS: tl.constexpr,
        HEADER_BYTES: tl.constexpr,
        TOTAL_ELEMS_GLOBAL: tl.constexpr,
        ALLOWED_PROBE_COUNT_DELTA: tl.constexpr,
        SAMPLE_BLOCKS_MIN: tl.constexpr,
        SAMPLE_BLOCKS_CAP: tl.constexpr,
        PROBE_MISMATCH_GROW_AFTER: tl.constexpr,
        UPDATE_SAMPLE_FROM_PROBE: tl.constexpr,
        USE_CAPACITY_HINT: tl.constexpr,
        MAGIC: tl.constexpr,
        VERSION: tl.constexpr,
        PREFIX: tl.constexpr,
        RECORD: tl.constexpr,
        SECTOR: tl.constexpr,
    ):
        selected_sum = tl.full((), 0, tl.int64)
        max_sum = tl.full((), 0, tl.int64)
        for i in range(0, N_EXPERTS):
            selected_sum += tl.maximum(tl.load(cursor_ptr + i), 0)
            max_sum += tl.maximum(tl.load(max_records_ptr + i), 0)

        initialized = tl.load(initialized_ptr)
        if UPDATE_SAMPLE_FROM_PROBE:
            expected_count = tl.maximum(tl.load(probe_expected_selected_count_ptr), 0)
            allowed_delta = tl.full((), ALLOWED_PROBE_COUNT_DELTA, tl.int64)
            diff = selected_sum - expected_count
            diff = tl.where(diff < 0, -diff, diff)
            probe_mismatch = (
                (initialized != 0)
                & (expected_count > 0)
                & (diff > allowed_delta)
            )
            prev_streak = tl.maximum(tl.load(probe_count_mismatch_streak_ptr), 0)
            next_streak = tl.where(probe_mismatch, prev_streak + 1, 0)
            grow_after = tl.maximum(tl.full((), PROBE_MISMATCH_GROW_AFTER, tl.int32), 1)
            grow = probe_mismatch & (next_streak >= grow_after)
            current_sample = tl.maximum(tl.load(sample_blocks_current_ptr), SAMPLE_BLOCKS_MIN)
            grown_sample = tl.minimum(current_sample * 2, SAMPLE_BLOCKS_CAP)
            tl.store(
                sample_blocks_current_ptr,
                tl.where(grow, grown_sample, current_sample),
            )
            tl.store(probe_count_mismatch_streak_ptr, tl.where(grow, 0, next_streak))
            tl.store(sample_grow_hold_ptr, tl.where(grow, 1, tl.load(sample_grow_hold_ptr)))
            tl.store(ci_shortfall_streak_ptr, tl.where(grow, 0, tl.load(ci_shortfall_streak_ptr)))
            current_probe_ratio = tl.load(probe_diagnostics_ptr + 0)
            current_expected = (
                current_probe_ratio
                * tl.full((), TOTAL_ELEMS_GLOBAL, tl.float32)
            ).to(tl.int64)
            valid_current = (
                (current_probe_ratio > 0.0)
                & (current_probe_ratio == current_probe_ratio)
                & (TOTAL_ELEMS_GLOBAL > 0)
            )
            tl.store(
                probe_expected_selected_count_ptr,
                tl.where(valid_current, current_expected, 0),
            )
        payload_budget = tl.maximum(MAX_TOTAL_BYTES - HEADER_BYTES - N_EXPERTS * PREFIX, 0)
        record_budget = payload_budget // RECORD
        target = tl.minimum(tl.minimum(tl.full((), TARGET_RECORDS, tl.int64), record_budget), max_sum)
        max_segment_records = tl.maximum((SEGMENT_BYTES - PREFIX) // RECORD, 0)

        demand_sum = tl.full((), 0, tl.int64)
        for i in range(0, N_EXPERTS):
            max_i = tl.maximum(tl.load(max_records_ptr + i), 0)
            hint = tl.full((), 0, tl.int64)
            if USE_CAPACITY_HINT:
                hint_f = tl.maximum(tl.load(capacity_demand_hint_ptr + i), 0.0)
                hint = tl.minimum(hint_f.to(tl.int64), max_i)
            # The direct-emission cursor is the raw selected count.  Overflow is
            # tracked separately for diagnostics and written count, so adding it
            # here would double-count capped records.
            previous_demand = tl.minimum(tl.maximum(tl.load(cursor_ptr + i), 0), max_i)
            demand = tl.maximum(hint, previous_demand)
            demand_sum += tl.minimum(demand, max_i)
        use_demand = USE_CAPACITY_HINT & (demand_sum > 0)

        for i in range(0, N_EXPERTS):
            max_i = tl.maximum(tl.load(max_records_ptr + i), 0)
            hint = tl.full((), 0, tl.int64)
            if USE_CAPACITY_HINT:
                hint_f = tl.maximum(tl.load(capacity_demand_hint_ptr + i), 0.0)
                hint = tl.minimum(hint_f.to(tl.int64), max_i)
            previous_demand = tl.minimum(tl.maximum(tl.load(cursor_ptr + i), 0), max_i)
            demand = tl.maximum(hint, previous_demand)
            weight = tl.where(use_demand, demand, max_i)
            denom = tl.where(use_demand, demand_sum, max_sum)
            cap = (weight * target) // tl.maximum(denom, 1)
            cap = tl.minimum(tl.maximum(cap, 0), max_i)
            cap = tl.minimum(cap, max_segment_records)
            tl.store(capacity_ptr + i, cap)

        tl.store(header_ptr + 0, tl.full((), MAGIC, tl.int64))
        tl.store(header_ptr + 1, tl.full((), VERSION, tl.int64))
        tl.store(header_ptr + 2, tl.full((), N_EXPERTS, tl.int64))
        tl.store(header_ptr + 3, tl.full((), HEADER_BYTES, tl.int64))
        prefix_blob = ((PREFIX + SECTOR - 1) // SECTOR) * SECTOR
        for i in range(0, N_EXPERTS):
            tl.store(offset_ptr + i, -1)
            tl.store(header_ptr + 4 + i, -1)
            tl.store(header_ptr + 4 + N_EXPERTS + i, 0)

        segment_ids = tl.arange(0, BLOCK_SEGMENTS)
        segment_mask = segment_ids < N_SEGMENTS
        segment_byte_off = segment_ids.to(tl.int64) * tl.full((), SEGMENT_BYTES, tl.int64)
        segment_limit = tl.minimum(
            tl.full((BLOCK_SEGMENTS,), SEGMENT_BYTES, tl.int64),
            tl.maximum(tl.full((BLOCK_SEGMENTS,), MAX_TOTAL_BYTES, tl.int64) - segment_byte_off, 0),
        )
        segment_used = tl.where(segment_ids == 0, tl.full((BLOCK_SEGMENTS,), HEADER_BYTES, tl.int64), 0)
        segment_used = tl.where(segment_mask, segment_used, segment_limit)

        for placement_i in range(0, N_EXPERTS):
            best_idx = tl.full((), -1, tl.int32)
            best_blob = tl.full((), -1, tl.int64)
            for j in range(0, N_EXPERTS):
                unplaced = tl.load(offset_ptr + j) == -1
                cap_j = tl.maximum(tl.load(capacity_ptr + j), 0)
                raw_j = PREFIX + cap_j * RECORD
                blob_j = ((raw_j + SECTOR - 1) // SECTOR) * SECTOR
                take_j = unplaced & (
                    (blob_j > best_blob)
                    | ((blob_j == best_blob) & ((best_idx < 0) | (j < best_idx)))
                )
                best_idx = tl.where(take_j, j, best_idx)
                best_blob = tl.where(take_j, blob_j, best_blob)
            cap = tl.maximum(tl.load(capacity_ptr + best_idx), 0)
            raw_requested = PREFIX + cap * RECORD
            blob_requested = ((raw_requested + SECTOR - 1) // SECTOR) * SECTOR

            segment_free = tl.maximum(segment_limit - segment_used, 0)
            full_fits = segment_mask & (segment_free >= blob_requested)
            free_after_full = segment_free - blob_requested
            huge = tl.full((BLOCK_SEGMENTS,), 0x7FFFFFFFFFFFFFFF, tl.int64)
            best_free_after = tl.min(tl.where(full_fits, free_after_full, huge), axis=0)
            full_found = best_free_after < tl.full((), 0x7FFFFFFFFFFFFFFF, tl.int64)
            full_seg = tl.min(
                tl.where(full_fits & (free_after_full == best_free_after), segment_ids, 2147483647),
                axis=0,
            )

            prefix_fits_segments = segment_mask & (segment_free >= prefix_blob)
            best_prefix_free = tl.max(
                tl.where(prefix_fits_segments, segment_free, tl.full((BLOCK_SEGMENTS,), -1, tl.int64)),
                axis=0,
            )
            prefix_found = best_prefix_free >= prefix_blob
            prefix_seg = tl.min(
                tl.where(prefix_fits_segments & (segment_free == best_prefix_free), segment_ids, 2147483647),
                axis=0,
            )
            chosen_seg = tl.where(full_found, full_seg, prefix_seg)
            placed = full_found | prefix_found
            chosen_used = tl.max(
                tl.where(segment_ids == chosen_seg, segment_used, tl.full((BLOCK_SEGMENTS,), 0, tl.int64)),
                axis=0,
            )
            layout_off = chosen_seg.to(tl.int64) * tl.full((), SEGMENT_BYTES, tl.int64) + chosen_used
            segment_local_off = layout_off - chosen_seg.to(tl.int64) * tl.full((), SEGMENT_BYTES, tl.int64)
            actual_segment_remaining = tl.minimum(
                tl.maximum(tl.full((), SEGMENT_BYTES, tl.int64) - segment_local_off, 0),
                tl.maximum(tl.full((), MAX_TOTAL_BYTES, tl.int64) - layout_off, 0),
            )
            record_bytes_available = tl.maximum(actual_segment_remaining - PREFIX, 0)
            cap_fit = record_bytes_available // RECORD
            cap = tl.where(placed, tl.minimum(cap, cap_fit), 0)
            raw = PREFIX + cap * RECORD
            blob = ((raw + SECTOR - 1) // SECTOR) * SECTOR
            device_layout_off = tl.where(placed, layout_off, -2)
            header_layout_off = tl.where(placed, layout_off, -1)
            tl.store(capacity_ptr + best_idx, cap)
            tl.store(offset_ptr + best_idx, device_layout_off)
            tl.store(header_ptr + 4 + best_idx, header_layout_off)
            tl.store(header_ptr + 4 + N_EXPERTS + best_idx, cap)
            segment_used = tl.where(
                segment_ids == chosen_seg,
                tl.where(placed, segment_used + blob, segment_used),
                segment_used,
            )
        tl.store(initialized_ptr, 1)


class GpuAosLayoutState:
    """Device-side expert capacity/offset state for one pinned AOS slab."""

    def __init__(
        self,
        *,
        max_records_by_expert: Sequence[int],
        target_record_capacity: int,
        max_total_bytes: int,
        segment_bytes: int,
        device: torch.device,
    ) -> None:
        self.max_records_by_expert = tuple(max(0, int(x)) for x in max_records_by_expert)
        self.n_experts = len(self.max_records_by_expert)
        self.target_record_capacity = max(0, int(target_record_capacity))
        self.max_total_bytes = int(max_total_bytes)
        self.segment_bytes = int(segment_bytes)
        self.n_segments = max(
            1,
            int(math.ceil(float(max(1, self.max_total_bytes)) / float(max(1, self.segment_bytes)))),
        )
        self.block_segments = 1
        while self.block_segments < self.n_segments:
            self.block_segments <<= 1
        self.header_bytes = aos_layout_header_bytes(self.n_experts)
        self.header_words = self.header_bytes // 8
        min_layout_bytes = self.header_bytes + self.n_experts * SECTOR_BYTES
        if self.segment_bytes < SECTOR_BYTES:
            raise ValueError(
                f"AOS segment_bytes must be at least one sector "
                f"({SECTOR_BYTES}), got {self.segment_bytes}"
            )
        if self.max_total_bytes < min_layout_bytes:
            raise ValueError(
                "AOS slab is too small for the layout header and per-expert "
                f"prefix sectors: {self.max_total_bytes} < {min_layout_bytes}"
            )
        self.device = device
        n = max(1, self.n_experts)
        max_host = torch.empty(n, dtype=torch.int64, pin_memory=True)
        for i, value in enumerate(self.max_records_by_expert):
            max_host[i] = int(value)
        self.max_records_device = torch.empty(n, dtype=torch.int64, device=device)
        self.max_records_device.copy_(max_host, non_blocking=True)
        self.offset_device = torch.full((n,), -1, dtype=torch.int64, device=device)
        self.capacity_device = torch.zeros(n, dtype=torch.int64, device=device)
        self.header_device = torch.zeros(
            max(1, self.header_words), dtype=torch.int64, device=device,
        )
        self.initialized_device = torch.zeros(1, dtype=torch.int32, device=device)

    def launch_update(
        self,
        *,
        cursor: torch.Tensor,
        stream: torch.cuda.Stream,
        sample_blocks_current: torch.Tensor | None = None,
        probe_count_mismatch_streak: torch.Tensor | None = None,
        sample_grow_hold: torch.Tensor | None = None,
        ci_shortfall_streak: torch.Tensor | None = None,
        probe_diagnostics: torch.Tensor | None = None,
        probe_expected_selected_count: torch.Tensor | None = None,
        capacity_demand_hint: torch.Tensor | None = None,
        total_elems_global: int = 0,
        allowed_probe_count_delta: int = 0,
        sample_blocks_min: int = 1,
        sample_blocks_cap: int = 1,
        probe_mismatch_grow_after: int = 3,
    ) -> None:
        if self.n_experts <= 0:
            return
        if not _HAS_TRITON:
            raise RuntimeError("GPU AOS layout requires Triton")
        if cursor.numel() < self.n_experts:
            raise RuntimeError("direct cursor tensor is too small")
        update_sample = (
            sample_blocks_current is not None
            and probe_count_mismatch_streak is not None
            and sample_grow_hold is not None
            and ci_shortfall_streak is not None
            and probe_diagnostics is not None
            and probe_expected_selected_count is not None
            and int(total_elems_global) > 0
            and int(sample_blocks_cap) > 0
        )
        sample_ptr = sample_blocks_current if update_sample else self.initialized_device
        mismatch_streak_ptr = (
            probe_count_mismatch_streak if update_sample else self.initialized_device
        )
        grow_hold_ptr = sample_grow_hold if update_sample else self.initialized_device
        ci_streak_ptr = ci_shortfall_streak if update_sample else self.initialized_device
        diagnostics_ptr = probe_diagnostics if update_sample else self.header_device
        expected_ptr = (
            probe_expected_selected_count if update_sample else self.initialized_device
        )
        use_capacity_hint = (
            capacity_demand_hint is not None
            and int(capacity_demand_hint.numel()) >= int(self.n_experts)
        )
        capacity_hint_ptr = (
            capacity_demand_hint if use_capacity_hint else self.max_records_device
        )
        with torch.cuda.stream(stream):
            _build_aos_layout_kernel[(1,)](
                self.max_records_device,
                cursor,
                self.offset_device,
                self.capacity_device,
                self.header_device,
                self.initialized_device,
                sample_ptr,
                mismatch_streak_ptr,
                grow_hold_ptr,
                ci_streak_ptr,
                diagnostics_ptr,
                expected_ptr,
                capacity_hint_ptr,
                N_EXPERTS=int(self.n_experts),
                TARGET_RECORDS=int(self.target_record_capacity),
                MAX_TOTAL_BYTES=int(self.max_total_bytes),
                SEGMENT_BYTES=int(self.segment_bytes),
                N_SEGMENTS=int(self.n_segments),
                BLOCK_SEGMENTS=int(self.block_segments),
                HEADER_BYTES=int(self.header_bytes),
                TOTAL_ELEMS_GLOBAL=max(0, int(total_elems_global)),
                ALLOWED_PROBE_COUNT_DELTA=max(0, int(allowed_probe_count_delta)),
                SAMPLE_BLOCKS_MIN=max(1, int(sample_blocks_min)),
                SAMPLE_BLOCKS_CAP=max(1, int(sample_blocks_cap)),
                PROBE_MISMATCH_GROW_AFTER=max(1, int(probe_mismatch_grow_after)),
                UPDATE_SAMPLE_FROM_PROBE=bool(update_sample),
                USE_CAPACITY_HINT=bool(use_capacity_hint),
                MAGIC=int(AOS_LAYOUT_MAGIC),
                VERSION=int(AOS_LAYOUT_VERSION),
                PREFIX=int(PREFIX_BYTES),
                RECORD=int(RECORD_BYTES),
                SECTOR=int(SECTOR_BYTES),
                num_warps=1,
            )


__all__ = [
    "AOS_LAYOUT_MAGIC",
    "AOS_LAYOUT_VERSION",
    "AosLayoutRow",
    "GpuAosLayoutState",
    "PinnedAosLayoutMirror",
    "aos_layout_header_bytes",
    "read_lfsy_prefix_count",
]
