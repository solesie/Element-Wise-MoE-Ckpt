"""Direct lossy-record write kernels.

Production materializes lossy expert records only through direct emission:
the same score predicate selects records, updates actual-written age state, and
writes through a per-expert cursor capacity cap. Host-side selection masks and
cutoff wrappers are not part of this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


@dataclass(frozen=True)
class LossyRecordFormat:
    """Single source of truth for the on-disk lossy record format."""

    record_bytes: int = 16
    record_words: int = 4
    prefix_bytes: int = 8
    sector_bytes: int = 4096


RECORD = LossyRecordFormat()


# On-disk constants — must match emission/aos_emitter.py and storage/expert_writer.py.
RECORD_BYTES: int = RECORD.record_bytes
RECORD_WORDS: int = RECORD.record_words
PREFIX_BYTES: int = RECORD.prefix_bytes
SECTOR_BYTES: int = RECORD.sector_bytes
GROUP_SCAN_TARGET_ELEMENTS: int = 256
DIRECT_EMIT_WORKERS_PER_SM: int = 32


@dataclass(frozen=True)
class AdamHParams:
    """Adam scalars needed to compute Δθ² for one lossy CP iteration."""

    lr_over_bc1: float
    sqrt_bc2_inv: float
    eps: float


@dataclass
class LossyRecordWriteInputs:
    """Inputs consumed by the direct lossy record writer for one CP iter."""

    bf16_groups_mvph: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]
    name_list: List[str]
    group_size: int
    within_off_per_name: List[int] = field(default_factory=list)
    group_count_per_name: List[int] = field(default_factory=list)
    h_lut_values: Optional[torch.Tensor] = None
    h_lut_log_v_min: Optional[torch.Tensor] = None
    h_lut_inv_log_v_width: Optional[torch.Tensor] = None
    total_th: Optional[torch.Tensor] = None
    adam_hparams: Optional[AdamHParams] = None
    adam_scalar_device: Optional[torch.Tensor] = None
    step_device: Optional[torch.Tensor] = None
    random_mode: bool = False
    random_salt: int = 0
    direct_emission_enabled: bool = False
    direct_record_cursor: Optional[torch.Tensor] = None
    direct_overflow_count: Optional[torch.Tensor] = None
    sample_blocks_current: Optional[torch.Tensor] = None
    probe_count_mismatch_streak: Optional[torch.Tensor] = None
    sample_grow_hold: Optional[torch.Tensor] = None
    ci_shortfall_streak: Optional[torch.Tensor] = None
    probe_diagnostics: Optional[torch.Tensor] = None
    probe_expected_selected_count: Optional[torch.Tensor] = None
    capacity_demand_hint: Optional[torch.Tensor] = None
    total_elems_global: int = 0
    allowed_probe_count_delta: int = 0
    sample_blocks_min: int = 1
    sample_blocks_cap: int = 1
    probe_count_mismatch_grow_after: int = 3
    aos_layout_offsets: Optional[torch.Tensor] = None
    aos_layout_capacities: Optional[torch.Tensor] = None
    aos_layout_header_device: Optional[torch.Tensor] = None
    aos_layout_header_words: int = 0
    age2bit_words: Optional[torch.Tensor] = None
    age2bit_floor_mid: Optional[torch.Tensor] = None
    age2bit_floor_old: Optional[torch.Tensor] = None


class LossyRecordWriter:
    """Single entry object for one iteration's lossy record write.

    The lossy checkpoint owner supplies the per-iter state; this object keeps
    callers on the V2 write entry point and away from the lower-level AOS
    pipeline.
    """

    def __init__(self, write_for_iter: Callable[..., Any]) -> None:
        self._write_for_iter = write_for_iter

    def write_for_iter(self, *args: Any, **kwargs: Any) -> Any:
        return self._write_for_iter(*args, **kwargs)


def _triton_random_salt(random_mode: bool, random_salt: int) -> int:
    return (int(random_salt) & 0xFFFFFFFF) if bool(random_mode) else 0


# ---------------------------------------------------------------------------
# Triton kernels — production path.
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _lookup_h_and_floors_from_lut(
        v_fp32,
        h_lut_ptr,
        log_v_min_ptr,
        inv_log_v_width_ptr,
        floor_mid_ptr,
        floor_old_ptr,
        H_LUT_BINS: tl.constexpr,
        N_VBINS: tl.constexpr,
    ):
        log_v_min = tl.load(log_v_min_ptr)
        inv_width = tl.load(inv_log_v_width_ptr)
        log_v = tl.log(tl.maximum(v_fp32, 1.0e-30))
        pos = (log_v - log_v_min) * inv_width
        pos = tl.maximum(tl.minimum(pos, H_LUT_BINS - 1.0), 0.0)
        idx_lo = pos.to(tl.int32)
        idx_hi = tl.minimum(idx_lo + 1, H_LUT_BINS - 1)
        frac = pos - idx_lo.to(tl.float32)
        h_lo = tl.load(h_lut_ptr + idx_lo)
        h_hi = tl.load(h_lut_ptr + idx_hi)
        h = h_lo + frac * (h_hi - h_lo)
        vbin_pos = pos * (N_VBINS / (H_LUT_BINS - 1))
        floor_pos = tl.maximum(tl.minimum(vbin_pos - 0.5, N_VBINS - 1.0), 0.0)
        floor_lo_idx = floor_pos.to(tl.int32)
        floor_hi_idx = tl.minimum(floor_lo_idx + 1, N_VBINS - 1)
        floor_frac = floor_pos - floor_lo_idx.to(tl.float32)

        mid_lo = tl.load(floor_mid_ptr + floor_lo_idx)
        mid_hi = tl.load(floor_mid_ptr + floor_hi_idx)
        old_lo = tl.load(floor_old_ptr + floor_lo_idx)
        old_hi = tl.load(floor_old_ptr + floor_hi_idx)
        mid_interp = mid_lo + floor_frac * (mid_hi - mid_lo)
        old_interp = old_lo + floor_frac * (old_hi - old_lo)
        mid = tl.where((mid_lo > 0.0) & (mid_hi > 0.0), mid_interp, tl.maximum(mid_lo, mid_hi))
        old = tl.where((old_lo > 0.0) & (old_hi > 0.0), old_interp, tl.maximum(old_lo, old_hi))
        return h, mid, old

    @triton.jit
    def _age2_load_code(age_words_ptr, global_elem_idx):
        word_idx = (global_elem_idx >> 4).to(tl.int64)
        shift = ((global_elem_idx & 15) << 1).to(tl.int32)
        word = tl.load(age_words_ptr + word_idx, mask=global_elem_idx >= 0, other=0)
        return (word >> shift) & 3

    @triton.jit
    def _age2_store_code_atomic(age_words_ptr, global_elem_idx, code, mask):
        word_idx = (global_elem_idx >> 4).to(tl.int64)
        shift = ((global_elem_idx & 15) << 1).to(tl.int32)
        bit_mask = (3 << shift).to(tl.int32)
        clear_mask = ~bit_mask
        value = (code.to(tl.int32) << shift).to(tl.int32)
        tl.atomic_and(
            age_words_ptr + word_idx,
            clear_mask,
            mask=mask,
            sem="relaxed",
        )
        tl.atomic_or(
            age_words_ptr + word_idx,
            value,
            mask=mask,
            sem="relaxed",
        )

    @triton.jit
    def _age2_phase_hit(global_elem_idx, step_value, period: tl.constexpr, salt: tl.constexpr):
        x = global_elem_idx.to(tl.uint32) ^ tl.full((), salt, tl.uint32)
        x = (x ^ (x >> 16)) * tl.full((), 0x7FEB352D, tl.uint32)
        x = (x ^ (x >> 15)) * tl.full((), 0x846CA68B, tl.uint32)
        x = x ^ (x >> 16)
        phase = (x % period).to(tl.int32)
        return ((step_value.to(tl.int32) + phase) % period) == 0

    @triton.jit
    def _random_unit_from_index(global_idx, salt: tl.constexpr):
        x = global_idx.to(tl.uint32)
        x = x ^ tl.full((), salt, tl.uint32)
        x = (x ^ (x >> 16)) * tl.full((), 0x7FEB352D, tl.uint32)
        x = (x ^ (x >> 15)) * tl.full((), 0x846CA68B, tl.uint32)
        x = x ^ (x >> 16)
        mant = x & tl.full((), 0x00FFFFFF, tl.uint32)
        return (mant.to(tl.float32) + 1.0) * (1.0 / 16777217.0)

    @triton.jit
    def _build_segment_entry_table_kernel(
        entry_expert_idx_ptr,
        entry_group_count_ptr,
        aos_layout_offset_ptr,
        aos_layout_capacity_ptr,
        segment_entry_indices_ptr,
        segment_entry_counts_ptr,
        segment_entry_max_group_count_ptr,
        segment_entry_overflow_ptr,
        n_entries,
        n_segments,
        table_capacity,
        SEGMENT_BYTES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_entries
        expert_idx = tl.load(entry_expert_idx_ptr + offs, mask=mask, other=0)
        blob_off = tl.load(aos_layout_offset_ptr + expert_idx, mask=mask, other=-1)
        seg_id = (blob_off // SEGMENT_BYTES).to(tl.int32)
        active = mask & (blob_off >= 0) & (seg_id >= 0) & (seg_id < n_segments)
        slot = tl.atomic_add(
            segment_entry_counts_ptr + seg_id,
            1,
            mask=active,
            sem="relaxed",
        )
        fits = active & (slot < table_capacity)
        tl.store(
            segment_entry_indices_ptr + seg_id * table_capacity + slot,
            offs.to(tl.int32),
            mask=fits,
        )
        tl.atomic_add(
            segment_entry_overflow_ptr + seg_id,
            1,
            mask=active & ~fits,
            sem="relaxed",
        )
        group_count = tl.load(entry_group_count_ptr + offs, mask=active, other=0)
        tl.atomic_max(
            segment_entry_max_group_count_ptr + seg_id,
            group_count,
            mask=active,
            sem="relaxed",
        )

    @triton.jit
    def _write_direct_records_segment_workqueue_kernel(
        m_ptr, v_ptr, p_ptr,
        age_words_ptr, floor_mid_ptr, floor_old_ptr,
        h_lut_ptr, log_v_min_ptr, inv_log_v_width_ptr,
        entry_elem_start_ptr,
        entry_group_start_ptr,
        entry_group_count_ptr,
        entry_t_off_ptr,
        entry_expert_idx_ptr,
        aos_layout_offset_ptr,
        aos_layout_capacity_ptr,
        expert_record_cursor_ptr,
        expert_overflow_count_ptr,
        total_th_ptr,
        adam_scalar_ptr,
        step_ptr,
        record_out_i32_ptr,
        segment_entry_indices_ptr,
        segment_entry_counts_ptr,
        segment_entry_max_group_count_ptr,
        segment_byte_off,
        segment_n_bytes,
        segment_id,
        segment_table_capacity,
        n_entries,
        master_base_elem,
        lr_over_bc1, sqrt_bc2_inv, eps,
        G: tl.constexpr,
        GROUPS_PER_BLOCK: tl.constexpr,
        H_LUT_BINS: tl.constexpr,
        N_VBINS: tl.constexpr,
        RANDOM_MODE: tl.constexpr,
        RANDOM_SALT: tl.constexpr,
        USE_ADAM_SCALAR_PTR: tl.constexpr,
        WORKER_COUNT: tl.constexpr,
    ):
        BLOCK_SIZE: tl.constexpr = G * GROUPS_PER_BLOCK
        worker_id = tl.program_id(axis=0)
        active_entries = tl.load(segment_entry_counts_ptr + segment_id)
        active_max_group_count = tl.load(segment_entry_max_group_count_ptr + segment_id)
        if (active_entries <= 0) | (active_max_group_count <= 0):
            return
        group_blocks = (active_max_group_count + GROUPS_PER_BLOCK - 1) // GROUPS_PER_BLOCK
        total_tasks = active_entries * group_blocks
        task = worker_id
        while task < total_tasks:
            entry_slot = task % active_entries
            block_group_id = task // active_entries
            block_group_start = block_group_id * GROUPS_PER_BLOCK
            entry_id = tl.load(
                segment_entry_indices_ptr
                + segment_id * segment_table_capacity
                + entry_slot
            )

            entry_elem_start = tl.load(entry_elem_start_ptr + entry_id)
            entry_group_start = tl.load(entry_group_start_ptr + entry_id)
            entry_group_count = tl.load(entry_group_count_ptr + entry_id)
            entry_t_off = tl.load(entry_t_off_ptr + entry_id)
            expert_idx = tl.load(entry_expert_idx_ptr + entry_id)
            expert_blob_byte_off = tl.load(aos_layout_offset_ptr + expert_idx)
            expert_capacity = tl.load(aos_layout_capacity_ptr + expert_idx)
            segment_lo = tl.cast(segment_byte_off, tl.int64)
            segment_hi = segment_lo + tl.cast(segment_n_bytes, tl.int64)
            entry_active = (
                (entry_id < n_entries)
                & (expert_blob_byte_off >= segment_lo)
                & (expert_blob_byte_off < segment_hi)
            )
            expert_record_byte_off = expert_blob_byte_off - segment_lo + 8

            gid_local = block_group_start + tl.arange(0, GROUPS_PER_BLOCK)
            gid_mask = (gid_local < entry_group_count) & entry_active
            elem_lane_2d = tl.broadcast_to(
                tl.arange(0, G)[None, :], (GROUPS_PER_BLOCK, G),
            )
            gid_local_2d = tl.broadcast_to(
                gid_local[:, None], (GROUPS_PER_BLOCK, G),
            )
            elem_off = tl.reshape(
                entry_elem_start + gid_local_2d * G + elem_lane_2d,
                (BLOCK_SIZE,),
            )
            elem_mask = tl.reshape(
                (gid_local_2d < entry_group_count) & entry_active,
                (BLOCK_SIZE,),
            )

            m = tl.load(m_ptr + elem_off, mask=elem_mask, other=0.0)
            v = tl.load(v_ptr + elem_off, mask=elem_mask, other=0.0)
            v_fp32 = v.to(tl.float32)
            lr_scalar = lr_over_bc1
            sqrt_bc2_scalar = sqrt_bc2_inv
            eps_scalar = eps
            if USE_ADAM_SCALAR_PTR:
                lr_scalar = tl.load(adam_scalar_ptr + 0)
                sqrt_bc2_scalar = tl.load(adam_scalar_ptr + 1)
                eps_scalar = tl.load(adam_scalar_ptr + 2)
            denom = tl.sqrt(v) * sqrt_bc2_scalar + eps_scalar
            delta = lr_scalar * m / denom
            delta_sq = delta * delta
            h_per_elem, floor_mid, floor_old = _lookup_h_and_floors_from_lut(
                v_fp32,
                h_lut_ptr,
                log_v_min_ptr,
                inv_log_v_width_ptr,
                floor_mid_ptr,
                floor_old_ptr,
                H_LUT_BINS,
                N_VBINS,
            )
            global_elem_idx_for_score = (
                elem_off.to(tl.int64) + tl.cast(master_base_elem, tl.int64)
            )
            age_code = _age2_load_code(age_words_ptr, global_elem_idx_for_score)
            age_sq = tl.where(
                age_code == 0,
                1.0,
                tl.where(age_code == 1, 16.0, tl.where(age_code == 2, 256.0, 4096.0)),
            )
            floor = tl.where(
                age_code == 2,
                floor_mid,
                tl.where(age_code == 3, floor_old, 0.0),
            )
            delta_eff_sq = tl.maximum(delta_sq, floor)
            elem_score = h_per_elem * age_sq * delta_eff_sq
            if RANDOM_MODE:
                elem_score = _random_unit_from_index(
                    global_elem_idx_for_score, RANDOM_SALT,
                )

            th = tl.load(total_th_ptr)
            step_value = tl.load(step_ptr).to(tl.int32)
            force = (age_code == 3) & _age2_phase_hit(
                global_elem_idx_for_score, step_value, 128, 0x51ED270B,
            ) & elem_mask
            if RANDOM_MODE:
                force = tl.full((BLOCK_SIZE,), False, tl.int1)
            elem_selected = ((elem_score >= th) | force) & elem_mask
            elem_selected_i = elem_selected.to(tl.int64)
            elem_records_n = tl.sum(elem_selected_i, axis=0)
            elem_base = tl.atomic_add(
                expert_record_cursor_ptr + expert_idx,
                elem_records_n,
                mask=elem_records_n > 0,
                sem="relaxed",
            )
            elem_remaining = tl.maximum(expert_capacity - elem_base, 0)
            elem_written_n = tl.minimum(elem_remaining, elem_records_n)
            elem_overflow_n = elem_records_n - elem_written_n
            tl.atomic_add(
                expert_overflow_count_ptr + expert_idx,
                elem_overflow_n,
                mask=elem_overflow_n > 0,
                sem="relaxed",
            )
            elem_rank = tl.cumsum(elem_selected_i, 0) - 1
            elem_slot = elem_base + elem_rank
            elem_write_mask = elem_selected & (elem_slot < expert_capacity)
            elem_idx_local = elem_off.to(tl.int64) - entry_elem_start.to(tl.int64)
            elem_disk_idx = (elem_idx_local + entry_t_off.to(tl.int64)).to(tl.int32)
            elem_word_off = (
                (expert_record_byte_off + elem_slot * 16) >> 2
            ).to(tl.int32)
            p = tl.load(
                p_ptr + elem_off,
                mask=elem_write_mask,
                other=0.0,
            )
            tl.store(record_out_i32_ptr + elem_word_off + 0, elem_disk_idx, mask=elem_write_mask)
            tl.store(record_out_i32_ptr + elem_word_off + 1, p.to(tl.int32, bitcast=True), mask=elem_write_mask)
            tl.store(record_out_i32_ptr + elem_word_off + 2, m.to(tl.int32, bitcast=True), mask=elem_write_mask)
            tl.store(record_out_i32_ptr + elem_word_off + 3, v.to(tl.int32, bitcast=True), mask=elem_write_mask)

            promote0 = (age_code == 0) & _age2_phase_hit(
                global_elem_idx_for_score, step_value, 6, 0xA511E9B3,
            )
            promote1 = (age_code == 1) & _age2_phase_hit(
                global_elem_idx_for_score, step_value, 24, 0x63D83595,
            )
            promote2 = (age_code == 2) & _age2_phase_hit(
                global_elem_idx_for_score, step_value, 96, 0xB5297A4D,
            )
            promoted = tl.minimum(age_code + 1, 3)
            next_age_code = tl.where(
                elem_write_mask,
                0,
                tl.where(promote0 | promote1 | promote2, promoted, age_code),
            )
            age_update_mask = elem_write_mask | promote0 | promote1 | promote2
            _age2_store_code_atomic(
                age_words_ptr,
                global_elem_idx_for_score,
                next_age_code,
                elem_mask & age_update_mask,
            )
            task += WORKER_COUNT

    @triton.jit
    def _stamp_direct_expert_prefix_kernel(
        record_out_i32_ptr,
        expert_idx_ptr,
        aos_layout_offset_ptr,
        aos_layout_capacity_ptr,
        expert_record_cursor_ptr,
        expert_overflow_count_ptr,
        step_ptr,
        step,
        segment_byte_off,
        segment_n_bytes,
        n_experts,
        BLOCK_EXPERTS: tl.constexpr,
        USE_STEP_PTR: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_EXPERTS + tl.arange(0, BLOCK_EXPERTS)
        mask = offs < n_experts
        expert_idx = tl.load(expert_idx_ptr + offs, mask=mask, other=0)
        expert_blob_byte_off = tl.load(aos_layout_offset_ptr + expert_idx, mask=mask, other=-1)
        segment_lo = tl.cast(segment_byte_off, tl.int64)
        segment_hi = segment_lo + tl.cast(segment_n_bytes, tl.int64)
        mask = mask & (expert_blob_byte_off >= segment_lo) & (expert_blob_byte_off < segment_hi)
        prefix_word = ((expert_blob_byte_off - segment_lo) >> 2).to(tl.int64)
        cap = tl.load(aos_layout_capacity_ptr + expert_idx, mask=mask, other=0)
        raw_count = tl.load(expert_record_cursor_ptr + expert_idx, mask=mask, other=0)
        overflow = tl.load(expert_overflow_count_ptr + expert_idx, mask=mask, other=0)
        count = tl.minimum(tl.maximum(raw_count - overflow, 0), cap).to(tl.int32)
        step_value = step
        if USE_STEP_PTR:
            step_value = tl.load(step_ptr).to(tl.int32)
        tl.store(record_out_i32_ptr + prefix_word + 0, count, mask=mask)
        tl.store(
            record_out_i32_ptr + prefix_word + 1,
            step_value + tl.zeros((BLOCK_EXPERTS,), tl.int32),
            mask=mask,
        )


def launch_build_segment_entry_table(
    *,
    entry_expert_idx: torch.Tensor,
    entry_group_count: torch.Tensor,
    aos_layout_offsets: torch.Tensor,
    aos_layout_capacities: torch.Tensor,
    segment_entry_indices: torch.Tensor,
    segment_entry_counts: torch.Tensor,
    segment_entry_max_group_count: torch.Tensor,
    segment_entry_overflow: torch.Tensor,
    segment_bytes: int,
) -> None:
    if not _HAS_TRITON:
        return
    n_entries = int(entry_expert_idx.numel())
    if n_entries <= 0:
        return
    if int(entry_group_count.numel()) < n_entries:
        raise RuntimeError("entry_group_count is smaller than entry_expert_idx")
    if segment_entry_indices.dim() != 2:
        raise RuntimeError("segment_entry_indices must be a 2D tensor")
    n_segments = int(segment_entry_indices.shape[0])
    table_capacity = int(segment_entry_indices.shape[1])
    if n_segments <= 0 or table_capacity <= 0:
        return
    if int(segment_entry_counts.numel()) < n_segments:
        raise RuntimeError("segment_entry_counts is smaller than n_segments")
    if int(segment_entry_max_group_count.numel()) < n_segments:
        raise RuntimeError("segment_entry_max_group_count is smaller than n_segments")
    if int(segment_entry_overflow.numel()) < n_segments:
        raise RuntimeError("segment_entry_overflow is smaller than n_segments")
    block = 256
    grid = ((n_entries + block - 1) // block,)
    _build_segment_entry_table_kernel[grid](
        entry_expert_idx,
        entry_group_count,
        aos_layout_offsets,
        aos_layout_capacities,
        segment_entry_indices,
        segment_entry_counts,
        segment_entry_max_group_count,
        segment_entry_overflow,
        int(n_entries),
        int(n_segments),
        int(table_capacity),
        SEGMENT_BYTES=int(segment_bytes),
        BLOCK=int(block),
    )


def launch_direct_records_for_segment_entry_workqueue(
    *,
    m_view: torch.Tensor,
    v_view: torch.Tensor,
    p_view: torch.Tensor,
    h_lut_values: torch.Tensor,
    h_lut_log_v_min: torch.Tensor,
    h_lut_inv_log_v_width: torch.Tensor,
    entry_elem_start: torch.Tensor,
    entry_group_start: torch.Tensor,
    entry_group_count: torch.Tensor,
    entry_t_off: torch.Tensor,
    entry_expert_idx: torch.Tensor,
    aos_layout_offsets: torch.Tensor,
    aos_layout_capacities: torch.Tensor,
    expert_record_cursor: torch.Tensor,
    expert_overflow_count: torch.Tensor,
    total_th: torch.Tensor,
    adam_scalar_device: Optional[torch.Tensor] = None,
    record_out_i32: torch.Tensor,
    segment_entry_indices: torch.Tensor,
    segment_entry_counts: torch.Tensor,
    segment_entry_max_group_count: torch.Tensor,
    segment_id: int,
    segment_byte_off: int,
    segment_n_bytes: int,
    master_base_elem: int,
    static_max_group_count: int,
    group_size: int,
    lr_over_bc1: float,
    sqrt_bc2_inv: float,
    eps: float,
    random_mode: bool = False,
    random_salt: int = 0,
    age2bit_words: Optional[torch.Tensor] = None,
    age2bit_floor_mid: Optional[torch.Tensor] = None,
    age2bit_floor_old: Optional[torch.Tensor] = None,
    step_device: Optional[torch.Tensor] = None,
) -> None:
    if not _HAS_TRITON:
        return
    n_entries = int(entry_group_count.numel())
    if n_entries <= 0:
        return
    if segment_entry_indices.dim() != 2:
        raise RuntimeError("segment_entry_indices must be a 2D tensor")
    table_capacity = int(segment_entry_indices.shape[1])
    if table_capacity <= 0:
        return
    if h_lut_values is None or total_th is None:
        raise RuntimeError("direct emission requires H LUT and total_th")
    if age2bit_words is None or age2bit_floor_mid is None or age2bit_floor_old is None:
        raise RuntimeError("direct emission requires Age2Bit state")
    if step_device is None:
        raise RuntimeError("direct emission requires step_device for Age2Bit aging")
    use_adam_scalar_ptr = adam_scalar_device is not None
    if use_adam_scalar_ptr:
        if adam_scalar_device.dtype != torch.float32 or adam_scalar_device.numel() < 3:
            raise RuntimeError(
                "adam_scalar_device must be a float32 tensor with at least 3 elements"
            )
    gpb = _groups_per_program(group_size)
    max_group_count = int(static_max_group_count)
    if max_group_count <= 0:
        return
    max_group_blocks = (max_group_count + gpb - 1) // gpb
    max_tasks = max(1, int(table_capacity) * max(1, int(max_group_blocks)))
    workers = min(max_tasks, _direct_emit_worker_count(record_out_i32))
    _write_direct_records_segment_workqueue_kernel[(workers,)](
        m_view,
        v_view,
        p_view,
        age2bit_words,
        age2bit_floor_mid,
        age2bit_floor_old,
        h_lut_values,
        h_lut_log_v_min,
        h_lut_inv_log_v_width,
        entry_elem_start,
        entry_group_start,
        entry_group_count,
        entry_t_off,
        entry_expert_idx,
        aos_layout_offsets,
        aos_layout_capacities,
        expert_record_cursor,
        expert_overflow_count,
        total_th,
        adam_scalar_device if adam_scalar_device is not None else total_th,
        step_device,
        record_out_i32,
        segment_entry_indices,
        segment_entry_counts,
        segment_entry_max_group_count,
        int(segment_byte_off),
        int(segment_n_bytes),
        int(segment_id),
        int(table_capacity),
        n_entries,
        int(master_base_elem),
        float(lr_over_bc1),
        float(sqrt_bc2_inv),
        float(eps),
        G=int(group_size),
        GROUPS_PER_BLOCK=gpb,
        H_LUT_BINS=int(h_lut_values.numel()),
        N_VBINS=int(age2bit_floor_mid.numel()),
        RANDOM_MODE=bool(random_mode),
        RANDOM_SALT=_triton_random_salt(random_mode, random_salt),
        USE_ADAM_SCALAR_PTR=bool(use_adam_scalar_ptr),
        WORKER_COUNT=int(workers),
    )


def launch_stamp_direct_expert_prefixes(
    *,
    record_out_i32: torch.Tensor,
    expert_idx: torch.Tensor,
    aos_layout_offsets: torch.Tensor,
    aos_layout_capacities: torch.Tensor,
    expert_record_cursor: torch.Tensor,
    expert_overflow_count: torch.Tensor,
    step_device: Optional[torch.Tensor] = None,
    step: int,
    segment_byte_off: int,
    segment_n_bytes: int,
    n_experts: int,
) -> None:
    if not _HAS_TRITON:
        return
    n = min(
        max(0, int(n_experts)),
        int(expert_idx.numel()),
    )
    if n <= 0:
        return
    use_step_ptr = step_device is not None
    if use_step_ptr and (step_device.dtype != torch.int32 or step_device.numel() < 1):
        raise RuntimeError("step_device must be an int32 tensor with at least 1 element")
    block = 128
    grid = ((n + block - 1) // block,)
    _stamp_direct_expert_prefix_kernel[grid](
        record_out_i32,
        expert_idx,
        aos_layout_offsets,
        aos_layout_capacities,
        expert_record_cursor,
        expert_overflow_count,
        step_device if step_device is not None else expert_idx,
        int(step),
        int(segment_byte_off),
        int(segment_n_bytes),
        int(n),
        BLOCK_EXPERTS=int(block),
        USE_STEP_PTR=bool(use_step_ptr),
    )


def _groups_per_program(
    group_size: int,
    target_block: Optional[int] = None,
) -> int:
    if target_block is None:
        target_block = int(GROUP_SCAN_TARGET_ELEMENTS)
    gpb = max(1, int(target_block) // max(1, int(group_size)))
    p = 1
    while p < gpb:
        p *= 2
    return p


def _direct_emit_worker_count(tensor: torch.Tensor) -> int:
    device = tensor.device
    if device.type != "cuda":
        return 1
    props = torch.cuda.get_device_properties(device)
    return max(1, int(props.multi_processor_count) * DIRECT_EMIT_WORKERS_PER_SM)


__all__ = [
    "LossyRecordFormat",
    "RECORD",
    "RECORD_BYTES",
    "RECORD_WORDS",
    "PREFIX_BYTES",
    "SECTOR_BYTES",
    "launch_build_segment_entry_table",
    "launch_direct_records_for_segment_entry_workqueue",
    "launch_stamp_direct_expert_prefixes",
    "LossyRecordWriteInputs",
    "LossyRecordWriter",
]
