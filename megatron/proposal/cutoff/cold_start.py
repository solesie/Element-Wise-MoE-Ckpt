"""Cold-start score histogram for the first cutoff scale bootstrap.

This module is not a steady-state validation path. It runs only while the
adaptive controller lacks an initial tau scale, before age2bit state has
diverged from all-zero.  After that, refresh and steady-state cutoff must use
the normal approx probe and direct emission score helper.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _lookup_h_from_lut(
        v_fp32,
        h_lut_ptr,
        log_v_min_ptr,
        inv_log_v_width_ptr,
        H_LUT_BINS: tl.constexpr,
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
        return h_lo + frac * (h_hi - h_lo)

    @triton.jit
    def _hash_u32(x):
        x = x.to(tl.uint32)
        x = (x ^ (x >> 16)) * tl.full((), 0x7FEB352D, tl.uint32)
        x = (x ^ (x >> 15)) * tl.full((), 0x846CA68B, tl.uint32)
        return x ^ (x >> 16)

    @triton.jit
    def _random_unit_from_index(global_idx, salt: tl.constexpr):
        x = global_idx.to(tl.uint32) ^ tl.full((), salt, tl.uint32)
        x = _hash_u32(x)
        mant = x & tl.full((), 0x00FFFFFF, tl.uint32)
        return (mant.to(tl.float32) + 1.0) * (1.0 / 16777217.0)

    @triton.jit
    def _score_block(
        m_ptr, v_ptr,
        h_lut_ptr, log_v_min_ptr, inv_log_v_width_ptr,
        entry_elem_start_ptr,
        entry_group_start_ptr,
        entry_group_count_ptr,
        entry_block_entry_idx_ptr,
        entry_block_local_group_start_ptr,
        entry_block_id,
        master_base_elem,
        lr_over_bc1, sqrt_bc2_inv, eps,
        G: tl.constexpr,
        GROUPS_PER_BLOCK: tl.constexpr,
        H_LUT_BINS: tl.constexpr,
        RANDOM_MODE: tl.constexpr,
        RANDOM_SALT: tl.constexpr,
    ):
        BLOCK_SIZE: tl.constexpr = G * GROUPS_PER_BLOCK
        entry_id = tl.load(entry_block_entry_idx_ptr + entry_block_id)
        block_group_start = tl.load(entry_block_local_group_start_ptr + entry_block_id)
        entry_elem_start = tl.load(entry_elem_start_ptr + entry_id)
        entry_group_start = tl.load(entry_group_start_ptr + entry_id)
        entry_group_count = tl.load(entry_group_count_ptr + entry_id)

        gid_local = block_group_start + tl.arange(0, GROUPS_PER_BLOCK)
        gid_mask = gid_local < entry_group_count
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
        elem_mask = tl.reshape(gid_local_2d < entry_group_count, (BLOCK_SIZE,))
        m = tl.load(m_ptr + elem_off, mask=elem_mask, other=0.0)
        v = tl.load(v_ptr + elem_off, mask=elem_mask, other=0.0)
        v_fp32 = v.to(tl.float32)
        denom = tl.sqrt(v) * sqrt_bc2_inv + eps
        delta = lr_over_bc1 * m / denom
        delta_sq = delta * delta
        h_per_elem = _lookup_h_from_lut(
            v_fp32, h_lut_ptr, log_v_min_ptr, inv_log_v_width_ptr, H_LUT_BINS,
        )
        score = h_per_elem * delta_sq
        if RANDOM_MODE:
            global_elem_idx = elem_off.to(tl.int64) + tl.cast(master_base_elem, tl.int64)
            score = _random_unit_from_index(global_elem_idx, RANDOM_SALT)
        return score, elem_mask

    @triton.jit
    def _reduce_score_minmax_entries_kernel(
        m_ptr, v_ptr,
        h_lut_ptr, log_v_min_ptr, inv_log_v_width_ptr,
        entry_elem_start_ptr,
        entry_group_start_ptr,
        entry_group_count_ptr,
        entry_block_entry_idx_ptr,
        entry_block_local_group_start_ptr,
        smin_ptr,
        smax_ptr,
        n_entry_blocks,
        master_base_elem,
        lr_over_bc1, sqrt_bc2_inv, eps,
        G: tl.constexpr,
        GROUPS_PER_BLOCK: tl.constexpr,
        H_LUT_BINS: tl.constexpr,
        RANDOM_MODE: tl.constexpr,
        RANDOM_SALT: tl.constexpr,
    ):
        entry_block_id = tl.program_id(axis=0)
        if entry_block_id >= n_entry_blocks:
            return
        score, mask = _score_block(
            m_ptr, v_ptr,
            h_lut_ptr, log_v_min_ptr, inv_log_v_width_ptr,
            entry_elem_start_ptr,
            entry_group_start_ptr,
            entry_group_count_ptr,
            entry_block_entry_idx_ptr,
            entry_block_local_group_start_ptr,
            entry_block_id,
            master_base_elem,
            lr_over_bc1, sqrt_bc2_inv, eps,
            G,
            GROUPS_PER_BLOCK,
            H_LUT_BINS,
            RANDOM_MODE,
            RANDOM_SALT,
        )
        positive = mask & (score > 0.0) & (score == score)
        block_min = tl.min(tl.where(positive, score, float("inf")), axis=0)
        block_max = tl.max(tl.where(positive, score, 0.0), axis=0)
        tl.atomic_min(smin_ptr, block_min, sem="relaxed")
        tl.atomic_max(smax_ptr, block_max, sem="relaxed")

    @triton.jit
    def _build_score_log_hist_entries_kernel(
        m_ptr, v_ptr,
        h_lut_ptr, log_v_min_ptr, inv_log_v_width_ptr,
        entry_elem_start_ptr,
        entry_group_start_ptr,
        entry_group_count_ptr,
        entry_block_entry_idx_ptr,
        entry_block_local_group_start_ptr,
        hist_ptr,
        smin_ptr,
        smax_ptr,
        n_bins,
        n_entry_blocks,
        master_base_elem,
        lr_over_bc1, sqrt_bc2_inv, eps,
        G: tl.constexpr,
        GROUPS_PER_BLOCK: tl.constexpr,
        H_LUT_BINS: tl.constexpr,
        RANDOM_MODE: tl.constexpr,
        RANDOM_SALT: tl.constexpr,
    ):
        entry_block_id = tl.program_id(axis=0)
        if entry_block_id >= n_entry_blocks:
            return
        score, mask = _score_block(
            m_ptr, v_ptr,
            h_lut_ptr, log_v_min_ptr, inv_log_v_width_ptr,
            entry_elem_start_ptr,
            entry_group_start_ptr,
            entry_group_count_ptr,
            entry_block_entry_idx_ptr,
            entry_block_local_group_start_ptr,
            entry_block_id,
            master_base_elem,
            lr_over_bc1, sqrt_bc2_inv, eps,
            G,
            GROUPS_PER_BLOCK,
            H_LUT_BINS,
            RANDOM_MODE,
            RANDOM_SALT,
        )
        lo = tl.load(smin_ptr)
        hi = tl.load(smax_ptr)
        valid_bounds = (lo > 0.0) & (hi > lo)
        log_lo = tl.log(lo)
        inv = n_bins / (tl.log(hi + 1.0e-30) - log_lo + 1.0e-30)
        positive = mask & (score > 0.0) & (score == score) & valid_bounds
        lx = tl.log(tl.where(positive, score, 1.0))
        b = ((lx - log_lo) * inv).to(tl.int32)
        b = tl.maximum(tl.minimum(b, n_bins - 1), 0)
        tl.atomic_add(hist_ptr + b, 1, mask=positive, sem="relaxed")


def _groups_per_program(group_size: int) -> int:
    gpb = max(1, 256 // max(1, int(group_size)))
    power = 1
    while power < gpb:
        power <<= 1
    return int(power)


def launch_reduce_score_minmax_for_group(
    *,
    m_flat: torch.Tensor,
    v_flat: torch.Tensor,
    h_lut_values: torch.Tensor,
    h_lut_log_v_min: torch.Tensor,
    h_lut_inv_log_v_width: torch.Tensor,
    entry_elem_start: torch.Tensor,
    entry_group_start: torch.Tensor,
    entry_group_count: torch.Tensor,
    entry_block_entry_idx: torch.Tensor,
    entry_block_local_group_start: torch.Tensor,
    smin: torch.Tensor,
    smax: torch.Tensor,
    master_off_elem: int,
    group_size: int,
    lr_over_bc1: float,
    sqrt_bc2_inv: float,
    eps: float,
    random_mode: bool = False,
    random_salt: int = 0,
) -> None:
    if not _HAS_TRITON:
        return
    n_entry_blocks = int(entry_block_entry_idx.numel())
    if n_entry_blocks <= 0:
        return
    _reduce_score_minmax_entries_kernel[(n_entry_blocks,)](
        m_flat,
        v_flat,
        h_lut_values,
        h_lut_log_v_min,
        h_lut_inv_log_v_width,
        entry_elem_start,
        entry_group_start,
        entry_group_count,
        entry_block_entry_idx,
        entry_block_local_group_start,
        smin,
        smax,
        int(n_entry_blocks),
        int(master_off_elem),
        float(lr_over_bc1),
        float(sqrt_bc2_inv),
        float(eps),
        G=int(group_size),
        GROUPS_PER_BLOCK=_groups_per_program(group_size),
        H_LUT_BINS=int(h_lut_values.numel()),
        RANDOM_MODE=bool(random_mode),
        RANDOM_SALT=(int(random_salt) & 0xFFFFFFFF) if bool(random_mode) else 0,
    )


def launch_build_score_log_hist_for_group(
    *,
    m_flat: torch.Tensor,
    v_flat: torch.Tensor,
    h_lut_values: torch.Tensor,
    h_lut_log_v_min: torch.Tensor,
    h_lut_inv_log_v_width: torch.Tensor,
    entry_elem_start: torch.Tensor,
    entry_group_start: torch.Tensor,
    entry_group_count: torch.Tensor,
    entry_block_entry_idx: torch.Tensor,
    entry_block_local_group_start: torch.Tensor,
    hist: torch.Tensor,
    smin: torch.Tensor,
    smax: torch.Tensor,
    master_off_elem: int,
    group_size: int,
    lr_over_bc1: float,
    sqrt_bc2_inv: float,
    eps: float,
    random_mode: bool = False,
    random_salt: int = 0,
) -> None:
    if not _HAS_TRITON:
        return
    n_entry_blocks = int(entry_block_entry_idx.numel())
    if n_entry_blocks <= 0 or int(hist.numel()) <= 1:
        return
    _build_score_log_hist_entries_kernel[(n_entry_blocks,)](
        m_flat,
        v_flat,
        h_lut_values,
        h_lut_log_v_min,
        h_lut_inv_log_v_width,
        entry_elem_start,
        entry_group_start,
        entry_group_count,
        entry_block_entry_idx,
        entry_block_local_group_start,
        hist,
        smin,
        smax,
        int(hist.numel()),
        int(n_entry_blocks),
        int(master_off_elem),
        float(lr_over_bc1),
        float(sqrt_bc2_inv),
        float(eps),
        G=int(group_size),
        GROUPS_PER_BLOCK=_groups_per_program(group_size),
        H_LUT_BINS=int(h_lut_values.numel()),
        RANDOM_MODE=bool(random_mode),
        RANDOM_SALT=(int(random_salt) & 0xFFFFFFFF) if bool(random_mode) else 0,
    )
