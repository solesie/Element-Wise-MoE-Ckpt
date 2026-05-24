"""Lossy-checkpoint outlier + group selection kernels (production).

Single source of truth for the proposal subsystem's element-/group-
pool selection. The CP write path issues exactly four Triton kernels
per iter; no host-side packed FIFO or K-bounded persistent slab.

Per-iter kernel pipeline
========================

  [tracker_stream — Pass A pre-loop, per BF16 group]
  _outlier_count_kernel:
    * Per element: ``elem_score = h_g × Δθ²`` (Adam m, v).
    * If ``elem_score >= prev_th`` AND group is not LB-picked
      (= ``h_g × prev_dg >= group_th``):
        - ``atomic_add(name_outlier_count[name_idx], 1)``
        - sub-sample ``elem_score`` into the persistent
          stochastic-threshold buffer (deterministic position
          ``global_elem_idx // sample_stride`` — no atomic).
    * Does NOT fold dg. Fold is deferred to ``_aos_write_kernel`` so
      Pass A, ``_group_select_kernel``, and Pass B all see the
      same PRE-fold dg for ``group_picked`` consistency.

  [tracker_stream — pass2, single launch over all groups]
  _group_select_kernel:
    * Per group: ``group_score = h_g × dg``.
    * If ``group_score >= group_th``:
        - ``atomic_add(name_group_count[name_idx], 1)``
        - ``selected_mask[global_gid] = 1`` (i8, persistent).
    * No record write, no fold.

  [tracker_stream — _walk_thresholds, end of step()]
  Stochastic sub-sampling threshold: sort the per-element sample
  buffer, pick K-th order statistic (K = topk_ratio × sample_count).
  Replaces the legacy linear-bin histc walk that over-selected
  under heavy-tail score distributions. Writes new ``elem_th`` and
  ``group_th = elem_th × g_blk`` for the next iter.

  [worker_stream — Pass B per segment, per BF16 group]
  _aos_write_kernel:
    * Re-scans m, v, p with the same condition as Pass A.
    * For selected elements: ``atomic_add(name_slot[name_idx], 1)``
      claims a slot, then writes the 20-byte AOS record (idx_lo,
      idx_hi, p, m, v) into segment staging at
      ``name_offset_bytes[name_idx] + slot × 20``.
    * For non-selected: selective fold ``dg += Σ Δθ²``. Per-segment
      gating ensures each group is folded exactly once across all
      segments that touch its name.

  [worker_stream — Pass B-group per segment, per BF16 group]
  _aos_group_write_kernel:
    * Reads ``selected_mask[global_gid]`` (NOT recomputing score)
      so the decision matches pass2 even if dg drifts between the
      tracker_stream sweep and the worker_stream rescan.
    * For groups marked selected AND in current segment:
        - ``atomic_add(name_group_slot[name_idx], 1)`` claims slot
        - emits g_blk records at
          ``name_group_byte_off[name_idx] + slot × g_blk × 20``
        - writes ``dg = 0`` for the group (= "captured" reset).

Memory
======

Persistent additions beyond tracker buffers:
  * sub-sample buffer (64K × f32 = 256 KB)
  * sample sorted + indices (64K × f32 + 64K × i64 = 768 KB)
  * name_outlier_count / name_group_count / name_slot /
    name_group_slot / name_t_off / aos_byte_off_per_name
    (N_names × i64 each; ~tens of KB at any model scale)
  * selected_mask (N_groups × i8; same scale class as tracker
    buffers, e.g. ~125 MB at 100B per-GPU)

Zero K_total-bounded persistent (= "K_max slab" eliminated). All
records flow directly into the 256-MB segment double buffer at
write time.

Multi-GPU
=========

All counters/masks are per-GPU. For EP/TP partitioning each GPU's
selection is local. A future allgather of the sub-sample buffer
(~8K f32 per rank) would unify the global threshold; not yet wired.

Stress test: ``_test_outlier_per_segment.py`` validates that splitting a
name's elements across multiple kernel launches yields the same AOS
records as a single full-range launch (including ``master_off_elem >
2^31`` for the 100 B-scale path).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# On-disk constants — must match gpu_aos.py / expert_state.py.
RECORD_BYTES: int = 20
RECORD_WORDS: int = 5
PREFIX_BYTES: int = 8
SECTOR_BYTES: int = 4096


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


# ---------------------------------------------------------------------------
# Triton kernels — production path.
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _group_select_kernel(
        h_g_ptr,                          # *bf16, n_groups
        dg_ptr,                           # *f32,  n_groups
        exc_mask_ptr,                     # *bool, n_groups (True = NOT excluded)
        offsets_t_ptr,                    # *i64,  n_names+1 (group-space, ascending)
        name_group_count_ptr,             # *i64,  n_names (output)
        selected_mask_ptr,                # *bool, n_groups (output) — group passes threshold
        threshold_t_ptr,                  # *f32 scalar
        chunk_start,                      # i64 scalar — global group offset
        n_chunk,                          # i32 scalar — groups in this chunk
        n_names: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Per-chunk pass2 kernel. Each group with (h_g × dg) ≥ threshold,
        atomic_inc its name's counter. No sel_buf, no scatter — fully
        atomic-distributed across n_names counters (typically 수천 →
        contention low). Binary search for name_idx (log2 n_names loads
        per group, ~13 at N_names=5K)."""
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_chunk
        # (i64 fix) Explicit cast — chunk_start defaults to i32 in Triton
        # for values < 2^31, but late BF16 groups at 100B scale exceed
        # this. Cast before any arithmetic to prevent silent wrap to
        # negative downstream (= same fix outlier path applied via
        # ``master_off_elem.to(tl.int64)``).
        chunk_start_i64 = tl.cast(chunk_start, tl.int64)
        global_gid = chunk_start_i64 + offs.to(tl.int64)

        # h_g/dg/exc are FULL master-flat tensors; advance by chunk_start.
        load_off = chunk_start_i64 + offs.to(tl.int64)
        h = tl.load(h_g_ptr + load_off, mask=mask, other=0.0)
        dg = tl.load(dg_ptr + load_off, mask=mask, other=0.0)
        exc = tl.load(exc_mask_ptr + load_off, mask=mask, other=False)
        score = h.to(tl.float32) * dg
        # Excluded groups contribute 0 (= won't pass).
        score = tl.where(exc, score, 0.0)
        th = tl.load(threshold_t_ptr)
        selected = (score >= th) & mask

        # Binary search for name_idx: largest n s.t. ofs[n] <= global_gid.
        # ofs_t monotone, length n_names+1.
        lo = tl.zeros((BLOCK,), dtype=tl.int32)
        hi = tl.full((BLOCK,), n_names, dtype=tl.int32)
        for _ in tl.static_range(30):
            in_range = lo < hi
            mid = (lo + hi) >> 1
            pivot = tl.load(offsets_t_ptr + mid)
            cond = in_range & (pivot <= global_gid)
            lo = tl.where(cond, mid + 1, lo)
            hi = tl.where(in_range & ~cond, mid, hi)
        name_idx = lo - 1
        name_idx = tl.maximum(name_idx, 0)
        name_idx = tl.minimum(name_idx, n_names - 1)

        tl.atomic_add(name_group_count_ptr + name_idx, 1, mask=selected)
        # Per-group selection mask (int8): Pass B-group reads this
        # (instead of recomputing score) so it cannot select more than
        # pass2 even if dg has drifted between the two streams.
        tl.store(
            selected_mask_ptr + load_off,
            selected.to(tl.int8),
            mask=mask,
        )


    @triton.jit
    def _outlier_count_kernel(
        m_ptr, v_ptr, accum_ptr, h_g_ptr,
        name_count_ptr,
        ofs_elem_ptr,
        prev_th_ptr, floor_ptr, group_th_ptr,
        n_groups, n_names,
        master_off_elem,
        lr_over_bc1, sqrt_bc2_inv, eps,
        # Sub-sample: persistent buffer for stochastic threshold est.
        # Each element with (global_elem_idx % sample_stride == 0) writes
        # its score to sample_buf at position (global_elem_idx //
        # sample_stride). Deterministic positions → no atomic, no contention.
        sample_buf_ptr,                   # *f32, length SAMPLE_SIZE
        sample_stride,                    # i64 scalar; 0 → sub-sampling disabled
        sample_size,                      # i32 scalar
        G: tl.constexpr,
        GROUPS_PER_BLOCK: tl.constexpr,
    ):
        """Pass A: histogram update + per-expert selection count.

        Mirrors ``_accum_with_outlier_kernel`` for the score / floor /
        group-prediction / selection logic but:
          * No FIFO write (no K_max slab).
          * Selected → atomic_inc(name_count[name]); name = binary
            search ofs_elem for the global elem idx.
          * Does NOT fold into accum (deferred to Pass B).
        """
        BLOCK_SIZE: tl.constexpr = G * GROUPS_PER_BLOCK
        pid = tl.program_id(axis=0)
        gid_start = pid * GROUPS_PER_BLOCK
        elem_off = gid_start * G + tl.arange(0, BLOCK_SIZE)
        elem_mask = elem_off < n_groups * G

        m = tl.load(m_ptr + elem_off, mask=elem_mask, other=0.0)
        v = tl.load(v_ptr + elem_off, mask=elem_mask, other=0.0)
        denom = tl.sqrt(v) * sqrt_bc2_inv + eps
        delta = lr_over_bc1 * m / denom
        delta_sq = delta * delta

        gid_off = gid_start + tl.arange(0, GROUPS_PER_BLOCK)
        gid_mask = gid_off < n_groups
        h_g_per_g = tl.load(h_g_ptr + gid_off, mask=gid_mask, other=0.0)
        h_g_2d = tl.broadcast_to(h_g_per_g[:, None], (GROUPS_PER_BLOCK, G))
        delta_sq_2d = tl.reshape(delta_sq, (GROUPS_PER_BLOCK, G))
        elem_score_2d = h_g_2d.to(tl.float32) * delta_sq_2d
        elem_score = tl.reshape(elem_score_2d, (BLOCK_SIZE,))

        # Floor (atomic prune threshold). Histogram removed (Phase A:
        # threshold from element-axis sub-sample, not histc walk).
        floor = tl.load(floor_ptr)

        # Group-pick lower-bound prediction (same as production).
        prev_dg = tl.load(accum_ptr + gid_off, mask=gid_mask, other=0.0)
        group_score_lb = h_g_per_g.to(tl.float32) * prev_dg
        group_th = tl.load(group_th_ptr)
        group_picked = group_score_lb >= group_th
        group_picked_2d = tl.broadcast_to(
            group_picked[:, None], (GROUPS_PER_BLOCK, G),
        )
        group_picked_flat = tl.reshape(group_picked_2d, (BLOCK_SIZE,))

        prev_th = tl.load(prev_th_ptr)
        selected = (elem_score >= prev_th) & elem_mask & ~group_picked_flat

        # Per-element global idx → name (binary search ofs_elem).
        global_elem_idx = elem_off.to(tl.int64) + master_off_elem.to(tl.int64)
        # Binary search: find largest name such that ofs_elem[name] <= idx.
        # ofs_elem has length n_names+1, monotone increasing.
        lo = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        hi = tl.full((BLOCK_SIZE,), n_names, dtype=tl.int32)
        # log2(N_names_max) iterations; bound at 30 (covers up to 1B names).
        for _ in tl.static_range(30):
            in_range = lo < hi
            mid = (lo + hi) >> 1
            pivot = tl.load(ofs_elem_ptr + mid)
            cond = in_range & (pivot <= global_elem_idx)
            lo = tl.where(cond, mid + 1, lo)
            hi = tl.where(in_range & ~cond, mid, hi)
        name_idx = lo - 1
        name_idx = tl.maximum(name_idx, 0)
        name_idx = tl.minimum(name_idx, n_names - 1)

        # Atomic per-name count for selected elements.
        tl.atomic_add(
            name_count_ptr + name_idx, 1, mask=selected,
        )

        # Sub-sample element score at deterministic positions. Each
        # global_elem_idx that is a multiple of sample_stride writes
        # its score to sample_buf[global_elem_idx // sample_stride].
        # No atomics — positions are unique across all kernel launches
        # (within a single iter all calls use the same sample_stride).
        if sample_stride > 0:
            sample_pos = global_elem_idx // sample_stride
            sample_residual = global_elem_idx - sample_pos * sample_stride
            sample_write = (
                (sample_residual == 0) & elem_mask
                & (sample_pos < sample_size.to(tl.int64))
            )
            tl.store(
                sample_buf_ptr + sample_pos,
                elem_score,
                mask=sample_write,
            )

        # Selective fold MOVED to Pass B (outlier writer). Pass A must
        # leave dg unchanged so that Pass B + pass2 + Pass B-group all
        # read identical PRE-FOLD dg for group_picked decisions.
        # Folding here would make Pass B's group_picked use post-fold
        # dg → mismatch with Pass A's outlier count → atomic slot
        # underflow → trailing record slots stay zero (= INVALID idx
        # written as raw 0, then read back as out-of-range disk_idx).


    @triton.jit
    def _aos_write_kernel(
        m_ptr, v_ptr, p_ptr, accum_ptr, h_g_ptr,
        ofs_elem_ptr,
        name_offset_bytes_ptr,
        name_t_off_ptr,
        name_slot_ptr,
        aos_out_i32_ptr,
        prev_th_ptr, group_th_ptr,
        n_groups, n_names,
        master_off_elem,
        lr_over_bc1, sqrt_bc2_inv, eps,
        G: tl.constexpr,
        GROUPS_PER_BLOCK: tl.constexpr,
        DO_FOLD: tl.constexpr = True,
    ):
        """Pass B: AOS write for selected + selective fold for
        non-selected. Re-scans m, v, p with same selection condition
        as Pass A.

        Race: each element atomically claims its slot via
        ``atomic_add(name_slot[name], 1)``; the returned slot value is
        used to compute the AOS byte offset.
        """
        BLOCK_SIZE: tl.constexpr = G * GROUPS_PER_BLOCK
        pid = tl.program_id(axis=0)
        gid_start = pid * GROUPS_PER_BLOCK
        elem_off = gid_start * G + tl.arange(0, BLOCK_SIZE)
        elem_mask = elem_off < n_groups * G

        m = tl.load(m_ptr + elem_off, mask=elem_mask, other=0.0)
        v = tl.load(v_ptr + elem_off, mask=elem_mask, other=0.0)
        denom = tl.sqrt(v) * sqrt_bc2_inv + eps
        delta = lr_over_bc1 * m / denom
        delta_sq = delta * delta
        delta_sq_2d = tl.reshape(delta_sq, (GROUPS_PER_BLOCK, G))

        gid_off = gid_start + tl.arange(0, GROUPS_PER_BLOCK)
        gid_mask = gid_off < n_groups
        h_g_per_g = tl.load(h_g_ptr + gid_off, mask=gid_mask, other=0.0)
        h_g_2d = tl.broadcast_to(h_g_per_g[:, None], (GROUPS_PER_BLOCK, G))
        elem_score_2d = h_g_2d.to(tl.float32) * delta_sq_2d
        elem_score = tl.reshape(elem_score_2d, (BLOCK_SIZE,))

        prev_dg = tl.load(accum_ptr + gid_off, mask=gid_mask, other=0.0)
        group_score_lb = h_g_per_g.to(tl.float32) * prev_dg
        group_th = tl.load(group_th_ptr)
        group_picked = group_score_lb >= group_th
        group_picked_2d = tl.broadcast_to(
            group_picked[:, None], (GROUPS_PER_BLOCK, G),
        )
        group_picked_flat = tl.reshape(group_picked_2d, (BLOCK_SIZE,))

        prev_th = tl.load(prev_th_ptr)
        selected = (elem_score >= prev_th) & elem_mask & ~group_picked_flat

        # AOS write for selected.
        global_elem_idx = elem_off.to(tl.int64) + master_off_elem.to(tl.int64)
        lo = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        hi = tl.full((BLOCK_SIZE,), n_names, dtype=tl.int32)
        for _ in tl.static_range(30):
            in_range = lo < hi
            mid = (lo + hi) >> 1
            pivot = tl.load(ofs_elem_ptr + mid)
            cond = in_range & (pivot <= global_elem_idx)
            lo = tl.where(cond, mid + 1, lo)
            hi = tl.where(in_range & ~cond, mid, hi)
        name_idx = lo - 1
        name_idx = tl.maximum(name_idx, 0)
        name_idx = tl.minimum(name_idx, n_names - 1)

        # per-segment filter: if name_offset_bytes[name] < 0,
        # this name is NOT in the current segment — skip atomic_inc + write.
        # Allows one Pass B kernel call to process the full BF16 group
        # while only writing records for in-segment names.
        name_byte_off_raw = tl.load(name_offset_bytes_ptr + name_idx)
        in_segment = name_byte_off_raw >= 0
        valid_write = selected & in_segment

        # Atomic claim per-name slot.
        slot = tl.atomic_add(
            name_slot_ptr + name_idx, 1, mask=valid_write,
        )

        # Defensive: clamp sentinel (-1) byte_off to 0 + invalidated
        # slot for arithmetic, so the pointer arithmetic below stays
        # in-bounds even on lanes with mask=False. CUDA debug builds /
        # compute-sanitiser treat OOB pointer COMPUTATION as illegal
        # even when the store is masked out.
        name_byte_off = tl.where(
            in_segment, name_byte_off_raw, tl.zeros_like(name_byte_off_raw),
        )
        slot_safe = tl.where(valid_write, slot, tl.zeros_like(slot))

        # Compute byte offset within AOS layout. RECORD_BYTES=20 inlined
        # because Triton's @jit can't read module-level globals; see
        # NOTE at top of file.
        rec_byte_off = name_byte_off + slot_safe.to(tl.int64) * 20
        word_off = (rec_byte_off >> 2).to(tl.int32)

        # On-disk idx = local-idx-within-expert + name_t_off.
        name_start = tl.load(ofs_elem_ptr + name_idx)
        local_idx_in_expert = (
            global_elem_idx - name_start
            + tl.load(name_t_off_ptr + name_idx)
        )

        p = tl.load(p_ptr + elem_off, mask=elem_mask, other=0.0)
        # Split disk_idx into low/high 32-bit words.
        idx_lo = local_idx_in_expert.to(tl.int32)
        idx_hi = (local_idx_in_expert >> 32).to(tl.int32)

        p_i = p.to(tl.int32, bitcast=True)
        m_i = m.to(tl.int32, bitcast=True)
        v_i = v.to(tl.int32, bitcast=True)

        tl.store(aos_out_i32_ptr + word_off + 0, idx_lo, mask=valid_write)
        tl.store(aos_out_i32_ptr + word_off + 1, idx_hi, mask=valid_write)
        tl.store(aos_out_i32_ptr + word_off + 2, p_i,    mask=valid_write)
        tl.store(aos_out_i32_ptr + word_off + 3, m_i,    mask=valid_write)
        tl.store(aos_out_i32_ptr + word_off + 4, v_i,    mask=valid_write)

        # Selective fold (now name_idx is in scope from bsearch above).
        # per-segment gating: each group is folded ONLY by the
        # segment that owns it. Other segments skip fold for that group
        # to avoid multi-fold. All g_blk elements of a group share the
        # same name_idx (name boundaries align with group boundaries),
        # so we take name_idx of the first element of each group →
        # in_segment mask per group.
        if DO_FOLD:
            # All g_blk elements within a group share the same
            # name_idx (names align with groups), so axis=1 max ==
            # any element's name_idx. Avoids 2D slicing which Triton
            # doesn't support.
            name_idx_2d = tl.reshape(name_idx, (GROUPS_PER_BLOCK, G))
            group_name_idx = tl.max(name_idx_2d, axis=1)
            nob_first = tl.load(
                name_offset_bytes_ptr + group_name_idx, mask=gid_mask, other=-1,
            )
            group_in_segment = (nob_first >= 0)
            selected_2d = tl.reshape(selected, (GROUPS_PER_BLOCK, G))
            delta_sq_kept = tl.where(selected_2d, 0.0, delta_sq_2d)
            block_sums = tl.sum(delta_sq_kept, axis=1)
            fold_mask = gid_mask & group_in_segment
            tl.store(accum_ptr + gid_off, prev_dg + block_sums, mask=fold_mask)


def launch_aos_write_for_subrange(
    m_view: torch.Tensor,           # bf16/f32, n_sub_elements
    v_view: torch.Tensor,           # bf16/f32, same shape as m_view
    p_view: torch.Tensor,           # bf16/f32, same shape
    accum_view: torch.Tensor,       # f32, n_sub_groups (= n_sub_elements // g_blk)
    h_g_view: torch.Tensor,         # bf16, n_sub_groups
    ofs_elem_full: torch.Tensor,    # i64, N_names+1 (full, kernel binary-searches in this)
    name_offset_bytes: torch.Tensor,# i64, N_names — outlier byte_off per name in AOS layout
    name_t_off: torch.Tensor,       # i64, N_names — on-disk idx offset per name
    name_slot: torch.Tensor,        # i64, N_names — atomic slot counter per name
    aos_out_i32: torch.Tensor,      # i32, segment_words — segment staging buffer
    prev_th_t: torch.Tensor,
    group_th_t: torch.Tensor,
    master_off_elem: int,           # GLOBAL element offset of m_view[0]
    g_blk: int,
    lr_over_bc1: float,
    sqrt_bc2_inv: float,
    eps: float,
    n_names: int,
    do_fold: bool = True,
) -> None:
    """Per-segment Pass B launcher. Caller provides a *slice* of one
    BF16 group's m/v/p flat (the slice covers exactly the elements
    whose names belong to the current segment's expert range).
    ``master_off_elem`` is the GLOBAL element index of ``m_view[0]``;
    the kernel adds it to per-element local index to form the global
    index used for the binary-search → name lookup.

    AOS write target: ``aos_out_i32`` is the int32-view of the
    segment's staging buffer. Records for name N go into
    ``name_offset_bytes[N] + slot × 20`` byte offset within this
    buffer. ``name_offset_bytes`` must already be ADJUSTED for this
    segment (i.e. ``= absolute_byte_off - seg_byte_lo``).

    Selective fold writes ``accum_view`` in-place (non-atomic; each
    program owns disjoint gid range).
    """
    if not _HAS_TRITON:
        return
    n_sub_groups = int(accum_view.numel())
    if n_sub_groups == 0:
        return
    target_block = 256
    gpb = max(1, target_block // max(1, g_blk))
    p = 1
    while p < gpb:
        p *= 2
    gpb = p
    n_progs = (n_sub_groups + gpb - 1) // gpb
    _aos_write_kernel[(n_progs,)](
        m_view, v_view, p_view, accum_view, h_g_view,
        ofs_elem_full,
        name_offset_bytes, name_t_off, name_slot,
        aos_out_i32,
        prev_th_t, group_th_t,
        n_sub_groups, n_names,
        int(master_off_elem),
        float(lr_over_bc1), float(sqrt_bc2_inv), float(eps),
        G=int(g_blk), GROUPS_PER_BLOCK=gpb,
        DO_FOLD=bool(do_fold),
    )


if _HAS_TRITON:
    @triton.jit
    def _aos_group_write_kernel(
        # AOS segment output buffer (viewed as i32 words).
        out_i32_ptr,
        # BF16-group flat tensors (persistent).
        m_flat_ptr,                       # *f32,  n_groups × g_blk
        v_flat_ptr,                       # *f32,  n_groups × g_blk
        p_flat_ptr,                       # *f32,  n_groups × g_blk
        # Per-group selection mask (from pass2). True = selected.
        selected_mask_ptr,                # *bool, total_master_groups
        offsets_t_ptr,                    # *i64,  n_names+1 (master ELEMENT coord)
        # Per-name slot counter (atomic_inc); zero'd by caller per iter.
        name_group_slot_ptr,              # *i64,  n_names
        # Per-name AOS byte offset of GROUP region start (= entry.aos_byte_off
        # + outlier_n * 20). -1 sentinel if name not in this segment.
        name_group_byte_off_ptr,          # *i64,  n_names
        # Per-name t_off (on-disk tname start offset, expert-local elem coords).
        name_t_off_ptr,                   # *i64,  n_names
        # Iteration range for this kernel launch (in master group coords).
        chunk_start,                      # i64 scalar
        n_chunk,                          # i32 scalar
        # Selective-fold target: dg set to 0 for picked groups.
        accum_ptr,                        # *f32, n_groups (= dg_ptr typically)
        n_names: tl.constexpr,
        g_blk: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Pass B-group: rescan all groups, write GROUP-pool records
        directly to AOS segment.

        Per group g in this kernel's range:
          - score = h_g[g] × dg[g] × exc_mask[g]
          - if score >= threshold:
              name = bisect(offsets_t, master_off_g + g)
              slot = atomic_add(name_group_slot[name], 1)
              byte_off = name_group_byte_off[name]   # GROUP region start
              if byte_off >= 0:                       # name in this segment
                  for j in 0..g_blk:
                      record_byte = byte_off + slot * g_blk * 20 + j * 20
                      elem_pos_bf16 = g * g_blk + j  (in BF16-group flat coord)
                      disk_idx = elem_pos_bf16 - winoff + t_off
                                = g * g_blk + j  (entry-local)
                                  + name_t_off[name]
                      write 5 i32 record
              # Selective fold: zero dg for picked group.
              accum[g] = 0
        """
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_chunk
        # Local group idx within this BF16 group's flat (0..n_chunk-1).
        local_gid = offs
        # (i64 fix) Explicit cast — chunk_start defaults to i32 in
        # Triton; large master offsets at 100B scale silently wrap to
        # negative. Cast before arithmetic. Same pattern as outlier
        # path's ``master_off_elem.to(tl.int64)``.
        chunk_start_i64 = tl.cast(chunk_start, tl.int64)
        # Master flat group idx (for selected_mask + bisect).
        global_gid = chunk_start_i64 + offs.to(tl.int64)

        # Read pass2's per-group selection result. No score recomputation
        # → consistent with pass2 regardless of dg drift between streams.
        selected_i8 = tl.load(
            selected_mask_ptr + global_gid, mask=mask, other=0,
        )
        selected = (selected_i8 > 0) & mask

        # Binary search for name_idx. offsets_t is in ELEMENT coords
        # (length N_names+1, monotone); compare against global_elem_idx.
        # Gate updates on (lo < hi) so post-convergence iterations
        # don't drift past the answer.
        global_elem_idx = global_gid * g_blk
        lo = tl.zeros((BLOCK,), dtype=tl.int32)
        hi = tl.full((BLOCK,), n_names, dtype=tl.int32)
        for _ in tl.static_range(30):
            in_range = lo < hi
            mid = (lo + hi) >> 1
            pivot = tl.load(offsets_t_ptr + mid)
            cond = in_range & (pivot <= global_elem_idx)
            lo = tl.where(cond, mid + 1, lo)
            hi = tl.where(in_range & ~cond, mid, hi)
        name_idx = lo - 1
        name_idx = tl.maximum(name_idx, 0)
        name_idx = tl.minimum(name_idx, n_names - 1)

        # name_byte_off: skip writing if name not in this segment.
        nbo = tl.load(name_group_byte_off_ptr + name_idx, mask=mask, other=-1)
        in_seg = nbo >= 0
        write_records = selected & in_seg

        # atomic_inc per-name slot for selected & in-seg.
        slot = tl.atomic_add(
            name_group_slot_ptr + name_idx, 1, mask=write_records,
        )

        t_off = tl.load(name_t_off_ptr + name_idx, mask=mask, other=0)
        # Entry-local: name's first elem offset in master = ofs_elem[name]
        # → entry_local_elem_start = global_elem_idx - ofs_elem[name].
        ofs_name_elem = tl.load(offsets_t_ptr + name_idx, mask=mask, other=0)
        entry_local_elem_base = global_elem_idx - ofs_name_elem

        # Write g_blk records per selected group at (nbo + slot*g_blk*20)
        # + j*20. We unroll g_blk via static_range.
        for j in tl.static_range(g_blk):
            # Byte offset of this record's first word in segment.
            byte_off_record = nbo + slot * g_blk * 20 + j * 20
            word_off = (byte_off_record // 4).to(tl.int64)

            # Compute disk_idx + bit-cast f32 → i32 for p/m/v.
            elem_pos = entry_local_elem_base + j  # entry-local elem
            disk_idx = elem_pos + t_off
            # m/v/p reads from BF16-group flats at LOCAL position
            # local_gid * g_blk + j.
            elem_pos_bf16 = local_gid.to(tl.int64) * g_blk + j
            p_f = tl.load(p_flat_ptr + elem_pos_bf16, mask=write_records, other=0.0)
            m_f = tl.load(m_flat_ptr + elem_pos_bf16, mask=write_records, other=0.0)
            v_f = tl.load(v_flat_ptr + elem_pos_bf16, mask=write_records, other=0.0)

            idx_lo = disk_idx.to(tl.int32)
            idx_hi = (disk_idx >> 32).to(tl.int32)
            p_i = p_f.to(tl.int32, bitcast=True)
            m_i = m_f.to(tl.int32, bitcast=True)
            v_i = v_f.to(tl.int32, bitcast=True)

            tl.store(out_i32_ptr + word_off + 0, idx_lo, mask=write_records)
            tl.store(out_i32_ptr + word_off + 1, idx_hi, mask=write_records)
            tl.store(out_i32_ptr + word_off + 2, p_i,    mask=write_records)
            tl.store(out_i32_ptr + word_off + 3, m_i,    mask=write_records)
            tl.store(out_i32_ptr + word_off + 4, v_i,    mask=write_records)

        # Selective fold: dg = 0 for picked groups (in BF16-group flat).
        # Only write if selected (whether or not in current segment —
        # fold must happen exactly once across all segments; if name not
        # in segment, the fold will be done by the OTHER segment's launch).
        # To avoid double-fold across segments, we gate fold on `in_seg`:
        # the segment containing this name does the fold; others skip.
        tl.store(
            accum_ptr + local_gid, 0.0,
            mask=write_records,
        )


def launch_aos_group_write(
    *,
    out_i32: torch.Tensor,
    m_flat: torch.Tensor,
    v_flat: torch.Tensor,
    p_flat: torch.Tensor,
    selected_mask: torch.Tensor,
    offsets_t: torch.Tensor,
    name_group_slot: torch.Tensor,
    name_group_byte_off: torch.Tensor,
    name_t_off: torch.Tensor,
    master_off_g: int,
    n_groups_in_bf16: int,
    n_names: int,
    g_blk: int,
    accum: torch.Tensor,
) -> None:
    """Launch Pass B-group over one BF16 group's range of master groups."""
    if not _HAS_TRITON or n_groups_in_bf16 <= 0:
        return
    BLOCK = 256
    n_progs = (n_groups_in_bf16 + BLOCK - 1) // BLOCK
    _aos_group_write_kernel[(n_progs,)](
        out_i32,
        m_flat, v_flat, p_flat,
        selected_mask, offsets_t,
        name_group_slot, name_group_byte_off, name_t_off,
        int(master_off_g),
        int(n_groups_in_bf16),
        accum,
        n_names=int(n_names),
        g_blk=int(g_blk),
        BLOCK=BLOCK,
    )


def launch_group_select(
    h_g_flat: torch.Tensor,            # bf16, n_groups
    dg_flat: torch.Tensor,             # f32,  n_groups
    exc_mask: torch.Tensor,            # bool, n_groups
    offsets_t: torch.Tensor,           # i64,  n_names+1 (group coord)
    name_group_count: torch.Tensor,    # i64,  n_names (output, caller zero's)
    selected_mask: torch.Tensor,       # bool, n_groups (output; per-group select)
    threshold_t: torch.Tensor,         # f32 scalar
    n_groups_total: int,
    n_names: int,
    chunk_size: int = 0,               # 0 = single launch over all
) -> None:
    """Launch pass2 over groups in chunks, computing per-name selected
    group counts. Single fused kernel replaces the torch mask + cumsum
    + scatter to sel_buf path. No K_max persistent buffer needed —
    only n_names × i64 counter (~tens of KB at scale)."""
    if not _HAS_TRITON or n_groups_total <= 0:
        return
    if chunk_size <= 0:
        chunk_size = n_groups_total
    BLOCK = 1024
    g_done = 0
    while g_done < n_groups_total:
        n_in_chunk = min(chunk_size, n_groups_total - g_done)
        n_progs = (n_in_chunk + BLOCK - 1) // BLOCK
        _group_select_kernel[(n_progs,)](
            h_g_flat, dg_flat, exc_mask, offsets_t, name_group_count,
            selected_mask,
            threshold_t,
            g_done, n_in_chunk,
            n_names=int(n_names),
            BLOCK=BLOCK,
        )
        g_done += n_in_chunk


def launch_outlier_count_for_group(
    m_flat: torch.Tensor,
    v_flat: torch.Tensor,
    accum: torch.Tensor,            # dg_buf for this BF16 group (read-only here; Pass B does fold)
    h_g: torch.Tensor,              # h_g_buf for this BF16 group
    name_count: torch.Tensor,       # global per-name outlier count (shared)
    ofs_elem_full: torch.Tensor,    # global element-space offsets, length N_names+1
    prev_th_t: torch.Tensor,
    floor_t: torch.Tensor,
    group_th_t: torch.Tensor,
    master_off_elem: int,
    g_blk: int,
    lr_over_bc1: float,
    sqrt_bc2_inv: float,
    eps: float,
    n_names: int,
    sample_buf: torch.Tensor,
    sample_stride: int,
    sample_size: int,
) -> None:
    """Launch Pass A kernel for one BF16 group. Counts per-name outlier
    selections + sub-samples element scores for stochastic threshold."""
    if not _HAS_TRITON:
        return
    total_a = (m_flat.numel() // g_blk) * g_blk
    n_groups = total_a // g_blk
    if n_groups == 0:
        return
    target_block = 256
    gpb = max(1, target_block // max(1, g_blk))
    p = 1
    while p < gpb:
        p *= 2
    gpb = p
    n_progs = (n_groups + gpb - 1) // gpb
    _outlier_count_kernel[(n_progs,)](
        m_flat, v_flat, accum, h_g,
        name_count, ofs_elem_full,
        prev_th_t, floor_t, group_th_t,
        n_groups, n_names,
        master_off_elem,
        lr_over_bc1, sqrt_bc2_inv, eps,
        sample_buf, int(sample_stride), int(sample_size),
        G=g_blk, GROUPS_PER_BLOCK=gpb,
    )


def _aos_multisets_equal(
    a: torch.Tensor, b: torch.Tensor,
    name_byte_off: list, n_records_per_name: list,
) -> bool:
    """Compare AOS buffers as MULTISETS of records per name (within-name
    record order may differ due to atomic_add slot ordering)."""
    if a.numel() != b.numel():
        return False
    for byte_off, n_rec in zip(name_byte_off, n_records_per_name):
        if n_rec == 0:
            continue
        word_off = byte_off // 4
        # Each record = 5 i32. Extract n_rec records, sort by first
        # word (idx_lo) as a stable comparison key.
        recs_a = a[word_off : word_off + n_rec * 5].view(n_rec, 5).cpu()
        recs_b = b[word_off : word_off + n_rec * 5].view(n_rec, 5).cpu()
        a_sorted = recs_a[recs_a[:, 0].argsort()]
        b_sorted = recs_b[recs_b[:, 0].argsort()]
        if not torch.equal(a_sorted, b_sorted):
            return False
    return True
