"""Element-wise importance tracker for MoE gradient checkpointing.

New design (see CLAUDE.md):

    score[j] = v̂_t[j] × δ_j²
    δ_j     = |θ_current[j] - θ_CP[j]|
    v̂_t[j]  = v_t[j] / (1 - β₂^t)

θ_CP is a BF16 GPU-resident copy of every tracked expert param's value at
the last time it was checkpointed (either via a per-expert FC or via a
single-element sparse state snapshot). This replaces the legacy error buffer.

Snapshotted or FC'd elements get δ = 0 → score = 0 → not selected, giving
natural rotation without an explicit error-feedback term.

Under EP > 1 a global Top-K across experts would require all-gathering
importance scores across EP ranks (a collective whose cost dominates step
time); this module is restricted to EP = 1 for now.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_DS_EXPERT_IDX_RE = re.compile(r"deepspeed_experts\.(\d+)\.")


def _ids_from_name(name: str) -> Optional[Tuple[int, int]]:
    m_e = _DS_EXPERT_IDX_RE.search(name)
    if m_e is None:
        return None
    m_l = _LAYER_RE.search(name)
    layer = int(m_l.group(1)) if m_l is not None else 0
    return (layer, int(m_e.group(1)))

from ._ds_state import get_mapper
from .expert_state import ExpertState

try:
    from deepspeed.utils import safe_get_full_grad
    _HAS_SAFE_GRAD = True
except ImportError:
    _HAS_SAFE_GRAD = False

    def safe_get_full_grad(param):  # type: ignore
        return param.grad


try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True

    @triton.jit
    def _accum_delta_sq_triton_kernel(
        m_ptr, v_ptr, accum_ptr,
        n_groups,
        lr_over_bc1, sqrt_bc2_inv, eps,
        G: tl.constexpr,           # group size (block_quant_g)
        GROUPS_PER_BLOCK: tl.constexpr,  # # groups handled per program
    ):
        """Each program handles ``GROUPS_PER_BLOCK`` consecutive groups
        (each group has G elements) → BLOCK_SIZE = G * GROUPS_PER_BLOCK
        elements. Larger blocks improve SM occupancy when G is small (e.g.
        g=8). Uses plain load/add/store (no atomic) since each gid is
        unique to one program — accumulation across iters is naturally
        ordered by stream.
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
        block_sums = tl.sum(delta_sq_2d, axis=1)
        gid_off = gid_start + tl.arange(0, GROUPS_PER_BLOCK)
        gid_mask = gid_off < n_groups
        prev = tl.load(accum_ptr + gid_off, mask=gid_mask, other=0.0)
        tl.store(accum_ptr + gid_off, prev + block_sums, mask=gid_mask)

    @triton.jit
    def _accum_with_outlier_kernel(
        m_ptr, v_ptr, accum_ptr, h_g_ptr,
        elem_hist_ptr, sel_idx_ptr, sel_count_ptr,
        prev_th_ptr, floor_ptr, group_th_ptr,
        n_groups, score_max, sel_max,
        master_off_elem,
        lr_over_bc1, sqrt_bc2_inv, eps,
        G: tl.constexpr,
        GROUPS_PER_BLOCK: tl.constexpr,
        N_BINS: tl.constexpr,
    ):
        """accum_sq + outlier-promotion fused kernel with bin-floor pruning.

        Per element i in group g, computes Δθ²_i, element score
        H_g·Δθ²_i, then:
          1. if score ≥ floor: atomically increments
             elem_hist[bin(score)]. Below-floor elements skip the
             histogram entirely — they cannot be selected (floor ≤
             prev_th by construction) and aren't needed for the
             K-th rank determination as long as |above-floor| > K.
             This collapses the atomic_add count from N_total down
             to ~5-15 % under heavy-tail δ² distributions.
          2. if score ≥ prev_th AND group will not be picked by pass2:
             stream-compact via cumsum + 1 scalar atomic per program
             block; emit global idx to sel_idx (capacity sel_max).
             The group-pick prediction uses the unfolded post-update
             δ_g (= prev_dg + Σ Δθ²) which is an UPPER BOUND on the
             actual folded δ_g pass2 will see (fold only decreases).
             Conservative: if (h_g × unfolded_dg) ≥ group_th the
             group is CERTAIN to be picked → skip element emission
             (pure dedup, no false-positives, only false-negatives
             where dedup is missed by ~few %).
          3. accumulates the group sum of Δθ²_i EXCLUDING elements
             that hit (2) — Intent E (self-correcting accumulator).
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
        delta_sq = delta * delta  # [BLOCK_SIZE]
        delta_sq_2d = tl.reshape(delta_sq, (GROUPS_PER_BLOCK, G))

        gid_off = gid_start + tl.arange(0, GROUPS_PER_BLOCK)
        gid_mask = gid_off < n_groups
        h_g_per_g = tl.load(h_g_ptr + gid_off, mask=gid_mask, other=0.0)
        h_g_2d = tl.broadcast_to(h_g_per_g[:, None], (GROUPS_PER_BLOCK, G))
        elem_score_2d = h_g_2d.to(tl.float32) * delta_sq_2d  # [GPB, G]
        elem_score = tl.reshape(elem_score_2d, (BLOCK_SIZE,))

        # 1) Histogram (floor-pruned): only bin scores ≥ floor.
        # Using ``mask=above_floor`` skips the atomic for ~95 % of
        # elements (those below the percentile floor), which dominates
        # this kernel's runtime — going from 188 M atomics → ~9 M.
        # observed_max is no longer atomic-tracked; it's recovered
        # from the histogram (highest non-empty bin) in the dispatch
        # walk, so the per-block atomic_max — 737 K contentious
        # atomics on a single scalar — is gone entirely.
        floor = tl.load(floor_ptr)
        above_floor = (elem_score >= floor) & elem_mask
        bin_idx = (elem_score / score_max * N_BINS).to(tl.int32)
        bin_idx = tl.minimum(bin_idx, N_BINS - 1)
        bin_idx = tl.maximum(bin_idx, 0)
        tl.atomic_add(elem_hist_ptr + bin_idx, 1, mask=above_floor)

        # 2a) Group-pick prediction (dedup): use ``prev_dg`` (= δ_g
        # accumulated up to this iter, BEFORE this iter's update) as
        # a STRICT LOWER BOUND on what pass2 will see post-fold.
        #   prev_dg  ≤  folded_dg  =  prev_dg + Σ_{i ∈ group, not selected} Δθ_i²
        #            ≤  unfolded_dg = prev_dg + Σ_{i ∈ group} Δθ_i²
        # If h_g · prev_dg ≥ group_th, then h_g · folded_dg ≥ group_th
        # holds for ANY fold decision (selected element set is empty,
        # all, or anywhere in between) → group is CERTAINLY picked.
        # Skipping the element under this condition is a SAFE dedup
        # with **zero false positives** (no element is dedup'd whose
        # group could have failed group_th after fold). Trade-off:
        # missed dedup opportunities for groups whose membership
        # changes pass / fail status across the fold (rare in steady
        # state where prev_dg dominates).
        prev_dg = tl.load(accum_ptr + gid_off, mask=gid_mask, other=0.0)
        group_score_lb = h_g_per_g.to(tl.float32) * prev_dg
        group_th = tl.load(group_th_ptr)
        group_picked = group_score_lb >= group_th  # [GROUPS_PER_BLOCK]
        group_picked_2d = tl.broadcast_to(
            group_picked[:, None], (GROUPS_PER_BLOCK, G),
        )
        group_picked_flat = tl.reshape(group_picked_2d, (BLOCK_SIZE,))

        # 2b) Selection: emit idx if score >= prev_th AND its group
        # is NOT certain to be picked. Skipping elements whose group
        # will be picked by pass2 prevents the redundant entry that
        # would otherwise consume K-budget for an element already
        # covered by its group's g-element entry.
        prev_th = tl.load(prev_th_ptr)
        selected = (elem_score >= prev_th) & elem_mask & ~group_picked_flat
        sel_int = selected.to(tl.int32)
        within_block_pos = tl.cumsum(sel_int, axis=0) - sel_int  # 0..block_n-1
        block_n = tl.sum(sel_int, axis=0)
        block_start = tl.atomic_add(sel_count_ptr, block_n)
        write_pos = block_start + within_block_pos
        write_ok = selected & (write_pos < sel_max)
        # Convert BF16-group-local elem_off → GLOBAL elem_off for sel_idx.
        # The dispatch path (line ~1735, ~1993) treats sel_idx contents as
        # GLOBAL coordinates: it does ``searchsorted(offsets_t, idx//g_blk)``
        # to map idx → name, then subtracts each BF16 group's
        # ``master_off_g_elem`` to convert back to BF16-local for index_select
        # on the local ``m_flat``. The kernel's loads (m_ptr/v_ptr/h_g_ptr)
        # use the BF16-group-local elem_off because those pointers are local
        # views; ONLY the sel_idx store needs to be globalised.
        # Pre-fix bug: kernel stored elem_off raw → every outlier from
        # BF16 group ≥ 1 was misattributed to the first names in BF16 group 0
        # (their LOCAL offsets collided with GLOBAL offsets of group 0's
        # leading tensors), concentrating ALL lossy dispatch on those names.
        tl.store(
            sel_idx_ptr + write_pos, elem_off + master_off_elem, mask=write_ok,
        )

        # 3) Selective fold: zero out selected elements before group sum.
        selected_2d = tl.reshape(selected, (GROUPS_PER_BLOCK, G))
        delta_sq_kept = tl.where(selected_2d, 0.0, delta_sq_2d)
        block_sums = tl.sum(delta_sq_kept, axis=1)
        tl.store(accum_ptr + gid_off, prev_dg + block_sums, mask=gid_mask)
except Exception:
    _HAS_TRITON = False


def _accum_delta_sq_triton(flat_m, flat_v, flat_accum, total_a, g, lr_over_bc1, sqrt_bc2_inv, eps):
    """Launch Triton kernel: each program covers ``GROUPS_PER_BLOCK`` groups
    of ``g`` elements (BLOCK_SIZE = G * GROUPS_PER_BLOCK). For small g (e.g.
    8) we pick GROUPS_PER_BLOCK so BLOCK_SIZE ~ 256 — the smallest size that
    saturates an A100 SM well (4 warps × 32 threads = 128 threads + 2× ILP)."""
    n_groups = total_a // g
    if n_groups == 0:
        return
    if not _HAS_TRITON:
        flat = (flat_m[:total_a].to(torch.float32) /
                ((flat_v[:total_a].to(torch.float32).sqrt() * sqrt_bc2_inv) + eps)
                ) * lr_over_bc1
        flat = flat * flat
        flat_accum.add_(flat.view(-1, g).sum(dim=1))
        return
    # Pick GROUPS_PER_BLOCK so the kernel block is at least 256 elements
    # (saturates an A100 SM). For g >= 256 we keep GROUPS_PER_BLOCK = 1.
    target_block = 256
    gpb = max(1, target_block // max(1, g))
    # Round to next power of 2 for Triton.
    p = 1
    while p < gpb:
        p *= 2
    gpb = p
    n_progs = (n_groups + gpb - 1) // gpb
    grid = (n_progs,)
    _accum_delta_sq_triton_kernel[grid](
        flat_m, flat_v, flat_accum,
        n_groups, lr_over_bc1, sqrt_bc2_inv, eps,
        G=g, GROUPS_PER_BLOCK=gpb,
    )


def _accum_with_outlier(
    flat_m, flat_v, flat_accum, h_g_flat,
    elem_hist, sel_idx, sel_count,
    total_a, g,
    prev_th, prev_floor, group_th,
    score_max, sel_max,
    master_off_elem,
    lr_over_bc1, sqrt_bc2_inv, eps,
):
    """Outlier-promotion path: same group sum as ``_accum_delta_sq_triton``
    plus element histogram + selection emission + selective fold (Intent E).
    Falls back to ``_accum_delta_sq_triton`` (group sum only, no outlier
    emission) when Triton is unavailable — outlier promotion requires the
    fused kernel."""
    n_groups = total_a // g
    if n_groups == 0:
        return
    if not _HAS_TRITON:
        # Non-Triton fallback: do the plain group sum so training stays
        # correct; outlier emission is silently disabled. Caller checks
        # sel_count for "did anything emit" semantics.
        flat = (flat_m[:total_a].to(torch.float32) /
                ((flat_v[:total_a].to(torch.float32).sqrt() * sqrt_bc2_inv) + eps)
                ) * lr_over_bc1
        flat = flat * flat
        flat_accum.add_(flat.view(-1, g).sum(dim=1))
        return
    target_block = 256
    gpb = max(1, target_block // max(1, g))
    p = 1
    while p < gpb:
        p *= 2
    gpb = p
    n_progs = (n_groups + gpb - 1) // gpb
    grid = (n_progs,)
    n_bins = int(elem_hist.numel())
    _accum_with_outlier_kernel[grid](
        flat_m, flat_v, flat_accum, h_g_flat,
        elem_hist, sel_idx, sel_count,
        prev_th, prev_floor, group_th,
        n_groups, score_max, sel_max,
        master_off_elem,
        lr_over_bc1, sqrt_bc2_inv, eps,
        G=g, GROUPS_PER_BLOCK=gpb, N_BINS=n_bins,
    )


def _fp32_grad(param: torch.nn.Parameter) -> Optional[torch.Tensor]:
    """FP32 reduced grad for a param (DeepSpeed hp mapping when available)."""
    g = safe_get_full_grad(param) if _HAS_SAFE_GRAD else None
    if g is None:
        g = param.grad
    if g is None:
        return None
    return g.detach().to(torch.float32)


def _fp32_v_t(
    optimizer: torch.optim.Optimizer, param: torch.nn.Parameter
) -> Optional[torch.Tensor]:
    """FP32 Adam exp_avg_sq routed through Fp32StateMapper (works under
    FP16/BF16/ZeRO-0..2)."""
    m = get_mapper()
    if m is not None:
        return m.get_state(param, "exp_avg_sq")
    return optimizer.state.get(param, {}).get("exp_avg_sq")


def _fp32_param(param: torch.nn.Parameter) -> torch.Tensor:
    """FP32 master weights via Fp32StateMapper."""
    m = get_mapper()
    if m is not None:
        return m.get_param(param)
    return param.data.to(torch.float32)


@dataclass
class TrackerStepStats:
    step: int
    total_tracked: int
    k: int
    topk_ratio: float
    max_A: float          # max v̂_t (diagnostic)
    max_delta: float      # max δ (diagnostic)
    score_mean: float
    score_max: float
    selected_bytes_sparse: int
    full_save_bytes: int
    full_save_numel: int
    compute_time_ms: float
    per_param_selected: Dict[str, int] = field(default_factory=dict)
    per_param_score_sum: Dict[str, float] = field(default_factory=dict)


class ImportanceTracker:
    """Element-wise importance tracker with θ_CP deviation-based score."""

    def __init__(
        self,
        model: torch.nn.Module,
        topk_ratio: float = 0.01,
        include_filter: Optional[Iterable[str]] = None,
        exclude_filter: Optional[Iterable[str]] = None,
        expert_only: bool = True,
        expert_pattern: Optional[str] = None,
        selection_mode: str = "hessian",
        num_experts: Optional[int] = None,
        random_seed: int = 0,
        async_writer: Optional["AsyncWriter"] = None,  # type: ignore[name-defined]
        block_quant_g: int = 1,
    ) -> None:
        self._random_seed_salt = int(random_seed) * 2654435761 & 0xFFFFFFFF
        self.model = model
        self.topk_ratio = float(topk_ratio)
        self.block_quant_g = max(1, int(block_quant_g))
        # Per-tensor per-group δ_g accumulator (Σ_step Δθ_i² summed within
        # each contiguous block_quant_g-block). H_g (mean over group) is
        # stored in _H_g and used at score time.
        self._delta_g_accum: Dict[str, torch.Tensor] = {}
        # Group-level H cache (computed at refresh time).
        self._H_g: Dict[str, torch.Tensor] = {}
        self._async_writer = async_writer
        self.expert_only = bool(expert_only)
        self.expert_pattern = expert_pattern
        if selection_mode not in ("v_hat", "random", "hessian"):
            raise ValueError(
                f"selection_mode must be 'v_hat'|'random'|'hessian', got {selection_mode!r}."
            )
        self.selection_mode = selection_mode
        # Production stores group-level diagonal Hessian only — see
        # ``self._H_g`` above (BF16 view into ``_g_fast_h_g_bf16_flat``).
        # When ``selection_mode == 'hessian'``, the score is:
        #     score_elem(i)  = |H_g(group(i))| × Δθ_i²        # single-iter
        #     score_group(g) = |H_g(g)|        × δ_g[g]       # time-window
        # Both pools live on the same per-element axis and compete in the
        # unified_topk walk under a single K-budget. Refresh
        # (``proposal_refresh_finite_diff``) overwrites ``_H_g`` directly
        # via ``_fold_h``; there is no element-level FP32 H cache, no
        # EMA, and no disk persist path. Post-fault recovery relies on
        # the next FD refresh to repopulate H_g.

        # Outlier-promotion (Intent A–E) — enabled ONLY at g_blk >= 2.
        # When ON, the accum_sq Triton kernel additionally
        #   (a) bins H_g·Δθ²_i into _outlier_elem_hist for next iter's
        #       element threshold,
        #   (b) tags elements whose score ≥ _outlier_prev_th into
        #       _outlier_sel_idx (atomic FIFO), and
        #   (c) excludes those elements' Δθ²_i from the dg group sum
        #       (Intent E selective fold).
        # The element pool only makes sense when groups smooth Δθ² —
        # i.e. g >= 2. At g=1 a "group" is one element, so the element
        # pool would compete against the SAME element's δ accumulator
        # under a different time scale — the user-specified semantics
        # is to feed every element's Δθ² straight into δ_g and rank on
        # ``H × δ_g`` alone. Outlier path is therefore force-disabled
        # at g=1 regardless of ``PROPOSAL_OUTLIER_PROMOTE``.
        # All outlier buffers stay None when disabled (zero VRAM cost).
        import os as _os_op
        # Default ON: the unified two-pool top-K (element pool + group
        # pool competing under a single K-budget) IS the production
        # algorithm per CLAUDE.md. The env override remains so the path
        # can still be flipped OFF for ablation. ``OFF``/``0``/``false``
        # disables; anything else (including the empty default) enables.
        _env_outlier_raw = _os_op.environ.get(
            "PROPOSAL_OUTLIER_PROMOTE", "1",
        )
        _env_outlier = _env_outlier_raw not in (
            "0", "false", "FALSE", "off", "OFF",
        )
        self._outlier_promote = _env_outlier and (self.block_quant_g >= 2)
        # Element-pool histogram bin count. Default matches the group-
        # pool histogram (``_g_fast_n_bins = 10000``) so the unified
        # two-pool top-K walk has equal score-axis precision on both
        # pools. Both histograms cover the per-element score axis after
        # the per-element conversion in dispatch (group bin → score /
        # g_blk), so equal bin count = equal threshold quantisation.
        # Override via ``PROPOSAL_OUTLIER_N_BINS`` for ablation.
        self._outlier_n_bins = int(_os_op.environ.get(
            "PROPOSAL_OUTLIER_N_BINS", "10000"
        ))
        # Lazy: alloc on first kernel call (need device + total numel).
        self._outlier_elem_hist: Optional[torch.Tensor] = None  # (n_bins,) i32
        self._outlier_prev_th: Optional[torch.Tensor] = None    # () fp32
        # Floor for histogram pruning: only scores ≥ floor enter the
        # atomic histogram. Cold start = 0 (no pruning); after walk,
        # set to prev_th × 0.5 — captures top ~5-15 % under heavy-tail
        # δ², 6-20× atomic_add reduction, K-th rank still computable.
        self._outlier_prev_floor: Optional[torch.Tensor] = None  # () fp32
        # observed element-score max: recovered from the histogram
        # itself (highest non-empty bin) in the dispatch walk. Element
        # scores are narrowly distributed (Adam normalises δ) and
        # ~40× smaller than group scores, so we can't reuse the
        # group s_max — it would squeeze the histogram into the
        # bottom bins. Per-iter atomic_max in the kernel was the
        # original design but cost ~30 ms in single-scalar contention
        # across 737 K blocks; the bin-quantised estimate is adequate
        # because we only need it to size the next iter's bin range.
        self._outlier_prev_smax: float = 1e-6                    # host
        self._outlier_sel_idx: Optional[torch.Tensor] = None    # (K_max,) i32
        self._outlier_sel_count: Optional[torch.Tensor] = None  # (1,) i32
        # Per-iter pending: list of selected element idx (CPU long).
        # 1-iter pipelined like group selection — dispatched at next
        # tracker.step's _dispatch_pending_payload.
        self._outlier_pending_idx_cpu: Optional[torch.Tensor] = None

        self.num_experts = int(num_experts) if num_experts is not None else None

        self._tracked_names: List[str] = []
        self._name_to_param: Dict[str, torch.nn.Parameter] = {}

        self._full_save_names: List[str] = []
        self._name_to_full_save_param: Dict[str, torch.nn.Parameter] = {}
        self._full_save_numel_per_name: Dict[str, int] = {}

        any_ds_tagged = any(
            getattr(p, "allreduce", True) is False
            for _, p in model.named_parameters()
            if p.requires_grad
        )
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if self.expert_only and any_ds_tagged:
                is_expert = getattr(p, "allreduce", True) is False
            else:
                include_ok = (
                    include_filter is None
                    or any(k in name for k in include_filter)
                    or (self.expert_pattern is not None and self.expert_pattern in name)
                )
                exclude_ok = (
                    exclude_filter is None
                    or not any(k in name for k in exclude_filter)
                )
                is_expert = include_ok and exclude_ok
            if is_expert:
                self._tracked_names.append(name)
                self._name_to_param[name] = p
            else:
                self._full_save_names.append(name)
                self._name_to_full_save_param[name] = p
                self._full_save_numel_per_name[name] = int(p.numel())

        self._flat_offsets: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        for name in self._tracked_names:
            p = self._name_to_param[name]
            self._flat_offsets[name] = (cursor, cursor + p.numel())
            cursor += p.numel()
        self._flat_total = cursor

        self._flat_offset_list: List[Tuple[str, int, int]] = [
            (n, s, e) for n, (s, e) in self._flat_offsets.items()
        ]

        # Persistent GPU scratch — allocated lazily in step().
        # Obsolete: per-tensor scoring (see step()) no longer uses flat
        # concat buffers. Left as None for backward compat with any legacy
        # caller that inspects these attributes.
        self._flat_v_gpu: Optional[torch.Tensor] = None
        self._flat_p_gpu: Optional[torch.Tensor] = None
        self._flat_fc_gpu: Optional[torch.Tensor] = None

        # Delayed-sync cache for histc bounds. step() previously paid a
        # blocking cpu() sync to fetch s_min/s_max for the histc range; with
        # delayed sync we reuse the previous step's actual s_max with a
        # safety margin and never block. Cold start (first call) still pays
        # one sync. s_min is mathematically 0 (scores are v×δ² or |H|×δ²,
        # both products of non-negatives) so we never need to track it.
        self._prev_s_max: Optional[float] = None

        # B2: separate cuda stream for tracker GPU work. Launching
        # update_delta_accum + tracker chunk loop on this stream lets
        # main-stream forward(T+1) / backward(T+1) overlap with tracker
        # work(T). The host-blocking sync is deferred to the NEXT iter's
        # tracker.step entry (1-iter pipeline). Race guard:
        # proposal_dc_snap_grads inserts main_stream.wait_event(prev event)
        # before optimizer.step(T+1) writes m, v.
        self._tracker_stream: Optional["torch.cuda.Stream"] = None
        # Pending state from previous iter's tracker.step. Carried here
        # because cp_hooks dispatches it on the next iter's tracker entry.
        self._pending_tracker_event: Optional["torch.cuda.Event"] = None
        self._pending_dispatch: Optional[Dict] = None

        self.total_tracked = self._flat_total
        self.total_full_save = sum(self._full_save_numel_per_name.values())
        self.full_save_bytes = self.total_full_save * 4

        if self.expert_only and not self._tracked_names:
            raise RuntimeError(
                "ImportanceTracker: expert_only=True but no expert params detected."
            )

        # Precompute (layer, expert_idx) per tracked tensor for rolling-state
        # scatter updates in .step(). O(num_tracked_tensors), runs once.
        self._tname_to_eid: Dict[str, Tuple[int, int]] = {}
        for name in self._tracked_names:
            eid = _ids_from_name(name)
            if eid is not None:
                self._tname_to_eid[name] = eid

        # ExpertState sink (set via set_expert_state). When present,
        # step() appends each expert's selected (indices, p, m, v) tuples
        # to the per-expert lossy FC append log via
        # ``ExpertState.lossy_fc_append``. Experts undergoing lossless FC
        # on the current iter are passed in ``lossless_fc_eids`` and
        # excluded from the top-K scoring + lossy append.
        self._expert_state: Optional[ExpertState] = None

    # ------------------------------------------------------------------
    def set_expert_state(
        self, expert_state: Optional[ExpertState],
    ) -> None:
        """Install an ExpertState sink. When set, ``step()`` routes each
        expert's top-K selections into ``expert_state.lossy_fc_append(...)``.
        Pass ``None`` to detach.
        """
        self._expert_state = expert_state

    # ------------------------------------------------------------------
    def num_tracked_params(self) -> int:
        return self.total_tracked

    def num_full_save_params(self) -> int:
        return self.total_full_save

    def tracked_names(self) -> List[str]:
        return list(self._tracked_names)

    def full_save_names(self) -> List[str]:
        return list(self._full_save_names)

    def full_save_numel_per_name(self) -> Dict[str, int]:
        return dict(self._full_save_numel_per_name)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def gpu_memory_breakdown(self) -> Tuple[float, List[Tuple[str, float]]]:
        """Sum of all GPU tensors held by this tracker (MB) + per-item
        breakdown. Used to compare against torch.cuda.memory_allocated()
        to detect missing accountings.
        """
        items: List[Tuple[str, float]] = []
        for attr in [
            "_g_fast_dg_flat", "_g_fast_h_g_bf16_flat", "_g_fast_exc_mask",
            "_g_fast_chunk_buf", "_g_fast_global_hist", "_g_fast_sel_buf",
            "_g_fast_offsets_t", "_g_fast_arange_g",
            "_g_fast_prev_threshold_t",
            "_dc_stage_gpu",
            "_chunk_scratch_m", "_chunk_scratch_v",
            "_flat_delta_g",
        ]:
            t = getattr(self, attr, None)
            if isinstance(t, torch.Tensor) and t.is_cuda:
                items.append((attr, t.element_size() * t.numel() / 1e6))
        for dict_attr in ("_H_g",):
            d = getattr(self, dict_attr, {}) or {}
            n = 0
            sz = 0.0
            for t in d.values():
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    sz += t.element_size() * t.numel() / 1e6
                    n += 1
            if n > 0:
                items.append((f"{dict_attr}[{n}]", sz))
        total = sum(v for _, v in items)
        return total, items

    @torch.no_grad()
    def _init_accum_sq_flat(self, optimizer) -> None:
        """One-time setup for accum_sq (option A: per-BF16-group ZERO-COPY).

        Splits tracked tensors by their owning BF16-Opt group (via
        Fp32StateMapper coords). For each group, attempts to build a flat
        view across the group's tracked m, v slices — these all live inside
        one master storage so the contiguity check passes (which it didn't
        for the merged-across-groups view). Falls back to chunked scratch
        per-group if some group still doesn't pass.
        """
        if getattr(self, "_flat_layout_done", False):
            return
        mapper = get_mapper()
        g = max(1, int(self.block_quant_g))

        # Bucket tracked tensors by BF16-Opt group index (g_idx). For each
        # bucket we capture m, v refs, per-tensor sizes, and the original
        # tracked-name order so _delta_g_accum[name] views align.
        buckets: Dict[int, Dict[str, list]] = {}
        for n in self._tracked_names:
            p = self._name_to_param.get(n)
            if p is None:
                continue
            m = (mapper.get_state(p, "exp_avg")
                 if mapper is not None
                 else optimizer.state.get(p, {}).get("exp_avg"))
            v = _fp32_v_t(optimizer, p)
            if m is None or v is None:
                continue
            # Resolve which BF16-Opt group this tensor lives in. Use the
            # mapper's coord table when available; otherwise fall back to a
            # synthetic single-bucket index 0 (best-effort for non-DS modes).
            g_idx = 0
            if mapper is not None and hasattr(mapper, "_coords"):
                coord = mapper._coords.get(id(p))
                if coord is not None:
                    g_idx = coord[0]
            b = buckets.setdefault(g_idx, {
                "names": [], "sizes": [], "m_refs": [], "v_refs": [],
                "p_refs": [],
            })
            b["names"].append(n)
            b["sizes"].append(int(p.numel()))
            b["m_refs"].append(m.detach().reshape(-1))
            b["v_refs"].append(v.detach().reshape(-1))
            # FP32 master view for batched p index_select. Same FP32
            # master buffer as ``master_flat = mapper._fp32_flat[g_idx]``,
            # so all p_refs in this bucket share storage and form a
            # zero-copy flat view over the BF16-Opt group's params.
            b["p_refs"].append(_fp32_param(p).detach().reshape(-1))

        if not buckets:
            return

        device = next(iter(buckets.values()))["m_refs"][0].device

        def _try_flat_view(refs):
            """Try to build a single 1-D flat view spanning refs in
            storage-offset order. Returns (flat or None, reason str)."""
            if not refs:
                return None, "empty"
            try:
                base = refs[0]
                base_storage = base.untyped_storage()
                base_ptr = base_storage.data_ptr()
                cur_off = base.storage_offset()
                esize = base.element_size()
                for idx, r in enumerate(refs):
                    if r.untyped_storage().data_ptr() != base_ptr:
                        return None, (
                            f"ref[{idx}] storage ptr "
                            f"{hex(r.untyped_storage().data_ptr())} "
                            f"!= base {hex(base_ptr)}"
                        )
                    if r.storage_offset() != cur_off:
                        return None, (
                            f"ref[{idx}] off={r.storage_offset()}, "
                            f"expected {cur_off}"
                        )
                    if r.element_size() != esize or r.dtype != base.dtype:
                        return None, f"ref[{idx}] dtype/size mismatch"
                    cur_off += r.numel()
                total_n = cur_off - base.storage_offset()
                flat = torch.empty(0, dtype=base.dtype, device=base.device)
                flat.set_(
                    base_storage,
                    storage_offset=base.storage_offset(),
                    size=(total_n,),
                    stride=(1,),
                )
                return flat, None
            except Exception as e:
                return None, f"exception: {e!r}"

        # === Z1 layout unification ===
        # Allocate ONE dg_flat in master order (= sorted-by-g_idx, name order
        # within each master). Per-group dg_buf becomes a narrow view into
        # this flat tensor, and per-name views live in the same memory. This
        # lets Phase 1 compute score = h_g × dg with a SINGLE torch.mul over
        # the whole flat buffer (instead of 160 per-name launches).
        total_elems = 0
        total_groups = 0
        for g_idx in sorted(buckets.keys()):
            b = buckets[g_idx]
            total_elems += sum(b["sizes"])
            total_groups += sum(sz // g for sz in b["sizes"])

        # dg_flat: per-group δ accumulator (state). FP32 required (Δθ²
        # accumulation precision).
        self._g_fast_dg_flat = torch.zeros(
            total_groups, dtype=torch.float32, device=device,
        )
        # h_g_flat: BF16 cache (was FP32). Memory halved. mul with FP32
        # dg auto-promotes to FP32 in chunk_buf.
        self._g_fast_h_g_bf16_flat = torch.zeros(
            total_groups, dtype=torch.bfloat16, device=device,
        )
        # Exclusion mask (bool, 1 byte/element). Was FP32 (4 bytes) → 1/4.
        # False = excluded. mul-with-bool requires cast at use site.
        self._g_fast_exc_mask = torch.ones(
            total_groups, dtype=torch.bool, device=device,
        )
        # X-C: chunk_buf replaces the per-iter flat_sg buffer. Score is
        # computed chunk-by-chunk into this buffer; histc + cumsum/scatter
        # all run on chunk-sized slices. Buffer is reused across chunks AND
        # across iters → fixed transient memory regardless of total_groups.
        # Size: min(total_groups, CHUNK_SIZE) — for small total_groups
        # (g large) we don't waste space.
        import os as _os_chunk
        _chunk_mb = int(_os_chunk.environ.get("PROPOSAL_DG_CHUNK_MB", "100"))
        chunk_size = min(total_groups, max(1, _chunk_mb) * 1024 * 1024 // 4)
        self._g_fast_chunk_size = chunk_size
        self._g_fast_chunk_buf = torch.empty(
            chunk_size, dtype=torch.float32, device=device,
        )
        # No flat_sg / sg_views — chunked path doesn't need them. (kept None
        # for any legacy reference safety.)
        self._g_fast_flat_sg = None
        self._g_fast_sg_views = None
        # Persistent global histogram (reused across chunks).
        self._g_fast_n_bins = 10000
        self._g_fast_global_hist = torch.zeros(
            self._g_fast_n_bins, dtype=torch.float32, device=device,
        )
        # Delayed threshold: this iter uses prev iter's threshold;
        # current iter's hist is built for next iter's threshold.
        self._g_fast_prev_threshold_t = torch.zeros((), dtype=torch.float32,
                                                    device=device)
        self._g_fast_h_built_at = -1  # rebuild marker

        # Build per-group state. Each group's dg_buf is a master-region view
        # into _g_fast_dg_flat.
        self._groups: List[Dict] = []
        self._delta_g_accum = {}
        # Per-name views into the same flat buffer (in master order).
        self._g_fast_name_list: List[str] = []
        self._g_fast_offsets_cpu: List[int] = [0]
        # Note: only h_g_views / exc_views are still used (h_g rebuild /
        # exc mask updates). dg_views / sg_views are no longer needed in
        # the chunked X-C path.
        self._g_fast_h_g_views: List[torch.Tensor] = []
        self._g_fast_exc_views: List[torch.Tensor] = []

        all_zero_copy = True
        zc_reasons: List[str] = []
        cur_off = 0  # cumulative position in flat (master order)
        for g_idx in sorted(buckets.keys()):
            b = buckets[g_idx]
            sizes = b["sizes"]
            names = b["names"]
            m_refs = b["m_refs"]
            v_refs = b["v_refs"]
            p_refs = b["p_refs"]
            group_total = sum(sizes)
            group_groups = group_total // g
            master_off = cur_off  # start of this group's region in flat
            master_dg = self._g_fast_dg_flat.narrow(0, cur_off, group_groups)
            master_h_g = self._g_fast_h_g_bf16_flat.narrow(
                0, cur_off, group_groups,
            )

            # Track this BF16 group's name range in the global name_list
            # — used by the Option-B dispatch path to slice top_idx_sorted
            # / outlier_sorted_global per BF16 group.
            start_name_in_grp = len(self._g_fast_name_list)

            # Per-name views (in this master's region).
            for n, sz in zip(names, sizes):
                n_g = sz // g
                if n_g == 0:
                    continue
                self._delta_g_accum[n] = self._g_fast_dg_flat.narrow(
                    0, cur_off, n_g,
                )
                self._g_fast_h_g_views.append(
                    self._g_fast_h_g_bf16_flat.narrow(0, cur_off, n_g),
                )
                self._g_fast_exc_views.append(
                    self._g_fast_exc_mask.narrow(0, cur_off, n_g),
                )
                self._g_fast_name_list.append(n)
                cur_off += n_g
                self._g_fast_offsets_cpu.append(cur_off)

            end_name_in_grp = len(self._g_fast_name_list)

            m_flat, m_why = _try_flat_view(m_refs)
            v_flat, v_why = _try_flat_view(v_refs)
            p_flat, p_why = _try_flat_view(p_refs)
            zc = (m_flat is not None and v_flat is not None
                  and m_flat.numel() == group_total
                  and v_flat.numel() == group_total)
            if not zc:
                all_zero_copy = False
                zc_reasons.append(
                    f"g_idx={g_idx} m_why={m_why} v_why={v_why}"
                )
            # p_flat is required for Option-B batched dispatch but its
            # absence does not invalidate zero_copy (m/v fallback path
            # still works). Track the reason for diagnostic logging.
            if p_flat is None or p_flat.numel() != group_total:
                p_flat = None  # signals dispatch to fall back per-name
            self._groups.append({
                "g_idx": g_idx,
                "names": names,
                "sizes": sizes,
                "m_refs": m_refs,
                "v_refs": v_refs,
                "p_refs": p_refs,
                "m_flat": m_flat,
                "v_flat": v_flat,
                "p_flat": p_flat,
                "dg_buf": master_dg,  # master-region view into dg_flat
                "h_g_buf": master_h_g,  # master-region view into h_g_bf16_flat
                "master_off": master_off,  # global flat offset (= dg/h_g start)
                "start_name_in_grp": start_name_in_grp,
                "end_name_in_grp": end_name_in_grp,
                "zero_copy": zc,
                "total": group_total,
                "total_aligned": (group_total // g) * g,
            })

        self._zero_copy = all_zero_copy

        # Direct VRAM measurement (real, not estimated).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            _post_alloc = torch.cuda.memory_allocated() / 1e6
            print(f"[proposal] accum_sq_flat init: VRAM after lazy buffers "
                  f"= {_post_alloc:.1f} MB allocated", flush=True)

        try:
            mode_str = "ZERO-COPY all groups" if all_zero_copy else \
                       "MIXED (some FALLBACK)"
            # Persistent memory accounting (only items truly retained
            # across iters; transient chunk-loop tensors like cumsum
            # results, full_kmax, chunk_arange are allocated per-iter and
            # the cuda allocator pools them — they don't persist).
            mem_dg = total_groups * 4 / 1e6
            mem_h_g = total_groups * 2 / 1e6
            mem_exc = total_groups * 1 / 1e6
            mem_chunk = chunk_size * 4 / 1e6
            mem_hist = self._g_fast_n_bins * 4 / 1e6
            # sel_buf is lazy-built on first chunk loop; size depends on
            # K_max which we don't know yet at init time. Estimate
            # K_groups + 5*avg_per_bin × 8 bytes (long).
            K_total_est = max(1, int(self.topk_ratio * total_elems))
            K_groups_est = max(1, K_total_est // g)
            avg_per_bin = max(1, total_groups // self._g_fast_n_bins)
            K_max_est = K_groups_est + 5 * avg_per_bin
            mem_sel = (K_max_est + 1) * 8 / 1e6
            mem_total = (mem_dg + mem_h_g + mem_exc + mem_chunk
                         + mem_hist + mem_sel)
            ideal_mb = total_groups * 4 / 1e6  # dg_flat alone
            print(f"[proposal] accum_sq init: {mode_str}, "
                  f"bf16_groups={len(self._groups)}, "
                  f"total={total_elems} elements, groups={total_groups}, "
                  f"n_tracked={len(self._delta_g_accum)}, "
                  f"chunk_size={chunk_size}",
                  flush=True)
            print(f"[proposal] accum_sq mem persistent: "
                  f"dg_flat={mem_dg:.1f}MB h_g_bf16={mem_h_g:.1f}MB "
                  f"exc_bool={mem_exc:.1f}MB chunk_buf={mem_chunk:.1f}MB "
                  f"sel_buf~={mem_sel:.1f}MB hist={mem_hist:.3f}MB → "
                  f"total={mem_total:.1f}MB "
                  f"(ideal/dg_only={ideal_mb:.1f}MB, "
                  f"ratio={mem_total/max(ideal_mb,1):.2f}x)",
                  flush=True)
            for grp in self._groups:
                print(f"[proposal] accum_sq g_idx={grp['g_idx']}: "
                      f"n_tensors={len(grp['names'])}, "
                      f"total={grp['total']}, "
                      f"zero_copy={grp['zero_copy']}",
                      flush=True)
            if zc_reasons:
                for r in zc_reasons:
                    print(f"[proposal] accum_sq zc-fail {r}", flush=True)
        except Exception:
            pass

        self._flat_layout_done = True

    @torch.no_grad()
    def update_delta_accum(
        self,
        optimizer: torch.optim.Optimizer,
        global_step: int,
    ) -> None:
        """accum_sq fast path. ZERO-COPY mode (preferred): reads DS flat m/v
        in place via storage view, computes Δθ² + group reduce + accumulator
        add in a single Triton kernel — no scratch tensors, no extra VRAM.
        FALLBACK mode: if storage isn't shared (older DS / unusual ZeRO
        config), falls back to flat-scratch path.
        """
        self._init_accum_sq_flat(optimizer)
        if not getattr(self, "_flat_layout_done", False):
            return
        import os as _os_dbg
        # B2 note: DEBUG sync would force a main-stream synchronize() that
        # serializes tracker_stream against forward — disabling 1-iter
        # pipeline. We disable measurement when called from within tracker_stream.
        _dbg_time = (
            _os_dbg.environ.get("PROPOSAL_DEBUG_TIME_ACCUM", "")
            in ("1", "true", "TRUE")
            and not getattr(self, "_inside_tracker_stream", False)
        )
        if _dbg_time:
            import time as _time_dbg
            torch.cuda.synchronize()
            _vram_alloc_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            _t0_dbg = _time_dbg.perf_counter()
        # Adam hyper-params
        beta1 = 0.9
        beta2 = 0.999
        eps_adam = 1e-8
        lr = 1.0
        for grp in optimizer.param_groups:
            if "betas" in grp:
                b = grp["betas"]
                if isinstance(b, (tuple, list)) and len(b) >= 2:
                    beta1 = float(b[0])
                    beta2 = float(b[1])
            if "lr" in grp:
                lr = float(grp["lr"])
            if "eps" in grp:
                eps_adam = float(grp["eps"])
            break
        gs = max(int(global_step), 1)
        bc1 = max(1.0 - beta1 ** gs, 1e-12)
        bc2 = max(1.0 - beta2 ** gs, 1e-12)
        sqrt_bc2_inv = (1.0 / (bc2 ** 0.5))
        lr_over_bc1 = lr / bc1

        g = max(1, int(self.block_quant_g))
        total = sum(grp["total"] for grp in self._groups)

        # Lazy chunked-fallback scratch (only allocated if some BF16-Opt
        # group fails the ZERO-COPY check). Reused across groups + steps.
        scratch_m = getattr(self, "_chunk_scratch_m", None)
        scratch_v = getattr(self, "_chunk_scratch_v", None)
        import os as _os_b
        _budget_mb = int(_os_b.environ.get("PROPOSAL_DG_SCRATCH_MB", "50"))
        CHUNK_BUDGET = max(1, _budget_mb) * 1024 * 1024 // 4  # FP32 elems

        # Outlier-promotion path: lazy-alloc per-iter buffers (prev_th,
        # elem_hist, sel_idx, sel_count) on first call. Atomic counter
        # and histogram are reset to 0 each iter so the kernel can
        # accumulate fresh distribution.
        outlier_on = (
            self._outlier_promote
            and all(grp["zero_copy"] for grp in self._groups)
            and _HAS_TRITON
        )
        if outlier_on:
            device = self._g_fast_dg_flat.device
            n_bins = self._outlier_n_bins
            if self._outlier_elem_hist is None:
                self._outlier_elem_hist = torch.zeros(
                    n_bins, dtype=torch.int32, device=device,
                )
                self._outlier_prev_th = torch.full(
                    (), float("inf"), dtype=torch.float32, device=device,
                )
                self._outlier_prev_floor = torch.zeros(
                    (), dtype=torch.float32, device=device,
                )
                self._outlier_sel_count = torch.zeros(
                    1, dtype=torch.int32, device=device,
                )
            # K_max sized to topk_ratio × N_total (element-equiv budget),
            # plus headroom for early-iter promotion ratio spikes (Intent D).
            K_max = max(1, int(self.topk_ratio * total * 1.5))
            if (self._outlier_sel_idx is None
                    or self._outlier_sel_idx.numel() < K_max):
                self._outlier_sel_idx = torch.empty(
                    K_max, dtype=torch.int32, device=device,
                )
            self._outlier_elem_hist.zero_()
            self._outlier_sel_count.zero_()
            # score_max for THIS iter's bin range = prev iter's actual
            # max + 50 % headroom. Cold start uses small default so
            # iter 1's hist is mostly clamped (acceptable — adapts in
            # 1 iter once observed max is recovered from the histogram).
            score_max = max(self._outlier_prev_smax * 1.5, 1e-30)
            sel_max = int(self._outlier_sel_idx.numel())

        for grp in self._groups:
            dg_buf = grp["dg_buf"]
            total_a = grp["total_aligned"]
            if outlier_on and grp["zero_copy"]:
                # group_th (= elem_th × g_blk; written by the unified-
                # topK walk in dispatch) is consumed by the kernel for
                # LB dedup: element emission is suppressed when its
                # group is certain to be picked by pass2. No false
                # positives → strict no-double-counting at steady state.
                # master_off_elem = this BF16 group's start offset in
                # ELEMENT coordinates (master_off is in groups; × g_blk).
                # The kernel adds this to its BF16-local elem_off when
                # storing to sel_idx so dispatch reads back GLOBAL
                # element indices — see kernel comment above the store.
                master_off_elem = int(grp["master_off"]) * g
                _accum_with_outlier(
                    grp["m_flat"], grp["v_flat"], dg_buf, grp["h_g_buf"],
                    self._outlier_elem_hist, self._outlier_sel_idx,
                    self._outlier_sel_count,
                    total_a, g,
                    self._outlier_prev_th, self._outlier_prev_floor,
                    self._g_fast_prev_threshold_t,
                    score_max, sel_max,
                    master_off_elem,
                    lr_over_bc1, sqrt_bc2_inv, eps_adam,
                )
            elif grp["zero_copy"]:
                _accum_delta_sq_triton(
                    grp["m_flat"], grp["v_flat"], dg_buf,
                    total_a, g, lr_over_bc1, sqrt_bc2_inv, eps_adam,
                )
            else:
                # Per-group chunked fallback (shared persistent scratch).
                sizes = grp["sizes"]
                m_refs = grp["m_refs"]
                v_refs = grp["v_refs"]
                n_params = len(sizes)
                cap = max(CHUNK_BUDGET, max(sizes))
                if (scratch_m is None or scratch_m.numel() < cap
                        or scratch_m.device != dg_buf.device):
                    self._chunk_scratch_m = torch.empty(
                        cap, dtype=torch.float32, device=dg_buf.device)
                    self._chunk_scratch_v = torch.empty(
                        cap, dtype=torch.float32, device=dg_buf.device)
                    scratch_m = self._chunk_scratch_m
                    scratch_v = self._chunk_scratch_v
                i = 0
                dg_offset = 0
                while i < n_params:
                    j = i
                    cnumel = 0
                    while j < n_params and (cnumel == 0 or
                                            cnumel + sizes[j] <= CHUNK_BUDGET):
                        cnumel += sizes[j]
                        j += 1
                    m_dst: List[torch.Tensor] = []
                    v_dst: List[torch.Tensor] = []
                    off = 0
                    for k in range(i, j):
                        sz = sizes[k]
                        m_dst.append(scratch_m.narrow(0, off, sz))
                        v_dst.append(scratch_v.narrow(0, off, sz))
                        off += sz
                    torch._foreach_copy_(m_dst, m_refs[i:j])
                    torch._foreach_copy_(v_dst, v_refs[i:j])
                    m_flat = scratch_m.narrow(0, 0, cnumel)
                    v_flat = scratch_v.narrow(0, 0, cnumel)
                    v_flat.sqrt_().mul_(sqrt_bc2_inv).add_(eps_adam)
                    m_flat.div_(v_flat).mul_(lr_over_bc1).pow_(2)
                    cgroups = cnumel // g
                    if cgroups > 0:
                        inc = m_flat.view(-1, g).sum(dim=1)
                        dg_buf.narrow(0, dg_offset, cgroups).add_(inc)
                    dg_offset += cgroups
                    i = j

        if _dbg_time:
            torch.cuda.synchronize()
            _t1_dbg = _time_dbg.perf_counter()
            _vram_peak = torch.cuda.max_memory_allocated()
            _vram_alloc_after = torch.cuda.memory_allocated()
            _peak_delta = (_vram_peak - _vram_alloc_before) / 1e6
            _persistent_delta = (_vram_alloc_after - _vram_alloc_before) / 1e6
            n_zc = sum(1 for grp in self._groups if grp["zero_copy"])
            n_fb = len(self._groups) - n_zc
            mode = f"ZC={n_zc}/FB={n_fb}"
            print(f"[proposal] update_delta_accum @iter{global_step} ({mode}): "
                  f"{(_t1_dbg - _t0_dbg)*1000:.2f} ms (total={total} elements), "
                  f"vram peak+={_peak_delta:.1f}MB persistent+={_persistent_delta:.1f}MB",
                  flush=True)

    @torch.no_grad()
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _walk_thresholds(
        self, pending: Dict[str, Any], prev_step: int, g_blk: int,
    ) -> None:
        """Compute per-element score thresholds for THIS iter's
        lossy-FC selection and write them into kernel state
        (``self._g_fast_prev_threshold_t``, ``self._outlier_prev_th``,
        ``self._outlier_prev_floor``, ``self._prev_s_max_g``,
        ``self._outlier_prev_smax``).

        Three branches based on group quantisation g_blk and the
        outlier-promotion flag:
          1. g_blk == 1                            : single-pool walk on group hist
          2. g_blk >= 2 and not outlier_promote    : group-pool only walk
          3. outlier_promote and elem_hist present : unified two-pool walk

        See the inline comments in each branch for detailed
        rationale (K-budget arithmetic, FLOOR_MULT, prev_smax
        adaptation, multi-GPU semantics)."""
        # === Unified two-pool top-K: merged-walk on two histograms ===
        # === under a single K-budget.                              ===
        # Role: produce ONE element-axis threshold (and its group-axis
        # equivalent ``elem_th × g_blk``) by ranking element-pool bins
        # and group-pool bins on the SAME per-element score axis, then
        # walking the merged-sorted list under a weighted cumsum until
        # the K-budget = ``topk_ratio × N_total_elements`` is reached.
        # element-pool bin: weight 1, group-pool bin: weight g_blk.
        #
        # CLAUDE.md spec: "production 그룹 score 와 동일 K-budget 에서
        # single-iter element score H_g·Δθ²_i 가 경쟁."
        #
        # Both histograms (element-level + group-level) are populated at
        # PREV iter — element by the outlier kernel's atomic_add, group
        # by pass2's chunk-loop ``histc``. After ev.synchronize above
        # both are valid on host. We:
        #   1. Read both hists to host (4096 + 10000 entries — ~56 KB).
        #   2. Convert each bin to a per-element-score axis:
        #        element bin i:  score = (i + 0.5) × elem_bin_width
        #        group   bin i:  score = (i + 0.5) × group_bin_width / g_blk
        #      (group bin's per-element score = its mean element score.)
        #   3. Each entry's K-budget cost: element = 1, group = g_blk.
        #   4. Walk merged-sorted (descending score) with weighted cumsum
        #      until ≥ K_total = topk_ratio × N_total_elements.
        #   5. The score at K_total → unified threshold:
        #        elem_th  = threshold (used by kernel for outlier emit)
        #        group_th = threshold × g_blk (used by pass2 chunk-loop)
        #   6. Floor analogously at K_total × FLOOR_MULT (kernel atomic
        #      pruning).
        # Effect: elements with high individual contribution displace
        # groups whose mean element contribution is lower — true unified
        # top-K under K-budget. Total selected ≤ K_total elements.
        # Branch by g_blk:
        #   g >= 2 : two-pool unified top-K walk (element pool + group
        #            pool compete on a per-element score axis).
        #   g == 1 : single-pool walk on the group histogram only (the
        #            "group" is one element, ranking on H × δ_g[i] is
        #            the canonical element-wise score; element pool is
        #            disabled — see ``_outlier_promote`` gate).
        if (g_blk == 1 and getattr(self, "_flat_layout_done", False)):
            import numpy as _np_walk
            group_hist = self._g_fast_global_hist
            total_groups_g1 = self._g_fast_dg_flat.numel()
            K_total_g1 = max(1, int(self.topk_ratio * total_groups_g1))

            group_smax = getattr(self, "_prev_s_max_g", None)
            if group_smax is None or group_smax <= 0.0:
                group_smax = 1e-6
            group_score_max_used = max(group_smax * 1.5, 1e-30)
            group_n_bins = group_hist.numel()
            group_bin_width = group_score_max_used / group_n_bins

            group_np = group_hist.cpu().numpy().astype(_np_walk.int64)
            # Descending walk: bin midpoint score, weight = 1 per element
            # (groups are 1 element at g=1).
            group_bin_idx = _np_walk.arange(group_n_bins)
            group_scores = (group_bin_idx + 0.5) * group_bin_width
            order = _np_walk.argsort(-group_scores, kind="stable")
            sorted_scores = group_scores[order]
            sorted_counts = group_np[order]
            cum = _np_walk.cumsum(sorted_counts)
            total_w = int(cum[-1]) if cum.size > 0 else 0

            if total_w < K_total_g1:
                # Cold / under-populated: open the gate.
                group_th_val = 0.0
            else:
                idx_th = int(_np_walk.searchsorted(
                    cum, K_total_g1, side="left",
                ))
                idx_th = min(idx_th, sorted_scores.size - 1)
                group_th_val = float(sorted_scores[idx_th])

            self._g_fast_prev_threshold_t.fill_(group_th_val)

            # Update prev_s_max_g from highest non-empty bin so next
            # iter's bin range adapts to the live distribution.
            group_nonempty = _np_walk.nonzero(group_np)[0]
            if group_nonempty.size > 0:
                top_group_bin = int(group_nonempty[-1])
                self._prev_s_max_g = max(
                    (top_group_bin + 1) * group_bin_width, 1e-30,
                )

            import os as _os_od
            if (_os_od.environ.get("PROPOSAL_OUTLIER_DEBUG", "")
                    in ("1", "true", "TRUE")
                    and prev_step % 5 == 0):
                print(
                    f"[proposal] g1_walk @iter{prev_step}: "
                    f"K_total={K_total_g1} "
                    f"group_hist_total={int(group_np.sum())} "
                    f"group_smax={group_score_max_used:.3e} "
                    f"group_th={group_th_val:.3e}",
                    flush=True,
                )
        elif (g_blk >= 2 and not self._outlier_promote
                and getattr(self, "_flat_layout_done", False)):
            # g_blk >= 2 path with outlier-promote OFF: walk group_hist
            # alone. Without this branch ``_g_fast_prev_threshold_t``
            # stays at its initial 0.0 (set at __init__) → pass2's
            # mask_chunk = chunk_view >= 0 selects EVERY positive group,
            # sel_buf overflows in chunk order, and only the first
            # ~K_max groups in master-flat order (= first names in the
            # first BF16 group, e.g. L01_E00/E01) survive. That misroutes
            # ALL lossy dispatch to those few names regardless of where
            # the real top scores live.
            #
            # K-budget: CLAUDE.md spec is K = topk_ratio × N_total_elements.
            # Each group bin represents one group = g_blk elements, so
            # picking it spends g_blk units of the element budget.
            # K_total_groups = K_total_elements / g_blk = topk_ratio ×
            # n_total_groups. Walk group_hist with weight=1 per bin
            # (group-level) under the K_total_groups budget; the
            # resulting threshold is already in GROUP score units, so
            # it can be written directly into ``_g_fast_prev_threshold_t``
            # (which pass2's chunk-loop compares against group scores).
            import numpy as _np_walk
            group_hist = self._g_fast_global_hist
            total_groups_g2 = self._g_fast_dg_flat.numel()
            K_total_groups = max(
                1, int(self.topk_ratio * total_groups_g2)
            )

            group_smax = getattr(self, "_prev_s_max_g", None)
            if group_smax is None or group_smax <= 0.0:
                group_smax = 1e-6
            group_score_max_used = max(group_smax * 1.5, 1e-30)
            group_n_bins = group_hist.numel()
            group_bin_width = group_score_max_used / group_n_bins

            group_np = group_hist.cpu().numpy().astype(_np_walk.int64)
            group_bin_idx = _np_walk.arange(group_n_bins)
            group_scores = (group_bin_idx + 0.5) * group_bin_width
            order = _np_walk.argsort(-group_scores, kind="stable")
            sorted_scores = group_scores[order]
            sorted_counts = group_np[order]
            cum = _np_walk.cumsum(sorted_counts)
            total_w = int(cum[-1]) if cum.size > 0 else 0

            if total_w < K_total_groups:
                group_th_val = 0.0
            else:
                idx_th = int(_np_walk.searchsorted(
                    cum, K_total_groups, side="left",
                ))
                idx_th = min(idx_th, sorted_scores.size - 1)
                group_th_val = float(sorted_scores[idx_th])

            self._g_fast_prev_threshold_t.fill_(group_th_val)

            group_nonempty = _np_walk.nonzero(group_np)[0]
            if group_nonempty.size > 0:
                top_group_bin = int(group_nonempty[-1])
                self._prev_s_max_g = max(
                    (top_group_bin + 1) * group_bin_width, 1e-30,
                )

            import os as _os_od
            if (_os_od.environ.get("PROPOSAL_OUTLIER_DEBUG", "")
                    in ("1", "true", "TRUE")
                    and prev_step % 5 == 0):
                print(
                    f"[proposal] g2_walk @iter{prev_step}: "
                    f"K_total_groups={K_total_groups} "
                    f"group_hist_total={int(group_np.sum())} "
                    f"group_smax={group_score_max_used:.3e} "
                    f"group_th={group_th_val:.3e}",
                    flush=True,
                )
        elif (self._outlier_promote
                and self._outlier_elem_hist is not None
                and getattr(self, "_flat_layout_done", False)):
            import numpy as _np_walk
            elem_hist = self._outlier_elem_hist
            total_groups_outlier = self._g_fast_dg_flat.numel()
            total_elems_outlier = total_groups_outlier * g_blk
            K_total = max(1, int(self.topk_ratio * total_elems_outlier))
            FLOOR_MULT = 5

            # --- score-axis parameters (host scalars) ---
            elem_score_max_used = max(self._outlier_prev_smax * 1.5, 1e-30)
            elem_n_bins = elem_hist.numel()
            elem_bin_width = elem_score_max_used / elem_n_bins

            group_hist = self._g_fast_global_hist
            group_smax = getattr(self, "_prev_s_max_g", None)
            if group_smax is None or group_smax <= 0.0:
                group_smax = 1e-6
            group_score_max_used = max(group_smax * 1.5, 1e-30)
            group_n_bins = group_hist.numel()
            group_bin_width = group_score_max_used / group_n_bins

            # --- pull both hists to host (small, ~56 KB total) ---
            elem_np = elem_hist.cpu().numpy().astype(_np_walk.int64)
            # group_hist is fp32 (histc accumulator); cast to int64 for
            # weighted cumsum arithmetic.
            group_np = group_hist.cpu().numpy().astype(_np_walk.int64)

            # Per-element score for each bin's midpoint, axis: descending.
            elem_bin_idx = _np_walk.arange(elem_n_bins)
            elem_scores = (elem_bin_idx + 0.5) * elem_bin_width
            group_bin_idx = _np_walk.arange(group_n_bins)
            group_scores_per_elem = (
                (group_bin_idx + 0.5) * group_bin_width / g_blk
            )

            # Concatenate the two pools, then sort by score descending.
            # ``weight``: 1 for element bins, g_blk for group bins.
            all_scores = _np_walk.concatenate([elem_scores, group_scores_per_elem])
            all_counts = _np_walk.concatenate([elem_np, group_np])
            all_weights = _np_walk.concatenate([
                _np_walk.full(elem_n_bins, 1, dtype=_np_walk.int64),
                _np_walk.full(group_n_bins, g_blk, dtype=_np_walk.int64),
            ])
            order = _np_walk.argsort(-all_scores, kind="stable")
            sorted_scores = all_scores[order]
            sorted_weighted = (all_counts[order] * all_weights[order])
            cum = _np_walk.cumsum(sorted_weighted)
            total_weighted = int(cum[-1]) if cum.size > 0 else 0

            # === EXACT unique-budget walk under K_total =================
            # Why "exact" without statistical approximation:
            #   ``elem_hist`` is populated by the outlier kernel only
            #   for elements that are NOT covered by a picked group
            #   (kernel mask ``~group_picked_flat`` keyed off lower-
            #   bound ``group_score_lb = h_g × prev_dg``). So an
            #   element bin's count and a group bin's count refer to
            #   DISJOINT element sets at the steady state — there is
            #   no double-counting to subtract. Adding 1 per element
            #   bin and ``g_blk`` per group bin therefore counts
            #   UNIQUE elements exactly. The walk is just a cumulative
            #   sum until the unique-element budget K_total is met.
            #
            # User intent reflected: element entries (single-iter
            # Δθ²) and group entries (accumulated δ) compete on the
            # same per-element score axis — whichever ranks higher
            # wins each unique slot. With the kernel dedup, the
            # combined saved set has |S| ≈ K_total unique elements
            # AND ⊇ the elements that would be picked by group-only
            # at the same K_total budget (because group entries can
            # only be displaced by element entries with strictly
            # higher per-element score).
            #
            # Multi-GPU: histograms (≈56 KB) are the only state that
            # must be allreduced. The walk runs identically on every
            # rank from the reduced histograms → identical threshold;
            # no per-rank overhead.
            # ============================================================

            # --- threshold at K_total (selection boundary) ---
            if total_weighted < K_total:
                # Cold / under-populated: open both gates — every element
                # / group eligible next iter.
                elem_th_val = 0.0
            else:
                idx_th = int(_np_walk.searchsorted(cum, K_total, side="left"))
                idx_th = min(idx_th, sorted_scores.size - 1)
                # Lower edge of the bin where cumsum reached K_total.
                # Using the bin score (midpoint) is good enough — bin
                # quantisation is the inherent noise floor anyway.
                elem_th_val = float(sorted_scores[idx_th])

            # --- floor at K_total × FLOOR_MULT (kernel atomic prune) ---
            if total_weighted < K_total * FLOOR_MULT:
                # Soft fallback when long tail too short: pull floor in
                # to half of next_th — avoids the all-bin scan oscillation.
                elem_floor_val = elem_th_val * 0.5
            else:
                idx_floor = int(_np_walk.searchsorted(
                    cum, K_total * FLOOR_MULT, side="left",
                ))
                idx_floor = min(idx_floor, sorted_scores.size - 1)
                elem_floor_val = float(sorted_scores[idx_floor])

            # --- write thresholds back (used by THIS iter's compute) ---
            zero_t = torch.zeros(
                (), dtype=torch.float32, device=elem_hist.device,
            )
            self._outlier_prev_th.fill_(elem_th_val)
            self._outlier_prev_floor.fill_(elem_floor_val)
            # Group threshold = elem_th × g_blk (since group score is
            # the SUM over g_blk elements; mean ≥ elem_th ⇔ sum ≥
            # elem_th × g_blk). pass2 reads this directly.
            group_th_val = elem_th_val * g_blk
            self._g_fast_prev_threshold_t.fill_(group_th_val)

            # --- update prev_smax (element & group) for next iter's
            # bin range. Element side: highest non-empty element bin.
            # Group side: highest non-empty group bin (in group score).
            elem_nonempty = _np_walk.nonzero(elem_np)[0]
            if elem_nonempty.size > 0:
                top_elem_bin = int(elem_nonempty[-1])
                self._outlier_prev_smax = max(
                    (top_elem_bin + 1) * elem_bin_width, 1e-30,
                )
            group_nonempty = _np_walk.nonzero(group_np)[0]
            if group_nonempty.size > 0:
                top_group_bin = int(group_nonempty[-1])
                self._prev_s_max_g = max(
                    (top_group_bin + 1) * group_bin_width, 1e-30,
                )

            import os as _os_od
            if (_os_od.environ.get("PROPOSAL_OUTLIER_DEBUG", "")
                    in ("1", "true", "TRUE")
                    and prev_step % 5 == 0):
                elem_total_unweighted = int(elem_np.sum())
                group_total_unweighted = int(group_np.sum())
                sel_count_val = (
                    int(pending.get("outlier_sel_count_snap").item())
                    if pending.get("outlier_sel_count_snap") is not None
                    else 0
                )
                # Predict K-budget split for next iter from current
                # walk's threshold: bins above elem_th in each pool.
                # Element budget = sum of element bin counts above th.
                elem_above = int(elem_np[
                    _np_walk.where(elem_scores >= elem_th_val)[0]
                ].sum())
                # Group budget (in elements) = g_blk × sum of group bin
                # counts whose per-elem score ≥ elem_th.
                group_above_groups = int(group_np[
                    _np_walk.where(group_scores_per_elem >= elem_th_val)[0]
                ].sum())
                group_above_elem_units = group_above_groups * g_blk
                print(
                    f"[proposal] outlier @iter{prev_step}: "
                    f"prev_n_sel={sel_count_val} K_total={K_total} "
                    f"elem_hist_total={elem_total_unweighted} "
                    f"group_hist_total={group_total_unweighted} "
                    f"elem_smax={elem_score_max_used:.3e} "
                    f"group_smax={group_score_max_used:.3e} "
                    f"elem_th={elem_th_val:.3e} "
                    f"group_th={group_th_val:.3e} "
                    f"floor={elem_floor_val:.3e} "
                    f"predict_next: elem={elem_above} "
                    f"group_grps={group_above_groups} "
                    f"(group_in_elem_units={group_above_elem_units}) "
                    f"sum={elem_above + group_above_elem_units}/{K_total}",
                    flush=True,
                )

    def _dispatch_pending_payload(
        self, optimizer: torch.optim.Optimizer,
        current_iter_excluded: Optional[set] = None,
    ) -> None:
        """B2: Dispatch the previous iter's tracker work — runs at the
        START of the next iter's tracker.step. By then the tracker_stream
        GPU work is finished (forward+backward+optimizer of the new iter
        run in parallel on main_stream), so the host-blocking sync here
        is short.

        Operations:
          1. event_tracker.synchronize()  — wait for prev tracker GPU work
          2. seg_bounds.cpu()             — small sync (n_names+1 longs)
          3. for each name: index_select p, m, v + delta_g reset + H2D
          4. lossy_fc_append per expert (async writer queue push)

        ``current_iter_excluded`` (CLAUDE.md correctness): experts that
        FIRED a lossless FC on the CURRENT iter (T) must be excluded
        from this dispatch even though pending was stashed at T-1 with
        T-1's excluded set. Reason: at T-1 the about-to-fire experts had
        non-zero score and may have ended up in top_idx_sorted; at T,
        ``ExpertLosslessFC.fire_if_due`` already enqueued a lossless
        write for them on the same raw-queue (FIFO). If we still append
        T-1's lossy data here, the worker would scatter T-1's stale
        values *on top of* the just-written T LFCL — a recovery
        regression. Skipping them here keeps T's LFCL authoritative.
        """
        import time as _time_dsp
        import os as _os_dsp
        _dsp_dbg = _os_dsp.environ.get("PROPOSAL_OUTLIER_DBG_DSP", "") in ("1",)
        # Narrow peak-memory window for dispatch only — set
        # PROPOSAL_OUTLIER_DBG_DSP_PEAK=1 to log dispatch's GPU
        # transient contribution (in addition to the iter-level peak
        # in cp_hooks). Reset at entry, read at exit; isolates dispatch
        # from forward/backward/optimizer's own transient peaks.
        _dsp_peak = _os_dsp.environ.get("PROPOSAL_OUTLIER_DBG_DSP_PEAK", "") in ("1",)
        _t_dsp_entry = _time_dsp.perf_counter() if _dsp_dbg else 0.0
        if _dsp_peak and torch.cuda.is_available():
            torch.cuda.synchronize()
            _dsp_alloc_at_entry = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        pending = self._pending_dispatch
        if pending is None:
            return
        ev = self._pending_tracker_event
        if ev is not None:
            try:
                ev.synchronize()
            except Exception:
                pass
            self._pending_tracker_event = None
        _t_after_evsync = _time_dsp.perf_counter() if _dsp_dbg else 0.0

        seg_bounds = pending["seg_bounds"]
        top_idx_sorted = pending["top_idx_sorted"]
        name_list = pending["name_list"]
        offsets_cpu = pending["offsets_cpu"]
        # Union T-1's stashed excluded set with T's fired set so any
        # expert that started a lossless FC at T is skipped from T-1's
        # pending lossy append. Without this, T-1 stale lossy values
        # would land on the fresh LFCL/dense buffer (recovery bug).
        excluded_eids = set(pending["excluded_eids"])
        if current_iter_excluded:
            excluded_eids |= set(current_iter_excluded)
        prev_step = pending["step"]

        seg_bounds_cpu = seg_bounds.cpu()
        _t_after_segcpu = _time_dsp.perf_counter() if _dsp_dbg else 0.0
        device = next(self.model.parameters()).device
        g_blk = max(1, int(self.block_quant_g))
        arange_g = self._g_fast_arange_g
        mapper = get_mapper()
        stream = self._tracker_stream

        self._walk_thresholds(pending, prev_step, g_blk)

        # Outlier-promotion: derive elem_idx from the GPU snapshot
        # stashed at prev iter's tracker.step. ev.synchronize() above
        # already waited for the snapshot clones to complete, so the
        # .item() here is essentially free (data already resident).
        sel_idx_snap = pending.get("outlier_sel_idx_snap", None)
        sel_count_snap = pending.get("outlier_sel_count_snap", None)
        outlier_elem_idx: Optional[torch.Tensor] = None
        if sel_idx_snap is not None and sel_count_snap is not None:
            n_sel = int(sel_count_snap.item())
            n_sel = min(n_sel, sel_idx_snap.numel())
            if n_sel > 0:
                outlier_elem_idx = sel_idx_snap[:n_sel].to(torch.long)

        # Pre-sort by tensor name_idx so each tensor's outliers are a
        # contiguous slice — the per-name dispatch loop can use plain
        # integer slicing instead of bool-masking (which would trigger
        # a host sync per name from variable-size output). One CPU sync
        # total here, zero per-name syncs.
        outlier_sorted_global: Optional[torch.Tensor] = None
        outlier_name_bounds_cpu: Optional[List[int]] = None
        # Skip Phase-4 outlier dispatch when env flag set — used to
        # quantify the cost contribution of outlier element dispatch.
        _phase4_skip = _os_dsp.environ.get(
            "PROPOSAL_OUTLIER_SKIP_PHASE4", ""
        ) in ("1",)
        if _phase4_skip:
            outlier_elem_idx = None
        if outlier_elem_idx is not None and outlier_elem_idx.numel() > 0:
            group_idx_per_o = outlier_elem_idx // g_blk
            offsets_t = self._g_fast_offsets_t
            name_idx_per_o = torch.searchsorted(
                offsets_t[1:], group_idx_per_o, right=True
            )
            name_idx_per_o.clamp_(min=0, max=len(name_list) - 1)
            # Sort by name_idx so per-name outlier slice is contiguous.
            sort_idx = name_idx_per_o.argsort()
            outlier_sorted_global = outlier_elem_idx[sort_idx]
            sorted_name_idx = name_idx_per_o[sort_idx]
            # Boundaries: for each name i, slice is [bnd[i]:bnd[i+1]).
            # Cache the arange (len(name_list)+1 is init invariant).
            _name_arange = getattr(self, "_name_arange_cache", None)
            if _name_arange is None or _name_arange.numel() != len(name_list) + 1:
                _name_arange = torch.arange(
                    len(name_list) + 1, device=device,
                )
                self._name_arange_cache = _name_arange
            bnd = torch.searchsorted(sorted_name_idx, _name_arange)
            outlier_name_bounds_cpu = bnd.cpu().tolist()  # ONE sync
        _t_after_olsort = _time_dsp.perf_counter() if _dsp_dbg else 0.0

        from contextlib import nullcontext as _nullcontext
        from .async_writer import get_d2h_stream
        d2h_stream = get_d2h_stream() if torch.cuda.is_available() else None
        sm_tracker = (torch.cuda.stream(stream) if stream is not None
                      else _nullcontext())
        sm_d2h = (torch.cuda.stream(d2h_stream) if d2h_stream is not None
                  else _nullcontext())

        # === Option B: per-BF16-group batched index_select ============
        # Pinned layout (per BF16 group g, sequential):
        #   pinned_field[g_off : g_off + grp_total_groups_elem]    ← all groups
        #   pinned_field[+grp_total_groups_elem : + grp_total]     ← all outliers
        # Order within each region: name iteration order (same as
        # ``name_list`` slice for that BF16 group).
        # Per-name extraction uses two narrows (group_part, outlier_part)
        # → selected_payload[name] = list of sub-entries; writer flattens
        # them into the lossy log naturally.
        # Per BF16 group: 3 batched index_select (m, v, p) on the
        # group's flat views + 1 cat for elem_idx. Total launches /
        # iter ≈ 3 BF16 groups × (1 cat + 3 index_select + 1 cast +
        # 4 d2h) = ~27, vs ~1280 per-name. dg reset is ALSO batched
        # (one global index_fill on _g_fast_dg_flat).
        # Peak transient VRAM: per BF16 group, sequential drain →
        # ~max_grp_total × 4 fields × 4 B (~48 MB at this scale).
        # Per-name fallback path activates if any group's p_flat is
        # None (i.e. _try_flat_view declined to build it for this
        # bucket). In that case the older per-name code path runs.
        # ==============================================================

        # Pre-compute per-name layout (host-side only). Counts are
        # derived directly from seg_bounds / outlier_name_bounds — i.e.
        # whatever pass2 actually produced. Excluded experts ARE
        # included in these counts when their groups/outliers slipped
        # through pass2's exclusion (e.g. lossless FC fired at T but
        # pass2 was stashed at T-1 so it didn't know). Including them
        # here keeps elem_idx_groups / elem_idx_outliers size in sync
        # with name_cnt_g / name_cnt_o (a previous force-to-0 caused
        # repeat_interleave shape mismatch). The writer filters
        # excluded eids when building per_expert_batch, so any wasted
        # bytes for excluded names never make it to disk / DRAM.
        name_cnts: List[int] = []     # full per-name (group_elems + outlier)
        name_cnt_g: List[int] = []    # per-name group element count (cnt_g × g_blk)
        name_cnt_o: List[int] = []    # per-name outlier count
        name_offs: List[int] = [0]
        for i, name in enumerate(name_list):
            s_b_pre = int(seg_bounds_cpu[i].item())
            e_b_pre = int(seg_bounds_cpu[i + 1].item())
            cnt_g_pre = e_b_pre - s_b_pre
            if outlier_name_bounds_cpu is not None:
                cnt_o_pre = max(
                    0,
                    outlier_name_bounds_cpu[i + 1] - outlier_name_bounds_cpu[i],
                )
            else:
                cnt_o_pre = 0
            cnt_g_elem = cnt_g_pre * g_blk
            name_cnt_g.append(cnt_g_elem)
            name_cnt_o.append(cnt_o_pre)
            cnt_e_pre = cnt_g_elem + cnt_o_pre
            name_cnts.append(cnt_e_pre)
            name_offs.append(name_offs[-1] + cnt_e_pre)
        total_pinned = name_offs[-1]

        # Per-BF16-group layout in the global pinned buffer.
        # grp_pinned_off[g] = absolute offset where group g's segment starts
        # grp_groups_size[g] = bytes/elements occupied by groups concat
        # grp_total[g] = groups_concat + outliers_concat for group g
        grp_pinned_off: List[int] = []
        grp_groups_size: List[int] = []
        grp_outliers_size: List[int] = []
        cur_pinned_off = 0
        for grp in self._groups:
            sn = grp["start_name_in_grp"]
            en = grp["end_name_in_grp"]
            g_size = sum(name_cnt_g[sn:en])
            o_size = sum(name_cnt_o[sn:en])
            grp_pinned_off.append(cur_pinned_off)
            grp_groups_size.append(g_size)
            grp_outliers_size.append(o_size)
            cur_pinned_off += g_size + o_size

        pinned_idx: Optional[torch.Tensor] = None
        pinned_p: Optional[torch.Tensor] = None
        pinned_m: Optional[torch.Tensor] = None
        pinned_v: Optional[torch.Tensor] = None
        if total_pinned > 0:
            pinned_idx = torch.empty(total_pinned, dtype=torch.int32, pin_memory=True)
            pinned_p = torch.empty(total_pinned, dtype=torch.float32, pin_memory=True)
            pinned_m = torch.empty(total_pinned, dtype=torch.float32, pin_memory=True)
            pinned_v = torch.empty(total_pinned, dtype=torch.float32, pin_memory=True)

        # selected_payload[name] = list of (idx, p, m, v) sub-entries
        # (groups + outliers, separate). Writer iterates sub-entries
        # and appends them sequentially to the lossy log / DRAM scatter.
        selected_payload: Dict[str, List[Tuple[torch.Tensor, torch.Tensor,
                                                torch.Tensor, torch.Tensor]]] = {}
        per_param_selected: Dict[str, int] = {}
        event_h2d: Optional["torch.cuda.Event"] = None

        # Phase A: tracker_stream — batched dg fill + per-BF16-group
        # batched index_select. GPU intermediates per group are kept in
        # a small list across A/B; sequential-per-group drain limits
        # transient peak to one BF16 group's worth.
        per_grp_gpu: List[Optional[Tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor, torch.Tensor]]] = [None] * len(self._groups)

        with sm_tracker:
            last_valid = int(seg_bounds_cpu[-1].item())
            if last_valid > 0:
                self._g_fast_dg_flat.index_fill_(
                    0, top_idx_sorted[:last_valid].to(torch.long), 0.0,
                )

            # Batched per-name count tensors: build ONE GPU tensor for
            # each of name_cnt_g / name_cnt_o (covering all 130 names
            # across all 12 BF16 groups), then narrow per BF16 group
            # inside the loop. Replaces 24 per-iter torch.tensor()
            # launches (one per group per kind) with 2 launches.
            # Counts vary each iter so we cannot cache the full tensor;
            # but we collapse the H2D into 2 calls instead of 24.
            _all_cnt_g_t = torch.tensor(
                name_cnt_g, dtype=torch.long, device=device,
            )
            _all_cnt_o_t = torch.tensor(
                name_cnt_o, dtype=torch.long, device=device,
            )

            # === Batched Phase A pre-computation ===
            # Replace per-grp [sub(master_off) + mul(g_blk) + add(arange)]
            # with ONE global sub + ONE global mul/add/flatten across the
            # whole top_idx_sorted[:last_valid]. Saves ~36 launches
            # (3 ops × 12 BF16 groups) → 3 launches.
            #
            # Per-grp narrow gives back the per-grp slice without launch.
            # n_per_grp_g (groups count per BF16 grp) comes from seg_bounds.
            # master_off_per_grp_t cached per init.
            _g_off_per_grp_cache = getattr(
                self, "_master_off_per_grp_t_cache", None)
            if _g_off_per_grp_cache is None:
                _g_off_per_grp_cache = torch.tensor(
                    [g["master_off"] for g in self._groups],
                    dtype=torch.long, device=device,
                )
                self._master_off_per_grp_t_cache = _g_off_per_grp_cache
            _master_off_per_grp_t = _g_off_per_grp_cache

            # Global within_off concat: (offsets_cpu[k] - master_off[grp_of_k])
            # × g_blk for all 130 names across all BF16 groups, in name
            # order. Cached at init (init-invariant). Used to globalize
            # the per-grp ``within_off_per_name.repeat_interleave(cnt)``
            # call inside the loop into ONE global repeat_interleave.
            _winoff_concat_cache = getattr(
                self, "_winoff_concat_cache", None)
            if _winoff_concat_cache is None:
                _winoff_list: List[int] = []
                for grp_init in self._groups:
                    sn_i = grp_init["start_name_in_grp"]
                    en_i = grp_init["end_name_in_grp"]
                    moff_i = grp_init["master_off"]
                    for k in range(en_i - sn_i):
                        _winoff_list.append(
                            (offsets_cpu[sn_i + k] - moff_i) * g_blk
                        )
                _winoff_concat_cache = torch.tensor(
                    _winoff_list, dtype=torch.long, device=device,
                )
                self._winoff_concat_cache = _winoff_concat_cache
            _within_off_concat = _winoff_concat_cache

            # Per-grp groups count (from seg_bounds_cpu).
            n_per_grp_g_list = [
                int(seg_bounds_cpu[g["end_name_in_grp"]].item())
                - int(seg_bounds_cpu[g["start_name_in_grp"]].item())
                for g in self._groups
            ]
            n_per_grp_g_t = torch.tensor(
                n_per_grp_g_list, dtype=torch.long, device=device,
            )

            top_active = top_idx_sorted[:last_valid] if last_valid > 0 else None
            elem_idx_groups_bf16_global: Optional[torch.Tensor] = None
            grp_g_starts_in_global: List[int] = [0]
            for n in n_per_grp_g_list:
                grp_g_starts_in_global.append(
                    grp_g_starts_in_global[-1] + n
                )

            if top_active is not None and top_active.numel() > 0:
                # Build per-element master_off via repeat_interleave.
                _moff_per_elem = (
                    _master_off_per_grp_t.repeat_interleave(n_per_grp_g_t)
                )
                local_g_in_grp_global = top_active.sub(_moff_per_elem)
                # elem_idx_groups_bf16_global has shape (last_valid * g_blk,)
                elem_idx_groups_bf16_global = (
                    local_g_in_grp_global.unsqueeze(1) * g_blk
                    + arange_g.unsqueeze(0)
                ).flatten()

            for gi, grp in enumerate(self._groups):
                sn = grp["start_name_in_grp"]
                en = grp["end_name_in_grp"]
                if sn == en:
                    continue
                g_size = grp_groups_size[gi]
                o_size = grp_outliers_size[gi]
                if g_size == 0 and o_size == 0:
                    continue
                master_off_g = grp["master_off"]            # in groups
                master_off_g_elem = master_off_g * g_blk    # in elements
                m_flat_g = grp["m_flat"]
                v_flat_g = grp["v_flat"]
                p_flat_g = grp["p_flat"]

                # Group part: narrow into the GLOBAL pre-computed
                # elem_idx_groups_bf16_global (built once outside the
                # loop with one sub + mul + add). Saves per-grp launches.
                s_b_lo = int(seg_bounds_cpu[sn].item())
                s_b_hi = int(seg_bounds_cpu[en].item())
                if s_b_hi > s_b_lo and elem_idx_groups_bf16_global is not None:
                    grp_start_g = grp_g_starts_in_global[gi] * g_blk
                    grp_n_elems_g = (s_b_hi - s_b_lo) * g_blk
                    elem_idx_groups_bf16 = elem_idx_groups_bf16_global.narrow(
                        0, grp_start_g, grp_n_elems_g,
                    )
                else:
                    elem_idx_groups_bf16 = torch.empty(
                        0, dtype=torch.long, device=device,
                    )

                # Outlier part — same idea (BF16-group-flat coords).
                if (outlier_sorted_global is not None
                        and outlier_name_bounds_cpu is not None):
                    o_lo = outlier_name_bounds_cpu[sn]
                    o_hi = outlier_name_bounds_cpu[en]
                    if o_hi > o_lo:
                        outlier_grp = outlier_sorted_global[o_lo:o_hi]
                        elem_idx_outliers_bf16 = outlier_grp.sub(master_off_g_elem)
                    else:
                        elem_idx_outliers_bf16 = torch.empty(
                            0, dtype=torch.long, device=device,
                        )
                else:
                    elem_idx_outliers_bf16 = torch.empty(
                        0, dtype=torch.long, device=device,
                    )

                # Concat: [groups_concat, outliers_concat] — matches the
                # pinned layout.
                if (elem_idx_groups_bf16.numel() > 0
                        and elem_idx_outliers_bf16.numel() > 0):
                    elem_idx_grp_bf16 = torch.cat([elem_idx_groups_bf16, elem_idx_outliers_bf16])
                elif elem_idx_groups_bf16.numel() > 0:
                    elem_idx_grp_bf16 = elem_idx_groups_bf16
                else:
                    elem_idx_grp_bf16 = elem_idx_outliers_bf16

                if elem_idx_grp_bf16.numel() == 0:
                    continue

                # Three batched index_select on BF16-group-flat coords.
                if (m_flat_g is not None and v_flat_g is not None
                        and p_flat_g is not None):
                    m_sel_grp = m_flat_g.index_select(0, elem_idx_grp_bf16)
                    v_sel_grp = v_flat_g.index_select(0, elem_idx_grp_bf16)
                    p_sel_grp = p_flat_g.index_select(0, elem_idx_grp_bf16)
                    if m_sel_grp.dtype != torch.float32:
                        m_sel_grp = m_sel_grp.to(torch.float32)
                    if v_sel_grp.dtype != torch.float32:
                        v_sel_grp = v_sel_grp.to(torch.float32)
                    if p_sel_grp.dtype != torch.float32:
                        p_sel_grp = p_sel_grp.to(torch.float32)
                else:
                    n_grp_elem = elem_idx_grp_bf16.numel()
                    m_sel_grp = torch.zeros(
                        n_grp_elem, dtype=torch.float32, device=device,
                    )
                    v_sel_grp = torch.zeros_like(m_sel_grp)
                    p_sel_grp = torch.zeros_like(m_sel_grp)

                # Convert IDX from BF16-group-flat → tensor-local (the
                # writer / DRAM scatter expect tensor-local indices).
                # Build per-element within-BF16-group offset via two
                # ``repeat_interleave`` (one per region: groups, outliers).
                # within_off_per_name[k] = (offsets_cpu[sn+k] - master_off_g) × g_blk
                #                                    (in BF16-group ELEMENT coords).
                # within_off_per_name is now a narrow into the global
                # concat (cached at init). Saves the per-grp dict lookup
                # and any potential rebuild path.
                within_off_per_name = _within_off_concat.narrow(
                    0, sn, en - sn,
                )
                groups_off_per_elem: Optional[torch.Tensor] = None
                outliers_off_per_elem: Optional[torch.Tensor] = None
                if elem_idx_groups_bf16.numel() > 0:
                    cnt_g_per_name_t = _all_cnt_g_t.narrow(0, sn, en - sn)
                    groups_off_per_elem = within_off_per_name.repeat_interleave(
                        cnt_g_per_name_t,
                    )
                if elem_idx_outliers_bf16.numel() > 0:
                    cnt_o_per_name_t = _all_cnt_o_t.narrow(0, sn, en - sn)
                    outliers_off_per_elem = within_off_per_name.repeat_interleave(
                        cnt_o_per_name_t,
                    )
                if (groups_off_per_elem is not None
                        and outliers_off_per_elem is not None):
                    off_per_elem = torch.cat([groups_off_per_elem, outliers_off_per_elem])
                elif groups_off_per_elem is not None:
                    off_per_elem = groups_off_per_elem
                else:
                    off_per_elem = outliers_off_per_elem
                elem_idx_grp_tl_i32 = (
                    (elem_idx_grp_bf16 - off_per_elem).to(torch.int32)
                )
                per_grp_gpu[gi] = (elem_idx_grp_tl_i32, p_sel_grp, m_sel_grp, v_sel_grp)

        event_compute: Optional["torch.cuda.Event"] = None
        if stream is not None and total_pinned > 0:
            event_compute = torch.cuda.Event()
            event_compute.record(stream)

        # Phase B: d2h_stream — ONE big copy per field per BF16 group
        # (4 d2h × N_BF16_groups = 12 d2h total; vs 640 per-name).
        # NOTE: tried `_foreach_copy_` batching here (v6) — measured
        # WORSE iter time (+7 ms). Likely cause: building 4 dst/src
        # lists × 12 narrows + foreach_copy fixed overhead exceeds the
        # savings of fewer launches at N=12. Per-grp copy_ in a tight
        # loop is the empirical winner.
        if d2h_stream is not None and event_compute is not None:
            d2h_stream.wait_event(event_compute)
        with sm_d2h:
            for gi, grp in enumerate(self._groups):
                gpu = per_grp_gpu[gi]
                if gpu is None:
                    continue
                elem_idx_grp_i32, p_sel_grp, m_sel_grp, v_sel_grp = gpu
                grp_pinned_start = grp_pinned_off[gi]
                grp_total = grp_groups_size[gi] + grp_outliers_size[gi]
                pinned_idx.narrow(0, grp_pinned_start, grp_total).copy_(
                    elem_idx_grp_i32, non_blocking=True,
                )
                pinned_p.narrow(0, grp_pinned_start, grp_total).copy_(
                    p_sel_grp, non_blocking=True,
                )
                pinned_m.narrow(0, grp_pinned_start, grp_total).copy_(
                    m_sel_grp, non_blocking=True,
                )
                pinned_v.narrow(0, grp_pinned_start, grp_total).copy_(
                    v_sel_grp, non_blocking=True,
                )
        # Cross-stream ownership transfer: Phase A allocated these on
        # tracker_stream, Phase B queued non_blocking copies onto
        # d2h_stream. We must NOT free the GPU staging until the H2D
        # provably completes — otherwise the caching allocator may
        # reuse the memory under tracker_stream while d2h_stream's
        # copy is still in flight (UB: writes garbage to pinned host).
        # Solution: hand ownership to the writer thread. Writer
        # synchronises on ``event_h2d`` (recorded on d2h_stream below)
        # then clears the list — at that point the H2D is provably
        # done. This is the explicit ownership counterpart of the
        # earlier ``record_stream`` patch.
        gpu_refs_for_writer: List[torch.Tensor] = [
            t for gpu in per_grp_gpu if gpu is not None for t in gpu
        ]
        per_grp_gpu = []  # type: ignore[assignment]

        # Build per-name selected_payload as a list of sub-entries.
        # For each name in BF16 group g: groups sub-entry (if cnt_g_i > 0)
        # then outliers sub-entry (if cnt_o_i > 0).
        for gi, grp in enumerate(self._groups):
            sn = grp["start_name_in_grp"]
            en = grp["end_name_in_grp"]
            grp_pinned_start = grp_pinned_off[gi]
            grp_g_seg_start = grp_pinned_start
            grp_o_seg_start = grp_pinned_start + grp_groups_size[gi]
            # Cumulative offsets within this BF16 group's groups / outliers
            # regions — populated as we walk names.
            cum_g = 0
            cum_o = 0
            for i in range(sn, en):
                name = name_list[i]
                cg = name_cnt_g[i]
                co = name_cnt_o[i]
                sub_entries: List[Tuple[torch.Tensor, torch.Tensor,
                                         torch.Tensor, torch.Tensor]] = []
                if cg > 0 and pinned_idx is not None:
                    abs_g = grp_g_seg_start + cum_g
                    sub_entries.append((
                        pinned_idx.narrow(0, abs_g, cg),
                        pinned_p.narrow(0, abs_g, cg),
                        pinned_m.narrow(0, abs_g, cg),
                        pinned_v.narrow(0, abs_g, cg),
                    ))
                if co > 0 and pinned_idx is not None:
                    abs_o = grp_o_seg_start + cum_o
                    sub_entries.append((
                        pinned_idx.narrow(0, abs_o, co),
                        pinned_p.narrow(0, abs_o, co),
                        pinned_m.narrow(0, abs_o, co),
                        pinned_v.narrow(0, abs_o, co),
                    ))
                # NOTE: pinned tensors store IDX in the BF16-group's
                # m_flat coordinates (we subtracted master_off_g_elem
                # in Phase A). The writer's lossy_fc_append_batch
                # subtracts the per-tensor ``offset`` (from the entry's
                # ``ent.global_offsets[tname]``) before writing; that
                # matches local-in-tensor convention. We need to
                # convert from BF16-group-flat → tensor-local. Those
                # two coord systems differ by the tensor's offset
                # within the BF16 group's m_flat (= offsets_cpu[i] -
                # master_off_g, in groups, × g_blk). The writer's
                # ``ent.global_offsets[tname]`` is the offset within
                # the EXPERT's concatenated address space (defined by
                # _ExpertEntry), unrelated to BF16-group offsets.
                # → the writer applies a +offset bias; the IDX in
                # pinned should be in TENSOR-LOCAL coords (same as
                # before). To keep that contract, we store:
                #   pinned IDX = elem_idx_in_tensor (= elem_idx in
                #   BF16-group coords - tensor's within-group offset).
                # Easiest fix: store BF16-group-flat coords here and
                # have writer's filter subtract (offsets_cpu[i] -
                # master_off_g) × g_blk (a precomputed bias).
                selected_payload[name] = sub_entries
                per_param_selected[name] = cg + co
                cum_g += cg
                cum_o += co

        if d2h_stream is not None and total_pinned > 0:
            event_h2d = torch.cuda.Event()
            event_h2d.record(d2h_stream)
        _t_after_loop = _time_dsp.perf_counter() if _dsp_dbg else 0.0
        _t_after_streamsync = _t_after_loop

        _t_lossy_skip = _os_dsp.environ.get(
            "PROPOSAL_OUTLIER_SKIP_LOSSY", ""
        ) in ("1",) if _dsp_dbg else False
        _t_after_batchbuild = _t_after_streamsync
        if self._expert_state is not None and not _t_lossy_skip:
            per_expert_batch: Dict[Tuple[int, int],
                                   List[Tuple[str, torch.Tensor,
                                              torch.Tensor, torch.Tensor,
                                              torch.Tensor]]] = {}
            # selected_payload[name] is a list of sub-entries (groups
            # part + outliers part). Flatten into the per-expert batch
            # — writer treats each sub-entry as a separate scatter /
            # log append for the same tensor (last-write-wins on
            # duplicate idx is naturally handled).
            for name, sub_entries in selected_payload.items():
                eid = self._tname_to_eid.get(name)
                if eid is None or eid in excluded_eids:
                    continue
                if not sub_entries:
                    continue
                bucket = per_expert_batch.setdefault(eid, [])
                for (idx_cpu, p_cpu, m_cpu, v_cpu) in sub_entries:
                    bucket.append((name, idx_cpu, p_cpu, m_cpu, v_cpu))
            _t_after_batchbuild = _time_dsp.perf_counter() if _dsp_dbg else 0.0
            # Single writer submission instead of one per expert — saves
            # ~100 ms training-thread time at ~80 experts × 1-2 ms per
            # submit_raw call (queue lock + counter increment).
            # Writer thread waits on event_h2d (records after the d2h
            # copies complete) before reading pinned host data, so the
            # training thread NEVER blocks on the H2D — main_thread
            # returns from dispatch as soon as the copies are queued.
            self._expert_state.lossy_fc_append_batch(
                step=int(prev_step),
                per_expert_updates=per_expert_batch,
                h2d_event=event_h2d,
                gpu_refs=gpu_refs_for_writer,
            )
            # Per-expert dispatch coverage diagnostic — runs at every
            # ``log_interval`` boundary to confirm routing is balanced.
            # ``n_experts_dispatched`` is the # of distinct experts that
            # received at least one entry this iter. A healthy run shows
            # this near ``num_expert_slots``; if it stays at 1-2 the
            # routing is concentrated (the bug we previously diagnosed
            # and fixed). Total entries equals K_total under steady
            # state. Cheap: O(n_experts) sums + one print.
            if prev_step > 0 and prev_step % 10 == 0:
                _n_experts = len(per_expert_batch)
                _total_entries = 0
                _max_entries = 0
                _min_entries = float("inf") if _n_experts > 0 else 0
                for _eid, _bucket in per_expert_batch.items():
                    # Each bucket entry is a list of (name, idx, p, m, v)
                    # tuples. The idx tensor's numel is the count for
                    # that sub-entry. Summing across sub-entries gives
                    # per-expert total.
                    _per_expert = 0
                    for (_n, _idx_cpu, _p, _m, _v) in _bucket:
                        _per_expert += int(_idx_cpu.numel())
                    _total_entries += _per_expert
                    if _per_expert > _max_entries:
                        _max_entries = _per_expert
                    if _per_expert < _min_entries:
                        _min_entries = _per_expert
                if _n_experts == 0:
                    _min_entries = 0
                _total_slots = (
                    len(getattr(self, "_tname_to_eid", {}))
                    or 0
                )
                # Total expert slots = unique eids in tname_to_eid (one
                # eid maps to multiple tensors so we need unique values).
                _unique_eids = len(set(self._tname_to_eid.values())) if self._tname_to_eid else 0
                print(
                    f"[proposal] dispatch_coverage @prev_iter{prev_step}: "
                    f"n_experts_dispatched={_n_experts}/{_unique_eids} "
                    f"total_entries={_total_entries} "
                    f"per_expert[min={_min_entries if _n_experts else 0} "
                    f"max={_max_entries}]",
                    flush=True,
                )

        self._pending_dispatch = None
        self._last_dispatch_per_param_selected = per_param_selected
        if _dsp_dbg and prev_step % 10 == 0:
            _t_dsp_end = _time_dsp.perf_counter()
            _ms = lambda a, b: (b - a) * 1000.0
            _peak_str = ""
            if _dsp_peak and torch.cuda.is_available():
                # Synchronize all streams so the peak captures
                # tracker_stream + d2h_stream's transient allocations.
                torch.cuda.synchronize()
                _dsp_peak_mb = (
                    torch.cuda.max_memory_allocated() - _dsp_alloc_at_entry
                ) / 1e6
                _peak_str = f" dsp_peak_delta=+{_dsp_peak_mb:.1f}MB"
            print(
                f"[proposal] dispatch_dbg @prev_iter{prev_step}: "
                f"total={_ms(_t_dsp_entry, _t_dsp_end):.1f}ms "
                f"evsync={_ms(_t_dsp_entry, _t_after_evsync):.1f} "
                f"+walk_segcpu={_ms(_t_after_evsync, _t_after_segcpu):.1f} "
                f"+olsort={_ms(_t_after_segcpu, _t_after_olsort):.1f} "
                f"+loop={_ms(_t_after_olsort, _t_after_loop):.1f} "
                f"+streamsync={_ms(_t_after_loop, _t_after_streamsync):.1f} "
                f"+batchbuild={_ms(_t_after_streamsync, _t_after_batchbuild):.1f} "
                f"+lossy_submit={_ms(_t_after_batchbuild, _t_dsp_end):.1f} "
                f"phase4={'OFF' if _phase4_skip else 'ON'} "
                f"lossy={'OFF' if _t_lossy_skip else 'ON'}"
                f"{_peak_str}",
                flush=True,
            )

    @torch.no_grad()
    def step(
        self,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        lossless_fc_eids: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[
        TrackerStepStats,
        Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        """Compute score = H_g × δ_g², pick group-level Top-K, dispatch
        the previous iter's selected payload to ExpertState.

        Args:
            optimizer: Adam (or variant). m / v are read for dispatch.
            global_step: 1-indexed iteration number.
            lossless_fc_eids: experts undergoing lossless FC THIS iter;
                their tensors are excluded from top-K scoring (score=0)
                AND from the lossy-FC append (no stale lossy on top of
                fresh LFCL). See _dispatch_pending_payload docstring.

        1-iter pipeline + tracker stream: the current iter LAUNCHES all
        GPU work on tracker_stream (no host-blocking sync here); the
        previous iter's pending result is dispatched at the start of
        THIS call (the only host-blocking sync, but short because the
        prev tracker GPU work has overlapped with this iter's
        forward+backward+optimizer in main_stream).
        """
        import time as _time
        _t0_entry = _time.perf_counter()
        excluded_eids = (
            set(lossless_fc_eids) if lossless_fc_eids else set()
        )
        device = next(self.model.parameters()).device
        g_blk = max(1, int(self.block_quant_g))
        mapper = get_mapper()
        # Bench env flags for empirical decomposition of score_step.
        import os as _os_bench_ss
        _bench_no_dispatch = _os_bench_ss.environ.get(
            "PROPOSAL_BENCH_NO_DISPATCH", "") in ("1", "true", "TRUE")
        _bench_no_tracker_kernel = _os_bench_ss.environ.get(
            "PROPOSAL_BENCH_NO_TRACKER_KERNEL", "") in ("1", "true", "TRUE")
        _bench_no_pass2 = _os_bench_ss.environ.get(
            "PROPOSAL_BENCH_NO_PASS2", "") in ("1", "true", "TRUE")

        # === STEP 0: dispatch previous iter's pending payload ===
        # Pass THIS iter's excluded set so dispatch can skip experts that
        # fired a lossless FC at T (their fresh LFCL would be clobbered
        # otherwise — see _dispatch_pending_payload docstring).
        _t0_dispatch_start = _time.perf_counter()
        if not _bench_no_dispatch:
            self._dispatch_pending_payload(
                optimizer, current_iter_excluded=excluded_eids,
            )
        _t0_dispatch_end = _time.perf_counter()
        phase_dispatch_ms = (_t0_dispatch_end - _t0_dispatch_start) * 1000.0

        # === STEP 1: launch current iter's GPU work on tracker_stream ===
        if self._tracker_stream is None and torch.cuda.is_available():
            self._tracker_stream = torch.cuda.Stream()
        stream = self._tracker_stream
        prev_stream = None
        if stream is not None:
            # Option 2: narrow the dependency. Instead of
            # wait_stream(main) (which waits for ALL prior main-stream
            # ops), record an event right at this point in main_stream
            # and wait only on it. This is equivalent in time (FIFO)
            # but is the more idiomatic stream-event pattern, and lets
            # main_stream issue forward(T+1) immediately without being
            # entangled in tracker_stream's prior wait_stream chain.
            cur_main = torch.cuda.current_stream()
            ev_opt = torch.cuda.Event()
            ev_opt.record(cur_main)
            stream.wait_event(ev_opt)
            # Switch default stream → tracker_stream so all GPU launches
            # below land in tracker_stream without an extra `with` block.
            prev_stream = cur_main
            torch.cuda.set_stream(stream)
        # Mark we're inside tracker_stream so update_delta_accum's DEBUG
        # sync (which would block main stream) is suppressed.
        self._inside_tracker_stream = True

        # update_delta_accum (m, v read + dg write) — same stream as score.
        if not _bench_no_tracker_kernel:
            self.update_delta_accum(optimizer, global_step)

        # B2 expert-lossless-FC compatibility: zero out the dg accumulator
        # for any expert that fired a lossless FC this iter. CLAUDE.md spec:
        # "Expert lossless FC 수행 시 ... θ_CP = θ_current → δ = 0".
        # ExpertLosslessFC.fire_if_due ALSO does this reset (race-free
        # there since same tracker_stream). This block is a defensive
        # backup for callers that pass lossless_fc_eids without going
        # through ExpertLosslessFC. Idempotent (zero-of-zero is fine).
        if excluded_eids and getattr(self, "_flat_layout_done", False):
            for name in self._g_fast_name_list:
                eid = self._tname_to_eid.get(name)
                if eid is not None and eid in excluded_eids:
                    d_g = self._delta_g_accum.get(name)
                    if d_g is not None and d_g.numel() > 0:
                        d_g.zero_()

        _t0_p1 = _time.perf_counter()

        # --- Phase 1: group-level score = h_g × dg in ONE torch.mul ------
        # Layout (set at init): flat_sg, dg_flat, h_g_bf16_flat, exc_mask
        # all share the same master-order layout (per-name views laid out
        # contiguously per BF16-Opt group). h_g_bf16_flat is rewritten in
        # place by ``_fold_h`` every Hessian refresh (~40 iters); excluded
        # mask is touched only for the (usually 0-few) experts undergoing
        # lossless FC this iter.
        if not getattr(self, "_flat_layout_done", False):
            # init not done yet — return early stats (defensive).
            self._inside_tracker_stream = False
            if prev_stream is not None:
                torch.cuda.set_stream(prev_stream)
            stats = TrackerStepStats(
                step=global_step, total_tracked=0, k=0, topk_ratio=0.0,
                max_A=0.0, max_delta=0.0, score_mean=0.0, score_max=0.0,
                selected_bytes_sparse=0,
                full_save_bytes=self.full_save_bytes,
                full_save_numel=self.total_full_save, compute_time_ms=0.0,
            )
            self.last_phase_ms = {
                "setup": 0.0, "pass1": 0.0, "pass2": 0.0,
                "pass3": 0.0, "post": 0.0,
            }
            return stats, {}

        # offsets_t / arange_g lazy-init (once).
        if not hasattr(self, "_g_fast_offsets_t"):
            self._g_fast_offsets_t = torch.tensor(
                self._g_fast_offsets_cpu, dtype=torch.long, device=device,
            )
            self._g_fast_arange_g = torch.arange(
                g_blk, dtype=torch.long, device=device,
            )

        name_list = self._g_fast_name_list
        offsets_cpu = self._g_fast_offsets_cpu
        offsets_t = self._g_fast_offsets_t
        arange_g = self._g_fast_arange_g
        dg_flat = self._g_fast_dg_flat
        h_g_flat = self._g_fast_h_g_bf16_flat
        exc_mask = self._g_fast_exc_mask
        chunk_buf = self._g_fast_chunk_buf
        chunk_size = self._g_fast_chunk_size
        global_hist = self._g_fast_global_hist
        n_bins = self._g_fast_n_bins
        total_groups = dg_flat.numel()
        total_elems = total_groups * g_blk

        # B2 simplification: _fold_h now writes h_g IN-PLACE directly into
        # _g_fast_h_g_bf16_flat via per-name views, and _H_g dict
        # entries point to the same views (storage shared). Hence no
        # rebuild copy needed here — h_g_flat always reflects the latest
        # _fold_h result. Saves a redundant ~47 MB BF16 copy per refresh
        # AND eliminates the _H_g dict's separate-storage allocation.

        # Build exclusion mask: True normally, False for experts in lossless FC.
        prev_exc = getattr(self, "_g_fast_prev_exc_eids", None)
        if prev_exc != excluded_eids:
            exc_mask.fill_(True)
            if excluded_eids:
                for i, n in enumerate(name_list):
                    eid = self._tname_to_eid.get(n)
                    if eid is not None and eid in excluded_eids:
                        self._g_fast_exc_views[i].fill_(False)
            self._g_fast_prev_exc_eids = set(excluded_eids)

        _t0_p2 = _time.perf_counter()
        phase_pass1_ms = (_t0_p2 - _t0_p1) * 1000.0

        # --- Phase 2 (X-C: unified chunked loop) ------------------------
        # Goal: process Phase 1 (score = h_g × dg × exc) + Phase 2 (histc
        # accumulation, cumsum/scatter packed selection) in a SINGLE chunk
        # loop with fixed chunk_buf transient memory (~100 MB). This makes
        # the path memory-stable across all g (1..256) — no flat_sg buffer
        # scaling with total_groups.
        #
        # Threshold: 1-iter delayed. Current iter's selection uses previous
        # iter's threshold; current iter's chunk-summed hist updates the
        # threshold for the next iter. Cold start: threshold = 0 → all
        # groups pass first iter → sel_buf[0..K_max-1] holds first
        # K_max indices (whatever they are). Top-K accuracy converges
        # within 2-3 iters as the threshold catches up.
        K_total = max(1, int(self.topk_ratio * total_elems))
        K_groups = max(1, K_total // g_blk)
        K_groups = min(K_groups, total_groups)
        avg_per_bin = max(1, total_groups // n_bins)
        K_max = K_groups + 5 * avg_per_bin

        # Persistent sel buffer (lazy-init / re-alloc on size change).
        sel_buf = getattr(self, "_g_fast_sel_buf", None)
        if sel_buf is None or sel_buf.numel() < K_max + 1:
            self._g_fast_sel_buf = torch.empty(
                K_max + 1, dtype=torch.long, device=device,
            )
            sel_buf = self._g_fast_sel_buf
        sel_buf.fill_(-1)
        global_hist.zero_()

        threshold_t = self._g_fast_prev_threshold_t  # GPU scalar

        # Delayed s_max bound for histc range (host-side carry).
        prev_smax_g = getattr(self, "_prev_s_max_g", None)
        if prev_smax_g is not None and prev_smax_g > 0.0:
            s_max = prev_smax_g * 1.5
        else:
            # Cold start: probe one chunk to seed s_max. Single sync.
            cold = chunk_buf[: min(chunk_size, total_groups)]
            # h_g (BF16) × dg (FP32) → FP32 in cold via .to() promotion.
            torch.mul(
                h_g_flat[: cold.numel()].to(torch.float32),
                dg_flat[: cold.numel()],
                out=cold,
            )
            cold.mul_(exc_mask[: cold.numel()].to(torch.float32))
            s_max = float(cold.max().item())
            if s_max <= 0.0:
                s_max = 1e-6
        bin_width = s_max / n_bins

        # Note: per-chunk arange / full_kmax are transient (PyTorch caching
        # allocator reuses across chunks + iters → no measurable benefit
        # from making them persistent, only memory waste).

        # GPU-side timing of the chunk loop + sort/searchsorted (cuda
        # events). This isolates the actual GPU work from the host-side
        # blocking sync at seg_bounds.cpu(), which absorbs prior-stream
        # wait. Set PROPOSAL_DEBUG_TIME_ACCUM=1 to print.
        import os as _os_pe
        _dbg_pe = _os_pe.environ.get("PROPOSAL_DEBUG_TIME_ACCUM", "") in (
            "1", "true", "TRUE",
        )
        _ev0 = _ev1 = None
        if _dbg_pe and torch.cuda.is_available():
            _ev0 = torch.cuda.Event(enable_timing=True)
            _ev1 = torch.cuda.Event(enable_timing=True)
            _ev0.record()

        # Running True-count cumulative (GPU scalar).
        true_cum = torch.zeros((), dtype=torch.long, device=device)

        # Bench: skip pass2 chunk loop entirely.
        import os as _os_p2
        _bench_no_pass2_local = _os_p2.environ.get(
            "PROPOSAL_BENCH_NO_PASS2", "") in ("1", "true", "TRUE")
        _chunk_iter = (
            iter(())  # empty iterator: skip whole loop
            if _bench_no_pass2_local
            else range(0, total_groups, chunk_size)
        )
        for chunk_start in _chunk_iter:
            chunk_end = min(chunk_start + chunk_size, total_groups)
            chunk_n = chunk_end - chunk_start
            chunk_view = chunk_buf[:chunk_n]
            # Score: h_g (BF16) × dg (FP32) → FP32 in chunk_view.
            torch.mul(
                h_g_flat.narrow(0, chunk_start, chunk_n).to(torch.float32),
                dg_flat.narrow(0, chunk_start, chunk_n),
                out=chunk_view,
            )
            # Apply exclusion mask (bool → multiply zeros out excluded).
            exc_chunk = exc_mask.narrow(0, chunk_start, chunk_n)
            chunk_view.mul_(exc_chunk)  # bool * float OK in PyTorch
            # Histogram accumulate (for NEXT iter's threshold).
            global_hist.add_(
                torch.histc(chunk_view, bins=n_bins, min=0.0, max=s_max)
            )
            # Mask + cumsum + scatter using THIS iter's (prev) threshold.
            # Use cumsum(dtype=long) to skip an explicit .long() cast +
            # in-place sub_/add_ to avoid transient new tensors.
            mask_chunk = chunk_view >= threshold_t
            positions_chunk = mask_chunk.cumsum(0, dtype=torch.long)
            positions_chunk.sub_(1).add_(true_cum)
            full_kmax = torch.full_like(positions_chunk, K_max)
            target_pos = torch.where(
                mask_chunk, positions_chunk, full_kmax,
            )
            target_pos.clamp_(min=0, max=K_max)
            chunk_idx = torch.arange(
                chunk_start, chunk_end, device=device, dtype=torch.long,
            )
            sel_buf.scatter_(0, target_pos, chunk_idx)
            # mask.sum(dtype=long) skips the explicit .long() cast.
            true_cum = true_cum + mask_chunk.sum(dtype=torch.long)

        # NOTE (unified_topk): the per-pool threshold computation is
        # GONE from pass2. The unified two-pool top-K walk in dispatch
        # (next iter) reads this iter's ``global_hist`` AND
        # ``_outlier_elem_hist`` and produces a UNIFIED elem_th +
        # group_th (= elem_th × g_blk) under a single K-budget.
        # ``self._prev_s_max_g`` is still carried (used by the unified
        # walk to pick group score range), but the merge walk now
        # updates it from the histogram itself — so the cold-start
        # ``s_max`` probe above is the only place the host-side carry
        # is set, and only on iter 1 (when the merge walk's
        # prev_s_max_g is None).
        if getattr(self, "_prev_s_max_g", None) is None:
            self._prev_s_max_g = float(s_max)

        # Outlier-promotion: histogram walk MOVED to dispatch (after
        # ev.synchronize()) — there the two host syncs (hist_total,
        # observed_max) are free because GPU work has already finished.
        # Keeping them here would block tracker.step from returning
        # for ~80-120 ms (pass2 atomic_max latency), defeating the
        # 1-iter pipelining and idling main_stream.

        # Replace -1 padding with total_groups so sort puts them at end.
        n_groups_full = torch.full(
            (1,), total_groups, dtype=torch.long, device=device,
        )
        valid_part = sel_buf[:K_max]
        valid_part = torch.where(
            valid_part >= 0, valid_part, n_groups_full.expand_as(valid_part),
        )
        top_idx_sorted, _ = valid_part.sort()
        seg_bounds = torch.searchsorted(top_idx_sorted, offsets_t)
        if _ev1 is not None:
            _ev1.record()
        # NOTE: NO seg_bounds.cpu() here — defer to next iter dispatch.

        _t0_p2_end = _time.perf_counter()
        phase_pass2_ms = (_t0_p2_end - _t0_p2) * 1000.0

        # Restore main default stream (we set tracker_stream as default
        # at STEP 1 entry).
        self._inside_tracker_stream = False
        if prev_stream is not None:
            torch.cuda.set_stream(prev_stream)

        # Outlier-promotion: stash selected element indices/count via
        # GPU clone on tracker_stream — NO host sync here. The next
        # iter's update_delta_accum will overwrite the live buffers,
        # so we snapshot now. Clones run on tracker_stream so the
        # event recorded below covers them; dispatch's ev.synchronize()
        # then makes the .item() call essentially free.
        sel_idx_snap = None
        sel_count_snap = None
        if (self._outlier_promote
                and self._outlier_sel_idx is not None
                and self._outlier_sel_count is not None):
            from contextlib import nullcontext as _nullctx_stash
            sm_stash = (
                torch.cuda.stream(stream) if stream is not None
                else _nullctx_stash()
            )
            with sm_stash:
                sel_idx_snap = self._outlier_sel_idx.clone()
                sel_count_snap = self._outlier_sel_count.clone()

        # Record tracker work completion event AFTER stash clones so
        # ev.synchronize() at next iter dispatch covers them too.
        event_tracker = None
        if stream is not None:
            event_tracker = torch.cuda.Event()
            event_tracker.record(stream)
        self._pending_tracker_event = event_tracker
        self._pending_dispatch = {
            "step": global_step,
            "seg_bounds": seg_bounds,
            "top_idx_sorted": top_idx_sorted,
            "name_list": list(name_list),
            "offsets_cpu": list(offsets_cpu),
            "excluded_eids": set(excluded_eids),
            "outlier_sel_idx_snap": sel_idx_snap,
            "outlier_sel_count_snap": sel_count_snap,
        }

        _t0_end = _time.perf_counter()
        phase_launch_ms = (_t0_end - _t0_dispatch_end) * 1000.0
        phase_setup_ms = (_t0_p1 - _t0_dispatch_end) * 1000.0
        self.last_phase_ms = {
            "setup": phase_setup_ms,
            "pass1": phase_pass1_ms,
            "pass2": phase_pass2_ms,
            "pass3": 0.0,  # deferred to next iter dispatch
            "post": 0.0,
            "dispatch": phase_dispatch_ms,
            "launch": phase_launch_ms,
        }

        # GPU event timing (debug only)
        if _ev0 is not None and _ev1 is not None:
            try:
                _ev1.synchronize()
                gpu_ms = _ev0.elapsed_time(_ev1)
                print(f"[proposal] tracker.step.p2 GPU work "
                      f"@iter{global_step}: gpu={gpu_ms:.2f} ms host="
                      f"{phase_pass2_ms:.2f} ms (launch only)",
                      flush=True)
            except Exception:
                pass

        # Diagnostics: zeros (selected_payload is dispatched next iter).
        per_param_score_sum = {n: 0.0 for n in self._tracked_names}
        per_param_selected = getattr(
            self, "_last_dispatch_per_param_selected", {},
        )
        for n in self._tracked_names:
            per_param_selected.setdefault(n, 0)

        stats = TrackerStepStats(
            step=global_step,
            total_tracked=total_elems,
            k=sum(per_param_selected.values()),
            topk_ratio=0.0,
            max_A=0.0,
            max_delta=0.0,
            score_mean=0.0,
            score_max=0.0,
            selected_bytes_sparse=0,
            full_save_bytes=self.full_save_bytes,
            full_save_numel=self.total_full_save,
            compute_time_ms=0.0,
            per_param_selected=per_param_selected,
            per_param_score_sum=per_param_score_sum,
        )
        return stats, {}
