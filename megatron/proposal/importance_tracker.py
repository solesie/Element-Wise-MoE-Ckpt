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
    def _vmap_kernel(
        v_ptr,                       # v_t FP32 element-level
        h_g_ptr,                     # BF16 group-level output
        v_q_ptr, h_q_ptr,            # FP32 lookup tables [N_Q]
        n_groups,
        master_off_group,
        eps,
        G: tl.constexpr,
        GROUPS_PER_BLOCK: tl.constexpr,
        N_Q: tl.constexpr,
        LOG_N_Q: tl.constexpr,
    ):
        """v_t element → H quantile lookup → group-mean reduce → BF16 store.

        Single fused launch per BF16-Opt bucket: each program handles
        G * GROUPS_PER_BLOCK contiguous v_t elements, runs an unrolled
        log2(N_Q)-step binary search against ``v_q``, lerps ``h_q``, then
        reduces groups (sum / G) and stores BF16 into ``h_g_ptr`` at
        ``master_off_group + gid_start``.

        Mirrors ``_accum_with_outlier_kernel`` layout: same elem_off /
        gid_off math, same group-mean reduction via tl.reshape + tl.sum.
        v_q / h_q are 16 KB each → resident in L2 across all programs.
        """
        BLOCK_SIZE: tl.constexpr = G * GROUPS_PER_BLOCK
        pid = tl.program_id(axis=0)
        gid_start = pid * GROUPS_PER_BLOCK
        elem_off = gid_start * G + tl.arange(0, BLOCK_SIZE)
        elem_mask = elem_off < n_groups * G
        v = tl.load(v_ptr + elem_off, mask=elem_mask, other=0.0).to(tl.float32)

        # Unrolled binary search (lower_bound) in v_q[0..N_Q-1].
        # After LOG_N_Q steps, ``bin`` is the largest index with v_q[bin] <= v.
        bin_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        for i in tl.static_range(LOG_N_Q):
            step = N_Q >> (i + 1)
            mid = bin_idx + step
            mid_clamped = tl.minimum(mid, N_Q - 1)
            q_mid = tl.load(v_q_ptr + mid_clamped)
            bin_idx = tl.where(v >= q_mid, mid, bin_idx)
        bin_idx = tl.maximum(tl.minimum(bin_idx, N_Q - 2), 0)

        v_lo = tl.load(v_q_ptr + bin_idx)
        v_hi = tl.load(v_q_ptr + bin_idx + 1)
        h_lo = tl.load(h_q_ptr + bin_idx)
        h_hi = tl.load(h_q_ptr + bin_idx + 1)
        alpha = (v - v_lo) / (v_hi - v_lo + eps)
        mapped = h_lo + alpha * (h_hi - h_lo)

        mapped_2d = tl.reshape(mapped, (GROUPS_PER_BLOCK, G))
        grp_avg = tl.sum(mapped_2d, axis=1) / G

        gid_local = gid_start + tl.arange(0, GROUPS_PER_BLOCK)
        gid_mask = gid_local < n_groups
        tl.store(
            h_g_ptr + master_off_group + gid_local,
            grp_avg.to(tl.bfloat16),
            mask=gid_mask,
        )

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
        # NOTE: dtype = int64. Was int32 — silently overflowed for models
        # with > 2^31 expert elements (e.g. 12 layers × 48 experts every
        # layer = 2.72 B elements). The Triton kernel stores
        # ``elem_off + master_off_elem`` (BF16-group-local elem_off plus
        # global master_off_elem) into this buffer; for late BF16 groups
        # ``master_off_elem`` alone is > INT32_MAX, and the int32 store
        # wraps to negative → index_select OOB downstream → CUDA assert.
        # int64 is the correctness fix; +4 B per slot is negligible.
        # sel_idx / sel_count FIFO removed — Pass A writes
        # per-name atomic counts directly; no K_max persistent slab.
        # Per-iter pending: list of selected element idx (CPU long).
        # 1-iter pipelined like group selection — dispatched at next
        # tracker.step's _dispatch_pending_payload.
        self._outlier_pending_idx_cpu: Optional[torch.Tensor] = None

        # Dispatch metadata pool : pinned-host + dev pair for the
        # per-iter `torch.tensor(name_cnt_*, n_per_grp_*_list)` calls in
        # ``_dispatch_pending_payload``. Each ``torch.tensor(host_list,
        # device=cuda)`` implicitly cudaStreamSynchronizes — replacing the
        # 3 per-iter calls with a single non_blocking H2D copy removes
        # ~3 sync points per iter on tracker_stream.
        #
        # Single buffer (not ping-pong) is OK here because dispatch is
        # called once per iter — long inter-call gap means the prev iter's
        # async H2D has long completed before next acquire(). Event sync
        # at begin() guards against pathological CPU speed-up.
        self._dispatch_meta_host: Optional[torch.Tensor] = None
        self._dispatch_meta_dev: Optional[torch.Tensor] = None
        self._dispatch_meta_event: Optional["torch.cuda.Event"] = None

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
        # Persistent pinned dispatch buffers (fixed base addresses for
        # C++ IO-passthru — every iter narrows into the same buffer at
        # varying offsets). Lazy-allocated on first dispatch sized to
        # ``topk_ratio × total_elems × 3``; never re-grown.
        # Phase 3: legacy SoA pinned dispatch buffers removed — see
        # comment in _init_accum_sq_flat. GPU AOS handles the SoA → AoS
        # interleave on device into segment_pipeline staging.
        # Pending state from previous iter's tracker.step. Carried here
        # because cp_hooks dispatches it on the next iter's tracker entry.
        self._pending_tracker_event: Optional["torch.cuda.Event"] = None
        self._pending_dispatch: Optional[Dict] = None

        # Phase 2c (GPU-side scatter / AOS): the non-chunked dispatch path
        # populates these alongside the legacy ``selected_payload`` /
        # ``pinned_*`` arrays. ``_gpu_payload_per_group`` keeps the
        # per-BF16-group SoA tensors alive past the dispatch return; the
        # consumer (gpu_aos kernel in the expert lossy worker) reads
        # directly from them, skipping the legacy D2H to pinned.
        # ``_gpu_payload_entries`` maps each tracked name to a list of
        # ``(bf16_group_idx, src_off_in_group, n_records)`` tuples — one
        # per sub-entry (groups pool, outliers pool). Both fields are
        # ``None`` when no dispatch has run or the chunked path is in
        # use (chunked path releases per-chunk GPU tensors before this
        # capture point; integration deferred to Phase 3 follow-up).
        self._gpu_payload_per_group: Optional[List[
            Optional[Tuple[torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]]
        ]] = None
        self._gpu_payload_entries: Optional[Dict[
            str, List[Tuple[int, int, int]]
        ]] = None

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

    def take_gpu_payload(self) -> Tuple[
        Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor]]]],
        Optional[Dict[str, List[Tuple[int, int, int]]]],
    ]:
        """Phase 2c handoff: return the per-BF16-group GPU SoA payload
        and per-name entry descriptors built by the most recent
        non-chunked dispatch, then clear the internal references.

        Returns ``(None, None)`` when no GPU payload is available
        (chunked dispatch path, or no dispatch has run since the last
        take). Caller must keep the returned tensor references alive
        until the gpu_aos kernel finishes — typically by holding them
        in the worker thread's task closure alongside the
        ``gpu_refs_for_writer`` list that already protects the same
        storages from the caching allocator.
        """
        per_group = self._gpu_payload_per_group
        entries = self._gpu_payload_entries
        self._gpu_payload_per_group = None
        self._gpu_payload_entries = None
        return per_group, entries

    @torch.no_grad()
    def _quantile_table_from_flat(
        self, flat: torch.Tensor, drop_nonpositive: bool,
    ) -> Optional[torch.Tensor]:
        """N_Q-point quantile lookup table from a flat BF16/FP32 tensor.

        Stride-samples ``min(numel, 1M)`` to bound transient at 4 MB
        (sample) + 4 MB (sort) ≈ 8 MB ≪ 80 MB. ``drop_nonpositive``
        starts the table at the positive minimum (H side semantics).
        """
        n = int(flat.numel())
        if n == 0:
            return None
        stride = max(1, n // 1_000_000)
        sample = flat[::stride].detach().to(torch.float32).abs()
        sample_sorted = sample.sort().values
        if drop_nonpositive:
            sample_sorted = sample_sorted[sample_sorted > 0]
        m = int(sample_sorted.numel())
        if m < self._N_Q:
            return None
        idx = (
            torch.arange(1, self._N_Q + 1, device=flat.device, dtype=torch.float32)
            * ((m - 1) / (self._N_Q + 1))
        ).long().clamp_(0, m - 1)
        return sample_sorted[idx]

    def _acquire_ring_event(self, role: str) -> "torch.cuda.Event":
        """acquire next ``cuda.Event`` from a lazily-allocated ring
        for the given role. Eliminates per-iter ``torch.cuda.Event()``
        allocations on the three hot-path roles (compute / ev_opt /
        tracker). Ring size = 4 (≥ pipelining depth 2 with safety
        margin); re-recording is permitted by PyTorch.
        """
        ring_attr = f"_event_ring_{role}"
        idx_attr = f"_event_ring_{role}_idx"
        ring = getattr(self, ring_attr, None)
        if ring is None:
            n = 2 if role == "evopt" else 4
            ring = [torch.cuda.Event() for _ in range(n)]
            setattr(self, ring_attr, ring)
        idx = getattr(self, idx_attr)
        ev = ring[idx]
        setattr(self, idx_attr, (idx + 1) % len(ring))
        return ev

    @torch.no_grad()
    def update_h_anchor_quantiles(self) -> None:
        """Anchor (cheap FD refresh) just wrote H_FD into
        ``_g_fast_h_g_bf16_flat``; extract the N_Q-point quantile lookup
        table now so subsequent ``update_v_mapping`` calls can map v_t
        onto this H distribution."""
        q = self._quantile_table_from_flat(
            self._g_fast_h_g_bf16_flat, drop_nonpositive=True,
        )
        if q is None:
            return
        self._g_fast_h_quantiles.copy_(q)
        self._h_anchor_initialised = True

    @torch.no_grad()
    def update_v_mapping(self) -> None:
        """v_t element → mapped(v_t) → group-mean → BF16 store into
        ``_g_fast_h_g_bf16_flat``. Single fused Triton launch per
        BF16-Opt bucket — same launch / SM occupancy pattern as the
        production ``_accum_with_outlier_kernel``.

        No-op until first anchor populates ``_g_fast_h_quantiles`` and
        when Triton is unavailable.

        Hard constraints: persistent VRAM addition 0 (writes in place);
        transient (v_t quantile sort sample) 8 MB ≪ 80 MB;
        main_stream blocking 0 (runs on tracker_stream via :meth:`step`).
        """
        if not self._h_anchor_initialised or not _HAS_TRITON:
            return
        g = max(1, int(self.block_quant_g))
        # 1-iter lag GPU timing — same pattern as phase_pass2.
        _vmap_ev0 = _vmap_ev1 = None
        if torch.cuda.is_available():
            _prev = getattr(self, "_vmap_ev_prev", None)
            if _prev is not None:
                try:
                    _prev[1].synchronize()
                    if not hasattr(self, "last_phase_ms") or self.last_phase_ms is None:
                        self.last_phase_ms = {}
                    self.last_phase_ms["v_mapping"] = float(_prev[0].elapsed_time(_prev[1]))
                except Exception:
                    pass
            _vmap_ev0 = torch.cuda.Event(enable_timing=True)
            _vmap_ev1 = torch.cuda.Event(enable_timing=True)
            _vmap_ev0.record()

        # Sub-phase timers (cuda.Event pairs, 1-iter lag) — splits the
        # ``v_mapping`` total into ``v_q_extract`` (sample-sort) and
        # ``v_q_kernel`` (Triton fused launch) so the dominant cost is
        # identifiable. Stamped onto last_phase_ms with the same lag
        # pattern as the parent v_mapping event.
        _qe_ev0 = _qk_ev0 = _qe_ev1 = _qk_ev1 = None
        if torch.cuda.is_available():
            _prev_sub = getattr(self, "_vmap_sub_ev_prev", None)
            if _prev_sub is not None:
                try:
                    qe0, qe1, qk0, qk1 = _prev_sub
                    qe1.synchronize(); qk1.synchronize()
                    if not hasattr(self, "last_phase_ms") or self.last_phase_ms is None:
                        self.last_phase_ms = {}
                    self.last_phase_ms["v_q_extract"] = float(qe0.elapsed_time(qe1))
                    self.last_phase_ms["v_q_kernel"] = float(qk0.elapsed_time(qk1))
                except Exception:
                    pass
            _qe_ev0 = torch.cuda.Event(enable_timing=True)
            _qe_ev1 = torch.cuda.Event(enable_timing=True)
            _qk_ev0 = torch.cuda.Event(enable_timing=True)
            _qk_ev1 = torch.cuda.Event(enable_timing=True)
            _qe_ev0.record()

        # v_t quantile table — refreshed every iter (v_t scale evolves
        # under Adam's β₂ EMA; matching the current iter's scale keeps
        # mapping fidelity).  all transients (cat/cast/sort/arange/
        # advanced index) replaced with fixed scratch slabs. The total
        # sample length is bounded by ~1M (per_bucket × num_buckets),
        # the slabs are allocated once on first call.
        v_flats: List[torch.Tensor] = []
        for group_dict in self._groups:
            v_flat = group_dict.get("v_flat")
            if v_flat is None:
                return
            v_flats.append(v_flat)
        per_bucket = max(1, 1_000_000 // max(1, len(v_flats)))
        # Compute per-v_flat strides + sample lengths (host arithmetic).
        strides = [max(1, int(v.numel()) // per_bucket) for v in v_flats]
        sample_lens = [
            (int(v.numel()) + s - 1) // s for v, s in zip(v_flats, strides)
        ]
        total_sample = sum(sample_lens)

        # Lazy-init / grow scratch slabs. Steady-state: hit the lazy path
        # only on iter 1. Growth path runs only if model param count
        # changes mid-training (unsupported in practice).
        if (self._v_sample_buf is None
                or int(self._v_sample_buf.numel()) < total_sample):
            cap = max(total_sample, 1_048_576)  # 1M float32 = 4 MB
            _dev = v_flats[0].device
            self._v_sample_buf = torch.empty(
                cap, dtype=torch.float32, device=_dev,
            )
            self._v_sorted_vals = torch.empty(
                cap, dtype=torch.float32, device=_dev,
            )
            self._v_sorted_idx = torch.empty(
                cap, dtype=torch.int64, device=_dev,
            )
        if self._v_qtile_arange_f is None:
            _dev = v_flats[0].device
            self._v_qtile_arange_f = torch.arange(
                1, self._N_Q + 1, dtype=torch.float32, device=_dev,
            )
            self._v_qtile_idx_i64 = torch.empty(
                self._N_Q, dtype=torch.int64, device=_dev,
            )

        # Stage strided samples into the contiguous scratch slab. ``copy_``
        # auto-casts BF16→FP32 inline; no intermediate ``.to(float32)``.
        sample_view = self._v_sample_buf.narrow(0, 0, total_sample)
        pos = 0
        for v, s, n in zip(v_flats, strides, sample_lens):
            src = v[::s]
            n_src = int(src.numel())
            if n_src != n:
                n = min(n_src, n)
            sample_view.narrow(0, pos, n).copy_(src.narrow(0, 0, n))
            pos += n
        # In case stride math yielded slightly under total_sample (rare),
        # trim the view to the actual filled range.
        if pos < total_sample:
            sample_view = self._v_sample_buf.narrow(0, 0, pos)
            total_sample = pos
        sorted_vals_view = self._v_sorted_vals.narrow(0, 0, total_sample)
        sorted_idx_view = self._v_sorted_idx.narrow(0, 0, total_sample)
        # torch.sort writes (values, indices) into out=. indices unused.
        torch.sort(sample_view, out=(sorted_vals_view, sorted_idx_view))

        m = total_sample
        if m >= self._N_Q:
            # Compute quantile indices in-place into _v_qtile_idx_i64.
            #   idx = (arange(1, N_Q+1) * ((m-1)/(N_Q+1))).long().clamp_(0, m-1)
            # Fused via mul (FP32) → to(int64, out=) → clamp_ in-place.
            scale = (m - 1) / (self._N_Q + 1)
            # Use _v_qtile_arange_f as source; output i64 to slab. PyTorch
            # 2.0+: torch.mul has no out= for cross-dtype, so do it in two
            # in-place steps on a small (N_Q=256) scratch — negligible.
            # Use the qtile_arange_f * scale and cast via _v_qtile_idx_i64.
            # Manual: write float then cast to int via a tmp view in-place.
            # _v_qtile_idx_i64 is i64 — torch.mul(out=) requires same dtype,
            # so cast scaled_f → i64 via direct copy_ (auto-truncates).
            scaled_f = self._v_qtile_arange_f.mul(scale)  # tiny (N_Q f32)
            self._v_qtile_idx_i64.copy_(scaled_f)         # truncating cast
            self._v_qtile_idx_i64.clamp_(0, m - 1)
            # Replace ``v_sorted[idx]`` advanced indexing with
            # ``index_select(out=)`` directly into _g_fast_v_quantiles
            # (size = N_Q, dtype = f32).
            torch.index_select(
                sorted_vals_view, 0, self._v_qtile_idx_i64,
                out=self._g_fast_v_quantiles,
            )
        if _qe_ev1 is not None:
            _qe_ev1.record()
            _qk_ev0.record()

        # Single fused Triton kernel launch per bucket. GROUPS_PER_BLOCK
        # bounds BLOCK_SIZE = G × GPB; production g=256 with GPB=256
        # produced BLOCK_SIZE=65536 → 17-30 s JIT compile + register spill.
        # GPB=32 keeps BLOCK_SIZE ≤ 8192 (g=256 → 8192, g=8 → 256), inside
        # Triton's effective sweet spot (≤ 16K) — JIT 3-5 s, grid 4-8× larger
        # but per-program register footprint drops below A100's 255 / thread.
        GROUPS_PER_BLOCK = 32
        LOG_N_Q = int(self._N_Q).bit_length() - 1   # log2(N_Q), N_Q is a power of 2
        for group_dict in self._groups:
            v_flat = group_dict["v_flat"]
            master_off = int(group_dict["master_off"])
            n = int(v_flat.numel())
            n_groups = n // g
            if n_groups == 0:
                continue
            grid = ((n_groups + GROUPS_PER_BLOCK - 1) // GROUPS_PER_BLOCK,)
            _vmap_kernel[grid](
                v_flat, self._g_fast_h_g_bf16_flat,
                self._g_fast_v_quantiles, self._g_fast_h_quantiles,
                n_groups, master_off, 1e-30,
                G=g, GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
                N_Q=self._N_Q, LOG_N_Q=LOG_N_Q,
            )

        if _qk_ev1 is not None:
            _qk_ev1.record()
            self._vmap_sub_ev_prev = (_qe_ev0, _qe_ev1, _qk_ev0, _qk_ev1)
        if _vmap_ev1 is not None:
            _vmap_ev1.record()
            self._vmap_ev_prev = (_vmap_ev0, _vmap_ev1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def gpu_memory_breakdown(self) -> Tuple[float, List[Tuple[str, float]]]:
        """Sum of all GPU tensors held by this tracker (MB) + per-item
        breakdown. Auto-enumerates ``self.__dict__`` so newly-added
        attributes are tracked without a whitelist sync. Detects missing
        accountings vs ``torch.cuda.memory_allocated()``.
        """
        items: List[Tuple[str, float]] = []
        for attr, t in vars(self).items():
            if isinstance(t, torch.Tensor) and t.is_cuda:
                items.append((attr, t.element_size() * t.numel() / 1e6))
        # _H_g entries are VIEWS into _g_fast_h_g_bf16_flat (no extra
        # storage). Track count only, not size (avoid double-counting
        # against torch.cuda.memory_allocated).
        for dict_attr in ("_H_g",):
            d = getattr(self, dict_attr, {}) or {}
            n = sum(
                1 for t in d.values()
                if isinstance(t, torch.Tensor) and t.is_cuda
            )
            if n > 0:
                items.append((f"{dict_attr}[count={n}]", 0.0))
        # Pending dispatch state — persists across iters (1-iter pipeline).
        # Dedup by underlying storage data_ptr so pending entries that
        # reference an already-tracked tensor (e.g. outlier_sel_idx_snap
        # after the clone→reference fix) are not double-counted against
        # torch.cuda.memory_allocated. Build a set of already-seen ids.
        seen_ptrs: set = set()
        for _attr, t in vars(self).items():
            if isinstance(t, torch.Tensor) and t.is_cuda:
                seen_ptrs.add(t.untyped_storage().data_ptr())
        pending = getattr(self, "_pending_dispatch", None) or {}
        pend_sz = 0.0
        pend_items: List[Tuple[str, float]] = []
        for key, val in pending.items():
            if isinstance(val, torch.Tensor) and val.is_cuda:
                ptr = val.untyped_storage().data_ptr()
                tag = "" if ptr not in seen_ptrs else " (alias)"
                sz = (
                    val.element_size() * val.numel() / 1e6
                    if ptr not in seen_ptrs
                    else 0.0
                )
                pend_sz += sz
                pend_items.append((f"pending.{key}{tag}", sz))
                seen_ptrs.add(ptr)
        if pend_sz > 0 or pend_items:
            for n, v in pend_items:
                items.append((n, v))
        # _v_mapping_v_flats: list of flat views (no own storage). Skip.
        # Per-group cached buffers inside self._groups
        groups = getattr(self, "_groups", None) or []
        grp_sz = 0.0
        for grp in groups:
            for key in ("m_flat", "v_flat", "p_flat", "dg_buf", "h_g_buf"):
                t = grp.get(key) if isinstance(grp, dict) else None
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    # These are VIEWS into other tracked buffers — don't
                    # double-count storage but record view size for visibility.
                    pass  # (view; already counted in parent tensor)
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
        # Init to +inf so cold-start (iter 0 / 1 before walk
        # has set a sane threshold) selects ZERO groups → no spurious
        # 1.36B-record dispatch. walk_thresholds overwrites this once
        # the first sample is collected.
        self._g_fast_prev_threshold_t = torch.full(
            (), float("inf"), dtype=torch.float32, device=device,
        )
        self._g_fast_h_built_at = -1  # rebuild marker

        # Build per-group state. Each group's dg_buf is a master-region view
        # into _g_fast_dg_flat.
        self._groups: List[Dict] = []
        self._delta_g_accum = {}
        # _H_g entries are views into the PREVIOUS _g_fast_h_g_bf16_flat
        # (before this refresh). Without clearing them here, those views
        # keep the OLD bf16_flat storage alive across the refresh — even
        # though we reallocate a fresh bf16_flat below. Result: each refresh
        # leaks one bf16_flat (~680 MB at 48ex). Clearing reverts views
        # to NEW storage as _fold_h re-registers them post-refresh.
        if hasattr(self, "_H_g") and self._H_g:
            self._H_g.clear()
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
            import os as _os_idiag
            if _os_idiag.environ.get("PROPOSAL_INIT_DIAG", "") in ("1",):
                _per_name_total = sum(sz // g for sz in sizes if sz // g > 0)
                _group_total_g = group_total // g
                _mflat_n = int(m_flat.numel()) if m_flat is not None else -1
                _pflat_n = int(p_flat.numel()) if p_flat is not None else -1
                print(f"[init_diag] g_idx={g_idx} master_off={master_off} "
                      f"group_total_elem={group_total} group_groups={_group_total_g} "
                      f"sum_per_name_groups={_per_name_total} "
                      f"m_flat.numel={_mflat_n} p_flat.numel={_pflat_n} "
                      f"start_name={start_name_in_grp} end_name={end_name_in_grp} "
                      f"sizes_uniq={sorted(set(sizes))[:5]}",
                      flush=True)
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

        # ---- v_t → H quantile-matching lookup tables ----
        # CDF matching: anchor refresh (cheap FD) populates
        # _g_fast_h_quantiles from the freshly written _g_fast_h_g_bf16_flat;
        # every iter ``update_v_mapping`` resamples v_t's quantile and
        # overwrites _g_fast_h_g_bf16_flat with mapped(v_t). Tables are
        # ``_N_Q``-point piecewise linear (256 ≈ 99.6% accuracy on the
        # lognormal H distribution, log2 step = 8 — keeps binary-search
        # latency well inside the per-element memory-load cycle).
        # Persistent VRAM = 2 × 256 × FP32 = 2 KB (≈ 0 vs the GB-scale
        # flat buffers above).
        self._N_Q = 256
        self._g_fast_v_quantiles = torch.zeros(
            self._N_Q, dtype=torch.float32, device=device,
        )
        self._g_fast_h_quantiles = torch.zeros(
            self._N_Q, dtype=torch.float32, device=device,
        )
        self._h_anchor_initialised = False

        #  update_v_mapping fixed scratch. Replaces the per-iter
        # torch.cat / .to(float32) / .sort() / torch.arange / advanced
        # indexing chain (~12 MB transient × per iter). Capacity is lazy
        # because num v_flats is decided after _init_accum_sq_flat() runs;
        # we allocate on first update_v_mapping() call.
        self._v_sample_buf: Optional[torch.Tensor] = None      # f32, CAP
        self._v_sorted_vals: Optional[torch.Tensor] = None     # f32, CAP
        self._v_sorted_idx: Optional[torch.Tensor] = None      # i64, CAP
        # Quantile-index helpers (N_Q-sized, ~2 KB each, init-once).
        self._v_qtile_arange_f: Optional[torch.Tensor] = None  # f32, N_Q
        self._v_qtile_idx_i64: Optional[torch.Tensor] = None   # i64, N_Q

        # Outlier-count kernel state (Pass A): per-name atomic counts +
        # element-offset bisect lookup. Slabs are tiny (N_names i64 each).
        self._outlier_name_count: Optional[torch.Tensor] = None    # i64, N_names
        self._outlier_elem_offsets: Optional[torch.Tensor] = None  # i64, N_names+1

        #  Event rings — avoid ``torch.cuda.Event()`` alloc per iter
        # for the three hot-path roles. Ring size = 4 (covers ≤ 4-iter
        # pipelining; production uses 1-iter so safe with 2× headroom).
        # Re-recording on an Event is fine; the worker thread sync'd on
        # iter T's Event has already returned by iter T+4. ev_opt is
        # local-to-step (no cross-thread), ring size 2 sufficient.
        self._event_ring_compute: Optional[List["torch.cuda.Event"]] = None
        self._event_ring_compute_idx: int = 0
        self._event_ring_evopt: Optional[List["torch.cuda.Event"]] = None
        self._event_ring_evopt_idx: int = 0
        self._event_ring_tracker: Optional[List["torch.cuda.Event"]] = None
        self._event_ring_tracker_idx: int = 0

        # Pre-allocate dispatch pinned buffers at tracker init with K_max
        # ceiling so per-iter dispatch NEVER calls cudaHostAlloc. Each
        # cudaHostAlloc of ~700MB blocks the main thread ~400ms (kernel
        # page-pinning rate ~1.5GB/s); without this pre-alloc, dispatch's
        # grow-only reuse cache was thrashing and re-allocating per iter,
        # adding ~1.15 sec/iter of main-thread blocking.
        #
        # Phase 3: legacy SoA pinned dispatch buffers (_dispatch_pinned_*)
        # removed. The GPU AOS path produces records directly into
        # segment_pipeline staging on GPU → ExpertState's pinned_aos
        # mirror; no per-(idx, p, m, v) host buffers needed any more.

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

        # Stash for next iter's dispatch (replace mode). Dispatch
        # consumes prev iter's name_count + matching prev iter's adam
        # scalars — these are written here and read in the next call to
        # _dispatch_pending_payload.
        self._outlier_lr_over_bc1 = float(lr_over_bc1)
        self._outlier_sqrt_bc2_inv = float(sqrt_bc2_inv)
        self._outlier_eps_adam = float(eps_adam)

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

        # Outlier kernel pipeline (production single path):
        #   Pass A (outlier count): atomic histogram of elements above prev_th
        #   pass2 (group select): atomic histogram of groups above group_th
        #   Pass B (AOS write): assemble records + selective fold
        if outlier_on:
            device = self._g_fast_dg_flat.device
            if self._outlier_prev_th is None:
                self._outlier_prev_th = torch.full(
                    (), float("inf"), dtype=torch.float32, device=device,
                )
                self._outlier_prev_floor = torch.zeros(
                    (), dtype=torch.float32, device=device,
                )

            # Persistent element-axis sub-sample buffer
            # (~1.5 MB total) — replaces linear-bin histc walk.
            _SAMPLE_SIZE = 65536
            if (getattr(self, "_g_fast_score_sample_buf", None) is None
                    or self._g_fast_score_sample_buf.numel() != _SAMPLE_SIZE):
                self._g_fast_score_sample_buf = torch.empty(
                    _SAMPLE_SIZE, dtype=torch.float32, device=device,
                )
                self._g_fast_score_sample_sorted_buf = torch.empty(
                    _SAMPLE_SIZE, dtype=torch.float32, device=device,
                )
                self._g_fast_score_sample_indices_buf = torch.empty(
                    _SAMPLE_SIZE, dtype=torch.int64, device=device,
                )
            # Reset to 0 — kernel writes only to positions multiples of stride.
            self._g_fast_score_sample_buf.zero_()
            _n_total_elems_tracked = total
            _sample_stride = max(1, _n_total_elems_tracked // _SAMPLE_SIZE)
            self._g_fast_score_sample_stride = int(_sample_stride)
            self._g_fast_score_sample_total = int(_n_total_elems_tracked)

            _n_names_now = len(self._tracked_names)
            if (self._outlier_name_count is None
                    or int(self._outlier_name_count.numel()) != _n_names_now):
                self._outlier_name_count = torch.zeros(
                    _n_names_now, dtype=torch.int64, device=device,
                )
            _ofs_t = getattr(self, "_g_fast_offsets_t", None)
            if _ofs_t is None:
                _ofs_t = torch.tensor(
                    self._g_fast_offsets_cpu, dtype=torch.long,
                    device=device,
                )
                self._g_fast_offsets_t = _ofs_t
                self._g_fast_arange_g = torch.arange(
                    g, device=device, dtype=torch.long,
                )
            if (self._outlier_elem_offsets is None
                    or self._outlier_elem_offsets.numel() != _ofs_t.numel()):
                self._outlier_elem_offsets = (_ofs_t.to(torch.int64) * g).contiguous()
            self._outlier_name_count.zero_()

        # Pass A pre-loop ALWAYS runs when outlier_on — produces
        # per-name outlier count + fills element-axis sub-sample buffer.
        if outlier_on:
            from .outlier_kernels import launch_outlier_count_for_group
            for grp in self._groups:
                if not grp["zero_copy"]:
                    continue
                master_off_elem = int(grp["master_off"]) * g
                total_a = grp["total_aligned"]
                launch_outlier_count_for_group(
                    m_flat=grp["m_flat"],
                    v_flat=grp["v_flat"],
                    accum=grp["dg_buf"],
                    h_g=grp["h_g_buf"],
                    name_count=self._outlier_name_count,
                    ofs_elem_full=self._outlier_elem_offsets,
                    prev_th_t=self._outlier_prev_th,
                    floor_t=self._outlier_prev_floor,
                    group_th_t=self._g_fast_prev_threshold_t,
                    master_off_elem=master_off_elem,
                    g_blk=g,
                    lr_over_bc1=float(lr_over_bc1),
                    sqrt_bc2_inv=float(sqrt_bc2_inv),
                    eps=float(eps_adam),
                    n_names=int(self._outlier_elem_offsets.numel() - 1),
                    sample_buf=self._g_fast_score_sample_buf,
                    sample_stride=self._g_fast_score_sample_stride,
                    sample_size=self._g_fast_score_sample_buf.numel(),
                )

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
                # Pass A pre-loop already ran for this group:
                # it computes Δθ² and does selective fold (dg += Σ
                # non-selected Δθ²). Pass B-group later zeros dg for
                # groups picked at group-pool level.
                pass
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
                and getattr(self, "_flat_layout_done", False)):
            # Stochastic sub-sampling on PER-ELEMENT axis.
            # Pass A kernel writes every Sth element's score to
            # ``_g_fast_score_sample_buf`` at deterministic positions
            # (no atomic). Sort + pick K-th order statistic gives the
            # accurate per-element threshold under heavy-tail without
            # linear-bin histc's over-selection failure mode.
            #
            # group_th = elem_th × g_blk (group score is sum-of-g_blk-
            # elements in code units → axis match).
            #
            # Memory: ~1.5 MB persistent (sample + sorted + indices),
            # FIXED size independent of model scale (100B / 1000B both
            # fit identically). No alloc/free per iter.
            #
            # Multi-GPU (EP/TP partition): each GPU samples LOCAL
            # element scores. Future: allgather ~8K samples per GPU
            # for unified threshold (~32 KB send per iter @ W=64).
            sample_buf = getattr(self, "_g_fast_score_sample_buf", None)
            sorted_buf = getattr(self, "_g_fast_score_sample_sorted_buf", None)
            indices_buf = getattr(self, "_g_fast_score_sample_indices_buf", None)
            sample_total = int(getattr(
                self, "_g_fast_score_sample_total", 0,
            ))
            sample_stride = int(getattr(
                self, "_g_fast_score_sample_stride", 0,
            ))
            total_groups_outlier = self._g_fast_dg_flat.numel()
            total_elems_outlier = total_groups_outlier * g_blk
            K_total = max(1, int(self.topk_ratio * total_elems_outlier))
            FLOOR_MULT = 5

            if (sample_buf is None or sorted_buf is None
                    or indices_buf is None
                    or sample_total <= 0 or sample_stride <= 0):
                # Cold start (iter 0): no samples → open the gate.
                elem_th_val = 0.0
                group_th_val = 0.0
                elem_floor_val = 0.0
            else:
                # Number of sample positions filled = ceil(sample_total / sample_stride),
                # capped by SAMPLE_SIZE.
                n_filled = min(
                    sample_buf.numel(),
                    (sample_total + sample_stride - 1) // sample_stride,
                )
                if n_filled <= 0:
                    elem_th_val = float("inf")
                    group_th_val = float("inf")
                    elem_floor_val = float("inf")
                else:
                    sample_view = sample_buf[:n_filled]
                    sorted_view = sorted_buf[:n_filled]
                    indices_view = indices_buf[:n_filled]
                    # Ascending sort via out= (no per-iter alloc).
                    torch.sort(sample_view, out=(sorted_view, indices_view))
                    # Detect cold/zero distribution: at iter 0 m, v ≈ 0
                    # → all scores ≈ 0. top-K of zeros = 0 → would let
                    # everything pass. Guard: if max sample ≈ 0, keep
                    # threshold at +inf (= 0 selected this iter; converges
                    # by iter 2-3 once Adam state has signal).
                    _top_sample = float(sorted_view[n_filled - 1].item())
                    if _top_sample <= 1e-20:
                        elem_th_val = float("inf")
                        group_th_val = float("inf")
                        elem_floor_val = float("inf")
                    else:
                        # K-th order statistic from top.
                        K_in_sample = max(
                            1, (K_total * n_filled) // max(1, sample_total),
                        )
                        K_in_sample = min(K_in_sample, n_filled)
                        idx_th = n_filled - K_in_sample
                        idx_th = max(0, min(idx_th, n_filled - 1))
                        elem_th_val = float(sorted_view[idx_th].item())
                        # Cold-start mitigation: if the K-th value is
                        # very small (= sample has many zero/near-zero
                        # scores from cold gradients), enforce a floor
                        # at half of sample max to prevent over-selection.
                        # As scores fill in, this clamp self-disengages.
                        _half_max = _top_sample * 0.5
                        if elem_th_val < _half_max:
                            elem_th_val = _half_max
                        # Floor = top (K_in_sample × FLOOR_MULT)-th.
                        K_in_sample_floor = min(
                            n_filled, K_in_sample * FLOOR_MULT,
                        )
                        idx_floor = n_filled - K_in_sample_floor
                        idx_floor = max(0, min(idx_floor, n_filled - 1))
                        elem_floor_val = float(sorted_view[idx_floor].item())
                        if elem_floor_val < _half_max * 0.5:
                            elem_floor_val = _half_max * 0.5
                        # group score (code sum units) >= group_th if mean
                        # per element ≥ elem_th → sum ≥ elem_th × g_blk.
                        group_th_val = elem_th_val * g_blk
                    # Update prev_smax for diagnostics.
                    self._outlier_prev_smax = max(_top_sample, 1e-30)
                    self._prev_s_max_g = max(_top_sample * g_blk, 1e-30)

            self._outlier_prev_th.fill_(elem_th_val)
            self._outlier_prev_floor.fill_(elem_floor_val)
            self._g_fast_prev_threshold_t.fill_(group_th_val)
            # Debug: print threshold every iter (until thrott).
            if prev_step <= 10:
                _top_dbg = float(sorted_buf[(sample_buf.numel()) - 1].item()) if (sample_buf is not None and sample_buf.numel() > 0) else -1.0
                _min_dbg = float(sorted_buf[0].item()) if (sample_buf is not None and sample_buf.numel() > 0) else -1.0
                print(f"[PHASE_B] @iter{prev_step} elem_th={elem_th_val:.3e} group_th={group_th_val:.3e} sample_max={_top_dbg:.3e} sample_min={_min_dbg:.3e}",
                      flush=True)

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
        # Default OFF for production; enable with PROPOSAL_DISPATCH_DBG=1 to
        # log the dispatch_dbg breakdown + dispatch_coverage per-iter.
        _dsp_dbg = _os_dsp.environ.get("PROPOSAL_DISPATCH_DBG", "") in ("1", "true", "TRUE")
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
        import os as _os_trace_dsp
        _os_trace_dsp.write(2, f"[DSP_ENTRY] pending_is_none={pending is None}\n".encode())
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

        # name_group_count replaces seg_bounds + top_idx_sorted.
        # i64 tensor of per-name selected group counts.
        name_group_count = pending["name_group_count"]
        prev_step_dbg = int(pending.get("step", -1))
        if prev_step_dbg <= 10:
            _ngc_sum = int(name_group_count.sum().item())
            _ngc_max = int(name_group_count.max().item()) if name_group_count.numel() > 0 else 0
            print(f"[PHASE_B_DSP] @prev_iter{prev_step_dbg} name_group_count_sum={_ngc_sum} max={_ngc_max} n_names={name_group_count.numel()}",
                  flush=True)
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

        name_group_count_cpu_list = name_group_count.cpu().tolist()
        # Synthesize seg_bounds_cpu as cumsum of per-name counts so the
        # downstream code that indexes by (i, i+1) keeps working with
        # minimal change: seg_bounds_cpu[i] = sum(name_group_count[:i]).
        # Build host-side tensor so subsequent ``.item()`` calls work.
        import torch as _t_cumsum
        _cs = [0]
        for _c in name_group_count_cpu_list:
            _cs.append(_cs[-1] + int(_c))
        seg_bounds_cpu = _t_cumsum.tensor(_cs, dtype=_t_cumsum.long)
        _t_after_segcpu = _time_dsp.perf_counter() if _dsp_dbg else 0.0
        device = next(self.model.parameters()).device
        g_blk = max(1, int(self.block_quant_g))
        arange_g = self._g_fast_arange_g
        mapper = get_mapper()
        stream = self._tracker_stream

        # Snapshot prev_th + group_th BEFORE walk_thresholds
        # overwrites them, so Pass B (which runs later in this same
        # worker call) uses the SAME thresholds as Pass A. Walk's new
        # threshold values are for the NEXT iter's Pass A — we must
        # not let them leak into THIS iter's Pass B.
        if (self._outlier_prev_th is not None
                and self._g_fast_prev_threshold_t is not None):
            if getattr(self, "_outlier_prev_th_snap", None) is None:
                self._outlier_prev_th_snap = torch.zeros(
                    (), dtype=torch.float32, device=device,
                )
                self._outlier_group_th_snap = torch.zeros(
                    (), dtype=torch.float32, device=device,
                )
            self._outlier_prev_th_snap.copy_(self._outlier_prev_th)
            self._outlier_group_th_snap.copy_(self._g_fast_prev_threshold_t)

        self._walk_thresholds(pending, prev_step, g_blk)

        # Outlier-promotion: derive elem_idx from the LIVE GPU buffer
        # (now stored as a reference in pending, not a clone — saves
        # +1.3 GB persistent VRAM). ev.synchronize() above already
        # waited for prev tracker writes to settle, so the .item()
        # and .to(long) reads here see consistent T-1 data. The
        # follow-up .cpu().tolist() at the end of olsort host-syncs
        # worker_stream, which guarantees this read completes before
        # update_delta_accum's overwrite kernel launches on
        # tracker_stream later in this worker task.
        # Legacy outlier sort path removed. Per-name outlier
        # counts come directly from Pass A's _outlier_name_count (no
        # FIFO + sort + searchsorted needed).
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
        # Fetch per-name outlier count from the outlier-count kernel's
        # GPU buffer (single host sync per iter).
        _outlier_name_count_host: List[int] = (
            self._outlier_name_count.cpu().tolist()
        )

        name_cnts: List[int] = []     # full per-name (group_elems + outlier)
        name_cnt_g: List[int] = []    # per-name group element count (cnt_g × g_blk)
        name_cnt_o: List[int] = []    # per-name outlier count
        name_offs: List[int] = [0]
        for i, name in enumerate(name_list):
            s_b_pre = int(seg_bounds_cpu[i].item())
            e_b_pre = int(seg_bounds_cpu[i + 1].item())
            cnt_g_pre = e_b_pre - s_b_pre
            cnt_o_pre = int(_outlier_name_count_host[i]) if i < len(_outlier_name_count_host) else 0
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

        # Phase 3: GPU AOS path is the ONLY dispatch path. The legacy
        # CPU-side scatter / SoA-pinned / chunked-OOM-mitigation are
        # removed — records are interleaved on GPU (gpu_aos kernel) and
        # streamed to pinned_aos via segment_pipeline at 128 MiB
        # granularity, so the per-iter transient peak is bounded by
        # segment_pipeline (256 MiB) regardless of K_total.
        if True:
            per_param_selected: Dict[str, int] = {}
            event_h2d: Optional["torch.cuda.Event"] = None

            per_grp_gpu: List[Optional[Tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor]]] = [None] * len(self._groups)

            with sm_tracker:
                last_valid = int(seg_bounds_cpu[-1].item())
                # dg_flat.index_fill_ skip — selective fold is
                # now performed by Pass B-group kernel during AOS write
                # (kernel atomic_inc per-name slot + rewrite dg for
                # selected groups). No host-side index_fill needed.

                _g_off_per_grp_cache = getattr(
                    self, "_master_off_per_grp_t_cache", None)
                if _g_off_per_grp_cache is None:
                    _g_off_per_grp_cache = torch.tensor(
                        [g["master_off"] for g in self._groups],
                        dtype=torch.long, device=device,
                    )
                    self._master_off_per_grp_t_cache = _g_off_per_grp_cache
                _master_off_per_grp_t = _g_off_per_grp_cache

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

                n_per_grp_g_list = [
                    int(seg_bounds_cpu[g["end_name_in_grp"]].item())
                    - int(seg_bounds_cpu[g["start_name_in_grp"]].item())
                    for g in self._groups
                ]

                #  stage all per-iter int64 metadata through a single
                # pinned-host/device pool (replaces 3 separate per-iter
                # `torch.tensor(host_list, device=cuda)` calls; each was
                # implicitly cudaStreamSynchronize-ing tracker_stream).
                _n_names_now = len(name_cnt_g)
                _n_grps_now = len(n_per_grp_g_list)
                _meta_total = _n_names_now * 2 + _n_grps_now
                if (self._dispatch_meta_host is None
                        or int(self._dispatch_meta_host.numel()) < _meta_total):
                    _cap = max(_meta_total, 4096)
                    self._dispatch_meta_host = torch.empty(
                        _cap, dtype=torch.int64, pin_memory=True,
                    )
                    self._dispatch_meta_dev = torch.empty(
                        _cap, dtype=torch.int64, device=device,
                    )
                if self._dispatch_meta_event is not None:
                    # Wait for prev iter's async H2D to complete before we
                    # overwrite the pinned host slab. Steady-state no-op.
                    self._dispatch_meta_event.synchronize()
                _meta_host = self._dispatch_meta_host
                _meta_dev = self._dispatch_meta_dev
                # Write all 3 lists into contiguous pinned slices via
                # torch.as_tensor (single C-level list→tensor conversion
                # per list + one CPU memcpy into pinned). Avoids the per-
                # call sync of torch.tensor(...,device=cuda).
                _meta_host.narrow(0, 0, _n_names_now).copy_(
                    torch.as_tensor(name_cnt_g, dtype=torch.int64)
                )
                _meta_host.narrow(0, _n_names_now, _n_names_now).copy_(
                    torch.as_tensor(name_cnt_o, dtype=torch.int64)
                )
                _meta_host.narrow(0, 2 * _n_names_now, _n_grps_now).copy_(
                    torch.as_tensor(n_per_grp_g_list, dtype=torch.int64)
                )
                _meta_dev.narrow(0, 0, _meta_total).copy_(
                    _meta_host.narrow(0, 0, _meta_total), non_blocking=True,
                )
                if self._dispatch_meta_event is None:
                    self._dispatch_meta_event = torch.cuda.Event()
                self._dispatch_meta_event.record()
                # Views into the device slab. Kernels read these directly.
                _all_cnt_g_t = _meta_dev.narrow(0, 0, _n_names_now)
                _all_cnt_o_t = _meta_dev.narrow(0, _n_names_now, _n_names_now)
                n_per_grp_g_t = _meta_dev.narrow(
                    0, 2 * _n_names_now, _n_grps_now
                )

                # top_idx_sorted-based elem_idx construction
                # removed. Group records are written by Pass B-group
                # kernel (rescans + atomic_inc per-name slot + emits).
                grp_g_starts_in_global: List[int] = [0]
                for n in n_per_grp_g_list:
                    grp_g_starts_in_global.append(
                        grp_g_starts_in_global[-1] + n
                    )

                for gi, grp in enumerate(self._groups):
                    sn = grp["start_name_in_grp"]
                    en = grp["end_name_in_grp"]
                    if sn == en:
                        continue
                    g_size = grp_groups_size[gi]
                    o_size = grp_outliers_size[gi]
                    if g_size == 0 and o_size == 0:
                        continue
                    master_off_g = grp["master_off"]
                    master_off_g_elem = master_off_g * g_blk
                    m_flat_g = grp["m_flat"]
                    v_flat_g = grp["v_flat"]
                    p_flat_g = grp["p_flat"]

                    # Pass m/v/p flats directly to Pass B-group
                    # via per_grp_gpu. The kernel does on-the-fly group
                    # selection (re-scan + score compare) and writes
                    # records — no host-side elem_idx materialisation.
                    if (m_flat_g is not None and v_flat_g is not None
                            and p_flat_g is not None):
                        # First slot is a dummy (kernel ignores src_idx in
                        # Phase B path; persistent buffer for lifetime).
                        per_grp_gpu[gi] = (
                            self._g_fast_h_g_bf16_flat,  # dummy persistent ref
                            p_flat_g, m_flat_g, v_flat_g,
                        )

            # Phase 3: tracker_event = end-of-Phase-A signal on tracker_stream.
            # Worker thread waits on this before launching the gpu_aos kernel,
            # so it sees a consistent per_grp_gpu view. No D2H to pinned here —
            # the GPU AOS kernel reads per_grp_gpu directly and writes records
            # straight into segment_pipeline staging.
            event_compute: Optional["torch.cuda.Event"] = None
            if stream is not None and total_pinned > 0:
                event_compute = self._acquire_ring_event("compute")
                event_compute.record(stream)
            gpu_refs_for_writer: List[torch.Tensor] = [
                t for gpu in per_grp_gpu if gpu is not None for t in gpu
            ]
            # Phase 2c: preserve the structured per-BF16-group payload
            # for the gpu_aos kernel. Lifetime piggy-backs on
            # ``gpu_refs_for_writer`` (= same tensors), so this doesn't
            # extend GPU memory residency.
            self._gpu_payload_per_group = list(per_grp_gpu)
            per_grp_gpu = []  # type: ignore[assignment]

        # Phase 3: build per-name GPU entry descriptors. Each entry is
        # ``(bf16_group_idx, src_off_in_group, n_records)`` — the slice
        # of ``self._gpu_payload_per_group[bf16_group_idx]`` that this
        # name's records occupy. The per-BF16-group GPU SoA tensors are
        # concatenated as ``[groups_records ; outliers_records]`` (see
        # the elem_idx_grp_bf16 cat site), so within a group:
        #   groups   region:  [0, grp_groups_size[gi])
        #   outliers region:  [grp_groups_size[gi], grp_groups_size[gi]
        #                       + grp_outliers_size[gi])
        # IDX inside the per-BF16-group SoA is BF16-group-local; the
        # AOS kernel converts to on-disk idx by adding the tname's
        # ``global_offsets[tname]`` (= ``t_off``) supplied via the
        # entry-spec table.
        gpu_entries: Dict[str, List[Tuple[int, int, int]]] = {}
        for gi, grp in enumerate(self._groups):
            sn = grp["start_name_in_grp"]
            en = grp["end_name_in_grp"]
            grp_o_src_start = grp_groups_size[gi]
            cum_g = 0
            cum_o = 0
            for i in range(sn, en):
                name = name_list[i]
                cg = name_cnt_g[i]
                co = name_cnt_o[i]
                gpu_sub: List[Tuple[int, int, int]] = []
                # Outlier records written by Pass B kernel
                # directly to AOS staging. Register the tname even when
                # cg=0 so plan_layout reserves the outlier byte slot
                # and Pass B's name_offset_bytes is non-sentinel.
                if cg > 0 or co > 0:
                    gpu_sub.append((gi, cum_g, cg))
                if gpu_sub:
                    gpu_entries[name] = gpu_sub
                per_param_selected[name] = cg + co
                cum_g += cg
                cum_o += co

        self._gpu_payload_entries = gpu_entries
        _t_after_loop = _time_dsp.perf_counter() if _dsp_dbg else 0.0
        _t_after_streamsync = _t_after_loop

        _t_after_batchbuild = _t_after_streamsync
        if self._expert_state is not None:
            # Phase 3: per_expert_batch values are GPU descriptors
            # (tname, bf16_group_idx, src_off, n_records) — no CPU
            # tensors. Worker uses gpu_aos kernel to interleave records
            # directly from ``self._gpu_payload_per_group`` into
            # segment_pipeline staging, then D2H to pinned_aos.
            per_expert_batch: Dict[Tuple[int, int],
                                   List[Tuple[str, int, int, int]]] = {}
            for name, sub_entries in gpu_entries.items():
                eid = self._tname_to_eid.get(name)
                if eid is None or eid in excluded_eids:
                    continue
                bucket = per_expert_batch.setdefault(eid, [])
                for (bf16_g, src_off, n_recs) in sub_entries:
                    bucket.append((name, bf16_g, src_off, n_recs))
            _t_after_batchbuild = _time_dsp.perf_counter() if _dsp_dbg else 0.0
            # Build LossyDispatchState + per-expert outlier counts.
            _lossy_state = None
            _per_expert_outlier_n = None
            if _outlier_name_count_host is not None:
                # Build per_expert_outlier_n: {eid: {tname: n_outlier}}.
                # Each name maps to exactly one expert via _tname_to_eid.
                _per_expert_outlier_n = {}
                for i, nm in enumerate(name_list):
                    n_o = int(_outlier_name_count_host[i]) if i < len(_outlier_name_count_host) else 0
                    if n_o <= 0:
                        continue
                    eid_nm = self._tname_to_eid.get(nm)
                    if eid_nm is None or eid_nm in excluded_eids:
                        continue
                    _per_expert_outlier_n.setdefault(eid_nm, {})[nm] = n_o

                # LossyDispatchState: collect per-BF16-group m/v/p/h_g/dg flats
                # + master_off_elem in element coords.
                _mvph: List[Optional[Tuple[
                    torch.Tensor, torch.Tensor, torch.Tensor,
                    torch.Tensor, torch.Tensor, int,
                ]]] = []
                for grp in self._groups:
                    if not grp.get("zero_copy", False):
                        _mvph.append(None)
                        continue
                    p_flat_g = grp.get("p_flat")
                    if p_flat_g is None:
                        # Group without flat p view — Pass B can't write
                        # outliers for this group. Skip (graceful
                        # degradation; outlier CP missing for this group).
                        _mvph.append(None)
                        continue
                    _mvph.append((
                        grp["m_flat"], grp["v_flat"], p_flat_g,
                        grp["h_g_buf"], grp["dg_buf"],
                        int(grp["master_off"]) * g_blk,
                    ))

                # name_slot: per-name atomic counter, zeroed per iter.
                # name_t_off: per-name on-disk t_off (computed from ent.global_offsets).
                # Build host-side then push to dev.
                _n_names = len(name_list)
                if (getattr(self, "_outlier_name_slot", None) is None
                        or self._outlier_name_slot.numel() != _n_names):
                    self._outlier_name_slot = torch.zeros(
                        _n_names, dtype=torch.int64, device=device,
                    )
                self._outlier_name_slot.zero_()

                # name_t_off resolved via expert_state.entries[eid].global_offsets[tname].
                if (getattr(self, "_outlier_name_t_off", None) is None
                        or self._outlier_name_t_off.numel() != _n_names):
                    self._outlier_name_t_off = torch.zeros(
                        _n_names, dtype=torch.int64, device=device,
                    )
                _t_off_host = []
                _expert_state_entries = getattr(
                    self._expert_state, "_entries", {},
                )
                for nm in name_list:
                    eid_nm = self._tname_to_eid.get(nm)
                    ent_nm = _expert_state_entries.get(eid_nm) if eid_nm is not None else None
                    if ent_nm is not None and nm in ent_nm.global_offsets:
                        _t_off_host.append(int(ent_nm.global_offsets[nm]))
                    else:
                        _t_off_host.append(0)
                self._outlier_name_t_off.copy_(
                    torch.as_tensor(_t_off_host, dtype=torch.int64),
                    non_blocking=True,
                )

                from .gpu_aos import LossyDispatchState
                # Use SNAPSHOTTED prev_th/group_th — same values
                # Pass A saw in update_delta_accum. _outlier_prev_th was
                # overwritten by walk_thresholds above with this iter's
                # NEW threshold (which is for NEXT iter). Pass B needs
                # the OLD value to match Pass A's selection.
                _outlier_prev_th_for_b = getattr(
                    self, "_outlier_prev_th_snap", self._outlier_prev_th,
                )
                _outlier_group_th_for_b = getattr(
                    self, "_outlier_group_th_snap", self._g_fast_prev_threshold_t,
                )
                # Build per-name within_off (host-side int list) by
                # reusing the already-computed ``_winoff_list`` order which
                # is parallel to name_list (per-BF16-group, name-within-
                # group). The same computation is performed in
                # ``_winoff_concat_cache`` build above; rebuilding the host
                # list here avoids materialising the GPU tensor host-side
                # (no .cpu() sync). This is cheap host arithmetic.
                _outlier_within_off_host: List[int] = []
                for grp_init_w in self._groups:
                    sn_iw = grp_init_w["start_name_in_grp"]
                    en_iw = grp_init_w["end_name_in_grp"]
                    moff_iw = grp_init_w["master_off"]
                    for kw in range(en_iw - sn_iw):
                        _outlier_within_off_host.append(
                            (offsets_cpu[sn_iw + kw] - moff_iw) * g_blk
                        )

                # Build per-name slot for GROUP records (Pass
                # B-group writes here atomically). Same shape as Pass A's
                # name_slot. Zero it per iter.
                if (getattr(self, "_g_fast_name_group_slot", None) is None
                        or self._g_fast_name_group_slot.numel() != _n_names):
                    self._g_fast_name_group_slot = torch.zeros(
                        _n_names, dtype=torch.int64, device=device,
                    )
                self._g_fast_name_group_slot.zero_()

                _lossy_state = LossyDispatchState(
                    bf16_groups_mvph=_mvph,
                    ofs_elem_full=self._outlier_elem_offsets,
                    name_list=list(name_list),  # tracker's authoritative name ordering
                    name_outlier_count_host=list(_outlier_name_count_host),
                    name_outlier_count_dev=self._outlier_name_count,
                    name_t_off=self._outlier_name_t_off,
                    name_slot=self._outlier_name_slot,  # outlier slot
                    prev_th_t=_outlier_prev_th_for_b,
                    group_th_t=_outlier_group_th_for_b,
                    lr_over_bc1=getattr(self, "_outlier_lr_over_bc1", 1e-4),
                    sqrt_bc2_inv=getattr(self, "_outlier_sqrt_bc2_inv", 1.0),
                    eps=getattr(self, "_outlier_eps_adam", 1e-8),
                    g_blk=int(g_blk),
                    aos_byte_off_per_name=[],   # filled by LossyAosPipeline.run
                    within_off_per_name=_outlier_within_off_host,
                    exc_mask_full=self._g_fast_exc_mask,
                    name_group_slot=self._g_fast_name_group_slot,
                    selected_mask=pending.get("selected_mask"),
                )

            self._expert_state.lossy_fc_append_batch(
                step=int(prev_step),
                per_expert_updates=per_expert_batch,
                tracker_event=event_compute,
                gpu_refs=gpu_refs_for_writer,
                bf16_groups=self._gpu_payload_per_group,
                lossy_state=_lossy_state,
                per_expert_outlier_n=_per_expert_outlier_n,
            )
            # Per-expert dispatch coverage diagnostic — runs every 10
            # iters to confirm routing is balanced.
            if prev_step > 0 and prev_step % 10 == 0:
                _n_experts = len(per_expert_batch)
                _total_entries = 0
                _max_entries = 0
                _min_entries = float("inf") if _n_experts > 0 else 0
                for _eid, _bucket in per_expert_batch.items():
                    _per_expert = 0
                    for (_n, _bf16_g, _src_off, _n_recs) in _bucket:
                        _per_expert += int(_n_recs)
                    _total_entries += _per_expert
                    if _per_expert > _max_entries:
                        _max_entries = _per_expert
                    if _per_expert < _min_entries:
                        _min_entries = _per_expert
                if _n_experts == 0:
                    _min_entries = 0
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
        _vram_probe = _os_bench_ss.environ.get(
            "PROPOSAL_VRAM_PEAK_PROBE", "") in ("1", "true", "True")
        if _vram_probe and torch.cuda.is_available():
            # Reset peak BEFORE dispatch so this iter's peak is isolated.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            _vp_before_dispatch = torch.cuda.memory_allocated() / 1e6
            _vp_peak_before = torch.cuda.max_memory_allocated() / 1e6
        # In Option A REPLACE mode, the dispatch is moved to AFTER
        # update_delta_accum (= same-iter dispatch). This ensures Pass B
        # reads the SAME m, v, p as Pass A — otherwise the optimizer.step()
        # between worker calls would mutate m, v and Pass B would compute
        # a different selection set than Pass A counted → slot overflow
        # → CUDA illegal memory access.
        # Replace mode is the sole production path: dispatch
        # runs at the END of step() (same-iter), not here.
        if _vram_probe and torch.cuda.is_available():
            torch.cuda.synchronize()
            _vp_after_dispatch = torch.cuda.memory_allocated() / 1e6
            _vp_peak_after_dispatch = torch.cuda.max_memory_allocated() / 1e6
            print(f"[vram_probe] step={global_step} "
                  f"BEFORE dispatch: alloc={_vp_before_dispatch:.0f}MB peak={_vp_peak_before:.0f}MB | "
                  f"AFTER dispatch: alloc={_vp_after_dispatch:.0f}MB peak={_vp_peak_after_dispatch:.0f}MB | "
                  f"Δpeak_in_dispatch={_vp_peak_after_dispatch - _vp_peak_before:+.0f}MB",
                  flush=True)
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
            ev_opt = self._acquire_ring_event("evopt")
            ev_opt.record(cur_main)
            stream.wait_event(ev_opt)
            # Switch default stream → tracker_stream so all GPU launches
            # below land in tracker_stream without an extra `with` block.
            prev_stream = cur_main
            torch.cuda.set_stream(stream)
        # Mark we're inside tracker_stream so update_delta_accum's DEBUG
        # sync (which would block main stream) is suppressed.
        self._inside_tracker_stream = True

        # v_t → mapped(v_t) into _g_fast_h_g_bf16_flat (no-op until the
        # first anchor populates _g_fast_h_quantiles). Runs on
        # tracker_stream so it overlaps freely with main_stream's next
        # forward/backward; no host-blocking sync.
        #
        if not _bench_no_tracker_kernel:
            self.update_v_mapping()
        # Anchor scale-shift correction: cp_hooks captures h ratio across
        # anchor's _fold_h (pre/post). Scale prev_threshold by the same
        # ratio so score and threshold stay on the same scale, preventing
        # all-scores-below-threshold dead pass2 iter after anchor refresh.
        if getattr(self, "_anchor_just_fired", False):
            self._anchor_just_fired = False
            ratio = float(getattr(self, "_anchor_h_ratio", 1.0))
            if (getattr(self, "_g_fast_prev_threshold_t", None) is not None
                    and ratio > 0 and abs(ratio - 1.0) > 1e-6):
                self._g_fast_prev_threshold_t.mul_(ratio)
                print(f"[ANCHOR_FIX] @iter{global_step} threshold scaled by ratio={ratio:.4f}",
                      flush=True)

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
            if not hasattr(self, "last_phase_ms") or self.last_phase_ms is None:
                self.last_phase_ms = {}
            self.last_phase_ms.update({
                "setup": 0.0, "pass1": 0.0, "pass2": 0.0,
                "pass3": 0.0, "post": 0.0,
            })
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
        # K-budget kept only for diagnostic / next-iter sample stride.
        # sel_buf, sort, searchsorted entirely removed.
        K_total = max(1, int(self.topk_ratio * total_elems))
        K_groups = max(1, K_total // g_blk)
        K_groups = min(K_groups, total_groups)
        threshold_t = self._g_fast_prev_threshold_t  # GPU scalar (set by walk)
        # Diagnostic-only s_max placeholder (used by some debug prints).
        s_max = max(getattr(self, "_prev_s_max_g", None) or 1e-6, 1e-30)

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

        # Running True-count cumulative (GPU scalar).  persistent
        # i64 scalar slab, reused per iter via in-place ``zero_()``.
        # Replaces per-iter ``torch.zeros(())`` + per-chunk
        # ``true_cum = true_cum + ...`` reassignment with single-slab
        # ``add_()``. _g_fast_true_cum is lazy because device wasn't
        # known at __init__ for the scalar dtype.
        true_cum = getattr(self, "_g_fast_true_cum", None)
        if true_cum is None:
            true_cum = torch.zeros((), dtype=torch.long, device=device)
            self._g_fast_true_cum = true_cum
        else:
            true_cum.zero_()

        # Replace per-chunk torch ops (mask/cumsum/scatter/
        # histc) with a single Triton kernel ``_pass2_select_count``
        # that does:
        #   - score = h_g × dg × exc_mask
        #   - if score >= threshold: atomic_inc(name_group_count[bisect
        #     ofs_t, global_gid])
        # No sel_buf, no global_hist, no top_idx_sorted, no
        # searchsorted — all of those K_max-sized buffers are gone.
        # name_group_count is N_names × i64 (~수십 KB) — fits the
        # user's "histc 진짜 조금 필요한 정도" budget.
        n_names_full = int(offsets_t.numel() - 1)
        if (getattr(self, "_g_fast_name_group_count", None) is None
                or self._g_fast_name_group_count.numel() != n_names_full):
            self._g_fast_name_group_count = torch.zeros(
                n_names_full, dtype=torch.int64, device=device,
            )
        self._g_fast_name_group_count.zero_()
        # Per-group selection mask (int8 — bool would need
        # special Triton support): pass2 writes selection result, Pass
        # B-group reads — decouples from dg drift across streams.
        # Per-group selection mask (int8): pass2 writes, Pass B-group reads.
        # Stash clones to a transient snapshot — keeping the clone (vs
        # N-deep double-buffer) is the safe path because the async worker
        # queue depth can exceed 2 under load.
        if (getattr(self, "_g_fast_selected_mask", None) is None
                or self._g_fast_selected_mask.numel() != total_groups):
            self._g_fast_selected_mask = torch.zeros(
                total_groups, dtype=torch.int8, device=device,
            )
        self._g_fast_selected_mask.zero_()

        import os as _os_p2
        _bench_no_pass2_local = _os_p2.environ.get(
            "PROPOSAL_BENCH_NO_PASS2", "") in ("1", "true", "TRUE")
        if not _bench_no_pass2_local:
            from .outlier_kernels import launch_group_select
            launch_group_select(
                h_g_flat=h_g_flat,
                dg_flat=dg_flat,
                exc_mask=exc_mask,
                offsets_t=offsets_t,
                name_group_count=self._g_fast_name_group_count,
                selected_mask=self._g_fast_selected_mask,
                threshold_t=threshold_t,
                n_groups_total=total_groups,
                n_names=n_names_full,
                chunk_size=chunk_size,
            )
        if getattr(self, "_prev_s_max_g", None) is None:
            self._prev_s_max_g = float(s_max)
        if _ev1 is not None:
            _ev1.record()

        _t0_p2_end = _time.perf_counter()
        phase_pass2_ms = (_t0_p2_end - _t0_p2) * 1000.0
        if _vram_probe and torch.cuda.is_available():
            torch.cuda.synchronize()
            _vp_after_p2 = torch.cuda.memory_allocated() / 1e6
            _vp_peak_after_p2 = torch.cuda.max_memory_allocated() / 1e6
            print(f"[vram_probe] step={global_step} "
                  f"AFTER pass2: alloc={_vp_after_p2:.0f}MB peak={_vp_peak_after_p2:.0f}MB | "
                  f"Δpeak_in_pass2={_vp_peak_after_p2 - _vp_peak_after_dispatch:+.0f}MB",
                  flush=True)

        # Restore main default stream (we set tracker_stream as default
        # at STEP 1 entry).
        self._inside_tracker_stream = False
        if prev_stream is not None:
            torch.cuda.set_stream(prev_stream)

        # Outlier-promotion: stash LIVE references (not clones) of
        # selected element indices/count. Reference is safe because the
        # consumer (next iter's dispatch) reads sel_idx exactly once at
        # the very start (``sel_idx_snap[:n_sel].to(torch.long)``) on
        # worker_stream, followed by a HOST sync at ``.cpu().tolist()``
        # (line ~1948) BEFORE update_delta_accum's overwrite kernel is
        # launched on tracker_stream — so the read-then-write order is
        # enforced by the host barrier inside the same worker task.
        # Eliminates the +1.3 GB persistent VRAM that the prior clone
        # held (K_max int64 buffer) without any algorithm change.
        # Record tracker work completion event AFTER stash clones so
        # ev.synchronize() at next iter dispatch covers them too.
        event_tracker = None
        if stream is not None:
            event_tracker = self._acquire_ring_event("tracker")
            event_tracker.record(stream)
        self._pending_tracker_event = event_tracker
        # Direct reference (no clone) — safe in same-iter dispatch path
        # since dispatch runs in the same worker task and self._pending_dispatch
        # is cleared at end. Anchor scale-shift bug is fixed by ANCHOR_FIX
        # cold-start path above.
        self._pending_dispatch = {
            "step": global_step,
            "name_group_count": self._g_fast_name_group_count,
            "selected_mask": self._g_fast_selected_mask,
            "name_list": list(name_list),
            "offsets_cpu": list(offsets_cpu),
            "excluded_eids": set(excluded_eids),
        }

        # Same-iter dispatch in replace mode: now that _pending_
        # dispatch holds CURRENT iter's state, run dispatch immediately.
        # Pass A and Pass B both read the same m, v, p (no optimizer
        # step happens within this single worker call), so the
        # selection set is consistent → name_count matches actual writes.
        if not _bench_no_dispatch:
            self._dispatch_pending_payload(
                optimizer, current_iter_excluded=excluded_eids,
            )
            # Clear: there is no leftover stash to consume at next call.
            self._pending_dispatch = None

        _t0_end = _time.perf_counter()
        phase_launch_ms = (_t0_end - _t0_dispatch_end) * 1000.0
        phase_setup_ms = (_t0_p1 - _t0_dispatch_end) * 1000.0
        # dict.update() preserves keys written earlier in the step (e.g.
        # ``v_mapping`` recorded by update_v_mapping's lagged GPU timer).
        if not hasattr(self, "last_phase_ms") or self.last_phase_ms is None:
            self.last_phase_ms = {}
        self.last_phase_ms.update({
            "setup": phase_setup_ms,
            "pass1": phase_pass1_ms,
            "pass2": phase_pass2_ms,
            "pass3": 0.0,  # deferred to next iter dispatch
            "post": 0.0,
            "dispatch": phase_dispatch_ms,
            "launch": phase_launch_ms,
        })

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
