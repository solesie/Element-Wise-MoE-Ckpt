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
class NonExpertDCStats:
    """Cost of one non-expert DC snapshot (LowDiff lossless path)."""
    step: int
    tensors: int
    numel: int
    bytes: int
    missing_grads: int
    compute_time_ms: float


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
    ) -> None:
        self._random_seed_salt = int(random_seed) * 2654435761 & 0xFFFFFFFF
        self.model = model
        self.topk_ratio = float(topk_ratio)
        self._async_writer = async_writer
        # Cached GPU staging + pinned CPU buffers for batched
        # snapshot_nonexpert_dc transfer. See ``FCManager._collect_param_payload``
        # for the same pattern — pageable D2H runs at ~1.2 GB/s while pinned
        # reaches ~20 GB/s. The pinned buffer is reused each step; we wait
        # for any in-flight torch.save reading the prior payload to finish
        # before overwriting (``drain_torch``).
        self._dc_stage_gpu: Optional[torch.Tensor] = None
        self._dc_pinned_cpu: Optional[torch.Tensor] = None
        self.last_dc_sub_ms: Dict[str, float] = {}
        self.expert_only = bool(expert_only)
        self.expert_pattern = expert_pattern
        if selection_mode not in ("v_hat", "random", "hessian"):
            raise ValueError(
                f"selection_mode must be 'v_hat'|'random'|'hessian', got {selection_mode!r}."
            )
        self.selection_mode = selection_mode
        # Diagonal Hessian EMA; updated by proposal_refresh_hessian every
        # `--proposal-hessian-refresh-every` iters as:
        #     H_ema = α · H_ema + (1-α) · H_new
        # (α = args.proposal_hessian_ema_alpha, default 0.9 → effective sample
        # count 1/(1-α) = 10, which absorbs the K=1-per-refresh variance.)
        # When selection_mode == 'hessian', H_ema replaces v̂_t in the score:
        #     score = |H_diag| × δ²       (instead of v̂_t × δ²)
        # Persisted to ``cp/hessian/h_ema_iter{N}.pt`` so recovery restores
        # EMA continuity — a fresh refresh after fault would otherwise only
        # reflect one sample, temporarily degrading selection quality.
        self._H_ema: Dict[str, torch.Tensor] = {}
        # Number of EMA updates applied. Used for diagnostics / bias correction.
        self._H_ema_updates: int = 0
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

        # θ_CP cache: BF16 GPU-resident snapshot of each tracked param's value
        # at the time of its last FC or per-element state snapshot. Initialize
        # to current param value so δ = 0 at construction (no elements selected
        # until some drift accumulates — consistent with a "fresh FC" view).
        self.theta_cp: Dict[str, torch.Tensor] = {}
        for name in self._tracked_names:
            p = self._name_to_param[name]
            self.theta_cp[name] = (
                _fp32_param(p).detach().to(torch.bfloat16).clone()
            )

        # Diagnostic θ_CP for NON-expert params. Used ONLY for measuring
        # non-expert score distribution to compare against expert. Refreshed
        # whenever non-expert FC fires (on_nonexpert_fc). BF16 to save RAM.
        self.theta_fc_nonexpert: Dict[str, torch.Tensor] = {}
        for name in self._full_save_names:
            p = self._name_to_full_save_param[name]
            self.theta_fc_nonexpert[name] = (
                _fp32_param(p).detach().to(torch.bfloat16).clone()
            )

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

        # Persistent FP32 + bool scratch for chunked cat in p2/p3. Without
        # this, each chunk allocates a fresh ~134 MB FP32 buffer (and a
        # ~33 MB bool mask) — 72 alloc/free per iter at exp_3 scale, which
        # progressively fragments the CUDA caching allocator. Empirically
        # this caused p2 to drift from ~70 ms (iter 10) to ~220 ms (iter
        # 50) as the run proceeded. A single scratch sized to the largest
        # possible chunk avoids all of that churn.
        #
        # Sized for the worst case: the CHUNK_SIZE largest tracked tensors.
        # On exp_3 the per-expert tensors are uniform so this matches a
        # typical chunk; on the legacy 12L config it slightly over-allocates.
        sizes_desc = sorted(
            (self._name_to_param[n].numel() for n in self._tracked_names),
            reverse=True,
        )
        self._max_chunk_numel = sum(sizes_desc[:64])  # CHUNK = 64
        # Lazy: alloc on first step() so __init__ stays cheap and avoids
        # claiming GPU memory before the model has placed all params.
        self._cat_scratch_fp32: Optional[torch.Tensor] = None
        self._cat_scratch_bool: Optional[torch.Tensor] = None

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
    def on_expert_fc(self, tensor_name: str) -> int:
        """Refresh θ_CP for one expert tensor to the current param value.

        Called by FCManager immediately after an expert FC snapshot lands in
        the write queue. DeepSpeed's separated layout means each tracked tensor
        belongs to exactly one expert, so this is a bulk copy of the full slice.
        """
        if tensor_name not in self.theta_cp:
            return 0
        p = self._name_to_param[tensor_name]
        buf = self.theta_cp[tensor_name]
        # _fp32_param -> bf16 copy (auto-cast via copy_).
        buf.copy_(_fp32_param(p).detach())
        return int(buf.numel())

    @torch.no_grad()
    def on_state_snapshot(
        self, tensor_name: str, flat_indices: torch.Tensor
    ) -> int:
        """Scatter-update θ_CP for the selected flat indices of one tensor."""
        if tensor_name not in self.theta_cp or flat_indices.numel() == 0:
            return 0
        p = self._name_to_param[tensor_name]
        buf = self.theta_cp[tensor_name]  # bf16, GPU
        dev = buf.device
        idx = flat_indices.to(device=dev, dtype=torch.long, non_blocking=True)
        src = _fp32_param(p).detach().reshape(-1).to(torch.bfloat16)
        buf.view(-1).index_copy_(0, idx, src.index_select(0, idx))
        return int(idx.numel())

    @torch.no_grad()
    def on_nonexpert_fc(self) -> int:
        """Refresh non-expert θ_FC diagnostic cache. Called when non-expert FC fires."""
        n = 0
        for name in self._full_save_names:
            p = self._name_to_full_save_param[name]
            buf = self.theta_fc_nonexpert[name]
            buf.copy_(_fp32_param(p).detach())
            n += buf.numel()
        return n

    # ------------------------------------------------------------------
    @torch.no_grad()
    def snapshot_nonexpert_dc(
        self,
        global_step: int,
    ) -> Tuple[Dict[str, torch.Tensor], NonExpertDCStats]:
        """Clone FP32 non-expert grads to CPU (lossless dense DC path).

        Batch path: concatenates all grads into a single GPU staging buffer
        and issues ONE pinned GPU→CPU transfer. Collapses hundreds of
        per-tensor CUDA syncs (observed 100-250 ms/iter) to a single
        contiguous transfer at pinned bandwidth.
        """
        import time as _time
        _t_enum0 = _time.perf_counter()

        entries: List[Tuple[str, torch.Tensor]] = []
        missing = 0
        for name in self._full_save_names:
            p = self._name_to_full_save_param[name]
            g = _fp32_grad(p)
            if g is None:
                missing += 1
                continue
            entries.append((name, g))
        _enum_ms = (_time.perf_counter() - _t_enum0) * 1000.0

        if not entries:
            self.last_dc_sub_ms = {
                "enum_ms": _enum_ms, "copy_ms": 0.0, "wait_ms": 0.0,
                "xfer_ms": 0.0, "nentries": 0,
            }
            return {}, NonExpertDCStats(
                step=global_step, tensors=0, numel=0, bytes=0,
                missing_grads=missing, compute_time_ms=_enum_ms,
            )

        plan: List[Tuple[str, int, int, torch.Size]] = []
        offset = 0
        for name, g in entries:
            n = int(g.numel())
            plan.append((name, offset, n, g.shape))
            offset += n
        total_numel = offset
        device = entries[0][1].device

        stage = self._dc_stage_gpu
        if stage is None or stage.numel() < total_numel or stage.device != device:
            self._dc_stage_gpu = torch.empty(
                total_numel, dtype=torch.float32, device=device,
            )
            stage = self._dc_stage_gpu

        _t_copy0 = _time.perf_counter()
        for (_n, off, n, _s), (_n2, src) in zip(plan, entries):
            stage.narrow(0, off, n).copy_(src.reshape(-1))
        _copy_ms = (_time.perf_counter() - _t_copy0) * 1000.0

        _t_wait0 = _time.perf_counter()
        if self._async_writer is not None and self._dc_pinned_cpu is not None:
            self._async_writer.drain_torch()
        _wait_ms = (_time.perf_counter() - _t_wait0) * 1000.0

        pinned = self._dc_pinned_cpu
        if pinned is None or pinned.numel() < total_numel:
            self._dc_pinned_cpu = torch.empty(
                total_numel, dtype=torch.float32, pin_memory=True,
            )
            pinned = self._dc_pinned_cpu

        _t_xfer0 = _time.perf_counter()
        flat_cpu = pinned.narrow(0, 0, total_numel)
        flat_cpu.copy_(stage.narrow(0, 0, total_numel), non_blocking=False)
        _xfer_ms = (_time.perf_counter() - _t_xfer0) * 1000.0

        payload: Dict[str, torch.Tensor] = {}
        for name, off, n, shape in plan:
            payload[name] = flat_cpu.narrow(0, off, n).view(shape)
        bytes_total = int(total_numel * 4)

        self.last_dc_sub_ms = {
            "enum_ms": _enum_ms, "copy_ms": _copy_ms, "wait_ms": _wait_ms,
            "xfer_ms": _xfer_ms, "nentries": len(entries),
        }
        stats = NonExpertDCStats(
            step=global_step, tensors=len(payload), numel=total_numel,
            bytes=bytes_total, missing_grads=missing,
            compute_time_ms=_enum_ms + _copy_ms + _xfer_ms,
        )
        return payload, stats

    # ------------------------------------------------------------------
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
        """Compute score = H × δ², pick Top-K, return sparse state triples.

        Args:
            optimizer: the Adam (or variant) optimizer — Adam m/v are
                copied out for the selected elements via ``_fp32_v_t``.
            global_step: 1-indexed iteration number.
            lossless_fc_eids: experts that are performing full-lossless
                FC on this iteration (via FCManager). Their tensors are
                excluded from the top-K scoring (score forced to 0) and
                from the lossy-FC append — per CLAUDE.md § "Expert lossless
                FC가 실행대상이 되는 expert들은 제외한 전체 expert 원소에
                대해 importance score를 계산".
        """
        excluded_eids = (
            set(lossless_fc_eids) if lossless_fc_eids else set()
        )
        t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if t0 is not None:
            t0.record()

        beta2 = 0.999
        for group in optimizer.param_groups:
            if "betas" in group:
                beta2 = float(group["betas"][1])
                break
        bias_corr_2 = max(1.0 - beta2 ** max(global_step, 1), 1e-12)

        device = next(self.model.parameters()).device
        total = self._flat_total
        use_hessian = (self.selection_mode == "hessian")
        use_random = (self.selection_mode == "random")
        mapper = get_mapper()

        # Chunk size for foreach + cat batching. Trade-off:
        #   - too small → kernel-launch overhead returns (~thousands of launches)
        #   - too large → transient GPU memory (FP32 cast lists) grows
        # 64 keeps per-chunk transient ≈ 200-300 MB on the exp_3 config
        # (avg expert tensor ~525K elem) while collapsing 2304 tracked
        # tensors into 36 foreach kernel batches.
        CHUNK = 64

        # Lazy-init persistent scratch for chunked cat (allocator churn fix).
        if (
            self._cat_scratch_fp32 is None
            or self._cat_scratch_fp32.device != device
        ):
            self._cat_scratch_fp32 = torch.empty(
                self._max_chunk_numel, dtype=torch.float32, device=device,
            )
            self._cat_scratch_bool = torch.empty(
                self._max_chunk_numel, dtype=torch.bool, device=device,
            )

        # Phase timers (CPU wall-clock per pass).
        import time as _time
        _p1_t0 = _time.perf_counter()

        # Partition tracked names: included (eligible for top-K scoring)
        # vs excluded (currently undergoing lossless FC — score forced to
        # 0 to keep them out of selection without polluting bin range).
        included_names: List[str] = []
        excluded_names: List[str] = []
        for (name, _s, _e) in self._flat_offset_list:
            eid = self._tname_to_eid.get(name)
            if eid is not None and eid in excluded_eids:
                excluded_names.append(name)
            else:
                included_names.append(name)

        scores_by_name: Dict[str, torch.Tensor] = {}
        name_to_m: Dict[str, Optional[torch.Tensor]] = {}

        # GPU scalar tensors (one per chunk) for late-bundled diagnostic
        # reductions. _foreach_norm(list, ord=inf) returns one scalar per
        # input — _stack().max()/.sum() at the end batches all chunks into
        # a single sync.
        chunk_smax_list: List[torch.Tensor] = []
        chunk_ssum_list: List[torch.Tensor] = []
        chunk_maxA_list: List[torch.Tensor] = []
        chunk_maxd_list: List[torch.Tensor] = []

        # Pre-fill excluded tensors with zero scores. m is fetched here so
        # Pass 3 doesn't have to re-resolve mapper state for excluded names.
        for name in excluded_names:
            p = self._name_to_param[name]
            scores_by_name[name] = torch.zeros(
                p.numel(), dtype=torch.float32, device=device,
            )
            name_to_m[name] = (
                mapper.get_state(p, "exp_avg") if mapper is not None
                else optimizer.state.get(p, {}).get("exp_avg")
            )

        # ---- Pass 1: chunked foreach pipeline ------------------------
        # For each chunk of up to CHUNK tracked tensors:
        #   1. Build aligned FP32 flat lists (pv, fc, v_or_h)
        #   2. Pipeline (all in-place on pv_list, no extra alloc):
        #        pv ← pv − fc                  (foreach_sub_)
        #        pv ← |pv|                     (foreach_abs_)         = δ
        #        pv ← pv²                      (foreach_pow_(2))      = δ²
        #        pv ← pv × A (= v or H)        (foreach_mul_)         = score
        #        if not hessian: pv ← pv / bias (foreach_div_)
        #   3. Record per-chunk diagnostics via _foreach_norm
        # Net: ~6-9 GPU kernels per chunk vs ~13 per *tensor* in the legacy path.
        for chunk_start in range(0, len(included_names), CHUNK):
            cnames = included_names[chunk_start:chunk_start + CHUNK]
            pv_list: List[torch.Tensor] = []
            fc_list: List[torch.Tensor] = []
            for n in cnames:
                p = self._name_to_param[n]
                pv_list.append(
                    _fp32_param(p).detach().reshape(-1).to(torch.float32)
                )
                fc_list.append(
                    self.theta_cp[n].view(-1).to(torch.float32)
                )
                name_to_m[n] = (
                    mapper.get_state(p, "exp_avg") if mapper is not None
                    else optimizer.state.get(p, {}).get("exp_avg")
                )

            # delta = |pv - fc|, in-place on pv_list
            torch._foreach_sub_(pv_list, fc_list)
            del fc_list  # free chunk's FC FP32 cast (~134 MB at exp_3 scale)
            torch._foreach_abs_(pv_list)
            # diagnostic: per-chunk max |delta| (scores ≥ 0 so abs irrelevant
            # for inf-norm, but we want this BEFORE the squaring step).
            chunk_maxd_list.append(
                torch.stack(torch._foreach_norm(pv_list, ord=float('inf'))).max()
            )
            # δ² in-place
            torch._foreach_pow_(pv_list, 2)

            # Build A (= |H_diag| in hessian mode, = v_t in v_hat/random mode).
            # Note: random mode still computes v_hat × δ² scores here so the
            # importance diagnostics (max_A, score_total, per_param_score_sum)
            # remain meaningful — these scores are then OVERWRITTEN with rand
            # below for the actual selection. Mirrors the legacy two-pass
            # invariant of "diagnostics over importance, selection over rand".
            v_list: List[torch.Tensor] = []
            for n in cnames:
                if use_hessian:
                    h = self._H_ema.get(n)
                    if h is None:
                        v_list.append(torch.zeros_like(pv_list[len(v_list)]))
                    else:
                        v_list.append(h.detach().reshape(-1))
                else:
                    p = self._name_to_param[n]
                    v = _fp32_v_t(optimizer, p)
                    if v is None:
                        v_list.append(torch.zeros_like(pv_list[len(v_list)]))
                    else:
                        v_list.append(v.detach().flatten().to(torch.float32))

            if use_hessian:
                chunk_maxA_list.append(
                    torch.stack(torch._foreach_norm(v_list, ord=float('inf'))).max()
                )
            else:
                chunk_maxA_list.append(
                    torch.stack(torch._foreach_norm(v_list, ord=float('inf'))).max()
                    / bias_corr_2
                )

            # score = δ² × A (in-place into pv_list)
            torch._foreach_mul_(pv_list, v_list)
            del v_list
            if not use_hessian:
                # Adam bias correction (skipped for |H| since the GNB/Hutchinson
                # estimators are already unbiased).
                torch._foreach_div_(pv_list, bias_corr_2)

            # Per-chunk score reductions (scores ≥ 0 → ord=1 norm = sum).
            chunk_smax_list.append(
                torch.stack(torch._foreach_norm(pv_list, ord=float('inf'))).max()
            )
            chunk_ssum_list.append(
                torch.stack(torch._foreach_norm(pv_list, ord=1.0)).sum()
            )

            # Move scores into the dict (pv_list elements ARE the score tensors).
            for i, n in enumerate(cnames):
                scores_by_name[n] = pv_list[i]

        # ---- Random mode: replace scores with rand for selection -----
        # Importance scores stay in `importance_scores_for_diag` so we can
        # still report per-param score sums (used by score_diag plotting).
        if use_random:
            gen = torch.Generator(device=device)
            gen.manual_seed(
                (int(global_step) * (2 ** 20) + 0xBADCAFE) ^ self._random_seed_salt
            )
            rand_scores: Dict[str, torch.Tensor] = {}
            rand_smax_list: List[torch.Tensor] = []
            rand_ssum_list: List[torch.Tensor] = []
            for chunk_start in range(0, len(included_names), CHUNK):
                cnames = included_names[chunk_start:chunk_start + CHUNK]
                r_list: List[torch.Tensor] = [
                    torch.rand(
                        scores_by_name[n].numel(), generator=gen,
                        device=device, dtype=torch.float32,
                    )
                    for n in cnames
                ]
                rand_smax_list.append(
                    torch.stack(torch._foreach_norm(r_list, ord=float('inf'))).max()
                )
                rand_ssum_list.append(
                    torch.stack(torch._foreach_norm(r_list, ord=1.0)).sum()
                )
                for i, n in enumerate(cnames):
                    rand_scores[n] = r_list[i]
            # Excluded experts: rand uses zeros (consistent with importance pass).
            for name in excluded_names:
                rand_scores[name] = scores_by_name[name]
            importance_scores_for_diag = scores_by_name
            scores_by_name = rand_scores
            # Selection histogram is over rand → swap the bound lists.
            chunk_smax_list = rand_smax_list
            chunk_ssum_list = rand_ssum_list
            # max_A / max_delta diagnostics carry over from importance pass.
        else:
            importance_scores_for_diag = scores_by_name

        _p2_t0 = _time.perf_counter()
        phase_pass1_ms = (_p2_t0 - _p1_t0) * 1000.0

        # ---- Pass 2: histc with delayed bounds (no GPU→CPU sync) -----
        # s_min = 0 always (scores are products of non-negatives:
        # v × δ² ≥ 0; |H| × δ² ≥ 0; rand ∈ [0,1)). No reduction needed.
        # s_max: reuse the previous step's actual s_max with 1.5× safety
        # margin so newly-emerged outliers fall inside the bin range. The
        # current step's actual s_max is computed on GPU and the host-side
        # value is captured in the end-of-step diagnostic bundle, where
        # the sync overlaps the index_select / theta_cp / H2D launches.
        n_bins = 10000
        s_min = 0.0
        if self._prev_s_max is not None and self._prev_s_max > 0.0:
            s_max = self._prev_s_max * 1.5
        else:
            # Cold start: one blocking sync to seed the cache. Subsequent
            # steps are sync-free.
            if chunk_smax_list:
                s_max = max(
                    float(torch.stack(chunk_smax_list).max().item()), 1e-6,
                )
            else:
                s_max = 1e-6
        if s_max <= 0.0:
            s_max = 1e-6

        # Chunked histc: cat 64-tensor groups before histc to amortize
        # kernel launch + atomic-bin contention. Uses the persistent
        # `_cat_scratch_fp32` buffer (sized to max_chunk_numel) — torch.cat
        # with `out=` writes directly into the scratch slice, eliminating
        # the per-chunk alloc/free that fragments the CUDA allocator.
        global_hist = torch.zeros(n_bins, dtype=torch.float32, device=device)
        score_items_list: List[Tuple[str, torch.Tensor]] = list(scores_by_name.items())
        for chunk_start in range(0, len(score_items_list), CHUNK):
            chunk_scores = [
                s for _, s in score_items_list[chunk_start:chunk_start + CHUNK]
            ]
            total_n = sum(s.numel() for s in chunk_scores)
            flat = self._cat_scratch_fp32[:total_n]
            torch.cat(chunk_scores, out=flat)
            global_hist += torch.histc(flat, bins=n_bins, min=s_min, max=s_max)

        # Threshold from cumulative top-K count.
        k = max(1, int(self.topk_ratio * total))
        cumsum = global_hist.flip(0).cumsum(0)
        bin_idx_t = (cumsum >= k).nonzero(as_tuple=True)[0]
        bin_width = (s_max - s_min) / n_bins
        if bin_idx_t.numel() > 0:
            threshold = s_max - float(bin_idx_t[0].item()) * bin_width
        else:
            threshold = s_min

        _p3_t0 = _time.perf_counter()
        phase_pass2_ms = (_p3_t0 - _p2_t0) * 1000.0

        # ---- Pass 3: chunked nonzero + searchsorted segment split ----
        # Per-tensor nonzero would force ~N (=2304 in exp_3 scale) implicit
        # syncs because nonzero produces a dynamic-shape tensor and any
        # downstream op needing the count blocks the CPU. Chunked nonzero
        # collapses these to ~N/CHUNK syncs:
        #
        #   1. cat the chunk's score tensors → one flat
        #   2. flat_mask = flat >= threshold  (1 launch)
        #   3. flat_sel = flat_mask.nonzero(...)  (1 launch + 1 SHAPE SYNC)
        #   4. searchsorted(flat_sel, chunk_offsets) → segment boundaries
        #      (1 launch, no sync; flat_sel is sorted ascending so the
        #       boundary positions exactly delimit each tensor's slice)
        #   5. seg_bounds.cpu()  (1 small sync — ~CHUNK+1 longs)
        #   6. for each tensor in chunk: slice flat_sel via CPU offsets,
        #      subtract chunk_offset to get tensor-local indices, then run
        #      the same index_select / theta_cp / H2D pipeline as before.
        #
        # Net: 2 syncs per chunk × ⌈N_tensors/CHUNK⌉ chunks vs the legacy
        # 1 sync × N_tensors. At N=2304, CHUNK=64: 72 syncs vs 2304.
        selected_payload: Dict[
            str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        per_param_selected: Dict[str, int] = {}
        per_param_score_sum: Dict[str, float] = {}
        selected_bytes = 0
        k_actual = 0

        # Per-param score sum for diagnostics. _foreach_norm(ord=1.0) gives
        # sum-of-abs which equals sum for our non-negative scores. Stack +
        # one .cpu() at the end of step() bundles the sync.
        diag_scores_items = list(importance_scores_for_diag.items())
        per_param_sum_gpu: List[torch.Tensor] = []
        for chunk_start in range(0, len(diag_scores_items), CHUNK):
            chunk_sc = [
                s for _, s in diag_scores_items[chunk_start:chunk_start + CHUNK]
            ]
            per_param_sum_gpu.extend(
                torch._foreach_norm(chunk_sc, ord=1.0)
            )

        # Process p3 in chunks: chunked nonzero + searchsorted-based split.
        for chunk_start in range(0, len(score_items_list), CHUNK):
            chunk_items = score_items_list[chunk_start:chunk_start + CHUNK]
            chunk_names_p3 = [n for n, _ in chunk_items]
            chunk_scores = [s for _, s in chunk_items]
            n_in_chunk = len(chunk_names_p3)

            total_n = sum(s.numel() for s in chunk_scores)
            flat = self._cat_scratch_fp32[:total_n]
            torch.cat(chunk_scores, out=flat)
            flat_mask = self._cat_scratch_bool[:total_n]
            torch.ge(flat, threshold, out=flat_mask)
            # Single nonzero per chunk: returns sorted indices into `flat`.
            flat_sel = flat_mask.nonzero(as_tuple=True)[0]
            # Note: flat / flat_mask are views into persistent scratch — no
            # del needed; the slices will be overwritten on next chunk.

            # Cumulative chunk-local offsets: chunk_off[i] is where tensor i
            # starts in the cat'd flat. Boundary at chunk_off[i+1] means
            # tensor i's selections live in flat_sel[searchsorted(flat_sel,
            # chunk_off[i]) : searchsorted(flat_sel, chunk_off[i+1])].
            chunk_off_cpu: List[int] = [0]
            for s in chunk_scores:
                chunk_off_cpu.append(chunk_off_cpu[-1] + s.numel())
            chunk_off_t = torch.tensor(
                chunk_off_cpu, dtype=torch.long, device=device,
            )
            seg_bounds = torch.searchsorted(flat_sel, chunk_off_t)
            seg_bounds_cpu = seg_bounds.cpu()  # small sync (CHUNK+1 longs)

            # Process each tensor in chunk using already-computed boundaries.
            # No further nonzero / shape syncs are needed: cnt is derived
            # from CPU-side seg_bounds, and `flat_sel[s:e]` is a known-size
            # view whose length we got from seg_bounds_cpu.
            for i, name in enumerate(chunk_names_p3):
                s = int(seg_bounds_cpu[i].item())
                e = int(seg_bounds_cpu[i + 1].item())
                cnt = e - s
                per_param_selected[name] = cnt
                k_actual += cnt
                if cnt == 0:
                    selected_payload[name] = (
                        torch.empty(0, dtype=torch.int32),
                        torch.empty(0, dtype=torch.float32),
                        torch.empty(0, dtype=torch.float32),
                        torch.empty(0, dtype=torch.float32),
                    )
                    continue

                # Tensor-local indices: subtract this tensor's offset within
                # the cat'd chunk. flat_sel[s:e] is a contiguous slice (no
                # launch); the subtract is one elementwise op.
                offset_i = chunk_off_cpu[i]
                local_sel = flat_sel[s:e].sub(offset_i)

                p = self._name_to_param[name]
                p_flat = _fp32_param(p).detach().reshape(-1)
                p_sel = p_flat.index_select(0, local_sel).to(torch.float32)
                m_src = name_to_m[name]
                if m_src is None:
                    m_sel_gpu = torch.zeros(cnt, dtype=torch.float32, device=device)
                else:
                    m_flat = m_src.detach().reshape(-1).to(torch.float32)
                    m_sel_gpu = m_flat.index_select(0, local_sel)
                v_src = _fp32_v_t(optimizer, p)
                if v_src is None:
                    v_sel_gpu = torch.zeros(cnt, dtype=torch.float32, device=device)
                else:
                    v_flat = v_src.detach().flatten().to(torch.float32)
                    v_sel_gpu = v_flat.index_select(0, local_sel)

                # Update θ_CP for selected positions (GPU-only, no PCIe).
                self.theta_cp[name].view(-1).index_copy_(
                    0, local_sel, p_sel.to(torch.bfloat16)
                )

                idx_cpu = local_sel.to(torch.int32).to("cpu", non_blocking=True)
                p_cpu = p_sel.to("cpu", non_blocking=True)
                m_cpu = m_sel_gpu.to("cpu", non_blocking=True)
                v_cpu = v_sel_gpu.to("cpu", non_blocking=True)
                selected_payload[name] = (idx_cpu, p_cpu, m_cpu, v_cpu)
                selected_bytes += (
                    idx_cpu.numel() * idx_cpu.element_size()
                    + p_cpu.numel() * p_cpu.element_size()
                    + m_cpu.numel() * m_cpu.element_size()
                    + v_cpu.numel() * v_cpu.element_size()
                )

        # ---- End-of-step bundled diagnostic sync ---------------------
        # Aggregate all per-chunk reductions + per-param sums into one
        # stacked tensor, single .cpu() transfer. By this point all p3 GPU
        # work has been launched, so the sync overlaps with the kernels
        # still in flight (index_select, index_copy_, H2D) — net cost ~0.
        if chunk_smax_list:
            cur_s_max_t = torch.stack(chunk_smax_list).max()
            cur_s_sum_t = torch.stack(chunk_ssum_list).sum()
            cur_max_A_t = torch.stack(chunk_maxA_list).max().clamp_min(1e-12)
            cur_max_d_t = torch.stack(chunk_maxd_list).max()
        else:
            zero = torch.zeros((), device=device)
            cur_s_max_t = zero
            cur_s_sum_t = zero
            cur_max_A_t = torch.full((), 1e-12, device=device)
            cur_max_d_t = zero

        if per_param_sum_gpu:
            per_param_sum_stack = torch.stack(per_param_sum_gpu)
        else:
            per_param_sum_stack = torch.zeros(0, device=device)

        # One bundled sync.
        diag_bundle = torch.cat([
            torch.stack([cur_s_max_t, cur_s_sum_t, cur_max_A_t, cur_max_d_t]),
            per_param_sum_stack,
        ]).cpu()
        s_max_actual = float(diag_bundle[0].item())
        score_total = float(diag_bundle[1].item())
        max_A = float(diag_bundle[2].item())
        max_delta = float(diag_bundle[3].item())
        score_mean = score_total / max(total, 1)
        score_max = s_max_actual

        # Match per-param sums to names (importance order, includes excluded zeros).
        for i, (name, _) in enumerate(diag_scores_items):
            per_param_score_sum[name] = float(diag_bundle[4 + i].item())
        # Excluded names whose scores are zeros are already in
        # diag_scores_items (importance pass populated zeros for them in
        # scores_by_name before random swap), so the loop above covers them.
        # Random mode swaps importance_scores_for_diag = original scores_by_name
        # which still contains the excluded zeros. Either way, every name
        # is represented.
        for name in self._tracked_names:
            per_param_score_sum.setdefault(name, 0.0)

        # Cache actual s_max for next step's delayed bound.
        self._prev_s_max = s_max_actual

        # Free scoring tensors (5+ GB at exp_3 scale).
        scores_by_name.clear()
        if use_random:
            importance_scores_for_diag.clear()

        if t1 is not None:
            t1.record()
            torch.cuda.synchronize()
            compute_ms = float(t0.elapsed_time(t1))
        else:
            compute_ms = 0.0

        _p3_end = _time.perf_counter()
        phase_pass3_ms = (_p3_end - _p3_t0) * 1000.0
        # Save most-recent phase breakdown for cp_hooks diagnostic print.
        self.last_phase_ms = {
            "pass1": phase_pass1_ms,
            "pass2": phase_pass2_ms,
            "pass3": phase_pass3_ms,
        }

        # Push selected element state into the ExpertState append log.
        # Collect each expert's tensor updates into one lossy_fc_append
        # call per CLAUDE.md spec. Experts already doing lossless FC this
        # iter (in excluded_eids) are skipped — their full state is being
        # captured by the lossless FC path. The sync above guarantees
        # non_blocking H2D copies have landed before we read these CPU
        # tensors.
        if self._expert_state is not None:
            per_expert_batch: Dict[Tuple[int, int],
                                   List[Tuple[str, torch.Tensor,
                                              torch.Tensor, torch.Tensor,
                                              torch.Tensor]]] = {}
            for name, entry in selected_payload.items():
                idx_cpu, p_cpu, m_cpu, v_cpu = entry
                eid = self._tname_to_eid.get(name)
                if eid is None or eid in excluded_eids:
                    continue
                per_expert_batch.setdefault(eid, []).append(
                    (name, idx_cpu, p_cpu, m_cpu, v_cpu)
                )
            for eid, updates in per_expert_batch.items():
                self._expert_state.lossy_fc_append(
                    expert_id=eid, step=int(global_step),
                    tensor_updates=updates,
                )

        stats = TrackerStepStats(
            step=global_step,
            total_tracked=total,
            k=k_actual,
            topk_ratio=k_actual / max(total, 1),
            max_A=max_A,
            max_delta=max_delta,
            score_mean=score_mean,
            score_max=score_max,
            selected_bytes_sparse=selected_bytes,
            full_save_bytes=self.full_save_bytes,
            full_save_numel=self.total_full_save,
            compute_time_ms=compute_ms,
            per_param_selected=per_param_selected,
            per_param_score_sum=per_param_score_sum,
        )
        return stats, selected_payload

    # ------------------------------------------------------------------
    @torch.no_grad()
    def measure_score_distribution(
        self,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        num_bins: int = 1024,
        topk_ratio_for_membership: Optional[float] = None,
    ) -> Dict:
        """Diagnostic: compute score = v̂_t × δ² for BOTH expert and non-expert
        params. Returns histograms + per-layer aggregates + overlap percentile.

        Expensive (~hundreds of ms). Call sparingly (e.g., every 100-200 iter).

        Returns dict::
            {
              "step": int,
              "expert": {
                "total_numel": int,
                "min": float, "max": float, "mean": float, "p50": float,
                "p90": float, "p99": float, "p99.9": float,
                "hist_edges": [...], "hist_counts": [...],
                "per_layer": {layer_idx: {"sum": float, "numel": int,
                                          "expert_count": int, "avg_per_expert": float}}
              },
              "nonexpert": {same structure, no per_layer aggregation},
              "overlap": {
                "nonexpert_p999_value": float,     # 99.9-percentile of non-expert score
                "expert_max_value": float,          # max of expert score
                "expert_elements_above_ne_p999": int,   # how many expert elems > ne-p99.9
                "expert_elements_above_ne_p999_pct": float,
              }
            }
        """
        beta2 = 0.999
        for group in optimizer.param_groups:
            if "betas" in group:
                beta2 = float(group["betas"][1])
                break
        bias_corr_2 = max(1.0 - beta2 ** max(global_step, 1), 1e-12)

        def _score_for_params(
            names: List[str], name_to_p: Dict[str, torch.nn.Parameter],
            theta_cp_dict: Dict[str, torch.Tensor],
        ) -> Tuple[torch.Tensor, Dict[int, Tuple[float, int]],
                   Dict[Tuple[int, int], List[torch.Tensor]], Dict[str, float]]:
            """Compute scores for given param set. Returns
            (flat_scores_cpu, per_layer_stats, per_expert_chunks, diag_stats).

            diag_stats includes max/mean of |δ|, |v̂_t|, |param|, and ΔL proxy
            (0.5 × Σ score), for detecting anomalies in score scale.
            """
            chunks: List[torch.Tensor] = []
            per_layer_sum: Dict[int, float] = {}
            per_layer_numel: Dict[int, int] = {}
            per_expert_chunks: Dict[Tuple[int, int], List[torch.Tensor]] = {}
            # Diag aggregates (max / mean across all tracked tensors)
            d_max = d_sum = d_count = 0.0
            v_max = v_sum = 0.0
            p_abs_max = p_abs_sum = 0.0
            score_sum = 0.0
            import re as _re
            layer_re = _re.compile(r"\.layers\.(\d+)\.")
            expert_re = _re.compile(r"deepspeed_experts\.(\d+)\.")
            for name in names:
                p = name_to_p[name]
                v = _fp32_v_t(optimizer, p)
                if v is None:
                    continue
                p_fp32 = _fp32_param(p).detach().flatten()
                fc_fp32 = theta_cp_dict[name].view(-1).float()
                delta = (p_fp32 - fc_fp32).abs()
                vhat = v.detach().flatten().div(bias_corr_2)
                score = vhat * delta * delta
                score_cpu = score.to("cpu")
                chunks.append(score_cpu)
                # Diag accumulation
                numel = delta.numel()
                d_max = max(d_max, float(delta.max().item()))
                d_sum += float(delta.sum().item())
                d_count += numel
                v_max = max(v_max, float(vhat.max().item()))
                v_sum += float(vhat.sum().item())
                p_abs = p_fp32.abs()
                p_abs_max = max(p_abs_max, float(p_abs.max().item()))
                p_abs_sum += float(p_abs.sum().item())
                score_sum += float(score.sum().item())
                # per-layer aggregation
                lm = layer_re.search(name)
                layer_idx = int(lm.group(1)) if lm else -1
                per_layer_sum[layer_idx] = per_layer_sum.get(layer_idx, 0.0) + float(score.sum().item())
                per_layer_numel[layer_idx] = per_layer_numel.get(layer_idx, 0) + int(score.numel())
                # per-(layer, expert) accumulation — only for expert tensors
                em = expert_re.search(name)
                if em is not None and lm is not None:
                    eid = (int(lm.group(1)), int(em.group(1)))
                    per_expert_chunks.setdefault(eid, []).append(score_cpu)
            diag_stats = {
                "delta_max": d_max,
                "delta_mean": d_sum / max(d_count, 1),
                "vhat_max": v_max,
                "vhat_mean": v_sum / max(d_count, 1),
                "param_abs_max": p_abs_max,
                "param_abs_mean": p_abs_sum / max(d_count, 1),
                "score_sum": score_sum,
                "delta_L_proxy": 0.5 * score_sum,  # ΔL ≈ 0.5 × Σ v̂ × δ²
                "numel_total": int(d_count),
            }
            if not chunks:
                return torch.empty(0, dtype=torch.float32), {}, {}, diag_stats
            flat = torch.cat(chunks)
            per_layer = {
                k: (per_layer_sum[k], per_layer_numel[k])
                for k in per_layer_sum
            }
            return flat, per_layer, per_expert_chunks, diag_stats

        def _summary(flat: torch.Tensor, num_bins: int = 1024) -> Dict:
            if flat.numel() == 0:
                return {"total_numel": 0, "min": 0.0, "max": 0.0, "mean": 0.0,
                        "p50": 0.0, "p90": 0.0, "p99": 0.0,
                        "p99.5": 0.0, "p99.9": 0.0,
                        "hist_edges": [], "hist_counts": []}
            fmin = float(flat.min().item())
            fmax = float(flat.max().item())
            fmean = float(flat.mean().item())
            # Percentiles via torch.quantile on a sample if too large.
            sample = flat if flat.numel() < 1_000_000 else flat[torch.randperm(flat.numel())[:1_000_000]]
            qs = torch.quantile(sample, torch.tensor([0.5, 0.9, 0.99, 0.995, 0.999]))
            # Histogram on linear bins from fmin to fmax. Previously used a
            # 1e-12 guard against fmax==fmin which dominated when scores
            # are tiny (~1e-21), forcing all values into bin 0. Use a
            # relative epsilon (or 1e-30 floor) so the histogram resolves
            # the actual data range.
            hi = fmax if fmax > fmin else fmin + max(abs(fmin) * 1e-6, 1e-30)
            edges = torch.linspace(fmin, hi, num_bins + 1)
            hist = torch.histc(flat, bins=num_bins, min=fmin, max=hi)
            return {
                "total_numel": int(flat.numel()),
                "min": fmin, "max": fmax, "mean": fmean,
                "p50": float(qs[0].item()), "p90": float(qs[1].item()),
                "p99": float(qs[2].item()),
                "p99.5": float(qs[3].item()), "p99.9": float(qs[4].item()),
                "hist_edges": edges.tolist(),
                "hist_counts": hist.tolist(),
            }

        # ==========================================================
        # (A) Expert score = v̂_t × δ²  —  expert-vs-expert 비교용
        # ==========================================================
        exp_flat, exp_per_layer, exp_per_expert_chunks, exp_diag = _score_for_params(
            self._tracked_names, self._name_to_param, self.theta_cp,
        )
        exp_summary = _summary(exp_flat, num_bins)
        exp_summary["diag"] = exp_diag  # δ, v̂, param, ΔL proxy
        import re as _re
        exp_idx_re = _re.compile(r"deepspeed_experts\.(\d+)\.")
        layer_re2 = _re.compile(r"\.layers\.(\d+)\.")
        per_layer_experts: Dict[int, set] = {}
        for name in self._tracked_names:
            lm = layer_re2.search(name)
            em = exp_idx_re.search(name)
            if lm and em:
                per_layer_experts.setdefault(int(lm.group(1)), set()).add(int(em.group(1)))
        exp_per_layer_out = {}
        for layer_idx, (s, n) in exp_per_layer.items():
            ncount = len(per_layer_experts.get(layer_idx, set())) or 1
            exp_per_layer_out[layer_idx] = {
                "sum": s, "numel": n,
                "expert_count": ncount,
                "avg_per_expert": s / ncount,
            }
        exp_summary["per_layer"] = exp_per_layer_out

        # Per-(layer, expert) summary for expert-vs-expert comparison.
        per_expert_out: Dict[str, Dict[str, float]] = {}
        per_expert_flat: Dict[str, torch.Tensor] = {}
        for (L, E), parts in exp_per_expert_chunks.items():
            ftens = torch.cat(parts) if len(parts) > 1 else parts[0]
            n = int(ftens.numel())
            if n == 0:
                continue
            per_expert_flat[f"L{L}_E{E}"] = ftens
            qs = torch.quantile(ftens, torch.tensor([0.5, 0.9, 0.99, 0.999]))
            per_expert_out[f"L{L}_E{E}"] = {
                "layer": int(L), "expert": int(E), "numel": n,
                "sum": float(ftens.sum().item()),
                "mean": float(ftens.mean().item()),
                "max": float(ftens.max().item()),
                "p50": float(qs[0].item()), "p90": float(qs[1].item()),
                "p99": float(qs[2].item()), "p99.9": float(qs[3].item()),
            }
        exp_summary["per_expert"] = per_expert_out

        # ==========================================================
        # (B) Top-K membership — expert score 상위 k%에 포함된 비율
        #     layer별, expert별로 집계
        # ==========================================================
        k_ratio = float(topk_ratio_for_membership or self.topk_ratio)
        topk_membership_out: Dict = {"topk_ratio": k_ratio}
        if exp_flat.numel() > 0:
            k = max(1, int(k_ratio * exp_flat.numel()))
            # Global threshold at top-k (descending): value at rank k
            sorted_scores, _ = exp_flat.sort(descending=True)
            threshold = float(sorted_scores[k - 1].item())
            topk_membership_out["threshold"] = threshold
            topk_membership_out["k"] = int(k)
            # Per-(layer, expert) inclusion count/fraction
            per_expert_topk: Dict[str, Dict[str, float]] = {}
            per_layer_topk_count: Dict[int, int] = {}
            per_layer_topk_numel: Dict[int, int] = {}
            for key, ftens in per_expert_flat.items():
                n_in = int((ftens >= threshold).sum().item())
                n_total = int(ftens.numel())
                per_expert_topk[key] = {
                    "count": n_in,
                    "numel": n_total,
                    "frac": n_in / max(n_total, 1),
                }
                L = per_expert_out[key]["layer"]
                per_layer_topk_count[L] = per_layer_topk_count.get(L, 0) + n_in
                per_layer_topk_numel[L] = per_layer_topk_numel.get(L, 0) + n_total
            topk_membership_out["per_expert"] = per_expert_topk
            topk_membership_out["per_layer"] = {
                L: {
                    "count": per_layer_topk_count[L],
                    "numel": per_layer_topk_numel[L],
                    "frac": per_layer_topk_count[L] / max(per_layer_topk_numel[L], 1),
                }
                for L in per_layer_topk_count
            }
        exp_summary["topk_membership"] = topk_membership_out

        # ==========================================================
        # (C) v_t percentile distributions  —  non-expert vs expert CDF 용
        #     각 카테고리의 v_t (FP32 Adam 2nd moment) 원본 분포를 측정.
        #     v̂_t(bias-correction)는 단조 스케일이므로 v_t 분위만 저장해도 동치.
        # ==========================================================
        # Dense percentile grid tailored for long-tail CDF visualization:
        # 0, 1, 2, ..., 98   (per-1%)
        # 99.0, 99.1, ..., 99.9  (per-0.1%)
        # 99.99
        pct_list = (
            [i * 1.0 for i in range(0, 99)]
            + [99.0 + i * 0.1 for i in range(0, 10)]
            + [99.99]
        )
        pct_keys = [f"p{p:.2f}" for p in pct_list]

        def _collect_vt(names: List[str], name_to_p: Dict[str, torch.nn.Parameter]) -> torch.Tensor:
            """Concat v_t across the given param set (CPU)."""
            chunks: List[torch.Tensor] = []
            for name in names:
                p = name_to_p[name]
                v = _fp32_v_t(optimizer, p)
                if v is None:
                    continue
                chunks.append(v.detach().flatten().to("cpu"))
            return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)

        def _percentiles(flat: torch.Tensor) -> Dict[str, float]:
            if flat.numel() == 0:
                return {k: 0.0 for k in pct_keys}
            # torch.quantile memory-limits at ~16M elems; sample if larger.
            sample = (
                flat if flat.numel() < 1_000_000
                else flat[torch.randperm(flat.numel())[:1_000_000]]
            )
            qs = torch.quantile(sample, torch.tensor([p / 100.0 for p in pct_list]))
            return {pct_keys[i]: float(qs[i].item()) for i in range(len(pct_list))}

        exp_vt_flat = _collect_vt(self._tracked_names, self._name_to_param)
        ne_vt_flat = _collect_vt(self._full_save_names, self._name_to_full_save_param)
        expert_vt_pcts = _percentiles(exp_vt_flat)
        nonexpert_vt_pcts = _percentiles(ne_vt_flat)

        # ---------------- |H| and score_H = |H|·δ² percentiles ----------------
        # Only populated when H cache is active (selection_mode='hessian'
        # or refresh_hessian called periodically). Gives user a per-iter CDF
        # of raw |H_diag| and the Hessian-based selection score.
        h_cache = getattr(self, "_H_ema", {}) or {}
        h_percentiles: Dict[str, float] = {}
        score_h_percentiles: Dict[str, float] = {}
        h_numel = 0
        score_h_numel = 0
        if h_cache:
            h_chunks: List[torch.Tensor] = []
            sh_chunks: List[torch.Tensor] = []
            for name in self._tracked_names:
                h = h_cache.get(name)
                if h is None:
                    continue
                h_flat = h.detach().flatten().to("cpu")
                h_chunks.append(h_flat)
                # score_H = |H| · δ²
                p = self._name_to_param[name]
                fc = self.theta_cp.get(name)
                if fc is not None:
                    p_fp32 = _fp32_param(p).detach().flatten()
                    delta_sq = (p_fp32 - fc.view(-1).float()).pow_(2).to("cpu")
                    sh_chunks.append(h_flat * delta_sq)
            h_flat_all = torch.cat(h_chunks) if h_chunks else torch.empty(0, dtype=torch.float32)
            sh_flat_all = torch.cat(sh_chunks) if sh_chunks else torch.empty(0, dtype=torch.float32)
            h_percentiles = _percentiles(h_flat_all)
            score_h_percentiles = _percentiles(sh_flat_all)
            h_numel = int(h_flat_all.numel())
            score_h_numel = int(sh_flat_all.numel())

        # ==========================================================
        # (D) Legacy non-expert score (v̂×δ², for backward compat only)
        # ==========================================================
        ne_flat, _, _, ne_diag = _score_for_params(
            self._full_save_names, self._name_to_full_save_param, self.theta_fc_nonexpert,
        )
        ne_summary = _summary(ne_flat, num_bins)
        ne_summary["diag"] = ne_diag

        ne_p999 = ne_summary["p99.9"]
        if exp_flat.numel() > 0 and ne_p999 > 0:
            n_above = int((exp_flat > ne_p999).sum().item())
            pct_above = n_above / exp_flat.numel() * 100.0
        else:
            n_above = 0
            pct_above = 0.0

        return {
            "step": int(global_step),
            "expert": exp_summary,           # (A) expert score + (B) top-K membership
            "nonexpert": ne_summary,         # legacy (v̂×δ² score for non-expert) — compat
            "vt_percentiles": {              # (C) v_t CDF for cross-category comparison
                "percentile_labels": pct_list,
                "expert": {
                    "numel": int(exp_vt_flat.numel()),
                    "values": expert_vt_pcts,
                },
                "nonexpert": {
                    "numel": int(ne_vt_flat.numel()),
                    "values": nonexpert_vt_pcts,
                },
            },
            # (E) |H| and score_H = |H|·δ² CDFs — populated when H
            # cache is active (selection_mode='hessian' or periodic refresh).
            "h_percentiles": {
                "percentile_labels": pct_list,
                "H_diag": {"numel": h_numel, "values": h_percentiles},
                "score_h": {"numel": score_h_numel, "values": score_h_percentiles},
            },
            "overlap": {
                "nonexpert_p999_value": ne_p999,
                "expert_max_value": exp_summary["max"],
                "expert_elements_above_ne_p999": n_above,
                "expert_elements_above_ne_p999_pct": pct_above,
            },
        }
