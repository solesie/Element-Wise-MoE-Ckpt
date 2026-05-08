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
class NonExpertDCStats:
    """Cost of one non-expert DC snapshot (LowDiff lossless path)."""
    step: int
    tensors: int
    numel: int
    bytes: int
    missing_grads: int
    compute_time_ms: float
    # CUDA event marking completion of the (non_blocking) D2H copy that
    # produced the payload. Caller forwards this to AsyncWriter.submit so
    # the worker thread waits for the copy before reading the host buffer.
    cuda_event: Optional["torch.cuda.Event"] = None


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
        self._H: Dict[str, torch.Tensor] = {}
        # Number of EMA updates applied. Used for diagnostics / bias correction.
        self._H_updates: int = 0  # bumped every refresh
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
        # proposal_pre_step inserts main_stream.wait_event(prev event)
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
        for dict_attr in ("_H", "_H_g"):
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

        # GPU stage UNUSED: per-source D2H directly to pinned host narrow
        # views. Saves ~total_numel × 4 bytes persistent GPU stage
        # (e.g. 388 MB for non-expert DC at exp_etc scale).
        _copy_ms = 0.0  # no GPU2GPU stage cat anymore

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
        # Per-source D2H directly to pinned host narrow views — no GPU
        # stage. main_event syncs main-stream prior work (grad ready);
        # all D2Hs run on the dedicated d2h_stream so they overlap with
        # main forward/tracker. cuda_event records the LAST D2H — writer
        # thread waits on it before SSD write.
        cuda_event: Optional["torch.cuda.Event"] = None
        if torch.cuda.is_available():
            from .async_writer import get_d2h_stream
            d2h_stream = get_d2h_stream()
            main_event = torch.cuda.Event()
            main_event.record()  # main: grads are ready by this point
            with torch.cuda.stream(d2h_stream):
                d2h_stream.wait_event(main_event)
                for (_n, off, n, _shape), (_n2, src) in zip(plan, entries):
                    dst = flat_cpu.narrow(0, off, n)
                    src_flat = src.reshape(-1)
                    if src_flat.dtype != torch.float32:
                        # FP32 cast first (in d2h_stream, transient buf)
                        dst.copy_(src_flat.to(torch.float32),
                                  non_blocking=True)
                    else:
                        dst.copy_(src_flat, non_blocking=True)
                cuda_event = torch.cuda.Event()
                cuda_event.record(d2h_stream)
        else:
            for (_n, off, n, _shape), (_n2, src) in zip(plan, entries):
                flat_cpu.narrow(0, off, n).copy_(
                    src.reshape(-1).to(torch.float32),
                )
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
            cuda_event=cuda_event,
        )
        return payload, stats

    # ------------------------------------------------------------------
    @torch.no_grad()
    def pre_step_snap(self) -> None:
        """accum_sq mode: NO-OP. Adam state derivation in update_delta_accum
        avoids needing a θ_before snapshot — Δθ is recomputed from m, v
        post-step using the fused-Adam formula: Δθ = -lr × m_hat / (sqrt(v_hat)
        + ε). Persistent memory cost: 0 extra beyond flat δ_g (~3 MB g=256).
        """
        return

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
            })
            b["names"].append(n)
            b["sizes"].append(int(p.numel()))
            b["m_refs"].append(m.detach().reshape(-1))
            b["v_refs"].append(v.detach().reshape(-1))

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
            group_total = sum(sizes)
            group_groups = group_total // g
            master_dg = self._g_fast_dg_flat.narrow(0, cur_off, group_groups)

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

            m_flat, m_why = _try_flat_view(m_refs)
            v_flat, v_why = _try_flat_view(v_refs)
            zc = (m_flat is not None and v_flat is not None
                  and m_flat.numel() == group_total
                  and v_flat.numel() == group_total)
            if not zc:
                all_zero_copy = False
                zc_reasons.append(
                    f"g_idx={g_idx} m_why={m_why} v_why={v_why}"
                )
            self._groups.append({
                "g_idx": g_idx,
                "names": names,
                "sizes": sizes,
                "m_refs": m_refs,
                "v_refs": v_refs,
                "m_flat": m_flat,
                "v_flat": v_flat,
                "dg_buf": master_dg,  # master-region view into dg_flat
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

        for grp in self._groups:
            dg_buf = grp["dg_buf"]
            total_a = grp["total_aligned"]
            if grp["zero_copy"]:
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
    def _aggregate_h_g(self) -> None:
        """Derive H_g (group mean) from per-element H_ema.
        Only runs when g=1 (degenerate, copies H_ema). For g>1, H_g is populated
        directly at refresh time by ``_fold_h`` (BF16) so this is a no-op."""
        g = max(1, int(self.block_quant_g))
        if g > 1:
            return  # H_g_ema is set by refresh fold; nothing to aggregate
        for n, h in self._H.items():
            h_flat = h.detach().reshape(-1).to(torch.float32)
            nt = h_flat.numel()
            nt_a = (nt // g) * g
            if nt_a == 0:
                self._H_g.pop(n, None)
                continue
            h_g = h_flat[:nt_a].reshape(-1, g).mean(dim=1)
            self._H_g[n] = h_g

    @torch.no_grad()
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
        device = next(self.model.parameters()).device
        g_blk = max(1, int(self.block_quant_g))
        arange_g = self._g_fast_arange_g
        mapper = get_mapper()
        stream = self._tracker_stream

        from contextlib import nullcontext as _nullcontext
        sm = torch.cuda.stream(stream) if stream is not None else _nullcontext()

        selected_payload: Dict[str, Tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor, torch.Tensor]] = {}
        per_param_selected: Dict[str, int] = {}

        with sm:
            for i, name in enumerate(name_list):
                s_b = int(seg_bounds_cpu[i].item())
                e_b = int(seg_bounds_cpu[i + 1].item())
                cnt_g = e_b - s_b
                # Early-skip: if this expert is in the effective exclusion
                # (T-1 fired ∪ T fired), don't bother with index_select +
                # D2H. The lossy_fc_append below would skip it anyway, but
                # this saves the wasted GPU work + pinned host transfer.
                eid_skip = self._tname_to_eid.get(name)
                if cnt_g == 0 or (
                    eid_skip is not None and eid_skip in excluded_eids
                ):
                    selected_payload[name] = (
                        torch.empty(0, dtype=torch.int32),
                        torch.empty(0, dtype=torch.float32),
                        torch.empty(0, dtype=torch.float32),
                        torch.empty(0, dtype=torch.float32),
                    )
                    per_param_selected[name] = 0
                    # NB: dg index_fill_ for T-1 selected groups under a
                    # T-fired expert is also skipped. That's fine — the
                    # whole-expert dg zero is already done by
                    # ExpertLosslessFC._fire_one (and tracker.step's
                    # defensive backup) for fired_T experts, so the dg
                    # state for these experts is uniformly 0 regardless.
                    continue
                local_g = top_idx_sorted[s_b:e_b].sub(offsets_cpu[i])
                elem_idx = (
                    local_g.unsqueeze(1) * g_blk + arange_g.unsqueeze(0)
                ).flatten()
                cnt_e = elem_idx.numel()

                p = self._name_to_param[name]
                p_flat = _fp32_param(p).detach().reshape(-1)
                p_sel = p_flat.index_select(0, elem_idx).to(torch.float32)
                m_src = (mapper.get_state(p, "exp_avg") if mapper is not None
                         else optimizer.state.get(p, {}).get("exp_avg"))
                if m_src is None:
                    m_sel = torch.zeros(cnt_e, dtype=torch.float32,
                                        device=device)
                else:
                    m_sel = m_src.detach().reshape(-1).to(torch.float32) \
                                  .index_select(0, elem_idx)
                v_src = _fp32_v_t(optimizer, p)
                if v_src is None:
                    v_sel = torch.zeros(cnt_e, dtype=torch.float32,
                                        device=device)
                else:
                    v_sel = v_src.detach().reshape(-1).to(torch.float32) \
                                  .index_select(0, elem_idx)

                d_g_buf = self._delta_g_accum.get(name)
                if d_g_buf is not None and d_g_buf.numel() > 0:
                    d_g_buf.index_fill_(0, local_g, 0.0)

                idx_cpu = elem_idx.to(torch.int32).to("cpu", non_blocking=True)
                p_cpu = p_sel.to("cpu", non_blocking=True)
                m_cpu = m_sel.to("cpu", non_blocking=True)
                v_cpu = v_sel.to("cpu", non_blocking=True)
                selected_payload[name] = (idx_cpu, p_cpu, m_cpu, v_cpu)
                per_param_selected[name] = cnt_e

        # Ensure all H2D non_blocking copies in tracker_stream are done
        # before the writer thread reads the CPU tensors.
        if stream is not None:
            stream.synchronize()

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
                    expert_id=eid, step=int(prev_step),
                    tensor_updates=updates,
                )

        self._pending_dispatch = None
        self._last_dispatch_per_param_selected = per_param_selected

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

        # === STEP 0: dispatch previous iter's pending payload ===
        # Pass THIS iter's excluded set so dispatch can skip experts that
        # fired a lossless FC at T (their fresh LFCL would be clobbered
        # otherwise — see _dispatch_pending_payload docstring).
        _t0_dispatch_start = _time.perf_counter()
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
        # Layout (set at init): flat_sg, dg_flat, h_g_fp32_flat, exc_mask
        # all share the same master-order layout (per-name views laid out
        # contiguously per BF16-Opt group). h_g_fp32_flat is rebuilt only
        # when _H_updates changes (every refresh ~40 iters); excluded
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

        for chunk_start in range(0, total_groups, chunk_size):
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

        # Update threshold for NEXT iter (GPU scalar; no host sync).
        cumsum_rev = global_hist.flip(0).cumsum(0)
        ge_k = cumsum_rev >= K_groups
        bin_idx_t = ge_k.to(torch.int8).argmax(dim=0)
        next_threshold = s_max - bin_idx_t.to(torch.float32) * bin_width
        self._g_fast_prev_threshold_t = next_threshold
        self._prev_s_max_g = float(s_max)  # host-side carry

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

        # Record tracker work completion event (so next iter dispatch can
        # synchronize and main stream race guard can wait_event).
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
