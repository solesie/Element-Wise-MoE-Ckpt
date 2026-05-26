"""DeepSpeed FP32 master / Adam state mapper.

Unifies access to FP32 master weights and Adam moments across the three
DeepSpeed wrappers we care about:

  - ``FP16_Optimizer`` (ZeRO-0 + FP16): holds ``fp16_groups`` (model params)
    and ``fp32_groups_flat`` (one flat FP32 master tensor per param_group).
    Inner torch optimizer's ``param_groups[i]['params'] = [fp32_groups_flat[i]]``.
  - ``BF16_Optimizer`` (ZeRO-0 + BF16): same shape with ``bf16_groups`` /
    ``fp32_groups_flat_partition``.
  - ``DeepSpeedZeroOptimizer`` (ZeRO-1/2): each model param carries
    ``_hp_mapping`` so DeepSpeed's ``safe_get_full_*`` APIs work directly.

ZeRO-3 is intentionally unsupported (init_proposal raises).

Usage::

    mapper = Fp32StateMapper(engine_optimizer, inner_torch_optimizer, model)
    fp32_param = mapper.get_param(p)            # FP32 view of master weight
    v = mapper.get_state(p, "exp_avg_sq")       # FP32 Adam second moment
    mapper.set_param(p, fp32_value)             # FP32 writeback to master
    mapper.set_state(p, fp32_value, "exp_avg")  # FP32 writeback to Adam state

The mapper does not synchronise FP32 → FP16/BF16 model params; DeepSpeed
takes care of that on its next ``engine.step()`` (or the user can call
``engine.optimizer._update_model_bit16_weights`` after recovery).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from deepspeed.utils import (
        safe_get_full_fp32_param,
        safe_get_full_optimizer_state,
        safe_set_full_fp32_param,
        safe_set_full_optimizer_state,
    )
    _HAS_SAFE_API = True
except ImportError:
    _HAS_SAFE_API = False

    def safe_get_full_fp32_param(p):  # type: ignore
        return None

    def safe_get_full_optimizer_state(p, k):  # type: ignore
        return None

    def safe_set_full_fp32_param(p, v):  # type: ignore
        pass

    def safe_set_full_optimizer_state(p, v, k):  # type: ignore
        pass


# Per-param coordinate inside the flat FP32 master tensor.
ParamCoord = Tuple[int, int, int]  # (group_idx, offset, numel)


class Fp32StateMapper:
    def __init__(
        self,
        engine_optimizer: object,
        inner_torch_optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
    ) -> None:
        self._engine_opt = engine_optimizer
        # Drill to the bare torch.optim.Optimizer (its .state dict is the
        # only place Adam moments actually live). Wrappers like Megatron's
        # MixedPrecisionOptimizer forward param_groups but not state.
        self._inner_opt = self._drill_to_torch_optimizer(inner_torch_optimizer)
        self._coords: Dict[int, ParamCoord] = {}
        self._fp32_flat: Optional[list] = None
        self._mode: str = "unknown"
        self._build()

    @staticmethod
    def _drill_to_torch_optimizer(o: object) -> torch.optim.Optimizer:
        """Walk .optimizer chain until we hit a torch.optim.Optimizer subclass
        whose ``state`` dict is keyed by the FP32 master tensors AdamW writes to.
        """
        cur = o
        for _ in range(8):
            if isinstance(cur, torch.optim.Optimizer) and not hasattr(cur, "fp32_groups") \
                    and not hasattr(cur, "fp32_from_float16_groups"):
                return cur
            inner = getattr(cur, "optimizer", None)
            if inner is None or inner is cur:
                break
            cur = inner
        return cur if isinstance(cur, torch.optim.Optimizer) else o  # type: ignore[return-value]

    def _build(self) -> None:
        eo = self._engine_opt

        # Try unwrapping one layer if the outer wrapper hides the master groups
        # but its .optimizer attribute exposes them (DeepSpeed FP16_UnfusedOptimizer
        # or DeepSpeed wrapping Megatron's Float16OptimizerWithFloat16Params).
        candidates = [eo]
        if hasattr(eo, "optimizer"):
            candidates.append(eo.optimizer)
            inner = eo.optimizer
            if hasattr(inner, "optimizer"):
                candidates.append(inner.optimizer)

        for c in candidates:
            if self._try_megatron_float16(c):
                return
            if self._try_unfused_fp16(c):
                return
            if self._try_zero12(c):
                return
            if self._try_fp16_zero0(c):
                return
            if self._try_bf16_zero0(c):
                return

        self._mode = "unrecognised"

    def _try_megatron_float16(self, c: object) -> bool:
        if hasattr(c, "fp32_from_float16_groups") and hasattr(c, "float16_groups"):
            self._mode = "megatron_float16"
            self._direct_master: Dict[int, torch.Tensor] = {}
            for f16_group, fp32_group in zip(c.float16_groups, c.fp32_from_float16_groups):
                for f16_p, fp32_p in zip(f16_group, fp32_group):
                    self._direct_master[id(f16_p)] = fp32_p
            return True
        return False

    def _try_unfused_fp16(self, c: object) -> bool:
        # DeepSpeed FP16_UnfusedOptimizer: 1:1 direct master, same shape per
        # param. Adam state is on whatever bare torch optimizer is at the
        # bottom of the wrapper chain (set in __init__ via _drill_to_torch_optimizer).
        if hasattr(c, "fp32_groups") and hasattr(c, "fp16_groups"):
            self._mode = "unfused_fp16"
            self._direct_master = {}
            for f16_group, fp32_group in zip(c.fp16_groups, c.fp32_groups):
                for f16_p, fp32_p in zip(f16_group, fp32_group):
                    self._direct_master[id(f16_p)] = fp32_p
            return True
        return False

    def _try_zero12(self, c: object) -> bool:
        if hasattr(c, "single_partition_of_fp32_groups"):
            self._fp32_flat = c.single_partition_of_fp32_groups
            self._mode = "zero12"
            self._coords = self._index_groups(c, "bit16_groups")
            return True
        return False

    def _try_fp16_zero0(self, c: object) -> bool:
        if hasattr(c, "fp32_groups_flat"):
            self._fp32_flat = c.fp32_groups_flat
            self._mode = "fp16_zero0"
            self._coords = self._index_groups(c, "fp16_groups")
            return True
        return False

    def _try_bf16_zero0(self, c: object) -> bool:
        if hasattr(c, "fp32_groups_flat_partition"):
            self._fp32_flat = c.fp32_groups_flat_partition
            self._mode = "bf16_zero0"
            self._coords = self._index_groups(c, "bf16_groups")
            return True
        return False
        # ZeRO-1/2: master is flat per group with _hp_mapping on each model param.
        if hasattr(eo, "single_partition_of_fp32_groups"):
            self._fp32_flat = eo.single_partition_of_fp32_groups
            self._mode = "zero12"
            self._coords = self._index_groups(eo, "bit16_groups")
            return
        # ZeRO-0 + FP16 (DeepSpeed FP16_Optimizer): master is flat per group.
        if hasattr(eo, "fp32_groups_flat"):
            self._fp32_flat = eo.fp32_groups_flat
            self._mode = "fp16_zero0"
            self._coords = self._index_groups(eo, "fp16_groups")
            return
        # ZeRO-0 + BF16 (DeepSpeed BF16_Optimizer): master is flat per group.
        if hasattr(eo, "fp32_groups_flat_partition"):
            self._fp32_flat = eo.fp32_groups_flat_partition
            self._mode = "bf16_zero0"
            self._coords = self._index_groups(eo, "bf16_groups")
            return
        self._mode = "unrecognised"

    @staticmethod
    def _index_groups(engine_opt: object, attr: str) -> Dict[int, ParamCoord]:
        """Walk engine_opt.<attr> (list of lists of model params) and record
        (group_idx, offset, numel) for each model param."""
        out: Dict[int, ParamCoord] = {}
        groups = getattr(engine_opt, attr, None)
        if groups is None:
            return out
        for g_idx, group in enumerate(groups):
            offset = 0
            for p in group:
                out[id(p)] = (g_idx, offset, p.numel())
                offset += p.numel()
        return out

    # ----- read paths --------------------------------------------------
    def get_param(self, p: torch.nn.Parameter) -> torch.Tensor:
        """FP32 master view of ``p`` (gathered when ZeRO partitions). Caller
        should clone before storing if persistence is needed."""
        if self._mode in ("megatron_float16", "unfused_fp16"):
            master = self._direct_master.get(id(p))
            if master is not None:
                return master
            return p.data.to(torch.float32)
        if _HAS_SAFE_API:
            full = safe_get_full_fp32_param(p)
            if full is not None:
                return full
        coord = self._coords.get(id(p))
        if coord is None:
            return p.data.to(torch.float32)
        g, off, n = coord
        return self._fp32_flat[g].narrow(0, off, n).view(p.shape)

    def get_state(
        self, p: torch.nn.Parameter, key: str,
    ) -> Optional[torch.Tensor]:
        """FP32 Adam state reshaped to ``p.shape``; ``None`` if uninitialised."""
        if self._mode in ("megatron_float16", "unfused_fp16"):
            master = self._direct_master.get(id(p))
            if master is None:
                return None
            st = self._inner_opt.state.get(master, {})
            return st.get(key)
        if _HAS_SAFE_API:
            full = safe_get_full_optimizer_state(p, key)
            if full is not None:
                return full
        coord = self._coords.get(id(p))
        if coord is None:
            st = self._inner_opt.state.get(p, {})
            return st.get(key)
        g, off, n = coord
        master_flat = self._fp32_flat[g]
        st = self._inner_opt.state.get(master_flat, {})
        v = st.get(key)
        if v is None:
            return None
        return v.narrow(0, off, n).view(p.shape)

    # ----- write paths -------------------------------------------------
    @torch.no_grad()
    def set_param(self, p: torch.nn.Parameter, value: torch.Tensor) -> None:
        if self._mode in ("megatron_float16", "unfused_fp16"):
            master = self._direct_master.get(id(p))
            if master is not None:
                master.data.copy_(value.to(device=master.device, dtype=master.dtype))
            p.data.copy_(value.to(device=p.device, dtype=p.dtype))
            return
        # safe_set_full_fp32_param is a no-op silently when neither _hp_mapping
        # (ZeRO-1/2) nor ds_id (ZeRO-3) is present on the model param, so we
        # must gate on those markers before trusting it.
        if _HAS_SAFE_API and (hasattr(p, "_hp_mapping") or hasattr(p, "ds_id")):
            try:
                safe_set_full_fp32_param(p, value)
                return
            except Exception:
                pass
        coord = self._coords.get(id(p))
        if coord is not None:
            g, off, n = coord
            # DeepSpeed's flat FP32 master has requires_grad=True; use .data
            # to avoid autograd's leaf-view in-place check.
            flat_slice = self._fp32_flat[g].data.narrow(0, off, n)
            flat_slice.copy_(value.flatten().to(dtype=flat_slice.dtype, device=flat_slice.device))
        # Always update the model param view so the next forward sees the
        # loaded weights (DeepSpeed normally does master→bit16 sync inside
        # step(); replay-only mode never calls step, so do it here).
        p.data.copy_(value.view(p.shape).to(device=p.device, dtype=p.dtype))

    @torch.no_grad()
    def set_state(
        self, p: torch.nn.Parameter, value: torch.Tensor, key: str,
    ) -> None:
        if self._mode in ("megatron_float16", "unfused_fp16"):
            master = self._direct_master.get(id(p))
            if master is None:
                return
            st = self._inner_opt.state.setdefault(master, {})
            slot = st.get(key)
            if slot is None:
                slot = torch.zeros_like(master)
                st[key] = slot
            slot.copy_(value.to(device=slot.device, dtype=slot.dtype))
            return
        if _HAS_SAFE_API and (hasattr(p, "_hp_mapping") or hasattr(p, "ds_id")):
            try:
                safe_set_full_optimizer_state(p, value, key)
                return
            except Exception:
                pass
        coord = self._coords.get(id(p))
        if coord is None:
            st = self._inner_opt.state.setdefault(p, {})
            slot = st.get(key)
            if slot is None:
                slot = torch.zeros_like(p, dtype=torch.float32)
                st[key] = slot
            slot.copy_(value.to(device=slot.device, dtype=slot.dtype))
            return
        g, off, n = coord
        master_flat = self._fp32_flat[g]
        st = self._inner_opt.state.setdefault(master_flat, {})
        slot = st.get(key)
        if slot is None:
            slot = torch.zeros_like(master_flat)
            st[key] = slot
        flat_slice = slot.narrow(0, off, n)
        flat_slice.copy_(value.flatten().to(dtype=flat_slice.dtype, device=flat_slice.device))

    @property
    def mode(self) -> str:
        return self._mode


# Module-level singleton, set by init_proposal and consumed by fc_manager,
# importance_tracker, recovery. Default-None implementation falls back to
# safe_get_full_* / .data behaviour.
_SINGLETON: Optional[Fp32StateMapper] = None


def set_mapper(m: Optional[Fp32StateMapper]) -> None:
    global _SINGLETON
    _SINGLETON = m


def get_mapper() -> Optional[Fp32StateMapper]:
    return _SINGLETON


# Anchor-time offload: stash FP32 master / Adam (m, v) / tracker
# accumulator state to pinned CPU so FD's +ε/-ε passes fit on one A100.
# Pinned buffers are kept alive across calls so we pay the page-locking
# cost only on the first refresh. Keyed by a STABLE str/tuple identity
# (e.g. ``("master", g)`` or ``("adam", g, "exp_avg")``) so subsequent
# refresh cycles always find their cached pinned buffer even if the
# underlying Python tensor object identity changes (DeepSpeed BF16
# optimizer can re-wrap master/state tensors between iters, which would
# invalidate any id(t)-based cache).
_PINNED_POOL: Dict[Any, torch.Tensor] = {}

# Tracker fields that own GPU state (not views) — preserved across offload.
_TRACKER_STATE_ATTRS = (
    "_g_fast_dg_flat", "_g_fast_exc_mask",
    "_g_fast_h_anchor_sample_buf",
)
# Extended tracker buffers offloaded only during anchor refresh
# (not used by the FD compute path — chunk_buf is scoring scratch,
# selected_mask is dispatch output, name_group_count is dispatch output,
# v_sample buffers are scoring inputs, _outlier_* are dispatch
# metadata). _g_fast_h_anchor_sample_buf is EXCLUDED — anchor writes to it.
_TRACKER_STATE_ATTRS_ANCHOR_EXTRA = (
    "_g_fast_chunk_buf", "_g_fast_selected_mask",
    "_g_fast_name_group_count", "_g_fast_offsets_t",
    "_g_fast_score_sample_buf", "_g_fast_score_sample_sorted_buf",
    "_g_fast_score_sample_indices_buf",
    "_outlier_name_count", "_outlier_name_slot",
    "_outlier_name_t_off", "_outlier_elem_offsets",
    "_g_fast_name_group_slot",
)
# Tracker fields that hold *views* into Adam storage — dropped so storage
# can be freed; rebuilt by restore via tracker._init_accum_sq_flat.
_TRACKER_VIEW_FIELDS = ("m_refs", "v_refs", "p_refs", "m_flat", "v_flat", "p_flat")


def _pinned_clone(t: torch.Tensor, key: Any) -> torch.Tensor:
    """Pinned CPU clone of ``t`` cached under stable ``key``. ``key`` must
    be a stable identifier across iterations (e.g. ``("master", g)`` or
    ``("adam", g, "exp_avg")``) — using ``id(t)`` is NOT safe because
    DeepSpeed BF16 optimizer can re-wrap master/state tensors between
    iters, causing cache misses and fresh ``cudaHostAlloc`` per refresh
    (~400 ms per 700 MB alloc on A100)."""
    buf = _PINNED_POOL.get(key)
    if buf is None or buf.shape != t.shape or buf.dtype != t.dtype:
        buf = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
        _PINNED_POOL[key] = buf
    buf.copy_(t, non_blocking=True)
    return buf


def _release(t: torch.Tensor) -> torch.Tensor:
    """Return a 0-element placeholder with the same dtype/device as ``t``;
    when assigned to ``t.data`` it drops the storage (if no other refs)."""
    return torch.empty(0, dtype=t.dtype, device=t.device)


def _purge_stale_pool_entries(live_keys: set) -> None:
    """Drop pool entries whose stable key is no longer live — keeps the
    pool bounded across many refresh cycles when tracker buffers get
    reallocated each time."""
    for k in list(_PINNED_POOL.keys()):
        if k not in live_keys:
            del _PINNED_POOL[k]


@torch.no_grad()
def offload_adam_state_to_cpu(tracker: Optional[object] = None) -> Dict[str, Any]:
    """Stash FP32 master + Adam (m, v) + tracker accumulator state to
    pinned CPU. Tracker views into Adam storage are dropped; pass the
    returned handle to ``restore_adam_state_from_cpu`` to rebuild them.

    Caller must drain async writer queues first — the writer thread
    accesses tracker views and will fault if they're dropped mid-task.
    """
    handle: Dict[str, Any] = {"adam": {}, "tracker": {}}
    mp = get_mapper()
    if mp is None or mp._fp32_flat is None:
        return handle

    # Drop tracker view caches so their refs don't keep storage alive.
    if tracker is not None and hasattr(tracker, "_groups"):
        for grp in tracker._groups:
            for f in _TRACKER_VIEW_FIELDS:
                grp[f] = [] if f.endswith("refs") else None
            grp["zero_copy"] = False
    if tracker is not None and hasattr(tracker, "_v_mapping_v_flats"):
        tracker._v_mapping_v_flats = []

    live_keys: set = set()

    # Tracker accumulator buffers — own state, save then release. Key by
    # stable attr name so the cached pinned buffer is reused across
    # cycles even though tracker reallocates the underlying GPU tensor
    # each refresh.
    for attr in _TRACKER_STATE_ATTRS:
        t = getattr(tracker, attr, None) if tracker is not None else None
        if isinstance(t, torch.Tensor) and t.is_cuda:
            key = ("tracker", attr)
            handle["tracker"][attr] = _pinned_clone(t, key)
            live_keys.add(key)
            t.data = _release(t)

    # FP32 master + Adam moments. Key by stable ``(g, slot)`` rather
    # than id(t) — DeepSpeed's BF16 optimizer can wrap master/state
    # tensors in fresh Python wrappers between iters which would
    # invalidate id-based keys and force cudaHostAlloc on every refresh
    # (~400 ms per 700 MB alloc).
    import os as _os_offl
    _diag = _os_offl.environ.get("PROPOSAL_OFFLOAD_DIAG", "") in ("1",)
    if _diag:
        print(f"[offload_diag] mp._fp32_flat has {len(mp._fp32_flat)} groups",
              flush=True)
    for g, master in enumerate(mp._fp32_flat):
        slot: Dict[str, torch.Tensor] = {}
        if _diag:
            print(f"[offload_diag] g={g} master.numel={master.numel()} "
                  f"dtype={master.dtype} data_ptr=0x{master.data_ptr():x}",
                  flush=True)
        master_key = ("adam", g, "master")
        slot["master"] = _pinned_clone(master, master_key)
        live_keys.add(master_key)
        master.data = _release(master)
        st = mp._inner_opt.state.get(master, {})
        for key_name in ("exp_avg", "exp_avg_sq"):
            t = st.get(key_name)
            if isinstance(t, torch.Tensor):
                if _diag:
                    print(f"[offload_diag] g={g} state[{key_name}].numel={t.numel()} "
                          f"dtype={t.dtype} data_ptr=0x{t.data_ptr():x}",
                          flush=True)
                state_key = ("adam", g, key_name)
                slot[key_name] = _pinned_clone(t, state_key)
                live_keys.add(state_key)
                t.data = _release(t)
        handle["adam"][g] = slot

    # Note: tracker GPU buffers get reallocated by _init_accum_sq_flat on
    # restore — but our keys are stable strings, so the same pinned host
    # buffer is reused. Adam state tensors are also keyed stably. Purge
    # only removes pool entries whose stable key wasn't used this cycle.
    _purge_stale_pool_entries(live_keys)

    if torch.cuda.is_available():
        torch.cuda.synchronize()       # drain pinned-D2H copies
        torch.cuda.empty_cache()
    return handle


@torch.no_grad()
def restore_adam_state_from_cpu(
    handle: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    tracker: Optional[object] = None,
) -> None:
    """Reverse of :func:`offload_adam_state_to_cpu`."""
    mp = get_mapper()
    if mp is None or mp._fp32_flat is None:
        return

    # FP32 master + Adam moments back to GPU (allocate fresh, copy from pinned).
    for g, master in enumerate(mp._fp32_flat):
        slot = handle["adam"].get(g, {})
        if "master" in slot:
            master.data = slot["master"].to(master.device, non_blocking=True)
        st = mp._inner_opt.state.setdefault(master, {})
        for key in ("exp_avg", "exp_avg_sq"):
            cpu_t = slot.get(key)
            if cpu_t is None:
                continue
            new_gpu = cpu_t.to(master.device, non_blocking=True)
            existing = st.get(key)
            if isinstance(existing, torch.Tensor):
                existing.data = new_gpu
            else:
                st[key] = new_gpu

    # Force tracker to rebuild views into the new GPU storage, then
    # overwrite the freshly-zeroed accumulator buffers with the saved state.
    if tracker is not None and hasattr(tracker, "_init_accum_sq_flat"):
        tracker._flat_layout_done = False
        tracker._init_accum_sq_flat(optimizer)
        for attr, cpu_t in handle["tracker"].items():
            t = getattr(tracker, attr, None)
            if isinstance(t, torch.Tensor) and t.is_cuda:
                t.copy_(cpu_t, non_blocking=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()       # ensure all H2D copies land before training resumes


@torch.no_grad()
def _offload_tracker_extras(tracker: Any) -> Dict[str, torch.Tensor]:
    """Pinned-clone + release the EXTENDED tracker buffers used during
    normal training but NOT by the anchor's FD compute. Returns a handle
    dict (attr → pinned-host tensor) for ``_restore_tracker_extras``.
    """
    handle: Dict[str, torch.Tensor] = {}
    if tracker is None:
        return handle
    for attr in _TRACKER_STATE_ATTRS_ANCHOR_EXTRA:
        t = getattr(tracker, attr, None)
        if isinstance(t, torch.Tensor) and t.is_cuda:
            key = ("tracker_extra", attr)
            handle[attr] = _pinned_clone(t, key)
            t.data = _release(t)
    return handle


@torch.no_grad()
def _restore_tracker_extras(tracker: Any, handle: Dict[str, torch.Tensor]) -> None:
    """Reverse of ``_offload_tracker_extras``. Critical: ``adam`` restore
    calls ``tracker._init_accum_sq_flat`` which re-allocates several
    extra buffers (e.g., ``_g_fast_chunk_buf``, ``_g_fast_global_hist``).
    If those are still in our ``handle``, we MUST copy IN-PLACE into the
    re-allocated tensor (not setattr a fresh tensor) — otherwise the
    re-allocated one leaks.
    """
    if tracker is None or not handle:
        return
    for attr, cpu_t in handle.items():
        existing = getattr(tracker, attr, None)
        if (isinstance(existing, torch.Tensor) and existing.is_cuda
                and existing.numel() > 0
                and existing.shape == cpu_t.shape
                and existing.dtype == cpu_t.dtype):
            # Re-allocated already by _init_accum_sq_flat → copy in place.
            existing.copy_(cpu_t, non_blocking=True)
        else:
            # Still the 0-elem placeholder from _release → alloc + rebind.
            new_gpu = torch.empty(cpu_t.shape, dtype=cpu_t.dtype, device="cuda")
            new_gpu.copy_(cpu_t, non_blocking=True)
            setattr(tracker, attr, new_gpu)


@torch.no_grad()
def _offload_segment_pipeline_bufs(seg_pipe: Any) -> List[torch.Tensor]:
    """Pinned-clone + release ``segment_pipeline._gpu_bufs`` (double-buf
    staging). Returns the original list so caller can rebind on restore.
    """
    if seg_pipe is None:
        return []
    bufs = getattr(seg_pipe, "_gpu_bufs", None)
    if not bufs:
        return []
    handle: List[torch.Tensor] = []
    for i, gb in enumerate(bufs):
        if isinstance(gb, torch.Tensor) and gb.is_cuda:
            key = ("seg_pipe_gpu_buf", i)
            handle.append(_pinned_clone(gb, key))
            gb.data = _release(gb)
        else:
            handle.append(None)  # placeholder
    return handle


@torch.no_grad()
def _restore_segment_pipeline_bufs(seg_pipe: Any, handle: List[torch.Tensor]) -> None:
    if seg_pipe is None or not handle:
        return
    bufs = getattr(seg_pipe, "_gpu_bufs", None)
    if not bufs:
        return
    for i, cpu_t in enumerate(handle):
        if cpu_t is None:
            continue
        new_gpu = torch.empty(cpu_t.shape, dtype=cpu_t.dtype, device="cuda")
        new_gpu.copy_(cpu_t, non_blocking=True)
        bufs[i].data = new_gpu


@contextmanager
def anchor_offload_context(model: Any, tracker: Any,
                           segment_pipeline: Any = None,
                           async_writer: Any = None,
                           *, offload_optimizer: bool = True,
                           offload_tracker_full: bool = True,
                           offload_segment: bool = True):
    """Unified swap-out → anchor → swap-in for refresh anchor.

    Pre-anchor:
      0. async_writer drain (free writer-held GPU refs to tracker views).
      1. ``torch.cuda.synchronize()`` (drain all streams).
      2. (optional) offload extended tracker buffers (chunk_buf,
         selected_mask, name_group_count, sub-sample, _outlier_*).
      3. (optional) offload segment_pipeline._gpu_bufs (2 × 128 MiB).
      4. (optional) ``offload_adam_state_to_cpu(tracker)`` — FP32 master
         + Adam (m, v) + the 3 core tracker buffers (dg_flat,
         h_anchor_sample_buf, exc_mask).
      5. ``torch.cuda.empty_cache()`` — release allocator pool pages.
      6. Snapshot ``pre_anchor_alloc`` for leak check.

    Yield — anchor's FD compute runs here. ``_g_fast_h_anchor_sample_buf``
    stays on GPU because anchor writes to it.

    Post-anchor:
      1. ``torch.cuda.synchronize()`` (drain anchor's streams).
      2. ``torch.cuda.empty_cache()`` (free anchor's transient pool).
      3. Restore in REVERSE order (segment / extras / adam) — caching
         allocator under expandable_segments tends to reuse the same
         virtual region for same-size allocations, minimising
         fragmentation. Each restore: alloc fresh GPU tensor of saved
         shape, H2D copy from pinned, rebind attr.
      4. ``torch.cuda.synchronize()``.
      5. Leak check: ``cuda.memory_allocated()`` vs ``pre_anchor_alloc``
         should be within ±2 MB. Larger drift = leak warning.
    """
    # Pre-anchor
    if async_writer is not None:
        try:
            async_writer.drain_non_expert()
            async_writer.drain_expert()
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    extras_handle: Dict[str, torch.Tensor] = {}
    seg_handle: List[torch.Tensor] = []
    adam_handle: Optional[Dict[str, Any]] = None
    if offload_tracker_full:
        extras_handle = _offload_tracker_extras(tracker)
    if offload_segment:
        seg_handle = _offload_segment_pipeline_bufs(segment_pipeline)
    if offload_optimizer:
        adam_handle = offload_adam_state_to_cpu(tracker)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pre_anchor_alloc = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )

    try:
        yield
    finally:
        # Post-anchor: drain + reclaim + restore in reverse.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if adam_handle is not None:
            # ``optimizer`` is the inner torch optimizer; the existing
            # restore reads it from the global mapper.
            restore_adam_state_from_cpu(
                adam_handle, get_mapper()._inner_opt if get_mapper() else None,
                tracker,
            )
        _restore_segment_pipeline_bufs(segment_pipeline, seg_handle)
        _restore_tracker_extras(tracker, extras_handle)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # Leak check.
            post_alloc = torch.cuda.memory_allocated()
            delta = (post_alloc - pre_anchor_alloc) / 1e6
            if abs(delta) > 2.0:
                print(
                    f"[anchor_offload] LEAK delta={delta:+.1f}MB "
                    f"(pre={pre_anchor_alloc/1e6:.1f}MB, "
                    f"post={post_alloc/1e6:.1f}MB)",
                    flush=True,
                )


# === Per-chunk backward support ============================================
# Snapshot/restore Megatron + DeepSpeed training state so per-chunk anchor
# backward (which fires multiple forwards/backwards) doesn't corrupt
# downstream training.
@torch.no_grad()
def snapshot_training_state(optimizer: Any, model: Any = None) -> Dict[str, Any]:
    """Capture state that anchor's multiple forward/backward passes would
    corrupt: cuda RNG, torch RNG, Megatron RNG tracker, DeepSpeed hp_grad
    accumulator, AND model lp.grad (DeepSpeed mode skips optimizer.zero_grad
    at train_step start, so anchor's residual lp.grad would accumulate into
    next iter's first mb grad). Restore with :func:`restore_training_state`.
    """
    snap: Dict[str, Any] = {}
    if torch.cuda.is_available():
        snap["cuda_rng"] = torch.cuda.get_rng_state_all()
    snap["torch_rng"] = torch.get_rng_state()
    try:
        from megatron.core.tensor_parallel import get_cuda_rng_tracker
        snap["mpu_rng"] = get_cuda_rng_tracker().get_states()
    except Exception:
        snap["mpu_rng"] = None
    # DeepSpeed BF16 hp_grad.
    hp_grads: List[Any] = []
    try:
        mp = get_mapper()
        if mp is not None and mp._fp32_flat is not None:
            for g, master in enumerate(mp._fp32_flat):
                g_ref = getattr(master, "grad", None)
                if isinstance(g_ref, torch.Tensor):
                    hp_grads.append(g_ref.detach().clone())
                else:
                    hp_grads.append(None)
    except Exception:
        pass
    snap["hp_grads"] = hp_grads
    # Model lp.grad — must be restored so anchor's N×backward leaves no
    # residual in next iter's grad accumulation.
    lp_grads: List[Tuple[Any, Any]] = []
    if model is not None:
        try:
            it = model.parameters() if hasattr(model, "parameters") else iter(model)
            for p in it:
                g_ref = getattr(p, "grad", None)
                if isinstance(g_ref, torch.Tensor):
                    lp_grads.append((p, g_ref.detach().clone()))
                else:
                    lp_grads.append((p, None))
        except Exception:
            pass
    snap["lp_grads"] = lp_grads
    # Model buffers (registered_buffer in nn.Module) — stateful state that
    # forward passes may update (running stats, attention mask cache,
    # ROPE freq cache, MoE expert utilization, etc). Per-chunk's N forwards
    # may drift these → next iter's forward sees corrupted buffer state.
    # Param.grad covers grad accumulation but NOT module buffers.
    buffers: List[Tuple[Any, Any]] = []
    if model is not None:
        try:
            m = model[0] if isinstance(model, (list, tuple)) else model
            for name, b in m.named_buffers():
                if isinstance(b, torch.Tensor):
                    buffers.append((b, b.detach().clone()))
        except Exception:
            pass
    snap["buffers"] = buffers
    return snap


@torch.no_grad()
def restore_training_state(optimizer: Any, snap: Dict[str, Any]) -> None:
    if torch.cuda.is_available() and snap.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(snap["cuda_rng"])
    if snap.get("torch_rng") is not None:
        torch.set_rng_state(snap["torch_rng"])
    if snap.get("mpu_rng") is not None:
        try:
            from megatron.core.tensor_parallel import get_cuda_rng_tracker
            get_cuda_rng_tracker().set_states(snap["mpu_rng"])
        except Exception:
            pass
    hp_grads = snap.get("hp_grads", [])
    if hp_grads:
        try:
            mp = get_mapper()
            if mp is not None and mp._fp32_flat is not None:
                for g, master in enumerate(mp._fp32_flat):
                    if g >= len(hp_grads):
                        break
                    saved = hp_grads[g]
                    cur = getattr(master, "grad", None)
                    if isinstance(saved, torch.Tensor) and isinstance(cur, torch.Tensor):
                        cur.copy_(saved)
                    elif saved is None and cur is not None:
                        cur.zero_()
        except Exception:
            pass
    lp_grads = snap.get("lp_grads", [])
    if lp_grads:
        for p, saved in lp_grads:
            cur = getattr(p, "grad", None)
            if isinstance(saved, torch.Tensor):
                if isinstance(cur, torch.Tensor) and cur.shape == saved.shape:
                    cur.copy_(saved)
                else:
                    p.grad = saved
            else:
                # anchor created grad where there was none — drop it.
                if cur is not None:
                    p.grad = None
