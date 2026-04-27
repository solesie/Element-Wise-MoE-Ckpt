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

from typing import Dict, Optional, Tuple

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
