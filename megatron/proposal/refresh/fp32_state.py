"""FP32 optimizer-state mapping and anchor snapshot helpers.

This module locates the exact FP32 master, Adam state, and grad accumulation
storage used by the active training configuration. H-refresh and lossy scoring
depend on that mapping, but neither owns DeepSpeed optimizer internals.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


def get_engine_optimizer(model: Any) -> Any:
    """Return the DeepSpeed engine optimizer wrapper when present."""

    module = model[0] if isinstance(model, (list, tuple)) else model
    return getattr(module, "optimizer", None)


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

    def safe_get_full_fp32_param(param):  # type: ignore
        return None

    def safe_get_full_optimizer_state(param, key):  # type: ignore
        return None

    def safe_set_full_fp32_param(param, value):  # type: ignore
        pass

    def safe_set_full_optimizer_state(param, value, key):  # type: ignore
        pass


ParamCoord = Tuple[int, int, int]  # (group_idx, offset, numel)
LocalPartitionFragment = Tuple[str, torch.nn.Parameter, int, int, int, int]
# (name, param, group_idx, local_offset, param_offset, numel)
TrainingStateSnapshot = Dict[str, Any]


class Fp32StateMapper:
    """Map model params to the optimizer's FP32 master and Adam state."""

    def __init__(
        self,
        engine_optimizer: object,
        inner_torch_optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
    ) -> None:
        del model
        self._engine_opt = engine_optimizer
        self._optimizer_root = inner_torch_optimizer
        self._inner_opt = self._drill_to_torch_optimizer(inner_torch_optimizer)
        self._coords: Dict[int, ParamCoord] = {}
        self._fp32_flat: Optional[List[torch.Tensor]] = None
        self._fp32_grad_flat: Optional[List[torch.Tensor]] = None
        self._direct_master: Dict[int, torch.Tensor] = {}
        self._zero_params_in_partition: List[List[torch.nn.Parameter]] = []
        self._zero_first_offset: List[int] = []
        self._zero_partition_size: List[int] = []
        self._mode = "unknown"
        self._candidate_summaries: List[str] = []
        self._build()

    @staticmethod
    def _drill_to_torch_optimizer(optimizer: object) -> torch.optim.Optimizer:
        cur = optimizer
        for _ in range(8):
            if (
                isinstance(cur, torch.optim.Optimizer)
                and not hasattr(cur, "fp32_groups")
                and not hasattr(cur, "fp32_from_float16_groups")
            ):
                return cur
            inner = getattr(cur, "optimizer", None)
            if inner is None or inner is cur:
                break
            cur = inner
        if isinstance(cur, torch.optim.Optimizer):
            return cur
        return optimizer  # type: ignore[return-value]

    def _build(self) -> None:
        candidates: List[object] = []
        seen = set()

        def add_chain(root: object) -> None:
            cur = root
            for _ in range(8):
                if cur is None:
                    return
                cur_id = id(cur)
                if cur_id in seen:
                    return
                seen.add(cur_id)
                candidates.append(cur)
                inner = getattr(cur, "optimizer", None)
                if inner is None or inner is cur:
                    return
                cur = inner

        add_chain(self._engine_opt)
        add_chain(self._optimizer_root)
        self._candidate_summaries = [
            self._summarize_candidate(candidate) for candidate in candidates
        ]

        for candidate in candidates:
            if self._try_megatron_float16(candidate):
                return
            if self._try_unfused_fp16(candidate):
                return
            if self._try_zero12(candidate):
                return
            if self._try_bf16_zero0(candidate):
                return
            if self._try_fp16_zero0(candidate):
                return
        self._mode = "unrecognised"

    @staticmethod
    def _summarize_candidate(candidate: object) -> str:
        if candidate is None:
            return "None"
        attrs = []
        for attr in (
            "bf16_groups",
            "fp16_groups",
            "bit16_groups",
            "fp32_groups_flat",
            "fp32_groups_flat_partition",
            "single_partition_of_fp32_groups",
            "fp32_groups_gradients_flat",
        ):
            if not hasattr(candidate, attr):
                continue
            value = getattr(candidate, attr)
            if isinstance(value, (list, tuple)):
                attrs.append(f"{attr}[{len(value)}]")
            else:
                attrs.append(attr)
        cls = candidate.__class__
        return f"{cls.__module__}.{cls.__name__}: {', '.join(attrs) or 'no-flat-attrs'}"

    def _try_megatron_float16(self, candidate: object) -> bool:
        if not (
            hasattr(candidate, "fp32_from_float16_groups")
            and hasattr(candidate, "float16_groups")
        ):
            return False
        self._mode = "megatron_float16"
        self._direct_master = {}
        for f16_group, fp32_group in zip(
            candidate.float16_groups, candidate.fp32_from_float16_groups
        ):
            for f16_param, fp32_param in zip(f16_group, fp32_group):
                self._direct_master[id(f16_param)] = fp32_param
        return True

    def _try_unfused_fp16(self, candidate: object) -> bool:
        if not (hasattr(candidate, "fp32_groups") and hasattr(candidate, "fp16_groups")):
            return False
        self._mode = "unfused_fp16"
        self._direct_master = {}
        for f16_group, fp32_group in zip(candidate.fp16_groups, candidate.fp32_groups):
            for f16_param, fp32_param in zip(f16_group, fp32_group):
                self._direct_master[id(f16_param)] = fp32_param
        return True

    def _try_zero12(self, candidate: object) -> bool:
        if not hasattr(candidate, "single_partition_of_fp32_groups"):
            return False
        self._fp32_flat = candidate.single_partition_of_fp32_groups
        self._fp32_grad_flat = getattr(candidate, "fp32_groups_gradients_flat", None)
        self._mode = "zero12"
        self._coords = self._index_groups(candidate, "bit16_groups")
        self._zero_params_in_partition = [
            list(group) for group in getattr(candidate, "params_in_partition", [])
        ]
        self._zero_first_offset = [
            int(v) for v in getattr(candidate, "first_offset", [])
        ]
        self._zero_partition_size = [
            int(v) for v in getattr(candidate, "partition_size", [])
        ]
        return True

    def _try_fp16_zero0(self, candidate: object) -> bool:
        if not (
            hasattr(candidate, "fp32_groups_flat")
            and hasattr(candidate, "fp16_groups")
        ):
            return False
        if not self._groups_match_any_dtype(
            candidate.fp16_groups, (torch.float16, torch.bfloat16)
        ):
            return False
        self._fp32_flat = candidate.fp32_groups_flat
        self._fp32_grad_flat = getattr(candidate, "fp32_groups_gradients_flat", None)
        self._coords = self._index_groups(candidate, "fp16_groups")
        if not self._coords:
            return False
        self._mode = "fp16_zero0"
        return True

    def _try_bf16_zero0(self, candidate: object) -> bool:
        if not (
            hasattr(candidate, "fp32_groups_flat_partition")
            and hasattr(candidate, "bf16_groups")
        ):
            return False
        if not self._groups_match_dtype(candidate.bf16_groups, torch.bfloat16):
            return False
        self._fp32_flat = candidate.fp32_groups_flat_partition
        self._fp32_grad_flat = getattr(candidate, "fp32_groups_gradients_flat", None)
        self._coords = self._index_groups(candidate, "bf16_groups")
        if not self._coords:
            return False
        self._mode = "bf16_zero0"
        return True

    @staticmethod
    def _groups_match_dtype(groups: object, dtype: torch.dtype) -> bool:
        return Fp32StateMapper._groups_match_any_dtype(groups, (dtype,))

    @staticmethod
    def _groups_match_any_dtype(groups: object, dtypes: Tuple[torch.dtype, ...]) -> bool:
        saw_param = False
        for group in groups or []:
            for param in group:
                saw_param = True
                if not isinstance(param, torch.Tensor) or param.dtype not in dtypes:
                    return False
        return saw_param

    @staticmethod
    def _index_groups(engine_opt: object, attr: str) -> Dict[int, ParamCoord]:
        out: Dict[int, ParamCoord] = {}
        groups = getattr(engine_opt, attr, None)
        if groups is None:
            return out
        for group_idx, group in enumerate(groups):
            offset = 0
            for param in group:
                out[id(param)] = (group_idx, offset, param.numel())
                offset += param.numel()
        return out

    def get_param(self, param: torch.nn.Parameter) -> torch.Tensor:
        if self._mode in ("megatron_float16", "unfused_fp16"):
            master = self._direct_master.get(id(param))
            if master is not None:
                return master
            return param.data.to(torch.float32)
        if _HAS_SAFE_API:
            full = safe_get_full_fp32_param(param)
            if full is not None:
                return full
        coord = self._coords.get(id(param))
        if coord is None or self._fp32_flat is None:
            return param.data.to(torch.float32)
        group_idx, offset, numel = coord
        flat = self._fp32_flat[group_idx]
        if offset < 0 or offset + numel > int(flat.numel()):
            raise RuntimeError(
                "FP32 full-param view is not local to this ZeRO partition; "
                "use a ZeRO partition-aware path or DeepSpeed safe full API."
            )
        return flat.narrow(0, offset, numel).view(param.shape)

    def get_state(self, param: torch.nn.Parameter, key: str) -> Optional[torch.Tensor]:
        if self._mode in ("megatron_float16", "unfused_fp16"):
            master = self._direct_master.get(id(param))
            if master is None:
                return None
            return self._inner_opt.state.get(master, {}).get(key)
        if _HAS_SAFE_API:
            try:
                full = safe_get_full_optimizer_state(param, key)
            except (TypeError, KeyError, AttributeError):
                full = None
            if full is not None:
                return full
        coord = self._coords.get(id(param))
        if coord is None or self._fp32_flat is None:
            return self._inner_opt.state.get(param, {}).get(key)
        group_idx, offset, numel = coord
        master_flat = self._fp32_flat[group_idx]
        state_value = self._inner_opt.state.get(master_flat, {}).get(key)
        if state_value is None:
            return None
        if offset < 0 or offset + numel > int(state_value.numel()):
            raise RuntimeError(
                "FP32 optimizer-state full-param view is not local to this "
                "ZeRO partition; use a ZeRO partition-aware path or "
                "DeepSpeed safe full API."
            )
        return state_value.narrow(0, offset, numel).view(param.shape)

    def get_local_param_view(self, param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        if self._mode in ("megatron_float16", "unfused_fp16"):
            master = self._direct_master.get(id(param))
            if master is not None:
                return master
        coord = self._coords.get(id(param))
        if coord is None or self._fp32_flat is None:
            return None
        group_idx, offset, numel = coord
        if group_idx >= len(self._fp32_flat):
            return None
        flat = self._fp32_flat[group_idx]
        if offset < 0 or offset + numel > int(flat.numel()):
            return None
        return flat.narrow(0, offset, numel).view(param.shape)

    def get_local_state_view(
        self, param: torch.nn.Parameter, key: str
    ) -> Optional[torch.Tensor]:
        if self._mode in ("megatron_float16", "unfused_fp16"):
            master = self._direct_master.get(id(param))
            if master is None:
                return None
            return self._inner_opt.state.get(master, {}).get(key)
        coord = self._coords.get(id(param))
        if coord is None or self._fp32_flat is None:
            return None
        group_idx, offset, numel = coord
        if group_idx >= len(self._fp32_flat):
            return None
        master_flat = self._fp32_flat[group_idx]
        state_value = self._inner_opt.state.get(master_flat, {}).get(key)
        if state_value is None:
            return None
        if offset < 0 or offset + numel > int(state_value.numel()):
            return None
        return state_value.narrow(0, offset, numel).view(param.shape)

    def zero12_local_fragments(
        self,
        named_params: Iterable[Tuple[str, torch.nn.Parameter]],
    ) -> List[LocalPartitionFragment]:
        """Return ZeRO-1/2 local partition fragments for the requested params.

        DeepSpeed's ZeRO-1/2 fp32 master is partitioned by flat element range,
        not by tensor object.  A non-expert tensor can straddle a partition
        boundary, so checkpoint code must use these local fragments instead of
        treating ``single_partition_of_fp32_groups`` as a full-param flat.
        """

        if self._mode != "zero12" or self._fp32_flat is None:
            return []
        wanted: Dict[int, Tuple[str, torch.nn.Parameter]] = {
            id(p): (name, p) for name, p in named_params
        }
        if not wanted:
            return []

        fragments: List[LocalPartitionFragment] = []
        n_groups = min(len(self._zero_params_in_partition), len(self._fp32_flat))
        for group_idx in range(n_groups):
            part_size = (
                self._zero_partition_size[group_idx]
                if group_idx < len(self._zero_partition_size)
                else int(self._fp32_flat[group_idx].numel())
            )
            part_size = min(int(part_size), int(self._fp32_flat[group_idx].numel()))
            first_offset = (
                self._zero_first_offset[group_idx]
                if group_idx < len(self._zero_first_offset)
                else 0
            )
            local_offset = 0
            for idx, param in enumerate(self._zero_params_in_partition[group_idx]):
                param_offset = int(first_offset) if idx == 0 else 0
                available = int(param.numel()) - int(param_offset)
                if available <= 0:
                    continue
                n = min(int(available), int(part_size) - int(local_offset))
                if n <= 0:
                    break
                hit = wanted.get(id(param))
                if hit is not None:
                    name, p = hit
                    fragments.append((
                        str(name),
                        p,
                        int(group_idx),
                        int(local_offset),
                        int(param_offset),
                        int(n),
                    ))
                local_offset += int(n)
                if local_offset >= part_size:
                    break
        return fragments

    def zero12_partition_view(
        self,
        group_idx: int,
        local_offset: int,
        numel: int,
        *,
        state_key: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        if self._mode != "zero12" or self._fp32_flat is None:
            return None
        group_idx = int(group_idx)
        local_offset = int(local_offset)
        numel = int(numel)
        if group_idx < 0 or group_idx >= len(self._fp32_flat):
            return None
        flat = self._fp32_flat[group_idx]
        if local_offset < 0 or local_offset + numel > int(flat.numel()):
            return None
        if state_key is None:
            return flat.narrow(0, local_offset, numel)
        state = self._inner_opt.state.setdefault(flat, {})
        slot = state.get(state_key)
        if slot is None:
            slot = torch.zeros_like(flat)
            state[state_key] = slot
        if local_offset + numel > int(slot.numel()):
            return None
        return slot.narrow(0, local_offset, numel)

    @torch.no_grad()
    def copy_zero12_partition_fragment(
        self,
        *,
        group_idx: int,
        local_offset: int,
        value: torch.Tensor,
        state_key: Optional[str] = None,
    ) -> None:
        dst = self.zero12_partition_view(
            int(group_idx),
            int(local_offset),
            int(value.numel()),
            state_key=state_key,
        )
        if dst is None:
            raise RuntimeError(
                "ZeRO partition payload does not match local optimizer layout: "
                f"group={group_idx} offset={local_offset} n={value.numel()}"
            )
        dst.copy_(value.reshape(-1).to(device=dst.device, dtype=dst.dtype))

    def sync_zero12_fp32_to_lp(self) -> None:
        """Refresh ZeRO low-precision model weights from local fp32 partitions."""

        if self._mode != "zero12":
            return
        update_lp = getattr(self._engine_opt, "update_lp_params", None)
        if callable(update_lp):
            update_lp()

    def get_grad_accum_view(self, param: torch.nn.Parameter) -> torch.Tensor:
        """Return the FP32 storage where training normally accumulates grad."""

        coord = self._coords.get(id(param))
        if (
            coord is not None
            and self._fp32_grad_flat is not None
            and coord[0] < len(self._fp32_grad_flat)
        ):
            group_idx, offset, numel = coord
            grad_flat = self._fp32_grad_flat[group_idx]
            return grad_flat.narrow(0, offset, numel).view(-1)

        hp_mapping = getattr(param, "_hp_mapping", None)
        param_group_index = getattr(param, "_index_in_param_group", None)
        if hp_mapping is not None and param_group_index is not None:
            try:
                hp_grad = hp_mapping.get_lp_grad_fragment(param_group_index)
            except (AttributeError, KeyError, TypeError, ValueError):
                hp_grad = None
            if isinstance(hp_grad, torch.Tensor):
                if hp_grad.dtype != torch.float32:
                    raise RuntimeError(
                        f"anchor requires FP32 hp grad fragment, got {hp_grad.dtype}"
                    )
                return hp_grad.view(-1)

        main_grad = getattr(param, "main_grad", None)
        if isinstance(main_grad, torch.Tensor):
            if main_grad.dtype != torch.float32:
                raise RuntimeError(
                    f"anchor requires FP32 grad accumulation storage, got {main_grad.dtype}"
                )
            return main_grad.view(-1)

        if param.grad is None:
            if param.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise RuntimeError(
                    "anchor could not locate supported training grad accumulation "
                    f"storage for dtype {param.dtype}"
                )
            param.grad = torch.zeros_like(param, dtype=param.dtype)
        return param.grad.view(-1)

    @torch.no_grad()
    def set_param(self, param: torch.nn.Parameter, value: torch.Tensor) -> None:
        if self._mode in ("megatron_float16", "unfused_fp16"):
            master = self._direct_master.get(id(param))
            if master is not None:
                master.data.copy_(value.to(device=master.device, dtype=master.dtype))
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))
            return
        if _HAS_SAFE_API and (hasattr(param, "_hp_mapping") or hasattr(param, "ds_id")):
            try:
                safe_set_full_fp32_param(param, value)
                return
            except Exception:
                pass
        coord = self._coords.get(id(param))
        if coord is not None and self._fp32_flat is not None:
            group_idx, offset, numel = coord
            flat_slice = self._fp32_flat[group_idx].data.narrow(0, offset, numel)
            flat_slice.copy_(
                value.flatten().to(device=flat_slice.device, dtype=flat_slice.dtype)
            )
        param.data.copy_(
            value.view(param.shape).to(device=param.device, dtype=param.dtype)
        )

    @torch.no_grad()
    def set_state(
        self, param: torch.nn.Parameter, value: torch.Tensor, key: str
    ) -> None:
        if self._mode in ("megatron_float16", "unfused_fp16"):
            master = self._direct_master.get(id(param))
            if master is None:
                return
            state = self._inner_opt.state.setdefault(master, {})
            slot = state.get(key)
            if slot is None:
                slot = torch.zeros_like(master)
                state[key] = slot
            slot.copy_(value.to(device=slot.device, dtype=slot.dtype))
            return
        if _HAS_SAFE_API and (hasattr(param, "_hp_mapping") or hasattr(param, "ds_id")):
            try:
                safe_set_full_optimizer_state(param, value, key)
                return
            except Exception:
                pass
        coord = self._coords.get(id(param))
        if coord is None or self._fp32_flat is None:
            state = self._inner_opt.state.setdefault(param, {})
            slot = state.get(key)
            if slot is None:
                slot = torch.zeros_like(param, dtype=torch.float32)
                state[key] = slot
            slot.copy_(value.to(device=slot.device, dtype=slot.dtype))
            return
        group_idx, offset, numel = coord
        master_flat = self._fp32_flat[group_idx]
        state = self._inner_opt.state.setdefault(master_flat, {})
        slot = state.get(key)
        if slot is None:
            slot = torch.zeros_like(master_flat)
            state[key] = slot
        slot.narrow(0, offset, numel).copy_(
            value.flatten().to(device=slot.device, dtype=slot.dtype)
        )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def candidate_summaries(self) -> List[str]:
        return list(self._candidate_summaries)


_MAPPER: Optional[Fp32StateMapper] = None


def set_mapper(mapper: Optional[Fp32StateMapper]) -> None:
    global _MAPPER
    _MAPPER = mapper


def get_mapper() -> Optional[Fp32StateMapper]:
    return _MAPPER


_PINNED_POOL: Dict[Any, torch.Tensor] = {}


def pinned_snapshot(tensor: torch.Tensor, key: Any) -> torch.Tensor:
    """Copy ``tensor`` into a reusable pinned CPU buffer keyed by ``key``."""

    buf = _PINNED_POOL.get(key)
    if buf is None or buf.shape != tensor.shape or buf.dtype != tensor.dtype:
        buf = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
        _PINNED_POOL[key] = buf
    buf.copy_(tensor, non_blocking=True)
    return buf


@torch.no_grad()
def snapshot_training_state(optimizer: Any, model: Any = None) -> TrainingStateSnapshot:
    """Capture mutable training state touched by anchor forward/backward passes."""

    del optimizer
    snap: Dict[str, Any] = {}
    if torch.cuda.is_available():
        snap["cuda_rng"] = torch.cuda.get_rng_state_all()
    snap["torch_rng"] = torch.get_rng_state()
    try:
        from megatron.core.tensor_parallel import get_cuda_rng_tracker

        snap["mpu_rng"] = get_cuda_rng_tracker().get_states()
    except Exception:
        snap["mpu_rng"] = None

    hp_grads: List[Any] = []
    try:
        mapper = get_mapper()
        grad_sources = (
            mapper._fp32_grad_flat or mapper._fp32_flat
            if mapper is not None
            else None
        )
        if mapper is not None and grad_sources is not None:
            for source in grad_sources:
                if mapper._fp32_grad_flat is not None:
                    hp_grads.append(
                        pinned_snapshot(
                            source.detach(),
                            key=("hp_grad_flat", len(hp_grads)),
                        )
                    )
                else:
                    grad = getattr(source, "grad", None)
                    hp_grads.append(
                        pinned_snapshot(
                            grad.detach(),
                            key=("hp_param_grad", len(hp_grads)),
                        )
                        if isinstance(grad, torch.Tensor) else None
                    )
    except Exception:
        pass
    snap["hp_grads"] = hp_grads

    lp_grads: List[Tuple[Any, Any]] = []
    if model is not None:
        try:
            module = model[0] if isinstance(model, (list, tuple)) else model
            for param in module.parameters():
                grad = getattr(param, "grad", None)
                lp_grads.append(
                    (
                        param,
                        pinned_snapshot(
                            grad.detach(),
                            key=("lp_grad", id(param)),
                        )
                        if isinstance(grad, torch.Tensor) else None,
                    )
                )
        except Exception:
            pass
    snap["lp_grads"] = lp_grads

    buffers: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if model is not None:
        try:
            module = model[0] if isinstance(model, (list, tuple)) else model
            for _name, buf in module.named_buffers():
                if isinstance(buf, torch.Tensor):
                    buffers.append((
                        buf,
                        pinned_snapshot(
                            buf.detach(),
                            key=("model_buffer", _name, tuple(buf.shape), buf.dtype),
                        ),
                    ))
        except Exception:
            pass
    snap["buffers"] = buffers
    return snap


@torch.no_grad()
def restore_training_state(optimizer: Any, snap: TrainingStateSnapshot) -> None:
    """Restore state captured by :func:`snapshot_training_state`."""

    del optimizer
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
            mapper = get_mapper()
            grad_sources = (
                mapper._fp32_grad_flat or mapper._fp32_flat
                if mapper is not None
                else None
            )
            if mapper is not None and grad_sources is not None:
                for group_idx, source in enumerate(grad_sources):
                    if group_idx >= len(hp_grads):
                        break
                    saved = hp_grads[group_idx]
                    if mapper._fp32_grad_flat is not None:
                        if isinstance(saved, torch.Tensor):
                            source.copy_(saved)
                    else:
                        current = getattr(source, "grad", None)
                        if isinstance(saved, torch.Tensor) and isinstance(current, torch.Tensor):
                            current.copy_(saved)
                        elif saved is None and current is not None:
                            current.zero_()
        except Exception:
            pass

    for param, saved in snap.get("lp_grads", []):
        current = getattr(param, "grad", None)
        if isinstance(saved, torch.Tensor):
            if isinstance(current, torch.Tensor) and current.shape == saved.shape:
                current.copy_(saved)
            else:
                param.grad = saved.to(device=param.device, non_blocking=True)
        elif current is not None:
            param.grad = None

    for buffer, saved in snap.get("buffers", []):
        if isinstance(buffer, torch.Tensor) and isinstance(saved, torch.Tensor):
            buffer.copy_(saved)
