"""Local expert partition schema for BF16/ZeRO optimizer layouts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import torch

from ..refresh.fp32_state import get_mapper


ExpertId = Tuple[int, int]
SchemaEntry = Tuple[str, torch.Size, int]


@dataclass(frozen=True)
class ExpertPartitionPlan:
    expert_tensor_map: Dict[ExpertId, List[SchemaEntry]]
    fragments_by_expert: Dict[ExpertId, List[Dict[str, Any]]]
    field_cache_by_expert: Dict[ExpertId, List[Dict[str, Any]]]
    fragment_to_expert: Dict[str, ExpertId]


@dataclass(frozen=True)
class ActiveTrackedFragmentSpec:
    """Stable parameter identity for one active rank-owned runtime fragment."""

    runtime_name: str
    expert_id: ExpertId
    parameter_name: str
    parameter_offset: int
    numel: int
    parameter_numel: int
    parameter_shape: Tuple[int, ...]
    optimizer_group_idx: int | None = None
    optimizer_local_offset: int | None = None


def build_active_tracked_fragment_specs(
    *,
    active_names: Iterable[str],
    name_to_expert: Mapping[str, ExpertId],
    fragment_meta: Mapping[str, Mapping[str, Any]],
    name_to_param: Mapping[str, torch.nn.Parameter],
) -> Tuple[ActiveTrackedFragmentSpec, ...]:
    """Resolve runtime names without exposing rank-local ``#frag`` parsing.

    The checkpoint layout owns fragment identity.  Telemetry consumes this
    immutable API and therefore never treats the rank-local runtime suffix as
    a cross-rank key.
    """

    names = tuple(str(name) for name in active_names)
    if len(set(names)) != len(names):
        raise RuntimeError("active tracked fragment names must be unique")
    meta_names = {str(name) for name in fragment_meta}
    if meta_names and set(names) != meta_names:
        missing = sorted(meta_names - set(names))[:8]
        extra = sorted(set(names) - meta_names)[:8]
        raise RuntimeError(
            "active partition fragment set does not match partition metadata: "
            f"missing_active={missing} missing_metadata={extra}"
        )

    out: list[ActiveTrackedFragmentSpec] = []
    for runtime_name in names:
        eid_raw = name_to_expert.get(runtime_name)
        if eid_raw is None:
            raise RuntimeError(
                f"active tracked fragment has no expert id: {runtime_name!r}"
            )
        meta = fragment_meta.get(runtime_name)
        if meta is not None:
            parameter_name = str(meta.get("name", ""))
            parameter_offset = int(meta.get("param_offset", 0) or 0)
            numel = int(meta.get("numel", 0) or 0)
            optimizer_group_idx = int(meta.get("group_idx", 0) or 0)
            optimizer_local_offset = int(meta.get("local_offset", 0) or 0)
        else:
            parameter_name = runtime_name
            parameter_offset = 0
            parameter = name_to_param.get(parameter_name)
            numel = int(parameter.numel()) if parameter is not None else 0
            optimizer_group_idx = None
            optimizer_local_offset = None
        parameter = name_to_param.get(parameter_name)
        if parameter is None:
            raise RuntimeError(
                "active tracked fragment cannot resolve original parameter: "
                f"runtime={runtime_name!r} parameter={parameter_name!r}"
            )
        parameter_numel = int(parameter.numel())
        if (
            numel <= 0
            or parameter_offset < 0
            or parameter_offset + numel > parameter_numel
        ):
            raise RuntimeError(
                "active tracked fragment interval is invalid: "
                f"runtime={runtime_name!r} offset={parameter_offset} "
                f"numel={numel} parameter_numel={parameter_numel}"
            )
        out.append(
            ActiveTrackedFragmentSpec(
                runtime_name=runtime_name,
                expert_id=(int(eid_raw[0]), int(eid_raw[1])),
                parameter_name=parameter_name,
                parameter_offset=parameter_offset,
                numel=numel,
                parameter_numel=parameter_numel,
                parameter_shape=tuple(int(x) for x in parameter.shape),
                optimizer_group_idx=optimizer_group_idx,
                optimizer_local_offset=optimizer_local_offset,
            )
        )
    return tuple(out)


def build_expert_partition_plan(
    named_params: Dict[str, torch.nn.Parameter],
    separated_tensors_by_slot: Dict[ExpertId, List[str]],
) -> ExpertPartitionPlan | None:
    """Describe local DP-owned expert fp32 partition fragments.

    DeepSpeed BF16 optimizer partitions fp32 master/Adam state by flat local
    ranges even when the training configuration is otherwise ZeRO-0.  Expert
    checkpoint code must therefore store local fragments, just like the
    existing non-expert partition payload path.
    """

    mapper = get_mapper()
    if mapper is None:
        return None
    mode = str(getattr(mapper, "mode", ""))
    if mode not in ("bf16_zero0", "zero12"):
        return None

    expert_tensor_map: Dict[ExpertId, List[SchemaEntry]] = {}
    fragments_by_expert: Dict[ExpertId, List[Dict[str, Any]]] = {}
    field_cache_by_expert: Dict[ExpertId, List[Dict[str, Any]]] = {}
    fragment_to_expert: Dict[str, ExpertId] = {}

    for eid in sorted(separated_tensors_by_slot):
        names = [n for n in separated_tensors_by_slot[eid] if n in named_params]
        raw_fragments = mapper.zero12_local_fragments(
            (name, named_params[name]) for name in names
        )
        raw_fragments = sorted(
            raw_fragments,
            key=lambda item: (
                int(item[2]),
                int(item[3]),
                str(item[0]),
                int(item[4]),
            ),
        )
        schema: List[SchemaEntry] = []
        meta: List[Dict[str, Any]] = []
        fields: List[Dict[str, Any]] = []
        for frag_idx, (name, param, group_idx, local_offset, param_offset, numel) in enumerate(raw_fragments):
            n = int(numel)
            if n <= 0:
                continue
            fragment_name = f"{name}#frag{frag_idx}"
            param_src = mapper.zero12_partition_view(
                int(group_idx),
                int(local_offset),
                n,
                state_key=None,
            )
            exp_avg = mapper.zero12_partition_view(
                int(group_idx),
                int(local_offset),
                n,
                state_key="exp_avg",
            )
            exp_avg_sq = mapper.zero12_partition_view(
                int(group_idx),
                int(local_offset),
                n,
                state_key="exp_avg_sq",
            )
            schema.append((fragment_name, torch.Size([n]), n))
            meta.append({
                "index": int(frag_idx),
                "fragment_name": fragment_name,
                "name": str(name),
                "group_idx": int(group_idx),
                "local_offset": int(local_offset),
                "param_offset": int(param_offset),
                "numel": n,
                "shape": [int(d) for d in param.shape],
            })
            fields.append({
                "name": fragment_name,
                "param": param,
                "shape": torch.Size([n]),
                "numel": n,
                "group_idx": int(group_idx),
                "local_offset": int(local_offset),
                "param_src": None if param_src is None else param_src.detach(),
                "exp_avg": None if exp_avg is None else exp_avg.detach(),
                "exp_avg_sq": None if exp_avg_sq is None else exp_avg_sq.detach(),
            })
            fragment_to_expert[fragment_name] = eid
        if schema:
            expert_tensor_map[eid] = schema
            fragments_by_expert[eid] = meta
            field_cache_by_expert[eid] = fields

    if not expert_tensor_map:
        return None
    return ExpertPartitionPlan(
        expert_tensor_map=expert_tensor_map,
        fragments_by_expert=fragments_by_expert,
        field_cache_by_expert=field_cache_by_expert,
        fragment_to_expert=fragment_to_expert,
    )
