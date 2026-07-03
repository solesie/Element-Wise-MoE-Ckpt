"""Local expert partition schema for BF16/ZeRO optimizer layouts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

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
