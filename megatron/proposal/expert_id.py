"""Expert layer/id parsing shared by proposal storage and recovery."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Tuple

if TYPE_CHECKING:
    import torch


LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
PIPELINE_LAYER_RE = re.compile(r"^(?:module\.)?(\d+)\.")
DS_EXPERT_IDX_RE = re.compile(r"deepspeed_experts\.(\d+)\.")


ExpertId = Tuple[int, int]
ExpertIdMap = Mapping[str, ExpertId]


def layer_from_name(name: str) -> int:
    """Return a non-pipeline model-layer id encoded in a parameter name.

    DeepSpeed PipelineModule's leading number is an execution-list position,
    so it is deliberately rejected here and must be resolved from the model
    object by :func:`build_expert_id_map`.
    """

    m_l = LAYER_RE.search(name)
    if m_l is not None:
        return int(m_l.group(1))
    if PIPELINE_LAYER_RE.match(name) is not None:
        raise ValueError(
            "PipelineModule parameter names contain an execution-list index, "
            "not a model layer id; build and pass the canonical expert-id map"
        )
    return 0


def ids_from_name(
    name: str,
    expert_id_map: Optional[ExpertIdMap] = None,
) -> Optional[ExpertId]:
    m_e = DS_EXPERT_IDX_RE.search(name)
    if m_e is None:
        return None
    if expert_id_map is not None:
        try:
            layer, expert = expert_id_map[name]
        except KeyError as exc:
            raise KeyError(
                f"expert parameter {name!r} is missing from the canonical "
                "expert-id map"
            ) from exc
        return (int(layer), int(expert))
    return (layer_from_name(name), int(m_e.group(1)))


def build_expert_id_map(
    model: "torch.nn.Module",
    *,
    num_layers: int,
) -> Dict[str, ExpertId]:
    """Resolve every expert parameter to its canonical model-layer ordinal.

    DeepSpeed registers a pipeline layer under its position in the complete
    execution list (for example ``"5"``), while the instantiated layer keeps
    the model-layer ordinal in ``layer_number``.  Storage keys must use the
    latter so their range remains exactly ``0..num_layers-1`` for every PP
    partition.
    """

    num_layers = int(num_layers)
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")

    modules = dict(model.named_modules())
    out: Dict[str, ExpertId] = {}
    for name, _param in model.named_parameters():
        expert_match = DS_EXPERT_IDX_RE.search(name)
        if expert_match is None:
            continue

        layer_match = LAYER_RE.search(name)
        if layer_match is not None:
            layer = int(layer_match.group(1))
        else:
            pipeline_match = PIPELINE_LAYER_RE.match(name)
            if pipeline_match is None:
                layer = 0
            else:
                module_name = name[:pipeline_match.end() - 1]
                layer_module = modules.get(module_name)
                if layer_module is None or not hasattr(layer_module, "layer_number"):
                    raise RuntimeError(
                        f"cannot resolve canonical model layer for pipeline "
                        f"expert parameter {name!r}: module {module_name!r} "
                        "has no layer_number"
                    )
                layer = int(layer_module.layer_number)

        if not 0 <= layer < num_layers:
            raise RuntimeError(
                f"expert parameter {name!r} resolved to model layer {layer}, "
                f"outside [0, {num_layers})"
            )
        out[name] = (layer, int(expert_match.group(1)))
    return out


__all__ = [
    "DS_EXPERT_IDX_RE",
    "ExpertId",
    "ExpertIdMap",
    "LAYER_RE",
    "PIPELINE_LAYER_RE",
    "build_expert_id_map",
    "ids_from_name",
    "layer_from_name",
]
