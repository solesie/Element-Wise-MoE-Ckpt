"""Expert layer/id parsing shared by proposal storage and recovery."""

from __future__ import annotations

import re
from typing import Optional, Tuple


LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
PIPELINE_LAYER_RE = re.compile(r"^(?:module\.)?(\d+)\.")
DS_EXPERT_IDX_RE = re.compile(r"deepspeed_experts\.(\d+)\.")


def layer_from_name(name: str) -> int:
    """Return the global transformer layer id encoded in a parameter name.

    Megatron modules usually expose ``...layers.N...`` names, while DeepSpeed
    ``PipelineModule`` exposes stage-local modules as leading numeric names
    such as ``9.mlp...``.  Treat the latter as authoritative only at the start
    of the name so ``deepspeed_experts.0`` is never mistaken for a layer.
    """

    m_l = LAYER_RE.search(name)
    if m_l is not None:
        return int(m_l.group(1))
    m_p = PIPELINE_LAYER_RE.match(name)
    if m_p is not None:
        return int(m_p.group(1))
    return 0


def ids_from_name(name: str) -> Optional[Tuple[int, int]]:
    m_e = DS_EXPERT_IDX_RE.search(name)
    if m_e is None:
        return None
    return (layer_from_name(name), int(m_e.group(1)))


__all__ = [
    "DS_EXPERT_IDX_RE",
    "LAYER_RE",
    "PIPELINE_LAYER_RE",
    "ids_from_name",
    "layer_from_name",
]
