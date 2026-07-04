"""Small step-level helpers for lossy checkpoint orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..expert_id import ids_from_name
from ..refresh.fp32_state import get_mapper


@torch.no_grad()
def global_selection_element_count(
    local_n_elements: int,
    scratch: torch.Tensor,
    *,
    process_group=None,
    world_size: int = 1,
) -> int:
    if (
        int(world_size) > 1
        and process_group is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        scratch.fill_(int(local_n_elements))
        torch.distributed.all_reduce(
            scratch,
            op=torch.distributed.ReduceOp.SUM,
            group=process_group,
        )
        return int(scratch.item())
    return int(local_n_elements)


def fp32_v_t(
    optimizer: torch.optim.Optimizer, param: torch.nn.Parameter
) -> Optional[torch.Tensor]:
    mapper = get_mapper()
    if mapper is not None:
        return mapper.get_state(param, "exp_avg_sq")
    return optimizer.state.get(param, {}).get("exp_avg_sq")


def fp32_param(param: torch.nn.Parameter) -> torch.Tensor:
    mapper = get_mapper()
    if mapper is not None:
        return mapper.get_param(param)
    return param.data.to(torch.float32)


@dataclass
class LossyStepStats:
    step: int
    total_tracked: int
    k: int
    topk_ratio: float
    max_A: float
    max_delta: float
    score_mean: float
    score_max: float
    selected_bytes_sparse: int
    full_save_bytes: int
    full_save_numel: int
    compute_time_ms: float
