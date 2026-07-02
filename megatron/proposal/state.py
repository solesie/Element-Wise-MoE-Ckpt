"""Runtime state objects for lossy checkpoint selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class BF16OptGroupView:
    """View over one DeepSpeed BF16 optimizer flat group."""

    m_flat: torch.Tensor
    v_flat: torch.Tensor
    p_flat: torch.Tensor
    master_off_elem: int
    theta_cp_flat: Optional[torch.Tensor] = None
    score_linear_flat: Optional[torch.Tensor] = None
    name_start: int = 0
    name_end: int = 0
    entry_elem_start: Optional[torch.Tensor] = None
    entry_group_start: Optional[torch.Tensor] = None
    entry_group_count: Optional[torch.Tensor] = None
    entry_name_idx: Optional[torch.Tensor] = None
    entry_block_entry_idx: Optional[torch.Tensor] = None
    entry_block_local_group_start: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ExpertParamIndex:
    """Immutable index metadata for expert parameters."""

    bf16_opt_groups: List[BF16OptGroupView]
    tname_list: List[str]
    tname_to_expert: Dict[str, Tuple[int, int]]
    elem_offsets: torch.Tensor
    group_offsets: torch.Tensor
    n_total_elements: int
    n_total_groups: int
    n_experts: int


@dataclass
class CutoffSearchScratch:
    """Persistent scratch for device-side cutoff state."""

    hist: torch.Tensor
    smin: torch.Tensor
    smax: torch.Tensor
    total_th: torch.Tensor


@dataclass
class LossyCpState:
    """All mutable state needed by one lossy checkpoint iteration."""

    index: ExpertParamIndex
    quantiles: "QuantileRefs"
    scratch: CutoffSearchScratch
    age2bit: Optional["Age2BitScoreState"] = None
    theta_cp: Optional["ThetaCpScoreState"] = None


from .refresh.quantile import QuantileRefs  # noqa: E402  (typing cycle only)
from .cutoff.age2bit import Age2BitScoreState  # noqa: E402  (typing cycle only)
from .cutoff.theta_cp import ThetaCpScoreState  # noqa: E402  (typing cycle only)
