"""Persistent 2-byte checkpoint-value state for residual scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ThetaCpScoreState:
    """Device-resident theta_cp snapshot used when e_hat proxy is disabled."""

    values: torch.Tensor


def allocate_theta_cp_score_state(
    *,
    n_elements: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> ThetaCpScoreState:
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"theta_cp dtype must be 2-byte fp, got {dtype}")
    values = torch.empty(int(n_elements), dtype=dtype, device=device)
    return ThetaCpScoreState(values=values)


__all__ = ["ThetaCpScoreState", "allocate_theta_cp_score_state"]
