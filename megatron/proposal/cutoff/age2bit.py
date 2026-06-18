"""Packed 2-bit age state and score LUT allocation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


AGE2BIT_CODE_VALUES = (1.0, 4.0, 16.0, 64.0)
AGE2BIT_WORD_BITS = 32
AGE2BIT_CODE_BITS = 2
AGE2BIT_CODES_PER_WORD = AGE2BIT_WORD_BITS // AGE2BIT_CODE_BITS


@dataclass
class Age2BitScoreState:
    """Persistent device state for age-based Taylor score policies."""

    age_words: torch.Tensor
    ehat_geom_lut: torch.Tensor


def allocate_age2bit_score_state(
    *,
    n_elements: int,
    device: torch.device,
    h_lut_bins: int,
) -> Age2BitScoreState:
    """Allocate packed 2-bit age state and e_hat geometry LUT once."""

    n_words = (int(n_elements) + AGE2BIT_CODES_PER_WORD - 1) // AGE2BIT_CODES_PER_WORD
    age_words = torch.zeros(n_words, dtype=torch.int32, device=device)
    ehat_geom_lut = torch.empty((4, int(h_lut_bins)), dtype=torch.float32, device=device)
    ehat_geom_lut[0].fill_(1.0)
    ehat_geom_lut[1].fill_(4.0)
    ehat_geom_lut[2].fill_(16.0)
    ehat_geom_lut[3].fill_(64.0)
    return Age2BitScoreState(
        age_words=age_words,
        ehat_geom_lut=ehat_geom_lut,
    )


__all__ = [
    "AGE2BIT_CODE_VALUES",
    "AGE2BIT_CODE_BITS",
    "AGE2BIT_CODES_PER_WORD",
    "Age2BitScoreState",
    "allocate_age2bit_score_state",
]
