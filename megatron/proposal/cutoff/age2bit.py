"""Age2Bit selection policy constants and state allocation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


AGE2BIT_CODE_VALUES = (1.0, 4.0, 16.0, 64.0)
AGE2BIT_WORD_BITS = 32
AGE2BIT_CODE_BITS = 2
AGE2BIT_CODES_PER_WORD = AGE2BIT_WORD_BITS // AGE2BIT_CODE_BITS
AGE2BIT_VBINS = 16
AGE2BIT_FLOOR_LOG_BINS = 256


@dataclass
class Age2BitFloorState:
    """Persistent device state for the Age2Bit v-bin floor policy."""

    age_words: torch.Tensor
    floor_mid: torch.Tensor
    floor_old: torch.Tensor
    floor_hist: torch.Tensor


def allocate_age2bit_floor_state(
    *,
    n_elements: int,
    device: torch.device,
    n_vbins: int = AGE2BIT_VBINS,
) -> Age2BitFloorState:
    """Allocate packed 2-bit age state and v-bin floor snapshots once."""

    n_words = (int(n_elements) + AGE2BIT_CODES_PER_WORD - 1) // AGE2BIT_CODES_PER_WORD
    age_words = torch.zeros(n_words, dtype=torch.int32, device=device)
    floor_mid = torch.zeros(int(n_vbins), dtype=torch.float32, device=device)
    floor_old = torch.zeros(int(n_vbins), dtype=torch.float32, device=device)
    floor_hist = torch.zeros(
        (int(n_vbins), AGE2BIT_FLOOR_LOG_BINS),
        dtype=torch.int32,
        device=device,
    )
    return Age2BitFloorState(
        age_words=age_words,
        floor_mid=floor_mid,
        floor_old=floor_old,
        floor_hist=floor_hist,
    )


__all__ = [
    "AGE2BIT_CODE_VALUES",
    "AGE2BIT_CODE_BITS",
    "AGE2BIT_CODES_PER_WORD",
    "AGE2BIT_VBINS",
    "AGE2BIT_FLOOR_LOG_BINS",
    "Age2BitFloorState",
    "allocate_age2bit_floor_state",
]
