"""Persistent descriptor tensor construction for lossy checkpoint kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def device_tensor_from_pinned_host(
    values: Iterable[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    values_list = [int(v) for v in values]
    if device.type != "cuda":
        return torch.tensor(values_list, dtype=dtype, device=device)
    host = torch.empty(len(values_list), dtype=dtype, pin_memory=True)
    if values_list:
        np_dtype = np.int64 if dtype in (torch.int64, torch.long) else np.int32
        host.numpy()[:] = np.asarray(values_list, dtype=np_dtype)
    return host.to(device=device, non_blocking=True)


@dataclass(frozen=True)
class StaticDescriptorToken:
    dtype: torch.dtype
    start: int
    length: int


class StaticDescriptorArena:
    """Bulk-build persistent integer descriptor tensors with pinned H2D sources."""

    def __init__(self, *, device: torch.device) -> None:
        self._device = device
        self._i64_values: List[int] = []
        self._i32_values: List[int] = []
        self._device_by_dtype: Dict[torch.dtype, torch.Tensor] = {}
        self._host_refs: List[torch.Tensor] = []

    @property
    def host_refs(self) -> Tuple[torch.Tensor, ...]:
        return tuple(self._host_refs)

    def add(self, values: Iterable[int], *, dtype: torch.dtype) -> StaticDescriptorToken:
        values_list = [int(v) for v in values]
        if dtype in (torch.int64, torch.long):
            start = len(self._i64_values)
            self._i64_values.extend(values_list)
            dtype = torch.int64
        elif dtype == torch.int32:
            start = len(self._i32_values)
            self._i32_values.extend(values_list)
        else:
            raise TypeError(f"unsupported descriptor dtype: {dtype}")
        return StaticDescriptorToken(dtype=dtype, start=start, length=len(values_list))

    def materialize(self) -> None:
        self._materialize_dtype(torch.int64, self._i64_values)
        self._materialize_dtype(torch.int32, self._i32_values)

    def tensor(self, token: StaticDescriptorToken) -> torch.Tensor:
        if token.length == 0:
            return torch.empty(0, dtype=token.dtype, device=self._device)
        return self._device_by_dtype[token.dtype].narrow(
            0, int(token.start), int(token.length),
        )

    def _materialize_dtype(
        self,
        dtype: torch.dtype,
        values: List[int],
    ) -> None:
        if not values:
            return
        if self._device.type != "cuda":
            self._device_by_dtype[dtype] = torch.tensor(
                values, dtype=dtype, device=self._device,
            )
            return
        host = torch.empty(len(values), dtype=dtype, pin_memory=True)
        np_dtype = np.int64 if dtype == torch.int64 else np.int32
        host.numpy()[:] = np.asarray(values, dtype=np_dtype)
        dev = torch.empty(len(values), dtype=dtype, device=self._device)
        dev.copy_(host, non_blocking=True)
        self._host_refs.append(host)
        self._device_by_dtype[dtype] = dev
