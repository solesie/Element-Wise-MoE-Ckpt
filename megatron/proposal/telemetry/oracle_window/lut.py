"""Independent expert/non-expert ``v -> |H|`` LUT construction.

This module reuses the active production histogram/quantile kernels, but owns
its buffers.  Consequently an oracle-window measurement cannot overwrite the
production proposal controller's quantiles or cutoff bridge state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch

from ...refresh.quantile import (
    QuantileRefs,
    launch_build_log_hist_from_bounds,
    launch_fill_quantiles_from_log_hist,
    launch_rebuild_h_lookup_lut,
    launch_reduce_positive_minmax,
    make_quantile_refs,
)
from .fragments import DOMAIN_EXPERT, DOMAIN_NONEXPERT


_LOW_PRECISION_FD_DTYPES = frozenset(("bfloat16", "float16"))


@dataclass(frozen=True)
class FiniteDifferenceAudit:
    """Observed perturbation representability for an FD anchor.

    A nominal epsilon does not guarantee that ``theta +/- epsilon*z`` differs
    after writing into BF16/FP16 model parameters.  The outer FD orchestrator
    must count actual changed low-precision elements for both signs and pass
    the result here.  A LUT built from an effectively rounded-away probe is
    rejected rather than silently labelled Hessian telemetry.
    """

    parameter_dtype: str
    epsilon: float
    compared_elements: int
    plus_changed_elements: int
    minus_changed_elements: int
    minimum_changed_fraction: float = 0.95
    estimator: str = "fd_hutchinson_diagonal"

    @property
    def plus_changed_fraction(self) -> float:
        return float(self.plus_changed_elements) / float(self.compared_elements)

    @property
    def minus_changed_fraction(self) -> float:
        return float(self.minus_changed_elements) / float(self.compared_elements)

    def validate(self) -> None:
        dtype = str(self.parameter_dtype).replace("torch.", "").lower()
        if dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError(f"unsupported FD parameter dtype {self.parameter_dtype!r}")
        if not math.isfinite(float(self.epsilon)) or float(self.epsilon) <= 0.0:
            raise ValueError("finite-difference epsilon must be finite and positive")
        total = int(self.compared_elements)
        plus = int(self.plus_changed_elements)
        minus = int(self.minus_changed_elements)
        if total <= 0 or not (0 <= plus <= total and 0 <= minus <= total):
            raise ValueError("FD changed-element counts are invalid")
        threshold = float(self.minimum_changed_fraction)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_changed_fraction must lie in [0, 1]")
        if dtype in _LOW_PRECISION_FD_DTYPES and (
            self.plus_changed_fraction < threshold
            or self.minus_changed_fraction < threshold
        ):
            raise RuntimeError(
                "low-precision finite-difference perturbation was rounded away: "
                f"dtype={dtype} eps={self.epsilon:g} "
                f"plus={self.plus_changed_fraction:.6f} "
                f"minus={self.minus_changed_fraction:.6f} "
                f"required={threshold:.6f}"
            )


@dataclass(frozen=True)
class DomainLutMetadata:
    domain: str
    local_population: int
    global_population: int
    global_h_positive: int
    global_v_positive: int
    h_min: float
    h_max: float
    v_min: float
    v_max: float
    n_quantiles: int
    n_histogram_bins: int
    n_lut_bins: int
    h_values_are_absolute: bool
    fd: FiniteDifferenceAudit

    @property
    def h_coverage(self) -> float:
        if self.global_population <= 0:
            return 0.0
        return float(self.global_h_positive) / float(self.global_population)

    @property
    def v_coverage(self) -> float:
        if self.global_population <= 0:
            return 0.0
        return float(self.global_v_positive) / float(self.global_population)


@dataclass(frozen=True, eq=False)
class DomainLut:
    """A domain-owned quantile mapping with no production-state alias."""

    domain: str
    refs: QuantileRefs
    metadata: DomainLutMetadata


def _validate_domain(domain: str) -> str:
    value = str(domain)
    if value not in (DOMAIN_EXPERT, DOMAIN_NONEXPERT):
        raise ValueError(
            f"domain must be {DOMAIN_EXPERT!r} or {DOMAIN_NONEXPERT!r}; got {value!r}"
        )
    return value


def _flat_cuda_views(
    values: Iterable[torch.Tensor],
    *,
    label: str,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    out: list[torch.Tensor] = []
    for value in values:
        if value is None or int(value.numel()) == 0:
            continue
        if value.device != device or value.device.type != "cuda":
            raise ValueError(f"{label} views must reside on {device}")
        if value.dtype != torch.float32 and not (
            label == "H" and value.dtype in (torch.float16, torch.bfloat16)
        ):
            raise ValueError(f"{label} views have unsupported dtype {value.dtype}")
        if not value.is_contiguous():
            raise ValueError(
                f"{label} views must be contiguous; refusing an implicit model-sized copy"
            )
        out.append(value.detach().view(-1))
    if not out:
        raise ValueError(f"at least one non-empty {label} view is required")
    return tuple(out)


class DomainLutBuilder:
    """Reusable, domain-private low-level histogram builder."""

    def __init__(
        self,
        *,
        domain: str,
        device: torch.device,
        n_quantiles: int = 4096,
        n_histogram_bins: int = 1024,
        n_lut_bins: int = 8192,
    ) -> None:
        self.domain = _validate_domain(domain)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DomainLutBuilder requires a CUDA device")
        if int(n_quantiles) < 2 or int(n_histogram_bins) < 2 or int(n_lut_bins) < 2:
            raise ValueError("quantile, histogram, and LUT sizes must be at least two")
        self.n_quantiles = int(n_quantiles)
        self.n_histogram_bins = int(n_histogram_bins)
        self.n_lut_bins = int(n_lut_bins)
        self.refs = make_quantile_refs(
            device=self.device,
            n_q=self.n_quantiles,
            h_lookup_lut_bins=self.n_lut_bins,
        )
        self._hist = torch.zeros(
            self.n_histogram_bins, dtype=torch.int64, device=self.device,
        )
        self._min = torch.empty((), dtype=torch.float32, device=self.device)
        self._max = torch.empty((), dtype=torch.float32, device=self.device)
        self._population = torch.empty((), dtype=torch.int64, device=self.device)

    @staticmethod
    def _all_reduce(
        tensor: torch.Tensor,
        op: torch.distributed.ReduceOp,
        process_group: object | None,
    ) -> None:
        if process_group is None:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("a process group was supplied but torch.distributed is inactive")
        torch.distributed.all_reduce(tensor, op=op, group=process_group)

    @torch.no_grad()
    def _build_quantiles(
        self,
        views: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        *,
        process_group: object | None,
    ) -> tuple[int, float, float]:
        self._min.fill_(float("inf"))
        self._max.zero_()
        for view in views:
            launch_reduce_positive_minmax(view, self._min, self._max)
        self._all_reduce(self._min, torch.distributed.ReduceOp.MIN, process_group)
        self._all_reduce(self._max, torch.distributed.ReduceOp.MAX, process_group)

        true_min = float(self._min.item())
        true_max = float(self._max.item())
        if not (math.isfinite(true_min) and math.isfinite(true_max)):
            raise RuntimeError("quantile source contains no globally positive finite values")
        if not (true_min > 0.0 and true_max > 0.0):
            raise RuntimeError("quantile source contains no globally positive values")

        # The active histogram kernel intentionally drops b == n_bins.  Pad
        # only this private scalar so the observed maximum lies in the final
        # interior bin; the reported metadata retains the unpadded range.
        padded_max = max(
            math.nextafter(true_max, math.inf),
            true_max * (1.0 + 1.0e-6),
            true_min * (1.0 + 1.0e-6),
        )
        self._max.fill_(padded_max)
        self._hist.zero_()
        for view in views:
            launch_build_log_hist_from_bounds(
                view,
                self._hist,
                min_bound=self._min,
                max_bound=self._max,
            )
        self._all_reduce(self._hist, torch.distributed.ReduceOp.SUM, process_group)
        positive = int(self._hist.sum().item())
        if positive <= 0:
            raise RuntimeError("positive quantile histogram is unexpectedly empty")
        launch_fill_quantiles_from_log_hist(
            self._hist,
            output,
            min_bound=self._min,
            max_bound=self._max,
        )
        return positive, true_min, true_max

    @torch.no_grad()
    def build(
        self,
        *,
        h_views: Iterable[torch.Tensor],
        v_views: Iterable[torch.Tensor],
        fd_audit: FiniteDifferenceAudit,
        h_values_are_absolute: bool,
        process_group: object | None = None,
    ) -> DomainLut:
        """Build one independent domain LUT from finalized FD-H and Adam-v.

        ``h_views`` must be the finalized resident ``abs(FD-Hutchinson)``
        views.  The production anchor already takes abs in place; accepting a
        false flag here would otherwise silently drop negative diagonal
        estimates in the positive-only histogram primitive.
        """

        fd_audit.validate()
        if not bool(h_values_are_absolute):
            raise ValueError(
                "DomainLut requires finalized abs(FD-Hutchinson) views; signed "
                "H views would be truncated by the positive histogram"
            )
        h = _flat_cuda_views(h_views, label="H", device=self.device)
        v = _flat_cuda_views(v_views, label="v", device=self.device)
        local_h_n = sum(int(view.numel()) for view in h)
        local_v_n = sum(int(view.numel()) for view in v)
        if local_h_n != local_v_n:
            raise ValueError(
                f"H/v population mismatch for {self.domain}: {local_h_n} != {local_v_n}"
            )

        self._population.fill_(int(local_h_n))
        self._all_reduce(
            self._population, torch.distributed.ReduceOp.SUM, process_group,
        )
        global_population = int(self._population.item())
        h_positive, h_min, h_max = self._build_quantiles(
            h, self.refs.h.values, process_group=process_group,
        )
        v_positive, v_min, v_max = self._build_quantiles(
            v, self.refs.v.values, process_group=process_group,
        )
        launch_rebuild_h_lookup_lut(
            self.refs.v.values,
            self.refs.h.values,
            self.refs.h_lut,
        )
        metadata = DomainLutMetadata(
            domain=self.domain,
            local_population=int(local_h_n),
            global_population=int(global_population),
            global_h_positive=int(h_positive),
            global_v_positive=int(v_positive),
            h_min=float(h_min),
            h_max=float(h_max),
            v_min=float(v_min),
            v_max=float(v_max),
            n_quantiles=self.n_quantiles,
            n_histogram_bins=self.n_histogram_bins,
            n_lut_bins=self.n_lut_bins,
            h_values_are_absolute=True,
            fd=fd_audit,
        )
        return DomainLut(domain=self.domain, refs=self.refs, metadata=metadata)


def make_domain_lut_builders(
    *,
    device: torch.device,
    n_quantiles: int = 4096,
    n_histogram_bins: int = 1024,
    n_lut_bins: int = 8192,
) -> tuple[DomainLutBuilder, DomainLutBuilder]:
    """Allocate distinct expert and non-expert LUT owners."""

    kwargs = dict(
        device=device,
        n_quantiles=n_quantiles,
        n_histogram_bins=n_histogram_bins,
        n_lut_bins=n_lut_bins,
    )
    return (
        DomainLutBuilder(domain=DOMAIN_EXPERT, **kwargs),
        DomainLutBuilder(domain=DOMAIN_NONEXPERT, **kwargs),
    )


__all__ = [
    "DomainLut",
    "DomainLutBuilder",
    "DomainLutMetadata",
    "FiniteDifferenceAudit",
    "make_domain_lut_builders",
]
