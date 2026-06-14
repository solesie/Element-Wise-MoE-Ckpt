"""Quantile tables and log-domain H lookup LUT for scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Generic histogram primitives (§4.3.A).
#
# These bin a *flat* positive tensor — used at the anchor boundary for v_q
# over v_t, the anchor Stage 3 h_q build over |grad_flat|, and as the building
# block the recursive quantile/cutoff search reuses. Direct cutoff/emission
# kernels consume the same age2bit/vbin element-wise score helper.
#
# ``out_hist`` is **i64** (100B+ MoE / multi-GPU all-reduce overflow safety).
# Non-positive values are dropped (Adam v_t ≥ 0; H assumed positive curvature).
# Out-of-range bins are skipped, so the caller fixes total count via the live
# range, not by piling mass into the edge bins.
# ---------------------------------------------------------------------------
@triton.jit
def reduce_positive_minmax_kernel(
    src_ptr, n_elems, min_ptr, max_ptr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    x = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    positive = mask & (x > 0.0)
    block_min = tl.min(tl.where(positive, x, float("inf")), axis=0)
    block_max = tl.max(tl.where(positive, x, 0.0), axis=0)
    tl.atomic_min(min_ptr, block_min, sem="relaxed")
    tl.atomic_max(max_ptr, block_max, sem="relaxed")


@triton.jit
def build_log_hist_from_bounds_kernel(
    src_ptr, n_elems, hist_ptr, n_bins, min_ptr, max_ptr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    x = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v_lo = tl.load(min_ptr)
    v_hi = tl.load(max_ptr)
    valid_bounds = (v_hi > v_lo) & (v_lo > 0.0)
    log_lo = tl.log(v_lo)
    inv_log_width = n_bins / (tl.log(v_hi + 1.0e-30) - log_lo + 1.0e-30)
    positive = mask & (x > 0.0) & valid_bounds
    lx = tl.log(tl.where(positive, x, 1.0))
    b = ((lx - log_lo) * inv_log_width).to(tl.int32)
    valid = positive & (b >= 0) & (b < n_bins)
    b_clamped = tl.maximum(tl.minimum(b, n_bins - 1), 0)
    tl.atomic_add(hist_ptr + b_clamped, 1, mask=valid, sem="relaxed")


@triton.jit
def fill_quantiles_from_log_hist_kernel(
    hist_ptr,
    out_ptr,
    n_bins: tl.constexpr,
    n_q: tl.constexpr,
    min_ptr,
    max_ptr,
    BLOCK_BINS: tl.constexpr,
):
    qid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_BINS)
    mask = offs < n_bins
    counts = tl.load(hist_ptr + offs, mask=mask, other=0).to(tl.float32)
    total = tl.sum(counts, axis=0)
    target = (qid + 1.0) * total / (n_q + 1.0)
    cum = tl.cumsum(counts, axis=0)
    crossing_mask = (cum >= target) & mask
    crossing = tl.min(tl.where(crossing_mask, offs, n_bins), axis=0)
    count_at = tl.sum(tl.where(offs == crossing, counts, 0.0), axis=0)
    prev_cum = tl.sum(tl.where(offs < crossing, counts, 0.0), axis=0)
    frac = (target - prev_cum) / tl.maximum(count_at, 1.0)
    v_lo = tl.load(min_ptr)
    v_hi = tl.load(max_ptr)
    log_lo = tl.log(v_lo)
    log_hi = tl.log(v_hi + 1.0e-30) + 1.0e-6
    width = (log_hi - log_lo) / n_bins
    value = tl.exp(log_lo + (crossing.to(tl.float32) + frac) * width)
    valid = (total > 0.0) & (v_hi > v_lo) & (v_lo > 0.0)
    tl.store(out_ptr + qid, tl.where(valid, value, 1.0))


@triton.jit
def rebuild_h_lookup_lut_kernel(
    v_q_ptr,
    h_q_ptr,
    lut_ptr,
    log_v_min_ptr,
    inv_log_v_width_ptr,
    n_q: tl.constexpr,
    n_lut: tl.constexpr,
    log_n_q: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_lut

    v_min = tl.maximum(tl.load(v_q_ptr + 0).to(tl.float32), 1.0e-30)
    v_max = tl.maximum(tl.load(v_q_ptr + n_q - 1).to(tl.float32), v_min)
    log_min = tl.log(v_min)
    log_max = tl.log(v_max)
    width = log_max - log_min
    inv_width = tl.where(width > 1.0e-12, (n_lut - 1) / width, 0.0)
    tl.store(log_v_min_ptr, log_min)
    tl.store(inv_log_v_width_ptr, inv_width)

    frac_lut = offs.to(tl.float32) / tl.maximum((n_lut - 1), 1)
    v_grid = tl.exp(log_min + frac_lut * width)

    idx = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in tl.static_range(log_n_q):
        step = n_q >> (i + 1)
        mid = idx + step
        mid_clamped = tl.minimum(mid, n_q - 1)
        q_mid = tl.load(v_q_ptr + mid_clamped)
        idx = tl.where(v_grid >= q_mid, mid, idx)
    idx = tl.maximum(tl.minimum(idx, n_q - 2), 0)
    v_lo = tl.load(v_q_ptr + idx)
    v_hi = tl.load(v_q_ptr + idx + 1)
    h_lo = tl.load(h_q_ptr + idx)
    h_hi = tl.load(h_q_ptr + idx + 1)
    alpha = (v_grid - v_lo) / (v_hi - v_lo + 1.0e-30)
    h = h_lo + tl.maximum(tl.minimum(alpha, 1.0), 0.0) * (h_hi - h_lo)
    h_same = tl.load(h_q_ptr + n_q - 1)
    tl.store(lut_ptr + offs, tl.where(inv_width > 0.0, h, h_same), mask=mask)


@triton.jit
def build_log_hist_kernel(
    src_ptr, n_elems, hist_ptr, n_bins,
    log_lo, inv_log_width,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    x = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    pos = x > 0.0
    lx = tl.log(tl.where(pos, x, 1.0))
    b = ((lx - log_lo) * inv_log_width).to(tl.int32)
    valid = mask & pos & (b >= 0) & (b < n_bins)
    b_clamped = tl.maximum(tl.minimum(b, n_bins - 1), 0)
    tl.atomic_add(hist_ptr + b_clamped, 1, mask=valid, sem="relaxed")


@triton.jit
def build_linear_hist_kernel(
    src_ptr, n_elems, hist_ptr, n_bins,
    lin_lo, inv_lin_width,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    x = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    pos = x > 0.0
    b = ((x - lin_lo) * inv_lin_width).to(tl.int32)
    valid = mask & pos & (b >= 0) & (b < n_bins)
    b_clamped = tl.maximum(tl.minimum(b, n_bins - 1), 0)
    tl.atomic_add(hist_ptr + b_clamped, 1, mask=valid, sem="relaxed")


_HIST_BLOCK = 1024


@torch.no_grad()
def launch_reduce_positive_minmax(
    src: torch.Tensor, min_out: torch.Tensor, max_out: torch.Tensor,
) -> None:
    n = int(src.numel())
    if n == 0:
        return
    grid = ((n + _HIST_BLOCK - 1) // _HIST_BLOCK,)
    reduce_positive_minmax_kernel[grid](
        src, n, min_out, max_out, BLOCK=_HIST_BLOCK,
    )


@torch.no_grad()
def launch_build_log_hist_from_bounds(
    src: torch.Tensor,
    hist: torch.Tensor,
    *,
    min_bound: torch.Tensor,
    max_bound: torch.Tensor,
) -> None:
    n = int(src.numel())
    n_bins = int(hist.numel())
    if n == 0 or n_bins == 0:
        return
    grid = ((n + _HIST_BLOCK - 1) // _HIST_BLOCK,)
    build_log_hist_from_bounds_kernel[grid](
        src, n, hist, n_bins, min_bound, max_bound, BLOCK=_HIST_BLOCK,
    )


@torch.no_grad()
def launch_fill_quantiles_from_log_hist(
    hist: torch.Tensor,
    out: torch.Tensor,
    *,
    min_bound: torch.Tensor,
    max_bound: torch.Tensor,
) -> None:
    n_bins = int(hist.numel())
    n_q = int(out.numel())
    if n_bins == 0 or n_q == 0:
        return
    block_bins = 1 << (n_bins - 1).bit_length()
    fill_quantiles_from_log_hist_kernel[(n_q,)](
        hist, out, n_bins, n_q, min_bound, max_bound,
        BLOCK_BINS=block_bins,
    )


@torch.no_grad()
def launch_rebuild_h_lookup_lut(
    v_q: torch.Tensor,
    h_q: torch.Tensor,
    h_lut: "HLookupLut",
) -> None:
    """Rebuild the preallocated log(v)->H LUT without allocating tensors."""

    n_q = int(v_q.numel())
    n_lut = int(h_lut.values.numel())
    if n_q < 2 or n_lut <= 0:
        return
    block = 256
    grid = ((n_lut + block - 1) // block,)
    rebuild_h_lookup_lut_kernel[grid](
        v_q,
        h_q,
        h_lut.values,
        h_lut.log_v_min,
        h_lut.inv_log_v_width,
        n_q,
        n_lut,
        int(n_q).bit_length() - 1,
        BLOCK=block,
    )


@torch.no_grad()
def launch_build_log_hist(
    src: torch.Tensor, hist: torch.Tensor, *, v_lo: float, v_hi: float,
) -> None:
    """Fill ``hist`` (i64, pre-zeroed by caller) with the log-bin histogram of
    positive ``src`` over ``[v_lo, v_hi]``. No allocation: ``hist`` is
    caller-owned scratch reused across passes (§0 R2)."""

    import math
    n = int(src.numel())
    n_bins = int(hist.numel())
    if n == 0 or n_bins == 0 or not (v_hi > v_lo > 0.0):
        return
    log_lo = math.log(v_lo)
    inv_log_width = n_bins / (math.log(v_hi) - log_lo)
    grid = ((n + _HIST_BLOCK - 1) // _HIST_BLOCK,)
    build_log_hist_kernel[grid](
        src, n, hist, n_bins,
        float(log_lo), float(inv_log_width),
        BLOCK=_HIST_BLOCK,
    )


@torch.no_grad()
def launch_build_linear_hist(
    src: torch.Tensor, hist: torch.Tensor, *, lo: float, hi: float,
) -> None:
    """Fill ``hist`` (i64, pre-zeroed) with the linear-bin histogram of positive
    ``src`` over ``[lo, hi]``. No allocation."""

    n = int(src.numel())
    n_bins = int(hist.numel())
    if n == 0 or n_bins == 0 or not (hi > lo):
        return
    inv_lin_width = n_bins / (hi - lo)
    grid = ((n + _HIST_BLOCK - 1) // _HIST_BLOCK,)
    build_linear_hist_kernel[grid](
        src, n, hist, n_bins,
        float(lo), float(inv_lin_width),
        BLOCK=_HIST_BLOCK,
    )


@dataclass(frozen=True)
class QuantileTable:
    """Monotone quantile table."""

    values: torch.Tensor
    n_q: int


@dataclass
class HLookupLut:
    """Runtime log(v)->H lookup table used by score kernels."""

    values: torch.Tensor
    log_v_min: torch.Tensor
    inv_log_v_width: torch.Tensor


@dataclass
class QuantileRefs:
    """Current v/H quantile tables and the derived runtime lookup."""

    v: QuantileTable
    h: QuantileTable
    h_lut: HLookupLut


class QuantileTableBuilder:
    """Build quantile tables and the runtime H lookup LUT."""

    @staticmethod
    @torch.no_grad()
    def refresh_quantile_from_flat(
        flat: torch.Tensor,
        out: torch.Tensor,
        *,
        drop_nonpositive: bool = True,
    ) -> bool:
        """Fill ``out`` with quantiles from ``flat``.

        This reference builder is used at refresh/anchor boundaries only.
        The production hot path consumes the prebuilt table/LUT and performs
        no per-element sort.
        """

        if out.ndim != 1:
            raise ValueError("quantile output must be a 1-D tensor")
        src = flat.detach().reshape(-1).to(torch.float32)
        if drop_nonpositive:
            src = src[src > 0]
        if src.numel() == 0:
            return False
        sorted_src = torch.sort(src).values
        if out.numel() == 1:
            out.copy_(sorted_src[-1:])
            return True
        pos = torch.linspace(
            0,
            sorted_src.numel() - 1,
            out.numel(),
            device=sorted_src.device,
            dtype=torch.float32,
        )
        lo = torch.floor(pos).to(torch.long)
        hi = torch.clamp(lo + 1, max=sorted_src.numel() - 1)
        frac = pos - lo.to(torch.float32)
        vals = sorted_src[lo] * (1.0 - frac) + sorted_src[hi] * frac
        out.copy_(vals.to(out.dtype))
        return True

    @staticmethod
    @torch.no_grad()
    def build_from_flat_hist(
        flat: torch.Tensor,
        out: torch.Tensor,
        hist: torch.Tensor,
        *,
        v_lo: Optional[float] = None,
        v_hi: Optional[float] = None,
    ) -> bool:
        """Production quantile build: log-bin histogram → cumsum → per-quantile
        lerp. No sort (§5.1 removes the sort path). ``hist`` (i64) is
        caller-owned scratch; ``out`` (n_q,) is overwritten in place.

        This is the anchor-boundary builder for v_q over v_t and h_q over
        ``|grad_flat|``. The histogram-quantile is exact to
        within one log-bin width; tail precision matches the sort reference at
        N_Q=4096 over a 1024-bin log grid for lognormal-tailed distributions.
        """

        import math
        src = flat.detach().reshape(-1).to(torch.float32)
        n_bins = int(hist.numel())
        n_q = int(out.numel())
        if src.numel() == 0 or n_bins == 0 or n_q == 0:
            return False
        if v_lo is None or v_hi is None:
            pos = src[src > 0]
            if pos.numel() == 0:
                return False
            v_lo = float(pos.min().item()) if v_lo is None else v_lo
            v_hi = float(pos.max().item()) if v_hi is None else v_hi
        if not (v_hi > v_lo > 0.0):
            out.fill_(float(v_hi if v_hi and v_hi > 0 else 1.0))
            return True
        # Pad hi so the max value bins interior (kernel skips b == n_bins).
        log_lo = math.log(v_lo)
        log_hi = math.log(v_hi) + 1e-6
        width = (log_hi - log_lo) / n_bins
        hist.zero_()
        launch_build_log_hist(src, hist, v_lo=v_lo, v_hi=math.exp(log_hi))
        cum = torch.cumsum(hist, 0).to(torch.float64)
        total = float(cum[-1].item())
        if total <= 0.0:
            return False
        device = out.device
        qpos = (
            (torch.arange(n_q, device=device, dtype=torch.float64) + 0.5)
            / n_q
        ) * total
        b = torch.searchsorted(cum, qpos, right=False).clamp_(max=n_bins - 1)
        cum_below = torch.where(
            b > 0,
            cum.index_select(0, (b - 1).clamp_(min=0)),
            torch.zeros_like(qpos),
        )
        hb = hist.to(torch.float64).index_select(0, b).clamp_(min=1.0)
        frac = ((qpos - cum_below) / hb).clamp_(0.0, 1.0)
        vq = torch.exp(log_lo + (b.to(torch.float64) + frac) * width)
        out.copy_(torch.cummax(vq, 0).values.to(out.dtype))  # enforce monotone
        return True

    @staticmethod
    @torch.no_grad()
    def rebuild_h_lookup_lut(
        v_q: torch.Tensor,
        h_q: torch.Tensor,
        h_lut: HLookupLut,
        *,
        eps: float = 1e-30,
    ) -> bool:
        """Rebuild the log-domain LUT from v/H quantile tables."""

        if v_q.numel() != h_q.numel():
            raise ValueError("v_q and h_q must have the same length")
        if h_lut.values.ndim != 1:
            raise ValueError("h_lut.values must be 1-D")
        v = v_q.detach().to(torch.float32)
        h = h_q.detach().to(torch.float32)
        positive = v > 0
        if not bool(positive.any()):
            return False
        v_pos = v[positive]
        h_pos = h[positive]
        log_v_min = torch.log(torch.clamp(v_pos[0], min=eps))
        log_v_max = torch.log(torch.clamp(v_pos[-1], min=eps))
        if not torch.isfinite(log_v_min) or not torch.isfinite(log_v_max):
            return False
        if float((log_v_max - log_v_min).abs().item()) < 1e-12:
            h_lut.values.fill_(float(h_pos[-1].item()))
            h_lut.log_v_min.fill_(float(log_v_min.item()))
            h_lut.inv_log_v_width.fill_(0.0)
            return True

        bins = h_lut.values.numel()
        grid = torch.linspace(
            float(log_v_min.item()),
            float(log_v_max.item()),
            bins,
            device=v.device,
            dtype=torch.float32,
        )
        v_grid = torch.exp(grid)
        idx_hi = torch.searchsorted(v_pos, v_grid, right=False)
        idx_hi = torch.clamp(idx_hi, min=1, max=v_pos.numel() - 1)
        idx_lo = idx_hi - 1
        v_lo = v_pos[idx_lo]
        v_hi = v_pos[idx_hi]
        denom = torch.clamp(v_hi - v_lo, min=eps)
        frac = torch.clamp((v_grid - v_lo) / denom, 0.0, 1.0)
        lut = h_pos[idx_lo] * (1.0 - frac) + h_pos[idx_hi] * frac
        h_lut.values.copy_(lut.to(h_lut.values.dtype))
        h_lut.log_v_min.fill_(float(log_v_min.item()))
        h_lut.inv_log_v_width.fill_(
            float((bins - 1) / max(float((log_v_max - log_v_min).item()), eps))
        )
        return True


def make_quantile_refs(
    *,
    device: torch.device,
    n_q: int,
    h_lookup_lut_bins: int,
    v_init: Optional[torch.Tensor] = None,
    h_init: Optional[torch.Tensor] = None,
) -> QuantileRefs:
    """Allocate small persistent quantile/LUT buffers."""

    v_values = torch.empty(n_q, dtype=torch.float32, device=device)
    h_values = torch.empty(n_q, dtype=torch.float32, device=device)
    if v_init is not None:
        v_values.copy_(v_init.to(device=device, dtype=torch.float32))
    else:
        v_values.fill_(1.0)
    if h_init is not None:
        h_values.copy_(h_init.to(device=device, dtype=torch.float32))
    else:
        h_values.fill_(1.0)
    lut = HLookupLut(
        values=torch.empty(h_lookup_lut_bins, dtype=torch.float32, device=device),
        log_v_min=torch.zeros((), dtype=torch.float32, device=device),
        inv_log_v_width=torch.zeros((), dtype=torch.float32, device=device),
    )
    QuantileTableBuilder.rebuild_h_lookup_lut(v_values, h_values, lut)
    return QuantileRefs(
        v=QuantileTable(values=v_values, n_q=n_q),
        h=QuantileTable(values=h_values, n_q=n_q),
        h_lut=lut,
    )
