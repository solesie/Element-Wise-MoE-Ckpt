"""Generic histogram primitives (quantile.py) vs torch.histc oracle."""

import math

import pytest
import torch

from megatron.proposal.refresh.quantile import (
    launch_build_linear_hist,
    launch_build_log_hist,
)

_CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not _CUDA, reason="histogram kernels require CUDA")
def test_build_log_hist_matches_histc():
    torch.manual_seed(2024)
    n, n_bins = 200_000, 1024
    # lognormal positive sample (heavy tail) on GPU.
    x = torch.exp(torch.randn(n, device="cuda", dtype=torch.float32) * 2.0)
    lx = torch.log(x)
    log_lo = float(lx.min().item())
    # nudge hi a hair above the max so the max value lands in an interior bin
    # (kernel skips b == n_bins; torch.histc closes the top interval).
    log_hi = float(lx.max().item()) + 1e-4
    v_lo, v_hi = math.exp(log_lo), math.exp(log_hi)

    hist = torch.zeros(n_bins, dtype=torch.int64, device="cuda")
    launch_build_log_hist(x, hist, v_lo=v_lo, v_hi=v_hi)
    torch.cuda.synchronize()

    ref = torch.histc(lx, bins=n_bins, min=log_lo, max=log_hi).to(torch.int64)
    assert int(hist.sum().item()) == n, "every positive element must be binned"
    # bin edges computed slightly differently (our log_lo vs histc internal) →
    # allow tiny boundary migration but total mass + shape must match closely.
    max_bin_diff = int((hist - ref).abs().max().item())
    assert max_bin_diff <= 3, f"per-bin disagreement too large: {max_bin_diff}"


@pytest.mark.skipif(not _CUDA, reason="histogram kernels require CUDA")
def test_build_linear_hist_matches_histc():
    torch.manual_seed(7)
    n, n_bins = 200_000, 512
    x = torch.rand(n, device="cuda", dtype=torch.float32) * 10.0 + 0.5
    lo = float(x.min().item())
    hi = float(x.max().item()) + 1e-4

    hist = torch.zeros(n_bins, dtype=torch.int64, device="cuda")
    launch_build_linear_hist(x, hist, lo=lo, hi=hi)
    torch.cuda.synchronize()

    ref = torch.histc(x, bins=n_bins, min=lo, max=hi).to(torch.int64)
    assert int(hist.sum().item()) == n
    assert int((hist - ref).abs().max().item()) <= 3


@pytest.mark.skipif(not _CUDA, reason="histogram kernels require CUDA")
def test_production_quantile_build_matches_sort_reference():
    import math
    import statistics
    from megatron.proposal.refresh.quantile import QuantileTableBuilder as Q

    torch.manual_seed(2024)
    n, n_q, n_bins = 2_000_000, 4096, 1024
    x = torch.exp(torch.randn(n, device="cuda") * 2.0)
    out_hist = torch.empty(n_q, device="cuda", dtype=torch.float32)
    hist = torch.zeros(n_bins, device="cuda", dtype=torch.int64)
    assert Q.build_from_flat_hist(x, out_hist, hist)
    torch.cuda.synchronize()
    out_ref = torch.empty(n_q, device="cuda", dtype=torch.float32)
    Q.refresh_quantile_from_flat(x, out_ref)
    assert bool((out_hist[1:] >= out_hist[:-1]).all().item()), "monotone"
    dl = [
        abs(math.log(max(a, 1e-30)) - math.log(max(b, 1e-30)))
        for a, b in zip(out_hist.tolist(), out_ref.tolist())
    ]
    assert statistics.median(dl) < 0.01, "bulk quantiles must track sort"


@pytest.mark.skipif(not _CUDA, reason="histogram kernels require CUDA")
def test_log_hist_drops_nonpositive_and_oob():
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 1e6], device="cuda")
    hist = torch.zeros(8, dtype=torch.int64, device="cuda")
    # range [1, 16] excludes the 1e6 outlier and the non-positive entries.
    launch_build_log_hist(x, hist, v_lo=1.0, v_hi=16.0)
    torch.cuda.synchronize()
    # 1,2,4,8 are in range; -1,0 dropped (non-positive); 1e6 dropped (oob).
    assert int(hist.sum().item()) == 4
