"""v_t vs FD-Hessian probe.

Measures distribution + ranking similarity between Adam ``v_t`` (FP32,
the optimizer's exp_avg_sq, used in-place — no copy) and a high-quality
FD finite-difference Hessian reference (FP16/BF16 group accumulator,
``g_blk=8``, ``ref_n_batches=20`` minibatches × ``ref_k=32`` K=1 calls).

Goal: determine whether ``v_t`` — at zero compute cost — can replace
the FD-H reference if first re-scaled into the FD axis. The probe
captures only the metrics needed for that question:

  * **Distribution** of |H_ref| and v_t (per-group):
      mean / median (p50) / p95 / p99 / p99.9 / max + log-domain
      mean, std, skew, excess-kurtosis.
  * **Ranking similarity** v_t vs |H_ref|:
      Spearman ρ (all + top-{1,3,5,10}%), Pearson on top-1% values,
      Jaccard at {1,3,5,10}% top-sets, top-bucket retention (M=10).

Spearman = Pearson on integer ranks (ties broken by argsort order).
Implementation uses ``argsort`` once per tensor and reuses the cached
ranks for top-X% subsets — verified equivalent to ``scipy.stats.spearmanr``.

All large tensor ops (sort / argsort / topk / set-intersection) execute
on CUDA when available; the CPU fallback is provided for environments
without CUDA only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


_TOP_PCTS: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10)

# Module-level stash for H_prev_base distribution comparison (cross-iter
# H stability). Holds the prior probe's H_base sorted-|values| tensor
# on host so a 188M GPU buffer is not pinned across iters.
_PREV_H_BASE_SORTED: Optional[torch.Tensor] = None


def _capture_batches(data_iterator: Any, n: int) -> List[Tuple[Any, ...]]:
    """Capture ``n`` consecutive batches via pretrain_gpt's get_batch."""
    from pretrain_gpt import get_batch as _get_batch_gpt
    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    return [_get_batch_gpt(di) for _ in range(int(n))]


def _flatten_h(h: List[torch.Tensor]) -> torch.Tensor:
    """Concat per-tensor diag-|H| list → 1-D FP32 tensor on the original
    device (GPU if available). Keeping the result on-device removes the
    188M-element D2H copy that dominated the previous CPU-pipeline."""
    return torch.cat(
        [t.detach().to(torch.float32).reshape(-1) for t in h], dim=0,
    )


def _mean_list(hs: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Elementwise mean of a list of per-tensor lists (FP32)."""
    out: List[torch.Tensor] = []
    for tens_per_call in zip(*hs):
        stack = torch.stack(
            [t.detach().to(torch.float32) for t in tens_per_call], dim=0,
        )
        out.append(stack.mean(dim=0))
    return out


def _jaccard_isin(idx_a: torch.Tensor, idx_b: torch.Tensor) -> float:
    """Jaccard via torch.isin on the original device (GPU if available)."""
    inter = int(torch.isin(idx_a, idx_b).sum().item())
    union = int(idx_a.numel()) + int(idx_b.numel()) - inter
    return inter / max(1, union)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson r on 1-D tensors (FP64 numerics). Runs on the tensor's
    current device — call on GPU tensors to avoid host fallback."""
    a64 = a.to(torch.float64); b64 = b.to(torch.float64)
    if a64.numel() < 2:
        return float("nan")
    a_c = a64 - a64.mean(); b_c = b64 - b64.mean()
    denom = float((a_c.norm() * b_c.norm()).item())
    if denom == 0.0:
        return float("nan")
    return float((a_c * b_c).sum().item() / denom)


def _ranks_from_argsort(argsort_desc: torch.Tensor, n: int) -> torch.Tensor:
    """Integer ranks from argsort(descending=True). On the same device
    as the argsort — scatter on GPU is ~10× faster than on CPU at 188M."""
    ranks = torch.empty(n, dtype=torch.long, device=argsort_desc.device)
    ranks[argsort_desc] = torch.arange(n, dtype=torch.long, device=argsort_desc.device)
    return ranks


def _spearman(rank_a: torch.Tensor, rank_b: torch.Tensor) -> float:
    """Spearman ρ ≡ Pearson on ranks (verified vs scipy)."""
    return _pearson(rank_a.to(torch.float64), rank_b.to(torch.float64))


def _distribution_stats(
    flat_abs: torch.Tensor, prefix: str, metrics: Dict[str, float],
) -> None:
    """Quantiles + log-domain shape (mean / std / skew / excess-kurt).
    All ops run on the input tensor's device — only the final scalars
    are pulled to host."""
    n = int(flat_abs.numel())
    if n == 0:
        return
    sorted_v = flat_abs.sort().values  # stays on input device
    for name, q in (("p1", 0.01), ("p50", 0.5), ("p95", 0.95),
                    ("p99", 0.99), ("p99.9", 0.999)):
        idx = max(0, min(n - 1, int(round(q * (n - 1)))))
        metrics[f"{prefix}_{name}"] = float(sorted_v[idx].item())
    metrics[f"{prefix}_max"] = float(sorted_v[-1].item())
    metrics[f"{prefix}_mean"] = float(flat_abs.to(torch.float64).mean().item())
    # Log-domain shape on-device (FP64 reduces overflow risk)
    pos = sorted_v[sorted_v > 0]
    if pos.numel() >= 2:
        log_v = torch.log(pos.to(torch.float64))
        m = log_v.mean(); s = log_v.std()
        metrics[f"{prefix}_log_mean"] = float(m.item())
        metrics[f"{prefix}_log_std"] = float(s.item())
        if float(s.item()) > 0:
            z = (log_v - m) / s
            metrics[f"{prefix}_log_skew"] = float((z ** 3).mean().item())
            metrics[f"{prefix}_log_excess_kurt"] = float((z ** 4).mean().item()) - 3.0


def _ranking_metrics(
    ref_flat: torch.Tensor, est_flat: torch.Tensor,
    prefix: str, metrics: Dict[str, float],
) -> None:
    """Spearman ρ (all + top-pct), Pearson on top-1%, Jaccard at top-pcts,
    QTM top-bucket retention (M=10). All ops stay on the input device
    (GPU); only final reductions are pulled to host."""
    n = int(ref_flat.numel())
    if n == 0 or n != int(est_flat.numel()):
        return
    # One argsort each on |values|, descending — reused for all subsets.
    ref_argsort = ref_flat.abs().argsort(descending=True)
    est_argsort = est_flat.abs().argsort(descending=True)
    ref_ranks = _ranks_from_argsort(ref_argsort, n)
    est_ranks = _ranks_from_argsort(est_argsort, n)

    metrics[f"{prefix}_spearman_all"] = _spearman(ref_ranks, est_ranks)

    for q in _TOP_PCTS:
        n_top = max(2, int(round(q * n)))
        ref_top_idx = ref_argsort[:n_top]
        est_top_idx = est_argsort[:n_top]
        sub_ref = ref_ranks[ref_top_idx]
        sub_est = est_ranks[ref_top_idx]
        metrics[f"{prefix}_spearman_top{int(q*100)}pct"] = _spearman(
            sub_ref, sub_est,
        )
        metrics[f"{prefix}_jaccard_top{int(q*100)}pct"] = _jaccard_isin(
            ref_top_idx, est_top_idx,
        )
        if int(q * 100) == 1:
            metrics[f"{prefix}_pearson_top1pct_values"] = _pearson(
                ref_flat[ref_top_idx], est_flat[ref_top_idx],
            )

    # QTM M=10 top-bucket retention (% of ref top-10% kept inside est top-10%)
    M = 10
    per_bucket = max(1, n // M)
    ref_bucket = (ref_ranks // per_bucket).clamp(max=M - 1)
    est_bucket = (est_ranks // per_bucket).clamp(max=M - 1)
    in_top_both = int(((ref_bucket == 0) & (est_bucket == 0)).sum().item())
    metrics[f"{prefix}_top_bucket_retention"] = (
        float(in_top_both) / max(1, per_bucket)
    )


def _log_hist(
    flat_abs_sorted: torch.Tensor, n_bins: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Log-spaced histogram of |values|. Input must be sorted ascending
    (so positive values are at the tail). Returns (counts, edges) on
    the input device."""
    pos = flat_abs_sorted[flat_abs_sorted > 0]
    if pos.numel() < 2:
        empty = torch.zeros(n_bins, dtype=torch.long, device=flat_abs_sorted.device)
        edges = torch.zeros(n_bins + 1, dtype=torch.float64, device=flat_abs_sorted.device)
        return empty, edges
    lo = pos[0].to(torch.float64)
    hi = pos[-1].to(torch.float64)
    if float(hi.item()) <= float(lo.item()):
        empty = torch.zeros(n_bins, dtype=torch.long, device=flat_abs_sorted.device)
        edges = torch.zeros(n_bins + 1, dtype=torch.float64, device=flat_abs_sorted.device)
        return empty, edges
    log_edges = torch.linspace(
        torch.log(lo).item(), torch.log(hi).item(),
        n_bins + 1, dtype=torch.float64, device=flat_abs_sorted.device,
    )
    edges = torch.exp(log_edges)
    counts = torch.histc(pos.to(torch.float64), bins=n_bins,
                         min=float(edges[0].item()), max=float(edges[-1].item()))
    return counts.to(torch.long), edges


def _distribution_diff(
    flat_a_abs: torch.Tensor, flat_b_abs: torch.Tensor,
    prefix: str, metrics: Dict[str, float], n_bins: int = 50,
) -> None:
    """Compare |a| vs |b| distributions (range + bin-count agreement).

    Both inputs are flat |values|; binning uses A's log-spaced edges so
    the two histograms share an axis and can be subtracted directly.
    Metrics:
      * {prefix}_min_ratio_b_over_a, _max_ratio_b_over_a
      * {prefix}_p1_ratio, _p50_ratio, _p99_ratio, _p99.9_ratio
      * {prefix}_bin_l1   (Σ|n_a − n_b| / N, in [0, 2])
      * {prefix}_bin_kl   (KL(a||b) on bin probabilities, with ε floor)
      * {prefix}_n_bins, {prefix}_log_lo, {prefix}_log_hi (axis spec)
    """
    a_sorted = flat_a_abs.sort().values
    b_sorted = flat_b_abs.sort().values
    n_a = int(a_sorted.numel()); n_b = int(b_sorted.numel())
    if n_a == 0 or n_b == 0:
        return

    def _q(t: torch.Tensor, q: float) -> float:
        n = int(t.numel()); k = max(0, min(n - 1, int(round(q * (n - 1)))))
        return float(t[k].item())

    metrics[f"{prefix}_min_ratio_b_over_a"] = _q(b_sorted, 0.0) / max(1e-300, _q(a_sorted, 0.0))
    metrics[f"{prefix}_max_ratio_b_over_a"] = _q(b_sorted, 1.0) / max(1e-300, _q(a_sorted, 1.0))
    for q, name in ((0.01, "p1"), (0.5, "p50"), (0.99, "p99"), (0.999, "p99.9")):
        a_q = _q(a_sorted, q); b_q = _q(b_sorted, q)
        metrics[f"{prefix}_{name}_ratio_b_over_a"] = (b_q / max(1e-300, a_q))
        metrics[f"{prefix}_{name}_a"] = a_q
        metrics[f"{prefix}_{name}_b"] = b_q

    # Bin-count comparison on A's log-spaced edges.
    counts_a, edges = _log_hist(a_sorted, n_bins=n_bins)
    pos_b = b_sorted[b_sorted > 0]
    if pos_b.numel() > 0 and float(edges[-1].item()) > float(edges[0].item()):
        counts_b = torch.histc(
            pos_b.to(torch.float64), bins=n_bins,
            min=float(edges[0].item()), max=float(edges[-1].item()),
        ).to(torch.long)
    else:
        counts_b = torch.zeros_like(counts_a)
    pa = counts_a.to(torch.float64) / max(1, int(counts_a.sum().item()))
    pb = counts_b.to(torch.float64) / max(1, int(counts_b.sum().item()))
    metrics[f"{prefix}_bin_l1"] = float((pa - pb).abs().sum().item())
    eps = 1e-12
    kl = (pa * (torch.log(pa.clamp_min(eps)) - torch.log(pb.clamp_min(eps)))).sum()
    metrics[f"{prefix}_bin_kl"] = float(kl.item())
    metrics[f"{prefix}_n_bins"] = int(n_bins)
    metrics[f"{prefix}_log_lo"] = float(torch.log(edges[0]).item())
    metrics[f"{prefix}_log_hi"] = float(torch.log(edges[-1]).item())


def _get_v_t_grouped(
    expert_ps: List[torch.nn.Parameter], g_blk: int,
) -> Optional[torch.Tensor]:
    """Read FP32 ``exp_avg_sq`` per expert param via Fp32StateMapper
    (view, no copy). At g_blk=1 the result is just a concat of the
    raw v_t views (zero extra VRAM); at g_blk>1 a per-tensor group-
    mean reduce produces a numel/g tensor (transient on-device).

    Stays on the original (GPU) device — matches FD-ref placement so
    downstream ranking compares without D2H transfers."""
    from ._ds_state import get_mapper
    mp = get_mapper()
    if mp is None:
        return None
    parts: List[torch.Tensor] = []
    g = max(1, int(g_blk))
    for p in expert_ps:
        v = mp.get_state(p, "exp_avg_sq")
        if v is None:
            return None
        v_flat = v.detach().reshape(-1).to(torch.float32)
        if g > 1:
            n_aligned = (v_flat.numel() // g) * g
            v_flat = v_flat[:n_aligned].view(-1, g).mean(dim=1)
        parts.append(v_flat)
    return torch.cat(parts, dim=0) if parts else None


def _quantile_table(
    flat_abs: torch.Tensor, n_q: int, drop_nonpositive: bool,
) -> Optional[torch.Tensor]:
    """N_Q-point quantile lookup — same semantics as production
    ``ImportanceTracker._quantile_table_from_flat``. Stride-samples up to
    1M to bound transient, sorts ascending, picks N_Q evenly-spaced
    quantile positions."""
    n = int(flat_abs.numel())
    if n == 0:
        return None
    stride = max(1, n // 1_000_000)
    sample = flat_abs[::stride].detach().to(torch.float32)
    sample_sorted = sample.sort().values
    if drop_nonpositive:
        sample_sorted = sample_sorted[sample_sorted > 0]
    m = int(sample_sorted.numel())
    if m < n_q:
        return None
    idx = (
        torch.arange(1, n_q + 1, device=flat_abs.device, dtype=torch.float32)
        * ((m - 1) / (n_q + 1))
    ).long().clamp_(0, m - 1)
    return sample_sorted[idx]


def _map_v_to_h(
    v_abs: torch.Tensor, v_q: torch.Tensor, h_q: torch.Tensor,
) -> torch.Tensor:
    """Element-wise piecewise-linear CDF map — same semantics as the
    production Triton ``_vmap_kernel``: ``bin = right-search(v_q, |v|)``
    clamped to ``[0, N_Q-2]``; lerp_t in ``(v_q[bin], v_q[bin+1])``;
    output = ``h_q[bin] + lerp_t × (h_q[bin+1] − h_q[bin])``."""
    n_q = int(v_q.numel())
    idx = torch.searchsorted(
        v_q, v_abs.contiguous(), right=False,
    ).clamp(0, n_q - 2)
    v_lo = v_q[idx]
    v_hi = v_q[idx + 1]
    h_lo = h_q[idx]
    h_hi = h_q[idx + 1]
    denom = (v_hi - v_lo).clamp_min(1e-30)
    t = ((v_abs - v_lo) / denom).clamp(0.0, 1.0)
    return h_lo + t * (h_hi - h_lo)


def run_noise_probe(
    *,
    iteration: int,
    model: Any,
    forward_step_func: Callable,
    data_iterator: Any,
    expert_ps: List[torch.nn.Parameter],
    eps: Optional[float],
    ref_k: int = 32,
    ref_n_batches: int = 20,
    log_path: Optional[str] = None,
    g_blk: int = 8,
    tracker: Optional[Any] = None,
) -> Dict[str, Any]:
    """Single probe pass: compute FD reference + v_t, log distribution
    + ranking similarity. No caseA/caseB/σ²/cross-iter — those are FD
    self-noise diagnostics, irrelevant to the v_t-replacement question.
    """
    from .hessian import compute_hessian_finite_diff as _fd

    if not expert_ps:
        return {"iter": int(iteration), "error": "expert_ps empty"}

    t0 = time.perf_counter()

    try:
        batches = _capture_batches(data_iterator, ref_n_batches)
    except Exception as exc:
        print(f"[fd_v_probe] @iter{iteration} batch capture failed: {exc}",
              flush=True)
        return {"iter": int(iteration), "error": f"batch_capture: {exc}"}

    # FD reference: ref_k K=1 calls, cycling over ref_n_batches batches.
    SEED_BASE = 0x10000
    ref_calls: List[List[torch.Tensor]] = []
    for k_i in range(int(ref_k)):
        b = batches[k_i % len(batches)]
        h = _fd(
            model, forward_step_func, data_iterator, expert_ps,
            k_samples=1, eps=eps, use_routing_freeze=True,
            batch_override=b, rng_seed=SEED_BASE + k_i,
            one_sided=False, track_k_values=None, g_blk=int(g_blk),
        )
        if h is None:
            return {"iter": int(iteration), "error": f"ref_kfail_{k_i}"}
        ref_calls.append(h)
    h_ref = _mean_list(ref_calls)
    ref_flat = _flatten_h(h_ref)
    t_ref = time.perf_counter()

    # v_t (Adam exp_avg_sq, FP32, in-place view → group-mean reduce).
    v_t_flat = _get_v_t_grouped(expert_ps, g_blk=int(g_blk))
    if v_t_flat is None:
        return {"iter": int(iteration), "error": "v_t_unavailable"}
    if v_t_flat.numel() != ref_flat.numel():
        return {
            "iter": int(iteration),
            "error": f"shape_mismatch_v={v_t_flat.numel()}_h={ref_flat.numel()}",
        }

    metrics: Dict[str, Any] = {
        "iter": int(iteration),
        "ref_k": int(ref_k),
        "ref_n_batches": int(ref_n_batches),
        "g_blk": int(g_blk),
        "n_total": int(ref_flat.numel()),
    }

    _distribution_stats(ref_flat.abs(), "H_abs", metrics)
    _distribution_stats(v_t_flat.abs(), "v_t", metrics)
    _ranking_metrics(ref_flat, v_t_flat, "v_vs_H", metrics)

    # Scale ratio H/v_t at multiple quantiles. Constant ratio across q
    # ⇒ single affine mapping suffices; q-varying ratio ⇒ piecewise
    # CDF mapping needed.
    for _q in ("p1", "p50", "p95", "p99", "p99.9", "max"):
        h_q = metrics.get(f"H_abs_{_q}", float("nan"))
        v_q = metrics.get(f"v_t_{_q}", float("nan"))
        if v_q and v_q > 0:
            metrics[f"scale_ratio_H_over_v_{_q}"] = float(h_q) / float(v_q)

    # H_cheap = single FD call (K=1, B=1, g=1). Per-element noisy but
    # the *distribution* (range, percentiles, bin counts) is robust:
    # Var(quantile) ∝ 1/N, ~10^16× smaller than Var(per-element) at
    # N=188M. This validates whether a free cheap measurement can
    # supply the H scale used in CDF-matching v_t → H_mapped.
    SEED_CHEAP = 0x60000
    h_cheap = _fd(
        model, forward_step_func, data_iterator, expert_ps,
        k_samples=1, eps=eps, use_routing_freeze=True,
        batch_override=batches[0], rng_seed=SEED_CHEAP,
        one_sided=False, track_k_values=None, g_blk=int(g_blk),
    )
    h_cheap_flat: Optional[torch.Tensor] = None
    if h_cheap is not None:
        _hc_flat = _flatten_h(h_cheap)
        if _hc_flat.numel() == ref_flat.numel():
            h_cheap_flat = _hc_flat
            _distribution_stats(h_cheap_flat.abs(), "H_cheap", metrics)
            _distribution_diff(
                ref_flat.abs(), h_cheap_flat.abs(),
                "diff_cheap_vs_base", metrics,
            )
        else:
            del _hc_flat
        del h_cheap

    # ============================================================
    # v_t mapping experiments (production CDF-matching).
    # Per element: bin = searchsorted(v_q, |v_t[i]|); mapped[i] =
    # lerp(h_q[bin], h_q[bin+1]). h_q derived from BOTH:
    #   • H_base (idealised — best possible mapping fidelity)
    #   • H_cheap (production — uses anchor FD K=1 distribution)
    # After element-level mapping the group-mean reduce (g=g_blk)
    # produces a group-level mapped value, matching the grain of
    # ref_flat and v_t_flat used for ranking comparisons:
    #   Exp A: rank(v_t_mapped) vs rank(v_t)   — does mapping preserve rank?
    #   Exp B: rank(v_t_mapped) vs rank(H_base) — does mapped(v_t) ≈ H rank?
    # N_Q=256 matches production lookup table size.
    # ============================================================
    N_Q = 256
    v_t_element = _get_v_t_grouped(expert_ps, g_blk=1)
    if v_t_element is not None and v_t_element.numel() >= N_Q:
        g_eff = max(1, int(g_blk))
        # v_q from element-level v_t (production: _g_fast_v_quantiles)
        v_q = _quantile_table(
            v_t_element.abs(), n_q=N_Q, drop_nonpositive=False,
        )
        if v_q is not None:
            h_sources: List[Tuple[str, torch.Tensor]] = [("base", ref_flat)]
            if h_cheap_flat is not None:
                h_sources.append(("cheap", h_cheap_flat))
            for h_src_name, h_src_flat in h_sources:
                # h_q from chosen H source (production: from h_g_bf16_flat
                # populated by anchor refresh, which is the cheap variant).
                h_q = _quantile_table(
                    h_src_flat.abs(), n_q=N_Q, drop_nonpositive=True,
                )
                if h_q is None:
                    continue
                # Element-level CDF map (matches Triton _vmap_kernel)
                mapped_elem = _map_v_to_h(v_t_element.abs(), v_q, h_q)
                # Group-mean reduce to ref_flat / v_t_flat grain
                if g_eff > 1:
                    n_align = (mapped_elem.numel() // g_eff) * g_eff
                    mapped_group = mapped_elem[:n_align].view(
                        -1, g_eff,
                    ).mean(dim=1)
                else:
                    mapped_group = mapped_elem
                # Align lengths to ref_flat / v_t_flat (defensive)
                n_min = min(
                    mapped_group.numel(),
                    ref_flat.numel(),
                    v_t_flat.numel(),
                )
                mg = mapped_group[:n_min]
                rf = ref_flat[:n_min]
                vf = v_t_flat[:n_min]
                _distribution_stats(mg, f"vmap_{h_src_name}", metrics)
                # Exp A: rank(v_t_mapped) vs rank(v_t)
                _ranking_metrics(vf, mg, f"vmap_{h_src_name}_vs_v", metrics)
                # Exp B: rank(v_t_mapped) vs rank(H_base)
                _ranking_metrics(rf, mg, f"vmap_{h_src_name}_vs_H", metrics)
                del mapped_elem, mapped_group, mg, h_q
            del v_q
        # ============================================================
        # vmap_prod : production simulation.
        # Uses the LIVE tracker quantile tables:
        #   • _g_fast_v_quantiles : refreshed every iter (current v_t scale)
        #   • _g_fast_h_quantiles : refreshed only at anchor refresh
        #     (HESSIAN_REFRESH_EVERY) — so STALE between refresh boundaries.
        # This reflects exactly what the production score = mapped(v_t) × δ²
        # would see at this iter. Compared against H_base (truth) to
        # quantify how much staleness costs in rank fidelity.
        # ============================================================
        if (tracker is not None
                and getattr(tracker, "_h_anchor_initialised", False)
                and getattr(tracker, "_g_fast_v_quantiles", None) is not None
                and getattr(tracker, "_g_fast_h_quantiles", None) is not None):
            prod_v_q = tracker._g_fast_v_quantiles
            prod_h_q = tracker._g_fast_h_quantiles
            if (prod_v_q.numel() == N_Q and prod_h_q.numel() == N_Q
                    and prod_v_q.device == v_t_element.device):
                mapped_elem_p = _map_v_to_h(v_t_element.abs(), prod_v_q, prod_h_q)
                if g_eff > 1:
                    n_align = (mapped_elem_p.numel() // g_eff) * g_eff
                    mapped_group_p = mapped_elem_p[:n_align].view(
                        -1, g_eff,
                    ).mean(dim=1)
                else:
                    mapped_group_p = mapped_elem_p
                n_min = min(
                    mapped_group_p.numel(),
                    ref_flat.numel(),
                    v_t_flat.numel(),
                )
                mg = mapped_group_p[:n_min]
                rf = ref_flat[:n_min]
                vf = v_t_flat[:n_min]
                _distribution_stats(mg, "vmap_prod", metrics)
                _ranking_metrics(vf, mg, "vmap_prod_vs_v", metrics)
                _ranking_metrics(rf, mg, "vmap_prod_vs_H", metrics)
                del mapped_elem_p, mapped_group_p, mg
        del v_t_element
    if h_cheap_flat is not None:
        del h_cheap_flat

    # Cross-iter H_base distribution stability: compare current vs prior
    # probe's H_base (sorted |values| stashed on host to spare GPU mem).
    global _PREV_H_BASE_SORTED
    ref_abs_sorted_host = ref_flat.abs().sort().values.cpu()
    if _PREV_H_BASE_SORTED is not None and _PREV_H_BASE_SORTED.numel() == ref_abs_sorted_host.numel():
        prev_dev = _PREV_H_BASE_SORTED.to(ref_flat.device)
        _distribution_diff(
            prev_dev, ref_abs_sorted_host.to(ref_flat.device),
            "diff_prev_vs_curr", metrics,
        )
        del prev_dev
    _PREV_H_BASE_SORTED = ref_abs_sorted_host

    metrics["wall_clock_s"] = float(time.perf_counter() - t0)
    metrics["ref_compute_s"] = float(t_ref - t0)

    # One human-readable summary line.
    print(
        f"[fd_v_probe] @iter{iteration} g={g_blk} N={metrics['n_total']} "
        f"K={ref_k} B={ref_n_batches}\n"
        f"  |H|: p50={metrics.get('H_abs_p50', float('nan')):.3e} "
        f"p99={metrics.get('H_abs_p99', float('nan')):.3e} "
        f"p99.9={metrics.get('H_abs_p99.9', float('nan')):.3e} "
        f"max={metrics.get('H_abs_max', float('nan')):.3e}  "
        f"log: μ={metrics.get('H_abs_log_mean', float('nan')):.3f} "
        f"σ={metrics.get('H_abs_log_std', float('nan')):.3f} "
        f"skew={metrics.get('H_abs_log_skew', float('nan')):.3f} "
        f"k={metrics.get('H_abs_log_excess_kurt', float('nan')):.3f}\n"
        f"  v_t: p50={metrics.get('v_t_p50', float('nan')):.3e} "
        f"p99={metrics.get('v_t_p99', float('nan')):.3e} "
        f"p99.9={metrics.get('v_t_p99.9', float('nan')):.3e} "
        f"max={metrics.get('v_t_max', float('nan')):.3e}  "
        f"log: μ={metrics.get('v_t_log_mean', float('nan')):.3f} "
        f"σ={metrics.get('v_t_log_std', float('nan')):.3f} "
        f"skew={metrics.get('v_t_log_skew', float('nan')):.3f} "
        f"k={metrics.get('v_t_log_excess_kurt', float('nan')):.3f}\n"
        f"  ρ_all={metrics.get('v_vs_H_spearman_all', float('nan')):.4f}  "
        f"ρ@1%={metrics.get('v_vs_H_spearman_top1pct', float('nan')):.4f}  "
        f"ρ@3%={metrics.get('v_vs_H_spearman_top3pct', float('nan')):.4f}  "
        f"ρ@5%={metrics.get('v_vs_H_spearman_top5pct', float('nan')):.4f}  "
        f"ρ@10%={metrics.get('v_vs_H_spearman_top10pct', float('nan')):.4f}\n"
        f"  J@1%={metrics.get('v_vs_H_jaccard_top1pct', float('nan')):.4f}  "
        f"J@3%={metrics.get('v_vs_H_jaccard_top3pct', float('nan')):.4f}  "
        f"J@5%={metrics.get('v_vs_H_jaccard_top5pct', float('nan')):.4f}  "
        f"J@10%={metrics.get('v_vs_H_jaccard_top10pct', float('nan')):.4f}  "
        f"top_bkt_ret={metrics.get('v_vs_H_top_bucket_retention', float('nan')):.4f}  "
        f"P@1%={metrics.get('v_vs_H_pearson_top1pct_values', float('nan')):.4f}\n"
        f"  EXP_A vmap_base_vs_v: "
        f"ρ_all={metrics.get('vmap_base_vs_v_spearman_all', float('nan')):.4f} "
        f"ρ@1%={metrics.get('vmap_base_vs_v_spearman_top1pct', float('nan')):.4f} "
        f"J@1%={metrics.get('vmap_base_vs_v_jaccard_top1pct', float('nan')):.4f} "
        f"J@4%={metrics.get('vmap_base_vs_v_jaccard_top3pct', float('nan')):.4f} "
        f"J@10%={metrics.get('vmap_base_vs_v_jaccard_top10pct', float('nan')):.4f}\n"
        f"  EXP_B vmap_base_vs_H: "
        f"ρ_all={metrics.get('vmap_base_vs_H_spearman_all', float('nan')):.4f} "
        f"ρ@1%={metrics.get('vmap_base_vs_H_spearman_top1pct', float('nan')):.4f} "
        f"J@1%={metrics.get('vmap_base_vs_H_jaccard_top1pct', float('nan')):.4f} "
        f"J@4%={metrics.get('vmap_base_vs_H_jaccard_top3pct', float('nan')):.4f} "
        f"J@10%={metrics.get('vmap_base_vs_H_jaccard_top10pct', float('nan')):.4f}\n"
        f"  EXP_B vmap_cheap_vs_H: "
        f"ρ_all={metrics.get('vmap_cheap_vs_H_spearman_all', float('nan')):.4f} "
        f"ρ@1%={metrics.get('vmap_cheap_vs_H_spearman_top1pct', float('nan')):.4f} "
        f"J@1%={metrics.get('vmap_cheap_vs_H_jaccard_top1pct', float('nan')):.4f} "
        f"J@4%={metrics.get('vmap_cheap_vs_H_jaccard_top3pct', float('nan')):.4f} "
        f"J@10%={metrics.get('vmap_cheap_vs_H_jaccard_top10pct', float('nan')):.4f}\n"
        f"  EXP_B vmap_prod_vs_H (LIVE tracker h_q, stale): "
        f"ρ_all={metrics.get('vmap_prod_vs_H_spearman_all', float('nan')):.4f} "
        f"ρ@1%={metrics.get('vmap_prod_vs_H_spearman_top1pct', float('nan')):.4f} "
        f"J@1%={metrics.get('vmap_prod_vs_H_jaccard_top1pct', float('nan')):.4f} "
        f"J@4%={metrics.get('vmap_prod_vs_H_jaccard_top3pct', float('nan')):.4f} "
        f"J@10%={metrics.get('vmap_prod_vs_H_jaccard_top10pct', float('nan')):.4f}\n"
        f"  wall={metrics['wall_clock_s']:.1f}s "
        f"(ref={metrics['ref_compute_s']:.1f}s)",
        flush=True,
    )

    if log_path:
        try:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as exc:
            print(f"[fd_v_probe] log write failed: {exc}", flush=True)

    return metrics
