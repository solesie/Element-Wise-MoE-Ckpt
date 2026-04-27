"""Diagonal Hessian estimators for the element-wise importance score.

Two estimators are provided:

* :func:`compute_hessian_hutchinson` — K-sample Rademacher on true-label
  CE via 2nd-order autograd (create_graph=True + Hz). Unbiased;
  K× more expensive than GNB. Matches the "Hutchinson" arm of the
  ``--proposal-hessian-estimator`` flag.
* :func:`compute_hessian_gnb` — Sophia-G (Gauss-Newton-Bartlett): sample
  labels from softmax, 1 forward + 1 backward, square the gradient,
  scale by batch size. Cheap; biased in non-CE regimes.

Both return per-parameter FP32 tensors with values ``|H_diag|`` (already
absolute-valued for Hutchinson, non-negative by construction for GNB)
ready to be consumed by ``ImportanceTracker._H_ema`` (EMA-folded in the
refresh hook).

:func:`compare_hutchinson_vs_gnb` computes Pearson / Jaccard similarity
between the two estimators on bare |H| and on the score = |H|·δ². It is
called from ``proposal_refresh_hutchinson`` when
``--proposal-hessian-cross-check`` is set.

These functions are state-free: all dependencies (model, data iterator,
tracked params, θ_CP cache) are passed in explicitly. That makes them
testable and independent of the cp_hooks singleton state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


# ----------------------------------------------------------------------
# Hutchinson estimator (K-sample Rademacher on true-label CE)
# ----------------------------------------------------------------------
def compute_hessian_hutchinson(
    model,
    forward_step_func,
    data_iterator,
    expert_ps: List[torch.nn.Parameter],
    k_samples: int,
) -> Optional[List[torch.Tensor]]:
    """Compute |H_diag| via K-sample Hutchinson estimator (Rademacher).
    Returns list of FP32 tensors matching ``expert_ps``, already pre-abs'd.
    Uses true-label CE via ``forward_step_func`` + create_graph=True + 2nd-order grad.
    Returns None on failure. Caller must handle RNG / engine state restoration.

    Memory: the refresh is wrapped in
    :func:`~megatron.proposal.checkpoint_ctx.hutchinson_cpu_offload_ctx`
    (opt-out via ``HUTCHINSON_NO_CPU_OFFLOAD=1``). It installs
    ``saved_tensors_hooks`` that park every saved-for-backward tensor on
    a pinned CPU buffer pool until autograd demands it. Empirically drops
    exp2-scale refresh peak from OOM (~74 GB) to ~43 GB, at ~17 s/refresh
    steady-state (1 in 40 iters, ~0.4 s/iter amortized).

    Regular training iters are unaffected — the ctx is entered only for
    this refresh call, which fires every ``--proposal-hessian-refresh-every``
    iterations.
    """
    import os as _os
    from contextlib import nullcontext as _nullcontext
    from .checkpoint_ctx import hutchinson_cpu_offload_ctx
    from ._functorch_probe import is_probe_enabled as _is_probe_enabled, _probe as _run_probe

    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    m0 = model[0] if isinstance(model, (list, tuple)) else model

    # Diagnostic: HUTCHINSON_FUNCTORCH_TEST=1 replaces this refresh with a
    # 3-step compatibility probe (functional_call → grad_fn → jvp) and
    # exits. Used to empirically verify torch.func viability before any
    # refactor claim.
    if _is_probe_enabled():
        _run_probe(model, forward_step_func, data_iterator, expert_ps)
        return None  # unreachable — probe os._exit(0) s
    # HUTCHINSON_NO_CPU_OFFLOAD=1: opt-out of the pinned-pool saved-tensor
    # offload. Default ON; this is the load-bearing mitigation.
    _cpu_offload = _os.environ.get("HUTCHINSON_NO_CPU_OFFLOAD", "") not in ("1", "true", "True")
    # Per-phase GPU memory instrumentation (HUTCHINSON_MEM_TRACE=1). Prints
    # ``torch.cuda.memory_allocated()`` at each key phase so the caller can
    # locate where peak memory is consumed inside the refresh.
    _trace = _os.environ.get("HUTCHINSON_MEM_TRACE", "") in ("1", "true", "True")
    import time as _time
    _t0 = _time.perf_counter()
    def _mem(tag: str) -> None:
        if not _trace or not torch.cuda.is_available():
            return
        # Synchronize so reported elapsed reflects completed device work,
        # not just kernel launches queued on the stream. Negligible overhead
        # relative to the phases we're timing (≥100 ms each).
        torch.cuda.synchronize()
        a = torch.cuda.memory_allocated() / 1024 / 1024
        r = torch.cuda.memory_reserved() / 1024 / 1024
        dt = _time.perf_counter() - _t0
        print(f"[hutch-mem] {tag}: t={dt:.2f}s alloc={a:.1f}MB reserved={r:.1f}MB",
              flush=True)
    if _trace and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _mem("entry")
    _ctx = hutchinson_cpu_offload_ctx() if _cpu_offload else _nullcontext()
    try:
        with _ctx:
            _mem("after ctx enter")
            try:
                loss, _ = forward_step_func(di, m0)
            except Exception as exc:
                print(f"[proposal] compute_hessian_hutchinson: forward failed: {exc}", flush=True)
                return None
            _mem("after forward")
            if not torch.is_tensor(loss):
                return None
            if loss.dim() > 0:
                loss = loss.mean()
            try:
                grads = torch.autograd.grad(
                    loss, expert_ps, create_graph=True, retain_graph=True, allow_unused=True,
                )
            except Exception as exc:
                import traceback as _tb
                print(
                    f"[proposal] compute_hessian_hutchinson: 1st-order grad failed: "
                    f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}",
                    flush=True,
                )
                del loss
                return None
            _mem("after 1st-order grad (create_graph=True)")
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, expert_ps)]
            gen = torch.Generator(device=expert_ps[0].device).manual_seed(0xBADCAFE)
            h_accum = [torch.zeros_like(p, dtype=torch.float32) for p in expert_ps]
            _mem("after h_accum alloc")
            for k in range(k_samples):
                zs = [
                    (torch.randint(0, 2, p.shape, generator=gen, device=p.device, dtype=torch.float32) * 2 - 1)
                    for p in expert_ps
                ]
                _mem(f"k={k} after zs alloc")
                dot = sum((g * z).sum() for g, z in zip(grads, zs))
                _mem(f"k={k} after dot (g·z sum)")
                try:
                    hz = torch.autograd.grad(
                        dot, expert_ps, retain_graph=(k < k_samples - 1), allow_unused=True,
                    )
                    _mem(f"k={k} after 2nd-order grad (Hv)")
                except Exception as exc:
                    import traceback as _tb
                    print(
                        f"[proposal] compute_hessian_hutchinson: 2nd-order grad failed at k={k}: "
                        f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}",
                        flush=True,
                    )
                    return None
                hz = [h if h is not None else torch.zeros_like(p) for h, p in zip(hz, expert_ps)]
                # Streaming accumulation: element-wise |z_i · (Hz)_i| folded
                # into h_accum[i] (fp32, pre-allocated once), then drop the
                # full hz and zs tensor lists. h_accum IS the diagonal-Hessian
                # estimate we need for the importance score — a scalar
                # trace-style reduction (v·Hv) would destroy the per-element
                # signal. Memory trace confirms zs/hz are released before the
                # next k-iter allocates: alloc stays flat across k.
                for i, (z, h) in enumerate(zip(zs, hz)):
                    h_accum[i].add_((z * h).to(torch.float32).abs_().detach().div_(k_samples))
                del zs, hz, dot
                _mem(f"k={k} after accumulate (end of k iter)")
            del loss, grads
            _mem("after del loss, grads")
            if _trace and torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                print(f"[hutch-mem] PEAK during refresh: {peak:.1f}MB", flush=True)
            return h_accum
    except Exception as exc:
        print(f"[proposal] compute_hessian_hutchinson: context failed: {exc}", flush=True)
        return None


# ----------------------------------------------------------------------
# GNB (Sophia-G) estimator
# ----------------------------------------------------------------------
def compute_hessian_gnb(
    model,
    data_iterator,
    expert_ps: List[torch.nn.Parameter],
    k_samples: int,
) -> Optional[List[torch.Tensor]]:
    """Compute |H_diag| via Sophia-G (Gauss-Newton-Bartlett). Returns list of
    FP32 non-negative tensors. Samples labels from softmax(logits), takes
    first-order grad, squares, scales by effective batch size.
    """
    import torch.nn.functional as F
    try:
        from pretrain_gpt import get_batch as _get_batch_gpt
    except Exception as _imp_exc:
        print(f"[proposal] compute_hessian_gnb: cannot import get_batch: {_imp_exc}", flush=True)
        return None
    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    m0 = model[0] if isinstance(model, (list, tuple)) else model
    h_accum = [torch.zeros_like(p, dtype=torch.float32) for p in expert_ps]
    for k in range(max(1, int(k_samples))):
        try:
            batch = _get_batch_gpt(di)
        except Exception as exc:
            print(f"[proposal] compute_hessian_gnb: get_batch failed at k={k}: {exc}", flush=True)
            return None
        tokens, _labels_true, loss_mask, attention_mask, position_ids = batch[:5]
        try:
            out = m0(tokens, position_ids, attention_mask)
        except Exception as exc:
            print(f"[proposal] compute_hessian_gnb: forward failed at k={k}: {exc}", flush=True)
            return None
        logits = out[0] if isinstance(out, tuple) else out
        if logits is None or not torch.is_tensor(logits):
            return None
        V = logits.size(-1) if logits.dim() >= 2 else None
        flat_logits = logits.reshape(-1, V) if V is not None else logits
        with torch.no_grad():
            probs = torch.softmax(flat_logits.detach().float(), dim=-1)
            y_hat = torch.multinomial(probs, num_samples=1).squeeze(-1)
        del probs
        if loss_mask is not None and loss_mask.numel() == flat_logits.size(0):
            per_tok = F.cross_entropy(flat_logits.float(), y_hat, reduction="none")
            denom = loss_mask.view(-1).float().sum().clamp_min(1.0)
            loss_gnb = (per_tok * loss_mask.view(-1).float()).sum() / denom
            B_eff = float(denom.item())
        else:
            loss_gnb = F.cross_entropy(flat_logits.float(), y_hat, reduction="mean")
            B_eff = float(flat_logits.size(0))
        try:
            grads = torch.autograd.grad(
                loss_gnb, expert_ps,
                retain_graph=False, create_graph=False, allow_unused=True,
            )
        except Exception as exc:
            print(f"[proposal] compute_hessian_gnb: grad failed at k={k}: {exc}", flush=True)
            return None
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, expert_ps)]
        inv_k = 1.0 / max(1, int(k_samples))
        for i, g in enumerate(grads):
            h_accum[i].add_(g.to(torch.float32).pow(2).mul_(B_eff * inv_k))
        del logits, out, flat_logits, y_hat, loss_gnb, grads, tokens, batch
    return h_accum


# ----------------------------------------------------------------------
# Hutchinson vs GNB ranking-similarity comparison
# ----------------------------------------------------------------------
def compare_hutchinson_vs_gnb(
    iteration: int,
    tracked: List[Tuple[str, torch.nn.Parameter]],
    h_hutch: List[torch.Tensor],
    h_gnb: List[torch.Tensor],
    theta_cp: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Compute Pearson/Jaccard similarity between Hutchinson and GNB |H| on
    bare and δ²-weighted scales.

    Per-tensor GPU reduction, log-bin histc thresholds; no CPU sync inside
    the hot loop. The ``theta_cp`` dict maps tensor name → BF16 θ_CP cache
    (``ImportanceTracker.theta_cp``) for δ computation. Pass an empty
    dict to skip δ² weighting (falls back to zeros).
    """
    device = h_hutch[0].device if h_hutch else torch.device("cpu")
    LOG_LO, LOG_HI, BINS = -40.0, 20.0, 600

    def _log_hist_add(vals, hist):
        log_v = torch.log10(vals.clamp_min(1e-40))
        idx = ((log_v - LOG_LO) * (BINS / (LOG_HI - LOG_LO))).clamp_(0, BINS - 1).long()
        hist += torch.bincount(idx, minlength=BINS)

    pears = {k: 0.0 for k in (
        "n",
        "sa_h", "sb_g", "sab_hg", "sa2_h", "sb2_g",
        "sa_sh", "sb_sg", "sab_shsg", "sa2_sh", "sb2_sg",
    )}
    hist_h = torch.zeros(BINS, dtype=torch.int64, device=device)
    hist_g = torch.zeros(BINS, dtype=torch.int64, device=device)
    hist_sh = torch.zeros(BINS, dtype=torch.int64, device=device)
    hist_sg = torch.zeros(BINS, dtype=torch.int64, device=device)

    # Pass 1: sums + histograms
    for (name, p), hh, hg in zip(tracked, h_hutch, h_gnb):
        hh_f = hh.detach().flatten()
        hg_f = hg.detach().flatten()
        fc = theta_cp.get(name)
        if fc is not None:
            d_sq = (p.data.to(torch.float32) - fc.to(torch.float32)).pow_(2).flatten()
        else:
            d_sq = torch.zeros_like(hh_f)
        sh_f = hh_f * d_sq
        sg_f = hg_f * d_sq
        pears["n"] += float(hh_f.numel())
        pears["sa_h"] += float(hh_f.sum().item())
        pears["sb_g"] += float(hg_f.sum().item())
        pears["sab_hg"] += float((hh_f * hg_f).sum().item())
        pears["sa2_h"] += float((hh_f * hh_f).sum().item())
        pears["sb2_g"] += float((hg_f * hg_f).sum().item())
        pears["sa_sh"] += float(sh_f.sum().item())
        pears["sb_sg"] += float(sg_f.sum().item())
        pears["sab_shsg"] += float((sh_f * sg_f).sum().item())
        pears["sa2_sh"] += float((sh_f * sh_f).sum().item())
        pears["sb2_sg"] += float((sg_f * sg_f).sum().item())
        _log_hist_add(hh_f, hist_h)
        _log_hist_add(hg_f, hist_g)
        _log_hist_add(sh_f, hist_sh)
        _log_hist_add(sg_f, hist_sg)
        del hh_f, hg_f, d_sq, sh_f, sg_f

    def _pearson_from_sums(n, sa, sb, sab, sa2, sb2):
        if n <= 0: return 0.0
        mean_a = sa / n; mean_b = sb / n
        cov = sab / n - mean_a * mean_b
        va = max(sa2 / n - mean_a * mean_a, 0.0)
        vb = max(sb2 / n - mean_b * mean_b, 0.0)
        d = (va * vb) ** 0.5
        return float(cov / d) if d > 0 else 0.0

    pearson_bare = _pearson_from_sums(
        pears["n"], pears["sa_h"], pears["sb_g"], pears["sab_hg"],
        pears["sa2_h"], pears["sb2_g"],
    )
    pearson_score = _pearson_from_sums(
        pears["n"], pears["sa_sh"], pears["sb_sg"], pears["sab_shsg"],
        pears["sa2_sh"], pears["sb2_sg"],
    )

    def _threshold_from_hist(hist, k_target):
        if k_target <= 0: return float("inf")
        rev_cum = torch.flip(torch.cumsum(torch.flip(hist, [0]), 0), [0])
        mask = rev_cum >= k_target
        if not bool(mask.any()): return 0.0
        last_idx = int(mask.nonzero(as_tuple=True)[0][-1].item())
        bin_width = (LOG_HI - LOG_LO) / BINS
        return 10.0 ** (LOG_LO + last_idx * bin_width)

    ratios = (0.001, 0.01, 0.1)
    N_total = int(pears["n"])
    thresholds = {}
    for r in ratios:
        k_target = r * N_total
        thresholds[r] = (
            _threshold_from_hist(hist_h, k_target),
            _threshold_from_hist(hist_g, k_target),
            _threshold_from_hist(hist_sh, k_target),
            _threshold_from_hist(hist_sg, k_target),
        )

    jacc = {r: {"inter_hg": 0, "A_h": 0, "A_g": 0,
                "inter_shsg": 0, "A_sh": 0, "A_sg": 0} for r in ratios}
    for (name, p), hh, hg in zip(tracked, h_hutch, h_gnb):
        hh_f = hh.detach().flatten()
        hg_f = hg.detach().flatten()
        fc = theta_cp.get(name)
        if fc is not None:
            d_sq = (p.data.to(torch.float32) - fc.to(torch.float32)).pow_(2).flatten()
        else:
            d_sq = torch.zeros_like(hh_f)
        sh_f = hh_f * d_sq
        sg_f = hg_f * d_sq
        for r in ratios:
            th_h, th_g, th_sh, th_sg = thresholds[r]
            mh = hh_f >= th_h; mg = hg_f >= th_g
            msh = sh_f >= th_sh; msg = sg_f >= th_sg
            jacc[r]["A_h"] += int(mh.sum().item())
            jacc[r]["A_g"] += int(mg.sum().item())
            jacc[r]["inter_hg"] += int((mh & mg).sum().item())
            jacc[r]["A_sh"] += int(msh.sum().item())
            jacc[r]["A_sg"] += int(msg.sum().item())
            jacc[r]["inter_shsg"] += int((msh & msg).sum().item())
        del hh_f, hg_f, d_sq, sh_f, sg_f

    def _j(inter, a, b):
        u = a + b - inter
        return (inter / u) if u > 0 else 0.0

    return {
        "iter": int(iteration),
        "total_expert_numel": N_total,
        "pearson_hutch_gnb_bare": pearson_bare,
        "jaccard_top0.1pct_bare": _j(jacc[0.001]["inter_hg"], jacc[0.001]["A_h"], jacc[0.001]["A_g"]),
        "jaccard_top1pct_bare":   _j(jacc[0.01]["inter_hg"],  jacc[0.01]["A_h"],  jacc[0.01]["A_g"]),
        "jaccard_top10pct_bare":  _j(jacc[0.1]["inter_hg"],   jacc[0.1]["A_h"],   jacc[0.1]["A_g"]),
        "pearson_hutch_gnb_score": pearson_score,
        "jaccard_top0.1pct_score": _j(jacc[0.001]["inter_shsg"], jacc[0.001]["A_sh"], jacc[0.001]["A_sg"]),
        "jaccard_top1pct_score":   _j(jacc[0.01]["inter_shsg"],  jacc[0.01]["A_sh"],  jacc[0.01]["A_sg"]),
        "jaccard_top10pct_score":  _j(jacc[0.1]["inter_shsg"],   jacc[0.1]["A_sh"],   jacc[0.1]["A_sg"]),
    }
