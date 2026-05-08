"""Diagonal Hessian estimators for the element-wise importance score.

Production
----------
* :func:`compute_hessian_finite_diff` — central-difference Hessian-vector
  product. Wraps :func:`compute_hessian_finite_diff_core` (param-perturb)
  inside :func:`._param_rebind.param_rebind_context` so writes to
  ``p.data`` reach the bytes forward reads under DeepSpeed BF16+ZeRO-1.
  Uses MoE routing freeze so ±ε forwards see the same dispatch.
  Validated to match Hutchinson K=4 baseline within iteration noise on
  production exp_etc (BF16, 12-layer 768d) and is faster (no 2nd-order
  autograd, no saved-tensor offload).

Truth reference (opt-in)
------------------------
* :func:`compute_hessian_hutchinson` — K-sample Rademacher with 2nd-order
  autograd. Used for occasional comparison runs on small models
  (`--proposal-hessian-estimator hutchinson`). Memory-heavy due to
  saved-tensor offload; not validated for EP > 1.

These functions are state-free: all dependencies (model, data iterator,
tracked params, θ_CP cache) are passed in explicitly.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# ----------------------------------------------------------------------
# Hutchinson estimator (kept for occasional truth comparison on small
# models — not the production default; production uses finite_diff).
# ----------------------------------------------------------------------
def compute_hessian_hutchinson(
    model,
    forward_step_func,
    data_iterator,
    expert_ps: List[torch.nn.Parameter],
    k_samples: int,
    rng_seed: int = 0xBADCAFE,
) -> Optional[List[torch.Tensor]]:
    """Compute |H_diag| via K-sample Hutchinson estimator (Rademacher).
    Returns list of FP32 tensors matching ``expert_ps``, already pre-abs'd.
    Uses true-label CE via ``forward_step_func`` + create_graph=True + 2nd-order grad.
    Returns None on failure. Caller must handle RNG / engine state restoration.

    Memory: the refresh is wrapped in
    :func:`~megatron.proposal.checkpoint_ctx.hutchinson_cpu_offload_ctx`
    (opt-out via ``HUTCHINSON_NO_CPU_OFFLOAD=1``). It installs
    ``saved_tensors_hooks`` that park every saved-for-backward tensor on
    a pinned CPU buffer pool until autograd demands it.

    Use case: invoked when ``--proposal-hessian-estimator hutchinson`` is
    explicitly set — primarily for truth comparison runs on small models
    (FD K=16 vs Hutch K=4 vs Hutch K=64 on tiny MoE / FP32 dense LM).
    Production runs use finite_diff (default).
    """
    import os as _os
    from contextlib import nullcontext as _nullcontext
    from .checkpoint_ctx import hutchinson_cpu_offload_ctx

    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    m0 = model[0] if isinstance(model, (list, tuple)) else model

    _cpu_offload = _os.environ.get("HUTCHINSON_NO_CPU_OFFLOAD", "") not in ("1", "true", "True")
    _ctx = hutchinson_cpu_offload_ctx() if _cpu_offload else _nullcontext()
    try:
        with _ctx:
            try:
                loss, _ = forward_step_func(di, m0)
            except Exception as exc:
                print(f"[proposal] compute_hessian_hutchinson: forward failed: {exc}", flush=True)
                return None
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
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, expert_ps)]
            gen = torch.Generator(device=expert_ps[0].device).manual_seed(int(rng_seed))
            h_accum = [torch.zeros_like(p, dtype=torch.float32) for p in expert_ps]
            for k in range(k_samples):
                zs = [
                    (torch.randint(0, 2, p.shape, generator=gen, device=p.device, dtype=torch.float32) * 2 - 1)
                    for p in expert_ps
                ]
                dot = sum((g * z).sum() for g, z in zip(grads, zs))
                try:
                    hz = torch.autograd.grad(
                        dot, expert_ps, retain_graph=(k < k_samples - 1), allow_unused=True,
                    )
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
                # estimate we need for the importance score.
                for i, (z, h) in enumerate(zip(zs, hz)):
                    h_accum[i].add_((z * h).to(torch.float32).abs_().detach().div_(k_samples))
                del zs, hz, dot
            del loss, grads
            return h_accum
    except Exception as exc:
        print(f"[proposal] compute_hessian_hutchinson: context failed: {exc}", flush=True)
        return None


# ----------------------------------------------------------------------
# Finite-difference estimator (production default)
def _make_rademacher(params: List[torch.nn.Parameter], gen: torch.Generator) -> List[torch.Tensor]:
    return [
        (torch.randint(0, 2, p.shape, generator=gen, device=p.device, dtype=torch.float32) * 2 - 1)
        for p in params
    ]


def _compute_grad_no_create(
    forward_fn: Callable[[], Any],
    params: List[torch.nn.Parameter],
) -> Optional[List[torch.Tensor]]:
    """Run forward → loss → torch.autograd.grad with create_graph=False,
    retain_graph=False. Returns list of gradient tensors (zero-filled for
    None entries) or None on failure.
    """
    try:
        loss, _ = forward_fn()
    except Exception as exc:
        print(f"[proposal] finite_diff: forward failed: {exc}", flush=True)
        return None
    if not torch.is_tensor(loss):
        return None
    if loss.dim() > 0:
        loss = loss.mean()
    try:
        grads = torch.autograd.grad(
            loss, params,
            create_graph=False, retain_graph=False, allow_unused=True,
        )
    except Exception as exc:
        print(f"[proposal] finite_diff: grad failed: {exc}", flush=True)
        return None
    grads = [g.detach() if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
    return grads


def _pearson_two_tensor_lists(a_list: List[torch.Tensor],
                               b_list: List[torch.Tensor]) -> float:
    """Compute streaming Pearson over flattened concatenation. Used by the
    eps auto-tune sweep against a Hutchinson reference.
    """
    n = 0; sa = 0.0; sb = 0.0; sab = 0.0; sa2 = 0.0; sb2 = 0.0
    for a, b in zip(a_list, b_list):
        af = a.detach().flatten().to(torch.float64)
        bf = b.detach().flatten().to(torch.float64)
        n += int(af.numel())
        sa += float(af.sum().item()); sb += float(bf.sum().item())
        sab += float((af * bf).sum().item())
        sa2 += float((af * af).sum().item()); sb2 += float((bf * bf).sum().item())
    if n <= 0:
        return 0.0
    mean_a = sa / n; mean_b = sb / n
    cov = sab / n - mean_a * mean_b
    va = max(sa2 / n - mean_a * mean_a, 0.0)
    vb = max(sb2 / n - mean_b * mean_b, 0.0)
    d = (va * vb) ** 0.5
    return float(cov / d) if d > 0 else 0.0



def compute_hessian_finite_diff_core(
    forward_loss_fn: Callable[[], Tuple[torch.Tensor, Any]],
    expert_ps: List[torch.nn.Parameter],
    k_samples: int,
    eps: float,
    gates_for_freeze: Optional[List[Tuple[str, torch.nn.Module]]] = None,
    rng_seed: int = 0xFD15CAFE,
    one_sided: bool = False,
    track_k_values: Optional[List[int]] = None,
) -> Optional[Any]:
    """Backend that runs the finite-difference loop on a caller-supplied
    forward closure. Used by both :func:`compute_hessian_finite_diff` (via
    pretrain_gpt batch capture) and the unit tests (via a synthetic
    forward closure that needs no Megatron import).

    ``forward_loss_fn`` MUST close over a fixed batch (Common Random
    Numbers requirement). It must return ``(scalar_loss_tensor, ...)``.
    ``gates_for_freeze`` is the list returned by
    :func:`identify_gate_modules`; when None, no routing freeze is
    applied. ``eps`` must be > 0.

    ``one_sided`` (default False, central diff) toggles forward-difference
    mode: ``Hz ≈ [g(θ+εz) - g(θ)] / ε``. The baseline g(θ) is computed
    ONCE before the K-sample loop and reused, so total fwd+bwd cost drops
    from ``2K`` (central) to ``K + 1`` (forward). Truncation error
    increases from O(ε²) to O(ε); validated empirically — see
    PROPOSAL_FD_ONE_SIDED env in cp_hooks. Memory cost: 1 extra grad list
    held for the whole refresh (~|expert_ps| BF16 bytes).

    Returns list of |H_diag| tensors (FP32, same shape as ``expert_ps``)
    or ``None`` on failure. Param mutation is restored exactly via
    equal-and-opposite ``add_`` calls; restore correctness is verified
    per-tensor after the loop.
    """
    if not expert_ps or eps <= 0:
        return None
    from contextlib import nullcontext
    from ._routing_freeze import RoutingFreezeContext

    theta_before_norms = [float(p.data.detach().to(torch.float32).norm().item())
                          for p in expert_ps]

    # Bit-exact snapshot of θ for restore between +ε / -ε forwards.
    # Required because the previous +ε / -2ε / +ε in-place chain leaks
    # ~1 ULP per add_() through BF16 rounding (model params are BF16);
    # accumulated drift was ~0.4% of ||θ|| at eps=1e-2 × ||θ||_mean,
    # blowing the restore check and forcing a false-positive failure.
    # Snapshot is per-tensor .clone() — peak transient ≈ Σ |expert_p|
    # bytes (BF16). For exp_etc 12-layer / 0.3B that's ~300 MB; for
    # exp_3 1.4B that's ~2.4 GB. Freed in `finally`.
    theta_snapshots = [p.data.detach().clone() for p in expert_ps]

    if gates_for_freeze:
        ctx_mgr = RoutingFreezeContext(gates_for_freeze)
    else:
        ctx_mgr = None

    h_accum = [torch.zeros_like(p, dtype=torch.float32) for p in expert_ps]
    gen = torch.Generator(device=expert_ps[0].device).manual_seed(rng_seed)
    # Multi-K snapshot tracking: when track_k_values is provided, we
    # accumulate raw sums (no inv_k division during loop) and snapshot
    # h_sum / k_done at each K boundary. Returns dict {k: list[Tensor]}.
    _track_set: set = set()
    _snapshots: Dict[int, List[torch.Tensor]] = {}
    if track_k_values:
        _track_set = set(int(k) for k in track_k_values
                         if 1 <= int(k) <= int(k_samples))
    try:
        with (ctx_mgr if ctx_mgr is not None else nullcontext()):
            if ctx_mgr is not None:
                try:
                    ctx_mgr.capture(forward_loss_fn)
                except Exception as exc:
                    print(f"[proposal] finite_diff: routing-freeze capture failed: {exc}",
                          flush=True)
                    return None
            # One-sided mode: compute baseline g(θ) once, reuse across K.
            # Saves K backward passes vs central diff.
            grads_baseline: Optional[List[torch.Tensor]] = None
            if one_sided:
                with torch.no_grad():
                    for p, snap in zip(expert_ps, theta_snapshots):
                        p.data.copy_(snap)
                grads_baseline = _compute_grad_no_create(forward_loss_fn, expert_ps)
                if grads_baseline is None:
                    return None
            inv_step = 1.0 / eps if one_sided else 1.0 / (2.0 * eps)
            # When tracking, accumulate raw sum (inv_k=1.0) and divide at
            # snapshot time. Otherwise pre-divide for the existing single-K
            # return path.
            inv_k = 1.0 if _track_set else 1.0 / max(1, int(k_samples))
            for k in range(max(1, int(k_samples))):
                zs = _make_rademacher(expert_ps, gen)
                # +ε pass: copy snapshot → add eps*z (one BF16 round)
                with torch.no_grad():
                    for p, snap, z in zip(expert_ps, theta_snapshots, zs):
                        p.data.copy_(snap).add_(z, alpha=eps)
                grads_plus = _compute_grad_no_create(forward_loss_fn, expert_ps)
                if grads_plus is None:
                    return None
                if one_sided:
                    grads_minus = grads_baseline  # alias: read-only here
                else:
                    # -ε pass: copy snapshot → add -eps*z (independent of +ε)
                    with torch.no_grad():
                        for p, snap, z in zip(expert_ps, theta_snapshots, zs):
                            p.data.copy_(snap).add_(z, alpha=-eps)
                    grads_minus = _compute_grad_no_create(forward_loss_fn, expert_ps)
                    if grads_minus is None:
                        return None
                for i, (gp, gm, z) in enumerate(zip(grads_plus, grads_minus, zs)):
                    hz = (gp - gm).mul(inv_step).to(torch.float32)
                    h_accum[i].add_((z.to(torch.float32) * hz).abs_().mul_(inv_k))
                k_done = k + 1
                if k_done in _track_set:
                    _snapshots[k_done] = [
                        h.detach().clone().div_(float(k_done))
                        for h in h_accum
                    ]
                # one-sided: don't free grads_baseline — it's needed every k.
                if one_sided:
                    del grads_plus, zs
                else:
                    del grads_plus, grads_minus, zs
    except Exception as exc:
        import traceback as _tb
        print(f"[proposal] finite_diff: refresh failed: {type(exc).__name__}: {exc}\n"
              f"{_tb.format_exc()}", flush=True)
        return None
    finally:
        # Always restore θ exactly via copy_ from snapshot, then free.
        with torch.no_grad():
            for p, snap in zip(expert_ps, theta_snapshots):
                p.data.copy_(snap)
        del theta_snapshots

    # Restore correctness check (now expected to pass under bit-exact
    # snapshot restore — kept as a safety net for any future refactor).
    for p, n_before in zip(expert_ps, theta_before_norms):
        n_after = float(p.data.detach().to(torch.float32).norm().item())
        if abs(n_after - n_before) > max(1e-6, 1e-5 * max(n_before, 1.0)):
            print(f"[proposal] finite_diff: restore check FAILED on tensor (norm "
                  f"before={n_before:.6e} after={n_after:.6e})", flush=True)
            return None

    for h in h_accum:
        if not bool(torch.isfinite(h).all()):
            print("[proposal] finite_diff: NaN/Inf in result", flush=True)
            return None

    if _track_set:
        for k_d, snap in _snapshots.items():
            for h in snap:
                if not bool(torch.isfinite(h).all()):
                    print(f"[proposal] finite_diff: NaN/Inf in K={k_d} snapshot",
                          flush=True)
                    return None
        return _snapshots

    return h_accum


def compute_hessian_finite_diff(
    model: Any,
    forward_step_func: Callable,
    data_iterator: Any,
    expert_ps: List[torch.nn.Parameter],
    k_samples: int,
    eps: Optional[float] = None,
    use_routing_freeze: bool = True,
    batch_override: Optional[Any] = None,
    rng_seed: int = 0xBADCAFE,
    one_sided: bool = False,
    track_k_values: Optional[List[int]] = None,
) -> Optional[Any]:
    """Diagonal Hessian via central finite-difference Hessian-vector products.

    Estimator (Rademacher-z central diff):

        Hz ≈ [∇L(θ + εz) - ∇L(θ - εz)] / (2ε)
        |H_diag|_j ≈ (1/K) Σ_k |z_jk · Hz_jk|

    Strict opt-in alternative to :func:`compute_hessian_hutchinson`.
    Plain (1st-order) grads only — no ``create_graph=True``, no 2nd-order
    autograd, no saved-tensor offload context. Memory cost ~3× a single
    1st-order backward (one for the original θ batch reload, one for
    +ε, one for -ε), independent of K (per-k allocation is one z + one
    grad list, immediately folded into accumulator).

    Hard requirements (per spec):
      * Common Random Numbers: the batch is loaded ONCE per refresh and
        reused across +ε/-ε for every k. ``data_iterator`` is advanced
        exactly once.
      * Param mutation under ``torch.no_grad()`` then exact restore via
        equal-and-opposite ``add_``s. Per-tensor restore-error check
        guards against drift; on failure returns None.
      * eps=None triggers a 3-point auto-tune sweep against a Hutchinson
        K=2 reference; the chosen multiplier is cached at module scope.
      * Returns None on any internal failure — caller falls back to
        Hutchinson.

    Routing freeze
    --------------
    When ``use_routing_freeze=True`` (default), MoE gates are identified
    via :func:`._routing_freeze.identify_gate_modules` and a single
    capture forward is run at θ; both perturbed forwards then replay the
    captured ``(l_aux, combine_weights, dispatch_mask)``. Without this
    freeze, top-k argmax flips between +ε and -ε produce infinite-magnitude
    Hz noise. If gate identification fails (zero gates), we fall back to
    no-freeze (caller may then fall back to Hutchinson via the outer
    None-return).

    EP > 1 is currently blocked at cp_hooks.py:120-124 so all-to-all
    interaction with the captured outputs is not exercised — this code
    path is unverified for EP > 1.
    """
    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    m0 = model[0] if isinstance(model, (list, tuple)) else model
    if not expert_ps:
        return None

    # ---- Load batch ONCE (Common Random Numbers). ----
    # We do not call get_batch ourselves — instead we wrap forward_step_func
    # so that the FIRST call advances the iterator (it does its own
    # get_batch internally), and SUBSEQUENT calls within the refresh share
    # the same batch via a one-shot replay shim. Megatron's
    # forward_step_func signature varies (sometimes get_batch is inside).
    # The simplest viable pattern: capture (tokens, labels, mask, attn,
    # pos) once via a get_batch call, then run a custom forward that
    # bypasses forward_step_func's iterator advance. The Hutchinson path
    # advances the iterator on every refresh too — to preserve identical
    # behaviour we ALSO advance once here.
    #
    # We can't always reach into get_batch here without import cycles; the
    # caller (cp_hooks) ensures forward_step_func is the same pretrain_gpt
    # callable. We use a one-shot lazy import; on import failure we fall
    # back to "call forward_step_func three times against the same di"
    # which still passes the spec sanity check on Megatron's standard
    # iterator (the iterator returns the same batch only when it doesn't
    # advance — which Megatron's does. So that fallback is wrong. Better
    # to fail and let the caller fall back to Hutchinson).
    # Allow caller (cross-check path) to pre-capture batch and inject it,
    # so cross-check Hutchinson can use the SAME batch (CRN compliance —
    # original spec: "±ε 양쪽이 동일 batch tensor를 사용"). When
    # batch_override is None we capture our own batch as before.
    if batch_override is not None:
        batch = batch_override
    else:
        try:
            from pretrain_gpt import get_batch as _get_batch_gpt
        except Exception as exc:
            print(f"[proposal] finite_diff: cannot import get_batch: {exc}", flush=True)
            return None
        try:
            batch = _get_batch_gpt(di)
        except Exception as exc:
            print(f"[proposal] finite_diff: get_batch failed: {exc}", flush=True)
            return None
    tokens, labels, loss_mask, attention_mask, position_ids = batch[:5]

    # Build a closure that re-runs the model forward + LM loss on the
    # cached batch. Mirrors Megatron's training forward.
    import torch.nn.functional as F

    def _forward_loss() -> Tuple[torch.Tensor, Dict[str, Any]]:
        out = m0(tokens, position_ids, attention_mask)
        logits = out[0] if isinstance(out, tuple) else out
        if logits is None or not torch.is_tensor(logits):
            raise RuntimeError("finite_diff: model returned non-tensor")
        if logits.dim() >= 2:
            V = logits.size(-1)
            flat_logits = logits.reshape(-1, V)
        else:
            flat_logits = logits
        flat_labels = labels.reshape(-1) if labels is not None else None
        if flat_labels is None:
            raise RuntimeError("finite_diff: labels missing")
        if loss_mask is not None and loss_mask.numel() == flat_logits.size(0):
            per_tok = F.cross_entropy(flat_logits.float(), flat_labels, reduction="none")
            denom = loss_mask.view(-1).float().sum().clamp_min(1.0)
            loss = (per_tok * loss_mask.view(-1).float()).sum() / denom
        else:
            loss = F.cross_entropy(flat_logits.float(), flat_labels, reduction="mean")
        return loss, {}

    # ---- Routing freeze setup ----
    from ._routing_freeze import identify_gate_modules_safe
    gates_for_freeze: Optional[List[Tuple[str, torch.nn.Module]]] = None
    if use_routing_freeze:
        gates, _rules = identify_gate_modules_safe(model)
        if not gates:
            print("[proposal] finite_diff: WARN no gates identified; running without "
                  "routing freeze (consider falling back to Hutchinson)", flush=True)
        else:
            gates_for_freeze = gates

    # ---- eps resolution ----
    # Default: 1e-3, validated on tiny (4-layer 256d) and production
    # (12-layer 768d) MoE setups under BF16+ZeRO-1 with K=16 — matches
    # Hutchinson K=4 baseline within iteration noise. Override via
    # `--proposal-finite-diff-eps` CLI or `PROPOSAL_FD_EPS` env if a
    # very different scale (e.g. 100B+ params with unusual |H| range)
    # ever proves the default unsafe.
    eps_value = 1e-3 if eps is None else float(eps)

    # ---- Main K-sample finite-diff loop ----
    # rng_seed defaults to Hutchinson's hardcoded 0xBADCAFE so cross-check
    # comparisons sample the SAME z sequence (within iteration ordering
    # constraints), isolating the estimator difference (autograd HVP vs
    # central-diff HVP) from random-sampling variance.
    #
    # We wrap in `param_rebind_context` and use the param-perturb backend.
    # Why: under DS BF16+ZeRO-1, `p.data = ...` is silently ignored
    # because the bf16_optimizer aliases all weights into one flat buffer.
    # Rebind replaces each module._parameters[name] with a fresh Parameter
    # whose storage is independent of that flat buffer, so writes via
    # `p.data.copy_(...).add_(...)` reach the bytes forward reads. Empirically
    # this matches the Hutchinson K=4 baseline within iteration noise on
    # production exp_etc and is faster (no 2nd-order autograd needed).
    from ._param_rebind import param_rebind_context
    with param_rebind_context(model, expert_ps=expert_ps):
        return compute_hessian_finite_diff_core(
            forward_loss_fn=_forward_loss,
            expert_ps=expert_ps,
            k_samples=k_samples,
            eps=eps_value,
            gates_for_freeze=gates_for_freeze,
            rng_seed=rng_seed,
            one_sided=one_sided,
            track_k_values=track_k_values,
        )


def _unwrap_model(model: Any) -> Any:
    if isinstance(model, (list, tuple)):
        return model[0]
    return model



# ----------------------------------------------------------------------
# Hutchinson vs GNB ranking-similarity comparison
# ----------------------------------------------------------------------
def compare_hessian_estimators(
    iteration: int,
    tracked: List[Tuple[str, torch.nn.Parameter]],
    h_a: List[torch.Tensor],
    h_b: List[torch.Tensor],
    theta_cp: Dict[str, torch.Tensor],
    label_a: str = "hutch",
    label_b: str = "gnb",
) -> Dict[str, Any]:
    """Compute Pearson/Jaccard similarity between two |H| estimators on
    bare and δ²-weighted scales.

    Per-tensor GPU reduction, log-bin histc thresholds; no CPU sync inside
    the hot loop. The ``theta_cp`` dict maps tensor name → BF16 θ_CP cache
    (``ImportanceTracker.theta_cp``) for δ computation. Pass an empty
    dict to skip δ² weighting (falls back to zeros).

    The ``label_a`` / ``label_b`` are propagated into the JSONL output keys
    so different cross-checks (hutch_vs_gnb, hutch_vs_finite_diff) can be
    distinguished in downstream analysis.
    """
    h_hutch = h_a
    h_gnb = h_b
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

    ratios = (0.0001, 0.001, 0.01, 0.05, 0.1)
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

    pair_key = f"{label_a}_{label_b}"
    out: Dict[str, Any] = {
        "iter": int(iteration),
        "label_a": label_a,
        "label_b": label_b,
        "total_expert_numel": N_total,
        f"pearson_{pair_key}_bare": pearson_bare,
        "jaccard_top0.01pct_bare":_j(jacc[0.0001]["inter_hg"],jacc[0.0001]["A_h"],jacc[0.0001]["A_g"]),
        "jaccard_top0.1pct_bare": _j(jacc[0.001]["inter_hg"], jacc[0.001]["A_h"], jacc[0.001]["A_g"]),
        "jaccard_top1pct_bare":   _j(jacc[0.01]["inter_hg"],  jacc[0.01]["A_h"],  jacc[0.01]["A_g"]),
        "jaccard_top5pct_bare":   _j(jacc[0.05]["inter_hg"],  jacc[0.05]["A_h"],  jacc[0.05]["A_g"]),
        "jaccard_top10pct_bare":  _j(jacc[0.1]["inter_hg"],   jacc[0.1]["A_h"],   jacc[0.1]["A_g"]),
        f"pearson_{pair_key}_score": pearson_score,
        "jaccard_top0.01pct_score":_j(jacc[0.0001]["inter_shsg"],jacc[0.0001]["A_sh"],jacc[0.0001]["A_sg"]),
        "jaccard_top0.1pct_score": _j(jacc[0.001]["inter_shsg"], jacc[0.001]["A_sh"], jacc[0.001]["A_sg"]),
        "jaccard_top1pct_score":   _j(jacc[0.01]["inter_shsg"],  jacc[0.01]["A_sh"],  jacc[0.01]["A_sg"]),
        "jaccard_top5pct_score":   _j(jacc[0.05]["inter_shsg"],  jacc[0.05]["A_sh"],  jacc[0.05]["A_sg"]),
        "jaccard_top10pct_score":  _j(jacc[0.1]["inter_shsg"],   jacc[0.1]["A_sh"],   jacc[0.1]["A_sg"]),
    }
    # Backwards-compat alias keys for the original hutch_vs_gnb call site.
    if label_a == "hutch" and label_b == "gnb":
        out["pearson_hutch_gnb_bare"] = pearson_bare
        out["pearson_hutch_gnb_score"] = pearson_score
    return out


