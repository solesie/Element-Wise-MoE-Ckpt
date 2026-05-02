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

from typing import Any, Callable, Dict, List, Optional, Tuple

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
# Finite-difference estimator (opt-in alternative to Hutchinson)
# ----------------------------------------------------------------------
# Module-level cache for the auto-tuned eps multiplier. Set on the first
# call when eps=None, reused on all subsequent calls so we don't redo the
# 3-point sweep every refresh. Keyed by id(model) so distinct models in
# the same process get independent locks.
_EPS_CACHE: Dict[int, float] = {}


def _norm_mean_theta(params: List[torch.nn.Parameter]) -> float:
    """Mean of per-tensor ||θ||_F. Base scale for the eps auto-tune sweep."""
    if not params:
        return 1.0
    norms = []
    for p in params:
        try:
            norms.append(float(p.data.detach().to(torch.float32).norm().item()))
        except Exception:
            norms.append(0.0)
    n = sum(norms) / max(len(norms), 1)
    return max(n, 1e-12)


_HOOK_PROBE_DONE = False


def _probe_hook_levels(
    model: Any, forward_loss_fn: Callable,
) -> None:
    """One-shot diagnostic: install passive log-only forward hooks on
    EVERY module under the "deepspeed_experts" subtree, run ONE forward,
    print which hooks actually fired. Identifies the right module level
    for activation perturbation when the inner Linear hooks don't fire
    (DeepSpeed MoE may bypass module.__call__ for fused expert paths).
    """
    global _HOOK_PROBE_DONE
    if _HOOK_PROBE_DONE:
        return
    _HOOK_PROBE_DONE = True

    inner = _unwrap_model(model)
    while hasattr(inner, "module") and not hasattr(inner, "named_parameters"):
        inner = inner.module
    if hasattr(inner, "module") and hasattr(inner.module, "named_parameters"):
        inner = inner.module

    fired: List[str] = []
    handles = []
    name_to_class: Dict[str, str] = {}
    candidates: List[Tuple[str, torch.nn.Module]] = []
    for name, module in inner.named_modules():
        # MoE-relevant subtree only (avoid hooking the entire model).
        if not any(kw in name for kw in
                   ("deepspeed_experts", "deepspeed_moe", "experts", "MoE", "moe")):
            continue
        candidates.append((name, module))
        name_to_class[name] = type(module).__name__

    print(f"\n[HOOK-PROBE] candidates: {len(candidates)} modules under MoE/expert subtree",
          flush=True)

    def make_hook(name):
        def hook(_m, _args, _output):
            fired.append(name)
            return None
        return hook

    for name, module in candidates:
        try:
            handles.append(module.register_forward_hook(make_hook(name)))
        except Exception:
            pass

    try:
        with torch.no_grad():
            _ = forward_loss_fn()
    except Exception as exc:
        print(f"[HOOK-PROBE] forward failed during probe: {exc}", flush=True)
    finally:
        for h in handles:
            try: h.remove()
            except Exception: pass

    # Group fired modules by class name for compact output.
    fired_by_class: Dict[str, List[str]] = {}
    for n in fired:
        cls = name_to_class.get(n, "?")
        fired_by_class.setdefault(cls, []).append(n)
    print(f"[HOOK-PROBE] FIRED ({len(fired)} hook events, "
          f"{len(set(fired))} distinct modules):", flush=True)
    for cls, names in fired_by_class.items():
        print(f"[HOOK-PROBE]   class={cls}  count={len(names)}  "
              f"sample={names[0] if names else '-'}", flush=True)

    # Did NOT fire (subset; avoid spam).
    not_fired = [n for n, _ in candidates if n not in set(fired)]
    print(f"[HOOK-PROBE] NOT-fired: {len(not_fired)} modules. Sample classes:",
          flush=True)
    seen_cls = set()
    for n in not_fired:
        cls = name_to_class.get(n, "?")
        if cls in seen_cls: continue
        seen_cls.add(cls)
        print(f"[HOOK-PROBE]   class={cls}  e.g. {n}", flush=True)
        if len(seen_cls) >= 8: break
    print("", flush=True)


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


def compute_hessian_finite_diff_activation_core(
    forward_loss_fn: Callable[[], Tuple[torch.Tensor, Any]],
    expert_ps: List[torch.nn.Parameter],
    expert_modules: List[Tuple[str, torch.nn.Module,
                                Dict[str, torch.nn.Parameter]]],
    k_samples: int,
    eps: float,
    gates_for_freeze: Optional[List[Tuple[str, torch.nn.Module]]] = None,
    rng_seed: int = 0xFD15CAFE,
) -> Optional[List[torch.Tensor]]:
    """Activation-perturbation backend (PRIMARY path).

    Mathematically equivalent to true param perturbation
    ``W ← W + εz`` for affine layers — see ``_activation_perturb`` for
    the algebra. Crucially this avoids touching ``param.data`` or any
    DeepSpeed BF16/FP32 master buffer, so it works under the
    BF16+ZeRO-1 setup where param perturbation silently fails (the
    perturbation never reaches the buffer that forward actually reads).

    Inputs
    ------
    expert_ps:
        Same list the caller maintains for h_accum indexing. Order
        determines the returned list.
    expert_modules:
        Output of :func:`._activation_perturb.identify_expert_linear_modules`.
        Each entry is ``(qualified_name, module, {"weight": p, ...})``.
        Hooks attach to these modules; per-(module, role) Rademacher z
        is tied back to ``expert_ps`` indices via ``id(p)``.

    Returns ``[|H_diag|_p_FP32 for p in expert_ps]`` or ``None``. Params
    not represented in ``expert_modules`` (e.g. gate weights, embeddings
    that slipped past the filter) get a zero ``|H_diag|`` accumulator —
    same convention as the param-perturb backend's "untouched param"
    behavior, so the H_ema fold's coverage check flags them.
    """
    if not expert_ps or eps <= 0:
        return None
    from contextlib import nullcontext
    from ._routing_freeze import RoutingFreezeContext
    from ._activation_perturb import ActivationPerturbContext

    if not expert_modules:
        print("[proposal] finite_diff: activation perturb requires "
              "expert_modules; returning None", flush=True)
        return None

    # Index map for tying per-(module, role) z back to h_accum slots.
    param_to_idx: Dict[int, int] = {id(p): i for i, p in enumerate(expert_ps)}

    h_accum = [torch.zeros_like(p, dtype=torch.float32) for p in expert_ps]
    gen = torch.Generator(device=expert_ps[0].device).manual_seed(rng_seed)

    if gates_for_freeze:
        ctx_mgr = RoutingFreezeContext(gates_for_freeze)
    else:
        ctx_mgr = None

    perturb_ctx = ActivationPerturbContext(expert_modules, param_to_idx)

    try:
        with (ctx_mgr if ctx_mgr is not None else nullcontext()):
            if ctx_mgr is not None:
                try:
                    ctx_mgr.capture(forward_loss_fn)
                except Exception as exc:
                    print(f"[proposal] finite_diff: routing-freeze capture failed: {exc}",
                          flush=True)
                    return None
            for k in range(max(1, int(k_samples))):
                perturb_ctx.sample_z(gen)
                z_by_idx = perturb_ctx.z_by_param_idx()

                # +ε pass
                perturb_ctx.install(eps_signed=+eps)
                try:
                    grads_plus = _compute_grad_no_create(forward_loss_fn, expert_ps)
                finally:
                    fire_p = perturb_ctx.fire_count()
                    perturb_ctx.remove()
                # -ε pass with same z, opposite sign
                perturb_ctx.install(eps_signed=-eps)
                try:
                    grads_minus = _compute_grad_no_create(forward_loss_fn, expert_ps)
                finally:
                    fire_m = perturb_ctx.fire_count()
                    perturb_ctx.remove()
                if k == 0:
                    print(f"[proposal] finite_diff: hooks fired +ε:{fire_p} -ε:{fire_m} "
                          f"(modules: {len(expert_modules)})", flush=True)
                if grads_plus is None or grads_minus is None:
                    return None

                inv_2eps = 1.0 / (2.0 * eps)
                inv_k = 1.0 / max(1, int(k_samples))
                for idx in range(len(expert_ps)):
                    z = z_by_idx.get(idx)
                    if z is None:
                        continue  # param outside hooked modules → zero accumulator
                    gp = grads_plus[idx]
                    gm = grads_minus[idx]
                    if gp is None or gm is None:
                        continue
                    # Subtract in FP32: when gp ≈ gm (catastrophic cancellation
                    # under BF16 grad noise), upcasting both operands first
                    # preserves the result's full dynamic range. Cheap (BF16→FP32
                    # is lossless), no extra memory beyond one extra tensor.
                    hz = (gp.to(torch.float32) - gm.to(torch.float32)).mul_(inv_2eps)
                    h_accum[idx].add_((z.to(torch.float32) * hz).abs_().mul_(inv_k))
                del grads_plus, grads_minus
    except Exception as exc:
        import traceback as _tb
        print(f"[proposal] finite_diff: refresh failed: {type(exc).__name__}: {exc}\n"
              f"{_tb.format_exc()}", flush=True)
        return None
    finally:
        perturb_ctx.remove()

    for h in h_accum:
        if not bool(torch.isfinite(h).all()):
            print("[proposal] finite_diff: NaN/Inf in result", flush=True)
            return None

    return h_accum


def compute_hessian_finite_diff_core(
    forward_loss_fn: Callable[[], Tuple[torch.Tensor, Any]],
    expert_ps: List[torch.nn.Parameter],
    k_samples: int,
    eps: float,
    gates_for_freeze: Optional[List[Tuple[str, torch.nn.Module]]] = None,
    rng_seed: int = 0xFD15CAFE,
) -> Optional[List[torch.Tensor]]:
    """Backend that runs the central-difference loop on a caller-supplied
    forward closure. Used by both :func:`compute_hessian_finite_diff` (via
    pretrain_gpt batch capture) and the unit tests (via a synthetic
    forward closure that needs no Megatron import).

    ``forward_loss_fn`` MUST close over a fixed batch (Common Random
    Numbers requirement). It must return ``(scalar_loss_tensor, ...)``.
    ``gates_for_freeze`` is the list returned by
    :func:`identify_gate_modules`; when None, no routing freeze is
    applied. ``eps`` must be > 0.

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
    try:
        with (ctx_mgr if ctx_mgr is not None else nullcontext()):
            if ctx_mgr is not None:
                try:
                    ctx_mgr.capture(forward_loss_fn)
                except Exception as exc:
                    print(f"[proposal] finite_diff: routing-freeze capture failed: {exc}",
                          flush=True)
                    return None
            for k in range(max(1, int(k_samples))):
                zs = _make_rademacher(expert_ps, gen)
                # +ε pass: copy snapshot → add eps*z (one BF16 round)
                with torch.no_grad():
                    for p, snap, z in zip(expert_ps, theta_snapshots, zs):
                        p.data.copy_(snap).add_(z, alpha=eps)
                grads_plus = _compute_grad_no_create(forward_loss_fn, expert_ps)
                # -ε pass: copy snapshot → add -eps*z (independent of +ε)
                with torch.no_grad():
                    for p, snap, z in zip(expert_ps, theta_snapshots, zs):
                        p.data.copy_(snap).add_(z, alpha=-eps)
                grads_minus = _compute_grad_no_create(forward_loss_fn, expert_ps)
                if grads_plus is None or grads_minus is None:
                    return None
                inv_2eps = 1.0 / (2.0 * eps)
                inv_k = 1.0 / max(1, int(k_samples))
                for i, (gp, gm, z) in enumerate(zip(grads_plus, grads_minus, zs)):
                    hz = (gp - gm).mul_(inv_2eps).to(torch.float32)
                    h_accum[i].add_((z.to(torch.float32) * hz).abs_().mul_(inv_k))
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
) -> Optional[List[torch.Tensor]]:
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

    # ---- One-shot diagnostic: which modules actually fire forward hooks? ----
    _probe_hook_levels(model, _forward_loss)

    # ---- Expert Linear module identification (activation perturb path) ----
    from ._activation_perturb import identify_expert_linear_modules_safe
    expert_modules, _expert_rules = identify_expert_linear_modules_safe(model)
    if not expert_modules:
        print("[proposal] finite_diff: WARN no expert Linear modules identified; "
              "activation perturb path requires them — falling back to None "
              "(caller will use Hutchinson)", flush=True)
        return None
    # One-time identifier debug.
    if not getattr(compute_hessian_finite_diff, "_announced", False):
        sample = [n for n, _m, _r in expert_modules[:5]]
        print(f"[proposal] finite_diff: identified {len(expert_modules)} expert Linear modules "
              f"(rules={_expert_rules}). Sample: {sample}", flush=True)
        compute_hessian_finite_diff._announced = True  # type: ignore[attr-defined]

    # ---- eps resolution ----
    # Cache stores the ABSOLUTE eps (not a multiplier × ||θ||_mean) for
    # activation-perturb. Reason: ||θ||_mean from param.data is unreliable
    # under DeepSpeed BF16 + Megatron Float16Module wrapping (stale view
    # post-step), so multiplying it gives drifting eps values that fall
    # below the BF16 ULP at output scale. The absolute eps from a
    # Hutchinson-anchored auto-tune sweep at iter 1 is the right scale
    # for activation perturbation since the perturbation magnitude
    # ε(x @ z.T) depends on x and z, not on ||θ||.
    cached_eps = _EPS_CACHE.get(id(_unwrap_model(model)))
    if eps is None and cached_eps is not None:
        eps_value = cached_eps
    elif eps is None:
        # Auto-tune sweep against a Hutchinson reference (returns absolute).
        eps_value = _autotune_eps(model, forward_step_func, data_iterator,
                                  expert_ps, batch_cached=batch,
                                  forward_loss_fn=_forward_loss,
                                  gates_for_freeze=gates_for_freeze,
                                  expert_modules=expert_modules)
        if eps_value is None:
            return None
    else:
        eps_value = float(eps)

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
    with param_rebind_context(model, expert_ps=expert_ps,
                               expert_modules=expert_modules):
        return compute_hessian_finite_diff_core(
            forward_loss_fn=_forward_loss,
            expert_ps=expert_ps,
            k_samples=k_samples,
            eps=eps_value,
            gates_for_freeze=gates_for_freeze,
            rng_seed=rng_seed,
        )


def _unwrap_model(model: Any) -> Any:
    if isinstance(model, (list, tuple)):
        return model[0]
    return model


def _autotune_eps(
    model: Any,
    forward_step_func: Callable,
    data_iterator: Any,
    expert_ps: List[torch.nn.Parameter],
    batch_cached: Any,
    forward_loss_fn: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
    gates_for_freeze: Optional[List[Tuple[str, torch.nn.Module]]],
    expert_modules: Optional[List[Tuple[str, torch.nn.Module,
                                          Dict[str, torch.nn.Parameter]]]] = None,
) -> Optional[float]:
    """3-point sweep over eps ∈ {1e-2, 1e-3, 1e-4} × ||θ||_mean. Computes a
    Hutchinson K=2 reference once, then for each eps runs a 1-sample
    finite-diff probe (via :func:`compute_hessian_finite_diff_core`) and
    measures Pearson against the reference. Returns the absolute eps with
    the highest Pearson, caches the multiplier in ``_EPS_CACHE``. Returns
    None if every probe fails.
    """
    base = _norm_mean_theta(expert_ps)
    # 5-point sweep with intermediate half-decade points. Original 3-point
    # log sweep (1e-2, 1e-3, 1e-4) misses sweet spots between adjacent
    # decades (e.g. 5e-3 or 5e-4) where the round-off and truncation
    # error envelopes cross. Extra cost is one-time at iter 1 only;
    # cached thereafter via _EPS_CACHE.
    multipliers = (1e-2, 5e-3, 1e-3, 5e-4, 1e-4)
    ref = compute_hessian_hutchinson(model, forward_step_func, data_iterator,
                                      expert_ps, k_samples=2)
    if ref is None:
        print("[proposal] finite_diff: auto-tune Hutchinson reference FAILED — "
              "cannot pick eps", flush=True)
        return None

    best_pearson = -2.0
    best_mult = None
    pearsons: Dict[float, Optional[float]] = {}
    for mult in multipliers:
        eps_try = mult * base
        try:
            # Activation-perturb path requires expert_modules; fall back to
            # the legacy param-perturb core only when the caller didn't
            # provide them (e.g. unit tests on FP32 toy models).
            if expert_modules is not None:
                probe = compute_hessian_finite_diff_activation_core(
                    forward_loss_fn=forward_loss_fn,
                    expert_ps=expert_ps,
                    expert_modules=expert_modules,
                    k_samples=1,
                    eps=eps_try,
                    gates_for_freeze=gates_for_freeze,
                    rng_seed=0xEC0EC0EC + int(mult * 1e6),
                )
            else:
                probe = compute_hessian_finite_diff_core(
                    forward_loss_fn=forward_loss_fn,
                    expert_ps=expert_ps,
                    k_samples=1,
                    eps=eps_try,
                    gates_for_freeze=gates_for_freeze,
                    rng_seed=0xEC0EC0EC + int(mult * 1e6),
                )
            if probe is None:
                pearsons[mult] = None
                continue
            r = _pearson_two_tensor_lists(probe, ref)
            pearsons[mult] = r
            if r > best_pearson:
                best_pearson = r
                best_mult = mult
        except torch.cuda.OutOfMemoryError as exc:
            print(f"[proposal] finite_diff: eps={eps_try:.3e} OOM, skipping ({exc})", flush=True)
            torch.cuda.empty_cache()
            pearsons[mult] = None
            continue
        except Exception as exc:
            print(f"[proposal] finite_diff: eps={eps_try:.3e} probe failed: {exc}", flush=True)
            pearsons[mult] = None
            continue
    if best_mult is None:
        print("[proposal] finite_diff: auto-tune found no viable eps", flush=True)
        return None
    best_eps = best_mult * base
    print(f"[proposal] finite_diff: auto-tune Pearson = {pearsons} "
          f"→ chose mult={best_mult:.0e} eps={best_eps:.3e}", flush=True)
    # Cache ABSOLUTE eps (not multiplier) — see compute_hessian_finite_diff
    # for the rationale. ||θ||_mean from param.data is unreliable under
    # DeepSpeed BF16 + Megatron Float16Module wrapping, so multiplying it
    # gives drifting eps. Storing the absolute value picked at iter 1
    # keeps the perturbation magnitude stable across all subsequent
    # refreshes (which is what activation perturb needs — the eps is a
    # function of x and z magnitudes, not θ magnitudes).
    _EPS_CACHE[id(_unwrap_model(model))] = best_eps
    return best_eps


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

    pair_key = f"{label_a}_{label_b}"
    out: Dict[str, Any] = {
        "iter": int(iteration),
        "label_a": label_a,
        "label_b": label_b,
        "total_expert_numel": N_total,
        f"pearson_{pair_key}_bare": pearson_bare,
        "jaccard_top0.1pct_bare": _j(jacc[0.001]["inter_hg"], jacc[0.001]["A_h"], jacc[0.001]["A_g"]),
        "jaccard_top1pct_bare":   _j(jacc[0.01]["inter_hg"],  jacc[0.01]["A_h"],  jacc[0.01]["A_g"]),
        "jaccard_top10pct_bare":  _j(jacc[0.1]["inter_hg"],   jacc[0.1]["A_h"],   jacc[0.1]["A_g"]),
        f"pearson_{pair_key}_score": pearson_score,
        "jaccard_top0.1pct_score": _j(jacc[0.001]["inter_shsg"], jacc[0.001]["A_sh"], jacc[0.001]["A_sg"]),
        "jaccard_top1pct_score":   _j(jacc[0.01]["inter_shsg"],  jacc[0.01]["A_sh"],  jacc[0.01]["A_sg"]),
        "jaccard_top10pct_score":  _j(jacc[0.1]["inter_shsg"],   jacc[0.1]["A_sh"],   jacc[0.1]["A_sg"]),
    }
    # Backwards-compat alias keys for the original hutch_vs_gnb call site.
    if label_a == "hutch" and label_b == "gnb":
        out["pearson_hutch_gnb_bare"] = pearson_bare
        out["pearson_hutch_gnb_score"] = pearson_score
    return out


def compare_hutchinson_vs_gnb(
    iteration: int,
    tracked: List[Tuple[str, torch.nn.Parameter]],
    h_hutch: List[torch.Tensor],
    h_gnb: List[torch.Tensor],
    theta_cp: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Backwards-compat wrapper around :func:`compare_hessian_estimators`
    with ``label_a='hutch'`` / ``label_b='gnb'``. Preserved so existing
    cp_hooks call sites and analysis scripts that consume the
    ``pearson_hutch_gnb_*`` keys continue to work.
    """
    return compare_hessian_estimators(
        iteration, tracked, h_hutch, h_gnb, theta_cp,
        label_a="hutch", label_b="gnb",
    )
