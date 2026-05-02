"""Activation perturbation hooks for finite-difference Hessian estimation.

Why this exists
---------------
The straightforward finite-diff implementation (perturb ``param.data``
in place, measure forward gradient) does NOT work under DeepSpeed
BF16 + ZeRO-1: the BF16 ``.data`` tensor returned by
``model.named_parameters()`` is a stale view, and the FP32 master
returned by the Fp32StateMapper is also unreliable post-step. Forward
reads from a third buffer (DeepSpeed's BF16Optimizer flat group)
that is not exposed via the standard parameter API. Diagnostic in
``results/finite_diff_diag/`` showed both BF16 .data and FP32 master
collapse to ~1e-5 / ~1e-10 respectively after a few optimizer steps
while training proceeds normally — proof that forward uses neither.

Mathematical rewrite
--------------------
For an affine layer ``y = W x + b``, perturbing ``W`` by ``εz`` and
adding ``ε(zx)`` to the forward output are mathematically identical:

    (W + εz) x + b  ==  W x + b + ε z x  ==  y + ε(z x)

So instead of touching the (inaccessible) weight buffer, we register a
forward hook on each expert Linear layer that adds ``ε(zx)`` (and
``εz_b`` for bias) to the output. Autograd then computes
``∇L(θ + εz)`` correctly because:

  * forward output downstream of the perturbed layer is identical to
    the true-perturb case (same input to next layer);
  * gradient ``∂L/∂W`` via chain rule is ``(∂L/∂y_pert) · x.T`` for
    both methods (the additive constant ``ε(zx)`` has no W-dependency);
  * gradient ``∂L/∂x`` flowing back through the layer picks up the
    ``εz.T`` contribution from the ``x @ z.T`` term — matching the
    true-perturb chain rule which gives ``(W + εz).T = W.T + εz.T``.

Limitations
-----------
* Only works for affine ops (Linear, Conv). MoE expert FFN is a
  Linear-Activation-Linear block, so all expert weight & bias params
  are covered.
* Routing freeze is still required for MoE: without it, top-k argmax
  flips between +ε and -ε passes ruin the central-difference estimate.
  Both hook sets coexist on different modules (gate vs expert linear)
  with no conflict.
* Captures a Rademacher z per expert linear weight + bias and reuses it
  across the +ε / -ε pair (Common Random Numbers).

Public API
----------
:func:`identify_expert_linear_modules` — walks a (DeepSpeed-wrapped)
model, returns the list of (qualified_name, module, role_to_param)
triples for every expert Linear-like layer.

:class:`ActivationPerturbContext` — context manager. Sample z per
target param via :meth:`sample_z`, then activate hooks for a signed
ε via :meth:`install(eps_signed)`. Hooks survive until :meth:`remove`
or context exit. Hooks for the +ε pass and -ε pass use the SAME z
tensors (CRN); only the sign of ε changes.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Module identification
# ---------------------------------------------------------------------------
_EXPERT_NAME_HINTS = ("deepspeed_experts.", )


def _unwrap_to_inner(model: Any) -> torch.nn.Module:
    """Mirror ``_routing_freeze._unwrap_to_inner`` for self-containment."""
    m = model[0] if isinstance(model, (list, tuple)) else model
    while hasattr(m, "module") and not hasattr(m, "named_parameters"):
        m = m.module
    if hasattr(m, "module") and hasattr(m.module, "named_parameters"):
        m = m.module
    return m


def _is_linear_like(module: torch.nn.Module) -> bool:
    """Return True if the module exposes a 2-D ``weight`` attribute and
    behaves like ``y = x @ weight.T + bias``. Catches ``nn.Linear``,
    ``ColumnParallelLinear``, custom expert linear shims, etc.
    """
    w = getattr(module, "weight", None)
    if w is None or not torch.is_tensor(w):
        return False
    return w.dim() == 2


def identify_expert_linear_modules(
    model: Any,
) -> Tuple[List[Tuple[str, torch.nn.Module, Dict[str, torch.nn.Parameter]]],
           Dict[str, int]]:
    """Walk model, return ``[(qualified_name, module, role_to_param), ...]``
    for every expert Linear-like layer. ``role_to_param`` is
    ``{"weight": p_w}`` or ``{"weight": p_w, "bias": p_b}``.

    Returns ``([], {})`` if nothing is identified — caller should warn
    and fall back (e.g. to Hutchinson).
    """
    inner = _unwrap_to_inner(model)
    rules: Dict[str, int] = {}
    out: List[Tuple[str, torch.nn.Module, Dict[str, torch.nn.Parameter]]] = []
    for name, module in inner.named_modules():
        # Only consider modules in expert subtrees.
        if not any(h in name for h in _EXPERT_NAME_HINTS):
            continue
        if not _is_linear_like(module):
            continue
        roles: Dict[str, torch.nn.Parameter] = {"weight": module.weight}
        b = getattr(module, "bias", None)
        if b is not None and torch.is_tensor(b):
            roles["bias"] = b
        out.append((name, module, roles))
        rules["expert_linear"] = rules.get("expert_linear", 0) + 1
    return out, rules


# ---------------------------------------------------------------------------
# Force-empty identifier (test injection point)
# ---------------------------------------------------------------------------
_FORCE_EMPTY = False


def _set_force_empty(v: bool) -> None:
    global _FORCE_EMPTY
    _FORCE_EMPTY = bool(v)


def identify_expert_linear_modules_safe(
    model: Any,
) -> Tuple[List[Tuple[str, torch.nn.Module, Dict[str, torch.nn.Parameter]]],
           Dict[str, int]]:
    if _FORCE_EMPTY:
        return [], {}
    return identify_expert_linear_modules(model)


# ---------------------------------------------------------------------------
# Activation perturbation context
# ---------------------------------------------------------------------------
class ActivationPerturbContext:
    """Manages the per-layer Rademacher z + signed forward hook.

    Workflow per refresh::

        ctx = ActivationPerturbContext(expert_modules, param_to_idx)
        ctx.sample_z(generator)           # one Rademacher per (module, role)
        ctx.install(eps_signed=+eps)      # hooks attach
        try: grads_plus = compute_grad(...)
        finally: ctx.remove()
        ctx.install(eps_signed=-eps)      # same z, opposite sign
        try: grads_minus = compute_grad(...)
        finally: ctx.remove()
        # repeat for K samples by re-calling sample_z + the two installs.

    The mapping ``param_to_idx`` is ``{id(param): index_in_expert_ps}``;
    used by the caller to align ``z`` (per-param) with the accumulator
    ``h_accum[idx]``.
    """

    def __init__(
        self,
        expert_modules: List[Tuple[str, torch.nn.Module,
                                    Dict[str, torch.nn.Parameter]]],
        param_to_idx: Dict[int, int],
    ) -> None:
        self._modules = expert_modules
        self._param_to_idx = param_to_idx
        # Per-(module_id, role) z tensor, FP32 on the param's device.
        self._z: Dict[Tuple[int, str], torch.Tensor] = {}
        self._handles: List[Any] = []

    def sample_z(self, generator: torch.Generator) -> None:
        """(Re)populate ``self._z`` with Rademacher tensors matching each
        tracked param's shape. Call ONCE per K-sample iteration; reuse
        the same z for the +ε and -ε pair (Common Random Numbers).
        """
        self._z.clear()
        for _name, module, roles in self._modules:
            mid = id(module)
            for role, p in roles.items():
                if id(p) not in self._param_to_idx:
                    continue
                # Rademacher ±1 on the param's device, FP32 for clean math.
                z = (
                    torch.randint(
                        0, 2, p.shape, generator=generator,
                        device=p.device, dtype=torch.float32,
                    ) * 2 - 1
                )
                self._z[(mid, role)] = z

    def install(self, eps_signed: float) -> None:
        """Attach forward hooks that add ``ε_signed * (z_w x + z_b)`` to
        each expert Linear's output. Idempotent against double-install
        (auto-removes any prior batch first).
        """
        if self._handles:
            self.remove()
        # Counter to track if hooks actually fire (debug aid).
        self._fire_counter = [0]
        for _name, module, roles in self._modules:
            mid = id(module)
            z_w = self._z.get((mid, "weight"))
            z_b = self._z.get((mid, "bias"))
            if z_w is None and z_b is None:
                continue
            handle = module.register_forward_hook(
                _make_perturb_hook(z_w, z_b, eps_signed, self._fire_counter)
            )
            self._handles.append(handle)

    def fire_count(self) -> int:
        return getattr(self, "_fire_counter", [0])[0]

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def z_by_param_idx(self) -> Dict[int, torch.Tensor]:
        """Return ``{idx_in_expert_ps: z_tensor}`` for the most recent
        sampled z. Caller uses this to compute ``|z·Hz|`` per param when
        accumulating ``|H_diag|``.
        """
        out: Dict[int, torch.Tensor] = {}
        for (mid, role), z in self._z.items():
            for _name, module, roles in self._modules:
                if id(module) != mid:
                    continue
                p = roles.get(role)
                if p is None:
                    continue
                idx = self._param_to_idx.get(id(p))
                if idx is None:
                    continue
                out[idx] = z
                break
        return out

    def __enter__(self) -> "ActivationPerturbContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()
        self._z.clear()


def _make_perturb_hook(
    z_w: Optional[torch.Tensor],
    z_b: Optional[torch.Tensor],
    eps_signed: float,
    fire_counter: Optional[List[int]] = None,
):
    """Closure that adds ``eps_signed * (x @ z_w.T + z_b)`` to the
    layer's forward output. Computed entirely in FP32 against an FP32
    cast of ``output`` to avoid the BF16-cast-then-add round-off that
    silently truncates small perturbations to zero.

    Handles both plain-tensor and tuple outputs. Megatron's
    ``ColumnParallelLinear`` / ``RowParallelLinear`` return
    ``(output_tensor, output_bias)`` tuples — the first element is the
    matmul result we need to perturb; the second is the bias to be
    added externally by the caller (or ``None`` when skip_bias_add is
    off). We rebuild the tuple with the perturbed first element so the
    downstream consumer sees the correct shape.
    """
    def _perturb_tensor(x_in: torch.Tensor, out_t: torch.Tensor) -> torch.Tensor:
        """Compute the perturbed output for the canonical y = x @ W.T + b form."""
        x32 = x_in.detach().to(torch.float32)
        perturb_fp32: Optional[torch.Tensor] = None
        if z_w is not None:
            # x: (..., in_features), z_w: (out_features, in_features)
            perturb_fp32 = x32.matmul(z_w.t())
        if z_b is not None:
            zb32 = z_b.to(torch.float32)
            perturb_fp32 = (
                zb32 if perturb_fp32 is None else perturb_fp32 + zb32
            )
        if perturb_fp32 is None:
            return out_t
        # FP32 add then cast back to preserve sub-ULP perturbations.
        out_fp32 = out_t.to(torch.float32) + eps_signed * perturb_fp32
        return out_fp32.to(out_t.dtype)

    def hook(module: torch.nn.Module, args, output):
        # args[0] is the layer's primary input tensor (BF16 typically).
        x = args[0] if (isinstance(args, tuple) and args) else None
        if x is None:
            return output
        # ColumnParallelLinear / RowParallelLinear return (out, bias_or_None).
        # We perturb the first element and rebuild the tuple.
        if isinstance(output, tuple):
            if not output or not torch.is_tensor(output[0]):
                return output
            if fire_counter is not None:
                fire_counter[0] += 1
            new_first = _perturb_tensor(x, output[0])
            return (new_first,) + tuple(output[1:])
        if not torch.is_tensor(output):
            return output
        if fire_counter is not None:
            fire_counter[0] += 1
        return _perturb_tensor(x, output)
    return hook
