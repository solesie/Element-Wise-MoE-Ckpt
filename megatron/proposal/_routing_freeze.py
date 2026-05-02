"""Routing freeze for MoE gates during finite-difference Hessian estimation.

Why this exists
---------------
DeepSpeed MoE gating (TopKGate) is stochastic in two ways:

  * ``noisy_gate_policy='Jitter'`` adds multiplicative noise per call.
  * top-k argmax over softmax logits flips when logits are perturbed
    near the decision boundary.

Finite-difference Hessian estimation requires evaluating
``∇L(θ + εz)`` and ``∇L(θ - εz)`` — two forwards on the same batch
where the only allowed source of variation is θ. If routing flips
between those two forwards, Hz becomes infinite-magnitude noise and
ruins the estimator. We freeze the routing decision: capture
``(l_aux, combine_weights, dispatch_mask)`` at the unperturbed θ, then
replay it on both perturbed forwards.

Public API
----------
:func:`identify_gate_modules` — walks a (possibly DeepSpeed-wrapped)
model and returns the list of TopKGate-like modules to freeze.

:class:`RoutingFreezeContext` — context manager. ``__enter__`` runs
one capture forward (caller invokes inside the with-block via
``ctx.capture(forward_fn)``) then monkey-patches each gate's
``forward`` to return the captured value. ``__exit__`` always
restores the originals (try/finally semantics).

Edge cases handled
------------------
* Gate output is a tuple of 3 tensors — deep-detached & cloned so
  autograd cannot back-prop through the captured value and CUDA
  doesn't free the memory mid-replay.
* Auxiliary load-balance loss (``l_aux``) is zeroed in the captured
  output so finite-diff sees only the task-loss component (matches
  Hutchinson which also runs only on the language-model loss).
* EP > 1 not exercised — caller blocks at cp_hooks.py:120-124. We
  comment that here for future-proofing.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Gate identification
# ---------------------------------------------------------------------------
_GATE_CLASS_NAME_HINTS = ("TopKGate", "MoEGate", "SparseDispatcher")
_MOE_PARENT_HINTS = ("MoE", "MOELayer", "MoELayer")


def _classify_module(name: str, module: torch.nn.Module,
                     parent_class_name: str) -> Optional[str]:
    """Return the rule that flagged this module as a gate, or None."""
    cls_name = type(module).__name__
    # Rule A: class-name keyword.
    for hint in _GATE_CLASS_NAME_HINTS:
        if hint in cls_name:
            return f"class_name~{hint}"
    # Rule B: module HAS a .gate attribute that is itself a Module
    # (catches a wrapper around an inner gate). We don't return this rule
    # for the wrapper itself — we prefer the inner gate. So skip Rule B
    # and let Rule A pick up the inner.
    # Rule C: parent class name contains MoE marker AND module class name
    # ends with "Gate". Catches subclasses we don't enumerate above.
    if parent_class_name and any(h in parent_class_name for h in _MOE_PARENT_HINTS):
        if cls_name.endswith("Gate"):
            return f"parent_moe~{parent_class_name}"
    return None


def _unwrap_to_inner(model: Any) -> torch.nn.Module:
    """Mirror cp_hooks._unwrap_ds_engine for self-containment."""
    m = model[0] if isinstance(model, (list, tuple)) else model
    while hasattr(m, "module") and not hasattr(m, "named_parameters"):
        m = m.module
    if hasattr(m, "module") and hasattr(m.module, "named_parameters"):
        m = m.module
    return m


def identify_gate_modules(
    model: Any,
) -> Tuple[List[Tuple[str, torch.nn.Module]], Dict[str, int]]:
    """Walk model and return [(qualified_name, gate_module), ...] plus
    rule-breakdown dict ``{rule_name: count}`` for diagnostics.

    The model may be a DeepSpeed engine wrapper, a list, or a plain
    nn.Module. We unwrap once via the same pattern cp_hooks uses.

    Returns ``([], {})`` if no gates are found — caller should warn and
    fall back to no-freeze (or to Hutchinson).
    """
    inner = _unwrap_to_inner(model)
    # Build name → parent map for parent-class-name lookups.
    parent_map: Dict[str, str] = {}
    for name, module in inner.named_modules():
        for child_name, _child in module.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_map[full] = type(module).__name__
    rules: Dict[str, int] = {}
    gates: List[Tuple[str, torch.nn.Module]] = []
    for name, module in inner.named_modules():
        parent_cls = parent_map.get(name, "")
        rule = _classify_module(name, module, parent_cls)
        if rule is None:
            continue
        # Avoid double-flagging: if a parent module already qualifies and
        # this is a child of it, the inner gate is what we want — but we
        # only register modules whose class actually matches Rule A or
        # Rule C. _classify_module already handles that.
        gates.append((name, module))
        rules[rule] = rules.get(rule, 0) + 1
    return gates, rules


# ---------------------------------------------------------------------------
# Capture / replay context
# ---------------------------------------------------------------------------
def _deep_detach_clone(obj: Any) -> Any:
    """Detach + clone every tensor in a nested tuple/list/dict, leave other
    objects untouched. Cloning is necessary because the original is
    autograd-tracked and may be freed when its graph is dropped.
    """
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, tuple):
        return tuple(_deep_detach_clone(o) for o in obj)
    if isinstance(obj, list):
        return [_deep_detach_clone(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _deep_detach_clone(v) for k, v in obj.items()}
    return obj


def _zero_aux_loss(captured: Any) -> Any:
    """For TopKGate the output is (l_aux, combine_weights, dispatch_mask).
    Zero out l_aux so the replay doesn't contribute load-balance loss to
    the finite-diff numerator. We detect this shape heuristically: a
    3-tuple whose first entry is a 0-dim tensor.
    """
    if isinstance(captured, tuple) and len(captured) == 3 and torch.is_tensor(captured[0]):
        l_aux = captured[0]
        if l_aux.dim() == 0:
            return (torch.zeros_like(l_aux), captured[1], captured[2])
    return captured


class RoutingFreezeContext:
    """Context manager that captures gate outputs at θ and replays them
    during ``+ε`` / ``-ε`` perturbed forwards.

    Usage::

        gates, rules = identify_gate_modules(model)
        with RoutingFreezeContext(gates) as ctx:
            ctx.capture(lambda: forward_step_func(di, m0))
            # All subsequent forwards inside the with-block return the
            # captured gate outputs.
            loss_plus  = forward_step_func(di, m0)[0]
            loss_minus = forward_step_func(di, m0)[0]
    """

    def __init__(self, gates: List[Tuple[str, torch.nn.Module]]) -> None:
        self._gates = gates
        self._originals: Dict[int, Callable] = {}
        self._captured: Dict[int, Any] = {}
        self._capture_hook_handles: List[Any] = []

    def __enter__(self) -> "RoutingFreezeContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Always remove forward hooks AND restore monkey-patched forwards.
        for h in self._capture_hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._capture_hook_handles.clear()
        for mid, fwd in self._originals.items():
            module = self._lookup_by_id(mid)
            if module is not None:
                try:
                    module.forward = fwd
                except Exception:
                    pass
        self._originals.clear()
        self._captured.clear()

    def _lookup_by_id(self, mid: int) -> Optional[torch.nn.Module]:
        for _name, m in self._gates:
            if id(m) == mid:
                return m
        return None

    def capture(self, forward_fn: Callable[[], Any]) -> Any:
        """Run a single forward pass at θ to capture each gate's output.
        Hooks are removed before this method returns; the captured
        outputs then drive replay via :meth:`install_replay`.
        """
        self._captured.clear()
        # Install a transient forward hook on each gate.
        def _make_hook(mid: int):
            def _hook(_module, _inputs, output):
                self._captured[mid] = _zero_aux_loss(_deep_detach_clone(output))
            return _hook
        for _name, module in self._gates:
            h = module.register_forward_hook(_make_hook(id(module)))
            self._capture_hook_handles.append(h)
        try:
            with torch.no_grad():
                out = forward_fn()
        finally:
            for h in self._capture_hook_handles:
                try: h.remove()
                except Exception: pass
            self._capture_hook_handles.clear()
        # Now install monkey-patched forwards for replay.
        self._install_replay()
        return out

    def _install_replay(self) -> None:
        for _name, module in self._gates:
            mid = id(module)
            if mid not in self._captured:
                # Module was not invoked during capture — skip; its
                # original forward stays in place.
                continue
            self._originals[mid] = module.forward
            captured_value = self._captured[mid]
            def _replay(*_args, _v=captured_value, **_kwargs):
                return _v
            module.forward = _replay  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Force-empty identifier (test injection point)
# ---------------------------------------------------------------------------
_FORCE_EMPTY = False


def _set_force_empty(v: bool) -> None:
    """Used by the failure-case unit test: temporarily force the
    identifier to return []."""
    global _FORCE_EMPTY
    _FORCE_EMPTY = bool(v)


def identify_gate_modules_safe(
    model: Any,
) -> Tuple[List[Tuple[str, torch.nn.Module]], Dict[str, int]]:
    """Public-facing wrapper that honours the test-injection flag."""
    if _FORCE_EMPTY:
        return [], {}
    return identify_gate_modules(model)
