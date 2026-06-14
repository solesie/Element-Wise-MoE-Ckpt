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
  output so finite-diff sees only the task-loss component.
* EP > 1 is blocked by proposal initialization until the cross-EP score
  protocol is implemented.
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
    """Unwrap common DeepSpeed/Megatron wrappers to the inner module."""
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
    nn.Module. We unwrap once using the same convention as the proposal pipeline.

    Returns ``([], {})`` if no gates are found — caller should warn and
    fall back to no-freeze.
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
def _deep_detach(obj: Any) -> Any:
    """Detach every tensor in a nested tuple/list/dict (no clone).

    Storage-sharing rationale: ``detach()`` returns a view sharing the
    original storage. As long as the returned tensor (held in
    ``self._captured``) keeps a reference, the storage refcount stays
    ≥ 1 and CUDA does NOT free the bytes — even after the original
    graph drops the source tensor.

    Cloning was previously used out of caution but cost +20 GB peak at
    exp_etc: each MoE gate's output includes a dense
    ``combine_weights`` of shape (tokens × experts × capacity) FP32
    (~2.7 GB / layer) and a bool ``dispatch_mask`` (~670 MB / layer).
    Six MoE layers ≈ +20 GB persistent during refresh. Detach-only
    reuses the original storage → ~0 extra alloc.

    Safety: the captured outputs are returned by patched gate forwards
    during ±ε replay. DeepSpeed MoE downstream (``dispatch_mask.type_as
    (input)``, ``einsum``) is read-only; no in-place mutation observed.
    """
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, tuple):
        return tuple(_deep_detach(o) for o in obj)
    if isinstance(obj, list):
        return [_deep_detach(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _deep_detach(v) for k, v in obj.items()}
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


def _zero_aux_loss_sparse(captured: Any) -> Any:
    """Sparse-aware variant of :func:`_zero_aux_loss`.

    Handles three captured shapes:
      * ``_SparseGateCapture`` (3-tuple gate output sparsified)
      * ``(_SparseGateCapture, exp_counts)`` (4-tuple variant)
      * Original 3- or 4-tuple (sparsification declined → fallback)
    """
    if isinstance(captured, _SparseGateCapture):
        l_aux = captured.l_aux
        if torch.is_tensor(l_aux) and l_aux.dim() == 0:
            captured.l_aux = torch.zeros_like(l_aux)
        return captured
    if (isinstance(captured, tuple) and len(captured) == 2
            and isinstance(captured[0], _SparseGateCapture)):
        sparse, trailing = captured
        l_aux = sparse.l_aux
        if torch.is_tensor(l_aux) and l_aux.dim() == 0:
            sparse.l_aux = torch.zeros_like(l_aux)
        return (sparse, trailing)
    # Fallback: defer to the original dense zeroer.
    return _zero_aux_loss(captured)


# ---------------------------------------------------------------------------
# Sparse capture / replay for MoE gate outputs
#
# Without sparsification, holding ALL MoE gate outputs across the K-sample
# loop costs ~20 GB persistent at exp_etc (6 MoE layers × ~3.4 GB dense
# combine_weights ~(tokens × experts × capacity) FP32 + ~0.7 GB bool
# dispatch_mask). In normal training each gate output is transient (freed
# after its MoE layer's forward returns), so routing-freeze inflates peak
# from ~42 GB to ~60 GB.
#
# Top-K routing (k≪num_experts·capacity) makes these tensors >99.99 %
# zero. We extract just the nonzero (token, expert, capacity, weight)
# tuples at capture time (~few hundred KB / layer), then materialise the
# dense form on-the-fly inside the replay closure. The dense replay
# tensor is transient — alive only during one MoE layer forward — so
# peak returns to training-level (~42 GB) with negligible time overhead.
# ---------------------------------------------------------------------------

# Tag class identifying a sparse-stored gate output. The replay closure
# detects this and rebuilds the dense (l_aux, combine, dispatch) tuple
# on each ±ε pass.
class _SparseGateCapture:
    __slots__ = (
        "l_aux", "shape", "nz_idx", "nz_weight",
        "weight_dtype", "weight_device", "dense_dtype",
    )

    def __init__(
        self,
        l_aux: torch.Tensor,
        shape: Tuple[int, ...],
        nz_idx: torch.Tensor,
        nz_weight: torch.Tensor,
        dense_dtype: torch.dtype,
    ) -> None:
        self.l_aux = l_aux                          # scalar (already zeroed)
        self.shape = shape                          # (S, E, C)
        self.nz_idx = nz_idx                        # (NNZ, 3) int64 — (s, e, c)
        self.nz_weight = nz_weight                  # (NNZ,) FP32 — gate weights
        self.weight_dtype = nz_weight.dtype
        self.weight_device = nz_weight.device
        self.dense_dtype = dense_dtype              # original combine_weights dtype


def _maybe_sparsify_gate_output(captured: Any) -> Any:
    """Detect a typical TopKGate output and convert combine/dispatch to
    sparse-coords form.  Recognised shapes:

      * 3-tuple ``(l_aux, combine_weights, dispatch_mask)``
      * 4-tuple ``(l_aux, combine_weights, dispatch_mask, exp_counts)``

    Falls through (returns input unchanged) on any other shape.
    Replay rebuilds the dense form via ``_replay_value(captured)``.
    """
    if not isinstance(captured, tuple):
        return captured
    if len(captured) not in (3, 4):
        return captured
    l_aux = captured[0]
    combine = captured[1]
    dispatch = captured[2]
    if not (torch.is_tensor(l_aux) and torch.is_tensor(combine)
            and torch.is_tensor(dispatch)):
        return captured
    if combine.dim() != 3 or combine.shape != dispatch.shape:
        return captured
    # Extract nonzero coordinates from combine (dispatch's True positions
    # match combine's nonzeros for top-K softmax routing).
    nz_idx = combine.nonzero(as_tuple=False)        # (NNZ, 3) int64
    if nz_idx.numel() == 0:
        # Fully zero gate output (shouldn't happen but degrade gracefully).
        return captured
    s_ix = nz_idx[:, 0]
    e_ix = nz_idx[:, 1]
    c_ix = nz_idx[:, 2]
    nz_w = combine[s_ix, e_ix, c_ix].contiguous()
    sparse = _SparseGateCapture(
        l_aux=l_aux,
        shape=tuple(combine.shape),
        nz_idx=nz_idx.contiguous(),
        nz_weight=nz_w,
        dense_dtype=combine.dtype,
    )
    if len(captured) == 4:
        return (sparse, captured[3])               # preserve exp_counts trailing
    return sparse


def _replay_value(captured: Any) -> Any:
    """Rebuild the dense gate-output tuple from a ``_SparseGateCapture``.

    Each call allocates fresh dense ``combine_weights`` (FP32) and
    ``dispatch_mask`` (bool) of the original shape, scatters the stored
    nonzero weights, and returns the original tuple shape. The dense
    tensors are transient — released as soon as the MoE layer's forward
    consumes them.
    """
    if isinstance(captured, tuple) and len(captured) == 2 and isinstance(
            captured[0], _SparseGateCapture):
        sparse: _SparseGateCapture = captured[0]
        exp_counts = captured[1]
        combine = torch.zeros(
            sparse.shape, dtype=sparse.dense_dtype, device=sparse.weight_device,
        )
        s_ix = sparse.nz_idx[:, 0]
        e_ix = sparse.nz_idx[:, 1]
        c_ix = sparse.nz_idx[:, 2]
        combine[s_ix, e_ix, c_ix] = sparse.nz_weight.to(sparse.dense_dtype)
        dispatch = combine.bool()
        return (sparse.l_aux, combine, dispatch, exp_counts)
    if isinstance(captured, _SparseGateCapture):
        sparse = captured
        combine = torch.zeros(
            sparse.shape, dtype=sparse.dense_dtype, device=sparse.weight_device,
        )
        s_ix = sparse.nz_idx[:, 0]
        e_ix = sparse.nz_idx[:, 1]
        c_ix = sparse.nz_idx[:, 2]
        combine[s_ix, e_ix, c_ix] = sparse.nz_weight.to(sparse.dense_dtype)
        dispatch = combine.bool()
        return (sparse.l_aux, combine, dispatch)
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
                # 1) detach (storage shared with original — no clone).
                # 2) sparsify TopKGate output if it matches the
                #    (l_aux, combine, dispatch[, exp_counts]) shape.
                #    Reduces ~3.4 GB / layer dense capture → ~few hundred
                #    KB sparse coords; replay rebuilds dense transiently.
                # 3) zero l_aux so the load-balance loss does not
                #    contribute to the FD numerator.
                detached = _deep_detach(output)
                sparse = _maybe_sparsify_gate_output(detached)
                self._captured[mid] = _zero_aux_loss_sparse(sparse)
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
            # Replay always passes through ``_replay_value`` so sparse
            # captures get materialised to the original dense tuple
            # shape on each ±ε pass. For non-sparse captures, the
            # function is a no-op pass-through — same behaviour as
            # before.
            def _replay(*_args, _v=captured_value, **_kwargs):
                return _replay_value(_v)
            module.forward = _replay  # type: ignore[assignment]


def identify_gate_modules_safe(
    model: Any,
) -> Tuple[List[Tuple[str, torch.nn.Module]], Dict[str, int]]:
    """Public-facing wrapper kept for callers that use the safe entry."""
    return identify_gate_modules(model)
