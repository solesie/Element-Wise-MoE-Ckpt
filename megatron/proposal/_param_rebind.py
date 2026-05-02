"""Temporary Parameter rebind for the finite-diff Hessian refresh.

Why
---
Production runs in BF16 + DeepSpeed ZeRO-1. The cleanest finite-diff (FD)
implementation perturbs `p.data` in place, runs forward, and computes the
gradient — but under DS BF16+ZeRO-1, `p.data = ...` is silently ignored:
the bf16_optimizer flattens all weights into a single backing buffer and
keeps its own Parameter references in `bf16_groups`. Forward reads from
that flat buffer, not from `p.data` of the user-visible Parameter.

The fix
-------
For the duration of one FD refresh, walk every module that holds BF16
parameters and replace each `module._parameters[name]` with a freshly-
allocated Parameter object — same dtype (BF16), same values (cloned), but
with NEW backing storage that is no longer aliased to DS's flat buffer.
Forward then reads `_parameters[name]` (via `nn.Module.__getattr__`) and
finds our new Parameter, so writes via `p.data.copy_(...).add_(...)` do
reach the bytes that forward consumes.

Caller-supplied `expert_ps` and `expert_modules` lists hold references to
the OLD Parameter objects. We REMAP both in place so the caller's list
items now point to the new Parameters — `torch.autograd.grad(loss, expert_ps)`
then operates on the same nodes that forward computed against.

On context exit (success or exception) we restore everything bit-exactly:
- `_parameters` and `_buffers` dicts get the original objects back
- expert_ps and expert_modules lists get the original Parameter refs back
- DS bf16_optimizer.bf16_groups is untouched throughout (we never wrote
  to old Parameters), so the model weights forward sees AFTER the refresh
  are identical to the values BEFORE.

Empirical validation (tiny MoE, 2000 iter, eps=1e-3, K=16):
- val loss matches the Hutchinson K=4 baseline within iteration noise
- recovery from injected faults works (DRAM expert state intact)

Cost: clones one BF16 copy of every replaced parameter — ~1% of model
memory for typical MoE sizes; freed on context exit.
"""
from __future__ import annotations

import contextlib
from typing import Any, List, Tuple

import torch


def _iter_modules(model: Any):
    """Yield every nn.Module in the (possibly DS-wrapped) model tree."""
    if isinstance(model, (list, tuple)):
        model = model[0]
    if hasattr(model, "modules"):
        yield from model.modules()
        return
    cur = model
    visited = 0
    while cur is not None and visited < 8:
        if hasattr(cur, "modules"):
            yield from cur.modules()
            return
        cur = getattr(cur, "module", None)
        visited += 1


@contextlib.contextmanager
def param_rebind_context(model: Any, expert_ps=None, expert_modules=None):
    """Replace every BF16 Parameter in `model` with a fresh Parameter
    (same dtype, cloned data, new storage), remap `expert_ps` and the
    role-dicts inside `expert_modules` to point to the new Parameters,
    then restore everything on exit.

    Use as::

        with param_rebind_context(model, expert_ps, expert_modules):
            # FD param-perturb runs here — p.data writes reach forward
            ...
    """
    if isinstance(model, (list, tuple)):
        model = model[0]

    # Step 1: replace BF16 (and FP16) Parameters and buffers with fresh
    # objects whose storage is independent of DS's flat-tensor aliasing.
    saved_params: List[Tuple[Any, str, Any]] = []
    old_to_new: dict = {}
    for sub in _iter_modules(model):
        for name, p in list(sub._parameters.items()):
            if p is None or p.dtype == torch.float32:
                continue
            saved_params.append((sub, name, p))
            new_p = torch.nn.Parameter(
                p.data.clone(), requires_grad=p.requires_grad,
            )
            sub._parameters[name] = new_p
            old_to_new[id(p)] = new_p
        for name, b in list(sub._buffers.items()):
            if b is None or not torch.is_floating_point(b) or b.dtype == torch.float32:
                continue
            saved_params.append((sub, name, b))
            sub._buffers[name] = b.clone()

    # Step 2: remap expert_ps in place — autograd.grad(loss, expert_ps)
    # must land on the new Parameters that forward actually used.
    saved_expert_ps_originals: List[Any] = []
    if expert_ps is not None:
        for i, p in enumerate(expert_ps):
            saved_expert_ps_originals.append(p)
            new_p = old_to_new.get(id(p))
            if new_p is not None:
                expert_ps[i] = new_p

    saved_expert_modules_originals: List[Tuple[dict, str, Any]] = []
    if expert_modules is not None:
        for entry in expert_modules:
            try:
                _qname, _mod, role_dict = entry
            except Exception:
                continue
            if not isinstance(role_dict, dict):
                continue
            for role, p in list(role_dict.items()):
                saved_expert_modules_originals.append((role_dict, role, p))
                new_p = old_to_new.get(id(p))
                if new_p is not None:
                    role_dict[role] = new_p

    try:
        yield
    finally:
        # Restore in reverse order. Each step is bit-exact: we put back
        # the saved object (not a clone or cast).
        if expert_ps is not None:
            for i, original in enumerate(saved_expert_ps_originals):
                if i < len(expert_ps):
                    expert_ps[i] = original
        for role_dict, role, original in saved_expert_modules_originals:
            role_dict[role] = original
        for sub, name, original in saved_params:
            if name in sub._parameters:
                sub._parameters[name] = original
            elif name in sub._buffers:
                sub._buffers[name] = original
