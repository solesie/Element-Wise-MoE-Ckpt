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
For the duration of one FD refresh, replace each
``module._parameters[name]`` for the params we will perturb with a
freshly-allocated Parameter object — same dtype (BF16), same values
(cloned), but with NEW backing storage independent of DS's flat buffer.
Forward then reads ``_parameters[name]`` (via ``nn.Module.__getattr__``)
and finds our new Parameter, so writes via ``p.data.copy_(...).add_(...)``
do reach the bytes that forward consumes.

The caller-supplied ``expert_ps`` list is REMAPPED in place so each entry
points to the new Parameter — ``torch.autograd.grad(loss, expert_ps)``
then operates on the same nodes that forward computed against.

Scope
-----
Only params in ``expert_ps`` are rebinded. All other model params keep
their DS flat-buffer aliasing untouched. This keeps the transient memory
cost ~|expert_params| (e.g. ~1B BF16 = 2 GB on a 1T total-param model)
instead of cloning the entire model. Forward correctness is preserved
because we only WRITE to expert_ps; everything else flows through DS's
normal storage.

On context exit (success or exception) we restore everything bit-exactly:
- ``_parameters`` dicts get the original Parameter objects back
- ``expert_ps`` list gets the original Parameter refs back
- DS bf16_optimizer.bf16_groups is untouched throughout (we never wrote
  to old Parameters), so the model weights forward sees AFTER the refresh
  are identical to the values BEFORE.

Empirical validation (tiny MoE 200 iter + production exp_etc 1000 iter,
eps=1e-3, K=16): val loss matches the Hutchinson K=4 baseline within
iteration noise; recovery from injected faults works (DRAM expert state
intact).

EP > 1 caveats
--------------
The actual EP-specific risks are NOT cross-rank gradient noise (see
"non-issue" note below) but:

1. **Routing freeze across ranks** (single-rank tested only). All-to-all
   dispatch must produce the SAME routing on +ε and -ε forwards.
   ``RoutingFreezeContext`` captures gate output once at θ and replays
   on the perturbed forwards — verify it works under cross-rank dispatch
   before declaring the EP path safe.

2. **DS expert grad reduce hooks**. DeepSpeed registers grad
   accumulation hooks on the original Parameter objects; verify these
   don't fire on our ``torch.autograd.grad`` call (which they shouldn't,
   since we don't trigger ``loss.backward()``, but worth confirming).

3. **All-to-all latency**. Each refresh costs 2K extra all-to-alls
   (one per ±ε forward). Multiplies refresh time by EP communication
   overhead — usually small but worth measuring.

Non-issue: cross-rank gradient variance
---------------------------------------
A naive concern is that under EP=N every rank perturbs its local
expert independently, and rank 0's gradient ``g_+ - g_-`` picks up the
other ranks' perturbations as noise. Closer analysis: under
``RoutingFreezeContext`` each token is dispatched to a fixed expert,
so within a single MoE layer the cross-expert Hessian
``H_{i,j} = ∂²L/∂W_i ∂W_j = 0`` for i ≠ j (different token sets, no
interaction). Across MoE layers cross-Hessians are non-zero (a token
visits experts at multiple layers), but that variance source is intrinsic
to multi-layer MoE and exists at EP=1 too — EP just redistributes the
computation, not the noise. So the per-element FD variance is the same
function of (#layers × #experts-per-layer), independent of EP topology.
"""
from __future__ import annotations

import contextlib
from typing import Any, List, Tuple

import torch


@contextlib.contextmanager
def param_rebind_context(model: Any, expert_ps: List[torch.nn.Parameter]):
    """Replace the Parameter objects in ``expert_ps`` (and their host
    modules) with fresh Parameters that have independent storage. On
    exit, restore the originals bit-exactly.

    Use as::

        with param_rebind_context(model, expert_ps):
            # FD param-perturb runs here — p.data writes reach forward
            ...
    """
    if not expert_ps:
        yield
        return

    # Locate (host_module, attr_name) for each expert Parameter via id() match.
    targets: List[Tuple[Any, str, torch.nn.Parameter]] = []
    expert_ids = {id(p) for p in expert_ps}
    inner = model[0] if isinstance(model, (list, tuple)) else model
    for sub in inner.modules():
        for name, p in list(sub._parameters.items()):
            if p is not None and id(p) in expert_ids:
                targets.append((sub, name, p))

    # Step 1: replace each target with a fresh Parameter (independent storage).
    saved: List[Tuple[Any, str, torch.nn.Parameter]] = []
    old_to_new: dict = {}
    for sub, name, old_p in targets:
        new_p = torch.nn.Parameter(
            old_p.data.clone(), requires_grad=old_p.requires_grad,
        )
        sub._parameters[name] = new_p
        old_to_new[id(old_p)] = new_p
        saved.append((sub, name, old_p))

    # Step 2: remap expert_ps in place → autograd.grad lands on the new graph nodes.
    saved_expert_ps: List[torch.nn.Parameter] = list(expert_ps)
    for i, p in enumerate(expert_ps):
        new_p = old_to_new.get(id(p))
        if new_p is not None:
            expert_ps[i] = new_p

    try:
        yield
    finally:
        # Restore expert_ps list contents.
        for i, original in enumerate(saved_expert_ps):
            if i < len(expert_ps):
                expert_ps[i] = original
        # Restore module._parameters entries (bit-exact: same Parameter object).
        for sub, name, original in saved:
            sub._parameters[name] = original
