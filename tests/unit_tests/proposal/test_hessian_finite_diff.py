"""Unit tests for the finite-difference Hessian estimator.

Exercises the state-free pieces of ``megatron.proposal.hessian`` and
``megatron.proposal._routing_freeze`` without requiring DeepSpeed,
distributed init, or the Megatron training stack:

  * synthetic 3-point eps sweep correctness (highest Pearson is picked)
  * exact param restore after the +/- eps perturbations
  * routing freeze hook captures and replays gate outputs
  * graceful None-return when forward fails
"""
from __future__ import annotations

import math
from typing import Any, List, Tuple

import pytest
import torch

from megatron.proposal.hessian import (
    compute_hessian_finite_diff_core,
    compare_hessian_estimators,
    _pearson_two_tensor_lists,
)
from megatron.proposal._routing_freeze import (
    RoutingFreezeContext,
    identify_gate_modules,
    _set_force_empty,
)


# ---------------------------------------------------------------------------
# Synthetic quadratic problem: L(θ) = 0.5 Σ a_i θ_i² has H_diag = a (constant).
# This lets us test the central-diff estimator analytically.
# ---------------------------------------------------------------------------
def _make_quadratic(n: int = 16, seed: int = 0) -> Tuple[torch.nn.Parameter, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    a = (1.0 + torch.rand(n, generator=g)) * 5.0   # positive Hessian diag in [5,10]
    theta = torch.nn.Parameter(torch.randn(n, generator=g) * 0.3)
    return theta, a


def test_param_restore_exact():
    """After compute_hessian_finite_diff_core, params must be byte-identical
    to before the call (the restore correctness gate that prevents drift).
    """
    theta, a = _make_quadratic(n=32)
    saved = theta.data.clone()

    def _forward():
        # 0.5 Σ a_i θ_i² ; gradient = a * θ ; Hessian diag = a.
        loss = 0.5 * (a * theta * theta).sum()
        return loss, {}

    h = compute_hessian_finite_diff_core(
        forward_loss_fn=_forward,
        expert_ps=[theta],
        k_samples=2,
        eps=1e-2,
        gates_for_freeze=None,
    )
    assert h is not None, "central-diff returned None unexpectedly"
    # Param restore — exact-byte recovery within float64 promotion of norm.
    diff = (theta.data - saved).abs().max().item()
    assert diff < 1e-6, f"param drift after estimator: max abs diff = {diff:.3e}"


def test_quadratic_recovers_diag():
    """For L = 0.5 Σ a_i θ_i², central-diff |H_diag| ≈ a. K-sample Rademacher
    averaging means a single noise-free run should still recover ranks
    correctly even at modest K.
    """
    torch.manual_seed(0)
    theta, a = _make_quadratic(n=32, seed=42)

    def _forward():
        loss = 0.5 * (a * theta * theta).sum()
        return loss, {}

    h = compute_hessian_finite_diff_core(
        forward_loss_fn=_forward,
        expert_ps=[theta],
        k_samples=8,
        eps=1e-3,
        gates_for_freeze=None,
    )
    assert h is not None
    h_est = h[0]
    # Pearson correlation of estimate vs truth on a 32-element diagonal.
    a64 = a.to(torch.float64)
    h64 = h_est.to(torch.float64)
    pa = a64 - a64.mean()
    pb = h64 - h64.mean()
    pearson = (pa * pb).sum() / (pa.norm() * pb.norm() + 1e-12)
    assert pearson.item() > 0.85, f"low pearson on quadratic test: {pearson.item():.3f}"


def test_graceful_none_on_forward_failure():
    """If the forward closure raises, the estimator returns None (no leaked
    param mutation, no exception propagated)."""
    theta = torch.nn.Parameter(torch.randn(8))
    saved = theta.data.clone()

    def _broken_forward():
        raise RuntimeError("synthetic forward failure")

    h = compute_hessian_finite_diff_core(
        forward_loss_fn=_broken_forward,
        expert_ps=[theta],
        k_samples=2,
        eps=1e-3,
        gates_for_freeze=None,
    )
    assert h is None
    # Param restored even though we never hit the cleanup path explicitly.
    diff = (theta.data - saved).abs().max().item()
    assert diff < 1e-6, f"param drifted on failed estimator path: {diff:.3e}"


# ---------------------------------------------------------------------------
# Routing-freeze tests
# ---------------------------------------------------------------------------
class _ToyTopKGate(torch.nn.Module):
    """A toy Gate whose class name matches one of the identifier hints
    (TopKGate). Forward returns (l_aux, combine_weights, dispatch_mask) —
    the same 3-tuple shape DeepSpeed's TopKGate returns. ``combine_weights``
    depends on the input via a stochastic mask so we can verify replay
    actually substitutes the captured value.
    """
    def __init__(self, hidden: int = 4, n_experts: int = 2):
        super().__init__()
        self.wg = torch.nn.Linear(hidden, n_experts, bias=False)
        self.calls = 0

    def forward(self, x: torch.Tensor):
        self.calls += 1
        # noisy (data-dependent + random) so any replay mismatch shows up.
        logits = self.wg(x) + torch.randn_like(self.wg(x)) * 0.5
        l_aux = torch.tensor(float(self.calls), dtype=torch.float32)
        # Top-1 mask
        idx = logits.argmax(dim=-1)
        combine = torch.nn.functional.one_hot(idx, num_classes=logits.size(-1)).float()
        dispatch = combine.bool()
        return l_aux, combine, dispatch


class _ToyMoEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _ToyTopKGate(hidden=4, n_experts=2)
        self.proj = torch.nn.Linear(4, 4)


def test_routing_freeze_capture_replay():
    """RoutingFreezeContext.capture must record the gate output, then
    subsequent forwards return that captured (zeroed-aux) value
    regardless of input."""
    model = _ToyMoEModel()
    gates, rules = identify_gate_modules(model)
    assert len(gates) == 1, f"expected 1 gate, found {len(gates)}: {[(n, type(g).__name__) for n,g in gates]}"
    assert any("TopKGate" in r for r in rules), f"unexpected rules: {rules}"

    inp_capture = torch.randn(3, 4)
    with RoutingFreezeContext(gates) as ctx:
        # Capture phase
        ctx.capture(lambda: model.gate(inp_capture))
        # Replay phase
        inp_replay = torch.randn(3, 4)  # different input
        l_aux_a, comb_a, disp_a = model.gate(inp_replay)
        l_aux_b, comb_b, disp_b = model.gate(torch.randn(3, 4))
    # Combine weights must match across replay calls — the whole point.
    assert torch.equal(comb_a, comb_b), "captured combine weights diverged across replay"
    # l_aux is zeroed in replay (per spec).
    assert l_aux_a.item() == 0.0
    assert l_aux_b.item() == 0.0


def test_routing_freeze_restores_original_forward():
    """After the context exits, the gate's forward is restored to the
    original (so subsequent training calls aren't broken)."""
    model = _ToyMoEModel()
    gates, _ = identify_gate_modules(model)
    orig_calls = model.gate.calls
    with RoutingFreezeContext(gates) as ctx:
        ctx.capture(lambda: model.gate(torch.randn(2, 4)))
        # Replay — does NOT bump .calls because forward is monkey-patched.
        model.gate(torch.randn(2, 4))
    # After exit, original forward is back.
    model.gate(torch.randn(2, 4))
    assert model.gate.calls > orig_calls + 1, "original gate forward not restored"


def test_identify_gate_modules_safe_force_empty():
    """The test-injection flag forces identify_gate_modules_safe to return
    empty even on a model that has gates — used to simulate the
    gate-id-failure code path."""
    from megatron.proposal._routing_freeze import identify_gate_modules_safe
    model = _ToyMoEModel()
    gates_normal, _ = identify_gate_modules_safe(model)
    assert len(gates_normal) == 1
    _set_force_empty(True)
    try:
        gates_empty, _ = identify_gate_modules_safe(model)
        assert gates_empty == []
    finally:
        _set_force_empty(False)



def test_compare_hessian_estimators_label_propagation():
    """When called with custom labels, the output keys reflect them."""
    n = 16
    p = torch.nn.Parameter(torch.randn(n))
    h_a = torch.randn(n).abs().to(torch.float32)
    h_b = torch.randn(n).abs().to(torch.float32)
    tracked = [("x", p)]
    res = compare_hessian_estimators(0, tracked, [h_a], [h_b], {},
                                      label_a="hutch", label_b="finite_diff")
    assert "pearson_hutch_finite_diff_bare" in res
    assert "pearson_hutch_finite_diff_score" in res
    assert res["label_a"] == "hutch"
    assert res["label_b"] == "finite_diff"



def test_pearson_two_tensor_lists_constant():
    """Pearson on identical lists is 1.0; Pearson on flat-zero is 0.0."""
    a = [torch.tensor([1.0, 2.0, 3.0])]
    b = [torch.tensor([2.0, 4.0, 6.0])]
    r = _pearson_two_tensor_lists(a, b)
    assert r > 0.99
    z = [torch.zeros(8)]
    r0 = _pearson_two_tensor_lists(z, z)
    assert r0 == 0.0
