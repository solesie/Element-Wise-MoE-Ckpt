"""Unit tests for the H-quantile anchor finite-difference helpers.

Exercises the state-free pieces of ``megatron.proposal.refresh.hessian_refresh`` and
``megatron.proposal.refresh.routing_freeze`` without requiring DeepSpeed,
distributed init, or the Megatron training stack:

  * the FD core requires theta_snap_ref + grad_offsets + grad_accum_views
  * CUDA/Triton FP32 grad accumulation matches a quadratic Hessian diagonal
  * routing freeze hook captures and replays gate outputs
"""
from __future__ import annotations

from typing import Tuple

import pytest
import torch

from megatron.proposal.refresh.hessian_refresh import (
    _HAS_TRITON_HESSIAN,
    compute_hessian_finite_diff_core,
)
from megatron.proposal.refresh.routing_freeze import (
    RoutingFreezeContext,
    identify_gate_modules,
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


def test_grad_flat_inputs_are_required():
    theta, a = _make_quadratic(n=32)
    saved = theta.data.clone()

    def _forward():
        # 0.5 Σ a_i θ_i² ; gradient = a * θ ; Hessian diag = a.
        loss = 0.5 * (a * theta * theta).sum()
        return loss, {}

    # Missing theta_snap_ref / grad_offsets / grad_accum_views → the core
    # refuses (no partial run) and leaves θ untouched.
    with pytest.raises(RuntimeError):
        compute_hessian_finite_diff_core(
            forward_loss_fn=_forward,
            expert_ps=[theta],
            k_samples=1,
            eps=1e-2,
            gates_for_freeze=None,
        )
    assert torch.equal(theta.data, saved)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_TRITON_HESSIAN,
    reason="grad-flat anchor path requires CUDA and Triton",
)
def test_grad_flat_quadratic_signal_matches_diagonal():
    torch.manual_seed(123)
    n = 64
    a = torch.linspace(1.0, 4.0, n, device="cuda", dtype=torch.float32)
    theta = torch.nn.Parameter(
        (torch.randn(n, device="cuda", dtype=torch.float32) * 0.1).to(torch.bfloat16)
    )
    theta_snap = [theta.data.detach().clone()]
    # Live FD path accumulates z·(g₊−g₋)/(2εK) into an FP32 per-expert view.
    grad_accum = [torch.zeros(n, device="cuda", dtype=torch.float32)]

    def _forward():
        loss = 0.5 * (a * theta.float() * theta.float()).sum()
        return loss, {}

    ret = compute_hessian_finite_diff_core(
        forward_loss_fn=_forward,
        expert_ps=[theta],
        k_samples=1,
        eps=1e-2,
        gates_for_freeze=None,
        theta_snap_ref=theta_snap,
        grad_offsets=[0],
        grad_accum_views=grad_accum,
    )
    torch.cuda.synchronize()
    try:
        assert ret == []
        h64 = grad_accum[0].abs().to(torch.float64)
    finally:
        theta.data.copy_(theta_snap[0])

    a64 = a.to(torch.float64)
    pa = a64 - a64.mean()
    pb = h64 - h64.mean()
    pearson = (pa * pb).sum() / (pa.norm() * pb.norm() + 1e-12)
    assert pearson.item() > 0.8


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_TRITON_HESSIAN,
    reason="grad-flat anchor path requires CUDA and Triton",
)
def test_grad_flat_forward_failure_returns_none_and_caller_restores():
    theta = torch.nn.Parameter(torch.randn(8, device="cuda", dtype=torch.bfloat16))
    snap = [theta.data.detach().clone()]
    grad_accum = [torch.zeros(theta.numel(), device="cuda", dtype=torch.float32)]

    def _broken_forward():
        raise RuntimeError("synthetic forward failure")

    h = compute_hessian_finite_diff_core(
        forward_loss_fn=_broken_forward,
        expert_ps=[theta],
        k_samples=2,
        eps=1e-3,
        gates_for_freeze=None,
        theta_snap_ref=snap,
        grad_offsets=[0],
        grad_accum_views=grad_accum,
    )
    assert h is None
    theta.data.copy_(snap[0])
    assert torch.equal(theta.data, snap[0])


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
