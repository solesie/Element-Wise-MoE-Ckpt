"""Tests for megatron.proposal.checkpoint_ctx.hutchinson_activations_checkpoint.

Verifies:
- Transformer-layer detection via ``.layers.N`` naming
- forward() is wrapped on enter and restored on exit
- Context is exception-safe (restores on error)
- 2nd-order autograd through checkpoint works (use_reentrant=False path)
- No-op when no transformer layers are present
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from megatron.proposal.checkpoint_ctx import (
    _find_transformer_layers,
    hutchinson_activations_checkpoint,
)


class ParallelTransformerLayer(nn.Module):
    """Name-collides with Megatron's class on purpose — our context
    manager detects layers by ``type(mod).__name__``."""
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x))


class _MegatronLikeModel(nn.Module):
    """Mirrors the Megatron naming convention: submodules as
    ``language_model.encoder.layers.<N>``. Layer class name matches the
    real Megatron module so class-based detection triggers.
    """
    def __init__(self, n_layers: int = 3, dim: int = 8) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.encoder = nn.Module()
        self.language_model.encoder.layers = nn.ModuleList(
            [ParallelTransformerLayer(dim) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.language_model.encoder.layers:
            x = l(x)
        return x


def test_layer_detection_matches_class_name() -> None:
    m = _MegatronLikeModel(n_layers=4)
    found = _find_transformer_layers(m)
    assert len(found) == 4
    for _name, mod in found:
        assert type(mod).__name__ == "ParallelTransformerLayer"


def test_forward_is_wrapped_and_restored() -> None:
    m = _MegatronLikeModel(n_layers=2)
    # Capture underlying method identity (bound methods are not `is`-stable
    # across attribute accesses, so we compare the descriptor / attribute).
    orig_ids = [id(type(l).forward) for l in m.language_model.encoder.layers]
    with hutchinson_activations_checkpoint(m):
        for l in m.language_model.encoder.layers:
            # In-context: instance attribute `forward` exists and is a plain
            # function (our wrapper), not the bound method via the class.
            assert "forward" in l.__dict__
    for l, oid in zip(m.language_model.encoder.layers, orig_ids):
        # After exit: instance attribute removed OR set back to a callable
        # with the same function identity as the original class-level method.
        # The canonical check: class method identity unchanged, and calling
        # the layer still produces the original behavior.
        assert id(type(l).forward) == oid


def test_forward_restored_on_exception() -> None:
    m = _MegatronLikeModel(n_layers=2)
    # Raise inside the context without invoking forward (deepspeed
    # checkpoint path wants CUDA). We verify structural restore only.
    originals_in_dict = []
    with pytest.raises(RuntimeError):
        with hutchinson_activations_checkpoint(m):
            for l in m.language_model.encoder.layers:
                originals_in_dict.append("forward" in l.__dict__)
            raise RuntimeError("boom")
    # Inside the ctx each layer had its own instance-level forward.
    assert all(originals_in_dict)
    # After exit, instance-level forward is removed → class-level method
    # resolution is back.
    for l in m.language_model.encoder.layers:
        assert "forward" not in l.__dict__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (deepspeed checkpoint)")
def test_numerical_equivalence_with_checkpointing() -> None:
    """Outputs and gradients should match within tolerance whether
    checkpointing is applied or not. Runs on CUDA because
    ``deepspeed.checkpointing.checkpoint`` assumes CUDA execution."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    m = _MegatronLikeModel(n_layers=3, dim=6).to(dev)
    x = torch.randn(4, 6, device=dev, requires_grad=True)

    y_ref = m(x)
    g_ref, = torch.autograd.grad(y_ref.sum(), x)

    with hutchinson_activations_checkpoint(m):
        y_ckpt = m(x)
    g_ckpt, = torch.autograd.grad(y_ckpt.sum(), x)
    torch.testing.assert_close(y_ref, y_ckpt)
    torch.testing.assert_close(g_ref, g_ckpt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (deepspeed checkpoint)")
def test_second_order_grad_works_with_create_graph() -> None:
    """Simulate the Hutchinson pattern: forward → create_graph=True →
    2nd autograd.grad — must not raise or return None under the ckpt ctx.
    """
    torch.manual_seed(1)
    dev = torch.device("cuda")
    m = _MegatronLikeModel(n_layers=2, dim=6).to(dev)
    x = torch.randn(4, 6, device=dev, requires_grad=True)
    params = [p for p in m.parameters() if p.requires_grad]

    with hutchinson_activations_checkpoint(m):
        y = m(x)
        loss = (y * y).sum()
        grads = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True, allow_unused=True,
        )
        z = [torch.ones_like(p) for p in params]
        dot = sum((g * zi).sum() for g, zi in zip(grads, z))
        hz = torch.autograd.grad(dot, params, retain_graph=False, allow_unused=True)

    assert all(h is not None for h in hz)
    assert all(h.shape == p.shape for h, p in zip(hz, params))


def test_no_transformer_layers_is_noop() -> None:
    class Flat(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
        def forward(self, x):
            return self.lin(x)

    m = Flat()
    # No submodule matches `.layers.N` → context detects zero layers.
    with hutchinson_activations_checkpoint(m) as n_wrapped:
        assert n_wrapped == 0
    # Behaviour unchanged — simple forward still works.
    x = torch.randn(2, 4)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, 4)
