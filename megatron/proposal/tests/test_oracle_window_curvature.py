from __future__ import annotations

import copy

import pytest
import torch

from megatron.proposal.refresh.quantile import make_quantile_refs
from megatron.proposal.telemetry.oracle_window import curvature
from megatron.proposal.telemetry.oracle_window.curvature import (
    collect_nonexpert_perturb_parameters,
    freeze_expert_lut,
    measure_nonexpert_curvature,
    select_nonexpert_measurement_parameters,
)
from megatron.proposal.telemetry.oracle_window.lut import (
    DomainLut,
    DomainLutMetadata,
    FiniteDifferenceAudit,
)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expert = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.router = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.dense = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.register_buffer("running", torch.tensor([7.0]))


def _metadata(domain: str, refs, fd: FiniteDifferenceAudit) -> DomainLutMetadata:
    return DomainLutMetadata(
        domain=domain,
        local_population=8,
        global_population=8,
        global_h_positive=8,
        global_v_positive=8,
        h_min=1.0,
        h_max=2.0,
        v_min=1.0,
        v_max=2.0,
        n_quantiles=int(refs.v.values.numel()),
        n_histogram_bins=8,
        n_lut_bins=int(refs.h_lut.values.numel()),
        h_values_are_absolute=True,
        fd=fd,
    )


def test_nonexpert_perturb_ownership_includes_router_and_is_canonical() -> None:
    first = _TinyModel()
    second = copy.deepcopy(first)
    kwargs = dict(
        expert_parameter_names=("expert.weight",),
        pp_rank=3,
        router_parameter_names=("router.weight",),
    )
    params0, routers0, offsets0 = collect_nonexpert_perturb_parameters(
        model=first, **kwargs,
    )
    params1, routers1, offsets1 = collect_nonexpert_perturb_parameters(
        model=second, **kwargs,
    )

    assert [name for name, _param in params0] == ["dense.weight", "router.weight"]
    assert routers0 == routers1 == ("router.weight",)
    assert offsets0 == offsets1 == (0, 4)


def test_nonexpert_perturb_ownership_fails_if_router_is_excluded() -> None:
    model = _TinyModel()
    with pytest.raises(RuntimeError, match="router parameters are absent"):
        collect_nonexpert_perturb_parameters(
            model=model,
            expert_parameter_names=("expert.weight", "router.weight"),
            pp_rank=0,
            router_parameter_names=("router.weight",),
        )


def test_frozen_expert_lut_does_not_alias_source() -> None:
    refs = make_quantile_refs(
        device=torch.device("cpu"), n_q=8, h_lookup_lut_bins=16,
    )
    refs.v.values.copy_(torch.linspace(1.0, 2.0, 8))
    refs.h.values.copy_(torch.linspace(3.0, 4.0, 8))
    refs.h_lut.values.copy_(torch.linspace(5.0, 6.0, 16))
    audit = FiniteDifferenceAudit(
        parameter_dtype="bfloat16",
        epsilon=0.01,
        compared_elements=8,
        plus_changed_elements=8,
        minus_changed_elements=8,
    )
    source = DomainLut(
        domain="expert",
        refs=refs,
        metadata=_metadata("expert", refs, audit),
    )

    frozen = freeze_expert_lut(source)
    refs.h_lut.values.add_(100.0)

    assert frozen.refs.h_lut.values.data_ptr() != refs.h_lut.values.data_ptr()
    torch.testing.assert_close(
        frozen.refs.h_lut.values,
        torch.linspace(5.0, 6.0, 16),
    )


def test_measurement_contract_requires_k4_and_dedicated_iterator() -> None:
    with pytest.raises(ValueError, match="K >= 4"):
        measure_nonexpert_curvature(
            model=None,
            forward_step_func=lambda *_args: None,
            data_iterator=None,
            mapper=None,
            inner_optimizer=None,
            nonexpert_lut_builder=None,
            expert_parameter_names=(),
            pp_rank=0,
            hutchinson_probes=3,
            dedicated_data_iterator=True,
        )
    with pytest.raises(ValueError, match="dedicated data iterator"):
        measure_nonexpert_curvature(
            model=None,
            forward_step_func=lambda *_args: None,
            data_iterator=None,
            mapper=None,
            inner_optimizer=None,
            nonexpert_lut_builder=None,
            expert_parameter_names=(),
            pp_rank=0,
            hutchinson_probes=4,
            dedicated_data_iterator=False,
        )


class _FakeMapper:
    mode = "bf16_zero0"

    def __init__(self, named_parameters):
        self.named = dict(named_parameters)
        self.order = list(self.named)
        self.p = torch.cat([
            parameter.detach().float().reshape(-1)
            for parameter in self.named.values()
        ])
        self.m = torch.ones_like(self.p)
        self.v = torch.full_like(self.p, 2.0)
        self.h = {
            name: torch.zeros(parameter.numel(), dtype=torch.float32)
            for name, parameter in self.named.items()
        }
        self.offset = {}
        cursor = 0
        for name, parameter in self.named.items():
            self.offset[name] = cursor
            cursor += parameter.numel()

    def zero12_local_fragments(self, named_parameters):
        out = []
        for name, parameter in named_parameters:
            if name not in self.named:
                continue
            out.append((name, parameter, 0, self.offset[name], 0, parameter.numel()))
        return out

    def zero12_partition_view(self, group_idx, local_offset, numel, *, state_key=None):
        source = {None: self.p, "exp_avg": self.m, "exp_avg_sq": self.v}[state_key]
        return source.narrow(0, int(local_offset), int(numel))

    def get_grad_accum_view(self, parameter, *, allow_create=False):
        del allow_create
        for name, candidate in self.named.items():
            if candidate is parameter:
                return self.h[name]
        raise KeyError("unknown parameter")


def test_replicated_measurement_ownership_shards_names_but_not_perturb_set() -> None:
    model = _TinyModel()
    perturb = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name != "expert.weight"
    )
    mapper = _FakeMapper(perturb)
    shard0, mode0 = select_nonexpert_measurement_parameters(
        mapper=mapper,
        perturb_parameters=perturb,
        owner_rank=0,
        owner_count=2,
    )
    shard1, mode1 = select_nonexpert_measurement_parameters(
        mapper=mapper,
        perturb_parameters=perturb,
        owner_rank=1,
        owner_count=2,
    )
    names0 = {name for name, _parameter in shard0}
    names1 = {name for name, _parameter in shard1}
    assert mode0 == mode1 == "replicated_name_shard"
    assert not names0 & names1
    assert names0 | names1 == {name for name, _parameter in perturb}


class _FakeOptimizer:
    def __init__(self) -> None:
        self.micro_steps = 11
        self.global_steps = 22


class _FakeEngine:
    def __init__(self, module, optimizer) -> None:
        self.module = module
        self.optimizer = optimizer
        self.micro_steps = 33


class _FakeLutBuilder:
    def build(self, *, h_views, v_views, fd_audit, **_kwargs):
        assert all(torch.count_nonzero(view).item() == view.numel() for view in h_views)
        assert sum(view.numel() for view in h_views) == sum(
            view.numel() for view in v_views
        )
        refs = make_quantile_refs(
            device=torch.device("cpu"), n_q=8, h_lookup_lut_bins=16,
        )
        return DomainLut(
            domain="nonexpert",
            refs=refs,
            metadata=_metadata("nonexpert", refs, fd_audit),
        )


def test_nonexpert_fd_restores_lp_rng_buffers_grads_and_ds_attrs(monkeypatch) -> None:
    model = _TinyModel()
    optimizer = _FakeOptimizer()
    engine = _FakeEngine(model, optimizer)
    perturb_names = ("dense.weight", "router.weight")
    mapper = _FakeMapper((name, dict(model.named_parameters())[name]) for name in perturb_names)
    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    original_buffer = model.running.clone()
    original_grad = torch.full_like(model.dense.weight, 0.25)
    model.dense.weight.grad = original_grad.clone()
    torch.manual_seed(123)
    original_rng = torch.get_rng_state().clone()

    def fake_snapshot(_optimizer, model=None):
        module = model.module
        return {
            "rng": torch.get_rng_state().clone(),
            "buffer": module.running.clone(),
            "grads": {
                name: None if p.grad is None else p.grad.clone()
                for name, p in module.named_parameters()
            },
        }

    def fake_restore(_optimizer, snapshot):
        torch.set_rng_state(snapshot["rng"])
        model.running.copy_(snapshot["buffer"])
        for name, parameter in model.named_parameters():
            saved = snapshot["grads"][name]
            parameter.grad = None if saved is None else saved.clone()

    monkeypatch.setattr(curvature, "snapshot_training_state", fake_snapshot)
    monkeypatch.setattr(curvature, "restore_training_state", fake_restore)
    monkeypatch.setattr(
        curvature,
        "pinned_snapshot",
        lambda tensor, key: tensor.detach().clone(),
    )
    monkeypatch.setattr(curvature, "_validate_finite_signal", lambda views: None)

    def fake_fd_runner(
        _engine,
        _forward,
        _iterator,
        parameters,
        probes,
        **kwargs,
    ):
        for parameter in parameters:
            parameter.data.add_(1.0)
            mapper.get_grad_accum_view(parameter).add_(float(probes))
        model.running.add_(5.0)
        model.dense.weight.grad = torch.ones_like(model.dense.weight)
        engine.micro_steps = -1
        optimizer.micro_steps = -2
        optimizer.global_steps = -3
        torch.manual_seed(999)
        per_sign = sum(parameter.numel() for parameter in parameters) * probes
        kwargs["perturb_changed_counts"][0].add_(per_sign)
        kwargs["perturb_changed_counts"][1].add_(per_sign)
        return []

    result = measure_nonexpert_curvature(
        model=engine,
        forward_step_func=lambda *_args: None,
        data_iterator=iter((object(),)),
        mapper=mapper,
        inner_optimizer=optimizer,
        nonexpert_lut_builder=_FakeLutBuilder(),
        expert_parameter_names=("expert.weight",),
        pp_rank=0,
        router_parameter_names=("router.weight",),
        epsilon=0.01,
        hutchinson_probes=4,
        data_batches=1,
        dedicated_data_iterator=True,
        fd_runner=fake_fd_runner,
    )

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original_parameters[name])
    torch.testing.assert_close(model.running, original_buffer)
    torch.testing.assert_close(model.dense.weight.grad, original_grad)
    assert engine.micro_steps == 33
    assert optimizer.micro_steps == 11
    assert optimizer.global_steps == 22
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert result.metadata.routing_freeze is False
    assert result.metadata.hutchinson_probes == 4
    assert result.metadata.router_parameter_names == ("router.weight",)
    assert result.metadata.fd.plus_changed_fraction == 1.0
    assert result.metadata.fd.minus_changed_fraction == 1.0
