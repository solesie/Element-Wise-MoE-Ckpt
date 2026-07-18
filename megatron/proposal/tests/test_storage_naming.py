from pathlib import Path

import pytest
import torch

from megatron.proposal.expert_id import build_expert_id_map, ids_from_name
from megatron.proposal.distributed import ProposalTopology
from megatron.proposal.storage.recovery import _classify_params, _scan_cp_dir


def test_nonexpert_dir_is_pp_scoped_not_ep_scoped():
    topo = ProposalTopology(
        global_rank=3,
        world_size=4,
        dp_rank=0,
        dp_size=1,
        pp_rank=1,
        pp_size=2,
        ep_rank=1,
        ep_size=2,
        tp_rank=0,
        tp_size=1,
    )

    assert topo.nonexpert_dir(Path("/cp")) == Path("/cp/dp0/pp1/non_expert")
    assert topo.expert_dir(Path("/cp")) == Path("/cp/dp0/pp1/ep1/expert")


def test_dp_topology_uses_dp0_expert_and_dp_ep_nonexpert_shards():
    topo = ProposalTopology(
        global_rank=7,
        world_size=8,
        dp_rank=1,
        dp_size=2,
        pp_rank=1,
        pp_size=2,
        ep_rank=1,
        ep_size=2,
        tp_rank=0,
        tp_size=1,
    )

    assert topo.writes_expert is True
    assert topo.writes_nonexpert is True
    assert topo.selection_ranks == list(range(8))
    assert topo.nonexpert_shard_rank == 3
    assert topo.nonexpert_shard_count == 4
    assert topo.nonexpert_dir(Path("/cp")) == Path("/cp/pp1/non_expert")
    assert topo.expert_dir(Path("/cp")) == Path("/cp/dp1/pp1/ep1/expert")


def test_recovery_scans_new_and_legacy_nonexpert_shard_names(tmp_path):
    ne_dir = tmp_path / "dp0" / "pp0" / "non_expert"
    ex_dir = tmp_path / "dp0" / "pp0" / "ep0" / "expert"
    ne_dir.mkdir(parents=True)
    ex_dir.mkdir(parents=True)
    (ne_dir / "fc_nonexpert_shard00of04_iter00000007.bin").touch()
    (ne_dir / "fc_nonexpert_ep01of04_iter00000009.bin").touch()

    nonexpert_fcs, expert_pairs = _scan_cp_dir(
        tmp_path,
        nonexpert_dir=ne_dir,
        expert_dir=ex_dir,
    )

    assert sorted(nonexpert_fcs) == [
        (7, ne_dir / "fc_nonexpert_shard00of04_iter00000007.bin"),
        (9, ne_dir / "fc_nonexpert_ep01of04_iter00000009.bin"),
    ]
    assert expert_pairs == {}


class _FakeExpertLayer(torch.nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = int(layer_number)
        self.deepspeed_experts = torch.nn.ModuleList([
            torch.nn.Linear(1, 1, bias=False),
        ])


def _fake_pipe(*, wrapped=False, bad_layer=None):
    pipe = torch.nn.Module()
    pipe.add_module("2", _FakeExpertLayer(0))
    pipe.add_module("5", _FakeExpertLayer(3 if bad_layer is None else bad_layer))
    if not wrapped:
        return pipe
    wrapper = torch.nn.Module()
    wrapper.add_module("module", pipe)
    return wrapper


def test_pipeline_module_expert_names_use_model_layer_number():
    model = _fake_pipe()
    expert_ids = build_expert_id_map(model, num_layers=4)

    assert expert_ids["2.deepspeed_experts.0.weight"] == (0, 0)
    assert expert_ids["5.deepspeed_experts.0.weight"] == (3, 0)
    assert ids_from_name(
        "5.deepspeed_experts.0.weight", expert_ids,
    ) == (3, 0)


def test_pipeline_module_expert_map_supports_module_wrapper():
    model = _fake_pipe(wrapped=True)
    expert_ids = build_expert_id_map(model, num_layers=4)

    assert expert_ids["module.2.deepspeed_experts.0.weight"] == (0, 0)
    assert expert_ids["module.5.deepspeed_experts.0.weight"] == (3, 0)


def test_recovery_classification_uses_same_canonical_layer_map():
    model = _fake_pipe()
    expert_ids = build_expert_id_map(model, num_layers=4)

    non_expert, by_expert = _classify_params(model, expert_ids)

    assert non_expert == {}
    assert set(by_expert) == {(0, 0), (3, 0)}


def test_pipeline_module_name_without_canonical_map_fails_closed():
    with pytest.raises(ValueError, match="canonical expert-id map"):
        ids_from_name("9.deepspeed_experts.14.weight")


def test_pipeline_module_layer_outside_model_range_fails_closed():
    with pytest.raises(RuntimeError, match=r"outside \[0, 4\)"):
        build_expert_id_map(_fake_pipe(bad_layer=4), num_layers=4)


def test_pipeline_module_without_layer_number_fails_closed():
    model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.add_module("deepspeed_experts", torch.nn.ModuleList([
        torch.nn.Linear(1, 1, bias=False),
    ]))
    model.add_module("2", layer)

    with pytest.raises(RuntimeError, match="has no layer_number"):
        build_expert_id_map(model, num_layers=1)


def test_nonpipeline_expert_name_keeps_model_layer_id():
    assert ids_from_name(
        "language_model.encoder.layers.5.mlp.deepspeed_moe.experts."
        "deepspeed_experts.7.wi.weight"
    ) == (5, 7)
