from pathlib import Path

from megatron.proposal.distributed import ProposalTopology
from megatron.proposal.storage.recovery import _scan_cp_dir


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

    assert topo.writes_expert is False
    assert topo.writes_nonexpert is True
    assert topo.selection_ranks == [0, 1, 4, 5]
    assert topo.nonexpert_shard_rank == 3
    assert topo.nonexpert_shard_count == 4
    assert topo.nonexpert_dir(Path("/cp")) == Path("/cp/pp1/non_expert")
    assert topo.expert_dir(Path("/cp")) == Path("/cp/dp0/pp1/ep1/expert")


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
