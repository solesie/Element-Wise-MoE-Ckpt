from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

from megatron.proposal.cli_args import add_proposal_args
from megatron.proposal.config import ProposalConfig
from megatron.proposal.storage.lossless_compaction import ExpertLosslessFC


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_proposal_args(parser)
    return parser


def _scheduler(
    layers: list[int],
    experts_per_layer: int,
    *,
    interval_iters: int = 4,
    moc_sequential_policy: str = "layer_rr",
    lfcl_fraction_pct: float = 1.0,
) -> ExpertLosslessFC:
    expert_ids = [
        (layer, expert)
        for layer in layers
        for expert in range(experts_per_layer)
    ]
    return ExpertLosslessFC(
        all_expert_ids=expert_ids,
        separated_tensors_by_slot={expert_id: [] for expert_id in expert_ids},
        sorted_layers=list(layers),
        expert_state=None,
        interval_iters=int(interval_iters),
        schedule="sequential",
        moc_sequential_policy=moc_sequential_policy,
        lfcl_fraction_pct=lfcl_fraction_pct,
        commit_payload=False,
    )


def test_sequential_interval_is_cli_configurable() -> None:
    args = _parser().parse_args(
        [
            "--proposal-cp-mode",
            "sequential",
            "--proposal-lfcl-interval-iters",
            "4",
        ]
    )
    config = ProposalConfig.from_args(args, "/tmp/moc-config-test")
    assert config.cp_mode == "sequential"
    assert config.lfcl_interval_iters == 4


def test_proposal_mode_keeps_fixed_one_iteration_lfcl() -> None:
    args = _parser().parse_args(["--proposal-lfcl-interval-iters", "4"])
    with pytest.raises(ValueError, match="only valid.*sequential"):
        ProposalConfig.from_args(args, "/tmp/proposal-config-test")


def test_expert_rr_pp_locked_is_explicitly_moc_only() -> None:
    args = _parser().parse_args(
        ["--proposal-moc-sequential-policy", "expert_rr_pp_locked"]
    )
    with pytest.raises(ValueError, match="only valid.*sequential"):
        ProposalConfig.from_args(args, "/tmp/moc-pp-locked-invalid")

    args = _parser().parse_args(
        [
            "--proposal-cp-mode",
            "sequential",
            "--proposal-moc-sequential-policy",
            "expert_rr_pp_locked",
            "--proposal-lfcl-interval-iters",
            "1",
            "--proposal-lfcl-fraction-pct",
            "1.0",
        ]
    )
    config = ProposalConfig.from_args(args, "/tmp/moc-pp-locked-valid")
    assert config.moc_sequential_policy == "expert_rr_pp_locked"


@pytest.mark.parametrize("fraction_pct", [1.0, 4.0])
def test_expert_rr_pp_back_locked_accepts_topology_derived_budgets(
    fraction_pct: float,
) -> None:
    args = _parser().parse_args(
        [
            "--proposal-cp-mode",
            "sequential",
            "--proposal-moc-sequential-policy",
            "expert_rr_pp_back_locked",
            "--proposal-lfcl-interval-iters",
            "1",
            "--proposal-lfcl-fraction-pct",
            str(fraction_pct),
        ]
    )
    config = ProposalConfig.from_args(args, "/tmp/moc-back-locked-valid")
    assert config.moc_sequential_policy == "expert_rr_pp_back_locked"
    assert config.lfcl_fraction_pct == pytest.approx(fraction_pct)


def test_expert_rr_pp_back_locked_also_supports_single_rank() -> None:
    scheduler = _scheduler(
        [1, 3, 5],
        experts_per_layer=4,
        interval_iters=1,
        moc_sequential_policy="expert_rr_pp_back_locked",
        lfcl_fraction_pct=1.0,
    )
    fired = [
        expert
        for step in range(1, 101)
        for expert in scheduler.plan_if_due(step, world_size=1)
    ]
    assert len(fired) == 12
    assert set(fired) == {
        (layer, expert)
        for layer in (1, 3, 5)
        for expert in range(4)
    }


def test_sequential_fires_one_whole_local_layer_every_four_steps() -> None:
    scheduler = _scheduler([1, 3, 5], experts_per_layer=4)
    fired_by_step = {
        step: scheduler.plan_if_due(step)
        for step in range(1, 14)
    }

    assert fired_by_step[1] == [(1, expert) for expert in range(4)]
    assert fired_by_step[5] == [(3, expert) for expert in range(4)]
    assert fired_by_step[9] == [(5, expert) for expert in range(4)]
    assert fired_by_step[13] == [(1, expert) for expert in range(4)]
    assert all(
        not fired_by_step[step]
        for step in (2, 3, 4, 6, 7, 8, 10, 11, 12)
    )


def test_pp2_stages_advance_matching_local_layer_indices() -> None:
    pp0 = _scheduler([1, 3, 5, 7, 9, 11], experts_per_layer=32)
    pp1 = _scheduler([13, 15, 17, 19, 21, 23], experts_per_layer=32)

    expected = [
        (1, 13),
        (3, 15),
        (5, 17),
        (7, 19),
        (9, 21),
        (11, 23),
        (1, 13),
    ]
    for step, (pp0_layer, pp1_layer) in zip(range(1, 26, 4), expected):
        pp0_fired = pp0.plan_if_due(step)
        pp1_fired = pp1.plan_if_due(step)
        assert len(pp0_fired) == 32
        assert len(pp1_fired) == 32
        assert {layer for layer, _expert in pp0_fired} == {pp0_layer}
        assert {layer for layer, _expert in pp1_fired} == {pp1_layer}


def test_warmup_boundary_keeps_absolute_four_iteration_cadence() -> None:
    scheduler = _scheduler([1, 3, 5, 7, 9, 11], experts_per_layer=32)

    assert len(scheduler.plan_if_due(81)) == 32
    assert not scheduler.plan_if_due(82)
    assert not scheduler.plan_if_due(83)
    assert not scheduler.plan_if_due(84)
    fired = scheduler.plan_if_due(85)
    assert len(fired) == 32
    assert {layer for layer, _expert in fired} == {3}


def test_one_percent_profile_fires_one_layer_every_sixteen_steps() -> None:
    scheduler = _scheduler(
        [1, 3, 5, 7, 9, 11],
        experts_per_layer=32,
        interval_iters=16,
    )

    fired_steps = []
    fired_layers = []
    for step in range(1, 50):
        fired = scheduler.plan_if_due(step)
        if fired:
            fired_steps.append(step)
            fired_layers.append(next(iter({layer for layer, _expert in fired})))
            assert len(fired) == 32
    assert fired_steps == [1, 17, 33, 49]
    assert fired_layers == [1, 3, 5, 7]


def test_expert_rr_pp_locked_writes_four_global_slots_then_idles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # PP0 owns the first three MoE layers on ranks 0..3; PP1 owns the last
    # three on ranks 4..7. EP4 gives every rank 16 local experts per layer.
    rank_layers = {
        rank: ([1, 3, 5] if rank < 4 else [7, 9, 11])
        for rank in range(8)
    }
    shards = []
    schedulers = []
    for rank in range(8):
        expert_ids = [
            (layer, expert)
            for layer in rank_layers[rank]
            for expert in range(16)
        ]
        shards.append(
            [
                (rank, idx, int(layer), int(expert))
                for idx, (layer, expert) in enumerate(expert_ids)
            ]
        )
        schedulers.append(
            _scheduler(
                rank_layers[rank],
                experts_per_layer=16,
                interval_iters=1,
                moc_sequential_policy="expert_rr_pp_locked",
                lfcl_fraction_pct=1.0,
            )
        )

    current_rank = {"value": 0}
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr(
        "torch.distributed.get_rank", lambda: current_rank["value"]
    )
    monkeypatch.setattr(
        "torch.distributed.get_world_size", lambda group=None: 8
    )

    def fake_all_gather_object(output, _local, group=None):
        del group
        output[:] = shards

    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        fake_all_gather_object,
    )

    frame_records = []
    first_step = None
    for step in range(1, 101):
        selected = []
        for rank, scheduler in enumerate(schedulers):
            current_rank["value"] = rank
            fired = scheduler.plan_if_due(
                step,
                process_group=object(),
                world_size=8,
            )
            selected.extend((rank, layer, expert) for layer, expert in fired)

        if step <= 96:
            assert len(selected) == 4
            frame_records.extend(selected)
            if first_step is None:
                first_step = list(selected)
            if step <= 48:
                assert {rank for rank, _layer, _expert in selected} <= set(range(4))
            else:
                assert {rank for rank, _layer, _expert in selected} <= set(range(4, 8))
        else:
            assert selected == []

    assert len(frame_records) == 384
    assert len(set(frame_records)) == 384

    repeated = []
    for rank, scheduler in enumerate(schedulers):
        current_rank["value"] = rank
        fired = scheduler.plan_if_due(
            101,
            process_group=object(),
            world_size=8,
        )
        repeated.extend((rank, layer, expert) for layer, expert in fired)
    assert repeated == first_step


@pytest.mark.parametrize(
    ("rank_layers", "experts_per_layer", "fraction_pct", "expected_repeats"),
    [
        (
            {
                rank: ([1, 3, 5] if rank < 4 else [7, 9, 11])
                for rank in range(8)
            },
            16,
            1.0,
            1,
        ),
        (
            {
                rank: (
                    [1, 3, 5, 7, 9, 11]
                    if rank < 4 else
                    [13, 15, 17, 19, 21, 23]
                )
                for rank in range(8)
            },
            32,
            1.0,
            1,
        ),
        (
            {
                rank: (
                    [1, 3, 5, 7, 9, 11]
                    if rank < 4 else
                    [13, 15, 17, 19, 21, 23]
                )
                for rank in range(8)
            },
            32,
            4.0,
            4,
        ),
    ],
)
def test_expert_rr_pp_back_locked_generalizes_and_repeats_fault_phase(
    monkeypatch: pytest.MonkeyPatch,
    rank_layers: dict[int, list[int]],
    experts_per_layer: int,
    fraction_pct: float,
    expected_repeats: int,
) -> None:
    shards = []
    schedulers = []
    for rank in range(8):
        expert_ids = [
            (layer, expert)
            for layer in rank_layers[rank]
            for expert in range(experts_per_layer)
        ]
        shards.append(
            [
                (rank, idx, int(layer), int(expert))
                for idx, (layer, expert) in enumerate(expert_ids)
            ]
        )
        schedulers.append(
            _scheduler(
                rank_layers[rank],
                experts_per_layer=experts_per_layer,
                interval_iters=1,
                moc_sequential_policy="expert_rr_pp_back_locked",
                lfcl_fraction_pct=fraction_pct,
            )
        )

    current_rank = {"value": 0}
    gather_calls = {"count": 0}
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr(
        "torch.distributed.get_rank", lambda: current_rank["value"]
    )
    monkeypatch.setattr(
        "torch.distributed.get_world_size", lambda group=None: 8
    )

    def fake_all_gather_object(output, _local, group=None):
        del group
        gather_calls["count"] += 1
        output[:] = shards

    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        fake_all_gather_object,
    )

    selected_by_step = []
    all_records = []
    for step in range(1, 101):
        selected = []
        for rank, scheduler in enumerate(schedulers):
            current_rank["value"] = rank
            fired = scheduler.plan_if_due(
                step,
                process_group=object(),
                world_size=8,
            )
            selected.extend((rank, layer, expert) for layer, expert in fired)
        selected_by_step.append(selected)
        all_records.extend(selected)

    total_slots = sum(len(records) for records in shards)
    assert len(all_records) == total_slots * expected_repeats
    counts = {
        record: all_records.count(record)
        for record in set(all_records)
    }
    assert len(counts) == total_slots
    assert set(counts.values()) == {expected_repeats}
    assert {layer for _rank, layer, _expert in selected_by_step[0]} == {
        max(layer for layers in rank_layers.values() for layer in layers)
    }
    assert {layer for _rank, layer, _expert in selected_by_step[-1]} == {
        min(layer for layers in rank_layers.values() for layer in layers)
    }

    # The topology gather is cached: one collective per scheduler/rank, not
    # one collective per training iteration.
    assert gather_calls["count"] == 8

    repeated_first_step = []
    for rank, scheduler in enumerate(schedulers):
        current_rank["value"] = rank
        fired = scheduler.plan_if_due(
            101,
            process_group=object(),
            world_size=8,
        )
        repeated_first_step.extend(
            (rank, layer, expert) for layer, expert in fired
        )
    assert repeated_first_step == selected_by_step[0]


def test_one_percent_proposal_budget_knobs_are_cli_configurable() -> None:
    args = _parser().parse_args(
        [
            "--proposal-topk-ratio",
            "0.01",
            "--proposal-lfcl-fraction-pct",
            "0.3",
            "--proposal-age-force-period",
            "512",
        ]
    )
    config = ProposalConfig.from_args(args, "/tmp/proposal-one-percent-test")
    assert config.topk_ratio == pytest.approx(0.01)
    assert config.lfcl_fraction_pct == pytest.approx(0.3)
    assert config.age_force_period == 512


def test_all_canonical_moc_launchers_require_lr_warmup_80() -> None:
    moe_dir = Path(__file__).resolve().parents[3] / "examples_deepspeed" / "MoE"
    launchers = [
        "ds_pretrain_gpt_125M_MoE64_10Kiter_moc.sh",
        "ds_pretrain_gpt_125M_MoE64_10Kiter_moc_1pct.sh",
        "ds_pretrain_gpt_125M_MoE64_10Kiter_moc_1pct_expert_rr_pp_locked.sh",
        "ds_pretrain_gpt_125M_MoE64_10Kiter_moc_1pct_pp_back_locked.sh",
        "ds_pretrain_gpt_125M_MoE64_10Kiter_moc_4pct_pp_back_locked.sh",
        "ds_pretrain_gpt_350M_MoE128_10Kiter_moc.sh",
        "ds_pretrain_gpt_350M_MoE128_10Kiter_moc_1pct.sh",
        "ds_pretrain_gpt_350M_MoE128_10Kiter_moc_1pct_pp_back_locked.sh",
        "ds_pretrain_gpt_350M_MoE128_10Kiter_moc_4pct_pp_back_locked.sh",
    ]
    for launcher in launchers:
        source = (moe_dir / launcher).read_text(encoding="utf-8")
        assert "WARMUP_ITERS=${WARMUP_ITERS:-80}" in source, launcher
        assert "WARMUP_ITERS=${WARMUP_ITERS:-0}" not in source, launcher

    for launcher in (
        "ds_pretrain_gpt_125M_MoE64_10Kiter_moc.sh",
        "ds_pretrain_gpt_350M_MoE128_10Kiter_moc.sh",
    ):
        source = (moe_dir / launcher).read_text(encoding="utf-8")
        assert "EXPECTED_WARMUP_TOKENS=" in source, launcher
        assert "ALLOW_MOC_ZERO_LR_WARMUP" in source, launcher
        assert "lr_warmup_iters=${WARMUP_ITERS}" in source, launcher


@pytest.mark.parametrize(
    ("flag", "value", "match"),
    [
        ("--proposal-lfcl-fraction-pct", "0", "0 < pct <= 100"),
        ("--proposal-lfcl-fraction-pct", "100.1", "0 < pct <= 100"),
        ("--proposal-age-force-period", "0", "1 <= period"),
    ],
)
def test_one_percent_budget_knobs_reject_invalid_values(
    flag: str,
    value: str,
    match: str,
) -> None:
    args = _parser().parse_args([flag, value])
    with pytest.raises(ValueError, match=match):
        ProposalConfig.from_args(args, "/tmp/proposal-invalid-budget-test")


def test_point_three_percent_global_lfcl_quota_rounds_to_five_slots() -> None:
    expert_ids = [(layer, expert) for layer in range(12) for expert in range(128)]
    scheduler = ExpertLosslessFC(
        all_expert_ids=expert_ids,
        separated_tensors_by_slot={expert_id: [] for expert_id in expert_ids},
        sorted_layers=list(range(12)),
        expert_state=None,
        interval_iters=1,
        schedule="proposal",
        lfcl_fraction_pct=0.3,
        commit_payload=False,
    )
    assert scheduler.num_expert_slots == 1536
    assert scheduler.experts_per_lfcl_event == 5
    assert len(scheduler.plan_if_due(1)) == 5


def test_distributed_round_robin_rounds_point_three_percent_globally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.proposal.storage.lfcl_plan import RoundRobinPlan

    # Eight rank shards of 192 distinct logical slots: the global quota must
    # be ceil(1536 * 0.3%) == 5, not one locally rounded slot per rank.
    shards = []
    for rank in range(8):
        records = []
        for idx in range(192):
            global_slot = rank * 192 + idx
            records.append((rank, idx, global_slot // 128, global_slot % 128))
        shards.append(records)

    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr(
        "torch.distributed.get_world_size",
        lambda group=None: 8,
    )

    def fake_all_gather_object(output, _local, group=None):
        del group
        output[:] = shards

    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        fake_all_gather_object,
    )
    rank0_ids = [(slot // 128, slot % 128) for slot in range(192)]
    scheduler = ExpertLosslessFC(
        all_expert_ids=rank0_ids,
        separated_tensors_by_slot={eid: [] for eid in rank0_ids},
        sorted_layers=[0, 1],
        expert_state=None,
        schedule="proposal",
        lfcl_fraction_pct=0.3,
        planner=RoundRobinPlan(),
        commit_payload=False,
    )
    fired = scheduler.plan_if_due(step=1, process_group=object(), world_size=8)
    assert fired == [(0, expert) for expert in range(5)]


def test_controller_threads_lfcl_fraction_to_capacity_and_scheduler_owners() -> None:
    controller_path = Path(__file__).resolve().parents[1] / "controller.py"
    tree = ast.parse(controller_path.read_text())
    calls: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else ""
        )
        if name in {"FCManager", "ExpertLosslessFC"}:
            calls[name] = node

    assert set(calls) == {"FCManager", "ExpertLosslessFC"}
    for call in calls.values():
        assert any(
            keyword.arg == "lfcl_fraction_pct"
            for keyword in call.keywords
        )
