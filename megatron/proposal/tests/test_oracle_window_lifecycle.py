from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from megatron.proposal.cli_args import add_proposal_args
from megatron.proposal.config import ProposalConfig
from megatron.proposal.telemetry.oracle_window.session import OracleWindowSession


class _RecordingCollector:
    def __init__(self) -> None:
        self.events = []

    def prepare_window(self, **kwargs) -> None:
        self.events.append(("start", kwargs["start_iteration"]))

    def observe_successful_update(self, **kwargs) -> None:
        self.events.append(
            (
                "update",
                kwargs["iteration"],
                kwargs["offset"],
                kwargs["snapshot_due"],
            )
        )

    def finish_window(self, **kwargs) -> None:
        self.events.append(
            ("finish", kwargs["start_iteration"], kwargs["end_iteration"])
        )


def test_window_offset_zero_then_100_successful_updates() -> None:
    collector = _RecordingCollector()
    session = OracleWindowSession(start_iterations=(100,))
    session.attach_collector(collector)

    assert session.maybe_start(
        model=object(), forward_step_func=object(), data_iterator=object(), iteration=100
    )
    assert collector.events == [("start", 100)]  # no offset-zero selection

    assert not session.observe_post_optimizer(
        model=object(), iteration=101, applied=False
    )
    for offset in range(1, 101):
        assert session.observe_post_optimizer(
            model=object(), iteration=101 + offset, applied=True
        )

    updates = [event for event in collector.events if event[0] == "update"]
    assert [event[2] for event in updates] == list(range(1, 101))
    assert [event[2] for event in updates if event[3]] == list(range(10, 101, 10))
    assert collector.events[-1] == ("finish", 100, 201)
    assert not session.active


def test_window_requires_collector_and_rejects_overlap() -> None:
    session = OracleWindowSession(start_iterations=(100,))
    with pytest.raises(RuntimeError, match="no attached collector"):
        session.maybe_start(
            model=None, forward_step_func=None, data_iterator=None, iteration=100
        )

    with pytest.raises(ValueError, match="overlap"):
        OracleWindowSession(start_iterations=(100, 200), window_updates=100)


def test_window_shutdown_is_fail_closed_for_active_or_unstarted_work() -> None:
    pending = OracleWindowSession(
        start_iterations=(100,), window_updates=1, snapshot_offsets=(1,),
    )
    pending.attach_collector(_RecordingCollector())
    with pytest.raises(RuntimeError, match="before configured"):
        pending.assert_complete()

    active = OracleWindowSession(
        start_iterations=(100,), window_updates=2, snapshot_offsets=(1, 2),
    )
    active.attach_collector(_RecordingCollector())
    active.maybe_start(
        model=None, forward_step_func=None, data_iterator=None, iteration=100,
    )
    active.observe_post_optimizer(model=None, iteration=101, applied=True)
    with pytest.raises(RuntimeError, match="active oracle"):
        active.assert_complete()

    complete = OracleWindowSession(
        start_iterations=(100,), window_updates=1, snapshot_offsets=(1,),
    )
    complete.attach_collector(_RecordingCollector())
    complete.maybe_start(
        model=None, forward_step_func=None, data_iterator=None, iteration=100,
    )
    complete.observe_post_optimizer(model=None, iteration=101, applied=True)
    complete.assert_complete()


def test_skips_can_extend_window_and_runtime_overlap_is_rejected() -> None:
    session = OracleWindowSession(
        start_iterations=(100, 201),
        window_updates=100,
        snapshot_offsets=(100,),
    )
    session.attach_collector(_RecordingCollector())
    session.maybe_start(
        model=None, forward_step_func=None, data_iterator=None, iteration=100
    )
    for iteration in range(101, 201):
        session.observe_post_optimizer(
            model=None,
            iteration=iteration,
            applied=False,
        )
    with pytest.raises(RuntimeError, match="overlap at runtime"):
        session.maybe_start(
            model=None, forward_step_func=None, data_iterator=None, iteration=201
        )


def _parse_config(*extra: str) -> ProposalConfig:
    parser = argparse.ArgumentParser()
    add_proposal_args(parser)
    args = parser.parse_args(list(extra))
    return ProposalConfig.from_args(args, "/tmp/oracle-window-config")


def test_measurement_mode_config_defaults_and_fault_guard() -> None:
    config = _parse_config(
        "--proposal-independent-oracle-enable",
        "--proposal-independent-oracle-iters",
        "1000",
        "--proposal-oracle-window-measurement-only",
    )
    assert config.oracle_window_measurement_only
    assert config.oracle_window_updates == 100
    assert config.oracle_window_snapshot_offsets == tuple(range(10, 101, 10))
    assert config.oracle_window_fd_epsilon == pytest.approx(1.0e-2)
    assert config.oracle_window_fd_probes == 4
    assert config.oracle_window_fd_data_batches == 1
    assert config.oracle_window_fd_min_changed_fraction == pytest.approx(0.95)
    assert config.fault_interval is None

    with pytest.raises(ValueError, match="fault-interval 0"):
        _parse_config(
            "--proposal-independent-oracle-enable",
            "--proposal-independent-oracle-iters",
            "1000",
            "--proposal-oracle-window-measurement-only",
            "--proposal-fault-interval",
            "10",
        )


def test_measurement_mode_requires_explicit_nonoverlapping_oracle_starts() -> None:
    with pytest.raises(ValueError, match="requires explicit"):
        _parse_config(
            "--proposal-independent-oracle-enable",
            "--proposal-oracle-window-measurement-only",
        )
    with pytest.raises(ValueError, match="overlap"):
        _parse_config(
            "--proposal-independent-oracle-enable",
            "--proposal-independent-oracle-iters",
            "100,200",
            "--proposal-oracle-window-measurement-only",
        )
    with pytest.raises(ValueError, match="selection-mode hessian"):
        _parse_config(
            "--proposal-independent-oracle-enable",
            "--proposal-independent-oracle-iters",
            "100",
            "--proposal-oracle-window-measurement-only",
            "--proposal-selection-mode",
            "random",
        )


def test_training_and_controller_source_keep_measurement_path_isolated() -> None:
    proposal_root = Path(__file__).resolve().parents[1]
    training = (proposal_root.parent / "training.py").read_text(encoding="utf-8")
    assert "proposal_oracle_window_post_optimizer(" in training
    assert "applied=(skipped_iter == 0)" in training
    refresh_end = training.find("\ndef ", training.index("def _proposal_h_refresh_due") + 1)
    if refresh_end < 0:
        refresh_end = len(training)
    refresh_due = training[
        training.index("def _proposal_h_refresh_due"):
        refresh_end
    ]
    assert "proposal_oracle_window_measurement_only" in refresh_due
    assert "return int(iteration) in starts" in refresh_due

    controller = (proposal_root / "controller.py").read_text(encoding="utf-8")
    assert "if (_oracle_window_only or cp_target != \"test_dram\")" in controller
    assert "build_rank_owned_oracle_window_engine(" in controller
    assert "_oracle_session.attach_collector(_oracle_collector)" in controller
    dispatch_start = controller.index("def proposal_dispatch_post_iter(")
    dispatch_end = controller.index("def proposal_finalize_post_iter(", dispatch_start)
    dispatch = controller[dispatch_start:dispatch_end]
    assert dispatch.index("_oracle_window_measurement_only") < dispatch.index(
        "_joint_saliency_collector"
    )

    activation_start = controller.index("def proposal_activate_after_warmup(")
    activation_end = controller.index("def proposal_prepare_after_optimizer(")
    activation = controller[activation_start:activation_end]
    assert activation.index("oracle_window_measurement_only") < activation.index(
        "bootstrap_expert_recovery_baseline"
    )

    engine = (
        proposal_root / "checkpoint" / "lossy_checkpoint_engine.py"
    ).read_text(encoding="utf-8")
    joint_init = engine[engine.index("# Joint saliency is part"):]
    assert "oracle_window_measurement_only" in joint_init
