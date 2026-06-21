import pytest
import torch

from megatron.proposal.cutoff.global_cpu_decision import (
    choose_global_probe_decision_cpu,
    retry_candidate_ladder_cpu,
)


def _decision(**kwargs):
    defaults = dict(
        stats=None,
        candidates=[1.0, 2.0],
        n_candidates=2,
        target_ratio=0.25,
        cutoff_tolerance=0.04,
        sample_blocks_min=4,
        sample_blocks_current=16,
        sample_blocks_cap=64,
        ci_shortfall_streak=0,
        sample_grow_hold=0,
        ci_shortfall_grow_after=3,
        run_if_decision_code=-1,
    )
    defaults.update(kwargs)
    return choose_global_probe_decision_cpu(**defaults)


def test_cpu_decision_first_pass_requests_retry_like_device_path():
    stats = torch.tensor(
        [
            [0.40, 0.10],
            [0.0, 0.0],
            [0.0, 0.0],
            [16.0, 16.0],
        ],
        dtype=torch.float32,
    )

    out = _decision(stats=stats)

    assert out["decision"] == 3
    assert out["needs_retry"] is True
    assert out["strict_accept"] is False
    assert out["chosen_ratio"] == pytest.approx(0.25)
    assert out["bracket_width"] > 0.0


def test_cpu_decision_retry_pass_uses_final_code_not_retry_code():
    stats = torch.tensor(
        [
            [0.40, 0.10],
            [0.0, 0.0],
            [0.0, 0.0],
            [16.0, 16.0],
        ],
        dtype=torch.float32,
    )

    out = _decision(stats=stats, run_if_decision_code=3)

    assert out["decision"] == 4
    assert out["needs_retry"] is False
    assert out["strict_accept"] is False
    assert out["chosen_ratio"] == pytest.approx(0.25)
    assert out["bracket_width"] == pytest.approx(0.0)


def test_cpu_decision_retry_pass_grows_sample_after_ci_shortfall():
    stats = torch.tensor(
        [
            [0.80, 0.70],
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, 2.0],
        ],
        dtype=torch.float32,
    )

    out = _decision(
        stats=stats,
        run_if_decision_code=3,
        ci_shortfall_streak=2,
        sample_blocks_current=16,
        sample_blocks_cap=64,
    )

    assert out["decision"] == 4
    assert out["needs_retry"] is False
    assert out["next_sample"] == 32
    assert out["stored_streak"] == 0
    assert out["grow_hold_value"] == 1


def test_cpu_decision_rejects_zero_ratio_candidates_like_device_path():
    stats = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [16.0, 16.0],
        ],
        dtype=torch.float32,
    )

    out = _decision(stats=stats)

    assert out["decision"] == 0
    assert out["accept"] is False
    assert out["chosen_tau"] == pytest.approx(0.0)


def test_retry_ladder_keeps_device_path_centering():
    decision = {
        "chosen_tau": 1.25e-9,
        "chosen_ratio": 0.04,
        "bracket_width": 1.6,
    }

    candidates = retry_candidate_ladder_cpu(
        decision,
        source_candidates=[1.0e-10, 5.0e-9],
        capacity=16,
        target_ratio=0.04,
    )

    assert len(candidates) == 16
    assert candidates[7] < decision["chosen_tau"] < candidates[8]
