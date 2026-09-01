from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from megatron.proposal.emission.record_kernels import LossyRecordWriteInputs
from megatron.proposal.telemetry.independent_oracle import (
    _age2_phase_hit_np,
    _age2_phase_hit_tensor,
)


_PROPOSAL_ROOT = Path(__file__).resolve().parents[1]


def test_production_probe_and_emission_use_configured_force_period() -> None:
    probe = (_PROPOSAL_ROOT / "cutoff" / "approx_probe_kernels.py").read_text()
    emission = (_PROPOSAL_ROOT / "emission" / "record_kernels.py").read_text()

    assert "AGE_FORCE_PERIOD: tl.constexpr" in probe
    assert "step_value, AGE_FORCE_PERIOD, 0x51ED270B" in probe
    assert "AGE_FORCE_PERIOD: tl.constexpr" in emission
    assert "AGE_FORCE_PERIOD," in emission
    assert "global_elem_idx_for_score, step_value, 128" not in emission


def _calls_named(source: str, name: str) -> list[ast.Call]:
    calls = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = (
            func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else ""
        )
        if func_name == name:
            calls.append(node)
    return calls


def test_every_probe_and_emission_launch_threads_force_period() -> None:
    cutoff_source = (
        _PROPOSAL_ROOT / "cutoff" / "cutoff_orchestrator.py"
    ).read_text()
    emission_source = (
        _PROPOSAL_ROOT / "emission" / "direct_emission.py"
    ).read_text()

    probe_calls = _calls_named(
        cutoff_source,
        "launch_generated_block_probe_for_group",
    )
    assert len(probe_calls) == 2  # first probe and guarded retry probe
    assert all(
        any(keyword.arg == "age_force_period" for keyword in call.keywords)
        for call in probe_calls
    )

    emission_calls = _calls_named(
        emission_source,
        "launch_direct_records_for_segment_entry_workqueue",
    )
    assert len(emission_calls) == 1
    assert any(
        keyword.arg == "age_force_period"
        for keyword in emission_calls[0].keywords
    )


def test_each_oldest_code_element_fires_once_per_512_step_cycle() -> None:
    element_ids = np.arange(4096, dtype=np.int64)
    hits = np.zeros(element_ids.shape, dtype=np.int32)
    for step in range(512):
        hits += _age2_phase_hit_np(
            element_ids,
            step=step,
            period=512,
            salt=0x51ED270B,
        ).astype(np.int32)

    assert np.all(hits == 1)


@pytest.mark.parametrize("period", [128, 512])
@pytest.mark.parametrize("step", [0, 1, 63, 127, 511, 4097])
def test_numpy_and_torch_force_predicates_are_bit_identical(
    period: int,
    step: int,
) -> None:
    element_ids = np.arange(8192, dtype=np.int64)
    expected = _age2_phase_hit_np(
        element_ids,
        step=step,
        period=period,
        salt=0x51ED270B,
    )
    actual = _age2_phase_hit_tensor(
        torch.from_numpy(element_ids),
        step=step,
        period=period,
        salt=0x51ED270B,
    ).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_lossy_write_inputs_carries_force_period_without_global_state() -> None:
    inputs = LossyRecordWriteInputs(
        bf16_groups_mvph=[],
        name_list=[],
        group_size=8,
        age_force_period=512,
    )
    assert inputs.age_force_period == 512
