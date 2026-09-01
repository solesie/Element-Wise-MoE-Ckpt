from __future__ import annotations

import os

import pytest
import torch

from megatron.proposal.telemetry.oracle_window.fragments import build_nonexpert_layout
from megatron.proposal.telemetry.oracle_window.gpu import (
    build_block_sampling_plan,
    canonical_element_id,
    deterministic_block_element,
    score_reference,
)


class _Mapper:
    def __init__(self, rows, total):
        self.rows = list(rows)
        self.p = torch.zeros(total, dtype=torch.float32)
        self.m = torch.zeros(total, dtype=torch.float32)
        self.v = torch.ones(total, dtype=torch.float32)

    def zero12_local_fragments(self, named_params):
        wanted = {name for name, _param in named_params}
        return [row for row in self.rows if row[0] in wanted]

    def zero12_partition_view(self, group_idx, local_offset, numel, *, state_key=None):
        source = {None: self.p, "exp_avg": self.m, "exp_avg_sq": self.v}[state_key]
        return source.narrow(0, int(local_offset), int(numel))


def _cpu_layout():
    large = torch.nn.Parameter(torch.zeros(1000))
    small = torch.nn.Parameter(torch.zeros(100))
    mapper = _Mapper(
        [
            ("large", large, 0, 0, 0, 1000),
            ("small", small, 0, 1000, 0, 100),
        ],
        1100,
    )
    return build_nonexpert_layout(
        mapper=mapper,
        named_params=(("large", large), ("small", small)),
        pp_rank=0,
        namespace="sampling-test",
    )


def test_score_reference_preserves_signed_raw_and_clips_only_selection() -> None:
    p = torch.tensor([2.0, 2.0])
    theta = torch.tensor([1.0, 1.0])
    m = torch.tensor([2.0, -2.0])
    h = torch.tensor([1.0, 1.0])

    raw, clipped = score_reference(p=p, m=m, h_abs=h, theta_cp=theta)

    torch.testing.assert_close(raw, torch.tensor([-1.5, 2.5]))
    torch.testing.assert_close(clipped, torch.tensor([0.0, 2.5]))


def test_block_sampling_plan_preserves_population_metadata_and_weights() -> None:
    layout = _cpu_layout()
    plan = build_block_sampling_plan(layout, block_elements=100, sample_blocks=6)

    assert plan.population_elements == 1100
    assert plan.population_blocks == 11
    assert plan.sampled_blocks == 6
    by_size = {part.population_elements: part.sampled_blocks for part in plan.fragments}
    assert by_size == {1000: 5, 100: 1}
    assert sum(plan.population_weights()) == pytest.approx(1100.0)


def test_block_sampling_plan_weights_each_actual_stratum_width() -> None:
    parameter = torch.nn.Parameter(torch.zeros(300))
    mapper = _Mapper(
        [("partial", parameter, 0, 0, 0, 300)],
        300,
    )
    layout = build_nonexpert_layout(
        mapper=mapper,
        named_params=(("partial", parameter),),
        pp_rank=0,
        namespace="partial-stratum-test",
    )

    plan = build_block_sampling_plan(
        layout,
        block_elements=256,
        sample_blocks=2,
    )

    # The two strata represent [0, 256) and [256, 300), not two equal halves.
    assert plan.population_weights() == pytest.approx((256.0, 44.0))
    assert sum(plan.population_weights()) == pytest.approx(300.0)


def test_round_salt_changes_deterministic_block_stratified_samples() -> None:
    kwargs = dict(
        n_samples=8,
        n_blocks=64,
        block_elements=128,
        fragment_numel=8192,
        canonical_parameter_id=0x123456789ABCDEF0,
    )
    round0 = [
        deterministic_block_element(sample_index=i, round_salt=7, **kwargs)
        for i in range(8)
    ]
    repeat = [
        deterministic_block_element(sample_index=i, round_salt=7, **kwargs)
        for i in range(8)
    ]
    round1 = [
        deterministic_block_element(sample_index=i, round_salt=8, **kwargs)
        for i in range(8)
    ]
    assert round0 == repeat
    assert round0 != round1
    assert not ({element for _block, element in round0} & {
        element for _block, element in round1
    })
    for sample_index, (block, element) in enumerate(round0):
        assert sample_index * 8 <= block < (sample_index + 1) * 8
        assert block * 128 <= element < (block + 1) * 128


@pytest.mark.parametrize(
    ("sample_index", "width"),
    ((0, 256), (1, 44)),
)
def test_round_salts_traverse_a_diverse_full_stratum_permutation(
    sample_index: int,
    width: int,
) -> None:
    kwargs = dict(
        sample_index=sample_index,
        n_samples=2,
        n_blocks=2,
        block_elements=256,
        fragment_numel=300,
        canonical_parameter_id=0x123456789ABCDEF0,
    )
    elements = [
        deterministic_block_element(round_salt=17 + round_index, **kwargs)[1]
        for round_index in range(width)
    ]
    lo = 0 if sample_index == 0 else 256

    assert len(set(elements)) == width
    assert set(elements) == set(range(lo, lo + width))
    if width > 4:
        modular_step = (elements[1] - elements[0]) % width
        assert modular_step not in (1, width - 1)


def test_canonical_element_identity_is_stable_and_offset_sensitive() -> None:
    parameter_id = 0xFEDCBA9876543210
    assert canonical_element_id(parameter_id, 17) == canonical_element_id(
        parameter_id, 17,
    )
    assert canonical_element_id(parameter_id, 17) != canonical_element_id(
        parameter_id, 18,
    )


@pytest.mark.skipif(
    os.environ.get("PROPOSAL_RUN_ORACLE_WINDOW_CUDA_TESTS") != "1"
    or not torch.cuda.is_available(),
    reason="explicit opt-in CUDA/Triton smoke test",
)
def test_cuda_primitives_are_opt_in_to_avoid_touching_live_experiments() -> None:
    # Full integration requires an active optimizer mapper and finalized FD-H
    # views; that belongs to the outer oracle orchestrator.  This marker keeps
    # regular unit runs CPU-only and documents the explicit CUDA test gate.
    assert torch.cuda.current_device() >= 0
