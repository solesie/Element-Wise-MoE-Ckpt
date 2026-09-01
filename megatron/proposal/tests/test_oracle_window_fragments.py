from __future__ import annotations

import pytest
import torch

from megatron.proposal.storage.expert_partition import ActiveTrackedFragmentSpec
from megatron.proposal.telemetry.oracle_window.fragments import (
    FragmentCoverageRecord,
    build_expert_layout,
    build_nonexpert_layout,
    stable_u64,
    validate_fragment_coverage,
)


class _Mapper:
    def __init__(self, rows, p, m, v):
        self.rows = list(rows)
        self.slots = {None: p, "exp_avg": m, "exp_avg_sq": v}

    def zero12_local_fragments(self, named_params):
        wanted = {name for name, _param in named_params}
        return [row for row in self.rows if row[0] in wanted]

    def zero12_partition_view(self, group_idx, local_offset, numel, *, state_key=None):
        assert int(group_idx) == 0
        return self.slots[state_key].narrow(0, int(local_offset), int(numel))


def _mapper_for_rows(rows, total):
    p = torch.arange(total, dtype=torch.float32)
    m = p + 100.0
    v = p + 200.0
    return _Mapper(rows, p, m, v), p, m, v


def test_nonexpert_layout_uses_local_optimizer_fragments_and_stable_ids() -> None:
    p0 = torch.nn.Parameter(torch.zeros(8))
    p1 = torch.nn.Parameter(torch.zeros(5))
    rows = [
        ("p1", p1, 0, 4, 0, 5),
        ("p0", p0, 0, 0, 2, 4),
    ]
    mapper, p, m, v = _mapper_for_rows(rows, 9)

    layout = build_nonexpert_layout(
        mapper=mapper,
        named_params=(("p0", p0), ("p1", p1)),
        pp_rank=1,
        namespace="test-model",
    )

    assert layout.total_numel == 9
    # Canonical ordering is independent of mapper enumeration order.
    assert [fragment.parameter_name for fragment in layout.fragments] == ["p0", "p1"]
    assert [fragment.packed_offset for fragment in layout.fragments] == [0, 4]
    first = layout.fragments[0]
    assert first.parameter_offset == 2
    assert first.canonical_parameter_id == stable_u64(first.canonical_key)
    assert first.canonical_key == "test-model/pp=1/nonexpert/p0"
    torch.testing.assert_close(first.p, p[:4])
    torch.testing.assert_close(first.m, m[:4])
    torch.testing.assert_close(first.v, v[:4])


def test_expert_layout_uses_global_expert_identity_not_runtime_suffix() -> None:
    parameter = torch.nn.Parameter(torch.zeros(6))
    rows = [("local.expert.weight", parameter, 0, 0, 1, 5)]
    mapper, _p, _m, _v = _mapper_for_rows(rows, 5)
    spec = ActiveTrackedFragmentSpec(
        runtime_name="rank-local#frag0",
        expert_id=(17, 42),
        parameter_name="local.expert.weight",
        parameter_offset=1,
        numel=5,
        parameter_numel=6,
        parameter_shape=(6,),
        optimizer_group_idx=0,
        optimizer_local_offset=0,
    )

    layout = build_expert_layout(
        mapper=mapper,
        specs=(spec,),
        name_to_param={"local.expert.weight": parameter},
        namespace="test-model",
        global_expert_base=32,
    )

    fragment = layout.fragments[0]
    assert fragment.runtime_name == "rank-local#frag0"
    assert fragment.expert_id == (17, 74)
    assert "/layer=17/expert=74/" in fragment.canonical_key
    assert "rank-local#frag0" not in fragment.canonical_key


def test_global_coverage_rejects_dp_duplicates_and_partition_gaps() -> None:
    key = "model/pp=0/nonexpert/w"
    identity = stable_u64(key)
    full = (
        FragmentCoverageRecord("nonexpert", key, identity, 0, 4, 8, 0),
        FragmentCoverageRecord("nonexpert", key, identity, 4, 4, 8, 1),
    )
    validate_fragment_coverage(full, require_complete=True)

    with pytest.raises(RuntimeError, match="overlapping"):
        validate_fragment_coverage(
            full + (FragmentCoverageRecord("nonexpert", key, identity, 0, 4, 8, 2),),
            require_complete=True,
        )
    with pytest.raises(RuntimeError, match="missing optimizer ownership"):
        validate_fragment_coverage(
            (
                FragmentCoverageRecord("nonexpert", key, identity, 0, 3, 8, 0),
                FragmentCoverageRecord("nonexpert", key, identity, 4, 4, 8, 1),
            ),
            require_complete=True,
        )


def test_fragment_builder_rejects_non_fp32_optimizer_state() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4))
    rows = [("p", parameter, 0, 0, 0, 4)]
    mapper = _Mapper(
        rows,
        torch.zeros(4, dtype=torch.float32),
        torch.zeros(4, dtype=torch.float16),
        torch.zeros(4, dtype=torch.float32),
    )
    with pytest.raises(RuntimeError, match="active FP32 optimizer slot"):
        build_nonexpert_layout(
            mapper=mapper,
            named_params=(("p", parameter),),
            pp_rank=0,
        )
