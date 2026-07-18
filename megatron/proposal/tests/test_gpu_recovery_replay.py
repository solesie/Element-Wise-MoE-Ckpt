"""Bit-exact and allocation tests for rank-local GPU LFSY replay."""

from __future__ import annotations

import struct
from typing import Iterable, Sequence, Tuple

import pytest
import torch

from megatron.proposal.storage.expert_writer import (
    BLOCK_PREFIX_BYTES,
    HEADER_BLOCK_BYTES,
    MAGIC_LFCL,
    MAGIC_LFSY,
    RECORD_BYTES,
    _pack_header,
    read_expert_state_from_blobs,
)
from megatron.proposal.storage.gpu_replay import (
    ReplayTarget,
    prepare_lfcl_restore,
    prepare_lfsy_replay,
    restore_lfcl_in_place,
    replay_lfsy_in_place,
)
from megatron.proposal.storage import recovery as recovery_module
from megatron.proposal.storage.proposal_io import SECTOR_BYTES


Record = Tuple[int, float, float, float]


def _round_sector(nbytes: int) -> int:
    return (
        (int(nbytes) + int(SECTOR_BYTES) - 1)
        // int(SECTOR_BYTES)
        * int(SECTOR_BYTES)
    )


def _u8_blob(parts: Iterable[bytes]) -> torch.Tensor:
    return torch.frombuffer(bytearray(b"".join(parts)), dtype=torch.uint8)


def _metadata(*, unique: bool = True):
    return {
        "total_numel": 12,
        "tensors": [
            {"name": "w0", "shape": [5], "numel": 5},
            {"name": "w1", "shape": [7], "numel": 7},
        ],
        "unique_idx_per_block": bool(unique),
    }


def _lfcl_blob(
    *, metadata=None, step: int = 4,
    layout: str = "param_m_v_concat_fp32",
) -> torch.Tensor:
    meta = {
        **(_metadata() if metadata is None else dict(metadata)),
        "step": int(step),
        "layout": str(layout),
    }
    values = []
    cursor = 1.0
    for entry in meta["tensors"]:
        n = int(entry["numel"])
        values.extend(cursor + i for i in range(n))
        cursor += 10.0
        values.extend(cursor + i for i in range(n))
        cursor += 10.0
        values.extend(cursor + i for i in range(n))
        cursor += 10.0
    payload = torch.tensor(values, dtype=torch.float32).view(torch.uint8)
    logical = int(HEADER_BLOCK_BYTES) + int(payload.numel())
    padded = _round_sector(logical)
    blob = torch.zeros(padded, dtype=torch.uint8)
    header = torch.frombuffer(bytearray(_pack_header(MAGIC_LFCL, meta)), dtype=torch.uint8)
    blob[: int(header.numel())].copy_(header)
    blob[int(HEADER_BLOCK_BYTES):logical].copy_(payload)
    return blob


def _pinned_copy(blob: torch.Tensor) -> torch.Tensor:
    out = torch.empty(int(blob.numel()), dtype=torch.uint8, pin_memory=True)
    out.copy_(blob)
    return out


def _lfsy_blob(
    blocks: Sequence[Tuple[int, Sequence[Record]]],
    *,
    unique: bool = True,
) -> torch.Tensor:
    parts = [_pack_header(MAGIC_LFSY, _metadata(unique=unique))]
    for step, records in blocks:
        block = bytearray(struct.pack("<II", len(records), int(step)))
        for idx, p, m, v in records:
            block.extend(struct.pack("<Ifff", int(idx), float(p), float(m), float(v)))
        block.extend(bytes(_round_sector(len(block)) - len(block)))
        parts.append(bytes(block))
    return _u8_blob(parts)


def _cpu_dense(payload, field: str) -> torch.Tensor:
    return torch.cat([
        payload[f"w0:{field}"].reshape(-1),
        payload[f"w1:{field}"].reshape(-1),
    ])


def test_prepare_lfcl_restore_validates_layout_size_step_and_schema():
    lfcl = _lfcl_blob()
    plan = prepare_lfcl_restore(
        lfcl,
        expected_metadata=_metadata(),
        t_fault=4,
        source="lfcl-plan-test",
    )
    assert plan is not None
    assert plan.entry_numels == (5, 7)
    assert plan.total_numel == 12
    assert plan.base_step == 4

    with pytest.raises(RuntimeError, match="outside recovery range"):
        prepare_lfcl_restore(
            _lfcl_blob(step=5),
            expected_metadata=_metadata(),
            t_fault=4,
            source="future-lfcl",
        )
    with pytest.raises(ValueError, match="unsupported LFCL payload layout"):
        prepare_lfcl_restore(
            _lfcl_blob(layout="unknown"),
            expected_metadata=_metadata(),
            t_fault=4,
            source="bad-layout",
        )
    with pytest.raises(ValueError, match="byte size mismatch"):
        prepare_lfcl_restore(
            lfcl[:-4],
            expected_metadata=_metadata(),
            t_fault=4,
            source="truncated-lfcl",
        )
    wrong_schema = _metadata()
    wrong_schema["tensors"] = [dict(x) for x in wrong_schema["tensors"]]
    wrong_schema["tensors"][0]["name"] = "other"
    with pytest.raises(ValueError, match="recovery schema mismatch"):
        prepare_lfcl_restore(
            lfcl,
            expected_metadata=wrong_schema,
            t_fault=4,
            source="schema-mismatch",
        )


def test_prepare_lfcl_restore_builds_zero_seed_from_lfsy_schema():
    plan = prepare_lfcl_restore(
        None,
        expected_metadata=_metadata(),
        t_fault=4,
        source="zero-seed",
    )
    assert plan is not None
    assert plan.blob is None
    assert plan.entry_numels == (5, 7)
    assert plan.total_numel == 12
    assert plan.base_step == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_replay_matches_cpu_reference_bitwise_across_staging_splits():
    lfcl = _pinned_copy(_lfcl_blob())
    lfsy = _lfsy_blob([
        (5, [(0, 10.0, 1.0, 0.1), (4, 14.0, 1.4, 0.14),
             (5, 15.0, 1.5, 0.15), (11, 21.0, 2.1, 0.21)]),
        (6, [(4, 64.0, 6.4, 0.64), (7, 67.0, 6.7, 0.67)]),
        # Invalid indices are skipped by both the CPU reference and GPU path.
        (7, [(0, 70.0, 7.0, 0.7), (99, 99.0, 9.9, 0.99)]),
    ])
    cpu = read_expert_state_from_blobs(
        lfcl, lfsy, layer=0, expert=0, source="gpu-replay-test",
    )
    plan = prepare_lfsy_replay(
        lfsy, t_fault=7, source="gpu-replay-test",
    )
    assert plan is not None

    p_cpu = _cpu_dense(cpu, "param")
    m_cpu = _cpu_dense(cpu, "exp_avg")
    v_cpu = _cpu_dense(cpu, "exp_avg_sq")
    p = torch.zeros(12, dtype=torch.float32, device="cuda")
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    targets = [
        ReplayTarget(0, p[:5], m[:5], v[:5]),
        ReplayTarget(5, p[5:], m[5:], v[5:]),
    ]
    lfcl_plan = prepare_lfcl_restore(
        lfcl,
        expected_metadata=plan.metadata,
        t_fault=7,
        source="gpu-replay-test",
    )
    assert lfcl_plan is not None
    restore_lfcl_in_place(lfcl_plan, targets)
    # Two records force every four-record block across staging boundaries.
    staging = torch.empty(2 * int(RECORD_BYTES), dtype=torch.uint8, device="cuda")
    stats = replay_lfsy_in_place(
        plan,
        targets,
        staging,
    )
    torch.cuda.synchronize()

    assert stats.blocks == 3
    assert stats.records == 8
    assert torch.equal(p.cpu().view(torch.int32), p_cpu.view(torch.int32))
    assert torch.equal(m.cpu().view(torch.int32), m_cpu.view(torch.int32))
    assert torch.equal(v.cpu().view(torch.int32), v_cpu.view(torch.int32))


def test_gpu_replay_rejects_future_step_before_mutation():
    lfsy = _lfsy_blob([(8, [(1, 1.0, 2.0, 3.0)])])
    with pytest.raises(RuntimeError, match="newer than requested fault step"):
        prepare_lfsy_replay(lfsy, t_fault=7, source="future-log")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_replay_requires_explicit_unique_block_contract():
    lfsy = _lfsy_blob([(1, [(1, 1.0, 2.0, 3.0)])], unique=False)
    plan = prepare_lfsy_replay(lfsy, t_fault=1, source="legacy-log")
    assert plan is not None and not plan.unique_idx_per_block
    target = torch.zeros(12, dtype=torch.float32, device="cuda")
    staging = torch.empty(4096, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="unique_idx_per_block"):
        replay_lfsy_in_place(
            plan,
            [ReplayTarget(0, target, target.clone(), target.clone())],
            staging,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_replay_reuses_fixed_workspace_without_allocator_growth():
    lfcl = _pinned_copy(_lfcl_blob(step=0))
    lfsy = _lfsy_blob([
        (1, [(1, 1.0, 2.0, 3.0), (3, 4.0, 5.0, 6.0)]),
        (2, [(1, 7.0, 8.0, 9.0)]),
    ])
    plan = prepare_lfsy_replay(lfsy, t_fault=2, source="allocation-test")
    assert plan is not None
    p = torch.zeros(12, dtype=torch.float32, device="cuda")
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    staging = torch.empty(4096, dtype=torch.uint8, device="cuda")
    targets = [ReplayTarget(0, p, m, v)]
    lfcl_plan = prepare_lfcl_restore(
        lfcl,
        expected_metadata=plan.metadata,
        t_fault=2,
        source="allocation-test",
    )
    assert lfcl_plan is not None

    restore_lfcl_in_place(lfcl_plan, targets)
    replay_lfsy_in_place(plan, targets, staging)
    torch.cuda.synchronize()
    ptr = int(staging.data_ptr())
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    stats_before = torch.cuda.memory_stats()
    for _ in range(20):
        restore_lfcl_in_place(lfcl_plan, targets)
        replay_lfsy_in_place(plan, targets, staging)
    torch.cuda.synchronize()
    stats_after = torch.cuda.memory_stats()

    assert int(staging.data_ptr()) == ptr
    assert int(torch.cuda.memory_allocated()) == allocated
    assert int(torch.cuda.memory_reserved()) == reserved
    for key in (
        "allocation.all.allocated",
        "allocation.all.freed",
        "num_device_alloc",
        "num_device_free",
    ):
        assert int(stats_after.get(key, 0)) == int(stats_before.get(key, 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_replay_rejects_non_fp32_destinations():
    lfsy = _lfsy_blob([(1, [(1, 1.0, 2.0, 3.0)])])
    plan = prepare_lfsy_replay(lfsy, t_fault=1, source="dtype-test")
    assert plan is not None
    p = torch.zeros(12, dtype=torch.bfloat16, device="cuda")
    m = torch.zeros(12, dtype=torch.float32, device="cuda")
    v = torch.zeros_like(m)
    staging = torch.empty(4096, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="FP32"):
        replay_lfsy_in_place(
            plan,
            [ReplayTarget(0, p, m, v)],
            staging,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero1_replay_targets_use_only_header_owned_fragments(monkeypatch):
    class Mapper:
        def __init__(self):
            self.p = [torch.full((16,), -1.0, device="cuda") for _ in range(2)]
            self.m = [torch.full((16,), -2.0, device="cuda") for _ in range(2)]
            self.v = [torch.full((16,), -3.0, device="cuda") for _ in range(2)]

        def zero12_partition_view(
            self, group_idx, local_offset, numel, *, state_key=None,
        ):
            table = self.p if state_key is None else self.m if state_key == "exp_avg" else self.v
            return table[int(group_idx)].narrow(0, int(local_offset), int(numel))

    mapper = Mapper()
    monkeypatch.setattr(recovery_module, "get_mapper", lambda: mapper)
    lfsy = _lfsy_blob([
        (1, [(0, 10.0, 20.0, 30.0), (4, 14.0, 24.0, 34.0),
             (5, 15.0, 25.0, 35.0), (11, 21.0, 31.0, 41.0)]),
    ])
    plan = prepare_lfsy_replay(lfsy, t_fault=1, source="zero1-target-test")
    assert plan is not None
    plan.metadata["payload_kind"] = "zero12_expert_partition"
    plan.metadata["fragments"] = [
        {
            "index": 0, "fragment_name": "w0", "name": "w0",
            "group_idx": 0, "local_offset": 3, "param_offset": 0,
            "numel": 5, "shape": [5],
        },
        {
            "index": 1, "fragment_name": "w1", "name": "w1",
            "group_idx": 1, "local_offset": 4, "param_offset": 0,
            "numel": 7, "shape": [7],
        },
    ]
    targets = recovery_module._build_gpu_replay_targets(plan, {}, object())
    lfcl = _pinned_copy(_lfcl_blob(metadata=plan.metadata, step=0))
    lfcl_plan = prepare_lfcl_restore(
        lfcl,
        expected_metadata=plan.metadata,
        t_fault=1,
        source="zero1-target-test",
    )
    assert lfcl_plan is not None
    restore_lfcl_in_place(lfcl_plan, targets)
    staging = torch.empty(4096, dtype=torch.uint8, device="cuda")
    replay_lfsy_in_place(plan, targets, staging)
    torch.cuda.synchronize()

    assert mapper.p[0][2].item() == -1.0
    assert mapper.p[0][3].item() == 10.0
    assert mapper.p[0][7].item() == 14.0
    assert mapper.p[0][8].item() == -1.0
    assert mapper.p[0][4].item() == 2.0
    assert mapper.p[1][3].item() == -1.0
    assert mapper.p[1][4].item() == 15.0
    assert mapper.p[1][10].item() == 21.0
    assert mapper.p[1][11].item() == -1.0
    assert mapper.m[0][3].item() == 20.0
    assert mapper.v[1][10].item() == 41.0
