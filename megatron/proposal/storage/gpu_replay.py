"""Rank-local GPU restore for expert LFCL baselines and LFSY logs.

This module owns only expert recovery-format validation and in-place GPU state
application.  It does not discover checkpoints, decide rank ownership, perform
collectives, or synchronize low-precision model weights; those remain recovery
orchestration responsibilities.

Blocks are applied in physical append order.  Direct emission visits each
expert-local element at most once per block, so parallel writes inside one
block are disjoint.  Serial block submission therefore has the same
latest-write-wins result as the CPU reference's reverse scan and seen bitmap.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only in CPU-only installs.
    triton = None
    tl = None
    _HAS_TRITON = False

from .expert_writer import (
    BLOCK_PREFIX_BYTES,
    FORMAT_VERSION,
    HEADER_BLOCK_BYTES,
    MAGIC_LFCL,
    MAGIC_LFSY,
    RECORD_BYTES,
    _unpack_header,
)
from .proposal_io import SECTOR_BYTES


@dataclass(frozen=True)
class LfsyBlock:
    """One validated physical append block in an LFSY object."""

    record_byte_offset: int
    record_count: int
    step: int


@dataclass(frozen=True)
class LfsyReplayPlan:
    """Immutable CPU-side description retained until H2D replay completes."""

    blob: torch.Tensor
    metadata: Dict[str, Any]
    total_numel: int
    blocks: Tuple[LfsyBlock, ...]
    max_step: int
    unique_idx_per_block: bool


@dataclass(frozen=True)
class ReplayTarget:
    """A contiguous interval in the expert-local dense recovery layout."""

    element_start: int
    param: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor

    @property
    def numel(self) -> int:
        return int(self.param.numel())


@dataclass(frozen=True)
class ReplayStats:
    blocks: int
    records: int
    bytes_staged: int


@dataclass(frozen=True)
class LfclRestorePlan:
    """Validated LFCL source for direct restore into resident FP32 state."""

    blob: Optional[torch.Tensor]
    metadata: Dict[str, Any]
    entry_numels: Tuple[int, ...]
    total_numel: int
    base_step: int


def _recovery_schema(metadata: Dict[str, Any], source: str):
    """Return one validated, directly comparable recovery schema."""

    entries = []
    names = set()
    total = 0
    for index, raw in enumerate(list(metadata.get("tensors", []) or [])):
        if not isinstance(raw, dict):
            raise ValueError(f"{source}: invalid tensor schema entry {index}")
        name = str(raw.get("name", ""))
        shape = tuple(int(dim) for dim in raw.get("shape", []) or [])
        numel = int(raw.get("numel", -1))
        shape_numel = 1
        for dim in shape:
            shape_numel *= dim
        if (
            not name or name in names or numel < 0
            or any(dim < 0 for dim in shape) or shape_numel != numel
        ):
            raise ValueError(f"{source}: invalid tensor schema for {name!r}")
        names.add(name)
        entries.append((name, shape, numel))
        total += numel
    if int(metadata.get("total_numel", -1)) != total:
        raise ValueError(f"{source}: total_numel does not match tensor schema")

    kind = str(metadata.get("payload_kind", ""))
    if kind not in ("", "zero12_expert_partition"):
        raise ValueError(f"{source}: unsupported payload kind {kind!r}")
    fragment_schema = ()
    if kind == "zero12_expert_partition":
        fragments = list(metadata.get("fragments", []) or [])
        if len(fragments) != len(entries):
            raise ValueError(f"{source}: ZeRO fragment/tensor count mismatch")
        normalized = []
        for entry, fragment in zip(entries, fragments):
            if not isinstance(fragment, dict):
                raise ValueError(f"{source}: invalid ZeRO fragment metadata")
            try:
                normalized_fragment = (
                    int(fragment["index"]), str(fragment["fragment_name"]),
                    str(fragment["name"]), int(fragment["group_idx"]),
                    int(fragment["local_offset"]), int(fragment["param_offset"]),
                    int(fragment["numel"]),
                    tuple(int(dim) for dim in fragment["shape"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}: invalid ZeRO fragment metadata") from exc
            if normalized_fragment[1] != entry[0] or normalized_fragment[6] != entry[2]:
                raise ValueError(f"{source}: ZeRO fragment does not match tensor")
            normalized.append(normalized_fragment)
        fragment_schema = tuple(normalized)
    return tuple(entries), int(total), kind, fragment_schema


def prepare_lfcl_restore(
    blob: Optional[torch.Tensor],
    *,
    expected_metadata: Optional[Dict[str, Any]],
    t_fault: int,
    source: str,
) -> Optional[LfclRestorePlan]:
    """Validate an LFCL before any resident state is mutated.

    A missing LFCL with a valid LFSY schema represents the pre-baseline zero
    seed used by the reference recovery path.
    """

    if blob is None or int(blob.numel()) == 0:
        if expected_metadata is None:
            return None
        schema, total_numel, _kind, _fragments = _recovery_schema(
            expected_metadata, f"{source}:zero-seed",
        )
        return LfclRestorePlan(
            blob=None,
            metadata=dict(expected_metadata),
            entry_numels=tuple(int(entry[2]) for entry in schema),
            total_numel=int(total_numel),
            base_step=0,
        )
    if blob.device.type != "cpu" or blob.dtype != torch.uint8:
        raise ValueError(f"{source}: LFCL blob must be a CPU uint8 tensor")
    if not blob.is_contiguous():
        raise ValueError(f"{source}: LFCL blob must be contiguous")
    if torch.cuda.is_available() and not blob.is_pinned():
        raise ValueError(f"{source}: direct LFCL restore requires pinned input")
    if int(blob.numel()) < int(HEADER_BLOCK_BYTES):
        raise ValueError(f"{source}: truncated LFCL header")

    header = bytes(
        blob.narrow(0, 0, int(HEADER_BLOCK_BYTES)).numpy().tobytes()
    )
    magic, version, metadata = _unpack_header(header)
    if magic != MAGIC_LFCL:
        raise ValueError(f"{source}: expected LFCL, got {magic!r}")
    if int(version) != int(FORMAT_VERSION):
        raise ValueError(f"{source}: unsupported LFCL version {int(version)}")
    if str(metadata.get("layout", "")) != "param_m_v_concat_fp32":
        raise ValueError(f"{source}: unsupported LFCL payload layout")
    schema, total_numel, kind, fragments = _recovery_schema(
        metadata, f"{source}:LFCL",
    )
    if expected_metadata is not None:
        expected = _recovery_schema(expected_metadata, f"{source}:LFSY")
        if (schema, total_numel, kind, fragments) != expected:
            raise ValueError(f"{source}: LFCL/LFSY recovery schema mismatch")
    base_step = int(metadata.get("step", 0))
    if base_step < 0 or base_step > int(t_fault):
        raise RuntimeError(
            f"{source}: LFCL step {base_step} is outside recovery range "
            f"[0, {int(t_fault)}]"
        )

    logical_bytes = int(HEADER_BLOCK_BYTES) + int(total_numel) * 3 * 4
    padded_bytes = _round_up_sector(logical_bytes)
    if int(blob.numel()) not in (logical_bytes, padded_bytes):
        raise ValueError(
            f"{source}: LFCL byte size mismatch: got={int(blob.numel())} "
            f"expected={logical_bytes} or {padded_bytes}"
        )
    return LfclRestorePlan(
        blob=blob,
        metadata=dict(metadata),
        entry_numels=tuple(int(entry[2]) for entry in schema),
        total_numel=int(total_numel),
        base_step=int(base_step),
    )


@torch.no_grad()
def restore_lfcl_in_place(
    plan: LfclRestorePlan,
    targets: Iterable[ReplayTarget],
    *,
    stream: Optional[torch.cuda.Stream] = None,
) -> int:
    """Restore an LFCL directly into existing FP32 p/m/v destinations."""

    ordered_targets = _validate_targets(targets, total_numel=plan.total_numel)
    if len(ordered_targets) != len(plan.entry_numels):
        raise ValueError("LFCL target/schema entry count mismatch")
    for target, numel in zip(ordered_targets, plan.entry_numels):
        if int(target.numel) != int(numel):
            raise ValueError("LFCL target/schema entry size mismatch")
    if not ordered_targets:
        return int(plan.base_step)

    restore_stream = stream or torch.cuda.current_stream(
        ordered_targets[0].param.device,
    )
    source_fields = []
    if plan.blob is not None:
        if not plan.blob.is_pinned():
            raise ValueError("direct LFCL restore requires pinned input")
        payload_numel = int(plan.total_numel) * 3
        payload = plan.blob.narrow(
            0, int(HEADER_BLOCK_BYTES), payload_numel * 4,
        ).view(torch.float32)
        cursor = 0
        for numel in plan.entry_numels:
            n = int(numel)
            source_fields.append((
                payload.narrow(0, cursor, n),
                payload.narrow(0, cursor + n, n),
                payload.narrow(0, cursor + 2 * n, n),
            ))
            cursor += 3 * n
        if cursor != payload_numel:
            raise ValueError("LFCL payload cursor mismatch")

    with torch.cuda.stream(restore_stream):
        if plan.blob is None:
            for target in ordered_targets:
                target.param.zero_()
                target.exp_avg.zero_()
                target.exp_avg_sq.zero_()
        else:
            for target, sources in zip(ordered_targets, source_fields):
                target.param.reshape(-1).copy_(sources[0], non_blocking=True)
                target.exp_avg.reshape(-1).copy_(sources[1], non_blocking=True)
                target.exp_avg_sq.reshape(-1).copy_(sources[2], non_blocking=True)
    return int(plan.base_step)


def _round_up_sector(nbytes: int) -> int:
    return (
        (int(nbytes) + int(SECTOR_BYTES) - 1)
        // int(SECTOR_BYTES)
        * int(SECTOR_BYTES)
    )


def prepare_lfsy_replay(
    blob: Optional[torch.Tensor],
    *,
    t_fault: int,
    source: str,
) -> Optional[LfsyReplayPlan]:
    """Validate an LFSY object and describe its blocks without replaying it.

    Validation happens before any resident model/optimizer state is mutated.
    This preserves the existing latest-only FDP contract while avoiding the
    old behavior where a future log could be noticed only after application.
    """

    if blob is None or int(blob.numel()) == 0:
        return None
    if blob.device.type != "cpu" or blob.dtype != torch.uint8:
        raise ValueError(f"{source}: LFSY blob must be a CPU uint8 tensor")
    if not blob.is_contiguous():
        raise ValueError(f"{source}: LFSY blob must be contiguous")
    if int(blob.numel()) < int(HEADER_BLOCK_BYTES):
        raise ValueError(f"{source}: truncated LFSY header")

    header = bytes(
        blob.narrow(0, 0, int(HEADER_BLOCK_BYTES)).numpy().tobytes()
    )
    magic, _version, metadata = _unpack_header(header)
    if magic != MAGIC_LFSY:
        raise ValueError(f"{source}: expected LFSY, got {magic!r}")
    total_numel = int(metadata.get("total_numel", -1))
    if total_numel < 0:
        raise ValueError(f"{source}: invalid total_numel={total_numel}")

    raw = memoryview(blob.numpy())
    file_bytes = int(blob.numel())
    offset = int(HEADER_BLOCK_BYTES)
    blocks = []
    max_step = 0
    while offset + int(BLOCK_PREFIX_BYTES) <= file_bytes:
        count, step = struct.unpack_from("<II", raw, offset)
        if int(count) == 0:
            offset += int(SECTOR_BYTES)
            continue
        record_off = offset + int(BLOCK_PREFIX_BYTES)
        payload_bytes = int(count) * int(RECORD_BYTES)
        payload_end = record_off + payload_bytes
        if payload_end > file_bytes:
            raise ValueError(
                f"{source}: truncated LFSY block at byte {offset} "
                f"(count={int(count)})"
            )
        step_i = int(step)
        if step_i > int(t_fault):
            raise RuntimeError(
                f"{source}: LFSY step {step_i} is newer than requested "
                f"fault step {int(t_fault)}"
            )
        blocks.append(LfsyBlock(record_off, int(count), step_i))
        max_step = max(max_step, step_i)
        offset = _round_up_sector(payload_end)

    return LfsyReplayPlan(
        blob=blob,
        metadata=dict(metadata),
        total_numel=total_numel,
        blocks=tuple(blocks),
        max_step=int(max_step),
        unique_idx_per_block=bool(metadata.get("unique_idx_per_block", False)),
    )


if _HAS_TRITON:

    @triton.jit
    def _scatter_lfsy_interval_kernel(
        records_i32_ptr,
        param_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        n_records,
        element_start,
        element_end,
        BLOCK_SIZE: tl.constexpr,
    ):
        record = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        active = record < n_records
        word = record * 4
        idx = tl.load(records_i32_ptr + word, mask=active, other=0).to(tl.uint32)
        idx64 = idx.to(tl.int64)
        selected = active & (idx64 >= element_start) & (idx64 < element_end)
        local = idx64 - element_start
        p_bits = tl.load(records_i32_ptr + word + 1, mask=selected, other=0)
        m_bits = tl.load(records_i32_ptr + word + 2, mask=selected, other=0)
        v_bits = tl.load(records_i32_ptr + word + 3, mask=selected, other=0)
        tl.store(
            param_ptr + local,
            p_bits.to(tl.float32, bitcast=True),
            mask=selected,
        )
        tl.store(
            exp_avg_ptr + local,
            m_bits.to(tl.float32, bitcast=True),
            mask=selected,
        )
        tl.store(
            exp_avg_sq_ptr + local,
            v_bits.to(tl.float32, bitcast=True),
            mask=selected,
        )


def _validate_targets(
    targets: Iterable[ReplayTarget],
    *,
    total_numel: int,
) -> Tuple[ReplayTarget, ...]:
    ordered = tuple(targets)
    cursor = 0
    device = None
    for target in ordered:
        if int(target.element_start) != cursor:
            raise ValueError(
                "GPU replay targets must exactly cover expert-local layout: "
                f"expected start {cursor}, got {int(target.element_start)}"
            )
        tensors = (target.param, target.exp_avg, target.exp_avg_sq)
        if any(t.device.type != "cuda" for t in tensors):
            raise ValueError("GPU replay destinations must be CUDA tensors")
        if len({t.device for t in tensors}) != 1:
            raise ValueError("GPU replay destination fields must share one device")
        if device is None:
            device = tensors[0].device
        elif tensors[0].device != device:
            raise ValueError("GPU replay targets must share one device")
        if any(not t.is_contiguous() for t in tensors):
            raise ValueError("GPU replay destinations must be contiguous")
        if any(t.dtype != torch.float32 for t in tensors):
            raise ValueError("GPU replay destinations must be FP32 tensors")
        if any(int(t.numel()) != int(target.numel) for t in tensors):
            raise ValueError("GPU replay destination sizes do not match")
        cursor += int(target.numel)
    if cursor != int(total_numel):
        raise ValueError(
            "GPU replay target coverage mismatch: "
            f"covered={cursor}, expected={int(total_numel)}"
        )
    return ordered


def replay_lfsy_in_place(
    plan: LfsyReplayPlan,
    targets: Iterable[ReplayTarget],
    staging_u8: torch.Tensor,
    *,
    stream: Optional[torch.cuda.Stream] = None,
) -> ReplayStats:
    """Apply a validated LFSY plan into resident GPU state in append order.

    ``staging_u8`` is caller-owned fixed workspace.  The function never
    allocates a model- or expert-sized CUDA tensor and never performs a
    collective.  All commands use one stream, which is the only ordering
    requirement for latest-write-wins correctness.
    """

    if not _HAS_TRITON:
        raise RuntimeError("GPU LFSY replay requires Triton")
    if not bool(plan.unique_idx_per_block):
        raise ValueError(
            "GPU chronological replay requires unique_idx_per_block; "
            "use the CPU reference for legacy LFSY objects"
        )
    if staging_u8.device.type != "cuda" or staging_u8.dtype != torch.uint8:
        raise ValueError("GPU replay staging must be a CUDA uint8 tensor")
    if not staging_u8.is_contiguous():
        raise ValueError("GPU replay staging must be contiguous")
    records_per_stage = int(staging_u8.numel()) // int(RECORD_BYTES)
    if records_per_stage <= 0:
        raise ValueError("GPU replay staging is smaller than one LFSY record")
    ordered_targets = _validate_targets(
        targets,
        total_numel=int(plan.total_numel),
    )
    if not ordered_targets and int(plan.total_numel) != 0:
        raise ValueError("GPU replay has no resident destination targets")
    if ordered_targets and ordered_targets[0].param.device != staging_u8.device:
        raise ValueError("GPU replay staging and destinations must share a device")

    replay_stream = stream or torch.cuda.current_stream(staging_u8.device)
    records_total = 0
    bytes_total = 0
    with torch.cuda.stream(replay_stream):
        for block in plan.blocks:
            block_record = 0
            while block_record < int(block.record_count):
                n_records = min(
                    records_per_stage,
                    int(block.record_count) - block_record,
                )
                nbytes = n_records * int(RECORD_BYTES)
                source_off = (
                    int(block.record_byte_offset)
                    + block_record * int(RECORD_BYTES)
                )
                source = plan.blob.narrow(0, source_off, nbytes)
                staged = staging_u8.narrow(0, 0, nbytes)
                staged.copy_(source, non_blocking=bool(source.is_pinned()))
                records_i32 = staged.view(torch.int32)
                for target in ordered_targets:
                    start = int(target.element_start)
                    end = start + int(target.numel)
                    grid = (triton.cdiv(n_records, 256),)
                    _scatter_lfsy_interval_kernel[grid](
                        records_i32,
                        target.param.reshape(-1),
                        target.exp_avg.reshape(-1),
                        target.exp_avg_sq.reshape(-1),
                        n_records,
                        element_start=start,
                        element_end=end,
                        BLOCK_SIZE=256,
                        num_warps=4,
                    )
                block_record += n_records
                records_total += n_records
                bytes_total += nbytes

    return ReplayStats(
        blocks=len(plan.blocks),
        records=int(records_total),
        bytes_staged=int(bytes_total),
    )


__all__ = [
    "LfclRestorePlan",
    "LfsyBlock",
    "LfsyReplayPlan",
    "ReplayStats",
    "ReplayTarget",
    "prepare_lfcl_restore",
    "prepare_lfsy_replay",
    "restore_lfcl_in_place",
    "replay_lfsy_in_place",
]
