"""Lossy-record emission kernels.

This module owns the Stage-2 segment emission path. Production emits directly
from the shared score predicate into per-expert capacity blobs with a device
cursor cap; pre-materialized selection masks are not a supported writer path.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .record_kernels import (
    LossyRecordWriteInputs,
    launch_build_segment_entry_table,
    launch_direct_records_for_segment_entry_workqueue,
    launch_stamp_direct_expert_prefixes,
)


@dataclass
class DirectEntryGroupPlan:
    bf16_group_idx: int
    max_group_count: int
    entry_elem_start: torch.Tensor
    entry_age_elem_start: torch.Tensor
    entry_group_start: torch.Tensor
    entry_group_count: torch.Tensor
    entry_t_off: torch.Tensor
    entry_expert_idx: torch.Tensor
    entry_record_byte_off: torch.Tensor
    segment_entry_indices: Optional[torch.Tensor] = None
    segment_entry_counts: Optional[torch.Tensor] = None
    segment_entry_max_group_count: Optional[torch.Tensor] = None
    segment_entry_overflow: Optional[torch.Tensor] = None
    segment_entry_counts_host: Optional[torch.Tensor] = None
    segment_entry_max_group_count_host: Optional[torch.Tensor] = None
    segment_table_capacity: int = 0
    active_entry_count: Optional[int] = None


@dataclass(frozen=True)
class DirectSegmentEmissionPlan:
    expert_idx: torch.Tensor
    entries_by_group: Tuple[DirectEntryGroupPlan, ...]
    descriptor_host_refs: Tuple[torch.Tensor, ...] = ()
    segment_id: int = -1
    segment_bytes: int = 0


def build_direct_all_entries_emission_plan(
    *,
    n_experts: int,
    entry_specs: Sequence[object],
    lossy_state: LossyRecordWriteInputs,
    device: torch.device,
    stream: Optional[torch.cuda.Stream] = None,
    n_segments: int = 0,
) -> DirectSegmentEmissionPlan:
    """Build static entry descriptors; layout offsets are supplied by GPU tables."""

    _validate_direct_lossy_state(lossy_state)
    group_size = int(lossy_state.group_size)
    entry_buckets: Dict[Tuple[int, int], Dict[str, List[int]]] = {}
    for ent in entry_specs:
        expert_idx = int(getattr(ent, "expert_index", -1))
        if expert_idx < 0:
            continue
        g_idx = int(getattr(ent, "bf16_group_idx", -1))
        if g_idx < 0 or g_idx >= len(lossy_state.bf16_groups_mvph):
            raise RuntimeError(f"direct entry references invalid BF16 group: {g_idx}")
        mvph = lossy_state.bf16_groups_mvph[g_idx]
        if mvph is None:
            raise RuntimeError(f"direct entry references missing BF16 group: {g_idx}")
        m_flat, _v_flat, _p_flat, _master_off_elem = mvph
        elem_start = int(getattr(ent, "entry_within_off", 0))
        age_elem_start = int(getattr(ent, "entry_age_off", elem_start))
        group_count = int(getattr(ent, "name_group_count", 0))
        if group_count <= 0:
            continue
        if elem_start % group_size != 0:
            raise RuntimeError(
                f"direct entry is not group-aligned: elem_start={elem_start}, "
                f"group_size={group_size}"
            )
        group_start = elem_start // group_size
        elem_n = group_count * group_size
        if elem_start < 0 or elem_start + elem_n > int(m_flat.numel()):
            raise RuntimeError("direct entry exceeds BF16 group span")
        bucket = entry_buckets.setdefault(
            (g_idx, _entry_size_bucket(group_count)),
            {
                "elem_start": [],
                "age_elem_start": [],
                "group_start": [],
                "group_count": [],
                "t_off": [],
                "expert_idx": [],
                "record_byte": [],
            },
        )
        bucket["elem_start"].append(elem_start)
        bucket["age_elem_start"].append(age_elem_start)
        bucket["group_start"].append(group_start)
        bucket["group_count"].append(group_count)
        bucket["t_off"].append(int(getattr(ent, "t_off", 0)))
        bucket["expert_idx"].append(expert_idx)
        bucket["record_byte"].append(0)

    arena = _DescriptorArena(device=device, stream=stream)
    group_plans: List[DirectEntryGroupPlan] = []
    pending_groups = []
    for (g_idx, _size_bucket), bucket in sorted(entry_buckets.items()):
        pending_groups.append((
            int(g_idx),
            max(int(x) for x in bucket["group_count"]),
            arena.add(bucket["elem_start"], dtype=torch.int64),
            arena.add(bucket["age_elem_start"], dtype=torch.int64),
            arena.add(bucket["group_start"], dtype=torch.int64),
            arena.add(bucket["group_count"], dtype=torch.int32),
            arena.add(bucket["t_off"], dtype=torch.int64),
            arena.add(bucket["expert_idx"], dtype=torch.int32),
            arena.add(bucket["record_byte"], dtype=torch.int64),
        ))
    arena.materialize()
    for (
        g_idx, max_group_count, elem_start, age_elem_start, group_start,
        group_count, t_off, expert_idx_group, record_byte,
    ) in pending_groups:
        table_capacity = int(group_count.length)
        segment_entry_indices = None
        segment_entry_counts = None
        segment_entry_max_group_count = None
        segment_entry_overflow = None
        if int(n_segments) > 0 and table_capacity > 0:
            segment_entry_indices = torch.empty(
                (int(n_segments), table_capacity),
                dtype=torch.int32,
                device=device,
            )
            segment_entry_counts = torch.empty(
                int(n_segments),
                dtype=torch.int32,
                device=device,
            )
            segment_entry_max_group_count = torch.empty(
                int(n_segments),
                dtype=torch.int32,
                device=device,
            )
            segment_entry_overflow = torch.empty(
                int(n_segments),
                dtype=torch.int32,
                device=device,
            )
            segment_entry_counts_host = torch.empty(
                int(n_segments),
                dtype=torch.int32,
                pin_memory=True,
            )
            segment_entry_max_group_count_host = torch.empty(
                int(n_segments),
                dtype=torch.int32,
                pin_memory=True,
            )
        else:
            segment_entry_counts_host = None
            segment_entry_max_group_count_host = None
        group_plans.append(DirectEntryGroupPlan(
            bf16_group_idx=g_idx,
            max_group_count=max_group_count,
            entry_elem_start=arena.tensor(elem_start),
            entry_age_elem_start=arena.tensor(age_elem_start),
            entry_group_start=arena.tensor(group_start),
            entry_group_count=arena.tensor(group_count),
            entry_t_off=arena.tensor(t_off),
            entry_expert_idx=arena.tensor(expert_idx_group),
            entry_record_byte_off=arena.tensor(record_byte),
            segment_entry_indices=segment_entry_indices,
            segment_entry_counts=segment_entry_counts,
            segment_entry_max_group_count=segment_entry_max_group_count,
            segment_entry_overflow=segment_entry_overflow,
            segment_entry_counts_host=segment_entry_counts_host,
            segment_entry_max_group_count_host=segment_entry_max_group_count_host,
            segment_table_capacity=table_capacity,
        ))
    expert_idx = torch.arange(max(0, int(n_experts)), dtype=torch.int32, device=device)
    return DirectSegmentEmissionPlan(
        expert_idx=expert_idx,
        entries_by_group=tuple(group_plans),
        descriptor_host_refs=arena.host_refs,
    )


def _entry_size_bucket(group_count: int) -> int:
    """Bound rectangular workqueue waste without falling back to per-fragment launches."""

    count = max(1, int(group_count))
    return int(count - 1).bit_length()

def with_segment_identity(
    plan: DirectSegmentEmissionPlan,
    *,
    segment_id: int,
    segment_bytes: int,
) -> DirectSegmentEmissionPlan:
    return DirectSegmentEmissionPlan(
        expert_idx=plan.expert_idx,
        entries_by_group=plan.entries_by_group,
        descriptor_host_refs=plan.descriptor_host_refs,
        segment_id=int(segment_id),
        segment_bytes=int(segment_bytes),
    )


def prepare_segment_entry_tables(
    *,
    segment_plan: DirectSegmentEmissionPlan,
    lossy_state: LossyRecordWriteInputs,
    segment_bytes: int,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    _validate_direct_lossy_state(lossy_state)
    active_groups = [
        group_plan
        for group_plan in segment_plan.entries_by_group
        if group_plan.segment_entry_indices is not None
    ]
    if not active_groups:
        return
    context = (
        torch.cuda.stream(stream)
        if stream is not None
        else torch.cuda.stream(torch.cuda.current_stream())
    )
    with context:
        for group_plan in active_groups:
            assert group_plan.segment_entry_counts is not None
            assert group_plan.segment_entry_max_group_count is not None
            assert group_plan.segment_entry_overflow is not None
            group_plan.segment_entry_counts.zero_()
            group_plan.segment_entry_max_group_count.zero_()
            group_plan.segment_entry_overflow.zero_()
            launch_build_segment_entry_table(
                entry_expert_idx=group_plan.entry_expert_idx,
                entry_group_count=group_plan.entry_group_count,
                aos_layout_offsets=lossy_state.aos_layout_offsets,
                aos_layout_capacities=lossy_state.aos_layout_capacities,
                segment_entry_indices=group_plan.segment_entry_indices,
                segment_entry_counts=group_plan.segment_entry_counts,
                segment_entry_max_group_count=group_plan.segment_entry_max_group_count,
                segment_entry_overflow=group_plan.segment_entry_overflow,
                segment_bytes=int(segment_bytes),
            )
    if os.environ.get("PROPOSAL_SEGMENT_TABLE_PROFILE", "") in ("1", "true", "TRUE"):
        print(
            "[proposal] " + describe_segment_entry_tables(segment_plan),
            flush=True,
        )


def describe_segment_entry_tables(
    segment_plan: DirectSegmentEmissionPlan,
) -> str:
    """Profile-only summary; callers must keep it out of the hot path."""

    group_summaries = []
    total_active = None
    total_overflow = None
    for group_plan in segment_plan.entries_by_group:
        if group_plan.segment_entry_counts is None:
            continue
        counts = group_plan.segment_entry_counts.detach().cpu()
        over = (
            group_plan.segment_entry_overflow.detach().cpu()
            if group_plan.segment_entry_overflow is not None
            else torch.zeros_like(counts)
        )
        if total_active is None:
            total_active = counts.to(torch.int64)
            total_overflow = over.to(torch.int64)
        else:
            total_active = total_active + counts.to(torch.int64)
            total_overflow = total_overflow + over.to(torch.int64)
        nonzero = int((counts > 0).sum().item())
        max_count = int(counts.max().item()) if counts.numel() else 0
        group_summaries.append(
            f"g{int(group_plan.bf16_group_idx)}:nz={nonzero},max={max_count},cap={int(group_plan.segment_table_capacity)}"
        )
    if total_active is None:
        return "segment_table=none"
    active_list = ",".join(str(int(x)) for x in total_active.tolist())
    overflow_list = ",".join(str(int(x)) for x in total_overflow.tolist())
    return (
        f"segment_table_active=[{active_list}] "
        f"segment_table_overflow=[{overflow_list}] "
        f"groups={' | '.join(group_summaries)}"
    )


def emit_segment_direct_from_plan(
    *,
    staging_u8: torch.Tensor,
    seg_byte_off: int,
    seg_n_bytes: int,
    segment_plan: DirectSegmentEmissionPlan,
    lossy_state: LossyRecordWriteInputs,
    step: int,
) -> None:
    """Fill one AOS segment with score-based direct emission and cursor caps."""

    import time

    profile = os.environ.get("PROPOSAL_DIRECT_EMIT_PROFILE", "") in (
        "1", "true", "TRUE",
    )
    t0 = time.perf_counter() if profile else 0.0
    _validate_segment_buffer(
        staging_u8=staging_u8,
        seg_byte_off=int(seg_byte_off),
        seg_n_bytes=int(seg_n_bytes),
    )
    _validate_direct_lossy_state(lossy_state)
    if int(seg_byte_off) == 0:
        header_words = int(getattr(lossy_state, "aos_layout_header_words", 0))
        header_device = getattr(lossy_state, "aos_layout_header_device", None)
        if header_words > 0 and header_device is not None:
            header_nbytes = header_words * 8
            if header_nbytes > int(seg_n_bytes):
                raise RuntimeError("AOS layout header does not fit in segment 0")
            staging_u8.narrow(0, 0, header_nbytes).view(torch.int64).copy_(
                header_device.narrow(0, 0, header_words),
                non_blocking=True,
            )
    t_after_header = time.perf_counter() if profile else 0.0
    seg_i32 = staging_u8.narrow(0, 0, int(seg_n_bytes)).view(torch.int32)

    hp = lossy_state.adam_hparams
    assert hp is not None
    total_entries = 0
    total_max_groups = 0
    total_table_capacity = 0
    for group_plan in segment_plan.entries_by_group:
        total_entries += int(group_plan.entry_group_count.numel())
        total_max_groups += int(group_plan.max_group_count)
        total_table_capacity += int(group_plan.segment_table_capacity)
        m_flat, v_flat, p_flat, master_off_elem = (
            lossy_state.bf16_groups_mvph[group_plan.bf16_group_idx]
        )
        theta_cp_view = None
        theta_cp_groups = getattr(lossy_state, "theta_cp_groups", None)
        if (
            theta_cp_groups is not None
            and 0 <= int(group_plan.bf16_group_idx) < len(theta_cp_groups)
        ):
            theta_cp_view = theta_cp_groups[group_plan.bf16_group_idx]
        if (
            group_plan.segment_entry_indices is not None
            and group_plan.segment_entry_counts is not None
            and group_plan.segment_entry_max_group_count is not None
            and int(segment_plan.segment_id) >= 0
        ):
            launch_direct_records_for_segment_entry_workqueue(
                m_view=m_flat,
                v_view=v_flat,
                p_view=p_flat,
                h_lut_values=lossy_state.h_lut_values,
                h_lut_log_v_min=lossy_state.h_lut_log_v_min,
                h_lut_inv_log_v_width=lossy_state.h_lut_inv_log_v_width,
                entry_elem_start=group_plan.entry_elem_start,
                entry_age_elem_start=group_plan.entry_age_elem_start,
                entry_group_start=group_plan.entry_group_start,
                entry_group_count=group_plan.entry_group_count,
                entry_t_off=group_plan.entry_t_off,
                entry_expert_idx=group_plan.entry_expert_idx,
                aos_layout_offsets=lossy_state.aos_layout_offsets,
                aos_layout_capacities=lossy_state.aos_layout_capacities,
                expert_record_cursor=lossy_state.direct_record_cursor,
                expert_overflow_count=lossy_state.direct_overflow_count,
                total_th=lossy_state.total_th,
                adam_scalar_device=lossy_state.adam_scalar_device,
                record_out_i32=seg_i32,
                segment_entry_indices=group_plan.segment_entry_indices,
                segment_entry_counts=group_plan.segment_entry_counts,
                segment_entry_max_group_count=group_plan.segment_entry_max_group_count,
                segment_id=int(segment_plan.segment_id),
                segment_byte_off=int(seg_byte_off),
                segment_n_bytes=int(seg_n_bytes),
                master_base_elem=int(master_off_elem),
                static_max_group_count=int(group_plan.max_group_count),
                group_size=int(lossy_state.group_size),
                lr_over_bc1=float(hp.lr_over_bc1),
                sqrt_bc2_inv=float(hp.sqrt_bc2_inv),
                eps=float(hp.eps),
                random_mode=bool(lossy_state.random_mode),
                random_salt=int(lossy_state.random_salt),
                age2bit_words=lossy_state.age2bit_words,
                ehat_geom_lut=lossy_state.ehat_geom_lut,
                theta_cp_view=theta_cp_view,
                use_e_hat_proxy=bool(
                    getattr(lossy_state, "use_e_hat_proxy", True)
                ),
                score_policy_id=int(lossy_state.score_policy_id),
                age_force_period=int(getattr(lossy_state, "age_force_period", 128)),
                lfcl_fired_epoch=getattr(lossy_state, "lfcl_fired_epoch", None),
                step_device=lossy_state.step_device,
            )
            continue
        raise RuntimeError(
            "direct emission requires GPU segment-entry tables; "
            "entry-block fallback is not part of the final path"
        )
    t_after_emit = time.perf_counter() if profile else 0.0

    launch_stamp_direct_expert_prefixes(
        record_out_i32=seg_i32,
        expert_idx=segment_plan.expert_idx,
        aos_layout_offsets=lossy_state.aos_layout_offsets,
        aos_layout_capacities=lossy_state.aos_layout_capacities,
        expert_record_cursor=lossy_state.direct_record_cursor,
        expert_overflow_count=lossy_state.direct_overflow_count,
        step_device=lossy_state.step_device,
        step=int(step),
        segment_byte_off=int(seg_byte_off),
        segment_n_bytes=int(seg_n_bytes),
        n_experts=int(segment_plan.expert_idx.numel()),
    )
    if profile:
        t_after_prefix = time.perf_counter()
        print(
            f"[proposal] direct_emit_profile: "
            f"seg_off={int(seg_byte_off)} seg_n={int(seg_n_bytes)} "
            f"groups={len(segment_plan.entries_by_group)} "
            f"entries={int(total_entries)} max_groups_sum={int(total_max_groups)} "
            f"table_cap={int(total_table_capacity)} "
            f"prefix_experts={int(segment_plan.expert_idx.numel())} "
            f"header={((t_after_header - t0) * 1000.0):.1f}ms "
            f"emit_enqueue={((t_after_emit - t_after_header) * 1000.0):.1f}ms "
            f"prefix_enqueue={((t_after_prefix - t_after_emit) * 1000.0):.1f}ms",
            flush=True,
        )


def _validate_segment_buffer(
    *,
    staging_u8: torch.Tensor,
    seg_byte_off: int,
    seg_n_bytes: int,
) -> None:
    if int(seg_byte_off) & 3:
        raise ValueError(
            f"emit_segment_direct_from_plan: seg_byte_off must be 4-byte aligned, "
            f"got {int(seg_byte_off)}"
        )
    if int(seg_n_bytes) & 3:
        raise ValueError(
            f"emit_segment_direct_from_plan: seg_n_bytes must be 4-byte aligned, "
            f"got {int(seg_n_bytes)}"
        )
    if staging_u8.numel() < int(seg_n_bytes):
        raise ValueError(
            f"emit_segment_direct_from_plan: staging too small "
            f"({int(staging_u8.numel())} < {int(seg_n_bytes)})"
        )
    if staging_u8.dtype != torch.uint8 or not staging_u8.is_contiguous():
        raise ValueError(
            "emit_segment_direct_from_plan: staging must be contiguous uint8"
        )


def _validate_direct_lossy_state(lossy_state: LossyRecordWriteInputs) -> None:
    required = {
        "h_lut_values": lossy_state.h_lut_values,
        "h_lut_log_v_min": lossy_state.h_lut_log_v_min,
        "h_lut_inv_log_v_width": lossy_state.h_lut_inv_log_v_width,
        "total_th": lossy_state.total_th,
        "adam_hparams": lossy_state.adam_hparams,
        "direct_record_cursor": lossy_state.direct_record_cursor,
        "direct_overflow_count": lossy_state.direct_overflow_count,
        "aos_layout_offsets": lossy_state.aos_layout_offsets,
        "aos_layout_capacities": lossy_state.aos_layout_capacities,
        "aos_layout_header_device": lossy_state.aos_layout_header_device,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            "direct emission missing required inputs: " + ", ".join(missing)
        )


@dataclass(frozen=True)
class _DescriptorToken:
    dtype: torch.dtype
    start: int
    length: int


class _DescriptorArena:
    """Bulk-build static descriptor tensors with one pinned H2D copy per dtype."""

    def __init__(
        self,
        *,
        device: torch.device,
        stream: Optional[torch.cuda.Stream],
    ) -> None:
        self._device = device
        self._stream = stream
        self._i64_values: List[int] = []
        self._i32_values: List[int] = []
        self._device_by_dtype: Dict[torch.dtype, torch.Tensor] = {}
        self._host_refs: List[torch.Tensor] = []

    @property
    def host_refs(self) -> Tuple[torch.Tensor, ...]:
        return tuple(self._host_refs)

    def add(self, values: Sequence[int], *, dtype: torch.dtype) -> _DescriptorToken:
        if dtype == torch.int64:
            start = len(self._i64_values)
            self._i64_values.extend(int(v) for v in values)
        elif dtype == torch.int32:
            start = len(self._i32_values)
            self._i32_values.extend(int(v) for v in values)
        else:
            raise TypeError(f"unsupported descriptor dtype: {dtype}")
        return _DescriptorToken(dtype=dtype, start=start, length=len(values))

    def materialize(self) -> None:
        self._materialize_dtype(torch.int64, self._i64_values)
        self._materialize_dtype(torch.int32, self._i32_values)

    def tensor(self, token: _DescriptorToken) -> torch.Tensor:
        if token.length == 0:
            return torch.empty(0, dtype=token.dtype, device=self._device)
        base = self._device_by_dtype[token.dtype]
        return base.narrow(0, int(token.start), int(token.length))

    def _materialize_dtype(
        self,
        dtype: torch.dtype,
        values: Sequence[int],
    ) -> None:
        if not values:
            return
        host = torch.empty(len(values), dtype=dtype, pin_memory=True)
        host.copy_(torch.as_tensor(values, dtype=dtype))
        dev = torch.empty(len(values), dtype=dtype, device=self._device)
        if self._stream is None:
            dev.copy_(host, non_blocking=True)
        else:
            with torch.cuda.stream(self._stream):
                dev.copy_(host, non_blocking=True)
        self._host_refs.append(host)
        self._device_by_dtype[dtype] = dev


__all__ = [
    "DirectSegmentEmissionPlan",
    "build_direct_all_entries_emission_plan",
    "emit_segment_direct_from_plan",
    "prepare_segment_entry_tables",
    "describe_segment_entry_tables",
    "with_segment_identity",
]
