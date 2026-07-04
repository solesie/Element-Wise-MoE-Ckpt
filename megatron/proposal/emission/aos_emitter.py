"""GPU-side scatter / AOS interleave for expert lossy FC.

Replaces the CPU SoA→AoS interleave path (formerly ``aos_write`` in
proposal_io.cpp). Builds the on-disk lossy record stream directly on
GPU into the segment_d2h staging buffer, eliminating the ~2 GB
CPU memory traffic per iter that caused L3 cache pollution and main
thread launch cadence perturbation.

Triton kernels write through the segment staging buffer viewed as a
contiguous int32 array (record = 4 × i32 = 16 bytes):

  * ``lossy_records``  — batch all tname slices for one BF16 optimizer
                         group in the segment and directly select/fold/emit
                         from the shared score predicate.

Launch order per segment (driven by ``run_segment_producer`` below):

   for BF16 optimizer group with entries in segment:
        lossy_records       (1 launch, grid = compact entry blocks)

The caller then records a ``cuda.Event`` on the worker stream and the
caller-supplied segment_d2h drives the D2H to pinned_aos.

VRAM: no global AOS buffer. Only the fixed segment_d2h staging
double-buffer plus compact entry descriptor tables built per segment.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .segment_d2h import DEFAULT_SEGMENT_BYTES
from .record_kernels import LossyRecordWriteInputs
from .aos_layout import (
    GpuAosLayoutState,
    PinnedAosLayoutMirror,
)


# On-disk constants (must match expert_state.RECORD_BYTES /
# BLOCK_PREFIX_BYTES / _RECORD_DTYPE and _csrc/proposal_io.cpp's
# AoSRecord struct).
RECORD_BYTES: int = 16            # uint32 idx + 3 × float32
RECORD_WORDS: int = 4             # in 4-byte words
PREFIX_BYTES: int = 8             # int32 count + int32 step
SECTOR_BYTES: int = 4096          # blob is padded up to this
SECTOR_WORDS: int = SECTOR_BYTES // 4


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Host-side descriptors
# ---------------------------------------------------------------------------

@dataclass
class ExpertSpec:
    """One expert's contribution to a single lossy-FC iter.

    ``aos_byte_off`` is the byte offset of THIS expert's blob within
    the global per-iter AOS layout (= cumulative size of prior
    experts' padded blobs).
    ``blob_padded_bytes`` is the 4 KiB-rounded blob size (= 8-byte prefix
    + 16-byte records, padded up).
    """
    aos_byte_off: int
    blob_padded_bytes: int
    n_records: int
    step: int
    prefix_on_gpu: bool = False
    expert_index: int = -1


@dataclass
class EntrySpec:
    """One (expert, tname) entry's contribution.

    ``bf16_group_idx`` is the index into the caller-supplied list of
    per-BF16-group GPU SoA tensors.
    ``src_off`` is the BF16-group-local element offset of this entry's
    first GROUP record in the SoA. Element records are read directly from
    m/v/p flat tensors by the element-record writer.
    ``aos_byte_off`` is the byte offset of this entry's FIRST RECORD
    (= element-record start) in the global AOS layout.
    ``n_records`` is total records this entry contributes to AOS
    (= elem_n_records + group_n_records).
    ``t_off`` is the per-tname offset added to BF16-group-local idx to
    form the on-disk idx (= ``tname_base_offset_in_expert``).
    ``elem_n_records`` is the count of element records that precede
    group records within this entry.
    """
    bf16_group_idx: int
    src_off: int
    n_records: int
    aos_byte_off: int
    t_off: int
    elem_n_records: int = 0
    name_idx: int = -1
    name_group_count: int = 0
    expert_index: int = -1
    expert_record_capacity: int = 0
    expert_record_byte_off: int = 0
    # Entry's start position in BF16-group-local ELEMENT coords.
    # Used by the group-record descriptor path to
    # convert bf16-local idx → entry-local idx → disk_idx:
    #   disk_idx = bf16_local - entry_within_off + t_off.
    # Default 0 means src_idx is already entry-local.
    entry_within_off: int = 0
    # Entry's start position in proposal compact ELEMENT coords
    # (age2bit/theta_cp address space).
    entry_age_off: int = 0


@dataclass
class _DirectAosPlan:
    expert_specs: List[ExpertSpec]
    entry_specs: List[EntrySpec]
    total_aos: int
    n_entries: int
    segments: List[Tuple[int, int]]
    segment_plans: dict[Tuple[int, int], object]
    layout_state: object
    all_entries_plan: object


class AosEmitter:
    """Owner of the per-iter GPU→pinned AOS staging path. Held by an
    ``ExpertState`` instance and reused across
    every iter.

    Caller pre-allocates ``pinned_aos`` once (sized by
    ``ExpertState.__init__`` for the local worst-case AOS payload) and
    passes it here. The pipeline NEVER grows or re-allocates it; overflow
    raises ``RuntimeError``.

    ``run(prepared, compute_event, gpu_refs, lossy_state)`` performs:
      * device-side ``compute_stream.wait_event(compute_event)`` when a handoff
        event is provided
      * builds the direct-emission AOS plan once, then reuses its
        device-resident segment descriptors every iter
      * drives ``SegmentD2H.run`` with a Triton-kernel producer
      * returns the final D2H event to ``ExpertState``, then clears
        ``gpu_refs``

    Returns ``(expert_specs, total_aos_bytes, d2h_done_event)``. The
    expert_launcher thread calls this method; ``ExpertState`` uses the
    returned event for the optimizer fence and expert-writer lease.

    ``prepared`` is a list of ``(ent, total_n, filtered)`` where
    ``filtered`` is a list of ``(tname, t_off, bf16_group_idx, src_off,
    n_records)`` per sub-entry — ordered tname-major within an expert.
    """

    def __init__(self,
                 *,
                 pinned_aos: torch.Tensor,
                 target_record_capacity: Optional[int] = None,
                 segment_bytes: int = DEFAULT_SEGMENT_BYTES,
                 num_segment_buffers: int = 2) -> None:
        if not pinned_aos.is_pinned():
            raise ValueError("AosEmitter: pinned_aos must be pinned")
        if pinned_aos.dtype != torch.uint8:
            raise ValueError("AosEmitter: pinned_aos must be uint8")
        self.pinned_aos = pinned_aos
        self._target_record_capacity = (
            None if target_record_capacity is None else max(0, int(target_record_capacity))
        )
        self._segment_bytes = int(segment_bytes)
        self._num_segment_buffers = int(num_segment_buffers)
        self._segment_d2h: Optional[object] = None  # SegmentD2H
        self._direct_plan: Optional[_DirectAosPlan] = None
        self._direct_plan_capacity_signature: Optional[Tuple[int, ...]] = None

    def _ensure_segment_d2h(self, device: torch.device) -> None:
        if self._segment_d2h is not None:
            return
        from .segment_d2h import SegmentD2H
        from ..runtime import get_compute_stream, get_d2h_stream
        d2h_stream = get_d2h_stream()
        compute_stream = get_compute_stream()
        if compute_stream is None:
            compute_stream = torch.cuda.current_stream(device)
        self._segment_d2h = SegmentD2H(
            device=device.index if device.index is not None else 0,
            compute_stream=compute_stream,
            d2h_stream=d2h_stream,
            segment_bytes=self._segment_bytes,
            num_buffers=self._num_segment_buffers,
        )

    def warmup(self, device: torch.device) -> None:
        """Allocate all GPU/event resources before the first checkpoint iter."""

        self._ensure_segment_d2h(device)

    def _build_direct_plan(
        self,
        *,
        step: int,
        prepared,
        lossy_state: LossyRecordWriteInputs,
        max_total_bytes: int,
        stream: "torch.cuda.Stream",
    ) -> _DirectAosPlan:
        from .direct_emission import (
            build_direct_all_entries_emission_plan,
            with_segment_identity,
        )

        name_to_winoff = {
            nm: int(lossy_state.within_off_per_name[i])
            for i, nm in enumerate(lossy_state.name_list)
        }
        age_offs = (
            lossy_state.age_off_per_name
            if len(lossy_state.age_off_per_name) == len(lossy_state.name_list)
            else lossy_state.within_off_per_name
        )
        name_to_ageoff = {
            nm: int(age_offs[i])
            for i, nm in enumerate(lossy_state.name_list)
        }
        name_to_idx = {nm: i for i, nm in enumerate(lossy_state.name_list)}
        expert_specs: List[ExpertSpec] = []
        entry_specs: List[EntrySpec] = []
        max_records_by_expert: List[int] = []
        for e_idx, (_ent_obj, total_n, filtered) in enumerate(prepared):
            max_records_by_expert.append(max(0, int(total_n)))
            expert_specs.append(ExpertSpec(
                aos_byte_off=-1,
                blob_padded_bytes=0,
                n_records=max(0, int(total_n)),
                step=int(step),
                prefix_on_gpu=True,
                expert_index=int(e_idx),
            ))
            for (tname_f, _t_off, _bf16_g, _src_off, _n_recs) in filtered:
                name_idx = int(name_to_idx.get(tname_f, -1))
                within_off = int(name_to_winoff.get(tname_f, 0))
                age_off = int(name_to_ageoff.get(tname_f, within_off))
                if 0 <= name_idx < len(lossy_state.group_count_per_name):
                    group_count = int(lossy_state.group_count_per_name[name_idx])
                else:
                    group_count = 0
                entry_specs.append(EntrySpec(
                    bf16_group_idx=int(_bf16_g),
                    src_off=int(_src_off),
                    n_records=int(_n_recs),
                    aos_byte_off=0,
                    t_off=int(_t_off),
                    elem_n_records=0,
                    name_idx=name_idx,
                    name_group_count=group_count,
                    expert_index=int(e_idx),
                    expert_record_capacity=0,
                    expert_record_byte_off=0,
                    entry_within_off=within_off,
                    entry_age_off=age_off,
                ))

        device = _lossy_state_device(lossy_state)
        layout_state = GpuAosLayoutState(
            max_records_by_expert=max_records_by_expert,
            target_record_capacity=(
                int(self._target_record_capacity)
                if self._target_record_capacity is not None
                else int(sum(max_records_by_expert))
            ),
            max_total_bytes=int(max_total_bytes),
            segment_bytes=int(self._segment_d2h.segment_bytes),
            device=device,
        )
        lossy_state.aos_layout_offsets = layout_state.offset_device
        lossy_state.aos_layout_capacities = layout_state.capacity_device
        lossy_state.aos_layout_header_device = layout_state.header_device
        lossy_state.aos_layout_header_words = int(layout_state.header_words)
        segments = _fixed_segments_for_bytes(
            int(max_total_bytes),
            int(self._segment_d2h.segment_bytes),
        )
        all_entries_plan = build_direct_all_entries_emission_plan(
            n_experts=len(expert_specs),
            entry_specs=entry_specs,
            lossy_state=lossy_state,
            device=device,
            stream=stream,
            n_segments=len(segments),
        )
        segment_plans = {
            (int(seg_off), int(seg_n)): with_segment_identity(
                all_entries_plan,
                segment_id=seg_id,
                segment_bytes=int(self._segment_d2h.segment_bytes),
            )
            for seg_id, (seg_off, seg_n) in enumerate(segments)
        }
        return _DirectAosPlan(
            expert_specs=expert_specs,
            entry_specs=entry_specs,
            total_aos=int(max_total_bytes),
            n_entries=len(entry_specs),
            segments=[(int(o), int(n)) for o, n in segments],
            segment_plans=segment_plans,
            layout_state=layout_state,
            all_entries_plan=all_entries_plan,
        )

    def run(
        self,
        *,
        step: int,
        prepared,
        compute_event,
        gpu_refs,
        lossy_state: Optional[LossyRecordWriteInputs] = None,
        pinned_aos: Optional[torch.Tensor] = None,
        layout_mirror: Optional[PinnedAosLayoutMirror] = None,
        copy_to_host: bool = True,
    ):
        """Drive one iter of the GPU AOS pipeline. Returns
        ``(expert_specs, total_aos_bytes, d2h_done_event)``.
        ``pinned_aos`` (pre-allocated by the caller in ``__init__``) holds the full AOS
        layout after this call — caller does the per-expert
        finalisation (``::write`` or DRAM scatter).

        ``lossy_state`` supplies live p/m/v, H LUT, τ, Age2Bit state, score LUT
        tables, and cursor-cap tensors. Selection and emit happen inside
        direct emission.
        """
        import time

        profile = os.environ.get("PROPOSAL_AOS_PROFILE", "") in (
            "1", "true", "TRUE",
        )
        profile_every = int(os.environ.get("PROPOSAL_AOS_PROFILE_EVERY", "10") or "10")
        profile_this = profile and (int(step) % max(1, profile_every) == 0)
        t_entry = time.perf_counter() if profile_this else 0.0
        # Handoff ordering is device-side. Host synchronization here would
        # serialize the final path and defeat the proposal stream contract.
        t_after_compute_wait = time.perf_counter() if profile_this else 0.0

        direct_emit = _lossy_state_can_direct_emit(lossy_state)
        if not direct_emit:
            raise RuntimeError("AosEmitter final path requires direct emission")
        target_pinned_aos = pinned_aos if pinned_aos is not None else self.pinned_aos
        device = _lossy_state_device(lossy_state)
        self._ensure_segment_d2h(device)
        capacity_signature = (
            tuple(int(total_n) for _ent, total_n, _filtered in prepared)
        )
        if (
            self._direct_plan is None
            or capacity_signature != self._direct_plan_capacity_signature
        ):
            self._direct_plan = self._build_direct_plan(
                step=int(step),
                prepared=prepared,
                lossy_state=lossy_state,
                max_total_bytes=int(target_pinned_aos.numel()),
                stream=self._segment_d2h._compute_stream,
            )
            self._direct_plan_capacity_signature = capacity_signature
        direct_plan = self._direct_plan
        lossy_state.aos_layout_offsets = direct_plan.layout_state.offset_device
        lossy_state.aos_layout_capacities = direct_plan.layout_state.capacity_device
        lossy_state.aos_layout_header_device = direct_plan.layout_state.header_device
        lossy_state.aos_layout_header_words = int(direct_plan.layout_state.header_words)
        expert_specs = direct_plan.expert_specs
        for spec in expert_specs:
            spec.step = int(step)
        total_aos = int(direct_plan.total_aos)
        t_after_layout = time.perf_counter() if profile_this else 0.0
        if total_aos == 0:
            if gpu_refs is not None:
                gpu_refs.clear()
            return expert_specs, 0, None

        if total_aos > int(target_pinned_aos.numel()):
            raise RuntimeError(
                f"AosEmitter: pinned_aos too small "
                f"({int(target_pinned_aos.numel())} < {total_aos}); the "
                f"pre-allocated pinned AOS slab was exceeded for this iter. "
                f"Check cutoff ratio, capacity margin, and model schema."
            )

        if compute_event is not None:
            self._segment_d2h._compute_stream.wait_event(compute_event)

        cursor = lossy_state.direct_record_cursor
        overflow = lossy_state.direct_overflow_count
        if cursor is None or overflow is None:
            raise RuntimeError("direct emission requires cursor/overflow tensors")
        if cursor.numel() < len(expert_specs) or overflow.numel() < len(expert_specs):
            raise RuntimeError(
                "direct emission cursor/overflow tensors are smaller than "
                f"the prepared expert count ({cursor.numel()}, {overflow.numel()} "
                f"< {len(expert_specs)})"
            )
        enable_sample_feedback = bool(
            getattr(lossy_state, "enable_direct_probe_sample_feedback", True)
        )
        direct_plan.layout_state.launch_update(
            cursor=cursor,
            stream=self._segment_d2h._compute_stream,
            sample_blocks_current=(
                getattr(lossy_state, "sample_blocks_current", None)
                if enable_sample_feedback
                else None
            ),
            probe_count_mismatch_streak=(
                getattr(lossy_state, "probe_count_mismatch_streak", None)
                if enable_sample_feedback
                else None
            ),
            sample_grow_hold=(
                getattr(lossy_state, "sample_grow_hold", None)
                if enable_sample_feedback
                else None
            ),
            ci_shortfall_streak=(
                getattr(lossy_state, "ci_shortfall_streak", None)
                if enable_sample_feedback
                else None
            ),
            probe_diagnostics=(
                getattr(lossy_state, "probe_diagnostics", None)
                if enable_sample_feedback
                else None
            ),
            probe_expected_selected_count=(
                getattr(lossy_state, "probe_expected_selected_count", None)
                if enable_sample_feedback
                else None
            ),
            capacity_demand_hint=getattr(lossy_state, "capacity_demand_hint", None),
            total_elems_global=int(getattr(lossy_state, "total_elems_global", 0)),
            allowed_probe_count_delta=int(
                getattr(lossy_state, "allowed_probe_count_delta", 0)
            ),
            sample_blocks_min=int(getattr(lossy_state, "sample_blocks_min", 1)),
            sample_blocks_cap=int(getattr(lossy_state, "sample_blocks_cap", 1)),
            probe_mismatch_grow_after=int(
                getattr(lossy_state, "probe_count_mismatch_grow_after", 3)
            ),
        )
        segments = direct_plan.segments
        from .direct_emission import prepare_segment_entry_tables
        first_segment_plan = direct_plan.segment_plans[segments[0]] if segments else None
        if first_segment_plan is not None:
            prepare_segment_entry_tables(
                segment_plan=first_segment_plan,
                lossy_state=lossy_state,
                segment_bytes=int(self._segment_d2h.segment_bytes),
                stream=self._segment_d2h._compute_stream,
            )
        with torch.cuda.stream(self._segment_d2h._compute_stream):
            cursor.narrow(0, 0, len(expert_specs)).zero_()
            overflow.narrow(0, 0, len(expert_specs)).zero_()

        t_after_segments = time.perf_counter() if profile_this else 0.0

        def _producer(seg_view: torch.Tensor, byte_off: int, n: int) -> None:
            run_segment_producer(
                staging_u8=seg_view,
                seg_byte_off=byte_off,
                seg_n_bytes=n,
                segment_plan=direct_plan.segment_plans[(int(byte_off), int(n))],
                lossy_state=lossy_state,
                step=int(step),
            )

        d2h_done_event, _producer_done_event = self._segment_d2h.run(
            _producer,
            segments,
            target_pinned_aos,
            copy_to_host=bool(copy_to_host),
        )
        t_after_run = time.perf_counter() if profile_this else 0.0
        t_after_final_sync = time.perf_counter() if profile_this else 0.0
        if gpu_refs is not None:
            gpu_refs.clear()
        if profile_this:
            ms = lambda a, b: (b - a) * 1000.0
            print(
                f"[proposal] aos_profile @iter{int(step)}: "
                f"experts={len(expert_specs)} entries={int(direct_plan.n_entries)} "
                f"segments={len(segments)} bytes={int(total_aos)} "
                f"writer=direct "
                f"compute_wait={ms(t_entry, t_after_compute_wait):.1f}ms "
                f"layout={ms(t_after_compute_wait, t_after_layout):.1f} "
                f"segments_build={ms(t_after_layout, t_after_segments):.1f} "
                f"enqueue={ms(t_after_segments, t_after_run):.1f} "
                f"final_sync={ms(t_after_run, t_after_final_sync):.1f}",
                flush=True,
            )
        return expert_specs, total_aos, d2h_done_event


def _fixed_segments_for_bytes(
    total_bytes: int,
    segment_bytes: int,
) -> List[Tuple[int, int]]:
    total = max(0, int(total_bytes))
    seg = max(1, int(segment_bytes))
    out: List[Tuple[int, int]] = []
    off = 0
    while off < total:
        n = min(seg, total - off)
        out.append((off, n))
        off += n
    return out


def bucket_expert_specs_by_segment(
    expert_specs: Sequence["ExpertSpec"],
    segments: Sequence[Tuple[int, int]],
) -> dict[Tuple[int, int], List["ExpertSpec"]]:
    buckets: dict[Tuple[int, int], List["ExpertSpec"]] = {
        (int(off), int(n)): [] for off, n in segments
    }
    if not expert_specs or not segments:
        return buckets
    seg_i = 0
    for spec in expert_specs:
        spec_lo = int(spec.aos_byte_off)
        spec_hi = spec_lo + int(spec.blob_padded_bytes)
        while seg_i < len(segments):
            seg_lo, seg_n = segments[seg_i]
            if spec_lo < int(seg_lo) + int(seg_n):
                break
            seg_i += 1
        if seg_i >= len(segments):
            raise RuntimeError(
                f"expert blob at byte range [{spec_lo}, {spec_hi}) has no AOS segment"
            )
        seg_lo, seg_n = segments[seg_i]
        seg_hi = int(seg_lo) + int(seg_n)
        if spec_lo < int(seg_lo) or spec_hi > seg_hi:
            raise RuntimeError(
                "expert blob unexpectedly crosses an AOS segment; "
                "segments must be cut at expert blob boundaries"
            )
        buckets[(int(seg_lo), int(seg_n))].append(spec)
    return buckets


def _lossy_state_can_direct_emit(lossy_state: Optional[LossyRecordWriteInputs]) -> bool:
    if lossy_state is None:
        return False
    return bool(
        getattr(lossy_state, "direct_emission_enabled", False)
        and getattr(lossy_state, "h_lut_values", None) is not None
        and getattr(lossy_state, "h_lut_log_v_min", None) is not None
        and getattr(lossy_state, "h_lut_inv_log_v_width", None) is not None
        and getattr(lossy_state, "total_th", None) is not None
        and getattr(lossy_state, "adam_hparams", None) is not None
        and getattr(lossy_state, "direct_record_cursor", None) is not None
        and getattr(lossy_state, "direct_overflow_count", None) is not None
    )


def _lossy_state_device(lossy_state: Optional[LossyRecordWriteInputs]) -> torch.device:
    if lossy_state is not None:
        for mvph in lossy_state.bf16_groups_mvph:
            if mvph is not None:
                return mvph[0].device
    raise RuntimeError(
        "AosEmitter requires LossyRecordWriteInputs with at least "
        "one live p/m/v BF16-group view"
    )


# ---------------------------------------------------------------------------
# Per-segment producer
# ---------------------------------------------------------------------------

def run_segment_producer(
    *,
    staging_u8: torch.Tensor,
    seg_byte_off: int,
    seg_n_bytes: int,
    segment_plan: object,
    lossy_state: Optional[LossyRecordWriteInputs] = None,
    step: int,
) -> None:
    """Fill ``staging_u8[0:seg_n_bytes]`` with the slice of the global
    AOS layout that lives at byte range ``[seg_byte_off, seg_byte_off + seg_n_bytes)``.

    Caller must invoke this on ``compute_stream`` (the stream the
    segment_d2h producer runs on). All kernel launches inherit
    the current stream.

    ``staging_u8`` must be a contiguous uint8 GPU tensor of length
    >= seg_n_bytes. Live BF16 optimizer views are supplied through
    ``lossy_state.bf16_groups_mvph``; the caller is responsible for keeping
    them alive until kernel completion.

    Byte-alignment requirement: ``seg_byte_off`` must be a multiple of
    4 (segment_d2h uses MiB-aligned segments → trivially aligned).
    Within an expert's blob, prefix and records are 4-byte aligned by
    construction (PREFIX_BYTES = 8, RECORD_BYTES = 16). Expert blob
    boundaries are 4 KiB-aligned.
    """
    if lossy_state is None:
        raise RuntimeError("AosEmitter requires LossyRecordWriteInputs")
    from .direct_emission import emit_segment_direct_from_plan

    if _lossy_state_can_direct_emit(lossy_state):
        emit_segment_direct_from_plan(
            staging_u8=staging_u8,
            seg_byte_off=int(seg_byte_off),
            seg_n_bytes=int(seg_n_bytes),
            segment_plan=segment_plan,
            lossy_state=lossy_state,
            step=int(step),
        )
        return

    raise RuntimeError("AosEmitter final path requires direct emission")
