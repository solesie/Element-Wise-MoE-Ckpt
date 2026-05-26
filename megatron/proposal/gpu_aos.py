"""GPU-side scatter / AOS interleave for expert lossy FC.

Replaces the CPU SoA→AoS interleave path (formerly ``aos_write`` in
proposal_io.cpp). Builds the on-disk lossy record stream directly on
GPU into the segment_pipeline staging buffer, eliminating the ~2 GB
CPU memory traffic per iter that caused L3 cache pollution and main
thread launch cadence perturbation (CLAUDE.md diagnosis +355 ms).

Three Triton kernels, all writing through the segment staging buffer
viewed as a contiguous int32 array (record = 5 × i32 = 20 bytes):

  * ``zero_segment``   — pre-zero the staging segment (clears per-expert
                         4 KiB padding regions and any gaps the other
                         kernels do not touch).
  * ``prefix_write``   — write the 8-byte (count, step) prefix at the
                         start of each expert's blob (positions that
                         fall inside the current segment only).
  * ``aos_records``    — interleave one BF16 group's SoA (idx i64,
                         p / m / v f32) into AoS records (20 B each)
                         at the correct positions in staging.

Launch order per segment (driven by ``run_segment_producer`` below):

   zero_segment             (1 launch)
   prefix_write             (1 launch, grid = experts in segment)
   for grp in bf16_groups_with_entries_in_segment:
        aos_records         (1 launch, grid = entries from that group)

The caller then records a ``cuda.Event`` on the worker stream and the
caller-supplied segment_pipeline drives the D2H to pinned_aos.

VRAM: no global AOS buffer. Only the segment_pipeline staging (256 MB
double-buffered) plus the small descriptor tables built per iter
(~100 KB total — pointers, offsets, counts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from .segment_pipeline import DEFAULT_SEGMENT_BYTES


# On-disk constants (must match expert_state.RECORD_BYTES /
# BLOCK_PREFIX_BYTES / _RECORD_DTYPE and _csrc/proposal_io.cpp's
# AoSRecord struct).
RECORD_BYTES: int = 20            # int64 idx + 3 × float32
RECORD_WORDS: int = 5             # in 4-byte words
PREFIX_BYTES: int = 8             # int32 count + int32 step
PREFIX_WORDS: int = 2
SECTOR_BYTES: int = 4096          # blob is padded up to this
SECTOR_WORDS: int = SECTOR_BYTES // 4


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _zero_segment_kernel(
    out_i32_ptr,            # *i32 — segment staging, viewed as i32
    n_words,                # number of i32 words to clear
    BLOCK: tl.constexpr,
):
    """Write 0 to ``out_i32_ptr[0 : n_words]``. Grid = ceil(n_words / BLOCK)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_words
    z = tl.zeros((BLOCK,), dtype=tl.int32)
    tl.store(out_i32_ptr + offs, z, mask=mask)


@triton.jit
def _prefix_write_kernel(
    out_i32_ptr,            # *i32 — segment staging viewed as i32
    pos_word_ptr,           # *i32 — per-expert i32 word position in segment
    count_ptr,              # *i32 — per-expert record count
    step_ptr,               # *i32 — per-expert iter step
    n_experts,
):
    """Each program writes one expert's 2-word (count, step) prefix at
    ``out[pos_word], out[pos_word + 1]``. Positions that fall outside
    the segment (negative or >= seg_words) MUST be filtered out on
    host before launch."""
    eid = tl.program_id(0)
    if eid >= n_experts:
        return
    pos = tl.load(pos_word_ptr + eid)
    cnt = tl.load(count_ptr + eid)
    stp = tl.load(step_ptr + eid)
    tl.store(out_i32_ptr + pos + 0, cnt)
    tl.store(out_i32_ptr + pos + 1, stp)



# ---------------------------------------------------------------------------
# Host-side descriptors
# ---------------------------------------------------------------------------

@dataclass
class ExpertSpec:
    """One expert's contribution to a single lossy-FC iter.

    ``aos_byte_off`` is the byte offset of THIS expert's blob within
    the global per-iter AOS layout (= cumulative size of prior
    experts' padded blobs).
    ``blob_padded_bytes`` is the 4 KiB-rounded blob size (= 8 + n_rec
    × 20 padded up).
    """
    aos_byte_off: int
    blob_padded_bytes: int
    n_records: int
    step: int


@dataclass
class EntrySpec:
    """One (expert, tname) entry's contribution.

    ``bf16_group_idx`` is the index into the caller-supplied list of
    per-BF16-group GPU SoA tensors.
    ``src_off`` is the BF16-group-local element offset of this entry's
    first GROUP record in the SoA (outliers, if any, are NOT in the SoA
    under Option A replace mode — Pass B reads m/v/p flat directly).
    ``aos_byte_off`` is the byte offset of this entry's FIRST RECORD
    (= outlier record start in Option A replace mode; = combined
    record start otherwise) in the global AOS layout.
    ``n_records`` is total records this entry contributes to AOS
    (= outlier_n_records + group_n_records).
    ``t_off`` is the per-tname offset added to BF16-group-local idx to
    form the on-disk idx (= ``tname_base_offset_in_expert``).
    ``outlier_n_records`` (Option A only) is the count of outlier
    records that precede group records within this entry. Pass B writes
    outliers to ``[aos_byte_off, aos_byte_off + outlier_n_records × 20)``;
    aos_records writes groups to ``[aos_byte_off + outlier_n_records ×
    20, aos_byte_off + n_records × 20)``. Default 0 (legacy mode: all
    n_records are group/combined, no outlier split).
    """
    bf16_group_idx: int
    src_off: int
    n_records: int
    aos_byte_off: int
    t_off: int
    outlier_n_records: int = 0
    # Entry's start position in BF16-group-local ELEMENT coords.
    # Used ONLY by ``_aos_records_kernel_v2`` (Option A v2 path) to
    # convert bf16-local idx → entry-local idx → disk_idx:
    #   disk_idx = bf16_local - entry_within_off + t_off.
    # Default 0 → equivalent to legacy v1 semantics (src_idx already
    # entry-local).
    entry_within_off: int = 0


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


# ---------------------------------------------------------------------------
# Option A (replace mode) state — drives Pass B inside the segment producer
# ---------------------------------------------------------------------------

@dataclass
class LossyDispatchState:
    """Bundle passed to ``run_segment_producer`` to enable Option A
    replace mode. When provided, the producer:
      * Computes per-segment ``name_outlier_byte_off`` (sentinel -1 for
        out-of-segment names) from ``name_outlier_count_host`` and the
        global AOS layout (``expert_specs`` + ``entry_specs``).
      * Calls ``launch_aos_write_for_subrange`` per BF16 group → writes
        outlier records to staging at byte offsets relative to seg start.
      * Calls ``_aos_records_kernel`` for GROUP records with byte
        offsets SHIFTED by ``n_outlier × RECORD_BYTES`` so outlier and
        group records share each entry's blob without overlap.

    Required fields:
      * ``bf16_groups_mvph``: list of (m_flat, v_flat, p_flat,
        dg_buf, master_off_elem) per BF16 group. h_g_buf dropped —
        H is computed inline in kernels via bsearch in v_q/h_q.
      * ``v_q``, ``h_q``: global FP32 quantile tables [N_Q] for inline
        H lookup. Persistent (set by caller in init).
      * ``n_q``: quantile table size.
      * ``ofs_elem_full``: global element-space offsets (i64, N+1).
      * ``name_outlier_count_host``: per-name outlier count (int list,
        from Pass A's ``_outlier_name_count.cpu()``). Length = N_names.
      * ``name_outlier_count_dev``: same on GPU.
      * ``name_t_off``: per-name on-disk t_off (i64, N_names) — same as
        existing aos_records' t_off.
      * ``name_slot``: per-name atomic slot counter (i64, N_names),
        zero'd by caller before each call.
      * ``prev_th_t``, ``group_th_t``: GPU scalar thresholds.
      * ``lr_over_bc1``, ``sqrt_bc2_inv``, ``eps``: Adam scalars.
      * ``g_blk``: group quantisation block size.
      * ``aos_byte_off_per_name``: per-name AOS byte offset (=entry
        start, global coord) — host int list, length N_names. Built by
        caller's plan_layout walk. -1 for names with 0 records this iter.
    """
    bf16_groups_mvph: List[Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, int,
    ]]
    v_q: torch.Tensor
    h_q: torch.Tensor
    n_q: int
    ofs_elem_full: torch.Tensor
    name_list: List[str]
    name_outlier_count_host: List[int]
    name_outlier_count_dev: torch.Tensor
    name_t_off: torch.Tensor
    name_slot: torch.Tensor
    prev_th_t: torch.Tensor
    group_th_t: torch.Tensor
    lr_over_bc1: float
    sqrt_bc2_inv: float
    eps: float
    g_blk: int
    aos_byte_off_per_name: List[int]
    # Per-name BF16-group-local element-space start offset (=
    # ``(offsets_cpu[name] - bf16_master_off) * g_blk``). Plumbed to
    # ``plan_layout`` via ``within_off_per_entry`` and stored in
    # ``EntrySpec.entry_within_off`` so the v2 kernel can derive
    # entry-local idx from bf16-local idx. Length = N_names (tracker
    # authoritative ordering). Default empty list → v2 disabled.
    within_off_per_name: List[int] = field(default_factory=list)
    # (A.4 v3) Per-name top_start = start position in top_idx_sorted
    # of this name's picked groups (= seg_bounds_cpu[i]). Length =
    # N_names + 1 (extra sentinel = total picked groups). Used by the
    # v3 kernel to compute elem_idx on the fly — no K_total scratch
    # alloc. Default empty list → v3 disabled (v2 path remains).
    top_start_per_name: List[int] = field(default_factory=list)
    # (A.4 v3) top_idx_sorted full tensor reference (persistent
    # K_groups-bound buffer in tracker). v3 kernel reads from this.
    top_idx_sorted: Optional[torch.Tensor] = None
    # Full master-flat exclusion mask (bool, N_total_groups).
    # Pass B-group kernel reads it indexed by global_gid.
    exc_mask_full: Optional[torch.Tensor] = None
    # Per-name slot counter for GROUP records (separate from
    # name_slot which is for outliers). Pass B-group atomic_inc's here.
    name_group_slot: Optional[torch.Tensor] = None
    # Per-group bool mask (master flat). Pass B-group reads
    # this instead of recomputing score (decouples from dg drift).
    selected_mask: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Pinned descriptor pool (replaces per-segment torch.tensor(host_list, device=))
# ---------------------------------------------------------------------------
# Each `torch.tensor(host_list, device=cuda)` call triggers an implicit
# cudaStreamSynchronize. With ~7 prefix tensors + 4 per BF16 group × ~10
# segments per iter, ~80 syncs land on worker_stream → measured +95~210 ms
# iter time inflation (CLAUDE.md "host CPU 측 잔여").
#
# Replace with: write descriptor lists into a pre-allocated pinned-host slab,
# single non_blocking copy to device, kernels read device views. 2-buf
# ping-pong because the async H2D DMA reads host memory at execution time
# (not queue time) — CPU must not overwrite host slab before DMA completes.


class _DescBuf:
    """One ping-pong slot of the descriptor pool. Holds 4 fixed-size
    GPU+pinned-host pairs (i32 and i64) plus an event for copy-completion
    tracking."""

    __slots__ = (
        "host_i32", "dev_i32", "host_i64", "dev_i64",
        "cur_i32", "cur_i64", "copy_event",
    )

    def __init__(self, device: torch.device, cap_i32: int, cap_i64: int):
        self.host_i32 = torch.empty(cap_i32, dtype=torch.int32, pin_memory=True)
        self.dev_i32 = torch.empty(cap_i32, dtype=torch.int32, device=device)
        self.host_i64 = torch.empty(cap_i64, dtype=torch.int64, pin_memory=True)
        self.dev_i64 = torch.empty(cap_i64, dtype=torch.int64, device=device)
        self.cur_i32 = 0
        self.cur_i64 = 0
        self.copy_event: Optional[torch.cuda.Event] = None

    def reset(self) -> None:
        self.cur_i32 = 0
        self.cur_i64 = 0

    def add_i32(self, data: Sequence[int]) -> torch.Tensor:
        """Stage a host-side i32 list into this buf, return the device view
        (valid AFTER ``flush()`` has been queued on the kernel stream)."""
        n = len(data)
        if n == 0:
            return self.dev_i32.narrow(0, 0, 0)
        end = self.cur_i32 + n
        if end > self.host_i32.numel():
            raise RuntimeError(
                f"_DescBuf: i32 overflow ({end} > {self.host_i32.numel()}); "
                f"increase cap_i32 in LossyAosPipeline._ensure_desc_pool"
            )
        # torch.as_tensor on a python list is a single C-level conversion;
        # then a CPU memcpy into the pinned slab. Both fast for ≤ 100K ints.
        self.host_i32.narrow(0, self.cur_i32, n).copy_(
            torch.as_tensor(data, dtype=torch.int32)
        )
        view = self.dev_i32.narrow(0, self.cur_i32, n)
        self.cur_i32 = end
        return view

    def add_i64(self, data: Sequence[int]) -> torch.Tensor:
        n = len(data)
        if n == 0:
            return self.dev_i64.narrow(0, 0, 0)
        end = self.cur_i64 + n
        if end > self.host_i64.numel():
            raise RuntimeError(
                f"_DescBuf: i64 overflow ({end} > {self.host_i64.numel()}); "
                f"increase cap_i64 in LossyAosPipeline._ensure_desc_pool"
            )
        self.host_i64.narrow(0, self.cur_i64, n).copy_(
            torch.as_tensor(data, dtype=torch.int64)
        )
        view = self.dev_i64.narrow(0, self.cur_i64, n)
        self.cur_i64 = end
        return view


class _DescPool:
    """Per-segment double-buffered descriptor pool. One pool per
    LossyAosPipeline (i.e., per ExpertState / DramExpertState).

    Lifecycle per segment (inside ``run_segment_producer``):
      1. ``buf = pool.acquire()`` — waits if prev cycle's DMA still pending
      2. ``buf.add_i32(...)`` / ``buf.add_i64(...)`` repeatedly
      3. ``pool.flush(buf)`` — queues non-blocking H2D copy + records event
      4. ``... kernels use the returned device views ...``

    Capacity: i32 covers (prefix_pos, prefix_cnt, prefix_step) + per group
    (src_off, n_rec, aos_word). i64 covers per group (t_off). Sizing
    is bounded by ``MAX_EXPERTS_PER_SEG × (3 + 3 × N_BF16_GROUPS)``;
    defaults are generous (1 MB i32 + 512 KB i64 per buf × 2 bufs ≈ 3 MB
    total, all in-budget)."""

    def __init__(self, device: torch.device, cap_i32: int, cap_i64: int):
        self._bufs = [_DescBuf(device, cap_i32, cap_i64) for _ in range(2)]
        self._cur = 0

    def acquire(self) -> _DescBuf:
        buf = self._bufs[self._cur]
        # Wait for previous cycle's H2D DMA from this buf's host slab to
        # complete before we overwrite it. On steady state, GPU has long
        # finished by the time CPU loops back — synchronize is no-op fast
        # path. Defensive against CPU running ahead.
        ev = buf.copy_event
        if ev is not None:
            ev.synchronize()
            buf.copy_event = None
        buf.reset()
        return buf

    def flush(self, buf: _DescBuf) -> None:
        """Queue async H2D for whatever was staged into ``buf``. Records
        a copy-completion event on the CURRENT stream (caller is expected
        to be inside ``with torch.cuda.stream(worker_stream):``)."""
        if buf.cur_i32 > 0:
            buf.dev_i32.narrow(0, 0, buf.cur_i32).copy_(
                buf.host_i32.narrow(0, 0, buf.cur_i32), non_blocking=True
            )
        if buf.cur_i64 > 0:
            buf.dev_i64.narrow(0, 0, buf.cur_i64).copy_(
                buf.host_i64.narrow(0, 0, buf.cur_i64), non_blocking=True
            )
        if buf.copy_event is None:
            buf.copy_event = torch.cuda.Event()
        # Record AFTER the copies; subsequent acquire() of this buf
        # synchronize()s here before reuse.
        buf.copy_event.record()
        self._cur = (self._cur + 1) % len(self._bufs)


class LossyAosPipeline:
    """Owner of the per-iter GPU→pinned AOS staging path. Held by an
    ``ExpertState`` / ``DramExpertState`` instance and reused across
    every iter.

    Caller pre-allocates ``pinned_aos`` once (sized to ~half the total
    expert state bytes, see ``ExpertState.__init__``) and passes it
    here. The pipeline NEVER grows or re-allocates it — overflow
    raises ``RuntimeError`` (defensive; should not happen under any
    realistic ``topk_ratio`` because half the total state covers
    100% selection of all elements at fp32 → 20-byte records ratio).

    ``run(prepared, tracker_event, gpu_refs, bf16_groups)`` performs:
      * ``tracker_event.synchronize()`` (worker_stream sees per-BF16-
        group SoA writes from tracker_stream)
      * builds the AOS layout (per-expert / per-entry specs) via
        ``plan_layout``
      * partitions into blob-aligned segments (≤ 128 MiB each)
      * drives ``SegmentPipeline.run`` with a Triton-kernel producer
      * waits on the final D2H event then clears ``gpu_refs``

    Returns ``(expert_specs, total_aos_bytes)``. The caller decides
    what to do per expert (``::write`` for disk mode, numpy scatter
    into a dense mirror for DRAM mode).

    ``prepared`` is a list of ``(ent, total_n, filtered)`` where
    ``filtered`` is a list of ``(tname, t_off, bf16_group_idx, src_off,
    n_records)`` per sub-entry — ordered tname-major within an expert.
    """

    # Per-segment descriptor pool capacity. i32 holds prefix_pos/cnt/step
    # plus per-group (src_off, n_rec, aos_word_pos); i64 holds per-group
    # t_off. Worst case per segment ≈ N_experts_in_seg × (3 + 3 × N_groups)
    # + N_experts_in_seg × N_groups for i64. Generous defaults absorb any
    # realistic exp-1/2/3/extreme-scale workload (CLAUDE.md scaled-OLMoE
    # has 16 layers × 48 experts × 6 tnames ≈ 4608 entries × 8 groups ≈
    # 37K entries per segment).
    _DESC_CAP_I32: int = 256 * 1024     # 1 MB / buf
    _DESC_CAP_I64: int = 64 * 1024      # 512 KB / buf

    def __init__(self,
                 *,
                 pinned_aos: torch.Tensor,
                 segment_bytes: int = DEFAULT_SEGMENT_BYTES,
                 num_segment_buffers: int = 2) -> None:
        if not pinned_aos.is_pinned():
            raise ValueError("LossyAosPipeline: pinned_aos must be pinned")
        if pinned_aos.dtype != torch.uint8:
            raise ValueError("LossyAosPipeline: pinned_aos must be uint8")
        self.pinned_aos = pinned_aos
        self._segment_bytes = int(segment_bytes)
        self._num_segment_buffers = int(num_segment_buffers)
        self._segment_pipeline: Optional[object] = None  # SegmentPipeline
        self._desc_pool: Optional[_DescPool] = None

    def _ensure_segment_pipeline(self, device: torch.device) -> None:
        if self._segment_pipeline is not None:
            return
        from .segment_pipeline import SegmentPipeline
        from .async_writer import get_d2h_stream
        d2h_stream = get_d2h_stream()
        worker_stream = torch.cuda.current_stream(device)
        self._segment_pipeline = SegmentPipeline(
            device=device.index if device.index is not None else 0,
            worker_stream=worker_stream,
            d2h_stream=d2h_stream,
            segment_bytes=self._segment_bytes,
            num_buffers=self._num_segment_buffers,
        )

    def _ensure_desc_pool(self, device: torch.device) -> None:
        if self._desc_pool is not None:
            return
        self._desc_pool = _DescPool(
            device, self._DESC_CAP_I32, self._DESC_CAP_I64,
        )

    def run(
        self,
        *,
        step: int,
        prepared,
        tracker_event,
        gpu_refs,
        bf16_groups,
        lossy_state: Optional[LossyDispatchState] = None,
        outlier_n_per_expert_entry: Optional[Sequence[Sequence[int]]] = None,
    ):
        """Drive one iter of the GPU AOS pipeline. Returns
        ``(expert_specs, total_aos_bytes)``. ``pinned_aos`` (pre-
        allocated by the caller in ``__init__``) holds the full AOS
        layout after this call — caller does the per-expert
        finalisation (``::write`` or DRAM scatter).

        ``lossy_state``: if provided, Pass B writes outlier records
        directly to staging within each segment (per-segment name
        filtering via sentinel byte_off). ``outlier_n_per_expert_entry``
        provides the per-(expert, tname) outlier count used by
        plan_layout to compute correct entry shifts.
        """
        if tracker_event is not None:
            try:
                tracker_event.synchronize()
            except Exception:
                pass

        experts_n_step = []
        entries_per_expert = []
        for ent, total_n, filtered in prepared:
            experts_n_step.append((total_n, int(step)))
            ents = []
            for (_tname, t_off, bf16_g, src_off, n_recs) in filtered:
                ents.append((bf16_g, src_off, n_recs, t_off))
            entries_per_expert.append(ents)
        # In v2 path (lossy_state provided), pass per-entry
        # BF16-group-local within_off so plan_layout can stamp it onto
        # EntrySpec. Producer uses it to subtract from t_off so disk_idx
        # encoding works with bf16-local src_idx (no host-side subtract).
        within_off_per_entry: Optional[List[List[int]]] = None
        if lossy_state is not None and lossy_state.within_off_per_name:
            _name_to_winoff = {
                nm: lossy_state.within_off_per_name[i]
                for i, nm in enumerate(lossy_state.name_list)
            }
            within_off_per_entry = []
            for _ent_obj, _total_n, filtered in prepared:
                per_expert_woff: List[int] = []
                for (tname_f, _t_off, _bf16_g, _src_off, _n_recs) in filtered:
                    per_expert_woff.append(int(_name_to_winoff.get(tname_f, 0)))
                within_off_per_entry.append(per_expert_woff)
        expert_specs, entry_specs, total_aos = plan_layout(
            experts_n_step, entries_per_expert,
            outlier_n_per_entry=outlier_n_per_expert_entry,
            within_off_per_entry=within_off_per_entry,
        )
        if total_aos == 0:
            if gpu_refs is not None:
                gpu_refs.clear()
            return expert_specs, 0

        if total_aos > int(self.pinned_aos.numel()):
            raise RuntimeError(
                f"LossyAosPipeline: pinned_aos too small "
                f"({int(self.pinned_aos.numel())} < {total_aos}); the "
                f"pre-allocated capacity (half total expert state bytes) "
                f"was undersized for this iter. Either the model size "
                f"or topk_ratio changed since __init__."
            )

        # Resolve device from the first non-None BF16 group.
        first_g_idx = next(i for i, g in enumerate(bf16_groups) if g is not None)
        device = bf16_groups[first_g_idx][0].device
        self._ensure_segment_pipeline(device)
        self._ensure_desc_pool(device)

        segments = cut_into_blob_aligned_segments(
            expert_specs,
            max_segment_bytes=self._segment_pipeline.segment_bytes,
        )

        desc_pool = self._desc_pool

        # Pre-compute the per-name (tracker name_list order) AOS
        # byte offset of OUTLIER record start. entry_specs is FILTERED
        # (only names with non-zero records this iter); name_list is
        # the full tracker vocabulary. We map tname → entry → byte_off
        # via a tname→entry-index dict built from ``prepared``. Names
        # not present this iter get -1 (Pass B sentinel = skip).
        #
        # ALSO populate name_t_off here from prepared's ``t_off`` values
        # (which the caller of lossy_fc_append_batch sets per-mode:
        # ExpertState uses ent.global_offsets, DramExpertState uses
        # tname_offsets cumulative cursor). The dispatch-time guess
        # cannot know the mode, so this is the authoritative source.
        if lossy_state is not None:
            tname_to_entry_idx: Dict[str, int] = {}
            tname_to_t_off: Dict[str, int] = {}
            cur_ent_idx = 0
            for ent_obj, _total_n, filtered in prepared:
                for (tname_f, t_off, _bf16_g, _src_off, _n_recs) in filtered:
                    tname_to_entry_idx[tname_f] = cur_ent_idx
                    tname_to_t_off[tname_f] = int(t_off)
                    cur_ent_idx += 1
            assert cur_ent_idx == len(entry_specs), (
                f"plan_layout entry count mismatch: {cur_ent_idx} vs {len(entry_specs)}"
            )
            aos_byte_off_per_name = []
            t_off_per_name = []
            for nm in lossy_state.name_list:
                ei = tname_to_entry_idx.get(nm)
                if ei is None:
                    aos_byte_off_per_name.append(-1)
                    t_off_per_name.append(0)
                else:
                    aos_byte_off_per_name.append(int(entry_specs[ei].aos_byte_off))
                    t_off_per_name.append(tname_to_t_off[nm])
            lossy_state.aos_byte_off_per_name = aos_byte_off_per_name
            # Overwrite name_t_off (dispatch-time value may be wrong for
            # dram_expert_state which has no ``_entries`` attribute).
            lossy_state.name_t_off.copy_(
                torch.as_tensor(t_off_per_name, dtype=torch.int64),
                non_blocking=True,
            )

        def _producer(seg_view: torch.Tensor, byte_off: int, n: int) -> None:
            run_segment_producer(
                staging_u8=seg_view,
                seg_byte_off=byte_off,
                seg_n_bytes=n,
                expert_specs=expert_specs,
                entry_specs=entry_specs,
                bf16_groups=bf16_groups,
                device=device,
                desc_pool=desc_pool,
                lossy_state=lossy_state,
            )

        final_event = self._segment_pipeline.run(
            _producer, segments, self.pinned_aos,
        )
        if final_event is not None:
            try:
                final_event.synchronize()
            except Exception:
                pass
        if gpu_refs is not None:
            gpu_refs.clear()
        return expert_specs, total_aos


def cut_into_blob_aligned_segments(
    expert_specs: Sequence["ExpertSpec"],
    max_segment_bytes: int,
) -> List[Tuple[int, int]]:
    """Partition the global AOS layout into segments of at most
    ``max_segment_bytes``, with every cut on an expert blob boundary.

    Guarantees: each segment is non-empty, and every expert's blob
    lies entirely within ONE segment — so records never straddle
    segment boundaries. Required because records are 20 B and
    segment_pipeline's fixed-size cuts (e.g. 128 MiB) would otherwise
    misalign with record boundaries.

    Returns a list of ``(byte_offset, byte_length)`` tuples.
    Raises ``ValueError`` if a single expert blob exceeds
    ``max_segment_bytes`` (the segment size must accommodate the
    largest single blob).
    """
    if max_segment_bytes <= 0:
        raise ValueError(
            f"max_segment_bytes must be positive: {max_segment_bytes}"
        )
    segments: List[Tuple[int, int]] = []
    if not expert_specs:
        return segments
    seg_lo = expert_specs[0].aos_byte_off
    seg_hi = seg_lo
    for spec in expert_specs:
        blob_end = spec.aos_byte_off + spec.blob_padded_bytes
        if blob_end - seg_lo > max_segment_bytes:
            # Cut before this expert (its inclusion would overflow).
            if seg_hi == seg_lo:
                raise ValueError(
                    f"expert blob {spec.blob_padded_bytes} bytes "
                    f"> max_segment_bytes {max_segment_bytes}"
                )
            segments.append((seg_lo, seg_hi - seg_lo))
            seg_lo = spec.aos_byte_off
        seg_hi = blob_end
    if seg_hi > seg_lo:
        segments.append((seg_lo, seg_hi - seg_lo))
    return segments


def plan_layout(
    experts: Sequence[Tuple[int, int]],
    entries_per_expert: Sequence[Sequence[Tuple[int, int, int, int]]],
    outlier_n_per_entry: Optional[Sequence[Sequence[int]]] = None,
    within_off_per_entry: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[List[ExpertSpec], List[EntrySpec], int]:
    """Build the global AOS layout plan.

    ``experts``: list of ``(n_records, step)`` per expert in canonical
    iteration order. Empty experts (n_records=0) must still be passed
    if their blob occupies a slot (the caller decides; typically empty
    experts are filtered out before this call).

    ``entries_per_expert``: per-expert list of entries; each entry is
    ``(bf16_group_idx, src_off, n_records, t_off)`` in the tname-major
    order they must appear in this expert's blob.

    Returns ``(expert_specs, entry_specs, total_aos_bytes)`` where
    ``total_aos_bytes`` is the size of the global AOS layout — used
    by the caller to size the segment_pipeline run and pinned_aos.
    """
    if len(experts) != len(entries_per_expert):
        raise ValueError(
            f"plan_layout: experts vs entries length mismatch "
            f"({len(experts)} vs {len(entries_per_expert)})"
        )
    expert_specs: List[ExpertSpec] = []
    entry_specs: List[EntrySpec] = []
    cursor = 0
    for e_idx, (n_rec_e, step_e) in enumerate(experts):
        blob_unpadded = PREFIX_BYTES + int(n_rec_e) * RECORD_BYTES
        blob_padded = _round_up(blob_unpadded, SECTOR_BYTES)
        expert_specs.append(ExpertSpec(
            aos_byte_off=cursor,
            blob_padded_bytes=blob_padded,
            n_records=int(n_rec_e),
            step=int(step_e),
        ))
        # Per-entry positions within this expert.
        entry_byte_off = cursor + PREFIX_BYTES
        running = 0
        for ent_i, ent in enumerate(entries_per_expert[e_idx]):
            bf16_g_idx, src_off, n_rec_t, t_off = ent
            if outlier_n_per_entry is not None:
                outlier_n = int(outlier_n_per_entry[e_idx][ent_i])
                if outlier_n > n_rec_t:
                    raise ValueError(
                        f"plan_layout: entry({e_idx},{ent_i}) outlier_n="
                        f"{outlier_n} > n_records={n_rec_t}"
                    )
            else:
                outlier_n = 0
            if within_off_per_entry is not None:
                _within_off = int(within_off_per_entry[e_idx][ent_i])
            else:
                _within_off = 0
            entry_specs.append(EntrySpec(
                bf16_group_idx=int(bf16_g_idx),
                src_off=int(src_off),
                n_records=int(n_rec_t),
                aos_byte_off=entry_byte_off + running * RECORD_BYTES,
                t_off=int(t_off),
                outlier_n_records=outlier_n,
                entry_within_off=_within_off,
            ))
            running += int(n_rec_t)
        if running != int(n_rec_e):
            raise ValueError(
                f"plan_layout: expert {e_idx} has n_records={n_rec_e} "
                f"but sum of entry record counts = {running}"
            )
        cursor += blob_padded
    return expert_specs, entry_specs, cursor


# ---------------------------------------------------------------------------
# Per-segment producer
# ---------------------------------------------------------------------------

def run_segment_producer(
    *,
    staging_u8: torch.Tensor,
    seg_byte_off: int,
    seg_n_bytes: int,
    expert_specs: Sequence[ExpertSpec],
    entry_specs: Sequence[EntrySpec],
    bf16_groups: Sequence[Tuple[torch.Tensor, torch.Tensor,
                                torch.Tensor, torch.Tensor]],
    device: torch.device,
    record_block: int = 64,
    desc_pool: Optional["_DescPool"] = None,
    lossy_state: Optional[LossyDispatchState] = None,
    name_to_entry_map: Optional[List[int]] = None,
) -> None:
    """Fill ``staging_u8[0:seg_n_bytes]`` with the slice of the global
    AOS layout that lives at byte range ``[seg_byte_off, seg_byte_off + seg_n_bytes)``.

    Caller must invoke this on ``worker_stream`` (the stream the
    segment_pipeline producer runs on). All kernel launches inherit
    the current stream.

    ``staging_u8`` must be a contiguous uint8 GPU tensor of length
    >= seg_n_bytes. ``bf16_groups[i]`` is a 4-tuple of GPU tensors
    (idx_i64, p_f32, m_f32, v_f32) for BF16 group i; the caller is
    responsible for keeping these alive until kernel completion.

    Byte-alignment requirement: ``seg_byte_off`` must be a multiple of
    4 (segment_pipeline uses 128 MiB segments → trivially aligned).
    Within an expert's blob, prefix and records are 4-byte aligned by
    construction (PREFIX_BYTES = 8, RECORD_BYTES = 20). Expert blob
    boundaries are 4 KiB-aligned.
    """
    if seg_byte_off & 3:
        raise ValueError(
            f"run_segment_producer: seg_byte_off must be 4-byte aligned, "
            f"got {seg_byte_off}"
        )
    if seg_n_bytes & 3:
        raise ValueError(
            f"run_segment_producer: seg_n_bytes must be 4-byte aligned, "
            f"got {seg_n_bytes}"
        )
    if staging_u8.numel() < seg_n_bytes:
        raise ValueError(
            f"run_segment_producer: staging too small "
            f"({staging_u8.numel()} < {seg_n_bytes})"
        )
    if staging_u8.dtype != torch.uint8 or not staging_u8.is_contiguous():
        raise ValueError(
            "run_segment_producer: staging must be contiguous uint8"
        )

    seg_words = seg_n_bytes // 4
    seg_i32 = staging_u8.narrow(0, 0, seg_n_bytes).view(torch.int32)

    # 1) Zero-fill the segment region. Single launch.
    ZERO_BLOCK = 4096
    n_progs = (seg_words + ZERO_BLOCK - 1) // ZERO_BLOCK
    _zero_segment_kernel[(n_progs,)](
        seg_i32, seg_words, BLOCK=ZERO_BLOCK,
    )

    # 2) Collect prefix descriptors (host side).
    seg_lo = int(seg_byte_off)
    seg_hi = seg_lo + int(seg_n_bytes)
    prefix_word_positions: List[int] = []
    prefix_counts: List[int] = []
    prefix_steps: List[int] = []
    for spec in expert_specs:
        prefix_byte = spec.aos_byte_off                # prefix sits at blob start
        if seg_lo <= prefix_byte < seg_hi:
            # Word position relative to segment start.
            prefix_word_positions.append((prefix_byte - seg_lo) // 4)
            prefix_counts.append(spec.n_records)
            prefix_steps.append(spec.step)

    # 3) Per BF16 group, collect (clipped src_off, n_in_seg, aos_word_pos
    # within segment, t_off) for each entry intersecting this segment.
    # One kernel launch per non-empty group.
    #
    # In Option A replace mode, aos_records writes ONLY the GROUP portion
    # of each entry. Entry layout is [outlier × n_o][group × n_g]; aos_
    # records starts at (entry.aos_byte_off + n_o × 20). Pass B (called
    # later) writes the outlier portion at entry.aos_byte_off.
    by_group_src_off:  List[List[int]] = [[] for _ in range(len(bf16_groups))]
    by_group_n_rec:    List[List[int]] = [[] for _ in range(len(bf16_groups))]
    by_group_aos_word: List[List[int]] = [[] for _ in range(len(bf16_groups))]
    by_group_t_off:    List[List[int]] = [[] for _ in range(len(bf16_groups))]
    # In Option A v2 path, t_off passed to the kernel is
    # ``ent.t_off - ent.entry_within_off`` so that ``disk_idx =
    # bf16_local + t_off_v2`` equals the v1 ``entry_local + ent.t_off``.
    # This lets src_idx stay as bf16-local (no host-side subtraction)
    # → eliminates the per-iter ``elem_idx_grp_bf16 - off_per_elem``
    # alloc + ``.to(int64)`` cast.
    _use_v2 = lossy_state is not None
    for ent in entry_specs:
        outlier_n = int(getattr(ent, "outlier_n_records", 0))
        group_n_total = int(ent.n_records) - outlier_n
        # Group-record region of this entry within global AOS layout.
        group_lo = ent.aos_byte_off + outlier_n * RECORD_BYTES
        group_hi = group_lo + group_n_total * RECORD_BYTES
        if group_n_total == 0 or group_hi <= seg_lo or group_lo >= seg_hi:
            continue
        # Clip to segment edges (within group-record range only).
        r_start = (max(group_lo, seg_lo) - group_lo) // RECORD_BYTES
        r_end   = (min(group_hi, seg_hi) - group_lo) // RECORD_BYTES
        n_in_seg = r_end - r_start
        if n_in_seg <= 0:
            continue
        first_rec_byte = group_lo + r_start * RECORD_BYTES
        g = ent.bf16_group_idx
        by_group_src_off[g].append(ent.src_off + r_start)
        by_group_n_rec[g].append(n_in_seg)
        by_group_aos_word[g].append((first_rec_byte - seg_lo) // 4)
        if _use_v2:
            # v2 kernel: src_idx stores bf16-local; t_off carries the
            # combined offset so disk_idx = bf16_local + (t_off - winoff).
            by_group_t_off[g].append(ent.t_off - ent.entry_within_off)
        else:
            by_group_t_off[g].append(ent.t_off)

    # 4) Single H2D transfer of all descriptors via the pinned pool.
    # Fallback: legacy torch.tensor(host_list, device=) path when desc_pool
    # is unavailable (e.g., unit tests). The fallback path keeps the old
    # implicit cudaStreamSynchronize behavior — only the production path
    # (LossyAosPipeline) gets the win.
    if desc_pool is not None:
        buf = desc_pool.acquire()
        pos_view = buf.add_i32(prefix_word_positions) if prefix_word_positions else None
        cnt_view = buf.add_i32(prefix_counts) if prefix_counts else None
        stp_view = buf.add_i32(prefix_steps) if prefix_steps else None

        # Cache per-group views in order; None entries mean "no kernel".
        grp_descs: List[Optional[Tuple[torch.Tensor, torch.Tensor,
                                        torch.Tensor, torch.Tensor]]] = []
        for g_idx in range(len(bf16_groups)):
            if bf16_groups[g_idx] is None or len(by_group_src_off[g_idx]) == 0:
                grp_descs.append(None)
                continue
            grp_descs.append((
                buf.add_i32(by_group_src_off[g_idx]),
                buf.add_i32(by_group_n_rec[g_idx]),
                buf.add_i32(by_group_aos_word[g_idx]),
                buf.add_i64(by_group_t_off[g_idx]),
            ))

        # Option A: stage per-segment name_outlier_byte_off (-1 sentinel
        # for names whose entry is NOT in this segment). Pass B kernel
        # skips elements whose name has sentinel. Length = N_names.
        nob_seg_view: Optional[torch.Tensor] = None
        if lossy_state is not None:
            nob_seg_host = []
            for boff_global in lossy_state.aos_byte_off_per_name:
                if boff_global >= 0 and seg_lo <= boff_global < seg_hi:
                    nob_seg_host.append(int(boff_global) - int(seg_lo))
                else:
                    nob_seg_host.append(-1)
            nob_seg_view = buf.add_i64(nob_seg_host)

        # Single non_blocking H2D for the whole descriptor batch. Records
        # an event so the next acquire() of this buf waits for completion.
        desc_pool.flush(buf)

        if pos_view is not None:
            _prefix_write_kernel[(len(prefix_word_positions),)](
                seg_i32, pos_view, cnt_view, stp_view,
                len(prefix_word_positions),
            )

        # Option A Pass B — writes outlier records directly to seg_i32 at
        # entry-start byte offsets (BEFORE the group records that
        # aos_records writes). Pass B does NOT fold (Pass A in update_
        # delta_accum already did the fold via DO_FOLD=True).
        if lossy_state is not None and nob_seg_view is not None:
            from .outlier_kernels import launch_aos_write_for_subrange
            n_names_total = len(lossy_state.aos_byte_off_per_name)
            for g_idx in range(len(bf16_groups)):
                if g_idx >= len(lossy_state.bf16_groups_mvph):
                    break
                mvph = lossy_state.bf16_groups_mvph[g_idx]
                if mvph is None:
                    continue
                m_view, v_view, p_view, dg_view, master_off_elem = mvph
                launch_aos_write_for_subrange(
                    m_view=m_view, v_view=v_view, p_view=p_view,
                    accum_view=dg_view,
                    v_q=lossy_state.v_q, h_q=lossy_state.h_q,
                    ofs_elem_full=lossy_state.ofs_elem_full,
                    name_offset_bytes=nob_seg_view,
                    name_t_off=lossy_state.name_t_off,
                    name_slot=lossy_state.name_slot,
                    aos_out_i32=seg_i32,
                    prev_th_t=lossy_state.prev_th_t,
                    group_th_t=lossy_state.group_th_t,
                    master_off_elem=master_off_elem,
                    g_blk=lossy_state.g_blk,
                    lr_over_bc1=lossy_state.lr_over_bc1,
                    sqrt_bc2_inv=lossy_state.sqrt_bc2_inv,
                    eps=lossy_state.eps,
                    n_names=n_names_total,
                    n_q=lossy_state.n_q,
                    do_fold=True,  # Pass B does the selective fold (Pass A leaves dg unchanged).
                )

        RECORD_BLOCK = int(record_block)
        # When lossy_state is provided: group records via
        # Pass B-group kernel (rescan + atomic_inc per-name slot + write).
        # No K_max scratch (no sel_buf / top_idx_sorted). For legacy
        # (lossy_state is None), keep _aos_records_kernel.
        if lossy_state is not None:
            from .outlier_kernels import launch_aos_group_write
            # Per-segment GROUP-region byte offsets per name.
            ng_off_host: List[int] = []
            for nm_i, boff_global in enumerate(lossy_state.aos_byte_off_per_name):
                if boff_global < 0:
                    ng_off_host.append(-1)
                    continue
                n_o = (int(lossy_state.name_outlier_count_host[nm_i])
                       if nm_i < len(lossy_state.name_outlier_count_host)
                       else 0)
                group_lo = int(boff_global) + n_o * RECORD_BYTES
                if seg_lo <= group_lo < seg_hi:
                    ng_off_host.append(group_lo - int(seg_lo))
                else:
                    ng_off_host.append(-1)
            ng_off_dev = buf.add_i64(ng_off_host)
            desc_pool.flush(buf)
            n_names_total = len(lossy_state.aos_byte_off_per_name)
            for g_idx in range(len(bf16_groups)):
                if g_idx >= len(lossy_state.bf16_groups_mvph):
                    break
                mvph = lossy_state.bf16_groups_mvph[g_idx]
                if mvph is None:
                    continue
                m_view, v_view, p_view, dg_view, master_off_elem = mvph
                master_off_g_groups = int(master_off_elem) // int(lossy_state.g_blk)
                n_groups_in_bf16 = int(dg_view.numel())
                launch_aos_group_write(
                    out_i32=seg_i32,
                    m_flat=m_view,
                    v_flat=v_view,
                    p_flat=p_view,
                    selected_mask=lossy_state.selected_mask,
                    offsets_t=lossy_state.ofs_elem_full,
                    name_group_slot=(lossy_state.name_group_slot
                                     if lossy_state.name_group_slot is not None
                                     else lossy_state.name_slot),
                    name_group_byte_off=ng_off_dev,
                    name_t_off=lossy_state.name_t_off,
                    master_off_g=master_off_g_groups,
                    n_groups_in_bf16=n_groups_in_bf16,
                    n_names=n_names_total,
                    g_blk=lossy_state.g_blk,
                    accum=dg_view,
                )
        # lossy_state is mandatory — legacy SoA path
        # removed. If reached without state, that's a caller bug.
        return
