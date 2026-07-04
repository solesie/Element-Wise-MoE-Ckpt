"""Per-expert state storage: lossy FC append log + LFCL compaction baseline.

This module implements the AGENTS.md V2 expert checkpoint contract:

* **Lossy FC (`lossy_expert_L{L:02d}_E{E:02d}.bin`)** — an append-only log.
  Every iteration, the top-K% element selections for this expert are
  packed into a single per-iter block and appended. File is NOT truncated
  during normal training; it grows bounded by the LFCL compaction
  cycle (``D × ⌈100/F⌉`` iters) and is unlinked at the next LFCL
  of this expert.

* **LFCL (`fc_expert_L{L:02d}_E{E:02d}.bin`)** — an expert recovery baseline
  compacted from a pinned payload materialized from live GPU p/m/v. After a
  successful fsync + atomic rename, the previous lossy FC file is unlinked
  and a fresh header-only lossy FC file is created so the next iter's append
  starts clean.

Binary formats:

``LFCL`` (pinned live-GPU payload compaction):
    [magic 4B "LFCL"][version u32][header_len u32][pickled meta]
    [pad to HEADER_BLOCK_BYTES = 4096]
    [fp32 payload: for each tensor in schema order,
        param_flat (numel × 4B), m_flat, v_flat]

``LFSY`` (Lossy FC append log):
    [magic 4B "LFSY"][version u32][header_len u32][pickled meta]
    [pad to HEADER_BLOCK_BYTES]
    per-iter block (repeated):
        [count u32][step u32]
        [count × 16B records: idx u32, p f32, m f32, v f32]

Recovery reads LFCL fully, then replays LFSY block-by-block in step order
(last write wins per index).

Write-path performance
======================

Production writes AOS records on GPU into a segment pipeline, D2H-copies
the result into one reusable pinned AOS slab, then lets the expert writer
append the leased slab to LFSY. The slab is released only after the writer
finishes, so bounded capacity failures are explicit rather than hidden by
unbounded host buffering.
"""
from __future__ import annotations

import os
import pickle
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from .proposal_io import (
    HEADER_BLOCK_BYTES as _BIN_HEADER_BLOCK_BYTES,
    SECTOR_BYTES as _BIN_SECTOR_BYTES,
    alloc_pinned_uint8_sector_aligned,
    append_lossy_log_from_pinned,
    backend_kind,
    close_fd,
    fdp_begin_expert_batch,
    fdp_end_expert_batch_stats,
    fdp_mark_writer_done,
    fdp_metadata_mmap_enabled,
    fsync_fd,
    open_lossy_log_for_append,
    scatter_lossy_records_to_dense_by_intervals,
    write_header_payload_from_pinned_keep_open,
    write_lfcs_from_pinned,
)
from ..emission.segment_d2h import DEFAULT_SEGMENT_BYTES
from ..emission.aos_emitter import AosEmitter
from ..emission.aos_layout import PinnedAosLayoutMirror, read_lfsy_prefix_count


ExpertId = Tuple[int, int]  # (layer, expert_idx)

# ---- Binary format constants -----------------------------------------------
MAGIC_LFCL = b"LFCL"  # dense recovery-image compaction baseline
MAGIC_LFSY = b"LFSY"  # lossy FC append log
FORMAT_VERSION: int = 1
# Reuse the file-format header block size from binary_io so PRPTENS,
# LFCL, and LFSY all share a single sector convention. 64 KiB is the
# current value (sized to fit the largest header — non-expert FC).
HEADER_BLOCK_BYTES: int = _BIN_HEADER_BLOCK_BYTES


def _require_pinned_cpu_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: Optional[torch.dtype] = None,
) -> None:
    if tensor.device.type != "cpu" or not tensor.is_pinned():
        raise RuntimeError(f"{name} must be a pinned CPU tensor")
    if dtype is not None and tensor.dtype != dtype:
        raise RuntimeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
RECORD_BYTES: int = 16  # uint32 idx + fp32 p + fp32 m + fp32 v
BLOCK_PREFIX_BYTES: int = 8  # count u32 + step u32

# Per-record numpy structured dtype (little-endian, packed to 16B).
# idx is uint32: per-expert combined offset = tname_base_offset + idx_local.
# ExpertState validates schema size before writing so this does not silently
# wrap on oversized expert shards.
_RECORD_DTYPE = np.dtype([
    ("idx", "<u4"),
    ("p", "<f4"),
    ("m", "<f4"),
    ("v", "<f4"),
])


def _lossless_name(layer: int, expert: int) -> str:
    return f"fc_expert_L{layer:02d}_E{expert:02d}.bin"


def _lossy_name(layer: int, expert: int) -> str:
    return f"lossy_expert_L{layer:02d}_E{expert:02d}.bin"


# ----------------------------------------------------------------------------
# Header pack / unpack (4 KiB prefix)
# ----------------------------------------------------------------------------
def _pack_header(magic: bytes, meta: Dict[str, Any]) -> bytes:
    payload = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
    prefix = magic + struct.pack("<II", FORMAT_VERSION, len(payload)) + payload
    if len(prefix) > HEADER_BLOCK_BYTES:
        raise RuntimeError(
            f"expert_state header ({len(prefix)} B) exceeds HEADER_BLOCK_BYTES "
            f"({HEADER_BLOCK_BYTES}); reduce tensor-schema bulk."
        )
    return prefix + b"\x00" * (HEADER_BLOCK_BYTES - len(prefix))


def _unpack_header(block: bytes) -> Tuple[bytes, int, Dict[str, Any]]:
    if len(block) < 12:
        raise ValueError(f"expert_state header too short: {len(block)} B")
    magic = block[:4]
    version, pickle_len = struct.unpack("<II", block[4:12])
    if magic not in (MAGIC_LFCL, MAGIC_LFSY):
        raise ValueError(f"unknown expert_state magic: {magic!r}")
    if 12 + pickle_len > HEADER_BLOCK_BYTES:
        raise ValueError(f"pickle_len {pickle_len} overflows 4 KiB block")
    meta = pickle.loads(block[12:12 + pickle_len])
    return magic, version, meta


# ----------------------------------------------------------------------------
# Per-expert writer state (internal)
# ----------------------------------------------------------------------------
@dataclass
class _ExpertEntry:
    """Bookkeeping for one expert's LFCL baseline + LFSY append log."""
    expert_id: ExpertId
    tensor_schema: List[Tuple[str, torch.Size, int]]
    global_offsets: Dict[str, int]
    total_numel: int
    lossless_path: Path
    lossy_path: Path
    partition_fragments: Optional[List[Dict[str, Any]]] = None
    # Open fd on lossy file (append mode), None until first lossy append.
    lossy_fd: Optional[int] = None
    lossy_start_step: int = -1
    lossless_step: int = -1  # last step at which LFCL compaction fired.
    step: int = -1


# ----------------------------------------------------------------------------
# Main class: ExpertState
# ----------------------------------------------------------------------------
class ExpertState:
    """Per-expert lossy FC append-only log + LFCL compaction baseline.

    Call pattern from the training loop (integrated via FCManager +
    lossy checkpoint engine):

        state = ExpertState(expert_tensor_map, output_dir, fresh=True)
        # ... each iter:
        #   1) FCManager determines experts doing LFCL compaction this iter.
        for eid in lfcl_expert_ids_this_iter:
            state.compact_lossy_recovery_image(eid, step=iter)
        # 2) lossy checkpoint engine computes top-K across expert elements.
        for eid, tensor_updates in selections.items():
            state.append_lossy_records_for_test(eid, step=iter, tensor_updates=tensor_updates)

    At the end: state.shutdown() to close fds.
    """

    def __init__(
        self,
        expert_tensor_map: Dict[ExpertId, List[Tuple[str, torch.Size, int]]],
        output_dir: Path,
        *,
        fresh: bool = True,
        checkpoint_workers: Optional[Any] = None,
        cp_target: str = "disk",
        pinned_aos_record_capacity: Optional[int] = None,
        commit_lossy_records: bool = True,
        test_skip_d2h: bool = False,
        partition_fragments_by_expert: Optional[
            Dict[ExpertId, List[Dict[str, Any]]]
        ] = None,
    ) -> None:
        """
        Args:
            expert_tensor_map: per-expert tensor schema.
            output_dir: where LFCL / LFSY files live.
            fresh: purge any existing files at that path.
            checkpoint_workers: a ``CheckpointWorkerPool`` whose
                ``submit_expert_write(fn, *args)`` dispatches raw-callable I/O jobs on its dedicated
                worker thread. When None, writes execute synchronously on
                the caller thread — intended for unit tests / offline
                tools where background I/O isn't desirable.
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[ExpertId, _ExpertEntry] = {}
        self._checkpoint_workers = checkpoint_workers
        if cp_target not in ("disk", "test_dram"):
            raise ValueError(
                "cp_target must be 'disk' or 'test_dram', "
                f"got {cp_target!r}"
            )
        self.cp_target = cp_target
        self.commit_lossy_records = bool(commit_lossy_records)
        self.test_skip_d2h = bool(test_skip_d2h)
        self._fdp_pending_lock = threading.Lock()
        self._fdp_pending_writes_by_step: Dict[int, int] = {}
        self._fdp_done_marked_steps: Set[int] = set()
        if self.test_skip_d2h and self.commit_lossy_records:
            raise ValueError(
                "test_skip_d2h requires commit_lossy_records=False; "
                "it is only valid for write-excluded iter-time profiling."
            )
        self._dense_buffers: Dict[ExpertId, Dict[str, Dict[str, Any]]] = {}
        self._dense_interval_cache: Dict[ExpertId, Tuple[
            torch.Tensor, torch.Tensor, List[str],
        ]] = {}
        self._dense_flat = torch.empty(0, dtype=torch.float32)
        self._lossy_log_bytes_by_expert: Dict[ExpertId, int] = {}
        self._lossy_log_counter_index: Dict[ExpertId, int] = {}
        self._lossy_log_bytes_device: Optional[torch.Tensor] = None
        self._lossy_log_bytes_device_dirty: set[ExpertId] = set()
        self._direct_record_cursor_device: Optional[torch.Tensor] = None
        self._direct_overflow_count_device: Optional[torch.Tensor] = None
        self._direct_record_cursor_pinned: Optional[torch.Tensor] = None
        self._direct_overflow_count_pinned: Optional[torch.Tensor] = None
        self._direct_capacity_pinned: Optional[torch.Tensor] = None
        self._direct_offset_pinned: Optional[torch.Tensor] = None
        self._direct_total_th_pinned: Optional[torch.Tensor] = None
        self._direct_feedback_event: Optional["torch.cuda.Event"] = None
        self._direct_feedback_step: int = -1
        self._direct_feedback_n_experts: int = 0
        self._direct_feedback_consumed_step: int = -1
        self._direct_feedback_tau_version: int = -1
        self._direct_feedback_tau_h_epoch: int = -1
        self._direct_feedback_tau_source_step: int = -1
        self._direct_feedback_tau_apply_from_step: int = -1
        self._direct_feedback_tau_lr_over_bc1: float = 0.0
        self._direct_feedback_expert_ids: List[ExpertId] = []
        self._latest_direct_feedback: Optional[Dict[str, Any]] = None
        self._direct_layout_feedback_enabled = (
            os.environ.get("PROPOSAL_APPROX_PROBE_TELEMETRY_DETAIL", "")
            in ("1", "true", "TRUE")
        )
        self._prepared_direct_updates_key: Optional[int] = None
        self._prepared_direct_updates: Optional[
            List[Tuple["_ExpertEntry", int, List[Tuple[str, int, int, int, int]]]]
        ] = None
        self._expert_order: List[ExpertId] = list(expert_tensor_map.keys())
        self._expert_max_records_by_id: Dict[ExpertId, int] = {
            eid: sum(int(numel) for (_tname, _shape, numel) in schema)
            for eid, schema in expert_tensor_map.items()
        }
        # Pre-allocate one pinned AOS slab. Unit/offline callers default to
        # local worst-case; production passes a K-budget plus tolerance record
        # capacity after adaptive threshold guards have bounded the count.
        _total_param_numel = sum(
            int(numel)
            for schema in expert_tensor_map.values()
            for (_tname, _shape, numel) in schema
        )
        if pinned_aos_record_capacity is None:
            _record_capacity = _total_param_numel
            _capacity_desc = "local worst-case"
        else:
            _record_capacity = min(
                _total_param_numel,
                max(
                    1 if _total_param_numel > 0 else 0,
                    int(pinned_aos_record_capacity),
                ),
            )
            _capacity_desc = f"configured capacity ({_record_capacity} records)"
        self._pinned_aos_record_capacity = int(_record_capacity)
        _sect = int(_BIN_SECTOR_BYTES)
        _n_local_experts = len(expert_tensor_map)
        _pinned_aos_bytes = (
            _record_capacity * RECORD_BYTES
            + _n_local_experts * (BLOCK_PREFIX_BYTES + _sect)
        )
        _pinned_aos_bytes = (
            (_pinned_aos_bytes + _sect - 1) // _sect
        ) * _sect
        if _pinned_aos_bytes <= 0:
            _pinned_aos_bytes = _sect
        self._pinned_aos_total_bytes = int(_pinned_aos_bytes)
        self._pinned_aos: torch.Tensor = alloc_pinned_uint8_sector_aligned(
            _pinned_aos_bytes
        )
        self._pinned_aos_ready = threading.Event()
        self._pinned_aos_ready.set()
        self._pinned_aos_layout_mirror = PinnedAosLayoutMirror(
            max_experts=len(expert_tensor_map)
        )
        print(
            f"[proposal] ExpertState pinned_aos pre-alloc: "
            f"{_pinned_aos_bytes / 1e9:.2f} GB "
            f"({_capacity_desc}; local_total={_total_param_numel / 1e9:.2f}G elements)",
            flush=True,
        )

        _max_expert_blob_bytes = _BIN_SECTOR_BYTES
        for schema in expert_tensor_map.values():
            _max_records_for_expert = sum(int(numel) for (_tname, _shape, numel) in schema)
            _blob_unpadded = BLOCK_PREFIX_BYTES + _max_records_for_expert * RECORD_BYTES
            _blob_bytes = (
                (_blob_unpadded + _BIN_SECTOR_BYTES - 1)
                // _BIN_SECTOR_BYTES
            ) * _BIN_SECTOR_BYTES
            _max_expert_blob_bytes = max(_max_expert_blob_bytes, _blob_bytes)
        _segment_bytes = DEFAULT_SEGMENT_BYTES
        _num_segment_buffers = 2

        # Lossy record pipeline shares ``self._pinned_aos`` — no internal
        # growth. GPU staging/descriptor pools are warmed by init_proposal
        # before the first checkpoint iter.
        self._record_pipeline: AosEmitter = AosEmitter(
            pinned_aos=self._pinned_aos,
            target_record_capacity=self._pinned_aos_record_capacity,
            segment_bytes=_segment_bytes,
            num_segment_buffers=_num_segment_buffers,
        )

        n_counter_slots = len(expert_tensor_map)
        try:
            self._lossy_log_bytes_pinned = torch.zeros(
                n_counter_slots, dtype=torch.int64, pin_memory=True,
            )
        except RuntimeError:
            self._lossy_log_bytes_pinned = torch.zeros(
                n_counter_slots, dtype=torch.int64,
            )

        for counter_idx, (eid, schema) in enumerate(expert_tensor_map.items()):
            L, E = eid
            # Compute global flat offsets within the expert's
            # concatenated (param|m|v) address space.
            offsets: Dict[str, int] = {}
            cursor = 0
            for (tn, _shape, numel) in schema:
                offsets[tn] = cursor
                cursor += int(numel)
            if cursor > 0xFFFFFFFF:
                raise ValueError(
                    f"expert {eid} has {cursor} elements in its recovery "
                    "image; V2 compact lossy records encode expert-local "
                    "indices as uint32"
                )
            entry = _ExpertEntry(
                expert_id=eid,
                tensor_schema=list(schema),
                global_offsets=offsets,
                total_numel=cursor,
                lossless_path=self._output_dir / _lossless_name(L, E),
                lossy_path=self._output_dir / _lossy_name(L, E),
                partition_fragments=(
                    None if partition_fragments_by_expert is None
                    else list(partition_fragments_by_expert.get(eid, []))
                ),
            )
            if fresh:
                from .proposal_io import unlink_path as _unlink
                for p in (entry.lossless_path, entry.lossy_path):
                    _unlink(p)
            self._entries[eid] = entry
            self._lossy_log_bytes_by_expert[eid] = 0
            self._lossy_log_counter_index[eid] = int(counter_idx)
        if self.cp_target == "test_dram" and self.commit_lossy_records:
            self._dense_flat = self._build_dense_recovery_image(expert_tensor_map)
            self._build_dense_interval_cache()

    def warmup_record_pipeline(self, device: torch.device) -> None:
        self._record_pipeline.warmup(device)

    @property
    def pinned_aos_total_bytes(self) -> int:
        return int(self._pinned_aos_total_bytes)

    def _ensure_direct_emission_counters(self, device: torch.device) -> None:
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != "cuda":
            return
        n = max(1, len(self._entries))
        if (
            self._direct_record_cursor_device is not None
            and self._direct_record_cursor_device.device == device
            and self._direct_record_cursor_device.numel() >= n
            and self._direct_overflow_count_device is not None
            and self._direct_overflow_count_device.device == device
            and self._direct_overflow_count_device.numel() >= n
            and self._direct_record_cursor_pinned is not None
            and self._direct_record_cursor_pinned.numel() >= n
            and self._direct_overflow_count_pinned is not None
            and self._direct_overflow_count_pinned.numel() >= n
            and (
                not self._direct_layout_feedback_enabled
                or (
                    self._direct_capacity_pinned is not None
                    and self._direct_capacity_pinned.numel() >= n
                    and self._direct_offset_pinned is not None
                    and self._direct_offset_pinned.numel() >= n
                )
            )
            and self._direct_total_th_pinned is not None
            and self._direct_feedback_event is not None
        ):
            return
        self._direct_record_cursor_device = torch.zeros(
            n, dtype=torch.int64, device=device,
        )
        self._direct_overflow_count_device = torch.zeros(
            n, dtype=torch.int64, device=device,
        )
        self._direct_record_cursor_pinned = torch.empty(
            n, dtype=torch.int64, pin_memory=True,
        )
        self._direct_overflow_count_pinned = torch.empty(
            n, dtype=torch.int64, pin_memory=True,
        )
        if self._direct_layout_feedback_enabled:
            self._direct_capacity_pinned = torch.empty(
                n, dtype=torch.int64, pin_memory=True,
            )
            self._direct_offset_pinned = torch.empty(
                n, dtype=torch.int64, pin_memory=True,
            )
        else:
            self._direct_capacity_pinned = None
            self._direct_offset_pinned = None
        self._direct_total_th_pinned = torch.empty(
            1, dtype=torch.float32, pin_memory=True,
        )
        self._direct_feedback_event = torch.cuda.Event()

    def _ensure_lossy_log_byte_device_counter(self, device: torch.device) -> None:
        """Allocate the optional CUDA mirror for LFCL log-size planning."""

        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != "cuda":
            return
        if (
            self._lossy_log_bytes_device is not None
            and self._lossy_log_bytes_device.device == device
            and self._lossy_log_bytes_device.numel()
            == self._lossy_log_bytes_pinned.numel()
        ):
            return
        self._lossy_log_bytes_device = torch.zeros(
            int(self._lossy_log_bytes_pinned.numel()),
            dtype=torch.int64,
            device=device,
        )
        self._lossy_log_bytes_device.copy_(
            self._lossy_log_bytes_pinned,
            non_blocking=True,
        )

    def _sync_lossy_log_counter_slot_to_device(self, expert_id: ExpertId) -> None:
        dev = self._lossy_log_bytes_device
        if dev is None:
            return
        idx = self._lossy_log_counter_index.get(expert_id)
        if idx is None:
            return
        dev.narrow(0, int(idx), 1).copy_(
            self._lossy_log_bytes_pinned.narrow(0, int(idx), 1),
            non_blocking=True,
        )

    def _set_lossy_log_bytes(self, expert_id: ExpertId, value: int) -> None:
        self._set_lossy_log_bytes_impl(expert_id, value, sync_device=True)

    def _set_lossy_log_bytes_impl(
        self,
        expert_id: ExpertId,
        value: int,
        *,
        sync_device: bool,
    ) -> None:
        value_i = max(0, int(value))
        self._lossy_log_bytes_by_expert[expert_id] = value_i
        idx = self._lossy_log_counter_index.get(expert_id)
        if idx is not None:
            self._lossy_log_bytes_pinned[int(idx)] = value_i
            if sync_device:
                self._sync_lossy_log_counter_slot_to_device(expert_id)
                self._lossy_log_bytes_device_dirty.discard(expert_id)
            else:
                self._lossy_log_bytes_device_dirty.add(expert_id)

    def _add_lossy_log_bytes(self, expert_id: ExpertId, delta_bytes: int) -> None:
        current = int(self._lossy_log_bytes_by_expert.get(expert_id, 0))
        self._set_lossy_log_bytes(expert_id, current + int(delta_bytes))

    def _add_lossy_log_bytes_from_writer(
        self,
        expert_id: ExpertId,
        delta_bytes: int,
    ) -> None:
        current = int(self._lossy_log_bytes_by_expert.get(expert_id, 0))
        self._set_lossy_log_bytes_impl(
            expert_id,
            current + int(delta_bytes),
            sync_device=False,
        )

    def _sync_dirty_lossy_log_counters_to_device(self) -> None:
        if self._lossy_log_bytes_device is None:
            self._lossy_log_bytes_device_dirty.clear()
            return
        if not self._lossy_log_bytes_device_dirty:
            return
        for expert_id in list(self._lossy_log_bytes_device_dirty):
            self._sync_lossy_log_counter_slot_to_device(expert_id)
        self._lossy_log_bytes_device_dirty.clear()

    def _build_dense_recovery_image(
        self,
        expert_tensor_map: Dict[ExpertId, List[Tuple[str, torch.Size, int]]],
    ) -> torch.Tensor:
        """Allocate the pinned host dense recovery image for test_dram."""

        total_numel = 0
        for schema in expert_tensor_map.values():
            for (_tname, _shape, numel) in schema:
                total_numel += int(numel) * 3
        if torch.cuda.is_available() and int(total_numel) > 0:
            flat = torch.empty(
                int(total_numel), dtype=torch.float32, pin_memory=True,
            )
            flat.zero_()
        else:
            flat = torch.zeros(int(total_numel), dtype=torch.float32)
        offset = 0
        for eid, schema in expert_tensor_map.items():
            tensor_buffers: Dict[str, Dict[str, Any]] = {}
            for (tname, shape, numel) in schema:
                n = int(numel)
                p_view = flat.narrow(0, offset, n); offset += n
                m_view = flat.narrow(0, offset, n); offset += n
                v_view = flat.narrow(0, offset, n); offset += n
                tensor_buffers[tname] = {
                    "p": p_view,
                    "m": m_view,
                    "v": v_view,
                    "p_np": p_view.numpy(),
                    "m_np": m_view.numpy(),
                    "v_np": v_view.numpy(),
                    "shape": shape,
                    "numel": n,
                }
            self._dense_buffers[eid] = tensor_buffers
        print(
            f"[proposal] ExpertState dense recovery image: "
            f"{flat.numel() * flat.element_size() / 1e9:.2f} GB "
            f"(cp_target={self.cp_target}, pinned={flat.is_pinned()})",
            flush=True,
        )
        return flat

    def _build_dense_interval_cache(self) -> None:
        """Build expert-local idx interval tables for test_dram scatter."""

        self._dense_interval_cache.clear()
        for eid, ent in self._entries.items():
            buffers = self._dense_buffers.get(eid, {})
            intervals: List[Tuple[int, int, str]] = []
            for tname, start in ent.global_offsets.items():
                buf = buffers.get(tname)
                if buf is None:
                    continue
                end = int(start) + int(buf["numel"])
                intervals.append((int(start), int(end), tname))
            intervals.sort(key=lambda item: item[0])
            for prev, cur in zip(intervals, intervals[1:]):
                if prev[1] > cur[0]:
                    raise RuntimeError(
                        f"ExpertState dense interval overlap for expert {eid}: "
                        f"{prev[2]} [{prev[0]}, {prev[1]}) overlaps "
                        f"{cur[2]} [{cur[0]}, {cur[1]})"
                    )
            starts = torch.tensor([x[0] for x in intervals], dtype=torch.int64)
            ends = torch.tensor([x[1] for x in intervals], dtype=torch.int64)
            names = [x[2] for x in intervals]
            self._dense_interval_cache[eid] = (starts, ends, names)

    @torch.no_grad()
    def seed_test_dram_dense_image_from_live(
        self,
        fields_by_expert: Dict[ExpertId, List[Dict[str, Any]]],
        *,
        step: int = 0,
    ) -> int:
        """Seed test_dram dense recovery image from live local FP32 views.

        test_dram recovery has no disk LFCL bootstrap.  Without this initial
        image, experts that have not yet fired LFCL are invisible to recovery
        and an in-process synthetic fault can leave exact live state behind.
        """

        if self.cp_target != "test_dram" or not self.commit_lossy_records:
            return 0
        seeded = 0
        for eid, ent in self._entries.items():
            buffers = self._dense_buffers.get(eid)
            fields = fields_by_expert.get(eid)
            if not buffers or not fields:
                continue
            for field in fields:
                name = str(field.get("name", ""))
                buf = buffers.get(name)
                if buf is None:
                    continue
                n = int(buf["numel"])
                sources = (
                    ("p", field.get("param_src")),
                    ("m", field.get("exp_avg")),
                    ("v", field.get("exp_avg_sq")),
                )
                for dst_key, src in sources:
                    dst = buf[dst_key]
                    if src is None:
                        dst.zero_()
                        continue
                    src_flat = src.detach().reshape(-1)
                    if int(src_flat.numel()) != n:
                        raise RuntimeError(
                            "test_dram seed source length mismatch for "
                            f"{eid} {name} {dst_key}: "
                            f"src={src_flat.numel()} dst={n}"
                        )
                    if src_flat.dtype != torch.float32:
                        raise RuntimeError(
                            "test_dram seed requires existing FP32 live "
                            f"source for {eid} {name} {dst_key}; got "
                            f"{src_flat.dtype}"
                        )
                    dst.copy_(src_flat, non_blocking=False)
            ent.step = int(step)
            ent.lossless_step = max(int(ent.lossless_step), int(step))
            seeded += 1
        return int(seeded)

    # ------------------------------------------------------------------
    # I/O dispatch helper — route to expert_writer if present,
    # else execute synchronously (test / offline path).
    # ------------------------------------------------------------------
    def _dispatch(self, fn: "Callable[..., None]", *args: Any, tag: str) -> None:
        # Route commit work to the expert_writer worker so the caller
        # (expert_launcher, in the middle of a lossy checkpoint step) can
        # return immediately to launch this iter's score on compute_stream.
        # Per-iter commits land on expert_writer FIFO in issue order. That is
        # enough for LFCL rotation before subsequent LFSY appends on the same
        # expert; the caller already serialises those submissions.
        del tag
        if self._checkpoint_workers is not None:
            self._checkpoint_workers.submit_expert_write(fn, *args)
        else:
            fn(*args)

    def _fdp_note_expert_write_scheduled(self, step: int) -> None:
        if not fdp_metadata_mmap_enabled():
            return
        with self._fdp_pending_lock:
            step_i = int(step)
            self._fdp_pending_writes_by_step[step_i] = (
                int(self._fdp_pending_writes_by_step.get(step_i, 0)) + 1
            )

    def _fdp_note_expert_write_done(self, step: int) -> None:
        if not fdp_metadata_mmap_enabled():
            return
        mark = False
        step_i = int(step)
        with self._fdp_pending_lock:
            left = int(self._fdp_pending_writes_by_step.get(step_i, 0)) - 1
            if left > 0:
                self._fdp_pending_writes_by_step[step_i] = left
            else:
                self._fdp_pending_writes_by_step.pop(step_i, None)
                if step_i not in self._fdp_done_marked_steps:
                    self._fdp_done_marked_steps.add(step_i)
                    mark = True
        if mark:
            fdp_mark_writer_done(step_i, "expert")

    def fdp_has_pending_writes(self, step: int) -> bool:
        if not fdp_metadata_mmap_enabled():
            return False
        with self._fdp_pending_lock:
            return int(self._fdp_pending_writes_by_step.get(int(step), 0)) > 0

    def _acquire_pinned_aos(self) -> Tuple[torch.Tensor, threading.Event]:
        ready = self._pinned_aos_ready
        ready.wait()
        ready.clear()
        return self._pinned_aos, ready

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def expert_ids(self) -> List[ExpertId]:
        return list(self._entries.keys())

    def lossless_fc_step(self, expert_id: ExpertId) -> int:
        """Iteration at which this expert's most recent LFCL compaction fired
        (-1 if never)."""
        ent = self._entries.get(expert_id)
        return ent.lossless_step if ent is not None else -1

    def has_lossless_fc(self, expert_id: ExpertId) -> bool:
        ent = self._entries.get(expert_id)
        return ent is not None and ent.lossless_step >= 0

    @property
    def total_bytes(self) -> int:
        if self.cp_target == "test_dram":
            return int(self._dense_flat.numel() * self._dense_flat.element_size())
        out = 0
        for e in self._entries.values():
            for p in (e.lossless_path, e.lossy_path):
                try:
                    if p.exists():
                        out += p.stat().st_size
                except OSError:
                    pass
        return out

    # ------------------------------------------------------------------
    # LFCL compaction (GPU live p/m/v -> pinned payload)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compact_lossy_recovery_image(
        self,
        expert_id: ExpertId,
        step: int,
    ) -> None:
        """Test-DRAM metadata-only LFCL hook.

        Disk production must call ``compact_lossy_recovery_image_from_pinned``;
        V2 does not compact LFCL from a resident CPU dense recovery image.
        """

        ent = self._entries.get(expert_id)
        if ent is None:
            return
        ent.lossless_step = int(step)
        ent.step = int(step)
        if self.cp_target == "test_dram":
            self._set_lossy_log_bytes(ent.expert_id, 0)
            return
        raise RuntimeError(
            "V2 disk LFCL compaction requires a pinned GPU payload; call "
            "compact_lossy_recovery_image_from_pinned()."
        )

    @torch.no_grad()
    def compact_lossy_recovery_image_from_pinned(
        self,
        *,
        expert_id: ExpertId,
        step: int,
        pinned_info: Dict[str, Any],
        cuda_event: Optional["torch.cuda.Event"] = None,
    ) -> None:
        """Commit a plan-target LFCL payload already D2H'd from live GPU state."""

        ent = self._entries.get(expert_id)
        if ent is None:
            return
        ent.lossless_step = int(step)
        ent.step = int(step)
        if self.cp_target == "test_dram":
            L, E = ent.expert_id
            self._dispatch(
                self._do_compact_lossy_recovery_image_test_dram_from_pinned,
                ent,
                int(step),
                pinned_info,
                cuda_event,
                tag=f"lfcl_test_dram_compact_L{L:02d}_E{E:02d}",
            )
            return
        header_meta = {
            **self._schema_header_meta(ent),
            "step": int(step),
            "layout": "param_m_v_concat_fp32",
            "source": "gpu_live_pmv",
        }
        header = _pack_header(MAGIC_LFCL, header_meta)
        L, E = ent.expert_id
        self._fdp_note_expert_write_scheduled(int(step))
        self._dispatch(
            self._do_compact_lossy_recovery_image_from_pinned,
            ent,
            int(step),
            header,
            pinned_info,
            cuda_event,
            tag=f"lfcl_gpu_compact_L{L:02d}_E{E:02d}",
        )

    @torch.no_grad()
    def compact_lossy_recovery_images_from_pinned_batch(
        self,
        *,
        step: int,
        batch_info: Dict[str, Any],
        cuda_event: Optional["torch.cuda.Event"] = None,
        commit_payload: bool = True,
    ) -> None:
        """Commit one iteration's fired LFCL payloads from one pinned slab."""

        entries = list(batch_info.get("entries", []))
        if not entries:
            ready_event = batch_info.get("ready_event")
            if ready_event is not None:
                ready_event.set()
            return
        self._fdp_note_expert_write_scheduled(int(step))
        self._dispatch(
            self._do_compact_lossy_recovery_images_from_pinned_batch,
            int(step),
            batch_info,
            cuda_event,
            bool(commit_payload),
            tag=f"lfcl_gpu_compact_batch_{int(step):08d}",
        )

    def _fsync_parent_dir(self, path: Path) -> None:
        import os as _os

        dir_fd = _os.open(str(Path(path).parent), _os.O_RDONLY)
        try:
            _os.fsync(dir_fd)
        finally:
            _os.close(dir_fd)

    def _schema_header_meta(self, ent: "_ExpertEntry") -> Dict[str, Any]:
        meta = {
            "expert_id": [int(ent.expert_id[0]), int(ent.expert_id[1])],
            "tensors": [
                {"name": tn, "shape": list(shape), "numel": int(n)}
                for (tn, shape, n) in ent.tensor_schema
            ],
            "global_offsets": dict(ent.global_offsets),
            "total_numel": int(ent.total_numel),
        }
        if ent.partition_fragments:
            meta["payload_kind"] = "zero12_expert_partition"
            meta["fragments"] = list(ent.partition_fragments)
        return meta

    def _lfcl_header_for_entry(self, ent: "_ExpertEntry", step: int) -> bytes:
        header_meta = {
            **self._schema_header_meta(ent),
            "step": int(step),
            "layout": "param_m_v_concat_fp32",
            "source": "gpu_live_pmv",
        }
        return _pack_header(MAGIC_LFCL, header_meta)

    def _commit_lfcl_payload_from_ready_pinned(
        self,
        ent: "_ExpertEntry",
        step: int,
        pinned_info: Dict[str, Any],
        header: Optional[bytes] = None,
        fdp_batch_active: bool = False,
    ) -> None:
        fdp_batch = bool(fdp_metadata_mmap_enabled()) and not bool(fdp_batch_active)
        if fdp_batch:
            fdp_begin_expert_batch()
        try:
            if ent.lossy_fd is not None:
                close_fd(ent.lossy_fd)
                ent.lossy_fd = None
            pinned = pinned_info["pinned"]
            _require_pinned_cpu_tensor(
                pinned,
                name="LFCL pinned payload source",
                dtype=torch.float32,
            )
            payload_offset_bytes = int(pinned_info.get("payload_offset_bytes", 0))
            payload_bytes = int(pinned_info["payload_bytes"])
            if payload_offset_bytes < HEADER_BLOCK_BYTES:
                raise RuntimeError(
                    "LFCL pinned payload must reserve a leading header block"
                )
            self._write_pinned_lfcl_file(
                ent,
                header if header is not None else self._lfcl_header_for_entry(ent, int(step)),
                pinned,
                payload_offset_bytes,
                payload_bytes,
                step=int(step),
            )
            ent.lossless_step = int(step)
            ent.step = int(step)
            from .proposal_io import unlink_path as _unlink
            _unlink(ent.lossy_path)
            self._fsync_parent_dir(ent.lossy_path)
            self._set_lossy_log_bytes(ent.expert_id, 0)
            self._open_fresh_lossy(ent, step=int(step))
            if fdp_batch:
                fdp_end_expert_batch_stats()
        except BaseException:
            if fdp_batch:
                try:
                    fdp_end_expert_batch_stats()
                except BaseException:
                    pass
            raise

    def _do_compact_lossy_recovery_image_from_pinned(
        self,
        ent: "_ExpertEntry",
        step: int,
        header: bytes,
        pinned_info: Dict[str, Any],
        cuda_event: Optional["torch.cuda.Event"],
    ) -> None:
        ready_event = pinned_info.get("ready_event")
        success = False
        try:
            if cuda_event is not None:
                cuda_event.synchronize()
            self._commit_lfcl_payload_from_ready_pinned(
                ent,
                int(step),
                pinned_info,
                header=header,
            )
            success = True
        finally:
            if ready_event is not None:
                ready_event.set()
            if success:
                self._fdp_note_expert_write_done(int(step))

    def _do_compact_lossy_recovery_images_from_pinned_batch(
        self,
        step: int,
        batch_info: Dict[str, Any],
        cuda_event: Optional["torch.cuda.Event"],
        commit_payload: bool,
    ) -> None:
        ready_event = batch_info.get("ready_event")
        success = False
        fdp_batch = False
        try:
            if not commit_payload:
                success = True
                return
            if cuda_event is not None:
                cuda_event.synchronize()
            fdp_batch = (
                bool(fdp_metadata_mmap_enabled()) and self.cp_target != "test_dram"
            )
            if fdp_batch:
                fdp_begin_expert_batch()
            for item in list(batch_info.get("entries", [])):
                expert_id = tuple(item.get("expert_id"))  # type: ignore[arg-type]
                ent = self._entries.get(expert_id)
                if ent is None:
                    continue
                pinned_info = dict(item["pinned_info"])
                if self.cp_target == "test_dram":
                    self._copy_lfcl_payload_to_dense_image(ent, pinned_info)
                    ent.lossless_step = int(step)
                    ent.step = int(step)
                    self._set_lossy_log_bytes(ent.expert_id, 0)
                else:
                    self._commit_lfcl_payload_from_ready_pinned(
                        ent,
                        int(step),
                        pinned_info,
                        fdp_batch_active=fdp_batch,
                    )
            if fdp_batch:
                fdp_end_expert_batch_stats()
                fdp_batch = False
            success = True
        finally:
            if fdp_batch:
                try:
                    fdp_end_expert_batch_stats()
                except BaseException:
                    pass
            if ready_event is not None:
                ready_event.set()
            if success:
                self._fdp_note_expert_write_done(int(step))

    def _do_compact_lossy_recovery_image_test_dram_from_pinned(
        self,
        ent: "_ExpertEntry",
        step: int,
        pinned_info: Dict[str, Any],
        cuda_event: Optional["torch.cuda.Event"],
    ) -> None:
        ready_event = pinned_info.get("ready_event")
        try:
            if cuda_event is not None:
                cuda_event.synchronize()
            self._copy_lfcl_payload_to_dense_image(ent, pinned_info)
            ent.lossless_step = int(step)
            ent.step = int(step)
            self._set_lossy_log_bytes(ent.expert_id, 0)
        finally:
            if ready_event is not None:
                ready_event.set()

    def _copy_lfcl_payload_to_dense_image(
        self,
        ent: "_ExpertEntry",
        pinned_info: Dict[str, Any],
    ) -> None:
        """Apply a live GPU p/m/v LFCL payload to the test-DRAM image."""

        buffers = self._dense_buffers.get(ent.expert_id)
        if buffers is None:
            return
        pinned = pinned_info["pinned"]
        _require_pinned_cpu_tensor(
            pinned,
            name="test_dram LFCL pinned payload source",
            dtype=torch.float32,
        )
        payload_offset_bytes = int(pinned_info.get("payload_offset_bytes", 0))
        payload_bytes = int(pinned_info["payload_bytes"])
        payload_offset_elems = payload_offset_bytes // 4
        payload_elems = payload_bytes // 4
        payload = pinned.narrow(0, payload_offset_elems, payload_elems)
        cursor = 0
        for tname, _shape, numel in ent.tensor_schema:
            buf = buffers.get(tname)
            n = int(numel)
            if buf is None:
                cursor += n * 3
                continue
            buf["p"].copy_(payload.narrow(0, cursor, n)); cursor += n
            buf["m"].copy_(payload.narrow(0, cursor, n)); cursor += n
            buf["v"].copy_(payload.narrow(0, cursor, n)); cursor += n
        ent.lossless_step = max(int(ent.lossless_step), int(ent.step))

    def _write_pinned_lfcl_file(
        self,
        ent: "_ExpertEntry",
        header: bytes,
        pinned: torch.Tensor,
        payload_offset_bytes: int,
        payload_bytes: int,
        step: int,
    ) -> None:
        import os as _os

        _require_pinned_cpu_tensor(
            pinned,
            name="LFCL pinned file buffer",
            dtype=torch.float32,
        )
        if (int(pinned.data_ptr()) & (_BIN_SECTOR_BYTES - 1)) != 0:
            raise RuntimeError("LFCL pinned file buffer is not 4 KiB aligned")
        if int(payload_offset_bytes) < HEADER_BLOCK_BYTES:
            raise RuntimeError("LFCL pinned file buffer is missing header space")
        total_bytes = int(payload_offset_bytes) + int(payload_bytes)
        if total_bytes > int(pinned.numel() * pinned.element_size()):
            raise RuntimeError("LFCL pinned file buffer is smaller than payload")
        if int(payload_offset_bytes) != HEADER_BLOCK_BYTES:
            raise RuntimeError(
                "LFCL writer expects payload immediately after the header block"
            )
        if backend_kind() == "fdp":
            L, E = ent.expert_id
            write_lfcs_from_pinned(
                ent.lossless_path,
                pinned,
                header,
                int(payload_bytes),
                layer=int(L),
                expert=int(E),
                step=int(step),
                generation=int(step),
            )
            return
        tmp_path = Path(str(ent.lossless_path) + ".tmp")
        fd, _padded = write_header_payload_from_pinned_keep_open(
            tmp_path,
            pinned,
            header,
            int(payload_bytes),
            mode=0o644,
        )
        try:
            fsync_fd(fd)
        finally:
            close_fd(fd)
        _os.replace(tmp_path, ent.lossless_path)
        self._fsync_parent_dir(ent.lossless_path)

    # ------------------------------------------------------------------
    # Lossy FC (append-only per iter)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def append_lossy_records_for_test(
        self,
        expert_id: ExpertId,
        step: int,
        tensor_updates: List[Tuple[str, torch.Tensor, torch.Tensor,
                                   torch.Tensor, torch.Tensor]],
    ) -> int:
        """CPU test/offline append path.

        Production uses ``write_for_iter`` and the GPU AOS
        pipeline. This method keeps the same exact dense-image and
        sector-padded LFSY semantics for unit tests and offline tools.
        """

        ent = self._entries.get(expert_id)
        if ent is None:
            return 0
        tensor_buffers = self._dense_buffers.get(expert_id, {})
        staged: List[Tuple[str, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]] = []
        total_n = 0
        for (tname, idx_local, p, m, v) in tensor_updates:
            if tname not in ent.global_offsets:
                continue
            n = int(idx_local.numel())
            if n <= 0:
                continue
            staged.append((tname, idx_local, p, m, v))
            total_n += n
        if total_n == 0:
            return 0

        recs = np.empty(int(total_n), dtype=_RECORD_DTYPE)
        cursor = 0
        for (tname, idx_local, p, m, v) in staged:
            n = int(idx_local.numel())
            idx_cpu = idx_local.detach().to(
                device="cpu", dtype=torch.int64,
            ).reshape(-1)
            p_cpu = p.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            m_cpu = m.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            v_cpu = v.detach().to(device="cpu", dtype=torch.float32).reshape(-1)

            buf = tensor_buffers.get(tname)
            if buf is not None:
                idx_long = idx_cpu.to(torch.long, copy=False)
                buf["p"].scatter_(0, idx_long, p_cpu)
                buf["m"].scatter_(0, idx_long, m_cpu)
                buf["v"].scatter_(0, idx_long, v_cpu)

            seg = recs[cursor:cursor + n]
            idx_np = (
                idx_cpu.numpy().astype(np.uint64, copy=False)
                + np.uint64(ent.global_offsets[tname])
            )
            if idx_np.size and int(idx_np.max()) > 0xFFFFFFFF:
                raise ValueError(
                    f"lossy record idx exceeds uint32 for expert {expert_id}"
                )
            seg["idx"] = idx_np.astype(np.uint32, copy=False)
            seg["p"] = p_cpu.numpy()
            seg["m"] = m_cpu.numpy()
            seg["v"] = v_cpu.numpy()
            cursor += n

        ent.step = int(step)
        if self.cp_target == "test_dram":
            blob_nbytes = BLOCK_PREFIX_BYTES + int(total_n) * RECORD_BYTES
            padded_nbytes = ((blob_nbytes + _BIN_SECTOR_BYTES - 1)
                             // _BIN_SECTOR_BYTES) * _BIN_SECTOR_BYTES
            self._add_lossy_log_bytes(ent.expert_id, int(padded_nbytes))
            return int(total_n)

        if ent.lossy_fd is None:
            self._open_fresh_lossy(ent, step=int(step))
        blob_nbytes = BLOCK_PREFIX_BYTES + int(total_n) * RECORD_BYTES
        padded_nbytes = ((blob_nbytes + _BIN_SECTOR_BYTES - 1)
                         // _BIN_SECTOR_BYTES) * _BIN_SECTOR_BYTES
        blob = alloc_pinned_uint8_sector_aligned(padded_nbytes)
        blob_np = blob.numpy()
        blob_np[:padded_nbytes] = 0
        blob_np[:BLOCK_PREFIX_BYTES] = np.frombuffer(
            struct.pack("<II", int(total_n), int(step)), dtype=np.uint8,
        )
        rec_bytes = recs.tobytes()
        rec_start = BLOCK_PREFIX_BYTES
        blob_np[rec_start:rec_start + len(rec_bytes)] = np.frombuffer(
            rec_bytes, dtype=np.uint8,
        )
        L, E = ent.expert_id
        append_lossy_log_from_pinned(
            int(ent.lossy_fd), blob, int(blob_nbytes),
            layer=int(L), expert=int(E), step=int(step),
            generation=max(0, int(ent.lossless_step)),
        )
        fsync_fd(int(ent.lossy_fd))
        self._add_lossy_log_bytes(ent.expert_id, int(padded_nbytes))
        return int(total_n)

    # ------------------------------------------------------------------

    def write_for_iter(
        self,
        step: int,
        per_expert_updates: Dict[ExpertId, List[Tuple[str, int, int, int]]],
        compute_event: Optional[Any] = None,
        gpu_refs: Optional[List[torch.Tensor]] = None,
        lossy_state: Optional[Any] = None,
        d2h_event_callback: Optional[Any] = None,
    ) -> int:
        """GPU AOS batch submit (Phase 3). See ``AosEmitter``.

        ``per_expert_updates[eid]`` is a list of static
        ``(tname, bf16_group_idx, src_off, n_records)`` descriptors.
        Direct emission decides selection/fold/write inside the GPU kernel.
        """
        cache_key = id(per_expert_updates)
        prepared = (
            self._prepared_direct_updates
            if self._prepared_direct_updates_key == cache_key else None
        )
        if prepared is None:
            prepared = []
            for eid, tensor_updates in per_expert_updates.items():
                ent = self._entries.get(eid)
                if ent is None:
                    continue
                total_n = 0
                filtered: List[Tuple[str, int, int, int, int]] = []
                for (tname, bf16_group_idx, src_off, n_recs) in tensor_updates:
                    if tname not in ent.global_offsets:
                        continue
                    n_total_recs = int(n_recs)
                    if n_total_recs <= 0:
                        continue
                    filtered.append((
                        tname,
                        ent.global_offsets[tname],
                        int(bf16_group_idx),
                        int(src_off),
                        n_total_recs,
                    ))
                    total_n += n_total_recs
                if total_n == 0:
                    continue
                prepared.append((ent, total_n, filtered))
            self._prepared_direct_updates_key = cache_key
            self._prepared_direct_updates = prepared
        for ent, _total_n, _filtered in prepared:
            ent.step = int(step)
        grand_total = sum(int(total_n) for _ent, total_n, _filtered in prepared)
        if not prepared:
            if gpu_refs is not None:
                gpu_refs.clear()
            return 0
        self._do_lossy_fc_aos_emit(
            int(step), prepared, compute_event, gpu_refs,
            lossy_state, d2h_event_callback,
        )
        return grand_total

    def _do_lossy_fc_aos_emit(
        self,
        step: int,
        prepared,
        compute_event,
        gpu_refs,
        lossy_state: Optional[Any] = None,
        d2h_event_callback: Optional[Any] = None,
    ) -> None:
        """Run GPU AOS emission + D2H on the expert_launcher thread.

        The resulting pinned AOS generation is handed to expert_writer for
        CPU-only persistence. The optimizer fence and writer both receive
        the final D2H event because the single reusable pinned slab cannot be
        overwritten until that copy has completed.
        """
        pinned_aos, pinned_aos_ready = self._acquire_pinned_aos()
        try:
            self._sync_dirty_lossy_log_counters_to_device()
            if lossy_state is not None and getattr(
                lossy_state, "direct_emission_enabled", False,
            ):
                device = None
                for mvph in getattr(lossy_state, "bf16_groups_mvph", ()) or ():
                    if mvph is not None:
                        device = mvph[0].device
                        break
                if device is not None:
                    self._ensure_direct_emission_counters(device)
                lossy_state.direct_record_cursor = self._direct_record_cursor_device
                lossy_state.direct_overflow_count = self._direct_overflow_count_device
            self._direct_feedback_expert_ids = [
                ent.expert_id for ent, _total_n, _filtered in prepared
            ]
            expert_specs, total_aos, d2h_done_event = self._record_pipeline.run(
                step=int(step),
                prepared=prepared,
                compute_event=compute_event,
                gpu_refs=gpu_refs,
                lossy_state=lossy_state,
                pinned_aos=pinned_aos,
                layout_mirror=self._pinned_aos_layout_mirror,
                copy_to_host=not self.test_skip_d2h,
            )
            ready_for_optimizer = self._record_direct_emission_feedback(
                step=int(step),
                n_experts=len(expert_specs),
                lossy_state=lossy_state,
                ready_event=d2h_done_event,
            )
            if d2h_event_callback is not None:
                d2h_event_callback(
                    ready_for_optimizer
                    if ready_for_optimizer is not None
                    else d2h_done_event
                )
        except BaseException:
            pinned_aos_ready.set()
            raise
        if total_aos == 0:
            pinned_aos_ready.set()
            return
        if not self.commit_lossy_records:
            # write-off+D2H-on exposes the final D2H/feedback event to the
            # optimizer fence above.  The next proposal dispatch happens only
            # after that fence and optimizer step, so a no-commit expert_writer
            # task that only synchronizes the same event before releasing the
            # reusable slab is redundant and can delay PP runs.
            pinned_aos_ready.set()
            return
        self._fdp_note_expert_write_scheduled(int(step))
        self._dispatch(
            self._persist_lossy_aos_from_pinned,
            int(step), expert_specs, prepared, pinned_aos,
            self._pinned_aos_layout_mirror, pinned_aos_ready,
            d2h_done_event,
            tag=f"lossy_persist_iter{int(step)}",
        )

    def _record_direct_emission_feedback(
        self,
        *,
        step: int,
        n_experts: int,
        lossy_state: Optional[Any],
        ready_event: Optional["torch.cuda.Event"],
    ) -> Optional["torch.cuda.Event"]:
        if (
            lossy_state is None
            or not bool(getattr(lossy_state, "direct_emission_enabled", False))
            or self._direct_record_cursor_device is None
            or self._direct_overflow_count_device is None
            or self._direct_record_cursor_pinned is None
            or self._direct_overflow_count_pinned is None
            or self._direct_total_th_pinned is None
            or self._direct_feedback_event is None
        ):
            return ready_event
        total_th = getattr(lossy_state, "total_th", None)
        if total_th is None:
            return ready_event
        n = max(0, min(int(n_experts), int(self._direct_record_cursor_device.numel())))
        if n <= 0:
            return ready_event
        capacities = getattr(lossy_state, "aos_layout_capacities", None)
        offsets = getattr(lossy_state, "aos_layout_offsets", None)
        from ..runtime import get_d2h_stream
        d2h_stream = get_d2h_stream()
        with torch.cuda.stream(d2h_stream):
            if ready_event is not None:
                d2h_stream.wait_event(ready_event)
            self._direct_record_cursor_pinned.narrow(0, 0, n).copy_(
                self._direct_record_cursor_device.narrow(0, 0, n),
                non_blocking=True,
            )
            self._direct_overflow_count_pinned.narrow(0, 0, n).copy_(
                self._direct_overflow_count_device.narrow(0, 0, n),
                non_blocking=True,
            )
            self._direct_total_th_pinned.narrow(0, 0, 1).copy_(
                total_th.reshape(1),
                non_blocking=True,
            )
            if (
                self._direct_layout_feedback_enabled
                and self._direct_capacity_pinned is not None
                and capacities is not None
                and int(capacities.numel()) >= n
            ):
                self._direct_capacity_pinned.narrow(0, 0, n).copy_(
                    capacities.narrow(0, 0, n),
                    non_blocking=True,
                )
            if (
                self._direct_layout_feedback_enabled
                and self._direct_offset_pinned is not None
                and offsets is not None
                and int(offsets.numel()) >= n
            ):
                self._direct_offset_pinned.narrow(0, 0, n).copy_(
                    offsets.narrow(0, 0, n),
                    non_blocking=True,
                )
            self._direct_feedback_event.record(d2h_stream)
        self._direct_feedback_step = int(step)
        self._direct_feedback_n_experts = int(n)
        self._direct_feedback_tau_version = int(getattr(lossy_state, "tau_version", -1))
        self._direct_feedback_tau_h_epoch = int(getattr(lossy_state, "tau_h_epoch", -1))
        self._direct_feedback_tau_source_step = int(
            getattr(lossy_state, "tau_source_step", -1)
        )
        self._direct_feedback_tau_apply_from_step = int(
            getattr(lossy_state, "tau_apply_from_step", -1)
        )
        self._direct_feedback_tau_lr_over_bc1 = float(
            getattr(lossy_state, "tau_lr_over_bc1", 0.0)
        )
        return self._direct_feedback_event

    def consume_direct_emission_feedback(self) -> Optional[Dict[str, Any]]:
        event = self._direct_feedback_event
        step = int(getattr(self, "_direct_feedback_step", -1))
        if (
            event is None
            or step < 0
            or step == int(getattr(self, "_direct_feedback_consumed_step", -1))
        ):
            return None
        try:
            if not event.query():
                return None
        except Exception:
            return None
        n = max(0, int(getattr(self, "_direct_feedback_n_experts", 0)))
        if (
            n <= 0
            or self._direct_record_cursor_pinned is None
            or self._direct_overflow_count_pinned is None
            or self._direct_total_th_pinned is None
        ):
            return None
        cursor = self._direct_record_cursor_pinned.narrow(0, 0, n)
        overflow = self._direct_overflow_count_pinned.narrow(0, 0, n)
        cursor_list = [int(x) for x in cursor.tolist()]
        overflow_list = [int(x) for x in overflow.tolist()]
        selected = int(sum(cursor_list))
        dropped = int(sum(overflow_list))
        written = max(0, selected - dropped)
        tau = float(self._direct_total_th_pinned[0].item())
        self._direct_feedback_consumed_step = int(step)
        feedback = {
            "step": int(step),
            "selected": int(selected),
            "written": int(written),
            "overflow": int(dropped),
            "tau": float(tau),
            "tau_version": int(getattr(self, "_direct_feedback_tau_version", -1)),
            "tau_h_epoch": int(getattr(self, "_direct_feedback_tau_h_epoch", -1)),
            "tau_source_step": int(
                getattr(self, "_direct_feedback_tau_source_step", -1)
            ),
            "tau_apply_from_step": int(
                getattr(self, "_direct_feedback_tau_apply_from_step", -1)
            ),
            "tau_lr_over_bc1": float(
                getattr(self, "_direct_feedback_tau_lr_over_bc1", 0.0)
            ),
            "n_experts": int(n),
            "selected_by_expert": cursor_list,
            "overflow_by_expert": overflow_list,
        }
        direct_feedback_log = os.environ.get("PROPOSAL_DIRECT_FEEDBACK_LOG", "")
        if direct_feedback_log:
            feedback["expert_ids"] = [
                [int(eid[0]), int(eid[1])]
                for eid in self._direct_feedback_expert_ids[:n]
            ]
        if (
            self._direct_layout_feedback_enabled
            and self._direct_capacity_pinned is not None
            and self._direct_offset_pinned is not None
        ):
            capacity = self._direct_capacity_pinned.narrow(0, 0, n)
            offset = self._direct_offset_pinned.narrow(0, 0, n)
            feedback.update({
                "capacity_by_expert": [int(x) for x in capacity.tolist()],
                "aos_offset_by_expert": [int(x) for x in offset.tolist()],
                "max_records_by_expert": [
                    int(self._expert_max_records_by_id.get(eid, 0))
                    for eid in self._direct_feedback_expert_ids[:n]
                ],
                "expert_ids": [
                    [int(eid[0]), int(eid[1])]
                    for eid in self._direct_feedback_expert_ids[:n]
                ],
            })
        if direct_feedback_log:
            self._append_direct_feedback_log(direct_feedback_log, feedback)
        self._latest_direct_feedback = feedback
        return feedback

    def _append_direct_feedback_log(
        self,
        path_spec: str,
        feedback: Dict[str, Any],
    ) -> None:
        import json

        try:
            rank = int(os.environ.get("RANK", "0") or "0")
        except ValueError:
            rank = 0
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
        except ValueError:
            local_rank = 0
        path_s = str(path_spec)
        if "{rank}" in path_s or "{local_rank}" in path_s:
            path_s = path_s.format(rank=rank, local_rank=local_rank)
        else:
            base, ext = os.path.splitext(path_s)
            path_s = f"{base}_rank{rank:03d}{ext or '.jsonl'}"
        path = Path(path_s)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "rank": int(rank),
            "local_rank": int(local_rank),
            **feedback,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _persist_lossy_aos_from_pinned(
        self,
        step: int,
        expert_specs,
        prepared,
        pinned_aos: torch.Tensor,
        layout_mirror: PinnedAosLayoutMirror,
        pinned_aos_ready: threading.Event,
        d2h_done_event: Optional["torch.cuda.Event"] = None,
    ) -> None:
        """CPU-only expert_writer task: append pinned AOS blobs to LFSY."""

        import os as _os_persist
        _profile = _os_persist.environ.get("PROPOSAL_PERSIST_PROFILE", "") in (
            "1", "true", "TRUE",
        )
        if _profile:
            import time as _time_profile
            _t0 = _time_profile.perf_counter()
        else:
            _time_profile = None
            _t0 = 0.0
        _t_ready = (
            _time_profile.perf_counter() if _profile and _time_profile is not None else 0.0
        )
        success = False
        try:
            if not self.commit_lossy_records and self.test_skip_d2h:
                if _profile and _time_profile is not None:
                    _t_done = _time_profile.perf_counter()
                    print(
                        f"[proposal] persist_profile @iter{int(step)}: "
                        f"commit_skipped={(_t_done - _t_ready) * 1000.0:.1f}ms "
                        f"target=proposal-test-skip-expert-commit",
                        flush=True,
                    )
                return
            if not self.commit_lossy_records:
                # write-off+D2H-on still fences the optimizer on d2h_done_event
                # before p/m/v can be mutated again. Since no CPU consumer reads
                # this slab in write-off mode, do not make the writer thread
                # block on the D2H event just to release the next lease.
                if _profile and _time_profile is not None:
                    _t_done = _time_profile.perf_counter()
                    print(
                        f"[proposal] persist_profile @iter{int(step)}: "
                        f"commit_skipped={(_t_done - _t_ready) * 1000.0:.1f}ms "
                        f"target=proposal-test-skip-expert-commit",
                        flush=True,
                    )
                return
            if d2h_done_event is not None:
                d2h_done_event.synchronize()
            _t_ready = (
                _time_profile.perf_counter()
                if _profile and _time_profile is not None else 0.0
            )
            layout_mirror.update_from_header(
                pinned_aos=pinned_aos,
                prepared=prepared,
                max_records_by_expert=self._expert_max_records_by_id,
            )
            if self.cp_target == "test_dram":
                self._scatter_lossy_aos_to_dense(
                    step,
                    layout_mirror, prepared, pinned_aos,
                )
                if _profile and _time_profile is not None:
                    _t_done = _time_profile.perf_counter()
                    print(
                        f"[proposal] persist_profile @iter{int(step)}: "
                        f"scatter={(_t_done - _t_ready) * 1000.0:.1f}ms "
                        f"target=test_dram",
                        flush=True,
                )
                return
            fdp_batch = bool(fdp_metadata_mmap_enabled())
            if fdp_batch:
                fdp_begin_expert_batch()
            try:
                for row in layout_mirror.rows():
                    if int(row.aos_byte_off) < 0:
                        continue
                    ent = self._entries.get(row.expert_id)
                    if ent is None:
                        raise RuntimeError(
                            f"missing expert entry for AOS layout row "
                            f"{row.expert_id}"
                        )
                    count = read_lfsy_prefix_count(
                        pinned_aos,
                        aos_byte_off=row.aos_byte_off,
                        record_capacity=row.record_capacity,
                        prefix_bytes=BLOCK_PREFIX_BYTES,
                    )
                    if count <= 0:
                        continue
                    append_nbytes = self._lossy_blob_padded_bytes_for_count(count)
                    if int(row.blob_padded_bytes) > 0 and append_nbytes > int(row.blob_padded_bytes):
                        raise RuntimeError(
                            f"GPU-stamped count exceeds reserved blob: "
                            f"expert={ent.expert_id} count={count} "
                            f"append={append_nbytes} reserved={row.blob_padded_bytes}"
                        )
                    if ent.lossy_fd is None:
                        self._open_fresh_lossy(
                            ent, step=int(step), reset_counter=False,
                        )
                    blob = pinned_aos.narrow(0, row.aos_byte_off, append_nbytes)
                    L, E = ent.expert_id
                    append_lossy_log_from_pinned(
                        int(ent.lossy_fd), blob, int(append_nbytes),
                        layer=int(L), expert=int(E), step=int(step),
                        generation=max(0, int(ent.lossless_step)),
                    )
                    fsync_fd(int(ent.lossy_fd))
                    self._add_lossy_log_bytes_from_writer(
                        ent.expert_id,
                        int(append_nbytes),
                    )
                if fdp_batch:
                    fdp_end_expert_batch_stats()
                    fdp_batch = False
            except BaseException:
                if fdp_batch:
                    try:
                        fdp_end_expert_batch_stats()
                    except BaseException:
                        pass
                raise
            if _profile and _time_profile is not None:
                _t_done = _time_profile.perf_counter()
                print(
                    f"[proposal] persist_profile @iter{int(step)}: "
                    f"scatter=0.0ms "
                    f"write_fsync={(_t_done - _t_ready) * 1000.0:.1f}ms "
                    f"target=disk",
                    flush=True,
                )
            success = True
        finally:
            pinned_aos_ready.set()
            if success:
                self._fdp_note_expert_write_done(int(step))

    def _lossy_blob_padded_bytes_for_count(self, count: int) -> int:
        raw = BLOCK_PREFIX_BYTES + max(0, int(count)) * RECORD_BYTES
        sector = int(_BIN_SECTOR_BYTES)
        return ((raw + sector - 1) // sector) * sector

    def _scatter_lossy_aos_to_dense(
        self,
        step: int,
        layout_mirror: PinnedAosLayoutMirror,
        prepared,
        pinned_aos: torch.Tensor,
    ) -> None:
        """Apply selected lossy records to the host dense recovery image."""

        for row, (ent, _total_n, _filtered) in zip(layout_mirror.rows(), prepared):
            if int(row.aos_byte_off) < 0:
                continue
            n_recs_total = read_lfsy_prefix_count(
                pinned_aos,
                aos_byte_off=row.aos_byte_off,
                record_capacity=row.record_capacity,
                prefix_bytes=BLOCK_PREFIX_BYTES,
            )
            if n_recs_total == 0:
                continue
            self._scatter_records_to_dense_with_cpp(
                ent,
                pinned_aos,
                int(row.aos_byte_off) + BLOCK_PREFIX_BYTES,
                n_recs_total,
            )
            ent.step = int(step)
            self._add_lossy_log_bytes_from_writer(
                ent.expert_id,
                self._lossy_blob_padded_bytes_for_count(n_recs_total),
            )

    def _scatter_records_to_dense_with_cpp(
        self,
        ent: "_ExpertEntry",
        pinned_aos: torch.Tensor,
        record_byte_offset: int,
        n_records: int,
    ) -> None:
        """Scatter self-contained records using expert-local idx intervals."""

        if int(n_records) == 0:
            return
        buffers = self._dense_buffers.get(ent.expert_id)
        if buffers is None:
            return
        starts, ends, names = self._dense_interval_cache.get(
            ent.expert_id, (
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
                [],
            ),
        )
        if int(starts.numel()) == 0:
            return
        p_tensors: List[torch.Tensor] = []
        m_tensors: List[torch.Tensor] = []
        v_tensors: List[torch.Tensor] = []
        for tname in names:
            buf = buffers.get(tname)
            if buf is None:
                raise RuntimeError(
                    f"ExpertState dense scatter missing buffer for {tname!r} "
                    f"in expert {ent.expert_id}"
                )
            p_tensors.append(buf["p"])
            m_tensors.append(buf["m"])
            v_tensors.append(buf["v"])
        try:
            scatter_lossy_records_to_dense_by_intervals(
                pinned_aos,
                int(record_byte_offset),
                int(n_records),
                starts,
                ends,
                p_tensors,
                m_tensors,
                v_tensors,
            )
        except RuntimeError as exc:
            import re as _re
            import struct as _struct

            detail = ""
            m = _re.search(r"record (\d+):", str(exc))
            if m is not None:
                rec_i = int(m.group(1))
                byte_i = int(record_byte_offset) + rec_i * RECORD_BYTES
                if 0 <= byte_i and byte_i + RECORD_BYTES <= int(pinned_aos.numel()):
                    raw = bytes(
                        pinned_aos.narrow(0, byte_i, RECORD_BYTES).cpu().numpy()
                    )
                    prev_raw = b""
                    if byte_i - RECORD_BYTES >= 0:
                        prev_raw = bytes(
                            pinned_aos.narrow(
                                0, byte_i - RECORD_BYTES, RECORD_BYTES,
                            ).cpu().numpy()
                        )
                    words = _struct.unpack("<IIII", raw)
                    floats = _struct.unpack("<ffff", raw)
                    prev_words = (
                        list(_struct.unpack("<IIII", prev_raw))
                        if prev_raw else []
                    )
                    prev_floats = (
                        [round(x, 8) for x in _struct.unpack("<ffff", prev_raw)]
                        if prev_raw else []
                    )
                    detail = (
                        f"; expert={ent.expert_id} record={rec_i} "
                        f"byte_off={byte_i} words={list(words)} "
                        f"floats={[round(x, 8) for x in floats]} "
                        f"prev_words={prev_words} prev_floats={prev_floats} "
                        f"interval_minmax=({int(starts[0].item())},"
                        f"{int(ends[-1].item())}) n_records={int(n_records)}"
                    )
            raise RuntimeError(f"{exc}{detail}") from exc

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    def _do_fsync_lossy(self, ent: "_ExpertEntry") -> None:
        if ent.lossy_fd is not None:
            fsync_fd(ent.lossy_fd)

    def flush(self, only_eids: Optional[List[ExpertId]] = None) -> None:
        """Drain pending async jobs, then fsync every open lossy FC fd.

        LFCL files are atomically persisted at compaction time, so this
        method only needs to cover the lossy append fds.
        """
        # Drain expert_writer first so pending writes land before
        # we fsync.
        if self._checkpoint_workers is not None:
            try:
                self._checkpoint_workers.drain_expert_write()
            except Exception:
                pass
        target = set(only_eids) if only_eids is not None else None
        for eid, ent in self._entries.items():
            if target is not None and eid not in target:
                continue
            if ent.lossy_fd is not None:
                L, E = ent.expert_id
                self._dispatch(
                    self._do_fsync_lossy, ent,
                    tag=f"fsync_lossy_L{L:02d}_E{E:02d}",
                )
        if self._checkpoint_workers is not None:
            try:
                self._checkpoint_workers.drain_expert_write()
            except Exception:
                pass

    def shutdown(self) -> None:
        """Drain pending I/O and close every open fd. Idempotent.

        Does NOT shut down ``self._checkpoint_workers``; the proposal pipeline
        owns that worker pool and tears it down separately.
        """
        try:
            self.flush()
        except Exception as exc:
            print(f"[expert_state] shutdown flush failed: {exc}", flush=True)
        # Close remaining fds. Safe only after flush() drained all jobs.
        for ent in self._entries.values():
            if ent.lossy_fd is not None:
                close_fd(ent.lossy_fd)
                ent.lossy_fd = None

    def read(self, expert_id: ExpertId) -> Optional[Dict[str, torch.Tensor]]:
        """Return this expert's dense recovery image for recovery."""

        ent = self._entries.get(expert_id)
        if ent is None or int(ent.lossless_step) < 0:
            return None
        tensor_buffers = self._dense_buffers.get(expert_id)
        if tensor_buffers is None:
            return None
        out: Dict[str, torch.Tensor] = {}
        if ent.partition_fragments:
            out["_payload_kind"] = "zero12_expert_partition"
            out["_fragments"] = list(ent.partition_fragments)
            for frag in ent.partition_fragments:
                frag_idx = int(frag["index"])
                tname = str(frag["fragment_name"])
                buf = tensor_buffers.get(tname)
                if buf is None:
                    continue
                out[f"frag{frag_idx}:param"] = buf["p"].view(-1)
                out[f"frag{frag_idx}:exp_avg"] = buf["m"].view(-1)
                out[f"frag{frag_idx}:exp_avg_sq"] = buf["v"].view(-1)
            out["_rolling_step"] = torch.tensor(
                int(ent.step),
                dtype=torch.long,
            )
            return out
        for tname, buf in tensor_buffers.items():
            shape = torch.Size(buf["shape"])
            out[f"{tname}:param"] = buf["p"].view(shape)
            out[f"{tname}:exp_avg"] = buf["m"].view(shape)
            out[f"{tname}:exp_avg_sq"] = buf["v"].view(shape)
        out["_rolling_step"] = torch.tensor(
            int(ent.step),
            dtype=torch.long,
        )
        return out

    def lossy_log_bytes(self, expert_id: ExpertId) -> int:
        idx = self._lossy_log_counter_index.get(expert_id)
        if idx is not None:
            return int(self._lossy_log_bytes_pinned[int(idx)].item())
        return int(self._lossy_log_bytes_by_expert.get(expert_id, 0))

    @property
    def lossy_log_bytes_device(self) -> Optional[torch.Tensor]:
        return self._lossy_log_bytes_device

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _open_fresh_lossy(
        self,
        ent: _ExpertEntry,
        step: int,
        *,
        reset_counter: bool = True,
    ) -> None:
        header_meta = {
            **self._schema_header_meta(ent),
            "start_step": int(step),
            "record_bytes": RECORD_BYTES,
        }
        header = _pack_header(MAGIC_LFSY, header_meta)
        L, E = ent.expert_id
        ent.lossy_fd = open_lossy_log_for_append(
            ent.lossy_path,
            header,
            layer=int(L),
            expert=int(E),
            step=int(step),
            generation=max(0, int(ent.lossless_step)),
        )
        fsync_fd(int(ent.lossy_fd))
        self._fsync_parent_dir(ent.lossy_path)
        ent.lossy_start_step = int(step)
        if reset_counter:
            self._set_lossy_log_bytes(ent.expert_id, 0)


# Phase 3 removed the per-tensor numpy helpers (``_to_fp32_numpy`` /
# ``_to_i64_numpy``) that supported the CPU SoA → AoS interleave path.
# The GPU AOS kernel reads directly from the per-BF16-group GPU SoA.


# ----------------------------------------------------------------------------
# Recovery reader
# ----------------------------------------------------------------------------
def read_expert_state_from_blobs(
    lfcl_blob: Optional[torch.Tensor],
    lfsy_blob: Optional[torch.Tensor],
    *,
    layer: int,
    expert: int,
    source: str = "expert_state",
) -> Dict[str, Any]:
    """Reconstruct one expert from already-read LFCL/LFSY byte blobs."""

    def _header_bytes(blob: torch.Tensor) -> bytes:
        n = min(int(blob.numel()), int(HEADER_BLOCK_BYTES))
        return bytes(blob.narrow(0, 0, n).numpy().tobytes())

    from .proposal_io import (
        load_lfcl_payload_interleaved,
        replay_lfsy_log_into_dense,
    )

    if lfcl_blob is not None and int(lfcl_blob.numel()) > 0:
        header_block = _header_bytes(lfcl_blob)
        magic, _ver, meta = _unpack_header(header_block)
        if magic != MAGIC_LFCL:
            raise ValueError(
                f"expected LFCL in {source} L{layer}/E{expert}, got {magic!r}"
            )
        tensors_meta = meta["tensors"]
        total_numel = int(meta["total_numel"])
        p_flat = torch.empty(total_numel, dtype=torch.float32)
        m_flat = torch.empty(total_numel, dtype=torch.float32)
        v_flat = torch.empty(total_numel, dtype=torch.float32)
        numels = torch.tensor(
            [int(entry["numel"]) for entry in tensors_meta],
            dtype=torch.int64,
        )
        load_lfcl_payload_interleaved(
            lfcl_blob,
            payload_offset_bytes=HEADER_BLOCK_BYTES,
            numels_i64=numels,
            p_flat=p_flat,
            m_flat=m_flat,
            v_flat=v_flat,
        )
        base_step = int(meta.get("step", 0))
    elif lfsy_blob is not None and int(lfsy_blob.numel()) > 0:
        # No LFCL compaction baseline yet: recover schema from LFSY header and
        # seed dense tensors with zeros before replaying sparse records.
        header_block = _header_bytes(lfsy_blob)
        magic, _ver, meta = _unpack_header(header_block)
        if magic != MAGIC_LFSY:
            return {"_rolling_step": 0}
        tensors_meta = meta["tensors"]
        total_numel = int(meta["total_numel"])
        p_flat = torch.zeros(total_numel, dtype=torch.float32)
        m_flat = torch.zeros(total_numel, dtype=torch.float32)
        v_flat = torch.zeros(total_numel, dtype=torch.float32)
        base_step = 0
    else:
        return {"_rolling_step": 0}

    max_step = base_step
    if lfsy_blob is not None and int(lfsy_blob.numel()) > 0:
        header_block = _header_bytes(lfsy_blob)
        magic, _ver, _meta = _unpack_header(header_block)
        if magic != MAGIC_LFSY:
            return {"_rolling_step": int(max_step)}
        replay = replay_lfsy_log_into_dense(
            lfsy_blob,
            total_numel=int(total_numel),
            p_flat=p_flat,
            m_flat=m_flat,
            v_flat=v_flat,
        )
        if int(replay.get("max_step", 0)) > max_step:
            max_step = int(replay["max_step"])

    payload: Dict[str, Any] = {"_rolling_step": int(max_step)}
    payload_kind = str(meta.get("payload_kind", ""))
    if payload_kind == "zero12_expert_partition":
        fragments = list(meta.get("fragments", []) or [])
        payload["_payload_kind"] = payload_kind
        payload["_fragments"] = fragments
        cursor = 0
        for entry_idx, entry in enumerate(tensors_meta):
            frag_idx = (
                int(fragments[entry_idx]["index"])
                if entry_idx < len(fragments) else int(entry_idx)
            )
            numel = int(entry["numel"])
            payload[f"frag{frag_idx}:param"] = p_flat.narrow(0, cursor, numel)
            payload[f"frag{frag_idx}:exp_avg"] = m_flat.narrow(0, cursor, numel)
            payload[f"frag{frag_idx}:exp_avg_sq"] = v_flat.narrow(0, cursor, numel)
            cursor += numel
        return payload

    cursor = 0
    for entry in tensors_meta:
        tname = entry["name"]
        numel = int(entry["numel"])
        shape = torch.Size(entry["shape"])
        payload[f"{tname}:param"] = p_flat.narrow(0, cursor, numel).view(shape)
        payload[f"{tname}:exp_avg"] = m_flat.narrow(0, cursor, numel).view(shape)
        payload[f"{tname}:exp_avg_sq"] = v_flat.narrow(0, cursor, numel).view(shape)
        cursor += numel
    return payload


def read_expert_state(
    expert_dir: Path,
    layer: int,
    expert: int,
) -> Dict[str, Any]:
    """Reconstruct one expert's state from its filesystem LFCL/LFSY pair."""
    expert_dir = Path(expert_dir)
    lossless_path = expert_dir / _lossless_name(layer, expert)
    lossy_path = expert_dir / _lossy_name(layer, expert)

    from .proposal_io import file_size as _fsize, read_file_aligned

    lfcl_blob: Optional[torch.Tensor] = None
    lfsy_blob: Optional[torch.Tensor] = None
    lossless_size = _fsize(lossless_path)
    if lossless_size >= 0:
        lfcl_blob = read_file_aligned(lossless_path, int(lossless_size))
    lossy_size = _fsize(lossy_path)
    if lossy_size >= 0:
        lfsy_blob = read_file_aligned(lossy_path, int(lossy_size))

    return read_expert_state_from_blobs(
        lfcl_blob,
        lfsy_blob,
        layer=int(layer),
        expert=int(expert),
        source=str(expert_dir),
    )
