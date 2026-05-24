"""Per-expert state storage: lossy FC (append-only log) + round-robin lossless FC.

This module implements the CLAUDE.md "Expert 저장 전략" spec:

* **Lossy FC (`lossy_expert_L{L:02d}_E{E:02d}.bin`)** — an append-only log.
  Every iteration, the top-K% element selections for this expert are
  packed into a single per-iter block and appended. File is NOT truncated
  during normal training; it grows bounded by the lossless-FC rotation
  cycle (``D × ⌈100/F⌉`` iters) and is unlinked at the next lossless FC
  of this expert.

* **Lossless FC (`fc_expert_L{L:02d}_E{E:02d}.bin`)** — a full param +
  optimizer snapshot. Written every ``D`` iterations for ``F%`` of
  experts in round-robin order. After a successful fsync + atomic rename,
  the previous lossless FC file AND the previous lossy FC file are
  unlinked, and a fresh (empty-payload) lossy FC file is created so the
  next iter's lossy-FC append starts clean.

Binary formats:

``LFCL`` (Lossless FC):
    [magic 4B "LFCL"][version u32][header_len u32][pickled meta]
    [pad to HEADER_BLOCK_BYTES = 4096]
    [fp32 payload: for each tensor in schema order,
        param_flat (numel × 4B), m_flat, v_flat]

``LFSY`` (Lossy FC append log):
    [magic 4B "LFSY"][version u32][header_len u32][pickled meta]
    [pad to HEADER_BLOCK_BYTES]
    per-iter block (repeated):
        [count u32][step u32]
        [count × 16B records: idx i32, p f32, m f32, v f32]

Recovery reads LFCL fully, then replays LFSY block-by-block in step order
(last write wins per index).

Write-path performance
======================

The append hot path is optimised to avoid ``torch.cat`` + bytes
concatenation overhead that plagued the earlier log-structured design:

* Pre-allocate a single numpy structured-dtype array per expert per iter.
* Fill columns directly from ``tensor.numpy()`` (zero-copy for CPU
  contiguous fp32/int32 tensors).
* Single ``os.write`` syscall flushes the (count, step) prefix and the
  AoS records — both assembled directly into a pinned host buffer so
  the future C++ SSD-passthrough writer can submit the same buffer
  with io_uring + O_DIRECT without any extra copy.

This keeps training-thread overhead in the write path to roughly one
numpy allocation per expert per iter, plus a single syscall.
"""
from __future__ import annotations

import os
import pickle
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .proposal_io import (
    HEADER_BLOCK_BYTES as _BIN_HEADER_BLOCK_BYTES,
    SECTOR_BYTES as _BIN_SECTOR_BYTES,
    alloc_pinned_uint8_sector_aligned,
    append_lossy_log_from_pinned,
    close_fd,
    fsync_fd,
    open_lossy_log_for_append,
    write_lossless_from_pinned,
)
from . import gpu_aos as _gpu_aos
from .segment_pipeline import SegmentPipeline, DEFAULT_SEGMENT_BYTES


ExpertId = Tuple[int, int]  # (layer, expert_idx)

# ---- Binary format constants -----------------------------------------------
MAGIC_LFCL = b"LFCL"  # lossless FC
MAGIC_LFSY = b"LFSY"  # lossy FC append log
FORMAT_VERSION: int = 1
# Reuse the file-format header block size from binary_io so PRPTENS,
# LFCL, and LFSY all share a single sector convention. 64 KiB is the
# current value (sized to fit the largest header — non-expert FC).
HEADER_BLOCK_BYTES: int = _BIN_HEADER_BLOCK_BYTES
RECORD_BYTES: int = 20  # int64 idx + fp32 p + fp32 m + fp32 v
BLOCK_PREFIX_BYTES: int = 8  # count u32 + step u32

# Filename conventions.
_LOSSLESS_RE = re.compile(r"^fc_expert_L(\d+)_E(\d+)\.bin$")
_LOSSY_RE = re.compile(r"^lossy_expert_L(\d+)_E(\d+)\.bin$")

# Per-record numpy structured dtype (little-endian, packed to 20B).
# idx is int64: per-expert combined offset = tname_base_offset + idx_local
# overflows int32 once per-expert (param+m+v) elements exceed ~2.1B.
_RECORD_DTYPE = np.dtype([
    ("idx", "<i8"),
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
    """Bookkeeping for one expert's lossless + lossy files."""
    expert_id: ExpertId
    tensor_schema: List[Tuple[str, torch.Size, int]]
    global_offsets: Dict[str, int]
    total_numel: int
    lossless_path: Path
    lossy_path: Path
    # Open fd on lossy file (append mode), None until first lossy append.
    lossy_fd: Optional[int] = None
    lossy_start_step: int = -1
    lossless_step: int = -1  # last step at which lossless FC fired; -1 if never.
    step: int = -1


# ----------------------------------------------------------------------------
# Main class: ExpertState
# ----------------------------------------------------------------------------
class ExpertState:
    """Per-expert lossy FC append-only log + lossless FC snapshot.

    Call pattern from the training loop (integrated via FCManager +
    ImportanceTracker):

        state = ExpertState(expert_tensor_map, output_dir, fresh=True)
        # ... each iter:
        #   1) FCManager determines experts doing lossless FC this iter.
        for eid in lossless_fc_expert_ids_this_iter:
            state.lossless_fc(eid, step=iter, initial_state={tname: (p, m, v)})
        # 2) ImportanceTracker computes top-K across remaining experts.
        for eid, tensor_updates in selections.items():
            state.lossy_fc_append(eid, step=iter, tensor_updates=tensor_updates)

    At the end: state.shutdown() to close fds.
    """

    def __init__(
        self,
        expert_tensor_map: Dict[ExpertId, List[Tuple[str, torch.Size, int]]],
        output_dir: Path,
        *,
        fresh: bool = True,
        async_writer: Optional[Any] = None,
    ) -> None:
        """
        Args:
            expert_tensor_map: per-expert tensor schema.
            output_dir: where LFCL / LFSY files live.
            fresh: purge any existing files at that path.
            async_writer: an ``AsyncWriter`` whose ``submit_raw(fn, *args)``
                dispatches raw-callable I/O jobs on its dedicated raw
                worker thread. When None, writes execute synchronously on
                the caller thread — intended for unit tests / offline
                tools where background I/O isn't desirable.
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[ExpertId, _ExpertEntry] = {}
        self._async_writer = async_writer

        # Phase 5: pre-allocate pinned_aos at half the total expert state
        # bytes (Σ per-expert numel × 3 fields × 4 bytes). This bounds
        # the per-iter AOS payload regardless of topk_ratio / outlier
        # spikes; raises if exceeded. Sized once, reused across iters.
        _total_param_numel = sum(
            int(numel)
            for schema in expert_tensor_map.values()
            for (_tname, _shape, numel) in schema
        )
        _total_state_bytes = _total_param_numel * 12  # p + m + v, fp32
        _pinned_aos_bytes = _total_state_bytes // 2
        _sect = int(_BIN_SECTOR_BYTES)
        _pinned_aos_bytes = (
            (_pinned_aos_bytes + _sect - 1) // _sect
        ) * _sect
        if _pinned_aos_bytes <= 0:
            # Empty expert map (defensive). Allocate a single sector so
            # the pipeline's size check stays well-defined.
            _pinned_aos_bytes = _sect
        self._pinned_aos: torch.Tensor = alloc_pinned_uint8_sector_aligned(
            _pinned_aos_bytes
        )
        print(
            f"[proposal] ExpertState pinned_aos pre-alloc: "
            f"{_pinned_aos_bytes / 1e9:.2f} GB "
            f"(half of {_total_state_bytes / 1e9:.2f} GB total expert state)",
            flush=True,
        )

        # GPU AOS pipeline shares ``self._pinned_aos`` — no internal
        # allocation, no lazy growth. The segment_pipeline staging
        # (256 MiB GPU) is lazy-created on first ``run()`` once the
        # worker thread's CUDA context is up.
        self._aos_pipeline: _gpu_aos.LossyAosPipeline = _gpu_aos.LossyAosPipeline(
            pinned_aos=self._pinned_aos,
        )

        for eid, schema in expert_tensor_map.items():
            L, E = eid
            # Compute global flat offsets within the expert's
            # concatenated (param|m|v) address space.
            offsets: Dict[str, int] = {}
            cursor = 0
            for (tn, _shape, numel) in schema:
                offsets[tn] = cursor
                cursor += int(numel)
            entry = _ExpertEntry(
                expert_id=eid,
                tensor_schema=list(schema),
                global_offsets=offsets,
                total_numel=cursor,
                lossless_path=self._output_dir / _lossless_name(L, E),
                lossy_path=self._output_dir / _lossy_name(L, E),
            )
            if fresh:
                from .proposal_io import unlink_path as _unlink
                for p in (entry.lossless_path, entry.lossy_path):
                    _unlink(p)
            self._entries[eid] = entry

    # ------------------------------------------------------------------
    # I/O dispatch helper — route to AsyncWriter's raw worker if present,
    # else execute synchronously (test / offline path).
    # ------------------------------------------------------------------
    def _dispatch(self, fn: "Callable[..., None]", *args: Any, tag: str) -> None:
        # Route commit work to the AsyncWriter's raw worker so the caller
        # (host_writer_expert, which is in the middle of tracker.step) can
        # return immediately to launch this iter's score on tracker_stream.
        # Per-iter commits land on raw's FIFO in the order issued — FIFO is
        # the only ordering guarantee callers need (lossless before any
        # subsequent lossy on the same expert, which the caller already
        # serialises within host_writer_expert).
        del tag
        if self._async_writer is not None:
            self._async_writer.submit_raw(fn, *args)
        else:
            fn(*args)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def expert_ids(self) -> List[ExpertId]:
        return list(self._entries.keys())

    def lossless_fc_step(self, expert_id: ExpertId) -> int:
        """Iteration at which this expert's most recent lossless FC fired
        (-1 if never)."""
        ent = self._entries.get(expert_id)
        return ent.lossless_step if ent is not None else -1

    def has_lossless_fc(self, expert_id: ExpertId) -> bool:
        ent = self._entries.get(expert_id)
        return ent is not None and ent.lossless_step >= 0

    @property
    def total_bytes(self) -> int:
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
    # Lossless FC (full expert snapshot, round-robin)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def lossless_fc(
        self,
        expert_id: ExpertId,
        step: int,
        initial_state: Dict[str, Tuple[torch.Tensor,
                                       Optional[torch.Tensor],
                                       Optional[torch.Tensor]]],
        cuda_event: Optional["torch.cuda.Event"] = None,
        pinned_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write full (param, m, v) snapshot for ``expert_id`` and reset
        its lossy FC log.

        Training-thread work is limited to building the 4 KiB header
        bytes + dispatching. The worker thread syncs on ``cuda_event``,
        stamps the header into the pinned buffer, and emits one
        ``os.write`` of [header | p|m|v] straight from pinned to the
        tmp file before fsync + atomic rename.

        ``pinned_info`` is mandatory and carries the pinned buffer +
        size produced by ``fc_manager._collect_param_payload(...,
        header_block_bytes=HEADER_BLOCK_BYTES, ...)``.
        """
        ent = self._entries.get(expert_id)
        if ent is None:
            return
        del initial_state  # no longer needed — pinned_info owns the data

        header_meta = {
            "expert_id": [int(ent.expert_id[0]), int(ent.expert_id[1])],
            "step": int(step),
            "tensors": [
                {"name": tn, "shape": list(shape), "numel": int(n)}
                for (tn, shape, n) in ent.tensor_schema
            ],
            "global_offsets": dict(ent.global_offsets),
            "total_numel": int(ent.total_numel),
            "layout": "param_m_v_concat_fp32",
        }
        header = _pack_header(MAGIC_LFCL, header_meta)

        L, E = ent.expert_id
        self._dispatch(
            self._do_lossless_fc, ent, int(step), header,
            cuda_event, pinned_info,
            tag=f"lossless_L{L:02d}_E{E:02d}",
        )

    # ------------------------------------------------------------------
    # Background lossless FC handler
    # ------------------------------------------------------------------
    def _do_lossless_fc(
        self,
        ent: "_ExpertEntry",
        step: int,
        header: bytes,
        cuda_event: Optional["torch.cuda.Event"],
        pinned_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        # 0) Wait for the d2h_stream copies that produced our pinned
        # source tensors. Done on this worker thread so the training
        # thread is never blocked on D2H or the host memcpy below.
        if cuda_event is not None:
            try:
                cuda_event.synchronize()
            except Exception:
                pass

        # 1) Close the current lossy fd (we own fd lifecycle here).
        if ent.lossy_fd is not None:
            close_fd(ent.lossy_fd)
            ent.lossy_fd = None

        # 2) Write the new LFCL file via the single pinned-inline path.
        # The collect helper reserved HEADER_BLOCK_BYTES at the front of
        # the pinned buffer and laid the payload (p|m|v per tensor in
        # schema order) right after it. We stamp the header in-place
        # and ship the whole region in one os.write to the tmp file
        # (atomic rename via the centralised helper).
        if pinned_info is None:
            raise RuntimeError(
                "lossless_fc: pinned_info is required — collect helper must "
                "be called with header_block_bytes=HEADER_BLOCK_BYTES"
            )
        pinned_tensor = pinned_info["pinned"]
        payload_bytes = int(pinned_info["payload_bytes"])
        if int(pinned_info.get("header_block_bytes", 0)) != _BIN_HEADER_BLOCK_BYTES:
            raise RuntimeError(
                "lossless_fc: pinned_info.header_block_bytes mismatch — "
                f"expected {_BIN_HEADER_BLOCK_BYTES}, got "
                f"{pinned_info.get('header_block_bytes')}"
            )
        L_id, E_id = ent.expert_id
        write_lossless_from_pinned(
            ent.lossless_path,
            pinned_tensor,
            header_bytes=header,
            payload_bytes=payload_bytes,
            layer=int(L_id),
            expert=int(E_id),
            step=int(step),
        )

        ent.lossless_step = int(step)
        ent.step = int(step)

        # 3) Unlink previous lossy log (if any).
        from .proposal_io import unlink_path as _unlink
        _unlink(ent.lossy_path)

        # 4) Pre-create fresh lossy FC file with its header for upcoming
        # appends. _open_fresh_lossy runs on this BG thread, which owns
        # all fd mutations.
        self._open_fresh_lossy(ent, step=int(step))

    # ------------------------------------------------------------------
    # Lossy FC (append-only per iter)
    # ------------------------------------------------------------------
    # Singular ``lossy_fc_append`` (CPU SoA → numpy AoS assembly per
    # expert) removed in Phase 3 — the legacy path was bypassable only
    # for tests; production callers use ``lossy_fc_append_batch`` which
    # now drives the GPU AOS pipeline.
    # ------------------------------------------------------------------

    def lossy_fc_append_batch(
        self,
        step: int,
        per_expert_updates: Dict[ExpertId, List[Tuple[str, int, int, int]]],
        tracker_event: Optional[Any] = None,
        gpu_refs: Optional[List[torch.Tensor]] = None,
        bf16_groups: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor, torch.Tensor]]]] = None,
        lossy_state: Optional[Any] = None,
        per_expert_outlier_n: Optional[Dict[ExpertId, Dict[str, int]]] = None,
    ) -> int:
        """GPU AOS batch submit (Phase 3). See ``LossyAosPipeline``.

        ``per_expert_updates[eid]`` is a list of
        ``(tname, bf16_group_idx, src_off, n_records)`` tuples — pure
        descriptors, no CPU tensors. In Option A replace mode, the
        ``n_records`` here is GROUP records only. Outlier records are
        added on top via ``per_expert_outlier_n[eid][tname]`` and Pass B
        writes them within the segment producer.
        """
        prepared: List[Tuple[_ExpertEntry, int,
                             List[Tuple[str, int, int, int, int]]]] = []
        outlier_n_per_expert_entry: List[List[int]] = []
        grand_total = 0
        for eid, tensor_updates in per_expert_updates.items():
            ent = self._entries.get(eid)
            if ent is None:
                continue
            total_n = 0
            filtered: List[Tuple[str, int, int, int, int]] = []
            outlier_ns_for_expert: List[int] = []
            outlier_map_for_expert = (
                per_expert_outlier_n.get(eid, {}) if per_expert_outlier_n else {}
            )
            for (tname, bf16_group_idx, src_off, n_recs) in tensor_updates:
                if tname not in ent.global_offsets:
                    continue
                n_outlier_for_entry = int(outlier_map_for_expert.get(tname, 0))
                # n_recs from importance_tracker is GROUP records count.
                # The entry's TOTAL records = group + outlier.
                n_total_recs = int(n_recs) + n_outlier_for_entry
                if n_total_recs <= 0:
                    continue
                filtered.append((
                    tname,
                    ent.global_offsets[tname],
                    int(bf16_group_idx),
                    int(src_off),
                    n_total_recs,
                ))
                outlier_ns_for_expert.append(n_outlier_for_entry)
                total_n += n_total_recs
            ent.step = int(step)
            if total_n == 0:
                continue
            prepared.append((ent, total_n, filtered))
            outlier_n_per_expert_entry.append(outlier_ns_for_expert)
            grand_total += total_n
        if not prepared:
            if gpu_refs is not None:
                gpu_refs.clear()
            return 0
        # Pass lossy_state + outlier_n_per_expert_entry through to the
        # worker task. Only used when lossy_state is not None.
        self._dispatch(
            self._do_lossy_fc_gpu_aos,
            int(step), prepared, tracker_event, gpu_refs, bf16_groups,
            lossy_state, outlier_n_per_expert_entry,
            tag=f"lossy_batch_iter{int(step)}",
        )
        return grand_total

    def _do_lossy_fc_gpu_aos(
        self,
        step: int,
        prepared,
        tracker_event,
        gpu_refs,
        bf16_groups,
        lossy_state: Optional[Any] = None,
        outlier_n_per_expert_entry: Optional[List[List[int]]] = None,
    ) -> None:
        """Disk-mode worker (Phase 3). Drives the shared GPU AOS pipeline
        then ``::write``s each expert's 4 KiB-padded blob to its lossy fd.
        """
        expert_specs, total_aos = self._aos_pipeline.run(
            step=int(step),
            prepared=prepared,
            tracker_event=tracker_event,
            gpu_refs=gpu_refs,
            bf16_groups=bf16_groups,
            lossy_state=lossy_state,
            outlier_n_per_expert_entry=outlier_n_per_expert_entry,
        )
        if total_aos == 0:
            return
        pinned_aos = self._aos_pipeline.pinned_aos
        # Ablation knob: PROPOSAL_DISK_SKIP_WRITE=1 exercises the full
        # GPU AOS + D2H to pinned_aos but skips the per-expert ::write
        # syscall. Used to isolate whether the kernel ``copy_from_user``
        # of ::write is the residual L3 polluter (cf. CLAUDE.md +355 ms
        # diagnosis). 학습 / 복구가 깨지므로 측정 전용.
        import os as _os_skip
        _skip_write = _os_skip.environ.get(
            "PROPOSAL_DISK_SKIP_WRITE", "") in ("1", "true", "True")
        for spec, (ent, _total_n, _filtered) in zip(expert_specs, prepared):
            if ent.lossy_fd is None:
                self._open_fresh_lossy(ent, step=int(step))
            blob = pinned_aos.narrow(0, spec.aos_byte_off,
                                     spec.blob_padded_bytes)
            L, E = ent.expert_id
            if _skip_write:
                continue
            append_lossy_log_from_pinned(
                int(ent.lossy_fd), blob, int(spec.blob_padded_bytes),
                layer=int(L), expert=int(E), step=int(step),
            )

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    def _do_fsync_lossy(self, ent: "_ExpertEntry") -> None:
        if ent.lossy_fd is not None:
            fsync_fd(ent.lossy_fd)

    def flush(self, only_eids: Optional[List[ExpertId]] = None) -> None:
        """Drain pending async jobs, then fsync every open lossy FC fd.

        Lossless FC files are atomically persisted at job time (fsync +
        rename on the AsyncWriter raw worker), so this method only needs
        to cover the lossy append fds.
        """
        # Drain AsyncWriter raw queue first so pending writes land before
        # we fsync.
        if self._async_writer is not None:
            try:
                self._async_writer.drain_raw()
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
        if self._async_writer is not None:
            try:
                self._async_writer.drain_raw()
            except Exception:
                pass

    def shutdown(self) -> None:
        """Drain pending I/O and close every open fd. Idempotent.

        Does NOT shut down ``self._async_writer`` — that is owned by the
        top-level cp_hooks module (shared with CheckpointManager/FCManager)
        and is torn down separately in ``proposal_shutdown``.
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _open_fresh_lossy(self, ent: _ExpertEntry, step: int) -> None:
        header_meta = {
            "expert_id": [int(ent.expert_id[0]), int(ent.expert_id[1])],
            "tensors": [
                {"name": tn, "shape": list(shape), "numel": int(n)}
                for (tn, shape, n) in ent.tensor_schema
            ],
            "global_offsets": dict(ent.global_offsets),
            "total_numel": int(ent.total_numel),
            "start_step": int(step),
            "record_bytes": RECORD_BYTES,
        }
        header = _pack_header(MAGIC_LFSY, header_meta)
        ent.lossy_fd = open_lossy_log_for_append(ent.lossy_path, header)
        ent.lossy_start_step = int(step)


# Phase 3 removed the per-tensor numpy helpers (``_to_fp32_numpy`` /
# ``_to_i64_numpy``) that supported the CPU SoA → AoS interleave path.
# The GPU AOS kernel reads directly from the per-BF16-group GPU SoA.


# ----------------------------------------------------------------------------
# Recovery reader
# ----------------------------------------------------------------------------
def read_expert_state(
    expert_dir: Path,
    layer: int,
    expert: int,
) -> Dict[str, Any]:
    """Reconstruct one expert's state at the end of its current lossy FC
    window. Returns a payload compatible with the recovery pipeline:

        {"_rolling_step": int,
         "{tname}:param":    fp32 tensor of shape meta["shape"],
         "{tname}:exp_avg":  ...,
         "{tname}:exp_avg_sq": ...}

    Algorithm:
      1. If ``fc_expert_L{L:02d}_E{E:02d}.bin`` exists, load it fully →
         dense payload baseline.
      2. If ``lossy_expert_L{L:02d}_E{E:02d}.bin`` exists, replay every
         block in its appended order, scattering each record into the
         dense flat arrays (later-step writes overwrite earlier).
      3. Slice the concatenated buffers back into per-tensor shapes.
    """
    expert_dir = Path(expert_dir)
    lossless_path = expert_dir / _lossless_name(layer, expert)
    lossy_path = expert_dir / _lossy_name(layer, expert)

    # ------------------------------------------------------------------
    # 1) Lossless FC → dense (param, m, v) flat buffers.
    # All file IO goes through proposal_io.read_file_range (binary_io.cpp).
    # ------------------------------------------------------------------
    from .proposal_io import read_file_range as _pread, file_size as _fsize
    lossless_size = _fsize(lossless_path)
    if lossless_size >= 0:
        header_block = _pread(lossless_path, 0, HEADER_BLOCK_BYTES)
        magic, _ver, meta = _unpack_header(header_block)
        if magic != MAGIC_LFCL:
            raise ValueError(f"expected LFCL in {lossless_path}, got {magic!r}")
        tensors_meta = meta["tensors"]
        total_numel = int(meta["total_numel"])
        p_flat = np.zeros(total_numel, dtype=np.float32)
        m_flat = np.zeros(total_numel, dtype=np.float32)
        v_flat = np.zeros(total_numel, dtype=np.float32)
        cursor = 0
        file_off = HEADER_BLOCK_BYTES
        for entry in tensors_meta:
            numel = int(entry["numel"])
            for buf in (p_flat, m_flat, v_flat):
                raw = _pread(lossless_path, file_off, numel * 4)
                buf[cursor:cursor + numel] = np.frombuffer(raw, dtype=np.float32)
                file_off += numel * 4
            cursor += numel
        base_step = int(meta.get("step", 0))
    else:
        # No lossless FC yet — read schema from lossy FC header only.
        lossy_size_probe = _fsize(lossy_path)
        if lossy_size_probe < 0:
            return {"_rolling_step": 0}
        header_block = _pread(lossy_path, 0, HEADER_BLOCK_BYTES)
        magic, _ver, meta = _unpack_header(header_block)
        if magic != MAGIC_LFSY:
            return {"_rolling_step": 0}
        tensors_meta = meta["tensors"]
        total_numel = int(meta["total_numel"])
        p_flat = np.zeros(total_numel, dtype=np.float32)
        m_flat = np.zeros(total_numel, dtype=np.float32)
        v_flat = np.zeros(total_numel, dtype=np.float32)
        base_step = 0

    # ------------------------------------------------------------------
    # 2) Replay lossy FC append log (if present).
    #
    # Per CLAUDE.md § 복구 성능 주의사항: dedupe by idx keeping the LAST
    # occurrence, then one scatter per field.
    # ------------------------------------------------------------------
    max_step = base_step
    lossy_size = _fsize(lossy_path)
    if lossy_size >= 0:
        block_arrays: List[np.ndarray] = []
        header_block = _pread(lossy_path, 0, HEADER_BLOCK_BYTES)
        magic, _ver, _meta = _unpack_header(header_block)
        if magic != MAGIC_LFSY:
            return {"_rolling_step": int(max_step)}
        # write path (append_lossy_log_from_pinned) pads every block to
        # SECTOR_BYTES (4 KiB) for O_DIRECT/io_uring-compatible writes,
        # so each block starts on a sector boundary. Walk by sector.
        from .proposal_io import SECTOR_BYTES, _round_up_sector
        offset = HEADER_BLOCK_BYTES
        while offset + BLOCK_PREFIX_BYTES <= lossy_size:
            prefix_bytes = _pread(lossy_path, offset, BLOCK_PREFIX_BYTES)
            if len(prefix_bytes) < BLOCK_PREFIX_BYTES:
                break
            count, step = struct.unpack("<II", prefix_bytes)
            if count == 0:
                # empty / padding block — advance to next sector
                offset += SECTOR_BYTES
                continue
            payload_bytes = count * RECORD_BYTES
            data_off = offset + BLOCK_PREFIX_BYTES
            if data_off + payload_bytes > lossy_size:
                break  # truncated / corrupt tail
            raw = _pread(lossy_path, data_off, payload_bytes)
            if step > max_step:
                max_step = int(step)
            block_arrays.append(
                np.frombuffer(raw, dtype=_RECORD_DTYPE).copy()
            )
            # Advance to next sector boundary — block is padded to
            # SECTOR_BYTES by the writer.
            offset = _round_up_sector(data_off + payload_bytes)

        if block_arrays:
            # Concatenate blocks in file order (oldest → newest).
            all_records = np.concatenate(block_arrays)
            # Dedupe keeping LAST occurrence per idx:
            # reverse, np.unique(return_index=True) yields first-in-rev
            # which equals last-in-original.
            rev = all_records[::-1]
            _, first_in_rev = np.unique(rev["idx"], return_index=True)
            last_per_idx = rev[first_in_rev]
            idx64 = last_per_idx["idx"].astype(np.int64)
            # Single scatter per field over deduped indices.
            p_flat[idx64] = last_per_idx["p"]
            m_flat[idx64] = last_per_idx["m"]
            v_flat[idx64] = last_per_idx["v"]

    # ------------------------------------------------------------------
    # 3) Slice flat buffers back into per-tensor views.
    # ------------------------------------------------------------------
    payload: Dict[str, Any] = {"_rolling_step": int(max_step)}
    cursor = 0
    for entry in tensors_meta:
        tname = entry["name"]
        numel = int(entry["numel"])
        shape = torch.Size(entry["shape"])
        payload[f"{tname}:param"] = torch.from_numpy(
            p_flat[cursor:cursor + numel].copy()
        ).view(shape)
        payload[f"{tname}:exp_avg"] = torch.from_numpy(
            m_flat[cursor:cursor + numel].copy()
        ).view(shape)
        payload[f"{tname}:exp_avg_sq"] = torch.from_numpy(
            v_flat[cursor:cursor + numel].copy()
        ).view(shape)
        cursor += numel
    return payload
