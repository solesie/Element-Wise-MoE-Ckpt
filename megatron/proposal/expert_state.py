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

from .binary_io import (
    HEADER_BLOCK_BYTES as _BIN_HEADER_BLOCK_BYTES,
    alloc_pinned_uint8_sector_aligned,
    append_lossy_log_from_pinned,
    close_fd,
    fsync_fd,
    open_lossy_log_for_append,
    write_lossless_from_pinned,
)


ExpertId = Tuple[int, int]  # (layer, expert_idx)

# ---- Binary format constants -----------------------------------------------
MAGIC_LFCL = b"LFCL"  # lossless FC
MAGIC_LFSY = b"LFSY"  # lossy FC append log
FORMAT_VERSION: int = 1
# Reuse the file-format header block size from binary_io so PRPTENS,
# LFCL, and LFSY all share a single sector convention. 64 KiB is the
# current value (sized to fit the largest header — non-expert FC).
HEADER_BLOCK_BYTES: int = _BIN_HEADER_BLOCK_BYTES
RECORD_BYTES: int = 16  # int32 idx + fp32 p + fp32 m + fp32 v
BLOCK_PREFIX_BYTES: int = 8  # count u32 + step u32

# Filename conventions.
_LOSSLESS_RE = re.compile(r"^fc_expert_L(\d+)_E(\d+)\.bin$")
_LOSSY_RE = re.compile(r"^lossy_expert_L(\d+)_E(\d+)\.bin$")

# Per-record numpy structured dtype (little-endian, packed to 16B).
_RECORD_DTYPE = np.dtype([
    ("idx", "<i4"),
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
                for p in (entry.lossless_path, entry.lossy_path):
                    if p.exists():
                        try:
                            p.unlink()
                        except OSError:
                            pass
            self._entries[eid] = entry

    # ------------------------------------------------------------------
    # I/O dispatch helper — route to AsyncWriter's raw worker if present,
    # else execute synchronously (test / offline path).
    # ------------------------------------------------------------------
    def _dispatch(self, fn: "Callable[..., None]", *args: Any, tag: str) -> None:
        # Always run inline. In the new arch every caller of ExpertState
        # (ExpertLosslessFC, ImportanceTracker.dispatch) already lives on
        # the host_writer_expert worker, so submitting back to that same
        # queue here would simply enqueue extra steps without parallelism
        # — and would self-join inside any later drain_expert.
        del tag
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
        pinned_mv = memoryview(pinned_tensor.numpy().view(np.uint8))
        L_id, E_id = ent.expert_id
        write_lossless_from_pinned(
            ent.lossless_path,
            pinned_mv,
            header_bytes=header,
            payload_bytes=payload_bytes,
            layer=int(L_id),
            expert=int(E_id),
            step=int(step),
        )

        ent.lossless_step = int(step)
        ent.step = int(step)

        # 3) Unlink previous lossy log (if any).
        if ent.lossy_path.exists():
            try:
                ent.lossy_path.unlink()
            except OSError:
                pass

        # 4) Pre-create fresh lossy FC file with its header for upcoming
        # appends. _open_fresh_lossy runs on this BG thread, which owns
        # all fd mutations.
        self._open_fresh_lossy(ent, step=int(step))

    # ------------------------------------------------------------------
    # Lossy FC (append-only per iter)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def lossy_fc_append(
        self,
        expert_id: ExpertId,
        step: int,
        tensor_updates: List[Tuple[str, torch.Tensor, torch.Tensor,
                                   torch.Tensor, torch.Tensor]],
    ) -> int:
        """Append one per-iter block for ``expert_id``.

        **Training-thread work**: build the per-iter numpy structured
        array from already-CPU input tensors. Zero-copy for fp32/int32
        contiguous inputs.

        **Background-thread work** (in ``_do_lossy_fc_assemble_and_write``):
        assemble the pinned (count, step, records) blob and emit a
        single ``os.write`` into the per-expert lossy FC log, lazily
        opening the fd if needed.
        """
        ent = self._entries.get(expert_id)
        if ent is None:
            return 0
        # Tally filtered total (skip unknown tensor names / empty
        # selections). All filtering work here is fast; the heavy
        # numpy structured-array assembly is moved to the writer
        # thread (see _do_lossy_fc_assemble_and_write) so the
        # training thread doesn't pay ~100-200 ms of CPU memcpy
        # per iter for ~9 M selected elements.
        total_n = 0
        filtered: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for (tname, idx_local, p, m, v) in tensor_updates:
            if tname not in ent.global_offsets:
                continue
            n_i = int(idx_local.numel())
            if n_i == 0:
                continue
            filtered.append((ent.global_offsets[tname], idx_local, p, m, v))
            total_n += n_i
        if total_n == 0:
            ent.step = int(step)
            return 0

        ent.step = int(step)
        # The CPU tensors in ``filtered`` were freshly produced by the
        # tracker dispatch (not aliased to any ongoing GPU buffer), so
        # ownership transfer to the writer thread is safe.
        L, E = ent.expert_id
        self._dispatch(
            self._do_lossy_fc_assemble_and_write,
            ent, int(step), filtered, int(total_n),
            tag=f"lossy_L{L:02d}_E{E:02d}",
        )
        return total_n

    def _do_lossy_fc_assemble_and_write(
        self,
        ent: "_ExpertEntry",
        step: int,
        filtered: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        total_n: int,
    ) -> None:
        """Writer-thread assembly + single ``os.write`` from a pinned
        AoS blob. Builds ``[prefix(8B) | records(N×16B)]`` directly
        into pinned memory so the C++ SSD-passthrough writer can later
        replace the ``os.write`` with a chunk-aligned io_uring submit
        without any extra copy. The blob is allocated 4 KiB-aligned
        (size padded up to next sector) so the data_ptr is O_DIRECT
        compatible regardless of total_n."""
        blob_bytes = BLOCK_PREFIX_BYTES + total_n * RECORD_BYTES
        blob = alloc_pinned_uint8_sector_aligned(blob_bytes)
        blob_np = blob.numpy()
        struct.pack_into("<II", blob_np, 0, int(total_n), int(step))
        rec_view = blob_np[
            BLOCK_PREFIX_BYTES:BLOCK_PREFIX_BYTES + total_n * RECORD_BYTES
        ].view(_RECORD_DTYPE)
        cursor = 0
        for (offset, idx_local, p, m, v) in filtered:
            n_i = int(idx_local.numel())
            idx_np = _to_i32_numpy(idx_local, n_i)
            rec_view["idx"][cursor:cursor + n_i] = idx_np + np.int32(offset)
            rec_view["p"][cursor:cursor + n_i] = _to_fp32_numpy(p, n_i)
            rec_view["m"][cursor:cursor + n_i] = _to_fp32_numpy(m, n_i)
            rec_view["v"][cursor:cursor + n_i] = _to_fp32_numpy(v, n_i)
            cursor += n_i
        if ent.lossy_fd is None:
            self._open_fresh_lossy(ent, step=int(step))
        L_id, E_id = ent.expert_id
        append_lossy_log_from_pinned(
            ent.lossy_fd, memoryview(blob_np), blob_bytes,
            layer=int(L_id), expert=int(E_id), step=int(step),
        )

    def lossy_fc_append_batch(
        self,
        step: int,
        per_expert_updates: Dict[ExpertId, List[Tuple[str, torch.Tensor,
                                                       torch.Tensor, torch.Tensor,
                                                       torch.Tensor]]],
        h2d_event: Optional[Any] = None,
        gpu_refs: Optional[List[torch.Tensor]] = None,
    ) -> int:
        """Batch variant of ``lossy_fc_append``: takes ALL per-expert
        updates for one iter and submits a SINGLE writer-thread job that
        handles every expert in sequence. Avoids ~80 per-iter
        submit_raw calls (each costs ~1-2 ms training-thread time
        from queue lock + counter increment), so the training-thread
        overhead drops to one submission. Heavy numpy assembly + writev
        still runs on the writer thread.

        ``h2d_event``: optional cuda.Event recorded after the d2h_stream
        copies that populate the pinned tensor data. The writer thread
        synchronises on this event before reading the pinned tensors —
        keeps the H2D off the training thread's critical path while
        guaranteeing the writer sees consistent data."""
        # Build a list of (entry, total_n, filtered) on training thread.
        # All work here is fast (filtering numel checks + dict lookups);
        # the heavy numpy memcpy is deferred to the writer thread.
        prepared: List[Tuple[_ExpertEntry, int, List[Tuple[int, torch.Tensor,
                                                            torch.Tensor, torch.Tensor,
                                                            torch.Tensor]]]] = []
        grand_total = 0
        for eid, tensor_updates in per_expert_updates.items():
            ent = self._entries.get(eid)
            if ent is None:
                continue
            total_n = 0
            filtered: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor,
                                 torch.Tensor]] = []
            for (tname, idx_local, p, m, v) in tensor_updates:
                if tname not in ent.global_offsets:
                    continue
                n_i = int(idx_local.numel())
                if n_i == 0:
                    continue
                filtered.append((ent.global_offsets[tname], idx_local, p, m, v))
                total_n += n_i
            ent.step = int(step)
            if total_n == 0:
                continue
            prepared.append((ent, total_n, filtered))
            grand_total += total_n
        if not prepared:
            return 0
        self._dispatch(
            self._do_lossy_fc_batch_write,
            int(step), prepared, h2d_event, gpu_refs,
            tag=f"lossy_batch_iter{int(step)}",
        )
        return grand_total

    def _do_lossy_fc_batch_write(
        self,
        step: int,
        prepared: List[Tuple["_ExpertEntry", int, List[Tuple[int, torch.Tensor,
                                                              torch.Tensor, torch.Tensor,
                                                              torch.Tensor]]]],
        h2d_event: Optional[Any] = None,
        gpu_refs: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """Writer-thread handler for the batched variant — performs the
        per-expert SoA→AoS interleave into a pinned packed buffer +
        single ``os.write`` sequentially.

        Synchronises on ``h2d_event`` first (when present) so we read
        the pinned tensors only after the d2h_stream copies that filled
        them have completed."""
        if h2d_event is not None:
            try:
                h2d_event.synchronize()
            except Exception:
                pass
        # H2D provably done → release Phase A GPU staging tensors so the
        # caching allocator can reuse their memory (was held alive by
        # the training thread to avoid cross-stream UB).
        if gpu_refs is not None:
            gpu_refs.clear()
        for (ent, total_n, filtered) in prepared:
            # Per-expert pinned blob: [prefix(8B) | records(N×16B)].
            # Allocated as a fresh pinned uint8 tensor so the C++
            # writer (future SSD passthrough) can submit the entire
            # blob in one chunk-aligned io_uring write. Torch's pinned
            # caching allocator recycles freed buffers so per-iter
            # alloc cost is small. Use sector-aligned helper so the
            # blob's data_ptr is 4 KiB-aligned (small uint8 allocs from
            # torch's caching pinned allocator may otherwise return
            # non-page-aligned pointers — incompatible with O_DIRECT).
            blob_bytes = BLOCK_PREFIX_BYTES + total_n * RECORD_BYTES
            blob = alloc_pinned_uint8_sector_aligned(blob_bytes)
            blob_np = blob.numpy()
            # Pack (count, step) prefix at the head of the pinned blob.
            struct.pack_into(
                "<II", blob_np, 0, int(total_n), int(step),
            )
            # Interleave SoA planar → AoS packed directly into pinned.
            rec_view = blob_np[
                BLOCK_PREFIX_BYTES:BLOCK_PREFIX_BYTES + total_n * RECORD_BYTES
            ].view(_RECORD_DTYPE)
            cursor = 0
            for (offset, idx_local, p, m, v) in filtered:
                n_i = int(idx_local.numel())
                idx_np = _to_i32_numpy(idx_local, n_i)
                rec_view["idx"][cursor:cursor + n_i] = idx_np + np.int32(offset)
                rec_view["p"][cursor:cursor + n_i] = _to_fp32_numpy(p, n_i)
                rec_view["m"][cursor:cursor + n_i] = _to_fp32_numpy(m, n_i)
                rec_view["v"][cursor:cursor + n_i] = _to_fp32_numpy(v, n_i)
                cursor += n_i
            if ent.lossy_fd is None:
                self._open_fresh_lossy(ent, step=int(step))
            L_id, E_id = ent.expert_id
            append_lossy_log_from_pinned(
                ent.lossy_fd, memoryview(blob_np), blob_bytes,
                layer=int(L_id), expert=int(E_id), step=int(step),
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
        # Drain AsyncWriter queues first so pending writes land before
        # we fsync.
        if self._async_writer is not None:
            try:
                self._async_writer.drain_expert()
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
                self._async_writer.drain_expert()
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


# ----------------------------------------------------------------------------
# Tensor → numpy helpers (zero-copy for CPU contiguous tensors)
# ----------------------------------------------------------------------------
def _to_fp32_numpy(t: torch.Tensor, expected_n: int) -> np.ndarray:
    """Return a 1-D fp32 numpy view of ``t`` with exactly ``expected_n``
    elements. Copies only if dtype/device/contiguity requires it."""
    t = t.detach()
    if t.device.type != "cpu":
        t = t.to("cpu")
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    if not t.is_contiguous():
        t = t.contiguous()
    arr = t.numpy().reshape(-1)
    if arr.size != expected_n:
        # Caller passed a wrong-size tensor; trim or pad to expected_n to
        # avoid hard failures (defensive: normal path has size match).
        if arr.size > expected_n:
            arr = arr[:expected_n]
        else:
            padded = np.zeros(expected_n, dtype=np.float32)
            padded[:arr.size] = arr
            arr = padded
    return arr


def _to_i32_numpy(t: torch.Tensor, expected_n: int) -> np.ndarray:
    t = t.detach()
    if t.device.type != "cpu":
        t = t.to("cpu")
    if t.dtype != torch.int32:
        t = t.to(torch.int32)
    if not t.is_contiguous():
        t = t.contiguous()
    arr = t.numpy().reshape(-1)
    if arr.size != expected_n:
        if arr.size > expected_n:
            arr = arr[:expected_n]
        else:
            padded = np.zeros(expected_n, dtype=np.int32)
            padded[:arr.size] = arr
            arr = padded
    return arr


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
    # ------------------------------------------------------------------
    if lossless_path.exists():
        with open(lossless_path, "rb") as f:
            header_block = f.read(HEADER_BLOCK_BYTES)
        magic, _ver, meta = _unpack_header(header_block)
        if magic != MAGIC_LFCL:
            raise ValueError(f"expected LFCL in {lossless_path}, got {magic!r}")
        tensors_meta = meta["tensors"]
        total_numel = int(meta["total_numel"])
        p_flat = np.zeros(total_numel, dtype=np.float32)
        m_flat = np.zeros(total_numel, dtype=np.float32)
        v_flat = np.zeros(total_numel, dtype=np.float32)
        fd = os.open(str(lossless_path), os.O_RDONLY)
        try:
            cursor = 0
            file_off = HEADER_BLOCK_BYTES
            for entry in tensors_meta:
                numel = int(entry["numel"])
                # Read param / m / v for this tensor from disk.
                for buf in (p_flat, m_flat, v_flat):
                    raw = os.pread(fd, numel * 4, file_off)
                    buf[cursor:cursor + numel] = np.frombuffer(raw, dtype=np.float32)
                    file_off += numel * 4
                cursor += numel
        finally:
            os.close(fd)
        base_step = int(meta.get("step", 0))
    else:
        # No lossless FC yet — read schema from lossy FC header only.
        if not lossy_path.exists():
            return {"_rolling_step": 0}
        with open(lossy_path, "rb") as f:
            header_block = f.read(HEADER_BLOCK_BYTES)
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
    # Per CLAUDE.md § 복구 성능 주의사항: the lossy log accumulates
    # duplicate element_index entries as the same "important" elements
    # get re-selected each iter. Naive per-block scatter would touch
    # every record (including duplicates). Optimised path: read every
    # block into a single concatenated ndarray, dedupe by idx keeping
    # the LAST occurrence, then apply one scatter per field. This keeps
    # the scatter cost proportional to the unique-index count, not the
    # total record count.
    # ------------------------------------------------------------------
    max_step = base_step
    if lossy_path.exists():
        block_arrays: List[np.ndarray] = []
        fd = os.open(str(lossy_path), os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            header_block = os.pread(fd, HEADER_BLOCK_BYTES, 0)
            magic, _ver, _meta = _unpack_header(header_block)
            if magic != MAGIC_LFSY:
                return {"_rolling_step": int(max_step)}
            offset = HEADER_BLOCK_BYTES
            while offset + BLOCK_PREFIX_BYTES <= size:
                prefix_bytes = os.pread(fd, BLOCK_PREFIX_BYTES, offset)
                if len(prefix_bytes) < BLOCK_PREFIX_BYTES:
                    break
                count, step = struct.unpack("<II", prefix_bytes)
                offset += BLOCK_PREFIX_BYTES
                if count == 0:
                    continue
                payload_bytes = count * RECORD_BYTES
                if offset + payload_bytes > size:
                    break  # truncated / corrupt tail
                raw = os.pread(fd, payload_bytes, offset)
                offset += payload_bytes
                if step > max_step:
                    max_step = int(step)
                block_arrays.append(
                    np.frombuffer(raw, dtype=_RECORD_DTYPE).copy()
                )
        finally:
            os.close(fd)

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
