"""In-memory (host DRAM) expert state for fast validation-loss experiments.

Drop-in replacement for ``ExpertState`` (the disk-backed lossy log + lossless
FC implementation). Activated by ``--proposal-cp-target dram``.

Why a separate class
====================

The disk implementation is shaped by SSD constraints:

* 4 KB block granularity → batched writev of (header + payload).
* Per-iter append-only log with magic header + structured records — the
  log is the only way to capture sparse updates between full rewrites.
* Recovery must read the whole log to replay last-write-wins per index,
  so log size grows as the run progresses and recovery slows.

When the storage backend is host DRAM the log structure buys nothing: we
can scatter the new ``(idx, p, m, v)`` directly into a per-expert
sparse-dense FP32 buffer. There is no concept of "log size" — the buffer
size is fixed at the expert's parameter count, and recovery reads it
directly. This is what the user spec calls "K%% 골라진 원소만 단순하게
DRAM에 업데이트".

Memory cost
===========

Per expert, per tracked tensor: 3 × fp32 × numel = 12 bytes per element.
For exp_3 (16 layers × 48 experts × 3 tensors × ~525 K elem ≈ 1.21 B
expert params) the buffer is ~14.5 GB host RAM. The 192 GiB host has
ample headroom.

API surface
===========

This class implements the same public methods as ``ExpertState`` plus a
recovery hook that ``recovery.recover()`` invokes when DRAM mode is
active:

* ``expert_ids() / lossless_fc_step(eid) / has_lossless_fc(eid) /
  total_bytes`` — same semantics as ExpertState.
* ``lossless_fc(eid, step, initial_state)`` — replace the per-tensor
  dense buffer wholesale; reset the "sparse drift" overlay.
* ``lossy_fc_append(eid, step, tensor_updates)`` — scatter the supplied
  ``(idx, p, m, v)`` quadruples into the dense buffer (last-write-wins
  on duplicate idx).
* ``flush(only_eids=None) / shutdown()`` — no-ops (no fds to fsync).
* ``read(eid)`` — returns the merged payload dict
  ``{"<tname>:param" / ":exp_avg" / ":exp_avg_sq": cpu_fp32_tensor}``
  for ``recovery._apply_payload_dict_to_model``.

What this class does NOT do
===========================

Non-expert FC and DC (gradient bundles) remain disk-backed even with
``--proposal-cp-target dram``. The user can redirect those to tmpfs by
pointing ``CP_DIR_OVERRIDE`` (or ``--proposal-cp-dir``) at ``/dev/shm``
if RAM-speed I/O for those paths is also desired. Sequential/MoC
checkpoint mode is similarly unaffected (still file-based).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# Tuple alias matches expert_state.ExpertId so callers can pass either
# class through the same expert-id-keyed maps without conversion.
ExpertId = Tuple[int, int]  # (layer, expert_idx)


class DramExpertState:
    """In-memory expert state. See module docstring for design rationale."""

    def __init__(
        self,
        expert_tensor_map: Dict[ExpertId, List[Tuple[str, torch.Size, int]]],
        output_dir: Path,
        *,
        fresh: bool = True,
        async_writer: Optional[Any] = None,
    ) -> None:
        """Same constructor signature as ``ExpertState`` so ``FCManager`` can
        instantiate either class without branching on type.

        Args:
            expert_tensor_map: per-expert list of (tensor_name, shape, numel).
            output_dir: accepted for API compatibility but unused — DRAM mode
                writes to host memory, not disk.
            fresh: accepted for compatibility; DRAM buffers are always fresh
                because they live in process memory.
            async_writer: optional AsyncWriter. When provided, ``lossless_fc``
                dispatches its (cuda_event.synchronize() + ~50 ms host copy)
                onto the writer's raw worker thread so the main training
                thread isn't blocked at iter boundaries with expert FC.
        """
        del output_dir, fresh  # unused in DRAM mode
        self._async_writer = async_writer

        # Per-expert: tensor_name -> {'p', 'm', 'v', 'shape', 'numel'}.
        # 'p'/'m'/'v' are flat CPU FP32 views into ONE contiguous backing
        # buffer (``self._flat``) of size 3 × Σ numel. Single allocation
        # avoids hundreds of small per-tensor torch.zeros() calls — the
        # previous design fragmented the CPU allocator and inflated
        # virtual memory bookkeeping under sustained training. With the
        # flat layout the OS sees a single anonymous mapping and the
        # views (.narrow + .view) are zero-cost.
        self._buffers: Dict[ExpertId, Dict[str, Dict[str, Any]]] = {}
        self._lossless_step: Dict[ExpertId, int] = {}
        self._last_step: Dict[ExpertId, int] = {}

        # Phase 1: compute total flat size (3 × Σ numel for p/m/v).
        total_numel = 0
        for eid, schema in expert_tensor_map.items():
            for (_tname, _shape, numel) in schema:
                total_numel += int(numel) * 3
        # Phase 2: single contiguous allocation, zero-initialised.
        self._flat = torch.zeros(int(total_numel), dtype=torch.float32)
        # Phase 3: hand out narrow() views to each (expert, tensor, role).
        # narrow returns a view sharing storage; scatter_/copy_ on the
        # view writes through to self._flat.
        offset = 0
        for eid, schema in expert_tensor_map.items():
            tensor_buffers: Dict[str, Dict[str, Any]] = {}
            for (tname, shape, numel) in schema:
                n = int(numel)
                p_view = self._flat.narrow(0, offset, n); offset += n
                m_view = self._flat.narrow(0, offset, n); offset += n
                v_view = self._flat.narrow(0, offset, n); offset += n
                tensor_buffers[tname] = {
                    "p": p_view,
                    "m": m_view,
                    "v": v_view,
                    "shape": shape,
                    "numel": n,
                }
            self._buffers[eid] = tensor_buffers
            self._lossless_step[eid] = -1
            self._last_step[eid] = -1
        assert offset == total_numel, (
            f"flat-buffer offset accounting mismatch: "
            f"used {offset} of {total_numel}"
        )

    # ------------------------------------------------------------------
    # Introspection — mirror ExpertState's API surface.
    # ------------------------------------------------------------------
    def expert_ids(self) -> List[ExpertId]:
        return list(self._buffers.keys())

    def lossless_fc_step(self, expert_id: ExpertId) -> int:
        return self._lossless_step.get(expert_id, -1)

    def has_lossless_fc(self, expert_id: ExpertId) -> bool:
        return self._lossless_step.get(expert_id, -1) >= 0

    @property
    def total_bytes(self) -> int:
        """Sum of all dense buffer bytes (3 × fp32 per element). The disk
        ExpertState reports cumulative file size including the lossy log
        overhead; the DRAM equivalent is just the bounded dense footprint
        because no overlay log exists."""
        total = 0
        for tensor_buffers in self._buffers.values():
            for buf in tensor_buffers.values():
                total += buf["numel"] * 4 * 3
        return total

    # ------------------------------------------------------------------
    # Lossless FC (full per-expert snapshot)
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
    ) -> None:
        """Replace dense (param, m, v) buffer with the supplied triples.

        Async path: dispatch the (cuda_event.synchronize() + host copy)
        to the AsyncWriter's raw worker queue so the main training
        thread isn't blocked on the D2H wait + ~50 ms host copy.
        lossy_fc_append uses the same queue → automatic FIFO ordering on
        the dense buffer, no race.
        """
        # Bypass for measurement-only runs.
        import os as _os
        if _os.environ.get("PROPOSAL_SKIP_EXPERT_WRITES", "") in ("1", "true", "True"):
            self._lossless_step[expert_id] = int(step)
            self._last_step[expert_id] = int(step)
            return
        tensor_buffers = self._buffers.get(expert_id)
        if tensor_buffers is None:
            return

        # Update step markers immediately (used by recovery — must be
        # consistent with the dispatched buffer write since FIFO worker
        # processes items in order).
        self._lossless_step[expert_id] = int(step)
        self._last_step[expert_id] = int(step)

        def _do_lossless_fc(
            triples=dict(initial_state),
            buffers=tensor_buffers,
            ev=cuda_event,
        ):
            if ev is not None:
                try:
                    ev.synchronize()
                except Exception:
                    pass
            for tname, triple in triples.items():
                buf = buffers.get(tname)
                if buf is None:
                    continue
                p, m, v = triple
                buf["p"].copy_(p.detach().to(torch.float32).reshape(-1))
                if m is not None:
                    buf["m"].copy_(m.detach().to(torch.float32).reshape(-1))
                else:
                    buf["m"].zero_()
                if v is not None:
                    buf["v"].copy_(v.detach().to(torch.float32).reshape(-1))
                else:
                    buf["v"].zero_()

        # Caller (ExpertLosslessFC._fire_one) already runs on the
        # host_writer_expert worker — inline avoids self-recursive
        # enqueueing into the same FIFO.
        _do_lossless_fc()

    # ------------------------------------------------------------------
    # Lossy FC (per-iter sparse updates → scatter into dense buffer)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def lossy_fc_append(
        self,
        expert_id: ExpertId,
        step: int,
        tensor_updates: List[Tuple[str, torch.Tensor, torch.Tensor,
                                   torch.Tensor, torch.Tensor]],
    ) -> int:
        """Scatter the supplied ``(idx, p, m, v)`` quadruples directly
        into this expert's dense FP32 buffer. Last-write-wins on
        duplicate ``idx`` happens naturally via ``Tensor.scatter_``.

        Returns the total number of elements written (sum of idx sizes
        across tensors), matching ``ExpertState.lossy_fc_append``.
        """
        # Bypass for measurement-only runs (see lossless_fc docstring).
        import os as _os
        if _os.environ.get("PROPOSAL_SKIP_EXPERT_WRITES", "") in ("1", "true", "True"):
            n = sum(int(idx.numel()) for (_, idx, _, _, _) in tensor_updates)
            if n > 0:
                self._last_step[expert_id] = int(step)
            return n
        tensor_buffers = self._buffers.get(expert_id)
        if tensor_buffers is None:
            return 0
        total_n = 0
        for (tname, idx_local, p, m, v) in tensor_updates:
            buf = tensor_buffers.get(tname)
            if buf is None:
                continue
            n = int(idx_local.numel())
            if n == 0:
                continue
            # Cast index to long for scatter. The values land in CPU FP32.
            idx_long = idx_local.to(torch.long, copy=False)
            buf["p"].scatter_(0, idx_long, p.to(torch.float32))
            buf["m"].scatter_(0, idx_long, m.to(torch.float32))
            buf["v"].scatter_(0, idx_long, v.to(torch.float32))
            total_n += n
        if total_n > 0:
            self._last_step[expert_id] = int(step)
        return total_n

    @torch.no_grad()
    def lossy_fc_append_batch(
        self,
        step: int,
        per_expert_updates: Dict[ExpertId, List[Tuple[str, torch.Tensor,
                                                       torch.Tensor, torch.Tensor,
                                                       torch.Tensor]]],
        h2d_event: Optional[Any] = None,
        gpu_refs: Optional[List[torch.Tensor]] = None,
    ) -> int:
        """Batch variant: package all per-expert scatter ops and submit
        ONE writer-thread job. Avoids ~80 per-iter ``submit_raw`` lock
        round-trips that the training thread paid in the per-call form,
        and moves the actual scatter compute off the training thread.

        ``h2d_event``: optional cuda.Event recorded after the d2h_stream
        copies that populated the pinned-host tensors in
        ``per_expert_updates``. Writer thread synchronises on it before
        running the scatter — keeps the H2D off the training-thread
        critical path while guaranteeing read-after-write ordering.

        ``gpu_refs``: optional list of GPU tensors that were the *source*
        of the H2D copies into ``per_expert_updates``'s pinned tensors.
        The training thread hands ownership to the writer thread; the
        writer releases (clears the list) only AFTER ``h2d_event``
        synchronises, guaranteeing the GPU memory cannot be reclaimed
        by the caching allocator before the H2D completes. Replaces the
        prior ``record_stream`` patch with explicit ownership transfer
        — see Phase B in ``_dispatch_pending_payload``."""
        import os as _os
        if _os.environ.get("PROPOSAL_SKIP_EXPERT_WRITES", "") in ("1", "true", "True"):
            grand = 0
            for eid, tensor_updates in per_expert_updates.items():
                n = sum(int(idx.numel()) for (_, idx, _, _, _) in tensor_updates)
                if n > 0:
                    self._last_step[eid] = int(step)
                grand += n
            return grand

        # Training-thread: light validation + package. The actual
        # scatter (CPU memcpy with random indices) happens in the
        # writer thread.
        prepared: List[Tuple[ExpertId,
                             List[Tuple[Dict[str, Any], torch.Tensor,
                                        torch.Tensor, torch.Tensor,
                                        torch.Tensor]]]] = []
        grand_total = 0
        for eid, tensor_updates in per_expert_updates.items():
            tensor_buffers = self._buffers.get(eid)
            if tensor_buffers is None:
                continue
            per_expert: List[Tuple[Dict[str, Any], torch.Tensor,
                                    torch.Tensor, torch.Tensor,
                                    torch.Tensor]] = []
            n_eid = 0
            for (tname, idx_local, p, m, v) in tensor_updates:
                buf = tensor_buffers.get(tname)
                if buf is None:
                    continue
                n_i = int(idx_local.numel())
                if n_i == 0:
                    continue
                per_expert.append((buf, idx_local, p, m, v))
                n_eid += n_i
            if n_eid > 0:
                self._last_step[eid] = int(step)
                prepared.append((eid, per_expert))
                grand_total += n_eid
        if not prepared:
            return 0
        # Caller (ImportanceTracker._dispatch_pending_payload) already
        # runs on the host_writer_expert worker — inline avoids a
        # self-recursive submit into the same FIFO.
        self._do_dram_lossy_batch(prepared, h2d_event, gpu_refs)
        return grand_total

    @torch.no_grad()
    def _do_dram_lossy_batch(
        self,
        prepared: List[Tuple[ExpertId,
                             List[Tuple[Dict[str, Any], torch.Tensor,
                                        torch.Tensor, torch.Tensor,
                                        torch.Tensor]]]],
        h2d_event: Optional[Any] = None,
        gpu_refs: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """Writer-thread handler — synchronises on ``h2d_event`` (if any)
        then runs the scatter ops sequentially. Releases ``gpu_refs``
        (Phase A staging tensors) AFTER the event sync so the caching
        allocator can safely reuse their memory — at which point the
        H2D is provably complete."""
        if h2d_event is not None:
            try:
                h2d_event.synchronize()
            except Exception:
                pass
        if gpu_refs is not None:
            gpu_refs.clear()
        for (eid, per_expert) in prepared:
            for (buf, idx_local, p, m, v) in per_expert:
                idx_long = idx_local.to(torch.long, copy=False)
                buf["p"].scatter_(0, idx_long, p.to(torch.float32))
                buf["m"].scatter_(0, idx_long, m.to(torch.float32))
                buf["v"].scatter_(0, idx_long, v.to(torch.float32))

    # ------------------------------------------------------------------
    # Bookkeeping no-ops (kept for API compatibility with ExpertState).
    # ------------------------------------------------------------------
    def flush(self, only_eids: Optional[List[ExpertId]] = None) -> None:
        """Drain pending writer-thread scatter jobs (if any). Required
        before recovery reads the dense buffer."""
        del only_eids
        if self._async_writer is not None:
            try:
                self._async_writer.drain_expert()
            except Exception:
                pass

    def shutdown(self) -> None:
        """No-op for symmetry with ExpertState. The buffers are released
        when this instance is garbage-collected. Note we don't proactively
        clear them here because some callers (recovery, replay-only mode)
        invoke ``shutdown()`` and then expect the data to still be
        readable for diagnostic purposes."""

    # ------------------------------------------------------------------
    # Recovery hook (DRAM-only path).
    # ------------------------------------------------------------------
    def read(self, expert_id: ExpertId) -> Optional[Dict[str, torch.Tensor]]:
        """Return the merged dense payload for one expert in the format
        expected by ``recovery._apply_payload_dict_to_model``. Returns
        ``None`` when the expert is unknown (defensive — same convention
        as ``read_expert_state`` returning empty when no files exist).
        """
        tensor_buffers = self._buffers.get(expert_id)
        if tensor_buffers is None:
            return None
        # Build the flat {"name:field": tensor} dict the loader uses to
        # scatter into model + optimizer state. Use the buffers directly
        # (no clone) because the loader does its own .to(device) and we
        # don't mutate them after recovery starts.
        out: Dict[str, torch.Tensor] = {}
        for tname, buf in tensor_buffers.items():
            out[f"{tname}:param"] = buf["p"]
            out[f"{tname}:exp_avg"] = buf["m"]
            out[f"{tname}:exp_avg_sq"] = buf["v"]
        # `_apply_payload_dict_to_model` reads `_rolling_step` if present
        # for diagnostics; supply the latest lossy/lossless step.
        out["_rolling_step"] = torch.tensor(
            int(self._last_step.get(expert_id, -1)), dtype=torch.long,
        )
        return out
