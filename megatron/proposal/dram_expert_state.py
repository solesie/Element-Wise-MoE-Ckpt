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
            async_writer: accepted for compatibility but unused; all updates
                are synchronous CPU scatters.
        """
        del output_dir, async_writer, fresh  # unused in DRAM mode

        # Per-expert: tensor_name -> {'p', 'm', 'v', 'shape', 'numel'}.
        # 'p'/'m'/'v' are flat CPU FP32 buffers of size 'numel'. 'shape'
        # is kept so callers that need the original tensor shape (e.g.
        # for diagnostics) can recover it without the model handle.
        self._buffers: Dict[ExpertId, Dict[str, Dict[str, Any]]] = {}
        # Most recent step that fired a lossless / lossy update for this
        # expert. `lossless_step` mirrors ExpertState's `_entries[eid].
        # lossless_step` so external code (e.g. FCManager.tick logging)
        # sees the same field.
        self._lossless_step: Dict[ExpertId, int] = {}
        self._last_step: Dict[ExpertId, int] = {}

        for eid, schema in expert_tensor_map.items():
            tensor_buffers: Dict[str, Dict[str, Any]] = {}
            for (tname, shape, numel) in schema:
                # Pre-allocate zero buffers so lossy FC can scatter even
                # before the first lossless FC fires for this expert. The
                # alternative — lazy alloc on first use — would make the
                # very first lossy_fc_append for an expert pay the full
                # alloc cost on the training thread.
                tensor_buffers[tname] = {
                    "p": torch.zeros(int(numel), dtype=torch.float32),
                    "m": torch.zeros(int(numel), dtype=torch.float32),
                    "v": torch.zeros(int(numel), dtype=torch.float32),
                    "shape": shape,
                    "numel": int(numel),
                }
            self._buffers[eid] = tensor_buffers
            self._lossless_step[eid] = -1
            self._last_step[eid] = -1

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
    ) -> None:
        """Replace the per-tensor dense buffer with the supplied
        (param, m, v) triples. Mirrors ``ExpertState.lossless_fc``: the
        provided tensors come from ``FCManager._collect_expert_payload``
        already on CPU; we only have to copy them into our buffer.

        Any sparse drift accumulated since the last lossless FC is
        implicitly discarded (overwritten by the new dense snapshot),
        which matches the disk semantics where the prior lossy FC log is
        unlinked at this point.
        """
        tensor_buffers = self._buffers.get(expert_id)
        if tensor_buffers is None:
            return
        for tname, triple in initial_state.items():
            buf = tensor_buffers.get(tname)
            if buf is None:
                continue
            p, m, v = triple
            # FP32 flatten; .copy_ keeps the buffer's identity (no realloc).
            buf["p"].copy_(p.detach().to(torch.float32).reshape(-1))
            if m is not None:
                buf["m"].copy_(m.detach().to(torch.float32).reshape(-1))
            else:
                buf["m"].zero_()
            if v is not None:
                buf["v"].copy_(v.detach().to(torch.float32).reshape(-1))
            else:
                buf["v"].zero_()
        self._lossless_step[expert_id] = int(step)
        self._last_step[expert_id] = int(step)

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

    # ------------------------------------------------------------------
    # Bookkeeping no-ops (kept for API compatibility with ExpertState).
    # ------------------------------------------------------------------
    def flush(self, only_eids: Optional[List[ExpertId]] = None) -> None:
        """No-op: there are no fds or async jobs to drain. Provided so
        callers (FCManager, cp_hooks) can call ``flush()`` uniformly."""
        del only_eids

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
