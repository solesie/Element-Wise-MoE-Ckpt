"""Round-robin expert lossless FC.

Separated from fc_manager to keep concerns split (CLAUDE.md § Expert 저장
전략):

  - **lossy expert FC**: per-iter top-K element snapshot → ImportanceTracker
    (this stays in tracker.step).
  - **lossless expert FC** (this module): every D iter, F% experts in
    round-robin slot order get a full (weight + m + v) snapshot.
  - **non-expert FC** (still in fc_manager): every W iter, full non-expert
    weight + m + v snapshot.

All GPU work in this module runs on **tracker_stream** so it serializes
naturally with `tracker.update_delta_accum` (same FIFO, race-free) and
overlaps with main-stream forward/backward/optimizer.

Per-expert post-fire actions (CLAUDE.md spec):
  1. submit (weight + m + v) payload to ExpertState (disk or dram)
  2. truncate that expert's lossy FC log (handled inside ExpertState)
  3. reset that expert's `_delta_g_accum` views in tracker (δ_g = 0)
"""
from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Tuple,
)

import torch

from .async_writer import AsyncWriter, WriteRecord
from .binary_io import save_binary
from .expert_state import ExpertState

ExpertId = Tuple[int, int]  # (layer, expert_idx) — same shape as fc_manager

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
DS_EXPERT_IDX_RE = re.compile(r"deepspeed_experts\.(\d+)\.")


class ExpertLosslessFC:
    """Round-robin expert lossless FC scheduler + fire-er.

    Initialise once after fc_manager has enumerated expert tensors. Call
    ``fire_if_due(step)`` from the post-step hook BEFORE tracker.step;
    the returned list of fired expert ids must be passed to tracker.step
    as ``lossless_fc_eids`` so the tracker excludes them from the lossy
    selection AND zeroes their dg accumulator.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        # expert enumeration (passed in from fc_manager — already done there)
        all_expert_ids: List[ExpertId],
        separated_tensors_by_slot: Dict[ExpertId, List[str]],
        sorted_layers: List[int],
        # I/O
        expert_state: Optional[Any],
        async_writer: Optional[AsyncWriter],
        expert_output_dir: Path,
        keep_last_per_expert: int = 1,
        # schedule
        D: int = 4,
        schedule: str = "proposal",
        lossless_fc_fraction_pct: float = 2.0,
        # tracker integration
        tracker: Optional[Any] = None,
        tracker_stream: Optional["torch.cuda.Stream"] = None,
        # collect helper (delegated to fc_manager's _collect_param_payload
        # to avoid code duplication)
        collect_payload_fn: Optional[
            Callable[[List[str]], Tuple[Dict[str, torch.Tensor], int]]
        ] = None,
        # Owner of the `_last_collect_cuda_event` attribute (typically
        # the fc_manager whose `_collect_param_payload` we delegate to).
        event_owner: Optional[Any] = None,
    ) -> None:
        if schedule not in ("proposal", "sequential"):
            raise ValueError(
                f"schedule must be 'proposal' or 'sequential', got {schedule!r}"
            )
        self._model = model
        self._all_expert_ids = list(all_expert_ids)
        self._separated_tensors_by_slot = separated_tensors_by_slot
        self._sorted_layers = sorted_layers
        self._num_layers = len(sorted_layers)
        self._num_expert_slots = len(self._all_expert_ids)
        self.D = max(1, int(D))
        self._schedule = schedule
        self.lossless_fc_fraction_pct = float(lossless_fc_fraction_pct)
        if self._num_expert_slots > 0 and self.lossless_fc_fraction_pct > 0:
            self._experts_per_lossless_event = max(
                1,
                math.ceil(
                    self._num_expert_slots
                    * self.lossless_fc_fraction_pct / 100.0
                ),
            )
        else:
            self._experts_per_lossless_event = 0
        self._rr_cursor = 0
        self._layer_rr_tick = 0  # sequential mode

        self._expert_state = expert_state
        self._async_writer = async_writer
        self.expert_output_dir = Path(expert_output_dir)
        self.expert_output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_per_expert = int(keep_last_per_expert)

        self._tracker = tracker
        self._tracker_stream = tracker_stream
        self._collect_payload_fn = collect_payload_fn
        self._event_owner = event_owner

        # Legacy disk-mode rolling files (sequential / non-ExpertState path).
        self._expert_fc_files: Dict[
            ExpertId, List[Tuple[Path, Optional[WriteRecord]]]
        ] = {}

        # Per-tick diagnostic timing (host wall clock).
        self.last_phase_ms: Dict[str, float] = {}

    # ------------------------------------------------------------------
    @property
    def num_expert_slots(self) -> int:
        return self._num_expert_slots

    @property
    def experts_per_lossless_event(self) -> int:
        return self._experts_per_lossless_event

    # ------------------------------------------------------------------
    def _select_fired_experts(self, step: int) -> List[ExpertId]:
        """Pure host work — decide which experts fire this iter."""
        fired: List[ExpertId] = []
        if step <= 0:
            return fired
        if self._schedule == "sequential":
            # MoC paper: every D iter, fire all experts of ONE layer.
            if (
                self._num_layers > 0
                and (step - 1) % max(self.D, 1) == 0
            ):
                layer_idx = self._layer_rr_tick % self._num_layers
                layer = self._sorted_layers[layer_idx]
                fired = [eid for eid in self._all_expert_ids if eid[0] == layer]
                self._layer_rr_tick += 1
        else:  # proposal
            if (
                self._num_expert_slots > 0
                and self._experts_per_lossless_event > 0
                and step % max(self.D, 1) == 0
            ):
                n_fire = self._experts_per_lossless_event
                for i in range(n_fire):
                    slot = (self._rr_cursor + i) % self._num_expert_slots
                    fired.append(self._all_expert_ids[slot])
                self._rr_cursor = (
                    (self._rr_cursor + n_fire) % self._num_expert_slots
                )
        return fired

    # ------------------------------------------------------------------
    def fire_if_due(self, step: int) -> List[ExpertId]:
        """Fire any experts due this iter; returns the list. Runs all
        GPU work on tracker_stream (so it serializes with
        tracker.update_delta_accum and avoids main-stream contention)."""
        _t0 = time.perf_counter()
        fired = self._select_fired_experts(step)
        sub: Dict[str, float] = {
            "n_experts_fired": float(len(fired)),
            "collect_expert_ms": 0.0,
            "lossless_dispatch_ms": 0.0,
        }
        if not fired:
            self.last_phase_ms = sub
            return fired

        # Switch to tracker_stream for all GPU launches (collect + submit
        # event recording). Restore main stream on exit.
        prev_stream = None
        if self._tracker_stream is not None and torch.cuda.is_available():
            self._tracker_stream.wait_stream(torch.cuda.current_stream())
            prev_stream = torch.cuda.current_stream()
            torch.cuda.set_stream(self._tracker_stream)
        try:
            for eid in fired:
                ph = self._fire_one(eid, step)
                if ph is not None:
                    for k in ("collect_expert_ms", "lossless_dispatch_ms"):
                        sub[k] += ph.get(k, 0.0)
        finally:
            if prev_stream is not None:
                torch.cuda.set_stream(prev_stream)
        sub["total_ms"] = (time.perf_counter() - _t0) * 1000.0
        self.last_phase_ms = sub
        return fired

    # ------------------------------------------------------------------
    def _fire_one(
        self, expert_id: ExpertId, step: int,
    ) -> Optional[Dict[str, float]]:
        """Collect + submit a single expert's lossless FC. Per-phase
        timing dict (ms) returned for diagnostics."""
        layer, expert_idx = expert_id
        tnames = list(self._separated_tensors_by_slot.get(expert_id, []))
        if not tnames:
            return None

        # 1. Collect (delegated to fc_manager's helper for code reuse;
        #    helper handles per-source D2H + cuda_event). cuda_event is
        #    stashed on a configurable host object to break the lambda
        #    closure (so it's reliably retrievable here).
        _t_col0 = time.perf_counter()
        if self._collect_payload_fn is None:
            return None
        payload, payload_bytes = self._collect_payload_fn(tnames)
        # `_event_owner` is set at construction; fc_manager._collect_param
        # _payload stashes _last_collect_cuda_event on itself.
        collect_event = (
            getattr(self._event_owner, "_last_collect_cuda_event", None)
            if self._event_owner is not None else None
        )
        _col_ms = (time.perf_counter() - _t_col0) * 1000.0

        # 2. Submit to ExpertState (disk or DRAM). Both implementations
        #    take responsibility for truncating the expert's lossy FC
        #    log. cuda_event from collect is forwarded so the writer
        #    waits for the D2H to finish before reading host buffers.
        _t_dp0 = time.perf_counter()
        if self._expert_state is not None:
            initial_state: Dict[str, Tuple[
                torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
            ]] = {}
            for tname in tnames:
                p_cpu = payload.get(f"{tname}:param")
                if p_cpu is None:
                    continue
                m_cpu = payload.get(f"{tname}:exp_avg")
                v_cpu = payload.get(f"{tname}:exp_avg_sq")
                initial_state[tname] = (p_cpu, m_cpu, v_cpu)
            if initial_state:
                # Both DramExpertState and disk ExpertState accept
                # cuda_event and synchronise on it inside the writer
                # worker thread before reading the pinned host views,
                # so the training thread is NEVER blocked on the per-
                # source D2H or the host memcpy.
                self._expert_state.lossless_fc(
                    expert_id=expert_id,
                    step=int(step),
                    initial_state=initial_state,
                    cuda_event=collect_event,
                )
        _dp_ms = (time.perf_counter() - _t_dp0) * 1000.0

        # 4. Reset tracker dg (δ_g = 0) for this expert. Same
        #    tracker_stream so it FIFO-orders after this iter's
        #    update_delta_accum.
        if self._tracker is not None:
            d_acc = getattr(self._tracker, "_delta_g_accum", None) or {}
            for tname in tnames:
                d_g = d_acc.get(tname)
                if isinstance(d_g, torch.Tensor) and d_g.numel() > 0:
                    d_g.zero_()

        return {
            "collect_expert_ms": _col_ms,
            "lossless_dispatch_ms": _dp_ms,
        }
