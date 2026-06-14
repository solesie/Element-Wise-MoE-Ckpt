"""Removable telemetry for approximate cutoff probe experiments.

This module is intentionally isolated from the production cutoff algorithm.
Set ``PROPOSAL_APPROX_PROBE_TELEMETRY_LOG=/path/to/file.jsonl`` to enable it;
with the env unset, callers pay only a cheap boolean check.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


class ApproxProbeTelemetry:
    """Async, env-gated probe telemetry.

    The class owns only measurement buffers/events. It does not choose tau and
    does not feed the controller. Delete this file plus the three callsites in
    ``checkpoint/lossy_checkpoint_engine.py`` to remove the experiment.
    """

    def __init__(self, path: str, *, ring_size: int = 32) -> None:
        self.enabled = bool(path)
        self._path = Path(path) if path else None
        self._ring_size = max(2, int(ring_size))
        self._handle = None
        self._diag_host: Optional[torch.Tensor] = None
        self._code_host: Optional[torch.Tensor] = None
        self._probe_reject_host: Optional[torch.Tensor] = None
        self._sum_q_host: Optional[torch.Tensor] = None
        self._sum_q2_host: Optional[torch.Tensor] = None
        self._n_blocks_host: Optional[torch.Tensor] = None
        self._copy_events = []
        self._probe_start_events = []
        self._probe_done_events = []
        self._decision_done_events = []
        self._pending: Dict[int, Dict[str, Any]] = {}
        if self.enabled:
            assert self._path is not None
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", buffering=1)

    @classmethod
    def from_env(cls) -> "ApproxProbeTelemetry":
        return cls(os.environ.get("PROPOSAL_APPROX_PROBE_TELEMETRY_LOG", ""))

    def ensure_buffers(self) -> None:
        if not self.enabled or self._diag_host is not None:
            return
        self._diag_host = torch.empty(
            (self._ring_size, 8), dtype=torch.float32, pin_memory=True,
        )
        self._code_host = torch.empty(
            self._ring_size, dtype=torch.int32, pin_memory=True,
        )
        self._probe_reject_host = torch.empty(
            self._ring_size, dtype=torch.int32, pin_memory=True,
        )
        for _ in range(self._ring_size):
            self._copy_events.append(torch.cuda.Event())
            self._probe_start_events.append(torch.cuda.Event(enable_timing=True))
            self._probe_done_events.append(torch.cuda.Event(enable_timing=True))
            self._decision_done_events.append(torch.cuda.Event(enable_timing=True))

    def slot_for_step(self, step: int) -> int:
        return int(step) % self._ring_size

    def _ensure_sample_buffers(self, *, n_strata: int, n_candidates: int) -> None:
        self.ensure_buffers()
        n_strata_i = max(1, int(n_strata))
        n_candidates_i = max(1, int(n_candidates))
        shape = (self._ring_size, n_strata_i, n_candidates_i)
        if self._sum_q_host is not None and tuple(self._sum_q_host.shape) == shape:
            return
        self._sum_q_host = torch.empty(
            shape,
            dtype=torch.float32,
            pin_memory=True,
        )
        self._sum_q2_host = torch.empty_like(self._sum_q_host)
        self._n_blocks_host = torch.empty(
            shape,
            dtype=torch.int32,
            pin_memory=True,
        )

    def record_probe_start(self, *, step: int, stream: torch.cuda.Stream) -> None:
        if not self.enabled:
            return
        self.ensure_buffers()
        self._probe_start_events[self.slot_for_step(step)].record(stream)

    def record_probe_done(self, *, step: int, stream: torch.cuda.Stream) -> None:
        if not self.enabled:
            return
        self._probe_done_events[self.slot_for_step(step)].record(stream)

    def record_decision_done(self, *, step: int, stream: torch.cuda.Stream) -> None:
        if not self.enabled:
            return
        self._decision_done_events[self.slot_for_step(step)].record(stream)

    def record_device_result(
        self,
        *,
        step: int,
        stream: torch.cuda.Stream,
        diagnostics: torch.Tensor,
        decision_code: torch.Tensor,
        probe_reject: torch.Tensor,
        n_candidates: int,
        n_strata: int,
        total_entry_blocks: int,
        block_elements: int,
        sum_q: Optional[torch.Tensor] = None,
        sum_q2: Optional[torch.Tensor] = None,
        n_blocks: Optional[torch.Tensor] = None,
        stratum_total_elems: Optional[Iterable[int]] = None,
    ) -> None:
        if not self.enabled:
            return
        self.ensure_buffers()
        assert self._diag_host is not None
        assert self._code_host is not None
        assert self._probe_reject_host is not None
        slot = self.slot_for_step(step)
        self._diag_host[slot].copy_(diagnostics.reshape(-1)[:8], non_blocking=True)
        self._code_host[slot].copy_(decision_code.reshape(()), non_blocking=True)
        self._probe_reject_host[slot].copy_(probe_reject.reshape(()), non_blocking=True)
        if sum_q is not None and sum_q2 is not None and n_blocks is not None:
            self._ensure_sample_buffers(
                n_strata=int(n_strata),
                n_candidates=int(n_candidates),
            )
            assert self._sum_q_host is not None
            assert self._sum_q2_host is not None
            assert self._n_blocks_host is not None
            self._sum_q_host[slot, :n_strata, :n_candidates].copy_(
                sum_q[:n_strata, :n_candidates],
                non_blocking=True,
            )
            self._sum_q2_host[slot, :n_strata, :n_candidates].copy_(
                sum_q2[:n_strata, :n_candidates],
                non_blocking=True,
            )
            self._n_blocks_host[slot, :n_strata, :n_candidates].copy_(
                n_blocks[:n_strata, :n_candidates],
                non_blocking=True,
            )
        self._copy_events[slot].record(stream)
        self._pending[int(step)] = {
            "slot": int(slot),
            "n_candidates": int(n_candidates),
            "n_strata": int(n_strata),
            "total_entry_blocks": int(total_entry_blocks),
            "block_elements": int(block_elements),
            "stratum_total_elems": list(stratum_total_elems or []),
        }

    def emit_with_direct_feedback(
        self,
        *,
        direct_feedback: Dict[str, Any],
        total_elems_global: int,
        target_ratio: float,
    ) -> None:
        if not self.enabled or self._handle is None:
            return
        step = int(direct_feedback.get("step", -1))
        pending = self._pending.pop(step, None)
        if pending is None:
            return
        slot = int(pending["slot"])
        if not self._copy_events[slot].query():
            self._pending[step] = pending
            return
        assert self._diag_host is not None
        assert self._code_host is not None
        assert self._probe_reject_host is not None
        diag = self._diag_host[slot].tolist()
        selected = int(direct_feedback.get("selected", 0))
        written = int(direct_feedback.get("written", 0))
        overflow = int(direct_feedback.get("overflow", 0))
        total = max(1, int(total_elems_global))
        actual_ratio = float(selected) / float(total)
        written_ratio = float(written) / float(total)
        approx_ratio = float(diag[0])
        target = max(float(target_ratio), 1.0e-12)
        sampled_blocks = max(0.0, float(diag[6]))
        sampled_elements = sampled_blocks * float(pending["block_elements"])
        record = {
            "step": step,
            "tau": float(direct_feedback.get("tau", 0.0)),
            "approx_ratio": approx_ratio,
            "actual_selected_ratio": actual_ratio,
            "actual_written_ratio": written_ratio,
            "target_ratio": target,
            "abs_error": abs(approx_ratio - actual_ratio),
            "rel_error_to_target": abs(approx_ratio - actual_ratio) / target,
            "k_rel_error": (actual_ratio / target) - 1.0,
            "selected": selected,
            "written": written,
            "overflow": overflow,
            "decision_code": int(self._code_host[slot].item()),
            "probe_reject": int(self._probe_reject_host[slot].item()),
            "strict_accept": bool(float(diag[5]) >= 0.5),
            "probe_accept": bool(float(diag[7]) >= 0.5),
            "chosen_se": float(diag[1]),
            "best_bound": float(diag[2]),
            "approx_tau": float(diag[3]),
            "next_sample_blocks": int(float(diag[4])),
            "sampled_blocks": int(sampled_blocks),
            "sampled_elements": int(sampled_elements),
            "sample_fraction": sampled_elements / float(total),
            "n_candidates": int(pending["n_candidates"]),
            "n_strata": int(pending["n_strata"]),
            "total_entry_blocks": int(pending["total_entry_blocks"]),
            "probe_ms": self._elapsed_ms(
                self._probe_start_events[slot], self._probe_done_events[slot],
            ),
            "decision_ms": self._elapsed_ms(
                self._probe_done_events[slot], self._decision_done_events[slot],
            ),
        }
        if self._sum_q_host is not None:
            record["candidate_sample"] = self._candidate_sample_record(
                slot=slot,
                pending=pending,
                target=target,
                total=total,
            )
        expert_summary = self._expert_overflow_summary(direct_feedback)
        if expert_summary:
            record["expert_overflow_summary"] = expert_summary
        self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _expert_overflow_summary(
        self,
        direct_feedback: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        selected = list(direct_feedback.get("selected_by_expert") or [])
        overflow = list(direct_feedback.get("overflow_by_expert") or [])
        capacity = list(direct_feedback.get("capacity_by_expert") or [])
        max_records = list(direct_feedback.get("max_records_by_expert") or [])
        offsets = list(direct_feedback.get("aos_offset_by_expert") or [])
        expert_ids = list(direct_feedback.get("expert_ids") or [])
        n = min(
            len(selected), len(overflow), len(capacity), len(max_records),
            len(offsets), len(expert_ids),
        )
        if n <= 0:
            return None
        topk = max(
            0,
            int(os.environ.get("PROPOSAL_EXPERT_OVERFLOW_TELEMETRY_TOPK", "8")),
        )
        rows = []
        for idx in range(n):
            sel = max(0, int(selected[idx]))
            over = max(0, int(overflow[idx]))
            cap = max(0, int(capacity[idx]))
            max_rec = max(0, int(max_records[idx]))
            written = max(0, sel - over)
            rows.append({
                "idx": int(idx),
                "expert": expert_ids[idx],
                "selected": sel,
                "written": written,
                "overflow": over,
                "capacity": cap,
                "max_records": max_rec,
                "offset": int(offsets[idx]),
                "overflow_over_selected": (
                    float(over) / float(sel) if sel > 0 else 0.0
                ),
                "selected_over_capacity": (
                    float(sel) / float(cap) if cap > 0 else 0.0
                ),
                "capacity_over_max_records": (
                    float(cap) / float(max_rec) if max_rec > 0 else 0.0
                ),
                "full_expert_aos_mb": float(max_rec) * 16.0 / 1.0e6,
            })
        rows_by_overflow = sorted(
            rows, key=lambda r: (int(r["overflow"]), int(r["selected"])),
            reverse=True,
        )
        rows_by_selected = sorted(
            rows, key=lambda r: int(r["selected"]), reverse=True,
        )
        return {
            "n_experts": int(n),
            "sum_capacity": int(sum(max(0, int(x)) for x in capacity[:n])),
            "sum_max_records": int(sum(max(0, int(x)) for x in max_records[:n])),
            "max_capacity": int(max(max(0, int(x)) for x in capacity[:n])),
            "max_records": int(max(max(0, int(x)) for x in max_records[:n])),
            "top_overflow": rows_by_overflow[:topk],
            "top_selected": rows_by_selected[:topk],
        }

    def _candidate_sample_record(
        self,
        *,
        slot: int,
        pending: Dict[str, Any],
        target: float,
        total: int,
    ) -> Dict[str, Any]:
        assert self._sum_q_host is not None
        assert self._sum_q2_host is not None
        assert self._n_blocks_host is not None
        n_candidates = int(pending["n_candidates"])
        n_strata = int(pending["n_strata"])
        stratum_total = [float(x) for x in pending.get("stratum_total_elems", [])]
        if len(stratum_total) < n_strata:
            stratum_total = stratum_total + [0.0] * (n_strata - len(stratum_total))
        weights = [x / float(max(1, total)) for x in stratum_total[:n_strata]]
        sample_ratios = []
        sample_se = []
        for cand in range(n_candidates):
            ratio = 0.0
            var = 0.0
            for sid in range(n_strata):
                n = int(self._n_blocks_host[slot, sid, cand].item())
                if n <= 0:
                    continue
                sum_q = float(self._sum_q_host[slot, sid, cand].item())
                sum_q2 = float(self._sum_q2_host[slot, sid, cand].item())
                mean = sum_q / float(n)
                ratio += weights[sid] * mean
                numer = max(sum_q2 - (sum_q * sum_q) / float(max(1, n)), 0.0)
                s2 = numer / float(max(1, n - 1)) if n > 1 else 0.0
                var += weights[sid] * weights[sid] * s2 / float(max(1, n))
            sample_ratios.append(ratio)
            sample_se.append(1.96 * (var ** 0.5))
        sample_best_idx = min(
            range(n_candidates),
            key=lambda idx: abs(sample_ratios[idx] - target),
        ) if n_candidates > 0 else -1
        return {
            "sample_ratios": sample_ratios,
            "sample_se": sample_se,
            "sample_best_idx": int(sample_best_idx),
            "sample_best_sample_ratio": (
                sample_ratios[sample_best_idx] if sample_best_idx >= 0 else 0.0
            ),
        }

    @staticmethod
    def _elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> Optional[float]:
        try:
            if not end.query():
                return None
            return float(start.elapsed_time(end))
        except Exception:
            return None
