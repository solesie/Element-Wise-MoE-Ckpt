"""Expert LFCL compaction scheduler.

Separated from fc_manager to keep checkpoint responsibilities split:

  - **lossy expert FC**: per-iter top-K element snapshot handled by
    lossy checkpoint engine.
  - **expert LFCL compaction** (this module): every iter, 1% of expert slots
    by the selected planner materialize live GPU p/m/v into an LFCL payload.
  - **non-expert FC** (still in fc_manager): every W iter, full non-expert
    weight + m + v snapshot.

Per-expert post-fire actions:
  1. plan fired experts before lossy selection
  2. publish the fired-expert mask so lossy probe/write skips them
  3. collect plan-target live GPU p/m/v into a pinned LFCL payload
  4. submit LFCL payload commit to ExpertState and rotate that expert's lossy log
"""
from __future__ import annotations

import heapq
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .lfcl_plan import (
    LFCLPlanner,
    PlannerState,
    RoundRobinPlan,
)

ExpertId = Tuple[int, int]  # (layer, expert_idx) — same shape as fc_manager


class ExpertLosslessFC:
    """Expert LFCL compaction scheduler + fire-er.

    Initialise once after fc_manager has enumerated expert tensors. The hot path
    calls ``plan_if_due(step)`` before lossy selection, publishes that plan as a
    fired mask, and then calls ``fire_planned(...)`` to materialize dense LFCL
    payloads.
    """

    def __init__(
        self,
        *,
        all_expert_ids: List[ExpertId],
        separated_tensors_by_slot: Dict[ExpertId, List[str]],
        sorted_layers: List[int],
        expert_state: Optional[Any],
        fc_manager: Optional[Any] = None,
        interval_iters: int = 1,
        schedule: str = "proposal",
        lfcl_fraction_pct: float = 1.0,
        planner: Optional[LFCLPlanner] = None,
        commit_payload: bool = True,
    ) -> None:
        if schedule not in ("proposal", "sequential"):
            raise ValueError(
                f"schedule must be 'proposal' or 'sequential', got {schedule!r}"
            )
        self._all_expert_ids = list(all_expert_ids)
        self._separated_tensors_by_slot = separated_tensors_by_slot
        self._sorted_layers = sorted_layers
        self._num_layers = len(sorted_layers)
        self._num_expert_slots = len(self._all_expert_ids)
        self.interval_iters = max(1, int(interval_iters))
        self._schedule = schedule
        self.lfcl_fraction_pct = float(lfcl_fraction_pct)
        if self._num_expert_slots > 0 and self.lfcl_fraction_pct > 0:
            self._experts_per_lfcl_event = max(
                1,
                math.ceil(self._num_expert_slots * self.lfcl_fraction_pct / 100.0),
            )
        else:
            self._experts_per_lfcl_event = 0
        self._planner = planner if planner is not None else RoundRobinPlan()
        self._layer_rr_tick = 0  # sequential mode

        self._expert_state = expert_state
        self._fc_manager = fc_manager
        self._commit_payload = bool(commit_payload)

        # Per-tick diagnostic timing (host wall clock).
        self.last_phase_ms: Dict[str, float] = {}
        self.last_d2h_event: Optional[Any] = None

    def _append_fire_log(self, step: int, fired: List[ExpertId]) -> None:
        path_tmpl = os.environ.get("PROPOSAL_LFCL_FIRE_LOG", "")
        if not path_tmpl or not fired:
            return
        rank = int(os.environ.get("RANK", "-1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = int(torch.distributed.get_rank())
        path = path_tmpl.format(rank=rank, local_rank=local_rank)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        import json

        record = {
            "step": int(step),
            "rank": rank,
            "local_rank": local_rank,
            "count": int(len(fired)),
            "experts": [[int(layer), int(expert)] for layer, expert in fired],
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    # ------------------------------------------------------------------
    @property
    def num_expert_slots(self) -> int:
        return self._num_expert_slots

    @property
    def experts_per_lfcl_event(self) -> int:
        return self._experts_per_lfcl_event

    # ------------------------------------------------------------------
    def _local_lossy_log_bytes(self) -> Dict[ExpertId, int]:
        lossy_log_bytes: Dict[ExpertId, int] = {}
        if self._expert_state is None:
            return lossy_log_bytes
        get_size = getattr(self._expert_state, "lossy_log_bytes", None)
        if not callable(get_size):
            return lossy_log_bytes
        for eid in self._all_expert_ids:
            lossy_log_bytes[eid] = int(get_size(eid))
        return lossy_log_bytes

    def _select_global_lossy_log_size_plan(
        self,
        *,
        step: int,
        process_group: Any,
    ) -> List[ExpertId]:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return self._select_fired_experts(step)
        rank = int(torch.distributed.get_rank())
        local_records = [
            (rank, idx, int(eid[0]), int(eid[1]), int(nbytes))
            for idx, (eid, nbytes) in enumerate(self._local_lossy_log_bytes().items())
        ]
        gathered: List[List[Tuple[int, int, int, int, int]]] = [
            [] for _ in range(int(torch.distributed.get_world_size(group=process_group)))
        ]
        torch.distributed.all_gather_object(
            gathered,
            local_records,
            group=process_group,
        )
        flat = [record for records in gathered for record in records]
        if not flat or self.lfcl_fraction_pct <= 0:
            return []
        n_to_fire = max(
            1,
            math.ceil(len(flat) * float(self.lfcl_fraction_pct) / 100.0),
        )
        winners = heapq.nsmallest(
            min(int(n_to_fire), len(flat)),
            flat,
            key=lambda item: (-int(item[4]), int(item[2]), int(item[3]), int(item[0])),
        )
        return [
            (int(layer), int(expert))
            for winner_rank, _idx, layer, expert, _nbytes in winners
            if int(winner_rank) == rank
        ]

    def _select_fired_experts(
        self,
        step: int,
        *,
        process_group: Any = None,
        world_size: int = 1,
    ) -> List[ExpertId]:
        """Pure host work — decide which experts fire this iter."""
        fired: List[ExpertId] = []
        if step <= 0:
            return fired
        if self._schedule == "sequential":
            # MoC paper baseline: fire all experts of one layer in round-robin order.
            if (
                self._num_layers > 0
                and (step - 1) % max(self.interval_iters, 1) == 0
            ):
                layer_idx = self._layer_rr_tick % self._num_layers
                layer = self._sorted_layers[layer_idx]
                fired = [eid for eid in self._all_expert_ids if eid[0] == layer]
                self._layer_rr_tick += 1
        else:
            due = step % max(self.interval_iters, 1) == 0
            can_fire_local = (
                self._num_expert_slots > 0 and self._experts_per_lfcl_event > 0
            )
            participates_global = int(world_size) > 1 and process_group is not None
            if due and (can_fire_local or participates_global):
                needs_log_bytes = bool(
                    getattr(
                        self._planner,
                        "requires_lossy_log_bytes",
                        lambda: True,
                    )()
                )
                if (
                    needs_log_bytes
                    and int(world_size) > 1
                    and process_group is not None
                ):
                    return self._select_global_lossy_log_size_plan(
                        step=int(step),
                        process_group=process_group,
                    )
                lossy_log_bytes = (
                    self._local_lossy_log_bytes() if needs_log_bytes else {}
                )
                fired = self._planner.select(
                    PlannerState(
                        expert_ids=list(self._all_expert_ids),
                        iter_T=int(step),
                        lossy_log_bytes=lossy_log_bytes,
                    ),
                    self._experts_per_lfcl_event,
                )
        return fired

    # ------------------------------------------------------------------
    def plan_if_due(
        self,
        step: int,
        *,
        process_group: Any = None,
        world_size: int = 1,
    ) -> List[ExpertId]:
        """Return experts due this iter without collecting payloads."""
        return self._select_fired_experts(
            int(step),
            process_group=process_group,
            world_size=int(world_size),
        )

    def fire_planned(
        self,
        step: int,
        fired: List[ExpertId],
        ready_event: Optional["Any"] = None,
    ) -> List[ExpertId]:
        """Materialize the already planned experts; returns the fired list."""
        _t0 = time.perf_counter()
        self.last_d2h_event = None
        sub: Dict[str, float] = {
            "n_experts_fired": float(len(fired)),
            "collect_expert_ms": 0.0,
            "lossless_dispatch_ms": 0.0,
        }
        if not fired:
            self.last_phase_ms = sub
            return fired

        if self._expert_state is not None:
            if self._fc_manager is None:
                raise RuntimeError(
                    "expert LFCL requires FCManager live-GPU pinned payload "
                    "collection in V2."
                )
            targets = [
                (eid, list(self._separated_tensors_by_slot.get(eid, [])))
                for eid in fired
                if self._separated_tensors_by_slot.get(eid)
            ]
            _t_dp0 = time.perf_counter()
            batch_info = self._fc_manager.collect_lfcl_batch_payload(
                targets,
                ready_event=ready_event,
            )
            collect_sub = dict(
                getattr(self._fc_manager, "_last_lfcl_batch_sub_ms", {}) or {}
            )
            cuda_event = getattr(
                self._fc_manager,
                "_last_lfcl_batch_cuda_event",
                None,
            )
            self.last_d2h_event = cuda_event
            if batch_info is not None:
                self._expert_state.compact_lossy_recovery_images_from_pinned_batch(
                    step=int(step),
                    batch_info=batch_info,
                    cuda_event=cuda_event,
                    commit_payload=self._commit_payload,
                )
            _dp_ms = (time.perf_counter() - _t_dp0) * 1000.0
            sub["lossless_dispatch_ms"] = _dp_ms
            sub["payload_bytes"] = float(
                0 if batch_info is None else batch_info.get("payload_bytes", 0)
            )
            for k in ("enum_ms", "xfer_ms", "d2h_copy_cmds", "wait_ms"):
                sub[k] = float(collect_sub.get(k, 0.0))
        sub["total_ms"] = (time.perf_counter() - _t0) * 1000.0
        self.last_phase_ms = sub
        self._append_fire_log(int(step), fired)
        if os.environ.get("PROPOSAL_LFCL_PROFILE", "") in ("1", "true", "TRUE"):
            print(
                f"[proposal] lfcl_profile @iter{int(step)}: "
                f"fired={len(fired)} commit={int(self._commit_payload)} "
                f"total={sub['total_ms']:.1f}ms "
                f"dispatch={sub['lossless_dispatch_ms']:.1f}ms "
                f"payload_mb={sub.get('payload_bytes', 0.0) / 1e6:.1f} "
                f"enum={sub.get('enum_ms', 0.0):.1f}ms "
                f"wait={sub.get('wait_ms', 0.0):.1f}ms "
                f"xfer_submit={sub.get('xfer_ms', 0.0):.1f}ms "
                f"d2h_cmds={int(sub.get('d2h_copy_cmds', 0.0))}",
                flush=True,
            )
        return fired

    def fire_if_due(
        self,
        step: int,
        ready_event: Optional["Any"] = None,
    ) -> List[ExpertId]:
        """Compatibility wrapper for callers that do not need pre-selection."""
        fired = self.plan_if_due(int(step))
        return self.fire_planned(int(step), fired, ready_event=ready_event)
