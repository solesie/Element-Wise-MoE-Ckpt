"""Expert LFCL compaction planners."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Tuple


ExpertId = Tuple[int, int]


@dataclass
class PlannerState:
    """CPU-side state visible to LFCL planners."""

    expert_ids: List[ExpertId]
    iter_T: int
    lossy_log_bytes: Dict[ExpertId, int] = field(default_factory=dict)


class LFCLPlanner(Protocol):
    def select(self, planner_state: PlannerState, n_to_fire: int) -> List[ExpertId]:
        """Return experts to compact this D-boundary."""


class RoundRobinPlan:
    """Deterministic round-robin expert compaction plan."""

    def __init__(self) -> None:
        self._cursor = 0

    def requires_lossy_log_bytes(self) -> bool:
        return False

    def select(self, planner_state: PlannerState, n_to_fire: int) -> List[ExpertId]:
        ids = list(planner_state.expert_ids)
        if not ids or n_to_fire <= 0:
            return []
        out: List[ExpertId] = []
        for _ in range(min(int(n_to_fire), len(ids))):
            out.append(ids[self._cursor % len(ids)])
            self._cursor = (self._cursor + 1) % len(ids)
        return out


class LossyLogSizeBasedPlan:
    """Compact experts with the largest current lossy-log byte size."""

    def requires_lossy_log_bytes(self) -> bool:
        return True

    def select(self, planner_state: PlannerState, n_to_fire: int) -> List[ExpertId]:
        if n_to_fire <= 0:
            return []
        n = min(int(n_to_fire), len(planner_state.lossy_log_bytes))
        if n <= 0:
            return []
        return [
            eid for eid, _size in heapq.nsmallest(
                n,
                planner_state.lossy_log_bytes.items(),
                key=lambda kv: (-int(kv[1]), kv[0]),
            )
        ]
