"""Versioned lagged cutoff state for PP-friendly selection.

This module owns only CPU metadata.  It does not change the cutoff math: strict
probe/direct-feedback observations still train ``AdaptiveCutoffController``;
this state only decides when a previously published tau is safe to reuse.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass(frozen=True)
class TauSnapshot:
    version: int
    tau: float
    source_step: int
    apply_from_step: int
    h_epoch: int
    score_source: str


class PPLaggedTauState:
    """Small version table for PP lagged cutoff reuse."""

    def __init__(
        self,
        *,
        target_ratio: float,
        band_fraction: float = 0.25,
        safety_lag: int = 1,
        max_staleness: int = 8,
        strict_after_bad: int = 1,
    ) -> None:
        self.target_ratio = max(float(target_ratio), 1.0e-12)
        self.band_fraction = max(0.0, float(band_fraction))
        self.safety_lag = max(1, int(safety_lag))
        self.max_staleness = max(self.safety_lag, int(max_staleness))
        self.strict_after_bad = max(1, int(strict_after_bad))
        self._latest: Optional[TauSnapshot] = None
        self._next_version = 1
        self._bad_streak = 0
        self._feedback_tau_rows: Deque[Tuple[int, float]] = deque(maxlen=8)
        self._publish_epoch: Optional[int] = None
        self._epoch_in_band_count = 0
        self._require_fresh_publish = False
        self.last_strict_reason = "init"

    @property
    def latest(self) -> Optional[TauSnapshot]:
        return self._latest

    def band(self) -> Tuple[float, float]:
        width = self.target_ratio * self.band_fraction
        return max(0.0, self.target_ratio - width), self.target_ratio + width

    def observe_ratio(self, ratio: float) -> bool:
        lo, hi = self.band()
        ratio_f = float(ratio)
        in_band = bool(lo <= ratio_f <= hi)
        self._bad_streak = 0 if in_band else self._bad_streak + 1
        if not in_band:
            self._require_fresh_publish = True
        return in_band

    def observe_publish_readiness(self, *, h_epoch: int, in_band: bool) -> bool:
        if self._publish_epoch != int(h_epoch):
            self._publish_epoch = int(h_epoch)
            self._epoch_in_band_count = 0
        if bool(in_band):
            self._epoch_in_band_count += 1
        else:
            self._epoch_in_band_count = 0
        return int(self._epoch_in_band_count) >= 2

    def observe_feedback_tau(
        self,
        *,
        step: int,
        tau: float,
        selected_ratio: float,
    ) -> None:
        tau_f = float(tau)
        ratio_f = float(selected_ratio)
        if (
            not math.isfinite(tau_f)
            or tau_f <= 0.0
            or not math.isfinite(ratio_f)
            or ratio_f <= 0.0
        ):
            return
        target = max(float(self.target_ratio), 1.0e-12)
        tau_at_target = tau_f * (ratio_f / target)
        if not math.isfinite(tau_at_target) or tau_at_target <= 0.0:
            return
        self._feedback_tau_rows.append((int(step), float(tau_at_target)))

    def trend_adjusted_tau(self, *, proposed_tau: float, source_step: int) -> float:
        proposed = float(proposed_tau)
        if not math.isfinite(proposed) or proposed <= 0.0:
            return proposed
        rows = [
            (int(step), float(tau))
            for step, tau in self._feedback_tau_rows
            if math.isfinite(float(tau)) and float(tau) > 0.0
        ]
        if len(rows) < 2:
            return proposed
        deltas = []
        for (prev_step, prev_tau), (curr_step, curr_tau) in zip(rows, rows[1:]):
            span = max(1, int(curr_step) - int(prev_step))
            delta = (math.log(curr_tau) - math.log(prev_tau)) / float(span)
            if math.isfinite(delta):
                deltas.append(float(delta))
        if not deltas:
            return proposed
        # Lagged reuse is most sensitive to fast one-step score-scale motion.
        # Use the risk-side recent drift: falling tau-star would under-select if
        # ignored, rising tau-star would over-select if ignored.
        falling = [d for d in deltas[-4:] if d < 0.0]
        rising = [d for d in deltas[-4:] if d > 0.0]
        if falling:
            drift = max(min(falling), -1.0)
        elif rising:
            drift = min(max(rising), 1.0)
        else:
            return proposed
        last_step, last_tau = rows[-1]
        horizon = max(1, int(source_step) + int(self.safety_lag) - int(last_step))
        trended = float(last_tau) * math.exp(float(drift) * float(horizon))
        if not math.isfinite(trended) or trended <= 0.0:
            return proposed
        if drift < 0.0:
            return min(proposed, float(trended))
        return max(proposed, float(trended))

    def bridge_guarded_tau(
        self,
        *,
        proposed_tau: float,
        feedback_tau: float,
        lr_bridge: float,
    ) -> float:
        proposed = float(proposed_tau)
        if not math.isfinite(proposed) or proposed <= 0.0:
            return proposed
        feedback = float(feedback_tau)
        bridge = float(lr_bridge)
        if (
            not math.isfinite(feedback)
            or feedback <= 0.0
            or not math.isfinite(bridge)
            or bridge <= 0.0
        ):
            return proposed
        center = feedback * bridge
        if not math.isfinite(center) or center <= 0.0:
            return proposed
        width = min(max(float(self.band_fraction), 0.0), 0.9)
        lower_factor = max(0.05, 1.0 - 1.5 * width)
        if bridge < max(0.05, 2.0 * width):
            lower_factor = 0.05
        lo = center * lower_factor
        hi = center * (1.0 + width)
        return min(max(proposed, lo), hi)

    def publish(
        self,
        *,
        tau: float,
        source_step: int,
        h_epoch: int,
        score_source: str,
    ) -> Optional[TauSnapshot]:
        tau_f = float(tau)
        if not math.isfinite(tau_f) or tau_f <= 0.0:
            return None
        snapshot = TauSnapshot(
            version=int(self._next_version),
            tau=tau_f,
            source_step=int(source_step),
            apply_from_step=int(source_step) + int(self.safety_lag),
            h_epoch=int(h_epoch),
            score_source=str(score_source),
        )
        self._next_version += 1
        self._latest = snapshot
        self._require_fresh_publish = False
        return snapshot

    def lookup(
        self,
        *,
        step: int,
        h_epoch: int,
        score_source: str,
        exact_required: bool,
    ) -> Tuple[Optional[TauSnapshot], str]:
        if bool(exact_required):
            self.last_strict_reason = "strict_required"
            return None, self.last_strict_reason
        snapshot = self._latest
        if snapshot is None:
            self.last_strict_reason = "no_snapshot"
            return None, self.last_strict_reason
        if int(snapshot.h_epoch) != int(h_epoch):
            self.last_strict_reason = "h_epoch_mismatch"
            return None, self.last_strict_reason
        if str(snapshot.score_source) != str(score_source):
            self.last_strict_reason = "score_source_mismatch"
            return None, self.last_strict_reason
        if int(step) < int(snapshot.apply_from_step):
            self.last_strict_reason = "not_yet_applicable"
            return None, self.last_strict_reason
        if int(step) - int(snapshot.source_step) > int(self.max_staleness):
            self.last_strict_reason = "stale_snapshot"
            return None, self.last_strict_reason
        if bool(self._require_fresh_publish):
            self.last_strict_reason = "publish_warmup"
            return None, self.last_strict_reason
        if int(self._bad_streak) >= int(self.strict_after_bad):
            self.last_strict_reason = "out_of_band"
            return None, self.last_strict_reason
        return snapshot, "lagged"
