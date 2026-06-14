"""Adaptive cutoff controller for steady-state lossy selection.

The controller itself owns no CUDA storage. It tracks recent selected ratios
and predicts the next score cutoff from the observed log(tau) ->
log(selected_ratio) curve. The async feedback adapter at the bottom of this
module only copies small sampled-ratio summaries into pinned host buffers so
the controller can learn without synchronizing the proposal stream.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class AdaptiveCutoffConfig:
    enabled: bool = True
    target_ratio: float = 0.04             # = K budget (topk_ratio); default 4%, set from topk_ratio at build
    tolerance: float = 0.015
    window_size: int = 16
    min_observations: int = 8
    warmup_iters: int = 0
    retry_tolerance: float = 0.02


@dataclass(frozen=True)
class CutoffObservation:
    step: int
    tau: float
    selected_ratio: float
    exact: bool


@dataclass(frozen=True)
class CutoffDecision:
    use_exact: bool
    tau: Optional[float]
    reason: str


@dataclass(frozen=True)
class StableTauCertificate:
    safe: bool
    reason: str
    predicted_ratio: Optional[float] = None
    residual_log_ratio: Optional[float] = None


@dataclass(frozen=True)
class _TauStarFit:
    predicted_log_tau_star: float
    gain: float
    residual_log_tau_star: float


class AdaptiveCutoffController:
    """Sliding-window controller for Amber-style threshold adjustment.

    The controller deliberately reasons over a window, not a single iteration:
    selected-ratio noise should be interpreted through the measured cutoff
    response curve rather than fixed gain constants.
    """

    def __init__(self, config: AdaptiveCutoffConfig) -> None:
        self.config = config
        self._observations: Deque[CutoffObservation] = deque(
            maxlen=max(1, int(config.window_size)),
        )
        self.measured_alpha_rows: Deque[float] = deque(maxlen=16)
        self.tau_star_rows: Deque[float] = deque(maxlen=16)
        self.tau_star_step_rows: Deque[Tuple[int, float]] = deque(maxlen=16)
        self.tau_error_gains: Deque[float] = deque(maxlen=16)

    @property
    def observations(self) -> tuple[CutoffObservation, ...]:
        return tuple(self._observations)

    def observe(
        self,
        *,
        step: int,
        tau: float,
        selected_ratio: float,
        exact: bool,
    ) -> None:
        if not math.isfinite(float(tau)) or float(tau) <= 0.0:
            return
        ratio = max(0.0, float(selected_ratio))
        if ratio <= 0.0 or not math.isfinite(ratio):
            return
        obs = CutoffObservation(
            step=int(step),
            tau=float(tau),
            selected_ratio=ratio,
            exact=bool(exact),
        )
        self._observations.append(obs)

    def decide(self, *, step: int) -> CutoffDecision:
        cfg = self.config
        if not cfg.enabled:
            return CutoffDecision(True, None, "disabled")
        if int(step) <= int(cfg.warmup_iters):
            return CutoffDecision(True, None, "warmup")
        if len(self._observations) < max(1, int(cfg.min_observations)):
            return CutoffDecision(True, None, "insufficient_window")
        tau = self._predict_tau(step=int(step))
        if tau is None:
            return CutoffDecision(True, None, "invalid_prediction")
        return CutoffDecision(False, tau, "adaptive")

    def count_rows_to_ratio_rows(
        self,
        rows: List[Tuple[float, int]],
        *,
        total_elems_global: int,
    ) -> List[Tuple[float, float]]:
        n_global = max(1, int(total_elems_global))
        deduped_reversed: List[Tuple[float, int]] = []
        for tau, count in reversed(rows):
            if not _valid_positive_threshold(float(tau)) or int(count) <= 0:
                continue
            if any(
                abs(math.log(float(tau)) - math.log(old_tau)) <= 1.0e-6
                for old_tau, _ in deduped_reversed
            ):
                continue
            deduped_reversed.append((float(tau), int(count)))
        ratio_rows: List[Tuple[float, float]] = []
        for tau, count in reversed(deduped_reversed):
            ratio = float(max(0, int(count))) / float(n_global)
            if math.isfinite(ratio) and 0.0 < ratio <= 1.0:
                ratio_rows.append((float(tau), float(ratio)))
        return ratio_rows

    def estimate_measured_alpha_tau(
        self,
        ratio_rows: List[Tuple[float, float]],
    ) -> Tuple[Optional[float], Optional[float], str]:
        return estimate_measured_alpha_tau(
            ratio_rows,
            target_ratio=float(self.config.target_ratio),
        )

    def remember_measured_alpha_estimate(
        self,
        *,
        tau_star: Optional[float],
        alpha: Optional[float],
        step: Optional[int] = None,
    ) -> None:
        if (
            tau_star is None
            or alpha is None
            or not _valid_positive_threshold(float(tau_star))
            or not math.isfinite(float(alpha))
            or float(alpha) <= 0.0
        ):
            return
        self.measured_alpha_rows.append(float(alpha))
        self.tau_star_rows.append(float(tau_star))
        if step is not None:
            self.tau_star_step_rows.append((int(step), float(tau_star)))
        gain = 1.0 / float(alpha)
        if math.isfinite(gain) and gain > 0.0:
            self.tau_error_gains.append(float(gain))

    def predicted_tau_star(self) -> Optional[float]:
        tau_stars = [
            float(t)
            for t in self.tau_star_rows
            if _valid_positive_threshold(float(t))
        ]
        if not tau_stars:
            return None
        predicted = float(tau_stars[-1]) * math.exp(
            self.recent_measured_tau_star_drift()
        )
        if not _valid_positive_threshold(float(predicted)):
            return None
        return float(predicted)

    def observe_final_count_feedback(
        self,
        rows: List[Tuple[float, int]],
        *,
        step: Optional[int] = None,
        current_tau: float,
        selected_ratio: float,
        total_elems_global: int,
        cutoff_tolerance: float,
    ) -> None:
        """Learn the tau-star coordinate from the final exact count row.

        Candidate count is not the only source of cutoff feedback. The accepted
        threshold is followed by direct-emission cursor feedback, so its realized
        ratio can keep the tau-star drift model current without host-side count scans.
        """

        if (
            not _valid_positive_threshold(float(current_tau))
            or not math.isfinite(float(selected_ratio))
            or float(selected_ratio) <= 0.0
        ):
            return
        ratio_rows = self.count_rows_to_ratio_rows(
            rows,
            total_elems_global=int(total_elems_global),
        )
        tau_star, alpha, _reason = self.estimate_measured_alpha_tau(ratio_rows)
        if (
            _valid_positive_threshold(float(tau_star or 0.0))
            and alpha is not None
            and math.isfinite(float(alpha))
            and float(alpha) > 0.0
        ):
            self.remember_measured_alpha_estimate(
                tau_star=float(tau_star),
                alpha=float(alpha),
                step=step,
            )
            return
        gain = self.recent_measured_alpha_gain()
        if gain is None or not math.isfinite(float(gain)) or float(gain) <= 0.0:
            return
        target = max(float(self.config.target_ratio), 1.0e-12)
        strict_slack = max(
            float(cutoff_tolerance),
            float(self.config.retry_tolerance),
        )
        if abs(float(selected_ratio) - target) > target * strict_slack:
            return
        self.remember_measured_alpha_estimate(
            tau_star=float(current_tau),
            alpha=1.0 / float(gain),
            step=step,
        )

    def bridge_refresh_scale(self, scale: float) -> bool:
        """Move the controller's tau coordinate after a score remap.

        Refresh changes H(v) but not p/m/v. If the score axis is multiplied
        by an approximate scalar, previous tau observations remain useful after
        the same scalar is applied to their tau coordinates.
        """

        kappa = float(scale)
        if not math.isfinite(kappa) or kappa <= 0.0:
            return False
        if abs(math.log(kappa)) <= 1.0e-9:
            return False
        self._observations = deque(
            (
                CutoffObservation(
                    step=o.step,
                    tau=float(o.tau) * kappa,
                    selected_ratio=o.selected_ratio,
                    exact=o.exact,
                )
                for o in self._observations
                if _valid_positive_threshold(float(o.tau) * kappa)
            ),
            maxlen=self._observations.maxlen,
        )
        self.tau_star_rows = deque(
            (
                float(t) * kappa
                for t in self.tau_star_rows
                if _valid_positive_threshold(float(t) * kappa)
            ),
            maxlen=self.tau_star_rows.maxlen,
        )
        self.tau_star_step_rows = deque(
            (
                (int(step), float(t) * kappa)
                for step, t in self.tau_star_step_rows
                if _valid_positive_threshold(float(t) * kappa)
            ),
            maxlen=self.tau_star_step_rows.maxlen,
        )
        return True

    def forget_tau_model(self) -> None:
        """Drop learned tau-scale rows after a non-scalar score remap."""

        self.measured_alpha_rows.clear()
        self.tau_star_rows.clear()
        self.tau_star_step_rows.clear()
        self.tau_error_gains.clear()

    def recent_measured_alpha_gain(self) -> Optional[float]:
        alphas = [
            float(a)
            for a in self.measured_alpha_rows
            if math.isfinite(float(a)) and float(a) > 0.0
        ]
        if not alphas:
            return None
        alphas.sort()
        alpha = float(alphas[len(alphas) // 2])
        if not math.isfinite(alpha) or alpha <= 0.0:
            return None
        return 1.0 / alpha

    def recent_measured_tau_star_drift(self) -> float:
        tau_stars = [
            float(t)
            for t in self.tau_star_rows
            if _valid_positive_threshold(float(t))
        ]
        if len(tau_stars) < 3:
            return 0.0
        deltas = [
            math.log(float(curr)) - math.log(float(prev))
            for prev, curr in zip(tau_stars, tau_stars[1:])
            if _valid_positive_threshold(float(prev))
            and _valid_positive_threshold(float(curr))
        ]
        deltas = [float(d) for d in deltas if math.isfinite(float(d))]
        if not deltas:
            return 0.0
        recent = sorted(deltas[-min(5, len(deltas)):])
        return float(recent[len(recent) // 2])

    def recent_measured_tau_star_abs_drift(self) -> Optional[float]:
        tau_stars = [
            float(t)
            for t in self.tau_star_rows
            if _valid_positive_threshold(float(t))
        ]
        if len(tau_stars) < 3:
            return None
        deltas = [
            abs(math.log(float(curr)) - math.log(float(prev)))
            for prev, curr in zip(tau_stars, tau_stars[1:])
            if _valid_positive_threshold(float(prev))
            and _valid_positive_threshold(float(curr))
        ]
        deltas = [float(d) for d in deltas if math.isfinite(float(d))]
        if not deltas:
            return None
        recent = sorted(deltas[-min(5, len(deltas)):])
        return float(recent[len(recent) // 2])

    def recent_measured_tau_star_upward_drift(self) -> Optional[float]:
        """Return recent positive tau-star motion, the over-selection risk side."""

        tau_stars = [
            float(t)
            for t in self.tau_star_rows
            if _valid_positive_threshold(float(t))
        ]
        if len(tau_stars) < 3:
            return None
        deltas = [
            math.log(float(curr)) - math.log(float(prev))
            for prev, curr in zip(tau_stars, tau_stars[1:])
            if _valid_positive_threshold(float(prev))
            and _valid_positive_threshold(float(curr))
        ]
        positive = [
            float(d)
            for d in deltas[-min(5, len(deltas)):]
            if math.isfinite(float(d)) and float(d) > 0.0
        ]
        if not positive:
            return None
        return float(max(positive))

    def _fit_recent_tau_star(self, *, step: int) -> Optional[_TauStarFit]:
        measured_fit = self._fit_measured_tau_star_rows(step=int(step))
        if measured_fit is not None:
            return measured_fit
        gain = self.recent_measured_alpha_gain()
        if gain is None or not math.isfinite(float(gain)) or float(gain) <= 0.0:
            return None
        target = max(float(self.config.target_ratio), 1.0e-12)
        obs = tuple(
            o
            for o in self._observations
            if (
                not o.exact
                and _valid_positive_threshold(float(o.tau))
                and math.isfinite(float(o.selected_ratio))
                and float(o.selected_ratio) > 0.0
            )
        )
        required = max(
            2,
            int(self.config.min_observations),
        )
        if len(obs) < required:
            return None
        max_window = min(len(obs), max(required, int(self.config.window_size)))
        best: Optional[_TauStarFit] = None
        best_score: Optional[Tuple[float, int]] = None
        for n_window in range(max_window, required - 1, -1):
            window = obs[-n_window:]
            xs = [float(o.step) for o in window]
            ys = [
                math.log(float(o.tau))
                + float(gain) * math.log(float(o.selected_ratio) / target)
                for o in window
            ]
            if any(not math.isfinite(y) for y in ys):
                continue
            x_mean = sum(xs) / float(len(xs))
            y_mean = sum(ys) / float(len(ys))
            denom = sum((x - x_mean) * (x - x_mean) for x in xs)
            if denom <= 0.0 or not math.isfinite(denom):
                slope = 0.0
            else:
                slope = (
                    sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
                    / denom
                )
            if not math.isfinite(slope):
                continue
            intercept = y_mean - slope * x_mean
            predicted_log_tau_star = intercept + slope * float(step)
            if not math.isfinite(predicted_log_tau_star):
                continue
            residual = max(
                abs(y - (intercept + slope * x))
                for x, y in zip(xs, ys)
            )
            if not math.isfinite(residual):
                continue
            fit = _TauStarFit(
                predicted_log_tau_star=float(predicted_log_tau_star),
                gain=float(gain),
                residual_log_tau_star=float(residual),
            )
            score = (float(residual) / float(gain), -int(n_window))
            if best_score is None or score < best_score:
                best = fit
                best_score = score
        return best

    def _fit_measured_tau_star_rows(self, *, step: int) -> Optional[_TauStarFit]:
        gain = self.recent_measured_alpha_gain()
        if gain is None or not math.isfinite(float(gain)) or float(gain) <= 0.0:
            return None
        rows = tuple(
            (int(s), float(t))
            for s, t in self.tau_star_step_rows
            if _valid_positive_threshold(float(t))
        )
        required = max(2, int(self.config.min_observations))
        if len(rows) < required:
            return None
        max_window = min(len(rows), max(required, int(self.config.window_size)))
        best: Optional[_TauStarFit] = None
        best_score: Optional[Tuple[float, int]] = None
        for n_window in range(max_window, required - 1, -1):
            window = rows[-n_window:]
            xs = [float(s) for s, _t in window]
            ys = [math.log(float(t)) for _s, t in window]
            if any(not math.isfinite(y) for y in ys):
                continue
            x_mean = sum(xs) / float(len(xs))
            y_mean = sum(ys) / float(len(ys))
            denom = sum((x - x_mean) * (x - x_mean) for x in xs)
            if denom <= 0.0 or not math.isfinite(denom):
                slope = 0.0
            else:
                slope = (
                    sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
                    / denom
                )
            if not math.isfinite(slope):
                continue
            intercept = y_mean - slope * x_mean
            predicted_log_tau_star = intercept + slope * float(step)
            if not math.isfinite(predicted_log_tau_star):
                continue
            residual = max(
                abs(y - (intercept + slope * x))
                for x, y in zip(xs, ys)
            )
            if not math.isfinite(residual):
                continue
            fit = _TauStarFit(
                predicted_log_tau_star=float(predicted_log_tau_star),
                gain=float(gain),
                residual_log_tau_star=float(residual),
            )
            score = (float(residual) / float(gain), -int(n_window))
            if best_score is None or score < best_score:
                best = fit
                best_score = score
        return best

    def certified_tau_from_recent_survival(
        self,
        *,
        step: int,
        cutoff_tolerance: float,
    ) -> Tuple[Optional[float], StableTauCertificate]:
        fit = self._fit_recent_tau_star(step=int(step))
        if fit is None:
            return None, StableTauCertificate(False, "survival_fit_unavailable")
        tau = math.exp(float(fit.predicted_log_tau_star))
        if not _valid_positive_threshold(float(tau)):
            return None, StableTauCertificate(False, "survival_fit_invalid_tau")
        cert = self.certify_predicted_tau(
            step=int(step),
            tau=float(tau),
            cutoff_tolerance=float(cutoff_tolerance),
        )
        if not cert.safe:
            return None, cert
        return float(tau), cert

    def certify_predicted_tau(
        self,
        *,
        step: int,
        tau: float,
        cutoff_tolerance: float,
    ) -> StableTauCertificate:
        """Certify that a predicted tau is safe enough to skip candidate count.

        The certificate uses only observed survival geometry.  For each recent
        final-count observation, estimate the tau-star coordinate

            log tau* = log tau + gain * log(selected_ratio / target)

        then fit a linear drift model over the active sliding window.  Candidate
        count can be skipped only when the predicted ratio for ``tau`` remains
        inside the strict K band even after the window's observed tau-star
        residual envelope is applied.
        """

        target = max(float(self.config.target_ratio), 1.0e-12)
        slack = max(float(cutoff_tolerance), float(self.config.retry_tolerance))
        if slack <= 0.0 or slack >= 1.0:
            return StableTauCertificate(False, "invalid_strict_band")
        if not _valid_positive_threshold(float(tau)):
            return StableTauCertificate(False, "invalid_predicted_tau")
        fit = self._fit_recent_tau_star(step=int(step))
        if fit is None:
            return StableTauCertificate(False, "missing_survival_gain")
        residual_log_ratio = (
            float(fit.residual_log_tau_star) / float(fit.gain)
        )
        predicted_log_ratio = (
            float(fit.predicted_log_tau_star) - math.log(float(tau))
        ) / float(fit.gain)
        if not math.isfinite(predicted_log_ratio):
            return StableTauCertificate(False, "certificate_invalid_ratio")
        lower_log = math.log(max(1.0 - slack, 1.0e-12))
        upper_log = math.log1p(slack)
        lo = float(predicted_log_ratio) - float(residual_log_ratio)
        hi = float(predicted_log_ratio) + float(residual_log_ratio)
        predicted_ratio = target * math.exp(float(predicted_log_ratio))
        if not (lower_log <= lo and hi <= upper_log):
            return StableTauCertificate(
                False,
                "certificate_ratio_risk",
                predicted_ratio=float(predicted_ratio),
                residual_log_ratio=float(residual_log_ratio),
            )
        return StableTauCertificate(
            True,
            "certified_stable_survival",
            predicted_ratio=float(predicted_ratio),
            residual_log_ratio=float(residual_log_ratio),
        )

    def tau_moves_towards_target_count(
        self,
        *,
        current_tau: float,
        candidate_tau: float,
        selected_ratio: float,
        cutoff_tolerance: float,
    ) -> bool:
        if (
            not _valid_positive_threshold(float(current_tau))
            or not _valid_positive_threshold(float(candidate_tau))
            or not math.isfinite(float(selected_ratio))
            or float(selected_ratio) <= 0.0
        ):
            return False
        target = max(float(self.config.target_ratio), 1.0e-12)
        tolerance = target * max(
            float(cutoff_tolerance),
            float(self.config.retry_tolerance),
        )
        ratio = float(selected_ratio)
        if abs(ratio - target) <= tolerance:
            return True
        log_step = math.log(float(candidate_tau)) - math.log(float(current_tau))
        if not math.isfinite(log_step):
            return False
        if ratio < target:
            return log_step < -1.0e-9
        return log_step > 1.0e-9

    def tau_from_recent_error_gain(
        self,
        *,
        current_tau: float,
        selected_ratio: float,
        cutoff_tolerance: float,
    ) -> Optional[float]:
        if (
            not _valid_positive_threshold(float(current_tau))
            or not math.isfinite(float(selected_ratio))
            or float(selected_ratio) <= 0.0
        ):
            return None
        gains = [
            float(g)
            for g in self.tau_error_gains
            if math.isfinite(float(g)) and float(g) > 0.0
        ]
        if not gains:
            return None
        gains.sort()
        gain = gains[len(gains) // 2]
        target = max(float(self.config.target_ratio), 1.0e-12)
        predicted = float(current_tau) * math.exp(
            gain * math.log(max(float(selected_ratio), 1.0e-12) / target)
        )
        if not _valid_positive_threshold(float(predicted)):
            return None
        if not self.tau_moves_towards_target_count(
            current_tau=float(current_tau),
            candidate_tau=float(predicted),
            selected_ratio=float(selected_ratio),
            cutoff_tolerance=float(cutoff_tolerance),
        ):
            return None
        return float(predicted)

    def candidate_taus(
        self,
        initial_tau: float,
        *,
        capacity: int,
    ) -> List[float]:
        cap = max(1, int(capacity))
        target = max(float(self.config.target_ratio), 1.0e-12)
        out: List[float] = []

        def add(value: float) -> None:
            _append_unique_threshold(out, value, capacity=cap)

        add(float(initial_tau))
        observations = _candidate_observation_window(self.observations)
        local_step = _median_log_tau_step(observations)
        probe_step = _max_optional(
            local_step,
            _latest_ratio_log_error(observations, target),
        )
        if observations:
            latest = observations[-1]
            tau_star_pred = self.predicted_tau_star()
            if tau_star_pred is not None:
                add(float(tau_star_pred))
                support = self.recent_measured_tau_star_abs_drift()
                if (
                    support is not None
                    and math.isfinite(float(support))
                    and float(support) > 0.0
                ):
                    support = min(max(float(support), 1.0e-4), 1.0)
                    add(float(tau_star_pred) * math.exp(-support))
                    add(float(tau_star_pred) * math.exp(support))
            gain = self.recent_measured_alpha_gain()
            if gain is not None:
                feedback_tau = _tau_from_ratio_feedback(
                    tau=float(latest.tau),
                    selected_ratio=float(latest.selected_ratio),
                    target_ratio=target,
                    gain=float(gain),
                )
                if feedback_tau is not None:
                    add(float(feedback_tau))
        if probe_step is not None and math.isfinite(float(probe_step)) and probe_step > 0.0:
            probe_step = min(max(float(probe_step), 1.0e-4), 1.0)
            for mul in (0.5, -0.5, 1.0, -1.0, 2.0, -2.0):
                add(float(initial_tau) * math.exp(float(mul) * probe_step))
        if observations:
            latest = observations[-1]
            if latest.selected_ratio > 0.0:
                add(
                    float(latest.tau)
                    * max(float(latest.selected_ratio), 1.0e-12)
                    / target
                )
            add(latest.tau)
            if local_step is not None:
                for mul in (-1.0, 1.0):
                    add(float(initial_tau) * math.exp(mul * float(local_step)))
        above = [o for o in observations if o.selected_ratio >= target]
        below = [o for o in observations if o.selected_ratio < target]
        if above:
            add(min(above, key=lambda o: abs(o.selected_ratio - target)).tau)
        if below:
            add(min(below, key=lambda o: abs(o.selected_ratio - target)).tau)

        step = local_step
        if step is None:
            ratios = [
                o.selected_ratio for o in observations[-4:]
                if o.selected_ratio > 0.0
            ]
            avg_ratio = sum(ratios) / float(len(ratios)) if ratios else target
            step = abs(math.log(max(avg_ratio, 1.0e-12) / target))
        if math.isfinite(step) and step > 0.0:
            step = min(max(step, 1.0e-4), 1.0)
            for mul in (1.0, -1.0, 2.0, -2.0):
                add(float(initial_tau) * math.exp(mul * step))
        return out[:cap]

    def _predict_tau(self, *, step: int) -> Optional[float]:
        obs = tuple(
            o for o in self._observations
            if o.tau > 0.0 and o.selected_ratio > 0.0
        )
        if not obs:
            return None
        last_tau = obs[-1].tau
        if not math.isfinite(last_tau) or last_tau <= 0.0:
            return None

        adaptive_obs = tuple(o for o in obs if not o.exact)
        control_source = adaptive_obs if len(adaptive_obs) >= 2 else obs
        control_obs = _recent_control_window(control_source)
        latest_ratio = obs[-1].selected_ratio
        avg_ratio = _linearly_weighted_mean([o.selected_ratio for o in control_obs])
        target = max(float(self.config.target_ratio), 1.0e-12)
        tolerance = target * float(self.config.tolerance)
        tau_stars = [
            float(t)
            for t in self.tau_star_rows
            if _valid_positive_threshold(float(t))
        ]
        stable_window = (
            abs(latest_ratio - target) <= tolerance
            and abs(avg_ratio - target) <= tolerance
        )
        if stable_window:
            if tau_stars:
                tau_star_pred = float(tau_stars[-1]) * math.exp(
                    self.recent_measured_tau_star_drift()
                )
                if _valid_positive_threshold(float(tau_star_pred)):
                    return float(tau_star_pred)
            certified_tau, _cert = self.certified_tau_from_recent_survival(
                step=int(step),
                cutoff_tolerance=max(
                    float(self.config.tolerance),
                    float(self.config.retry_tolerance),
                ),
            )
            if _valid_positive_threshold(float(certified_tau or 0.0)):
                return float(certified_tau)
            adjusted = self.tau_from_recent_error_gain(
                current_tau=float(last_tau),
                selected_ratio=float(latest_ratio),
                cutoff_tolerance=float(self.config.tolerance),
            )
            if _valid_positive_threshold(float(adjusted or 0.0)):
                return _clamp_tau_to_recent_motion(
                    predicted_tau=float(adjusted),
                    observations=control_obs,
                )
            return last_tau
        if tau_stars:
            tau_star_pred = float(tau_stars[-1]) * math.exp(
                self.recent_measured_tau_star_drift()
            )
            if _valid_positive_threshold(float(tau_star_pred)):
                predicted_from_alpha = self.tau_from_recent_error_gain(
                    current_tau=float(tau_star_pred),
                    selected_ratio=float(latest_ratio),
                    cutoff_tolerance=float(self.config.tolerance),
                )
                if _valid_positive_threshold(float(predicted_from_alpha or 0.0)):
                    return _clamp_tau_to_recent_motion(
                        predicted_tau=float(predicted_from_alpha),
                        observations=control_obs,
                    )

        predicted = _predict_tau_from_feedback_trend(
            control_obs,
            target,
            next_step=max(float(step), float(control_obs[-1].step + 1)),
        )
        if predicted is None and stable_window:
            predicted = last_tau
        if predicted is None:
            predicted = _interpolate_bracketed_log_tau(control_obs, target)
        if predicted is None:
            predicted = _scale_tau_by_observed_error(last_tau, avg_ratio, target)
        if predicted is None:
            return None
        if not math.isfinite(predicted) or predicted <= 0.0:
            return None
        return _clamp_tau_to_recent_motion(
            predicted_tau=predicted,
            observations=control_obs,
        )


def _interpolate_bracketed_log_tau(
    observations: tuple[CutoffObservation, ...],
    target_ratio: float,
) -> Optional[float]:
    pairs = sorted(
        (math.log(o.tau), math.log(o.selected_ratio))
        for o in observations
        if o.tau > 0.0 and o.selected_ratio > 0.0
    )
    target_log_ratio = math.log(max(float(target_ratio), 1.0e-12))
    best: Optional[float] = None
    best_width = float("inf")
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        if (y0 - target_log_ratio) * (y1 - target_log_ratio) > 0.0:
            continue
        if y0 == y1:
            x = 0.5 * (x0 + x1)
        else:
            t = (target_log_ratio - y0) / (y1 - y0)
            x = x0 + t * (x1 - x0)
        width = abs(x1 - x0)
        if width < best_width:
            best_width = width
            best = x
    if best is None or not math.isfinite(best):
        return None
    return math.exp(best)


def _scale_tau_by_observed_error(
    last_tau: float,
    avg_ratio: float,
    target_ratio: float,
) -> Optional[float]:
    if last_tau <= 0.0 or avg_ratio <= 0.0 or target_ratio <= 0.0:
        return None
    predicted = last_tau * (avg_ratio / target_ratio)
    if not math.isfinite(predicted) or predicted <= 0.0:
        return None
    return predicted


def _valid_positive_threshold(tau: float) -> bool:
    return bool(math.isfinite(float(tau)) and float(tau) > 0.0)


def _append_unique_threshold(
    out: List[float],
    value: float,
    *,
    capacity: int,
) -> None:
    if len(out) >= int(capacity):
        return
    tau = float(value)
    if not _valid_positive_threshold(tau):
        return
    if any(abs(math.log(tau) - math.log(old)) <= 1.0e-6 for old in out):
        return
    out.append(tau)


def _candidate_observation_window(
    observations: Tuple[CutoffObservation, ...],
) -> Tuple[CutoffObservation, ...]:
    rows = tuple(
        o for o in observations
        if o.tau > 0.0 and o.selected_ratio > 0.0
    )
    first_adaptive_step = min(
        (o.step for o in rows if not o.exact),
        default=None,
    )
    if first_adaptive_step is None:
        return rows
    return tuple(
        o for o in rows
        if (not o.exact) or int(o.step) > int(first_adaptive_step)
    )


def _median_log_tau_step(
    observations: Tuple[CutoffObservation, ...],
) -> Optional[float]:
    steps = [
        abs(math.log(curr.tau) - math.log(prev.tau))
        for prev, curr in zip(observations, observations[1:])
        if prev.tau > 0.0 and curr.tau > 0.0 and prev.tau != curr.tau
    ]
    steps = [
        float(step)
        for step in steps
        if math.isfinite(float(step)) and float(step) > 0.0
    ]
    if not steps:
        return None
    steps.sort()
    return float(steps[len(steps) // 2])


def _latest_ratio_log_error(
    observations: Tuple[CutoffObservation, ...],
    target_ratio: float,
) -> Optional[float]:
    if not observations:
        return None
    latest_ratio = max(float(observations[-1].selected_ratio), 1.0e-12)
    target = max(float(target_ratio), 1.0e-12)
    step = abs(math.log(latest_ratio / target))
    if not math.isfinite(step) or step <= 0.0:
        return None
    return float(step)


def _max_optional(*values: Optional[float]) -> Optional[float]:
    valid = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(float(value))
        and float(value) > 0.0
    ]
    if not valid:
        return None
    return max(valid)


def _tau_from_ratio_feedback(
    *,
    tau: float,
    selected_ratio: float,
    target_ratio: float,
    gain: float,
) -> Optional[float]:
    if (
        not _valid_positive_threshold(float(tau))
        or not math.isfinite(float(selected_ratio))
        or float(selected_ratio) <= 0.0
        or not math.isfinite(float(gain))
        or float(gain) <= 0.0
    ):
        return None
    target = max(float(target_ratio), 1.0e-12)
    predicted = float(tau) * math.exp(
        float(gain) * math.log(max(float(selected_ratio), 1.0e-12) / target)
    )
    if not _valid_positive_threshold(float(predicted)):
        return None
    return float(predicted)


def estimate_measured_alpha_tau(
    ratio_rows: List[Tuple[float, float]],
    *,
    target_ratio: float,
) -> Tuple[Optional[float], Optional[float], str]:
    target = max(float(target_ratio), 1.0e-12)
    target_log = math.log(target)
    points = sorted(
        (
            math.log(float(tau)),
            math.log(float(ratio)),
            float(tau),
            float(ratio),
        )
        for tau, ratio in ratio_rows
        if _valid_positive_threshold(float(tau))
        and math.isfinite(float(ratio))
        and 0.0 < float(ratio) <= 1.0
    )
    if len(points) < 2:
        return None, None, "measured_alpha_insufficient"

    alpha_samples: List[float] = []
    best_bracket: Optional[Tuple[float, float, Tuple[float, float]]] = None
    for i, (x0, y0, _tau0, ratio0) in enumerate(points):
        for x1, y1, _tau1, ratio1 in points[i + 1:]:
            dx = float(x1 - x0)
            if abs(dx) <= 1.0e-12:
                continue
            slope = float(y1 - y0) / dx
            if not math.isfinite(slope) or slope >= 0.0:
                continue
            alpha = -slope
            if not math.isfinite(alpha) or alpha <= 0.0:
                continue
            alpha_samples.append(float(alpha))
            raw_brackets_target = (
                (float(ratio0) - target) * (float(ratio1) - target) <= 0.0
            )
            if not raw_brackets_target:
                continue
            x_star = float(x0) + (target_log - float(y0)) / slope
            lo, hi = sorted((float(x0), float(x1)))
            if not math.isfinite(x_star) or not (lo <= x_star <= hi):
                continue
            score = (
                max(abs(float(ratio0) - target), abs(float(ratio1) - target))
                / target,
                hi - lo,
            )
            if best_bracket is None or score < best_bracket[2]:
                best_bracket = (float(x_star), float(alpha), score)

    if best_bracket is not None:
        tau = math.exp(float(best_bracket[0]))
        alpha = float(best_bracket[1])
        if _valid_positive_threshold(float(tau)):
            return float(tau), float(alpha), "measured_alpha_bracket"
    if not alpha_samples:
        return None, None, "measured_alpha_no_negative_slope"

    alpha_samples.sort()
    alpha = float(alpha_samples[len(alpha_samples) // 2])
    if not math.isfinite(alpha) or alpha <= 0.0:
        return None, None, "measured_alpha_invalid"

    x_min = min(x for x, _y, _tau, _ratio in points)
    x_max = max(x for x, _y, _tau, _ratio in points)
    span = max(float(x_max - x_min), 1.0e-12)
    x_star_samples: List[Tuple[float, float]] = []
    for x, y, _tau, ratio in points:
        x_star = float(x) + (float(y) - target_log) / alpha
        if not math.isfinite(x_star):
            continue
        x_star = min(max(x_star, float(x_min) - span), float(x_max) + span)
        ratio_error = abs(math.log(max(float(ratio), 1.0e-12) / target))
        x_star_samples.append((float(ratio_error), float(x_star)))
    if not x_star_samples:
        return None, None, "measured_alpha_no_tau_star"
    x_star_samples.sort(key=lambda item: item[0])
    nearest = sorted(x for _err, x in x_star_samples[: min(3, len(x_star_samples))])
    x_star = nearest[len(nearest) // 2]
    tau = math.exp(float(x_star))
    if not _valid_positive_threshold(float(tau)):
        return None, None, "measured_alpha_invalid_tau"
    return float(tau), float(alpha), "measured_alpha_extrapolated"


def _clamp_tau_to_recent_motion(
    *,
    predicted_tau: float,
    observations: tuple[CutoffObservation, ...],
) -> Optional[float]:
    if predicted_tau <= 0.0 or not math.isfinite(float(predicted_tau)):
        return None
    obs = tuple(o for o in observations if o.tau > 0.0)
    if len(obs) < 2:
        return float(predicted_tau)
    last_tau = float(obs[-1].tau)
    if last_tau <= 0.0 or not math.isfinite(last_tau):
        return float(predicted_tau)
    observed_steps = [
        abs(math.log(float(curr.tau)) - math.log(float(prev.tau)))
        for prev, curr in zip(obs, obs[1:])
        if prev.tau > 0.0 and curr.tau > 0.0
    ]
    observed_steps = [
        step for step in observed_steps
        if math.isfinite(float(step)) and float(step) > 0.0
    ]
    if not observed_steps:
        return float(predicted_tau)
    max_observed_step = max(observed_steps)
    predicted_log_step = (
        math.log(float(predicted_tau)) - math.log(last_tau)
    )
    if abs(predicted_log_step) <= max_observed_step:
        return float(predicted_tau)
    clamped_log_tau = math.log(last_tau) + math.copysign(
        max_observed_step,
        predicted_log_step,
    )
    clamped = math.exp(clamped_log_tau)
    if not math.isfinite(clamped) or clamped <= 0.0:
        return float(predicted_tau)
    return float(clamped)


def _predict_tau_from_feedback_trend(
    observations: tuple[CutoffObservation, ...],
    target_ratio: float,
    next_step: float,
) -> Optional[float]:
    target = max(float(target_ratio), 1.0e-12)
    feedback = [
        (float(o.step), o.tau * (o.selected_ratio / target))
        for o in observations
        if o.tau > 0.0 and o.selected_ratio > 0.0
    ]
    if not feedback:
        return None
    if len(feedback) == 1:
        return _linearly_weighted_mean([tau for _, tau in feedback])
    if len(feedback) == 2:
        (x0, tau0), (x1, tau1) = feedback
        if x1 == x0 or tau0 <= 0.0 or tau1 <= 0.0:
            return _linearly_weighted_mean([tau for _, tau in feedback])
        slope = (math.log(tau1) - math.log(tau0)) / (x1 - x0)
        predicted_log_tau = math.log(tau1) + slope * (float(next_step) - x1)
        if not math.isfinite(predicted_log_tau):
            return _linearly_weighted_mean([tau for _, tau in feedback])
        predicted = math.exp(predicted_log_tau)
        if not math.isfinite(predicted) or predicted <= 0.0:
            return _linearly_weighted_mean([tau for _, tau in feedback])
        return predicted

    weights = [float(i) for i in range(1, len(feedback) + 1)]
    xs = [x for x, _ in feedback]
    ys = [math.log(tau) for _, tau in feedback if tau > 0.0]
    if len(ys) != len(feedback):
        return None
    weight_total = sum(weights)
    x_mean = sum(w * x for w, x in zip(weights, xs)) / weight_total
    y_mean = sum(w * y for w, y in zip(weights, ys)) / weight_total
    denom = 0.0
    numer = 0.0
    for weight, x, y in zip(weights, xs, ys):
        dx = x - x_mean
        denom += weight * dx * dx
        numer += weight * dx * (y - y_mean)
    if denom <= 0.0:
        return _linearly_weighted_mean([tau for _, tau in feedback])
    slope = numer / denom
    intercept = y_mean - slope * x_mean
    predicted_log_tau = intercept + slope * float(next_step)
    if not math.isfinite(predicted_log_tau):
        return _linearly_weighted_mean([tau for _, tau in feedback])
    predicted = math.exp(predicted_log_tau)
    if not math.isfinite(predicted) or predicted <= 0.0:
        return _linearly_weighted_mean([tau for _, tau in feedback])
    return predicted


def _recent_control_window(
    observations: tuple[CutoffObservation, ...],
) -> tuple[CutoffObservation, ...]:
    n = len(observations)
    if n <= 1:
        return observations
    tail = max(2, (n + 3) // 4)
    return observations[-tail:]


def _linearly_weighted_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    weighted_sum = 0.0
    weight_total = 0.0
    for i, value in enumerate(values, start=1):
        weighted_sum += float(i) * float(value)
        weight_total += float(i)
    return weighted_sum / max(weight_total, 1.0)


class ApproxProbeRejectWarning:
    """Delayed warning for unexpected probe rejection without stream sync."""

    def __init__(self, *, ring_size: int = 8) -> None:
        self._ring_size = max(2, int(ring_size))
        self._decision_code = torch.empty(
            self._ring_size, dtype=torch.int32, pin_memory=True,
        )
        self._probe_reject = torch.empty(
            self._ring_size, dtype=torch.int32, pin_memory=True,
        )
        self._diagnostics = torch.empty(
            (self._ring_size, 8), dtype=torch.float32, pin_memory=True,
        )
        self._events = [torch.cuda.Event() for _ in range(self._ring_size)]
        self._steps = [-1 for _ in range(self._ring_size)]
        self._pending_slots: List[int] = []
        self._warned_steps: set[int] = set()

    def record(
        self,
        *,
        step: int,
        stream: torch.cuda.Stream,
        decision_code: torch.Tensor,
        probe_reject: torch.Tensor,
        diagnostics: torch.Tensor,
    ) -> None:
        slot = int(step) % self._ring_size
        if slot in self._pending_slots and not self._events[slot].query():
            return
        with torch.cuda.stream(stream):
            self._decision_code[slot].copy_(
                decision_code.reshape(()),
                non_blocking=True,
            )
            self._probe_reject[slot].copy_(
                probe_reject.reshape(()),
                non_blocking=True,
            )
            self._diagnostics[slot].copy_(
                diagnostics.reshape(-1)[:8],
                non_blocking=True,
            )
            self._events[slot].record(stream)
        self._steps[slot] = int(step)
        if slot not in self._pending_slots:
            self._pending_slots.append(slot)

    def emit_ready(self) -> None:
        still_pending: List[int] = []
        for slot in self._pending_slots:
            if not self._events[slot].query():
                still_pending.append(slot)
                continue
            step = int(self._steps[slot])
            code = int(self._decision_code[slot].item())
            reject = int(self._probe_reject[slot].item())
            if (code == 0 or reject != 0) and step not in self._warned_steps:
                diag = self._diagnostics[slot]
                print(
                    "[proposal] WARNING approx cutoff probe rejected "
                    f"@iter{step}: decision_code={code} "
                    f"probe_reject={reject} approx_ratio={float(diag[0].item()):.8g} "
                    f"se={float(diag[1].item()):.8g} "
                    f"bound={float(diag[2].item()):.8g} "
                    f"tau={float(diag[3].item()):.8g} "
                    f"sampled_blocks={int(float(diag[6].item()))}",
                    flush=True,
                )
                self._warned_steps.add(step)
        self._pending_slots = still_pending


__all__ = [
    "ApproxProbeRejectWarning",
    "AdaptiveCutoffConfig",
    "AdaptiveCutoffController",
    "CutoffDecision",
    "CutoffObservation",
    "estimate_measured_alpha_tau",
]
