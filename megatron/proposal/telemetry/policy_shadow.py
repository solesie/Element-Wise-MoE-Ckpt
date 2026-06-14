"""Experiment-only independent checkpoint-policy shadow simulation.

Enable with ``PROPOSAL_POLICY_SHADOW_LOG=/path/to/policy_shadow.jsonl``.

This telemetry answers a different question from one-step feature probes:
each policy owns its own ``theta_cp`` and, when selected, resets only its own
checkpoint snapshot.  It never reads or mutates production ``dg`` or write
state.  The only shared live inputs are p/m/v/H and Adam scalars.
"""

from __future__ import annotations

import json
import math
import os
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class _SampleSource:
    group_index: int
    local_offsets: torch.Tensor
    sample_indices: np.ndarray


@dataclass
class _PolicyState:
    name: str
    theta_cp: np.ndarray
    group_debt: np.ndarray
    last_written: np.ndarray
    last_written_step: np.ndarray
    age_code: np.ndarray
    cum_delta: np.ndarray
    cum_abs_delta: np.ndarray
    cum_delta_sq: np.ndarray
    max_abs_delta: np.ndarray
    prev_delta: np.ndarray
    sign_flip_count: np.ndarray
    lambda_floor: float
    fitted_age_power: float = 1.0
    fitted_delta_power: float = 0.0
    use_fitted_score: float = 0.0


class PolicyShadowTelemetry:
    """CPU-side sampled independent-policy simulator."""

    def __init__(
        self,
        *,
        log_path: Optional[str],
        sample_groups: int,
        every: int,
        seed: int,
        policies: Tuple[str, ...],
    ) -> None:
        self.log_path = Path(log_path).expanduser() if log_path else None
        self.sample_groups = max(1, int(sample_groups))
        self.every = max(1, int(every))
        self.seed = int(seed)
        self.policies = tuple(policies)
        self._initialized = False
        self._sources: List[_SampleSource] = []
        self._sample_group_ids: Optional[np.ndarray] = None
        self._sample_element_ids: Optional[np.ndarray] = None
        self._states: Dict[str, _PolicyState] = {}
        self._group_size = 1
        self._prev_delta_sq: Optional[np.ndarray] = None
        self._prev_h_delta_score: Optional[np.ndarray] = None
        self._delta_ema: Dict[int, np.ndarray] = {}
        self._group_delta_ema: Dict[int, np.ndarray] = {}
        self._theta_history: List[np.ndarray] = []
        self._passive_horizons: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
        self._beta1 = float(os.environ.get("PROPOSAL_POLICY_SHADOW_BETA1", "0.9") or "0.9")

    @classmethod
    def from_env(cls) -> "PolicyShadowTelemetry":
        return cls(
            log_path=os.environ.get("PROPOSAL_POLICY_SHADOW_LOG", ""),
            sample_groups=int(os.environ.get("PROPOSAL_POLICY_SHADOW_SAMPLE_GROUPS", "8192") or "8192"),
            every=int(os.environ.get("PROPOSAL_POLICY_SHADOW_EVERY", "1") or "1"),
            seed=int(os.environ.get("PROPOSAL_POLICY_SHADOW_SEED", "2024") or "2024"),
            policies=cls._parse_csv_env(
                "PROPOSAL_POLICY_SHADOW_POLICIES",
                "oracle,rr,h_delta,delta,h_age,h_age_q2,h_age_q3,h_age_q4,h_age_q8,h_age_cap8,h_age_cap16,h_logage,h_sqrtage,h_oldest1,h_oldest2,h_oldest4,h_rank2,h_groupmaxage,h_groupmeanage,h_groupmaxage_rank2,h_groupmeanage_rank2,h_groupmaxage_oldest1,h_groupmaxage_oldest2,h_groupmaxage_oldest4,h_stale8,h_stale16,mix_age_spike,mix_q4_spike,dg19,dg_age19",
            ),
        )

    @staticmethod
    def _parse_csv_env(name: str, default: str) -> Tuple[str, ...]:
        raw = os.environ.get(name, default) or default
        return tuple(item.strip() for item in raw.split(",") if item.strip())

    @property
    def enabled(self) -> bool:
        return self.log_path is not None

    def allocate_state(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record(
        self,
        *,
        step: int,
        topk_ratio: float,
        groups: Iterable[Dict[str, Any]],
        h_lut_values: torch.Tensor,
        h_lut_log_v_min: torch.Tensor,
        h_lut_inv_log_v_width: torch.Tensor,
        group_size: int,
        lr_over_bc1: float,
        sqrt_bc2_inv: float,
        eps: float,
        stream: Optional[torch.cuda.Stream],
    ) -> None:
        if not self.enabled or int(step) % self.every != 0:
            return
        groups_list = list(groups)
        if not self._initialized:
            self._initialize(groups_list=groups_list, group_size=int(group_size))
        if not self._sources:
            return

        p, m, v = self._gather_sample(groups_list=groups_list)
        h = self._lookup_h(
            v=v,
            h_lut_values=h_lut_values,
            h_lut_log_v_min=h_lut_log_v_min,
            h_lut_inv_log_v_width=h_lut_inv_log_v_width,
        )
        denom = np.sqrt(np.maximum(v, 0.0), dtype=np.float32) * float(sqrt_bc2_inv) + float(eps)
        delta = float(lr_over_bc1) * m / denom
        delta_sq = delta * delta
        h_delta_score = h * delta_sq
        n = int(p.shape[0])
        k = max(1, min(n, int(math.ceil(float(topk_ratio) * float(n)))))
        requested_ema_spans = self._requested_delta_ema_spans()
        self._update_delta_ema(delta_sq=delta_sq, spans=requested_ema_spans)
        diagnostics = self._delta_diagnostics(delta_sq=delta_sq, h_delta_score=h_delta_score, k=k)
        passive_diagnostics = self._passive_horizon_diagnostics(
            p=p,
            h=h,
            delta_sq=delta_sq,
            k=k,
        )
        residual_fit = self._residual_fit_diagnostics(
            step=int(step),
            p=p,
            h=h,
            v=v,
            delta_sq=delta_sq,
            k=k,
        )

        rows = []
        oracle_post_error = None
        for name, state in self._states.items():
            self._advance_policy_trace(state=state, delta=delta)
            truth_score = h * ((p - state.theta_cp) ** 2)
            pre_error = float(truth_score.sum())
            ideal = self._topk_mask(truth_score, k)
            ideal_mass = float(truth_score[ideal].sum())
            score = self._score_for_policy(
                name=name,
                state=state,
                step=int(step),
                p=p,
                m=m,
                v=v,
                h=h,
                delta_sq=delta_sq,
                h_delta_score=h_delta_score,
                n=n,
                k=k,
                h_lut_bins=int(h_lut_values.numel()),
                log_v_min=float(h_lut_log_v_min.detach().cpu().item()),
                inv_log_v_width=float(h_lut_inv_log_v_width.detach().cpu().item()),
            )
            selected = self._topk_mask(score, k)
            selected_mass = float(truth_score[selected].sum())
            intersect = selected & ideal
            missed = ideal & (~selected)
            missed_mass = float(truth_score[missed].sum())
            selected_delta_mean = float(delta_sq[selected].mean()) if selected.any() else 0.0
            ideal_delta_mean = float(delta_sq[ideal].mean()) if ideal.any() else 0.0
            missed_delta_mean = float(delta_sq[missed].mean()) if missed.any() else 0.0
            ideal_h_delta = float(h_delta_score[ideal].sum())
            missed_h_delta = float(h_delta_score[missed].sum())
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            v_rank = self._rank01(np.log(np.maximum(v, np.float32(1.0e-30))))
            over64 = age > 64.0
            over128 = age >= 128.0
            extra = selected & (~ideal)
            velocity_diag = self._velocity_proxy_diagnostics(
                residual=(p - state.theta_cp),
                delta=delta,
                age=age,
                state=state,
                ideal=ideal,
                selected=selected,
                missed=missed,
                extra=extra,
            )
            age_bin_diag = self._age_bin_trace_diagnostics(
                truth_score=truth_score,
                residual=(p - state.theta_cp),
                delta=delta,
                age=age,
                state=state,
                ideal=ideal,
                selected=selected,
                missed=missed,
                extra=extra,
            )
            age_code_diag = self._age_code_diagnostics(
                state=state,
                step=int(step),
                selected=selected,
            )

            self._update_policy(
                state=state,
                step=int(step),
                selected=selected,
                p=p,
                delta_sq=delta_sq,
            )
            post_truth_score = h * ((p - state.theta_cp) ** 2)
            post_error = float(post_truth_score.sum())
            if name == "oracle":
                oracle_post_error = post_error
            rows.append({
                "policy": name,
                "pre_error": pre_error,
                "post_error": post_error,
                "selected_mass": selected_mass,
                "ideal_mass": ideal_mass,
                "immediate_removal_efficiency": self._safe_div(selected_mass, ideal_mass),
                "recall_to_own_oracle": self._safe_div(float(intersect.sum()), float(ideal.sum())),
                "missed_dormant_mass": missed_mass,
                "missed_dormant_mass_ratio": self._safe_div(missed_mass, ideal_mass),
                "missed_current_h_delta_ratio": self._safe_div(missed_h_delta, ideal_h_delta),
                "selected_delta_sq_mean": selected_delta_mean,
                "ideal_delta_sq_mean": ideal_delta_mean,
                "missed_delta_sq_mean": missed_delta_mean,
                "missed_delta_sq_mean_over_ideal": self._safe_div(missed_delta_mean, ideal_delta_mean),
                **self._mask_age_v_stats(
                    age=age,
                    v_rank=v_rank,
                    truth_score=truth_score,
                    masks={
                        "ideal": ideal,
                        "selected": selected,
                        "missed": missed,
                        "extra": extra,
                    },
                ),
                "age_over64_ratio": float(over64.mean()),
                "selected_age_over64_ratio": float(over64[selected].mean()) if selected.any() else 0.0,
                "age_over128_ratio": float(over128.mean()),
                "selected_age_over128_ratio": float(over128[selected].mean()) if selected.any() else 0.0,
                "age_p95": float(np.percentile(age, 95.0)),
                "fitted_age_power": float(state.fitted_age_power),
                "fitted_delta_power": float(state.fitted_delta_power),
                **velocity_diag,
                **age_bin_diag,
                **age_code_diag,
            })
            if name.startswith("h_fit"):
                self._update_fitted_exponents(
                    state=state,
                    step=int(step),
                    p=p,
                    h=h,
                    delta_sq=delta_sq,
                    k=k,
                )

        if oracle_post_error is not None:
            for row in rows:
                row["post_error_over_oracle"] = self._safe_div(row["post_error"], oracle_post_error)

        self._append_record({
            "step": int(step),
            "mode": "policy_shadow",
            "sample_groups": int(n // max(1, int(group_size))),
            "sample_elements": int(n),
            "sample_k": int(k),
            "topk_ratio": float(topk_ratio),
            "diagnostics": diagnostics,
            "passive_horizon": passive_diagnostics,
            "residual_fit": residual_fit,
            "policies": rows,
        })
        self._prev_delta_sq = delta_sq.copy()
        self._prev_h_delta_score = h_delta_score.copy()
        self._theta_history.append(p.copy())
        max_horizon = max(self._passive_horizons)
        if len(self._theta_history) > max_horizon:
            del self._theta_history[: len(self._theta_history) - max_horizon]

    def _initialize(self, *, groups_list: List[Dict[str, Any]], group_size: int) -> None:
        ranges: List[Tuple[int, int, int]] = []
        total_groups = 0
        for idx, grp in enumerate(groups_list):
            if grp.get("p_flat") is None or grp.get("m_flat") is None or grp.get("v_flat") is None:
                continue
            group_count = int(grp.get("total_aligned", 0)) // int(group_size)
            if group_count <= 0:
                continue
            start = int(grp["master_off"])
            ranges.append((idx, start, start + group_count))
            total_groups = max(total_groups, start + group_count)
        if total_groups <= 0:
            self._initialized = True
            return
        n_sample_groups = min(int(self.sample_groups), int(total_groups))
        rng = np.random.default_rng(self.seed)
        sampled_groups = np.sort(
            rng.choice(total_groups, size=n_sample_groups, replace=False).astype(np.int64)
        )
        sources: List[_SampleSource] = []
        sample_group_ids: List[int] = []
        sample_element_ids: List[int] = []
        sample_index_base = 0
        sample_group_base = 0
        for group_idx, start, end in ranges:
            lo = int(np.searchsorted(sampled_groups, start, side="left"))
            hi = int(np.searchsorted(sampled_groups, end, side="left"))
            if hi <= lo:
                continue
            global_groups = sampled_groups[lo:hi]
            local_groups = global_groups - int(start)
            local_offsets_np = (
                local_groups[:, None] * int(group_size)
                + np.arange(int(group_size), dtype=np.int64)[None, :]
            ).reshape(-1)
            global_element_ids_np = (
                global_groups[:, None] * int(group_size)
                + np.arange(int(group_size), dtype=np.int64)[None, :]
            ).reshape(-1)
            sample_indices = np.arange(
                sample_index_base,
                sample_index_base + local_offsets_np.size,
                dtype=np.int64,
            )
            sample_index_base += int(local_offsets_np.size)
            device = groups_list[group_idx]["p_flat"].device
            local_offsets = torch.as_tensor(local_offsets_np, dtype=torch.long, device=device)
            sources.append(_SampleSource(
                group_index=int(group_idx),
                local_offsets=local_offsets,
                sample_indices=sample_indices,
            ))
            for local_sample_group_id in range(sample_group_base, sample_group_base + len(global_groups)):
                sample_group_ids.extend([int(local_sample_group_id)] * int(group_size))
            sample_element_ids.extend(int(x) for x in global_element_ids_np.tolist())
            sample_group_base += int(len(global_groups))
        self._group_size = int(group_size)
        self._sources = sources
        self._sample_group_ids = np.asarray(sample_group_ids, dtype=np.int64)
        self._sample_element_ids = np.asarray(sample_element_ids, dtype=np.uint64)
        self._initialized = True

    def _init_states(self, *, p: np.ndarray) -> None:
        if self._states:
            return
        n = int(p.shape[0])
        n_groups = max(1, n // max(1, self._group_size))
        for name in self.policies:
            lambda_floor = 0.0
            if name.startswith("dg_age"):
                raw = name[6:] or "1"
                lambda_floor = float(raw.replace("_", "."))
            elif name.startswith("dg"):
                raw = name[2:] or "1"
                lambda_floor = float(raw.replace("_", "."))
            self._states[name] = _PolicyState(
                name=name,
                theta_cp=p.copy(),
                group_debt=np.zeros(n_groups, dtype=np.float32),
                last_written=np.zeros(n, dtype=bool),
                last_written_step=np.zeros(n, dtype=np.int32),
                age_code=np.zeros(n, dtype=np.uint8),
                cum_delta=np.zeros(n, dtype=np.float32),
                cum_abs_delta=np.zeros(n, dtype=np.float32),
                cum_delta_sq=np.zeros(n, dtype=np.float32),
                max_abs_delta=np.zeros(n, dtype=np.float32),
                prev_delta=np.zeros(n, dtype=np.float32),
                sign_flip_count=np.zeros(n, dtype=np.uint16),
                lambda_floor=float(lambda_floor),
            )

    def _gather_sample(self, *, groups_list: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = sum(int(src.sample_indices.size) for src in self._sources)
        p = np.empty(n, dtype=np.float32)
        m = np.empty(n, dtype=np.float32)
        v = np.empty(n, dtype=np.float32)
        for src in self._sources:
            grp = groups_list[src.group_index]
            idx = src.sample_indices
            offsets = src.local_offsets
            p[idx] = torch.index_select(grp["p_flat"], 0, offsets).detach().float().cpu().numpy()
            m[idx] = torch.index_select(grp["m_flat"], 0, offsets).detach().float().cpu().numpy()
            v[idx] = torch.index_select(grp["v_flat"], 0, offsets).detach().float().cpu().numpy()
        self._init_states(p=p)
        return p, m, v

    def _lookup_h(
        self,
        *,
        v: np.ndarray,
        h_lut_values: torch.Tensor,
        h_lut_log_v_min: torch.Tensor,
        h_lut_inv_log_v_width: torch.Tensor,
    ) -> np.ndarray:
        lut = h_lut_values.detach().float().cpu().numpy()
        log_v_min = float(h_lut_log_v_min.detach().cpu().item())
        inv_width = float(h_lut_inv_log_v_width.detach().cpu().item())
        pos = (np.log(np.maximum(v, np.float32(1.0e-30))) - log_v_min) * inv_width
        pos = np.clip(pos, 0.0, float(len(lut) - 1))
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, len(lut) - 1)
        frac = (pos - lo.astype(np.float32)).astype(np.float32)
        return (lut[lo] + frac * (lut[hi] - lut[lo])).astype(np.float32)

    def _score_for_policy(
        self,
        *,
        name: str,
        state: _PolicyState,
        step: int,
        p: np.ndarray,
        m: np.ndarray,
        v: np.ndarray,
        h: np.ndarray,
        delta_sq: np.ndarray,
        h_delta_score: np.ndarray,
        n: int,
        k: int,
        h_lut_bins: int,
        log_v_min: float,
        inv_log_v_width: float,
    ) -> np.ndarray:
        if name == "oracle":
            return h * ((p - state.theta_cp) ** 2)
        if name == "rr":
            start = ((int(step) - 1) * int(k)) % max(1, int(n))
            dist = (np.arange(n, dtype=np.int64) - int(start)) % max(1, int(n))
            return -dist.astype(np.float32)
        if name == "h_delta":
            return h_delta_score
        if name == "delta":
            return delta_sq
        if name == "h_delta_skip1":
            return np.where(state.last_written, 0.0, h_delta_score)
        if name == "delta_skip1":
            return np.where(state.last_written, 0.0, delta_sq)
        if name == "h_delta_age":
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return h_delta_score * age
        if name == "delta_age":
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return delta_sq * age
        if name.startswith("h_agepow") and "_deltapow" in name:
            left, right = name.split("_deltapow", 1)
            age_power = float((left[len("h_agepow"):] or "1").replace("_", "."))
            delta_power = float((right or "1").replace("_", "."))
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return h * np.power(age, np.float32(age_power)) * np.power(delta_sq + np.float32(1.0e-30), np.float32(delta_power))
        if (
            name.startswith("h_agecode2b")
            or name.startswith("h_agedet2b")
            or name.startswith("h_agehard2b")
        ) and "_pow" in name:
            # Actual 2-bit state path.  Unlike h_ageq2b*, this does not read
            # exact age for scoring.  The code is updated only from actual
            # writes plus stateless hash promotions.  h_agecode2b uses
            # memoryless Morris-style promotion; h_agedet2b uses bounded
            # deterministic phase promotion with the same target dwell.
            force = name.endswith("_force")
            parse_name = name[:-len("_force")] if force else name
            v_mode = ""
            for suffix in (
                "_vbinmeanvelold",
                "_vbinmedianvelold",
                "_vbinp75velold",
                "_vbinp90velold",
                "_vbinmeanp90velold",
                "_vbinp75p90velold",
                "_vbinmeanvel11",
                "_vbinmedianvel11",
                "_vbinp75vel11",
                "_vbinp90vel11",
                "_vbinmeanaddold",
                "_vbinmedianaddold",
                "_vbinp75addold",
                "_vbinp90addold",
                "_vbinmeanadd11",
                "_vbinmedianadd11",
                "_vbinp75add11",
                "_vbinp90add11",
                "_vbinmeanold",
                "_vbinp75old",
                "_vbinp90old",
                "_vbinmeanp90old",
                "_vbinp75p90old",
                "_globalp75p90old",
                "_vbinshiftp75p90old",
                "_vbinrevp75p90old",
                "_vbinp75all",
                "_vbinp90all",
                "_lutfloorp75p90old",
                "_vrankold",
                "_vnormold",
            ):
                if parse_name.endswith(suffix):
                    v_mode = suffix[1:]
                    parse_name = parse_name[:-len(suffix)]
                    break
            left, rest = parse_name.split("_pow", 1)
            if left.startswith("h_agecode2b"):
                base = max(1.25, float(left[len("h_agecode2b"):] or "4"))
            elif left.startswith("h_agedet2b"):
                base = max(1.25, float(left[len("h_agedet2b"):] or "4"))
            else:
                base = max(1.25, float(left[len("h_agehard2b"):] or "4"))
            age_power_text, delta_text = rest.split("_deltapow", 1)
            age_power = float((age_power_text or "1").replace("_", "."))
            delta_power = float((delta_text or "0").replace("_", "."))
            quant_age = np.power(np.float32(base), state.age_code.astype(np.float32))
            scale_source = delta_sq + np.float32(1.0e-30)
            old_code = state.age_code >= np.uint8(2)
            if v_mode in {
                "vbinmeanvelold",
                "vbinmedianvelold",
                "vbinp75velold",
                "vbinp90velold",
                "vbinmeanp90velold",
                "vbinp75p90velold",
                "vbinmeanvel11",
                "vbinmedianvel11",
                "vbinp75vel11",
                "vbinp90vel11",
                "vbinmeanaddold",
                "vbinmedianaddold",
                "vbinp75addold",
                "vbinp90addold",
                "vbinmeanadd11",
                "vbinmedianadd11",
                "vbinp75add11",
                "vbinp90add11",
            }:
                young = state.age_code <= np.uint8(1)
                stat = "mean"
                if "median" in v_mode:
                    stat = "median"
                elif "p75" in v_mode:
                    stat = "p75"
                elif "p90" in v_mode:
                    stat = "p90"
                if "meanp90" in v_mode or "p75p90" in v_mode:
                    low_stat = "p75" if "p75p90" in v_mode else "mean"
                    low_activity = self._delta_stat_by_v_rank(
                        v=v,
                        delta_sq=delta_sq,
                        stat=low_stat,
                        mask=young,
                    )
                    high_activity = self._delta_stat_by_v_rank(
                        v=v,
                        delta_sq=delta_sq,
                        stat="p90",
                        mask=young,
                    )
                    activity = np.where(state.age_code >= np.uint8(3), high_activity, low_activity)
                else:
                    activity = self._delta_stat_by_v_rank(
                        v=v,
                        delta_sq=delta_sq,
                        stat=stat,
                        mask=young,
                    )
                corrected = old_code
                if v_mode.endswith("11"):
                    corrected = state.age_code >= np.uint8(3)
                recent = np.minimum(
                    quant_age,
                    np.float32(1.0 / max(1.0e-6, 1.0 - float(self._beta1))),
                )
                old = np.maximum(quant_age - recent, np.float32(0.0))
                if "add" in v_mode:
                    candidate = h * (
                        old * old * activity
                        + recent * recent * (delta_sq + np.float32(1.0e-30))
                    )
                else:
                    candidate = h * np.square(
                        old * np.sqrt(activity + np.float32(1.0e-30))
                        + recent * np.sqrt(delta_sq + np.float32(1.0e-30))
                    )
                base_score = (
                    h
                    * np.power(quant_age, np.float32(age_power))
                    * np.power(delta_sq + np.float32(1.0e-30), np.float32(delta_power))
                )
                score = np.where(corrected, candidate, base_score).astype(np.float32)
                if force:
                    force_mask = self._age_code_force_mask(state=state, step=step, base=base)
                    if force_mask.any():
                        score = score.copy()
                        finite = score[np.isfinite(score)]
                        high = float(np.max(finite)) if finite.size else 1.0
                        if not math.isfinite(high) or high <= 0.0:
                            high = 1.0
                        score[force_mask] = np.float32(high) * np.float32(1.0e6)
                return score.astype(np.float32)
            if v_mode in {
                "vbinmeanold",
                "vbinp75old",
                "vbinp90old",
                "vbinmeanp90old",
                "vbinp75p90old",
                "globalp75p90old",
                "vbinshiftp75p90old",
                "vbinrevp75p90old",
                "vbinp75all",
                "vbinp90all",
                "lutfloorp75p90old",
            }:
                if "p90" in v_mode:
                    stat = "p90"
                elif "p75" in v_mode:
                    stat = "p75"
                else:
                    stat = "mean"
                if v_mode == "lutfloorp75p90old":
                    low_floor = self._delta_stat_by_log_v_lut(
                        v=v,
                        delta_sq=delta_sq,
                        stat="p75",
                        h_lut_bins=int(h_lut_bins),
                        log_v_min=float(log_v_min),
                        inv_log_v_width=float(inv_log_v_width),
                        interpolate=True,
                    )
                    high_floor = self._delta_stat_by_log_v_lut(
                        v=v,
                        delta_sq=delta_sq,
                        stat="p90",
                        h_lut_bins=int(h_lut_bins),
                        log_v_min=float(log_v_min),
                        inv_log_v_width=float(inv_log_v_width),
                        interpolate=True,
                    )
                    floor = np.where(state.age_code >= np.uint8(3), high_floor, low_floor)
                elif v_mode == "globalp75p90old":
                    low_floor = np.full(
                        delta_sq.shape,
                        np.float32(np.percentile(delta_sq, 75.0)),
                        dtype=np.float32,
                    )
                    high_floor = np.full(
                        delta_sq.shape,
                        np.float32(np.percentile(delta_sq, 90.0)),
                        dtype=np.float32,
                    )
                    floor = np.where(state.age_code >= np.uint8(3), high_floor, low_floor)
                elif "meanp90" in v_mode or "p75p90" in v_mode:
                    low_stat = "p75" if "p75p90" in v_mode else "mean"
                    bin_shift = 0
                    reverse_bins = False
                    if v_mode == "vbinshiftp75p90old":
                        bin_shift = 8
                    elif v_mode == "vbinrevp75p90old":
                        reverse_bins = True
                    low_floor = self._delta_stat_by_v_rank(
                        v=v,
                        delta_sq=delta_sq,
                        stat=low_stat,
                        bin_shift=bin_shift,
                        reverse_bins=reverse_bins,
                    )
                    high_floor = self._delta_stat_by_v_rank(
                        v=v,
                        delta_sq=delta_sq,
                        stat="p90",
                        bin_shift=bin_shift,
                        reverse_bins=reverse_bins,
                    )
                    floor = np.where(state.age_code >= np.uint8(3), high_floor, low_floor)
                else:
                    floor = self._delta_stat_by_v_rank(
                        v=v,
                        delta_sq=delta_sq,
                        stat=stat,
                    )
                scale_source = scale_source.copy()
                if v_mode.endswith("all"):
                    scale_source = np.maximum(scale_source, floor)
                else:
                    scale_source[old_code] = np.maximum(scale_source[old_code], floor[old_code])
            score = (
                h
                * np.power(quant_age, np.float32(age_power))
                * np.power(scale_source, np.float32(delta_power))
            )
            if v_mode == "vrankold":
                # Scale-free old-code correction: v-rank is a slow activity
                # proxy, but only old 2-bit states get the boost.  Young
                # states stay governed by current delta.
                v_rank = self._rank01(np.log(np.maximum(v, np.float32(1.0e-30))))
                score = score * (np.float32(1.0) + old_code.astype(np.float32) * v_rank)
            elif v_mode == "vnormold":
                # Adam relation check: delta^2 * v is proportional to m^2.
                # Normalize by the current sampled median to keep this
                # experiment scale-free across layers/models.
                med_v = float(np.median(np.maximum(v, np.float32(1.0e-30))))
                if math.isfinite(med_v) and med_v > 0.0:
                    v_norm = np.maximum(v, np.float32(1.0e-30)) / np.float32(med_v)
                    score = score * np.maximum(
                        np.float32(1.0),
                        np.where(old_code, v_norm, np.float32(1.0)).astype(np.float32),
                    )
            if force:
                force_mask = self._age_code_force_mask(state=state, step=step, base=base)
                if force_mask.any():
                    score = score.copy()
                    finite = score[np.isfinite(score)]
                    high = float(np.max(finite)) if finite.size else 1.0
                    if not math.isfinite(high) or high <= 0.0:
                        high = 1.0
                    score[force_mask] = np.float32(high) * np.float32(1.0e6)
            return score.astype(np.float32)
        if name.startswith("h_fit"):
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            quant_age = age
            if "_q2b" in name:
                base_text = name.split("_q2b", 1)[1].split("_", 1)[0]
                base = max(1.25, float(base_text or "4"))
                bucket = np.minimum(np.floor(np.log(age) / math.log(base)).astype(np.int32), 3)
                quant_age = np.power(np.float32(base), bucket).astype(np.float32)
            elif "_q" in name:
                # h_fit_q4_age_delta, h_fit_q3_age_delta
                bit_text = name.split("_q", 1)[1].split("_", 1)[0]
                bits = max(1, min(8, int(bit_text or "4")))
                bucket = np.minimum(np.floor(np.log2(age)).astype(np.int32), (1 << bits) - 1)
                quant_age = np.exp2(bucket).astype(np.float32)
            calibrated = (
                h
                * np.power(quant_age, np.float32(state.fitted_age_power))
                * np.power(delta_sq + np.float32(1.0e-30), np.float32(state.fitted_delta_power))
            )
            if name.startswith("h_fitguard") and state.use_fitted_score < 0.5:
                return h * quant_age
            return calibrated
        if name.startswith("h_agepow") and "_vpow" in name:
            left, right = name.split("_vpow", 1)
            age_power = float((left[len("h_agepow"):] or "1").replace("_", "."))
            v_power = float((right or "1").replace("_", "."))
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return h * np.power(age, np.float32(age_power)) * np.power(np.maximum(v, np.float32(1.0e-30)), np.float32(v_power))
        if name.startswith("h_age_delta_ema"):
            span = max(1, int(name[len("h_age_delta_ema"):] or "16"))
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            ema = self._delta_ema.get(span, delta_sq)
            return h * age * ema
        if name.startswith("h_age_groupdelta_ema"):
            span = max(1, int(name[len("h_age_groupdelta_ema"):] or "16"))
            group_ids = self._sample_group_ids
            if group_ids is None:
                return h_delta_score
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            group_ema = self._group_delta_ema.get(span)
            if group_ema is None:
                return h * age
            return h * age * group_ema[group_ids]
        if name.startswith("h_ar_groupema"):
            # AR(1)-motivated conditional residual estimate:
            # E[(sum delta)^2 | current delta, sigma_g^2]
            # = A(age, rho) * sigma_g^2 + B(age, rho) * delta^2.
            span_text, rho_text = name[len("h_ar_groupema"):].split("_rho", 1)
            span = max(1, int(span_text or "64"))
            rho = float((rho_text or "0").replace("_", "."))
            group_ids = self._sample_group_ids
            group_ema = self._group_delta_ema.get(span)
            if group_ids is None or group_ema is None:
                return h_delta_score
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            a_factor, b_factor = self._ar_residual_factors(age=age, rho=rho)
            residual_est = a_factor * group_ema[group_ids] + b_factor * delta_sq
            return h * residual_est
        if name.startswith("h_latent_b"):
            # Latent drift/noise score:
            #   E[R_n] = S * n * (1 + n*kappa)
            # with S estimated by a shrinkage power of current delta^2 and
            # kappa estimated by Adam's short-window m^2/v.  This branch tests
            # whether the empirical age^a*delta^b law can be replaced by its
            # hidden-variable origin rather than a fixed exponent pair.
            suffix = name[len("h_latent_b"):]
            b_text, _, rest = suffix.partition("_")
            delta_power = float((b_text or "0p4").replace("p", "."))
            quant = "full"
            window = 16
            scale = 1.0
            use_denominator = True
            for part in (item for item in rest.split("_") if item):
                if part.startswith("q2b"):
                    quant = part
                elif part.startswith("q"):
                    quant = part
                elif part.startswith("w"):
                    window = max(1, int(part[1:] or "16"))
                elif part.startswith("s"):
                    scale = float((part[1:] or "1").replace("p", "."))
                elif part == "noden":
                    use_denominator = False
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            residual_age = age
            if quant.startswith("q2b"):
                base = max(1.25, float(quant[len("q2b"):] or "4"))
                bucket = np.minimum(np.floor(np.log(age) / math.log(base)).astype(np.int32), 3)
                residual_age = np.power(np.float32(base), bucket).astype(np.float32)
            elif quant.startswith("q"):
                bits = max(1, min(8, int(quant[1:] or "4")))
                bucket = np.minimum(np.floor(np.log2(age)).astype(np.int32), (1 << bits) - 1)
                residual_age = np.exp2(bucket).astype(np.float32)
            c2 = np.float32(scale) * (m * m) / np.maximum(v, np.float32(1.0e-30))
            snr_age = np.minimum(residual_age, np.float32(window))
            factor = np.float32(1.0) + snr_age * c2
            if use_denominator:
                factor = factor / (np.float32(1.0) + c2)
            scale_est = np.power(delta_sq + np.float32(1.0e-30), np.float32(delta_power))
            return h * scale_est * residual_age * factor
        if name.startswith("h_snr_delta_cf") or name.startswith("h_snr_cf"):
            # F14-style closed-form residual proxy.  The exponent policies use
            # age and delta^2 as indirect SNR carriers; this branch inserts the
            # resident Adam SNR proxy c^2=m^2/v directly, so the score has no
            # fitted powers.  The SNR term is clipped to a finite momentum
            # window; otherwise a short-horizon moment would be extrapolated
            # across arbitrarily old residual age.
            include_delta_scale = name.startswith("h_snr_delta_cf")
            suffix = name[len("h_snr_delta_cf"):] if include_delta_scale else name[len("h_snr_cf"):]
            quant = "full"
            window = 16
            scale = 1.0
            use_denominator = True
            for part in (item for item in suffix.split("_") if item):
                if part.startswith("q2b"):
                    quant = part
                elif part.startswith("q"):
                    quant = part
                elif part.startswith("w"):
                    window = max(1, int(part[1:] or "16"))
                elif part.startswith("s"):
                    scale = float((part[1:] or "1").replace("p", "."))
                elif part == "noden":
                    use_denominator = False
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            residual_age = age
            if quant.startswith("q2b"):
                base = max(1.25, float(quant[len("q2b"):] or "4"))
                bucket = np.minimum(np.floor(np.log(age) / math.log(base)).astype(np.int32), 3)
                residual_age = np.power(np.float32(base), bucket).astype(np.float32)
            elif quant.startswith("q"):
                bits = max(1, min(8, int(quant[1:] or "4")))
                bucket = np.minimum(np.floor(np.log2(age)).astype(np.int32), (1 << bits) - 1)
                residual_age = np.exp2(bucket).astype(np.float32)
            c2 = np.float32(scale) * (m * m) / (np.maximum(v, np.float32(1.0e-30)))
            snr_age = np.minimum(residual_age, np.float32(window))
            numerator = np.float32(1.0) + snr_age * c2
            if use_denominator:
                residual_est = residual_age * (numerator / (np.float32(1.0) + c2))
            else:
                residual_est = residual_age * numerator
            if include_delta_scale:
                residual_est = residual_est * (delta_sq + np.float32(1.0e-30))
            return h * residual_est.astype(np.float32)
        if name.startswith("h_selectgrid"):
            # Calibration-cache selector.  It does not invent a new score; it
            # chooses among a small mathematically motivated family using only
            # a held-out calibration subset that has true theta_cp residuals.
            # This tests whether the exponent pair is a phase/K-dependent
            # projection instead of a universal constant.
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            quant = "q4"
            if "_q2b4" in name:
                quant = "q2b4"
            elif "_full" in name:
                quant = "full"
            if quant == "full":
                quant_age = age
            elif quant == "q2b4":
                bucket = np.minimum(np.floor(np.log(age) / math.log(4.0)).astype(np.int32), 3)
                quant_age = np.power(np.float32(4.0), bucket).astype(np.float32)
            else:
                bucket = np.minimum(np.floor(np.log2(age)).astype(np.int32), 15)
                quant_age = np.exp2(bucket).astype(np.float32)
            group_ids = self._sample_group_ids
            if group_ids is None:
                cal_mask = (np.arange(n, dtype=np.int64) & 1) == 0
            else:
                cal_mask = (group_ids & 1) == 0
            truth = h * ((p - state.theta_cp) ** 2)
            k_cal = max(1, min(int(cal_mask.sum()), int(math.ceil(float(k) * self._safe_div(float(cal_mask.sum()), float(n))))))
            age_powers = (1.0, 1.25, 1.5, 1.75, 2.0)
            delta_powers = (0.0, 0.2, 0.4, 0.6)
            if "_narrow" in name:
                age_powers = (1.25, 1.5, 1.75, 2.0)
                delta_powers = (0.2, 0.35, 0.5)
            best_score = h * quant_age
            best_mass = -1.0
            cal_idx = np.nonzero(cal_mask)[0]
            for age_power in age_powers:
                age_part = np.power(quant_age, np.float32(age_power))
                for delta_power in delta_powers:
                    cand = h * age_part
                    if delta_power != 0.0:
                        cand = cand * np.power(delta_sq + np.float32(1.0e-30), np.float32(delta_power))
                    cal_score = cand[cal_idx]
                    if k_cal >= cal_score.size:
                        pick_local = np.ones(cal_score.shape, dtype=bool)
                    else:
                        pick_local = np.zeros(cal_score.shape, dtype=bool)
                        pick_local[np.argpartition(cal_score, -k_cal)[-k_cal:]] = True
                    mass = float(truth[cal_idx[pick_local]].sum())
                    if mass > best_mass:
                        best_mass = mass
                        best_score = cand
            return best_score.astype(np.float32)
        if name == "age":
            return np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
        if name == "h_age":
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return h * age
        if name.startswith("h_age_pow"):
            raw = name[len("h_age_pow"):] or "1"
            gamma = float(raw.replace("_", "."))
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return h * np.power(age, np.float32(gamma))
        if name == "h_logage":
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return h * np.log2(age + 1.0)
        if name == "h_sqrtage":
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return h * np.sqrt(age)
        if name == "age2":
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return age * age
        if name == "h_age2":
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return h * age * age
        if name == "h_age2_theta2":
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return h * age * age * (p * p)
        if name.startswith("h_age_cap"):
            cap = max(1, int(name[len("h_age_cap"):] or "16"))
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return h * np.minimum(age, np.float32(cap))
        if name == "h_age_q2wide":
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            bucket = np.minimum(np.floor(np.log(age) / np.log(np.float32(4.0))).astype(np.int32), 3)
            return h * np.exp2(2 * bucket).astype(np.float32)
        if name == "h_age_q2wide_force64":
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            bucket = np.minimum(np.floor(np.log(age) / np.log(np.float32(4.0))).astype(np.int32), 3)
            score = h * np.exp2(2 * bucket).astype(np.float32)
            force = age > 64.0
            if force.any():
                score = score.copy()
                score[force] += np.float32(1.0e20)
            return score
        if name.startswith("h_ageq") and not name.startswith("h_ageq2b") and "_pow" in name:
            # Example: h_ageq4_pow1_9_deltapow0_4
            force128 = name.endswith("_force128")
            parse_name = name[:-len("_force128")] if force128 else name
            left, rest = parse_name.split("_pow", 1)
            bits = max(1, min(8, int(left[len("h_ageq"):] or "4")))
            if "_thetapow" in rest:
                age_power_text, theta_text = rest.split("_thetapow", 1)
                theta_power = float((theta_text or "2").replace("_", "."))
                age_power = float((age_power_text or "1").replace("_", "."))
                age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
                bucket = np.minimum(np.floor(np.log2(age)).astype(np.int32), (1 << bits) - 1)
                quant_age = np.exp2(bucket).astype(np.float32)
                score = h * np.power(quant_age, np.float32(age_power)) * np.power(p * p + np.float32(1.0e-30), np.float32(theta_power * 0.5))
                if force128:
                    score = self._force_old_age(score=score, age=age, min_age=128.0)
                return score
            age_power_text, delta_text = rest.split("_deltapow", 1)
            age_power = float((age_power_text or "1").replace("_", "."))
            delta_power = float((delta_text or "0").replace("_", "."))
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            bucket = np.minimum(np.floor(np.log2(age)).astype(np.int32), (1 << bits) - 1)
            quant_age = np.exp2(bucket).astype(np.float32)
            score = (
                h
                * np.power(quant_age, np.float32(age_power))
                * np.power(delta_sq + np.float32(1.0e-30), np.float32(delta_power))
            )
            if force128:
                score = self._force_old_age(score=score, age=age, min_age=128.0)
            return score
        if name.startswith("h_ageq2b") and "_pow" in name:
            # Example: h_ageq2b4_pow1_9_deltapow0_4 gives a 2-bit
            # codebook {1, 4, 16, 64} before exponent calibration.
            force128 = name.endswith("_force128")
            parse_name = name[:-len("_force128")] if force128 else name
            left, rest = parse_name.split("_pow", 1)
            base = max(1.25, float(left[len("h_ageq2b"):] or "4"))
            if "_thetapow" in rest:
                age_power_text, theta_text = rest.split("_thetapow", 1)
                age_power = float((age_power_text or "1").replace("_", "."))
                theta_power = float((theta_text or "2").replace("_", "."))
                age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
                bucket = np.minimum(np.floor(np.log(age) / math.log(base)).astype(np.int32), 3)
                quant_age = np.power(np.float32(base), bucket).astype(np.float32)
                score = h * np.power(quant_age, np.float32(age_power)) * np.power(p * p + np.float32(1.0e-30), np.float32(theta_power * 0.5))
                if force128:
                    score = self._force_old_age(score=score, age=age, min_age=128.0)
                return score
            age_power_text, delta_text = rest.split("_deltapow", 1)
            age_power = float((age_power_text or "1").replace("_", "."))
            delta_power = float((delta_text or "0").replace("_", "."))
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            bucket = np.minimum(np.floor(np.log(age) / math.log(base)).astype(np.int32), 3)
            quant_age = np.power(np.float32(base), bucket).astype(np.float32)
            score = (
                h
                * np.power(quant_age, np.float32(age_power))
                * np.power(delta_sq + np.float32(1.0e-30), np.float32(delta_power))
            )
            if force128:
                score = self._force_old_age(score=score, age=age, min_age=128.0)
            return score
        if name.startswith("h_age_q2b"):
            base = max(1.25, float(name[len("h_age_q2b"):] or "4"))
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            bucket = np.minimum(np.floor(np.log(age) / math.log(base)).astype(np.int32), 3)
            return h * np.power(np.float32(base), bucket).astype(np.float32)
        if name.startswith("h_age_q"):
            bits = max(1, min(8, int(name[len("h_age_q"):] or "4")))
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            bucket = np.minimum(np.floor(np.log2(age)).astype(np.int32), (1 << bits) - 1)
            quant_age = np.exp2(bucket).astype(np.float32)
            return h * quant_age
        if name.startswith("h_stale"):
            th = max(1, int(name[len("h_stale"):] or "8"))
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return h * (age >= th).astype(np.float32)
        if name.startswith("h_oldest"):
            keep = max(1, min(self._group_size, int(name[len("h_oldest"):] or "1")))
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return h * age * self._oldest_lane_mask(age=age, keep=keep).astype(np.float32)
        if name == "h_rank2":
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return h * self._rank2_age_weight(age=age)
        if name in {"h_groupmaxage", "h_groupmeanage"}:
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            group_age = self._group_age(age=age, reduce="max" if "max" in name else "mean")
            return h * group_age
        if name in {"h_groupmaxage_rank2", "h_groupmeanage_rank2"}:
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            reduce = "max" if "max" in name else "mean"
            return h * self._group_age(age=age, reduce=reduce) * self._rank2_age_weight(age=age)
        if name.startswith("h_groupmaxage_oldest") or name.startswith("h_groupmeanage_oldest"):
            if name.startswith("h_groupmaxage_oldest"):
                keep = max(1, min(self._group_size, int(name[len("h_groupmaxage_oldest"):] or "1")))
                reduce = "max"
            else:
                keep = max(1, min(self._group_size, int(name[len("h_groupmeanage_oldest"):] or "1")))
                reduce = "mean"
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return (
                h
                * self._group_age(age=age, reduce=reduce)
                * self._oldest_lane_mask(age=age, keep=keep).astype(np.float32)
            )
        if name == "mix_age_spike":
            age_score = h * np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            return np.maximum(
                self._normalize_by_kth(age_score, k),
                self._normalize_by_kth(h_delta_score, k),
            )
        if name == "mix_q4_spike":
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            bucket = np.minimum(np.floor(np.log2(age)).astype(np.int32), 15)
            age_score = h * np.exp2(bucket).astype(np.float32)
            return np.maximum(
                self._normalize_by_kth(age_score, k),
                self._normalize_by_kth(h_delta_score, k),
            )
        if name.startswith("dg_age"):
            group_ids = self._sample_group_ids
            if group_ids is None:
                return h_delta_score
            n_groups = state.group_debt.size
            active = ~state.last_written
            age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
            weighted_age = age * active.astype(np.float32)
            age_sum = np.bincount(
                group_ids,
                weights=weighted_age.astype(np.float64),
                minlength=n_groups,
            ).astype(np.float32)
            floor = np.zeros_like(age, dtype=np.float32)
            valid = active & (age_sum[group_ids] > 0)
            floor[valid] = state.group_debt[group_ids[valid]] * age[valid] / age_sum[group_ids[valid]]
            return h * (np.float32(state.lambda_floor) * floor + delta_sq)
        if name.startswith("dg"):
            group_ids = self._sample_group_ids
            if group_ids is None:
                return h_delta_score
            n_groups = state.group_debt.size
            active = ~state.last_written
            active_counts = active.reshape(n_groups, self._group_size).sum(axis=1).astype(np.float32)
            floor = np.zeros(n_groups, dtype=np.float32)
            np.divide(
                state.group_debt,
                np.maximum(active_counts, np.float32(1.0)),
                out=floor,
                where=active_counts > 0,
            )
            return h * (np.float32(state.lambda_floor) * floor[group_ids] * active.astype(np.float32) + delta_sq)
        return h_delta_score

    def _update_policy(
        self,
        *,
        state: _PolicyState,
        step: int,
        selected: np.ndarray,
        p: np.ndarray,
        delta_sq: np.ndarray,
    ) -> None:
        if state.name.startswith("dg"):
            self._update_group_debt(
                state=state,
                step=int(step),
                selected=selected,
                delta_sq=delta_sq,
            )
        state.theta_cp[selected] = p[selected]
        state.last_written[:] = selected
        state.last_written_step[selected] = np.int32(step)
        if self._uses_age_code(state.name):
            self._update_age_code(state=state, step=int(step), written=selected)
        self._reset_policy_trace(state=state, written=selected)

    @staticmethod
    def _advance_policy_trace(*, state: _PolicyState, delta: np.ndarray) -> None:
        abs_delta = np.abs(delta).astype(np.float32)
        prev = state.prev_delta
        has_prev = state.cum_abs_delta > np.float32(0.0)
        flip = has_prev & (prev * delta < np.float32(0.0))
        if flip.any():
            state.sign_flip_count[flip] = np.minimum(
                state.sign_flip_count[flip].astype(np.uint32) + 1,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
        state.cum_delta += delta.astype(np.float32)
        state.cum_abs_delta += abs_delta
        state.cum_delta_sq += (delta * delta).astype(np.float32)
        np.maximum(state.max_abs_delta, abs_delta, out=state.max_abs_delta)
        state.prev_delta[:] = delta.astype(np.float32)

    @staticmethod
    def _reset_policy_trace(*, state: _PolicyState, written: np.ndarray) -> None:
        if not written.any():
            return
        state.cum_delta[written] = np.float32(0.0)
        state.cum_abs_delta[written] = np.float32(0.0)
        state.cum_delta_sq[written] = np.float32(0.0)
        state.max_abs_delta[written] = np.float32(0.0)
        state.prev_delta[written] = np.float32(0.0)
        state.sign_flip_count[written] = np.uint16(0)

    def _update_group_debt(
        self,
        *,
        state: _PolicyState,
        step: int,
        selected: np.ndarray,
        delta_sq: np.ndarray,
    ) -> None:
        group_ids = self._sample_group_ids
        if group_ids is None:
            return
        n_groups = state.group_debt.size
        active = ~state.last_written
        remove = selected & active
        if remove.any():
            if state.name.startswith("dg_age"):
                age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
                weighted_age = age * active.astype(np.float32)
                age_sum = np.bincount(
                    group_ids,
                    weights=weighted_age.astype(np.float64),
                    minlength=n_groups,
                ).astype(np.float32)
                floor_i = np.zeros_like(age, dtype=np.float32)
                valid = active & (age_sum[group_ids] > 0)
                floor_i[valid] = (
                    state.group_debt[group_ids[valid]]
                    * age[valid]
                    / age_sum[group_ids[valid]]
                )
                weights = floor_i[remove].astype(np.float64)
            else:
                active_counts = active.reshape(n_groups, self._group_size).sum(axis=1).astype(np.float32)
                floor = np.zeros(n_groups, dtype=np.float32)
                np.divide(
                    state.group_debt,
                    np.maximum(active_counts, np.float32(1.0)),
                    out=floor,
                    where=active_counts > 0,
                )
                weights = floor[group_ids[remove]].astype(np.float64)
            remove_mass = np.bincount(
                group_ids[remove],
                weights=weights,
                minlength=n_groups,
            ).astype(np.float32)
            state.group_debt -= remove_mass
            np.maximum(state.group_debt, np.float32(0.0), out=state.group_debt)
        add = ~selected
        if add.any():
            add_mass = np.bincount(
                group_ids[add],
                weights=delta_sq[add].astype(np.float64),
                minlength=n_groups,
            ).astype(np.float32)
            state.group_debt += add_mass
        state.last_written[:] = selected

    def _update_fitted_exponents(
        self,
        *,
        state: _PolicyState,
        step: int,
        p: np.ndarray,
        h: np.ndarray,
        delta_sq: np.ndarray,
        k: int,
    ) -> None:
        residual_sq = (p - state.theta_cp) ** 2
        valid = residual_sq > np.float32(1.0e-30)
        if int(valid.sum()) < 128:
            return
        age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
        fit_mask = valid.copy()
        truth_score = h * residual_sq
        weights = None
        group_ids = self._sample_group_ids
        train_mask = np.ones_like(valid, dtype=bool)
        holdout_mask = np.ones_like(valid, dtype=bool)
        if state.name.startswith("h_fitguard") and group_ids is not None:
            train_mask = (group_ids % 4) < 2
            holdout_mask = (group_ids % 4) == 2
            fit_mask &= train_mask
        if state.name.startswith("h_fitband"):
            n_fit = min(int(fit_mask.sum()), max(128, int(4 * max(1, k))))
            idx = np.argpartition(truth_score, -n_fit)[-n_fit:]
            fit_mask[:] = False
            fit_mask[idx] = True
            fit_mask &= valid
        elif state.name.startswith("h_fitmass"):
            weights = truth_score[valid].astype(np.float64)
            weights = weights / max(float(np.mean(weights)), 1.0e-30)
        target = np.log(residual_sq[fit_mask] + np.float32(1.0e-30)).astype(np.float64)
        features = [
            np.log(age[fit_mask] + np.float32(1.0e-30)).astype(np.float64),
            np.log(delta_sq[fit_mask] + np.float32(1.0e-30)).astype(np.float64),
        ]
        if state.name.startswith("h_fitmass") and weights is not None:
            _pred, _r2, coefs = self._fit_log_linear_weighted(
                target=target,
                features=features,
                weights=weights,
            )
        else:
            _pred, _r2, coefs = self._fit_log_linear(target=target, features=features)
        # Clamp only prevents pathological early rows from inverting the
        # monotonic stale signal.  Fast variants use the last completed fit
        # directly; others smooth the calibration estimate.
        new_age = float(np.clip(coefs[1], 0.25, 2.5))
        new_delta = float(np.clip(coefs[2], 0.0, 0.75))
        beta = 1.0 if "fast" in state.name else 0.10
        state.fitted_age_power = (1.0 - beta) * float(state.fitted_age_power) + beta * new_age
        state.fitted_delta_power = (1.0 - beta) * float(state.fitted_delta_power) + beta * new_delta
        if state.name.startswith("h_fitguard"):
            eval_mask = valid & holdout_mask
            if int(eval_mask.sum()) < 128:
                state.use_fitted_score = 0.0
                return
            eval_idx = np.nonzero(eval_mask)[0]
            k_eval = max(1, min(int(eval_idx.size), int(math.ceil(float(k) * self._safe_div(float(eval_idx.size), float(delta_sq.size))))))
            quant_age = age
            if "_q2b" in state.name:
                base_text = state.name.split("_q2b", 1)[1].split("_", 1)[0]
                base = max(1.25, float(base_text or "4"))
                bucket = np.minimum(np.floor(np.log(age) / math.log(base)).astype(np.int32), 3)
                quant_age = np.power(np.float32(base), bucket).astype(np.float32)
            elif "_q" in state.name:
                bit_text = state.name.split("_q", 1)[1].split("_", 1)[0]
                bits = max(1, min(8, int(bit_text or "4")))
                bucket = np.minimum(np.floor(np.log2(age)).astype(np.int32), (1 << bits) - 1)
                quant_age = np.exp2(bucket).astype(np.float32)
            base_score = (h * quant_age)[eval_idx]
            fit_score = (
                h
                * np.power(quant_age, np.float32(state.fitted_age_power))
                * np.power(delta_sq + np.float32(1.0e-30), np.float32(state.fitted_delta_power))
            )[eval_idx]
            truth_eval = truth_score[eval_idx]
            base_pick = np.zeros(eval_idx.size, dtype=bool)
            fit_pick = np.zeros(eval_idx.size, dtype=bool)
            base_pick[np.argpartition(base_score, -k_eval)[-k_eval:]] = True
            fit_pick[np.argpartition(fit_score, -k_eval)[-k_eval:]] = True
            base_mass = float(truth_eval[base_pick].sum())
            fit_mass = float(truth_eval[fit_pick].sum())
            state.use_fitted_score = 1.0 if fit_mass >= base_mass else 0.0

    @staticmethod
    def _topk_mask(score: np.ndarray, k: int) -> np.ndarray:
        k = max(1, min(int(k), int(score.size)))
        mask = np.zeros(score.shape[0], dtype=bool)
        if k >= score.size:
            mask[:] = True
            return mask
        idx = np.argpartition(score, -k)[-k:]
        mask[idx] = True
        return mask

    def _oldest_lane_mask(self, *, age: np.ndarray, keep: int) -> np.ndarray:
        n_groups = max(1, int(age.size // max(1, self._group_size)))
        age_g = age.reshape(n_groups, self._group_size)
        order = np.argsort(-age_g, axis=1)
        mask = np.zeros_like(age_g, dtype=bool)
        rows = np.arange(n_groups)[:, None]
        mask[rows, order[:, :keep]] = True
        return mask.reshape(age.shape[0])

    def _rank2_age_weight(self, *, age: np.ndarray) -> np.ndarray:
        n_groups = max(1, int(age.size // max(1, self._group_size)))
        age_g = age.reshape(n_groups, self._group_size)
        order = np.argsort(-age_g, axis=1)
        weights = np.zeros_like(age_g, dtype=np.float32)
        # Four ordinal classes emulate 2-bit group-local rank.  For G=8 this
        # gives two lanes per class: 8, 4, 2, 1.
        class_size = max(1, int(math.ceil(float(self._group_size) / 4.0)))
        class_values = np.asarray([8.0, 4.0, 2.0, 1.0], dtype=np.float32)
        rows = np.arange(n_groups)[:, None]
        for cls, value in enumerate(class_values):
            lo = cls * class_size
            hi = min(self._group_size, (cls + 1) * class_size)
            if lo >= hi:
                continue
            weights[rows, order[:, lo:hi]] = value
        return weights.reshape(age.shape[0])

    def _group_age(self, *, age: np.ndarray, reduce: str) -> np.ndarray:
        n_groups = max(1, int(age.size // max(1, self._group_size)))
        age_g = age.reshape(n_groups, self._group_size)
        if reduce == "mean":
            group_age = age_g.mean(axis=1)
        else:
            group_age = age_g.max(axis=1)
        return np.repeat(group_age.astype(np.float32), self._group_size)

    @staticmethod
    def _normalize_by_kth(score: np.ndarray, k: int) -> np.ndarray:
        k = max(1, min(int(k), int(score.size)))
        if k >= score.size:
            scale = float(np.max(score))
        else:
            scale = float(np.partition(score, -k)[-k])
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return score / np.float32(scale)

    @staticmethod
    def _force_old_age(*, score: np.ndarray, age: np.ndarray, min_age: float) -> np.ndarray:
        force = age >= np.float32(min_age)
        if not force.any():
            return score
        forced = score.copy()
        finite = forced[np.isfinite(forced)]
        base = float(np.max(finite)) if finite.size else 1.0
        if not math.isfinite(base) or base <= 0.0:
            base = 1.0
        forced[force] = np.float32(base) * np.float32(1.0e6)
        return forced

    @staticmethod
    def _uses_age_code(name: str) -> bool:
        return (
            name.startswith("h_agecode2b")
            or name.startswith("h_agedet2b")
            or name.startswith("h_agehard2b")
        )

    @staticmethod
    def _age_code_base(name: str) -> float:
        if name.startswith("h_agecode2b"):
            raw = name[len("h_agecode2b"):].split("_", 1)[0]
        elif name.startswith("h_agedet2b"):
            raw = name[len("h_agedet2b"):].split("_", 1)[0]
        elif name.startswith("h_agehard2b"):
            raw = name[len("h_agehard2b"):].split("_", 1)[0]
        else:
            raw = "4"
        return max(1.25, float(raw or "4"))

    @staticmethod
    def _age_hard_periods(base: float) -> Tuple[int, int, int, int]:
        # 2-bit hard-horizon schedule.  For base=4 this gives codebook
        # {1,4,16,64} and a force horizon near 128 without storing exact age:
        #   transition gaps: 3, 12, 48, remaining-to-128.
        # The first three gaps are differences between adjacent logarithmic
        # representatives; the final period is the stale-force budget.
        horizon = max(4, int(round(2.0 * (float(base) ** 3))))
        first = max(1, int(round(float(base) - 1.0)))
        second = max(1, int(round(float(base) * (float(base) - 1.0))))
        third = max(1, int(round((float(base) ** 2) * (float(base) - 1.0))))
        force = max(1, horizon - first - second - third)
        return first, second, third, force

    @staticmethod
    def _rank01(values: np.ndarray) -> np.ndarray:
        n = int(values.size)
        if n <= 0:
            return np.zeros_like(values, dtype=np.float32)
        order = np.argsort(values.astype(np.float64), kind="mergesort")
        rank = np.empty(n, dtype=np.float32)
        rank[order] = (np.arange(n, dtype=np.float32) + np.float32(0.5)) / np.float32(n)
        return rank

    def _delta_stat_by_v_rank(
        self,
        *,
        v: np.ndarray,
        delta_sq: np.ndarray,
        stat: str,
        mask: Optional[np.ndarray] = None,
        bins: int = 16,
        bin_shift: int = 0,
        reverse_bins: bool = False,
    ) -> np.ndarray:
        rank = self._rank01(np.log(np.maximum(v, np.float32(1.0e-30))))
        bin_id = np.minimum((rank * np.float32(bins)).astype(np.int32), bins - 1)
        floor = np.zeros(bins, dtype=np.float32)
        global_mean = float(np.mean(delta_sq)) if delta_sq.size else 0.0
        source_mask = np.ones(delta_sq.shape, dtype=bool) if mask is None else mask.astype(bool, copy=False)
        for b in range(bins):
            bin_mask = bin_id == b
            use = bin_mask & source_mask
            if not use.any():
                use = bin_mask
            if not use.any():
                floor[b] = np.float32(global_mean)
            elif stat == "mean":
                floor[b] = np.float32(np.mean(delta_sq[use]))
            elif stat == "median":
                floor[b] = np.float32(np.percentile(delta_sq[use], 50.0))
            elif stat == "p75":
                floor[b] = np.float32(np.percentile(delta_sq[use], 75.0))
            elif stat == "p90":
                floor[b] = np.float32(np.percentile(delta_sq[use], 90.0))
            else:
                raise ValueError(f"unknown delta v-rank statistic: {stat}")
        lookup_bin = bin_id
        if reverse_bins:
            lookup_bin = (bins - 1) - lookup_bin
        elif int(bin_shift) != 0:
            lookup_bin = np.remainder(lookup_bin + int(bin_shift), bins).astype(np.int32)
        return floor[lookup_bin].astype(np.float32)

    def _delta_stat_by_log_v_lut(
        self,
        *,
        v: np.ndarray,
        delta_sq: np.ndarray,
        stat: str,
        h_lut_bins: int,
        log_v_min: float,
        inv_log_v_width: float,
        bins: int = 16,
        mask: Optional[np.ndarray] = None,
        interpolate: bool = True,
    ) -> np.ndarray:
        if int(h_lut_bins) <= 1 or int(bins) <= 1:
            return np.full(delta_sq.shape, np.float32(np.mean(delta_sq)), dtype=np.float32)
        log_v = np.log(np.maximum(v, np.float32(1.0e-30))).astype(np.float32)
        pos = (log_v - np.float32(log_v_min)) * np.float32(inv_log_v_width)
        pos = np.clip(pos, np.float32(0.0), np.float32(int(h_lut_bins) - 1))
        vbin_pos = pos * (np.float32(bins) / np.float32(int(h_lut_bins) - 1))
        bin_id = np.minimum(vbin_pos.astype(np.int32), int(bins) - 1)
        floor = np.zeros(int(bins), dtype=np.float32)
        global_mean = float(np.mean(delta_sq)) if delta_sq.size else 0.0
        source_mask = np.ones(delta_sq.shape, dtype=bool) if mask is None else mask.astype(bool, copy=False)
        for b in range(int(bins)):
            use = (bin_id == b) & source_mask
            if not use.any():
                use = bin_id == b
            if not use.any():
                floor[b] = np.float32(global_mean)
            elif stat == "mean":
                floor[b] = np.float32(np.mean(delta_sq[use]))
            elif stat == "median":
                floor[b] = np.float32(np.percentile(delta_sq[use], 50.0))
            elif stat == "p75":
                floor[b] = np.float32(np.percentile(delta_sq[use], 75.0))
            elif stat == "p90":
                floor[b] = np.float32(np.percentile(delta_sq[use], 90.0))
            else:
                raise ValueError(f"unknown log-v delta statistic: {stat}")
        if not bool(interpolate):
            return floor[bin_id].astype(np.float32)
        floor_pos = np.clip(vbin_pos - np.float32(0.5), np.float32(0.0), np.float32(int(bins) - 1))
        lo = np.floor(floor_pos).astype(np.int32)
        hi = np.minimum(lo + 1, int(bins) - 1)
        frac = (floor_pos - lo.astype(np.float32)).astype(np.float32)
        lo_v = floor[lo]
        hi_v = floor[hi]
        interp = lo_v + frac * (hi_v - lo_v)
        return np.where((lo_v > 0.0) & (hi_v > 0.0), interp, np.maximum(lo_v, hi_v)).astype(np.float32)

    @staticmethod
    def _mask_age_v_stats(
        *,
        age: np.ndarray,
        v_rank: np.ndarray,
        truth_score: np.ndarray,
        masks: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, mask in masks.items():
            if not mask.any():
                out[f"{name}_age_mean"] = 0.0
                out[f"{name}_age_p50"] = 0.0
                out[f"{name}_age_p90"] = 0.0
                out[f"{name}_age_mass_mean"] = 0.0
                out[f"{name}_v_rank_mean"] = 0.0
                out[f"{name}_v_rank_p90"] = 0.0
                continue
            score = truth_score[mask].astype(np.float64)
            score_sum = float(score.sum())
            out[f"{name}_age_mean"] = float(age[mask].mean())
            out[f"{name}_age_p50"] = float(np.percentile(age[mask], 50.0))
            out[f"{name}_age_p90"] = float(np.percentile(age[mask], 90.0))
            out[f"{name}_age_mass_mean"] = (
                float(np.sum(age[mask].astype(np.float64) * score) / max(score_sum, 1.0e-30))
            )
            out[f"{name}_v_rank_mean"] = float(v_rank[mask].mean())
            out[f"{name}_v_rank_p90"] = float(np.percentile(v_rank[mask], 90.0))
        return out

    def _sample_ids(self, n: int) -> np.ndarray:
        if self._sample_element_ids is not None and int(self._sample_element_ids.size) == int(n):
            return self._sample_element_ids
        return np.arange(int(n), dtype=np.uint64)

    @staticmethod
    def _hash_u64(ids: np.ndarray, *, step: int, salt: int) -> np.ndarray:
        x = ids.astype(np.uint64, copy=False)
        x = x + np.uint64(0x9E3779B97F4A7C15) * np.uint64(int(step) + 1)
        x = x ^ np.uint64(int(salt) & 0xFFFFFFFFFFFFFFFF)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))

    @classmethod
    def _hash_uniform01(cls, ids: np.ndarray, *, step: int, salt: int) -> np.ndarray:
        x = cls._hash_u64(ids, step=step, salt=salt)
        return ((x >> np.uint64(11)).astype(np.float64)) * (1.0 / float(1 << 53))

    def _age_code_salt(self, name: str, *, extra: int = 0) -> int:
        return int(zlib.crc32(name.encode("utf-8")) ^ int(extra))

    def _age_code_force_mask(self, *, state: _PolicyState, step: int, base: float) -> np.ndarray:
        code = state.age_code
        ids = self._sample_ids(int(code.size))
        if state.name.startswith("h_agehard2b"):
            period = self._age_hard_periods(base)[3]
            h = self._hash_u64(
                ids,
                step=0,
                salt=self._age_code_salt(state.name, extra=0xA501),
            )
            phase = np.remainder(h, np.uint64(period)).astype(np.int64)
            return (code >= 3) & (phase == (int(step) % period))
        if state.name.startswith("h_agedet2b"):
            period = max(1, int(round(2.0 * float(base) ** 3)))
            h = self._hash_u64(
                ids,
                step=0,
                salt=self._age_code_salt(state.name, extra=0xA501),
            )
            phase = np.remainder(h, np.uint64(period)).astype(np.int64)
            return (code >= 3) & (phase == (int(step) % period))
        p_force = 1.0 / max(float(base) ** 3, 1.0)
        u = self._hash_uniform01(
            ids,
            step=int(step),
            salt=self._age_code_salt(state.name, extra=0xA501),
        )
        return (code >= 3) & (u < p_force)

    def _update_age_code(self, *, state: _PolicyState, step: int, written: np.ndarray) -> None:
        code = state.age_code
        if code.size == 0:
            return
        code[written] = np.uint8(0)
        not_written = ~written
        promotable = not_written & (code < 3)
        if not promotable.any():
            return
        base = self._age_code_base(state.name)
        ids = self._sample_ids(int(code.size))
        if state.name.startswith("h_agedet2b") or state.name.startswith("h_agehard2b"):
            if state.name.startswith("h_agehard2b"):
                # Hard-horizon variant: periods are the logarithmic codebook
                # gaps, so the sum of worst-case waits is bounded by the
                # stale-force horizon (base=4 -> about 128 steps).
                periods = list(self._age_hard_periods(base)[:3])
            else:
                # Deterministic bounded analogue of dwell means 3,12,48:
                # a period of 2*dwell gives the same mean wait under a random
                # element phase, but avoids geometric long tails.
                periods = [
                    max(1, int(round(2.0 * (base - 1.0) * (base ** idx))))
                    for idx in range(3)
                ]
            promote = np.zeros(code.shape, dtype=bool)
            for value, period in enumerate(periods):
                mask = promotable & (code == value)
                if not mask.any():
                    continue
                h = self._hash_u64(
                    ids,
                    step=0,
                    salt=self._age_code_salt(state.name, extra=0xD00D + value),
                )
                phase = np.remainder(h, np.uint64(period)).astype(np.int64)
                promote |= mask & (phase == (int(step) % period))
            code[promote] = np.minimum(code[promote] + np.uint8(1), np.uint8(3))
            return
        # Codebook {1,4,16,64}.  Promotion dwell times are the distance to
        # the next representative: 3, 12, 48.  Code 3 has a separate force
        # event with mean dwell 64, giving total age near 128.
        probs = np.asarray([1.0 / 3.0, 1.0 / 12.0, 1.0 / 48.0, 0.0], dtype=np.float64)
        u = self._hash_uniform01(
            ids,
            step=int(step),
            salt=self._age_code_salt(state.name, extra=0xC0DE),
        )
        promote = promotable & (u < probs[code])
        code[promote] = np.minimum(code[promote] + np.uint8(1), np.uint8(3))

    def _age_code_diagnostics(
        self,
        *,
        state: _PolicyState,
        step: int,
        selected: np.ndarray,
    ) -> Dict[str, float]:
        if not self._uses_age_code(state.name):
            return {}
        code = state.age_code
        out: Dict[str, float] = {}
        exact_age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
        for value in range(4):
            mask = code == value
            out[f"age_code{value}_ratio"] = float(mask.mean())
            out[f"selected_age_code{value}_ratio"] = float(mask[selected].mean()) if selected.any() else 0.0
            out[f"age_code{value}_exact_age_mean"] = float(exact_age[mask].mean()) if mask.any() else 0.0
            out[f"age_code{value}_exact_age_p90"] = float(np.percentile(exact_age[mask], 90.0)) if mask.any() else 0.0
        force = self._age_code_force_mask(state=state, step=int(step), base=self._age_code_base(state.name))
        out["age_code_force_candidate_ratio"] = float(force.mean())
        out["selected_age_code_force_ratio"] = float(force[selected].mean()) if selected.any() else 0.0
        return out

    def _velocity_proxy_diagnostics(
        self,
        *,
        residual: np.ndarray,
        delta: np.ndarray,
        age: np.ndarray,
        state: _PolicyState,
        ideal: np.ndarray,
        selected: np.ndarray,
        missed: np.ndarray,
        extra: np.ndarray,
    ) -> Dict[str, float]:
        eps = np.float32(1.0e-30)
        velocity = age.astype(np.float32) * delta.astype(np.float32)
        ratio = np.abs(residual.astype(np.float32)) / (np.abs(velocity) + eps)
        sign_match = (residual * delta) >= 0.0
        mean_abs_delta = state.cum_abs_delta / np.maximum(age.astype(np.float32), np.float32(1.0))
        current_over_mean_abs = np.abs(delta.astype(np.float32)) / (mean_abs_delta + eps)
        max_over_current = state.max_abs_delta / (np.abs(delta.astype(np.float32)) + eps)
        directional_efficiency = np.abs(state.cum_delta) / (state.cum_abs_delta + eps)
        residual_path_efficiency = np.abs(residual.astype(np.float32)) / (state.cum_abs_delta + eps)
        flip_rate = state.sign_flip_count.astype(np.float32) / np.maximum(age.astype(np.float32), np.float32(1.0))
        rms_delta = np.sqrt(state.cum_delta_sq / np.maximum(age.astype(np.float32), np.float32(1.0)))
        current_over_rms = np.abs(delta.astype(np.float32)) / (rms_delta + eps)

        def mean_bool(mask: np.ndarray) -> float:
            return float(sign_match[mask].mean()) if mask.any() else 0.0

        def median_ratio(mask: np.ndarray) -> float:
            return float(np.median(ratio[mask])) if mask.any() else 0.0

        def p90_ratio(mask: np.ndarray) -> float:
            return float(np.percentile(ratio[mask], 90.0)) if mask.any() else 0.0

        def median_of(values: np.ndarray, mask: np.ndarray) -> float:
            return float(np.median(values[mask])) if mask.any() else 0.0

        def p90_of(values: np.ndarray, mask: np.ndarray) -> float:
            return float(np.percentile(values[mask], 90.0)) if mask.any() else 0.0

        return {
            "velocity_ratio_ideal_median": median_ratio(ideal),
            "velocity_ratio_selected_median": median_ratio(selected),
            "velocity_ratio_missed_median": median_ratio(missed),
            "velocity_ratio_extra_median": median_ratio(extra),
            "velocity_ratio_missed_p90": p90_ratio(missed),
            "velocity_sign_match_ideal": mean_bool(ideal),
            "velocity_sign_match_selected": mean_bool(selected),
            "velocity_sign_match_missed": mean_bool(missed),
            "velocity_sign_match_extra": mean_bool(extra),
            "trace_current_over_mean_abs_missed_median": median_of(current_over_mean_abs, missed),
            "trace_current_over_mean_abs_extra_median": median_of(current_over_mean_abs, extra),
            "trace_current_over_rms_missed_median": median_of(current_over_rms, missed),
            "trace_current_over_rms_extra_median": median_of(current_over_rms, extra),
            "trace_max_over_current_missed_median": median_of(max_over_current, missed),
            "trace_max_over_current_missed_p90": p90_of(max_over_current, missed),
            "trace_max_over_current_extra_median": median_of(max_over_current, extra),
            "trace_directional_eff_missed_median": median_of(directional_efficiency, missed),
            "trace_directional_eff_extra_median": median_of(directional_efficiency, extra),
            "trace_residual_path_eff_missed_median": median_of(residual_path_efficiency, missed),
            "trace_residual_path_eff_extra_median": median_of(residual_path_efficiency, extra),
            "trace_flip_rate_missed_median": median_of(flip_rate, missed),
            "trace_flip_rate_extra_median": median_of(flip_rate, extra),
        }

    def _age_bin_trace_diagnostics(
        self,
        *,
        truth_score: np.ndarray,
        residual: np.ndarray,
        delta: np.ndarray,
        age: np.ndarray,
        state: _PolicyState,
        ideal: np.ndarray,
        selected: np.ndarray,
        missed: np.ndarray,
        extra: np.ndarray,
    ) -> Dict[str, float]:
        eps = np.float32(1.0e-30)
        total_ideal_mass = max(float(truth_score[ideal].sum()), 1.0e-30)
        total_selected_mass = max(float(truth_score[selected].sum()), 1.0e-30)
        mean_abs_delta = state.cum_abs_delta / np.maximum(age.astype(np.float32), np.float32(1.0))
        current_over_mean_abs = np.abs(delta.astype(np.float32)) / (mean_abs_delta + eps)
        max_over_current = state.max_abs_delta / (np.abs(delta.astype(np.float32)) + eps)
        directional_efficiency = np.abs(state.cum_delta) / (state.cum_abs_delta + eps)
        flip_rate = state.sign_flip_count.astype(np.float32) / np.maximum(age.astype(np.float32), np.float32(1.0))
        velocity = age.astype(np.float32) * np.abs(delta.astype(np.float32))
        residual_over_velocity = np.abs(residual.astype(np.float32)) / (velocity + eps)

        bins = (
            ("lt32", age < 32.0),
            ("32_64", (age >= 32.0) & (age < 64.0)),
            ("64_128", (age >= 64.0) & (age < 128.0)),
            ("ge128", age >= 128.0),
        )
        out: Dict[str, float] = {}

        def share(mask: np.ndarray, base: np.ndarray, denom: float) -> float:
            return float(truth_score[mask & base].sum()) / denom

        def count_share(mask: np.ndarray, base: np.ndarray) -> float:
            denom = max(int(base.sum()), 1)
            return float((mask & base).sum()) / float(denom)

        def median(values: np.ndarray, mask: np.ndarray) -> float:
            return float(np.median(values[mask])) if mask.any() else 0.0

        def mean_bool(values: np.ndarray, mask: np.ndarray) -> float:
            return float(values[mask].mean()) if mask.any() else 0.0

        sign_match = residual.astype(np.float32) * delta.astype(np.float32) >= np.float32(0.0)
        cumulative_sign_match = state.cum_delta * delta.astype(np.float32) >= np.float32(0.0)
        for label, bin_mask in bins:
            prefix = f"agebin_{label}"
            missed_bin = missed & bin_mask
            extra_bin = extra & bin_mask
            selected_bin = selected & bin_mask
            out[f"{prefix}_ideal_mass_share"] = share(bin_mask, ideal, total_ideal_mass)
            out[f"{prefix}_missed_mass_share"] = share(bin_mask, missed, total_ideal_mass)
            out[f"{prefix}_selected_mass_share"] = share(bin_mask, selected, total_selected_mass)
            out[f"{prefix}_ideal_count_share"] = count_share(bin_mask, ideal)
            out[f"{prefix}_missed_count_share"] = count_share(bin_mask, missed)
            out[f"{prefix}_selected_count_share"] = count_share(bin_mask, selected)
            out[f"{prefix}_missed_current_over_mean_abs_median"] = median(current_over_mean_abs, missed_bin)
            out[f"{prefix}_missed_max_over_current_median"] = median(max_over_current, missed_bin)
            out[f"{prefix}_missed_residual_over_velocity_median"] = median(residual_over_velocity, missed_bin)
            out[f"{prefix}_missed_directional_eff_median"] = median(directional_efficiency, missed_bin)
            out[f"{prefix}_missed_flip_rate_median"] = median(flip_rate, missed_bin)
            out[f"{prefix}_missed_sign_match"] = mean_bool(sign_match, missed_bin)
            out[f"{prefix}_missed_cumulative_sign_match"] = mean_bool(cumulative_sign_match, missed_bin)
            out[f"{prefix}_extra_current_over_mean_abs_median"] = median(current_over_mean_abs, extra_bin)
            out[f"{prefix}_extra_residual_over_velocity_median"] = median(residual_over_velocity, extra_bin)
            out[f"{prefix}_selected_current_over_mean_abs_median"] = median(current_over_mean_abs, selected_bin)
        return out

    @staticmethod
    def _ar_residual_factors(*, age: np.ndarray, rho: float) -> Tuple[np.ndarray, np.ndarray]:
        rho = float(max(0.0, min(0.999, rho)))
        a = np.maximum(1.0, age.astype(np.float32))
        if rho <= 1.0e-6:
            # Independent signed updates: S = current + independent past.
            return np.maximum(a - 1.0, 0.0).astype(np.float32), np.ones_like(a, dtype=np.float32)
        r = np.float32(rho)
        one_minus = np.float32(1.0 - rho)
        r_pow_a = np.power(r, a).astype(np.float32)
        r_pow_am1 = np.power(r, np.maximum(a - 1.0, 0.0)).astype(np.float32)
        c = (np.float32(1.0) - r_pow_a) / one_minus
        s1 = r * (np.float32(1.0) - r_pow_am1) / one_minus
        s2 = r * (
            np.float32(1.0)
            - a * r_pow_am1
            + (a - np.float32(1.0)) * r_pow_a
        ) / (one_minus * one_minus)
        v_sum = a + np.float32(2.0) * (a * s1 - s2)
        b_factor = c * c
        a_factor = np.maximum(v_sum - b_factor, np.float32(0.0))
        return a_factor.astype(np.float32), b_factor.astype(np.float32)

    def _requested_delta_ema_spans(self) -> Tuple[int, ...]:
        spans = set()
        for name in self.policies:
            if name.startswith("h_age_delta_ema"):
                spans.add(max(1, int(name[len("h_age_delta_ema"):] or "16")))
            elif name.startswith("h_age_groupdelta_ema"):
                spans.add(max(1, int(name[len("h_age_groupdelta_ema"):] or "16")))
            elif name.startswith("h_ar_groupema"):
                span_text = name[len("h_ar_groupema"):].split("_rho", 1)[0] or "64"
                spans.add(max(1, int(span_text)))
        return tuple(sorted(spans))

    def _update_delta_ema(self, *, delta_sq: np.ndarray, spans: Tuple[int, ...]) -> None:
        if not spans:
            return
        group_ids = self._sample_group_ids
        n_groups = max(1, int(delta_sq.size // max(1, self._group_size)))
        if group_ids is not None:
            group_sum = np.bincount(group_ids, weights=delta_sq.astype(np.float64), minlength=n_groups).astype(np.float32)
            group_count = np.bincount(group_ids, minlength=n_groups).astype(np.float32)
            group_mean = np.zeros(n_groups, dtype=np.float32)
            np.divide(group_sum, np.maximum(group_count, np.float32(1.0)), out=group_mean, where=group_count > 0)
        else:
            group_mean = np.zeros(n_groups, dtype=np.float32)
        for span in spans:
            alpha = np.float32((float(span) - 1.0) / float(span))
            if span not in self._delta_ema:
                self._delta_ema[span] = delta_sq.copy()
            else:
                self._delta_ema[span] = alpha * self._delta_ema[span] + (np.float32(1.0) - alpha) * delta_sq
            if span not in self._group_delta_ema:
                self._group_delta_ema[span] = group_mean.copy()
            else:
                self._group_delta_ema[span] = alpha * self._group_delta_ema[span] + (np.float32(1.0) - alpha) * group_mean

    def _delta_diagnostics(self, *, delta_sq: np.ndarray, h_delta_score: np.ndarray, k: int) -> Dict[str, float]:
        diagnostics: Dict[str, float] = {
            "delta_sq_mean": float(delta_sq.mean()),
            "delta_sq_p99": float(np.percentile(delta_sq, 99.0)),
            "h_delta_mean": float(h_delta_score.mean()),
            "h_delta_p99": float(np.percentile(h_delta_score, 99.0)),
        }
        if self._prev_delta_sq is None or self._prev_h_delta_score is None:
            return diagnostics
        eps = np.float32(1.0e-30)
        log_cur = np.log(delta_sq + eps)
        log_prev = np.log(self._prev_delta_sq + eps)
        log_h_cur = np.log(h_delta_score + eps)
        log_h_prev = np.log(self._prev_h_delta_score + eps)
        diagnostics["delta_log_corr_prev"] = self._pearson(log_cur, log_prev)
        diagnostics["h_delta_log_corr_prev"] = self._pearson(log_h_cur, log_h_prev)
        cur_top = self._topk_mask(delta_sq, k)
        prev_top = self._topk_mask(self._prev_delta_sq, k)
        h_cur_top = self._topk_mask(h_delta_score, k)
        h_prev_top = self._topk_mask(self._prev_h_delta_score, k)
        diagnostics["delta_topk_overlap_prev"] = self._safe_div(float((cur_top & prev_top).sum()), float(k))
        diagnostics["h_delta_topk_overlap_prev"] = self._safe_div(float((h_cur_top & h_prev_top).sum()), float(k))
        return diagnostics

    def _passive_horizon_diagnostics(
        self,
        *,
        p: np.ndarray,
        h: np.ndarray,
        delta_sq: np.ndarray,
        k: int,
    ) -> Dict[str, float]:
        if not self._theta_history:
            return {}
        eps = np.float32(1.0e-30)
        log_horizon_parts = []
        log_residual_parts = []
        log_delta_parts = []
        horizon_means = []
        out: Dict[str, float] = {}
        for horizon in self._passive_horizons:
            if len(self._theta_history) < horizon:
                continue
            prev = self._theta_history[-horizon]
            residual_sq = (p - prev) ** 2
            score = h * residual_sq
            valid = residual_sq > eps
            if int(valid.sum()) < 128:
                continue
            mean_res = float(np.mean(residual_sq[valid]))
            top = self._topk_mask(score, k)
            top_mean_res = float(np.mean(residual_sq[top])) if top.any() else 0.0
            out[f"h{horizon}_residual_sq_mean"] = mean_res
            out[f"h{horizon}_topk_residual_sq_mean"] = top_mean_res
            horizon_means.append((float(horizon), mean_res))
            log_horizon_parts.append(np.full(int(valid.sum()), math.log(float(horizon)), dtype=np.float64))
            log_residual_parts.append(np.log(residual_sq[valid] + eps).astype(np.float64))
            log_delta_parts.append(np.log(delta_sq[valid] + eps).astype(np.float64))
        if len(horizon_means) >= 2:
            x = np.log(np.asarray([item[0] for item in horizon_means], dtype=np.float64))
            y = np.log(np.asarray([item[1] for item in horizon_means], dtype=np.float64) + 1.0e-30)
            _pred, r2, coef = self._fit_log_linear(target=y, features=[x])
            out["mean_residual_log_horizon_slope"] = float(coef[1])
            out["mean_residual_hurst"] = float(0.5 * coef[1])
            out["mean_residual_log_horizon_r2"] = float(r2)
        if log_horizon_parts:
            log_horizon = np.concatenate(log_horizon_parts)
            log_residual = np.concatenate(log_residual_parts)
            log_delta = np.concatenate(log_delta_parts)
            _pred, r2, coef = self._fit_log_linear(
                target=log_residual,
                features=[log_horizon, log_delta],
            )
            out["point_residual_horizon_delta_r2"] = float(r2)
            out["point_residual_horizon_coef"] = float(coef[1])
            out["point_residual_delta_coef"] = float(coef[2])
        return out

    def _residual_fit_diagnostics(
        self,
        *,
        step: int,
        p: np.ndarray,
        h: np.ndarray,
        v: np.ndarray,
        delta_sq: np.ndarray,
        k: int,
    ) -> Dict[str, float]:
        state = self._states.get("h_age") or self._states.get("h_age_q4") or self._states.get("oracle")
        if state is None:
            return {}
        residual_sq = (p - state.theta_cp) ** 2
        truth_score = h * residual_sq
        if not np.isfinite(truth_score).all() or float(truth_score.sum()) <= 0.0:
            return {}
        age = np.maximum(1, int(step) - state.last_written_step).astype(np.float32)
        eps = np.float32(1.0e-30)
        target = np.log(residual_sq + eps).astype(np.float64)
        feature_sets = {
            "age": [np.log(age + eps)],
            "age_theta": [np.log(age + eps), np.log(p * p + eps)],
            "age_v": [np.log(age + eps), np.log(v + eps)],
            "age_delta": [np.log(age + eps), np.log(delta_sq + eps)],
            "age_theta_v_delta": [
                np.log(age + eps),
                np.log(p * p + eps),
                np.log(v + eps),
                np.log(delta_sq + eps),
            ],
        }
        out: Dict[str, float] = {
            "ref_policy": 1.0 if state.name == "h_age" else 0.0,
            "residual_sq_mean": float(residual_sq.mean()),
            "residual_sq_p99": float(np.percentile(residual_sq, 99.0)),
        }
        hurst_slope, hurst_r2 = self._age_bin_log_slope(
            age=age,
            residual_sq=residual_sq,
            weights=truth_score,
        )
        out["age_bin_log_residual_slope"] = float(hurst_slope)
        out["age_bin_hurst"] = float(0.5 * hurst_slope)
        out["age_bin_log_residual_r2"] = float(hurst_r2)
        ideal = self._topk_mask(truth_score, k)
        ideal_mass = float(truth_score[ideal].sum())
        for name, features in feature_sets.items():
            pred_log, r2, coefs = self._fit_log_linear(target=target, features=features)
            pred_score = h * np.exp(pred_log).astype(np.float32)
            selected = self._topk_mask(pred_score, k)
            selected_mass = float(truth_score[selected].sum())
            out[f"{name}_r2"] = float(r2)
            out[f"{name}_mass_recall"] = self._safe_div(selected_mass, ideal_mass)
            for idx, coef in enumerate(coefs[:5]):
                out[f"{name}_coef{idx}"] = float(coef)
        weights = truth_score.astype(np.float64)
        pred_log, r2, coefs = self._fit_log_linear_weighted(
            target=target,
            features=feature_sets["age_delta"],
            weights=weights,
        )
        pred_score = h * np.exp(pred_log).astype(np.float32)
        selected = self._topk_mask(pred_score, k)
        out["age_delta_massw_r2"] = float(r2)
        out["age_delta_massw_mass_recall"] = self._safe_div(float(truth_score[selected].sum()), ideal_mass)
        out["age_delta_massw_coef1"] = float(coefs[1])
        out["age_delta_massw_coef2"] = float(coefs[2])
        return out

    @staticmethod
    def _age_bin_log_slope(
        *,
        age: np.ndarray,
        residual_sq: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[float, float]:
        valid = (age >= 1.0) & (residual_sq > np.float32(1.0e-30)) & np.isfinite(residual_sq)
        if int(valid.sum()) < 128:
            return 0.0, 0.0
        bucket = np.floor(np.log2(age[valid])).astype(np.int32)
        log_age_vals = []
        log_res_vals = []
        bin_weights = []
        for b in sorted(set(int(x) for x in bucket.tolist())):
            mask = bucket == b
            if int(mask.sum()) < 16:
                continue
            w = weights[valid][mask].astype(np.float64)
            if float(np.sum(w)) <= 0.0:
                w = np.ones_like(w, dtype=np.float64)
            log_age_vals.append(float(np.mean(np.log(age[valid][mask]))))
            log_res_vals.append(float(np.sum(w * np.log(residual_sq[valid][mask] + np.float32(1.0e-30))) / max(float(np.sum(w)), 1.0e-30)))
            bin_weights.append(float(np.sum(w)))
        if len(log_age_vals) < 2:
            return 0.0, 0.0
        x = np.asarray(log_age_vals, dtype=np.float64)
        y = np.asarray(log_res_vals, dtype=np.float64)
        w = np.asarray(bin_weights, dtype=np.float64)
        pred, r2, coef = PolicyShadowTelemetry._fit_log_linear_weighted(
            target=y,
            features=[x],
            weights=w,
        )
        return float(coef[1]), float(r2)

    @staticmethod
    def _fit_log_linear(*, target: np.ndarray, features: List[np.ndarray]) -> Tuple[np.ndarray, float, np.ndarray]:
        y = target.astype(np.float64)
        cols = [np.ones_like(y, dtype=np.float64)]
        for feat in features:
            x = feat.astype(np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            cols.append(x)
        xmat = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(xmat, y, rcond=None)
        pred = xmat @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        return pred.astype(np.float32), float(r2), coef.astype(np.float32)

    @staticmethod
    def _fit_log_linear_weighted(
        *,
        target: np.ndarray,
        features: List[np.ndarray],
        weights: np.ndarray,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        y = target.astype(np.float64)
        w = np.maximum(weights.astype(np.float64), 0.0)
        if float(np.sum(w)) <= 0.0:
            w = np.ones_like(y, dtype=np.float64)
        cols = [np.ones_like(y, dtype=np.float64)]
        for feat in features:
            x = feat.astype(np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            cols.append(x)
        xmat = np.stack(cols, axis=1)
        sqrt_w = np.sqrt(w / max(float(np.mean(w)), 1.0e-30))
        coef, *_ = np.linalg.lstsq(xmat * sqrt_w[:, None], y * sqrt_w, rcond=None)
        pred = xmat @ coef
        y_mean = float(np.sum(w * y) / max(float(np.sum(w)), 1.0e-30))
        ss_res = float(np.sum(w * (y - pred) ** 2))
        ss_tot = float(np.sum(w * (y - y_mean) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        return pred.astype(np.float32), float(r2), coef.astype(np.float32)

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        av = a - a.mean()
        bv = b - b.mean()
        den = float(np.sqrt(np.sum(av * av) * np.sum(bv * bv)))
        if den <= 0.0 or not math.isfinite(den):
            return 0.0
        return float(np.sum(av * bv) / den)

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return float(num) / float(den) if float(den) != 0.0 else 0.0

    def _append_record(self, record: Dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = ["PolicyShadowTelemetry"]
