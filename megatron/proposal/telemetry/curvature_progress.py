"""Experiment-only telemetry for signed-curvature progress scoring.

This module is intentionally isolated from production selection.  It samples
whole element groups, records signed FD-Hessian values before the production
refresh takes ``abs()``, and runs independent CPU shadow policies on the
sampled elements.
"""

from __future__ import annotations

import json
import math
import os
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
class _ShadowState:
    name: str
    theta_cp: np.ndarray
    last_written_step: np.ndarray
    age_code: np.ndarray


class CurvatureProgressTelemetry:
    """Sampled telemetry for TODO1-3 in the signed-H score investigation."""

    def __init__(
        self,
        *,
        log_path: str,
        sample_groups: int,
        every: int,
        seed: int,
    ) -> None:
        self.log_path = Path(log_path).expanduser() if log_path else None
        self.sample_groups = max(1, int(sample_groups))
        self.every = max(1, int(every))
        self.seed = int(seed)
        self._initialized = False
        self._group_size = 1
        self._sources: List[_SampleSource] = []
        self._sample_group_ids: Optional[np.ndarray] = None
        self._sample_element_ids: Optional[np.ndarray] = None
        self._sample_by_name: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._signed_h: Optional[np.ndarray] = None
        self._prev_signed_h: Optional[np.ndarray] = None
        self._states: Dict[str, _ShadowState] = {}
        self._last_refresh_step: Optional[int] = None

    @classmethod
    def from_env(cls) -> "CurvatureProgressTelemetry":
        return cls(
            log_path=os.environ.get("PROPOSAL_CURVATURE_PROGRESS_LOG", ""),
            sample_groups=int(os.environ.get("PROPOSAL_CURVATURE_PROGRESS_SAMPLE_GROUPS", "2048") or "2048"),
            every=int(os.environ.get("PROPOSAL_CURVATURE_PROGRESS_EVERY", "1") or "1"),
            seed=int(os.environ.get("PROPOSAL_CURVATURE_PROGRESS_SEED", "2024") or "2024"),
        )

    @property
    def enabled(self) -> bool:
        return self.log_path is not None

    def record_signed_h_refresh(
        self,
        *,
        step: int,
        groups: Iterable[Dict[str, Any]],
        group_size: int,
        names: Iterable[str],
        signed_h_views: Iterable[torch.Tensor],
        n_anchor_batches: int,
    ) -> None:
        if not self.enabled:
            return
        groups_list = list(groups)
        if not self._initialized:
            self._initialize(groups_list=groups_list, group_size=int(group_size))
        if not self._sample_by_name or self._sample_element_ids is None:
            return
        n = int(self._sample_element_ids.shape[0])
        signed = np.zeros(n, dtype=np.float32)
        filled = np.zeros(n, dtype=bool)
        view_by_name = {
            str(name): view.detach().reshape(-1)
            for name, view in zip(names, signed_h_views)
            if view is not None and view.numel() > 0
        }
        for name, (sample_idx, local_off) in self._sample_by_name.items():
            view = view_by_name.get(name)
            if view is None or sample_idx.size <= 0:
                continue
            offsets = torch.as_tensor(local_off, dtype=torch.long, device=view.device)
            vals = torch.index_select(view, 0, offsets).detach().float().cpu().numpy()
            signed[sample_idx] = vals.astype(np.float32, copy=False)
            filled[sample_idx] = True
        if int(filled.sum()) <= 0:
            return
        self._prev_signed_h = None if self._signed_h is None else self._signed_h.copy()
        self._signed_h = signed
        self._last_refresh_step = int(step)
        v = self._gather_v(groups_list=groups_list)
        record = {
            "mode": "curvature_progress_h_refresh",
            "step": int(step),
            "sample_elements": int(n),
            "sample_groups": int(n // max(1, self._group_size)),
            "n_anchor_batches": int(n_anchor_batches),
            **self._signed_h_diagnostics(signed=signed, filled=filled, v=v),
        }
        self._append(record)

    def record_iteration(
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
        beta1: float = 0.9,
    ) -> None:
        if not self.enabled or int(step) % self.every != 0:
            return
        groups_list = list(groups)
        if not self._initialized:
            self._initialize(groups_list=groups_list, group_size=int(group_size))
        if not self._sources or self._sample_element_ids is None:
            return
        p, m, v = self._gather_pmv(groups_list=groups_list)
        self._init_states(p=p)
        h_abs_lut = self._lookup_h_abs(
            v=v,
            h_lut_values=h_lut_values,
            h_lut_log_v_min=h_lut_log_v_min,
            h_lut_inv_log_v_width=h_lut_inv_log_v_width,
        )
        signed_h_raw = self._signed_h
        if signed_h_raw is None or signed_h_raw.shape[0] != p.shape[0]:
            signed_h = h_abs_lut.copy()
            signed_available = False
        else:
            signed_h = signed_h_raw.astype(np.float32, copy=False)
            signed_available = True
        denom = np.sqrt(np.maximum(v, 0.0), dtype=np.float32) * float(sqrt_bc2_inv) + float(eps)
        # Adam descent mutates theta by -lr*m/denom.  Production only needs
        # delta^2, but the progress score is signed, so telemetry must keep
        # the actual theta displacement direction.
        delta = -float(lr_over_bc1) * m / denom
        delta_sq = delta * delta
        n = int(p.shape[0])
        k = max(1, min(n, int(math.ceil(float(topk_ratio) * float(n)))))
        rho_stats = self._rho_diagnostics(
            h_abs=h_abs_lut,
            v=v,
            lr_over_bc1=float(lr_over_bc1),
            sqrt_bc2_inv=float(sqrt_bc2_inv),
            eps=float(eps),
            beta1=float(beta1),
        )
        policies = []
        for state in self._states.values():
            e_true = p - state.theta_cp
            risk_truth = h_abs_lut * (e_true * e_true)
            progress_truth_signed = -m * e_true + np.float32(0.5) * signed_h * (e_true * e_true)
            progress_truth_pos = np.maximum(progress_truth_signed, np.float32(0.0))
            progress_truth_abs = np.abs(progress_truth_signed)
            progress_truth_abs_h_pos = np.maximum(
                -m * e_true + np.float32(0.5) * h_abs_lut * (e_true * e_true),
                np.float32(0.0),
            )

            e_hat = self._curvature_residual_proxy(
                delta=delta,
                v=v,
                h_abs=h_abs_lut,
                age=np.maximum(1, int(step) - state.last_written_step).astype(np.float32),
                lr_over_bc1=float(lr_over_bc1),
                sqrt_bc2_inv=float(sqrt_bc2_inv),
                eps=float(eps),
                beta1=float(beta1),
            )
            e_hat_age2 = self._curvature_residual_proxy_age2bit(
                delta=delta,
                v=v,
                h_abs=h_abs_lut,
                age_code=state.age_code,
                lr_over_bc1=float(lr_over_bc1),
                sqrt_bc2_inv=float(sqrt_bc2_inv),
                eps=float(eps),
                beta1=float(beta1),
            )
            current_score = self._current_age2bit_score(
                h_abs=h_abs_lut,
                delta_sq=delta_sq,
                age_code=state.age_code,
                step=int(step),
                element_ids=self._sample_element_ids,
            )
            curv_score = h_abs_lut * (e_hat * e_hat)
            curv_age2_score = h_abs_lut * (e_hat_age2 * e_hat_age2)
            progress_score = np.maximum(
                -m * e_hat + np.float32(0.5) * signed_h * (e_hat * e_hat),
                np.float32(0.0),
            )
            progress_abs_h_score = np.maximum(
                -m * e_hat + np.float32(0.5) * h_abs_lut * (e_hat * e_hat),
                np.float32(0.0),
            )
            progress_abs_h_age2_score = np.maximum(
                -m * e_hat_age2 + np.float32(0.5) * h_abs_lut * (e_hat_age2 * e_hat_age2),
                np.float32(0.0),
            )
            progress_abs_score = np.abs(
                -m * e_hat + np.float32(0.5) * signed_h * (e_hat * e_hat),
            )

            selected_by_policy = {
                "oracle_risk": self._topk_mask(risk_truth, k),
                "oracle_progress_pos": self._topk_mask(progress_truth_pos, k),
                "oracle_abs_h_progress_pos": self._topk_mask(progress_truth_abs_h_pos, k),
                "current_age2bit_delta": self._topk_mask(current_score, k),
                "curvature_proxy": self._topk_mask(curv_score, k),
                "curvature_proxy_age2bit": self._topk_mask(curv_age2_score, k),
                "progress_proxy_pos": self._topk_mask(progress_score, k),
                "progress_proxy_abs_h_pos": self._topk_mask(progress_abs_h_score, k),
                "progress_proxy_abs_h_pos_age2bit": self._topk_mask(progress_abs_h_age2_score, k),
                "progress_proxy_abs": self._topk_mask(progress_abs_score, k),
            }
            row = {
                "state": state.name,
                "signed_h_available": bool(signed_available),
                "risk_pre_error": float(risk_truth.sum()),
                "progress_pos_pre_error": float(progress_truth_pos.sum()),
                "progress_abs_pre_error": float(progress_truth_abs.sum()),
                "progress_abs_h_pos_pre_error": float(progress_truth_abs_h_pos.sum()),
                **self._proxy_quality(
                    e_true=e_true,
                    e_hat=e_hat,
                    risk_truth=risk_truth,
                    progress_truth_pos=progress_truth_pos,
                    progress_truth_abs=progress_truth_abs,
                    progress_truth_abs_h_pos=progress_truth_abs_h_pos,
                    scores={
                        "current_age2bit_delta": current_score,
                        "curvature_proxy": curv_score,
                        "curvature_proxy_age2bit": curv_age2_score,
                        "progress_proxy_pos": progress_score,
                        "progress_proxy_abs_h_pos": progress_abs_h_score,
                        "progress_proxy_abs_h_pos_age2bit": progress_abs_h_age2_score,
                        "progress_proxy_abs": progress_abs_score,
                    },
                    selected_by_policy=selected_by_policy,
                    k=k,
                ),
                **self._value_sanity(
                    m=m,
                    e_true=e_true,
                    e_hat=e_hat,
                    signed_h=signed_h,
                    progress_raw=(-m * e_hat + np.float32(0.5) * signed_h * (e_hat * e_hat)),
                ),
                **self._selection_failure_diagnostics(
                    m=m,
                    delta=delta,
                    age=np.maximum(1, int(step) - state.last_written_step).astype(np.float32),
                    e_true=e_true,
                    e_hat=e_hat,
                    h_abs=h_abs_lut,
                    signed_h=signed_h,
                    risk_truth=risk_truth,
                    progress_truth_pos=progress_truth_pos,
                    progress_truth_abs_h_pos=progress_truth_abs_h_pos,
                    selected_by_policy=selected_by_policy,
                    k=k,
                ),
            }
            selected = selected_by_policy.get(state.name)
            if selected is None:
                selected = selected_by_policy["progress_proxy_pos"]
            self._update_state(
                state=state,
                step=int(step),
                selected=selected,
                p=p,
                element_ids=self._sample_element_ids,
            )
            risk_post = h_abs_lut * ((p - state.theta_cp) ** 2)
            progress_post_raw = -m * (p - state.theta_cp) + np.float32(0.5) * signed_h * ((p - state.theta_cp) ** 2)
            progress_post_abs_h_raw = -m * (p - state.theta_cp) + np.float32(0.5) * h_abs_lut * ((p - state.theta_cp) ** 2)
            row["risk_post_error"] = float(risk_post.sum())
            row["progress_pos_post_error"] = float(np.maximum(progress_post_raw, np.float32(0.0)).sum())
            row["progress_abs_post_error"] = float(np.abs(progress_post_raw).sum())
            row["progress_abs_h_pos_post_error"] = float(np.maximum(progress_post_abs_h_raw, np.float32(0.0)).sum())
            policies.append(row)
        self._append({
            "mode": "curvature_progress_iter",
            "step": int(step),
            "last_refresh_step": self._last_refresh_step,
            "sample_elements": int(n),
            "sample_groups": int(n // max(1, int(group_size))),
            "sample_k": int(k),
            "topk_ratio": float(topk_ratio),
            **self._h_lut_diagnostics(h_abs_lut=h_abs_lut, v=v),
            **rho_stats,
            "policies": policies,
        })

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
        sampled_groups = np.sort(rng.choice(total_groups, size=n_sample_groups, replace=False).astype(np.int64))
        sources: List[_SampleSource] = []
        sample_group_ids: List[int] = []
        sample_element_ids: List[int] = []
        by_name: Dict[str, Tuple[List[int], List[int]]] = {}
        sample_index_base = 0
        sample_group_base = 0
        for group_idx, start, end in ranges:
            lo = int(np.searchsorted(sampled_groups, start, side="left"))
            hi = int(np.searchsorted(sampled_groups, end, side="left"))
            if hi <= lo:
                continue
            grp = groups_list[group_idx]
            global_groups = sampled_groups[lo:hi]
            local_groups = global_groups - int(start)
            lane = np.arange(int(group_size), dtype=np.int64)
            local_offsets_np = (local_groups[:, None] * int(group_size) + lane[None, :]).reshape(-1)
            global_element_ids_np = (global_groups[:, None] * int(group_size) + lane[None, :]).reshape(-1)
            sample_indices = np.arange(sample_index_base, sample_index_base + local_offsets_np.size, dtype=np.int64)
            sample_index_base += int(local_offsets_np.size)
            sources.append(_SampleSource(
                group_index=int(group_idx),
                local_offsets=torch.as_tensor(local_offsets_np, dtype=torch.long, device=grp["p_flat"].device),
                sample_indices=sample_indices,
            ))
            self._append_name_mapping(
                grp=grp,
                local_offsets=local_offsets_np,
                sample_indices=sample_indices,
                by_name=by_name,
            )
            for local_sample_group_id in range(sample_group_base, sample_group_base + len(global_groups)):
                sample_group_ids.extend([int(local_sample_group_id)] * int(group_size))
            sample_element_ids.extend(int(x) for x in global_element_ids_np.tolist())
            sample_group_base += int(len(global_groups))
        self._group_size = int(group_size)
        self._sources = sources
        self._sample_group_ids = np.asarray(sample_group_ids, dtype=np.int64)
        self._sample_element_ids = np.asarray(sample_element_ids, dtype=np.uint64)
        self._sample_by_name = {
            name: (np.asarray(idx, dtype=np.int64), np.asarray(off, dtype=np.int64))
            for name, (idx, off) in by_name.items()
        }
        self._initialized = True

    @staticmethod
    def _append_name_mapping(
        *,
        grp: Dict[str, Any],
        local_offsets: np.ndarray,
        sample_indices: np.ndarray,
        by_name: Dict[str, Tuple[List[int], List[int]]],
    ) -> None:
        names = list(grp.get("names") or [])
        sizes = [int(x) for x in (grp.get("sizes") or [])]
        if not names or not sizes:
            return
        ends = np.cumsum(np.asarray(sizes, dtype=np.int64))
        name_idx = np.searchsorted(ends, local_offsets, side="right")
        starts = np.concatenate(([0], ends[:-1]))
        for sample_i, local_i, ni in zip(sample_indices, local_offsets, name_idx):
            if int(ni) < 0 or int(ni) >= len(names):
                continue
            name = str(names[int(ni)])
            off = int(local_i) - int(starts[int(ni)])
            idx_list, off_list = by_name.setdefault(name, ([], []))
            idx_list.append(int(sample_i))
            off_list.append(int(off))

    def _init_states(self, *, p: np.ndarray) -> None:
        if self._states:
            return
        n = int(p.shape[0])
        for name in (
            "oracle_risk",
            "oracle_progress_pos",
            "oracle_abs_h_progress_pos",
            "current_age2bit_delta",
            "curvature_proxy",
            "curvature_proxy_age2bit",
            "progress_proxy_pos",
            "progress_proxy_abs_h_pos",
            "progress_proxy_abs_h_pos_age2bit",
            "progress_proxy_abs",
        ):
            self._states[name] = _ShadowState(
                name=name,
                theta_cp=p.copy(),
                last_written_step=np.zeros(n, dtype=np.int32),
                age_code=np.zeros(n, dtype=np.uint8),
            )

    def _gather_pmv(self, *, groups_list: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return p, m, v

    def _gather_v(self, *, groups_list: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        if not self._sources:
            return None
        n = sum(int(src.sample_indices.size) for src in self._sources)
        v = np.empty(n, dtype=np.float32)
        for src in self._sources:
            grp = groups_list[src.group_index]
            v[src.sample_indices] = torch.index_select(
                grp["v_flat"], 0, src.local_offsets,
            ).detach().float().cpu().numpy()
        return v

    @staticmethod
    def _lookup_h_abs(
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

    @staticmethod
    def _curvature_residual_proxy(
        *,
        delta: np.ndarray,
        v: np.ndarray,
        h_abs: np.ndarray,
        age: np.ndarray,
        lr_over_bc1: float,
        sqrt_bc2_inv: float,
        eps: float,
        beta1: float,
    ) -> np.ndarray:
        rho = CurvatureProgressTelemetry._adam_curvature_contraction(
            h_abs=h_abs,
            v=v,
            lr_over_bc1=float(lr_over_bc1),
            sqrt_bc2_inv=float(sqrt_bc2_inv),
            eps=float(eps),
            beta1=float(beta1),
        )
        horizon = np.minimum(age, np.float32(64.0))
        near_one = np.abs(np.float32(1.0) - rho) < np.float32(1.0e-4)
        inv_rho = np.float32(1.0) / rho
        geom_pow = np.power(inv_rho.astype(np.float32), horizon.astype(np.float32)).astype(np.float32)
        geom = (geom_pow - np.float32(1.0)) / (inv_rho - np.float32(1.0) + np.float32(1.0e-20))
        geom = np.where(near_one, horizon, geom).astype(np.float32)
        return delta * geom

    @staticmethod
    def _curvature_residual_proxy_age2bit(
        *,
        delta: np.ndarray,
        v: np.ndarray,
        h_abs: np.ndarray,
        age_code: np.ndarray,
        lr_over_bc1: float,
        sqrt_bc2_inv: float,
        eps: float,
        beta1: float,
    ) -> np.ndarray:
        rho = CurvatureProgressTelemetry._adam_curvature_contraction(
            h_abs=h_abs,
            v=v,
            lr_over_bc1=float(lr_over_bc1),
            sqrt_bc2_inv=float(sqrt_bc2_inv),
            eps=float(eps),
            beta1=float(beta1),
        )
        horizon = np.choose(age_code, (1.0, 4.0, 16.0, 64.0)).astype(np.float32)
        near_one = np.abs(np.float32(1.0) - rho) < np.float32(1.0e-4)
        inv_rho = np.float32(1.0) / rho
        geom_pow = np.power(inv_rho.astype(np.float32), horizon.astype(np.float32)).astype(np.float32)
        geom = (geom_pow - np.float32(1.0)) / (inv_rho - np.float32(1.0) + np.float32(1.0e-20))
        geom = np.where(near_one, horizon, geom).astype(np.float32)
        return delta * geom

    @staticmethod
    def _adam_curvature_contraction(
        *,
        h_abs: np.ndarray,
        v: np.ndarray,
        lr_over_bc1: float,
        sqrt_bc2_inv: float,
        eps: float,
        beta1: float,
    ) -> np.ndarray:
        denom = (
            np.sqrt(np.maximum(v, np.float32(0.0)), dtype=np.float32)
            * np.float32(float(sqrt_bc2_inv))
            + np.float32(float(eps))
        )
        mu = np.float32(abs(lr_over_bc1)) * h_abs / np.maximum(denom, np.float32(1.0e-30))
        beta = np.float32(min(max(float(beta1), 0.0), 0.9999))
        trace = np.float32(1.0) + beta - (np.float32(1.0) - beta) * mu
        disc = trace * trace - np.float32(4.0) * beta
        sqrt_disc = np.sqrt(np.maximum(disc, np.float32(0.0)), dtype=np.float32)
        real_a = np.float32(0.5) * (trace + sqrt_disc)
        real_b = np.float32(0.5) * (trace - sqrt_disc)
        real_radius = np.maximum(np.abs(real_a), np.abs(real_b))
        oscillatory_mag = np.float32(math.sqrt(float(beta)))
        rho = np.where(disc >= 0.0, real_radius, oscillatory_mag)
        return np.clip(rho, np.float32(0.9), np.float32(1.0)).astype(np.float32)

    @staticmethod
    def _rho_diagnostics(
        *,
        h_abs: np.ndarray,
        v: np.ndarray,
        lr_over_bc1: float,
        sqrt_bc2_inv: float,
        eps: float,
        beta1: float,
    ) -> Dict[str, float]:
        denom = (
            np.sqrt(np.maximum(v, np.float32(0.0)), dtype=np.float32)
            * np.float32(float(sqrt_bc2_inv))
            + np.float32(float(eps))
        )
        mu = np.float32(abs(lr_over_bc1)) * h_abs / np.maximum(denom, np.float32(1.0e-30))
        rho_sgd = np.float32(1.0) - mu
        rho = CurvatureProgressTelemetry._adam_curvature_contraction(
            h_abs=h_abs,
            v=v,
            lr_over_bc1=float(lr_over_bc1),
            sqrt_bc2_inv=float(sqrt_bc2_inv),
            eps=float(eps),
            beta1=float(beta1),
        )
        beta = np.float32(min(max(float(beta1), 0.0), 0.9999))
        trace = np.float32(1.0) + beta - (np.float32(1.0) - beta) * mu
        disc = trace * trace - np.float32(4.0) * beta
        valid = np.isfinite(mu) & np.isfinite(rho)
        if int(valid.sum()) <= 0:
            return {}
        m = mu[valid]
        r_sgd = rho_sgd[valid]
        r = rho[valid]
        d = disc[valid]
        return {
            "rho_mu_min": float(np.min(m)),
            "rho_mu_p01": float(np.percentile(m, 1.0)),
            "rho_mu_p50": float(np.percentile(m, 50.0)),
            "rho_mu_p99": float(np.percentile(m, 99.0)),
            "rho_mu_max": float(np.max(m)),
            "rho_sgd_raw_min": float(np.min(r_sgd)),
            "rho_sgd_raw_p50": float(np.percentile(r_sgd, 50.0)),
            "rho_sgd_raw_p99": float(np.percentile(r_sgd, 99.0)),
            "rho_adam_p01": float(np.percentile(r, 1.0)),
            "rho_adam_p50": float(np.percentile(r, 50.0)),
            "rho_adam_p99": float(np.percentile(r, 99.0)),
            "rho_adam_clip_low_ratio": float((r <= np.float32(0.900001)).mean()),
            "rho_adam_near_one_1e_3_ratio": float((r >= np.float32(0.999)).mean()),
            "rho_adam_complex_mode_ratio": float((d < 0.0).mean()),
            "rho_adam_nonpositive_real_ratio": float(((d >= 0.0) & (trace <= 0.0)).mean()),
        }

    @staticmethod
    def _current_age2bit_score(
        *,
        h_abs: np.ndarray,
        delta_sq: np.ndarray,
        age_code: np.ndarray,
        step: int,
        element_ids: Optional[np.ndarray],
    ) -> np.ndarray:
        age_sq = np.choose(age_code, (1.0, 16.0, 256.0, 4096.0)).astype(np.float32)
        score = h_abs * age_sq * delta_sq
        if element_ids is None:
            element_ids = np.arange(age_code.shape[0], dtype=np.uint64)
        forced = (age_code == 3) & CurvatureProgressTelemetry._age2_phase_hit_np(
            element_ids=element_ids,
            step=int(step),
            period=128,
            salt=0x51ED270B,
        )
        if forced.any():
            high = float(np.max(score[np.isfinite(score)])) if np.isfinite(score).any() else 1.0
            score = score.copy()
            score[forced] = np.float32(high * 1.0e6 + 1.0)
        return score.astype(np.float32)

    @staticmethod
    def _age2_phase_hit_np(
        *,
        element_ids: np.ndarray,
        step: int,
        period: int,
        salt: int,
    ) -> np.ndarray:
        x = element_ids.astype(np.uint32, copy=False) ^ np.uint32(int(salt))
        x = (x ^ (x >> np.uint32(16))) * np.uint32(0x7FEB352D)
        x = (x ^ (x >> np.uint32(15))) * np.uint32(0x846CA68B)
        x = x ^ (x >> np.uint32(16))
        phase = (x % np.uint32(int(period))).astype(np.int32)
        return ((np.int32(int(step)) + phase) % np.int32(int(period))) == 0

    def _signed_h_diagnostics(
        self,
        *,
        signed: np.ndarray,
        filled: np.ndarray,
        v: Optional[np.ndarray],
    ) -> Dict[str, float]:
        valid = filled & np.isfinite(signed)
        out: Dict[str, float] = {
            "valid_ratio": float(valid.mean()) if valid.size else 0.0,
        }
        if int(valid.sum()) <= 0:
            return out
        s = signed[valid]
        out.update({
            "signed_h_mean": float(s.mean()),
            "signed_h_abs_mean": float(np.abs(s).mean()),
            "signed_h_pos_ratio": float((s > 0.0).mean()),
            "signed_h_neg_ratio": float((s < 0.0).mean()),
            "signed_h_zero_ratio": float((s == 0.0).mean()),
        })
        if self._prev_signed_h is not None and self._prev_signed_h.shape == signed.shape:
            prev = self._prev_signed_h[valid]
            out["signed_h_sign_flip_vs_prev"] = float((np.sign(prev) != np.sign(s)).mean())
            out["signed_h_abs_log_corr_prev"] = self._pearson(np.log(np.abs(prev) + 1.0e-30), np.log(np.abs(s) + 1.0e-30))
            out["signed_h_spearman_prev"] = self._spearman(prev, s)
            out.update(self._conditional_sign_stability(
                current=signed,
                previous=self._prev_signed_h,
                valid=valid,
                prefix="signed_h",
            ))
            out.update({
                f"group_{k}": val
                for k, val in self._group_refresh_stability(
                    current=signed,
                    previous=self._prev_signed_h,
                    valid=valid,
                ).items()
            })
        if v is not None and v.shape == signed.shape:
            vv = v[valid]
            out["raw_signed_h_vs_logv_spearman"] = self._spearman(s, np.log(vv + 1.0e-30))
            out["raw_abs_signed_h_vs_logv_spearman"] = self._spearman(np.abs(s), np.log(vv + 1.0e-30))
            out.update({
                f"raw_abs_signed_h_group_{k}": val
                for k, val in self._group_mean_spearman(
                    values=np.abs(signed),
                    other=v,
                    valid=valid,
                ).items()
            })
        out.update(self._row_locality(signed=signed, valid=valid))
        return out

    def _h_lut_diagnostics(self, *, h_abs_lut: np.ndarray, v: np.ndarray) -> Dict[str, float]:
        valid = np.isfinite(h_abs_lut) & np.isfinite(v) & (v > 0.0)
        out: Dict[str, float] = {}
        if int(valid.sum()) <= 1:
            return out
        out["h_lut_abs_vs_logv_spearman"] = self._spearman(
            h_abs_lut[valid],
            np.log(v[valid] + 1.0e-30),
        )
        out.update({
            f"h_lut_group_{k}": val
            for k, val in self._group_mean_spearman(
                values=h_abs_lut,
                other=v,
                valid=valid,
            ).items()
        })
        out.update({
            f"h_lut_{k}": v
            for k, v in self._row_abs_diagnostics(values=h_abs_lut, valid=valid).items()
        })
        return out

    def _group_mean_spearman(
        self,
        *,
        values: np.ndarray,
        other: np.ndarray,
        valid: np.ndarray,
    ) -> Dict[str, float]:
        """Compare group-mean value ranks against group-mean v ranks.

        This is experiment-only because the old 5/14 diagnostics discussed
        group-level H, while the signed-H refresh telemetry samples raw
        elements.  Keeping both axes prevents mixing those definitions.
        """
        g = int(self._group_size)
        n_groups = int(values.shape[0]) // max(1, g)
        if n_groups <= 1 or other.shape != values.shape:
            return {}
        v0 = values[: n_groups * g].reshape(n_groups, g)
        o0 = other[: n_groups * g].reshape(n_groups, g)
        m0 = valid[: n_groups * g].reshape(n_groups, g)
        row_valid = m0.sum(axis=1) > 0
        if int(row_valid.sum()) <= 1:
            return {}
        cnt = np.maximum(m0.sum(axis=1), 1)
        value_mean = np.sum(v0 * m0, axis=1) / cnt
        other_mean = np.sum(o0 * m0, axis=1) / cnt
        log_other_mean = np.sum(np.log(o0 + 1.0e-30) * m0, axis=1) / cnt
        value_mean = value_mean[row_valid]
        other_mean = other_mean[row_valid]
        log_other_mean = log_other_mean[row_valid]
        return {
            "mean_vs_v_mean_spearman": self._spearman(value_mean, other_mean),
            "mean_vs_logv_mean_spearman": self._spearman(value_mean, log_other_mean),
        }

    def _row_abs_diagnostics(self, *, values: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
        g = int(self._group_size)
        n_groups = int(values.shape[0]) // max(1, g)
        if n_groups <= 0:
            return {}
        x = np.abs(values[: n_groups * g]).reshape(n_groups, g)
        m = valid[: n_groups * g].reshape(n_groups, g)
        valid_rows = m.sum(axis=1) >= max(2, min(g, 2))
        if not valid_rows.any():
            return {}
        ax = x[valid_rows]
        mx = m[valid_rows]
        cnt = mx.sum(axis=1)
        adjacent_valid = mx[:, 1:] & mx[:, :-1]
        log_adj_lo = np.log(ax[:, :-1] + 1.0e-30)[adjacent_valid]
        log_adj_hi = np.log(ax[:, 1:] + 1.0e-30)[adjacent_valid]
        row_mean = np.sum(ax * mx, axis=1) / np.maximum(cnt, 1)
        row_var = np.sum(((ax - row_mean[:, None]) ** 2) * mx, axis=1) / np.maximum(cnt, 1)
        row_cv = np.sqrt(row_var) / np.maximum(row_mean, 1.0e-30)
        return {
            "row_abs_cv_mean": float(row_cv.mean()),
            "row_abs_adjacent_log_corr": self._pearson(log_adj_lo, log_adj_hi),
            "row_abs_adjacent_log_absdiff_median": float(np.median(np.abs(log_adj_hi - log_adj_lo))) if log_adj_lo.size else 0.0,
        }

    def _row_locality(self, *, signed: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
        if self._sample_group_ids is None:
            return {}
        g = int(self._group_size)
        n_groups = int(signed.shape[0]) // max(1, g)
        if n_groups <= 0:
            return {}
        x = signed[: n_groups * g].reshape(n_groups, g)
        m = valid[: n_groups * g].reshape(n_groups, g)
        signs = np.sign(x)
        absx = np.abs(x)
        valid_rows = m.sum(axis=1) >= max(2, min(g, 2))
        if not valid_rows.any():
            return {}
        sx = signs[valid_rows]
        ax = absx[valid_rows]
        mx = m[valid_rows]
        pos = ((sx > 0) & mx).sum(axis=1)
        neg = ((sx < 0) & mx).sum(axis=1)
        cnt = mx.sum(axis=1)
        majority = np.maximum(pos, neg) / np.maximum(cnt, 1)
        adjacent_valid = mx[:, 1:] & mx[:, :-1]
        adjacent_same = (sx[:, 1:] == sx[:, :-1]) & adjacent_valid
        log_adj_lo = np.log(ax[:, :-1] + 1.0e-30)[adjacent_valid]
        log_adj_hi = np.log(ax[:, 1:] + 1.0e-30)[adjacent_valid]
        row_mean = np.sum(ax * mx, axis=1) / np.maximum(cnt, 1)
        row_var = np.sum(((ax - row_mean[:, None]) ** 2) * mx, axis=1) / np.maximum(cnt, 1)
        row_cv = np.sqrt(row_var) / np.maximum(row_mean, 1.0e-30)
        return {
            "row_sign_all_same_ratio": float((majority >= 1.0).mean()),
            "row_sign_majority_mean": float(majority.mean()),
            "row_adjacent_sign_agreement": float(adjacent_same.sum() / max(1, int(adjacent_valid.sum()))),
            "row_signed_adjacent_corr": self._pearson(x[:, :-1][adjacent_valid], x[:, 1:][adjacent_valid]),
            "row_signed_adjacent_spearman": self._spearman(x[:, :-1][adjacent_valid], x[:, 1:][adjacent_valid]),
            "row_abs_cv_mean": float(row_cv.mean()),
            "row_abs_adjacent_log_corr": self._pearson(log_adj_lo, log_adj_hi),
            "row_abs_adjacent_log_absdiff_median": float(np.median(np.abs(log_adj_hi - log_adj_lo))) if log_adj_lo.size else 0.0,
        }

    def _conditional_sign_stability(
        self,
        *,
        current: np.ndarray,
        previous: np.ndarray,
        valid: np.ndarray,
        prefix: str,
    ) -> Dict[str, float]:
        pair_valid = valid & np.isfinite(current) & np.isfinite(previous)
        if int(pair_valid.sum()) <= 0:
            return {}
        c = current[pair_valid].astype(np.float32, copy=False)
        p = previous[pair_valid].astype(np.float32, copy=False)
        c_sign = np.sign(c)
        p_sign = np.sign(p)
        nonzero = (c_sign != 0.0) & (p_sign != 0.0)
        if int(nonzero.sum()) <= 0:
            return {}
        c = c[nonzero]
        p = p[nonzero]
        c_abs = np.abs(c)
        p_abs = np.abs(p)
        pair_abs = np.maximum(c_abs, p_abs)
        flip = (np.sign(c) != np.sign(p))
        out: Dict[str, float] = {
            f"{prefix}_sign_flip_pair_abs_weighted": self._safe_div(
                float((pair_abs * flip.astype(np.float32)).sum()),
                float(pair_abs.sum()),
            ),
        }
        for pct in (1, 5, 10, 25):
            mask = self._top_fraction_mask(pair_abs, float(pct) / 100.0)
            if mask.any():
                out[f"{prefix}_sign_flip_pair_abs_top{pct}pct"] = float(flip[mask].mean())
                out[f"{prefix}_sign_agree_pair_abs_top{pct}pct"] = float((~flip[mask]).mean())
        bucket_edges = (0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
        order = np.argsort(pair_abs, kind="mergesort")
        ranks = np.empty(order.shape[0], dtype=np.float32)
        ranks[order] = (np.arange(order.shape[0], dtype=np.float32) + 0.5) / float(order.shape[0])
        for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
            mask = (ranks >= lo) & (ranks < hi)
            if mask.any():
                label = f"q{int(lo * 100):02d}_{int(hi * 100):02d}"
                out[f"{prefix}_sign_agree_pair_abs_{label}"] = float((~flip[mask]).mean())
                out[f"{prefix}_sign_flip_pair_abs_{label}"] = float(flip[mask].mean())
        return out

    def _group_refresh_stability(
        self,
        *,
        current: np.ndarray,
        previous: np.ndarray,
        valid: np.ndarray,
    ) -> Dict[str, float]:
        g = int(self._group_size)
        n_groups = int(current.shape[0]) // max(1, g)
        if n_groups <= 1 or previous.shape != current.shape:
            return {}
        c = current[: n_groups * g].reshape(n_groups, g)
        p = previous[: n_groups * g].reshape(n_groups, g)
        m = valid[: n_groups * g].reshape(n_groups, g)
        row_valid = m.sum(axis=1) > 0
        if int(row_valid.sum()) <= 1:
            return {}
        cnt = np.maximum(m.sum(axis=1), 1)
        c_signed = np.sum(c * m, axis=1) / cnt
        p_signed = np.sum(p * m, axis=1) / cnt
        c_abs = np.sum(np.abs(c) * m, axis=1) / cnt
        p_abs = np.sum(np.abs(p) * m, axis=1) / cnt
        c_signed = c_signed[row_valid]
        p_signed = p_signed[row_valid]
        c_abs = c_abs[row_valid]
        p_abs = p_abs[row_valid]
        nonzero_sign = (c_signed != 0.0) & (p_signed != 0.0)
        out = {
            "signed_mean_spearman_prev": self._spearman(c_signed, p_signed),
            "signed_mean_corr_prev": self._pearson(c_signed, p_signed),
            "sign_flip_prev": float((np.sign(c_signed[nonzero_sign]) != np.sign(p_signed[nonzero_sign])).mean()) if nonzero_sign.any() else 0.0,
            "abs_mean_spearman_prev": self._spearman(c_abs, p_abs),
            "abs_mean_log_corr_prev": self._pearson(np.log(c_abs + 1.0e-30), np.log(p_abs + 1.0e-30)),
        }
        if nonzero_sign.any():
            pair_abs = np.maximum(c_abs, p_abs)
            flip = np.sign(c_signed) != np.sign(p_signed)
            out["sign_flip_pair_abs_weighted_prev"] = self._safe_div(
                float((pair_abs[nonzero_sign] * flip[nonzero_sign].astype(np.float32)).sum()),
                float(pair_abs[nonzero_sign].sum()),
            )
            for pct in (1, 5, 10, 25):
                mask = self._top_fraction_mask(pair_abs, float(pct) / 100.0) & nonzero_sign
                if mask.any():
                    out[f"abs_top{pct}pct_signed_sign_flip_prev"] = float(flip[mask].mean())
                    out[f"abs_top{pct}pct_signed_sign_agree_prev"] = float((~flip[mask]).mean())
        return out

    @staticmethod
    def _top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
        n = int(values.size)
        mask = np.zeros(n, dtype=bool)
        if n <= 0:
            return mask
        k = max(1, min(n, int(math.ceil(float(fraction) * float(n)))))
        finite = np.where(np.isfinite(values), values, -np.inf)
        idx = np.argpartition(finite, -k)[-k:]
        mask[idx] = True
        return mask

    def _proxy_quality(
        self,
        *,
        e_true: np.ndarray,
        e_hat: np.ndarray,
        risk_truth: np.ndarray,
        progress_truth_pos: np.ndarray,
        progress_truth_abs: np.ndarray,
        progress_truth_abs_h_pos: np.ndarray,
        scores: Dict[str, np.ndarray],
        selected_by_policy: Dict[str, np.ndarray],
        k: int,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {
            "e_hat_abs_spearman": self._spearman(np.abs(e_hat), np.abs(e_true)),
            "e_hat_signed_corr": self._pearson(e_hat, e_true),
            "e_hat_abs_median_over_true": self._safe_div(float(np.median(np.abs(e_hat))), float(np.median(np.abs(e_true)))),
            "e_hat_abs_p95_over_true": self._safe_div(float(np.percentile(np.abs(e_hat), 95.0)), float(np.percentile(np.abs(e_true), 95.0))),
        }
        truth_sets = {
            "risk": risk_truth,
            "progress_pos": progress_truth_pos,
            "progress_abs": progress_truth_abs,
            "progress_abs_h_pos": progress_truth_abs_h_pos,
        }
        for truth_name, truth in truth_sets.items():
            ideal = self._topk_mask(truth, k)
            ideal_mass = max(float(truth[ideal].sum()), 1.0e-30)
            for score_name, score in scores.items():
                selected = selected_by_policy.get(score_name)
                if selected is None:
                    selected = self._topk_mask(score, k)
                out[f"{score_name}_mass_recall_vs_{truth_name}"] = self._safe_div(float(truth[selected].sum()), ideal_mass)
                out[f"{score_name}_overlap_vs_{truth_name}"] = self._safe_div(float((selected & ideal).sum()), float(max(1, int(ideal.sum()))))
        return out

    @staticmethod
    def _value_sanity(
        *,
        m: np.ndarray,
        e_true: np.ndarray,
        e_hat: np.ndarray,
        signed_h: np.ndarray,
        progress_raw: np.ndarray,
    ) -> Dict[str, float]:
        linear = -m * e_hat
        quad = np.float32(0.5) * signed_h * (e_hat * e_hat)
        true_linear = -m * e_true
        true_quad = np.float32(0.5) * signed_h * (e_true * e_true)
        return {
            "progress_raw_pos_ratio": float((progress_raw > 0.0).mean()),
            "progress_raw_neg_ratio": float((progress_raw < 0.0).mean()),
            "linear_abs_mean": float(np.abs(linear).mean()),
            "quad_abs_mean": float(np.abs(quad).mean()),
            "quad_over_linear_abs_mean": CurvatureProgressTelemetry._safe_div(float(np.abs(quad).mean()), float(np.abs(linear).mean())),
            "true_quad_over_linear_abs_mean": CurvatureProgressTelemetry._safe_div(float(np.abs(true_quad).mean()), float(np.abs(true_linear).mean())),
            "signed_h_negative_quad_ratio": float((quad < 0.0).mean()),
        }

    @staticmethod
    def _selection_failure_diagnostics(
        *,
        m: np.ndarray,
        delta: np.ndarray,
        age: np.ndarray,
        e_true: np.ndarray,
        e_hat: np.ndarray,
        h_abs: np.ndarray,
        signed_h: np.ndarray,
        risk_truth: np.ndarray,
        progress_truth_pos: np.ndarray,
        progress_truth_abs_h_pos: np.ndarray,
        selected_by_policy: Dict[str, np.ndarray],
        k: int,
    ) -> Dict[str, float]:
        """Explain where progress-like scores spend or lose K-budget.

        These diagnostics are intentionally telemetry-only.  They compare a
        policy's selected set with the risk/progress oracle sets for the same
        shadow trajectory; they do not feed production decisions.
        """
        out: Dict[str, float] = {}
        n = int(e_true.size)
        if n <= 0 or k <= 0:
            return out
        risk_ideal = CurvatureProgressTelemetry._topk_mask(risk_truth, k)
        progress_ideal = CurvatureProgressTelemetry._topk_mask(progress_truth_pos, k)
        abs_h_progress_ideal = CurvatureProgressTelemetry._topk_mask(progress_truth_abs_h_pos, k)
        risk_mass = max(float(risk_truth[risk_ideal].sum()), 1.0e-30)
        progress_mass = max(float(progress_truth_pos[progress_ideal].sum()), 1.0e-30)
        abs_h_progress_mass = max(float(progress_truth_abs_h_pos[abs_h_progress_ideal].sum()), 1.0e-30)

        abs_true = np.abs(e_true).astype(np.float32, copy=False)
        abs_hat = np.abs(e_hat).astype(np.float32, copy=False)
        naive_abs = (np.maximum(age, np.float32(1.0)) * np.abs(delta)).astype(np.float32, copy=False)
        resid_over_naive = abs_true / np.maximum(naive_abs, np.float32(1.0e-30))
        # Case labels are diagnostics only:
        # accel: old displacement was likely smaller than current-delta extrapolation.
        # decel/dormant: current delta under-explains accumulated residual.
        accel_case = resid_over_naive < np.float32(0.5)
        decel_case = resid_over_naive > np.float32(2.0)
        sign_match = (np.sign(e_hat) == np.sign(e_true)) & (abs_true > 0.0) & (abs_hat > 0.0)
        neg_signed_h = signed_h < 0.0
        linear_abs_h = -m * e_hat
        quad_abs_h = np.float32(0.5) * h_abs * (e_hat * e_hat)
        raw_abs_h = linear_abs_h + quad_abs_h

        def add_stats(prefix: str, mask: np.ndarray) -> None:
            count = int(mask.sum())
            out[f"{prefix}_count_ratio"] = float(count) / float(max(1, k))
            if count <= 0:
                return
            out[f"{prefix}_risk_mass_over_ideal"] = CurvatureProgressTelemetry._safe_div(
                float(risk_truth[mask].sum()),
                risk_mass,
            )
            out[f"{prefix}_progress_mass_over_ideal"] = CurvatureProgressTelemetry._safe_div(
                float(progress_truth_pos[mask].sum()),
                progress_mass,
            )
            out[f"{prefix}_abs_h_progress_mass_over_ideal"] = CurvatureProgressTelemetry._safe_div(
                float(progress_truth_abs_h_pos[mask].sum()),
                abs_h_progress_mass,
            )
            out[f"{prefix}_neg_signed_h_ratio"] = float(neg_signed_h[mask].mean())
            out[f"{prefix}_ehat_sign_match_ratio"] = float(sign_match[mask].mean())
            out[f"{prefix}_resid_over_naive_median"] = float(np.median(resid_over_naive[mask]))
            out[f"{prefix}_ehat_over_true_median"] = float(np.median(abs_hat[mask] / np.maximum(abs_true[mask], np.float32(1.0e-30))))
            out[f"{prefix}_accel_case_ratio"] = float(accel_case[mask].mean())
            out[f"{prefix}_decel_case_ratio"] = float(decel_case[mask].mean())
            out[f"{prefix}_linear_abs_mean"] = float(np.abs(linear_abs_h[mask]).mean())
            out[f"{prefix}_quad_abs_h_mean"] = float(quad_abs_h[mask].mean())
            out[f"{prefix}_raw_abs_h_pos_ratio"] = float((raw_abs_h[mask] > 0.0).mean())

        for policy_name in (
            "curvature_proxy",
            "curvature_proxy_age2bit",
            "progress_proxy_abs_h_pos",
            "progress_proxy_abs_h_pos_age2bit",
        ):
            selected = selected_by_policy.get(policy_name)
            if selected is None:
                continue
            missed_risk = risk_ideal & ~selected
            extra_vs_risk = selected & ~risk_ideal
            recovered_progress = selected & progress_ideal
            missed_progress = progress_ideal & ~selected
            recovered_abs_h_progress = selected & abs_h_progress_ideal
            missed_abs_h_progress = abs_h_progress_ideal & ~selected
            add_stats(f"{policy_name}_missed_risk", missed_risk)
            add_stats(f"{policy_name}_extra_vs_risk", extra_vs_risk)
            add_stats(f"{policy_name}_recovered_progress", recovered_progress)
            add_stats(f"{policy_name}_missed_progress", missed_progress)
            add_stats(f"{policy_name}_recovered_abs_h_progress", recovered_abs_h_progress)
            add_stats(f"{policy_name}_missed_abs_h_progress", missed_abs_h_progress)
        return out

    @staticmethod
    def _update_state(
        *,
        state: _ShadowState,
        step: int,
        selected: np.ndarray,
        p: np.ndarray,
        element_ids: Optional[np.ndarray],
    ) -> None:
        state.theta_cp[selected] = p[selected]
        state.last_written_step[selected] = int(step)
        if element_ids is None:
            element_ids = np.arange(state.age_code.shape[0], dtype=np.uint64)
        promote0 = (state.age_code == 0) & CurvatureProgressTelemetry._age2_phase_hit_np(
            element_ids=element_ids,
            step=int(step),
            period=6,
            salt=0xA511E9B3,
        )
        promote1 = (state.age_code == 1) & CurvatureProgressTelemetry._age2_phase_hit_np(
            element_ids=element_ids,
            step=int(step),
            period=24,
            salt=0x63D83595,
        )
        promote2 = (state.age_code == 2) & CurvatureProgressTelemetry._age2_phase_hit_np(
            element_ids=element_ids,
            step=int(step),
            period=96,
            salt=0xB5297A4D,
        )
        next_code = state.age_code.copy()
        promote = promote0 | promote1 | promote2
        next_code[promote] = np.minimum(next_code[promote] + 1, 3).astype(np.uint8)
        next_code[selected] = 0
        state.age_code = next_code

    @staticmethod
    def _topk_mask(score: np.ndarray, k: int) -> np.ndarray:
        k = max(1, min(int(k), int(score.size)))
        mask = np.zeros(score.shape[0], dtype=bool)
        if k >= score.size:
            mask[:] = True
            return mask
        finite_score = np.where(np.isfinite(score), score, -np.inf)
        idx = np.argpartition(finite_score, -k)[-k:]
        mask[idx] = True
        return mask

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or b.size != a.size:
            return 0.0
        aa = a.astype(np.float64)
        bb = b.astype(np.float64)
        aa -= aa.mean()
        bb -= bb.mean()
        denom = math.sqrt(float((aa * aa).sum()) * float((bb * bb).sum()))
        return float((aa * bb).sum() / denom) if denom > 0.0 else 0.0

    @staticmethod
    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or b.size != a.size:
            return 0.0
        return CurvatureProgressTelemetry._pearson(
            CurvatureProgressTelemetry._rank(a),
            CurvatureProgressTelemetry._rank(b),
        )

    @staticmethod
    def _rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(order.shape[0], dtype=np.float64)
        ranks[order] = np.arange(order.shape[0], dtype=np.float64)
        return ranks

    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        return float(a) / float(b) if b not in (0.0, -0.0) and math.isfinite(float(b)) else 0.0

    def _append(self, record: Dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
