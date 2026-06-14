"""Sampled sparse-cache shadow experiment for Oracle-B residual locality.

Enable with ``PROPOSAL_SPARSE_CACHE_SHADOW_LOG=/path/to/cache.jsonl``.

This module is experiment-only. It samples full groups (all lanes in a group),
keeps CPU-side cache states for several admission/eviction policies, and
compares each policy against an Oracle-B sampled top-K trace:

    truth_score_i = H_i * (theta_i - theta_cp_oracle_i)^2

It never reads or mutates production ``dg``/write state.
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


@dataclass(frozen=True)
class _VariantSpec:
    admit: str
    evict: str
    capacity_fraction: float
    score_mode: str
    lambda_floor: float


@dataclass
class _VariantState:
    present: np.ndarray
    theta_cp: np.ndarray
    last_access: np.ndarray
    hit_count: np.ndarray
    score_ema: np.ndarray
    group_debt: np.ndarray
    last_written: np.ndarray


class SparseCacheShadowTelemetry:
    """CPU-side sampled cache simulator, isolated from production logic."""

    def __init__(
        self,
        *,
        log_path: Optional[str],
        sample_groups: int,
        every: int,
        seed: int,
        capacity_fractions: Tuple[float, ...],
        admits: Tuple[str, ...],
        evicts: Tuple[str, ...],
        score_modes: Tuple[str, ...],
        dg_lambdas: Tuple[float, ...],
        ema_decay: float,
    ) -> None:
        self.log_path = Path(log_path).expanduser() if log_path else None
        self.sample_groups = max(1, int(sample_groups))
        self.every = max(1, int(every))
        self.seed = int(seed)
        self.capacity_fractions = tuple(
            sorted({float(c) for c in capacity_fractions if float(c) > 0.0})
        )
        self.admits = tuple(admits)
        self.evicts = tuple(evicts)
        self.score_modes = tuple(score_modes)
        self.dg_lambdas = tuple(dg_lambdas)
        self.ema_decay = min(max(float(ema_decay), 0.0), 0.9999)
        self._initialized = False
        self._sources: List[_SampleSource] = []
        self._sample_group_ids: Optional[np.ndarray] = None
        self._sample_lane_ids: Optional[np.ndarray] = None
        self._oracle_theta_cp: Optional[np.ndarray] = None
        self._oracle_select_count: Optional[np.ndarray] = None
        self._variants: Dict[str, Tuple[_VariantSpec, _VariantState]] = {}
        self._group_size = 1

    @classmethod
    def from_env(cls) -> "SparseCacheShadowTelemetry":
        caps_raw = os.environ.get(
            "PROPOSAL_SPARSE_CACHE_SHADOW_CAPS",
            "0.001,0.0025,0.005,0.01,0.02,0.03",
        )
        caps: List[float] = []
        for item in caps_raw.split(","):
            item = item.strip()
            if not item:
                continue
            caps.append(float(item))
        return cls(
            log_path=os.environ.get("PROPOSAL_SPARSE_CACHE_SHADOW_LOG", ""),
            sample_groups=int(
                os.environ.get("PROPOSAL_SPARSE_CACHE_SHADOW_SAMPLE_GROUPS", "2048")
                or "2048"
            ),
            every=int(os.environ.get("PROPOSAL_SPARSE_CACHE_SHADOW_EVERY", "1") or "1"),
            seed=int(os.environ.get("PROPOSAL_SPARSE_CACHE_SHADOW_SEED", "2024") or "2024"),
            capacity_fractions=tuple(caps),
            admits=cls._parse_csv_env(
                "PROPOSAL_SPARSE_CACHE_SHADOW_ADMITS",
                "oracle_all,self_all,self_repeat2,oracle_singleton",
            ),
            evicts=cls._parse_csv_env("PROPOSAL_SPARSE_CACHE_SHADOW_EVICTS", "lru,ema"),
            score_modes=cls._parse_csv_env(
                "PROPOSAL_SPARSE_CACHE_SHADOW_SCORE_MODES",
                "spike,dg1,dg19,cache_spike,cache_dg1,cache_dg19",
            ),
            dg_lambdas=cls._parse_float_csv_env(
                "PROPOSAL_SPARSE_CACHE_SHADOW_DG_LAMBDAS",
                "1,19",
            ),
            ema_decay=float(
                os.environ.get("PROPOSAL_SPARSE_CACHE_SHADOW_EMA_DECAY", "0.95")
                or "0.95"
            ),
        )

    @staticmethod
    def _parse_csv_env(name: str, default: str) -> Tuple[str, ...]:
        raw = os.environ.get(name, default) or default
        return tuple(item.strip() for item in raw.split(",") if item.strip())

    @staticmethod
    def _parse_float_csv_env(name: str, default: str) -> Tuple[float, ...]:
        raw = os.environ.get(name, default) or default
        vals = []
        for item in raw.split(","):
            item = item.strip()
            if item:
                vals.append(float(item))
        return tuple(vals)

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
        if not self.enabled:
            return
        if int(step) % self.every != 0:
            return
        groups_list = list(groups)
        if not self._initialized:
            self._initialize(groups_list=groups_list, group_size=int(group_size))
        if not self._sources:
            return
        p, m, v = self._gather_sample(groups_list=groups_list)
        if self._oracle_theta_cp is None:
            self._oracle_theta_cp = p.copy()
        if self._oracle_select_count is None:
            self._oracle_select_count = np.zeros(p.shape[0], dtype=np.int32)

        h = self._lookup_h(
            v=v,
            h_lut_values=h_lut_values,
            h_lut_log_v_min=h_lut_log_v_min,
            h_lut_inv_log_v_width=h_lut_inv_log_v_width,
        )
        denom = np.sqrt(np.maximum(v, 0.0), dtype=np.float32) * float(sqrt_bc2_inv) + float(eps)
        delta = float(lr_over_bc1) * m / denom
        delta_sq = delta * delta
        spike_score = h * delta_sq
        truth_score = h * ((p - self._oracle_theta_cp) ** 2)
        n = int(p.shape[0])
        group_size = int(group_size)
        n_groups = n // group_size
        k = max(1, min(n, int(math.ceil(float(topk_ratio) * float(n)))))
        oracle_selected = self._topk_mask(truth_score, k)
        oracle_mass = float(truth_score[oracle_selected].sum())
        oracle_group_counts = oracle_selected.reshape(n_groups, group_size).sum(axis=1)

        records: List[Dict[str, Any]] = []
        for key, (spec, state) in self._variants.items():
            cache_score = self._score_for_variant(
                spec=spec,
                state=state,
                p=p,
                h=h,
                delta_sq=delta_sq,
                spike_score=spike_score,
                n_groups=n_groups,
                group_size=group_size,
            )
            predicted = self._topk_mask(cache_score, k)
            intersection = predicted & oracle_selected
            cache_hit_oracle = oracle_selected & state.present
            selected_cache_backed = predicted & state.present

            pred_group_counts = predicted.reshape(n_groups, group_size).sum(axis=1)
            admit = self._admit_mask(
                spec=spec,
                oracle_selected=oracle_selected,
                predicted=predicted,
                oracle_group_counts=oracle_group_counts,
                pred_group_counts=pred_group_counts,
                oracle_select_count=self._oracle_select_count,
            )
            self._update_variant(
                state=state,
                spec=spec,
                step=int(step),
                p=p,
                delta_sq=delta_sq,
                truth_score=truth_score,
                predicted=predicted,
                oracle_selected=oracle_selected,
                admit=admit,
            )

            records.append({
                "variant": key,
                "admit": spec.admit,
                "evict": spec.evict,
                "capacity_fraction": spec.capacity_fraction,
                "score_mode": spec.score_mode,
                "lambda_floor": spec.lambda_floor,
                "capacity_entries": int(self._capacity_entries(spec.capacity_fraction, n)),
                "cache_occupancy": int(state.present.sum()),
                "group_debt_mean": (
                    float(state.group_debt.mean())
                    if spec.score_mode in ("dg", "cache_dg") and state.group_debt.size
                    else 0.0
                ),
                "recall_at_k": self._safe_div(float(intersection.sum()), float(oracle_selected.sum())),
                "oracle_mass_recovered": self._safe_div(
                    float(truth_score[intersection].sum()), oracle_mass,
                ),
                "predicted_truth_mass_over_oracle_mass": self._safe_div(
                    float(truth_score[predicted].sum()), oracle_mass,
                ),
                "oracle_selected_cache_hit_ratio": self._safe_div(
                    float(cache_hit_oracle.sum()), float(oracle_selected.sum()),
                ),
                "predicted_cache_backed_ratio": self._safe_div(
                    float(selected_cache_backed.sum()), float(predicted.sum()),
                ),
                "admit_count": int(admit.sum()),
            })

        self._oracle_theta_cp[oracle_selected] = p[oracle_selected]
        self._oracle_select_count[oracle_selected] += 1
        self._append_record({
            "step": int(step),
            "mode": "sparse_cache_shadow",
            "sample_groups": int(n_groups),
            "sample_elements": int(n),
            "topk_ratio": float(topk_ratio),
            "sample_k": int(k),
            "oracle_mass": oracle_mass,
            "oracle_selected_ratio": self._safe_div(float(oracle_selected.sum()), float(n)),
            "oracle_singleton_group_ratio": self._safe_div(
                float((oracle_group_counts == 1).sum()),
                float((oracle_group_counts > 0).sum()),
            ),
            "variants": records,
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
        sampled_groups = np.sort(
            rng.choice(total_groups, size=n_sample_groups, replace=False).astype(np.int64)
        )
        sources: List[_SampleSource] = []
        sample_group_ids: List[int] = []
        sample_lane_ids: List[int] = []
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
                sample_lane_ids.extend(range(int(group_size)))
            sample_group_base += int(len(global_groups))
        n = int(sample_index_base)
        self._group_size = int(group_size)
        self._sources = sources
        self._sample_group_ids = np.asarray(sample_group_ids, dtype=np.int64)
        self._sample_lane_ids = np.asarray(sample_lane_ids, dtype=np.int16)
        self._oracle_select_count = np.zeros(n, dtype=np.int32)
        self._init_variants(n)
        self._initialized = True

    def _init_variants(self, n: int) -> None:
        admits = self.admits
        evicts = self.evicts
        n_groups = max(1, n // max(1, self._group_size))
        base_specs: List[_VariantSpec] = []
        if "spike" in self.score_modes:
            base_specs.append(_VariantSpec(
                admit="none",
                evict="none",
                capacity_fraction=0.0,
                score_mode="spike",
                lambda_floor=0.0,
            ))
        if "dg" in self.score_modes or "dg1" in self.score_modes or "dg19" in self.score_modes:
            lambdas = list(self.dg_lambdas)
            if "dg1" in self.score_modes and 1.0 not in lambdas:
                lambdas.append(1.0)
            if "dg19" in self.score_modes and 19.0 not in lambdas:
                lambdas.append(19.0)
            for lambda_floor in sorted(set(lambdas)):
                base_specs.append(_VariantSpec(
                    admit="none",
                    evict="none",
                    capacity_fraction=0.0,
                    score_mode="dg",
                    lambda_floor=float(lambda_floor),
                ))
        for spec in base_specs:
            key = self._variant_key(spec)
            self._variants[key] = (spec, self._new_state(n=n, n_groups=n_groups))
        cache_score_modes: List[Tuple[str, float]] = []
        if "cache_spike" in self.score_modes:
            cache_score_modes.append(("cache_spike", 0.0))
        if "cache_dg" in self.score_modes or "cache_dg1" in self.score_modes or "cache_dg19" in self.score_modes:
            lambdas = list(self.dg_lambdas)
            if "cache_dg1" in self.score_modes and 1.0 not in lambdas:
                lambdas.append(1.0)
            if "cache_dg19" in self.score_modes and 19.0 not in lambdas:
                lambdas.append(19.0)
            for lambda_floor in sorted(set(lambdas)):
                cache_score_modes.append(("cache_dg", float(lambda_floor)))
        for admit in admits:
            for evict in evicts:
                for cap in self.capacity_fractions:
                    for score_mode, lambda_floor in cache_score_modes:
                        spec = _VariantSpec(
                            admit=admit,
                            evict=evict,
                            capacity_fraction=float(cap),
                            score_mode=score_mode,
                            lambda_floor=float(lambda_floor),
                        )
                        self._variants[self._variant_key(spec)] = (
                            spec,
                            self._new_state(n=n, n_groups=n_groups),
                        )

    @staticmethod
    def _variant_key(spec: _VariantSpec) -> str:
        if spec.score_mode == "spike":
            return "no_cache:spike"
        if spec.score_mode == "dg":
            return f"no_cache:dg_l{spec.lambda_floor:g}"
        return (
            f"{spec.score_mode}_l{spec.lambda_floor:g}:"
            f"{spec.admit}:{spec.evict}:cap{spec.capacity_fraction:g}"
        )

    @staticmethod
    def _new_state(*, n: int, n_groups: int) -> _VariantState:
        return _VariantState(
            present=np.zeros(n, dtype=bool),
            theta_cp=np.zeros(n, dtype=np.float32),
            last_access=np.zeros(n, dtype=np.int32),
            hit_count=np.zeros(n, dtype=np.int32),
            score_ema=np.zeros(n, dtype=np.float32),
            group_debt=np.zeros(n_groups, dtype=np.float32),
            last_written=np.zeros(n, dtype=bool),
        )

    def _score_for_variant(
        self,
        *,
        spec: _VariantSpec,
        state: _VariantState,
        p: np.ndarray,
        h: np.ndarray,
        delta_sq: np.ndarray,
        spike_score: np.ndarray,
        n_groups: int,
        group_size: int,
    ) -> np.ndarray:
        if spec.score_mode == "spike":
            return spike_score

        cache_score = h * ((p - state.theta_cp) ** 2)
        if spec.score_mode == "cache_spike":
            return np.where(state.present, cache_score, spike_score)

        active = ~state.last_written
        if spec.score_mode == "cache_dg":
            active &= ~state.present
        active_counts = active.reshape(n_groups, group_size).sum(axis=1).astype(np.float32)
        floor = np.zeros(n_groups, dtype=np.float32)
        np.divide(
            state.group_debt,
            np.maximum(active_counts, np.float32(1.0)),
            out=floor,
            where=active_counts > 0,
        )
        floor_i = np.repeat(floor, group_size)
        dg_score = h * (np.float32(spec.lambda_floor) * floor_i * active.astype(np.float32) + delta_sq)
        if spec.score_mode == "cache_dg":
            return np.where(state.present, cache_score, dg_score)
        return dg_score

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

    def _admit_mask(
        self,
        *,
        spec: _VariantSpec,
        oracle_selected: np.ndarray,
        predicted: np.ndarray,
        oracle_group_counts: np.ndarray,
        pred_group_counts: np.ndarray,
        oracle_select_count: np.ndarray,
    ) -> np.ndarray:
        group_size = int(oracle_selected.size // oracle_group_counts.size)
        oracle_counts_per_elem = np.repeat(oracle_group_counts, group_size)
        pred_counts_per_elem = np.repeat(pred_group_counts, group_size)
        if spec.admit == "oracle_all":
            return oracle_selected.copy()
        if spec.admit == "oracle_singleton":
            return oracle_selected & (oracle_counts_per_elem == 1)
        if spec.admit == "oracle_up_to2":
            return oracle_selected & (oracle_counts_per_elem <= 2)
        if spec.admit == "oracle_miss":
            return oracle_selected & (~predicted)
        if spec.admit == "self_singleton":
            return predicted & (pred_counts_per_elem == 1)
        if spec.admit == "self_repeat2":
            return predicted & (oracle_select_count > 0)
        return predicted.copy()

    def _update_variant(
        self,
        *,
        state: _VariantState,
        spec: _VariantSpec,
        step: int,
        p: np.ndarray,
        delta_sq: np.ndarray,
        truth_score: np.ndarray,
        predicted: np.ndarray,
        oracle_selected: np.ndarray,
        admit: np.ndarray,
    ) -> None:
        if spec.score_mode in ("dg", "cache_dg"):
            self._update_group_debt(
                state=state,
                spec=spec,
                p=p,
                delta_sq=delta_sq,
                predicted=predicted,
            )
        present_hit = state.present & oracle_selected
        state.score_ema[state.present] *= np.float32(self.ema_decay)
        state.last_access[state.present & predicted] = int(step)
        state.hit_count[present_hit] += 1
        state.score_ema[present_hit] = np.maximum(
            state.score_ema[present_hit],
            truth_score[present_hit],
        )
        if admit.any() and spec.capacity_fraction > 0.0:
            state.present[admit] = True
            state.theta_cp[admit] = p[admit]
            state.last_access[admit] = int(step)
            state.hit_count[admit] += 1
            state.score_ema[admit] = np.maximum(state.score_ema[admit], truth_score[admit])
        cap = self._capacity_entries(spec.capacity_fraction, p.shape[0])
        occ = int(state.present.sum())
        if occ <= cap:
            return
        present_idx = np.flatnonzero(state.present)
        if cap <= 0:
            evict = state.present.copy()
            if spec.score_mode == "cache_dg":
                self._fold_evicted_cache_into_debt(state=state, p=p, evict=evict)
            state.present[:] = False
            state.theta_cp[:] = 0.0
            state.last_access[:] = 0
            state.hit_count[:] = 0
            state.score_ema[:] = 0.0
            return
        if spec.evict == "lfu":
            key = state.hit_count[present_idx].astype(np.float64) * 1.0e9 + state.last_access[present_idx]
        elif spec.evict == "ema":
            key = state.score_ema[present_idx].astype(np.float64)
        else:
            key = state.last_access[present_idx].astype(np.float64)
        keep_local = np.argpartition(key, -cap)[-cap:]
        keep = present_idx[keep_local]
        new_present = np.zeros_like(state.present)
        new_present[keep] = True
        evict = state.present & (~new_present)
        if spec.score_mode == "cache_dg":
            self._fold_evicted_cache_into_debt(state=state, p=p, evict=evict)
        state.present[evict] = False
        state.theta_cp[evict] = 0.0
        state.last_access[evict] = 0
        state.hit_count[evict] = 0
        state.score_ema[evict] = 0.0

    def _update_group_debt(
        self,
        *,
        state: _VariantState,
        spec: _VariantSpec,
        p: np.ndarray,
        delta_sq: np.ndarray,
        predicted: np.ndarray,
    ) -> None:
        group_ids = self._sample_group_ids
        if group_ids is None:
            return
        n_groups = state.group_debt.size
        group_size = max(1, self._group_size)
        active = ~state.last_written
        if spec.score_mode == "cache_dg":
            active &= ~state.present
        active_counts = active.reshape(n_groups, group_size).sum(axis=1).astype(np.float32)
        floor = np.zeros(n_groups, dtype=np.float32)
        np.divide(
            state.group_debt,
            np.maximum(active_counts, np.float32(1.0)),
            out=floor,
            where=active_counts > 0,
        )
        remove = predicted & active
        if remove.any():
            remove_mass = np.bincount(
                group_ids[remove],
                weights=floor[group_ids[remove]].astype(np.float64),
                minlength=n_groups,
            ).astype(np.float32)
            state.group_debt -= remove_mass
            np.maximum(state.group_debt, np.float32(0.0), out=state.group_debt)
        add = ~predicted
        if spec.score_mode == "cache_dg":
            add &= ~state.present
        if add.any():
            add_mass = np.bincount(
                group_ids[add],
                weights=delta_sq[add].astype(np.float64),
                minlength=n_groups,
            ).astype(np.float32)
            state.group_debt += add_mass
        state.last_written[:] = predicted

    def _fold_evicted_cache_into_debt(
        self,
        *,
        state: _VariantState,
        p: np.ndarray,
        evict: np.ndarray,
    ) -> None:
        if not evict.any() or state.group_debt.size == 0:
            return
        group_ids = self._sample_group_ids
        if group_ids is None:
            return
        resid = (p[evict] - state.theta_cp[evict]) ** 2
        fold = np.bincount(
            group_ids[evict],
            weights=resid.astype(np.float64),
            minlength=state.group_debt.size,
        ).astype(np.float32)
        state.group_debt += fold

    @staticmethod
    def _capacity_entries(capacity_fraction: float, n: int) -> int:
        if float(capacity_fraction) <= 0.0:
            return 0
        return max(1, min(int(n), int(math.floor(float(capacity_fraction) * float(n)))))

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return float(num) / float(den) if float(den) != 0.0 else 0.0

    def _append_record(self, record: Dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = ["SparseCacheShadowTelemetry"]
