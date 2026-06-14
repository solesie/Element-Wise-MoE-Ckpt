"""Cutoff orchestration for the production GPU approx-probe path."""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional

import torch

from ..checkpoint.static_descriptors import device_tensor_from_pinned_host
from ..checkpoint.step_context import global_selection_element_count
from .adaptive_controller import CutoffDecision

try:
    import triton
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

_APPROX_PROBE_CI_SHORTFALL_GROW_AFTER = 3
_APPROX_PROBE_COUNT_MISMATCH_GROW_AFTER = 3


class CutoffOrchestrationMixin:
    def _valid_positive_threshold(self, tau: float) -> bool:
        return bool(math.isfinite(float(tau)) and float(tau) > 0.0)

    def _stage_candidate_taus(
        self,
        values: List[float],
        *,
        tau_center: float,
    ) -> int:
        taus_dev = getattr(self, "_selection_candidate_taus", None)
        total_th = getattr(self, "_selection_total_th", None)
        if taus_dev is None or total_th is None:
            return 0
        from ..cutoff.approx_probe_kernels import launch_stage_candidate_taus

        return launch_stage_candidate_taus(
            candidate_taus=taus_dev,
            total_th=total_th,
            values=tuple(float(v) for v in values),
            tau_center=float(max(tau_center, 1.0e-12)),
        )

    def _initial_cutoff_center(self, decision: Any) -> float:
        tau = getattr(decision, "tau", None)
        if tau is not None and self._valid_positive_threshold(float(tau)):
            return float(tau)
        tau_star = self._adaptive_cutoff.predicted_tau_star()
        if tau_star is not None and self._valid_positive_threshold(float(tau_star)):
            return float(tau_star)
        observations = self._adaptive_cutoff.observations
        for obs in reversed(observations):
            if self._valid_positive_threshold(float(obs.tau)):
                return float(obs.tau)
        current = float(getattr(self, "_selection_total_th_host", 1.0))
        if self._valid_positive_threshold(current):
            return float(current)
        return 1.0

    def _cold_start_candidate_ladder(
        self,
        *,
        tau: float,
        log_bin_width: float,
    ) -> List[float]:
        cap = int(getattr(self, "_selection_candidate_capacity", 16))
        center = float(tau)
        if not self._valid_positive_threshold(center):
            return []
        step = max(float(log_bin_width), 1.0e-6)
        out: List[float] = []
        for mul in (0.0, -4.0, 4.0, -2.0, 2.0, -1.0, 1.0, -8.0,
                    8.0, -12.0, 12.0, -16.0, 16.0):
            if len(out) >= cap:
                break
            value = center * math.exp(float(mul) * step)
            if self._valid_positive_threshold(value):
                out.append(float(value))
        return out

    def _choose_tau_from_cold_start_hist(
        self,
        *,
        hist: torch.Tensor,
        smin: float,
        smax: float,
        total_elems_global: int,
    ) -> Optional[Tuple[float, float]]:
        if (
            not math.isfinite(float(smin))
            or not math.isfinite(float(smax))
            or float(smin) <= 0.0
            or float(smax) <= float(smin)
        ):
            return None
        target = max(1, int(math.ceil(float(self.topk_ratio) * float(total_elems_global))))
        counts = hist.detach().cpu().numpy()
        running = 0
        crossing = -1
        for idx in range(int(counts.shape[0]) - 1, -1, -1):
            running += int(counts[idx])
            if running >= target:
                crossing = int(idx)
                break
        if crossing < 0:
            return None
        log_lo = math.log(float(smin))
        log_hi = math.log(float(smax))
        width = (log_hi - log_lo) / float(max(1, int(counts.shape[0])))
        tau = math.exp(log_lo + (float(crossing) + 0.5) * width)
        if not self._valid_positive_threshold(tau):
            return None
        return float(tau), float(width)

    def _cold_start_cutoff_bootstrap(
        self,
        *,
        runtime_state: Any,
        hparams: Any,
        total_elems_global: int,
        group_size: int,
        process_group: Any,
        world_size: int,
        random_mode: bool,
        random_salt: int,
    ) -> Optional[Tuple[float, float]]:
        """Bootstrap tau scale with a full score histogram during cold start."""

        from ..cutoff.cold_start import (
            launch_build_score_log_hist_for_group,
            launch_reduce_score_minmax_for_group,
        )

        hist = self._selection_hist
        smin = self._selection_smin
        smax = self._selection_smax
        smin.fill_(float("inf"))
        smax.zero_()
        for grp in self._groups:
            launch_reduce_score_minmax_for_group(
                m_flat=grp["m_flat"],
                v_flat=grp["v_flat"],
                h_lut_values=runtime_state.quantiles.h_lut.values,
                h_lut_log_v_min=runtime_state.quantiles.h_lut.log_v_min,
                h_lut_inv_log_v_width=runtime_state.quantiles.h_lut.inv_log_v_width,
                entry_elem_start=grp["entry_elem_start"],
                entry_group_start=grp["entry_group_start"],
                entry_group_count=grp["entry_group_count"],
                entry_block_entry_idx=grp["entry_block_entry_idx"],
                entry_block_local_group_start=grp["entry_block_local_group_start"],
                smin=smin,
                smax=smax,
                master_off_elem=int(grp["master_off"]) * int(group_size),
                group_size=int(group_size),
                lr_over_bc1=float(hparams.lr_over_bc1),
                sqrt_bc2_inv=float(hparams.sqrt_bc2_inv),
                eps=float(hparams.eps),
                random_mode=bool(random_mode),
                random_salt=int(random_salt),
            )
        if (
            int(world_size) > 1
            and process_group is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(smin, op=torch.distributed.ReduceOp.MIN, group=process_group)
            torch.distributed.all_reduce(smax, op=torch.distributed.ReduceOp.MAX, group=process_group)
        hist.zero_()
        for grp in self._groups:
            launch_build_score_log_hist_for_group(
                m_flat=grp["m_flat"],
                v_flat=grp["v_flat"],
                h_lut_values=runtime_state.quantiles.h_lut.values,
                h_lut_log_v_min=runtime_state.quantiles.h_lut.log_v_min,
                h_lut_inv_log_v_width=runtime_state.quantiles.h_lut.inv_log_v_width,
                entry_elem_start=grp["entry_elem_start"],
                entry_group_start=grp["entry_group_start"],
                entry_group_count=grp["entry_group_count"],
                entry_block_entry_idx=grp["entry_block_entry_idx"],
                entry_block_local_group_start=grp["entry_block_local_group_start"],
                hist=hist,
                smin=smin,
                smax=smax,
                master_off_elem=int(grp["master_off"]) * int(group_size),
                group_size=int(group_size),
                lr_over_bc1=float(hparams.lr_over_bc1),
                sqrt_bc2_inv=float(hparams.sqrt_bc2_inv),
                eps=float(hparams.eps),
                random_mode=bool(random_mode),
                random_salt=int(random_salt),
            )
        if (
            int(world_size) > 1
            and process_group is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(hist, op=torch.distributed.ReduceOp.SUM, group=process_group)
        pinned = getattr(self, "_cold_start_hist_pinned", None)
        if pinned is None or pinned.numel() != hist.numel():
            pinned = torch.empty(hist.numel(), dtype=torch.int64, pin_memory=True)
            self._cold_start_hist_pinned = pinned
        pinned.copy_(hist, non_blocking=True)
        stream = torch.cuda.current_stream()
        stream.synchronize()
        return self._choose_tau_from_cold_start_hist(
            hist=pinned,
            smin=float(smin.item()),
            smax=float(smax.item()),
            total_elems_global=int(total_elems_global),
        )

    def _global_selection_element_count_cached(
        self,
        local_n_elements: int,
        *,
        process_group: Any,
        world_size: int,
    ) -> int:
        cached = getattr(self, "_global_selection_element_count_host", None)
        if cached is not None and int(cached) > 0:
            return int(cached)
        count = global_selection_element_count(
            int(local_n_elements),
            self._selection_global_n_elements,
            process_group=process_group,
            world_size=int(world_size),
        )
        self._global_selection_element_count_host = int(count)
        return int(count)

    def _consume_direct_emission_feedback(
        self,
        *,
        total_elems_global: int,
    ) -> None:
        warning = getattr(self, "_approx_probe_reject_warning", None)
        if warning is not None:
            warning.emit_ready()
        n_global = max(1, int(total_elems_global))
        expert_state = getattr(self, "_expert_state", None)
        if expert_state is None:
            return
        consume = getattr(expert_state, "consume_direct_emission_feedback", None)
        if consume is None:
            return
        feedback = consume()
        if not feedback:
            return
        tau = float(feedback.get("tau", 0.0))
        written = int(feedback.get("written", 0))
        selected = int(feedback.get("selected", written))
        ratio = float(selected) / float(n_global)
        telemetry = getattr(self, "_approx_probe_telemetry", None)
        if telemetry is not None and getattr(telemetry, "enabled", False):
            telemetry.emit_with_direct_feedback(
                direct_feedback=feedback,
                total_elems_global=int(n_global),
                target_ratio=float(self.topk_ratio),
            )
        self._emit_lfcl_overlap_telemetry(
            direct_feedback=feedback,
            total_elems_global=int(n_global),
        )
        if not self._valid_positive_threshold(tau) or selected <= 0:
            return
        self._adaptive_cutoff.observe(
            step=int(feedback.get("step", 0)),
            tau=float(tau),
            selected_ratio=float(ratio),
            exact=False,
        )
        self._adaptive_cutoff.observe_final_count_feedback(
            [(float(tau), int(selected))],
            step=int(feedback.get("step", 0)),
            current_tau=float(tau),
            selected_ratio=float(ratio),
            total_elems_global=int(n_global),
            cutoff_tolerance=float(self.cutoff_K_tolerance),
        )

    def _emit_lfcl_overlap_telemetry(
        self,
        *,
        direct_feedback: Dict[str, Any],
        total_elems_global: int,
    ) -> None:
        import json
        import os

        path = os.environ.get("PROPOSAL_LFCL_OVERLAP_LOG", "")
        if not path:
            return
        step = int(direct_feedback.get("step", 0))
        fired_by_step = getattr(self, "_lfcl_fired_experts_by_step", {}) or {}
        fired = set(tuple(x) for x in fired_by_step.pop(step, []) or [])
        if not fired:
            return
        selected_by_expert = list(direct_feedback.get("selected_by_expert") or [])
        expert_ids = [tuple(x) for x in (direct_feedback.get("expert_ids") or [])]
        n = min(len(selected_by_expert), len(expert_ids))
        selected_total = max(0, int(direct_feedback.get("selected", 0)))
        if selected_total <= 0:
            selected_total = sum(max(0, int(x)) for x in selected_by_expert[:n])
        overlap = 0
        fired_with_selected = 0
        for idx in range(n):
            if expert_ids[idx] in fired:
                selected_i = max(0, int(selected_by_expert[idx]))
                overlap += selected_i
                if selected_i > 0:
                    fired_with_selected += 1
        record = {
            "step": int(step),
            "target_ratio": float(self.topk_ratio),
            "selected_total": int(selected_total),
            "lfcl_fired_experts": len(fired),
            "lfcl_fired_with_selected": int(fired_with_selected),
            "selected_in_lfcl_experts": int(overlap),
            "selected_in_lfcl_over_selected": (
                float(overlap) / float(selected_total)
                if selected_total > 0 else 0.0
            ),
            "selected_in_lfcl_over_global": (
                float(overlap) / float(max(1, int(total_elems_global)))
            ),
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _launch_gpu_approx_cutoff(
        self,
        *,
        global_step: int,
        runtime_state: Any,
        hparams: Any,
        n_candidates: int,
        total_elems_global: int,
        group_size: int,
        process_group: Any,
        world_size: int,
        random_mode: bool,
        random_salt: int,
    ) -> str:
        from ..cutoff.approx_probe_kernels import (
            launch_extract_age2bit_floors,
            launch_generated_block_probe_for_group,
            launch_probe_decision,
            launch_retry_candidates_if_code,
            launch_zero_probe_accumulators_if_code,
        )

        n_cand = max(0, int(n_candidates))
        if n_cand <= 0:
            self._approx_probe_reject.fill_(1)
            return "gpu_probe:no_candidates"
        age2bit_state = getattr(runtime_state, "age2bit", None)
        if age2bit_state is None:
            raise RuntimeError("Age2Bit state is required for production cutoff")

        profile = os.environ.get("PROPOSAL_APPROX_LAUNCH_PROFILE", "") in (
            "1", "true", "TRUE",
        )
        t_entry = time.perf_counter() if profile else 0.0
        self._approx_probe_sum_q.zero_()
        self._approx_probe_sum_q2.zero_()
        self._approx_probe_n_blocks.zero_()
        self._approx_probe_capacity_demand_hint.zero_()
        self._approx_probe_reject.zero_()
        self._approx_probe_decision_code.zero_()
        # Consume the previous iteration's sampled delta2 histogram before
        # zeroing it.  The freshly extracted floor snapshot is then used by
        # both this iteration's probe and direct emission, while this probe
        # fills the histogram for the next iteration.  Updating floors after
        # the probe would break probe/direct score identity for the current
        # tau decision.
        launch_extract_age2bit_floors(
            floor_hist=age2bit_state.floor_hist,
            floor_mid=age2bit_state.floor_mid,
            floor_old=age2bit_state.floor_old,
        )
        age2bit_state.floor_hist.zero_()
        t_after_zero = time.perf_counter() if profile else 0.0
        telemetry_sinks = [
            telemetry
            for telemetry in (
                getattr(self, "_approx_probe_telemetry", None),
            )
            if telemetry is not None and getattr(telemetry, "enabled", False)
        ]
        telemetry_enabled = bool(telemetry_sinks)
        telemetry_stream = torch.cuda.current_stream()
        if telemetry_enabled:
            for telemetry in telemetry_sinks:
                telemetry.record_probe_start(
                    step=int(global_step),
                    stream=telemetry_stream,
                )

        for grp in self._groups:
            group_blocks = int(grp.get("approx_entry_block_count", 0))
            if group_blocks <= 0:
                continue
            max_samples_for_group = int(math.ceil(
                float(self._approx_probe_sample_blocks_cap)
                * float(group_blocks)
                / float(max(1, int(self._approx_probe_total_entry_blocks)))
            ))
            max_samples_for_group = max(1, min(max_samples_for_group, group_blocks))
            launch_generated_block_probe_for_group(
                m_flat=grp["m_flat"],
                v_flat=grp["v_flat"],
                age2bit_words=age2bit_state.age_words,
                age2bit_floor_mid=age2bit_state.floor_mid,
                age2bit_floor_old=age2bit_state.floor_old,
                age2bit_floor_hist=age2bit_state.floor_hist,
                h_lut_values=runtime_state.quantiles.h_lut.values,
                h_lut_log_v_min=runtime_state.quantiles.h_lut.log_v_min,
                h_lut_inv_log_v_width=runtime_state.quantiles.h_lut.inv_log_v_width,
                entry_elem_start=grp["entry_elem_start"],
                entry_group_start=grp["entry_group_start"],
                entry_group_count=grp["entry_group_count"],
                entry_stratum_idx=grp["entry_stratum_idx"],
                entry_block_entry_idx=grp["entry_block_entry_idx"],
                entry_block_local_group_start=grp["entry_block_local_group_start"],
                candidate_taus=self._selection_candidate_taus,
                sum_q=self._approx_probe_sum_q,
                sum_q2=self._approx_probe_sum_q2,
                n_blocks=self._approx_probe_n_blocks,
                sample_blocks_current=self._approx_probe_sample_blocks_current,
                total_entry_blocks=int(self._approx_probe_total_entry_blocks),
                master_off_elem=int(grp["master_off"]) * int(group_size),
                group_size=int(group_size),
                lr_over_bc1=float(hparams.lr_over_bc1),
                sqrt_bc2_inv=float(hparams.sqrt_bc2_inv),
                eps=float(hparams.eps),
                step=int(global_step),
                random_mode=bool(random_mode),
                random_salt=int(random_salt),
                n_candidates=int(n_cand),
                max_samples_per_group=int(max_samples_for_group),
            )
        t_after_first_probe = time.perf_counter() if profile else 0.0
        if telemetry_enabled:
            for telemetry in telemetry_sinks:
                telemetry.record_probe_done(
                    step=int(global_step),
                    stream=telemetry_stream,
                )

        launch_probe_decision(
            candidate_taus=self._selection_candidate_taus,
                sum_q=self._approx_probe_sum_q,
                sum_q2=self._approx_probe_sum_q2,
                n_blocks=self._approx_probe_n_blocks,
                stratum_total_elems=self._approx_probe_stratum_total_elems_device,
                stratum_expert_idx=self._approx_probe_stratum_expert_idx_device,
                capacity_demand_hint=self._approx_probe_capacity_demand_hint,
                total_th=runtime_state.scratch.total_th,
            probe_reject=self._approx_probe_reject,
            decision_code=self._approx_probe_decision_code,
            sample_blocks_current=self._approx_probe_sample_blocks_current,
            ci_shortfall_streak=self._approx_probe_ci_shortfall_streak,
            diagnostics=self._approx_probe_diagnostics,
            sample_grow_hold=self._approx_probe_sample_grow_hold,
            total_elems_global=int(total_elems_global),
            target_ratio=float(self.topk_ratio),
            cutoff_tolerance=float(self.cutoff_K_tolerance),
            sample_blocks_min=int(self._approx_probe_sample_blocks_min),
            sample_blocks_cap=int(self._approx_probe_sample_blocks_cap),
            ci_shortfall_grow_after=int(_APPROX_PROBE_CI_SHORTFALL_GROW_AFTER),
            n_candidates=int(n_cand),
        )
        t_after_decision1 = time.perf_counter() if profile else 0.0

        retry_n_cand = int(self._selection_candidate_taus.numel())
        launch_retry_candidates_if_code(
            candidate_taus=self._selection_candidate_taus,
            diagnostics=self._approx_probe_diagnostics,
            decision_code=self._approx_probe_decision_code,
            n_candidates=int(retry_n_cand),
            source_n_candidates=int(n_cand),
            target_ratio=float(self.topk_ratio),
            run_if_decision_code=3,
        )
        t_after_retry_candidates = time.perf_counter() if profile else 0.0
        launch_zero_probe_accumulators_if_code(
            sum_q=self._approx_probe_sum_q,
            sum_q2=self._approx_probe_sum_q2,
            n_blocks=self._approx_probe_n_blocks,
            decision_code=self._approx_probe_decision_code,
            run_if_decision_code=3,
        )
        t_after_guarded_zero = time.perf_counter() if profile else 0.0
        for grp in self._groups:
            group_blocks = int(grp.get("approx_entry_block_count", 0))
            if group_blocks <= 0:
                continue
            max_samples_for_group = int(math.ceil(
                float(self._approx_probe_sample_blocks_cap)
                * float(group_blocks)
                / float(max(1, int(self._approx_probe_total_entry_blocks)))
            ))
            max_samples_for_group = max(1, min(max_samples_for_group, group_blocks))
            launch_generated_block_probe_for_group(
                m_flat=grp["m_flat"],
                v_flat=grp["v_flat"],
                age2bit_words=age2bit_state.age_words,
                age2bit_floor_mid=age2bit_state.floor_mid,
                age2bit_floor_old=age2bit_state.floor_old,
                age2bit_floor_hist=age2bit_state.floor_hist,
                h_lut_values=runtime_state.quantiles.h_lut.values,
                h_lut_log_v_min=runtime_state.quantiles.h_lut.log_v_min,
                h_lut_inv_log_v_width=runtime_state.quantiles.h_lut.inv_log_v_width,
                entry_elem_start=grp["entry_elem_start"],
                entry_group_start=grp["entry_group_start"],
                entry_group_count=grp["entry_group_count"],
                entry_stratum_idx=grp["entry_stratum_idx"],
                entry_block_entry_idx=grp["entry_block_entry_idx"],
                entry_block_local_group_start=grp["entry_block_local_group_start"],
                candidate_taus=self._selection_candidate_taus,
                sum_q=self._approx_probe_sum_q,
                sum_q2=self._approx_probe_sum_q2,
                n_blocks=self._approx_probe_n_blocks,
                sample_blocks_current=self._approx_probe_sample_blocks_current,
                total_entry_blocks=int(self._approx_probe_total_entry_blocks),
                master_off_elem=int(grp["master_off"]) * int(group_size),
                group_size=int(group_size),
                lr_over_bc1=float(hparams.lr_over_bc1),
                sqrt_bc2_inv=float(hparams.sqrt_bc2_inv),
                eps=float(hparams.eps),
                step=int(global_step),
                random_mode=bool(random_mode),
                random_salt=int(random_salt),
                n_candidates=int(retry_n_cand),
                max_samples_per_group=int(max_samples_for_group),
                decision_code_guard=self._approx_probe_decision_code,
                run_if_decision_code=3,
            )
        t_after_second_probe = time.perf_counter() if profile else 0.0
        launch_probe_decision(
            candidate_taus=self._selection_candidate_taus,
                sum_q=self._approx_probe_sum_q,
                sum_q2=self._approx_probe_sum_q2,
                n_blocks=self._approx_probe_n_blocks,
                stratum_total_elems=self._approx_probe_stratum_total_elems_device,
                stratum_expert_idx=self._approx_probe_stratum_expert_idx_device,
                capacity_demand_hint=self._approx_probe_capacity_demand_hint,
                total_th=runtime_state.scratch.total_th,
            probe_reject=self._approx_probe_reject,
            decision_code=self._approx_probe_decision_code,
            sample_blocks_current=self._approx_probe_sample_blocks_current,
            ci_shortfall_streak=self._approx_probe_ci_shortfall_streak,
            diagnostics=self._approx_probe_diagnostics,
            sample_grow_hold=self._approx_probe_sample_grow_hold,
            total_elems_global=int(total_elems_global),
            target_ratio=float(self.topk_ratio),
            cutoff_tolerance=float(self.cutoff_K_tolerance),
            sample_blocks_min=int(self._approx_probe_sample_blocks_min),
            sample_blocks_cap=int(self._approx_probe_sample_blocks_cap),
            ci_shortfall_grow_after=int(_APPROX_PROBE_CI_SHORTFALL_GROW_AFTER),
            n_candidates=int(retry_n_cand),
            run_if_decision_code=3,
        )
        t_after_decision2 = time.perf_counter() if profile else 0.0
        if telemetry_enabled:
            for telemetry in telemetry_sinks:
                telemetry.record_decision_done(
                    step=int(global_step),
                    stream=telemetry_stream,
                )
        warning = getattr(self, "_approx_probe_reject_warning", None)
        if warning is not None:
            warning.record(
                step=int(global_step),
                stream=torch.cuda.current_stream(),
                decision_code=self._approx_probe_decision_code,
                probe_reject=self._approx_probe_reject,
                diagnostics=self._approx_probe_diagnostics,
            )
        t_after_feedback_record = time.perf_counter() if profile else 0.0

        if telemetry_enabled:
            for telemetry in telemetry_sinks:
                telemetry.record_device_result(
                    step=int(global_step),
                    stream=telemetry_stream,
                    diagnostics=self._approx_probe_diagnostics,
                    decision_code=self._approx_probe_decision_code,
                    probe_reject=self._approx_probe_reject,
                    n_candidates=int(n_cand),
                    n_strata=int(getattr(self, "_approx_probe_n_strata", 0)),
                    total_entry_blocks=int(
                        getattr(self, "_approx_probe_total_entry_blocks", 0)
                    ),
                    block_elements=int(
                        getattr(self, "_approx_probe_block_elements", group_size)
                    ),
                    sum_q=self._approx_probe_sum_q,
                    sum_q2=self._approx_probe_sum_q2,
                    n_blocks=self._approx_probe_n_blocks,
                    stratum_total_elems=getattr(
                        self, "_approx_probe_stratum_total_elems", []
                    ),
                )
        if profile:
            print(
                f"[proposal] approx_launch_profile @iter{int(global_step)}: "
                f"zero={(t_after_zero - t_entry) * 1000.0:.1f}ms "
                f"probe1={(t_after_first_probe - t_after_zero) * 1000.0:.1f}ms "
                f"decision1={(t_after_decision1 - t_after_first_probe) * 1000.0:.1f}ms "
                f"retry_candidates={(t_after_retry_candidates - t_after_decision1) * 1000.0:.1f}ms "
                f"retry_zero={(t_after_guarded_zero - t_after_retry_candidates) * 1000.0:.1f}ms "
                f"probe2={(t_after_second_probe - t_after_guarded_zero) * 1000.0:.1f}ms "
                f"decision2={(t_after_decision2 - t_after_second_probe) * 1000.0:.1f}ms "
                f"feedback_record={(t_after_feedback_record - t_after_decision2) * 1000.0:.1f}ms",
                flush=True,
            )
        return "gpu_probe"

    def prepare_selection_state(
        self,
        optimizer: torch.optim.Optimizer,
        global_step: int,
    ) -> None:
        """Refresh Adam scalars and quantile snapshots for this proposal iter."""
        self._init_lossy_flat_layout(optimizer)
        if not getattr(self, "_flat_layout_done", False):
            return
        # Adam hyper-params
        beta1 = 0.9
        beta2 = 0.999
        eps_adam = 1e-8
        lr = 1.0
        for grp in optimizer.param_groups:
            if "betas" in grp:
                b = grp["betas"]
                if isinstance(b, (tuple, list)) and len(b) >= 2:
                    beta1 = float(b[0])
                    beta2 = float(b[1])
            if "lr" in grp:
                lr = float(grp["lr"])
            if "eps" in grp:
                eps_adam = float(grp["eps"])
            break
        gs = max(int(global_step), 1)
        bc1 = max(1.0 - beta1 ** gs, 1e-12)
        bc2 = max(1.0 - beta2 ** gs, 1e-12)
        sqrt_bc2_inv = (1.0 / (bc2 ** 0.5))
        lr_over_bc1 = lr / bc1

        # Adam scalars consumed by the current iter's lossy record writer.
        self._adam_lr_over_bc1_for_selection = float(lr_over_bc1)
        self._adam_sqrt_bc2_inv_for_selection = float(sqrt_bc2_inv)
        self._adam_eps_for_selection = float(eps_adam)

        selection_ready = all(grp["zero_copy"] for grp in self._groups) and _HAS_TRITON
        if not selection_ready:
            raise RuntimeError(
                "V2 lossy CP requires Triton zero-copy BF16 optimizer views."
            )

        # Selection pipeline: GPU approximate cutoff chooses total_th; direct
        # emission later applies the same score predicate and cursor cap.
        device = self._selection_hist.device
        if self._selection_total_th is None:
            self._selection_total_th = torch.full(
                (), float("inf"), dtype=torch.float32, device=device,
            )

        _ofs_t = getattr(self, "_group_offsets_t", None)
        if _ofs_t is None:
            _ofs_t = device_tensor_from_pinned_host(
                self._group_offsets_cpu, dtype=torch.long,
                device=device,
            )
            self._group_offsets_t = _ofs_t
        if (self._elem_offsets_for_selection is None
                or self._elem_offsets_for_selection.numel() != _ofs_t.numel()):
            self._elem_offsets_for_selection = (
                _ofs_t.to(torch.int64) * max(1, int(self.group_size))
            ).contiguous()
        self._selection_v_q_for_write = self._v_quantile_values
        self._selection_h_q_for_write = self._h_quantile_values

    def _find_selection_cutoff_for_current_iter(
        self,
        *,
        global_step: int,
        group_size: int,
        lr_over_bc1: float,
        sqrt_bc2_inv: float,
        eps_adam: float,
        stream: Optional["torch.cuda.Stream"],
    ) -> Optional["torch.cuda.Event"]:
        """Choose τ on device and leave materialization to direct emission."""

        profile = os.environ.get("PROPOSAL_SELECTION_PHASE_PROFILE", "") in (
            "1", "true", "TRUE",
        )
        t_entry = time.perf_counter() if profile else 0.0
        if not getattr(self, "_flat_layout_done", False):
            return None
        from ..emission.record_kernels import AdamHParams
        runtime_state = self.lossy_cp_state
        if runtime_state is None:
            self._refresh_lossy_cp_state_refs()
            runtime_state = self.lossy_cp_state
        if runtime_state is None:
            return None

        world = int(getattr(self, "_proposal_world_size", 1))
        pg = getattr(self, "_proposal_pg", None)
        total_elems_global = self._global_selection_element_count_cached(
            int(runtime_state.index.n_total_elements),
            process_group=pg,
            world_size=world,
        )
        t_after_global = time.perf_counter() if profile else 0.0
        self._consume_direct_emission_feedback(
            total_elems_global=int(total_elems_global),
        )
        t_after_feedback = time.perf_counter() if profile else 0.0
        hparams = AdamHParams(
            lr_over_bc1=float(lr_over_bc1),
            sqrt_bc2_inv=float(sqrt_bc2_inv),
            eps=float(eps_adam),
        )
        random_mode = bool(self.selection_mode == "random")
        random_salt = (
            int(self._random_seed_salt)
            ^ ((int(global_step) * 0x9E3779B9) & 0xFFFFFFFF)
        ) & 0xFFFFFFFF
        self._selection_random_mode = bool(random_mode)
        self._selection_random_salt = int(random_salt)
        self._selection_direct_emission_will_fold = bool(
            self._expert_state is not None
            and not bool(getattr(self, "test_expert_selection_only", False))
        )

        cutoff_decision = self._adaptive_cutoff.decide(step=int(global_step))
        tau_center = self._initial_cutoff_center(cutoff_decision)
        exact_reason = str(getattr(cutoff_decision, "reason", ""))
        initial_cold_start = int(global_step) <= 1
        use_cold_start_hist = (
            bool(getattr(cutoff_decision, "use_exact", False))
            and bool(initial_cold_start)
            and not bool(getattr(self, "_cold_start_cutoff_ready", False))
        )
        if use_cold_start_hist:
            cold_start = self._cold_start_cutoff_bootstrap(
                runtime_state=runtime_state,
                hparams=hparams,
                total_elems_global=int(total_elems_global),
                group_size=int(group_size),
                process_group=pg,
                world_size=int(world),
                random_mode=bool(random_mode),
                random_salt=int(random_salt),
            )
            if cold_start is not None:
                tau_center, cold_log_width = cold_start
                self._cold_start_cutoff_ready = True
                candidates = self._cold_start_candidate_ladder(
                    tau=float(tau_center),
                    log_bin_width=float(cold_log_width),
                )
            else:
                candidates = []
            t_after_decide = time.perf_counter() if profile else 0.0
        else:
            candidates = self._adaptive_cutoff.candidate_taus(
                float(tau_center),
                capacity=int(getattr(self, "_selection_candidate_capacity", 16)),
            )
            if (
                bool(getattr(cutoff_decision, "use_exact", False))
                and exact_reason == "insufficient_window"
                and not candidates
            ):
                candidates = self._cold_start_candidate_ladder(
                    tau=float(tau_center),
                    log_bin_width=0.25,
                )
            t_after_decide = time.perf_counter() if profile else 0.0
        if not candidates:
            candidates = [float(max(tau_center, 1.0e-12))]
        n_candidates = self._stage_candidate_taus(
            candidates,
            tau_center=float(tau_center),
        )
        t_after_stage = time.perf_counter() if profile else 0.0
        self._launch_gpu_approx_cutoff(
            global_step=int(global_step),
            runtime_state=runtime_state,
            hparams=hparams,
            n_candidates=int(n_candidates),
            total_elems_global=int(total_elems_global),
            group_size=int(group_size),
            process_group=pg,
            world_size=int(world),
            random_mode=bool(random_mode),
            random_salt=int(random_salt),
        )
        t_after_probe = time.perf_counter() if profile else 0.0
        cache_telemetry = getattr(self, "_sparse_cache_shadow_telemetry", None)
        if cache_telemetry is not None and getattr(cache_telemetry, "enabled", False):
            cache_telemetry.record(
                step=int(global_step),
                topk_ratio=float(self.topk_ratio),
                groups=self._groups,
                h_lut_values=runtime_state.quantiles.h_lut.values,
                h_lut_log_v_min=runtime_state.quantiles.h_lut.log_v_min,
                h_lut_inv_log_v_width=runtime_state.quantiles.h_lut.inv_log_v_width,
                group_size=int(group_size),
                lr_over_bc1=float(hparams.lr_over_bc1),
                sqrt_bc2_inv=float(hparams.sqrt_bc2_inv),
                eps=float(hparams.eps),
                stream=stream,
            )
        policy_telemetry = getattr(self, "_policy_shadow_telemetry", None)
        if policy_telemetry is not None and getattr(policy_telemetry, "enabled", False):
            policy_telemetry.record(
                step=int(global_step),
                topk_ratio=float(self.topk_ratio),
                groups=self._groups,
                h_lut_values=runtime_state.quantiles.h_lut.values,
                h_lut_log_v_min=runtime_state.quantiles.h_lut.log_v_min,
                h_lut_inv_log_v_width=runtime_state.quantiles.h_lut.inv_log_v_width,
                group_size=int(group_size),
                lr_over_bc1=float(hparams.lr_over_bc1),
                sqrt_bc2_inv=float(hparams.sqrt_bc2_inv),
                eps=float(hparams.eps),
                stream=stream,
            )
        t_after_telemetry = time.perf_counter() if profile else 0.0
        self._selection_total_th_host = float(tau_center)
        ev = self._acquire_ring_event("selection_count")
        ev.record(stream if stream is not None else torch.cuda.current_stream())
        if profile:
            print(
                f"[proposal] selection_phase_profile @iter{int(global_step)}: "
                f"global={((t_after_global - t_entry) * 1000.0):.1f}ms "
                f"feedback={((t_after_feedback - t_after_global) * 1000.0):.1f}ms "
                f"decide={((t_after_decide - t_after_feedback) * 1000.0):.1f}ms "
                f"stage={((t_after_stage - t_after_decide) * 1000.0):.1f}ms "
                f"probe={((t_after_probe - t_after_stage) * 1000.0):.1f}ms "
                f"telemetry={((t_after_telemetry - t_after_probe) * 1000.0):.1f}ms "
                f"candidates={int(n_candidates)}",
                flush=True,
            )
        return ev
