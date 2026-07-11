"""Direct-emission orchestration for lossy checkpoint writes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

_APPROX_PROBE_COUNT_MISMATCH_GROW_AFTER = 3


class EmissionOrchestrationMixin:
    def _static_direct_write_inputs(self) -> Dict[str, Any]:
        cached = getattr(self, "_direct_write_static_inputs", None)
        if cached is not None:
            return cached
        group_size = max(1, int(self.group_size))
        name_list = tuple(self._tname_list)
        offsets_cpu = tuple(int(x) for x in self._group_offsets_cpu)
        name_cnt_g: List[int] = []
        total_records = 0
        for i in range(len(name_list)):
            cnt_g_elem = (int(offsets_cpu[i + 1]) - int(offsets_cpu[i])) * group_size
            name_cnt_g.append(int(cnt_g_elem))
            total_records += int(cnt_g_elem)

        per_expert_batch: Dict[Tuple[int, int], List[Tuple[str, int, int, int]]] = {}
        for gi, grp in enumerate(self._groups):
            sn = int(grp["start_name_in_grp"])
            en = int(grp["end_name_in_grp"])
            cum_g = 0
            for i in range(sn, en):
                name = name_list[i]
                cg = int(name_cnt_g[i])
                if cg > 0:
                    eid = self._tname_to_eid.get(name)
                    if eid is not None:
                        per_expert_batch.setdefault(eid, []).append((name, gi, cum_g, cg))
                cum_g += cg
        per_expert_batch = {
            eid: per_expert_batch[eid]
            for eid in sorted(per_expert_batch)
        }
        write_expert_ids = tuple(per_expert_batch)
        hint_expert_ids = tuple(getattr(self, "_local_expert_ids", ()))
        if hint_expert_ids != write_expert_ids:
            raise RuntimeError(
                "capacity-demand hint expert order does not match the "
                "rank-local direct-emission layout: "
                f"hint={hint_expert_ids[:8]} ({len(hint_expert_ids)} slots), "
                f"write={write_expert_ids[:8]} ({len(write_expert_ids)} slots)"
            )

        mvph: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        theta_cp_groups: List[Optional[torch.Tensor]] = []
        gpu_refs: List[torch.Tensor] = []
        for grp in self._groups:
            p_flat_g = grp.get("p_flat")
            if (not grp.get("zero_copy", False)) or p_flat_g is None:
                raise RuntimeError(
                    "LossyRecordWriteInputs requires zero-copy m/v/p views "
                    "for every BF16 optimizer group."
                )
            mvph.append((
                grp["m_flat"], grp["v_flat"], p_flat_g,
                int(grp["master_off"]) * group_size,
            ))
            theta_cp_g = grp.get("theta_cp_flat")
            theta_cp_groups.append(theta_cp_g)
            gpu_refs.extend((p_flat_g, grp["m_flat"], grp["v_flat"]))
            if theta_cp_g is not None:
                gpu_refs.append(theta_cp_g)

        element_within_off_host: List[int] = []
        element_age_off_host: List[int] = []
        group_count_per_name_host: List[int] = []
        for grp_init_w in self._groups:
            sn_iw = int(grp_init_w["start_name_in_grp"])
            en_iw = int(grp_init_w["end_name_in_grp"])
            phys_starts = list(grp_init_w.get("entry_elem_start_host", ()))
            age_starts = list(grp_init_w.get("entry_age_elem_start_host", ()))
            for entry_iw, name_iw in enumerate(range(sn_iw, en_iw)):
                element_within_off_host.append(int(phys_starts[entry_iw]))
                element_age_off_host.append(int(age_starts[entry_iw]))
                group_count_per_name_host.append(
                    int(offsets_cpu[name_iw + 1]) - int(offsets_cpu[name_iw])
                )

        cached = {
            "name_list": name_list,
            "per_expert_batch": per_expert_batch,
            "capacity_hint_expert_ids": hint_expert_ids,
            "mvph": tuple(mvph),
            "theta_cp_groups": tuple(theta_cp_groups),
            "within_off_per_name": tuple(element_within_off_host),
            "age_off_per_name": tuple(element_age_off_host),
            "group_count_per_name": tuple(group_count_per_name_host),
            "gpu_refs": tuple(gpu_refs),
            "total_records": int(total_records),
        }
        self._direct_write_static_inputs = cached
        return cached

    def _update_direct_write_device_scalars(
        self,
        *,
        device: torch.device,
        step: int,
        hparams: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fp32_host = getattr(self, "_direct_write_fp32_scalar_host", None)
        fp32_dev = getattr(self, "_direct_write_fp32_scalar_device", None)
        step_host = getattr(self, "_direct_write_step_host", None)
        step_dev = getattr(self, "_direct_write_step_device", None)
        if (
            fp32_host is None
            or fp32_dev is None
            or step_host is None
            or step_dev is None
            or tuple(fp32_host.shape) != (3,)
            or tuple(step_host.shape) != (1,)
            or fp32_dev.device != device
            or step_dev.device != device
            or tuple(fp32_dev.shape) != (3,)
            or tuple(step_dev.shape) != (1,)
        ):
            fp32_host = torch.empty(3, dtype=torch.float32, pin_memory=True)
            step_host = torch.empty(1, dtype=torch.int32, pin_memory=True)
            fp32_dev = torch.empty(3, dtype=torch.float32, device=device)
            step_dev = torch.empty(1, dtype=torch.int32, device=device)
            self._direct_write_fp32_scalar_host = fp32_host
            self._direct_write_step_host = step_host
            self._direct_write_fp32_scalar_device = fp32_dev
            self._direct_write_step_device = step_dev
        fp32_host[0] = float(hparams.lr_over_bc1)
        fp32_host[1] = float(hparams.sqrt_bc2_inv)
        fp32_host[2] = float(hparams.eps)
        step_host[0] = int(step)
        stream = self._compute_stream
        if stream is not None:
            with torch.cuda.stream(stream):
                fp32_dev.copy_(fp32_host, non_blocking=True)
                step_dev.copy_(step_host, non_blocking=True)
        else:
            fp32_dev.copy_(fp32_host, non_blocking=True)
            step_dev.copy_(step_host, non_blocking=True)
        return fp32_dev, step_dev

    def _snapshot_total_th_for_direct_write(
        self,
        *,
        device: torch.device,
        step: int,
        total_th: torch.Tensor,
    ) -> torch.Tensor:
        ring = getattr(self, "_direct_write_total_th_ring_device", None)
        if (
            ring is None
            or ring.device != device
            or ring.dtype != torch.float32
            or tuple(ring.shape) != (1,)
        ):
            ring = torch.empty(1, dtype=torch.float32, device=device)
            self._direct_write_total_th_ring_device = ring
        stream = self._compute_stream
        src = total_th.reshape(1)
        if stream is not None:
            with torch.cuda.stream(stream):
                ring.copy_(src, non_blocking=True)
        else:
            ring.copy_(src, non_blocking=True)
        return ring

    def _write_lossy_records_for_current_iter(
        self, optimizer: torch.optim.Optimizer,
        lossy_write_inputs: Dict[str, Any],
        selection_event: Optional["torch.cuda.Event"] = None,
        d2h_event_callback: Optional[Any] = None,
    ) -> None:
        """Submit one direct-emission lossy-record iteration.

        Selection is not materialized as host counts or masks here. The GPU
        writer scans static tname spans and the per-expert cursor cap is the
        hard overflow guard.
        """
        pending = lossy_write_inputs
        prev_step = pending["step"]
        direct_emission_enabled = bool(
            pending.get("direct_emission_enabled", False)
        )
        if (
            not bool(direct_emission_enabled)
            and self._expert_state is not None
            and not bool(getattr(self, "test_expert_selection_only", False))
        ):
            raise RuntimeError("final lossy checkpoint path requires direct emission")
        group_size = max(1, int(self.group_size))
        stream = self._compute_stream

        static_inputs = self._static_direct_write_inputs()
        gpu_refs_for_writer = list(static_inputs["gpu_refs"])
        age2bit_state = getattr(self, "_age2bit_score_state", None)
        if age2bit_state is not None:
            gpu_refs_for_writer.extend((
                age2bit_state.age_words,
                age2bit_state.ehat_geom_lut,
            ))

        if bool(getattr(self, "test_expert_selection_only", False)):
            if d2h_event_callback is not None:
                if torch.cuda.is_available():
                    event_no_aos = self._acquire_ring_event("compute")
                    event_no_aos.record(
                        stream if stream is not None else torch.cuda.current_stream()
                    )
                    d2h_event_callback(event_no_aos)
                else:
                    d2h_event_callback(None)
            return

        if self._expert_state is not None:
            _lossy_state = None

            from ..emission.record_kernels import LossyRecordWriteInputs
            hparams = pending.get("adam_hparams")
            scalar_dev, step_dev = self._update_direct_write_device_scalars(
                device=next(self.model.parameters()).device,
                step=int(prev_step),
                hparams=hparams,
            )
            total_th_for_write = pending.get("total_th")
            if total_th_for_write is not None:
                total_th_for_write = self._snapshot_total_th_for_direct_write(
                    device=next(self.model.parameters()).device,
                    step=int(prev_step),
                    total_th=total_th_for_write,
                )
            total_elems_for_actual_k = 0
            runtime_state = self.lossy_cp_state
            if runtime_state is not None:
                total_elems_for_actual_k = self._global_selection_element_count_cached(
                    int(runtime_state.index.n_total_elements),
                    process_group=getattr(self, "_proposal_cpu_pg", None),
                    world_size=int(getattr(self, "_proposal_world_size", 1)),
                )
            target_selected_count = int(round(
                float(self.topk_ratio) * float(max(1, total_elems_for_actual_k))
            ))
            allowed_probe_count_delta = int(math.ceil(
                float(target_selected_count) * float(self.cutoff_K_tolerance)
            ))
            _lossy_state = LossyRecordWriteInputs(
                bf16_groups_mvph=static_inputs["mvph"],
                name_list=static_inputs["name_list"],
                group_size=int(group_size),
                theta_cp_groups=static_inputs["theta_cp_groups"],
                within_off_per_name=static_inputs["within_off_per_name"],
                age_off_per_name=static_inputs["age_off_per_name"],
                group_count_per_name=static_inputs["group_count_per_name"],
                h_lut_values=pending.get("h_lut_values"),
                h_lut_log_v_min=pending.get("h_lut_log_v_min"),
                h_lut_inv_log_v_width=pending.get("h_lut_inv_log_v_width"),
                total_th=total_th_for_write,
                adam_hparams=hparams,
                adam_scalar_device=scalar_dev,
                step_device=step_dev,
                random_mode=bool(pending.get("random_mode", False)),
                random_salt=int(pending.get("random_salt", 0)),
                tau_version=int(pending.get("tau_version", -1)),
                tau_h_epoch=int(pending.get("tau_h_epoch", -1)),
                tau_source_step=int(pending.get("tau_source_step", -1)),
                tau_apply_from_step=int(pending.get("tau_apply_from_step", -1)),
                tau_lr_over_bc1=float(pending.get("tau_lr_over_bc1", 0.0)),
                direct_emission_enabled=bool(
                    pending.get("direct_emission_enabled", False)
                ),
                sample_blocks_current=self._approx_probe_sample_blocks_current,
                probe_count_mismatch_streak=self._approx_probe_count_mismatch_streak,
                sample_grow_hold=self._approx_probe_sample_grow_hold,
                ci_shortfall_streak=self._approx_probe_ci_shortfall_streak,
                probe_diagnostics=self._approx_probe_diagnostics,
                probe_expected_selected_count=(
                    self._approx_probe_expected_selected_count
                ),
                capacity_demand_hint=self._approx_probe_capacity_demand_hint,
                capacity_demand_hint_expert_ids=static_inputs[
                    "capacity_hint_expert_ids"
                ],
                total_elems_global=int(total_elems_for_actual_k),
                allowed_probe_count_delta=max(1, int(allowed_probe_count_delta)),
                sample_blocks_min=int(self._approx_probe_sample_blocks_min),
                sample_blocks_cap=int(self._approx_probe_sample_blocks_cap),
                probe_count_mismatch_grow_after=int(
                    _APPROX_PROBE_COUNT_MISMATCH_GROW_AFTER
                ),
                enable_direct_probe_sample_feedback=True,
                age2bit_words=(
                    age2bit_state.age_words if age2bit_state is not None else None
                ),
                ehat_geom_lut=(
                    age2bit_state.ehat_geom_lut if age2bit_state is not None else None
                ),
                use_e_hat_proxy=bool(getattr(self, "e_hat_proxy", True)),
                score_policy_id=int(getattr(self, "score_policy_id", 0)),
                lfcl_fired_epoch=getattr(self, "_lfcl_fired_epoch_device", None),
            )

            # Phase 3: compute_event = end-of-Phase-A + metadata handoff
            # signal on compute_stream. The expert worker waits on this
            # before launching AOS emission, so it sees both payload tensors and
            # per-name slot metadata in a consistent state.
            event_compute: Optional["torch.cuda.Event"] = None
            if stream is not None and int(static_inputs["total_records"]) > 0:
                event_compute = self._acquire_ring_event("compute")
                event_compute.record(stream)

            self._expert_state.write_for_iter(
                step=int(prev_step),
                per_expert_updates=static_inputs["per_expert_batch"],
                compute_event=event_compute,
                gpu_refs=gpu_refs_for_writer,
                lossy_state=_lossy_state,
                d2h_event_callback=d2h_event_callback,
            )
        else:
            return
