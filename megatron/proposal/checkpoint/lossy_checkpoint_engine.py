"""Element-wise expert lossy checkpoint selection.

The active path ranks elements with one score:

    default score(i) = max(0, -m_i * e_hat_i + 0.5 * |H_i| * e_hat_i^2)

``LossyCpState`` holds the runtime views used by cutoff search and record
write. ``LossyCheckpointPipeline`` owns one lossy checkpoint iteration.
"""

from __future__ import annotations

import math
import time
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from ..cutoff.adaptive_controller import CutoffDecision
from ..cutoff.cutoff_orchestrator import CutoffOrchestrationMixin
from ..emission.emission_orchestrator import EmissionOrchestrationMixin
from ..refresh.fp32_state import get_mapper
from ..refresh.refresh_orchestrator import RefreshOrchestrationMixin
from .static_descriptors import (
    StaticDescriptorArena,
    device_tensor_from_pinned_host,
)
from .step_context import (
    LossyStepStats,
    fp32_param,
    fp32_v_t,
)
from ..expert_id import ids_from_name
from ..storage.expert_partition import build_expert_partition_plan

_APPROX_PROBE_DEFAULT_SAMPLE_FRACTION = 0.01
_APPROX_PROBE_MAX_SAMPLE_FRACTION = 0.05

from ..storage.expert_writer import ExpertState


class LossyCheckpointPipeline(
    RefreshOrchestrationMixin,
    CutoffOrchestrationMixin,
    EmissionOrchestrationMixin,
):
    """Owner for expert lossy checkpoint selection and write state."""

    def __init__(
        self,
        model: torch.nn.Module,
        topk_ratio: float = 0.01,
        include_filter: Optional[Iterable[str]] = None,
        exclude_filter: Optional[Iterable[str]] = None,
        expert_only: bool = True,
        expert_pattern: Optional[str] = None,
        selection_mode: str = "hessian",
        num_experts: Optional[int] = None,
        random_seed: int = 0,
        checkpoint_workers: Optional[Any] = None,
        group_size: int = 1,
        proposal_config: Optional[Any] = None,
        expert_id_map: Optional[Mapping[str, Tuple[int, int]]] = None,
    ) -> None:
        self._random_seed_salt = int(random_seed) * 2654435761 & 0xFFFFFFFF
        self.model = model
        self.proposal_config = proposal_config
        self._expert_id_map = expert_id_map
        self.topk_ratio = float(topk_ratio)
        if not (0.0 < self.topk_ratio <= 1.0):
            raise ValueError(
                f"topk_ratio must satisfy 0 < topk_ratio <= 1, got {self.topk_ratio!r}"
            )
        self.group_size = max(1, int(group_size))
        self.n_bins_cutoff = max(2, int(
            getattr(proposal_config, "n_bins_cutoff", 4096)
            if proposal_config is not None else 4096
        ))
        self.cutoff_K_tolerance = float(
            getattr(proposal_config, "cutoff_K_tolerance", 0.015)
            if proposal_config is not None else 0.015
        )
        self.n_q = int(
            getattr(proposal_config, "n_q", 4096)
            if proposal_config is not None else 4096
        )
        self.n_bins_quantile_log = int(
            getattr(proposal_config, "n_bins_quantile_log", 1024)
            if proposal_config is not None else 1024
        )
        self.h_lookup_lut_bins = int(
            getattr(proposal_config, "h_lookup_lut_bins", 8192)
            if proposal_config is not None else 8192
        )
        self._last_vq_refresh_step: Optional[int] = None
        self._last_refresh_bridge_scale: Optional[float] = None
        self._last_refresh_bridge_reason: str = ""
        from ..cutoff.adaptive_controller import (
            AdaptiveCutoffConfig,
            AdaptiveCutoffController,
        )
        self._adaptive_cutoff = AdaptiveCutoffController(
            AdaptiveCutoffConfig(
                enabled=True,
                target_ratio=float(topk_ratio),
                tolerance=float(self.cutoff_K_tolerance),
                window_size=int(
                    getattr(proposal_config, "adaptive_cutoff_window", 16)
                    if proposal_config is not None else 16
                ),
                min_observations=int(
                    getattr(proposal_config, "adaptive_cutoff_min_observations", 8)
                    if proposal_config is not None else 8
                ),
                warmup_iters=int(
                    getattr(proposal_config, "adaptive_cutoff_warmup_iters", 0)
                    if proposal_config is not None else 0
                ),
                retry_tolerance=float(
                    getattr(
                        proposal_config,
                        "adaptive_cutoff_retry_tolerance",
                        0.02,
                    )
                    if proposal_config is not None else 0.02
                ),
            )
        )
        from ..telemetry.approx_probe import ApproxProbeTelemetry
        self._approx_probe_telemetry = ApproxProbeTelemetry.from_env()
        from ..telemetry.sparse_cache_shadow import SparseCacheShadowTelemetry
        self._sparse_cache_shadow_telemetry = SparseCacheShadowTelemetry.from_env()
        from ..telemetry.policy_shadow import PolicyShadowTelemetry
        self._policy_shadow_telemetry = PolicyShadowTelemetry.from_env()
        from ..telemetry.curvature_progress import CurvatureProgressTelemetry
        self._curvature_progress_telemetry = CurvatureProgressTelemetry.from_env()
        from ..cutoff.adaptive_controller import ApproxProbeRejectWarning
        self._approx_probe_reject_warning = ApproxProbeRejectWarning()
        self._checkpoint_workers = checkpoint_workers
        self.expert_only = bool(expert_only)
        self.expert_pattern = expert_pattern
        if selection_mode not in ("v_hat", "random", "hessian"):
            raise ValueError(
                f"selection_mode must be 'v_hat'|'random'|'hessian', got {selection_mode!r}."
            )
        self.selection_mode = selection_mode
        from ..score_policy import DEFAULT_SCORE_POLICY, score_policy_id
        self.score_policy = str(
            getattr(proposal_config, "score_policy", DEFAULT_SCORE_POLICY)
            if proposal_config is not None else DEFAULT_SCORE_POLICY
        )
        self.score_policy_id = int(score_policy_id(self.score_policy))
        self.e_hat_proxy = bool(
            getattr(proposal_config, "e_hat_proxy", True)
            if proposal_config is not None else True
        )
        self._current_global_step_for_selection = -1
        self._theta_cp_score_state = None
        # Production reads H through the quantile table. In hessian mode the
        # single hot-path score is the configured abs-H Taylor policy.  Probe
        # and direct emission must use the same helper/LUT snapshot; group-dg,
        # vbin-floor, and outlier-priority scoring are not production paths.

        self._selection_total_th: Optional[torch.Tensor] = None    # () fp32
        self._selection_total_th_host: float = 1.0
        self._selection_tau_meta: Dict[str, Any] = {}
        async_queue = int(
            getattr(proposal_config, "async_queue", 5)
            if proposal_config is not None else 5
        )
        self._direct_write_total_th_ring_device: Optional[torch.Tensor] = None
        self._global_selection_element_count_host: Optional[int] = None
        self._local_selection_element_count_host: Optional[int] = None
        self._selection_candidate_capacity = 16
        self._cold_start_hist_iters = 4
        self._cold_start_first_global_step = 1
        self.test_expert_selection_only = bool(
            getattr(proposal_config, "test_expert_selection_only", False)
            if proposal_config is not None else False
        )
        self._adaptive_skip_warmup_iters = max(
            0,
            int(
                getattr(
                    proposal_config,
                    "adaptive_cutoff_skip_warmup_iters",
                    0,
                )
                if proposal_config is not None else 0
            ),
        )

        self.num_experts = int(num_experts) if num_experts is not None else None
        self.lossy_cp_state = None
        self._h_lookup_lut = None
        from ..emission.record_kernels import LossyRecordWriter
        self._lossy_record_writer = LossyRecordWriter(
            self._write_lossy_records_for_current_iter,
        )

        self._tracked_names: List[str] = []
        self._name_to_param: Dict[str, torch.nn.Parameter] = {}

        self._full_save_names: List[str] = []
        self._name_to_full_save_param: Dict[str, torch.nn.Parameter] = {}
        self._full_save_numel_per_name: Dict[str, int] = {}

        any_ds_tagged = any(
            getattr(p, "allreduce", True) is False
            for _, p in model.named_parameters()
            if p.requires_grad
        )
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if self.expert_only and any_ds_tagged:
                is_expert = getattr(p, "allreduce", True) is False
            else:
                include_ok = (
                    include_filter is None
                    or any(k in name for k in include_filter)
                    or (self.expert_pattern is not None and self.expert_pattern in name)
                )
                exclude_ok = (
                    exclude_filter is None
                    or not any(k in name for k in exclude_filter)
                )
                is_expert = include_ok and exclude_ok
            if is_expert:
                self._tracked_names.append(name)
                self._name_to_param[name] = p
            else:
                self._full_save_names.append(name)
                self._name_to_full_save_param[name] = p
                self._full_save_numel_per_name[name] = int(p.numel())

        self._flat_offsets: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        for name in self._tracked_names:
            p = self._name_to_param[name]
            self._flat_offsets[name] = (cursor, cursor + p.numel())
            cursor += p.numel()
        self._flat_total = cursor

        self._flat_offset_list: List[Tuple[str, int, int]] = [
            (n, s, e) for n, (s, e) in self._flat_offsets.items()
        ]

        # compute_stream for all per-iter proposal kernels.
        self._compute_stream: Optional["torch.cuda.Stream"] = None
        # ``_gpu_payload_entries`` maps each tracked name to a list of
        # ``(bf16_group_idx, src_off_in_group, n_records)`` tuples.
        self._gpu_payload_entries: Optional[Dict[
            str, List[Tuple[int, int, int]]
        ]] = None

        self.total_tracked = self._flat_total
        self.total_full_save = sum(self._full_save_numel_per_name.values())
        self.full_save_bytes = self.total_full_save * 4

        if self.expert_only and not self._tracked_names:
            raise RuntimeError(
                "LossyCheckpointPipeline: expert_only=True but no expert params detected."
            )

        # Precompute (layer, expert_idx) per tracked tensor for rolling-state
        # scatter updates in .step(). O(num_tracked_tensors), runs once.
        self._tname_to_eid: Dict[str, Tuple[int, int]] = {}
        for name in self._tracked_names:
            eid = ids_from_name(name, self._expert_id_map)
            if eid is not None:
                self._tname_to_eid[name] = eid

        # ExpertState sink (set via set_expert_state). When present,
        # step() appends each expert's selected (indices, p, m, v) tuples
        # to the per-expert lossy FC append log.
        self._expert_state: Optional[ExpertState] = None

    # ------------------------------------------------------------------
    def _ensure_h_lookup_lut(self, device: Optional[torch.device] = None):
        """Allocate the small persistent log(v)->H LUT once."""

        if self._h_lookup_lut is not None:
            return self._h_lookup_lut
        if device is None:
            device = self._v_quantile_values.device
        from ..refresh.quantile import HLookupLut

        self._h_lookup_lut = HLookupLut(
            values=torch.empty(
                self.h_lookup_lut_bins, dtype=torch.float32, device=device,
            ),
            log_v_min=torch.zeros((), dtype=torch.float32, device=device),
            inv_log_v_width=torch.zeros((), dtype=torch.float32, device=device),
        )
        self._h_lookup_lut.values.fill_(1.0)
        return self._h_lookup_lut

    # ------------------------------------------------------------------
    def set_expert_state(
        self, expert_state: Optional[ExpertState],
    ) -> None:
        """Install an ExpertState sink. When set, ``step()`` routes each
        expert's top-K selections into ``expert_state.append_lossy_records_for_test(...)``.
        Pass ``None`` to detach.
        """
        self._expert_state = expert_state

    def publish_lfcl_fired_experts(
        self,
        *,
        step: int,
        expert_ids: Iterable[Tuple[int, int]],
    ) -> None:
        """Publish LFCL-fired local experts for the current lossy pass.

        The device mask is epoch based, so old fired entries do not need a
        per-iter clear.  Kernels only skip entries whose epoch equals ``step``.
        """

        host = getattr(self, "_lfcl_fired_epoch_host", None)
        device = getattr(self, "_lfcl_fired_epoch_device", None)
        local_idx = getattr(self, "_local_expert_to_idx", {})
        if host is None or device is None or not local_idx:
            return
        wrote = False
        for eid_raw in expert_ids:
            eid = (int(eid_raw[0]), int(eid_raw[1]))
            idx = local_idx.get(eid)
            if idx is None:
                continue
            host[int(idx)] = int(step)
            wrote = True
        if wrote:
            if device.is_cuda and torch.cuda.is_available():
                if self._compute_stream is None:
                    from ..runtime import get_compute_stream
                    self._compute_stream = get_compute_stream()
                stream = self._compute_stream
                if stream is not None:
                    with torch.cuda.stream(stream):
                        device.copy_(host, non_blocking=True)
                    return
            device.copy_(host, non_blocking=True)

    def _active_expert_ids_for_current_layout(self) -> Tuple[Tuple[int, int], ...]:
        """Return expert slots represented by the active rank-local layout.

        Original parameter names remain in ``_tname_to_eid`` after a ZeRO
        partition plan is installed.  Restricting the lookup to
        ``_tname_list`` excludes experts for which this DP rank owns no flat
        fragment and makes the order identical to the sorted direct-write
        batch.
        """

        return tuple(sorted({
            self._tname_to_eid[name]
            for name in self._tname_list
            if name in self._tname_to_eid
        }))

    def _build_approx_probe_metadata(self, *, device: torch.device) -> None:
        """Build static metadata for GPU-generated block-stratified probe.

        The production probe does not keep per-iter sample tables. Each kernel
        launch derives sampled entry-block ordinals directly from the local
        contiguous entry-block population of each optimizer flat group.
        """

        # ``_tname_to_eid`` intentionally retains the original model names so
        # recovery can resolve them.  Under ZeRO-1/2, however, some of those
        # experts have no fragment on this DP rank.  Capacity hints must use
        # the exact same expert slots as direct emission, which are represented
        # by the active, partitioned ``_tname_list`` only.
        local_expert_ids = self._active_expert_ids_for_current_layout()
        self._local_expert_ids = tuple(local_expert_ids)
        self._local_expert_to_idx = {
            eid: idx for idx, eid in enumerate(local_expert_ids)
        }
        eid_to_expert_idx = self._local_expert_to_idx
        stratum_total_elems: List[int] = []
        stratum_total_blocks: List[int] = []
        stratum_expert_idx: List[int] = []
        total_entry_blocks = 0
        g = max(1, int(self.group_size))
        groups_per_program = max(1, 256 // g)
        power = 1
        while power < groups_per_program:
            power *= 2
        groups_per_program = int(power)
        self._approx_probe_block_elements = int(groups_per_program * g)

        descriptor_arena = StaticDescriptorArena(device=device)
        for grp in self._groups:
            entry_stratum_idx = []
            entry_names = list(grp.get("entry_name_idx_host", ()))
            entry_group_counts = list(grp.get("entry_group_count_host", ()))
            block_cursor = 0
            for entry_id, entry_group_count_raw in enumerate(entry_group_counts):
                entry_group_count = max(0, int(entry_group_count_raw))
                n_blocks = (
                    (entry_group_count + groups_per_program - 1)
                    // groups_per_program
                )
                name_idx = int(entry_names[entry_id]) if entry_id < len(entry_names) else -1
                sid = -1
                if 0 <= name_idx < len(self._tname_list):
                    eid = self._tname_to_eid.get(self._tname_list[name_idx])
                    expert_idx = eid_to_expert_idx.get(eid, -1)
                    if expert_idx >= 0:
                        sid = len(stratum_total_elems)
                        stratum_total_elems.append(int(entry_group_count) * g)
                        stratum_total_blocks.append(int(n_blocks))
                        stratum_expert_idx.append(int(expert_idx))
                entry_stratum_idx.append(int(sid))
                block_cursor += int(n_blocks)
            total_entry_blocks += int(block_cursor)
            grp["_entry_stratum_idx_token"] = descriptor_arena.add(
                entry_stratum_idx, dtype=torch.int32,
            )
            grp["approx_entry_block_count"] = int(block_cursor)

        if not stratum_total_elems:
            stratum_total_elems = [0]
            stratum_total_blocks = [0]
            stratum_expert_idx = [-1]
        n_strata = len(stratum_total_elems)
        cand_cap = int(self._selection_candidate_taus.numel())
        self._approx_probe_n_strata = int(n_strata)
        self._approx_probe_total_entry_blocks = max(1, int(total_entry_blocks))
        self._approx_probe_stratum_total_blocks = list(stratum_total_blocks)
        self._approx_probe_stratum_total_elems = list(stratum_total_elems)
        stratum_total_elems_token = descriptor_arena.add(
            stratum_total_elems, dtype=torch.int64,
        )
        stratum_expert_idx_token = descriptor_arena.add(
            stratum_expert_idx, dtype=torch.int32,
        )
        descriptor_arena.materialize()
        for grp in self._groups:
            grp["entry_stratum_idx"] = descriptor_arena.tensor(
                grp.pop("_entry_stratum_idx_token"),
            )
        self._approx_probe_stratum_total_elems_device = descriptor_arena.tensor(
            stratum_total_elems_token,
        )
        self._approx_probe_stratum_expert_idx_device = descriptor_arena.tensor(
            stratum_expert_idx_token,
        )
        n_local_experts = max(1, len(local_expert_ids))
        self._lfcl_fired_epoch_device = torch.zeros(
            n_local_experts, dtype=torch.int32, device=device,
        )
        try:
            self._lfcl_fired_epoch_host = torch.zeros(
                n_local_experts, dtype=torch.int32, pin_memory=True,
            )
        except RuntimeError:
            self._lfcl_fired_epoch_host = torch.zeros(
                n_local_experts, dtype=torch.int32,
            )
        self._static_descriptor_host_refs.extend(descriptor_arena.host_refs)
        self._approx_probe_sum_q = torch.zeros(
            (n_strata, cand_cap), dtype=torch.float32, device=device,
        )
        self._approx_probe_sum_q2 = torch.zeros_like(self._approx_probe_sum_q)
        self._approx_probe_n_blocks = torch.zeros(
            (n_strata, cand_cap), dtype=torch.int32, device=device,
        )
        self._approx_probe_candidate_stats = torch.zeros(
            (4, cand_cap), dtype=torch.float32, device=device,
        )
        self._approx_probe_candidate_stats_pinned = torch.empty(
            (4, cand_cap), dtype=torch.float32, pin_memory=True,
        )
        self._approx_probe_reject = torch.ones(
            (), dtype=torch.int32, device=device,
        )
        self._approx_probe_decision_code = torch.zeros(
            (), dtype=torch.int32, device=device,
        )
        self._approx_probe_diagnostics = torch.zeros(
            10, dtype=torch.float32, device=device,
        )
        self._approx_probe_capacity_demand_hint = torch.zeros(
            max(1, len(eid_to_expert_idx)), dtype=torch.float32, device=device,
        )
        default_sample_fraction = float(_APPROX_PROBE_DEFAULT_SAMPLE_FRACTION)
        max_sample_fraction = float(_APPROX_PROBE_MAX_SAMPLE_FRACTION)
        default_sample_fraction = max(0.0, min(1.0, default_sample_fraction))
        max_sample_fraction = max(default_sample_fraction, min(1.0, max_sample_fraction))
        total_blocks = max(1, int(self._approx_probe_total_entry_blocks))
        initial_samples = int(math.ceil(
            float(total_blocks) * default_sample_fraction
        ))
        initial_samples = max(1, min(initial_samples, total_blocks))
        sample_cap = int(math.ceil(
            float(total_blocks) * max_sample_fraction
        ))
        sample_cap = max(initial_samples, min(sample_cap, total_blocks))
        self._approx_probe_sample_blocks_current = torch.full(
            (), int(initial_samples), dtype=torch.int32, device=device,
        )
        self._approx_probe_ci_shortfall_streak = torch.zeros(
            (), dtype=torch.int32, device=device,
        )
        self._approx_probe_count_mismatch_streak = torch.zeros(
            (), dtype=torch.int32, device=device,
        )
        self._approx_probe_expected_selected_count = torch.zeros(
            (), dtype=torch.int64, device=device,
        )
        self._approx_probe_sample_grow_hold = torch.zeros(
            (), dtype=torch.int32, device=device,
        )
        self._approx_probe_sample_blocks_min = max(1, int(initial_samples))
        self._approx_probe_sample_blocks_cap = max(1, int(sample_cap))
    # ------------------------------------------------------------------
    def num_tracked_params(self) -> int:
        return self.total_tracked

    def num_full_save_params(self) -> int:
        return self.total_full_save

    def tracked_names(self) -> List[str]:
        return list(self._tracked_names)

    def full_save_names(self) -> List[str]:
        return list(self._full_save_names)

    def full_save_numel_per_name(self) -> Dict[str, int]:
        return dict(self._full_save_numel_per_name)

    def _acquire_ring_event(self, role: str) -> "torch.cuda.Event":
        """Return the single reusable event for a hot-path role."""

        attr = f"_event_{role}"
        ev = getattr(self, attr, None)
        if ev is None:
            raise RuntimeError(f"proposal event {role!r} was not initialized")
        return ev

    def _acquire_timing_event(self, role: str) -> "torch.cuda.Event":
        """Return the single reusable timing event for a debug role."""

        attr = f"_timing_event_{role}"
        ev = getattr(self, attr, None)
        if ev is None:
            ev = torch.cuda.Event(enable_timing=True)
            setattr(self, attr, ev)
        return ev

    @torch.no_grad()
    def gpu_memory_breakdown(self) -> Tuple[float, List[Tuple[str, float]]]:
        """Sum of all GPU tensors held by this lossy checkpoint path (MB) + per-item
        breakdown. Auto-enumerates ``self.__dict__`` so newly-added
        attributes are tracked without a whitelist sync. Detects missing
        accountings vs ``torch.cuda.memory_allocated()``.
        """
        items: List[Tuple[str, float]] = []
        seen_storage: set[tuple[int, int]] = set()

        def add_owned_tensor(name: str, tensor: torch.Tensor) -> None:
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                return
            try:
                storage = tensor.untyped_storage()
                key = (int(storage.data_ptr()), int(storage.nbytes()))
                size_mb = float(storage.nbytes()) / 1e6
            except Exception:
                key = (int(tensor.data_ptr()), int(tensor.element_size() * tensor.numel()))
                size_mb = float(tensor.element_size() * tensor.numel()) / 1e6
            if key in seen_storage:
                return
            seen_storage.add(key)
            items.append((name, size_mb))

        for attr, t in vars(self).items():
            add_owned_tensor(attr, t)
        age2bit_state = getattr(self, "_age2bit_score_state", None)
        if age2bit_state is not None:
            add_owned_tensor("age2bit.age_words", getattr(age2bit_state, "age_words", None))
            add_owned_tensor("age2bit.ehat_geom_lut", getattr(age2bit_state, "ehat_geom_lut", None))
        theta_cp_state = getattr(self, "_theta_cp_score_state", None)
        if theta_cp_state is not None:
            add_owned_tensor("theta_cp.values", getattr(theta_cp_state, "values", None))
        # Current dispatch carries no cross-iteration pending stash; current
        # lossy write inputs are local to ``step()``.
        # _v_mapping_v_flats and m/v/p group entries are flat views into
        # optimizer/proposal storage. Per-group descriptor and partial-count
        # scratch tensors own storage and are part of the proposal footprint.
        groups = getattr(self, "_groups", None) or []
        skip_group_view_keys = {"m_flat", "v_flat", "p_flat", "theta_cp_flat"}
        for group_idx, grp in enumerate(groups):
            if not isinstance(grp, dict):
                continue
            for key, tensor in grp.items():
                if key in skip_group_view_keys:
                    continue
                add_owned_tensor(f"group[{group_idx}].{key}", tensor)
        total = sum(v for _, v in items)
        return total, items

    @torch.no_grad()
    def _init_lossy_flat_layout(self, optimizer) -> None:
        """One-time setup for BF16-Optimizer zero-copy lossy CP views."""
        if getattr(self, "_flat_layout_done", False):
            return
        mapper = get_mapper()
        g = max(1, int(self.group_size))

        # Bucket tracked tensors by BF16-Opt group index (g_idx). For each
        # bucket we capture m, v, p refs, per-tensor sizes, and the original
        # tracked-name order.
        buckets: Dict[int, Dict[str, list]] = {}
        separated: Dict[Tuple[int, int], List[str]] = {}
        for n in self._tracked_names:
            eid = ids_from_name(n, self._expert_id_map)
            if eid is not None:
                separated.setdefault(eid, []).append(n)
        partition_plan = (
            build_expert_partition_plan(self._name_to_param, separated)
            if separated else None
        )
        self._expert_partition_layout = partition_plan is not None
        if partition_plan is not None:
            self._tname_to_eid.update(partition_plan.fragment_to_expert)
            self._expert_partition_fragment_meta = {
                str(frag["fragment_name"]): dict(frag)
                for fragments in partition_plan.fragments_by_expert.values()
                for frag in fragments
            }
            pending_adam_state = False
            for eid in sorted(partition_plan.expert_tensor_map):
                schema = partition_plan.expert_tensor_map[eid]
                fragments = partition_plan.fragments_by_expert.get(eid, [])
                for (frag_name, _shape, numel), frag in zip(schema, fragments):
                    n_elem = int(numel)
                    group_idx = int(frag["group_idx"])
                    local_offset = int(frag["local_offset"])
                    p_group = mapper.zero12_flat_group_view(
                        group_idx, state_key=None,
                    )
                    m_group = mapper.zero12_flat_group_view(
                        group_idx, state_key="exp_avg",
                    )
                    v_group = mapper.zero12_flat_group_view(
                        group_idx, state_key="exp_avg_sq",
                    )
                    if p_group is None:
                        raise RuntimeError(
                            "expert partition fragment is missing local "
                            f"fp32 source: {frag_name}"
                        )
                    if m_group is None or v_group is None:
                        pending_adam_state = True
                        break
                    if local_offset < 0 or local_offset + n_elem > int(p_group.numel()):
                        raise RuntimeError(
                            "expert partition fragment exceeds local flat "
                            f"group: {frag_name}"
                        )
                    b = buckets.setdefault(group_idx, {
                        "names": [], "sizes": [],
                        "m_refs": [m_group.detach().reshape(-1)],
                        "v_refs": [v_group.detach().reshape(-1)],
                        "p_refs": [p_group.detach().reshape(-1)],
                        "opt_offsets": [0],
                        "physical_offsets": [],
                        "optimizer_group_idx": [group_idx],
                        "flat_total": int(p_group.numel()),
                    })
                    b["names"].append(frag_name)
                    b["sizes"].append(n_elem)
                    b["physical_offsets"].append(local_offset)
                if pending_adam_state:
                    break
            if pending_adam_state:
                if not getattr(self, "_flat_layout_waiting_for_adam_state_logged", False):
                    print(
                        "[proposal] lossy CP layout deferred until DeepSpeed "
                        "initializes BF16 Adam state",
                        flush=True,
                    )
                    self._flat_layout_waiting_for_adam_state_logged = True
                return

        tracked_names_for_full = [] if partition_plan is not None else self._tracked_names
        for n in tracked_names_for_full:
            p = self._name_to_param.get(n)
            if p is None:
                continue
            m = (
                mapper.get_local_state_view(p, "exp_avg")
                if mapper is not None
                else optimizer.state.get(p, {}).get("exp_avg")
            )
            v = (
                mapper.get_local_state_view(p, "exp_avg_sq")
                if mapper is not None
                else optimizer.state.get(p, {}).get("exp_avg_sq")
            )
            if m is None or v is None:
                continue
            # Expert hot path needs rank-local zero-copy views. Public mapper
            # getters may intentionally return full-state safe API tensors for
            # non-expert FC/recovery, so use local views here.
            p_local = (
                mapper.get_local_param_view(p)
                if mapper is not None
                else fp32_param(p)
            )
            if p_local is None:
                continue
            # Resolve which BF16-Opt group this tensor lives in. Use the
            # mapper's coord table when available; otherwise fall back to a
            # synthetic single-bucket index 0 (best-effort for non-DS modes).
            g_idx = 0
            opt_elem_offset = 0
            if mapper is not None and hasattr(mapper, "_coords"):
                coord = mapper._coords.get(id(p))
                if coord is not None:
                    g_idx = coord[0]
                    opt_elem_offset = int(coord[1])
                elif getattr(mapper, "mode", "") in (
                    "megatron_float16",
                    "unfused_fp16",
                ):
                    # These optimizers keep one FP32 master tensor plus Adam
                    # state per low-precision parameter, not a shared flat
                    # group. Treat each tracked tensor as its own local group
                    # so the hot path still gets zero-copy p/m/v views.
                    g_idx = len(buckets)
                    opt_elem_offset = 0
            b = buckets.setdefault(g_idx, {
                "names": [], "sizes": [], "m_refs": [], "v_refs": [],
                "p_refs": [], "opt_offsets": [],
            })
            b["names"].append(n)
            b["sizes"].append(int(p.numel()))
            b["m_refs"].append(m.detach().reshape(-1))
            b["v_refs"].append(v.detach().reshape(-1))
            b["p_refs"].append(p_local.detach().reshape(-1))
            b["opt_offsets"].append(int(opt_elem_offset))

        if not buckets:
            return

        device = next(iter(buckets.values()))["m_refs"][0].device

        def _try_flat_view(refs):
            """Try to build a single 1-D flat view spanning refs in
            storage-offset order. Returns (flat or None, reason str)."""
            if not refs:
                return None, "empty"
            try:
                base = refs[0]
                base_storage = base.untyped_storage()
                base_ptr = base_storage.data_ptr()
                cur_off = base.storage_offset()
                esize = base.element_size()
                for idx, r in enumerate(refs):
                    if r.untyped_storage().data_ptr() != base_ptr:
                        return None, (
                            f"ref[{idx}] storage ptr "
                            f"{hex(r.untyped_storage().data_ptr())} "
                            f"!= base {hex(base_ptr)}"
                        )
                    if r.storage_offset() != cur_off:
                        return None, (
                            f"ref[{idx}] off={r.storage_offset()}, "
                            f"expected {cur_off}"
                        )
                    if r.element_size() != esize or r.dtype != base.dtype:
                        return None, f"ref[{idx}] dtype/size mismatch"
                    cur_off += r.numel()
                total_n = cur_off - base.storage_offset()
                flat = torch.empty(0, dtype=base.dtype, device=base.device)
                flat.set_(
                    base_storage,
                    storage_offset=base.storage_offset(),
                    size=(total_n,),
                    stride=(1,),
                )
                return flat, None
            except Exception as e:
                return None, f"exception: {e!r}"

        total_elems = 0
        total_groups = 0
        misaligned_tensors: List[str] = []
        for g_idx in sorted(buckets.keys()):
            b = buckets[g_idx]
            total_elems += sum(b["sizes"])
            total_groups += sum(sz // g for sz in b["sizes"])
            for name, size in zip(b["names"], b["sizes"]):
                if int(size) % g != 0:
                    misaligned_tensors.append(f"{name}:{int(size)}")
        if misaligned_tensors:
            sample = ", ".join(misaligned_tensors[:8])
            raise RuntimeError(
                "V2 lossy CP requires each tracked expert tensor numel to be "
                f"divisible by group_size={g}; otherwise group ownership can "
                f"cross tensor boundaries. Offenders: {sample}"
            )
        self._flat_total = int(total_elems)
        self.total_tracked = int(total_elems)

        from ..cutoff.age2bit import allocate_age2bit_score_state
        self._age2bit_score_state = allocate_age2bit_score_state(
            n_elements=int(total_elems),
            device=device,
            h_lut_bins=int(self.h_lookup_lut_bins),
        )
        # Selection/write kernels compute per-element H through the prebuilt
        # log(v)->H LUT.
        # Persistent cutoff-search scratch reused by every Stage 1 pass.
        self._n_bins_cutoff = int(self.n_bins_cutoff)
        self._selection_hist = torch.zeros(
            self._n_bins_cutoff, dtype=torch.int64, device=device,
        )
        self._cold_start_hist_pinned = torch.empty(
            self._n_bins_cutoff, dtype=torch.int64, pin_memory=True,
        )
        self._cold_start_minmax_pinned = torch.empty(
            2, dtype=torch.float32, pin_memory=True,
        )
        self._selection_smin = torch.full(
            (), float("inf"), dtype=torch.float32, device=device,
        )
        self._selection_smax = torch.zeros(
            (), dtype=torch.float32, device=device,
        )
        self._selection_global_n_elements = torch.zeros(
            (), dtype=torch.int64, device=device,
        )
        cand_cap = int(getattr(self, "_selection_candidate_capacity", 16))
        self._candidate_ladder_sync_cpu = torch.empty(
            cand_cap + 1, dtype=torch.float64,
        )
        self._selection_global_n_elements_cpu = torch.empty(
            1, dtype=torch.int64,
        )
        self._direct_feedback_counts_cpu = torch.empty(
            14, dtype=torch.int64,
        )
        self._selection_candidate_taus = torch.empty(
            cand_cap, dtype=torch.float32, device=device,
        )
        self._approx_probe_decision_code = torch.zeros(
            (), dtype=torch.int32, device=device,
        )
        self._direct_write_fp32_scalar_host = torch.empty(
            3, dtype=torch.float32, pin_memory=True,
        )
        self._direct_write_step_host = torch.empty(
            1, dtype=torch.int32, pin_memory=True,
        )
        self._direct_write_fp32_scalar_device = torch.empty(
            3, dtype=torch.float32, device=device,
        )
        self._direct_write_step_device = torch.empty(
            1, dtype=torch.int32, device=device,
        )
        self._direct_write_total_th_ring_device = torch.empty(
            1, dtype=torch.float32, device=device,
        )
        self._h_quantile_built_at = -1  # rebuild marker

        # Build per-group descriptor state. Production score/write kernels no
        # Score kernels no longer read group residual state; descriptors keep
        # only element/group geometry.
        self._groups: List[Dict] = []
        # Per-name views into the same flat buffer (in master order).
        self._tname_list: List[str] = []
        self._group_offsets_cpu: List[int] = [0]
        all_zero_copy = True
        zc_reasons: List[str] = []
        cur_off = 0  # cumulative position in flat (master order)
        descriptor_arena = StaticDescriptorArena(device=device)
        for g_idx in sorted(buckets.keys()):
            b = buckets[g_idx]
            sizes = b["sizes"]
            names = b["names"]
            m_refs = b["m_refs"]
            v_refs = b["v_refs"]
            p_refs = b["p_refs"]
            opt_offsets = b.get("opt_offsets", [])
            physical_offsets = b.get("physical_offsets")
            optimizer_group_indices = b.get("optimizer_group_idx", [g_idx])
            optimizer_group_idx = (
                int(optimizer_group_indices[0])
                if optimizer_group_indices else int(g_idx)
            )
            tracked_total = sum(sizes)
            group_total = int(b.get("flat_total", tracked_total))
            group_groups = tracked_total // g
            master_off = cur_off  # start of this group's region in flat
            # h_g views removed — selection/write kernels do inline H lookup.

            # Track this BF16 group's name range in the global name_list.
            start_name_in_grp = len(self._tname_list)

            # Per-name views (in this master's region).
            entry_elem_start_host: List[int] = []
            entry_age_elem_start_host: List[int] = []
            entry_group_start_host: List[int] = []
            entry_group_count_host: List[int] = []
            entry_name_idx_host: List[int] = []
            entry_block_entry_idx_host: List[int] = []
            entry_block_local_group_start_host: List[int] = []
            local_elem_off = 0
            groups_per_program = max(1, 256 // max(1, int(g)))
            power = 1
            while power < groups_per_program:
                power *= 2
            groups_per_program = power
            if physical_offsets is not None and len(physical_offsets) != len(sizes):
                raise RuntimeError(
                    "expert partition physical offset count does not match "
                    "tracked fragment count"
                )
            for entry_i, (n, sz) in enumerate(zip(names, sizes)):
                n_g = sz // g
                if n_g == 0:
                    local_elem_off += int(sz)
                    continue
                name_idx = len(self._tname_list)
                entry_elem_start = (
                    int(physical_offsets[entry_i])
                    if physical_offsets is not None
                    else int(local_elem_off)
                )
                entry_age_elem_start = int(cur_off) * int(g)
                entry_elem_start_host.append(int(entry_elem_start))
                entry_age_elem_start_host.append(int(entry_age_elem_start))
                entry_group_start_host.append(int(entry_elem_start) // g)
                entry_group_count_host.append(int(n_g))
                entry_name_idx_host.append(int(name_idx))
                local_entry_id = len(entry_name_idx_host) - 1
                for block_group_start in range(0, int(n_g), groups_per_program):
                    entry_block_entry_idx_host.append(int(local_entry_id))
                    entry_block_local_group_start_host.append(
                        int(block_group_start),
                    )
                self._tname_list.append(n)
                cur_off += n_g
                self._group_offsets_cpu.append(cur_off)
                local_elem_off += int(sz)

            end_name_in_grp = len(self._tname_list)
            entry_descriptor_tokens = (
                descriptor_arena.add(entry_elem_start_host, dtype=torch.int64),
                descriptor_arena.add(entry_age_elem_start_host, dtype=torch.int64),
                descriptor_arena.add(entry_group_start_host, dtype=torch.int64),
                descriptor_arena.add(entry_group_count_host, dtype=torch.int32),
                descriptor_arena.add(entry_name_idx_host, dtype=torch.int32),
                descriptor_arena.add(entry_block_entry_idx_host, dtype=torch.int32),
                descriptor_arena.add(
                    entry_block_local_group_start_host,
                    dtype=torch.int32,
                ),
            )

            m_flat, m_why = _try_flat_view(m_refs)
            v_flat, v_why = _try_flat_view(v_refs)
            p_flat, p_why = _try_flat_view(p_refs)
            zc = (m_flat is not None and v_flat is not None
                  and p_flat is not None
                  and m_flat.numel() >= group_total
                  and v_flat.numel() >= group_total
                  and p_flat.numel() >= group_total)
            if not zc:
                all_zero_copy = False
                zc_reasons.append(
                    f"g_idx={g_idx} m_why={m_why} v_why={v_why} p_why={p_why}"
                )
            import os as _os_idiag
            if _os_idiag.environ.get("PROPOSAL_INIT_DIAG", "") in ("1",):
                _per_name_total = sum(sz // g for sz in sizes if sz // g > 0)
                _group_total_g = group_total // g
                _mflat_n = int(m_flat.numel()) if m_flat is not None else -1
                _pflat_n = int(p_flat.numel()) if p_flat is not None else -1
                print(f"[init_diag] g_idx={g_idx} master_off={master_off} "
                      f"group_total_elem={group_total} tracked_elem={tracked_total} "
                      f"group_groups={_group_total_g} "
                      f"sum_per_name_groups={_per_name_total} "
                      f"m_flat.numel={_mflat_n} p_flat.numel={_pflat_n} "
                      f"start_name={start_name_in_grp} end_name={end_name_in_grp} "
                      f"sizes_uniq={sorted(set(sizes))[:5]}",
                      flush=True)
            self._groups.append({
                "g_idx": g_idx,
                "optimizer_group_idx": optimizer_group_idx,
                "names": names,
                "sizes": sizes,
                "m_refs": m_refs,
                "v_refs": v_refs,
                "p_refs": p_refs,
                "m_flat": m_flat,
                "v_flat": v_flat,
                "p_flat": p_flat,
                "optimizer_elem_offset": (
                    int(opt_offsets[0]) if opt_offsets else 0
                ),
                "master_off": master_off,  # global flat element offset
                "start_name_in_grp": start_name_in_grp,
                "end_name_in_grp": end_name_in_grp,
                "_entry_descriptor_tokens": entry_descriptor_tokens,
                "entry_elem_start_host": list(entry_elem_start_host),
                "entry_age_elem_start_host": list(entry_age_elem_start_host),
                "entry_group_start_host": list(entry_group_start_host),
                "entry_name_idx_host": list(entry_name_idx_host),
                "entry_group_count_host": list(entry_group_count_host),
                "entry_block_entry_idx_host": list(entry_block_entry_idx_host),
                "entry_block_local_group_start_host": list(
                    entry_block_local_group_start_host,
                ),
                "zero_copy": zc,
                "total": tracked_total,
                "flat_total": group_total,
                "total_aligned": (tracked_total // g) * g,
            })

        descriptor_arena.materialize()
        for grp in self._groups:
            (
                entry_elem_start_t,
                entry_age_elem_start_t,
                entry_group_start_t,
                entry_group_count_t,
                entry_name_idx_t,
                entry_block_entry_idx_t,
                entry_block_local_group_start_t,
            ) = (
                descriptor_arena.tensor(tok)
                for tok in grp.pop("_entry_descriptor_tokens")
            )
            grp["entry_elem_start"] = entry_elem_start_t
            grp["entry_age_elem_start"] = entry_age_elem_start_t
            grp["entry_group_start"] = entry_group_start_t
            grp["entry_group_count"] = entry_group_count_t
            grp["entry_name_idx"] = entry_name_idx_t
            grp["entry_block_entry_idx"] = entry_block_entry_idx_t
            grp["entry_block_local_group_start"] = entry_block_local_group_start_t
        self._static_descriptor_host_refs = list(descriptor_arena.host_refs)
        self._zero_copy = all_zero_copy
        self._build_approx_probe_metadata(device=device)

        # ---- v_t → H quantile-matching lookup tables ----
        # CDF matching: anchor refresh builds the H quantile table from the
        # hook-scaled grad_flat, then ``update_v_mapping`` refreshes v_q from
        # live Adam v_t before training resumes. The derived LUT is rebuilt
        # after v_q/h_q changes; kernels do O(1) lookup per element.
        self._N_Q = int(self.n_q)
        self._v_quantile_values = torch.zeros(
            self._N_Q, dtype=torch.float32, device=device,
        )
        self._h_quantile_values = torch.zeros(
            self._N_Q, dtype=torch.float32, device=device,
        )
        # Anchor validation scratch.  H refresh scans the resident gradient
        # views with Triton and atomically marks this scalar on NaN/Inf.  Keep
        # it resident so validation never materializes a model-sized bool
        # tensor (``torch.isfinite(view)``) or allocates per refresh.
        self._h_refresh_nonfinite_flag = torch.zeros(
            1, dtype=torch.int32, device=device,
        )
        # One int per tracked tensor is enough to retain exact zero/nonzero
        # coverage without reducing each model-sized H view into a newly
        # allocated CUDA scalar.  The pinned mirror is the only D2H target.
        _h_refresh_view_capacity = max(1, len(self._tracked_names))
        self._h_refresh_view_coverage = torch.zeros(
            _h_refresh_view_capacity, dtype=torch.int32, device=device,
        )
        _anchor_validation_pin = bool(torch.cuda.is_available())
        self._h_refresh_view_coverage_host = torch.empty(
            _h_refresh_view_capacity,
            dtype=torch.int32,
            pin_memory=_anchor_validation_pin,
        )
        self._last_h_quantile_summary = None
        self._selection_v_q_for_write = self._v_quantile_values
        self._selection_h_q_for_write = self._h_quantile_values
        self._h_anchor_initialised = False

        # Element-offset lookup used to map name slices onto BF16 flat groups.
        self._elem_offsets_for_selection: Optional[torch.Tensor] = None  # i64, N_names+1

        if torch.cuda.is_available():
            self._event_compute: Optional["torch.cuda.Event"] = torch.cuda.Event()
            self._event_evopt: Optional["torch.cuda.Event"] = torch.cuda.Event()
            self._event_selection_count: Optional["torch.cuda.Event"] = torch.cuda.Event()
        else:
            self._event_compute = None
            self._event_evopt = None
            self._event_selection_count = None

        if not all_zero_copy:
            detail = "; ".join(zc_reasons[:8])
            raise RuntimeError(
                "V2 lossy CP requires zero-copy BF16 optimizer m/v/p flat "
                f"views for every group; refusing fallback. {detail}"
            )

        if not bool(getattr(self, "e_hat_proxy", True)):
            from ..cutoff.theta_cp import allocate_theta_cp_score_state
            self._theta_cp_score_state = allocate_theta_cp_score_state(
                n_elements=int(total_elems),
                device=device,
                dtype=torch.bfloat16,
            )
            theta_values = self._theta_cp_score_state.values
            for grp in self._groups:
                grp["theta_cp_flat"] = theta_values
            if int(
                getattr(self.proposal_config, "h_refresh_warmup_iters", 0) or 0
            ) <= 0:
                self.reset_checkpoint_score_state_from_live()
        else:
            self._theta_cp_score_state = None
            for grp in self._groups:
                grp["theta_cp_flat"] = None

        try:
            # Persistent memory accounting (only items truly retained
            # across iters).
            def _tensor_mb(tensor) -> float:
                if tensor is None:
                    return 0.0
                try:
                    return float(tensor.untyped_storage().nbytes()) / 1e6
                except Exception:
                    return float(int(tensor.numel()) * int(tensor.element_size())) / 1e6

            age2bit_state = getattr(self, "_age2bit_score_state", None)
            mem_age2bit = _tensor_mb(
                age2bit_state.age_words if age2bit_state is not None else None
            )
            mem_score_lut = _tensor_mb(
                age2bit_state.ehat_geom_lut if age2bit_state is not None else None
            )
            theta_cp_state = getattr(self, "_theta_cp_score_state", None)
            mem_theta_cp = _tensor_mb(
                theta_cp_state.values if theta_cp_state is not None else None
            )
            mem_hist = self._n_bins_cutoff * 8 / 1e6
            mem_total = mem_age2bit + mem_score_lut + mem_theta_cp + mem_hist
            print(f"[proposal] lossy CP state init: ZERO-COPY all groups, "
                  f"optimizer_groups={len(self._groups)}, "
                  f"total={total_elems} elements, groups={total_groups}, "
                  f"n_tracked={len(self._tname_list)}",
                  flush=True)
            print(f"[proposal] lossy CP mem persistent: "
                  f"age2bit={mem_age2bit:.1f}MB "
                  f"score_lut={mem_score_lut:.3f}MB "
                  f"theta_cp={mem_theta_cp:.1f}MB "
                  f"hist_i64={mem_hist:.3f}MB -> "
                  f"total={mem_total:.1f}MB",
                  flush=True)
            preview_groups = list(self._groups)
            if len(preview_groups) > 16:
                hidden = len(preview_groups) - 12
                preview_groups = preview_groups[:8] + preview_groups[-4:]
                print(
                    f"[proposal] bf16_group detail truncated: "
                    f"showing 12/{len(self._groups)} groups "
                    f"({hidden} omitted)",
                    flush=True,
                )
            for grp in preview_groups:
                print(f"[proposal] bf16_group g_idx={grp['g_idx']}: "
                      f"n_tensors={len(grp['names'])}, "
                      f"total={grp['total']}, "
                      f"zero_copy={grp['zero_copy']}",
                      flush=True)
            if zc_reasons:
                for r in zc_reasons:
                    print(f"[proposal] bf16_group zc-fail {r}", flush=True)
        except Exception:
            pass

        # Controller setup can cache the unpartitioned model-domain count
        # before DeepSpeed creates its rank-local ZeRO flat.  Every rank
        # reaches this one-time layout boundary, so invalidate collectively
        # here and let the first real selection rebuild the global count from
        # ``LossyCpState.index.n_total_elements``.
        self._invalidate_global_selection_element_count_cache()
        self._flat_layout_done = True

    @torch.no_grad()
    def reset_checkpoint_score_state_from_live(
        self, *, next_global_step: int = 1,
    ) -> None:
        """Start selection from the current exact recovery baseline."""

        theta_state = getattr(self, "_theta_cp_score_state", None)
        if theta_state is not None:
            theta_values = theta_state.values
            group_size = max(1, int(self.group_size))
            for grp in self._groups:
                p_flat = grp["p_flat"]
                for elem_start, age_start, group_count in zip(
                    grp.get("entry_elem_start_host", ()),
                    grp.get("entry_age_elem_start_host", ()),
                    grp.get("entry_group_count_host", ()),
                ):
                    n_elem = int(group_count) * group_size
                    if n_elem <= 0:
                        continue
                    theta_values.narrow(0, int(age_start), n_elem).copy_(
                        p_flat.narrow(0, int(elem_start), n_elem),
                        non_blocking=True,
                    )
                grp["theta_cp_flat"] = theta_values
        age_state = getattr(self, "_age2bit_score_state", None)
        if age_state is not None:
            age_state.age_words.zero_()
        # Preserve the original four-global-step cutoff bootstrap window,
        # shifted so its first step is the first proposal selection.
        self._cold_start_first_global_step = int(next_global_step)

    @torch.no_grad()
    def _refresh_lossy_cp_state_refs(self) -> None:
        """Refresh the V2 holder with the active tensor views.

        The dataclass owns no storage; it makes the active selection/write
        path address state through the V2 role names while preserving the
        existing allocation lifetime.
        """
        required = (
            getattr(self, "_selection_hist", None),
            getattr(self, "_selection_smin", None),
            getattr(self, "_selection_smax", None),
            getattr(self, "_selection_total_th", None),
            getattr(self, "_elem_offsets_for_selection", None),
            getattr(self, "_group_offsets_t", None),
        )
        if any(x is None for x in required):
            return

        from ..refresh.quantile import QuantileRefs, QuantileTable
        from ..state import (
            BF16OptGroupView,
            CutoffSearchScratch,
            ExpertParamIndex,
            LossyCpState,
        )

        device = self._selection_hist.device
        h_lut = self._ensure_h_lookup_lut(device=device)

        bf16_opt_groups: List[BF16OptGroupView] = []
        for grp in self._groups:
            if not grp.get("zero_copy", False):
                raise RuntimeError(
                    "LossyCpState requires zero-copy BF16 optimizer groups."
                )
            m_flat = grp.get("m_flat")
            v_flat = grp.get("v_flat")
            p_flat = grp.get("p_flat")
            theta_cp_flat = grp.get("theta_cp_flat")
            if m_flat is None or v_flat is None or p_flat is None:
                raise RuntimeError(
                    "LossyCpState requires m/v/p flat views for every group."
                )
            bf16_opt_groups.append(BF16OptGroupView(
                m_flat=m_flat,
                v_flat=v_flat,
                p_flat=p_flat,
                master_off_elem=int(grp["master_off"]) * int(self.group_size),
                theta_cp_flat=theta_cp_flat,
                name_start=int(grp["start_name_in_grp"]),
                name_end=int(grp["end_name_in_grp"]),
                entry_elem_start=grp.get("entry_elem_start"),
                entry_age_elem_start=grp.get("entry_age_elem_start"),
                entry_group_start=grp.get("entry_group_start"),
                entry_group_count=grp.get("entry_group_count"),
                entry_name_idx=grp.get("entry_name_idx"),
                entry_block_entry_idx=grp.get("entry_block_entry_idx"),
                entry_block_local_group_start=grp.get(
                    "entry_block_local_group_start",
                ),
            ))

        n_experts = (
            int(self.num_experts)
            if self.num_experts is not None
            else len(set(self._tname_to_eid.values()))
        )
        quantiles = QuantileRefs(
            v=QuantileTable(
                values=self._selection_v_q_for_write,
                n_q=int(self._N_Q),
            ),
            h=QuantileTable(
                values=self._selection_h_q_for_write,
                n_q=int(self._N_Q),
            ),
            h_lut=h_lut,
        )
        index = ExpertParamIndex(
            bf16_opt_groups=bf16_opt_groups,
            tname_list=list(self._tname_list),
            tname_to_expert=dict(self._tname_to_eid),
            elem_offsets=self._elem_offsets_for_selection,
            group_offsets=self._group_offsets_t,
            n_total_elements=int(self._group_offsets_cpu[-1]) * int(self.group_size),
            n_total_groups=int(self._group_offsets_cpu[-1]),
            n_experts=n_experts,
        )
        scratch = CutoffSearchScratch(
            hist=self._selection_hist,
            smin=self._selection_smin,
            smax=self._selection_smax,
            total_th=self._selection_total_th,
        )
        self.lossy_cp_state = LossyCpState(
            index=index,
            quantiles=quantiles,
            scratch=scratch,
            age2bit=getattr(self, "_age2bit_score_state", None),
            theta_cp=getattr(self, "_theta_cp_score_state", None),
        )

    @torch.no_grad()
    def step(
        self,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        ready_event: Optional["torch.cuda.Event"] = None,
        d2h_event_callback: Optional[Any] = None,
        emission_release_event: Optional[Any] = None,
        emission_cuda_event: Optional["torch.cuda.Event"] = None,
    ) -> Tuple[
        LossyStepStats,
        Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        """Run one lossy checkpoint iteration.

        Args:
            optimizer: Adam (or variant). m / v are read for dispatch.
            global_step: 1-indexed iteration number.
            ready_event: event recorded by the training thread after the
                previous optimizer step. The proposal compute stream waits
                on it device-side; the worker thread must not synchronize
                on it.
            d2h_event_callback: called with the expert AOS final-D2H event
                used by the optimizer fence and pinned-slab writer lease.
            emission_release_event: optional host event used only by PP stage
                0. Stage 1 cutoff is launched normally, then the expert worker
                waits here before the full direct-emission scan and D2H.
            emission_cuda_event: reusable event recorded immediately before
                stage 0 posts its first RecvGrad. The host release occurs only
                after the original RecvGrad instruction has posted all NCCL
                receives; the proposal compute stream then waits on this
                pre-receive event.
        """
        import time as _time
        _t0_entry = _time.perf_counter()
        self._current_global_step_for_selection = int(global_step)
        device = next(self.model.parameters()).device
        group_size = max(1, int(self.group_size))
        _t0_dispatch_start = _time.perf_counter()
        _t0_dispatch_end = _time.perf_counter()
        phase_dispatch_ms = (_t0_dispatch_end - _t0_dispatch_start) * 1000.0

        if self._compute_stream is None and torch.cuda.is_available():
            from ..runtime import get_compute_stream
            self._compute_stream = get_compute_stream()
        stream = self._compute_stream
        prev_stream = None
        if stream is not None:
            if ready_event is not None:
                stream.wait_event(ready_event)
            else:
                cur_stream = torch.cuda.current_stream()
                ev_opt = self._acquire_ring_event("evopt")
                ev_opt.record(cur_stream)
                stream.wait_event(ev_opt)
            # Switch default stream → compute_stream so all GPU launches
            # below land in compute_stream without an extra `with` block.
            prev_stream = torch.cuda.current_stream()
            torch.cuda.set_stream(stream)
        # Mark we're inside compute_stream so prepare_selection_state's DEBUG
        # sync (which would block main stream) is suppressed.
        self._inside_compute_stream = True

        self.prepare_selection_state(optimizer, global_step)

        _t0_p1 = _time.perf_counter()

        if not getattr(self, "_flat_layout_done", False):
            # init not done yet — return early stats (defensive).
            self._inside_compute_stream = False
            if prev_stream is not None:
                torch.cuda.set_stream(prev_stream)
            stats = LossyStepStats(
                step=global_step, total_tracked=0, k=0, topk_ratio=0.0,
                max_A=0.0, max_delta=0.0, score_mean=0.0, score_max=0.0,
                selected_bytes_sparse=0,
                full_save_bytes=self.full_save_bytes,
                full_save_numel=self.total_full_save, compute_time_ms=0.0,
            )
            if not hasattr(self, "last_phase_ms") or self.last_phase_ms is None:
                self.last_phase_ms = {}
            self.last_phase_ms.update({
                "setup": 0.0,
                "selection_state": 0.0,
                "selection": 0.0,
                "write_lossy_records": 0.0,
                "post": 0.0,
            })
            return stats, {}

        # offsets_t lazy-init (once).
        if not hasattr(self, "_group_offsets_t"):
            self._group_offsets_t = device_tensor_from_pinned_host(
                self._group_offsets_cpu, dtype=torch.long, device=device,
            )

        total_elems = int(self._group_offsets_cpu[-1]) * group_size

        _t0_p2 = _time.perf_counter()
        phase_selection_state_ms = (_t0_p2 - _t0_p1) * 1000.0

        # Stage 1: device-side cutoff decision. Direct emission handles
        # selection/fold/materialization.
        self._refresh_lossy_cp_state_refs()
        selection_event = self._find_selection_cutoff_for_current_iter(
            global_step=global_step,
            group_size=group_size,
            lr_over_bc1=float(getattr(self, "_adam_lr_over_bc1_for_selection", 1e-4)),
            sqrt_bc2_inv=float(getattr(self, "_adam_sqrt_bc2_inv_for_selection", 1.0)),
            eps_adam=float(getattr(self, "_adam_eps_for_selection", 1e-8)),
            stream=stream,
        )

        _t0_p2_end = _time.perf_counter()
        phase_selection_ms = (_t0_p2_end - _t0_p2) * 1000.0
        # Restore main default stream (we set compute_stream as default
        # at STEP 1 entry).
        self._inside_compute_stream = False
        if prev_stream is not None:
            torch.cuda.set_stream(prev_stream)

        from ..emission.record_kernels import AdamHParams as _LossyAdamHParams
        _runtime_state_for_write = self.lossy_cp_state
        _h_lut_for_write = (
            _runtime_state_for_write.quantiles.h_lut
            if _runtime_state_for_write is not None else None
        )
        lossy_write_inputs = {
            "step": global_step,
            "direct_emission_enabled": bool(
                getattr(self, "_selection_direct_emission_will_fold", False)
            ),
            "h_lut_values": (
                _h_lut_for_write.values if _h_lut_for_write is not None else None
            ),
            "h_lut_log_v_min": (
                _h_lut_for_write.log_v_min if _h_lut_for_write is not None else None
            ),
            "h_lut_inv_log_v_width": (
                _h_lut_for_write.inv_log_v_width if _h_lut_for_write is not None else None
            ),
            "total_th": self._selection_total_th,
            "adam_hparams": _LossyAdamHParams(
                lr_over_bc1=float(getattr(self, "_adam_lr_over_bc1_for_selection", 1e-4)),
                sqrt_bc2_inv=float(getattr(self, "_adam_sqrt_bc2_inv_for_selection", 1.0)),
                eps=float(getattr(self, "_adam_eps_for_selection", 1e-8)),
            ),
            "random_mode": bool(getattr(self, "_selection_random_mode", False)),
            "random_salt": int(getattr(self, "_selection_random_salt", 0)),
        }
        lossy_write_inputs.update(dict(getattr(self, "_selection_tau_meta", {}) or {}))

        if emission_release_event is not None:
            # This blocks only the expert worker thread. The pipeline training
            # thread records ``emission_cuda_event`` before stage 0's first
            # RecvGrad, posts that receive, and only then sets the host event.
            # The worker therefore cannot submit scan ahead of NCCL, while its
            # compute stream is ordered after the preceding forward kernels.
            if not emission_release_event.wait(timeout=300.0):
                raise RuntimeError(
                    "timed out waiting for PP stage0 backward-bubble release"
                )
            if stream is not None and emission_cuda_event is not None:
                stream.wait_event(emission_cuda_event)

        self._lossy_record_writer.write_for_iter(
            optimizer,
            lossy_write_inputs=lossy_write_inputs,
            selection_event=selection_event,
            d2h_event_callback=d2h_event_callback,
        )
        _t0_end = _time.perf_counter()
        phase_launch_ms = (_t0_end - _t0_dispatch_end) * 1000.0
        phase_setup_ms = (_t0_p1 - _t0_dispatch_end) * 1000.0
        # dict.update() preserves keys written earlier by diagnostics.
        if not hasattr(self, "last_phase_ms") or self.last_phase_ms is None:
            self.last_phase_ms = {}
        self.last_phase_ms.update({
            "setup": phase_setup_ms,
            "selection_state": phase_selection_state_ms,
            "selection": phase_selection_ms,
            "write_lossy_records": 0.0,
            "post": 0.0,
            "dispatch": phase_dispatch_ms,
            "launch": phase_launch_ms,
        })
        if os.environ.get("PROPOSAL_LOSSY_STEP_PROFILE", "") in ("1", "true", "TRUE"):
            print(
                f"[proposal] lossy_step_profile @iter{int(global_step)}: "
                f"setup={phase_setup_ms:.1f}ms "
                f"selection_state={phase_selection_state_ms:.1f}ms "
                f"selection={phase_selection_ms:.1f}ms "
                f"launch={phase_launch_ms:.1f}ms",
                flush=True,
            )

        stats = LossyStepStats(
            step=global_step,
            total_tracked=total_elems,
            k=0,
            topk_ratio=0.0,
            max_A=0.0,
            max_delta=0.0,
            score_mean=0.0,
            score_max=0.0,
            selected_bytes_sparse=0,
            full_save_bytes=self.full_save_bytes,
            full_save_numel=self.total_full_save,
            compute_time_ms=0.0,
        )

        return stats, {}
