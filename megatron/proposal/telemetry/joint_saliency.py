"""Oracle-time all-element joint expert/parameter saliency telemetry.

Two quantities must remain distinct:

* this module scans **all tracked expert elements** and records the production
  diagonal score computed from the BF16 ``theta_cp`` scoring residual; and
* :mod:`independent_oracle` retains the sampled directional-HVP reference.

The proxy is not labelled as actual per-element loss.  Its formula and the
post-dispatch checkpoint phase are embedded in every manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from .joint_saliency_analysis import DEFAULT_BUDGETS, histogram_budget_summary
from .joint_saliency_gpu import (
    CLIPPED_HIGH,
    CLIPPED_LOW,
    NONFINITE,
    POSITIVE,
    TOTAL,
    ZERO,
    JointSaliencyWorkspace,
    launch_joint_saliency_group,
)
from .joint_saliency_io import write_joint_saliency_shard


@dataclass(frozen=True)
class JointSaliencySettings:
    enabled: bool = False
    n_bins: int = 512
    log10_min: float = -30.0
    log10_max: float = 10.0
    budgets: Tuple[float, ...] = DEFAULT_BUDGETS
    output_dir: Path = Path("joint_saliency")
    global_expert_base: int = 0
    ep_rank: int = 0
    ep_size: int = 1
    experts_per_layer: int = 0

    def validated(self) -> "JointSaliencySettings":
        if self.n_bins < 16:
            raise ValueError("joint saliency n_bins must be >= 16")
        if not math.isfinite(self.log10_min) or not math.isfinite(self.log10_max):
            raise ValueError("joint saliency log bounds must be finite")
        if self.log10_max <= self.log10_min:
            raise ValueError("joint saliency log10_max must exceed log10_min")
        if not self.budgets or any(not (0.0 < x <= 1.0) for x in self.budgets):
            raise ValueError("joint saliency budgets must satisfy 0 < ratio <= 1")
        if self.global_expert_base < 0:
            raise ValueError("joint saliency global_expert_base must be non-negative")
        if self.ep_size < 1 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("joint saliency EP topology is invalid")
        if self.experts_per_layer < 0 or (
            self.enabled and self.experts_per_layer < 1
        ):
            raise ValueError(
                "enabled joint saliency requires positive experts_per_layer"
            )
        return self


@dataclass(frozen=True)
class JointSaliencyRow:
    row: int
    name: str
    layer: int
    local_expert: int
    global_expert: int
    numel: int
    group_idx: int
    entry_idx: int
    parameter_name: str
    parameter_offset: int
    parameter_total_numel: int
    parameter_shape: Tuple[int, ...]


@dataclass(frozen=True)
class _GroupLayout:
    group_idx: int
    row_index: torch.Tensor
    max_group_count: int


class JointSaliencyCollector:
    """Own the static row mapping, fixed workspace, and oracle-time capture."""

    def __init__(
        self,
        *,
        settings: JointSaliencySettings,
        rows: Sequence[JointSaliencyRow],
        group_layouts: Sequence[_GroupLayout],
        workspace: JointSaliencyWorkspace,
    ) -> None:
        self.settings = settings.validated()
        self.rows = tuple(rows)
        self._group_layouts = tuple(group_layouts)
        self.workspace = workspace

    @classmethod
    def initialize(
        cls,
        lossy_cp: Any,
        settings: JointSaliencySettings,
    ) -> "JointSaliencyCollector | None":
        settings = settings.validated()
        if not settings.enabled:
            return None
        selection_mode = str(getattr(lossy_cp, "selection_mode", "hessian"))
        if selection_mode != "hessian":
            raise RuntimeError(
                "joint saliency telemetry is defined only for the hessian "
                f"selection mode, got {selection_mode!r}"
            )
        if bool(getattr(lossy_cp, "e_hat_proxy", True)):
            raise RuntimeError(
                "joint saliency all-element telemetry requires "
                "--proposal-e-hat-proxy-off so the BF16 theta_cp residual "
                "is resident on GPU"
            )
        groups = list(getattr(lossy_cp, "_groups", ()) or ())
        if not groups:
            raise RuntimeError("joint saliency telemetry requires initialized flat groups")
        theta_state = getattr(lossy_cp, "_theta_cp_score_state", None)
        theta_values = getattr(theta_state, "values", None)
        if not isinstance(theta_values, torch.Tensor) or not theta_values.is_cuda:
            raise RuntimeError("joint saliency telemetry requires resident CUDA theta_cp")
        device = theta_values.device

        tname_list = list(getattr(lossy_cp, "_tname_list", ()) or ())
        resolve_fragments = getattr(lossy_cp, "active_tracked_fragment_specs", None)
        if not callable(resolve_fragments):
            raise RuntimeError(
                "joint saliency requires the active fragment identity API"
            )
        active_fragments = tuple(resolve_fragments())
        fragment_by_name = {
            str(spec.runtime_name): spec for spec in active_fragments
        }
        if len(fragment_by_name) != len(active_fragments):
            raise RuntimeError("joint saliency active fragment identities are not unique")
        rows: list[JointSaliencyRow] = []
        layouts: list[_GroupLayout] = []
        for group_idx, grp in enumerate(groups):
            name_indices = list(grp.get("entry_name_idx_host", ()) or ())
            group_counts = list(grp.get("entry_group_count_host", ()) or ())
            elem_starts = list(grp.get("entry_elem_start_host", ()) or ())
            if len(elem_starts) != len(group_counts):
                raise RuntimeError("joint saliency physical entry metadata is misaligned")
            row_indices: list[int] = []
            for entry_idx, (name_idx_raw, group_count_raw) in enumerate(
                zip(name_indices, group_counts)
            ):
                name_idx = int(name_idx_raw)
                name = (
                    str(tname_list[name_idx])
                    if 0 <= name_idx < len(tname_list)
                    else ""
                )
                spec = fragment_by_name.get(name)
                if spec is None:
                    raise RuntimeError(
                        "joint saliency cannot map tracked fragment to an "
                        f"expert: group={group_idx} entry={entry_idx} name={name!r}"
                    )
                row_numel = int(group_count_raw) * max(
                    1, int(lossy_cp.group_size)
                )
                if int(spec.numel) != row_numel:
                    raise RuntimeError(
                        "joint saliency fragment metadata mismatch: "
                        f"name={name!r} fragment_numel={spec.numel} "
                        f"row_numel={row_numel}"
                    )
                if spec.optimizer_group_idx is not None and (
                    int(spec.optimizer_group_idx)
                    != int(grp.get("optimizer_group_idx", -1))
                    or int(spec.optimizer_local_offset)
                    != int(elem_starts[entry_idx])
                ):
                    raise RuntimeError(
                        "joint saliency optimizer fragment placement mismatch: "
                        f"name={name!r} expected_group="
                        f"{spec.optimizer_group_idx} actual_group="
                        f"{grp.get('optimizer_group_idx')} expected_offset="
                        f"{spec.optimizer_local_offset} actual_offset="
                        f"{elem_starts[entry_idx]}"
                    )
                row = len(rows)
                row_indices.append(row)
                local_expert = int(spec.expert_id[1])
                global_expert = int(settings.global_expert_base) + local_expert
                if (
                    int(settings.experts_per_layer) > 0
                    and global_expert >= int(settings.experts_per_layer)
                ):
                    raise RuntimeError(
                        "joint saliency global expert id is outside the configured "
                        f"layer range: local={local_expert} "
                        f"base={settings.global_expert_base} "
                        f"global={global_expert} "
                        f"experts_per_layer={settings.experts_per_layer}"
                    )
                rows.append(
                    JointSaliencyRow(
                        row=row,
                        name=name,
                        layer=int(spec.expert_id[0]),
                        local_expert=local_expert,
                        global_expert=global_expert,
                        numel=row_numel,
                        group_idx=int(group_idx),
                        entry_idx=int(entry_idx),
                        parameter_name=str(spec.parameter_name),
                        parameter_offset=int(spec.parameter_offset),
                        parameter_total_numel=int(spec.parameter_numel),
                        parameter_shape=tuple(spec.parameter_shape),
                    )
                )
            if len(row_indices) != len(group_counts):
                raise RuntimeError("joint saliency entry metadata is misaligned")
            row_tensor = torch.tensor(
                row_indices,
                dtype=torch.int32,
                device=device,
            )
            layouts.append(
                _GroupLayout(
                    group_idx=int(group_idx),
                    row_index=row_tensor,
                    max_group_count=max((int(x) for x in group_counts), default=0),
                )
            )
        if not rows:
            raise RuntimeError("joint saliency found no active expert fragments")
        mapped_numel = sum(int(row.numel) for row in rows)
        expected_numel = int(getattr(lossy_cp, "total_tracked", -1))
        if expected_numel < 0 or mapped_numel != expected_numel:
            raise RuntimeError(
                "joint saliency tracked coverage mismatch: "
                f"mapped={mapped_numel} expected={expected_numel}"
            )
        workspace = JointSaliencyWorkspace(
            n_rows=len(rows),
            n_bins=int(settings.n_bins),
            device=device,
        )
        return cls(
            settings=settings,
            rows=rows,
            group_layouts=layouts,
            workspace=workspace,
        )

    def owned_cuda_tensors(self) -> Mapping[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {
            f"workspace.{name}": tensor
            for name, tensor in self.workspace.owned_cuda_tensors().items()
        }
        for layout in self._group_layouts:
            out[f"group[{layout.group_idx}].row_index"] = layout.row_index
        return out

    @torch.no_grad()
    def capture(self, *, lossy_cp: Any, step: int) -> tuple[Path, Path]:
        """Capture one pre-dispatch rank shard without mutating model state."""

        h_lut = getattr(lossy_cp, "_h_lookup_lut", None)
        h_values = getattr(h_lut, "values", None)
        h_log_min = getattr(h_lut, "log_v_min", None)
        h_inv_width = getattr(h_lut, "inv_log_v_width", None)
        age_state = getattr(lossy_cp, "_age2bit_score_state", None)
        age_words = getattr(age_state, "age_words", None)
        required = (h_values, h_log_min, h_inv_width, age_words)
        if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in required):
            raise RuntimeError("joint saliency telemetry source state is incomplete")

        stream = torch.cuda.current_stream(device=self.workspace.device)
        started = time.perf_counter()
        from . import diagnostics as _diagnostics

        _diagnostics.set_iter(int(step))
        _diagnostics.vram_audit_step(
            "oracle_joint_saliency_begin",
            lossy_cp=lossy_cp,
            extra={"joint_saliency_workspace_bytes": self.workspace.workspace_bytes},
        )
        self.workspace.reset()
        for layout in self._group_layouts:
            grp = lossy_cp._groups[int(layout.group_idx)]
            if int(layout.max_group_count) <= 0:
                continue
            launch_joint_saliency_group(
                linear=grp["m_flat"],
                v=grp["v_flat"],
                p=grp["p_flat"],
                theta_cp=grp["theta_cp_flat"],
                age_words=age_words,
                h_lut_values=h_values,
                h_lut_log_v_min=h_log_min,
                h_lut_inv_log_v_width=h_inv_width,
                entry_elem_start=grp["entry_elem_start"],
                entry_age_start=grp["entry_age_elem_start"],
                entry_group_count=grp["entry_group_count"],
                entry_row=layout.row_index,
                workspace=self.workspace,
                step=int(step),
                group_size=int(lossy_cp.group_size),
                score_policy_id=int(getattr(lossy_cp, "score_policy_id", 0)),
                age_force_period=int(getattr(lossy_cp, "age_force_period", 128)),
                log10_min=float(self.settings.log10_min),
                log10_max=float(self.settings.log10_max),
                max_group_count=int(layout.max_group_count),
            )
        host = self.workspace.copy_to_host(stream)
        _diagnostics.vram_audit_step(
            "oracle_joint_saliency_end",
            lossy_cp=lossy_cp,
            extra={"joint_saliency_workspace_bytes": self.workspace.workspace_bytes},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        max_parameter_ndim = max(
            (len(row.parameter_shape) for row in self.rows),
            default=0,
        )
        parameter_shapes = np.full(
            (len(self.rows), max_parameter_ndim),
            -1,
            dtype=np.int64,
        )
        parameter_ndim = np.zeros(len(self.rows), dtype=np.int16)
        for row in self.rows:
            parameter_ndim[row.row] = len(row.parameter_shape)
            if row.parameter_shape:
                parameter_shapes[row.row, :len(row.parameter_shape)] = row.parameter_shape

        arrays = {
            "expert_ids": np.asarray(
                [(row.layer, row.global_expert) for row in self.rows],
                dtype=np.int32,
            ),
            "local_expert_ids": np.asarray(
                [(row.layer, row.local_expert) for row in self.rows],
                dtype=np.int32,
            ),
            "row_names": np.asarray([row.name for row in self.rows], dtype=np.str_),
            "row_numel": np.asarray([row.numel for row in self.rows], dtype=np.int64),
            "row_group_entry": np.asarray(
                [(row.group_idx, row.entry_idx) for row in self.rows], dtype=np.int32,
            ),
            "parameter_names": np.asarray(
                [row.parameter_name for row in self.rows], dtype=np.str_,
            ),
            "parameter_offsets": np.asarray(
                [row.parameter_offset for row in self.rows], dtype=np.int64,
            ),
            "parameter_total_numel": np.asarray(
                [row.parameter_total_numel for row in self.rows], dtype=np.int64,
            ),
            "parameter_ndim": parameter_ndim,
            "parameter_shapes": parameter_shapes,
            "log10_edges": np.linspace(
                self.settings.log10_min,
                self.settings.log10_max,
                self.settings.n_bins + 1,
                dtype=np.float32,
            ),
            "hist_count": host.hist_count.numpy().astype(np.int64, copy=True),
            "hist_mass": host.hist_mass.numpy().astype(np.float64, copy=True),
            "age_count": host.age_count.numpy().astype(np.int64, copy=True),
            "force_count": host.force_count.numpy().astype(np.int64, copy=True),
            "int_stats": host.int_stats.numpy().astype(np.int64, copy=True),
            "score_sum": host.score_sum.numpy().astype(np.float64, copy=True),
            "score_min_log10": host.score_min_log.numpy().copy(),
            "score_max_log10": host.score_max_log.numpy().copy(),
        }
        rank, world = _rank_world()
        int_stats = arrays["int_stats"]
        local_total = int(int_stats[:, TOTAL].sum())
        local_positive = int(int_stats[:, POSITIVE].sum())
        local_zero = int(int_stats[:, ZERO].sum())
        local_nonfinite = int(int_stats[:, NONFINITE].sum())
        local_clipped_low = int(int_stats[:, CLIPPED_LOW].sum())
        local_clipped_high = int(int_stats[:, CLIPPED_HIGH].sum())
        expected_total = sum(int(row.numel) for row in self.rows)
        observed_age = int(arrays["age_count"].sum())
        observed_hist = int(arrays["hist_count"].sum())
        row_expected = arrays["row_numel"]
        row_total = int_stats[:, TOTAL]
        row_classified = (
            int_stats[:, POSITIVE]
            + int_stats[:, ZERO]
            + int_stats[:, NONFINITE]
        )
        row_age = arrays["age_count"].sum(axis=1)
        row_hist = arrays["hist_count"].sum(axis=1)
        if (
            local_total != expected_total
            or observed_age != expected_total
            or local_positive + local_zero + local_nonfinite != expected_total
            or observed_hist != local_positive
            or not np.array_equal(row_total, row_expected)
            or not np.array_equal(row_classified, row_expected)
            or not np.array_equal(row_age, row_expected)
            or not np.array_equal(row_hist, int_stats[:, POSITIVE])
        ):
            raise RuntimeError(
                "joint saliency capture coverage mismatch: "
                f"expected={expected_total} total={local_total} "
                f"age={observed_age} classified="
                f"{local_positive + local_zero + local_nonfinite} "
                f"hist={observed_hist} positive={local_positive}"
            )
        summary = histogram_budget_summary(
            hist_count=arrays["hist_count"],
            hist_mass=arrays["hist_mass"],
            log_edges=arrays["log10_edges"],
            total_count=int(int_stats[:, TOTAL].sum()),
            budgets=self.settings.budgets,
            nonfinite_count=local_nonfinite,
            clipped_low_count=local_clipped_low,
            clipped_high_count=local_clipped_high,
        )
        score_policy_id = int(getattr(lossy_cp, "score_policy_id", 0))
        score_policy_name = str(
            getattr(
                lossy_cp,
                "score_policy",
                "abs_h_taylor_first_and_second_order",
            )
        )
        if score_policy_id == 0:
            score_formula = (
                "max(0,-m_t*e+0.5*abs(H_tilde(v_t))*e^2); "
                "e=p_live_fp32-theta_cp_bf16"
            )
        elif score_policy_id == 1:
            score_formula = (
                "abs(H_tilde(v_t))*e^2; "
                "e=p_live_fp32-theta_cp_bf16"
            )
        else:
            raise RuntimeError(
                f"joint saliency unsupported score_policy_id={score_policy_id}"
            )
        manifest = {
            "mode": "oracle_joint_saliency",
            "step": int(step),
            "rank": int(rank),
            "world": int(world),
            "expert_id_space": "global",
            "global_expert_base": int(self.settings.global_expert_base),
            "ep_rank": int(self.settings.ep_rank),
            "ep_size": int(self.settings.ep_size),
            "experts_per_layer": int(self.settings.experts_per_layer),
            "state_phase": "pre_dispatch_selection_input_for_T",
            "checkpoint_visible_through": int(step) - 1,
            "selection_input_step": int(step),
            "all_tracked_elements": True,
            "score_source": "production_bf16_theta_cp_scoring_residual",
            "score_source_dtype": "theta_cp=bfloat16;p_live,m_t,v_t=float32",
            "selection_mode": "hessian",
            "score_policy_id": score_policy_id,
            "score_policy_name": score_policy_name,
            "score_formula": score_formula,
            "is_actual_loss": False,
            "directional_hvp_reference": (
                "separate_sampled_post_dispatch_independent_oracle;not_same_phase"
            ),
            "cutoff_method": "rank_local_histogram_estimate",
            "budgets": summary,
            "budget_ratios": [float(x) for x in self.settings.budgets],
            "n_rows": int(len(self.rows)),
            "n_bins": int(self.settings.n_bins),
            "local_total_elements": local_total,
            "local_positive_elements": local_positive,
            "local_zero_elements": local_zero,
            "local_nonfinite_elements": local_nonfinite,
            "local_clipped_low": local_clipped_low,
            "local_clipped_high": local_clipped_high,
            "local_age_code_count": arrays["age_count"].sum(axis=0).tolist(),
            "local_force_candidate_count": int(arrays["force_count"].sum()),
            "age_force_period": int(getattr(lossy_cp, "age_force_period", 128)),
            "h_lut_built_at": int(getattr(lossy_cp, "_h_quantile_built_at", -1)),
            "hist_count_dtype": "int32_gpu;int64_npz",
            "hist_mass_accumulator_dtype": "float32_gpu_atomic;float64_npz",
            "score_sum_accumulator_dtype": "float32_gpu_atomic;float64_npz",
            "workspace_bytes": int(self.workspace.workspace_bytes),
            "elapsed_ms": float(elapsed_ms),
        }
        return write_joint_saliency_shard(
            output_dir=self.settings.output_dir,
            step=int(step),
            rank=int(rank),
            arrays=arrays,
            manifest=manifest,
        )


def _rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
    return 0, 1


__all__ = [
    "DEFAULT_BUDGETS",
    "JointSaliencyCollector",
    "JointSaliencyRow",
    "JointSaliencySettings",
    "histogram_budget_summary",
]
