from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from megatron.proposal.telemetry.joint_saliency_analysis import DEFAULT_BUDGETS
from megatron.proposal.telemetry.joint_saliency_io import (
    merge_joint_saliency_shards,
    write_joint_saliency_shard,
)


def _arrays(
    *,
    expert: int,
    count: tuple[int, int] = (8, 2),
    total: int = 20,
    nonfinite: int = 0,
    clipped_low: int = 0,
    clipped_high: int = 0,
    parameter: bool = False,
    offset: int = 0,
    row_numel: int = 10,
    parameter_total: int = 10,
    fragment_name: str = "fragment",
) -> dict[str, np.ndarray]:
    positive = int(sum(count))
    arrays = {
        "expert_ids": np.asarray([[3, expert]], dtype=np.int32),
        "log10_edges": np.asarray([-2.0, -1.0, 0.0], dtype=np.float32),
        "hist_count": np.asarray([count], dtype=np.int64),
        "hist_mass": np.asarray([[float(count[0]), 10.0 * count[1]]], dtype=np.float64),
        "age_count": np.asarray([[total, 0, 0, 0]], dtype=np.int64),
        "force_count": np.asarray([0], dtype=np.int64),
        "int_stats": np.asarray(
            [[
                total,
                positive,
                total - positive - nonfinite,
                nonfinite,
                clipped_low,
                clipped_high,
            ]],
            dtype=np.int64,
        ),
        "score_sum": np.asarray([float(count[0] + 10 * count[1])], dtype=np.float64),
        "score_min_log10": np.asarray([-1.9], dtype=np.float32),
        "score_max_log10": np.asarray([0.5 if clipped_high else -0.1], dtype=np.float32),
    }
    if parameter:
        arrays.update(
            {
                "row_names": np.asarray([fragment_name], dtype=np.str_),
                "row_numel": np.asarray([row_numel], dtype=np.int64),
                "parameter_names": np.asarray(["w1.weight"], dtype=np.str_),
                "parameter_offsets": np.asarray([offset], dtype=np.int64),
                "parameter_total_numel": np.asarray(
                    [parameter_total], dtype=np.int64,
                ),
                "parameter_ndim": np.asarray([1], dtype=np.int16),
                "parameter_shapes": np.asarray(
                    [[parameter_total]], dtype=np.int64,
                ),
            }
        )
    return arrays


def _manifest(*, rank: int, world: int = 2) -> dict[str, object]:
    return {
        "mode": "oracle_joint_saliency",
        "step": 100,
        "rank": rank,
        "world": world,
        "state_phase": "pre_dispatch_selection_input_for_T",
        "checkpoint_visible_through": 99,
        "selection_input_step": 100,
        "score_source": "production_bf16_theta_cp_scoring_residual",
        "score_source_dtype": "theta_cp=bfloat16;p_live,m_t,v_t=float32",
        "selection_mode": "hessian",
        "score_policy_id": 0,
        "score_policy_name": "abs_h_taylor_first_and_second_order",
        "score_formula": "score",
        "is_actual_loss": False,
        "all_tracked_elements": True,
        "n_bins": 2,
        "age_force_period": 512,
        "h_lut_built_at": 100,
        "budget_ratios": list(DEFAULT_BUDGETS),
        # Deliberately bogus rank-local values: merge must never average or
        # copy them into the global summary.
        "budgets": [
            {"budget": ratio, "target_count": 999_999}
            for ratio in DEFAULT_BUDGETS
        ],
    }


def _write(
    root: Path,
    *,
    rank: int,
    arrays: dict[str, np.ndarray] | None = None,
    manifest: dict[str, object] | None = None,
) -> Path:
    _, path = write_joint_saliency_shard(
        output_dir=root,
        step=100,
        rank=rank,
        arrays=arrays or _arrays(expert=rank),
        manifest=manifest or _manifest(rank=rank),
    )
    return path


def test_merge_recomputes_canonical_global_budgets_from_all_elements(
    tmp_path: Path,
) -> None:
    paths = [
        _write(tmp_path, rank=0, arrays=_arrays(expert=0, count=(8, 2), total=20)),
        _write(tmp_path, rank=1, arrays=_arrays(expert=1, count=(10, 10), total=30)),
    ]

    summary, _merged = merge_joint_saliency_shards(paths)
    assert summary["global_budget_ratios"] == list(DEFAULT_BUDGETS)
    assert summary["global_total_elements"] == 50
    assert summary["global_positive_elements"] == 30
    assert [row["target_count"] for row in summary["budgets"]] == [1, 2, 5, 10]
    assert all(row["denominator_count"] == 50 for row in summary["budgets"])
    assert all(row["target_count"] != 999_999 for row in summary["budgets"])


def test_merge_uses_validated_manifest_budget_ratios(tmp_path: Path) -> None:
    ratios = [0.02, 0.05]
    paths = []
    for rank in range(2):
        manifest = _manifest(rank=rank)
        manifest["budget_ratios"] = ratios
        manifest["budgets"] = [
            {"budget": ratio, "target_count": 123}
            for ratio in ratios
        ]
        paths.append(_write(tmp_path, rank=rank, manifest=manifest))

    summary, _merged = merge_joint_saliency_shards(paths)
    assert summary["global_budget_ratios"] == ratios
    assert [row["budget"] for row in summary["budgets"]] == ratios


def test_merge_rejects_duplicate_rank_even_when_manifest_count_matches_world(
    tmp_path: Path,
) -> None:
    rank0 = _write(tmp_path, rank=0)
    with pytest.raises(ValueError, match="duplicate.*rank"):
        merge_joint_saliency_shards([rank0, rank0])


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("state_phase", "post_dispatch"),
        ("score_source_dtype", "float16"),
        ("score_policy_id", 1),
        ("h_lut_built_at", 99),
        ("budget_ratios", [0.02, 0.04, 0.10, 0.20]),
    ],
)
def test_merge_rejects_semantic_invariant_mismatch(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    first = _write(tmp_path, rank=0)
    second_manifest = _manifest(rank=1)
    second_manifest[field] = replacement
    if field == "budget_ratios":
        second_manifest["budgets"] = [
            {"budget": ratio, "target_count": 999_999}
            for ratio in replacement
        ]
    second = _write(tmp_path, rank=1, manifest=second_manifest)
    with pytest.raises(ValueError, match="configuration mismatch"):
        merge_joint_saliency_shards([first, second])


def test_nonfinite_and_clipped_high_tail_expose_validity_and_open_edge(
    tmp_path: Path,
) -> None:
    paths = [
        _write(
            tmp_path,
            rank=0,
            arrays=_arrays(
                expert=0,
                count=(7, 2),
                total=20,
                nonfinite=1,
                clipped_high=2,
            ),
        ),
        _write(tmp_path, rank=1, arrays=_arrays(expert=1, count=(8, 2), total=20)),
    ]
    summary, _merged = merge_joint_saliency_shards(paths)
    top = summary["budgets"][0]
    assert top["budget_valid"] is False
    assert top["budget_validity"] == "invalid_nonfinite_or_hist_mass"
    assert top["cutoff_log10_upper"] is None
    assert top["cutoff_edge_semantics"] == (
        "upper_bound_open_due_clipped_high_tail"
    )
    assert top["nonfinite_count"] == 1
    assert top["clipped_high_count"] == 2


def test_parameter_fragments_merge_by_canonical_name_and_preserve_fragment_ids(
    tmp_path: Path,
) -> None:
    first = _write(
        tmp_path,
        rank=0,
        arrays=_arrays(
            expert=7,
            count=(4, 2),
            total=6,
            parameter=True,
            offset=0,
            row_numel=6,
            parameter_total=10,
            fragment_name="zero_fragment_rank0",
        ),
    )
    second = _write(
        tmp_path,
        rank=1,
        arrays=_arrays(
            expert=7,
            count=(2, 2),
            total=4,
            parameter=True,
            offset=6,
            row_numel=4,
            parameter_total=10,
            fragment_name="zero_fragment_rank1",
        ),
    )

    summary, merged = merge_joint_saliency_shards([second, first])
    assert summary["global_parameter_count"] == 1
    assert summary["global_fragment_count"] == 2
    np.testing.assert_array_equal(merged["parameter_expert_ids"], [[3, 7]])
    np.testing.assert_array_equal(merged["parameter_names"], ["w1.weight"])
    np.testing.assert_array_equal(
        merged["parameter_hist_count"],
        np.asarray([[6, 4]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        merged["row_names"],
        ["zero_fragment_rank0", "zero_fragment_rank1"],
    )
    np.testing.assert_array_equal(merged["fragment_offsets"], [0, 6])


def test_ep_shards_merge_full_parameters_by_global_expert_id(
    tmp_path: Path,
) -> None:
    paths = []
    for rank, (ep_rank, base) in enumerate(((0, 0), (1, 32))):
        arrays = _arrays(
            expert=base,
            count=(8, 2),
            total=10,
            parameter=True,
            row_numel=10,
            parameter_total=10,
            fragment_name=f"ep{ep_rank}_local0",
        )
        arrays["local_expert_ids"] = np.asarray([[3, 0]], dtype=np.int32)
        manifest = _manifest(rank=rank)
        manifest.update(
            {
                "expert_id_space": "global",
                "global_expert_base": base,
                "ep_rank": ep_rank,
                "ep_size": 4,
                "experts_per_layer": 128,
            }
        )
        paths.append(
            _write(
                tmp_path,
                rank=rank,
                arrays=arrays,
                manifest=manifest,
            )
        )

    summary, merged = merge_joint_saliency_shards(paths)
    assert summary["global_parameter_count"] == 2
    np.testing.assert_array_equal(merged["expert_ids"], [[3, 0], [3, 32]])
    np.testing.assert_array_equal(
        merged["parameter_expert_ids"], [[3, 0], [3, 32]],
    )


def test_ep_global_expert_mapping_mismatch_is_rejected(tmp_path: Path) -> None:
    arrays = _arrays(expert=31)
    arrays["local_expert_ids"] = np.asarray([[3, 0]], dtype=np.int32)
    manifest = _manifest(rank=0, world=1)
    manifest.update(
        {
            "expert_id_space": "global",
            "global_expert_base": 32,
            "ep_rank": 1,
            "ep_size": 4,
            "experts_per_layer": 128,
        }
    )
    path = _write(
        tmp_path,
        rank=0,
        arrays=arrays,
        manifest=manifest,
    )
    with pytest.raises(ValueError, match="local/global expert id mapping"):
        merge_joint_saliency_shards([path])


@pytest.mark.parametrize("second_offset", [5, 7])
def test_parameter_fragment_overlap_or_gap_is_rejected(
    tmp_path: Path,
    second_offset: int,
) -> None:
    first = _write(
        tmp_path,
        rank=0,
        arrays=_arrays(
            expert=7,
            count=(6, 0),
            total=6,
            parameter=True,
            offset=0,
            row_numel=6,
            parameter_total=10,
            fragment_name="first",
        ),
    )
    second_numel = 10 - second_offset
    second = _write(
        tmp_path,
        rank=1,
        arrays=_arrays(
            expert=7,
            count=(second_numel, 0),
            total=second_numel,
            parameter=True,
            offset=second_offset,
            row_numel=second_numel,
            parameter_total=10,
            fragment_name="second",
        ),
    )
    with pytest.raises(ValueError, match="fragment (overlap|gap)"):
        merge_joint_saliency_shards([first, second])


def test_manifest_writer_rejects_conflicting_step_or_rank(tmp_path: Path) -> None:
    bad = _manifest(rank=1)
    with pytest.raises(ValueError, match="does not match writer"):
        write_joint_saliency_shard(
            output_dir=tmp_path,
            step=100,
            rank=0,
            arrays=_arrays(expert=0),
            manifest=bad,
        )
