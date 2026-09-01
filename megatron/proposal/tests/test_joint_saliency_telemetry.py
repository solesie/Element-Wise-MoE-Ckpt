from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from megatron.proposal.cli_args import add_proposal_args
from megatron.proposal.config import ProposalConfig
from megatron.proposal.telemetry.joint_saliency import (
    JointSaliencyCollector,
    JointSaliencySettings,
    histogram_budget_summary,
)
from megatron.proposal.telemetry.joint_saliency_gpu import (
    TOTAL,
    JointSaliencyWorkspace,
    launch_joint_saliency_group,
)
from megatron.proposal.telemetry.joint_saliency_io import (
    load_joint_saliency_shard,
    merge_joint_saliency_shards,
    write_joint_saliency_shard,
)
from megatron.proposal.storage.expert_partition import (
    build_active_tracked_fragment_specs,
)


def test_joint_saliency_is_disabled_without_allocating_workspace() -> None:
    assert (
        JointSaliencyCollector.initialize(
            object(),
            JointSaliencySettings(enabled=False),
        )
        is None
    )


def test_joint_saliency_settings_fail_closed() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        JointSaliencySettings(enabled=True, n_bins=8).validated()
    with pytest.raises(ValueError, match="log10_max"):
        JointSaliencySettings(
            enabled=True,
            log10_min=2.0,
            log10_max=2.0,
        ).validated()
    with pytest.raises(ValueError, match="experts_per_layer"):
        JointSaliencySettings(enabled=True).validated()
    with pytest.raises(ValueError, match="global_expert_base"):
        JointSaliencySettings(
            enabled=True,
            experts_per_layer=128,
            global_expert_base=-1,
        ).validated()
    settings = JointSaliencySettings(
        enabled=True,
        experts_per_layer=128,
        global_expert_base=32,
        ep_rank=1,
        ep_size=4,
    ).validated()
    assert settings.global_expert_base == 32


def test_fragment_identity_uses_original_name_and_offset_not_runtime_suffix() -> None:
    parameter = torch.nn.Parameter(torch.zeros(10))
    specs = build_active_tracked_fragment_specs(
        active_names=["runtime-rank-local-id"],
        name_to_expert={"runtime-rank-local-id": (3, 7)},
        fragment_meta={
            "runtime-rank-local-id": {
                "name": "layers.3.experts.7.w1.weight",
                "param_offset": 4,
                "numel": 6,
                "group_idx": 2,
                "local_offset": 19,
            }
        },
        name_to_param={"layers.3.experts.7.w1.weight": parameter},
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.runtime_name == "runtime-rank-local-id"
    assert spec.parameter_name == "layers.3.experts.7.w1.weight"
    assert (spec.parameter_offset, spec.numel, spec.parameter_numel) == (4, 6, 10)
    assert spec.expert_id == (3, 7)


def test_fragment_identity_requires_active_metadata_set_identity() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(RuntimeError, match="fragment set"):
        build_active_tracked_fragment_specs(
            active_names=["active"],
            name_to_expert={"active": (1, 0)},
            fragment_meta={
                "different": {
                    "name": "p",
                    "param_offset": 0,
                    "numel": 4,
                }
            },
            name_to_param={"p": parameter},
        )


def test_histogram_budget_summary_handles_heavy_tail_without_linear_bins() -> None:
    # Eight log bins with a deliberately heavy upper tail.  The summary must
    # be monotone and must advertise its boundary-bin approximation.
    count = np.asarray([[900, 50, 20, 10, 8, 6, 4, 2]], dtype=np.int64)
    mass = np.asarray([[1, 2, 3, 4, 8, 16, 32, 64]], dtype=np.float64)
    edges = np.linspace(-8.0, 0.0, 9)
    rows = histogram_budget_summary(
        hist_count=count,
        hist_mass=mass,
        log_edges=edges,
        budgets=(0.01, 0.04, 0.10, 0.20),
    )

    assert [row["budget"] for row in rows] == [0.01, 0.04, 0.10, 0.20]
    assert all(row["cutoff_method"] == "histogram_estimate" for row in rows)
    selected = [int(row["selected_count_including_boundary_bin"]) for row in rows]
    captured = [float(row["captured_hist_mass_ratio"]) for row in rows]
    assert selected == sorted(selected)
    assert captured == sorted(captured)
    assert captured[0] > 0.4  # the top 1% carries most of this synthetic mass


def test_histogram_budget_denominator_is_all_elements_not_only_positive() -> None:
    rows = histogram_budget_summary(
        hist_count=np.asarray([[10, 10]], dtype=np.int64),
        hist_mass=np.asarray([[1.0, 9.0]], dtype=np.float64),
        log_edges=np.asarray([-2.0, -1.0, 0.0]),
        total_count=100,
        budgets=(0.10, 0.40),
    )

    assert rows[0]["target_count"] == 10
    assert rows[0]["denominator_count"] == 100
    assert rows[0]["positive_count"] == 20
    assert rows[0]["cutoff_method"] == "histogram_estimate"
    # At 40% the cutoff reaches the 80 zero-score elements.  A >= 0 predicate
    # ties the whole population, which must be made visible in the manifest.
    assert rows[1]["target_count"] == 40
    assert rows[1]["cutoff_method"] == "zero_score_tie"
    assert rows[1]["selected_count_including_boundary_bin"] == 100


def test_atomic_npz_manifest_roundtrip_and_checksum(tmp_path: Path) -> None:
    arrays = {
        "hist_count": np.arange(12, dtype=np.int64).reshape(3, 4),
        "row_names": np.asarray(["a", "b", "c"], dtype=np.str_),
    }
    _payload, manifest_path = write_joint_saliency_shard(
        output_dir=tmp_path,
        step=100,
        rank=3,
        arrays=arrays,
        manifest={"mode": "oracle_joint_saliency", "rank": 3, "world": 8},
    )
    manifest, loaded = load_joint_saliency_shard(manifest_path)
    assert manifest["mode"] == "oracle_joint_saliency"
    np.testing.assert_array_equal(loaded["hist_count"], arrays["hist_count"])
    np.testing.assert_array_equal(loaded["row_names"], arrays["row_names"])

    payload_path = tmp_path / manifest["payload"]
    with payload_path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_joint_saliency_shard(manifest_path)


def test_cli_config_carries_joint_saliency_settings(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    add_proposal_args(parser)
    args = parser.parse_args(
        [
            "--proposal-independent-oracle-enable",
            "--proposal-joint-saliency-bins",
            "1024",
            "--proposal-joint-saliency-log-dir",
            str(tmp_path),
        ]
    )
    config = ProposalConfig.from_args(args, "/tmp/joint-saliency-config")
    assert config.independent_oracle_enable is True
    assert not hasattr(config, "joint_saliency_telemetry_enable")
    assert config.joint_saliency_bins == 1024
    assert config.joint_saliency_log_dir == tmp_path


def test_joint_saliency_has_no_enable_gate_besides_independent_oracle() -> None:
    parser = argparse.ArgumentParser()
    add_proposal_args(parser)
    action_dests = {action.dest for action in parser._actions}
    assert "proposal_joint_saliency_telemetry_enable" not in action_dests
    assert "joint_saliency_telemetry_enable" not in ProposalConfig.__dataclass_fields__

    proposal_root = Path(__file__).resolve().parents[1]
    engine = (proposal_root / "checkpoint" / "lossy_checkpoint_engine.py").read_text(
        encoding="utf-8"
    )
    init_start = engine.index("# Joint saliency is part of the independent-oracle")
    init_end = engine.index("        try:", init_start)
    init_body = engine[init_start:init_end]
    assert "independent_oracle_enable" in init_body
    assert "joint_saliency_telemetry_enable" not in init_body


def test_world_merge_rejects_incomplete_shards_and_sums_duplicate_experts(
    tmp_path: Path,
) -> None:
    manifests = []
    for rank in range(2):
        _payload, manifest_path = write_joint_saliency_shard(
            output_dir=tmp_path,
            step=100,
            rank=rank,
            arrays={
                "expert_ids": np.asarray([[3, 7]], dtype=np.int32),
                "log10_edges": np.asarray([-2.0, -1.0, 0.0], dtype=np.float32),
                "hist_count": np.asarray([[rank + 1, 2]], dtype=np.int64),
                "hist_mass": np.asarray([[1.0, rank + 2.0]], dtype=np.float64),
                "age_count": np.asarray([[1, 2, 3, 4]], dtype=np.int64),
                "force_count": np.asarray([rank], dtype=np.int64),
                "int_stats": np.asarray([[10, 9, 1, 0, 0, 0]], dtype=np.int64),
                "score_sum": np.asarray([3.0], dtype=np.float64),
                "score_min_log10": np.asarray([-1.5], dtype=np.float32),
                "score_max_log10": np.asarray([-0.1], dtype=np.float32),
            },
            manifest={
                "mode": "oracle_joint_saliency",
                "step": 100,
                "rank": rank,
                "world": 2,
                "state_phase": "post_dispatch_recovery_image_visible_through_T",
                "score_source": "diag",
                "score_formula": "score",
                "is_actual_loss": False,
                "n_bins": 2,
                "age_force_period": 512,
            },
        )
        manifests.append(manifest_path)

    with pytest.raises(ValueError, match="incomplete"):
        merge_joint_saliency_shards(manifests[:1])
    summary, merged = merge_joint_saliency_shards(manifests)
    assert summary["ranks"] == [0, 1]
    np.testing.assert_array_equal(merged["expert_ids"], [[3, 7]])
    np.testing.assert_array_equal(merged["expert_hist_count"], [[3, 4]])
    assert int(merged["expert_force_count"][0]) == 1


def test_gpu_aggregator_has_no_raw_score_cpu_or_sort_path() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "telemetry"
        / "joint_saliency_gpu.py"
    ).read_text(encoding="utf-8")
    assert ".cpu()" not in source
    assert "torch.sort" not in source
    assert "torch.topk" not in source
    assert "_proposal_element_score_from_theta_cp" in source
    assert "_age2_load_code(age_words_ptr, age_idx, mask)" in source
    assert "grid = (triton.cdiv(max_elements, block), n_entries)" in source


def test_joint_saliency_capture_is_inside_the_single_dispatch_gateway() -> None:
    controller = (
        Path(__file__).resolve().parents[1] / "controller.py"
    ).read_text(encoding="utf-8")
    start = controller.index("def proposal_dispatch_post_iter")
    end = controller.index("def proposal_finalize_post_iter", start)
    body = controller[start:end]
    assert body.count("joint_saliency.capture") == 1
    assert body.index("joint_saliency.capture") < body.index(
        "_submit_post_step_checkpoint_task(task)"
    )
    assert "pre_dispatch_selection_input_for_T" in (
        Path(__file__).resolve().parents[1]
        / "telemetry"
        / "joint_saliency.py"
    ).read_text(encoding="utf-8")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "group_counts",
    [
        (1, 2),
        # 65,536+ 256-element blocks: this used to exceed CUDA grid.y on
        # 1.3B/13B-sized parameter entries before the grid-axis fix.
        ((65535 * 256 + 8) // 8,),
    ],
)
def test_joint_saliency_h100_launch_partial_and_large_entry(
    group_counts: tuple[int, ...],
) -> None:
    device = torch.device("cuda")
    group_size = 8
    counts = [int(x) for x in group_counts]
    starts = []
    cursor = 0
    for count in counts:
        starts.append(cursor)
        cursor += count * group_size
    total = cursor
    flat = torch.zeros(total, dtype=torch.float32, device=device)
    theta_cp = torch.zeros(total, dtype=torch.bfloat16, device=device)
    age_words = torch.zeros(
        (total + 15) // 16,
        dtype=torch.int32,
        device=device,
    )
    h_lut = torch.ones(8, dtype=torch.float32, device=device)
    log_v_min = torch.full((1,), -70.0, dtype=torch.float32, device=device)
    inv_log_v_width = torch.ones(1, dtype=torch.float32, device=device)
    entry_starts = torch.tensor(starts, dtype=torch.int64, device=device)
    entry_counts = torch.tensor(counts, dtype=torch.int32, device=device)
    entry_rows = torch.arange(len(counts), dtype=torch.int32, device=device)
    workspace = JointSaliencyWorkspace(
        n_rows=len(counts),
        n_bins=16,
        device=device,
    )
    workspace.reset()
    launch_joint_saliency_group(
        linear=flat,
        v=flat,
        p=flat,
        theta_cp=theta_cp,
        age_words=age_words,
        h_lut_values=h_lut,
        h_lut_log_v_min=log_v_min,
        h_lut_inv_log_v_width=inv_log_v_width,
        entry_elem_start=entry_starts,
        entry_age_start=entry_starts,
        entry_group_count=entry_counts,
        entry_row=entry_rows,
        workspace=workspace,
        step=100,
        group_size=group_size,
        score_policy_id=0,
        age_force_period=512,
        log10_min=-30.0,
        log10_max=10.0,
        max_group_count=max(counts),
    )
    host = workspace.copy_to_host(torch.cuda.current_stream())
    assert int(host.int_stats[:, TOTAL].sum().item()) == total
    assert int(host.age_count.sum().item()) == total
