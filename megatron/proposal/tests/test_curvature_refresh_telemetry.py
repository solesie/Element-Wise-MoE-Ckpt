import json
import pathlib
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from megatron.proposal.refresh.hessian_refresh import (
    _finalize_curvature_views_and_record_telemetry,
)
from megatron.proposal.telemetry.curvature_progress import CurvatureProgressTelemetry


def test_partition_fragment_uses_original_h_offset_and_physical_v_offset(
    tmp_path,
    monkeypatch,
):
    """Reproduce the BF16 optimizer layout used by the real 8-rank run."""

    monkeypatch.setenv("RANK", "1")
    runtime_name = "layers.3.experts.7.weight#frag0"
    parameter_name = "layers.3.experts.7.weight"
    spec = SimpleNamespace(
        runtime_name=runtime_name,
        parameter_name=parameter_name,
        parameter_offset=8,
        numel=8,
        parameter_numel=24,
    )
    telemetry = CurvatureProgressTelemetry(
        log_path=str(tmp_path / "h_rank_rank{rank}.jsonl"),
        refresh_snapshot_dir=str(tmp_path / "snapshots"),
        sample_groups=2,
        every=10,
        seed=2024,
        refresh_only=True,
        fragment_specs_provider=lambda: [spec],
    )
    v_flat = torch.full((32,), -1.0, dtype=torch.float32)
    v_flat[12:20] = torch.arange(101, 109, dtype=torch.float32)
    groups = [{
        "p_flat": torch.zeros(32, dtype=torch.float32),
        "m_flat": torch.zeros(32, dtype=torch.float32),
        "v_flat": v_flat,
        "total_aligned": 8,
        "master_off": 0,
        "names": [runtime_name],
        "sizes": [8],
        "entry_elem_start_host": [12],
        "entry_age_elem_start_host": [0],
    }]
    original_h_view = torch.arange(24, dtype=torch.float32)

    telemetry.record_signed_h_refresh(
        step=10,
        groups=groups,
        group_size=4,
        names=[parameter_name],
        signed_h_views=[original_h_view],
        n_anchor_batches=1,
    )

    row = json.loads((tmp_path / "h_rank_rank1.jsonl").read_text())
    assert row["sample_elements"] == 8
    assert row["valid_ratio"] == 1.0
    assert row["abs_h_vs_v_spearman"] == 1.0
    with np.load(row["refresh_snapshot"], allow_pickle=False) as payload:
        # H comes from original parameter offset 8, while v comes from the
        # optimizer flat's physical offset 12.  Reading either as offset zero
        # would make this assertion fail.
        assert payload["abs_h"].tolist() == list(range(8, 16))
        assert payload["v_t"].tolist() == list(range(101, 109))
        assert payload["element_ids"].tolist() == list(range(8))
        assert payload["group_ids"].tolist() == [0] * 4 + [1] * 4


def test_partition_fragment_without_identity_metadata_fails(tmp_path):
    runtime_name = "expert.weight#frag0"
    telemetry = CurvatureProgressTelemetry(
        log_path=str(tmp_path / "h.jsonl"),
        refresh_snapshot_dir=str(tmp_path / "snapshots"),
        sample_groups=1,
        every=1,
        seed=2024,
        refresh_only=True,
        fragment_specs_provider=lambda: [],
    )
    groups = [{
        "p_flat": torch.zeros(8),
        "m_flat": torch.zeros(8),
        "v_flat": torch.ones(8),
        "total_aligned": 8,
        "master_off": 0,
        "names": [runtime_name],
        "sizes": [8],
        "entry_elem_start_host": [0],
        "entry_age_elem_start_host": [0],
    }]
    with pytest.raises(RuntimeError, match="cannot resolve partition fragment"):
        telemetry.record_signed_h_refresh(
            step=1,
            groups=groups,
            group_size=8,
            names=["expert.weight"],
            signed_h_views=[torch.ones(8)],
            n_anchor_batches=1,
        )


def test_spearman_uses_average_ranks_for_ties():
    values = np.asarray([0.0, 0.0, 2.0, 4.0], dtype=np.float32)
    ranks = CurvatureProgressTelemetry._rank(values)
    assert ranks.tolist() == [0.5, 0.5, 2.0, 3.0]
    assert CurvatureProgressTelemetry._spearman(values, values) == pytest.approx(1.0)


def test_refresh_telemetry_cadence_rank_shard_and_replayable_snapshot(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "3")
    telemetry = CurvatureProgressTelemetry(
        log_path=str(tmp_path / "h_rank_rank{rank}.jsonl"),
        refresh_snapshot_dir=str(tmp_path / "snapshots"),
        sample_groups=2,
        every=1000,
        seed=7,
        refresh_only=True,
    )
    v = torch.arange(1, 9, dtype=torch.float32)
    groups = [{
        "p_flat": torch.zeros(8, dtype=torch.float32),
        "m_flat": torch.zeros(8, dtype=torch.float32),
        "v_flat": v,
        "total_aligned": 8,
        "master_off": 0,
        "names": ["expert.weight"],
        "sizes": [8],
    }]
    signed_h = torch.arange(1, 9, dtype=torch.float32)

    telemetry.record_signed_h_refresh(
        step=100,
        groups=groups,
        group_size=4,
        names=["expert.weight"],
        signed_h_views=[signed_h],
        n_anchor_batches=2,
    )
    assert not list(tmp_path.glob("*.jsonl"))

    telemetry.record_signed_h_refresh(
        step=1000,
        groups=groups,
        group_size=4,
        names=["expert.weight"],
        signed_h_views=[signed_h],
        n_anchor_batches=2,
    )
    telemetry.record_signed_h_refresh(
        step=2000,
        groups=groups,
        group_size=4,
        names=["expert.weight"],
        signed_h_views=[signed_h.flip(0)],
        n_anchor_batches=2,
    )

    log_path = tmp_path / "h_rank_rank3.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [1000, 2000]
    assert all(row["rank"] == 3 for row in rows)
    assert rows[0]["abs_h_vs_v_spearman"] == 1.0
    assert "abs_h_spearman_prev" not in rows[0]
    assert rows[1]["abs_h_spearman_prev"] == -1.0
    assert rows[0]["abs_h_min"] == 1.0
    assert rows[0]["abs_h_max"] == 8.0

    snapshot_path = tmp_path / "snapshots" / "h_rank_step0001000_rank00003.npz"
    assert snapshot_path.is_file()
    with np.load(snapshot_path, allow_pickle=False) as payload:
        assert payload["schema_version"].tolist() == [2]
        assert payload["step"].tolist() == [1000]
        assert payload["rank"].tolist() == [3]
        assert payload["abs_h"].tolist() == list(range(1, 9))
        assert payload["v_t"].tolist() == list(range(1, 9))
        assert payload["element_ids"].shape == (8,)
        assert payload["group_ids"].shape == (8,)
    assert len(rows[0]["refresh_snapshot_sha256"]) == 64


def test_anchor_integration_seam_records_group8_and_is_fail_closed(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("RANK", "0")
    requested_log = tmp_path / "integrated_rank0.jsonl"
    monkeypatch.setenv("PROPOSAL_CURVATURE_PROGRESS_LOG", str(requested_log))
    telemetry = CurvatureProgressTelemetry(
        log_path=str(requested_log),
        refresh_snapshot_dir=str(tmp_path / "integrated_snapshots"),
        sample_groups=1,
        every=10,
        seed=2024,
        refresh_only=True,
    )
    v = torch.arange(1, 9, dtype=torch.float32)
    groups = [{
        "p_flat": torch.zeros(8, dtype=torch.float32),
        "m_flat": torch.zeros(8, dtype=torch.float32),
        "v_flat": v,
        "total_aligned": 8,
        "master_off": 0,
        "names": ["expert.weight"],
        "sizes": [8],
    }]
    lossy_cp = type("Lossy", (), {})()
    lossy_cp._groups = groups
    lossy_cp._curvature_progress_telemetry = telemetry
    signed_view = -torch.arange(1, 9, dtype=torch.float32)

    due = _finalize_curvature_views_and_record_telemetry(
        lossy_cp=lossy_cp,
        iteration=10,
        group_size=8,
        names=["expert.weight"],
        views=[signed_view],
        successful_anchor_batches=2,
    )
    assert due is True
    assert signed_view.tolist() == [x / 2.0 for x in range(1, 9)]
    row = json.loads(requested_log.read_text().strip())
    assert row["step"] == 10
    assert row["sample_groups"] == 1
    assert row["sample_elements"] == 8
    assert row["abs_h_vs_v_spearman"] == 1.0
    assert pathlib.Path(row["refresh_snapshot"]).is_file()

    missing = type("Lossy", (), {"_groups": groups})()
    with pytest.raises(RuntimeError, match="not attached"):
        _finalize_curvature_views_and_record_telemetry(
            lossy_cp=missing,
            iteration=10,
            group_size=8,
            names=["expert.weight"],
            views=[torch.ones(8)],
            successful_anchor_batches=1,
        )
