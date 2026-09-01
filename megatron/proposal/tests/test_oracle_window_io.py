from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from megatron.proposal.telemetry.oracle_window.io import (
    build_rank_arrays,
    load_oracle_window_shard,
    merge_oracle_window_shards,
    write_oracle_window_shard,
)


def _rank_arrays(rank: int) -> dict[str, np.ndarray]:
    domain = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int16)
    rounds = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16)
    ids = rank * 100 + np.asarray([0, 1, 50, 51, 2, 3, 52, 53], dtype=np.int64)
    raw = np.asarray([-2.0, 1.0, 4.0, 8.0, -1.0, 2.0, 5.0, 9.0]) + rank
    return build_rank_arrays(
        population_count=np.asarray([90, 10]),
        sample_domain=domain,
        sample_round=rounds,
        sample_global_id=ids,
        sample_raw=raw,
        sample_clipped=np.maximum(raw, 0.0),
        selected_count=np.asarray([3 + rank, 1]),
        tau=8.0,
        target_count=10,
        min_effective_tail_count=0.0,
    )


def _write(root: Path, rank: int, *, world: int = 2, arrays=None, **manifest):
    record = {
        "world": world,
        "topk_ratio": 0.04,
        "sampling_seed": 2024,
        "min_effective_tail_count": 0.0,
        "curvature_provenance": {
            "expert": {"estimator": "production_fd", "epsilon": 1.0e-3},
            "nonexpert": {"estimator": "dedicated_fd", "epsilon": 1.0e-2},
        },
        **manifest,
    }
    return write_oracle_window_shard(
        output_dir=root,
        window_start=1000,
        offset=10,
        rank=rank,
        arrays=_rank_arrays(rank) if arrays is None else arrays,
        manifest=record,
    )[1]


def test_rank_shard_roundtrip_keeps_raw_clipped_and_checksum(tmp_path: Path) -> None:
    path = _write(tmp_path, 0, world=1)
    manifest, arrays = load_oracle_window_shard(path)
    assert manifest["rank"] == 0
    assert manifest["step"] == 1010
    assert manifest["selection_score"] == "clipped"
    assert manifest["score_formula_raw"] == "-m*e+0.5*abs_h*e^2"
    assert manifest["curvature_provenance"]["expert"]["epsilon"] == 1.0e-3
    assert manifest["curvature_provenance"]["nonexpert"]["epsilon"] == 1.0e-2
    assert np.any(arrays["sample_raw"] < 0.0)
    assert np.all(arrays["sample_clipped"] >= 0.0)

    payload = tmp_path / manifest["payload"]
    with payload.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_oracle_window_shard(path)


def test_merge_retains_rank_arrays_and_recomputes_global_distributions(tmp_path: Path) -> None:
    paths = [_write(tmp_path, rank) for rank in range(2)]
    summary, merged = merge_oracle_window_shards(paths[::-1])
    assert summary["ranks"] == [0, 1]
    assert summary["actual_selected_count"] == 9
    assert merged["rank_raw_percentiles"].shape == (2, 2, 6)
    assert merged["rank_clipped_percentiles"].shape == (2, 2, 6)
    assert merged["rank_effective_tail_count"].shape == (2, 2, 6)
    assert merged["rank_percentile_valid"].shape == (2, 2, 6)
    np.testing.assert_array_equal(merged["rank_population_count"], [[90, 10], [90, 10]])
    np.testing.assert_array_equal(merged["global_population_count"], [180, 20])
    np.testing.assert_array_equal(np.unique(merged["sample_rank"]), [0, 1])
    assert merged["global_raw_percentiles"].shape == (2, 6)
    assert merged["global_all_clipped_percentiles"].shape == (6,)


def test_merge_rejects_incomplete_duplicate_or_semantically_different_world(tmp_path: Path) -> None:
    rank0 = _write(tmp_path, 0)
    with pytest.raises(ValueError, match="incomplete"):
        merge_oracle_window_shards([rank0])
    with pytest.raises(ValueError, match="duplicate"):
        merge_oracle_window_shards([rank0, rank0])

    rank1 = _write(tmp_path, 1, topk_ratio=0.01)
    with pytest.raises(ValueError, match="configuration mismatch: topk_ratio"):
        merge_oracle_window_shards([rank0, rank1])


def test_merge_rejects_cross_rank_replica_sample_identity(tmp_path: Path) -> None:
    rank0 = _write(tmp_path, 0)
    duplicate = _rank_arrays(1)
    duplicate["sample_global_id"] = _rank_arrays(0)["sample_global_id"].copy()
    rank1 = _write(tmp_path, 1, arrays=duplicate)
    with pytest.raises(ValueError, match="duplicated across ranks"):
        merge_oracle_window_shards([rank0, rank1])


def test_writer_rejects_inconsistent_clipped_or_rank_summary(tmp_path: Path) -> None:
    bad_clipped = _rank_arrays(0)
    bad_clipped["sample_clipped"] = bad_clipped["sample_clipped"].copy()
    bad_clipped["sample_clipped"][0] = 3.0
    with pytest.raises(ValueError, match="clipped saliency"):
        _write(tmp_path, 0, world=1, arrays=bad_clipped)

    bad_summary = _rank_arrays(0)
    bad_summary["raw_percentiles"] = bad_summary["raw_percentiles"].copy()
    bad_summary["raw_percentiles"][0, 0] += 1.0
    with pytest.raises(ValueError, match="raw_percentiles"):
        _write(tmp_path, 0, world=1, arrays=bad_summary)
