from __future__ import annotations

import numpy as np
import pytest

from megatron.proposal.telemetry.oracle_window.stats import (
    PERCENTILE_LABELS,
    PERCENTILE_PROBABILITIES,
    aggregate_sample_rounds,
    compute_saliency,
    deterministic_round_indices,
    summarize_by_domain,
    weighted_cutoff,
    weighted_percentiles,
)


def test_saliency_preserves_negative_raw_and_clips_only_selection_score() -> None:
    raw, clipped = compute_saliency(
        m=np.asarray([2.0, -2.0]),
        e=np.asarray([1.0, 1.0]),
        abs_h=np.asarray([1.0, -1.0]),
    )
    np.testing.assert_allclose(raw, [-1.5, 2.5])
    np.testing.assert_allclose(clipped, [0.0, 2.5])


def test_percentile_grid_and_p99_tail_resolution_are_explicit() -> None:
    assert PERCENTILE_LABELS == (
        "p10", "p25", "p50", "p75", "p90", "p99",
    )
    np.testing.assert_array_equal(
        PERCENTILE_PROBABILITIES,
        [0.10, 0.25, 0.50, 0.75, 0.90, 0.99],
    )
    stats = weighted_percentiles(np.arange(10_000, dtype=np.float64))
    assert stats["effective_sample_size"] == pytest.approx(10_000.0)
    assert stats["effective_tail_count"][-1] == pytest.approx(100.0)
    assert bool(stats["valid"][-1]) is True
    assert bool(stats["valid"][5]) is True


def test_deterministic_rounds_repeat_but_use_different_round_samples() -> None:
    first = deterministic_round_indices(10_000, 512, seed=7, round_index=0, stream=1)
    replay = deterministic_round_indices(10_000, 512, seed=7, round_index=0, stream=1)
    second = deterministic_round_indices(10_000, 512, seed=7, round_index=1, stream=1)
    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, second)
    assert np.intersect1d(first, second).size == 0
    assert np.unique(first).size == first.size
    assert int(first.min()) >= 0 and int(first.max()) < 10_000


def test_multi_round_weights_preserve_domain_population_not_sample_ratio() -> None:
    aggregate = aggregate_sample_rounds(
        [
            {"expert": np.asarray([1.0, 2.0]), "nonexpert": np.asarray([10.0, 11.0])},
            {"expert": np.asarray([3.0, 4.0]), "nonexpert": np.asarray([12.0, 13.0])},
        ],
        population_sizes={"expert": 900, "nonexpert": 100},
    )
    weights = aggregate["weights"]
    domain = aggregate["domain_index"]
    assert float(weights[domain == 0].sum()) == pytest.approx(900.0)
    assert float(weights[domain == 1].sum()) == pytest.approx(100.0)
    for round_id in (0, 1):
        round_mask = aggregate["round_index"] == round_id
        assert float(weights[round_mask & (domain == 0)].sum()) == pytest.approx(450.0)
        assert float(weights[round_mask & (domain == 1)].sum()) == pytest.approx(50.0)


def test_tau_uses_clipped_scores_and_population_weights() -> None:
    clipped = np.asarray([0.0, 1.0, 2.0, 2.0], dtype=np.float32)
    weights = np.asarray([90.0, 5.0, 3.0, 2.0])
    cutoff = weighted_cutoff(clipped, weights, topk_ratio=0.04)
    assert cutoff["tau"] == pytest.approx(2.0)
    assert cutoff["tie_fraction"] == pytest.approx(0.8)
    assert cutoff["estimated_selected_ratio"] == pytest.approx(0.04)
    with pytest.raises(ValueError, match="clipped"):
        weighted_cutoff(np.asarray([-1.0, 2.0]), None, topk_ratio=0.5)


def test_domain_summary_records_raw_and_clipped_distributions() -> None:
    raw = np.asarray([-4.0, -1.0, 2.0, 3.0, -2.0, 1.0, 4.0, 8.0])
    clipped = np.maximum(raw, 0.0)
    domain = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    summary = summarize_by_domain(
        raw=raw,
        clipped=clipped,
        weights=np.ones(raw.size),
        domain_index=domain,
        n_domains=2,
        min_effective_tail_count=0.0,
    )
    assert summary["raw_percentiles"].shape == (2, 6)
    assert summary["clipped_percentiles"].shape == (2, 6)
    assert summary["raw_percentiles"][0, 0] < 0.0
    assert summary["clipped_percentiles"][0, 0] == 0.0
    assert bool(np.all(summary["percentile_valid"]))
