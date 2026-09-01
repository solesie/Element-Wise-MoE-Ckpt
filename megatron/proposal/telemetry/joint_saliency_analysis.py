"""NumPy-only summaries for compressed joint-saliency telemetry."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


DEFAULT_BUDGETS: Tuple[float, ...] = (0.01, 0.04, 0.10, 0.20)


def histogram_budget_summary(
    *,
    hist_count: np.ndarray,
    hist_mass: np.ndarray,
    log_edges: np.ndarray,
    total_count: int | None = None,
    budgets: Iterable[float] = DEFAULT_BUDGETS,
    nonfinite_count: int = 0,
    clipped_low_count: int = 0,
    clipped_high_count: int = 0,
) -> list[dict[str, float | int | str | None]]:
    """Estimate all-element top-budget cutoffs from a log-score histogram.

    Positive-score counts and masses are folded into bins on GPU.  The cutoff
    within a boundary bin is approximate.  Zero scores are outside the log
    histogram but remain part of the requested-budget denominator.
    """

    counts = np.asarray(hist_count, dtype=np.int64)
    mass = np.asarray(hist_mass, dtype=np.float64)
    if counts.ndim == 2:
        counts = counts.sum(axis=0, dtype=np.int64)
    if mass.ndim == 2:
        mass = mass.sum(axis=0, dtype=np.float64)
    edges = np.asarray(log_edges, dtype=np.float64)
    if counts.ndim != 1 or mass.shape != counts.shape:
        raise ValueError("joint saliency histograms must have aligned 1-D bins")
    if edges.shape != (counts.size + 1,):
        raise ValueError("joint saliency log edges do not match histogram bins")

    positive_n = int(counts.sum())
    denominator_n = positive_n if total_count is None else int(total_count)
    if denominator_n < positive_n:
        raise ValueError(
            "joint saliency total_count cannot be smaller than positive count"
        )
    if np.any(counts < 0):
        raise ValueError("joint saliency histogram counts must be non-negative")
    nonfinite_n = int(nonfinite_count)
    clipped_low_n = int(clipped_low_count)
    clipped_high_n = int(clipped_high_count)
    if min(nonfinite_n, clipped_low_n, clipped_high_n) < 0:
        raise ValueError("joint saliency diagnostic counts must be non-negative")
    if denominator_n < positive_n + nonfinite_n:
        raise ValueError(
            "joint saliency total_count cannot be smaller than the represented "
            "positive and non-finite populations"
        )

    # Histogram mass is accumulated independently from the integer counters.
    # Preserve a usable count summary if FP accumulation itself overflowed, but
    # never present the resulting mass ratio as valid.
    mass_valid = bool(np.all(np.isfinite(mass)) and np.all(mass >= 0.0))
    safe_mass = np.where(np.isfinite(mass) & (mass >= 0.0), mass, 0.0)
    total_mass = float(safe_mass.sum())
    descending_count = np.cumsum(counts[::-1], dtype=np.int64)
    descending_mass = np.cumsum(safe_mass[::-1], dtype=np.float64)
    out: list[dict[str, float | int | str | None]] = []
    for ratio_raw in budgets:
        ratio = float(ratio_raw)
        if not (0.0 < ratio <= 1.0):
            raise ValueError("joint saliency budgets must satisfy 0 < ratio <= 1")
        target = max(1, int(math.ceil(ratio * max(1, denominator_n))))
        if target > positive_n:
            # A non-negative score threshold at zero includes the whole tied
            # zero population.  Expose that tie rather than silently changing
            # the denominator to positive scores only.
            selected_count = int(denominator_n)
            out.append(
                {
                    "budget": ratio,
                    "target_count": int(target),
                    "denominator_count": int(denominator_n),
                    "positive_count": int(positive_n),
                    "cutoff_log10_lower": None,
                    "cutoff_log10_upper": None,
                    "selected_count_including_boundary_bin": selected_count,
                    "selected_ratio_including_boundary_bin": (
                        float(selected_count) / float(denominator_n)
                        if denominator_n else 0.0
                    ),
                    "captured_hist_mass_ratio": (
                        1.0 if total_mass > 0.0 and mass_valid else None
                    ),
                    "cutoff_method": "zero_score_tie",
                    "budget_valid": bool(nonfinite_n == 0 and mass_valid),
                    "budget_validity": (
                        (
                            "valid_zero_score_tie_with_clipped_tail"
                            if clipped_low_n > 0 or clipped_high_n > 0
                            else "valid_zero_score_tie"
                        )
                        if nonfinite_n == 0 and mass_valid
                        else "invalid_nonfinite_or_hist_mass"
                    ),
                    "cutoff_edge_semantics": "zero_score_population_tie",
                    "cutoff_bounds_complete": True,
                    "nonfinite_count": nonfinite_n,
                    "clipped_low_count": clipped_low_n,
                    "clipped_high_count": clipped_high_n,
                }
            )
            continue

        rev_idx = int(np.searchsorted(descending_count, target, side="left"))
        rev_idx = min(max(rev_idx, 0), max(0, counts.size - 1))
        bin_idx = counts.size - 1 - rev_idx
        selected_count = int(descending_count[rev_idx]) if positive_n else 0
        selected_mass = float(descending_mass[rev_idx]) if positive_n else 0.0
        lower_unbounded = bool(bin_idx == 0 and clipped_low_n > 0)
        upper_unbounded = bool(
            bin_idx == counts.size - 1 and clipped_high_n > 0
        )
        if lower_unbounded and upper_unbounded:
            edge_semantics = "both_bounds_open_due_clipped_tails"
        elif lower_unbounded:
            edge_semantics = "lower_bound_open_due_clipped_low_tail"
        elif upper_unbounded:
            edge_semantics = "upper_bound_open_due_clipped_high_tail"
        elif clipped_low_n > 0 or clipped_high_n > 0:
            edge_semantics = "finite_boundary_bin_with_clipped_tail_elsewhere"
        else:
            edge_semantics = "finite_histogram_bin_bounds"
        budget_valid = bool(nonfinite_n == 0 and mass_valid)
        out.append(
            {
                "budget": ratio,
                "target_count": int(target),
                "denominator_count": int(denominator_n),
                "positive_count": int(positive_n),
                "cutoff_log10_lower": (
                    None if lower_unbounded else float(edges[bin_idx])
                ),
                "cutoff_log10_upper": (
                    None if upper_unbounded else float(edges[bin_idx + 1])
                ),
                "selected_count_including_boundary_bin": selected_count,
                "selected_ratio_including_boundary_bin": (
                    float(selected_count) / float(denominator_n)
                    if denominator_n else 0.0
                ),
                "captured_hist_mass_ratio": (
                    selected_mass / total_mass
                    if total_mass > 0.0 and mass_valid else None
                ),
                "cutoff_method": "histogram_estimate",
                "budget_valid": budget_valid,
                "budget_validity": (
                    (
                        "valid_histogram_estimate_with_clipped_tail"
                        if clipped_low_n > 0 or clipped_high_n > 0
                        else "valid_histogram_boundary_estimate"
                    )
                    if budget_valid
                    else "invalid_nonfinite_or_hist_mass"
                ),
                "cutoff_edge_semantics": edge_semantics,
                "cutoff_bounds_complete": bool(
                    not lower_unbounded and not upper_unbounded
                ),
                "nonfinite_count": nonfinite_n,
                "clipped_low_count": clipped_low_n,
                "clipped_high_count": clipped_high_n,
            }
        )
    return out


__all__ = ["DEFAULT_BUDGETS", "histogram_budget_summary"]
