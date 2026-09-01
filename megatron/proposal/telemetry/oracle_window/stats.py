"""Pure NumPy statistics for the oracle saliency-window experiment."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


PERCENTILE_PROBABILITIES = np.asarray(
    (0.10, 0.25, 0.50, 0.75, 0.90, 0.99),
    dtype=np.float64,
)
PERCENTILE_LABELS = (
    "p10", "p25", "p50", "p75", "p90", "p99",
)


def compute_saliency(
    m: np.ndarray,
    e: np.ndarray,
    abs_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the signed Taylor saliency and its selection-time clipping.

    Analytics retain ``raw`` exactly.  Only cutoff/selection consumes
    ``clipped = max(0, raw)``.
    """

    m_arr, e_arr, h_arr = np.broadcast_arrays(
        np.asarray(m, dtype=np.float32),
        np.asarray(e, dtype=np.float32),
        np.asarray(abs_h, dtype=np.float32),
    )
    raw = -m_arr * e_arr + np.float32(0.5) * np.abs(h_arr) * e_arr * e_arr
    raw = raw.astype(np.float32, copy=False)
    return raw, np.maximum(raw, np.float32(0.0)).astype(np.float32, copy=False)


def deterministic_round_indices(
    population_size: int,
    sample_size: int,
    *,
    seed: int,
    round_index: int,
    stream: int = 0,
) -> np.ndarray:
    """Draw one reproducible, without-replacement sample for a CDF round."""

    population = int(population_size)
    count = int(sample_size)
    if population < 0 or count < 0 or count > population:
        raise ValueError("sample_size must satisfy 0 <= sample_size <= population_size")
    if int(round_index) < 0:
        raise ValueError("round_index must be non-negative")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if (int(round_index) + 1) * count > population:
        raise ValueError(
            "sample rounds must remain disjoint; requested rounds exceed population"
        )
    # Consecutive slices of one deterministic affine permutation make rounds
    # disjoint without materializing an O(population) permutation table.
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(stream), int(population), int(count)])
    )
    multiplier = int(rng.integers(1, min(population, 1_000_003) + 1))
    while math.gcd(multiplier, population) != 1:
        multiplier += 1
        if multiplier > 1_000_003:
            multiplier = 1
    shift = int(rng.integers(0, population))
    start = int(round_index) * count
    positions = start + np.arange(count, dtype=np.int64)
    picked = (positions * np.int64(multiplier) + np.int64(shift)) % np.int64(population)
    return np.sort(picked.astype(np.int64, copy=False))


def population_weights(
    domain_index: np.ndarray,
    round_index: np.ndarray,
    population_count: np.ndarray,
) -> np.ndarray:
    """Give every domain its population mass, shared equally across rounds.

    This prevents equal expert/non-expert sample counts from accidentally
    turning the mixed CDF into a 50:50 domain mixture.
    """

    domains = np.asarray(domain_index, dtype=np.int64).reshape(-1)
    rounds = np.asarray(round_index, dtype=np.int64).reshape(-1)
    populations = np.asarray(population_count, dtype=np.int64).reshape(-1)
    if domains.shape != rounds.shape:
        raise ValueError("domain_index and round_index must have the same shape")
    if np.any(populations < 0) or np.any(domains < 0) or np.any(rounds < 0):
        raise ValueError("population, domain, and round values must be non-negative")
    if domains.size and int(domains.max()) >= int(populations.size):
        raise ValueError("sample domain index is out of range")

    weights = np.empty(domains.size, dtype=np.float64)
    for domain in range(int(populations.size)):
        domain_mask = domains == domain
        sample_n = int(domain_mask.sum())
        population = int(populations[domain])
        if population == 0:
            if sample_n:
                raise ValueError("zero-population domain cannot contain samples")
            continue
        if sample_n == 0:
            raise ValueError("positive-population domain must contain samples")
        domain_rounds = np.unique(rounds[domain_mask])
        for round_id in domain_rounds:
            mask = domain_mask & (rounds == int(round_id))
            n_round = int(mask.sum())
            if n_round <= 0:
                raise AssertionError("unreachable empty sample round")
            weights[mask] = (
                float(population) / float(domain_rounds.size) / float(n_round)
            )
    return weights


def aggregate_sample_rounds(
    rounds: Sequence[Mapping[str, np.ndarray]],
    *,
    population_sizes: Mapping[str, int],
    domain_names: Sequence[str] = ("expert", "nonexpert"),
) -> dict[str, np.ndarray]:
    """Flatten deterministic per-domain CDF rounds with population weights."""

    names = tuple(str(name) for name in domain_names)
    if not rounds or not names or len(set(names)) != len(names):
        raise ValueError("sample rounds and unique domain names are required")
    if set(population_sizes) != set(names):
        raise ValueError("population_sizes must exactly match domain_names")

    values: list[np.ndarray] = []
    domains: list[np.ndarray] = []
    round_ids: list[np.ndarray] = []
    for ridx, sample_round in enumerate(rounds):
        if set(sample_round) != set(names):
            raise ValueError("every sample round must contain every domain")
        for didx, name in enumerate(names):
            part = np.asarray(sample_round[name], dtype=np.float32).reshape(-1)
            if part.size <= 0 or not np.all(np.isfinite(part)):
                raise ValueError("each domain round must contain finite samples")
            values.append(part)
            domains.append(np.full(part.size, didx, dtype=np.int16))
            round_ids.append(np.full(part.size, ridx, dtype=np.int16))
    domain_index = np.concatenate(domains)
    round_index = np.concatenate(round_ids)
    populations = np.asarray([population_sizes[name] for name in names], dtype=np.int64)
    return {
        "domain_names": np.asarray(names, dtype=np.str_),
        "population_count": populations,
        "values": np.concatenate(values),
        "domain_index": domain_index,
        "round_index": round_index,
        "weights": population_weights(domain_index, round_index, populations),
    }


def _validated_values_weights(
    values: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    weight = (
        np.ones(arr.size, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    if arr.shape != weight.shape or arr.size <= 0:
        raise ValueError("values and weights must be non-empty and aligned")
    valid = np.isfinite(arr) & np.isfinite(weight) & (weight > 0.0)
    if not bool(np.all(valid)):
        raise ValueError("values and weights must be finite and weights positive")
    return arr, weight


def weighted_percentiles(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    probabilities: np.ndarray = PERCENTILE_PROBABILITIES,
    min_effective_tail_count: float = 8.0,
) -> dict[str, np.ndarray | float | int]:
    """Compute weighted percentiles and disclose finite-tail resolution."""

    arr, weight = _validated_values_weights(values, weights)
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size <= 0 or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("percentile probabilities must be within [0, 1]")
    if float(min_effective_tail_count) < 0.0:
        raise ValueError("min_effective_tail_count must be non-negative")

    order = np.argsort(arr, kind="mergesort")
    sorted_values = arr[order]
    sorted_weights = weight[order]
    cumulative = np.cumsum(sorted_weights, dtype=np.float64)
    total_weight = float(cumulative[-1])
    positions = (cumulative - 0.5 * sorted_weights) / total_weight
    quantiles = np.interp(
        probs,
        positions,
        sorted_values,
        left=float(sorted_values[0]),
        right=float(sorted_values[-1]),
    )
    effective_n = total_weight * total_weight / float(np.dot(weight, weight))
    tail_count = effective_n * np.minimum(probs, 1.0 - probs)
    return {
        "probabilities": probs.copy(),
        "values": quantiles.astype(np.float64, copy=False),
        "effective_sample_size": float(effective_n),
        "effective_tail_count": tail_count.astype(np.float64, copy=False),
        "valid": (tail_count >= float(min_effective_tail_count)),
        "sample_count": int(arr.size),
        "weight_sum": float(total_weight),
    }


def summarize_by_domain(
    *,
    raw: np.ndarray,
    clipped: np.ndarray,
    weights: np.ndarray,
    domain_index: np.ndarray,
    n_domains: int,
    min_effective_tail_count: float = 8.0,
) -> dict[str, np.ndarray]:
    """Return aligned raw/clipped percentile arrays for each domain."""

    raw_arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    clipped_arr = np.asarray(clipped, dtype=np.float32).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    domain_arr = np.asarray(domain_index, dtype=np.int64).reshape(-1)
    if not (raw_arr.shape == clipped_arr.shape == weight_arr.shape == domain_arr.shape):
        raise ValueError("saliency samples, weights, and domains must align")
    expected = np.maximum(raw_arr, np.float32(0.0))
    if not np.allclose(clipped_arr, expected, rtol=1.0e-6, atol=0.0):
        raise ValueError("clipped saliency must equal max(0, raw saliency)")

    p = int(PERCENTILE_PROBABILITIES.size)
    raw_q = np.empty((int(n_domains), p), dtype=np.float64)
    clipped_q = np.empty_like(raw_q)
    effective_tail = np.empty_like(raw_q)
    valid = np.empty((int(n_domains), p), dtype=np.bool_)
    sample_count = np.zeros(int(n_domains), dtype=np.int64)
    effective_n = np.zeros(int(n_domains), dtype=np.float64)
    for domain in range(int(n_domains)):
        mask = domain_arr == domain
        if not bool(mask.any()):
            raise ValueError(f"domain {domain} has no samples")
        raw_stats = weighted_percentiles(
            raw_arr[mask],
            weight_arr[mask],
            min_effective_tail_count=min_effective_tail_count,
        )
        clipped_stats = weighted_percentiles(
            clipped_arr[mask],
            weight_arr[mask],
            min_effective_tail_count=min_effective_tail_count,
        )
        raw_q[domain] = raw_stats["values"]
        clipped_q[domain] = clipped_stats["values"]
        effective_tail[domain] = raw_stats["effective_tail_count"]
        valid[domain] = raw_stats["valid"]
        sample_count[domain] = int(raw_stats["sample_count"])
        effective_n[domain] = float(raw_stats["effective_sample_size"])
    return {
        "percentile_prob": PERCENTILE_PROBABILITIES.copy(),
        "raw_percentiles": raw_q,
        "clipped_percentiles": clipped_q,
        "effective_tail_count": effective_tail,
        "percentile_valid": valid,
        "sample_count": sample_count,
        "effective_sample_size": effective_n,
    }


def weighted_cutoff(
    clipped_scores: np.ndarray,
    weights: np.ndarray | None,
    *,
    topk_ratio: float,
) -> dict[str, float | int]:
    """Estimate a top-K threshold from population-weighted clipped scores."""

    score, weight = _validated_values_weights(clipped_scores, weights)
    if np.any(score < 0.0):
        raise ValueError("tau must be estimated from clipped, non-negative saliency")
    ratio = float(topk_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError("topk_ratio must satisfy 0 < ratio <= 1")
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_weight = weight[order]
    total = float(sorted_weight.sum())
    target = ratio * total
    idx = min(
        int(np.searchsorted(np.cumsum(sorted_weight), target, side="left")),
        int(sorted_score.size - 1),
    )
    tau = float(sorted_score[idx])
    greater = score > tau
    equal = score == tau
    greater_mass = float(weight[greater].sum())
    equal_mass = float(weight[equal].sum())
    tie_mass = max(0.0, min(equal_mass, target - greater_mass))
    tie_fraction = tie_mass / equal_mass if equal_mass > 0.0 else 0.0
    return {
        "tau": tau,
        "tie_fraction": float(tie_fraction),
        "population_weight": total,
        "target_weight": float(target),
        "greater_weight": greater_mass,
        "equal_weight": equal_mass,
        "estimated_selected_ratio": (
            (greater_mass + tie_fraction * equal_mass) / total
            if total > 0.0 else 0.0
        ),
        "sample_count": int(score.size),
    }


__all__ = [
    "PERCENTILE_LABELS",
    "PERCENTILE_PROBABILITIES",
    "aggregate_sample_rounds",
    "compute_saliency",
    "deterministic_round_indices",
    "population_weights",
    "summarize_by_domain",
    "weighted_cutoff",
    "weighted_percentiles",
]
