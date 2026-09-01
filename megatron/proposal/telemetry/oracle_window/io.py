"""Atomic rank shards and world merge for oracle-window telemetry."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .stats import (
    PERCENTILE_LABELS,
    PERCENTILE_PROBABILITIES,
    population_weights,
    summarize_by_domain,
    weighted_cutoff,
    weighted_percentiles,
)


# Version 3 fixes the reported grid to p10/p25/p50/p75/p90/p99.
SCHEMA_VERSION = 3
DOMAIN_NAMES = ("expert", "nonexpert")
RAW_FORMULA = "-m*e+0.5*abs_h*e^2"
CLIPPED_FORMULA = "max(0,raw_saliency)"

_REQUIRED_ARRAYS = (
    "domain_names",
    "population_count",
    "sample_domain",
    "sample_round",
    "sample_global_id",
    "sample_weight",
    "sample_raw",
    "sample_clipped",
    "percentile_prob",
    "raw_percentiles",
    "clipped_percentiles",
    "effective_tail_count",
    "percentile_valid",
    "sample_count",
    "effective_sample_size",
    "selected_count",
    "tau",
    "target_count",
)

_INVARIANTS = (
    "mode",
    "window_start",
    "window_length",
    "offset",
    "step",
    "world",
    "topk_ratio",
    "score_formula_raw",
    "score_formula_clipped",
    "selection_score",
    "min_effective_tail_count",
    "sampling_seed",
    "fd_epsilon",
    "fd_probes",
    "fd_data_batches",
    "fd_min_changed_fraction",
)


def build_rank_arrays(
    *,
    population_count: np.ndarray,
    sample_domain: np.ndarray,
    sample_round: np.ndarray,
    sample_global_id: np.ndarray,
    sample_raw: np.ndarray,
    sample_clipped: np.ndarray,
    sample_weight: np.ndarray | None = None,
    selected_count: np.ndarray,
    tau: float,
    target_count: int,
    min_effective_tail_count: float = 8.0,
) -> dict[str, np.ndarray]:
    """Build one rank payload from sampled raw/clipped saliency."""

    populations = np.asarray(population_count, dtype=np.int64).reshape(-1)
    domains = np.asarray(sample_domain, dtype=np.int16).reshape(-1)
    rounds = np.asarray(sample_round, dtype=np.int16).reshape(-1)
    global_ids = np.asarray(sample_global_id, dtype=np.int64).reshape(-1)
    raw = np.asarray(sample_raw, dtype=np.float32).reshape(-1)
    clipped = np.asarray(sample_clipped, dtype=np.float32).reshape(-1)
    if populations.shape != (len(DOMAIN_NAMES),):
        raise ValueError("population_count must contain expert and nonexpert counts")
    if not (domains.shape == rounds.shape == global_ids.shape == raw.shape == clipped.shape):
        raise ValueError("sample arrays must be aligned")
    weights = (
        population_weights(domains, rounds, populations)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    )
    if weights.shape != raw.shape or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("sample_weight must be finite, positive, and aligned")
    summary = summarize_by_domain(
        raw=raw,
        clipped=clipped,
        weights=weights,
        domain_index=domains,
        n_domains=len(DOMAIN_NAMES),
        min_effective_tail_count=float(min_effective_tail_count),
    )
    return {
        "domain_names": np.asarray(DOMAIN_NAMES, dtype=np.str_),
        "population_count": populations,
        "sample_domain": domains,
        "sample_round": rounds,
        "sample_global_id": global_ids,
        "sample_weight": weights,
        "sample_raw": raw,
        "sample_clipped": clipped,
        "percentile_prob": summary["percentile_prob"],
        "raw_percentiles": summary["raw_percentiles"],
        "clipped_percentiles": summary["clipped_percentiles"],
        "effective_tail_count": summary["effective_tail_count"],
        "percentile_valid": summary["percentile_valid"],
        "sample_count": summary["sample_count"],
        "effective_sample_size": summary["effective_sample_size"],
        "selected_count": np.asarray(selected_count, dtype=np.int64).reshape(-1),
        "tau": np.asarray([float(tau)], dtype=np.float64),
        "target_count": np.asarray([int(target_count)], dtype=np.int64),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_parent(path: Path) -> None:
    fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    min_effective_tail_count: float,
) -> None:
    missing = [name for name in _REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"oracle-window shard missing arrays: {missing}")
    names = tuple(str(value) for value in np.asarray(arrays["domain_names"]).reshape(-1))
    if names != DOMAIN_NAMES:
        raise ValueError("oracle-window domains must be expert, nonexpert")
    populations = np.asarray(arrays["population_count"], dtype=np.int64).reshape(-1)
    domains = np.asarray(arrays["sample_domain"], dtype=np.int64).reshape(-1)
    rounds = np.asarray(arrays["sample_round"], dtype=np.int64).reshape(-1)
    global_ids = np.asarray(arrays["sample_global_id"], dtype=np.int64).reshape(-1)
    weights = np.asarray(arrays["sample_weight"], dtype=np.float64).reshape(-1)
    raw = np.asarray(arrays["sample_raw"], dtype=np.float32).reshape(-1)
    clipped = np.asarray(arrays["sample_clipped"], dtype=np.float32).reshape(-1)
    if populations.shape != (2,) or np.any(populations < 0):
        raise ValueError("oracle-window population_count is invalid")
    if not (domains.shape == rounds.shape == global_ids.shape == weights.shape == raw.shape == clipped.shape):
        raise ValueError("oracle-window sample arrays are not aligned")
    if raw.size <= 0 or np.any(domains < 0) or np.any(domains >= 2) or np.any(rounds < 0):
        raise ValueError("oracle-window sample identity is invalid")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(clipped)):
        raise ValueError("oracle-window saliency must be finite")
    if not np.allclose(clipped, np.maximum(raw, 0.0), rtol=1.0e-6, atol=0.0):
        raise ValueError("oracle-window clipped saliency must equal max(0, raw)")
    # Fragment-stratified sampling may assign unequal weights inside one
    # domain/round. Tiny exhausted strata can also repeat an element across
    # rounds; the engine folds those weights onto one canonical observation.
    # The invariant is therefore exact total domain mass, not equal mass in
    # every surviving round.
    for domain in range(2):
        domain_rounds = np.unique(rounds[domains == domain])
        if domain_rounds.size <= 0:
            raise ValueError("positive-population domain has no sample rounds")
        actual_mass = float(weights[domains == domain].sum())
        if not np.isclose(
            actual_mass,
            float(populations[domain]),
            rtol=1.0e-10,
            atol=1.0e-8,
        ):
            raise ValueError(
                "oracle-window fragment weights do not preserve domain population"
            )
    # Repeated rounds must add information rather than resample the same
    # coordinate and overstate effective-tail resolution.
    keys = np.stack((domains, global_ids), axis=1)
    if np.unique(keys, axis=0).shape[0] != keys.shape[0]:
        raise ValueError("oracle-window sample identity is duplicated across rounds")

    probs = np.asarray(arrays["percentile_prob"], dtype=np.float64).reshape(-1)
    if not np.array_equal(probs, PERCENTILE_PROBABILITIES):
        raise ValueError("oracle-window percentile grid mismatch")
    expected = summarize_by_domain(
        raw=raw,
        clipped=clipped,
        weights=weights,
        domain_index=domains,
        n_domains=2,
        min_effective_tail_count=float(min_effective_tail_count),
    )
    for name in (
        "raw_percentiles",
        "clipped_percentiles",
        "effective_tail_count",
        "effective_sample_size",
    ):
        if not np.allclose(np.asarray(arrays[name]), expected[name], rtol=1.0e-10, atol=0.0):
            raise ValueError(f"oracle-window {name} does not match samples")
    for name in ("percentile_valid", "sample_count"):
        if not np.array_equal(np.asarray(arrays[name]), expected[name]):
            raise ValueError(f"oracle-window {name} does not match samples")
    selected = np.asarray(arrays["selected_count"], dtype=np.int64).reshape(-1)
    if selected.shape != (2,) or np.any(selected < 0) or np.any(selected > populations):
        raise ValueError("oracle-window selected_count is invalid")
    tau = np.asarray(arrays["tau"], dtype=np.float64).reshape(-1)
    target = np.asarray(arrays["target_count"], dtype=np.int64).reshape(-1)
    if tau.shape != (1,) or not np.isfinite(tau[0]) or tau[0] < 0.0:
        raise ValueError("oracle-window tau is invalid")
    if target.shape != (1,) or int(target[0]) < 0:
        raise ValueError("oracle-window target_count is invalid")


def write_oracle_window_shard(
    *,
    output_dir: Path,
    window_start: int,
    offset: int,
    step: int | None = None,
    rank: int,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Atomically write one rank's sampled telemetry and checksum manifest."""

    record = dict(manifest)
    world = int(record.get("world", 0))
    window_length = int(record.get("window_length", 100))
    actual_step = (
        int(window_start) + int(offset)
        if step is None
        else int(step)
    )
    if actual_step < int(window_start) + int(offset):
        raise ValueError(
            "oracle-window actual step cannot precede successful-update offset"
        )
    expected = {
        "mode": "oracle_saliency_window",
        "window_start": int(window_start),
        "window_length": window_length,
        "offset": int(offset),
        "step": actual_step,
        "rank": int(rank),
        "score_formula_raw": RAW_FORMULA,
        "score_formula_clipped": CLIPPED_FORMULA,
        "selection_score": "clipped",
    }
    for key, value in expected.items():
        if key in record and record[key] != value:
            raise ValueError(f"oracle-window manifest {key} conflicts with writer")
        record[key] = value
    if world <= 0 or not 0 <= int(rank) < world:
        raise ValueError("oracle-window rank/world is invalid")
    if not 0 < int(offset) <= window_length:
        raise ValueError("oracle-window offset is outside the window")
    if "topk_ratio" not in record or not (0.0 < float(record["topk_ratio"]) <= 1.0):
        raise ValueError("oracle-window topk_ratio is required")
    min_tail = float(record.get("min_effective_tail_count", 8.0))
    record["min_effective_tail_count"] = min_tail
    record.setdefault("sampling_seed", 0)
    record.setdefault("fd_epsilon", 1.0e-2)
    record.setdefault("fd_probes", 4)
    record.setdefault("fd_data_batches", 1)
    record.setdefault("fd_min_changed_fraction", 0.95)
    provenance = record.get("curvature_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(DOMAIN_NAMES):
        raise ValueError(
            "oracle-window shard requires expert/nonexpert curvature provenance"
        )
    for domain in DOMAIN_NAMES:
        metadata = provenance[domain]
        if not isinstance(metadata, Mapping):
            raise ValueError(f"oracle-window {domain} provenance must be a mapping")
        if not str(metadata.get("estimator", "")):
            raise ValueError(f"oracle-window {domain} estimator provenance is missing")
    expert_meta = provenance["expert"]
    nonexpert_meta = provenance["nonexpert"]
    record["expert_fd_signature"] = {
        key: expert_meta.get(key)
        for key in (
            "estimator", "parameter_dtype", "epsilon",
            "hutchinson_probes", "data_batches",
            "minimum_changed_fraction", "global_population",
            "n_quantiles", "n_histogram_bins", "n_lut_bins",
        )
    }
    record["nonexpert_fd_signature"] = {
        key: nonexpert_meta.get(key)
        for key in (
            "estimator", "parameter_dtype", "epsilon",
            "hutchinson_probes", "data_batches",
            "minimum_changed_fraction", "global_population",
            "measurement_owner_count",
            "n_quantiles", "n_histogram_bins", "n_lut_bins",
        )
    }
    record["percentile_labels"] = list(PERCENTILE_LABELS)
    record["percentile_probabilities"] = PERCENTILE_PROBABILITIES.tolist()
    _validate_arrays(arrays, min_effective_tail_count=min_tail)

    output = Path(output_dir)
    stem = (
        f"oracle_saliency_start{int(window_start):07d}_"
        f"offset{int(offset):03d}_rank{int(rank):03d}"
    )
    payload_path = output / f"{stem}.npz"
    manifest_path = output / f"{stem}.json"
    _atomic_npz(payload_path, arrays)
    record.update({
        "schema_version": SCHEMA_VERSION,
        "payload": payload_path.name,
        "payload_sha256": _sha256(payload_path),
    })
    _atomic_json(manifest_path, record)
    return payload_path, manifest_path


def load_oracle_window_shard(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported oracle-window schema")
    payload = path.parent / str(manifest.get("payload", ""))
    if _sha256(payload) != str(manifest.get("payload_sha256", "")):
        raise ValueError("oracle-window payload checksum mismatch")
    with np.load(payload, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    _validate_arrays(
        arrays,
        min_effective_tail_count=float(manifest["min_effective_tail_count"]),
    )
    return manifest, arrays


def merge_oracle_window_shards(
    manifest_paths: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate a complete world and retain both rank and global summaries."""

    loaded = [load_oracle_window_shard(Path(path)) for path in manifest_paths]
    if not loaded:
        raise ValueError("oracle-window merge requires rank shards")
    manifests = [item[0] for item in loaded]
    shards = [item[1] for item in loaded]
    world_values = {int(item.get("world", -1)) for item in manifests}
    if len(world_values) != 1:
        raise ValueError("oracle-window shard world mismatch")
    world = next(iter(world_values))
    rank_list = [int(item.get("rank", -1)) for item in manifests]
    if len(set(rank_list)) != len(rank_list):
        raise ValueError("duplicate oracle-window rank shard")
    if world <= 0 or set(rank_list) != set(range(world)) or len(rank_list) != world:
        raise ValueError("incomplete oracle-window world")
    order = np.argsort(np.asarray(rank_list, dtype=np.int64))
    manifests = [manifests[int(idx)] for idx in order]
    shards = [shards[int(idx)] for idx in order]
    first = manifests[0]
    for key in _INVARIANTS:
        present = [key in item for item in manifests]
        if not all(present) or any(item[key] != first[key] for item in manifests[1:]):
            raise ValueError(f"oracle-window shard configuration mismatch: {key}")
    for key in ("expert_fd_signature", "nonexpert_fd_signature"):
        if any(item.get(key) != first.get(key) for item in manifests[1:]):
            raise ValueError(
                f"oracle-window shard curvature provenance mismatch: {key}"
            )
    if any(item.get("percentile_labels") != list(PERCENTILE_LABELS) for item in manifests):
        raise ValueError("oracle-window percentile labels mismatch")

    tau_values = np.asarray([float(shard["tau"][0]) for shard in shards])
    targets = np.asarray([int(shard["target_count"][0]) for shard in shards])
    if not np.allclose(tau_values, tau_values[0], rtol=1.0e-12, atol=0.0):
        raise ValueError("oracle-window ranks did not use one global tau")
    if not np.all(targets == targets[0]):
        raise ValueError("oracle-window ranks disagree on target count")

    sample_rank = np.concatenate([
        np.full(shard["sample_raw"].size, rank, dtype=np.int32)
        for rank, shard in enumerate(shards)
    ])
    sample_domain = np.concatenate([shard["sample_domain"] for shard in shards])
    sample_round = np.concatenate([shard["sample_round"] for shard in shards])
    sample_global_id = np.concatenate([shard["sample_global_id"] for shard in shards])
    sample_weight = np.concatenate([shard["sample_weight"] for shard in shards])
    sample_raw = np.concatenate([shard["sample_raw"] for shard in shards])
    sample_clipped = np.concatenate([shard["sample_clipped"] for shard in shards])
    canonical = np.stack((sample_domain, sample_global_id), axis=1)
    if np.unique(canonical, axis=0).shape[0] != canonical.shape[0]:
        raise ValueError("oracle-window sampled canonical element is duplicated across ranks")

    min_tail = float(first["min_effective_tail_count"])
    global_domain = summarize_by_domain(
        raw=sample_raw,
        clipped=sample_clipped,
        weights=sample_weight,
        domain_index=sample_domain,
        n_domains=2,
        min_effective_tail_count=min_tail,
    )
    global_raw = weighted_percentiles(
        sample_raw,
        sample_weight,
        min_effective_tail_count=min_tail,
    )
    global_clipped = weighted_percentiles(
        sample_clipped,
        sample_weight,
        min_effective_tail_count=min_tail,
    )
    cutoff = weighted_cutoff(
        sample_clipped,
        sample_weight,
        topk_ratio=float(first["topk_ratio"]),
    )
    stored_tau = float(tau_values[0])
    estimated_tau = float(cutoff["tau"])
    if not np.isclose(estimated_tau, stored_tau, rtol=1.0e-6, atol=1.0e-12):
        raise ValueError(
            "oracle-window merged weighted cutoff disagrees with stored tau: "
            f"estimated={estimated_tau:.9g} stored={stored_tau:.9g}"
        )
    rank_population = np.stack([shard["population_count"] for shard in shards])
    rank_selected = np.stack([shard["selected_count"] for shard in shards])
    merged = {
        "domain_names": np.asarray(DOMAIN_NAMES, dtype=np.str_),
        "percentile_prob": PERCENTILE_PROBABILITIES.copy(),
        "ranks": np.arange(world, dtype=np.int32),
        "rank_population_count": rank_population,
        "rank_selected_count": rank_selected,
        "rank_sample_count": np.stack([shard["sample_count"] for shard in shards]),
        "rank_raw_percentiles": np.stack([shard["raw_percentiles"] for shard in shards]),
        "rank_clipped_percentiles": np.stack([shard["clipped_percentiles"] for shard in shards]),
        "rank_effective_tail_count": np.stack([shard["effective_tail_count"] for shard in shards]),
        "rank_percentile_valid": np.stack([shard["percentile_valid"] for shard in shards]),
        "sample_rank": sample_rank,
        "sample_domain": sample_domain,
        "sample_round": sample_round,
        "sample_global_id": sample_global_id,
        "sample_weight": sample_weight,
        "sample_raw": sample_raw,
        "sample_clipped": sample_clipped,
        "global_population_count": rank_population.sum(axis=0, dtype=np.int64),
        "global_selected_count": rank_selected.sum(axis=0, dtype=np.int64),
        "global_raw_percentiles": global_domain["raw_percentiles"],
        "global_clipped_percentiles": global_domain["clipped_percentiles"],
        "global_effective_tail_count": global_domain["effective_tail_count"],
        "global_percentile_valid": global_domain["percentile_valid"],
        "global_all_raw_percentiles": np.asarray(global_raw["values"]),
        "global_all_clipped_percentiles": np.asarray(global_clipped["values"]),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": first["mode"],
        "window_start": int(first["window_start"]),
        "offset": int(first["offset"]),
        "step": int(first["step"]),
        "world": world,
        "ranks": list(range(world)),
        "topk_ratio": float(first["topk_ratio"]),
        "tau": float(tau_values[0]),
        "target_count": int(targets[0]),
        "actual_selected_count": int(rank_selected.sum()),
        "estimated_tau": cutoff,
        "score_formula_raw": RAW_FORMULA,
        "score_formula_clipped": CLIPPED_FORMULA,
        "selection_score": "clipped",
        "percentile_labels": list(PERCENTILE_LABELS),
    }
    return summary, merged


__all__ = [
    "CLIPPED_FORMULA",
    "DOMAIN_NAMES",
    "RAW_FORMULA",
    "SCHEMA_VERSION",
    "build_rank_arrays",
    "load_oracle_window_shard",
    "merge_oracle_window_shards",
    "write_oracle_window_shard",
]
