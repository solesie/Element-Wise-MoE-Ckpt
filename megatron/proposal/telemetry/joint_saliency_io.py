"""Serialization for rank-sharded oracle joint-saliency telemetry."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .joint_saliency_analysis import DEFAULT_BUDGETS, histogram_budget_summary


SCHEMA_VERSION = 1

_REQUIRED_MERGE_ARRAYS = (
    "expert_ids",
    "log10_edges",
    "hist_count",
    "hist_mass",
    "age_count",
    "force_count",
    "int_stats",
    "score_sum",
    "score_min_log10",
    "score_max_log10",
)

_PARAMETER_METADATA_ARRAYS = (
    "row_names",
    "row_numel",
    "parameter_names",
    "parameter_offsets",
    "parameter_total_numel",
    "parameter_ndim",
    "parameter_shapes",
)

# These values define what a score means.  Optional fields keep schema-v1
# shards readable, but presence must be consistent across a merged world.
_MANIFEST_INVARIANTS = (
    "mode",
    "state_phase",
    "checkpoint_phase",
    "checkpoint_visible_through",
    "selection_input_step",
    "score_source",
    "score_source_dtype",
    "source_dtype",
    "parameter_dtype",
    "master_dtype",
    "score_policy",
    "score_policy_id",
    "score_policy_name",
    "selection_mode",
    "score_formula",
    "is_actual_loss",
    "all_tracked_elements",
    "expert_id_space",
    "ep_size",
    "experts_per_layer",
    "n_bins",
    "age_force_period",
    "h_lut_built_at",
    "h_lut_epoch",
    "directional_hvp_reference",
    "hist_count_dtype",
    "hist_mass_accumulator_dtype",
    "score_sum_accumulator_dtype",
)


def _validate_expert_id_space(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Prove that new shards use canonical global EP expert ordinals."""

    if manifest.get("expert_id_space") != "global":
        return
    required = (
        "global_expert_base",
        "ep_rank",
        "ep_size",
        "experts_per_layer",
    )
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(
            f"joint saliency global expert metadata is incomplete: {missing}"
        )
    base = int(manifest["global_expert_base"])
    ep_rank = int(manifest["ep_rank"])
    ep_size = int(manifest["ep_size"])
    experts_per_layer = int(manifest["experts_per_layer"])
    if (
        base < 0
        or ep_size < 1
        or not 0 <= ep_rank < ep_size
        or experts_per_layer < 1
    ):
        raise ValueError("joint saliency global expert topology is invalid")

    global_ids = np.asarray(arrays["expert_ids"], dtype=np.int64).reshape(-1, 2)
    if np.any(global_ids[:, 1] < 0) or np.any(
        global_ids[:, 1] >= experts_per_layer
    ):
        raise ValueError("joint saliency global expert id is out of range")
    if "local_expert_ids" not in arrays:
        raise ValueError("joint saliency global shard is missing local_expert_ids")
    local_ids = np.asarray(
        arrays["local_expert_ids"], dtype=np.int64,
    ).reshape(-1, 2)
    if local_ids.shape != global_ids.shape:
        raise ValueError("joint saliency local/global expert id shape mismatch")
    if not np.array_equal(local_ids[:, 0], global_ids[:, 0]) or not np.array_equal(
        local_ids[:, 1] + base,
        global_ids[:, 1],
    ):
        raise ValueError("joint saliency local/global expert id mapping mismatch")


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(path.parent), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_parent(path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_parent(path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_joint_saliency_shard(
    *,
    output_dir: Path,
    step: int,
    rank: int,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Atomically write one compressed rank shard and its JSON manifest."""

    output_dir = Path(output_dir)
    stem = f"joint_saliency_step{int(step):07d}_rank{int(rank):03d}"
    payload_path = output_dir / f"{stem}.npz"
    manifest_path = output_dir / f"{stem}.json"
    record = dict(manifest)
    for key, expected in (("step", int(step)), ("rank", int(rank))):
        if key in record and int(record[key]) != expected:
            raise ValueError(
                f"joint saliency manifest {key}={record[key]!r} does not "
                f"match writer {key}={expected}"
            )
        record[key] = expected
    _atomic_npz(payload_path, arrays)
    record.update(
        {
            "schema_version": int(SCHEMA_VERSION),
            "payload": payload_path.name,
            "payload_sha256": sha256_file(payload_path),
        }
    )
    encoded = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    _atomic_bytes(manifest_path, encoded)
    return payload_path, manifest_path


def load_joint_saliency_shard(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load and checksum-verify one shard for tests/offline plotting."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported joint saliency schema")
    payload_path = manifest_path.parent / str(manifest["payload"])
    actual = sha256_file(payload_path)
    if actual != str(manifest.get("payload_sha256", "")):
        raise ValueError("joint saliency payload checksum mismatch")
    with np.load(payload_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return manifest, arrays


def _budget_ratios(manifest: Mapping[str, Any]) -> tuple[float, ...] | None:
    def _parse(raw: Any) -> tuple[float, ...] | None:
        if raw is None:
            return None
        if not isinstance(raw, (list, tuple)):
            raise ValueError("joint saliency budget ratios must be a sequence")
        ratios: list[float] = []
        for item in raw:
            value = item.get("budget") if isinstance(item, Mapping) else item
            ratio = float(value)
            if not (0.0 < ratio <= 1.0):
                raise ValueError(
                    "joint saliency budget ratios must satisfy 0 < ratio <= 1"
                )
            ratios.append(ratio)
        if not ratios or len(set(ratios)) != len(ratios):
            raise ValueError(
                "joint saliency budget ratios must be non-empty and unique"
            )
        return tuple(ratios)

    explicit = _parse(manifest.get("budget_ratios"))
    rows = _parse(manifest.get("budgets"))
    if explicit is not None and rows is not None and explicit != rows:
        raise ValueError(
            "joint saliency manifest budget_ratios/budgets mismatch"
        )
    return explicit if explicit is not None else rows


def _validate_merge_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[int, int, dict[str, str]]:
    missing = [name for name in _REQUIRED_MERGE_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"joint saliency shard missing arrays: {missing}")
    ids = np.asarray(arrays["expert_ids"])
    edges = np.asarray(arrays["log10_edges"])
    if ids.ndim != 2 or ids.shape[1] != 2:
        raise ValueError("joint saliency expert_ids must have shape [rows, 2]")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("joint saliency log10_edges must be a non-empty 1-D grid")
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
        raise ValueError("joint saliency log10_edges must be finite and increasing")
    rows = int(ids.shape[0])
    bins = int(edges.size - 1)
    expected_shapes = {
        "hist_count": (rows, bins),
        "hist_mass": (rows, bins),
        "age_count": (rows, 4),
        "force_count": (rows,),
        "int_stats": (rows, 6),
        "score_sum": (rows,),
        "score_min_log10": (rows,),
        "score_max_log10": (rows,),
    }
    for name, shape in expected_shapes.items():
        if np.asarray(arrays[name]).shape != shape:
            raise ValueError(
                f"joint saliency {name} shape mismatch: "
                f"{np.asarray(arrays[name]).shape} != {shape}"
            )
    for name in ("hist_count", "age_count", "force_count", "int_stats"):
        value = np.asarray(arrays[name])
        if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
            raise ValueError(
                f"joint saliency {name} must contain non-negative integers"
            )
    dtypes = {
        name: np.asarray(arrays[name]).dtype.str
        for name in _REQUIRED_MERGE_ARRAYS
    }
    return rows, bins, dtypes


def _validate_manifest_invariants(
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first = manifests[0]
    invariant_values: dict[str, Any] = {}
    for key in _MANIFEST_INVARIANTS:
        present = [key in manifest for manifest in manifests]
        if any(present) and not all(present):
            raise ValueError(
                f"joint saliency shard configuration mismatch: {key} presence"
            )
        if not all(present):
            continue
        expected = first[key]
        if any(manifest[key] != expected for manifest in manifests[1:]):
            raise ValueError(
                f"joint saliency shard configuration mismatch: {key}"
            )
        invariant_values[key] = expected

    ratios = [_budget_ratios(manifest) for manifest in manifests]
    present = [ratio is not None for ratio in ratios]
    if any(present) and not all(present):
        raise ValueError(
            "joint saliency shard configuration mismatch: budget ratio presence"
        )
    if all(present):
        expected_ratios = ratios[0]
        if any(ratio != expected_ratios for ratio in ratios[1:]):
            raise ValueError(
                "joint saliency shard configuration mismatch: budget ratios"
            )
        invariant_values["rank_budget_ratios"] = list(expected_ratios or ())
    return invariant_values


def _merge_parameter_fragments(
    shards: Sequence[Mapping[str, np.ndarray]],
    *,
    n_bins: int,
) -> dict[str, np.ndarray]:
    """Reassemble canonical parameters and prove exact fragment coverage."""

    presence = [
        tuple(name in shard for name in _PARAMETER_METADATA_ARRAYS)
        for shard in shards
    ]
    if not any(any(row) for row in presence):
        return {}
    if any(not all(row) for row in presence):
        raise ValueError(
            "joint saliency parameter fragment metadata is incomplete"
        )

    fragments: list[dict[str, Any]] = []
    parameter_meta: dict[tuple[int, int, str], tuple[int, tuple[int, ...]]] = {}
    parameter_intervals: dict[
        tuple[int, int, str], list[tuple[int, int, str]]
    ] = {}
    for shard_idx, shard in enumerate(shards):
        ids = np.asarray(shard["expert_ids"], dtype=np.int32).reshape(-1, 2)
        rows = int(ids.shape[0])
        names = np.asarray(shard["parameter_names"]).reshape(-1)
        fragment_names = np.asarray(shard["row_names"]).reshape(-1)
        offsets = np.asarray(shard["parameter_offsets"], dtype=np.int64).reshape(-1)
        totals = np.asarray(
            shard["parameter_total_numel"], dtype=np.int64,
        ).reshape(-1)
        row_numel = np.asarray(shard["row_numel"], dtype=np.int64).reshape(-1)
        ndim = np.asarray(shard["parameter_ndim"], dtype=np.int64).reshape(-1)
        shapes = np.asarray(shard["parameter_shapes"], dtype=np.int64)
        if any(
            value.shape != (rows,)
            for value in (
                names,
                fragment_names,
                offsets,
                totals,
                row_numel,
                ndim,
            )
        ) or shapes.ndim != 2 or shapes.shape[0] != rows:
            raise ValueError(
                "joint saliency parameter fragment metadata shape mismatch"
            )

        for row in range(rows):
            n_dim = int(ndim[row])
            if n_dim < 0 or n_dim > int(shapes.shape[1]):
                raise ValueError("joint saliency parameter_ndim is out of range")
            shape = tuple(int(x) for x in shapes[row, :n_dim])
            if any(dim < 0 for dim in shape):
                raise ValueError("joint saliency parameter shape is invalid")
            total = int(totals[row])
            if total <= 0 or (shape and math.prod(shape) != total):
                raise ValueError(
                    "joint saliency parameter shape/total_numel mismatch"
                )
            offset = int(offsets[row])
            length = int(row_numel[row])
            if offset < 0 or length <= 0 or offset + length > total:
                raise ValueError(
                    "joint saliency parameter fragment range is invalid"
                )
            row_stats = np.asarray(shard["int_stats"][row], dtype=np.int64)
            row_age = np.asarray(shard["age_count"][row], dtype=np.int64)
            row_hist_count = np.asarray(
                shard["hist_count"][row], dtype=np.int64,
            )
            if (
                int(row_stats[0]) != length
                or int(row_age.sum()) != length
                or int(row_stats[1]) != int(row_hist_count.sum())
                or int(row_stats[1] + row_stats[2] + row_stats[3]) != length
            ):
                raise ValueError(
                    "joint saliency parameter fragment element coverage mismatch"
                )
            key = (int(ids[row, 0]), int(ids[row, 1]), str(names[row]))
            metadata = (total, shape)
            if key in parameter_meta and parameter_meta[key] != metadata:
                raise ValueError(
                    f"joint saliency parameter metadata mismatch for {key}"
                )
            parameter_meta[key] = metadata
            fragment_id = str(fragment_names[row])
            parameter_intervals.setdefault(key, []).append(
                (offset, offset + length, fragment_id)
            )
            fragments.append(
                {
                    "key": key,
                    "shard": shard_idx,
                    "row": row,
                    "fragment_id": fragment_id,
                    "offset": offset,
                    "length": length,
                    "total": total,
                    "shape": shape,
                }
            )

    for key, intervals_raw in parameter_intervals.items():
        intervals = sorted(intervals_raw, key=lambda value: (value[0], value[1]))
        cursor = 0
        for start, end, fragment_id in intervals:
            if start != cursor:
                kind = "overlap" if start < cursor else "gap"
                raise ValueError(
                    f"joint saliency parameter fragment {kind} for {key}: "
                    f"cursor={cursor} interval=({start},{end}) "
                    f"fragment={fragment_id!r}"
                )
            cursor = end
        expected = int(parameter_meta[key][0])
        if cursor != expected:
            raise ValueError(
                f"joint saliency parameter fragment gap for {key}: "
                f"covered={cursor} total={expected}"
            )

    keys = sorted(parameter_meta)
    key_to_row = {key: row for row, key in enumerate(keys)}
    n_parameters = len(keys)
    out: dict[str, np.ndarray] = {
        "parameter_expert_ids": np.asarray(
            [(key[0], key[1]) for key in keys], dtype=np.int32,
        ),
        "parameter_names": np.asarray([key[2] for key in keys], dtype=np.str_),
        "parameter_total_numel": np.asarray(
            [parameter_meta[key][0] for key in keys], dtype=np.int64,
        ),
        "parameter_hist_count": np.zeros(
            (n_parameters, n_bins), dtype=np.int64,
        ),
        "parameter_hist_mass": np.zeros(
            (n_parameters, n_bins), dtype=np.float64,
        ),
        "parameter_age_count": np.zeros((n_parameters, 4), dtype=np.int64),
        "parameter_force_count": np.zeros(n_parameters, dtype=np.int64),
        "parameter_int_stats": np.zeros((n_parameters, 6), dtype=np.int64),
        "parameter_score_sum": np.zeros(n_parameters, dtype=np.float64),
        "parameter_score_min_log10": np.full(
            n_parameters, np.inf, dtype=np.float32,
        ),
        "parameter_score_max_log10": np.full(
            n_parameters, -np.inf, dtype=np.float32,
        ),
    }
    max_ndim = max((len(parameter_meta[key][1]) for key in keys), default=0)
    out["parameter_ndim"] = np.asarray(
        [len(parameter_meta[key][1]) for key in keys], dtype=np.int16,
    )
    out["parameter_shapes"] = np.full(
        (n_parameters, max_ndim), -1, dtype=np.int64,
    )
    for row, key in enumerate(keys):
        shape = parameter_meta[key][1]
        out["parameter_shapes"][row, :len(shape)] = shape

    # Keep fragment identifiers and placement for audit/debug only.  They are
    # deliberately not used as canonical parameter identities.
    fragments.sort(
        key=lambda item: (
            item["key"], item["offset"], item["shard"], item["row"],
        )
    )
    out["row_names"] = np.asarray(
        [item["fragment_id"] for item in fragments], dtype=np.str_,
    )
    out["fragment_expert_ids"] = np.asarray(
        [(item["key"][0], item["key"][1]) for item in fragments],
        dtype=np.int32,
    )
    out["fragment_parameter_names"] = np.asarray(
        [item["key"][2] for item in fragments], dtype=np.str_,
    )
    out["fragment_offsets"] = np.asarray(
        [item["offset"] for item in fragments], dtype=np.int64,
    )
    out["fragment_numel"] = np.asarray(
        [item["length"] for item in fragments], dtype=np.int64,
    )

    for item in fragments:
        target = key_to_row[item["key"]]
        shard = shards[int(item["shard"])]
        source = int(item["row"])
        out["parameter_hist_count"][target] += shard["hist_count"][source]
        out["parameter_hist_mass"][target] += shard["hist_mass"][source]
        out["parameter_age_count"][target] += shard["age_count"][source]
        out["parameter_force_count"][target] += shard["force_count"][source]
        out["parameter_int_stats"][target] += shard["int_stats"][source]
        out["parameter_score_sum"][target] += shard["score_sum"][source]
        out["parameter_score_min_log10"][target] = min(
            out["parameter_score_min_log10"][target],
            shard["score_min_log10"][source],
        )
        out["parameter_score_max_log10"][target] = max(
            out["parameter_score_max_log10"][target],
            shard["score_max_log10"][source],
        )
    return out


def merge_joint_saliency_shards(
    manifest_paths: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate a complete world and merge fragment rows by canonical expert."""

    loaded = [load_joint_saliency_shard(Path(path)) for path in manifest_paths]
    if not loaded:
        raise ValueError("joint saliency merge requires at least one shard")
    manifests = [item[0] for item in loaded]
    shards = [item[1] for item in loaded]
    worlds = {int(item.get("world", -1)) for item in manifests}
    steps = {int(item.get("step", -1)) for item in manifests}
    if len(worlds) != 1 or len(steps) != 1:
        raise ValueError("joint saliency shard world/step mismatch")
    world = next(iter(worlds))
    rank_list = [int(item.get("rank", -1)) for item in manifests]
    ranks = set(rank_list)
    if len(rank_list) != len(ranks):
        raise ValueError(
            f"duplicate joint saliency rank manifests: ranks={rank_list}"
        )
    if (
        world <= 0
        or len(manifests) != world
        or len(ranks) != world
        or ranks != set(range(world))
    ):
        raise ValueError(
            "incomplete joint saliency world: "
            f"shards={len(manifests)} ranks={sorted(ranks)} world={world}"
        )
    rank_order = sorted(range(len(manifests)), key=lambda idx: rank_list[idx])
    manifests = [manifests[idx] for idx in rank_order]
    shards = [shards[idx] for idx in rank_order]
    first_manifest = manifests[0]
    invariant_values = _validate_manifest_invariants(manifests)

    validated = []
    for manifest, shard in zip(manifests, shards):
        validated.append(_validate_merge_arrays(shard))
        _validate_expert_id_space(manifest, shard)
    dtype_signatures = [item[2] for item in validated]
    if any(signature != dtype_signatures[0] for signature in dtype_signatures[1:]):
        raise ValueError("joint saliency shard source dtype mismatch")
    first_edges = np.asarray(shards[0]["log10_edges"])
    if any(
        not np.array_equal(np.asarray(shard["log10_edges"]), first_edges)
        for shard in shards[1:]
    ):
        raise ValueError("joint saliency shard histogram edges mismatch")

    expert_keys = sorted(
        {
            (int(layer), int(expert))
            for shard in shards
            for layer, expert in np.asarray(shard["expert_ids"]).reshape(-1, 2)
        }
    )
    key_to_row = {key: idx for idx, key in enumerate(expert_keys)}
    n_experts = len(expert_keys)
    n_bins = int(first_edges.size - 1)
    merged = {
        "expert_ids": np.asarray(expert_keys, dtype=np.int32),
        "log10_edges": first_edges.copy(),
        "expert_hist_count": np.zeros((n_experts, n_bins), dtype=np.int64),
        "expert_hist_mass": np.zeros((n_experts, n_bins), dtype=np.float64),
        "expert_age_count": np.zeros((n_experts, 4), dtype=np.int64),
        "expert_force_count": np.zeros(n_experts, dtype=np.int64),
        "expert_int_stats": np.zeros((n_experts, 6), dtype=np.int64),
        "expert_score_sum": np.zeros(n_experts, dtype=np.float64),
        "expert_score_min_log10": np.full(n_experts, np.inf, dtype=np.float32),
        "expert_score_max_log10": np.full(n_experts, -np.inf, dtype=np.float32),
    }
    for shard in shards:
        ids = np.asarray(shard["expert_ids"], dtype=np.int32).reshape(-1, 2)
        for source_row, eid_raw in enumerate(ids):
            target = key_to_row[(int(eid_raw[0]), int(eid_raw[1]))]
            merged["expert_hist_count"][target] += shard["hist_count"][source_row]
            merged["expert_hist_mass"][target] += shard["hist_mass"][source_row]
            merged["expert_age_count"][target] += shard["age_count"][source_row]
            merged["expert_force_count"][target] += shard["force_count"][source_row]
            merged["expert_int_stats"][target] += shard["int_stats"][source_row]
            merged["expert_score_sum"][target] += shard["score_sum"][source_row]
            merged["expert_score_min_log10"][target] = min(
                merged["expert_score_min_log10"][target],
                shard["score_min_log10"][source_row],
            )
            merged["expert_score_max_log10"][target] = max(
                merged["expert_score_max_log10"][target],
                shard["score_max_log10"][source_row],
            )

    parameter_arrays = _merge_parameter_fragments(shards, n_bins=n_bins)
    merged.update(parameter_arrays)

    global_stats = merged["expert_int_stats"].sum(axis=0, dtype=np.int64)
    global_total = int(global_stats[0])
    global_positive = int(merged["expert_hist_count"].sum(dtype=np.int64))
    global_nonfinite = int(global_stats[3])
    global_clipped_low = int(global_stats[4])
    global_clipped_high = int(global_stats[5])
    global_budget_ratios = tuple(
        float(value)
        for value in invariant_values.get(
            "rank_budget_ratios",
            DEFAULT_BUDGETS,
        )
    )
    global_budgets = histogram_budget_summary(
        hist_count=merged["expert_hist_count"],
        hist_mass=merged["expert_hist_mass"],
        log_edges=merged["log10_edges"],
        total_count=global_total,
        budgets=global_budget_ratios,
        nonfinite_count=global_nonfinite,
        clipped_low_count=global_clipped_low,
        clipped_high_count=global_clipped_high,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "step": next(iter(steps)),
        "world": world,
        "ranks": sorted(ranks),
        "state_phase": first_manifest.get("state_phase"),
        "score_source": first_manifest.get("score_source"),
        "score_formula": first_manifest.get("score_formula"),
        "is_actual_loss": first_manifest.get("is_actual_loss"),
        "cutoff_method": "global_histogram_estimate",
        "global_budget_ratios": list(global_budget_ratios),
        "budgets": global_budgets,
        "global_total_elements": global_total,
        "global_positive_elements": global_positive,
        "global_zero_elements": int(global_stats[2]),
        "global_nonfinite_elements": global_nonfinite,
        "global_clipped_low": global_clipped_low,
        "global_clipped_high": global_clipped_high,
        "payload_dtypes": dtype_signatures[0],
        "global_parameter_count": int(
            np.asarray(parameter_arrays.get("parameter_names", ())).size
        ),
        "global_fragment_count": int(
            np.asarray(parameter_arrays.get("row_names", ())).size
        ),
        **invariant_values,
    }
    return summary, merged


__all__ = [
    "SCHEMA_VERSION",
    "load_joint_saliency_shard",
    "merge_joint_saliency_shards",
    "sha256_file",
    "write_joint_saliency_shard",
]
