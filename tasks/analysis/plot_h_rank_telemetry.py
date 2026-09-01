#!/usr/bin/env python3
"""Plot sampled |H_ii| progress and its rank agreement with Adam v_t."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_SNAPSHOT_RE = re.compile(r"h_rank_step(?P<step>\d+)_rank(?P<rank>\d+)\.npz$")


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = np.asarray(values)[order]
    starts = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.flatnonzero(sorted_values[1:] != sorted_values[:-1]).astype(np.int64) + 1,
    ))
    ends = np.concatenate((starts[1:], np.asarray([order.size], dtype=np.int64)))
    average = (starts.astype(np.float64) + ends.astype(np.float64) - 1.0) * 0.5
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.repeat(average, ends - starts)
    return ranks / float(max(1, order.size - 1))


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size <= 1 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _top_overlap(left: np.ndarray, right: np.ndarray, ratio: float = 0.01) -> float:
    k = max(1, int(np.ceil(float(ratio) * left.size)))
    left_top = set(np.argpartition(left, -k)[-k:].tolist())
    right_top = set(np.argpartition(right, -k)[-k:].tolist())
    return float(len(left_top & right_top)) / float(k)


def _checksums(input_dir: Path) -> dict[Path, str]:
    expected: dict[Path, str] = {}
    for log_path in sorted(input_dir.glob("h_rank_rank*.jsonl")):
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            snapshot = row.get("refresh_snapshot")
            digest = row.get("refresh_snapshot_sha256")
            if snapshot and digest:
                expected[Path(snapshot).resolve()] = str(digest)
    return expected


def load_snapshots(input_dir: Path):
    expected_digest = _checksums(input_dir)
    by_step: dict[int, dict[int, dict[str, np.ndarray]]] = defaultdict(dict)
    for path in sorted(input_dir.glob("h_rank_step*_rank*.npz")):
        match = _SNAPSHOT_RE.match(path.name)
        if match is None:
            continue
        step = int(match.group("step"))
        rank = int(match.group("rank"))
        resolved = path.resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if resolved not in expected_digest:
            raise RuntimeError(f"snapshot has no JSONL checksum record: {path}")
        if digest != expected_digest[resolved]:
            raise RuntimeError(f"snapshot checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            abs_h = np.asarray(payload["abs_h"], dtype=np.float64)
            v_t = np.asarray(payload["v_t"], dtype=np.float64)
            element_ids = np.asarray(payload["element_ids"], dtype=np.uint64)
            group_ids = np.asarray(payload["group_ids"], dtype=np.int64)
            group_size = int(np.asarray(payload["group_size"]).reshape(-1)[0])
        if not (
            abs_h.shape == v_t.shape == element_ids.shape == group_ids.shape
        ):
            raise RuntimeError(f"snapshot array shape mismatch: {path}")
        if group_size <= 0:
            raise RuntimeError(f"snapshot has invalid group_size={group_size}: {path}")
        if abs_h.size == 0 or not np.isfinite(abs_h).all() or not np.isfinite(v_t).all():
            raise RuntimeError(f"snapshot contains empty/non-finite samples: {path}")
        if np.any(abs_h < 0.0) or np.any(v_t < 0.0):
            raise RuntimeError(f"snapshot contains negative |H| or v_t: {path}")
        by_step[step][rank] = {
            "abs_h": abs_h,
            "v_t": v_t,
            "element_ids": element_ids,
            "group_ids": group_ids,
            "group_size": np.asarray([group_size], dtype=np.int64),
        }
    if not by_step:
        raise RuntimeError(f"no H-rank snapshots found under {input_dir}")
    ranks = sorted(next(iter(by_step.values())))
    for step, shards in by_step.items():
        if sorted(shards) != ranks:
            raise RuntimeError(f"rank set changed at step {step}: {sorted(shards)} != {ranks}")
    first_step = min(by_step)
    for step, shards in by_step.items():
        for rank in ranks:
            if not np.array_equal(
                shards[rank]["element_ids"],
                by_step[first_step][rank]["element_ids"],
            ):
                raise RuntimeError(
                    f"fixed sample identity changed: step={step} rank={rank}"
                )
    return dict(sorted(by_step.items())), ranks


def _pooled(by_rank: dict[int, dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.concatenate([by_rank[rank][key] for rank in sorted(by_rank)])


def _row_group_means(
    shard: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return AdaHessian-style contiguous-row group means for one rank."""

    group_size = int(shard["group_size"][0])
    group_ids = shard["group_ids"]
    order = np.argsort(group_ids, kind="mergesort")
    sorted_ids = group_ids[order]
    starts = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]).astype(np.int64) + 1,
    ))
    ends = np.concatenate((starts[1:], np.asarray([order.size], dtype=np.int64)))
    counts = ends - starts
    if np.any(counts != group_size):
        bad = counts[counts != group_size][:8].tolist()
        raise RuntimeError(
            "row-group sample is incomplete: "
            f"expected={group_size} observed_counts={bad}"
        )
    h_sorted = shard["abs_h"][order]
    v_sorted = shard["v_t"][order]
    return (
        np.add.reduceat(h_sorted, starts) / float(group_size),
        np.add.reduceat(v_sorted, starts) / float(group_size),
    )


def _pooled_row_group_means(
    by_rank: dict[int, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    h_parts = []
    v_parts = []
    for rank in sorted(by_rank):
        h_mean, v_mean = _row_group_means(by_rank[rank])
        h_parts.append(h_mean)
        v_parts.append(v_mean)
    return np.concatenate(h_parts), np.concatenate(v_parts)


def _pooled_element_v_and_repeated_row_h(
    by_rank: dict[int, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pair each element v_t with its AdaHessian-style row-8 mean |H|."""

    v_parts = []
    repeated_h_parts = []
    for rank in sorted(by_rank):
        shard = by_rank[rank]
        group_size = int(shard["group_size"][0])
        order = np.argsort(shard["group_ids"], kind="mergesort")
        row_h, _ = _row_group_means(shard)
        v_parts.append(shard["v_t"][order])
        repeated_h_parts.append(np.repeat(row_h, group_size))
    return np.concatenate(v_parts), np.concatenate(repeated_h_parts)


def plot(
    input_dir: Path,
    output: Path,
    *,
    steady_start: int | None = None,
) -> dict[str, object]:
    by_step, ranks = load_snapshots(input_dir)
    steps = list(by_step)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)

    observed_ratios: dict[int, float] = {}
    row8_by_step: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for step in steps:
        abs_h = _pooled(by_step[step], "abs_h")
        observed_ratios[step] = float(np.count_nonzero(abs_h > 0.0)) / float(abs_h.size)
        row8_by_step[step] = _pooled_row_group_means(by_step[step])

    steady_steps = [
        step for step in steps
        if steady_start is None or int(step) >= int(steady_start)
    ]
    if not steady_steps:
        raise RuntimeError(
            f"no snapshots at or after steady_start={steady_start}"
        )

    # (a) Three easy-to-read linear percentile curves.  Stop at p99 so a
    # handful of extreme values cannot flatten the other 99% of the sample.
    selected_steps = sorted({
        steady_steps[0],
        steady_steps[len(steady_steps) // 2],
        steady_steps[-1],
    })
    percentile = np.linspace(0.0, 99.0, 300)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(selected_steps)))
    for color, step in zip(colors, selected_steps):
        row8_h, _ = row8_by_step[step]
        positive_h = row8_h[row8_h > 0.0]
        if positive_h.size == 0:
            raise RuntimeError(f"step {step} has no observed positive row-8 |H| sample")
        axes[0].plot(
            percentile,
            np.percentile(positive_h, percentile) * 1.0e6,
            color=color,
            linewidth=2.0,
            label=f"iter {step}",
        )
    last = steps[-1]
    row8_h, row8_v = row8_by_step[last]
    observed_row8 = row8_h > 0.0
    observed_row8_h = row8_h[observed_row8]
    observed_row8_v = row8_v[observed_row8]
    axes[0].set_xlim(0.0, 99.0)
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_xlabel("Row-8 block percentile")
    axes[0].set_ylabel(r"Mean $|\hat H_{ii}|$  ($\times 10^{-6}$)")
    axes[0].set_title("(a) Curvature distribution")
    axes[0].text(
        0.04,
        0.95,
        f"iter {last}: {observed_row8.mean():.1%} nonzero blocks",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.82))

    # (b) Collapse the rank comparison to ten Adam-v deciles.  Each point is
    # the median curvature rank in that decile; whiskers are the middle 50%.
    abs_h = _pooled(by_step[last], "abs_h")
    v_t = _pooled(by_step[last], "v_t")
    observed = abs_h > 0.0
    observed_h = abs_h[observed]
    observed_v = v_t[observed]
    element_rho = _corr(_rank(observed_v), _rank(observed_h))
    element_tail_overlap = _top_overlap(observed_v, observed_h)
    h_rank = _rank(observed_row8_h)
    v_rank = _rank(observed_row8_v)
    edges = np.linspace(0.0, 1.0, 11)
    deciles = np.arange(1, 11, dtype=np.float64)
    median = np.empty(10, dtype=np.float64)
    low = np.empty(10, dtype=np.float64)
    high = np.empty(10, dtype=np.float64)
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (v_rank >= left) & (v_rank < right if idx < 9 else v_rank <= right)
        values = h_rank[mask]
        low[idx], median[idx], high[idx] = np.percentile(values, [25.0, 50.0, 75.0])
    rho = _corr(v_rank, h_rank)
    tail_overlap = _top_overlap(observed_row8_v, observed_row8_h)
    element_v, repeated_row8_h = _pooled_element_v_and_repeated_row_h(
        by_step[last]
    )
    observed_repeated_row8 = repeated_row8_h > 0.0
    element_v_vs_row8_h_rho = _corr(
        _rank(element_v[observed_repeated_row8]),
        _rank(repeated_row8_h[observed_repeated_row8]),
    )
    median_pct = median * 100.0
    axes[1].errorbar(
        deciles,
        median_pct,
        yerr=np.vstack(((median - low) * 100.0, (high - median) * 100.0)),
        color="#4C78A8",
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        capsize=2.5,
        label="Observed median (IQR)",
    )
    axes[1].plot(
        deciles,
        (deciles - 0.5) * 10.0,
        color="0.55",
        linestyle="--",
        linewidth=1.0,
        label="Perfect rank match",
    )
    axes[1].set_xlim(0.5, 10.5)
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_xticks(deciles)
    axes[1].set_xlabel(r"Adam $v_t$ rank decile")
    axes[1].set_ylabel(r"Median $|\hat H_{ii}|$ rank (%)")
    axes[1].set_title("(b) Adam–curvature agreement")
    axes[1].text(
        0.04, 0.96,
        (
            rf"$\rho(\overline{{v}}_8,\overline{{|H|}}_8)$ = {rho:.2f}"
            "\n"
            rf"$\rho(v_i,\overline{{|H|}}_8)$ = {element_v_vs_row8_h_rho:.2f}"
        ),
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")

    # (c) Replace the cold-start rank-agreement panel with directly readable
    # scale statistics.  All values are linear and use the same physical unit
    # as panel (a).
    quantile_labels = ("Median", "P90", "P99")
    quantile_probs = (50.0, 90.0, 99.0)
    quantiles_by_step: dict[int, list[float]] = {}
    for step in steady_steps:
        step_h = row8_by_step[step][0]
        step_h = step_h[step_h > 0.0]
        quantiles_by_step[step] = [
            float(value)
            for value in np.percentile(step_h, quantile_probs)
        ]
    quantile_matrix = np.asarray(
        [quantiles_by_step[step] for step in steady_steps],
        dtype=np.float64,
    ) * 1.0e6
    for idx, label in enumerate(quantile_labels):
        axes[2].plot(
            steady_steps,
            quantile_matrix[:, idx],
            marker="o",
            linewidth=2.0,
            label=label,
        )
    axes[2].set_ylim(bottom=0.0)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel(r"Mean $|\hat H_{ii}|$  ($\times 10^{-6}$)")
    axes[2].set_title("(c) Curvature scale over time")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    summary = {
        "steps": steps,
        "ranks": ranks,
        "sample_elements_last": int(abs_h.size),
        "observed_ratio_by_step": observed_ratios,
        "observed_elements_last": int(observed_h.size),
        "sample_row8_groups_last": int(row8_h.size),
        "observed_row8_groups_last": int(observed_row8_h.size),
        "steady_start": steady_start,
        "steady_steps": steady_steps,
        "distribution_steps": selected_steps,
        "last_abs_h_min": float(np.min(abs_h)),
        "last_observed_abs_h_min": float(np.min(observed_h)),
        "last_observed_abs_h_median": float(np.median(observed_h)),
        "last_observed_abs_h_p99": float(np.percentile(observed_h, 99.0)),
        "last_observed_abs_h_max": float(np.max(observed_h)),
        "last_observed_abs_h_vs_v_element_spearman": element_rho,
        "last_observed_abs_h_vs_v_element_top1_overlap": element_tail_overlap,
        "last_observed_row8_mean_abs_h_vs_row8_mean_v_spearman": rho,
        "last_observed_row8_mean_abs_h_vs_row8_mean_v_top1_overlap": tail_overlap,
        "last_observed_row8_mean_abs_h_repeated_vs_element_v_spearman": (
            element_v_vs_row8_h_rho
        ),
        "row8_abs_h_quantiles": {
            str(step): {
                label: quantiles_by_step[step][idx]
                for idx, label in enumerate(quantile_labels)
            }
            for step in steady_steps
        },
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steady-start", type=int)
    args = parser.parse_args()
    print(json.dumps(
        plot(args.input_dir, args.output, steady_start=args.steady_start),
        indent=2,
    ))


if __name__ == "__main__":
    main()
