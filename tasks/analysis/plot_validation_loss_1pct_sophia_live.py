#!/usr/bin/env python3
"""Render a read-only Sophia-style snapshot of the live 1% experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COMMON_SCRIPT = ROOT / "tasks/analysis/plot_validation_loss_4pct_live_extended.py"
OUTPUT = ROOT / "results/validation_loss_1pct_sophia_style_live.png"

CURRENT_DIR = (
    ROOT
    / "results/ds_pretrain_gpt_350M_MoE128_15Kiter_moc_1pct_pp2_ep4_back_locked/log"
)
PROPOSAL_1PCT_DIR = (
    ROOT / "results/ds_pretrain_gpt_350M_MoE128_10Kiter_proposal_1pct/log"
)


def load_common_module():
    spec = importlib.util.spec_from_file_location("plot4_common", COMMON_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helpers from {COMMON_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def only_log(log_dir: Path) -> Path:
    candidates = list(log_dir.glob("gpt-*.log"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one log under {log_dir}: {candidates}")
    return candidates[0]


def first_downward_crossing(
    steps: np.ndarray, losses: np.ndarray, target: float
) -> float:
    for right in range(1, len(steps)):
        left = right - 1
        if losses[right] <= target < losses[left]:
            x0, x1 = float(steps[left]), float(steps[right])
            y0, y1 = float(losses[left]), float(losses[right])
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    raise RuntimeError(f"curve never reaches target {target:.6f}")


def main() -> None:
    common = load_common_module()
    baseline_x, baseline_raw = common.read_curve(
        common.complete_10k_log(common.BASELINE_DIR)
    )
    proposal_x, proposal_raw = common.read_curve(
        common.complete_10k_log(PROPOSAL_1PCT_DIR)
    )
    moc_x, moc_raw = common.read_curve(only_log(CURRENT_DIR))

    baseline_y = common.moving_average(baseline_raw, window=5)
    proposal_y = common.moving_average(proposal_raw, window=5)
    moc_y = common.moving_average(moc_raw, window=5)

    # A centered five-point smoother needs two observations on each side.  Do
    # not use the live padded edge as the comparison endpoint.
    if moc_x.size < 7:
        raise RuntimeError("too few live validation points")
    endpoint = int(moc_x[-3])
    endpoint_index = int(np.flatnonzero(moc_x == endpoint)[0])
    target_loss = float(moc_y[endpoint_index])
    baseline_target = first_downward_crossing(baseline_x, baseline_y, target_loss)
    proposal_target = first_downward_crossing(proposal_x, proposal_y, target_loss)
    recovery_overhead = endpoint - proposal_target

    keep_baseline = baseline_x <= endpoint
    keep_proposal = proposal_x <= endpoint
    keep_moc = moc_x <= endpoint

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(11.88, 7.08), constrained_layout=True)
    ax.plot(
        baseline_x[keep_baseline], baseline_y[keep_baseline],
        color="#252525", lw=3.0, label="No fault",
    )
    ax.plot(
        proposal_x[keep_proposal], proposal_y[keep_proposal],
        color="#1555FF", lw=3.0, label="Proposal",
    )
    ax.plot(
        moc_x[keep_moc], moc_y[keep_moc],
        color="#2A9D38", lw=3.0, label="MoC",
    )

    lower = target_loss - 0.075
    arrow_y = target_loss - 0.018
    fault_steps = np.arange(100, endpoint + 1, 100, dtype=np.int64)
    ax.scatter(
        fault_steps,
        np.full(fault_steps.shape, lower + 0.006, dtype=np.float64),
        color="#B97878", marker="|", s=13, alpha=0.42,
        linewidths=0.7, zorder=3, label="Faults (every 100 steps)",
    )
    ax.annotate(
        "",
        xy=(proposal_target, arrow_y),
        xytext=(endpoint, arrow_y),
        arrowprops={
            "arrowstyle": "-|>", "color": "#858585", "lw": 4.2,
            "mutation_scale": 14, "shrinkA": 0, "shrinkB": 0,
        },
        zorder=6,
    )
    ax.vlines(
        [proposal_target, endpoint], arrow_y, target_loss,
        colors="#858585", linewidth=1.2, alpha=0.85, zorder=5,
    )
    ax.scatter(
        [baseline_target, proposal_target, endpoint],
        [target_loss, target_loss, target_loss],
        color=["#252525", "#1555FF", "#2A9D38"], s=34, zorder=7,
    )
    ax.text(
        (proposal_target + endpoint) / 2.0,
        arrow_y - 0.010,
        f"{recovery_overhead / 1000:.2f}K recovery overhead",
        ha="center", va="top", fontsize=14, color="#333333",
        bbox={"facecolor": "#EAEAF2", "edgecolor": "none", "alpha": 0.88, "pad": 0.15},
        zorder=8,
    )

    ax.set_title("(a) 1% Checkpoint Budget", fontsize=22, pad=10)
    ax.set_xlabel("Number of Training Steps", fontsize=17)
    ax.set_ylabel("Validation Loss", fontsize=17)
    ax.set_xlim(-100, endpoint + 100)
    ax.set_ylim(lower, 3.80)
    ticks = np.asarray([0, 2_500, 5_000, 7_500, 10_000], dtype=np.int64)
    ticks = ticks[ticks <= endpoint]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0K" if x == 0 else f"{x / 1000:g}K" for x in ticks])
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.52, linewidth=1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=15)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"live_validation_max={int(moc_x[-1])}")
    print(f"centered_endpoint={endpoint}")
    print(f"target_loss={target_loss:.6f}")
    print(f"no_fault_target={baseline_target:.3f}")
    print(f"proposal_target={proposal_target:.3f}")
    print(f"recovery_overhead={recovery_overhead:.3f}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
