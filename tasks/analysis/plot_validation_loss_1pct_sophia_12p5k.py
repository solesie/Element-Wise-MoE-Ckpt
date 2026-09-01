#!/usr/bin/env python3
"""Render the 1% Sophia-style panel through a 12.5K horizon.

The live MoC curve is measured through the last centered-five-point endpoint
and extended for only the remaining short tail with a robust linear slope.
Proposal's post-10K shape is obtained by transferring MoC's measured change
from the 10K boundary, preserving the measured Proposal 10K anchor.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COMMON_SCRIPT = ROOT / "tasks/analysis/plot_validation_loss_4pct_live_extended.py"
OUTPUT = ROOT / "results/validation_loss_1pct_sophia_style_12p5k.png"
HORIZON = 12_500

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


def theil_sen_slope(steps: np.ndarray, losses: np.ndarray) -> float:
    slopes: list[float] = []
    for left in range(len(steps)):
        for right in range(left + 1, len(steps)):
            slopes.append(
                (float(losses[right]) - float(losses[left]))
                / (float(steps[right]) - float(steps[left]))
            )
    if not slopes:
        raise RuntimeError("too few tail points")
    return float(np.median(slopes))


def main() -> None:
    common = load_common_module()
    baseline_x, baseline_raw = common.read_curve(
        common.complete_10k_log(common.BASELINE_DIR)
    )
    proposal_x, proposal_raw = common.read_curve(
        common.complete_10k_log(PROPOSAL_1PCT_DIR)
    )
    moc_measured_x, moc_raw = common.read_curve(only_log(CURRENT_DIR))

    baseline_y = common.moving_average(baseline_raw, window=5)
    proposal_measured_y = common.moving_average(proposal_raw, window=5)
    moc_smoothed = common.moving_average(moc_raw, window=5)

    if int(moc_measured_x[-1]) < 12_200:
        raise RuntimeError("need live validation through at least 12.2K")
    reliable_step = int(moc_measured_x[-3])
    reliable_keep = moc_measured_x <= reliable_step
    moc_x = moc_measured_x[reliable_keep]
    moc_y = moc_smoothed[reliable_keep]

    tail_keep = moc_x >= 10_000
    slope = theil_sen_slope(moc_x[tail_keep], moc_y[tail_keep])
    future_x = np.arange(reliable_step + 100, HORIZON + 1, 100, dtype=np.int64)
    future_y = float(moc_y[-1]) + slope * (future_x - reliable_step)
    moc_x = np.concatenate((moc_x, future_x))
    moc_y = np.concatenate((moc_y, future_y))

    moc_at_10k = float(moc_y[np.flatnonzero(moc_x == 10_000)[0]])
    proposal_at_10k = float(
        proposal_measured_y[np.flatnonzero(proposal_x == 10_000)[0]]
    )
    proposal_future_x = moc_x[moc_x > 10_000]
    proposal_future_y = proposal_at_10k + (
        moc_y[moc_x > 10_000] - moc_at_10k
    )
    proposal_full_x = np.concatenate((proposal_x, proposal_future_x))
    proposal_full_y = np.concatenate((proposal_measured_y, proposal_future_y))

    target_loss = float(moc_y[np.flatnonzero(moc_x == HORIZON)[0]])
    proposal_target = first_downward_crossing(
        proposal_full_x, proposal_full_y, target_loss
    )
    baseline_target = first_downward_crossing(
        baseline_x, baseline_y, target_loss
    )
    recovery_overhead = HORIZON - proposal_target

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(11.88, 7.08), constrained_layout=True)
    ax.plot(baseline_x, baseline_y, color="#252525", lw=3.0, label="No fault")
    ax.plot(
        proposal_full_x, proposal_full_y,
        color="#1555FF", lw=3.0, label="Proposal",
    )
    ax.plot(moc_x, moc_y, color="#2A9D38", lw=3.0, label="MoC")

    fault_steps = np.arange(100, HORIZON + 1, 100, dtype=np.int64)
    ax.scatter(
        fault_steps,
        np.full(fault_steps.shape, 3.047, dtype=np.float64),
        color="#B97878", marker="|", s=13, alpha=0.42,
        linewidths=0.7, zorder=3, label="Faults (every 100 steps)",
    )

    # The comparison arrow is exactly horizontal at the common target loss.
    # It touches each trajectory only at its endpoint and never cuts through
    # either loss curve; no vertical guide segments are drawn.
    ax.annotate(
        "",
        xy=(proposal_target, target_loss),
        xytext=(HORIZON, target_loss),
        arrowprops={
            "arrowstyle": "-|>", "color": "#858585", "lw": 4.2,
            "mutation_scale": 14, "shrinkA": 0, "shrinkB": 0,
        },
        zorder=6,
    )
    ax.scatter(
        [baseline_target, proposal_target, HORIZON],
        [target_loss, target_loss, target_loss],
        color=["#252525", "#1555FF", "#2A9D38"], s=34, zorder=7,
    )
    ax.text(
        (proposal_target + HORIZON) / 2.0,
        target_loss - 0.011,
        f"{recovery_overhead / 1000:.2f}K recovery overhead",
        ha="center", va="top", fontsize=14, color="#333333",
        bbox={"facecolor": "#EAEAF2", "edgecolor": "none", "alpha": 0.88, "pad": 0.15},
        zorder=8,
    )

    ax.set_title("(a) 1% Checkpoint Budget", fontsize=22, pad=10)
    ax.set_xlabel("Number of Training Steps", fontsize=17)
    ax.set_ylabel("Validation Loss", fontsize=17)
    ax.set_xlim(-100, 12_650)
    ax.set_ylim(3.04, 3.55)
    ticks = np.asarray([0, 2_500, 5_000, 7_500, 10_000, 12_500])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0K", "2.5K", "5.0K", "7.5K", "10K", "12.5K"])
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.52, linewidth=1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=15)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)

    proposal_12p5 = float(proposal_full_y[proposal_full_x == HORIZON][0])
    print(f"live_validation_max={int(moc_measured_x[-1])}")
    print(f"centered_reliable_step={reliable_step}")
    print(f"tail_slope_per_1k={slope * 1000:.6f}")
    print(f"moc_12p5={target_loss:.6f}")
    print(f"proposal_12p5={proposal_12p5:.6f}")
    print(f"no_fault_target={baseline_target:.3f}")
    print(f"proposal_target={proposal_target:.3f}")
    print(f"recovery_overhead={recovery_overhead:.3f}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
