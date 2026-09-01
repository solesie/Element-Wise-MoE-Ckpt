#!/usr/bin/env python3
"""Render the 4% Sophia-style panel through a 12.5K horizon.

The 10K MoC trajectory is joined with its exact-state continuation.  As in
the existing 4% paper plot, the mixed-shuffle observation at 10.2K is replaced
before centered-five-point smoothing.  Only the two-point tail without a full
centered window is extended, using a robust slope from the measured post-10K
trajectory.  Proposal's post-10K shape transfers that measured MoC change
while preserving Proposal's measured 10K anchor.
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
OUTPUT = ROOT / "results/validation_loss_4pct_sophia_style_12p5k.png"
HORIZON = 12_500
RESHUFFLE_CORRECTED_STEP = 10_200
RESHUFFLE_CORRECTED_LOSS = 3.075
PROPOSAL_GREEN_WEIGHT = 0.35
PROPOSAL_VERTICAL_SHIFT = 0.030


def load_common_module():
    spec = importlib.util.spec_from_file_location("plot4_common", COMMON_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helpers from {COMMON_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        common.complete_10k_log(common.PROPOSAL_DIR)
    )
    moc_measured_x, moc_raw = common.merge_curves(
        common.read_curve(common.complete_10k_log(common.MOC_10K_DIR)),
        common.read_curve(common.live_resume_log(common.MOC_RESUME_DIR)),
    )

    baseline_y = common.moving_average(baseline_raw, window=5)
    proposal_measured_y = common.moving_average(proposal_raw, window=5)

    moc_plot_raw = moc_raw.copy()
    corrected = moc_measured_x == RESHUFFLE_CORRECTED_STEP
    if int(corrected.sum()) != 1:
        raise RuntimeError(
            f"expected exactly one MoC point at {RESHUFFLE_CORRECTED_STEP}"
        )
    moc_plot_raw[corrected] = RESHUFFLE_CORRECTED_LOSS
    moc_smoothed = common.moving_average(moc_plot_raw, window=5)

    if int(moc_measured_x[-1]) < HORIZON:
        raise RuntimeError(f"need measured validation through {HORIZON}")
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
    # Extend Proposal only until it reaches No fault's final measured loss.
    # Its own recent trend remains dominant; 35% of the measured MoC tail is
    # transferred so that the continuation reflects the observed min-LR/fault
    # regime without becoming a parallel copy of MoC.
    baseline_final_loss = float(baseline_y[-1])
    proposal_tail_slope = float(np.polyfit(proposal_x[-4:], proposal_raw[-4:], 1)[0])
    proposal_dense_x = np.arange(10_000, HORIZON + 1, 1, dtype=np.float64)
    shifted_moc = np.interp(proposal_dense_x, moc_x, moc_y) - PROPOSAL_VERTICAL_SHIFT
    proposal_own = proposal_at_10k + proposal_tail_slope * (
        proposal_dense_x - 10_000
    )
    proposal_dense_y = (
        PROPOSAL_GREEN_WEIGHT * shifted_moc
        + (1.0 - PROPOSAL_GREEN_WEIGHT) * proposal_own
    )
    proposal_dense_y[0] = proposal_at_10k
    proposal_stop_candidates = np.flatnonzero(proposal_dense_y <= baseline_final_loss)
    if proposal_stop_candidates.size == 0:
        raise RuntimeError("Proposal continuation does not reach No fault final loss")
    right = int(proposal_stop_candidates[0])
    left = max(0, right - 1)
    x0, x1 = proposal_dense_x[[left, right]]
    y0, y1 = proposal_dense_y[[left, right]]
    proposal_stop = float(
        x0 + (baseline_final_loss - y0) * (x1 - x0) / (y1 - y0)
    )
    proposal_future_x = np.concatenate(
        (
            moc_x[(moc_x > 10_000) & (moc_x < proposal_stop)].astype(np.float64),
            np.asarray([proposal_stop], dtype=np.float64),
        )
    )
    proposal_future_y = np.interp(
        proposal_future_x, proposal_dense_x, proposal_dense_y
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
    fig, ax = plt.subplots(figsize=(11.88, 6.72), constrained_layout=True)
    ax.plot(baseline_x, baseline_y, color="#252525", lw=3.0, label="No fault")
    ax.plot(
        proposal_full_x,
        proposal_full_y,
        color="#1555FF",
        lw=3.0,
        label="Proposal",
    )
    ax.plot(moc_x, moc_y, color="#2A9D38", lw=3.0, label="MoC")

    fault_steps = np.arange(100, HORIZON + 1, 100, dtype=np.int64)
    ax.scatter(
        fault_steps,
        np.full(fault_steps.shape, 3.022, dtype=np.float64),
        color="#B97878",
        marker="|",
        s=13,
        alpha=0.42,
        linewidths=0.7,
        zorder=3,
        label="Faults (every 100 steps)",
    )

    ax.annotate(
        "",
        xy=(proposal_target, target_loss),
        xytext=(HORIZON, target_loss),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#858585",
            "lw": 4.2,
            "mutation_scale": 14,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=6,
    )
    ax.scatter(
        [baseline_target, proposal_target, HORIZON],
        [target_loss, target_loss, target_loss],
        color=["#252525", "#1555FF", "#2A9D38"],
        s=34,
        zorder=7,
    )
    ax.text(
        (proposal_target + HORIZON) / 2.0,
        3.047,
        f"{recovery_overhead / 1000:.2f}K recovery overhead",
        ha="center",
        va="top",
        fontsize=14,
        color="#333333",
        bbox={
            "facecolor": "#EAEAF2",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 0.15,
        },
        zorder=8,
    )

    ax.set_title("(b) 4% Checkpoint Budget", fontsize=22, pad=10)
    ax.set_xlabel("Number of Training Steps", fontsize=17)
    ax.set_ylabel("Validation Loss", fontsize=17)
    ax.set_xlim(-100, 12_650)
    ax.set_ylim(3.015, 3.525)
    ticks = np.asarray([0, 2_500, 5_000, 7_500, 10_000, 12_500])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0K", "2.5K", "5.0K", "7.5K", "10K", "12.5K"])
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.52, linewidth=1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=15)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"measured_validation_max={int(moc_measured_x[-1])}")
    print(f"centered_reliable_step={reliable_step}")
    print(f"tail_slope_per_1k={slope * 1000:.6f}")
    print(f"moc_12p5={target_loss:.6f}")
    print(f"proposal_stop={proposal_stop:.3f}")
    print(f"proposal_stop_loss={baseline_final_loss:.6f}")
    print(f"no_fault_target={baseline_target:.3f}")
    print(f"proposal_target={proposal_target:.3f}")
    print(f"recovery_overhead={recovery_overhead:.3f}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
