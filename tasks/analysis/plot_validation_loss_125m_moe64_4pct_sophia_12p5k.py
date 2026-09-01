#!/usr/bin/env python3
"""Plot the live 125M-MoE64 4% comparison in the Sophia paper style.

All three curves use a centered five-point moving average.  Once the MoC run
has reached 12.5K, its full measured trajectory is used.  Proposal's post-10K
continuation transfers the same measured MoC loss change from Proposal's
measured 10K anchor and stops at the measured No-fault 10K loss.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASELINE_LOG = ROOT / (
    "results/ds_pretrain_gpt_125M_MoE64_10Kiter_baseline/log/"
    "gpt-0.125B-lr-4.5e-4-minlr-4.5e-06-bs-256-gpus-8-mp-1-pp-2-"
    "ep-64-mlc-0.01-cap-1.0-drop-true_cloud-pomvGq_2026.07.31-05.52.25.log"
)
PROPOSAL_LOG = ROOT / (
    "results/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal/log/"
    "gpt-0.125B-lr-4.5e-4-minlr-4.5e-06-bs-256-gpus-8-mp-1-pp-2-"
    "ep-64-mlc-0.01-cap-1.0-drop-true_cloud-pomvGq_2026.07.31-01.07.36.log"
)
MOC_LOG = ROOT / (
    "results/ds_pretrain_gpt_125M_MoE64_12p5Kiter_moc_4pct_pp2_ep4_back_locked/"
    "log/gpt-0.125B-lr-4.5e-4-minlr-4.5e-06-bs-256-gpus-8-mp-1-pp-2-"
    "ep-64-mlc-0.01-cap-1.0-drop-true_cloud-pomvGq_2026.08.05-12.44.22.log"
)
OUTPUT = ROOT / "results/validation_loss_125m_moe64_4pct_sophia_12p5k.png"
HORIZON = 12_500
VALIDATION_RE = re.compile(
    r"validation loss at iteration\s+(\d+)\s*\|\s*"
    r"lm loss value:\s*([0-9.Ee+-]+)\s*\|"
)


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values: dict[int, float] = {}
    for step, loss in VALIDATION_RE.findall(path.read_text(errors="replace")):
        values[int(step)] = float(loss)
    if not values:
        raise RuntimeError(f"no regular validation records in {path}")
    steps = np.asarray(sorted(values), dtype=np.int64)
    losses = np.asarray([values[int(step)] for step in steps], dtype=np.float64)
    return steps, losses


def centered_moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window % 2 != 1:
        raise ValueError("centered moving-average window must be odd")
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def theil_sen_slope(steps: np.ndarray, losses: np.ndarray) -> float:
    slopes: list[float] = []
    for left in range(len(steps)):
        for right in range(left + 1, len(steps)):
            slopes.append(
                (float(losses[right]) - float(losses[left]))
                / (float(steps[right]) - float(steps[left]))
            )
    if not slopes:
        raise RuntimeError("too few post-10K points for a robust tail slope")
    return float(np.median(slopes))


def downward_crossing(
    steps: np.ndarray, losses: np.ndarray, target: float
) -> float | None:
    exact = np.flatnonzero(losses == target)
    if exact.size:
        return float(steps[int(exact[0])])
    for right in range(1, len(steps)):
        left = right - 1
        if losses[right] <= target < losses[left]:
            x0, x1 = float(steps[left]), float(steps[right])
            y0, y1 = float(losses[left]), float(losses[right])
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return None


def truncate_at_crossing(
    steps: np.ndarray, losses: np.ndarray, target: float
) -> tuple[np.ndarray, np.ndarray, float]:
    crossing = downward_crossing(steps, losses, target)
    if crossing is None:
        return steps, losses, float("nan")
    keep = steps < crossing
    return (
        np.concatenate((steps[keep].astype(np.float64), [crossing])),
        np.concatenate((losses[keep], [target])),
        crossing,
    )


def main() -> None:
    baseline_x, baseline_raw = read_curve(BASELINE_LOG)
    proposal_x, proposal_raw = read_curve(PROPOSAL_LOG)
    moc_measured_x, moc_raw = read_curve(MOC_LOG)

    baseline_y = centered_moving_average(baseline_raw)
    proposal_measured_y = centered_moving_average(proposal_raw)
    moc_smoothed = centered_moving_average(moc_raw)

    if int(baseline_x[-1]) != 10_000 or int(proposal_x[-1]) != 10_000:
        raise RuntimeError("No-fault and Proposal curves must end at 10K")
    if int(moc_measured_x[-1]) < 10_400:
        raise RuntimeError("too few measured post-10K MoC points")

    if int(moc_measured_x[-1]) >= HORIZON:
        measured = moc_measured_x <= HORIZON
        moc_full_x = moc_measured_x[measured]
        moc_full_y = moc_smoothed[measured]
        reliable_step = HORIZON
        slope = float("nan")
    else:
        # The final two centered values depend on duplicated edge padding.
        # Use only points with two measured neighbours on each side, then
        # robustly extend the still-running trajectory.
        reliable_step = int(moc_measured_x[-3])
        reliable = moc_measured_x <= reliable_step
        moc_reliable_x = moc_measured_x[reliable]
        moc_reliable_y = moc_smoothed[reliable]
        tail = moc_reliable_x >= 10_000
        slope = theil_sen_slope(moc_reliable_x[tail], moc_reliable_y[tail])
        future_x = np.arange(
            reliable_step + 100, HORIZON + 1, 100, dtype=np.int64
        )
        future_y = float(moc_reliable_y[-1]) + slope * (
            future_x - reliable_step
        )
        moc_full_x = np.concatenate((moc_reliable_x, future_x))
        moc_full_y = np.concatenate((moc_reliable_y, future_y))

    moc_at_10k = float(moc_full_y[np.flatnonzero(moc_full_x == 10_000)[0]])
    proposal_at_10k = float(proposal_measured_y[-1])
    proposal_future_x = moc_full_x[moc_full_x > 10_000]
    proposal_future_y = proposal_at_10k + (
        moc_full_y[moc_full_x > 10_000] - moc_at_10k
    )
    proposal_full_x = np.concatenate((proposal_x, proposal_future_x))
    proposal_full_y = np.concatenate((proposal_measured_y, proposal_future_y))

    baseline_target = float(baseline_y[-1])
    proposal_plot_x, proposal_plot_y, proposal_crossing = truncate_at_crossing(
        proposal_full_x, proposal_full_y, baseline_target
    )
    moc_baseline_crossing = downward_crossing(
        moc_full_x, moc_full_y, baseline_target
    )
    moc_plot_x, moc_plot_y = moc_full_x, moc_full_y

    # Sophia-style comparison at the loss MoC has at the end of 12.5K.
    # This keeps the MoC endpoint fully measured; only Proposal after 10K is
    # the requested difference-based continuation.
    comparison_target = float(moc_full_y[-1])
    proposal_target_crossing = downward_crossing(
        proposal_full_x, proposal_full_y, comparison_target
    )
    baseline_target_crossing = downward_crossing(
        baseline_x, baseline_y, comparison_target
    )

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(11.88, 6.72), constrained_layout=True)
    ax.plot(baseline_x, baseline_y, color="#252525", lw=3.0, label="No fault")
    ax.plot(proposal_plot_x, proposal_plot_y, color="#1555FF", lw=3.0, label="Proposal")
    ax.plot(moc_plot_x, moc_plot_y, color="#2A9D38", lw=3.0, label="MoC")

    fault_steps = np.arange(100, HORIZON + 1, 100, dtype=np.int64)
    ax.scatter(
        fault_steps,
        np.full(fault_steps.shape, 3.052, dtype=np.float64),
        color="#B97878",
        marker="|",
        s=13,
        alpha=0.42,
        linewidths=0.7,
        zorder=3,
        label="Faults (every 100 steps)",
    )

    if proposal_target_crossing is not None and baseline_target_crossing is not None:
        overhead = HORIZON - proposal_target_crossing
        ax.annotate(
            "",
            xy=(proposal_target_crossing, comparison_target),
            xytext=(HORIZON, comparison_target),
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
            [baseline_target_crossing, proposal_target_crossing, HORIZON],
            [comparison_target, comparison_target, comparison_target],
            color=["#252525", "#1555FF", "#2A9D38"],
            s=34,
            zorder=7,
        )
        ax.text(
            (proposal_target_crossing + HORIZON) / 2.0,
            comparison_target - 0.023,
            f"{overhead / 1000:.2f}K recovery overhead",
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
    ax.set_ylim(3.045, 3.525)
    ticks = np.asarray([0, 2_500, 5_000, 7_500, 10_000, 12_500])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0K", "2.5K", "5.0K", "7.5K", "10K", "12.5K"])
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.52, linewidth=1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=15)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)

    measured_moc_min_i = int(np.argmin(moc_smoothed))
    print(f"measured_validation_max={int(moc_measured_x[-1])}")
    print(f"centered_reliable_step={reliable_step}")
    print(
        f"tail_slope_per_1k={slope * 1000:.6f}"
        if np.isfinite(slope)
        else "tail_slope_per_1k=not_used_full_measurement"
    )
    print(f"no_fault_10k={baseline_target:.6f}")
    print(f"proposal_10k={proposal_at_10k:.6f}")
    print(f"proposal_10k_gap={proposal_at_10k - baseline_target:.6f}")
    print(
        "moc_measured_centered_min="
        f"{moc_smoothed[measured_moc_min_i]:.6f}@{int(moc_measured_x[measured_moc_min_i])}"
    )
    print(
        f"proposal_baseline_crossing={proposal_crossing:.3f}"
        if np.isfinite(proposal_crossing)
        else "proposal_baseline_crossing=not_reached"
    )
    print(
        f"moc_baseline_crossing={moc_baseline_crossing:.3f}"
        if moc_baseline_crossing is not None
        else "moc_baseline_crossing=not_reached"
    )
    print(f"comparison_target={comparison_target:.6f}")
    print(
        f"no_fault_comparison_crossing={baseline_target_crossing:.3f}"
        if baseline_target_crossing is not None
        else "no_fault_comparison_crossing=not_reached"
    )
    print(
        f"proposal_comparison_crossing={proposal_target_crossing:.3f}"
        if proposal_target_crossing is not None
        else "proposal_comparison_crossing=not_reached"
    )
    if proposal_target_crossing is not None:
        print(f"recovery_overhead={HORIZON - proposal_target_crossing:.3f}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
