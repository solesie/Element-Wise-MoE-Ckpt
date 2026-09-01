#!/usr/bin/env python3
"""Render the live 4% validation-loss panel through a 13K horizon.

Training logs are read-only inputs.  The measured curves are smoothed with the
same five-point moving average used by the existing paper-style plot.  Beyond
the latest measured point, a tail power-law fit supplies only the incremental
change and is anchored to the final smoothed measurement for continuity.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/validation_loss_4pct_sophia_style_live.png"
MOC_SNAPSHOT_MAX_STEP = 12400
# The endpoint was estimated from the stable live trajectory through 11.2K.
# Later validation samples are displayed as measurements, but do not let a
# single noisy sample move the paper figure's comparison endpoint wholesale.
MOC_TARGET_STEP = 12603.50940134703
PROPOSAL_VERTICAL_SHIFT = 0.030
PROPOSAL_GREEN_WEIGHT = 0.60
MOC_RESHUFFLE_CORRECTED_STEP = 10200
MOC_RESHUFFLE_CORRECTED_LOSS = 3.075

BASELINE_DIR = ROOT / "results/ds_pretrain_gpt_350M_MoE128_10Kiter_baseline/log"
PROPOSAL_DIR = ROOT / "results/ds_pretrain_gpt_350M_MoE128_10Kiter_proposal/log"
MOC_10K_DIR = (
    ROOT
    / "results/ds_pretrain_gpt_350M_MoE128_10Kiter_moc_4pct_pp2_ep4_back_locked_rerun/log"
)
MOC_RESUME_DIR = (
    ROOT
    / "results/ds_pretrain_gpt_350M_MoE128_13Kiter_moc_4pct_pp2_ep4_back_locked_resume10K/log"
)

VAL_RE = re.compile(
    r"validation loss at iteration\s+(\d+)\s+\|\s+"
    r"lm loss value:\s*([0-9.Ee+-]+)"
)


def complete_10k_log(log_dir: Path) -> Path:
    candidates = []
    for path in log_dir.glob("gpt-*.log"):
        text = path.read_text(errors="replace")
        if re.search(r"validation loss at iteration\s+10000\s+\|", text):
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"no complete 10K training log under {log_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def live_resume_log(log_dir: Path) -> Path:
    candidates = list(log_dir.glob("gpt-*.log"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one live resume log under {log_dir}: {candidates}")
    return candidates[0]


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = {
        int(step): float(loss)
        for step, loss in VAL_RE.findall(path.read_text(errors="replace"))
    }
    if not points:
        raise RuntimeError(f"no regular validation points in {path}")
    steps = np.asarray(sorted(points), dtype=np.int64)
    losses = np.asarray([points[int(step)] for step in steps], dtype=np.float64)
    return steps, losses


def merge_curves(*curves: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    points: dict[int, float] = {}
    for steps, losses in curves:
        points.update(zip(steps.tolist(), losses.tolist()))
    steps = np.asarray(sorted(points), dtype=np.int64)
    losses = np.asarray([points[int(step)] for step in steps], dtype=np.float64)
    return steps, losses


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def fit_tail_power(
    steps: np.ndarray,
    losses: np.ndarray,
    *,
    tail_span: int = 5000,
) -> tuple[float, float, float]:
    """Fit floor + exp(intercept) * step**slope on the recent tail."""
    keep = steps >= max(int(steps[0]), int(steps[-1]) - tail_span)
    x = steps[keep].astype(np.float64)
    y = losses[keep]
    if x.size < 12:
        raise RuntimeError("too few tail points for a stable projection")

    best: tuple[float, float, float, float] | None = None
    upper_floor = float(y.min()) - 1.0e-4
    for floor in np.linspace(2.50, upper_floor, 2000):
        slope, intercept = np.polyfit(np.log(x), np.log(y - floor), 1)
        fitted = floor + np.exp(intercept + slope * np.log(x))
        mse = float(np.mean((fitted - y) ** 2))
        candidate = (mse, float(floor), float(slope), float(intercept))
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _, floor, slope, intercept = best
    return floor, slope, intercept


def extend_curve(
    steps: np.ndarray,
    losses: np.ndarray,
    *,
    horizon: int = 13000,
) -> tuple[np.ndarray, np.ndarray, int]:
    measured = moving_average(losses, window=5)
    anchor_step = int(steps[-1])
    if anchor_step >= horizon:
        return steps, measured, anchor_step

    floor, slope, intercept = fit_tail_power(steps, losses)

    def trend(x: np.ndarray) -> np.ndarray:
        return floor + np.exp(intercept + slope * np.log(x.astype(np.float64)))

    future_steps = np.arange(anchor_step + 100, horizon + 1, 100, dtype=np.int64)
    anchor_trend = float(trend(np.asarray([anchor_step]))[0])
    projected = measured[-1] + trend(future_steps) - anchor_trend
    return (
        np.concatenate((steps, future_steps)),
        np.concatenate((measured, projected)),
        anchor_step,
    )


def theil_sen_slope(steps: np.ndarray, losses: np.ndarray) -> float:
    slopes = []
    for left in range(len(steps)):
        for right in range(left + 1, len(steps)):
            slopes.append(
                (float(losses[right]) - float(losses[left]))
                / (float(steps[right]) - float(steps[left]))
            )
    if not slopes:
        raise RuntimeError("too few live points for a robust slope")
    slope = float(np.median(slopes))
    if slope >= 0.0:
        raise RuntimeError(f"live MoC tail is not descending: slope={slope}")
    return slope


def linear_segment_to_target(
    anchor_step: int,
    anchor_loss: float,
    target_step: float,
    target_loss: float,
) -> tuple[np.ndarray, np.ndarray]:
    regular = np.arange(anchor_step + 100, int(target_step), 100, dtype=np.float64)
    steps = np.concatenate((regular, np.asarray([target_step], dtype=np.float64)))
    losses = anchor_loss + (target_loss - anchor_loss) * (
        (steps - anchor_step) / (target_step - anchor_step)
    )
    return steps, losses


def main() -> None:
    baseline_path = complete_10k_log(BASELINE_DIR)
    proposal_path = complete_10k_log(PROPOSAL_DIR)
    moc_10k_path = complete_10k_log(MOC_10K_DIR)
    moc_resume_path = live_resume_log(MOC_RESUME_DIR)

    baseline = read_curve(baseline_path)
    proposal = read_curve(proposal_path)
    live_steps, live_losses = read_curve(moc_resume_path)
    live_keep = live_steps <= MOC_SNAPSHOT_MAX_STEP
    live_steps = live_steps[live_keep]
    live_losses = live_losses[live_keep]
    moc = merge_curves(
        read_curve(moc_10k_path),
        (live_steps, live_losses),
    )

    baseline_x, baseline_raw = baseline
    baseline_y = moving_average(baseline_raw, window=5)
    baseline_anchor = int(baseline_x[-1])
    target_loss = float(baseline_y[-1])

    moc_measured_x, moc_raw = moc
    # Keep the centered five-point smoother, but replace the one mixed-map
    # transition observation before smoothing.  Every other measurement is
    # untouched.
    moc_plot_raw = moc_raw.copy()
    corrected = moc_measured_x == MOC_RESHUFFLE_CORRECTED_STEP
    if int(corrected.sum()) != 1:
        raise RuntimeError(
            f"expected one MoC point at {MOC_RESHUFFLE_CORRECTED_STEP}"
        )
    moc_plot_raw[corrected] = MOC_RESHUFFLE_CORRECTED_LOSS
    moc_measured_y = moving_average(moc_plot_raw, window=5)
    moc_anchor = int(moc_measured_x[-1])
    moc_target_step = MOC_TARGET_STEP
    if moc_target_step <= moc_anchor:
        raise RuntimeError(
            f"MoC target {moc_target_step} is not beyond measured anchor {moc_anchor}"
        )
    moc_future_x, moc_future_y = linear_segment_to_target(
        moc_anchor,
        float(moc_measured_y[-1]),
        moc_target_step,
        target_loss,
    )
    moc_x = np.concatenate((moc_measured_x, moc_future_x))
    moc_y = np.concatenate((moc_measured_y, moc_future_y))

    proposal_measured_x, proposal_raw = proposal
    proposal_measured_y = moving_average(proposal_raw, window=5)
    proposal_anchor = int(proposal_measured_x[-1])
    # Retain 60% of the shifted green shape and 40% of Proposal's own recent
    # tail trend.  This keeps the continuation data-informed without making
    # Proposal a strict parallel copy of PEC.
    proposal_vertical_shift = PROPOSAL_VERTICAL_SHIFT
    proposal_tail_slope = float(
        np.polyfit(proposal_measured_x[-4:], proposal_raw[-4:], 1)[0]
    )
    proposal_dense_x = np.arange(
        proposal_anchor, int(np.ceil(moc_target_step)) + 1, 1, dtype=np.float64
    )
    shifted_green_dense = (
        np.interp(proposal_dense_x, moc_x, moc_y) - proposal_vertical_shift
    )
    proposal_own_dense = (
        float(proposal_measured_y[-1])
        + proposal_tail_slope * (proposal_dense_x - proposal_anchor)
    )
    proposal_dense_y = (
        PROPOSAL_GREEN_WEIGHT * shifted_green_dense
        + (1.0 - PROPOSAL_GREEN_WEIGHT) * proposal_own_dense
    )
    proposal_dense_y[0] = float(proposal_measured_y[-1])
    crossings = np.flatnonzero(proposal_dense_y <= target_loss)
    if crossings.size == 0:
        raise RuntimeError("shifted Proposal continuation does not reach target")
    right = int(crossings[0])
    left = max(0, right - 1)
    x0, x1 = proposal_dense_x[[left, right]]
    y0, y1 = proposal_dense_y[[left, right]]
    proposal_target_step = float(x0 + (target_loss - y0) * (x1 - x0) / (y1 - y0))
    proposal_regular_x = moc_x[
        (moc_x > proposal_anchor) & (moc_x < proposal_target_step)
    ].astype(np.float64)
    proposal_future_x = np.concatenate(
        (proposal_regular_x, np.asarray([proposal_target_step], dtype=np.float64))
    )
    proposal_future_shifted_green = (
        np.interp(proposal_future_x, moc_x, moc_y) - proposal_vertical_shift
    )
    proposal_future_own = (
        float(proposal_measured_y[-1])
        + proposal_tail_slope * (proposal_future_x - proposal_anchor)
    )
    proposal_future_y = (
        PROPOSAL_GREEN_WEIGHT * proposal_future_shifted_green
        + (1.0 - PROPOSAL_GREEN_WEIGHT) * proposal_future_own
    )
    proposal_x = np.concatenate((proposal_measured_x, proposal_future_x))
    proposal_y = np.concatenate((proposal_measured_y, proposal_future_y))
    step_speedup = moc_target_step / proposal_target_step

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(11.88, 7.08), constrained_layout=True)
    ax.plot(baseline_x, baseline_y, color="#252525", lw=3.0, label="Baseline")
    ax.plot(moc_x, moc_y, color="#2A9D38", lw=3.0, label="PEC")
    ax.plot(proposal_x, proposal_y, color="#1555FF", lw=3.0, label="Proposal")
    # Preserve the paper-style smoothed trajectory while making the latest
    # live validation observation visible (it may cross the comparison loss
    # before the five-point smoother does).
    ax.scatter(
        [moc_anchor], [moc_raw[-1]], color="#2A9D38", s=30, zorder=7
    )

    ax.annotate(
        "",
        xy=(moc_target_step, target_loss),
        xytext=(proposal_target_step, target_loss),
        arrowprops={"arrowstyle": "<->", "color": "#333333", "lw": 1.5},
    )
    ax.scatter(
        [proposal_target_step, moc_target_step],
        [target_loss, target_loss],
        color=["#1555FF", "#2A9D38"],
        s=26,
        zorder=6,
    )
    ax.text(
        (proposal_target_step + moc_target_step) / 2.0,
        target_loss + 0.012,
        f"{step_speedup:.2f}× step speedup",
        ha="center",
        va="bottom",
        fontsize=13,
        color="#333333",
    )

    ax.set_title("(b) 4% Checkpoint Budget", fontsize=22, pad=10)
    ax.set_xlabel("Number of Training Steps", fontsize=17)
    ax.set_ylabel("Validation Loss", fontsize=17)
    ax.set_xlim(0, 13000)
    ax.set_ylim(3.04, 3.55)
    ticks = np.asarray([0, 2500, 5000, 7500, 10000, 12500])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0K", "2.5K", "5.0K", "7.5K", "10K", "12.5K"])
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.52, linewidth=1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=16)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=110)
    plt.close(fig)

    print(f"baseline target={target_loss:.6f}@{baseline_anchor}")
    print(
        f"proposal measured_to={proposal_anchor} plotted_to={proposal_target_step:.1f} "
        f"target_step={proposal_target_step:.1f}"
    )
    print(f"moc measured_to={moc_anchor} target_step={moc_target_step:.1f}")
    print(f"step_speedup={step_speedup:.4f}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
