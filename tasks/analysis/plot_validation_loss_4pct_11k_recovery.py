#!/usr/bin/env python3
"""Plot the measured 4% recovery comparison at the 11K MoC endpoint.

This analysis is intentionally read-only with respect to training.  It reads
completed log files, applies the paper figure's centered five-point smoothing,
and writes a plot plus a normalized recovery-overhead table.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COMMON_SCRIPT = ROOT / "tasks/analysis/plot_validation_loss_4pct_live_extended.py"
PLOT_OUTPUT = ROOT / "results/validation_loss_4pct_sophia_style_11k.png"
TABLE_OUTPUT = ROOT / "results/recovery_overhead_4pct_normalized_table.png"
METRICS_OUTPUT = ROOT / "results/recovery_overhead_4pct_normalized.tsv"

PROPOSAL_FAULTS = (
    ROOT
    / "results/ds_pretrain_gpt_350M_MoE128_10Kiter_proposal/log/proposal_faults.jsonl"
)
MOC_FAULTS_10K = (
    ROOT
    / "results/ds_pretrain_gpt_350M_MoE128_10Kiter_moc_4pct_pp2_ep4_back_locked_rerun/log/moc_faults.jsonl"
)
MOC_FAULTS_RESUME = (
    ROOT
    / "results/ds_pretrain_gpt_350M_MoE128_13Kiter_moc_4pct_pp2_ep4_back_locked_resume10K/log/moc_faults.jsonl"
)

MOC_ENDPOINT = 11_000
FULL_CHECKPOINT_OVERHEAD = 5_000.0
RESHUFFLE_CORRECTED_STEP = 10_200
RESHUFFLE_CORRECTED_LOSS = 3.075


def load_common_module():
    spec = importlib.util.spec_from_file_location("plot4_common", COMMON_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import plotting helpers from {COMMON_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_downward_crossing(
    steps: np.ndarray, losses: np.ndarray, target: float
) -> float:
    """Linearly interpolate the first descending crossing of target."""
    for right in range(1, len(steps)):
        left = right - 1
        if losses[right] <= target < losses[left]:
            x0, x1 = float(steps[left]), float(steps[right])
            y0, y1 = float(losses[left]), float(losses[right])
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    raise RuntimeError(f"curve never reaches target loss {target:.6f}")


def read_fault_steps(*paths: Path, maximum: int) -> np.ndarray:
    steps: set[int] = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            step = int(json.loads(line)["step"])
            if step <= maximum:
                steps.add(step)
    return np.asarray(sorted(steps), dtype=np.int64)


def save_metrics(rows: list[tuple[str, float, float]]) -> None:
    lines = ["method\trecovery_overhead_steps\tnormalized_vs_full_checkpoint"]
    lines.extend(f"{name}\t{steps:.3f}\t{normalized:.6f}" for name, steps, normalized in rows)
    METRICS_OUTPUT.write_text("\n".join(lines) + "\n")


def render_table(rows: list[tuple[str, float, float]]) -> None:
    fig, ax = plt.subplots(figsize=(8.1, 2.35), constrained_layout=True)
    ax.axis("off")
    cell_text = [
        [name, f"{steps / 1000:.2f}K", f"{normalized:.2f}×"]
        for name, steps, normalized in rows
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Recovery policy", "Recovery overhead", "Normalized"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.48, 0.29, 0.23],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12.5)
    table.scale(1.0, 1.65)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        if row == 0:
            cell.set_facecolor("#E7EBF2")
            cell.set_text_props(weight="bold")
        elif row % 2:
            cell.set_facecolor("#F7F8FA")
    ax.set_title(
        "Normalized Recovery Overhead (5K Full-Checkpoint Reference)",
        fontsize=14,
        pad=3,
    )
    fig.savefig(TABLE_OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    common = load_common_module()

    baseline_x, baseline_raw = common.read_curve(
        common.complete_10k_log(common.BASELINE_DIR)
    )
    proposal_x, proposal_raw = common.read_curve(
        common.complete_10k_log(common.PROPOSAL_DIR)
    )
    moc_x, moc_raw = common.merge_curves(
        common.read_curve(common.complete_10k_log(common.MOC_10K_DIR)),
        common.read_curve(common.live_resume_log(common.MOC_RESUME_DIR)),
    )

    # Preserve the correction and centered smoother used in the existing live
    # paper plot.  No other measurement is modified.
    moc_plot_raw = moc_raw.copy()
    corrected = moc_x == RESHUFFLE_CORRECTED_STEP
    if int(corrected.sum()) != 1:
        raise RuntimeError(f"expected one MoC point at {RESHUFFLE_CORRECTED_STEP}")
    moc_plot_raw[corrected] = RESHUFFLE_CORRECTED_LOSS

    baseline_y = common.moving_average(baseline_raw, window=5)
    proposal_y = common.moving_average(proposal_raw, window=5)
    moc_y = common.moving_average(moc_plot_raw, window=5)

    endpoint_index = np.flatnonzero(moc_x == MOC_ENDPOINT)
    if endpoint_index.size != 1:
        raise RuntimeError(f"expected one MoC point at {MOC_ENDPOINT}")
    target_loss = float(moc_y[int(endpoint_index[0])])
    baseline_target_step = first_downward_crossing(
        baseline_x, baseline_y, target_loss
    )
    proposal_target_step = first_downward_crossing(
        proposal_x, proposal_y, target_loss
    )

    moc_overhead = MOC_ENDPOINT - baseline_target_step
    proposal_overhead = proposal_target_step - baseline_target_step
    rows = [
        ("Full checkpoint (5K rollback)", FULL_CHECKPOINT_OVERHEAD, 1.0),
        (
            "MoC (4%)",
            moc_overhead,
            moc_overhead / FULL_CHECKPOINT_OVERHEAD,
        ),
        (
            "Proposal (4%)",
            proposal_overhead,
            proposal_overhead / FULL_CHECKPOINT_OVERHEAD,
        ),
    ]

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(11.88, 7.08), constrained_layout=True)
    ax.plot(baseline_x, baseline_y, color="#252525", lw=3.0, label="No fault")
    ax.plot(proposal_x, proposal_y, color="#1555FF", lw=3.0, label="Proposal")
    moc_keep = moc_x <= MOC_ENDPOINT
    ax.plot(moc_x[moc_keep], moc_y[moc_keep], color="#2A9D38", lw=3.0, label="MoC")

    proposal_fault_steps = read_fault_steps(PROPOSAL_FAULTS, maximum=10_000)
    moc_fault_steps = read_fault_steps(
        MOC_FAULTS_10K, MOC_FAULTS_RESUME, maximum=MOC_ENDPOINT
    )
    fault_steps = np.asarray(
        sorted(set(proposal_fault_steps.tolist()) | set(moc_fault_steps.tolist())),
        dtype=np.int64,
    )
    # A muted event rug keeps the 100-step cadence visible without painting
    # red dots over the trajectories.
    ax.scatter(
        fault_steps,
        np.full(fault_steps.shape, 3.047, dtype=np.float64),
        color="#B97878",
        marker="|",
        s=13,
        alpha=0.42,
        linewidths=0.7,
        zorder=3,
        label="Faults (every 100 steps)",
    )

    # Sophia-style horizontal comparison: Proposal reaches the same target
    # loss much earlier than MoC, while remaining close to the no-fault curve.
    arrow_y = target_loss - 0.0065
    ax.annotate(
        "",
        xy=(proposal_target_step, arrow_y),
        xytext=(MOC_ENDPOINT, arrow_y),
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
    ax.vlines(
        [proposal_target_step, MOC_ENDPOINT],
        arrow_y,
        target_loss,
        colors="#858585",
        linewidth=1.2,
        alpha=0.85,
        zorder=5,
    )
    ax.scatter(
        [baseline_target_step, proposal_target_step, MOC_ENDPOINT],
        [target_loss, target_loss, target_loss],
        color=["#252525", "#1555FF", "#2A9D38"],
        s=34,
        zorder=7,
    )
    ax.text(
        (proposal_target_step + MOC_ENDPOINT) / 2.0,
        arrow_y - 0.005,
        f"{(MOC_ENDPOINT - proposal_target_step) / 1000:.2f}K recovery overhead",
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
    ax.set_xlim(0, 11_250)
    ax.set_ylim(3.04, 3.55)
    ticks = np.asarray([0, 2_500, 5_000, 7_500, 10_000, 11_000])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0K", "2.5K", "5.0K", "7.5K", "10K", "11K"])
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.52, linewidth=1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=15)

    PLOT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    render_table(rows)
    save_metrics(rows)

    print(f"target_loss={target_loss:.6f}")
    print(f"baseline_target_step={baseline_target_step:.3f}")
    print(f"proposal_target_step={proposal_target_step:.3f}")
    print(f"moc_endpoint={MOC_ENDPOINT}")
    print(f"moc_overhead={moc_overhead:.3f}")
    print(f"proposal_overhead={proposal_overhead:.3f}")
    print(f"plot={PLOT_OUTPUT}")
    print(f"table={TABLE_OUTPUT}")
    print(f"metrics={METRICS_OUTPUT}")


if __name__ == "__main__":
    main()
