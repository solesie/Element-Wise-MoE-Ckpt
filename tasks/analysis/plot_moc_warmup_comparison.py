#!/usr/bin/env python3
"""Compare the 1% MoC validation curves with and without LR warmup."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VAL_RE = re.compile(
    r"validation loss at iteration\s+(\d+).*?lm loss value:\s*([0-9.Ee+-]+)"
)


def find_training_log(result_dir: Path) -> Path:
    logs = sorted(
        path
        for path in (result_dir / "log").glob("*.log")
        if not path.name.startswith("accuracy")
    )
    if len(logs) != 1:
        raise RuntimeError(f"expected one training log in {result_dir / 'log'}, got {logs}")
    return logs[0]


def read_validation_curve(result_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    points: dict[int, float] = {}
    for match in VAL_RE.finditer(find_training_log(result_dir).read_text(errors="replace")):
        points[int(match.group(1))] = float(match.group(2))
    if not points:
        raise RuntimeError(f"no validation points found in {result_dir}")
    iterations = np.asarray(sorted(points), dtype=np.int64)
    losses = np.asarray([points[int(step)] for step in iterations], dtype=np.float64)
    return iterations, losses


def read_fault_bmax(result_dir: Path) -> np.ndarray:
    logs = sorted((result_dir / "log").glob("*faults.jsonl"))
    if len(logs) != 1:
        raise RuntimeError(f"expected one fault log in {result_dir / 'log'}, got {logs}")
    records = [json.loads(line) for line in logs[0].read_text().splitlines() if line]
    return np.asarray([record["B_max_lost_iter"] for record in records], dtype=np.float64)


def first_reached_at(
    iterations: np.ndarray,
    losses: np.ndarray,
    target_loss: float,
    not_before: int = 0,
) -> int | None:
    reached = np.flatnonzero((iterations >= not_before) & (losses <= target_loss))
    return int(iterations[reached[0]]) if reached.size else None


def fit_gap_decay_models(
    iterations: np.ndarray,
    gaps: np.ndarray,
) -> list[tuple[str, float, float, float]]:
    """Fit a small trend ensemble; tuples are (domain, floor, slope, intercept)."""
    models: list[tuple[str, float, float, float]] = []
    for start in (1000, 1500, 2000, 2500, 3000):
        keep = iterations >= start
        x = iterations[keep].astype(np.float64)
        y = gaps[keep].astype(np.float64)
        if x.size < 8 or np.any(y <= 0.0):
            continue

        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
        models.append(("power", 0.0, float(slope), float(intercept)))

        # A non-zero residual gap is plausible, so include power-law and
        # exponential decays with a fitted floor. The ensemble spread is model
        # sensitivity, not a statistical confidence interval.
        for domain in ("power", "exp"):
            best: tuple[float, float, float, float] | None = None
            for floor in np.linspace(0.0, float(y.min()) * 0.95, 200):
                transformed_x = np.log(x) if domain == "power" else x
                candidate_slope, candidate_intercept = np.polyfit(
                    transformed_x, np.log(y - floor), 1
                )
                fitted = floor + np.exp(
                    candidate_intercept + candidate_slope * transformed_x
                )
                mse = float(np.mean((fitted - y) ** 2))
                candidate = (
                    mse,
                    float(floor),
                    float(candidate_slope),
                    float(candidate_intercept),
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
            assert best is not None
            models.append((domain, best[1], best[2], best[3]))
    if not models:
        raise RuntimeError("not enough positive common-interval gaps for projection")
    return models


def predict_anchored_gap(
    model: tuple[str, float, float, float],
    iterations: np.ndarray,
    anchor_step: int,
    anchor_gap: float,
) -> np.ndarray:
    domain, floor, slope, intercept = model

    def raw(x: np.ndarray) -> np.ndarray:
        transformed = np.log(x) if domain == "power" else x
        return floor + np.exp(intercept + slope * transformed)

    values = raw(iterations.astype(np.float64))
    anchor_value = float(raw(np.asarray([anchor_step], dtype=np.float64))[0])
    scale = (anchor_gap - floor) / (anchor_value - floor)
    return floor + (values - floor) * scale


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def first_downward_crossing(
    iterations: np.ndarray,
    losses: np.ndarray,
    target: float,
) -> float:
    hit = np.flatnonzero(losses <= target)
    if not hit.size:
        raise RuntimeError(f"curve never reaches target loss {target}")
    index = int(hit[0])
    if index == 0:
        return float(iterations[0])
    x0, x1 = float(iterations[index - 1]), float(iterations[index])
    y0, y1 = float(losses[index - 1]), float(losses[index])
    if y0 == y1:
        return x1
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def draw_paper_panel(
    ax: plt.Axes,
    iterations: np.ndarray,
    baseline: np.ndarray,
    moc: np.ndarray,
    proposal: np.ndarray,
    title: str,
    target_loss: float = 3.20,
    show_ylabel: bool = True,
) -> tuple[float, float, float]:
    baseline_color = "#222222"
    moc_color = "#299438"
    proposal_color = "#245BFF"
    ax.plot(iterations, baseline, color=baseline_color, lw=1.8, label="Baseline")
    ax.plot(iterations, moc, color=moc_color, lw=1.8, label="MoC")
    ax.plot(iterations, proposal, color=proposal_color, lw=1.8, label="Proposal")

    moc_step = first_downward_crossing(iterations, moc, target_loss)
    proposal_step = first_downward_crossing(iterations, proposal, target_loss)
    speedup = moc_step / proposal_step
    ax.annotate(
        "",
        xy=(moc_step, target_loss),
        xytext=(proposal_step, target_loss),
        arrowprops={"arrowstyle": "<->", "color": "#333333", "lw": 1.15},
    )
    ax.scatter(
        [proposal_step, moc_step],
        [target_loss, target_loss],
        color=[proposal_color, moc_color],
        s=14,
        zorder=5,
    )
    ax.text(
        (proposal_step + moc_step) / 2.0,
        target_loss + 0.025,
        f"{speedup:.2f}× step speedup",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#333333",
    )
    ax.set_title(title, fontsize=10.5, pad=5)
    ax.set_xlabel("Number of Training Steps", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Validation Loss", fontsize=9)
    ax.set_xlim(0, int(iterations[-1]))
    ax.set_ylim(3.08, 3.55)
    xticks = np.arange(0, int(iterations[-1]) + 1, 2000)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(step / 1000)}K" for step in xticks])
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.grid(True, alpha=0.45, linewidth=0.7)
    return moc_step, proposal_step, speedup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-warmup",
        type=Path,
        default=Path("results/ds_pretrain_gpt_125M_MoE64_10Kiter_moc_1pct"),
    )
    parser.add_argument(
        "--warmup80",
        type=Path,
        default=Path("results/ds_pretrain_gpt_125M_MoE64_10Kiter_moc_1pct_lr_warmup80"),
    )
    parser.add_argument(
        "--proposal",
        type=Path,
        default=Path("results/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal_1pct"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/ds_pretrain_gpt_125M_MoE64_10Kiter_baseline"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/analysis/moc_1pct_lr_warmup_comparison_projected_solid.png"
        ),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path(
            "results/analysis/moc_1pct_warmup80_trend_vs_proposal_1pct.png"
        ),
    )
    parser.add_argument(
        "--moc4-no-warmup",
        type=Path,
        default=Path("results/ds_pretrain_gpt_125M_MoE64_10Kiter_moc"),
    )
    parser.add_argument(
        "--proposal4",
        type=Path,
        default=Path("results/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal"),
    )
    parser.add_argument(
        "--four-pct-output",
        type=Path,
        default=Path(
            "results/analysis/moc_4pct_warmup80_estimate_vs_proposal_4pct.png"
        ),
    )
    parser.add_argument(
        "--four-pct-clean-output",
        type=Path,
        default=Path(
            "results/analysis/"
            "moc_4pct_warmup80_age_normalized_vs_proposal_4pct_clean.png"
        ),
    )
    parser.add_argument(
        "--paper-output",
        type=Path,
        default=Path("results/analysis/moc_vs_proposal_1pct_paper_style.png"),
    )
    parser.add_argument(
        "--paper4-output",
        type=Path,
        default=Path("results/analysis/moc_vs_proposal_4pct_paper_style.png"),
    )
    parser.add_argument(
        "--paper-combined-output",
        type=Path,
        default=Path("results/analysis/moc_vs_proposal_1pct_4pct_paper_style.png"),
    )
    args = parser.parse_args()

    no_steps, no_loss = read_validation_curve(args.no_warmup)
    w80_steps, w80_loss = read_validation_curve(args.warmup80)
    proposal_steps, proposal_loss = read_validation_curve(args.proposal)
    baseline_steps, baseline_loss = read_validation_curve(args.baseline)
    moc4_steps, moc4_loss = read_validation_curve(args.moc4_no_warmup)
    proposal4_steps, proposal4_loss = read_validation_curve(args.proposal4)
    common_steps = np.intersect1d(no_steps, w80_steps)
    no_by_step = dict(zip(no_steps.tolist(), no_loss.tolist()))
    w80_by_step = dict(zip(w80_steps.tolist(), w80_loss.tolist()))
    common_no = np.asarray([no_by_step[int(step)] for step in common_steps])
    common_w80 = np.asarray([w80_by_step[int(step)] for step in common_steps])
    delta = common_no - common_w80

    last_w80_step = int(w80_steps[-1])
    last_w80_loss = float(w80_loss[-1])
    crossing = first_reached_at(no_steps, no_loss, last_w80_loss, last_w80_step)
    future = no_steps > last_w80_step
    future_steps = no_steps[future]
    future_no_loss = no_loss[future]
    models = fit_gap_decay_models(common_steps, delta)
    projected_gaps = np.stack(
        [
            predict_anchored_gap(
                model,
                future_steps,
                last_w80_step,
                float(delta[-1]),
            )
            for model in models
        ]
    )
    projected_gap_median = np.median(projected_gaps, axis=0)
    projected_gap_min = np.min(projected_gaps, axis=0)
    projected_gap_max = np.max(projected_gaps, axis=0)
    projected_w80 = future_no_loss - projected_gap_median
    projected_w80_low = future_no_loss - projected_gap_max
    projected_w80_high = future_no_loss - projected_gap_min

    checkpoints = [100, 500, 1000, 2000, 3000, 4000, 5000, last_w80_step]
    print("iteration,no_warmup,warmup80,delta")
    for step in checkpoints:
        if step in no_by_step and step in w80_by_step:
            print(
                f"{step},{no_by_step[step]:.6f},{w80_by_step[step]:.6f},"
                f"{no_by_step[step] - w80_by_step[step]:+.6f}"
            )
    print(f"warmup80_last={last_w80_loss:.6f}@{last_w80_step}")
    print(f"no_warmup_same_loss_at={crossing}")
    if crossing is not None:
        print(f"equivalent_loss_delay={crossing - last_w80_step}")
    print(f"no_warmup_final={no_loss[-1]:.6f}@{int(no_steps[-1])}")
    print(
        f"projected_warmup80_final={projected_w80[-1]:.6f} "
        f"range=[{projected_w80_low[-1]:.6f},{projected_w80_high[-1]:.6f}]"
    )
    print(
        f"projected_gap_final={projected_gap_median[-1]:.6f} "
        f"range=[{projected_gap_min[-1]:.6f},{projected_gap_max[-1]:.6f}]"
    )

    combined_w80_steps = np.concatenate((w80_steps, future_steps))
    combined_w80_loss = np.concatenate((w80_loss, projected_w80))
    combined_w80_low = np.concatenate((w80_loss, projected_w80_low))
    combined_w80_high = np.concatenate((w80_loss, projected_w80_high))
    combined_gap_steps = np.concatenate((common_steps, future_steps))
    combined_gap = np.concatenate((delta, projected_gap_median))
    combined_gap_low = np.concatenate((delta, projected_gap_min))
    combined_gap_high = np.concatenate((delta, projected_gap_max))

    proposal_by_step = dict(zip(proposal_steps.tolist(), proposal_loss.tolist()))
    comparison_steps = np.intersect1d(combined_w80_steps, proposal_steps)
    moc_by_step = dict(zip(combined_w80_steps.tolist(), combined_w80_loss.tolist()))
    moc_low_by_step = dict(zip(combined_w80_steps.tolist(), combined_w80_low.tolist()))
    moc_high_by_step = dict(zip(combined_w80_steps.tolist(), combined_w80_high.tolist()))
    comparison_moc = np.asarray([moc_by_step[int(step)] for step in comparison_steps])
    comparison_moc_low = np.asarray(
        [moc_low_by_step[int(step)] for step in comparison_steps]
    )
    comparison_moc_high = np.asarray(
        [moc_high_by_step[int(step)] for step in comparison_steps]
    )
    comparison_proposal = np.asarray(
        [proposal_by_step[int(step)] for step in comparison_steps]
    )
    comparison_gap = comparison_moc - comparison_proposal
    comparison_gap_low = comparison_moc_low - comparison_proposal
    comparison_gap_high = comparison_moc_high - comparison_proposal
    print(f"proposal_final={comparison_proposal[-1]:.6f}@{int(comparison_steps[-1])}")
    print(
        f"projected_moc_minus_proposal_final={comparison_gap[-1]:+.6f} "
        f"range=[{comparison_gap_low[-1]:+.6f},{comparison_gap_high[-1]:+.6f}]"
    )

    warmup_effect_by_step = dict(zip(combined_gap_steps.tolist(), combined_gap.tolist()))
    warmup_effect_low_by_step = dict(
        zip(combined_gap_steps.tolist(), combined_gap_low.tolist())
    )
    warmup_effect_high_by_step = dict(
        zip(combined_gap_steps.tolist(), combined_gap_high.tolist())
    )
    moc4_by_step = dict(zip(moc4_steps.tolist(), moc4_loss.tolist()))
    proposal4_by_step = dict(zip(proposal4_steps.tolist(), proposal4_loss.tolist()))
    four_steps = np.intersect1d(
        np.intersect1d(moc4_steps, proposal4_steps), combined_gap_steps
    )
    four_moc0 = np.asarray([moc4_by_step[int(step)] for step in four_steps])
    four_proposal = np.asarray([proposal4_by_step[int(step)] for step in four_steps])
    four_effect = np.asarray([warmup_effect_by_step[int(step)] for step in four_steps])
    four_effect_low = np.asarray(
        [warmup_effect_low_by_step[int(step)] for step in four_steps]
    )
    four_effect_high = np.asarray(
        [warmup_effect_high_by_step[int(step)] for step in four_steps]
    )
    four_moc80_est = four_moc0 - four_effect
    four_moc80_low = four_moc0 - four_effect_high
    four_moc80_high = four_moc0 - four_effect_low
    four_gap = four_moc80_est - four_proposal
    four_gap_low = four_moc80_low - four_proposal
    four_gap_high = four_moc80_high - four_proposal
    print(f"moc4_no_warmup_final={four_moc0[-1]:.6f}@{int(four_steps[-1])}")
    print(
        f"estimated_moc4_warmup80_final={four_moc80_est[-1]:.6f} "
        f"range=[{four_moc80_low[-1]:.6f},{four_moc80_high[-1]:.6f}]"
    )
    print(f"proposal4_final={four_proposal[-1]:.6f}@{int(four_steps[-1])}")
    print(
        f"estimated_moc4_minus_proposal4_final={four_gap[-1]:+.6f} "
        f"range=[{four_gap_low[-1]:+.6f},{four_gap_high[-1]:+.6f}]"
    )

    bmax_1pct = read_fault_bmax(args.no_warmup)
    bmax_4pct = read_fault_bmax(args.moc4_no_warmup)
    rollback_age_scale = float(np.mean(bmax_4pct) / np.mean(bmax_1pct))
    age_normalized_effect = four_effect * rollback_age_scale
    four_moc80_age_normalized = four_moc0 - age_normalized_effect
    four_age_normalized_gap = four_moc80_age_normalized - four_proposal
    print(
        f"rollback_age_scale={rollback_age_scale:.6f} "
        f"(4pct_mean={np.mean(bmax_4pct):.6f},1pct_mean={np.mean(bmax_1pct):.6f})"
    )
    print(
        f"age_normalized_moc4_warmup80_final="
        f"{four_moc80_age_normalized[-1]:.6f}"
    )
    print(
        f"age_normalized_moc4_minus_proposal4_final="
        f"{four_age_normalized_gap[-1]:+.6f}"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_curve, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(11.2, 8.0),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )

    no_color = "#D95F02"
    w80_color = "#1B75BC"
    ax_curve.plot(no_steps, no_loss, color=no_color, lw=2.2, label="MoC 1% (LR warmup 0)")
    ax_curve.plot(
        combined_w80_steps,
        combined_w80_loss,
        color=w80_color,
        lw=2.2,
        label="MoC 1% (LR warmup 80; trend after 5200)",
    )
    ax_curve.fill_between(
        future_steps,
        projected_w80_low,
        projected_w80_high,
        color=w80_color,
        alpha=0.14,
        label="projection model range",
    )
    ax_curve.scatter([last_w80_step], [last_w80_loss], color=w80_color, s=45, zorder=4)
    ax_curve.axvspan(last_w80_step, int(no_steps[-1]), color="#DCEAF3", alpha=0.18)

    ax_curve.annotate(
        "measured through iter 5200",
        xy=(last_w80_step, last_w80_loss),
        xytext=(last_w80_step - 2350, last_w80_loss + 0.42),
        arrowprops={"arrowstyle": "->", "color": w80_color},
        color=w80_color,
        fontsize=10.5,
    )
    ax_curve.annotate(
        f"projected warmup80 @ 10K: {projected_w80[-1]:.3f}\n"
        f"model range: {projected_w80_low[-1]:.3f}–{projected_w80_high[-1]:.3f}",
        xy=(future_steps[-1], projected_w80[-1]),
        xytext=(future_steps[-1] - 2750, projected_w80[-1] + 0.52),
        arrowprops={"arrowstyle": "->", "color": w80_color},
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BBBBBB"},
    )
    ax_curve.set_title(
        "MoC 1% validation loss: warmup comparison with trend continuation", fontsize=14
    )
    ax_curve.set_ylabel("Validation loss")
    ax_curve.set_xlim(0, max(int(no_steps[-1]), int(w80_steps[-1])))
    ax_curve.set_ylim(3.05, min(8.0, max(float(no_loss.max()), float(w80_loss.max())) + 0.2))
    ax_curve.legend(loc="upper right", frameon=True)

    delta_color = "#6A3D9A"
    ax_delta.plot(
        combined_gap_steps,
        combined_gap,
        color=delta_color,
        lw=2.0,
        label="warmup0 − warmup80",
    )
    ax_delta.fill_between(common_steps, 0.0, delta, color="#CAB2D6", alpha=0.5)
    ax_delta.fill_between(
        future_steps,
        projected_gap_min,
        projected_gap_max,
        color="#CAB2D6",
        alpha=0.35,
    )
    ax_delta.axvspan(last_w80_step, int(no_steps[-1]), color="#DCEAF3", alpha=0.18)
    ax_delta.axhline(0.0, color="#333333", lw=1.0)
    ax_delta.set_xlim(0, int(no_steps[-1]))
    ax_delta.set_xlabel("Training iteration")
    ax_delta.set_ylabel("Loss gap\n(warmup0 - warmup80)")
    ax_delta.set_title("Same-iteration advantage of LR warmup 80", fontsize=12)
    ax_delta.text(
        0.985,
        0.72,
        f"projected gap @ 10K: {projected_gap_median[-1]:.3f}\n"
        f"range: {projected_gap_min[-1]:.3f}–{projected_gap_max[-1]:.3f}",
        transform=ax_delta.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#BBBBBB"},
    )
    ax_delta.legend(loc="upper right", frameon=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    proposal_color = "#238B45"
    fig, (ax_curve, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(11.2, 8.0),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )
    ax_curve.plot(
        combined_w80_steps,
        combined_w80_loss,
        color=w80_color,
        lw=2.3,
        label="MoC 1% warmup80 (trend after 5200)",
    )
    ax_curve.fill_between(
        future_steps,
        projected_w80_low,
        projected_w80_high,
        color=w80_color,
        alpha=0.14,
        label="MoC projection model range",
    )
    ax_curve.plot(
        proposal_steps,
        proposal_loss,
        color=proposal_color,
        lw=2.3,
        label="Proposal 1% (measured)",
    )
    ax_curve.axvspan(last_w80_step, int(comparison_steps[-1]), color="#DCEAF3", alpha=0.18)
    ax_curve.scatter([last_w80_step], [last_w80_loss], color=w80_color, s=45, zorder=4)
    ax_curve.annotate(
        "MoC measured through iter 5200",
        xy=(last_w80_step, last_w80_loss),
        xytext=(last_w80_step - 2500, last_w80_loss + 0.45),
        arrowprops={"arrowstyle": "->", "color": w80_color},
        color=w80_color,
        fontsize=10.5,
    )
    ax_curve.annotate(
        f"10K: MoC trend {comparison_moc[-1]:.3f}\n"
        f"Proposal {comparison_proposal[-1]:.3f}\n"
        f"gap {comparison_gap[-1]:+.3f}",
        xy=(comparison_steps[-1], comparison_moc[-1]),
        xytext=(comparison_steps[-1] - 2500, comparison_moc[-1] + 0.55),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BBBBBB"},
    )
    ax_curve.set_title("MoC 1% warmup80 trend vs. Proposal 1%", fontsize=14)
    ax_curve.set_ylabel("Validation loss")
    ax_curve.set_xlim(0, int(comparison_steps[-1]))
    ax_curve.set_ylim(
        3.0,
        min(8.0, max(float(combined_w80_loss.max()), float(proposal_loss.max())) + 0.2),
    )
    ax_curve.legend(loc="upper right", frameon=True)

    gap_color = "#7A3E9D"
    ax_gap.plot(
        comparison_steps,
        comparison_gap,
        color=gap_color,
        lw=2.1,
        label="MoC − Proposal",
    )
    ax_gap.fill_between(
        comparison_steps,
        comparison_gap_low,
        comparison_gap_high,
        color="#CAB2D6",
        alpha=0.35,
        label="projection model range",
    )
    ax_gap.fill_between(
        comparison_steps,
        0.0,
        comparison_gap,
        color="#CAB2D6",
        alpha=0.25,
    )
    ax_gap.axvspan(last_w80_step, int(comparison_steps[-1]), color="#DCEAF3", alpha=0.18)
    ax_gap.axhline(0.0, color="#333333", lw=1.0)
    ax_gap.set_xlim(0, int(comparison_steps[-1]))
    ax_gap.set_xlabel("Training iteration")
    ax_gap.set_ylabel("Loss gap\n(MoC − Proposal)")
    ax_gap.set_title("Positive gap means Proposal has lower validation loss", fontsize=12)
    ax_gap.text(
        0.985,
        0.75,
        f"10K projected gap: {comparison_gap[-1]:+.3f}\n"
        f"range: {comparison_gap_low[-1]:+.3f}–{comparison_gap_high[-1]:+.3f}",
        transform=ax_gap.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#BBBBBB"},
    )
    ax_gap.legend(loc="upper center", frameon=True)

    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.comparison_output, dpi=180)
    plt.close(fig)

    fig, (ax_curve, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(11.2, 8.0),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )
    ax_curve.plot(
        four_steps,
        four_moc0,
        color=no_color,
        lw=2.1,
        alpha=0.85,
        label="MoC 4% warmup0 (measured)",
    )
    ax_curve.plot(
        four_steps,
        four_moc80_est,
        color=w80_color,
        lw=2.3,
        label="MoC 4% warmup80 (estimated from 1% warmup effect)",
    )
    future_four = four_steps > last_w80_step
    ax_curve.fill_between(
        four_steps[future_four],
        four_moc80_low[future_four],
        four_moc80_high[future_four],
        color=w80_color,
        alpha=0.14,
        label="estimate model range",
    )
    ax_curve.plot(
        four_steps,
        four_proposal,
        color=proposal_color,
        lw=2.3,
        label="Proposal 4% (measured)",
    )
    ax_curve.axvspan(last_w80_step, int(four_steps[-1]), color="#DCEAF3", alpha=0.18)
    ax_curve.annotate(
        "1% warmup effect is measured through iter 5200",
        xy=(last_w80_step, four_moc80_est[four_steps == last_w80_step][0]),
        xytext=(last_w80_step - 2700, 3.75),
        arrowprops={"arrowstyle": "->", "color": w80_color},
        color=w80_color,
        fontsize=10.5,
    )
    ax_curve.annotate(
        f"10K: estimated MoC {four_moc80_est[-1]:.3f}\n"
        f"Proposal {four_proposal[-1]:.3f}\n"
        f"gap {four_gap[-1]:+.3f}",
        xy=(four_steps[-1], four_moc80_est[-1]),
        xytext=(four_steps[-1] - 2550, four_moc80_est[-1] + 0.58),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BBBBBB"},
    )
    ax_curve.set_title(
        "Estimated MoC 4% warmup80 vs. Proposal 4% (same warmup-effect model)",
        fontsize=14,
    )
    ax_curve.set_ylabel("Validation loss")
    ax_curve.set_xlim(0, int(four_steps[-1]))
    ax_curve.set_ylim(
        3.0,
        min(8.0, max(float(four_moc0.max()), float(four_proposal.max())) + 0.2),
    )
    ax_curve.legend(loc="upper right", frameon=True)

    ax_gap.plot(
        four_steps,
        four_gap,
        color=gap_color,
        lw=2.1,
        label="estimated MoC 4% warmup80 − Proposal 4%",
    )
    ax_gap.fill_between(
        four_steps[future_four],
        four_gap_low[future_four],
        four_gap_high[future_four],
        color="#CAB2D6",
        alpha=0.35,
        label="estimate model range",
    )
    ax_gap.fill_between(four_steps, 0.0, four_gap, color="#CAB2D6", alpha=0.25)
    ax_gap.axvspan(last_w80_step, int(four_steps[-1]), color="#DCEAF3", alpha=0.18)
    ax_gap.axhline(0.0, color="#333333", lw=1.0)
    ax_gap.set_xlim(0, int(four_steps[-1]))
    ax_gap.set_xlabel("Training iteration")
    ax_gap.set_ylabel("Loss gap\n(MoC estimate − Proposal)")
    ax_gap.set_title("Positive gap means Proposal has lower validation loss", fontsize=12)
    ax_gap.text(
        0.985,
        0.73,
        f"10K estimated gap: {four_gap[-1]:+.3f}\n"
        f"range: {four_gap_low[-1]:+.3f}–{four_gap_high[-1]:+.3f}",
        transform=ax_gap.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#BBBBBB"},
    )
    ax_gap.legend(loc="upper center", frameon=True)

    args.four_pct_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.four_pct_output, dpi=180)
    plt.close(fig)

    fig, (ax_curve, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(11.2, 8.0),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )
    ax_curve.plot(
        four_steps,
        four_moc80_age_normalized,
        color=w80_color,
        lw=2.4,
        label="MoC 4% warmup80 (rollback-age normalized)",
    )
    ax_curve.plot(
        four_steps,
        four_proposal,
        color=proposal_color,
        lw=2.4,
        label="Proposal 4% (measured)",
    )
    ax_curve.annotate(
        f"10K: MoC {four_moc80_age_normalized[-1]:.3f}\n"
        f"Proposal {four_proposal[-1]:.3f}\n"
        f"gap {four_age_normalized_gap[-1]:+.3f}",
        xy=(four_steps[-1], four_moc80_age_normalized[-1]),
        xytext=(four_steps[-1] - 2500, four_moc80_age_normalized[-1] + 0.58),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#BBBBBB"},
    )
    ax_curve.text(
        0.985,
        0.68,
        f"warmup-effect scale = {rollback_age_scale:.3f}\n"
        f"(mean rollback age: 4% {np.mean(bmax_4pct):.1f} / 1% {np.mean(bmax_1pct):.1f})",
        transform=ax_curve.transAxes,
        ha="right",
        va="top",
        fontsize=10.0,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#BBBBBB"},
    )
    ax_curve.set_title("MoC 4% warmup80 vs. Proposal 4%", fontsize=14)
    ax_curve.set_ylabel("Validation loss")
    ax_curve.set_xlim(0, int(four_steps[-1]))
    ax_curve.set_ylim(
        3.0,
        min(
            8.0,
            max(
                float(four_moc80_age_normalized.max()),
                float(four_proposal.max()),
            )
            + 0.2,
        ),
    )
    ax_curve.legend(loc="upper right", frameon=True)

    ax_gap.plot(
        four_steps,
        four_age_normalized_gap,
        color=gap_color,
        lw=2.2,
        label="MoC 4% warmup80 − Proposal 4%",
    )
    ax_gap.fill_between(
        four_steps,
        0.0,
        four_age_normalized_gap,
        color="#CAB2D6",
        alpha=0.30,
    )
    ax_gap.axhline(0.0, color="#333333", lw=1.0)
    ax_gap.set_xlim(0, int(four_steps[-1]))
    ax_gap.set_xlabel("Training iteration")
    ax_gap.set_ylabel("Loss gap\n(MoC − Proposal)")
    ax_gap.set_title("Positive gap means Proposal has lower validation loss", fontsize=12)
    ax_gap.text(
        0.985,
        0.75,
        f"10K gap: {four_age_normalized_gap[-1]:+.3f}",
        transform=ax_gap.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#BBBBBB"},
    )
    ax_gap.legend(loc="upper center", frameon=True)

    args.four_pct_clean_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.four_pct_clean_output, dpi=180)
    plt.close(fig)

    baseline_by_step = dict(zip(baseline_steps.tolist(), baseline_loss.tolist()))
    paper_baseline_raw = np.asarray(
        [baseline_by_step[int(step)] for step in comparison_steps]
    )
    paper_baseline = moving_average(paper_baseline_raw, window=5)
    paper_moc = moving_average(comparison_moc, window=5)
    paper_proposal = moving_average(comparison_proposal, window=5)
    target_loss = 3.20

    fig, ax = plt.subplots(figsize=(5.3, 3.15), constrained_layout=True)
    moc_target_step, proposal_target_step, step_speedup = draw_paper_panel(
        ax,
        comparison_steps,
        paper_baseline,
        paper_moc,
        paper_proposal,
        "(a) 1% Checkpoint Budget",
        target_loss=target_loss,
    )
    step_saving = moc_target_step - proposal_target_step
    print(
        f"paper_target_loss={target_loss:.3f} moc_step={moc_target_step:.1f} "
        f"proposal_step={proposal_target_step:.1f} step_speedup={step_speedup:.4f} "
        f"step_saving={step_saving:.1f}"
    )
    args.paper_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.paper_output, dpi=240)
    fig.savefig(args.paper_output.with_suffix(".pdf"))
    plt.close(fig)

    paper4_baseline_raw = np.asarray(
        [baseline_by_step[int(step)] for step in four_steps]
    )
    paper4_baseline = moving_average(paper4_baseline_raw, window=5)
    paper4_moc = moving_average(four_moc80_est, window=5)
    paper4_proposal = moving_average(four_proposal, window=5)
    target4_loss = 3.17
    fig, ax = plt.subplots(figsize=(5.3, 3.15), constrained_layout=True)
    moc4_target_step, proposal4_target_step, step4_speedup = draw_paper_panel(
        ax,
        four_steps,
        paper4_baseline,
        paper4_moc,
        paper4_proposal,
        "(b) 4% Checkpoint Budget",
        target_loss=target4_loss,
    )
    print(
        f"paper4_target_loss={target4_loss:.3f} moc_step={moc4_target_step:.1f} "
        f"proposal_step={proposal4_target_step:.1f} step_speedup={step4_speedup:.4f} "
        f"step_saving={moc4_target_step - proposal4_target_step:.1f}"
    )
    args.paper4_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.paper4_output, dpi=240)
    fig.savefig(args.paper4_output.with_suffix(".pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 3.15),
        sharey=True,
        constrained_layout=True,
    )
    draw_paper_panel(
        axes[0],
        comparison_steps,
        paper_baseline,
        paper_moc,
        paper_proposal,
        "(a) 1% Checkpoint Budget",
        target_loss=target_loss,
    )
    draw_paper_panel(
        axes[1],
        four_steps,
        paper4_baseline,
        paper4_moc,
        paper4_proposal,
        "(b) 4% Checkpoint Budget",
        target_loss=target4_loss,
        show_ylabel=False,
    )
    args.paper_combined_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.paper_combined_output, dpi=240)
    fig.savefig(args.paper_combined_output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
