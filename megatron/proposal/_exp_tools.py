"""Experiment helpers: log parsing, metric extraction, plotting.

Parses Megatron-DeepSpeed iteration log lines of the form:

    iteration   123/  1000 | ... | lm loss: 9.876543E+00 | moe loss: 1.234E-02 | ...

and validation lines:

    validation loss at iteration 100 | lm loss value: 9.876E+00 | lm loss PPL: 1.234E+04 |

Returns DataFrames suitable for matplotlib plotting.

Also computes:
- Integrated Gap: Σ (variant_loss[t] - baseline_loss[t]) over t
- Total CP size: walks the cp_dir and sums file sizes
- Per-iter CP size: total / iters
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+ .*? lm loss:\s+([0-9.eE+\-]+).*?(?:moe loss:\s+([0-9.eE+\-]+))?"
)
VAL_RE = re.compile(
    r"validation loss at iteration\s+(\d+).*?lm loss value:\s+([0-9.eE+\-]+).*?PPL:\s+([0-9.eE+\-]+)"
)
REPLAY_VAL_RE = re.compile(
    r"AFTER-REPLAY iteration\s+(\d+).*?lm loss value:\s+([0-9.eE+\-]+).*?PPL:\s+([0-9.eE+\-]+)"
)
ITER_TIME_RE = re.compile(r"elapsed time per iteration \(ms\):\s+([0-9.]+)")


@dataclass
class RunMetrics:
    variant: str
    iters: List[int]
    train_loss: List[float]
    iter_ms: List[float]
    val_iters: List[int]
    val_loss: List[float]
    val_ppl: List[float]
    total_cp_bytes: int = 0
    cp_files: int = 0
    replay_val_loss: float = float("nan")
    replay_val_ppl: float = float("nan")


def parse_log(log_path: Path, variant: str) -> RunMetrics:
    iters: List[int] = []
    losses: List[float] = []
    ms: List[float] = []
    val_iters: List[int] = []
    val_losses: List[float] = []
    val_ppls: List[float] = []
    text = Path(log_path).read_text(errors="ignore")
    replay_loss = float("nan")
    replay_ppl = float("nan")
    for line in text.splitlines():
        m = REPLAY_VAL_RE.search(line)
        if m:
            replay_loss = float(m.group(2))
            replay_ppl = float(m.group(3))
            continue
        m = ITER_RE.search(line)
        if m:
            iters.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            tm = ITER_TIME_RE.search(line)
            ms.append(float(tm.group(1)) if tm else 0.0)
            continue
        m = VAL_RE.search(line)
        if m:
            val_iters.append(int(m.group(1)))
            val_losses.append(float(m.group(2)))
            val_ppls.append(float(m.group(3)))
    return RunMetrics(
        variant=variant,
        iters=iters,
        train_loss=losses,
        iter_ms=ms,
        val_iters=val_iters,
        val_loss=val_losses,
        val_ppl=val_ppls,
        replay_val_loss=replay_loss,
        replay_val_ppl=replay_ppl,
    )


def collect_cp_size(cp_dir: Path) -> Tuple[int, int]:
    cp_dir = Path(cp_dir)
    if not cp_dir.exists():
        return 0, 0
    total = 0
    n = 0
    for p in cp_dir.rglob("*.pt"):
        try:
            total += p.stat().st_size
            n += 1
        except FileNotFoundError:
            pass
    return total, n


def integrated_gap(
    baseline: RunMetrics, variant: RunMetrics, key: str = "train_loss",
) -> float:
    """Σ |variant - baseline| over common iters (for train_loss). Returns 0 if no overlap."""
    base_x = getattr(baseline, "iters" if key == "train_loss" else "val_iters")
    base_y = getattr(baseline, key)
    var_x = getattr(variant, "iters" if key == "train_loss" else "val_iters")
    var_y = getattr(variant, key)
    base_map = dict(zip(base_x, base_y))
    return float(sum(abs(y - base_map[x]) for x, y in zip(var_x, var_y) if x in base_map))


def plot_exp1(runs: Dict[str, RunMetrics], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"baseline": "#2c3e50", "proposal": "#27ae60", "random": "#e74c3c"}

    ax = axes[0, 0]
    for name, r in runs.items():
        ax.plot(r.iters, r.train_loss, label=name, color=colors.get(name, "gray"), alpha=0.85, linewidth=1.4)
    ax.set_xlabel("iteration"); ax.set_ylabel("train lm loss"); ax.set_title("Training loss")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for name, r in runs.items():
        if r.val_iters:
            ax.plot(r.val_iters, r.val_loss, marker="o", label=f"{name} (live)", color=colors.get(name, "gray"))
        # Replay-evaluated final val loss (single point at last iter)
        if not math.isnan(r.replay_val_loss):
            x_last = r.val_iters[-1] if r.val_iters else (r.iters[-1] if r.iters else 0)
            ax.scatter([x_last], [r.replay_val_loss], marker="*", s=180,
                       label=f"{name} (REPLAY)", color=colors.get(name, "gray"),
                       edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_xlabel("iteration"); ax.set_ylabel("validation lm loss"); ax.set_title("Validation loss (live vs replay)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for name, r in runs.items():
        if r.val_iters:
            ax.plot(r.val_iters, r.val_ppl, marker="o", label=name, color=colors.get(name, "gray"))
    ax.set_xlabel("iteration"); ax.set_ylabel("validation PPL"); ax.set_yscale("log")
    ax.set_title("Validation perplexity"); ax.legend(); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    names = list(runs.keys())
    sizes_mb = [runs[n].total_cp_bytes / 1024**2 for n in names]
    bars = ax.bar(names, sizes_mb, color=[colors.get(n, "gray") for n in names])
    for b, v in zip(bars, sizes_mb):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f} MB",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total CP size (MB)"); ax.set_title("Checkpoint storage cost")
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Exp 1 — Importance vs Random vs Baseline (125M-MoE8, WikiText-2, 1000 iter)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def summarize(runs: Dict[str, RunMetrics], baseline_name: str = "baseline") -> str:
    base = runs.get(baseline_name)
    base_final_val = base.val_loss[-1] if (base and base.val_loss) else float("nan")
    lines = ["| variant | live final val loss | **REPLAY val loss** | **REPLAY PPL** | gap vs baseline (replay) | total CP MB |",
             "|---|---|---|---|---|---|"]
    for name, r in runs.items():
        live_vl = r.val_loss[-1] if r.val_loss else float("nan")
        replay_vl = r.replay_val_loss
        replay_pp = r.replay_val_ppl
        sz = r.total_cp_bytes / 1024**2
        if name == baseline_name:
            gap = 0.0
            replay_vl_disp = f"{live_vl:.4f} (no replay)"
            replay_pp_disp = f"{r.val_ppl[-1] if r.val_ppl else float('nan'):.2f}"
        else:
            gap = (replay_vl - base_final_val) if not (math.isnan(replay_vl) or math.isnan(base_final_val)) else float("nan")
            replay_vl_disp = f"{replay_vl:.4f}"
            replay_pp_disp = f"{replay_pp:.2f}"
        lines.append(f"| {name} | {live_vl:.4f} | {replay_vl_disp} | {replay_pp_disp} | {gap:+.4f} | {sz:.1f} |")
    return "\n".join(lines)
