#!/usr/bin/env python3
"""Plot H / Score / v̂_t CDFs in hutch_H_full-style 4-panel layout.

Layout per metric:
  (top-left)   Full CDF 0-100 pct, linear y
  (top-right)  Top 10% zoom (90-100 pct), linear y
  (bottom-left)  Evolution of key percentiles over iterations, linear y
  (bottom-right) Evolution of key percentiles over iterations, log y
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCORE_DIAG = "/home/asdf1234/Megatron-DeepSpeed/results/exp_3/log/exp3_run1_seed2024.score_diag.jsonl"
OUT_DIR = Path("/home/asdf1234/Megatron-DeepSpeed/results/exp_3/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

records = [json.loads(l) for l in open(SCORE_DIAG) if l.strip()]
records.sort(key=lambda r: r["step"])
ALL_ITERS = [r["step"] for r in records]
print(f"Loaded {len(records)} records, iters: {ALL_ITERS}")


def extract(record, key: str):
    """Return (percentile_x, value_y) sorted arrays for one record's CDF."""
    if key in ("H_diag", "score_h"):
        d = record["h_percentiles"][key]["values"]
    elif key == "expert":
        d = record["vt_percentiles"]["expert"]["values"]
    else:
        raise KeyError(key)
    items = sorted(d.items(), key=lambda kv: float(kv[0][1:]))
    pcts = np.array([float(k[1:]) for k, _ in items])
    vals = np.array([v for _, v in items], dtype=float)
    return pcts, vals


# Overlay iters: equally-spaced selection across the available records.
sel_iters = [ALL_ITERS[i] for i in
             np.linspace(0, len(ALL_ITERS) - 1, min(6, len(ALL_ITERS))).astype(int)]
cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(sel_iters)))
sel_records = [r for r in records if r["step"] in sel_iters]

# Fixed "key percentile" tracks for bottom panels.
KEY_PCT = {
    "top50%": 50.0,
    "top10%": 90.0,
    "top1%":  99.0,
    "top0.1%": 99.9,
    "max":    99.99,
}
KEY_COLORS = {
    "top50%": "#1f77b4",
    "top10%": "#2ca02c",
    "top1%":  "#ff7f0e",
    "top0.1%": "#d62728",
    "max":    "#8e44ad",
}


def value_at_pct(record, key: str, pct: float) -> float:
    pcts, vals = extract(record, key)
    if pct in pcts:
        idx = int(np.where(pcts == pct)[0][0])
        return float(vals[idx])
    return float(np.interp(pct, pcts, vals))


def plot_metric_4panel(key: str, label: str, fname: str, annotate: str = ""):
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    # ── A: full CDF 0-100, linear y ──
    ax = axs[0, 0]
    for r, color in zip(sel_records, cmap):
        pcts, vals = extract(r, key)
        ax.plot(pcts, vals, color=color, label=f"iter {r['step']}", lw=1.6)
    ax.set_xlabel("percentile (0 = smallest, 100 = largest)")
    ax.set_ylabel(f"{label}")
    ax.set_title(f"A. {label} CDF — full 0-100")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # ── B: top 10% zoom (90-100), linear y ──
    ax = axs[0, 1]
    for r, color in zip(sel_records, cmap):
        pcts, vals = extract(r, key)
        mask = pcts >= 90.0
        ax.plot(pcts[mask], vals[mask], color=color, label=f"iter {r['step']}", lw=1.6)
    ax.set_xlabel("percentile (zoom: 90 → 100)")
    ax.set_ylabel(f"{label}")
    ax.set_title(f"B. {label} CDF — top 10% zoom")
    ax.set_xlim(90, 100)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # ── C: evolution over iters, linear y ──
    ax = axs[1, 0]
    for label_pct, pct_val in KEY_PCT.items():
        xs = [r["step"] for r in records]
        ys = [value_at_pct(r, key, pct_val) for r in records]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.5, color=KEY_COLORS[label_pct], label=label_pct)
    ax.set_xlabel("iter")
    ax.set_ylabel(f"{label}")
    ax.set_title(f"C. {label} — evolution over training")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    # ── D: evolution over iters, log y ──
    ax = axs[1, 1]
    for label_pct, pct_val in KEY_PCT.items():
        xs = [r["step"] for r in records]
        ys = [max(value_at_pct(r, key, pct_val), 1e-40) for r in records]
        ax.semilogy(xs, ys, marker="o", ms=4, lw=1.5, color=KEY_COLORS[label_pct], label=label_pct)
    ax.set_xlabel("iter")
    ax.set_ylabel(f"{label} (log scale)")
    ax.set_title(f"D. {label} — log-scale evolution")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3, which="both")

    suptitle = f"{label} — exp3 run1 (K=1%, D=1, F=0.5%)"
    if annotate:
        suptitle += f"  · {annotate}"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    path = OUT_DIR / fname
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  → {path}")


plot_metric_4panel("H_diag",  "|H_diag| (Hutchinson K=4)", "cdf_h_full.png")
plot_metric_4panel("score_h", "Score = |H|·δ²",            "cdf_score_full.png")
plot_metric_4panel("expert",  "v̂_t (Adam)",                 "cdf_vhat_full.png")


# Combined 3-metric comparison at the latest iter, using the same style.
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
r = sel_records[-1]
for key, color, lbl in [("H_diag", "#1f77b4", "|H_diag|"),
                        ("expert", "#d62728", "v̂_t"),
                        ("score_h", "#2ca02c", "Score")]:
    pcts, vals = extract(r, key)
    # Panel 1: full linear
    axs[0].plot(pcts, vals, color=color, label=lbl, lw=2.0)
    # Panel 2: zoom top 10%, linear
    mask = pcts >= 90.0
    axs[1].plot(pcts[mask], vals[mask], color=color, label=lbl, lw=2.0)
axs[0].set_xlabel("percentile")
axs[0].set_ylabel("value")
axs[0].set_title(f"3-metric CDF @ iter {r['step']} — full")
axs[0].set_xlim(0, 100); axs[0].grid(alpha=0.3); axs[0].legend()
axs[1].set_xlabel("percentile (zoom 90-100)")
axs[1].set_ylabel("value")
axs[1].set_title(f"3-metric CDF @ iter {r['step']} — top 10%")
axs[1].set_xlim(90, 100); axs[1].grid(alpha=0.3); axs[1].legend()
fig.suptitle("|H_diag|, v̂_t, Score — latest iter comparison (linear y)", fontsize=12)
fig.tight_layout()
p = OUT_DIR / "cdf_three_latest.png"
fig.savefig(p, dpi=130)
plt.close(fig)
print(f"  → {p}")
