"""Plot score_diag JSONL: v_t CDF (expert vs non-expert) + top-K membership heatmaps."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse(path: Path):
    recs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return recs


def plot_vt_cdf(recs, out_path):
    """For the last 4 measurement iters, overlay expert vs non-expert v_t CDF."""
    recs_with_vt = [r for r in recs if "vt_percentiles" in r]
    if not recs_with_vt:
        return
    steps_to_plot = recs_with_vt[::max(1, len(recs_with_vt) // 4)][-4:]
    fig, axes = plt.subplots(1, len(steps_to_plot), figsize=(4 * len(steps_to_plot), 4), squeeze=False)
    for i, r in enumerate(steps_to_plot):
        ax = axes[0, i]
        labels = r["vt_percentiles"]["percentile_labels"]
        exp_vals = [r["vt_percentiles"]["expert"]["values"][f"p{p:.2f}"] for p in labels]
        ne_vals = [r["vt_percentiles"]["nonexpert"]["values"][f"p{p:.2f}"] for p in labels]
        exp_n = r["vt_percentiles"]["expert"]["numel"]
        ne_n = r["vt_percentiles"]["nonexpert"]["numel"]
        # Avoid log(0)
        exp_vals = [max(v, 1e-40) for v in exp_vals]
        ne_vals = [max(v, 1e-40) for v in ne_vals]
        ax.plot(labels, exp_vals, label=f"expert (n={exp_n:,})", color="#27ae60", linewidth=1.5)
        ax.plot(labels, ne_vals, label=f"non-expert (n={ne_n:,})", color="#e67e22", linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("percentile")
        ax.set_ylabel("v_t value (log scale)")
        ax.set_title(f"iter {r['step']}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("v_t distribution — expert vs non-expert (percentile CDF, log-y)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_vt_expert_to_ne_mapping(recs, out_path):
    """For each expert v_t percentile x, find the percentile y in non-expert
    distribution whose value equals expert's p_x value. Interpolate in log-v.

    Interpretation:
      - x-axis: expert percentile x (0-100)
      - y-axis: equivalent non-expert percentile (where exp's v_t at pct x
                would land in non-expert's distribution)
      - y=100 means "expert top X% value exceeds all non-expert values"
      - y<x means expert at that percentile is LOWER than non-expert's same pct
    """
    recs_with_vt = [r for r in recs if "vt_percentiles" in r]
    if not recs_with_vt:
        return
    r = recs_with_vt[-1]  # final measurement
    labels = r["vt_percentiles"]["percentile_labels"]
    exp_vals = [r["vt_percentiles"]["expert"]["values"][f"p{p:.2f}"] for p in labels]
    ne_vals = [r["vt_percentiles"]["nonexpert"]["values"][f"p{p:.2f}"] for p in labels]

    # Log-space interpolation over non-expert (percentile, value) curve.
    # Build monotone-log lookup: ne_pct (x) → log10(ne_val) (y).
    ne_pct_arr = np.array(labels, dtype=float)
    ne_val_arr = np.array([max(v, 1e-40) for v in ne_vals], dtype=float)
    # Sort by non-expert value ascending (should already be if percentiles are monotone).
    order = np.argsort(ne_val_arr)
    ne_val_sorted = ne_val_arr[order]
    ne_pct_sorted = ne_pct_arr[order]
    log_ne_val_sorted = np.log10(ne_val_sorted)

    def val_to_ne_pct(v):
        lv = np.log10(max(v, 1e-40))
        if lv <= log_ne_val_sorted[0]:
            return 0.0
        if lv >= log_ne_val_sorted[-1]:
            return 100.0
        # Linear interp in (log_val → pct)
        return float(np.interp(lv, log_ne_val_sorted, ne_pct_sorted))

    equiv_ne_pct = [val_to_ne_pct(v) for v in exp_vals]

    # Zoomed: top 1% of expert (pct 99-100) where the action happens.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(labels, equiv_ne_pct, marker=".", markersize=3, color="#8e44ad", linewidth=1.5,
            label="expert v_t → ne v_t percentile")
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--", linewidth=1, label="y=x (identical dist.)")
    ax.set_xlabel("expert percentile x (%)")
    ax.set_ylabel("equivalent non-expert percentile y (%)")
    ax.set_title(f"Cross-category v_t mapping @ iter {r['step']}")
    ax.legend(); ax.grid(alpha=0.3)

    # Zoom on expert tail
    ax = axes[1]
    # Only use percentiles >= 90 for zoom
    idxs = [i for i, p in enumerate(labels) if p >= 90]
    lx = [labels[i] for i in idxs]
    ly = [equiv_ne_pct[i] for i in idxs]
    ax.plot(lx, ly, marker="o", markersize=4, color="#8e44ad", linewidth=1.5)
    ax.plot([90, 100], [90, 100], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("expert percentile x (%, zoom 90-100)")
    ax.set_ylabel("equivalent non-expert percentile y (%)")
    ax.set_title("Expert tail (p90-p99.99) — where does it land in ne distribution?")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_vt_cdf_overlap_zoom(recs, out_path):
    """Focus on 95-99.99 percentile range where expert and non-expert may overlap."""
    recs_with_vt = [r for r in recs if "vt_percentiles" in r]
    if not recs_with_vt:
        return
    r = recs_with_vt[-1]  # final measurement
    labels = r["vt_percentiles"]["percentile_labels"]
    exp_vals = [r["vt_percentiles"]["expert"]["values"][f"p{p:.2f}"] for p in labels]
    ne_vals = [r["vt_percentiles"]["nonexpert"]["values"][f"p{p:.2f}"] for p in labels]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (xmin, xmax, title) in zip(axes, [(0, 100, "full range"), (95, 100, "zoomed tail 95-100%")]):
        mask = [xmin <= p <= xmax for p in labels]
        lx = [labels[i] for i, m in enumerate(mask) if m]
        le = [max(exp_vals[i], 1e-40) for i, m in enumerate(mask) if m]
        ln = [max(ne_vals[i], 1e-40) for i, m in enumerate(mask) if m]
        ax.plot(lx, le, marker="o", markersize=3, label="expert", color="#27ae60", linewidth=1.5)
        ax.plot(lx, ln, marker="s", markersize=3, label="non-expert", color="#e67e22", linewidth=1.5)
        ax.set_yscale("log"); ax.set_xlabel("percentile"); ax.set_ylabel("v_t value")
        ax.set_title(f"iter {r['step']} — {title}")
        ax.legend(); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_per_layer_topk(recs, out_path):
    """Per-layer top-K membership fraction over iters."""
    layers = set()
    for r in recs:
        pl = r.get("expert", {}).get("topk_membership", {}).get("per_layer", {})
        layers.update(int(k) for k in pl.keys())
    if not layers:
        return
    layers = sorted(layers)
    steps = [r["step"] for r in recs]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    for i, L in enumerate(layers):
        fracs = []
        for r in recs:
            pl = r.get("expert", {}).get("topk_membership", {}).get("per_layer", {})
            key = str(L)
            if key not in pl:
                key = L  # numeric key fallback
            entry = pl.get(key) or pl.get(str(L)) or {}
            fracs.append(entry.get("frac", 0.0) * 100)
        ax.plot(steps, fracs, marker="o", markersize=3, label=f"layer {L}", color=colors[i])
    k_ratio_pct = recs[0].get("expert", {}).get("topk_membership", {}).get("topk_ratio", 0.01) * 100
    ax.axhline(k_ratio_pct, color="red", linestyle="--", linewidth=1,
               label=f"uniform expectation ({k_ratio_pct:.2f}%)")
    ax.set_xlabel("iteration"); ax.set_ylabel("% of layer's expert elements in Top-K")
    ax.set_title("Per-layer Top-K membership fraction")
    ax.legend(ncol=2, fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_per_expert_topk_heatmap(recs, out_path):
    """Heatmap: rows = (L, E), cols = iters; color = top-K frac %."""
    expert_keys = set()
    for r in recs:
        pe = r.get("expert", {}).get("topk_membership", {}).get("per_expert", {})
        expert_keys.update(pe.keys())
    if not expert_keys:
        return
    # Sort by (layer, expert_idx)
    def _sort_key(k):
        import re
        m = re.match(r"L(\d+)_E(\d+)", k)
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)
    expert_keys = sorted(expert_keys, key=_sort_key)
    steps = [r["step"] for r in recs]
    matrix = np.zeros((len(expert_keys), len(steps)))
    for j, r in enumerate(recs):
        pe = r.get("expert", {}).get("topk_membership", {}).get("per_expert", {})
        for i, k in enumerate(expert_keys):
            matrix[i, j] = pe.get(k, {}).get("frac", 0.0) * 100
    fig, ax = plt.subplots(figsize=(12, max(6, len(expert_keys) * 0.2)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(expert_keys)))
    ax.set_yticklabels(expert_keys, fontsize=6)
    ax.set_xlabel("iteration"); ax.set_ylabel("(layer, expert)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03)
    cbar.set_label("% in Top-K (higher = more often selected)")
    ax.set_title("Per-expert Top-K membership over iterations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_expert_score_per_layer_avg(recs, out_path):
    """Per-layer average score (v̂×δ²) per-expert over iters."""
    layers = set()
    for r in recs:
        pl = r.get("expert", {}).get("per_layer", {})
        layers.update(int(k) for k in pl.keys())
    if not layers:
        return
    layers = sorted(layers)
    steps = [r["step"] for r in recs]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    for i, L in enumerate(layers):
        avgs = []
        for r in recs:
            pl = r.get("expert", {}).get("per_layer", {})
            entry = pl.get(str(L)) or pl.get(L) or {}
            avgs.append(entry.get("avg_per_expert", 0.0))
        ax.plot(steps, avgs, marker="o", markersize=3, label=f"layer {L}", color=colors[i])
    ax.set_yscale("log")
    ax.set_xlabel("iteration"); ax.set_ylabel("avg v̂×δ² per expert (log)")
    ax.set_title("Per-layer average score per expert (log-y)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_gnb_cdf(recs, out_path):
    """|H_diag| percentile CDF across measurement iters (one curve per iter).
    Backward-compat: accepts either new 'H_diag' or legacy 'hessian' key."""
    def _hkey(r):
        hp = r.get("h_percentiles", {})
        return "H_diag" if hp.get("H_diag", {}).get("numel", 0) > 0 else "hessian"
    recs_h = [r for r in recs
              if r.get("h_percentiles", {}).get(_hkey(r), {}).get("numel", 0) > 0]
    if not recs_h:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(recs_h), 1)))
    for ax, key, title in [
        (axes[0, 0], "H_diag", "|H_diag| CDF (GNB / Hutchinson estimate)"),
        (axes[0, 1], "score_h", "score_H = |H|·δ² CDF (GNB-based score)"),
    ]:
        for i, r in enumerate(recs_h):
            hp = r["h_percentiles"]
            use_key = key
            if key == "H_diag" and use_key not in hp:
                use_key = "hessian"  # legacy
            labels = hp["percentile_labels"]
            vals = [hp[use_key]["values"][f"p{p:.2f}"] for p in labels]
            vals = [max(v, 1e-40) for v in vals]
            n = hp[use_key]["numel"]
            ax.plot(labels, vals, color=colors[i], alpha=0.7, linewidth=1.2,
                    label=f"iter {r['step']} (n={n:,})" if i % max(1, len(recs_h)//5) == 0 else None)
        ax.set_yscale("log")
        ax.set_xlabel("percentile (0→99.99)")
        ax.set_ylabel(f"{key} value (log scale)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("GNB-based score distribution across training iters")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_vt_vs_gnb_compare(recs, out_path):
    """Overlay v_t CDF (expert) and |H| CDF on same axes at last measurement iter.
    Backward-compat: accepts either 'H_diag' or legacy 'hessian' key."""
    def _hkey(r):
        hp = r.get("h_percentiles", {})
        return "H_diag" if hp.get("H_diag", {}).get("numel", 0) > 0 else "hessian"
    recs_both = [r for r in recs
                 if r.get("vt_percentiles", {}).get("expert", {}).get("numel", 0) > 0
                 and r.get("h_percentiles", {}).get(_hkey(r), {}).get("numel", 0) > 0]
    if not recs_both:
        return
    r = recs_both[-1]
    hk = _hkey(r)
    labels = r["vt_percentiles"]["percentile_labels"]
    vt = [max(r["vt_percentiles"]["expert"]["values"][f"p{p:.2f}"], 1e-40) for p in labels]
    hh = [max(r["h_percentiles"][hk]["values"][f"p{p:.2f}"], 1e-40) for p in labels]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(labels, vt, label=f"v_t (expert, n={r['vt_percentiles']['expert']['numel']:,})",
            color="#2c7bb6", linewidth=1.5)
    ax.plot(labels, hh, label=f"|H_diag| (expert, n={r['h_percentiles'][hk]['numel']:,})",
            color="#d7191c", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("percentile"); ax.set_ylabel("value (log)")
    ax.set_title(f"Expert: v_t vs |H_diag| CDF @ iter {r['step']}")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    recs = parse(args.inp)
    print(f"Parsed {len(recs)} score_diag records from {args.inp}")
    if not recs:
        print("No records; nothing to plot.")
        return

    plot_vt_cdf(recs, args.out_dir / "vt_cdf_4iters.png")
    plot_vt_cdf_overlap_zoom(recs, args.out_dir / "vt_cdf_zoom_final.png")
    plot_vt_expert_to_ne_mapping(recs, args.out_dir / "vt_exp_to_ne_mapping.png")
    plot_per_layer_topk(recs, args.out_dir / "topk_per_layer.png")
    plot_per_expert_topk_heatmap(recs, args.out_dir / "topk_per_expert_heatmap.png")
    plot_expert_score_per_layer_avg(recs, args.out_dir / "score_per_layer_avg.png")
    plot_gnb_cdf(recs, args.out_dir / "gnb_score_cdf.png")
    plot_vt_vs_gnb_compare(recs, args.out_dir / "vt_vs_gnb_cdf.png")
    print(f"Plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
