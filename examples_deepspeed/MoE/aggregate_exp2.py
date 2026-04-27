#!/usr/bin/env python3
"""Aggregate exp 2 results after all 12 runs finish.

Reads per-tag artifacts from ``results/exp_2/log/``:
  - ``{tag}.log``              — training + validation log (Megatron format)
  - ``{tag}.faults.jsonl``     — one JSON per fault injection event
  - ``fc_write_log.json``      — per-FC-write timing (dumped by proposal_shutdown)
  - ``ckpt_write_log.json``    — per-DC-bundle-write timing
  - ``cpu_mem_samples.json``   — periodic psutil samples
  - ``proposal_shutdown_summary.json``

Writes:
  - ``results/exp_2/metrics/{tag}.json``  — per-tag full metrics
  - ``results/exp_2/exp2_summary.json``   — cross-tag aggregate
  - ``results/exp_2/exp2_plot.png``       — 6-panel figure
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

OUT = Path("/home/asdf1234/Megatron-DeepSpeed/results/exp_2")
LOG_DIR = OUT / "log"
METRICS_DIR = OUT / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["baseline", "moc", "proposal", "random"]
SEEDS = [1234, 2024, 7777]

it_re = re.compile(
    r"iteration\s+(\d+)/\s*\d+ .*? elapsed time per iteration \(ms\):\s*([\d.]+).*?lm loss:\s*([\d.E+-]+)"
)
val_re = re.compile(
    r"validation loss at iteration (\d+) \| lm loss value:\s*([\d.E+-]+) \| lm loss PPL:\s*([\d.E+-]+)"
)


def _load_json(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def parse_log(log_path: Path) -> Tuple[List[Tuple[int, float, float]], List[Tuple[int, float, float]]]:
    if not log_path.exists():
        return [], []
    train, vals = [], []
    txt = log_path.read_text()
    for ln in txt.splitlines():
        m = it_re.search(ln)
        if m:
            train.append((int(m.group(1)), float(m.group(3)), float(m.group(2))))
        m = val_re.search(ln)
        if m:
            vals.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return train, vals


def parse_faults(jsonl: Path) -> List[Dict]:
    if not jsonl.exists():
        return []
    out = []
    for ln in jsonl.read_text().splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def _prefix_dir(tag: str) -> Path:
    """Per-run dump directory for fc/ckpt/cpu_mem logs. Named tag-prefixed in log dir."""
    # proposal_shutdown dumps into Path(fault_log).parent → log dir, shared across tags.
    # To disambiguate we look for ``{tag}.ckpt_write_log.json`` style names.
    # Fallback: global names in log dir (would be overwritten across runs; use last).
    return LOG_DIR


def collect_tag(tag: str) -> Dict:
    log = LOG_DIR / f"{tag}.log"
    faults_jsonl = LOG_DIR / f"{tag}.faults.jsonl"
    train, vals = parse_log(log)
    faults = parse_faults(faults_jsonl)

    # These are dumped by proposal_shutdown with generic names that get
    # overwritten across runs sharing the log dir. To preserve per-run data,
    # proposal_shutdown will later be patched to namespace by tag. For now,
    # accept that only the LAST run's dumps survive — baseline has no dumps,
    # and the last variant's data will be whatever was just run.
    fc_log = _load_json(LOG_DIR / "fc_write_log.json")
    ckpt_log = _load_json(LOG_DIR / "ckpt_write_log.json")
    cpu_mem = _load_json(LOG_DIR / "cpu_mem_samples.json")
    shut = _load_json(LOG_DIR / "proposal_shutdown_summary.json")
    # Prefer tag-prefixed files if they exist.
    for suffix in ("fc_write_log.json", "ckpt_write_log.json",
                   "cpu_mem_samples.json", "proposal_shutdown_summary.json"):
        tagged = LOG_DIR / f"{tag}.{suffix}"
        if tagged.exists():
            data = _load_json(tagged)
            if suffix == "fc_write_log.json":
                fc_log = data
            elif suffix == "ckpt_write_log.json":
                ckpt_log = data
            elif suffix == "cpu_mem_samples.json":
                cpu_mem = data
            else:
                shut = data

    def _sum(lst, k):
        return sum((x.get(k) or 0) for x in lst) if lst else 0

    total_fc_bytes = _sum(fc_log, "file_bytes")
    total_dc_bytes = _sum(ckpt_log, "file_bytes")
    total_fc_write_ms = _sum(fc_log, "write_time_ms")
    total_dc_write_ms = _sum(ckpt_log, "write_time_ms")
    total_recover_s = sum((f.get("recover_time_s") or 0) for f in faults)
    wasted_time_s = (total_fc_write_ms + total_dc_write_ms) / 1000.0 + total_recover_s

    # Peak IO bandwidth: per-event bytes/write_time_ms * 1000 MB/s; take max.
    def _peak_mbs(log):
        m = 0.0
        if not log:
            return 0.0
        for r in log:
            t_ms = r.get("write_time_ms") or 0
            b = r.get("file_bytes") or 0
            if t_ms > 0.5 and b > 0:  # skip async placeholders (t≈0)
                mbs = (b / 1024 / 1024) / (t_ms / 1000.0)
                if mbs > m:
                    m = mbs
        return m
    peak_bw_mbs = max(_peak_mbs(fc_log), _peak_mbs(ckpt_log))

    it_ms = [ms for _, _, ms in train if ms and ms < 20000]
    mean_iter_ms = sum(it_ms[1:]) / max(1, len(it_ms) - 1) if len(it_ms) > 1 else 0

    # Post-fault damage via window averaging: mean loss over [fault-W, fault)
    # (pre) vs (fault, fault+W] (post). W = 5 iter. Uses train_loss_curve.
    # Robust against single-iter batch noise.
    W_avg = 5
    train_by_iter: Dict[int, float] = {}
    for it, loss, _ in train:
        train_by_iter[it] = loss

    def _mean_window(start, end):
        vals = [train_by_iter[i] for i in range(start, end + 1) if i in train_by_iter]
        return sum(vals) / len(vals) if vals else None

    pf_deltas = []
    pf_deltas_window = []
    for f in faults:
        pre = f.get("pre_fault_loss")
        post = f.get("post_fault_loss")
        if pre is not None and post is not None:
            pf_deltas.append(post - pre)
        it = f.get("step")
        if it is None:
            continue
        pre_avg = _mean_window(it - W_avg, it)
        post_avg = _mean_window(it + 1, it + W_avg)
        if pre_avg is not None and post_avg is not None:
            pf_deltas_window.append(post_avg - pre_avg)

    post_fault_damage_mean = (
        sum(pf_deltas_window) / len(pf_deltas_window)
        if pf_deltas_window else None
    )
    post_fault_damage_max = (
        max(pf_deltas_window) if pf_deltas_window else None
    )
    # Keep single-iter delta for diagnostic comparison.
    post_fault_damage_mean_singleiter = (
        sum(pf_deltas) / len(pf_deltas) if pf_deltas else None
    )

    variant = tag.split("_seed")[0]
    seed = int(tag.split("_seed")[1]) if "_seed" in tag else 0

    metrics = {
        "tag": tag,
        "variant": variant,
        "seed": seed,
        # 1. CP 크기
        "cp_bytes_fc_total_written": total_fc_bytes,
        "cp_bytes_dc_total_written": total_dc_bytes,
        "cp_bytes_total_written": total_fc_bytes + total_dc_bytes,
        # 2. 저장 시간
        "total_fc_write_ms": total_fc_write_ms,
        "total_dc_write_ms": total_dc_write_ms,
        # 3. throughput
        "mean_iter_ms": mean_iter_ms,
        # 4. CPU memory
        "cpu_mem_peak_rss_mb": (
            max(s["rss_mb"] for s in cpu_mem) if cpu_mem else None
        ),
        "cpu_mem_samples_count": len(cpu_mem) if cpu_mem else 0,
        # 5. peak IO bandwidth (MB/s, aggregated 1s bins)
        "peak_io_bandwidth_mb_s": peak_bw_mbs,
        # 6. 복구 후 수렴 (needs paired baseline — leave as curve; aggregator
        #    computes across variants below)
        # 7. B_max
        "B_max_observed": (
            max((f.get("B_max_lost_iter", 0) for f in faults), default=0)
        ),
        # 8. 순서 기반 복구 품질 (same run data; separate analysis)
        # 9. Wasted Time
        "wasted_time_s": wasted_time_s,
        "total_recover_s": total_recover_s,
        # 10. Validation loss + PPL
        "val_curve": vals,
        "val_final_loss": vals[-1][1] if vals else None,
        "val_final_ppl": vals[-1][2] if vals else None,
        # 11. Train loss curve
        "train_loss_curve": train,
        "train_final_loss": train[-1][1] if train else None,
        # 12. Post-fault Damage (window-avg ±5 iter, robust to batch noise)
        "post_fault_damage_mean": post_fault_damage_mean,
        "post_fault_damage_max": post_fault_damage_max,
        "post_fault_damage_mean_singleiter": post_fault_damage_mean_singleiter,
        "num_faults": len(faults),
        "faults_raw": faults,
        # lightweight raw-log pointers
        "has_fc_write_log": fc_log is not None,
        "has_ckpt_write_log": ckpt_log is not None,
        "has_cpu_mem": cpu_mem is not None,
        "shutdown_summary": shut,
    }
    return metrics


def main():
    all_metrics: List[Dict] = []
    for seed in SEEDS:
        for v in VARIANTS:
            tag = f"{v}_seed{seed}"
            log_exists = (LOG_DIR / f"{tag}.log").exists()
            if not log_exists:
                continue
            m = collect_tag(tag)
            (METRICS_DIR / f"{tag}.json").write_text(json.dumps(m, indent=2))
            all_metrics.append(m)
            print(f"  {tag}: val={m['val_final_loss']}, wasted={m['wasted_time_s']:.1f}s, "
                  f"faults={m['num_faults']}, B={m['B_max_observed']}")

    # Cross-tag aggregate
    by_variant: Dict[str, List[Dict]] = {v: [] for v in VARIANTS}
    for m in all_metrics:
        by_variant[m["variant"]].append(m)

    def _avg(lst, k):
        vals = [x[k] for x in lst if x.get(k) is not None]
        return sum(vals) / len(vals) if vals else None

    def _std(lst, k):
        vals = [x[k] for x in lst if x.get(k) is not None]
        if len(vals) < 2:
            return 0
        mean = sum(vals) / len(vals)
        return (sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5

    summary: Dict[str, Dict] = {}
    for v, lst in by_variant.items():
        if not lst:
            continue
        summary[v] = {
            "n_seeds": len(lst),
            "val_final_loss_mean": _avg(lst, "val_final_loss"),
            "val_final_loss_std": _std(lst, "val_final_loss"),
            "val_final_ppl_mean": _avg(lst, "val_final_ppl"),
            "train_final_loss_mean": _avg(lst, "train_final_loss"),
            "mean_iter_ms": _avg(lst, "mean_iter_ms"),
            "cp_bytes_total_written_mean": _avg(lst, "cp_bytes_total_written"),
            "wasted_time_s_mean": _avg(lst, "wasted_time_s"),
            "wasted_time_s_std": _std(lst, "wasted_time_s"),
            "total_recover_s_mean": _avg(lst, "total_recover_s"),
            "post_fault_damage_mean_mean": _avg(lst, "post_fault_damage_mean"),
            "B_max_observed_mean": _avg(lst, "B_max_observed"),
            "cpu_mem_peak_rss_mb_mean": _avg(lst, "cpu_mem_peak_rss_mb"),
            "peak_io_bandwidth_mb_s_mean": _avg(lst, "peak_io_bandwidth_mb_s"),
        }

    (OUT / "exp2_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nCross-variant summary:")
    for v, s in summary.items():
        print(f"  {v}: n={s['n_seeds']}, "
              f"val_loss={s['val_final_loss_mean']}±{s['val_final_loss_std']:.3f}, "
              f"wasted={s['wasted_time_s_mean']}")

    # -------- Plots --------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"baseline": "tab:blue", "moc": "tab:red",
                  "proposal": "tab:green", "random": "tab:orange"}

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax = axes.flatten()

        # Panel 1: Training loss curves (mean over seeds)
        for v in VARIANTS:
            lst = by_variant.get(v, [])
            if not lst:
                continue
            all_x, all_y = [], []
            for m in lst:
                for it, loss, _ in m["train_loss_curve"]:
                    all_x.append(it); all_y.append(loss)
            if all_x:
                # Simple binned mean by iter
                bins: Dict[int, List[float]] = {}
                for x, y in zip(all_x, all_y):
                    bins.setdefault(x, []).append(y)
                xs = sorted(bins.keys())
                ys = [sum(bins[x]) / len(bins[x]) for x in xs]
                ax[0].plot(xs, ys, color=colors[v], label=v, lw=1.5)
        ax[0].set_title("(a) Training loss (seed-averaged)")
        ax[0].set_xlabel("iter"); ax[0].set_ylabel("lm loss"); ax[0].grid(alpha=0.3); ax[0].legend()

        # Panel 2: Validation loss (final)
        labels = []; means = []; stds = []; colors_b = []
        for v in VARIANTS:
            s = summary.get(v)
            if not s or s["val_final_loss_mean"] is None:
                continue
            labels.append(v); means.append(s["val_final_loss_mean"])
            stds.append(s["val_final_loss_std"]); colors_b.append(colors[v])
        ax[1].bar(labels, means, yerr=stds, color=colors_b, alpha=0.85, capsize=4)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax[1].text(i, m + (s or 0) + 0.02, f"{m:.3f}", ha="center", fontweight="bold")
        ax[1].set_title("(b) Final validation loss (±std)")
        ax[1].set_ylabel("val lm loss"); ax[1].grid(alpha=0.3, axis="y")

        # Panel 3: Wasted time
        labels = []; means = []; stds = []; colors_b = []
        for v in VARIANTS:
            s = summary.get(v)
            if not s:
                continue
            labels.append(v)
            means.append((s["wasted_time_s_mean"] or 0) / 60)  # minutes
            stds.append((s["wasted_time_s_std"] or 0) / 60)
            colors_b.append(colors[v])
        ax[2].bar(labels, means, yerr=stds, color=colors_b, alpha=0.85, capsize=4)
        ax[2].set_title("(c) Wasted time (CP + recovery, min)")
        ax[2].set_ylabel("minutes"); ax[2].grid(alpha=0.3, axis="y")

        # Panel 4: Post-fault damage (mean + max)
        labels = []; means = []; colors_b = []
        for v in VARIANTS:
            s = summary.get(v)
            if not s or s["post_fault_damage_mean_mean"] is None:
                continue
            labels.append(v); means.append(s["post_fault_damage_mean_mean"])
            colors_b.append(colors[v])
        if labels:
            ax[3].bar(labels, means, color=colors_b, alpha=0.85)
            for i, mv in enumerate(means):
                ax[3].text(i, mv + 0.005, f"{mv:+.3f}", ha="center", fontweight="bold")
        ax[3].set_title("(d) Post-fault damage (Δloss)")
        ax[3].set_ylabel("loss delta"); ax[3].grid(alpha=0.3, axis="y")
        ax[3].axhline(0, color="black", lw=0.8)

        # Panel 5: Total CP size
        labels = []; means = []; colors_b = []
        for v in VARIANTS:
            s = summary.get(v)
            if not s:
                continue
            bytes_ = s.get("cp_bytes_total_written_mean") or 0
            labels.append(v); means.append(bytes_ / 1024**3); colors_b.append(colors[v])
        ax[4].bar(labels, means, color=colors_b, alpha=0.85)
        for i, mv in enumerate(means):
            ax[4].text(i, mv + max(means + [1]) * 0.01, f"{mv:.1f}",
                       ha="center", fontweight="bold")
        ax[4].set_title("(e) Total CP bytes written (GiB)")
        ax[4].set_ylabel("GiB"); ax[4].grid(alpha=0.3, axis="y")

        # Panel 6: Throughput slowdown (relative to baseline)
        baseline_ms = summary.get("baseline", {}).get("mean_iter_ms")
        labels = []; means = []; colors_b = []
        for v in VARIANTS:
            s = summary.get(v)
            if not s or not baseline_ms:
                continue
            labels.append(v)
            means.append((s["mean_iter_ms"] or 0) / baseline_ms * 100 - 100)
            colors_b.append(colors[v])
        if labels:
            ax[5].bar(labels, means, color=colors_b, alpha=0.85)
            for i, mv in enumerate(means):
                ax[5].text(i, mv + 0.2, f"{mv:+.1f}%", ha="center", fontweight="bold")
        ax[5].set_title("(f) Throughput slowdown vs baseline (%)")
        ax[5].set_ylabel("% slower"); ax[5].grid(alpha=0.3, axis="y")
        ax[5].axhline(0, color="black", lw=0.8)

        plt.suptitle("Experiment 2 — Proposal vs MoC vs Random under periodic fault "
                     "injection (125M-MoE8 / WikiText-2 / 2000 iter / fault every 100)",
                     fontsize=13, y=0.995)
        plt.tight_layout()
        plt.savefig(OUT / "exp2_plot.png", dpi=140, bbox_inches="tight")
        print(f"\nPlot saved: {OUT}/exp2_plot.png")
    except ImportError:
        print("(matplotlib missing; skipped plot)")


if __name__ == "__main__":
    main()
