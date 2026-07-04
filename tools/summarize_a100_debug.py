#!/usr/bin/env python3
"""Summarize reduced 2-GPU proposal debug runs."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from statistics import mean


def _read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _summarize_faults(case: Path) -> str:
    path = case / "log" / "proposal_faults.jsonl"
    rows = list(_read_jsonl(path) or [])
    if not rows:
        return "faults: none"
    deltas = [
        float(r.get("post_fault_loss", 0.0)) - float(r.get("pre_fault_loss", 0.0))
        for r in rows
        if "post_fault_loss" in r and "pre_fault_loss" in r
    ]
    missing = max(int(r.get("expert_missing_count", 0)) for r in rows)
    b_lost = max(int(r.get("B_max_lost_iter", 0)) for r in rows)
    if not deltas:
        return f"faults: n={len(rows)} missing_max={missing} B_lost_max={b_lost}"
    return (
        "faults: "
        f"n={len(rows)} mean_delta={mean(deltas):+.4f} "
        f"max_abs_delta={max(abs(x) for x in deltas):.4f} "
        f"missing_max={missing} B_lost_max={b_lost}"
    )


def _summarize_direct(case: Path) -> str:
    paths = sorted((case / "log").glob("direct_feedback_rank*.jsonl"))
    if not paths:
        return "direct_feedback: none"
    by_rank = collections.Counter()
    by_rank_layer = collections.Counter()
    slot_counts = collections.defaultdict(set)
    steps = set()
    for path in paths:
        for row in _read_jsonl(path) or []:
            rank = int(row.get("rank", -1))
            steps.add(int(row.get("step", -1)))
            selected = list(row.get("selected_by_expert") or [])
            experts = list(row.get("expert_ids") or [])
            for sel, eid in zip(selected, experts):
                layer = int(eid[0])
                expert = int(eid[1])
                value = max(0, int(sel))
                by_rank[rank] += value
                by_rank_layer[(rank, layer)] += value
                slot_counts[rank].add((layer, expert))
    total = sum(by_rank.values())
    rank_parts = []
    for rank, val in sorted(by_rank.items()):
        frac = float(val) / float(total) if total > 0 else 0.0
        layers = {
            layer: by_rank_layer[(rank, layer)]
            for (r, layer) in by_rank_layer
            if r == rank
        }
        top_layers = ",".join(
            f"L{layer}:{count / max(1, val):.2f}"
            for layer, count in sorted(layers.items(), key=lambda kv: kv[1], reverse=True)[:4]
        )
        rank_parts.append(
            f"r{rank}={frac:.3f}/slots{len(slot_counts[rank])}"
            + (f"/{top_layers}" if top_layers else "")
        )
    return (
        f"direct_feedback: steps={len(steps)} total_selected={total} "
        + " ".join(rank_parts)
    )


def _summarize_lfcl(case: Path) -> str:
    paths = sorted((case / "log").glob("lfcl_fire_rank*.jsonl"))
    if not paths:
        return "lfcl: none"
    by_rank = collections.Counter()
    by_rank_layer = collections.Counter()
    for path in paths:
        for row in _read_jsonl(path) or []:
            rank = int(row.get("rank", -1))
            for eid in row.get("experts", []) or []:
                layer = int(eid[0])
                by_rank[rank] += 1
                by_rank_layer[(rank, layer)] += 1
    total = sum(by_rank.values())
    parts = []
    for rank, val in sorted(by_rank.items()):
        frac = float(val) / float(total) if total > 0 else 0.0
        layers = {
            layer: by_rank_layer[(rank, layer)]
            for (r, layer) in by_rank_layer
            if r == rank
        }
        top_layers = ",".join(
            f"L{layer}:{count}"
            for layer, count in sorted(layers.items(), key=lambda kv: kv[1], reverse=True)[:4]
        )
        parts.append(f"r{rank}={frac:.3f}" + (f"/{top_layers}" if top_layers else ""))
    return f"lfcl: total_fires={total} " + " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    cases = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not cases:
        cases = [root]
    for case in cases:
        print(f"\n== {case.name} ==")
        print(_summarize_faults(case))
        print(_summarize_direct(case))
        print(_summarize_lfcl(case))


if __name__ == "__main__":
    main()
