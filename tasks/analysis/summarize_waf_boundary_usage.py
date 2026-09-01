#!/usr/bin/env python3
"""Summarize per-iteration non-compaction writes from proposal WAF traces."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


STEP_RE = re.compile(br'"step":(\d+)')
WRITE_RE = re.compile(
    br'\["([ycn])","([arc])",-?\d+,-?\d+,(\d+),\d+\]'
)

RUNS = (
    ("1pct", 1.0, "ds_pretrain_gpt_350M_MoE128_10Kiter_proposal_1pct"),
    ("4pct", 4.0, "ds_pretrain_gpt_350M_MoE128_10Kiter_proposal"),
)


def parse_trace_dir(trace_dir: Path) -> dict[int, dict[str, int]]:
    totals: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    files = sorted(trace_dir.glob("waf_rank*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no WAF traces found under {trace_dir}")

    for path in files:
        with path.open("rb", buffering=1024 * 1024) as stream:
            for line in stream:
                step_match = STEP_RE.search(line)
                if step_match is None:
                    continue
                step = int(step_match.group(1))
                for kind, op, byte_text in WRITE_RE.findall(line):
                    totals[step][f"{kind.decode()}_{op.decode()}"] += int(byte_text)
    return totals


def ratio(numerator: int, denominator: int) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/waf_boundary_noncompaction_by_iter.csv"),
    )
    parser.add_argument("--last-iteration", type=int, default=10_000)
    args = parser.parse_args()

    fieldnames = (
        "run",
        "configured_boundary_pct",
        "iteration",
        "phase",
        "lfsy_append_bytes",
        "nefc_replace_bytes",
        "noncompaction_bytes",
        "lfsy_physical_pct_of_expert_full",
        "lfsy_padded_record_equiv_selected_pct",
        "noncompaction_pct_of_full_state",
        "lfcl_bytes_excluded",
        "lfsy_create_bytes_excluded",
        "expert_full_reference_bytes",
        "nonexpert_full_reference_bytes",
        "full_state_reference_bytes",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for label, boundary_pct, run_dirname in RUNS:
            totals = parse_trace_dir(args.results_root / run_dirname / "waf_trace")

            # Iteration 80 is the exact bootstrap. Its LFCL and NEFC sizes form
            # the physical full-state reference across all ranks.
            expert_full = totals[80].get("c_r", 0)
            nonexpert_full = totals[80].get("n_r", 0)
            full_state = expert_full + nonexpert_full
            if not expert_full or not nonexpert_full:
                raise RuntimeError(f"missing iteration-80 bootstrap reference for {label}")

            for iteration in range(1, args.last_iteration + 1):
                values = totals.get(iteration, {})
                lfsy_append = values.get("y_a", 0)
                nefc_replace = values.get("n_r", 0)
                lfcl_excluded = sum(
                    byte_count
                    for key, byte_count in values.items()
                    if key.startswith("c_")
                )
                lfsy_create_excluded = values.get("y_c", 0)
                noncompaction = lfsy_append + nefc_replace
                phase = "pre_warmup" if iteration < 80 else (
                    "bootstrap" if iteration == 80 else "active"
                )

                writer.writerow(
                    {
                        "run": label,
                        "configured_boundary_pct": f"{boundary_pct:.1f}",
                        "iteration": iteration,
                        "phase": phase,
                        "lfsy_append_bytes": lfsy_append,
                        "nefc_replace_bytes": nefc_replace,
                        "noncompaction_bytes": noncompaction,
                        "lfsy_physical_pct_of_expert_full": f"{ratio(lfsy_append, expert_full):.9f}",
                        # A lossy record is 16 B (idx,p,m,v), while the dense
                        # p,m,v reference is 12 B/parameter. Sector padding and
                        # 8-byte prefixes make this an upper-bound estimate.
                        "lfsy_padded_record_equiv_selected_pct": f"{ratio(lfsy_append * 12, expert_full * 16):.9f}",
                        "noncompaction_pct_of_full_state": f"{ratio(noncompaction, full_state):.9f}",
                        "lfcl_bytes_excluded": lfcl_excluded,
                        "lfsy_create_bytes_excluded": lfsy_create_excluded,
                        "expert_full_reference_bytes": expert_full,
                        "nonexpert_full_reference_bytes": nonexpert_full,
                        "full_state_reference_bytes": full_state,
                    }
                )


if __name__ == "__main__":
    main()
