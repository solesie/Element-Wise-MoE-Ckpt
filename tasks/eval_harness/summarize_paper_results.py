#!/usr/bin/env python3
"""Render the primary paper metrics as percentages."""

import argparse
import json
from pathlib import Path


PRIMARY = {
    "piqa": "acc_norm",
    "hellaswag": "acc_norm",
    "race": "acc",
    "mathqa": "acc_norm",
    "boolq": "acc",
    "arc_challenge": "acc_norm",
    "winogrande": "acc",
}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("results", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    with args.results.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload["results"]
    summary = {}
    print("task\tmetric\taccuracy_percent\tstderr_percent")
    for task, metric in PRIMARY.items():
        if task not in results:
            continue
        value = 100.0 * float(results[task][metric])
        stderr = results[task].get(metric + "_stderr")
        stderr_percent = None if stderr is None else 100.0 * float(stderr)
        summary[task] = {
            "metric": metric,
            "accuracy_percent": value,
            "stderr_percent": stderr_percent,
        }
        stderr_text = "" if stderr_percent is None else f"{stderr_percent:.3f}"
        print(f"{task}\t{metric}\t{value:.3f}\t{stderr_text}")
    if not summary:
        raise RuntimeError("results contain none of the paper tasks")
    summary["macro_average_percent"] = sum(
        row["accuracy_percent"] for row in summary.values() if isinstance(row, dict)
    ) / len(summary)
    print(f"macro_average\tprimary\t{summary['macro_average_percent']:.3f}\t")

    output = args.json_out or args.results.with_suffix(".percent.json")
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
