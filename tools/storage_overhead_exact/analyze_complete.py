#!/usr/bin/env python3
"""Build an additive wall-time waterfall plus non-headline shadow diagnostics."""

import argparse
import json
import re
from pathlib import Path


def syscall_names():
    names = {}
    header = Path("/usr/include/x86_64-linux-gnu/asm/unistd_64.h")
    if header.exists():
        for line in header.read_text().splitlines():
            m = re.match(r"#define __NR_(\w+)\s+(\d+)$", line)
            if m:
                names[int(m.group(2))] = m.group(1)
    return names


def read_bpf(path):
    maps = {}
    if not path or not Path(path).exists():
        return maps
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") in ("map", "hist"):
            maps.update(obj.get("data", {}))
    return maps


def scalar(maps, key):
    value = maps.get(key, 0)
    return value if isinstance(value, (int, float)) else 0


def sum_dict(maps, key):
    value = maps.get(key, {})
    return sum(value.values()) if isinstance(value, dict) else 0


def parse_perf(path):
    out = {}
    if not path or not Path(path).exists():
        return out
    for line in Path(path).read_text(errors="replace").splitlines():
        fields = line.split(",")
        if len(fields) < 3:
            continue
        try:
            value = float(fields[0])
        except ValueError:
            continue
        out[fields[2].strip()] = {"value": value, "unit": fields[1].strip()}
    return out


def parse_pyspy(path):
    out = {"all_samples": 0, "timed_loop_gil_samples": 0,
           "timed_loop_classes": {"torch_save": 0, "path_build": 0,
                                  "buffer_flush": 0, "other_python": 0}}
    if not path or not Path(path).exists():
        return out
    timed = re.compile(r"worker \(exact_waf_replay[.]py:1(?:3[4-9]|4[0-9]|5[0-9]|6[0-9]|7[0-9])\)")
    for line in Path(path).read_text(errors="replace").splitlines():
        m = re.search(r" (\d+)$", line)
        if not m:
            continue
        count = int(m.group(1)); out["all_samples"] += count
        if not timed.search(line):
            continue
        out["timed_loop_gil_samples"] += count
        if "torch/serialization.py:" in line:
            key = "torch_save"
        elif "paths (exact_waf_replay.py:" in line:
            key = "path_build"
        elif ";flush (" in line:
            key = "buffer_flush"
        else:
            key = "other_python"
        out["timed_loop_classes"][key] += count
    n = out["timed_loop_gil_samples"]
    out["timed_loop_class_percent"] = {
        k: (100.0 * v / n if n else 0) for k, v in out["timed_loop_classes"].items()
    }
    return out


def parse_dso(path):
    out = {}
    if not path or not Path(path).exists():
        return out
    for line in Path(path).read_text(errors="replace").splitlines():
        m = re.match(r"\s+([0-9.]+)%\s+(.+?)\s*$", line)
        if m:
            out[m.group(2)] = float(m.group(1))
    return out


def classify_offcpu(maps):
    categories = {k: 0 for k in ("gil_or_futex", "journal_wait", "block_io_wait",
                                  "filesystem_lock", "filesystem_work_reschedule",
                                  "scheduler_or_other")}
    stacks = maps.get("@blocked_stack_ns", {})
    if not isinstance(stacks, dict):
        stacks = {}
    for stack, ns in stacks.items():
        s = stack.lower()
        if "futex" in s or "take_gil" in s:
            key = "gil_or_futex"
        elif "jbd2_" in s or "ext4_fc_commit" in s or "wait_transaction_locked" in s:
            key = "journal_wait"
        elif any(x in s for x in ("io_schedule", "folio_wait_writeback", "filemap_fdatawait",
                                   "submit_bio_wait", "blkdev_issue_flush", "wait_on_bit")):
            key = "block_io_wait"
        elif any(x in s for x in ("mutex_lock", "rwsem", "down_write", "queued_read_lock",
                                   "queued_write_lock", "lock_buffer")):
            key = "filesystem_lock"
        elif any(x in s for x in ("ext4_", "vfs_write", "generic_perform_write")):
            key = "filesystem_work_reschedule"
        else:
            key = "scheduler_or_other"
        categories[key] += int(ns)
    total = sum(categories.values())
    return {
        "captured_summed_s_across_ranks": total / 1e9,
        "category_summed_s_across_ranks": {k: v / 1e9 for k, v in categories.items()},
        "category_percent_of_captured": {k: (100.0 * v / total if total else 0)
                                         for k, v in categories.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True)
    p.add_argument("--bpf")
    p.add_argument("--perf-stat")
    p.add_argument("--perf-dso")
    p.add_argument("--pyspy")
    p.add_argument("--ssd-gbps", type=float, default=4.1)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    cases = {x["case"]: x for x in map(json.loads, Path(a.cases).read_text().splitlines())}
    t = lambda name: float(cases[name]["seconds"])
    payload = int(cases["full_fsync_proc8"]["payload_bytes"])
    media = payload / (a.ssd_gbps * 1e9)

    points = [
        ("device_ideal_4.1GBps", media),
        ("raw_kernel_filesystem", t("raw_preopen_proc1") - media),
        ("eight_rank_net", t("raw_preopen_proc8") - t("raw_preopen_proc1")),
        ("open_close", t("raw_actual_proc8") - t("raw_preopen_proc8")),
        ("torch_save_python", t("torch_none_proc8") - t("raw_actual_proc8")),
        ("explicit_flush", t("torch_flush_proc8") - t("torch_none_proc8")),
        ("fsync_durability", t("full_fsync_proc8") - t("torch_flush_proc8")),
    ]
    headline = t("full_fsync_proc8")
    overhead = headline - media
    waterfall = {k: v for k, v in points}
    shares = {k: 100.0 * v / overhead for k, v in points[1:]}

    maps = read_bpf(a.bpf)
    ids = syscall_names()
    counts = maps.get("@sys_count", {}) if isinstance(maps.get("@sys_count", {}), dict) else {}
    times = maps.get("@sys_ns", {}) if isinstance(maps.get("@sys_ns", {}), dict) else {}
    syscalls = {}
    for raw_id in sorted(set(counts) | set(times), key=lambda x: int(x)):
        nr = int(raw_id)
        count = int(counts.get(raw_id, 0))
        ns = int(times.get(raw_id, 0))
        syscalls[ids.get(nr, f"syscall_{nr}")] = {
            "count": count,
            "summed_s_across_ranks": ns / 1e9,
            "mean_us": ns / count / 1e3 if count else 0,
        }

    shadow = {
        "not_used_for_headline_time": True,
        "bpf_markers": scalar(maps, "@markers"),
        "syscalls": syscalls,
        "scheduler": {
            "blocked_summed_s_across_ranks": sum_dict(maps, "@blocked_ns") / 1e9,
            "runnable_summed_s_across_ranks": sum_dict(maps, "@runnable_ns") / 1e9,
            "blocked_switches": sum_dict(maps, "@blocked_count"),
            "runnable_switches": sum_dict(maps, "@runnable_count"),
        },
        "offcpu_stack_breakdown": classify_offcpu(maps),
        "kernel_lock": {
            "contention_summed_s_across_ranks": sum_dict(maps, "@lock_ns") / 1e9,
            "events": sum_dict(maps, "@lock_count"),
        },
        "ext4_sync": {
            "count": scalar(maps, "@ext4_sync_count"),
            "summed_s_across_ranks": scalar(maps, "@ext4_sync_ns") / 1e9,
        },
        "jbd2": {
            "commits": sum_dict(maps, "@jbd2_count"),
            "summed_commit_s": sum_dict(maps, "@jbd2_ns") / 1e9,
        },
        "writeback": {
            "starts": scalar(maps, "@writeback_start_count"),
            "written_events": scalar(maps, "@writeback_written_count"),
            "dirty_balance_events": scalar(maps, "@dirty_balance_count"),
            "dirty_pause_raw": scalar(maps, "@dirty_pause"),
        },
        "block": {
            "issues": scalar(maps, "@block_issue_count"),
            "issue_bytes": scalar(maps, "@block_issue_bytes"),
            "matched_completions": scalar(maps, "@block_complete_count"),
            "summed_device_service_s": scalar(maps, "@block_latency_ns") / 1e9,
        },
        "perf_stat": parse_perf(a.perf_stat),
        "on_cpu_cycle_dso_percent": parse_dso(a.perf_dso),
        "gil_holding_samples": parse_pyspy(a.pyspy),
    }

    result = {
        "scope": {"steps": cases["full_fsync_proc8"]["steps"], "payload_bytes": payload,
                  "physical_ops": cases["full_fsync_proc8"]["physical_ops"],
                  "topology": cases["full_fsync_proc8"]["topology"]},
        "headline": {"seconds": headline, "GBps": payload / headline / 1e9,
                     "device_ideal_seconds_at_4.1GBps": media,
                     "software_overhead_seconds": overhead,
                     "software_overhead_percent": 100.0 * overhead / headline},
        "additive_waterfall_seconds": waterfall,
        "software_overhead_share_percent": shares,
        "waterfall_sum_seconds": sum(waterfall.values()),
        "waterfall_closure_error_seconds": sum(waterfall.values()) - headline,
        "shadow": shadow,
        "interpretation": [
            "Waterfall deltas are ordered counterfactuals and therefore add exactly to headline wall time.",
            "Shadow syscall/off-CPU/device durations are concurrent rank sums; they explain causes but are not added to wall time.",
            "The 4.1 GB/s passthru result is supplied, not remeasured on this VM.",
        ],
    }
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
