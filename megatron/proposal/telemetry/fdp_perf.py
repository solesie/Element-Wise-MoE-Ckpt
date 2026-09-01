"""Low-disturbance, absolute-timeline diagnostics for the FDP backend.

The hot path only calls ``clock_gettime``/``thread_time`` and appends a small
record while ``PROPOSAL_FDP_PERF_DIAG=1``.  No per-iteration file or console
I/O is performed.  Runtime scheduler/cgroup samples are taken once at the
first active dispatch and once at shutdown.
"""

from __future__ import annotations

import json
import os
import resource
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar


_T = TypeVar("_T")
_ENABLED = os.environ.get("PROPOSAL_FDP_PERF_DIAG", "").strip().lower() in (
    "1", "true", "yes", "on",
)
_RECORDS: list[Dict[str, Any]] = []
_INIT_RUNTIME: Optional[Dict[str, Any]] = None
_START_RUNTIME: Optional[Dict[str, Any]] = None
_END_RUNTIME: Optional[Dict[str, Any]] = None
_RUNTIME_LOCK = threading.Lock()


def enabled() -> bool:
    return bool(_ENABLED)


def monotonic_raw_ns() -> int:
    clock_id = getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC)
    return int(time.clock_gettime_ns(clock_id))


def _schedstat(tid: int) -> Optional[Dict[str, int]]:
    try:
        fields = Path(f"/proc/self/task/{int(tid)}/schedstat").read_text().split()
        return {
            "run_ns": int(fields[0]),
            "runqueue_wait_ns": int(fields[1]),
            "timeslices": int(fields[2]),
        }
    except (OSError, ValueError, IndexError):
        return None


def _fdinfo(ring_fd: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if int(ring_fd) < 0:
        return out
    try:
        lines = Path(f"/proc/self/fdinfo/{int(ring_fd)}").read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in ("SqThread", "SqThreadCpu"):
            try:
                out[key] = int(value, 0)
            except ValueError:
                out[key] = value
    sq_tid = out.get("SqThread")
    if isinstance(sq_tid, int):
        try:
            fields = Path(f"/proc/{sq_tid}/schedstat").read_text().split()
            out["SqThreadSchedstat"] = {
                "run_ns": int(fields[0]),
                "runqueue_wait_ns": int(fields[1]),
                "timeslices": int(fields[2]),
            }
        except (OSError, ValueError, IndexError):
            pass
    return out


def _cgroup_sample() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        rel = "/"
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0":
                rel = parts[2]
                break
        root = Path("/sys/fs/cgroup") / rel.lstrip("/")
        for name in ("cpu.stat", "cpu.pressure", "cpuset.cpus.effective"):
            try:
                out[name] = (root / name).read_text().strip()
            except OSError:
                pass
    except OSError:
        pass
    return out


def _runtime_sample(label: str) -> Dict[str, Any]:
    from ..storage.proposal_io import fdp_perf_runtime_stats

    native_tid = threading.get_native_id()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rings = fdp_perf_runtime_stats()
    for key in ("nefc_ring", "expert_ring"):
        ring = rings.get(key)
        if not isinstance(ring, dict):
            continue
        tid = int(ring.get("completion_tid", -1) or -1)
        ring["completion_schedstat"] = _schedstat(tid)
        ring["fdinfo"] = _fdinfo(int(ring.get("ring_fd", -1) or -1))
    return {
        "label": str(label),
        "mono_raw_ns": monotonic_raw_ns(),
        "pid": int(os.getpid()),
        "native_tid": int(native_tid),
        "main_schedstat": _schedstat(native_tid),
        "process_user_cpu_s": float(usage.ru_utime),
        "process_system_cpu_s": float(usage.ru_stime),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
        "rings": rings,
        "cgroup": _cgroup_sample(),
    }


def capture_runtime_init() -> None:
    global _INIT_RUNTIME
    if not _ENABLED or _INIT_RUNTIME is not None:
        return
    with _RUNTIME_LOCK:
        if _INIT_RUNTIME is None:
            _INIT_RUNTIME = _runtime_sample("init")


def capture_runtime_start() -> None:
    global _START_RUNTIME
    if not _ENABLED or _START_RUNTIME is not None:
        return
    with _RUNTIME_LOCK:
        if _START_RUNTIME is None:
            _START_RUNTIME = _runtime_sample("start")


def capture_runtime_end() -> None:
    global _END_RUNTIME
    if not _ENABLED:
        return
    with _RUNTIME_LOCK:
        _END_RUNTIME = _runtime_sample("end")


def record_interval(
    phase: str,
    step: int,
    start_mono_raw_ns: int,
    end_mono_raw_ns: int,
    start_thread_cpu_ns: int,
    end_thread_cpu_ns: int,
    **fields: Any,
) -> None:
    if not _ENABLED:
        return
    _RECORDS.append({
        "phase": str(phase),
        "step": int(step),
        "pid": int(os.getpid()),
        "native_tid": int(threading.get_native_id()),
        "start_mono_raw_ns": int(start_mono_raw_ns),
        "end_mono_raw_ns": int(end_mono_raw_ns),
        "wall_ns": max(0, int(end_mono_raw_ns) - int(start_mono_raw_ns)),
        "thread_cpu_ns": max(
            0, int(end_thread_cpu_ns) - int(start_thread_cpu_ns),
        ),
        **fields,
    })


def measured_call(
    phase: str,
    step: int,
    fn: Callable[[], _T],
    **fields: Any,
) -> _T:
    if not _ENABLED:
        return fn()
    start_wall = monotonic_raw_ns()
    start_cpu = time.thread_time_ns()
    try:
        return fn()
    finally:
        end_wall = monotonic_raw_ns()
        end_cpu = time.thread_time_ns()
        record_interval(
            phase, step, start_wall, end_wall, start_cpu, end_cpu, **fields,
        )


def measured_mark_writer_done(
    *, step: int, writer_kind: str, site: str, fn: Callable[[], bool],
) -> bool:
    if not _ENABLED:
        return bool(fn())
    start_wall = monotonic_raw_ns()
    start_cpu = time.thread_time_ns()
    did_flush = False
    try:
        did_flush = bool(fn())
        return did_flush
    finally:
        end_wall = monotonic_raw_ns()
        end_cpu = time.thread_time_ns()
        metadata: Optional[Dict[str, Any]] = None
        if did_flush:
            from ..storage.proposal_io import fdp_perf_runtime_stats
            raw = fdp_perf_runtime_stats().get("last_metadata_commit")
            if isinstance(raw, dict) and int(raw.get("iter", -1)) == int(step):
                metadata = dict(raw)
        record_interval(
            "mark_writer_done",
            step,
            start_wall,
            end_wall,
            start_cpu,
            end_cpu,
            writer_kind=str(writer_kind),
            site=str(site),
            did_metadata_flush=bool(did_flush),
            metadata_commit=metadata,
        )


def dump(directory: Path, prefix: str = "") -> Optional[Path]:
    if not _ENABLED:
        return None
    capture_runtime_end()
    rank = int(os.environ.get("PROPOSAL_FDP_RANK", "0") or 0)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}fdp_perf_rank{rank:05d}.json"
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    payload = {
        "schema": "proposal-fdp-perf-v1",
        "rank": rank,
        "pid": int(os.getpid()),
        "clock": "CLOCK_MONOTONIC_RAW",
        "runtime_init": _INIT_RUNTIME,
        "runtime_start": _START_RUNTIME,
        "runtime_end": _END_RUNTIME,
        "records": list(_RECORDS),
    }
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(tmp, path)
    return path
