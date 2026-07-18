"""Compact, rank-local write trace for offline SSD WAF experiments.

One row describes one logical storage write, not one NVMe SQE.  The training
path only timestamps a write batch and enqueues immutable tuples; JSON encoding
and trace-file I/O run on a daemon thread.
"""

from __future__ import annotations

import atexit
import itertools
import json
import os
import queue
import socket
import threading
import time
from typing import Iterable, NamedTuple, Optional, Tuple


TRACE_DIR_ENV = "PROPOSAL_IO_BACKLOG_TRACE_DIR"
_SCHEMA = "proposal-waf-write-v1"
_QUEUE_MAX_BATCHES = 256
_TRACE_INSTANCE_SEQ = itertools.count()
_STOP = object()
_KIND_CODE = {"lfsy": "y", "lfcl": "c", "nefc": "n"}
_OP_CODE = {"append": "a", "replace": "r", "create": "c"}
_WRITE_FIELDS = (
    "kind", "op", "layer", "global_expert",
    "io_bytes", "offset",
)


class WriteObject(NamedTuple):
    """One logical host write used to reconstruct the SSD workload."""

    kind: str
    op: str
    layer: int = -1
    local_expert: int = -1
    io_bytes: int = 0
    offset: int = 0


class _WriteBatch(NamedTuple):
    seq: int
    mono_ns: int
    step: int
    writes: Tuple[WriteObject, ...]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return int(default)


def _topology() -> dict:
    ep_rank = _env_int("PROPOSAL_FDP_EP_RANK", 0)
    ep_size = max(1, _env_int("PROPOSAL_FDP_EP_SIZE", 1))
    experts = max(0, _env_int("PROPOSAL_FDP_EXPERTS_PER_LAYER", 0))
    local_experts = experts // ep_size if experts and experts % ep_size == 0 else 0
    ids_global = os.environ.get(
        "PROPOSAL_FDP_EXPERT_IDS_GLOBAL", "",
    ).strip().lower() in ("1", "true", "yes", "on")
    expert_base = 0 if ids_global else _env_int(
        "PROPOSAL_FDP_EXPERT_ID_BASE", ep_rank * local_experts,
    )
    return {
        "rank": _env_int("RANK", _env_int("PROPOSAL_FDP_RANK", -1)),
        "local_rank": _env_int("LOCAL_RANK", -1),
        "world_size": _env_int(
            "WORLD_SIZE", _env_int("PROPOSAL_FDP_WORLD_SIZE", 1),
        ),
        "dp_rank": _env_int("PROPOSAL_FDP_DP_RANK", 0),
        "dp_size": max(1, _env_int("PROPOSAL_FDP_DP_SIZE", 1)),
        "pp_rank": _env_int("PROPOSAL_FDP_PP_RANK", 0),
        "pp_size": max(1, _env_int("PROPOSAL_FDP_PP_SIZE", 1)),
        "ep_rank": ep_rank,
        "ep_size": ep_size,
        "experts_per_layer": experts,
        "global_expert_base": expert_base,
        "sector_bytes": 4096,
        "max_io_bytes": max(0, _env_int("PROPOSAL_FDP_MAX_IO_KB", 256)) * 1024,
        "nonexpert_qd": max(1, _env_int("PROPOSAL_FDP_NEFC_QD", 64)),
        "expert_qd": max(1, _env_int("PROPOSAL_FDP_EXPERT_QD", 64)),
        "source_cp_target": os.environ.get("PROPOSAL_CP_TARGET", "").strip(),
        "source_io_backend": os.environ.get(
            "PROPOSAL_IO_BACKEND", "file",
        ).strip() or "file",
    }


def capture_timestamp() -> int:
    return time.monotonic_ns()


def io_backlog_enabled() -> bool:
    return bool(os.environ.get(TRACE_DIR_ENV, "").strip())


def _json_line(record: dict) -> bytes:
    return (json.dumps(record, separators=(",", ":")) + "\n").encode()


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short WAF trace write")
        view = view[written:]


class _Tracer:
    def __init__(self, trace_dir: str) -> None:
        self.pid = os.getpid()
        self.wall0 = time.time_ns()
        self.mono0 = time.monotonic_ns()
        self.topology = _topology()
        rank = int(self.topology["rank"])
        rank_label = f"{rank:05d}" if rank >= 0 else "unknown"
        instance = next(_TRACE_INSTANCE_SEQ)
        self.path = os.path.join(
            os.path.abspath(trace_dir),
            f"waf_rank{rank_label}_pid{self.pid}_{self.wall0}_{instance}.jsonl",
        )
        self.queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX_BATCHES)
        self.seq = itertools.count()
        self.lock = threading.Lock()
        self.closed = False
        self.error = ""
        self.producer_dropped = 0
        self.thread = threading.Thread(
            target=self._run,
            name=f"proposal-waf-trace-rank{rank_label}",
            daemon=True,
        )
        self.thread.start()

    def submit(
        self,
        writes: Tuple[WriteObject, ...],
        *,
        step: int,
        timestamp: int,
    ) -> bool:
        with self.lock:
            if self.closed or self.error:
                return False
            try:
                self.queue.put_nowait(_WriteBatch(
                    next(self.seq), int(timestamp), int(step), writes,
                ))
            except queue.Full:
                self.producer_dropped += 1
                return False
        return True

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.queue.put(_STOP)
        if threading.current_thread() is not self.thread:
            self.thread.join()

    def _fail(self, exc: BaseException) -> None:
        if self.error:
            return
        self.error = f"{type(exc).__name__}: {exc}"
        try:
            os.write(2, f"[proposal-waf-trace] {self.error}\n".encode())
        except Exception:
            pass

    def _metadata(self) -> dict:
        record = {
            "schema": _SCHEMA,
            "record": "metadata",
            "host": socket.gethostname(),
            "pid": self.pid,
            "created_wall_ns": self.wall0,
            "created_mono_ns": self.mono0,
            "timestamp": (
                "monotonic CPU time when the listed logical writes became "
                "storage-eligible; wall time is created_wall_ns + "
                "(mono_ns - created_mono_ns)"
            ),
            "granularity": "logical write object, never individual SQE",
            "replay": (
                "reissue logical operations through the FDP backend; it "
                "derives SQE splits and metadata writes"
            ),
            "write_fields": list(_WRITE_FIELDS),
            "kind_codes": {value: key for key, value in _KIND_CODE.items()},
            "op_codes": {value: key for key, value in _OP_CODE.items()},
            "queue_max_batches": _QUEUE_MAX_BATCHES,
        }
        record.update(self.topology)
        return record

    def _line(self, batch: _WriteBatch) -> bytes:
        expert_base = int(self.topology["global_expert_base"])
        rows = []
        for write in batch.writes:
            kind = _KIND_CODE[write.kind]
            op = _OP_CODE[write.op]
            expert = (
                expert_base + int(write.local_expert)
                if int(write.local_expert) >= 0 else -1
            )
            rows.append([
                kind, op, int(write.layer), expert,
                int(write.io_bytes), int(write.offset),
            ])
        return _json_line({
            "schema": _SCHEMA,
            "record": "write_batch",
            "mono_ns": batch.mono_ns,
            "batch_seq": batch.seq,
            "step": batch.step,
            "writes": rows,
        })

    def _run(self) -> None:
        fd: Optional[int] = None
        batches = writes = dropped = 0
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            _write_all(fd, _json_line(self._metadata()))
        except Exception as exc:
            self._fail(exc)

        stop = False
        while not stop:
            item = self.queue.get()
            if item is _STOP:
                break
            pending = [item]
            while len(pending) < 64:
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    break
                if item is _STOP:
                    stop = True
                    break
                pending.append(item)
            if fd is None or self.error:
                dropped += len(pending)
                continue
            try:
                _write_all(fd, b"".join(self._line(batch) for batch in pending))
                batches += len(pending)
                writes += sum(len(batch.writes) for batch in pending)
            except Exception as exc:
                dropped += len(pending)
                self._fail(exc)

        if fd is not None:
            try:
                _write_all(fd, _json_line({
                    "schema": _SCHEMA,
                    "record": "footer",
                    "status": "error" if self.error else "ok",
                    "batches": batches,
                    "writes": writes,
                    "dropped_batches": dropped + self.producer_dropped,
                    "error": self.error,
                }))
                os.fsync(fd)
            except Exception as exc:
                self._fail(exc)
            finally:
                os.close(fd)


_TRACER: Optional[_Tracer] = None
_TRACER_LOCK = threading.Lock()


def _get_tracer() -> Optional[_Tracer]:
    global _TRACER
    if not io_backlog_enabled():
        return None
    pid = os.getpid()
    if _TRACER is not None and _TRACER.pid == pid:
        return _TRACER
    with _TRACER_LOCK:
        if _TRACER is not None and _TRACER.pid == pid:
            return _TRACER
        trace_dir = os.environ.get(TRACE_DIR_ENV, "").strip()
        if not trace_dir:
            return None
        try:
            _TRACER = _Tracer(trace_dir)
        except Exception as exc:
            try:
                os.write(2, f"[proposal-waf-trace] init failed: {exc}\n".encode())
            except Exception:
                pass
            _TRACER = None
        return _TRACER


def record_write_batch(
    writes: Iterable[WriteObject],
    *,
    step: int,
    timestamp: Optional[int] = None,
) -> bool:
    """Queue one logical write batch; disabled tracing does not consume it."""

    if not io_backlog_enabled():
        return False
    try:
        values = tuple(writes)
        if not values:
            return False
        tracer = _get_tracer()
        return bool(tracer and tracer.submit(
            values,
            step=int(step),
            timestamp=capture_timestamp() if timestamp is None else timestamp,
        ))
    except Exception as exc:
        if _TRACER is not None:
            _TRACER._fail(exc)
        return False


def close_io_backlog_trace() -> None:
    global _TRACER
    with _TRACER_LOCK:
        tracer, _TRACER = _TRACER, None
    if tracer is not None and tracer.pid == os.getpid():
        tracer.close()


def _reset_io_backlog_trace_for_tests() -> None:
    close_io_backlog_trace()


atexit.register(close_io_backlog_trace)


__all__ = [
    "TRACE_DIR_ENV", "WriteObject", "capture_timestamp",
    "close_io_backlog_trace", "io_backlog_enabled", "record_write_batch",
    "_reset_io_backlog_trace_for_tests",
]
