"""
AsyncWriter — background thread that takes (payload, path) work items off a
queue and calls ``torch.save``, decoupling training-loop critical path from
disk IO. 

Design choices
--------------
- **Single writer thread.** Phase 4 established that the disk is bandwidth-
  bound at ~950 MiB/s, so parallelizing writers buys nothing and only adds
  synchronization overhead. One consumer thread is enough.

- **Bounded queue.** If the training thread is submitting faster than the
  writer drains, the queue back-pressures via a blocking ``put`` on a
  full queue. In practice this only happens if `compute_per_iter < write_per_iter`
  for an extended stretch — a healthy warning that the machine is under-
  provisioned for the checkpoint strategy.

- **WriteRecord handoff.** `submit()` returns a mutable ``WriteRecord``.
  The main thread stores this record alongside its own stats and reads
  timing/byte fields *after* ``drain()`` completes. This avoids any cross-
  thread mutable state beyond the record itself.

- **Clean shutdown.** ``drain()`` waits for all submitted work; ``shutdown()``
  sends a sentinel and joins the worker thread. Both are idempotent-safe.

Error handling
--------------
If ``torch.save`` raises inside the worker, the exception is recorded on
``WriteRecord.error`` (string) and the worker continues. The main thread
can inspect records after ``drain()``. We do **not** propagate exceptions
across threads silently — any error field is reported in the stats.

Consistency guarantee
---------------------
The caller is responsible for ensuring the payload tensors are **CPU-
resident** and **not mutated after submission**. Both CheckpointManager
and FCManager already clone tensors to CPU before calling ``submit``, so
this invariant holds by construction.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

import torch


@dataclass
class WriteRecord:
    """Mutable handoff record. Main thread creates it, worker fills it in.

    Fields below ``submit_time`` are written by the worker thread only;
    read them **after** ``AsyncWriter.drain()`` returns.

    ``on_complete`` is an optional list of zero-argument callables invoked
    (on the worker thread) **after** a successful ``torch.save``. The
    intended use is per-write cleanup — for instance, "delete these older
    DC files now that the FC that supersedes them is safely on disk". If
    ``torch.save`` raises, ``on_complete`` is **not** invoked and the
    cleanup is skipped so the caller can still roll back / retry.
    """
    tag: str                   # "dc" | "fc" — for log grouping only
    path: Path
    submit_time: float         # time.perf_counter() at submit
    done: bool = False
    queue_delay_ms: float = 0.0
    write_time_ms: float = 0.0
    file_bytes: int = 0
    error: Optional[str] = None
    on_complete: List[Callable[[], None]] = field(default_factory=list)


class AsyncWriter:
    """Single-thread async writer for torch.save payloads.

    Usage::

        writer = AsyncWriter(max_queue_size=32)
        rec = writer.submit(payload, path, tag="dc")
        ...
        writer.drain()            # block until queue empty
        assert rec.done
        print(rec.write_time_ms, rec.file_bytes)
        writer.shutdown()         # terminate worker thread
    """

    _SENTINEL = object()

    def __init__(
        self,
        max_queue_size: int = 32,
        raw_queue_size: int = 1024,
    ) -> None:
        # Queue A: pickled payloads written via torch.save (non-expert DC,
        # non-expert FC, legacy per-iter expert FC in MoC mode).
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        # Queue B: raw callables for ExpertState's binary LFCL/LFSY format
        # (submit_raw). Separate worker so DC and expert writes can proceed
        # in parallel and don't share a head-of-line block.
        self._queue_raw: queue.Queue = queue.Queue(maxsize=int(raw_queue_size))
        self._thread = threading.Thread(
            target=self._run, name="AsyncWriter-torch", daemon=True,
        )
        self._thread_raw = threading.Thread(
            target=self._run_raw, name="AsyncWriter-raw", daemon=True,
        )
        self._submissions = 0
        self._completions = 0
        self._submissions_raw = 0
        self._completions_raw = 0
        self._total_write_ms = 0.0
        self._total_file_bytes = 0
        self._max_queue_depth = 0
        self._max_queue_depth_raw = 0
        self._lock = threading.Lock()  # protects counters
        self._shutdown_sent = False
        self._thread.start()
        self._thread_raw.start()

    # ------------------------------------------------------------------
    def submit(
        self,
        payload: Any,
        path: Path,
        tag: str = "dc",
        on_complete: Optional[List[Callable[[], None]]] = None,
        save_fn: Optional[Callable[[Any, Path], None]] = None,
    ) -> WriteRecord:
        """Enqueue a payload for async writing. Returns a mutable
        :class:`WriteRecord` that the worker will fill in.

        ``on_complete`` is a list of zero-argument callables invoked on
        the worker thread after a successful write (see WriteRecord
        docstring).

        ``save_fn`` lets the caller swap the default ``torch.save`` for a
        custom writer, e.g. the raw-binary ``save_binary`` used by
        non-expert FC and H_ema CPs to bypass pickle overhead. Signature
        must be ``save_fn(payload, path) -> None``; exceptions are captured
        on ``rec.error`` just like torch.save's.
        """
        if self._shutdown_sent:
            raise RuntimeError("AsyncWriter.submit() called after shutdown()")
        rec = WriteRecord(
            tag=tag,
            path=Path(path),
            submit_time=time.perf_counter(),
            on_complete=list(on_complete) if on_complete else [],
        )
        self._queue.put((payload, rec, save_fn))
        with self._lock:
            self._submissions += 1
            depth = self._queue.qsize()
            if depth > self._max_queue_depth:
                self._max_queue_depth = depth
        return rec

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._SENTINEL:
                    return
                payload, rec, save_fn = item
                start = time.perf_counter()
                rec.queue_delay_ms = (start - rec.submit_time) * 1000.0
                try:
                    if save_fn is not None:
                        save_fn(payload, rec.path)
                    else:
                        torch.save(payload, rec.path)
                    try:
                        rec.file_bytes = int(rec.path.stat().st_size)
                    except FileNotFoundError:
                        rec.file_bytes = 0
                except Exception as exc:
                    rec.error = f"{type(exc).__name__}: {exc}"
                end = time.perf_counter()
                rec.write_time_ms = (end - start) * 1000.0
                rec.done = True
                with self._lock:
                    self._completions += 1
                    self._total_write_ms += rec.write_time_ms
                    self._total_file_bytes += rec.file_bytes
                # Run post-success cleanups (ONLY on success) — e.g., unlink
                # the older DC bundles that this FC has just superseded.
                # Done on the worker thread so the main loop never blocks.
                if rec.error is None:
                    for cb in rec.on_complete:
                        try:
                            cb()
                        except Exception as cb_exc:
                            # Record the first callback failure on the rec
                            # so the main thread can inspect it, but keep
                            # processing subsequent callbacks.
                            if rec.error is None:
                                rec.error = (
                                    f"on_complete: "
                                    f"{type(cb_exc).__name__}: {cb_exc}"
                                )
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------
    def submit_raw(
        self,
        fn: Callable[..., None],
        *args: Any,
        tag: str = "raw",
    ) -> None:
        """Enqueue a raw-callable job for the dedicated raw worker thread.
        ``fn(*args)`` runs on the raw worker — used by ExpertState to
        offload its binary LFCL/LFSY writes (including ``os.fsync``) so
        they don't block the training thread.

        No WriteRecord is returned; the raw path is fire-and-forget.
        Exceptions are logged to stdout but do not propagate.
        """
        if self._shutdown_sent:
            raise RuntimeError("AsyncWriter.submit_raw() called after shutdown()")
        self._queue_raw.put((fn, args, tag))
        with self._lock:
            self._submissions_raw += 1
            depth = self._queue_raw.qsize()
            if depth > self._max_queue_depth_raw:
                self._max_queue_depth_raw = depth

    def _run_raw(self) -> None:
        while True:
            item = self._queue_raw.get()
            try:
                if item is self._SENTINEL:
                    return
                fn, args, tag = item
                try:
                    fn(*args)
                except Exception as exc:
                    print(f"[async_writer] raw job '{tag}' failed: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                with self._lock:
                    self._completions_raw += 1
            finally:
                self._queue_raw.task_done()

    # ------------------------------------------------------------------
    def drain(self) -> None:
        """Block until every submitted work item (torch + raw) has been
        processed."""
        self._queue.join()
        self._queue_raw.join()

    def drain_torch(self) -> None:
        """Block until the torch.save queue is empty. The raw queue is
        independent and is NOT waited on — used by FCManager to ensure a
        prior torch.save has finished reading its pinned CPU payload buffer
        before the main thread overwrites that buffer for the next FC."""
        self._queue.join()

    def shutdown(self) -> None:
        """Drain both queues, stop both worker threads. Idempotent."""
        if self._shutdown_sent:
            return
        self._shutdown_sent = True
        self.drain()
        self._queue.put(self._SENTINEL)
        self._queue_raw.put(self._SENTINEL)
        self._thread.join()
        self._thread_raw.join()

    # ------------------------------------------------------------------
    @property
    def submissions(self) -> int:
        with self._lock:
            return self._submissions

    @property
    def completions(self) -> int:
        with self._lock:
            return self._completions

    @property
    def pending(self) -> int:
        with self._lock:
            return self._submissions - self._completions

    @property
    def max_queue_depth(self) -> int:
        with self._lock:
            return self._max_queue_depth

    @property
    def total_write_ms(self) -> float:
        with self._lock:
            return self._total_write_ms

    @property
    def total_file_bytes(self) -> int:
        with self._lock:
            return self._total_file_bytes

    def summary(self) -> dict:
        with self._lock:
            return {
                "submissions": self._submissions,
                "completions": self._completions,
                "pending": self._submissions - self._completions,
                "max_queue_depth": self._max_queue_depth,
                "submissions_raw": self._submissions_raw,
                "completions_raw": self._completions_raw,
                "pending_raw": self._submissions_raw - self._completions_raw,
                "max_queue_depth_raw": self._max_queue_depth_raw,
                "total_write_ms": self._total_write_ms,
                "total_file_bytes": self._total_file_bytes,
            }
