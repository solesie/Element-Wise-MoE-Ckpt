"""Domain-split asynchronous I/O for proposal CP.

Two single-thread workers, each owning one domain:

* ``host_writer_non_expert`` — all non-expert lossless FC writes (and
  recovery reads). Owned by ``FCManager``.
* ``host_writer_expert``     — all expert work: ``tracker_stream`` GPU
  launches (proposal mode), expert lossless FC, expert lossy FC log
  scatter/write, and expert recovery reads. Owned by ``ExpertState`` /
  ``ExpertLosslessFC`` / ``ImportanceTracker.step``.

Each worker:
  * sets ``torch.cuda.set_device(...)`` once at startup so it can launch
    GPU kernels and record / wait on ``cuda.Event`` directly,
  * processes tasks FIFO from its private queue (back-pressure via
    bounded ``queue.Queue``),
  * holds **all** state mutated by its domain — there is no
    cross-thread access to ``_pending_dispatch``, ``ExpertState`` dense
    buffers, or pinned host CP buffers.

The training thread never launches GPU work for proposal CP. Its only
proposal-related interaction is ``queue.put`` of small task closures at
the iteration boundary (``proposal_arm_post_iter_fc`` /
``proposal_release_post_iter_fc``).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


_d2h_stream: Optional["torch.cuda.Stream"] = None


def get_d2h_stream() -> Optional["torch.cuda.Stream"]:
    """Lazy-init the shared D2H stream. Both domain workers launch their
    GPU→pinned-host copies on this single stream — PCIe is the
    bottleneck and a second D2H stream would only fragment kernel
    submission order without freeing bandwidth."""
    global _d2h_stream
    if _d2h_stream is None and torch.cuda.is_available():
        _d2h_stream = torch.cuda.Stream()
    return _d2h_stream


@dataclass
class WriteRecord:
    """Mutable handoff for a single non-expert FC submission. The worker
    fills in ``done`` / timings after processing; the main thread reads
    them after ``drain_non_expert()`` returns."""

    tag: str
    path: "Optional[object]" = None
    submit_time: float = 0.0
    done: bool = False
    queue_delay_ms: float = 0.0
    write_time_ms: float = 0.0
    file_bytes: int = 0
    error: Optional[str] = None


class _DomainWorker:
    """Single-thread FIFO worker bound to one CUDA device.

    Accepts arbitrary callables and runs them on the worker thread. The
    callable is responsible for any GPU synchronisation it needs (e.g.
    ``cuda_event.synchronize()`` before reading the result of a D2H
    copy). The worker only guarantees FIFO ordering and that
    ``torch.cuda.set_device(...)`` was set on its thread before the
    first task runs.
    """

    _SENTINEL = object()

    def __init__(
        self,
        name: str,
        device: Optional[int],
        max_queue_size: int,
    ) -> None:
        self._name = name
        self._device = device
        self._queue: queue.Queue = queue.Queue(maxsize=int(max_queue_size))
        self._thread = threading.Thread(
            target=self._run, name=name, daemon=True,
        )
        self._submissions = 0
        self._completions = 0
        self._max_queue_depth = 0
        self._lock = threading.Lock()
        self._shutdown_sent = False
        self._thread.start()

    def submit(self, fn: Callable[..., None], *args: Any) -> None:
        if self._shutdown_sent:
            raise RuntimeError(f"{self._name}.submit() after shutdown()")
        self._queue.put((fn, args))
        with self._lock:
            self._submissions += 1
            depth = self._queue.qsize()
            if depth > self._max_queue_depth:
                self._max_queue_depth = depth

    def drain(self) -> None:
        self._queue.join()

    def shutdown(self) -> None:
        if self._shutdown_sent:
            return
        self._shutdown_sent = True
        self._queue.join()
        self._queue.put(self._SENTINEL)
        self._thread.join()

    def _run(self) -> None:
        if self._device is not None and torch.cuda.is_available():
            torch.cuda.set_device(self._device)
            # Move this worker's current_stream off the legacy NULL
            # stream onto an explicit NonBlocking stream. Without this,
            # torch ops invoked inside worker tasks (e.g. importance
            # tracker's torch.sort / torch.searchsorted in dispatch)
            # land on cuda_stream=0x0, triggering CUDA's NULL-stream
            # implicit sync rule and cascade-blocking other streams
            # (verified via nsys: 62 NULL kernels/run with sort family
            # from worker thread before this fix).
            self._worker_stream = torch.cuda.Stream()
            torch.cuda.set_stream(self._worker_stream)
        while True:
            item = self._queue.get()
            try:
                if item is self._SENTINEL:
                    return
                fn, args = item
                try:
                    fn(*args)
                except Exception as exc:
                    print(
                        f"[async_writer] {self._name}: task failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                with self._lock:
                    self._completions += 1
            finally:
                self._queue.task_done()

    def summary(self) -> dict:
        with self._lock:
            return {
                "name": self._name,
                "submissions": self._submissions,
                "completions": self._completions,
                "pending": self._submissions - self._completions,
                "max_queue_depth": self._max_queue_depth,
            }


class AsyncWriter:
    """Owner of the two domain workers.

    Usage::

        writer = AsyncWriter(device=0)
        writer.submit_non_expert(do_ne_fc, ne_task)
        writer.submit_expert(do_ex_fc, ex_task)
        writer.drain()           # both queues empty
        writer.shutdown()
    """

    def __init__(
        self,
        device: Optional[int] = None,
        non_expert_queue_size: int = 8,
        expert_queue_size: int = 32,
    ) -> None:
        if device is None and torch.cuda.is_available():
            device = torch.cuda.current_device()
        self.non_expert = _DomainWorker(
            "host_writer_non_expert", device, non_expert_queue_size,
        )
        self.expert = _DomainWorker(
            "host_writer_expert", device, expert_queue_size,
        )

    def submit_non_expert(self, fn: Callable[..., None], *args: Any) -> None:
        self.non_expert.submit(fn, *args)

    def submit_expert(self, fn: Callable[..., None], *args: Any) -> None:
        self.expert.submit(fn, *args)

    def drain_non_expert(self) -> None:
        self.non_expert.drain()

    def drain_expert(self) -> None:
        self.expert.drain()

    def drain(self) -> None:
        self.non_expert.drain()
        self.expert.drain()

    def shutdown(self) -> None:
        self.non_expert.shutdown()
        self.expert.shutdown()

    def summary(self) -> dict:
        return {
            "non_expert": self.non_expert.summary(),
            "expert": self.expert.summary(),
        }
