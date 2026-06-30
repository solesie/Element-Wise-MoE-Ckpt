"""CUDA stream registry + checkpoint worker pool.

Two layers:

* ``CudaStreams`` — the three streams every proposal stage runs on.
  ``compute_stream`` carries all proposal compute kernels (the previous
  V1 compute split was already host-serialised, so a second
  stream bought no GPU overlap — see AGENTS.md §3.2). ``main`` is
  the training stream, referenced (not owned) so the optimizer-fence can
  ``wait_stream`` on ``d2h``.

* ``CheckpointWorker`` / ``CheckpointWorkerPool`` — checkpoint-domain
  FIFO workers, one per checkpoint domain. The training thread never
  launches proposal GPU work; it only enqueues a ``PostStepCheckpointTask``
  via the pipeline immediately before the next forward (§1.-1).

Domains (AGENTS.md §3.1):
  * ``ne_writer``        — non-expert lossless FC (GPU→pinned D2H handoff
                           and disk commit).
  * ``expert_launcher``  — per-iter selection/emission/D2H launches + segment drive.
  * ``expert_writer``    — Stage 4 LFSY/LFCL disk commit or test-DRAM commit.

Each worker sets ``torch.cuda.set_device`` once on its thread so it can
launch kernels / record / wait on ``cuda.Event`` directly, and processes
tasks FIFO from a bounded queue (back-pressure). All state mutated by a
domain lives behind that domain's single thread — no cross-thread access
to lossy-write inputs, dense recovery buffers, or pinned host buffers.
"""

from __future__ import annotations

import queue
import threading
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch


_COMPUTE_STREAM: Optional["torch.cuda.Stream"] = None
_D2H_STREAM: Optional["torch.cuda.Stream"] = None


def make_cuda_stream(priority_env: str = "", default_priority: int = 0) -> "torch.cuda.Stream":
    """Create an explicit CUDA stream with optional env-controlled priority."""

    priority = int(default_priority)
    if priority_env:
        raw = os.environ.get(priority_env, "")
        if raw:
            try:
                priority = int(raw)
            except ValueError:
                priority = int(default_priority)
    return torch.cuda.Stream(priority=priority)


def get_d2h_stream() -> Optional["torch.cuda.Stream"]:
    """Shared explicit D2H stream for GPU-to-pinned checkpoint copies."""

    global _D2H_STREAM
    if _D2H_STREAM is None and torch.cuda.is_available():
        _D2H_STREAM = make_cuda_stream("PROPOSAL_D2H_STREAM_PRIORITY")
    return _D2H_STREAM


def get_compute_stream() -> Optional["torch.cuda.Stream"]:
    """Shared explicit proposal compute stream.

    Warmup may run on the training thread, so AOS emission must not capture
    ``torch.cuda.current_stream()``. Stage 1 cutoff and Stage 2 direct
    emission share this stream.
    """

    global _COMPUTE_STREAM
    if _COMPUTE_STREAM is None and torch.cuda.is_available():
        _COMPUTE_STREAM = make_cuda_stream("PROPOSAL_COMPUTE_STREAM_PRIORITY")
    return _COMPUTE_STREAM


# ---------------------------------------------------------------------------
# Stream registry.
# ---------------------------------------------------------------------------
@dataclass
class CudaStreams:
    """The three streams every proposal stage runs on (§3.2).

    ``main`` is the training stream (external owner, referenced only).
    ``compute`` carries every per-iter proposal GPU kernel (selection/emission)
    and the anchor h_q/v_q build. ``d2h`` carries every GPU→pinned
    host copy (per-iter Stage 3, anchor snapshot/restore, non-expert FC
    payload) — a single D2H stream because PCIe, not stream count, is the
    bottleneck.
    """

    main: "torch.cuda.Stream"
    compute: "torch.cuda.Stream"
    d2h: "torch.cuda.Stream"

    @classmethod
    def create(cls) -> Optional["CudaStreams"]:
        """Build the registry for the current device, or ``None`` off-CUDA.

        Streams are lightweight handles (no VRAM allocation) — §0 R1 safe.
        """

        if not torch.cuda.is_available():
            return None
        return cls(
            main=torch.cuda.current_stream(),
            compute=get_compute_stream(),
            d2h=make_cuda_stream("PROPOSAL_D2H_STREAM_PRIORITY"),
        )


# ---------------------------------------------------------------------------
# Non-expert FC submission handoff.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Single-domain FIFO worker.
# ---------------------------------------------------------------------------
class CheckpointWorker:
    """Single-thread FIFO worker bound to one CUDA device (was ``_DomainWorker``).

    Accepts arbitrary callables and runs them on the worker thread. The
    callable owns any GPU ordering it needs. Worker tasks must prefer
    stream-side ``wait_event`` over host-side CUDA synchronisation; blocking
    CUDA API calls on these Python threads can delay the training thread's
    kernel launches through the shared CUDA context.
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
        self._submissions = 0
        self._completions = 0
        self._max_queue_depth = 0
        self._lock = threading.Lock()
        self._shutdown_sent = False
        self._domain_stream: Optional["torch.cuda.Stream"] = None
        self._thread = threading.Thread(
            target=self._run, name=name, daemon=True,
        )
        self._thread.start()

    @property
    def name(self) -> str:
        return self._name

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
            # Move this worker's current_stream off the legacy NULL stream
            # onto an explicit non-blocking stream. Without this, torch ops
            # invoked inside worker tasks land on cuda_stream=0x0, triggering
            # CUDA's NULL-stream implicit-sync rule and cascade-blocking
            # other streams.
            self._domain_stream = make_cuda_stream("PROPOSAL_WORKER_STREAM_PRIORITY")
            torch.cuda.set_stream(self._domain_stream)
        while True:
            item = self._queue.get()
            try:
                if item is self._SENTINEL:
                    return
                fn, args = item
                try:
                    fn(*args)
                except Exception as exc:
                    import traceback as _tb
                    print(
                        f"[checkpoint_worker] {self._name}: task failed: "
                        f"{type(exc).__name__}: {exc}\n"
                        f"{_tb.format_exc()}",
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


# ---------------------------------------------------------------------------
# Checkpoint-domain worker pool.
# ---------------------------------------------------------------------------
class CheckpointWorkerPool:
    """Checkpoint-domain workers, one clear responsibility each
    See module docstring / §3.1.

    * ``ne_writer``       — non-expert lossless FC
                            (GPU→pinned D2H + file write/fsync).
    * ``expert_launcher`` — per-iter selection/emission/D2H launches; forwards the
                            heavy commit to ``expert_writer`` so the next
                            iter's score can launch on time.
    * ``expert_writer``   — Stage 4 commit (lossy log append + LFCL file
                            write in disk mode, or DRAM scatter).
    """

    def __init__(
        self,
        device: Optional[int] = None,
        non_expert_queue_size: int = 1,
        expert_queue_size: int = 1,
        expert_writer_queue_size: int = 2,
    ) -> None:
        if device is None and torch.cuda.is_available():
            device = torch.cuda.current_device()
        self.ne_writer = CheckpointWorker(
            "ne_writer", device, non_expert_queue_size,
        )
        self.expert_launcher = CheckpointWorker(
            "expert_launcher", device, expert_queue_size,
        )
        self.expert_writer = CheckpointWorker(
            "expert_writer", device, expert_writer_queue_size,
        )

    def submit_non_expert(self, fn: Callable[..., None], *args: Any) -> None:
        self.ne_writer.submit(fn, *args)

    def submit_expert(self, fn: Callable[..., None], *args: Any) -> None:
        self.expert_launcher.submit(fn, *args)

    def submit_expert_write(self, fn: Callable[..., None], *args: Any) -> None:
        self.expert_writer.submit(fn, *args)

    def drain_non_expert(self) -> None:
        self.ne_writer.drain()

    def drain_expert(self) -> None:
        self.expert_launcher.drain()

    def drain_expert_write(self) -> None:
        self.expert_writer.drain()

    def drain(self) -> None:
        self.ne_writer.drain()
        self.expert_launcher.drain()
        self.expert_writer.drain()

    def shutdown(self) -> None:
        self.ne_writer.shutdown()
        self.expert_launcher.shutdown()
        self.expert_writer.shutdown()

    def summary(self) -> Dict[str, dict]:
        return {
            "ne_writer": self.ne_writer.summary(),
            "expert_launcher": self.expert_launcher.summary(),
            "expert_writer": self.expert_writer.summary(),
        }


__all__ = [
    "CudaStreams",
    "make_cuda_stream",
    "get_compute_stream",
    "get_d2h_stream",
    "WriteRecord",
    "CheckpointWorker",
    "CheckpointWorkerPool",
]
