"""Double-buffered GPU→pinned segment D2H for proposal CP.

Architectural contract (AGENTS.md): the expert lossy AOS path's
GPU→pinned D2H flows through this primitive.

  * Caller supplies a ``producer(seg_view, byte_off, n)`` callable. The
    pipeline invokes it on ``compute_stream`` with a contiguous uint8
    GPU slice of length ``n``; the producer fills that slice with the
    payload bytes ``[byte_off, byte_off + n)``.
  * The pipeline ping-pongs between ``num_buffers`` (default 2) GPU
    staging buffers so the producer can build segment k+1 while the
    d2h_stream is still draining segment k.
  * Caller provides a single pre-allocated pinned destination of size
    ≥ max(byte_off + n) across the segments (no per-iter pinned alloc).
  * Returns a ``cuda.Event`` recorded on ``d2h_stream`` after the final
    segment's D2H. With a single pinned AOS slab, this is also the safe
    optimizer-fence event: the next iter cannot reuse the slab before the
    copy is complete.

Segments are caller-supplied by the AOS layout. Each segment has its own
``(byte_offset, byte_length)`` tuple; the pipeline does not assume a
fixed segment size and only requires that every segment fit within
``segment_bytes`` (the size of the GPU staging buffers allocated at
construction).

GPU staging is allocated **once** at construction and reused. Persistent
VRAM footprint = ``segment_bytes × num_buffers`` per pipeline.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple
import os

import torch


DEFAULT_SEGMENT_BYTES: int = 125 * 1024 * 1024  # 125 MiB


class SegmentD2H:
    """Owns N GPU staging buffers and drives segmented GPU→pinned D2H."""

    def __init__(
        self,
        *,
        device: int,
        compute_stream: "torch.cuda.Stream",
        d2h_stream: "torch.cuda.Stream",
        segment_bytes: int = DEFAULT_SEGMENT_BYTES,
        num_buffers: int = 2,
    ) -> None:
        if segment_bytes <= 0:
            raise ValueError(f"segment_bytes must be positive: {segment_bytes}")
        if num_buffers < 1:
            raise ValueError(f"num_buffers must be >= 1: {num_buffers}")
        self._compute_stream = compute_stream
        self._d2h_stream = d2h_stream
        self._segment_bytes = int(segment_bytes)
        self._gpu_bufs: List[torch.Tensor] = [
            torch.empty(self._segment_bytes, dtype=torch.uint8, device=device)
            for _ in range(num_buffers)
        ]
        # Lazily created on first use; construction can happen before the
        # caller has set the final CUDA device.
        self._prod_done: List[Optional[torch.cuda.Event]] = [None] * num_buffers
        self._d2h_done: List[Optional[torch.cuda.Event]] = [None] * num_buffers
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_warmup_done = False
        self._graph_signature: Optional[Tuple[Tuple[Tuple[int, int], ...], int, bool]] = None
        self._graph_last_event: Optional[torch.cuda.Event] = None
        self._graph_last_producer_event: Optional[torch.cuda.Event] = None
        self._graph_external_d2h_done: Optional[torch.cuda.Event] = None
        self._graph_external_producer_done: Optional[torch.cuda.Event] = None

    @property
    def cuda_graph_ready(self) -> bool:
        return self._graph is not None

    @property
    def segment_bytes(self) -> int:
        return self._segment_bytes

    @property
    def num_buffers(self) -> int:
        return len(self._gpu_bufs)

    def run(
        self,
        producer: Callable[[torch.Tensor, int, int], None],
        segments: Iterable[Tuple[int, int]],
        dst: torch.Tensor,
        *,
        copy_to_host: bool = True,
    ) -> Tuple[Optional["torch.cuda.Event"], Optional["torch.cuda.Event"]]:
        segment_list = [(int(o), int(n)) for o, n in segments if int(n) > 0]
        if not segment_list:
            return None, None
        return self._run_graph(
            producer,
            segment_list,
            dst,
            copy_to_host=bool(copy_to_host),
        )

    def _validate_run_inputs(
        self,
        segments: Iterable[Tuple[int, int]],
        dst: torch.Tensor,
        *,
        copy_to_host: bool,
    ) -> List[Tuple[int, int]]:
        if dst.device.type != "cpu":
            raise ValueError("SegmentD2H.run: dst must be a CPU tensor")
        if dst.dtype != torch.uint8 or not dst.is_contiguous():
            raise ValueError("SegmentD2H.run: dst must be contiguous uint8")
        if not dst.is_pinned():
            raise ValueError("SegmentD2H.run: dst must be pinned")

        segment_list = [(int(o), int(n)) for o, n in segments if int(n) > 0]
        for byte_off, n in segment_list:
            if n > self._segment_bytes:
                raise ValueError(
                    f"SegmentD2H.run: segment size {n} exceeds "
                    f"staging capacity {self._segment_bytes}"
                )
            if copy_to_host and byte_off + n > dst.numel():
                raise ValueError(
                    f"SegmentD2H.run: segment ({byte_off}, {n}) "
                    f"exceeds dst ({dst.numel()} bytes)"
                )
        return segment_list

    def _graph_run_signature(
        self,
        segments: List[Tuple[int, int]],
        dst: torch.Tensor,
        *,
        copy_to_host: bool,
    ) -> Tuple[Tuple[Tuple[int, int], ...], int, bool]:
        return tuple(segments), int(dst.data_ptr()), bool(copy_to_host)

    def _run_graph(
        self,
        producer: Callable[[torch.Tensor, int, int], None],
        segments: List[Tuple[int, int]],
        dst: torch.Tensor,
        *,
        copy_to_host: bool,
    ):
        segment_list = self._validate_run_inputs(
            segments, dst, copy_to_host=bool(copy_to_host),
        )
        signature = self._graph_run_signature(
            segment_list, dst, copy_to_host=bool(copy_to_host),
        )
        if self._graph is not None:
            if signature != self._graph_signature:
                raise RuntimeError(
                    "SegmentD2H CUDA graph signature changed; fixed graph "
                    "topology/address contract was violated"
                )
            with torch.cuda.stream(self._compute_stream):
                self._graph.replay()
            return self._graph_last_event, self._graph_last_producer_event

        if not self._graph_warmup_done:
            self._graph_warmup_done = True
            return self._enqueue_segment_commands(
                producer,
                segment_list,
                dst,
                copy_to_host=bool(copy_to_host),
            )

        self._prod_done = [None] * len(self._gpu_bufs)
        self._d2h_done = [None] * len(self._gpu_bufs)
        self._graph_external_d2h_done = torch.cuda.Event(external=True)
        self._graph_external_producer_done = torch.cuda.Event(external=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            stream=self._compute_stream,
            capture_error_mode="thread_local",
        ):
            last_event, last_producer_event = self._enqueue_segment_commands(
                producer,
                segment_list,
                dst,
                copy_to_host=bool(copy_to_host),
                join_d2h_on_compute=True,
                external_done_events=(
                    self._graph_external_d2h_done,
                    self._graph_external_producer_done,
                ),
            )
        self._graph = graph
        self._graph_signature = signature
        self._graph_last_event = last_event
        self._graph_last_producer_event = last_producer_event
        with torch.cuda.stream(self._compute_stream):
            self._graph.replay()
        return self._graph_last_event, self._graph_last_producer_event

    def _enqueue_segment_commands(
        self,
        producer: Callable[[torch.Tensor, int, int], None],
        segments: Iterable[Tuple[int, int]],
        dst: torch.Tensor,
        *,
        copy_to_host: bool = True,
        join_d2h_on_compute: bool = False,
        external_done_events: Optional[Tuple["torch.cuda.Event", "torch.cuda.Event"]] = None,
    ) -> Optional["torch.cuda.Event"]:
        """Stream caller-supplied segments from GPU through ``producer``
        into ``dst``.

        ``segments`` is an iterable of ``(byte_offset, byte_length)``
        tuples; each entry's ``byte_length`` must satisfy
        ``0 < byte_length <= segment_bytes`` (the construction-time
        staging buffer size). Segments may have variable size.

        ``producer(seg_view, byte_off, n)`` is invoked on
        ``compute_stream``. ``seg_view`` is a contiguous uint8 GPU slice
        of length ``n``; the producer fills it with payload bytes
        ``[byte_off, byte_off + n)``.

        ``dst`` must be a contiguous, pinned, uint8 CPU tensor with
        ``numel() >= byte_off + n`` for every segment when
        ``copy_to_host`` is true. ``dst[byte_off : byte_off + n]`` is
        overwritten per segment.

        Returns ``(d2h_done, producer_done)``. ``d2h_done`` is recorded on
        ``d2h_stream`` after the final D2H, or equals ``producer_done`` when
        ``copy_to_host`` is false. ``producer_done`` is recorded on
        ``compute_stream`` after the final producer kernel has materialized
        records into staging. Optimizer fences and writer leases wait for
        ``d2h_done`` in the production contract.
        Returns ``(None, None)`` if ``segments`` is empty.
        """
        profile = os.environ.get("PROPOSAL_SEGMENT_D2H_PROFILE", "") in (
            "1", "true", "TRUE",
        )
        if profile:
            import time
            t_run0 = time.perf_counter()
            backpressure_ms = 0.0
            producer_ms = 0.0
            d2h_enqueue_ms = 0.0
            copy_cmds = 0
            copied_bytes = 0
        else:
            time = None
            t_run0 = 0.0
            backpressure_ms = producer_ms = d2h_enqueue_ms = 0.0
            copy_cmds = copied_bytes = 0
        segment_list = self._validate_run_inputs(
            segments, dst, copy_to_host=bool(copy_to_host),
        )
        seg = self._segment_bytes
        d2h_chunk = max(1 * 1024 * 1024, int(seg))
        nb = len(self._gpu_bufs)
        buf = 0
        last_event: Optional[torch.cuda.Event] = None
        last_join_event: Optional[torch.cuda.Event] = None
        last_producer_event: Optional[torch.cuda.Event] = None
        last_index = len(segment_list) - 1
        external_d2h_done = external_done_events[0] if external_done_events else None
        external_producer_done = external_done_events[1] if external_done_events else None
        for seg_index, (byte_off, n) in enumerate(segment_list):
            # Back-pressure: producer must not overwrite a staging buffer
            # whose previous d2h is still in flight.
            if self._d2h_done[buf] is not None:
                t0 = time.perf_counter() if profile and time is not None else 0.0
                self._compute_stream.wait_event(self._d2h_done[buf])
                if profile and time is not None:
                    backpressure_ms += (time.perf_counter() - t0) * 1000.0
            seg_view = self._gpu_bufs[buf].narrow(0, 0, n)
            with torch.cuda.stream(self._compute_stream):
                t0 = time.perf_counter() if profile and time is not None else 0.0
                producer(seg_view, byte_off, n)
                if profile and time is not None:
                    producer_ms += (time.perf_counter() - t0) * 1000.0
            if self._prod_done[buf] is None:
                self._prod_done[buf] = torch.cuda.Event()
            self._prod_done[buf].record(self._compute_stream)
            producer_event_for_wait = self._prod_done[buf]
            last_producer_event = producer_event_for_wait
            if seg_index == last_index and external_producer_done is not None:
                external_producer_done.record(self._compute_stream)
                last_producer_event = external_producer_done
            if not copy_to_host:
                last_event = producer_event_for_wait
                last_join_event = producer_event_for_wait
                if seg_index == last_index and external_d2h_done is not None:
                    external_d2h_done.record(self._compute_stream)
                    last_event = external_d2h_done
                buf = (buf + 1) % nb
                continue
            self._d2h_stream.wait_event(producer_event_for_wait)
            with torch.cuda.stream(self._d2h_stream):
                t0 = time.perf_counter() if profile and time is not None else 0.0
                chunk_off = 0
                while chunk_off < n:
                    chunk_n = min(d2h_chunk, n - chunk_off)
                    dst.narrow(0, byte_off + chunk_off, chunk_n).copy_(
                        seg_view.narrow(0, chunk_off, chunk_n),
                        non_blocking=True,
                    )
                    copy_cmds += 1
                    copied_bytes += int(chunk_n)
                    chunk_off += chunk_n
                if profile and time is not None:
                    d2h_enqueue_ms += (time.perf_counter() - t0) * 1000.0
            if seg_index == last_index and external_d2h_done is not None:
                external_d2h_done.record(self._d2h_stream)
                last_event = external_d2h_done
            if self._d2h_done[buf] is None:
                self._d2h_done[buf] = torch.cuda.Event()
            self._d2h_done[buf].record(self._d2h_stream)
            if last_event is None:
                last_event = self._d2h_done[buf]
            last_join_event = self._d2h_done[buf]
            buf = (buf + 1) % nb
        if bool(join_d2h_on_compute) and copy_to_host and last_join_event is not None:
            self._compute_stream.wait_event(last_join_event)
        if profile and time is not None:
            total_ms = (time.perf_counter() - t_run0) * 1000.0
            total_bytes = sum(int(n) for _o, n in segment_list)
            print(
                f"[proposal] segment_d2h_profile: "
                f"segments={len(segment_list)} segment_bytes={int(self._segment_bytes)} "
                f"bytes={int(total_bytes)} copy_to_host={int(bool(copy_to_host))} "
                f"copy_cmds={int(copy_cmds)} copied_bytes={int(copied_bytes)} "
                f"backpressure={backpressure_ms:.1f}ms "
                f"producer_enqueue={producer_ms:.1f}ms "
                f"d2h_enqueue={d2h_enqueue_ms:.1f}ms "
                f"total_host={total_ms:.1f}ms",
                flush=True,
            )
        return last_event, last_producer_event

    def _ensure_events(self, *, replace: bool = False) -> None:
        for i in range(len(self._gpu_bufs)):
            if replace or self._prod_done[i] is None:
                self._prod_done[i] = torch.cuda.Event()
            if replace or self._d2h_done[i] is None:
                self._d2h_done[i] = torch.cuda.Event()
