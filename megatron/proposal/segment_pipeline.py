"""Double-buffered GPU→pinned D2H pipeline for proposal CP.

Architectural contract (CLAUDE.md): the expert lossy AOS path's
GPU→pinned D2H flows through this primitive.

  * Caller supplies a ``producer(seg_view, byte_off, n)`` callable. The
    pipeline invokes it on ``worker_stream`` with a contiguous uint8
    GPU slice of length ``n``; the producer fills that slice with the
    payload bytes ``[byte_off, byte_off + n)``.
  * The pipeline ping-pongs between ``num_buffers`` (default 2) GPU
    staging buffers so the producer can build segment k+1 while the
    d2h_stream is still draining segment k.
  * Caller provides a single pre-allocated pinned destination of size
    ≥ max(byte_off + n) across the segments (no per-iter pinned alloc).
  * Returns a ``cuda.Event`` recorded on ``d2h_stream`` after the final
    segment's H2D — the worker thread is expected to
    ``event.synchronize()`` before reading ``dst`` on host.

Segments are caller-supplied — the AOS layout is partitioned at
expert-blob boundaries (see ``gpu_aos.cut_into_blob_aligned_segments``)
so records never straddle. Each segment has its own
``(byte_offset, byte_length)`` tuple; the pipeline does not assume a
fixed segment size and only requires that every segment fit within
``segment_bytes`` (the size of the GPU staging buffers allocated at
construction).

GPU staging is allocated **once** at construction and reused. Persistent
VRAM footprint = ``segment_bytes × num_buffers`` per pipeline.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

import torch


DEFAULT_SEGMENT_BYTES: int = 128 * 1024 * 1024  # 128 MiB


class SegmentPipeline:
    """Owns N GPU staging buffers and drives segmented GPU→pinned D2H."""

    def __init__(
        self,
        *,
        device: int,
        worker_stream: "torch.cuda.Stream",
        d2h_stream: "torch.cuda.Stream",
        segment_bytes: int = DEFAULT_SEGMENT_BYTES,
        num_buffers: int = 2,
    ) -> None:
        if segment_bytes <= 0:
            raise ValueError(f"segment_bytes must be positive: {segment_bytes}")
        if num_buffers < 1:
            raise ValueError(f"num_buffers must be >= 1: {num_buffers}")
        self._worker_stream = worker_stream
        self._d2h_stream = d2h_stream
        self._segment_bytes = int(segment_bytes)
        self._gpu_bufs: List[torch.Tensor] = [
            torch.empty(self._segment_bytes, dtype=torch.uint8, device=device)
            for _ in range(num_buffers)
        ]
        # Lazily created on first use — keeps construction cheap and
        # avoids cuda-event handles being captured before set_device().
        self._prod_done: List[Optional[torch.cuda.Event]] = [None] * num_buffers
        self._d2h_done: List[Optional[torch.cuda.Event]] = [None] * num_buffers

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
    ) -> Optional["torch.cuda.Event"]:
        """Stream caller-supplied segments from GPU through ``producer``
        into ``dst``.

        ``segments`` is an iterable of ``(byte_offset, byte_length)``
        tuples; each entry's ``byte_length`` must satisfy
        ``0 < byte_length <= segment_bytes`` (the construction-time
        staging buffer size). Segments may have variable size — the
        expert lossy path partitions at expert-blob boundaries (see
        ``gpu_aos.cut_into_blob_aligned_segments``).

        ``producer(seg_view, byte_off, n)`` is invoked on
        ``worker_stream``. ``seg_view`` is a contiguous uint8 GPU slice
        of length ``n``; the producer fills it with payload bytes
        ``[byte_off, byte_off + n)``.

        ``dst`` must be a contiguous, pinned, uint8 CPU tensor with
        ``numel() >= byte_off + n`` for every segment.
        ``dst[byte_off : byte_off + n]`` is overwritten per segment.

        Returns the ``cuda.Event`` recorded on ``d2h_stream`` after the
        final segment's H2D, or ``None`` if ``segments`` is empty.
        """
        if dst.device.type != "cpu":
            raise ValueError("SegmentPipeline.run: dst must be a CPU tensor")
        if dst.dtype != torch.uint8 or not dst.is_contiguous():
            raise ValueError("SegmentPipeline.run: dst must be contiguous uint8")
        if not dst.is_pinned():
            raise ValueError("SegmentPipeline.run: dst must be pinned")

        seg = self._segment_bytes
        nb = len(self._gpu_bufs)
        buf = 0
        last_event: Optional[torch.cuda.Event] = None
        for byte_off, n in segments:
            if n <= 0:
                continue
            if n > seg:
                raise ValueError(
                    f"SegmentPipeline.run: segment size {n} exceeds "
                    f"staging capacity {seg}"
                )
            if byte_off + n > dst.numel():
                raise ValueError(
                    f"SegmentPipeline.run: segment ({byte_off}, {n}) "
                    f"exceeds dst ({dst.numel()} bytes)"
                )
            # Back-pressure: producer must not overwrite a staging buffer
            # whose previous d2h is still in flight.
            if self._d2h_done[buf] is not None:
                self._worker_stream.wait_event(self._d2h_done[buf])
            seg_view = self._gpu_bufs[buf].narrow(0, 0, n)
            with torch.cuda.stream(self._worker_stream):
                producer(seg_view, byte_off, n)
            if self._prod_done[buf] is None:
                self._prod_done[buf] = torch.cuda.Event()
            self._prod_done[buf].record(self._worker_stream)
            self._d2h_stream.wait_event(self._prod_done[buf])
            with torch.cuda.stream(self._d2h_stream):
                dst.narrow(0, byte_off, n).copy_(seg_view, non_blocking=True)
            if self._d2h_done[buf] is None:
                self._d2h_done[buf] = torch.cuda.Event()
            self._d2h_done[buf].record(self._d2h_stream)
            last_event = self._d2h_done[buf]
            buf = (buf + 1) % nb
        return last_event
