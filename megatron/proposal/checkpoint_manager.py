"""Non-expert DC manager.

Expert state is handled entirely by ``ExpertState`` (per-expert lossy FC
append log + round-robin lossless FC snapshot, K% selections appended
each iter). Expert DC bundles are NOT written here — legacy
expert-sparse and expert-grad-replay paths have been removed.

This class only writes per-iter non-expert DENSE grad bundles
(``dc_iter{N}_dense_ne.pt``), one per iter except when non-expert FC
fires that iter. Retention is N (non-expert FC supersedes dense DC).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .async_writer import AsyncWriter, WriteRecord
from .binary_io import save_binary


@dataclass
class DCBundleStats:
    step: int
    path: Path
    nonexpert_bytes: int
    payload_bytes: int
    file_bytes: int = 0
    write_time_ms: float = 0.0
    queue_delay_ms: float = 0.0
    async_mode: bool = False


class CheckpointManager:
    """Per-iter non-expert dense DC writer with retention."""

    def __init__(
        self,
        output_dir: Path,
        retention: int,
        async_writer: Optional[AsyncWriter] = None,
    ) -> None:
        if retention < 1:
            raise ValueError(f"retention must be ≥1, got {retention}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.retention = int(retention)
        self._async_writer = async_writer

        self.write_log: List[DCBundleStats] = []
        self._write_records: List[Optional[WriteRecord]] = []
        self._live_dense_ne: List[Tuple[Path, Optional[WriteRecord]]] = []

    # ------------------------------------------------------------------
    def record(
        self,
        step: int,
        nonexpert_payload: Dict[str, torch.Tensor],
        nonexpert_did_fc: bool = False,
    ) -> DCBundleStats:
        """Write non-expert dense DC bundle for this iter.

        Skipped when non-expert FC fires this iter (FC supersedes DC).
        """
        if nonexpert_did_fc or not nonexpert_payload:
            stats = DCBundleStats(
                step=int(step),
                path=Path("/dev/null"),
                nonexpert_bytes=0,
                payload_bytes=0,
                async_mode=self._async_writer is not None,
            )
            self.write_log.append(stats)
            self._write_records.append(None)
            return stats

        ne_bytes = 0
        for t in nonexpert_payload.values():
            ne_bytes += t.numel() * t.element_size()

        # Binary format (PRPTENS). DC is written every iter so pickle
        # overhead was non-trivial (~300 ms / bundle). Tensors go in the
        # body, ``step`` goes in the scalar header.
        dense_path = self.output_dir / f"dc_iter{step:08d}_dense_ne.bin"
        scalars = {"step": int(step)}

        stats = DCBundleStats(
            step=int(step),
            path=dense_path,
            nonexpert_bytes=ne_bytes,
            payload_bytes=ne_bytes,
            async_mode=self._async_writer is not None,
        )

        def _save_dc(tensor_dict, p):
            save_binary(p, tensor_dict, scalars=scalars)

        to_unlink: List[Path] = []
        if self._async_writer is None:
            import time as _time
            t0 = _time.perf_counter()
            _save_dc(nonexpert_payload, dense_path)
            t1 = _time.perf_counter()
            stats.write_time_ms = (t1 - t0) * 1000.0
            try:
                stats.file_bytes = int(dense_path.stat().st_size)
            except FileNotFoundError:
                stats.file_bytes = 0
            self._live_dense_ne.append((dense_path, None))
            while len(self._live_dense_ne) > self.retention:
                old_path, _ = self._live_dense_ne.pop(0)
                try:
                    old_path.unlink()
                except FileNotFoundError:
                    pass
            rec: Optional[WriteRecord] = None
        else:
            self._live_dense_ne.append((dense_path, None))
            while len(self._live_dense_ne) > self.retention:
                old_path, _ = self._live_dense_ne.pop(0)
                to_unlink.append(old_path)

            def _unlink_many(paths: List[Path] = to_unlink) -> None:
                for p in paths:
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass

            rec = self._async_writer.submit(
                nonexpert_payload, dense_path, tag="dc-dense",
                on_complete=[_unlink_many] if to_unlink else None,
                save_fn=_save_dc,
            )
            self._live_dense_ne[-1] = (dense_path, rec)

        self.write_log.append(stats)
        self._write_records.append(rec)
        return stats

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        if self._async_writer is None:
            return
        self._async_writer.drain()
        for stats, rec in zip(self.write_log, self._write_records):
            if rec is None:
                continue
            stats.write_time_ms = rec.write_time_ms
            stats.file_bytes = rec.file_bytes
            stats.queue_delay_ms = rec.queue_delay_ms

    # ------------------------------------------------------------------
    @property
    def num_bundles(self) -> int:
        return len(self.write_log)

    @property
    def total_bytes_written(self) -> int:
        return sum(s.file_bytes for s in self.write_log)

    def write_log_dict(self) -> List[Dict]:
        return [
            {
                "step": s.step,
                "path": str(s.path),
                "nonexpert_bytes": s.nonexpert_bytes,
                "payload_bytes": s.payload_bytes,
                "file_bytes": s.file_bytes,
                "write_time_ms": s.write_time_ms,
                "queue_delay_ms": s.queue_delay_ms,
                "async_mode": s.async_mode,
            }
            for s in self.write_log
        ]

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        for path, _rec in self._live_dense_ne:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._live_dense_ne.clear()
        try:
            self.output_dir.rmdir()
        except OSError:
            pass
