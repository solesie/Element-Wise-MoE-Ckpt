"""Optional FDP write/read byte validation for smoke tests."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class _ReadbackRecord:
    kind: str
    context: str
    layer: int = -1
    expert: int = -1
    logical_bytes: int = 0
    nbytes: int = 0
    tensor: Optional[torch.Tensor] = None
    header: bytes = b""


class FdpReadbackValidator:
    """Compare FDP writes against raw reads after batch CQE completion."""

    def __init__(
        self,
        *,
        header_block_bytes: int,
        sector_bytes: int,
        enabled: Callable[[], bool],
        get_fdp: Callable[[], Any],
        alloc_cpu_u8: Callable[[int], torch.Tensor],
    ) -> None:
        self._header_block_bytes = int(header_block_bytes)
        self._sector_bytes = int(sector_bytes)
        self._enabled = enabled
        self._get_fdp = get_fdp
        self._alloc_cpu_u8 = alloc_cpu_u8
        self._lock = threading.Lock()
        self._nefc_pending: List[_ReadbackRecord] = []
        self._expert_pending: List[_ReadbackRecord] = []

    def enabled(self) -> bool:
        return bool(self._enabled())

    def queue_nefc(
        self,
        *,
        pinned_tensor: torch.Tensor,
        logical_bytes: int,
        context: str,
    ) -> None:
        if not self.enabled():
            return
        self._queue(
            self._nefc_pending,
            _ReadbackRecord(
                kind="nefc",
                tensor=pinned_tensor,
                logical_bytes=int(logical_bytes),
                context=str(context),
            ),
        )

    def queue_lfcs(
        self,
        *,
        pinned_tensor: torch.Tensor,
        logical_bytes: int,
        layer: int,
        expert: int,
        context: str,
    ) -> None:
        if not self.enabled():
            return
        self._queue(
            self._expert_pending,
            _ReadbackRecord(
                kind="lfcs",
                tensor=pinned_tensor,
                logical_bytes=int(logical_bytes),
                layer=int(layer),
                expert=int(expert),
                context=str(context),
            ),
        )

    def queue_lfsy_header(
        self,
        *,
        header: bytes,
        layer: int,
        expert: int,
        context: str,
    ) -> None:
        if not self.enabled():
            return
        self._queue(
            self._expert_pending,
            _ReadbackRecord(
                kind="lfsy_header",
                header=bytes(header),
                layer=int(layer),
                expert=int(expert),
                context=str(context),
            ),
        )

    def queue_lfsy_append(
        self,
        *,
        pinned_tensor: torch.Tensor,
        nbytes: int,
        layer: int,
        expert: int,
        context: str,
    ) -> None:
        if not self.enabled():
            return
        self._queue(
            self._expert_pending,
            _ReadbackRecord(
                kind="lfsy_append",
                tensor=pinned_tensor,
                nbytes=int(nbytes),
                layer=int(layer),
                expert=int(expert),
                context=str(context),
            ),
        )

    def validate_nefc(self) -> None:
        if not self.enabled():
            return
        pending = self._take(self._nefc_pending)
        if not pending:
            return
        total = 0
        for rec in pending:
            actual = self._read_nefc(rec.logical_bytes)
            expected = self._u8_view(rec.tensor, int(actual.numel()))
            total += self._assert_equal(
                kind="nefc",
                expected=expected,
                actual=actual,
                context=rec.context,
            )
        print(
            f"[proposal] FDP readback validation OK: kind=nefc "
            f"objects={len(pending)} bytes={total}",
            flush=True,
        )

    def validate_expert(self) -> None:
        if not self.enabled():
            return
        pending = self._take(self._expert_pending)
        if not pending:
            return

        total = 0
        lfcs_count = 0
        lfsy_header_count = 0
        lfsy_append_count = 0

        latest_lfcs: Dict[Tuple[int, int], _ReadbackRecord] = {}
        ordered: List[_ReadbackRecord] = []
        for rec in pending:
            if rec.kind == "lfcs":
                key = (int(rec.layer), int(rec.expert))
                if key not in latest_lfcs:
                    ordered.append(rec)
                latest_lfcs[key] = rec
                continue
            ordered.append(rec)

        for rec in ordered:
            if rec.kind == "lfcs":
                rec = latest_lfcs[(int(rec.layer), int(rec.expert))]
                total += self._validate_lfcs(rec)
                lfcs_count += 1
            elif rec.kind == "lfsy_header":
                total += self._validate_lfsy_header(rec)
                lfsy_header_count += 1

        lfsy_appends = [r for r in pending if r.kind == "lfsy_append"]
        if lfsy_appends:
            count, nbytes = self._validate_lfsy_appends(lfsy_appends)
            lfsy_append_count += count
            total += nbytes

        print(
            f"[proposal] FDP readback validation OK: kind=expert "
            f"lfcs={lfcs_count} lfsy_headers={lfsy_header_count} "
            f"lfsy_appends={lfsy_append_count} bytes={total}",
            flush=True,
        )

    def validate_all(self) -> None:
        self.validate_nefc()
        self.validate_expert()

    def _queue(self, target: List[_ReadbackRecord],
               rec: _ReadbackRecord) -> None:
        with self._lock:
            target.append(rec)

    def _take(self, target: List[_ReadbackRecord]) -> List[_ReadbackRecord]:
        with self._lock:
            out = list(target)
            target.clear()
        return out

    def _validate_lfcs(self, rec: _ReadbackRecord) -> int:
        actual = self._read_lfcs(rec.layer, rec.expert, rec.logical_bytes)
        expected = self._u8_view(rec.tensor, int(actual.numel()))
        return self._assert_equal(
            kind="lfcs",
            expected=expected,
            actual=actual,
            context=rec.context,
        )

    def _validate_lfsy_header(self, rec: _ReadbackRecord) -> int:
        expected_block = bytearray(self._header_block_bytes)
        expected_block[:len(rec.header)] = rec.header
        actual_bytes = self._get_fdp().debug_read_lfsy(
            int(rec.layer), int(rec.expert), 0, self._header_block_bytes
        )
        actual = torch.frombuffer(bytearray(actual_bytes), dtype=torch.uint8)
        expected = torch.frombuffer(expected_block, dtype=torch.uint8)
        return self._assert_equal(
            kind="lfsy_header",
            expected=expected,
            actual=actual,
            context=rec.context,
        )

    def _validate_lfsy_appends(
        self,
        records: List[_ReadbackRecord],
    ) -> Tuple[int, int]:
        final_tail = self._lfsy_tail_by_local_expert()
        grouped: Dict[Tuple[int, int], List[_ReadbackRecord]] = {}
        for rec in records:
            grouped.setdefault((rec.layer, rec.expert), []).append(rec)

        count = 0
        total = 0
        for eid, group in grouped.items():
            cursor = int(final_tail.get(eid, 0))
            if cursor <= 0:
                raise RuntimeError(
                    f"FDP readback validation cannot find LFSY metadata for "
                    f"L{eid[0]}/E{eid[1]}"
                )
            offsets: List[int] = []
            for rec in reversed(group):
                cursor -= int(rec.nbytes)
                offsets.append(cursor)
            offsets.reverse()
            for rec, offset in zip(group, offsets):
                if offset < 0:
                    raise RuntimeError(
                        f"FDP readback validation computed negative LFSY "
                        f"offset for L{eid[0]}/E{eid[1]} nbytes={rec.nbytes}"
                    )
                actual_bytes = self._get_fdp().debug_read_lfsy(
                    int(rec.layer), int(rec.expert), int(offset), int(rec.nbytes)
                )
                actual = torch.frombuffer(bytearray(actual_bytes), dtype=torch.uint8)
                expected = self._u8_view(rec.tensor, int(rec.nbytes))
                total += self._assert_equal(
                    kind="lfsy",
                    expected=expected,
                    actual=actual,
                    context=rec.context,
                    absolute_offset=int(offset),
                )
                count += 1
        return count, total

    def _read_nefc(self, logical_bytes: int) -> torch.Tensor:
        fdp = self._get_fdp()
        return self._read_object(
            logical_bytes,
            fdp.read_nefc_into,
            source="fdp_readback_nefc",
        )

    def _read_lfcs(
        self,
        layer: int,
        expert: int,
        logical_bytes: int,
    ) -> torch.Tensor:
        fdp = self._get_fdp()
        return self._read_object(
            logical_bytes,
            lambda addr, nbytes: fdp.read_lfcs_into(
                int(layer), int(expert), int(addr), int(nbytes)
            ),
            source=f"fdp_readback_lfcs(L{int(layer)}/E{int(expert)})",
        )

    def _read_object(self, logical_bytes: int, read_fn,
                     *, source: str) -> torch.Tensor:
        logical = int(logical_bytes)
        if logical <= 0:
            return torch.empty(0, dtype=torch.uint8)
        padded = self._round_up_sector(logical)
        buf = self._alloc_cpu_u8(padded)
        got = int(read_fn(int(buf.data_ptr()), int(padded)))
        if got < 0 or got > padded:
            raise RuntimeError(f"{source}: invalid read byte count {got}")
        return buf.narrow(0, 0, int(got))

    def _lfsy_tail_by_local_expert(self) -> Dict[Tuple[int, int], int]:
        base = int(os.environ.get("PROPOSAL_FDP_EXPERT_ID_BASE", "0") or "0")
        ep_size = int(os.environ.get("PROPOSAL_FDP_EP_SIZE", "1") or "1")
        dp_rank = int(os.environ.get("PROPOSAL_FDP_DP_RANK", "0") or "0")
        pp_rank = int(os.environ.get("PROPOSAL_FDP_PP_RANK", "0") or "0")
        ep_rank = int(os.environ.get("PROPOSAL_FDP_EP_RANK", "0") or "0")
        global_ids = os.environ.get("PROPOSAL_FDP_EXPERT_IDS_GLOBAL", "").lower() in (
            "1", "true", "yes",
        )
        out: Dict[Tuple[int, int], int] = {}
        for raw in self._get_fdp().recovery_entries().get("lfsy", []):
            entry = dict(raw)
            if not bool(entry.get("valid", False)):
                continue
            if (
                int(entry.get("dp_rank", dp_rank)) != dp_rank
                or int(entry.get("pp_rank", pp_rank)) != pp_rank
                or int(entry.get("ep_rank", ep_rank)) != ep_rank
            ):
                continue
            layer = int(entry.get("layer_id", -1))
            expert = int(entry.get("expert_id", -1))
            local_expert = expert if global_ids or ep_size <= 1 else expert - base
            if local_expert >= 0:
                out[(layer, local_expert)] = int(
                    entry.get("logical_tail", 0) or 0
                )
        return out

    def _round_up_sector(self, nbytes: int) -> int:
        n = int(nbytes)
        return ((n + self._sector_bytes - 1) // self._sector_bytes) * self._sector_bytes

    @staticmethod
    def _u8_view(
        tensor: Optional[torch.Tensor],
        nbytes: int,
        *,
        offset: int = 0,
    ) -> torch.Tensor:
        if tensor is None:
            raise RuntimeError("FDP readback validation missing source tensor")
        u8 = tensor.detach().view(torch.uint8)
        start = int(offset)
        n = int(nbytes)
        if start < 0 or n < 0 or start + n > int(u8.numel()):
            raise RuntimeError(
                f"FDP readback validation source range is out of bounds: "
                f"offset={start} nbytes={n} source={int(u8.numel())}"
            )
        return u8.narrow(0, start, n)

    @staticmethod
    def _assert_equal(
        *,
        kind: str,
        expected: torch.Tensor,
        actual: torch.Tensor,
        context: str,
        absolute_offset: int = 0,
    ) -> int:
        if expected.device.type != "cpu" or actual.device.type != "cpu":
            raise RuntimeError(f"{kind} readback validation requires CPU tensors")
        if expected.dtype != torch.uint8 or actual.dtype != torch.uint8:
            raise RuntimeError(f"{kind} readback validation requires uint8 tensors")
        if int(expected.numel()) != int(actual.numel()):
            raise RuntimeError(
                f"FDP readback mismatch ({kind}): size differs for {context}: "
                f"expected={int(expected.numel())} actual={int(actual.numel())}"
            )
        if torch.equal(expected, actual):
            return int(expected.numel())
        diff = torch.nonzero(expected != actual, as_tuple=False)
        rel = int(diff[0].item()) if int(diff.numel()) > 0 else -1
        exp_b = int(expected[rel].item()) if rel >= 0 else -1
        got_b = int(actual[rel].item()) if rel >= 0 else -1
        raise RuntimeError(
            f"FDP readback mismatch ({kind}): {context} first_bad_offset="
            f"{int(absolute_offset) + rel} expected_byte={exp_b} "
            f"actual_byte={got_b} nbytes={int(expected.numel())}"
        )
