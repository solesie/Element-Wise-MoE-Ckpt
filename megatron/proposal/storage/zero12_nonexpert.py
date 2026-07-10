"""ZeRO-1/2 non-expert PRPTENS payload planning.

This module owns the ZeRO partition-specific non-expert FC collect path:
mapping replicated non-expert tensors to local fp32 partition fragments,
packing those fragments into a pinned PRPTENS payload, and enqueueing D2H.
FCManager still owns scheduling, PRPTENS header stamping, and file commit.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..refresh.fp32_state import get_mapper
from ..runtime import CheckpointWorkerPool, get_d2h_stream
from . import proposal_io as binary_io


PayloadEntry = Tuple[str, Optional[torch.Tensor], torch.Size, int]
PlanEntry = Tuple[str, int, int, torch.Size]


@dataclass(frozen=True)
class Zero12NonExpertPayload:
    entries: List[PayloadEntry]
    plan: List[PlanEntry]
    total_numel: int
    scalars: Dict[str, Any]


@dataclass
class Zero12CollectResult:
    payload: Dict[str, torch.Tensor]
    payload_bytes: int
    pinned_info: Dict[str, Any]
    cuda_event: Optional["torch.cuda.Event"]
    read_event: Optional["torch.cuda.Event"]
    sub_ms: Dict[str, float]


def _alloc_sector_aligned_pinned_fp32(numel: int) -> torch.Tensor:
    raw = binary_io.alloc_pinned_uint8_sector_aligned(int(numel) * 4)
    return raw.view(torch.float32)


def _launch_d2h_copies(
    dsts: List[torch.Tensor],
    srcs: List[torch.Tensor],
) -> None:
    if len(dsts) != len(srcs):
        raise ValueError("D2H copy source/destination list length mismatch")
    for dst, src in zip(dsts, srcs):
        if int(dst.numel()) != int(src.numel()):
            raise ValueError(
                f"D2H copy length mismatch: dst={dst.numel()} src={src.numel()}"
            )
    for dst, src in zip(dsts, srcs):
        dst.copy_(src, non_blocking=True)


def _dist_rank_scalars() -> Dict[str, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return {
            "owner_global_rank": int(torch.distributed.get_rank()),
            "world_size": int(torch.distributed.get_world_size()),
        }
    return {"owner_global_rank": 0, "world_size": 1}


def build_zero12_nonexpert_payload(
    named_params: Dict[str, torch.nn.Parameter],
    non_expert_names: List[str],
) -> Zero12NonExpertPayload:
    """Build local ZeRO-1/2 non-expert partition entries.

    ZeRO-1/2 partitions fp32 master/state by flat local range, not by tensor
    object.  A tensor can straddle a partition boundary, so the checkpoint
    payload is expressed as local partition fragments instead of whole tensors.
    """

    mapper = get_mapper()
    if mapper is None:
        raise RuntimeError("partitioned non-expert payload requested without mapper")

    fragments = mapper.zero12_local_fragments(
        (name, named_params[name]) for name in non_expert_names
    )
    entries: List[PayloadEntry] = []
    fragment_meta: List[Dict[str, Any]] = []

    for frag_idx, (name, p, group_idx, local_offset, param_offset, numel) in enumerate(fragments):
        fragment_meta.append({
            "index": int(frag_idx),
            "name": str(name),
            "group_idx": int(group_idx),
            "local_offset": int(local_offset),
            "param_offset": int(param_offset),
            "numel": int(numel),
            "shape": [int(d) for d in p.shape],
        })
        for field_name, state_key in (
            ("param", None),
            ("exp_avg", "exp_avg"),
            ("exp_avg_sq", "exp_avg_sq"),
        ):
            src = mapper.zero12_partition_view(
                int(group_idx),
                int(local_offset),
                int(numel),
                state_key=state_key,
            )
            entries.append((
                f"frag{frag_idx}:{field_name}",
                None if src is None else src.detach(),
                torch.Size([int(numel)]),
                int(numel),
            ))

    plan: List[PlanEntry] = []
    offset = 0
    for key, _src, shape, n in entries:
        n = int(n)
        plan.append((key, offset, n, shape))
        offset += n

    return Zero12NonExpertPayload(
        entries=entries,
        plan=plan,
        total_numel=int(offset),
        scalars={
            "payload_kind": "zero12_nonexpert_partition",
            **_dist_rank_scalars(),
            "fragments": fragment_meta,
        },
    )


class Zero12NonExpertCollector:
    """Collect ZeRO-1/2 local non-expert partition payloads."""

    def __init__(
        self,
        model: torch.nn.Module,
        non_expert_names: List[str],
        *,
        cp_target: str,
        checkpoint_workers: Optional[CheckpointWorkerPool],
    ) -> None:
        self.model = model
        self.non_expert_names = non_expert_names
        self.cp_target = str(cp_target)
        self.checkpoint_workers = checkpoint_workers
        self._named_params_cache: Optional[Dict[str, torch.nn.Parameter]] = None
        self._pinned_cpu: Optional[torch.Tensor] = None
        self._pinned_ready = threading.Event()
        self._pinned_ready.set()
        if torch.cuda.is_available():
            self._main_event: Optional["torch.cuda.Event"] = torch.cuda.Event()
            self._d2h_event: Optional["torch.cuda.Event"] = torch.cuda.Event()
        else:
            self._main_event = None
            self._d2h_event = None

    def preallocate(self) -> None:
        """Allocate pinned capacity once before the training hot path."""

        named_params = self._named_params_cache
        if named_params is None:
            named_params = dict(self.model.named_parameters())
            self._named_params_cache = named_params
        mapper = get_mapper()
        if mapper is None:
            return
        fragments = mapper.zero12_local_fragments(
            (name, named_params[name]) for name in self.non_expert_names
        )
        total_numel = 0
        for _name, _param, _group_idx, _local_offset, _param_offset, numel in fragments:
            total_numel += int(numel) * 3
        header_bytes = int(binary_io.HEADER_BLOCK_BYTES)
        chunk = int(binary_io.PIN_CHUNK_BYTES)
        padded_bytes = ((header_bytes + total_numel * 4 + chunk - 1) // chunk) * chunk
        if padded_bytes & 3:
            padded_bytes = (padded_bytes + 3) & ~3
        pinned_numel = max(1, padded_bytes // 4)
        if self._pinned_cpu is None or int(self._pinned_cpu.numel()) < int(pinned_numel):
            self._pinned_cpu = _alloc_sector_aligned_pinned_fp32(pinned_numel)

    def collect(
        self,
        *,
        ready_event: Optional["torch.cuda.Event"] = None,
    ) -> Zero12CollectResult:
        t0 = time.perf_counter()
        named_params = self._named_params_cache
        if named_params is None:
            named_params = dict(self.model.named_parameters())
            self._named_params_cache = named_params
        desc = build_zero12_nonexpert_payload(
            named_params,
            self.non_expert_names,
        )
        enum_ms = (time.perf_counter() - t0) * 1000.0
        if not desc.entries:
            pinned = self._pinned_cpu
            if pinned is None:
                pinned = _alloc_sector_aligned_pinned_fp32(
                    int(binary_io.HEADER_BLOCK_BYTES) // 4
                )
                self._pinned_cpu = pinned
            return Zero12CollectResult(
                payload={},
                payload_bytes=0,
                pinned_info={
                    "pinned": pinned,
                    "header_block_bytes": int(binary_io.HEADER_BLOCK_BYTES),
                    "payload_offset_bytes": int(binary_io.HEADER_BLOCK_BYTES),
                    "payload_bytes": 0,
                    "total_pinned_bytes": int(binary_io.HEADER_BLOCK_BYTES),
                    "plan": [],
                    "scalars": dict(desc.scalars),
                },
                cuda_event=None,
                read_event=None,
                sub_ms={
                    "enum_ms": enum_ms,
                    "copy_ms": 0.0,
                    "xfer_ms": 0.0,
                    "wait_ms": 0.0,
                    "nentries": 0,
                    "d2h_copy_cmds": 0,
                },
            )

        entries = desc.entries
        total_numel = int(desc.total_numel)
        payload_bytes = int(total_numel) * 4
        header_bytes = int(binary_io.HEADER_BLOCK_BYTES)
        chunk = int(binary_io.PIN_CHUNK_BYTES)
        padded_bytes = ((header_bytes + payload_bytes + chunk - 1) // chunk) * chunk
        if padded_bytes & 3:
            padded_bytes = (padded_bytes + 3) & ~3
        pinned_numel = padded_bytes // 4

        use_writer_lease = (
            self.cp_target == "disk" and self.checkpoint_workers is not None
        )
        ready_event_for_writer: Optional[threading.Event] = None
        slab_d2h_event: Optional["torch.cuda.Event"] = None
        wait_ms = 0.0
        if use_writer_lease:
            ready_event_for_writer = self._pinned_ready
            tw = time.perf_counter()
            ready_event_for_writer.wait()
            wait_ms = (time.perf_counter() - tw) * 1000.0
            ready_event_for_writer.clear()
            slab_d2h_event = self._d2h_event
            pinned = self._pinned_cpu
            if pinned is None:
                raise RuntimeError("ZeRO12 non-expert slab was not preallocated")
            if int(pinned.numel()) < int(pinned_numel):
                raise RuntimeError(
                    f"zero12 non-expert pinned capacity {pinned.numel()} "
                    f"< required {pinned_numel}"
                )
        else:
            pinned = self._pinned_cpu
            if pinned is None:
                pinned = _alloc_sector_aligned_pinned_fp32(int(pinned_numel * 1.5))
                self._pinned_cpu = pinned
            elif int(pinned.numel()) < int(pinned_numel):
                raise RuntimeError(
                    f"zero12 non-expert pinned capacity {pinned.numel()} "
                    f"< required {pinned_numel}"
                )

        header_numel = header_bytes // 4
        flat_cpu = pinned.narrow(0, header_numel, total_numel)
        payload_template: Dict[str, torch.Tensor] = {}
        dsts: List[torch.Tensor] = []
        srcs: List[torch.Tensor] = []
        offset = 0
        for key, src, shape, n in entries:
            n = int(n)
            dst = flat_cpu.narrow(0, offset, n)
            payload_template[key] = dst.view(shape)
            if src is None:
                dst.zero_()
            else:
                dsts.append(dst)
                srcs.append(src.reshape(-1))
            offset += n

        xfer_t0 = time.perf_counter()
        cuda_event: Optional["torch.cuda.Event"] = None
        read_event: Optional["torch.cuda.Event"] = None
        if torch.cuda.is_available():
            d2h_stream = get_d2h_stream()
            d2h_event = (
                slab_d2h_event
                if slab_d2h_event is not None
                else self._d2h_event
            )
            if d2h_event is None or self._main_event is None:
                raise RuntimeError("ZeRO12 non-expert collect events are not initialized")
            if ready_event is None:
                self._main_event.record()
            with torch.cuda.stream(d2h_stream):
                if ready_event is not None:
                    d2h_stream.wait_event(ready_event)
                else:
                    d2h_stream.wait_event(self._main_event)
                _launch_d2h_copies(dsts, srcs)
                cuda_event = d2h_event
                cuda_event.record(d2h_stream)
            read_event = cuda_event
        else:
            torch._foreach_copy_(dsts, srcs)
        xfer_ms = (time.perf_counter() - xfer_t0) * 1000.0

        pinned_info: Dict[str, Any] = {
            "pinned": pinned,
            "header_block_bytes": header_bytes,
            "payload_offset_bytes": header_bytes,
            "payload_bytes": payload_bytes,
            "total_pinned_bytes": padded_bytes,
            "plan": list(desc.plan),
            "scalars": dict(desc.scalars),
        }
        if ready_event_for_writer is not None:
            pinned_info["ready_event"] = ready_event_for_writer

        return Zero12CollectResult(
            payload=payload_template,
            payload_bytes=payload_bytes,
            pinned_info=pinned_info,
            cuda_event=cuda_event,
            read_event=read_event,
            sub_ms={
                "enum_ms": enum_ms,
                "copy_ms": 0.0,
                "xfer_ms": xfer_ms,
                "wait_ms": wait_ms,
                "nentries": len(entries),
                "d2h_copy_cmds": len(srcs),
            },
        )
