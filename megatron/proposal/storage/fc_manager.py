"""Full Checkpoint manager — expert LFCL scheduling + periodic non-expert FC.

DeepSpeed MoE only: expert params live in separated per-expert tensors with
name pattern ``...deepspeed_experts.{idx}...``.
"""
from __future__ import annotations

import re
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

import math

from ..runtime import CheckpointWorkerPool, WriteRecord, get_d2h_stream
from . import proposal_io as binary_io  # facade keeps existing callsites unchanged
from ..refresh.fp32_state import get_mapper
from .expert_writer import ExpertState, HEADER_BLOCK_BYTES


def _fp32_param_live_view(p: torch.nn.Parameter) -> torch.Tensor:
    """Return an existing FP32 param view; never allocate a conversion tensor."""

    m = get_mapper()
    if m is not None:
        direct = getattr(m, "_direct_master", {}).get(id(p))
        if direct is not None:
            if direct.dtype != torch.float32:
                raise RuntimeError("FP32 mapper returned non-FP32 direct master")
            return direct
        coord = getattr(m, "_coords", {}).get(id(p))
        flat = getattr(m, "_fp32_flat", None)
        if coord is not None and flat is not None:
            group_idx, offset, numel = coord
            out = flat[group_idx].narrow(0, offset, numel).view(p.shape)
            if out.dtype != torch.float32:
                raise RuntimeError("FP32 mapper returned non-FP32 flat view")
            return out
    if p.data.dtype == torch.float32:
        return p.data
    raise RuntimeError(
        "expert LFCL requires an existing FP32 live param view; refusing "
        "to allocate a transient dtype-conversion staging tensor"
    )


def _fp32_opt_state_live_view(
    optimizer: torch.optim.Optimizer,
    p: torch.nn.Parameter,
    key: str,
) -> Optional[torch.Tensor]:
    """Return an existing local FP32 Adam-state view; never materialize full state."""

    m = get_mapper()
    if m is not None:
        local = m.get_local_state_view(p, key)
        if local is not None:
            return local
    return optimizer.state.get(p, {}).get(key)


LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
DS_EXPERT_IDX_RE = re.compile(r"deepspeed_experts\.(\d+)\.")


ExpertId = Tuple[int, int]  # (layer, expert_idx)


def _alloc_sector_aligned_pinned_fp32(numel: int) -> torch.Tensor:
    raw = binary_io.alloc_pinned_uint8_sector_aligned(int(numel) * 4)
    return raw.view(torch.float32)


def _round_up_bytes(value: int, align: int) -> int:
    align = max(1, int(align))
    return ((int(value) + align - 1) // align) * align


def _launch_d2h_copies(
    dsts: List[torch.Tensor],
    srcs: List[torch.Tensor],
) -> None:
    """Launch one GPU→pinned copy per contiguous FC tensor slice."""

    if not dsts:
        return
    if len(dsts) != len(srcs):
        raise ValueError("D2H copy source/destination list length mismatch")

    for dst, src in zip(dsts, srcs):
        n = int(src.numel())
        if n != int(dst.numel()):
            raise ValueError(
                f"D2H copy length mismatch: dst={dst.numel()} src={src.numel()}"
            )

    for dst, src in zip(dsts, srcs):
        dst.copy_(src, non_blocking=True)


def _count_d2h_copies(srcs: List[torch.Tensor]) -> int:
    return int(len(srcs))


def _detect_expert_layout(named_params, num_experts: int) -> str:
    """Sanity-check: assert the model uses DeepSpeed 'separated' expert layout.

    Raises if any param flagged with ``allreduce=False`` does not match the
    expected ``deepspeed_experts.{idx}.`` regex.
    """
    has_separated = False
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if getattr(p, "allreduce", True) is False:
            if not DS_EXPERT_IDX_RE.search(name):
                raise RuntimeError(
                    f"Param {name!r} is flagged allreduce=False (DeepSpeed "
                    f"expert marker) but does not match 'deepspeed_experts.<idx>.'; "
                    f"unexpected model layout."
                )
            has_separated = True
        elif DS_EXPERT_IDX_RE.search(name):
            has_separated = True
    return "separated" if has_separated else "none"


@dataclass
class FCEventStats:
    kind: str
    step: int
    expert_id: Optional[ExpertId]
    tensor_names: List[str]
    path: Path
    payload_bytes: int
    file_bytes: int = 0
    write_time_ms: float = 0.0
    queue_delay_ms: float = 0.0
    async_mode: bool = False


class FCManager:
    """Staggered per-expert FC + periodic non-expert FC (DeepSpeed MoE)."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        output_dir: Path,
        num_experts,  # int or List[int] (PR-MoE: per-layer counts)
        nonexpert_fc_W: int = 10,
        checkpoint_workers: Optional[CheckpointWorkerPool] = None,
        keep_last_per_expert: int = 1,
        keep_last_nonexpert: int = 1,
        schedule: str = "proposal",
        lfcl_fraction_pct: float = 1.0,
        expert_output_dir: Optional[Path] = None,
        fresh_expert_state: bool = True,
        cp_target: str = "disk",
        pinned_aos_capacity_ratio: Optional[float] = None,
        pinned_aos_capacity_tolerance: float = 0.75,
        enable_expert_state: bool = True,
        commit_expert_lossy_records: bool = True,
        commit_nonexpert_fc: bool = True,
        test_skip_expert_d2h: bool = False,
        test_skip_nonexpert_d2h: bool = False,
        nonexpert_shard_rank: int = 0,
        nonexpert_shard_count: int = 1,
    ) -> None:
        if nonexpert_fc_W < 1:
            raise ValueError(f"nonexpert_fc_W must be ≥1, got {nonexpert_fc_W}")

        self.model = model
        self.optimizer = optimizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Expert rolling files can live in a separate directory (split CP
        # layout). Default: same directory as non-expert FC.
        self.expert_output_dir = Path(expert_output_dir) if expert_output_dir is not None else self.output_dir
        self.expert_output_dir.mkdir(parents=True, exist_ok=True)
        # num_experts: keep raw spec; resolve per-layer counts after enumeration.
        self._num_experts_spec = num_experts
        self.W = int(nonexpert_fc_W)
        self._checkpoint_workers = checkpoint_workers
        self.keep_last_per_expert = int(keep_last_per_expert)
        self.keep_last_nonexpert = int(keep_last_nonexpert)

        self._expert_tensors_by_layer: Dict[int, List[str]] = {}
        self._non_expert_names: List[str] = []
        self._enumerate_params()
        self._nonexpert_shard_rank = max(0, int(nonexpert_shard_rank))
        self._nonexpert_shard_count = max(1, int(nonexpert_shard_count))
        if self._nonexpert_shard_rank >= self._nonexpert_shard_count:
            raise ValueError(
                "nonexpert_shard_rank must be < nonexpert_shard_count, "
                f"got {self._nonexpert_shard_rank} / {self._nonexpert_shard_count}"
            )
        if self._nonexpert_shard_count > 1:
            self._apply_nonexpert_name_shard()
        self._sorted_layers: List[int] = sorted(self._expert_tensors_by_layer.keys())
        self._num_layers = len(self._sorted_layers)

        # PR-MoE support: per-layer expert count from observed expert tensors.
        # _all_expert_ids enumerates every (layer, expert_idx) tuple actually
        # present, giving a flat slot index that handles heterogeneous counts.
        self._all_expert_ids: List[ExpertId] = sorted(self._separated_tensors_by_slot.keys())
        self._slot_index: Dict[ExpertId, int] = {
            eid: i for i, eid in enumerate(self._all_expert_ids)
        }
        self._num_experts_by_layer: Dict[int, int] = {
            layer: 0 for layer in self._sorted_layers
        }
        for layer, _expert_idx in self._all_expert_ids:
            self._num_experts_by_layer[layer] += 1
        self.num_experts = (
            max(self._num_experts_by_layer.values()) if self._num_experts_by_layer else 0
        )
        self._num_expert_slots = len(self._all_expert_ids)
        self.lfcl_interval_iters = 1

        if schedule not in ("proposal", "sequential"):
            raise ValueError(
                f"schedule must be 'proposal' or 'sequential', got {schedule!r}"
            )
        self._schedule = schedule

        # Proposal LFCL is fixed by V2: every iter, 1% expert slots by planner.
        self._rr_cursor = 0
        self.lfcl_fraction_pct = float(lfcl_fraction_pct)
        if self._num_expert_slots > 0 and self.lfcl_fraction_pct > 0:
            self._experts_per_lfcl_event = max(
                1,
                math.ceil(
                    self._num_expert_slots * self.lfcl_fraction_pct / 100.0
                ),
            )
        else:
            self._experts_per_lfcl_event = 0

        # MoC sequential baseline: layer-wise round robin, driven by
        # _layer_rr_tick.  This is an explicit ablation mode, not the proposal
        # LFCL default above.
        self._layer_rr_tick = 0

        self._expert_fc_files: Dict[ExpertId, List[Tuple[Path, Optional[WriteRecord]]]] = {}
        self._nonexpert_fc_files: List[Tuple[Path, Optional[WriteRecord]]] = []

        self.write_log: List[FCEventStats] = []
        self._write_records: List[Optional[WriteRecord]] = []

        # Last-tick phase breakdown (diagnostic). Populated by ``tick()``.
        # Keys: nonexpert_fc_ms, n_experts_fired.
        self.last_tick_phase_ms: Dict[str, float] = {}

        # Cached pinned CPU buffers for fast GPU→CPU transfers. Non-expert
        # FC has one owner and reuses a single slab/ring. Expert LFCL uses
        # one per-iteration contiguous slab and one writer lease, matching
        # the AOS ownership model.
        self._collect_pinned_cpu: Optional[torch.Tensor] = None
        _ne_ring_slots = 4
        self._collect_nonexpert_pinned_ring: List[Optional[torch.Tensor]] = [
            None for _ in range(_ne_ring_slots)
        ]
        self._collect_nonexpert_pinned_ready: List[threading.Event] = [
            threading.Event() for _ in self._collect_nonexpert_pinned_ring
        ]
        for _ev in self._collect_nonexpert_pinned_ready:
            _ev.set()
        self._collect_nonexpert_d2h_events: List[Optional["torch.cuda.Event"]] = []
        self._collect_nonexpert_pinned_cursor: int = 0
        self._lfcl_batch_pinned: Optional[torch.Tensor] = None
        self._lfcl_batch_ready = threading.Event()
        self._lfcl_batch_ready.set()
        self._lfcl_batch_capacity_bytes: int = 0
        self._lfcl_payload_numel_by_expert: Dict[ExpertId, int] = {}
        self._lfcl_field_cache_by_expert: Dict[ExpertId, List[Dict[str, Any]]] = {}
        self._lfcl_adam_source_cache_warmed: bool = False
        self._last_lfcl_batch_cuda_event: Optional["torch.cuda.Event"] = None
        self._last_lfcl_batch_info: Optional[Dict[str, Any]] = None
        self._last_lfcl_batch_sub_ms: Dict[str, float] = {}

        # Generic collect is now non-expert only. The lock serialises rare
        # cache-miss writes; cache hits are stable dict reads.
        self._collect_cache_lock = threading.Lock()
        if torch.cuda.is_available():
            self._collect_main_event: Optional["torch.cuda.Event"] = torch.cuda.Event()
            self._collect_d2h_event: Optional["torch.cuda.Event"] = torch.cuda.Event()
            self._collect_nonexpert_d2h_events = [
                torch.cuda.Event() for _ in self._collect_nonexpert_pinned_ring
            ]
            self._lfcl_batch_main_event: Optional["torch.cuda.Event"] = torch.cuda.Event()
            self._lfcl_batch_d2h_event: Optional["torch.cuda.Event"] = torch.cuda.Event()
        else:
            self._collect_main_event = None
            self._collect_d2h_event = None
            self._collect_nonexpert_d2h_events = [
                None for _ in self._collect_nonexpert_pinned_ring
            ]
            self._lfcl_batch_main_event = None
            self._lfcl_batch_d2h_event = None

        # Per-expert state (LFSY append log + round-robin LFCL compaction
        # baseline). See ``expert_state.ExpertState``. Skipped in
        # ``sequential`` mode (MoC-System baseline: no element-wise lossy FC).
        # cp_target selects durable disk persistence or the test-only pinned
        # in-memory recovery image. Both targets share GPU selection/AOS/D2H;
        # test_dram skips only filesystem writes.
        #
        # ExpertState is created for BOTH schedules. The modes differ only
        # in which experts fire each iter and whether the proposal pipeline's
        # lossy write path runs. Sequential mode only emits dense
        # expert snapshots.
        if cp_target not in ("disk", "test_dram"):
            raise ValueError(
                "cp_target must be 'disk' or 'test_dram', "
                f"got {cp_target!r}"
            )
        self.cp_target = cp_target
        self.commit_nonexpert_fc = bool(commit_nonexpert_fc)
        self.test_skip_nonexpert_d2h = bool(test_skip_nonexpert_d2h)
        if self.test_skip_nonexpert_d2h and self.commit_nonexpert_fc:
            raise ValueError(
                "test_skip_nonexpert_d2h requires commit_nonexpert_fc=False; "
                "it is only valid for write-excluded iter-time profiling."
            )
        self._nonexpert_test_dram_payload: Optional[Dict[str, torch.Tensor]] = None
        self._nonexpert_test_dram_step: int = -1
        self._expert_state: Optional[ExpertState] = None
        self._named_params_cache: Optional[Dict[str, torch.nn.Parameter]] = None
        if enable_expert_state:
            named_params = dict(self.model.named_parameters())
            self._named_params_cache = named_params
            expert_tensor_map: Dict[ExpertId, List[Tuple[str, torch.Size, int]]] = {}
            for eid, tnames in self._separated_tensors_by_slot.items():
                entries: List[Tuple[str, torch.Size, int]] = []
                for tname in tnames:
                    p = named_params[tname]
                    entries.append((tname, p.shape, int(p.numel())))
                expert_tensor_map[eid] = entries
            self._init_lfcl_batch_pinned(expert_tensor_map)
            pinned_aos_record_capacity: Optional[int] = None
            if pinned_aos_capacity_ratio is not None:
                local_expert_numel = sum(
                    int(numel)
                    for schema in expert_tensor_map.values()
                    for (_tname, _shape, numel) in schema
                )
                ratio = max(0.0, float(pinned_aos_capacity_ratio))
                tolerance = max(0.0, float(pinned_aos_capacity_tolerance))
                pinned_aos_record_capacity = min(
                    local_expert_numel,
                    max(
                        1 if local_expert_numel > 0 else 0,
                        math.ceil(local_expert_numel * ratio * (1.0 + tolerance)),
                    ),
                )
            else:
                pinned_aos_record_capacity = None
            self._expert_state = ExpertState(
                expert_tensor_map, output_dir=self.expert_output_dir,
                fresh=fresh_expert_state,
                checkpoint_workers=self._checkpoint_workers,
                cp_target=cp_target,
                pinned_aos_record_capacity=pinned_aos_record_capacity,
                commit_lossy_records=commit_expert_lossy_records,
                test_skip_d2h=bool(test_skip_expert_d2h),
            )
        self._init_nonexpert_pinned_buffers()

    # ------------------------------------------------------------------
    def _enumerate_params(self) -> None:
        expert_tensor_names: Set[str] = set()
        self._separated_tensors_by_slot: Dict[ExpertId, List[str]] = {}

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            m_e = DS_EXPERT_IDX_RE.search(name)
            if m_e is None:
                continue
            expert_idx = int(m_e.group(1))
            m_l = LAYER_RE.search(name)
            layer = int(m_l.group(1)) if m_l is not None else 0
            self._separated_tensors_by_slot.setdefault(
                (layer, expert_idx), []
            ).append(name)
            self._expert_tensors_by_layer.setdefault(layer, []).append(name)
            expert_tensor_names.add(name)

        for layer in self._expert_tensors_by_layer:
            self._expert_tensors_by_layer[layer].sort()
        for slot in self._separated_tensors_by_slot:
            self._separated_tensors_by_slot[slot].sort()
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name in expert_tensor_names:
                continue
            self._non_expert_names.append(name)

    def _apply_nonexpert_name_shard(self) -> None:
        """Keep only this EP replica's size-balanced non-expert tensor shard."""
        if not self._non_expert_names:
            return
        named_params = dict(self.model.named_parameters())
        shard_names: List[List[str]] = [
            [] for _ in range(self._nonexpert_shard_count)
        ]
        shard_loads = [0 for _ in range(self._nonexpert_shard_count)]
        weighted: List[Tuple[int, str]] = []
        for name in self._non_expert_names:
            p = named_params[name]
            weighted.append((int(p.numel()), name))
        if not weighted:
            self._non_expert_names = []
            return
        for numel, name in sorted(weighted, reverse=True):
            shard = min(
                range(self._nonexpert_shard_count),
                key=lambda idx: (shard_loads[idx], idx),
            )
            shard_names[shard].append(name)
            shard_loads[shard] += int(numel)
        selected = set(shard_names[self._nonexpert_shard_rank])
        self._non_expert_names = [
            name for name in self._non_expert_names if name in selected
        ]

    def _init_nonexpert_pinned_buffers(self) -> None:
        """Allocate non-expert FC pinned destination buffers before training.

        The hot path must only enqueue D2H copies into already-owned pinned
        memory.  Lazy cudaHostAlloc in the first few iterations shows up as a
        multi-second proposal stall and violates the allocation contract.
        """

        if self.test_skip_nonexpert_d2h or not self._non_expert_names:
            return
        named_params = self._named_params_cache
        if named_params is None:
            named_params = dict(self.model.named_parameters())
            self._named_params_cache = named_params
        total_numel = 0
        for name in self._non_expert_names:
            p = named_params[name]
            total_numel += int(p.numel()) * 3
        header_bytes = int(binary_io.HEADER_BLOCK_BYTES)
        padded_bytes = _round_up_bytes(
            header_bytes + int(total_numel) * 4,
            int(binary_io.PIN_CHUNK_BYTES),
        )
        pinned_numel = padded_bytes // 4
        if (
            self.cp_target == "disk"
            and self._checkpoint_workers is not None
            and len(self._collect_nonexpert_pinned_ring) > 1
        ):
            for i in range(len(self._collect_nonexpert_pinned_ring)):
                if self._collect_nonexpert_pinned_ring[i] is None:
                    self._collect_nonexpert_pinned_ring[i] = (
                        _alloc_sector_aligned_pinned_fp32(pinned_numel)
                    )
            return
        if self._collect_pinned_cpu is None:
            self._collect_pinned_cpu = _alloc_sector_aligned_pinned_fp32(
                pinned_numel,
            )

    # ------------------------------------------------------------------
    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_expert_slots(self) -> int:
        return self._num_expert_slots

    @property
    def expert_state(self) -> Optional[ExpertState]:
        return self._expert_state

    @property
    def expert_aos_total_bytes(self) -> int:
        state = self._expert_state
        if state is None:
            return 0
        return int(state.pinned_aos_total_bytes)

    def _init_lfcl_batch_pinned(
        self,
        expert_tensor_map: Dict[ExpertId, List[Tuple[str, torch.Size, int]]],
    ) -> None:
        """Allocate the single contiguous pinned LFCL batch slab once."""

        sector = int(binary_io.SECTOR_BYTES)
        per_expert_slot_bytes: Dict[ExpertId, int] = {}
        for eid, schema in expert_tensor_map.items():
            payload_numel = sum(int(numel) * 3 for (_tname, _shape, numel) in schema)
            self._lfcl_payload_numel_by_expert[eid] = int(payload_numel)
            slot_bytes = _round_up_bytes(
                int(HEADER_BLOCK_BYTES) + int(payload_numel) * 4,
                sector,
            )
            per_expert_slot_bytes[eid] = int(slot_bytes)
            named_params = self._named_params_cache
            if named_params is not None:
                self._lfcl_field_cache_by_expert[eid] = [
                    {
                        "name": tname,
                        "param": named_params[tname],
                        "shape": torch.Size(shape),
                        "numel": int(numel),
                        "param_src": _fp32_param_live_view(named_params[tname]).detach(),
                        "exp_avg": None,
                        "exp_avg_sq": None,
                    }
                    for (tname, shape, numel) in schema
                ]

        if not per_expert_slot_bytes:
            return
        if self._schedule == "sequential":
            max_fired = 0
            for layer in self._sorted_layers:
                max_fired = max(
                    max_fired,
                    sum(1 for eid in self._all_expert_ids if eid[0] == layer),
                )
        else:
            max_fired = int(self._experts_per_lfcl_event)
        max_fired = max(0, min(max_fired, len(per_expert_slot_bytes)))
        if max_fired <= 0:
            return

        largest_slots = sorted(per_expert_slot_bytes.values(), reverse=True)[:max_fired]
        capacity_bytes = _round_up_bytes(sum(largest_slots), sector)
        self._lfcl_batch_capacity_bytes = int(capacity_bytes)
        self._lfcl_batch_pinned = _alloc_sector_aligned_pinned_fp32(
            (capacity_bytes + 3) // 4,
        )

    def _lfcl_cached_state_source(
        self,
        field: Dict[str, Any],
        key: str,
    ) -> Optional[torch.Tensor]:
        src = field.get(key)
        if src is not None:
            return src
        v = _fp32_opt_state_live_view(self.optimizer, field["param"], key)
        if v is None:
            return None
        src = v.detach()
        field[key] = src
        return src

    def _warm_lfcl_adam_source_cache_once(self) -> None:
        if self._lfcl_adam_source_cache_warmed:
            return
        for fields in self._lfcl_field_cache_by_expert.values():
            for field in fields:
                self._lfcl_cached_state_source(field, "exp_avg")
                self._lfcl_cached_state_source(field, "exp_avg_sq")
        self._lfcl_adam_source_cache_warmed = True

    def _slot_of(self, layer: int, expert_idx: int) -> int:
        return self._slot_index[(layer, expert_idx)]

    def _expert_from_slot(self, slot: int) -> ExpertId:
        return self._all_expert_ids[slot]

    def collect_lfcl_batch_payload(
        self,
        targets: List[Tuple[ExpertId, List[str]]],
        *,
        ready_event: Optional["torch.cuda.Event"] = None,
    ) -> Optional[Dict[str, Any]]:
        """Materialize all fired LFCL experts into one leased pinned slab.

        The returned batch owns one ready event.  Writer code releases that
        single lease after the LFCL D2H event has completed and commit/skip
        handling is done.
        """

        if not targets:
            self._last_lfcl_batch_cuda_event = None
            self._last_lfcl_batch_info = None
            self._last_lfcl_batch_sub_ms = {
                "enum_ms": 0.0,
                "copy_ms": 0.0,
                "xfer_ms": 0.0,
                "wait_ms": 0.0,
                "nentries": 0,
                "d2h_copy_cmds": 0,
            }
            return None
        if self._lfcl_batch_pinned is None:
            raise RuntimeError("expert LFCL batch pinned slab was not initialized")

        wait_t0 = time.perf_counter()
        batch_ready = self._lfcl_batch_ready
        batch_ready.wait()
        batch_ready.clear()
        wait_ms = (time.perf_counter() - wait_t0) * 1000.0
        try:
            enum_t0 = time.perf_counter()
            self._warm_lfcl_adam_source_cache_once()
            pinned = self._lfcl_batch_pinned
            sector = int(binary_io.SECTOR_BYTES)
            header_numel = int(HEADER_BLOCK_BYTES) // 4
            cursor_bytes = 0
            batch_entries: List[Dict[str, Any]] = []
            dsts: List[torch.Tensor] = []
            srcs: List[torch.Tensor] = []
            total_payload_bytes = 0
            n_plan_entries = 0
            first_src: Optional[torch.Tensor] = None

            for expert_id, tnames in targets:
                payload_numel = int(
                    self._lfcl_payload_numel_by_expert.get(expert_id, 0)
                )
                if payload_numel <= 0:
                    continue
                payload_bytes = int(payload_numel) * 4
                slot_bytes = _round_up_bytes(
                    int(HEADER_BLOCK_BYTES) + payload_bytes,
                    sector,
                )
                cursor_bytes = _round_up_bytes(cursor_bytes, sector)
                if cursor_bytes + slot_bytes > self._lfcl_batch_capacity_bytes:
                    raise RuntimeError(
                        "expert LFCL batch pinned slab capacity exceeded: "
                        f"used={cursor_bytes} need={slot_bytes} "
                        f"capacity={self._lfcl_batch_capacity_bytes}"
                    )

                slot = pinned.narrow(0, cursor_bytes // 4, slot_bytes // 4)
                flat_cpu = slot.narrow(0, header_numel, payload_numel)
                offset = 0
                fields_for_expert = self._lfcl_field_cache_by_expert.get(expert_id)
                if fields_for_expert is None:
                    named_params = self._named_params_cache
                    if named_params is None:
                        named_params = dict(self.model.named_parameters())
                        self._named_params_cache = named_params
                    fields_for_expert = []
                    for name in tnames:
                        p = named_params[name]
                        fields_for_expert.append({
                            "name": name,
                            "param": p,
                            "shape": torch.Size(p.shape),
                            "numel": int(p.numel()),
                            "param_src": _fp32_param_live_view(p).detach(),
                            "exp_avg": None,
                            "exp_avg_sq": None,
                        })
                    self._lfcl_field_cache_by_expert[expert_id] = fields_for_expert
                for field in fields_for_expert:
                    n = int(field["numel"])
                    fields: List[Tuple[str, Optional[torch.Tensor]]] = [
                        ("param", field["param_src"]),
                        ("exp_avg", self._lfcl_cached_state_source(field, "exp_avg")),
                        ("exp_avg_sq", self._lfcl_cached_state_source(field, "exp_avg_sq")),
                    ]
                    for key, src in fields:
                        dst = flat_cpu.narrow(0, offset, n)
                        if src is None:
                            dst.zero_()
                        else:
                            src_flat = src.detach().reshape(-1)
                            if src_flat.dtype != torch.float32:
                                raise RuntimeError(
                                    "expert LFCL source must be FP32; refusing "
                                    "to allocate a transient dtype-conversion "
                                    "staging tensor"
                                )
                            if first_src is None:
                                first_src = src_flat
                            dsts.append(dst)
                            srcs.append(src_flat)
                        offset += n
                        n_plan_entries += 1
                if offset != payload_numel:
                    raise RuntimeError(
                        f"LFCL payload size mismatch for expert {expert_id}: "
                        f"planned={offset} expected={payload_numel}"
                    )
                batch_entries.append({
                    "expert_id": expert_id,
                    "pinned_info": {
                        "pinned": slot,
                        "header_block_bytes": int(HEADER_BLOCK_BYTES),
                        "payload_offset_bytes": int(HEADER_BLOCK_BYTES),
                        "payload_bytes": int(payload_bytes),
                        "total_pinned_bytes": int(slot_bytes),
                    },
                })
                total_payload_bytes += payload_bytes
                cursor_bytes += slot_bytes

            enum_ms = (time.perf_counter() - enum_t0) * 1000.0
            if not batch_entries:
                batch_ready.set()
                self._last_lfcl_batch_cuda_event = None
                self._last_lfcl_batch_info = None
                self._last_lfcl_batch_sub_ms = {
                    "enum_ms": enum_ms,
                    "copy_ms": 0.0,
                    "xfer_ms": 0.0,
                    "wait_ms": wait_ms,
                    "nentries": 0,
                    "d2h_copy_cmds": 0,
                }
                return None
            if first_src is None:
                raise RuntimeError("expert LFCL batch has no live GPU source tensors")

            xfer_t0 = time.perf_counter()
            cuda_event: Optional["torch.cuda.Event"] = None
            d2h_copy_cmds = _count_d2h_copies(srcs)
            if torch.cuda.is_available():
                d2h_stream = get_d2h_stream()
                d2h_event = self._lfcl_batch_d2h_event
                main_event = self._lfcl_batch_main_event
                if d2h_event is None or main_event is None:
                    raise RuntimeError("LFCL batch D2H events were not initialized")
                if ready_event is None:
                    main_event.record()
                with torch.cuda.stream(d2h_stream):
                    if ready_event is not None:
                        d2h_stream.wait_event(ready_event)
                    else:
                        d2h_stream.wait_event(main_event)
                    _launch_d2h_copies(dsts, srcs)
                    d2h_event.record(d2h_stream)
                cuda_event = d2h_event
            else:
                torch._foreach_copy_(dsts, srcs)
            xfer_ms = (time.perf_counter() - xfer_t0) * 1000.0

            batch_info = {
                "pinned": pinned,
                "ready_event": batch_ready,
                "entries": batch_entries,
                "used_bytes": int(cursor_bytes),
                "payload_bytes": int(total_payload_bytes),
                "total_pinned_bytes": int(self._lfcl_batch_capacity_bytes),
            }
            self._last_lfcl_batch_cuda_event = cuda_event
            self._last_lfcl_batch_info = batch_info
            self._last_lfcl_batch_sub_ms = {
                "enum_ms": enum_ms,
                "copy_ms": 0.0,
                "xfer_ms": xfer_ms,
                "wait_ms": wait_ms,
                "nentries": int(n_plan_entries),
                "d2h_copy_cmds": int(d2h_copy_cmds),
            }
            return batch_info
        except Exception:
            batch_ready.set()
            raise

    # ------------------------------------------------------------------
    def fire_nonexpert_fc_if_due(
        self,
        step: int,
        ready_event: Optional["torch.cuda.Event"] = None,
        d2h_event_callback: Optional[Any] = None,
    ) -> Tuple[List[ExpertId], bool]:
        """Fire non-expert FC at iteration ``step`` (1-indexed) if due
        (step == 1 or step % W == 0).

        Returns ``(fired_experts, nonexpert_fired)``. NOTE (post-B2):
        expert LFCL compaction is dispatched OUT-OF-BAND by ExpertLosslessFC
        (separate module, runs on compute_stream). This method now
        ONLY handles non-expert FC; ``fired_experts`` returned here is
        always [] for backward compatibility — callers that need the
        fired list must read it from ExpertLosslessFC.fire_if_due()
        directly.
        """
        fired_experts: List[ExpertId] = []
        sub: Dict[str, float] = {
            "nonexpert_fc_ms": 0.0,
            "n_experts_fired": 0.0,
        }

        nonexpert_fired = False
        # Fire non-expert FC at iter 1 unconditionally so that any early
        # fault (iter < W) has a loadable non-expert FC. Without this,
        # recover() raises "No non-expert FC found with iter ≤ t_fault".
        if step == 1 or step % self.W == 0:
            _t_ne0 = time.perf_counter()
            self._fire_nonexpert_fc(
                step,
                ready_event=ready_event,
                d2h_event_callback=d2h_event_callback,
            )
            sub["nonexpert_fc_ms"] = (time.perf_counter() - _t_ne0) * 1000.0
            nonexpert_fired = True
        elif d2h_event_callback is not None:
            d2h_event_callback(None)

        self.last_tick_phase_ms = sub
        return fired_experts, nonexpert_fired

    # ------------------------------------------------------------------
    def _collect_param_payload(
        self,
        names: List[str],
        named_params: Optional[Dict[str, torch.nn.Parameter]] = None,
        *,
        ready_event: Optional["torch.cuda.Event"] = None,
        header_block_bytes: int = 0,
        chunk_pad_bytes: int = 0,
        preserve_order: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Snapshot FP32 master + FP32 Adam state for each param.

        Per-source path: launches one non_blocking D2H per source straight
        into a pinned host buffer, returns payload-dict views into that
        buffer. ``cuda_event`` is recorded on d2h_stream and stashed in
        thread-local metadata so the caller can forward it to the writer
        (which must wait on it before reading the host data).
        If ``ready_event`` is provided, the producer/D2H stream waits on it
        device-side. The worker thread must not host-sync on the event.

        ``header_block_bytes`` (≥ 0, multiple of 4): reserve a leading
        region inside the pinned buffer for a file-format header. D2H
        copies land at byte offset ``header_block_bytes``, leaving the
        front of the buffer untouched for the caller to stamp the header
        later (writer thread, after the d2h event). Must be a multiple of
        4 because the pinned buffer dtype is fp32.

        ``chunk_pad_bytes`` (≥ 0): if non-zero, round the pinned alloc
        up to a multiple of this byte count. Used to pre-shape pinned
        buffers for SSD passthrough (io_uring + O_DIRECT) — the C++
        writer manages LBA in chunk-sized blocks, so emitting an
        already-padded buffer lets it submit a single chunk-aligned
        write. The on-disk file is still written at actual size only.

        Pinned info for the caller is stashed on
        ``self._last_collect_pinned_info`` as a dict with keys
        ``pinned``, ``header_block_bytes``, ``payload_offset_bytes``,
        ``payload_bytes``, ``total_pinned_bytes``.
        """
        if header_block_bytes < 0 or (header_block_bytes & 3):
            raise ValueError(
                f"header_block_bytes must be a non-negative multiple of 4, "
                f"got {header_block_bytes}"
            )
        if chunk_pad_bytes < 0:
            raise ValueError(
                f"chunk_pad_bytes must be non-negative, got {chunk_pad_bytes}"
            )
        # Optimization B: cache the per-name (FP32 master view) entries
        # + plan + total_numel. Each iter's enum loop calls
        # ``_fp32_param`` and ``_fp32_opt_state`` on 130 params × 3 keys
        # = 390 mapper lookups + ``narrow().view()`` view creations
        # (~10 µs each → 4 ms enum cost). The returned views share storage
        # with the DS FP32 master flat which is stable post-init —
        # caching the views is invariant. Cache key includes ``names``
        # tuple identity and fixed pinned-layout parameters.
        _t_enum0 = time.perf_counter()
        cache_key = (
            id(names), len(names),
            int(header_block_bytes), int(chunk_pad_bytes),
            bool(preserve_order),
        )
        # Store one entry per non-expert collect shape.
        if not hasattr(self, "_collect_enum_cache_by_key"):
            self._collect_enum_cache_by_key: Dict[Any, Dict[str, Any]] = {}
        # ``preserve_order=True`` (LFCL path): bypass the cache entirely.
        # Reason: an expert that has not yet received any token has no Adam
        # state. We reserve a zero-filled pinned destination slot so the
        # on-disk layout still matches the schema. Caching that zero slot
        # would be wrong once real Adam state appears for the expert.
        cached = (
            None if preserve_order
            else self._collect_enum_cache_by_key.get(cache_key)
        )
        if cached is not None:
            entries = cached["entries"]
            plan = cached["plan"]
            total_numel = cached["total_numel"]
            device = cached["device"]
        else:
            # Cache miss — only here do we need the named_params dict.
            if named_params is None:
                named_params = self._named_params_cache
                if named_params is None:
                    named_params = dict(self.model.named_parameters())
                    self._named_params_cache = named_params
            entries: List[Tuple[str, Optional[torch.Tensor], torch.Size, int]] = []
            for name in names:
                p = named_params[name]
                src = _fp32_param_live_view(p).detach()
                entries.append((f"{name}:param", src, src.shape, int(src.numel())))
                for key in ("exp_avg", "exp_avg_sq"):
                    v = _fp32_opt_state_live_view(self.optimizer, p, key)
                    if v is None:
                        if preserve_order:
                            # LFCL path: reserve a zero slot in the pinned
                            # destination. Do not allocate a GPU zero tensor;
                            # missing Adam state is logically zero and the
                            # file layout still needs the param|m|v slot.
                            entries.append((
                                f"{name}:{key}",
                                None,
                                torch.Size(p.shape),
                                int(p.numel()),
                            ))
                        # Non-LFCL path: skip the entry as before.
                        continue
                    src_v = v.detach()
                    entries.append((
                        f"{name}:{key}",
                        src_v,
                        src_v.shape,
                        int(src_v.numel()),
                    ))
            if not entries:
                _enum_ms = (time.perf_counter() - _t_enum0) * 1000.0
                self._last_collect_sub_ms = {"enum_ms": _enum_ms, "copy_ms": 0.0, "xfer_ms": 0.0, "wait_ms": 0.0, "nentries": 0}
                return {}, 0
            # Sort entries by (storage_data_ptr, storage_offset) so that
            # entries living in the SAME underlying flat tensor and at
            # contiguous offsets become adjacent → enables the merged
            # cudaMemcpyAsync path below. Stable sort preserves
            # within-storage ordering when offsets coincide.
            #
            # ``preserve_order=True`` (LFCL path): skip the sort so that
            # the on-disk byte order matches the caller-provided ``names``
            # iteration order (p|m|v per name). Trade-off: more
            # D2H launches per expert (typically a handful), no
            # measurable cost since expert collects are tiny.
            def _sort_key(it):
                _k, t, _shape, _n = it
                if t is None:
                    return (0, 0)
                try:
                    return (t.untyped_storage().data_ptr(), t.storage_offset())
                except Exception:
                    return (0, 0)
            if not preserve_order:
                entries = sorted(entries, key=_sort_key)
            plan: List[Tuple[str, int, int, torch.Size]] = []
            offset = 0
            for key, _src, shape, n in entries:
                n = int(n)
                plan.append((key, offset, n, shape))
                offset += n
            total_numel = offset
            first_src = next((src for _k, src, _shape, _n in entries if src is not None), None)
            if first_src is None:
                raise RuntimeError("_collect_param_payload: no GPU source entries")
            device = first_src.device
            if not preserve_order:
                with self._collect_cache_lock:
                    self._collect_enum_cache_by_key[cache_key] = {
                        "entries": entries,
                        "plan": plan,
                        "total_numel": total_numel,
                        "device": device,
                    }
        _enum_ms = (time.perf_counter() - _t_enum0) * 1000.0
        if not entries:
            self._last_collect_sub_ms = {"enum_ms": _enum_ms, "copy_ms": 0.0, "xfer_ms": 0.0, "wait_ms": 0.0, "nentries": 0}
            return {}, 0

        # GPU stage UNUSED: per-source D2H directly to pinned host narrow
        # views. Saves ~total_numel × 4 bytes persistent GPU stage
        # (e.g. 1163 MB for full FC of non-expert + Adam state).
        _copy_ms = 0.0

        # No buffer-reuse barrier needed: the non-expert worker is the
        # sole caller and processes tasks FIFO, so by
        # the time this method runs the previous _do_save has already
        # returned (the worker only pulls the next task after
        # ``queue.task_done`` for the current one). Calling
        # ``drain_non_expert`` here would self-join the worker's own
        # queue and deadlock.
        _wait_ms = 0.0

        # SSD-passthrough pinned layout: optional 4-KiB-aligned header
        # region at the front, payload contiguous after it, optional
        # chunk-pad at the tail. All sizes are in fp32 elements internally
        # but exposed back to the caller as bytes via
        # ``_last_collect_pinned_info``.
        header_numel = int(header_block_bytes) // 4
        payload_bytes = int(total_numel) * 4
        total_with_header_bytes = int(header_block_bytes) + payload_bytes
        if chunk_pad_bytes > 0:
            padded_bytes = (
                (total_with_header_bytes + int(chunk_pad_bytes) - 1)
                // int(chunk_pad_bytes)
            ) * int(chunk_pad_bytes)
        else:
            padded_bytes = total_with_header_bytes
        if padded_bytes & 3:
            # All allocs are in fp32 elements — must be 4-byte multiple.
            padded_bytes = (padded_bytes + 3) & ~3
        pinned_numel = padded_bytes // 4

        use_nonexpert_ring = (
            self.cp_target == "disk"
            and self._checkpoint_workers is not None
            and len(self._collect_nonexpert_pinned_ring) > 1
        )
        producer_ready_event = ready_event
        ready_event_for_writer: Optional[threading.Event] = None
        ring_d2h_event: Optional["torch.cuda.Event"] = None
        if use_nonexpert_ring:
            ring = self._collect_nonexpert_pinned_ring
            slot = int(self._collect_nonexpert_pinned_cursor) % len(ring)
            ready_event_for_writer = self._collect_nonexpert_pinned_ready[slot]
            ready_event_for_writer.wait()
            ready_event_for_writer.clear()
            if slot < len(self._collect_nonexpert_d2h_events):
                ring_d2h_event = self._collect_nonexpert_d2h_events[slot]
            pinned = ring[slot]
            if pinned is None:
                init_numel = int(max(pinned_numel, pinned_numel * 1.5))
                pinned = _alloc_sector_aligned_pinned_fp32(init_numel)
                ring[slot] = pinned
            elif pinned.numel() < pinned_numel:
                init_numel = int(max(pinned_numel, pinned_numel * 1.5))
                pinned = _alloc_sector_aligned_pinned_fp32(init_numel)
                ring[slot] = pinned
            self._collect_nonexpert_pinned_cursor += 1
        else:
            # Phase 5: NE-FC pinned is allocated ONCE on first call with
            # 1.5× headroom over that first iter's footprint. Subsequent
            # iters MUST fit (we raise if not — the FP32 master + Adam
            # state is fixed at init, so first-iter sizing is exact).
            pinned = self._collect_pinned_cpu
            if pinned is None:
                init_numel = int(pinned_numel * 1.5)
                self._collect_pinned_cpu = _alloc_sector_aligned_pinned_fp32(
                    init_numel,
                )
                pinned = self._collect_pinned_cpu
            elif pinned.numel() < pinned_numel:
                raise RuntimeError(
                    f"_collect_pinned_cpu: pre-allocated capacity "
                    f"{pinned.numel()} < required {pinned_numel}. NE-FC "
                    f"payload size grew after first iter — should not "
                    f"happen with fixed FP32 master + Adam layout."
                )

        _t_xfer0 = time.perf_counter()
        # ``flat_cpu`` is the payload region only — D2H lands here. The
        # header region (pinned[:header_numel]) is left untouched for
        # the caller to stamp later on the writer thread.
        flat_cpu = pinned.narrow(0, header_numel, total_numel)

        # Optimization B (continued): cache dsts + srcs_resolved (already
        # cast to FP32 if needed) + payload_template. All stable while
        # ``pinned`` and ``entries`` don't rebuild. Per-iter dtype check
        # eliminated by capturing the resolved srcs in cache.
        xfer_cache_key = (
            cache_key, pinned.data_ptr(), total_numel, header_numel,
        )
        if not hasattr(self, "_collect_xfer_cache_by_key"):
            self._collect_xfer_cache_by_key: Dict[Any, Dict[str, Any]] = {}
        xfer_cached = (
            None if preserve_order
            else self._collect_xfer_cache_by_key.get(xfer_cache_key)
        )
        if xfer_cached is not None:
            dsts = xfer_cached["dsts"]
            srcs_resolved = xfer_cached["srcs_resolved"]
            payload_template = xfer_cached["payload_template"]
            merged_dsts = xfer_cached["merged_dsts"]
            merged_srcs = xfer_cached["merged_srcs"]
        else:
            dsts: List[torch.Tensor] = []
            srcs_flat: List[torch.Tensor] = []
            copy_items: List[Tuple[str, int, int, torch.Size, torch.Tensor]] = []
            payload_template: Dict[str, torch.Tensor] = {}
            for (key, off, n, shape), (_k2, src, _shape2, _n2) in zip(plan, entries):
                dst = flat_cpu.narrow(0, off, n)
                payload_template[key] = dst.view(shape)
                if src is None:
                    dst.zero_()
                    continue
                dsts.append(dst)
                srcs_flat.append(src.reshape(-1))
                copy_items.append((key, off, n, shape, src))
            if all(s.dtype == torch.float32 for s in srcs_flat):
                srcs_resolved = srcs_flat
            else:
                srcs_resolved = [
                    s if s.dtype == torch.float32 else s.to(torch.float32)
                    for s in srcs_flat
                ]
            # === MERGED COPY PATH ====================================
            # Group consecutive entries whose source is contiguous in
            # the SAME underlying storage. Each run becomes a SINGLE
            # cudaMemcpyAsync (390 small launches → ~6 big launches in
            # the typical MoE layout where non-expert params are packed
            # into 1-2 DS group flat tensors). dst side is always
            # contiguous by construction (plan offsets are sequential).
            merged_dsts: List[torch.Tensor] = []
            merged_srcs: List[torch.Tensor] = []
            run_storage = None       # underlying storage data_ptr
            run_src_offset = -1      # next expected element offset in source storage
            run_dst_offset = -1      # starting plan offset of current run
            run_total_n = 0          # accumulated elements in current run
            def _flush_run():
                nonlocal run_total_n, run_storage, run_dst_offset
                if run_total_n <= 0 or run_storage is None:
                    return
                # Use storage of the original src tensor to build a flat view.
                # All entries in this run share the same storage; we use
                # the first entry's src.untyped_storage().
                # Rebuild via narrow on a flat view of the storage.
                start_off = run_dst_offset
                length = run_total_n
                merged_dsts.append(flat_cpu.narrow(0, start_off, length))
                # src_view via storage offset:
                merged_srcs.append(run_storage.narrow(0, run_src_offset - length, length))
            for key, off, n, shape, src in copy_items:
                src_resolved = src
                if src_resolved.dtype != torch.float32:
                    # Non-FP32 source can't be merged into the flat-storage path
                    # (precision conversion required). Flush current run, treat
                    # this entry alone via the small-copy fallback.
                    _flush_run()
                    run_storage = None; run_src_offset = -1; run_total_n = 0
                    # Append as a singleton merge entry pointing at the
                    # converted tensor (still 1 cudaMemcpyAsync — equivalent
                    # to small-copy path for this entry).
                    converted = src_resolved.to(torch.float32).reshape(-1)
                    merged_dsts.append(flat_cpu.narrow(0, off, n))
                    merged_srcs.append(converted)
                    continue
                src_view = src_resolved.reshape(-1)
                storage = src_view.untyped_storage()
                # element offset of src_view's first element inside storage
                src_elem_offset = src_view.storage_offset()
                # underlying flat-view tensor sharing the storage (FP32)
                # used so we can later .narrow(0, off, n) cleanly
                flat_storage_view = torch.empty(0, dtype=torch.float32,
                                                device=src_view.device).set_(
                    storage, 0, (storage.nbytes() // 4,), (1,))
                if (run_storage is not None
                        and storage.data_ptr() == run_storage.untyped_storage().data_ptr()
                        and src_elem_offset == run_src_offset):
                    # Extend current run
                    run_total_n += n
                    run_src_offset += n
                else:
                    _flush_run()
                    run_storage = flat_storage_view
                    run_src_offset = src_elem_offset + n
                    run_dst_offset = off
                    run_total_n = n
            _flush_run()
            if not preserve_order:
                with self._collect_cache_lock:
                    self._collect_xfer_cache_by_key[xfer_cache_key] = {
                        "dsts": dsts,
                        "srcs_resolved": srcs_resolved,
                        "payload_template": payload_template,
                        "merged_dsts": merged_dsts,
                        "merged_srcs": merged_srcs,
                    }

        cuda_event: Optional["torch.cuda.Event"] = None
        read_event: Optional["torch.cuda.Event"] = None
        d2h_copy_cmds = _count_d2h_copies(merged_srcs)
        if torch.cuda.is_available():
            d2h_stream = get_d2h_stream()
            d2h_event = ring_d2h_event if ring_d2h_event is not None else self._collect_d2h_event
            if d2h_event is None:
                raise RuntimeError("FCManager collect events were not initialized")
            main_event = self._collect_main_event
            if main_event is None:
                raise RuntimeError("FCManager collect events were not initialized")
            if producer_ready_event is None:
                main_event.record()
            with torch.cuda.stream(d2h_stream):
                if producer_ready_event is not None:
                    d2h_stream.wait_event(producer_ready_event)
                else:
                    d2h_stream.wait_event(main_event)
                _launch_d2h_copies(merged_dsts, merged_srcs)
                cuda_event = d2h_event
                cuda_event.record(d2h_stream)
            read_event = cuda_event
        else:
            torch._foreach_copy_(merged_dsts, merged_srcs)
        _xfer_ms = (time.perf_counter() - _t_xfer0) * 1000.0

        payload: Dict[str, torch.Tensor] = payload_template
        total_bytes = int(total_numel * 4)
        self._last_collect_sub_ms = {
            "enum_ms": _enum_ms, "copy_ms": _copy_ms,
            "xfer_ms": _xfer_ms, "wait_ms": _wait_ms,
            "nentries": len(entries),
            "d2h_copy_cmds": int(d2h_copy_cmds),
        }
        # Keep thread-local metadata for the non-expert worker.
        if not hasattr(self, "_collect_tls"):
            self._collect_tls = threading.local()
        self._collect_tls.cuda_event = cuda_event
        self._collect_tls.read_event = read_event
        self._collect_tls.pinned_info = {
            "pinned": pinned,
            "header_block_bytes": int(header_block_bytes),
            "payload_offset_bytes": int(header_block_bytes),
            "payload_bytes": int(payload_bytes),
            "total_pinned_bytes": int(padded_bytes),
            "plan": list(plan),
        }
        if ready_event_for_writer is not None:
            self._collect_tls.pinned_info["ready_event"] = ready_event_for_writer
        return payload, total_bytes


    def _collect_nonexpert_payload(
        self,
        *,
        ready_event: Optional["torch.cuda.Event"] = None,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        # named_params=None: the callee will lazily build it ONLY if the
        # enum cache misses. Steady-state (cache hit) skips the
        # ~few-ms dict(model.named_parameters()) walk.
        return self._collect_param_payload(
            self._non_expert_names, None,
            ready_event=ready_event,
            header_block_bytes=binary_io.HEADER_BLOCK_BYTES,
            chunk_pad_bytes=binary_io.PIN_CHUNK_BYTES,
        )

    def _last_collect_pinned_info(self) -> Optional[Dict[str, Any]]:
        """Return the most recent ``_collect_param_payload`` pinned info
        for THIS thread (set by the call inside the same worker). The
        caller reads this immediately after the collect to drive a
        single-write emit from pinned. Returns None if no collect has
        run on this thread."""
        tls = getattr(self, "_collect_tls", None)
        if tls is None:
            return None
        return getattr(tls, "pinned_info", None)

    def _fire_nonexpert_fc(
        self,
        step: int,
        ready_event: Optional["torch.cuda.Event"] = None,
        d2h_event_callback: Optional[Any] = None,
    ) -> None:
        if self.test_skip_nonexpert_d2h:
            self._last_nonexpert_collect_ms = 0.0
            self._last_nonexpert_collect_sub_ms = {
                "enum_ms": 0.0,
                "copy_ms": 0.0,
                "xfer_ms": 0.0,
                "wait_ms": 0.0,
                "nentries": 0,
            }
            if d2h_event_callback is not None:
                d2h_event_callback(None)
            return
        _t_col0 = time.perf_counter()
        payload, payload_bytes = self._collect_nonexpert_payload(
            ready_event=ready_event,
        )
        self._last_nonexpert_collect_ms = (time.perf_counter() - _t_col0) * 1000.0
        self._last_nonexpert_collect_sub_ms = dict(getattr(self, "_last_collect_sub_ms", {}))
        if os.environ.get("PROPOSAL_NONEXPERT_PROFILE", "") in ("1", "true", "TRUE"):
            try:
                every = max(
                    1,
                    int(os.environ.get("PROPOSAL_NONEXPERT_PROFILE_EVERY", "10") or "10"),
                )
            except ValueError:
                every = 10
            if int(step) % every == 0:
                sub = self._last_nonexpert_collect_sub_ms
                print(
                    f"[proposal] nonexpert_profile @iter{int(step)}: "
                    f"payload_bytes={int(payload_bytes)} "
                    f"total={self._last_nonexpert_collect_ms:.1f}ms "
                    f"enum={float(sub.get('enum_ms', 0.0)):.1f} "
                    f"copy={float(sub.get('copy_ms', 0.0)):.1f} "
                    f"xfer={float(sub.get('xfer_ms', 0.0)):.1f} "
                    f"wait={float(sub.get('wait_ms', 0.0)):.1f} "
                    f"nentries={int(sub.get('nentries', 0))} "
                    f"copy_cmds={int(sub.get('d2h_copy_cmds', 0))}",
                    flush=True,
                )
        # Raw-binary format (PRPTENS magic).  Single-rank keeps the historical
        # filename.  Multi-EP disk mode writes one disjoint tensor-name shard per
        # EP replica into the same PP-local non_expert directory.
        if self._nonexpert_shard_count > 1:
            path = self.output_dir / (
                f"fc_nonexpert_ep{self._nonexpert_shard_rank:02d}"
                f"of{self._nonexpert_shard_count:02d}_iter{step:08d}.bin"
            )
        else:
            path = self.output_dir / f"fc_nonexpert_iter{step:08d}.bin"

        to_unlink: List[Path] = []
        self._nonexpert_fc_files.append((path, None))
        while len(self._nonexpert_fc_files) > self.keep_last_nonexpert:
            old_path, _ = self._nonexpert_fc_files.pop(0)
            to_unlink.append(old_path)

        def _unlink(paths: List[Path] = to_unlink) -> None:
            for p in paths:
                binary_io.unlink_path(p)

        stats = FCEventStats(
            kind="nonexpert",
            step=int(step),
            expert_id=None,
            tensor_names=list(self._non_expert_names),
            path=path,
            payload_bytes=payload_bytes,
            async_mode=self._checkpoint_workers is not None,
        )

        tls = getattr(self, "_collect_tls", None)
        read_event = getattr(tls, "read_event", None) if tls is not None else None
        if d2h_event_callback is not None:
            d2h_event_callback(read_event)
        pinned_info = self._last_collect_pinned_info()
        if pinned_info is None:
            raise RuntimeError(
                "non-expert FC: pinned_info missing — collect did not stash "
                "the pinned buffer metadata (thread-local lost?)"
            )
        rec = WriteRecord(tag="fc", path=path, submit_time=time.perf_counter())
        if not self.commit_nonexpert_fc:
            _ready_event = pinned_info.get("ready_event")
            # write-off+D2H-on still exposes the D2H event to the optimizer
            # fence through ``d2h_event_callback`` above.  That fence completes
            # before the next proposal dispatch can reuse this FC slot, so a
            # second host-side cuda_event.synchronize() in the commit worker is
            # pure overhead and can delay the next ring lease in PP runs.
            if _ready_event is not None:
                _ready_event.set()
            rec.queue_delay_ms = 0.0
            rec.write_time_ms = 0.0
            rec.file_bytes = 0
            rec.done = True
            stats.write_time_ms = 0.0
            stats.file_bytes = 0
            stats.queue_delay_ms = 0.0
            self._nonexpert_fc_files[-1] = (path, rec)
            self.write_log.append(stats)
            self._write_records.append(rec)
            return
        # Pre-build the header bytes on the training/owner thread — small
        # CPU work (~tens of µs). The writer thread does the stamp +
        # single os.write after syncing on the D2H read event.
        header_bytes = binary_io.build_prptens_header_block(
            pinned_info["plan"],
            scalars={"step": int(step)},
        )

        def _do_save(
            _path=path, _pinned_info=pinned_info, _header=header_bytes,
            _cuda_event=read_event, _on_complete=_unlink, _rec=rec,
            _stats=stats, _step=int(step), _payload=payload,
        ) -> None:
            t0 = time.perf_counter()
            _rec.queue_delay_ms = (t0 - _rec.submit_time) * 1000.0
            _ready_event = _pinned_info.get("ready_event")

            def _release_ready_event() -> None:
                if _ready_event is not None:
                    try:
                        _ready_event.set()
                    except Exception:
                        pass

            def _finish_durable_write(_fd: int, _write_start: float = t0) -> None:
                try:
                    binary_io.fsync_fd(int(_fd))
                except Exception as exc:
                    _rec.error = f"{type(exc).__name__}: {exc}"
                finally:
                    try:
                        binary_io.close_fd(int(_fd))
                    except Exception:
                        pass
                _rec.write_time_ms = (time.perf_counter() - _write_start) * 1000.0
                _rec.done = True
                _stats.write_time_ms = _rec.write_time_ms
                _stats.file_bytes = _rec.file_bytes
                _stats.queue_delay_ms = _rec.queue_delay_ms
                if _rec.error is None:
                    _on_complete()
                _release_ready_event()

            def _write_from_leased_slot() -> None:
                write_start = time.perf_counter()
                _rec.queue_delay_ms = (write_start - _rec.submit_time) * 1000.0
                try:
                    if _cuda_event is not None:
                        _cuda_event.synchronize()
                    pinned_tensor = _pinned_info["pinned"]
                    payload_bytes_local = _pinned_info["payload_bytes"]
                    _rec.file_bytes = int(binary_io.write_prptens_from_pinned(
                        _path, pinned_tensor, _header, payload_bytes_local,
                        step=_step, fsync=True,
                    ))
                except Exception as exc:
                    _rec.error = f"{type(exc).__name__}: {exc}"
                _rec.write_time_ms = (time.perf_counter() - write_start) * 1000.0
                _rec.done = True
                _stats.write_time_ms = _rec.write_time_ms
                _stats.file_bytes = _rec.file_bytes
                _stats.queue_delay_ms = _rec.queue_delay_ms
                if _rec.error is None:
                    _on_complete()
                _release_ready_event()

            try:
                if (
                    self.cp_target == "disk"
                    and _ready_event is not None
                    and self._checkpoint_workers is not None
                ):
                    self._checkpoint_workers.submit_non_expert_commit(
                        _write_from_leased_slot,
                    )
                    return
                if _cuda_event is not None:
                    _cuda_event.synchronize()
                pinned_tensor = _pinned_info["pinned"]
                payload_bytes_local = _pinned_info["payload_bytes"]
                if self.cp_target == "disk":
                    fd, padded_bytes = binary_io.write_prptens_from_pinned_keep_open(
                        _path, pinned_tensor, _header, payload_bytes_local,
                        step=_step,
                    )
                    _rec.file_bytes = int(padded_bytes)
                    if self._checkpoint_workers is not None:
                        self._checkpoint_workers.submit_non_expert_commit(
                            _finish_durable_write, int(fd),
                        )
                    else:
                        _finish_durable_write(int(fd))
                    return
                elif self.cp_target == "test_dram":
                    self._nonexpert_test_dram_payload = dict(_payload)
                    self._nonexpert_test_dram_step = int(_step)
                try:
                    _rec.file_bytes = int(_path.stat().st_size)
                except FileNotFoundError:
                    _rec.file_bytes = 0
            except Exception as exc:
                _rec.error = f"{type(exc).__name__}: {exc}"
            _rec.write_time_ms = (time.perf_counter() - t0) * 1000.0
            _rec.done = True
            _stats.write_time_ms = _rec.write_time_ms
            _stats.file_bytes = _rec.file_bytes
            _stats.queue_delay_ms = _rec.queue_delay_ms
            if _rec.error is None:
                _on_complete()
            _release_ready_event()

        # Run inline. ``_fire_nonexpert_fc`` is invoked either from the
        # host_writer_non_expert worker (new arch) or from the training
        # thread directly (no-writer fallback). In the worker
        # case we already own the CUDA context and the FIFO ordering
        # guarantees the previous fire fully completed; submitting back
        # to our own queue would self-join during ``drain_non_expert``
        # callers downstream, so we just synchronously synchronise on
        # the cuda_event and call save_fn here.
        _do_save()
        self._nonexpert_fc_files[-1] = (path, rec)
        self.write_log.append(stats)
        self._write_records.append(rec)

    # ------------------------------------------------------------------
    def finalize(self) -> None:
        if self._checkpoint_workers is None:
            return
        self._checkpoint_workers.drain_non_expert()

    # ------------------------------------------------------------------
    def write_log_dict(self) -> List[Dict]:
        return [
            {
                "kind": s.kind,
                "step": s.step,
                "expert_id": list(s.expert_id) if s.expert_id is not None else None,
                "tensor_names": s.tensor_names,
                "path": str(s.path),
                "payload_bytes": s.payload_bytes,
                "file_bytes": s.file_bytes,
                "write_time_ms": s.write_time_ms,
                "queue_delay_ms": s.queue_delay_ms,
                "async_mode": s.async_mode,
            }
            for s in self.write_log
        ]

    def read_nonexpert_test_dram_state(self) -> Optional[Dict[str, Any]]:
        """Return the latest in-memory non-expert FC snapshot."""

        if self._nonexpert_test_dram_payload is None:
            return None
        out: Dict[str, Any] = dict(self._nonexpert_test_dram_payload)
        out["_rolling_step"] = torch.tensor(
            int(self._nonexpert_test_dram_step),
            dtype=torch.long,
        )
        return out

    def schedule_debug(self) -> Dict:
        per_expert_cycle = (
            math.ceil(100.0 / max(self.lfcl_fraction_pct, 0.001))
            if self._schedule == "proposal"
            else self.lfcl_interval_iters * max(self._num_layers, 1)
        )
        return {
            "schedule": self._schedule,
            "num_experts_by_layer": dict(self._num_experts_by_layer),
            "num_layers": self._num_layers,
            "num_expert_slots": self._num_expert_slots,
            "lfcl_interval_iters": self.lfcl_interval_iters,
            "nonexpert_fc_W": self.W,
            "lfcl_fraction_pct": self.lfcl_fraction_pct,
            "experts_per_lfcl_event": getattr(self, "_experts_per_lfcl_event", 0),
            "per_expert_lfcl_cycle_iter": per_expert_cycle,
        }

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        for history in self._expert_fc_files.values():
            for path, _rec in history:
                binary_io.unlink_path(path)
        self._expert_fc_files.clear()
        for path, _rec in self._nonexpert_fc_files:
            binary_io.unlink_path(path)
        self._nonexpert_fc_files.clear()
        try:
            self.output_dir.rmdir()
        except OSError:
            pass
