"""Full Checkpoint manager — round-robin F% expert lossless FC + periodic non-expert FC.

DeepSpeed MoE only: expert params live in separated per-expert tensors with
name pattern ``...deepspeed_experts.{idx}...``.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

import math

from .async_writer import AsyncWriter, WriteRecord
from . import proposal_io as binary_io  # facade keeps existing callsites unchanged
from ._ds_state import get_mapper
from .expert_state import ExpertState


def _fp32_param(p: torch.nn.Parameter) -> torch.Tensor:
    m = get_mapper()
    if m is not None:
        return m.get_param(p)
    return p.data.to(torch.float32)


def _fp32_opt_state(
    optimizer: torch.optim.Optimizer,
    p: torch.nn.Parameter,
    key: str,
) -> Optional[torch.Tensor]:
    m = get_mapper()
    if m is not None:
        return m.get_state(p, key)
    return optimizer.state.get(p, {}).get(key)


LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
DS_EXPERT_IDX_RE = re.compile(r"deepspeed_experts\.(\d+)\.")


ExpertId = Tuple[int, int]  # (layer, expert_idx)


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
        expert_fc_D: Optional[int] = None,
        nonexpert_fc_W: int = 10,
        async_writer: Optional[AsyncWriter] = None,
        keep_last_per_expert: int = 1,
        keep_last_nonexpert: int = 1,
        schedule: str = "proposal",
        lossless_fc_fraction_pct: float = 2.0,
        expert_output_dir: Optional[Path] = None,
        fresh_expert_state: bool = True,
        cp_target: str = "disk",
    ) -> None:
        if expert_fc_D is not None and expert_fc_D < 1:
            raise ValueError(f"expert_fc_D must be ≥1, got {expert_fc_D}")
        if nonexpert_fc_W < 1:
            raise ValueError(f"nonexpert_fc_W must be ≥1, got {nonexpert_fc_W}")

        self.model = model
        self.optimizer = optimizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Expert rolling files can live in a separate directory (split CP
        # layout). Default: same as non-expert FC (legacy layout).
        self.expert_output_dir = Path(expert_output_dir) if expert_output_dir is not None else self.output_dir
        self.expert_output_dir.mkdir(parents=True, exist_ok=True)
        # num_experts: keep raw spec; resolve per-layer counts after enumeration.
        self._num_experts_spec = num_experts
        # D is the FC-period — every D iters each expert fires once. Default
        # ties D to the TOTAL number of expert slots (sum across all MoE
        # layers), so staggered schedule fires 1 expert per iter on average.
        self._D_spec = expert_fc_D
        self.W = int(nonexpert_fc_W)
        self._async_writer = async_writer
        self.keep_last_per_expert = int(keep_last_per_expert)
        self.keep_last_nonexpert = int(keep_last_nonexpert)

        self._expert_tensors_by_layer: Dict[int, List[str]] = {}
        self._non_expert_names: List[str] = []
        self._enumerate_params()
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
        # Resolve D now that we know total slot count.
        # - proposal (CLAUDE.md spec): None/0 → 4 (per Key Parameters table).
        #   With F%=2 and 48 experts, one expert gets lossless FC every D=4
        #   iter → per-expert cycle = D × ⌈100/F⌉ = 4 × 50 = 200 iter.
        # - sequential (MoC baseline): None/0 → 1 (fire 1 expert per iter,
        #   matches MoC-System paper spec).
        if self._D_spec is None or int(self._D_spec) <= 0:
            self.D = 1 if schedule == "sequential" else 4
        else:
            self.D = int(self._D_spec)

        if schedule not in ("proposal", "sequential"):
            raise ValueError(
                f"schedule must be 'proposal' or 'sequential', got {schedule!r}"
            )
        self._schedule = schedule

        # Round-robin cursor for the "proposal" schedule: every D iter, the
        # next ⌈num_experts × F / 100⌉ experts in slot order get lossless FC.
        # Full cycle = D × ⌈100 / F⌉ iters (each expert FC'd exactly once).
        self._rr_cursor = 0
        self.lossless_fc_fraction_pct = float(lossless_fc_fraction_pct)
        if self._num_expert_slots > 0 and self.lossless_fc_fraction_pct > 0:
            self._experts_per_lossless_event = max(
                1,
                math.ceil(
                    self._num_expert_slots * self.lossless_fc_fraction_pct / 100.0
                ),
            )
        else:
            self._experts_per_lossless_event = 0

        # Legacy "round_robin" (MoC sequential): LAYER-wise round robin —
        # every D iter, fire all experts of ONE layer. Preserved for MoC
        # baseline compatibility; driven by _layer_rr_tick.
        self._layer_rr_tick = 0

        self._expert_fc_files: Dict[ExpertId, List[Tuple[Path, Optional[WriteRecord]]]] = {}
        self._nonexpert_fc_files: List[Tuple[Path, Optional[WriteRecord]]] = []

        self.write_log: List[FCEventStats] = []
        self._write_records: List[Optional[WriteRecord]] = []

        # Last-tick phase breakdown (diagnostic). Populated by ``tick()``.
        # Keys: nonexpert_fc_ms, n_experts_fired.
        self.last_tick_phase_ms: Dict[str, float] = {}

        # Cached GPU staging buffer for batched _collect_param_payload. Grows
        # to the largest FC payload observed (typically non-expert ≈ 1 GB on
        # 300M-MoE). Reused across DC-snapshot and FC-collect since those
        # never overlap in time.
        self._collect_stage_gpu: Optional[torch.Tensor] = None

        # Cached pinned CPU buffer for fast GPU→CPU transfers. Reused across
        # non-expert FC calls (which all run on the host_writer_non_expert
        # worker so the buffer has a single owner). The expert path passes
        # private=True and gets a fresh pinned allocation per call.
        self._collect_pinned_cpu: Optional[torch.Tensor] = None

        # ``_collect_param_payload`` is invoked from the two domain workers
        # (private=True on host_writer_expert, private=False on
        # host_writer_non_expert) so the per-call enum / xfer caches are
        # touched by two threads. CPython dict ops are GIL-atomic but a
        # concurrent resize during ``__setitem__`` is unsafe; the lock
        # serialises the rare cache-miss writes. Cache hits don't take the
        # lock — they read a stable dict via ``.get()`` which is atomic.
        self._collect_cache_lock = threading.Lock()

        # Per-expert state (lossy FC append-only log + round-robin lossless
        # FC snapshot). See ``expert_state.ExpertState``. Skipped in
        # ``sequential`` mode (MoC-System baseline: only legacy FC files,
        # no element-wise lossy FC).
        # cp_target selects between the disk-backed ExpertState (default)
        # and the host-DRAM-backed DramExpertState. Both implement the same
        # public surface used downstream (lossless_fc / lossy_fc_append /
        # flush / shutdown / total_bytes / lossless_fc_step). Recovery has
        # a separate fast path for DRAM mode — see recovery.recover().
        #
        # ExpertState is created for BOTH schedules. The two modes differ
        # only in (a) which experts fire each iter (see
        # ExpertLosslessFC._select_fired_experts) and (b) whether the
        # importance tracker is exercised (proposal yes, sequential no
        # — see cp_hooks.proposal_score_step). In sequential mode the
        # lossy-FC entry point (lossy_fc_append_batch) is never invoked
        # — the only writer traffic is the per-D-iter layer-RR
        # lossless_fc snapshots. Without an ExpertState the fired
        # payload would be silently dropped, which is exactly the bug
        # that made sequential mode a no-op before this fix.
        if cp_target not in ("disk", "dram"):
            raise ValueError(
                f"cp_target must be 'disk' or 'dram', got {cp_target!r}"
            )
        self.cp_target = cp_target
        named_params = dict(self.model.named_parameters())
        expert_tensor_map: Dict[ExpertId, List[Tuple[str, torch.Size, int]]] = {}
        for eid, tnames in self._separated_tensors_by_slot.items():
            entries: List[Tuple[str, torch.Size, int]] = []
            for tname in tnames:
                p = named_params[tname]
                entries.append((tname, p.shape, int(p.numel())))
            expert_tensor_map[eid] = entries
        if cp_target == "dram":
            from .dram_expert_state import DramExpertState
            self._expert_state = DramExpertState(
                expert_tensor_map, output_dir=self.expert_output_dir,
                fresh=fresh_expert_state, async_writer=self._async_writer,
            )
        else:
            self._expert_state = ExpertState(
                expert_tensor_map, output_dir=self.expert_output_dir,
                fresh=fresh_expert_state, async_writer=self._async_writer,
            )

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

    def _slot_of(self, layer: int, expert_idx: int) -> int:
        return self._slot_index[(layer, expert_idx)]

    def _expert_from_slot(self, slot: int) -> ExpertId:
        return self._all_expert_ids[slot]

    # ------------------------------------------------------------------
    def fire_nonexpert_fc_if_due(self, step: int) -> Tuple[List[ExpertId], bool]:
        """Fire non-expert FC at iteration ``step`` (1-indexed) if due
        (step == 1 or step % W == 0).

        Returns ``(fired_experts, nonexpert_fired)``. NOTE (post-B2):
        expert lossless FC is dispatched OUT-OF-BAND by ExpertLosslessFC
        (separate module, runs on tracker_stream). This method now
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
            self._fire_nonexpert_fc(step)
            sub["nonexpert_fc_ms"] = (time.perf_counter() - _t_ne0) * 1000.0
            nonexpert_fired = True

        self.last_tick_phase_ms = sub
        return fired_experts, nonexpert_fired

    # ------------------------------------------------------------------
    def _collect_param_payload(
        self,
        names: List[str],
        named_params: Optional[Dict[str, torch.nn.Parameter]] = None,
        *, private: bool = False,
        header_block_bytes: int = 0,
        chunk_pad_bytes: int = 0,
        preserve_order: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Snapshot FP32 master + FP32 Adam state for each param.

        Per-source path: launches one non_blocking D2H per source straight
        into a pinned host buffer, returns payload-dict views into that
        buffer. ``cuda_event`` is recorded on d2h_stream and stashed on
        ``self._last_collect_cuda_event`` so the caller can forward it to
        the writer (which must wait on it before reading the host data).

        ``private=False`` (default, non-expert FC path): reuses the cached
        ``self._collect_pinned_cpu`` buffer; previous in-flight torch-queue
        write is drained first.

        ``private=True`` (expert FC path): allocates a FRESH pinned buffer
        for THIS call. Required when several callers share the writer's
        raw worker queue — the worker may not have consumed the previous
        payload yet, so reusing the cached buffer would corrupt it. Torch's
        caching pinned allocator recycles freed buffers, so steady-state
        cost is small.

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
        # tuple identity + ``private`` (private path uses fresh pinned
        # so the dst/payload caches must differ).
        _t_enum0 = time.perf_counter()
        cache_key = (
            id(names), private, len(names),
            int(header_block_bytes), int(chunk_pad_bytes),
            bool(preserve_order),
        )
        # Multi-key cache: non-expert FC (private=False) and expert FC
        # (private=True) hit this function with DIFFERENT cache_keys.
        # A single-entry cache thrashed between them so every call paid
        # the full enum cost (~5-6 ms / call). Storing one entry per
        # cache_key keeps both paths warm. Per-key memory is small (list
        # of tensor views + plan tuples), so the dict adds <100 KB
        # total across all callers.
        if not hasattr(self, "_collect_enum_cache_by_key"):
            self._collect_enum_cache_by_key: Dict[Any, Dict[str, Any]] = {}
        # ``preserve_order=True`` (LFCL path): bypass the cache entirely.
        # Reason: an expert that has not yet received any token has no
        # Adam state (exp_avg/exp_avg_sq are None). We substitute a zero
        # placeholder so the on-disk layout always matches the schema,
        # but on a later iter that same expert may have real Adam state
        # — caching the zero placeholder would then silently write
        # zeros forever. LFCL collects are tiny (~few MB / expert) so
        # the per-call enum + plan rebuild is sub-millisecond.
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
                named_params = dict(self.model.named_parameters())
            entries: List[Tuple[str, torch.Tensor]] = []
            for name in names:
                p = named_params[name]
                src = _fp32_param(p).detach()
                entries.append((f"{name}:param", src))
                for key in ("exp_avg", "exp_avg_sq"):
                    v = _fp32_opt_state(self.optimizer, p, key)
                    if v is None:
                        if preserve_order:
                            # LFCL path: reserve the slot with a GPU
                            # zero placeholder so the on-disk byte
                            # layout is param|m|v per schema. Without
                            # this, the file would be shorter than the
                            # header claims (recovery would over-read).
                            v = torch.zeros(
                                int(p.numel()), dtype=torch.float32,
                                device=src.device,
                            )
                            entries.append((f"{name}:{key}", v))
                        # Non-LFCL path: skip the entry as before.
                        continue
                    entries.append((f"{name}:{key}", v.detach()))
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
                _k, t = it
                try:
                    return (t.untyped_storage().data_ptr(), t.storage_offset())
                except Exception:
                    return (0, 0)
            if not preserve_order:
                entries = sorted(entries, key=_sort_key)
            plan: List[Tuple[str, int, int, torch.Size]] = []
            offset = 0
            for key, t in entries:
                n = int(t.numel())
                plan.append((key, offset, n, t.shape))
                offset += n
            total_numel = offset
            device = entries[0][1].device
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
        # sole caller with private=False and processes tasks FIFO, so by
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

        if private:
            # Fresh pinned buffer for THIS call only — caller (typically
            # ExpertLosslessFC) hands the resulting payload views to the
            # raw-worker queue, where the closure owns the buffer until
            # consumption. Torch's pinned caching allocator recycles
            # freed buffers, so steady-state alloc cost is small.
            pinned = torch.empty(
                pinned_numel, dtype=torch.float32, pin_memory=True,
            )
        else:
            # Phase 5: NE-FC pinned is allocated ONCE on first call with
            # 1.5× headroom over that first iter's footprint. Subsequent
            # iters MUST fit (we raise if not — the FP32 master + Adam
            # state is fixed at init, so first-iter sizing is exact).
            pinned = self._collect_pinned_cpu
            if pinned is None:
                init_numel = int(pinned_numel * 1.5)
                self._collect_pinned_cpu = torch.empty(
                    init_numel, dtype=torch.float32, pin_memory=True,
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
        # Same rationale as the enum cache: bypass for preserve_order
        # (LFCL path) so a placeholder zero src never gets cached and
        # later misused once Adam state exists for that expert.
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
            payload_template: Dict[str, torch.Tensor] = {}
            for (key, off, n, shape), (_k2, src) in zip(plan, entries):
                dsts.append(flat_cpu.narrow(0, off, n))
                srcs_flat.append(src.reshape(-1))
                payload_template[key] = flat_cpu.narrow(0, off, n).view(shape)
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
            for (key, off, n, shape), (_k2, src) in zip(plan, entries):
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
            # One-time diagnostic on cache miss only.
            try:
                ds_pinned = [bool(d.is_pinned()) for d in dsts[:3]]
                ds_contig = [bool(d.is_contiguous()) for d in dsts[:3]]
                src_contig = [bool(s.is_contiguous()) for s in merged_srcs[:3]]
                src_strides = [tuple(s.stride()) for s in merged_srcs[:3]]
                print(f"[ne-diag] dst_pinned[:3]={ds_pinned} dst_contig[:3]={ds_contig} "
                      f"src_contig[:3]={src_contig} src_strides[:3]={src_strides} "
                      f"merged_count={len(merged_srcs)} total_numel={total_numel}",
                      flush=True)
            except Exception:
                pass
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
        if torch.cuda.is_available():
            from .async_writer import get_d2h_stream
            d2h_stream = get_d2h_stream()
            main_event = torch.cuda.Event()
            main_event.record()
            with torch.cuda.stream(d2h_stream):
                d2h_stream.wait_event(main_event)
                torch._foreach_copy_(merged_dsts, merged_srcs, non_blocking=True)
                cuda_event = torch.cuda.Event()
                cuda_event.record(d2h_stream)
        else:
            torch._foreach_copy_(merged_dsts, merged_srcs)
        _xfer_ms = (time.perf_counter() - _t_xfer0) * 1000.0

        payload: Dict[str, torch.Tensor] = payload_template
        total_bytes = int(total_numel * 4)
        self._last_collect_sub_ms = {
            "enum_ms": _enum_ms, "copy_ms": _copy_ms,
            "xfer_ms": _xfer_ms, "wait_ms": _wait_ms,
            "nentries": len(entries),
        }
        # Stash the event + pinned info for the immediate caller's
        # submit(). Stashed in a thread-local because the non-expert
        # worker (private=False) and the expert worker (private=True)
        # can both enter this method concurrently — a single-slot
        # ``self.`` attribute would race between them.
        if not hasattr(self, "_collect_tls"):
            self._collect_tls = threading.local()
        self._collect_tls.cuda_event = cuda_event
        self._collect_tls.pinned_info = {
            "pinned": pinned,
            "header_block_bytes": int(header_block_bytes),
            "payload_offset_bytes": int(header_block_bytes),
            "payload_bytes": int(payload_bytes),
            "total_pinned_bytes": int(padded_bytes),
            "plan": list(plan),
        }
        # Back-compat alias for callers that still read the single-slot
        # ``self._last_collect_cuda_event``. Safe under sequential
        # access from a SINGLE worker thread; concurrent calls from
        # both workers should read the thread-local instead.
        self._last_collect_cuda_event = cuda_event
        return payload, total_bytes


    def _collect_nonexpert_payload(self) -> Tuple[Dict[str, torch.Tensor], int]:
        # named_params=None: the callee will lazily build it ONLY if the
        # enum cache misses. Steady-state (cache hit) skips the
        # ~few-ms dict(model.named_parameters()) walk.
        return self._collect_param_payload(
            self._non_expert_names, None,
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

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _fire_nonexpert_fc(self, step: int) -> None:
        _t_col0 = time.perf_counter()
        payload, payload_bytes = self._collect_nonexpert_payload()
        self._last_nonexpert_collect_ms = (time.perf_counter() - _t_col0) * 1000.0
        self._last_nonexpert_collect_sub_ms = dict(getattr(self, "_last_collect_sub_ms", {}))
        # Raw-binary format (PRPTENS magic).
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
            async_mode=self._async_writer is not None,
        )

        cuda_event = getattr(self, "_last_collect_cuda_event", None)
        pinned_info = self._last_collect_pinned_info()
        if pinned_info is None:
            raise RuntimeError(
                "non-expert FC: pinned_info missing — collect did not stash "
                "the pinned buffer metadata (thread-local lost?)"
            )
        # Pre-build the header bytes on the training/owner thread — small
        # CPU work (~tens of µs). The writer thread does the stamp +
        # single os.write after syncing on cuda_event.
        header_bytes = binary_io.build_prptens_header_block(
            pinned_info["plan"],
            scalars={"step": int(step)},
        )
        rec = WriteRecord(tag="fc", path=path, submit_time=time.perf_counter())

        def _do_save(
            _path=path, _pinned_info=pinned_info, _header=header_bytes,
            _cuda_event=cuda_event, _on_complete=_unlink, _rec=rec,
            _stats=stats, _step=int(step),
        ) -> None:
            t0 = time.perf_counter()
            _rec.queue_delay_ms = (t0 - _rec.submit_time) * 1000.0
            try:
                if _cuda_event is not None:
                    _cuda_event.synchronize()
                pinned_tensor = _pinned_info["pinned"]
                payload_bytes_local = _pinned_info["payload_bytes"]
                # Ablation knob: PROPOSAL_DISK_SKIP_WRITE=1 skips the
                # NE-FC ::write (the 1.2 GB pinned → pagecache memcpy
                # that CLAUDE.md +355 ms diagnosis identified as the
                # dominant L3 polluter). 측정 전용 — 복구 데이터 안 만듦.
                import os as _os_skip
                _skip_write = _os_skip.environ.get(
                    "PROPOSAL_DISK_SKIP_WRITE", "") in ("1", "true", "True")
                if not _skip_write:
                    binary_io.write_prptens_from_pinned(
                        _path, pinned_tensor, _header, payload_bytes_local,
                        step=_step,
                    )
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

        # Run inline. ``_fire_nonexpert_fc`` is invoked either from the
        # host_writer_non_expert worker (new arch) or from the training
        # thread directly (legacy / no-writer fallback). In the worker
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
        if self._async_writer is None:
            return
        self._async_writer.drain_non_expert()

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

    def schedule_debug(self) -> Dict:
        per_expert_cycle = (
            self.D * math.ceil(100.0 / max(self.lossless_fc_fraction_pct, 0.001))
            if self._schedule == "proposal" else self.D * max(self._num_layers, 1)
        )
        return {
            "schedule": self._schedule,
            "num_experts_by_layer": dict(self._num_experts_by_layer),
            "num_layers": self._num_layers,
            "num_expert_slots": self._num_expert_slots,
            "expert_fc_D": self.D,
            "nonexpert_fc_W": self.W,
            "lossless_fc_fraction_pct": self.lossless_fc_fraction_pct,
            "experts_per_lossless_event": getattr(
                self, "_experts_per_lossless_event", 0,
            ),
            "per_expert_lossless_cycle_iter": per_expert_cycle,
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
