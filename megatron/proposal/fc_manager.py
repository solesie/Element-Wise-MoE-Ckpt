"""Full Checkpoint manager — round-robin F% expert lossless FC + periodic non-expert FC.

DeepSpeed MoE only: expert params live in separated per-expert tensors with
name pattern ``...deepspeed_experts.{idx}...``.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch

import math

from .async_writer import AsyncWriter, WriteRecord
from .binary_io import save_binary
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

        # Cached pinned CPU buffer for fast GPU→CPU transfers. Pageable D2H
        # tops out at ~1.2 GB/s (observed 820 ms/GB); pinned achieves
        # ~10-25 GB/s. Reused across FC calls; the writer must finish the
        # previous torch.save (which pickles from this buffer) before we
        # overwrite it — enforced via ``async_writer.drain_torch()`` below.
        self._collect_pinned_cpu: Optional[torch.Tensor] = None

        # Per-expert state (lossy FC append-only log + round-robin lossless
        # FC snapshot). See ``expert_state.ExpertState``. Skipped in
        # ``sequential`` mode (MoC-System baseline: only legacy FC files,
        # no element-wise lossy FC).
        # cp_target selects between the disk-backed ExpertState (default)
        # and the host-DRAM-backed DramExpertState. Both implement the same
        # public surface used downstream (lossless_fc / lossy_fc_append /
        # flush / shutdown / total_bytes / lossless_fc_step). Recovery has
        # a separate fast path for DRAM mode — see recovery.recover().
        # Sequential (MoC) mode does not use ExpertState at all, so the
        # cp_target flag is a no-op for that schedule.
        if cp_target not in ("disk", "dram"):
            raise ValueError(
                f"cp_target must be 'disk' or 'dram', got {cp_target!r}"
            )
        self.cp_target = cp_target
        if self._schedule == "sequential":
            self._expert_state: Optional[Any] = None
        else:
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
    def tick(self, step: int) -> Tuple[List[ExpertId], bool]:
        """Advance FC schedule for iteration ``step`` (1-indexed).

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
        self, names: List[str], named_params: Dict[str, torch.nn.Parameter],
        *, private: bool = False,
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
        """
        _t_enum0 = time.perf_counter()
        entries: List[Tuple[str, torch.Tensor]] = []
        for name in names:
            p = named_params[name]
            src = _fp32_param(p).detach()
            entries.append((f"{name}:param", src))
            for key in ("exp_avg", "exp_avg_sq"):
                v = _fp32_opt_state(self.optimizer, p, key)
                if v is None:
                    continue
                entries.append((f"{name}:{key}", v.detach()))
        _enum_ms = (time.perf_counter() - _t_enum0) * 1000.0
        if not entries:
            self._last_collect_sub_ms = {"enum_ms": _enum_ms, "copy_ms": 0.0, "xfer_ms": 0.0, "wait_ms": 0.0, "nentries": 0}
            return {}, 0

        plan: List[Tuple[str, int, int, torch.Size]] = []
        offset = 0
        for key, t in entries:
            n = int(t.numel())
            plan.append((key, offset, n, t.shape))
            offset += n
        total_numel = offset
        device = entries[0][1].device

        # GPU stage UNUSED: per-source D2H directly to pinned host narrow
        # views. Saves ~total_numel × 4 bytes persistent GPU stage
        # (e.g. 1163 MB for full FC of non-expert + Adam state).
        _copy_ms = 0.0

        _t_wait0 = time.perf_counter()
        if (not private and self._async_writer is not None
                and self._collect_pinned_cpu is not None):
            # Cached-buffer path: previous torch-queue write must finish
            # before we overwrite. Private path allocates fresh, so no
            # drain is required (and the raw queue isn't drained by
            # drain_torch anyway).
            self._async_writer.drain_torch()
        _wait_ms = (time.perf_counter() - _t_wait0) * 1000.0

        if private:
            # Fresh pinned buffer for THIS call only — caller (typically
            # ExpertLosslessFC) hands the resulting payload views to the
            # raw-worker queue, where the closure owns the buffer until
            # consumption. Torch's pinned caching allocator recycles
            # freed buffers, so steady-state alloc cost is small.
            pinned = torch.empty(
                total_numel, dtype=torch.float32, pin_memory=True,
            )
        else:
            pinned = self._collect_pinned_cpu
            if pinned is None or pinned.numel() < total_numel:
                # Grow-only pinned allocation. Cost is paid once (or very
                # rarely) per training run.
                self._collect_pinned_cpu = torch.empty(
                    total_numel, dtype=torch.float32, pin_memory=True,
                )
                pinned = self._collect_pinned_cpu

        _t_xfer0 = time.perf_counter()
        flat_cpu = pinned.narrow(0, 0, total_numel)
        # Per-source D2H directly to pinned narrow views — no GPU stage.
        cuda_event: Optional["torch.cuda.Event"] = None
        if torch.cuda.is_available():
            from .async_writer import get_d2h_stream
            d2h_stream = get_d2h_stream()
            main_event = torch.cuda.Event()
            main_event.record()
            with torch.cuda.stream(d2h_stream):
                d2h_stream.wait_event(main_event)
                for (_k, off, n, _shape), (_k2, src) in zip(plan, entries):
                    dst = flat_cpu.narrow(0, off, n)
                    src_flat = src.reshape(-1)
                    if src_flat.dtype != torch.float32:
                        dst.copy_(src_flat.to(torch.float32),
                                  non_blocking=True)
                    else:
                        dst.copy_(src_flat, non_blocking=True)
                cuda_event = torch.cuda.Event()
                cuda_event.record(d2h_stream)
        else:
            for (_k, off, n, _shape), (_k2, src) in zip(plan, entries):
                flat_cpu.narrow(0, off, n).copy_(
                    src.reshape(-1).to(torch.float32),
                )
        _xfer_ms = (time.perf_counter() - _t_xfer0) * 1000.0

        payload: Dict[str, torch.Tensor] = {}
        for key, off, n, shape in plan:
            payload[key] = flat_cpu.narrow(0, off, n).view(shape)
        total_bytes = int(total_numel * 4)
        self._last_collect_sub_ms = {
            "enum_ms": _enum_ms, "copy_ms": _copy_ms,
            "xfer_ms": _xfer_ms, "wait_ms": _wait_ms,
            "nentries": len(entries),
        }
        # Stash the event for the immediate caller's submit().
        self._last_collect_cuda_event = cuda_event
        return payload, total_bytes


    def _collect_nonexpert_payload(self) -> Tuple[Dict[str, torch.Tensor], int]:
        named_params = dict(self.model.named_parameters())
        return self._collect_param_payload(self._non_expert_names, named_params)

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
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

        stats = FCEventStats(
            kind="nonexpert",
            step=int(step),
            expert_id=None,
            tensor_names=list(self._non_expert_names),
            path=path,
            payload_bytes=payload_bytes,
            async_mode=self._async_writer is not None,
        )

        def _save_binary(payload, p):
            save_binary(p, payload)

        if self._async_writer is None:
            t0 = time.perf_counter()
            _save_binary(payload, path)
            t1 = time.perf_counter()
            stats.write_time_ms = (t1 - t0) * 1000.0
            try:
                stats.file_bytes = int(path.stat().st_size)
            except FileNotFoundError:
                stats.file_bytes = 0
            _unlink()
            rec: Optional[WriteRecord] = None
        else:
            rec = self._async_writer.submit(
                payload, path, tag="fc",
                on_complete=[_unlink] if to_unlink else None,
                save_fn=_save_binary,
                cuda_event=getattr(self, "_last_collect_cuda_event", None),
            )
            self._nonexpert_fc_files[-1] = (path, rec)

        self.write_log.append(stats)
        self._write_records.append(rec)

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
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        self._expert_fc_files.clear()
        for path, _rec in self._nonexpert_fc_files:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._nonexpert_fc_files.clear()
        try:
            self.output_dir.rmdir()
        except OSError:
            pass
