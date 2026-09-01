"""Recovery for the proposal CP pipeline (DeepSpeed MoE separated layout).

Rebuilds (model, optimizer) state at iteration ``t_fault`` by:

  1. Loading latest non-expert FC (``fc_nonexpert_iter{t}.bin``) with t ≤ t_fault.
  2. For each (L, E) expert, loading the most-recent LFCL compaction baseline
     (``fc_expert_L{L:02d}_E{E:02d}.bin``) and replaying its LFSY append
     log (``lossy_expert_L{L:02d}_E{E:02d}.bin``). Fresh FDP logs use
     rank-local GPU replay; legacy/file logs retain the CPU reference path.

``cp_target=test_dram`` bypasses filesystem recovery and consumes the pinned
dense expert image maintained by ``ExpertState`` directly; it performs no LFSY
replay.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from ..expert_id import DS_EXPERT_IDX_RE as _DS_EXPERT_IDX_RE
from ..expert_id import ids_from_name as _ids_from_param_name
from ..refresh.fp32_state import get_mapper


_EXPERT_LOSSLESS_RE = re.compile(r"^fc_expert_L(\d+)_E(\d+)\.bin$")
# Lossy FC append log (discovered for sanity; actual read is driven by
# expert_state.read_expert_state which locates it by convention).
_EXPERT_LOSSY_RE = re.compile(r"^lossy_expert_L(\d+)_E(\d+)\.bin$")
_NONEXPERT_FC_RE = re.compile(r"^fc_nonexpert_iter(\d+)\.bin$")
_NONEXPERT_FC_SHARD_RE = re.compile(
    r"^fc_nonexpert_shard(\d+)of(\d+)_iter(\d+)\.bin$"
)
_NONEXPERT_FC_LEGACY_EP_SHARD_RE = re.compile(
    r"^fc_nonexpert_ep(\d+)of(\d+)_iter(\d+)\.bin$"
)
ExpertId = Tuple[int, int]

# A single dense LFCL already contributes enough max-I/O-sized commands
# (default 256 KiB) to fill the FDP expert ring (default QD 64).  Pairing
# experts also interleaves LFCL/LFSY reads while bounding the extra raw CPU
# buffers held before batch completion.
_FDP_EXPERT_READ_BATCH_SIZE = 2


@dataclass
class RecoveryDiagnostics:
    nonexpert_fc_iter: int = -1
    nonexpert_fc_path: Optional[str] = None
    expert_fc_count: int = 0
    expert_expected_count: int = 0
    expert_missing_count: int = 0
    expert_fc_iters: Dict[ExpertId, int] = field(default_factory=dict)
    min_fc_iter: int = -1
    rolling_experts_loaded: int = 0
    errors_vs_gt: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: Dict[str, float] = field(default_factory=dict)


@dataclass
class _DecodedExpertState:
    """Either a legacy CPU payload or validated direct-GPU recovery plans."""

    payload: Optional[Dict[str, Any]] = None
    lfcl_plan: Optional[Any] = None
    replay_plan: Optional[Any] = None


class _GpuReplayLanePool:
    """Own source lifetimes and back-pressure for two fixed replay lanes."""

    def __init__(self, lanes: Iterable[Tuple[Any, Any, Any]]) -> None:
        self._lanes = tuple(lanes)
        if len(self._lanes) != 2:
            raise RuntimeError("FDP GPU replay requires exactly two lanes")
        for staging, stream, done in self._lanes:
            if (
                not isinstance(staging, torch.Tensor)
                or staging.device.type != "cuda"
                or staging.dtype != torch.uint8
                or not staging.is_contiguous()
            ):
                raise ValueError("invalid FDP GPU replay staging buffer")
            if not callable(getattr(stream, "synchronize", None)):
                raise ValueError("invalid FDP GPU replay stream")
            if not callable(getattr(done, "record", None)) or not callable(
                getattr(done, "synchronize", None)
            ):
                raise ValueError("invalid FDP GPU replay event")
        self._busy = [False, False]
        self._recorded = [False, False]
        self._keepalive: List[Optional[Any]] = [None, None]

    def acquire(self, index: int, keepalive: Any) -> Tuple[Any, Any]:
        self.wait(index)
        self._busy[index] = True
        self._recorded[index] = False
        self._keepalive[index] = keepalive
        staging, stream, _done = self._lanes[index]
        return staging, stream

    def record(self, index: int) -> None:
        _staging, stream, done = self._lanes[index]
        done.record(stream)
        self._recorded[index] = True

    def wait(self, index: int) -> None:
        if not self._busy[index]:
            return
        _staging, stream, done = self._lanes[index]
        if self._recorded[index]:
            done.synchronize()
        else:
            stream.synchronize()
        self._keepalive[index] = None
        self._busy[index] = False
        self._recorded[index] = False

    def drain(self) -> None:
        first_error: Optional[Exception] = None
        for index in range(2):
            try:
                self.wait(index)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def _scan_cp_dir(
    cp_dir: Path,
    *,
    nonexpert_dir: Optional[Path] = None,
    expert_dir: Optional[Path] = None,
) -> Tuple[
    List[Tuple[int, Path]],
    Dict[ExpertId, Tuple[Path, Path]],
]:
    """Discover all CP artifacts in ``cp_dir``.

    Returns a 3-tuple:
      - ``nonexpert_fcs``: non-expert full/shard FC files.  Sharded
        multi-EP files share the same iter and are all applied at recovery.
      - ``expert_pairs``: per-expert ``(lossless_path, lossy_path)`` pair.
        Either path may point at a non-existent file if that half hasn't
        been written yet — ``read_expert_state`` tolerates missing files.
    """
    scan_dirs = [
        Path(nonexpert_dir) if nonexpert_dir is not None else cp_dir / "non_expert",
    ]
    scan_dirs.append(Path(expert_dir) if expert_dir is not None else cp_dir / "expert")
    expert_new: Dict[ExpertId, Dict[str, Path]] = {}
    nonexpert_fcs: List[Tuple[int, Path]] = []

    for d in scan_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_dir():
                continue
            m = _EXPERT_LOSSLESS_RE.match(f.name)
            if m is not None:
                L, E = int(m.group(1)), int(m.group(2))
                expert_new.setdefault((L, E), {})["lossless"] = f
                continue
            m = _EXPERT_LOSSY_RE.match(f.name)
            if m is not None:
                L, E = int(m.group(1)), int(m.group(2))
                expert_new.setdefault((L, E), {})["lossy"] = f
                continue
            m = _NONEXPERT_FC_RE.match(f.name)
            if m is not None:
                nonexpert_fcs.append((int(m.group(1)), f))
                continue
            m = _NONEXPERT_FC_SHARD_RE.match(f.name)
            if m is not None:
                nonexpert_fcs.append((int(m.group(3)), f))
                continue
            m = _NONEXPERT_FC_LEGACY_EP_SHARD_RE.match(f.name)
            if m is not None:
                nonexpert_fcs.append((int(m.group(3)), f))

    expert_pairs: Dict[ExpertId, Tuple[Path, Path]] = {}
    for eid, parts in expert_new.items():
        any_path = parts.get("lossless") or parts.get("lossy")
        if any_path is None:
            continue
        expert_dir = any_path.parent
        L, E = eid
        expert_pairs[eid] = (
            expert_dir / f"fc_expert_L{L:02d}_E{E:02d}.bin",
            expert_dir / f"lossy_expert_L{L:02d}_E{E:02d}.bin",
        )
    return nonexpert_fcs, expert_pairs


def _ensure_optimizer_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    """Pre-populate optimizer state for non-mapper backends.

    Skipped entirely when the Fp32StateMapper is active (DeepSpeed BF16/Zero
    paths): in those modes the actual optimizer state lives in the
    mapper-managed FP32 flat buffers (already allocated at init), and
    forcing an extra ``optimizer.state[p]`` slot per param adds a
    redundant BF16 copy of m + v for **every** param. Mapper-aware
    get/set_state paths in this file read and write only the mapper's
    flat buffers, so the optimizer.state[p] duplicate is never used and
    can be safely skipped here.
    """
    if get_mapper() is not None:
        return
    for p in model.parameters():
        if not p.requires_grad:
            continue
        state = optimizer.state.setdefault(p, {})
        state.setdefault("step", torch.tensor(0.0))
        state.setdefault(
            "exp_avg", torch.zeros_like(p, memory_format=torch.preserve_format)
        )
        state.setdefault(
            "exp_avg_sq",
            torch.zeros_like(p, memory_format=torch.preserve_format),
        )


def _classify_params(
    model: torch.nn.Module,
    expert_id_map: Optional[Mapping[str, ExpertId]] = None,
) -> Tuple[Dict[str, torch.nn.Parameter], Dict[ExpertId, Dict[str, torch.nn.Parameter]]]:
    non_expert: Dict[str, torch.nn.Parameter] = {}
    by_expert: Dict[ExpertId, Dict[str, torch.nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ids = (
            expert_id_map.get(name)
            if expert_id_map is not None
            else _ids_from_param_name(name)
        )
        if (
            expert_id_map is not None
            and ids is None
            and _DS_EXPERT_IDX_RE.search(name) is not None
        ):
            raise RuntimeError(
                f"expert parameter {name!r} is missing from the canonical "
                "layer/expert mapping"
            )
        if ids is None:
            non_expert[name] = p
        else:
            by_expert.setdefault(ids, {})[name] = p
    return non_expert, by_expert


def _partition_owned_expert_params(
    by_expert: Dict[ExpertId, Dict[str, torch.nn.Parameter]],
    expected_expert_ids: Optional[Iterable[ExpertId]],
) -> Dict[ExpertId, Dict[str, torch.nn.Parameter]]:
    """Restrict recovery accounting to this rank's optimizer ownership.

    The low-precision model still enumerates every logical expert under
    ZeRO-1/2, while LFCL/LFSY stores only the FP32 master/Adam fragments owned
    by the current DP rank.  Treating the model enumeration as the recovery
    denominator therefore reports valid, unowned experts as missing.
    """

    if expected_expert_ids is None:
        return dict(by_expert)
    owned = {
        (int(expert_id[0]), int(expert_id[1]))
        for expert_id in expected_expert_ids
    }
    return {
        expert_id: params
        for expert_id, params in by_expert.items()
        if expert_id in owned
    }


def _apply_payload_dict_to_model(
    data: Dict[str, Any],
    target_params: Dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    sync_lp: bool = True,
) -> int:
    """Scatter a pre-read CPU payload dict into model + optimizer state."""
    mapper = get_mapper()
    rolling_step = int(data.get("_rolling_step", 0)) if isinstance(data, dict) else 0
    if str(data.get("_payload_kind", "")) == "zero12_expert_partition":
        return _apply_zero12_partition_payload(
            data,
            {
                "fragments": list(data.get("_fragments", []) or []),
                "step": int(rolling_step),
            },
            sync_lp=sync_lp,
        )
    for key, tensor in data.items():
        if not isinstance(key, str) or ":" not in key:
            continue
        name, field_name = key.rsplit(":", 1)
        p = target_params.get(name)
        if p is None:
            continue
        value = tensor.to(device=p.device, dtype=torch.float32, non_blocking=True)
        if field_name == "param":
            if mapper is not None:
                mapper.set_param(p, value)
            else:
                p.data.copy_(value.to(dtype=p.dtype), non_blocking=True)
        elif field_name in ("exp_avg", "exp_avg_sq"):
            if mapper is not None:
                mapper.set_state(p, value, field_name)
            else:
                state = optimizer.state.setdefault(p, {})
                slot = state.get(field_name)
                if slot is None:
                    slot = torch.zeros_like(p, dtype=torch.float32)
                    state[field_name] = slot
                slot.copy_(value.to(dtype=slot.dtype), non_blocking=True)
    return rolling_step


def _build_gpu_replay_targets(
    schema_plan: Any,
    target_params: Dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> List[Any]:
    """Map expert-local disk intervals to resident rank-owned GPU state."""

    from .gpu_replay import ReplayTarget

    metadata = dict(schema_plan.metadata)
    entries = list(metadata.get("tensors", []) or [])
    mapper = get_mapper()
    targets: List[Any] = []
    cursor = 0

    if str(metadata.get("payload_kind", "")) == "zero12_expert_partition":
        if mapper is None:
            raise RuntimeError("ZeRO partition GPU replay requires FP32 mapper")
        fragments = list(metadata.get("fragments", []) or [])
        if len(fragments) != len(entries):
            raise RuntimeError(
                "ZeRO partition replay schema/fragment count mismatch: "
                f"entries={len(entries)} fragments={len(fragments)}"
            )
        for entry, fragment in zip(entries, fragments):
            numel = int(entry["numel"])
            if int(fragment.get("numel", numel)) != numel:
                raise RuntimeError("ZeRO partition replay fragment size mismatch")
            group_idx = int(fragment["group_idx"])
            local_offset = int(fragment["local_offset"])
            p = mapper.zero12_partition_view(
                group_idx, local_offset, numel,
            )
            m = mapper.zero12_partition_view(
                group_idx, local_offset, numel, state_key="exp_avg",
            )
            v = mapper.zero12_partition_view(
                group_idx, local_offset, numel, state_key="exp_avg_sq",
            )
            if p is None or m is None or v is None:
                raise RuntimeError(
                    "ZeRO partition replay destination does not match local "
                    f"optimizer layout: group={group_idx} "
                    f"offset={local_offset} n={numel}"
                )
            targets.append(ReplayTarget(cursor, p, m, v))
            cursor += numel
    else:
        for entry in entries:
            name = str(entry["name"])
            numel = int(entry["numel"])
            param = target_params.get(name)
            if param is None:
                raise RuntimeError(f"GPU replay target parameter is missing: {name}")
            if mapper is not None:
                p = mapper.get_local_param_view(param)
                m = mapper.get_local_state_view(param, "exp_avg")
                v = mapper.get_local_state_view(param, "exp_avg_sq")
            else:
                p = param.data
                state = optimizer.state.get(param, {})
                m = state.get("exp_avg")
                v = state.get("exp_avg_sq")
            if p is None or m is None or v is None:
                raise RuntimeError(f"GPU replay destination is unavailable: {name}")
            if any(int(t.numel()) != numel for t in (p, m, v)):
                raise RuntimeError(f"GPU replay destination size mismatch: {name}")
            targets.append(ReplayTarget(cursor, p, m, v))
            cursor += numel

    if cursor != int(schema_plan.total_numel):
        raise RuntimeError(
            "GPU replay schema coverage mismatch: "
            f"covered={cursor} expected={int(schema_plan.total_numel)}"
        )
    return targets


def _apply_zero12_partition_payload(
    data: Dict[str, Any],
    scalars: Dict[str, Any],
    *,
    sync_lp: bool = True,
) -> int:
    owner = scalars.get("owner_global_rank", None)
    if owner is not None:
        cur_rank = (
            int(torch.distributed.get_rank())
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if int(owner) != int(cur_rank):
            return int(scalars.get("step", 0) or 0)
    mapper = get_mapper()
    if mapper is None:
        raise RuntimeError(
            "partition PRPTENS payload requires an active FP32 mapper"
        )
    fragments = list(scalars.get("fragments", []) or [])
    for frag in fragments:
        frag_idx = int(frag["index"])
        group_idx = int(frag["group_idx"])
        local_offset = int(frag["local_offset"])
        for field_name, state_key in (
            ("param", None),
            ("exp_avg", "exp_avg"),
            ("exp_avg_sq", "exp_avg_sq"),
        ):
            tensor = data.get(f"frag{frag_idx}:{field_name}")
            if tensor is None:
                continue
            mapper.copy_zero12_partition_fragment(
                group_idx=group_idx,
                local_offset=local_offset,
                value=tensor,
                state_key=state_key,
            )
    if sync_lp:
        mapper.sync_zero12_fp32_to_lp()
    return int(scalars.get("step", 0) or 0)


def _apply_fc_payload_to_model(
    data: Dict[str, Any],
    scalars: Dict[str, Any],
    target_params: Dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    sync_lp: bool = True,
) -> int:
    if str(scalars.get("payload_kind", "")) == "zero12_nonexpert_partition":
        return _apply_zero12_partition_payload(data, scalars, sync_lp=sync_lp)
    rolling_step = _apply_payload_dict_to_model(
        data, target_params, optimizer, sync_lp=sync_lp
    )
    return int(scalars.get("step", rolling_step) or rolling_step)


def _sync_recovered_zero12_lp() -> None:
    mapper = get_mapper()
    if mapper is not None:
        mapper.sync_zero12_fp32_to_lp()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    return int(raw)


def _fdp_global_expert_id(local_expert_id: int) -> int:
    if os.environ.get("PROPOSAL_FDP_EXPERT_IDS_GLOBAL", "").lower() in (
        "1", "true", "yes",
    ):
        return int(local_expert_id)
    ep_size = _env_int("PROPOSAL_FDP_EP_SIZE", 1)
    if ep_size <= 1:
        return int(local_expert_id)
    return _env_int("PROPOSAL_FDP_EXPERT_ID_BASE", 0) + int(local_expert_id)


def _fdp_entry_matches_current_topology(entry: Dict[str, Any]) -> bool:
    dp = _env_int("PROPOSAL_FDP_DP_RANK", 0)
    pp = _env_int("PROPOSAL_FDP_PP_RANK", 0)
    ep = _env_int("PROPOSAL_FDP_EP_RANK", 0)
    return (
        int(entry.get("dp_rank", dp)) == dp
        and int(entry.get("pp_rank", pp)) == pp
        and int(entry.get("ep_rank", ep)) == ep
    )


def _nefc_shards_to_read(
    entries: Iterable[Dict[str, Any]], *, partitioned: bool
) -> List[Dict[str, Any]]:
    shards = [dict(entry) for entry in entries]
    if not partitioned:
        return shards
    dp = _env_int("PROPOSAL_FDP_DP_RANK", 0)
    ep = _env_int("PROPOSAL_FDP_EP_RANK", 0)
    owned = [
        entry for entry in shards
        if int(entry.get("dp_rank", dp)) == dp
        and int(entry.get("ep_rank", ep)) == ep
    ]
    if len(owned) != 1:
        raise RuntimeError(
            "FDP recovery: expected exactly one owned partitioned NEFC shard, "
            f"got {len(owned)} for dp={dp}, ep={ep}"
        )
    return owned


def _load_fc_payload_into(
    payload_path: Path,
    target_params: Dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    expert_id: Optional[ExpertId] = None,
    *,
    sync_lp: bool = True,
) -> int:
    """Load FC payload into model + optimizer state.

    ``payload_path`` is either a PRPTENS non-expert FC file or an expert
    directory. For expert directories, ``expert_state.read_expert_state``
    merges the LFCL dense baseline and LFSY append log into a scatter-ready
    payload.
    """
    if payload_path.is_dir():
        if expert_id is None:
            raise ValueError(
                "Directory payload requires expert_id for read_expert_state"
            )
        from .expert_writer import read_expert_state
        data = read_expert_state(payload_path, layer=expert_id[0],
                                  expert=expert_id[1])
    else:
        from .proposal_io import load_binary
        tensor_dict, scalars = load_binary(payload_path)
        data = tensor_dict
        return _apply_fc_payload_to_model(
            data, scalars, target_params, optimizer, sync_lp=sync_lp
        )
    return _apply_payload_dict_to_model(
        data, target_params, optimizer, sync_lp=sync_lp
    )


def _read_fdp_expert_states_batched(
    expert_ids: Iterable[ExpertId],
    *,
    lfcs_by_eid: Dict[ExpertId, Dict[str, Any]],
    lfsy_by_eid: Dict[ExpertId, Dict[str, Any]],
    t_fault: int,
    begin_batch: Any,
    end_batch: Any,
    read_lfcs: Any,
    read_lfsy: Any,
    decode: Any,
    consume_batch: Optional[Callable[[Dict[ExpertId, Any]], None]] = None,
    batch_size: int = _FDP_EXPERT_READ_BATCH_SIZE,
) -> Tuple[Dict[ExpertId, Any], Dict[str, float]]:
    """Read FDP expert blobs with one completion fence per bounded batch.

    Read functions allocate the sector-aligned destination and enqueue raw
    NVMe commands.  ``pending`` retains every destination until ``end_batch``
    confirms completion; decoding starts only after that fence.  A
    ``consume_batch`` callback may retain decoded sources until its own GPU
    completion event, allowing the next bounded read window to overlap.
    """
    import time as _time

    eids = list(expert_ids)
    if not eids:
        return {}, {}
    if int(batch_size) <= 0:
        raise ValueError("FDP expert read batch size must be positive")

    # Reject a future baseline before opening an io_uring batch.
    for layer, expert in eids:
        global_eid = (layer, _fdp_global_expert_id(expert))
        lfcs = lfcs_by_eid.get(global_eid)
        if lfcs is None:
            continue
        lfcs_iter = int(lfcs.get("iter", -1))
        if lfcs_iter > int(t_fault):
            raise RuntimeError(
                "FDP recovery: latest LFCS "
                f"L{layer}/E{expert} iter {lfcs_iter} is newer than "
                f"requested fault step {int(t_fault)}"
            )

    out: Dict[ExpertId, Any] = {}
    batch_count = 0
    command_count = 0
    submit_count = 0
    submit_calls = 0
    logical_bytes = 0
    io_ms = 0.0

    for start in range(0, len(eids), int(batch_size)):
        pending: List[
            Tuple[ExpertId, Optional[torch.Tensor], Optional[torch.Tensor]]
        ] = []
        batch_keepalive: List[torch.Tensor] = []
        batch_open = False
        io_t0 = _time.perf_counter()
        try:
            begin_batch()
            batch_open = True
            for layer, expert in eids[start:start + int(batch_size)]:
                global_eid = (layer, _fdp_global_expert_id(expert))
                lfcs = lfcs_by_eid.get(global_eid)
                lfsy = lfsy_by_eid.get(global_eid)
                lfcl_blob: Optional[torch.Tensor] = None
                lfsy_blob: Optional[torch.Tensor] = None

                if lfcs is not None:
                    nbytes = int(lfcs["logical_bytes"])
                    lfcl_blob = read_lfcs(
                        layer,
                        expert,
                        nbytes,
                        _batch_keepalive=batch_keepalive,
                    )
                    logical_bytes += nbytes
                if lfsy is not None:
                    nbytes = int(lfsy["logical_tail"])
                    lfsy_blob = read_lfsy(
                        layer,
                        expert,
                        nbytes,
                        _batch_keepalive=batch_keepalive,
                    )
                    logical_bytes += nbytes
                pending.append(((layer, expert), lfcl_blob, lfsy_blob))

            stats = dict(end_batch())
            batch_open = False
        except BaseException:
            if batch_open:
                # Drain commands already submitted so their destination
                # tensors cannot be released while the device still owns them.
                try:
                    end_batch()
                except BaseException:
                    pass
            raise

        io_ms += (_time.perf_counter() - io_t0) * 1000.0
        batch_count += 1
        command_count += int(stats.get("commands", 0))
        submit_count += int(stats.get("submitted", 0))
        submit_calls += int(stats.get("submit_calls", 0))
        decoded_batch: Dict[ExpertId, Any] = {}
        for (layer, expert), lfcl_blob, lfsy_blob in pending:
            decoded_batch[(layer, expert)] = decode(
                lfcl_blob,
                lfsy_blob,
                layer=layer,
                expert=expert,
                source="fdp",
            )
        if consume_batch is None:
            out.update(decoded_batch)
        else:
            consume_batch(decoded_batch)

    return out, {
        "fdp_expert_read_batches": float(batch_count),
        "fdp_expert_read_commands": float(command_count),
        "fdp_expert_read_submitted": float(submit_count),
        "fdp_expert_read_submit_calls": float(submit_calls),
        "fdp_expert_read_bytes": float(logical_bytes),
        "fdp_expert_read_io_ms": float(io_ms),
    }


def _recover_from_fdp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    t_fault: int,
    load_expert_fcs: bool,
    expected_expert_ids: Optional[Iterable[ExpertId]],
    expert_id_map: Optional[Mapping[str, ExpertId]],
    gpu_replay_lanes: Optional[Iterable[Tuple[Any, Any, Any]]],
    diag: RecoveryDiagnostics,
    t_all0: float,
) -> RecoveryDiagnostics:
    """Recover from the raw FDP backend's latest metadata image.

    FDP metadata is latest-only in the current implementation. If the latest
    durable object is newer than ``t_fault``, this path fails explicitly rather
    than silently recovering a future state.
    """
    import time as _time

    _phase: Dict[str, float] = {}

    _t0 = _time.perf_counter()
    non_expert_params, by_expert_params = _classify_params(
        model,
        expert_id_map,
    )
    owned_expert_params = _partition_owned_expert_params(
        by_expert_params,
        expected_expert_ids,
    )
    diag.expert_expected_count = len(owned_expert_params)
    from .proposal_io import (
        fdp_begin_expert_read_batch,
        fdp_end_expert_read_batch_stats,
        fdp_read_lfcs_blob,
        fdp_read_lfsy_blob,
        fdp_read_nefc_blob,
        fdp_recovery_entries,
        load_binary_blob,
    )

    entries = fdp_recovery_entries()
    _phase["scan_classify_ms"] = (_time.perf_counter() - _t0) * 1000.0

    _t0 = _time.perf_counter()
    nefc_shards = [dict(e) for e in entries.get("nefc_shards", [])]
    if not nefc_shards:
        nefc_shards = [dict(entries.get("nefc", {}))]
    cur_pp = _env_int("PROPOSAL_FDP_PP_RANK", 0)
    dp_size = _env_int("PROPOSAL_FDP_DP_SIZE", 1)
    ep_size = _env_int("PROPOSAL_FDP_EP_SIZE", 1)
    valid_nefc = [
        e for e in nefc_shards
        if bool(e.get("valid", False))
        and int(e.get("pp_rank", cur_pp)) == int(cur_pp)
        and int(e.get("iter", -1)) <= int(t_fault)
    ]
    if not valid_nefc:
        raise RuntimeError("FDP recovery: no valid NEFC shard in metadata")
    best_ne_iter = max(int(e.get("iter", -1)) for e in valid_nefc)
    best_nefc = [
        e for e in valid_nefc if int(e.get("iter", -1)) == int(best_ne_iter)
    ]
    if dp_size > 1 or ep_size > 1:
        shard_seen = {
            (int(e.get("dp_rank", -1)), int(e.get("ep_rank", -1)))
            for e in best_nefc
        }
        missing = [
            (dp, ep)
            for dp in range(int(dp_size))
            for ep in range(int(ep_size))
            if (dp, ep) not in shard_seen
        ]
        if missing:
            raise RuntimeError(
                f"FDP recovery: missing NEFC shard(s) at iter "
                f"{best_ne_iter}: {missing}"
            )
    applied_steps: List[int] = []
    mapper = get_mapper()
    read_nefc = _nefc_shards_to_read(
        best_nefc,
        partitioned=(
            mapper is not None and getattr(mapper, "mode", "") == "zero12"
        ),
    )
    sorted_nefc = sorted(
        read_nefc,
        key=lambda e: (
            int(e.get("dp_rank", 0)),
            int(e.get("ep_rank", 0)),
            int(e.get("shard_index", 0)),
        ),
    )
    for shard in sorted_nefc:
        shard_index = int(shard.get("shard_index", 0))
        nefc_nbytes = int(shard["logical_bytes"])
        ne_blob = fdp_read_nefc_blob(
            nefc_nbytes,
            shard_index=shard_index,
        )
        ne_payload, ne_scalars = load_binary_blob(
            ne_blob,
            source=f"fdp:nefc_shard{shard_index}",
        )
        applied_steps.append(
            int(_apply_fc_payload_to_model(
                ne_payload, ne_scalars, non_expert_params, optimizer,
                sync_lp=False,
            ))
        )
    diag.nonexpert_fc_iter = int(best_ne_iter)
    diag.nonexpert_fc_path = ",".join(
        f"fdp:nefc_shard{int(e.get('shard_index', 0))}"
        for e in sorted_nefc
    )
    _phase["nonexpert_fc_ms"] = (_time.perf_counter() - _t0) * 1000.0

    log_loaded: Dict[ExpertId, int] = {}
    _t0 = _time.perf_counter()
    if load_expert_fcs:
        from .expert_writer import (
            read_expert_state_from_blobs,
        )
        from .gpu_replay import (
            prepare_lfcl_restore,
            prepare_lfsy_replay,
            replay_lfsy_in_place,
            restore_lfcl_in_place,
        )

        lane_pool = (
            _GpuReplayLanePool(gpu_replay_lanes)
            if gpu_replay_lanes is not None and torch.cuda.is_available()
            else None
        )
        use_gpu_replay = lane_pool is not None

        lfcs_by_eid: Dict[ExpertId, Dict[str, Any]] = {}
        for raw in entries.get("lfcs", []):
            e = dict(raw)
            if not bool(e.get("valid", False)):
                continue
            if not _fdp_entry_matches_current_topology(e):
                continue
            eid = (int(e.get("layer_id", -1)), int(e.get("expert_id", -1)))
            lfcs_by_eid[eid] = e

        lfsy_by_eid: Dict[ExpertId, Dict[str, Any]] = {}
        for raw in entries.get("lfsy", []):
            e = dict(raw)
            if not bool(e.get("valid", False)):
                continue
            if not _fdp_entry_matches_current_topology(e):
                continue
            eid = (int(e.get("layer_id", -1)), int(e.get("expert_id", -1)))
            lfsy_by_eid[eid] = e

        eids_with_fdp_state = [
            eid for eid in owned_expert_params
            if (
                (eid[0], _fdp_global_expert_id(eid[1])) in lfcs_by_eid
                or (eid[0], _fdp_global_expert_id(eid[1])) in lfsy_by_eid
            )
        ]

        def _decode_expert(
            lfcl_blob: Optional[torch.Tensor],
            lfsy_blob: Optional[torch.Tensor],
            *,
            layer: int,
            expert: int,
            source: str,
        ) -> _DecodedExpertState:
            replay_plan = prepare_lfsy_replay(
                lfsy_blob,
                t_fault=int(t_fault),
                source=f"{source}:L{int(layer)}/E{int(expert)}",
            )
            if (
                use_gpu_replay
                and (
                    replay_plan is None
                    or bool(replay_plan.unique_idx_per_block)
                )
            ):
                lfcl_plan = prepare_lfcl_restore(
                    lfcl_blob,
                    expected_metadata=(
                        None if replay_plan is None else replay_plan.metadata
                    ),
                    t_fault=int(t_fault),
                    source=f"{source}:L{int(layer)}/E{int(expert)}",
                )
                if lfcl_plan is None:
                    raise RuntimeError(
                        f"{source}: L{int(layer)}/E{int(expert)} has no "
                        "recoverable LFCL or LFSY schema"
                    )
                return _DecodedExpertState(
                    lfcl_plan=lfcl_plan,
                    replay_plan=replay_plan,
                )
            # Legacy objects lack the per-block uniqueness contract.  Keep the
            # exact CPU reverse/seen implementation for those files.
            return _DecodedExpertState(
                payload=read_expert_state_from_blobs(
                    lfcl_blob,
                    lfsy_blob,
                    layer=int(layer),
                    expert=int(expert),
                    source=source,
                ),
            )

        gpu_replay_blocks = 0
        gpu_replay_records = 0
        gpu_replay_bytes = 0
        gpu_replay_experts = 0
        expert_apply_ms = 0.0
        # Recovery-only timing: retain one event pair per LFSY replay call so
        # the intrinsic pinned-H2D + scatter work can be separated from NVMe
        # reads and the dense LFCL restore.  Events are allocated only during
        # a fault recovery, never on the per-iteration checkpoint hot path.
        gpu_replay_events = [[], []]

        def _apply_decoded_batch(
            decoded_batch: Dict[ExpertId, _DecodedExpertState],
        ) -> None:
            nonlocal gpu_replay_blocks, gpu_replay_records
            nonlocal gpu_replay_bytes, gpu_replay_experts, expert_apply_ms
            apply_t0 = _time.perf_counter()
            legacy_enqueued = False
            try:
                if lane_pool is not None and len(decoded_batch) > 2:
                    raise RuntimeError("FDP replay batch exceeds two GPU lanes")
                for lane_index, (eid, decoded) in enumerate(decoded_batch.items()):
                    if decoded.lfcl_plan is not None:
                        replay_targets = _build_gpu_replay_targets(
                            decoded.lfcl_plan,
                            owned_expert_params[eid],
                            optimizer,
                        )
                        if lane_pool is None:
                            raise RuntimeError(
                                "FDP direct GPU replay has no staging lanes"
                            )
                        staging, stream = lane_pool.acquire(
                            lane_index,
                            (decoded, replay_targets),
                        )
                        try:
                            log_step = restore_lfcl_in_place(
                                decoded.lfcl_plan,
                                replay_targets,
                                stream=stream,
                            )
                            if decoded.replay_plan is not None:
                                replay_start = torch.cuda.Event(enable_timing=True)
                                replay_end = torch.cuda.Event(enable_timing=True)
                                replay_start.record(stream)
                                replay_stats = replay_lfsy_in_place(
                                    decoded.replay_plan,
                                    replay_targets,
                                    staging,
                                    stream=stream,
                                )
                                replay_end.record(stream)
                                gpu_replay_events[lane_index].append(
                                    (replay_start, replay_end)
                                )
                                log_step = max(
                                    log_step,
                                    int(decoded.replay_plan.max_step),
                                )
                                gpu_replay_blocks += int(replay_stats.blocks)
                                gpu_replay_records += int(replay_stats.records)
                                gpu_replay_bytes += int(replay_stats.bytes_staged)
                                gpu_replay_experts += 1
                        finally:
                            lane_pool.record(lane_index)
                    else:
                        if decoded.payload is None:
                            raise RuntimeError(
                                f"FDP recovery: expert {eid} has no payload"
                            )
                        if lane_pool is not None:
                            lane_pool.drain()
                        log_step = _apply_payload_dict_to_model(
                            decoded.payload,
                            owned_expert_params[eid],
                            optimizer,
                            sync_lp=False,
                        )
                        legacy_enqueued = True
                    if log_step > int(t_fault):
                        raise RuntimeError(
                            f"FDP recovery: expert L{eid[0]}/E{eid[1]} "
                            f"recovered step {log_step} is newer than "
                            f"requested fault step {int(t_fault)}"
                        )
                    log_loaded[eid] = int(log_step)
                    diag.expert_fc_iters[eid] = int(log_step)
            finally:
                # Legacy CPU payloads are not retained by a replay lane.
                if legacy_enqueued and torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()
                expert_apply_ms += (
                    _time.perf_counter() - apply_t0
                ) * 1000.0

        _t_read0 = _time.perf_counter()
        try:
            _unused_pair_data, read_batch_stats = _read_fdp_expert_states_batched(
                eids_with_fdp_state,
                lfcs_by_eid=lfcs_by_eid,
                lfsy_by_eid=lfsy_by_eid,
                t_fault=int(t_fault),
                begin_batch=fdp_begin_expert_read_batch,
                end_batch=fdp_end_expert_read_batch_stats,
                read_lfcs=fdp_read_lfcs_blob,
                read_lfsy=fdp_read_lfsy_blob,
                decode=_decode_expert,
                consume_batch=_apply_decoded_batch,
            )
        finally:
            if lane_pool is not None:
                drain_t0 = _time.perf_counter()
                lane_pool.drain()
                expert_apply_ms += (
                    _time.perf_counter() - drain_t0
                ) * 1000.0
        gpu_replay_lane_ms = [
            sum(float(start.elapsed_time(end)) for start, end in events)
            for events in gpu_replay_events
        ]
        expert_pipeline_ms = (_time.perf_counter() - _t_read0) * 1000.0
        _phase["expert_read_ms"] = float(
            read_batch_stats.get("fdp_expert_read_io_ms", 0.0)
        )
        _phase["expert_apply_ms"] = float(expert_apply_ms)
        _phase["expert_pipeline_ms"] = float(expert_pipeline_ms)
        _phase.update(read_batch_stats)
        read_io_ms = read_batch_stats.get("fdp_expert_read_io_ms", 0.0)
        if read_io_ms > 0.0:
            _phase["fdp_expert_read_gib_s"] = (
                read_batch_stats.get("fdp_expert_read_bytes", 0.0)
                / (1024.0 ** 3)
                / (read_io_ms / 1000.0)
            )

        _phase["gpu_replay_experts"] = float(gpu_replay_experts)
        _phase["gpu_replay_blocks"] = float(gpu_replay_blocks)
        _phase["gpu_replay_records"] = float(gpu_replay_records)
        _phase["gpu_replay_bytes"] = float(gpu_replay_bytes)
        _phase["gpu_replay_lanes"] = float(2 if lane_pool is not None else 0)
        _phase["gpu_replay_lane0_ms"] = float(gpu_replay_lane_ms[0])
        _phase["gpu_replay_lane1_ms"] = float(gpu_replay_lane_ms[1])
        _phase["gpu_replay_sum_ms"] = float(sum(gpu_replay_lane_ms))
        _phase["gpu_replay_critical_ms"] = float(max(gpu_replay_lane_ms))

        diag.rolling_experts_loaded = len(log_loaded)
        diag.expert_fc_count = len(log_loaded)
        diag.expert_missing_count = max(
            0,
            diag.expert_expected_count - diag.expert_fc_count,
        )
        if diag.expert_missing_count != 0:
            missing = sorted(set(owned_expert_params) - set(log_loaded))
            raise RuntimeError(
                "FDP recovery: missing rank-owned expert state: "
                f"loaded={diag.expert_fc_count}/"
                f"{diag.expert_expected_count} missing={missing}"
            )
    _phase["expert_load_ms"] = (_time.perf_counter() - _t0) * 1000.0

    diag.min_fc_iter = int(diag.nonexpert_fc_iter)
    _t0 = _time.perf_counter()
    _sync_recovered_zero12_lp()
    _phase["lp_sync_ms"] = (_time.perf_counter() - _t0) * 1000.0
    _phase["total_ms"] = (_time.perf_counter() - t_all0) * 1000.0
    diag.summary.update(_phase)
    _print_recovery_phases(_phase, diag)
    return diag


def recover(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cp_dir: Path,
    t_fault: int,
    num_experts: int = 0,
    load_expert_fcs: bool = True,
    expert_state_image: Optional[Any] = None,
    nonexpert_state_image: Optional[Any] = None,
    nonexpert_dir: Optional[Path] = None,
    expert_dir: Optional[Path] = None,
    expected_expert_ids: Optional[Iterable[ExpertId]] = None,
    expert_id_map: Optional[Mapping[str, ExpertId]] = None,
    gpu_replay_lanes: Optional[Iterable[Tuple[Any, Any, Any]]] = None,
) -> RecoveryDiagnostics:
    """Rebuild state from CP files at ``t_fault``.

    Loads the latest non-expert FC at iter ≤ t_fault and the per-expert
    LFCL + LFSY pair for every (L, E) in the model. No DC replay:
    the gradient-bundle path was incompatible with gradient accumulation
    (``param.grad`` retains only the last micro-batch's grad after
    DeepSpeed's per-mb ``update_hp_grads`` flush) and has been removed in
    favour of W-scheduled non-expert FC every iter, which is exact.

    ``expert_state_image`` / ``nonexpert_state_image``: optional filesystem-off
    diagnostic inputs for cp_target=test_dram. Production recovery scans disk.
    ``expected_expert_ids`` is the current rank's partition-owned expert set;
    when omitted, the unpartitioned model enumeration is used.
    """
    import time as _time
    _phase: Dict[str, float] = {}
    _t_all0 = _time.perf_counter()
    diag = RecoveryDiagnostics()
    _ensure_optimizer_state(model, optimizer)

    if expert_state_image is None and nonexpert_state_image is None:
        from .proposal_io import backend_kind

        if backend_kind() == "fdp":
            return _recover_from_fdp(
                model,
                optimizer,
                t_fault=int(t_fault),
                load_expert_fcs=bool(load_expert_fcs),
                expected_expert_ids=expected_expert_ids,
                expert_id_map=expert_id_map,
                gpu_replay_lanes=gpu_replay_lanes,
                diag=diag,
                t_all0=_t_all0,
            )

    cp_dir = Path(cp_dir)
    _t0 = _time.perf_counter()
    nonexpert_fcs, expert_pairs = _scan_cp_dir(
        cp_dir,
        nonexpert_dir=nonexpert_dir,
        expert_dir=expert_dir,
    )
    non_expert_params, by_expert_params = _classify_params(
        model,
        expert_id_map,
    )
    owned_expert_params = _partition_owned_expert_params(
        by_expert_params,
        expected_expert_ids,
    )
    diag.expert_expected_count = len(owned_expert_params)
    _phase["scan_classify_ms"] = (_time.perf_counter() - _t0) * 1000.0

    # 1. Non-expert FC
    _t0 = _time.perf_counter()
    if nonexpert_state_image is not None:
        if callable(getattr(nonexpert_state_image, "read_nonexpert_test_dram_state", None)):
            ne_payload = nonexpert_state_image.read_nonexpert_test_dram_state()
        else:
            ne_payload = nonexpert_state_image
        if ne_payload is None:
            raise RuntimeError("test_dram non-expert state is empty")
        ne_scalars = dict(ne_payload.get("_prptens_scalars", {}))
        ne_step_raw = ne_payload.get("_rolling_step", -1)
        ne_step = int(ne_step_raw.item()) if torch.is_tensor(ne_step_raw) else int(ne_step_raw)
        if ne_step > int(t_fault):
            raise RuntimeError(
                f"test_dram non-expert state step {ne_step} is newer "
                f"than requested fault step {int(t_fault)}"
            )
        _apply_fc_payload_to_model(
            ne_payload, ne_scalars, non_expert_params, optimizer,
            sync_lp=False,
        )
        diag.nonexpert_fc_iter = int(ne_step)
        diag.nonexpert_fc_path = "test_dram"
    else:
        best_ne_iter: Optional[int] = None
        for it, path in nonexpert_fcs:
            if it > t_fault:
                continue
            if best_ne_iter is None or it > best_ne_iter:
                best_ne_iter = int(it)
        if best_ne_iter is None:
            raise RuntimeError(
                f"No non-expert FC found with iter ≤ {t_fault} in {cp_dir}"
            )
        best_ne_paths = [
            path for it, path in nonexpert_fcs if int(it) == int(best_ne_iter)
        ]
        shard_seen: Dict[int, int] = {}
        shard_count: Optional[int] = None
        for path in best_ne_paths:
            m = _NONEXPERT_FC_SHARD_RE.match(path.name)
            if m is None:
                m = _NONEXPERT_FC_LEGACY_EP_SHARD_RE.match(path.name)
            if m is None:
                continue
            shard = int(m.group(1))
            count = int(m.group(2))
            if shard_count is None:
                shard_count = count
            elif shard_count != count:
                raise RuntimeError(
                    f"Mixed non-expert shard counts at iter {best_ne_iter}: "
                    f"{shard_count} and {count}"
                )
            shard_seen[shard] = count
        if shard_count is not None and len(shard_seen) != int(shard_count):
            missing = [
                i for i in range(int(shard_count)) if i not in shard_seen
            ]
            raise RuntimeError(
                f"Missing non-expert FC shard(s) at iter {best_ne_iter}: "
                f"{missing}"
            )
        for path in sorted(best_ne_paths):
            _load_fc_payload_into(
                path, non_expert_params, optimizer, sync_lp=False
            )
        diag.nonexpert_fc_iter = int(best_ne_iter)
        diag.nonexpert_fc_path = ",".join(str(p) for p in sorted(best_ne_paths))
    _phase["nonexpert_fc_ms"] = (_time.perf_counter() - _t0) * 1000.0

    # 2. Per-expert state. Disk mode loads the V2 ``(LFCL, LFSY)`` pair and
    # applies the merged dense payload, last write wins per expert-local
    # index. test_dram consumes its pinned dense image directly.
    log_loaded: Dict[ExpertId, int] = {}

    _t0 = _time.perf_counter()
    if load_expert_fcs and expert_state_image is not None:
        # test_dram mode: each expert's merged dense (p, m, v) buffer lives in
        # host RAM. LFCL payloads seed the image and lossy AOS appends update it
        # at write time.
        # Recovery is just an O(num_experts × tensors_per_expert × full_size)
        # CPU→GPU copy. We bypass disk scan entirely for the expert path.
        _t_dram0 = _time.perf_counter()
        missing: List[ExpertId] = []
        for (L, E), target in owned_expert_params.items():
            payload = expert_state_image.read((L, E))
            if payload is None:
                missing.append((L, E))
                continue
            log_step = _apply_payload_dict_to_model(
                payload, target, optimizer, sync_lp=False
            )
            if log_step >= 0 and log_step <= t_fault:
                log_loaded[(L, E)] = log_step
                diag.expert_fc_iters[(L, E)] = log_step
        _phase["expert_dram_apply_ms"] = (
            _time.perf_counter() - _t_dram0
        ) * 1000.0
        diag.rolling_experts_loaded = len(log_loaded)
        diag.expert_fc_count = len(log_loaded)
        diag.expert_missing_count = len(missing)
        if diag.expert_fc_count == 0:
            raise RuntimeError(
                "test_dram expert state contained no recoverable experts; "
                "did you call compact_lossy_recovery_image / "
                "append_lossy_records_for_test before recover?"
            )
    elif load_expert_fcs:
        # Phase A (parallel, CPU-bound): read each expert's (LFCL, LFSY)
        # pair via read_expert_state, which uses one aligned read per object
        # and C++ reverse replay. Phase B (serial, GPU) applies the resulting
        # CPU dicts to model/optimizer so we don't race on CUDA stream or
        # mapper state.
        from concurrent.futures import ThreadPoolExecutor
        from .expert_writer import read_expert_state

        eids_with_new_pair: List[ExpertId] = []
        for (L, E), _target in owned_expert_params.items():
            pair = expert_pairs.get((L, E))
            if pair is not None and (pair[0].exists() or pair[1].exists()):
                eids_with_new_pair.append((L, E))

        _t_read0 = _time.perf_counter()
        new_pair_data: Dict[ExpertId, Dict[str, Any]] = {}
        if eids_with_new_pair:
            # Recovery reads whole LFCL/LFSY objects into aligned pinned buffers.
            # Keep concurrency bounded so large logs do not pin many GB at once.
            n_workers = min(
                2,
                max(1, len(eids_with_new_pair)),
            )
            def _read_one(eid: ExpertId) -> Tuple[ExpertId, Dict[str, Any]]:
                L, E = eid
                pair = expert_pairs[eid]
                data = read_expert_state(
                    pair[0].parent,
                    layer=L,
                    expert=E,
                )
                return eid, data

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for eid, data in pool.map(_read_one, eids_with_new_pair):
                    new_pair_data[eid] = data
        _phase["expert_read_ms"] = (_time.perf_counter() - _t_read0) * 1000.0
        diag.expert_missing_count = max(
            0,
            len(owned_expert_params) - len(eids_with_new_pair),
        )

        # Phase B: apply CPU payloads serially.  This keeps recovery's GPU
        # transient lifetime predictable after the one-read CPU replay.
        _t_apply0 = _time.perf_counter()
        for eid in eids_with_new_pair:
            data = new_pair_data[eid]
            target = owned_expert_params[eid]
            log_step = _apply_payload_dict_to_model(
                data, target, optimizer, sync_lp=False
            )
            if log_step <= t_fault:
                log_loaded[eid] = log_step
                diag.expert_fc_iters[eid] = log_step
        _phase["expert_apply_ms"] = (_time.perf_counter() - _t_apply0) * 1000.0

        diag.rolling_experts_loaded = len(log_loaded)
        diag.expert_fc_count = len(log_loaded)
        if diag.expert_fc_count == 0:
            raise RuntimeError(
                f"No expert LFCL/LFSY state found with "
                f"iter ≤ {t_fault} in {expert_dir or cp_dir}"
            )
    _phase["expert_load_ms"] = (_time.perf_counter() - _t0) * 1000.0

    diag.min_fc_iter = int(diag.nonexpert_fc_iter)
    _t0 = _time.perf_counter()
    _sync_recovered_zero12_lp()
    _phase["lp_sync_ms"] = (_time.perf_counter() - _t0) * 1000.0

    _phase["total_ms"] = (_time.perf_counter() - _t_all0) * 1000.0
    diag.summary.update(_phase)
    _print_recovery_phases(_phase, diag)
    return diag


def _print_recovery_phases(phase: Dict[str, float], diag: RecoveryDiagnostics) -> None:
    """Emit a one-line phase breakdown for recover() — used by benchmarks
    to separate scan / non-expert FC load / expert load."""
    fdp_read = ""
    if "fdp_expert_read_gib_s" in phase:
        fdp_read = (
            f"fdp_read={phase['fdp_expert_read_gib_s']:.2f}GiB/s "
            f"cmds={phase.get('fdp_expert_read_commands', 0.0):.0f} "
        )
    gpu_replay = ""
    if phase.get("gpu_replay_experts", 0.0) > 0.0:
        gpu_replay = (
            f"gpu_replay={phase['gpu_replay_experts']:.0f}e/"
            f"{phase.get('gpu_replay_records', 0.0):.0f}r/"
            f"{phase.get('gpu_replay_bytes', 0.0) / (1024.0 ** 3):.2f}GiB "
            f"pure={phase.get('gpu_replay_critical_ms', 0.0):.3f}ms"
            f"(sum={phase.get('gpu_replay_sum_ms', 0.0):.3f},"
            f"lanes={phase.get('gpu_replay_lane0_ms', 0.0):.3f}/"
            f"{phase.get('gpu_replay_lane1_ms', 0.0):.3f}) "
        )
    print(
        f"[recovery] total={phase.get('total_ms', 0.0):.1f}ms  "
        f"scan={phase.get('scan_classify_ms', 0.0):.1f} "
        f"ne_fc={phase.get('nonexpert_fc_ms', 0.0):.1f} "
        f"expert_load={phase.get('expert_load_ms', 0.0):.1f}"
        f"(read={phase.get('expert_read_ms', 0.0):.1f}+apply={phase.get('expert_apply_ms', 0.0):.1f}) "
        f"{fdp_read}"
        f"{gpu_replay}"
        f"lp_sync={phase.get('lp_sync_ms', 0.0):.1f} "
        f"experts={diag.expert_fc_count}/{diag.expert_expected_count} "
        f"missing={diag.expert_missing_count}",
        flush=True,
    )


def compare_states(
    recovered: torch.nn.Module,
    ground_truth: torch.nn.Module,
    num_experts: int = 0,
) -> Dict[str, Dict[str, float]]:
    gt_named = dict(ground_truth.named_parameters())
    errors: Dict[str, Dict[str, float]] = {}
    for name, p in recovered.named_parameters():
        gt = gt_named.get(name)
        if gt is None:
            continue
        r = p.data.to(torch.float32)
        g = gt.data.to(torch.float32)
        diff = (r - g).abs()
        denom = g.abs().mean().clamp_min(1e-12)
        l2_gt = g.norm().clamp_min(1e-12)
        errors[name] = {
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
            "l2_diff": float((r - g).norm().item()),
            "relative_l2": float((r - g).norm().item() / l2_gt.item()),
            "relative_mean": float(diff.mean().item() / denom.item()),
        }
    return errors


def summarize_errors(
    errors: Dict[str, Dict[str, float]],
    num_experts: int = 0,
) -> Dict[str, float]:
    expert_names = [n for n in errors if _DS_EXPERT_IDX_RE.search(n) is not None]
    nonexpert_names = [n for n in errors if n not in set(expert_names)]

    def _agg(names: List[str], metric: str) -> float:
        return max((errors[n][metric] for n in names), default=0.0)

    def _mean(names: List[str], metric: str) -> float:
        if not names:
            return 0.0
        return sum(errors[n][metric] for n in names) / len(names)

    return {
        "expert_count": len(expert_names),
        "nonexpert_count": len(nonexpert_names),
        "expert_max_abs_diff": _agg(expert_names, "max_abs_diff"),
        "expert_mean_abs_diff": _mean(expert_names, "mean_abs_diff"),
        "expert_max_relative_l2": _agg(expert_names, "relative_l2"),
        "expert_mean_relative_l2": _mean(expert_names, "relative_l2"),
        "nonexpert_max_abs_diff": _agg(nonexpert_names, "max_abs_diff"),
        "nonexpert_mean_abs_diff": _mean(nonexpert_names, "mean_abs_diff"),
        "nonexpert_max_relative_l2": _agg(nonexpert_names, "relative_l2"),
        "nonexpert_mean_relative_l2": _mean(nonexpert_names, "relative_l2"),
    }
