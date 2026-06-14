"""Recovery for the proposal CP pipeline (DeepSpeed MoE separated layout).

Rebuilds (model, optimizer) state at iteration ``t_fault`` by:

  1. Loading latest non-expert FC (``fc_nonexpert_iter{t}.bin``) with t ≤ t_fault.
  2. For each (L, E) expert, loading the most-recent LFCL compaction baseline
     (``fc_expert_L{L:02d}_E{E:02d}.bin``) and replaying its LFSY append
     log (``lossy_expert_L{L:02d}_E{E:02d}.bin``) in order via
     ``expert_state.read_expert_state``.

``cp_target=test_dram`` bypasses filesystem recovery and consumes the pinned
dense expert image maintained by ``ExpertState`` directly; it performs no LFSY
replay.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..refresh.fp32_state import get_mapper


_EXPERT_LOSSLESS_RE = re.compile(r"^fc_expert_L(\d+)_E(\d+)\.bin$")
# Lossy FC append log (discovered for sanity; actual read is driven by
# expert_state.read_expert_state which locates it by convention).
_EXPERT_LOSSY_RE = re.compile(r"^lossy_expert_L(\d+)_E(\d+)\.bin$")
_NONEXPERT_FC_RE = re.compile(r"^fc_nonexpert_iter(\d+)\.bin$")
_LAYER_IN_NAME = re.compile(r"\.layers\.(\d+)\.")
_DS_EXPERT_IDX_RE = re.compile(r"deepspeed_experts\.(\d+)\.")


ExpertId = Tuple[int, int]


def _ids_from_param_name(name: str) -> Optional[ExpertId]:
    m_e = _DS_EXPERT_IDX_RE.search(name)
    if m_e is None:
        return None
    m_l = _LAYER_IN_NAME.search(name)
    layer = int(m_l.group(1)) if m_l is not None else 0
    return (layer, int(m_e.group(1)))


@dataclass
class RecoveryDiagnostics:
    nonexpert_fc_iter: int = -1
    nonexpert_fc_path: Optional[str] = None
    expert_fc_count: int = 0
    expert_fc_iters: Dict[ExpertId, int] = field(default_factory=dict)
    min_fc_iter: int = -1
    rolling_experts_loaded: int = 0
    errors_vs_gt: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: Dict[str, float] = field(default_factory=dict)


def _scan_cp_dir(
    cp_dir: Path,
) -> Tuple[
    List[Tuple[int, Path]],
    Dict[ExpertId, Tuple[Path, Path]],
]:
    """Discover all CP artifacts in ``cp_dir``.

    Returns a 3-tuple:
      - ``nonexpert_fcs``: non-expert full FC files (``fc_nonexpert_iter{T}.bin``).
      - ``expert_pairs``: per-expert ``(lossless_path, lossy_path)`` pair.
        Either path may point at a non-existent file if that half hasn't
        been written yet — ``read_expert_state`` tolerates missing files.
    """
    scan_dirs = [cp_dir / "non_expert", cp_dir / "expert"]
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
) -> Tuple[Dict[str, torch.nn.Parameter], Dict[ExpertId, Dict[str, torch.nn.Parameter]]]:
    non_expert: Dict[str, torch.nn.Parameter] = {}
    by_expert: Dict[ExpertId, Dict[str, torch.nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ids = _ids_from_param_name(name)
        if ids is None:
            non_expert[name] = p
        else:
            by_expert.setdefault(ids, {})[name] = p
    return non_expert, by_expert


def _apply_payload_dict_to_model(
    data: Dict[str, Any],
    target_params: Dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    stream: Optional["torch.cuda.Stream"] = None,
) -> int:
    """Scatter a pre-read CPU payload dict into model + optimizer state.
    Serial GPU-side application; split out from ``_load_fc_payload_into``
    so Phase A (binary read on CPU) can run in parallel across experts.
    """
    mapper = get_mapper()
    rolling_step = int(data.get("_rolling_step", 0)) if isinstance(data, dict) else 0
    ctx = (
        torch.cuda.stream(stream)
        if stream is not None and torch.cuda.is_available()
        else None
    )
    if ctx is None:
        from contextlib import nullcontext
        ctx = nullcontext()
    with ctx:
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


def _load_fc_payload_into(
    payload_path: Path,
    target_params: Dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    expert_id: Optional[ExpertId] = None,
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
        tensor_dict, _scalars = load_binary(payload_path)
        data = tensor_dict
    return _apply_payload_dict_to_model(data, target_params, optimizer)


def recover(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cp_dir: Path,
    t_fault: int,
    num_experts: int = 0,
    load_expert_fcs: bool = True,
    expert_state_image: Optional[Any] = None,
    nonexpert_state_image: Optional[Any] = None,
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
    """
    import time as _time
    _phase: Dict[str, float] = {}
    _t_all0 = _time.perf_counter()
    diag = RecoveryDiagnostics()
    _ensure_optimizer_state(model, optimizer)

    cp_dir = Path(cp_dir)
    _t0 = _time.perf_counter()
    nonexpert_fcs, expert_pairs = _scan_cp_dir(cp_dir)
    non_expert_params, by_expert_params = _classify_params(model)
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
        ne_step_raw = ne_payload.get("_rolling_step", -1)
        ne_step = int(ne_step_raw.item()) if torch.is_tensor(ne_step_raw) else int(ne_step_raw)
        if ne_step > int(t_fault):
            raise RuntimeError(
                f"test_dram non-expert state step {ne_step} is newer "
                f"than requested fault step {int(t_fault)}"
            )
        _apply_payload_dict_to_model(ne_payload, non_expert_params, optimizer)
        diag.nonexpert_fc_iter = int(ne_step)
        diag.nonexpert_fc_path = "test_dram"
    else:
        best_ne: Optional[Tuple[int, Path]] = None
        for it, path in nonexpert_fcs:
            if it > t_fault:
                continue
            if best_ne is None or it > best_ne[0]:
                best_ne = (it, path)
        if best_ne is None:
            raise RuntimeError(
                f"No non-expert FC found with iter ≤ {t_fault} in {cp_dir}"
            )
        _load_fc_payload_into(best_ne[1], non_expert_params, optimizer)
        diag.nonexpert_fc_iter = best_ne[0]
        diag.nonexpert_fc_path = str(best_ne[1])
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
        for (L, E), target in by_expert_params.items():
            payload = expert_state_image.read((L, E))
            if payload is None:
                continue
            log_step = _apply_payload_dict_to_model(payload, target, optimizer)
            if log_step >= 0 and log_step <= t_fault:
                log_loaded[(L, E)] = log_step
                diag.expert_fc_iters[(L, E)] = log_step
        _phase["expert_dram_apply_ms"] = (
            _time.perf_counter() - _t_dram0
        ) * 1000.0
        diag.rolling_experts_loaded = len(log_loaded)
        diag.expert_fc_count = len(log_loaded)
        if diag.expert_fc_count == 0:
            raise RuntimeError(
                "test_dram expert state contained no recoverable experts; "
                "did you call compact_lossy_recovery_image / "
                "append_lossy_records_for_test before recover?"
            )
    elif load_expert_fcs:
        # Phase A (parallel, CPU-bound): read each expert's (LFCL, LFSY)
        # pair from disk via read_expert_state: pure numpy scatter, GIL-free
        # during I/O + numpy kernels. Phase B (serial, GPU) applies the
        # resulting CPU dicts to model/optimizer so we don't race on the
        # default CUDA stream or on mapper state.
        from concurrent.futures import ThreadPoolExecutor
        from .expert_writer import read_expert_state

        eids_with_new_pair: List[ExpertId] = []
        for (L, E), _target in by_expert_params.items():
            pair = expert_pairs.get((L, E))
            if pair is not None and (pair[0].exists() or pair[1].exists()):
                eids_with_new_pair.append((L, E))

        _t_read0 = _time.perf_counter()
        new_pair_data: Dict[ExpertId, Dict[str, Any]] = {}
        if eids_with_new_pair:
            # Thread pool width: min(32, n_experts). Diminishing returns past
            # ~8 due to GIL on dict mutation + file descriptor contention,
            # but headroom helps when experts vary in lossy-log size.
            n_workers = min(32, max(1, len(eids_with_new_pair)))

            def _read_one(eid: ExpertId) -> Tuple[ExpertId, Dict[str, Any]]:
                L, E = eid
                pair = expert_pairs[eid]
                data = read_expert_state(pair[0].parent, layer=L, expert=E)
                return eid, data

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for eid, data in pool.map(_read_one, eids_with_new_pair):
                    new_pair_data[eid] = data
        _phase["expert_read_ms"] = (_time.perf_counter() - _t_read0) * 1000.0

        # Phase B: apply the CPU payloads serially on the default CUDA stream.
        _t_apply0 = _time.perf_counter()
        stream_pool: List["torch.cuda.Stream"] = []
        if torch.cuda.is_available() and eids_with_new_pair:
            stream_pool = [
                torch.cuda.Stream()
                for _ in range(min(8, len(eids_with_new_pair)))
            ]
        for apply_i, eid in enumerate(eids_with_new_pair):
            data = new_pair_data[eid]
            target = by_expert_params[eid]
            stream = (
                stream_pool[apply_i % len(stream_pool)]
                if stream_pool else None
            )
            log_step = _apply_payload_dict_to_model(
                data, target, optimizer, stream=stream,
            )
            if log_step <= t_fault:
                log_loaded[eid] = log_step
                diag.expert_fc_iters[eid] = log_step
        for stream in stream_pool:
            stream.synchronize()
        _phase["expert_apply_ms"] = (_time.perf_counter() - _t_apply0) * 1000.0

        diag.rolling_experts_loaded = len(log_loaded)
        diag.expert_fc_count = len(log_loaded)
        if diag.expert_fc_count == 0:
            raise RuntimeError(
                f"No expert LFCL/LFSY state found with "
                f"iter ≤ {t_fault} in {cp_dir}"
            )
    _phase["expert_load_ms"] = (_time.perf_counter() - _t0) * 1000.0

    diag.min_fc_iter = int(diag.nonexpert_fc_iter)

    _phase["total_ms"] = (_time.perf_counter() - _t_all0) * 1000.0
    diag.summary.update(_phase)
    _print_recovery_phases(_phase, diag)
    return diag


def _print_recovery_phases(phase: Dict[str, float], diag: RecoveryDiagnostics) -> None:
    """Emit a one-line phase breakdown for recover() — used by benchmarks
    to separate scan / non-expert FC load / expert load."""
    print(
        f"[recovery] total={phase.get('total_ms', 0.0):.1f}ms  "
        f"scan={phase.get('scan_classify_ms', 0.0):.1f} "
        f"ne_fc={phase.get('nonexpert_fc_ms', 0.0):.1f} "
        f"expert_load={phase.get('expert_load_ms', 0.0):.1f}"
        f"(read={phase.get('expert_read_ms', 0.0):.1f}+apply={phase.get('expert_apply_ms', 0.0):.1f}) "
        f"experts={diag.expert_fc_count}",
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
