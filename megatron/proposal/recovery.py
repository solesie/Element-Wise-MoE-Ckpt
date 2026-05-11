"""Recovery for the proposal CP pipeline (DeepSpeed MoE separated layout).

Rebuilds (model, optimizer) state at iteration ``t_fault`` by:

  1. Loading latest non-expert FC (``fc_nonexpert_iter{t}.pt``) with t ≤ t_fault.
  2. For each (L, E) expert, loading the most-recent lossless FC file
     (``fc_expert_L{L:02d}_E{E:02d}.bin``) and replaying its lossy FC
     append log (``lossy_expert_L{L:02d}_E{E:02d}.bin``) in order via
     ``expert_state.read_expert_state``. MoC (sequential) baseline still
     writes legacy ``fc_expert_L{L}_E{E}_iter{t}.pt`` and is loaded via
     the legacy path when no lossless/lossy pair exists.
  3. Replaying non-expert DC bundles (``dc_iter{t}_dense_ne.pt``) for t in
     (nonexpert_fc_iter, t_fault] via fused AdamW (lossless).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ._ds_state import get_mapper


# Per-iter expert FC for MoC (sequential) mode. Accepts both .bin (current,
# PRPTENS) and .pt (legacy torch.save pickle from pre-bin runs).
_EXPERT_FC_RE = re.compile(r"^fc_expert_L(\d+)_E(\d+)_iter(\d+)\.(?:bin|pt)$")
# New per-expert lossless FC: single .bin file per (L, E).
_EXPERT_LOSSLESS_RE = re.compile(r"^fc_expert_L(\d+)_E(\d+)\.bin$")
# Lossy FC append log (discovered for sanity; actual read is driven by
# expert_state.read_expert_state which locates it by convention).
_EXPERT_LOSSY_RE = re.compile(r"^lossy_expert_L(\d+)_E(\d+)\.bin$")
_NONEXPERT_FC_RE = re.compile(r"^fc_nonexpert_iter(\d+)\.(pt|bin)$")
# Non-expert dense DC bundle (lossless grad). No expert DC files.
_DC_DENSE_NE_RE = re.compile(r"^dc_iter(\d+)_dense_ne\.(pt|bin)$")
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
    dc_bundles_found: int = 0
    dc_bundles_replayed: int = 0
    replay_range: Tuple[int, int] = (-1, -1)
    # Log-structured recovery: how many (L, E) experts loaded from
    # <cp>/expert/L{L:02d}_E{E:02d}/ directories (one compacted + segments
    # merge per expert, no DC replay for expert state).
    rolling_experts_loaded: int = 0
    errors_vs_gt: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: Dict[str, float] = field(default_factory=dict)


def _scan_cp_dir(
    cp_dir: Path,
) -> Tuple[
    Dict[ExpertId, List[Tuple[int, Path]]],
    List[Tuple[int, Path]],
    List[Tuple[int, Path]],
    Dict[ExpertId, Tuple[Path, Path]],
]:
    """Discover all CP artifacts in ``cp_dir``.

    Returns a 4-tuple:
      - ``expert_fcs``: legacy per-iter expert FC files (MoC sequential mode).
      - ``nonexpert_fcs``: non-expert full FC files (``fc_nonexpert_iter{T}.pt``).
      - ``dc_bundles``: non-expert dense DC bundles (``dc_iter{T}_dense_ne.pt``).
      - ``expert_new``: per-expert ``(lossless_path, lossy_path)`` pair for
        the new backend. Either path may point at a non-existent file if
        that half hasn't been written yet — ``read_expert_state`` tolerates
        missing files.

    Scans both the current split layout (``non_expert/`` + ``expert/``) and
    the legacy flat layout (``fc/`` + ``dc/``). Filenames are unique enough
    that mixing is safe.
    """
    scan_dirs = [
        cp_dir / "non_expert", cp_dir / "expert",  # current default (split)
        cp_dir / "fc",         cp_dir / "dc",      # legacy layout
    ]
    expert_fcs: Dict[ExpertId, List[Tuple[int, Path]]] = {}
    expert_new: Dict[ExpertId, Dict[str, Path]] = {}
    nonexpert_fcs: List[Tuple[int, Path]] = []
    dc_bundles: List[Tuple[int, Path]] = []

    for d in scan_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_dir():
                continue
            # New per-expert lossless FC (single .bin per L/E).
            m = _EXPERT_LOSSLESS_RE.match(f.name)
            if m is not None:
                L, E = int(m.group(1)), int(m.group(2))
                expert_new.setdefault((L, E), {})["lossless"] = f
                continue
            # New per-expert lossy FC append log.
            m = _EXPERT_LOSSY_RE.match(f.name)
            if m is not None:
                L, E = int(m.group(1)), int(m.group(2))
                expert_new.setdefault((L, E), {})["lossy"] = f
                continue
            # Legacy MoC per-iter FC.
            m = _EXPERT_FC_RE.match(f.name)
            if m is not None:
                L, E, it = map(int, m.groups())
                expert_fcs.setdefault((L, E), []).append((it, f))
                continue
            m = _NONEXPERT_FC_RE.match(f.name)
            if m is not None:
                nonexpert_fcs.append((int(m.group(1)), f))
                continue
            m = _DC_DENSE_NE_RE.match(f.name)
            if m is not None:
                dc_bundles.append((int(m.group(1)), f))

    expert_pairs: Dict[ExpertId, Tuple[Path, Path]] = {}
    for eid, parts in expert_new.items():
        # ``read_expert_state`` derives both filenames from the expert_dir
        # + (L, E) arguments; we just need the containing directory. Both
        # halves (if present) must live in the same directory.
        any_path = parts.get("lossless") or parts.get("lossy")
        if any_path is None:
            continue
        expert_dir = any_path.parent
        # Reuse the expected filenames as the pair record (existence is
        # checked at read time).
        L, E = eid
        expert_pairs[eid] = (
            expert_dir / f"fc_expert_L{L:02d}_E{E:02d}.bin",
            expert_dir / f"lossy_expert_L{L:02d}_E{E:02d}.bin",
        )
    return expert_fcs, nonexpert_fcs, dc_bundles, expert_pairs


def _ensure_optimizer_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> None:
    """Pre-populate optimizer state for non-mapper backends.

    Skipped entirely when the Fp32StateMapper is active (DeepSpeed BF16/Zero
    paths): in those modes the actual optimizer state lives in the
    mapper-managed FP32 flat buffers (already allocated at init), and
    forcing an extra ``optimizer.state[p]`` slot per param adds a
    redundant BF16 copy of m + v for **every** param. On exp_etc that
    duplicate is ~785 MB persistent VRAM, which the first fault recovery
    triggers and never frees. Mapper-aware get/set_state paths in this
    file (``_apply_payload_dict_to_model``, ``_gather_state_fp32``) read
    and write only the mapper's flat buffers, so the optimizer.state[p]
    duplicate is never used and can be safely skipped here.
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
) -> int:
    """Scatter a pre-read CPU payload dict into model + optimizer state.
    Serial GPU-side application; split out from ``_load_fc_payload_into``
    so Phase A (binary read on CPU) can run in parallel across experts.
    """
    mapper = get_mapper()
    rolling_step = int(data.get("_rolling_step", 0)) if isinstance(data, dict) else 0
    for key, tensor in data.items():
        if not isinstance(key, str) or ":" not in key:
            continue
        name, field_name = key.rsplit(":", 1)
        p = target_params.get(name)
        if p is None:
            continue
        value = tensor.to(device=p.device, dtype=torch.float32)
        if field_name == "param":
            if mapper is not None:
                mapper.set_param(p, value)
            else:
                p.data.copy_(value.to(dtype=p.dtype))
        elif field_name in ("exp_avg", "exp_avg_sq"):
            if mapper is not None:
                mapper.set_state(p, value, field_name)
            else:
                state = optimizer.state.setdefault(p, {})
                slot = state.get(field_name)
                if slot is None:
                    slot = torch.zeros_like(p, dtype=torch.float32)
                    state[field_name] = slot
                slot.copy_(value.to(dtype=slot.dtype))
    return rolling_step


def _load_fc_payload_into(
    payload_path: Path,
    target_params: Dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    expert_id: Optional[ExpertId] = None,
) -> int:
    """Load FC payload into model + optimizer state.

    Two input formats are supported:

      - Legacy FC pickle dict (``fc_expert_L*_E*_iter*.pt``,
        ``fc_nonexpert_*.pt``): ``torch.load``.
      - New per-expert ``(lossless, lossy)`` pair: when ``payload_path``
        is a directory, ``expert_id`` must be supplied and
        ``expert_state.read_expert_state(dir, L, E)`` merges the
        lossless FC + lossy FC append log into a scatter-ready payload.
    """
    if payload_path.is_dir():
        if expert_id is None:
            raise ValueError(
                "Directory payload requires expert_id for read_expert_state"
            )
        from .expert_state import read_expert_state
        data = read_expert_state(payload_path, layer=expert_id[0],
                                  expert=expert_id[1])
    else:
        from .binary_io import is_binary_format, load_binary
        if is_binary_format(payload_path):
            tensor_dict, _scalars = load_binary(payload_path)
            data = tensor_dict
        else:
            data = torch.load(payload_path, map_location="cpu")
    return _apply_payload_dict_to_model(data, target_params, optimizer)


def _fused_adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    step_prev: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> int:
    """One AdamW update via ``torch._fused_adamw_`` (kernel underlying
    ``torch.optim.AdamW(fused=True)``). Used only for non-expert dense-grad
    replay; expert replay is direct state overwrite with no Adam step.
    """
    if grad.dtype != param.dtype:
        grad = grad.to(dtype=param.dtype)
    step_new = step_prev + 1
    step_tensor = torch.tensor(
        float(step_new), dtype=torch.float32, device=param.device,
    )
    torch._fused_adamw_(
        [param], [grad], [m], [v], [], [step_tensor],
        amsgrad=False,
        lr=lr, beta1=beta1, beta2=beta2,
        weight_decay=weight_decay, eps=eps,
        maximize=False,
        grad_scale=None, found_inf=None,
    )
    return step_new


def recover(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cp_dir: Path,
    t_fault: int,
    num_experts: int = 0,
    dc_replay: bool = True,
    load_expert_fcs: bool = True,
    dram_expert_state: Optional[Any] = None,
) -> RecoveryDiagnostics:
    """Rebuild state from CP files at ``t_fault``.

    ``dram_expert_state``: when provided (proposal cp_target=dram), per-expert
    state is read from the in-memory DramExpertState buffer instead of
    scanning disk for ``(lossless, lossy)`` pairs. Non-expert FC and DC
    bundles still come from disk — DRAM mode currently only optimizes the
    expert path. Set CP_DIR_OVERRIDE / --proposal-cp-dir to /dev/shm to also
    eliminate disk I/O for non-expert + DC.
    """
    import time as _time
    _phase: Dict[str, float] = {}
    _t_all0 = _time.perf_counter()
    diag = RecoveryDiagnostics()
    _ensure_optimizer_state(model, optimizer)

    cp_dir = Path(cp_dir)
    _t0 = _time.perf_counter()
    expert_fcs, nonexpert_fcs, dc_bundles, expert_pairs = _scan_cp_dir(cp_dir)
    non_expert_params, by_expert_params = _classify_params(model)
    _phase["scan_classify_ms"] = (_time.perf_counter() - _t0) * 1000.0

    # 1. Non-expert FC
    _t0 = _time.perf_counter()
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
    _phase["nonexpert_fc_ms"] = (_time.perf_counter() - _t0) * 1000.0
    diag.nonexpert_fc_iter = best_ne[0]
    diag.nonexpert_fc_path = str(best_ne[1])

    # 2. Per-expert state. Prefer the new ``(lossless, lossy)`` pair
    #    (lossless FC loaded + lossy FC append log replayed, last-write-
    #    wins per index) over legacy per-iter FC files. Fall back to
    #    legacy per-iter FC (+ DC replay) only when no new-format pair
    #    exists.
    expert_fc_best: Dict[ExpertId, Tuple[int, Path]] = {}
    log_loaded: Dict[ExpertId, int] = {}
    legacy_experts: Dict[ExpertId, Dict[str, torch.nn.Parameter]] = {}

    _t0 = _time.perf_counter()
    if load_expert_fcs and dram_expert_state is not None:
        # DRAM mode: each expert's merged dense (p, m, v) buffer lives in
        # host RAM. No log replay needed — DramExpertState.lossy_fc_append
        # already scattered every update into the buffer at write time.
        # Recovery is just an O(num_experts × tensors_per_expert × full_size)
        # CPU→GPU copy. We bypass disk scan entirely for the expert path.
        _t_dram0 = _time.perf_counter()
        for (L, E), target in by_expert_params.items():
            payload = dram_expert_state.read((L, E))
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
                "DRAM expert state contained no recoverable experts; "
                "did you call lossless_fc / lossy_fc_append before recover?"
            )
        # min_fc_iter / expert_load_ms / DC replay are handled by the
        # shared code path below — no log replay window for DRAM mode
        # (every selected element was already scattered into the dense
        # buffer at write time), so the DC replay floor is best_ne[0].
    elif load_expert_fcs:
        # Phase A (parallel, CPU-bound): read each expert's (lossless, lossy)
        # pair from disk via read_expert_state — pure numpy scatter, GIL-free
        # during I/O + numpy kernels. Phase B (serial, GPU) applies the
        # resulting CPU dicts to model/optimizer so we don't race on the
        # default CUDA stream or on mapper state.
        from concurrent.futures import ThreadPoolExecutor
        from .expert_state import read_expert_state

        eids_with_new_pair: List[ExpertId] = []
        eids_legacy: List[Tuple[ExpertId, Tuple[int, Path]]] = []
        for (L, E), _target in by_expert_params.items():
            pair = expert_pairs.get((L, E))
            if pair is not None and (pair[0].exists() or pair[1].exists()):
                eids_with_new_pair.append((L, E))
            else:
                best: Optional[Tuple[int, Path]] = None
                for it, path in expert_fcs.get((L, E), []):
                    if it > t_fault:
                        continue
                    if best is None or it > best[0]:
                        best = (it, path)
                if best is not None:
                    eids_legacy.append(((L, E), best))

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
        for eid in eids_with_new_pair:
            data = new_pair_data[eid]
            target = by_expert_params[eid]
            log_step = _apply_payload_dict_to_model(data, target, optimizer)
            if log_step <= t_fault:
                log_loaded[eid] = log_step
                diag.expert_fc_iters[eid] = log_step
            else:
                # Stale on-disk state (newer than t_fault); fall through to
                # legacy per-iter FC if available for this expert.
                for it, path in expert_fcs.get(eid, []):
                    if it > t_fault:
                        continue
                    eids_legacy.append((eid, (it, path)))
                    break

        for eid, (it, path) in eids_legacy:
            target = by_expert_params[eid]
            _load_fc_payload_into(path, target, optimizer)
            expert_fc_best[eid] = (it, path)
            diag.expert_fc_iters[eid] = it
            legacy_experts[eid] = target
        _phase["expert_apply_ms"] = (_time.perf_counter() - _t_apply0) * 1000.0

        diag.rolling_experts_loaded = len(log_loaded)
        diag.expert_fc_count = len(log_loaded) + len(expert_fc_best)
        if diag.expert_fc_count == 0:
            raise RuntimeError(
                f"No expert lossless FCs or legacy FCs found with "
                f"iter ≤ {t_fault} in {cp_dir}"
            )
    _phase["expert_load_ms"] = (_time.perf_counter() - _t0) * 1000.0

    # min_fc_iter bounds the DC replay window. Log-loaded experts already
    # encode state AT log_step with no replay needed — they do NOT widen
    # the replay window. We use the non-expert FC iter and any legacy-FC
    # experts' iters to define the window.
    if expert_fc_best:
        min_fc_iter = min(
            best_ne[0],
            min(it for (it, _) in expert_fc_best.values()),
        )
    else:
        min_fc_iter = best_ne[0]
    diag.min_fc_iter = min_fc_iter

    # 3. DC replay
    if not dc_replay:
        _phase["total_ms"] = (_time.perf_counter() - _t_all0) * 1000.0
        diag.summary.update(_phase)
        _print_recovery_phases(_phase, diag)
        return diag
    _t0 = _time.perf_counter()

    replay_window = sorted(
        ((it, path) for it, path in dc_bundles if min_fc_iter < it <= t_fault),
        key=lambda x: x[0],
    )
    diag.dc_bundles_found = len(replay_window)
    diag.replay_range = (
        (replay_window[0][0], replay_window[-1][0])
        if replay_window
        else (-1, -1)
    )

    # AdamW hyperparams — needed for non-expert dense-grad replay only.
    bare_opt = optimizer
    for _ in range(8):
        if isinstance(bare_opt, torch.optim.Optimizer) and hasattr(bare_opt, "defaults"):
            break
        nxt = getattr(bare_opt, "optimizer", None)
        if nxt is None or nxt is bare_opt:
            break
        bare_opt = nxt
    opt_defs = getattr(bare_opt, "defaults", {})
    lr = float(opt_defs.get("lr", 1e-4))
    betas = opt_defs.get("betas", (0.9, 0.999))
    beta1, beta2 = float(betas[0]), float(betas[1])
    eps = float(opt_defs.get("eps", 1e-8))
    weight_decay = float(opt_defs.get("weight_decay", 0.0))

    mapper = get_mapper()

    def _gather_fp32(p: torch.nn.Parameter) -> torch.Tensor:
        if mapper is not None:
            return mapper.get_param(p).detach().clone()
        return p.data.detach().to(torch.float32).clone()

    def _gather_state_fp32(p: torch.nn.Parameter, key: str) -> torch.Tensor:
        if mapper is not None:
            v = mapper.get_state(p, key)
            if v is not None:
                return v.detach().clone()
        st = optimizer.state.get(p, {})
        v = st.get(key)
        if v is not None:
            return v.detach().to(torch.float32).clone()
        return torch.zeros_like(p, dtype=torch.float32)

    # Non-expert dense DC replay via fused AdamW. Expert state is ALREADY
    # live from the rolling file load above — no replay needed for experts.
    nonexpert_shadow: Dict[str, Dict] = {}
    for name, p in non_expert_params.items():
        nonexpert_shadow[name] = {
            "step": int(diag.nonexpert_fc_iter),
            "param": _gather_fp32(p),
            "m": _gather_state_fp32(p, "exp_avg"),
            "v": _gather_state_fp32(p, "exp_avg_sq"),
        }

    for bundle_iter, bundle_path in replay_window:
        ne_needed = any(
            bundle_iter > sh["step"] for sh in nonexpert_shadow.values()
        )
        if not ne_needed:
            diag.dc_bundles_replayed += 1
            continue
        # Magic-detect binary format. Legacy .pt pickle bundles had
        # ``{"nonexpert": {name: grad}, "step": int}``; the new .bin format
        # stores tensors directly at the top level (scalars in header).
        from .binary_io import is_binary_format, load_binary
        if is_binary_format(bundle_path):
            ne_dict, _scalars = load_binary(bundle_path)
        else:
            bundle = torch.load(bundle_path, map_location="cpu")
            ne_dict = bundle.get("nonexpert", {}) or {}
        for name, flat_grad in ne_dict.items():
            shadow = nonexpert_shadow.get(name)
            if shadow is None:
                continue
            if bundle_iter <= shadow["step"]:
                continue
            grad_tensor = flat_grad.view(shadow["param"].shape).to(
                device=shadow["param"].device, dtype=torch.float32,
            )
            shadow["step"] = _fused_adamw_step(
                shadow["param"], grad_tensor,
                shadow["m"], shadow["v"], shadow["step"],
                lr, beta1, beta2, eps, weight_decay,
            )
        diag.dc_bundles_replayed += 1

    # Writeback non-expert shadow → live model + optimizer state.
    for name, shadow in nonexpert_shadow.items():
        p = non_expert_params[name]
        if mapper is not None:
            mapper.set_param(p, shadow["param"])
            mapper.set_state(p, shadow["m"], "exp_avg")
            mapper.set_state(p, shadow["v"], "exp_avg_sq")
        else:
            p.data.copy_(shadow["param"].to(dtype=p.dtype))
            st = optimizer.state.setdefault(p, {})
            st.setdefault("exp_avg", torch.zeros_like(p, dtype=torch.float32)).copy_(shadow["m"])
            st.setdefault("exp_avg_sq", torch.zeros_like(p, dtype=torch.float32)).copy_(shadow["v"])
        st = optimizer.state.setdefault(p, {})
        st["step"] = torch.tensor(float(shadow["step"]))

    _phase["dc_replay_ms"] = (_time.perf_counter() - _t0) * 1000.0
    _phase["total_ms"] = (_time.perf_counter() - _t_all0) * 1000.0
    diag.summary.update(_phase)
    _print_recovery_phases(_phase, diag)
    return diag


def _print_recovery_phases(phase: Dict[str, float], diag: RecoveryDiagnostics) -> None:
    """Emit a one-line phase breakdown for recover() — used by benchmarks
    to separate scan / non-expert FC load / 40-expert load / DC replay."""
    print(
        f"[recovery] total={phase.get('total_ms', 0.0):.1f}ms  "
        f"scan={phase.get('scan_classify_ms', 0.0):.1f} "
        f"ne_fc={phase.get('nonexpert_fc_ms', 0.0):.1f} "
        f"expert_load={phase.get('expert_load_ms', 0.0):.1f}"
        f"(read={phase.get('expert_read_ms', 0.0):.1f}+apply={phase.get('expert_apply_ms', 0.0):.1f}) "
        f"dc_replay={phase.get('dc_replay_ms', 0.0):.1f}  "
        f"experts={diag.expert_fc_count} "
        f"dc_bundles={diag.dc_bundles_replayed}/{diag.dc_bundles_found}",
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
