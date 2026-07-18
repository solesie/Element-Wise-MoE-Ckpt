"""Telemetry for one independent policy run.

This module measures the checkpoint image owned by the current run.  It does
not simulate several policies from one shared checkpoint image.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..refresh.fp32_state import (
    discard_zero12_grad_state,
    get_engine_optimizer,
)
from .ideal_oracle import (
    _all_reduce_mean_float,
    _all_reduce_sum_float,
    _all_reduce_sum_int,
    _build_forward_loss_bank,
    _capture_rng_state,
    _eval_loss,
    _eval_loss_bank,
    _is_pipeline_engine,
    _restore_rng_state,
)

_SNAPSHOT_ATTR = "_independent_oracle_live_param_snapshot"
_PRODUCTION_FIRST_ORDER_SOURCE = "momentum"
_PRODUCTION_FIRST_ORDER_KEY = "m_flat"
_PRODUCTION_FIRST_ORDER_NAME = "m_t"
_ORACLE_SEED = 20240625
_HVP_BATCHES = 1


def independent_oracle_enabled(proposal_config: Any, iteration: int) -> bool:
    if not bool(getattr(proposal_config, "independent_oracle_enable", False)):
        return False
    iters = set(int(x) for x in getattr(proposal_config, "independent_oracle_iters", ()) or ())
    return not iters or int(iteration) in iters


def run_independent_oracle(
    *,
    model: Any,
    data_iterator: Any,
    iteration: int,
    lossy_cp: Any,
    proposal_config: Any,
) -> None:
    if not independent_oracle_enabled(proposal_config, int(iteration)):
        return
    rank, world = _rank_world()
    log_dir = _log_dir(proposal_config)
    if not getattr(lossy_cp, "_groups", None):
        _append_jsonl(log_dir / f"independent_skip_rank{rank:03d}.jsonl", {
            "mode": "independent_oracle_skip",
            "step": int(iteration),
            "rank": int(rank),
            "world": int(world),
            "reason": "no_lossy_groups",
        })
        return
    t0 = time.perf_counter()
    log_dir.mkdir(parents=True, exist_ok=True)

    hvp_batches = _HVP_BATCHES
    requested_hvp_batches = _HVP_BATCHES
    m0 = model[0] if isinstance(model, (list, tuple)) else model
    pipeline_engine = bool(_is_pipeline_engine(m0))
    batch_seed = _ORACLE_SEED
    outer_rng = _capture_rng_state()
    prev_snapshot = getattr(lossy_cp, _SNAPSHOT_ATTR, None)
    live_snapshot = None
    try:
        torch.manual_seed(batch_seed + int(iteration))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed + int(iteration))
        loss_bank = _build_forward_loss_bank(
            model=model,
            data_iterator=data_iterator,
            n_batches=hvp_batches,
        )
        if not loss_bank:
            _append_jsonl(log_dir / f"independent_skip_rank{rank:03d}.jsonl", {
                "mode": "independent_oracle_skip",
                "step": int(iteration),
                "rank": int(rank),
                "world": int(world),
                "reason": "empty_loss_bank",
                "requested_hvp_batches": int(requested_hvp_batches),
            })
            return

        live_snapshot = _LiveParamSnapshot.capture(lossy_cp)
        setattr(lossy_cp, _SNAPSHOT_ATTR, live_snapshot)
        _run_independent_rank_only(
            model=model,
            loss_bank=loss_bank,
            iteration=int(iteration),
            lossy_cp=lossy_cp,
            proposal_config=proposal_config,
            rank=rank,
            world=world,
            log_dir=log_dir,
            started_at=t0,
            requested_hvp_batches=requested_hvp_batches,
            pipeline_engine=pipeline_engine,
        )
        return
    finally:
        _restore_current_params(lossy_cp)
        # Pipeline HVP runs ReduceGrads but intentionally skips OptimizerStep.
        # Drop that ZeRO scratch table just as Hessian refresh does; otherwise
        # the first oracle leaves one local gradient partition resident.
        discard_zero12_grad_state(get_engine_optimizer(model))
        if live_snapshot is not None:
            max_diff = live_snapshot.max_abs_diff()
            if max_diff > float(os.environ.get(
                "PROPOSAL_INDEPENDENT_ORACLE_RESTORE_ATOL",
                "0.0",
            ) or 0.0):
                _append_jsonl(log_dir / f"independent_restore_rank{rank:03d}.jsonl", {
                    "mode": "independent_oracle_restore_error",
                    "step": int(iteration),
                    "rank": int(rank),
                    "world": int(world),
                    "max_abs_diff_after_restore": float(max_diff),
                })
                raise RuntimeError(
                    "independent oracle failed to restore live model params: "
                    f"max_abs_diff={max_diff:.6e}"
                )
            _append_jsonl(log_dir / f"independent_restore_rank{rank:03d}.jsonl", {
                "mode": "independent_oracle_restore_check",
                "step": int(iteration),
                "rank": int(rank),
                "world": int(world),
                "max_abs_diff_after_restore": float(max_diff),
                "status": "ok",
            })
        if prev_snapshot is None:
            try:
                delattr(lossy_cp, _SNAPSHOT_ATTR)
            except AttributeError:
                pass
        else:
            setattr(lossy_cp, _SNAPSHOT_ATTR, prev_snapshot)
        _restore_rng_state(outer_rng)


def _run_independent_rank_only(
    *,
    model: Any,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    iteration: int,
    lossy_cp: Any,
    proposal_config: Any,
    rank: int,
    world: int,
    log_dir: Path,
    started_at: float,
    requested_hvp_batches: int,
    pipeline_engine: bool,
) -> None:
    """Measure ranking quality without running unrelated loss/proxy telemetry."""

    error_rms = _global_error_rms(lossy_cp)
    sample = _build_sample(lossy_cp, proposal_config, rank=rank)
    sample_payload = _gather_sample_payload(
        lossy_cp,
        sample,
        iteration=int(iteration),
    )
    fd_eps = float(
        getattr(proposal_config, "ideal_oracle_fd_eps", None)
        or getattr(proposal_config, "finite_diff_eps", 1.0e-3)
        or 1.0e-3
    )
    hvp_sample: Optional[np.ndarray] = None
    hvp_batches = min(int(requested_hvp_batches), len(loss_bank))
    if hvp_batches > 0 and error_rms > 0.0:
        _r2_unused, hvp_sample = _hvp_for_current_error(
            lossy_cp=lossy_cp,
            model=model,
            loss_bank=loss_bank[:hvp_batches],
            fd_eps=fd_eps,
            rms_e=float(error_rms),
            sample=sample,
            compute_full_quadratic=False,
        )

    rank_rows = _rank_metric_rows(
        sample_payload=sample_payload,
        hvp_sample=hvp_sample,
        proposal_config=proposal_config,
        iteration=int(iteration),
        rank=rank,
        world=world,
        score_source=_PRODUCTION_FIRST_ORDER_SOURCE,
    )
    rank_rows = [
        row for row in rank_rows
        if row.get("mode") == "independent_oracle_rank"
    ]
    for row in rank_rows:
        _append_jsonl(log_dir / f"independent_rank_rank{rank:03d}.jsonl", row)

    _append_jsonl(log_dir / f"independent_timing_rank{rank:03d}.jsonl", {
        "mode": "independent_oracle_timing",
        "step": int(iteration),
        "rank": int(rank),
        "elapsed_ms": float((time.perf_counter() - started_at) * 1000.0),
        "actual_batches": 0,
        "hvp_batches": int(hvp_batches),
        "requested_hvp_batches": int(requested_hvp_batches),
        "hvp_skip_reason": None if hvp_sample is not None else "zero_checkpoint_error",
        "pipeline_engine": bool(pipeline_engine),
        "rank_only": True,
        "sample_elements": int(sample_payload["e"].size),
        "rank_rows": int(len(rank_rows)),
    })


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "0")).strip().lower() in ("1", "true", "yes", "on")


def _param_ref_for_name(
    lossy_cp: Any,
    name: str,
) -> Tuple[Optional[torch.nn.Parameter], int, int]:
    param = lossy_cp._name_to_param.get(str(name))
    if param is not None:
        return param, 0, int(param.numel())
    meta = getattr(lossy_cp, "_expert_partition_fragment_meta", {}) or {}
    frag = meta.get(str(name))
    if not isinstance(frag, dict):
        return None, 0, 0
    base = str(frag.get("name", ""))
    param = lossy_cp._name_to_param.get(base)
    if param is None:
        return None, 0, 0
    return param, int(frag.get("param_offset", 0) or 0), int(frag.get("numel", 0) or 0)


def _copy_flat_to_param(
    param: torch.nn.Parameter,
    value: torch.Tensor,
    *,
    param_offset: int = 0,
    n: Optional[int] = None,
) -> None:
    n_elem = int(value.numel() if n is None else n)
    dst = param.data.detach().reshape(-1).narrow(0, int(param_offset), n_elem)
    dst.copy_(
        value.reshape(-1).narrow(0, 0, n_elem).to(
            device=dst.device,
            dtype=dst.dtype,
        ),
        non_blocking=True,
    )


def _grad_slice_for_spec(raw_grad: torch.Tensor, spec: Dict[str, Any]) -> torch.Tensor:
    off = int(spec.get("param_offset", 0) or 0)
    n = int(spec["n"])
    return raw_grad.detach().reshape(-1).narrow(0, off, n)


def _group_entry_specs(
    lossy_cp: Any,
    group_idx: int,
    grp: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return the same entry geometry used by production score/write kernels.

    ``elem_off`` addresses live optimizer flat views (p/m/v/g), while
    ``age_off`` addresses the compact theta/age space.  They are identical in
    simple layouts, but diverge under expert partitioning / ZeRO flat groups.
    ``logical_off`` is only the compact per-group coordinate used for sampling.
    """

    g = max(1, int(getattr(lossy_cp, "group_size", 1)))
    master_base_elem = int(grp.get("master_off", 0)) * int(g)
    tname_list = list(getattr(lossy_cp, "_tname_list", ()) or ())
    elem_starts = list(grp.get("entry_elem_start_host", ()) or ())
    age_starts = list(grp.get("entry_age_elem_start_host", ()) or ())
    counts = list(grp.get("entry_group_count_host", ()) or ())
    name_idxs = list(grp.get("entry_name_idx_host", ()) or ())
    specs: List[Dict[str, Any]] = []

    if (
        elem_starts
        and len(elem_starts) == len(age_starts) == len(counts) == len(name_idxs)
    ):
        group_names = list(grp.get("names", ()) or ())
        for entry_idx, (elem_start, age_start, count, name_idx) in enumerate(
            zip(elem_starts, age_starts, counts, name_idxs)
        ):
            n = int(count) * int(g)
            if n <= 0:
                continue
            ni = int(name_idx)
            if 0 <= ni < len(tname_list):
                name = str(tname_list[ni])
            elif entry_idx < len(group_names):
                name = str(group_names[entry_idx])
            else:
                name = str(ni)
            specs.append({
                "group_idx": int(group_idx),
                "grp": grp,
                "name": name,
                "entry_idx": int(entry_idx),
                "name_idx": int(ni),
                "logical_off": int(age_start) - int(master_base_elem),
                "local_off": int(age_start) - int(master_base_elem),
                "elem_off": int(elem_start),
                "age_off": int(age_start),
                "n": int(n),
            })
        return specs

    logical_off = 0
    flat_off = 0
    for entry_idx, (name, size) in enumerate(zip(grp.get("names", ()), grp.get("sizes", ()))):
        size_i = int(size)
        n = (size_i // int(g)) * int(g)
        if n > 0:
            specs.append({
                "group_idx": int(group_idx),
                "grp": grp,
                "name": str(name),
                "entry_idx": int(entry_idx),
                "name_idx": int(entry_idx),
                "logical_off": int(logical_off),
                "local_off": int(logical_off),
                "elem_off": int(flat_off),
                "age_off": int(master_base_elem) + int(logical_off),
                "n": int(n),
            })
            logical_off += int(n)
        flat_off += int(size_i)
    return specs


def _elem_off(spec: Dict[str, Any]) -> int:
    return int(spec.get("elem_off", spec.get("local_off", 0)) or 0)


def _age_off(spec: Dict[str, Any]) -> int:
    return int(spec.get("age_off", spec.get("local_off", 0)) or 0)


def _logical_off(spec: Dict[str, Any]) -> int:
    return int(spec.get("logical_off", spec.get("local_off", 0)) or 0)


def _checkpoint_tensor_for_spec(
    lossy_cp: Any,
    spec: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    return _checkpoint_tensor_for_name(
        lossy_cp,
        spec["grp"],
        str(spec["name"]),
        _logical_off(spec),
        int(spec["n"]),
        checkpoint_off=_age_off(spec),
        device=device,
        dtype=dtype,
    )


class _LiveParamSnapshot:
    """Immutable telemetry snapshot of the live low-precision model params.

    The oracle temporarily writes rollback candidates into ``param.data`` for
    paired validation.  Restoring from ``grp["p_flat"]`` is not safe enough
    because that object can be a live master/view with optimizer-specific
    aliasing.  This snapshot captures exactly what forward saw at measurement
    start and uses it for every restore during the oracle.
    """

    def __init__(
        self,
        entries: Sequence[Tuple[torch.nn.Parameter, torch.Tensor]],
    ) -> None:
        self.entries = list(entries)

    @classmethod
    @torch.no_grad()
    def capture(cls, lossy_cp: Any) -> "_LiveParamSnapshot":
        entries: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        seen = set()
        for grp in getattr(lossy_cp, "_groups", ()) or ():
            for name in grp.get("names", ()):
                param, _param_offset, _n = _param_ref_for_name(lossy_cp, str(name))
                if param is None or id(param) in seen:
                    continue
                seen.add(id(param))
                entries.append((param, param.detach().cpu().clone()))
        return cls(entries)

    @torch.no_grad()
    def restore(self) -> None:
        for param, value_cpu in self.entries:
            param.data.copy_(value_cpu, non_blocking=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @torch.no_grad()
    def max_abs_diff(self) -> float:
        max_diff = 0.0
        for param, value_cpu in self.entries:
            ref = value_cpu.to(device=param.device, dtype=param.dtype, non_blocking=False)
            diff = torch.max(torch.abs(param.detach().float() - ref.detach().float()))
            max_diff = max(max_diff, float(diff.detach().item()))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return float(max_diff)


@torch.no_grad()
def _eval_loss_bank_values(
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    *,
    group: Any = None,
) -> Tuple[float, List[float]]:
    if not loss_bank:
        return 0.0, []
    values: List[float] = []
    for forward_loss, rng_state in loss_bank:
        local = float(_eval_loss(forward_loss, rng_state=rng_state))
        values.append(float(_all_reduce_mean_float(local, group=group)))
    mean = float(sum(values) / float(max(1, len(values))))
    return mean, values


@torch.no_grad()
def _eval_checkpoint_loss_values(
    *,
    lossy_cp: Any,
    model: Any,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    group: Any = None,
) -> Tuple[float, List[float]]:
    try:
        _set_params_to_checkpoint(lossy_cp)
        return _eval_loss_bank_values(loss_bank, group=group)
    finally:
        _restore_current_params(lossy_cp)


def _pipeline_forward_loss_state(
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
) -> Optional[Tuple[Any, Tuple[Any, ...]]]:
    engine = getattr(forward_loss, "_proposal_pipeline_engine", None)
    cached = getattr(forward_loss, "_proposal_pipeline_cached", None)
    if engine is None or cached is None or not _is_pipeline_engine(engine):
        return None
    return engine, tuple(cached)


def _pipeline_backward_from_cached(engine: Any, cached_microbatches: Sequence[Any]) -> None:
    from ..refresh.hessian_refresh import _pipeline_exec_batch

    _pipeline_exec_batch(
        engine,
        list(cached_microbatches),
        backward=True,
    )


def _backward_from_forward_loss(
    *,
    model: Any,
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
) -> None:
    pipeline_state = _pipeline_forward_loss_state(forward_loss)
    if pipeline_state is not None:
        engine, cached = pipeline_state
        _pipeline_backward_from_cached(engine, cached)
        return

    module = model[0] if isinstance(model, (list, tuple)) else model
    loss, _ = forward_loss()
    if loss.dim() > 0:
        loss = loss.mean()
    if hasattr(module, "backward") and hasattr(module, "optimizer"):
        module.backward(loss)
    else:
        loss.backward(retain_graph=False)


def _paired_delta_stats(
    *,
    base: Sequence[float],
    policy: Sequence[float],
    seed: int,
) -> Dict[str, Any]:
    base_np = np.asarray(list(base), dtype=np.float64)
    policy_np = np.asarray(list(policy), dtype=np.float64)
    n = int(min(base_np.size, policy_np.size))
    if n <= 0:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "positive_ratio": 0.0,
            "min": 0.0,
            "max": 0.0,
            "base_batches": [],
            "policy_batches": [],
            "delta_batches": [],
        }
    base_np = base_np[:n]
    policy_np = policy_np[:n]
    delta = policy_np - base_np
    bootstrap_n = max(0, int(os.environ.get(
        "PROPOSAL_INDEPENDENT_ORACLE_BOOTSTRAP",
        "2000",
    ) or 0))
    if n >= 2 and bootstrap_n > 0:
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, n, size=(bootstrap_n, n), endpoint=False)
        means = delta[idx].mean(axis=1)
        ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    else:
        ci_low = ci_high = float(delta.mean())

    save_values = _env_flag("PROPOSAL_INDEPENDENT_ORACLE_SAVE_BATCH_VALUES")
    if "PROPOSAL_INDEPENDENT_ORACLE_SAVE_BATCH_VALUES" not in os.environ:
        save_values = True
    return {
        "n": int(n),
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "std": float(delta.std(ddof=1)) if n >= 2 else 0.0,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "positive_ratio": float(np.count_nonzero(delta > 0.0) / float(n)),
        "min": float(delta.min()),
        "max": float(delta.max()),
        "base_batches": [float(x) for x in base_np.tolist()] if save_values else None,
        "policy_batches": [float(x) for x in policy_np.tolist()] if save_values else None,
        "delta_batches": [float(x) for x in delta.tolist()] if save_values else None,
    }


@torch.no_grad()
def _eval_checkpoint_loss(
    *,
    lossy_cp: Any,
    model: Any,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
) -> float:
    try:
        _set_params_to_checkpoint(lossy_cp)
        return float(
            _eval_loss_bank(
                loss_bank,
                group=getattr(lossy_cp, "_proposal_cpu_pg", None),
            )
        )
    finally:
        _restore_current_params(lossy_cp)


def _checkpoint_tensor_for_name(
    lossy_cp: Any,
    grp: Dict[str, Any],
    name: str,
    local_off: int,
    n: int,
    *,
    checkpoint_off: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    """Return this run's checkpoint image for one tracked tensor.

    Older oracle/test paths kept a per-group ``theta_cp_flat``.  ZeRO1/2
    test_dram keeps the same logical image in ExpertState dense buffers, keyed
    by tensor name.  This helper keeps the rest of the oracle independent of
    that storage detail.
    """

    n = int(n)
    if n <= 0:
        return None
    cp_off = int(local_off if checkpoint_off is None else checkpoint_off)
    c_flat = grp.get("theta_cp_flat")
    if isinstance(c_flat, torch.Tensor):
        if cp_off < 0 or cp_off + n > int(c_flat.numel()):
            return None
        out = c_flat.detach().reshape(-1).narrow(0, cp_off, n)
    else:
        expert_state = getattr(lossy_cp, "_expert_state", None)
        dense_buffers = getattr(expert_state, "_dense_buffers", None)
        tname_to_eid = getattr(lossy_cp, "_tname_to_eid", {}) or {}
        if not isinstance(dense_buffers, dict):
            return None
        eid = tname_to_eid.get(str(name))
        buffers = dense_buffers.get(eid)
        entry = buffers.get(str(name)) if isinstance(buffers, dict) else None
        tensor = entry.get("p") if isinstance(entry, dict) else None
        if not isinstance(tensor, torch.Tensor) or int(tensor.numel()) < n:
            return None
        out = tensor.detach().reshape(-1).narrow(0, 0, n)
    if device is not None or dtype is not None:
        out = out.to(
            device=device if device is not None else out.device,
            dtype=dtype if dtype is not None else out.dtype,
            non_blocking=True,
        )
    return out


def _checkpoint_values_for_offsets(
    lossy_cp: Any,
    grp: Dict[str, Any],
    local_offsets: np.ndarray,
    *,
    checkpoint_offsets: Optional[np.ndarray] = None,
) -> np.ndarray:
    local_offsets = np.asarray(local_offsets, dtype=np.int64).reshape(-1)
    cp_offsets = (
        local_offsets
        if checkpoint_offsets is None
        else np.asarray(checkpoint_offsets, dtype=np.int64).reshape(-1)
    )
    out = np.zeros(local_offsets.shape[0], dtype=np.float32)
    if local_offsets.size == 0:
        return out
    if cp_offsets.shape[0] != local_offsets.shape[0]:
        raise ValueError("checkpoint_offsets must have the same length as local_offsets")
    c_flat = grp.get("theta_cp_flat")
    if isinstance(c_flat, torch.Tensor):
        idx = torch.as_tensor(cp_offsets, dtype=torch.long, device=c_flat.device)
        return torch.index_select(c_flat.detach().reshape(-1), 0, idx).detach().float().cpu().numpy()

    group_idx = -1
    for i, cand in enumerate(getattr(lossy_cp, "_groups", ()) or ()):
        if cand is grp:
            group_idx = int(i)
            break
    for spec in _group_entry_specs(lossy_cp, group_idx, grp):
        local_base = _logical_off(spec)
        n = int(spec["n"])
        lo = int(np.searchsorted(local_offsets, local_base, side="left"))
        hi = int(np.searchsorted(local_offsets, local_base + n, side="left"))
        if hi > lo:
            offsets = local_offsets[lo:hi] - np.int64(local_base)
            cp = _checkpoint_tensor_for_name(
                lossy_cp,
                grp,
                str(spec["name"]),
                local_base,
                n,
                checkpoint_off=_age_off(spec),
            )
            if cp is not None:
                if cp.is_cuda:
                    idx = torch.as_tensor(offsets, dtype=torch.long, device=cp.device)
                    vals = torch.index_select(cp.detach().reshape(-1), 0, idx)
                    out[lo:hi] = vals.detach().float().cpu().numpy()
                else:
                    out[lo:hi] = cp.detach().reshape(-1).cpu().numpy()[offsets].astype(
                        np.float32, copy=False,
                    )
    return out


@torch.no_grad()
def _set_params_to_checkpoint(lossy_cp: Any) -> None:
    for group_idx, grp in enumerate(getattr(lossy_cp, "_groups", ()) or ()):
        for spec in _group_entry_specs(lossy_cp, group_idx, grp):
            n = int(spec["n"])
            param, param_offset, param_n = _param_ref_for_name(lossy_cp, str(spec["name"]))
            if param is not None and n > 0:
                src = _checkpoint_tensor_for_spec(lossy_cp, spec)
                if src is None:
                    continue
                _copy_flat_to_param(
                    param,
                    src,
                    param_offset=param_offset,
                    n=min(n, param_n),
                )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _restore_current_params(lossy_cp: Any) -> None:
    snapshot = getattr(lossy_cp, _SNAPSHOT_ATTR, None)
    if snapshot is not None:
        snapshot.restore()
        return
    for group_idx, grp in enumerate(getattr(lossy_cp, "_groups", ()) or ()):
        p_flat = grp.get("p_flat")
        if p_flat is None:
            continue
        for spec in _group_entry_specs(lossy_cp, group_idx, grp):
            n = int(spec["n"])
            param, param_offset, param_n = _param_ref_for_name(lossy_cp, str(spec["name"]))
            if param is not None and n > 0:
                _copy_flat_to_param(
                    param,
                    p_flat.narrow(0, _elem_off(spec), n),
                    param_offset=param_offset,
                    n=min(n, param_n),
                )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _set_params_perturbed(lossy_cp: Any, *, scale: float) -> None:
    for group_idx, grp in enumerate(getattr(lossy_cp, "_groups", ()) or ()):
        p_flat = grp.get("p_flat")
        if p_flat is None:
            continue
        for spec in _group_entry_specs(lossy_cp, group_idx, grp):
            n = int(spec["n"])
            param, param_offset, param_n = _param_ref_for_name(lossy_cp, str(spec["name"]))
            if param is not None and n > 0:
                p = p_flat.narrow(0, _elem_off(spec), n)
                c = _checkpoint_tensor_for_spec(lossy_cp, spec, device=p.device, dtype=p.dtype)
                if c is None:
                    continue
                value = p + float(scale) * (p - c)
                _copy_flat_to_param(
                    param,
                    value,
                    param_offset=param_offset,
                    n=min(n, param_n),
                )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _full_e_stats(lossy_cp: Any) -> Dict[str, float]:
    local_sumsq = 0.0
    local_abs = 0.0
    local_max = 0.0
    local_n = 0
    for group_idx, grp in enumerate(getattr(lossy_cp, "_groups", ()) or ()):
        p = grp.get("p_flat")
        if p is None:
            continue
        for spec in _group_entry_specs(lossy_cp, group_idx, grp):
            n = int(spec["n"])
            if n <= 0:
                continue
            p_slice = p.narrow(0, _elem_off(spec), n)
            c = _checkpoint_tensor_for_spec(lossy_cp, spec, device=p_slice.device, dtype=p_slice.dtype)
            if c is not None:
                e = (p_slice - c).detach().float()
                local_sumsq += float(torch.dot(e, e).detach().item())
                local_abs += float(torch.sum(torch.abs(e)).detach().item())
                local_max = max(local_max, float(torch.max(torch.abs(e)).detach().item()))
                local_n += int(e.numel())
    global_sumsq = _all_reduce_sum_float(local_sumsq, group=getattr(lossy_cp, "_proposal_cpu_pg", None))
    global_abs = _all_reduce_sum_float(local_abs, group=getattr(lossy_cp, "_proposal_cpu_pg", None))
    global_n = _all_reduce_sum_int(local_n, group=getattr(lossy_cp, "_proposal_cpu_pg", None))
    global_max = _all_reduce_max_float(local_max, group=getattr(lossy_cp, "_proposal_cpu_pg", None))
    return {
        "global_l2": math.sqrt(max(0.0, float(global_sumsq))),
        "global_rms": math.sqrt(max(0.0, float(global_sumsq)) / float(max(1, global_n))),
        "global_abs_mean": float(global_abs) / float(max(1, global_n)),
        "global_max": float(global_max),
    }


def _global_error_rms(lossy_cp: Any) -> float:
    """Return only the checkpoint-error RMS needed to scale sampled HVP."""

    local_sumsq = 0.0
    local_n = 0
    for group_idx, grp in enumerate(getattr(lossy_cp, "_groups", ()) or ()):
        p = grp.get("p_flat")
        if p is None:
            continue
        for spec in _group_entry_specs(lossy_cp, group_idx, grp):
            n = int(spec["n"])
            if n <= 0:
                continue
            p_slice = p.narrow(0, _elem_off(spec), n)
            checkpoint = _checkpoint_tensor_for_spec(
                lossy_cp,
                spec,
                device=p_slice.device,
                dtype=p_slice.dtype,
            )
            if checkpoint is None:
                continue
            error = (p_slice - checkpoint).detach().float()
            local_sumsq += float(torch.dot(error, error).detach().item())
            local_n += int(error.numel())
    group = getattr(lossy_cp, "_proposal_cpu_pg", None)
    global_sumsq = _all_reduce_sum_float(local_sumsq, group=group)
    global_n = _all_reduce_sum_int(local_n, group=group)
    return math.sqrt(max(0.0, global_sumsq) / float(max(1, global_n)))


def _full_first_order_dot(lossy_cp: Any) -> float:
    local = 0.0
    for group_idx, grp in enumerate(getattr(lossy_cp, "_groups", ()) or ()):
        p = grp.get("p_flat")
        lin = grp.get(_PRODUCTION_FIRST_ORDER_KEY)
        if p is None or lin is None:
            continue
        for spec in _group_entry_specs(lossy_cp, group_idx, grp):
            n = int(spec["n"])
            if n <= 0:
                continue
            off = _elem_off(spec)
            p_slice = p.narrow(0, off, n)
            c = _checkpoint_tensor_for_spec(lossy_cp, spec, device=p_slice.device, dtype=p_slice.dtype)
            if c is not None:
                e = (p_slice - c).detach().float()
                linear = lin.narrow(0, off, n).detach().float()
                local += float(torch.dot(-linear, e).detach().item())
    return _all_reduce_sum_float(local, group=getattr(lossy_cp, "_proposal_cpu_pg", None))


def _first_order_available(lossy_cp: Any) -> bool:
    for grp in getattr(lossy_cp, "_groups", ()) or ():
        if isinstance(grp.get(_PRODUCTION_FIRST_ORDER_KEY), torch.Tensor):
            return True
    return False


@torch.no_grad()
def _full_agebit_proxy_terms(lossy_cp: Any) -> Optional[Tuple[float, float]]:
    h_lut = getattr(lossy_cp, "_h_lookup_lut", None)
    h_values = getattr(h_lut, "values", None)
    log_v_min = getattr(h_lut, "log_v_min", None)
    inv_width = getattr(h_lut, "inv_log_v_width", None)
    age_state = getattr(lossy_cp, "_age2bit_score_state", None)
    age_words = getattr(age_state, "age_words", None)
    geom_lut = getattr(age_state, "ehat_geom_lut", None)
    if not (
        isinstance(h_values, torch.Tensor)
        and isinstance(log_v_min, torch.Tensor)
        and isinstance(inv_width, torch.Tensor)
        and isinstance(age_words, torch.Tensor)
        and isinstance(geom_lut, torch.Tensor)
    ):
        return None
    bins = int(h_values.numel())
    if bins <= 0:
        return None
    if not _first_order_available(lossy_cp):
        return None

    lr_over_bc1 = float(getattr(lossy_cp, "_adam_lr_over_bc1_for_selection", 1.0e-4))
    sqrt_bc2_inv = float(getattr(lossy_cp, "_adam_sqrt_bc2_inv_for_selection", 1.0))
    eps = float(getattr(lossy_cp, "_adam_eps_for_selection", 1.0e-8))
    log_min = log_v_min.detach().float()
    inv_w = inv_width.detach().float()
    flat_geom = geom_lut.detach().float().reshape(-1)
    h_vals = h_values.detach().float()
    chunk = max(1024, int(os.environ.get(
        "PROPOSAL_INDEPENDENT_ORACLE_PROXY_CHUNK",
        "4194304",
    ) or 4194304))

    local_r1 = 0.0
    local_r2 = 0.0
    for group_idx, grp in enumerate(getattr(lossy_cp, "_groups", ()) or ()):
        m_all = grp.get("m_flat")
        v_all = grp.get("v_flat")
        lin_all = grp.get(_PRODUCTION_FIRST_ORDER_KEY)
        if not (
            isinstance(m_all, torch.Tensor)
            and isinstance(v_all, torch.Tensor)
            and isinstance(lin_all, torch.Tensor)
        ):
            continue
        device = v_all.device
        for spec in _group_entry_specs(lossy_cp, group_idx, grp):
            elem = _elem_off(spec)
            age = _age_off(spec)
            total = int(spec["n"])
            if total <= 0:
                continue
            for rel_start in range(0, total, chunk):
                n = min(chunk, total - rel_start)
                global_idx = torch.arange(
                    int(age) + rel_start,
                    int(age) + rel_start + n,
                    device=device,
                    dtype=torch.long,
                )
                word_idx = torch.div(global_idx, 16, rounding_mode="floor")
                valid = word_idx < int(age_words.numel())
                age_code = torch.zeros(n, dtype=torch.long, device=device)
                if bool(valid.any().item()):
                    words = torch.index_select(age_words, 0, word_idx[valid]).to(torch.int64)
                    shift = ((global_idx[valid] & 15) << 1).to(torch.int64)
                    age_code[valid] = (words >> shift) & 3

                m = m_all.narrow(0, elem + rel_start, n).detach().float()
                v = v_all.narrow(0, elem + rel_start, n).detach().float()
                linear = lin_all.narrow(0, elem + rel_start, n).detach().float()
                pos = (torch.log(torch.clamp(v, min=1.0e-30)) - log_min) * inv_w
                pos = torch.clamp(pos, min=0.0, max=float(bins - 1))
                lo = torch.floor(pos).to(torch.long)
                hi = torch.clamp(lo + 1, max=bins - 1)
                frac = pos - lo.float()
                h_abs = h_vals[lo] + frac * (h_vals[hi] - h_vals[lo])
                geom_idx_lo = torch.clamp(age_code * bins + lo, max=int(flat_geom.numel()) - 1)
                geom_idx_hi = torch.clamp(age_code * bins + hi, max=int(flat_geom.numel()) - 1)
                geom = flat_geom[geom_idx_lo] + frac * (flat_geom[geom_idx_hi] - flat_geom[geom_idx_lo])
                denom = torch.sqrt(torch.clamp(v, min=0.0)) * float(sqrt_bc2_inv) + float(eps)
                e_hat = (-float(lr_over_bc1) * m / denom) * geom
                local_r1 += float(torch.sum(-linear * e_hat).detach().item())
                local_r2 += float(torch.sum(0.5 * h_abs * e_hat * e_hat).detach().item())

    group = getattr(lossy_cp, "_proposal_cpu_pg", None)
    return (
        _all_reduce_sum_float(local_r1, group=group),
        _all_reduce_sum_float(local_r2, group=group),
    )


def _build_sample(lossy_cp: Any, proposal_config: Any, *, rank: int) -> Dict[str, Any]:
    max_samples = int(getattr(proposal_config, "independent_oracle_rank_sample", 1048576) or 0)
    groups = list(getattr(lossy_cp, "_groups", ()) or ())
    spans: List[Tuple[int, Dict[str, Any], int, int]] = []
    total = 0
    for group_idx, grp in enumerate(groups):
        for spec in _group_entry_specs(lossy_cp, group_idx, grp):
            n = int(spec["n"])
            if n <= 0:
                continue
            spans.append((group_idx, spec, total, total + n))
            total += n
    n_sample = min(max_samples, total)
    if n_sample <= 0:
        return {"items": [], "n": 0}
    seed = _ORACLE_SEED
    rng = np.random.default_rng(seed + int(rank) * 1000003)
    ids = np.sort(rng.choice(total, size=n_sample, replace=False).astype(np.int64))
    items = []
    for group_idx, spec, lo, hi in spans:
        left = int(np.searchsorted(ids, lo, side="left"))
        right = int(np.searchsorted(ids, hi, side="left"))
        if right <= left:
            continue
        rel_offsets = ids[left:right] - np.int64(lo)
        local_offsets = np.int64(_logical_off(spec)) + rel_offsets
        elem_offsets = np.int64(_elem_off(spec)) + rel_offsets
        age_offsets = np.int64(_age_off(spec)) + rel_offsets
        device = groups[group_idx]["p_flat"].device
        items.append({
            "group_idx": int(group_idx),
            "name": str(spec["name"]),
            "entry_idx": int(spec.get("entry_idx", -1)),
            "positions": np.arange(left, right, dtype=np.int64),
            "entry_offsets_np": rel_offsets.astype(np.int64, copy=False),
            "entry_offsets_t": torch.as_tensor(rel_offsets, dtype=torch.long, device=device),
            "local_offsets_np": local_offsets.astype(np.int64, copy=False),
            "local_offsets_t": torch.as_tensor(local_offsets, dtype=torch.long, device=device),
            "elem_offsets_np": elem_offsets.astype(np.int64, copy=False),
            "elem_offsets_t": torch.as_tensor(elem_offsets, dtype=torch.long, device=device),
            "age_offsets_np": age_offsets.astype(np.int64, copy=False),
        })
    return {"items": items, "n": int(n_sample)}


@torch.no_grad()
def _gather_sample_payload(
    lossy_cp: Any,
    sample: Dict[str, Any],
    *,
    iteration: int,
) -> Dict[str, np.ndarray]:
    n = int(sample.get("n", 0))
    out = {
        "e": np.zeros(n, dtype=np.float32),
        "m": np.zeros(n, dtype=np.float32),
        "v": np.zeros(n, dtype=np.float32),
        "h_abs": np.zeros(n, dtype=np.float32),
        "e_age": np.zeros(n, dtype=np.float32),
        "age_code": np.zeros(n, dtype=np.uint8),
        "global_id": np.zeros(n, dtype=np.int64),
        "force": np.zeros(n, dtype=bool),
        "score_m_diag": np.zeros(n, dtype=np.float32),
        "score_agebit": np.zeros(n, dtype=np.float32),
    }
    h_lut = getattr(lossy_cp, "_h_lookup_lut", None)
    h_values = getattr(h_lut, "values", None)
    log_v_min = getattr(h_lut, "log_v_min", None)
    inv_width = getattr(h_lut, "inv_log_v_width", None)
    geom_lut = getattr(getattr(lossy_cp, "_age2bit_score_state", None), "ehat_geom_lut", None)
    age_words = getattr(getattr(lossy_cp, "_age2bit_score_state", None), "age_words", None)
    h_values_np = h_values.detach().float().cpu().numpy() if h_values is not None else np.zeros(1, dtype=np.float32)
    geom_np = (
        geom_lut.detach().float().cpu().numpy().reshape(-1)
        if geom_lut is not None
        else np.zeros(4 * h_values_np.size, dtype=np.float32)
    )
    log_min_f = float(log_v_min.detach().float().item()) if log_v_min is not None else 0.0
    inv_w_f = float(inv_width.detach().float().item()) if inv_width is not None else 1.0
    age_words_np = age_words.detach().cpu().numpy().astype(np.uint32, copy=True) if age_words is not None else None
    lr_over_bc1 = float(getattr(lossy_cp, "_adam_lr_over_bc1_for_selection", 1.0e-4))
    sqrt_bc2_inv = float(getattr(lossy_cp, "_adam_sqrt_bc2_inv_for_selection", 1.0))
    eps = float(getattr(lossy_cp, "_adam_eps_for_selection", 1.0e-8))
    for item in sample.get("items", ()):
        grp = lossy_cp._groups[int(item["group_idx"])]
        pos = item["positions"]
        idx = item.get("elem_offsets_t", item["local_offsets_t"])
        local_offsets_np = np.asarray(item["local_offsets_np"], dtype=np.int64)
        age_offsets_np = np.asarray(item.get("age_offsets_np", local_offsets_np), dtype=np.int64)
        p = torch.index_select(grp["p_flat"], 0, idx).detach().float().cpu().numpy()
        c = _checkpoint_values_for_offsets(
            lossy_cp,
            grp,
            local_offsets_np,
            checkpoint_offsets=age_offsets_np,
        )
        m = torch.index_select(grp["m_flat"], 0, idx).detach().float().cpu().numpy()
        v = torch.index_select(grp["v_flat"], 0, idx).detach().float().cpu().numpy()
        e = (p - c).astype(np.float32, copy=False)
        h_abs, lut_idx = _lookup_h_np(v, h_values_np, log_min_f, inv_w_f)
        global_ids = age_offsets_np
        age_code = _unpack_age(age_words_np, global_ids)
        geom = _lookup_geom_np(geom_np, lut_idx, age_code, h_values_np.size)
        denom = np.sqrt(np.maximum(v, 0.0)).astype(np.float32) * np.float32(sqrt_bc2_inv) + np.float32(eps)
        e_age = (-np.float32(lr_over_bc1) * m / denom * geom).astype(np.float32, copy=False)
        force = (age_code == 3) & _age2_phase_hit_np(
            global_ids,
            step=int(iteration),
            period=128,
            salt=0x51ED270B,
        )
        out["e"][pos] = e
        out["m"][pos] = m
        out["v"][pos] = v
        out["h_abs"][pos] = h_abs
        out["e_age"][pos] = e_age
        out["age_code"][pos] = age_code
        out["global_id"][pos] = global_ids
        out["force"][pos] = force
        out["score_m_diag"][pos] = np.maximum(-m * e + 0.5 * h_abs * e * e, 0.0)
        out["score_agebit"][pos] = np.maximum(-m * e_age + 0.5 * h_abs * e_age * e_age, 0.0)
    return out


def _hvp_for_current_error(
    *,
    lossy_cp: Any,
    model: Any,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    fd_eps: float,
    rms_e: float,
    sample: Dict[str, Any],
    compute_full_quadratic: bool = True,
) -> Tuple[float, np.ndarray]:
    if not loss_bank:
        return 0.0, np.zeros(int(sample.get("n", 0)), dtype=np.float32)
    sample_acc = np.zeros(int(sample.get("n", 0)), dtype=np.float32)
    r2_acc = 0.0
    inv_rms = 1.0 / float(rms_e)
    try:
        for forward_loss, rng_state in loss_bank:
            _restore_current_params(lossy_cp)
            _set_params_perturbed(lossy_cp, scale=float(fd_eps) * inv_rms)
            dot_plus, grad_plus = _backward_dot_and_sample(
                lossy_cp=lossy_cp,
                model=model,
                forward_loss=forward_loss,
                rng_state=rng_state,
                sample=sample,
                compute_full_dot=compute_full_quadratic,
            )
            _restore_current_params(lossy_cp)
            _set_params_perturbed(lossy_cp, scale=-float(fd_eps) * inv_rms)
            dot_minus, grad_minus = _backward_dot_and_sample(
                lossy_cp=lossy_cp,
                model=model,
                forward_loss=forward_loss,
                rng_state=rng_state,
                sample=sample,
                compute_full_dot=compute_full_quadratic,
            )
            if compute_full_quadratic:
                local_ehe = (
                    float(rms_e)
                    * (float(dot_plus) - float(dot_minus))
                    / (2.0 * float(fd_eps))
                )
                global_ehe = _all_reduce_sum_float(
                    local_ehe,
                    group=getattr(lossy_cp, "_proposal_cpu_pg", None),
                )
                r2_acc += 0.5 * float(global_ehe)
            sample_acc += (
                float(rms_e)
                * (grad_plus.astype(np.float32, copy=False) - grad_minus.astype(np.float32, copy=False))
                / np.float32(2.0 * float(fd_eps))
            ).astype(np.float32, copy=False)
    finally:
        _restore_current_params(lossy_cp)
    inv = np.float32(1.0 / float(max(1, len(loss_bank))))
    return float(r2_acc) / float(max(1, len(loss_bank))), (sample_acc * inv).astype(np.float32, copy=False)


def _run_oneshot_hvp_actual(
    *,
    lossy_cp: Any,
    model: Any,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    hvp_loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    fd_eps: float,
    sample: Dict[str, Any],
    loss_base: float,
    loss_base_batches: Sequence[float],
    proposal_config: Any,
    iteration: int,
    rank: int,
    world: int,
) -> Tuple[float, np.ndarray, List[Dict[str, Any]]]:
    backend = str(os.environ.get(
        "PROPOSAL_INDEPENDENT_ORACLE_ONESHOT_BACKEND",
        "gpu",
    )).strip().lower()
    if backend != "cpu_numpy":
        return _run_oneshot_hvp_actual_gpu(
            lossy_cp=lossy_cp,
            model=model,
            loss_bank=loss_bank,
            hvp_loss_bank=hvp_loss_bank,
            fd_eps=fd_eps,
            sample=sample,
            loss_base=loss_base,
            loss_base_batches=loss_base_batches,
            proposal_config=proposal_config,
            iteration=iteration,
            rank=rank,
            world=world,
        )

    specs = _tensor_specs(lossy_cp)
    e_parts = _direction_parts(lossy_cp, specs)
    total_n_local = int(sum(int(e.size) for e in e_parts))
    total_n = int(_all_reduce_sum_int(total_n_local, group=getattr(lossy_cp, "_proposal_cpu_pg", None)))
    if total_n <= 0:
        return 0.0, np.zeros(int(sample.get("n", 0)), dtype=np.float32), []

    h_e_parts = _hvp_parts_for_direction(
        lossy_cp=lossy_cp,
        specs=specs,
        e_parts=e_parts,
        model=model,
        loss_bank=hvp_loss_bank,
        fd_eps=float(fd_eps),
        total_n=int(total_n),
    )
    r2_full = _quadratic_from_parts(e_parts, h_e_parts, lossy_cp=lossy_cp)
    hvp_sample = _sample_parts_by_offsets(lossy_cp, specs, h_e_parts, sample)
    linear_parts = _first_order_parts(lossy_cp, specs)
    h_abs_parts = _diag_abs_h_parts(lossy_cp, specs)
    age_score_parts = _agebit_score_parts(
        lossy_cp,
        specs,
        iteration=int(iteration),
    )

    score_hvp_directional = [
        np.maximum(-lin * e + np.float32(0.5) * e * h_e, np.float32(0.0)).astype(np.float32, copy=False)
        for lin, e, h_e in zip(linear_parts, e_parts, h_e_parts)
    ]
    score_diag = [
        np.maximum(-lin * e + np.float32(0.5) * h * e * e, np.float32(0.0)).astype(np.float32, copy=False)
        for lin, e, h in zip(linear_parts, e_parts, h_abs_parts)
    ]

    ratio = float(getattr(proposal_config, "topk_ratio", 0.0) or 0.0)
    rows: List[Dict[str, Any]] = []
    compute_selected_hvp = _env_flag("PROPOSAL_INDEPENDENT_ORACLE_SELECTED_HVP")
    use_sample_cdf_threshold = _env_flag(
        "PROPOSAL_INDEPENDENT_ORACLE_ONESHOT_SAMPLE_CDF_THRESHOLD"
    )

    first_order_name = _PRODUCTION_FIRST_ORDER_NAME
    for policy_name, score_parts in (
        (f"{first_order_name}+diag_abs_h+agebit", age_score_parts),
        (f"{first_order_name}+diag_abs_h+true_e", score_diag),
        (f"{first_order_name}+hvp_directional+true_e", score_hvp_directional),
    ):
        if use_sample_cdf_threshold:
            score_sample = _sample_parts_by_offsets(lossy_cp, specs, score_parts, sample)
            masks, info = _select_masks_sample_cdf_threshold(
                score_parts,
                score_sample,
                ratio,
                lossy_cp=lossy_cp,
            )
        else:
            masks, info = _select_masks_global_exact(score_parts, ratio, lossy_cp=lossy_cp)
        loss_rb = _eval_selected_masks_loss(
            lossy_cp=lossy_cp,
            specs=specs,
            selected_masks=masks,
            loss_bank=loss_bank,
        )
        actual_remaining = float(loss_rb) - float(loss_base)
        first_remaining = _linear_sum_parts(
            linear_parts,
            _remaining_parts(e_parts, masks),
            lossy_cp=lossy_cp,
        )
        full_quad_remaining = None
        full_proxy = None
        third = None
        if compute_selected_hvp:
            e_remaining = _remaining_parts(e_parts, masks)
            full_quad_remaining = _hvp_quadratic_for_direction(
                lossy_cp=lossy_cp,
                specs=specs,
                e_parts=e_remaining,
                model=model,
                loss_bank=hvp_loss_bank,
                fd_eps=float(fd_eps),
                total_n=int(total_n),
            )
            full_proxy = float(first_remaining) + float(full_quad_remaining)
            third = float(actual_remaining) - float(full_proxy)
            del e_remaining
        rows.append({
            "mode": "independent_oneshot_hvp_actual",
            "step": int(iteration),
            "rank": int(rank),
            "world": int(world),
            "policy": policy_name,
            "score_first_order_name": str(first_order_name),
            "topk_ratio": float(ratio),
            "selection_scope": str(info["selection_scope"]),
            "local_elements": int(total_n_local),
            "global_elements": int(total_n),
            "local_selected": int(info["local_selected"]),
            "global_selected": int(info["global_selected"]),
            "global_selected_ratio": (
                float(info["global_selected_ratio"])
                if "global_selected_ratio" in info else _safe_div(float(info["global_selected"]), float(total_n))
            ),
            "threshold": info.get("threshold"),
            "sample_elements_for_threshold": info.get("sample_elements"),
            "sample_global_elements_for_threshold": info.get("sample_global_elements"),
            "dkw_epsilon_95": info.get("dkw_epsilon_95"),
            "dkw_epsilon_99": info.get("dkw_epsilon_99"),
            "loss_base": float(loss_base),
            "loss_after_selected_save": float(loss_rb),
            "actual_loss_remaining": float(actual_remaining),
            "first_order_remaining": float(first_remaining),
            "second_order_full_hvp_remaining": (
                float(full_quad_remaining) if full_quad_remaining is not None else None
            ),
            "first_plus_full_hvp_remaining": (
                float(full_proxy) if full_proxy is not None else None
            ),
            "third_order_residual_vs_full_hvp": (
                float(third) if third is not None else None
            ),
            "third_order_abs_over_actual_abs": (
                _safe_div(abs(float(third)), abs(float(actual_remaining)))
                if third is not None else None
            ),
            "third_order_abs_over_proxy_abs": (
                _safe_div(abs(float(third)), abs(float(full_proxy)))
                if third is not None and full_proxy is not None else None
            ),
            "selected_hvp_residual_computed": bool(compute_selected_hvp),
            "fd_eps": float(fd_eps),
            "actual_batches": int(len(loss_bank)),
            "hvp_batches": int(len(hvp_loss_bank)),
        })
    _restore_current_params(lossy_cp)
    return float(r2_full), hvp_sample, rows


def _run_oneshot_hvp_actual_gpu(
    *,
    lossy_cp: Any,
    model: Any,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    hvp_loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    fd_eps: float,
    sample: Dict[str, Any],
    loss_base: float,
    loss_base_batches: Sequence[float],
    proposal_config: Any,
    iteration: int,
    rank: int,
    world: int,
) -> Tuple[float, np.ndarray, List[Dict[str, Any]]]:
    specs = _tensor_specs(lossy_cp)
    total_n_local = int(sum(int(spec["n"]) for spec in specs))
    group = getattr(lossy_cp, "_proposal_cpu_pg", None)
    total_n = int(_all_reduce_sum_int(total_n_local, group=group))
    if total_n <= 0:
        return 0.0, np.zeros(int(sample.get("n", 0)), dtype=np.float32), []

    h_e_tensors = _hvp_cpu_tensors_for_current_error(
        lossy_cp=lossy_cp,
        specs=specs,
        model=model,
        loss_bank=hvp_loss_bank,
        fd_eps=float(fd_eps),
        total_n=int(total_n),
    )
    r2_full = _quadratic_current_error_with_hvp_parts(
        lossy_cp=lossy_cp,
        specs=specs,
        h_e_tensors=h_e_tensors,
    )
    hvp_sample = _sample_tensors_by_offsets(lossy_cp, specs, h_e_tensors, sample)
    sample_payload = _gather_sample_payload(
        lossy_cp,
        sample,
        iteration=int(iteration),
    )

    linear_sample = sample_payload["m"]
    diag_true_sample = sample_payload["score_m_diag"]
    agebit_sample = sample_payload["score_agebit"]

    e_sample = sample_payload["e"]
    h_abs_sample = sample_payload["h_abs"]
    hvp_directional_sample = np.maximum(
        -linear_sample * e_sample + np.float32(0.5) * e_sample * hvp_sample,
        np.float32(0.0),
    ).astype(np.float32, copy=False)
    agebit_force_sample = _score_with_force(agebit_sample, sample_payload["force"])

    first_order_name = _PRODUCTION_FIRST_ORDER_NAME
    threshold_policies: Tuple[Tuple[str, str, np.ndarray], ...] = (
        (f"{first_order_name}+diag_abs_h+agebit+force", "agebit_force", agebit_force_sample),
        (f"{first_order_name}+diag_abs_h+true_e", "diag_true_e", diag_true_sample),
        (f"{first_order_name}+hvp_directional+true_e", "hvp_directional", hvp_directional_sample),
    )
    direct_policies: Tuple[Tuple[str, str], ...] = (
        ("random_hash", "random_hash"),
        ("round_robin_id", "round_robin"),
    )

    ratio = float(getattr(proposal_config, "topk_ratio", 0.0) or 0.0)
    rows: List[Dict[str, Any]] = []
    for policy_name, policy_kind, score_sample in threshold_policies:
        threshold_info = _threshold_from_sample_scores(
            score_sample,
            ratio,
            lossy_cp=lossy_cp,
        )
        loss_rb, loss_rb_batches, select_info = _eval_gpu_policy_loss(
            lossy_cp=lossy_cp,
            specs=specs,
            h_e_tensors=h_e_tensors,
            threshold=float(threshold_info["threshold"]),
            tie_prob=float(threshold_info["tie_prob"]),
            policy_kind=policy_kind,
            iteration=int(iteration),
            ratio=float(ratio),
            loss_bank=loss_bank,
        )
        rows.append(_oneshot_actual_row(
            iteration=iteration,
            rank=rank,
            world=world,
            policy_name=policy_name,
            first_order_name=first_order_name,
            ratio=ratio,
            total_n_local=total_n_local,
            total_n=total_n,
            loss_base=loss_base,
            loss_rb=loss_rb,
            select_info=select_info,
            threshold_info=threshold_info,
            fd_eps=fd_eps,
            actual_batches=len(loss_bank),
            hvp_batches=len(hvp_loss_bank),
            loss_base_batches=loss_base_batches,
            loss_after_batches=loss_rb_batches,
            delta_seed=int(20240625) + int(iteration),
        ))

    for policy_name, policy_kind in direct_policies:
        loss_rb, loss_rb_batches, select_info = _eval_gpu_policy_loss(
            lossy_cp=lossy_cp,
            specs=specs,
            h_e_tensors=h_e_tensors,
            threshold=float("nan"),
            tie_prob=0.0,
            policy_kind=policy_kind,
            iteration=int(iteration),
            ratio=float(ratio),
            loss_bank=loss_bank,
        )
        rows.append(_oneshot_actual_row(
            iteration=iteration,
            rank=rank,
            world=world,
            policy_name=policy_name,
            first_order_name=first_order_name,
            ratio=ratio,
            total_n_local=total_n_local,
            total_n=total_n,
            loss_base=loss_base,
            loss_rb=loss_rb,
            select_info=select_info,
            threshold_info={
                "threshold": None,
                "tie_prob": 0.0,
                "sample_elements": int(sample_payload["e"].size),
                "sample_global_elements": int(_all_reduce_sum_int(
                    int(sample_payload["e"].size),
                    group=group,
                )),
                "dkw_epsilon_95": None,
                "dkw_epsilon_99": None,
                "selection_scope": "gpu_direct_baseline",
            },
            fd_eps=fd_eps,
            actual_batches=len(loss_bank),
            hvp_batches=len(hvp_loss_bank),
            loss_base_batches=loss_base_batches,
            loss_after_batches=loss_rb_batches,
            delta_seed=int(20240625) + int(iteration),
        ))
    _restore_current_params(lossy_cp)
    return float(r2_full), hvp_sample, rows


def _hvp_cpu_tensors_for_current_error(
    *,
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    model: Any,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    fd_eps: float,
    total_n: int,
) -> List[torch.Tensor]:
    rms_e = _rms_for_current_error(lossy_cp=lossy_cp, specs=specs, total_n=total_n)
    if rms_e <= 0.0 or not loss_bank:
        return [torch.zeros(int(spec["n"]), dtype=torch.float32) for spec in specs]
    h_acc = [torch.zeros(int(spec["n"]), dtype=torch.float32) for spec in specs]
    scale = float(fd_eps) / float(rms_e)
    coef = float(rms_e) / (2.0 * float(fd_eps) * float(max(1, len(loss_bank))))
    try:
        for forward_loss, rng_state in loss_bank:
            _restore_current_params(lossy_cp)
            _set_params_plus_current_error(lossy_cp, specs, scale=scale)
            g_plus = _backward_grad_cpu_tensors(
                model=model,
                specs=specs,
                forward_loss=forward_loss,
                rng_state=rng_state,
            )
            _restore_current_params(lossy_cp)
            _set_params_plus_current_error(lossy_cp, specs, scale=-scale)
            g_minus = _backward_grad_cpu_tensors(
                model=model,
                specs=specs,
                forward_loss=forward_loss,
                rng_state=rng_state,
            )
            for dst, gp, gm in zip(h_acc, g_plus, g_minus):
                dst.add_(gp.sub(gm), alpha=float(coef))
            del g_plus, g_minus
    finally:
        _restore_current_params(lossy_cp)
    return h_acc


def _backward_grad_cpu_tensors(
    *,
    model: Any,
    specs: Sequence[Dict[str, Any]],
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
    rng_state: Optional[Dict[str, Any]],
) -> List[torch.Tensor]:
    if rng_state is not None:
        _restore_rng_state(rng_state)
    module = model[0] if isinstance(model, (list, tuple)) else model
    try:
        for param in module.parameters():
            param.grad = None
    except Exception:
        pass
    grads: List[Optional[torch.Tensor]] = [None] * len(specs)
    handles = []
    for idx, spec in enumerate(specs):
        param = spec["param"]

        def _hook(raw_grad, *, _idx=idx, _spec=spec):
            grads[_idx] = _grad_slice_for_spec(raw_grad, _spec).float().cpu()
            return torch.zeros_like(raw_grad)

        handles.append(param.register_hook(_hook))
    try:
        _backward_from_forward_loss(model=model, forward_loss=forward_loss)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        try:
            for param in module.parameters():
                param.grad = None
        except Exception:
            pass
    out: List[torch.Tensor] = []
    for grad, spec in zip(grads, specs):
        if grad is not None:
            out.append(grad)
        else:
            out.append(torch.zeros(int(spec["n"]), dtype=torch.float32))
    return out


def _rms_for_current_error(
    *,
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    total_n: int,
) -> float:
    local = 0.0
    for spec in specs:
        grp = spec["grp"]
        n = int(spec["n"])
        off = _elem_off(spec)
        p = grp["p_flat"].narrow(0, off, n)
        c = _checkpoint_tensor_for_spec(lossy_cp, spec, device=p.device, dtype=p.dtype)
        if c is None:
            continue
        e = p.detach().float() - c.detach().float()
        local += float(torch.dot(e, e).detach().item())
    global_sum = _all_reduce_sum_float(local, group=getattr(lossy_cp, "_proposal_cpu_pg", None))
    return math.sqrt(max(0.0, float(global_sum)) / float(max(1, int(total_n))))


@torch.no_grad()
def _set_params_plus_current_error(
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    *,
    scale: float,
) -> None:
    for spec in specs:
        grp = spec["grp"]
        n = int(spec["n"])
        off = _elem_off(spec)
        param = spec["param"]
        p = grp["p_flat"].narrow(0, off, n)
        c = _checkpoint_tensor_for_spec(lossy_cp, spec, device=p.device, dtype=p.dtype)
        if c is None:
            value = p.detach().float()
        else:
            value = p.detach().float() + float(scale) * (p.detach().float() - c.detach().float())
        _copy_flat_to_param(
            param,
            value,
            param_offset=int(spec.get("param_offset", 0) or 0),
            n=n,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _quadratic_current_error_with_hvp_parts(
    *,
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    h_e_tensors: Sequence[torch.Tensor],
) -> float:
    local = 0.0
    for spec, h_cpu in zip(specs, h_e_tensors):
        grp = spec["grp"]
        n = int(spec["n"])
        off = _elem_off(spec)
        p = grp["p_flat"].narrow(0, off, n)
        c = _checkpoint_tensor_for_spec(lossy_cp, spec, device=p.device, dtype=p.dtype)
        if c is None:
            continue
        e = p.detach().float() - c.detach().float()
        h = h_cpu.to(device=p.device, dtype=torch.float32)
        local += 0.5 * float(torch.dot(e, h).detach().item())
    return _all_reduce_sum_float(local, group=getattr(lossy_cp, "_proposal_cpu_pg", None))


def _tensor_specs(lossy_cp: Any) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for group_idx, grp in enumerate(getattr(lossy_cp, "_groups", ()) or ()):
        for spec in _group_entry_specs(lossy_cp, group_idx, grp):
            n = int(spec["n"])
            param, param_offset, param_n = _param_ref_for_name(lossy_cp, str(spec["name"]))
            n = int(min(n, param_n))
            if param is not None and n > 0:
                enriched = dict(spec)
                enriched["n"] = int(n)
                enriched["param"] = param
                enriched["param_offset"] = int(param_offset)
                specs.append(enriched)
    return specs


def _direction_parts(lossy_cp: Any, specs: Sequence[Dict[str, Any]]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for spec in specs:
        grp = spec["grp"]
        n = int(spec["n"])
        off = _elem_off(spec)
        p = grp["p_flat"].narrow(0, off, n)
        c = _checkpoint_tensor_for_spec(lossy_cp, spec, device=p.device, dtype=p.dtype)
        if c is None:
            out.append(np.zeros(n, dtype=np.float32))
        else:
            out.append((p.detach().float() - c.detach().float()).cpu().numpy().astype(np.float32, copy=True))
    return out


def _first_order_parts(lossy_cp: Any, specs: Sequence[Dict[str, Any]]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for spec in specs:
        grp = spec["grp"]
        n = int(spec["n"])
        off = _elem_off(spec)
        lin = grp.get(_PRODUCTION_FIRST_ORDER_KEY)
        if isinstance(lin, torch.Tensor):
            out.append(lin.narrow(0, off, n).detach().float().cpu().numpy().astype(np.float32, copy=True))
        else:
            out.append(np.zeros(n, dtype=np.float32))
    return out


def _diag_abs_h_parts(lossy_cp: Any, specs: Sequence[Dict[str, Any]]) -> List[np.ndarray]:
    h_lut = getattr(lossy_cp, "_h_lookup_lut", None)
    h_values = getattr(h_lut, "values", None)
    log_v_min = getattr(h_lut, "log_v_min", None)
    inv_width = getattr(h_lut, "inv_log_v_width", None)
    h_values_np = h_values.detach().float().cpu().numpy() if h_values is not None else np.zeros(1, dtype=np.float32)
    log_min_f = float(log_v_min.detach().float().item()) if log_v_min is not None else 0.0
    inv_w_f = float(inv_width.detach().float().item()) if inv_width is not None else 1.0
    out: List[np.ndarray] = []
    for spec in specs:
        grp = spec["grp"]
        n = int(spec["n"])
        off = _elem_off(spec)
        v = grp["v_flat"].narrow(0, off, n).detach().float().cpu().numpy()
        h_abs, _ = _lookup_h_np(v, h_values_np, log_min_f, inv_w_f)
        out.append(h_abs.astype(np.float32, copy=True))
    return out


def _agebit_score_parts(
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    *,
    iteration: int,
) -> List[np.ndarray]:
    h_lut = getattr(lossy_cp, "_h_lookup_lut", None)
    h_values = getattr(h_lut, "values", None)
    log_v_min = getattr(h_lut, "log_v_min", None)
    inv_width = getattr(h_lut, "inv_log_v_width", None)
    age_state = getattr(lossy_cp, "_age2bit_score_state", None)
    age_words = getattr(age_state, "age_words", None)
    geom_lut = getattr(age_state, "ehat_geom_lut", None)
    h_values_np = h_values.detach().float().cpu().numpy() if h_values is not None else np.zeros(1, dtype=np.float32)
    geom_np = (
        geom_lut.detach().float().cpu().numpy().reshape(-1)
        if geom_lut is not None else np.zeros(4 * h_values_np.size, dtype=np.float32)
    )
    age_words_np = age_words.detach().cpu().numpy().astype(np.uint32, copy=True) if age_words is not None else None
    log_min_f = float(log_v_min.detach().float().item()) if log_v_min is not None else 0.0
    inv_w_f = float(inv_width.detach().float().item()) if inv_width is not None else 1.0
    lr_over_bc1 = float(getattr(lossy_cp, "_adam_lr_over_bc1_for_selection", 1.0e-4))
    sqrt_bc2_inv = float(getattr(lossy_cp, "_adam_sqrt_bc2_inv_for_selection", 1.0))
    eps = float(getattr(lossy_cp, "_adam_eps_for_selection", 1.0e-8))
    out: List[np.ndarray] = []
    for spec in specs:
        grp = spec["grp"]
        n = int(spec["n"])
        off = _elem_off(spec)
        m = grp["m_flat"].narrow(0, off, n).detach().float().cpu().numpy()
        v = grp["v_flat"].narrow(0, off, n).detach().float().cpu().numpy()
        lin_t = grp.get(_PRODUCTION_FIRST_ORDER_KEY)
        lin = (
            lin_t.narrow(0, off, n).detach().float().cpu().numpy()
            if isinstance(lin_t, torch.Tensor) else np.zeros(n, dtype=np.float32)
        )
        h_abs, lut_idx = _lookup_h_np(v, h_values_np, log_min_f, inv_w_f)
        age = _age_off(spec)
        global_ids = np.arange(age, age + n, dtype=np.int64)
        age_code = _unpack_age(age_words_np, global_ids)
        geom = _lookup_geom_np(geom_np, lut_idx, age_code, h_values_np.size)
        denom = np.sqrt(np.maximum(v, 0.0)).astype(np.float32) * np.float32(sqrt_bc2_inv) + np.float32(eps)
        e_age = (-np.float32(lr_over_bc1) * m / denom * geom).astype(np.float32, copy=False)
        force = (age_code == 3) & _age2_phase_hit_np(
            global_ids,
            step=int(iteration),
            period=128,
            salt=0x51ED270B,
        )
        score = np.maximum(-lin * e_age + np.float32(0.5) * h_abs * e_age * e_age, np.float32(0.0))
        if bool(force.any()):
            finite = score[np.isfinite(score)]
            high = np.float32(float(np.max(finite)) if finite.size else 1.0)
            score = score.astype(np.float32, copy=True)
            score[force] = high * np.float32(1.0e6) + np.float32(1.0)
        out.append(score.astype(np.float32, copy=False))
    return out


def _hvp_parts_for_direction(
    *,
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    e_parts: Sequence[np.ndarray],
    model: Any,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    fd_eps: float,
    total_n: int,
) -> List[np.ndarray]:
    rms_e = _rms_for_numpy_parts(e_parts, total_n=total_n, lossy_cp=lossy_cp)
    if rms_e <= 0.0 or not loss_bank:
        return [np.zeros_like(e, dtype=np.float32) for e in e_parts]
    h_acc = [np.zeros_like(e, dtype=np.float32) for e in e_parts]
    scale = float(fd_eps) / float(rms_e)
    inv = np.float32(1.0 / float(max(1, len(loss_bank))))
    try:
        for forward_loss, rng_state in loss_bank:
            _restore_current_params(lossy_cp)
            _set_params_plus_direction_parts(lossy_cp, specs, e_parts, scale=scale)
            g_plus = _backward_grad_parts(model=model, specs=specs, forward_loss=forward_loss, rng_state=rng_state)
            _restore_current_params(lossy_cp)
            _set_params_plus_direction_parts(lossy_cp, specs, e_parts, scale=-scale)
            g_minus = _backward_grad_parts(model=model, specs=specs, forward_loss=forward_loss, rng_state=rng_state)
            coef = np.float32(float(rms_e) / (2.0 * float(fd_eps)))
            for dst, gp, gm in zip(h_acc, g_plus, g_minus):
                dst += (coef * (gp.astype(np.float32, copy=False) - gm.astype(np.float32, copy=False)) * inv).astype(np.float32, copy=False)
            del g_plus, g_minus
    finally:
        _restore_current_params(lossy_cp)
    return h_acc


def _hvp_quadratic_for_direction(
    *,
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    e_parts: Sequence[np.ndarray],
    model: Any,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    fd_eps: float,
    total_n: int,
) -> float:
    rms_e = _rms_for_numpy_parts(e_parts, total_n=total_n, lossy_cp=lossy_cp)
    if rms_e <= 0.0 or not loss_bank:
        return 0.0
    acc = 0.0
    scale = float(fd_eps) / float(rms_e)
    try:
        for forward_loss, rng_state in loss_bank:
            _restore_current_params(lossy_cp)
            _set_params_plus_direction_parts(lossy_cp, specs, e_parts, scale=scale)
            dot_plus = _backward_dot_direction(model=model, specs=specs, e_parts=e_parts, forward_loss=forward_loss, rng_state=rng_state)
            _restore_current_params(lossy_cp)
            _set_params_plus_direction_parts(lossy_cp, specs, e_parts, scale=-scale)
            dot_minus = _backward_dot_direction(model=model, specs=specs, e_parts=e_parts, forward_loss=forward_loss, rng_state=rng_state)
            local_ehe = float(rms_e) * (float(dot_plus) - float(dot_minus)) / (2.0 * float(fd_eps))
            global_ehe = _all_reduce_sum_float(local_ehe, group=getattr(lossy_cp, "_proposal_cpu_pg", None))
            acc += 0.5 * float(global_ehe)
    finally:
        _restore_current_params(lossy_cp)
    return float(acc) / float(max(1, len(loss_bank)))


def _oneshot_actual_row(
    *,
    iteration: int,
    rank: int,
    world: int,
    policy_name: str,
    first_order_name: str,
    ratio: float,
    total_n_local: int,
    total_n: int,
    loss_base: float,
    loss_rb: float,
    select_info: Dict[str, Any],
    threshold_info: Dict[str, Any],
    fd_eps: float,
    actual_batches: int,
    hvp_batches: int,
    loss_base_batches: Sequence[float],
    loss_after_batches: Sequence[float],
    delta_seed: int,
) -> Dict[str, Any]:
    actual_remaining = float(loss_rb) - float(loss_base)
    delta_stats = _paired_delta_stats(
        base=loss_base_batches,
        policy=loss_after_batches,
        seed=int(delta_seed),
    )
    threshold = threshold_info.get("threshold")
    return {
        "mode": "independent_oneshot_hvp_actual",
        "step": int(iteration),
        "rank": int(rank),
        "world": int(world),
        "policy": str(policy_name),
        "score_first_order_name": str(first_order_name),
        "topk_ratio": float(ratio),
        "selection_scope": str(threshold_info.get("selection_scope", "gpu")),
        "local_elements": int(total_n_local),
        "global_elements": int(total_n),
        "local_selected": int(select_info["local_selected"]),
        "global_selected": int(select_info["global_selected"]),
        "global_selected_ratio": _safe_div(float(select_info["global_selected"]), float(total_n)),
        "threshold": (float(threshold) if threshold is not None and math.isfinite(float(threshold)) else None),
        "tie_prob": float(threshold_info.get("tie_prob", 0.0) or 0.0),
        "sample_elements_for_threshold": int(threshold_info.get("sample_elements", 0) or 0),
        "sample_global_elements_for_threshold": int(threshold_info.get("sample_global_elements", 0) or 0),
        "dkw_epsilon_95": threshold_info.get("dkw_epsilon_95"),
        "dkw_epsilon_99": threshold_info.get("dkw_epsilon_99"),
        "loss_base": float(loss_base),
        "loss_after_selected_save": float(loss_rb),
        "actual_loss_remaining": float(actual_remaining),
        "actual_delta_mean": float(delta_stats["mean"]),
        "actual_delta_median": float(delta_stats["median"]),
        "actual_delta_std": float(delta_stats["std"]),
        "actual_delta_ci95_low": float(delta_stats["ci95_low"]),
        "actual_delta_ci95_high": float(delta_stats["ci95_high"]),
        "actual_delta_positive_ratio": float(delta_stats["positive_ratio"]),
        "actual_delta_min": float(delta_stats["min"]),
        "actual_delta_max": float(delta_stats["max"]),
        "loss_base_batches": delta_stats.get("base_batches"),
        "loss_after_selected_save_batches": delta_stats.get("policy_batches"),
        "actual_delta_batches": delta_stats.get("delta_batches"),
        "selected_hvp_residual_computed": False,
        "selected_hvp_residual_reason": (
            "not computed: selected-set residual needs a fresh HVP of the remaining vector r, "
            "while this one-shot attribution uses H*e at the original rollback vector"
        ),
        "fd_eps": float(fd_eps),
        "actual_batches": int(actual_batches),
        "hvp_batches": int(hvp_batches),
        "hvp_backend": "gpu_threshold",
    }


def _direction_tensors(lossy_cp: Any, specs: Sequence[Dict[str, Any]]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for spec in specs:
        grp = spec["grp"]
        n = int(spec["n"])
        off = _elem_off(spec)
        p = grp["p_flat"].narrow(0, off, n)
        c = _checkpoint_tensor_for_spec(lossy_cp, spec, device=p.device, dtype=p.dtype)
        if c is None:
            out.append(torch.zeros(n, dtype=torch.float32, device=p.device))
        else:
            out.append((p.detach().float() - c.detach().float()).clone())
    return out


def _hvp_tensors_for_direction(
    *,
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    e_tensors: Sequence[torch.Tensor],
    model: Any,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    fd_eps: float,
    total_n: int,
) -> List[torch.Tensor]:
    rms_e = _rms_for_tensor_parts(e_tensors, total_n=total_n, lossy_cp=lossy_cp)
    if rms_e <= 0.0 or not loss_bank:
        return [torch.zeros_like(e, dtype=torch.float32) for e in e_tensors]
    h_acc = [torch.zeros_like(e, dtype=torch.float32) for e in e_tensors]
    scale = float(fd_eps) / float(rms_e)
    coef = float(rms_e) / (2.0 * float(fd_eps) * float(max(1, len(loss_bank))))
    try:
        for forward_loss, rng_state in loss_bank:
            _set_params_plus_direction_tensors(lossy_cp, specs, e_tensors, scale=scale)
            g_plus = _backward_grad_tensors(
                model=model,
                specs=specs,
                forward_loss=forward_loss,
                rng_state=rng_state,
            )
            _set_params_plus_direction_tensors(lossy_cp, specs, e_tensors, scale=-scale)
            g_minus = _backward_grad_tensors(
                model=model,
                specs=specs,
                forward_loss=forward_loss,
                rng_state=rng_state,
            )
            for dst, gp, gm in zip(h_acc, g_plus, g_minus):
                dst.add_(gp.sub(gm), alpha=float(coef))
            del g_plus, g_minus
    finally:
        _restore_current_params(lossy_cp)
    return h_acc


def _backward_grad_tensors(
    *,
    model: Any,
    specs: Sequence[Dict[str, Any]],
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
    rng_state: Optional[Dict[str, Any]],
) -> List[torch.Tensor]:
    if rng_state is not None:
        _restore_rng_state(rng_state)
    module = model[0] if isinstance(model, (list, tuple)) else model
    try:
        for param in module.parameters():
            param.grad = None
    except Exception:
        pass
    grads: List[Optional[torch.Tensor]] = [None] * len(specs)
    handles = []
    for idx, spec in enumerate(specs):
        param = spec["param"]

        def _hook(raw_grad, *, _idx=idx, _spec=spec):
            grads[_idx] = _grad_slice_for_spec(raw_grad, _spec).float().clone()
            return torch.zeros_like(raw_grad)

        handles.append(param.register_hook(_hook))
    try:
        _backward_from_forward_loss(model=model, forward_loss=forward_loss)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        try:
            for param in module.parameters():
                param.grad = None
        except Exception:
            pass
    out: List[torch.Tensor] = []
    for grad, spec in zip(grads, specs):
        if grad is not None:
            out.append(grad)
        else:
            out.append(torch.zeros(int(spec["n"]), dtype=torch.float32, device=spec["param"].device))
    return out


@torch.no_grad()
def _set_params_plus_direction_tensors(
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    e_tensors: Sequence[torch.Tensor],
    *,
    scale: float,
) -> None:
    for spec, e_t in zip(specs, e_tensors):
        grp = spec["grp"]
        n = int(spec["n"])
        off = _elem_off(spec)
        param = spec["param"]
        p = grp["p_flat"].narrow(0, off, n)
        value = p.detach().float() + float(scale) * e_t.detach().float()
        _copy_flat_to_param(
            param,
            value,
            param_offset=int(spec.get("param_offset", 0) or 0),
            n=n,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _quadratic_from_tensors(
    e_tensors: Sequence[torch.Tensor],
    h_e_tensors: Sequence[torch.Tensor],
    *,
    lossy_cp: Any,
) -> float:
    local_tensors = [
        torch.dot(e.detach().float(), h.detach().float())
        for e, h in zip(e_tensors, h_e_tensors)
        if int(e.numel()) > 0 and int(h.numel()) == int(e.numel())
    ]
    local = 0.5 * float(torch.stack(local_tensors).sum().detach().item()) if local_tensors else 0.0
    return _all_reduce_sum_float(local, group=getattr(lossy_cp, "_proposal_cpu_pg", None))


def _rms_for_tensor_parts(
    e_tensors: Sequence[torch.Tensor],
    *,
    total_n: int,
    lossy_cp: Any,
) -> float:
    local_tensors = [
        torch.dot(e.detach().float(), e.detach().float())
        for e in e_tensors
        if int(e.numel()) > 0
    ]
    local = float(torch.stack(local_tensors).sum().detach().item()) if local_tensors else 0.0
    global_sum = _all_reduce_sum_float(local, group=getattr(lossy_cp, "_proposal_cpu_pg", None))
    return math.sqrt(max(0.0, float(global_sum)) / float(max(1, int(total_n))))


def _sample_tensors_by_offsets(
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    tensors: Sequence[torch.Tensor],
    sample: Dict[str, Any],
) -> np.ndarray:
    out = np.zeros(int(sample.get("n", 0)), dtype=np.float32)
    by_group: Dict[int, List[Tuple[int, int, torch.Tensor]]] = {}
    for spec, tensor in zip(specs, tensors):
        by_group.setdefault(int(spec["group_idx"]), []).append(
            (_logical_off(spec), int(spec["n"]), tensor)
        )
    for item in sample.get("items", ()):
        group_idx = int(item["group_idx"])
        local_offsets = np.asarray(item["local_offsets_np"], dtype=np.int64)
        positions = np.asarray(item["positions"], dtype=np.int64)
        for start, n, tensor in by_group.get(group_idx, ()):
            lo = int(np.searchsorted(local_offsets, start, side="left"))
            hi = int(np.searchsorted(local_offsets, start + n, side="left"))
            if hi <= lo:
                continue
            offs_np = local_offsets[lo:hi] - np.int64(start)
            device = tensor.device
            offs = torch.as_tensor(offs_np, dtype=torch.long, device=device)
            vals = torch.index_select(tensor.detach().float(), 0, offs)
            out[positions[lo:hi]] = vals.detach().cpu().numpy().astype(np.float32, copy=False)
    return out


def _threshold_from_sample_scores(
    score_sample: np.ndarray,
    ratio: float,
    *,
    lossy_cp: Any,
) -> Dict[str, Any]:
    ratio = max(0.0, min(1.0, float(ratio)))
    local_sample = np.asarray(score_sample, dtype=np.float32).reshape(-1).copy()
    np.nan_to_num(local_sample, nan=-np.inf, posinf=np.inf, neginf=-np.inf, copy=False)
    group = getattr(lossy_cp, "_proposal_cpu_pg", None)
    gathered = _all_gather_float_arrays(local_sample, group=group)
    union = np.concatenate(gathered) if len(gathered) > 1 else gathered[0]
    sample_n = int(union.size)
    if sample_n <= 0 or ratio <= 0.0:
        threshold = float("inf")
        tie_prob = 0.0
    elif ratio >= 1.0:
        threshold = float(np.min(union))
        tie_prob = 1.0
    else:
        k = max(1, int(math.ceil(ratio * float(sample_n))))
        kth = max(0, sample_n - k)
        union.partition(kth)
        threshold = float(union[kth])
        greater = int(np.count_nonzero(union > np.float32(threshold)))
        equal = int(np.count_nonzero(union == np.float32(threshold)))
        need = max(0, min(equal, int(k - greater)))
        tie_prob = float(need) / float(equal) if equal > 0 else 0.0
    out = {
        "threshold": float(threshold),
        "tie_prob": float(max(0.0, min(1.0, tie_prob))),
        "sample_elements": int(local_sample.size),
        "sample_global_elements": int(sample_n),
        "dkw_epsilon_95": _dkw_epsilon(sample_n, alpha=0.05),
        "dkw_epsilon_99": _dkw_epsilon(sample_n, alpha=0.01),
        "selection_scope": "gpu_sample_cdf_threshold_full_apply",
    }
    del local_sample, gathered, union
    return out


@torch.no_grad()
def _eval_gpu_policy_loss(
    *,
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    h_e_tensors: Sequence[torch.Tensor],
    threshold: float,
    tie_prob: float,
    policy_kind: str,
    iteration: int,
    ratio: float,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
) -> Tuple[float, List[float], Dict[str, int]]:
    local_selected = 0
    try:
        for spec, h_e in zip(specs, h_e_tensors):
            selected = _gpu_policy_selected_mask(
                lossy_cp=lossy_cp,
                spec=spec,
                h_e=h_e,
                threshold=float(threshold),
                tie_prob=float(tie_prob),
                policy_kind=str(policy_kind),
                iteration=int(iteration),
                ratio=float(ratio),
            )
            local_selected += int(selected.sum().detach().item())
            grp = spec["grp"]
            n = int(spec["n"])
            off = _elem_off(spec)
            param = spec["param"]
            p = grp["p_flat"].narrow(0, off, n)
            c = _checkpoint_tensor_for_spec(lossy_cp, spec, device=p.device, dtype=p.dtype)
            if c is None:
                continue
            value = torch.where(selected, p.detach(), c.detach())
            _copy_flat_to_param(
                param,
                value,
                param_offset=int(spec.get("param_offset", 0) or 0),
                n=n,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        loss, loss_batches = _eval_loss_bank_values(
            loss_bank,
            group=getattr(lossy_cp, "_proposal_cpu_pg", None),
        )
    finally:
        _restore_current_params(lossy_cp)
    group = getattr(lossy_cp, "_proposal_cpu_pg", None)
    global_selected = int(_all_reduce_sum_int(int(local_selected), group=group))
    return float(loss), [float(x) for x in loss_batches], {
        "local_selected": int(local_selected),
        "global_selected": int(global_selected),
    }


def _gpu_policy_selected_mask(
    *,
    lossy_cp: Any,
    spec: Dict[str, Any],
    h_e: torch.Tensor,
    threshold: float,
    tie_prob: float,
    policy_kind: str,
    iteration: int,
    ratio: float,
) -> torch.Tensor:
    n = int(spec["n"])
    if n <= 0:
        return torch.zeros(0, dtype=torch.bool, device=spec["param"].device)
    if policy_kind == "random_hash":
        global_ids = _global_ids_for_spec(lossy_cp, spec)
        return _hash_uniform_tensor(global_ids, seed=0xABC000 + int(iteration)) < float(ratio)
    if policy_kind == "round_robin":
        global_ids = _global_ids_for_spec(lossy_cp, spec)
        period = max(1, int(round(1.0 / max(1.0e-12, float(ratio)))))
        return torch.remainder(global_ids + int(iteration), int(period)) == 0

    grp = spec["grp"]
    off = _elem_off(spec)
    p = grp["p_flat"].narrow(0, off, n)
    c = _checkpoint_tensor_for_spec(lossy_cp, spec, device=p.device, dtype=p.dtype)
    e = torch.zeros(n, dtype=torch.float32, device=p.device) if c is None else (p.detach().float() - c.detach().float())
    lin_t = grp.get(_PRODUCTION_FIRST_ORDER_KEY)
    linear = (
        lin_t.narrow(0, off, n).detach().float()
        if isinstance(lin_t, torch.Tensor) else torch.zeros(n, dtype=torch.float32, device=p.device)
    )
    h_abs = _diag_abs_h_tensor(lossy_cp, spec)
    force = None
    if policy_kind == "diag_true_e":
        score = torch.clamp(-linear * e + 0.5 * h_abs * e * e, min=0.0)
    elif policy_kind == "hvp_directional":
        h_e_dev = h_e.detach().to(device=p.device, dtype=torch.float32)
        score = torch.clamp(-linear * e + 0.5 * e * h_e_dev, min=0.0)
    elif policy_kind == "hvp_absdiag_marginal":
        h_e_dev = h_e.detach().to(device=p.device, dtype=torch.float32)
        score = torch.clamp(
            -linear * e + e * h_e_dev - 0.5 * h_abs * e * e,
            min=0.0,
        )
    elif policy_kind == "agebit_force":
        score, force = _agebit_score_tensor(
            lossy_cp=lossy_cp,
            spec=spec,
            iteration=int(iteration),
        )
    else:
        raise ValueError(f"unknown GPU oracle policy kind: {policy_kind}")

    score = torch.nan_to_num(score, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf"))
    if not math.isfinite(float(threshold)):
        selected = torch.zeros(n, dtype=torch.bool, device=score.device)
    else:
        th = torch.tensor(float(threshold), dtype=score.dtype, device=score.device)
        selected = score > th
        if float(tie_prob) > 0.0:
            global_ids = _global_ids_for_spec(lossy_cp, spec)
            tie = score == th
            selected = selected | (tie & (_hash_uniform_tensor(global_ids, seed=0x71E000 + int(iteration)) < float(tie_prob)))
    if force is not None:
        selected = selected | force
    return selected


def _diag_abs_h_tensor(lossy_cp: Any, spec: Dict[str, Any]) -> torch.Tensor:
    grp = spec["grp"]
    n = int(spec["n"])
    off = _elem_off(spec)
    v = grp["v_flat"].narrow(0, off, n).detach().float()
    h_lut = getattr(lossy_cp, "_h_lookup_lut", None)
    h_values = getattr(h_lut, "values", None)
    log_v_min = getattr(h_lut, "log_v_min", None)
    inv_width = getattr(h_lut, "inv_log_v_width", None)
    if not (
        isinstance(h_values, torch.Tensor)
        and isinstance(log_v_min, torch.Tensor)
        and isinstance(inv_width, torch.Tensor)
        and int(h_values.numel()) > 1
    ):
        return torch.zeros_like(v, dtype=torch.float32)
    bins = int(h_values.numel())
    pos = (torch.log(torch.clamp(v, min=1.0e-30)) - log_v_min.detach().float()) * inv_width.detach().float()
    pos = torch.clamp(pos, min=0.0, max=float(bins - 1))
    lo = torch.floor(pos).to(torch.long)
    hi = torch.clamp(lo + 1, max=bins - 1)
    frac = pos - lo.float()
    vals = h_values.detach().float()
    return vals[lo] + frac * (vals[hi] - vals[lo])


def _agebit_score_tensor(
    *,
    lossy_cp: Any,
    spec: Dict[str, Any],
    iteration: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grp = spec["grp"]
    n = int(spec["n"])
    off = _elem_off(spec)
    device = grp["p_flat"].device
    m = grp["m_flat"].narrow(0, off, n).detach().float()
    v = grp["v_flat"].narrow(0, off, n).detach().float()
    lin_t = grp.get(_PRODUCTION_FIRST_ORDER_KEY)
    linear = (
        lin_t.narrow(0, off, n).detach().float()
        if isinstance(lin_t, torch.Tensor) else torch.zeros(n, dtype=torch.float32, device=device)
    )
    h_abs = _diag_abs_h_tensor(lossy_cp, spec)
    age_code = _age_code_tensor(lossy_cp, spec)
    geom = _geom_tensor(lossy_cp, spec, age_code=age_code)
    lr_over_bc1 = float(getattr(lossy_cp, "_adam_lr_over_bc1_for_selection", 1.0e-4))
    sqrt_bc2_inv = float(getattr(lossy_cp, "_adam_sqrt_bc2_inv_for_selection", 1.0))
    eps = float(getattr(lossy_cp, "_adam_eps_for_selection", 1.0e-8))
    denom = torch.sqrt(torch.clamp(v, min=0.0)) * float(sqrt_bc2_inv) + float(eps)
    e_hat = (-float(lr_over_bc1) * m / denom) * geom
    score = torch.clamp(-linear * e_hat + 0.5 * h_abs * e_hat * e_hat, min=0.0)
    global_ids = _global_ids_for_spec(lossy_cp, spec)
    force = (age_code == 3) & _age2_phase_hit_tensor(
        global_ids,
        step=int(iteration),
        period=128,
        salt=0x51ED270B,
    )
    return score, force


def _geom_tensor(lossy_cp: Any, spec: Dict[str, Any], *, age_code: torch.Tensor) -> torch.Tensor:
    grp = spec["grp"]
    n = int(spec["n"])
    off = _elem_off(spec)
    v = grp["v_flat"].narrow(0, off, n).detach().float()
    h_lut = getattr(lossy_cp, "_h_lookup_lut", None)
    h_values = getattr(h_lut, "values", None)
    log_v_min = getattr(h_lut, "log_v_min", None)
    inv_width = getattr(h_lut, "inv_log_v_width", None)
    geom_lut = getattr(getattr(lossy_cp, "_age2bit_score_state", None), "ehat_geom_lut", None)
    if not (
        isinstance(h_values, torch.Tensor)
        and isinstance(log_v_min, torch.Tensor)
        and isinstance(inv_width, torch.Tensor)
        and isinstance(geom_lut, torch.Tensor)
        and int(h_values.numel()) > 1
    ):
        return torch.ones(n, dtype=torch.float32, device=v.device)
    bins = int(h_values.numel())
    pos = (torch.log(torch.clamp(v, min=1.0e-30)) - log_v_min.detach().float()) * inv_width.detach().float()
    pos = torch.clamp(pos, min=0.0, max=float(bins - 1))
    lo = torch.floor(pos).to(torch.long)
    hi = torch.clamp(lo + 1, max=bins - 1)
    frac = pos - lo.float()
    flat = geom_lut.detach().float().reshape(-1)
    idx_lo = torch.clamp(age_code.to(torch.long) * bins + lo, max=int(flat.numel()) - 1)
    idx_hi = torch.clamp(age_code.to(torch.long) * bins + hi, max=int(flat.numel()) - 1)
    return flat[idx_lo] + frac * (flat[idx_hi] - flat[idx_lo])


def _age_code_tensor(lossy_cp: Any, spec: Dict[str, Any]) -> torch.Tensor:
    age_state = getattr(lossy_cp, "_age2bit_score_state", None)
    age_words = getattr(age_state, "age_words", None)
    global_ids = _global_ids_for_spec(lossy_cp, spec)
    if not isinstance(age_words, torch.Tensor) or int(age_words.numel()) <= 0:
        return torch.zeros_like(global_ids, dtype=torch.long)
    word_idx = torch.div(global_ids, 16, rounding_mode="floor")
    valid = word_idx < int(age_words.numel())
    safe_word_idx = torch.where(valid, word_idx, torch.zeros_like(word_idx))
    words = torch.index_select(age_words.detach().to(torch.int64), 0, safe_word_idx)
    shift = ((global_ids & 15) << 1).to(torch.int64)
    code = (words >> shift) & 3
    return torch.where(valid, code, torch.zeros_like(code)).to(torch.long)


def _global_ids_for_spec(lossy_cp: Any, spec: Dict[str, Any]) -> torch.Tensor:
    grp = spec["grp"]
    n = int(spec["n"])
    off = _age_off(spec)
    device = grp["p_flat"].device
    return torch.arange(off, off + n, dtype=torch.long, device=device)


def _hash_uniform_tensor(global_ids: torch.Tensor, *, seed: int) -> torch.Tensor:
    mod = 1000003
    x = global_ids.to(torch.long) * 1103515245 + int(seed) * 12345 + 12345
    return torch.remainder(x, mod).to(torch.float32) / float(mod)


def _age2_phase_hit_tensor(
    global_ids: torch.Tensor,
    *,
    step: int,
    period: int,
    salt: int,
) -> torch.Tensor:
    period = max(1, int(period))
    mask32 = (1 << 32) - 1
    x = torch.bitwise_xor(global_ids.to(torch.long), int(salt) & mask32)
    x = torch.bitwise_and((torch.bitwise_xor(x, x >> 16) * 0x7FEB352D), mask32)
    x = torch.bitwise_and((torch.bitwise_xor(x, x >> 15) * 0x846CA68B), mask32)
    x = torch.bitwise_xor(x, x >> 16)
    phase = torch.remainder(x, period)
    return torch.remainder(phase + int(step), period) == 0


@torch.no_grad()
def _set_params_plus_direction_parts(
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    e_parts: Sequence[np.ndarray],
    *,
    scale: float,
) -> None:
    for spec, e_np in zip(specs, e_parts):
        grp = spec["grp"]
        n = int(spec["n"])
        off = _elem_off(spec)
        param = spec["param"]
        p = grp["p_flat"].narrow(0, off, n)
        e_t = torch.as_tensor(e_np, device=p.device, dtype=p.dtype)
        value = p + float(scale) * e_t
        _copy_flat_to_param(
            param,
            value,
            param_offset=int(spec.get("param_offset", 0) or 0),
            n=n,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _backward_grad_parts(
    *,
    model: Any,
    specs: Sequence[Dict[str, Any]],
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
    rng_state: Optional[Dict[str, Any]],
) -> List[np.ndarray]:
    if rng_state is not None:
        _restore_rng_state(rng_state)
    module = model[0] if isinstance(model, (list, tuple)) else model
    try:
        for param in module.parameters():
            param.grad = None
    except Exception:
        pass
    grads: List[Optional[np.ndarray]] = [None] * len(specs)
    handles = []
    for idx, spec in enumerate(specs):
        param = spec["param"]

        def _hook(raw_grad, *, _idx=idx, _spec=spec):
            grads[_idx] = _grad_slice_for_spec(raw_grad, _spec).float().cpu().numpy().astype(np.float32, copy=True)
            return torch.zeros_like(raw_grad)

        handles.append(param.register_hook(_hook))
    try:
        _backward_from_forward_loss(model=model, forward_loss=forward_loss)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        try:
            for param in module.parameters():
                param.grad = None
        except Exception:
            pass
    return [
        g if g is not None else np.zeros(int(spec["n"]), dtype=np.float32)
        for g, spec in zip(grads, specs)
    ]


def _backward_dot_direction(
    *,
    model: Any,
    specs: Sequence[Dict[str, Any]],
    e_parts: Sequence[np.ndarray],
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
    rng_state: Optional[Dict[str, Any]],
) -> float:
    if rng_state is not None:
        _restore_rng_state(rng_state)
    module = model[0] if isinstance(model, (list, tuple)) else model
    try:
        for param in module.parameters():
            param.grad = None
    except Exception:
        pass
    dot_tensors: List[torch.Tensor] = []
    handles = []
    for spec, e_np in zip(specs, e_parts):
        param = spec["param"]
        e_t = torch.as_tensor(e_np, device=param.device, dtype=torch.float32)

        def _hook(raw_grad, *, _e=e_t, _spec=spec):
            flat = _grad_slice_for_spec(raw_grad, _spec).float()
            dot_tensors.append(torch.sum(flat * _e))
            return torch.zeros_like(raw_grad)

        handles.append(param.register_hook(_hook))
    try:
        _backward_from_forward_loss(model=model, forward_loss=forward_loss)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return float(torch.stack(dot_tensors).sum().detach().item()) if dot_tensors else 0.0
    finally:
        for handle in handles:
            handle.remove()
        try:
            for param in module.parameters():
                param.grad = None
        except Exception:
            pass


def _sample_parts_by_offsets(
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    parts: Sequence[np.ndarray],
    sample: Dict[str, Any],
) -> np.ndarray:
    out = np.zeros(int(sample.get("n", 0)), dtype=np.float32)
    by_group: Dict[int, List[Tuple[int, int, np.ndarray]]] = {}
    for spec, part in zip(specs, parts):
        group_idx = int(spec["group_idx"])
        by_group.setdefault(group_idx, []).append((_logical_off(spec), int(spec["n"]), part))
    for item in sample.get("items", ()):
        group_idx = int(item["group_idx"])
        local_offsets = np.asarray(item["local_offsets_np"], dtype=np.int64)
        positions = np.asarray(item["positions"], dtype=np.int64)
        for start, n, part in by_group.get(group_idx, ()):
            lo = int(np.searchsorted(local_offsets, start, side="left"))
            hi = int(np.searchsorted(local_offsets, start + n, side="left"))
            if hi > lo:
                offs = local_offsets[lo:hi] - np.int64(start)
                out[positions[lo:hi]] = part[offs].astype(np.float32, copy=False)
    return out


def _select_masks_global_exact(
    score_parts: Sequence[np.ndarray],
    ratio: float,
    *,
    lossy_cp: Any,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    ratio = max(0.0, min(1.0, float(ratio)))
    local_n = int(sum(int(score.size) for score in score_parts))
    group = getattr(lossy_cp, "_proposal_cpu_pg", None)
    global_n = int(_all_reduce_sum_int(local_n, group=group))
    k_global = int(math.ceil(ratio * float(max(1, global_n))))
    if k_global <= 0 or local_n <= 0 or global_n <= 0:
        return [np.zeros(score.shape, dtype=bool) for score in score_parts], {
            "threshold": None,
            "local_selected": 0,
            "global_selected": 0,
            "selection_scope": "global_exact_partition",
        }
    if k_global >= global_n:
        masks = [np.ones(score.shape, dtype=bool) for score in score_parts]
        local_selected = int(sum(int(mask.sum()) for mask in masks))
        global_selected = int(_all_reduce_sum_int(local_selected, group=group))
        return masks, {
            "threshold": None,
            "local_selected": int(local_selected),
            "global_selected": int(global_selected),
            "selection_scope": "global_exact_partition",
        }
    views = [np.asarray(score, dtype=np.float32).reshape(-1) for score in score_parts]
    local_candidate_k = min(int(k_global), int(local_n))
    flat = np.concatenate(views) if len(views) > 1 else views[0].copy()
    np.nan_to_num(flat, nan=-np.inf, posinf=np.inf, neginf=-np.inf, copy=False)
    kth_index = int(local_n - local_candidate_k)
    flat.partition(kth_index)
    local_candidates = flat[kth_index:].copy()
    del flat
    gathered = _all_gather_float_arrays(local_candidates, group=group)
    del local_candidates
    union = np.concatenate(gathered) if len(gathered) > 1 else gathered[0]
    if int(union.size) < k_global:
        threshold = np.float32(np.min(union)) if union.size else np.float32(-np.inf)
    else:
        kth_union = int(union.size - k_global)
        union.partition(kth_union)
        threshold = np.float32(union[kth_union])
    del union, gathered
    greater_count = int(sum(int(np.count_nonzero(v > threshold)) for v in views))
    equal_count = int(sum(int(np.count_nonzero(v == threshold)) for v in views))
    global_greater = int(_all_reduce_sum_int(greater_count, group=group))
    global_equal = int(_all_reduce_sum_int(equal_count, group=group))
    ties_needed = max(0, min(global_equal, int(k_global - global_greater)))
    local_tie_quota = _tie_quota_for_rank(equal_count, ties_needed, group=group)
    masks: List[np.ndarray] = []
    for score, view in zip(score_parts, views):
        mask = view > threshold
        if local_tie_quota > 0:
            tie_idx = np.flatnonzero(view == threshold)
            take = min(int(local_tie_quota), int(tie_idx.size))
            if take > 0:
                mask = mask.copy()
                mask[tie_idx[:take]] = True
                local_tie_quota -= take
        masks.append(mask.reshape(score.shape))
    local_selected = int(sum(int(mask.sum()) for mask in masks))
    global_selected = int(_all_reduce_sum_int(local_selected, group=group))
    return masks, {
        "threshold": float(threshold),
        "local_selected": int(local_selected),
        "global_selected": int(global_selected),
        "selection_scope": "global_exact_partition",
    }


def _select_masks_sample_cdf_threshold(
    score_parts: Sequence[np.ndarray],
    score_sample: np.ndarray,
    ratio: float,
    *,
    lossy_cp: Any,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Estimate the global top-K threshold from a uniform element sample.

    This is not an exact top-K selector.  It is a CDF/quantile estimator used
    only by expensive telemetry one-shot actual loss checks, and it records the
    DKW bound plus the full-model selected ratio so the approximation is
    auditable.
    """

    ratio = max(0.0, min(1.0, float(ratio)))
    local_n = int(sum(int(score.size) for score in score_parts))
    group = getattr(lossy_cp, "_proposal_cpu_pg", None)
    global_n = int(_all_reduce_sum_int(local_n, group=group))
    local_sample = np.asarray(score_sample, dtype=np.float32).reshape(-1).copy()
    np.nan_to_num(
        local_sample,
        nan=-np.inf,
        posinf=np.inf,
        neginf=-np.inf,
        copy=False,
    )
    gathered = _all_gather_float_arrays(local_sample, group=group)
    union = np.concatenate(gathered) if len(gathered) > 1 else gathered[0]
    sample_n = int(union.size)
    if ratio <= 0.0 or local_n <= 0 or global_n <= 0 or sample_n <= 0:
        masks = [np.zeros(score.shape, dtype=bool) for score in score_parts]
        threshold = float("inf")
    elif ratio >= 1.0:
        masks = [np.ones(score.shape, dtype=bool) for score in score_parts]
        threshold = float(np.min(union))
    else:
        k_sample = max(1, int(math.ceil(ratio * float(sample_n))))
        kth = max(0, int(sample_n) - k_sample)
        union.partition(kth)
        threshold = float(union[kth])
        views = [
            np.nan_to_num(
                np.asarray(score, dtype=np.float32).reshape(-1),
                nan=-np.inf,
                posinf=np.inf,
                neginf=-np.inf,
            )
            for score in score_parts
        ]
        greater_count = int(sum(int(np.count_nonzero(v > threshold)) for v in views))
        equal_count = int(sum(int(np.count_nonzero(v == threshold)) for v in views))
        k_global = int(math.ceil(ratio * float(max(1, global_n))))
        global_greater = int(_all_reduce_sum_int(greater_count, group=group))
        global_equal = int(_all_reduce_sum_int(equal_count, group=group))
        ties_needed = max(0, min(global_equal, int(k_global - global_greater)))
        local_tie_quota = _tie_quota_for_rank(equal_count, ties_needed, group=group)
        masks = []
        for score, view in zip(score_parts, views):
            mask = view > threshold
            if local_tie_quota > 0:
                tie_idx = _first_equal_indices(
                    view,
                    np.float32(threshold),
                    int(local_tie_quota),
                )
                take = int(tie_idx.size)
                if take > 0:
                    mask = mask.copy()
                    mask[tie_idx] = True
                    local_tie_quota -= take
            masks.append(mask.reshape(score.shape))
    local_selected = int(sum(int(mask.sum()) for mask in masks))
    global_selected = int(_all_reduce_sum_int(local_selected, group=group))
    del gathered, union, local_sample
    return masks, {
        "threshold": float(threshold),
        "local_selected": int(local_selected),
        "global_selected": int(global_selected),
        "global_selected_ratio": _safe_div(float(global_selected), float(global_n)),
        "selection_scope": "sample_cdf_threshold_full_apply",
        "sample_elements": int(np.asarray(score_sample).size),
        "sample_global_elements": int(sample_n),
        "dkw_epsilon_95": _dkw_epsilon(sample_n, alpha=0.05),
        "dkw_epsilon_99": _dkw_epsilon(sample_n, alpha=0.01),
    }


def _all_gather_float_arrays(arr: np.ndarray, *, group: Any) -> List[np.ndarray]:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return [arr]
    world = int(torch.distributed.get_world_size(group=group))
    local_n = int(arr.size)
    lengths = [torch.zeros(1, dtype=torch.int64, device="cpu") for _ in range(world)]
    local_len_t = torch.tensor([local_n], dtype=torch.int64, device="cpu")
    torch.distributed.all_gather(lengths, local_len_t, group=group)
    max_n = max(int(x.item()) for x in lengths)
    padded = np.full(max_n, -np.inf, dtype=np.float32)
    if local_n > 0:
        padded[:local_n] = arr
    send = torch.from_numpy(padded)
    recv = [torch.empty_like(send) for _ in range(world)]
    torch.distributed.all_gather(recv, send, group=group)
    out: List[np.ndarray] = []
    for tensor, length in zip(recv, lengths):
        n = int(length.item())
        out.append(tensor.numpy()[:n].copy())
    return out


def _tie_quota_for_rank(local_equal: int, ties_needed: int, *, group: Any) -> int:
    if ties_needed <= 0 or local_equal <= 0:
        return 0
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return min(int(local_equal), int(ties_needed))
    rank = int(torch.distributed.get_rank(group=group))
    world = int(torch.distributed.get_world_size(group=group))
    counts = [torch.zeros(1, dtype=torch.int64, device="cpu") for _ in range(world)]
    local = torch.tensor([int(local_equal)], dtype=torch.int64, device="cpu")
    torch.distributed.all_gather(counts, local, group=group)
    before = sum(int(counts[i].item()) for i in range(rank))
    remaining = int(ties_needed) - int(before)
    return max(0, min(int(local_equal), int(remaining)))


def _first_equal_indices(values: np.ndarray, threshold: np.float32, limit: int) -> np.ndarray:
    limit = max(0, int(limit))
    if limit <= 0:
        return np.zeros(0, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    chunks: List[np.ndarray] = []
    taken = 0
    chunk_n = max(1, int(os.environ.get(
        "PROPOSAL_INDEPENDENT_ORACLE_TIE_SCAN_CHUNK",
        "4194304",
    ) or 4194304))
    for start in range(0, int(values.size), chunk_n):
        if taken >= limit:
            break
        view = values[start:start + chunk_n]
        idx = np.flatnonzero(view == threshold)
        if idx.size <= 0:
            continue
        need = min(int(idx.size), limit - taken)
        chunks.append((idx[:need].astype(np.int64, copy=False) + np.int64(start)))
        taken += need
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


def _dkw_epsilon(n: int, *, alpha: float) -> float:
    n = int(n)
    if n <= 0:
        return float("inf")
    alpha = max(1.0e-12, min(1.0, float(alpha)))
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * float(n)))


def _remaining_parts(e_parts: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for e, mask in zip(e_parts, masks):
        rem = e.copy()
        rem[np.asarray(mask, dtype=bool)] = np.float32(0.0)
        out.append(rem)
    return out


@torch.no_grad()
def _eval_selected_masks_loss(
    *,
    lossy_cp: Any,
    specs: Sequence[Dict[str, Any]],
    selected_masks: Sequence[np.ndarray],
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
) -> float:
    try:
        for spec, mask_np in zip(specs, selected_masks):
            grp = spec["grp"]
            n = int(spec["n"])
            off = _elem_off(spec)
            param = spec["param"]
            p = grp["p_flat"].narrow(0, off, n)
            c = _checkpoint_tensor_for_spec(lossy_cp, spec, device=p.device, dtype=p.dtype)
            if c is None:
                continue
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=p.device)
            value = torch.where(mask, p, c)
            _copy_flat_to_param(
                param,
                value,
                param_offset=int(spec.get("param_offset", 0) or 0),
                n=n,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return float(
            _eval_loss_bank(
                loss_bank,
                group=getattr(lossy_cp, "_proposal_cpu_pg", None),
            )
        )
    finally:
        _restore_current_params(lossy_cp)


def _linear_sum_parts(
    linear_parts: Sequence[np.ndarray],
    e_parts: Sequence[np.ndarray],
    *,
    lossy_cp: Any,
) -> float:
    local = 0.0
    for lin, e in zip(linear_parts, e_parts):
        local += -float(np.dot(lin.astype(np.float32, copy=False), e.astype(np.float32, copy=False)))
    return _all_reduce_sum_float(local, group=getattr(lossy_cp, "_proposal_cpu_pg", None))


def _quadratic_from_parts(
    e_parts: Sequence[np.ndarray],
    h_e_parts: Sequence[np.ndarray],
    *,
    lossy_cp: Any,
) -> float:
    local = 0.5 * sum(float(np.dot(e.astype(np.float32, copy=False), h.astype(np.float32, copy=False))) for e, h in zip(e_parts, h_e_parts))
    return _all_reduce_sum_float(local, group=getattr(lossy_cp, "_proposal_cpu_pg", None))


def _rms_for_numpy_parts(
    e_parts: Sequence[np.ndarray],
    *,
    total_n: int,
    lossy_cp: Any,
) -> float:
    local = sum(float(np.dot(e.astype(np.float32, copy=False), e.astype(np.float32, copy=False))) for e in e_parts)
    global_sum = _all_reduce_sum_float(local, group=getattr(lossy_cp, "_proposal_cpu_pg", None))
    return math.sqrt(max(0.0, float(global_sum)) / float(max(1, int(total_n))))


def _safe_div(num: float, den: float) -> float:
    if abs(float(den)) <= 1.0e-30:
        return 0.0
    return float(num) / float(den)


def _backward_dot_and_sample(
    *,
    lossy_cp: Any,
    model: Any,
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
    rng_state: Optional[Dict[str, Any]],
    sample: Dict[str, Any],
    compute_full_dot: bool = True,
) -> Tuple[float, np.ndarray]:
    if rng_state is not None:
        _restore_rng_state(rng_state)
    module = model[0] if isinstance(model, (list, tuple)) else model
    try:
        for param in module.parameters():
            param.grad = None
    except Exception:
        pass
    sample_by_name = _sample_by_name(lossy_cp, sample)
    dot_tensors: List[torch.Tensor] = []
    sample_out = np.zeros(int(sample.get("n", 0)), dtype=np.float32)
    handles = []
    for group_idx, grp in enumerate(getattr(lossy_cp, "_groups", ()) or ()):
        p_flat = grp.get("p_flat")
        if p_flat is None:
            continue
        for spec in _group_entry_specs(lossy_cp, group_idx, grp):
            n = int(spec["n"])
            param, param_offset, param_n = _param_ref_for_name(lossy_cp, str(spec["name"]))
            n = min(n, int(param_n))
            if param is None or n <= 0:
                continue
            off = _elem_off(spec)
            e_slice = None
            if compute_full_dot:
                p_slice = p_flat.narrow(0, off, n)
                c_slice = _checkpoint_tensor_for_spec(
                    lossy_cp,
                    spec,
                    device=p_slice.device,
                    dtype=p_slice.dtype,
                )
                if c_slice is None:
                    continue
                e_slice = (p_slice - c_slice).detach()
            sample_info = sample_by_name.get(str(spec["name"]))

            def _hook(raw_grad, *, _e=e_slice, _sample=sample_info, _param_offset=param_offset, _n=n):
                flat = None
                if _e is not None or _sample is not None:
                    flat = (
                        raw_grad.detach()
                        .reshape(-1)
                        .narrow(0, int(_param_offset), int(_n))
                        .float()
                    )
                if _e is not None and flat is not None:
                    dot_tensors.append(torch.sum(flat * _e.float()))
                if _sample is not None:
                    assert flat is not None
                    vals = torch.index_select(flat, 0, _sample["offsets_t"]).detach().cpu().numpy()
                    sample_out[_sample["positions"]] = vals.astype(np.float32, copy=False)
                return torch.zeros_like(raw_grad)

            handles.append(param.register_hook(_hook))
    try:
        _backward_from_forward_loss(model=model, forward_loss=forward_loss)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        local_dot = float(torch.stack(dot_tensors).sum().detach().item()) if dot_tensors else 0.0
    finally:
        for handle in handles:
            handle.remove()
        try:
            for param in module.parameters():
                param.grad = None
        except Exception:
            pass
    return local_dot, sample_out


def _sample_by_name(lossy_cp: Any, sample: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in sample.get("items", ()):
        grp = lossy_cp._groups[int(item["group_idx"])]
        name = str(item.get("name", ""))
        positions = np.asarray(item["positions"], dtype=np.int64)
        if not name:
            continue
        offsets_np = np.asarray(item.get("entry_offsets_np", ()), dtype=np.int64)
        if offsets_np.size <= 0:
            continue
        offsets_t = torch.as_tensor(offsets_np, dtype=torch.long, device=grp["p_flat"].device)
        if name not in out:
            out[name] = {
                "positions": positions,
                "offsets_t": offsets_t,
            }
        else:
            prev = out[name]
            prev["positions"] = np.concatenate([
                np.asarray(prev["positions"], dtype=np.int64),
                positions,
            ])
            prev["offsets_t"] = torch.cat([prev["offsets_t"], offsets_t], dim=0)
    return out


def _rank_metric_rows(
    *,
    sample_payload: Dict[str, np.ndarray],
    hvp_sample: Optional[np.ndarray],
    proposal_config: Any,
    iteration: int,
    rank: int,
    world: int,
    score_source: str,
) -> List[Dict[str, Any]]:
    e = sample_payload["e"]
    m = sample_payload["m"]
    force = sample_payload["force"]
    score_agebit = sample_payload["score_agebit"]
    score_m_diag = sample_payload["score_m_diag"]
    if hvp_sample is None:
        return []
    score_source = str(score_source)
    prefix = "m_t"
    score_age = score_agebit
    score_true = score_m_diag
    score_magnitude = np.abs(e)
    target_directional_name = "m_t_hvp_directional"
    linear = m
    target_directional = np.maximum(
        -linear * e + 0.5 * e * hvp_sample,
        0.0,
    ).astype(np.float32, copy=False)
    score_age_force = _score_with_force(score_age, force)
    ratios = (float(getattr(proposal_config, "topk_ratio", 0.0) or 0.0),)
    rows: List[Dict[str, Any]] = []

    comparisons = (
        (f"{prefix}+diag_abs_h+true_e", f"{prefix}_true_e_diag_abs_h", score_true, target_directional_name, target_directional, None),
        ("abs_true_checkpoint_error", "abs_true_checkpoint_error", score_magnitude, target_directional_name, target_directional, None),
        (f"{prefix}+diag_abs_h+agebit", f"agebit_{prefix}_diag_abs_h", score_age, target_directional_name, target_directional, None),
        (f"{prefix}+diag_abs_h+agebit+force", f"agebit_{prefix}_diag_abs_h_force", score_age_force, target_directional_name, target_directional, force),
        (f"{prefix}+hvp_directional+true_e", target_directional_name, target_directional, target_directional_name, target_directional, None),
        (f"{prefix}+diag_abs_h+agebit", f"agebit_{prefix}_diag_abs_h", score_age, f"{prefix}_true_e_diag_abs_h", score_true, None),
    )
    for policy_name, score_name, score, target_label, target_values, force_mask in comparisons:
        row = {
            "mode": "independent_oracle_rank",
            "step": int(iteration),
            "rank": int(rank),
            "world": int(world),
            "policy": policy_name,
            "score_first_order_source": (
                "none" if score_name == "abs_true_checkpoint_error" else score_source
            ),
            "topk_ratio": float(getattr(proposal_config, "topk_ratio", 0.0)),
            "score": score_name,
            "target": target_label,
            "sample_elements": int(score.size),
            "spearman": _spearman(score, target_values),
            "pearson": _pearson(score, target_values),
        }
        for ratio in ratios:
            if ratio <= 0.0:
                continue
            metrics = _top_heavy_metrics(
                score,
                target_values,
                ratio,
                force_mask=force_mask,
            )
            key = f"@{ratio:.2f}"
            for mkey, val in metrics.items():
                row[f"{mkey}{key}"] = val
        rows.append(row)

    return rows


def _ehat_diagnostic_from_rank_inputs(
    *,
    sample_payload: Dict[str, np.ndarray],
    hvp_sample: Optional[np.ndarray],
    proposal_config: Any,
    iteration: int,
    rank: int,
    world: int,
    score_source: str,
) -> Optional[Dict[str, Any]]:
    """Build the non-ranking e-hat diagnostic for the full oracle only."""

    if hvp_sample is None:
        return None
    e = sample_payload["e"]
    target_name = "m_t_hvp_directional"
    target = np.maximum(
        -sample_payload["m"] * e + 0.5 * e * hvp_sample,
        0.0,
    ).astype(np.float32, copy=False)
    return _ehat_diagnostic_row(
        e=e,
        e_age=sample_payload["e_age"],
        age_code=sample_payload["age_code"],
        force=sample_payload["force"],
        score=sample_payload["score_agebit"],
        target=target,
        target_name=target_name,
        proposal_config=proposal_config,
        iteration=int(iteration),
        rank=rank,
        world=world,
        score_source=score_source,
    )


def _top_heavy_metrics(
    score: np.ndarray,
    target: np.ndarray,
    ratio: float,
    *,
    force_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    if force_mask is None:
        s_mask, _ = _top_mask(score, ratio)
    else:
        s_mask, _ = _force_aware_top_mask(score, np.asarray(force_mask, dtype=bool).reshape(-1), ratio)
    t_mask, _ = _top_mask(target, ratio)
    denom_count = max(1, int(np.count_nonzero(t_mask)))
    overlap = float(np.count_nonzero(s_mask & t_mask)) / float(denom_count)
    target_clean = np.maximum(np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    oracle_mass = float(target_clean[t_mask].sum())
    policy_mass = float(target_clean[s_mask].sum())
    random_mass = float(target_clean.sum()) * max(0.0, min(1.0, float(ratio)))
    mass_capture = policy_mass / oracle_mass if oracle_mass > 0.0 else 0.0
    regret = oracle_mass - policy_mass
    ndcg = _ndcg_at_mask(score, target_clean, ratio)
    rank_p50, rank_p90 = _oracle_winner_rank_percentiles(score, t_mask)
    return {
        "top_overlap": float(overlap),
        "oracle_mass_capture": float(mass_capture),
        "oracle_topk_mass": float(oracle_mass),
        "policy_oracle_mass": float(policy_mass),
        "random_expected_mass": float(random_mass),
        "random_lift": float(policy_mass / random_mass) if random_mass > 0.0 else 0.0,
        "oracle_gap": float(oracle_mass / policy_mass) if policy_mass > 0.0 else 0.0,
        "regret": float(regret),
        "ndcg": float(ndcg),
        "oracle_top_rank_p50": float(rank_p50),
        "oracle_top_rank_p90": float(rank_p90),
        "policy_selected_ratio": float(np.count_nonzero(s_mask)) / float(max(1, score.size)),
    }


def _score_with_force(score: np.ndarray, force: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    force = np.asarray(force, dtype=bool).reshape(-1)
    if score.size == 0 or force.size != score.size or not bool(force.any()):
        return score
    finite = score[np.isfinite(score)]
    high = float(np.max(finite)) if finite.size else 1.0
    out = score.copy()
    out[force] = np.float32(high * 1.0e6 + 1.0)
    return out


def _autograd_gold_rank_rows(
    *,
    gold_arrays: Dict[str, np.ndarray],
    sample_payload: Dict[str, np.ndarray],
    proposal_config: Any,
    iteration: int,
    rank: int,
    world: int,
    score_source: str,
    first_order_name: str,
) -> List[Dict[str, Any]]:
    if not gold_arrays:
        return []
    ratio = float(getattr(proposal_config, "topk_ratio", 0.0) or 0.0)
    force = np.asarray(sample_payload.get("force", np.zeros(0, dtype=bool)), dtype=bool)
    prod = np.asarray(gold_arrays["q_prod_cached_linear_ehat"], dtype=np.float32)
    prod_force = _score_with_force(prod, force) if force.size == prod.size else prod

    comparisons: Tuple[Tuple[str, str, np.ndarray, str, np.ndarray, Optional[np.ndarray], str], ...] = (
        (
            "offdiag_share",
            "q_diag_abs_hutch_autograd_true_e",
            gold_arrays["q_diag_abs"],
            "q_share_autograd_he_true_e",
            gold_arrays["q_share"],
            None,
            "Does autograd E|zHz| abs-diagonal true-e preserve full-quadratic current-loss attribution?",
        ),
        (
            "offdiag_share_abs_signed_mean",
            "q_diag_abs_signed_mean_autograd_hii_true_e",
            gold_arrays["q_diag_abs_signed_mean"],
            "q_share_autograd_he_true_e",
            gold_arrays["q_share"],
            None,
            "Does |E zHz| signed-diagonal true-e preserve full-quadratic current-loss attribution?",
        ),
        (
            "offdiag_share_signed",
            "q_diag_signed_autograd_hii_true_e",
            gold_arrays["q_diag_signed"],
            "q_share_autograd_he_true_e",
            gold_arrays["q_share"],
            None,
            "Does signed diagonal true-e preserve full-quadratic current-loss attribution?",
        ),
        (
            "offdiag_singleton_marginal",
            "q_diag_abs_hutch_autograd_true_e",
            gold_arrays["q_diag_abs"],
            "q_marg_autograd_he_hii_true_e",
            gold_arrays["q_marg"],
            None,
            "Does autograd E|zHz| abs-diagonal true-e preserve singleton full-quadratic marginal?",
        ),
        (
            "offdiag_singleton_marginal_abs_signed_mean",
            "q_diag_abs_signed_mean_autograd_hii_true_e",
            gold_arrays["q_diag_abs_signed_mean"],
            "q_marg_autograd_he_hii_true_e",
            gold_arrays["q_marg"],
            None,
            "Does |E zHz| signed-diagonal true-e preserve singleton full-quadratic marginal?",
        ),
        (
            "offdiag_singleton_marginal_signed",
            "q_diag_signed_autograd_hii_true_e",
            gold_arrays["q_diag_signed"],
            "q_marg_autograd_he_hii_true_e",
            gold_arrays["q_marg"],
            None,
            "Does signed diagonal true-e preserve singleton full-quadratic marginal?",
        ),
        (
            "lut_loss",
            "q_lut_gold_linear_true_e",
            gold_arrays["q_lut_gold_linear_true_e"],
            "q_diag_abs_hutch_autograd_true_e",
            gold_arrays["q_diag_abs"],
            None,
            "How much does the production H LUT lose relative to autograd E|zHz| abs-H with true e?",
        ),
        (
            "lut_loss_abs_signed_mean",
            "q_lut_gold_linear_true_e",
            gold_arrays["q_lut_gold_linear_true_e"],
            "q_diag_abs_signed_mean_autograd_hii_true_e",
            gold_arrays["q_diag_abs_signed_mean"],
            None,
            "How much does the production H LUT lose relative to |E zHz| signed diagonal with true e?",
        ),
        (
            "cached_linear_loss",
            "q_lut_cached_linear_true_e",
            gold_arrays["q_lut_cached_linear_true_e"],
            "q_lut_gold_linear_true_e",
            gold_arrays["q_lut_gold_linear_true_e"],
            None,
            "How much does cached production first-order signal differ from validation autograd g under the same H LUT and true e?",
        ),
        (
            "ehat_age_loss",
            "q_prod_cached_linear_ehat",
            prod,
            "q_lut_cached_linear_true_e",
            gold_arrays["q_lut_cached_linear_true_e"],
            None,
            "How much does age2bit e_hat lose relative to the same H LUT with true e?",
        ),
        (
            "ehat_age_force_loss",
            "q_prod_cached_linear_ehat_force",
            prod_force,
            "q_lut_cached_linear_true_e",
            gold_arrays["q_lut_cached_linear_true_e"],
            force if force.size == prod.size else None,
            "How much does production-effective force selection recover relative to H LUT with true e?",
        ),
    )

    rows: List[Dict[str, Any]] = []
    for stage, score_name, score, target_name, target, force_mask, question in comparisons:
        score = np.asarray(score, dtype=np.float32).reshape(-1)
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        if score.size <= 0 or score.size != target.size:
            continue
        row: Dict[str, Any] = {
            "mode": "independent_autograd_gold_rank",
            "step": int(iteration),
            "rank": int(rank),
            "world": int(world),
            "stage": str(stage),
            "policy": str(score_name),
            "score": str(score_name),
            "target": str(target_name),
            "question": str(question),
            "score_first_order_source": str(score_source),
            "gold_first_order_name": "autograd_validation_g",
            "production_first_order_name": str(first_order_name),
            "topk_ratio": float(ratio),
            "sample_elements": int(score.size),
            "spearman": _spearman(score, target),
            "pearson": _pearson(score, target),
        }
        if ratio > 0.0:
            metrics = _top_heavy_metrics(score, target, ratio, force_mask=force_mask)
            key = f"@{ratio:.2f}"
            for mkey, val in metrics.items():
                row[f"{mkey}{key}"] = val
        rows.append(row)

    rows.append({
        "mode": "independent_autograd_gold_linear_diag",
        "step": int(iteration),
        "rank": int(rank),
        "world": int(world),
        "score_first_order_source": str(score_source),
        "topk_ratio": float(ratio),
        "sample_elements": int(np.asarray(gold_arrays["g_autograd"]).size),
        "cached_first_order_name": str(first_order_name),
        "g_autograd_vs_cached_spearman": _spearman(
            np.asarray(gold_arrays["g_autograd"], dtype=np.float32),
            np.asarray(sample_payload["m"], dtype=np.float32),
        ),
        "g_autograd_vs_cached_pearson": _pearson(
            np.asarray(gold_arrays["g_autograd"], dtype=np.float32),
            np.asarray(sample_payload["m"], dtype=np.float32),
        ),
    })
    return rows


def _force_aware_top_mask(score: np.ndarray, force: np.ndarray, ratio: float) -> Tuple[np.ndarray, float]:
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    force = np.asarray(force, dtype=bool).reshape(-1)
    n = int(score.size)
    if n <= 0 or force.size != n:
        return _top_mask(score, ratio)
    k = int(math.ceil(max(0.0, min(1.0, float(ratio))) * float(max(1, n))))
    out = force.copy()
    remaining = max(0, k - int(np.count_nonzero(out)))
    if remaining <= 0:
        return out, float("nan")
    clean = np.nan_to_num(score.copy(), nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    clean[out] = -np.inf
    non_force = int(np.count_nonzero(~out))
    if non_force <= 0:
        return out, float("nan")
    if remaining >= non_force:
        out |= ~out
        return out, float(np.nanmin(score))
    threshold = np.partition(clean, n - remaining)[n - remaining]
    add = clean > threshold
    need = remaining - int(np.count_nonzero(add))
    if need > 0:
        ties = np.flatnonzero(clean == threshold)
        add[ties[:need]] = True
    out |= add
    return out, float(threshold)


def _ehat_diagnostic_row(
    *,
    e: np.ndarray,
    e_age: np.ndarray,
    age_code: np.ndarray,
    force: np.ndarray,
    score: np.ndarray,
    target: np.ndarray,
    target_name: str,
    proposal_config: Any,
    iteration: int,
    rank: int,
    world: int,
    score_source: str,
) -> Dict[str, Any]:
    e = np.asarray(e, dtype=np.float32).reshape(-1)
    e_age = np.asarray(e_age, dtype=np.float32).reshape(-1)
    age_code = np.asarray(age_code, dtype=np.uint8).reshape(-1)
    force = np.asarray(force, dtype=bool).reshape(-1)
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    target_clean = np.maximum(
        np.nan_to_num(np.asarray(target, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    ratio = float(getattr(proposal_config, "topk_ratio", 0.0) or 0.0)
    oracle_mask, _ = _top_mask(target_clean, ratio)
    policy_mask, _ = _top_mask(score, ratio)
    force_policy_mask, _ = _force_aware_top_mask(score, force, ratio)
    oracle_mass = max(float(target_clean[oracle_mask].sum()), 1.0e-30)

    abs_e = np.abs(e)
    abs_e_age = np.abs(e_age)
    valid_ratio = np.isfinite(abs_e) & np.isfinite(abs_e_age) & (abs_e > np.float32(1.0e-30))
    ratio_values = abs_e_age[valid_ratio] / (abs_e[valid_ratio] + np.float32(1.0e-30))
    valid_sign = np.isfinite(e) & np.isfinite(e_age) & (abs_e > np.float32(1.0e-30)) & (abs_e_age > np.float32(1.0e-30))
    sign_match = (e[valid_sign] * e_age[valid_sign]) >= np.float32(0.0)

    row: Dict[str, Any] = {
        "mode": "independent_oracle_ehat_diag",
        "step": int(iteration),
        "rank": int(rank),
        "world": int(world),
        "policy": "m_t+diag_abs_h+agebit",
        "target": str(target_name),
            "score_first_order_source": score_source,
        "topk_ratio": float(ratio),
        "sample_elements": int(e.size),
        "ehat_abs_mean": float(np.mean(abs_e_age)) if e_age.size else 0.0,
        "true_e_abs_mean": float(np.mean(abs_e)) if e.size else 0.0,
        "ehat_over_true_e_p50": _percentile_or_zero(ratio_values, 50.0),
        "ehat_over_true_e_p90": _percentile_or_zero(ratio_values, 90.0),
        "ehat_over_true_e_p99": _percentile_or_zero(ratio_values, 99.0),
        "ehat_true_e_sign_match": float(np.mean(sign_match)) if sign_match.size else 0.0,
        "force_candidate_ratio": float(np.mean(force)) if force.size else 0.0,
        "force_in_oracle_top_ratio": (
            float(np.count_nonzero(force & oracle_mask)) / float(max(1, np.count_nonzero(oracle_mask)))
            if force.size == oracle_mask.size else 0.0
        ),
        "force_in_policy_top_ratio": (
            float(np.count_nonzero(force & policy_mask)) / float(max(1, np.count_nonzero(policy_mask)))
            if force.size == policy_mask.size else 0.0
        ),
        "force_in_forceaware_policy_ratio": (
            float(np.count_nonzero(force & force_policy_mask)) / float(max(1, np.count_nonzero(force_policy_mask)))
            if force.size == force_policy_mask.size else 0.0
        ),
        "force_mass_capture_vs_oracle_top": (
            float(target_clean[force].sum()) / oracle_mass
            if force.size == target_clean.size else 0.0
        ),
    }
    for code in range(4):
        mask = age_code == code
        row[f"age_code{code}_ratio"] = float(np.mean(mask)) if mask.size else 0.0
        row[f"oracle_top_age_code{code}_ratio"] = (
            float(np.count_nonzero(mask & oracle_mask)) / float(max(1, np.count_nonzero(oracle_mask)))
            if mask.size == oracle_mask.size else 0.0
        )
        row[f"policy_top_age_code{code}_ratio"] = (
            float(np.count_nonzero(mask & policy_mask)) / float(max(1, np.count_nonzero(policy_mask)))
            if mask.size == policy_mask.size else 0.0
        )
        row[f"forceaware_policy_age_code{code}_ratio"] = (
            float(np.count_nonzero(mask & force_policy_mask)) / float(max(1, np.count_nonzero(force_policy_mask)))
            if mask.size == force_policy_mask.size else 0.0
        )
        row[f"missed_oracle_age_code{code}_mass_share"] = (
            float(target_clean[mask & oracle_mask & ~policy_mask].sum()) / oracle_mass
            if mask.size == target_clean.size else 0.0
        )
    return row


def _percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size <= 0:
        return 0.0
    return float(np.percentile(finite, float(percentile)))


def _top_mask(values: np.ndarray, ratio: float) -> Tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    n = int(values.size)
    k = int(math.ceil(max(0.0, min(1.0, float(ratio))) * float(max(1, n))))
    if k <= 0 or n <= 0:
        return np.zeros(n, dtype=bool), float("nan")
    if k >= n:
        return np.ones(n, dtype=bool), float(np.nanmin(values))
    clean = np.nan_to_num(values.copy(), nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    threshold = np.partition(clean, n - k)[n - k]
    mask = clean > threshold
    need = k - int(mask.sum())
    if need > 0:
        ties = np.flatnonzero(clean == threshold)
        mask[ties[:need]] = True
    return mask, float(threshold)


def _ndcg_at_mask(score: np.ndarray, gain: np.ndarray, ratio: float) -> float:
    n = int(score.size)
    k = min(n, int(math.ceil(float(ratio) * float(max(1, n)))))
    if k <= 0:
        return 0.0
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf))[:k]
    ideal = np.argsort(-np.nan_to_num(gain, nan=-np.inf))[:k]
    denom = np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float((gain[order].astype(np.float64) / denom).sum())
    idcg = float((gain[ideal].astype(np.float64) / denom).sum())
    return dcg / idcg if idcg > 0.0 else 0.0


def _oracle_winner_rank_percentiles(score: np.ndarray, oracle_mask: np.ndarray) -> Tuple[float, float]:
    n = int(score.size)
    if n <= 0 or not bool(np.any(oracle_mask)):
        return 1.0, 1.0
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf))
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, dtype=np.int64)
    pct = (ranks[oracle_mask].astype(np.float64) + 1.0) / float(n)
    return float(np.percentile(pct, 50)), float(np.percentile(pct, 90))


def _lookup_h_np(v: np.ndarray, h_values: np.ndarray, log_min: float, inv_width: float) -> Tuple[np.ndarray, np.ndarray]:
    bins = int(h_values.size)
    if bins <= 1:
        return np.zeros_like(v, dtype=np.float32), np.zeros_like(v, dtype=np.int64)
    pos = (np.log(np.maximum(v.astype(np.float32), np.float32(1.0e-30))) - np.float32(log_min)) * np.float32(inv_width)
    pos = np.clip(pos, 0.0, float(bins - 1))
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, bins - 1)
    frac = pos - lo.astype(np.float32)
    h = h_values[lo] + frac * (h_values[hi] - h_values[lo])
    return h.astype(np.float32, copy=False), lo


def _lookup_geom_np(geom: np.ndarray, lut_idx: np.ndarray, age_code: np.ndarray, bins: int) -> np.ndarray:
    if geom.size < 4 * max(1, bins):
        return np.ones_like(lut_idx, dtype=np.float32)
    idx = age_code.astype(np.int64) * int(bins) + lut_idx.astype(np.int64)
    idx = np.clip(idx, 0, geom.size - 1)
    return geom[idx].astype(np.float32, copy=False)


def _age2_phase_hit_np(
    global_ids: np.ndarray,
    *,
    step: int,
    period: int,
    salt: int,
) -> np.ndarray:
    ids = np.asarray(global_ids, dtype=np.uint64).reshape(-1)
    if ids.size <= 0:
        return np.zeros(0, dtype=bool)
    x = ids.astype(np.uint32, copy=False) ^ np.uint32(int(salt) & 0xFFFFFFFF)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint32(16))) * np.uint32(0x7FEB352D)
        x = (x ^ (x >> np.uint32(15))) * np.uint32(0x846CA68B)
        x = x ^ (x >> np.uint32(16))
    phase = (x % np.uint32(max(1, int(period)))).astype(np.int32)
    return ((np.int32(int(step)) + phase) % np.int32(max(1, int(period)))) == 0


def _unpack_age(words: Optional[np.ndarray], global_ids: np.ndarray) -> np.ndarray:
    if words is None:
        return np.zeros(global_ids.shape, dtype=np.uint8)
    word_idx = global_ids >> 4
    valid = word_idx < int(words.shape[0])
    out = np.zeros(global_ids.shape, dtype=np.uint8)
    if not bool(valid.any()):
        return out
    shift = ((global_ids[valid] & 15) << 1).astype(np.uint32)
    out[valid] = ((words[word_idx[valid]] >> shift) & np.uint32(3)).astype(np.uint8, copy=False)
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size != a.size:
        return 0.0
    return _pearson(_rankdata(a), _rankdata(b))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size != a.size:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0.0 else 0.0


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _all_reduce_max_float(value: float, *, group=None) -> float:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return float(value)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)
    return float(t.item())


def _rank_world() -> Tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
    return 0, 1


def _log_dir(proposal_config: Any) -> Path:
    raw = getattr(proposal_config, "independent_oracle_log_dir", None)
    if raw:
        return Path(raw)
    return Path(getattr(proposal_config, "cp_dir", Path("./proposal_cp"))) / "independent_oracle"


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
