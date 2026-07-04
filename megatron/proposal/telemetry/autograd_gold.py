"""Autograd gold-reference telemetry for sampled proposal coordinates.

This module is intentionally telemetry-only.  It computes full-quadratic
reference scores on a deterministic sampled coordinate set without storing a
full HVP tensor or a full Hessian diagonal tensor on GPU.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class GoldSample:
    arrays: Dict[str, np.ndarray]
    meta: Dict[str, Any]


@dataclass
class _SampleEntry:
    spec_index: int
    positions: np.ndarray
    offsets: torch.Tensor


def autograd_gold_enabled() -> bool:
    return str(os.environ.get("PROPOSAL_INDEPENDENT_ORACLE_AUTOGRAD_GOLD", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def collect_autograd_gold_sample(
    *,
    lossy_cp: Any,
    model: Any,
    specs: Sequence[Dict[str, Any]],
    sample: Dict[str, Any],
    sample_payload: Dict[str, np.ndarray],
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    checkpoint_tensor_for_name: Callable[..., Optional[torch.Tensor]],
    restore_current_params: Callable[[Any], None],
    restore_rng_state: Callable[[Dict[str, Any]], None],
    iteration: int,
    rank: int,
) -> GoldSample:
    """Return autograd-HVP reference values on the independent-oracle sample.

    The first autograd pass builds ``g = grad(loss, theta, create_graph=True)``.
    From that graph we compute:

    * ``(H e)_I`` by differentiating ``g^T e`` only w.r.t. sampled params.
    * ``H_ii`` by Hutchinson on sampled coordinates only:
      ``E_z[z_i * (H z)_i]``.

    The full ``H e`` and full ``H_ii`` tensors are never materialized.
    """

    n = int(sample.get("n", 0))
    if n <= 0 or not loss_bank:
        return GoldSample(arrays={}, meta={"enabled": True, "sample_elements": n, "reason": "empty"})

    sample_entries = _sample_entries_by_spec(specs, sample)
    sampled_param_indices = sorted({entry.spec_index for entry in sample_entries})
    if not sampled_param_indices:
        return GoldSample(arrays={}, meta={"enabled": True, "sample_elements": n, "reason": "empty_sample_entries"})

    gold_batches = _env_int("PROPOSAL_INDEPENDENT_ORACLE_AUTOGRAD_GOLD_BATCHES", 1, minimum=1)
    h_ii_z = _env_int("PROPOSAL_INDEPENDENT_ORACLE_AUTOGRAD_GOLD_HII_Z", 4, minimum=0)
    loss_bank = list(loss_bank[:gold_batches])
    params = [spec["param"] for spec in specs]
    sampled_params = [specs[i]["param"] for i in sampled_param_indices]

    g_acc = np.zeros(n, dtype=np.float64)
    he_acc = np.zeros(n, dtype=np.float64)
    hii_acc = np.zeros(n, dtype=np.float64)
    hii_sq_acc = np.zeros(n, dtype=np.float64)
    hii_abs_acc = np.zeros(n, dtype=np.float64)
    hii_abs_sq_acc = np.zeros(n, dtype=np.float64)
    hii_obs = 0
    z_seed_base = (
        _env_int("PROPOSAL_INDEPENDENT_ORACLE_AUTOGRAD_GOLD_Z_SEED", 20240625, minimum=0)
        + int(iteration) * 1009
        + int(rank) * 9176
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        alloc_start = int(torch.cuda.memory_allocated())
        reserved_start = int(torch.cuda.memory_reserved())
    else:
        alloc_start = reserved_start = 0
    t0 = time.perf_counter()
    status = "ok"
    reason = None

    try:
        with _deepspeed_autograd_gold_guard(model):
            for batch_idx, (forward_loss, rng_state) in enumerate(loss_bank):
                restore_current_params(lossy_cp)
                restore_rng_state(rng_state)
                _zero_param_grads(_module(model))
                loss, _ = forward_loss()
                if loss.dim() > 0:
                    loss = loss.mean()
                grads = torch.autograd.grad(
                    loss,
                    params,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )

                g_acc += _sample_from_grad_list(grads, specs, sample_entries, n).astype(np.float64, copy=False)

                s_e = _dot_grads_with_current_error(
                    grads=grads,
                    specs=specs,
                    lossy_cp=lossy_cp,
                    checkpoint_tensor_for_name=checkpoint_tensor_for_name,
                )
                if not bool(s_e.requires_grad):
                    raise RuntimeError(
                        "autograd gold could not build a second-order graph for g^T e. "
                        "This usually means the current forward path uses a checkpointing "
                        "or custom autograd mode that is not differentiable to second order; "
                        "disable reentrant activation checkpointing for autograd-gold telemetry."
                    )
                he_grads = torch.autograd.grad(
                    s_e,
                    sampled_params,
                    retain_graph=(h_ii_z > 0),
                    create_graph=False,
                    allow_unused=True,
                )
                he_acc += _sample_from_indexed_grad_list(
                    he_grads,
                    sampled_param_indices,
                    specs,
                    sample_entries,
                    n,
                ).astype(np.float64, copy=False)

                for z_idx in range(h_ii_z):
                    s_z, signs = _dot_full_grads_with_rademacher(
                        grads=grads,
                        specs=specs,
                        sample_entries=sample_entries,
                        device_fallback=loss.device,
                        z_seed=int(z_seed_base + batch_idx * 1_000_003 + z_idx),
                    )
                    if not bool(s_z.requires_grad):
                        raise RuntimeError(
                            "autograd gold could not build a second-order graph for full Hutchinson z."
                        )
                    hz_grads = torch.autograd.grad(
                        s_z,
                        sampled_params,
                        retain_graph=(z_idx + 1 < h_ii_z),
                        create_graph=False,
                        allow_unused=True,
                    )
                    hz = _sample_from_indexed_grad_list(
                        hz_grads,
                        sampled_param_indices,
                        specs,
                        sample_entries,
                        n,
                    ).astype(np.float64, copy=False)
                    obs = hz * signs.astype(np.float64, copy=False)
                    hii_acc += obs
                    hii_sq_acc += obs * obs
                    abs_obs = np.abs(obs)
                    hii_abs_acc += abs_obs
                    hii_abs_sq_acc += abs_obs * abs_obs
                    hii_obs += 1

                del loss, grads, s_e, he_grads
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    except RuntimeError as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "error"
        reason = str(exc)
        if torch.cuda.is_available() and status == "oom":
            torch.cuda.empty_cache()
        raise
    finally:
        restore_current_params(lossy_cp)
        _zero_param_grads(_module(model))

    batch_n = max(1, len(loss_bank))
    g = (g_acc / float(batch_n)).astype(np.float32, copy=False)
    he = (he_acc / float(batch_n)).astype(np.float32, copy=False)
    if hii_obs > 0:
        hii_mean = (hii_acc / float(hii_obs)).astype(np.float32, copy=False)
        hii_var = np.maximum((hii_sq_acc / float(hii_obs)) - (hii_acc / float(hii_obs)) ** 2, 0.0)
        hii_se = np.sqrt(hii_var / float(hii_obs)).astype(np.float32, copy=False)
        hii_abs_mean = (hii_abs_acc / float(hii_obs)).astype(np.float32, copy=False)
        hii_abs_var = np.maximum(
            (hii_abs_sq_acc / float(hii_obs)) - (hii_abs_acc / float(hii_obs)) ** 2,
            0.0,
        )
        hii_abs_se = np.sqrt(hii_abs_var / float(hii_obs)).astype(np.float32, copy=False)
    else:
        hii_mean = np.zeros(n, dtype=np.float32)
        hii_se = np.zeros(n, dtype=np.float32)
        hii_abs_mean = np.zeros(n, dtype=np.float32)
        hii_abs_se = np.zeros(n, dtype=np.float32)

    e = np.asarray(sample_payload["e"], dtype=np.float32)
    v = np.asarray(sample_payload.get("v", np.zeros_like(e)), dtype=np.float32)
    h_lut = np.asarray(sample_payload["h_abs"], dtype=np.float32)
    cached_linear = np.asarray(sample_payload["m"], dtype=np.float32)
    e_age = np.asarray(sample_payload["e_age"], dtype=np.float32)
    global_id = np.asarray(sample_payload.get("global_id", np.zeros_like(e, dtype=np.int64)), dtype=np.int64)
    sqrt_v = np.sqrt(np.maximum(v, np.float32(0.0)), dtype=np.float32)
    h_lut_abs = np.abs(h_lut).astype(np.float32, copy=False)

    arrays = {
        "g_autograd": g,
        "he_autograd": he,
        "hii_autograd": hii_mean,
        "hii_autograd_se": hii_se,
        "hii_autograd_abs_hutch": hii_abs_mean,
        "hii_autograd_abs_hutch_se": hii_abs_se,
        "q_share": _positive(-g * e + np.float32(0.5) * e * he),
        "q_marg": _positive(-g * e + e * he - np.float32(0.5) * hii_mean * e * e),
        "q_diag_signed": _positive(-g * e + np.float32(0.5) * hii_mean * e * e),
        "q_diag_abs_signed_mean": _positive(-g * e + np.float32(0.5) * np.abs(hii_mean) * e * e),
        "q_diag_abs": _positive(-g * e + np.float32(0.5) * hii_abs_mean * e * e),
        "q_lut_gold_linear_true_e": _positive(-g * e + np.float32(0.5) * h_lut * e * e),
        "q_lut_cached_linear_true_e": _positive(-cached_linear * e + np.float32(0.5) * h_lut * e * e),
        "q_prod_cached_linear_ehat": _positive(-cached_linear * e_age + np.float32(0.5) * h_lut * e_age * e_age),
    }
    meta = {
        "enabled": True,
        "status": status,
        "reason": reason,
        "sample_elements": int(n),
        "gold_batches": int(batch_n),
        "hii_z": int(h_ii_z),
        "hii_observations": int(hii_obs),
        "elapsed_ms": float((time.perf_counter() - t0) * 1000.0),
        "cuda_alloc_start": int(alloc_start),
        "cuda_reserved_start": int(reserved_start),
        "cuda_peak_alloc": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "cuda_peak_reserved": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        "hii_abs_mean": float(np.mean(np.abs(hii_mean))) if n else 0.0,
        "hii_se_abs_mean": float(np.mean(np.abs(hii_se))) if n else 0.0,
        "hii_abs_hutch_mean": float(np.mean(hii_abs_mean)) if n else 0.0,
        "hii_abs_hutch_se_mean": float(np.mean(hii_abs_se)) if n else 0.0,
        "hii_relative_se_p50": _relative_se_percentile(hii_mean, hii_se, 50.0),
        "hii_relative_se_p90": _relative_se_percentile(hii_mean, hii_se, 90.0),
        "hii_abs_hutch_relative_se_p50": _relative_se_percentile(hii_abs_mean, hii_abs_se, 50.0),
        "hii_abs_hutch_relative_se_p90": _relative_se_percentile(hii_abs_mean, hii_abs_se, 90.0),
        "g_abs_mean": float(np.mean(np.abs(g))) if n else 0.0,
        "he_abs_mean": float(np.mean(np.abs(he))) if n else 0.0,
        "e_abs_mean": float(np.mean(np.abs(e))) if n else 0.0,
        "e_age_abs_mean": float(np.mean(np.abs(e_age))) if n else 0.0,
        "h_lut_abs_mean": float(np.mean(np.abs(h_lut))) if n else 0.0,
        "cached_linear_abs_mean": float(np.mean(np.abs(cached_linear))) if n else 0.0,
    }
    meta.update(_prefix_stats("v_t", v))
    meta.update(_prefix_stats("sqrt_v_t", sqrt_v))
    meta.update(_prefix_stats("production_h_lut_abs", h_lut_abs))
    meta.update(_prefix_stats("autograd_abs_hii_hutch", hii_abs_mean))
    meta.update(_ratio_stats("production_h_lut_abs_over_autograd_abs_hii_hutch", h_lut_abs, hii_abs_mean))
    meta.update({
        "sqrt_v_t_vs_autograd_abs_hii_hutch_spearman": _spearman_np(sqrt_v, hii_abs_mean),
        "sqrt_v_t_vs_autograd_abs_hii_hutch_pearson": _pearson_np(sqrt_v, hii_abs_mean),
        "v_t_vs_autograd_abs_hii_hutch_spearman": _spearman_np(v, hii_abs_mean),
        "v_t_vs_autograd_abs_hii_hutch_pearson": _pearson_np(v, hii_abs_mean),
        "production_h_lut_abs_vs_autograd_abs_hii_hutch_spearman": _spearman_np(h_lut_abs, hii_abs_mean),
        "production_h_lut_abs_vs_autograd_abs_hii_hutch_pearson": _pearson_np(h_lut_abs, hii_abs_mean),
    })
    meta.update(_block_curvature_stats(
        global_id=global_id,
        sqrt_v=sqrt_v,
        h_lut_abs=h_lut_abs,
        autograd_abs_h=hii_abs_mean,
    ))
    return GoldSample(arrays=arrays, meta=meta)


def _sample_entries_by_spec(specs: Sequence[Dict[str, Any]], sample: Dict[str, Any]) -> List[_SampleEntry]:
    entries: List[_SampleEntry] = []
    specs_by_group: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, spec in enumerate(specs):
        specs_by_group.setdefault(int(spec["group_idx"]), []).append((idx, spec))
    for item in sample.get("items", ()):
        group_idx = int(item["group_idx"])
        local_offsets = np.asarray(item["local_offsets_np"], dtype=np.int64)
        positions = np.asarray(item["positions"], dtype=np.int64)
        for spec_idx, spec in specs_by_group.get(group_idx, ()):
            start = int(spec["local_off"])
            end = start + int(spec["n"])
            lo = int(np.searchsorted(local_offsets, start, side="left"))
            hi = int(np.searchsorted(local_offsets, end, side="left"))
            if hi <= lo:
                continue
            offsets_np = (
                local_offsets[lo:hi]
                - np.int64(start)
                + np.int64(int(spec.get("param_offset", 0) or 0))
            )
            entries.append(_SampleEntry(
                spec_index=int(spec_idx),
                positions=positions[lo:hi].astype(np.int64, copy=True),
                offsets=torch.as_tensor(
                    offsets_np,
                    dtype=torch.long,
                    device=spec["param"].device,
                ),
            ))
    return entries


def _sample_from_grad_list(
    grads: Sequence[Optional[torch.Tensor]],
    specs: Sequence[Dict[str, Any]],
    entries: Sequence[_SampleEntry],
    n: int,
) -> np.ndarray:
    out = np.zeros(int(n), dtype=np.float32)
    for entry in entries:
        grad = grads[int(entry.spec_index)]
        if grad is None:
            continue
        vals = torch.index_select(grad.reshape(-1).detach().float(), 0, entry.offsets)
        out[entry.positions] = vals.cpu().numpy().astype(np.float32, copy=False)
    return out


def _sample_from_indexed_grad_list(
    grads: Sequence[Optional[torch.Tensor]],
    grad_spec_indices: Sequence[int],
    specs: Sequence[Dict[str, Any]],
    entries: Sequence[_SampleEntry],
    n: int,
) -> np.ndarray:
    by_spec = {int(spec_idx): grad for spec_idx, grad in zip(grad_spec_indices, grads)}
    out = np.zeros(int(n), dtype=np.float32)
    for entry in entries:
        grad = by_spec.get(int(entry.spec_index))
        if grad is None:
            continue
        vals = torch.index_select(grad.reshape(-1).detach().float(), 0, entry.offsets)
        out[entry.positions] = vals.cpu().numpy().astype(np.float32, copy=False)
    return out


def _dot_grads_with_current_error(
    *,
    grads: Sequence[Optional[torch.Tensor]],
    specs: Sequence[Dict[str, Any]],
    lossy_cp: Any,
    checkpoint_tensor_for_name: Callable[..., Optional[torch.Tensor]],
) -> torch.Tensor:
    total: Optional[torch.Tensor] = None
    for grad, spec in zip(grads, specs):
        if grad is None:
            continue
        grp = spec["grp"]
        n = int(spec["n"])
        logical_off = int(spec["local_off"])
        elem_off = int(spec.get("elem_off", logical_off) or 0)
        age_off = int(spec.get("age_off", logical_off) or 0)
        p = grp["p_flat"].narrow(0, elem_off, n)
        c = checkpoint_tensor_for_name(
            lossy_cp,
            grp,
            str(spec["name"]),
            logical_off,
            n,
            checkpoint_off=age_off,
            device=p.device,
            dtype=p.dtype,
        )
        if c is None:
            continue
        e = (p.detach().float() - c.detach().float()).reshape(-1)
        grad_slice = grad.reshape(-1).narrow(
            0,
            int(spec.get("param_offset", 0) or 0),
            n,
        ).float()
        term = torch.sum(grad_slice * e)
        total = term if total is None else total + term
    if total is None:
        device = specs[0]["param"].device if specs else torch.device("cpu")
        total = torch.zeros((), device=device, dtype=torch.float32)
    return total


def _dot_full_grads_with_rademacher(
    *,
    grads: Sequence[Optional[torch.Tensor]],
    specs: Sequence[Dict[str, Any]],
    sample_entries: Sequence[_SampleEntry],
    device_fallback: torch.device,
    z_seed: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    total: Optional[torch.Tensor] = None
    signs = np.zeros(
        max((int(pos) for entry in sample_entries for pos in entry.positions), default=-1) + 1,
        dtype=np.float32,
    )
    entries_by_spec: Dict[int, List[_SampleEntry]] = {}
    for entry in sample_entries:
        entries_by_spec.setdefault(int(entry.spec_index), []).append(entry)

    for spec_idx, grad in enumerate(grads):
        if grad is None:
            continue
        grad_f = grad.reshape(-1).float()
        generator = torch.Generator(device=grad_f.device)
        generator.manual_seed(int(z_seed + spec_idx * 104_729))
        z = torch.empty(
            grad_f.shape,
            dtype=torch.float32,
            device=grad_f.device,
        )
        z.bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
        term = torch.sum(grad_f * z)
        total = term if total is None else total + term
        for entry in entries_by_spec.get(int(spec_idx), ()):
            vals = torch.index_select(z.detach(), 0, entry.offsets)
            signs[entry.positions] = vals.cpu().numpy().astype(np.float32, copy=False)
    if total is None:
        total = torch.zeros((), device=device_fallback, dtype=torch.float32)
    return total, signs


def _zero_param_grads(module: Any) -> None:
    try:
        for param in module.parameters():
            param.grad = None
    except Exception:
        pass


@contextmanager
def _deepspeed_autograd_gold_guard(model: Any):
    """Disable DeepSpeed ZeRO grad side effects while torch.autograd.grad runs.

    Autograd gold needs graph derivatives, not ZeRO gradient reduction or engine
    backward epilogues.  DeepSpeed registers post-accumulate hooks on params and
    a post-backward callback on its optimizer; direct ``autograd.grad`` can trip
    those hooks as if it were a training backward.  This guard removes only those
    side-effect hooks for the telemetry window and recreates them afterwards.
    """

    engine = _engine(model)
    opt = getattr(engine, "optimizer", None)
    grad_hooks = getattr(opt, "_grad_acc_hooks", None)
    recreate_hooks = getattr(opt, "create_gradient_handling_hooks", None)
    hook_state = getattr(opt, "_backward_hook_state", None)

    removed_grad_hooks = False
    saved_post_hooks = None
    saved_hook_state: Dict[str, Any] = {}
    saved_engine_state: Dict[str, Any] = {}

    for attr in ("_running_engine_backward", "_manual_backward_expected", "_scale_wrt_gas"):
        if hasattr(engine, attr):
            try:
                saved_engine_state[attr] = getattr(engine, attr)
            except Exception:
                pass
    # DeepSpeedEngine.forward installs output backward hooks that divide
    # gradients by GAS when ``_scale_wrt_gas`` is true.  Autograd-gold calls
    # torch.autograd.grad directly, not engine.backward(), so that hook would
    # silently shrink g/H by gradient_accumulation_steps.  Keep ZeRO side
    # effects disabled and measure the unscaled mathematical loss.
    if hasattr(engine, "_scale_wrt_gas"):
        try:
            engine._scale_wrt_gas = False
        except Exception:
            pass
    if hook_state is not None:
        for attr in (
            "remaining_grad_acc_hooks",
            "backward_active_depth",
            "backward_seen_this_step",
            "epilogue_ran_this_backward",
            "hooks_fired_this_backward",
            "max_expected_hooks_seen",
            "post_backward_callback_queued",
            "post_backward_callback_graph_task_id",
        ):
            if hasattr(hook_state, attr):
                try:
                    saved_hook_state[attr] = getattr(hook_state, attr)
                except Exception:
                    pass
        post_hooks = getattr(hook_state, "_grad_acc_post_hooks", None)
        if isinstance(post_hooks, list):
            saved_post_hooks = list(post_hooks)
            hook_state._grad_acc_post_hooks = []

    if isinstance(grad_hooks, list) and grad_hooks and callable(recreate_hooks):
        for handle in list(grad_hooks):
            try:
                handle.remove()
            except Exception:
                pass
        opt._grad_acc_hooks = []
        removed_grad_hooks = True

    try:
        yield
    finally:
        if saved_post_hooks is not None and hook_state is not None:
            hook_state._grad_acc_post_hooks = saved_post_hooks
        for attr, value in saved_hook_state.items():
            try:
                setattr(hook_state, attr, value)
            except Exception:
                pass
        for attr, value in saved_engine_state.items():
            try:
                setattr(engine, attr, value)
            except Exception:
                pass
        if removed_grad_hooks:
            try:
                recreate_hooks()
            except Exception:
                pass


def _engine(model: Any) -> Any:
    return model[0] if isinstance(model, (list, tuple)) and model else model


def _module(model: Any) -> Any:
    return model[0] if isinstance(model, (list, tuple)) else model


def _positive(values: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        np.float32(0.0),
    ).astype(np.float32, copy=False)


def _relative_se_percentile(mean: np.ndarray, se: np.ndarray, pct: float) -> float:
    denom = np.abs(np.asarray(mean, dtype=np.float64))
    rel = np.asarray(se, dtype=np.float64) / np.maximum(denom, 1.0e-30)
    finite = rel[np.isfinite(rel)]
    if finite.size <= 0:
        return 0.0
    return float(np.percentile(finite, float(pct)))


def _prefix_stats(prefix: str, values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {f"{prefix}_{key}": 0.0 for key in ("mean", "p50", "p90", "p99", "max")}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_p50": float(np.percentile(arr, 50.0)),
        f"{prefix}_p90": float(np.percentile(arr, 90.0)),
        f"{prefix}_p99": float(np.percentile(arr, 99.0)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _block_curvature_stats(
    *,
    global_id: np.ndarray,
    sqrt_v: np.ndarray,
    h_lut_abs: np.ndarray,
    autograd_abs_h: np.ndarray,
) -> Dict[str, float]:
    """Block-level v/H rank diagnostics for telemetry.

    Coordinate-level v_t is noisy.  The proposal's quantile matching is closer
    to a distribution/block statement, so also report whether block-averaged
    sqrt(v_t) and H_LUT order block-averaged autograd |H_ii|.
    """

    block_size = _env_int("PROPOSAL_INDEPENDENT_ORACLE_AUTOGRAD_GOLD_BLOCK_SIZE", 16384, minimum=1)
    min_count = _env_int("PROPOSAL_INDEPENDENT_ORACLE_AUTOGRAD_GOLD_BLOCK_MIN_SAMPLES", 16, minimum=1)
    gids = np.asarray(global_id, dtype=np.int64).reshape(-1)
    if gids.size <= 0:
        return {
            "block_diag_block_size": float(block_size),
            "block_diag_blocks": 0.0,
            "sqrt_v_block_mean_vs_autograd_abs_hii_hutch_spearman": 0.0,
            "sqrt_v_block_mean_vs_autograd_abs_hii_hutch_pearson": 0.0,
            "production_h_lut_block_mean_vs_autograd_abs_hii_hutch_spearman": 0.0,
            "production_h_lut_block_mean_vs_autograd_abs_hii_hutch_pearson": 0.0,
        }
    bid = gids // np.int64(block_size)
    order = np.argsort(bid, kind="mergesort")
    bid_s = bid[order]
    sqrt_s = np.asarray(sqrt_v, dtype=np.float64).reshape(-1)[order]
    lut_s = np.asarray(h_lut_abs, dtype=np.float64).reshape(-1)[order]
    h_s = np.asarray(autograd_abs_h, dtype=np.float64).reshape(-1)[order]
    b_sqrt: List[float] = []
    b_lut: List[float] = []
    b_h: List[float] = []
    start = 0
    n = int(bid_s.size)
    while start < n:
        end = start + 1
        while end < n and bid_s[end] == bid_s[start]:
            end += 1
        if end - start >= min_count:
            b_sqrt.append(float(np.mean(sqrt_s[start:end])))
            b_lut.append(float(np.mean(lut_s[start:end])))
            b_h.append(float(np.mean(h_s[start:end])))
        start = end
    sqrt_arr = np.asarray(b_sqrt, dtype=np.float64)
    lut_arr = np.asarray(b_lut, dtype=np.float64)
    h_arr = np.asarray(b_h, dtype=np.float64)
    return {
        "block_diag_block_size": float(block_size),
        "block_diag_blocks": float(h_arr.size),
        "sqrt_v_block_mean_vs_autograd_abs_hii_hutch_spearman": _spearman_np(sqrt_arr, h_arr),
        "sqrt_v_block_mean_vs_autograd_abs_hii_hutch_pearson": _pearson_np(sqrt_arr, h_arr),
        "production_h_lut_block_mean_vs_autograd_abs_hii_hutch_spearman": _spearman_np(lut_arr, h_arr),
        "production_h_lut_block_mean_vs_autograd_abs_hii_hutch_pearson": _pearson_np(lut_arr, h_arr),
        **_sample_order_block_curvature_stats(
            sqrt_v=sqrt_v,
            h_lut_abs=h_lut_abs,
            autograd_abs_h=autograd_abs_h,
        ),
    }


def _sample_order_block_curvature_stats(
    *,
    sqrt_v: np.ndarray,
    h_lut_abs: np.ndarray,
    autograd_abs_h: np.ndarray,
) -> Dict[str, float]:
    """Chunk the deterministic rank sample into equal-count coordinate-order blocks.

    ``global_id`` is the agebit word address and is group-local for multiple
    optimizer groups.  It is correct for age lookup, but it collapses distinct
    groups when used as a model-coordinate block id.  The sample itself is
    sorted by the global flat sample order, so equal-count chunks provide a
    stable telemetry-only block diagnostic without changing production logic.
    """

    block_size = _env_int(
        "PROPOSAL_INDEPENDENT_ORACLE_AUTOGRAD_GOLD_SAMPLE_BLOCK_SIZE",
        4096,
        minimum=1,
    )
    sqrt_arr = np.asarray(sqrt_v, dtype=np.float64).reshape(-1)
    lut_arr = np.asarray(h_lut_abs, dtype=np.float64).reshape(-1)
    h_arr = np.asarray(autograd_abs_h, dtype=np.float64).reshape(-1)
    n = int(min(sqrt_arr.size, lut_arr.size, h_arr.size))
    if n <= 0:
        return {
            "sample_block_size": float(block_size),
            "sample_blocks": 0.0,
            "sqrt_v_sample_block_mean_vs_autograd_abs_hii_hutch_spearman": 0.0,
            "sqrt_v_sample_block_mean_vs_autograd_abs_hii_hutch_pearson": 0.0,
            "production_h_lut_sample_block_mean_vs_autograd_abs_hii_hutch_spearman": 0.0,
            "production_h_lut_sample_block_mean_vs_autograd_abs_hii_hutch_pearson": 0.0,
        }
    usable = (n // block_size) * block_size
    if usable < block_size:
        return {
            "sample_block_size": float(block_size),
            "sample_blocks": 0.0,
            "sqrt_v_sample_block_mean_vs_autograd_abs_hii_hutch_spearman": 0.0,
            "sqrt_v_sample_block_mean_vs_autograd_abs_hii_hutch_pearson": 0.0,
            "production_h_lut_sample_block_mean_vs_autograd_abs_hii_hutch_spearman": 0.0,
            "production_h_lut_sample_block_mean_vs_autograd_abs_hii_hutch_pearson": 0.0,
        }
    shape = (-1, block_size)
    b_sqrt = sqrt_arr[:usable].reshape(shape).mean(axis=1)
    b_lut = lut_arr[:usable].reshape(shape).mean(axis=1)
    b_h = h_arr[:usable].reshape(shape).mean(axis=1)
    return {
        "sample_block_size": float(block_size),
        "sample_blocks": float(b_h.size),
        "sqrt_v_sample_block_mean_vs_autograd_abs_hii_hutch_spearman": _spearman_np(b_sqrt, b_h),
        "sqrt_v_sample_block_mean_vs_autograd_abs_hii_hutch_pearson": _pearson_np(b_sqrt, b_h),
        "production_h_lut_sample_block_mean_vs_autograd_abs_hii_hutch_spearman": _spearman_np(b_lut, b_h),
        "production_h_lut_sample_block_mean_vs_autograd_abs_hii_hutch_pearson": _pearson_np(b_lut, b_h),
    }


def _ratio_stats(prefix: str, numer: np.ndarray, denom: np.ndarray) -> Dict[str, float]:
    n = np.asarray(numer, dtype=np.float64).reshape(-1)
    d = np.asarray(denom, dtype=np.float64).reshape(-1)
    mask = np.isfinite(n) & np.isfinite(d) & (np.abs(d) > 1.0e-30)
    if not bool(np.any(mask)):
        return {f"{prefix}_{key}": 0.0 for key in ("mean", "p50", "p90", "p99")}
    ratio = n[mask] / d[mask]
    return {
        f"{prefix}_mean": float(np.mean(ratio)),
        f"{prefix}_p50": float(np.percentile(ratio, 50.0)),
        f"{prefix}_p90": float(np.percentile(ratio, 90.0)),
        f"{prefix}_p99": float(np.percentile(ratio, 99.0)),
    }


def _pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    x = x[mask] - float(np.mean(x[mask]))
    y = y[mask] - float(np.mean(y[mask]))
    denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    return _pearson_np(_rank_average(x[mask]), _rank_average(y[mask]))


def _rank_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=np.float64)
    sorted_x = x[order]
    start = 0
    n = int(sorted_x.size)
    while start < n:
        end = start + 1
        while end < n and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1)
        start = end
    return ranks


def _env_int(name: str, default: int, *, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except Exception:
        value = int(default)
    return max(int(minimum), int(value))
