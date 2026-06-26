"""Telemetry for one independent policy run.

This module measures the checkpoint image owned by the current run.  It does
not simulate several policies from one shared checkpoint image.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .ideal_oracle import (
    _all_reduce_mean_float,
    _all_reduce_sum_float,
    _all_reduce_sum_int,
    _build_forward_loss_bank,
    _eval_loss_bank,
    _restore_rng_state,
)


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
    if not getattr(lossy_cp, "_groups", None):
        return
    rank, world = _rank_world()
    t0 = time.perf_counter()
    log_dir = _log_dir(proposal_config)
    log_dir.mkdir(parents=True, exist_ok=True)

    actual_batches = max(1, int(getattr(proposal_config, "independent_oracle_actual_batches", 4) or 4))
    hvp_batches = max(0, int(getattr(proposal_config, "independent_oracle_hvp_batches", 1) or 0))
    loss_bank = _build_forward_loss_bank(
        model=model,
        data_iterator=data_iterator,
        n_batches=actual_batches,
    )
    if not loss_bank:
        return

    loss_base = _eval_loss_bank(loss_bank)
    loss_policy = _eval_checkpoint_loss(
        lossy_cp=lossy_cp,
        model=model,
        loss_bank=loss_bank,
        checkpoint_key="theta_cp_flat",
    )
    loss_initial = _eval_checkpoint_loss(
        lossy_cp=lossy_cp,
        model=model,
        loss_bank=loss_bank,
        checkpoint_key="theta_cp_initial_flat",
    )
    actual_remaining = float(loss_policy) - float(loss_base)
    actual_initial = float(loss_initial) - float(loss_base)
    captured_since_init = float(actual_initial) - float(actual_remaining)

    e_stats = _full_e_stats(lossy_cp)
    r1_m = _full_linear_dot(lossy_cp, linear_key="m_flat")
    r1_g = _full_linear_dot(lossy_cp, linear_key="g_score_flat")
    r2_full = None
    third_m = None
    third_g = None
    hvp_sample: Optional[np.ndarray] = None

    sample = _build_sample(lossy_cp, proposal_config, rank=rank)
    sample_payload = _gather_sample_payload(lossy_cp, sample)

    if hvp_batches > 0 and e_stats["global_rms"] > 0.0:
        r2_full, hvp_sample = _hvp_for_current_error(
            lossy_cp=lossy_cp,
            model=model,
            loss_bank=loss_bank[:hvp_batches],
            fd_eps=float(getattr(proposal_config, "ideal_oracle_fd_eps", None) or getattr(proposal_config, "finite_diff_eps", 1.0e-3) or 1.0e-3),
            rms_e=float(e_stats["global_rms"]),
            sample=sample,
        )
        third_m = float(actual_remaining) - (float(r1_m) + float(r2_full))
        third_g = float(actual_remaining) - (float(r1_g) + float(r2_full))

    row = {
        "mode": "independent_oracle_loss",
        "step": int(iteration),
        "rank": int(rank),
        "world": int(world),
        "policy": str(getattr(proposal_config, "experiment_score_policy", "disabled")),
        "topk_ratio": float(getattr(proposal_config, "topk_ratio", 0.0)),
        "loss_base": float(loss_base),
        "loss_policy_checkpoint": float(loss_policy),
        "loss_initial_checkpoint": float(loss_initial),
        "actual_loss_remaining": float(actual_remaining),
        "actual_loss_initial_checkpoint": float(actual_initial),
        "captured_since_init": float(captured_since_init),
        "e_global_rms": float(e_stats["global_rms"]),
        "e_global_l2": float(e_stats["global_l2"]),
        "e_global_abs_mean": float(e_stats["global_abs_mean"]),
        "e_global_max": float(e_stats["global_max"]),
        "R1_m_t": float(r1_m),
        "R1_g": float(r1_g),
        "R2_full_hvp": float(r2_full) if r2_full is not None else None,
        "third_order_residual_m_t": float(third_m) if third_m is not None else None,
        "third_order_residual_g": float(third_g) if third_g is not None else None,
        "actual_batches": int(len(loss_bank)),
        "hvp_batches": int(hvp_batches),
    }
    _append_jsonl(log_dir / f"independent_loss_rank{rank:03d}.jsonl", row)

    rank_rows = _rank_metric_rows(
        sample_payload=sample_payload,
        hvp_sample=hvp_sample,
        proposal_config=proposal_config,
        iteration=int(iteration),
        rank=rank,
        world=world,
    )
    for rank_row in rank_rows:
        _append_jsonl(log_dir / f"independent_rank_rank{rank:03d}.jsonl", rank_row)

    timing = {
        "mode": "independent_oracle_timing",
        "step": int(iteration),
        "rank": int(rank),
        "elapsed_ms": float((time.perf_counter() - t0) * 1000.0),
        "actual_batches": int(len(loss_bank)),
        "hvp_batches": int(hvp_batches),
        "sample_elements": int(sample_payload["e"].size),
    }
    _append_jsonl(log_dir / f"independent_timing_rank{rank:03d}.jsonl", timing)
    _restore_current_params(lossy_cp)


@torch.no_grad()
def _eval_checkpoint_loss(
    *,
    lossy_cp: Any,
    model: Any,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    checkpoint_key: str,
) -> float:
    try:
        _set_params_from_key(lossy_cp, checkpoint_key)
        return float(_eval_loss_bank(loss_bank))
    finally:
        _restore_current_params(lossy_cp)


@torch.no_grad()
def _set_params_from_key(lossy_cp: Any, key: str) -> None:
    for grp in getattr(lossy_cp, "_groups", ()) or ():
        src_flat = grp.get(key)
        if src_flat is None:
            continue
        local_off = 0
        for name, size in zip(grp.get("names", ()), grp.get("sizes", ())):
            n = int(size)
            param = lossy_cp._name_to_param.get(str(name))
            if param is not None and n > 0:
                src = src_flat.narrow(0, local_off, n).view(param.shape)
                param.data.copy_(src.to(device=param.device, dtype=param.dtype), non_blocking=True)
            local_off += n
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _restore_current_params(lossy_cp: Any) -> None:
    for grp in getattr(lossy_cp, "_groups", ()) or ():
        p_flat = grp.get("p_flat")
        if p_flat is None:
            continue
        local_off = 0
        for name, size in zip(grp.get("names", ()), grp.get("sizes", ())):
            n = int(size)
            param = lossy_cp._name_to_param.get(str(name))
            if param is not None and n > 0:
                src = p_flat.narrow(0, local_off, n).view(param.shape)
                param.data.copy_(src.to(device=param.device, dtype=param.dtype), non_blocking=True)
            local_off += n
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _set_params_perturbed(lossy_cp: Any, *, scale: float) -> None:
    for grp in getattr(lossy_cp, "_groups", ()) or ():
        p_flat = grp.get("p_flat")
        c_flat = grp.get("theta_cp_flat")
        if p_flat is None or c_flat is None:
            continue
        local_off = 0
        for name, size in zip(grp.get("names", ()), grp.get("sizes", ())):
            n = int(size)
            param = lossy_cp._name_to_param.get(str(name))
            if param is not None and n > 0:
                p = p_flat.narrow(0, local_off, n)
                c = c_flat.narrow(0, local_off, n)
                value = p + float(scale) * (p - c)
                param.data.copy_(value.view(param.shape).to(device=param.device, dtype=param.dtype), non_blocking=True)
            local_off += n
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _full_e_stats(lossy_cp: Any) -> Dict[str, float]:
    local_sumsq = 0.0
    local_abs = 0.0
    local_max = 0.0
    local_n = 0
    for grp in getattr(lossy_cp, "_groups", ()) or ():
        p = grp.get("p_flat")
        c = grp.get("theta_cp_flat")
        if p is None or c is None:
            continue
        e = (p - c).detach().float()
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


def _full_linear_dot(lossy_cp: Any, *, linear_key: str) -> float:
    local = 0.0
    for grp in getattr(lossy_cp, "_groups", ()) or ():
        p = grp.get("p_flat")
        c = grp.get("theta_cp_flat")
        lin = grp.get(linear_key)
        if p is None or c is None or lin is None:
            continue
        e = (p - c).detach().float()
        linear = lin.detach().float()
        local += float(torch.dot(-linear, e).detach().item())
    return _all_reduce_sum_float(local, group=getattr(lossy_cp, "_proposal_cpu_pg", None))


def _build_sample(lossy_cp: Any, proposal_config: Any, *, rank: int) -> Dict[str, Any]:
    max_samples = int(getattr(proposal_config, "independent_oracle_rank_sample", 1048576) or 0)
    groups = list(getattr(lossy_cp, "_groups", ()) or ())
    spans: List[Tuple[int, int, int]] = []
    total = 0
    for group_idx, grp in enumerate(groups):
        n = int(grp.get("total_aligned", grp.get("total", 0)) or 0)
        if n <= 0:
            continue
        spans.append((group_idx, total, total + n))
        total += n
    n_sample = min(max_samples, total)
    if n_sample <= 0:
        return {"items": [], "n": 0}
    seed = int(getattr(proposal_config, "independent_oracle_rank_sample_seed", 20240625) or 20240625)
    rng = np.random.default_rng(seed + int(rank) * 1000003)
    ids = np.sort(rng.choice(total, size=n_sample, replace=False).astype(np.int64))
    items = []
    for group_idx, lo, hi in spans:
        left = int(np.searchsorted(ids, lo, side="left"))
        right = int(np.searchsorted(ids, hi, side="left"))
        if right <= left:
            continue
        local_offsets = ids[left:right] - np.int64(lo)
        items.append({
            "group_idx": int(group_idx),
            "positions": np.arange(left, right, dtype=np.int64),
            "local_offsets_np": local_offsets.astype(np.int64, copy=False),
            "local_offsets_t": torch.as_tensor(local_offsets, dtype=torch.long, device=groups[group_idx]["p_flat"].device),
        })
    return {"items": items, "n": int(n_sample)}


@torch.no_grad()
def _gather_sample_payload(lossy_cp: Any, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
    n = int(sample.get("n", 0))
    out = {
        "e": np.zeros(n, dtype=np.float32),
        "m": np.zeros(n, dtype=np.float32),
        "g": np.zeros(n, dtype=np.float32),
        "v": np.zeros(n, dtype=np.float32),
        "h_abs": np.zeros(n, dtype=np.float32),
        "score_m_diag": np.zeros(n, dtype=np.float32),
        "score_g_diag": np.zeros(n, dtype=np.float32),
        "score_agebit": np.zeros(n, dtype=np.float32),
        "score_g_agebit": np.zeros(n, dtype=np.float32),
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
        idx = item["local_offsets_t"]
        p = torch.index_select(grp["p_flat"], 0, idx).detach().float().cpu().numpy()
        c = torch.index_select(grp["theta_cp_flat"], 0, idx).detach().float().cpu().numpy()
        m = torch.index_select(grp["m_flat"], 0, idx).detach().float().cpu().numpy()
        g = torch.index_select(grp["g_score_flat"], 0, idx).detach().float().cpu().numpy()
        v = torch.index_select(grp["v_flat"], 0, idx).detach().float().cpu().numpy()
        e = (p - c).astype(np.float32, copy=False)
        h_abs, lut_idx = _lookup_h_np(v, h_values_np, log_min_f, inv_w_f)
        global_ids = (
            np.asarray(item["local_offsets_np"], dtype=np.int64)
            + int(grp.get("master_off", 0)) * int(getattr(lossy_cp, "group_size", 1))
        )
        age_code = _unpack_age(age_words_np, global_ids)
        geom = _lookup_geom_np(geom_np, lut_idx, age_code, h_values_np.size)
        denom = np.sqrt(np.maximum(v, 0.0)).astype(np.float32) * np.float32(sqrt_bc2_inv) + np.float32(eps)
        e_age = (-np.float32(lr_over_bc1) * m / denom * geom).astype(np.float32, copy=False)
        out["e"][pos] = e
        out["m"][pos] = m
        out["g"][pos] = g
        out["v"][pos] = v
        out["h_abs"][pos] = h_abs
        out["score_m_diag"][pos] = np.maximum(-m * e + 0.5 * h_abs * e * e, 0.0)
        out["score_g_diag"][pos] = np.maximum(-g * e + 0.5 * h_abs * e * e, 0.0)
        out["score_agebit"][pos] = np.maximum(-m * e_age + 0.5 * h_abs * e_age * e_age, 0.0)
        out["score_g_agebit"][pos] = np.maximum(-g * e_age + 0.5 * h_abs * e_age * e_age, 0.0)
    return out


def _hvp_for_current_error(
    *,
    lossy_cp: Any,
    model: Any,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    fd_eps: float,
    rms_e: float,
    sample: Dict[str, Any],
) -> Tuple[float, np.ndarray]:
    if not loss_bank:
        return 0.0, np.zeros(int(sample.get("n", 0)), dtype=np.float32)
    sample_acc = np.zeros(int(sample.get("n", 0)), dtype=np.float32)
    r2_acc = 0.0
    inv_rms = 1.0 / float(rms_e)
    try:
        for forward_loss, rng_state in loss_bank:
            _set_params_perturbed(lossy_cp, scale=float(fd_eps) * inv_rms)
            dot_plus, grad_plus = _backward_dot_and_sample(
                lossy_cp=lossy_cp,
                model=model,
                forward_loss=forward_loss,
                rng_state=rng_state,
                sample=sample,
            )
            _set_params_perturbed(lossy_cp, scale=-float(fd_eps) * inv_rms)
            dot_minus, grad_minus = _backward_dot_and_sample(
                lossy_cp=lossy_cp,
                model=model,
                forward_loss=forward_loss,
                rng_state=rng_state,
                sample=sample,
            )
            local_ehe = float(rms_e) * (float(dot_plus) - float(dot_minus)) / (2.0 * float(fd_eps))
            global_ehe = _all_reduce_sum_float(local_ehe, group=getattr(lossy_cp, "_proposal_cpu_pg", None))
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


def _backward_dot_and_sample(
    *,
    lossy_cp: Any,
    model: Any,
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
    rng_state: Optional[Dict[str, Any]],
    sample: Dict[str, Any],
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
    for grp in getattr(lossy_cp, "_groups", ()) or ():
        p_flat = grp.get("p_flat")
        c_flat = grp.get("theta_cp_flat")
        if p_flat is None or c_flat is None:
            continue
        local_off = 0
        for name, size in zip(grp.get("names", ()), grp.get("sizes", ())):
            n = int(size)
            param = lossy_cp._name_to_param.get(str(name))
            if param is None or n <= 0:
                local_off += n
                continue
            e_slice = (p_flat.narrow(0, local_off, n) - c_flat.narrow(0, local_off, n)).detach()
            sample_info = sample_by_name.get(str(name))

            def _hook(raw_grad, *, _e=e_slice, _sample=sample_info):
                flat = raw_grad.detach().reshape(-1).float()
                dot_tensors.append(torch.sum(flat * _e.float()))
                if _sample is not None:
                    vals = torch.index_select(flat, 0, _sample["offsets_t"]).detach().cpu().numpy()
                    sample_out[_sample["positions"]] = vals.astype(np.float32, copy=False)
                return torch.zeros_like(raw_grad)

            handles.append(param.register_hook(_hook))
            local_off += n
    try:
        loss, _ = forward_loss()
        if loss.dim() > 0:
            loss = loss.mean()
        if hasattr(module, "backward") and hasattr(module, "optimizer"):
            module.backward(loss)
        else:
            loss.backward(retain_graph=False)
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
        local_offsets = np.asarray(item["local_offsets_np"], dtype=np.int64)
        positions = np.asarray(item["positions"], dtype=np.int64)
        local_base = 0
        for name, size in zip(grp.get("names", ()), grp.get("sizes", ())):
            n = int(size)
            lo = int(np.searchsorted(local_offsets, local_base, side="left"))
            hi = int(np.searchsorted(local_offsets, local_base + n, side="left"))
            if hi > lo:
                offs = local_offsets[lo:hi] - np.int64(local_base)
                out[str(name)] = {
                    "positions": positions[lo:hi],
                    "offsets_t": torch.as_tensor(
                        offs,
                        dtype=torch.long,
                        device=grp["p_flat"].device,
                    ),
                }
            local_base += n
    return out


def _rank_metric_rows(
    *,
    sample_payload: Dict[str, np.ndarray],
    hvp_sample: Optional[np.ndarray],
    proposal_config: Any,
    iteration: int,
    rank: int,
    world: int,
) -> List[Dict[str, Any]]:
    e = sample_payload["e"]
    m = sample_payload["m"]
    g = sample_payload["g"]
    h_abs = sample_payload["h_abs"]
    score_m_diag = sample_payload["score_m_diag"]
    score_g_diag = sample_payload["score_g_diag"]
    score_agebit = sample_payload["score_agebit"]
    score_g_agebit = sample_payload["score_g_agebit"]
    candidates = {
        "m_t_diag_abs_h": score_m_diag,
        "g_diag_abs_h": score_g_diag,
        "agebit_m_t_diag_abs_h": score_agebit,
        "agebit_g_diag_abs_h": score_g_agebit,
        "m_t_first_only": np.maximum(-m * e, 0.0).astype(np.float32, copy=False),
        "g_first_only": np.maximum(-g * e, 0.0).astype(np.float32, copy=False),
    }
    targets = {
        "m_t_diag_abs_h": score_m_diag,
        "g_diag_abs_h": score_g_diag,
    }
    if hvp_sample is not None:
        targets["m_t_hvp_offdiag"] = np.maximum(-m * e + 0.5 * e * hvp_sample, 0.0).astype(np.float32, copy=False)
        targets["g_hvp_offdiag"] = np.maximum(-g * e + 0.5 * e * hvp_sample, 0.0).astype(np.float32, copy=False)
    ratios = (0.02, 0.04, 0.08)
    rows: List[Dict[str, Any]] = []
    for target_name, target in targets.items():
        for score_name, score in candidates.items():
            row = {
                "mode": "independent_oracle_rank",
                "step": int(iteration),
                "rank": int(rank),
                "world": int(world),
                "policy": str(getattr(proposal_config, "experiment_score_policy", "disabled")),
                "topk_ratio": float(getattr(proposal_config, "topk_ratio", 0.0)),
                "score": score_name,
                "target": target_name,
                "sample_elements": int(score.size),
                "spearman": _spearman(score, target),
                "pearson": _pearson(score, target),
            }
            for ratio in ratios:
                metrics = _top_heavy_metrics(score, target, ratio)
                key = f"@{ratio:.2f}"
                for mkey, val in metrics.items():
                    row[f"{mkey}{key}"] = val
            rows.append(row)
    return rows


def _top_heavy_metrics(score: np.ndarray, target: np.ndarray, ratio: float) -> Dict[str, float]:
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    s_mask, _ = _top_mask(score, ratio)
    t_mask, _ = _top_mask(target, ratio)
    denom_count = max(1, int(np.count_nonzero(t_mask)))
    overlap = float(np.count_nonzero(s_mask & t_mask)) / float(denom_count)
    target_clean = np.maximum(np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    oracle_mass = float(target_clean[t_mask].sum())
    policy_mass = float(target_clean[s_mask].sum())
    mass_capture = policy_mass / oracle_mass if oracle_mass > 0.0 else 0.0
    regret = oracle_mass - policy_mass
    ndcg = _ndcg_at_mask(score, target_clean, ratio)
    rank_p50, rank_p90 = _oracle_winner_rank_percentiles(score, t_mask)
    return {
        "top_overlap": float(overlap),
        "oracle_mass_capture": float(mass_capture),
        "regret": float(regret),
        "ndcg": float(ndcg),
        "oracle_top_rank_p50": float(rank_p50),
        "oracle_top_rank_p90": float(rank_p90),
    }


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
