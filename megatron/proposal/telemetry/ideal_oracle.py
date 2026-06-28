"""Telemetry-only mixed-age rollback Taylor/HVP oracle.

This module is deliberately outside the production cutoff/write path.  It runs
only from the H-refresh training halt, consumes the test_dram recovery image as
the checkpoint vector c, and restores model parameters after every probe.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class _OracleTensor:
    name: str
    param: torch.nn.Parameter
    theta: np.ndarray
    checkpoint: np.ndarray
    m: np.ndarray
    v: np.ndarray
    h_signed: np.ndarray
    h_abs_lut: np.ndarray
    e_full: np.ndarray
    e_agebit: np.ndarray
    age_code: np.ndarray
    global_start: int
    score_abs_lut: np.ndarray
    g: Optional[np.ndarray] = None


def ideal_oracle_enabled(proposal_config: Any, iteration: int) -> bool:
    if proposal_config is None:
        return False
    if not bool(getattr(proposal_config, "ideal_oracle_enable", False)):
        return False
    iters = set(int(x) for x in getattr(proposal_config, "ideal_oracle_iters", ()) or ())
    return int(iteration) in iters


def capture_signed_h_cpu(signed_h_views: Iterable[torch.Tensor]) -> List[np.ndarray]:
    """Copy signed diagonal-H estimates before H refresh mutates them to abs()."""

    out: List[np.ndarray] = []
    for view in signed_h_views:
        out.append(view.detach().reshape(-1).float().cpu().numpy().copy())
    return out


def run_ideal_oracle_from_anchor(
    *,
    model: Any,
    data_iterator: Any,
    iteration: int,
    lossy_cp: Any,
    proposal_config: Any,
    mapper: Any,
    names: Sequence[str],
    expert_params: Sequence[torch.nn.Parameter],
    signed_h_cpu: Sequence[np.ndarray],
    grad_accum_views: Sequence[torch.Tensor],
) -> Optional[Dict[str, Any]]:
    """Run the oracle from inside the H-refresh anchor halt.

    The caller already drained checkpoint workers, detached DeepSpeed grad hooks,
    snapshotted RNG/buffers/grads, and will restore them in its outer finally.
    """

    t0 = time.perf_counter()
    rank, world = _rank_world()
    if _is_pipeline_engine(model[0] if isinstance(model, (list, tuple)) else model):
        record = {
            "mode": "ideal_oracle_skip",
            "step": int(iteration),
            "rank": int(rank),
            "reason": "pipeline_engine_not_supported",
        }
        _append_jsonl(_log_path(proposal_config, "ideal_oracle_terms", rank), record)
        return record
    if str(getattr(proposal_config, "cp_target", "")) != "test_dram":
        record = {
            "mode": "ideal_oracle_skip",
            "step": int(iteration),
            "rank": int(rank),
            "reason": "requires_cp_target_test_dram",
        }
        _append_jsonl(_log_path(proposal_config, "ideal_oracle_terms", rank), record)
        return record
    if len(names) != len(expert_params) or len(names) != len(signed_h_cpu):
        raise RuntimeError("ideal oracle got inconsistent tensor/name/H views")

    records = _build_tensor_records(
        lossy_cp=lossy_cp,
        mapper=mapper,
        names=names,
        expert_params=expert_params,
        signed_h_cpu=signed_h_cpu,
    )
    del signed_h_cpu
    gc.collect()
    local_n = int(sum(r.e_full.size for r in records))
    if local_n <= 0:
        return None

    total_n = int(_all_reduce_sum_int(local_n, group=getattr(lossy_cp, "_proposal_cpu_pg", None)))
    if total_n <= 0:
        return None

    actual_loss_bank = _build_forward_loss_bank(
        model=model,
        data_iterator=data_iterator,
        n_batches=int(os.environ.get("PROPOSAL_IDEAL_ORACLE_ACTUAL_BATCHES", "1") or "1"),
    )
    if not actual_loss_bank:
        return None
    forward_loss, paired_rng_state = actual_loss_bank[0]

    _restore_theta(records)
    loss_base_global = _eval_loss_bank(actual_loss_bank)
    if bool(int(os.environ.get("PROPOSAL_IDEAL_ORACLE_MEASURE_NOISE", "0") or "0")):
        _measure_loss_noise(
            model=model,
            data_iterator=data_iterator,
            proposal_config=proposal_config,
            records=records,
            iteration=int(iteration),
            base_loss=float(loss_base_global),
            rank=rank,
        )

    _restore_theta(records)
    grad_batches = max(
        1,
        min(
            len(actual_loss_bank),
            int(os.environ.get("PROPOSAL_IDEAL_ORACLE_GRAD_BATCHES", "1") or "1"),
        ),
    )
    g_parts = _run_backward_bank_to_cpu_arrays(
        loss_bank=actual_loss_bank[:grad_batches],
        expert_params=expert_params,
        grad_accum_views=grad_accum_views,
        model=model,
    )
    for rec, g in zip(records, g_parts):
        rec.g = g

    k_ratios = tuple(float(x) for x in getattr(proposal_config, "ideal_oracle_k_ratios", ()) or ())
    if not k_ratios:
        k_ratios = (0.0, 0.01, 0.02, 0.04, 0.08, 0.16)
    fd_eps = getattr(proposal_config, "ideal_oracle_fd_eps", None)
    if fd_eps is None:
        fd_eps = getattr(proposal_config, "finite_diff_eps", 1.0e-3)
    fd_eps = float(fd_eps or 1.0e-3)

    k_ratios = tuple(x for x in k_ratios if float(x) > 0.0)
    if not k_ratios:
        k_ratios = (0.01, 0.04, 0.08)

    term_rows, rank_rows = _run_selection_oracle_v2(
        records=records,
        k_ratios=k_ratios,
        iteration=int(iteration),
        rank=int(rank),
        world=int(world),
        local_n=int(local_n),
        total_n=int(total_n),
        loss_base=float(loss_base_global),
        actual_loss_bank=actual_loss_bank,
        forward_loss=forward_loss,
        paired_rng_state=paired_rng_state,
        fd_eps=float(fd_eps),
        proposal_config=proposal_config,
        lossy_cp=lossy_cp,
        expert_params=expert_params,
        grad_accum_views=grad_accum_views,
        model=model,
    )

    _restore_theta(records)
    for view in grad_accum_views:
        view.zero_()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    timing = {
        "mode": "ideal_oracle_timing",
        "step": int(iteration),
        "rank": int(rank),
        "elapsed_ms": float(elapsed_ms),
        "n_terms": int(len(term_rows)),
        "n_rank_rows": int(len(rank_rows)),
        "local_elements": int(local_n),
        "global_elements": int(total_n),
        "actual_eval_batches": int(len(actual_loss_bank)),
        "grad_batches": int(grad_batches),
    }
    _append_jsonl(_log_path(proposal_config, "ideal_oracle_timing", rank), timing)
    return timing


def _run_selection_oracle_v2(
    *,
    records: Sequence[_OracleTensor],
    k_ratios: Sequence[float],
    iteration: int,
    rank: int,
    world: int,
    local_n: int,
    total_n: int,
    loss_base: float,
    actual_loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
    paired_rng_state: Dict[str, Any],
    fd_eps: float,
    proposal_config: Any,
    lossy_cp: Any,
    expert_params: Sequence[torch.nn.Parameter],
    grad_accum_views: Sequence[torch.Tensor],
    model: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Selection-centered oracle used by the report experiments.

    Actual rollback is always evaluated with the true mixed-age error
    ``theta_cur - theta_cp``.  ``e_source=agebit`` only changes the score used
    to choose the saved top-K set.
    """

    group = getattr(lossy_cp, "_proposal_cpu_pg", None)
    sample_seed_raw = os.environ.get("PROPOSAL_IDEAL_ORACLE_SAMPLE_SEED", "")
    sample_seed = (
        int(sample_seed_raw)
        if str(sample_seed_raw).strip()
        else 0x1DEA2026 + int(iteration)
    )
    sample_plan = _make_sample_plan(
        records,
        max_samples=int(getattr(proposal_config, "ideal_oracle_rank_sample", 1048576) or 0),
        seed=int(sample_seed),
    )
    exact_selected_hvp = str(
        os.environ.get("PROPOSAL_IDEAL_ORACLE_EXACT_SELECTED_HVP", "0")
    ).lower() in ("1", "true", "yes", "on")
    e_source_filter = set(_env_csv_tuple("PROPOSAL_IDEAL_ORACLE_E_SOURCES", ("true", "agebit")))
    linear_filter = set(_env_csv_tuple("PROPOSAL_IDEAL_ORACLE_LINEAR_SOURCES", ("m_t", "g")))
    second_filter = set(_env_csv_tuple(
        "PROPOSAL_IDEAL_ORACLE_SECOND_ORDERS",
        ("none", "diag_abs_h", "hvp_offdiag"),
    ))
    rank_second_filter = set(_env_csv_tuple(
        "PROPOSAL_IDEAL_ORACLE_RANK_SECOND_ORDERS",
        tuple(second_filter),
    ))
    exact_hvp_e_filter = set(_env_csv_tuple(
        "PROPOSAL_IDEAL_ORACLE_EXACT_SELECTED_HVP_E_SOURCES",
        tuple(e_source_filter),
    ))
    exact_hvp_linear_filter = set(_env_csv_tuple(
        "PROPOSAL_IDEAL_ORACLE_EXACT_SELECTED_HVP_LINEAR_SOURCES",
        tuple(linear_filter),
    ))
    exact_hvp_second_filter = set(_env_csv_tuple(
        "PROPOSAL_IDEAL_ORACLE_EXACT_SELECTED_HVP_SECOND_ORDERS",
        tuple(second_filter),
    ))
    exact_hvp_k_filter = set(
        float(x)
        for x in _env_csv_tuple(
            "PROPOSAL_IDEAL_ORACLE_EXACT_SELECTED_HVP_K_RATIOS",
            tuple(str(float(x)) for x in k_ratios),
        )
    )
    exact_hvp_policy_filter = _parse_policy_combo_filter(
        os.environ.get("PROPOSAL_IDEAL_ORACLE_EXACT_SELECTED_HVP_POLICIES", "")
    )
    pairwise_second_filter = set(_env_csv_tuple(
        "PROPOSAL_IDEAL_ORACLE_PAIRWISE_SECOND_ORDERS",
        ("none", "diag_abs_h"),
    ))
    skip_hvp = bool(int(os.environ.get("PROPOSAL_IDEAL_ORACLE_SKIP_HVP", "0") or "0"))
    skip_rank = bool(int(os.environ.get("PROPOSAL_IDEAL_ORACLE_SKIP_RANK", "0") or "0"))
    hvp_batches = max(
        1,
        min(
            len(actual_loss_bank),
            int(os.environ.get("PROPOSAL_IDEAL_ORACLE_HVP_BATCHES", "1") or "1"),
        ),
    )
    hvp_loss_bank = actual_loss_bank[:hvp_batches]

    e_true_parts = [rec.e_full for rec in records]
    _restore_theta(records)
    _set_theta_minus_e(records, e_true_parts)
    loss_no_save = _eval_loss_bank(actual_loss_bank)
    actual_no_save = float(loss_no_save) - float(loss_base)
    _restore_theta(records)

    all_e_parts_by_source: Dict[str, List[np.ndarray]] = {
        "true": e_true_parts,
        "agebit": [rec.e_agebit for rec in records],
    }
    e_parts_by_source: Dict[str, List[np.ndarray]] = {
        name: parts for name, parts in all_e_parts_by_source.items() if name in e_source_filter
    }
    hvp_parts_by_source: Dict[str, List[np.ndarray]] = {}
    for e_source, e_parts in e_parts_by_source.items():
        rms_e = _rms_for_parts(e_parts, total_n=total_n, group=group)
        if skip_hvp:
            r2_base = 0.0
            h_e_parts = [np.zeros_like(e) for e in e_parts]
        else:
            r2_base, hvp_payload = _hvp_quadratic_form_bank(
                records=records,
                e_parts=e_parts,
                rms_e=float(rms_e),
                fd_eps=float(fd_eps),
                loss_bank=hvp_loss_bank,
                expert_params=expert_params,
                grad_accum_views=grad_accum_views,
                model=model,
                sample_plan=None,
                lossy_cp=lossy_cp,
                return_h_e_parts=True,
            )
            h_e_parts = []
            if hvp_payload and "h_e_parts" in hvp_payload:
                h_e_parts = list(hvp_payload["h_e_parts"])  # type: ignore[arg-type]
            else:
                h_e_parts = [np.zeros_like(e) for e in e_parts]
        hvp_parts_by_source[e_source] = h_e_parts
        first_m = _linear_loss_sum(
            records,
            e_parts=e_parts,
            linear_source="m_t",
            group=group,
        )
        first_g = _linear_loss_sum(
            records,
            e_parts=e_parts,
            linear_source="g",
            group=group,
        )
        first_full_m = float(first_m) + float(r2_base)
        first_full_g = float(first_g) + float(r2_base)
        row = {
            "mode": "ideal_oracle_baseline",
            "step": int(iteration),
            "rank": int(rank),
            "world": int(world),
            "e_source": str(e_source),
            "local_elements": int(local_n),
            "global_elements": int(total_n),
            "loss_base": float(loss_base),
            "loss_no_save_rollback": float(loss_no_save),
            "actual_loss_no_save": float(actual_no_save),
            "first_order_m_t_no_save": float(first_m),
            "first_order_g_no_save": float(first_g),
            "second_order_full_hvp_for_e_source": float(r2_base),
            "first_plus_full_hvp_m_t_no_save": float(first_full_m),
            "first_plus_full_hvp_g_no_save": float(first_full_g),
            "actual_minus_first_plus_full_hvp_m_t_no_save": float(actual_no_save) - float(first_full_m),
            "actual_minus_first_plus_full_hvp_g_no_save": float(actual_no_save) - float(first_full_g),
            "e_rms": float(rms_e),
            "fd_eps": float(fd_eps),
            "age_code0_ratio": _age_code_ratio(records, 0),
            "age_code1_ratio": _age_code_ratio(records, 1),
            "age_code2_ratio": _age_code_ratio(records, 2),
            "age_code3_ratio": _age_code_ratio(records, 3),
            "actual_eval_batches": int(len(actual_loss_bank)),
            "grad_batches": int(os.environ.get("PROPOSAL_IDEAL_ORACLE_GRAD_BATCHES", "1") or "1"),
            "hvp_batches": int(hvp_batches),
            "hvp_skipped": bool(skip_hvp),
        }
        _append_jsonl(_log_path(proposal_config, "ideal_oracle_baseline", rank), row)

    rank_rows: List[Dict[str, Any]] = []
    term_rows: List[Dict[str, Any]] = []
    for e_source, e_parts in e_parts_by_source.items():
        h_e_parts = hvp_parts_by_source[e_source]
        if (
            not skip_rank
            and "m_t" in linear_filter
            and "g" in linear_filter
            and sample_plan["record_idx"].size > 0
        ):
            for second_order in ("none", "diag_abs_h"):
                if second_order not in pairwise_second_filter:
                    continue
                rank_rows.extend(_record_pairwise_score_metrics_v2(
                    records=records,
                    e_source=e_source,
                    second_order=second_order,
                    left_linear_source="m_t",
                    right_linear_source="g",
                    sample_plan=sample_plan,
                    k_ratios=k_ratios,
                    iteration=iteration,
                    rank=rank,
                    total_n=total_n,
                    proposal_config=proposal_config,
                    h_e_parts=h_e_parts,
                ))
        for linear_source in ("m_t", "g"):
            if linear_source not in linear_filter:
                continue
            score_none = _score_parts_v2(
                records,
                e_parts=e_parts,
                h_e_parts=h_e_parts,
                linear_source=linear_source,
                second_order="none",
                clamp=True,
            )
            score_diag = _score_parts_v2(
                records,
                e_parts=e_parts,
                h_e_parts=h_e_parts,
                linear_source=linear_source,
                second_order="diag_abs_h",
                clamp=True,
            )
            score_hvp = _score_parts_v2(
                records,
                e_parts=e_parts,
                h_e_parts=h_e_parts,
                linear_source=linear_source,
                second_order="hvp_offdiag",
                clamp=True,
            )
            score_parts_by_name = {
                "first_order_only": score_none,
                "diag_abs_h": score_diag,
                "hvp_offdiag": score_hvp,
            }
            rank_score_parts_by_name = {
                name: parts
                for name, parts in score_parts_by_name.items()
                if name == "hvp_offdiag"
                or (name == "first_order_only" and "none" in rank_second_filter)
                or name in rank_second_filter
            }
            cache_enabled = bool(int(os.environ.get("PROPOSAL_IDEAL_ORACLE_CACHE_MASKS", "1")))
            cache_max_local = int(os.environ.get(
                "PROPOSAL_IDEAL_ORACLE_CACHE_MAX_LOCAL_ELEMENTS",
                "800000000",
            ))
            selected_cache: Optional[Dict[Tuple[str, float], Tuple[List[np.ndarray], Dict[str, Any]]]]
            selected_cache = {} if cache_enabled and int(local_n) <= cache_max_local else None
            if selected_cache is not None:
                cache_score_names = set()
                if "none" in second_filter:
                    cache_score_names.add("first_order_only")
                if "diag_abs_h" in second_filter:
                    cache_score_names.add("diag_abs_h")
                if "hvp_offdiag" in second_filter:
                    cache_score_names.add("hvp_offdiag")
                for score_name, score_parts in score_parts_by_name.items():
                    if score_name not in cache_score_names:
                        continue
                    for k_ratio in k_ratios:
                        selected_cache[(score_name, float(k_ratio))] = _select_masks_global(
                            score_parts,
                            k_ratio=float(k_ratio),
                            total_n=int(total_n),
                            group=group,
                        )
            if not skip_rank and not skip_hvp:
                rank_rows.extend(_record_rank_metrics_v2(
                    records=records,
                    e_source=e_source,
                    linear_source=linear_source,
                    score_parts_by_name=rank_score_parts_by_name,
                    target_name="hvp_offdiag",
                    sample_plan=sample_plan,
                    k_ratios=k_ratios,
                    iteration=iteration,
                    rank=rank,
                    total_n=total_n,
                    proposal_config=proposal_config,
                    group=group,
                    selected_cache=selected_cache,
                ))
            for second_order, score_parts in (
                ("none", score_none),
                ("diag_abs_h", score_diag),
                ("hvp_offdiag", score_hvp),
            ):
                if second_order not in second_filter:
                    continue
                score_cache_name = "first_order_only" if second_order == "none" else second_order
                for k_ratio in k_ratios:
                    if selected_cache is not None:
                        selected_masks, select_info = selected_cache[(score_cache_name, float(k_ratio))]
                    else:
                        selected_masks, select_info = _select_masks_global(
                            score_parts,
                            k_ratio=float(k_ratio),
                            total_n=int(total_n),
                            group=group,
                        )
                    e_remaining_true = _remaining_true_parts(records, selected_masks)
                    _restore_theta(records)
                    _set_theta_minus_e(records, e_remaining_true)
                    loss_rb = _eval_loss_bank(actual_loss_bank)
                    actual_remaining = float(loss_rb) - float(loss_base)
                    actual_captured = float(actual_no_save) - float(actual_remaining)

                    first_remaining = _linear_loss_sum(
                        records,
                        e_parts=e_remaining_true,
                        linear_source=linear_source,
                        group=group,
                    )
                    diag_quad_remaining = _diag_abs_quad_sum(
                        records,
                        e_parts=e_remaining_true,
                        group=group,
                    )
                    diag_proxy_remaining = float(first_remaining) + float(diag_quad_remaining)
                    full_quad_remaining = None
                    full_proxy_remaining = None
                    third_order_full = None
                    do_exact_selected_hvp = (
                        exact_selected_hvp
                        and e_source in exact_hvp_e_filter
                        and linear_source in exact_hvp_linear_filter
                        and second_order in exact_hvp_second_filter
                        and float(k_ratio) in exact_hvp_k_filter
                        and (
                            not exact_hvp_policy_filter
                            or (
                                str(e_source),
                                str(linear_source),
                                str(second_order),
                            ) in exact_hvp_policy_filter
                        )
                    )
                    if do_exact_selected_hvp:
                        rms_remaining = _rms_for_parts(
                            e_remaining_true,
                            total_n=total_n,
                            group=group,
                        )
                        full_quad_remaining, _ = _hvp_quadratic_form_bank(
                            records=records,
                            e_parts=e_remaining_true,
                            rms_e=float(rms_remaining),
                            fd_eps=float(fd_eps),
                            loss_bank=hvp_loss_bank,
                            expert_params=expert_params,
                            grad_accum_views=grad_accum_views,
                            model=model,
                            sample_plan=None,
                            lossy_cp=lossy_cp,
                            return_h_e_parts=False,
                        )
                        full_proxy_remaining = float(first_remaining) + float(full_quad_remaining)
                        third_order_full = float(actual_remaining) - float(full_proxy_remaining)
                    row = {
                        "mode": "ideal_oracle_selection",
                        "step": int(iteration),
                        "rank": int(rank),
                        "world": int(world),
                        "selection_scope": select_info.get("selection_scope", "rank_local_exact_partition"),
                        "e_source_for_score": str(e_source),
                        "linear_source_for_score": str(linear_source),
                        "second_order_for_score": str(second_order),
                        "k_ratio": float(k_ratio),
                        "local_elements": int(local_n),
                        "global_elements": int(total_n),
                        "local_selected": int(select_info["local_selected"]),
                        "global_selected": int(select_info["global_selected"]),
                        "selected_ratio_global": float(select_info["global_selected"]) / float(max(1, total_n)),
                        "score_threshold": select_info["threshold"],
                        "loss_base": float(loss_base),
                        "loss_no_save_rollback": float(loss_no_save),
                        "actual_loss_no_save": float(actual_no_save),
                        "loss_after_selected_save": float(loss_rb),
                        "actual_loss_remaining": float(actual_remaining),
                        "actual_loss_captured": float(actual_captured),
                        "actual_capture_ratio_of_no_save": _safe_div(
                            float(actual_captured),
                            float(actual_no_save),
                        ),
                        "first_plus_diag_abs_h_loss_remaining": float(diag_proxy_remaining),
                        "actual_minus_first_plus_diag_abs_h": float(actual_remaining) - float(diag_proxy_remaining),
                        "exact_selected_hvp": bool(do_exact_selected_hvp),
                        "first_plus_full_hvp_loss_remaining": (
                            float(full_proxy_remaining) if full_proxy_remaining is not None else None
                        ),
                        "third_order_residual_vs_full_hvp": (
                            float(third_order_full) if third_order_full is not None else None
                        ),
                        "fd_eps": float(fd_eps),
                        "actual_eval_batches": int(len(actual_loss_bank)),
                        "hvp_batches": int(hvp_batches),
                    }
                    _append_jsonl(_log_path(proposal_config, "ideal_oracle_selection", rank), row)
                    term_rows.append(row)
                    _restore_theta(records)
            del selected_cache, score_parts_by_name, score_none, score_diag, score_hvp
            gc.collect()
    return term_rows, rank_rows


def _score_parts_v2(
    records: Sequence[_OracleTensor],
    *,
    e_parts: Sequence[np.ndarray],
    h_e_parts: Sequence[np.ndarray],
    linear_source: str,
    second_order: str,
    clamp: bool,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for rec, e, h_e in zip(records, e_parts, h_e_parts):
        lin = rec.m if linear_source == "m_t" else (
            rec.g if rec.g is not None else np.zeros_like(rec.e_full)
        )
        score = (-lin.astype(np.float32, copy=False) * e.astype(np.float32, copy=False)).astype(
            np.float32,
            copy=False,
        )
        if second_order == "diag_abs_h":
            score = score + np.float32(0.5) * rec.h_abs_lut * (e * e)
        elif second_order == "hvp_offdiag":
            score = score + np.float32(0.5) * e * h_e
        elif second_order != "none":
            raise ValueError(f"unknown second_order={second_order!r}")
        if clamp:
            score = np.maximum(score, np.float32(0.0))
        out.append(score.astype(np.float32, copy=False))
    return out


def _remaining_true_parts(
    records: Sequence[_OracleTensor],
    selected_masks: Sequence[np.ndarray],
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for rec, mask in zip(records, selected_masks):
        e = rec.e_full.copy()
        e[mask] = np.float32(0.0)
        out.append(e)
    return out


def _rms_for_parts(
    e_parts: Sequence[np.ndarray],
    *,
    total_n: int,
    group: Any,
) -> float:
    local_e_sq = float(sum(float(np.dot(e, e)) for e in e_parts))
    global_e_sq = _all_reduce_sum_float(local_e_sq, group=group)
    if global_e_sq <= 0.0:
        return 0.0
    return math.sqrt(float(global_e_sq) / float(max(1, int(total_n))))


def _linear_loss_sum(
    records: Sequence[_OracleTensor],
    *,
    e_parts: Sequence[np.ndarray],
    linear_source: str,
    group: Any,
) -> float:
    local = 0.0
    for rec, e in zip(records, e_parts):
        lin = rec.m if linear_source == "m_t" else (
            rec.g if rec.g is not None else np.zeros_like(rec.e_full)
        )
        local += -float(np.dot(lin, e))
    return _all_reduce_sum_float(local, group=group)


def _diag_abs_quad_sum(
    records: Sequence[_OracleTensor],
    *,
    e_parts: Sequence[np.ndarray],
    group: Any,
) -> float:
    local = 0.5 * sum(float(np.dot(rec.h_abs_lut, e * e)) for rec, e in zip(records, e_parts))
    return _all_reduce_sum_float(local, group=group)


def _select_masks_global(
    score_parts: Sequence[np.ndarray],
    *,
    k_ratio: float,
    total_n: int,
    group: Any,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    del total_n
    ratio = max(0.0, min(1.0, float(k_ratio)))
    local_n = int(sum(int(score.size) for score in score_parts))
    k_local = int(math.ceil(ratio * float(max(1, local_n))))
    if k_local <= 0 or local_n <= 0:
        masks = [np.zeros(score.shape, dtype=bool) for score in score_parts]
        return masks, {
            "threshold": None,
            "local_selected": 0,
            "global_selected": 0,
            "selection_scope": "rank_local_exact_partition",
        }
    if k_local >= local_n:
        masks = [np.ones(score.shape, dtype=bool) for score in score_parts]
        local_selected = int(local_n)
        global_selected = int(_all_reduce_sum_int(local_selected, group=group))
        return masks, {
            "threshold": None,
            "local_selected": int(local_selected),
            "global_selected": int(global_selected),
            "selection_scope": "rank_local_exact_partition",
        }

    score_views = [np.asarray(score, dtype=np.float32).reshape(-1) for score in score_parts]
    flat = np.concatenate(score_views) if len(score_views) > 1 else score_views[0].copy()
    np.nan_to_num(flat, nan=-np.inf, posinf=np.inf, neginf=-np.inf, copy=False)
    kth_index = int(local_n - k_local)
    flat.partition(kth_index)
    threshold = np.float32(flat[kth_index])

    greater_count = int(sum(int(np.count_nonzero(part > threshold)) for part in score_views))
    remaining_ties = max(0, int(k_local - greater_count))
    masks: List[np.ndarray] = []
    for score, score_view in zip(score_parts, score_views):
        mask_flat = score_view > threshold
        if remaining_ties > 0:
            tie_idx = np.flatnonzero(score_view == threshold)
            take = min(int(remaining_ties), int(tie_idx.size))
            if take > 0:
                mask_flat = mask_flat.copy()
                mask_flat[tie_idx[:take]] = True
                remaining_ties -= take
        masks.append(mask_flat.reshape(score.shape))
    local_selected = int(sum(int(mask.sum()) for mask in masks))
    global_selected = int(_all_reduce_sum_int(local_selected, group=group))
    return masks, {
        "threshold": float(threshold),
        "local_selected": int(local_selected),
        "global_selected": int(global_selected),
        "selection_scope": "rank_local_exact_partition",
    }


def _record_rank_metrics_v2(
    *,
    records: Sequence[_OracleTensor],
    e_source: str,
    linear_source: str,
    score_parts_by_name: Dict[str, Sequence[np.ndarray]],
    target_name: str,
    sample_plan: Dict[str, np.ndarray],
    k_ratios: Sequence[float],
    iteration: int,
    rank: int,
    total_n: int,
    proposal_config: Any,
    group: Any,
    selected_cache: Optional[Dict[Tuple[str, float], Tuple[List[np.ndarray], Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    target_parts = list(score_parts_by_name[target_name])
    target_sample = _sample_parts(target_parts, sample_plan)
    rows: List[Dict[str, Any]] = []
    for score_name, score_parts_seq in score_parts_by_name.items():
        if score_name == target_name:
            continue
        score_parts = list(score_parts_seq)
        score_sample = _sample_parts(score_parts, sample_plan)
        row: Dict[str, Any] = {
            "mode": "ideal_oracle_rank",
            "step": int(iteration),
            "rank": int(rank),
            "e_source_for_score": str(e_source),
            "linear_source_for_score": str(linear_source),
            "score": str(score_name),
            "target": str(target_name),
            "spearman_sample": _spearman(score_sample, target_sample),
            "pearson_sample": _pearson(score_sample, target_sample),
            "sample_elements": int(score_sample.size),
            "global_elements": int(total_n),
            "top_overlap_scope": "deterministic_sample_partition",
        }
        for ratio in k_ratios:
            ratio_key = float(ratio)
            score_mask, score_info = _sample_top_mask(score_sample, ratio_key)
            target_mask, target_info = _sample_top_mask(target_sample, ratio_key)
            intersection = int(np.count_nonzero(score_mask & target_mask))
            denom = max(1, int(target_info["selected"]))
            row[f"top_overlap@{float(ratio):.4f}"] = float(intersection) / float(denom)
            row[f"score_selected_sample@{float(ratio):.4f}"] = int(score_info["selected"])
            row[f"target_selected_sample@{float(ratio):.4f}"] = int(target_info["selected"])
        _append_jsonl(_log_path(proposal_config, "ideal_oracle_rank", rank), row)
        rows.append(row)
    return rows


def _score_sample_v2(
    records: Sequence[_OracleTensor],
    *,
    e_source: str,
    linear_source: str,
    second_order: str,
    sample_plan: Dict[str, np.ndarray],
    h_e_parts: Sequence[np.ndarray],
    clamp: bool,
) -> np.ndarray:
    e_attr = "e_agebit" if str(e_source) == "agebit" else "e_full"
    e = _sample_arrays(records, sample_plan, e_attr)
    if linear_source == "m_t":
        lin = _sample_arrays(records, sample_plan, "m")
    elif linear_source == "g":
        lin = _sample_arrays_with_optional_g(records, sample_plan)
    else:
        raise ValueError(f"unknown linear_source={linear_source!r}")

    score = (-lin * e).astype(np.float32, copy=False)
    if second_order == "diag_abs_h":
        h_abs = _sample_arrays(records, sample_plan, "h_abs_lut")
        score = score + np.float32(0.5) * h_abs * (e * e)
    elif second_order == "hvp_offdiag":
        h_e = _sample_parts(h_e_parts, sample_plan)
        score = score + np.float32(0.5) * e * h_e
    elif second_order != "none":
        raise ValueError(f"unknown second_order={second_order!r}")
    if clamp:
        score = np.maximum(score, np.float32(0.0))
    return score.astype(np.float32, copy=False)


def _record_pairwise_score_metrics_v2(
    *,
    records: Sequence[_OracleTensor],
    e_source: str,
    second_order: str,
    left_linear_source: str,
    right_linear_source: str,
    sample_plan: Dict[str, np.ndarray],
    k_ratios: Sequence[float],
    iteration: int,
    rank: int,
    total_n: int,
    proposal_config: Any,
    h_e_parts: Sequence[np.ndarray],
) -> List[Dict[str, Any]]:
    left = _score_sample_v2(
        records,
        e_source=e_source,
        linear_source=left_linear_source,
        second_order=second_order,
        sample_plan=sample_plan,
        h_e_parts=h_e_parts,
        clamp=True,
    )
    right = _score_sample_v2(
        records,
        e_source=e_source,
        linear_source=right_linear_source,
        second_order=second_order,
        sample_plan=sample_plan,
        h_e_parts=h_e_parts,
        clamp=True,
    )
    row: Dict[str, Any] = {
        "mode": "ideal_oracle_pairwise_rank",
        "step": int(iteration),
        "rank": int(rank),
        "e_source_for_score": str(e_source),
        "second_order_for_score": str(second_order),
        "left_linear_source": str(left_linear_source),
        "right_linear_source": str(right_linear_source),
        "left_score": f"{left_linear_source}+{second_order}",
        "right_score": f"{right_linear_source}+{second_order}",
        "spearman_sample": _spearman(left, right),
        "pearson_sample": _pearson(left, right),
        "sample_elements": int(left.size),
        "global_elements": int(total_n),
        "top_overlap_scope": "deterministic_sample_partition",
    }
    for ratio in k_ratios:
        ratio_key = float(ratio)
        left_mask, left_info = _sample_top_mask(left, ratio_key)
        right_mask, right_info = _sample_top_mask(right, ratio_key)
        intersection = int(np.count_nonzero(left_mask & right_mask))
        denom = max(1, int(right_info["selected"]))
        row[f"top_overlap@{ratio_key:.4f}"] = float(intersection) / float(denom)
        row[f"left_selected_sample@{ratio_key:.4f}"] = int(left_info["selected"])
        row[f"right_selected_sample@{ratio_key:.4f}"] = int(right_info["selected"])
    _append_jsonl(_log_path(proposal_config, "ideal_oracle_pairwise_rank", rank), row)
    return [row]


def _sample_top_mask(values: np.ndarray, ratio: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    n = int(arr.size)
    k = int(math.ceil(max(0.0, min(1.0, float(ratio))) * float(max(1, n))))
    if k <= 0 or n <= 0:
        return np.zeros(arr.shape, dtype=bool), {"selected": 0, "threshold": None}
    if k >= n:
        return np.ones(arr.shape, dtype=bool), {"selected": int(n), "threshold": None}

    clean = arr.copy()
    np.nan_to_num(clean, nan=-np.inf, posinf=np.inf, neginf=-np.inf, copy=False)
    kth_index = int(n - k)
    kth = clean.copy()
    kth.partition(kth_index)
    threshold = np.float32(kth[kth_index])

    mask = clean > threshold
    selected = int(np.count_nonzero(mask))
    remaining = int(k - selected)
    if remaining > 0:
        ties = np.flatnonzero(clean == threshold)
        take = min(remaining, int(ties.size))
        if take > 0:
            mask = mask.copy()
            mask[ties[:take]] = True
            selected += int(take)
    return mask, {"selected": int(selected), "threshold": float(threshold)}


def _age_code_ratio(records: Sequence[_OracleTensor], code: int) -> float:
    n = int(sum(rec.age_code.size for rec in records))
    if n <= 0:
        return 0.0
    count = sum(int(np.count_nonzero(rec.age_code == np.uint8(code))) for rec in records)
    return float(count) / float(n)


def _build_tensor_records(
    *,
    lossy_cp: Any,
    mapper: Any,
    names: Sequence[str],
    expert_params: Sequence[torch.nn.Parameter],
    signed_h_cpu: Sequence[np.ndarray],
) -> List[_OracleTensor]:
    expert_state = getattr(lossy_cp, "_expert_state", None)
    dense_buffers = getattr(expert_state, "_dense_buffers", None)
    if not isinstance(dense_buffers, dict):
        raise RuntimeError("ideal oracle requires test_dram expert dense buffers")
    tname_to_eid = getattr(lossy_cp, "_tname_to_eid", {}) or {}
    h_lut = getattr(lossy_cp, "_h_lookup_lut", None)
    global_start_by_name = _global_element_start_by_name(lossy_cp)
    age_words_cpu = _age_words_cpu(lossy_cp)
    lr_over_bc1 = float(getattr(lossy_cp, "_adam_lr_over_bc1_for_selection", 1.0e-4))
    sqrt_bc2_inv = float(getattr(lossy_cp, "_adam_sqrt_bc2_inv_for_selection", 1.0))
    eps_adam = float(getattr(lossy_cp, "_adam_eps_for_selection", 1.0e-8))
    beta1 = float(getattr(lossy_cp, "_adam_beta1_for_selection", 0.9))
    records: List[_OracleTensor] = []
    rank, _ = _rank_world()
    built_elements = 0
    empty_f32 = np.empty(0, dtype=np.float32)
    for name, param, h_signed_np in zip(names, expert_params, signed_h_cpu):
        eid = tname_to_eid.get(name)
        if eid is None:
            continue
        buffers = dense_buffers.get(eid)
        if not buffers or name not in buffers:
            continue
        cp_tensor = buffers[name]["p"].detach().reshape(-1).float().cpu()
        theta_tensor = mapper.get_param(param).detach().reshape(-1).float().cpu()
        m_state = mapper.get_state(param, "exp_avg")
        v_state = mapper.get_state(param, "exp_avg_sq")
        if m_state is None:
            m_tensor = torch.zeros_like(theta_tensor)
        else:
            m_tensor = m_state.detach().reshape(-1).float().cpu()
        if v_state is None:
            v_tensor = torch.zeros_like(theta_tensor)
        else:
            v_tensor = v_state.detach().reshape(-1).float().cpu()
        theta = theta_tensor.numpy().astype(np.float32, copy=False)
        checkpoint_view = cp_tensor.numpy().astype(np.float32, copy=False)
        m = m_tensor.numpy().astype(np.float32, copy=False)
        v = v_tensor.numpy().astype(np.float32, copy=False)
        h_signed = np.asarray(h_signed_np, dtype=np.float32).reshape(-1)
        if h_signed.shape[0] != theta.shape[0]:
            raise RuntimeError(f"signed-H size mismatch for {name}")
        h_abs_lut = _lookup_h_abs_lut_np(v, h_lut)
        global_start = int(global_start_by_name.get(str(name), 0))
        age_code = _unpack_age_codes_np(
            age_words_cpu,
            global_start=int(global_start),
            n=int(theta.shape[0]),
        )
        e_full = (theta - checkpoint_view).astype(np.float32, copy=False)
        e_agebit = _agebit_e_hat_np(
            m=m,
            v=v,
            h_abs=h_abs_lut,
            age_code=age_code,
            lr_over_bc1=lr_over_bc1,
            sqrt_bc2_inv=sqrt_bc2_inv,
            eps=eps_adam,
            beta1=beta1,
        )
        records.append(_OracleTensor(
            name=str(name),
            param=param,
            theta=theta,
            checkpoint=empty_f32,
            m=m,
            v=empty_f32,
            h_signed=empty_f32,
            h_abs_lut=h_abs_lut,
            e_full=e_full,
            e_agebit=e_agebit,
            age_code=age_code,
            global_start=global_start,
            score_abs_lut=empty_f32,
        ))
        built_elements += int(theta.shape[0])
        if len(records) == 1 or len(records) % 64 == 0:
            print(
                f"[proposal] ideal_oracle record build rank{rank}: "
                f"tensors={len(records)} elements={built_elements:,}",
                flush=True,
            )
        del cp_tensor, checkpoint_view, v_tensor, v, h_signed
    return records


def _global_element_start_by_name(lossy_cp: Any) -> Dict[str, int]:
    """Map tracked tensor name to the production age2bit global element offset."""

    out: Dict[str, int] = {}
    group_size = int(getattr(lossy_cp, "group_size", 1) or 1)
    for grp in getattr(lossy_cp, "_groups", ()) or ():
        names = list(grp.get("names", ()) or ())
        sizes = list(grp.get("sizes", ()) or ())
        master_base_elem = int(grp.get("master_off", 0)) * int(group_size)
        local_elem_off = 0
        for name, size in zip(names, sizes):
            n = int(size)
            if (n // max(1, group_size)) > 0:
                out[str(name)] = int(master_base_elem + local_elem_off)
            local_elem_off += n
    return out


def _age_words_cpu(lossy_cp: Any) -> Optional[np.ndarray]:
    state = getattr(lossy_cp, "_age2bit_score_state", None)
    words = getattr(state, "age_words", None)
    if words is None:
        return None
    return words.detach().cpu().numpy().astype(np.uint32, copy=True)


def _unpack_age_codes_np(
    words: Optional[np.ndarray],
    *,
    global_start: int,
    n: int,
) -> np.ndarray:
    if words is None or int(n) <= 0:
        return np.zeros(max(0, int(n)), dtype=np.uint8)
    idx = np.arange(int(global_start), int(global_start) + int(n), dtype=np.int64)
    word_idx = idx >> 4
    valid = word_idx < int(words.shape[0])
    out = np.zeros(int(n), dtype=np.uint8)
    if not bool(valid.any()):
        return out
    shift = ((idx[valid] & 15) << 1).astype(np.uint32, copy=False)
    out[valid] = ((words[word_idx[valid]] >> shift) & np.uint32(3)).astype(
        np.uint8,
        copy=False,
    )
    return out


def _agebit_e_hat_np(
    *,
    m: np.ndarray,
    v: np.ndarray,
    h_abs: np.ndarray,
    age_code: np.ndarray,
    lr_over_bc1: float,
    sqrt_bc2_inv: float,
    eps: float,
    beta1: float,
) -> np.ndarray:
    v_pos = np.maximum(v.astype(np.float32, copy=False), np.float32(0.0))
    denom = (
        np.sqrt(v_pos, dtype=np.float32) * np.float32(float(sqrt_bc2_inv))
        + np.float32(float(eps))
    )
    delta_signed = (
        -np.float32(float(lr_over_bc1)) * m.astype(np.float32, copy=False)
        / np.maximum(denom, np.float32(1.0e-30))
    )
    b = np.float32(min(max(float(beta1), 0.0), 0.999999))
    mu = (
        np.float32(abs(float(lr_over_bc1))) * h_abs.astype(np.float32, copy=False)
        / np.maximum(denom, np.float32(1.0e-30))
    )
    trace = np.float32(1.0) + b - (np.float32(1.0) - b) * mu
    disc = trace * trace - np.float32(4.0) * b
    sqrt_disc = np.sqrt(np.maximum(disc, np.float32(0.0)), dtype=np.float32)
    root0 = np.float32(0.5) * (trace + sqrt_disc)
    root1 = np.float32(0.5) * (trace - sqrt_disc)
    spectral = np.maximum(np.abs(root0), np.abs(root1))
    complex_rho = np.float32(math.sqrt(float(b)))
    rho = np.where(disc >= np.float32(0.0), spectral, complex_rho).astype(
        np.float32,
        copy=False,
    )
    rho = np.minimum(np.maximum(rho, np.float32(0.9)), np.float32(1.0))
    age_values = np.choose(
        age_code.astype(np.int64, copy=False),
        (
            np.float32(1.0),
            np.float32(4.0),
            np.float32(16.0),
            np.float32(64.0),
        ),
    ).astype(np.float32, copy=False)
    inv_rho = np.float32(1.0) / np.maximum(rho, np.float32(1.0e-30))
    denom_geom = inv_rho - np.float32(1.0)
    pow_term = np.exp(np.log(inv_rho.astype(np.float64)) * age_values.astype(np.float64))
    geom = ((pow_term - 1.0) / np.maximum(denom_geom.astype(np.float64), 1.0e-30)).astype(
        np.float32,
        copy=False,
    )
    geom = np.where(np.abs(denom_geom) < np.float32(1.0e-6), age_values, geom)
    return (delta_signed * geom.astype(np.float32, copy=False)).astype(
        np.float32,
        copy=False,
    )


def _lookup_h_abs_lut_np(v: np.ndarray, h_lut: Any) -> np.ndarray:
    if h_lut is None or getattr(h_lut, "values", None) is None:
        return np.ones_like(v, dtype=np.float32)
    values = h_lut.values.detach().float().cpu().numpy().astype(np.float32, copy=True)
    if values.size <= 1:
        return np.full_like(v, float(values[0]) if values.size else 1.0, dtype=np.float32)
    log_v_min = float(h_lut.log_v_min.detach().float().cpu().item())
    inv_width = float(h_lut.inv_log_v_width.detach().float().cpu().item())
    if not math.isfinite(inv_width) or inv_width <= 0.0:
        return np.full_like(v, float(values[0]), dtype=np.float32)
    pos = (np.log(np.maximum(v.astype(np.float64), 1.0e-30)) - log_v_min) * inv_width
    pos = np.clip(pos, 0.0, float(values.size - 1))
    idx_lo = np.floor(pos).astype(np.int64)
    idx_hi = np.minimum(idx_lo + 1, values.size - 1)
    frac = (pos - idx_lo.astype(np.float64)).astype(np.float32)
    out = values[idx_lo] + frac * (values[idx_hi] - values[idx_lo])
    return out.astype(np.float32, copy=False)


def _build_forward_loss(model: Any, data_iterator: Any) -> Optional[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]]]:
    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    m0 = model[0] if isinstance(model, (list, tuple)) else model
    try:
        from pretrain_gpt import get_batch as _get_batch_gpt
    except Exception as exc:
        print(f"[proposal] ideal_oracle: cannot import get_batch: {exc}", flush=True)
        return None
    try:
        batch = _get_batch_gpt(di)
    except Exception as exc:
        print(f"[proposal] ideal_oracle: get_batch failed: {exc}", flush=True)
        return None
    return _forward_loss_from_batch(m0, batch)


def _build_forward_loss_bank(
    *,
    model: Any,
    data_iterator: Any,
    n_batches: int,
) -> List[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]]:
    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    m0 = model[0] if isinstance(model, (list, tuple)) else model
    try:
        from pretrain_gpt import get_batch as _get_batch_gpt
    except Exception as exc:
        print(f"[proposal] ideal_oracle: cannot import get_batch: {exc}", flush=True)
        return []
    bank: List[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]] = []
    for batch_idx in range(max(1, int(n_batches))):
        try:
            batch = _get_batch_gpt(di)
        except Exception as exc:
            if not bank:
                print(f"[proposal] ideal_oracle: get_batch failed: {exc}", flush=True)
            break
        loss_fn = _forward_loss_from_batch(m0, batch)
        bank.append((loss_fn, _capture_rng_state()))
        if batch_idx == 0 or (batch_idx + 1) == int(n_batches):
            print(
                f"[proposal] ideal_oracle eval bank: built {batch_idx + 1}/{int(n_batches)} batches",
                flush=True,
            )
    return bank


def _forward_loss_from_batch(model_obj: Any, batch: Any) -> Callable[[], Tuple[torch.Tensor, Dict[str, Any]]]:
    tokens, labels, loss_mask, attention_mask, position_ids = batch[:5]
    try:
        from megatron import get_args
        from megatron.core import tensor_parallel
        from pretrain_gpt import loss_func as _pretrain_loss_func
    except Exception as exc:
        raise RuntimeError(f"ideal_oracle: cannot import training loss path: {exc}") from exc

    def _forward_loss() -> Tuple[torch.Tensor, Dict[str, Any]]:
        args = get_args()
        if bool(getattr(args, "mos", False)) or bool(getattr(args, "kd", False)):
            logits, other_losses = model_obj(tokens, position_ids, attention_mask)
            output_tensor = tensor_parallel.vocab_parallel_cross_entropy(
                logits.contiguous().float(),
                labels,
            )
            mos_loss = torch.zeros((), device=output_tensor.device, dtype=output_tensor.dtype)
        else:
            output_tensor, other_losses = model_obj(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
            )
            mos_loss = 0
        moe_losses = [
            loss for loss in (other_losses or ())
            if loss is not None
        ]
        if moe_losses:
            moe_loss = sum(moe_losses) * float(getattr(args, "moe_loss_coeff", 0.0))
        else:
            moe_loss = torch.zeros((), device=output_tensor.device, dtype=output_tensor.dtype)
        loss, meta = _pretrain_loss_func(loss_mask, moe_loss, mos_loss, output_tensor)
        return loss, dict(meta or {})

    return _forward_loss


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mpu_rng": None,
    }
    try:
        from megatron.core.tensor_parallel import get_cuda_rng_tracker
        state["mpu_rng"] = get_cuda_rng_tracker().get_states()
    except Exception:
        state["mpu_rng"] = None
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    try:
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if state.get("mpu_rng") is not None:
            try:
                from megatron.core.tensor_parallel import get_cuda_rng_tracker
                get_cuda_rng_tracker().set_states(state["mpu_rng"])
            except Exception:
                pass
    except Exception:
        pass


@torch.no_grad()
def _eval_loss(
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
    *,
    rng_state: Optional[Dict[str, Any]] = None,
) -> float:
    if rng_state is not None:
        _restore_rng_state(rng_state)
    loss, _ = forward_loss()
    if loss.dim() > 0:
        loss = loss.mean()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return float(loss.detach().float().item())


@torch.no_grad()
def _eval_loss_bank(
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
) -> float:
    if not loss_bank:
        return 0.0
    local_sum = 0.0
    for forward_loss, rng_state in loss_bank:
        local_sum += float(_eval_loss(forward_loss, rng_state=rng_state))
    local_mean = local_sum / float(max(1, len(loss_bank)))
    return _all_reduce_mean_float(local_mean)


def _run_backward_to_cpu_arrays(
    *,
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
    expert_params: Sequence[torch.nn.Parameter],
    grad_accum_views: Sequence[torch.Tensor],
    model: Any,
    rng_state: Optional[Dict[str, Any]] = None,
) -> List[np.ndarray]:
    if rng_state is not None:
        _restore_rng_state(rng_state)
    for view in grad_accum_views:
        view.zero_()
    module = model[0] if isinstance(model, (list, tuple)) else model
    try:
        for param in module.parameters():
            param.grad = None
    except Exception:
        for param in expert_params:
            param.grad = None
    handles = []
    out_parts = [
        np.zeros(int(param.numel()), dtype=np.float32)
        for param in expert_params
    ]
    for idx, param in enumerate(expert_params):
        def _hook(raw_grad, *, _idx=idx):
            out_parts[_idx] += (
                raw_grad.detach().reshape(-1).float().cpu().numpy()
            )
            return torch.zeros_like(raw_grad)
        handles.append(param.register_hook(_hook))
    try:
        loss, _ = forward_loss()
        if loss.dim() > 0:
            loss = loss.mean()
        loss.backward(retain_graph=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        try:
            for param in module.parameters():
                param.grad = None
        except Exception:
            for param in expert_params:
                param.grad = None
        for view in grad_accum_views:
            view.zero_()
    return out_parts


def _run_backward_bank_to_cpu_arrays(
    *,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    expert_params: Sequence[torch.nn.Parameter],
    grad_accum_views: Sequence[torch.Tensor],
    model: Any,
) -> List[np.ndarray]:
    if not loss_bank:
        return [
            np.zeros(int(param.numel()), dtype=np.float32)
            for param in expert_params
        ]
    acc_parts: Optional[List[np.ndarray]] = None
    for forward_loss, rng_state in loss_bank:
        parts = _run_backward_to_cpu_arrays(
            forward_loss=forward_loss,
            expert_params=expert_params,
            grad_accum_views=grad_accum_views,
            model=model,
            rng_state=rng_state,
        )
        if acc_parts is None:
            acc_parts = [part.astype(np.float32, copy=True) for part in parts]
        else:
            for acc, part in zip(acc_parts, parts):
                acc += part.astype(np.float32, copy=False)
        del parts
        gc.collect()
    assert acc_parts is not None
    inv = np.float32(1.0 / float(max(1, len(loss_bank))))
    for acc in acc_parts:
        acc *= inv
    return acc_parts


@torch.no_grad()
def _restore_theta(records: Sequence[_OracleTensor]) -> None:
    for rec in records:
        src = torch.from_numpy(rec.theta).view(rec.param.shape)
        rec.param.data.copy_(
            src.to(device=rec.param.device, dtype=rec.param.dtype),
            non_blocking=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _set_theta_minus_e(records: Sequence[_OracleTensor], e_parts: Sequence[np.ndarray]) -> None:
    for rec, e in zip(records, e_parts):
        value = rec.theta - e
        src = torch.from_numpy(value.astype(np.float32, copy=False)).view(rec.param.shape)
        rec.param.data.copy_(
            src.to(device=rec.param.device, dtype=rec.param.dtype),
            non_blocking=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _set_theta_plus_direction(
    records: Sequence[_OracleTensor],
    e_parts: Sequence[np.ndarray],
    *,
    scale: float,
) -> None:
    for rec, e in zip(records, e_parts):
        value = rec.theta + float(scale) * e
        src = torch.from_numpy(value.astype(np.float32, copy=False)).view(rec.param.shape)
        rec.param.data.copy_(
            src.to(device=rec.param.device, dtype=rec.param.dtype),
            non_blocking=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_e_for_local_budget(
    records: Sequence[_OracleTensor],
    k_ratio: float,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    return _make_e_for_score_parts(
        records,
        [rec.score_abs_lut for rec in records],
        k_ratio,
    )


def _make_e_for_score_parts(
    records: Sequence[_OracleTensor],
    score_parts: Sequence[np.ndarray],
    k_ratio: float,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    n_total = int(sum(rec.e_full.size for rec in records))
    k = int(math.ceil(max(0.0, min(1.0, float(k_ratio))) * float(n_total)))
    if k <= 0:
        return [rec.e_full.copy() for rec in records], {
            "selected_count": 0,
            "threshold": None,
        }
    k = min(k, n_total)
    scores = np.concatenate([score.reshape(-1) for score in score_parts])
    finite = np.where(np.isfinite(scores), scores, -np.inf)
    if k >= n_total:
        threshold = float(np.min(finite))
    else:
        threshold = float(np.partition(finite, n_total - k)[n_total - k])
    e_parts: List[np.ndarray] = []
    selected = 0
    for rec, score in zip(records, score_parts):
        mask = score >= np.float32(threshold)
        e = rec.e_full.copy()
        e[mask] = np.float32(0.0)
        selected += int(mask.sum())
        e_parts.append(e)
    return e_parts, {
        "selected_count": int(selected),
        "threshold": float(threshold),
    }


def _score_parts_for_policy(
    records: Sequence[_OracleTensor],
    *,
    policy: str,
    seed: int,
) -> List[np.ndarray]:
    if policy == "diag_abs_lut":
        return [rec.score_abs_lut for rec in records]
    if policy == "diag_signed_fd":
        return [
            np.maximum(
                -rec.m * rec.e_full
                + np.float32(0.5) * rec.h_signed * (rec.e_full * rec.e_full),
                np.float32(0.0),
            ).astype(np.float32, copy=False)
            for rec in records
        ]
    if policy == "first_m":
        return [
            np.maximum(-rec.m * rec.e_full, np.float32(0.0)).astype(np.float32, copy=False)
            for rec in records
        ]
    if policy == "random":
        rng = np.random.default_rng(int(seed))
        return [
            rng.random(rec.e_full.shape[0]).astype(np.float32, copy=False)
            for rec in records
        ]
    raise ValueError(f"unknown ideal oracle loss-sweep policy: {policy}")


def _record_loss_sweep(
    *,
    records: Sequence[_OracleTensor],
    k_ratio: float,
    iteration: int,
    rank: int,
    total_n: int,
    loss_base: float,
    diag_abs_lut_e_parts: Sequence[np.ndarray],
    diag_abs_lut_select_info: Dict[str, Any],
    diag_abs_lut_loss: float,
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
    paired_rng_state: Dict[str, Any],
    proposal_config: Any,
    lossy_cp: Any,
) -> None:
    for policy in ("diag_abs_lut", "diag_signed_fd", "first_m", "random"):
        if policy == "diag_abs_lut":
            e_parts = list(diag_abs_lut_e_parts)
            select_info = dict(diag_abs_lut_select_info)
            loss_rb_global = float(diag_abs_lut_loss)
        else:
            scores = _score_parts_for_policy(
                records,
                policy=policy,
                seed=0x51EED + int(iteration) * 9973 + int(round(float(k_ratio) * 1.0e6)),
            )
            e_parts, select_info = _make_e_for_score_parts(records, scores, float(k_ratio))
            _restore_theta(records)
            _set_theta_minus_e(records, e_parts)
            loss_rb_global = _all_reduce_mean_float(
                _eval_loss(forward_loss, rng_state=paired_rng_state)
            )
        selected_global = int(
            _all_reduce_sum_int(
                int(select_info["selected_count"]),
                group=getattr(lossy_cp, "_proposal_cpu_pg", None),
            )
        )
        row = {
            "mode": "ideal_oracle_loss_sweep",
            "step": int(iteration),
            "rank": int(rank),
            "selection_scope": "rank_local_ratio",
            "policy": str(policy),
            "k_ratio": float(k_ratio),
            "global_elements": int(total_n),
            "global_selected": int(selected_global),
            "selected_ratio_global": float(selected_global) / float(max(1, total_n)),
            "threshold": select_info["threshold"],
            "loss_base": float(loss_base),
            "loss_rollback": float(loss_rb_global),
            "R_actual": float(loss_rb_global) - float(loss_base),
        }
        _append_jsonl(_log_path(proposal_config, "ideal_oracle_loss_sweep", rank), row)
    _restore_theta(records)


def _hvp_quadratic_form(
    *,
    records: Sequence[_OracleTensor],
    e_parts: Sequence[np.ndarray],
    rms_e: float,
    fd_eps: float,
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict[str, Any]]],
    expert_params: Sequence[torch.nn.Parameter],
    grad_accum_views: Sequence[torch.Tensor],
    model: Any,
    rng_state: Optional[Dict[str, Any]],
    sample_plan: Optional[Dict[str, np.ndarray]],
    lossy_cp: Any,
    return_h_e_parts: bool = False,
) -> Tuple[float, Optional[Dict[str, np.ndarray]]]:
    if rms_e <= 0.0:
        return 0.0, None
    inv_rms = 1.0 / float(rms_e)
    sample_out: Optional[Dict[str, np.ndarray]] = None

    _restore_theta(records)
    _set_theta_plus_direction(records, e_parts, scale=float(fd_eps) * inv_rms)
    g_plus_parts = _run_backward_to_cpu_arrays(
        forward_loss=forward_loss,
        expert_params=expert_params,
        grad_accum_views=grad_accum_views,
        model=model,
        rng_state=rng_state,
    )
    dot_plus = _dot_parts_with_e(g_plus_parts, e_parts)
    if sample_plan is not None:
        sample_out = {"g_plus": _sample_parts(g_plus_parts, sample_plan)}

    _restore_theta(records)
    _set_theta_plus_direction(records, e_parts, scale=-float(fd_eps) * inv_rms)
    g_minus_parts = _run_backward_to_cpu_arrays(
        forward_loss=forward_loss,
        expert_params=expert_params,
        grad_accum_views=grad_accum_views,
        model=model,
        rng_state=rng_state,
    )
    dot_minus = _dot_parts_with_e(g_minus_parts, e_parts)
    if sample_out is not None:
        sample_out["g_minus"] = _sample_parts(g_minus_parts, sample_plan)
        sample_out["h_e"] = (
            float(rms_e)
            * (sample_out["g_plus"] - sample_out["g_minus"])
            / np.float32(2.0 * float(fd_eps))
        ).astype(np.float32, copy=False)
    if return_h_e_parts:
        if sample_out is None:
            sample_out = {}
        sample_out["h_e_parts"] = [
            (
                float(rms_e)
                * (gp.astype(np.float32, copy=False) - gm.astype(np.float32, copy=False))
                / np.float32(2.0 * float(fd_eps))
            ).astype(np.float32, copy=False)
            for gp, gm in zip(g_plus_parts, g_minus_parts)
        ]

    local_e_h_e = float(rms_e) * (float(dot_plus) - float(dot_minus)) / (2.0 * float(fd_eps))
    global_e_h_e = _all_reduce_sum_float(
        local_e_h_e,
        group=getattr(lossy_cp, "_proposal_cpu_pg", None),
    )
    _restore_theta(records)
    return 0.5 * float(global_e_h_e), sample_out


def _hvp_quadratic_form_bank(
    *,
    records: Sequence[_OracleTensor],
    e_parts: Sequence[np.ndarray],
    rms_e: float,
    fd_eps: float,
    loss_bank: Sequence[Tuple[Callable[[], Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]],
    expert_params: Sequence[torch.nn.Parameter],
    grad_accum_views: Sequence[torch.Tensor],
    model: Any,
    sample_plan: Optional[Dict[str, np.ndarray]],
    lossy_cp: Any,
    return_h_e_parts: bool = False,
) -> Tuple[float, Optional[Dict[str, np.ndarray]]]:
    if not loss_bank:
        return 0.0, None
    r2_sum = 0.0
    h_e_acc: Optional[List[np.ndarray]] = None
    sample_acc: Optional[Dict[str, np.ndarray]] = None
    for forward_loss, rng_state in loss_bank:
        r2, payload = _hvp_quadratic_form(
            records=records,
            e_parts=e_parts,
            rms_e=float(rms_e),
            fd_eps=float(fd_eps),
            forward_loss=forward_loss,
            expert_params=expert_params,
            grad_accum_views=grad_accum_views,
            model=model,
            rng_state=rng_state,
            sample_plan=sample_plan,
            lossy_cp=lossy_cp,
            return_h_e_parts=return_h_e_parts,
        )
        r2_sum += float(r2)
        if payload is not None:
            if return_h_e_parts and "h_e_parts" in payload:
                parts = list(payload["h_e_parts"])  # type: ignore[arg-type]
                if h_e_acc is None:
                    h_e_acc = [part.astype(np.float32, copy=True) for part in parts]
                else:
                    for acc, part in zip(h_e_acc, parts):
                        acc += part.astype(np.float32, copy=False)
            if sample_plan is not None and "h_e" in payload:
                if sample_acc is None:
                    sample_acc = {"h_e": payload["h_e"].astype(np.float32, copy=True)}
                else:
                    sample_acc["h_e"] += payload["h_e"].astype(np.float32, copy=False)
    inv = np.float32(1.0 / float(max(1, len(loss_bank))))
    out_payload: Optional[Dict[str, np.ndarray]] = None
    if h_e_acc is not None:
        for acc in h_e_acc:
            acc *= inv
        out_payload = {"h_e_parts": h_e_acc}  # type: ignore[dict-item]
    if sample_acc is not None:
        sample_acc["h_e"] *= inv
        if out_payload is None:
            out_payload = sample_acc
        else:
            out_payload.update(sample_acc)
    return float(r2_sum) / float(max(1, len(loss_bank))), out_payload


def _dot_parts_with_e(parts: Sequence[np.ndarray], e_parts: Sequence[np.ndarray]) -> float:
    total = 0.0
    for g, e in zip(parts, e_parts):
        total += float(np.dot(g, e))
    return total


def _make_sample_plan(
    records: Sequence[_OracleTensor],
    *,
    max_samples: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    total = int(sum(rec.e_full.size for rec in records))
    if max_samples <= 0 or total <= 0:
        return {"record_idx": np.empty(0, dtype=np.int64), "local_idx": np.empty(0, dtype=np.int64)}
    n = min(int(max_samples), int(total))
    rng = np.random.default_rng(int(seed))
    flat_idx = np.sort(rng.choice(total, size=n, replace=False).astype(np.int64))
    offsets = np.cumsum([0] + [int(rec.e_full.size) for rec in records]).astype(np.int64)
    record_idx = np.searchsorted(offsets[1:], flat_idx, side="right").astype(np.int64)
    local_idx = (flat_idx - offsets[record_idx]).astype(np.int64)
    return {"record_idx": record_idx, "local_idx": local_idx}


def _sample_arrays(
    records: Sequence[_OracleTensor],
    sample_plan: Dict[str, np.ndarray],
    attr: str,
) -> np.ndarray:
    record_idx = sample_plan["record_idx"]
    local_idx = sample_plan["local_idx"]
    out = np.empty(record_idx.shape[0], dtype=np.float32)
    cursor = 0
    for ridx in np.unique(record_idx):
        mask = record_idx == ridx
        vals = getattr(records[int(ridx)], attr)
        cnt = int(mask.sum())
        out[cursor:cursor + cnt] = vals[local_idx[mask]]
        cursor += cnt
    return out


def _sample_parts(
    parts: Sequence[np.ndarray],
    sample_plan: Dict[str, np.ndarray],
) -> np.ndarray:
    record_idx = sample_plan["record_idx"]
    local_idx = sample_plan["local_idx"]
    out = np.empty(record_idx.shape[0], dtype=np.float32)
    cursor = 0
    for ridx in np.unique(record_idx):
        mask = record_idx == ridx
        vals = parts[int(ridx)]
        cnt = int(mask.sum())
        out[cursor:cursor + cnt] = vals[local_idx[mask]]
        cursor += cnt
    return out


def _record_rank_metrics(
    *,
    records: Sequence[_OracleTensor],
    sample_plan: Dict[str, np.ndarray],
    hvp_sample: Dict[str, np.ndarray],
    iteration: int,
    rank: int,
    proposal_config: Any,
    k_ratios: Sequence[float],
) -> List[Dict[str, Any]]:
    if sample_plan["record_idx"].size <= 0:
        return []
    e = _sample_arrays(records, sample_plan, "e_full")
    m = _sample_arrays(records, sample_plan, "m")
    g = _sample_arrays_with_optional_g(records, sample_plan)
    h_signed = _sample_arrays(records, sample_plan, "h_signed")
    h_abs_lut = _sample_arrays(records, sample_plan, "h_abs_lut")
    h_e = hvp_sample["h_e"]
    diag_signed = -m * e + np.float32(0.5) * h_signed * (e * e)
    diag_abs_lut = np.maximum(
        -m * e + np.float32(0.5) * h_abs_lut * (e * e),
        np.float32(0.0),
    )
    first_m = -m * e
    first_g = -g * e
    full_directional = -g * e + np.float32(0.5) * e * h_e
    full_marginal = -g * e + e * h_e - np.float32(0.5) * h_signed * (e * e)
    scores = {
        "diag_signed_fd": diag_signed,
        "diag_abs_lut": diag_abs_lut,
        "first_m": first_m,
        "first_g": first_g,
        "full_directional_hvp": full_directional,
        "full_marginal_hvp": full_marginal,
    }
    rows: List[Dict[str, Any]] = []
    for target_name in ("full_directional_hvp", "full_marginal_hvp"):
        target = scores[target_name]
        for score_name, score in scores.items():
            if score_name == target_name:
                continue
            row: Dict[str, Any] = {
                "mode": "ideal_oracle_rank",
                "step": int(iteration),
                "rank": int(rank),
                "sample_elements": int(e.size),
                "target": target_name,
                "score": score_name,
                "spearman": _spearman(score, target),
                "pearson": _pearson(score, target),
            }
            for ratio in k_ratios:
                row[f"top_overlap@{ratio:.4f}"] = _top_overlap(score, target, float(ratio))
            _append_jsonl(_log_path(proposal_config, "ideal_oracle_rank", rank), row)
            rows.append(row)
    return rows


def _sample_arrays_with_optional_g(
    records: Sequence[_OracleTensor],
    sample_plan: Dict[str, np.ndarray],
) -> np.ndarray:
    record_idx = sample_plan["record_idx"]
    local_idx = sample_plan["local_idx"]
    out = np.empty(record_idx.shape[0], dtype=np.float32)
    cursor = 0
    for ridx in np.unique(record_idx):
        mask = record_idx == ridx
        vals = records[int(ridx)].g
        if vals is None:
            vals = np.zeros_like(records[int(ridx)].e_full)
        cnt = int(mask.sum())
        out[cursor:cursor + cnt] = vals[local_idx[mask]]
        cursor += cnt
    return out


def _measure_loss_noise(
    *,
    model: Any,
    data_iterator: Any,
    proposal_config: Any,
    records: Sequence[_OracleTensor],
    iteration: int,
    base_loss: float,
    rank: int,
) -> Dict[str, Any]:
    n_batches = int(os.environ.get("PROPOSAL_IDEAL_ORACLE_NOISE_BATCHES", "4") or "4")
    losses = [float(base_loss)]
    if n_batches <= 1:
        return {}
    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    m0 = model[0] if isinstance(model, (list, tuple)) else model
    try:
        from pretrain_gpt import get_batch as _get_batch_gpt
    except Exception:
        return {}
    _restore_theta(records)
    for _ in range(n_batches - 1):
        try:
            batch = _get_batch_gpt(di)
            loss_fn = _forward_loss_from_batch(m0, batch)
            losses.append(_all_reduce_mean_float(_eval_loss(loss_fn)))
        except Exception:
            break
    if len(losses) <= 1:
        return {}
    arr = np.asarray(losses, dtype=np.float64)
    rec = {
        "mode": "ideal_oracle_noise",
        "step": int(iteration),
        "rank": int(rank),
        "loss_count": int(arr.size),
        "loss_mean": float(arr.mean()),
        "loss_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "loss_min": float(arr.min()),
        "loss_max": float(arr.max()),
    }
    _append_jsonl(_log_path(proposal_config, "ideal_oracle_noise", rank), rec)
    return rec


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return 0.0
    return _pearson(_rankdata(a[mask]), _rankdata(b[mask]))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return 0.0
    x = a[mask].astype(np.float64, copy=False)
    y = b[mask].astype(np.float64, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return float(np.dot(x, y) / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _top_overlap(score: np.ndarray, target: np.ndarray, ratio: float) -> float:
    n = int(score.size)
    if n <= 0:
        return 0.0
    k = max(1, min(n, int(math.ceil(max(0.0, min(1.0, ratio)) * float(n)))))
    a = np.argpartition(np.where(np.isfinite(score), score, -np.inf), n - k)[n - k:]
    b = np.argpartition(np.where(np.isfinite(target), target, -np.inf), n - k)[n - k:]
    return float(np.intersect1d(a, b, assume_unique=False).size) / float(k)


def _safe_div(num: float, den: float) -> float:
    den_f = float(den)
    if den_f == 0.0 or not math.isfinite(den_f):
        return 0.0
    return float(num) / den_f


def _env_csv_tuple(name: str, default: Sequence[str]) -> Tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return tuple(str(x) for x in default)
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def _parse_policy_combo_filter(raw: str) -> set[Tuple[str, str, str]]:
    """Parse e_source:linear_source:second_order filters."""

    out: set[Tuple[str, str, str]] = set()
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        bits = tuple(x.strip() for x in part.split(":"))
        if len(bits) == 3 and all(bits):
            out.add((bits[0], bits[1], bits[2]))
    return out


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return None
    return obj


def _log_path(proposal_config: Any, stem: str, rank: int) -> Path:
    log_dir = getattr(proposal_config, "ideal_oracle_log_dir", None)
    if log_dir is None:
        log_dir = Path(getattr(proposal_config, "cp_dir", Path("./proposal_cp"))) / "ideal_oracle"
    return Path(log_dir) / f"{stem}_rank{int(rank):03d}.jsonl"


def _rank_world() -> Tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
    return 0, 1


def _all_reduce_sum_float(value: float, group: Any = None) -> float:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return float(value)
    if group is not None:
        tensor = torch.tensor([float(value)], dtype=torch.float64, device="cpu")
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)
        return float(tensor.item())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item())


def _all_reduce_extreme_float(value: float, *, op: str, group: Any = None) -> float:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return float(value)
    reduce_op = (
        torch.distributed.ReduceOp.MAX
        if str(op) == "max"
        else torch.distributed.ReduceOp.MIN
    )
    if group is not None:
        tensor = torch.tensor([float(value)], dtype=torch.float64, device="cpu")
        torch.distributed.all_reduce(tensor, op=reduce_op, group=group)
        return float(tensor.item())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=reduce_op)
    return float(tensor.item())


def _all_reduce_sum_int(value: int, group: Any = None) -> int:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return int(value)
    if group is not None:
        tensor = torch.tensor([int(value)], dtype=torch.int64, device="cpu")
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)
        return int(tensor.item())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([int(value)], dtype=torch.int64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return int(tensor.item())


def _all_reduce_mean_float(value: float) -> float:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return float(value)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([float(value), 1.0], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    denom = max(float(tensor[1].item()), 1.0)
    return float(tensor[0].item()) / denom


def _is_pipeline_engine(obj: Any) -> bool:
    return obj.__class__.__name__ == "PipelineEngine"
