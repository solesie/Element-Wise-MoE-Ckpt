"""CPU/Gloo helpers for tiny multi-rank proposal aggregates."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch


def valid_positive_threshold(tau: float) -> bool:
    return bool(math.isfinite(float(tau)) and float(tau) > 0.0)


def sync_candidate_ladder_cpu(
    values: List[float],
    *,
    tau_center: float,
    scratch: Optional[torch.Tensor],
    capacity: int,
    cpu_process_group: Any,
    world_size: int,
    src_rank: int,
) -> Tuple[List[float], torch.Tensor]:
    """Broadcast rank-0 candidate taus so all ranks reduce matching columns."""

    cap = max(1, min(int(capacity), 16))
    if int(world_size) <= 1:
        if scratch is None or int(scratch.numel()) != cap + 1:
            scratch = torch.empty(cap + 1, dtype=torch.float64)
        return values, scratch
    if (
        cpu_process_group is None
        or not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        raise RuntimeError("multi-rank cutoff candidates require CPU/Gloo group")
    if scratch is None or int(scratch.numel()) != cap + 1:
        scratch = torch.empty(cap + 1, dtype=torch.float64)
    n = min(len(values), cap)
    if n <= 0:
        n = 1
        local_values = [float(max(tau_center, 1.0e-12))]
    else:
        local_values = [float(v) for v in values[:n]]
    fill = float(max(tau_center, 1.0e-12))
    scratch[0] = float(n)
    for i in range(cap):
        scratch[i + 1] = local_values[i] if i < n else fill
    torch.distributed.broadcast(
        scratch,
        src=int(src_rank),
        group=cpu_process_group,
    )
    synced_n = max(1, min(cap, int(round(float(scratch[0].item())))))
    return [float(scratch[i + 1].item()) for i in range(synced_n)], scratch


def choose_global_probe_decision_cpu(
    *,
    stats: torch.Tensor,
    candidates: List[float],
    n_candidates: int,
    target_ratio: float,
    cutoff_tolerance: float,
    sample_blocks_min: int,
    sample_blocks_current: int,
    sample_blocks_cap: int,
    ci_shortfall_streak: int,
    sample_grow_hold: int,
    ci_shortfall_grow_after: int = 3,
    run_if_decision_code: int = -1,
) -> Dict[str, Any]:
    """Choose tau from globally reduced candidate stats.

    ``stats`` rows are ratio, variance, missing mass, and sampled block count.
    The objective mirrors the single-rank device decision: minimize
    ``abs(ratio-target) + 1.96*sqrt(var) + missing_mass`` and use a log-linear
    bracket interpolation when two valid candidates straddle the target.
    """

    n_current = min(int(n_candidates), len(candidates), int(stats.shape[1]))
    target = max(float(target_ratio), 1.0e-12)
    allowed = target * max(float(cutoff_tolerance), 1.0e-12)

    best_tau = 0.0
    best_ratio = 0.0
    best_se = float("inf")
    best_bound = float("inf")
    best_valid = False
    best_idx = -1
    sampled_blocks = 0.0

    bracket_tau = 0.0
    bracket_score = float("inf")
    bracket_found = False
    bracket_se = float("inf")
    bracket_width = 0.0

    rows = stats[:, :n_current]
    for i in range(n_current):
        tau_i = float(candidates[i])
        ratio_i = float(rows[0, i].item())
        var_i = float(rows[1, i].item())
        missing_i = float(rows[2, i].item())
        if i == 0:
            sampled_blocks = float(rows[3, i].item())
        se_i = 1.96 * math.sqrt(max(var_i, 0.0)) + missing_i
        bound_i = abs(ratio_i - target) + se_i
        valid_i = (
            tau_i > 0.0
            and math.isfinite(tau_i)
            and ratio_i > 0.0
            and math.isfinite(ratio_i)
            and math.isfinite(se_i)
        )
        if valid_i and bound_i < best_bound:
            best_tau = tau_i
            best_ratio = ratio_i
            best_se = se_i
            best_bound = bound_i
            best_valid = True
            best_idx = int(i)

        if not valid_i:
            continue
        for j in range(i + 1, n_current):
            tau_j = float(candidates[j])
            ratio_j = float(rows[0, j].item())
            var_j = float(rows[1, j].item())
            missing_j = float(rows[2, j].item())
            se_j = 1.96 * math.sqrt(max(var_j, 0.0)) + missing_j
            valid_j = (
                tau_j > 0.0
                and math.isfinite(tau_j)
                and ratio_j > 0.0
                and math.isfinite(ratio_j)
                and math.isfinite(se_j)
            )
            if not valid_j:
                continue
            if (ratio_i - target) * (ratio_j - target) > 0.0:
                continue
            x_i = math.log(tau_i)
            x_j = math.log(tau_j)
            y_i = math.log(max(ratio_i, 1.0e-12))
            y_j = math.log(max(ratio_j, 1.0e-12))
            if abs(x_j - x_i) <= 1.0e-12 or abs(y_j - y_i) <= 1.0e-12:
                continue
            slope = (y_j - y_i) / (x_j - x_i)
            if slope >= 0.0:
                continue
            pred_x = x_i + (math.log(target) - y_i) / slope
            x_lo = min(x_i, x_j)
            x_hi = max(x_i, x_j)
            if pred_x < x_lo or pred_x > x_hi:
                continue
            pair_se = max(se_i, se_j)
            pair_score = pair_se + (x_hi - x_lo) * target * 1.0e-3
            if pair_score < bracket_score:
                bracket_tau = math.exp(pred_x)
                bracket_score = pair_score
                bracket_se = pair_se
                bracket_width = x_hi - x_lo
                bracket_found = True

    best_accept = bool(best_valid and best_bound <= allowed)
    first_pass = int(run_if_decision_code) < 0
    use_bracket_for_reprobe = bool(first_pass and bracket_found and not best_accept)
    use_bracket_final = bool((not first_pass) and bracket_found and not best_accept)
    use_bracket_tau = bool(use_bracket_for_reprobe or use_bracket_final)
    accept = bool(bracket_found or best_valid)
    decision_code = (
        2 if best_accept
        else 3 if (use_bracket_for_reprobe or (first_pass and best_valid))
        else 4 if best_valid
        else 0
    )
    cur = max(int(sample_blocks_current), int(sample_blocks_min))
    shrunk = max((cur * 7) // 8, int(sample_blocks_min))
    grow_after = max(int(ci_shortfall_grow_after), 1)
    ci_shortfall = bool(decision_code == 4 and not best_accept and not use_bracket_final)
    prev_streak = max(int(ci_shortfall_streak), 0)
    next_streak = min(prev_streak + 1, grow_after)
    grow = bool(ci_shortfall and next_streak >= grow_after)
    grown = min(cur * 2, max(int(sample_blocks_cap), int(sample_blocks_min)))
    chosen_se = float(bracket_se if use_bracket_tau else best_se)
    hold_growth = int(sample_grow_hold) != 0
    if grow:
        next_sample = int(grown)
    elif best_accept and chosen_se < allowed * 0.25 and not hold_growth:
        next_sample = int(shrunk)
    else:
        next_sample = int(cur)
    if best_accept or grow:
        stored_streak = 0
    elif ci_shortfall:
        stored_streak = int(next_streak)
    else:
        stored_streak = int(prev_streak)
    return {
        "chosen_tau": float(bracket_tau if use_bracket_tau else best_tau),
        "chosen_ratio": float(target if use_bracket_tau else best_ratio),
        "chosen_se": float(chosen_se),
        "best_bound": float(best_bound),
        "next_sample": int(next_sample),
        "strict_accept": bool(best_accept),
        "sampled_blocks": float(sampled_blocks),
        "accept": bool(accept),
        "bracket_width": float(bracket_width if use_bracket_for_reprobe else 0.0),
        "decision": int(decision_code),
        "best_idx": int(best_idx),
        "needs_retry": bool(decision_code == 3),
        "stored_streak": int(stored_streak),
        "grow_hold_value": int(1 if grow else 0),
    }


def retry_candidate_ladder_cpu(
    decision: Dict[str, Any],
    *,
    source_candidates: List[float],
    capacity: int,
    target_ratio: float,
) -> List[float]:
    """Build the second probe ladder from a reduced global decision."""

    n_candidates = max(1, int(capacity))
    target = max(float(target_ratio), 1.0e-12)
    best_tau = float(decision.get("chosen_tau", 0.0))
    best_ratio = float(decision.get("chosen_ratio", 0.0))
    if not valid_positive_threshold(best_tau) or best_ratio <= 0.0:
        return [float(max(best_tau, 1.0e-12))]

    log_best = math.log(best_tau)
    min_spacing = float("inf")
    for tau in source_candidates:
        tau_f = float(tau)
        if not valid_positive_threshold(tau_f):
            continue
        spacing = abs(math.log(tau_f) - log_best)
        if spacing > 1.0e-7:
            min_spacing = min(min_spacing, spacing)

    ratio_step = abs(math.log(max(best_ratio, 1.0e-12) / target))
    has_spacing = bool(min_spacing < float("inf"))
    bracket_width = float(decision.get("bracket_width", 0.0))
    centered = ratio_step < 1.0e-6
    has_bracket_width = bool(
        centered and math.isfinite(bracket_width) and bracket_width > 1.0e-7
    )
    step = max(ratio_step, min_spacing * 0.5) if has_spacing else ratio_step
    if has_bracket_width:
        step = bracket_width / max(float(n_candidates - 1), 1.0)
    if not math.isfinite(step) or step <= 1.0e-7:
        step = min_spacing if has_spacing else 0.25
    step = min(max(float(step), 1.0e-4), 2.0)

    direction = 1.0 if best_ratio > target else -1.0
    center = float(n_candidates - 1) * 0.5
    out: List[float] = []
    for cand_i in range(n_candidates):
        if centered:
            offset = float(cand_i) - center
            tau = best_tau * math.exp(offset * step)
        else:
            tau = best_tau * math.exp(direction * float(cand_i) * step)
        if valid_positive_threshold(tau):
            out.append(float(tau))
    return out or [float(max(best_tau, 1.0e-12))]


def pack_direct_feedback_counts(
    feedback: Optional[Dict[str, Any]],
    counts: torch.Tensor,
) -> None:
    """Pack one rank's direct-emission feedback into a tiny CPU tensor."""
    local_written = int(feedback.get("written", 0)) if feedback else 0
    local_selected = (
        int(feedback.get("selected", local_written)) if feedback else 0
    )
    local_overflow = (
        int(feedback.get("overflow", max(0, local_selected - local_written)))
        if feedback
        else 0
    )
    local_step = int(feedback.get("step", 0)) if feedback else 0
    local_tau_version = int(feedback.get("tau_version", -1)) if feedback else -1
    local_h_epoch = int(feedback.get("tau_h_epoch", -1)) if feedback else -1

    counts[0] = 1 if feedback else 0
    counts[1] = local_selected
    counts[2] = local_written
    counts[3] = local_overflow
    counts[4] = local_step
    counts[5] = local_step * local_step
    if int(counts.numel()) >= 10:
        counts[6] = local_tau_version
        counts[7] = local_tau_version * local_tau_version
        counts[8] = local_h_epoch
        counts[9] = local_h_epoch * local_h_epoch
    if int(counts.numel()) >= 14:
        counts[10] = 0
        counts[11] = int(feedback.get("tau_source_step", -1)) if feedback else -1
        counts[12] = int(feedback.get("tau_apply_from_step", -1)) if feedback else -1
        counts[13] = 0


def finish_direct_feedback_reduce_cpu(
    *,
    feedback: Optional[Dict[str, Any]],
    counts: torch.Tensor,
    world_size: int,
) -> Optional[Dict[str, Any]]:
    """Parse reduced direct-emission feedback if every rank reported same key."""

    ready_ranks = int(counts[0].item())
    if feedback is None or ready_ranks != int(world_size):
        return None
    step_sum = int(counts[4].item())
    step_sq_sum = int(counts[5].item())
    if step_sum * step_sum != ready_ranks * step_sq_sum:
        return None
    if int(counts.numel()) >= 10:
        version_sum = int(counts[6].item())
        version_sq_sum = int(counts[7].item())
        if version_sum * version_sum != ready_ranks * version_sq_sum:
            return None
        epoch_sum = int(counts[8].item())
        epoch_sq_sum = int(counts[9].item())
        if epoch_sum * epoch_sum != ready_ranks * epoch_sq_sum:
            return None

    out = dict(feedback)
    out.update({
        "selected": int(counts[1].item()),
        "written": int(counts[2].item()),
        "overflow": int(counts[3].item()),
        "step": int(step_sum // max(1, ready_ranks)),
    })
    if int(counts.numel()) >= 10:
        out.update({
            "tau_version": int(int(counts[6].item()) // max(1, ready_ranks)),
            "tau_h_epoch": int(int(counts[8].item()) // max(1, ready_ranks)),
        })
    return out


def reduce_direct_feedback_cpu(
    *,
    feedback: Optional[Dict[str, Any]],
    counts: torch.Tensor,
    process_group: Any,
    world_size: int,
) -> Optional[Dict[str, Any]]:
    """All-reduce tiny direct-emission feedback and return a global record."""

    pack_direct_feedback_counts(feedback, counts)
    torch.distributed.all_reduce(
        counts,
        op=torch.distributed.ReduceOp.SUM,
        group=process_group,
    )
    return finish_direct_feedback_reduce_cpu(
        feedback=feedback,
        counts=counts,
        world_size=int(world_size),
    )
