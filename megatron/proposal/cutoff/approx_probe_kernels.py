"""GPU-only approximate cutoff probe and device tau decision."""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:
    from ..score_policy import _proposal_element_score_from_lut

    @triton.jit
    def _stage_candidate_taus_kernel(
        candidate_tau_ptr,
        total_th_ptr,
        n_candidates,
        tau_center,
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    ):
        tl.store(total_th_ptr, tl.maximum(tau_center, 1.0e-12))
        tl.store(candidate_tau_ptr + 0, tl.where(n_candidates > 0, v0, 0.0))
        tl.store(candidate_tau_ptr + 1, tl.where(n_candidates > 1, v1, 0.0))
        tl.store(candidate_tau_ptr + 2, tl.where(n_candidates > 2, v2, 0.0))
        tl.store(candidate_tau_ptr + 3, tl.where(n_candidates > 3, v3, 0.0))
        tl.store(candidate_tau_ptr + 4, tl.where(n_candidates > 4, v4, 0.0))
        tl.store(candidate_tau_ptr + 5, tl.where(n_candidates > 5, v5, 0.0))
        tl.store(candidate_tau_ptr + 6, tl.where(n_candidates > 6, v6, 0.0))
        tl.store(candidate_tau_ptr + 7, tl.where(n_candidates > 7, v7, 0.0))
        tl.store(candidate_tau_ptr + 8, tl.where(n_candidates > 8, v8, 0.0))
        tl.store(candidate_tau_ptr + 9, tl.where(n_candidates > 9, v9, 0.0))
        tl.store(candidate_tau_ptr + 10, tl.where(n_candidates > 10, v10, 0.0))
        tl.store(candidate_tau_ptr + 11, tl.where(n_candidates > 11, v11, 0.0))
        tl.store(candidate_tau_ptr + 12, tl.where(n_candidates > 12, v12, 0.0))
        tl.store(candidate_tau_ptr + 13, tl.where(n_candidates > 13, v13, 0.0))
        tl.store(candidate_tau_ptr + 14, tl.where(n_candidates > 14, v14, 0.0))
        tl.store(candidate_tau_ptr + 15, tl.where(n_candidates > 15, v15, 0.0))

    @triton.jit
    def _age2_load_code(age_words_ptr, global_elem_idx):
        word_idx = (global_elem_idx >> 4).to(tl.int64)
        shift = ((global_elem_idx & 15) << 1).to(tl.int32)
        word = tl.load(age_words_ptr + word_idx, mask=global_elem_idx >= 0, other=0)
        return (word >> shift) & 3

    @triton.jit
    def _hash_u32(x):
        x = x.to(tl.uint32)
        x = (x ^ (x >> 16)) * tl.full((), 0x7FEB352D, tl.uint32)
        x = (x ^ (x >> 15)) * tl.full((), 0x846CA68B, tl.uint32)
        return x ^ (x >> 16)

    @triton.jit
    def _random_unit_from_index(global_idx, salt: tl.constexpr):
        x = global_idx.to(tl.uint32) ^ tl.full((), salt, tl.uint32)
        x = _hash_u32(x)
        mant = x & tl.full((), 0x00FFFFFF, tl.uint32)
        return (mant.to(tl.float32) + 1.0) * (1.0 / 16777217.0)

    @triton.jit
    def _age2_phase_hit(global_elem_idx, step_value, period: tl.constexpr, salt: tl.constexpr):
        x = global_elem_idx.to(tl.uint32) ^ tl.full((), salt, tl.uint32)
        x = _hash_u32(x)
        phase = (x % period).to(tl.int32)
        return ((step_value + phase) % period) == 0

    @triton.jit
    def _generated_probe_entries_kernel(
        m_ptr, v_ptr,
        age_words_ptr, ehat_geom_lut_ptr,
        h_lut_ptr, log_v_min_ptr, inv_log_v_width_ptr,
        entry_elem_start_ptr,
        entry_group_start_ptr,
        entry_group_count_ptr,
        entry_stratum_idx_ptr,
        entry_block_entry_idx_ptr,
        entry_block_local_group_start_ptr,
        candidate_tau_ptr,
        sum_q_ptr,
        sum_q2_ptr,
        n_blocks_ptr,
        sample_blocks_current_ptr,
        decision_code_guard_ptr,
        n_entry_blocks,
        n_candidates,
        total_entry_blocks,
        master_base_elem,
        lr_over_bc1, sqrt_bc2_inv, eps,
        step_value,
        G: tl.constexpr,
        GROUPS_PER_BLOCK: tl.constexpr,
        H_LUT_BINS: tl.constexpr,
        SCORE_POLICY: tl.constexpr,
        MAX_CANDIDATES: tl.constexpr,
        STRATUM_STRIDE: tl.constexpr,
        MAX_SAMPLES_PER_GROUP: tl.constexpr,
        RANDOM_MODE: tl.constexpr,
        RANDOM_SALT: tl.constexpr,
        RUN_IF_DECISION_CODE: tl.constexpr,
    ):
        if RUN_IF_DECISION_CODE >= 0:
            if tl.load(decision_code_guard_ptr) != RUN_IF_DECISION_CODE:
                return
        BLOCK_SIZE: tl.constexpr = G * GROUPS_PER_BLOCK
        sample_id = tl.program_id(axis=0)
        block_count = n_entry_blocks
        if block_count <= 0:
            return
        sample_target = tl.maximum(tl.load(sample_blocks_current_ptr), 1)
        total_blocks = tl.maximum(total_entry_blocks, 1)
        take = (
            sample_target.to(tl.int64) * block_count.to(tl.int64)
            + total_blocks - 1
        ) // total_blocks
        take = tl.maximum(take, 1)
        take = tl.minimum(take, block_count)
        take = tl.minimum(take, MAX_SAMPLES_PER_GROUP)
        if sample_id >= take:
            return

        lo = (sample_id.to(tl.int64) * block_count.to(tl.int64)) // take.to(tl.int64)
        hi = ((sample_id.to(tl.int64) + 1) * block_count.to(tl.int64)) // take.to(tl.int64)
        hi = tl.maximum(hi, lo + 1)
        width = tl.maximum(hi - lo, 1)
        salt = (
            ((tl.cast(master_base_elem, tl.int64) // G).to(tl.uint32) + 1)
            * tl.full((), 0x9E3779B1, tl.uint32)
            ^ (sample_id.to(tl.uint32) + 1) * tl.full((), 0x85EBCA6B, tl.uint32)
        )
        picked = lo + (_hash_u32(salt).to(tl.int64) % width)
        entry_block_id = picked.to(tl.int32)

        entry_id = tl.load(entry_block_entry_idx_ptr + entry_block_id)
        stratum_idx = tl.load(entry_stratum_idx_ptr + entry_id)
        if stratum_idx < 0:
            return
        block_group_start = tl.load(
            entry_block_local_group_start_ptr + entry_block_id,
        )
        entry_elem_start = tl.load(entry_elem_start_ptr + entry_id)
        entry_group_start = tl.load(entry_group_start_ptr + entry_id)
        entry_group_count = tl.load(entry_group_count_ptr + entry_id)

        gid_local = block_group_start + tl.arange(0, GROUPS_PER_BLOCK)
        gid_mask = gid_local < entry_group_count
        elem_lane_2d = tl.broadcast_to(
            tl.arange(0, G)[None, :], (GROUPS_PER_BLOCK, G),
        )
        gid_local_2d = tl.broadcast_to(
            gid_local[:, None], (GROUPS_PER_BLOCK, G),
        )
        elem_off = tl.reshape(
            entry_elem_start + gid_local_2d * G + elem_lane_2d,
            (BLOCK_SIZE,),
        )
        elem_mask = tl.reshape(gid_local_2d < entry_group_count, (BLOCK_SIZE,))
        block_elem_count = tl.sum(elem_mask.to(tl.float32), axis=0)

        m = tl.load(m_ptr + elem_off, mask=elem_mask, other=0.0)
        v = tl.load(v_ptr + elem_off, mask=elem_mask, other=0.0)
        v_fp32 = v.to(tl.float32)
        global_elem_idx = elem_off.to(tl.int64) + tl.cast(master_base_elem, tl.int64)
        age_code = _age2_load_code(age_words_ptr, global_elem_idx)
        elem_score = _proposal_element_score_from_lut(
            m,
            v_fp32,
            age_code,
            h_lut_ptr,
            log_v_min_ptr,
            inv_log_v_width_ptr,
            ehat_geom_lut_ptr,
            lr_over_bc1,
            sqrt_bc2_inv,
            eps,
            H_LUT_BINS,
            SCORE_POLICY,
        )
        force = (age_code == 3) & _age2_phase_hit(
            global_elem_idx, step_value, 128, 0x51ED270B,
        ) & elem_mask
        if RANDOM_MODE:
            elem_score = _random_unit_from_index(global_elem_idx, RANDOM_SALT)
            force = tl.full((BLOCK_SIZE,), False, tl.int1)

        for cand_i in tl.static_range(0, MAX_CANDIDATES):
            enabled = cand_i < n_candidates
            th = tl.load(candidate_tau_ptr + cand_i, mask=enabled, other=float("inf"))
            elem_selected = ((elem_score >= th) | force) & elem_mask & enabled
            selected_n = tl.sum(elem_selected.to(tl.float32), axis=0)
            q = selected_n / tl.maximum(block_elem_count, 1.0)
            off = stratum_idx * STRATUM_STRIDE + cand_i
            tl.atomic_add(sum_q_ptr + off, q, mask=enabled, sem="relaxed")
            tl.atomic_add(sum_q2_ptr + off, q * q, mask=enabled, sem="relaxed")
            tl.atomic_add(n_blocks_ptr + off, 1, mask=enabled, sem="relaxed")

    @triton.jit
    def _zero_probe_accumulators_if_code_kernel(
        sum_q_ptr,
        sum_q2_ptr,
        n_blocks_ptr,
        decision_code_ptr,
        n_total,
        RUN_IF_DECISION_CODE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        if tl.load(decision_code_ptr) != RUN_IF_DECISION_CODE:
            return
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_total
        tl.store(sum_q_ptr + offs, 0.0, mask=mask)
        tl.store(sum_q2_ptr + offs, 0.0, mask=mask)
        tl.store(n_blocks_ptr + offs, 0, mask=mask)

    @triton.jit
    def _build_retry_candidates_if_code_kernel(
        candidate_tau_ptr,
        diagnostics_ptr,
        decision_code_ptr,
        n_candidates,
        source_n_candidates,
        target_ratio,
        RUN_IF_DECISION_CODE: tl.constexpr,
        MAX_CANDIDATES: tl.constexpr,
    ):
        if tl.load(decision_code_ptr) != RUN_IF_DECISION_CODE:
            return
        best_ratio = tl.load(diagnostics_ptr + 0)
        best_tau = tl.load(diagnostics_ptr + 3)
        target = tl.maximum(target_ratio, 1.0e-12)
        valid = (
            (best_tau > 0.0)
            & (best_tau == best_tau)
            & (best_ratio > 0.0)
            & (best_ratio == best_ratio)
        )
        if not valid:
            tl.store(decision_code_ptr, 0)
            return
        log_best = tl.log(best_tau)
        bracket_width = tl.load(diagnostics_ptr + 8)
        min_spacing = float("inf")
        for cand_i in tl.static_range(0, MAX_CANDIDATES):
            enabled = cand_i < source_n_candidates
            tau_i = tl.load(candidate_tau_ptr + cand_i, mask=enabled, other=0.0)
            ok = enabled & (tau_i > 0.0) & (tau_i == tau_i)
            d = tl.abs(tl.log(tau_i) - log_best)
            min_spacing = tl.minimum(min_spacing, tl.where(ok & (d > 1.0e-7), d, float("inf")))
        ratio_step = tl.abs(tl.log(tl.maximum(best_ratio, 1.0e-12) / target))
        has_spacing = min_spacing < float("inf")
        centered = tl.abs(tl.log(tl.maximum(best_ratio, 1.0e-12) / target)) < 1.0e-6
        has_bracket_width = centered & (bracket_width > 1.0e-7) & (bracket_width == bracket_width)
        bracket_step = bracket_width / tl.maximum((n_candidates - 1).to(tl.float32), 1.0)
        step = tl.where(has_spacing, tl.maximum(ratio_step, min_spacing * 0.5), ratio_step)
        step = tl.where(has_bracket_width, bracket_step, step)
        step = tl.where(
            (step != step) | (step <= 1.0e-7),
            tl.where(has_spacing, min_spacing, 0.25),
            step,
        )
        step = tl.minimum(tl.maximum(step, 1.0e-4), 2.0)
        direction = tl.where(best_ratio > target, 1.0, -1.0)
        center = (n_candidates - 1).to(tl.float32) * 0.5
        for cand_i in tl.static_range(0, MAX_CANDIDATES):
            enabled = cand_i < n_candidates
            mul = cand_i * 1.0
            offset = mul - center
            tau_directed = best_tau * tl.exp(direction * mul * step)
            tau_centered = best_tau * tl.exp(offset * step)
            tau = tl.where(centered, tau_centered, tau_directed)
            tl.store(candidate_tau_ptr + cand_i, tl.where(enabled, tau, 0.0))

    @triton.jit
    def _probe_decision_kernel(
        candidate_tau_ptr,
        sum_q_ptr,
        sum_q2_ptr,
        n_blocks_ptr,
        stratum_total_elem_ptr,
        stratum_expert_idx_ptr,
        capacity_demand_hint_ptr,
        total_th_ptr,
        probe_reject_ptr,
        decision_code_ptr,
        sample_blocks_current_ptr,
        ci_shortfall_streak_ptr,
        sample_grow_hold_ptr,
        diagnostics_ptr,
        population_elements,
        target_ratio,
        cutoff_tolerance,
        sample_blocks_min,
        sample_blocks_cap,
        ci_shortfall_grow_after,
        run_if_decision_code,
        USE_SAMPLE_GROW_HOLD: tl.constexpr,
        STRATUM_STRIDE: tl.constexpr,
        N_STRATA: tl.constexpr,
        N_CANDIDATES: tl.constexpr,
        BLOCK_STRATA: tl.constexpr,
        MAX_CANDIDATES: tl.constexpr,
        STORE_CAPACITY_HINT: tl.constexpr,
    ):
        if run_if_decision_code >= 0:
            if tl.load(decision_code_ptr) != run_if_decision_code:
                return
        offs = tl.arange(0, BLOCK_STRATA)
        mask = offs < N_STRATA
        total_elems = tl.load(
            stratum_total_elem_ptr + offs,
            mask=mask,
            other=0,
        ).to(tl.float32)
        pop = tl.maximum(population_elements, 1.0)
        weights = total_elems / pop
        active = mask & (total_elems > 0.0)
        target = tl.maximum(target_ratio, 1.0e-12)
        tol = tl.maximum(cutoff_tolerance, 1.0e-12)
        allowed = target * tol

        best_tau = 0.0
        best_ratio = 0.0
        best_se = float("inf")
        best_bound = float("inf")
        best_valid = False
        best_mean = tl.full((BLOCK_STRATA,), 0.0, tl.float32)
        best_hint = tl.full((BLOCK_STRATA,), 0.0, tl.float32)
        sampled_blocks = 0.0

        bracket_tau = 0.0
        bracket_score = float("inf")
        bracket_found = False
        bracket_se = float("inf")
        bracket_width = float("inf")
        bracket_mean = tl.full((BLOCK_STRATA,), 0.0, tl.float32)
        bracket_hint = tl.full((BLOCK_STRATA,), 0.0, tl.float32)

        for cand_i in tl.static_range(0, MAX_CANDIDATES):
            enabled_i = cand_i < N_CANDIDATES
            tau_i = tl.load(candidate_tau_ptr + cand_i, mask=enabled_i, other=0.0)
            n_i = tl.load(
                n_blocks_ptr + offs * STRATUM_STRIDE + cand_i,
                mask=mask & enabled_i,
                other=0,
            ).to(tl.float32)
            if cand_i == 0:
                sampled_blocks = tl.sum(tl.where(active, n_i, 0.0), axis=0)
            sum_i = tl.load(
                sum_q_ptr + offs * STRATUM_STRIDE + cand_i,
                mask=mask & enabled_i,
                other=0.0,
            )
            sum2_i = tl.load(
                sum_q2_ptr + offs * STRATUM_STRIDE + cand_i,
                mask=mask & enabled_i,
                other=0.0,
            )
            mean_i = tl.where(n_i > 0.0, sum_i / n_i, 0.0)
            numer_i = tl.maximum(sum2_i - (sum_i * sum_i) / tl.maximum(n_i, 1.0), 0.0)
            s2_i = tl.where(n_i > 1.0, numer_i / tl.maximum(n_i - 1.0, 1.0), 0.25)
            se_blocks_i = tl.sqrt(s2_i / tl.maximum(n_i, 1.0))
            hint_i = tl.minimum(
                mean_i + 1.96 * se_blocks_i,
                1.0,
            )
            ratio_i = tl.sum(tl.where(active, weights * mean_i, 0.0), axis=0)
            var_i = tl.sum(
                tl.where(active, weights * weights * s2_i / tl.maximum(n_i, 1.0), 0.0),
                axis=0,
            )
            missing_mass_i = tl.sum(
                tl.where(active & (n_i <= 0.0), weights, 0.0),
                axis=0,
            )
            se_i = 1.96 * tl.sqrt(tl.maximum(var_i, 0.0)) + missing_mass_i
            bound_i = tl.abs(ratio_i - target) + se_i
            valid_i = (
                enabled_i
                & (tau_i > 0.0)
                & (ratio_i > 0.0)
                & (ratio_i == ratio_i)
                & (se_i == se_i)
            )
            take_i = valid_i & (bound_i < best_bound)
            best_tau = tl.where(take_i, tau_i, best_tau)
            best_ratio = tl.where(take_i, ratio_i, best_ratio)
            best_se = tl.where(take_i, se_i, best_se)
            best_bound = tl.where(take_i, bound_i, best_bound)
            best_mean = tl.where(take_i, mean_i, best_mean)
            best_hint = tl.where(take_i, hint_i, best_hint)
            best_valid = best_valid | valid_i

            for cand_j in tl.static_range(0, MAX_CANDIDATES):
                enabled_j = (cand_j < N_CANDIDATES) & (cand_j > cand_i)
                tau_j = tl.load(candidate_tau_ptr + cand_j, mask=enabled_j, other=0.0)
                n_j = tl.load(
                    n_blocks_ptr + offs * STRATUM_STRIDE + cand_j,
                    mask=mask & enabled_j,
                    other=0,
                ).to(tl.float32)
                sum_j = tl.load(
                    sum_q_ptr + offs * STRATUM_STRIDE + cand_j,
                    mask=mask & enabled_j,
                    other=0.0,
                )
                sum2_j = tl.load(
                    sum_q2_ptr + offs * STRATUM_STRIDE + cand_j,
                    mask=mask & enabled_j,
                    other=0.0,
                )
                mean_j = tl.where(n_j > 0.0, sum_j / n_j, 0.0)
                numer_j = tl.maximum(sum2_j - (sum_j * sum_j) / tl.maximum(n_j, 1.0), 0.0)
                s2_j = tl.where(n_j > 1.0, numer_j / tl.maximum(n_j - 1.0, 1.0), 0.25)
                se_blocks_j = tl.sqrt(s2_j / tl.maximum(n_j, 1.0))
                hint_j = tl.minimum(
                    mean_j + 1.96 * se_blocks_j,
                    1.0,
                )
                ratio_j = tl.sum(tl.where(active, weights * mean_j, 0.0), axis=0)
                var_j = tl.sum(
                    tl.where(active, weights * weights * s2_j / tl.maximum(n_j, 1.0), 0.0),
                    axis=0,
                )
                missing_mass_j = tl.sum(
                    tl.where(active & (n_j <= 0.0), weights, 0.0),
                    axis=0,
                )
                se_j = 1.96 * tl.sqrt(tl.maximum(var_j, 0.0)) + missing_mass_j
                valid_j = (
                    enabled_j
                    & (tau_j > 0.0)
                    & (ratio_j > 0.0)
                    & (ratio_j == ratio_j)
                    & (se_j == se_j)
                )
                crosses = (
                    valid_i
                    & valid_j
                    & (((ratio_i - target) * (ratio_j - target)) <= 0.0)
                )
                x_i = tl.log(tau_i)
                x_j = tl.log(tau_j)
                y_i = tl.log(tl.maximum(ratio_i, 1.0e-12))
                y_j = tl.log(tl.maximum(ratio_j, 1.0e-12))
                slope = (y_j - y_i) / (x_j - x_i)
                pred_x = x_i + (tl.log(target) - y_i) / slope
                x_lo = tl.minimum(x_i, x_j)
                x_hi = tl.maximum(x_i, x_j)
                pred_ok = (
                    crosses
                    & (tl.abs(x_j - x_i) > 1.0e-12)
                    & (tl.abs(y_j - y_i) > 1.0e-12)
                    & (slope < 0.0)
                    & (pred_x >= x_lo)
                    & (pred_x <= x_hi)
                )
                pair_se = tl.maximum(se_i, se_j)
                pair_score = pair_se + (x_hi - x_lo) * target * 1.0e-3
                take_pair = pred_ok & (pair_score < bracket_score)
                endpoint_mean = tl.where(
                    tl.abs(ratio_i - target) <= tl.abs(ratio_j - target),
                    mean_i,
                    mean_j,
                )
                endpoint_hint = tl.maximum(hint_i, hint_j)
                bracket_tau = tl.where(take_pair, tl.exp(pred_x), bracket_tau)
                bracket_score = tl.where(take_pair, pair_score, bracket_score)
                bracket_se = tl.where(take_pair, pair_se, bracket_se)
                bracket_width = tl.where(take_pair, x_hi - x_lo, bracket_width)
                bracket_mean = tl.where(take_pair, endpoint_mean, bracket_mean)
                bracket_hint = tl.where(take_pair, endpoint_hint, bracket_hint)
                bracket_found = bracket_found | take_pair

        best_accept = best_valid & (best_bound <= allowed)
        first_pass = run_if_decision_code < 0
        use_bracket_for_reprobe = first_pass & bracket_found & (~best_accept)
        use_bracket_final = (~first_pass) & bracket_found & (~best_accept)
        strict_accept = best_accept
        accept = bracket_found | best_valid
        use_bracket_tau = use_bracket_for_reprobe | use_bracket_final
        chosen_tau = tl.where(use_bracket_tau, bracket_tau, best_tau)
        chosen_ratio = tl.where(use_bracket_tau, target, best_ratio)
        chosen_se = tl.where(use_bracket_tau, bracket_se, best_se)
        chosen_mean = tl.where(use_bracket_tau, bracket_mean, best_mean)
        chosen_hint = tl.where(use_bracket_tau, bracket_hint, best_hint)
        decision = tl.where(
            best_accept,
            2,
            tl.where(
                use_bracket_for_reprobe | (first_pass & best_valid),
                3,
                tl.where(best_valid, 4, 0),
            ),
        )
        tl.store(total_th_ptr, chosen_tau, mask=accept)
        tl.store(probe_reject_ptr, tl.where(accept, 0, 1))
        tl.store(decision_code_ptr, decision)

        cur = tl.maximum(tl.load(sample_blocks_current_ptr), sample_blocks_min)
        shrunk = tl.maximum((cur * 7) // 8, sample_blocks_min)
        hold_growth = False
        if USE_SAMPLE_GROW_HOLD:
            hold_growth = tl.load(sample_grow_hold_ptr) != 0
        ci_shortfall = (decision == 4) & (~strict_accept) & (~use_bracket_final)
        prev_streak = tl.maximum(tl.load(ci_shortfall_streak_ptr), 0)
        next_streak = tl.minimum(
            prev_streak + 1,
            tl.maximum(ci_shortfall_grow_after, 1),
        )
        grow = ci_shortfall & (next_streak >= tl.maximum(ci_shortfall_grow_after, 1))
        grown = tl.minimum(cur * 2, sample_blocks_cap)
        next_sample = tl.where(
            grow,
            grown,
            tl.where(
                strict_accept & (chosen_se < allowed * 0.25) & (~hold_growth),
                shrunk,
                cur,
            ),
        )
        stored_streak = tl.where(
            strict_accept | grow,
            0,
            tl.where(ci_shortfall, next_streak, prev_streak),
        )
        tl.store(ci_shortfall_streak_ptr, stored_streak)
        tl.store(sample_blocks_current_ptr, next_sample)
        if USE_SAMPLE_GROW_HOLD:
            tl.store(sample_grow_hold_ptr, tl.where(grow, 1, 0))
        tl.store(diagnostics_ptr + 0, chosen_ratio)
        tl.store(diagnostics_ptr + 1, chosen_se)
        tl.store(diagnostics_ptr + 2, best_bound)
        tl.store(diagnostics_ptr + 3, chosen_tau)
        tl.store(diagnostics_ptr + 4, next_sample.to(tl.float32))
        tl.store(diagnostics_ptr + 5, tl.where(strict_accept, 1.0, 0.0))
        tl.store(diagnostics_ptr + 6, sampled_blocks)
        tl.store(diagnostics_ptr + 7, tl.where(accept, 1.0, 0.0))
        tl.store(
            diagnostics_ptr + 8,
            tl.where(use_bracket_for_reprobe, bracket_width, 0.0),
        )
        if STORE_CAPACITY_HINT:
            hint = tl.maximum(chosen_hint, 0.0) * total_elems
            expert_idx = tl.load(stratum_expert_idx_ptr + offs, mask=active, other=-1)
            expert_idx_safe = tl.maximum(expert_idx, 0)
            tl.atomic_add(
                capacity_demand_hint_ptr + expert_idx_safe,
                hint,
                mask=active & (expert_idx >= 0),
                sem="relaxed",
            )

    @triton.jit
    def _reduce_probe_candidate_stats_kernel(
        sum_q_ptr,
        sum_q2_ptr,
        n_blocks_ptr,
        stratum_total_elem_ptr,
        stats_ptr,
        population_elements,
        STRATUM_STRIDE: tl.constexpr,
        N_STRATA: tl.constexpr,
        N_CANDIDATES: tl.constexpr,
        BLOCK_STRATA: tl.constexpr,
        MAX_CANDIDATES: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK_STRATA)
        mask = offs < N_STRATA
        total_elems = tl.load(
            stratum_total_elem_ptr + offs,
            mask=mask,
            other=0,
        ).to(tl.float32)
        pop = tl.maximum(population_elements, 1.0)
        weights = total_elems / pop
        active = mask & (total_elems > 0.0)

        for cand_i in tl.static_range(0, MAX_CANDIDATES):
            enabled_i = cand_i < N_CANDIDATES
            n_i = tl.load(
                n_blocks_ptr + offs * STRATUM_STRIDE + cand_i,
                mask=mask & enabled_i,
                other=0,
            ).to(tl.float32)
            sum_i = tl.load(
                sum_q_ptr + offs * STRATUM_STRIDE + cand_i,
                mask=mask & enabled_i,
                other=0.0,
            )
            sum2_i = tl.load(
                sum_q2_ptr + offs * STRATUM_STRIDE + cand_i,
                mask=mask & enabled_i,
                other=0.0,
            )
            mean_i = tl.where(n_i > 0.0, sum_i / n_i, 0.0)
            numer_i = tl.maximum(
                sum2_i - (sum_i * sum_i) / tl.maximum(n_i, 1.0),
                0.0,
            )
            s2_i = tl.where(n_i > 1.0, numer_i / tl.maximum(n_i - 1.0, 1.0), 0.25)
            ratio_i = tl.sum(tl.where(active, weights * mean_i, 0.0), axis=0)
            var_i = tl.sum(
                tl.where(
                    active,
                    weights * weights * s2_i / tl.maximum(n_i, 1.0),
                    0.0,
                ),
                axis=0,
            )
            missing_i = tl.sum(
                tl.where(active & (n_i <= 0.0), weights, 0.0),
                axis=0,
            )
            sampled_i = tl.sum(tl.where(active, n_i, 0.0), axis=0)
            tl.store(stats_ptr + 0 * MAX_CANDIDATES + cand_i, ratio_i, mask=enabled_i)
            tl.store(stats_ptr + 1 * MAX_CANDIDATES + cand_i, var_i, mask=enabled_i)
            tl.store(stats_ptr + 2 * MAX_CANDIDATES + cand_i, missing_i, mask=enabled_i)
            tl.store(stats_ptr + 3 * MAX_CANDIDATES + cand_i, sampled_i, mask=enabled_i)

    @triton.jit
    def _store_global_probe_decision_kernel(
        total_th_ptr,
        probe_reject_ptr,
        decision_code_ptr,
        sample_blocks_current_ptr,
        ci_shortfall_streak_ptr,
        sample_grow_hold_ptr,
        diagnostics_ptr,
        chosen_tau,
        chosen_ratio,
        chosen_se,
        best_bound,
        next_sample,
        strict_accept,
        sampled_blocks,
        accept,
        bracket_width,
        decision,
        stored_streak,
        grow_hold_value,
    ):
        ok = accept != 0
        tl.store(total_th_ptr, chosen_tau, mask=ok)
        tl.store(probe_reject_ptr, tl.where(ok, 0, 1))
        tl.store(decision_code_ptr, decision)
        tl.store(sample_blocks_current_ptr, next_sample)
        tl.store(ci_shortfall_streak_ptr, stored_streak)
        tl.store(sample_grow_hold_ptr, grow_hold_value)
        tl.store(diagnostics_ptr + 0, chosen_ratio)
        tl.store(diagnostics_ptr + 1, chosen_se)
        tl.store(diagnostics_ptr + 2, best_bound)
        tl.store(diagnostics_ptr + 3, chosen_tau)
        tl.store(diagnostics_ptr + 4, next_sample.to(tl.float32))
        tl.store(diagnostics_ptr + 5, tl.where(strict_accept != 0, 1.0, 0.0))
        tl.store(diagnostics_ptr + 6, sampled_blocks)
        tl.store(diagnostics_ptr + 7, tl.where(ok, 1.0, 0.0))
        tl.store(diagnostics_ptr + 8, bracket_width)

    @triton.jit
    def _capacity_hint_from_probe_kernel(
        candidate_tau_ptr,
        sum_q_ptr,
        sum_q2_ptr,
        n_blocks_ptr,
        stratum_total_elem_ptr,
        stratum_expert_idx_ptr,
        total_th_ptr,
        capacity_demand_hint_ptr,
        STRATUM_STRIDE: tl.constexpr,
        N_STRATA: tl.constexpr,
        N_CANDIDATES: tl.constexpr,
        BLOCK_STRATA: tl.constexpr,
        MAX_CANDIDATES: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK_STRATA)
        mask = offs < N_STRATA
        total_elems = tl.load(
            stratum_total_elem_ptr + offs,
            mask=mask,
            other=0,
        ).to(tl.float32)
        active = mask & (total_elems > 0.0)
        chosen_tau = tl.maximum(tl.load(total_th_ptr), 1.0e-30)
        log_chosen = tl.log(chosen_tau)

        best_tau_below = tl.full((BLOCK_STRATA,), 0.0, tl.float32)
        best_hint_below = tl.full((BLOCK_STRATA,), 0.0, tl.float32)
        best_log_below = tl.full((BLOCK_STRATA,), -float("inf"), tl.float32)
        fallback_hint = tl.full((BLOCK_STRATA,), 0.0, tl.float32)
        fallback_dist = tl.full((BLOCK_STRATA,), float("inf"), tl.float32)
        valid_any = tl.full((BLOCK_STRATA,), False, tl.int1)

        for cand_i in tl.static_range(0, MAX_CANDIDATES):
            enabled = cand_i < N_CANDIDATES
            tau_i = tl.load(candidate_tau_ptr + cand_i, mask=enabled, other=0.0)
            n_i = tl.load(
                n_blocks_ptr + offs * STRATUM_STRIDE + cand_i,
                mask=mask & enabled,
                other=0,
            ).to(tl.float32)
            sum_i = tl.load(
                sum_q_ptr + offs * STRATUM_STRIDE + cand_i,
                mask=mask & enabled,
                other=0.0,
            )
            sum2_i = tl.load(
                sum_q2_ptr + offs * STRATUM_STRIDE + cand_i,
                mask=mask & enabled,
                other=0.0,
            )
            valid_i = enabled & active & (tau_i > 0.0) & (tau_i == tau_i) & (n_i > 0.0)
            mean_i = tl.where(n_i > 0.0, sum_i / tl.maximum(n_i, 1.0), 0.0)
            numer = tl.maximum(sum2_i - (sum_i * sum_i) / tl.maximum(n_i, 1.0), 0.0)
            sample_var = tl.where(n_i > 1.0, numer / tl.maximum(n_i - 1.0, 1.0), 0.25)
            se_i = tl.sqrt(sample_var / tl.maximum(n_i, 1.0))
            cap_hint_i = tl.minimum(
                mean_i + 1.96 * se_i,
                1.0,
            )
            log_tau = tl.log(tau_i)
            below = valid_i & (tau_i <= chosen_tau) & (log_tau > best_log_below)
            best_tau_below = tl.where(below, tau_i, best_tau_below)
            best_log_below = tl.where(below, log_tau, best_log_below)
            best_hint_below = tl.where(below, cap_hint_i, best_hint_below)

            dist = tl.abs(log_tau - log_chosen)
            take_fallback = valid_i & (dist < fallback_dist)
            fallback_dist = tl.where(take_fallback, dist, fallback_dist)
            fallback_hint = tl.where(take_fallback, cap_hint_i, fallback_hint)
            valid_any = valid_any | valid_i

        use_below = best_tau_below > 0.0
        cap_hint = tl.where(use_below, best_hint_below, fallback_hint)
        hint = tl.maximum(cap_hint, 0.0) * total_elems
        expert_idx = tl.load(stratum_expert_idx_ptr + offs, mask=active, other=-1)
        expert_idx_safe = tl.maximum(expert_idx, 0)
        tl.atomic_add(
            capacity_demand_hint_ptr + expert_idx_safe,
            hint,
            mask=active & valid_any & (expert_idx >= 0),
            sem="relaxed",
        )

def _groups_per_program(group_size: int, target_block: Optional[int] = None) -> int:
    if target_block is None:
        target_block = 256
    gpb = max(1, int(target_block) // max(1, int(group_size)))
    power = 1
    while power < gpb:
        power *= 2
    return int(power)


def launch_stage_candidate_taus(
    *,
    candidate_taus: torch.Tensor,
    total_th: torch.Tensor,
    values: tuple[float, ...],
    tau_center: float,
) -> int:
    if not _HAS_TRITON:
        return 0
    n_cand = min(len(values), int(candidate_taus.numel()), 16)
    if n_cand <= 0:
        return 0
    padded = [0.0] * 16
    for i, value in enumerate(values[:n_cand]):
        padded[i] = float(value)
    _stage_candidate_taus_kernel[(1,)](
        candidate_taus,
        total_th,
        int(n_cand),
        float(tau_center),
        float(padded[0]),
        float(padded[1]),
        float(padded[2]),
        float(padded[3]),
        float(padded[4]),
        float(padded[5]),
        float(padded[6]),
        float(padded[7]),
        float(padded[8]),
        float(padded[9]),
        float(padded[10]),
        float(padded[11]),
        float(padded[12]),
        float(padded[13]),
        float(padded[14]),
        float(padded[15]),
    )
    return int(n_cand)


def launch_generated_block_probe_for_group(
    *,
    m_flat: torch.Tensor,
    v_flat: torch.Tensor,
    age2bit_words: torch.Tensor,
    ehat_geom_lut: torch.Tensor,
    h_lut_values: torch.Tensor,
    h_lut_log_v_min: torch.Tensor,
    h_lut_inv_log_v_width: torch.Tensor,
    entry_elem_start: torch.Tensor,
    entry_group_start: torch.Tensor,
    entry_group_count: torch.Tensor,
    entry_stratum_idx: torch.Tensor,
    entry_block_entry_idx: torch.Tensor,
    entry_block_local_group_start: torch.Tensor,
    candidate_taus: torch.Tensor,
    sum_q: torch.Tensor,
    sum_q2: torch.Tensor,
    n_blocks: torch.Tensor,
    sample_blocks_current: torch.Tensor,
    n_candidates: int,
    total_entry_blocks: int,
    max_samples_per_group: int,
    master_off_elem: int,
    group_size: int,
    lr_over_bc1: float,
    sqrt_bc2_inv: float,
    eps: float,
    step: int,
    score_policy_id: int,
    random_mode: bool = False,
    random_salt: int = 0,
    decision_code_guard: Optional[torch.Tensor] = None,
    run_if_decision_code: int = -1,
) -> None:
    if not _HAS_TRITON:
        return
    n_entry_blocks = int(entry_block_entry_idx.numel())
    n_cand = min(int(n_candidates), int(candidate_taus.numel()), int(sum_q.shape[1]))
    candidate_capacity = min(int(candidate_taus.numel()), int(sum_q.shape[1]))
    if n_entry_blocks <= 0 or n_cand <= 0:
        return
    gpb = _groups_per_program(group_size)
    max_samples = max(1, min(int(max_samples_per_group), int(n_entry_blocks)))
    guard = decision_code_guard if decision_code_guard is not None else sample_blocks_current
    _generated_probe_entries_kernel[(max_samples,)](
        m_flat,
        v_flat,
        age2bit_words,
        ehat_geom_lut,
        h_lut_values,
        h_lut_log_v_min,
        h_lut_inv_log_v_width,
        entry_elem_start,
        entry_group_start,
        entry_group_count,
        entry_stratum_idx,
        entry_block_entry_idx,
        entry_block_local_group_start,
        candidate_taus,
        sum_q,
        sum_q2,
        n_blocks,
        sample_blocks_current,
        guard,
        int(n_entry_blocks),
        int(n_cand),
        int(max(1, total_entry_blocks)),
        int(master_off_elem),
        float(lr_over_bc1),
        float(sqrt_bc2_inv),
        float(eps),
        int(step),
        G=int(group_size),
        GROUPS_PER_BLOCK=gpb,
        H_LUT_BINS=int(h_lut_values.numel()),
        SCORE_POLICY=int(score_policy_id),
        MAX_CANDIDATES=int(candidate_capacity),
        STRATUM_STRIDE=int(sum_q.shape[1]),
        MAX_SAMPLES_PER_GROUP=max_samples,
        RANDOM_MODE=bool(random_mode),
        RANDOM_SALT=(int(random_salt) & 0xFFFFFFFF) if bool(random_mode) else 0,
        RUN_IF_DECISION_CODE=int(run_if_decision_code),
    )


def launch_zero_probe_accumulators_if_code(
    *,
    sum_q: torch.Tensor,
    sum_q2: torch.Tensor,
    n_blocks: torch.Tensor,
    decision_code: torch.Tensor,
    run_if_decision_code: int,
) -> None:
    if not _HAS_TRITON:
        return
    n_total = min(int(sum_q.numel()), int(sum_q2.numel()), int(n_blocks.numel()))
    if n_total <= 0:
        return
    block = 256
    grid = (triton.cdiv(n_total, block),)
    _zero_probe_accumulators_if_code_kernel[grid](
        sum_q,
        sum_q2,
        n_blocks,
        decision_code,
        int(n_total),
        RUN_IF_DECISION_CODE=int(run_if_decision_code),
        BLOCK=block,
    )


def launch_retry_candidates_if_code(
    *,
    candidate_taus: torch.Tensor,
    diagnostics: torch.Tensor,
    decision_code: torch.Tensor,
    n_candidates: int,
    target_ratio: float,
    run_if_decision_code: int,
    source_n_candidates: Optional[int] = None,
) -> None:
    if not _HAS_TRITON:
        return
    n_cand = min(max(0, int(n_candidates)), int(candidate_taus.numel()))
    source_n_cand = min(
        max(0, int(source_n_candidates if source_n_candidates is not None else n_candidates)),
        int(candidate_taus.numel()),
    )
    candidate_capacity = int(candidate_taus.numel())
    if n_cand <= 0:
        return
    _build_retry_candidates_if_code_kernel[(1,)](
        candidate_taus,
        diagnostics,
        decision_code,
        int(n_cand),
        int(source_n_cand),
        float(target_ratio),
        RUN_IF_DECISION_CODE=int(run_if_decision_code),
        MAX_CANDIDATES=int(candidate_capacity),
    )


def launch_probe_decision(
    *,
    candidate_taus: torch.Tensor,
    sum_q: torch.Tensor,
    sum_q2: torch.Tensor,
    n_blocks: torch.Tensor,
    stratum_total_elems: torch.Tensor,
    stratum_expert_idx: Optional[torch.Tensor] = None,
    capacity_demand_hint: Optional[torch.Tensor] = None,
    total_th: torch.Tensor,
    probe_reject: torch.Tensor,
    decision_code: torch.Tensor,
    sample_blocks_current: torch.Tensor,
    ci_shortfall_streak: torch.Tensor,
    diagnostics: torch.Tensor,
    n_candidates: int,
    total_elems_global: int,
    target_ratio: float,
    cutoff_tolerance: float,
    sample_blocks_min: int,
    sample_blocks_cap: int,
    sample_grow_hold: Optional[torch.Tensor] = None,
    ci_shortfall_grow_after: int = 3,
    run_if_decision_code: int = -1,
) -> None:
    if not _HAS_TRITON:
        return
    n_cand = min(max(0, int(n_candidates)), int(candidate_taus.numel()), int(sum_q.shape[1]))
    candidate_capacity = min(int(candidate_taus.numel()), int(sum_q.shape[1]))
    n_strata = min(int(sum_q.shape[0]), int(stratum_total_elems.numel()))
    if n_cand <= 0 or n_strata <= 0:
        probe_reject.fill_(1)
        return
    use_sample_grow_hold = sample_grow_hold is not None
    grow_hold_ptr = sample_grow_hold if use_sample_grow_hold else ci_shortfall_streak
    store_capacity_hint = capacity_demand_hint is not None
    capacity_hint_ptr = (
        capacity_demand_hint if store_capacity_hint else stratum_total_elems
    )
    stratum_expert_idx_ptr = (
        stratum_expert_idx if stratum_expert_idx is not None else stratum_total_elems
    )
    block_strata = 1
    while block_strata < n_strata:
        block_strata <<= 1
    _probe_decision_kernel[(1,)](
        candidate_taus,
        sum_q,
        sum_q2,
        n_blocks,
        stratum_total_elems,
        stratum_expert_idx_ptr,
        capacity_hint_ptr,
        total_th,
        probe_reject,
        decision_code,
        sample_blocks_current,
        ci_shortfall_streak,
        grow_hold_ptr,
        diagnostics,
        float(max(1, int(total_elems_global))),
        float(target_ratio),
        float(cutoff_tolerance),
        int(max(1, sample_blocks_min)),
        int(max(1, sample_blocks_cap)),
        int(max(1, ci_shortfall_grow_after)),
        int(run_if_decision_code),
        USE_SAMPLE_GROW_HOLD=bool(use_sample_grow_hold),
        STRATUM_STRIDE=int(sum_q.shape[1]),
        N_STRATA=int(n_strata),
        N_CANDIDATES=int(candidate_capacity),
        BLOCK_STRATA=int(block_strata),
        MAX_CANDIDATES=int(candidate_capacity),
        STORE_CAPACITY_HINT=bool(store_capacity_hint),
    )


def launch_reduce_probe_candidate_stats(
    *,
    sum_q: torch.Tensor,
    sum_q2: torch.Tensor,
    n_blocks: torch.Tensor,
    stratum_total_elems: torch.Tensor,
    candidate_stats: torch.Tensor,
    n_candidates: int,
    total_elems_global: int,
) -> None:
    if not _HAS_TRITON:
        return
    n_cand = min(
        max(0, int(n_candidates)),
        int(sum_q.shape[1]),
        int(candidate_stats.shape[1]),
    )
    n_strata = min(int(sum_q.shape[0]), int(stratum_total_elems.numel()))
    if n_cand <= 0 or n_strata <= 0:
        return
    block_strata = 1
    while block_strata < n_strata:
        block_strata <<= 1
    _reduce_probe_candidate_stats_kernel[(1,)](
        sum_q,
        sum_q2,
        n_blocks,
        stratum_total_elems,
        candidate_stats,
        float(max(1, int(total_elems_global))),
        STRATUM_STRIDE=int(sum_q.shape[1]),
        N_STRATA=int(n_strata),
        N_CANDIDATES=int(candidate_stats.shape[1]),
        BLOCK_STRATA=int(block_strata),
        MAX_CANDIDATES=int(candidate_stats.shape[1]),
    )


def launch_store_global_probe_decision(
    *,
    total_th: torch.Tensor,
    probe_reject: torch.Tensor,
    decision_code: torch.Tensor,
    sample_blocks_current: torch.Tensor,
    ci_shortfall_streak: torch.Tensor,
    sample_grow_hold: torch.Tensor,
    diagnostics: torch.Tensor,
    chosen_tau: float,
    chosen_ratio: float,
    chosen_se: float,
    best_bound: float,
    next_sample: int,
    strict_accept: bool,
    sampled_blocks: float,
    accept: bool,
    bracket_width: float,
    decision: int,
    stored_streak: int,
    grow_hold_value: int,
) -> None:
    if not _HAS_TRITON:
        return
    _store_global_probe_decision_kernel[(1,)](
        total_th,
        probe_reject,
        decision_code,
        sample_blocks_current,
        ci_shortfall_streak,
        sample_grow_hold,
        diagnostics,
        float(chosen_tau),
        float(chosen_ratio),
        float(chosen_se),
        float(best_bound),
        int(next_sample),
        int(1 if strict_accept else 0),
        float(sampled_blocks),
        int(1 if accept else 0),
        float(bracket_width),
        int(decision),
        int(stored_streak),
        int(grow_hold_value),
    )


def launch_capacity_hint_from_probe(
    *,
    candidate_taus: torch.Tensor,
    sum_q: torch.Tensor,
    sum_q2: torch.Tensor,
    n_blocks: torch.Tensor,
    stratum_total_elems: torch.Tensor,
    stratum_expert_idx: torch.Tensor,
    total_th: torch.Tensor,
    capacity_demand_hint: torch.Tensor,
    n_candidates: int,
) -> None:
    if not _HAS_TRITON:
        return
    n_cand = min(max(0, int(n_candidates)), int(candidate_taus.numel()), int(sum_q.shape[1]))
    n_strata = min(
        int(sum_q.shape[0]),
        int(stratum_total_elems.numel()),
        int(stratum_expert_idx.numel()),
    )
    if n_cand <= 0 or n_strata <= 0 or int(capacity_demand_hint.numel()) <= 0:
        return
    block_strata = 1
    while block_strata < n_strata:
        block_strata <<= 1
    _capacity_hint_from_probe_kernel[(1,)](
        candidate_taus,
        sum_q,
        sum_q2,
        n_blocks,
        stratum_total_elems,
        stratum_expert_idx,
        total_th,
        capacity_demand_hint,
        STRATUM_STRIDE=int(sum_q.shape[1]),
        N_STRATA=int(n_strata),
        N_CANDIDATES=int(sum_q.shape[1]),
        BLOCK_STRATA=int(block_strata),
        MAX_CANDIDATES=int(sum_q.shape[1]),
    )


__all__ = [
    "launch_generated_block_probe_for_group",
    "launch_capacity_hint_from_probe",
    "launch_probe_decision",
    "launch_reduce_probe_candidate_stats",
    "launch_store_global_probe_decision",
    "launch_retry_candidates_if_code",
    "launch_stage_candidate_taus",
    "launch_zero_probe_accumulators_if_code",
    "_groups_per_program",
]
