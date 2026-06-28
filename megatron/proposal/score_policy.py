"""Shared production selection score policy.

The hot path has one element-score helper used by both cutoff probe and direct
emission.  Per-element rho/geom math is intentionally kept out of the scan
kernels; ``launch_update_ehat_geom_lut`` refreshes a tiny device LUT once per
iteration.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


SCORE_POLICY_ABS_H_TAYLOR_FIRST_AND_SECOND_ORDER = 0
SCORE_POLICY_ABS_H_TAYLOR_ONLY_SECOND_ORDER = 1

SCORE_POLICY_NAMES = {
    "abs_h_taylor_first_and_second_order": (
        SCORE_POLICY_ABS_H_TAYLOR_FIRST_AND_SECOND_ORDER
    ),
    "abs_h_taylor_only_second_order": (
        SCORE_POLICY_ABS_H_TAYLOR_ONLY_SECOND_ORDER
    ),
}
DEFAULT_SCORE_POLICY = "abs_h_taylor_first_and_second_order"


def score_policy_id(name: str) -> int:
    try:
        return int(SCORE_POLICY_NAMES[str(name)])
    except KeyError as exc:
        valid = ", ".join(sorted(SCORE_POLICY_NAMES))
        raise ValueError(f"proposal score policy must be one of {valid}; got {name!r}") from exc


if _HAS_TRITON:

    @triton.jit
    def _lookup_h_and_geom_from_lut(
        v_fp32,
        age_code,
        h_lut_ptr,
        log_v_min_ptr,
        inv_log_v_width_ptr,
        ehat_geom_lut_ptr,
        H_LUT_BINS: tl.constexpr,
    ):
        log_v_min = tl.load(log_v_min_ptr)
        inv_width = tl.load(inv_log_v_width_ptr)
        log_v = tl.log(tl.maximum(v_fp32, 1.0e-30))
        pos = (log_v - log_v_min) * inv_width
        pos = tl.maximum(tl.minimum(pos, H_LUT_BINS - 1.0), 0.0)
        idx_lo = pos.to(tl.int32)
        idx_hi = tl.minimum(idx_lo + 1, H_LUT_BINS - 1)
        frac = pos - idx_lo.to(tl.float32)

        h_lo = tl.load(h_lut_ptr + idx_lo)
        h_hi = tl.load(h_lut_ptr + idx_hi)
        h = h_lo + frac * (h_hi - h_lo)

        geom_base = age_code.to(tl.int64) * H_LUT_BINS
        geom_lo = tl.load(ehat_geom_lut_ptr + geom_base + idx_lo)
        geom_hi = tl.load(ehat_geom_lut_ptr + geom_base + idx_hi)
        geom = geom_lo + frac * (geom_hi - geom_lo)
        return h, geom

    @triton.jit
    def _proposal_element_score_from_lut(
        m,
        linear,
        v,
        age_code,
        h_lut_ptr,
        log_v_min_ptr,
        inv_log_v_width_ptr,
        ehat_geom_lut_ptr,
        lr_over_bc1,
        sqrt_bc2_inv,
        eps,
        H_LUT_BINS: tl.constexpr,
        SCORE_POLICY: tl.constexpr,
    ):
        v_fp32 = v.to(tl.float32)
        h_abs, geom = _lookup_h_and_geom_from_lut(
            v_fp32,
            age_code,
            h_lut_ptr,
            log_v_min_ptr,
            inv_log_v_width_ptr,
            ehat_geom_lut_ptr,
            H_LUT_BINS,
        )
        denom = tl.sqrt(v_fp32) * sqrt_bc2_inv + eps
        delta_signed = -lr_over_bc1 * m / denom
        e_hat = delta_signed * geom
        second_order = h_abs * e_hat * e_hat
        if SCORE_POLICY == 1:
            return second_order
        first_order = -linear * e_hat
        return tl.maximum(first_order + 0.5 * second_order, 0.0)

    @triton.jit
    def _geom_sum_from_rho(rho, age_value: tl.constexpr):
        # sum_{k=0}^{age-1} rho^{-k}; rho is clipped close to [0.9, 1].
        inv_rho = 1.0 / rho
        denom = inv_rho - 1.0
        pow_term = tl.exp(tl.log(inv_rho) * age_value)
        geom = (pow_term - 1.0) / denom
        return tl.where(tl.abs(denom) < 1.0e-6, age_value + 0.0, geom)

    @triton.jit
    def _update_ehat_geom_lut_kernel(
        h_lut_ptr,
        log_v_min_ptr,
        inv_log_v_width_ptr,
        geom_lut_ptr,
        lr_over_bc1,
        sqrt_bc2_inv,
        eps,
        beta1,
        H_LUT_BINS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < H_LUT_BINS
        inv_width = tl.load(inv_log_v_width_ptr)
        log_v_min = tl.load(log_v_min_ptr)
        safe_inv_width = tl.where(inv_width > 0.0, inv_width, 1.0)
        log_v = log_v_min + offs.to(tl.float32) / safe_inv_width
        v = tl.exp(log_v)
        h_abs = tl.load(h_lut_ptr + offs, mask=mask, other=0.0)
        denom = tl.sqrt(tl.maximum(v, 0.0)) * sqrt_bc2_inv + eps
        mu = tl.abs(lr_over_bc1) * h_abs / tl.maximum(denom, 1.0e-30)
        b = tl.minimum(tl.maximum(beta1, 0.0), 0.999999)
        trace = 1.0 + b - (1.0 - b) * mu
        disc = trace * trace - 4.0 * b
        sqrt_disc = tl.sqrt(tl.maximum(disc, 0.0))
        root0 = 0.5 * (trace + sqrt_disc)
        root1 = 0.5 * (trace - sqrt_disc)
        spectral = tl.maximum(tl.abs(root0), tl.abs(root1))
        complex_rho = tl.sqrt(b)
        rho = tl.where(disc >= 0.0, spectral, complex_rho)
        rho = tl.minimum(tl.maximum(rho, 0.9), 1.0)
        if H_LUT_BINS > 0:
            tl.store(
                geom_lut_ptr + offs,
                _geom_sum_from_rho(rho, 1),
                mask=mask,
            )
            tl.store(
                geom_lut_ptr + H_LUT_BINS + offs,
                _geom_sum_from_rho(rho, 4),
                mask=mask,
            )
            tl.store(
                geom_lut_ptr + H_LUT_BINS * 2 + offs,
                _geom_sum_from_rho(rho, 16),
                mask=mask,
            )
            tl.store(
                geom_lut_ptr + H_LUT_BINS * 3 + offs,
                _geom_sum_from_rho(rho, 64),
                mask=mask,
            )


def launch_update_ehat_geom_lut(
    *,
    h_lut_values: torch.Tensor,
    h_lut_log_v_min: torch.Tensor,
    h_lut_inv_log_v_width: torch.Tensor,
    ehat_geom_lut: torch.Tensor,
    lr_over_bc1: float,
    sqrt_bc2_inv: float,
    eps: float,
    beta1: float,
) -> None:
    """Refresh the small e_hat geometry LUT for the current Adam scalars."""

    if not _HAS_TRITON:
        raise RuntimeError("proposal score policy requires Triton")
    n_bins = int(h_lut_values.numel())
    if n_bins <= 0:
        return
    if ehat_geom_lut.numel() < 4 * n_bins:
        raise RuntimeError("ehat_geom_lut must contain 4 * h_lut_bins entries")
    block = 256
    grid = (triton.cdiv(n_bins, block),)
    _update_ehat_geom_lut_kernel[grid](
        h_lut_values,
        h_lut_log_v_min,
        h_lut_inv_log_v_width,
        ehat_geom_lut,
        float(lr_over_bc1),
        float(sqrt_bc2_inv),
        float(eps),
        float(beta1),
        H_LUT_BINS=int(n_bins),
        BLOCK=int(block),
    )


__all__ = [
    "DEFAULT_SCORE_POLICY",
    "SCORE_POLICY_ABS_H_TAYLOR_FIRST_AND_SECOND_ORDER",
    "SCORE_POLICY_ABS_H_TAYLOR_ONLY_SECOND_ORDER",
    "SCORE_POLICY_NAMES",
    "score_policy_id",
    "launch_update_ehat_geom_lut",
]
