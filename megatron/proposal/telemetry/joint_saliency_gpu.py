"""GPU-only aggregation for oracle-time joint saliency telemetry.

The collector intentionally never materializes a model-sized score tensor.
Each Triton program computes the production theta-checkpoint score in
registers and folds it into a fixed, init-time workspace.  Only the compressed
workspace is copied to pinned host memory at an explicitly requested oracle
iteration.

This is removable experiment telemetry.  It must not feed the production
cutoff controller or record writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised on CPU-only test hosts.
    triton = None
    tl = None
    _HAS_TRITON = False


# int_stats columns.  Keep this schema next to the kernel that writes it.
TOTAL = 0
POSITIVE = 1
ZERO = 2
NONFINITE = 3
CLIPPED_LOW = 4
CLIPPED_HIGH = 5
N_INT_STATS = 6


if _HAS_TRITON:
    from ..score_policy import _proposal_element_score_from_theta_cp

    @triton.jit
    def _age2_load_code(age_words_ptr, global_elem_idx, valid):
        word_idx = (global_elem_idx >> 4).to(tl.int64)
        shift = ((global_elem_idx & 15) << 1).to(tl.int32)
        word = tl.load(
            age_words_ptr + word_idx,
            mask=valid & (global_elem_idx >= 0),
            other=0,
        )
        return (word >> shift) & 3

    @triton.jit
    def _hash_u32(x):
        x = x.to(tl.uint32)
        x = (x ^ (x >> 16)) * tl.full((), 0x7FEB352D, tl.uint32)
        x = (x ^ (x >> 15)) * tl.full((), 0x846CA68B, tl.uint32)
        return x ^ (x >> 16)

    @triton.jit
    def _oldest_age_phase_hit(
        global_elem_idx,
        step_value,
        period: tl.constexpr,
    ):
        x = global_elem_idx.to(tl.uint32) ^ tl.full(
            (), 0x51ED270B, tl.uint32,
        )
        phase = (_hash_u32(x) % period).to(tl.int32)
        return ((step_value.to(tl.int32) + phase) % period) == 0

    @triton.jit
    def _joint_saliency_hist_kernel(
        linear_ptr,
        v_ptr,
        p_ptr,
        theta_cp_ptr,
        age_words_ptr,
        h_lut_ptr,
        log_v_min_ptr,
        inv_log_v_width_ptr,
        entry_elem_start_ptr,
        entry_age_start_ptr,
        entry_group_count_ptr,
        entry_row_ptr,
        hist_count_ptr,
        hist_mass_ptr,
        age_count_ptr,
        force_count_ptr,
        int_stats_ptr,
        score_sum_ptr,
        score_min_log_ptr,
        score_max_log_ptr,
        step_value,
        n_entries,
        LOG10_MIN: tl.constexpr,
        LOG10_MAX: tl.constexpr,
        N_BINS: tl.constexpr,
        G: tl.constexpr,
        H_LUT_BINS: tl.constexpr,
        SCORE_POLICY: tl.constexpr,
        AGE_FORCE_PERIOD: tl.constexpr,
        N_STATS: tl.constexpr,
        TOTAL_COL: tl.constexpr,
        POSITIVE_COL: tl.constexpr,
        ZERO_COL: tl.constexpr,
        NONFINITE_COL: tl.constexpr,
        CLIPPED_LOW_COL: tl.constexpr,
        CLIPPED_HIGH_COL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # The block dimension can exceed CUDA grid.y's 65,535 limit for a
        # single 1.3B/13B expert matrix. Keep it on grid.x; entry count is the
        # small descriptor dimension.
        block = tl.program_id(0)
        entry = tl.program_id(1)
        row = tl.load(entry_row_ptr + entry, mask=entry < n_entries, other=-1)
        n = tl.load(
            entry_group_count_ptr + entry,
            mask=entry < n_entries,
            other=0,
        ).to(tl.int64) * G
        rel = block * BLOCK + tl.arange(0, BLOCK)
        mask = (entry < n_entries) & (row >= 0) & (rel < n)

        elem_start = tl.load(
            entry_elem_start_ptr + entry,
            mask=entry < n_entries,
            other=0,
        ).to(tl.int64)
        age_start = tl.load(
            entry_age_start_ptr + entry,
            mask=entry < n_entries,
            other=0,
        ).to(tl.int64)
        elem_idx = elem_start + rel
        age_idx = age_start + rel

        linear = tl.load(linear_ptr + elem_idx, mask=mask, other=0.0)
        v = tl.load(v_ptr + elem_idx, mask=mask, other=0.0)
        p = tl.load(p_ptr + elem_idx, mask=mask, other=0.0)
        theta_cp = tl.load(theta_cp_ptr + age_idx, mask=mask, other=0.0)
        score = _proposal_element_score_from_theta_cp(
            linear,
            v,
            p,
            theta_cp,
            h_lut_ptr,
            log_v_min_ptr,
            inv_log_v_width_ptr,
            H_LUT_BINS,
            SCORE_POLICY,
        )

        # The production helper is non-negative.  The explicit finite guard
        # keeps NaN/Inf visible in telemetry rather than silently binning it.
        finite = mask & (score == score) & (score < 3.402823466e38)
        positive = finite & (score > 0.0)
        zero = finite & (score == 0.0)
        nonfinite = mask & ~finite

        safe_score = tl.where(positive, score, 1.0)
        log10_score = tl.log(safe_score) * 0.4342944819032518
        below = positive & (log10_score < LOG10_MIN)
        above = positive & (log10_score >= LOG10_MAX)
        scaled = (
            (log10_score - LOG10_MIN)
            * (N_BINS / (LOG10_MAX - LOG10_MIN))
        )
        bin_idx = tl.maximum(
            0,
            tl.minimum(N_BINS - 1, scaled.to(tl.int32)),
        )
        hist_off = row.to(tl.int64) * N_BINS + bin_idx.to(tl.int64)
        tl.atomic_add(hist_count_ptr + hist_off, 1, mask=positive)
        tl.atomic_add(
            hist_mass_ptr + hist_off,
            score.to(tl.float32),
            mask=positive,
        )

        row_valid = (entry < n_entries) & (row >= 0)
        block_total = tl.sum(mask.to(tl.int32), axis=0)
        block_positive = tl.sum(positive.to(tl.int32), axis=0)
        block_zero = tl.sum(zero.to(tl.int32), axis=0)
        block_nonfinite = tl.sum(nonfinite.to(tl.int32), axis=0)
        block_low = tl.sum(below.to(tl.int32), axis=0)
        block_high = tl.sum(above.to(tl.int32), axis=0)
        stats_base = row.to(tl.int64) * N_STATS
        tl.atomic_add(
            int_stats_ptr + stats_base + TOTAL_COL, block_total, mask=row_valid,
        )
        tl.atomic_add(
            int_stats_ptr + stats_base + POSITIVE_COL, block_positive, mask=row_valid,
        )
        tl.atomic_add(int_stats_ptr + stats_base + ZERO_COL, block_zero, mask=row_valid)
        tl.atomic_add(
            int_stats_ptr + stats_base + NONFINITE_COL,
            block_nonfinite,
            mask=row_valid,
        )
        tl.atomic_add(
            int_stats_ptr + stats_base + CLIPPED_LOW_COL, block_low, mask=row_valid,
        )
        tl.atomic_add(
            int_stats_ptr + stats_base + CLIPPED_HIGH_COL, block_high, mask=row_valid,
        )
        tl.atomic_add(
            score_sum_ptr + row,
            tl.sum(tl.where(positive, score, 0.0), axis=0),
            mask=row_valid,
        )

        local_min = tl.min(
            tl.where(positive, log10_score, float("inf")),
            axis=0,
        )
        local_max = tl.max(
            tl.where(positive, log10_score, -float("inf")),
            axis=0,
        )
        tl.atomic_min(
            score_min_log_ptr + row,
            local_min,
            mask=row_valid & (block_positive > 0),
        )
        tl.atomic_max(
            score_max_log_ptr + row,
            local_max,
            mask=row_valid & (block_positive > 0),
        )

        age_code = _age2_load_code(age_words_ptr, age_idx, mask)
        for code in tl.static_range(0, 4):
            count = tl.sum((mask & (age_code == code)).to(tl.int32), axis=0)
            tl.atomic_add(
                age_count_ptr + row * 4 + code,
                count,
                mask=row_valid,
            )
        forced = (
            mask
            & (age_code == 3)
            & _oldest_age_phase_hit(
                age_idx,
                step_value,
                AGE_FORCE_PERIOD,
            )
        )
        tl.atomic_add(
            force_count_ptr + row,
            tl.sum(forced.to(tl.int32), axis=0),
            mask=row_valid,
        )


@dataclass(frozen=True)
class JointSaliencyHostView:
    """Pinned-host snapshot produced by :class:`JointSaliencyWorkspace`."""

    hist_count: torch.Tensor
    hist_mass: torch.Tensor
    age_count: torch.Tensor
    force_count: torch.Tensor
    int_stats: torch.Tensor
    score_sum: torch.Tensor
    score_min_log: torch.Tensor
    score_max_log: torch.Tensor


class JointSaliencyWorkspace:
    """Fixed GPU and pinned-host buffers allocated once during proposal init."""

    def __init__(
        self,
        *,
        n_rows: int,
        n_bins: int,
        device: torch.device,
    ) -> None:
        if n_rows <= 0:
            raise ValueError("joint saliency workspace requires at least one row")
        if n_bins < 16:
            raise ValueError("joint saliency histogram requires at least 16 bins")
        if device.type != "cuda":
            raise ValueError("joint saliency GPU workspace requires a CUDA device")
        self.n_rows = int(n_rows)
        self.n_bins = int(n_bins)
        self.device = device

        self.hist_count = torch.zeros(
            (n_rows, n_bins), dtype=torch.int32, device=device,
        )
        self.hist_mass = torch.zeros(
            (n_rows, n_bins), dtype=torch.float32, device=device,
        )
        self.age_count = torch.zeros(
            (n_rows, 4), dtype=torch.int32, device=device,
        )
        self.force_count = torch.zeros(n_rows, dtype=torch.int32, device=device)
        self.int_stats = torch.zeros(
            (n_rows, N_INT_STATS), dtype=torch.int32, device=device,
        )
        self.score_sum = torch.zeros(n_rows, dtype=torch.float32, device=device)
        self.score_min_log = torch.full(
            (n_rows,), float("inf"), dtype=torch.float32, device=device,
        )
        self.score_max_log = torch.full(
            (n_rows,), -float("inf"), dtype=torch.float32, device=device,
        )

        self._host = JointSaliencyHostView(
            hist_count=torch.empty_like(self.hist_count, device="cpu", pin_memory=True),
            hist_mass=torch.empty_like(self.hist_mass, device="cpu", pin_memory=True),
            age_count=torch.empty_like(self.age_count, device="cpu", pin_memory=True),
            force_count=torch.empty_like(self.force_count, device="cpu", pin_memory=True),
            int_stats=torch.empty_like(self.int_stats, device="cpu", pin_memory=True),
            score_sum=torch.empty_like(self.score_sum, device="cpu", pin_memory=True),
            score_min_log=torch.empty_like(
                self.score_min_log, device="cpu", pin_memory=True,
            ),
            score_max_log=torch.empty_like(
                self.score_max_log, device="cpu", pin_memory=True,
            ),
        )
        self._copy_done = torch.cuda.Event()

    def reset(self) -> None:
        self.hist_count.zero_()
        self.hist_mass.zero_()
        self.age_count.zero_()
        self.force_count.zero_()
        self.int_stats.zero_()
        self.score_sum.zero_()
        self.score_min_log.fill_(float("inf"))
        self.score_max_log.fill_(-float("inf"))

    def copy_to_host(self, stream: torch.cuda.Stream) -> JointSaliencyHostView:
        for name in JointSaliencyHostView.__dataclass_fields__:
            getattr(self._host, name).copy_(
                getattr(self, name), non_blocking=True,
            )
        self._copy_done.record(stream)
        self._copy_done.synchronize()
        return self._host

    def owned_cuda_tensors(self) -> Dict[str, torch.Tensor]:
        return {
            name: getattr(self, name)
            for name in JointSaliencyHostView.__dataclass_fields__
        }

    @property
    def workspace_bytes(self) -> int:
        return sum(
            int(t.numel()) * int(t.element_size())
            for t in self.owned_cuda_tensors().values()
        )


def launch_joint_saliency_group(
    *,
    linear: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    theta_cp: torch.Tensor,
    age_words: torch.Tensor,
    h_lut_values: torch.Tensor,
    h_lut_log_v_min: torch.Tensor,
    h_lut_inv_log_v_width: torch.Tensor,
    entry_elem_start: torch.Tensor,
    entry_age_start: torch.Tensor,
    entry_group_count: torch.Tensor,
    entry_row: torch.Tensor,
    workspace: JointSaliencyWorkspace,
    step: int,
    group_size: int,
    score_policy_id: int,
    age_force_period: int,
    log10_min: float,
    log10_max: float,
    max_group_count: int,
) -> None:
    """Launch one all-element fused aggregation pass for an optimizer group."""

    if not _HAS_TRITON:
        raise RuntimeError("joint saliency telemetry requires Triton")
    if theta_cp.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "joint saliency production residual source requires 2-byte theta_cp"
        )
    n_entries = int(entry_group_count.numel())
    if n_entries <= 0:
        return
    if int(entry_row.numel()) != n_entries:
        raise RuntimeError("joint saliency entry-row map is misaligned")
    max_groups = int(max_group_count)
    if max_groups <= 0:
        return
    block = 256
    max_elements = max_groups * max(1, int(group_size))
    if n_entries > 65535:
        raise RuntimeError(
            "joint saliency entry count exceeds CUDA grid.y capacity"
        )
    grid = (triton.cdiv(max_elements, block), n_entries)
    _joint_saliency_hist_kernel[grid](
        linear,
        v,
        p,
        theta_cp,
        age_words,
        h_lut_values,
        h_lut_log_v_min,
        h_lut_inv_log_v_width,
        entry_elem_start,
        entry_age_start,
        entry_group_count,
        entry_row,
        workspace.hist_count,
        workspace.hist_mass,
        workspace.age_count,
        workspace.force_count,
        workspace.int_stats,
        workspace.score_sum,
        workspace.score_min_log,
        workspace.score_max_log,
        int(step),
        n_entries,
        LOG10_MIN=float(log10_min),
        LOG10_MAX=float(log10_max),
        N_BINS=int(workspace.n_bins),
        G=max(1, int(group_size)),
        H_LUT_BINS=int(h_lut_values.numel()),
        SCORE_POLICY=int(score_policy_id),
        AGE_FORCE_PERIOD=max(1, int(age_force_period)),
        N_STATS=N_INT_STATS,
        TOTAL_COL=TOTAL,
        POSITIVE_COL=POSITIVE,
        ZERO_COL=ZERO,
        NONFINITE_COL=NONFINITE,
        CLIPPED_LOW_COL=CLIPPED_LOW,
        CLIPPED_HIGH_COL=CLIPPED_HIGH,
        BLOCK=block,
    )


__all__ = [
    "CLIPPED_HIGH",
    "CLIPPED_LOW",
    "JointSaliencyHostView",
    "JointSaliencyWorkspace",
    "NONFINITE",
    "N_INT_STATS",
    "POSITIVE",
    "TOTAL",
    "ZERO",
    "launch_joint_saliency_group",
]
