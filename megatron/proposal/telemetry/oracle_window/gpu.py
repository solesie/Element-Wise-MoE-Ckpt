"""Bounded GPU primitives for the oracle saliency window.

The only model-sized allocation owned here is one packed FP32 ``theta_cp``
shadow.  Scores are either written to a caller-owned bounded sample buffer or
reduced directly into scalar counts; a dense model-sized score tensor is never
materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .fragments import OracleFragment, PackedFragmentLayout
from .lut import DomainLut

try:
    import triton
    import triton.language as tl

    from ...score_policy import _lookup_h_from_lut

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised by CPU-only installations.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


_KERNEL_BLOCK = 256
MAX_BOUNDED_SCORE_ELEMENTS = 1 << 16
UINT32_MAX = (1 << 32) - 1


def _mix32_cpu(value: int) -> int:
    value &= UINT32_MAX
    value ^= value >> 16
    value = (value * 0x7FEB352D) & UINT32_MAX
    value ^= value >> 15
    value = (value * 0x846CA68B) & UINT32_MAX
    value ^= value >> 16
    return value & UINT32_MAX


def _mix64_cpu(value: int) -> int:
    value &= (1 << 64) - 1
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    value ^= value >> 31
    return value & ((1 << 64) - 1)


def canonical_element_id(
    canonical_parameter_id: int,
    parameter_element_offset: int,
) -> int:
    """Return the signed-int64 identity written to rank telemetry shards."""

    offset = int(parameter_element_offset)
    if offset < 0:
        raise ValueError("parameter element offset must be non-negative")
    value = (
        int(canonical_parameter_id)
        ^ _mix64_cpu(offset + 0x9E3779B97F4A7C15)
    ) & ((1 << 64) - 1)
    return value if value < (1 << 63) else value - (1 << 64)


def deterministic_block_element(
    *,
    sample_index: int,
    n_samples: int,
    n_blocks: int,
    block_elements: int,
    fragment_numel: int,
    canonical_parameter_id: int,
    round_salt: int,
) -> tuple[int, int]:
    """CPU mirror of the GPU interval-stratified sampler."""

    sid = int(sample_index)
    samples = int(n_samples)
    blocks = int(n_blocks)
    block_size = int(block_elements)
    n = int(fragment_numel)
    if not (0 <= sid < samples <= blocks):
        raise ValueError("sample_index must lie in 0 <= i < n_samples <= n_blocks")
    if block_size <= 0 or n <= 0:
        raise ValueError("block_elements and fragment_numel must be positive")
    identity = int(canonical_parameter_id)
    lo = identity & UINT32_MAX
    hi = (identity >> 32) & UINT32_MAX
    round_id = int(round_salt)
    if round_id < 0:
        raise ValueError("round_salt must be non-negative")
    round_id &= UINT32_MAX
    hashed = _mix32_cpu(sid ^ lo ^ _mix32_cpu(hi))
    stratum_lo = (sid * blocks) // samples
    stratum_hi = ((sid + 1) * blocks) // samples
    element_lo = stratum_lo * block_size
    element_hi = min(stratum_hi * block_size, n)
    width = element_hi - element_lo
    if width <= 0:
        raise ValueError("empty block-sampling stratum")
    # This affine step is coprime to ``width`` for every positive integer:
    #   odd or divisible by 4 -> floor(width / 2) + 1
    #   width == 2 (mod 4)    -> floor(width / 2) + 2
    # Consequently consecutive round salts traverse a full permutation of the
    # stratum before repeating.  The roughly half-stratum stride also avoids
    # treating adjacent parameter elements as independent CDF rounds.
    stride = width // 2 + (2 if width % 4 == 2 else 1)
    element = element_lo + (hashed + round_id * stride) % width
    block = element // block_size
    return int(block), int(element)


@dataclass(frozen=True)
class FragmentSamplePlan:
    fragment_index: int
    population_elements: int
    population_blocks: int
    sampled_blocks: int
    output_offset: int
    sample_population_weights: tuple[float, ...]


@dataclass(frozen=True)
class BlockSamplingPlan:
    domain: str
    block_elements: int
    population_elements: int
    population_blocks: int
    requested_sample_blocks: int
    sampled_blocks: int
    fragments: tuple[FragmentSamplePlan, ...]

    @property
    def output_elements(self) -> int:
        return int(self.sampled_blocks)

    def population_weights(self) -> tuple[float, ...]:
        weights: list[float] = []
        for part in self.fragments:
            if len(part.sample_population_weights) != int(part.sampled_blocks):
                raise RuntimeError("fragment sample weights do not match its strata")
            weights.extend(float(value) for value in part.sample_population_weights)
        return tuple(weights)


def _proportional_block_allocation(
    block_counts: Sequence[int],
    requested: int,
) -> list[int]:
    counts = [int(value) for value in block_counts]
    if any(value <= 0 for value in counts):
        raise ValueError("every sampled fragment must contain at least one block")
    target = min(int(requested), sum(counts))
    if target <= 0:
        raise ValueError("sample_blocks must be positive")
    allocation = [0] * len(counts)

    # If capacity permits, cover every fragment once before proportional
    # allocation.  This avoids silently omitting small tensors from a domain.
    if target >= len(counts):
        allocation = [1] * len(counts)
        target_left = target - len(counts)
        remaining = [value - 1 for value in counts]
    else:
        target_left = target
        remaining = list(counts)
    remaining_total = sum(remaining)
    if target_left <= 0 or remaining_total <= 0:
        return allocation

    exact = [target_left * value / remaining_total for value in remaining]
    floors = [min(remaining[i], int(math.floor(exact[i]))) for i in range(len(counts))]
    for i, value in enumerate(floors):
        allocation[i] += value
    left = target_left - sum(floors)
    priority = sorted(
        range(len(counts)),
        key=lambda i: (exact[i] - floors[i], remaining[i], -i),
        reverse=True,
    )
    while left > 0:
        progressed = False
        for i in priority:
            if allocation[i] >= counts[i]:
                continue
            allocation[i] += 1
            left -= 1
            progressed = True
            if left == 0:
                break
        if not progressed:
            raise AssertionError("sample allocation cannot satisfy target")
    return allocation


def build_block_sampling_plan(
    layout: PackedFragmentLayout,
    *,
    block_elements: int,
    sample_blocks: int,
) -> BlockSamplingPlan:
    """Allocate block strata in proportion to each fragment population."""

    if int(block_elements) <= 0:
        raise ValueError("block_elements must be positive")
    if not layout.fragments or layout.total_numel <= 0:
        raise ValueError("sampling requires a non-empty fragment layout")
    block_counts = [
        (int(fragment.numel) + int(block_elements) - 1) // int(block_elements)
        for fragment in layout.fragments
    ]
    allocation = _proportional_block_allocation(block_counts, int(sample_blocks))
    parts: list[FragmentSamplePlan] = []
    output_offset = 0
    for index, (fragment, n_blocks, n_samples) in enumerate(
        zip(layout.fragments, block_counts, allocation)
    ):
        if n_samples <= 0:
            continue
        sample_weights: list[float] = []
        for sample_index in range(int(n_samples)):
            stratum_lo = (sample_index * int(n_blocks)) // int(n_samples)
            stratum_hi = ((sample_index + 1) * int(n_blocks)) // int(n_samples)
            element_lo = stratum_lo * int(block_elements)
            element_hi = min(
                stratum_hi * int(block_elements),
                int(fragment.numel),
            )
            width = int(element_hi) - int(element_lo)
            if width <= 0:
                raise AssertionError("sampling plan produced an empty stratum")
            # One uniform draw represents exactly this stratum's population.
            sample_weights.append(float(width))
        parts.append(
            FragmentSamplePlan(
                fragment_index=int(index),
                population_elements=int(fragment.numel),
                population_blocks=int(n_blocks),
                sampled_blocks=int(n_samples),
                output_offset=int(output_offset),
                sample_population_weights=tuple(sample_weights),
            )
        )
        output_offset += int(n_samples)
    return BlockSamplingPlan(
        domain=layout.domain,
        block_elements=int(block_elements),
        population_elements=int(layout.total_numel),
        population_blocks=int(sum(block_counts)),
        requested_sample_blocks=int(sample_blocks),
        sampled_blocks=int(output_offset),
        fragments=tuple(parts),
    )


@dataclass(eq=False)
class PackedFp32Shadow:
    domain: str
    values: torch.Tensor
    total_numel: int
    initialized: bool = False

    @classmethod
    def allocate(cls, layout: PackedFragmentLayout) -> "PackedFp32Shadow":
        if not layout.fragments or layout.total_numel <= 0:
            raise ValueError("cannot allocate a shadow for an empty layout")
        device = layout.fragments[0].p.device
        if device.type != "cuda":
            raise ValueError("packed oracle shadow requires CUDA FP32 fragments")
        for fragment in layout.fragments:
            if fragment.p.device != device or fragment.p.dtype != torch.float32:
                raise ValueError("all shadow source fragments must be same-device FP32")
        return cls(
            domain=layout.domain,
            values=torch.empty(
                int(layout.total_numel), dtype=torch.float32, device=device,
            ),
            total_numel=int(layout.total_numel),
            initialized=False,
        )

    @torch.no_grad()
    def begin(self, layout: PackedFragmentLayout) -> None:
        """Copy the exact active FP32 master so every initial residual is zero."""

        _validate_shadow_layout(self, layout)
        _require_triton()
        for fragment in layout.fragments:
            destination = self.values.narrow(
                0, int(fragment.packed_offset), int(fragment.numel),
            )
            _launch_copy(fragment.p, destination)
        self.initialized = True


@dataclass(eq=False)
class SampleBuffers:
    raw: torch.Tensor
    clipped: torch.Tensor
    packed_index: torch.Tensor
    canonical_element_id: torch.Tensor
    fragment_index: torch.Tensor
    capacity: int

    @classmethod
    def allocate(cls, capacity: int, *, device: torch.device) -> "SampleBuffers":
        count = int(capacity)
        if count <= 0:
            raise ValueError("sample capacity must be positive")
        return cls(
            raw=torch.empty(count, dtype=torch.float32, device=device),
            clipped=torch.empty(count, dtype=torch.float32, device=device),
            packed_index=torch.empty(count, dtype=torch.int64, device=device),
            canonical_element_id=torch.empty(
                count, dtype=torch.int64, device=device,
            ),
            fragment_index=torch.empty(count, dtype=torch.int32, device=device),
            capacity=count,
        )


@dataclass(eq=False)
class FullCountWorkspace:
    """Scalar full-scan counts; clipped selection uses greater/equal only."""

    counters: torch.Tensor

    GREATER = 0
    EQUAL = 1
    RAW_NEGATIVE = 2
    RAW_ZERO = 3
    RAW_POSITIVE = 4
    NONFINITE = 5

    @classmethod
    def allocate(cls, *, device: torch.device) -> "FullCountWorkspace":
        return cls(counters=torch.zeros(6, dtype=torch.int64, device=device))

    def reset(self) -> None:
        self.counters.zero_()


@dataclass(eq=False)
class SelectionCountWorkspace:
    count: torch.Tensor

    @classmethod
    def allocate(cls, *, device: torch.device) -> "SelectionCountWorkspace":
        return cls(count=torch.zeros((), dtype=torch.int64, device=device))

    def reset(self) -> None:
        self.count.zero_()


def score_reference(
    *,
    p: torch.Tensor,
    m: torch.Tensor,
    h_abs: torch.Tensor,
    theta_cp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small CPU/GPU reference used by focused correctness tests."""

    p32, m32, h32, theta32 = torch.broadcast_tensors(
        p.to(torch.float32),
        m.to(torch.float32),
        h_abs.to(torch.float32).abs(),
        theta_cp.to(torch.float32),
    )
    e = p32 - theta32
    raw = -m32 * e + 0.5 * h32 * e * e
    return raw, torch.clamp_min(raw, 0.0)


def _require_triton() -> None:
    if not _HAS_TRITON:
        raise RuntimeError("oracle-window GPU primitives require Triton")


def _validate_shadow_layout(
    shadow: PackedFp32Shadow,
    layout: PackedFragmentLayout,
) -> None:
    if shadow.domain != layout.domain or shadow.total_numel != layout.total_numel:
        raise ValueError("packed shadow and fragment layout do not match")
    if int(shadow.values.numel()) != int(layout.total_numel):
        raise ValueError("packed shadow storage has the wrong size")


def _validate_fragment_lut(fragment: OracleFragment, lut: DomainLut) -> None:
    if fragment.domain != lut.domain:
        raise ValueError(
            f"cannot score {fragment.domain} fragment with {lut.domain} LUT"
        )
    if fragment.p.device != lut.refs.h_lut.values.device:
        raise ValueError("fragment and domain LUT devices do not match")


if _HAS_TRITON:

    @triton.jit
    def _mix32_gpu(value):
        value = value.to(tl.uint32)
        value = value ^ (value >> 16)
        value = value * 0x7FEB352D
        value = value ^ (value >> 15)
        value = value * 0x846CA68B
        value = value ^ (value >> 16)
        return value

    @triton.jit
    def _mix64_gpu(value):
        value = value.to(tl.uint64)
        value = value ^ (value >> 30)
        value = value * 0xBF58476D1CE4E5B9
        value = value ^ (value >> 27)
        value = value * 0x94D049BB133111EB
        value = value ^ (value >> 31)
        return value

    @triton.jit
    def _raw_and_clipped_score(
        p,
        m,
        v,
        theta_cp,
        h_lut_ptr,
        log_v_min_ptr,
        inv_log_v_width_ptr,
        H_LUT_BINS: tl.constexpr,
    ):
        h_abs = _lookup_h_from_lut(
            v.to(tl.float32),
            h_lut_ptr,
            log_v_min_ptr,
            inv_log_v_width_ptr,
            H_LUT_BINS,
        )
        residual = p.to(tl.float32) - theta_cp.to(tl.float32)
        raw = -m.to(tl.float32) * residual + 0.5 * h_abs * residual * residual
        return raw, tl.maximum(raw, 0.0)

    @triton.jit
    def _copy_fp32_kernel(src_ptr, dst_ptr, n_elems, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elems
        value = tl.load(src_ptr + offsets, mask=mask).to(tl.float32)
        tl.store(dst_ptr + offsets, value, mask=mask)

    @triton.jit
    def _compute_scores_kernel(
        p_ptr,
        m_ptr,
        v_ptr,
        shadow_ptr,
        raw_out_ptr,
        clipped_out_ptr,
        fragment_start,
        output_count,
        h_lut_ptr,
        log_v_min_ptr,
        inv_log_v_width_ptr,
        H_LUT_BINS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        output_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = output_offset < output_count
        source_offset = fragment_start + output_offset
        p = tl.load(p_ptr + source_offset, mask=mask, other=0.0)
        m = tl.load(m_ptr + source_offset, mask=mask, other=0.0)
        v = tl.load(v_ptr + source_offset, mask=mask, other=0.0)
        theta = tl.load(shadow_ptr + source_offset, mask=mask, other=0.0)
        raw, clipped = _raw_and_clipped_score(
            p, m, v, theta,
            h_lut_ptr, log_v_min_ptr, inv_log_v_width_ptr,
            H_LUT_BINS,
        )
        tl.store(raw_out_ptr + output_offset, raw, mask=mask)
        tl.store(clipped_out_ptr + output_offset, clipped, mask=mask)

    @triton.jit
    def _sample_scores_kernel(
        p_ptr,
        m_ptr,
        v_ptr,
        shadow_ptr,
        raw_out_ptr,
        clipped_out_ptr,
        packed_index_out_ptr,
        canonical_element_id_out_ptr,
        fragment_index_out_ptr,
        fragment_numel,
        parameter_offset,
        packed_base,
        fragment_index,
        output_base,
        n_samples,
        n_blocks,
        round_salt,
        canonical_lo,
        canonical_hi,
        h_lut_ptr,
        log_v_min_ptr,
        inv_log_v_width_ptr,
        H_LUT_BINS: tl.constexpr,
        BLOCK_ELEMENTS: tl.constexpr,
    ):
        sample_id = tl.program_id(0)
        identity_mix = _mix32_gpu(canonical_hi)
        hashed = _mix32_gpu(
            sample_id.to(tl.uint32)
            ^ canonical_lo.to(tl.uint32)
            ^ identity_mix
        )
        stratum_lo = (sample_id * n_blocks) // n_samples
        stratum_hi = ((sample_id + 1) * n_blocks) // n_samples
        element_lo = stratum_lo * BLOCK_ELEMENTS
        element_hi = tl.minimum(stratum_hi * BLOCK_ELEMENTS, fragment_numel)
        width = tl.maximum(element_hi - element_lo, 1)
        # Coprime affine stride: consecutive salts form a permutation of every
        # stratum before repeating, while avoiding adjacent-element rounds.
        stride = width // 2 + tl.where((width % 4) == 2, 2, 1)
        source_offset = element_lo + (
            (
                hashed.to(tl.uint64)
                + round_salt.to(tl.uint64) * stride.to(tl.uint64)
            ) % width.to(tl.uint64)
        ).to(tl.int64)

        p = tl.load(p_ptr + source_offset)
        m = tl.load(m_ptr + source_offset)
        v = tl.load(v_ptr + source_offset)
        theta = tl.load(shadow_ptr + source_offset)
        raw, clipped = _raw_and_clipped_score(
            p, m, v, theta,
            h_lut_ptr, log_v_min_ptr, inv_log_v_width_ptr,
            H_LUT_BINS,
        )
        output = output_base + sample_id
        tl.store(raw_out_ptr + output, raw)
        tl.store(clipped_out_ptr + output, clipped)
        tl.store(packed_index_out_ptr + output, packed_base + source_offset)
        canonical_parameter = (
            canonical_lo.to(tl.uint64)
            | (canonical_hi.to(tl.uint64) << 32)
        )
        canonical_offset = parameter_offset.to(tl.uint64) + source_offset.to(tl.uint64)
        global_element_id = canonical_parameter ^ _mix64_gpu(
            canonical_offset + 0x9E3779B97F4A7C15
        )
        tl.store(canonical_element_id_out_ptr + output, global_element_id)
        tl.store(fragment_index_out_ptr + output, fragment_index)

    @triton.jit
    def _full_count_kernel(
        p_ptr,
        m_ptr,
        v_ptr,
        shadow_ptr,
        counters_ptr,
        n_elems,
        tau,
        h_lut_ptr,
        log_v_min_ptr,
        inv_log_v_width_ptr,
        H_LUT_BINS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elems
        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
        theta = tl.load(shadow_ptr + offsets, mask=mask, other=0.0)
        raw, clipped = _raw_and_clipped_score(
            p, m, v, theta,
            h_lut_ptr, log_v_min_ptr, inv_log_v_width_ptr,
            H_LUT_BINS,
        )
        finite = mask & (raw == raw) & (tl.abs(raw) < float("inf"))
        tl.atomic_add(counters_ptr + 0, tl.sum(finite & (clipped > tau), axis=0))
        tl.atomic_add(counters_ptr + 1, tl.sum(finite & (clipped == tau), axis=0))
        tl.atomic_add(counters_ptr + 2, tl.sum(finite & (raw < 0.0), axis=0))
        tl.atomic_add(counters_ptr + 3, tl.sum(finite & (raw == 0.0), axis=0))
        tl.atomic_add(counters_ptr + 4, tl.sum(finite & (raw > 0.0), axis=0))
        tl.atomic_add(counters_ptr + 5, tl.sum(mask & ~finite, axis=0))

    @triton.jit
    def _update_selected_kernel(
        p_ptr,
        m_ptr,
        v_ptr,
        shadow_ptr,
        selected_count_ptr,
        n_elems,
        parameter_offset,
        tau,
        tie_hash_limit,
        tie_salt,
        canonical_lo,
        canonical_hi,
        h_lut_ptr,
        log_v_min_ptr,
        inv_log_v_width_ptr,
        H_LUT_BINS: tl.constexpr,
        TAKE_TIES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elems
        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
        theta = tl.load(shadow_ptr + offsets, mask=mask, other=0.0)
        raw, clipped = _raw_and_clipped_score(
            p, m, v, theta,
            h_lut_ptr, log_v_min_ptr, inv_log_v_width_ptr,
            H_LUT_BINS,
        )
        global_parameter_offset = parameter_offset + offsets
        tie_hash = _mix32_gpu(
            global_parameter_offset.to(tl.uint32)
            ^ canonical_lo.to(tl.uint32)
            ^ _mix32_gpu(canonical_hi)
            ^ tie_salt.to(tl.uint32)
        )
        if TAKE_TIES:
            take_tie = tie_hash.to(tl.uint64) <= tie_hash_limit.to(tl.uint64)
        else:
            take_tie = tl.zeros((BLOCK,), dtype=tl.int1)
        finite = (raw == raw) & (tl.abs(raw) < float("inf"))
        selected = mask & finite & (
            (clipped > tau) | ((clipped == tau) & take_tie)
        )
        tl.store(shadow_ptr + offsets, p.to(tl.float32), mask=selected)
        tl.atomic_add(selected_count_ptr, tl.sum(selected, axis=0))


@torch.no_grad()
def _launch_copy(source: torch.Tensor, destination: torch.Tensor) -> None:
    _require_triton()
    n = int(source.numel())
    if n != int(destination.numel()):
        raise ValueError("copy source/destination sizes differ")
    _copy_fp32_kernel[(triton.cdiv(n, _KERNEL_BLOCK),)](
        source, destination, n, BLOCK=_KERNEL_BLOCK,
    )


@torch.no_grad()
def launch_compute_fragment_scores(
    *,
    fragment: OracleFragment,
    shadow: PackedFp32Shadow,
    lut: DomainLut,
    fragment_start: int,
    count: int,
    raw_out: torch.Tensor,
    clipped_out: torch.Tensor,
) -> None:
    """Compute one explicitly bounded score slice for diagnostics/tests."""

    _require_triton()
    _validate_fragment_lut(fragment, lut)
    start, n = int(fragment_start), int(count)
    if n <= 0 or n > MAX_BOUNDED_SCORE_ELEMENTS:
        raise ValueError(
            f"bounded score count must be in [1, {MAX_BOUNDED_SCORE_ELEMENTS}]"
        )
    if start < 0 or start + n > int(fragment.numel):
        raise ValueError("bounded score slice exceeds fragment")
    if int(raw_out.numel()) < n or int(clipped_out.numel()) < n:
        raise ValueError("bounded score output buffers are too small")
    shadow_view = shadow.values.narrow(
        0, int(fragment.packed_offset), int(fragment.numel),
    )
    _compute_scores_kernel[(triton.cdiv(n, _KERNEL_BLOCK),)](
        fragment.p,
        fragment.m,
        fragment.v,
        shadow_view,
        raw_out,
        clipped_out,
        start,
        n,
        lut.refs.h_lut.values,
        lut.refs.h_lut.log_v_min,
        lut.refs.h_lut.inv_log_v_width,
        H_LUT_BINS=int(lut.refs.h_lut.values.numel()),
        BLOCK=_KERNEL_BLOCK,
    )


@torch.no_grad()
def launch_sample_scores(
    *,
    layout: PackedFragmentLayout,
    shadow: PackedFp32Shadow,
    lut: DomainLut,
    plan: BlockSamplingPlan,
    round_salt: int,
    output: SampleBuffers,
) -> None:
    """Fill one deterministic, block-stratified multi-fragment sample round."""

    _require_triton()
    _validate_shadow_layout(shadow, layout)
    if not shadow.initialized:
        raise RuntimeError("packed shadow must begin from exact p before sampling")
    if plan.domain != layout.domain or lut.domain != layout.domain:
        raise ValueError("sampling layout, plan, shadow, and LUT domains must match")
    if output.capacity < plan.output_elements:
        raise ValueError("sample output capacity is smaller than the sampling plan")
    for part in plan.fragments:
        fragment = layout.fragments[int(part.fragment_index)]
        _validate_fragment_lut(fragment, lut)
        shadow_view = shadow.values.narrow(
            0, int(fragment.packed_offset), int(fragment.numel),
        )
        id_lo, id_hi = fragment.canonical_id_words
        _sample_scores_kernel[(int(part.sampled_blocks),)](
            fragment.p,
            fragment.m,
            fragment.v,
            shadow_view,
            output.raw,
            output.clipped,
            output.packed_index,
            output.canonical_element_id,
            output.fragment_index,
            int(fragment.numel),
            int(fragment.parameter_offset),
            int(fragment.packed_offset),
            int(part.fragment_index),
            int(part.output_offset),
            int(part.sampled_blocks),
            int(part.population_blocks),
            int(round_salt) & UINT32_MAX,
            int(id_lo),
            int(id_hi),
            lut.refs.h_lut.values,
            lut.refs.h_lut.log_v_min,
            lut.refs.h_lut.inv_log_v_width,
            H_LUT_BINS=int(lut.refs.h_lut.values.numel()),
            BLOCK_ELEMENTS=int(plan.block_elements),
        )


@torch.no_grad()
def launch_full_count(
    *,
    layout: PackedFragmentLayout,
    shadow: PackedFp32Shadow,
    lut: DomainLut,
    tau: float,
    workspace: FullCountWorkspace,
) -> None:
    """Exact full-population clipped count plus signed raw diagnostics."""

    _require_triton()
    _validate_shadow_layout(shadow, layout)
    if not shadow.initialized:
        raise RuntimeError("packed shadow must begin from exact p before counting")
    if not math.isfinite(float(tau)) or float(tau) < 0.0:
        raise ValueError("selection tau must be finite and non-negative")
    if workspace.counters.device != shadow.values.device:
        raise ValueError("full-count workspace is on the wrong device")
    for fragment in layout.fragments:
        _validate_fragment_lut(fragment, lut)
        shadow_view = shadow.values.narrow(
            0, int(fragment.packed_offset), int(fragment.numel),
        )
        _full_count_kernel[(triton.cdiv(int(fragment.numel), _KERNEL_BLOCK),)](
            fragment.p,
            fragment.m,
            fragment.v,
            shadow_view,
            workspace.counters,
            int(fragment.numel),
            float(tau),
            lut.refs.h_lut.values,
            lut.refs.h_lut.log_v_min,
            lut.refs.h_lut.inv_log_v_width,
            H_LUT_BINS=int(lut.refs.h_lut.values.numel()),
            BLOCK=_KERNEL_BLOCK,
        )


@torch.no_grad()
def launch_update_selected(
    *,
    layout: PackedFragmentLayout,
    shadow: PackedFp32Shadow,
    lut: DomainLut,
    tau: float,
    tie_hash_limit: int,
    tie_salt: int,
    workspace: SelectionCountWorkspace,
) -> None:
    """Advance ``theta_cp`` only for clipped-score selections.

    Elements strictly above ``tau`` are always selected.  Equality mass uses
    the stable canonical element identity and an inclusive 32-bit hash limit;
    ``-1`` selects no ties and ``UINT32_MAX`` selects every tie.  A coordinator
    can use the preceding exact ``greater/equal`` count to choose the limit.
    """

    _require_triton()
    _validate_shadow_layout(shadow, layout)
    if not shadow.initialized:
        raise RuntimeError("packed shadow must begin from exact p before updating")
    limit = int(tie_hash_limit)
    if limit < -1 or limit > UINT32_MAX:
        raise ValueError(f"tie_hash_limit must lie in [-1, {UINT32_MAX}]")
    if not math.isfinite(float(tau)) or float(tau) < 0.0:
        raise ValueError("selection tau must be finite and non-negative")
    kernel_limit = 0 if limit < 0 else limit
    for fragment in layout.fragments:
        _validate_fragment_lut(fragment, lut)
        shadow_view = shadow.values.narrow(
            0, int(fragment.packed_offset), int(fragment.numel),
        )
        id_lo, id_hi = fragment.canonical_id_words
        _update_selected_kernel[(
            triton.cdiv(int(fragment.numel), _KERNEL_BLOCK),
        )](
            fragment.p,
            fragment.m,
            fragment.v,
            shadow_view,
            workspace.count,
            int(fragment.numel),
            int(fragment.parameter_offset),
            float(tau),
            int(kernel_limit),
            int(tie_salt) & UINT32_MAX,
            int(id_lo),
            int(id_hi),
            lut.refs.h_lut.values,
            lut.refs.h_lut.log_v_min,
            lut.refs.h_lut.inv_log_v_width,
            H_LUT_BINS=int(lut.refs.h_lut.values.numel()),
            TAKE_TIES=bool(limit >= 0),
            BLOCK=_KERNEL_BLOCK,
        )


__all__ = [
    "BlockSamplingPlan",
    "FragmentSamplePlan",
    "FullCountWorkspace",
    "MAX_BOUNDED_SCORE_ELEMENTS",
    "PackedFp32Shadow",
    "SampleBuffers",
    "SelectionCountWorkspace",
    "UINT32_MAX",
    "build_block_sampling_plan",
    "canonical_element_id",
    "deterministic_block_element",
    "launch_compute_fragment_scores",
    "launch_full_count",
    "launch_sample_scores",
    "launch_update_selected",
    "score_reference",
]
