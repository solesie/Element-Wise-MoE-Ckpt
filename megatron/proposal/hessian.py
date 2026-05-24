"""Diagonal Hessian estimator for the element-wise importance score.

* :func:`compute_hessian_finite_diff` — central-difference Hessian-vector
  product (production). Perturbs the DS-managed expert lp ``.data`` in
  place; lp ``.data`` is a view into ``bf16_groups_flat`` (see
  ``deepspeed/runtime/bf16_optimizer.py:283-286``) so writes reach the
  bytes forward reads under BF16 across all ZeRO stages. θ_0 is preserved
  via a pinned-host snapshot (single source of truth for both ±ε reset and
  post-anchor restore). Uses MoE routing freeze so ±ε forwards see the
  same dispatch.

State-free: all dependencies (model, data iterator, tracked params)
are passed in explicitly.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON_HESSIAN = True
except ImportError:
    _HAS_TRITON_HESSIAN = False


# ---------------------------------------------------------------------------
# Phase 3 — streaming H accumulation kernel.
# ---------------------------------------------------------------------------
# Eliminates the per-K-sample (gp-gm).to(f32) + (z*hz).abs_().mul_() +
# reshape-mean intermediates from the central-diff loop. At 10B-scale
# (ΣP_expert ~ 7B), those temporaries are ~28 GB × 2 = 56 GB transient
# per K-sample; the streaming kernel reads gp/gm/z element-by-element
# (register-level) and writes group-mean BF16 contributions directly
# into h_accum. No tensor of size ΣP is ever allocated.

if _HAS_TRITON_HESSIAN:
    @triton.jit
    def _h_accum_streaming_kernel(
        gp_ptr,                 # *bf16/f32, n_elements (grad at +ε)
        gm_ptr,                 # *bf16/f32, n_elements (grad at -ε)
        z_ptr,                  # *f32,      n_elements (Rademacher ±1)
        h_accum_ptr,            # *bf16,     n_groups (accumulator slab)
        n_groups,
        inv_step,
        inv_k,
        G: tl.constexpr,        # g_blk
        GROUPS_PER_BLOCK: tl.constexpr,
    ):
        """Per-iter K-sample fused reduce:

        For each group g of G elements:
          contrib[g] = (1/G) Σ_{i ∈ g} |z_i · (gp_i - gm_i) · inv_step| · inv_k
          h_accum[g] += BF16(contrib[g])

        No atomics needed: each program owns a disjoint gid range. The
        BF16 RMW (load → cast f32 → add → cast bf16 → store) is safe
        because the outer K-loop is sequential.
        """
        BLOCK: tl.constexpr = G * GROUPS_PER_BLOCK
        pid = tl.program_id(0)
        gid_start = pid * GROUPS_PER_BLOCK
        elem_off = gid_start * G + tl.arange(0, BLOCK)
        elem_mask = elem_off < n_groups * G

        gp = tl.load(gp_ptr + elem_off, mask=elem_mask, other=0.0).to(tl.float32)
        gm = tl.load(gm_ptr + elem_off, mask=elem_mask, other=0.0).to(tl.float32)
        z  = tl.load(z_ptr  + elem_off, mask=elem_mask, other=0.0).to(tl.float32)

        # Fused: |z · (gp-gm) · inv_step| · inv_k, no intermediate alloc.
        contrib = tl.abs(z * (gp - gm) * inv_step) * inv_k

        # Group-reduce: mean over G elements per group.
        contrib_2d = tl.reshape(contrib, (GROUPS_PER_BLOCK, G))
        group_mean = tl.sum(contrib_2d, axis=1) / G

        gid_off = gid_start + tl.arange(0, GROUPS_PER_BLOCK)
        gid_mask = gid_off < n_groups
        # Load existing accumulator (bf16) → f32 → add → bf16 → store.
        h_old = tl.load(h_accum_ptr + gid_off, mask=gid_mask, other=0.0).to(tl.float32)
        h_new = h_old + group_mean
        tl.store(h_accum_ptr + gid_off, h_new.to(tl.bfloat16), mask=gid_mask)


def _h_accum_streaming_for_tensor(
    gp: torch.Tensor,
    gm: torch.Tensor,
    z: torch.Tensor,
    h_accum: torch.Tensor,
    g_blk: int,
    inv_step: float,
    inv_k: float,
) -> None:
    """Launch ``_h_accum_streaming_kernel`` for one (gp, gm, z, h_accum)
    quadruple. n_groups = h_accum.numel(). Same GROUPS_PER_BLOCK
    heuristic as ``_accum_delta_sq_triton`` (target block ≈ 256).
    """
    if not _HAS_TRITON_HESSIAN:
        # Fallback: original chain (kept for non-Triton environments).
        hz = (gp - gm).mul(inv_step).to(torch.float32)
        contrib = (z.to(torch.float32) * hz).abs_().mul_(inv_k)
        n_aligned = (contrib.numel() // g_blk) * g_blk
        if n_aligned > 0:
            h_accum.add_(
                contrib[:n_aligned].reshape(-1, g_blk).mean(dim=1).to(torch.bfloat16)
            )
        del hz, contrib
        return
    n_groups = int(h_accum.numel())
    if n_groups == 0:
        return
    target_block = 256
    gpb = max(1, target_block // max(1, g_blk))
    p = 1
    while p < gpb:
        p *= 2
    gpb = p
    n_progs = (n_groups + gpb - 1) // gpb
    _h_accum_streaming_kernel[(n_progs,)](
        gp, gm, z, h_accum,
        n_groups, float(inv_step), float(inv_k),
        G=int(g_blk), GROUPS_PER_BLOCK=gpb,
    )


if _HAS_TRITON_HESSIAN:
    @triton.jit
    def _h_accum_streaming_from_diff_kernel(
        diff_ptr,               # *bf16/f32, n_elements  (= gp - gm, pre-computed)
        z_ptr,                  # *bf16,     n_elements
        h_accum_ptr,            # *bf16,     n_groups
        n_groups,
        inv_step,
        inv_k,
        G: tl.constexpr,
        GROUPS_PER_BLOCK: tl.constexpr,
    ):
        """Variant of ``_h_accum_streaming_kernel`` that reads a
        pre-computed ``diff = gp - gm`` from a single tensor instead of
        materialising ``gp`` and ``gm`` separately. The ``.grad``
        accumulator trick in the central-diff loop computes ``diff``
        in-place via ``backward(L_plus) + backward(-L_minus)`` — no
        second full grad list is held on GPU."""
        BLOCK: tl.constexpr = G * GROUPS_PER_BLOCK
        pid = tl.program_id(0)
        gid_start = pid * GROUPS_PER_BLOCK
        elem_off = gid_start * G + tl.arange(0, BLOCK)
        elem_mask = elem_off < n_groups * G

        diff = tl.load(diff_ptr + elem_off, mask=elem_mask, other=0.0).to(tl.float32)
        z    = tl.load(z_ptr    + elem_off, mask=elem_mask, other=0.0).to(tl.float32)

        contrib = tl.abs(z * diff * inv_step) * inv_k
        contrib_2d = tl.reshape(contrib, (GROUPS_PER_BLOCK, G))
        group_mean = tl.sum(contrib_2d, axis=1) / G

        gid_off = gid_start + tl.arange(0, GROUPS_PER_BLOCK)
        gid_mask = gid_off < n_groups
        h_old = tl.load(h_accum_ptr + gid_off, mask=gid_mask, other=0.0).to(tl.float32)
        h_new = h_old + group_mean
        tl.store(h_accum_ptr + gid_off, h_new.to(tl.bfloat16), mask=gid_mask)


def _h_accum_streaming_from_diff(
    diff: torch.Tensor,
    z: torch.Tensor,
    h_accum: torch.Tensor,
    g_blk: int,
    inv_step: float,
    inv_k: float,
) -> None:
    """Per-tensor streaming fold consuming a pre-computed ``diff = gp - gm``.
    Used by the central-diff path's ``.grad`` accumulator trick: the
    per-tensor ``.grad`` is exactly ``gp - gm`` after the two-backward
    sequence, so we never materialise a full second grad list."""
    if not _HAS_TRITON_HESSIAN:
        contrib = (z.to(torch.float32) * diff.to(torch.float32)).mul_(inv_step).abs_().mul_(inv_k)
        n_aligned = (contrib.numel() // g_blk) * g_blk
        if n_aligned > 0:
            h_accum.add_(
                contrib[:n_aligned].reshape(-1, g_blk).mean(dim=1).to(torch.bfloat16)
            )
        del contrib
        return
    n_groups = int(h_accum.numel())
    if n_groups == 0:
        return
    target_block = 256
    gpb = max(1, target_block // max(1, g_blk))
    p = 1
    while p < gpb:
        p *= 2
    gpb = p
    n_progs = (n_groups + gpb - 1) // gpb
    _h_accum_streaming_from_diff_kernel[(n_progs,)](
        diff, z, h_accum,
        n_groups, float(inv_step), float(inv_k),
        G=int(g_blk), GROUPS_PER_BLOCK=gpb,
    )


# ---------------------------------------------------------------------------
# Phase 1/2 signed streaming helpers (production FD path).
# ---------------------------------------------------------------------------
# gp and gm are NEVER alive simultaneously — each phase's backward leaves
# expert grads in p.grad, the per-expert loop reduces to group-mean
# (signed), writes to h_accum, and releases the grad. abs is applied
# once after both phases complete (final |group_mean(z · (gp - gm) /
# 2ε)|). Per-call transient ≈ p.numel * 4 B FP32 (one expert at a time).

_Z_SANITY_DONE = False


def _z_seed(rng_seed: int, k: int, e_idx: int) -> int:
    """Deterministic int seed for lazy z. Same (rng_seed, k, e_idx) →
    same seed → torch.Generator → bit-exact same Rademacher. Required
    for FD cancellation: Phase 1 perturb, Phase 1 streaming, Phase 2
    perturb, Phase 2 streaming all call with the same key."""
    a = int(rng_seed) & ((1 << 64) - 1)
    b = (int(k) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    c = (int(e_idx) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    return int((a ^ b ^ c) & ((1 << 63) - 1))


def _lazy_rademacher(p: torch.Tensor, seed: int) -> torch.Tensor:
    """±1 BF16 Rademacher with shape p.shape. Deterministic given seed.
    Caller frees promptly (transient ~ p.numel * 2 B)."""
    gen = torch.Generator(device=p.device).manual_seed(int(seed))
    return (torch.randint(0, 2, p.shape, generator=gen,
                          device=p.device, dtype=torch.bfloat16) * 2 - 1)


def _h_signed_streaming_add(grad: torch.Tensor, z: torch.Tensor,
                            h_accum_view: torch.Tensor, g_blk: int,
                            scale: float, sign: float) -> None:
    """h_accum_view[gg] += sign * scale * group_mean_G(z * grad).

    Signed (no abs). Used by Phase 1 (+sign) and Phase 2 (-sign) to
    incrementally build group_mean(z · (gp - gm)) in h_accum_view.
    abs is applied once at end of K loop.
    """
    n = int(grad.numel())
    n_aligned = (n // max(1, int(g_blk))) * max(1, int(g_blk))
    if n_aligned <= 0:
        return
    # FP32 transient (avoid mutating original grad/z, BF16 underflow safety).
    g32 = grad.detach().reshape(-1)[:n_aligned].to(torch.float32, copy=True)
    z32 = z.detach().reshape(-1)[:n_aligned].to(torch.float32, copy=True)
    g32.mul_(z32)
    group = g32.reshape(-1, int(g_blk)).mean(dim=1)
    group.mul_(float(scale) * float(sign))
    h_accum_view.add_(group.to(torch.bfloat16))
    del g32, z32, group


def _backward_inplace(forward_fn: Callable[[], Any],
                      expert_ps: List[torch.nn.Parameter],
                      model_for_full_zero: Any,
                      preserve_expert_grad_storage: bool = False) -> bool:
    """Forward + backward; leaves expert grads alive in ``p.grad``.
    Clears non-expert grads after backward. Returns True on success.

    ``preserve_expert_grad_storage`` (fast-flat path): caller has assigned
    ``expert.grad`` as a view of a pre-allocated flat tensor. Do NOT set
    expert.grad = None pre-backward (would destroy the view). Caller is
    responsible for zeroing the flat tensor before the next backward.
    """
    try:
        loss, _ = forward_fn()
    except Exception as exc:
        print(f"[proposal] finite_diff: forward failed: {exc}", flush=True)
        return False
    if not torch.is_tensor(loss):
        return False
    if loss.dim() > 0:
        loss = loss.mean()
    _mfz = (model_for_full_zero[0]
            if isinstance(model_for_full_zero, (list, tuple))
            else model_for_full_zero) if model_for_full_zero is not None else None
    if _mfz is not None:
        _exp_ids = {id(p) for p in expert_ps} if preserve_expert_grad_storage else None
        try:
            for _p in _mfz.parameters():
                if _p.grad is None:
                    continue
                if preserve_expert_grad_storage and id(_p) in _exp_ids:
                    continue  # keep flat-view, caller zeroed already
                _p.grad = None
        except Exception:
            pass
    try:
        loss.backward(retain_graph=False)
    except Exception as exc:
        print(f"[proposal] finite_diff: backward failed: {exc}", flush=True)
        return False
    if _mfz is not None:
        _params_id = {id(p) for p in expert_ps}
        try:
            for _p in _mfz.parameters():
                if id(_p) not in _params_id and _p.grad is not None:
                    _p.grad = None
        except Exception as _exc:
            print(f"[FD] non-expert grad clear err: {_exc}", flush=True)
    return True


# ---------------------------------------------------------------------------
# Fast-flat path: single Triton kernel for perturb & stream, on-the-fly
# Rademacher z generated inside the kernel (register-only, 0 VRAM for z).
# Eliminates 4032+ host-side launches of the per-expert path.
# ---------------------------------------------------------------------------

if _HAS_TRITON_HESSIAN:
    @triton.jit
    def _flat_z_perturb_kernel(
        lp_ptr,          # *bf16, n_elem (in-place add)
        n_elem,
        eps_signed,      # +eps or -eps (float)
        global_offset,   # offset of lp_ptr[0] in the GLOBAL flat (for hash)
        rng_seed_lo,     # split int64 → 2 × int32 for triton portability
        rng_seed_hi,
        k_lo, k_hi,
        BLOCK: tl.constexpr,
    ):
        """In-place ``lp[i] += eps_signed * z(global_offset + i, k, seed)``.
        z = ±1 via element-wise PCG-like hash on the GLOBAL element index
        (so perturb's z matches stream's z even when lp is per-tensor and
        grad is a flat view). Register-only z, 0 VRAM."""
        pid = tl.program_id(0)
        local_offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = local_offs < n_elem
        global_offs = local_offs + global_offset
        lp = tl.load(lp_ptr + local_offs, mask=mask, other=0.0).to(tl.float32)
        x = global_offs.to(tl.uint32) * tl.full((), 0x9E3779B9, tl.uint32)
        x = x ^ tl.full((), rng_seed_lo, tl.uint32)
        x = x ^ (global_offs.to(tl.uint32) >> 16) * tl.full((), 0x85EBCA6B, tl.uint32)
        x = x ^ tl.full((), k_lo, tl.uint32) * tl.full((), 0xC2B2AE35, tl.uint32)
        x = x ^ (x >> 13)
        x = x * tl.full((), 0x27D4EB2D, tl.uint32)
        x = x ^ (x >> 16)
        z = tl.where((x & tl.full((), 1, tl.uint32)) == 0, 1.0, -1.0)
        tl.store(lp_ptr + local_offs,
                 (lp + eps_signed * z).to(tl.bfloat16), mask=mask)


def _flat_z_perturb(lp_tensor: torch.Tensor, eps_signed: float,
                    global_offset: int,
                    rng_seed: int, k: int) -> None:
    """Launch ``_flat_z_perturb_kernel`` on a contiguous BF16 tensor view.
    ``global_offset`` lets the same hash drive both per-tensor perturb and
    a flat-grad signed stream — z is bit-exact identical."""
    n = int(lp_tensor.numel())
    if n == 0:
        return
    if not _HAS_TRITON_HESSIAN:
        gen = torch.Generator(device=lp_tensor.device).manual_seed(
            int(rng_seed) ^ int(k) ^ int(global_offset))
        z = (torch.randint(0, 2, (n,), generator=gen,
                           device=lp_tensor.device,
                           dtype=torch.bfloat16) * 2 - 1)
        lp_tensor.view(-1).add_(z, alpha=float(eps_signed))
        return
    BLOCK = 8192
    n_progs = (n + BLOCK - 1) // BLOCK
    seed64 = int(rng_seed) & ((1 << 64) - 1)
    k64 = int(k) & ((1 << 64) - 1)
    _flat_z_perturb_kernel[(n_progs,)](
        lp_tensor, n,
        float(eps_signed),
        int(global_offset),
        seed64 & 0xFFFFFFFF, (seed64 >> 32) & 0xFFFFFFFF,
        k64 & 0xFFFFFFFF, (k64 >> 32) & 0xFFFFFFFF,
        BLOCK=BLOCK,
    )


if _HAS_TRITON_HESSIAN:
    @triton.jit
    def _flat_signed_stream_kernel(
        grad_ptr,        # *bf16, n_groups * G
        h_accum_ptr,     # *bf16, n_groups
        n_groups,
        sign_scale,      # sign × inv_step × inv_k (float)
        rng_seed_lo, rng_seed_hi,
        k_lo, k_hi,
        G: tl.constexpr,
        GROUPS_PER_BLOCK: tl.constexpr,
    ):
        """h_accum[g] += sign_scale × group_mean_G(z(offset) × grad[offset]).
        Signed (no abs). z generated on-the-fly via element-wise hash —
        SAME hash as _flat_z_perturb_kernel so FD cancellation is bit-exact."""
        BLOCK: tl.constexpr = G * GROUPS_PER_BLOCK
        pid = tl.program_id(0)
        gid_start = pid * GROUPS_PER_BLOCK
        elem_off = gid_start * G + tl.arange(0, BLOCK)
        elem_mask = elem_off < n_groups * G
        grad = tl.load(grad_ptr + elem_off, mask=elem_mask,
                       other=0.0).to(tl.float32)
        x = elem_off.to(tl.uint32) * tl.full((), 0x9E3779B9, tl.uint32)
        x = x ^ tl.full((), rng_seed_lo, tl.uint32)
        x = x ^ (elem_off.to(tl.uint32) >> 16) * tl.full((), 0x85EBCA6B, tl.uint32)
        x = x ^ tl.full((), k_lo, tl.uint32) * tl.full((), 0xC2B2AE35, tl.uint32)
        x = x ^ (x >> 13)
        x = x * tl.full((), 0x27D4EB2D, tl.uint32)
        x = x ^ (x >> 16)
        z = tl.where((x & tl.full((), 1, tl.uint32)) == 0, 1.0, -1.0)
        contrib = z * grad * sign_scale
        contrib_2d = tl.reshape(contrib, (GROUPS_PER_BLOCK, G))
        group_sum = tl.sum(contrib_2d, axis=1) / G
        gid_off = gid_start + tl.arange(0, GROUPS_PER_BLOCK)
        gid_mask = gid_off < n_groups
        h_old = tl.load(h_accum_ptr + gid_off, mask=gid_mask,
                        other=0.0).to(tl.float32)
        tl.store(h_accum_ptr + gid_off,
                 (h_old + group_sum).to(tl.bfloat16), mask=gid_mask)


def _flat_signed_stream(grad_flat: torch.Tensor, h_accum_flat: torch.Tensor,
                        g_blk: int, sign_scale: float,
                        rng_seed: int, k: int) -> None:
    """Launch ``_flat_signed_stream_kernel``. n_groups = h_accum_flat.numel()."""
    n_groups = int(h_accum_flat.numel())
    if n_groups == 0:
        return
    if not _HAS_TRITON_HESSIAN:
        # Fallback (slow): PyTorch ops on flat. Same hash semantics via fast path.
        n = n_groups * int(g_blk)
        gen = torch.Generator(device=grad_flat.device).manual_seed(int(rng_seed) ^ int(k))
        z = (torch.randint(0, 2, (n,), generator=gen,
                           device=grad_flat.device,
                           dtype=torch.bfloat16) * 2 - 1)
        g32 = grad_flat[:n].to(torch.float32).mul_(z.to(torch.float32))
        group = g32.reshape(-1, int(g_blk)).mean(dim=1).mul_(float(sign_scale))
        h_accum_flat.add_(group.to(torch.bfloat16))
        return
    target_block = 256
    gpb = max(1, target_block // max(1, int(g_blk)))
    p = 1
    while p < gpb:
        p *= 2
    gpb = p
    n_progs = (n_groups + gpb - 1) // gpb
    seed64 = int(rng_seed) & ((1 << 64) - 1)
    k64 = int(k) & ((1 << 64) - 1)
    _flat_signed_stream_kernel[(n_progs,)](
        grad_flat, h_accum_flat,
        n_groups, float(sign_scale),
        seed64 & 0xFFFFFFFF, (seed64 >> 32) & 0xFFFFFFFF,
        k64 & 0xFFFFFFFF, (k64 >> 32) & 0xFFFFFFFF,
        G=int(g_blk), GROUPS_PER_BLOCK=gpb,
    )


def _build_expert_flat_view(expert_ps: List[torch.nn.Parameter]
                            ) -> Tuple[Optional[torch.Tensor],
                                        Optional[List[int]], bool]:
    """Return (lp_flat_view, per-expert element offsets in flat, ok).

    Verifies all expert_ps share the same underlying storage AND are
    contiguous with no gaps in element-offset order. If so, builds a
    1-D BF16 view spanning the whole expert region — the same bytes that
    DS ``bf16_groups_flat[expert_group]`` exposes.
    """
    import os as _os_dbg
    _dbg = _os_dbg.environ.get("PROPOSAL_FD_FAST_FLAT_DBG", "0") in ("1",)
    if not expert_ps:
        return None, None, False
    p0 = expert_ps[0]
    if not p0.is_contiguous():
        if _dbg:
            print(f"[fast-flat-dbg] p0 not contiguous", flush=True)
        return None, None, False
    storage = p0.untyped_storage()
    storage_ptr = storage.data_ptr()
    dtype = p0.dtype
    dtype_bytes = p0.element_size()
    items: List[Tuple[int, int, int]] = []  # (offset_bytes, numel, orig_idx)
    distinct_storages: Dict[int, int] = {}
    for i, p in enumerate(expert_ps):
        s_ptr = p.untyped_storage().data_ptr()
        distinct_storages[s_ptr] = distinct_storages.get(s_ptr, 0) + 1
        if (p.dtype != dtype or not p.is_contiguous()
                or s_ptr != storage_ptr):
            if _dbg:
                print(f"[fast-flat-dbg] mismatch at i={i}: "
                      f"dtype={p.dtype} contig={p.is_contiguous()} "
                      f"ptr={hex(s_ptr)} vs base={hex(storage_ptr)} "
                      f"distinct_storage_count={len(distinct_storages)}",
                      flush=True)
            return None, None, False
        items.append((p.data.storage_offset() * dtype_bytes,
                      p.numel(), i))
    items.sort(key=lambda x: x[0])
    expected = items[0][0]
    for off_b, n, _ in items:
        if off_b != expected:
            if _dbg:
                print(f"[fast-flat-dbg] gap at offset={off_b} expected={expected} "
                      f"(diff={off_b-expected} bytes)", flush=True)
            return None, None, False
        expected += n * dtype_bytes
    min_off_b = items[0][0]
    total_n = sum(n for _, n, _ in items)
    lp_flat = torch.empty(0, dtype=dtype, device=p0.device).set_(
        storage, storage_offset=min_off_b // dtype_bytes,
        size=(total_n,), stride=(1,),
    )
    offsets = [0] * len(expert_ps)
    cum = 0
    for off_b, n, orig_i in items:
        offsets[orig_i] = cum
        cum += n
    if _dbg:
        print(f"[fast-flat-dbg] OK n_experts={len(expert_ps)} "
              f"total_n={total_n} ({total_n*dtype_bytes/1e9:.2f}GB)",
              flush=True)
    return lp_flat, offsets, True



def _compute_fd_fast_flat(
    forward_loss_fn: Callable[[], Tuple[torch.Tensor, Any]],
    expert_ps: List[torch.nn.Parameter],
    k_samples: int,
    eps: float,
    gates_for_freeze: Optional[List[Tuple[str, torch.nn.Module]]],
    rng_seed: int,
    g_blk: int,
    model_for_full_zero: Any,
    grad_flat: torch.Tensor,
    grad_offsets: List[int],
    h_accum_flat: torch.Tensor,
    restore_src_list: List[torch.Tensor],
) -> Optional[List[torch.Tensor]]:
    """Fast FD with batched grad-flat stream + per-tensor hash-based perturb.

    - Each expert.grad is already a view into ``grad_flat`` (caller assigns).
    - Perturb: per-expert Triton kernel using GLOBAL ``grad_offsets[i]`` for
      the z hash → identical z used by the single flat stream kernel.
    - Stream: single Triton kernel reads ``grad_flat`` → group-mean → signed
      accumulate into ``h_accum_flat`` (BF16). z register-only.
    - lp storage is per-expert (FP16_Optimizer / mixed groups); no lp_flat
      required. theta_snap is per-expert pinned (caller-managed).
    """
    from ._routing_freeze import RoutingFreezeContext

    import os as _os_san
    _timing_on = (_os_san.environ.get("PROPOSAL_FD_TIMING", "0")
                  in ("1", "true", "TRUE"))
    import time as _time

    def _sync():
        if _timing_on and torch.cuda.is_available():
            torch.cuda.synchronize()

    theta_before_norms = [float(p.data.detach().to(torch.float32).norm().item())
                          for p in expert_ps]

    g = max(1, int(g_blk))
    inv_step = 1.0 / (2.0 * float(eps))
    inv_k = 1.0 / max(1, int(k_samples))

    ctx_mgr = (RoutingFreezeContext(gates_for_freeze)
               if gates_for_freeze else None)

    _tsum = {"perturb": 0.0, "bwd": 0.0, "stream": 0.0, "zero": 0.0}
    try:
        with (ctx_mgr if ctx_mgr is not None else nullcontext()):
            if ctx_mgr is not None:
                try:
                    ctx_mgr.capture(forward_loss_fn)
                except Exception as exc:
                    print(f"[proposal] finite_diff: routing-freeze capture failed: {exc}",
                          flush=True)
                    return None
            _sync()
            _t_loop = _time.perf_counter()
            for k in range(max(1, int(k_samples))):
                for sign in (+1.0, -1.0):
                    _sync(); _t0 = _time.perf_counter()
                    # Per-tensor perturb: copy snap → add sign×eps × z(global_off+i).
                    with torch.no_grad():
                        for i, p in enumerate(expert_ps):
                            p.data.copy_(restore_src_list[i], non_blocking=True)
                            _flat_z_perturb(
                                p.data, sign * float(eps),
                                global_offset=int(grad_offsets[i]),
                                rng_seed=int(rng_seed), k=int(k),
                            )
                    _sync(); _t1 = _time.perf_counter()
                    if not _backward_inplace(forward_loss_fn, expert_ps,
                                              model_for_full_zero,
                                              preserve_expert_grad_storage=True):
                        return None
                    _sync(); _t2 = _time.perf_counter()
                    _flat_signed_stream(
                        grad_flat, h_accum_flat,
                        g_blk=g,
                        sign_scale=sign * inv_step * inv_k,
                        rng_seed=int(rng_seed), k=int(k),
                    )
                    _sync(); _t3 = _time.perf_counter()
                    grad_flat.zero_()
                    _sync(); _t4 = _time.perf_counter()
                    _tsum["perturb"] += (_t1 - _t0)
                    _tsum["bwd"]     += (_t2 - _t1)
                    _tsum["stream"]  += (_t3 - _t2)
                    _tsum["zero"]    += (_t4 - _t3)
            if _timing_on:
                _sync()
                _t_done = _time.perf_counter()
                print(f"[FD_FAST] loop={(_t_done-_t_loop)*1000:.0f}ms "
                      f"(perturb={_tsum['perturb']*1000:.0f} "
                      f"bwd={_tsum['bwd']*1000:.0f} "
                      f"stream={_tsum['stream']*1000:.0f} "
                      f"zero={_tsum['zero']*1000:.0f})",
                      flush=True)
    except Exception as exc:
        import traceback as _tb
        print(f"[proposal] finite_diff: fast-flat refresh failed: {type(exc).__name__}: {exc}\n"
              f"{_tb.format_exc()}", flush=True)
        return None
    # NB: lp left in stale -εz state; caller's finally restores from snaps.
    # h_accum_flat carries the SIGNED running sum across the multi-batch
    # loop; caller takes the final abs + per-expert split after dividing
    # by the success count. Sentinel non-None return signals success.
    if not bool(torch.isfinite(h_accum_flat).all()):
        print("[proposal] finite_diff: NaN/Inf in fast-flat result", flush=True)
        return None
    return []  # success sentinel; cp_hooks owns final reduction


def compute_hessian_finite_diff_core(
    forward_loss_fn: Callable[[], Tuple[torch.Tensor, Any]],
    expert_ps: List[torch.nn.Parameter],
    k_samples: int,
    eps: float,
    gates_for_freeze: Optional[List[Tuple[str, torch.nn.Module]]] = None,
    rng_seed: int = 0xFD15CAFE,
    one_sided: bool = False,
    track_k_values: Optional[List[int]] = None,
    g_blk: int = 1,
    model_for_full_zero: Any = None,
    theta_snap_ref: Optional[List[torch.Tensor]] = None,
    grad_flat: Optional[torch.Tensor] = None,
    grad_offsets: Optional[List[int]] = None,
    h_accum_flat: Optional[torch.Tensor] = None,
) -> Optional[List[torch.Tensor]]:
    """FD diagonal Hessian estimate via Phase-1/Phase-2 streaming.

    H̃_g ≈ |group_mean_G(z · (gp - gm) / 2ε)|, K-averaged.

      Phase 1 (+ε): per-expert in-place perturb θ+εz → forward+backward
        → per-expert streaming adds  +group_mean(z · gp · inv_step · inv_k)
        into ``h_accum[i]``  → expert.grad = None.
      Phase 2 (-ε): θ restore + per-expert in-place perturb θ-εz → forward
        +backward → per-expert streaming subtracts the same shape from
        ``h_accum[i]`` (sign flip) → expert.grad = None.
      Finalize: h_accum.abs_().

    Architectural invariant: ``gp`` and ``gm`` are NEVER alive simultaneously.
    Each expert's grad is folded into the group-mean accumulator and freed
    before the next backward. ``z`` is generated lazily per (k, expert_idx)
    with a deterministic seed; the same z is used in Phase 1 perturb,
    Phase 1 streaming, Phase 2 perturb, Phase 2 streaming — bit-exact.

    ``one_sided`` / ``track_k_values`` are accepted for API compatibility
    but unsupported in the streaming path (logged + ignored).
    Returns list of BF16 group-level |H| tensors, or ``None`` on failure.
    """
    if not expert_ps or eps <= 0:
        return None
    if one_sided:
        print("[proposal] finite_diff: one_sided unsupported in streaming path; "
              "using central diff", flush=True)
    if track_k_values:
        print("[proposal] finite_diff: track_k_values unsupported in streaming "
              "path; ignored", flush=True)
    from ._routing_freeze import RoutingFreezeContext
    from ._ds_state import _pinned_clone

    # Fast-flat dispatch: caller provides (grad_flat, grad_offsets,
    # h_accum_flat, theta_snap_ref). lp stays per-tensor (no constraint on
    # DS optimizer layout); grad gets a contiguous flat backing for single
    # Triton stream kernel; perturb uses per-tensor kernel with hash on the
    # GLOBAL offset so its z matches stream's z bit-exactly.
    import os as _os_ff
    _fast_flat_env = (_os_ff.environ.get("PROPOSAL_FD_FAST_FLAT", "1")
                      not in ("0", "false", "off", "OFF", "False"))
    if (_fast_flat_env and grad_flat is not None
            and grad_offsets is not None
            and h_accum_flat is not None
            and theta_snap_ref is not None
            and _HAS_TRITON_HESSIAN):
        return _compute_fd_fast_flat(
            forward_loss_fn=forward_loss_fn,
            expert_ps=expert_ps,
            k_samples=k_samples,
            eps=eps,
            gates_for_freeze=gates_for_freeze,
            rng_seed=rng_seed,
            g_blk=g_blk,
            model_for_full_zero=model_for_full_zero,
            grad_flat=grad_flat,
            grad_offsets=grad_offsets,
            h_accum_flat=h_accum_flat,
            restore_src_list=theta_snap_ref,
        )

    import os as _os_t0
    _timing_outer = (_os_t0.environ.get("PROPOSAL_FD_TIMING", "0")
                     in ("1", "true", "TRUE"))
    import time as _time_t0

    def _sync_for_timing():
        if _timing_outer and torch.cuda.is_available():
            torch.cuda.synchronize()

    _sync_for_timing()
    _t_norm0 = _time_t0.perf_counter()
    theta_before_norms = [float(p.data.detach().to(torch.float32).norm().item())
                          for p in expert_ps]
    _sync_for_timing()
    _t_norm1 = _time_t0.perf_counter()

    # θ_0 snapshot. If caller provided one (multi-batch anchor session),
    # reuse it — skip per-batch D2H + post-batch H2D restore (caller owns
    # the lifecycle). Otherwise allocate via ``_pinned_clone`` pool.
    _do_internal_restore = theta_snap_ref is None
    if theta_snap_ref is not None and len(theta_snap_ref) == len(expert_ps):
        _restore_src = theta_snap_ref
    else:
        _restore_src = [
            _pinned_clone(p.data.detach(), key=("anchor_theta", i))
            for i, p in enumerate(expert_ps)
        ]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    _t_snap1 = _time_t0.perf_counter()
    if _timing_outer:
        print(f"[FD_OUTER] theta_norms={(_t_norm1-_t_norm0)*1000:.0f}ms "
              f"backup_D2H={(_t_snap1-_t_norm1)*1000:.0f}ms"
              f"{' (cached)' if not _do_internal_restore else ''}",
              flush=True)

    if gates_for_freeze:
        ctx_mgr = RoutingFreezeContext(gates_for_freeze)
    else:
        ctx_mgr = None

    # Per-name h_accum (BF16 group-level). Single allocation per anchor
    # call, no chunk-local extras. At medium expert_ps ~2.72 GB BF16 and
    # g_blk=8: total ~340 MB BF16.
    h_accum = [
        torch.zeros(
            (p.numel() // max(1, int(g_blk)),),
            dtype=torch.bfloat16, device=p.device,
        )
        for p in expert_ps
    ]
    inv_step = 1.0 / (2.0 * eps)
    inv_k = 1.0 / max(1, int(k_samples))

    import os as _os_san
    _sanity_on = (_os_san.environ.get("PROPOSAL_FD_Z_SANITY", "1")
                  not in ("0", "false", "off", "OFF", "False"))
    _timing_on = (_os_san.environ.get("PROPOSAL_FD_TIMING", "0")
                  in ("1", "true", "TRUE"))
    _mem_dump = (_os_san.environ.get("PROPOSAL_FD_MEM_DUMP", "0")
                 in ("1", "true", "TRUE"))
    import time as _time
    global _Z_SANITY_DONE
    _t_setup = _time.perf_counter()

    def _sync_now():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _mem(tag: str):
        if _mem_dump and torch.cuda.is_available():
            torch.cuda.synchronize()
            a = torch.cuda.memory_allocated() / 1e6
            p = torch.cuda.max_memory_allocated() / 1e6
            r = torch.cuda.memory_reserved() / 1e6
            print(f"[FD_MEM] {tag:30s} alloc={a:8.1f} peak={p:8.1f} reserved={r:8.1f}",
                  flush=True)

    _tsum = {"perturb_p1": 0.0, "bwd_p1": 0.0, "stream_p1": 0.0,
             "perturb_p2": 0.0, "bwd_p2": 0.0, "stream_p2": 0.0}
    try:
        with (ctx_mgr if ctx_mgr is not None else nullcontext()):
            if ctx_mgr is not None:
                try:
                    ctx_mgr.capture(forward_loss_fn)
                except Exception as exc:
                    print(f"[proposal] finite_diff: routing-freeze capture failed: {exc}",
                          flush=True)
                    return None

            # One-time sanity: lazy_rademacher must be deterministic for
            # the same seed (Phase 1 perturb / streaming / Phase 2 perturb
            # / streaming all rely on this for FD cancellation).
            if _sanity_on and not _Z_SANITY_DONE and expert_ps:
                _seed0 = _z_seed(int(rng_seed), 0, 0)
                _z_a = _lazy_rademacher(expert_ps[0], _seed0)
                _z_b = _lazy_rademacher(expert_ps[0], _seed0)
                _same = bool(torch.equal(_z_a, _z_b))
                print(f"[FD_SANITY] z_deterministic={_same} (seed={_seed0})",
                      flush=True)
                del _z_a, _z_b
                if not _same:
                    print("[FD_SANITY] FATAL: lazy_rademacher is non-deterministic",
                          flush=True)
                    return None
                _Z_SANITY_DONE = True

            if _timing_on:
                _sync_now()
            _t_loop = _time.perf_counter()
            _mem("00_loop_entry")

            for k in range(max(1, int(k_samples))):
                if _timing_on: _sync_now()
                _t0 = _time.perf_counter()
                # Phase 1 (+ε): per-expert in-place perturb θ → θ+εz.
                with torch.no_grad():
                    for i, (p, snap) in enumerate(zip(expert_ps, _restore_src)):
                        z_e = _lazy_rademacher(p, _z_seed(int(rng_seed), k, i))
                        p.data.copy_(snap).add_(z_e, alpha=eps)
                        del z_e
                if _timing_on: _sync_now()
                _t1 = _time.perf_counter()
                _mem("10_after_perturb_p1")
                if not _backward_inplace(forward_loss_fn, expert_ps,
                                         model_for_full_zero):
                    return None
                if _timing_on: _sync_now()
                _t2 = _time.perf_counter()
                _mem("11_after_bwd_p1")
                # Phase 1 streaming: signed add into h_accum.
                for i, p in enumerate(expert_ps):
                    if p.grad is None:
                        continue
                    z_e = _lazy_rademacher(p, _z_seed(int(rng_seed), k, i))
                    _h_signed_streaming_add(
                        p.grad, z_e, h_accum[i],
                        g_blk=int(g_blk),
                        scale=float(inv_step) * float(inv_k),
                        sign=+1.0,
                    )
                    p.grad = None
                    del z_e
                if _timing_on: _sync_now()
                _t3 = _time.perf_counter()
                _mem("12_after_stream_p1")

                # Phase 2 (-ε): restore θ + perturb θ-εz.
                with torch.no_grad():
                    for i, (p, snap) in enumerate(zip(expert_ps, _restore_src)):
                        z_e = _lazy_rademacher(p, _z_seed(int(rng_seed), k, i))
                        p.data.copy_(snap).add_(z_e, alpha=-eps)
                        del z_e
                if _timing_on: _sync_now()
                _t4 = _time.perf_counter()
                _mem("20_after_perturb_p2")
                if not _backward_inplace(forward_loss_fn, expert_ps,
                                         model_for_full_zero):
                    return None
                if _timing_on: _sync_now()
                _t5 = _time.perf_counter()
                _mem("21_after_bwd_p2")
                # Phase 2 streaming: signed sub from h_accum.
                for i, p in enumerate(expert_ps):
                    if p.grad is None:
                        continue
                    z_e = _lazy_rademacher(p, _z_seed(int(rng_seed), k, i))
                    _h_signed_streaming_add(
                        p.grad, z_e, h_accum[i],
                        g_blk=int(g_blk),
                        scale=float(inv_step) * float(inv_k),
                        sign=-1.0,
                    )
                    p.grad = None
                    del z_e
                if _timing_on: _sync_now()
                _t6 = _time.perf_counter()
                _mem("22_after_stream_p2")
                _tsum["perturb_p1"] += (_t1 - _t0)
                _tsum["bwd_p1"]     += (_t2 - _t1)
                _tsum["stream_p1"]  += (_t3 - _t2)
                _tsum["perturb_p2"] += (_t4 - _t3)
                _tsum["bwd_p2"]     += (_t5 - _t4)
                _tsum["stream_p2"]  += (_t6 - _t5)
    except Exception as exc:
        import traceback as _tb
        print(f"[proposal] finite_diff: refresh failed: {type(exc).__name__}: {exc}\n"
              f"{_tb.format_exc()}", flush=True)
        return None
    finally:
        _sync_for_timing()
        _t_rs0 = _time_t0.perf_counter()
        if _do_internal_restore:
            with torch.no_grad():
                for p, snap in zip(expert_ps, _restore_src):
                    p.data.copy_(snap)
        _sync_for_timing()
        _t_rs1 = _time_t0.perf_counter()
        if _timing_outer:
            print(f"[FD_OUTER] restore_H2D={(_t_rs1-_t_rs0)*1000:.0f}ms"
                  f"{' (skipped, caller owns)' if not _do_internal_restore else ''}",
                  flush=True)
        del _restore_src

    if _timing_on:
        _sync_now()
        _t_done = _time.perf_counter()
        _t_setup_dur = _t_loop - _t_setup
        _t_loop_dur = _t_done - _t_loop
        print(f"[FD_TIMING] setup={_t_setup_dur*1000:.0f}ms "
              f"loop={_t_loop_dur*1000:.0f}ms "
              f"(perturb_p1={_tsum['perturb_p1']*1000:.0f} "
              f"bwd_p1={_tsum['bwd_p1']*1000:.0f} "
              f"stream_p1={_tsum['stream_p1']*1000:.0f} | "
              f"perturb_p2={_tsum['perturb_p2']*1000:.0f} "
              f"bwd_p2={_tsum['bwd_p2']*1000:.0f} "
              f"stream_p2={_tsum['stream_p2']*1000:.0f})",
              flush=True)

    # Finalize: |group_mean_G(z · (gp - gm) · inv_step / K)|.
    for h in h_accum:
        h.abs_()

    # Restore correctness check (norm).
    for p, n_before in zip(expert_ps, theta_before_norms):
        n_after = float(p.data.detach().to(torch.float32).norm().item())
        if abs(n_after - n_before) > max(1e-6, 1e-5 * max(n_before, 1.0)):
            print(f"[proposal] finite_diff: restore check FAILED on tensor (norm "
                  f"before={n_before:.6e} after={n_after:.6e})", flush=True)
            return None

    for h in h_accum:
        if not bool(torch.isfinite(h).all()):
            print("[proposal] finite_diff: NaN/Inf in result", flush=True)
            return None

    return h_accum


def compute_hessian_finite_diff(
    model: Any,
    forward_step_func: Callable,
    data_iterator: Any,
    expert_ps: List[torch.nn.Parameter],
    k_samples: int,
    eps: Optional[float] = None,
    use_routing_freeze: bool = True,
    batch_override: Optional[Any] = None,
    rng_seed: int = 0xBADCAFE,
    one_sided: bool = False,
    track_k_values: Optional[List[int]] = None,
    g_blk: int = 1,
    theta_snap_ref: Optional[List[torch.Tensor]] = None,
    grad_flat: Optional[torch.Tensor] = None,
    grad_offsets: Optional[List[int]] = None,
    h_accum_flat: Optional[torch.Tensor] = None,
) -> Optional[Any]:
    """Diagonal Hessian via central finite-difference Hessian-vector products.

    Estimator (Rademacher-z central diff):

        Hz ≈ [∇L(θ + εz) - ∇L(θ - εz)] / (2ε)
        |H_diag|_j ≈ (1/K) Σ_k |z_jk · Hz_jk|

    Plain (1st-order) grads only — no ``create_graph=True``, no 2nd-order
    autograd. Memory cost ~3× a single 1st-order backward (one for the
    original θ batch reload, one for +ε, one for -ε), independent of K
    (per-k allocation is one z + one grad list, immediately folded into
    accumulator).

    Hard requirements (per spec):
      * Common Random Numbers: the batch is loaded ONCE per refresh and
        reused across +ε/-ε for every k. ``data_iterator`` is advanced
        exactly once.
      * Param mutation under ``torch.no_grad()`` then exact restore via
        equal-and-opposite ``add_``s. Per-tensor restore-error check
        guards against drift; on failure returns None.
      * eps=None triggers a 3-point auto-tune sweep; the chosen
        multiplier is cached at module scope.
      * Returns None on any internal failure.

    Routing freeze
    --------------
    When ``use_routing_freeze=True`` (default), MoE gates are identified
    via :func:`._routing_freeze.identify_gate_modules` and a single
    capture forward is run at θ; both perturbed forwards then replay the
    captured ``(l_aux, combine_weights, dispatch_mask)``. Without this
    freeze, top-k argmax flips between +ε and -ε produce infinite-magnitude
    Hz noise. If gate identification fails (zero gates), we fall back to
    no-freeze.

    EP > 1 is currently blocked at cp_hooks.py:120-124 so all-to-all
    interaction with the captured outputs is not exercised — this code
    path is unverified for EP > 1.
    """
    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    m0 = model[0] if isinstance(model, (list, tuple)) else model
    if not expert_ps:
        return None

    # ---- Load batch ONCE (Common Random Numbers). ----
    # Capture (tokens, labels, mask, attn, pos) once via get_batch then
    # run a custom forward that bypasses forward_step_func's iterator
    # advance. Caller may also pre-capture and inject via batch_override.
    if batch_override is not None:
        batch = batch_override
    else:
        try:
            from pretrain_gpt import get_batch as _get_batch_gpt
        except Exception as exc:
            print(f"[proposal] finite_diff: cannot import get_batch: {exc}", flush=True)
            return None
        try:
            batch = _get_batch_gpt(di)
        except Exception as exc:
            print(f"[proposal] finite_diff: get_batch failed: {exc}", flush=True)
            return None
    tokens, labels, loss_mask, attention_mask, position_ids = batch[:5]

    # Build a closure that re-runs the model forward + LM loss on the
    # cached batch. Mirrors Megatron's training forward.
    import torch.nn.functional as F

    def _forward_loss() -> Tuple[torch.Tensor, Dict[str, Any]]:
        out = m0(tokens, position_ids, attention_mask)
        logits = out[0] if isinstance(out, tuple) else out
        if logits is None or not torch.is_tensor(logits):
            raise RuntimeError("finite_diff: model returned non-tensor")
        if logits.dim() >= 2:
            V = logits.size(-1)
            flat_logits = logits.reshape(-1, V)
        else:
            flat_logits = logits
        flat_labels = labels.reshape(-1) if labels is not None else None
        if flat_labels is None:
            raise RuntimeError("finite_diff: labels missing")
        if loss_mask is not None and loss_mask.numel() == flat_logits.size(0):
            per_tok = F.cross_entropy(flat_logits.float(), flat_labels, reduction="none")
            denom = loss_mask.view(-1).float().sum().clamp_min(1.0)
            loss = (per_tok * loss_mask.view(-1).float()).sum() / denom
        else:
            loss = F.cross_entropy(flat_logits.float(), flat_labels, reduction="mean")
        return loss, {}

    # ---- Routing freeze setup ----
    from ._routing_freeze import identify_gate_modules_safe
    gates_for_freeze: Optional[List[Tuple[str, torch.nn.Module]]] = None
    if use_routing_freeze:
        gates, _rules = identify_gate_modules_safe(model)
        if not gates:
            print("[proposal] finite_diff: WARN no gates identified; running "
                  "without routing freeze", flush=True)
        else:
            gates_for_freeze = gates

    # ---- eps resolution ----
    # Default: 1e-3, validated on tiny (4-layer 256d) and production
    # (12-layer 768d) MoE setups under BF16+ZeRO-1 with K=16. Override
    # via `--proposal-finite-diff-eps` CLI or `PROPOSAL_FD_EPS` env if a
    # very different scale ever proves the default unsafe.
    eps_value = 1e-3 if eps is None else float(eps)

    # ---- Main K-sample finite-diff loop ----
    # Perturb expert lp directly (DS BF16 optimizer stores each lp's .data
    # as a view into the bf16_groups_flat backing buffer — verified at
    # bf16_optimizer.py:283-286: `p.data = q.data` where q is unflatten
    # view), so ``p.data.copy_(snap).add_(z, alpha=eps)`` updates the
    # bytes forward reads. theta_0 is preserved in a single pinned-host
    # snapshot (allocated once via _pinned_clone pool, reused across
    # anchors). Activation checkpoint follows the model's normal training
    # setting (no anchor-only override).
    return compute_hessian_finite_diff_core(
        forward_loss_fn=_forward_loss,
        expert_ps=expert_ps,
        k_samples=k_samples,
        eps=eps_value,
        gates_for_freeze=gates_for_freeze,
        rng_seed=rng_seed,
        one_sided=one_sided,
        track_k_values=track_k_values,
        g_blk=g_blk,
        model_for_full_zero=model,
        theta_snap_ref=theta_snap_ref,
        grad_flat=grad_flat,
        grad_offsets=grad_offsets,
        h_accum_flat=h_accum_flat,
    )



