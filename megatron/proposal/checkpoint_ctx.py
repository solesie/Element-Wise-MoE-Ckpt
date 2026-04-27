"""CPU offload of saved-for-backward tensors during Hutchinson refresh.

Used by :func:`megatron.proposal.hessian.compute_hessian_hutchinson` to
cap GPU memory under ``create_graph=True``. Installs
``torch.autograd.graph.saved_tensors_hooks`` that move every
above-threshold saved tensor to a pinned CPU buffer (async D2H) and
stream it back on demand (async H2D). Buffers are recycled through a
module-global pool so repeated refreshes amortize pinning cost to zero.

Per-tensor threshold (``HUTCHINSON_OFFLOAD_MIN_BYTES``, default 2 MiB)
keeps small tensors on GPU — the PCIe round-trip is not worth it for
LayerNorm params / router logits.

Empirical (exp2, 24-layer × 192 expert × topk=1, K=4):
    - Refresh peak VRAM:   OOM (~74 GB)  →  43 GB
    - Refresh time:        — (crash)     →  17 s steady-state
                                            (first refresh 70 s for
                                             pinned-pool warmup)
    - Pinned RAM pool:     0             →  79 GB (bounded; stable across refreshes)
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch


# Design notes
# ------------
# Why offload instead of activation checkpointing:
#   ``create_graph=True`` disables the memory savings that
#   ``torch.utils.checkpoint`` normally provides — the 1st-order backward
#   re-enters each checkpoint region to build the 2nd-order graph, and
#   the recomputed activations are pinned in that graph for the HVP loop.
#   On exp2 (24 layer MoE), checkpoint alone saved ~2 GB. CPU offload
#   intercepts the same tensors via ``saved_tensors_hooks`` and parks
#   them on the host, which is the actual root-cause fix.
#
# Why pinned CPU buffers + non_blocking:
#   Pageable CPU memory caps D2H throughput at ~2-6 GB/s; with ~80 GB
#   offloaded per refresh that dominates wall-clock. Page-locked memory
#   lets PCIe 4.0 x16 hit its full ~24 GB/s and allows async H2D on
#   unpack (overlapping with compute).
#
# Why a process-global pool:
#   Pinning pages with the driver is slow the first time a buffer is
#   minted, so we recycle buffers across refreshes. The pool is keyed on
#   ``(numel, dtype)``, grows monotonically until shape distribution
#   stabilises (~8k entries for exp2), then stays flat.
#
# Why defer buffer release until ctx exit:
#   Two distinct hazards it avoids:
#     1. Within the HVP K-loop, retain_graph=True causes ``unpack`` to
#        fire N times per saved tensor — releasing on every unpack would
#        enqueue duplicates and the pool would grow unboundedly.
#     2. ``non_blocking=True`` H2D from a pinned buffer can still be
#        in-flight when unpack returns. A subsequent pack that reuses
#        the buffer (``copy_`` on the CPU side) would race with the
#        in-flight DMA read.
#   We release everything in a ``finally`` block after a single
#   ``cuda.synchronize()``, so every pack within a ctx sees a buffer
#   whose DMA history is complete.
#
# Per-tensor size threshold (``HUTCHINSON_OFFLOAD_MIN_BYTES``, default
# 2 MiB): below-threshold tensors stay on GPU because the PCIe round-trip
# cost outweighs the VRAM benefit for LayerNorm params, router logits,
# etc.


# Module-global pool of pinned CPU buffers, keyed on (numel, dtype). Each
# entry is a list of free buffers. A context leases buffers on pack and
# returns them at ctx exit (not on unpack — see pack_to_cpu docstring for
# why). Steady-state size ≈ unique-shape saved-tensor count × peak
# concurrency (~8k tensors on exp2, ~1-2 GB pool).
_PINNED_POOL: "Dict[Tuple[int, torch.dtype], List[torch.Tensor]]" = {}


def _acquire_pinned_buffer(numel: int, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Lease a pinned CPU buffer of the requested shape/dtype from the pool,
    or allocate a new one. Returns None on allocation failure (caller
    should fall back to pageable transfer)."""
    key = (numel, dtype)
    pool = _PINNED_POOL.setdefault(key, [])
    if pool:
        return pool.pop()
    try:
        return torch.empty(numel, dtype=dtype, pin_memory=True)
    except Exception:
        return None


def _release_pinned_buffer(buf: torch.Tensor, numel: int, dtype: torch.dtype) -> None:
    """Return a pinned buffer to the pool for reuse on the next refresh."""
    key = (numel, dtype)
    _PINNED_POOL.setdefault(key, []).append(buf)


@contextmanager
def hutchinson_cpu_offload_ctx(min_bytes: Optional[int] = None):
    """Context manager that moves every saved-for-backward tensor to CPU.

    Uses pinned (page-locked) CPU buffers + ``non_blocking=True`` for
    near-max-PCIe D2H on pack and overlapping H2D on unpack. Buffers are
    leased from a module-global pool (``_PINNED_POOL``) so repeated
    refreshes amortize the pinning cost to ~0.

    Tensors below ``min_bytes`` (default 2 MiB, overridable via
    ``HUTCHINSON_OFFLOAD_MIN_BYTES``) stay on GPU — the PCIe round-trip is
    not worth the savings for small activations (LayerNorm params, router
    logits, etc.), and the pinned-allocator per-call overhead is non-trivial.

    Fallback paths:
      * If ``pin_memory=True`` allocation fails (OOM on pinned pool, rare),
        degrades to pageable ``.to("cpu")`` for that tensor.
      * If ``torch.autograd.graph.saved_tensors_hooks`` is unavailable
        (PyTorch < 1.11), yields immediately (no-op).
    """
    import os as _os
    _debug = _os.environ.get("HUTCHINSON_OFFLOAD_DEBUG", "") in ("1", "true", "True")
    if min_bytes is None:
        try:
            # Default 2 MiB = ~1M BF16 elements = one moderate activation.
            # Smaller threshold (1 MiB) measured at 0.3% VRAM gain in
            # exchange for 30% extra pack/unpack churn on exp2.
            min_bytes = int(_os.environ.get("HUTCHINSON_OFFLOAD_MIN_BYTES", str(2 << 20)))
        except ValueError:
            min_bytes = 2 << 20

    try:
        from torch.autograd.graph import saved_tensors_hooks as _saved_hooks
    except Exception as exc:
        print(f"[offload_ctx] saved_tensors_hooks unavailable ({exc}); no-op", flush=True)
        yield 0
        return

    n_packed = [0]
    n_kept = [0]
    n_fallback_pageable = [0]
    bytes_offloaded = [0]
    # Buffers leased during THIS ctx invocation. They are returned to the
    # global pool at ctx exit (after a single cudaDeviceSynchronize), not
    # during unpack — two reasons:
    #   (1) ``retain_graph=True`` in the HVP loop causes ``unpack`` to
    #       fire multiple times for the same saved tensor. Returning the
    #       buffer on every unpack would enqueue N copies of the same
    #       buffer into the pool, growing it unboundedly per refresh.
    #   (2) ``non_blocking=True`` H2D from a pinned buffer may still be
    #       in-flight when unpack returns. If the pool immediately lends
    #       the buffer to a subsequent ``pack`` within the same ctx, the
    #       CPU ``copy_`` would race with the in-flight DMA read.
    # By deferring release, we guarantee every pack inside a ctx gets a
    # buffer whose DMA history is complete (= safe to overwrite), and
    # each buffer is returned exactly once.
    ctx_leased: List[Tuple[torch.Tensor, int, torch.dtype]] = []

    def pack_to_cpu(tensor):
        try:
            nb = tensor.element_size() * tensor.numel()
        except Exception:
            nb = 0
        if nb < min_bytes or not tensor.is_cuda:
            n_kept[0] += 1
            return ("keep", tensor)
        device = tensor.device
        numel = tensor.numel()
        dtype = tensor.dtype
        shape = tensor.shape
        buf = _acquire_pinned_buffer(numel, dtype)
        if buf is None:
            n_fallback_pageable[0] += 1
            cpu_copy = tensor.detach().to("cpu", copy=True)
            n_packed[0] += 1
            bytes_offloaded[0] += nb
            return ("cpu_pageable", cpu_copy, device, tensor.requires_grad, shape)
        buf_view = buf.view(shape)
        buf_view.copy_(tensor.detach(), non_blocking=True)
        ctx_leased.append((buf, numel, dtype))
        n_packed[0] += 1
        bytes_offloaded[0] += nb
        return ("cpu_pinned", buf_view, device, tensor.requires_grad)

    def unpack_from_cpu(packed):
        tag = packed[0]
        if tag == "keep":
            return packed[1]
        if tag == "cpu_pageable":
            _, cpu_tensor, device, requires_grad, _shape = packed
            gpu_tensor = cpu_tensor.to(device, non_blocking=False)
        else:  # cpu_pinned
            _, buf_view, device, requires_grad = packed
            gpu_tensor = buf_view.to(device, non_blocking=True)
        if requires_grad and not gpu_tensor.requires_grad:
            gpu_tensor.requires_grad_(True)
        return gpu_tensor

    if _debug:
        pooled = sum(len(v) for v in _PINNED_POOL.values())
        print(
            f"[offload_ctx] activating saved_tensors_hooks: "
            f"min_bytes={min_bytes:_}, pinned_pool_entries={pooled}",
            flush=True,
        )

    with _saved_hooks(pack_to_cpu, unpack_from_cpu):
        try:
            yield n_packed
        finally:
            # Wait for any still-in-flight H2D (from the last unpack calls)
            # to complete before returning buffers — otherwise the pool's
            # next consumer could race with the tail of an H2D.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for buf, numel, dtype in ctx_leased:
                _release_pinned_buffer(buf, numel, dtype)
            ctx_leased.clear()
            if _debug:
                mb = bytes_offloaded[0] / (1024 * 1024)
                pooled_after = sum(len(v) for v in _PINNED_POOL.values())
                pool_mb = sum(
                    (k[0] * torch.tensor([], dtype=k[1]).element_size()) * len(v)
                    for k, v in _PINNED_POOL.items()
                ) / (1024 * 1024)
                print(
                    f"[offload_ctx] packed={n_packed[0]} kept_on_gpu={n_kept[0]} "
                    f"fallback_pageable={n_fallback_pageable[0]} "
                    f"offloaded={mb:.1f}MB pool_entries={pooled_after} "
                    f"pool_size={pool_mb:.1f}MB",
                    flush=True,
                )
