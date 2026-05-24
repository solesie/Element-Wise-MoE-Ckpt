"""Single-source VRAM audit instrumentation for the proposal subsystem.

Replaces scattered ad-hoc memory probes (`_vram_probe` blocks in
importance_tracker.py, GPU mem log block in cp_hooks.py) with one entry
point that captures:

  * ``cuda.memory_allocated()``           — steady-state alloc
  * ``cuda.max_memory_allocated()``       — peak since last reset
  * ``cuda.memory_reserved()``            — caching-allocator pool size
  * ``cuda.memory_stats()['allocation.all.allocated']``
                                          — monotonic count of allocs
                                            (per-iter delta detects R2 violations)
  * ``['inactive_split_bytes.all.current']``
                                          — pool fragmentation indicator

Outputs JSONL to ``$PROPOSAL_VRAM_AUDIT_LOG`` (one record per call).
Used by ``tests/vram_audit_eval.py`` to emit PASS/FAIL per Requirement.

R2 alloc-trace mode (``PROPOSAL_PER_ITER_ALLOC_TRACE=1``): monkey-patches
``torch.empty/zeros/tensor/full`` and ``Tensor.clone`` for callsites
inside ``megatron/proposal/``. Records (callsite, size_bytes, dtype)
between iter 30 and 130; any entry with size > 1 MB and count >= 100
flags a per-iter alloc on the proposal hot path.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch


_LOG_PATH: Optional[str] = None
_ALLOC_TRACE_ON: bool = False
_ALLOC_TRACE: List[Tuple[str, int, str]] = []
_ALLOC_TRACE_ITER_RANGE: Tuple[int, int] = (30, 130)
_PATCHED: bool = False
_PROPOSAL_DIR = os.path.dirname(os.path.abspath(__file__))
_CURRENT_ITER: int = 0


def is_enabled() -> bool:
    return os.environ.get("PROPOSAL_VRAM_AUDIT", "") in ("1", "true", "True")


def _ensure_log_path() -> Optional[str]:
    global _LOG_PATH
    if _LOG_PATH is not None:
        return _LOG_PATH
    p = os.environ.get("PROPOSAL_VRAM_AUDIT_LOG", "")
    if not p:
        base = os.environ.get("OUTPUT_BASEPATH", "/tmp")
        p = os.path.join(base, "vram_audit.jsonl")
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        # Truncate on first use.
        with open(p, "w") as f:
            f.write("")
        _LOG_PATH = p
        print(f"[vram_audit] log path: {p}", flush=True)
    except Exception as exc:
        print(f"[vram_audit] cannot open log {p}: {exc}", flush=True)
        _LOG_PATH = None
    return _LOG_PATH


def set_iter(step: int) -> None:
    """Record current training iteration; used to bound alloc-trace window."""
    global _CURRENT_ITER
    _CURRENT_ITER = int(step)


def _record(label: str, extra: Optional[Dict[str, Any]] = None) -> None:
    if not is_enabled() or not torch.cuda.is_available():
        return
    p = _ensure_log_path()
    if p is None:
        return
    try:
        torch.cuda.synchronize()
        stats = torch.cuda.memory_stats()
        rec = {
            "ts": time.time(),
            "iter": _CURRENT_ITER,
            "label": label,
            "alloc_mb": torch.cuda.memory_allocated() / 1e6,
            "peak_mb": torch.cuda.max_memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "alloc_count": int(stats.get("allocation.all.allocated", 0)),
            "free_count": int(stats.get("allocation.all.freed", 0)),
            "inactive_split_bytes": int(stats.get("inactive_split_bytes.all.current", 0)),
        }
        if extra:
            rec.update(extra)
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as exc:
        print(f"[vram_audit] record failed @{label}: {exc}", flush=True)


def vram_audit_step(label: str, *, dump_breakdown: bool = False,
                    tracker: Optional[Any] = None,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    """Capture VRAM snapshot at a named phase.

    ``label`` examples:
      iter_begin / iter_end
      tracker_step_begin / tracker_step_end
      dispatch_begin / dispatch_end
      pre_anchor / pre_anchor_offload / inside_anchor_fd_peak
      post_anchor_swapin / post_anchor

    ``dump_breakdown=True`` adds per-attribute tracker buffer sizes
    via ``tracker.gpu_memory_breakdown()`` to the record.
    """
    if not is_enabled():
        return
    bk: Dict[str, Any] = dict(extra) if extra else {}
    if dump_breakdown and tracker is not None:
        try:
            if hasattr(tracker, "gpu_memory_breakdown"):
                total, items = tracker.gpu_memory_breakdown()
                bk["tracker_total_mb"] = float(total)
                bk["tracker_items"] = [(n, float(s)) for n, s in items if s > 0.01]
        except Exception:
            pass
    _record(label, bk)


# ---------------------------------------------------------------------------
# Per-iter alloc trace (R2 precision check)
# ---------------------------------------------------------------------------
def _alloc_trace_active() -> bool:
    if not _ALLOC_TRACE_ON:
        return False
    lo, hi = _ALLOC_TRACE_ITER_RANGE
    return lo <= _CURRENT_ITER <= hi


def _alloc_trace_callsite_proposal() -> Optional[str]:
    """Return file:lineno of nearest caller inside megatron/proposal/,
    else None (skip recording)."""
    frame = sys._getframe(2)  # skip this function + the wrapped call
    while frame is not None:
        fn = frame.f_code.co_filename
        if fn.startswith(_PROPOSAL_DIR):
            return f"{os.path.basename(fn)}:{frame.f_lineno}"
        frame = frame.f_back
    return None


def _make_alloc_patch(orig_fn, name: str):
    def wrapper(*args, **kwargs):
        t = orig_fn(*args, **kwargs)
        if _alloc_trace_active():
            try:
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    cs = _alloc_trace_callsite_proposal()
                    if cs is not None:
                        nbytes = t.element_size() * t.numel()
                        _ALLOC_TRACE.append((cs, nbytes, str(t.dtype)))
            except Exception:
                pass
        return t
    wrapper.__name__ = name
    return wrapper


def _make_clone_patch(orig_clone):
    def wrapper(self_t, *args, **kwargs):
        t = orig_clone(self_t, *args, **kwargs)
        if _alloc_trace_active():
            try:
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    cs = _alloc_trace_callsite_proposal()
                    if cs is not None:
                        nbytes = t.element_size() * t.numel()
                        _ALLOC_TRACE.append((cs, nbytes, str(t.dtype) + ".clone"))
            except Exception:
                pass
        return t
    wrapper.__name__ = "clone"
    return wrapper


def install_alloc_trace() -> None:
    """Install monkey-patches for tensor creation. Idempotent."""
    global _ALLOC_TRACE_ON, _PATCHED
    if os.environ.get("PROPOSAL_PER_ITER_ALLOC_TRACE", "") not in ("1", "true", "True"):
        return
    _ALLOC_TRACE_ON = True
    if _PATCHED:
        return
    torch._orig_empty = torch.empty
    torch._orig_zeros = torch.zeros
    torch._orig_tensor = torch.tensor
    torch._orig_full = torch.full
    torch._orig_clone = torch.Tensor.clone
    torch.empty = _make_alloc_patch(torch._orig_empty, "empty")
    torch.zeros = _make_alloc_patch(torch._orig_zeros, "zeros")
    torch.tensor = _make_alloc_patch(torch._orig_tensor, "tensor")
    torch.full = _make_alloc_patch(torch._orig_full, "full")
    torch.Tensor.clone = _make_clone_patch(torch._orig_clone)
    _PATCHED = True
    print("[vram_audit] alloc trace patches installed", flush=True)


def alloc_trace_summary() -> List[Tuple[str, int, int, str]]:
    """Return (callsite, total_bytes, count, dtype) sorted by total_bytes desc.

    R2 violations: entries with size > 1 MB AND count >= 100.
    """
    from collections import defaultdict
    agg: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
    for (cs, sz, dt) in _ALLOC_TRACE:
        key = (cs, dt)
        agg[key][0] += sz  # total
        agg[key][1] += 1   # count
    out = sorted(
        [(cs, total, count, dt) for (cs, dt), (total, count) in agg.items()],
        key=lambda x: -x[1],
    )
    return out


def dump_alloc_trace_to_log() -> None:
    """Append alloc trace summary to audit log as one record."""
    if not is_enabled():
        return
    summary = alloc_trace_summary()
    # Keep top 30 by total bytes.
    top = summary[:30]
    entries = [
        {"callsite": cs, "total_mb": total / 1e6, "count": count, "dtype": dt}
        for (cs, total, count, dt) in top
    ]
    _record("alloc_trace_summary", {
        "alloc_trace": entries,
        "total_records": len(_ALLOC_TRACE),
    })
