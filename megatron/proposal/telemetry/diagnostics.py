"""Diagnostics and audit helpers for proposal checkpointing."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class IterCheckpointStats:
    """Per-iteration lossy checkpoint statistics."""

    global_step: int
    n_total_elems: int
    n_selected_elems: int
    total_th: float
    phase_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class HRefreshStats:
    """Per-anchor H quantile refresh statistics."""

    iter: int
    n_batches: int
    elapsed_ms: float
    h_quantile_log_extent: float
    h_min: float
    h_max: float


_VRAM_AUDIT_LOG_PATH: Optional[str] = None
_ALLOC_TRACE_ENABLED: bool = False
_ALLOC_TRACE_RECORDS: List[Tuple[str, int, str]] = []
_ALLOC_TRACE_ITER_RANGE: Tuple[int, int] = (30, 130)
_ALLOC_TRACE_PATCHED: bool = False
_PROPOSAL_DIR = os.path.dirname(os.path.abspath(__file__))
_CURRENT_ITER: int = 0
_BASELINE_ALLOC_MB: Optional[float] = None


def is_vram_audit_enabled() -> bool:
    return os.environ.get("PROPOSAL_VRAM_AUDIT", "") in ("1", "true", "True")


def _ensure_vram_audit_log_path() -> Optional[str]:
    global _VRAM_AUDIT_LOG_PATH
    if _VRAM_AUDIT_LOG_PATH is not None:
        return _VRAM_AUDIT_LOG_PATH
    path = os.environ.get("PROPOSAL_VRAM_AUDIT_LOG", "")
    if not path:
        base = os.environ.get("OUTPUT_BASEPATH", "/tmp")
        path = os.path.join(base, "vram_audit.jsonl")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a"):
            pass
        _VRAM_AUDIT_LOG_PATH = path
        print(f"[vram_audit] log path: {path}", flush=True)
    except Exception as exc:
        print(f"[vram_audit] cannot open log {path}: {exc}", flush=True)
        _VRAM_AUDIT_LOG_PATH = None
    return _VRAM_AUDIT_LOG_PATH


def set_iter(step: int) -> None:
    global _CURRENT_ITER
    _CURRENT_ITER = int(step)


def set_baseline_alloc_mb(value: float) -> None:
    global _BASELINE_ALLOC_MB
    _BASELINE_ALLOC_MB = float(value)


def get_baseline_alloc_mb() -> float:
    return 0.0 if _BASELINE_ALLOC_MB is None else float(_BASELINE_ALLOC_MB)


def _record_vram_audit(label: str, extra: Optional[Dict[str, Any]] = None) -> None:
    if not is_vram_audit_enabled() or not torch.cuda.is_available():
        return
    path = _ensure_vram_audit_log_path()
    if path is None:
        return
    try:
        torch.cuda.synchronize()
        stats = torch.cuda.memory_stats()
        record = {
            "ts": time.time(),
            "iter": _CURRENT_ITER,
            "rank": int(os.environ.get("RANK", "-1")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
            "label": label,
            "alloc_mb": torch.cuda.memory_allocated() / 1e6,
            "peak_mb": torch.cuda.max_memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "alloc_count": int(stats.get("allocation.all.allocated", 0)),
            "free_count": int(stats.get("allocation.all.freed", 0)),
            "inactive_split_bytes": int(stats.get("inactive_split_bytes.all.current", 0)),
        }
        if extra:
            record.update(extra)
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    except Exception as exc:
        print(f"[vram_audit] record failed @{label}: {exc}", flush=True)


def vram_audit_step(
    label: str,
    *,
    dump_breakdown: bool = False,
    lossy_cp: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    synchronize: bool = True,
) -> None:
    """Record one VRAM sample when the audit is explicitly enabled.

    ``synchronize`` is part of the caller contract used at anchor boundaries.
    The current recorder has always synchronized before sampling; rejecting
    the keyword made every H-refresh fail even when auditing was disabled.
    Keep the disabled production path as an immediate no-op and reject the
    only unsupported mode (an unsynchronized sample) explicitly instead of
    silently changing its meaning.
    """
    if not is_vram_audit_enabled():
        return
    if not synchronize:
        raise RuntimeError(
            "unsynchronized proposal VRAM samples are not supported"
        )
    audit_extra: Dict[str, Any] = dict(extra) if extra else {}
    if dump_breakdown and lossy_cp is not None:
        try:
            if hasattr(lossy_cp, "gpu_memory_breakdown"):
                total, items = lossy_cp.gpu_memory_breakdown()
                audit_extra["lossy_cp_total_mb"] = float(total)
                audit_extra["lossy_cp_items"] = [
                    (name, float(size_mb)) for name, size_mb in items if size_mb > 0.01
                ]
        except Exception:
            pass
    _record_vram_audit(label, audit_extra)


def _allocator_history_requested() -> bool:
    return os.environ.get("PROPOSAL_PER_ITER_ALLOC_TRACE", "") in (
        "1",
        "true",
        "True",
    )


def _configured_steady_allocator_history_iters() -> set[int]:
    raw = os.environ.get("PROPOSAL_ALLOC_TRACE_ITERS", "").strip()
    if not raw:
        return set()
    try:
        return {int(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError as exc:
        raise RuntimeError(
            "PROPOSAL_ALLOC_TRACE_ITERS must be a comma-separated list of integers"
        ) from exc


def _raise_allocator_history_unavailable(region: str, step: int) -> None:
    raise RuntimeError(
        "exact CUDA allocator-history audit was requested for "
        f"{region}@{int(step)}, but no allocator-history backend is installed; "
        "refusing to emit a misleading R2/R4 PASS"
    )


def begin_allocator_history(
    region: str,
    step: int,
    *,
    force: bool = False,
) -> bool:
    """Enter an exact allocator-history region, if one was requested.

    ``force`` means that an enabled backend should include this region even
    when it is outside the steady-iteration selection.  It deliberately does
    not override the environment gate: normal training must remain a no-op.
    This tree currently has no exact allocator-history backend, so an explicit
    request fails closed rather than falling back to the incomplete Python
    tensor-factory tracer.
    """

    del force
    if not _allocator_history_requested():
        return False
    _raise_allocator_history_unavailable(str(region), int(step))
    return False  # pragma: no cover - documents the bool API


def end_allocator_history(region: str, step: int) -> None:
    """Leave an allocator-history region begun by ``begin_allocator_history``."""

    if not _allocator_history_requested():
        return
    _raise_allocator_history_unavailable(str(region), int(step))


def begin_steady_allocator_history(step: int) -> bool:
    """Enter one explicitly selected steady checkpoint region.

    Unselected iterations return immediately so enabling ordinary VRAM
    sampling never serializes or perturbs the 10K hot path.
    """

    if not _allocator_history_requested():
        return False
    selected = _configured_steady_allocator_history_iters()
    if int(step) not in selected:
        return False
    _raise_allocator_history_unavailable("steady", int(step))
    return False  # pragma: no cover - documents the bool API


def end_steady_allocator_history(step: int) -> None:
    """Leave a selected steady allocator-history region."""

    if not _allocator_history_requested():
        return
    if int(step) not in _configured_steady_allocator_history_iters():
        return
    _raise_allocator_history_unavailable("steady", int(step))


def _alloc_trace_active() -> bool:
    if not _ALLOC_TRACE_ENABLED:
        return False
    lo, hi = _ALLOC_TRACE_ITER_RANGE
    return lo <= _CURRENT_ITER <= hi


def _alloc_trace_callsite_proposal() -> Optional[str]:
    frame = sys._getframe(2)
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename.startswith(_PROPOSAL_DIR):
            return f"{os.path.basename(filename)}:{frame.f_lineno}"
        frame = frame.f_back
    return None


def _make_alloc_patch(orig_fn, name: str):
    def wrapper(*args, **kwargs):
        tensor = orig_fn(*args, **kwargs)
        if _alloc_trace_active():
            try:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    callsite = _alloc_trace_callsite_proposal()
                    if callsite is not None:
                        nbytes = tensor.element_size() * tensor.numel()
                        _ALLOC_TRACE_RECORDS.append((callsite, nbytes, str(tensor.dtype)))
            except Exception:
                pass
        return tensor

    wrapper.__name__ = name
    return wrapper


def _make_clone_patch(orig_clone):
    def wrapper(self_tensor, *args, **kwargs):
        tensor = orig_clone(self_tensor, *args, **kwargs)
        if _alloc_trace_active():
            try:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    callsite = _alloc_trace_callsite_proposal()
                    if callsite is not None:
                        nbytes = tensor.element_size() * tensor.numel()
                        _ALLOC_TRACE_RECORDS.append(
                            (callsite, nbytes, str(tensor.dtype) + ".clone")
                        )
            except Exception:
                pass
        return tensor

    wrapper.__name__ = "clone"
    return wrapper


def install_alloc_trace() -> None:
    global _ALLOC_TRACE_ENABLED, _ALLOC_TRACE_PATCHED
    if os.environ.get("PROPOSAL_PER_ITER_ALLOC_TRACE", "") not in ("1", "true", "True"):
        return
    _ALLOC_TRACE_ENABLED = True
    if _ALLOC_TRACE_PATCHED:
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
    _ALLOC_TRACE_PATCHED = True
    print("[vram_audit] alloc trace patches installed", flush=True)


def alloc_trace_summary() -> List[Tuple[str, int, int, str]]:
    from collections import defaultdict

    aggregate: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
    for callsite, size_bytes, dtype in _ALLOC_TRACE_RECORDS:
        key = (callsite, dtype)
        aggregate[key][0] += size_bytes
        aggregate[key][1] += 1
    return sorted(
        [
            (callsite, total_bytes, count, dtype)
            for (callsite, dtype), (total_bytes, count) in aggregate.items()
        ],
        key=lambda item: -item[1],
    )


def dump_alloc_trace_to_log() -> None:
    if not is_vram_audit_enabled():
        return
    entries = [
        {
            "callsite": callsite,
            "total_mb": total_bytes / 1e6,
            "count": count,
            "dtype": dtype,
        }
        for callsite, total_bytes, count, dtype in alloc_trace_summary()[:30]
    ]
    _record_vram_audit(
        "alloc_trace_summary",
        {
            "alloc_trace": entries,
            "total_records": len(_ALLOC_TRACE_RECORDS),
        },
    )


# ---------------------------------------------------------------------------
# Per-iter diagnostics.
# ---------------------------------------------------------------------------
def _proposal_post_iter_diagnostics(model, step: int) -> None:
    """No-op hook retained for the proposal facade."""

    return
