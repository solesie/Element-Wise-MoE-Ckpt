"""CPU-only contract tests for proposal diagnostics call sites."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_DIAGNOSTICS_PATH = (
    Path(__file__).resolve().parents[2]
    / "megatron"
    / "proposal"
    / "telemetry"
    / "diagnostics.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "proposal_diagnostics_contract_test",
    _DIAGNOSTICS_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
diagnostics = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = diagnostics
_SPEC.loader.exec_module(diagnostics)


_AUDIT_ENV = (
    "PROPOSAL_VRAM_AUDIT",
    "PROPOSAL_PER_ITER_ALLOC_TRACE",
    "PROPOSAL_ALLOC_TRACE_ITERS",
)


def _clear_audit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AUDIT_ENV:
        monkeypatch.delenv(name, raising=False)


def test_disabled_diagnostics_are_immediate_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_audit_env(monkeypatch)

    # These are the exact signatures used by H refresh and the steady
    # dispatch hook.  They must be callable without touching CUDA in a normal
    # training run.
    diagnostics.vram_audit_step("anchor", synchronize=True)
    assert diagnostics.begin_allocator_history("h_refresh", 100, force=True) is False
    diagnostics.end_allocator_history("h_refresh", 100)
    assert diagnostics.begin_steady_allocator_history(111) is False
    diagnostics.end_steady_allocator_history(111)


def test_ordinary_vram_sampling_does_not_enable_exact_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_audit_env(monkeypatch)
    monkeypatch.setenv("PROPOSAL_VRAM_AUDIT", "1")

    assert diagnostics.begin_allocator_history("h_refresh", 100, force=True) is False
    assert diagnostics.begin_steady_allocator_history(111) is False


def test_unsupported_exact_history_request_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_audit_env(monkeypatch)
    monkeypatch.setenv("PROPOSAL_PER_ITER_ALLOC_TRACE", "1")
    monkeypatch.setenv("PROPOSAL_ALLOC_TRACE_ITERS", "111, 112")

    with pytest.raises(RuntimeError, match="no allocator-history backend"):
        diagnostics.begin_allocator_history("h_refresh", 100, force=True)
    assert diagnostics.begin_steady_allocator_history(110) is False
    with pytest.raises(RuntimeError, match="refusing to emit a misleading R2/R4 PASS"):
        diagnostics.begin_steady_allocator_history(111)


def test_invalid_steady_trace_selection_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_audit_env(monkeypatch)
    monkeypatch.setenv("PROPOSAL_PER_ITER_ALLOC_TRACE", "true")
    monkeypatch.setenv("PROPOSAL_ALLOC_TRACE_ITERS", "111,bad")

    with pytest.raises(RuntimeError, match="comma-separated list of integers"):
        diagnostics.begin_steady_allocator_history(111)


def test_unsynchronized_vram_sample_is_rejected_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_audit_env(monkeypatch)
    diagnostics.vram_audit_step("disabled", synchronize=False)

    monkeypatch.setenv("PROPOSAL_VRAM_AUDIT", "1")
    with pytest.raises(RuntimeError, match="unsynchronized"):
        diagnostics.vram_audit_step("enabled", synchronize=False)
