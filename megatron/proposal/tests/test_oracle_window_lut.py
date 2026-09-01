from __future__ import annotations

import pytest

from megatron.proposal.telemetry.oracle_window.lut import FiniteDifferenceAudit


def test_bf16_fd_audit_rejects_rounded_away_perturbations() -> None:
    audit = FiniteDifferenceAudit(
        parameter_dtype="bfloat16",
        epsilon=1.0e-3,
        compared_elements=1000,
        plus_changed_elements=792,
        minus_changed_elements=801,
        minimum_changed_fraction=0.95,
    )
    with pytest.raises(RuntimeError, match="rounded away"):
        audit.validate()


def test_bf16_fd_audit_accepts_measured_representable_probe() -> None:
    audit = FiniteDifferenceAudit(
        parameter_dtype="torch.bfloat16",
        epsilon=1.0e-2,
        compared_elements=1000,
        plus_changed_elements=997,
        minus_changed_elements=996,
        minimum_changed_fraction=0.95,
    )
    audit.validate()
    assert audit.plus_changed_fraction == pytest.approx(0.997)
    assert audit.minus_changed_fraction == pytest.approx(0.996)


def test_fd_audit_rejects_nonpositive_epsilon() -> None:
    audit = FiniteDifferenceAudit(
        parameter_dtype="float32",
        epsilon=0.0,
        compared_elements=10,
        plus_changed_elements=10,
        minus_changed_elements=10,
    )
    with pytest.raises(ValueError, match="epsilon"):
        audit.validate()
