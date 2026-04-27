"""Tests for DramExpertState — in-memory drop-in replacement for ExpertState.

Covers:
- API parity (same public methods, callable with same signatures)
- Lossless FC overwrites the dense buffer wholesale
- Lossy FC append scatters into the buffer (last-write-wins on duplicate idx)
- read() returns a payload dict shaped for ``_apply_payload_dict_to_model``
- Total bytes accounting matches the bounded dense footprint (no log growth)
"""
from pathlib import Path

import pytest
import torch

from megatron.proposal.dram_expert_state import DramExpertState


def _toy_schema():
    """Two experts (0,0)+(0,1) each with two tensors. Sized so any test
    can index a few specific positions without needing large tensors."""
    return {
        (0, 0): [
            ("layer.0.expert.0.w1", torch.Size([4, 4]), 16),
            ("layer.0.expert.0.w2", torch.Size([2, 2]), 4),
        ],
        (0, 1): [
            ("layer.0.expert.1.w1", torch.Size([4, 4]), 16),
            ("layer.0.expert.1.w2", torch.Size([2, 2]), 4),
        ],
    }


def test_construct_zero_buffers(tmp_path):
    state = DramExpertState(_toy_schema(), output_dir=tmp_path)
    assert sorted(state.expert_ids()) == [(0, 0), (0, 1)]
    # Total bytes = 3 (p,m,v) * 4 bytes/fp32 * (16 + 4) per expert * 2 experts
    assert state.total_bytes == 3 * 4 * (16 + 4) * 2
    # No lossless yet for either expert.
    for eid in state.expert_ids():
        assert state.lossless_fc_step(eid) == -1
        assert not state.has_lossless_fc(eid)


def test_lossless_fc_overwrites_dense(tmp_path):
    state = DramExpertState(_toy_schema(), output_dir=tmp_path)
    p = torch.arange(16, dtype=torch.float32)
    m = torch.full((16,), 0.5)
    v = torch.full((16,), 0.25)
    state.lossless_fc(
        (0, 0), step=42,
        initial_state={"layer.0.expert.0.w1": (p, m, v)},
    )
    assert state.lossless_fc_step((0, 0)) == 42
    assert state.has_lossless_fc((0, 0))
    payload = state.read((0, 0))
    assert payload is not None
    # Buffer == provided tensor at every position.
    assert torch.allclose(payload["layer.0.expert.0.w1:param"], p)
    assert torch.allclose(payload["layer.0.expert.0.w1:exp_avg"], m)
    assert torch.allclose(payload["layer.0.expert.0.w1:exp_avg_sq"], v)
    # The other tensor (no triple supplied for it) stays at zero.
    assert torch.all(payload["layer.0.expert.0.w2:param"] == 0)


def test_lossy_fc_append_scatter(tmp_path):
    state = DramExpertState(_toy_schema(), output_dir=tmp_path)
    # Lossy update at indices [0, 2, 5, 11] of w1.
    idx = torch.tensor([0, 2, 5, 11], dtype=torch.int32)
    p_vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
    m_vals = torch.tensor([0.1, 0.2, 0.3, 0.4])
    v_vals = torch.tensor([0.01, 0.02, 0.03, 0.04])
    written = state.lossy_fc_append(
        (0, 0), step=10,
        tensor_updates=[
            ("layer.0.expert.0.w1", idx, p_vals, m_vals, v_vals),
        ],
    )
    assert written == 4
    payload = state.read((0, 0))
    p_buf = payload["layer.0.expert.0.w1:param"]
    # Scattered positions equal the new values.
    for i, src_i in enumerate(idx.tolist()):
        assert p_buf[src_i].item() == pytest.approx(p_vals[i].item())
    # All other positions remain zero (no lossless FC fired).
    not_set = set(range(16)) - set(idx.tolist())
    for j in not_set:
        assert p_buf[j].item() == 0.0


def test_lossy_fc_last_write_wins(tmp_path):
    """Two consecutive lossy_fc_append calls with overlapping indices —
    the second's value must win (matches ExpertState's log replay
    semantics)."""
    state = DramExpertState(_toy_schema(), output_dir=tmp_path)
    state.lossy_fc_append(
        (0, 0), step=10,
        tensor_updates=[(
            "layer.0.expert.0.w1",
            torch.tensor([3, 5], dtype=torch.int32),
            torch.tensor([100.0, 200.0]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.1, 0.2]),
        )],
    )
    state.lossy_fc_append(
        (0, 0), step=11,
        tensor_updates=[(
            "layer.0.expert.0.w1",
            torch.tensor([3], dtype=torch.int32),  # overlaps idx=3 from above
            torch.tensor([999.0]),                  # new value
            torch.tensor([9.0]),
            torch.tensor([0.9]),
        )],
    )
    payload = state.read((0, 0))
    p_buf = payload["layer.0.expert.0.w1:param"]
    # Second write wins at idx=3.
    assert p_buf[3].item() == pytest.approx(999.0)
    # First write still in place at idx=5.
    assert p_buf[5].item() == pytest.approx(200.0)


def test_lossless_clears_prior_lossy_drift(tmp_path):
    """After a fresh lossless FC, prior lossy updates must be dropped
    (consistent with ExpertState unlinking the lossy log on lossless)."""
    state = DramExpertState(_toy_schema(), output_dir=tmp_path)
    # Apply a lossy update first.
    state.lossy_fc_append(
        (0, 0), step=10,
        tensor_updates=[(
            "layer.0.expert.0.w1",
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([7.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
        )],
    )
    # Then a lossless FC with all-zero tensors.
    p_zero = torch.zeros(16)
    m_zero = torch.zeros(16)
    v_zero = torch.zeros(16)
    state.lossless_fc(
        (0, 0), step=20,
        initial_state={"layer.0.expert.0.w1": (p_zero, m_zero, v_zero)},
    )
    payload = state.read((0, 0))
    # The 7.0 we wrote at idx=0 must be GONE — wholesale overwrite.
    assert payload["layer.0.expert.0.w1:param"][0].item() == 0.0


def test_unknown_tensor_name_skipped(tmp_path):
    """Passing an out-of-schema tensor name must not raise; we silently
    skip (matches ExpertState — it filters via global_offsets membership)."""
    state = DramExpertState(_toy_schema(), output_dir=tmp_path)
    n = state.lossy_fc_append(
        (0, 0), step=1,
        tensor_updates=[(
            "not.in.schema",
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
        )],
    )
    assert n == 0


def test_read_includes_rolling_step(tmp_path):
    """The payload must carry _rolling_step so
    ``_apply_payload_dict_to_model`` can record the latest captured iter."""
    state = DramExpertState(_toy_schema(), output_dir=tmp_path)
    state.lossy_fc_append(
        (0, 0), step=42,
        tensor_updates=[(
            "layer.0.expert.0.w1",
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
        )],
    )
    payload = state.read((0, 0))
    assert "_rolling_step" in payload
    assert int(payload["_rolling_step"].item()) == 42


def test_flush_and_shutdown_are_safe_noops(tmp_path):
    """No fds, but the API contract requires both to exist and not
    raise — FCManager / cp_hooks call them uniformly."""
    state = DramExpertState(_toy_schema(), output_dir=tmp_path)
    state.flush()
    state.flush(only_eids=[(0, 0)])
    state.shutdown()
    # Buffers are still readable post-shutdown (recovery does this).
    assert state.read((0, 0)) is not None


def test_unknown_expert_returns_none(tmp_path):
    state = DramExpertState(_toy_schema(), output_dir=tmp_path)
    assert state.read((99, 99)) is None
    # Calls on unknown eid silently no-op.
    state.lossless_fc((99, 99), step=1, initial_state={})
    assert state.lossy_fc_append((99, 99), step=1, tensor_updates=[]) == 0
