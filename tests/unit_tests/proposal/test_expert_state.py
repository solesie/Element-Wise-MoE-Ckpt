"""Unit tests for expert_state.py (lossy FC append + round-robin lossless FC)."""
import os
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from megatron.proposal.expert_state import (
    HEADER_BLOCK_BYTES,
    MAGIC_LFCL,
    MAGIC_LFSY,
    RECORD_BYTES,
    ExpertState,
    _unpack_header,
    read_expert_state,
)


@pytest.fixture
def tiny_expert_map():
    return {
        (0, 0): [("mlp.w1", torch.Size([8]), 8),
                 ("mlp.w2", torch.Size([16]), 16)],
        (0, 1): [("mlp.w1", torch.Size([8]), 8),
                 ("mlp.w2", torch.Size([16]), 16)],
    }


def test_fresh_creates_no_files(tmp_path: Path, tiny_expert_map):
    state = ExpertState(tiny_expert_map, output_dir=tmp_path, fresh=True)
    # Until lossless_fc or lossy_fc_append is called, no files exist.
    assert list(tmp_path.iterdir()) == []
    state.shutdown()


def test_lossless_fc_writes_full_snapshot_and_opens_fresh_lossy(
    tmp_path: Path, tiny_expert_map,
):
    state = ExpertState(tiny_expert_map, output_dir=tmp_path, fresh=True)
    init = {
        "mlp.w1": (torch.arange(8, dtype=torch.float32),
                   torch.zeros(8), torch.zeros(8)),
        "mlp.w2": (torch.arange(16, dtype=torch.float32) + 100,
                   torch.ones(16), torch.full((16,), 0.5)),
    }
    state.lossless_fc((0, 0), step=10, initial_state=init)

    lossless = tmp_path / "fc_expert_L00_E00.bin"
    lossy = tmp_path / "lossy_expert_L00_E00.bin"
    assert lossless.exists()
    assert lossy.exists()
    # Lossy file has header only (no append yet).
    assert lossy.stat().st_size == HEADER_BLOCK_BYTES
    # Lossless payload size = (8 + 16) * 3 * 4 = 288 bytes.
    assert lossless.stat().st_size == HEADER_BLOCK_BYTES + 288

    # Verify lossless header + content.
    with open(lossless, "rb") as f:
        hdr = f.read(HEADER_BLOCK_BYTES)
        payload = f.read()
    magic, _ver, meta = _unpack_header(hdr)
    assert magic == MAGIC_LFCL
    assert meta["step"] == 10
    assert meta["total_numel"] == 24
    # payload = param (8) | m (8) | v (8) | param (16) | m (16) | v (16)
    p1 = np.frombuffer(payload[0:32], dtype=np.float32)
    m1 = np.frombuffer(payload[32:64], dtype=np.float32)
    v1 = np.frombuffer(payload[64:96], dtype=np.float32)
    assert np.allclose(p1, np.arange(8, dtype=np.float32))
    assert np.allclose(m1, np.zeros(8, dtype=np.float32))
    assert np.allclose(v1, np.zeros(8, dtype=np.float32))

    assert state.lossless_fc_step((0, 0)) == 10
    state.shutdown()


def test_lossy_fc_append_writes_per_iter_block(tmp_path: Path, tiny_expert_map):
    state = ExpertState(tiny_expert_map, output_dir=tmp_path, fresh=True)
    # Seed with lossless FC first so lossy file has a header.
    init = {
        "mlp.w1": (torch.zeros(8), torch.zeros(8), torch.zeros(8)),
        "mlp.w2": (torch.zeros(16), torch.zeros(16), torch.zeros(16)),
    }
    state.lossless_fc((0, 0), step=0, initial_state=init)

    # Append at step=1: w1[idx=3]=11.0, w2[idx=5]=55.0
    updates = [
        ("mlp.w1", torch.tensor([3], dtype=torch.int32),
         torch.tensor([11.0]), torch.tensor([1.0]), torch.tensor([0.1])),
        ("mlp.w2", torch.tensor([5], dtype=torch.int32),
         torch.tensor([55.0]), torch.tensor([5.0]), torch.tensor([0.5])),
    ]
    n = state.lossy_fc_append((0, 0), step=1, tensor_updates=updates)
    assert n == 2
    state.shutdown()

    # Verify on-disk block.
    lossy = tmp_path / "lossy_expert_L00_E00.bin"
    with open(lossy, "rb") as f:
        hdr = f.read(HEADER_BLOCK_BYTES)
        block = f.read()
    magic, _ver, meta = _unpack_header(hdr)
    assert magic == MAGIC_LFSY
    assert meta["start_step"] == 0
    # Block layout: count(4) + step(4) + 2*16 = 40 B.
    count, step = struct.unpack("<II", block[:8])
    assert count == 2 and step == 1
    records = np.frombuffer(block[8:40],
                             dtype=np.dtype([("idx", "<i4"), ("p", "<f4"),
                                             ("m", "<f4"), ("v", "<f4")]))
    # w1 offset=0, w2 offset=8 → idx=3 and idx=8+5=13.
    assert sorted(records["idx"].tolist()) == [3, 13]


def test_read_expert_state_combines_lossless_and_lossy(
    tmp_path: Path, tiny_expert_map,
):
    state = ExpertState(tiny_expert_map, output_dir=tmp_path, fresh=True)
    # Seed: param = [0..7] for w1, [100..115] for w2.
    init = {
        "mlp.w1": (torch.arange(8, dtype=torch.float32),
                   torch.zeros(8), torch.zeros(8)),
        "mlp.w2": (torch.arange(16, dtype=torch.float32) + 100,
                   torch.zeros(16), torch.zeros(16)),
    }
    state.lossless_fc((0, 0), step=0, initial_state=init)
    # Overwrite w1[3] = 999 at step=1; w2[5] at step=2; w1[3] again at step=3.
    state.lossy_fc_append(
        (0, 0), step=1,
        tensor_updates=[
            ("mlp.w1", torch.tensor([3], dtype=torch.int32),
             torch.tensor([999.0]), torch.tensor([0.0]), torch.tensor([0.0])),
        ],
    )
    state.lossy_fc_append(
        (0, 0), step=2,
        tensor_updates=[
            ("mlp.w2", torch.tensor([5], dtype=torch.int32),
             torch.tensor([555.0]), torch.tensor([0.0]), torch.tensor([0.0])),
        ],
    )
    state.lossy_fc_append(
        (0, 0), step=3,
        tensor_updates=[
            ("mlp.w1", torch.tensor([3], dtype=torch.int32),
             torch.tensor([42.0]), torch.tensor([0.0]), torch.tensor([0.0])),
        ],
    )
    state.shutdown()

    payload = read_expert_state(tmp_path, layer=0, expert=0)
    assert payload["_rolling_step"] == 3
    w1 = payload["mlp.w1:param"].reshape(-1)
    w2 = payload["mlp.w2:param"].reshape(-1)
    # w1[3] last written at step 3 → 42.0
    assert float(w1[3].item()) == pytest.approx(42.0)
    # Other w1 positions = seed values.
    for i in (0, 1, 2, 4, 5, 6, 7):
        assert float(w1[i].item()) == pytest.approx(float(i))
    # w2[5] = 555.0 (step 2); other w2 = seed.
    assert float(w2[5].item()) == pytest.approx(555.0)
    for i in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        assert float(w2[i].item()) == pytest.approx(100.0 + i)


def test_lossless_fc_unlinks_previous_lossy_log(tmp_path: Path, tiny_expert_map):
    """Spec: expert lossless FC wipes the prior lossy FC log."""
    state = ExpertState(tiny_expert_map, output_dir=tmp_path, fresh=True)
    init_zeros = {
        "mlp.w1": (torch.zeros(8), torch.zeros(8), torch.zeros(8)),
        "mlp.w2": (torch.zeros(16), torch.zeros(16), torch.zeros(16)),
    }
    state.lossless_fc((0, 0), step=0, initial_state=init_zeros)
    # Grow the lossy file.
    for s in range(1, 5):
        state.lossy_fc_append(
            (0, 0), step=s,
            tensor_updates=[
                ("mlp.w1", torch.tensor([0, 1, 2, 3], dtype=torch.int32),
                 torch.ones(4) * float(s), torch.zeros(4), torch.zeros(4)),
            ],
        )
    lossy_path = tmp_path / "lossy_expert_L00_E00.bin"
    size_before = lossy_path.stat().st_size
    assert size_before > HEADER_BLOCK_BYTES  # grew past header-only

    # Fire a second lossless FC → should truncate the lossy log back to
    # header-only.
    state.lossless_fc((0, 0), step=10, initial_state=init_zeros)
    size_after = lossy_path.stat().st_size
    assert size_after == HEADER_BLOCK_BYTES  # reset
    assert state.lossless_fc_step((0, 0)) == 10
    state.shutdown()


def test_independent_experts(tmp_path: Path, tiny_expert_map):
    """Writes to (0,0) must not affect (0,1)."""
    state = ExpertState(tiny_expert_map, output_dir=tmp_path, fresh=True)
    state.lossless_fc(
        (0, 0), step=0,
        initial_state={
            "mlp.w1": (torch.ones(8) * 1.0, torch.zeros(8), torch.zeros(8)),
            "mlp.w2": (torch.ones(16) * 2.0, torch.zeros(16), torch.zeros(16)),
        },
    )
    state.lossless_fc(
        (0, 1), step=0,
        initial_state={
            "mlp.w1": (torch.ones(8) * 9.0, torch.zeros(8), torch.zeros(8)),
            "mlp.w2": (torch.ones(16) * 8.0, torch.zeros(16), torch.zeros(16)),
        },
    )
    state.shutdown()

    p0 = read_expert_state(tmp_path, 0, 0)
    p1 = read_expert_state(tmp_path, 0, 1)
    assert torch.allclose(p0["mlp.w1:param"].reshape(-1), torch.ones(8) * 1.0)
    assert torch.allclose(p1["mlp.w1:param"].reshape(-1), torch.ones(8) * 9.0)


def test_empty_append_is_noop(tmp_path: Path, tiny_expert_map):
    state = ExpertState(tiny_expert_map, output_dir=tmp_path, fresh=True)
    # Append with empty indices — should NOT create a file or raise.
    n = state.lossy_fc_append(
        (0, 0), step=5,
        tensor_updates=[
            ("mlp.w1", torch.empty(0, dtype=torch.int32),
             torch.empty(0), torch.empty(0), torch.empty(0)),
        ],
    )
    assert n == 0
    assert not (tmp_path / "lossy_expert_L00_E00.bin").exists()
    state.shutdown()


def test_read_expert_state_returns_minimal_when_no_files(tmp_path: Path):
    # empty dir → {_rolling_step: 0}
    payload = read_expert_state(tmp_path, layer=0, expert=0)
    assert payload == {"_rolling_step": 0}
