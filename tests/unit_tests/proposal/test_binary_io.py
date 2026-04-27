"""Round-trip tests for megatron.proposal.binary_io.

Covers: magic detection, FP32 tensors (the common non-expert FC / H_ema
case), scalar metadata roundtrip, dtype variety, and rejection of mismatched
or corrupted files.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from megatron.proposal.binary_io import (
    MAGIC,
    is_binary_format,
    load_binary,
    save_binary,
)


def test_fp32_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "fp32.bin"
    a = torch.randn(4, 8, dtype=torch.float32)
    b = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    save_binary(path, {"a": a, "b": b})
    tensors, scalars = load_binary(path)
    assert scalars == {}
    assert set(tensors.keys()) == {"a", "b"}
    torch.testing.assert_close(tensors["a"], a)
    torch.testing.assert_close(tensors["b"], b)


def test_scalars_roundtrip(tmp_path: Path) -> None:
    """H_ema style: scalar header carries iter/updates/alpha plus tensor dict."""
    path = tmp_path / "h_ema.bin"
    tensors = {f"exp.{i}.w": torch.randn(3, 3) for i in range(3)}
    scalars = {"iter": 100, "updates": 9, "alpha": 0.9}
    save_binary(path, tensors, scalars=scalars)
    got_tensors, got_scalars = load_binary(path)
    assert got_scalars == scalars
    for k in tensors:
        torch.testing.assert_close(got_tensors[k], tensors[k])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32,
                                    torch.int64, torch.uint8, torch.bfloat16])
def test_dtype_roundtrip(tmp_path: Path, dtype) -> None:
    path = tmp_path / f"dt_{dtype}.bin"
    if dtype.is_floating_point:
        t = torch.randn(5, dtype=torch.float32).to(dtype)
    else:
        t = torch.arange(5, dtype=torch.int64).to(dtype)
    save_binary(path, {"x": t})
    tensors, _ = load_binary(path)
    assert tensors["x"].dtype == dtype
    torch.testing.assert_close(tensors["x"].to(torch.float32), t.to(torch.float32))


def test_is_binary_format_positive(tmp_path: Path) -> None:
    path = tmp_path / "probe.bin"
    save_binary(path, {"x": torch.zeros(1)})
    assert is_binary_format(path)


def test_is_binary_format_negative_on_pickle(tmp_path: Path) -> None:
    """A torch.save pickle file must NOT be detected as binary."""
    path = tmp_path / "pickle.pt"
    torch.save({"x": torch.zeros(1)}, path)
    assert not is_binary_format(path)


def test_is_binary_format_missing_file(tmp_path: Path) -> None:
    assert not is_binary_format(tmp_path / "doesnotexist.bin")


def test_rejects_bad_magic(tmp_path: Path) -> None:
    """load_binary raises on non-PRPTENS files (not just pickle)."""
    path = tmp_path / "garbage.bin"
    path.write_bytes(b"NOTOURFORMAT" + b"\x00" * 32)
    with pytest.raises(ValueError, match="bad magic"):
        load_binary(path)


def test_magic_bytes_constant() -> None:
    """Sanity: MAGIC length stays at 8 and starts with PRPTENS."""
    assert len(MAGIC) == 8
    assert MAGIC.startswith(b"PRPTENS")


def test_large_contiguous_payload(tmp_path: Path) -> None:
    """Simulate non-expert FC payload style — tensors are views into one flat
    pinned buffer. Ensure save preserves order + shape."""
    path = tmp_path / "big.bin"
    flat = torch.arange(100 * 12, dtype=torch.float32)
    tensors = {
        f"layer{i}.weight": flat.narrow(0, i * 100, 100).view(10, 10)
        for i in range(12)
    }
    save_binary(path, tensors)
    got, _ = load_binary(path)
    for k, v in tensors.items():
        torch.testing.assert_close(got[k], v)


def test_device_move(tmp_path: Path) -> None:
    """device= kwarg moves tensors during load (lazily)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    path = tmp_path / "cuda.bin"
    save_binary(path, {"x": torch.randn(3, 3)})
    got, _ = load_binary(path, device=torch.device("cuda"))
    assert got["x"].is_cuda
