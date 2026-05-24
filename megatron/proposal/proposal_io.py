"""Single Python facade for proposal CP I/O.

C++ extension (``_csrc/proposal_io.cpp`` + ``_csrc/binary_io.cpp``) is
JIT-built on first import via ``torch.utils.cpp_extension.load`` and
cached under ~/.cache/torch_extensions. First import ~30 s; subsequent
imports immediate. If the build fails, the module raises a clear
ImportError — there is no Python fallback for these hot paths.

Layout:
  * C++ exposed via _cext:  stamp_header_into_pinned / write_prptens /
                             write_lossless / open_lossy_log_for_append /
                             append_lossy_log / fsync_fd / close_fd.
    Constants exposed as module attrs (HEADER_BLOCK_BYTES, SECTOR_BYTES).
    (Phase 3 removed the legacy CPU compute paths
    ``scatter_into_flat`` / ``aos_write`` — see gpu_aos.py for the
    on-device replacement.)
  * Python here:           alloc_pinned_uint8_sector_aligned (uses
                             torch.empty(pin_memory=True)),
                             build_prptens_header_block (json/struct),
                             load_binary / is_binary_format (recovery
                             read path), thin write_*_from_pinned
                             wrappers that compute padded sizes and
                             call C++.

The pinned-buffer contract (base + length both 4 KiB-aligned) is
enforced inside the C++ binary_io layer; alloc_pinned_uint8_sector_aligned
guarantees the base alignment, and the write wrappers below round the
length up to SECTOR_BYTES before passing to C++.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.cpp_extension import load


# ---------------------------------------------------------------------------
# C++ extension load (no Python fallback — these are hot paths).
# ---------------------------------------------------------------------------
_src_dir = Path(__file__).parent / "_csrc"
_cext = load(
    name="proposal_io_cext",
    sources=[
        str(_src_dir / "binary_io.cpp"),
        str(_src_dir / "proposal_io.cpp"),
    ],
    extra_cflags=["-O3", "-std=c++17", "-fopenmp"],
    extra_ldflags=["-lgomp"],
    extra_include_paths=[str(_src_dir)],
    verbose=False,
)


# Re-export constants from C++ side.
HEADER_BLOCK_BYTES: int = int(_cext.HEADER_BLOCK_BYTES)
SECTOR_BYTES: int = int(_cext.SECTOR_BYTES)

# Pin-buffer chunk granularity (used by callers that want to round up
# the per-file pinned allocation to a larger chunk so the C++ writer's
# LBA bookkeeping stays uniform). Override via PROPOSAL_PIN_CHUNK_MB.
_DEFAULT_PIN_CHUNK_MB = 64
try:
    PIN_CHUNK_BYTES: int = max(
        1, int(os.environ.get("PROPOSAL_PIN_CHUNK_MB", _DEFAULT_PIN_CHUNK_MB))
    ) * 1024 * 1024
except ValueError:
    PIN_CHUNK_BYTES = _DEFAULT_PIN_CHUNK_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Pinned-buffer allocation (Python — uses torch.empty(pin_memory=True)).
# ---------------------------------------------------------------------------
def round_up_to_chunk(nbytes: int, chunk: int = PIN_CHUNK_BYTES) -> int:
    if chunk <= 0:
        return int(nbytes)
    n = int(nbytes)
    return ((n + chunk - 1) // chunk) * chunk


def _round_up_sector(nbytes: int) -> int:
    n = int(nbytes)
    return ((n + SECTOR_BYTES - 1) // SECTOR_BYTES) * SECTOR_BYTES


def alloc_pinned_uint8_sector_aligned(nbytes: int) -> torch.Tensor:
    """Pinned uint8 buffer whose base ptr and length are both 4 KiB-aligned.

    Torch's caching pinned allocator does NOT guarantee 4 KiB-aligned
    base pointers (verified: first alloc is page-aligned, subsequent ones
    sit at arbitrary 512 B offsets inside the same pinned arena). We
    over-allocate by SECTOR_BYTES-1 and slice forward to the next
    4 KiB-aligned offset. The returned tensor's storage holds the full
    over-allocated buffer so the slice stays valid.
    Pad tail is zero-filled for determinism.
    """
    n = int(nbytes)
    padded = _round_up_sector(n)
    over = torch.empty(padded + SECTOR_BYTES - 1, dtype=torch.uint8, pin_memory=True)
    base = int(over.data_ptr())
    misalign = base & (SECTOR_BYTES - 1)
    skip = 0 if misalign == 0 else (SECTOR_BYTES - misalign)
    t = over.narrow(0, skip, padded)
    assert (int(t.data_ptr()) & (SECTOR_BYTES - 1)) == 0, "aligned slice failed"
    if padded > n:
        t[n:].zero_()
    return t


# ---------------------------------------------------------------------------
# PRPTENS on-disk format.
# ---------------------------------------------------------------------------
MAGIC: bytes = b"PRPTENS\x00"
VERSION: int = 2

_TORCH_DTYPE_STR = {
    torch.float32: "float32", torch.float16: "float16",
    torch.bfloat16: "bfloat16", torch.int32: "int32",
    torch.int64: "int64", torch.uint8: "uint8", torch.int8: "int8",
}
_STR_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_DTYPE_STR.items()}
_STR_TO_NUMPY_DTYPE = {
    "float32": np.float32, "float16": np.float16, "int32": np.int32,
    "int64": np.int64, "uint8": np.uint8, "int8": np.int8,
}


def build_prptens_header_block(
    plan: Iterable[Tuple[str, int, int, "torch.Size"]],
    scalars: Optional[Dict[str, Any]] = None,
    *,
    dtype: str = "float32",
) -> bytes:
    """Build a PRPTENS header that fits inside HEADER_BLOCK_BYTES."""
    entries: List[Dict[str, Any]] = []
    for key, fp32_off, fp32_n, shape in plan:
        n = int(fp32_n)
        entries.append({
            "name": str(key),
            "dtype": dtype,
            "shape": [int(d) for d in shape],
            "offset": int(fp32_off) * 4,
            "nbytes": n * 4,
        })
    header: Dict[str, Any] = {"entries": entries}
    if scalars:
        header["scalars"] = dict(scalars)
    header_json = json.dumps(header).encode("utf-8")
    prefix = MAGIC + struct.pack("<II", VERSION, len(header_json)) + header_json
    if len(prefix) > HEADER_BLOCK_BYTES:
        raise RuntimeError(
            f"PRPTENS header ({len(prefix)} B) exceeds HEADER_BLOCK_BYTES "
            f"({HEADER_BLOCK_BYTES})"
        )
    return prefix


def is_binary_format(path: Path) -> bool:
    """Magic-byte probe; safe on missing/truncated files."""
    if _cext.file_size(str(path)) < len(MAGIC):
        return False
    return _cext.read_file_range(str(path), 0, len(MAGIC)) == MAGIC


def load_binary(
    path: Path, device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Read a PRPTENS file. Returns (tensor_dict, scalars). All file IO
    goes through the C++ binary_io::read_file_range; this function is
    pure header parse + tensor reconstruction."""
    path_s = str(Path(path))
    header_block = _cext.read_file_range(path_s, 0, HEADER_BLOCK_BYTES)
    if len(header_block) < 16:
        raise ValueError(f"load_binary: header too short in {path}")
    if header_block[:len(MAGIC)] != MAGIC:
        raise ValueError(
            f"load_binary: bad magic in {path}; expected {MAGIC!r}")
    version, header_len = struct.unpack("<II", header_block[8:16])
    if version != VERSION:
        raise ValueError(
            f"load_binary: unsupported version {version} (expected {VERSION})")
    header = json.loads(header_block[16:16 + header_len].decode("utf-8"))
    entries = header["entries"]
    scalars: Dict[str, Any] = dict(header.get("scalars", {}))
    out: Dict[str, torch.Tensor] = {}
    for e in entries:
        off = int(e["offset"])
        shape = tuple(int(x) for x in e["shape"])
        dtype_str = e["dtype"]
        nbytes = int(e["nbytes"])
        raw = _cext.read_file_range(path_s, HEADER_BLOCK_BYTES + off, nbytes)
        if dtype_str == "bfloat16":
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()
            t = torch.from_numpy(arr).view(torch.bfloat16)
        else:
            np_dtype = _STR_TO_NUMPY_DTYPE.get(dtype_str)
            if np_dtype is None:
                raise ValueError(
                    f"load_binary: no numpy dtype mapping for {dtype_str!r}")
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
            t = torch.from_numpy(arr)
        if device is not None:
            t = t.to(device)
        out[e["name"]] = t
    return out, scalars


# ---------------------------------------------------------------------------
# Write helpers (thin wrappers that compute padded sizes + call C++).
# ---------------------------------------------------------------------------
def write_prptens_from_pinned(
    final_path: Path,
    pinned_tensor: torch.Tensor,
    header_bytes: bytes,
    payload_bytes: int,
    *,
    step: int,
    mode: int = 0o644,
    fsync: bool = False,
) -> int:
    """Write a PRPTENS file in one full_write from a pinned uint8 buffer.

    ``pinned_tensor`` MUST be the tensor returned by
    alloc_pinned_uint8_sector_aligned (base ptr 4 KiB-aligned, length
    rounded up to SECTOR_BYTES). The C++ side asserts both.

    The write covers (HEADER_BLOCK_BYTES + payload_bytes), rounded up
    to the next SECTOR_BYTES multiple. Pad bytes between data end and
    sector boundary are already zero-filled by the allocator.
    """
    del step  # routing metadata only — consumed by io_uring path
    total_padded = _round_up_sector(HEADER_BLOCK_BYTES + int(payload_bytes))
    return int(_cext.write_prptens(
        str(Path(final_path)), int(pinned_tensor.data_ptr()),
        header_bytes, int(total_padded), int(mode), bool(fsync),
    ))


def write_lossless_from_pinned(
    final_path: Path,
    pinned_tensor: torch.Tensor,
    header_bytes: bytes,
    payload_bytes: int,
    *,
    layer: int,
    expert: int,
    step: int,
    mode: int = 0o644,
) -> int:
    """Atomic LFCL write: pinned → <final_path>.tmp → fsync → rename."""
    del layer, expert, step
    total_padded = _round_up_sector(HEADER_BLOCK_BYTES + int(payload_bytes))
    return int(_cext.write_lossless(
        str(Path(final_path)), int(pinned_tensor.data_ptr()),
        header_bytes, int(total_padded), int(mode),
    ))


def open_lossy_log_for_append(
    path: Path, header_bytes: bytes, *, mode: int = 0o644,
) -> int:
    """Create+truncate the lossy file, write a HEADER_BLOCK_BYTES header
    region (chunk-aligned), return the open fd. Reader treats the first
    HEADER_BLOCK_BYTES of the file as the header block (tail zero-filled
    if the actual header is smaller)."""
    # Allocate a single-shot 4 KiB-aligned pinned buffer for the header,
    # let C++ stamp it + write it, then drop the ref. Steady-state lossy
    # writes use append_lossy_log_from_pinned with already-padded blobs.
    hdr_pin = alloc_pinned_uint8_sector_aligned(HEADER_BLOCK_BYTES)
    return int(_cext.open_lossy_log_for_append(
        str(Path(path)), int(hdr_pin.data_ptr()),
        header_bytes, int(mode),
    ))


def append_lossy_log_from_pinned(
    fd: int, pinned_tensor: torch.Tensor, total_bytes: int,
    *,
    layer: int,
    expert: int,
    step: int,
) -> int:
    """Append a (count u32, step u32, records...) blob to ``fd`` via
    full_write_pinned. ``pinned_tensor`` must be 4 KiB-aligned (base +
    length); ``total_bytes`` is rounded up to the next SECTOR_BYTES
    multiple — the pad tail in pinned_tensor is already zero-filled."""
    del layer, expert, step
    total_padded = _round_up_sector(total_bytes)
    return int(_cext.append_lossy_log(
        int(fd), int(pinned_tensor.data_ptr()), int(total_padded),
    ))


def fsync_fd(fd: int) -> None:
    _cext.fsync_fd(int(fd))


def close_fd(fd: int) -> None:
    _cext.close_fd(int(fd))


def unlink_path(path) -> None:
    """``::unlink(path)``; swallows ENOENT for idempotent callers."""
    _cext.unlink_path(str(path))


def read_file_range(path, offset: int, nbytes: int) -> bytes:
    """``::pread(path, offset, nbytes)``; returns bytes (≤ nbytes if EOF)."""
    return _cext.read_file_range(str(path), int(offset), int(nbytes))


def file_size(path) -> int:
    """``::stat(path).st_size`` or -1 if the file is missing."""
    return int(_cext.file_size(str(path)))
