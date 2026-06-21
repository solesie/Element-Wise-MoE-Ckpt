"""Single Python facade for proposal CP I/O.

C++ extension (``_csrc/proposal_io.cpp`` + ``_csrc/binary_io.cpp``) is
JIT-built on first import via ``torch.utils.cpp_extension.load`` and
cached under ~/.cache/torch_extensions. First import ~30 s; subsequent
imports immediate. If the build fails, the module raises a clear
ImportError — there is no Python fallback for these hot paths.

Layout:
  * C++ exposed via _cext:  stamp_header_into_pinned / write_prptens /
                             open_lossy_log_for_append /
                             append_lossy_log / fsync_fd / close_fd.
    Constants exposed as module attrs (HEADER_BLOCK_BYTES, SECTOR_BYTES).
    The hot production AOS builder lives on GPU. A tiny CPU scatter helper
    remains for ``cp_target=test_dram`` only, where pinned host dense images
    are used for fast in-memory recovery checks.
  * Python here:           alloc_pinned_uint8_sector_aligned (uses
                             torch.empty(pin_memory=True)),
                             build_prptens_header_block (json/struct),
                             load_binary / is_binary_format (recovery
                             read path), thin pinned-buffer wrappers that
                             compute padded sizes and call C++.

The pinned-buffer contract (base + length both 4 KiB-aligned) is
enforced inside the C++ binary_io layer; alloc_pinned_uint8_sector_aligned
guarantees the base alignment, and the read/write wrappers below round the
length up to SECTOR_BYTES before passing to C++.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
from torch.utils.cpp_extension import load


# ---------------------------------------------------------------------------
# C++ extension load (no Python fallback — these are hot paths).
# ---------------------------------------------------------------------------
_src_dir = Path(__file__).resolve().parents[1] / "_csrc"
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

# Pin-buffer chunk granularity.  Keep fixed so checkpoint sizing is
# reproducible across runs.
PIN_CHUNK_BYTES: int = 64 * 1024 * 1024


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


def _require_pinned_aligned_buffer(
    tensor: torch.Tensor,
    *,
    where: str,
    total_padded: int,
) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{where}: tensor must be on CPU")
    if not tensor.is_contiguous():
        raise ValueError(f"{where}: tensor must be contiguous")
    if not tensor.is_pinned():
        raise ValueError(f"{where}: tensor must be pinned")
    if (int(tensor.data_ptr()) & (SECTOR_BYTES - 1)) != 0:
        raise ValueError(f"{where}: base pointer is not 4 KiB aligned")
    if (int(total_padded) & (SECTOR_BYTES - 1)) != 0:
        raise ValueError(f"{where}: length is not 4 KiB aligned")
    if int(total_padded) > int(tensor.numel() * tensor.element_size()):
        raise ValueError(f"{where}: requested write exceeds pinned tensor")


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


def scatter_lossy_records_to_dense_by_intervals(
    aos_u8: torch.Tensor,
    record_byte_offset: int,
    n_records: int,
    starts_i64: torch.Tensor,
    ends_i64: torch.Tensor,
    p_tensors: List[torch.Tensor],
    m_tensors: List[torch.Tensor],
    v_tensors: List[torch.Tensor],
) -> None:
    """Scatter one expert's pinned AOS records into pinned dense p/m/v views."""

    _cext.scatter_lossy_records_to_dense_by_intervals(
        aos_u8,
        int(record_byte_offset),
        int(n_records),
        starts_i64,
        ends_i64,
        p_tensors,
        m_tensors,
        v_tensors,
    )


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
    """Magic-byte probe through the same aligned read path as recovery."""
    nbytes = int(_cext.file_size(str(path)))
    if nbytes < len(MAGIC):
        return False
    blob = read_file_aligned(path, min(nbytes, HEADER_BLOCK_BYTES))
    return bytes(blob.narrow(0, 0, len(MAGIC)).numpy().tobytes()) == MAGIC


def load_binary(
    path: Path, device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Read a PRPTENS file. Returns (tensor_dict, scalars).

    Recovery reads the whole file once, then reconstructs tensors from byte
    views. This avoids one pread/open round-trip per tensor entry.
    """
    path_s = str(Path(path))
    nbytes_total = int(_cext.file_size(path_s))
    if nbytes_total < 16:
        raise ValueError(f"load_binary: header too short in {path}")
    blob = read_file_aligned(path_s, nbytes_total)
    got = int(blob.numel())
    if got < HEADER_BLOCK_BYTES:
        raise ValueError(f"load_binary: header too short in {path}")
    if got < nbytes_total:
        blob = blob.narrow(0, 0, got)
    header_block = bytes(
        blob.narrow(0, 0, min(int(blob.numel()), HEADER_BLOCK_BYTES))
        .numpy()
        .tobytes()
    )
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
        data_off = HEADER_BLOCK_BYTES + off
        if data_off < 0 or data_off + nbytes > int(blob.numel()):
            raise ValueError(f"load_binary: truncated entry {e['name']!r} in {path}")
        dtype = _STR_TO_TORCH_DTYPE.get(dtype_str)
        if dtype is None:
            raise ValueError(f"load_binary: no torch dtype mapping for {dtype_str!r}")
        raw_u8 = blob.narrow(0, data_off, nbytes)
        t = raw_u8.view(dtype).clone().view(shape)
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
    _require_pinned_aligned_buffer(
        pinned_tensor,
        where="write_prptens_from_pinned",
        total_padded=int(total_padded),
    )
    return int(_cext.write_prptens(
        str(Path(final_path)), int(pinned_tensor.data_ptr()),
        header_bytes, int(total_padded), int(mode), bool(fsync),
    ))


def write_prptens_from_pinned_keep_open(
    final_path: Path,
    pinned_tensor: torch.Tensor,
    header_bytes: bytes,
    payload_bytes: int,
    *,
    step: int,
    mode: int = 0o644,
) -> Tuple[int, int]:
    """Write a PRPTENS file and return ``(fd, padded_bytes)`` for fsync.

    The full write completes before this returns, so the caller may safely
    reuse the pinned source buffer after handing the fd to an async fsync
    worker.
    """

    del step
    total_padded = _round_up_sector(HEADER_BLOCK_BYTES + int(payload_bytes))
    _require_pinned_aligned_buffer(
        pinned_tensor,
        where="write_prptens_from_pinned_keep_open",
        total_padded=int(total_padded),
    )
    fd = int(_cext.write_prptens_keep_open(
        str(Path(final_path)), int(pinned_tensor.data_ptr()),
        header_bytes, int(total_padded), int(mode),
    ))
    return fd, int(total_padded)


def write_header_payload_from_pinned_keep_open(
    final_path: Path,
    pinned_tensor: torch.Tensor,
    header_bytes: bytes,
    payload_bytes: int,
    *,
    mode: int = 0o644,
) -> Tuple[int, int]:
    """Write a header+payload file from one pinned buffer, leaving fd open.

    This is the LFCL/common variant of ``write_prptens_from_pinned_keep_open``:
    the caller keeps tmp/rename/fsync policy, while the C++ layer owns the
    aligned full-write syscall. Only ``HEADER_BLOCK_BYTES + payload_bytes`` is
    written, rounded to a 4 KiB sector boundary.
    """

    total_bytes = HEADER_BLOCK_BYTES + int(payload_bytes)
    total_padded = _round_up_sector(total_bytes)
    _require_pinned_aligned_buffer(
        pinned_tensor,
        where="write_header_payload_from_pinned_keep_open",
        total_padded=int(total_padded),
    )
    if total_padded > total_bytes:
        pinned_tensor.view(torch.uint8).narrow(
            0, int(total_bytes), int(total_padded - total_bytes)
        ).zero_()
    fd = int(_cext.write_header_payload_keep_open(
        str(Path(final_path)), int(pinned_tensor.data_ptr()),
        header_bytes, int(total_padded), int(mode),
    ))
    return fd, int(total_padded)


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
    the C++ LFSY append helper. ``pinned_tensor`` must be 4 KiB-aligned
    (base + length); ``total_bytes`` is rounded up to the next
    SECTOR_BYTES multiple — the pad tail in pinned_tensor is already
    zero-filled."""
    del layer, expert, step
    if pinned_tensor.dtype != torch.uint8 or not pinned_tensor.is_contiguous():
        raise ValueError("append_lossy_log_from_pinned: tensor must be contiguous uint8")
    total_padded = _round_up_sector(total_bytes)
    _require_pinned_aligned_buffer(
        pinned_tensor,
        where="append_lossy_log_from_pinned",
        total_padded=int(total_padded),
    )
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


def read_file_into_tensor(path, dst_u8: torch.Tensor, offset: int, nbytes: int) -> int:
    """Read bytes directly into a CPU uint8 tensor via one C++ entry point."""
    if dst_u8.device.type != "cpu" or dst_u8.dtype != torch.uint8:
        raise ValueError("read_file_into_tensor: dst_u8 must be a CPU uint8 tensor")
    if not dst_u8.is_contiguous():
        raise ValueError("read_file_into_tensor: dst_u8 must be contiguous")
    if int(offset) & (SECTOR_BYTES - 1):
        raise ValueError("read_file_into_tensor: offset must be 4 KiB aligned")
    _require_pinned_aligned_buffer(
        dst_u8,
        where="read_file_into_tensor",
        total_padded=int(nbytes),
    )
    return int(_cext.read_file_into_tensor(
        str(path), dst_u8, int(offset), int(nbytes),
    ))


def read_file_aligned(path, nbytes: int) -> torch.Tensor:
    """Read one CP object through the aligned C++ read path.

    The pread request length is rounded to 4 KiB. The returned tensor is
    narrowed to the actual bytes read, so old truncated files still report
    their real length after satisfying the read alignment contract.
    """

    logical = int(nbytes)
    if logical < 0:
        raise ValueError("read_file_aligned: negative nbytes")
    padded = _round_up_sector(logical)
    buf = alloc_pinned_uint8_sector_aligned(padded)
    got = read_file_into_tensor(path, buf, 0, padded)
    return buf.narrow(0, 0, int(got))


def file_size(path) -> int:
    """``::stat(path).st_size`` or -1 if the file is missing."""
    return int(_cext.file_size(str(path)))


def load_lfcl_payload_interleaved(
    lfcl_u8: torch.Tensor,
    *,
    payload_offset_bytes: int,
    numels_i64: torch.Tensor,
    p_flat: torch.Tensor,
    m_flat: torch.Tensor,
    v_flat: torch.Tensor,
) -> None:
    _cext.load_lfcl_payload_interleaved(
        lfcl_u8,
        int(payload_offset_bytes),
        numels_i64,
        p_flat,
        m_flat,
        v_flat,
    )


def replay_lfsy_log_into_dense(
    lfsy_u8: torch.Tensor,
    *,
    total_numel: int,
    p_flat: torch.Tensor,
    m_flat: torch.Tensor,
    v_flat: torch.Tensor,
) -> Dict[str, int]:
    result = _cext.replay_lfsy_log_into_dense(
        lfsy_u8,
        int(total_numel),
        p_flat,
        m_flat,
        v_flat,
        int(HEADER_BLOCK_BYTES),
        int(SECTOR_BYTES),
        8,
        16,
    )
    keys = (
        "max_step",
        "n_blocks",
        "blocks_replayed",
        "records_seen",
        "records_applied",
        "remaining",
    )
    return {k: int(v) for k, v in zip(keys, result)}
