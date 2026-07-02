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
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
from torch.utils.cpp_extension import load

from ..telemetry.fdp_readback_validation import FdpReadbackValidator


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
_fdp_cext = None
_FDP_COMPLETED_FD = -(1 << 30)


def backend_kind() -> str:
    """Return the active proposal I/O backend name."""
    kind = os.environ.get("PROPOSAL_IO_BACKEND", "file").strip().lower()
    if kind not in ("file", "fdp"):
        raise ValueError(
            "PROPOSAL_IO_BACKEND must be 'file' or 'fdp', "
            f"got {kind!r}"
        )
    return kind


def _fdp_global_expert_id_for_read(expert: int) -> int:
    if os.environ.get("PROPOSAL_FDP_EXPERT_IDS_GLOBAL", "").lower() in (
        "1", "true", "yes",
    ):
        return int(expert)
    ep_size = int(os.environ.get("PROPOSAL_FDP_EP_SIZE", "1") or "1")
    if ep_size <= 1:
        return int(expert)
    base = int(os.environ.get("PROPOSAL_FDP_EXPERT_ID_BASE", "0") or "0")
    return int(base) + int(expert)


def _get_fdp():
    """Lazy-load and open the raw NVMe FDP backend session."""
    global _fdp_cext
    if _fdp_cext is None:
        _fdp_cext = load(
            name="proposal_fdp_cext",
            sources=[
                str(_src_dir / "fdp_store.cpp"),
                str(_src_dir / "fdp_passthru.cpp"),
                str(_src_dir / "proposal_fdp_io.cpp"),
            ],
            extra_cflags=["-O3", "-std=c++17", "-fopenmp", "-pthread"],
            extra_ldflags=["-lgomp", "-luring", "-pthread"],
            extra_include_paths=[str(_src_dir)],
            verbose=False,
        )
        _fdp_cext.open_session_from_env()
    return _fdp_cext


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


def fdp_readback_validation_enabled() -> bool:
    return backend_kind() == "fdp" and _env_bool("PROPOSAL_FDP_VALIDATE_READBACK")


def fdp_metadata_mmap_enabled() -> bool:
    return backend_kind() == "fdp" and _env_bool("PROPOSAL_FDP_METADATA_MMAP")


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


def _require_cpu_aligned_buffer(
    tensor: torch.Tensor,
    *,
    where: str,
    total_padded: int,
) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{where}: tensor must be on CPU")
    if not tensor.is_contiguous():
        raise ValueError(f"{where}: tensor must be contiguous")
    if (int(tensor.data_ptr()) & (SECTOR_BYTES - 1)) != 0:
        raise ValueError(f"{where}: base pointer is not 4 KiB aligned")
    if (int(total_padded) & (SECTOR_BYTES - 1)) != 0:
        raise ValueError(f"{where}: length is not 4 KiB aligned")
    if int(total_padded) > int(tensor.numel() * tensor.element_size()):
        raise ValueError(f"{where}: requested write exceeds tensor")


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


def alloc_cpu_uint8_sector_aligned(
    nbytes: int,
    *,
    pin_memory: bool = False,
) -> torch.Tensor:
    """CPU uint8 buffer whose base ptr and length are both 4 KiB-aligned.

    FDP recovery reads directly into this buffer from raw NVMe. It does not
    require CUDA pinned memory, which keeps CPU-only smoke/recovery paths
    usable on systems without a GPU.
    """
    n = int(nbytes)
    padded = _round_up_sector(n)
    try:
        over = torch.empty(
            padded + SECTOR_BYTES - 1,
            dtype=torch.uint8,
            pin_memory=bool(pin_memory),
        )
    except RuntimeError:
        if pin_memory:
            over = torch.empty(padded + SECTOR_BYTES - 1, dtype=torch.uint8)
        else:
            raise
    base = int(over.data_ptr())
    misalign = base & (SECTOR_BYTES - 1)
    skip = 0 if misalign == 0 else (SECTOR_BYTES - misalign)
    t = over.narrow(0, skip, padded)
    assert (int(t.data_ptr()) & (SECTOR_BYTES - 1)) == 0, "aligned slice failed"
    if padded > n:
        t[n:].zero_()
    return t


_fdp_readback_validator = FdpReadbackValidator(
    header_block_bytes=HEADER_BLOCK_BYTES,
    sector_bytes=SECTOR_BYTES,
    enabled=fdp_readback_validation_enabled,
    get_fdp=_get_fdp,
    alloc_cpu_u8=alloc_cpu_uint8_sector_aligned,
)


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


def load_binary_blob(
    blob: torch.Tensor,
    device: Optional[torch.device] = None,
    *,
    source: str = "<blob>",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Parse one in-memory PRPTENS object. Returns (tensor_dict, scalars)."""
    if blob.device.type != "cpu" or blob.dtype != torch.uint8:
        raise ValueError(f"load_binary_blob: {source} must be a CPU uint8 tensor")
    if not blob.is_contiguous():
        raise ValueError(f"load_binary_blob: {source} must be contiguous")
    got = int(blob.numel())
    if got < HEADER_BLOCK_BYTES:
        raise ValueError(f"load_binary_blob: header too short in {source}")
    header_block = bytes(
        blob.narrow(0, 0, min(int(blob.numel()), HEADER_BLOCK_BYTES))
        .numpy()
        .tobytes()
    )
    if header_block[:len(MAGIC)] != MAGIC:
        raise ValueError(
            f"load_binary_blob: bad magic in {source}; expected {MAGIC!r}")
    version, header_len = struct.unpack("<II", header_block[8:16])
    if version != VERSION:
        raise ValueError(
            "load_binary_blob: unsupported version "
            f"{version} in {source} (expected {VERSION})"
        )
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
            raise ValueError(
                f"load_binary_blob: truncated entry {e['name']!r} in {source}"
            )
        dtype = _STR_TO_TORCH_DTYPE.get(dtype_str)
        if dtype is None:
            raise ValueError(
                f"load_binary_blob: no torch dtype mapping for {dtype_str!r}"
            )
        raw_u8 = blob.narrow(0, data_off, nbytes)
        t = raw_u8.view(dtype).clone().view(shape)
        if device is not None:
            t = t.to(device)
        out[e["name"]] = t
    return out, scalars


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
    return load_binary_blob(blob, device=device, source=path_s)


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
    total_padded = _round_up_sector(HEADER_BLOCK_BYTES + int(payload_bytes))
    if backend_kind() == "fdp":
        _require_cpu_aligned_buffer(
            pinned_tensor,
            where="write_prptens_from_pinned",
            total_padded=int(total_padded),
        )
        del mode
        fdp = _get_fdp()
        written = int(fdp.write_nefc(
            str(Path(final_path).name),
            int(pinned_tensor.data_ptr()),
            header_bytes,
            int(payload_bytes),
            int(total_padded),
            int(step),
        ))
        _fdp_readback_validator.queue_nefc(
            pinned_tensor=pinned_tensor,
            logical_bytes=int(HEADER_BLOCK_BYTES + int(payload_bytes)),
            context=f"{Path(final_path).name}@iter{int(step)}",
        )
        if bool(fsync) and not fdp_metadata_mmap_enabled():
            fdp.flush_metadata(int(step))
        return written
    _require_pinned_aligned_buffer(
        pinned_tensor,
        where="write_prptens_from_pinned",
        total_padded=int(total_padded),
    )
    del step  # routing metadata only — consumed by file path
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

    total_padded = _round_up_sector(HEADER_BLOCK_BYTES + int(payload_bytes))
    if backend_kind() == "fdp":
        _require_cpu_aligned_buffer(
            pinned_tensor,
            where="write_prptens_from_pinned_keep_open",
            total_padded=int(total_padded),
        )
        written = write_prptens_from_pinned(
            final_path,
            pinned_tensor,
            header_bytes,
            int(payload_bytes),
            step=int(step),
            mode=int(mode),
            fsync=True,
        )
        return _FDP_COMPLETED_FD, int(written)
    _require_pinned_aligned_buffer(
        pinned_tensor,
        where="write_prptens_from_pinned_keep_open",
        total_padded=int(total_padded),
    )
    del step
    fd = int(_cext.write_prptens_keep_open(
        str(Path(final_path)), int(pinned_tensor.data_ptr()),
        header_bytes, int(total_padded), int(mode),
    ))
    return fd, int(total_padded)


def write_lfcs_from_pinned(
    final_path: Path,
    pinned_tensor: torch.Tensor,
    header_bytes: bytes,
    payload_bytes: int,
    *,
    layer: int,
    expert: int,
    step: int,
    generation: int = 0,
    mode: int = 0o644,
) -> int:
    """Write an LFCL/LFCS object through the active backend."""
    total_padded = _round_up_sector(HEADER_BLOCK_BYTES + int(payload_bytes))
    if backend_kind() == "fdp":
        _require_cpu_aligned_buffer(
            pinned_tensor,
            where="write_lfcs_from_pinned",
            total_padded=int(total_padded),
        )
        del mode
        written = int(_get_fdp().write_lfcs(
            str(Path(final_path).name),
            int(pinned_tensor.data_ptr()),
            header_bytes,
            int(payload_bytes),
            int(total_padded),
            int(layer),
            int(expert),
            int(step),
            int(generation),
        ))
        _fdp_readback_validator.queue_lfcs(
            pinned_tensor=pinned_tensor,
            logical_bytes=int(HEADER_BLOCK_BYTES + int(payload_bytes)),
            layer=int(layer),
            expert=int(expert),
            context=(
                f"{Path(final_path).name}@iter{int(step)}"
                f":L{int(layer)}/E{int(expert)}"
            ),
        )
        return written
    _require_pinned_aligned_buffer(
        pinned_tensor,
        where="write_lfcs_from_pinned",
        total_padded=int(total_padded),
    )
    fd, padded = write_header_payload_from_pinned_keep_open(
        final_path,
        pinned_tensor,
        header_bytes,
        int(payload_bytes),
        mode=int(mode),
    )
    try:
        fsync_fd(fd)
    finally:
        close_fd(fd)
    return int(padded)


def write_lossless_from_pinned(
    final_path: Path,
    pinned_tensor: torch.Tensor,
    header_bytes: bytes,
    payload_bytes: int,
    *,
    layer: int,
    expert: int,
    step: int,
    generation: int = 0,
    mode: int = 0o644,
) -> int:
    """Backward-compatible name for LFCL/LFCS writes."""
    return write_lfcs_from_pinned(
        final_path,
        pinned_tensor,
        header_bytes,
        int(payload_bytes),
        layer=int(layer),
        expert=int(expert),
        step=int(step),
        generation=int(generation),
        mode=int(mode),
    )


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
    path: Path,
    header_bytes: bytes,
    *,
    layer: Optional[int] = None,
    expert: Optional[int] = None,
    step: Optional[int] = None,
    generation: Optional[int] = None,
    mode: int = 0o644,
) -> int:
    """Create+truncate the lossy file, write a HEADER_BLOCK_BYTES header
    region (chunk-aligned), return the open fd. Reader treats the first
    HEADER_BLOCK_BYTES of the file as the header block (tail zero-filled
    if the actual header is smaller)."""
    # Allocate a single-shot 4 KiB-aligned pinned buffer for the header,
    # let C++ stamp it + write it, then drop the ref. Steady-state lossy
    # writes use append_lossy_log_from_pinned with already-padded blobs.
    if backend_kind() == "fdp":
        del mode
        if layer is None or expert is None or generation is None:
            raise RuntimeError("FDP LFSY open requires layer/expert/generation")
        fd = int(_get_fdp().open_lfsy(
            str(Path(path).name),
            header_bytes,
            int(layer),
            int(expert),
            int(step if step is not None else -1),
            int(generation),
        ))
        _fdp_readback_validator.queue_lfsy_header(
            header=bytes(header_bytes),
            layer=int(layer),
            expert=int(expert),
            context=(
                f"{Path(path).name}@iter"
                f"{int(step if step is not None else -1)}"
                f":L{int(layer)}/E{int(expert)}:gen{int(generation)}"
            ),
        )
        return fd
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
    generation: Optional[int] = None,
) -> int:
    """Append a (count u32, step u32, records...) blob to ``fd`` via
    the C++ LFSY append helper. ``pinned_tensor`` must be 4 KiB-aligned
    (base + length); ``total_bytes`` is rounded up to the next
    SECTOR_BYTES multiple — the pad tail in pinned_tensor is already
    zero-filled."""
    if pinned_tensor.dtype != torch.uint8 or not pinned_tensor.is_contiguous():
        raise ValueError(
            "append_lossy_log_from_pinned: tensor must be contiguous uint8"
        )
    total_padded = _round_up_sector(total_bytes)
    if backend_kind() == "fdp":
        _require_cpu_aligned_buffer(
            pinned_tensor,
            where="append_lossy_log_from_pinned",
            total_padded=int(total_padded),
        )
        if generation is None:
            raise RuntimeError("FDP LFSY append requires generation")
        written = int(_get_fdp().append_lfsy(
            int(fd),
            int(pinned_tensor.data_ptr()),
            int(total_bytes),
            int(total_padded),
            int(layer),
            int(expert),
            int(step),
            int(generation),
        ))
        _fdp_readback_validator.queue_lfsy_append(
            pinned_tensor=pinned_tensor,
            nbytes=int(total_padded),
            layer=int(layer),
            expert=int(expert),
            context=(
                f"lfsy@iter{int(step)}:L{int(layer)}/E{int(expert)}"
                f":gen{int(generation)}"
            ),
        )
        return written
    _require_pinned_aligned_buffer(
        pinned_tensor,
        where="append_lossy_log_from_pinned",
        total_padded=int(total_padded),
    )
    del layer, expert, step, generation
    return int(_cext.append_lossy_log(
        int(fd), int(pinned_tensor.data_ptr()), int(total_padded),
    ))


def fsync_fd(fd: int) -> None:
    if backend_kind() == "fdp":
        if int(fd) == _FDP_COMPLETED_FD:
            return
        if fdp_metadata_mmap_enabled():
            return
        _get_fdp().fsync_handle(int(fd))
        return
    _cext.fsync_fd(int(fd))


def close_fd(fd: int) -> None:
    if backend_kind() == "fdp":
        if int(fd) == _FDP_COMPLETED_FD:
            return
        _get_fdp().close_lfsy(int(fd))
        return
    _cext.close_fd(int(fd))


def unlink_path(path) -> None:
    """``::unlink(path)``; swallows ENOENT for idempotent callers."""
    if backend_kind() == "fdp":
        return
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


def flush_metadata(iter: int = 0) -> None:
    """Flush the FDP metadata image when the FDP backend is active."""
    if backend_kind() == "fdp":
        _get_fdp().flush_metadata(int(iter))


def fdp_begin_checkpoint_iter(iter: int) -> None:
    if backend_kind() != "fdp":
        return
    if not fdp_metadata_mmap_enabled():
        return
    _get_fdp().begin_checkpoint_iter(int(iter))


def fdp_mark_writer_done(iter: int, writer_kind: str) -> bool:
    if backend_kind() != "fdp":
        return False
    if not fdp_metadata_mmap_enabled():
        return False
    return bool(_get_fdp().mark_writer_done(int(iter), str(writer_kind)))


def fdp_begin_data_batch() -> None:
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_begin_data_batch requires PROPOSAL_IO_BACKEND=fdp")
    _get_fdp().begin_data_batch()


def fdp_end_data_batch() -> Dict[str, Any]:
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_end_data_batch requires PROPOSAL_IO_BACKEND=fdp")
    return dict(_get_fdp().end_data_batch())


def fdp_end_data_batch_stats() -> Dict[str, Any]:
    if backend_kind() != "fdp":
        raise RuntimeError(
            "fdp_end_data_batch_stats requires PROPOSAL_IO_BACKEND=fdp"
        )
    stats = dict(_get_fdp().end_data_batch_stats())
    _fdp_readback_validator.validate_all()
    return stats


def fdp_begin_nefc_batch() -> None:
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_begin_nefc_batch requires PROPOSAL_IO_BACKEND=fdp")
    _get_fdp().begin_nefc_batch()


def fdp_end_nefc_batch() -> int:
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_end_nefc_batch requires PROPOSAL_IO_BACKEND=fdp")
    out = int(_get_fdp().end_nefc_batch())
    _fdp_readback_validator.validate_nefc()
    return out


def fdp_end_nefc_batch_stats() -> Dict[str, Any]:
    if backend_kind() != "fdp":
        raise RuntimeError(
            "fdp_end_nefc_batch_stats requires PROPOSAL_IO_BACKEND=fdp"
        )
    stats = dict(_get_fdp().end_nefc_batch_stats())
    _fdp_readback_validator.validate_nefc()
    return stats


def fdp_begin_expert_batch() -> None:
    if backend_kind() != "fdp":
        raise RuntimeError(
            "fdp_begin_expert_batch requires PROPOSAL_IO_BACKEND=fdp"
        )
    _get_fdp().begin_expert_batch()


def fdp_end_expert_batch() -> int:
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_end_expert_batch requires PROPOSAL_IO_BACKEND=fdp")
    out = int(_get_fdp().end_expert_batch())
    _fdp_readback_validator.validate_expert()
    return out


def fdp_end_expert_batch_stats() -> Dict[str, Any]:
    if backend_kind() != "fdp":
        raise RuntimeError(
            "fdp_end_expert_batch_stats requires PROPOSAL_IO_BACKEND=fdp"
        )
    stats = dict(_get_fdp().end_expert_batch_stats())
    _fdp_readback_validator.validate_expert()
    return stats


def fdp_stats() -> Dict[str, Any]:
    if backend_kind() != "fdp":
        return {"backend": "file"}
    return dict(_get_fdp().stats())


def open_fdp_session() -> None:
    if backend_kind() != "fdp":
        raise RuntimeError("open_fdp_session requires PROPOSAL_IO_BACKEND=fdp")
    _get_fdp()


def fdp_format_or_load() -> None:
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_format_or_load requires PROPOSAL_IO_BACKEND=fdp")
    _get_fdp().format_or_load()


def fdp_debug_read_metadata_summary() -> Dict[str, Any]:
    if backend_kind() != "fdp":
        return {"backend": "file"}
    return dict(_get_fdp().debug_read_metadata_summary())


def fdp_debug_read_lfsy_entries() -> List[Dict[str, Any]]:
    if backend_kind() != "fdp":
        return []
    return [dict(entry) for entry in _get_fdp().debug_read_lfsy_entries()]


def fdp_recovery_entries() -> Dict[str, Any]:
    """Return the latest FDP metadata entries needed by recovery."""
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_recovery_entries requires PROPOSAL_IO_BACKEND=fdp")
    raw = dict(_get_fdp().recovery_entries())
    raw["nefc"] = dict(raw.get("nefc", {}))
    raw["nefc_shards"] = [dict(entry) for entry in raw.get("nefc_shards", [])]
    raw["lfcs"] = [dict(entry) for entry in raw.get("lfcs", [])]
    raw["lfsy"] = [dict(entry) for entry in raw.get("lfsy", [])]
    return raw


def _fdp_read_object_blob(
    logical_bytes: int,
    read_fn,
    *,
    source: str,
) -> torch.Tensor:
    logical = int(logical_bytes)
    if logical <= 0:
        return torch.empty(0, dtype=torch.uint8)
    padded = _round_up_sector(logical)
    buf = alloc_cpu_uint8_sector_aligned(padded)
    _require_cpu_aligned_buffer(
        buf,
        where=source,
        total_padded=int(padded),
    )
    got = int(read_fn(int(buf.data_ptr()), int(padded)))
    if got < 0 or got > padded:
        raise RuntimeError(f"{source}: invalid read byte count {got}")
    return buf.narrow(0, 0, int(got))


def fdp_read_nefc_blob(
    logical_bytes: Optional[int] = None,
    *,
    shard_index: Optional[int] = None,
) -> torch.Tensor:
    """Read the latest NEFC PRPTENS object from FDP chunks."""
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_read_nefc_blob requires PROPOSAL_IO_BACKEND=fdp")
    if logical_bytes is None:
        entries = fdp_recovery_entries()
        if shard_index is None:
            entry = entries["nefc"]
        else:
            entry = next(
                (
                    e for e in entries.get("nefc_shards", [])
                    if int(e.get("shard_index", -1)) == int(shard_index)
                ),
                None,
            )
            if entry is None:
                raise RuntimeError(
                    f"fdp_read_nefc_blob: NEFC shard {int(shard_index)} missing"
                )
        if not bool(entry.get("valid", False)):
            raise RuntimeError("fdp_read_nefc_blob: NEFC entry is not valid")
        logical_bytes = int(entry["logical_bytes"])
    fdp = _get_fdp()
    read_fn = (
        fdp.read_nefc_into
        if shard_index is None
        else lambda addr, nbytes: fdp.read_nefc_shard_into(
            int(shard_index), int(addr), int(nbytes)
        )
    )
    return _fdp_read_object_blob(
        int(logical_bytes),
        read_fn,
        source=(
            "fdp_read_nefc_blob"
            if shard_index is None
            else f"fdp_read_nefc_blob(shard={int(shard_index)})"
        ),
    )


def fdp_read_lfcs_blob(
    layer: int,
    expert: int,
    logical_bytes: Optional[int] = None,
) -> torch.Tensor:
    """Read one LFCS/LFCL baseline object from FDP chunks."""
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_read_lfcs_blob requires PROPOSAL_IO_BACKEND=fdp")
    if logical_bytes is None:
        entries = fdp_recovery_entries()["lfcs"]
        global_expert = _fdp_global_expert_id_for_read(int(expert))
        match = next(
            (
                e for e in entries
                if bool(e.get("valid", False))
                and int(e.get("layer_id", -1)) == int(layer)
                and int(e.get("expert_id", -1)) == int(global_expert)
            ),
            None,
        )
        if match is None:
            raise RuntimeError(
                f"fdp_read_lfcs_blob: LFCS entry missing for L{layer}/E{expert}"
            )
        logical_bytes = int(match["logical_bytes"])
    fdp = _get_fdp()
    return _fdp_read_object_blob(
        int(logical_bytes),
        lambda addr, nbytes: fdp.read_lfcs_into(
            int(layer), int(expert), int(addr), int(nbytes)
        ),
        source=f"fdp_read_lfcs_blob(L{int(layer)}/E{int(expert)})",
    )


def fdp_read_lfsy_blob(
    layer: int,
    expert: int,
    logical_bytes: Optional[int] = None,
) -> torch.Tensor:
    """Read one LFSY append log object from FDP chunks."""
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_read_lfsy_blob requires PROPOSAL_IO_BACKEND=fdp")
    if logical_bytes is None:
        entries = fdp_recovery_entries()["lfsy"]
        global_expert = _fdp_global_expert_id_for_read(int(expert))
        match = next(
            (
                e for e in entries
                if bool(e.get("valid", False))
                and int(e.get("layer_id", -1)) == int(layer)
                and int(e.get("expert_id", -1)) == int(global_expert)
            ),
            None,
        )
        if match is None:
            raise RuntimeError(
                f"fdp_read_lfsy_blob: LFSY entry missing for L{layer}/E{expert}"
            )
        logical_bytes = int(match["logical_tail"])
    fdp = _get_fdp()
    return _fdp_read_object_blob(
        int(logical_bytes),
        lambda addr, nbytes: fdp.read_lfsy_into(
            int(layer), int(expert), int(addr), int(nbytes)
        ),
        source=f"fdp_read_lfsy_blob(L{int(layer)}/E{int(expert)})",
    )


def fdp_debug_raw_write_read_smoke() -> bool:
    if backend_kind() != "fdp":
        return False
    return bool(_get_fdp().debug_raw_write_read_smoke())


def fdp_debug_read_nefc(offset: int, nbytes: int) -> bytes:
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_debug_read_nefc requires PROPOSAL_IO_BACKEND=fdp")
    return bytes(_get_fdp().debug_read_nefc(int(offset), int(nbytes)))


def fdp_debug_read_lfcs(layer: int, expert: int,
                        offset: int, nbytes: int) -> bytes:
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_debug_read_lfcs requires PROPOSAL_IO_BACKEND=fdp")
    return bytes(_get_fdp().debug_read_lfcs(
        int(layer), int(expert), int(offset), int(nbytes)
    ))


def fdp_debug_read_lfsy(layer: int, expert: int,
                        offset: int, nbytes: int) -> bytes:
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_debug_read_lfsy requires PROPOSAL_IO_BACKEND=fdp")
    return bytes(_get_fdp().debug_read_lfsy(
        int(layer), int(expert), int(offset), int(nbytes)
    ))


def fdp_debug_alloc_chunk() -> int:
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_debug_alloc_chunk requires PROPOSAL_IO_BACKEND=fdp")
    return int(_get_fdp().debug_alloc_chunk())


def fdp_debug_free_chunk(chunk_id: int) -> None:
    if backend_kind() != "fdp":
        raise RuntimeError("fdp_debug_free_chunk requires PROPOSAL_IO_BACKEND=fdp")
    _get_fdp().debug_free_chunk(int(chunk_id))


def close_fdp_session() -> None:
    global _fdp_cext
    if _fdp_cext is not None:
        _fdp_cext.close_session()
        _fdp_cext = None
