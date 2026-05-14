"""Raw binary serialization for proposal CP payloads (non-expert FC, H_ema)
and for ExpertState's per-expert lossless / lossy FC files.

This is the single home for **all** direct file ``write()`` calls in the
proposal package. The 4-stream architecture pins each kind of work to its
own owner:

  main_stream    — training (forward / backward / optimizer)
  tracker_stream — proposal expert score / index_select / dg reset
  d2h_stream     — every GPU → pinned-host copy
  writer thread  — pinned host → DRAM buf or disk file (this module)

Keeping ``open / write / writev / fsync`` confined to this module makes the
contract observable: searching for ``os.write`` etc. across ``proposal/``
should hit only this file (plus reads in expert_state.py). Each helper
below is intended to run on a CPU writer thread; they take ``bytes`` /
``memoryview`` / numpy arrays (no torch tensors, no CUDA dependency) so
they cannot accidentally touch a GPU stream.

Replaces ``torch.save`` / ``torch.load`` for flat ``{str: Tensor}`` payloads.
Eliminates Python pickle decode (~300-500 MB/s CPU-bound) — the bytes on
disk are literally the tensors' raw byte storage, preceded by a small JSON
metadata header.

File layout
-----------
::

    offset  size    field
    0       8       MAGIC           b"PRPTENS\\x00" (distinguishes from torch.save pickle)
    8       4       VERSION         uint32 LE (currently 1)
    12      4       HEADER_LEN      uint32 LE — length of the following JSON
    16      H       HEADER_JSON     UTF-8 JSON, structure below
    16+H    ...     DATA            concatenation of each entry's raw bytes

JSON header structure::

    {
      "scalars": {"iter": int, "updates": int, "alpha": float, ...},  # optional
      "entries": [
        {"name": str, "dtype": str, "shape": [int, ...],
         "offset": int, "nbytes": int},
        ...
      ]
    }

``scalars`` carries non-tensor metadata (e.g., H_ema's iter/updates/alpha).
For non-expert FC payloads (pure {name: tensor}) ``scalars`` is absent or empty.

Rationale
---------
Our CP payloads are already assembled as contiguous tensor views into a
single pinned CPU buffer (``_collect_pinned_cpu``, ``_h_ema_pinned_cpu``).
torch.save pickles each view — useful for general Python objects, wasteful
for our flat-buffer case. Writing the pinned buffer verbatim with a small
header is faster on both ends and matches the raw binary format already used
for ExpertState (LFCL/LFSY).

Why not SafeTensors / numpy.save?
---------------------------------
- SafeTensors: nice format, but adds a dependency and restricts tensor
  metadata to its own schema. We don't need its cross-framework guarantees.
- ``numpy.save``: writes a single ndarray, not a dict. Would need custom
  envelope anyway. bf16 unsupported natively.

This module is intentionally minimal (~100 LoC) and self-contained.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


MAGIC: bytes = b"PRPTENS\x00"
# On-disk layout (single version):
#   [0      .. 8)   MAGIC
#   [8      .. 12)  VERSION (u32 LE)
#   [12     .. 16)  HEADER_LEN (u32 LE, JSON bytes length)
#   [16     .. 16+HEADER_LEN)   JSON header (entries + scalars)
#   [16+HEADER_LEN .. HEADER_BLOCK_BYTES)  zero padding
#   [HEADER_BLOCK_BYTES .. EOF) concatenated tensor data
# All header lives inside the leading HEADER_BLOCK_BYTES sector so the
# pinned-inline writer can flush header + payload in ONE os.write.
VERSION: int = 2
_FIXED_HEADER_BYTES = len(MAGIC) + 4 + 4  # magic + version + header_len = 16

# Header block at the front of each pinned file buffer — must be a
# multiple of the O_DIRECT sector size (typically 4 KiB). Sized large
# enough to fit the LARGEST file's JSON metadata (the non-expert FC
# header packs ~390 entries — name + dtype + shape + offset — at
# ~150 B each, so ~60 KB). 64 KiB rounds that comfortably while
# staying trivial overhead vs the multi-GB payloads.
# Sized to also fit the LFCL/LFSY pickled metadata (tiny: ~few KB).
HEADER_BLOCK_BYTES: int = 65536

# Chunk granularity for SSD passthrough (io_uring + O_DIRECT). The C++
# writer manages LBA in N × CHUNK chunks; we pad each per-file pinned
# allocation up to the next chunk multiple so the C++ side can do a
# single chunk-aligned submit without needing scratch buffers.
# File size on disk stays equal to the actual payload (we write only
# ``actual_size`` bytes for now); the padding lives in the pinned alloc
# only. Override via ``PROPOSAL_PIN_CHUNK_MB`` env (default 64 MiB).
_DEFAULT_PIN_CHUNK_MB = 64
try:
    PIN_CHUNK_BYTES: int = max(
        1, int(os.environ.get("PROPOSAL_PIN_CHUNK_MB", _DEFAULT_PIN_CHUNK_MB))
    ) * 1024 * 1024
except ValueError:
    PIN_CHUNK_BYTES = _DEFAULT_PIN_CHUNK_MB * 1024 * 1024


# Sector size used for O_DIRECT alignment (block device sector = 4 KiB
# on virtually every modern Linux + NVMe stack).
SECTOR_BYTES: int = 4096


def round_up_to_chunk(nbytes: int, chunk: int = PIN_CHUNK_BYTES) -> int:
    """Round ``nbytes`` up to the next multiple of ``chunk``. Used to pad
    pinned-buffer allocations to the C++ writer's LBA chunk granularity."""
    if chunk <= 0:
        return int(nbytes)
    n = int(nbytes)
    return ((n + chunk - 1) // chunk) * chunk


def alloc_pinned_uint8_sector_aligned(nbytes: int) -> "torch.Tensor":
    """Allocate a pinned uint8 buffer whose base pointer is 4 KiB-aligned.

    Torch's caching pinned allocator does NOT guarantee page-aligned
    pointers for sub-page allocations (tested: 1608 B → ptr % 4096 ==
    1024). Empirically, allocations whose size is ≥ 4 KiB AND a
    multiple of 4 KiB are always page-aligned by the allocator's
    block-size class machinery.

    We round ``nbytes`` up to the next 4 KiB boundary so the returned
    tensor's ``data_ptr()`` lands on a sector boundary — making the
    buffer usable as-is by an O_DIRECT writer. The trailing pad bytes
    are zero-initialised so the file on disk is deterministic if the
    full padded buffer is written.
    """
    n = int(nbytes)
    padded = ((n + SECTOR_BYTES - 1) // SECTOR_BYTES) * SECTOR_BYTES
    t = torch.empty(padded, dtype=torch.uint8, pin_memory=True)
    if padded > n:
        # Zero-fill the pad so leftover bytes (if the C++ writer flushes
        # the whole padded buffer) are deterministic.
        t[n:].zero_()
    return t

# dtype round-trips between torch / numpy via a compact string tag.
_TORCH_DTYPE_STR = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.int8: "int8",
}
_STR_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_DTYPE_STR.items()}
# numpy dtype mapping for the subset it supports natively. bfloat16 is
# bit-cast via uint16 (numpy has no bf16).
_STR_TO_NUMPY_DTYPE = {
    "float32": np.float32,
    "float16": np.float16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "int8": np.int8,
}


def _as_numpy(t: torch.Tensor):
    """Return a CPU-contiguous numpy view of ``t`` for zero-copy write.
    bfloat16 is bit-cast to uint16 (numpy lacks bf16); reconstruction uses
    the same bit-cast in reverse."""
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy()
    return t.numpy()


def build_prptens_header_block(
    plan: Iterable[Tuple[str, int, int, "torch.Size"]],
    scalars: Optional[Dict[str, Any]] = None,
    *,
    dtype: str = "float32",
) -> bytes:
    """Build a PRPTENS header *block* that fits inside
    ``HEADER_BLOCK_BYTES`` (= 64 KiB). ``plan`` is a list of
    ``(key, fp32_offset, fp32_numel, shape)`` tuples as produced by
    ``FCManager._collect_param_payload`` — i.e. the actual byte order
    of the payload inside the pinned buffer.

    Non-expert FC's JSON header (~60 KB at ~390 entries × ~150 B) is
    what sets the block size — LFCL/LFSY headers are much smaller
    (~few hundred bytes) but share the same sector constant for a
    single on-disk convention.

    Element byte offsets are recorded as ``offset_bytes = fp32_offset * 4``
    so the on-disk header matches what's at ``pinned[HEADER_BLOCK:]``.

    Caller copies the returned bytes into ``pinned[:HEADER_BLOCK_BYTES]``
    (via ``_stamp_header_into_pinned``) just before emitting the single
    ``os.write``.
    """
    entries = []
    for (key, fp32_off, fp32_n, shape) in plan:
        n = int(fp32_n)
        entries.append({
            "name": str(key),
            "dtype": dtype,
            "shape": list(int(d) for d in shape),
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
            f"({HEADER_BLOCK_BYTES}); reduce tensor count or simplify names."
        )
    return prefix


def is_binary_format(path: Path) -> bool:
    """Fast magic-byte probe without full read. Safe on missing / truncated
    files — returns False rather than raising."""
    try:
        with open(path, "rb") as f:
            return f.read(len(MAGIC)) == MAGIC
    except (OSError, FileNotFoundError):
        return False


def load_binary(
    path: Path, device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load a PRPTENS binary file. Returns ``(tensor_dict, scalars)``.

    Read path: ``np.fromfile(f, dtype, count)`` reads tensor bytes directly
    from the open file descriptor into a fresh writable numpy buffer. This
    avoids the ``f.read() → frombuffer(ro) → .copy()`` pattern that would
    temporarily hold ~2× the payload in memory (once in ``data`` bytes,
    once in the copied array). With fromfile, only the final array lives.
    """
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(
                f"load_binary: bad magic {magic!r} in {path}. "
                f"Expected {MAGIC!r}; is this really a PRPTENS file?"
            )
        version, header_len = struct.unpack("<II", f.read(8))
        if version != VERSION:
            raise ValueError(
                f"load_binary: unsupported version {version} (expected "
                f"{VERSION}) in {path}"
            )
        header = json.loads(f.read(header_len).decode("utf-8"))

        entries = header["entries"]
        scalars: Dict[str, Any] = dict(header.get("scalars", {}))

        # Data region starts at the fixed HEADER_BLOCK_BYTES offset
        # (header is zero-padded inside the first sector).
        data_start = HEADER_BLOCK_BYTES
        out: Dict[str, torch.Tensor] = {}
        for e in entries:
            off = int(e["offset"])
            shape = tuple(int(x) for x in e["shape"])
            dtype_str = e["dtype"]
            numel = 1
            for d in shape:
                numel *= d
            f.seek(data_start + off)
            if dtype_str == "bfloat16":
                arr = np.fromfile(f, dtype=np.uint16, count=numel).reshape(shape)
                t = torch.from_numpy(arr).view(torch.bfloat16)
            else:
                np_dtype = _STR_TO_NUMPY_DTYPE.get(dtype_str)
                if np_dtype is None:
                    raise ValueError(
                        f"load_binary: no numpy dtype mapping for {dtype_str!r} "
                        f"(entry {e['name']!r})"
                    )
                arr = np.fromfile(f, dtype=np_dtype, count=numel).reshape(shape)
                t = torch.from_numpy(arr)
            if device is not None:
                t = t.to(device)
            out[e["name"]] = t
    return out, scalars


# ---------------------------------------------------------------------------
# Pinned-inline single-write helpers (SSD-passthrough prep).
#
# All three file types (PRPTENS / LFCL / LFSY) end up laid out as a single
# contiguous byte sequence inside a 4 KiB-aligned pinned buffer:
#
#     [header (HEADER_BLOCK_BYTES, zero-padded)] [payload bytes] [chunk-pad]
#
# The helpers below take that pinned buffer + (header_bytes, payload_size)
# and emit ONE ``os.write`` covering header + payload (the chunk-pad lives
# in the pinned alloc only — the on-disk file stays at actual size).
#
# Wiring a future C++ writer (io_uring + O_DIRECT) only needs to replace
# the ``os.write(fd, mv)`` call with the C++ queue submit — the buffer
# layout is already what the C++ side expects.
# ---------------------------------------------------------------------------


def _stamp_header_into_pinned(
    pinned_mv: memoryview, header_bytes: bytes,
) -> None:
    """Copy ``header_bytes`` into ``pinned_mv[0:HEADER_BLOCK_BYTES]`` and
    zero-fill the remainder of the block. ``pinned_mv`` must be a uint8
    memoryview backing the pinned host buffer."""
    n = len(header_bytes)
    if n > HEADER_BLOCK_BYTES:
        raise ValueError(
            f"header ({n} B) exceeds HEADER_BLOCK_BYTES ({HEADER_BLOCK_BYTES})"
        )
    pinned_mv[0:n] = header_bytes
    if n < HEADER_BLOCK_BYTES:
        # Zero-fill the tail of the header block (cudaHostAlloc memory is
        # uninitialised, so this is required for deterministic on-disk
        # bytes between header end and payload start).
        zero_tail = bytes(HEADER_BLOCK_BYTES - n)
        pinned_mv[n:HEADER_BLOCK_BYTES] = zero_tail


def write_prptens_from_pinned(
    final_path: Path,
    pinned_mv: memoryview,
    header_bytes: bytes,
    payload_bytes: int,
    *,
    step: int,
    mode: int = 0o644,
    fsync: bool = False,
) -> int:
    """Write a PRPTENS file in ONE ``os.write`` call from a pinned buffer
    whose first ``HEADER_BLOCK_BYTES`` are reserved for the header and
    whose next ``payload_bytes`` bytes hold the concatenated tensor data.

    ``header_bytes`` (≤ HEADER_BLOCK_BYTES) is stamped in-place at the
    start of ``pinned_mv``; the on-disk file is exactly
    ``HEADER_BLOCK_BYTES + payload_bytes`` bytes.

    ``step`` is included so the SSD-passthrough writer can route /
    log / index this file by iteration. The Python ``os.write`` path
    does not use it directly; it is part of the C++ swap-in contract
    (see CPP_HANDOFF.md).
    """
    del step  # routing metadata only — consumed when swapped to C++ submit
    final_path = Path(final_path)
    total = HEADER_BLOCK_BYTES + int(payload_bytes)
    _stamp_header_into_pinned(pinned_mv, header_bytes)
    fd = os.open(str(final_path), os.O_CREAT | os.O_RDWR | os.O_TRUNC, mode)
    try:
        os.write(fd, pinned_mv[:total])
        if fsync:
            os.fsync(fd)
    finally:
        os.close(fd)
    return total


def write_lossless_from_pinned(
    final_path: Path,
    pinned_mv: memoryview,
    header_bytes: bytes,
    payload_bytes: int,
    *,
    layer: int,
    expert: int,
    step: int,
    mode: int = 0o644,
) -> int:
    """Atomic LFCL writer that emits header + payload in a single
    ``os.write`` from a pinned buffer. The 4 KiB header block lives at
    ``pinned_mv[:HEADER_BLOCK_BYTES]``; tensor payload follows
    immediately. Writes to ``<final_path>.tmp``, ``fsync``s, then
    ``os.replace``s into place. Returns total bytes written.

    ``layer`` / ``expert`` / ``step`` route this expert's snapshot for
    the SSD-passthrough writer (logging, LBA bookkeeping). The
    Python ``os.write`` path does not consume them; see CPP_HANDOFF.md.
    """
    del layer, expert, step  # routing metadata — consumed by C++ submit
    final_path = Path(final_path)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    total = HEADER_BLOCK_BYTES + int(payload_bytes)
    _stamp_header_into_pinned(pinned_mv, header_bytes)
    fd = os.open(str(tmp_path), os.O_CREAT | os.O_RDWR | os.O_TRUNC, mode)
    try:
        os.write(fd, pinned_mv[:total])
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp_path), str(final_path))
    return total


def append_lossy_log_from_pinned(
    fd: int, pinned_mv: memoryview, total_bytes: int,
    *,
    layer: int,
    expert: int,
    step: int,
) -> int:
    """Append one ``(count, step, records)`` block to the lossy log fd
    via a single ``os.write`` from a pinned buffer that already contains
    the (count u32, step u32, records...) packed bytes in order.

    ``total_bytes`` = BLOCK_PREFIX_BYTES + count × RECORD_BYTES.

    ``layer`` / ``expert`` / ``step`` route this append for the
    SSD-passthrough writer (per-expert chunk LBA bookkeeping). The
    step value is also packed inside ``pinned_mv[4:8]`` per the file
    format; it is passed explicitly here for symmetry with the other
    helpers and to let the C++ submit path avoid re-parsing the blob.
    """
    del layer, expert, step  # routing metadata — consumed by C++ submit
    return int(os.write(fd, pinned_mv[:int(total_bytes)]))


# ---------------------------------------------------------------------------
# ExpertState LFSY (lossy log) file open helper.
#
# Lossy log layout (one file per expert):
#   header:  ``_pack_header(MAGIC_LFSY, meta)``  (caller supplies bytes)
#   appends: [ <count: u32> <step: u32> <records: count × RECORD_BYTES> ]*
# Per-append writes go through ``append_lossy_log_from_pinned`` above —
# the writer-thread assembles the entire ``[count, step, records]`` blob
# inside a pinned buffer and emits one ``os.write``.
# ---------------------------------------------------------------------------


def open_lossy_log_for_append(
    path: Path,
    header_bytes: bytes,
    *,
    mode: int = 0o644,
) -> int:
    """Create+truncate the lossy file at ``path``, write the header, and
    return the open fd. Caller (writer thread) holds the fd until close.
    The ``O_APPEND`` flag guarantees that every subsequent ``os.write``
    is atomic at the kernel level even under signal interruption."""
    flags = os.O_CREAT | os.O_RDWR | os.O_TRUNC | os.O_APPEND
    fd = os.open(str(path), flags, mode)
    try:
        os.write(fd, header_bytes)
    except Exception:
        os.close(fd)
        raise
    return fd


def fsync_fd(fd: int) -> None:
    """``os.fsync`` wrapper that swallows ``OSError`` (e.g. EBADF on
    already-closed fd) so callers do not need a try/except at every site."""
    try:
        os.fsync(fd)
    except OSError:
        pass


def close_fd(fd: int) -> None:
    """``os.close`` wrapper that swallows ``OSError`` for idempotent
    teardown paths."""
    try:
        os.close(fd)
    except OSError:
        pass
