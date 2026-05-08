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
VERSION: int = 1
_FIXED_HEADER_BYTES = len(MAGIC) + 4 + 4  # magic + version + header_len = 16

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


def save_binary(
    path: Path,
    tensor_dict: Dict[str, torch.Tensor],
    scalars: Optional[Dict[str, Any]] = None,
) -> int:
    """Write a flat ``{name: tensor}`` payload + optional scalar metadata to
    ``path`` in the PRPTENS binary format. Returns the total bytes written.

    Calling convention mirrors ``torch.save({**scalars, **tensor_dict}, path)``
    semantically but bypasses pickle. Tensors are concatenated in insertion
    order for maximum write locality.

    Write path: ``ndarray.tofile(f)`` does a C-level zero-copy dump from the
    buffer to the file descriptor. Avoids the ~1 GB userspace memcpy that
    ``.tobytes()`` would incur per non-expert FC.
    """
    path = Path(path)
    entries = []
    offset = 0
    arrays: list = []
    for name, t in tensor_dict.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"save_binary: expected torch.Tensor for key {name!r}, "
                f"got {type(t).__name__}"
            )
        dtype_str = _TORCH_DTYPE_STR.get(t.dtype)
        if dtype_str is None:
            raise ValueError(
                f"save_binary: unsupported dtype {t.dtype} for key {name!r}. "
                f"Add to _TORCH_DTYPE_STR if needed."
            )
        arr = _as_numpy(t)
        nbytes = int(arr.nbytes)
        entries.append({
            "name": name,
            "dtype": dtype_str,
            "shape": list(t.shape),
            "offset": offset,
            "nbytes": nbytes,
        })
        arrays.append(arr)
        offset += nbytes

    header: Dict[str, Any] = {"entries": entries}
    if scalars:
        header["scalars"] = dict(scalars)
    header_json = json.dumps(header).encode("utf-8")

    total = _FIXED_HEADER_BYTES + len(header_json) + offset
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(header_json)))
        f.write(header_json)
        for arr in arrays:
            arr.tofile(f)
    return total


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
                f"load_binary: unsupported version {version} (expected {VERSION}) "
                f"in {path}"
            )
        header = json.loads(f.read(header_len).decode("utf-8"))

        entries = header["entries"]
        scalars: Dict[str, Any] = dict(header.get("scalars", {}))

        data_start = f.tell()
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
# ExpertState LFCL (lossless full-checkpoint) writer.
#
# The LFCL format is independent of the PRPTENS dict format above — it's a
# header + per-tensor [p, m, v] byte-stream chosen by ExpertState. Centralising
# the actual ``os.open / os.write / os.fsync`` here keeps every write syscall
# in this module while leaving ExpertState in charge of the format details
# (header packing, tensor schema, atomic-rename policy).
# ---------------------------------------------------------------------------

LFCL_PMV = Tuple[bytes, bytes, bytes]  # (param_bytes, m_bytes, v_bytes)


def write_lossless_payload_atomic(
    final_path: Path,
    header_bytes: bytes,
    tensor_bytes: Iterable[Tuple[str, bytes, bytes, bytes]],
    *,
    mode: int = 0o644,
) -> int:
    """Atomically write a lossless FC file: ``header || (p|m|v)*`` for each
    tensor in order. Writes to ``<final_path>.tmp`` first, ``fsync`` the fd,
    then ``os.replace`` to ``final_path``.

    Returns the total number of bytes written (header + every p/m/v byte
    string). Designed to be called from a writer-thread callback so the
    fsync round-trip does not stall the training thread.
    """
    final_path = Path(final_path)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    total = len(header_bytes)
    fd = os.open(str(tmp_path), os.O_CREAT | os.O_RDWR | os.O_TRUNC, mode)
    try:
        os.write(fd, header_bytes)
        for (_tname, p_bytes, m_bytes, v_bytes) in tensor_bytes:
            os.write(fd, p_bytes)
            os.write(fd, m_bytes)
            os.write(fd, v_bytes)
            total += len(p_bytes) + len(m_bytes) + len(v_bytes)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp_path), str(final_path))
    return total


# ---------------------------------------------------------------------------
# ExpertState LFSY (lossy log) writer.
#
# Lossy log layout (one file per expert):
#   header:  ``_pack_header(MAGIC_LFSY, meta)``  (caller supplies bytes)
#   appends: [ <count: u32> <step: u32> <records: count × RECORD_BYTES> ]*
# Each append is a single ``writev`` of (prefix, record_view). Writer-thread
# only — the lossy fd lifecycle is owned by the writer worker so two
# threads never race on the same descriptor.
# ---------------------------------------------------------------------------


def open_lossy_log_for_append(
    path: Path,
    header_bytes: bytes,
    *,
    mode: int = 0o644,
) -> int:
    """Create+truncate the lossy file at ``path``, write the header, and
    return the open fd. Caller (writer thread) holds the fd until close.
    The ``O_APPEND`` flag guarantees that every subsequent ``writev`` is
    atomic at the kernel level even under signal interruption."""
    flags = os.O_CREAT | os.O_RDWR | os.O_TRUNC | os.O_APPEND
    fd = os.open(str(path), flags, mode)
    try:
        os.write(fd, header_bytes)
    except Exception:
        os.close(fd)
        raise
    return fd


def append_lossy_log(
    fd: int, count: int, step: int, record_view: memoryview,
) -> int:
    """Append one ``(count, step, records)`` block to the lossy log fd via
    a single ``writev`` syscall. Returns total bytes written.
    ``record_view`` should be a contiguous memoryview over the structured
    record buffer (idx/p/m/v interleaved)."""
    prefix = struct.pack("<II", int(count), int(step))
    nbytes = os.writev(fd, [prefix, record_view])
    return int(nbytes)


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
