"""CPU-only FDP raw-write smoke test.

This script simulates the "GPU filled a pinned checkpoint buffer" state by
building 4KiB-aligned CPU uint8 tensors, filling them with deterministic test
payloads, then sending them through proposal_io's FDP write path.

Default behavior is dry-run only. Actual raw writes require:

    python -m megatron.proposal.fdp_cpu_smoke --run --yes

WARNING: --run writes directly to the configured NVMe namespace LBA ranges.
Use only on a disposable namespace/range.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import pickle
import re
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_DEVICE = "/dev/nvme1n1"
DEFAULT_NSID = 1
DEFAULT_METADATA_MIB = 64
DEFAULT_CHUNK_MIB = 64
DEFAULT_LAYERS = 2
DEFAULT_EXPERTS_PER_LAYER = 4


def _infer_block_device(device: str) -> str:
    """Return the block namespace device used for sysfs geometry."""
    path = Path(device)
    name = path.name
    m = re.fullmatch(r"ng(\d+)n(\d+)", name)
    if m:
        return str(path.with_name(f"nvme{m.group(1)}n{m.group(2)}"))
    return str(path)


def _infer_uring_device(device: str) -> str:
    """Prefer the NVMe generic namespace char device for uring_cmd IO."""
    path = Path(device)
    name = path.name
    m = re.fullmatch(r"nvme(\d+)n(\d+)", name)
    if not m:
        return str(path)
    generic = path.with_name(f"ng{m.group(1)}n{m.group(2)}")
    if generic.exists():
        return str(generic)
    return str(path)


def _sysfs_block_name(device: str) -> str:
    name = Path(device).name
    if not name:
        raise ValueError(f"bad device path: {device!r}")
    return name


def _read_int(path: Path) -> int:
    return int(path.read_text().strip())


def _device_geometry(device: str) -> Tuple[int, int, int]:
    """Return (logical_block_size, namespace_nlb, total_bytes) from sysfs."""
    name = _sysfs_block_name(device)
    base = Path("/sys/block") / name
    if not base.exists():
        raise FileNotFoundError(
            f"{base} not found. Is {device} a block device visible in sysfs?"
        )
    logical_block_size = _read_int(base / "queue" / "logical_block_size")
    sectors_512 = _read_int(base / "size")
    total_bytes = sectors_512 * 512
    if logical_block_size <= 0:
        raise RuntimeError(f"invalid logical_block_size={logical_block_size}")
    namespace_nlb = total_bytes // logical_block_size
    if namespace_nlb <= 0:
        raise RuntimeError(f"invalid namespace_nlb={namespace_nlb}")
    return logical_block_size, namespace_nlb, total_bytes


def _device_max_io_kb(device: str) -> int:
    name = _sysfs_block_name(device)
    base = Path("/sys/block") / name / "queue"
    max_sectors = _read_int(base / "max_sectors_kb")
    max_hw_sectors = _read_int(base / "max_hw_sectors_kb")
    return min(int(max_sectors), int(max_hw_sectors))


def _round_up(n: int, align: int) -> int:
    return ((int(n) + int(align) - 1) // int(align)) * int(align)


def _ceil_div(n: int, d: int) -> int:
    return (int(n) + int(d) - 1) // int(d)


def _alloc_aligned_cpu_uint8(nbytes: int, sector_bytes: int = 4096):
    """Allocate a CPU uint8 tensor with a sector-aligned data_ptr."""
    import torch

    padded = _round_up(nbytes, sector_bytes)
    over = torch.empty(padded + sector_bytes - 1, dtype=torch.uint8)
    base = int(over.data_ptr())
    misalign = base & (sector_bytes - 1)
    skip = 0 if misalign == 0 else sector_bytes - misalign
    out = over.narrow(0, skip, padded)
    if (int(out.data_ptr()) & (sector_bytes - 1)) != 0:
        raise RuntimeError("failed to allocate sector-aligned CPU tensor")
    out.zero_()
    return out


def _copy_bytes(dst, offset: int, data: bytes) -> None:
    import numpy as np
    import torch

    arr = np.frombuffer(data, dtype=np.uint8).copy()
    dst.narrow(0, int(offset), int(len(data))).copy_(torch.from_numpy(arr))


def _pack_expert_header(magic: bytes, meta: Dict[str, object],
                        header_bytes: int) -> bytes:
    payload = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
    prefix = magic + struct.pack("<II", 1, len(payload)) + payload
    if len(prefix) > header_bytes:
        raise RuntimeError(f"header too large: {len(prefix)} > {header_bytes}")
    return prefix + b"\x00" * (header_bytes - len(prefix))


def _make_lossy_blob(step: int, count: int) -> Tuple[object, int, bytes]:
    """Build one LFSY append block: count u32, step u32, records."""
    import numpy as np

    record_dtype = np.dtype([
        ("idx", "<i8"),
        ("p", "<f4"),
        ("m", "<f4"),
        ("v", "<f4"),
    ])
    records = np.zeros(int(count), dtype=record_dtype)
    records["idx"] = np.arange(int(count), dtype=np.int64)
    records["p"] = np.linspace(1.0, 2.0, int(count), dtype=np.float32)
    records["m"] = np.linspace(3.0, 4.0, int(count), dtype=np.float32)
    records["v"] = np.linspace(5.0, 6.0, int(count), dtype=np.float32)
    raw = struct.pack("<II", int(count), int(step)) + records.tobytes()
    buf = _alloc_aligned_cpu_uint8(len(raw), 4096)
    _copy_bytes(buf, 0, raw)
    return buf, len(raw), raw


def _make_large_lossy_blob(step: int, layer: int, expert: int,
                           nbytes: int) -> Tuple[object, int, bytes]:
    """Build a deterministic fixed-size LFSY append payload."""
    if nbytes <= 0:
        raise ValueError("large LFSY payload size must be positive")
    seed = (int(step) * 17 + int(layer) * 31 + int(expert) * 47) & 0xff
    pattern = bytes(((i + seed) & 0xff) for i in range(4096))
    full, rem = divmod(int(nbytes), len(pattern))
    raw = pattern * full + pattern[:rem]
    buf = _alloc_aligned_cpu_uint8(len(raw), 4096)
    _copy_bytes(buf, 0, raw)
    return buf, len(raw), raw


def _expert_slots(layers: int, experts_per_layer: int):
    return [
        (layer, expert)
        for layer in range(int(layers))
        for expert in range(int(experts_per_layer))
    ]


def _make_lfcs_payload(layer: int, expert: int, step: int,
                       generation: int, numel: int = 1024) -> bytes:
    import numpy as np

    base = float(layer * 1000 + expert * 100 + generation * 10 + step)
    payload = np.concatenate([
        np.full(numel, base + 1.0, dtype=np.float32),
        np.full(numel, base + 2.0, dtype=np.float32),
        np.full(numel, base + 3.0, dtype=np.float32),
    ])
    return payload.tobytes()


def _make_lfcs_header(layer: int, expert: int, step: int, generation: int,
                      header_bytes: int, numel: int = 1024) -> bytes:
    return _pack_expert_header(
        b"LFCL",
        {
            "expert_id": [int(layer), int(expert)],
            "step": int(step),
            "generation": int(generation),
            "tensors": [{"name": "cpu_smoke.expert", "shape": [int(numel)],
                         "numel": int(numel)}],
            "global_offsets": {"cpu_smoke.expert": 0},
            "total_numel": int(numel),
            "layout": "param_m_v_concat_fp32",
            "source": "fdp_cpu_smoke",
        },
        header_bytes,
    )


def _make_lfsy_header(layer: int, expert: int, start_step: int,
                      generation: int, header_bytes: int) -> bytes:
    return _pack_expert_header(
        b"LFSY",
        {
            "expert_id": [int(layer), int(expert)],
            "start_step": int(start_step),
            "generation": int(generation),
            "record_bytes": 20,
            "source": "fdp_cpu_smoke",
        },
        header_bytes,
    )


def _padded_bytes(raw: bytes, padded_len: int) -> bytes:
    return raw + b"\x00" * (int(padded_len) - len(raw))


def _make_nefc_payload(step: int, numel: int = 4096) -> bytes:
    import numpy as np

    payload = np.arange(numel, dtype=np.float32) + float(step * 100000)
    return payload.tobytes()


def _make_nefc_header(proposal_io, step: int, numel: int = 4096) -> bytes:
    import torch

    plan = [("cpu_smoke.nefc_tensor", 0, numel, torch.Size([numel]))]
    return proposal_io.build_prptens_header_block(
        plan,
        scalars={
            "step": int(step),
            "source": "fdp_cpu_smoke",
            "mode": "train_like",
        },
    )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _assert_bytes_equal(label: str, got: bytes, expected: bytes) -> None:
    if got == expected:
        print(
            f"[fdp-smoke] readback OK {label}: "
            f"bytes={len(expected)} sha256={_digest(expected)}"
        )
        return
    mismatch = next(
        (i for i, (a, b) in enumerate(zip(got, expected)) if a != b),
        min(len(got), len(expected)),
    )
    got_byte = got[mismatch] if mismatch < len(got) else None
    exp_byte = expected[mismatch] if mismatch < len(expected) else None
    raise RuntimeError(
        f"readback mismatch {label}: got_len={len(got)} "
        f"expected_len={len(expected)} first_mismatch={mismatch} "
        f"got={got_byte} expected={exp_byte} "
        f"got_sha256={_digest(got)} expected_sha256={_digest(expected)}"
    )


def _set_fdp_env(args: argparse.Namespace) -> Dict[str, str]:
    geometry_device = args.geometry_device or _infer_block_device(args.device)
    uring_device = args.device if args.no_auto_ng else _infer_uring_device(args.device)
    logical_block_size, namespace_nlb, total_bytes = _device_geometry(
        geometry_device
    )
    max_io_kb = int(args.max_io_kb)
    if max_io_kb <= 0:
        max_io_kb = _device_max_io_kb(geometry_device)

    metadata_nlb = args.metadata_nlb
    if metadata_nlb is None:
        metadata_bytes = int(args.metadata_mib) * 1024 * 1024
        metadata_nlb = _round_up(metadata_bytes, logical_block_size) // logical_block_size

    data_start_lba = args.data_start_lba
    if data_start_lba is None:
        data_start_lba = int(args.metadata_start_lba) + int(metadata_nlb)

    data_nlb = args.data_nlb
    if data_nlb is None:
        data_nlb = int(namespace_nlb) - int(data_start_lba)

    if data_nlb <= 0:
        raise RuntimeError(
            f"computed data_nlb={data_nlb}; check metadata/data layout"
        )

    env = {
        "PROPOSAL_IO_BACKEND": "fdp",
        "PROPOSAL_FDP_DEVICE": str(uring_device),
        "PROPOSAL_FDP_NSID": str(args.nsid),
        "PROPOSAL_FDP_LBA_SIZE": str(logical_block_size),
        "PROPOSAL_FDP_NAMESPACE_NLB": str(namespace_nlb),
        "PROPOSAL_FDP_ASSUME_FDP_SUPPORTED": "1",
        "PROPOSAL_FDP_METADATA_START_LBA": str(args.metadata_start_lba),
        "PROPOSAL_FDP_METADATA_NLB": str(metadata_nlb),
        "PROPOSAL_FDP_DATA_START_LBA": str(data_start_lba),
        "PROPOSAL_FDP_DATA_NLB": str(data_nlb),
        "PROPOSAL_FDP_CHUNK_MB": str(args.chunk_mib),
        "PROPOSAL_FDP_WORLD_SIZE": "1",
        "PROPOSAL_FDP_RANK": "0",
        "PROPOSAL_FDP_NUM_MOE_LAYERS": str(args.layers),
        "PROPOSAL_FDP_EXPERTS_PER_LAYER": str(args.experts_per_layer),
        "PROPOSAL_FDP_NEFC_QD": str(args.nefc_qd),
        "PROPOSAL_FDP_EXPERT_QD": str(args.expert_qd),
        "PROPOSAL_FDP_METADATA_QD": str(args.metadata_qd),
        "PROPOSAL_FDP_MAX_IO_KB": str(max_io_kb),
        "PROPOSAL_FDP_NVME_FLUSH": "1" if args.nvme_flush else "0",
        "PROPOSAL_FDP_WRITE_FUA": "1" if args.write_fua else "0",
        "PROPOSAL_FDP_ALLOW_FORMAT": "1" if args.format else "0",
        "PROPOSAL_FDP_FORCE_FORMAT": "1" if args.force_format else "0",
    }
    os.environ.update(env)

    env["_sysfs_logical_block_size"] = str(logical_block_size)
    env["_sysfs_namespace_nlb"] = str(namespace_nlb)
    env["_sysfs_total_bytes"] = str(total_bytes)
    env["_sysfs_total_gib"] = f"{total_bytes / (1024 ** 3):.3f}"
    env["_sysfs_max_io_kb"] = str(max_io_kb)
    env["_data_region_bytes"] = str(int(data_nlb) * int(logical_block_size))
    env["_data_region_gib"] = (
        f"{(int(data_nlb) * int(logical_block_size)) / (1024 ** 3):.3f}"
    )
    env["_data_chunk_count_estimate"] = str(
        (int(data_nlb) * int(logical_block_size)) //
        (int(args.chunk_mib) * 1024 * 1024)
    )
    env["_geometry_device"] = str(geometry_device)
    env["_requested_device"] = str(args.device)
    return env


def _print_env(env: Dict[str, str]) -> None:
    printable = {k: v for k, v in env.items() if k.startswith("PROPOSAL_")}
    printable["_sysfs_logical_block_size"] = env["_sysfs_logical_block_size"]
    printable["_sysfs_namespace_nlb"] = env["_sysfs_namespace_nlb"]
    printable["_sysfs_total_bytes"] = env["_sysfs_total_bytes"]
    printable["_sysfs_total_gib"] = env["_sysfs_total_gib"]
    printable["_sysfs_max_io_kb"] = env["_sysfs_max_io_kb"]
    printable["_data_region_bytes"] = env["_data_region_bytes"]
    printable["_data_region_gib"] = env["_data_region_gib"]
    printable["_data_chunk_count_estimate"] = env["_data_chunk_count_estimate"]
    printable["_geometry_device"] = env["_geometry_device"]
    printable["_requested_device"] = env["_requested_device"]
    print(json.dumps(printable, indent=2, sort_keys=True))


def _require_runtime_deps() -> None:
    missing = []
    for name in ("numpy", "torch"):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name == name:
                missing.append(name)
            else:
                raise
    if missing:
        exe = sys.executable
        raise SystemExit(
            "missing Python runtime dependency for --run: "
            + ", ".join(missing)
            + "\nRun this script with the same Python environment used for "
              "training, for example:\n"
            + f"  sudo -E {exe} {Path(__file__).name} --device /dev/nvme1n1 "
              "--format --raw-smoke --run --yes"
        )


def _ensure_python_bin_on_path() -> None:
    """Make torch cpp_extension find tools installed in this Python env.

    sudo often replaces PATH even with -E. torch.utils.cpp_extension checks
    ninja by executing ``ninja --version``, so the active env's bin directory
    must be present in PATH.
    """
    py_bin = str(Path(sys.executable).resolve().parent)
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    if py_bin not in parts:
        os.environ["PATH"] = py_bin + (os.pathsep + path if path else "")


def _load_local_proposal_io():
    """Load storage/proposal_io.py directly without importing megatron/__init__.py."""
    module_path = Path(__file__).resolve().parent / "storage" / "proposal_io.py"
    spec = importlib.util.spec_from_file_location(
        "_fdp_cpu_smoke_proposal_io", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_train_like_smoke(args: argparse.Namespace, proposal_io,
                          header_bytes: int, sector_bytes: int) -> None:
    slots = _expert_slots(args.layers, args.experts_per_layer)
    if not slots:
        raise RuntimeError("train-like smoke requires at least one expert slot")
    layers = int(args.layers)
    experts_per_layer = int(args.experts_per_layer)
    large_nefc_bytes = int(round(float(args.train_like_nefc_mib) *
                                 1024 * 1024))
    large_lfcs_bytes = int(round(float(args.train_like_lfcs_mib) *
                                 1024 * 1024))
    large_lossy_bytes = int(round(float(args.train_like_lossy_mib) *
                                  1024 * 1024))
    nefc_numel = (
        max(1, _ceil_div(large_nefc_bytes, 4))
        if large_nefc_bytes > 0
        else 4096
    )
    lfcs_numel = (
        max(1, _ceil_div(large_lfcs_bytes, 12))
        if large_lfcs_bytes > 0
        else 1024
    )

    generations = {eid: 0 for eid in slots}
    handles = {}
    lfcs_payloads = {}
    lfsy_expected = {}
    lfsy_expected_len = {}
    final_nefc_payload = b""
    stats = proposal_io.fdp_stats()
    chunk_bytes = int(stats["chunk_bytes"])
    max_io_bytes = int(stats.get("max_io_bytes", 0))
    if max_io_bytes <= 0:
        raise RuntimeError(f"invalid FDP max_io_bytes={max_io_bytes}")
    expected_cmds = {"nefc": 0, "expert": 0}

    def empty_batch_stats() -> Dict[str, float]:
        return {
            "commands": -1,
            "submitted": 0,
            "first_submit_ns": 0,
            "last_cqe_ns": 0,
            "submit_lifetime_ns": 0,
            "submit_lifetime_ms": 0.0,
        }

    def normalize_batch_stats(stats_dict) -> Dict[str, float]:
        if not stats_dict:
            return empty_batch_stats()
        return {
            "commands": int(stats_dict.get("commands", -1)),
            "submitted": int(stats_dict.get("submitted", 0)),
            "first_submit_ns": int(stats_dict.get("first_submit_ns", 0)),
            "last_cqe_ns": int(stats_dict.get("last_cqe_ns", 0)),
            "submit_lifetime_ns": int(stats_dict.get("submit_lifetime_ns", 0)),
            "submit_lifetime_ms": float(stats_dict.get("submit_lifetime_ms", 0.0)),
        }

    def data_write_cmds(nbytes: int) -> int:
        return _round_up(int(nbytes), max_io_bytes) // max_io_bytes

    def append_cmds(offset: int, nbytes: int) -> int:
        total = 0
        logical = int(offset)
        remaining = int(nbytes)
        while remaining > 0:
            chunk_left = int(chunk_bytes) - (logical % int(chunk_bytes))
            segment = min(remaining, chunk_left)
            total += data_write_cmds(segment)
            logical += segment
            remaining -= segment
        return total

    def close_handle(eid) -> None:
        handle = handles.pop(eid, None)
        if handle is not None:
            proposal_io.close_fd(int(handle))

    def open_lfsy(eid, step: int, header: bytes | None = None) -> None:
        layer, expert = eid
        if header is None:
            header = _make_lfsy_header(
                layer, expert, int(step), int(generations[eid]), header_bytes
            )
        handle = proposal_io.open_lossy_log_for_append(
            Path(f"lossy_expert_L{layer:02d}_E{expert:02d}.bin"),
            header,
            layer=int(layer),
            expert=int(expert),
            step=int(step),
            generation=int(generations[eid]),
        )
        handles[eid] = int(handle)
        lfsy_expected_len[eid] = len(header)
        if not args.skip_readback:
            lfsy_expected[eid] = bytes(header)
        expected_cmds["expert"] += data_write_cmds(len(header))

    def lossless_eids_for_step(step: int):
        if args.round_robin_scope == "slot":
            return [slots[(int(step) - 1) % len(slots)]]
        expert = (int(step) - 1) % experts_per_layer
        return [(layer, expert) for layer in range(layers)]

    rr_desc = (
        "expert-index round-robin"
        if args.round_robin_scope == "expert"
        else "slot round-robin"
    )
    lossy_desc = (
        f"fixed_bytes={large_lossy_bytes}"
        if large_lossy_bytes > 0
        else f"records_base={args.train_like_lossy_records}"
    )
    nefc_desc = (
        f"target_bytes={large_nefc_bytes} actual_bytes={nefc_numel * 4}"
        if large_nefc_bytes > 0
        else f"numel={nefc_numel}"
    )
    lfcs_desc = (
        f"target_bytes={large_lfcs_bytes} actual_bytes={lfcs_numel * 12}"
        if large_lfcs_bytes > 0
        else f"numel={lfcs_numel}"
    )
    print(
        "[fdp-smoke] train-like mode: "
        f"iters={args.train_like_iters} slots={len(slots)} "
        f"scope={rr_desc} "
        f"writers={'two-thread' if args.two_writer_threads else 'single-thread'} "
        "payload_mode=prebuilt-per-iter "
        f"nefc_payload={nefc_desc} "
        f"lfcs_payload={lfcs_desc} "
        f"lossy_payload={lossy_desc} "
        f"max_io_bytes={max_io_bytes}"
    )

    for step in range(1, int(args.train_like_iters) + 1):
        lossless_eids = lossless_eids_for_step(step)
        lossless_set = set(lossless_eids)
        expected_cmds["nefc"] = 0
        expected_cmds["expert"] = 0

        batch_counts = {"nefc_commands": -1, "expert_commands": -1,
                        "total_commands": -1}
        nefc_prepare_ms = 0.0
        expert_prepare_ms = 0.0
        nefc_fdp_write_ms = 0.0
        expert_fdp_write_ms = 0.0
        nefc_end_batch_ms = 0.0
        expert_end_batch_ms = 0.0
        nefc_submit_lifetime_ms = 0.0
        expert_submit_lifetime_ms = 0.0
        nefc_first_submit_ns = 0
        nefc_last_cqe_ns = 0
        expert_first_submit_ns = 0
        expert_last_cqe_ns = 0
        global_batch_submit_ms = 0.0
        nefc_written = 0
        lfcs_written_total = 0
        lfsy_written_total = 0
        selected_labels = []
        selected_generations = []
        lossy_count = 0

        prepare_start = time.perf_counter()
        nefc_payload = _make_nefc_payload(step, nefc_numel)
        nefc_header = _make_nefc_header(proposal_io, step, nefc_numel)
        nefc_buf = _alloc_aligned_cpu_uint8(
            header_bytes + len(nefc_payload), sector_bytes
        )
        _copy_bytes(nefc_buf, header_bytes, nefc_payload)
        prebuilt_nefc = {
            "payload": nefc_payload,
            "header": nefc_header,
            "buf": nefc_buf,
        }
        nefc_prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

        prepare_start = time.perf_counter()
        prebuilt_lfcs = []
        prebuilt_lfsy_opens = {}
        prebuilt_lfsy_appends = []

        for lossless_eid in lossless_eids:
            lossless_layer, lossless_expert = lossless_eid
            new_generation = int(generations[lossless_eid]) + 1
            payload = _make_lfcs_payload(
                lossless_layer, lossless_expert, step, new_generation,
                lfcs_numel
            )
            header = _make_lfcs_header(
                lossless_layer, lossless_expert, step, new_generation,
                header_bytes, lfcs_numel
            )
            buf = _alloc_aligned_cpu_uint8(
                header_bytes + len(payload), sector_bytes
            )
            _copy_bytes(buf, header_bytes, payload)
            prebuilt_lfcs.append({
                "eid": lossless_eid,
                "layer": int(lossless_layer),
                "expert": int(lossless_expert),
                "generation": int(new_generation),
                "payload": payload,
                "header": header,
                "buf": buf,
            })
            prebuilt_lfsy_opens[lossless_eid] = _make_lfsy_header(
                lossless_layer, lossless_expert, int(step),
                int(new_generation), header_bytes
            )

        for eid in slots:
            if eid in lossless_set:
                continue
            layer, expert = eid
            open_header = None
            if eid not in handles:
                open_header = _make_lfsy_header(
                    layer, expert, int(step), int(generations[eid]),
                    header_bytes
                )
            if large_lossy_bytes > 0:
                blob, logical_bytes, raw_blob = _make_large_lossy_blob(
                    step, layer, expert, large_lossy_bytes
                )
            else:
                n_records = max(
                    1,
                    int(args.train_like_lossy_records)
                    + ((int(step) + int(layer) + int(expert)) % 3),
                )
                blob, logical_bytes, raw_blob = _make_lossy_blob(
                    step, n_records
                )
            prebuilt_lfsy_appends.append({
                "eid": eid,
                "layer": int(layer),
                "expert": int(expert),
                "generation": int(generations[eid]),
                "open_header": open_header,
                "blob": blob,
                "logical_bytes": int(logical_bytes),
                "raw_blob": raw_blob,
            })
        expert_prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

        def run_nefc_write(keepalive_refs=None):
            refs = [] if keepalive_refs is None else keepalive_refs
            batch_started = False
            if args.batch_per_iter and args.two_writer_threads:
                proposal_io.fdp_begin_nefc_batch()
                batch_started = True
            try:
                nefc_payload = prebuilt_nefc["payload"]
                nefc_header = prebuilt_nefc["header"]
                nefc_buf = prebuilt_nefc["buf"]
                refs.append(nefc_buf)
                fdp_write_start = time.perf_counter()
                written = proposal_io.write_prptens_from_pinned(
                    Path("nefc_cpu_smoke.bin"),
                    nefc_buf,
                    nefc_header,
                    len(nefc_payload),
                    step=int(step),
                )
                expected_cmds["nefc"] += data_write_cmds(int(written))
                actual_cmds = -1
                end_batch_ms = 0.0
                batch_stats = empty_batch_stats()
                if args.batch_per_iter and args.two_writer_threads:
                    end_batch_start = time.perf_counter()
                    batch_stats = normalize_batch_stats(
                        proposal_io.fdp_end_nefc_batch_stats()
                    )
                    actual_cmds = int(batch_stats["commands"])
                    end_batch_ms = (
                        time.perf_counter() - end_batch_start
                    ) * 1000.0
                    batch_started = False
                    refs.clear()
                    if int(actual_cmds) != int(expected_cmds["nefc"]):
                        raise RuntimeError(
                            f"iter {step}: expected {expected_cmds['nefc']} "
                            f"NEFC batch commands, got {actual_cmds}"
                    )
                return {
                    "payload": nefc_payload,
                    "written": int(written),
                    "commands": int(actual_cmds),
                    "prepare_ms": float(nefc_prepare_ms),
                    "end_batch_ms": float(end_batch_ms),
                    "batch_stats": batch_stats,
                    "fdp_write_ms": (
                        time.perf_counter() - fdp_write_start
                    ) * 1000.0,
                }
            finally:
                if batch_started:
                    try:
                        proposal_io.fdp_end_nefc_batch()
                    except Exception:
                        pass

        def run_expert_write(keepalive_refs=None):
            refs = [] if keepalive_refs is None else keepalive_refs
            batch_started = False
            if args.batch_per_iter and args.two_writer_threads:
                proposal_io.fdp_begin_expert_batch()
                batch_started = True
            try:
                fdp_write_ms = 0.0
                local_lfcs_written_total = 0
                local_lfsy_written_total = 0
                local_selected_labels = []
                local_selected_generations = []
                local_lossy_count = 0

                for eid in lossless_eids:
                    close_handle(eid)

                for item in prebuilt_lfcs:
                    lossless_eid = item["eid"]
                    lossless_layer = int(item["layer"])
                    lossless_expert = int(item["expert"])
                    new_generation = int(item["generation"])
                    payload = item["payload"]
                    header = item["header"]
                    buf = item["buf"]
                    refs.append(buf)
                    fdp_write_start = time.perf_counter()
                    written = proposal_io.write_lossless_from_pinned(
                        Path(
                            f"fc_expert_L{lossless_layer:02d}_"
                            f"E{lossless_expert:02d}.bin"
                        ),
                        buf,
                        header,
                        len(payload),
                        layer=int(lossless_layer),
                        expert=int(lossless_expert),
                        step=int(step),
                        generation=int(new_generation),
                    )
                    fdp_write_ms += (
                        time.perf_counter() - fdp_write_start
                    ) * 1000.0
                    expected_cmds["expert"] += data_write_cmds(int(written))
                    local_lfcs_written_total += int(written)
                    generations[lossless_eid] = int(new_generation)
                    lfcs_payloads[lossless_eid] = payload
                    local_selected_labels.append(
                        f"L{lossless_layer:02d}/E{lossless_expert:02d}"
                    )
                    local_selected_generations.append(str(int(new_generation)))

                    # Match ExpertState behavior: after LFCS replacement, start
                    # a fresh LFSY object for future lossy appends.
                    fdp_write_start = time.perf_counter()
                    open_lfsy(
                        lossless_eid, step,
                        header=prebuilt_lfsy_opens[lossless_eid],
                    )
                    fdp_write_ms += (
                        time.perf_counter() - fdp_write_start
                    ) * 1000.0
                    local_lfsy_written_total += header_bytes

                for item in prebuilt_lfsy_appends:
                    eid = item["eid"]
                    layer = int(item["layer"])
                    expert = int(item["expert"])
                    if eid not in handles:
                        fdp_write_start = time.perf_counter()
                        open_lfsy(eid, step, header=item["open_header"])
                        fdp_write_ms += (
                            time.perf_counter() - fdp_write_start
                        ) * 1000.0
                        local_lfsy_written_total += header_bytes
                    blob = item["blob"]
                    logical_bytes = int(item["logical_bytes"])
                    raw_blob = item["raw_blob"]
                    refs.append(blob)
                    fdp_write_start = time.perf_counter()
                    padded_written = proposal_io.append_lossy_log_from_pinned(
                        int(handles[eid]),
                        blob,
                        logical_bytes,
                        layer=int(layer),
                        expert=int(expert),
                        step=int(step),
                        generation=int(generations[eid]),
                    )
                    fdp_write_ms += (
                        time.perf_counter() - fdp_write_start
                    ) * 1000.0
                    append_offset = int(lfsy_expected_len[eid])
                    expected_cmds["expert"] += append_cmds(
                        append_offset, int(padded_written)
                    )
                    local_lfsy_written_total += int(padded_written)
                    lfsy_expected_len[eid] = append_offset + int(padded_written)
                    if not args.skip_readback:
                        lfsy_expected[eid] += _padded_bytes(
                            raw_blob, int(padded_written)
                        )
                    local_lossy_count += 1

                actual_cmds = -1
                end_batch_ms = 0.0
                batch_stats = empty_batch_stats()
                if args.batch_per_iter and args.two_writer_threads:
                    fdp_write_start = time.perf_counter()
                    end_batch_start = fdp_write_start
                    batch_stats = normalize_batch_stats(
                        proposal_io.fdp_end_expert_batch_stats()
                    )
                    actual_cmds = int(batch_stats["commands"])
                    end_batch_ms = (
                        time.perf_counter() - end_batch_start
                    ) * 1000.0
                    fdp_write_ms += (
                        time.perf_counter() - fdp_write_start
                    ) * 1000.0
                    batch_started = False
                    refs.clear()
                    if int(actual_cmds) != int(expected_cmds["expert"]):
                        raise RuntimeError(
                            f"iter {step}: expected "
                            f"{expected_cmds['expert']} expert batch "
                            f"commands, got {actual_cmds}"
                        )
                return {
                    "lfcs_written_total": int(local_lfcs_written_total),
                    "lfsy_written_total": int(local_lfsy_written_total),
                    "selected_labels": local_selected_labels,
                    "selected_generations": local_selected_generations,
                    "lossy_count": int(local_lossy_count),
                    "commands": int(actual_cmds),
                    "prepare_ms": float(expert_prepare_ms),
                    "end_batch_ms": float(end_batch_ms),
                    "batch_stats": batch_stats,
                    "fdp_write_ms": float(fdp_write_ms),
                }
            finally:
                if batch_started:
                    try:
                        proposal_io.fdp_end_expert_batch()
                    except Exception:
                        pass

        data_write_start = time.perf_counter()
        if args.two_writer_threads:
            results = {}
            errors = []
            errors_lock = threading.Lock()

            def capture(name, fn):
                try:
                    results[name] = fn()
                except BaseException as exc:
                    with errors_lock:
                        errors.append(exc)

            nefc_thread = threading.Thread(
                target=capture, args=("nefc", run_nefc_write),
                name=f"fdp-smoke-nefc-{step}", daemon=True,
            )
            expert_thread = threading.Thread(
                target=capture, args=("expert", run_expert_write),
                name=f"fdp-smoke-expert-{step}", daemon=True,
            )
            nefc_thread.start()
            expert_thread.start()
            nefc_thread.join()
            expert_thread.join()
            if errors:
                raise errors[0]

            final_nefc_payload = results["nefc"]["payload"]
            nefc_written = int(results["nefc"]["written"])
            lfcs_written_total = int(results["expert"]["lfcs_written_total"])
            selected_labels = results["expert"]["selected_labels"]
            selected_generations = results["expert"]["selected_generations"]
            lossy_count = int(results["expert"]["lossy_count"])
            nefc_prepare_ms = float(results["nefc"]["prepare_ms"])
            expert_prepare_ms = float(results["expert"]["prepare_ms"])
            nefc_fdp_write_ms = float(results["nefc"]["fdp_write_ms"])
            expert_fdp_write_ms = float(results["expert"]["fdp_write_ms"])
            nefc_end_batch_ms = float(results["nefc"]["end_batch_ms"])
            expert_end_batch_ms = float(results["expert"]["end_batch_ms"])
            nefc_stats = normalize_batch_stats(results["nefc"].get("batch_stats"))
            expert_stats = normalize_batch_stats(
                results["expert"].get("batch_stats")
            )
            nefc_submit_lifetime_ms = float(nefc_stats["submit_lifetime_ms"])
            expert_submit_lifetime_ms = float(expert_stats["submit_lifetime_ms"])
            nefc_first_submit_ns = int(nefc_stats["first_submit_ns"])
            nefc_last_cqe_ns = int(nefc_stats["last_cqe_ns"])
            expert_first_submit_ns = int(expert_stats["first_submit_ns"])
            expert_last_cqe_ns = int(expert_stats["last_cqe_ns"])
            lfsy_written_total = int(results["expert"]["lfsy_written_total"])
            batch_counts["nefc_commands"] = int(results["nefc"]["commands"])
            batch_counts["expert_commands"] = int(results["expert"]["commands"])
            if (batch_counts["nefc_commands"] >= 0 and
                    batch_counts["expert_commands"] >= 0):
                batch_counts["total_commands"] = (
                    batch_counts["nefc_commands"] +
                    batch_counts["expert_commands"]
                )
        else:
            batch_refs = []
            if args.batch_per_iter:
                proposal_io.fdp_begin_data_batch()

            nefc_result = run_nefc_write(batch_refs)
            final_nefc_payload = nefc_result["payload"]
            nefc_written = int(nefc_result["written"])
            nefc_prepare_ms = float(nefc_result["prepare_ms"])
            nefc_fdp_write_ms = float(nefc_result["fdp_write_ms"])

            expert_result = run_expert_write(batch_refs)
            lfcs_written_total = int(expert_result["lfcs_written_total"])
            selected_labels = expert_result["selected_labels"]
            selected_generations = expert_result["selected_generations"]
            lossy_count = int(expert_result["lossy_count"])
            expert_prepare_ms = float(expert_result["prepare_ms"])
            expert_fdp_write_ms = float(expert_result["fdp_write_ms"])
            lfsy_written_total = int(expert_result["lfsy_written_total"])

            if args.batch_per_iter:
                global_batch_submit_start = time.perf_counter()
                batch_stats = proposal_io.fdp_end_data_batch_stats()
                global_batch_submit_ms = (
                    time.perf_counter() - global_batch_submit_start
                ) * 1000.0
                nefc_stats = normalize_batch_stats(batch_stats.get("nefc"))
                expert_stats = normalize_batch_stats(batch_stats.get("expert"))
                nefc_submit_lifetime_ms = float(
                    nefc_stats["submit_lifetime_ms"]
                )
                expert_submit_lifetime_ms = float(
                    expert_stats["submit_lifetime_ms"]
                )
                nefc_first_submit_ns = int(nefc_stats["first_submit_ns"])
                nefc_last_cqe_ns = int(nefc_stats["last_cqe_ns"])
                expert_first_submit_ns = int(expert_stats["first_submit_ns"])
                expert_last_cqe_ns = int(expert_stats["last_cqe_ns"])
                batch_counts = {
                    "nefc_commands": int(batch_stats["nefc_commands"]),
                    "expert_commands": int(batch_stats["expert_commands"]),
                    "total_commands": int(batch_stats["total_commands"]),
                }
                batch_refs.clear()
                if int(batch_counts["nefc_commands"]) != int(expected_cmds["nefc"]):
                    raise RuntimeError(
                        f"iter {step}: expected {expected_cmds['nefc']} NEFC "
                        f"batch commands, "
                        f"got {batch_counts['nefc_commands']}"
                    )
                if int(batch_counts["expert_commands"]) != int(
                        expected_cmds["expert"]):
                    raise RuntimeError(
                        f"iter {step}: expected {expected_cmds['expert']} "
                        f"expert batch commands, "
                        f"got {batch_counts['expert_commands']}"
                    )
        data_total_ms = (time.perf_counter() - data_write_start) * 1000.0
        prepare_ms = nefc_prepare_ms + expert_prepare_ms
        fdp_write_sum_ms = (
            nefc_fdp_write_ms + expert_fdp_write_ms + global_batch_submit_ms
        )
        fdp_write_wall_ms = (
            max(nefc_fdp_write_ms, expert_fdp_write_ms)
            if args.two_writer_threads
            else fdp_write_sum_ms
        )
        metadata_flush_ms = 0.0
        if args.per_iter_flush:
            metadata_flush_start = time.perf_counter()
            proposal_io.flush_metadata(int(step))
            metadata_flush_ms = (
                time.perf_counter() - metadata_flush_start
            ) * 1000.0
        write_ms = fdp_write_wall_ms + metadata_flush_ms
        data_written_total = int(nefc_written) + int(lfcs_written_total) + int(
            lfsy_written_total
        )
        expert_written_total = int(lfcs_written_total) + int(lfsy_written_total)
        fdp_write_mib_s = 0.0
        if fdp_write_wall_ms > 0:
            fdp_write_mib_s = (
                float(data_written_total) / (1024.0 * 1024.0)
            ) / (fdp_write_wall_ms / 1000.0)
        io_wall_mib_s = 0.0
        if data_total_ms > 0:
            io_wall_mib_s = (
                float(data_written_total) / (1024.0 * 1024.0)
            ) / (data_total_ms / 1000.0)
        nefc_submit_mib_s = 0.0
        if nefc_submit_lifetime_ms > 0:
            nefc_submit_mib_s = (
                float(nefc_written) / (1024.0 * 1024.0)
            ) / (nefc_submit_lifetime_ms / 1000.0)
        expert_submit_mib_s = 0.0
        if expert_submit_lifetime_ms > 0:
            expert_submit_mib_s = (
                float(expert_written_total) / (1024.0 * 1024.0)
            ) / (expert_submit_lifetime_ms / 1000.0)
        submit_lifetime_ms = 0.0
        submit_mib_s = 0.0
        first_submit_times = [
            t for t in (nefc_first_submit_ns, expert_first_submit_ns) if t > 0
        ]
        last_cqe_times = [
            t for t in (nefc_last_cqe_ns, expert_last_cqe_ns) if t > 0
        ]
        if first_submit_times and last_cqe_times:
            submit_lifetime_ms = (
                max(last_cqe_times) - min(first_submit_times)
            ) / 1000000.0
            if submit_lifetime_ms > 0:
                submit_mib_s = (
                    float(data_written_total) / (1024.0 * 1024.0)
                ) / (submit_lifetime_ms / 1000.0)

        print(
            f"[fdp-smoke] iter={step:03d} "
            f"LFCS={','.join(selected_labels)} "
            f"gen={','.join(selected_generations)} "
            f"nefc_bytes={nefc_written} "
            f"lfcs_bytes={lfcs_written_total} "
            f"lfsy_bytes={lfsy_written_total} LFSY_appends={lossy_count} "
            f"batch_nefc_cmds={batch_counts['nefc_commands']} "
            f"batch_expert_cmds={batch_counts['expert_commands']} "
            f"prepare_ms={prepare_ms:.3f} "
            f"nefc_fdp_write_ms={nefc_fdp_write_ms:.3f} "
            f"expert_fdp_write_ms={expert_fdp_write_ms:.3f} "
            f"nefc_end_batch_ms={nefc_end_batch_ms:.3f} "
            f"expert_end_batch_ms={expert_end_batch_ms:.3f} "
            f"nefc_submit_lifetime_ms={nefc_submit_lifetime_ms:.3f} "
            f"expert_submit_lifetime_ms={expert_submit_lifetime_ms:.3f} "
            f"submit_lifetime_ms={submit_lifetime_ms:.3f} "
            f"nefc_submit_mib_s={nefc_submit_mib_s:.1f} "
            f"expert_submit_mib_s={expert_submit_mib_s:.1f} "
            f"submit_mib_s={submit_mib_s:.1f} "
            f"fdp_write_ms={fdp_write_wall_ms:.3f} "
            f"fdp_write_sum_ms={fdp_write_sum_ms:.3f} "
            f"fdp_write_mib_s={fdp_write_mib_s:.1f} "
            f"io_wall_mib_s={io_wall_mib_s:.1f} "
            f"metadata_flush_ms={metadata_flush_ms:.3f} "
            f"write_ms={write_ms:.3f} "
            f"data_total_ms={data_total_ms:.3f}"
        )
        if args.iter_sleep_sec > 0 and step < int(args.train_like_iters):
            time.sleep(float(args.iter_sleep_sec))

    for handle in list(handles.values()):
        proposal_io.fsync_fd(int(handle))
    if not args.per_iter_flush:
        final_flush_start = time.perf_counter()
        proposal_io.flush_metadata(int(args.train_like_iters))
        print(
            "[fdp-smoke] final metadata flush: "
            f"metadata_flush_ms={(time.perf_counter() - final_flush_start) * 1000.0:.3f}"
        )

    summary = proposal_io.fdp_debug_read_metadata_summary()
    print("[fdp-smoke] metadata summary after train-like writes:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    stats = proposal_io.fdp_stats()
    chunk_bytes = int(stats["chunk_bytes"])
    lfsy_entries = proposal_io.fdp_debug_read_lfsy_entries()
    valid_lfsy_entries = [
        dict(entry) for entry in lfsy_entries if bool(entry.get("valid"))
    ]
    print("[fdp-smoke] LFSY entry summary after train-like writes:")
    for entry in sorted(
        valid_lfsy_entries,
        key=lambda e: (int(e["layer_id"]), int(e["expert_id"])),
    ):
        chunks = [int(c) for c in entry.get("chunks", [])]
        preview = chunks[:4]
        suffix = "..." if len(chunks) > len(preview) else ""
        print(
            "[fdp-smoke] "
            f"L{int(entry['layer_id']):02d}/E{int(entry['expert_id']):02d} "
            f"gen={int(entry['generation'])} "
            f"tail={int(entry['logical_tail'])} "
            f"chunk_count={int(entry['chunk_count'])} "
            f"chunks={preview}{suffix}"
        )

    expected_lfcs = len(lfcs_payloads)
    expected_lfsy = len(slots)
    expected_nefc_chunks = (
        _round_up(header_bytes + len(final_nefc_payload), chunk_bytes)
        // chunk_bytes
    )
    expected_lfcs_chunks = sum(
        _round_up(header_bytes + len(payload), chunk_bytes) // chunk_bytes
        for payload in lfcs_payloads.values()
    )
    expected_lfcs_max_chunks = max(
        (
            _round_up(header_bytes + len(payload), chunk_bytes) // chunk_bytes
            for payload in lfcs_payloads.values()
        ),
        default=0,
    )
    entries_by_eid = {
        (int(entry["layer_id"]), int(entry["expert_id"])): entry
        for entry in valid_lfsy_entries
    }
    expected_lfsy_chunks = 0
    max_lfsy_chunks = 0
    for eid in slots:
        if eid not in entries_by_eid:
            raise RuntimeError(f"missing valid LFSY entry for {eid}")
        entry = entries_by_eid[eid]
        expected_tail = int(lfsy_expected_len[eid])
        expected_chunk_count = (
            _round_up(expected_tail, chunk_bytes) // chunk_bytes
        )
        actual_tail = int(entry["logical_tail"])
        actual_chunk_count = int(entry["chunk_count"])
        if actual_tail != expected_tail:
            raise RuntimeError(
                f"LFSY tail mismatch for {eid}: "
                f"got={actual_tail} expected={expected_tail}"
            )
        if actual_chunk_count != expected_chunk_count:
            raise RuntimeError(
                f"LFSY chunk_count mismatch for {eid}: "
                f"got={actual_chunk_count} expected={expected_chunk_count}"
            )
        expected_lfsy_chunks += expected_chunk_count
        max_lfsy_chunks = max(max_lfsy_chunks, actual_chunk_count)

    actual_nefc_chunks = int(summary.get("nefc_chunk_count", 0))
    actual_lfcs_chunks = int(summary.get("lfcs_total_chunks", -1))
    actual_lfcs_max_chunks = int(summary.get("lfcs_max_chunk_count", -1))
    print(
        "[fdp-smoke] expected live chunk summary: "
        f"nefc={expected_nefc_chunks} "
        f"lfcs_total={expected_lfcs_chunks} "
        f"lfcs_max={expected_lfcs_max_chunks} "
        f"lfsy_total={expected_lfsy_chunks}"
    )
    expected_chunks = (
        expected_nefc_chunks + expected_lfcs_chunks + expected_lfsy_chunks
    )
    if actual_nefc_chunks != expected_nefc_chunks:
        raise RuntimeError(
            f"unexpected nefc_chunk_count={actual_nefc_chunks} "
            f"expected={expected_nefc_chunks}"
        )
    if actual_lfcs_chunks != -1 and actual_lfcs_chunks != expected_lfcs_chunks:
        raise RuntimeError(
            f"unexpected lfcs_total_chunks={actual_lfcs_chunks} "
            f"expected={expected_lfcs_chunks}"
        )
    if (actual_lfcs_max_chunks != -1 and
            actual_lfcs_max_chunks != expected_lfcs_max_chunks):
        raise RuntimeError(
            f"unexpected lfcs_max_chunk_count={actual_lfcs_max_chunks} "
            f"expected={expected_lfcs_max_chunks}"
        )
    if int(summary.get("lfcs_valid_count", -1)) != expected_lfcs:
        raise RuntimeError(
            f"unexpected lfcs_valid_count={summary.get('lfcs_valid_count')} "
            f"expected={expected_lfcs}"
        )
    if int(summary.get("lfsy_valid_count", -1)) != expected_lfsy:
        raise RuntimeError(
            f"unexpected lfsy_valid_count={summary.get('lfsy_valid_count')} "
            f"expected={expected_lfsy}"
        )
    if int(summary.get("allocated_chunks", -1)) != expected_chunks:
        raise RuntimeError(
            f"unexpected allocated_chunks={summary.get('allocated_chunks')} "
            f"expected={expected_chunks}"
        )
    if (int(args.expect_lfsy_min_chunks) > 0 and
            max_lfsy_chunks < int(args.expect_lfsy_min_chunks)):
        raise RuntimeError(
            f"max LFSY chunk_count={max_lfsy_chunks}, "
            f"expected at least {args.expect_lfsy_min_chunks}"
        )
    if (int(args.expect_nefc_min_chunks) > 0 and
            actual_nefc_chunks < int(args.expect_nefc_min_chunks)):
        raise RuntimeError(
            f"NEFC chunk_count={actual_nefc_chunks}, "
            f"expected at least {args.expect_nefc_min_chunks}"
        )
    lfcs_max_for_min = (
        actual_lfcs_max_chunks
        if actual_lfcs_max_chunks != -1
        else expected_lfcs_max_chunks
    )
    if (int(args.expect_lfcs_min_chunks) > 0 and
            lfcs_max_for_min < int(args.expect_lfcs_min_chunks)):
        raise RuntimeError(
            f"LFCS max chunk_count={lfcs_max_for_min}, "
            f"expected at least {args.expect_lfcs_min_chunks}"
        )

    if not args.skip_readback:
        print("[fdp-smoke] verifying train-like payload readback")
        _assert_bytes_equal(
            "NEFC payload",
            proposal_io.fdp_debug_read_nefc(
                header_bytes, len(final_nefc_payload)
            ),
            final_nefc_payload,
        )
        for eid in slots:
            layer, expert = eid
            if eid in lfcs_payloads:
                _assert_bytes_equal(
                    f"LFCS L{layer:02d}/E{expert:02d}",
                    proposal_io.fdp_debug_read_lfcs(
                        layer, expert, header_bytes, len(lfcs_payloads[eid])
                    ),
                    lfcs_payloads[eid],
                )
            _assert_bytes_equal(
                f"LFSY L{layer:02d}/E{expert:02d}",
                proposal_io.fdp_debug_read_lfsy(
                    layer, expert, 0, len(lfsy_expected[eid])
                ),
                lfsy_expected[eid],
            )

    for eid in list(handles):
        close_handle(eid)


def run_smoke(args: argparse.Namespace) -> None:
    env = _set_fdp_env(args)
    print("[fdp-smoke] resolved environment:")
    _print_env(env)

    metadata_start = int(env["PROPOSAL_FDP_METADATA_START_LBA"])
    metadata_nlb = int(env["PROPOSAL_FDP_METADATA_NLB"])
    data_start = int(env["PROPOSAL_FDP_DATA_START_LBA"])
    data_nlb = int(env["PROPOSAL_FDP_DATA_NLB"])
    print(
        "[fdp-smoke] raw write ranges: "
        f"metadata=[{metadata_start}, {metadata_start + metadata_nlb}), "
        f"data=[{data_start}, {data_start + data_nlb})"
    )
    print("[fdp-smoke] current FDP placement PID wiring uses PID 0 only.")

    if not args.run:
        print("[fdp-smoke] dry-run only. Add --run --yes to write to the device.")
        return
    if not args.yes:
        raise SystemExit(
            "--run requires --yes because this writes directly to the NVMe namespace"
        )

    _require_runtime_deps()
    _ensure_python_bin_on_path()

    # Import after env setup so the lazy FDP extension opens with this config.
    import numpy as np
    import torch

    proposal_io = _load_local_proposal_io()

    header_bytes = int(proposal_io.HEADER_BLOCK_BYTES)
    sector_bytes = int(proposal_io.SECTOR_BYTES)

    print("[fdp-smoke] opening FDP session")
    proposal_io.open_fdp_session()
    print("[fdp-smoke] stats after open:")
    print(json.dumps(proposal_io.fdp_stats(), indent=2, sort_keys=True))

    if args.raw_smoke:
        ok = proposal_io.fdp_debug_raw_write_read_smoke()
        print(f"[fdp-smoke] raw metadata write/read smoke: {ok}")

    if int(args.train_like_iters) > 0:
        _run_train_like_smoke(args, proposal_io, header_bytes, sector_bytes)
        proposal_io.close_fdp_session()
        print("[fdp-smoke] done")
        return

    # NEFC: PRPTENS-like non-expert checkpoint payload.
    nefc_numel = 4096
    nefc_payload = np.arange(nefc_numel, dtype=np.float32)
    nefc_payload_bytes = nefc_payload.tobytes()
    nefc_plan = [("cpu_smoke.nefc_tensor", 0, nefc_numel, torch.Size([nefc_numel]))]
    nefc_header = proposal_io.build_prptens_header_block(
        nefc_plan, scalars={"step": 1, "source": "fdp_cpu_smoke"}
    )
    nefc_buf = _alloc_aligned_cpu_uint8(
        header_bytes + len(nefc_payload_bytes), sector_bytes
    )
    _copy_bytes(nefc_buf, header_bytes, nefc_payload_bytes)
    nefc_written = proposal_io.write_prptens_from_pinned(
        Path("nefc_cpu_smoke.bin"),
        nefc_buf,
        nefc_header,
        len(nefc_payload_bytes),
        step=1,
    )
    print(f"[fdp-smoke] wrote NEFC bytes={nefc_written}")

    # LFCS: one expert full snapshot.
    layer_id = int(args.test_layer)
    expert_id = int(args.test_expert)
    lfcs_numel = 1024
    lfcs_payload = np.concatenate([
        np.full(lfcs_numel, 11.0, dtype=np.float32),
        np.full(lfcs_numel, 22.0, dtype=np.float32),
        np.full(lfcs_numel, 33.0, dtype=np.float32),
    ])
    lfcs_payload_bytes = lfcs_payload.tobytes()
    lfcs_header = _pack_expert_header(
        b"LFCL",
        {
            "expert_id": [layer_id, expert_id],
            "step": 2,
            "tensors": [{"name": "cpu_smoke.expert", "shape": [lfcs_numel],
                         "numel": lfcs_numel}],
            "global_offsets": {"cpu_smoke.expert": 0},
            "total_numel": lfcs_numel,
            "layout": "param_m_v_concat_fp32",
            "source": "fdp_cpu_smoke",
        },
        header_bytes,
    )
    lfcs_buf = _alloc_aligned_cpu_uint8(
        header_bytes + len(lfcs_payload_bytes), sector_bytes
    )
    _copy_bytes(lfcs_buf, header_bytes, lfcs_payload_bytes)
    lfcs_written = proposal_io.write_lossless_from_pinned(
        Path(f"fc_expert_L{layer_id:02d}_E{expert_id:02d}.bin"),
        lfcs_buf,
        lfcs_header,
        len(lfcs_payload_bytes),
        layer=layer_id,
        expert=expert_id,
        step=2,
        generation=1,
    )
    print(f"[fdp-smoke] wrote LFCS bytes={lfcs_written}")

    # LFSY: open a fresh lossy log and append two synthetic blocks.
    lfsy_header = _pack_expert_header(
        b"LFSY",
        {
            "expert_id": [layer_id, expert_id],
            "start_step": 2,
            "generation": 1,
            "record_bytes": 20,
            "source": "fdp_cpu_smoke",
        },
        header_bytes,
    )
    handle = proposal_io.open_lossy_log_for_append(
        Path(f"lossy_expert_L{layer_id:02d}_E{expert_id:02d}.bin"),
        lfsy_header,
        layer=layer_id,
        expert=expert_id,
        step=2,
        generation=1,
    )
    print(f"[fdp-smoke] opened LFSY handle={handle}")
    lfsy_blocks = []
    for step, count in [(3, 5), (4, 7)]:
        blob, logical_bytes, raw_blob = _make_lossy_blob(step, count)
        written = proposal_io.append_lossy_log_from_pinned(
            handle,
            blob,
            logical_bytes,
            layer=layer_id,
            expert=expert_id,
            step=step,
            generation=1,
        )
        print(
            f"[fdp-smoke] appended LFSY step={step} "
            f"logical={logical_bytes} padded_written={written}"
        )
        lfsy_blocks.append((step, raw_blob, written))
    proposal_io.fsync_fd(handle)
    proposal_io.close_fd(handle)

    proposal_io.flush_metadata(4)
    summary = proposal_io.fdp_debug_read_metadata_summary()
    print("[fdp-smoke] metadata summary after writes:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.skip_readback:
        print("[fdp-smoke] verifying payload readback")
        _assert_bytes_equal(
            "NEFC payload",
            proposal_io.fdp_debug_read_nefc(
                header_bytes, len(nefc_payload_bytes)
            ),
            nefc_payload_bytes,
        )
        _assert_bytes_equal(
            "LFCS payload",
            proposal_io.fdp_debug_read_lfcs(
                layer_id, expert_id, header_bytes, len(lfcs_payload_bytes)
            ),
            lfcs_payload_bytes,
        )
        _assert_bytes_equal(
            "LFSY header",
            proposal_io.fdp_debug_read_lfsy(
                layer_id, expert_id, 0, len(lfsy_header)
            ),
            lfsy_header,
        )
        lfsy_offset = header_bytes
        for step, raw_blob, padded_written in lfsy_blocks:
            expected = raw_blob + b"\x00" * (int(padded_written) - len(raw_blob))
            _assert_bytes_equal(
                f"LFSY append step={step}",
                proposal_io.fdp_debug_read_lfsy(
                    layer_id, expert_id, lfsy_offset, int(padded_written)
                ),
                expected,
            )
            lfsy_offset += int(padded_written)
    proposal_io.close_fdp_session()
    print("[fdp-smoke] done")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--geometry-device", default=None,
                   help="block device used only for sysfs geometry")
    p.add_argument("--no-auto-ng", action="store_true",
                   help="do not auto-map /dev/nvmeXnY to /dev/ngXnY")
    p.add_argument("--nsid", type=int, default=DEFAULT_NSID)
    p.add_argument("--metadata-start-lba", type=int, default=0)
    p.add_argument("--metadata-nlb", type=int, default=None)
    p.add_argument("--metadata-mib", type=int, default=DEFAULT_METADATA_MIB)
    p.add_argument("--data-start-lba", type=int, default=None)
    p.add_argument("--data-nlb", type=int, default=None)
    p.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_MIB)
    p.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    p.add_argument("--experts-per-layer", type=int,
                   default=DEFAULT_EXPERTS_PER_LAYER)
    p.add_argument("--test-layer", type=int, default=0)
    p.add_argument("--test-expert", type=int, default=0)
    p.add_argument("--train-like-iters", type=int, default=0,
                   help=("run a training-like expert loop: each iter writes "
                         "NEFC, selected LFCS, and LFSY for the rest"))
    p.add_argument("--round-robin-scope", choices=("expert", "slot"),
                   default="expert",
                   help=("LFCS selection for train-like mode: expert selects "
                         "the same expert index across all layers; slot "
                         "selects one flattened layer/expert slot"))
    p.add_argument("--preset-2g-iter", action="store_true",
                   help=("set train-like payload sizes to roughly 2GiB of "
                         "data-region writes per iteration"))
    p.add_argument("--train-like-lossy-records", type=int, default=5,
                   help="base LFSY record count per lossy expert append")
    p.add_argument("--train-like-nefc-mib", type=float, default=0.0,
                   help=("use a larger train-like NEFC payload targeting this "
                         "many MiB; 0 keeps the small default payload"))
    p.add_argument("--train-like-lfcs-mib", type=float, default=0.0,
                   help=("use larger train-like LFCS payloads targeting this "
                         "many MiB per selected expert; 0 keeps the small "
                         "default payload"))
    p.add_argument("--train-like-lossy-mib", type=float, default=0.0,
                   help=("use a fixed-size deterministic LFSY append payload "
                         "of this many MiB; 0 keeps the small record payload"))
    p.add_argument("--expect-nefc-min-chunks", type=int, default=0,
                   help=("fail unless the final NEFC entry has at least this "
                         "many chunks"))
    p.add_argument("--expect-lfcs-min-chunks", type=int, default=0,
                   help=("fail unless at least one final LFCS entry has this "
                         "many chunks"))
    p.add_argument("--expect-lfsy-min-chunks", type=int, default=0,
                   help=("fail unless at least one final LFSY entry has this "
                         "many chunks"))
    p.add_argument("--iter-sleep-sec", type=float, default=0.05,
                   help="sleep duration between train-like iterations")
    p.add_argument("--no-batch-per-iter", dest="batch_per_iter",
                   action="store_false",
                   help="disable per-iteration io_uring data batching")
    p.add_argument("--two-writer-threads", action="store_true",
                   help=("in train-like mode, run NEFC writes and expert "
                         "LFCS/LFSY writes on separate Python threads with "
                         "independent per-ring io_uring batches"))
    p.add_argument("--no-per-iter-flush", dest="per_iter_flush",
                   action="store_false",
                   help="flush metadata only once after the train-like loop")
    p.set_defaults(batch_per_iter=True, per_iter_flush=True)
    p.add_argument("--nefc-qd", type=int, default=32)
    p.add_argument("--expert-qd", type=int, default=64)
    p.add_argument("--metadata-qd", type=int, default=8)
    p.add_argument("--max-io-kb", type=int, default=0,
                   help=("maximum data write size per NVMe command in KiB; "
                         "0 reads sysfs queue max_sectors_kb"))
    p.add_argument("--format", action="store_true",
                   help="allow creating a fresh FDP superblock/metadata image")
    p.add_argument("--force-format", action="store_true",
                   help=("overwrite any existing FDP superblock/metadata "
                         "using the current namespace geometry"))
    p.add_argument("--no-nvme-flush", dest="nvme_flush", action="store_false")
    p.add_argument("--write-fua", action="store_true",
                   help=("set NVMe write FUA so completion waits for "
                         "FDPVirt NAND program path"))
    p.add_argument("--raw-smoke", action="store_true",
                   help="also run one metadata LBA write/read smoke")
    p.add_argument("--skip-readback", action="store_true",
                   help="skip byte-for-byte payload readback checks")
    p.add_argument("--run", action="store_true",
                   help="perform raw writes; default is dry-run only")
    p.add_argument("--yes", action="store_true",
                   help="required with --run")
    args = p.parse_args()
    if args.preset_2g_iter:
        args.train_like_nefc_mib = 256.0
        args.train_like_lfcs_mib = 256.0
        args.train_like_lossy_mib = 214.0
        args.expect_nefc_min_chunks = max(int(args.expect_nefc_min_chunks), 5)
        args.expect_lfcs_min_chunks = max(int(args.expect_lfcs_min_chunks), 5)
        args.expect_lfsy_min_chunks = max(int(args.expect_lfsy_min_chunks), 11)
    return args


def main() -> None:
    run_smoke(parse_args())


if __name__ == "__main__":
    main()
