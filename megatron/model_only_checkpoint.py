"""DeepSpeed-compatible BF16 model-only checkpoints.

The training checkpoint path stores optimizer/master-weight state as well as
the model.  Downstream evaluation only needs the live low-precision model, so
this module reuses DeepSpeed's native MoE shard writer while deliberately
disabling optimizer and ZeRO-state serialization.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from megatron import get_args
from megatron.checkpointing import deepspeed_checkpoint_state_dict
from megatron.utils import print_rank_0


FORMAT_NAME = "deepspeed_moe_bf16_model_only"
FORMAT_VERSION = 1
MANIFEST_NAME = "model_only_manifest.json"


def _dist_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank_world() -> tuple[int, int]:
    if not _dist_ready():
        return 0, 1
    return torch.distributed.get_rank(), torch.distributed.get_world_size()


def _barrier() -> None:
    if _dist_ready():
        torch.distributed.barrier()


def _broadcast_object(value, src: int = 0):
    if not _dist_ready():
        return value
    payload = [value]
    torch.distributed.broadcast_object_list(payload, src=src)
    return payload[0]


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _payload_files(tag_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in tag_dir.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    )


def _distributed_file_records(tag_dir: Path) -> List[Dict[str, object]]:
    rank, world = _rank_world()
    files = _payload_files(tag_dir)
    local = []
    for index, path in enumerate(files):
        if index % world != rank:
            continue
        checksum = sha256_file(path)
        _fsync_file(path)
        local.append({
            "path": path.relative_to(tag_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": checksum,
        })

    if not _dist_ready():
        return local
    gathered: List[List[Dict[str, object]]] = [None] * world
    torch.distributed.all_gather_object(gathered, local)
    return sorted(
        (record for rank_records in gathered for record in rank_records),
        key=lambda record: str(record["path"]),
    )


def _copy_file_durable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as src, open(destination, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def _write_sidecar_metadata(tag_dir: Path, args) -> None:
    tokenizer_dir = tag_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_sources = (
        (getattr(args, "vocab_file", None), tokenizer_dir / "gpt2-vocab.json"),
        (getattr(args, "merge_file", None), tokenizer_dir / "gpt2-merges.txt"),
    )
    for source, destination in tokenizer_sources:
        if not source:
            raise RuntimeError(f"missing tokenizer source for {destination.name}")
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"tokenizer file does not exist: {source_path}")
        _copy_file_durable(source_path, destination)

    model_fields = (
        "num_layers", "hidden_size", "num_attention_heads",
        "num_key_value_heads", "seq_length", "max_position_embeddings",
        "padded_vocab_size", "num_experts", "tensor_model_parallel_size",
        "pipeline_model_parallel_size", "moe_expert_parallel_size",
    )
    model_config = {
        key: getattr(args, key)
        for key in model_fields
        if getattr(args, key, None) is not None
    }
    model_config["parameter_dtype"] = "bfloat16"
    _atomic_write_text(
        tag_dir / "model_config.json",
        json.dumps(model_config, indent=2, sort_keys=True) + "\n",
    )

    ds_config = getattr(args, "deepspeed_config_dict", None)
    if isinstance(ds_config, dict):
        _atomic_write_text(
            tag_dir / "deepspeed_config.json",
            json.dumps(ds_config, indent=2, sort_keys=True) + "\n",
        )
    _fsync_dir(tokenizer_dir)


def _validate_bf16_parameters(engine) -> None:
    bad = []
    for name, parameter in engine.module.named_parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            bad.append(f"{name}:{parameter.dtype}")
            if len(bad) >= 8:
                break
    if _dist_ready():
        rank, world = _rank_world()
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, (rank, bad))
        bad_by_rank = [(r, values) for r, values in gathered if values]
    else:
        bad_by_rank = [(0, bad)] if bad else []
    if bad_by_rank:
        detail = "; ".join(
            f"rank {rank}: {', '.join(values)}" for rank, values in bad_by_rank
        )
        raise RuntimeError(
            "BF16 model-only save requires every floating model parameter to "
            f"already be BF16; found {detail}"
        )


def _topology(args) -> Dict[str, int]:
    tp = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    pp = int(getattr(args, "pipeline_model_parallel_size", 1) or 1)
    ep = int(getattr(args, "moe_expert_parallel_size", 1) or 1)
    _, world = _rank_world()
    denominator = max(1, tp * pp * ep)
    if world % denominator:
        raise RuntimeError(
            "cannot publish model-only checkpoint with inconsistent topology: "
            f"world={world}, tp={tp}, pp={pp}, ep={ep}"
        )
    return {
        "world_size": world,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "expert_parallel_size": ep,
        "data_parallel_size": world // denominator,
    }


def _manifest(args, iteration: int, files: Iterable[Dict[str, object]]) -> Dict[str, object]:
    file_records = list(files)
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "complete": True,
        "iteration": int(iteration),
        "state_semantics": "live_post_optimizer_step",
        "parameter_dtype": "bfloat16",
        "optimizer_state": False,
        "fp32_master_weights": False,
        "rng_state": False,
        "consumed_train_tokens": int(getattr(args, "consumed_train_tokens", 0) or 0),
        "topology": _topology(args),
        "files": file_records,
        "total_bytes": sum(int(record["bytes"]) for record in file_records),
    }


def save_bf16_model_checkpoint(iteration: int, model, save_dir: str | None = None) -> str:
    """Save the final live model in DeepSpeed's native MoE shard layout.

    This is a collective operation.  It saves neither optimizer/momentum/
    variance tensors nor FP32 master weights and is intended for inference,
    not training resume.
    """
    args = get_args()
    target = save_dir or getattr(args, "save_bf16_model", None)
    if not target:
        raise ValueError("save_bf16_model_checkpoint requires an output directory")
    if not getattr(args, "deepspeed", False):
        raise RuntimeError("BF16 model-only save currently requires DeepSpeed")
    if not getattr(args, "bf16", False):
        raise RuntimeError("BF16 model-only save requires --bf16 training")
    if len(model) != 1 or not hasattr(model[0], "save_checkpoint"):
        raise RuntimeError("expected one initialized DeepSpeed engine")

    engine = model[0]
    _validate_bf16_parameters(engine)

    rank, _ = _rank_world()
    root = Path(target).expanduser().resolve()
    tag = f"global_step{int(iteration)}"
    final_tag_dir = root / tag

    preflight_error = None
    if rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            if final_tag_dir.exists():
                raise FileExistsError(
                    f"refusing to replace completed model checkpoint: {final_tag_dir}"
                )
        except Exception as exc:  # propagate rank-zero filesystem failures
            preflight_error = f"{type(exc).__name__}: {exc}"
    preflight_error = _broadcast_object(preflight_error)
    if preflight_error:
        raise RuntimeError(preflight_error)

    token = _broadcast_object(uuid.uuid4().hex if rank == 0 else None)
    temporary_root = root / f".incomplete-{tag}-{token}"
    client_state = {
        "args": args,
        "checkpoint_version": 3.0,
        "iteration": int(iteration),
        "tokens": int(getattr(args, "consumed_train_tokens", 0) or 0),
        "model_only_checkpoint": {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "state_semantics": "live_post_optimizer_step",
            "parameter_dtype": "bfloat16",
        },
    }

    # DeepSpeed's MoE writer already knows the exact PP/EP shard naming and
    # load format.  Remove optimizer references only for this terminal save
    # and restore the live engine even if serialization fails.
    original_optimizer = engine.optimizer
    original_save_zero = engine.save_zero_checkpoint
    original_save_non_zero = engine.save_non_zero_checkpoint
    try:
        engine.optimizer = None
        engine.save_zero_checkpoint = False
        with deepspeed_checkpoint_state_dict(
                engine, getattr(args, "no_pipeline_parallel", False)):
            engine.save_checkpoint(
                str(temporary_root),
                tag=tag,
                client_state=client_state,
                save_latest=False,
            )
    finally:
        engine.optimizer = original_optimizer
        engine.save_zero_checkpoint = original_save_zero
        engine.save_non_zero_checkpoint = original_save_non_zero

    temporary_tag_dir = temporary_root / tag
    sidecar_error = None
    if rank == 0:
        try:
            for path in temporary_tag_dir.glob("*optim_states.pt"):
                path.unlink()
            _write_sidecar_metadata(temporary_tag_dir, args)
        except Exception as exc:
            sidecar_error = f"{type(exc).__name__}: {exc}"
    sidecar_error = _broadcast_object(sidecar_error)
    if sidecar_error:
        raise RuntimeError(sidecar_error)
    _barrier()

    file_records = _distributed_file_records(temporary_tag_dir)
    finalize_error = None
    total_bytes = 0
    if rank == 0:
        try:
            manifest = _manifest(args, iteration, file_records)
            total_bytes = int(manifest["total_bytes"])
            _atomic_write_text(
                temporary_tag_dir / MANIFEST_NAME,
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
            _fsync_dir(temporary_tag_dir)
            os.replace(temporary_tag_dir, final_tag_dir)
            _fsync_dir(root)
            _atomic_write_text(root / "latest", tag)
            temporary_root.rmdir()
        except Exception as exc:
            finalize_error = f"{type(exc).__name__}: {exc}"
    finalize_error = _broadcast_object(finalize_error)
    if finalize_error:
        raise RuntimeError(finalize_error)
    _barrier()

    print_rank_0(
        f"saved BF16 model-only checkpoint at iteration {iteration} to "
        f"{final_tag_dir} ({int(total_bytes) / (1024 ** 3):.2f} GiB, checksummed)"
    )
    return str(final_tag_dir)
