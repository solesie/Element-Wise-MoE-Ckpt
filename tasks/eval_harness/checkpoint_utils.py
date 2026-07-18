"""Small, dependency-free helpers for locating DeepSpeed eval checkpoints."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch


MODEL_ONLY_MANIFEST = "model_only_manifest.json"
MODEL_ONLY_FORMAT = "deepspeed_moe_bf16_model_only"
MODEL_ONLY_FORMAT_VERSION = 1


def prepare_bf16_eval_module(module: torch.nn.Module) -> None:
    """Mirror DeepSpeed's BF16 module cast without its optimizer wrapper."""

    module.bfloat16()
    bad = []
    for kind, values in (
        ("parameter", module.named_parameters()),
        ("buffer", module.named_buffers()),
    ):
        for name, value in values:
            if value.is_floating_point() and value.dtype != torch.bfloat16:
                bad.append(f"{kind} {name}:{value.dtype}")
                if len(bad) >= 8:
                    break
        if len(bad) >= 8:
            break
    if bad:
        raise RuntimeError(
            "BF16 evaluator module contains non-BF16 floating state: "
            + ", ".join(bad)
        )


def concat_pipeline_logits(outputs, expected_batch_size: int) -> torch.Tensor:
    """Convert PipelineEngine's seq-first microbatch outputs to batch-first."""

    if not outputs:
        raise RuntimeError("PipelineEngine returned no evaluation logits")

    batch_first = []
    for index, output in enumerate(outputs):
        if not torch.is_tensor(output) or output.ndim != 3:
            raise RuntimeError(
                "expected seq-first [seq, batch, vocab] pipeline logits, "
                f"got output {index} with type/shape "
                f"{type(output).__name__}/{getattr(output, 'shape', None)}"
            )
        batch_first.append(output.transpose(0, 1).contiguous())

    logits = torch.cat(batch_first, dim=0)
    expected = int(expected_batch_size)
    if expected < 0 or logits.size(0) < expected:
        raise RuntimeError(
            f"pipeline returned {logits.size(0)} rows for {expected} inputs"
        )
    return logits[:expected]


def deepspeed_model_only_eval_config(
    training_config: dict,
    *,
    micro_batch_size: int,
    data_parallel_size: int,
) -> dict:
    """Build a no-optimizer DeepSpeed config for loading BF16 model shards.

    Model dtype is owned by Megatron's checkpoint arguments.  Leaving the
    training config's ZeRO-1 or DeepSpeed BF16 optimizer mode enabled makes an
    inference Engine create a DummyOptim wrapper and fail before model load.
    """

    config = copy.deepcopy(training_config)
    micro = max(1, int(micro_batch_size))
    dp = max(1, int(data_parallel_size))
    config.pop("optimizer", None)
    config.pop("scheduler", None)
    config.pop("torch_autocast", None)
    config["zero_optimization"] = {"stage": 0}
    config["fp16"] = {"enabled": False}
    config["bf16"] = {"enabled": False}
    config["train_micro_batch_size_per_gpu"] = micro
    config["gradient_accumulation_steps"] = 1
    config["train_batch_size"] = micro * dp
    return config


def _model_state_files(path: Path):
    return sorted(path.glob("mp_rank_*_model_states.pt"))


def resolve_deepspeed_checkpoint_dir(
    path: str,
    *,
    require_model_only: bool = False,
    expected_topology: dict[str, int] | None = None,
) -> str:
    """Accept either a DeepSpeed checkpoint root or its concrete tag dir."""
    checkpoint = Path(path).expanduser().resolve()
    if _model_state_files(checkpoint):
        tag_dir = checkpoint
    else:
        latest = checkpoint / "latest"
        if not latest.is_file():
            raise FileNotFoundError(
                f"no mp_rank model shard or DeepSpeed latest pointer under {checkpoint}"
            )
        tag = latest.read_text(encoding="utf-8").strip()
        if not tag or "/" in tag or tag in {".", ".."}:
            raise RuntimeError(f"invalid DeepSpeed latest tag {tag!r} in {latest}")
        tag_dir = checkpoint / tag
        if not _model_state_files(tag_dir):
            raise FileNotFoundError(f"no mp_rank model shard under {tag_dir}")

    manifest_path = tag_dir / MODEL_ONLY_MANIFEST
    if not manifest_path.is_file():
        if require_model_only:
            raise FileNotFoundError(f"no model-only manifest: {manifest_path}")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            raise RuntimeError(f"incomplete model-only checkpoint: {manifest_path}")
        if manifest.get("optimizer_state") is not False:
            raise RuntimeError(f"invalid model-only manifest: {manifest_path}")
        required = {
            "format": MODEL_ONLY_FORMAT,
            "format_version": MODEL_ONLY_FORMAT_VERSION,
            "parameter_dtype": "bfloat16",
            "fp32_master_weights": False,
        }
        for key, value in required.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    f"invalid model-only manifest {key}: "
                    f"{manifest.get(key)!r} != {value!r}"
                )
        if expected_topology is not None:
            actual = manifest.get("topology")
            expected = {key: int(value) for key, value in expected_topology.items()}
            if actual != expected:
                raise RuntimeError(
                    f"checkpoint topology {actual!r} does not match evaluator "
                    f"topology {expected!r}"
                )
    return str(tag_dir)


def load_deepspeed_checkpoint_args(tag_dir: str):
    """Read Megatron arguments without invoking the legacy DS converter."""
    checkpoint = Path(resolve_deepspeed_checkpoint_dir(tag_dir))
    model_files = _model_state_files(checkpoint)
    try:
        state = torch.load(model_files[0], map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        state = torch.load(model_files[0], map_location="cpu")
    args = state.get("args")
    if args is None:
        raise RuntimeError(f"checkpoint does not contain Megatron args: {model_files[0]}")
    return args
