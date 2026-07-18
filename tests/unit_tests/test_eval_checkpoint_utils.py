import pytest
import torch
import json

from tasks.eval_harness.checkpoint_utils import (
    concat_pipeline_logits,
    deepspeed_model_only_eval_config,
    prepare_bf16_eval_module,
    resolve_deepspeed_checkpoint_dir,
)


def test_bf16_eval_cast_covers_plain_parameters_and_floating_buffers():
    module = torch.nn.Module()
    module.embedding = torch.nn.Embedding(8, 4)
    module.register_buffer("floating", torch.ones(3, dtype=torch.float32))
    module.register_buffer("integer", torch.ones(3, dtype=torch.int64))

    prepare_bf16_eval_module(module)

    assert module.embedding.weight.dtype == torch.bfloat16
    assert module.floating.dtype == torch.bfloat16
    assert module.integer.dtype == torch.int64


def test_pipeline_logits_are_converted_from_seq_first_to_batch_first():
    first = torch.arange(3 * 2 * 5).reshape(3, 2, 5)
    second = 100 + torch.arange(3 * 2 * 5).reshape(3, 2, 5)

    actual = concat_pipeline_logits([first, second], expected_batch_size=3)

    expected = torch.cat([first.transpose(0, 1), second.transpose(0, 1)], dim=0)[:3]
    assert actual.shape == (3, 3, 5)
    assert torch.equal(actual, expected)


def test_pipeline_logits_reject_invalid_or_short_outputs():
    with pytest.raises(RuntimeError, match="no evaluation logits"):
        concat_pipeline_logits([], expected_batch_size=1)
    with pytest.raises(RuntimeError, match="seq-first"):
        concat_pipeline_logits([torch.zeros(2, 3)], expected_batch_size=1)
    with pytest.raises(RuntimeError, match="returned 1 rows"):
        concat_pipeline_logits([torch.zeros(2, 1, 3)], expected_batch_size=2)


def test_model_only_resolver_rejects_wrong_evaluator_topology(tmp_path):
    tag = tmp_path / "global_step8"
    tag.mkdir()
    (tag / "mp_rank_00_model_states.pt").write_bytes(b"model")
    manifest = {
        "format": "deepspeed_moe_bf16_model_only",
        "format_version": 1,
        "complete": True,
        "parameter_dtype": "bfloat16",
        "optimizer_state": False,
        "fp32_master_weights": False,
        "topology": {
            "world_size": 2,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 2,
            "data_parallel_size": 1,
        },
    }
    (tag / "model_only_manifest.json").write_text(json.dumps(manifest))

    assert resolve_deepspeed_checkpoint_dir(
        str(tag),
        require_model_only=True,
        expected_topology=manifest["topology"],
    ) == str(tag.resolve())
    wrong = dict(manifest["topology"], pipeline_parallel_size=2)
    with pytest.raises(RuntimeError, match="does not match evaluator topology"):
        resolve_deepspeed_checkpoint_dir(
            str(tag), require_model_only=True, expected_topology=wrong
        )


def test_model_only_eval_config_disables_training_optimizer_wrappers():
    training = {
        "train_batch_size": 256,
        "train_micro_batch_size_per_gpu": 8,
        "zero_optimization": {"stage": 1, "overlap_comm": True},
        "bf16": {"enabled": True},
        "fp16": {"enabled": False, "loss_scale": 0},
        "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    }

    evaluation = deepspeed_model_only_eval_config(
        training,
        micro_batch_size=2,
        data_parallel_size=1,
    )

    assert evaluation["zero_optimization"] == {"stage": 0}
    assert evaluation["bf16"] == {"enabled": False}
    assert evaluation["fp16"] == {"enabled": False}
    assert evaluation["gradient_accumulation_steps"] == 1
    assert evaluation["train_micro_batch_size_per_gpu"] == 2
    assert evaluation["train_batch_size"] == 2
    assert "optimizer" not in evaluation
    assert training["zero_optimization"]["stage"] == 1
    assert training["bf16"]["enabled"] is True
