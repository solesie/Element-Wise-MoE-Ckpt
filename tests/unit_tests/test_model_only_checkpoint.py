from argparse import Namespace
from pathlib import Path

import torch

from megatron import model_only_checkpoint as model_checkpoint
from tasks.eval_harness.checkpoint_utils import (
    load_deepspeed_checkpoint_args,
    resolve_deepspeed_checkpoint_dir,
)


class _FakeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return {
            "model": super().state_dict(prefix=prefix, keep_vars=keep_vars)
        }


class _FakeDeepSpeedEngine:
    def __init__(self):
        self.module = _FakeModule()
        self.optimizer = object()
        self.lr_scheduler = object()
        self.save_zero_checkpoint = True
        self.save_non_zero_checkpoint = True
        self.observed_model_only_state = False
        self.observed_scheduler = None

    def save_checkpoint(self, save_dir, tag, client_state, save_latest):
        assert save_latest is False
        self.observed_model_only_state = (
            self.optimizer is None
            and self.save_zero_checkpoint is False
        )
        self.observed_scheduler = self.lr_scheduler
        # DeepSpeed's real MoE branch makes this mutation internally.
        self.save_non_zero_checkpoint = False
        output = Path(save_dir) / tag
        output.mkdir(parents=True)
        torch.save(
            {"module": self.module.state_dict(), **client_state},
            output / "mp_rank_00_model_states.pt",
        )
        # The real MoE writer emits this empty compatibility file. The
        # model-only finalizer must remove it before publishing the tag.
        torch.save(
            {"optimizer": None},
            output / "expp_rank_0_mp_rank_00_optim_states.pt",
        )


def _args(tmp_path):
    vocab = tmp_path / "vocab.json"
    merges = tmp_path / "merges.txt"
    vocab.write_text('{"x": 0}\n', encoding="utf-8")
    merges.write_text("#version: 0.2\n", encoding="utf-8")
    return Namespace(
        deepspeed=True,
        no_pipeline_parallel=True,
        bf16=True,
        vocab_file=str(vocab),
        merge_file=str(merges),
        consumed_train_tokens=123,
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        seq_length=16,
        max_position_embeddings=16,
        padded_vocab_size=32,
        num_experts=4,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        moe_expert_parallel_size=1,
        deepspeed_config_dict={"bf16": {"enabled": True}},
        save_bf16_model=str(tmp_path / "published"),
    )


def test_model_only_checkpoint_is_atomic_loadable_and_restores_engine(monkeypatch, tmp_path):
    args = _args(tmp_path)
    engine = _FakeDeepSpeedEngine()
    optimizer = engine.optimizer
    scheduler = engine.lr_scheduler
    monkeypatch.setattr(model_checkpoint, "get_args", lambda: args)

    tag_dir = Path(model_checkpoint.save_bf16_model_checkpoint(10000, [engine]))

    assert engine.observed_model_only_state
    assert engine.observed_scheduler is scheduler
    assert engine.optimizer is optimizer
    assert engine.lr_scheduler is scheduler
    assert engine.save_zero_checkpoint is True
    assert engine.save_non_zero_checkpoint is True
    assert tag_dir.name == "global_step10000"
    assert (tag_dir.parent / "latest").read_text() == "global_step10000"
    assert not list(tag_dir.glob("*optim_states.pt"))
    assert not list(tag_dir.parent.glob(".incomplete-*"))
    assert (tag_dir / "tokenizer" / "gpt2-vocab.json").is_file()
    assert (tag_dir / "tokenizer" / "gpt2-merges.txt").is_file()

    manifest = __import__("json").loads(
        (tag_dir / model_checkpoint.MANIFEST_NAME).read_text()
    )
    assert manifest["complete"] is True
    assert manifest["state_semantics"] == "live_post_optimizer_step"
    assert manifest["optimizer_state"] is False
    assert manifest["fp32_master_weights"] is False
    assert all(len(record["sha256"]) == 64 for record in manifest["files"])
    for record in manifest["files"]:
        assert model_checkpoint.sha256_file(tag_dir / record["path"]) == record["sha256"]

    assert resolve_deepspeed_checkpoint_dir(str(tag_dir.parent)) == str(tag_dir)
    assert resolve_deepspeed_checkpoint_dir(str(tag_dir)) == str(tag_dir)
    loaded_args = load_deepspeed_checkpoint_args(str(tag_dir.parent))
    assert loaded_args.num_layers == args.num_layers
    state = torch.load(
        tag_dir / "mp_rank_00_model_states.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert state["module"]["model"]["weight"].dtype == torch.bfloat16


def test_model_only_checkpoint_rejects_non_bf16_parameter(monkeypatch, tmp_path):
    args = _args(tmp_path)
    engine = _FakeDeepSpeedEngine()
    engine.module.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    monkeypatch.setattr(model_checkpoint, "get_args", lambda: args)

    try:
        model_checkpoint.save_bf16_model_checkpoint(1, [engine])
    except RuntimeError as error:
        assert "every floating model parameter" in str(error)
    else:
        raise AssertionError("non-BF16 parameter was accepted")


def test_pipeline_model_keeps_deepspeed_native_state_dict(monkeypatch, tmp_path):
    args = _args(tmp_path)
    args.no_pipeline_parallel = False
    engine = _FakeDeepSpeedEngine()
    monkeypatch.setattr(model_checkpoint, "get_args", lambda: args)

    tag_dir = Path(model_checkpoint.save_bf16_model_checkpoint(1, [engine]))
    state = torch.load(
        tag_dir / "mp_rank_00_model_states.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert state["module"]["weight"].dtype == torch.bfloat16
    assert "model" not in state["module"]


def test_model_only_checkpoint_rejects_inconsistent_topology(monkeypatch, tmp_path):
    args = _args(tmp_path)
    args.pipeline_model_parallel_size = 2
    args.moe_expert_parallel_size = 2
    monkeypatch.setattr(model_checkpoint, "_rank_world", lambda: (0, 2))

    try:
        model_checkpoint._topology(args)
    except RuntimeError as error:
        assert "inconsistent topology" in str(error)
    else:
        raise AssertionError("inconsistent model-only topology was accepted")
