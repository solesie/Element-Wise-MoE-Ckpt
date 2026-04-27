"""Unit tests for recovery.py file discovery + param classification.

Verifies that the recovery module correctly parses checkpoint filenames
and routes params to the expected expert / non-expert buckets. The
heavyweight ``recover()`` path (DC replay via fused AdamW) requires a
real training-ready model + optimizer and is exercised via integration
rather than unit tests.
"""
from pathlib import Path

import pytest
import torch

from megatron.proposal.recovery import (
    _classify_params,
    _ids_from_param_name,
    _scan_cp_dir,
)


def test_ids_from_param_name_parses_ds_expert_names():
    # DeepSpeed separated-expert naming: ``...deepspeed_experts.{E}...``
    # and layer index in ``...layers.{L}...``.
    assert _ids_from_param_name(
        "module.language_model.transformer.layers.3.mlp.deepspeed_experts.5.dense_h_to_4h.weight"
    ) == (3, 5)
    assert _ids_from_param_name(
        "encoder.layers.0.mlp.deepspeed_experts.0.w.weight"
    ) == (0, 0)


def test_ids_from_param_name_returns_none_for_non_expert():
    # No deepspeed_experts token → not an expert.
    assert _ids_from_param_name("encoder.embedding.weight") is None
    assert _ids_from_param_name("attn.query.weight") is None


def test_ids_from_param_name_defaults_layer_zero_when_missing():
    # ``deepspeed_experts`` token present but no ``layers.N`` → layer=0.
    assert _ids_from_param_name("mlp.deepspeed_experts.7.weight") == (0, 7)


def test_scan_cp_dir_recognises_filename_patterns(tmp_path: Path):
    (tmp_path / "non_expert").mkdir()
    (tmp_path / "expert").mkdir()

    # Create CP files with all supported names.
    (tmp_path / "non_expert" / "fc_nonexpert_iter100.pt").touch()
    (tmp_path / "non_expert" / "dc_iter101_dense_ne.pt").touch()
    (tmp_path / "non_expert" / "dc_iter102_dense_ne.pt").touch()
    (tmp_path / "expert" / "fc_expert_L03_E05_iter100.pt").touch()  # legacy per-iter FC
    # New flat per-expert layout: one lossless + one lossy FC file per
    # (L, E), both living directly under <cp>/expert/.
    (tmp_path / "expert" / "fc_expert_L00_E00.bin").touch()
    (tmp_path / "expert" / "lossy_expert_L00_E00.bin").touch()
    (tmp_path / "expert" / "fc_expert_L00_E01.bin").touch()  # only lossless
    # Decoys that must NOT be classified as CP entries.
    (tmp_path / "non_expert" / "README.txt").touch()
    (tmp_path / "expert" / "temp_unknown.bin").touch()

    expert_fcs, ne_fcs, dc_bundles, expert_pairs = _scan_cp_dir(tmp_path)

    # Non-expert FC → one entry at iter 100.
    assert len(ne_fcs) == 1
    assert ne_fcs[0][0] == 100
    # DC bundles → two, iters 101 and 102.
    assert sorted(i for i, _ in dc_bundles) == [101, 102]
    # Legacy expert FC → (L=3, E=5) at iter 100.
    assert (3, 5) in expert_fcs
    assert expert_fcs[(3, 5)][0][0] == 100
    # New per-expert pairs → (0, 0) has both, (0, 1) has only lossless.
    assert set(expert_pairs.keys()) == {(0, 0), (0, 1)}
    lossless_00, lossy_00 = expert_pairs[(0, 0)]
    assert lossless_00.name == "fc_expert_L00_E00.bin"
    assert lossy_00.name == "lossy_expert_L00_E00.bin"


def test_scan_cp_dir_accepts_legacy_layout(tmp_path: Path):
    """Old runs used ``<cp>/fc/`` + ``<cp>/dc/`` instead of split
    non_expert/ expert/. Scanner must also pick those up."""
    (tmp_path / "fc").mkdir()
    (tmp_path / "dc").mkdir()
    (tmp_path / "fc" / "fc_nonexpert_iter50.pt").touch()
    (tmp_path / "dc" / "dc_iter51_dense_ne.pt").touch()
    (tmp_path / "fc" / "fc_expert_L01_E02_iter50.pt").touch()

    expert_fcs, ne_fcs, dc_bundles, expert_pairs = _scan_cp_dir(tmp_path)
    assert ne_fcs and ne_fcs[0][0] == 50
    assert dc_bundles and dc_bundles[0][0] == 51
    assert (1, 2) in expert_fcs


def test_scan_cp_dir_ignores_missing_dir(tmp_path: Path):
    # tmp_path has no non_expert/ or expert/ subdir — scanner should
    # return empty collections, not raise.
    expert_fcs, ne_fcs, dc_bundles, expert_pairs = _scan_cp_dir(tmp_path)
    assert expert_fcs == {}
    assert ne_fcs == []
    assert dc_bundles == []
    assert expert_pairs == {}


class _ExpertParam(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden, hidden))


class _Layer(torch.nn.Module):
    """Uses ``deepspeed_experts = ModuleList([...])`` so named_parameters
    produces ``layers.{L}.deepspeed_experts.{E}.weight`` — matching the
    ``deepspeed_experts.{idx}.`` substring that _ids_from_param_name keys on."""
    def __init__(self, hidden: int, n: int):
        super().__init__()
        self.deepspeed_experts = torch.nn.ModuleList(
            [_ExpertParam(hidden) for _ in range(n)]
        )


def test_classify_params_splits_expert_vs_non_expert():
    # Outer wrapper so named_parameters produces
    # ``inner.layers.{L}.deepspeed_experts.{E}.weight`` — the ``.layers.N.``
    # substring regex requires a dot before the ``layers`` token.
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Parameter(torch.randn(4))
            inner = torch.nn.Module()
            inner.layers = torch.nn.ModuleList([_Layer(4, 3) for _ in range(2)])
            self.inner = inner

    non_expert, by_expert = _classify_params(_M())
    # embedding is non-expert.
    assert any("embedding" in n for n in non_expert)
    # 2 layers × 3 experts = 6 expert buckets.
    assert len(by_expert) == 6, f"got {list(by_expert.keys())}"
    for (L, E), tensors in by_expert.items():
        assert 0 <= L < 2 and 0 <= E < 3
        assert any(name.endswith("weight") for name in tensors)
