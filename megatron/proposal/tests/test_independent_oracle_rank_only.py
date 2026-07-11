from pathlib import Path
from types import SimpleNamespace

import numpy as np

from megatron.proposal.telemetry import independent_oracle as oracle
from megatron.proposal.refresh.fp32_state import discard_zero12_grad_state


def test_discard_zero12_grad_state_releases_tables_and_param_grads():
    import torch

    param = torch.nn.Parameter(torch.ones(4))
    param.grad = torch.ones_like(param)
    param.grad_accum = torch.ones_like(param)
    optimizer = SimpleNamespace(
        averaged_gradients={0: [torch.ones(4)]},
        all_grad_tensors=[torch.ones(4)],
        bit16_groups=[[param]],
    )

    discard_zero12_grad_state(optimizer)

    assert optimizer.averaged_gradients == {0: None}
    assert optimizer.all_grad_tensors == [None]
    assert param.grad is None
    assert param.grad_accum is None


def test_rank_only_outer_finally_discards_hvp_grad_state(monkeypatch, tmp_path: Path):
    discarded = []

    class Snapshot:
        def restore(self):
            pass

        def max_abs_diff(self):
            return 0.0

    snapshot = Snapshot()
    engine_optimizer = object()
    lossy_cp = SimpleNamespace(_groups=[{}])
    config = SimpleNamespace(
        independent_oracle_enable=True,
        independent_oracle_iters=(50,),
        independent_oracle_log_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        oracle,
        "_build_forward_loss_bank",
        lambda **_kwargs: [(object(), {})],
    )
    monkeypatch.setattr(
        oracle._LiveParamSnapshot,
        "capture",
        classmethod(lambda _cls, _cp: snapshot),
    )
    monkeypatch.setattr(oracle, "_run_independent_rank_only", lambda **_kwargs: None)
    monkeypatch.setattr(oracle, "get_engine_optimizer", lambda _model: engine_optimizer)
    monkeypatch.setattr(
        oracle,
        "discard_zero12_grad_state",
        lambda optimizer: discarded.append(optimizer),
    )

    oracle.run_independent_oracle(
        model=object(),
        data_iterator=object(),
        iteration=50,
        lossy_cp=lossy_cp,
        proposal_config=config,
    )

    assert discarded == [engine_optimizer]
    assert not hasattr(lossy_cp, oracle._SNAPSHOT_ATTR)


def test_rank_only_oracle_writes_only_rank_rows(monkeypatch, tmp_path: Path):
    writes = []
    payload = {"e": np.ones(8, dtype=np.float32)}

    monkeypatch.setattr(oracle, "_global_error_rms", lambda _cp: 1.0)
    monkeypatch.setattr(
        oracle,
        "_build_sample",
        lambda _cp, _config, rank: {"n": 8, "items": []},
    )
    monkeypatch.setattr(
        oracle,
        "_gather_sample_payload",
        lambda _cp, _sample, iteration: payload,
    )
    monkeypatch.setattr(
        oracle,
        "_hvp_for_current_error",
        lambda **_kwargs: (0.0, np.ones(8, dtype=np.float32)),
    )
    monkeypatch.setattr(
        oracle,
        "_rank_metric_rows",
        lambda **_kwargs: [
            {"mode": "independent_oracle_rank", "ndcg@0.04": 0.5},
            {"mode": "independent_oracle_ehat_diag"},
        ],
    )
    monkeypatch.setattr(
        oracle,
        "_append_jsonl",
        lambda path, row: writes.append((Path(path).name, row)),
    )

    oracle._run_independent_rank_only(
        model=object(),
        loss_bank=[(object(), {})],
        iteration=50,
        lossy_cp=object(),
        proposal_config=SimpleNamespace(
            finite_diff_eps=1.0e-3,
            topk_ratio=0.04,
        ),
        rank=0,
        world=8,
        log_dir=tmp_path,
        started_at=0.0,
        requested_hvp_batches=1,
        pipeline_engine=True,
    )

    rank_writes = [row for name, row in writes if name.startswith("independent_rank_")]
    timing_writes = [row for name, row in writes if name.startswith("independent_timing_")]
    assert rank_writes == [{"mode": "independent_oracle_rank", "ndcg@0.04": 0.5}]
    assert len(timing_writes) == 1
    assert timing_writes[0]["rank_only"] is True
    assert timing_writes[0]["actual_batches"] == 0
    assert timing_writes[0]["hvp_batches"] == 1


def test_sample_only_hvp_skips_full_quadratic_collective(monkeypatch):
    calls = []
    monkeypatch.setattr(oracle, "_restore_current_params", lambda _cp: None)
    monkeypatch.setattr(oracle, "_set_params_perturbed", lambda _cp, scale: None)
    monkeypatch.setattr(
        oracle,
        "_backward_dot_and_sample",
        lambda **kwargs: (
            calls.append(kwargs["compute_full_dot"]) or 123.0,
            np.ones(4, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        oracle,
        "_all_reduce_sum_float",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sample-only HVP must not reduce the full quadratic")
        ),
    )

    r2, sample = oracle._hvp_for_current_error(
        lossy_cp=object(),
        model=object(),
        loss_bank=[(object(), {})],
        fd_eps=1.0e-3,
        rms_e=1.0,
        sample={"n": 4},
        compute_full_quadratic=False,
    )

    assert r2 == 0.0
    assert np.array_equal(sample, np.zeros(4, dtype=np.float32))
    assert calls == [False, False]
