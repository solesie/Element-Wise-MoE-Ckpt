"""GPU-free checks for delayed proposal activation after LR warmup."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from megatron.proposal.config import checkpoint_warmup_iters_from_args


_ROOT = Path(__file__).resolve().parents[2]


def _function(relative: str, name: str, class_name: str | None = None) -> ast.FunctionDef:
    tree = ast.parse((_ROOT / relative).read_text(encoding="utf-8"))
    scope: list[ast.stmt] = tree.body
    if class_name is not None:
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        scope = cls.body
    return next(
        node for node in scope
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _source(relative: str, name: str, class_name: str | None = None) -> str:
    path = _ROOT / relative
    return ast.get_source_segment(
        path.read_text(encoding="utf-8"),
        _function(relative, name, class_name),
    ) or ""


def test_refresh_cadence_starts_at_warmup_then_keeps_absolute_period() -> None:
    fn = _function("training.py", "_proposal_h_refresh_due")
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "checkpoint_warmup_iters_from_args": lambda args: int(args.warmup),
    }
    exec(compile(module, "training.py", "exec"), namespace)
    due = namespace["_proposal_h_refresh_due"]
    args = SimpleNamespace(
        proposal_hessian_refresh_every=100,
        warmup=80,
    )
    assert [step for step in range(1, 301) if due(args, step)] == [80, 100, 200, 300]


def test_activation_uses_existing_exact_baseline_paths() -> None:
    init = _source("proposal/controller.py", "init_proposal")
    assert "defer_initial_expert_bootstrap" in init

    activation = _source("proposal/controller.py", "proposal_activate_after_warmup")
    assert "bootstrap_expert_recovery_baseline" in activation
    assert "bootstrap_nonexpert_recovery_baseline" in activation
    assert "reset_checkpoint_score_state_from_live" in activation
    assert "ctrl._active = True" in activation

    dispatch = _source("proposal/controller.py", "proposal_dispatch_post_iter")
    assert "not get_controller()._active" in dispatch
    assert "iter_T <= int(get_controller()._warmup_end_iter)" in dispatch

    train = _source("training.py", "train")
    assert "iteration > checkpoint_warmup_end" in train
    assert "iteration == checkpoint_warmup_end" in train
    assert "iteration > lr_warmup_end" in train


def test_sequential_moc_decouples_lr_warmup_from_checkpoint_cold_start() -> None:
    common = {
        "lr_warmup_tokens": None,
        "lr_warmup_iters": 80,
        "lr_warmup_samples": 0,
        "lr_warmup_fraction": None,
    }
    assert checkpoint_warmup_iters_from_args(
        SimpleNamespace(proposal_cp_mode="sequential", **common),
    ) == 0
    assert checkpoint_warmup_iters_from_args(
        SimpleNamespace(proposal_cp_mode="proposal", **common),
    ) == 80
    assert checkpoint_warmup_iters_from_args(
        SimpleNamespace(
            proposal_cp_mode="proposal",
            proposal_checkpoint_warmup_iters=10,
            **common,
        ),
    ) == 10

    init = _source("proposal/controller.py", "init_proposal")
    assert "checkpoint_warmup_iters_from_args(args)" in init

    train = _source("training.py", "train")
    assert "checkpoint_warmup_end =" in train
    assert "checkpoint_warmup_iters_from_args(args)" in train
    # Sequential dispatch is gated by checkpoint policy (zero warmup), while
    # fault injection remains gated by the optimizer's real LR warmup.
    assert "iteration > checkpoint_warmup_end" in train
    assert "iteration > lr_warmup_end" in train


def test_h_refresh_uses_checkpoint_bootstrap_boundary() -> None:
    fn = _function("training.py", "_proposal_h_refresh_due")
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "checkpoint_warmup_iters_from_args": lambda args: int(args.cp_warmup),
    }
    exec(compile(module, "training.py", "exec"), namespace)
    due = namespace["_proposal_h_refresh_due"]
    args = SimpleNamespace(
        proposal_hessian_refresh_every=100,
        cp_warmup=10,
    )
    assert [step for step in range(1, 301) if due(args, step)] == [10, 100, 200, 300]


def test_activation_runs_baselines_once_at_exact_boundary() -> None:
    calls: list[object] = []

    class _Event:
        def record(self, _stream) -> None:
            calls.append("record")

    class _Workers:
        def drain(self) -> None:
            calls.append("drain")

    class _FC:
        def bootstrap_expert_recovery_baseline(self, *, step: int) -> int:
            calls.append(("expert", step))
            return 2

        def bootstrap_nonexpert_recovery_baseline(self, *, step: int, ready_event) -> bool:
            calls.append(("nonexpert", step, ready_event is event))
            return True

    class _Lossy:
        _flat_layout_done = False

        def _init_lossy_flat_layout(self, optimizer) -> None:
            calls.append(("init_layout", optimizer))
            self._flat_layout_done = True

        def reset_checkpoint_score_state_from_live(
            self, *, next_global_step: int = 1,
        ) -> None:
            calls.append(("reset", next_global_step))

    event = _Event()
    ctrl = SimpleNamespace(
        _enabled=True,
        _active=False,
        _warmup_end_iter=80,
        _checkpoint_workers=_Workers(),
        _fc=_FC(),
        _lossy_cp=_Lossy(),
        _config={"_inner_opt": "optimizer"},
        acquire_dispatch_ready_event=lambda: event,
    )
    fn = _function("proposal/controller.py", "proposal_activate_after_warmup")
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "get_controller": lambda: ctrl,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(current_stream=lambda: object()),
        ),
        "os": SimpleNamespace(environ={}),
        "_should_write": lambda: False,
    }
    exec(compile(module, "controller.py", "exec"), namespace)
    activate = namespace["proposal_activate_after_warmup"]
    assert activate(object(), 79) is False
    assert activate(object(), 80) is True
    assert calls == [
        "drain",
        ("init_layout", "optimizer"),
        "record",
        ("expert", 80),
        ("nonexpert", 80, True),
        "drain",
        ("reset", 81),
    ]
    assert ctrl._active is True
    assert activate(object(), 80) is False


def test_cutoff_cold_start_is_activation_relative() -> None:
    cutoff = _source(
        "proposal/cutoff/cutoff_orchestrator.py",
        "_find_selection_cutoff_for_current_iter",
        "CutoffOrchestrationMixin",
    )
    assert 'getattr(self, "_cold_start_first_global_step", 1)' in cutoff
    assert "cold_start_first_step + cold_start_hist_iters" in cutoff


def test_adam_source_cache_retries_until_optimizer_state_exists() -> None:
    fn = _function(
        "proposal/storage/fc_manager.py",
        "_warm_lfcl_adam_source_cache_once",
        "FCManager",
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, "fc_manager.py", "exec"), namespace)
    warm = namespace["_warm_lfcl_adam_source_cache_once"]

    class _FC:
        _lfcl_adam_source_cache_warmed = False
        _lfcl_field_cache_by_expert = {(0, 0): [{}]}
        ready = False
        calls = 0

        def _lfcl_cached_state_source(self, _field, _key):
            self.calls += 1
            return object() if self.ready else None

    fc = _FC()
    assert warm(fc) is False
    assert fc._lfcl_adam_source_cache_warmed is False
    fc.ready = True
    assert warm(fc) is True
    assert fc._lfcl_adam_source_cache_warmed is True
    calls = fc.calls
    assert warm(fc) is True
    assert fc.calls == calls

    bootstrap = _source(
        "proposal/storage/fc_manager.py",
        "bootstrap_expert_recovery_baseline",
        "FCManager",
    )
    assert "int(step) > 0 and not adam_sources_ready" in bootstrap


if __name__ == "__main__":
    test_refresh_cadence_starts_at_warmup_then_keeps_absolute_period()
    test_activation_uses_existing_exact_baseline_paths()
    test_activation_runs_baselines_once_at_exact_boundary()
    test_cutoff_cold_start_is_activation_relative()
    test_adam_source_cache_retries_until_optimizer_state_exists()
    print("warmup activation contract: PASS")
