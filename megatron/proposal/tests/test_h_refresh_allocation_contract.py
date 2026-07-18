"""Static guards for allocation-free H-refresh validation."""

from __future__ import annotations

import ast
from pathlib import Path

_PROPOSAL = Path(__file__).resolve().parents[1]
_H_REFRESH = _PROPOSAL / "refresh" / "hessian_refresh.py"
_ENGINE = _PROPOSAL / "checkpoint" / "lossy_checkpoint_engine.py"
_FP32_STATE = _PROPOSAL / "refresh" / "fp32_state.py"
_REFRESH_ORCHESTRATOR = _PROPOSAL / "refresh" / "refresh_orchestrator.py"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _call_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            names.append(func.id)
        elif isinstance(func, ast.Attribute):
            names.append(func.attr)
        elif isinstance(func, ast.Subscript) and isinstance(func.value, ast.Name):
            names.append(func.value.id)
    return names


def test_h_refresh_does_not_materialize_full_view_diagnostics() -> None:
    tree = ast.parse(_H_REFRESH.read_text())

    # Coverage is returned by the resident scan; diagnostics must not launch
    # one CUDA scalar reduction for every expert view.
    assert not {"abs", "mean", "isfinite"}.intersection(
        _call_names(_function(tree, "_fold_h"))
    )

    # The FD loop must not create a bool tensor for every resident grad view.
    assert "isfinite" not in _call_names(_function(tree, "_compute_fd_fast_flat"))

    # The removed scale-probe option converted each raw model-sized grad to
    # FP32 and abs() inside the hook.  Production hooks now only transform the
    # incoming grad in place.
    assert not {"float", "abs", "mean"}.intersection(
        _call_names(_function(tree, "_register_fd_grad_transform_hook"))
    )
    assert "PROPOSAL_H_REFRESH_SCALE_PROBE" not in _H_REFRESH.read_text()


def test_h_refresh_finite_check_reuses_init_time_scalar() -> None:
    h_tree = ast.parse(_H_REFRESH.read_text())
    finite_check = _function(h_tree, "_resident_views_are_finite")
    calls = _call_names(finite_check)
    assert "zero_" in calls
    assert "_mark_nonfinite_kernel" in calls
    assert "item" in calls
    assert not {"empty", "zeros", "isfinite", "reshape"}.intersection(calls)

    engine_text = _ENGINE.read_text()
    assert "self._h_refresh_nonfinite_flag = torch.zeros(" in engine_text
    assert '1, dtype=torch.int32, device=device' in engine_text
    assert "self._h_refresh_view_coverage = torch.zeros(" in engine_text
    assert "self._h_refresh_view_coverage_host = torch.empty(" in engine_text


def test_h_refresh_reuses_grad_storage_and_never_creates_param_grad() -> None:
    h_text = _H_REFRESH.read_text()
    assert "allow_create=False" in h_text

    fp32_tree = ast.parse(_FP32_STATE.read_text())
    grad_view = _function(fp32_tree, "get_grad_accum_view")
    source = ast.get_source_segment(_FP32_STATE.read_text(), grad_view)
    assert source is not None
    fail_closed = source.index("if not allow_create")
    allocating_fallback = source.index("param.grad = torch.zeros_like")
    assert fail_closed < allocating_fallback


def test_production_h_refresh_contains_no_explicit_cuda_factory() -> None:
    tree = ast.parse(_H_REFRESH.read_text())
    refresh_calls = _call_names(
        _function(tree, "_refresh_h_quantile_from_finite_diff")
    )
    assert not {
        "empty",
        "empty_like",
        "zeros",
        "zeros_like",
        "ones",
        "ones_like",
        "rand",
        "randn",
        "randint",
        "full",
        "clone",
    }.intersection(refresh_calls)

    # The only mapper fallback that can create param.grad remains available
    # for non-anchor callers; H refresh explicitly opts out above.
    fp32_tree = ast.parse(_FP32_STATE.read_text())
    grad_view = _function(fp32_tree, "get_grad_accum_view")
    assert "zeros_like" in _call_names(grad_view)


def test_pipeline_zero1_keeps_cross_pass_grad_and_ipg_storage() -> None:
    source = _H_REFRESH.read_text()
    tree = ast.parse(source)

    fd = _function(tree, "_compute_fd_fast_flat")
    fd_source = ast.get_source_segment(source, fd)
    assert fd_source is not None
    assert "if is_last_backward:" in fd_source
    assert "_clear_pipeline_grads()" in fd_source
    # Releasing ZeRO's IPG buffer here would make setup_buckets() allocate a
    # new reduce bucket on every +/- pass.  Final ReduceGrads owns its release.
    assert "_release_ipg_buffers" not in fd_source
    # In PipelineEngine + ZeRO-1, p.grad is the BF16 +/- accumulator.  The
    # generic post-pass clear is legal only outside the pipeline path.
    assert "preserve_param_grad_storage = pipeline_engine is not None or any(" in fd_source


def test_zero1_anchor_cleanup_drops_every_cuda_owning_reference() -> None:
    """The BF16 accumulator may live across passes, never across anchors."""

    fp32_source = _FP32_STATE.read_text()
    fp32_tree = ast.parse(fp32_source)
    discard_source = ast.get_source_segment(
        fp32_source,
        _function(fp32_tree, "discard_zero12_grad_state"),
    )
    assert discard_source is not None
    for table_name in ("averaged_gradients", "all_grad_tensors"):
        assert f'"{table_name}"' in discard_source
    assert "table[key] = None" in discard_source
    assert "table[index] = None" in discard_source
    assert "param.grad = None" in discard_source
    assert "param.grad_accum = None" in discard_source

    h_source = _H_REFRESH.read_text()
    refresh_source = ast.get_source_segment(
        h_source,
        _function(ast.parse(h_source), "_refresh_h_quantile_from_finite_diff"),
    )
    assert refresh_source is not None
    # DeepSpeed tables are cleared both at the inner FD boundary and again in
    # the outer finally; local views are dropped before training can resume.
    assert refresh_source.count("discard_zero12_grad_state(_ds_engine_opt)") >= 2
    for local_name in ("_grad_accum_views", "_final_views", "_view", "_h"):
        assert f"{local_name} = None" in refresh_source
    # Anchor scratch must be discarded before the exact pre-anchor gradients
    # are restored.  A later discard would silently erase restored training
    # state when the anchor runs without OptimizerStep.
    last_discard = refresh_source.rfind(
        "discard_zero12_grad_state(_ds_engine_opt)"
    )
    restore = refresh_source.rfind(
        'restore_training_state(_config.get("_inner_opt"), _train_state_snap)'
    )
    assert 0 <= last_discard < restore
    assert "discard_zero12_grad_state(_ds_engine_opt)" not in refresh_source[restore:]
    assert "empty_cache" not in _call_names(
        _function(ast.parse(h_source), "_refresh_h_quantile_from_finite_diff")
    )
