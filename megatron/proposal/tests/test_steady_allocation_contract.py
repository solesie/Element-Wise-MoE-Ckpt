"""GPU-free structural checks for the steady proposal allocation contract.

The first lossy dispatch may build immutable optimizer/AOS descriptors.  Once
that cold-start state exists, the functions below are restricted to views,
in-place kernels and copies into persistent buffers.  These tests intentionally
parse source instead of importing Megatron/Torch, so they also run on a CPU-only
checkout and fail when a CUDA materialization is reintroduced.
"""

from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _path(relative: str) -> Path:
    return _ROOT / relative


def _function(relative: str, name: str, class_name: str | None = None) -> ast.FunctionDef:
    tree = ast.parse(_path(relative).read_text(encoding="utf-8"))
    scope: list[ast.stmt] = tree.body
    if class_name is not None:
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        scope = cls.body
    for node in scope:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing {relative}:{class_name or '<module>'}.{name}")


def _source(relative: str, name: str, class_name: str | None = None) -> str:
    path = _path(relative)
    return ast.get_source_segment(
        path.read_text(encoding="utf-8"),
        _function(relative, name, class_name),
    ) or ""


_FACTORIES = {
    "arange", "as_tensor", "cat", "concat", "empty", "empty_like",
    "full", "full_like", "gather", "index_select", "ones", "ones_like",
    "rand", "rand_like", "randint", "randn", "randn_like", "stack",
    "tensor", "zeros", "zeros_like",
}
_LIKE_FACTORIES = {name for name in _FACTORIES if name.endswith("_like")}
_ALLOCATING_METHODS = {
    "bfloat16", "clone", "contiguous", "cuda", "float", "half", "int",
    "long", "new_empty", "new_full", "new_ones", "new_tensor", "new_zeros",
    "to",
}


def _cuda_materializations(fn: ast.FunctionDef) -> list[str]:
    found: list[str] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "torch":
                if func.attr not in _FACTORIES:
                    continue
                device_kw = next(
                    (kw.value for kw in node.keywords if kw.arg == "device"),
                    None,
                )
                cpu_only = (
                    isinstance(device_kw, ast.Constant)
                    and str(device_kw.value) == "cpu"
                )
                if func.attr in _LIKE_FACTORIES or (device_kw is not None and not cpu_only):
                    found.append(f"torch.{func.attr}@{node.lineno}")
            elif func.attr in _ALLOCATING_METHODS:
                found.append(f".{func.attr}@{node.lineno}")
        elif isinstance(func, ast.Name) and func.id == "device_tensor_from_pinned_host":
            found.append(f"{func.id}@{node.lineno}")
    return found


_SEALED_HOT_FUNCTIONS = (
    ("proposal/controller.py", None, "_execute_non_expert_post_iter"),
    ("proposal/controller.py", None, "_execute_expert_post_iter"),
    ("proposal/controller.py", None, "proposal_dispatch_post_iter"),
    ("proposal/controller.py", None, "proposal_wait_d2h_before_optimizer"),
    ("proposal/checkpoint/lossy_checkpoint_engine.py", "LossyCheckpointPipeline", "publish_lfcl_fired_experts"),
    ("proposal/checkpoint/lossy_checkpoint_engine.py", "LossyCheckpointPipeline", "_refresh_lossy_cp_state_refs"),
    ("proposal/cutoff/cutoff_orchestrator.py", "CutoffOrchestrationMixin", "_launch_gpu_approx_cutoff"),
    ("proposal/cutoff/cutoff_orchestrator.py", "CutoffOrchestrationMixin", "_find_selection_cutoff_for_current_iter"),
    ("proposal/emission/emission_orchestrator.py", "EmissionOrchestrationMixin", "_update_direct_write_device_scalars"),
    ("proposal/emission/emission_orchestrator.py", "EmissionOrchestrationMixin", "_snapshot_total_th_for_direct_write"),
    ("proposal/emission/emission_orchestrator.py", "EmissionOrchestrationMixin", "_write_lossy_records_for_current_iter"),
    ("proposal/emission/direct_emission.py", None, "prepare_segment_entry_tables"),
    ("proposal/emission/direct_emission.py", None, "emit_segment_direct_from_plan"),
    ("proposal/emission/segment_d2h.py", "SegmentD2H", "_enqueue_segment_commands"),
    ("proposal/storage/zero12_nonexpert.py", None, "build_zero12_nonexpert_payload"),
    ("proposal/storage/zero12_nonexpert.py", "Zero12NonExpertCollector", "collect"),
    ("proposal/storage/fc_manager.py", "FCManager", "collect_lfcl_batch_payload"),
    ("proposal/storage/fc_manager.py", "FCManager", "_collect_zero12_nonexpert_payload"),
    ("proposal/storage/expert_writer.py", "ExpertState", "write_for_iter"),
    ("proposal/storage/expert_writer.py", "ExpertState", "_do_lossy_fc_aos_emit"),
    ("proposal/storage/expert_writer.py", "ExpertState", "_record_direct_emission_feedback"),
    ("proposal/storage/lossless_compaction.py", "ExpertLosslessFC", "fire_planned"),
    ("proposal/refresh/fp32_state.py", "Fp32StateMapper", "zero12_partition_view"),
)


def test_sealed_steady_functions_have_no_cuda_materialization() -> None:
    violations: list[str] = []
    for relative, class_name, name in _SEALED_HOT_FUNCTIONS:
        calls = _cuda_materializations(_function(relative, name, class_name))
        violations.extend(
            f"{relative}:{class_name or '<module>'}.{name}:{call}"
            for call in calls
        )
    assert not violations, "steady CUDA materialization(s):\n" + "\n".join(violations)


def test_cold_allocators_are_one_shot_and_fail_closed_after_warmup() -> None:
    engine_init = _source(
        "proposal/checkpoint/lossy_checkpoint_engine.py",
        "_init_lossy_flat_layout",
        "LossyCheckpointPipeline",
    )
    assert 'getattr(self, "_flat_layout_done", False)' in engine_init
    assert "self._flat_layout_done = True" in engine_init

    prepare = _source(
        "proposal/cutoff/cutoff_orchestrator.py",
        "prepare_selection_state",
        "CutoffOrchestrationMixin",
    )
    assert "if self._selection_total_th is None:" in prepare
    assert "if _ofs_t is None:" in prepare
    assert "self._elem_offsets_for_selection is None" in prepare

    step = _source(
        "proposal/checkpoint/lossy_checkpoint_engine.py",
        "step",
        "LossyCheckpointPipeline",
    )
    assert 'if not hasattr(self, "_group_offsets_t"):' in step

    aos_run = _source("proposal/emission/aos_emitter.py", "run", "AosEmitter")
    assert "if self._direct_plan is None:" in aos_run
    assert aos_run.count("self._build_direct_plan(") == 1
    assert "capacity_signature != self._direct_plan_capacity_signature" in aos_run
    assert "refusing a steady-path CUDA reallocation" in aos_run

    warmup = _source(
        "proposal/storage/expert_writer.py", "warmup_record_pipeline", "ExpertState",
    )
    assert "self._ensure_direct_emission_counters(device)" in warmup
    assert "self._ensure_lossy_log_byte_device_counter(device)" in warmup
    hot_emit = _source(
        "proposal/storage/expert_writer.py", "_do_lossy_fc_aos_emit", "ExpertState",
    )
    assert "_require_direct_emission_counters" in hot_emit
    assert "_ensure_direct_emission_counters" not in hot_emit

    controller_init = _source("proposal/controller.py", "init_proposal")
    assert ".warmup_record_pipeline(" in controller_init


def test_zero1_checkpoint_sources_remain_storage_views() -> None:
    for name in ("_fp32_param_lfcl_source", "_fp32_opt_state_lfcl_source"):
        fn = _function("proposal/storage/fc_manager.py", name)
        assert not _cuda_materializations(fn), name
    mapper = _source(
        "proposal/refresh/fp32_state.py", "zero12_partition_view", "Fp32StateMapper",
    )
    assert "return flat.narrow(" in mapper
    assert "return slot.narrow(" in mapper


if __name__ == "__main__":
    test_sealed_steady_functions_have_no_cuda_materialization()
    test_cold_allocators_are_one_shot_and_fail_closed_after_warmup()
    test_zero1_checkpoint_sources_remain_storage_views()
    print("steady allocation contract: PASS")
