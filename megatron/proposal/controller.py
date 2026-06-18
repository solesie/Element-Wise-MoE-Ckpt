"""Proposal checkpoint pipeline implementation.

The checkpoint path dispatches the previous optimizer-step result at the start of the next
training iteration, immediately before forward. The training thread records a
CUDA event on the current main stream, submits a small task to the proposal
workers, and returns. Worker threads own all proposal GPU/D2H/file work.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .runtime import CheckpointWorkerPool, get_d2h_stream
from .refresh.fp32_state import (
    Fp32StateMapper,
    get_engine_optimizer,
    set_mapper,
)
from .refresh.hessian_refresh import HQuantileAnchor
from .config import ProposalConfig
from .storage.fc_manager import FCManager, _detect_expert_layout, DS_EXPERT_IDX_RE
from .checkpoint.lossy_checkpoint_engine import LossyCheckpointPipeline
from .storage.recovery import recover


@dataclass
class PostStepCheckpointTask:
    """Checkpoint task for the just-completed optimizer step."""

    cuda_event: Optional["torch.cuda.Event"]
    iter_T: int
    submitted_at: float = field(default_factory=time.perf_counter)
    non_expert_read_ready: threading.Event = field(default_factory=threading.Event)
    expert_read_ready: threading.Event = field(default_factory=threading.Event)
    non_expert_d2h_event: Optional["torch.cuda.Event"] = None
    expert_d2h_event: Optional["torch.cuda.Event"] = None
    non_expert_d2h_start_event: Optional["torch.cuda.Event"] = None
    expert_d2h_start_event: Optional["torch.cuda.Event"] = None
    non_expert_error: Optional[BaseException] = None
    expert_error: Optional[BaseException] = None
    non_expert_started_at: float = 0.0
    expert_started_at: float = 0.0
    non_expert_ready_at: float = 0.0
    expert_ready_at: float = 0.0


class ProposalController:
    """Owns proposal runtime state and exposes training-loop entry points."""

    def __init__(self) -> None:
        self._lossy_cp = None                          # LossyCheckpointPipeline
        self._fc = None                               # FCManager
        self._checkpoint_workers = None               # CheckpointWorkerPool
        self._expert_lossless_fc = None               # LFCL scheduler
        self._enabled = False
        self._cp_dir = None
        self._proposal_config = None                  # ProposalConfig
        self._h_anchor = None                         # HQuantileAnchor
        self._latest_post_step_checkpoint_task = None
        self._last_dispatched_iter_T = -1
        self._config = {}
        self._pending_fault_record = None
        self._dispatch_ready_event = None

    def init(self, model: Any, optimizer: Any, args: Any, cp_root: str) -> None:
        init_proposal(model, optimizer, args, cp_root)

    def dispatch_post_step_task(self, model: Any, iter_T: int) -> None:
        proposal_dispatch_post_iter(model, iter_T)

    def step(
        self,
        optimizer: Any,
        global_step: int,
        ready_event: Optional["torch.cuda.Event"] = None,
        d2h_event_callback: Optional[Any] = None,
        expert_d2h_start_event: Optional["torch.cuda.Event"] = None,
    ) -> Any:
        if self._lossy_cp is None:
            return None
        return self._lossy_cp.step(
            optimizer,
            global_step=global_step,
            ready_event=ready_event,
            d2h_event_callback=d2h_event_callback,
            expert_d2h_start_event=expert_d2h_start_event,
        )

    def wait_d2h_before_optimizer(self) -> None:
        proposal_wait_d2h_before_optimizer()

    def record_loss(self, step: int, loss: Optional[float]) -> None:
        proposal_record_loss(step, loss)

    def vram_audit_iter_begin(self, step: int) -> None:
        proposal_vram_audit_iter_begin(step)

    def vram_audit_iter_end(self, step: int) -> None:
        proposal_vram_audit_iter_end(step)

    def aos_cuda_graph_ready(self) -> bool:
        return bool(
            self._lossy_cp is not None
            and getattr(self._lossy_cp, "aos_cuda_graph_ready", lambda: False)()
        )

    def aos_cuda_graph_capture_required(self) -> bool:
        if self._lossy_cp is None:
            return False
        return not bool(getattr(self._lossy_cp, "test_expert_selection_only", False))

    def refresh_h_quantile(self, *args: Any, **kwargs: Any) -> Any:
        if self._h_anchor is not None:
            return self._h_anchor.refresh(*args, **kwargs)
        return proposal_refresh_h_quantile(*args, **kwargs)

    def set_anchor_data_iter(self, it: Any) -> None:
        if self._h_anchor is not None:
            self._h_anchor.set_anchor_data_iter(it)
            return
        proposal_set_anchor_data_iter(it)

    def attach_expert_state(self, expert_state: Any) -> None:
        if self._lossy_cp is None:
            raise RuntimeError("proposal lossy checkpoint pipeline is not initialized")
        self._lossy_cp.set_expert_state(expert_state)

    def simulate_fault(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return proposal_simulate_fault(*args, **kwargs)

    def recover(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return proposal_recover(*args, **kwargs)

    def shutdown(self) -> Dict[str, Any]:
        return proposal_shutdown()

    def acquire_dispatch_ready_event(self) -> Optional["torch.cuda.Event"]:
        if not torch.cuda.is_available():
            return None
        if self._dispatch_ready_event is None:
            raise RuntimeError("proposal dispatch event was not initialized")
        return self._dispatch_ready_event


_CONTROLLER = ProposalController()


def get_controller() -> ProposalController:
    return _CONTROLLER

def _unwrap_ds_engine(model):
    m = model[0] if isinstance(model, (list, tuple)) else model
    while hasattr(m, "module") and not hasattr(m, "named_parameters"):
        m = m.module
    if hasattr(m, "module") and hasattr(m.module, "named_parameters"):
        m = m.module
    return m


def _get_ds_optimizer(model):
    """Return the inner torch optimizer, unwrapping DeepSpeed ZeRO shell."""
    m = model[0] if isinstance(model, (list, tuple)) else model
    if hasattr(m, "optimizer"):
        opt = m.optimizer
        if hasattr(opt, "optimizer"):
            return opt.optimizer
        return opt
    return None


def _should_write() -> bool:
    """File writes are gated to DP rank 0. Tracker compute runs on all ranks."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def init_proposal(
    model,
    optimizer,
    args,
    cp_root: str,
) -> None:
    """Build singletons. Short-circuits if args.proposal_enable is False."""
    # PR_SET_PTRACER = ANY: allow py-spy / gdb attach when yama ptrace_scope = 1.
    # Diagnostic only — no behavior change. Env-gated.
    import os as _os_ptr
    if _os_ptr.environ.get("PROPOSAL_ALLOW_PTRACE", "") in ("1", "true", "True"):
        try:
            import ctypes as _ctypes_ptr
            _ctypes_ptr.CDLL('libc.so.6').prctl(0x59616d61, 0xffffffff, 0, 0, 0)
            print("[proposal] PR_SET_PTRACER=ANY (py-spy attach enabled)", flush=True)
        except Exception as e:
            print(f"[proposal] prctl failed: {e}", flush=True)

    get_controller()._enabled = bool(getattr(args, "proposal_enable", False))
    if not get_controller()._enabled:
        return
    get_controller()._proposal_config = ProposalConfig.from_args(args, cp_root)
    get_controller()._h_anchor = HQuantileAnchor(
        finite_diff_eps=get_controller()._proposal_config.finite_diff_eps,
        n_anchor_batches=max(
            1, int(get_controller()._proposal_config.n_anchor_batches or 1),
        ),
    )
    if torch.cuda.is_available() and get_controller()._dispatch_ready_event is None:
        get_controller()._dispatch_ready_event = torch.cuda.Event()

    # PR-MoE support: args.num_experts may be a list of per-layer counts.
    # FCManager re-derives per-layer counts from observed expert tensors;
    # we pass the raw spec through and use max() as the scalar metadata for
    # checkpoint managers, which only use it for sizing/logging.
    ne_arg = args.num_experts
    if isinstance(ne_arg, list):
        if any(int(x) < 1 for x in ne_arg):
            raise ValueError(f"num_experts entries must be ≥1: {ne_arg}")
        num_experts_spec = [int(x) for x in ne_arg]
        num_experts_scalar = max(num_experts_spec)
    else:
        num_experts_spec = int(ne_arg)
        num_experts_scalar = int(ne_arg)

    # B3: guard unsupported configs.
    zero_stage = int(getattr(args, "zero_stage", 0) or 0)
    ep_size = int(getattr(args, "moe_expert_parallel_size", getattr(args, "ep_size", 1)) or 1)
    tp_size = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    pp_size = int(getattr(args, "pipeline_model_parallel_size", 1) or 1)
    if zero_stage >= 3:
        raise NotImplementedError(
            f"proposal: ZeRO-3 partitions params; not supported (zero_stage={zero_stage})."
        )
    if ep_size > 1:
        raise NotImplementedError(
            f"proposal: expert-parallel size > 1 requires cross-EP all-gather of importance "
            f"scores (out of scope); got ep_size={ep_size}."
        )
    if tp_size > 1 or pp_size > 1:
        raise NotImplementedError(
            f"proposal: tensor/pipeline parallel > 1 unsupported "
            f"(tp={tp_size}, pp={pp_size})."
        )

    inner_model = _unwrap_ds_engine(model)
    inner_opt = _get_ds_optimizer(model) if optimizer is None else optimizer
    engine_opt = get_engine_optimizer(model)
    # Baseline VRAM snapshot before any proposal-side allocation.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        _baseline_alloc = torch.cuda.memory_allocated()
        _baseline_reserved = torch.cuda.memory_reserved()
        print(f"[proposal] BASELINE VRAM (before proposal init): "
              f"alloc={_baseline_alloc/1e6:.1f} MB, "
              f"reserved={_baseline_reserved/1e6:.1f} MB", flush=True)
        get_controller()._config.setdefault("_baseline_alloc", _baseline_alloc) if False else None
        # Stash on module so we can compare after init.
        globals()["_baseline_alloc_mb"] = _baseline_alloc / 1e6
        # Audit: capture baseline + install alloc trace patches (env-gated).
        from .telemetry import diagnostics as _proposal_diagnostics
        _proposal_diagnostics.set_baseline_alloc_mb(_baseline_alloc / 1e6)
        _proposal_diagnostics.install_alloc_trace()
        _proposal_diagnostics.vram_audit_step("baseline_pre_init")
    # Build the FP32 master / Adam state mapper. This handles ZeRO-0+FP16
    # (FP16_Optimizer), ZeRO-0+BF16 (BF16_Optimizer), and ZeRO-1/2
    # (DeepSpeedZeroOptimizer with _hp_mapping) in one interface so the rest
    # of the pipeline never touches DeepSpeed internals directly.
    mapper = Fp32StateMapper(engine_opt, inner_opt, inner_model)
    set_mapper(mapper)

    # B1: detect expert params by DeepSpeed's param.allreduce=False tag.
    expert_param_names: List[str] = []
    for name, p in inner_model.named_parameters():
        if getattr(p, "allreduce", True) is False:
            # Confirm layout with the DeepSpeed naming regex; a flagged param
            # that doesn't match means the model layout is unexpected.
            if not DS_EXPERT_IDX_RE.search(name):
                raise RuntimeError(
                    f"Param {name!r} has allreduce=False (DeepSpeed expert marker) "
                    f"but its name does not match the expected "
                    f"'deepspeed_experts.<idx>.' pattern; model layout unexpected."
                )
            # B3 (runtime): cross-check param.group_name implied EP size.
            gname = getattr(p, "group_name", "")
            m = re.fullmatch(r"ep_size_(\d+)", gname) if gname else None
            if m is not None and int(m.group(1)) > 1:
                raise NotImplementedError(
                    f"proposal: detected expert param {name!r} with group_name={gname!r} "
                    f"implying EP>1; unsupported."
                )
            expert_param_names.append(name)

    if not expert_param_names:
        raise RuntimeError(
            "proposal: no expert params detected (no param with allreduce=False). "
            "Is this an MoE model?"
        )

    if torch.distributed.get_rank() == 0:
        print(
            f"[proposal] Fp32StateMapper mode={mapper.mode}  "
            f"masters={len(getattr(mapper, '_direct_master', {}))} "
            f"adam_state_keys={len(getattr(mapper._inner_opt, 'state', {}))}",
            flush=True,
        )
        if mapper.mode == "unrecognised":
            for summary in mapper.candidate_summaries:
                print(f"[proposal] Fp32StateMapper candidate: {summary}", flush=True)


    # DeepSpeed MoE expert tensors must be separated by expert.
    layout = _detect_expert_layout(
        list(inner_model.named_parameters()), num_experts_scalar,
    )
    if layout != "separated":
        raise RuntimeError(
            f"proposal: expected DeepSpeed 'separated' expert layout, detected {layout!r}."
        )

    get_controller()._cp_dir = Path(cp_root)
    if _should_write():
        get_controller()._cp_dir.mkdir(parents=True, exist_ok=True)

    get_controller()._config = {
        "selection_mode": get_controller()._proposal_config.selection_mode,
        "topk_ratio": get_controller()._proposal_config.topk_ratio,
        "lfcl_interval_iters": get_controller()._proposal_config.lfcl_interval_iters,
        "lfcl_fraction_pct": get_controller()._proposal_config.lfcl_fraction_pct,
        "nonexpert_fc_W": get_controller()._proposal_config.nonexpert_fc_w,
        "async_queue": get_controller()._proposal_config.async_queue,
        "cp_mode": get_controller()._proposal_config.cp_mode,
        "num_experts": num_experts_scalar,
        "num_experts_spec": num_experts_spec,
        "proposal_config": get_controller()._proposal_config,
    }
    _test_skip_expert_commit = bool(
        getattr(args, "proposal_test_skip_expert_commit", False)
    )
    _test_skip_expert_d2h = bool(
        getattr(args, "proposal_test_skip_expert_d2h", False)
    )
    _test_expert_selection_only = bool(
        getattr(args, "proposal_test_expert_selection_only", False)
    )
    _test_skip_nonexpert_commit = bool(
        getattr(args, "proposal_test_skip_nonexpert_commit", False)
    )
    _test_skip_nonexpert_d2h = bool(
        getattr(args, "proposal_test_skip_nonexpert_d2h", False)
    )
    if _test_skip_expert_d2h and not _test_skip_expert_commit:
        raise ValueError(
            "--proposal-test-skip-expert-d2h requires "
            "--proposal-test-skip-expert-commit"
        )
    if _test_expert_selection_only and not _test_skip_expert_d2h:
        raise ValueError(
            "--proposal-test-expert-selection-only requires "
            "--proposal-test-skip-expert-d2h"
        )
    if _test_skip_nonexpert_d2h and not _test_skip_nonexpert_commit:
        raise ValueError(
            "--proposal-test-skip-nonexpert-d2h requires "
            "--proposal-test-skip-nonexpert-commit"
        )
    get_controller()._config["expert_lossy_enabled"] = (
        get_controller()._config["cp_mode"] != "sequential"
    )
    get_controller()._config["expert_lossy_commit_enabled"] = (
        not _test_skip_expert_commit
    )
    get_controller()._config["nonexpert_commit_enabled"] = (
        not _test_skip_nonexpert_commit
    )

    get_controller()._checkpoint_workers = (
        CheckpointWorkerPool(
            non_expert_queue_size=get_controller()._config["async_queue"],
            expert_queue_size=max(get_controller()._config["async_queue"] * 4, 16),
        )
        if _should_write() else None
    )

    if get_controller()._config["expert_lossy_enabled"]:
        get_controller()._lossy_cp = LossyCheckpointPipeline(
            inner_model,
            topk_ratio=get_controller()._config["topk_ratio"],
            expert_only=True,
            selection_mode=get_controller()._config["selection_mode"],
            num_experts=num_experts_scalar,
            random_seed=int(getattr(args, "seed", 0) or 0),
            checkpoint_workers=get_controller()._checkpoint_workers,
            group_size=get_controller()._proposal_config.group_size,
            proposal_config=get_controller()._proposal_config,
        )
        get_controller()._lossy_cp.test_expert_selection_only = bool(
            _test_expert_selection_only,
        )
        # Eager-allocate the per-BF16-group flat buffers and selection scratch.
        # Required so iter-1 H refresh has valid metadata before the first
        # lossy checkpoint step runs on expert_launcher.
        get_controller()._lossy_cp._init_lossy_flat_layout(inner_opt)
    else:
        get_controller()._lossy_cp = None

    # ---- NCCL process group for proposal histogram all-reduce ----
    # Per-rank score histograms are reduced so all ranks derive the same
    # global cutoff. Single-GPU runs leave `_proposal_pg = None`.
    _proposal_pg = None
    _proposal_world_size = 1
    try:
        if (torch.distributed.is_available()
                and torch.distributed.is_initialized()):
            _proposal_world_size = int(torch.distributed.get_world_size())
            if _proposal_world_size > 1:
                _proposal_pg = torch.distributed.new_group(
                    ranks=list(range(_proposal_world_size)),
                )
                if _should_write():
                    print(f"[proposal] NCCL proposal_pg created "
                          f"(world_size={_proposal_world_size})",
                          flush=True)
    except Exception as _exc:
        _proposal_pg = None
        _proposal_world_size = 1
        if _should_write():
            print(f"[proposal] proposal_pg init failed (single-GPU path): "
                  f"{_exc}", flush=True)
    if get_controller()._lossy_cp is not None:
        get_controller()._lossy_cp._proposal_pg = _proposal_pg
        get_controller()._lossy_cp._proposal_world_size = _proposal_world_size

    # CP layout: split non-expert + expert into separate subdirectories
    # so they can be mounted on different devices.
    #   <cp>/non_expert/  — non-expert full FC (W-scheduled)
    #   <cp>/expert/      — expert rolling LFSY + LFCL compaction
    # H_g is GPU-only (BF16 in ``_h_quantile_values``). Recovery
    # rebuilds it on the next FD refresh; no disk persist path.
    _nonexpert_cp_dir = get_controller()._cp_dir / "non_expert"
    _expert_cp_dir = get_controller()._cp_dir / "expert"

    # MoC-System baseline uses round_robin single-expert sequential FC.
    fc_schedule = (
        "sequential" if get_controller()._config["cp_mode"] == "sequential" else "proposal"
    )
    if _should_write():
        get_controller()._fc = FCManager(
            model=inner_model,
            optimizer=inner_opt,
            output_dir=_nonexpert_cp_dir,
            expert_output_dir=_expert_cp_dir,
            num_experts=num_experts_spec,
            nonexpert_fc_W=get_controller()._config["nonexpert_fc_W"],
            checkpoint_workers=get_controller()._checkpoint_workers,
            schedule=fc_schedule,
            lfcl_fraction_pct=get_controller()._config["lfcl_fraction_pct"],
            fresh_expert_state=True,
            # cp_target: 'disk' (default) or 'test_dram'. Both targets use
            # the same proposal GPU path; test_dram is an in-memory
            # correctness/recovery diagnostic target.
            cp_target=get_controller()._proposal_config.cp_target,
            pinned_aos_capacity_ratio=float(
                get_controller()._proposal_config.topk_ratio,
            ),
            pinned_aos_capacity_tolerance=float(
                get_controller()._proposal_config.pinned_aos_capacity_tolerance,
            ),
            enable_expert_state=True,
            commit_expert_lossy_records=(
                get_controller()._config["expert_lossy_commit_enabled"]
            ),
            commit_nonexpert_fc=(
                get_controller()._config["nonexpert_commit_enabled"]
            ),
            test_skip_expert_d2h=bool(_test_skip_expert_d2h),
            test_skip_nonexpert_d2h=bool(_test_skip_nonexpert_d2h),
        )
        # Wire the ExpertState sink into the lossy checkpoint path. Each
        # proposal step then routes selected expert records into
        # ExpertState.write_for_iter. Sequential mode never calls the lossy
        # path, so it skips this wiring.
        if get_controller()._config["cp_mode"] != "sequential":
            get_controller().attach_expert_state(get_controller()._fc.expert_state)
            if torch.cuda.is_available():
                get_controller()._fc.expert_state.warmup_record_pipeline(
                    next(inner_model.parameters()).device,
        )
        if _test_skip_expert_commit:
            if _test_expert_selection_only:
                print("[proposal] TEST MODE proposal-test-expert-selection-only: "
                      "expert selection/cutoff and host layout run; GPU "
                      "record emission, D2H, CPU scatter, LFSY writes, and "
                      "LFCL commit are skipped.",
                      flush=True)
            elif _test_skip_expert_d2h:
                print("[proposal] TEST MODE proposal-test-skip-expert-d2h: "
                      "expert selection/AOS materialization run; AOS D2H, "
                      "CPU scatter, LFSY writes, and LFCL commit are skipped.",
                      flush=True)
            else:
                print("[proposal] TEST MODE proposal-test-skip-expert-commit: "
                      "expert selection/AOS/D2H and LFCL payload D2H still "
                      "run; CPU scatter, LFSY writes, and LFCL commit are "
                      "skipped.",
                      flush=True)
        if _test_skip_nonexpert_commit:
            if _test_skip_nonexpert_d2h:
                print("[proposal] TEST MODE proposal-test-skip-nonexpert-d2h: "
                      "non-expert FC D2H and PRPTENS file write/fsync are "
                      "skipped.",
                      flush=True)
            else:
                print("[proposal] TEST MODE proposal-test-skip-nonexpert-commit: "
                      "non-expert FC D2H still runs; PRPTENS file write/fsync "
                      "is skipped.",
                      flush=True)

        # Expert LFCL compaction materializes plan-target live GPU p/m/v
        # into a pinned LFCL payload; ExpertState owns durable file commit.
        # ``fc_manager.fire_nonexpert_fc_if_due()`` only handles non-expert
        # FC after this split.
        if _test_skip_expert_d2h:
            get_controller()._expert_lossless_fc = None
        else:
            from .storage.lossless_compaction import ExpertLosslessFC
            from .storage.lfcl_plan import LossyLogSizeBasedPlan, RoundRobinPlan
            lfcl_policy = get_controller()._proposal_config.lfcl_plan_policy
            if lfcl_policy == "round_robin":
                lfcl_planner = RoundRobinPlan()
            elif lfcl_policy == "lossy_log_size":
                lfcl_planner = LossyLogSizeBasedPlan()
            else:
                raise ValueError(f"unknown LFCL plan policy: {lfcl_policy!r}")
            get_controller()._expert_lossless_fc = ExpertLosslessFC(
                all_expert_ids=get_controller()._fc._all_expert_ids,
                separated_tensors_by_slot=get_controller()._fc._separated_tensors_by_slot,
                sorted_layers=get_controller()._fc._sorted_layers,
                expert_state=get_controller()._fc.expert_state,
                fc_manager=get_controller()._fc,
                interval_iters=get_controller()._proposal_config.lfcl_interval_iters,
                schedule=fc_schedule,
                lfcl_fraction_pct=get_controller()._config["lfcl_fraction_pct"],
                planner=lfcl_planner,
                commit_payload=not _test_skip_expert_commit,
            )
    else:
        get_controller()._fc = None
        get_controller()._expert_lossless_fc = None

    get_controller()._config["_inner_model"] = inner_model
    get_controller()._config["_inner_opt"] = inner_opt
    # Persist fault log path early so proposal_shutdown dumps land next to it
    # even if no faults fire (or training crashes before first fault).
    fault_log = getattr(args, "proposal_fault_log", None)
    if fault_log:
        get_controller()._config["_fault_log_path"] = fault_log

    # finite_diff config.
    get_controller()._config["_finite_diff_eps"] = get_controller()._proposal_config.finite_diff_eps
    # Diagnostic env override: PROPOSAL_FD_EPS=<float> bypasses autotune
    # and forces a fixed eps. Used to test BF16 noise-floor hypotheses
    # (autotune's iter-1 chosen eps may be below BF16 ULP at output scale).
    import os as _os_eps
    _eps_env = _os_eps.environ.get("PROPOSAL_FD_EPS", "")
    if _eps_env:
        try:
            get_controller()._config["_finite_diff_eps"] = float(_eps_env)
        except ValueError:
            pass
    # Diagnostic: per-refresh expert coverage / skipped tensor ratio / eps.
    get_controller()._config["_h_diag_records"] = []
    # ANALYSIS: non-expert |H| probe — comparison of expert vs non-expert |H|
    # distribution at score_diag_iters; retained for measurement purposes.

    # POST-INIT VRAM diff vs baseline. Large selection buffers have already
    # been initialized, so this captures the steady V2 allocation footprint.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        _post_init_alloc = torch.cuda.memory_allocated()
        _baseline = globals().get("_baseline_alloc_mb", 0.0) * 1e6
        delta = (_post_init_alloc - _baseline) / 1e6
        print(f"[proposal] POST-INIT VRAM: alloc={_post_init_alloc/1e6:.1f} MB "
              f"(Δ from baseline = {delta:+.1f} MB; lazy buffers add at iter 1)",
              flush=True)
    # Audit: post_init record (R1 baseline). dump_breakdown captures
    # lossy checkpoint GPU buffers if initialized.
    from .telemetry import diagnostics as _proposal_diagnostics
    _proposal_diagnostics.vram_audit_step(
        "post_init", dump_breakdown=True, lossy_cp=get_controller()._lossy_cp,
    )


def proposal_record_loss(step: int, loss: Optional[float]) -> None:
    """Record the latest training loss and proposal diagnostics."""
    del step
    if not get_controller()._enabled:
        return
    if get_controller()._pending_fault_record is not None and loss is not None:
        get_controller()._pending_fault_record["post_fault_loss"] = float(loss)
        _flush_fault_record(get_controller()._pending_fault_record)
        get_controller()._pending_fault_record = None


def _proposal_tracked_gpu_memory_breakdown() -> Tuple[float, List[Tuple[str, float]]]:
    """Audit-only accounting for proposal-owned CUDA tensors.

    Training model/optimizer tensors are intentionally excluded. This is used
    only under ``PROPOSAL_VRAM_AUDIT`` to make R1 compare allocated VRAM
    against proposal-owned buffers that are expected to persist.
    """

    items: List[Tuple[str, float]] = []
    seen: set[Tuple[int, int]] = set()

    def add_tensor(name: str, tensor: Any) -> None:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            return
        try:
            storage = tensor.untyped_storage()
            key = (int(storage.data_ptr()), int(storage.nbytes()))
            size_mb = float(storage.nbytes()) / 1e6
        except Exception:
            key = (int(tensor.data_ptr()), int(tensor.element_size() * tensor.numel()))
            size_mb = float(tensor.element_size() * tensor.numel()) / 1e6
        if key in seen:
            return
        seen.add(key)
        items.append((name, size_mb))

    lossy_cp = get_controller()._lossy_cp
    if lossy_cp is not None and hasattr(lossy_cp, "gpu_memory_breakdown"):
        try:
            _lossy_total, lossy_items = lossy_cp.gpu_memory_breakdown()
            for name, size_mb in lossy_items:
                items.append((f"lossy_cp.{name}", float(size_mb)))
        except Exception:
            pass

    fc = get_controller()._fc
    if fc is not None:
        expert_state = getattr(fc, "expert_state", None)
        if expert_state is not None:
            add_tensor(
                "expert_state._lossy_log_bytes_device",
                getattr(expert_state, "_lossy_log_bytes_device", None),
            )
            add_tensor(
                "expert_state._direct_record_cursor_device",
                getattr(expert_state, "_direct_record_cursor_device", None),
            )
            add_tensor(
                "expert_state._direct_overflow_count_device",
                getattr(expert_state, "_direct_overflow_count_device", None),
            )
            record_pipeline = getattr(expert_state, "_record_pipeline", None)
            segment_d2h = getattr(record_pipeline, "_segment_d2h", None)
            if segment_d2h is not None:
                for idx, tensor in enumerate(getattr(segment_d2h, "_gpu_bufs", []) or []):
                    add_tensor(f"aos_segment._gpu_bufs[{idx}]", tensor)
            direct_plan = getattr(record_pipeline, "_direct_plan", None)
            layout_state = getattr(direct_plan, "layout_state", None)
            if layout_state is not None:
                add_tensor("aos_layout.offset_device", getattr(layout_state, "offset_device", None))
                add_tensor("aos_layout.capacity_device", getattr(layout_state, "capacity_device", None))
                add_tensor("aos_layout.header_device", getattr(layout_state, "header_device", None))

    return sum(size_mb for _name, size_mb in items), items


def _proposal_vram_audit_extra() -> Dict[str, Any]:
    total_mb, items = _proposal_tracked_gpu_memory_breakdown()
    return {
        "tracked_total_mb": float(total_mb),
        "tracked_items": [
            (name, float(size_mb)) for name, size_mb in items if size_mb > 0.01
        ],
    }


def proposal_vram_audit_iter_begin(step: int) -> None:
    from .telemetry import diagnostics as _proposal_diagnostics

    if not _proposal_diagnostics.is_vram_audit_enabled():
        return
    _proposal_diagnostics.set_iter(int(step))
    _proposal_diagnostics.vram_audit_step(
        "iter_begin",
        dump_breakdown=True,
        lossy_cp=get_controller()._lossy_cp,
        extra=_proposal_vram_audit_extra(),
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def proposal_vram_audit_iter_end(step: int) -> None:
    from .telemetry import diagnostics as _proposal_diagnostics

    if not _proposal_diagnostics.is_vram_audit_enabled():
        return
    _proposal_diagnostics.set_iter(int(step))
    _proposal_diagnostics.vram_audit_step(
        "iter_end",
        dump_breakdown=True,
        lossy_cp=get_controller()._lossy_cp,
        extra=_proposal_vram_audit_extra(),
    )


def _execute_non_expert_post_iter(task: PostStepCheckpointTask) -> None:
    """Body of the host_writer_non_expert worker task.

    The worker has CUDA context set. It must not host-sync on the iter T
    optimizer-step event; FCManager wires that event into its D2H stream so
    the training thread can continue launching forward kernels.
    """
    def _mark_non_expert_read_ready(
        d2h_event: Optional["torch.cuda.Event"],
    ) -> None:
        task.non_expert_d2h_event = d2h_event
        task.non_expert_ready_at = time.perf_counter()
        task.non_expert_read_ready.set()

    try:
        task.non_expert_started_at = time.perf_counter()
        if get_controller()._fc is not None:
            get_controller()._fc.fire_nonexpert_fc_if_due(
                step=task.iter_T,
                ready_event=task.cuda_event,
                d2h_event_callback=_mark_non_expert_read_ready,
                d2h_start_event=task.non_expert_d2h_start_event,
            )
    except BaseException as exc:
        task.non_expert_error = exc
        raise
    finally:
        if task.non_expert_ready_at == 0.0:
            task.non_expert_ready_at = time.perf_counter()
        task.non_expert_read_ready.set()


def _execute_expert_post_iter(task: PostStepCheckpointTask) -> None:
    """Body of the host_writer_expert worker task.

    Two responsibilities, both single-threaded on this worker:
      * fire the schedule-due expert LFCL compactions (layer-RR in sequential
        mode, planner-selected 1% per iter in proposal mode). ``ExpertLosslessFC.fire_if_due``
        internally launches the D2H copies on ``d2h_stream``.
      * when expert lossy checkpointing is enabled, run
        ``LossyCheckpointPipeline.step``. This dispatches this iter's lossy
        records after scoring on ``compute_stream``. The expert worker owns
        the whole path, so single-thread invariants hold without locks.
    """
    def _mark_expert_d2h_event(
        d2h_event: Optional["torch.cuda.Event"],
    ) -> None:
        task.expert_d2h_event = d2h_event

    try:
        task.expert_started_at = time.perf_counter()
        if (
            get_controller()._config.get("expert_lossy_enabled", True)
            and get_controller()._lossy_cp is not None
        ):
            inner_opt = get_controller()._config.get("_inner_opt")
            if inner_opt is not None:
                get_controller().step(
                    inner_opt,
                    global_step=task.iter_T,
                    ready_event=task.cuda_event,
                    d2h_event_callback=_mark_expert_d2h_event,
                    expert_d2h_start_event=task.expert_d2h_start_event,
                )
        fired_experts: List[Tuple[int, int]] = []
        if get_controller()._expert_lossless_fc is not None:
            fired_experts = get_controller()._expert_lossless_fc.fire_if_due(
                task.iter_T,
                ready_event=task.cuda_event,
            )
            lfcl_d2h_event = getattr(
                get_controller()._expert_lossless_fc,
                "last_d2h_event",
                None,
            )
            if lfcl_d2h_event is not None:
                _mark_expert_d2h_event(lfcl_d2h_event)
        if get_controller()._lossy_cp is not None:
            fired_by_step = getattr(
                get_controller()._lossy_cp,
                "_lfcl_fired_experts_by_step",
                None,
            )
            if fired_by_step is None:
                fired_by_step = {}
                setattr(
                    get_controller()._lossy_cp,
                    "_lfcl_fired_experts_by_step",
                    fired_by_step,
                )
            fired_by_step[int(task.iter_T)] = list(fired_experts)
        get_controller()._config["_last_fired_experts"] = fired_experts
        task.expert_ready_at = time.perf_counter()
        task.expert_read_ready.set()
    except BaseException as exc:
        task.expert_error = exc
        raise
    finally:
        if task.expert_ready_at == 0.0:
            task.expert_ready_at = time.perf_counter()
        task.expert_read_ready.set()


def proposal_wait_d2h_before_optimizer() -> None:
    """Order optimizer.step after this iter's checkpoint state reads.

    The stream-level wait below only covers work already enqueued on
    ``d2h_stream``. Dispatch is worker-owned, so the optimizer fence
    must first observe that the submitted ``PostStepCheckpointTask`` has
    reached its read-complete point; otherwise optimizer.step can mutate
    ``p/m/v`` while Stage 1/Stage 2 is still reading live Adam state.
    """
    if not get_controller()._enabled or not torch.cuda.is_available():
        return
    import os as _os_probe_fence
    _probe = _os_probe_fence.environ.get("PROPOSAL_MAIN_PROBE", "") in ("1", "true", "True")
    import time as _time_probe
    _t0 = _time_probe.perf_counter() if _probe else 0
    task = get_controller()._latest_post_step_checkpoint_task
    if task is not None:
        current = torch.cuda.current_stream()
        if task.non_expert_d2h_start_event is not None:
            task.non_expert_d2h_start_event.record(current)
        if task.expert_d2h_start_event is not None:
            task.expert_d2h_start_event.record(current)
        _t_before_non_expert = _time_probe.perf_counter() if _probe else 0
        task.non_expert_read_ready.wait()
        _t_after_non_expert = _time_probe.perf_counter() if _probe else 0
        task.expert_read_ready.wait()
        _t_after_expert = _time_probe.perf_counter() if _probe else 0
        if task.non_expert_error is not None:
            raise RuntimeError(
                f"proposal non-expert checkpoint task failed at "
                f"iter {task.iter_T}"
            ) from task.non_expert_error
        if task.expert_error is not None:
            raise RuntimeError(
                f"proposal expert checkpoint task failed at iter {task.iter_T}"
            ) from task.expert_error
        _non_event_ready = None
        _expert_event_ready = None
        if _probe:
            try:
                _non_event_ready = (
                    True if task.non_expert_d2h_event is None
                    else bool(task.non_expert_d2h_event.query())
                )
            except Exception:
                _non_event_ready = False
            try:
                _expert_event_ready = (
                    True if task.expert_d2h_event is None
                    else bool(task.expert_d2h_event.query())
                )
            except Exception:
                _expert_event_ready = False
        if task.non_expert_d2h_event is not None:
            current.wait_event(task.non_expert_d2h_event)
        if task.expert_d2h_event is not None:
            current.wait_event(task.expert_d2h_event)
        _t_after_event_waits = _time_probe.perf_counter() if _probe else 0
        if task is get_controller()._latest_post_step_checkpoint_task:
            get_controller()._latest_post_step_checkpoint_task = None
        if _probe:
            _non_wait_ms = (_t_after_non_expert - _t_before_non_expert) * 1000.0
            _expert_wait_ms = (_t_after_expert - _t_after_non_expert) * 1000.0
            _event_wait_ms = (_t_after_event_waits - _t_after_expert) * 1000.0
            try:
                _probe_every = max(
                    1,
                    int(_os_probe_fence.environ.get("PROPOSAL_MAIN_PROBE_EVERY", "10") or "10"),
                )
            except ValueError:
                _probe_every = 10
            _force_probe_row = int(task.iter_T) % _probe_every == 0
            if _force_probe_row or max(_non_wait_ms, _expert_wait_ms, _event_wait_ms) > 0.5:
                _queue_ne_ms = (
                    (task.non_expert_started_at - task.submitted_at) * 1000.0
                    if task.non_expert_started_at else -1.0
                )
                _queue_ex_ms = (
                    (task.expert_started_at - task.submitted_at) * 1000.0
                    if task.expert_started_at else -1.0
                )
                _run_ne_ms = (
                    (task.non_expert_ready_at - task.non_expert_started_at) * 1000.0
                    if task.non_expert_started_at and task.non_expert_ready_at else -1.0
                )
                _run_ex_ms = (
                    (task.expert_ready_at - task.expert_started_at) * 1000.0
                    if task.expert_started_at and task.expert_ready_at else -1.0
                )
                print(
                    f"[probe] fence_detail@iter{task.iter_T}: "
                    f"wait_non_expert={_non_wait_ms:.2f}ms "
                    f"wait_expert={_expert_wait_ms:.2f}ms "
                    f"event_wait={_event_wait_ms:.2f}ms "
                    f"non_event_ready={int(bool(_non_event_ready))} "
                    f"expert_event_ready={int(bool(_expert_event_ready))} "
                    f"queue_ne={_queue_ne_ms:.2f}ms "
                    f"queue_ex={_queue_ex_ms:.2f}ms "
                    f"run_ne={_run_ne_ms:.2f}ms "
                    f"run_ex={_run_ex_ms:.2f}ms",
                    flush=True,
                )
    _t_before_wait_stream = _t_after_wait_stream = 0
    if _probe:
        _t1 = _time_probe.perf_counter()
        if (_t1 - _t0) * 1000 > 0.5:
            _wait_stream_ms = 0.0
            print(
                f"[probe] fence: {(_t1-_t0)*1000:.2f}ms "
                f"wait_stream={_wait_stream_ms:.2f}ms",
                flush=True,
            )


def _submit_post_step_checkpoint_task(task: PostStepCheckpointTask) -> None:
    """Submit a post-step checkpoint task to the proposal workers."""

    if get_controller()._checkpoint_workers is None:
        _execute_non_expert_post_iter(task)
        _execute_expert_post_iter(task)
        return
    if (
        get_controller().aos_cuda_graph_capture_required()
        and not get_controller().aos_cuda_graph_ready()
    ):
        get_controller()._checkpoint_workers.submit_non_expert(
            _execute_non_expert_post_iter, task,
        )
        get_controller()._checkpoint_workers.drain_non_expert()
        get_controller()._checkpoint_workers.submit_expert(_execute_expert_post_iter, task)
        get_controller()._checkpoint_workers.drain_expert()
        return
    get_controller()._checkpoint_workers.submit_non_expert(_execute_non_expert_post_iter, task)
    get_controller()._checkpoint_workers.submit_expert(_execute_expert_post_iter, task)


def proposal_dispatch_post_iter(model, iter_T: int) -> None:
    """Submit iter ``T`` checkpoint work before next forward.

    The caller invokes this as the first action of iteration ``T+1`` after
    iter ``T``'s optimizer step has completed. The task is submitted
    immediately; no forward hook or arm/release stash is used.
    """

    if not get_controller()._enabled:
        return
    iter_T = int(iter_T)
    if iter_T <= int(getattr(get_controller(), "_last_dispatched_iter_T", -1)):
        return
    import os as _os_probe_dispatch
    _probe = _os_probe_dispatch.environ.get(
        "PROPOSAL_MAIN_PROBE", "") in ("1", "true", "True")
    import time as _time_probe
    _t0 = _time_probe.perf_counter() if _probe else 0.0
    event = get_controller().acquire_dispatch_ready_event()
    if event is not None:
        event.record()
    task = PostStepCheckpointTask(
        cuda_event=event,
        iter_T=iter_T,
        non_expert_d2h_start_event=(
            torch.cuda.Event()
            if torch.cuda.is_available()
            and (
                os.environ.get("PROPOSAL_DEFER_NONEXPERT_D2H_TO_FENCE", "")
                in ("1", "true", "TRUE")
                or os.environ.get("PROPOSAL_DEFER_ALL_D2H_TO_FENCE", "")
                in ("1", "true", "TRUE")
            )
            else None
        ),
        expert_d2h_start_event=(
            torch.cuda.Event()
            if torch.cuda.is_available()
            and (
                os.environ.get("PROPOSAL_DEFER_AOS_D2H_TO_FENCE", "")
                in ("1", "true", "TRUE")
                or os.environ.get("PROPOSAL_DEFER_ALL_D2H_TO_FENCE", "")
                in ("1", "true", "TRUE")
            )
            else None
        ),
    )
    get_controller()._latest_post_step_checkpoint_task = task
    _submit_post_step_checkpoint_task(task)
    get_controller()._last_dispatched_iter_T = iter_T
    if _probe:
        _t1 = _time_probe.perf_counter()
        if (_t1 - _t0) * 1000 > 0.5:
            print(
                f"[probe] dispatch@iter{iter_T}: {(_t1-_t0)*1000:.2f}ms",
                flush=True,
            )
    from .telemetry import diagnostics as _d
    _d._proposal_post_iter_diagnostics(model, int(iter_T))

def proposal_recover(model, optimizer, t_fault: int) -> Optional[Dict[str, Any]]:
    if not get_controller()._enabled or get_controller()._cp_dir is None:
        return None
    if get_controller()._checkpoint_workers is not None:
        get_controller()._checkpoint_workers.drain()
    # Flush the log-structured expert state so end-of-training replay sees
    # the same on-disk snapshot that fault injection would see.
    if get_controller()._fc is not None and get_controller()._fc.expert_state is not None:
        try:
            get_controller()._fc.expert_state.flush()
        except Exception as exc:
            print(
                f"[proposal] WARNING end-of-training expert-state flush failed: {exc}; "
                f"continuing without a freshly flushed expert image.",
                flush=True,
            )
    inner_model = get_controller()._config["_inner_model"]
    inner_opt = get_controller()._config["_inner_opt"] if optimizer is None else optimizer
    num_experts = get_controller()._config["num_experts"]
    # cp_target=test_dram is a filesystem-off diagnostic mode. If in-memory
    # images exist, recovery consumes them; production disk mode reconstructs
    # from LFCL/LFSY + non-expert FC files.
    _expert_state_image = None
    _nonexpert_state_image = None
    if (
        get_controller()._fc is not None and get_controller()._fc.expert_state is not None
        and getattr(get_controller()._fc, "cp_target", "disk") == "test_dram"
    ):
        _expert_state_image = get_controller()._fc.expert_state
        _nonexpert_state_image = get_controller()._fc
    import time as _time
    _t_rec0 = _time.perf_counter()
    diag = recover(
        model=inner_model,
        optimizer=inner_opt,
        cp_dir=get_controller()._cp_dir,
        t_fault=t_fault,
        num_experts=num_experts,
        expert_state_image=_expert_state_image,
        nonexpert_state_image=_nonexpert_state_image,
    )
    _t_rec1 = _time.perf_counter()
    # H_g is GPU-only; the next FD refresh will repopulate it. No disk
    # restore step.
    _t_rec2 = _time.perf_counter()
    recover_s = _t_rec2 - _t_rec0
    print(
        f"[proposal] proposal_recover: recover_s={recover_s:.3f}  "
        f"(recover={_t_rec1 - _t_rec0:.3f}s h_load={_t_rec2 - _t_rec1:.3f}s)  "
        f"ne_fc={diag.nonexpert_fc_iter} expert_fc_count={diag.expert_fc_count}",
        flush=True,
    )
    return {
        "recover_time_s": float(recover_s),
        "recover_time_main_s": float(_t_rec1 - _t_rec0),
        "recover_time_h_s": float(_t_rec2 - _t_rec1),
        "nonexpert_fc_iter": diag.nonexpert_fc_iter,
        "expert_fc_count": diag.expert_fc_count,
    }


def _flush_fault_record(rec: Dict[str, Any]) -> None:
    import json
    log_path = get_controller()._config.get("_fault_log_path") if get_controller()._config else None
    if not log_path:
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


@torch.no_grad()
def _simulate_fault_impl(
    model, optimizer, step: int, args,
    pre_fault_loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a synthetic mid-training recovery check.

    Drains async writer, reconstructs model + optimizer state from the
    existing checkpoint files via ``proposal_recover``, and
    returns diagnostic info (recovery time, diag, pre/post loss).

    Training caller is expected to resume from the recovered state. We do
    NOT reset the FCManager or lossy checkpoint pipeline — those stateful objects keep
    running as if no fault occurred (correct: the CP they wrote is what we
    just reloaded, so their bookkeeping is already consistent).
    """
    import time
    import json

    if not get_controller()._enabled:
        return {}
    # The normal V2 cadence dispatches iter T at the beginning of iter T+1,
    # before the next forward. Synthetic fault injection runs immediately
    # after iter T's train step, before that next-loop dispatch point. Flush
    # the current post-step state here so the fault experiment measures the
    # same checkpoint visibility as old end-of-iter ETC runs.
    if int(step) > int(getattr(get_controller(), "_last_dispatched_iter_T", -1)):
        proposal_dispatch_post_iter(model, iter_T=int(step))
    if get_controller()._checkpoint_workers is not None:
        get_controller()._checkpoint_workers.drain()

    # Before recovery, fsync the log-structured expert state so segment
    # contents are durable on disk. recover() then reads each expert's
    # compacted.bin + remaining segments via read_expert_state(expert_dir).
    flushed_paths: List[Path] = []
    if get_controller()._fc is not None and get_controller()._fc.expert_state is not None and get_controller()._cp_dir is not None:
        try:
            flushed_paths = get_controller()._fc.expert_state.flush()
        except Exception as exc:
            print(
                f"[proposal] WARNING expert-state flush failed @iter {step}: {exc}; "
                f"continuing without freshly flushed expert paths.",
                flush=True,
            )
            flushed_paths = []

    t0 = time.perf_counter()
    inner_model = get_controller()._config["_inner_model"]
    inner_opt = get_controller()._config["_inner_opt"]
    num_experts = get_controller()._config["num_experts"]
    # Same test_dram short-circuit as proposal_recover() (above): only the
    # filesystem-off diagnostic path may pass in-memory images.
    _expert_state_image = None
    _nonexpert_state_image = None
    if (
        get_controller()._fc is not None and get_controller()._fc.expert_state is not None
        and getattr(get_controller()._fc, "cp_target", "disk") == "test_dram"
    ):
        _expert_state_image = get_controller()._fc.expert_state
        _nonexpert_state_image = get_controller()._fc
    diag = recover(
        model=inner_model,
        optimizer=inner_opt,
        cp_dir=get_controller()._cp_dir,
        t_fault=int(step),
        num_experts=num_experts,
        expert_state_image=_expert_state_image,
        nonexpert_state_image=_nonexpert_state_image,
    )
    # H_g is GPU-only; next FD refresh repopulates it.
    recover_s = time.perf_counter() - t0

    # Maximum lost iteration window "B" = step - min(expert_fc_iter, ne_fc_iter).
    # For MoC (sequential, no DC), B is max iters since last FC for any expert.
    # For proposal/random, B ≈ iters since last same-expert LFCL.
    min_fc_iter = diag.nonexpert_fc_iter
    if diag.expert_fc_iters:
        min_fc_iter = min(
            diag.nonexpert_fc_iter,
            *diag.expert_fc_iters.values(),
        )
    B_max = int(step) - int(min_fc_iter)

    rec = {
        "step": int(step),
        "recover_time_s": float(recover_s),
        "pre_fault_loss": (
            float(pre_fault_loss) if pre_fault_loss is not None else None
        ),
        "post_fault_loss": None,  # filled on next post_step by caller
        "nonexpert_fc_iter": int(diag.nonexpert_fc_iter),
        "expert_fc_count": int(diag.expert_fc_count),
        "B_max_lost_iter": B_max,
    }
    # Store pending — post_fault_loss is filled on next train step.
    get_controller()._pending_fault_record = rec
    # Register fault log path for _flush_fault_record callback.
    log_path = getattr(args, "proposal_fault_log", None)
    if log_path:
        get_controller()._config["_fault_log_path"] = log_path
    return rec


def proposal_simulate_fault(
    model, optimizer, step: int, args, pre_fault_loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the synthetic mid-training fault path."""

    return _simulate_fault_impl(
        model, optimizer, step=step, args=args, pre_fault_loss=pre_fault_loss,
    )


def proposal_set_anchor_data_iter(it: Any) -> None:
    from .refresh import hessian_refresh as h_refresh
    h_refresh.proposal_set_anchor_data_iter(it)


def proposal_refresh_h_quantile(
    model, forward_step_func, data_iterator, iteration,
) -> Optional[Dict[str, Any]]:
    from .refresh import hessian_refresh as h_refresh
    return h_refresh.proposal_refresh_h_quantile(
        model, forward_step_func, data_iterator, iteration)


def proposal_shutdown() -> Dict[str, Any]:
    """Drain writers, dump write logs and diagnostic summaries, then cleanup."""
    if not get_controller()._enabled:
        return {}
    import json
    summary: Dict[str, Any] = {}
    dump_dir: Optional[Path] = None
    dump_prefix = ""
    fl = get_controller()._config.get("_fault_log_path") if get_controller()._config else None
    if fl:
        dump_dir = Path(fl).parent
        # Extract tag from fault log basename: "{tag}.faults.jsonl" → tag prefix
        base = Path(fl).name
        if base.endswith(".faults.jsonl"):
            dump_prefix = base[: -len(".faults.jsonl")] + "."
    elif get_controller()._cp_dir is not None:
        dump_dir = get_controller()._cp_dir
    if get_controller()._fc is not None:
        # Shutdown ExpertState: fsync any open lossy-FC fds and close them
        # cleanly so recovery sees a consistent on-disk snapshot.
        if get_controller()._fc.expert_state is not None:
            try:
                get_controller()._fc.expert_state.shutdown()
            except Exception as exc:
                print(f"[proposal] WARNING shutdown expert-state close failed: {exc}", flush=True)
        get_controller()._fc.finalize()
        summary["fc_expert_events"] = sum(
            1 for s in get_controller()._fc.write_log if s.kind == "expert"
        )
        summary["fc_ne_events"] = sum(
            1 for s in get_controller()._fc.write_log if s.kind == "nonexpert"
        )
        if dump_dir is not None:
            try:
                (dump_dir / f"{dump_prefix}fc_write_log.json").write_text(
                    json.dumps(get_controller()._fc.write_log_dict(), indent=2)
                )
            except Exception:
                pass
    if get_controller()._checkpoint_workers is not None:
        ws = get_controller()._checkpoint_workers.summary()
        summary["async_total_write_ms"] = ws.get("total_write_ms", 0.0)
        get_controller()._checkpoint_workers.shutdown()
    # Audit: dump per-iter alloc trace summary to audit log (R2 evidence).
    try:
        from .telemetry import diagnostics as _proposal_diagnostics
        _proposal_diagnostics.dump_alloc_trace_to_log()
    except Exception:
        pass
    if dump_dir is not None:
        try:
            (dump_dir / f"{dump_prefix}proposal_shutdown_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
        except Exception:
            pass
    # Dump per-refresh diagnostic record (expert coverage, skip ratio, eps).
    h_diag_records = get_controller()._config.get("_h_diag_records") if get_controller()._config else None
    if dump_dir is not None and h_diag_records:
        try:
            (dump_dir / f"{dump_prefix}h_diag_records.jsonl").write_text(
                "\n".join(json.dumps(r) for r in h_diag_records) + "\n"
            )
        except Exception:
            pass
    get_controller()._enabled = False
    return summary
