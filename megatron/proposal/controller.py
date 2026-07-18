"""Proposal checkpoint pipeline implementation.

The checkpoint path dispatches the previous optimizer-step result at the start of the next
training iteration, immediately before forward. The training thread records a
CUDA event on the current main stream, submits a small task to the proposal
workers, and returns. Worker threads own all proposal GPU/D2H/file work.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .runtime import CheckpointWorkerPool
from .distributed import ProposalTopology, topology_from_args
from .refresh.fp32_state import (
    Fp32StateMapper,
    get_engine_optimizer,
    set_mapper,
)
from .refresh.hessian_refresh import HQuantileAnchor
from .config import ProposalConfig, lr_warmup_iters_from_args
from .expert_id import build_expert_id_map
from .storage.fc_manager import FCManager, _detect_expert_layout, DS_EXPERT_IDX_RE
from .checkpoint.lossy_checkpoint_engine import LossyCheckpointPipeline
from .storage.recovery import recover


@dataclass
class PostStepCheckpointTask:
    """Checkpoint task for the just-completed optimizer step."""

    cuda_event: Optional["torch.cuda.Event"]
    iter_T: int
    sync_processes: bool = True
    dispatch_release_ready: threading.Event = field(
        default_factory=threading.Event,
    )
    submitted_at: float = field(default_factory=time.perf_counter)
    non_expert_read_ready: threading.Event = field(default_factory=threading.Event)
    expert_read_ready: threading.Event = field(default_factory=threading.Event)
    non_expert_d2h_event: Optional["torch.cuda.Event"] = None
    expert_d2h_event: Optional["torch.cuda.Event"] = None
    non_expert_error: Optional[BaseException] = None
    expert_error: Optional[BaseException] = None
    non_expert_started_at: float = 0.0
    expert_started_at: float = 0.0
    non_expert_ready_at: float = 0.0
    expert_ready_at: float = 0.0
    defer_stage0_emission: bool = False
    stage0_emission_release: threading.Event = field(
        default_factory=threading.Event,
    )
    stage0_emission_cuda_event: Optional["torch.cuda.Event"] = None


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
        self._topology: Optional[ProposalTopology] = None
        self._h_anchor = None                         # HQuantileAnchor
        self._latest_post_step_checkpoint_task = None
        self._last_dispatched_iter_T = -1
        self._config = {}
        self._pending_fault_record = None
        self._dispatch_ready_event = None
        # Re-recorded once per normal PP-stage-0 iteration.  The object itself
        # is allocated once at proposal init so the steady path creates no
        # CUDA events.
        self._stage0_emission_cuda_event = None
        self._proposal_cpu_pg = None
        self._proposal_world_size = 1
        self._warmup_end_iter = 0
        self._active = False

    def init(self, model: Any, optimizer: Any, args: Any, cp_root: str) -> None:
        init_proposal(model, optimizer, args, cp_root)

    def dispatch_post_step_task(
        self,
        model: Any,
        iter_T: int,
        *,
        sync_processes: bool = True,
        defer_stage0_emission: bool = False,
    ) -> None:
        proposal_dispatch_post_iter(
            model,
            iter_T,
            sync_processes=bool(sync_processes),
            defer_stage0_emission=bool(defer_stage0_emission),
        )

    def finalize_post_step(self, model: Any, iteration: int) -> None:
        proposal_finalize_post_iter(model, iteration)

    def activate_after_warmup(self, model: Any, iteration: int) -> bool:
        return proposal_activate_after_warmup(model, iteration)

    def prepare_after_optimizer(self) -> None:
        proposal_prepare_after_optimizer()

    def step(
        self,
        optimizer: Any,
        global_step: int,
        ready_event: Optional["torch.cuda.Event"] = None,
        d2h_event_callback: Optional[Any] = None,
        emission_release_event: Optional[threading.Event] = None,
        emission_cuda_event: Optional["torch.cuda.Event"] = None,
    ) -> Any:
        if self._lossy_cp is None:
            return None
        return self._lossy_cp.step(
            optimizer,
            global_step=global_step,
            ready_event=ready_event,
            d2h_event_callback=d2h_event_callback,
            emission_release_event=emission_release_event,
            emission_cuda_event=emission_cuda_event,
        )

    def wait_d2h_before_optimizer(self) -> None:
        proposal_wait_d2h_before_optimizer()

    def record_loss(self, step: int, loss: Optional[float]) -> None:
        proposal_record_loss(step, loss)

    def run_independent_oracle(
        self,
        model: Any,
        data_iterator: Any,
        iteration: int,
    ) -> None:
        proposal_run_independent_oracle(model, data_iterator, iteration)

    def vram_audit_iter_begin(self, step: int) -> None:
        proposal_vram_audit_iter_begin(step)

    def vram_audit_iter_end(self, step: int) -> None:
        proposal_vram_audit_iter_end(step)

    def vram_audit_dispatch_post_iter(self, model: Any, iter_T: int) -> bool:
        return proposal_vram_audit_dispatch_post_iter(model, iter_T)

    def refresh_h_quantile(self, *args: Any, **kwargs: Any) -> Any:
        if self._h_anchor is not None:
            return self._h_anchor.refresh(*args, **kwargs)
        return proposal_refresh_h_quantile(*args, **kwargs)

    def set_anchor_data_iter(self, it: Any) -> None:
        if self._h_anchor is not None:
            self._h_anchor.set_anchor_data_iter(it)
            return
        proposal_set_anchor_data_iter(it)

    def set_oracle_data_iter(self, it: Any) -> None:
        proposal_set_oracle_data_iter(it)

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
    """Rank-0 console/log side effects only. Data ownership uses topology."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _stage0_bubble_deferral_enabled(
    topology: ProposalTopology,
    *,
    enable_zbh1_pipeline: bool,
) -> bool:
    """Whether this rank has the standard 1F1B stage-0 backward bubble."""

    return bool(
        int(topology.pp_size) > 1
        and int(topology.pp_rank) == 0
        and not bool(enable_zbh1_pipeline)
    )


def _release_stage0_emission(*, record_cuda_event: bool = True) -> None:
    """Release deferred stage-0 checkpoint reads at the gradient-wait boundary.

    The host event wakes both checkpoint workers.  A reusable CUDA event orders
    expert emission and non-expert D2H after the main-stream forward work
    preceding the first ``RecvGrad``.  The normal PipelineEngine hook records
    that event, posts the complete receive through the original instruction,
    and only then calls this helper with ``record_cuda_event=False``.  That host
    ordering is essential: checkpoint traffic must never race ahead of posting
    the NCCL receive.  Repeated calls are harmless and provide an optimizer-step
    fallback for custom PipelineEngine schedules that omit the standard hook.
    """

    ctrl = get_controller()
    task = getattr(ctrl, "_latest_post_step_checkpoint_task", None)
    if task is None or not bool(task.defer_stage0_emission):
        return
    if task.stage0_emission_release.is_set():
        return
    event = task.stage0_emission_cuda_event
    try:
        if record_cuda_event and event is not None and torch.cuda.is_available():
            event.record(torch.cuda.current_stream())
    finally:
        # Never strand either worker if CUDA/P2P fails while entering the
        # bubble.  The failed training step will still propagate its error;
        # this release is only for worker/teardown liveness.
        task.stage0_emission_release.set()


def _wait_stage0_emission_boundary(
    task: PostStepCheckpointTask,
) -> Optional["torch.cuda.Event"]:
    """Wait on the worker thread and return the bubble boundary event."""

    ready_event = task.cuda_event
    if not task.defer_stage0_emission:
        return ready_event
    if not task.stage0_emission_release.wait(timeout=300.0):
        raise RuntimeError(
            "timed out waiting for PP stage0 backward-bubble release"
        )
    boundary_event = task.stage0_emission_cuda_event
    return boundary_event if boundary_event is not None else ready_event


def _install_pipeline_optimizer_fence(model: Any) -> None:
    """Wire proposal bubble release/fence into DeepSpeed PipelineEngine.

    PipelineEngine executes forward/backward and optimizer.step inside
    ``train_batch()``, so the normal Megatron optimizer-fence call site is not
    reached.  Install an instance-local schedule hook that preserves the
    single-GPU contract: proposal D2H/read work for the previous step must be
    complete before the next optimizer mutation of p/m/v.
    """
    engine = model[0] if isinstance(model, (list, tuple)) else model
    if getattr(engine, "_proposal_pipeline_fence_installed", False):
        return
    instr_map = getattr(engine, "_INSTRUCTION_MAP", None)
    if not isinstance(instr_map, dict) or not hasattr(engine, "train_batch"):
        return

    opt_key = None
    original_opt = None
    recv_grad_key = None
    original_recv_grad = None
    for key, fn in instr_map.items():
        if getattr(key, "__name__", "") == "OptimizerStep":
            opt_key = key
            original_opt = fn
        elif getattr(key, "__name__", "") == "RecvGrad":
            recv_grad_key = key
            original_recv_grad = fn
    if opt_key is None or original_opt is None:
        return

    def _proposal_optimizer_step(self, *args: Any, **kwargs: Any) -> Any:
        # Correctness fallback for a custom schedule without the standard
        # late-backward cooldown hook.
        # Release first, then wait for the resulting read/D2H completion.
        _release_stage0_emission(record_cuda_event=True)
        proposal_wait_d2h_before_optimizer()
        return original_opt(self, *args, **kwargs)

    def _proposal_recv_grad(self, *args: Any, **kwargs: Any) -> Any:
        task = get_controller()._latest_post_step_checkpoint_task
        first_deferred_recv = bool(
            task is not None
            and task.defer_stage0_emission
            and not task.stage0_emission_release.is_set()
        )
        try:
            if first_deferred_recv:
                # Seal only the preceding F1/.../F(P-1) work.  This event must
                # be recorded before the original receive, because c10d makes
                # the main stream wait for that receive before subsequent
                # backward.
                event = task.stage0_emission_cuda_event
                if event is not None and torch.cuda.is_available():
                    event.record(torch.cuda.current_stream())
            # DeepSpeed posts every gradient tensor receive here.  Its CPU
            # call returns after enqueue while the NCCL kernel remains active.
            return original_recv_grad(self, *args, **kwargs)
        finally:
            if first_deferred_recv:
                # Wake the worker only after NCCL receive submission.  The
                # worker waits the pre-receive event above, not an event placed
                # behind the main-stream receive dependency.  On a CUDA/P2P
                # exception this release is only a teardown-liveness measure;
                # the optimizer is not allowed to continue that failed step.
                _release_stage0_emission(record_cuda_event=False)

    local_map = dict(instr_map)
    local_map[opt_key] = _proposal_optimizer_step
    if recv_grad_key is not None and original_recv_grad is not None:
        local_map[recv_grad_key] = _proposal_recv_grad
    engine._INSTRUCTION_MAP = local_map
    engine._proposal_pipeline_fence_installed = True
    if _should_write():
        print(
            "[proposal] PipelineEngine optimizer fence/stage0 bubble hook installed",
            flush=True,
        )


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
    get_controller()._warmup_end_iter = max(
        0, int(lr_warmup_iters_from_args(args)),
    )
    get_controller()._active = get_controller()._warmup_end_iter == 0
    get_controller()._h_anchor = HQuantileAnchor(
        finite_diff_eps=get_controller()._proposal_config.finite_diff_eps,
        n_anchor_batches=max(
            1, int(get_controller()._proposal_config.n_anchor_batches or 1),
        ),
    )
    if torch.cuda.is_available() and get_controller()._dispatch_ready_event is None:
        get_controller()._dispatch_ready_event = torch.cuda.Event()

    # args.num_experts may be a list of per-layer expert counts.
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

    topology = topology_from_args(args)
    get_controller()._topology = topology
    _defer_stage0_emission = _stage0_bubble_deferral_enabled(
        topology,
        enable_zbh1_pipeline=bool(
            getattr(args, "enable_zbh1_pipeline", False)
        ),
    )
    if (
        _defer_stage0_emission
        and torch.cuda.is_available()
        and get_controller()._stage0_emission_cuda_event is None
    ):
        get_controller()._stage0_emission_cuda_event = torch.cuda.Event()
    # These topology keys also identify objects in the rank-local I/O backlog,
    # so publish them for test_dram/file runs as well as the FDP backend.
    os.environ.setdefault("PROPOSAL_FDP_RANK", str(int(topology.global_rank)))
    os.environ.setdefault("PROPOSAL_FDP_WORLD_SIZE", str(int(topology.world_size)))
    os.environ.setdefault("PROPOSAL_FDP_DP_RANK", str(int(topology.dp_rank)))
    os.environ.setdefault("PROPOSAL_FDP_DP_SIZE", str(int(topology.dp_size)))
    os.environ.setdefault("PROPOSAL_FDP_PP_RANK", str(int(topology.pp_rank)))
    os.environ.setdefault("PROPOSAL_FDP_PP_SIZE", str(int(topology.pp_size)))
    os.environ.setdefault("PROPOSAL_FDP_EP_RANK", str(int(topology.ep_rank)))
    os.environ.setdefault("PROPOSAL_FDP_EP_SIZE", str(int(topology.ep_size)))
    num_layers = int(getattr(args, "num_layers", 0) or 0)
    if num_layers > 0:
        configured_layers = os.environ.get("PROPOSAL_FDP_NUM_MOE_LAYERS")
        if (
            configured_layers is not None
            and int(configured_layers) != int(num_layers)
        ):
            raise ValueError(
                "PROPOSAL_FDP_NUM_MOE_LAYERS must equal the model layer "
                f"count ({num_layers}), got {configured_layers}"
            )
        os.environ["PROPOSAL_FDP_NUM_MOE_LAYERS"] = str(num_layers)
    os.environ.setdefault(
        "PROPOSAL_FDP_EXPERTS_PER_LAYER", str(int(num_experts_scalar)),
    )
    if int(topology.ep_size) > 1 and int(num_experts_scalar) > 0:
        if int(num_experts_scalar) % int(topology.ep_size) == 0:
            local_experts = int(num_experts_scalar) // int(topology.ep_size)
            os.environ.setdefault(
                "PROPOSAL_FDP_EXPERT_ID_BASE",
                str(int(topology.ep_rank) * int(local_experts)),
            )
        elif (
            os.environ.get("PROPOSAL_IO_BACKEND", "").lower() == "fdp"
            and "PROPOSAL_FDP_EXPERT_ID_BASE" not in os.environ
        ):
            raise RuntimeError(
                "FDP global expert-slot mapping requires num_experts to be "
                "divisible by ep_size unless PROPOSAL_FDP_EXPERT_ID_BASE "
                "is set explicitly."
            )

    # B3: guard unsupported configs.
    zero_stage = int(getattr(args, "zero_stage", 0) or 0)
    ep_size = int(topology.ep_size)
    tp_size = int(topology.tp_size)
    pp_size = int(topology.pp_size)
    if zero_stage >= 3:
        raise NotImplementedError(
            f"proposal: ZeRO-3 partitions params; not supported (zero_stage={zero_stage})."
        )
    if tp_size > 1:
        raise NotImplementedError(
            f"proposal: tensor parallel > 1 unsupported (tp={tp_size})."
        )

    inner_model = _unwrap_ds_engine(model)
    inner_opt = _get_ds_optimizer(model) if optimizer is None else optimizer
    engine_opt = get_engine_optimizer(model)
    # Baseline VRAM snapshot before any proposal-side allocation.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        _baseline_alloc = torch.cuda.memory_allocated()
        _baseline_reserved = torch.cuda.memory_reserved()
        if _should_write():
            print(f"[proposal] BASELINE VRAM (before proposal init): "
                  f"alloc={_baseline_alloc/1e6:.1f} MB, "
                  f"reserved={_baseline_reserved/1e6:.1f} MB", flush=True)
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
            expert_param_names.append(name)

    if not expert_param_names:
        raise RuntimeError(
            "proposal: no expert params detected (no param with allreduce=False). "
            "Is this an MoE model?"
        )

    # DeepSpeed PipelineModule prefixes names with an execution-list position,
    # not a model-layer number.  Resolve each expert name through the actual
    # layer object's ``layer_number`` once, then use that canonical map in all
    # storage and recovery paths.
    expert_id_map = build_expert_id_map(
        inner_model,
        num_layers=num_layers,
    )
    if set(expert_id_map) != set(expert_param_names):
        missing = sorted(set(expert_param_names) - set(expert_id_map))
        extra = sorted(set(expert_id_map) - set(expert_param_names))
        raise RuntimeError(
            "canonical expert-id map does not match rank-local expert "
            f"parameters: missing={missing} extra={extra}"
        )

    if topology.is_global_rank0:
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
    get_controller()._cp_dir.mkdir(parents=True, exist_ok=True)

    get_controller()._config = {
        "selection_mode": get_controller()._proposal_config.selection_mode,
        "topk_ratio": get_controller()._proposal_config.topk_ratio,
        "e_hat_proxy": get_controller()._proposal_config.e_hat_proxy,
        "lfcl_interval_iters": get_controller()._proposal_config.lfcl_interval_iters,
        "lfcl_fraction_pct": get_controller()._proposal_config.lfcl_fraction_pct,
        "nonexpert_fc_W": get_controller()._proposal_config.nonexpert_fc_w,
        "async_queue": get_controller()._proposal_config.async_queue,
        "cp_mode": get_controller()._proposal_config.cp_mode,
        "num_experts": num_experts_scalar,
        "num_experts_spec": num_experts_spec,
        "proposal_config": get_controller()._proposal_config,
        "topology": topology,
        "rank_namespace": topology.namespace,
        "expert_id_map": expert_id_map,
        "writes_expert": topology.writes_expert,
        "writes_nonexpert": topology.writes_nonexpert,
        "defer_stage0_emission": bool(_defer_stage0_emission),
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
        and bool(topology.writes_expert)
    )
    cp_target = str(get_controller()._proposal_config.cp_target)
    # The backlog tracer is environment-gated but must report the effective
    # CLI/config target, including callers that did not export this variable.
    os.environ["PROPOSAL_CP_TARGET"] = cp_target
    get_controller()._config["expert_lossy_commit_enabled"] = (
        (not _test_skip_expert_commit) and topology.writes_expert
    )
    get_controller()._config["nonexpert_commit_enabled"] = (
        (not _test_skip_nonexpert_commit)
        and (topology.writes_nonexpert or cp_target == "test_dram")
    )

    get_controller()._checkpoint_workers = CheckpointWorkerPool(
        non_expert_queue_size=get_controller()._config["async_queue"],
        expert_queue_size=max(get_controller()._config["async_queue"] * 4, 16),
    )
    _install_pipeline_optimizer_fence(model)

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
            expert_id_map=expert_id_map,
        )
        get_controller()._lossy_cp.test_expert_selection_only = bool(
            _test_expert_selection_only,
        )
        # Eager-allocate the per-BF16-group flat buffers and selection scratch.
        # Required so the warmup-boundary H refresh has valid metadata before
        # the first lossy checkpoint step runs on expert_launcher.
        get_controller()._lossy_cp._init_lossy_flat_layout(inner_opt)
    else:
        get_controller()._lossy_cp = None

    # ---- Process groups for proposal reductions ----
    # Per-iteration cutoff/feedback aggregates are tiny. Keep them on a CPU
    # group for every multi-GPU topology so EP, PP, and EP+PP share one
    # ordering contract and proposal workers never inject independent NCCL
    # collectives into the training pipeline.
    _proposal_pg = None
    _proposal_cpu_pg = None
    _proposal_world_size = 1
    try:
        if (torch.distributed.is_available()
                and torch.distributed.is_initialized()):
            _proposal_world_size = int(len(topology.selection_ranks))
            _proposal_ranks = list(topology.selection_ranks)
            if _proposal_world_size > 1:
                _proposal_pg = torch.distributed.new_group(
                    ranks=_proposal_ranks,
                )
                _proposal_cpu_pg = torch.distributed.new_group(
                    ranks=_proposal_ranks,
                    backend="gloo",
                )
                if _should_write():
                    print(
                        f"[proposal] proposal process groups created "
                        f"(world_size={_proposal_world_size})",
                        flush=True,
                    )
    except Exception as _exc:
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and int(torch.distributed.get_world_size()) > 1
        ):
            raise RuntimeError(
                "proposal: multi-rank mode requires proposal CPU process "
                "groups for global LFCL planning and strict tau."
            ) from _exc
        _proposal_pg = None
        _proposal_cpu_pg = None
        _proposal_world_size = 1
        if _should_write():
            print(
                f"[proposal] proposal process-group init failed; "
                f"falling back to local cutoff: {_exc}",
                flush=True,
            )
    if get_controller()._lossy_cp is not None:
        get_controller()._lossy_cp._proposal_pg = _proposal_pg
        get_controller()._lossy_cp._proposal_cpu_pg = _proposal_cpu_pg
        get_controller()._lossy_cp._proposal_world_size = _proposal_world_size
        get_controller()._lossy_cp._topology = topology
    get_controller()._proposal_cpu_pg = _proposal_cpu_pg
    get_controller()._proposal_world_size = int(_proposal_world_size)

    topk_ratio = float(
        get_controller()._proposal_config.topk_ratio,
    )
    pinned_aos_capacity_ratio = float(topk_ratio)
    pinned_aos_capacity_tolerance = float(
        get_controller()._proposal_config.pinned_aos_capacity_tolerance,
    )

    # CP layout: split non-expert + expert into separate subdirectories.
    # DP>1 stores non-expert shards in a PP-common directory, while expert
    # shards use the canonical dp0 namespace just like DeepSpeed MoE.
    # H_g is GPU-only (BF16 in ``_h_quantile_values``). Recovery
    # rebuilds it on the next FD refresh; no disk persist path.
    _nonexpert_cp_dir = topology.nonexpert_dir(get_controller()._cp_dir)
    _expert_cp_dir = topology.expert_dir(get_controller()._cp_dir)
    get_controller()._config["nonexpert_cp_dir"] = _nonexpert_cp_dir
    get_controller()._config["expert_cp_dir"] = _expert_cp_dir

    # MoC-System baseline uses round_robin single-expert sequential FC.
    fc_schedule = (
        "sequential" if get_controller()._config["cp_mode"] == "sequential" else "proposal"
    )
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
        cp_target=cp_target,
        pinned_aos_capacity_ratio=float(pinned_aos_capacity_ratio),
        pinned_aos_capacity_tolerance=float(pinned_aos_capacity_tolerance),
        enable_expert_state=bool(topology.writes_expert),
        commit_expert_lossy_records=(
            get_controller()._config["expert_lossy_commit_enabled"]
        ),
        commit_nonexpert_fc=(
            get_controller()._config["nonexpert_commit_enabled"]
        ),
        test_skip_expert_d2h=bool(_test_skip_expert_d2h),
        test_skip_nonexpert_d2h=bool(
            _test_skip_nonexpert_d2h
            or (not topology.writes_nonexpert and cp_target != "test_dram")
        ),
        lfcl_capacity_world_size=int(_proposal_world_size),
        expert_id_map=expert_id_map,
        defer_initial_expert_bootstrap=(
            int(get_controller()._warmup_end_iter) > 0
        ),
        nonexpert_shard_rank=(
            int(topology.nonexpert_shard_rank) if cp_target != "test_dram" else 0
        ),
        nonexpert_shard_count=(
            int(topology.nonexpert_shard_count) if cp_target != "test_dram" else 1
        ),
    )
    # Wire the ExpertState sink into the lossy checkpoint path. Each proposal
    # step then routes selected expert records into ExpertState.write_for_iter.
    # Sequential mode never calls the lossy path, so it skips this wiring.
    if get_controller()._config["expert_lossy_enabled"]:
        get_controller().attach_expert_state(get_controller()._fc.expert_state)
        if (
            get_controller()._fc.expert_state is not None
            and torch.cuda.is_available()
        ):
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

    # Expert LFCL compaction materializes plan-target live GPU p/m/v into a
    # pinned LFCL payload; ExpertState owns durable file commit.
    # ``fc_manager.fire_nonexpert_fc_if_due()`` only handles non-expert FC.
    if _test_skip_expert_d2h or not bool(topology.writes_expert):
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
            commit_payload=get_controller()._config["expert_lossy_commit_enabled"],
        )
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

    # POST-INIT VRAM diff vs baseline. Large selection buffers have already
    # been initialized, so this captures the steady V2 allocation footprint.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        _post_init_alloc = torch.cuda.memory_allocated()
        _baseline = globals().get("_baseline_alloc_mb", 0.0) * 1e6
        delta = (_post_init_alloc - _baseline) / 1e6
        if _should_write():
            print(
                f"[proposal] POST-INIT VRAM: alloc={_post_init_alloc/1e6:.1f} MB "
                f"(Δ from baseline = {delta:+.1f} MB; lazy buffers add at first active step)",
                flush=True,
            )
    # Audit: post_init record (R1 baseline). dump_breakdown captures
    # lossy checkpoint GPU buffers if initialized.
    from .telemetry import diagnostics as _proposal_diagnostics
    _proposal_diagnostics.vram_audit_step(
        "post_init", dump_breakdown=True, lossy_cp=get_controller()._lossy_cp,
    )


def proposal_activate_after_warmup(model, iteration: int) -> bool:
    """Create the first learned recovery baseline at the LR-warmup boundary."""

    del model
    ctrl = get_controller()
    iteration = int(iteration)
    if not ctrl._enabled:
        return False
    if ctrl._active:
        return False
    warmup_end = int(ctrl._warmup_end_iter)
    if iteration < warmup_end:
        return False
    if iteration != warmup_end:
        raise RuntimeError(
            "proposal activation missed the LR-warmup boundary: "
            f"iteration={iteration} warmup_end={warmup_end}"
        )
    workers = ctrl._checkpoint_workers
    if workers is not None:
        workers.drain()
    if ctrl._lossy_cp is not None:
        ctrl._lossy_cp._init_lossy_flat_layout(ctrl._config["_inner_opt"])
        if not getattr(ctrl._lossy_cp, "_flat_layout_done", False):
            raise RuntimeError(
                "proposal activation requires initialized Adam/layout state"
            )
    ready_event = ctrl.acquire_dispatch_ready_event()
    if ready_event is not None:
        ready_event.record(torch.cuda.current_stream())
    fc = ctrl._fc
    if fc is None:
        raise RuntimeError("proposal activation requires FCManager")
    expert_count = fc.bootstrap_expert_recovery_baseline(step=iteration)
    nonexpert_saved = fc.bootstrap_nonexpert_recovery_baseline(
        step=iteration,
        ready_event=ready_event,
    )
    if workers is not None:
        workers.drain()
    if ctrl._lossy_cp is not None:
        ctrl._lossy_cp.reset_checkpoint_score_state_from_live(
            next_global_step=iteration + 1,
        )
    ctrl._active = True
    if _should_write():
        print(
            f"[proposal] activated after LR warmup @iter{iteration}: "
            f"expert_baselines={int(expert_count)} "
            f"nonexpert_baseline={int(bool(nonexpert_saved))}",
            flush=True,
        )
    return True


def proposal_prepare_after_optimizer() -> None:
    """Finish init-once layout allocation after Adam state first exists."""

    ctrl = get_controller()
    lossy_cp = ctrl._lossy_cp
    if not ctrl._enabled or ctrl._active or lossy_cp is None:
        return
    if not getattr(lossy_cp, "_flat_layout_done", False):
        lossy_cp._init_lossy_flat_layout(ctrl._config["_inner_opt"])


def proposal_record_loss(step: int, loss: Optional[float]) -> None:
    """Record the latest training loss and proposal diagnostics."""
    del step
    if not get_controller()._enabled or not get_controller()._active:
        return
    if get_controller()._pending_fault_record is not None and loss is not None:
        get_controller()._pending_fault_record["post_fault_loss"] = float(loss)
        _flush_fault_record(get_controller()._pending_fault_record)
        get_controller()._pending_fault_record = None


def proposal_run_independent_oracle(model, data_iterator, iteration: int) -> None:
    """Run telemetry-only rollback oracle for the current checkpoint image."""

    ctrl = get_controller()
    if (
        not ctrl._enabled
        or not ctrl._active
        or ctrl._lossy_cp is None
        or ctrl._proposal_config is None
    ):
        if os.environ.get("PROPOSAL_INDEPENDENT_ORACLE_DEBUG", "0") == "1" and _should_write():
            print(
                "[proposal] independent_oracle skip: "
                f"enabled={bool(ctrl._enabled)} "
                f"lossy_cp={ctrl._lossy_cp is not None} "
                f"config={ctrl._proposal_config is not None} "
                f"iter={int(iteration)}",
                flush=True,
            )
        return
    try:
        from .telemetry.independent_oracle import independent_oracle_enabled
        if not independent_oracle_enabled(ctrl._proposal_config, int(iteration)):
            if os.environ.get("PROPOSAL_INDEPENDENT_ORACLE_DEBUG", "0") == "1" and _should_write():
                iters = tuple(int(x) for x in getattr(ctrl._proposal_config, "independent_oracle_iters", ()) or ())
                print(
                    "[proposal] independent_oracle skip: "
                    f"disabled_for_iter iter={int(iteration)} iters={iters}",
                    flush=True,
                )
            return
    except Exception as exc:
        if _should_write():
            print(f"[proposal] independent_oracle: enable check failed: {exc}", flush=True)
        return
    if os.environ.get("PROPOSAL_INDEPENDENT_ORACLE_DEBUG", "0") == "1" and _should_write():
        print(f"[proposal] independent_oracle run: iter={int(iteration)}", flush=True)

    # Normal V2 dispatches iter T at the beginning of iter T+1.  The oracle
    # measures a fault at iter T, so make the current post-step image visible
    # before evaluating rollback loss.
    if int(iteration) > int(getattr(ctrl, "_last_dispatched_iter_T", -1)):
        proposal_dispatch_post_iter(
            model,
            iter_T=int(iteration),
            sync_processes=False,
        )
    if ctrl._checkpoint_workers is not None:
        ctrl._checkpoint_workers.drain()

    from .telemetry.independent_oracle import run_independent_oracle
    oracle_data_iterator = ctrl._config.get(
        "_oracle_dedicated_valid_iter",
        data_iterator,
    )
    run_independent_oracle(
        model=model,
        data_iterator=oracle_data_iterator,
        iteration=int(iteration),
        lossy_cp=ctrl._lossy_cp,
        proposal_config=ctrl._proposal_config,
    )


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
        "proposal_enabled": bool(get_controller()._enabled),
        "tracked_total_mb": float(total_mb),
        "tracked_items": [
            (name, float(size_mb)) for name, size_mb in items if size_mb > 0.01
        ],
    }


def proposal_vram_audit_iter_begin(step: int) -> None:
    from .telemetry import diagnostics as _proposal_diagnostics

    if not _proposal_diagnostics.is_vram_audit_enabled():
        return
    # install_alloc_trace is intentionally called here as well as proposal
    # init: a proposal-off baseline never enters init_proposal(), but still
    # needs the same audit metadata and long-running allocator counters.
    _proposal_diagnostics.install_alloc_trace()
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


def proposal_vram_audit_dispatch_post_iter(model: Any, iter_T: int) -> bool:
    """Run one selected checkpoint dispatch inside exact allocator history.

    A full training step creates hundreds of thousands of unrelated autograd
    allocation events.  The exact R2 audit therefore traces only the proposal
    dispatch/read/D2H region, synchronously and with pipeline deferral disabled.
    Returning ``False`` tells the caller to use the normal asynchronous
    dispatch path.  This audit-only path never changes optimizer state.
    """

    from .telemetry import diagnostics as _proposal_diagnostics

    if not get_controller()._enabled or not get_controller()._active:
        return False
    iter_T = int(iter_T)
    if not _proposal_diagnostics.begin_steady_allocator_history(iter_T):
        return False
    # The training-loop prologue is iteration T+1, while this dispatch and the
    # trace both describe checkpoint state produced by optimizer step T.
    _proposal_diagnostics.set_iter(iter_T)
    try:
        if iter_T <= int(getattr(get_controller(), "_last_dispatched_iter_T", -1)):
            raise RuntimeError(
                "selected allocator-trace iteration was already dispatched; "
                "choose a non-anchor/non-oracle steady iteration "
                f"(iter_T={iter_T})"
            )
        proposal_dispatch_post_iter(
            model,
            iter_T=iter_T,
            sync_processes=True,
            defer_stage0_emission=False,
        )
        if get_controller()._latest_post_step_checkpoint_task is None:
            raise RuntimeError(
                f"selected allocator-trace dispatch created no task (iter_T={iter_T})"
            )
        # Wait for the worker-owned scan and both D2H domains to enqueue on
        # the main stream.  end_allocator_history() then synchronizes once so
        # all frees/segment changes belong to this exact region.
        proposal_wait_d2h_before_optimizer()
    finally:
        try:
            _proposal_diagnostics.end_steady_allocator_history(iter_T)
        finally:
            _proposal_diagnostics.set_iter(iter_T + 1)
    return True


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
        task.dispatch_release_ready.wait()
        # On PP stage 0 this is the first-RecvGrad boundary event, which is
        # later than the post-optimizer dispatch event.  FCManager wires it
        # into the D2H stream without blocking the training thread.
        ready_event = _wait_stage0_emission_boundary(task)
        if get_controller()._fc is not None:
            get_controller()._fc.fire_nonexpert_fc_if_due(
                step=task.iter_T,
                ready_event=ready_event,
                d2h_event_callback=_mark_non_expert_read_ready,
            )
        from .storage import proposal_io as _proposal_io
        _proposal_io.fdp_mark_writer_done(task.iter_T, "nonexpert")
    except BaseException as exc:
        task.non_expert_error = exc
        raise
    finally:
        if task.non_expert_ready_at == 0.0:
            task.non_expert_ready_at = time.perf_counter()
        task.non_expert_read_ready.set()


def _execute_expert_post_iter(task: PostStepCheckpointTask) -> None:
    """Body of the host_writer_expert worker task.

    The worker first plans LFCL compaction, publishes the fired-expert mask to
    lossy selection, then runs lossy direct emission and finally materializes
    the dense LFCL payloads.  The expert worker owns the whole path, so
    single-thread invariants hold without locks.
    """
    def _mark_expert_d2h_event(
        d2h_event: Optional["torch.cuda.Event"],
    ) -> None:
        task.expert_d2h_event = d2h_event

    try:
        ctrl = get_controller()
        task.expert_started_at = time.perf_counter()
        if not bool(ctrl._config.get("writes_expert", True)):
            task.dispatch_release_ready.set()
            from .storage import proposal_io as _proposal_io
            _proposal_io.fdp_mark_writer_done(task.iter_T, "expert")
            task.expert_ready_at = time.perf_counter()
            task.expert_read_ready.set()
            return
        try:
            if bool(getattr(task, "sync_processes", True)):
                _sync_post_iter_dispatch(task.iter_T)
        finally:
            task.dispatch_release_ready.set()
        fired_experts: List[Tuple[int, int]] = []
        if ctrl._expert_lossless_fc is not None:
            fired_experts = ctrl._expert_lossless_fc.plan_if_due(
                task.iter_T,
                process_group=getattr(ctrl._lossy_cp, "_proposal_cpu_pg", None),
                world_size=int(getattr(ctrl._lossy_cp, "_proposal_world_size", 1)),
            )
        if ctrl._lossy_cp is not None:
            publish_fired = getattr(ctrl._lossy_cp, "publish_lfcl_fired_experts", None)
            if callable(publish_fired):
                publish_fired(step=int(task.iter_T), expert_ids=fired_experts)
        if (
            ctrl._config.get("expert_lossy_enabled", True)
            and ctrl._lossy_cp is not None
        ):
            inner_opt = ctrl._config.get("_inner_opt")
            if inner_opt is not None:
                ctrl.step(
                    inner_opt,
                    global_step=task.iter_T,
                    ready_event=task.cuda_event,
                    d2h_event_callback=_mark_expert_d2h_event,
                    emission_release_event=(
                        task.stage0_emission_release
                        if task.defer_stage0_emission else None
                    ),
                    emission_cuda_event=(
                        task.stage0_emission_cuda_event
                        if task.defer_stage0_emission else None
                    ),
                )
        if ctrl._expert_lossless_fc is not None:
            # The lossy path above already gates direct scan/AOS D2H after
            # cutoff.  Select the same boundary explicitly for LFCL D2H as
            # well, including sequential/LFCL-only configurations.
            lfcl_ready_event = _wait_stage0_emission_boundary(task)
            fired_experts = ctrl._expert_lossless_fc.fire_planned(
                task.iter_T,
                fired_experts,
                ready_event=lfcl_ready_event,
            )
            lfcl_d2h_event = getattr(
                ctrl._expert_lossless_fc,
                "last_d2h_event",
                None,
            )
            if lfcl_d2h_event is not None:
                _mark_expert_d2h_event(lfcl_d2h_event)
        if ctrl._lossy_cp is not None:
            fired_by_step = getattr(
                ctrl._lossy_cp,
                "_lfcl_fired_experts_by_step",
                None,
            )
            if fired_by_step is None:
                fired_by_step = {}
                setattr(
                    ctrl._lossy_cp,
                    "_lfcl_fired_experts_by_step",
                    fired_by_step,
                )
            fired_by_step[int(task.iter_T)] = list(fired_experts)
        ctrl._config["_last_fired_experts"] = fired_experts
        from .storage import proposal_io as _proposal_io
        expert_state = (
            getattr(ctrl._fc, "expert_state", None)
            if ctrl._fc is not None else None
        )
        has_pending = (
            callable(getattr(expert_state, "fdp_has_pending_writes", None))
            and expert_state.fdp_has_pending_writes(task.iter_T)
        )
        if not has_pending:
            _proposal_io.fdp_mark_writer_done(task.iter_T, "expert")
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
    task = get_controller()._latest_post_step_checkpoint_task
    if task is not None:
        current = torch.cuda.current_stream()
        task.non_expert_read_ready.wait()
        task.expert_read_ready.wait()
        if task.non_expert_error is not None:
            raise RuntimeError(
                f"proposal non-expert checkpoint task failed at "
                f"iter {task.iter_T}"
            ) from task.non_expert_error
        if task.expert_error is not None:
            raise RuntimeError(
                f"proposal expert checkpoint task failed at iter {task.iter_T}"
            ) from task.expert_error
        if task.non_expert_d2h_event is not None:
            current.wait_event(task.non_expert_d2h_event)
        if task.expert_d2h_event is not None:
            current.wait_event(task.expert_d2h_event)
        if task is get_controller()._latest_post_step_checkpoint_task:
            get_controller()._latest_post_step_checkpoint_task = None


def _sync_post_iter_dispatch(iter_T: int) -> None:
    """Release all proposal owner ranks at the same post-optimizer boundary."""

    ctrl = get_controller()
    world_size = int(getattr(ctrl, "_proposal_world_size", 1) or 1)
    if world_size <= 1:
        return
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        raise RuntimeError(
            "proposal: synchronized dispatch requires initialized distributed "
            f"state (iter={int(iter_T)})"
        )
    cpu_pg = getattr(ctrl, "_proposal_cpu_pg", None)
    if cpu_pg is None:
        raise RuntimeError(
            "proposal: synchronized dispatch requires proposal CPU process "
            f"group (iter={int(iter_T)})"
        )
    torch.distributed.barrier(group=cpu_pg)


def _proposal_cpu_barrier(where: str) -> None:
    """Synchronize proposal ranks without launching NCCL work on GPUs."""

    del where
    ctrl = get_controller()
    world_size = int(getattr(ctrl, "_proposal_world_size", 1) or 1)
    if world_size <= 1:
        return
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        return
    cpu_pg = getattr(ctrl, "_proposal_cpu_pg", None)
    if cpu_pg is None:
        raise RuntimeError("proposal multi-rank sync requires CPU/Gloo group")
    torch.distributed.barrier(group=cpu_pg)


def _proposal_drain_cuda(where: str) -> None:
    """Finish outstanding CUDA work before proposal mutates recovered state."""

    del where
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _acquire_gpu_replay_lanes():
    """Lease fixed expert replay lanes; topology stays elsewhere."""

    from contextlib import nullcontext

    ctrl = get_controller()
    fc = getattr(ctrl, "_fc", None)
    expert_state = getattr(fc, "expert_state", None) if fc is not None else None
    if (
        expert_state is None
        or not torch.cuda.is_available()
        or getattr(fc, "cp_target", "disk") == "test_dram"
    ):
        return nullcontext(None)
    return expert_state.acquire_recovery_lanes()


def _submit_post_step_checkpoint_task(task: PostStepCheckpointTask) -> None:
    """Submit a post-step checkpoint task to the proposal workers."""

    if get_controller()._checkpoint_workers is None:
        # A deferred task requires the training thread to advance to
        # RecvGrad, which is impossible while running this fallback inline.
        task.defer_stage0_emission = False
        task.stage0_emission_cuda_event = None
        task.stage0_emission_release.set()
        _execute_expert_post_iter(task)
        _execute_non_expert_post_iter(task)
        return
    get_controller()._checkpoint_workers.submit_non_expert(_execute_non_expert_post_iter, task)
    get_controller()._checkpoint_workers.submit_expert(_execute_expert_post_iter, task)


def proposal_dispatch_post_iter(
    model,
    iter_T: int,
    *,
    sync_processes: bool = True,
    defer_stage0_emission: bool = False,
) -> None:
    """Submit iter ``T`` checkpoint work before next forward.

    The caller invokes this as the first action of iteration ``T+1`` after
    iter ``T``'s optimizer step has completed. The task is submitted
    immediately; no forward hook or arm/release stash is used.
    """

    if not get_controller()._enabled or not get_controller()._active:
        return
    iter_T = int(iter_T)
    if iter_T <= int(get_controller()._warmup_end_iter):
        return
    if iter_T <= int(getattr(get_controller(), "_last_dispatched_iter_T", -1)):
        return
    from .storage import proposal_io as _proposal_io
    _prev_iter_T = int(getattr(get_controller(), "_last_dispatched_iter_T", -1))
    if (
        _prev_iter_T >= 0
        and _proposal_io.backend_kind() == "fdp"
        and _proposal_io.fdp_metadata_mmap_enabled()
    ):
        if get_controller()._checkpoint_workers is not None:
            get_controller()._checkpoint_workers.drain_non_expert()
            get_controller()._checkpoint_workers.drain_expert()
            get_controller()._checkpoint_workers.drain_expert_write()
        _proposal_io.fdp_mark_writer_done(_prev_iter_T, "all")
        _proposal_cpu_barrier("fdp_mark_writer_done")
    _proposal_io.fdp_begin_checkpoint_iter(iter_T)
    event = get_controller().acquire_dispatch_ready_event()
    if event is not None:
        event.record()
    ctrl = get_controller()
    defer_stage0 = bool(
        defer_stage0_emission
        and ctrl._config.get("defer_stage0_emission", False)
    )
    task = PostStepCheckpointTask(
        cuda_event=event,
        iter_T=iter_T,
        sync_processes=bool(sync_processes),
        defer_stage0_emission=defer_stage0,
        stage0_emission_cuda_event=(
            ctrl._stage0_emission_cuda_event if defer_stage0 else None
        ),
    )
    if not defer_stage0:
        task.stage0_emission_release.set()
    if not bool(sync_processes):
        task.dispatch_release_ready.set()
    get_controller()._latest_post_step_checkpoint_task = task
    _submit_post_step_checkpoint_task(task)
    get_controller()._last_dispatched_iter_T = iter_T
    from .telemetry import diagnostics as _d
    _d._proposal_post_iter_diagnostics(model, int(iter_T))


def proposal_finalize_post_iter(model, iteration: int) -> None:
    """Commit the final optimizer result when no next iteration exists.

    Normal training dispatches iteration ``T`` immediately before iteration
    ``T+1``.  At training termination there is no next-loop prologue, so this
    terminal-only hook performs the same dispatch without pipeline deferral,
    then drains every checkpoint domain before model save/evaluation starts.
    The dispatch's existing iteration guard makes this idempotent when a final
    H refresh, oracle, or fault path already made the state visible.
    """

    ctrl = get_controller()
    iteration = int(iteration)
    if not ctrl._enabled or not ctrl._active or iteration <= 0:
        return

    proposal_dispatch_post_iter(
        model,
        iter_T=iteration,
        sync_processes=True,
        defer_stage0_emission=False,
    )
    workers = ctrl._checkpoint_workers
    local_failure: Optional[str] = None
    try:
        if workers is not None:
            workers.drain()
    except BaseException as exc:
        local_failure = f"worker drain: {type(exc).__name__}: {exc}"

    task = ctrl._latest_post_step_checkpoint_task
    if task is not None and local_failure is None:
        if task.non_expert_error is not None:
            exc = task.non_expert_error
            local_failure = (
                f"non-expert task at iter {task.iter_T}: "
                f"{type(exc).__name__}: {exc}"
            )
        elif task.expert_error is not None:
            exc = task.expert_error
            local_failure = (
                f"expert task at iter {task.iter_T}: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            # Terminal-only host synchronization is outside the steady
            # training path and guarantees pinned payloads are complete before
            # save/eval.
            try:
                for event in (
                    task.non_expert_d2h_event,
                    task.expert_d2h_event,
                ):
                    if event is not None:
                        event.synchronize()
            except BaseException as exc:
                local_failure = (
                    f"D2H completion at iter {task.iter_T}: "
                    f"{type(exc).__name__}: {exc}"
                )
    ctrl._latest_post_step_checkpoint_task = None

    # Every rank must make the same success/failure decision before the next
    # collective BF16 save.  Raising a rank-local worker error before this
    # Gloo collective would strand the healthy ranks in model save.
    world_size = int(getattr(ctrl, "_proposal_world_size", 1) or 1)
    failures: List[Optional[str]] = [local_failure]
    if world_size > 1:
        cpu_pg = getattr(ctrl, "_proposal_cpu_pg", None)
        if (
            cpu_pg is None
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "proposal terminal finalize requires the proposal CPU/Gloo "
                "process group"
            )
        failures = [None] * world_size
        torch.distributed.all_gather_object(
            failures,
            local_failure,
            group=cpu_pg,
        )
    for rank, failure in enumerate(failures):
        if failure:
            raise RuntimeError(
                f"proposal terminal checkpoint failed on rank {rank}: "
                f"{failure}"
            )


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
    _proposal_drain_cuda("proposal_recover_pre")
    _proposal_cpu_barrier("proposal_recover_pre")
    with _acquire_gpu_replay_lanes() as gpu_replay_lanes:
        try:
            diag = recover(
                model=inner_model,
                optimizer=inner_opt,
                cp_dir=get_controller()._cp_dir,
                t_fault=t_fault,
                num_experts=num_experts,
                expert_state_image=_expert_state_image,
                nonexpert_state_image=_nonexpert_state_image,
                nonexpert_dir=get_controller()._config.get("nonexpert_cp_dir"),
                expert_dir=get_controller()._config.get("expert_cp_dir"),
                expected_expert_ids=(
                    get_controller()._fc.expert_state.expert_ids()
                    if (
                        get_controller()._fc is not None
                        and get_controller()._fc.expert_state is not None
                    )
                    else None
                ),
                expert_id_map=get_controller()._config.get("expert_id_map"),
                gpu_replay_lanes=gpu_replay_lanes,
            )
        finally:
            _proposal_drain_cuda("proposal_recover_post")
    _proposal_cpu_barrier("proposal_recover")
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
        "expert_expected_count": getattr(diag, "expert_expected_count", 0),
        "expert_missing_count": getattr(diag, "expert_missing_count", 0),
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


def _poison_live_state_before_synthetic_recover(
    model: torch.nn.Module,
    optimizer: Any,
) -> Dict[str, int]:
    """Erase local live state so synthetic fault recovery cannot reuse it.

    A real crash restarts from checkpoint material only.  The in-process fault
    injector must therefore wipe the current model/optimizer state before
    calling recover(); otherwise missing expert payloads silently leave exact
    live tensors in place and make the rollback loss meaningless.
    """

    from .refresh.fp32_state import get_mapper

    mapper = get_mapper()
    n_params = 0
    n_master = 0
    n_state = 0
    with torch.no_grad():
        for p in model.parameters():
            if not getattr(p, "requires_grad", False):
                continue
            n_params += 1
            p.data.zero_()
            if mapper is not None:
                local = mapper.get_local_param_view(p)
                if local is not None:
                    local.zero_()
                    n_master += 1
                for key in ("exp_avg", "exp_avg_sq"):
                    slot = mapper.get_local_state_view(p, key)
                    if slot is not None:
                        slot.zero_()
                        n_state += 1
            elif optimizer is not None:
                state = optimizer.state.get(p, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    slot = state.get(key)
                    if isinstance(slot, torch.Tensor):
                        slot.zero_()
                        n_state += 1
    return {
        "params": int(n_params),
        "master": int(n_master),
        "state": int(n_state),
    }


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
        proposal_dispatch_post_iter(
            model,
            iter_T=int(step),
            sync_processes=False,
        )
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

    _proposal_drain_cuda("fault_pre_recover")
    _proposal_cpu_barrier("fault_pre_recover")

    t0 = time.perf_counter()
    inner_model = get_controller()._config["_inner_model"]
    inner_opt = get_controller()._config["_inner_opt"]
    num_experts = get_controller()._config["num_experts"]
    poison_stats = _poison_live_state_before_synthetic_recover(
        inner_model,
        inner_opt,
    )
    _proposal_drain_cuda("fault_poison_live_state")
    _proposal_cpu_barrier("fault_poison_live_state")
    if _should_write():
        print(
            "[proposal] synthetic_fault_poison: "
            f"params={poison_stats['params']} "
            f"master={poison_stats['master']} "
            f"state={poison_stats['state']}",
            flush=True,
        )
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
    with _acquire_gpu_replay_lanes() as gpu_replay_lanes:
        try:
            diag = recover(
                model=inner_model,
                optimizer=inner_opt,
                cp_dir=get_controller()._cp_dir,
                t_fault=int(step),
                num_experts=num_experts,
                expert_state_image=_expert_state_image,
                nonexpert_state_image=_nonexpert_state_image,
                nonexpert_dir=get_controller()._config.get("nonexpert_cp_dir"),
                expert_dir=get_controller()._config.get("expert_cp_dir"),
                expected_expert_ids=(
                    get_controller()._fc.expert_state.expert_ids()
                    if (
                        get_controller()._fc is not None
                        and get_controller()._fc.expert_state is not None
                    )
                    else None
                ),
                expert_id_map=get_controller()._config.get("expert_id_map"),
                gpu_replay_lanes=gpu_replay_lanes,
            )
        finally:
            _proposal_drain_cuda("fault_post_recover")
    _proposal_cpu_barrier("fault_post_recover")
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
        "expert_expected_count": int(getattr(diag, "expert_expected_count", 0)),
        "expert_missing_count": int(getattr(diag, "expert_missing_count", 0)),
        "B_max_lost_iter": B_max,
        "poisoned_params": int(poison_stats["params"]),
        "poisoned_master": int(poison_stats["master"]),
        "poisoned_state": int(poison_stats["state"]),
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


def proposal_set_oracle_data_iter(it: Any) -> None:
    get_controller()._config["_oracle_dedicated_valid_iter"] = it


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
    # An exception may reach shutdown before stage 0 posts its first RecvGrad.
    # Release a pending task before FCManager/worker drains to avoid a teardown
    # deadlock.  Normal completed tasks make this a no-op.
    try:
        _release_stage0_emission(record_cuda_event=True)
    except Exception:
        # _release_stage0_emission sets the host gate in ``finally``.  CUDA is
        # already failing, so continue draining CPU workers and metadata.
        pass
    # Drain launcher domains before ExpertState closes files or releases slab
    # leases.  This is also required on exception teardown, where a task may
    # have been submitted but not yet reached expert_writer.
    if get_controller()._checkpoint_workers is not None:
        get_controller()._checkpoint_workers.drain()
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
    try:
        from .telemetry.io_backlog import close_io_backlog_trace
        close_io_backlog_trace()
    except Exception:
        pass
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
