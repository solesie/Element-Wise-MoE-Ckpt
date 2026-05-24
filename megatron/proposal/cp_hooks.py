"""Integration hooks between Megatron-DeepSpeed's training loop and the proposal CP pipeline.

The training thread runs **only** training (fwd/bwd/optimizer). All proposal CP
work — non-expert lossless FC, expert lossless FC, expert lossy FC append, and
the proposal-mode importance tracker compute — is performed on the two domain
workers owned by :class:`AsyncWriter`. The handshake is the
``proposal_arm_post_iter_fc`` / ``proposal_release_post_iter_fc`` pair:

* :func:`proposal_arm_post_iter_fc` runs at the end of ``train_step`` on the
  training thread. It records a ``cuda.Event`` that signals iter T's
  ``optimizer.step`` completion and stashes a small task descriptor.
* :func:`proposal_release_post_iter_fc` is fired by a ``register_forward_hook``
  on the unwrapped model at the END of the first micro-batch's forward in iter
  T+1 (between fwd and bwd of mb0). It pushes the descriptor onto the two
  domain queues and returns immediately. Subsequent micro-batches see the
  descriptor already consumed (``_armed_post_iter_fc_task is None``) and the
  hook is a no-op. The d2h is queued on ``d2h_stream`` while main_stream
  proceeds with mb0's backward — backward contains no host-side ``.to('cuda')``
  triggers so the PCIe copy-engine FIFO has no contender, and the 44 ms d2h
  completes well inside the ~1.8 s remaining bwd/fwd/bwd/.../opt window.

Each domain worker thread is bound to one CUDA device via ``set_device`` at
startup, synchronises on the recorded cuda_event before reading any optimizer
state, and runs FIFO so single-thread invariants (e.g. ``_pending_dispatch``
inside :class:`ImportanceTracker`) hold without locks.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .async_writer import AsyncWriter
from ._ds_state import Fp32StateMapper, set_mapper
from .fc_manager import FCManager, _detect_expert_layout, DS_EXPERT_IDX_RE
from .importance_tracker import ImportanceTracker
from .recovery import recover


@dataclass
class _PostIterFCTask:
    """Descriptor handed from the training thread to the domain workers.

    The event captures iter T's ``optimizer.step`` completion; both workers
    synchronise on it before reading any FP32 master / Adam state. ``iter_T``
    is forwarded into the schedule predicates (W-boundary for non-expert,
    D-boundary for expert lossless FC).
    """

    cuda_event: Optional["torch.cuda.Event"]
    iter_T: int


_tracker: Optional[ImportanceTracker] = None
_fc: Optional[FCManager] = None
_writer: Optional[AsyncWriter] = None
_expert_lossless_fc: Optional[Any] = None  # ExpertLosslessFC, lazy import
_armed_post_iter_fc_task: Optional[_PostIterFCTask] = None
_enabled: bool = False
_cp_dir: Optional[Path] = None


_config: Dict[str, Any] = {}

# Exp 2 metric plumbing.
_pending_fault_record: Optional[Dict[str, Any]] = None
_cpu_mem_samples: List[Dict[str, Any]] = []
_cpu_mem_every_n: int = 50

try:
    import psutil as _psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


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


def _get_ds_engine_optimizer(model):
    """Return the DeepSpeed engine's wrapping optimizer (FP16/BF16/ZeRO),
    NOT the inner torch optimizer. Needed to map model param → FP32 master
    slice under the FP16/BF16_Optimizer flat-master layout."""
    m = model[0] if isinstance(model, (list, tuple)) else model
    return getattr(m, "optimizer", None)


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
    global _tracker, _fc, _writer, _enabled, _cp_dir, _config
    global _expert_lossless_fc
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

    _enabled = bool(getattr(args, "proposal_enable", False))
    if not _enabled:
        return

    # CUDA stream priority promotion: per atomic per-micro-batch
    # observation (PROPOSAL_BENCH_MB + nsys profile), iter-time gap
    # comes from main_stream's kernels (stream 7 in nsys) being
    # SM-contended by worker thread's tracker_stream + d2h_stream
    # heavy GPU work (Triton kernels, index_select). On A100 the
    # default stream priority is 0 (least). cudaDeviceGetStreamPriorityRange
    # returns leastPriority=0, greatestPriority=-5. Promoting the
    # main compute path to a higher-priority stream (e.g. -5) lets
    # the scheduler preempt low-priority worker kernels, so main's
    # forward/backward kernels run at baseline speed even while
    # worker streams are active.
    # Env-gated (PROPOSAL_PROMOTE_MAIN_STREAM=1) so the change can
    # be A/B compared against the baseline (priority-0 default).
    import os as _os_prio
    _promote = _os_prio.environ.get(
        "PROPOSAL_PROMOTE_MAIN_STREAM", "") in ("1", "true", "True")
    if _promote and torch.cuda.is_available():
        # PyTorch's "current_stream" is what Megatron/DeepSpeed enqueue
        # forward/backward/optimizer kernels on. We create a high-
        # priority stream and set it as the current_stream for the
        # main training thread. Worker threads (host_writer_*) still
        # set their own per-thread current_stream in _DomainWorker._run,
        # which we leave at default priority 0 — so they yield to
        # this high-priority main stream under SM contention.
        _hp_stream = torch.cuda.Stream(
            device=torch.cuda.current_device(), priority=-5,
        )
        torch.cuda.set_stream(_hp_stream)
        _config["_main_hp_stream"] = _hp_stream
        print(
            f"[proposal] PROMOTE_MAIN_STREAM: created high-priority "
            f"main_stream (priority=-5) and set as current_stream "
            f"on training thread.",
            flush=True,
        )

    # PR-MoE support: args.num_experts may be a list of per-layer counts.
    # FCManager re-derives per-layer counts from observed expert tensors;
    # we pass the raw spec through and use max() as the scalar metadata for
    # tracker/checkpoint managers (which only use it for sizing/logging).
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
    engine_opt = _get_ds_engine_optimizer(model)
    # Baseline VRAM snapshot before any proposal-side allocation.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        _baseline_alloc = torch.cuda.memory_allocated()
        _baseline_reserved = torch.cuda.memory_reserved()
        print(f"[proposal] BASELINE VRAM (before proposal init): "
              f"alloc={_baseline_alloc/1e6:.1f} MB, "
              f"reserved={_baseline_reserved/1e6:.1f} MB", flush=True)
        _config.setdefault("_baseline_alloc", _baseline_alloc) if False else None
        # Stash on module so we can compare after init.
        globals()["_baseline_alloc_mb"] = _baseline_alloc / 1e6
        # Audit: capture baseline + install alloc trace patches (env-gated).
        from . import _vram_audit
        _vram_audit.install_alloc_trace()
        _vram_audit.vram_audit_step("baseline_pre_init")
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


    # B2: layout must be 'separated' for DeepSpeed MoE.
    layout = _detect_expert_layout(
        list(inner_model.named_parameters()), num_experts_scalar,
    )
    if layout != "separated":
        raise RuntimeError(
            f"proposal: expected DeepSpeed 'separated' expert layout, detected {layout!r}."
        )

    _cp_dir = Path(cp_root)
    if _should_write():
        _cp_dir.mkdir(parents=True, exist_ok=True)

    # cp_mode (CLAUDE.md § Expert 저장 전략):
    #   proposal (default): Lossy FC (append-only log, top-K% elements each
    #       iter) + round-robin F% expert lossless FC every D iter.
    #   sequential: MoC-System baseline — layer-wise round-robin full expert
    #       FC, no element-wise lossy FC.
    cp_mode = getattr(args, "proposal_cp_mode", "proposal")
    if cp_mode not in ("proposal", "sequential"):
        raise ValueError(
            f"proposal_cp_mode must be 'proposal'|'sequential', got {cp_mode!r}."
        )
    _config = {
        "selection_mode": getattr(args, "proposal_selection_mode", "hessian"),
        "topk_ratio": float(getattr(args, "proposal_topk_ratio", 0.05)),
        "expert_fc_D": (
            int(getattr(args, "proposal_expert_fc_D"))
            if getattr(args, "proposal_expert_fc_D", None)
            else None
        ),
        "nonexpert_fc_W": int(getattr(args, "proposal_nonexpert_fc_W", 10)),
        "async_queue": int(getattr(args, "proposal_async_queue", 5)),
        "cp_mode": cp_mode,
        "lossless_fc_fraction_pct": float(
            getattr(args, "proposal_lossless_fc_fraction", 2.0)
        ),
        "num_experts": num_experts_scalar,
        "num_experts_spec": num_experts_spec,
    }

    _writer = (
        AsyncWriter(
            non_expert_queue_size=_config["async_queue"],
            expert_queue_size=max(_config["async_queue"] * 4, 16),
        )
        if _should_write() else None
    )

    _tracker = ImportanceTracker(
        inner_model,
        topk_ratio=_config["topk_ratio"],
        expert_only=True,
        selection_mode=_config["selection_mode"],
        num_experts=num_experts_scalar,
        random_seed=int(getattr(args, "seed", 0) or 0),
        async_writer=_writer,
        block_quant_g=int(getattr(args, "proposal_block_quant_g", 1)),
    )
    # Eager-allocate the per-BF16-group flat buffers (dg, h_g, exc, sel
    # scratch) so the iter-1 Hessian refresh has valid target views to
    # write into. With the new arch the first tracker.step runs on the
    # host_writer_expert worker AFTER iter 1's H refresh on the training
    # thread; if we left this lazy, ``_fold_h`` would silently skip every
    # tensor (no view yet) and ``_g_fast_h_g_bf16_flat`` would stay zero
    # for the full first refresh cycle.
    _tracker._init_accum_sq_flat(inner_opt)

    # CP layout: split non-expert + expert into separate subdirectories
    # so they can be mounted on different devices.
    #   <cp>/non_expert/  — non-expert full FC (W-scheduled)
    #   <cp>/expert/      — expert rolling (lossy-FC) + lossless FC (D-scheduled)
    # H_g is GPU-only (BF16 in ``_g_fast_h_g_bf16_flat``). Recovery
    # rebuilds it on the next FD refresh; no disk persist path.
    _nonexpert_cp_dir = _cp_dir / "non_expert"
    _expert_cp_dir = _cp_dir / "expert"

    # MoC-System baseline uses round_robin single-expert sequential FC.
    fc_schedule = (
        "sequential" if _config["cp_mode"] == "sequential" else "proposal"
    )
    if _should_write():
        _fc = FCManager(
            model=inner_model,
            optimizer=inner_opt,
            output_dir=_nonexpert_cp_dir,
            expert_output_dir=_expert_cp_dir,
            num_experts=num_experts_spec,
            expert_fc_D=_config["expert_fc_D"],
            nonexpert_fc_W=_config["nonexpert_fc_W"],
            async_writer=_writer,
            schedule=fc_schedule,
            lossless_fc_fraction_pct=_config["lossless_fc_fraction_pct"],
            fresh_expert_state=True,
            # cp_target: 'disk' (default) or 'dram'. DRAM mode swaps in
            # DramExpertState — see fc_manager.py and dram_expert_state.py.
            cp_target=str(getattr(args, "proposal_cp_target", "disk")),
        )
        # Wire the ExpertState sink into the tracker. tracker.step() then
        # routes each expert's top-K selections into lossy_fc_append. For
        # sequential (MoC) mode, tracker.step is never called so we skip
        # the wiring.
        _disable_expert_state = bool(getattr(args, "proposal_disable_rolling", False))
        if _config["cp_mode"] != "sequential" and not _disable_expert_state:
            _tracker.set_expert_state(_fc.expert_state)
        if _disable_expert_state:
            print("[proposal] expert state DISABLED (debug flag); expert "
                  "lossy-FC skipped entirely.", flush=True)

        # B2 separation: expert lossless FC is now its own module driven by
        # tracker_stream (not main stream). It shares fc_manager's collect
        # helper for DRY but is otherwise independent.
        # ``fc_manager.fire_nonexpert_fc_if_due()`` only handles non-expert
        # FC after this split.
        from .expert_lossless_fc import ExpertLosslessFC
        # Bind the collect helper as a callable that returns
        # (payload, total_bytes); ExpertLosslessFC pulls cuda_event from
        # fc._last_collect_cuda_event after the call.
        def _collect_helper(tnames):
            # private=True: each fired expert gets its OWN pinned buffer
            # so the AsyncWriter raw worker can consume payloads at its
            # own pace without the next _fire_one overwriting in-flight
            # data. Steady-state alloc cost is small (pinned caching
            # allocator recycles freed buffers).
            # named_params=None: the callee lazily builds it ONLY on cache
            # miss.
            # header_block_bytes / chunk_pad_bytes / preserve_order:
            # reserve the 4 KiB LFCL header at the front of the pinned
            # buffer, pad the alloc up to the SSD-passthrough chunk so
            # the C++ writer can submit chunk-aligned, and write p|m|v
            # in schema (caller) order so the on-disk byte layout
            # matches what recovery expects.
            from .proposal_io import HEADER_BLOCK_BYTES as _HB, PIN_CHUNK_BYTES as _CB
            return _fc._collect_param_payload(
                tnames, None, private=True,
                header_block_bytes=_HB,
                chunk_pad_bytes=_CB,
                preserve_order=True,
            )
        _expert_lossless_fc = ExpertLosslessFC(
            model=inner_model,
            all_expert_ids=_fc._all_expert_ids,
            separated_tensors_by_slot=_fc._separated_tensors_by_slot,
            sorted_layers=_fc._sorted_layers,
            expert_state=_fc.expert_state,
            async_writer=_writer,
            expert_output_dir=_expert_cp_dir,
            keep_last_per_expert=_fc.keep_last_per_expert,
            D=_fc.D,
            schedule=fc_schedule,
            lossless_fc_fraction_pct=_config["lossless_fc_fraction_pct"],
            tracker=_tracker,
            tracker_stream=None,  # filled lazily on first fire (tracker
                                  # creates its stream on first step)
            collect_payload_fn=_collect_helper,
            event_owner=_fc,  # for retrieving _last_collect_cuda_event
        )
    else:
        _fc = None
        _expert_lossless_fc = None

    _config["_inner_model"] = inner_model
    _config["_inner_opt"] = inner_opt
    # Register a forward hook on the model so that the previous-iter
    # release fires at the END of the first micro-batch's forward
    # (instead of at the start, before any forward kernels run).
    # Timing rationale: at iter-start release, the d2h on d2h_stream
    # queues on the GPU PCIe copy engine FIFO ahead of the small H2D
    # transfers that PyTorch's `tensor.to('cuda')` issues during
    # forward (e.g. `torch.tensor(0.0, device=)` in Megatron
    # transformer's MoE path × 36 calls/iter). Each such .to() ends
    # with cudaStreamSynchronize which waits for the queue to drain
    # — directly observed in perf stack (cuStreamSynchronize /
    # __clock_gettime in libcuda.so) blocking main thread ~44ms.
    # By delaying release to *after* mb0's forward, d2h instead
    # overlaps with mb0's backward (which contains no H2D triggers),
    # and finishes well before mb1's forward starts. Worker overlap
    # window is reduced by ~mb0-fwd time (~230 ms) but file-write
    # (~900 ms) still fits comfortably inside the remaining iter.
    if inner_model is not None:
        # Remove any prior hook (idempotent re-init).
        _prev = _config.get("_release_hook_handle")
        if _prev is not None:
            try:
                _prev.remove()
            except Exception:
                pass
        # Release at end of first mb forward: worker overlaps with mb0_bwd
        # (the only contention window allowed by design — single mb_bwd of
        # HBM/SM contention; subsequent mb_bwds run at baseline speed).
        def _release_on_first_fwd(module, _inputs, _output):
            import os as _os_dbg_rel
            _dbg_rel = _os_dbg_rel.environ.get("PROPOSAL_RELEASE_DBG", "") in ("1", "true", "True")
            if _dbg_rel:
                torch.cuda.nvtx.range_push(
                    f"hook_fire(training={module.training},anchor={bool(_config.get('_in_anchor_refresh'))},task={'!N' if _armed_post_iter_fc_task is not None else 'N'})"
                )
                torch.cuda.nvtx.range_pop()
                import time as _t_dbg
                _t_now = _t_dbg.monotonic()
                if not hasattr(_release_on_first_fwd, "_t0"):
                    _release_on_first_fwd._t0 = _t_now
                _rel = (_t_now - _release_on_first_fwd._t0) * 1000.0
                print(f"[hook_fire] t={_rel:.1f}ms training={module.training} "
                      f"anchor={bool(_config.get('_in_anchor_refresh'))} "
                      f"task={'!None' if _armed_post_iter_fc_task is not None else 'None'}",
                      flush=True)
            # Eval-mode forwards bypass the iter-level arm/release
            # contract; do not consume the armed task there.
            if not module.training:
                return
            # FD anchor's +ε/-ε forwards run with .train() too — skip
            # release so they don't fire writer tasks against tracker
            # views currently offloaded to CPU.
            if _config.get("_in_anchor_refresh"):
                return
            # `_armed_post_iter_fc_task` is the single gate. release
            # pops it (→ None) — subsequent micro-batches see None
            # and the call is a no-op.
            if _armed_post_iter_fc_task is not None:
                if _dbg_rel:
                    torch.cuda.nvtx.range_push("release_fire")
                proposal_release_post_iter_fc(None)
                if _dbg_rel:
                    torch.cuda.nvtx.range_pop()

        _config["_release_hook_handle"] = (
            inner_model.register_forward_hook(_release_on_first_fwd)
        )
    # Persist fault log path early so proposal_shutdown dumps land next to it
    # even if no faults fire (or training crashes before first fault).
    fault_log = getattr(args, "proposal_fault_log", None)
    if fault_log:
        _config["_fault_log_path"] = fault_log
    _fc_iters = getattr(args, "proposal_fisher_coverage_iters", None) or []
    _config["_fisher_coverage_iters"] = set(int(x) for x in _fc_iters)
    _config["_fisher_coverage_done_set"] = set()
    _config["_fisher_coverage_records"] = []

    # finite_diff config.
    _config["_finite_diff_eps"] = getattr(args, "proposal_finite_diff_eps", None)
    # Diagnostic env override: PROPOSAL_FD_EPS=<float> bypasses autotune
    # and forces a fixed eps. Used to test BF16 noise-floor hypotheses
    # (autotune's iter-1 chosen eps may be below BF16 ULP at output scale).
    import os as _os_eps
    _eps_env = _os_eps.environ.get("PROPOSAL_FD_EPS", "")
    if _eps_env:
        try:
            _config["_finite_diff_eps"] = float(_eps_env)
        except ValueError:
            pass
    _config["_fd_multi_k_track"] = bool(
        getattr(args, "proposal_fd_multi_k_track", False)
    )
    _config["_fd_multi_k_values"] = list(
        getattr(args, "proposal_fd_multi_k_values", [1, 2, 4, 8, 16, 32])
        or [1, 2, 4, 8, 16, 32]
    )
    _config["_fd_multi_k_log"] = getattr(args, "proposal_fd_multi_k_log", None)
    _config["_finite_diff_flip_cutoff"] = float(
        getattr(args, "proposal_finite_diff_flip_cutoff", 100.0)
    )
    # Anchor uses half of one iteration's worth of micro-batches.
    # mb_per_iter = global_batch_size / micro_batch_size.
    # Coverage rationale: empirically n_anchor_batches = mb_per_iter/2 gives
    # ≥0.93 H coverage on exp_etc while staying under 50% of the per-iter
    # forward+backward budget. See CLAUDE.md `Anchor batch count`.
    _mb_per_iter = max(1, int(args.global_batch_size) // max(1, int(args.micro_batch_size)))
    _config["_anchor_n_batches"] = max(1, _mb_per_iter // 2)
    # FD noise probe (diagnostic) — populated only when the CLI flag is set.
    _fd_probe_iters = getattr(args, "proposal_fd_noise_probe_iters", None) or []
    _config["_fd_noise_probe_iters"] = frozenset(int(x) for x in _fd_probe_iters)
    _config["_fd_noise_probe_log"] = getattr(
        args, "proposal_fd_noise_probe_log", None,
    )
    _config["_fd_noise_probe_ref_k"] = int(
        getattr(args, "proposal_fd_noise_probe_ref_k", 32)
    )
    _config["_fd_noise_probe_ref_n_batches"] = int(
        getattr(args, "proposal_fd_noise_probe_ref_n_batches", 10)
    )
    _config["_fd_noise_probe_g_blk"] = int(
        getattr(args, "proposal_fd_noise_probe_g_blk", 1)
    )
    # Diagnostic: per-refresh expert_coverage / flip_skip_ratio / eps.
    _config["_h_diag_records"] = []
    # ANALYSIS: non-expert |H| probe — comparison of expert vs non-expert |H|
    # distribution at score_diag_iters; retained for measurement purposes.

    # POST-INIT VRAM diff vs baseline (run before _init_accum_sq_flat fires
    # at first step; this only measures init-time alloc — chunk_buf, dg_flat,
    # h_g_bf16, exc_bool, sel_buf are lazy-built on first tracker.step,
    # so the bigger jump appears after iter 1).
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        _post_init_alloc = torch.cuda.memory_allocated()
        _baseline = globals().get("_baseline_alloc_mb", 0.0) * 1e6
        delta = (_post_init_alloc - _baseline) / 1e6
        print(f"[proposal] POST-INIT VRAM: alloc={_post_init_alloc/1e6:.1f} MB "
              f"(Δ from baseline = {delta:+.1f} MB; lazy buffers add at iter 1)",
              flush=True)
    # Audit: post_init record (R1 baseline). dump_breakdown captures
    # tracker.gpu_memory_breakdown() if tracker is initialized.
    from . import _vram_audit
    _vram_audit.vram_audit_step(
        "post_init", dump_breakdown=True, tracker=_tracker,
    )


def proposal_record_iter_loss(step: int, loss_value: Optional[float]) -> None:
    """Called from the pretrain loop after train_step returns. Resolves
    post-fault loss and samples CPU memory.
    """
    global _pending_fault_record
    if not _enabled:
        return
    if _pending_fault_record is not None and loss_value is not None:
        _pending_fault_record["post_fault_loss"] = float(loss_value)
        _flush_fault_record(_pending_fault_record)
        _pending_fault_record = None
    if _HAS_PSUTIL and step % _cpu_mem_every_n == 0:
        try:
            vm = _psutil.virtual_memory()
            proc = _psutil.Process()
            _cpu_mem_samples.append({
                "step": int(step),
                "rss_mb": proc.memory_info().rss / 1024 / 1024,
                "sys_used_mb": vm.used / 1024 / 1024,
                "sys_avail_mb": vm.available / 1024 / 1024,
            })
        except Exception:
            pass


def _execute_non_expert_post_iter(task: _PostIterFCTask) -> None:
    """Body of the host_writer_non_expert worker task.

    The worker has CUDA context set; we sync on the iter T optimizer.step
    event then defer scheduling (W-boundary) to ``FCManager``. The
    in-flight FC's D2H + disk/DRAM write all run on this thread, FIFO.
    """
    if task.cuda_event is not None:
        task.cuda_event.synchronize()
    if _fc is not None:
        _fc.fire_nonexpert_fc_if_due(step=task.iter_T)


def _execute_expert_post_iter(task: _PostIterFCTask) -> None:
    """Body of the host_writer_expert worker task.

    Two responsibilities, both single-threaded on this worker:
      * fire the schedule-due expert lossless FCs (layer-RR in sequential
        mode, F%-RR in proposal mode). ``ExpertLosslessFC.fire_if_due``
        internally launches the D2H copies on ``d2h_stream``.
      * in proposal mode, run ``ImportanceTracker.step``. This dispatches
        the prev iter's pending payload (D2H on ``d2h_stream``) and
        launches this iter's scoring compute on ``tracker_stream``. The
        ``_pending_dispatch`` 1-iter pipeline state is mutated only here,
        so single-thread invariants hold without locks.
    """
    if task.cuda_event is not None:
        task.cuda_event.synchronize()
    fired_experts: List[Tuple[int, int]] = []
    if _expert_lossless_fc is not None:
        if _expert_lossless_fc._tracker_stream is None:
            _expert_lossless_fc._tracker_stream = getattr(
                _tracker, "_tracker_stream", None,
            )
        fired_experts = _expert_lossless_fc.fire_if_due(task.iter_T)
    _config["_last_fired_experts"] = fired_experts
    if _config["cp_mode"] != "sequential" and _tracker is not None:
        inner_opt = _config.get("_inner_opt")
        if inner_opt is not None:
            _tracker.step(
                inner_opt,
                global_step=task.iter_T,
                lossless_fc_eids=fired_experts,
            )


def proposal_fence_before_optimizer() -> None:
    """Order this iter's optimizer.step AFTER any in-flight d2h on
    ``d2h_stream`` (worker's master-read for non-expert FC payload).

    Worker's H2D launches at iter T+1's first mb forward (from the
    release hook) and finishes well within the mb_bwd window; by the
    time fence is reached the wait is typically no-op (us-level).
    Protects against opt.step's master-write racing d2h_stream's
    master-read when worker pickup or PCIe is unexpectedly delayed.
    """
    if not _enabled or not torch.cuda.is_available():
        return
    import os as _os_probe_fence
    _probe = _os_probe_fence.environ.get("PROPOSAL_MAIN_PROBE", "") in ("1", "true", "True")
    import time as _time_probe
    _t0 = _time_probe.perf_counter() if _probe else 0
    from .async_writer import get_d2h_stream
    d2h = get_d2h_stream()
    if d2h is not None:
        torch.cuda.current_stream().wait_stream(d2h)
    if _probe:
        _t1 = _time_probe.perf_counter()
        if (_t1 - _t0) * 1000 > 0.5:
            print(f"[probe] fence: {(_t1-_t0)*1000:.2f}ms", flush=True)


def proposal_arm_post_iter_fc(model, iter_T: int) -> None:
    """Arm the iter T post-optimizer FC work for next-iter execution.

    Called from the training thread at the tail of ``train_step``, right
    after ``model[0].step()`` returns. Records a ``cuda.Event`` on the
    main stream — the event signals iter T's ``optimizer.step``
    completion to the domain workers when they synchronise on it. The
    descriptor is then stashed for :func:`proposal_release_post_iter_fc`
    to consume at the END of the first micro-batch's forward in iter
    T+1 (via the ``register_forward_hook`` on the unwrapped model).

    Diagnostics (per-iter probe, GPU memory report, Fisher coverage)
    capture this iter's training state and are emitted inline because
    they read the same FP32 master / H_g views as the worker tasks; the
    worker has not started yet so there is no race.
    """
    global _armed_post_iter_fc_task
    if not _enabled or _tracker is None:
        return
    import os as _os_probe_arm
    _probe = _os_probe_arm.environ.get("PROPOSAL_MAIN_PROBE", "") in ("1", "true", "True")
    import time as _time_probe
    _t0 = _time_probe.perf_counter() if _probe else 0
    # CPU vs wall: thread_time = main thread's user+sys CPU time. If main
    # thread is blocked on GIL or sync, wall >> thread_time. Report per-iter.
    _cpu_iter_probe = _os_probe_arm.environ.get(
        "PROPOSAL_CPU_PROBE", "") in ("1", "true", "True")
    if _cpu_iter_probe:
        _now_wall = _time_probe.monotonic()
        _now_cpu = _time_probe.thread_time()
        _prev = _config.get("_cpu_probe_prev")
        if _prev is not None:
            dw = _now_wall - _prev[0]
            dc = _now_cpu - _prev[1]
            ratio = (dc / dw * 100) if dw > 0 else 0
            print(f"[cpu_probe] iter={iter_T}  wall={dw*1000:.0f}ms  main_thread_cpu={dc*1000:.0f}ms  busy_ratio={ratio:.1f}%",
                  flush=True)
        _config["_cpu_probe_prev"] = (_now_wall, _now_cpu)
    import os as _os_dbg_arm
    _dbg_arm = _os_dbg_arm.environ.get("PROPOSAL_RELEASE_DBG", "") in ("1", "true", "True")
    if _dbg_arm:
        torch.cuda.nvtx.range_push(f"arm(iter_T={iter_T})")
        import time as _t_dbg_arm
        _t_now_a = _t_dbg_arm.monotonic()
        if not hasattr(proposal_arm_post_iter_fc, "_t0"):
            proposal_arm_post_iter_fc._t0 = _t_now_a
        _rel_a = (_t_now_a - proposal_arm_post_iter_fc._t0) * 1000.0
        print(f"[arm] t={_rel_a:.1f}ms iter_T={iter_T}", flush=True)
    event = None
    if torch.cuda.is_available():
        event = torch.cuda.Event()
        event.record()
    _armed_post_iter_fc_task = _PostIterFCTask(
        cuda_event=event, iter_T=int(iter_T),
    )
    if _dbg_arm:
        torch.cuda.nvtx.range_pop()
    if _probe:
        _t1 = _time_probe.perf_counter()
        _proposal_post_iter_diagnostics(model, int(iter_T))
        _t2 = _time_probe.perf_counter()
        body_ms = (_t1-_t0)*1000
        diag_ms = (_t2-_t1)*1000
        if body_ms + diag_ms > 0.5:
            print(f"[probe] arm@iter{iter_T}: body={body_ms:.2f}ms diag={diag_ms:.2f}ms", flush=True)
    else:
        _proposal_post_iter_diagnostics(model, int(iter_T))


def proposal_release_post_iter_fc(model) -> None:
    """Release the armed task onto the two domain workers.

    Fired by the ``register_forward_hook`` on the unwrapped model at
    the END of the first micro-batch's forward in iter T+1. Both
    submissions are ``queue.put`` calls returning in ~1 microsecond,
    so the training thread proceeds straight into mb0's backward on
    ``main_stream``. Each worker independently synchronises on the
    iter T optimizer.step event before reading optimizer state."""
    global _armed_post_iter_fc_task
    task = _armed_post_iter_fc_task
    _armed_post_iter_fc_task = None
    if task is None:
        return
    import os as _os_probe_rel
    _probe = _os_probe_rel.environ.get("PROPOSAL_MAIN_PROBE", "") in ("1", "true", "True")
    import time as _time_probe
    _t0 = _time_probe.perf_counter() if _probe else 0
    if _writer is None:
        # Inline fallback (no worker pool) — runs both phases here.
        _execute_non_expert_post_iter(task)
        _execute_expert_post_iter(task)
        if _probe:
            _t1 = _time_probe.perf_counter()
            if (_t1-_t0)*1000 > 0.5:
                print(f"[probe] release(inline)@iter{task.iter_T}: {(_t1-_t0)*1000:.2f}ms", flush=True)
        return

    _writer.submit_non_expert(_execute_non_expert_post_iter, task)
    _writer.submit_expert(_execute_expert_post_iter, task)
    if _probe:
        _t1 = _time_probe.perf_counter()
        if (_t1-_t0)*1000 > 0.5:
            print(f"[probe] release@iter{task.iter_T}: {(_t1-_t0)*1000:.2f}ms", flush=True)


def _proposal_post_iter_diagnostics(model, step: int) -> None:
    """Internal stage E: per-iter probes and observability emit.

    Called from the tail of ``proposal_dc_record``. All work here is
    either env-gated (peritr group probe, fisher coverage) or runs
    every 10 iters only (AsyncWriter / GPU mem report).
    """
    if not _enabled or _tracker is None:
        return
    seq_mode = _config["cp_mode"] == "sequential"
    inner_opt = _config["_inner_opt"]

    # === PER-ITER PROBE (env-flagged, measurement-only) ===========
    # Two independent measurements per probe iter:
    #
    # (A) "Element-wise (optimal)": maintain per-element residual that
    #     accumulates δ² since last save. Each iter, select top-K by
    #     H × residual_wise → captured_score_wise = Σ(H × residual_wise
    #     of selected) → reset residual_wise[selected] = 0.
    #
    # (B) "Element pool only": same fold, but selection by H × δ²_thisiter
    #     (production single-iter signal). Maintain SEPARATE residual.
    #     captured_score_pool = Σ(H × residual_pool of selected at this
    #     iter) → reset residual_pool[selected] = 0.
    #
    # Output: captured_score_wise vs captured_score_pool per iter.
    # Selection signal differs (accumulated optimal vs single-iter),
    # state evolves separately. Window-start resets all state.
    #
    # Probe is MEASUREMENT-ONLY — does not touch production tracker or
    # parameters. Reads ``_tracker._H_g`` and FP32 master weights.
    #
    # Env (sampling pattern):
    #   PROPOSAL_PERITER_GROUP_PROBE_START=N    window start iter
    #   PROPOSAL_PERITER_GROUP_PROBE_PERIOD=P   interval (0 = single window)
    #   PROPOSAL_PERITER_GROUP_PROBE_LEN=M      window length
    #   PROPOSAL_PERITER_GROUP_K_PCT=0.04       K-budget fraction
    # NOTE: first iter of each window has no prev θ → no measurement.
    import os as _os_pi
    _pi_start = int(_os_pi.environ.get("PROPOSAL_PERITER_GROUP_PROBE_START", "0"))
    _pi_len = int(_os_pi.environ.get("PROPOSAL_PERITER_GROUP_PROBE_LEN", "0"))
    _pi_period = int(_os_pi.environ.get("PROPOSAL_PERITER_GROUP_PROBE_PERIOD", "0"))
    if _pi_period > 0 and _pi_len > 0:
        if step >= _pi_start:
            offset = (step - _pi_start) % _pi_period
            _pi_in_window = offset < _pi_len
            if offset == 0:
                _config.pop("_peritr_theta_snap", None)
                _config.pop("_peritr_residual_wise", None)
                _config.pop("_peritr_residual_pool", None)
                _config.pop("_peritr_residual_uni", None)
        else:
            _pi_in_window = False
    elif _pi_len > 0:
        _pi_in_window = (_pi_start <= step < _pi_start + _pi_len)
        if step == _pi_start:
            _config.pop("_peritr_theta_snap", None)
            _config.pop("_peritr_residual_wise", None)
            _config.pop("_peritr_residual_pool", None)
            _config.pop("_peritr_residual_uni", None)
    else:
        _pi_in_window = False
    if _pi_in_window:
        try:
            from ._ds_state import get_mapper as _gm_pi
            _mp_pi = _gm_pi()
            _kpct_pi = float(_os_pi.environ.get(
                "PROPOSAL_PERITER_GROUP_K_PCT", "0.04"))
            tracked_pi = _tracker._tracked_names if hasattr(
                _tracker, "_tracked_names") else None
            if tracked_pi is not None:
                # Build flat H (element-level) + θ_now
                h_list_pi = []
                theta_now_pi = []
                # Probe always reads H_g from production at its current
                # block_quant_g granularity, then broadcasts back to
                # element resolution for fair element-level comparison.
                prod_g = max(1, int(getattr(_tracker, "block_quant_g", 1)))
                for nm in tracked_pi:
                    p_pi = _tracker._name_to_param.get(nm)
                    if p_pi is None:
                        continue
                    h_g_view = _tracker._H_g.get(nm)
                    if h_g_view is None:
                        continue
                    h_elem = h_g_view.detach().to(torch.float32).repeat_interleave(prod_g)
                    n_t = h_elem.numel()
                    if n_t == 0:
                        continue
                    h_list_pi.append(h_elem)
                    fp32_now = (_mp_pi.get_param(p_pi)
                                if _mp_pi is not None
                                else p_pi.data.to(torch.float32))
                    theta_now_pi.append((nm, fp32_now.detach().clone(), n_t))
                if not h_list_pi:
                    return None
                H_all_pi = torch.cat(h_list_pi)
                prev_pi = _config.get("_peritr_theta_snap")
                if prev_pi is not None:
                    delta_list = []
                    for nm, t_now, n_t in theta_now_pi:
                        if nm in prev_pi:
                            d = (t_now - prev_pi[nm]).reshape(-1)[:n_t]
                            delta_list.append(d)
                    if len(delta_list) == len(theta_now_pi):
                        delta_signed = torch.cat(delta_list)
                        # δ² for this iter only (PER-ITER, no accumulator).
                        delta_sq = delta_signed ** 2
                        n_total = delta_sq.numel()
                        k_select = max(1, int(_kpct_pi * n_total))
                        device = delta_sq.device
                        # Per-algorithm residual (reset on save = production
                        # fold semantic). Each algorithm maintains its OWN
                        # residual since its OWN saved sets differ.
                        res_wise = _config.get("_peritr_residual_wise")
                        res_pool = _config.get("_peritr_residual_pool")
                        res_uni = _config.get("_peritr_residual_uni")
                        if res_wise is None or res_wise.numel() != n_total:
                            res_wise = torch.zeros(n_total, dtype=torch.float32, device=device)
                        if res_pool is None or res_pool.numel() != n_total:
                            res_pool = torch.zeros(n_total, dtype=torch.float32, device=device)
                        if res_uni is None or res_uni.numel() != n_total:
                            res_uni = torch.zeros(n_total, dtype=torch.float32, device=device)
                        # 1) update residuals with this iter's δ²
                        res_wise.add_(delta_sq)
                        res_pool.add_(delta_sq)
                        res_uni.add_(delta_sq)
                        # 2) Element-wise (optimal): select by H × residual_wise
                        score_wise = H_all_pi * res_wise
                        th_wise = float(torch.kthvalue(
                            score_wise, n_total - k_select + 1).values.item())
                        saved_wise = score_wise >= th_wise
                        captured_wise = float(score_wise[saved_wise].sum().item())
                        # 3) Element pool only: select by H × δ²_thisiter (single iter)
                        score_pool_signal = H_all_pi * delta_sq
                        th_pool = float(torch.kthvalue(
                            score_pool_signal, n_total - k_select + 1).values.item())
                        saved_pool = score_pool_signal >= th_pool
                        captured_pool = float((H_all_pi * res_pool)[saved_pool].sum().item())
                        # 4) Unified at probe_g: select by max(element pool
                        #    signal, group residual mean signal) — production
                        #    unified semantic.
                        g_uni = int(_os_pi.environ.get(
                            "PROPOSAL_PERITER_UNIFIED_G", "8"))
                        ng_uni = n_total // g_uni
                        H_g_uni = H_all_pi.view(ng_uni, g_uni).mean(dim=1)
                        res_uni_g_mean = res_uni.view(ng_uni, g_uni).mean(dim=1)
                        elem_sig = H_all_pi * delta_sq  # element-level single-iter
                        group_sig_per_elem = (H_g_uni * res_uni_g_mean).repeat_interleave(g_uni)
                        m_score = torch.maximum(elem_sig, group_sig_per_elem)
                        th_uni = float(torch.kthvalue(
                            m_score, n_total - k_select + 1).values.item())
                        saved_uni = m_score >= th_uni
                        captured_uni = float((H_all_pi * res_uni)[saved_uni].sum().item())
                        # 5) reset selected residuals (= production fold)
                        res_wise[saved_wise] = 0
                        res_pool[saved_pool] = 0
                        res_uni[saved_uni] = 0
                        # 6) write state back
                        _config["_peritr_residual_wise"] = res_wise
                        _config["_peritr_residual_pool"] = res_pool
                        _config["_peritr_residual_uni"] = res_uni
                        n_saved_wise = int(saved_wise.sum().item())
                        n_saved_pool = int(saved_pool.sum().item())
                        n_saved_uni = int(saved_uni.sum().item())
                        ratio_pool_over_wise = (
                            captured_pool / max(captured_wise, 1e-30)
                        )
                        ratio_uni_over_wise = (
                            captured_uni / max(captured_wise, 1e-30)
                        )
                        if _should_write():
                            print(
                                f"[proposal] peritr_probe K={_kpct_pi:.3f} "
                                f"g_uni={g_uni} @iter{step}: "
                                f"captured_wise={captured_wise:.4e} "
                                f"captured_pool={captured_pool:.4e} "
                                f"captured_uni={captured_uni:.4e} "
                                f"pool/wise={ratio_pool_over_wise:.4f} "
                                f"uni/wise={ratio_uni_over_wise:.4f} "
                                f"n_saved_wise={n_saved_wise} "
                                f"n_saved_pool={n_saved_pool} "
                                f"n_saved_uni={n_saved_uni}",
                                flush=True,
                            )
                # Save θ snapshot for next iter's per-iter delta.
                _config["_peritr_theta_snap"] = {
                    nm: t for nm, t, _ in theta_now_pi
                }
        except Exception as _piexc:
            if _should_write():
                import traceback as _tb_pi
                print(f"[proposal] peritr_probe FAILED @iter{step}: "
                      f"{_piexc}\n{_tb_pi.format_exc()}", flush=True)

    # Diagnostic: every 10 iters, report worker queue state. Per-iter
    # phase timings disappeared along with the synchronous PRE/POST
    # hooks — the new architecture moves all of that off the training
    # thread, so what matters now is queue depth (steady-state should
    # show pending ≤ 1 on both workers; growing depth means the writer
    # is falling behind the training rate).
    #
    # Default OFF for production. Enable with PROPOSAL_DISPATCH_DBG=1
    # (shares the gate with importance_tracker's dispatch_dbg/coverage).
    import os as _os_diag
    _diag_on = _os_diag.environ.get("PROPOSAL_DISPATCH_DBG", "") in ("1", "true", "TRUE")
    _diag_every = int(_os_diag.environ.get("PROPOSAL_DISPATCH_DBG_EVERY", "10") or "10")
    if _diag_on and _writer is not None and step > 0 and step % _diag_every == 0:
        s = _writer.summary()
        ne = s["non_expert"]
        ex = s["expert"]
        print(
            f"[proposal] AsyncWriter@iter{step}: "
            f"non_expert pending={ne['pending']} "
            f"max={ne['max_queue_depth']} "
            f"subs={ne['submissions']} comps={ne['completions']}  "
            f"expert pending={ex['pending']} "
            f"max={ex['max_queue_depth']} "
            f"subs={ex['submissions']} comps={ex['completions']}",
            flush=True,
        )
        # GPU memory breakdown vs cuda.memory_allocated.
        try:
            if torch.cuda.is_available() and hasattr(_tracker, "gpu_memory_breakdown"):
                torch.cuda.synchronize()
                cuda_alloc = torch.cuda.memory_allocated() / 1e6
                tracker_total, items = _tracker.gpu_memory_breakdown()
                # Add fc_manager's GPU buffers.
                fc_total = 0.0
                if _fc is not None:
                    for attr in ["_collect_stage_gpu"]:
                        t = getattr(_fc, attr, None)
                        if isinstance(t, torch.Tensor) and t.is_cuda:
                            sz = t.element_size() * t.numel() / 1e6
                            fc_total += sz
                            items.append((f"fc.{attr}", sz))
                # Add expert_state / dram_expert_state GPU buffers
                # (segment_pipeline staging + AOS pipeline state). The
                # actual staging tensors live on the AOS pipeline's
                # ``_segment_pipeline._gpu_bufs`` (double-buffered).
                ex_total = 0.0
                _expert_state = getattr(_fc, "_expert_state", None) if _fc is not None else None
                if _expert_state is not None:
                    pipe = getattr(_expert_state, "_aos_pipeline", None)
                    if pipe is not None:
                        seg_pipe = getattr(pipe, "_segment_pipeline", None)
                        for i, gb in enumerate(
                                getattr(seg_pipe, "_gpu_bufs", []) or []):
                            if isinstance(gb, torch.Tensor) and gb.is_cuda:
                                sz = gb.element_size() * gb.numel() / 1e6
                                ex_total += sz
                                items.append((f"seg_pipe.gpu_buf[{i}]", sz))
                # Add ExpertLosslessFC GPU buffers.
                elfc_total = 0.0
                if _expert_lossless_fc is not None:
                    for attr in ["_collect_stage_gpu", "_collect_p_gpu",
                                 "_collect_m_gpu", "_collect_v_gpu"]:
                        t = getattr(_expert_lossless_fc, attr, None)
                        if isinstance(t, torch.Tensor) and t.is_cuda:
                            sz = t.element_size() * t.numel() / 1e6
                            elfc_total += sz
                            items.append((f"elfc.{attr}", sz))
                items_str = " ".join(f"{n}={v:.1f}" for n, v in items if v > 0.5)
                _baseline_mb = globals().get("_baseline_alloc_mb", 0.0)
                added = cuda_alloc - _baseline_mb
                tracked_total = tracker_total + fc_total + ex_total + elfc_total
                # Diagnostic: dump cuda memory summary on selected iter
                # (untracked spike origin tracing)
                import os as _os_dump
                _dump_at = int(_os_dump.environ.get("PROPOSAL_VRAM_DUMP_AT", "0") or "0")
                if _dump_at > 0 and step == _dump_at:
                    print(f"[proposal] cuda.memory_summary() @iter{step}:", flush=True)
                    print(torch.cuda.memory_summary(abbreviated=False), flush=True)
                untracked = added - tracked_total
                # Peak (max) memory tracking — captures TRANSIENT spikes
                # caused by dispatch's per-iter GPU intermediates that
                # ``alloc`` (steady-state allocated) misses. Reset after
                # each log so each report's peak reflects the iters
                # since the previous log only.
                cuda_peak = torch.cuda.max_memory_allocated() / 1e6
                peak_added = cuda_peak - _baseline_mb
                # Audit BEFORE reset_peak so the per-iter peak is captured.
                from . import _vram_audit
                _vram_audit.set_iter(int(step))
                _vram_audit.vram_audit_step(
                    "iter_end", dump_breakdown=True, tracker=_tracker,
                    extra={
                        "untracked_mb": float(untracked),
                        "tracked_total_mb": float(tracked_total),
                    },
                )
                torch.cuda.reset_peak_memory_stats()
                print(
                    f"[proposal] GPU mem @iter{step}: "
                    f"alloc={cuda_alloc:.1f}MB Δfrom_baseline={added:+.1f}MB "
                    f"peak={cuda_peak:.1f}MB Δpeak={peak_added:+.1f}MB "
                    f"proposal_buffers={tracked_total:.1f}MB "
                    f"untracked={untracked:+.1f}MB | {items_str}",
                    flush=True,
                )
                # When untracked is large, dump biggest live GPU tensors
                # by walking gc.get_objects(). Helps find leaks that the
                # static breakdown misses (DeepSpeed internal caches,
                # per-iter tensors held by C++ engine refs, etc.).
                import os as _os_dump
                if _os_dump.environ.get("PROPOSAL_MEM_DUMP", "") in ("1",):
                    import gc as _gc
                    cur_storages: dict = {}  # data_ptr → (size_mb, shape, dtype)
                    for obj in _gc.get_objects():
                        try:
                            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                                sid = obj.untyped_storage().data_ptr()
                                if sid in cur_storages:
                                    continue
                                cur_storages[sid] = (
                                    obj.untyped_storage().nbytes() / 1e6,
                                    list(obj.shape), str(obj.dtype),
                                )
                        except Exception:
                            pass
                    # Diff vs previous snapshot
                    prev_storages = globals().get("_prev_mem_storages", {})
                    new_sids = set(cur_storages.keys()) - set(prev_storages.keys())
                    gone_sids = set(prev_storages.keys()) - set(cur_storages.keys())
                    new_entries = sorted(
                        [(cur_storages[s][0], cur_storages[s][1], cur_storages[s][2]) for s in new_sids],
                        reverse=True,
                    )
                    gone_entries = sorted(
                        [(prev_storages[s][0], prev_storages[s][1], prev_storages[s][2]) for s in gone_sids],
                        reverse=True,
                    )
                    new_total = sum(s for s, *_ in new_entries)
                    gone_total = sum(s for s, *_ in gone_entries)
                    print(f"[proposal] mem diff @iter{step}: "
                          f"NEW={len(new_entries)} (+{new_total:.1f}MB) "
                          f"GONE={len(gone_entries)} (-{gone_total:.1f}MB) "
                          f"NET={new_total - gone_total:+.1f}MB",
                          flush=True)
                    # Bucket new/gone by (size_mb, shape, dtype) signature
                    from collections import Counter as _Ctr_diff
                    new_sig_ctr = _Ctr_diff()
                    for sz, shp, dt in new_entries:
                        new_sig_ctr[(round(sz, 1), tuple(shp), dt)] += 1
                    gone_sig_ctr = _Ctr_diff()
                    for sz, shp, dt in gone_entries:
                        gone_sig_ctr[(round(sz, 1), tuple(shp), dt)] += 1
                    print(f"[proposal] mem diff @iter{step} NEW by signature:", flush=True)
                    for (sz, shp, dt), cnt in sorted(
                        new_sig_ctr.items(), key=lambda x: -x[0][0]*x[1])[:15]:
                        print(f"  +{cnt:3d} × {sz:7.1f} MB  shape={list(shp)} dtype={dt}  total={cnt*sz:.1f}MB",
                              flush=True)
                    if gone_sig_ctr:
                        print(f"[proposal] mem diff @iter{step} GONE by signature:", flush=True)
                        for (sz, shp, dt), cnt in sorted(
                            gone_sig_ctr.items(), key=lambda x: -x[0][0]*x[1])[:15]:
                            print(f"  -{cnt:3d} × {sz:7.1f} MB  shape={list(shp)} dtype={dt}  total={cnt*sz:.1f}MB",
                                  flush=True)
                    globals()["_prev_mem_storages"] = cur_storages
                    # Total snapshot by signature — surfaces UNTRACKED
                    # tensors that live persistently (not visible in diff
                    # at steady state). Compares against the known
                    # proposal_buffers items so the remaining is what
                    # we don't account for.
                    snap_ctr = _Ctr_diff()
                    for (sz_mb, shp, dt) in cur_storages.values():
                        snap_ctr[(round(sz_mb, 1), tuple(shp), dt)] += 1
                    # Bucket and print top size×count
                    top = sorted(
                        snap_ctr.items(),
                        key=lambda x: -x[0][0] * x[1],
                    )[:20]
                    print(f"[proposal] mem snapshot @iter{step} TOP by total:", flush=True)
                    for (sz, shp, dt), cnt in top:
                        if sz * cnt < 1.0:
                            continue
                        print(
                            f"  {cnt:4d} × {sz:7.1f} MB  shape={list(shp)} "
                            f"dtype={dt}  total={cnt*sz:.1f}MB",
                            flush=True,
                        )
                # Tracker phase ms (existing instrumentation: setup/pass1/
                # pass2/dispatch/launch + the new v_mapping). v_mapping is
                # lagged by 1 iter (cuda.Event sync only at next call).
                lpm = getattr(_tracker, "last_phase_ms", None)
                if lpm:
                    parts = " ".join(f"{k}={v:.2f}ms" for k, v in lpm.items())
                    print(f"[proposal] tracker phase @iter{step}: {parts}",
                          flush=True)
        except Exception:
            pass

    # Fisher mass concentration measurement.
    # - differential mode (proposal/random): element-wise C(K) over all expert v_t
    # - sequential mode (MoC): per-expert v_t mass + fired-expert coverage
    fc_iters = _config.get("_fisher_coverage_iters", set())
    fc_done = _config.get("_fisher_coverage_done_set", set())
    if step in fc_iters and step not in fc_done:
        log_path = _config.get("_fault_log_path") or ""
        if log_path:
            base = log_path
            if base.endswith(".faults.jsonl"):
                base = base[: -len(".faults.jsonl")]
            out_path = base + ".fisher_coverage.json"
        else:
            out_path = "/tmp/fisher_coverage.json"
        if seq_mode:
            rec = proposal_measure_moc_fisher_coverage(inner_opt, step, out_path)
        else:
            rec = proposal_measure_fisher_coverage(inner_opt, step, out_path)
        # Append into the running list and rewrite the JSON file.
        records = _config.setdefault("_fisher_coverage_records", [])
        records.append(rec)
        try:
            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                import json as _json
                with open(out_path, "w") as f:
                    _json.dump(records, f, indent=2)
                if seq_mode:
                    fired_cov = rec.get("fired_coverage", 0.0) * 100
                    bestk = rec.get("best_k_experts_coverage", {})
                    bk_str = " ".join(
                        f"{k}:{v*100:.1f}%" for k, v in bestk.items()
                    )
                    print(
                        f"[proposal] MoC Fisher @iter {step}: "
                        f"fired={rec.get('fired_expert')} "
                        f"fired_cov={fired_cov:.2f}%  best_k_experts: {bk_str}",
                        flush=True,
                    )
                else:
                    cov = rec.get("coverage", {})
                    cov_str = " ".join(f"{k}:{v*100:.1f}%" for k, v in cov.items())
                    print(
                        f"[proposal] Fisher coverage @iter {step}: {cov_str}",
                        flush=True,
                    )
        except Exception:
            pass
        fc_done.add(step)


# H_g (BF16 group-level diagonal Hessian) is GPU-only. After a fault,
# recovery's first FD refresh repopulates it; there is no disk persist
# / restore path.


def proposal_recover(model, optimizer, t_fault: int) -> Optional[Dict[str, Any]]:
    if not _enabled or _cp_dir is None:
        return None
    if _writer is not None:
        _writer.drain()
    # Flush the log-structured expert state so end-of-training replay sees
    # the same on-disk snapshot that fault injection would see.
    if _fc is not None and _fc.expert_state is not None:
        try:
            _fc.expert_state.flush()
        except Exception as exc:
            print(
                f"[proposal] WARNING end-of-training expert-state flush failed: {exc}; "
                f"falling back to legacy DC replay.",
                flush=True,
            )
    inner_model = _config["_inner_model"]
    inner_opt = _config["_inner_opt"] if optimizer is None else optimizer
    num_experts = _config["num_experts"]
    # When proposal cp_target=dram, the per-expert state was kept in a
    # host-RAM DramExpertState rather than written to disk. Hand the
    # in-memory state to recover() so the per-expert load short-circuits
    # the disk scan (see recovery.recover()'s dram_expert_state branch).
    _dram_es = None
    if (
        _fc is not None and _fc.expert_state is not None
        and getattr(_fc, "cp_target", "disk") == "dram"
    ):
        _dram_es = _fc.expert_state
    import time as _time
    _t_rec0 = _time.perf_counter()
    diag = recover(
        model=inner_model,
        optimizer=inner_opt,
        cp_dir=_cp_dir,
        t_fault=t_fault,
        num_experts=num_experts,
        dram_expert_state=_dram_es,
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
    log_path = _config.get("_fault_log_path") if _config else None
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
def proposal_measure_fisher_coverage(
    optimizer, step: int, output_path: str,
) -> Dict[str, Any]:
    """Measure Fisher mass concentration C(K) over expert params.

    C(K) = sum of |v_t| of top-K% elements / sum of |v_t| of all expert elements.
    Heavy-tail distributions yield C(K) >> K. Dumped as JSON for later analysis.
    """
    import json
    import time
    if _tracker is None:
        return {}
    t0 = time.perf_counter()
    bias_corr_2 = 1.0
    for group in optimizer.param_groups:
        if "betas" in group:
            beta2 = float(group["betas"][1])
            bias_corr_2 = max(1.0 - beta2 ** max(step, 1), 1e-12)
            break

    chunks = []
    n_missing = 0
    for name in _tracker._tracked_names:
        p = _tracker._name_to_param[name]
        from .importance_tracker import _fp32_v_t  # late import to avoid cycle
        v = _fp32_v_t(optimizer, p)
        if v is None:
            n_missing += 1
            continue
        chunks.append(v.detach().abs().flatten())
    if not chunks:
        return {"error": "no v_t available", "step": step}
    flat_v = torch.cat(chunks).to(torch.float32)
    # Bias-correct: v̂_t = v_t / (1 - β₂^t)
    flat_v.div_(bias_corr_2)

    total = flat_v.sum().item()
    N = int(flat_v.numel())
    sorted_v, _ = torch.sort(flat_v, descending=True)
    cumsum = sorted_v.cumsum(0)

    # Sweep low end finely to characterize heavy-tail shape near K→0.
    # k = max(1, int(K/100 × N)); for N=96M, K=0.001% gives top-960 elements.
    K_pcts = [
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
        0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0,
    ]
    coverage: Dict[str, float] = {}
    for kp in K_pcts:
        k = max(1, int(kp / 100.0 * N))
        c = float(cumsum[k - 1].item()) / max(total, 1e-30)
        coverage[f"K={kp}%"] = c

    elapsed = time.perf_counter() - t0
    rec = {
        "step": int(step),
        "n_total_expert_elements": N,
        "n_missing_v_t_params": n_missing,
        "bias_corr_2": bias_corr_2,
        "v_t_sum_total": total,
        "v_t_max": float(sorted_v[0].item()),
        "v_t_top1_share": float(sorted_v[0].item()) / max(total, 1e-30),
        "coverage": coverage,
        "compute_time_s": elapsed,
    }
    # Free memory
    del flat_v, sorted_v, cumsum
    torch.cuda.empty_cache()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return rec
    try:
        with open(output_path, "w") as f:
            json.dump(rec, f, indent=2)
    except Exception:
        pass
    return rec


@torch.no_grad()
def proposal_measure_moc_fisher_coverage(
    optimizer, step: int, output_path: str,
) -> Dict[str, Any]:
    """MoC analogue of Fisher coverage measurement.

    Computes, at iter ``step``:
      - per-expert v̂_t mass (96 numbers, indexed by (layer, expert_idx))
      - identifies the single expert MoC fires at this iter (round-robin)
      - reports (numerator = fired expert v̂_t sum) / (denom = all-expert v̂_t sum)
      - also reports cumulative "best-K experts" coverage (sorted descending)
        for comparison with proposal's element-wise C(K)
    """
    import json
    import time
    from collections import defaultdict
    if _tracker is None or _fc is None:
        return {}
    t0 = time.perf_counter()
    bias_corr_2 = 1.0
    for group in optimizer.param_groups:
        if "betas" in group:
            beta2 = float(group["betas"][1])
            bias_corr_2 = max(1.0 - beta2 ** max(step, 1), 1e-12)
            break

    from .importance_tracker import _fp32_v_t
    from .fc_manager import LAYER_RE, DS_EXPERT_IDX_RE
    # Per-expert v̂_t sum, keyed by (layer, expert_idx).
    per_expert: Dict[Tuple[int, int], float] = defaultdict(float)
    n_total_elements = 0
    n_missing_v_t = 0
    for name in _tracker._tracked_names:
        p = _tracker._name_to_param[name]
        v = _fp32_v_t(optimizer, p)
        if v is None:
            n_missing_v_t += 1
            continue
        s = float(v.detach().abs().sum().item()) / bias_corr_2
        n_total_elements += int(v.numel())
        m_l = LAYER_RE.search(name)
        m_e = DS_EXPERT_IDX_RE.search(name)
        if m_l is None or m_e is None:
            continue
        per_expert[(int(m_l.group(1)), int(m_e.group(1)))] += s

    total = sum(per_expert.values())
    if total <= 0:
        return {"error": "total v_t sum is zero", "step": step}

    # Identify the expert MoC fires at this iter (round_robin: slot=(step-1)%N).
    n_slots = len(_fc._all_expert_ids)
    fired_slot = (step - 1) % n_slots if n_slots > 0 else -1
    fired_expert = _fc._all_expert_ids[fired_slot] if fired_slot >= 0 else None
    fired_sum = per_expert.get(tuple(fired_expert), 0.0) if fired_expert else 0.0

    # Cumulative "best-K experts" coverage (sort experts by v̂_t mass desc).
    sorted_exp = sorted(per_expert.items(), key=lambda kv: -kv[1])
    cum = 0.0
    bestk = []
    for k_idx, (eid, s) in enumerate(sorted_exp, 1):
        cum += s
        bestk.append({"k_experts": k_idx, "coverage": cum / total,
                      "k_pct_of_slots": k_idx / n_slots * 100})
    # Also derive specific K_pct points to match proposal's K-format (approx).
    K_pcts = [1, 2, 5, 10, 25, 50, 100]
    bestk_summary = {}
    for kp in K_pcts:
        k = max(1, int(kp / 100.0 * n_slots))
        bestk_summary[f"K_experts={kp}%"] = bestk[k - 1]["coverage"]

    # Per-expert sums (sorted descending) for full distribution analysis.
    per_expert_list = [
        {"layer": L, "expert_idx": E, "v_t_sum": s,
         "share": s / total}
        for (L, E), s in sorted_exp
    ]

    elapsed = time.perf_counter() - t0
    rec = {
        "step": int(step),
        "mode": "moc_round_robin",
        "n_expert_slots": n_slots,
        "n_total_expert_elements": n_total_elements,
        "n_missing_v_t_params": n_missing_v_t,
        "bias_corr_2": bias_corr_2,
        "total_v_t_sum": total,
        "fired_slot": fired_slot,
        "fired_expert": list(fired_expert) if fired_expert else None,
        "fired_v_t_sum": fired_sum,
        "fired_coverage": fired_sum / total,
        "best_k_experts_coverage": bestk_summary,
        "per_expert_sorted_desc": per_expert_list,
        "compute_time_s": elapsed,
    }

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return rec
    try:
        with open(output_path, "w") as f:
            json.dump(rec, f, indent=2)
    except Exception:
        pass
    return rec


def proposal_inject_fault(
    model, optimizer, step: int, args,
    pre_fault_loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Exp 2 mid-training fault injection.

    Drains async writer, reconstructs model + optimizer state from the
    existing CP files (FCs + DC bundles) via ``proposal_recover``, and
    returns diagnostic info (recovery time, diag, pre/post loss).

    Training caller is expected to resume from the recovered state. We do
    NOT reset the FCManager or tracker — those stateful objects keep
    running as if no fault occurred (correct: the CP they wrote is what we
    just reloaded, so their bookkeeping is already consistent).
    """
    import time
    import json

    if not _enabled:
        return {}
    if _writer is not None:
        _writer.drain()

    # Before recovery, fsync the log-structured expert state so segment
    # contents are durable on disk. recover() then reads each expert's
    # compacted.bin + remaining segments via read_expert_state(expert_dir).
    flushed_paths: List[Path] = []
    if _fc is not None and _fc.expert_state is not None and _cp_dir is not None:
        try:
            flushed_paths = _fc.expert_state.flush()
        except Exception as exc:
            print(
                f"[proposal] WARNING expert-state flush failed @iter {step}: {exc}; "
                f"falling back to legacy DC replay.",
                flush=True,
            )
            flushed_paths = []

    t0 = time.perf_counter()
    inner_model = _config["_inner_model"]
    inner_opt = _config["_inner_opt"]
    num_experts = _config["num_experts"]
    # Same DRAM-mode short-circuit as proposal_recover() (above): when
    # cp_target=dram the per-expert state lives in DramExpertState, not on
    # disk. Pass it through so recover() reads from the in-memory buffer.
    _dram_es = None
    if (
        _fc is not None and _fc.expert_state is not None
        and getattr(_fc, "cp_target", "disk") == "dram"
    ):
        _dram_es = _fc.expert_state
    diag = recover(
        model=inner_model,
        optimizer=inner_opt,
        cp_dir=_cp_dir,
        t_fault=int(step),
        num_experts=num_experts,
        dram_expert_state=_dram_es,
    )
    # H_g is GPU-only; next FD refresh repopulates it.
    recover_s = time.perf_counter() - t0

    # Maximum lost iteration window "B" = step - min(expert_fc_iter, ne_fc_iter).
    # For MoC (sequential, no DC), B is max iters since last FC for any expert.
    # For proposal/random, B ≈ iters since last same-expert FC (≤ D typically).
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
    global _pending_fault_record
    _pending_fault_record = rec
    # Register fault log path for _flush_fault_record callback.
    log_path = getattr(args, "proposal_fault_log", None)
    if log_path:
        _config["_fault_log_path"] = log_path
    return rec


def set_hessian_valid_iter(it: Any) -> None:
    """Install a dedicated validation data iterator used ONLY by
    proposal_refresh_hessian. Keeps the shared valid_data_iterator
    untouched so periodic evaluate_and_print_results draws the same
    sample across (hessian | v_hat | baseline) variants, making val_loss
    comparable."""
    _config["_hessian_dedicated_valid_iter"] = it


def _fold_h(names: List[str], h_new_list: List[torch.Tensor],
            flip_cutoff: Optional[float] = None) -> Dict[str, Any]:
    """Overwrite ``_tracker._H_g`` (group-level |H|) with a fresh estimate.

    Plain assignment per refresh (no temporal smoothing). Group H lives
    in BF16 in ``_g_fast_h_g_bf16_flat`` per-name views; the dict
    ``_tracker._H_g`` holds the same views (storage shared).

    Outlier protection: per-tensor Hz-norm is compared to the median
    across tensors; tensors exceeding ``flip_cutoff × median`` are
    skipped (residual safety net beyond routing freeze). Tensors with
    zero coverage are also skipped. ``flip_cutoff=None`` disables.

    Returns ``{expert_coverage, flip_skip_ratio, n_skipped}``.
    """
    if _tracker is None:
        return {"expert_coverage": 0.0, "flip_skip_ratio": 0.0, "n_skipped": 0}

    # Compute per-tensor Hz norms (mean abs) on GPU.
    hz_norms = []
    coverage_flags = []
    for h in h_new_list:
        s = float(h.detach().abs().mean().item()) if h.numel() > 0 else 0.0
        hz_norms.append(s)
        coverage_flags.append(s > 0)
    n = len(h_new_list)
    expert_coverage = float(sum(coverage_flags)) / max(n, 1)
    skip_mask = [False] * n
    if flip_cutoff is not None and n > 0:
        positives = [v for v in hz_norms if v > 0]
        if positives:
            positives.sort()
            med = positives[len(positives) // 2]
            for i, v in enumerate(hz_norms):
                if v > flip_cutoff * med:
                    skip_mask[i] = True
    # H_DIAG: per-anchor, list zero-coverage + flip-skipped tensors.
    import os as _os_h_diag
    if _os_h_diag.environ.get("PROPOSAL_H_DIAG", "") in ("1",):
        n_zero = sum(1 for v in hz_norms if v == 0)
        n_flip = sum(1 for v in skip_mask if v)
        positives = sorted([v for v in hz_norms if v > 0])
        if positives:
            mn = positives[0]; mx = positives[-1]
            md = positives[len(positives)//2]
            print(f"[H_DIAG] n_total={n} n_zero={n_zero} n_flip={n_flip} "
                  f"hz_norm: min={mn:.3e} median={md:.3e} max={mx:.3e}",
                  flush=True)
            # List first 8 zero-coverage tensor names
            zero_names = [names[i] for i in range(n) if hz_norms[i] == 0][:8]
            flip_names = [names[i] for i in range(n) if skip_mask[i]][:8]
            if zero_names:
                print(f"[H_DIAG] zero_names (first 8): {zero_names}", flush=True)
            if flip_names:
                print(f"[H_DIAG] flip_names (first 8): {flip_names}", flush=True)
    g_blk = max(1, int(getattr(_tracker, "block_quant_g", 1)))
    n_skipped = 0
    for i, (name, h_new) in enumerate(zip(names, h_new_list)):
        if not coverage_flags[i] or skip_mask[i]:
            # H=0 or outlier: skip fold (keep existing H). update_v_mapping
            # writes mapped(v_t) — nonzero (Adam v_t > 0) — into the buffer
            # every iter, so even un-folded experts get nonzero score at the
            # lowest selection bracket. Explicit floor was tested and proven
            # redundant.
            n_skipped += 1
            continue
        # Reduce h_new to per-group mean (BF16) — drop element-level.
        # Write IN-PLACE into _g_fast_h_g_bf16_flat's per-name view; the
        # dict ``_tracker._H_g`` holds that same view (storage shared,
        # no extra alloc).
        view_idx_map = getattr(_tracker, "_g_fast_name_to_idx", None)
        if view_idx_map is None and hasattr(_tracker, "_g_fast_name_list"):
            view_idx_map = {nm: idx for idx, nm
                            in enumerate(_tracker._g_fast_name_list)}
            _tracker._g_fast_name_to_idx = view_idx_map
        if view_idx_map is None or name not in view_idx_map:
            continue
        view = _tracker._g_fast_h_g_views[view_idx_map[name]]
        # Detect the FD per-group-BF16 fast path: h_new is already shaped
        # ``(numel_view,)`` with dtype BF16 (compute_hessian_finite_diff
        # has folded element→group inline). In that case copy directly.
        if (h_new.dtype == torch.bfloat16
                and h_new.numel() == view.numel()):
            view.copy_(h_new.detach().reshape(-1))
            _tracker._H_g[name] = view
            continue
        # Element-level path (multi-K diagnostic).
        h_flat = h_new.detach().reshape(-1).to(torch.float32)
        nt = h_flat.numel()
        nt_a = (nt // g_blk) * g_blk
        if nt_a == 0:
            continue
        h_g_new = h_flat[:nt_a].reshape(-1, g_blk).mean(dim=1)
        del h_flat
        view.copy_(h_g_new.to(torch.bfloat16))
        _tracker._H_g[name] = view
        del h_g_new
    # Anchor 시점 lookup-table 갱신: _g_fast_h_quantiles ← quantile of
    # _g_fast_h_g_bf16_flat (just written above). update_v_mapping uses
    # this every iter to overwrite the same buffer with mapped(v_t).
    try:
        _tracker.update_h_anchor_quantiles()
    except Exception as _exc:
        if _should_write():
            print(f"[proposal] update_h_anchor_quantiles failed: {_exc}",
                  flush=True)
    return {
        "expert_coverage": expert_coverage,
        "flip_skip_ratio": float(n_skipped) / max(n, 1),
        "n_skipped": int(n_skipped),
    }


def proposal_fd_noise_probe(
    model, forward_step_func, data_iterator, iteration,
) -> Optional[Dict[str, Any]]:
    """Diagnostic FD Hessian noise probe — fires only at iters listed in
    ``--proposal-fd-noise-probe-iters``. Side-channel: does NOT mutate
    ``tracker._H_g`` (production refresh runs separately).

    Captures the same ``tracked`` expert_ps set as the production
    refresh, the same eps, and the same routing-freeze convention,
    then runs the multi-call probe defined in ``fd_noise_probe``.
    """
    if not _enabled or _tracker is None:
        return None
    probe_iters = _config.get("_fd_noise_probe_iters") or ()
    if not probe_iters or int(iteration) not in probe_iters:
        return None
    from .fd_noise_probe import run_noise_probe

    inner_model = _config["_inner_model"]
    tracked = [(n, p) for n, p in inner_model.named_parameters()
               if p.requires_grad and getattr(p, "allreduce", True) is False]
    if not tracked:
        return None
    expert_ps = [p for _, p in tracked]

    # Use the dedicated valid iterator if installed (same as production
    # refresh). The probe will advance this iterator by ref_n_batches.
    dedicated = _config.get("_hessian_dedicated_valid_iter")
    di = dedicated if dedicated is not None else data_iterator

    # Match production refresh's pre/post wrapping so model state is
    # restored cleanly (training mode, RNG state, ds_engine flag).
    _cpu_rng_state = torch.get_rng_state()
    _cuda_rng_state = (torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
    if hasattr(model[0], "module"):
        net = model[0].module
    else:
        net = model[0]
    was_training = net.training
    net.train()
    m0 = model[0] if isinstance(model, (list, tuple)) else model
    ds_engine = m0
    prev_flag = getattr(ds_engine, "_running_engine_backward", False)
    try:
        ds_engine._running_engine_backward = True
    except Exception:
        pass

    eps = _config.get("_finite_diff_eps")
    log_path = _config.get("_fd_noise_probe_log")
    ref_k = int(_config.get("_fd_noise_probe_ref_k", 32))
    ref_n_batches = int(_config.get("_fd_noise_probe_ref_n_batches", 10))
    g_blk = int(_config.get("_fd_noise_probe_g_blk", 1))

    try:
        metrics = run_noise_probe(
            iteration=int(iteration),
            model=model,
            forward_step_func=forward_step_func,
            data_iterator=di,
            expert_ps=expert_ps,
            eps=eps,
            ref_k=ref_k,
            ref_n_batches=ref_n_batches,
            log_path=log_path,
            g_blk=g_blk,
            tracker=_tracker,
        )
    finally:
        try:
            ds_engine._running_engine_backward = prev_flag
        except Exception:
            pass
        net.train(was_training)
        try:
            torch.set_rng_state(_cpu_rng_state)
            if _cuda_rng_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(_cuda_rng_state)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return metrics


def proposal_refresh_hessian(
    model, forward_step_func, data_iterator, iteration,
    estimator: str = "finite_diff",
    k_samples: int = 1,
) -> Optional[Dict[str, Any]]:
    """Refresh ``tracker._H_g`` (BF16 group-level diagonal Hessian)
    with a fresh Finite-Difference estimate. Production refresh path
    is FD-only; ``estimator`` is kept for backward-compatible call
    sites but ignored. ``proposal_refresh_finite_diff`` overwrites the
    cached H_g views in place (no temporal smoothing).
    """
    del estimator  # FD-only now
    return proposal_refresh_finite_diff(
        model, forward_step_func, data_iterator, iteration,
        k_samples=k_samples,
    )


def proposal_refresh_finite_diff(
    model, forward_step_func, data_iterator, iteration,
    k_samples: int = 4,
) -> Optional[Dict[str, Any]]:
    """Overwrite ``tracker._H_g`` (BF16 group-level views into
    ``_g_fast_h_g_bf16_flat``) with a fresh finite-difference
    Hessian-vector |H_diag| estimate. Plain 1st-order grads × 2
    (central diff) per Rademacher sample; routing freeze always
    applied around the +/- forwards (production default).

    """
    from .hessian import (
        compute_hessian_finite_diff as _finite_diff_compute_H,
    )
    if not _enabled or _tracker is None:
        return None
    dedicated = _config.get("_hessian_dedicated_valid_iter")
    if dedicated is not None:
        data_iterator = dedicated

    import time as _time
    inner_model = _config["_inner_model"]
    tracked = [(n, p) for n, p in inner_model.named_parameters()
               if p.requires_grad and getattr(p, "allreduce", True) is False]
    if not tracked:
        return None

    _cpu_rng_state = torch.get_rng_state()
    _cuda_rng_state = (torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
    if hasattr(model[0], "module"):
        net = model[0].module
    else:
        net = model[0]
    was_training = net.training
    net.train()
    m0 = model[0] if isinstance(model, (list, tuple)) else model
    ds_engine = m0
    prev_flag = getattr(ds_engine, "_running_engine_backward", False)
    try: ds_engine._running_engine_backward = True
    except Exception: pass

    def _restore_rng():
        try:
            torch.set_rng_state(_cpu_rng_state)
            if _cuda_rng_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(_cuda_rng_state)
        except Exception: pass

    def _finalize(ret):
        try: ds_engine._running_engine_backward = prev_flag
        except Exception: pass
        net.train(was_training); _restore_rng()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ret

    expert_ps = [p for _, p in tracked]
    names = [n for n, _ in tracked]

    eps = _config.get("_finite_diff_eps")

    # PROPOSAL_FD_ONE_SIDED=1 → forward-difference instead of central.
    # Cuts fwd+bwd from 2K to K+1 (47% at K=16) at the cost of O(ε) vs
    # O(ε²) truncation.
    import os as _os_fd1
    _one_sided = _os_fd1.environ.get("PROPOSAL_FD_ONE_SIDED", "") in ("1", "true", "TRUE")

    multi_k_track = bool(_config.get("_fd_multi_k_track", False))
    multi_k_values = sorted(set(int(v) for v in
                                _config.get("_fd_multi_k_values", []) or []
                                if int(v) >= 1))
    multi_k_log_path = _config.get("_fd_multi_k_log")
    multi_k_max = max(multi_k_values) if (multi_k_track and multi_k_values) else None
    fd_k_used = int(multi_k_max) if multi_k_max is not None else int(k_samples)

    t0 = _time.perf_counter()
    # Audit: pre-anchor snapshot.
    from . import _vram_audit
    _vram_audit.vram_audit_step("pre_anchor", dump_breakdown=True, tracker=_tracker)
    # FD core accumulates BF16 directly into a per-group buffer of size
    # ``numel/g_blk`` (saves transient memory; final |H_g| is BF16 per
    # CLAUDE.md spec). Multi-K diagnostic mode forces FP32 element-level
    # accumulation inside the core (``track_k_values`` branch) so cross-K
    # Pearson is computed on a shared FP32 axis.
    _g_blk_for_fd = max(1, int(getattr(_tracker, "block_quant_g", 1)))

    # FD's +ε/-ε grads + activations don't fit on a single A100 alongside
    # billion-scale Adam state. Stash master+m+v to pinned CPU during the
    # refresh; ~5s pinned-pool D2H+H2D overhead amortises to <1 ms/iter at
    # refresh_every≥1000. Disable with PROPOSAL_FD_NO_OFFLOAD=1.
    import os as _os_off
    _do_offload = _os_off.environ.get(
        "PROPOSAL_FD_NO_OFFLOAD", "") not in ("1", "true", "TRUE")
    _offload_handle = None
    if _do_offload:
        from ._ds_state import offload_adam_state_to_cpu
        if _writer is not None:
            try:
                _writer.drain_non_expert(); _writer.drain_expert()
            except Exception:
                pass
        _t_off0 = _time.perf_counter()
        _offload_handle = offload_adam_state_to_cpu(_tracker)
        if _should_write():
            print(f"[proposal] anchor offload D2H "
                  f"{(_time.perf_counter()-_t_off0)*1000:.0f}ms", flush=True)
        _vram_audit.vram_audit_step("pre_anchor_offload", dump_breakdown=False)

    # Suppress release-hook firing for the anchor's +ε/-ε forwards.
    _config["_in_anchor_refresh"] = True
    # Snapshot training state — anchor's multiple forward/backward passes
    # advance Megatron's CudaRNGTracker and may write into model param
    # .grad / DeepSpeed hp_grad. Restore in finally so training continues
    # bit-exact regardless of how many anchor passes fired.
    from ._ds_state import snapshot_training_state, restore_training_state
    _train_state_snap = snapshot_training_state(_config.get("_inner_opt"), model=model)
    _ds_engine_opt = _get_ds_engine_optimizer(model)
    # FULL STATE DIFF DIAGNOSTIC — capture every state hash pre/post anchor
    # to pinpoint what per-chunk's N×backward changes. Caller runs once on
    # first per-chunk anchor; output identifies which buffers / optimizer
    # state / RNG keys drift.
    def _capture_full_state(_dso):
        st = {}
        _m = model[0] if isinstance(model, (list, tuple)) else model
        for n, p in _m.named_parameters():
            st["P:" + n] = (float(p.detach().float().norm().item()),
                            float(p.detach().float().sum().item()))
            g = p.grad
            st["G:" + n] = (float(g.float().norm().item()) if g is not None else None,
                            float(g.float().sum().item()) if g is not None else None)
        for n, b in _m.named_buffers():
            st["B:" + n] = (float(b.detach().float().norm().item()),
                            float(b.detach().float().sum().item()))
        # DS engine state
        if _dso is not None:
            for attr in ("micro_steps", "global_steps", "loss_scale",
                         "cur_iter", "cur_scale"):
                if hasattr(_dso, attr):
                    try:
                        st["DS:" + attr] = getattr(_dso, attr)
                    except Exception:
                        pass
            for fa in ("fp32_groups_flat", "fp32_groups_gradients_flat"):
                gl = getattr(_dso, fa, None)
                if gl:
                    for i, t in enumerate(gl):
                        if isinstance(t, torch.Tensor):
                            st[f"DS:{fa}[{i}]"] = (float(t.float().norm().item()),
                                                    float(t.float().sum().item()))
        # Inner optimizer state (Adam m, v)
        _io = _config.get("_inner_opt")
        if _io is not None and hasattr(_io, "state"):
            for i, (p, s) in enumerate(_io.state.items()):
                for k, v in s.items():
                    if isinstance(v, torch.Tensor):
                        st[f"OS:{i}:{k}"] = (float(v.float().norm().item()),
                                             float(v.float().sum().item()))
                    else:
                        st[f"OS:{i}:{k}"] = v
        if torch.cuda.is_available():
            try:
                import hashlib
                st["RNG:cuda"] = hashlib.sha256(torch.cuda.get_rng_state(0).numpy().tobytes()).hexdigest()[:16]
            except Exception:
                pass
        return st
    import os as _os_diag_anchor
    _full_diag = _os_diag_anchor.environ.get("PROPOSAL_ANCHOR_FULL_DIAG", "") in ("1",)
    _pre_state = _capture_full_state(_ds_engine_opt) if _full_diag else None
    # CRITICAL fix for per-chunk anchor: DeepSpeed BF16Optimizer registers
    # grad-accumulate hooks on every model param. Each anchor backward fires
    # these hooks → accumulates lp.grad into fp32_groups_gradients_flat
    # (DS internal flat buffer). Per-chunk's N×2 backwards thus poison
    # the DS flat buffer with ~2N·g(θ₀) for non-chunk params; next iter's
    # optimizer.step consumes this corruption → lm_loss drift at iter T+2.
    # p.grad=None in _compute_grad_no_create only clears the Python view,
    # NOT the DS flat buffer. Fix: detach all hooks before anchor, restore
    # after.
    # DS BF16Optimizer hook detach (hypothesis 1 of analyst's diagnosis).
    # Effective only if the DS optimizer is BF16_Optimizer; for FP16_Optimizer
    # this is a no-op because FP16_Optimizer.backward is just loss.backward()
    # without grad-accumulator hooks. Kept here as a safety net for BF16 path.
    _saved_grad_acc_hooks = None
    _saved_hp_grads_flat = None
    if _ds_engine_opt is not None and hasattr(_ds_engine_opt, "_grad_acc_hooks"):
        _saved_grad_acc_hooks = _ds_engine_opt._grad_acc_hooks
        for _h in _saved_grad_acc_hooks:
            try: _h.remove()
            except Exception: pass
        _ds_engine_opt._grad_acc_hooks = []
        try:
            _fp32_grads = getattr(_ds_engine_opt, "fp32_groups_gradients_flat", None)
            if _fp32_grads is not None:
                _saved_hp_grads_flat = [t.detach().clone() for t in _fp32_grads]
        except Exception:
            _saved_hp_grads_flat = None
    try:
        # MULTI-BATCH ANCHOR: each batch's routing activates only ~67% of
        # experts; averaging over N batches (each with a different routing)
        # raises H coverage to ≥0.93. N = mb_per_iter/2 (set at init_proposal).
        _n_anchor_batches = max(1, int(_config["_anchor_n_batches"]))
        _use_freeze = (_os_off.environ.get("PROPOSAL_ROUTING_FREEZE_ENABLE", "")
                       in ("1", "true", "True"))
        fd_ret = None
        _n_succ = 0
        _mb_timing_on = (_os_off.environ.get("PROPOSAL_MB_TIMING", "0")
                         in ("1", "true", "TRUE"))
        import time as _time_mb
        # Fast-flat: grad-flat (single Triton stream kernel) + per-tensor
        # hash-based perturb. No constraint on lp storage (works for FP16/
        # BF16 optimizer + arbitrary param-group layout).
        from megatron.proposal._ds_state import _pinned_clone as _pc
        try:
            from megatron.proposal.hessian import _HAS_TRITON_HESSIAN as _HAS_TRITON_GUARD
        except Exception:
            _HAS_TRITON_GUARD = False
        _fast_flat_env = (_os_off.environ.get("PROPOSAL_FD_FAST_FLAT", "1")
                          not in ("0", "false", "off", "OFF", "False"))
        # Tracker buffer layout dump: are expert positions contiguous?
        if _os_off.environ.get("PROPOSAL_FD_FAST_FLAT_DBG", "0") in ("1",):
            try:
                _expert_names = [_n for _n, _ in tracked]
                _name_to_idx = getattr(_tracker, "_g_fast_name_to_idx", None)
                if _name_to_idx is None and hasattr(_tracker, "_g_fast_name_list"):
                    _name_to_idx = {nm: idx for idx, nm
                                    in enumerate(_tracker._g_fast_name_list)}
                _total_names = len(_tracker._g_fast_name_list)
                _expert_idxs = sorted([_name_to_idx[_n] for _n in _expert_names
                                       if _n in _name_to_idx])
                _is_contig = (_expert_idxs == list(range(min(_expert_idxs),
                                                         max(_expert_idxs)+1)))
                # Check group-level (BF16) buffer contiguity
                _views = [_tracker._g_fast_h_g_views[_name_to_idx[_n]]
                          for _n in _expert_names if _n in _name_to_idx]
                _v0 = _views[0]
                _storage_ptr = _v0.untyped_storage().data_ptr()
                _bpe = _v0.element_size()
                _expected = _v0.storage_offset() * _bpe
                _gv_contig = True
                _total_groups = 0
                for _v in _views:
                    if (_v.untyped_storage().data_ptr() != _storage_ptr
                            or _v.storage_offset() * _bpe != _expected):
                        _gv_contig = False
                        break
                    _expected += _v.numel() * _bpe
                    _total_groups += _v.numel()
                print(f"[fast-flat-dbg] tracker buffer: n_total_names={_total_names} "
                      f"n_experts={len(_expert_idxs)} "
                      f"expert_idx_range=[{min(_expert_idxs)},{max(_expert_idxs)}] "
                      f"idx_contig={_is_contig}",
                      flush=True)
                print(f"[fast-flat-dbg] expert group-views: "
                      f"storage_contig={_gv_contig} "
                      f"n_groups_in_expert_region={_total_groups} "
                      f"buf_total_numel={_tracker._g_fast_h_g_bf16_flat.numel()} "
                      f"buf_dtype={_tracker._g_fast_h_g_bf16_flat.dtype}",
                      flush=True)
            except Exception as _e:
                print(f"[fast-flat-dbg] tracker dump err: {_e}", flush=True)
        # DS optimizer fp16_groups / fp16_groups_flat structure dump
        if _os_off.environ.get("PROPOSAL_FD_FAST_FLAT_DBG", "0") in ("1",):
            try:
                _opt = _ds_engine_opt
                _groups_attr = None
                _flats_attr = None
                for _ga, _fa in (("fp16_groups", "fp16_groups_flat"),
                                 ("bf16_groups", "bf16_groups_flat"),
                                 ("bit16_groups", "bit16_groups_flat")):
                    if hasattr(_opt, _ga) and hasattr(_opt, _fa):
                        _groups_attr = _ga; _flats_attr = _fa; break
                if _groups_attr is None:
                    print(f"[fast-flat-dbg] no known flat-group attrs on {type(_opt).__name__}; "
                          f"attrs={[a for a in dir(_opt) if 'group' in a.lower()]}",
                          flush=True)
                else:
                    _grps = getattr(_opt, _groups_attr)
                    _flats = getattr(_opt, _flats_attr)
                    _ep_set = {id(p) for p in expert_ps}
                    print(f"[fast-flat-dbg] DS opt={type(_opt).__name__} "
                          f"{_groups_attr}: {len(_grps)} groups", flush=True)
                    for _gi, _grp in enumerate(_grps):
                        _exp_in = sum(1 for p in _grp if id(p) in _ep_set)
                        _gf = _flats[_gi]
                        _ne_grp = sum(p.numel() for p in _grp)
                        print(f"  group{_gi}: n_params={len(_grp)} "
                              f"experts_in_group={_exp_in} "
                              f"flat_ptr={hex(_gf.untyped_storage().data_ptr())} "
                              f"flat_numel={_gf.numel()} "
                              f"sum_param_numel={_ne_grp}",
                              flush=True)
                    # Map expert → group
                    _exp_grp_count = {}
                    for p in expert_ps:
                        _p_ptr = p.untyped_storage().data_ptr()
                        for _gi, _gf in enumerate(_flats):
                            if _gf.untyped_storage().data_ptr() == _p_ptr:
                                _exp_grp_count[_gi] = _exp_grp_count.get(_gi, 0) + 1
                                break
                    print(f"[fast-flat-dbg] expert→DS-group: {dict(_exp_grp_count)} "
                          f"(total experts={len(expert_ps)})",
                          flush=True)
            except Exception as _e:
                print(f"[fast-flat-dbg] DS dump err: {_e}", flush=True)
        # Fast-flat setup. Reorder expert_ps to tracker.name_list order so
        # the tracker's _g_fast_h_g_bf16_flat (single contiguous BF16 buffer,
        # already in R1 budget) serves as h_accum_flat — zero extra VRAM
        # for the H accumulator. grad_flat is still allocated fresh
        # (same size as training's lp.grad transient → 0 over training peak).
        _grad_flat = None
        _grad_offsets = None
        _h_accum_flat = None
        _grad_view_set = False
        _expert_ps_sorted = expert_ps
        _names_sorted = [_n for _n, _ in tracked]
        if _fast_flat_env and _HAS_TRITON_GUARD:
            try:
                _name_to_idx = getattr(_tracker, "_g_fast_name_to_idx", None)
                if _name_to_idx is None and hasattr(_tracker, "_g_fast_name_list"):
                    _name_to_idx = {nm: idx for idx, nm
                                    in enumerate(_tracker._g_fast_name_list)}
                    _tracker._g_fast_name_to_idx = _name_to_idx
                _missing = [_n for _n in _names_sorted if _n not in _name_to_idx]
                if _missing:
                    raise RuntimeError(
                        f"{len(_missing)} expert names not in tracker; first: {_missing[:3]}")
                # Sort expert_ps + names by tracker idx ascending
                _order = sorted(range(len(_expert_ps_sorted)),
                                key=lambda i: _name_to_idx[_names_sorted[i]])
                _expert_ps_sorted = [expert_ps[_i] for _i in _order]
                _names_sorted = [_names_sorted[_i] for _i in _order]
                # Verify sorted name idxs are exactly [0..N-1] (= full tracker buffer)
                _sorted_idxs = [_name_to_idx[_n] for _n in _names_sorted]
                _N = len(_sorted_idxs)
                if _sorted_idxs != list(range(_N)):
                    raise RuntimeError(
                        f"expert idxs not [0..{_N-1}] after sort; "
                        f"min={_sorted_idxs[0]} max={_sorted_idxs[-1]} len={_N}")
                # Build grad_flat (fresh alloc, same size as lp.grad transient)
                _p0 = _expert_ps_sorted[0]
                _ne = [int(p.numel()) for p in _expert_ps_sorted]
                _grad_offsets = [0]
                for _n in _ne[:-1]:
                    _grad_offsets.append(_grad_offsets[-1] + _n)
                _total_n = sum(_ne)
                _g_blk_int = max(1, int(_g_blk_for_fd))
                _total_groups = _total_n // _g_blk_int
                _grad_flat = torch.zeros(_total_n, dtype=_p0.dtype,
                                         device=_p0.device)
                # h_accum_flat = tracker buffer directly (NO new alloc)
                _h_accum_flat = _tracker._g_fast_h_g_bf16_flat
                if int(_h_accum_flat.numel()) != int(_total_groups):
                    raise RuntimeError(
                        f"tracker buffer numel {int(_h_accum_flat.numel())} "
                        f"!= expected expert groups {int(_total_groups)}")
                # Assign each expert.grad = view of grad_flat
                for _i, _p in enumerate(_expert_ps_sorted):
                    _off = _grad_offsets[_i]
                    _p.grad = _grad_flat[_off:_off + _ne[_i]].view_as(_p)
                _grad_view_set = True
                if _should_write():
                    print(f"[proposal] anchor fast-flat ON "
                          f"(n_elem={_total_n} grad_flat={_total_n*2/1e9:.2f}GB "
                          f"h_accum=tracker_view {_total_groups*2/1e6:.0f}MB no-alloc)",
                          flush=True)
            except Exception as _e:
                _grad_flat = None
                _grad_offsets = None
                _h_accum_flat = None
                _grad_view_set = False
                _expert_ps_sorted = expert_ps
                _names_sorted = [_n for _n, _ in tracked]
                if _should_write():
                    print(f"[proposal] anchor fast-flat OFF (alloc err: {_e})",
                          flush=True)
        # Per-tensor θ_0 snapshot (Task 1; pool reuses across anchor calls).
        # NOTE: use _expert_ps_sorted (matches fast-flat ordering for
        # grad_offsets / h_accum_flat; identical to expert_ps when fast-flat
        # is disabled).
        _theta_snap_session = [
            _pc(_p.data.detach(), key=("anchor_theta", _i))
            for _i, _p in enumerate(_expert_ps_sorted)
        ]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        try:
            for _bi in range(_n_anchor_batches):
                if _mb_timing_on and torch.cuda.is_available():
                    torch.cuda.synchronize()
                _mb_t0 = _time_mb.perf_counter()
                # Zero h_accum at multi-batch loop entry (fast-flat path
                # accumulates into the SAME flat across batches — single
                # storage, no per-batch return list to sum).
                if _grad_flat is not None and _bi == 0:
                    _h_accum_flat.zero_()
                _ret = _finite_diff_compute_H(
                    model, forward_step_func, data_iterator,
                    _expert_ps_sorted, fd_k_used,
                    eps=eps, use_routing_freeze=_use_freeze,
                    batch_override=None,
                    one_sided=_one_sided,
                    track_k_values=(multi_k_values if multi_k_track else None),
                    g_blk=_g_blk_for_fd,
                    theta_snap_ref=_theta_snap_session,
                    grad_flat=_grad_flat,
                    grad_offsets=_grad_offsets,
                    h_accum_flat=_h_accum_flat,
                )
                if _mb_timing_on and torch.cuda.is_available():
                    torch.cuda.synchronize()
                if _mb_timing_on:
                    print(f"[MB_TIMING] batch={_bi} {(_time_mb.perf_counter()-_mb_t0)*1000:.0f}ms",
                          flush=True)
                if _ret is None:
                    continue
                _n_succ += 1
                if _grad_flat is not None:
                    # Fast-flat: core self-accumulates into h_accum_flat;
                    # _ret is just a success sentinel ([]). fd_ret built
                    # post-loop from h_accum_flat (div + abs + split).
                    pass
                else:
                    if fd_ret is None:
                        fd_ret = _ret  # list of BF16 group-H tensors (~47 MB total)
                    else:
                        # In-place add (BF16 → cast briefly to FP32 to avoid sat)
                        for _hi, _h in enumerate(_ret):
                            fd_ret[_hi].add_(_h)
                        del _ret
            # Fast-flat: build fd_ret from accumulated h_accum_flat
            # (div by success count + abs + per-expert view). fd_ret order
            # follows _names_sorted (tracker idx order).
            if _grad_flat is not None and _n_succ > 0:
                _h_accum_flat.div_(float(_n_succ)).abs_()
                fd_ret = []
                _g_int = max(1, int(_g_blk_for_fd))
                for _i, _p in enumerate(_expert_ps_sorted):
                    _off = _grad_offsets[_i]
                    _ne_p = _p.numel()
                    _ng = _ne_p // _g_int
                    if _ng == 0:
                        fd_ret.append(torch.zeros(0, dtype=torch.bfloat16,
                                                  device=_h_accum_flat.device))
                        continue
                    _gv = _h_accum_flat[_off // _g_int : _off // _g_int + _ng]
                    # detach + clone so subsequent fast-path callers don't
                    # mutate the result while _fold_h is still reading.
                    fd_ret.append(_gv.detach().clone())
        finally:
            # Drop expert grad views (so del _grad_flat actually frees).
            if _grad_view_set:
                for _p in _expert_ps_sorted:
                    _p.grad = None
            if _grad_flat is not None:
                del _grad_flat
            # NOTE: do NOT del _h_accum_flat — it's a tracker buffer view.
            _h_accum_flat = None
            # Guaranteed θ_0 restore for the whole anchor session.
            with torch.no_grad():
                for _p, _snap in zip(_expert_ps_sorted, _theta_snap_session):
                    _p.data.copy_(_snap)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        # Average
        if fd_ret is not None and _n_succ > 1:
            for _h in fd_ret:
                _h.div_(_n_succ)
        if _should_write() and _n_anchor_batches > 1:
            print(f"[proposal] anchor multi-batch: N={_n_anchor_batches} succ={_n_succ}",
                  flush=True)
        _vram_audit.vram_audit_step("inside_anchor_fd_peak", dump_breakdown=False)
    finally:
        _config["_in_anchor_refresh"] = False
        # Restore DS flat grad buffer FIRST (before lp.grad reset so the
        # restored buffer reflects pre-anchor state exactly).
        if (_ds_engine_opt is not None and _saved_hp_grads_flat is not None):
            try:
                _fp32_grads = getattr(_ds_engine_opt, "fp32_groups_gradients_flat", None)
                if _fp32_grads is not None:
                    for cur, saved in zip(_fp32_grads, _saved_hp_grads_flat):
                        cur.copy_(saved)
            except Exception:
                pass
        # Re-register DS grad-acc hooks so next training iter's backward
        # accumulates into the flat buffer again.
        if _ds_engine_opt is not None and _saved_grad_acc_hooks is not None:
            try:
                _ds_engine_opt.create_grad_acc_hooks()
            except Exception:
                pass
        restore_training_state(_config.get("_inner_opt"), _train_state_snap)
        # Final belt-and-suspenders: force-clear any residual lp.grad.
        # model may be a list/tuple (Megatron DDP wrapper) — unwrap first.
        _m_clear = model[0] if isinstance(model, (list, tuple)) else model
        try:
            for _p in _m_clear.parameters():
                if _p.grad is not None:
                    _p.grad = None
        except Exception as _exc:
            print(f"[ANCHOR_FIX] zero_grad err: {_exc}", flush=True)
        # DIFF report — print any state that changed across anchor.
        if _full_diag and _pre_state is not None:
            _post_state = _capture_full_state(_ds_engine_opt)
            _keys = set(_pre_state) | set(_post_state)
            _diffs = []
            for k in sorted(_keys):
                a = _pre_state.get(k)
                b = _post_state.get(k)
                if a != b:
                    _diffs.append(f"  {k}: PRE={a} POST={b}")
            print(f"[ANCHOR_DIFF] @iter{iteration} n_diff={len(_diffs)}", flush=True)
            for d in _diffs[:60]:
                print(d, flush=True)
            if len(_diffs) > 60:
                print(f"  ... and {len(_diffs)-60} more", flush=True)
        # Fragmentation mitigation: empty_cache returns per-chunk loop's
        # transient free blocks (chunk_snaps, chunk_zs, gp, gm) to OS.
        # ~10 ms cost, prevents next iter's reserved-allocated waste growth.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if _offload_handle is not None:
        from ._ds_state import restore_adam_state_from_cpu
        _t_res0 = _time.perf_counter()
        restore_adam_state_from_cpu(
            _offload_handle, _config["_inner_opt"], _tracker)
        if _should_write():
            print(f"[proposal] anchor offload H2D "
                  f"{(_time.perf_counter()-_t_res0)*1000:.0f}ms", flush=True)
        _vram_audit.vram_audit_step("post_anchor_swapin", dump_breakdown=False)
    _vram_audit.vram_audit_step(
        "post_anchor", dump_breakdown=False,
        extra={"anchor_total_ms": (_time.perf_counter() - t0) * 1000.0},
    )
    # Anchor changes H scale → next iter's prev_threshold becomes invalid
    # on the new scale. _anchor_h_ratio (set after _fold_h above) is used
    # by tracker.step to rescale prev_threshold; this flag triggers it.
    _tracker._anchor_just_fired = True

    t1 = _time.perf_counter()
    fd_snapshots: Optional[Dict[int, List[torch.Tensor]]] = None
    if multi_k_track and isinstance(fd_ret, dict):
        fd_snapshots = fd_ret
        h_fd = fd_ret.get(int(multi_k_max)) if fd_ret else None
    else:
        h_fd = fd_ret

    if h_fd is None:
        if _should_write():
            print(f"[proposal] finite_diff failed @iter{iteration}", flush=True)
        return _finalize(None)

    # Capture H scale BEFORE fold (which overwrites _g_fast_h_g_bf16_flat
    # with new anchor distribution). Ratio is used by tracker.step to
    # rescale prev_threshold (computed on OLD H scale) onto the NEW H scale,
    # preventing all-scores-below-threshold pass2 dead-iter.
    try:
        _h_pre_fold = float(_tracker._g_fast_h_g_bf16_flat.abs().sum().item())
    except Exception:
        _h_pre_fold = None
    # Fold into H cache with outlier protection. Names follow the order
    # used to build h_fd: fast-flat re-sorted to tracker idx; legacy path
    # keeps original tracked order (names == _names_sorted in that case).
    _fold_diag = _fold_h(_names_sorted, h_fd,
                         flip_cutoff=_config.get("_finite_diff_flip_cutoff"))
    try:
        _h_post_fold = float(_tracker._g_fast_h_g_bf16_flat.abs().sum().item())
    except Exception:
        _h_post_fold = None
    if _h_pre_fold is not None and _h_post_fold is not None and _h_pre_fold > 0:
        _tracker._anchor_h_ratio = _h_post_fold / _h_pre_fold
    else:
        _tracker._anchor_h_ratio = 1.0

    elapsed_ms = (t1 - t0) * 1000.0
    n_elem = sum(p.numel() for p in expert_ps)

    # Resolve eps actually used: explicit override > default 1e-3.
    # (Autotune was removed when finite_diff became the only estimator.)
    eps_actual = float(eps) if eps is not None else 1e-3

    diag_record = {
        "iter": int(iteration), "estimator": "finite_diff",
        "expert_coverage": _fold_diag["expert_coverage"],
        "flip_skip_ratio": _fold_diag["flip_skip_ratio"],
        "eps": eps_actual,
        "routing_freeze": True,
        "one_sided": bool(_one_sided),
    }
    _config["_h_diag_records"].append(diag_record)

    # Magnitude log (|H| only — δ tracking is now group-level via accum_sq
    # and not exposed at refresh time).
    if _should_write():
        try:
            import statistics as _st
            h_abs_means = [float(h.mean().item()) for h in h_fd]
            h_abs_maxs = [float(h.max().item()) for h in h_fd]
            print(
                f"[proposal] scale @iter{iteration} (finite_diff): "
                f"|H| mean={_st.mean(h_abs_means):.3e} max={max(h_abs_maxs):.3e}  "
                f"(eps={eps_actual:.3e}, coverage={_fold_diag['expert_coverage']:.3f}, "
                f"flip_skip={_fold_diag['flip_skip_ratio']:.4f})",
                flush=True,
            )
        except Exception:
            pass

    # ---- Multi-K FD pearson + Jaccard logging ----
    # When --proposal-fd-multi-k-track is set, compute pearson + Jaccard@top%
    # for each K_x vs K_max, plus same vs prev_refresh K_max for stability.
    if fd_snapshots is not None:
        fd_snapshots.clear()

    if _should_write():
        print(
            f"[proposal] refresh_finite_diff @iter{iteration} K={fd_k_used} "
            f"{elapsed_ms:.0f} ms  ({n_elem:,} expert elements)",
            flush=True,
        )

    return _finalize({"iter": iteration, "k_samples": k_samples,
                      "ms": elapsed_ms, "numel": n_elem,
                      "diag": diag_record})



def proposal_shutdown() -> Dict[str, Any]:
    """Drain writers, dump write-logs + cpu-mem samples to disk, cleanup."""
    global _tracker, _fc, _writer, _enabled
    if not _enabled:
        return {}
    _h = _config.get("_release_hook_handle") if _config else None
    if _h is not None:
        try:
            _h.remove()
        except Exception:
            pass
        _config["_release_hook_handle"] = None
    import json
    summary: Dict[str, Any] = {}
    dump_dir: Optional[Path] = None
    dump_prefix = ""
    fl = _config.get("_fault_log_path") if _config else None
    if fl:
        dump_dir = Path(fl).parent
        # Extract tag from fault log basename: "{tag}.faults.jsonl" → tag prefix
        base = Path(fl).name
        if base.endswith(".faults.jsonl"):
            dump_prefix = base[: -len(".faults.jsonl")] + "."
    elif _cp_dir is not None:
        dump_dir = _cp_dir
    if _fc is not None:
        # Shutdown ExpertState: fsync any open lossy-FC fds and close them
        # cleanly so recovery sees a consistent on-disk snapshot.
        if _fc.expert_state is not None:
            try:
                _fc.expert_state.shutdown()
            except Exception as exc:
                print(f"[proposal] WARNING shutdown expert-state close failed: {exc}", flush=True)
        _fc.finalize()
        summary["fc_expert_events"] = sum(
            1 for s in _fc.write_log if s.kind == "expert"
        )
        summary["fc_ne_events"] = sum(
            1 for s in _fc.write_log if s.kind == "nonexpert"
        )
        if dump_dir is not None:
            try:
                (dump_dir / f"{dump_prefix}fc_write_log.json").write_text(
                    json.dumps(_fc.write_log_dict(), indent=2)
                )
            except Exception:
                pass
    if _writer is not None:
        ws = _writer.summary()
        summary["async_total_write_ms"] = ws.get("total_write_ms", 0.0)
        _writer.shutdown()
    # Audit: dump per-iter alloc trace summary to audit log (R2 evidence).
    try:
        from . import _vram_audit
        _vram_audit.dump_alloc_trace_to_log()
    except Exception:
        pass
    # Dump CPU memory samples + final summary.
    if dump_dir is not None and _cpu_mem_samples:
        try:
            (dump_dir / f"{dump_prefix}cpu_mem_samples.json").write_text(
                json.dumps(_cpu_mem_samples, indent=2)
            )
        except Exception:
            pass
    if dump_dir is not None:
        try:
            (dump_dir / f"{dump_prefix}proposal_shutdown_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
        except Exception:
            pass
    # Dump per-refresh diagnostic record (expert_coverage, flip_skip_ratio, eps).
    h_diag_records = _config.get("_h_diag_records") if _config else None
    if dump_dir is not None and h_diag_records:
        try:
            (dump_dir / f"{dump_prefix}h_diag_records.jsonl").write_text(
                "\n".join(json.dumps(r) for r in h_diag_records) + "\n"
            )
        except Exception:
            pass
    _enabled = False
    return summary


# add_proposal_args moved to cli_args.py (re-exported below for backward
# compatibility with pretrain_gpt.py's `from megatron.proposal.cp_hooks
# import add_proposal_args` call site, if any).
from .cli_args import add_proposal_args  # noqa: F401,E402


