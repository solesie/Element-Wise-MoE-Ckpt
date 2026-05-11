"""Integration hooks between Megatron-DeepSpeed's training loop and the proposal CP pipeline."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .async_writer import AsyncWriter
from .checkpoint_manager import CheckpointManager
from ._ds_state import Fp32StateMapper, set_mapper
from .fc_manager import FCManager, _detect_expert_layout, DS_EXPERT_IDX_RE
from .importance_tracker import ImportanceTracker
from .recovery import recover


_tracker: Optional[ImportanceTracker] = None
_ckpt: Optional[CheckpointManager] = None
_fc: Optional[FCManager] = None
_writer: Optional[AsyncWriter] = None
_expert_lossless_fc: Optional[Any] = None  # ExpertLosslessFC, lazy import
_ne_payload: Optional[Dict[str, torch.Tensor]] = None
_ne_payload_cuda_event: Optional["torch.cuda.Event"] = None
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
    global _tracker, _ckpt, _fc, _writer, _enabled, _cp_dir, _config
    global _expert_lossless_fc

    _enabled = bool(getattr(args, "proposal_enable", False))
    if not _enabled:
        return

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
        "dc_retention": int(getattr(args, "proposal_dc_retention", 0)),
        "async_queue": int(getattr(args, "proposal_async_queue", 5)),
        "cp_mode": cp_mode,
        "lossless_fc_fraction_pct": float(
            getattr(args, "proposal_lossless_fc_fraction", 2.0)
        ),
        "num_experts": num_experts_scalar,
        "num_experts_spec": num_experts_spec,
    }

    _writer = AsyncWriter(max_queue_size=_config["async_queue"]) if _should_write() else None

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

    # Non-expert DC retention: prefer the explicit ``--proposal-dc-
    # retention`` (default 20 — see cli_args). Floor of W+2 ensures at
    # least one FC supersedes the oldest DC; floor of 3 covers tiny W
    # corner cases. DC is non-expert only now.
    _dc_retention_arg = int(_config.get("dc_retention", 0))
    ne_retention = max(_dc_retention_arg, _config["nonexpert_fc_W"] + 2, 3)

    # CP layout: split non-expert + expert into separate subdirectories
    # so they can be mounted on different devices.
    #   <cp>/non_expert/  — non-expert dense DC + non-expert full FC
    #   <cp>/expert/      — expert rolling (lossy-FC) + legacy expert full FC
    # H_g is GPU-only (BF16 in ``_g_fast_h_g_bf16_flat``). Recovery
    # rebuilds it on the next FD refresh; no disk persist path.
    _nonexpert_cp_dir = _cp_dir / "non_expert"
    _expert_cp_dir = _cp_dir / "expert"

    if _should_write():
        _ckpt = CheckpointManager(
            output_dir=_nonexpert_cp_dir,
            retention=ne_retention,
            async_writer=_writer,
        )
    else:
        _ckpt = None

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
            named_params = dict(inner_model.named_parameters())
            return _fc._collect_param_payload(
                tnames, named_params, private=True,
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
    _config["_finite_diff_cross_check"] = bool(
        getattr(args, "proposal_finite_diff_cross_check", False)
    )
    _config["_finite_diff_cross_check_log"] = getattr(
        args, "proposal_finite_diff_cross_check_log", None,
    )
    _config["_finite_diff_flip_cutoff"] = float(
        getattr(args, "proposal_finite_diff_flip_cutoff", 100.0)
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


def proposal_dc_snap_grads(model) -> None:
    """Pre-optimizer hook (stage A): snapshot non-expert grads for DC.

    Must run BEFORE ``engine.step()``: DS engine calls
    ``optimizer.zero_grad(set_to_none=True)`` after the inner optimizer
    step, which sets ``param.grad = None`` and zeroes the FP32 master
    flat (via ``clear_hp_grads``). After step, no recoverable grad
    exists.

    Storage: pinned host BF16 (source dtype, no cast). Recovery casts
    BF16 → FP32 in ``_apply_payload_dict_to_model`` before AdamW
    replay. Since the source gradient is BF16 in DS BF16 training,
    storing as FP32 would only pad zeros without adding information —
    so BF16 storage is numerically lossless and saves 50% pinned host
    + 1 cast kernel per param + 50% PCIe bandwidth.

    Output: stashes (payload_dict, cuda_event) on module globals for
    consumption by ``proposal_dc_record`` (post-optimizer stage D).
    """
    global _ne_payload, _ne_payload_cuda_event
    if not _enabled or _tracker is None:
        return
    # Bench: skip dc_snap entirely (isolation flag for empirical
    # decomposition of per-iter overhead).
    import os as _os_bench_dc
    if _os_bench_dc.environ.get("PROPOSAL_BENCH_NO_DC_SNAP", "") in ("1", "true", "TRUE"):
        _ne_payload = None
        _ne_payload_cuda_event = None
        return

    # MoC-System baseline (sequential): no DC, no non-expert dense snapshot.
    seq_mode = _config["cp_mode"] == "sequential"
    if not seq_mode:
        import time as _time
        _t0 = _time.perf_counter()
        _ne_payload, _ne_stats = _tracker.snapshot_nonexpert_dc(
            global_step=0,
        )
        _ne_payload_cuda_event = getattr(_ne_stats, "cuda_event", None)
        _config["_phase_ms_dc_snap"] = (_time.perf_counter() - _t0) * 1000.0
    else:
        _ne_payload = None
        _ne_payload_cuda_event = None

    # accum_sq mode: snapshot expert θ before optimizer.step so update_delta_accum
    # can compute Δθ = θ_after - θ_before in post_step (way cheaper than re-
    # deriving from Adam m/v).
    try:
        _tracker.pre_step_snap()
    except Exception as _ps_exc:
        if _should_write():
            print(f"[proposal] pre_step_snap FAILED: {_ps_exc}", flush=True)

    # B2 race guard: main stream waits for the previous iter's tracker
    # work to finish reading m, v BEFORE this iter's optimizer.step
    # writes them. The wait is on a cuda Event (stream-level dependency,
    # not host-blocking) — typically a no-op because forward+backward
    # GPU work (~hundreds of ms) >> tracker work (~ms), but we add it
    # for safety. Without this, fast forward+backward could race with
    # tracker_stream's m/v read.
    prev_ev = getattr(_tracker, "_pending_tracker_event", None)
    if prev_ev is not None and torch.cuda.is_available():
        try:
            torch.cuda.current_stream().wait_event(prev_ev)
        except Exception:
            pass


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


def proposal_score_step(
    model, step: int,
) -> Optional[Tuple[List[Tuple[int, int]], bool]]:
    """Post-optimizer stage B: expert lossless FC + score compute.

    Runs AFTER ``engine.step()`` (m, v are now updated):
      ① ``ExpertLosslessFC.fire_if_due`` → fired list (round-robin
         F% expert lossless FC; collect+dispatch on tracker_stream).
      ② ``tracker.step`` → dispatch prev iter pending payload (small
         main-stream D2Hs) + launch tracker_stream score compute
         (Triton outlier kernel + group score chunk-loop). Must run
         BEFORE ``proposal_fc_fire`` (stage C) so dispatch's small D2H
         launches on the d2h_stream do not queue behind FC's 75 ms
         non-expert payload.

    The ``fired_experts`` list is passed into ``tracker.step`` as an
    exclusion set so dispatch can skip experts whose fresh LFCL would
    be clobbered by stale lossy entries.

    Returns ``(fired_experts, False)`` for backward compatibility with
    callers that read both the expert and non-expert fire flags from a
    single call. Non-expert flag is always False here — see
    ``proposal_fc_fire``.
    """
    if not _enabled or _tracker is None:
        return None
    import os as _os_bench_ss
    if _os_bench_ss.environ.get("PROPOSAL_BENCH_NO_SCORE_STEP", "") in ("1", "true", "TRUE"):
        return [], False

    seq_mode = _config["cp_mode"] == "sequential"
    inner_opt = _config["_inner_opt"]

    fired_experts: List[Tuple[int, int]] = []

    import time as _time

    if seq_mode:
        # MoC baseline runs all FC scheduling synchronously inside
        # proposal_fc_fire; nothing to do here.
        return fired_experts, False

    if _expert_lossless_fc is not None:
        if _expert_lossless_fc._tracker_stream is None:
            _expert_lossless_fc._tracker_stream = getattr(
                _tracker, "_tracker_stream", None,
            )
        fired_experts = _expert_lossless_fc.fire_if_due(step)

    _t_trk0 = _time.perf_counter()
    _tracker.step(
        inner_opt, global_step=step,
        lossless_fc_eids=fired_experts,
    )
    _config["_phase_ms_score_step"] = (_time.perf_counter() - _t_trk0) * 1000.0

    # Stash for proposal_dc_record (downstream stage may want fired list
    # for diagnostics; current pipeline does not).
    _config["_last_fired_experts"] = fired_experts
    return fired_experts, False


def proposal_fc_fire(
    model, step: int,
) -> bool:
    """Post-optimizer stage C: non-expert lossless FC (W-scheduled).

    Runs AFTER ``proposal_score_step`` (so dispatch's small D2H
    launches go onto d2h_stream first; then FC's 75 ms heavy load
    queues behind them on the same DMA engine without blocking
    dispatch).

    Action: when ``step == 1`` or ``step % W == 0``, snapshot all
    non-expert params + Adam (m, v) FP32 master state to a pinned host
    buffer (single batched D2H on d2h_stream), then queue a writer
    job to persist to disk. Otherwise no-op.

    Returns ``True`` iff a non-expert FC fire occurred this step.
    Side effect: stashes the flag on ``_config['_last_nonexpert_fired']``
    so ``proposal_dc_record`` can mark the DC bundle accordingly.
    """
    if not _enabled or _tracker is None:
        return False
    import os as _os_bench_fc
    if _os_bench_fc.environ.get("PROPOSAL_BENCH_NO_FC_FIRE", "") in ("1", "true", "TRUE"):
        _config["_last_nonexpert_fired"] = False
        return False

    seq_mode = _config["cp_mode"] == "sequential"

    import time as _time

    nonexpert_fired = False
    _t0 = _time.perf_counter()
    if _fc is not None:
        # Sequential mode reuses the same fire path — both branches just
        # call into FCManager. Difference is only in surrounding stages.
        _fired_unused, nonexpert_fired = _fc.fire_nonexpert_fc_if_due(step=step)
    if not seq_mode:
        _config["_phase_ms_fc_fire"] = (_time.perf_counter() - _t0) * 1000.0

    _config["_last_nonexpert_fired"] = nonexpert_fired
    return nonexpert_fired


def proposal_dc_record(
    model, step: int,
    ne_payload: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Post-optimizer stage D: queue DC bundle to writer.

    Runs LAST in the post-optimizer chain (after stages B, C). Consumes
    the ``ne_payload`` produced by ``proposal_dc_snap_grads`` (stage A)
    and the non-expert FC flag from ``proposal_fc_fire`` (stage C).

    ``ne_payload`` may be passed explicitly; if ``None`` the function
    falls back to the module-stashed ``_ne_payload`` (default path —
    matches the existing implicit handoff between stages A and D).

    The writer thread receives a ``cuda.Event`` (recorded on
    d2h_stream during stage A) and calls ``event.synchronize()`` on
    the worker side before reading the pinned host data — training
    thread never blocks.
    """
    global _ne_payload, _ne_payload_cuda_event
    if not _enabled or _tracker is None:
        return
    if _config["cp_mode"] == "sequential":
        return
    import os as _os_bench_dr
    if _os_bench_dr.environ.get("PROPOSAL_BENCH_NO_DC_RECORD", "") in ("1", "true", "TRUE"):
        _ne_payload = None
        _ne_payload_cuda_event = None
        return

    if ne_payload is None:
        ne_payload = _ne_payload
    nonexpert_fired = bool(_config.get("_last_nonexpert_fired", False))

    import time as _time
    _t0 = _time.perf_counter()
    if _ckpt is not None:
        _ckpt.record(
            step=step,
            nonexpert_payload=ne_payload or {},
            nonexpert_did_fc=nonexpert_fired,
            cuda_event=_ne_payload_cuda_event,
        )
    _config["_phase_ms_dc_record"] = (_time.perf_counter() - _t0) * 1000.0
    _ne_payload = None
    _ne_payload_cuda_event = None

    # Stage E: post-iter observability (probes, AsyncWriter / GPU mem
    # breakdown, Fisher coverage). All cheap or env-gated; runs after
    # writer queue submit so it never delays disk I/O.
    _proposal_post_iter_diagnostics(model, step)


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

    # Per-iter group probe (env-flagged): measures whether group score tracks
    # element score on PER-ITER δ (1 step accumulation). Captures the
    # production semantics where δ varies every iter.
    # Env (single window mode):
    #   PROPOSAL_PERITER_GROUP_PROBE_START=N PROPOSAL_PERITER_GROUP_PROBE_LEN=M
    #   → probe fires at iter [N, N+M)
    # Env (periodic mode):
    #   PROPOSAL_PERITER_GROUP_PROBE_PERIOD=P PROPOSAL_PERITER_GROUP_PROBE_LEN=M
    #   → probe fires at iter [0, M), [P, P+M), [2P, 2P+M), ...
    import os as _os_pi
    _pi_start = int(_os_pi.environ.get("PROPOSAL_PERITER_GROUP_PROBE_START", "0"))
    _pi_len = int(_os_pi.environ.get("PROPOSAL_PERITER_GROUP_PROBE_LEN", "0"))
    _pi_period = int(_os_pi.environ.get("PROPOSAL_PERITER_GROUP_PROBE_PERIOD", "0"))
    if _pi_period > 0 and _pi_len > 0:
        _pi_in_window = (step % _pi_period) < _pi_len
        # Reset snapshot at start of each new window so δ is 1-step within window
        if step % _pi_period == 0:
            _config.pop("_peritr_theta_snap", None)
    elif _pi_len > 0:
        _pi_in_window = (_pi_start <= step < _pi_start + _pi_len)
    else:
        _pi_in_window = False
    if _pi_in_window:
        try:
            from ._ds_state import get_mapper as _gm_pi
            _mp_pi = _gm_pi()
            _gs_pi = [int(g) for g in _os_pi.environ.get(
                "PROPOSAL_PERITER_GROUP_SIZES", "2,4,8,16,32").split(",")]
            _kpct_pi = float(_os_pi.environ.get(
                "PROPOSAL_PERITER_GROUP_K_PCT", "0.01"))
            tracked_pi = _tracker._tracked_names if hasattr(
                _tracker, "_tracked_names") else None
            if tracked_pi is not None:
                # Build flat H, θ_now (FP32 master)
                h_list_pi = []
                theta_now_pi = []
                names_pi = []
                # Production stores group-level H (_H_g, BF16, 1 entry per
                # production group). Probe needs element-level H to compare
                # different probe g values against the element-level baseline.
                # Broadcast _H_g back to element resolution — H locality is
                # high (CV_H ≪ CV_|δ|) so this is a faithful proxy for the
                # purposes of cap_g vs cap_e ratio measurement.
                prod_g = max(1, int(getattr(_tracker, "block_quant_g", 1)))
                for nm in tracked_pi:
                    p_pi = _tracker._name_to_param.get(nm)
                    if p_pi is None:
                        continue
                    h_g_view = _tracker._H_g.get(nm)
                    if h_g_view is None:
                        continue
                    h_elem = h_g_view.detach().to(torch.float32).repeat_interleave(prod_g)
                    max_g = max(_gs_pi)
                    n_t = (h_elem.numel() // max_g) * max_g
                    if n_t == 0:
                        continue
                    h_list_pi.append(h_elem[:n_t])
                    fp32_now = (_mp_pi.get_param(p_pi)
                                if _mp_pi is not None
                                else p_pi.data.to(torch.float32))
                    theta_now_pi.append((nm, fp32_now.detach().clone(), n_t))
                    names_pi.append(nm)
                if not h_list_pi:
                    # _H_g not yet populated (first refresh hasn't fired);
                    # silently skip until refresh runs.
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
                        delta_abs = delta_signed.abs()
                        delta_sq = delta_signed ** 2
                        score_elem = H_all_pi * delta_sq
                        total = float(score_elem.sum().item())
                        n_total = score_elem.numel()
                        k_elem = max(1, int(_kpct_pi * n_total))
                        if total > 0:
                            elem_th = float(torch.kthvalue(
                                score_elem, n_total - k_elem + 1).values.item())
                            elem_mask = score_elem >= elem_th
                            cap_e = float(score_elem[elem_mask].sum().item()) / total
                            # H-only top-K (δ²=1 가정): isolates impact of H block
                            # quantization on selection
                            H_total = float(H_all_pi.sum().item())
                            H_elem_th = float(torch.kthvalue(
                                H_all_pi, n_total - k_elem + 1).values.item())
                            H_elem_mask = H_all_pi >= H_elem_th
                            cap_h_elem = float(H_all_pi[H_elem_mask].sum().item()) / max(H_total, 1e-30)
                            # δ²-only top-K (H=1 가정): isolates δ² block quant
                            d2_total = float(delta_sq.sum().item())
                            d2_elem_th = float(torch.kthvalue(
                                delta_sq, n_total - k_elem + 1).values.item())
                            d2_elem_mask = delta_sq >= d2_elem_th
                            cap_d2_elem = float(delta_sq[d2_elem_mask].sum().item()) / max(d2_total, 1e-30)
                            for g in _gs_pi:
                                ng = n_total // g
                                # Match production (rank-fair):
                                #   H_g     = mean_{i∈group}(|H_i|)
                                #   δ_g     = (1/g) · Σ_{i∈group} Δθ_i²
                                #   score_g = H_g × δ_g
                                # Production storage uses bare Σ Δθ_i² (no
                                # /g) but g is uniform across groups so the
                                # 1/g constant is rank-invariant. Use mean
                                # form here for unit consistency with H_g.
                                H_g = H_all_pi.view(ng, g).mean(dim=1)
                                d_g_sq = delta_sq.view(ng, g).mean(dim=1)
                                score_g = H_g * d_g_sq
                                k_g = max(1, int(_kpct_pi * ng))
                                g_th = float(torch.kthvalue(
                                    score_g, ng - k_g + 1).values.item())
                                g_mask = score_g >= g_th
                                elem_from_g = g_mask.repeat_interleave(g)
                                cap_g = float(score_elem[elem_from_g].sum().item()) / total
                                # H-only group quantization: pick top groups by H_g
                                Hg_th = float(torch.kthvalue(
                                    H_g, ng - k_g + 1).values.item())
                                Hg_mask = H_g >= Hg_th
                                elem_from_Hg = Hg_mask.repeat_interleave(g)
                                cap_h_g = float(H_all_pi[elem_from_Hg].sum().item()) / max(H_total, 1e-30)
                                # δ²-only group quantization: pick top groups
                                # by Σ_{i∈g} δ_i² (matches production).
                                d2g_th = float(torch.kthvalue(
                                    d_g_sq, ng - k_g + 1).values.item())
                                d2g_mask = d_g_sq >= d2g_th
                                elem_from_d2g = d2g_mask.repeat_interleave(g)
                                cap_d2_g = float(delta_sq[elem_from_d2g].sum().item()) / max(d2_total, 1e-30)
                                # Pearson(score_elem, score_g_broadcast)
                                score_g_bcast = score_g.repeat_interleave(g)
                                se_mean = score_elem.mean()
                                sg_mean = score_g_bcast.mean()
                                se_d = score_elem - se_mean
                                sg_d = score_g_bcast - sg_mean
                                num = float((se_d * sg_d).sum().item())
                                den = float((se_d.norm() * sg_d.norm()).item())
                                pearson_sg = num / max(den, 1e-30)
                                d_abs_g = delta_abs.view(ng, g)
                                cv_d = (d_abs_g.std(dim=1) /
                                        d_abs_g.mean(dim=1).clamp_min(1e-30))
                                cv_d_med = float(cv_d.median().item())

                                # Outlier-promotion variant: union pool of
                                #   element entry  (score=H_g·Δθ²_i, weight=1)
                                #   group   entry  (score=H_g·δ_g,  weight=g)
                                # Greedy by score with element-equivalent
                                # budget K_total = k_elem.
                                H_g_be = H_g.repeat_interleave(g)
                                elem_pool = H_g_be * delta_sq
                                combined = torch.cat([elem_pool, score_g])
                                weights = torch.cat([
                                    torch.ones(elem_pool.numel(), device=combined.device),
                                    torch.full((ng,), float(g), device=combined.device),
                                ])
                                order = torch.argsort(combined, descending=True)
                                cum_w = torch.cumsum(weights[order], dim=0)
                                sel_n = int((cum_w < float(k_elem)).sum().item()) + 1
                                sel = torch.zeros_like(combined, dtype=torch.bool)
                                sel[order[:sel_n]] = True
                                e_sel = sel[:elem_pool.numel()]
                                g_sel = sel[elem_pool.numel():]
                                saved = e_sel | g_sel.repeat_interleave(g)
                                cap_promote = float(score_elem[saved].sum().item()) / total
                                alpha_e = float(e_sel.sum().item()) / max(int(k_elem), 1)

                                if _should_write():
                                    print(
                                        f"[proposal] peritr_probe g={g:>2d} "
                                        f"@iter{step}: cap_g={cap_g:.3f} "
                                        f"cap_promote={cap_promote:.3f} "
                                        f"(elem={cap_e:.3f}) "
                                        f"alpha_e={alpha_e:.2f} "
                                        f"cap_H={cap_h_g:.3f}/{cap_h_elem:.3f} "
                                        f"cap_d2={cap_d2_g:.3f}/{cap_d2_elem:.3f} "
                                        f"pearson={pearson_sg:.3f} "
                                        f"CV_|δ|={cv_d_med:.2f}",
                                        flush=True,
                                    )
                # Save snapshot for next iter
                _config["_peritr_theta_snap"] = {
                    nm: t for nm, t, _ in theta_now_pi
                }
        except Exception as _piexc:
            if _should_write():
                import traceback as _tb_pi
                print(f"[proposal] peritr_probe FAILED @iter{step}: "
                      f"{_piexc}\n{_tb_pi.format_exc()}", flush=True)

    # Diagnostic: every 10 iters, report AsyncWriter (torch + raw) queue
    # state + proposal phase breakdown. Identifies which bucket the
    # per-iter overhead lives in.
    if _writer is not None and step > 0 and step % 10 == 0:
        s = _writer.summary()
        qmax = _writer._queue.maxsize if hasattr(_writer, "_queue") else -1
        qrmax = _writer._queue_raw.maxsize if hasattr(_writer, "_queue_raw") else -1
        ckpt_log = _ckpt.write_log_dict() if _ckpt is not None else []
        recent = ckpt_log[-10:] if ckpt_log else []
        avg_queue_delay = (
            sum(r.get("queue_delay_ms", 0.0) for r in recent) / max(len(recent), 1)
        )
        avg_write_ms = (
            sum(r.get("write_time_ms", 0.0) for r in recent) / max(len(recent), 1)
        )
        phase_dc_snap = _config.get("_phase_ms_dc_snap", 0.0)
        phase_fc_fire = _config.get("_phase_ms_fc_fire", 0.0)
        phase_score = _config.get("_phase_ms_score_step", 0.0)
        phase_dc_rec = _config.get("_phase_ms_dc_record", 0.0)
        trk_phases = getattr(_tracker, "last_phase_ms", None) or {}
        tp1 = trk_phases.get("pass1", 0.0)
        tp2 = trk_phases.get("pass2", 0.0)
        tp3 = trk_phases.get("pass3", 0.0)
        t_setup = trk_phases.get("setup", 0.0)
        t_post = trk_phases.get("post", 0.0)
        t_dispatch = trk_phases.get("dispatch", 0.0)
        t_launch = trk_phases.get("launch", 0.0)
        tick_sub = getattr(_fc, "last_tick_phase_ms", None) or {} if _fc is not None else {}
        t_col = tick_sub.get("collect_expert_ms", 0.0)
        t_cb = tick_sub.get("on_expert_fc_ms", 0.0)
        t_dp = tick_sub.get("lossless_dispatch_ms", 0.0)
        t_ne = tick_sub.get("nonexpert_fc_ms", 0.0)
        n_fired = int(tick_sub.get("n_experts_fired", 0.0))
        ne_col = getattr(_fc, "_last_nonexpert_collect_ms", 0.0) if _fc is not None else 0.0
        ne_sub = getattr(_fc, "_last_nonexpert_collect_sub_ms", {}) if _fc is not None else {}
        ne_enum = ne_sub.get("enum_ms", 0.0)
        ne_copy = ne_sub.get("copy_ms", 0.0)
        ne_xfer = ne_sub.get("xfer_ms", 0.0)
        ne_wait = ne_sub.get("wait_ms", 0.0)
        ne_n = int(ne_sub.get("nentries", 0))
        dc_sub = getattr(_tracker, "last_dc_sub_ms", {}) or {}
        dc_xfer = dc_sub.get("xfer_ms", 0.0)
        dc_copy = dc_sub.get("copy_ms", 0.0)
        dc_enum = dc_sub.get("enum_ms", 0.0)
        dc_wait = dc_sub.get("wait_ms", 0.0)
        print(
            f"[proposal] AsyncWriter@iter{step}: "
            f"torch pending={s['pending']}/{qmax} max={s['max_queue_depth']} "
            f"subs={s['submissions']} comps={s['completions']}  "
            f"raw pending={s['pending_raw']}/{qrmax} max={s['max_queue_depth_raw']} "
            f"subs={s['submissions_raw']} comps={s['completions_raw']}  "
            f"torch-write last10 q_delay={avg_queue_delay:.1f}ms "
            f"write={avg_write_ms:.1f}ms  "
            f"phase_ms[dc_snap={phase_dc_snap:.1f}"
            f"(enum={dc_enum:.1f}+copy={dc_copy:.1f}+wait={dc_wait:.1f}+"
            f"xfer={dc_xfer:.1f}) score_step={phase_score:.1f}"
            f"(dispatch={t_dispatch:.1f}+launch={t_launch:.1f}"
            f": setup={t_setup:.1f}+p1={tp1:.1f}+p2={tp2:.1f}+p3={tp3:.1f}"
            f"+post={t_post:.1f}) fc_fire={phase_fc_fire:.1f}"
            f"(nfired={n_fired} collect={t_col:.1f}+cb={t_cb:.1f}+"
            f"dispatch={t_dp:.1f}+ne={t_ne:.1f}"
            f"[col={ne_col:.1f}:enum={ne_enum:.1f}+copy={ne_copy:.1f}+"
            f"wait={ne_wait:.1f}+xfer={ne_xfer:.1f} N={ne_n}]) "
            f"dc_record={phase_dc_rec:.1f}]",
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
                items_str = " ".join(f"{n}={v:.1f}" for n, v in items)
                _baseline_mb = globals().get("_baseline_alloc_mb", 0.0)
                added = cuda_alloc - _baseline_mb
                tracked_total = tracker_total + fc_total
                untracked = added - tracked_total
                # Peak (max) memory tracking — captures TRANSIENT spikes
                # caused by dispatch's per-iter GPU intermediates that
                # ``alloc`` (steady-state allocated) misses. Reset after
                # each log so each report's peak reflects the iters
                # since the previous log only.
                cuda_peak = torch.cuda.max_memory_allocated() / 1e6
                peak_added = cuda_peak - _baseline_mb
                torch.cuda.reset_peak_memory_stats()
                print(
                    f"[proposal] GPU mem @iter{step}: "
                    f"alloc={cuda_alloc:.1f}MB Δfrom_baseline={added:+.1f}MB "
                    f"peak={cuda_peak:.1f}MB Δpeak={peak_added:+.1f}MB "
                    f"proposal_buffers={tracked_total:.1f}MB "
                    f"untracked={untracked:+.1f}MB | {items_str}",
                    flush=True,
                )
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
        dc_replay=True,
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
        f"ne_fc={diag.nonexpert_fc_iter} expert_fc_count={diag.expert_fc_count} "
        f"dc_replayed={diag.dc_bundles_replayed}/{list(diag.replay_range)}",
        flush=True,
    )
    return {
        "recover_time_s": float(recover_s),
        "recover_time_main_s": float(_t_rec1 - _t_rec0),
        "recover_time_h_s": float(_t_rec2 - _t_rec1),
        "nonexpert_fc_iter": diag.nonexpert_fc_iter,
        "expert_fc_count": diag.expert_fc_count,
        "dc_bundles_replayed": diag.dc_bundles_replayed,
        "dc_bundles_found": diag.dc_bundles_found,
        "replay_range": list(diag.replay_range),
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
        dc_replay=True,
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
        "dc_bundles_replayed": int(diag.dc_bundles_replayed),
        "replay_range": list(diag.replay_range),
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
    g_blk = max(1, int(getattr(_tracker, "block_quant_g", 1)))
    n_skipped = 0
    for i, (name, h_new) in enumerate(zip(names, h_new_list)):
        if not coverage_flags[i] or skip_mask[i]:
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
        # Element-level path (Hutch fallback / multi-K diagnostic).
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
    return {
        "expert_coverage": expert_coverage,
        "flip_skip_ratio": float(n_skipped) / max(n, 1),
        "n_skipped": int(n_skipped),
    }


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

    Optional cross-check: when ``--proposal-finite-diff-cross-check`` is
    set, Hutchinson is run on the same param state and the Hutch-vs-FD
    Pearson/Jaccard is dumped via :func:`compare_hessian_estimators`.
    """
    from .hessian import (
        compute_hessian_finite_diff as _finite_diff_compute_H,
        compute_hessian_hutchinson as _hutchinson_compute_H,
        compare_hessian_estimators as _compare_hessian_estimators,
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
    cross_check_on = bool(_config.get("_finite_diff_cross_check", False))

    # CRN compliance for cross-check: capture the batch ONCE at refresh
    # entry, then feed the same batch tensor to BOTH FD and Hutchinson.
    # Per the original spec ("±ε 양쪽이 동일 batch tensor를 사용") the
    # CRN principle should also extend to any direct comparison between
    # estimators — otherwise batch-to-batch Hessian variance dominates
    # the cross-check Pearson and masks the actual estimator agreement.
    crn_batch = None
    crn_step_func = forward_step_func
    crn_di = data_iterator
    if cross_check_on:
        try:
            from pretrain_gpt import get_batch as _get_batch_gpt
            di_for_capture = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
            crn_batch = _get_batch_gpt(di_for_capture)
            tokens, labels, loss_mask, attn_mask, position_ids = crn_batch[:5]

            import torch.nn.functional as _F
            def _crn_step(_di, m):
                """forward_step_func wrapper that ignores the iterator and
                uses the pre-captured CRN batch. Mirrors Megatron's LM
                forward + cross-entropy reduction. Used for the
                cross-check Hutchinson call so it sees the SAME batch FD
                used."""
                out = m(tokens, position_ids, attn_mask)
                logits = out[0] if isinstance(out, tuple) else out
                if logits is None or not torch.is_tensor(logits):
                    raise RuntimeError("CRN: model returned non-tensor")
                if logits.dim() >= 2:
                    flat = logits.reshape(-1, logits.size(-1))
                else:
                    flat = logits
                if labels is None:
                    raise RuntimeError("CRN: labels missing")
                lbl = labels.reshape(-1)
                if loss_mask is not None and loss_mask.numel() == flat.size(0):
                    per_tok = _F.cross_entropy(flat.float(), lbl, reduction="none")
                    denom = loss_mask.view(-1).float().sum().clamp_min(1.0)
                    loss = (per_tok * loss_mask.view(-1).float()).sum() / denom
                else:
                    loss = _F.cross_entropy(flat.float(), lbl, reduction="mean")
                return loss, {}
            crn_step_func = _crn_step
            # Stub iterator — _crn_step ignores it.
            class _StubIter:
                def __next__(self): return None
            crn_di = _StubIter()
        except Exception as _exc:
            if _should_write():
                print(f"[proposal] CRN batch capture failed: {_exc} — "
                      f"falling back to per-estimator batch (CRN violation, "
                      f"cross-check Pearson will reflect batch noise)", flush=True)
            crn_batch = None
            crn_step_func = forward_step_func
            crn_di = data_iterator

    # PROPOSAL_FD_ONE_SIDED=1 → forward-difference instead of central.
    # Cuts fwd+bwd from 2K to K+1 (47% at K=16) at the cost of O(ε) vs
    # O(ε²) truncation. Validated against central baseline — see
    # results/exp_etc_FDvsHutch_K16_* runs.
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
    # Pass g_blk so the FD core accumulates BF16 directly into a
    # per-group buffer of size ``numel/g_blk`` (saves ~700 MB transient
    # at production g=256; final |H_g| is BF16 per CLAUDE.md spec).
    # At g_blk=1 the FD core still produces BF16, just element-level
    # shape (group of 1 = element). Multi-K diagnostic mode forces
    # FP32 element-level accumulation inside the core
    # (``track_k_values`` branch) so cross-K Pearson is computed on a
    # shared FP32 axis. When cross-check is ON we force g_blk=1 so FD
    # output is element-shape, matching Hutch's element-level FP32 for
    # the per-element Pearson/Jaccard (dtype promotion handled in
    # ``compare_hessian_estimators``).
    _g_blk_for_fd = max(1, int(getattr(_tracker, "block_quant_g", 1)))
    if cross_check_on:
        _g_blk_for_fd = 1
    fd_ret = _finite_diff_compute_H(
        model, crn_step_func if crn_batch is not None else forward_step_func,
        crn_di if crn_batch is not None else data_iterator,
        expert_ps, fd_k_used,
        eps=eps, use_routing_freeze=True,
        batch_override=crn_batch,
        one_sided=_one_sided,
        track_k_values=(multi_k_values if multi_k_track else None),
        g_blk=_g_blk_for_fd,
    )
    t1 = _time.perf_counter()
    fd_snapshots: Optional[Dict[int, List[torch.Tensor]]] = None
    if multi_k_track and isinstance(fd_ret, dict):
        fd_snapshots = fd_ret
        h_fd = fd_ret.get(int(multi_k_max)) if fd_ret else None
    else:
        h_fd = fd_ret

    if h_fd is None:
        # Graceful fallback to Hutchinson — print warning and re-run.
        if _should_write():
            print(f"[proposal] finite_diff failed @iter{iteration}, falling back "
                  f"to hutchinson", flush=True)
        h_hutch = _hutchinson_compute_H(model, forward_step_func, data_iterator,
                                         expert_ps, k_samples)
        if h_hutch is None:
            return _finalize(None)
        _fold_diag = _fold_h(names, h_hutch,
                             flip_cutoff=_config.get("_finite_diff_flip_cutoff"))
        _config["_h_diag_records"].append({
            "iter": int(iteration), "estimator": "fallback_hutch",
            "expert_coverage": _fold_diag["expert_coverage"],
            "flip_skip_ratio": _fold_diag["flip_skip_ratio"],
        })
        elapsed_ms = (_time.perf_counter() - t0) * 1000.0
        return _finalize({"iter": iteration, "k_samples": k_samples,
                          "ms": elapsed_ms, "fallback": True})

    # Fold into H cache with outlier protection.
    _fold_diag = _fold_h(names, h_fd,
                         flip_cutoff=_config.get("_finite_diff_flip_cutoff"))

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
    # h_fd above is K_max snapshot already.
    if (fd_snapshots is not None and multi_k_max is not None
            and len(multi_k_values) >= 1):
        try:
            from .hessian import compare_hessian_estimators as _cmp_h
            ref_list = fd_snapshots[int(multi_k_max)]
            # δ-weighted score requires theta_cp (legacy); fast path stores
            # δ_g per-group only. Pass empty dict so the comparator computes
            # bare-|H| Pearson/Jaccard (the production-relevant metrics).
            theta_cp_for_cmp: Dict[str, torch.Tensor] = {}

            rec: Dict[str, Any] = {
                "iter": int(iteration),
                "k_max": int(multi_k_max),
                "values": list(multi_k_values),
            }
            console_pieces: List[str] = []
            for kk in multi_k_values:
                if kk == multi_k_max or kk not in fd_snapshots:
                    continue
                cand_list = fd_snapshots[int(kk)]
                agg = _cmp_h(
                    iteration, tracked, cand_list, ref_list,
                    theta_cp_for_cmp,
                    label_a=f"K{kk}", label_b=f"K{multi_k_max}",
                )
                p_bare_key = next((k for k in agg if k.startswith("pearson_") and k.endswith("_bare")), None)
                p_bare = float(agg.get(p_bare_key, 0.0)) if p_bare_key else 0.0
                rec[f"K{kk}_pearson_bare"] = p_bare
                rec[f"K{kk}_jaccard_top0.1pct_bare"] = float(agg.get("jaccard_top0.1pct_bare", 0.0))
                rec[f"K{kk}_jaccard_top1pct_bare"] = float(agg.get("jaccard_top1pct_bare", 0.0))
                rec[f"K{kk}_jaccard_top5pct_bare"] = float(agg.get("jaccard_top5pct_bare", 0.0))
                p_score_key = next((k for k in agg if k.startswith("pearson_") and k.endswith("_score")), None)
                rec[f"K{kk}_pearson_score"] = float(agg.get(p_score_key, 0.0)) if p_score_key else 0.0
                rec[f"K{kk}_jaccard_top1pct_score"] = float(agg.get("jaccard_top1pct_score", 0.0))
                console_pieces.append(
                    f"K{kk}=P{p_bare:+.2f}/J@1%{rec[f'K{kk}_jaccard_top1pct_bare']:.2f}"
                )

            # prev-refresh stability
            prev_list_cpu = _config.get("_fd_multi_k_prev_max_list")
            if (prev_list_cpu is not None and len(prev_list_cpu) == len(ref_list)
                    and all(p.numel() == r.numel() for p, r in zip(prev_list_cpu, ref_list))):
                prev_list_gpu = [p.to(r.device, non_blocking=True)
                                 for p, r in zip(prev_list_cpu, ref_list)]
                agg_prev = _cmp_h(
                    iteration, tracked, prev_list_gpu, ref_list,
                    theta_cp_for_cmp,
                    label_a="prev", label_b=f"K{multi_k_max}",
                )
                p_bare_key = next((k for k in agg_prev if k.startswith("pearson_") and k.endswith("_bare")), None)
                pp = float(agg_prev.get(p_bare_key, 0.0)) if p_bare_key else 0.0
                rec["prev_pearson_bare"] = pp
                rec["prev_jaccard_top0.1pct_bare"] = float(agg_prev.get("jaccard_top0.1pct_bare", 0.0))
                rec["prev_jaccard_top1pct_bare"] = float(agg_prev.get("jaccard_top1pct_bare", 0.0))
                rec["prev_jaccard_top5pct_bare"] = float(agg_prev.get("jaccard_top5pct_bare", 0.0))
                console_pieces.append(
                    f"prev=P{pp:+.2f}/J@1%{rec['prev_jaccard_top1pct_bare']:.2f}"
                )
                del prev_list_gpu
            else:
                rec["prev_pearson_bare"] = None
                console_pieces.append("prev=N/A")
            # Cache CURRENT K_max snapshot to CPU for next refresh's prev-comparison.
            _config["_fd_multi_k_prev_max_list"] = [t.detach().cpu() for t in ref_list]

            if _should_write():
                print(
                    f"[proposal] multi_k @iter{iteration} K_max={multi_k_max}: "
                    + " ".join(console_pieces),
                    flush=True,
                )
            if multi_k_log_path and _should_write():
                import json as _json_mk
                with open(multi_k_log_path, "a") as _f:
                    _f.write(_json_mk.dumps(rec) + "\n")
            fd_snapshots.clear()
        except Exception as _mkexc:
            if _should_write():
                import traceback as _tb_mk
                print(f"[proposal] multi_k FAILED @iter{iteration}: "
                      f"{_mkexc}\n{_tb_mk.format_exc()}", flush=True)

    # ---- Optional cross-check: 4-way estimator comparison vs truth + ----
    # ---- per-estimator temporal stability (i-th vs (i+1)-th refresh). ----
    if cross_check_on:
        try:
            from .hessian import _unwrap_model as _uw
            from ._routing_freeze import (
                RoutingFreezeContext as _RFC,
                identify_gate_modules_safe as _id_gates,
            )
            gates_xc, _ = _id_gates(model)
            hutch_step = (crn_step_func if crn_batch is not None
                          else forward_step_func)
            hutch_di = (crn_di if crn_batch is not None else data_iterator)

            # PROPOSAL_TRUTH_K: Hutch K used as truth reference (default 64).
            # PROPOSAL_FD_KS:    comma list of FD K values (default "4,8,16").
            # PROPOSAL_HUTCH_KS: comma list of Hutch K values (default "4").
            import os as _os_xc
            truth_k = int(_os_xc.environ.get("PROPOSAL_TRUTH_K", "64"))
            fd_ks = [int(x) for x in _os_xc.environ.get(
                "PROPOSAL_FD_KS", "4,8,16").split(",") if x.strip()]
            hutch_ks = [int(x) for x in _os_xc.environ.get(
                "PROPOSAL_HUTCH_KS", "4").split(",") if x.strip()]

            # Iter-varying base seed + per-candidate offset breaks two
            # artifacts:
            #  (a) vs-PREV: same z reused across refreshes makes the noise
            #      term pick the same "spurious top" consistently → inflated
            #      J@.01%. Iter mixing fixes this.
            #  (b) vs-TRUTH: truth (Hutch K=64) and a candidate (e.g. Hutch
            #      K=4) sharing seed → candidate's z's are a SUBSET of
            #      truth's → inflated J@X from sample reuse, not from real
            #      H agreement. Per-candidate offset fixes this — each
            #      estimator draws independent z's from the same batch.
            _seed_base = 0xBADCAFE ^ (int(iteration) * 2654435761 & 0xFFFFFFFF)

            def _seed_for(name: str) -> int:
                # Stable per-name offset (hash) so the same candidate always
                # gets the same offset across refreshes — only iteration
                # varies the absolute seed.
                _h = 0
                for _c in name:
                    _h = (_h * 131 + ord(_c)) & 0xFFFFFFFF
                return (_seed_base ^ _h) & 0xFFFFFFFF

            def _run_hutch(k, step_fn, di, seed: int):
                """Hutchinson with routing freeze (caller-wrapped)."""
                if gates_xc:
                    with _RFC(gates_xc) as _ctx_h:
                        try:
                            _ctx_h.capture(
                                lambda: step_fn(di, _uw(model))
                            )
                        except Exception:
                            pass
                        return _hutchinson_compute_H(
                            model, step_fn, di, expert_ps, k,
                            rng_seed=seed,
                        )
                return _hutchinson_compute_H(
                    model, step_fn, di, expert_ps, k,
                    rng_seed=seed,
                )

            def _run_fd(k, batch, step_fn, di, seed: int):
                return _finite_diff_compute_H(
                    model, step_fn, di, expert_ps, k,
                    eps=eps, use_routing_freeze=True,
                    batch_override=batch, one_sided=False,
                    rng_seed=seed,
                )

            def _build_candidates(batch, step_fn, di, include_truth: bool):
                """Compute Hutch + FD candidates against (batch, step_fn, di).
                Each estimator gets an independent seed so vs-TRUTH J@X
                isn't inflated by Rademacher z subset-sharing.
                Returns (h_truth or None, dict[name] -> H tensors)."""
                _truth = (_run_hutch(truth_k, step_fn, di, _seed_for("truth"))
                          if include_truth else None)
                _cands: Dict[str, Optional[List[torch.Tensor]]] = {}
                for kk in hutch_ks:
                    nm = f"hutch_k{kk}"
                    _cands[nm] = _run_hutch(kk, step_fn, di, _seed_for(nm))
                for kk in fd_ks:
                    nm = f"fd_k{kk}"
                    _cands[nm] = _run_fd(kk, batch, step_fn, di, _seed_for(nm))
                return _truth, _cands

            # ---- Steps 1+2: truth + candidate estimators on batch A (CRN) ----
            h_truth, candidates = _build_candidates(
                crn_batch, hutch_step, hutch_di, include_truth=True,
            )
            if h_truth is None:
                raise RuntimeError("truth Hutch K=%d failed" % truth_k)

            cross_log = _config.get("_finite_diff_cross_check_log")
            import json as _json

            def _emit(agg, tag, name):
                key_p = next((k for k in agg
                              if k.startswith("pearson_") and k.endswith("_bare")), None)
                p = float(agg.get(key_p, 0.0)) if key_p else 0.0
                if _should_write():
                    print(
                        f"[proposal] {tag} @iter{iteration}: {name:>10s} "
                        f"P={p:+.3f} "
                        f"J@.01%={agg.get('jaccard_top0.01pct_bare',0):.3f} "
                        f"J@.1%={agg.get('jaccard_top0.1pct_bare',0):.3f} "
                        f"J@1%={agg.get('jaccard_top1pct_bare',0):.3f} "
                        f"J@5%={agg.get('jaccard_top5pct_bare',0):.3f} "
                        f"J@10%={agg.get('jaccard_top10pct_bare',0):.3f}",
                        flush=True,
                    )
                    if cross_log:
                        agg_out = dict(agg)
                        agg_out["tag"] = tag
                        agg_out["name"] = name
                        Path(cross_log).parent.mkdir(parents=True, exist_ok=True)
                        with open(cross_log, "a") as _f:
                            _f.write(_json.dumps(agg_out) + "\n")

            # ---- Step 3: vs-TRUTH per candidate ----
            for name, h in candidates.items():
                if h is None:
                    if _should_write():
                        print(f"[proposal] vs-TRUTH @iter{iteration}: {name} FAILED",
                              flush=True)
                    continue
                agg = _compare_hessian_estimators(
                    iteration, tracked, h_truth, h, {},
                    label_a="truth", label_b=name,
                )
                agg["truth_k"] = int(truth_k)
                _emit(agg, "vs-TRUTH", name)

            # ---- Step 4: temporal stability — prev refresh vs current per estimator ----
            prev_hs = _config.get("_prev_cross_h", {})
            for name, h in candidates.items():
                if h is None:
                    continue
                prev_h = prev_hs.get(name)
                if prev_h is None:
                    continue
                agg_t = _compare_hessian_estimators(
                    iteration, tracked, prev_h, h, {},
                    label_a="prev", label_b=name,
                )
                _emit(agg_t, "vs-PREV", name)

            # Save current candidates for next-refresh comparison.
            # Memory: |candidates| × |expert_ps| FP32. For 4M expert × 4 cands ≈ 64 MB.
            _config["_prev_cross_h"] = {
                name: [t.detach().clone() for t in h]
                for name, h in candidates.items() if h is not None
            }

            # ---- Step 4d: Group-wise scoring sweep ----
            # PROPOSAL_GROUP_SIZE_PROBE=1: for each group_size in {2,4,8,16,32}:
            #   capture_rate = ΔL fraction protected by group-wise top-K (vs element)
            #   missed_severity = max/sum of element scores missed
            #   CV_H, CV_delta = within-group spread (lower = more correlation)
            # Uses TRUTH H (Hutch K=64) and current θ - θ_CP for δ.
            group_probe_on = _os_xc.environ.get(
                "PROPOSAL_GROUP_SIZE_PROBE", "") in ("1", "true", "TRUE")
            if group_probe_on:
                try:
                    K_PCT = float(_os_xc.environ.get(
                        "PROPOSAL_GROUP_K_PCT", "0.01"))
                    GROUP_SIZES = [int(g) for g in _os_xc.environ.get(
                        "PROPOSAL_GROUP_SIZES", "2,4,8,16,32").split(",")]
                    # Concatenate ALL expert tensors into one flat vector for
                    # global top-K analysis (matches production behavior).
                    # δ_fresh = |FP32(θ_now) - FP32(θ_prev_snapshot)|
                    # Use FP32 master weights from optimizer (mapper.get_param)
                    # to avoid BF16 quantization noise (BF16 ULP ≈ 5e-4 at θ=0.1
                    # > typical 40-iter Δθ ~ 4e-4 → most elements get rounded to 0).
                    from ._ds_state import get_mapper as _get_mapper_gp
                    _mapper_gp = _get_mapper_gp()
                    h_flat_list = []
                    delta_fresh_list = []
                    theta_now_list = []  # for next probe's snapshot
                    have_fresh = "_group_probe_theta_snap" in _config
                    if have_fresh:
                        prev_snap = _config["_group_probe_theta_snap"]
                    for (name, p), h in zip(tracked, h_truth):
                        max_g = max(GROUP_SIZES)
                        n_tensor = (h.numel() // max_g) * max_g
                        h_flat_list.append(h.detach().reshape(-1)[:n_tensor].to(torch.float32))
                        # FP32 master weight (precision-safe for δ measurement)
                        fp32_now = (_mapper_gp.get_param(p)
                                    if _mapper_gp is not None
                                    else p.data.to(torch.float32))
                        if have_fresh and name in prev_snap:
                            # Keep SIGNED delta (no abs) so we can measure both:
                            #   mean(|δ|)        — current (no cancellation)
                            #   |mean(δ_signed)| — production group-reset semantics
                            d_signed_fresh = (fp32_now - prev_snap[name]).reshape(-1)[:n_tensor]
                            delta_fresh_list.append(d_signed_fresh)
                        theta_now_list.append((name, fp32_now.detach().clone()))
                    H_all = torch.cat(h_flat_list)
                    if have_fresh and len(delta_fresh_list) == len(h_flat_list):
                        delta_fresh_all = torch.cat(delta_fresh_list)
                    else:
                        delta_fresh_all = None
                    # Save snapshot for next probe (memory: ~|expert_ps| FP32 = 2× BF16)
                    _config["_group_probe_theta_snap"] = {
                        name: t for name, t in theta_now_list
                    }
                    # Only fresh delta now (rot removed; was rotation-noise-dominated)
                    if delta_fresh_all is None:
                        if _should_write():
                            print(f"[proposal] group_probe @iter{iteration}: "
                                  f"snapshot init only", flush=True)
                        delta_variants = []
                    else:
                        delta_variants = [("fresh", delta_fresh_all)]
                    # delta_all is now SIGNED. Compute element-wise abs for element score.
                    for delta_label, delta_signed_all in delta_variants:
                        delta_abs_all = delta_signed_all.abs()
                        delta_sq_all = delta_signed_all ** 2  # δ² (sign doesn't matter)
                        score_elem = H_all * delta_sq_all
                        total_score = float(score_elem.sum().item())
                        n_total = score_elem.numel()
                        k_elem = max(1, int(K_PCT * n_total))
                        if total_score <= 0:
                            if _should_write():
                                print(f"[proposal] group_probe[{delta_label}] "
                                      f"@iter{iteration}: score sum=0, skipped",
                                      flush=True)
                            continue
                        elem_threshold = float(torch.kthvalue(
                            score_elem, n_total - k_elem + 1).values.item())
                        elem_mask = score_elem >= elem_threshold
                        captured_elem = float(score_elem[elem_mask].sum().item())
                        capture_rate_elem = captured_elem / total_score

                        delta_topk_th = float(torch.kthvalue(
                            delta_abs_all, n_total - k_elem + 1).values.item())
                        delta_top_mask = delta_abs_all >= delta_topk_th
                        H_topk_th = float(torch.kthvalue(
                            H_all, n_total - k_elem + 1).values.item())
                        H_top_mask = H_all >= H_topk_th

                        for g in GROUP_SIZES:
                            n_groups = n_total // g
                            H_g = H_all.view(n_groups, g).mean(dim=1)
                            # Three group score variants:
                            #   _abs:  mean(|δ|)²        — no sign cancellation, /g² scaling
                            #   _sgn:  |mean(δ_signed)|² — sign cancellation, /g² scaling
                            #   _prod: Σ_{i∈g} δ_i²      — PRODUCTION (matches
                            #          _accum_delta_sq_triton kernel + _fold_h
                            #          mean-H rule). This is the only variant
                            #          that fairly mirrors what production
                            #          ranks groups by.
                            d_g_abs = delta_abs_all.view(n_groups, g).mean(dim=1)
                            d_g_signed = delta_signed_all.view(n_groups, g).mean(dim=1).abs_()
                            # _prod: production-fair δ_g = (1/g)·Σ Δθ_i²
                            d_g_prod = delta_sq_all.view(n_groups, g).mean(dim=1)
                            score_g_abs = H_g * (d_g_abs ** 2)
                            score_g_signed = H_g * (d_g_signed ** 2)
                            score_g_prod = H_g * d_g_prod
                            k_g = max(1, int(K_PCT * n_groups))
                            # Capture rate for ABS variant
                            g_thr_abs = float(torch.kthvalue(
                                score_g_abs, n_groups - k_g + 1).values.item())
                            g_mask_abs = score_g_abs >= g_thr_abs
                            elem_from_abs = g_mask_abs.repeat_interleave(g)
                            cap_abs = float(score_elem[elem_from_abs].sum().item()) / total_score
                            # Capture rate for SIGNED variant
                            g_thr_signed = float(torch.kthvalue(
                                score_g_signed, n_groups - k_g + 1).values.item())
                            g_mask_signed = score_g_signed >= g_thr_signed
                            elem_from_signed = g_mask_signed.repeat_interleave(g)
                            cap_signed = float(score_elem[elem_from_signed].sum().item()) / total_score
                            # Capture rate for PRODUCTION variant
                            g_thr_prod = float(torch.kthvalue(
                                score_g_prod, n_groups - k_g + 1).values.item())
                            g_mask_prod = score_g_prod >= g_thr_prod
                            elem_from_prod = g_mask_prod.repeat_interleave(g)
                            cap_prod = float(score_elem[elem_from_prod].sum().item()) / total_score
                            # CVs (use abs for CV_δ as ranking is on magnitude)
                            H_grp = H_all.view(n_groups, g)
                            d_abs_grp = delta_abs_all.view(n_groups, g)
                            score_grp = score_elem.view(n_groups, g)
                            cv_h = (H_grp.std(dim=1) /
                                    H_grp.mean(dim=1).clamp_min(1e-30))
                            cv_d = (d_abs_grp.std(dim=1) /
                                    d_abs_grp.mean(dim=1).clamp_min(1e-30))
                            d_sq_grp = d_abs_grp ** 2
                            cv_d_sq = (d_sq_grp.std(dim=1) /
                                       d_sq_grp.mean(dim=1).clamp_min(1e-30))
                            cv_score = (score_grp.std(dim=1) /
                                        score_grp.mean(dim=1).clamp_min(1e-30))
                            # Sign-cancellation ratio:
                            # |mean(δ_signed)| / mean(|δ|)
                            #   = 1 (all same sign) ~ 0 (perfect cancellation)
                            cancellation_ratio = float(
                                (d_g_signed / d_g_abs.clamp_min(1e-30)).median().item()
                            )
                            cv_h_med = float(cv_h.median().item())
                            cv_d_med = float(cv_d.median().item())
                            cv_d_sq_med = float(cv_d_sq.median().item())
                            cv_score_med = float(cv_score.median().item())

                            if _should_write():
                                print(
                                    f"[proposal] group_probe[{delta_label}] g={g:>2d} "
                                    f"@iter{iteration}: "
                                    f"cap_prod={cap_prod:.3f} cap_abs={cap_abs:.3f} "
                                    f"cap_sgn={cap_signed:.3f} "
                                    f"(elem={capture_rate_elem:.3f}) "
                                    f"CV_H={cv_h_med:.2f} CV_|δ|={cv_d_med:.2f} "
                                    f"CV_δ²={cv_d_sq_med:.2f} CV_score={cv_score_med:.2f} "
                                    f"cancel={cancellation_ratio:.3f}",
                                    flush=True,
                                )
                except Exception as _gpexc:
                    if _should_write():
                        import traceback as _tb
                        print(f"[proposal] group_probe FAILED @iter{iteration}: "
                              f"{_gpexc}\n{_tb.format_exc()}", flush=True)

            # ---- Step 4c: EMA-vs-TRUTH probe ----
            # Test whether EMA over multiple raw refreshes converges to truth.
            # Hypothesis: raw H ~= truth + independent_noise per refresh.
            # If true: EMA averages noise out → EMA-vs-TRUTH J@1% > raw-vs-TRUTH J@1%.
            # If false: noise is correlated or H drifts faster than EMA averages →
            # EMA stays similar to raw.
            # Uses the SAME α as production (CLAUDE.md default 0.9).
            _xc_ema_alpha = float(_os_xc.environ.get("PROPOSAL_XC_EMA_ALPHA", "0.9"))
            _ema_store = _config.get("_xc_ema_h", {})
            for name, h in candidates.items():
                if h is None:
                    continue
                prev_ema = _ema_store.get(name)
                if prev_ema is None:
                    # Initialize EMA with first sample.
                    _ema_store[name] = [t.detach().clone() for t in h]
                    continue
                # Update EMA: ema = α·prev + (1-α)·new
                new_ema = []
                for prev_t, new_t in zip(prev_ema, h):
                    e = (prev_t * _xc_ema_alpha
                         + new_t * (1.0 - _xc_ema_alpha)).detach()
                    new_ema.append(e)
                _ema_store[name] = new_ema
                # Compare EMA against truth (CURRENT truth, since production
                # would use EMA H to compare against current data).
                agg_e = _compare_hessian_estimators(
                    iteration, tracked, h_truth, new_ema, {},
                    label_a="truth", label_b=f"{name}_ema",
                )
                agg_e["truth_k"] = int(truth_k)
                agg_e["ema_alpha"] = _xc_ema_alpha
                _emit(agg_e, "vs-TRUTH-EMA", name)
            _config["_xc_ema_h"] = _ema_store

            # ---- Step 5: optional batch-noise test ----
            # PROPOSAL_BATCH_NOISE_TEST=1: capture a SECOND batch (advance
            # iterator) and recompute candidates against the same θ. The
            # vs-BATCH J@X is "pure batch noise" — no model drift since we
            # haven't taken any optimizer step. Comparing to vs-PREV J@X
            # tells us how much of the temporal turnover is batch noise vs
            # real model drift over the refresh interval.
            batch_noise_on = _os_xc.environ.get(
                "PROPOSAL_BATCH_NOISE_TEST", "") in ("1", "true", "TRUE")
            if batch_noise_on:
                try:
                    from pretrain_gpt import get_batch as _get_batch_b
                    di_b = data_iterator[0] if isinstance(
                        data_iterator, (list, tuple)) else data_iterator
                    batch_b = _get_batch_b(di_b)
                    tokens_b, labels_b, lm_b, am_b, pid_b = batch_b[:5]
                    import torch.nn.functional as _Fb
                    def _step_b(_di, m):
                        out = m(tokens_b, pid_b, am_b)
                        logits = out[0] if isinstance(out, tuple) else out
                        flat = logits.reshape(-1, logits.size(-1))
                        lbl = labels_b.reshape(-1)
                        if lm_b is not None and lm_b.numel() == flat.size(0):
                            per = _Fb.cross_entropy(flat.float(), lbl, reduction="none")
                            denom = lm_b.view(-1).float().sum().clamp_min(1.0)
                            loss = (per * lm_b.view(-1).float()).sum() / denom
                        else:
                            loss = _Fb.cross_entropy(flat.float(), lbl, reduction="mean")
                        return loss, {}
                    class _StubB:
                        def __next__(self): return None
                    di_b_stub = _StubB()
                    _, candidates_b = _build_candidates(
                        batch_b, _step_b, di_b_stub, include_truth=False,
                    )
                    for name, h_a in candidates.items():
                        h_b = candidates_b.get(name)
                        if h_a is None or h_b is None:
                            continue
                        agg_b = _compare_hessian_estimators(
                            iteration, tracked, h_a, h_b, {},
                            label_a="batchA", label_b=name,
                        )
                        _emit(agg_b, "vs-BATCH", name)
                    del candidates_b
                except Exception as _bexc:
                    if _should_write():
                        print(f"[proposal] batch-noise test FAILED @iter{iteration}: "
                              f"{_bexc}", flush=True)

            del h_truth, candidates
        except Exception as _exc:
            if _should_write():
                import traceback as _tb
                print(f"[proposal] cross-check 4way+temporal FAILED @iter{iteration}: "
                      f"{_exc}\n{_tb.format_exc()}", flush=True)

    if _should_write():
        print(
            f"[proposal] refresh_finite_diff @iter{iteration} K={fd_k_used} "
            f"{elapsed_ms:.0f} ms  ({n_elem:,} expert elements)",
            flush=True,
        )

    return _finalize({"iter": iteration, "k_samples": k_samples,
                      "ms": elapsed_ms, "numel": n_elem,
                      "diag": diag_record})


def proposal_hutchinson_probe(
    model, optimizer, forward_step_func, data_iterator, iteration,
    k_samples: int = 4, out_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Option C smoke: compare v_t (from Adam state) vs H_diag (Hutchinson K-sample)
    as importance-score components, on the live post-training model state.

    Uses the same expert-param detection as the live tracker, and pulls v_t via
    the Fp32StateMapper so it matches precisely what live training uses for score.
    """
    if not _enabled or _tracker is None:
        return None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return None

    import json
    import math
    import time

    inner_model = _config["_inner_model"]
    inner_opt = _config["_inner_opt"] if optimizer is None else optimizer
    from ._ds_state import get_mapper
    mapper = get_mapper()

    # Collect expert params (the tracked set the live tracker uses).
    tracked = []
    for name, p in inner_model.named_parameters():
        if not p.requires_grad:
            continue
        if getattr(p, "allreduce", True) is False:
            tracked.append((name, p))
    if not tracked:
        print("[proposal] hutchinson probe: no expert params detected", flush=True)
        return None

    # Read v_t (FP32 bias-corrected) via the mapper (handles every DS wrapper path).
    beta2 = 0.999
    for g in inner_opt.param_groups:
        if "betas" in g:
            beta2 = float(g["betas"][1]); break
    step_now = int(iteration) if iteration else 1
    bias_corr2 = max(1.0 - beta2 ** step_now, 1e-12)
    v_list, h_list, names = [], [], []
    v_map: Dict[str, torch.Tensor] = {}
    for name, p in tracked:
        v = mapper.get_state(p, "exp_avg_sq") if mapper is not None else None
        if v is None:
            st = inner_opt.state.get(p, {})
            v = st.get("exp_avg_sq")
        if v is None:
            v_f = torch.zeros_like(p, dtype=torch.float32)
        else:
            v_f = (v / bias_corr2).to(torch.float32).detach()
        v_map[name] = v_f

    # Forward + loss via Megatron's forward_step_func. It returns (loss, loss_reduced).
    # We bypass DeepSpeed backward: compute grads directly via torch.autograd.grad
    # with create_graph=True so we can take a second-order grad for Hutchinson.
    # Put in train mode for dropout-stable numerics, then switch back to eval at end.
    if hasattr(model[0], "module"):
        net = model[0].module
    else:
        net = model[0]
    was_training = net.training
    net.train()  # enable grad flow; DS engine usually keeps this on anyway

    # Run one microbatch via forward_step_func but intercept its loss.
    # Megatron's forward_step_func signature: (data_iter, model) → (loss_tensor, loss_reduced_dict)
    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    m0 = model[0] if isinstance(model, (list, tuple)) else model
    try:
        loss, _ = forward_step_func(di, m0)
    except Exception as exc:
        print(f"[proposal] hutchinson probe: forward_step failed: {exc}", flush=True)
        net.train(was_training)
        return None

    if not torch.is_tensor(loss):
        net.train(was_training)
        return None
    # Some Megatron configs return a scalar already; ensure so.
    if loss.dim() > 0:
        loss = loss.mean()

    expert_ps = [p for (_, p) in tracked]
    t_h0 = time.perf_counter()
    ds_engine = m0
    prev_flag = getattr(ds_engine, "_running_engine_backward", False)
    try:
        ds_engine._running_engine_backward = True
    except Exception:
        pass
    try:
        grads = torch.autograd.grad(loss, expert_ps, create_graph=True, retain_graph=True, allow_unused=True)
    except Exception as exc:
        print(f"[proposal] hutchinson probe: first-order grad failed: {exc}", flush=True)
        try:
            ds_engine._running_engine_backward = prev_flag
        except Exception:
            pass
        net.train(was_training)
        return None
    # Replace None (allow_unused) with zeros for uniformity.
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, expert_ps)]
    gen = torch.Generator(device=expert_ps[0].device).manual_seed(
        int(1_000_003 * step_now + 7777) & 0x7fffffff
    )
    h_accum = [torch.zeros_like(p, dtype=torch.float32) for p in expert_ps]
    for k in range(k_samples):
        zs = [
            (torch.randint(0, 2, p.shape, generator=gen, device=p.device, dtype=torch.float32) * 2 - 1)
            for p in expert_ps
        ]
        dot = sum((g * z).sum() for g, z in zip(grads, zs))
        try:
            hz_list = torch.autograd.grad(dot, expert_ps, retain_graph=(k < k_samples - 1), allow_unused=True)
        except Exception as exc:
            print(f"[proposal] hutchinson probe: second-order grad failed at k={k}: {exc}", flush=True)
            try:
                ds_engine._running_engine_backward = prev_flag
            except Exception:
                pass
            net.train(was_training)
            return None
        hz_list = [hz if hz is not None else torch.zeros_like(p) for hz, p in zip(hz_list, expert_ps)]
        for i, (z, hz) in enumerate(zip(zs, hz_list)):
            h_accum[i] = h_accum[i] + (z * hz).to(torch.float32).detach() / k_samples
    t_h1 = time.perf_counter()
    try:
        ds_engine._running_engine_backward = prev_flag
    except Exception:
        pass
    net.train(was_training)

    # Compute per-param + aggregate stats.
    def _pearson(a, b):
        a = a.flatten().to(torch.float64); b = b.flatten().to(torch.float64)
        a = a - a.mean(); b = b - b.mean()
        n = (a * b).sum().item(); d = (a.norm().item() * b.norm().item())
        return n / d if d > 0 else 0.0

    def _topk_jaccard(a, b, k_ratio):
        n = a.numel(); k = max(1, int(k_ratio * n))
        a_top = set(torch.topk(a.flatten(), k, largest=True).indices.tolist())
        b_top = set(torch.topk(b.flatten(), k, largest=True).indices.tolist())
        inter = len(a_top & b_top)
        return inter / (2 * k - inter) if (2 * k - inter) > 0 else 0.0

    # Compute δ = |θ_current − θ_CP| per expert tensor. tracker.theta_cp holds
    # the FC-reference weights (BF16, GPU). Used to form the δ²-weighted score
    # that matches the actual selection formula: score = (v or |H|) × δ².
    per_param_rows = []
    cat_v, cat_h = [], []                 # bare v_t, |H|
    cat_sv, cat_sh = [], []               # score_v = v·δ², score_h = |H|·δ²
    for (name, p), h in zip(tracked, h_accum):
        v = v_map[name]
        h_abs = h.abs()
        # δ² from tracker's theta_cp cache (same buffer the live selection uses).
        fc = {}.get(name)
        if fc is not None:
            delta_sq = (p.data.to(torch.float32) - fc.to(torch.float32)).pow_(2).flatten()
        else:
            delta_sq = torch.zeros_like(v.flatten())
        sv = v.flatten() * delta_sq                  # score_v = v × δ²
        sh = h_abs.flatten() * delta_sq              # score_h = |H| × δ²
        row = {
            "name": name, "numel": int(p.numel()),
            "v_mean": float(v.mean()), "v_max": float(v.max()),
            "h_mean": float(h.mean()), "h_abs_max": float(h_abs.max()),
            "h_neg_frac": float((h < 0).float().mean()),
            "delta_sq_mean": float(delta_sq.mean()), "delta_sq_max": float(delta_sq.max()),
            # bare
            "pearson_v_habs": _pearson(v, h_abs),
            "jaccard_top0.1pct_bare": _topk_jaccard(v, h_abs, 0.001),
            "jaccard_top1pct_bare": _topk_jaccard(v, h_abs, 0.01),
            "jaccard_top10pct_bare": _topk_jaccard(v, h_abs, 0.1),
            # δ²-weighted (matches actual selection)
            "pearson_score": _pearson(sv, sh),
            "jaccard_top0.1pct_score": _topk_jaccard(sv, sh, 0.001),
            "jaccard_top1pct_score": _topk_jaccard(sv, sh, 0.01),
            "jaccard_top10pct_score": _topk_jaccard(sv, sh, 0.1),
        }
        per_param_rows.append(row)
        cat_v.append(v.flatten().cpu()); cat_h.append(h_abs.flatten().cpu())
        cat_sv.append(sv.cpu()); cat_sh.append(sh.cpu())
    cv = torch.cat(cat_v); ch = torch.cat(cat_h)
    csv_ = torch.cat(cat_sv); csh = torch.cat(cat_sh)
    agg = {
        "iter": step_now, "k_samples": k_samples,
        "hutchinson_seconds": t_h1 - t_h0,
        "total_expert_numel": int(cv.numel()),
        # bare (v_t vs |H|)
        "pearson_v_habs_all": _pearson(cv, ch),
        "jaccard_top0.1pct_all": _topk_jaccard(cv, ch, 0.001),
        "jaccard_top1pct_all": _topk_jaccard(cv, ch, 0.01),
        "jaccard_top10pct_all": _topk_jaccard(cv, ch, 0.1),
        # δ²-weighted (score_v = v·δ² vs score_h = |H|·δ²) — matches actual Top-K
        "pearson_score_all": _pearson(csv_, csh),
        "jaccard_top0.1pct_score_all": _topk_jaccard(csv_, csh, 0.001),
        "jaccard_top1pct_score_all": _topk_jaccard(csv_, csh, 0.01),
        "jaccard_top10pct_score_all": _topk_jaccard(csv_, csh, 0.1),
        "neg_h_frac_mean": float(sum(r["h_neg_frac"] for r in per_param_rows) / max(len(per_param_rows), 1)),
    }
    print(
        f"[proposal] hutchinson probe @iter{step_now} K={k_samples} "
        f"({agg['hutchinson_seconds']*1000:.0f} ms) neg_H={agg['neg_h_frac_mean']*100:.1f}%\n"
        f"  bare  (v_t vs |H|):     Pearson={agg['pearson_v_habs_all']:+.4f}  "
        f"J@0.1%={agg['jaccard_top0.1pct_all']:.4f}  "
        f"J@1%={agg['jaccard_top1pct_all']:.4f}  "
        f"J@10%={agg['jaccard_top10pct_all']:.4f}\n"
        f"  score (v·δ² vs |H|·δ²): Pearson={agg['pearson_score_all']:+.4f}  "
        f"J@0.1%={agg['jaccard_top0.1pct_score_all']:.4f}  "
        f"J@1%={agg['jaccard_top1pct_score_all']:.4f}  "
        f"J@10%={agg['jaccard_top10pct_score_all']:.4f}",
        flush=True,
    )
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as f:
            f.write(json.dumps({"aggregate": agg, "per_param": per_param_rows}) + "\n")
    return agg


def proposal_shutdown() -> Dict[str, Any]:
    """Drain writers, dump write-logs + cpu-mem samples to disk, cleanup."""
    global _tracker, _ckpt, _fc, _writer, _enabled
    if not _enabled:
        return {}
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
    if _ckpt is not None:
        _ckpt.finalize()
        summary["dc_total_bytes"] = _ckpt.total_bytes_written
        summary["dc_num_bundles"] = _ckpt.num_bundles
        if dump_dir is not None:
            try:
                (dump_dir / f"{dump_prefix}ckpt_write_log.json").write_text(
                    json.dumps(_ckpt.write_log_dict(), indent=2)
                )
            except Exception:
                pass
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


