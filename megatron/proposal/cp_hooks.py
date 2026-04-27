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
from .recovery import load_h_ema, recover


_tracker: Optional[ImportanceTracker] = None
_ckpt: Optional[CheckpointManager] = None
_fc: Optional[FCManager] = None
_writer: Optional[AsyncWriter] = None
_ne_payload: Optional[Dict[str, torch.Tensor]] = None
_enabled: bool = False
_cp_dir: Optional[Path] = None

# Module-level staging buffers for _persist_h_ema. H_ema lives on
# GPU (per-expert |H|, ~900 MB on 300M-MoE). Naively ``.to("cpu")`` per tensor
# repeats the pageable-D2H slow path we just removed from FC/DC: pinned batch
# collapses N syncs to one pinned transfer. Allocated lazily on first refresh;
# pinned buffer is reused across refreshes (drain_torch() before overwrite
# ensures the previous torch.save has finished reading it).
_h_ema_stage_gpu: Optional[torch.Tensor] = None
_h_ema_pinned_cpu: Optional[torch.Tensor] = None
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
    )

    # Non-expert DC retention: W+2 is sufficient (each non-expert FC
    # supersedes older dense files). DC is non-expert only now.
    ne_retention = max(_config["nonexpert_fc_W"] + 2, 3)

    # CP layout: split non-expert, expert, and Hessian-EMA into separate
    # subdirectories so they can be mounted on different devices.
    #   <cp>/non_expert/  — non-expert dense DC + non-expert full FC
    #   <cp>/expert/      — expert rolling (lossy-FC) + legacy expert full FC
    #   <cp>/hessian/     — |H_diag| EMA state (persisted so recovery
    #                        continues the EMA rather than restarting it)
    _nonexpert_cp_dir = _cp_dir / "non_expert"
    _expert_cp_dir = _cp_dir / "expert"
    _hessian_cp_dir = _cp_dir / "hessian"
    # Only create the hessian dir when persistence is actually requested.
    # Avoids empty subdirs under experiment runs that stay in-memory.
    if _should_write() and bool(getattr(args, "proposal_hessian_persist", False)):
        _hessian_cp_dir.mkdir(parents=True, exist_ok=True)
    _config["_hessian_cp_dir"] = _hessian_cp_dir
    _config["_hessian_mode"] = str(getattr(args, "proposal_hessian_mode", "ema"))
    _config["_hessian_ema_alpha"] = float(getattr(args, "proposal_hessian_ema_alpha", 0.9))
    _config["_hessian_persist"] = bool(getattr(args, "proposal_hessian_persist", False))
    _config["_hessian_ema_history"] = []  # [(path, rec)] for retention

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
    _is_replay_only = bool(getattr(args, "proposal_replay_only", False))
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
            on_expert_fc_fn=_tracker.on_expert_fc,
            on_nonexpert_fc_fn=_tracker.on_nonexpert_fc,
            schedule=fc_schedule,
            lossless_fc_fraction_pct=_config["lossless_fc_fraction_pct"],
            # Replay-only: preserve on-disk expert state so recover() can
            # read it back. Normal training starts with fresh=True so the
            # previous run's stale append-log doesn't pollute the new state.
            fresh_expert_state=not _is_replay_only,
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
    else:
        _fc = None

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

    # Score distribution diagnostic: measure expert vs non-expert score
    # distribution at specific iterations and dump per-layer stats.
    _sd_iters = getattr(args, "proposal_score_diag_iters", None) or []
    _config["_score_diag_iters"] = set(int(x) for x in _sd_iters)
    _config["_score_diag_log"] = getattr(args, "proposal_score_diag_log", None)
    _config["_score_diag_every"] = int(getattr(args, "proposal_score_diag_every", 0) or 0)
    _config["_score_diag_done"] = set()
    _config["_hessian_comparison_log"] = getattr(args, "proposal_hessian_comparison_log", None)
    _config["_hessian_estimator"] = getattr(args, "proposal_hessian_estimator", "hutchinson")
    _config["_hessian_cross_check"] = bool(getattr(args, "proposal_hessian_cross_check", False))
    _config["_hessian_cross_check_log"] = getattr(args, "proposal_hessian_cross_check_log", None)
    # ANALYSIS: non-expert |H| probe — comparison of expert vs non-expert |H|
    # distribution at score_diag_iters; retained for measurement purposes.
    _config["_hessian_nonexpert_probe_log"] = getattr(args, "proposal_hessian_nonexpert_probe_log", None)


def proposal_pre_step(model) -> None:
    """Snapshot non-expert grads before optimizer.step consumes them.

    Expert payload is built post-step from Adam state (p, m, v) via tracker.step,
    so no expert grad snapshot is needed here.
    """
    global _ne_payload
    if not _enabled or _tracker is None:
        return

    # MoC-System baseline (sequential): no DC, no non-expert dense snapshot.
    seq_mode = _config["cp_mode"] == "sequential"
    if not seq_mode:
        import time as _time
        _t0 = _time.perf_counter()
        _ne_payload, _ = _tracker.snapshot_nonexpert_dc(global_step=0)
        _config["_phase_ms_pre_step"] = (_time.perf_counter() - _t0) * 1000.0
    else:
        _ne_payload = None


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


def proposal_post_step(
    model, step: int,
) -> Optional[Tuple[List[Tuple[int, int]], bool]]:
    """Run tracker.step + FC tick + DC record after optimizer.step."""
    global _ne_payload
    if not _enabled or _tracker is None:
        return None

    seq_mode = _config["cp_mode"] == "sequential"
    inner_opt = _config["_inner_opt"]

    fired_experts: List[Tuple[int, int]] = []
    nonexpert_fired = False

    import time as _time

    if seq_mode:
        if _fc is not None:
            fired_experts, nonexpert_fired = _fc.tick(step=step)
    else:
        # Order matters:
        #   1) FCManager.tick fires expert lossless FC (round-robin F%) +
        #      non-expert FC, and resets θ_CP for fired experts via the
        #      tracker.on_expert_fc callback.
        #   2) tracker.step computes top-K scores; fired experts are
        #      passed as the exclusion list so their tensors get score=0
        #      and are skipped from the lossy_fc_append.
        #   3) CheckpointManager records non-expert dense DC for iters
        #      that did NOT fire a non-expert FC.
        _t_tick0 = _time.perf_counter()
        if _fc is not None:
            fired_experts, nonexpert_fired = _fc.tick(step=step)
        _phase_tick = (_time.perf_counter() - _t_tick0) * 1000.0

        _t_trk0 = _time.perf_counter()
        _tracker.step(
            inner_opt, global_step=step,
            lossless_fc_eids=fired_experts,
        )
        _phase_tracker = (_time.perf_counter() - _t_trk0) * 1000.0

        _t_ckpt0 = _time.perf_counter()
        if _ckpt is not None:
            _ckpt.record(
                step=step,
                nonexpert_payload=_ne_payload or {},
                nonexpert_did_fc=nonexpert_fired,
            )
        _phase_ckpt = (_time.perf_counter() - _t_ckpt0) * 1000.0
        _ne_payload = None

        _config["_phase_ms_tick"] = _phase_tick
        _config["_phase_ms_tracker_step"] = _phase_tracker
        _config["_phase_ms_ckpt_record"] = _phase_ckpt

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
        phase_pre = _config.get("_phase_ms_pre_step", 0.0)
        phase_tick = _config.get("_phase_ms_tick", 0.0)
        phase_trk = _config.get("_phase_ms_tracker_step", 0.0)
        phase_ckpt = _config.get("_phase_ms_ckpt_record", 0.0)
        trk_phases = getattr(_tracker, "last_phase_ms", None) or {}
        tp1 = trk_phases.get("pass1", 0.0)
        tp2 = trk_phases.get("pass2", 0.0)
        tp3 = trk_phases.get("pass3", 0.0)
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
            f"phase_ms[pre_step={phase_pre:.1f}"
            f"(enum={dc_enum:.1f}+copy={dc_copy:.1f}+wait={dc_wait:.1f}+"
            f"xfer={dc_xfer:.1f}) tick={phase_tick:.1f}"
            f"(nfired={n_fired} collect={t_col:.1f}+cb={t_cb:.1f}+"
            f"dispatch={t_dp:.1f}+ne={t_ne:.1f}"
            f"[col={ne_col:.1f}:enum={ne_enum:.1f}+copy={ne_copy:.1f}+"
            f"wait={ne_wait:.1f}+xfer={ne_xfer:.1f} N={ne_n}]) "
            f"tracker_step={phase_trk:.1f}(p1={tp1:.1f}+p2={tp2:.1f}+"
            f"p3={tp3:.1f}) ckpt_record={phase_ckpt:.1f}]",
            flush=True,
        )

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

    # Score distribution diagnostic (expert vs non-expert scores + per-layer).
    sd_iters = _config.get("_score_diag_iters", set())
    sd_done = _config.get("_score_diag_done", set())
    sd_log = _config.get("_score_diag_log")
    sd_every = _config.get("_score_diag_every", 0) or 0
    # `--proposal-score-diag-every N`: trigger measurement every N iters
    # (overrides `_score_diag_iters` list-based trigger). Used for dense
    # diagnostic sampling during smoke/analysis.
    should_fire_sd = (
        (step in sd_iters and step not in sd_done)
        or (sd_every > 0 and step > 0 and step % sd_every == 0)
    )
    if should_fire_sd and _tracker is not None and sd_log:
        try:
            rec = _tracker.measure_score_distribution(inner_opt, step)
            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                import json as _json
                with open(sd_log, "a") as f:
                    f.write(_json.dumps(rec) + "\n")
                exp_max = rec["expert"]["max"]
                exp_p999 = rec["expert"]["p99.9"]
                ne_max = rec["nonexpert"]["max"]
                ne_p999 = rec["nonexpert"]["p99.9"]
                overlap_pct = rec["overlap"]["expert_elements_above_ne_p999_pct"]
                exp_diag = rec["expert"].get("diag", {})
                d_max = exp_diag.get("delta_max", 0.0)
                v_max = exp_diag.get("vhat_max", 0.0)
                dL = exp_diag.get("delta_L_proxy", 0.0)
                p_abs_mean = exp_diag.get("param_abs_mean", 0.0)
                print(
                    f"[proposal] ScoreDiag @iter {step}: "
                    f"expert max={exp_max:.3e} p99.9={exp_p999:.3e} | "
                    f"non-expert max={ne_max:.3e} p99.9={ne_p999:.3e} | "
                    f"overlap: {overlap_pct:.3f}%\n"
                    f"          [expert diag] δ_max={d_max:.3e} v̂_max={v_max:.3e} "
                    f"|θ|_mean={p_abs_mean:.3e} ΔL≈{dL:.3e}",
                    flush=True,
                )
        except Exception as e:
            print(f"[proposal] ScoreDiag @iter {step} FAILED: {e}", flush=True)
        sd_done.add(step)

    return fired_experts, nonexpert_fired


def _restore_h_ema_from_cp(t_fault: int) -> None:
    """Populate ``_tracker._H_ema`` from the most-recent ``h_ema_iter*.bin``
    at or before ``t_fault``. Called after ``recover()`` so the tracker
    resumes scoring with the same |H| state rather than restarting.

    No-op when ``--proposal-hessian-persist`` is off (default) — the
    tracker's in-memory H state is used as-is, and the next scheduled
    refresh rebuilds it (for EMA mode: warm-start from H_new since
    H_ema was empty; for plain mode: always fresh H_new).
    """
    if _tracker is None:
        return
    if not bool(_config.get("_hessian_persist", False)):
        return
    hessian_dir = _config.get("_hessian_cp_dir")
    if hessian_dir is None or not Path(hessian_dir).exists():
        return
    device = next(_config["_inner_model"].parameters()).device
    obj = load_h_ema(Path(hessian_dir), int(t_fault), device=device)
    if obj is None:
        return
    _tracker._H_ema = dict(obj.get("h_ema", {}))
    _tracker._H_ema_updates = int(obj.get("updates", 0))


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
    _restore_h_ema_from_cp(t_fault)
    _t_rec2 = _time.perf_counter()
    recover_s = _t_rec2 - _t_rec0
    print(
        f"[proposal] proposal_recover: recover_s={recover_s:.3f}  "
        f"(recover={_t_rec1 - _t_rec0:.3f}s h_ema_load={_t_rec2 - _t_rec1:.3f}s)  "
        f"ne_fc={diag.nonexpert_fc_iter} expert_fc_count={diag.expert_fc_count} "
        f"dc_replayed={diag.dc_bundles_replayed}/{list(diag.replay_range)}",
        flush=True,
    )
    return {
        "recover_time_s": float(recover_s),
        "recover_time_main_s": float(_t_rec1 - _t_rec0),
        "recover_time_h_ema_s": float(_t_rec2 - _t_rec1),
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
    _restore_h_ema_from_cp(int(step))
    recover_s = time.perf_counter() - t0
    # θ_CP cache: after recovery, param values have been overwritten to
    # reconstruct state. We refresh θ_CP for every tracked tensor so δ = 0
    # at the post-recovery iter (consistent with a "fresh FC" view).
    if _tracker is not None:
        for name in _tracker.tracked_names():
            _tracker.on_expert_fc(name)

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


def _fold_h(names: List[str], h_new_list: List[torch.Tensor]) -> None:
    """Update ``_tracker._H_ema`` with a fresh |H| estimate.

    Two modes, controlled by ``--proposal-hessian-mode``:
      * ``ema`` (default): H_ema ← α·H_ema + (1-α)·H_new per refresh,
        α from ``--proposal-hessian-ema-alpha``. Smooths K-sample variance
        across the EMA window. First refresh initialises H_ema = H_new.
      * ``plain``: H_ema ← H_new each refresh — no temporal smoothing.

    Bumps ``_H_ema_updates`` so recovery can check continuity. This
    function only touches GPU state; disk persistence is orthogonal and
    handled by :func:`_persist_h_ema` when ``--proposal-hessian-persist``
    is set.
    """
    if _tracker is None:
        return
    mode = str(_config.get("_hessian_mode", "ema"))
    if mode == "plain":
        for name, h_new in zip(names, h_new_list):
            _tracker._H_ema[name] = h_new.detach().clone()
    else:  # ema
        alpha = float(_config.get("_hessian_ema_alpha", 0.9))
        for name, h_new in zip(names, h_new_list):
            cur = _tracker._H_ema.get(name)
            if cur is None:
                _tracker._H_ema[name] = h_new.detach().clone()
            else:
                # In-place EMA to avoid a transient alloc of shape(h_new).
                cur.mul_(alpha).add_(h_new.detach(), alpha=(1.0 - alpha))
    _tracker._H_ema_updates += 1


def _persist_h_ema(iteration: int) -> None:
    """Async-save the current ``_tracker._H_ema`` to
    ``<cp>/hessian/h_ema_iter{N}.bin``. No-op when:
      - ``--proposal-hessian-persist`` is off (default), OR
      - this rank is not the writing rank, OR
      - no async writer / hessian_cp_dir is configured.

    Payload is ~900 MB across hundreds of per-expert tensors; batched via
    GPU staging + pinned CPU for a single ~20 GB/s D2H transfer (vs
    ~1.2 GB/s pageable per-tensor loop). drain_torch() blocks only if a
    prior refresh's write is still reading the pinned buffer — refreshes
    are 40 iters apart so this is typically a no-op.
    """
    if _tracker is None:
        return
    if not bool(_config.get("_hessian_persist", False)):
        return
    hessian_dir = _config.get("_hessian_cp_dir")
    if not _should_write() or _writer is None or hessian_dir is None:
        return

    names_ord = list(_tracker._H_ema.keys())
    srcs = [_tracker._H_ema[n] for n in names_ord]
    plan: List[Tuple[str, int, int, torch.Size]] = []
    off = 0
    for name, src in zip(names_ord, srcs):
        n = int(src.numel())
        plan.append((name, off, n, src.shape))
        off += n
    total_numel = off
    device = srcs[0].device if srcs else None
    if device is None:
        return

    global _h_ema_stage_gpu, _h_ema_pinned_cpu
    if (_h_ema_stage_gpu is None
            or _h_ema_stage_gpu.numel() < total_numel
            or _h_ema_stage_gpu.device != device):
        _h_ema_stage_gpu = torch.empty(total_numel, dtype=torch.float32, device=device)
    stage = _h_ema_stage_gpu
    for (_n, off_i, n_i, _s), src in zip(plan, srcs):
        stage.narrow(0, off_i, n_i).copy_(src.reshape(-1))

    if _h_ema_pinned_cpu is not None:
        _writer.drain_torch()
    if _h_ema_pinned_cpu is None or _h_ema_pinned_cpu.numel() < total_numel:
        _h_ema_pinned_cpu = torch.empty(total_numel, dtype=torch.float32, pin_memory=True)
    flat_cpu = _h_ema_pinned_cpu.narrow(0, 0, total_numel)
    flat_cpu.copy_(stage.narrow(0, 0, total_numel), non_blocking=False)

    h_ema_cpu = {name: flat_cpu.narrow(0, off_i, n_i).view(shape)
                 for name, off_i, n_i, shape in plan}
    # Raw-binary payload (PRPTENS). ``save_binary`` takes a flat
    # {name: tensor} dict + separate scalar metadata; on load, recovery
    # reconstructs the same structure as the legacy pickle dict.
    from .binary_io import save_binary as _save_binary
    scalars = {
        "iter": int(iteration),
        "updates": int(_tracker._H_ema_updates),
        "mode": str(_config.get("_hessian_mode", "ema")),
        "alpha": float(_config.get("_hessian_ema_alpha", 0.9)),
    }
    path = Path(hessian_dir) / f"h_ema_iter{int(iteration):08d}.bin"

    # Retention: keep last 2 (one being overwritten + one for rollback safety).
    # Older files unlinked from the writer thread once the new one is durable.
    history: List[Tuple[Path, Any]] = _config.setdefault("_hessian_ema_history", [])
    to_unlink: List[Path] = []
    history.append((path, None))
    while len(history) > 2:
        old_path, _ = history.pop(0)
        to_unlink.append(old_path)

    def _unlink(paths: List[Path] = to_unlink) -> None:
        for p in paths:
            try: p.unlink()
            except FileNotFoundError: pass

    def _save_h_ema(tensor_dict, p):
        _save_binary(p, tensor_dict, scalars=scalars)

    rec = _writer.submit(
        h_ema_cpu, path, tag="h_ema",
        on_complete=[_unlink] if to_unlink else None,
        save_fn=_save_h_ema,
    )
    history[-1] = (path, rec)


def proposal_refresh_hessian(
    model, forward_step_func, data_iterator, iteration,
    estimator: str = "hutchinson",
    k_samples: int = 1,
) -> Optional[Dict[str, Any]]:
    """Fold a fresh |H_diag| estimate into tracker._H_ema. Dispatches to
    the chosen estimator:
        estimator='gnb'        → _refresh_gnb_impl (Sophia-G, 1 fwd + 1 bwd)
        estimator='hutchinson' → proposal_refresh_hutchinson (K× 2nd-order bwd)
    EMA result is also async-persisted to <cp>/hessian/.
    When selection_mode='hessian', the EMA H replaces v̂_t in Pass 1 of
    tracker.step.
    """
    if estimator == "hutchinson":
        return proposal_refresh_hutchinson(
            model, forward_step_func, data_iterator, iteration,
            k_samples=k_samples,
        )
    return _refresh_gnb_impl(
        model, forward_step_func, data_iterator, iteration, k_samples=k_samples,
    )


def _refresh_gnb_impl(
    model, forward_step_func, data_iterator, iteration,
    k_samples: int = 1,
) -> Optional[Dict[str, Any]]:
    """GNB (Sophia-G) estimator for diag(H). See proposal_refresh_hessian.

    Algorithm (Liu et al. 2023, "Sophia"):
      1. Forward pass, obtain per-token logits ℓ.
      2. Sample ŷ_t ~ Categorical(softmax(ℓ_t)) for each token t.
      3. Compute L̂ = CE(ℓ, ŷ).
      4. Single backward → g = ∇_θ L̂.
      5. |H_diag| ≈ B · g²  (Bartlett identity on diagonal GN).

    Advantages vs Hutchinson:
      - 1 forward + 1 backward (no create_graph, no 2nd-order autograd)
      - No Rademacher allocation, no Hz vectors
      - Peak refresh memory drops from ~7-12 GB to ~1-2 GB (189M expert params)

    The `k_samples` argument now controls how many label draws are averaged
    (variance reduction). Default 1 matches Sophia-G spec; ≥2 runs one extra
    forward+backward per sample.
    """
    if not _enabled or _tracker is None:
        return None

    # Prefer dedicated iterator if set, so the shared valid iterator isn't
    # consumed by refresh (avoids val-sample shift across variants).
    dedicated = _config.get("_hessian_dedicated_valid_iter")
    if dedicated is not None:
        data_iterator = dedicated

    import time as _time
    import torch.nn.functional as F

    # Lazy import get_batch to avoid circular import at module load.
    try:
        from pretrain_gpt import get_batch as _get_batch_gpt
    except Exception as _imp_exc:
        print(f"[proposal] refresh_hessian(gnb): cannot import get_batch from pretrain_gpt: {_imp_exc}", flush=True)
        return None

    inner_model = _config["_inner_model"]

    tracked = []
    for name, p in inner_model.named_parameters():
        if not p.requires_grad:
            continue
        if getattr(p, "allreduce", True) is False:
            tracked.append((name, p))
    if not tracked:
        return None

    # Snapshot RNG state (CPU + CUDA) so dropout masks, MoE routing, and
    # label-sampling multinomial draws consumed by the refresh forward+backward
    # don't shift the training RNG stream.
    _cpu_rng_state = torch.get_rng_state()
    _cuda_rng_state = (
        torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    )

    if hasattr(model[0], "module"):
        net = model[0].module
    else:
        net = model[0]
    was_training = net.training
    net.train()

    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    m0 = model[0] if isinstance(model, (list, tuple)) else model

    def _restore_rng() -> None:
        try:
            torch.set_rng_state(_cpu_rng_state)
            if _cuda_rng_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(_cuda_rng_state)
        except Exception:
            pass

    expert_ps = [p for _, p in tracked]
    names = [n for n, _ in tracked]

    # DS engine prologue bypass (same as legacy Hutchinson path).
    ds_engine = m0
    prev_flag = getattr(ds_engine, "_running_engine_backward", False)
    try:
        ds_engine._running_engine_backward = True
    except Exception:
        pass

    h_accum = [torch.zeros_like(p, dtype=torch.float32) for p in expert_ps]
    t0 = _time.perf_counter()

    for k in range(max(1, int(k_samples))):
        # ---- Get a fresh batch (advances data_iterator). ----
        try:
            batch = _get_batch_gpt(di)
        except Exception as exc:
            print(f"[proposal] refresh_gnb: get_batch failed at k={k}: {exc}", flush=True)
            try: ds_engine._running_engine_backward = prev_flag
            except Exception: pass
            net.train(was_training); _restore_rng()
            return None
        # pretrain_gpt.get_batch returns (tokens, labels, loss_mask, attn_mask, pos_ids)
        tokens, _labels_true, loss_mask, attention_mask, position_ids = batch[:5]

        # ---- Forward WITHOUT labels → logits (post_language_model_processing
        # returns logits shape [b, s, v] when labels is None). ----
        try:
            out = m0(tokens, position_ids, attention_mask)
        except Exception as exc:
            print(f"[proposal] refresh_gnb: forward failed at k={k}: {exc}", flush=True)
            try: ds_engine._running_engine_backward = prev_flag
            except Exception: pass
            net.train(was_training); _restore_rng()
            return None
        logits = out[0] if isinstance(out, tuple) else out
        if logits is None or not torch.is_tensor(logits):
            print(f"[proposal] refresh_gnb: unexpected forward output at k={k}", flush=True)
            try: ds_engine._running_engine_backward = prev_flag
            except Exception: pass
            net.train(was_training); _restore_rng()
            return None

        # Flatten to [N, V] for CE.
        if logits.dim() >= 2:
            V = logits.size(-1)
            flat_logits = logits.reshape(-1, V)
        else:
            flat_logits = logits

        # ---- Sample labels from model's predictive distribution. ----
        with torch.no_grad():
            probs = torch.softmax(flat_logits.detach().float(), dim=-1)
            y_hat = torch.multinomial(probs, num_samples=1).squeeze(-1)
        del probs

        # ---- Compute CE with sampled labels. ----
        # Use loss_mask if provided to match training's effective batch
        # (drops padding tokens from the estimator).
        try:
            if loss_mask is not None and loss_mask.numel() == flat_logits.size(0):
                per_tok = F.cross_entropy(
                    flat_logits.float(), y_hat, reduction="none",
                )
                denom = loss_mask.view(-1).float().sum().clamp_min(1.0)
                loss_gnb = (per_tok * loss_mask.view(-1).float()).sum() / denom
                B_eff = float(denom.item())
            else:
                loss_gnb = F.cross_entropy(
                    flat_logits.float(), y_hat, reduction="mean",
                )
                B_eff = float(flat_logits.size(0))
        except Exception as exc:
            print(f"[proposal] refresh_gnb: CE failed at k={k}: {exc}", flush=True)
            try: ds_engine._running_engine_backward = prev_flag
            except Exception: pass
            net.train(was_training); _restore_rng()
            return None

        # ---- Single backward (no create_graph → activations freed). ----
        try:
            grads = torch.autograd.grad(
                loss_gnb, expert_ps,
                retain_graph=False, create_graph=False, allow_unused=True,
            )
        except Exception as exc:
            print(f"[proposal] refresh_gnb: grad failed at k={k}: {exc}", flush=True)
            try: ds_engine._running_engine_backward = prev_flag
            except Exception: pass
            net.train(was_training); _restore_rng()
            return None
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, expert_ps)]

        # ---- GNB diagonal accumulator: H_diag_k = B_eff · g² ----
        # The Bartlett identity gives E[g g^T] = H (when label is sampled from
        # the model). The diagonal estimator under the chosen CE reduction is
        # B_eff · g² (undoing the 1/B mean). Top-K is scale-invariant, so the
        # factor only matters for the magnitude log — it keeps |H| in the same
        # physical scale as a per-token Fisher.
        inv_k = 1.0 / max(1, int(k_samples))
        for i, g in enumerate(grads):
            h_accum[i].add_(g.to(torch.float32).pow(2).mul_(B_eff * inv_k))

        # Explicit cleanup to release the forward graph. Avoid per-loop
        # empty_cache(): the allocator will reclaim freed blocks on demand,
        # and repeated cache invalidation can perturb torch.compile graphs
        # and slow down the next training step. Single empty_cache after
        # the loop is enough.
        del logits, out, flat_logits, y_hat, loss_gnb, grads, tokens, batch

    t1 = _time.perf_counter()

    # Restore DS engine flag.
    try:
        ds_engine._running_engine_backward = prev_flag
    except Exception:
        pass
    net.train(was_training)

    # Fold the fresh |H| into H_ema (see _fold_h). Since g² ≥ 0 already,
    # no abs_() needed. Disk persist is opt-in via --proposal-hessian-persist.
    _fold_h(names, h_accum)
    _persist_h_ema(int(iteration))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed_ms = (t1 - t0) * 1000.0
    n_elem = sum(p.numel() for p in expert_ps)

    # ---------- Magnitude log: |H| and |δ| scale ----------
    # Helps calibrate score = |H| × δ² — e.g., if |H| ~ 1e-6 and |δ| ~ 1e-3,
    # score scale ~ 1e-12 which is meaningful for Top-K thresholding.
    try:
        if (not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0):
            import torch as _t
            h_abs_means = []; h_abs_maxs = []
            d_abs_means = []; d_abs_maxs = []
            for (name, p) in tracked:
                h_cache = _tracker._H_ema.get(name)
                if h_cache is not None:
                    h_abs_means.append(float(h_cache.mean().item()))
                    h_abs_maxs.append(float(h_cache.max().item()))
                fc = _tracker.theta_cp.get(name)
                if fc is not None:
                    delta = (p.data.to(_t.float32) - fc.to(_t.float32)).abs_()
                    d_abs_means.append(float(delta.mean().item()))
                    d_abs_maxs.append(float(delta.max().item()))
            if h_abs_means:
                import statistics as _st
                print(
                    f"[proposal] scale @iter{iteration}: "
                    f"|H| mean={_st.mean(h_abs_means):.3e} max={max(h_abs_maxs):.3e}  "
                    f"|\u03b4| mean={_st.mean(d_abs_means):.3e} max={max(d_abs_maxs):.3e}  "
                    f"(score=|H|\u00b7\u03b4\u00b2 \u2248 {_st.mean(h_abs_means) * _st.mean(d_abs_means)**2:.3e})",
                    flush=True,
                )
    except Exception as _exc:
        if (not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0):
            print(f"[proposal] scale log FAILED: {_exc}", flush=True)

    # ---------- Periodic v_t vs |H| comparison dump ----------
    # Per-tensor, GPU-resident Pearson + histc-based Jaccard for 0.1%/1%/10%
    # top-K thresholds. No concat, no CPU fallback, no Python sets. 2-pass:
    #   pass 1: Pearson sums + log-bin histograms (global per quantity)
    #   → compute thresholds τ_v, τ_h, τ_sv, τ_sh from cumulative histograms
    #   pass 2: per-tensor masks + intersection/union counts
    # Total cost ≲ 300 ms/refresh for 189M expert params (vs ~75s CPU version).
    try:
        from ._ds_state import get_mapper as _get_mapper
        _mapper = _get_mapper()
        beta2 = 0.999
        for g in (_config.get("_inner_opt") or []).param_groups if isinstance(_config.get("_inner_opt"), torch.optim.Optimizer) else []:
            if "betas" in g: beta2 = float(g["betas"][1]); break
        inner_opt_for_v = _config.get("_inner_opt")
        bias_corr2 = max(1.0 - beta2 ** int(iteration), 1e-12)

        device = expert_ps[0].device
        # Log-scale histogram covering [1e-40, 1e+20] with 0.1-decade bins.
        # Works across any value scale without a pre-computed max.
        LOG_LO, LOG_HI, BINS = -40.0, 20.0, 600
        def _quant_pair_tensors(name, p, h):
            # Returns (v_flat, h_flat, delta_sq_flat) as GPU fp32 views.
            v_raw = _mapper.get_state(p, "exp_avg_sq") if _mapper is not None else None
            if v_raw is None and inner_opt_for_v is not None:
                v_raw = inner_opt_for_v.state.get(p, {}).get("exp_avg_sq")
            if v_raw is None:
                v_f = torch.zeros_like(p, dtype=torch.float32).flatten()
            else:
                v_f = (v_raw / bias_corr2).to(torch.float32).detach().flatten()
            h_f = h.detach().flatten()  # already fp32, non-negative
            fc = _tracker.theta_cp.get(name)
            if fc is not None:
                d_sq = (p.data.to(torch.float32) - fc.to(torch.float32)).pow_(2).flatten()
            else:
                d_sq = torch.zeros_like(v_f)
            return v_f, h_f, d_sq

        def _log_hist_add(vals_flat, hist):
            # Accumulate log-scale histogram in-place.
            log_v = torch.log10(vals_flat.clamp_min(1e-40))
            idx = ((log_v - LOG_LO) * (BINS / (LOG_HI - LOG_LO))).clamp_(0, BINS - 1).long()
            hist += torch.bincount(idx, minlength=BINS)

        # Pass 1: Pearson sums + histograms.
        pears = {k: 0.0 for k in ("n", "sa_v", "sb_h", "sab_vh", "sa2_v", "sb2_h",
                                   "sa_sv", "sb_sh", "sab_svsh", "sa2_sv", "sb2_sh")}
        hist_v = torch.zeros(BINS, dtype=torch.int64, device=device)
        hist_h = torch.zeros(BINS, dtype=torch.int64, device=device)
        hist_sv = torch.zeros(BINS, dtype=torch.int64, device=device)
        hist_sh = torch.zeros(BINS, dtype=torch.int64, device=device)
        for (name, p), h in zip(tracked, h_accum):
            v_f, h_f, d_sq = _quant_pair_tensors(name, p, h)
            sv_f = v_f * d_sq
            sh_f = h_f * d_sq
            pears["n"] += float(v_f.numel())
            pears["sa_v"] += float(v_f.sum().item())
            pears["sb_h"] += float(h_f.sum().item())
            pears["sab_vh"] += float((v_f * h_f).sum().item())
            pears["sa2_v"] += float((v_f * v_f).sum().item())
            pears["sb2_h"] += float((h_f * h_f).sum().item())
            pears["sa_sv"] += float(sv_f.sum().item())
            pears["sb_sh"] += float(sh_f.sum().item())
            pears["sab_svsh"] += float((sv_f * sh_f).sum().item())
            pears["sa2_sv"] += float((sv_f * sv_f).sum().item())
            pears["sb2_sh"] += float((sh_f * sh_f).sum().item())
            _log_hist_add(v_f, hist_v)
            _log_hist_add(h_f, hist_h)
            _log_hist_add(sv_f, hist_sv)
            _log_hist_add(sh_f, hist_sh)
            del v_f, h_f, d_sq, sv_f, sh_f

        def _pearson_from_sums(n, sa, sb, sab, sa2, sb2):
            if n <= 0:
                return 0.0
            mean_a = sa / n; mean_b = sb / n
            cov = sab / n - mean_a * mean_b
            va = max(sa2 / n - mean_a * mean_a, 0.0)
            vb = max(sb2 / n - mean_b * mean_b, 0.0)
            d = (va * vb) ** 0.5
            return float(cov / d) if d > 0 else 0.0

        pearson_vh = _pearson_from_sums(
            pears["n"], pears["sa_v"], pears["sb_h"], pears["sab_vh"],
            pears["sa2_v"], pears["sb2_h"],
        )
        pearson_svh = _pearson_from_sums(
            pears["n"], pears["sa_sv"], pears["sb_sh"], pears["sab_svsh"],
            pears["sa2_sv"], pears["sb2_sh"],
        )

        def _threshold_from_hist(hist, k_target):
            # Find smallest bin index b such that sum(hist[b:]) >= k_target.
            # Returns log10 value at the lower edge of that bin (τ in linear scale via 10**).
            if k_target <= 0:
                return float("inf")
            # Reverse cumulative sum.
            rev_cum = torch.flip(torch.cumsum(torch.flip(hist, [0]), 0), [0])
            # Indices where rev_cum >= k_target; pick the largest (lowest threshold satisfying count).
            mask = rev_cum >= k_target
            if not bool(mask.any()):
                return 0.0  # below everything → keep all
            last_idx = int(mask.nonzero(as_tuple=True)[0][-1].item())
            bin_width = (LOG_HI - LOG_LO) / BINS
            log_edge = LOG_LO + last_idx * bin_width
            return 10.0 ** log_edge

        ratios = (0.001, 0.01, 0.1)
        N_total = int(pears["n"])
        thresholds = {}  # {ratio: (τ_v, τ_h, τ_sv, τ_sh)}
        for r in ratios:
            k_target = r * N_total
            thresholds[r] = (
                _threshold_from_hist(hist_v, k_target),
                _threshold_from_hist(hist_h, k_target),
                _threshold_from_hist(hist_sv, k_target),
                _threshold_from_hist(hist_sh, k_target),
            )

        # Pass 2: masks + intersection/union counts per ratio.
        jacc = {r: {"inter_vh": 0, "A_v": 0, "A_h": 0,
                    "inter_svh": 0, "A_sv": 0, "A_sh": 0} for r in ratios}
        for (name, p), h in zip(tracked, h_accum):
            v_f, h_f, d_sq = _quant_pair_tensors(name, p, h)
            sv_f = v_f * d_sq
            sh_f = h_f * d_sq
            for r in ratios:
                tv, th_, tsv, tsh = thresholds[r]
                mv = v_f >= tv; mh = h_f >= th_
                msv = sv_f >= tsv; msh = sh_f >= tsh
                jacc[r]["A_v"] += int(mv.sum().item())
                jacc[r]["A_h"] += int(mh.sum().item())
                jacc[r]["inter_vh"] += int((mv & mh).sum().item())
                jacc[r]["A_sv"] += int(msv.sum().item())
                jacc[r]["A_sh"] += int(msh.sum().item())
                jacc[r]["inter_svh"] += int((msv & msh).sum().item())
            del v_f, h_f, d_sq, sv_f, sh_f

        def _j(inter, a, b):
            u = a + b - inter
            return (inter / u) if u > 0 else 0.0

        agg_cmp = {
            "iter": int(iteration),
            "k_samples": int(k_samples),
            "total_expert_numel": N_total,
            "pearson_v_habs": pearson_vh,
            "jaccard_top0.1pct_bare": _j(jacc[0.001]["inter_vh"], jacc[0.001]["A_v"], jacc[0.001]["A_h"]),
            "jaccard_top1pct_bare":   _j(jacc[0.01]["inter_vh"],  jacc[0.01]["A_v"],  jacc[0.01]["A_h"]),
            "jaccard_top10pct_bare":  _j(jacc[0.1]["inter_vh"],   jacc[0.1]["A_v"],   jacc[0.1]["A_h"]),
            "pearson_score": pearson_svh,
            "jaccard_top0.1pct_score": _j(jacc[0.001]["inter_svh"], jacc[0.001]["A_sv"], jacc[0.001]["A_sh"]),
            "jaccard_top1pct_score":   _j(jacc[0.01]["inter_svh"],  jacc[0.01]["A_sv"],  jacc[0.01]["A_sh"]),
            "jaccard_top10pct_score":  _j(jacc[0.1]["inter_svh"],   jacc[0.1]["A_sv"],   jacc[0.1]["A_sh"]),
        }
        if (not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0):
            print(
                f"[proposal] v_t-vs-H @iter{iteration}: "
                f"bare P={agg_cmp['pearson_v_habs']:+.3f} "
                f"J@1%={agg_cmp['jaccard_top1pct_bare']:.3f} "
                f"J@10%={agg_cmp['jaccard_top10pct_bare']:.3f}  "
                f"score P={agg_cmp['pearson_score']:+.3f} "
                f"J@1%={agg_cmp['jaccard_top1pct_score']:.3f} "
                f"J@10%={agg_cmp['jaccard_top10pct_score']:.3f}",
                flush=True,
            )
            cmp_log = _config.get("_hessian_comparison_log")
            if cmp_log:
                import json as _json
                Path(cmp_log).parent.mkdir(parents=True, exist_ok=True)
                with open(cmp_log, "a") as _f:
                    _f.write(_json.dumps(agg_cmp) + "\n")
    except Exception as _exc:
        if (not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0):
            print(f"[proposal] v_t-vs-H comparison FAILED @iter{iteration}: {_exc}", flush=True)

    if (not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0):
        print(
            f"[proposal] refresh_gnb @iter{iteration} K={k_samples} "
            f"{elapsed_ms:.0f} ms  ({n_elem:,} expert elements)",
            flush=True,
        )
    # Restore RNG so training's dropout/MoE-routing stream is unaffected
    # by this refresh probe. (Same restoration also runs on all error paths
    # via _restore_rng defined earlier.)
    _restore_rng()

    return {"iter": iteration, "k_samples": k_samples, "ms": elapsed_ms, "numel": n_elem}


# _hutchinson_compute_H, _gnb_compute_H, _hutchinson_vs_gnb_similarity
# moved to hessian.py (state-free helpers). Imported below for use by
# proposal_refresh_hutchinson.
from .hessian import (
    compute_hessian_hutchinson as _hutchinson_compute_H,
    compute_hessian_gnb as _gnb_compute_H,
    compare_hutchinson_vs_gnb as _hutchinson_vs_gnb_similarity,
)


def proposal_refresh_hutchinson(
    model, forward_step_func, data_iterator, iteration,
    k_samples: int = 4,
) -> Optional[Dict[str, Any]]:
    """Fold a K-sample Hutchinson |H_diag| estimate (true-label CE, 2nd-order
    autograd) into tracker._H_ema, then async-persist it. Optionally runs a
    parallel GNB pass on the same param state and dumps Hutchinson-vs-GNB
    Pearson/Jaccard to `_hessian_cross_check_log`."""
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

    t0 = _time.perf_counter()
    h_hutch = _hutchinson_compute_H(model, forward_step_func, data_iterator, expert_ps, k_samples)
    if h_hutch is None:
        return _finalize(None)
    t1 = _time.perf_counter()

    # Fold Hutchinson |H| into H_ema; disk persist is opt-in.
    _fold_h(names, h_hutch)
    _persist_h_ema(int(iteration))

    elapsed_ms = (t1 - t0) * 1000.0
    n_elem = sum(p.numel() for p in expert_ps)

    # Magnitude log (reuses refresh_gnb formatting).
    try:
        if _should_write():
            import statistics as _st
            h_abs_means = [float(h.mean().item()) for h in h_hutch]
            h_abs_maxs  = [float(h.max().item())  for h in h_hutch]
            d_abs_means = []; d_abs_maxs = []
            for (name, p) in tracked:
                fc = _tracker.theta_cp.get(name)
                if fc is not None:
                    d = (p.data.to(torch.float32) - fc.to(torch.float32)).abs_()
                    d_abs_means.append(float(d.mean().item()))
                    d_abs_maxs.append(float(d.max().item()))
            print(
                f"[proposal] scale @iter{iteration}: "
                f"|H| mean={_st.mean(h_abs_means):.3e} max={max(h_abs_maxs):.3e}  "
                f"|\u03b4| mean={_st.mean(d_abs_means):.3e} max={max(d_abs_maxs):.3e}  "
                f"(score=|H|\u00b7\u03b4\u00b2 \u2248 "
                f"{_st.mean(h_abs_means) * (_st.mean(d_abs_means) ** 2):.3e})",
                flush=True,
            )
    except Exception as _exc:
        if _should_write():
            print(f"[proposal] scale log FAILED: {_exc}", flush=True)

    # -----------------------------------------------------------------------
    # TEMP ANALYSIS: non-expert |H| probe. Disabled (commented-out) — the
    # experiment has finished collecting data. Re-enable by uncommenting.
    # Runs only on score_diag_iters to bound overhead. Dumps expert vs
    # non-expert |H| percentile comparison when --proposal-h-nonexpert-probe-log
    # is set. Adds ~10 GB peak GPU memory on each probe iter (fresh forward
    # for non-expert + Hutchinson 2nd-order).
    # -----------------------------------------------------------------------
    # _ne_probe_log = _config.get("_h_nonexpert_probe_log")
    # _sd_iters = _config.get("_score_diag_iters") or set()
    # if _ne_probe_log and int(iteration) in _sd_iters:
    #     try:
    #         nonexpert_ps = [
    #             p for n, p in inner_model.named_parameters()
    #             if p.requires_grad and getattr(p, "allreduce", True) is True
    #         ]
    #         if nonexpert_ps:
    #             h_ne = _hutchinson_compute_H(
    #                 model, forward_step_func, data_iterator, nonexpert_ps, k_samples,
    #             )
    #             if h_ne is not None:
    #                 LOG_LO, LOG_HI, BINS = -40.0, 20.0, 600
    #                 def _pct_stats(h_list):
    #                     device = h_list[0].device
    #                     hist = torch.zeros(BINS, dtype=torch.int64, device=device)
    #                     total = 0; mean_sum = 0.0; max_val = 0.0
    #                     for hh in h_list:
    #                         hf = hh.detach().flatten()
    #                         total += hf.numel()
    #                         mean_sum += float(hf.sum().item())
    #                         max_val = max(max_val, float(hf.max().item()))
    #                         log_v = torch.log10(hf.clamp_min(1e-40))
    #                         idx = ((log_v - LOG_LO) * (BINS / (LOG_HI - LOG_LO))).clamp_(0, BINS - 1).long()
    #                         hist += torch.bincount(idx, minlength=BINS)
    #                     cum = torch.cumsum(hist, 0)
    #                     def _pct(p):
    #                         target = p / 100.0 * total
    #                         nz = (cum >= target).nonzero(as_tuple=True)[0]
    #                         if nz.numel() == 0:
    #                             return max_val
    #                         i = int(nz[0].item())
    #                         bw = (LOG_HI - LOG_LO) / BINS
    #                         return 10.0 ** (LOG_LO + i * bw)
    #                     return {
    #                         "numel": total,
    #                         "mean": mean_sum / max(total, 1),
    #                         "max": max_val,
    #                         "p50": _pct(50), "p90": _pct(90), "p99": _pct(99),
    #                         "p99.9": _pct(99.9), "p99.99": _pct(99.99),
    #                     }
    #                 agg_ne = {
    #                     "iter": int(iteration),
    #                     "k_samples": int(k_samples),
    #                     "expert_stats": _pct_stats(h_hutch),
    #                     "nonexpert_stats": _pct_stats(h_ne),
    #                 }
    #                 if _should_write():
    #                     es = agg_ne["expert_stats"]
    #                     ns = agg_ne["nonexpert_stats"]
    #                     print(
    #                         f"[proposal] H-probe @iter{iteration}: "
    #                         f"expert  mean={es['mean']:.3e} max={es['max']:.3e} "
    #                         f"p99={es['p99']:.3e}  "
    #                         f"nonexpert mean={ns['mean']:.3e} max={ns['max']:.3e} "
    #                         f"p99={ns['p99']:.3e}",
    #                         flush=True,
    #                     )
    #                     import json as _json
    #                     Path(_ne_probe_log).parent.mkdir(parents=True, exist_ok=True)
    #                     with open(_ne_probe_log, "a") as _f:
    #                         _f.write(_json.dumps(agg_ne) + "\n")
    #                 del h_ne
    #     except Exception as _exc:
    #         if _should_write():
    #             print(f"[proposal] H-probe nonexpert FAILED @iter{iteration}: {_exc}", flush=True)
    # ---------- end temp analysis block ----------

    # -----------------------------------------------------------------------
    # DISABLED: Hutchinson-vs-GNB cross-check block. Commented out after
    # analysis run (exp_etc_hutch / exp_etc_hutch_k4pct) — Pearson/Jaccard
    # data collected. Re-enable by uncommenting when needed. Adds ~5 GB peak
    # GPU memory (fresh GNB forward pass + similarity computation).
    # -----------------------------------------------------------------------
    # if _config.get("_hessian_cross_check", False):
    #     try:
    #         h_gnb = _gnb_compute_H(model, data_iterator, expert_ps, k_samples=1)
    #         if h_gnb is not None:
    #             agg = _hutchinson_vs_gnb_similarity(
    #                 iteration, tracked, h_hutch, h_gnb, _tracker.theta_cp)
    #             if _should_write():
    #                 print(
    #                     f"[proposal] hutch-vs-gnb @iter{iteration}: "
    #                     f"bare P={agg['pearson_hutch_gnb_bare']:+.3f} "
    #                     f"J@1%={agg['jaccard_top1pct_bare']:.3f} "
    #                     f"J@10%={agg['jaccard_top10pct_bare']:.3f}  "
    #                     f"score P={agg['pearson_hutch_gnb_score']:+.3f} "
    #                     f"J@1%={agg['jaccard_top1pct_score']:.3f} "
    #                     f"J@10%={agg['jaccard_top10pct_score']:.3f}",
    #                     flush=True,
    #                 )
    #                 cross_log = _config.get("_h_cross_check_log")
    #                 if cross_log:
    #                     import json as _json
    #                     Path(cross_log).parent.mkdir(parents=True, exist_ok=True)
    #                     with open(cross_log, "a") as _f:
    #                         _f.write(_json.dumps(agg) + "\n")
    #             del h_gnb
    #     except Exception as _exc:
    #         if _should_write():
    #             print(f"[proposal] hutch-vs-gnb comparison FAILED @iter{iteration}: {_exc}", flush=True)

    if _should_write():
        print(
            f"[proposal] refresh_hutchinson @iter{iteration} K={k_samples} "
            f"{elapsed_ms:.0f} ms  ({n_elem:,} expert elements)",
            flush=True,
        )

    return _finalize({"iter": iteration, "k_samples": k_samples, "ms": elapsed_ms, "numel": n_elem})


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
        fc = _tracker.theta_cp.get(name)
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
    _enabled = False
    return summary


# add_proposal_args moved to cli_args.py (re-exported below for backward
# compatibility with pretrain_gpt.py's `from megatron.proposal.cp_hooks
# import add_proposal_args` call site, if any).
from .cli_args import add_proposal_args  # noqa: F401,E402


