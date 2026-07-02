"""H-quantile anchor: FD-Hessian refresh of the score's v→H mapping.

Owns the anchor body. FP32 optimizer-state mapping and reusable pinned
snapshots live in ``fp32_state.py`` because lossy scoring and recovery also
need that training-state view.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .fp32_state import (
    get_engine_optimizer,
    get_mapper,
    pinned_snapshot,
    restore_training_state,
    snapshot_training_state,
)

try:
    import triton
    import triton.language as tl
    _HAS_TRITON_HESSIAN = True
except ImportError:
    _HAS_TRITON_HESSIAN = False


def _pipe():
    from ..controller import get_controller
    return get_controller()


def _should_write() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _zero12_clear_param_grad_refs(ds_opt: Any) -> None:
    if ds_opt is None:
        return
    try:
        for group in getattr(ds_opt, "bit16_groups", ()):
            for p in group:
                p.grad = None
                if hasattr(p, "grad_accum"):
                    p.grad_accum = None
    except Exception:
        pass


def _zero12_discard_anchor_grad_state(ds_opt: Any) -> None:
    """Discard FD-H anchor gradients from ZeRO1/2's normal grad stores.

    The anchor intentionally writes H estimates into DeepSpeed's ordinary
    gradient-accumulation path.  After h_q is built, those values must not reach
    optimizer.step().  Keep this cleanup narrow: do not reset ZeRO reducer
    bookkeeping that the normal forward/backward lifecycle owns.
    """

    if ds_opt is None:
        return
    try:
        for name in ("averaged_gradients", "all_grad_tensors"):
            table = getattr(ds_opt, name, None)
            if isinstance(table, dict):
                for key in list(table.keys()):
                    table[key] = None
            elif isinstance(table, list):
                for idx in range(len(table)):
                    table[idx] = None
        _zero12_clear_param_grad_refs(ds_opt)
    except Exception:
        pass


def _zero12_expert_h_views_from_table(
    ds_opt: Any,
    table_name: str,
    names: List[str],
    params: List[torch.nn.Parameter],
) -> Tuple[List[str], List[torch.Tensor]]:
    table = getattr(ds_opt, table_name, None)
    params_in_partition = getattr(ds_opt, "params_in_partition", None)
    if not isinstance(table, dict) or not isinstance(params_in_partition, list):
        return [], []
    name_by_id = {id(p): str(n) for n, p in zip(names, params)}
    out_names: List[str] = []
    views: List[torch.Tensor] = []
    for group_idx, group_params in enumerate(params_in_partition):
        grad_list = table.get(group_idx)
        if not isinstance(grad_list, list):
            continue
        for param, grad in zip(group_params, grad_list):
            name = name_by_id.get(id(param))
            if name is None or not isinstance(grad, torch.Tensor) or grad.numel() == 0:
                continue
            out_names.append(name)
            views.append(grad.view(-1))
    return out_names, views


def _unwrap_engine(model: Any) -> Any:
    if isinstance(model, (list, tuple)):
        return model[0] if model else None
    return model


def _loss_scale_for_backward(engine: Any) -> float:
    import math

    opt = getattr(engine, "optimizer", None)
    scale = getattr(opt, "loss_scale", 1.0)
    try:
        scale_f = float(scale)
    except Exception:
        scale_f = 1.0
    if not math.isfinite(scale_f) or scale_f <= 0.0:
        return 1.0
    return scale_f


def _gas_scale_for_backward(engine: Any) -> float:
    getter = getattr(engine, "gradient_accumulation_steps", None)
    if not callable(getter):
        return 1.0
    try:
        gas = float(getter())
    except Exception:
        gas = 1.0
    return gas if gas > 0.0 else 1.0


def _capture_anchor_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mpu": None,
    }
    try:
        from megatron.core.tensor_parallel import get_cuda_rng_tracker
        state["mpu"] = get_cuda_rng_tracker().get_states()
    except Exception:
        state["mpu"] = None
    return state


def _restore_anchor_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    try:
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
        if state.get("mpu") is not None:
            try:
                from megatron.core.tensor_parallel import get_cuda_rng_tracker
                get_cuda_rng_tracker().set_states(state["mpu"])
            except Exception:
                pass
    except Exception:
        pass


class _DSGradAccumulationWindow:
    """Match Megatron's no-pipeline DeepSpeed GAS boundary pattern."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._setter = getattr(engine, "set_gradient_accumulation_boundary", None)
        self._opt = getattr(engine, "optimizer", None)
        self._old_engine = None
        self._old_opt = None

    def __enter__(self):
        if not callable(self._setter):
            return self
        self._old_engine = getattr(self._engine, "_is_gradient_accumulation_boundary", None)
        self._old_opt = getattr(self._opt, "is_gradient_accumulation_boundary", None)
        self.set_boundary(False)
        return self

    def set_boundary(self, is_boundary: bool) -> None:
        if callable(self._setter):
            self._setter(bool(is_boundary))

    def __exit__(self, exc_type, exc, tb) -> None:
        if not callable(self._setter):
            return
        try:
            self._engine._is_gradient_accumulation_boundary = self._old_engine
        except Exception:
            pass
        if self._opt is not None:
            try:
                self._opt.is_gradient_accumulation_boundary = self._old_opt
            except Exception:
                pass


def _collect_anchor_h_named_views(
    mapper: Any,
    names: List[str],
    params: List[torch.nn.Parameter],
    *,
    allow_create: bool,
) -> Tuple[List[str], List[torch.Tensor]]:
    out_names: List[str] = []
    views: List[torch.Tensor] = []
    for name, param in zip(names, params):
        try:
            view = mapper.get_grad_accum_view(param).view(-1)
        except Exception:
            if allow_create:
                raise
            continue
        if isinstance(view, torch.Tensor) and view.numel() > 0:
            out_names.append(name)
            views.append(view)
    return out_names, views


def _register_fd_grad_transform_hook(
    param: torch.nn.Parameter,
    *,
    sign_scale: float,
    global_offset: int,
    rng_seed: int,
    k: int,
    scale_probe: Optional[Dict[str, Any]] = None,
):
    def _pre_accumulate_hook(raw_grad):
        if scale_probe is not None and not bool(scale_probe.get("captured", False)):
            try:
                scale_probe["raw_abs_mean"] = float(raw_grad.detach().float().abs().mean().item())
            except Exception:
                scale_probe["raw_abs_mean"] = 0.0
        out = _scale_grad_inplace_with_pcg_z(
            raw_grad,
            sign_scale=sign_scale,
            global_offset=global_offset,
            rng_seed=int(rng_seed),
            k=int(k),
        )
        if scale_probe is not None and not bool(scale_probe.get("captured", False)):
            try:
                scale_probe["scaled_abs_mean"] = float(out.detach().float().abs().mean().item())
            except Exception:
                scale_probe["scaled_abs_mean"] = 0.0
            scale_probe["captured"] = True
        return out

    return param.register_hook(_pre_accumulate_hook)


def _trainable_params(model_or_engine: Any) -> List[torch.nn.Parameter]:
    module = getattr(model_or_engine, "module", model_or_engine)
    if isinstance(module, (list, tuple)):
        modules = module
    else:
        modules = (module,)
    params: List[torch.nn.Parameter] = []
    seen = set()
    for mod in modules:
        if not hasattr(mod, "parameters"):
            continue
        for p in mod.parameters():
            if not getattr(p, "requires_grad", False):
                continue
            ident = id(p)
            if ident in seen:
                continue
            seen.add(ident)
            params.append(p)
    return params


class _ReplayDataIterator:
    """Iterator that replays one raw dataset item without advancing data."""

    def __init__(self, item: Any) -> None:
        self._item = item

    def __iter__(self):
        return self

    def __next__(self):
        return self._item


def proposal_set_anchor_data_iter(it: Any) -> None:
    """Install a dedicated validation data iterator used ONLY by
    proposal_refresh_h_quantile. Keeps the shared valid_data_iterator
    untouched so periodic evaluate_and_print_results draws the same
    sample across (hessian | v_hat | baseline) variants, making val_loss
    comparable."""
    _config = _pipe()._config
    _config["_hessian_dedicated_valid_iter"] = it


def _fold_h(names: List[str], h_new_list: List[torch.Tensor]) -> Dict[str, Any]:
    """Return anchor diagnostics for compatibility with existing logs."""
    _lossy_cp = _pipe()._lossy_cp
    if _lossy_cp is None:
        return {"expert_coverage": 0.0, "skip_ratio": 0.0, "n_skipped": 0}

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
    # H_DIAG: per-anchor, list zero-coverage tensors.
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
            if zero_names:
                print(f"[H_DIAG] zero_names (first 8): {zero_names}", flush=True)
    n_skipped = 0
    for i, (name, h_new) in enumerate(zip(names, h_new_list)):
        if not coverage_flags[i] or skip_mask[i]:
            # H=0 or non-finite: skip fold (keep existing H). The cached
            # anchor-boundary v_q/H LUT still maps Adam v_t to nonzero scores
            # for later selection. Explicit floor was tested and proven
            # redundant.
            n_skipped += 1
            continue
    return {
        "expert_coverage": expert_coverage,
        "skip_ratio": float(n_skipped) / max(n, 1),
        "n_skipped": int(n_skipped),
    }


def proposal_refresh_h_quantile(
    model, forward_step_func, data_iterator, iteration,
) -> Optional[Dict[str, Any]]:
    """Refresh the H quantile mapping."""

    if _pipe()._config.get("cp_mode") == "sequential":
        return {
            "skipped": "sequential_cp",
            "iteration": int(iteration),
        }
    return _refresh_h_quantile_from_finite_diff(
        model, forward_step_func, data_iterator, iteration, k_samples=1,
    )


def _refresh_h_quantile_from_finite_diff(
    model, forward_step_func, data_iterator, iteration,
    k_samples: int = 4,
) -> Optional[Dict[str, Any]]:
    """Overwrite ``H quantile table`` (BF16 group-level views into
    ``_h_quantile_values``) with a fresh finite-difference
    Hessian-vector |H_diag| estimate. Plain 1st-order grads × 2
    (central diff) per Rademacher sample; routing freeze always
    applied around the +/- forwards (production default).

    """
    _ps = _pipe()
    _lossy_cp = _ps._lossy_cp
    _config = _ps._config
    _proposal_config = _ps._proposal_config
    _enabled = _ps._enabled
    _checkpoint_workers = _ps._checkpoint_workers
    if not _enabled:
        return None
    selection_mode = (
        getattr(_lossy_cp, "selection_mode", None)
        if _lossy_cp is not None else None
    )
    if selection_mode is None:
        selection_mode = str(_config.get("selection_mode", "hessian"))
    if selection_mode == "v_hat":
        if _lossy_cp is not None:
            _lossy_cp.update_h_anchor_quantiles(None, refresh_step=int(iteration))
            _lossy_cp.update_v_mapping(global_step=int(iteration))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "skipped": "v_hat",
            "iteration": int(iteration),
        }
    dedicated = _config.get("_hessian_dedicated_valid_iter")
    if dedicated is not None:
        data_iterator = dedicated
    m0 = model[0] if isinstance(model, (list, tuple)) else model
    _actual_pipeline_anchor = _is_pipeline_engine(m0)

    # Anchor/H refresh is an explicit training halt (§0.2 exception). It
    # must not overlap with an older lossy-AOS task, because that task
    # recomputes record predicates from the selection quantile tables.
    # Draining here is not a per-iteration proposal fallback: it only
    # happens at the configured refresh cadence before the anchor begins.
    if _checkpoint_workers is not None:
        _checkpoint_workers.drain()

    import time as _time
    inner_model = _config["_inner_model"]
    tracked = [(n, p) for n, p in inner_model.named_parameters()
               if p.requires_grad and getattr(p, "allreduce", True) is False]
    if not tracked:
        return None

    _cpu_rng_state = torch.get_rng_state()
    _cuda_rng_state = (torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
    if hasattr(m0, "module"):
        net = m0.module
    else:
        net = m0
    was_training = net.training
    net.train()
    def _restore_rng():
        try:
            torch.set_rng_state(_cpu_rng_state)
            if _cuda_rng_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(_cuda_rng_state)
        except Exception: pass

    def _finalize(ret):
        net.train(was_training); _restore_rng()
        return ret

    expert_ps = [p for _, p in tracked]
    names = [n for n, _ in tracked]

    eps = _config.get("_finite_diff_eps")

    fd_k_used = 1

    t0 = _time.perf_counter()
    # Audit: pre-anchor snapshot.
    from ..telemetry import diagnostics as _proposal_diagnostics
    _proposal_diagnostics.vram_audit_step("pre_anchor", dump_breakdown=True, lossy_cp=_lossy_cp)
    # FD core accumulates hook-scaled per-element values into training grad storage.
    # K is fixed to one; robustness comes from the multi-batch anchor loop.
    _group_size_for_fd = max(
        1,
        int(
            getattr(
                _lossy_cp,
                "group_size",
                getattr(_proposal_config, "group_size", 1),
            )
        ),
    )

    import os as _os_off

    # Suppress proposal dispatch side work for the anchor's +eps/-eps forwards.
    _config["_in_anchor_refresh"] = True
    # Snapshot training state — anchor's multiple forward/backward passes
    # advance Megatron's CudaRNGTracker and may write into model param
    # .grad / DeepSpeed hp_grad. Restore in finally so training continues
    # bit-exact regardless of how many anchor passes fired.
    _train_state_snap = snapshot_training_state(_config.get("_inner_opt"), model=model)
    _ds_engine_opt = get_engine_optimizer(model)
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
    _ds_attr_snapshot: Dict[str, Any] = {}
    if _ds_engine_opt is not None:
        for _attr in ("micro_steps", "global_steps", "loss_scale",
                      "cur_iter", "cur_scale"):
            if hasattr(_ds_engine_opt, _attr):
                try:
                    _ds_attr_snapshot[_attr] = getattr(_ds_engine_opt, _attr)
                except Exception:
                    pass
    try:
        # MULTI-BATCH ANCHOR: each batch's routing activates only ~67% of
        # experts; averaging over N batches (each with a different routing)
        # raises H coverage to ≥0.93. N = mb_per_iter/2 (set at init_proposal).
        _n_anchor_batches = max(1, int(_proposal_config.n_anchor_batches))
        # Routing freeze defaults ON (production): the +εz/−εz pair must share
        # one routing decision, else the FD estimate absorbs routing noise.
        # Set PROPOSAL_ROUTING_FREEZE_ENABLE=0 to opt out.
        _use_freeze = (_os_off.environ.get("PROPOSAL_ROUTING_FREEZE_ENABLE", "1")
                       not in ("0", "false", "False"))
        fd_ret = None
        _fold_diag = None
        _h_scale_stats = None
        _n_succ = 0
        _mb_timing_on = (_os_off.environ.get("PROPOSAL_MB_TIMING", "0")
                         in ("1", "true", "TRUE"))
        import time as _time_mb
        _grad_offsets = None
        _grad_accum_views = None
        _expert_ps_sorted = expert_ps
        _names_sorted = [_n for _n, _ in tracked]
        if not _HAS_TRITON_HESSIAN:
            raise RuntimeError("H-quantile anchor requires Triton kernels")
        _get_mapper = get_mapper
        _mapper = _get_mapper()
        if _use_freeze and getattr(_mapper, "_mode", "") == "zero12":
            # ZeRO-1/2 reduces partitioned param grads from post-accumulate
            # hooks. Routing-freeze replay can make the same expert param
            # appear twice to that reducer under activation checkpointing,
            # tripping DeepSpeed's duplicate-reduction invariant. Keep the
            # normal backward path intact and rely on multi-batch averaging
            # for routing noise in this layout.
            _use_freeze = False
            if _should_write() and not _config.get("_warned_zero12_no_routing_freeze", False):
                print(
                    "[proposal] finite_diff: routing freeze disabled for ZeRO1/2 "
                    "anchor; using normal DeepSpeed backward with multi-batch FD.",
                    flush=True,
                )
                _config["_warned_zero12_no_routing_freeze"] = True
        # Mode-aware FD accumulator: use the resident FP32 storage where
        # training normally accumulates gradients. The mapper locates each
        # expert's grad accumulation view for every supported layout
        # (fp16_zero0 / bf16_zero0 / zero12). Accumulating there preserves
        # the same reducer/loss-scale/shard contract as training.
        if _mapper is None:
            raise RuntimeError(
                "anchor needs Fp32StateMapper to locate training grad storage"
            )
        _name_to_idx = (
            getattr(_lossy_cp, "_tname_to_index", None)
            if _lossy_cp is not None else None
        )
        if (
            _name_to_idx is None
            and _lossy_cp is not None
            and hasattr(_lossy_cp, "_tname_list")
        ):
            _name_to_idx = {nm: idx for idx, nm
                            in enumerate(_lossy_cp._tname_list)}
            _lossy_cp._tname_to_index = _name_to_idx
        if _name_to_idx is None:
            _name_to_idx = {nm: idx for idx, nm in enumerate(_names_sorted)}
        _missing = [_n for _n in _names_sorted if _n not in _name_to_idx]
        if _missing:
            raise RuntimeError(
                f"{len(_missing)} expert names not in lossy checkpoint table; first: {_missing[:3]}")
        _order = sorted(range(len(_expert_ps_sorted)),
                        key=lambda i: _name_to_idx[_names_sorted[i]])
        _expert_ps_sorted = [expert_ps[_i] for _i in _order]
        _names_sorted = [_names_sorted[_i] for _i in _order]
        _ne = [int(p.numel()) for p in _expert_ps_sorted]
        _grad_offsets = [0]
        for _n in _ne[:-1]:
            _grad_offsets.append(_grad_offsets[-1] + _n)
        _zero12_mode = getattr(_mapper, "_mode", "") == "zero12"
        if _zero12_mode:
            _h_view_names = []
            _grad_accum_views = []
            _zero12_discard_anchor_grad_state(_ds_engine_opt)
            for _p in _expert_ps_sorted:
                _p.grad = None
                if hasattr(_p, "grad_accum"):
                    _p.grad_accum = None
        else:
            _h_view_names, _grad_accum_views = _collect_anchor_h_named_views(
                _mapper,
                _names_sorted,
                _expert_ps_sorted,
                allow_create=True,
            )
            if not _grad_accum_views:
                raise RuntimeError("anchor could not locate training grad storage")
            for _view in _grad_accum_views:
                if _view.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    raise RuntimeError(f"unsupported grad accumulation storage dtype: {_view.dtype}")
            for _view in _grad_accum_views:
                _view.zero_()
        # Per-tensor θ_0 snapshot (Task 1; pool reuses across anchor calls).
        # NOTE: use _expert_ps_sorted (matches fast-flat grad_offsets;
        # identical to expert_ps when fast-flat is disabled).
        _theta_snap_session = [
            pinned_snapshot(_p.data.detach(), key=("anchor_theta", _i))
            for _i, _p in enumerate(_expert_ps_sorted)
        ]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        try:
            _fd_passes_per_anchor_batch = max(1, int(fd_k_used)) * 2
            _fd_total_backward_passes = int(_n_anchor_batches) * _fd_passes_per_anchor_batch
            for _bi in range(_n_anchor_batches):
                if _mb_timing_on and torch.cuda.is_available():
                    torch.cuda.synchronize()
                _mb_t0 = _time_mb.perf_counter()
                _ret = compute_hessian_finite_diff(
                    model, forward_step_func, data_iterator,
                    _expert_ps_sorted, fd_k_used,
                    eps=eps, use_routing_freeze=_use_freeze,
                    batch_override=None,
                    group_size=_group_size_for_fd,
                    theta_snap_ref=_theta_snap_session,
                    grad_offsets=_grad_offsets,
                    grad_accum_views=_grad_accum_views,
                    ds_pass_base=_bi * _fd_passes_per_anchor_batch,
                    ds_total_backward_passes=_fd_total_backward_passes,
                    final_grad_accum_boundary=True,
                )
                if _mb_timing_on and torch.cuda.is_available():
                    torch.cuda.synchronize()
                if _mb_timing_on:
                    print(f"[MB_TIMING] batch={_bi} {(_time_mb.perf_counter()-_mb_t0)*1000:.0f}ms",
                          flush=True)
                if _ret is None:
                    continue
                _n_succ += 1
                # FD hooks accumulate the signed z·(g₊−g₋)/(2εK) into the
                # same FP32 grad storage training uses; _ret is just a success sentinel.
            if _zero12_mode:
                _final_names, _final_views = _zero12_expert_h_views_from_table(
                    _ds_engine_opt,
                    "averaged_gradients",
                    _names_sorted,
                    _expert_ps_sorted,
                )
                if _final_views:
                    _h_view_names = _final_names
                    _grad_accum_views = _final_views
            else:
                _final_names, _final_views = _collect_anchor_h_named_views(
                    _mapper,
                    _names_sorted,
                    _expert_ps_sorted,
                    allow_create=False,
                )
                if _final_views:
                    _h_view_names = _final_names
                    _grad_accum_views = _final_views
            # After all batches: divide by the success count and take abs in
            # place, then build h_q from the per-expert grad accumulation views.
            if _grad_accum_views and _n_succ > 0:
                _inv_succ = 1.0 / float(_n_succ)
                _curvature_telemetry = (
                    getattr(_lossy_cp, "_curvature_progress_telemetry", None)
                    if _lossy_cp is not None else None
                )
                if _curvature_telemetry is not None and getattr(_curvature_telemetry, "enabled", False):
                    for _view in _grad_accum_views:
                        _view.mul_(_inv_succ)
                    _curvature_telemetry.record_signed_h_refresh(
                        step=int(iteration),
                        groups=getattr(_lossy_cp, "_groups", ()),
                        group_size=int(_group_size_for_fd),
                        names=_h_view_names,
                        signed_h_views=_grad_accum_views,
                        n_anchor_batches=int(_n_succ),
                    )
                    for _view in _grad_accum_views:
                        _view.abs_()
                else:
                    for _view in _grad_accum_views:
                        _view.mul_(_inv_succ).abs_()
                if _lossy_cp is not None:
                    _lossy_cp.update_h_anchor_quantiles(
                        _grad_accum_views,
                        refresh_step=int(iteration),
                    )
                _fold_diag = _fold_h(_h_view_names, _grad_accum_views)
                try:
                    import statistics as _st
                    _h_abs_means = [float(_h.mean().item()) for _h in _grad_accum_views]
                    _h_abs_maxs = [float(_h.max().item()) for _h in _grad_accum_views]
                    _h_scale_stats = (
                        float(_st.mean(_h_abs_means)) if _h_abs_means else 0.0,
                        float(max(_h_abs_maxs)) if _h_abs_maxs else 0.0,
                    )
                except Exception:
                    _h_scale_stats = None
                fd_ret = []
        finally:
            if _zero12_mode:
                _zero12_discard_anchor_grad_state(_ds_engine_opt)
                try:
                    _final_views = None
                except Exception:
                    pass
            for _p in _expert_ps_sorted:
                _p.grad = None
                if hasattr(_p, "grad_accum"):
                    _p.grad_accum = None
            # Restore θ_0 (grad accumulation storage is restored by
            # restore_training_state in the outer finally).
            with torch.no_grad():
                for _p, _snap in zip(_expert_ps_sorted, _theta_snap_session):
                    _p.data.copy_(_snap)
            _grad_accum_views = None
            _h_view_names = None
            _final_names = None
            _final_views = None
            _view = None
            _h = None
            _h_abs_means = None
            _h_abs_maxs = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        # Average
        if fd_ret is not None and _n_succ > 1:
            for _h in fd_ret:
                _h.div_(_n_succ)
        if _should_write() and _n_anchor_batches > 1:
            print(f"[proposal] anchor multi-batch: N={_n_anchor_batches} succ={_n_succ}",
                  flush=True)
        _proposal_diagnostics.vram_audit_step("inside_anchor_fd_peak", dump_breakdown=False)
    finally:
        _config["_in_anchor_refresh"] = False
        if _ds_engine_opt is not None:
            for _attr, _value in _ds_attr_snapshot.items():
                try:
                    setattr(_ds_engine_opt, _attr, _value)
                except Exception:
                    pass
        restore_training_state(_config.get("_inner_opt"), _train_state_snap)
        _zero12_discard_anchor_grad_state(_ds_engine_opt)
        # Final belt-and-suspenders: force-clear any residual lp.grad.
        # model may be a list/tuple (Megatron DDP wrapper) — unwrap first.
        _m_clear = model[0] if isinstance(model, (list, tuple)) else model
        try:
            for _p in _m_clear.parameters():
                if _p.grad is not None:
                    _p.grad = None
                if hasattr(_p, "grad_accum"):
                    _p.grad_accum = None
        except Exception as _exc:
            print(f"[proposal] anchor grad clear err: {_exc}", flush=True)
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
        # V2 keeps anchor snapshots in the reusable pinned pool and does not
        # call empty_cache from the anchor path.

    _proposal_diagnostics.vram_audit_step(
        "post_anchor", dump_breakdown=False,
        extra={"anchor_total_ms": (_time.perf_counter() - t0) * 1000.0},
    )
    t1 = _time.perf_counter()
    if fd_ret is None:
        if _should_write():
            print(f"[proposal] finite_diff failed @iter{iteration}", flush=True)
        return _finalize(None)

    # _fold_h is now mainly diagnostic; h_q was built from DS grad views above.
    if _fold_diag is None:
        _fold_diag = _fold_h(_names_sorted, [])

    elapsed_ms = (t1 - t0) * 1000.0
    n_elem = sum(p.numel() for p in expert_ps)

    # Resolve eps actually used: explicit override > default 1e-3.
    # (Autotune was removed when finite_diff became the only estimator.)
    eps_actual = float(eps) if eps is not None else 1e-3

    diag_record = {
        "iter": int(iteration), "estimator": "finite_diff",
        "expert_coverage": _fold_diag["expert_coverage"],
        "skip_ratio": _fold_diag["skip_ratio"],
        "eps": eps_actual,
        "routing_freeze": bool(_use_freeze and not _actual_pipeline_anchor),
        "pipeline_fd_anchor": bool(_actual_pipeline_anchor),
    }
    _config["_h_diag_records"].append(diag_record)

    # Magnitude log (|H| only — selection delta is computed inline during
    # cutoff/write and is not exposed at refresh time).
    if _should_write():
        if _h_scale_stats is not None:
            print(
                f"[proposal] scale @iter{iteration} (finite_diff): "
                f"|H| mean={_h_scale_stats[0]:.3e} max={_h_scale_stats[1]:.3e}  "
                f"(eps={eps_actual:.3e}, coverage={_fold_diag['expert_coverage']:.3f}, "
                f"skip_ratio={_fold_diag['skip_ratio']:.4f})",
                flush=True,
            )

    if _should_write():
        print(
            f"[proposal] refresh_finite_diff @iter{iteration} K={fd_k_used} "
            f"{elapsed_ms:.0f} ms  ({n_elem:,} expert elements)",
            flush=True,
        )

    ret = _finalize({"iter": iteration, "k_samples": k_samples,
                     "ms": elapsed_ms, "numel": n_elem,
                     "diag": diag_record})

    if _lossy_cp is None:
        return ret

    vq_t0 = _time.perf_counter()
    _lossy_cp.update_v_mapping(global_step=int(iteration))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    vq_refresh_ms = (_time.perf_counter() - vq_t0) * 1000.0
    diag_record["vq_refresh_ms"] = float(vq_refresh_ms)
    if _should_write():
        print(
            f"[proposal] refresh_v_quantile_after_h @iter{iteration} "
            f"{vq_refresh_ms:.0f} ms",
            flush=True,
        )

    return ret


class HQuantileAnchor:
    """Anchor entry. The FD-Hessian refresh body lives in this module's
    ``proposal_refresh_h_quantile`` / ``_refresh_h_quantile_from_finite_diff``."""

    def __init__(self, *, finite_diff_eps: float, n_anchor_batches: int) -> None:
        self.finite_diff_eps = float(finite_diff_eps)
        self.n_anchor_batches = int(n_anchor_batches)

    def set_anchor_data_iter(self, it: Any) -> None:
        proposal_set_anchor_data_iter(it)

    def refresh(self, model, forward_step_func, data_iterator, iteration):
        return proposal_refresh_h_quantile(
            model, forward_step_func, data_iterator, iteration)


def _backward_inplace(forward_fn: Callable[[], Any],
                      expert_ps: List[torch.nn.Parameter],
                      model_for_full_zero: Any,
                      preserve_expert_grad_storage: bool = False,
                      use_engine_backward: bool = False) -> bool:
    """Forward + backward for one FD phase.

    With DeepSpeed engines this calls ``engine.backward`` so the normal
    reducer/epilogue owns where the transformed grad lands. Non-engine fallback
    leaves expert grads in ``p.grad``. Returns True on success.

    ``preserve_expert_grad_storage`` (fast-flat path): caller has assigned
    ``expert.grad`` as a view of a pre-allocated flat tensor. Do NOT set
    expert.grad = None pre-backward (would destroy the view). Caller is
    responsible for zeroing the flat tensor before the next backward.
    """
    try:
        loss, _ = forward_fn()
    except Exception as exc:
        print(f"[proposal] finite_diff: forward failed: {exc}", flush=True)
        return False
    if not torch.is_tensor(loss):
        return False
    if loss.dim() > 0:
        loss = loss.mean()
    _mfz = (model_for_full_zero[0]
            if isinstance(model_for_full_zero, (list, tuple))
            else model_for_full_zero) if model_for_full_zero is not None else None
    if _mfz is not None and not use_engine_backward:
        _exp_ids = {id(p) for p in expert_ps} if preserve_expert_grad_storage else None
        try:
            for _p in _mfz.parameters():
                if _p.grad is None:
                    continue
                if preserve_expert_grad_storage and id(_p) in _exp_ids:
                    continue  # keep flat-view, caller zeroed already
                _p.grad = None
        except Exception:
            pass
    try:
        if use_engine_backward and _mfz is not None and callable(getattr(_mfz, "backward", None)):
            _mfz.backward(loss, retain_graph=False, scale_wrt_gas=True)
        else:
            loss.backward(retain_graph=False)
    except Exception as exc:
        print(f"[proposal] finite_diff: backward failed: {exc}", flush=True)
        return False
    if _mfz is not None and not use_engine_backward:
        _params_id = {id(p) for p in expert_ps}
        try:
            for _p in _mfz.parameters():
                if id(_p) not in _params_id and _p.grad is not None:
                    _p.grad = None
        except Exception as _exc:
            print(f"[FD] non-expert grad clear err: {_exc}", flush=True)
    return True


def _schedule_backward_inplace(
    *,
    forward_step_func: Callable,
    data_iterator: Any,
    model: Any,
) -> bool:
    """Run one no-pipeline Megatron microbatch through the normal schedule.

    This is the anchor equivalent of the body used by
    ``forward_backward_no_pipelining``: schedule.forward_step builds the loss,
    and schedule.backward_step calls ``model.backward`` for DeepSpeed.
    """

    try:
        from megatron.core.pipeline_parallel.schedules import (
            backward_step as _schedule_backward_step,
            forward_step as _schedule_forward_step,
        )
        from megatron.core.utils import get_model_config, get_model_type

        config = get_model_config(model)
        model_type = get_model_type(model)
        forward_data_store: List[Any] = []
        output_tensor = _schedule_forward_step(
            forward_step_func,
            data_iterator,
            model,
            1,
            None,
            forward_data_store,
            config,
            False,
        )
        _schedule_backward_step(
            None,
            output_tensor,
            None,
            model_type,
            config,
            model,
        )
        return True
    except Exception as exc:
        print(f"[proposal] finite_diff: schedule backward failed: {exc}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Fast-flat path: single Triton kernel for perturb & stream, on-the-fly
# Rademacher z generated inside the kernel (register-only, 0 VRAM for z).
# Eliminates 4032+ host-side launches of the per-expert path.
# ---------------------------------------------------------------------------

if _HAS_TRITON_HESSIAN:
    @triton.jit
    def _flat_z_perturb_kernel(
        lp_ptr,          # *bf16, n_elem (in-place add)
        n_elem,
        eps_signed,      # +eps or -eps (float)
        global_offset,   # offset of lp_ptr[0] in the GLOBAL flat (for hash)
        rng_seed_lo,     # split int64 → 2 × int32 for triton portability
        rng_seed_hi,
        k_lo, k_hi,
        BLOCK: tl.constexpr,
    ):
        """In-place ``lp[i] += eps_signed * z(global_offset + i, k, seed)``.
        z = ±1 via element-wise PCG-like hash on the GLOBAL element index
        (so perturb's z matches stream's z even when lp is per-tensor and
        grad is a flat view). Register-only z, 0 VRAM."""
        pid = tl.program_id(0)
        local_offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = local_offs < n_elem
        global_offs = local_offs + global_offset
        lp = tl.load(lp_ptr + local_offs, mask=mask, other=0.0).to(tl.float32)
        x = global_offs.to(tl.uint32) * tl.full((), 0x9E3779B9, tl.uint32)
        x = x ^ tl.full((), rng_seed_lo, tl.uint32)
        x = x ^ (global_offs.to(tl.uint32) >> 16) * tl.full((), 0x85EBCA6B, tl.uint32)
        x = x ^ tl.full((), k_lo, tl.uint32) * tl.full((), 0xC2B2AE35, tl.uint32)
        x = x ^ (x >> 13)
        x = x * tl.full((), 0x27D4EB2D, tl.uint32)
        x = x ^ (x >> 16)
        z = tl.where((x & tl.full((), 1, tl.uint32)) == 0, 1.0, -1.0)
        tl.store(lp_ptr + local_offs,
                 (lp + eps_signed * z).to(tl.bfloat16), mask=mask)


def _flat_z_perturb(lp_tensor: torch.Tensor, eps_signed: float,
                    global_offset: int,
                    rng_seed: int, k: int) -> None:
    """Launch ``_flat_z_perturb_kernel`` on a contiguous BF16 tensor view.
    ``global_offset`` lets the same hash drive both per-tensor perturb and
    a flat-grad signed stream — z is bit-exact identical."""
    n = int(lp_tensor.numel())
    if n == 0:
        return
    if not _HAS_TRITON_HESSIAN:
        gen = torch.Generator(device=lp_tensor.device).manual_seed(
            int(rng_seed) ^ int(k) ^ int(global_offset))
        z = (torch.randint(0, 2, (n,), generator=gen,
                           device=lp_tensor.device,
                           dtype=torch.bfloat16) * 2 - 1)
        lp_tensor.view(-1).add_(z, alpha=float(eps_signed))
        return
    BLOCK = 8192
    n_progs = (n + BLOCK - 1) // BLOCK
    seed64 = int(rng_seed) & ((1 << 64) - 1)
    k64 = int(k) & ((1 << 64) - 1)
    _flat_z_perturb_kernel[(n_progs,)](
        lp_tensor, n,
        float(eps_signed),
        int(global_offset),
        seed64 & 0xFFFFFFFF, (seed64 >> 32) & 0xFFFFFFFF,
        k64 & 0xFFFFFFFF, (k64 >> 32) & 0xFFFFFFFF,
        BLOCK=BLOCK,
    )


def _is_pipeline_engine(obj: Any) -> bool:
    return obj.__class__.__name__ == "PipelineEngine"


def _pipeline_cached_microbatches(engine: Any, data_iterator: Any) -> List[Any]:
    """Read exactly one pipeline batch and keep it reusable for +/- FD passes."""

    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    n_micro = max(1, int(getattr(engine, "micro_batches", 1)))
    return [next(di) for _ in range(n_micro)]


def _pipeline_replay_iterator(cached: List[Any]):
    for item in cached:
        yield item


def _pipeline_exec_batch(
    engine: Any,
    cached_microbatches: List[Any],
    *,
    backward: bool,
) -> None:
    """Run PipelineEngine schedule on cached microbatches.

    Backward mode uses the normal 1F1B train schedule but drops only the final
    reduce/optimizer instructions. Forward-only mode is used by routing-freeze
    capture so the same gate decisions are replayed for +eps and -eps.
    """

    from deepspeed.runtime.pipe import schedule as _ds_schedule

    class _TrainNoStepSchedule(_ds_schedule.TrainSchedule):
        def steps(self):  # type: ignore[override]
            skip = (
                _ds_schedule.ReduceTiedGrads,
                _ds_schedule.ReduceGrads,
                _ds_schedule.OptimizerStep,
            )
            for cmds in super().steps():
                kept = [cmd for cmd in cmds if not isinstance(cmd, skip)]
                yield kept

    sched_cls = _TrainNoStepSchedule if backward else _ds_schedule.InferenceSchedule
    sched = sched_cls(
        micro_batches=int(getattr(engine, "micro_batches", 1)),
        stages=int(getattr(engine, "num_stages", 1)),
        stage_id=int(getattr(engine, "stage_id", 0)),
    )
    old_iter = getattr(engine, "data_iterator", None)
    old_total_loss = getattr(engine, "total_loss", None)
    old_total_additional = getattr(engine, "total_additional_losses", None)
    old_compute_loss = getattr(engine, "_compute_loss", True)
    old_outputs = getattr(engine, "fwd_outputs", None)
    try:
        engine.set_dataiterator(_pipeline_replay_iterator(cached_microbatches))
        engine.module.train()
        engine.total_loss = None
        engine.total_additional_losses = None
        engine._compute_loss = True
        engine._exec_schedule(sched)
    finally:
        engine.set_dataiterator(old_iter)
        engine.total_loss = old_total_loss
        engine.total_additional_losses = old_total_additional
        engine._compute_loss = old_compute_loss
        if old_outputs is not None:
            engine.fwd_outputs = old_outputs


if _HAS_TRITON_HESSIAN:
    @triton.jit
    def _scale_grad_inplace_with_pcg_z_kernel(
        grad_ptr,
        n_elements,
        sign_scale,
        global_offset,
        rng_seed_lo, rng_seed_hi,
        k_lo, k_hi,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        local_off = pid * BLOCK + tl.arange(0, BLOCK)
        mask = local_off < n_elements
        global_elem_off = local_off + global_offset
        grad = tl.load(grad_ptr + local_off, mask=mask, other=0.0).to(tl.float32)
        x = global_elem_off.to(tl.uint32) * tl.full((), 0x9E3779B9, tl.uint32)
        x = x ^ tl.full((), rng_seed_lo, tl.uint32)
        x = x ^ (global_elem_off.to(tl.uint32) >> 16) * tl.full((), 0x85EBCA6B, tl.uint32)
        x = x ^ tl.full((), k_lo, tl.uint32) * tl.full((), 0xC2B2AE35, tl.uint32)
        x = x ^ (x >> 13)
        x = x * tl.full((), 0x27D4EB2D, tl.uint32)
        x = x ^ (x >> 16)
        z = tl.where((x & tl.full((), 1, tl.uint32)) == 0, 1.0, -1.0)
        tl.store(grad_ptr + local_off, grad * z * sign_scale, mask=mask)


def _scale_grad_inplace_with_pcg_z(
    grad: torch.Tensor,
    *,
    sign_scale: float,
    global_offset: int,
    rng_seed: int,
    k: int,
) -> torch.Tensor:
    if not grad.is_contiguous():
        raise RuntimeError("anchor grad hook requires contiguous grad")
    n_total = int(grad.numel())
    if n_total == 0:
        return grad
    if not _HAS_TRITON_HESSIAN:
        raise RuntimeError("anchor grad scaling requires Triton")
    BLOCK = 8192
    n_progs = (n_total + BLOCK - 1) // BLOCK
    seed64 = int(rng_seed) & ((1 << 64) - 1)
    k64 = int(k) & ((1 << 64) - 1)
    _scale_grad_inplace_with_pcg_z_kernel[(n_progs,)](
        grad.reshape(-1), n_total, float(sign_scale),
        int(global_offset),
        seed64 & 0xFFFFFFFF, (seed64 >> 32) & 0xFFFFFFFF,
        k64 & 0xFFFFFFFF, (k64 >> 32) & 0xFFFFFFFF,
        BLOCK=BLOCK,
    )
    return grad


def _compute_fd_fast_flat(
    forward_loss_fn: Callable[[], Tuple[torch.Tensor, Any]],
    expert_ps: List[torch.nn.Parameter],
    k_samples: int,
    eps: float,
    gates_for_freeze: Optional[List[Tuple[str, torch.nn.Module]]],
    rng_seed: int,
    group_size: int,
    model_for_full_zero: Any,
    grad_offsets: List[int],
    restore_src_list: List[torch.Tensor],
    grad_accum_views: List[torch.Tensor],
    ds_pass_base: int = 0,
    ds_total_backward_passes: Optional[int] = None,
    pipeline_engine: Optional[Any] = None,
    pipeline_microbatches: Optional[List[Any]] = None,
    schedule_backward_fn: Optional[Callable[[], bool]] = None,
    final_grad_accum_boundary: bool = True,
    phase_rng_state: Optional[Dict[str, Any]] = None,
) -> Optional[List[torch.Tensor]]:
    """Finite-difference anchor over the normal training backward path.

    - Proposal hooks transform the raw autograd grad in-place to
      ``sign * z * grad / (2 eps K)`` before DeepSpeed consumes it.
    - DeepSpeed/Megatron then reduce/accumulate that transformed grad using
      the same storage and epilogue as a normal backward.
    - Perturb: per-expert Triton kernel using GLOBAL ``grad_offsets[i]``.
    - lp storage is per-expert (FP16_Optimizer / mixed groups); no lp_flat
      required. theta_snap is per-expert pinned (caller-managed).
    """
    from .routing_freeze import RoutingFreezeContext

    if grad_accum_views is None:
        raise RuntimeError("H-quantile anchor requires resident FP32 grad views")
    preserve_param_grad_storage = any(
        isinstance(view, torch.Tensor) and view.dtype != torch.float32
        for view in grad_accum_views
    )

    import os as _os_san
    _timing_on = (_os_san.environ.get("PROPOSAL_FD_TIMING", "0")
                  in ("1", "true", "TRUE"))
    import time as _time

    def _sync():
        if _timing_on and torch.cuda.is_available():
            torch.cuda.synchronize()

    inv_step = 1.0 / (2.0 * float(eps))
    inv_k = 1.0 / max(1, int(k_samples))

    ctx_mgr = (RoutingFreezeContext(gates_for_freeze)
               if gates_for_freeze else None)

    def _restore_theta0() -> None:
        with torch.no_grad():
            for _p, _snap in zip(expert_ps, restore_src_list):
                _p.data.copy_(_snap, non_blocking=True)

    def _clear_pipeline_grads() -> None:
        if pipeline_engine is None:
            return
        try:
            for _p in pipeline_engine.module.parameters():
                _p.grad = None
        except Exception:
            for _p in expert_ps:
                _p.grad = None

    _tsum = {"perturb": 0.0, "hook": 0.0, "bwd": 0.0}
    ds_engine = _unwrap_engine(model_for_full_zero)
    use_engine_backward = (
        pipeline_engine is None
        and ds_engine is not None
        and callable(getattr(ds_engine, "backward", None))
        and getattr(ds_engine, "optimizer", None) is not None
    )
    ds_opt = getattr(ds_engine, "optimizer", None) if use_engine_backward else None
    expert_param_ids = {id(p) for p in expert_ps}
    nonexpert_params = [
        p for p in _trainable_params(ds_engine if use_engine_backward else model_for_full_zero)
        if id(p) not in expert_param_ids
    ]
    next_nonexpert_offset = sum(int(p.numel()) for p in expert_ps)
    nonexpert_offsets: Dict[int, int] = {}
    for p in nonexpert_params:
        nonexpert_offsets[id(p)] = next_nonexpert_offset
        next_nonexpert_offset += int(p.numel())
    local_backward_passes = max(1, int(k_samples)) * 2
    total_backward_passes = (
        int(ds_total_backward_passes)
        if ds_total_backward_passes is not None
        else local_backward_passes
    )
    nonfinal_backward_context = nullcontext
    if schedule_backward_fn is not None and ds_engine is not None:
        try:
            from megatron.core.utils import get_model_config
            config = get_model_config(ds_engine)
            no_sync_func = getattr(config, "no_sync_func", None)
            if callable(no_sync_func):
                nonfinal_backward_context = no_sync_func
        except Exception:
            nonfinal_backward_context = nullcontext
    pass_idx = 0
    try:
        with (ctx_mgr if ctx_mgr is not None else nullcontext()):
            if ctx_mgr is not None:
                try:
                    _restore_anchor_rng_state(phase_rng_state)
                    if pipeline_engine is not None:
                        if pipeline_microbatches is None:
                            raise RuntimeError("pipeline FD missing cached microbatches")
                        _restore_theta0()
                        ctx_mgr.capture(
                            lambda: _pipeline_exec_batch(
                                pipeline_engine,
                                pipeline_microbatches,
                                backward=False,
                            )
                        )
                    else:
                        ctx_mgr.capture(forward_loss_fn)
                except Exception as exc:
                    print(f"[proposal] finite_diff: routing-freeze capture failed: {exc}",
                          flush=True)
                    return None
            _sync()
            _t_loop = _time.perf_counter()
            ds_window_ctx = (
                _DSGradAccumulationWindow(ds_engine)
                if use_engine_backward and pipeline_engine is None
                else nullcontext()
            )
            with ds_window_ctx as ds_window:
                _scale_probe_on = str(os.environ.get("PROPOSAL_H_REFRESH_SCALE_PROBE", "0")).lower() in (
                    "1", "true", "yes", "on",
                )
                _scale_probe_printed = False
                for k in range(max(1, int(k_samples))):
                    for sign in (+1.0, -1.0):
                        pass_idx += 1
                        global_pass_idx = int(ds_pass_base) + pass_idx
                        is_last_backward = (
                            bool(final_grad_accum_boundary)
                            and global_pass_idx >= total_backward_passes
                        )
                        if is_last_backward and isinstance(ds_window, _DSGradAccumulationWindow):
                            ds_window.set_boundary(True)
                        _sync(); _t0 = _time.perf_counter()
                        # Per-tensor perturb: copy snap → add sign×eps × z(global_off+i).
                        with torch.no_grad():
                            for i, p in enumerate(expert_ps):
                                p.data.copy_(restore_src_list[i], non_blocking=True)
                                _flat_z_perturb(
                                    p.data, sign * float(eps),
                                    global_offset=int(grad_offsets[i]),
                                    rng_seed=int(rng_seed), k=int(k),
                                )
                        _sync(); _t1 = _time.perf_counter()
                        handles = []
                        loss_scale = (
                            _loss_scale_for_backward(ds_engine)
                            if use_engine_backward else 1.0
                        )
                        gas_scale = (
                            _gas_scale_for_backward(ds_engine)
                            if use_engine_backward else 1.0
                        )
                        sign_scale = sign * inv_step * inv_k * float(gas_scale) / float(loss_scale)
                        scale_probe = None
                        if _scale_probe_on and not _scale_probe_printed:
                            scale_probe = {
                                "captured": False,
                                "loss_scale": float(loss_scale),
                                "gas_scale": float(gas_scale),
                                "inv_step": float(inv_step),
                                "inv_k": float(inv_k),
                                "sign_scale": float(sign_scale),
                            }
                        for hook_i, hook_param in enumerate(expert_ps):
                            hook_global_offset = int(grad_offsets[hook_i])
                            handles.append(
                                _register_fd_grad_transform_hook(
                                    hook_param,
                                    sign_scale=sign_scale,
                                    global_offset=hook_global_offset,
                                    rng_seed=int(rng_seed),
                                    k=int(k),
                                    scale_probe=scale_probe,
                                )
                            )
                        for hook_param in nonexpert_params:
                            handles.append(
                                _register_fd_grad_transform_hook(
                                    hook_param,
                                    sign_scale=sign_scale,
                                    global_offset=int(nonexpert_offsets[id(hook_param)]),
                                    rng_seed=int(rng_seed),
                                    k=int(k),
                                    scale_probe=scale_probe,
                                )
                            )
                        _sync(); _t_hook = _time.perf_counter()
                        try:
                            bwd_context = (
                                nullcontext()
                                if is_last_backward
                                else nonfinal_backward_context()
                            )
                            with bwd_context:
                                _restore_anchor_rng_state(phase_rng_state)
                                if pipeline_engine is not None:
                                    if pipeline_microbatches is None:
                                        raise RuntimeError("pipeline FD missing cached microbatches")
                                    _pipeline_exec_batch(
                                        pipeline_engine,
                                        pipeline_microbatches,
                                        backward=True,
                                    )
                                    _clear_pipeline_grads()
                                elif schedule_backward_fn is not None:
                                    _ok = schedule_backward_fn()
                                    if not _ok:
                                        return None
                                else:
                                    if not _backward_inplace(
                                        forward_loss_fn,
                                        expert_ps,
                                        model_for_full_zero,
                                        preserve_expert_grad_storage=preserve_param_grad_storage,
                                        use_engine_backward=use_engine_backward,
                                    ):
                                        return None
                        finally:
                            for handle in handles:
                                handle.remove()
                            if not preserve_param_grad_storage:
                                for p in expert_ps:
                                    p.grad = None
                        _sync(); _t2 = _time.perf_counter()
                        if scale_probe is not None and not _scale_probe_printed:
                            print(
                                "[proposal] h_refresh_scale_probe "
                                f"loss_scale={scale_probe.get('loss_scale', 1.0):.6g} "
                                f"gas={scale_probe.get('gas_scale', 1.0):.6g} "
                                f"inv_step={scale_probe.get('inv_step', 0.0):.6g} "
                                f"inv_k={scale_probe.get('inv_k', 0.0):.6g} "
                                f"sign_scale={scale_probe.get('sign_scale', 0.0):.6e} "
                                f"raw_abs_mean={scale_probe.get('raw_abs_mean', 0.0):.6e} "
                                f"scaled_abs_mean={scale_probe.get('scaled_abs_mean', 0.0):.6e}",
                                flush=True,
                            )
                            _scale_probe_printed = True
                        _tsum["perturb"] += (_t1 - _t0)
                        _tsum["hook"]    += (_t_hook - _t1)
                        _tsum["bwd"]     += (_t2 - _t_hook)
            if _timing_on:
                _sync()
                _t_done = _time.perf_counter()
                print(f"[FD_FAST] loop={(_t_done-_t_loop)*1000:.0f}ms "
                      f"(perturb={_tsum['perturb']*1000:.0f} "
                      f"hook={_tsum['hook']*1000:.0f} "
                      f"bwd={_tsum['bwd']*1000:.0f} "
                      f")",
                      flush=True)
    except Exception as exc:
        import traceback as _tb
        print(f"[proposal] finite_diff: fast-flat refresh failed: {type(exc).__name__}: {exc}\n"
              f"{_tb.format_exc()}", flush=True)
        return None
    # NB: lp left in stale -εz state; caller's finally restores from snaps.
    # DeepSpeed-owned grad accumulation views carry the signed running sum
    # across the multi-batch loop.
    # The caller takes the final abs + quantile build after dividing by the
    # success count. Sentinel non-None return signals success.
    for view in grad_accum_views:
        if not bool(torch.isfinite(view).all()):
            print("[proposal] finite_diff: NaN/Inf in resident grad result", flush=True)
            return None
    return []  # success sentinel; h_refresh owns final reduction


def compute_hessian_finite_diff_core(
    forward_loss_fn: Callable[[], Tuple[torch.Tensor, Any]],
    expert_ps: List[torch.nn.Parameter],
    k_samples: int,
    eps: float,
    gates_for_freeze: Optional[List[Tuple[str, torch.nn.Module]]] = None,
    rng_seed: int = 0xFD15CAFE,
    group_size: int = 1,
    model_for_full_zero: Any = None,
    theta_snap_ref: Optional[List[torch.Tensor]] = None,
    grad_offsets: Optional[List[int]] = None,
    grad_accum_views: Optional[List[torch.Tensor]] = None,
    ds_pass_base: int = 0,
    ds_total_backward_passes: Optional[int] = None,
    pipeline_engine: Optional[Any] = None,
    pipeline_microbatches: Optional[List[Any]] = None,
    schedule_backward_fn: Optional[Callable[[], bool]] = None,
    final_grad_accum_boundary: bool = True,
    phase_rng_state: Optional[Dict[str, Any]] = None,
) -> Optional[List[torch.Tensor]]:
    """Accumulate one finite-difference batch through the normal backward path.

    The anchor caller provides per-expert offsets and a theta snapshot. Hooks
    scale each raw grad by deterministic PCG ``z`` before DeepSpeed's existing
    grad handling consumes it. The removed per-tensor group accumulator path is
    intentionally not available.
    """
    if not expert_ps or eps <= 0:
        return None
    # Fast dispatch: lp stays per-tensor (no constraint on DS optimizer
    # layout); hooks scale raw grads before the normal DS reducer sees them.
    if (grad_offsets is not None
            and theta_snap_ref is not None
            and grad_accum_views is not None
            and _HAS_TRITON_HESSIAN):
        return _compute_fd_fast_flat(
            forward_loss_fn=forward_loss_fn,
            expert_ps=expert_ps,
            k_samples=k_samples,
            eps=eps,
            gates_for_freeze=gates_for_freeze,
            rng_seed=rng_seed,
            group_size=group_size,
            model_for_full_zero=model_for_full_zero,
            grad_offsets=grad_offsets,
            restore_src_list=theta_snap_ref,
            grad_accum_views=grad_accum_views,
            ds_pass_base=ds_pass_base,
            ds_total_backward_passes=ds_total_backward_passes,
            pipeline_engine=pipeline_engine,
            pipeline_microbatches=pipeline_microbatches,
            schedule_backward_fn=schedule_backward_fn,
            final_grad_accum_boundary=final_grad_accum_boundary,
            phase_rng_state=phase_rng_state,
        )

    raise RuntimeError(
        "H-quantile anchor refresh requires grad_offsets, "
        "theta_snap_ref, and Triton kernels"
    )


def compute_hessian_finite_diff(
    model: Any,
    forward_step_func: Callable,
    data_iterator: Any,
    expert_ps: List[torch.nn.Parameter],
    k_samples: int,
    eps: Optional[float] = None,
    use_routing_freeze: bool = True,
    batch_override: Optional[Any] = None,
    rng_seed: int = 0xBADCAFE,
    group_size: int = 1,
    theta_snap_ref: Optional[List[torch.Tensor]] = None,
    grad_offsets: Optional[List[int]] = None,
    grad_accum_views: Optional[List[torch.Tensor]] = None,
    ds_pass_base: int = 0,
    ds_total_backward_passes: Optional[int] = None,
    final_grad_accum_boundary: bool = True,
    phase_rng_state: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Run one central finite-difference batch into training grad storage.

    The public compatibility wrapper still builds the cached batch and routing
    freeze context. Actual H-quantile construction happens outside this module:
    the caller averages the DS grad views, takes ``abs()``, and refreshes
    ``h_q``.
    """
    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    m0 = model[0] if isinstance(model, (list, tuple)) else model
    if not expert_ps:
        return None

    pipeline_engine = m0 if _is_pipeline_engine(m0) else None
    pipeline_microbatches = None
    if pipeline_engine is not None:
        try:
            pipeline_microbatches = _pipeline_cached_microbatches(pipeline_engine, di)
        except Exception as exc:
            print(f"[proposal] finite_diff: pipeline batch cache failed: {exc}", flush=True)
            return None
        batch = None
    else:
        # ---- Load raw batch ONCE (Common Random Numbers). ----
        # Replay the raw dataset item through forward_step_func on every
        # +/- phase.  This preserves the normal GPT loss path
        # (labels=labels, vocab-parallel CE, MoE loss) without advancing the
        # shared iterator or hand-building a different loss graph.
        if batch_override is not None:
            raw_batch = batch_override
        else:
            try:
                raw_batch = next(di) if di is not None else None
            except Exception as exc:
                print(f"[proposal] finite_diff: raw batch capture failed: {exc}", flush=True)
                return None

    if pipeline_engine is None:
        def _forward_loss() -> Tuple[torch.Tensor, Dict[str, Any]]:
            output_tensor, loss_func = forward_step_func(
                _ReplayDataIterator(raw_batch),
                m0,
            )
            loss, loss_dict = loss_func(output_tensor)
            return loss, loss_dict

        def _schedule_backward() -> bool:
            return _schedule_backward_inplace(
                forward_step_func=forward_step_func,
                data_iterator=_ReplayDataIterator(raw_batch),
                model=m0,
            )
    else:
        def _forward_loss() -> Tuple[torch.Tensor, Dict[str, Any]]:
            raise RuntimeError("pipeline FD uses PipelineEngine schedules")
        _schedule_backward = None

    # ---- Routing freeze setup ----
    from .routing_freeze import identify_gate_modules_safe
    gates_for_freeze: Optional[List[Tuple[str, torch.nn.Module]]] = None
    if use_routing_freeze and pipeline_engine is not None:
        if _should_write():
            print(
                "[proposal] finite_diff: DeepSpeed PipelineEngine schedule "
                "anchor disables routing-freeze replay; cached +/- "
                "microbatches are still shared.",
                flush=True,
            )
    elif use_routing_freeze:
        gates, _rules = identify_gate_modules_safe(model)
        if not gates:
            print("[proposal] finite_diff: WARN no gates identified; running "
                  "without routing freeze", flush=True)
        else:
            gates_for_freeze = gates

    if phase_rng_state is None:
        phase_rng_state = _capture_anchor_rng_state()

    # ---- eps resolution ----
    # Default: 1e-3, validated on tiny (4-layer 256d) and production
    # (12-layer 768d) MoE setups under BF16+ZeRO-1 with K=16. Override
    # via `--proposal-finite-diff-eps` CLI or `PROPOSAL_FD_EPS` env if a
    # very different scale ever proves the default unsafe.
    eps_value = 1e-3 if eps is None else float(eps)

    # ---- Main K-sample finite-diff loop ----
    # Perturb expert lp directly (DS BF16 optimizer stores each lp's .data
    # as a view into the bf16_groups_flat backing buffer — verified at
    # bf16_optimizer.py:283-286: `p.data = q.data` where q is unflatten
    # view), so ``p.data.copy_(snap).add_(z, alpha=eps)`` updates the
    # bytes forward reads. theta_0 is preserved in a single pinned-host
    # snapshot (allocated once via pinned_snapshot pool, reused across
    # anchors). Activation checkpoint follows the model's normal training
    # setting (no anchor-only override).
    return compute_hessian_finite_diff_core(
        forward_loss_fn=_forward_loss,
        expert_ps=expert_ps,
        k_samples=k_samples,
        eps=eps_value,
        gates_for_freeze=gates_for_freeze,
        rng_seed=rng_seed,
        group_size=group_size,
        model_for_full_zero=model,
        theta_snap_ref=theta_snap_ref,
        grad_offsets=grad_offsets,
        grad_accum_views=grad_accum_views,
        ds_pass_base=ds_pass_base,
        ds_total_backward_passes=ds_total_backward_passes,
        pipeline_engine=pipeline_engine,
        pipeline_microbatches=pipeline_microbatches,
        schedule_backward_fn=_schedule_backward,
        final_grad_accum_boundary=final_grad_accum_boundary,
        phase_rng_state=phase_rng_state,
    )
