"""H-quantile anchor: FD-Hessian refresh of the score's v→H mapping.

Owns the anchor body. FP32 optimizer-state mapping and reusable pinned
snapshots live in ``fp32_state.py`` because lossy scoring and recovery also
need that training-state view.
"""
from __future__ import annotations

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

    if not _pipe()._config.get("expert_lossy_enabled", True):
        return {
            "skipped": "expert_lossy_disabled",
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
    if not _enabled or _lossy_cp is None:
        return None
    if getattr(_lossy_cp, "selection_mode", "hessian") == "v_hat":
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
        return ret

    expert_ps = [p for _, p in tracked]
    names = [n for n, _ in tracked]

    eps = _config.get("_finite_diff_eps")

    fd_k_used = 1

    t0 = _time.perf_counter()
    # Audit: pre-anchor snapshot.
    from ..telemetry import diagnostics as _proposal_diagnostics
    _proposal_diagnostics.vram_audit_step("pre_anchor", dump_breakdown=True, lossy_cp=_lossy_cp)
    # FD core accumulates hook-scaled per-element values into grad_flat.
    # K is fixed to one; robustness comes from the multi-batch anchor loop.
    _group_size_for_fd = max(1, int(getattr(_lossy_cp, "group_size", 1)))

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
    _saved_grad_acc_hooks = None
    if _ds_engine_opt is not None and hasattr(_ds_engine_opt, "_grad_acc_hooks"):
        # BF16_Optimizer routes param grads into its own flat buffer via these
        # hooks; detach them so the anchor's FD hooks own the grads. FP16/ZeRO
        # optimizers have no such hooks (branch skipped).
        _saved_grad_acc_hooks = _ds_engine_opt._grad_acc_hooks
        for _h in _saved_grad_acc_hooks:
            try: _h.remove()
            except Exception: pass
        _ds_engine_opt._grad_acc_hooks = []
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
        # Mode-aware FD accumulator: use the resident FP32 storage where
        # training normally accumulates gradients. The mapper locates each
        # expert's grad accumulation view for every supported layout
        # (fp16_zero0 / bf16_zero0 / zero12). Accumulating there preserves
        # the same reducer/loss-scale/shard contract as training.
        if _mapper is None:
            raise RuntimeError(
                "anchor needs Fp32StateMapper to locate training grad storage"
            )
        _name_to_idx = getattr(_lossy_cp, "_tname_to_index", None)
        if _name_to_idx is None and hasattr(_lossy_cp, "_tname_list"):
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
        _grad_accum_views = []
        for _p in _expert_ps_sorted:
            _view = _mapper.get_grad_accum_view(_p).view(-1)
            if _view.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise RuntimeError(f"unsupported grad accumulation storage dtype: {_view.dtype}")
            if int(_view.numel()) != int(_p.numel()):
                raise RuntimeError("grad accumulation view size mismatch")
            _grad_accum_views.append(_view)
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
            # After all batches: divide by the success count and take abs in
            # place, then build h_q from the per-expert grad accumulation views.
            if _grad_accum_views and _n_succ > 0:
                _inv_succ = 1.0 / float(_n_succ)
                for _view in _grad_accum_views:
                    _view.mul_(_inv_succ).abs_()
                _lossy_cp.update_h_anchor_quantiles(
                    _grad_accum_views,
                    refresh_step=int(iteration),
                )
                fd_ret = []
        finally:
            for _p in _expert_ps_sorted:
                _p.grad = None
            # Restore θ_0 (grad accumulation storage is restored by
            # restore_training_state in the outer finally).
            with torch.no_grad():
                for _p, _snap in zip(_expert_ps_sorted, _theta_snap_session):
                    _p.data.copy_(_snap)
            _grad_accum_views = None
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
        # Re-register DS grad-acc hooks so next training iter's backward
        # accumulates into the flat buffer again (BF16_Optimizer only).
        if _ds_engine_opt is not None and _saved_grad_acc_hooks is not None:
            try:
                _ds_engine_opt.create_grad_acc_hooks()
            except Exception:
                pass
        if _ds_engine_opt is not None:
            for _attr, _value in _ds_attr_snapshot.items():
                try:
                    setattr(_ds_engine_opt, _attr, _value)
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
    h_fd = fd_ret

    if h_fd is None:
        if _should_write():
            print(f"[proposal] finite_diff failed @iter{iteration}", flush=True)
        return _finalize(None)

    # _fold_h is now mainly diagnostic; h_q was built from grad_flat above.
    _fold_diag = _fold_h(_names_sorted, h_fd)

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
        "routing_freeze": True,
    }
    _config["_h_diag_records"].append(diag_record)

    # Magnitude log (|H| only — selection delta is computed inline during
    # cutoff/write and is not exposed at refresh time).
    if _should_write():
        try:
            import statistics as _st
            h_abs_means = [float(h.mean().item()) for h in h_fd]
            h_abs_maxs = [float(h.max().item()) for h in h_fd]
            print(
                f"[proposal] scale @iter{iteration} (finite_diff): "
                f"|H| mean={_st.mean(h_abs_means):.3e} max={max(h_abs_maxs):.3e}  "
                f"(eps={eps_actual:.3e}, coverage={_fold_diag['expert_coverage']:.3f}, "
                f"skip_ratio={_fold_diag['skip_ratio']:.4f})",
                flush=True,
            )
        except Exception:
            pass

    if _should_write():
        print(
            f"[proposal] refresh_finite_diff @iter{iteration} K={fd_k_used} "
            f"{elapsed_ms:.0f} ms  ({n_elem:,} expert elements)",
            flush=True,
        )

    ret = _finalize({"iter": iteration, "k_samples": k_samples,
                     "ms": elapsed_ms, "numel": n_elem,
                     "diag": diag_record})

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
                      preserve_expert_grad_storage: bool = False) -> bool:
    """Forward + backward; leaves expert grads alive in ``p.grad``.
    Clears non-expert grads after backward. Returns True on success.

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
    if _mfz is not None:
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
        loss.backward(retain_graph=False)
    except Exception as exc:
        print(f"[proposal] finite_diff: backward failed: {exc}", flush=True)
        return False
    if _mfz is not None:
        _params_id = {id(p) for p in expert_ps}
        try:
            for _p in _mfz.parameters():
                if id(_p) not in _params_id and _p.grad is not None:
                    _p.grad = None
        except Exception as _exc:
            print(f"[FD] non-expert grad clear err: {_exc}", flush=True)
    return True


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


if _HAS_TRITON_HESSIAN:
    @triton.jit
    def _accumulate_grad_to_fp32_with_pcg_z_kernel(
        grad_ptr,
        target_ptr,
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
        prev = tl.load(target_ptr + local_off, mask=mask, other=0.0).to(tl.float32)
        x = global_elem_off.to(tl.uint32) * tl.full((), 0x9E3779B9, tl.uint32)
        x = x ^ tl.full((), rng_seed_lo, tl.uint32)
        x = x ^ (global_elem_off.to(tl.uint32) >> 16) * tl.full((), 0x85EBCA6B, tl.uint32)
        x = x ^ tl.full((), k_lo, tl.uint32) * tl.full((), 0xC2B2AE35, tl.uint32)
        x = x ^ (x >> 13)
        x = x * tl.full((), 0x27D4EB2D, tl.uint32)
        x = x ^ (x >> 16)
        z = tl.where((x & tl.full((), 1, tl.uint32)) == 0, 1.0, -1.0)
        tl.store(target_ptr + local_off, prev + grad * z * sign_scale, mask=mask)


def _accumulate_grad_to_fp32_with_pcg_z(
    grad: torch.Tensor,
    target: torch.Tensor,
    *,
    sign_scale: float,
    global_offset: int,
    rng_seed: int,
    k: int,
) -> torch.Tensor:
    """Accumulate one raw grad into a resident FP32 DS grad slice."""

    if not grad.is_contiguous() or not target.is_contiguous():
        raise RuntimeError("anchor grad hook requires contiguous grad and target")
    n_total = int(grad.numel())
    if n_total == 0:
        return grad
    if target.numel() != n_total or target.dtype != torch.float32:
        raise RuntimeError("anchor fp32 grad target shape/dtype mismatch")
    if not _HAS_TRITON_HESSIAN:
        raise RuntimeError("anchor fp32 grad accumulation requires Triton")
    BLOCK = 8192
    n_progs = (n_total + BLOCK - 1) // BLOCK
    seed64 = int(rng_seed) & ((1 << 64) - 1)
    k64 = int(k) & ((1 << 64) - 1)
    _accumulate_grad_to_fp32_with_pcg_z_kernel[(n_progs,)](
        grad.reshape(-1), target.reshape(-1), n_total, float(sign_scale),
        int(global_offset),
        seed64 & 0xFFFFFFFF, (seed64 >> 32) & 0xFFFFFFFF,
        k64 & 0xFFFFFFFF, (k64 >> 32) & 0xFFFFFFFF,
        BLOCK=BLOCK,
    )
    grad.zero_()
    return grad


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
    grad_flat: torch.Tensor,
    grad_offsets: List[int],
    restore_src_list: List[torch.Tensor],
    grad_accum_views: List[torch.Tensor],
) -> Optional[List[torch.Tensor]]:
    """Fast FD with batched grad-flat stream + per-tensor hash-based perturb.

    - Hooks accumulate directly into resident FP32 DS grad slices
      (``grad_accum_views``), then zero raw grads.
    - Perturb: per-expert Triton kernel using GLOBAL ``grad_offsets[i]``.
    - Backward hook: scale raw grad by ``z * sign / (2 eps K)`` before it
      accumulates into the caller-provided ``grad_flat`` view.
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

    _tsum = {"perturb": 0.0, "hook": 0.0, "bwd": 0.0}
    try:
        with (ctx_mgr if ctx_mgr is not None else nullcontext()):
            if ctx_mgr is not None:
                try:
                    ctx_mgr.capture(forward_loss_fn)
                except Exception as exc:
                    print(f"[proposal] finite_diff: routing-freeze capture failed: {exc}",
                          flush=True)
                    return None
            _sync()
            _t_loop = _time.perf_counter()
            for k in range(max(1, int(k_samples))):
                for sign in (+1.0, -1.0):
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
                    sign_scale = sign * inv_step * inv_k
                    for hook_i, hook_param in enumerate(expert_ps):
                        hook_global_offset = int(grad_offsets[hook_i])
                        hook_target = grad_accum_views[hook_i]

                        def _scale_hook(
                            raw_grad,
                            *,
                            _global_offset=hook_global_offset,
                            _target=hook_target,
                            _sign_scale=sign_scale,
                            _k=int(k),
                        ):
                            if _target.dtype == torch.float32:
                                return _accumulate_grad_to_fp32_with_pcg_z(
                                    raw_grad,
                                    _target,
                                    sign_scale=_sign_scale,
                                    global_offset=_global_offset,
                                    rng_seed=int(rng_seed),
                                    k=_k,
                                )
                            return _scale_grad_inplace_with_pcg_z(
                                raw_grad,
                                sign_scale=_sign_scale,
                                global_offset=_global_offset,
                                rng_seed=int(rng_seed),
                                k=_k,
                            )

                        handles.append(hook_param.register_hook(_scale_hook))
                    _sync(); _t_hook = _time.perf_counter()
                    try:
                        if not _backward_inplace(forward_loss_fn, expert_ps,
                                                  model_for_full_zero,
                                                  preserve_expert_grad_storage=preserve_param_grad_storage):
                            return None
                    finally:
                        for handle in handles:
                            handle.remove()
                        if not preserve_param_grad_storage:
                            for p in expert_ps:
                                p.grad = None
                    _sync(); _t2 = _time.perf_counter()
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
    # grad_flat or grad_accum_views carry the signed running sum across the multi-batch loop.
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
    grad_flat: Optional[torch.Tensor] = None,
    grad_offsets: Optional[List[int]] = None,
    grad_accum_views: Optional[List[torch.Tensor]] = None,
) -> Optional[List[torch.Tensor]]:
    """Accumulate one finite-difference batch into caller-owned ``grad_flat``.

    The anchor caller provides ``grad_flat`` plus per-expert offsets and a
    theta snapshot. Hooks scale each raw grad by deterministic PCG ``z`` before
    PyTorch accumulates into ``grad_flat``. The removed per-tensor group
    accumulator path is intentionally not available.
    """
    if not expert_ps or eps <= 0:
        return None
    # Fast-flat dispatch: caller provides (grad_flat, grad_offsets,
    # theta_snap_ref). lp stays per-tensor (no constraint on DS optimizer
    # layout); grad gets a contiguous flat backing and hooks scale raw grads
    # before they accumulate into it.
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
            grad_flat=grad_flat,
            grad_offsets=grad_offsets,
            restore_src_list=theta_snap_ref,
            grad_accum_views=grad_accum_views,
        )

    raise RuntimeError(
        "H-quantile anchor refresh requires resident grad_flat, grad_offsets, "
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
    grad_flat: Optional[torch.Tensor] = None,
    grad_offsets: Optional[List[int]] = None,
    grad_accum_views: Optional[List[torch.Tensor]] = None,
) -> Optional[Any]:
    """Run one central finite-difference batch into caller-owned ``grad_flat``.

    The public compatibility wrapper still builds the cached batch and routing
    freeze context. Actual H-quantile construction happens outside this module:
    the caller averages ``grad_flat``, takes ``abs()``, and refreshes ``h_q``.
    """
    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    m0 = model[0] if isinstance(model, (list, tuple)) else model
    if not expert_ps:
        return None

    # ---- Load batch ONCE (Common Random Numbers). ----
    # Capture (tokens, labels, mask, attn, pos) once via get_batch then
    # run a custom forward that bypasses forward_step_func's iterator
    # advance. Caller may also pre-capture and inject via batch_override.
    if batch_override is not None:
        batch = batch_override
    else:
        try:
            from pretrain_gpt import get_batch as _get_batch_gpt
        except Exception as exc:
            print(f"[proposal] finite_diff: cannot import get_batch: {exc}", flush=True)
            return None
        try:
            batch = _get_batch_gpt(di)
        except Exception as exc:
            print(f"[proposal] finite_diff: get_batch failed: {exc}", flush=True)
            return None
    tokens, labels, loss_mask, attention_mask, position_ids = batch[:5]

    # Build a closure that re-runs the model forward + LM loss on the
    # cached batch. Mirrors Megatron's training forward.
    import torch.nn.functional as F

    def _forward_loss() -> Tuple[torch.Tensor, Dict[str, Any]]:
        out = m0(tokens, position_ids, attention_mask)
        logits = out[0] if isinstance(out, tuple) else out
        if logits is None or not torch.is_tensor(logits):
            raise RuntimeError("finite_diff: model returned non-tensor")
        if logits.dim() >= 2:
            V = logits.size(-1)
            flat_logits = logits.reshape(-1, V)
        else:
            flat_logits = logits
        flat_labels = labels.reshape(-1) if labels is not None else None
        if flat_labels is None:
            raise RuntimeError("finite_diff: labels missing")
        if loss_mask is not None and loss_mask.numel() == flat_logits.size(0):
            per_tok = F.cross_entropy(flat_logits.float(), flat_labels, reduction="none")
            denom = loss_mask.view(-1).float().sum().clamp_min(1.0)
            loss = (per_tok * loss_mask.view(-1).float()).sum() / denom
        else:
            loss = F.cross_entropy(flat_logits.float(), flat_labels, reduction="mean")
        return loss, {}

    # ---- Routing freeze setup ----
    from .routing_freeze import identify_gate_modules_safe
    gates_for_freeze: Optional[List[Tuple[str, torch.nn.Module]]] = None
    if use_routing_freeze:
        gates, _rules = identify_gate_modules_safe(model)
        if not gates:
            print("[proposal] finite_diff: WARN no gates identified; running "
                  "without routing freeze", flush=True)
        else:
            gates_for_freeze = gates

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
        grad_flat=grad_flat,
        grad_offsets=grad_offsets,
        grad_accum_views=grad_accum_views,
    )
