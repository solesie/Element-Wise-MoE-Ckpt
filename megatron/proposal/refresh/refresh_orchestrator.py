"""Refresh-boundary orchestration used by LossyCheckpointPipeline."""

from __future__ import annotations

import math
import os
from typing import List, Optional

import torch

try:
    import triton
    _HAS_TRITON_REFRESH = True
except Exception:
    _HAS_TRITON_REFRESH = False


class RefreshOrchestrationMixin:
    def _device_all_reduce_if_needed(self, tensor: torch.Tensor, op) -> None:
        pg = getattr(self, "_proposal_pg", None)
        world = int(getattr(self, "_proposal_world_size", 1))
        if (
            world > 1
            and pg is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(tensor, op=op, group=pg)

    def _quantile_operating_band_slice(self, n: int) -> slice:
        width = max(1, int(math.ceil(float(n) * float(self.topk_ratio))))
        start = max(0, int(n) - int(width))
        return slice(start, int(n))

    def _compute_h_refresh_bridge_scale(
        self,
        old_h_q: torch.Tensor,
        new_h_q: torch.Tensor,
    ) -> Optional[float]:
        if old_h_q.numel() <= 0 or old_h_q.numel() != new_h_q.numel():
            return None
        band = self._quantile_operating_band_slice(int(old_h_q.numel()))
        old = old_h_q.detach().to(torch.float64).cpu()[band]
        new = new_h_q.detach().to(torch.float64).cpu()[band]
        mask = torch.isfinite(old) & torch.isfinite(new) & (old > 0.0) & (new > 0.0)
        if int(mask.sum().item()) <= 0:
            return None
        log_ratio = torch.log(new[mask] / old[mask])
        log_ratio = log_ratio[torch.isfinite(log_ratio)]
        if int(log_ratio.numel()) <= 0:
            return None
        scale = float(math.exp(float(torch.median(log_ratio).item())))
        return scale if math.isfinite(scale) and scale > 0.0 else None

    def _interp_h_for_v_host(
        self,
        v_points: torch.Tensor,
        v_q: torch.Tensor,
        h_q: torch.Tensor,
    ) -> torch.Tensor:
        v = v_q.detach().to(torch.float64).cpu().clamp_min(1.0e-30)
        h = h_q.detach().to(torch.float64).cpu().clamp_min(1.0e-30)
        x = v_points.detach().to(torch.float64).cpu().clamp_min(1.0e-30)
        if v.numel() != h.numel() or v.numel() <= 1:
            return torch.empty(0, dtype=torch.float64)
        order = torch.argsort(v)
        v = v[order]
        h = h[order]
        v = torch.cummax(v, dim=0).values
        idx = torch.searchsorted(v, x, right=False)
        idx = torch.clamp(idx, 1, int(v.numel()) - 1)
        v_lo = v[idx - 1]
        v_hi = v[idx]
        h_lo = h[idx - 1]
        h_hi = h[idx]
        denom = torch.clamp(v_hi - v_lo, min=1.0e-30)
        t = torch.clamp((x - v_lo) / denom, 0.0, 1.0)
        return h_lo + t * (h_hi - h_lo)

    def _compute_v_refresh_bridge_scale(
        self,
        old_v_q: torch.Tensor,
        new_v_q: torch.Tensor,
        h_q: torch.Tensor,
    ) -> Optional[float]:
        if old_v_q.numel() <= 1 or old_v_q.numel() != new_v_q.numel():
            return None
        if h_q.numel() != old_v_q.numel():
            return None
        band = self._quantile_operating_band_slice(int(new_v_q.numel()))
        v_points = new_v_q.detach()[band]
        new_h = self._interp_h_for_v_host(v_points, new_v_q, h_q)
        old_h = self._interp_h_for_v_host(v_points, old_v_q, h_q)
        if new_h.numel() <= 0 or old_h.numel() != new_h.numel():
            return None
        mask = (
            torch.isfinite(old_h)
            & torch.isfinite(new_h)
            & (old_h > 0.0)
            & (new_h > 0.0)
        )
        if int(mask.sum().item()) <= 0:
            return None
        log_ratio = torch.log(new_h[mask] / old_h[mask])
        log_ratio = log_ratio[torch.isfinite(log_ratio)]
        if int(log_ratio.numel()) <= 0:
            return None
        scale = float(torch.exp(log_ratio.mean()).item())
        return scale if math.isfinite(scale) and scale > 0.0 else None

    def _bridge_adaptive_refresh_scale(
        self,
        *,
        scale: Optional[float],
        reason: str,
        refresh_step: Optional[int] = None,
        apply: bool = True,
    ) -> None:
        if scale is None or not math.isfinite(float(scale)) or float(scale) <= 0.0:
            return
        applied = (
            self._adaptive_cutoff.bridge_refresh_scale(float(scale))
            if bool(apply) else False
        )
        if applied:
            self._last_refresh_bridge_scale = float(scale)
            self._last_refresh_bridge_reason = str(reason)

    def update_h_anchor_quantiles(self, h_flat=None, refresh_step: Optional[int] = None) -> None:
        """Extract the N_Q-point H quantile table from the anchor FD output.

        ``h_flat`` is one tensor or a list of per-expert tensors (|H| estimate).
        The log-hist is built across all of them (atomic min/max + atomic_add),
        so no contiguous concatenation of the per-expert grads is required.
        """

        if self.selection_mode == "v_hat":
            self._h_quantile_values.copy_(self._v_quantile_values)
            self._h_anchor_initialised = True
            self._ensure_h_lookup_lut()
            from ..refresh.quantile import launch_rebuild_h_lookup_lut
            launch_rebuild_h_lookup_lut(
                self._v_quantile_values,
                self._h_quantile_values,
                self._h_lookup_lut,
            )
            return
        if h_flat is None:
            return
        if not getattr(self, "_flat_layout_done", False):
            return
        runtime_state = self.lossy_cp_state
        if runtime_state is not None:
            cpu_pg = getattr(self, "_proposal_cpu_pg", None)
            total_elems_global = self._global_selection_element_count_cached(
                int(runtime_state.index.n_total_elements),
                process_group=cpu_pg,
                world_size=int(getattr(self, "_proposal_world_size", 1)),
            )
            self._consume_direct_emission_feedback(
                total_elems_global=int(total_elems_global),
                world_size=int(getattr(self, "_proposal_world_size", 1)),
                process_group=cpu_pg,
            )
        old_h_q = (
            self._h_quantile_values.detach().clone()
            if getattr(self, "_h_anchor_initialised", False) else None
        )
        srcs = h_flat if isinstance(h_flat, (list, tuple)) else [h_flat]
        flats = [s.detach().reshape(-1) for s in srcs
                 if s is not None and s.numel() > 0]
        if not flats:
            return
        n_bins = min(int(self.n_bins_quantile_log), int(self._selection_hist.numel()))
        if n_bins <= 1:
            return
        from ..refresh.quantile import (
            launch_build_log_hist_from_bounds,
            launch_fill_quantiles_from_log_hist,
            launch_rebuild_h_lookup_lut,
            launch_reduce_positive_minmax,
        )
        hist = self._selection_hist.narrow(0, 0, n_bins)
        h_lo_t = self._selection_smin
        h_hi_t = self._selection_smax
        h_lo_t.fill_(float("inf"))
        h_hi_t.zero_()
        for flat in flats:
            launch_reduce_positive_minmax(flat, h_lo_t, h_hi_t)
        self._device_all_reduce_if_needed(h_lo_t, torch.distributed.ReduceOp.MIN)
        self._device_all_reduce_if_needed(h_hi_t, torch.distributed.ReduceOp.MAX)
        hist.zero_()
        for flat in flats:
            launch_build_log_hist_from_bounds(
                flat, hist, min_bound=h_lo_t, max_bound=h_hi_t,
            )
        self._device_all_reduce_if_needed(hist, torch.distributed.ReduceOp.SUM)
        self._oracle_window_h_hist_metadata = {
            "positive": int(hist.sum().item()),
            "minimum": float(h_lo_t.item()),
            "maximum": float(h_hi_t.item()),
        }
        launch_fill_quantiles_from_log_hist(
            hist, self._h_quantile_values, min_bound=h_lo_t, max_bound=h_hi_t,
        )
        self._last_h_quantile_summary = (
            float(self._h_quantile_values.mean().item()),
            float(self._h_quantile_values.max().item()),
        )
        self._h_anchor_initialised = True
        self._ensure_h_lookup_lut()
        launch_rebuild_h_lookup_lut(
            self._v_quantile_values,
            self._h_quantile_values,
            self._h_lookup_lut,
        )
        if old_h_q is not None:
            bridge_scale = self._compute_h_refresh_bridge_scale(
                old_h_q,
                self._h_quantile_values,
            )
            self._bridge_adaptive_refresh_scale(
                scale=bridge_scale,
                reason="h_refresh",
                refresh_step=refresh_step,
                apply=True,
            )

    def update_v_mapping(self, global_step: Optional[int] = None) -> None:
        """Refresh the runtime v quantile table with log-bin histograms.

        This is an anchor-boundary operation. Production hessian mode calls
        it immediately after H refresh, before training resumes. It is not a
        per-iteration Stage 0 and has no independent cadence knob.

        The hot path reuses the Stage 1 histogram scratch; no sampled sort
        path remains.
        """
        if not _HAS_TRITON_REFRESH:
            return
        if self.selection_mode != "v_hat" and not getattr(self, "_h_anchor_initialised", False):
            return
        if not getattr(self, "_flat_layout_done", False):
            return
        # 1-iter lag GPU timing — same pattern as selection timing.
        _vmap_ev0 = _vmap_ev1 = None
        import os as _os_vmap_timing
        _vmap_timing = (
            _os_vmap_timing.environ.get("PROPOSAL_VMAP_TIMING", "")
            in ("1", "true", "TRUE")
        )
        if _vmap_timing and torch.cuda.is_available():
            _prev = getattr(self, "_vmap_ev_prev", None)
            if _prev is not None:
                try:
                    _prev[1].synchronize()
                    if not hasattr(self, "last_phase_ms") or self.last_phase_ms is None:
                        self.last_phase_ms = {}
                    self.last_phase_ms["v_mapping"] = float(_prev[0].elapsed_time(_prev[1]))
                except Exception:
                    pass
            _vmap_ev0 = self._acquire_timing_event("vmap0")
            _vmap_ev1 = self._acquire_timing_event("vmap1")
            _vmap_ev0.record()

        # Sub-phase timers (cuda.Event pairs, 1-iter lag) split host-side
        # quantile extraction from histogram kernel launch time.
        _qe_ev0 = _qk_ev0 = _qe_ev1 = _qk_ev1 = None
        if _vmap_timing and torch.cuda.is_available():
            _prev_sub = getattr(self, "_vmap_sub_ev_prev", None)
            if _prev_sub is not None:
                try:
                    qe0, qe1, qk0, qk1 = _prev_sub
                    qe1.synchronize(); qk1.synchronize()
                    if not hasattr(self, "last_phase_ms") or self.last_phase_ms is None:
                        self.last_phase_ms = {}
                    self.last_phase_ms["v_q_extract"] = float(qe0.elapsed_time(qe1))
                    self.last_phase_ms["v_q_kernel"] = float(qk0.elapsed_time(qk1))
                except Exception:
                    pass
            _qe_ev0 = self._acquire_timing_event("vq_extract0")
            _qe_ev1 = self._acquire_timing_event("vq_extract1")
            _qk_ev0 = self._acquire_timing_event("vq_kernel0")
            _qk_ev1 = self._acquire_timing_event("vq_kernel1")
            _qe_ev0.record()

        # v_t quantile table. This scans each BF16-Opt v_flat into the
        # shared log histogram and extracts quantiles from the small
        # histogram while training is halted for the anchor boundary.
        v_flats: List[torch.Tensor] = []
        for group_dict in self._groups:
            v_flat = group_dict.get("v_flat")
            if v_flat is None:
                return
            v_flats.append(v_flat)
        n_bins = min(int(self.n_bins_quantile_log), int(self._selection_hist.numel()))
        if n_bins <= 1:
            return
        hist = self._selection_hist.narrow(0, 0, n_bins)
        can_bridge_v_refresh = self._last_vq_refresh_step is not None
        old_v_q = self._v_quantile_values.detach().clone()
        old_h_q = self._h_quantile_values.detach().clone()

        v_lo_t = self._selection_smin
        v_hi_t = self._selection_smax
        v_lo_t.fill_(float("inf"))
        v_hi_t.zero_()
        from ..refresh.quantile import (
            launch_build_log_hist_from_bounds,
            launch_fill_quantiles_from_log_hist,
            launch_rebuild_h_lookup_lut,
            launch_reduce_positive_minmax,
        )
        for v in v_flats:
            launch_reduce_positive_minmax(v.detach().reshape(-1), v_lo_t, v_hi_t)
        self._device_all_reduce_if_needed(v_lo_t, torch.distributed.ReduceOp.MIN)
        self._device_all_reduce_if_needed(v_hi_t, torch.distributed.ReduceOp.MAX)
        hist.zero_()
        for v in v_flats:
            launch_build_log_hist_from_bounds(
                v.detach().reshape(-1),
                hist,
                min_bound=v_lo_t,
                max_bound=v_hi_t,
            )
        self._device_all_reduce_if_needed(hist, torch.distributed.ReduceOp.SUM)
        self._oracle_window_v_hist_metadata = {
            "positive": int(hist.sum().item()),
            "minimum": float(v_lo_t.item()),
            "maximum": float(v_hi_t.item()),
        }
        launch_fill_quantiles_from_log_hist(
            hist,
            self._v_quantile_values,
            min_bound=v_lo_t,
            max_bound=v_hi_t,
        )
        if self.selection_mode == "v_hat":
            self._h_quantile_values.copy_(self._v_quantile_values)
            self._h_anchor_initialised = True
        self._ensure_h_lookup_lut()
        launch_rebuild_h_lookup_lut(
            self._v_quantile_values,
            self._h_quantile_values,
            self._h_lookup_lut,
        )
        if self.selection_mode != "v_hat" and can_bridge_v_refresh:
            self._bridge_adaptive_refresh_scale(
                scale=self._compute_v_refresh_bridge_scale(
                    old_v_q,
                    self._v_quantile_values,
                    old_h_q,
                ),
                reason="v_refresh",
                refresh_step=global_step,
            )
        if global_step is not None:
            self._last_vq_refresh_step = int(global_step)
        if _qe_ev1 is not None:
            _qe_ev1.record()
            _qk_ev0.record()

        # Selection/write kernels read the prebuilt log(v)->H LUT.

        if _qk_ev1 is not None:
            _qk_ev1.record()
            self._vmap_sub_ev_prev = (_qe_ev0, _qe_ev1, _qk_ev0, _qk_ev1)
        if _vmap_ev1 is not None:
            _vmap_ev1.record()
            self._vmap_ev_prev = (_vmap_ev0, _vmap_ev1)

    # ------------------------------------------------------------------
