"""Short-lived gradient leases for proposal scoring.

The proposal score can use gradients that DeepSpeed has already materialized
for the optimizer step.  This module deliberately does not clone or cache those
tensors; it only keeps references alive until proposal kernels have finished
reading them.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

import torch


class GradScoreLease:
    """Lease DeepSpeed optimizer grad flats for one proposal step."""

    def __init__(self, optimizer: object) -> None:
        self._optimizer = optimizer
        self._installed = False
        self._orig_optimizer_step = None
        self._orig_clear_hp_grads = None
        self._orig_update_hp_grads = None
        self._leases: Dict[int, torch.Tensor] = {}
        self._mode = ""
        self._bf16_clear_deferred = False
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self._installed

    @property
    def has_leases(self) -> bool:
        with self._lock:
            return bool(self._leases)

    def install(self) -> bool:
        """Wrap the active DeepSpeed optimizer grad-clear point if available."""

        if self._installed:
            return True
        return self._install_zero_partition_step() or self._install_bf16_grad_flat()

    def _install_zero_partition_step(self) -> bool:
        opt = self._optimizer
        if not hasattr(opt, "single_partition_of_fp32_groups"):
            return False
        orig = getattr(opt, "_optimizer_step", None)
        if orig is None or not callable(orig):
            return False

        lease = self

        def _wrapped_optimizer_step(group_no, *args, **kwargs):
            grad = None
            try:
                groups = getattr(opt, "single_partition_of_fp32_groups", ())
                if 0 <= int(group_no) < len(groups):
                    grad = getattr(groups[int(group_no)], "grad", None)
            except Exception:
                grad = None
            if isinstance(grad, torch.Tensor) and grad.is_cuda:
                with lease._lock:
                    lease._leases[int(group_no)] = grad
            return orig(group_no, *args, **kwargs)

        self._orig_optimizer_step = orig
        setattr(opt, "_optimizer_step", _wrapped_optimizer_step)
        self._installed = True
        self._mode = "zero_partition"
        return True

    def _install_bf16_grad_flat(self) -> bool:
        opt = self._optimizer
        flats = getattr(opt, "fp32_groups_gradients_flat", None)
        if not isinstance(flats, (list, tuple)):
            return False
        orig_clear = getattr(opt, "clear_hp_grads", None)
        orig_update = getattr(opt, "update_hp_grads", None)
        if not callable(orig_clear) or not callable(orig_update):
            return False

        lease = self

        def _wrapped_clear_hp_grads(*args, **kwargs):
            with lease._lock:
                if lease._bf16_clear_deferred:
                    lease._reset_bf16_gradient_flags()
                    return None
            if not lease._capture_bf16_grad_flats():
                return orig_clear(*args, **kwargs)
            return None

        def _wrapped_update_hp_grads(*args, **kwargs):
            lease.release()
            return orig_update(*args, **kwargs)

        self._orig_clear_hp_grads = orig_clear
        self._orig_update_hp_grads = orig_update
        setattr(opt, "clear_hp_grads", _wrapped_clear_hp_grads)
        setattr(opt, "update_hp_grads", _wrapped_update_hp_grads)
        self._installed = True
        self._mode = "bf16_grad_flat"
        return True

    def uninstall(self) -> None:
        if self._installed:
            if self._orig_optimizer_step is not None:
                setattr(self._optimizer, "_optimizer_step", self._orig_optimizer_step)
            if self._orig_clear_hp_grads is not None:
                setattr(self._optimizer, "clear_hp_grads", self._orig_clear_hp_grads)
            if self._orig_update_hp_grads is not None:
                setattr(self._optimizer, "update_hp_grads", self._orig_update_hp_grads)
        self._installed = False
        self._orig_optimizer_step = None
        self._orig_clear_hp_grads = None
        self._orig_update_hp_grads = None
        self._mode = ""
        self.release()

    def view_for_group(
        self,
        group_idx: int,
        offset: int,
        expected_numel: int,
    ) -> Optional[torch.Tensor]:
        with self._lock:
            grad = self._leases.get(int(group_idx))
        if not isinstance(grad, torch.Tensor) or not grad.is_cuda:
            return None
        start = int(offset)
        n = int(expected_numel)
        if start < 0 or n <= 0 or start + n > int(grad.numel()):
            return None
        return grad.detach().reshape(-1).narrow(0, start, n)

    def _bf16_has_gradient_flags(self) -> bool:
        flags = getattr(self._optimizer, "fp32_groups_has_gradients", None)
        if not isinstance(flags, (list, tuple)):
            return True
        for group in flags:
            if any(bool(v) for v in group):
                return True
        return False

    def _reset_bf16_gradient_flags(self) -> None:
        flags = getattr(self._optimizer, "fp32_groups_has_gradients", None)
        grads = getattr(self._optimizer, "fp32_groups_gradients", None)
        if not isinstance(flags, list) or not isinstance(grads, (list, tuple)):
            return
        for idx, group in enumerate(grads):
            if idx < len(flags):
                flags[idx] = [False] * len(group)

    def _capture_bf16_grad_flats(self) -> bool:
        if not self._bf16_has_gradient_flags():
            return False
        flats = getattr(self._optimizer, "fp32_groups_gradients_flat", None)
        if not isinstance(flats, (list, tuple)):
            return False
        captured: Dict[int, torch.Tensor] = {}
        for idx, grad in enumerate(flats):
            if isinstance(grad, torch.Tensor) and grad.is_cuda:
                captured[int(idx)] = grad
        if not captured:
            return False
        with self._lock:
            self._leases.update(captured)
            self._bf16_clear_deferred = True
            self._reset_bf16_gradient_flags()
        return True

    def _clear_deferred_bf16_grads_locked(self) -> None:
        if self._mode != "bf16_grad_flat" or not self._bf16_clear_deferred:
            return
        for grad in self._leases.values():
            grad.zero_()
        devices = {
            int(grad.device.index)
            for grad in self._leases.values()
            if isinstance(grad, torch.Tensor) and grad.is_cuda
        }
        for device_idx in devices:
            torch.cuda.current_stream(device_idx).synchronize()
        self._bf16_clear_deferred = False

    def release(self) -> None:
        with self._lock:
            self._clear_deferred_bf16_grads_locked()
            self._leases.clear()

    def release_after(self, event: Optional["torch.cuda.Event"]) -> None:
        """Release leases after a CUDA event that follows all gradient reads."""

        with self._lock:
            if event is not None:
                event.synchronize()
            self._clear_deferred_bf16_grads_locked()
            self._leases.clear()


__all__ = ["GradScoreLease"]
