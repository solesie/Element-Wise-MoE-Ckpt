"""Thin training-loop hooks.

Each function delegates to ``ProposalController`` so the training loop imports
one stable proposal checkpointing surface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .controller import get_controller


def init_proposal(model: Any, optimizer: Any, args: Any, cp_root: str) -> None:
    get_controller().init(model, optimizer, args, cp_root)


def proposal_activate_after_warmup(model: Any, iteration: int) -> bool:
    return get_controller().activate_after_warmup(model, int(iteration))


def proposal_prepare_after_optimizer() -> None:
    get_controller().prepare_after_optimizer()


def proposal_dispatch_post_iter(
    model: Any,
    iter_T: int,
    *,
    sync_processes: bool = True,
    defer_stage0_emission: bool = False,
) -> None:
    get_controller().dispatch_post_step_task(
        model,
        iter_T,
        sync_processes=bool(sync_processes),
        defer_stage0_emission=bool(defer_stage0_emission),
    )


def proposal_finalize_post_iter(model: Any, iteration: int) -> None:
    get_controller().finalize_post_step(model, int(iteration))


def proposal_wait_d2h_before_optimizer() -> None:
    get_controller().wait_d2h_before_optimizer()


def proposal_record_loss(step: int, loss: Optional[float]) -> None:
    get_controller().record_loss(step, loss)


def proposal_run_independent_oracle(model: Any, data_iterator: Any, iteration: int) -> None:
    get_controller().run_independent_oracle(model, data_iterator, iteration)


def proposal_vram_audit_iter_begin(step: int) -> None:
    get_controller().vram_audit_iter_begin(step)


def proposal_vram_audit_iter_end(step: int) -> None:
    get_controller().vram_audit_iter_end(step)


def proposal_vram_audit_dispatch_post_iter(model: Any, iter_T: int) -> bool:
    return get_controller().vram_audit_dispatch_post_iter(model, int(iter_T))


def proposal_refresh_h_quantile(*args: Any, **kwargs: Any) -> Any:
    return get_controller().refresh_h_quantile(*args, **kwargs)


def proposal_set_anchor_data_iter(it: Any) -> None:
    get_controller().set_anchor_data_iter(it)


def proposal_set_oracle_data_iter(it: Any) -> None:
    get_controller().set_oracle_data_iter(it)


def proposal_simulate_fault(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
    return get_controller().simulate_fault(*args, **kwargs)


def proposal_recover(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
    return get_controller().recover(*args, **kwargs)


def proposal_shutdown() -> Dict[str, Any]:
    return get_controller().shutdown()


__all__ = [
    "init_proposal",
    "proposal_activate_after_warmup",
    "proposal_prepare_after_optimizer",
    "proposal_dispatch_post_iter",
    "proposal_finalize_post_iter",
    "proposal_wait_d2h_before_optimizer",
    "proposal_record_loss",
    "proposal_run_independent_oracle",
    "proposal_vram_audit_iter_begin",
    "proposal_vram_audit_iter_end",
    "proposal_vram_audit_dispatch_post_iter",
    "proposal_refresh_h_quantile",
    "proposal_set_anchor_data_iter",
    "proposal_set_oracle_data_iter",
    "proposal_simulate_fault",
    "proposal_recover",
    "proposal_shutdown",
]
