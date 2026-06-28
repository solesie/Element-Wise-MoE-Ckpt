"""Proposal: element-wise importance MoE checkpointing for Megatron-DeepSpeed."""
from .config import ProposalConfig  # noqa: F401
from .telemetry.diagnostics import HRefreshStats, IterCheckpointStats  # noqa: F401
from .hooks import (  # noqa: F401
    init_proposal,
    proposal_dispatch_post_iter,
    proposal_wait_d2h_before_optimizer,
    proposal_record_loss,
    proposal_run_independent_oracle,
    proposal_vram_audit_iter_begin,
    proposal_vram_audit_iter_end,
    proposal_refresh_h_quantile,
    proposal_set_anchor_data_iter,
    proposal_simulate_fault,
    proposal_recover,
    proposal_shutdown,
)
from .cli_args import add_proposal_args  # noqa: F401

__all__ = [
    "ProposalConfig",
    "IterCheckpointStats",
    "HRefreshStats",
    "init_proposal",
    "proposal_dispatch_post_iter",
    "proposal_wait_d2h_before_optimizer",
    "proposal_record_loss",
    "proposal_run_independent_oracle",
    "proposal_vram_audit_iter_begin",
    "proposal_vram_audit_iter_end",
    "proposal_refresh_h_quantile",
    "proposal_set_anchor_data_iter",
    "proposal_simulate_fault",
    "proposal_recover",
    "proposal_shutdown",
    "add_proposal_args",
]
