"""Typed configuration for the proposal checkpointing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_LFCL_INTERVAL_ITERS = 1
DEFAULT_LFCL_FRACTION_PCT = 1.0
DEFAULT_LFCL_PLAN_POLICY = "lossy_log_size"


def _ceil_div(num: int, den: int) -> int:
    return (max(0, int(num)) + max(1, int(den)) - 1) // max(1, int(den))


def lr_warmup_iters_from_args(args) -> int:
    warmup_tokens = getattr(args, "lr_warmup_tokens", None)
    if warmup_tokens is not None and int(warmup_tokens or 0) > 0:
        tokens_per_iter = (
            max(1, int(getattr(args, "global_batch_size", 1) or 1))
            * max(1, int(getattr(args, "seq_length", 1) or 1))
        )
        return _ceil_div(int(warmup_tokens), tokens_per_iter)

    warmup_iters = int(getattr(args, "lr_warmup_iters", 0) or 0)
    if warmup_iters > 0:
        return warmup_iters

    warmup_samples = int(getattr(args, "lr_warmup_samples", 0) or 0)
    if warmup_samples > 0:
        return _ceil_div(
            warmup_samples,
            max(1, int(getattr(args, "global_batch_size", 1) or 1)),
        )

    warmup_fraction = getattr(args, "lr_warmup_fraction", None)
    if warmup_fraction is not None:
        decay_iters = int(
            getattr(args, "lr_decay_iters", None)
            or getattr(args, "train_iters", 0)
            or 0
        )
        return int(math.ceil(float(warmup_fraction) * max(0, decay_iters)))

    return 0


@dataclass(frozen=True)
class ProposalConfig:
    """All CLI-derived proposal parameters in one immutable object."""

    # Selection.
    topk_ratio: float
    group_size: int
    selection_mode: str
    score_policy: str
    score_linear_source: str
    e_hat_proxy: bool = True

    # Selection cutoff search.
    n_bins_cutoff: int = 4096
    cutoff_K_tolerance: float = 0.04
    adaptive_cutoff_window: int = 16
    adaptive_cutoff_min_observations: int = 8
    adaptive_cutoff_warmup_iters: int = 0
    adaptive_cutoff_skip_warmup_iters: int = 0
    adaptive_cutoff_retry_tolerance: float = 0.04
    # Quantile tables.
    n_q: int = 4096
    n_bins_quantile_log: int = 1024
    h_lookup_lut_bins: int = 8192

    # Anchor.
    finite_diff_eps: float = 1e-3
    h_refresh_every: int = 50
    h_refresh_warmup_every: int = 20
    h_refresh_warmup_iters: int = 0
    n_anchor_batches: Optional[int] = None

    # FC schedule. Expert LFCL is fixed by V2: every iter, 1% of expert slots.
    lfcl_interval_iters: int = DEFAULT_LFCL_INTERVAL_ITERS
    lfcl_fraction_pct: float = DEFAULT_LFCL_FRACTION_PCT
    lfcl_plan_policy: str = DEFAULT_LFCL_PLAN_POLICY
    nonexpert_fc_w: int = 1

    # Storage.
    cp_mode: str = "proposal"
    cp_target: str = "disk"
    cp_dir: Path = Path("./cp")
    pinned_aos_capacity_tolerance: float = 0.75

    # Worker pool.
    async_queue: int = 5

    # Optional experiment knobs.
    fault_interval: Optional[int] = None
    fault_warmup: int = 0
    fault_log: Optional[Path] = None
    test_expert_selection_only: bool = False

    # Telemetry-only independent oracle. This is intentionally outside the
    # production selection/write path.
    independent_oracle_enable: bool = False
    independent_oracle_iters: Tuple[int, ...] = ()
    independent_oracle_log_dir: Optional[Path] = None
    independent_oracle_rank_sample: int = 1048576
    independent_oracle_rank_sample_seed: int = 20240625
    independent_oracle_actual_batches: int = 4
    independent_oracle_hvp_batches: int = 4
    independent_oracle_batch_seed: int = 20240625

    @classmethod
    def from_args(cls, args, cp_root: str) -> "ProposalConfig":
        """Build config from Megatron argparse namespace."""

        selection_mode = getattr(args, "proposal_selection_mode", "hessian")
        if selection_mode not in ("hessian", "v_hat", "random"):
            raise ValueError(
                "proposal_selection_mode must be one of "
                f"'hessian', 'v_hat', 'random'; got {selection_mode!r}"
            )
        score_policy = getattr(
            args,
            "proposal_score_policy",
            "abs_h_taylor_first_and_second_order",
        )
        valid_score_policies = (
            "abs_h_taylor_first_and_second_order",
            "abs_h_taylor_only_second_order",
        )
        if score_policy not in valid_score_policies:
            raise ValueError(
                "proposal_score_policy must be one of "
                f"{valid_score_policies}; got {score_policy!r}"
            )
        score_linear_source = getattr(args, "proposal_score_linear_source", "grad")
        if score_linear_source not in ("grad", "momentum"):
            raise ValueError(
                "proposal_score_linear_source must be 'grad' or 'momentum'; "
                f"got {score_linear_source!r}"
            )
        cp_mode = getattr(args, "proposal_cp_mode", "proposal")
        if cp_mode not in ("proposal", "sequential"):
            raise ValueError(
                f"proposal_cp_mode must be 'proposal'|'sequential', got {cp_mode!r}"
            )
        cp_target = getattr(args, "proposal_cp_target", "disk")
        if cp_target not in ("disk", "test_dram"):
            raise ValueError(
                "proposal_cp_target must be 'disk' or 'test_dram', "
                f"got {cp_target!r}"
            )
        lfcl_plan_policy = getattr(
            args,
            "proposal_lfcl_plan_policy",
            DEFAULT_LFCL_PLAN_POLICY,
        )
        if lfcl_plan_policy not in ("round_robin", "lossy_log_size"):
            raise ValueError(
                "proposal_lfcl_plan_policy must be "
                f"'round_robin'|'lossy_log_size', got {lfcl_plan_policy!r}"
            )

        finite_diff_eps = getattr(args, "proposal_finite_diff_eps", None)
        if finite_diff_eps is None:
            finite_diff_eps = 1e-3
        h_refresh_every = int(
            getattr(args, "proposal_hessian_refresh_every", 50) or 0
        )
        h_refresh_warmup_every = int(
            getattr(args, "proposal_hessian_refresh_warmup_every", 20) or 0
        )

        fault_interval_raw = int(getattr(args, "proposal_fault_interval", 0) or 0)
        fault_log_raw = getattr(args, "proposal_fault_log", None)

        topk_ratio = float(getattr(args, "proposal_topk_ratio", 0.04))
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError(
                f"proposal_topk_ratio must satisfy 0 < topk_ratio <= 1, got {topk_ratio!r}"
            )
        adaptive_warmup_iters = getattr(
            args,
            "proposal_adaptive_cutoff_warmup_iters",
            0,
        )
        skip_warmup_iters = lr_warmup_iters_from_args(args)

        # Anchor processes half of one iteration's micro-batches (mb_per_iter/2):
        # empirically ≥0.93 H coverage while under half the fwd+bwd budget.
        _mb_per_iter = max(
            1, int(getattr(args, "global_batch_size", 1))
            // max(1, int(getattr(args, "micro_batch_size", 1)))
        )
        n_anchor_batches = max(1, _mb_per_iter // 2)

        def _parse_int_tuple(raw) -> Tuple[int, ...]:
            if raw is None:
                return ()
            if isinstance(raw, (list, tuple)):
                vals = raw
            else:
                vals = str(raw).replace(";", ",").split(",")
            out = []
            for item in vals:
                text = str(item).strip()
                if text:
                    out.append(int(text))
            return tuple(out)

        independent_oracle_log_dir_raw = getattr(
            args,
            "proposal_independent_oracle_log_dir",
            None,
        )

        return cls(
            topk_ratio=topk_ratio,
            group_size=max(1, int(getattr(args, "proposal_block_quant_g", 8))),
            selection_mode=str(selection_mode),
            score_policy=str(score_policy),
            score_linear_source=str(score_linear_source),
            e_hat_proxy=not bool(
                getattr(args, "proposal_e_hat_proxy_off", False)
            ),
            n_bins_cutoff=int(getattr(args, "proposal_n_bins_cutoff", 4096)),
            cutoff_K_tolerance=float(
                getattr(args, "proposal_cutoff_k_tolerance", 0.04)
            ),
            adaptive_cutoff_window=max(
                2, int(getattr(args, "proposal_adaptive_cutoff_window", 16) or 16)
            ),
            adaptive_cutoff_min_observations=max(
                1,
                int(
                    getattr(
                        args,
                        "proposal_adaptive_cutoff_min_observations",
                        8,
                    ) or 8
                ),
            ),
            adaptive_cutoff_warmup_iters=max(
                0,
                int(adaptive_warmup_iters or 0),
            ),
            adaptive_cutoff_skip_warmup_iters=max(
                0,
                int(skip_warmup_iters or 0),
            ),
            adaptive_cutoff_retry_tolerance=max(
                0.0,
                float(
                    getattr(
                        args,
                        "proposal_adaptive_cutoff_retry_tolerance",
                        0.04,
                    ) or 0.04
                ),
            ),
            n_q=int(getattr(args, "proposal_n_q", 4096)),
            h_lookup_lut_bins=int(
                getattr(args, "proposal_h_lookup_lut_bins", 8192)
            ),
            finite_diff_eps=float(finite_diff_eps),
            h_refresh_every=h_refresh_every,
            h_refresh_warmup_every=max(0, h_refresh_warmup_every),
            h_refresh_warmup_iters=max(0, int(skip_warmup_iters or 0)),
            n_anchor_batches=n_anchor_batches,
            lfcl_interval_iters=DEFAULT_LFCL_INTERVAL_ITERS,
            lfcl_fraction_pct=DEFAULT_LFCL_FRACTION_PCT,
            lfcl_plan_policy=str(lfcl_plan_policy),
            nonexpert_fc_w=int(getattr(args, "proposal_nonexpert_fc_W", 1)),
            cp_mode=str(cp_mode),
            cp_target=str(cp_target),
            cp_dir=Path(cp_root),
            pinned_aos_capacity_tolerance=max(
                0.0,
                float(
                    getattr(
                        args,
                        "proposal_pinned_aos_capacity_tolerance",
                        0.75,
                    ) or 0.75
                ),
            ),
            async_queue=int(getattr(args, "proposal_async_queue", 5)),
            fault_interval=(fault_interval_raw or None),
            fault_warmup=int(getattr(args, "proposal_fault_warmup", 0) or 0),
            fault_log=Path(fault_log_raw) if fault_log_raw else None,
            test_expert_selection_only=bool(
                getattr(args, "proposal_test_expert_selection_only", False)
            ),
            independent_oracle_enable=bool(
                getattr(args, "proposal_independent_oracle_enable", False)
            ),
            independent_oracle_iters=_parse_int_tuple(
                getattr(args, "proposal_independent_oracle_iters", "")
            ),
            independent_oracle_log_dir=(
                Path(independent_oracle_log_dir_raw)
                if independent_oracle_log_dir_raw else None
            ),
            independent_oracle_rank_sample=max(
                0,
                int(
                    getattr(
                        args,
                        "proposal_independent_oracle_rank_sample",
                        1048576,
                    ) or 0
                ),
            ),
            independent_oracle_rank_sample_seed=int(
                getattr(
                    args,
                    "proposal_independent_oracle_rank_sample_seed",
                    20240625,
                ) or 20240625
            ),
            independent_oracle_actual_batches=max(
                1,
                int(
                    getattr(
                        args,
                        "proposal_independent_oracle_actual_batches",
                        4,
                    ) or 4
                ),
            ),
            independent_oracle_hvp_batches=max(
                0,
                int(
                    getattr(
                        args,
                        "proposal_independent_oracle_hvp_batches",
                        4,
                    ) or 0
                ),
            ),
            independent_oracle_batch_seed=int(
                getattr(
                    args,
                    "proposal_independent_oracle_batch_seed",
                    20240625,
                ) or 20240625
            ),
        )
