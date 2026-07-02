"""CLI argument registration for the MoE element-wise importance
checkpointing pipeline.

Registered via ``pretrain_gpt.py``'s ``extra_args_provider``. Defaults
track AGENTS.md: selection_mode=hessian, topk_ratio=0.04,
W=1, async_queue=5.

Kept as a separate module so the argparse wiring can be imported without
triggering proposal runtime initialization
— useful for tooling that inspects flags without initialising the
checkpoint pipeline.
"""
from __future__ import annotations


def add_proposal_args(parser):
    """Register proposal CLI flags. Invoked as extra_args_provider."""
    group = parser.add_argument_group(title="proposal")
    group.add_argument("--proposal-enable", action="store_true",
                       help="Enable element-wise importance MoE checkpointing.")
    group.add_argument("--proposal-topk-ratio", type=float, default=0.04,
                       help="Fraction of expert elements to snapshot per lossy-FC "
                            "(score-based top-K). Default 4%%.")
    group.add_argument("--proposal-block-quant-g", type=int, default=8,
                       help="Block quantization group size for descriptor traversal. "
                            "Default 8. Production selection is element-wise; this "
                            "value controls contiguous descriptor grouping, not a "
                            "separate group-score pool.")
    group.add_argument("--proposal-selection-mode", type=str, default="hessian",
                       choices=["v_hat", "random", "hessian"],
                       help="Score metric for element selection: "
                            "'v_hat' = Adam bias-corrected v̂_t × δ² ; "
                            "'hessian' = |H_diag| × δ² (Finite-Difference) ; "
                            "'random' = uniform random (ablation). Default 'hessian'.")
    group.add_argument("--proposal-score-policy", type=str,
                       default="abs_h_taylor_first_and_second_order",
                       choices=[
                           "abs_h_taylor_first_and_second_order",
                           "abs_h_taylor_only_second_order",
                       ],
                       help="Production hessian score policy. The default "
                            "uses the configured first-order source plus "
                            "0.5*|H|*e_hat^2; the second-order ablation uses "
                            "|H|*e_hat^2.")
    group.add_argument("--proposal-score-linear-source", type=str,
                       default="grad",
                       choices=["grad", "momentum"],
                       help="First-order Taylor source for hessian scoring. "
                            "'grad' leases DeepSpeed ZeRO-1/2 optimizer "
                            "gradients when available and falls back to "
                            "momentum otherwise. 'momentum' preserves the "
                            "legacy -m*e_hat term.")
    group.add_argument("--proposal-e-hat-proxy-off",
                       "--proposal-ehat-proxy-off",
                       dest="proposal_e_hat_proxy_off",
                       action="store_true",
                       help="Use a persistent 2-byte expert theta_cp snapshot "
                            "for residual scoring instead of the default "
                            "age/Adam e_hat proxy.")
    group.add_argument("--proposal-n-bins-cutoff", type=int, default=4096,
                       help="Selection cutoff-search histogram bins. Default 4096.")
    group.add_argument("--proposal-cutoff-k-tolerance", type=float, default=0.04,
                       help="Allowed relative K-budget tolerance for cutoff "
                            "search. Default 0.04.")
    group.add_argument("--proposal-pinned-aos-capacity-tolerance", type=float,
                       default=0.75,
                       help="Host pinned AOS slab capacity margin over "
                            "--proposal-topk-ratio. This is a buffer-capacity "
                            "contract, not a cutoff accuracy tolerance. "
                            "Default 0.75.")
    group.add_argument("--proposal-adaptive-cutoff-window", type=int, default=16,
                       help="Sliding-window length for adaptive cutoff selected-ratio "
                            "control. Default 16.")
    group.add_argument("--proposal-adaptive-cutoff-min-observations", type=int, default=8,
                       help="Minimum window observations before adaptive cutoff may "
                            "predict tau. Default 8.")
    group.add_argument("--proposal-adaptive-cutoff-warmup-iters", type=int, default=0,
                       help="Initial proposal iterations treated as the warmup/limit "
                            "region for adaptive cutoff exact histogram. Default 0; "
                            "LR warmup only suppresses stable-skip candidate-count "
                            "elision.")
    group.add_argument("--proposal-adaptive-cutoff-retry-tolerance", type=float, default=0.04,
                       help="Relative selected-ratio slack used by adaptive cutoff "
                            "certification and guards. Default 0.04.")
    group.add_argument("--proposal-n-q", type=int, default=4096,
                       help="Number of v/h quantile points. Default 4096.")
    group.add_argument("--proposal-h-lookup-lut-bins", type=int, default=8192,
                       help="Number of log(v)->H lookup LUT bins. Default 8192.")
    group.add_argument("--proposal-finite-diff-eps", type=float, default=None,
                       help="Absolute eps for finite-difference Hz (2-point central). "
                            "Default None → 1e-3 (validated on production exp_etc, "
                            "BF16 + MoE). Override via this flag or PROPOSAL_FD_EPS "
                            "env if a very different scale ever proves the default unsafe.")
    group.add_argument("--proposal-hessian-refresh-every", type=int, default=50,
                       help="With --proposal-selection-mode=hessian: re-estimate "
                            "|H_diag| every N iters and overwrite the cached H. "
                            "Default 50 after LR warmup. Set 0 to disable.")
    group.add_argument("--proposal-hessian-refresh-warmup-every", type=int, default=20,
                       help="During LR warmup, refresh |H_diag| every N iters. "
                            "Default 20. Ignored when "
                            "--proposal-hessian-refresh-every=0.")
    group.add_argument("--proposal-test-skip-expert-commit", action="store_true",
                       help="Test/profiling only: run expert selection, AOS "
                            "materialization, and D2H, then skip CPU dense "
                            "scatter, LFSY file writes, and LFCL commit.")
    group.add_argument("--proposal-test-skip-expert-d2h", action="store_true",
                       help="Test/profiling only: with "
                            "--proposal-test-skip-expert-commit, run expert "
                            "selection and AOS materialization but skip AOS "
                            "GPU-to-host copy. This isolates non-D2H proposal "
                            "overhead and is invalid for recovery/durability.")
    group.add_argument("--proposal-test-expert-selection-only", action="store_true",
                       help="Test/profiling only: with "
                            "--proposal-test-skip-expert-d2h, run expert "
                            "selection/count and host layout only, then skip "
                            "GPU record emission. This isolates expert "
                            "selection/cutoff overhead and is invalid for "
                            "recovery/durability.")
    group.add_argument("--proposal-test-skip-nonexpert-commit", action="store_true",
                       help="Test/profiling only: run non-expert FC D2H, then "
                            "skip PRPTENS file write/fsync/retention cleanup. "
                            "Use with --proposal-test-skip-expert-commit to "
                            "measure write-excluded proposal overhead.")
    group.add_argument("--proposal-test-skip-nonexpert-d2h", action="store_true",
                       help="Test/profiling only: with "
                            "--proposal-test-skip-nonexpert-commit, skip "
                            "non-expert FC GPU-to-host copy as well. This "
                            "isolates non-D2H proposal overhead and is invalid "
                            "for recovery/durability.")
    group.add_argument("--proposal-lfcl-plan-policy", type=str,
                       default="lossy_log_size",
                       choices=["round_robin", "lossy_log_size"],
                       help="Expert LFCL compaction planner. lossy_log_size "
                            "is the V2 default and compacts experts with the "
                            "largest current LFSY files; round_robin is kept "
                            "as a deterministic baseline.")
    group.add_argument("--proposal-nonexpert-fc-W", type=int, default=1,
                       help="Non-expert lossless FC interval (iters). Default 1 "
                            "(V2 spec: W=1, lossless non-expert FC every iter; DC removed).")
    group.add_argument("--proposal-async-queue", type=int, default=5)
    group.add_argument("--proposal-cp-mode", type=str, default="proposal",
                       choices=["proposal", "sequential"],
                       help="Expert checkpoint strategy. "
                            "'proposal' (default): Lossy FC (append-only log, "
                            "top-K%% elements each iter) + 1%% expert LFCL "
                            "compaction every iter. "
                            "'sequential': MoC-System baseline (layer-wise "
                            "round-robin dense expert checkpoint, no element-wise lossy).")
    group.add_argument("--proposal-cp-target", type=str, default="disk",
                       choices=["disk", "test_dram"],
                       help="Where to persist expert checkpoints. "
                            "'disk' (default): write LFCL/LFSY expert files "
                            "to --proposal-cp-dir on storage; recovery reads "
                            "files back. SSD 4KB write granularity forces the "
                            "log format. "
                            "'test_dram': test-only target that keeps the "
                            "latest non-expert FC and expert LFCL/LFSY "
                            "recovery image in host RAM, skipping filesystem "
                            "durability.")
    group.add_argument("--proposal-cp-dir", type=str, default="./proposal_cp",
                       help="Directory for proposal checkpoint files.")
    group.add_argument("--proposal-fault-interval", type=int, default=0,
                       help="Synthetic recovery check: every N iters, call "
                            "proposal_recover() in the middle of training and continue "
                            "from the reconstructed state. 0 disables.")
    group.add_argument("--proposal-fault-warmup", type=int, default=0,
                       help="Skip synthetic recovery checks for the first N iters "
                            "(warm-up). First fault fires at iter ≥ N (and == multiple "
                            "of --proposal-fault-interval).")
    group.add_argument("--proposal-fault-log", type=str, default=None,
                       help="Path to append JSON-per-line fault records "
                            "(iter, pre_fault_loss, post_fault_loss, recover_time_s).")
    group.add_argument("--proposal-independent-oracle-enable", action="store_true",
                       help="Telemetry-only: run independent rollback oracle "
                            "at selected iterations. This does not change the "
                            "production selection/write path.")
    group.add_argument("--proposal-independent-oracle-iters", type=str, default="",
                       help="Comma-separated iterations for independent oracle "
                            "measurement. Empty means every iteration.")
    group.add_argument("--proposal-independent-oracle-log-dir", type=str, default=None,
                       help="Directory for independent oracle JSONL outputs.")
    group.add_argument("--proposal-independent-oracle-rank-sample", type=int,
                       default=1048576,
                       help="Per-rank element sample count for rank/mass oracle "
                            "metrics. Default 1048576.")
    group.add_argument("--proposal-independent-oracle-rank-sample-seed", type=int,
                       default=20240625,
                       help="Deterministic sample seed for independent oracle "
                            "rank/mass metrics.")
    group.add_argument("--proposal-independent-oracle-actual-batches", type=int,
                       default=4,
                       help="Number of validation batches used for paired actual "
                            "rollback loss. Default 4.")
    group.add_argument("--proposal-independent-oracle-hvp-batches", type=int,
                       default=4,
                       help="Number of the same validation batches used for HVP "
                            "quadratic term. Default 4.")
    group.add_argument("--proposal-independent-oracle-batch-seed", type=int,
                       default=20240625,
                       help="Telemetry metadata seed for paired oracle batches. "
                            "Runs still use the normal deterministic validation "
                            "iterator.")
    return parser
