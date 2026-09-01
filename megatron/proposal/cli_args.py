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
                       help="Deprecated compatibility option. Proposal I/O and "
                            "H refresh now start only after LR warmup.")
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
    group.add_argument("--proposal-lfcl-fraction-pct", type=float, default=1.0,
                       help="Global expert-slot percentage compacted into dense "
                            "LFCL state at each proposal LFCL event. Proposal "
                            "events round up globally; cumulative-quota MoC "
                            "policies preserve the exact long-run ratio. "
                            "Default 1.0.")
    group.add_argument("--proposal-nonexpert-fc-W", type=int, default=1,
                       help="Non-expert lossless FC interval (iters). Default 1 "
                            "(V2 spec: W=1, lossless non-expert FC every iter; DC removed).")
    group.add_argument("--proposal-async-queue", type=int, default=5)
    group.add_argument("--proposal-cp-mode", type=str, default="proposal",
                       choices=["proposal", "sequential"],
                       help="Expert checkpoint strategy. "
                            "'proposal' (default): Lossy FC (append-only log, "
                            "top-K%% elements each iter) + configured expert LFCL "
                            "compaction every iter. "
                            "'sequential': MoC-System baseline (layer-wise "
                            "round-robin dense expert checkpoint, no element-wise lossy).")
    group.add_argument(
        "--proposal-checkpoint-warmup-iters",
        type=int,
        default=None,
        help="Override only the proposal checkpoint cold-start/bootstrap "
             "iteration. LR warmup and synthetic-fault gating remain "
             "unchanged. Default None uses the optimizer LR warmup; "
             "sequential MoC always uses zero checkpoint cold start.",
    )
    group.add_argument(
        "--proposal-moc-sequential-policy",
        type=str,
        default="layer_rr",
        choices=[
            "layer_rr",
            "expert_rr_pp_locked",
            "expert_rr_pp_back_locked",
        ],
        help="MoC-only dense checkpoint order. 'layer_rr' preserves the "
             "canonical whole-layer round robin. 'expert_rr_pp_locked' is "
             "the 125M-MoE64 PP2/EP4 stress profile: four global experts "
             "per iteration for 96 iterations, followed by four idle "
             "iterations in each 100-iteration frame. "
             "'expert_rr_pp_back_locked' is topology-derived and writes an "
             "exact cumulative global quota every iteration, ordered from "
             "the deepest layer to the shallowest layer.",
    )
    group.add_argument("--proposal-lfcl-interval-iters", type=int, default=1,
                       help="Dense expert LFCL event interval. In sequential "
                            "MoC mode, all local experts in one pipeline-local "
                            "layer fire every N iterations. Proposal mode is "
                            "fixed at N=1. Default 1.")
    group.add_argument("--proposal-age-force-period", type=int, default=128,
                       help="Phase-sharded force-flush period for elements in "
                            "the oldest 2-bit age code. This is not an exact "
                            "maximum element age. Default 128.")
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
                       help="Telemetry-only: at selected iterations run both "
                            "the sampled-HVP independent ranking oracle and "
                            "all-element expert/parameter saliency histograms. "
                            "This does not change the production selection/write path.")
    group.add_argument("--proposal-independent-oracle-iters", type=str, default="",
                       help="Comma-separated iterations for independent oracle "
                            "measurement. Empty means every iteration.")
    group.add_argument("--proposal-independent-oracle-log-dir", type=str, default=None,
                       help="Directory for independent oracle JSONL outputs.")
    group.add_argument("--proposal-independent-oracle-rank-sample", type=int,
                       default=1048576,
                       help="Per-rank element sample count for rank/mass oracle "
                            "metrics. Default 1048576.")
    group.add_argument(
        "--proposal-oracle-window-measurement-only",
        action="store_true",
        help="Special measurement-only mode: at each explicit independent-oracle "
             "iteration reset expert/non-expert shadow state, observe only "
             "successful optimizer updates, and disable production checkpoint "
             "bootstrap/dispatch/finalize and fault injection.",
    )
    group.add_argument(
        "--proposal-oracle-window-updates",
        type=int,
        default=100,
        help="Number of successful optimizer updates in each measurement window. "
             "Default 100.",
    )
    group.add_argument(
        "--proposal-oracle-window-snapshot-offsets",
        type=str,
        default="10,20,30,40,50,60,70,80,90,100",
        help="Sorted comma-separated successful-update offsets to snapshot "
             "before shadow-state reset.",
    )
    group.add_argument(
        "--proposal-oracle-window-log-dir",
        type=str,
        default=None,
        help="Rank-sharded oracle-window output directory. Defaults to "
             "<independent-oracle-log-dir>/oracle_window or "
             "<proposal-cp-dir>/independent_oracle/oracle_window.",
    )
    group.add_argument(
        "--proposal-oracle-window-sample-rounds",
        type=int,
        default=4,
        help="Initial number of distinct block-stratified CDF rounds. Minimum 4.",
    )
    group.add_argument(
        "--proposal-oracle-window-max-sample-rounds",
        type=int,
        default=8,
        help="Maximum distinct CDF rounds used when exact K-band validation "
             "requires cutoff refinement. Default 8.",
    )
    group.add_argument(
        "--proposal-oracle-window-sample-blocks",
        type=int,
        default=4096,
        help="Block-stratified samples per domain and round. Default 4096.",
    )
    group.add_argument(
        "--proposal-oracle-window-block-elements",
        type=int,
        default=256,
        help="Elements per sampling block. Default 256.",
    )
    group.add_argument(
        "--proposal-oracle-window-min-effective-tail-count",
        type=float,
        default=8.0,
        help="Minimum effective tail sample count for percentile validity.",
    )
    group.add_argument(
        "--proposal-oracle-window-fd-epsilon",
        type=float,
        default=1.0e-2,
        help="Explicit BF16-representable finite-difference epsilon for the "
             "router-inclusive non-expert oracle probe. Default 1e-2.",
    )
    group.add_argument(
        "--proposal-oracle-window-fd-probes",
        type=int,
        default=4,
        help="Non-expert FD-Hutchinson probes at window T0. Minimum 4.",
    )
    group.add_argument(
        "--proposal-oracle-window-fd-data-batches",
        type=int,
        default=1,
        help="Dedicated validation batches used by non-expert FD at T0.",
    )
    group.add_argument(
        "--proposal-oracle-window-fd-min-changed-fraction",
        type=float,
        default=0.95,
        help="Fail-closed minimum +/- BF16 perturb changed fraction.",
    )
    group.add_argument(
        "--proposal-joint-saliency-bins",
        type=int,
        default=512,
        help="Number of log10 score bins for the all-element telemetry that "
             "runs at every enabled independent-oracle iteration. Default 512.",
    )
    group.add_argument(
        "--proposal-joint-saliency-log-dir",
        type=str,
        default=None,
        help="Rank-sharded NPZ/manifest output directory. Defaults to a "
             "joint_saliency subdirectory of the independent-oracle log dir.",
    )
    return parser
