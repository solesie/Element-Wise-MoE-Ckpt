"""CLI argument registration for the MoE element-wise importance
checkpointing pipeline.

Registered via ``pretrain_gpt.py``'s ``extra_args_provider``. Defaults
track CLAUDE.md: cp_mode=element_wise, selection_mode=hessian,
hessian_estimator=finite_diff (default), topk_ratio=0.05, W=10, async_queue=5.

Kept as a separate module (vs. inline in cp_hooks.py) so the argparse
wiring can be imported without triggering the heavier cp_hooks module
— useful for tooling that inspects flags without initialising the
checkpoint pipeline.
"""
from __future__ import annotations


def add_proposal_args(parser):
    """Register proposal CLI flags. Invoked as extra_args_provider."""
    group = parser.add_argument_group(title="proposal")
    group.add_argument("--proposal-enable", action="store_true",
                       help="Enable element-wise importance MoE checkpointing.")
    group.add_argument("--proposal-topk-ratio", type=float, default=0.05,
                       help="Fraction of expert elements to snapshot per lossy-FC "
                            "(score-based top-K). Default 5%%.")
    group.add_argument("--proposal-block-quant-g", type=int, default=1,
                       help="Block quantization group size for selection. Default 1 "
                            "(no block quant). When g>1, contiguous g-element groups "
                            "share a group score (sum of element scores) and are "
                            "selected as a unit. Saves use element-level values "
                            "(Option B): K-element budget preserved.")
    # FD multi-K diagnostic (per-refresh K_x vs K_max comparison).
    group.add_argument("--proposal-fd-multi-k-track", action="store_true",
                       help="Track FD H estimates at multiple K values (per refresh). "
                            "Snapshots accumulated h_sum / k_done at each requested K "
                            "boundary; pearson(K_x, K_max) and pearson(prev_K_max, "
                            "current_K_max) logged.")
    group.add_argument("--proposal-fd-multi-k-values", type=int, nargs="+",
                       default=[1, 2, 4, 8, 16, 32],
                       help="K boundary values for multi-K tracking. Last value used "
                            "as truth for cross-K pearson. Refresh runs k_samples = "
                            "max(values).")
    group.add_argument("--proposal-fd-multi-k-log", type=str, default=None,
                       help="JSONL output path for multi-K pearson records.")
    group.add_argument("--proposal-selection-mode", type=str, default="hessian",
                       choices=["v_hat", "random", "hessian"],
                       help="Score metric for element selection: "
                            "'v_hat' = Adam bias-corrected v̂_t × δ² ; "
                            "'hessian' = |H_diag| × δ² (Finite-Difference) ; "
                            "'random' = uniform random (ablation). Default 'hessian'.")
    group.add_argument("--proposal-finite-diff-eps", type=float, default=None,
                       help="Absolute eps for finite-difference Hz (2-point central). "
                            "Default None → 1e-3 (validated on production exp_etc, "
                            "BF16 + MoE). Override via this flag or PROPOSAL_FD_EPS "
                            "env if a very different scale ever proves the default unsafe.")
    group.add_argument("--proposal-finite-diff-cross-check", action="store_true",
                       help="ALSO run Hutchinson on the same param state (CRN batch) "
                            "at refresh and log Pearson/Jaccard via "
                            "compare_hessian_estimators. Production refresh is "
                            "FD-only; this flag is the ONLY surviving Hutchinson "
                            "path, kept for FD-vs-Hutch validation runs.")
    group.add_argument("--proposal-finite-diff-cross-check-log", type=str, default=None,
                       help="JSONL output path for the FD-vs-Hutch cross-check "
                            "(populated when --proposal-finite-diff-cross-check is "
                            "set). When None, results print to stdout only.")
    group.add_argument("--proposal-finite-diff-flip-cutoff", type=float,
                       default=100.0,
                       help="Outlier protection: per-tensor Hz mean-abs is "
                            "compared to the median across tensors; tensors "
                            "exceeding cutoff*median are skipped from the H "
                            "update (residual flip protection beyond routing "
                            "freeze). Default 100.")
    group.add_argument("--proposal-hessian-refresh-every", type=int, default=40,
                       help="With --proposal-selection-mode=hessian: re-estimate "
                            "|H_diag| every N iters and overwrite the cached H. "
                            "Default 40. Set 0 to disable.")
    group.add_argument("--proposal-hessian-num-samples", type=int, default=4,
                       help="Label draws averaged per Hessian refresh. Default 4.")
    group.add_argument("--proposal-hutchinson-at-end", action="store_true",
                       help="At end of training, run one Hutchinson probe and write "
                            "a JSONL log (post-hoc cross-check; production refresh is "
                            "always FD).")
    group.add_argument("--proposal-hutchinson-k-samples", type=int, default=4)
    group.add_argument("--proposal-hutchinson-log", type=str, default=None)
    # DEBUG: disable the ExpertState lossy FC writer (skip both lossy-FC
    # and lossless-FC of experts). Used to isolate expert-side write cost
    # when profiling. Legacy flag name preserved for compat.
    group.add_argument("--proposal-disable-rolling", action="store_true",
                       help="Debug: skip tracker.set_expert_state() so expert "
                            "lossy-FC is disabled entirely.")
    # D: expert lossless FC interval. Per CLAUDE.md Key Parameters, default 4.
    # Per-expert lossless FC cycle = D × ⌈100/F⌉ iters.
    group.add_argument("--proposal-expert-fc-D", type=int, default=None,
                       help="Expert lossless FC interval in iterations "
                            "(default: 4 for proposal mode, 1 for sequential).")
    group.add_argument("--proposal-lossless-fc-fraction", type=float, default=2.0,
                       help="F%% — fraction of total experts that receive a "
                            "lossless FC snapshot each time the D-interval "
                            "fires. Per CLAUDE.md default 2.0 → 1 expert per "
                            "fire for 48-expert model → 200-iter cycle.")
    group.add_argument("--proposal-nonexpert-fc-W", type=int, default=10)
    group.add_argument("--proposal-async-queue", type=int, default=5)
    group.add_argument("--proposal-cp-mode", type=str, default="proposal",
                       choices=["proposal", "sequential"],
                       help="Expert checkpoint strategy. "
                            "'proposal' (default): Lossy FC (append-only log, "
                            "top-K%% elements each iter) + round-robin F%% "
                            "expert lossless FC every D iter. "
                            "'sequential': MoC-System baseline (layer-wise "
                            "round-robin lossless FC, no element-wise lossy).")
    group.add_argument("--proposal-cp-target", type=str, default="disk",
                       choices=["disk", "dram"],
                       help="Where to persist expert checkpoints. "
                            "'disk' (default): write lossless/lossy FC files "
                            "to --proposal-cp-dir on storage; recovery reads "
                            "files back. SSD 4KB write granularity forces the "
                            "log format. "
                            "'dram': proposal mode keeps a per-expert "
                            "sparse-dense FP32 buffer in host RAM (no log, no "
                            "file I/O). Lossy FC scatters new (idx, p, m, v) "
                            "directly into the buffer; recovery reads the "
                            "buffer in O(expert_size) regardless of how many "
                            "lossy appends accumulated. Use for fast "
                            "validation-loss experiments where storage "
                            "cost/recovery time is not the variable under "
                            "study. Sequential (MoC) mode currently still "
                            "uses disk regardless of this flag — set "
                            "CP_DIR_OVERRIDE to a tmpfs path (e.g. /dev/shm) "
                            "for that mode if you want RAM-speed I/O.")
    group.add_argument("--proposal-cp-dir", type=str, default="./proposal_cp",
                       help="Directory for proposal checkpoint files.")
    group.add_argument("--proposal-fault-interval", type=int, default=0,
                       help="Exp 2 fault injection: every N iters, call "
                            "proposal_recover() in the middle of training and continue "
                            "from the reconstructed state. 0 disables.")
    group.add_argument("--proposal-fault-warmup", type=int, default=0,
                       help="Exp 2: skip fault injection for the first N iters "
                            "(warm-up). First fault fires at iter ≥ N (and == multiple "
                            "of --proposal-fault-interval).")
    group.add_argument("--proposal-fault-log", type=str, default=None,
                       help="Path to append JSON-per-line fault records "
                            "(iter, pre_fault_loss, post_fault_loss, recover_time_s).")
    group.add_argument("--proposal-fisher-coverage-iters", type=int, nargs="+",
                       default=None,
                       help="Measure Fisher mass concentration C(K) at each of "
                            "these iters (using current Adam v_t over expert "
                            "params only — ZeRO-aware via Fp32StateMapper). "
                            "Dumps JSON list next to fault log.")
    return parser
