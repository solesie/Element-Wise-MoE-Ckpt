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
    group.add_argument("--proposal-selection-mode", type=str, default="hessian",
                       choices=["v_hat", "random", "hessian"],
                       help="Score metric for element selection: "
                            "'v_hat' = Adam bias-corrected v̂_t × δ² ; "
                            "'hessian' = |H_diag| × δ² (estimator configurable via "
                            "--proposal-hessian-estimator) ; "
                            "'random' = uniform random (ablation). Default 'hessian'.")
    group.add_argument("--proposal-hessian-estimator", type=str, default="finite_diff",
                       choices=["finite_diff", "hutchinson"],
                       help="Estimator for |H_diag|. 'finite_diff' (default, "
                            "production) = central-difference Hz via 1st-order "
                            "grads with MoE routing freeze + Parameter rebind to "
                            "bypass DS BF16+ZeRO storage aliasing. 'hutchinson' "
                            "(opt-in) = K-sample Rademacher with 2nd-order autograd; "
                            "kept as a truth reference for occasional comparison runs "
                            "on small models — heavier (saved-tensor offload) and not "
                            "validated for EP > 1, so don't use for production.")
    group.add_argument("--proposal-finite-diff-eps", type=float, default=None,
                       help="Absolute eps for finite-difference Hz (2-point central). "
                            "Default None → 1e-3 (validated on production exp_etc, "
                            "BF16 + MoE). Override via this flag or PROPOSAL_FD_EPS "
                            "env if a very different scale (e.g. 100B+ params with "
                            "unusual |H| range) ever proves the default unsafe.")
    group.add_argument("--proposal-finite-diff-routing-freeze", action="store_true",
                       default=True,
                       help="Freeze MoE gating decisions during finite-diff "
                            "+/- eps forwards. Default ON; without freeze, "
                            "top-k argmax flips between perturbed forwards "
                            "produce noise that swamps Hz.")
    group.add_argument("--proposal-no-finite-diff-routing-freeze",
                       dest="proposal_finite_diff_routing_freeze",
                       action="store_false",
                       help="Ablation: turn the routing freeze OFF.")
    group.add_argument("--proposal-finite-diff-cross-check", action="store_true",
                       help="When estimator=finite_diff, ALSO run Hutchinson on "
                            "the same param state and log Pearson/Jaccard via "
                            "compare_hessian_estimators(label_a='hutch', "
                            "label_b='finite_diff') to the cross-check log.")
    group.add_argument("--proposal-finite-diff-flip-cutoff", type=float,
                       default=100.0,
                       help="Outlier protection: per-tensor Hz mean-abs is "
                            "compared to the median across tensors; tensors "
                            "exceeding cutoff*median are skipped from the EMA "
                            "update (residual flip protection beyond routing "
                            "freeze). Default 100.")
    group.add_argument("--proposal-hessian-refresh-every", type=int, default=40,
                       help="With --proposal-selection-mode=hessian: re-estimate "
                            "|H_diag| every N iters and fold into the H EMA. "
                            "Default 40. Set 0 to disable.")
    group.add_argument("--proposal-hessian-num-samples", type=int, default=4,
                       help="Label draws averaged per Hessian refresh. Default 4 "
                            "(matches the current exp_etc_new baseline). K=1 + "
                            "aggressive EMA is a viable low-memory alternative "
                            "but degrades recovery quality slightly.")
    group.add_argument("--proposal-hessian-mode", type=str, default="ema",
                       choices=["ema", "plain"],
                       help="|H_diag| accumulation across refreshes. "
                            "'ema' (default): H_ema = α·H_ema + (1-α)·H_new per "
                            "refresh, smooths K-sample variance across the EMA "
                            "window. 'plain': H_ema is overwritten with H_new "
                            "each refresh — no temporal smoothing, use when "
                            "K_samples is large enough on its own or when "
                            "comparing against a fresh snapshot baseline.")
    group.add_argument("--proposal-hessian-ema-alpha", type=float, default=0.9,
                       help="EMA decay for |H_diag| when --proposal-hessian-mode=ema. "
                            "H_ema = α·H_ema + (1-α)·H_new per refresh. α=0.9 "
                            "gives effective sample count 1/(1-α)=10, which "
                            "dominates the K=1 per-refresh variance. "
                            "Ignored when mode=plain. Default 0.9.")
    group.add_argument("--proposal-hessian-persist", action="store_true",
                       help="Persist |H_diag| (EMA or plain) to "
                            "<cp-dir>/hessian/h_ema_iter*.bin after every refresh. "
                            "Default OFF — the refresh path stays in GPU only, "
                            "no async-writer pressure, no retention bookkeeping. "
                            "Turn ON when recovery needs to resume H state across "
                            "faults (--proposal-fault-interval with EMA continuity), "
                            "or for post-hoc analysis of |H| trajectory.")
    group.add_argument("--proposal-hessian-comparison-log", type=str, default=None,
                       help="JSONL path for periodic v̂_t vs |H_diag| comparison "
                            "(Pearson / Jaccard over bare and score=X·δ²). "
                            "Populated on each refresh_hessian call.")
    group.add_argument("--proposal-hessian-cross-check", action="store_true",
                       help="When estimator=hutchinson, also run GNB on the same "
                            "param state and log Hutchinson-vs-GNB ranking "
                            "similarity (Pearson + Jaccard).")
    group.add_argument("--proposal-hessian-cross-check-log", type=str, default=None,
                       help="JSONL path for Hutchinson-vs-GNB cross-check dump "
                            "(populated each refresh when "
                            "--proposal-hessian-cross-check is set).")
    group.add_argument("--proposal-hutchinson-at-end", action="store_true",
                       help="At end of training, run one Hutchinson probe to compare "
                            "v̂_t vs |H_diag| as importance proxy. Writes JSONL.")
    group.add_argument("--proposal-hutchinson-k-samples", type=int, default=4)
    group.add_argument("--proposal-hutchinson-log", type=str, default=None)
    # ANALYSIS: non-expert |H| probe — retained for expert-vs-non-expert score
    # distribution measurement (see CLAUDE.md overlap measurement plan).
    group.add_argument("--proposal-hessian-nonexpert-probe-log", type=str, default=None,
                       help="JSONL path for non-expert |H| probe. When set, "
                            "refresh_hessian also computes |H| for non-expert "
                            "params on score_diag_iters and dumps percentile "
                            "comparison with expert |H|.")
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
    group.add_argument("--proposal-dc-retention", type=int, default=0)
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
    group.add_argument("--proposal-replay-at-end", action="store_true",
                       help="After training completes, call proposal_recover() and re-run "
                            "validation. Used for exp 1 (CP reconstruction quality measurement).")
    group.add_argument("--proposal-replay-only", action="store_true",
                       help="Skip training entirely; load existing CP files via "
                            "proposal_recover() then run validation. Use to re-evaluate "
                            "an existing CP dir without re-training.")
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
    group.add_argument("--proposal-score-diag-iters", type=int, nargs="+",
                       default=None,
                       help="Measure expert vs non-expert score distribution "
                            "(v̂_t × δ²) at each of these iters, plus per-layer "
                            "sum + avg per expert. Dumps JSONL to --proposal-score-diag-log.")
    group.add_argument("--proposal-score-diag-log", type=str, default=None,
                       help="Path to JSONL file for score distribution diagnostic records.")
    group.add_argument("--proposal-score-diag-every", type=int, default=0,
                       help="If >0, measure score distribution every N iter "
                            "(dense periodic sampling; use with --proposal-score-diag-log).")
    return parser
