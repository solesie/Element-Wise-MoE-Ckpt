#!/bin/bash
# ETC experiment template — 125M PR-MoE / SlimPajama / 2000 iter / seed=2024
# Fault injection every FAULT_INTERVAL iters (default 200).
#
# Variants (set via VARIANT env var):
#   baseline        — no CP
#   moc_m2          — MoC-System baseline, M=2 (1 expert FC every 2 iter)
#   proposal_s4_k01 — ETC-2: Proposal FC every 4 iter + lossy K=0.1%
#   proposal_nofc   — ETC-1: Proposal iter-1 burst only + lossy K=0.15%
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}

# Default config = exp1 (125M PR-MoE, 12 layers, 40 experts). Override via
# env vars for exp2 etc. All model dims exposed here:
SEQ_LEN=${SEQ_LEN:-2048}
NUM_LAYERS=${NUM_LAYERS:-12}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
NUM_ATTN_HEADS=${NUM_ATTN_HEADS:-12}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-24}
BATCH_SIZE=${BATCH_SIZE:-8}
EXPERT_INTERVAL=${EXPERT_INTERVAL:-2}
# Reduced expert count for better per-expert token saturation:
# Original "16 16 32 32 48 48" with cap=1.0 gave ~0.43× tokens/expert.
# New "4 4 8 8 8 8" (40 total) + topk=2 + cap=1.25 restores healthy token flow.
# exp2 override example: PRMOE_SPEC="8 8 8 8 8 8 8 8 8 8 8 8" (24 layers × 8).
PRMOE_SPEC=${PRMOE_SPEC:-"4 4 8 8 8 8"}
NUM_SLOTS=$(echo ${PRMOE_SPEC} | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; print s}')

TRAIN_ITERS=${TRAIN_ITERS:-2000}
SEED=${SEED:-2024}
LR=${LR:-1.5e-4}
MIN_LR=${MIN_LR:-1.5e-5}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-50}
INIT_STD=0.014
MLC=0.01

VARIANT=${VARIANT:?must set VARIANT=baseline|moc_m2|moc_d18|proposal_s4_k01|element_wise|proposal_gnb|proposal_hutch|proposal_hutch_k4pct|proposal_hutch_k3pct|proposal_hutch_k2pct|proposal_hutch_k10pct|proposal_ema_hutch|proposal_exp2_k2|proposal_exp2_k4|proposal_nofc|exp3_run1|exp3_run2|exp3_run3_moc_d4}
OUTPUT_BASEPATH=${OUTPUT_BASEPATH:?must set OUTPUT_BASEPATH}
TAG="${VARIANT}_seed${SEED}"
FAULT_INTERVAL=${FAULT_INTERVAL:-100}
FAULT_WARMUP=${FAULT_WARMUP:-300}
mkdir -p "${OUTPUT_BASEPATH}/log" "${OUTPUT_BASEPATH}/cp"
LOG_FILE="${OUTPUT_BASEPATH}/log/${TAG}.log"
FAULT_LOG="${OUTPUT_BASEPATH}/log/${TAG}.faults.jsonl"
CP_DIR="${CP_DIR_OVERRIDE:-${OUTPUT_BASEPATH}/cp/${TAG}}"
# Preserve pre-created subdirs (e.g. symlinks to separate mount points for
# non_expert/ and expert/). Only clean the files inside, not the dirs.
if [ -d "${CP_DIR}" ]; then
    find "${CP_DIR}" -mindepth 1 -type f -delete 2>/dev/null || true
else
    mkdir -p "${CP_DIR}"
fi
mkdir -p "${CP_DIR}/non_expert" "${CP_DIR}/expert"
rm -f "${FAULT_LOG}"

# DC retention: default max(D, W). Expert FC supersedes older expert
# sparse DC, non-expert FC supersedes older non-expert dense DC.
DC_RETENTION=0  # 0 = use default max(D, W) in cp_hooks

PROPOSAL_FLAGS=""
case "$VARIANT" in
    baseline)
        PROPOSAL_FLAGS=""  # no CP
        ;;
    moc_m2)
        # MoC-System M=2: 1 expert FC every 2 iter (round_robin with D=2).
        # Non-expert FC every iter (W=1) — original MoC behavior.
        # No DC (sequential mode skips ckpt.record). DC retention irrelevant.
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode sequential \
            --proposal-expert-fc-D 2 \
            --proposal-nonexpert-fc-W 1 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG}"
        ;;
    moc_d18)
        # MoC-System with D=18: layer-wise round robin, fire all experts of
        # one layer every 18 iter. Full cycle = 18 × num_MoE_layers.
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode sequential \
            --proposal-expert-fc-D 18 \
            --proposal-nonexpert-fc-W 1 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG}"
        ;;
    proposal_s4_k01)
        # ETC-2: S=4 (D = 4 × num_slots), K=0.5% lossy-FC every iter.
        # W=10: non-expert FC every 10 iters. Rolling lossy-FC state in CPU RAM.
        # Storage budget per iter ≈ MoC m2 (28 MB):
        #   FC amortized = (1/4) × per_expert × 12B ≈ 14 MB
        #   sparse = 0.005 × total_expert_numel × 16B ≈ 14 MB
        FC_D=$((4 * NUM_SLOTS))
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode staggered \
            --proposal-selection-mode v_hat \
            --proposal-expert-fc-D ${FC_D} \
            --proposal-topk-ratio 0.005 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-dc-retention ${DC_RETENTION} \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG} \
            --proposal-score-diag-iters 100 200 400 600 800 1000 1200 1400 1600 1800 2000 \
            --proposal-score-diag-log ${SCORE_DIAG_LOG}"
        ;;
    element_wise)
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode element_wise \
            --proposal-selection-mode v_hat \
            --proposal-topk-ratio 0.01 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-dc-retention ${TRAIN_ITERS} \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG} \
            --proposal-score-diag-iters 100 200 400 600 800 1000 1200 1400 1600 1800 2000 \
            --proposal-score-diag-log ${SCORE_DIAG_LOG}"
        ;;
    proposal_gnb)
        # Same as element_wise but score uses |H| (Sophia-G
        # Gauss-Newton-Bartlett estimate of diag(H)) instead of v_t.
        # Periodically dumps v_t vs |H| Pearson/Jaccard diagnostics.
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        H_CMP_LOG="${OUTPUT_BASEPATH}/log/${TAG}.gnb_vs_vt.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode element_wise \
            --proposal-selection-mode hessian \
            --proposal-hessian-refresh-every 10 \
            --proposal-hessian-num-samples 1 \
            --proposal-hessian-comparison-log ${H_CMP_LOG} \
            --proposal-topk-ratio 0.01 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-dc-retention ${TRAIN_ITERS} \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG} \
            --proposal-score-diag-iters 100 200 400 600 800 1000 1200 1400 1600 1800 2000 \
            --proposal-score-diag-log ${SCORE_DIAG_LOG}"
        ;;
    proposal_hutch)
        # Same data path as proposal_gnb but score uses |H| from K-sample Hutchinson
        # (Rademacher estimator on true-label CE). Runs GNB cross-check on same
        # param state each refresh to log Hutchinson-vs-GNB ranking similarity.
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        H_CMP_LOG="${OUTPUT_BASEPATH}/log/${TAG}.gnb_vs_vt.jsonl"
        HUTCH_GNB_LOG="${OUTPUT_BASEPATH}/log/${TAG}.hutch_vs_gnb.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode element_wise \
            --proposal-selection-mode hessian \
            --proposal-hessian-estimator hutchinson \
            --proposal-hessian-cross-check \
            --proposal-hessian-cross-check-log ${HUTCH_GNB_LOG} \
            --proposal-hessian-refresh-every 40 \
            --proposal-hessian-num-samples 4 \
            --proposal-hessian-comparison-log ${H_CMP_LOG} \
            --proposal-topk-ratio 0.01 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-dc-retention ${TRAIN_ITERS} \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG} \
            --proposal-score-diag-iters 100 200 400 600 800 1000 1200 1400 1600 1800 2000 \
            --proposal-score-diag-log ${SCORE_DIAG_LOG}"
        ;;
    proposal_hutch_k2pct)
        # Same as proposal_hutch but topk-ratio 2% (vs 1%). Split CP layout
        # (non_expert/ + expert/ dirs) is the default now.
        # Rolling state ENABLED intentionally — this variant is for observing
        # random-write WAF impact on nvmev-backed storage. Expect slower iter
        # time (~6 s/iter vs 2.5s baseline) due to mmap scatter-writeback on
        # nvmev. That slowness IS the measurement target.
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode element_wise \
            --proposal-selection-mode hessian \
            --proposal-hessian-estimator hutchinson \
            --proposal-hessian-refresh-every 40 \
            --proposal-hessian-num-samples 4 \
            --proposal-topk-ratio 0.02 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-dc-retention ${TRAIN_ITERS} \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG} \
            --proposal-score-diag-iters 100 200 400 600 800 1000 1200 1400 1600 1800 2000 \
            --proposal-score-diag-log ${SCORE_DIAG_LOG}"
        ;;
    proposal_hutch_k3pct)
        # Same as proposal_hutch_k2pct but topk-ratio 3% (vs 2%).
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode element_wise \
            --proposal-selection-mode hessian \
            --proposal-hessian-estimator hutchinson \
            --proposal-hessian-refresh-every 40 \
            --proposal-hessian-num-samples 4 \
            --proposal-topk-ratio 0.03 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-dc-retention ${TRAIN_ITERS} \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG} \
            --proposal-score-diag-iters 100 200 400 600 800 1000 1200 1400 1600 1800 2000 \
            --proposal-score-diag-log ${SCORE_DIAG_LOG}"
        ;;
    proposal_hutch_k10pct)
        # K=10% + per-iter rolling fsync (TEMP durability test per user).
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode element_wise \
            --proposal-selection-mode hessian \
            --proposal-hessian-estimator hutchinson \
            --proposal-hessian-refresh-every 40 \
            --proposal-hessian-num-samples 4 \
            --proposal-topk-ratio 0.10 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-dc-retention ${TRAIN_ITERS} \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG} \
            --proposal-score-diag-iters 100 200 400 600 800 1000 1200 1400 1600 1800 2000 \
            --proposal-score-diag-log ${SCORE_DIAG_LOG}"
        ;;
    proposal_hutch_k4pct)
        # Same as proposal_hutch but topk-ratio 4% (vs 1%). Includes temp
        # non-expert |H| probe on score_diag iters (expert vs non-expert
        # |H| distribution analysis).
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        H_CMP_LOG="${OUTPUT_BASEPATH}/log/${TAG}.gnb_vs_vt.jsonl"
        HUTCH_GNB_LOG="${OUTPUT_BASEPATH}/log/${TAG}.hutch_vs_gnb.jsonl"
        H_NE_PROBE_LOG="${OUTPUT_BASEPATH}/log/${TAG}.H_nonexpert_probe.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode element_wise \
            --proposal-selection-mode hessian \
            --proposal-hessian-estimator hutchinson \
            --proposal-hessian-cross-check \
            --proposal-hessian-cross-check-log ${HUTCH_GNB_LOG} \
            --proposal-hessian-nonexpert-probe-log ${H_NE_PROBE_LOG} \
            --proposal-hessian-refresh-every 40 \
            --proposal-hessian-num-samples 4 \
            --proposal-hessian-comparison-log ${H_CMP_LOG} \
            --proposal-topk-ratio 0.04 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-dc-retention ${TRAIN_ITERS} \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG} \
            --proposal-score-diag-iters 100 200 400 600 800 1000 1200 1400 1600 1800 2000 \
            --proposal-score-diag-log ${SCORE_DIAG_LOG}"
        ;;
    proposal_ema_hutch)
        # Lossy FC (append-only) + round-robin expert lossless FC with
        # Hutchinson K=4 + H EMA (α=0.9, refresh every 40 iter). CP params:
        #   K(importance top-ratio) = 3%       --proposal-topk-ratio 0.03
        #   D(expert lossless FC interval) = 2 --proposal-expert-fc-D 2
        #   F(experts per lossless event)  = 2% --proposal-lossless-fc-fraction 2.0
        #     → ceil(40 × 0.02) = 1 expert / 2 iter → per-expert cycle 100 iter
        #   W(non-expert FC interval) = 10     --proposal-nonexpert-fc-W 10
        # H EMA state async-persisted to ${CP_DIR}/hessian/ (created by cp_hooks).
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode proposal \
            --proposal-selection-mode ${SELECTION_MODE:-hessian} \
            --proposal-hessian-estimator hutchinson \
            --proposal-hessian-refresh-every ${HESSIAN_REFRESH_EVERY:-40} \
            --proposal-hessian-num-samples ${HESSIAN_NUM_SAMPLES:-2} \
            --proposal-hessian-mode ${HESSIAN_MODE:-ema} \
            --proposal-hessian-ema-alpha ${HESSIAN_EMA_ALPHA:-0.9} \
            ${HESSIAN_PERSIST:+--proposal-hessian-persist} \
            --proposal-topk-ratio 0.03 \
            --proposal-expert-fc-D 2 \
            --proposal-lossless-fc-fraction 2.0 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG} \
            --proposal-score-diag-iters 100 200 400 600 800 1000 1200 1400 1600 1800 2000 \
            --proposal-score-diag-log ${SCORE_DIAG_LOG}"
        ;;
    proposal_exp2_k2)
        # Exp2 run 1: proposal_ema_hutch-style (Hutchinson K=4 refresh=40,
        # EMA α=0.9) with K_top=2%, D=1, F=1% (→ ceil(192 × 0.01) = 2 experts
        # per D=1 iter event). Diagnostics pared back from exp_etc_hutch_k4pct:
        # GNB cross-check removed (GNB retirement + OOM risk), non-expert H
        # probe removed (extra Hutchinson pass → OOM risk at 71+ GB peak).
        # Expert H and score distributions still come from score_diag.jsonl.
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode proposal \
            --proposal-selection-mode ${SELECTION_MODE:-hessian} \
            --proposal-hessian-estimator hutchinson \
            --proposal-hessian-refresh-every ${HESSIAN_REFRESH_EVERY:-40} \
            --proposal-hessian-num-samples ${HESSIAN_NUM_SAMPLES:-2} \
            --proposal-hessian-mode ${HESSIAN_MODE:-ema} \
            --proposal-hessian-ema-alpha ${HESSIAN_EMA_ALPHA:-0.9} \
            ${HESSIAN_PERSIST:+--proposal-hessian-persist} \
            --proposal-topk-ratio 0.02 \
            --proposal-expert-fc-D 1 \
            --proposal-lossless-fc-fraction 1.0 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG} \
            --proposal-score-diag-iters 100 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 6600 6800 7000 7200 7400 7600 7800 8000 \
            --proposal-score-diag-log ${SCORE_DIAG_LOG}"
        ;;
    proposal_exp2_k4)
        # Exp2 run 2: same as proposal_exp2_k2 but K_top=4%.
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode proposal \
            --proposal-selection-mode ${SELECTION_MODE:-hessian} \
            --proposal-hessian-estimator hutchinson \
            --proposal-hessian-refresh-every ${HESSIAN_REFRESH_EVERY:-40} \
            --proposal-hessian-num-samples ${HESSIAN_NUM_SAMPLES:-2} \
            --proposal-hessian-mode ${HESSIAN_MODE:-ema} \
            --proposal-hessian-ema-alpha ${HESSIAN_EMA_ALPHA:-0.9} \
            ${HESSIAN_PERSIST:+--proposal-hessian-persist} \
            --proposal-topk-ratio 0.04 \
            --proposal-expert-fc-D 1 \
            --proposal-lossless-fc-fraction 1.0 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG} \
            --proposal-score-diag-iters 100 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 6600 6800 7000 7200 7400 7600 7800 8000 \
            --proposal-score-diag-log ${SCORE_DIAG_LOG}"
        ;;
    proposal_nofc)
        # ETC-1: stagger=0 → iter-1 full burst + DC-only (no ongoing FC),
        # K=0.15%. W=10. DC retention=TRAIN_ITERS because expert FC never
        # fires again → expert sparse DC is never superseded.
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode element_wise \
            --proposal-selection-mode v_hat \
            --proposal-topk-ratio 0.0015 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-dc-retention ${TRAIN_ITERS} \
            --proposal-replay-at-end"
        ;;
    exp3_run1)
        # Exp3 Run 1 — proposed system, K_top=1%, D=1, F=0.5%
        # F=0.5% × 768 total experts = ceil(3.84) = 4 experts per fire.
        # Score diag + v̂_t vs |H| comparison enabled for percentile/Pearson plots.
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        HCOMP_LOG="${OUTPUT_BASEPATH}/log/${TAG}.vhat_vs_h.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode proposal \
            --proposal-selection-mode hessian \
            --proposal-hessian-estimator hutchinson \
            --proposal-hessian-refresh-every ${HESSIAN_REFRESH_EVERY:-40} \
            --proposal-hessian-num-samples ${HESSIAN_NUM_SAMPLES:-4} \
            --proposal-hessian-mode ${HESSIAN_MODE:-ema} \
            --proposal-hessian-ema-alpha ${HESSIAN_EMA_ALPHA:-0.9} \
            ${HESSIAN_PERSIST:+--proposal-hessian-persist} \
            --proposal-topk-ratio 0.01 \
            --proposal-expert-fc-D 1 \
            --proposal-lossless-fc-fraction 0.5 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG} \
            --proposal-score-diag-iters 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 6600 \
            --proposal-score-diag-log ${SCORE_DIAG_LOG} \
            --proposal-hessian-comparison-log ${HCOMP_LOG}"
        ;;
    exp3_run2)
        # Exp3 Run 2 — proposed system, K_top=4%, D=1, F=1%
        # F=1% × 768 total experts = ceil(7.68) = 8 experts per fire.
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        HCOMP_LOG="${OUTPUT_BASEPATH}/log/${TAG}.vhat_vs_h.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode proposal \
            --proposal-selection-mode hessian \
            --proposal-hessian-estimator hutchinson \
            --proposal-hessian-refresh-every ${HESSIAN_REFRESH_EVERY:-40} \
            --proposal-hessian-num-samples ${HESSIAN_NUM_SAMPLES:-4} \
            --proposal-hessian-mode ${HESSIAN_MODE:-ema} \
            --proposal-hessian-ema-alpha ${HESSIAN_EMA_ALPHA:-0.9} \
            ${HESSIAN_PERSIST:+--proposal-hessian-persist} \
            --proposal-topk-ratio 0.04 \
            --proposal-expert-fc-D 1 \
            --proposal-lossless-fc-fraction 1.0 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG} \
            --proposal-score-diag-iters 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5200 5400 5600 5800 6000 6200 6400 6600 \
            --proposal-score-diag-log ${SCORE_DIAG_LOG} \
            --proposal-hessian-comparison-log ${HCOMP_LOG}"
        ;;
    exp3_run3_moc_d4)
        # Exp3 Run 3 — MoC PEC: D=4, layer-wise round-robin (each event FCs all
        # 48 experts of one layer). Non-expert FC every iter (W=1) — original
        # MoC behavior. No score-based selection (sequential mode skips
        # importance tracker).
        # Explicitly disable Hutchinson refresh (selection-mode=v_hat means H
        # is never read; refresh-every=0 means the estimator is never run).
        # Defaults are hessian + every 40, which would burn ~11s every 40 iter
        # for nothing in this variant.
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode sequential \
            --proposal-selection-mode v_hat \
            --proposal-hessian-refresh-every 0 \
            --proposal-expert-fc-D 4 \
            --proposal-nonexpert-fc-W 1 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG}"
        ;;
    *)
        echo "Unknown VARIANT=$VARIANT"; exit 1 ;;
esac

VOCAB_PATH=${DIR}/../rebase/gpt2-vocab.json
MERGE_PATH=${DIR}/../rebase/gpt2-merges.txt
DATA_PATH=${DATA_PATH:-/home/asdf1234/data/slimpajama_6b/slimpajama_6b_train_text_document}
VALID_DATA_PATH=${VALID_DATA_PATH:-/home/asdf1234/data/slimpajama_6b/slimpajama_6b_valid_text_document}

data_options=" \
    --vocab-file ${VOCAB_PATH} \
    --merge-file ${MERGE_PATH} \
    --data-path ${DATA_PATH} \
    --valid-data-path ${VALID_DATA_PATH} \
    --data-impl mmap"

megatron_options=" \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 --adam-beta2 0.95 \
    --tensor-model-parallel-size 1 --moe-expert-parallel-size 1 \
    --num-experts ${PRMOE_SPEC} --expert-interval ${EXPERT_INTERVAL} \
    --moe-loss-coeff ${MLC} \
    --moe-train-capacity-factor ${MOE_TRAIN_CAP:-1.25} --moe-eval-capacity-factor ${MOE_EVAL_CAP:-1.25} --moe-min-capacity 4 \
    --topk ${MOE_TOPK:-2} \
    --init-method-std ${INIT_STD} \
    --micro-batch-size ${BATCH_SIZE} --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --num-layers ${NUM_LAYERS} --hidden-size ${HIDDEN_SIZE} --num-attention-heads ${NUM_ATTN_HEADS} \
    --seq-length ${SEQ_LEN} --max-position-embeddings ${SEQ_LEN} \
    --train-iters ${TRAIN_ITERS} \
    --lr ${LR} --min-lr ${MIN_LR} --lr-decay-style cosine --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --split 98,2,0 \
    --log-interval 10 --eval-interval 100 --eval-iters 128 \
    --save-interval 100000 \
    --weight-decay 0.1 --clip-grad 1.0 --hysteresis 2 \
    --num-workers 2 --bf16 --use-flash-attn-v2 --ds_fused_adam \
    --seed ${SEED} \
    ${EXTRA_MEGATRON_FLAGS:-} \
    ${PROPOSAL_FLAGS}"

template_json="${DIR}/ds_config_gpt_TEMPLATE.json"
config_json="${OUTPUT_BASEPATH}/ds_config_${TAG}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/10/" \
    | sed "s/ZERO_STAGE/0/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/false/" \
    | sed "s/CONFIG_BF16_ENABLED/true/" \
    | sed "s/CONFIG_CL_ENABLED/false/" \
    | sed "s/CONFIG_CL_MIN/80/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/0/" > ${config_json}

deepspeed_options=" \
    --deepspeed --deepspeed_config ${config_json} \
    --pipeline-model-parallel-size 1 --no-pipeline-parallel \
    --create-moe-param-group"

run_cmd="deepspeed --num_gpus 1 --master_port 29500 ${DIR}/../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options}"
echo "[etc] TAG=${TAG} NUM_SLOTS=${NUM_SLOTS} LOG=${LOG_FILE}"
eval ${run_cmd} 2>&1 | tee "${LOG_FILE}"
