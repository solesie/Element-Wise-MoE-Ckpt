#!/bin/bash
# ETC experiment template — 125M PR-MoE / SlimPajama / 2000 iter / seed=2024
# Fault injection every FAULT_INTERVAL iters (default 200).
#
# Variants (set via VARIANT env var):
#   baseline        — no CP
#   moc_m2          — MoC-System baseline, M=2 (1 expert FC every 2 iter)
#   proposal_s4_k01 — ETC-2: Proposal FC every 4 iter + lossy K=0.1%
#   proposal_nofc   — ETC-1: Proposal iter-1 burst only + lossy K=0.15%
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$DIR/../.." && pwd)
export LD_LIBRARY_PATH=$(python3 -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null):${LD_LIBRARY_PATH:-}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:${PATH}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Default config = exp1 (125M PR-MoE, 12 layers, 40 experts). Override via
# env vars for exp2 etc. All model dims exposed here:
SEQ_LEN=${SEQ_LEN:-2048}
NUM_LAYERS=${NUM_LAYERS:-12}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
NUM_ATTN_HEADS=${NUM_ATTN_HEADS:-12}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-24}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_GPUS=${NUM_GPUS:-1}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
EP_PARALLEL_SIZE=${EP_PARALLEL_SIZE:-1}
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

VARIANT=${VARIANT:?must set VARIANT=baseline|moc_m2|moc_d18|moc_test_dram_d6|proposal_s4_k01|element_wise|proposal_gnb|proposal_hutch_k1pct_finite_diff_F1|proposal_exp2_k2|proposal_exp2_k4|proposal_nofc|exp3_run1|exp3_run2|exp3_run3_moc_d4|verify_vram|iter_time_debug|proposal_fd_kpct}
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
if [ "${NUM_GPUS}" = "1" ] && [ "${PP_SIZE}" = "1" ] && [ "${EP_PARALLEL_SIZE}" = "1" ]; then
    mkdir -p "${CP_DIR}/non_expert" "${CP_DIR}/expert"
fi
rm -f "${FAULT_LOG}"

PROPOSAL_FLAGS=""
case "$VARIANT" in
    baseline)
        PROPOSAL_FLAGS=""  # no CP
        # Baseline must remain a hard no-proposal run even if the parent
        # shell has leftover profiling/test proposal flags.
        EXTRA_PROPOSAL_FLAGS=""
        ;;
    moc_m2)
        # MoC-System M=2: 1 expert FC every 2 iter (round_robin with D=2).
        # Non-expert FC every iter (W=1) — original MoC behavior.
        # No DC (sequential mode skips ckpt.record). DC retention irrelevant.
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode sequential \
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
            --proposal-nonexpert-fc-W 1 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG}"
        ;;
    moc_test_dram_d6)
        # MoC sequential baseline aligned for direct comparison vs the
        # proposal (proposal_hutch_k1pct_finite_diff_F1) at matched
        # non-expert protection. Only difference from proposal is the
        # expert side: layer-RR dense checkpoint every D=6 iter (one MoE
        # layer at a time) — no element-wise lossy FC, no importance
        # tracker, no Hessian.
        #
        # Knobs:
        #   NONEXPERT_FC_W    non-expert FC interval            (default 10)
        # Non-expert path = identical to proposal: DC every iter +
        # non-expert FC every W iter. Recovery uses test_dram ExpertState
        # so the fault recover_s should match the proposal's ~1.2 s.
        #
        # Selection knobs are pinned to ``v_hat`` / refresh=0 because
        # sequential mode never reads any importance score — those
        # flags exist only to suppress training-loop Hessian refreshes
        # that would otherwise run on every refresh_every iter for
        # nothing (lazy Hessian buffers in the tracker stay
        # unallocated, but the FD recompute itself still runs).
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode sequential \
            --proposal-cp-target ${CP_TARGET:-test_dram} \
            --proposal-selection-mode v_hat \
            --proposal-hessian-refresh-every 0 \
            --proposal-nonexpert-fc-W ${NONEXPERT_FC_W:-10} \
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
            --proposal-topk-ratio 0.005 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG}"
        ;;
    element_wise)
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode element_wise \
            --proposal-selection-mode v_hat \
            --proposal-topk-ratio 0.01 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG}"
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
            --proposal-topk-ratio 0.01 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG}"
        ;;
    proposal_hutch_k1pct_finite_diff_F1|verify_vram|iter_time_debug|proposal_fd_kpct)
        # Production settings (prodsim verified 2026-05-15):
        #   • g=8 (AGENTS.md production)
        #   • refresh_every=100 (50-iter staleness → zero rank impact)
        #   • K=1 anchor (K=1 quantile-equivalent to K=32 truth)
        #   • W=1 non-expert FC every iter (AGENTS.md spec)
        # cp-target=disk (production). test_dram mode is available via
        # CP_TARGET=test_dram for fast in-memory correctness/recovery checks.
        # Expert LFCL compaction is enabled by default: every iter the
        # selected plan targets 1% of expert slots and materializes live
        # GPU p/m/v into the LFCL baseline used to bound LFSY replay cost.
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        H_CMP_LOG="${OUTPUT_BASEPATH}/log/${TAG}.gnb_vs_vt.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode proposal \
            --proposal-cp-target ${CP_TARGET:-disk} \
            --proposal-block-quant-g ${BLOCK_QUANT_G:-8} \
            --proposal-selection-mode ${SELECTION_MODE:-hessian} \
            --proposal-hessian-refresh-every ${HESSIAN_REFRESH_EVERY:-100} \
            --proposal-topk-ratio ${PROPOSAL_TOPK:-0.04} \
            --proposal-nonexpert-fc-W ${NONEXPERT_FC_W:-1} \
            --proposal-async-queue ${ASYNC_QUEUE:-5} \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG}"
        ;;
    proposal_exp2_k2)
        # Exp2 run 1: K_top=2%, D=1, F=1% (→ ceil(192 × 0.01) = 2
        # experts per D=1 iter event). Diagnostics pared back to keep
        # peak VRAM under budget. Expert H and score distributions
        # come from score_diag.jsonl.
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode proposal \
            --proposal-selection-mode ${SELECTION_MODE:-hessian} \
            --proposal-hessian-refresh-every ${HESSIAN_REFRESH_EVERY:-40} \
            --proposal-topk-ratio 0.02 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG}"
        ;;
    proposal_exp2_k4)
        # Exp2 run 2: same as proposal_exp2_k2 but K_top=4%.
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode proposal \
            --proposal-selection-mode ${SELECTION_MODE:-hessian} \
            --proposal-hessian-refresh-every ${HESSIAN_REFRESH_EVERY:-40} \
            --proposal-topk-ratio 0.04 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG}"
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
            --proposal-async-queue 5"
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
            --proposal-hessian-refresh-every ${HESSIAN_REFRESH_EVERY:-40} \
            --proposal-topk-ratio 0.01 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG}"
        ;;
    exp3_run2)
        # Exp3 Run 2 — proposed system, K_top=4%, D=1, F=1%
        # F=1% × 768 total experts = ceil(7.68) = 8 experts per fire.
        SCORE_DIAG_LOG="${OUTPUT_BASEPATH}/log/${TAG}.score_diag.jsonl"
        HCOMP_LOG="${OUTPUT_BASEPATH}/log/${TAG}.vhat_vs_h.jsonl"
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode proposal \
            --proposal-selection-mode hessian \
            --proposal-hessian-refresh-every ${HESSIAN_REFRESH_EVERY:-40} \
            --proposal-topk-ratio 0.04 \
            --proposal-nonexpert-fc-W 10 \
            --proposal-async-queue 5 \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG}"
        ;;
    exp3_run3_moc_d4)
        # Exp3 Run 3 — MoC PEC: D=4, layer-wise round-robin (each event FCs all
        # 48 experts of one layer). Non-expert FC every iter (W=1) — original
        # MoC behavior. No score-based selection (sequential mode skips
        # importance tracker).
        # Explicitly disable H refresh (selection-mode=v_hat means H
        # is never read; refresh-every=0 means the estimator is never run).
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode sequential \
            --proposal-selection-mode v_hat \
            --proposal-hessian-refresh-every 0 \
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
DEFAULT_DATA_ROOT="/home/elice/data/slimpajama_6b"
if [ ! -e "${DEFAULT_DATA_ROOT}/slimpajama_6b_train_text_document.idx" ]; then
    DEFAULT_DATA_ROOT="/home/asdf1234/data/slimpajama_6b"
fi
DATA_PATH=${DATA_PATH:-${DEFAULT_DATA_ROOT}/slimpajama_6b_train_text_document}
VALID_DATA_PATH=${VALID_DATA_PATH:-${DEFAULT_DATA_ROOT}/slimpajama_6b_valid_text_document}

data_options=" \
    --vocab-file ${VOCAB_PATH} \
    --merge-file ${MERGE_PATH} \
    --data-path ${DATA_PATH} \
    --valid-data-path ${VALID_DATA_PATH} \
    --data-impl mmap"

megatron_options=" \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 --adam-beta2 0.95 \
    --tensor-model-parallel-size ${TP_SIZE} --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
    --num-experts ${PRMOE_SPEC} \
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
    --log-interval ${LOG_INTERVAL:-10} --eval-interval ${EVAL_INTERVAL:-100} --eval-iters ${EVAL_ITERS:-128} \
    --save-interval 100000 \
    --weight-decay 0.1 --clip-grad 1.0 --hysteresis 2 \
    --num-workers 2 --bf16 --use-flash-attn-v2 --ds_fused_adam \
    --seed ${SEED} \
    ${ACTIVATION_CHECKPOINT:+--checkpoint-activations --deepspeed-activation-checkpointing} \
    ${EXTRA_MEGATRON_FLAGS:-} \
    ${PROPOSAL_FLAGS} \
    ${EXTRA_PROPOSAL_FLAGS:-}"

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

PIPELINE_FLAG=""
if [ "${PP_SIZE}" -eq 1 ]; then
    PIPELINE_FLAG="--no-pipeline-parallel"
fi

deepspeed_options=" \
    --deepspeed --deepspeed_config ${config_json} \
    --pipeline-model-parallel-size ${PP_SIZE} ${PIPELINE_FLAG} \
    --create-moe-param-group"

if [ -n "${NSYS_PROFILE:-}" ]; then
    NSYS_TRACE=${NSYS_TRACE:-cuda,osrt,nvtx}
    NSYS_PREFIX="nsys profile -o ${NSYS_OUT:-/tmp/nsys_run} --force-overwrite true --trace=${NSYS_TRACE} -s none --cudabacktrace=none ${NSYS_EXTRA_ARGS:-}"
else
    NSYS_PREFIX=""
fi
run_cmd="${NSYS_PREFIX} deepspeed --num_gpus ${NUM_GPUS} --master_port ${MASTER_PORT:-29500} ${DIR}/../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options}"
echo "[etc] TAG=${TAG} NUM_SLOTS=${NUM_SLOTS} GPUS=${NUM_GPUS} EP=${EP_PARALLEL_SIZE} PP=${PP_SIZE} LOG=${LOG_FILE}"
eval ${run_cmd} 2>&1 | tee "${LOG_FILE}"
