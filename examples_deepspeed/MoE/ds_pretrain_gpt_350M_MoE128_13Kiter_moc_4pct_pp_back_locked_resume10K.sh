#!/usr/bin/env bash
set -euo pipefail

# Continue the completed 10K MoC run from its full DeepSpeed checkpoint.
# The original decay horizon is intentionally retained so every resumed
# optimizer step stays at MIN_LR instead of reopening the cosine schedule.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)

RESUME_ROOT=${RESUME_ROOT:-/mnt/nvme0/moc4pct_full_checkpoint}
EXPECTED_TAG=global_step10000
EXPECTED_START_ITER=10000
EXPECTED_START_TOKENS=5242880000

if [ ! -f "${RESUME_ROOT}/latest" ] || \
   [ "$(<"${RESUME_ROOT}/latest")" != "${EXPECTED_TAG}" ]; then
    echo "Invalid resume checkpoint tracker under ${RESUME_ROOT}" >&2
    exit 2
fi
if [ ! -d "${RESUME_ROOT}/${EXPECTED_TAG}" ]; then
    echo "Missing resume checkpoint tag ${RESUME_ROOT}/${EXPECTED_TAG}" >&2
    exit 2
fi
if [ "$(find "${RESUME_ROOT}/${EXPECTED_TAG}" -maxdepth 1 -type f | wc -l)" -ne 1554 ]; then
    echo "Resume checkpoint shard inventory is incomplete" >&2
    exit 2
fi

export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_350M_MoE128_13Kiter_moc_4pct_pp2_ep4_back_locked_resume10K"}
if [ -e "${OUTPUT_BASEPATH}" ] && [ "${ALLOW_EXISTING_RESUME_OUTPUT:-0}" != "1" ]; then
    echo "Refusing to reuse existing resume output: ${OUTPUT_BASEPATH}" >&2
    echo "Existing logs are never deleted; choose a new OUTPUT_BASEPATH." >&2
    exit 2
fi

export TRAIN_ITERS=13000
export GLOBAL_BATCH_SIZE=256
export SEQ_LEN=2048
export TRAIN_TOKENS=$((TRAIN_ITERS * GLOBAL_BATCH_SIZE * SEQ_LEN))
export WARMUP_ITERS=80
export WARMUP_TOKENS=$((WARMUP_ITERS * GLOBAL_BATCH_SIZE * SEQ_LEN))
export LR_DECAY_TOKENS=${EXPECTED_START_TOKENS}
export LR=2.0e-4
export MIN_LR=2e-06

export NUM_GPUS=8
export NUM_GPUS_PERNODE=8
export MP_SIZE=1
export PP_SIZE=2
export MOE_EXPERT_PARALLEL_SIZE=4
export EP_SIZE=128
export ZERO_STAGE=1
export BATCH_SIZE=8
export ACTIVATION_CHECKPOINT=false
export TRAIN_PRECISION=bf16
export NUM_LAYERS=24
export EXPERT_INTERVAL=2

export MOC_SEQUENTIAL_POLICY=expert_rr_pp_back_locked
export MOC_LFCL_INTERVAL_ITERS=1
export MOC_LFCL_FRACTION_PCT=4.0
export FAULT_INTERVAL=100
export FAULT_WARMUP=0
export EVAL_INTERVAL=100
export EVAL_ITERS=10
export LOG_INTERVAL=10
export SAVE_INTERVAL=1000000
export PROPOSAL_CP_TARGET=test_dram
export PROPOSAL_CP_DIR="${OUTPUT_BASEPATH}/proposal_cp"

# Loading restores model, Adam, scheduler, RNG, sampler position, iteration,
# consumed samples, and consumed tokens.  No terminal BF16/full save is made.
export NO_MEGATRON_CKPT=1
export FINAL_BF16_MODEL_PATH=""
export MOC_EXTRA_MEGATRON_FLAGS="--load ${RESUME_ROOT}"

echo "[350M-13K-moc-resume] load=${RESUME_ROOT}/${EXPECTED_TAG}"
echo "[350M-13K-moc-resume] iter=${EXPECTED_START_ITER}->${TRAIN_ITERS} tokens=${EXPECTED_START_TOKENS}->${TRAIN_TOKENS}"
echo "[350M-13K-moc-resume] lr_decay_tokens=${LR_DECAY_TOKENS} min_lr=${MIN_LR} bf16_save=disabled"
echo "[350M-13K-moc-resume] fault=10100..12900 every=100; boundary 9900->10100 is the sole 200-iter gap"
echo "[350M-13K-moc-resume] output=${OUTPUT_BASEPATH}"

exec bash "${DIR}/ds_pretrain_gpt_350M_MoE128_10Kiter_moc_4pct_pp_back_locked.sh"
