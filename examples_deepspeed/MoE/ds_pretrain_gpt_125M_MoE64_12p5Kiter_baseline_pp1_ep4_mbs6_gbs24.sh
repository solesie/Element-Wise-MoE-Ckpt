#!/usr/bin/env bash
set -euo pipefail

# GPT-2B/64E baseline with the paper model settings unchanged.  Only the
# topology, batch sizes, and 12.5K horizon differ from the canonical run.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)

export NUM_GPUS=4
export NUM_GPUS_PERNODE=4
export MP_SIZE=1
export PP_SIZE=1
export MOE_EXPERT_PARALLEL_SIZE=4
export EP_SIZE=64
export NUM_LAYERS=12
export EXPERT_INTERVAL=2
export GLOBAL_BATCH_SIZE=24
export BATCH_SIZE=6
export SEQ_LEN=2048

export TRAIN_ITERS=12500
export TRAIN_TOKENS=614400000
export WARMUP_ITERS=80
export WARMUP_TOKENS=3932160
export LR_DECAY_TOKENS=491520000
export SAVE_INTERVAL=1000000

export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_125M_MoE64_12p5Kiter_baseline_pp1_ep4_mbs6_gbs24"}
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH-"${OUTPUT_BASEPATH}/bf16_model"}

echo "[125M-12p5K-baseline-pp1-ep4-mbs6-gbs24] warmup=80 decay_end=10000 min_lr_tail=2500"

exec bash "${DIR}/ds_pretrain_gpt_125M_MoE64_10Kiter_baseline.sh"
