#!/usr/bin/env bash
set -euo pipefail

# ~1.0B-parameter scaled-down Mixtral-style baseline.  Training, data,
# optimizer, LR schedule, topology, and batch settings match the existing
# 12.5K baseline; only the model architecture changes.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)

export NUM_GPUS=4
export NUM_GPUS_PERNODE=4
export MP_SIZE=1
export PP_SIZE=1
export MOE_EXPERT_PARALLEL_SIZE=4

# Model: approximately 1.00B total parameters and 0.32B active parameters.
export MODEL_SIZE=1.0
export EP_SIZE=8
export NUM_LAYERS=24
export HIDDEN_SIZE=768
export NUM_ATTN_HEADS=12
export EXPERT_INTERVAL=1
export BASELINE_EXTRA_MEGATRON_FLAGS="--topk 2 --expert-interval ${EXPERT_INTERVAL}"

export GLOBAL_BATCH_SIZE=24
export BATCH_SIZE=6
export SEQ_LEN=2048

export TRAIN_ITERS=12500
export TRAIN_TOKENS=614400000
export WARMUP_ITERS=80
export WARMUP_TOKENS=3932160
export LR_DECAY_TOKENS=491520000
export SAVE_INTERVAL=1000000

export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_mixtral_1B_MoE8_12p5Kiter_baseline_pp1_ep4_mbs6_gbs24"}
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH-"${OUTPUT_BASEPATH}/bf16_model"}

echo "[mixtral-1B-MoE8-12p5K-baseline] layers=24 hidden=768 experts=8 topk=2 expert_interval=1"
echo "[mixtral-1B-MoE8-12p5K-baseline] warmup=80 decay_end=10000 min_lr_tail=2500"

exec bash "${DIR}/ds_pretrain_gpt_125M_MoE64_10Kiter_baseline.sh"
