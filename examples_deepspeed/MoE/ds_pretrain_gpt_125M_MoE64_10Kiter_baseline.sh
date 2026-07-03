#!/usr/bin/env bash
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)
BASE_SCRIPT="${DIR}/ds_pretrain_gpt_125M_MoE64.sh"

DEFAULT_DATA_ROOT="/home/elice/data/slimpajama_6b"
if [ ! -e "${DEFAULT_DATA_ROOT}/slimpajama_6b_train_text_document.idx" ]; then
    DEFAULT_DATA_ROOT="/home/asdf1234/data/slimpajama_6b"
fi

export VOCAB_PATH=${VOCAB_PATH:-"${DIR}/../rebase/gpt2-vocab.json"}
export MERGE_PATH=${MERGE_PATH:-"${DIR}/../rebase/gpt2-merges.txt"}
export DATA_PATH=${DATA_PATH:-"${DEFAULT_DATA_ROOT}/slimpajama_6b_train_text_document"}
export VALID_DATA_PATH=${VALID_DATA_PATH:-"${DEFAULT_DATA_ROOT}/slimpajama_6b_valid_text_document"}

export TRAIN_ITERS=${TRAIN_ITERS:-10000}
export SEQ_LEN=${SEQ_LEN:-2048}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
export WARMUP_ITERS=${WARMUP_ITERS:-100}
export TRAIN_TOKENS=${TRAIN_TOKENS:-$((TRAIN_ITERS * GLOBAL_BATCH_SIZE * SEQ_LEN))}
export WARMUP_TOKENS=${WARMUP_TOKENS:-$((WARMUP_ITERS * GLOBAL_BATCH_SIZE * SEQ_LEN))}
export LR_DECAY_TOKENS=${LR_DECAY_TOKENS:-${TRAIN_TOKENS}}

export NUM_GPUS=${NUM_GPUS:-8}
export NUM_GPUS_PERNODE=${NUM_GPUS_PERNODE:-8}
export MP_SIZE=${MP_SIZE:-1}
export PP_SIZE=${PP_SIZE:-2}
export EP_SIZE=${EP_SIZE:-64}
export MOE_EXPERT_PARALLEL_SIZE=${MOE_EXPERT_PARALLEL_SIZE:-2}
export ZERO_STAGE=${ZERO_STAGE:-0}
export BATCH_SIZE=${BATCH_SIZE:-4}
export ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-false}

DENOM=$((MP_SIZE * PP_SIZE * MOE_EXPERT_PARALLEL_SIZE))
if [ "${DENOM}" -le 0 ] || [ $((NUM_GPUS % DENOM)) -ne 0 ]; then
    echo "Invalid parallelism: NUM_GPUS=${NUM_GPUS}, MP=${MP_SIZE}, PP=${PP_SIZE}, EP=${MOE_EXPERT_PARALLEL_SIZE}" >&2
    exit 2
fi
DP_SIZE=$((NUM_GPUS / DENOM))
if [ "${DP_SIZE}" -ne 2 ]; then
    echo "warning: requested defaults target DP=2, current computed DP=${DP_SIZE}" >&2
fi

export EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
export EVAL_ITERS=${EVAL_ITERS:-20}
export LOG_INTERVAL=${LOG_INTERVAL:-10}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}
export NO_MEGATRON_CKPT=1
export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_125M_MoE64_10Kiter_baseline"}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"${REPO_ROOT}/.torch_extensions"}
export DEEPSPEED_LAUNCH_ARGS=${DEEPSPEED_LAUNCH_ARGS:-"--num_gpus ${NUM_GPUS}"}

# Hard baseline: no proposal flags and no Megatron checkpoint save/load.
export EXTRA_MEGATRON_FLAGS=${BASELINE_EXTRA_MEGATRON_FLAGS:-}

echo "[10K-baseline] dataset=${DATA_PATH}"
echo "[10K-baseline] dp=${DP_SIZE} pp=${PP_SIZE} ep=${MOE_EXPERT_PARALLEL_SIZE} experts=${EP_SIZE} gpus=${NUM_GPUS}"
echo "[10K-baseline] train_iters=${TRAIN_ITERS} warmup_iters=${WARMUP_ITERS} warmup_tokens=${WARMUP_TOKENS} decay_tokens=${LR_DECAY_TOKENS}"
echo "[10K-baseline] proposal=off megatron_ckpt=off"

exec bash "${BASE_SCRIPT}"
