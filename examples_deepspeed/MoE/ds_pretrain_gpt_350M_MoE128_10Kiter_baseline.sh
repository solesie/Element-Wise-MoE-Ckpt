#!/usr/bin/env bash
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)
if [ -x "${REPO_ROOT}/.venv/bin/deepspeed" ]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi
BASE_SCRIPT="${DIR}/ds_pretrain_gpt_350M_MoE128.sh"

DEFAULT_DATA_ROOT=${SLIMPAJAMA_DATA_ROOT:-/home/paul/data/slimpajama_6b}
DEFAULT_TRAIN_PREFIX="${DEFAULT_DATA_ROOT}/slimpajama_6b_train_text_document"
DEFAULT_VALID_PREFIX="${DEFAULT_DATA_ROOT}/slimpajama_6b_valid_text_document"
export VOCAB_PATH=${VOCAB_PATH:-"${DIR}/../rebase/gpt2-vocab.json"}
export MERGE_PATH=${MERGE_PATH:-"${DIR}/../rebase/gpt2-merges.txt"}
export DATA_PATH=${DATA_PATH:-"${DEFAULT_TRAIN_PREFIX}"}
export VALID_DATA_PATH=${VALID_DATA_PATH:-"${DEFAULT_VALID_PREFIX}"}
export DATA_CACHE_PATH=${DATA_CACHE_PATH:-"${REPO_ROOT}/results/data-cache"}
if [ ! -e "${DATA_PATH}.idx" ] || [ ! -e "${DATA_PATH}.bin" ] || \
   [ ! -e "${VALID_DATA_PATH}.idx" ] || [ ! -e "${VALID_DATA_PATH}.bin" ]; then
    echo "SlimPajama 6B mmap dataset is not ready; run dataset/download_slimpajama_6b.sh." >&2
    exit 2
fi

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
export EP_SIZE=${EP_SIZE:-128}
export MOE_EXPERT_PARALLEL_SIZE=${MOE_EXPERT_PARALLEL_SIZE:-4}
export ZERO_STAGE=${ZERO_STAGE:-1}
export BATCH_SIZE=${BATCH_SIZE:-8}
export ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-false}
export TRAIN_PRECISION=${TRAIN_PRECISION:-bf16}
export LR=${LR:-2.0e-4}
export MIN_LR=${MIN_LR:-2e-06}

DENOM=$((MP_SIZE * PP_SIZE * MOE_EXPERT_PARALLEL_SIZE))
if [ "${DENOM}" -le 0 ] || [ $((NUM_GPUS % DENOM)) -ne 0 ]; then
    echo "Invalid parallelism: NUM_GPUS=${NUM_GPUS}, TP=${MP_SIZE}, PP=${PP_SIZE}, EP=${MOE_EXPERT_PARALLEL_SIZE}" >&2
    exit 2
fi
DP_SIZE=$((NUM_GPUS / DENOM))
if [ "${DP_SIZE}" -ne 1 ]; then
    echo "Expected DP=1, got DP=${DP_SIZE}" >&2
    exit 2
fi

export EVAL_INTERVAL=${EVAL_INTERVAL:-100}
export EVAL_ITERS=${EVAL_ITERS:-10}
export LOG_INTERVAL=${LOG_INTERVAL:-10}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}
export NO_MEGATRON_CKPT=1
export NO_TENSORBOARD=${NO_TENSORBOARD:-1}
export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_350M_MoE128_10Kiter_baseline"}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"${REPO_ROOT}/.torch_extensions"}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export MEGATRON_FUSED_KERNEL_ARCH_LIST=${MEGATRON_FUSED_KERNEL_ARCH_LIST:-90}
export DEEPSPEED_LAUNCH_ARGS=${DEEPSPEED_LAUNCH_ARGS:-"--num_gpus ${NUM_GPUS}"}

export EXTRA_MEGATRON_FLAGS="--use-flash-attn-v2 ${BASELINE_EXTRA_MEGATRON_FLAGS:-}"

echo "[350M-10K-baseline] dataset=${DATA_PATH}"
echo "[350M-10K-baseline] dp=${DP_SIZE} pp=${PP_SIZE} ep=${MOE_EXPERT_PARALLEL_SIZE} experts=${EP_SIZE} gpus=${NUM_GPUS}"
echo "[350M-10K-baseline] iters=${TRAIN_ITERS} tokens=${TRAIN_TOKENS} warmup=${WARMUP_TOKENS} decay=${LR_DECAY_TOKENS}"
echo "[350M-10K-baseline] micro_batch=${BATCH_SIZE} zero=${ZERO_STAGE} activation_checkpoint=${ACTIVATION_CHECKPOINT} precision=${TRAIN_PRECISION}"
echo "[350M-10K-baseline] eval_interval=${EVAL_INTERVAL} eval_iters=${EVAL_ITERS} proposal=off"

exec bash "${BASE_SCRIPT}"
