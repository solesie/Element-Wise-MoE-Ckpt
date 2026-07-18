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
export WARMUP_ITERS=${WARMUP_ITERS:-80}
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
export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_350M_MoE128_10Kiter_proposal"}
DEFAULT_FINAL_BF16_MODEL_PATH=""
if [ "${TRAIN_ITERS}" -eq 10000 ]; then
    DEFAULT_FINAL_BF16_MODEL_PATH="${OUTPUT_BASEPATH}/bf16_model"
fi
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH-${DEFAULT_FINAL_BF16_MODEL_PATH}}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"${REPO_ROOT}/.torch_extensions"}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export MEGATRON_FUSED_KERNEL_ARCH_LIST=${MEGATRON_FUSED_KERNEL_ARCH_LIST:-90}
export DEEPSPEED_LAUNCH_ARGS=${DEEPSPEED_LAUNCH_ARGS:-"--num_gpus ${NUM_GPUS}"}

export PROPOSAL_IO_BACKEND=${PROPOSAL_IO_BACKEND:-file}
export PROPOSAL_CP_TARGET=${PROPOSAL_CP_TARGET:-test_dram}
export PROPOSAL_CP_DIR=${PROPOSAL_CP_DIR:-"${OUTPUT_BASEPATH}/proposal_cp"}
# Logical write-only trace for offline SSD WAF replay. Empty disables it.
export PROPOSAL_IO_BACKLOG_TRACE_DIR=${PROPOSAL_IO_BACKLOG_TRACE_DIR-"${OUTPUT_BASEPATH}/waf_trace"}
export PROPOSAL_TOPK=${PROPOSAL_TOPK:-0.04}
export LFCL_PLAN_POLICY=${LFCL_PLAN_POLICY:-lossy_log_size}
export HESSIAN_REFRESH_EVERY=${HESSIAN_REFRESH_EVERY:-100}
export FAULT_INTERVAL=${FAULT_INTERVAL:-100}
export FAULT_WARMUP=${FAULT_WARMUP:-0}
export FAULT_LOG=${FAULT_LOG:-"${OUTPUT_BASEPATH}/log/proposal_faults.jsonl"}

PROPOSAL_ORACLE_ENABLE=${PROPOSAL_ORACLE_ENABLE:-1}
case "${PROPOSAL_ORACLE_ENABLE}" in
    1|true|TRUE|yes|YES|on|ON) PROPOSAL_ORACLE_ENABLE=1 ;;
    *) PROPOSAL_ORACLE_ENABLE=0 ;;
esac
export PROPOSAL_ORACLE_ENABLE
export ORACLE_INTERVAL=${ORACLE_INTERVAL:-1000}
ORACLE_ITERS=${ORACLE_ITERS:-}
if [ "${PROPOSAL_ORACLE_ENABLE}" = "1" ] && [ "${ORACLE_INTERVAL}" -gt 0 ] && [ -z "${ORACLE_ITERS}" ]; then
    for ((iter = ORACLE_INTERVAL; iter <= TRAIN_ITERS; iter += ORACLE_INTERVAL)); do
        ORACLE_ITERS="${ORACLE_ITERS:+${ORACLE_ITERS},}${iter}"
    done
fi

EXTRA_PROPOSAL_FLAGS=" \
    --proposal-enable \
    --proposal-cp-mode proposal \
    --proposal-cp-target ${PROPOSAL_CP_TARGET} \
    --proposal-cp-dir ${PROPOSAL_CP_DIR} \
    --proposal-topk-ratio ${PROPOSAL_TOPK} \
    --proposal-e-hat-proxy-off \
    --proposal-nonexpert-fc-W ${NONEXPERT_FC_W:-1} \
    --proposal-lfcl-plan-policy ${LFCL_PLAN_POLICY} \
    --proposal-hessian-refresh-every ${HESSIAN_REFRESH_EVERY} \
    --proposal-fault-interval ${FAULT_INTERVAL} \
    --proposal-fault-warmup ${FAULT_WARMUP} \
    --proposal-fault-log ${FAULT_LOG}"

if [ "${PROPOSAL_ORACLE_ENABLE}" = "1" ] && [ -n "${ORACLE_ITERS}" ]; then
    EXTRA_PROPOSAL_FLAGS="${EXTRA_PROPOSAL_FLAGS} \
        --proposal-independent-oracle-enable \
        --proposal-independent-oracle-iters ${ORACLE_ITERS} \
        --proposal-independent-oracle-log-dir ${ORACLE_LOG_DIR:-${OUTPUT_BASEPATH}/oracle} \
        --proposal-independent-oracle-rank-sample ${ORACLE_RANK_SAMPLE:-262144}"
fi

export EXTRA_MEGATRON_FLAGS="--use-flash-attn-v2 ${EXTRA_MEGATRON_FLAGS:-} ${EXTRA_PROPOSAL_FLAGS}"

echo "[350M-10K-proposal] dataset=${DATA_PATH}"
echo "[350M-10K-proposal] dp=${DP_SIZE} pp=${PP_SIZE} ep=${MOE_EXPERT_PARALLEL_SIZE} experts=${EP_SIZE} gpus=${NUM_GPUS}"
echo "[350M-10K-proposal] iters=${TRAIN_ITERS} tokens=${TRAIN_TOKENS} warmup=${WARMUP_TOKENS} decay=${LR_DECAY_TOKENS}"
echo "[350M-10K-proposal] micro_batch=${BATCH_SIZE} zero=${ZERO_STAGE} activation_checkpoint=${ACTIVATION_CHECKPOINT} precision=${TRAIN_PRECISION}"
echo "[350M-10K-proposal] eval=${EVAL_INTERVAL}x${EVAL_ITERS} cp_target=${PROPOSAL_CP_TARGET} h_refresh=${HESSIAN_REFRESH_EVERY} fault=${FAULT_INTERVAL} oracle=${ORACLE_INTERVAL}:${ORACLE_ITERS}"
echo "[350M-10K-proposal] waf_trace=${PROPOSAL_IO_BACKLOG_TRACE_DIR:-disabled}"
echo "[350M-10K-proposal] final_bf16_model=${FINAL_BF16_MODEL_PATH:-disabled}"

exec bash "${BASE_SCRIPT}"
