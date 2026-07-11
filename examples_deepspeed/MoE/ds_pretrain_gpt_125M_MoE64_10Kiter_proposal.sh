#!/usr/bin/env bash
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)
if [ -x "${REPO_ROOT}/.venv/bin/deepspeed" ]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi
BASE_SCRIPT="${DIR}/ds_pretrain_gpt_125M_MoE64.sh"

export PROPOSAL_IO_BACKEND=${PROPOSAL_IO_BACKEND:-fdp}
export PROPOSAL_FDP_BLOCK_DEVICE=${PROPOSAL_FDP_BLOCK_DEVICE:-/dev/nvme8n1}
export PROPOSAL_FDP_DEVICE=${PROPOSAL_FDP_DEVICE:-/dev/ng8n1}

if [ "${PROPOSAL_IO_BACKEND}" = "fdp" ] && [ "${EUID}" -ne 0 ]; then
    exec sudo -n -E env \
        PATH="${PATH}" \
        LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        PYTHONPATH="${PYTHONPATH:-}" \
        VIRTUAL_ENV="${VIRTUAL_ENV:-}" \
        PROPOSAL_IO_BACKEND="${PROPOSAL_IO_BACKEND}" \
        PROPOSAL_FDP_BLOCK_DEVICE="${PROPOSAL_FDP_BLOCK_DEVICE}" \
        PROPOSAL_FDP_DEVICE="${PROPOSAL_FDP_DEVICE}" \
        PROPOSAL_FDP_VALIDATE_READBACK="${PROPOSAL_FDP_VALIDATE_READBACK:-0}" \
        DATA_PATH="${DATA_PATH:-}" \
        VALID_DATA_PATH="${VALID_DATA_PATH:-}" \
        DATA_CACHE_PATH="${DATA_CACHE_PATH:-}" \
        OUTPUT_BASEPATH="${OUTPUT_BASEPATH:-}" \
        NO_TENSORBOARD="${NO_TENSORBOARD:-}" \
        TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-}" \
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}" \
        MEGATRON_FUSED_KERNEL_ARCH_LIST="${MEGATRON_FUSED_KERNEL_ARCH_LIST:-}" \
        NUM_GPUS="${NUM_GPUS:-}" \
        NUM_GPUS_PERNODE="${NUM_GPUS_PERNODE:-}" \
        MP_SIZE="${MP_SIZE:-}" \
        PP_SIZE="${PP_SIZE:-}" \
        EP_SIZE="${EP_SIZE:-}" \
        MOE_EXPERT_PARALLEL_SIZE="${MOE_EXPERT_PARALLEL_SIZE:-}" \
        ZERO_STAGE="${ZERO_STAGE:-}" \
        TRAIN_PRECISION="${TRAIN_PRECISION:-}" \
        BATCH_SIZE="${BATCH_SIZE:-}" \
        GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}" \
        SEQ_LEN="${SEQ_LEN:-}" \
        TRAIN_ITERS="${TRAIN_ITERS:-}" \
        EVAL_INTERVAL="${EVAL_INTERVAL:-}" \
        EVAL_ITERS="${EVAL_ITERS:-}" \
        PROPOSAL_CP_TARGET="${PROPOSAL_CP_TARGET:-}" \
        PROPOSAL_CP_DIR="${PROPOSAL_CP_DIR:-}" \
        PROPOSAL_TOPK="${PROPOSAL_TOPK:-}" \
        LFCL_PLAN_POLICY="${LFCL_PLAN_POLICY:-}" \
        HESSIAN_REFRESH_EVERY="${HESSIAN_REFRESH_EVERY:-}" \
        HESSIAN_REFRESH_WARMUP_EVERY="${HESSIAN_REFRESH_WARMUP_EVERY:-}" \
        PROPOSAL_ORACLE_ENABLE="${PROPOSAL_ORACLE_ENABLE:-}" \
        ORACLE_INTERVAL="${ORACLE_INTERVAL:-}" \
        ORACLE_ITERS="${ORACLE_ITERS:-}" \
        ORACLE_LOG_DIR="${ORACLE_LOG_DIR:-}" \
        FAULT_INTERVAL="${FAULT_INTERVAL:-}" \
        FAULT_WARMUP="${FAULT_WARMUP:-}" \
        FAULT_LOG="${FAULT_LOG:-}" \
        PROPOSAL_FDP_FORCE_FORMAT="${PROPOSAL_FDP_FORCE_FORMAT:-}" \
        PROPOSAL_FDP_ALLOW_FORMAT="${PROPOSAL_FDP_ALLOW_FORMAT:-}" \
        bash "$0" "$@"
fi

DEFAULT_DATA_ROOT="/home/paul/data/slimpajama_6b"
if [ ! -e "${DEFAULT_DATA_ROOT}/slimpajama_6b_train_text_document.idx" ]; then
    DEFAULT_DATA_ROOT="/home/elice/data/slimpajama_6b"
fi
if [ ! -e "${DEFAULT_DATA_ROOT}/slimpajama_6b_train_text_document.idx" ]; then
    DEFAULT_DATA_ROOT="/home/asdf1234/data/slimpajama_6b"
fi

export VOCAB_PATH=${VOCAB_PATH:-"${DIR}/../rebase/gpt2-vocab.json"}
export MERGE_PATH=${MERGE_PATH:-"${DIR}/../rebase/gpt2-merges.txt"}
export DATA_PATH=${DATA_PATH:-"${DEFAULT_DATA_ROOT}/slimpajama_6b_train_text_document"}
export VALID_DATA_PATH=${VALID_DATA_PATH:-"${DEFAULT_DATA_ROOT}/slimpajama_6b_valid_text_document"}
export DATA_CACHE_PATH=${DATA_CACHE_PATH:-"${REPO_ROOT}/results/data-cache"}

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
export TRAIN_PRECISION=${TRAIN_PRECISION:-bf16}

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
export NO_TENSORBOARD=${NO_TENSORBOARD:-1}
export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal"}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"${REPO_ROOT}/.torch_extensions"}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export MEGATRON_FUSED_KERNEL_ARCH_LIST=${MEGATRON_FUSED_KERNEL_ARCH_LIST:-90}
export DEEPSPEED_LAUNCH_ARGS=${DEEPSPEED_LAUNCH_ARGS:-"--num_gpus ${NUM_GPUS}"}

cleanup_output_owner() {
    if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ] && [ -d "${OUTPUT_BASEPATH}" ]; then
        chown -R "${SUDO_USER}:${SUDO_USER}" "${OUTPUT_BASEPATH}" || true
    fi
}
trap cleanup_output_owner EXIT

if [ "${PROPOSAL_IO_BACKEND}" = "fdp" ]; then
    FDP_BLOCK_NAME=$(basename "${PROPOSAL_FDP_BLOCK_DEVICE}")
    FDP_SYSFS="/sys/block/${FDP_BLOCK_NAME}"
    if [ ! -e "${FDP_SYSFS}/queue/logical_block_size" ] || [ ! -e "${FDP_SYSFS}/size" ]; then
        echo "FDP block geometry not found for ${PROPOSAL_FDP_BLOCK_DEVICE} (${FDP_SYSFS})" >&2
        exit 1
    fi
    if [ ! -e /usr/include/liburing.h ]; then
        echo "liburing development headers are missing. Install liburing-dev before FDP runs." >&2
        exit 1
    fi
    export PROPOSAL_FDP_NSID=${PROPOSAL_FDP_NSID:-1}
    export PROPOSAL_FDP_LBA_SIZE=${PROPOSAL_FDP_LBA_SIZE:-$(cat "${FDP_SYSFS}/queue/logical_block_size")}
    export PROPOSAL_FDP_NAMESPACE_NLB=${PROPOSAL_FDP_NAMESPACE_NLB:-$(( $(cat "${FDP_SYSFS}/size") * 512 / PROPOSAL_FDP_LBA_SIZE ))}
    export PROPOSAL_FDP_METADATA_START_LBA=${PROPOSAL_FDP_METADATA_START_LBA:-0}
    export PROPOSAL_FDP_METADATA_NLB=${PROPOSAL_FDP_METADATA_NLB:-$(( 64 * 1024 * 1024 / PROPOSAL_FDP_LBA_SIZE ))}
    export PROPOSAL_FDP_DATA_START_LBA=${PROPOSAL_FDP_DATA_START_LBA:-$(( PROPOSAL_FDP_METADATA_START_LBA + PROPOSAL_FDP_METADATA_NLB ))}
    export PROPOSAL_FDP_DATA_NLB=${PROPOSAL_FDP_DATA_NLB:-$(( PROPOSAL_FDP_NAMESPACE_NLB - PROPOSAL_FDP_DATA_START_LBA ))}
    export PROPOSAL_FDP_ASSUME_FDP_SUPPORTED=${PROPOSAL_FDP_ASSUME_FDP_SUPPORTED:-1}
    export PROPOSAL_FDP_METADATA_MMAP=${PROPOSAL_FDP_METADATA_MMAP:-1}
    export PROPOSAL_FDP_CHUNK_MB=${PROPOSAL_FDP_CHUNK_MB:-64}
    export PROPOSAL_FDP_MAX_IO_KB=${PROPOSAL_FDP_MAX_IO_KB:-$(cat "${FDP_SYSFS}/queue/max_sectors_kb")}
    export PROPOSAL_FDP_NUM_MOE_LAYERS=${PROPOSAL_FDP_NUM_MOE_LAYERS:-${NUM_LAYERS:-12}}
    export PROPOSAL_FDP_EXPERTS_PER_LAYER=${PROPOSAL_FDP_EXPERTS_PER_LAYER:-${EP_SIZE}}
    export PROPOSAL_FDP_ALLOW_FORMAT=${PROPOSAL_FDP_ALLOW_FORMAT:-1}
    export PROPOSAL_FDP_FORCE_FORMAT=${PROPOSAL_FDP_FORCE_FORMAT:-0}
fi

PROPOSAL_ORACLE_ENABLE=${PROPOSAL_ORACLE_ENABLE:-0}
case "${PROPOSAL_ORACLE_ENABLE}" in
    1|true|TRUE|yes|YES|on|ON) PROPOSAL_ORACLE_ENABLE=1 ;;
    *) PROPOSAL_ORACLE_ENABLE=0 ;;
esac
ORACLE_INTERVAL=${ORACLE_INTERVAL:-1000}
ORACLE_ITERS=${ORACLE_ITERS:-}
if [ "${PROPOSAL_ORACLE_ENABLE}" != "1" ]; then
    ORACLE_ITERS=""
elif [ -z "${ORACLE_ITERS}" ] && [ "${ORACLE_INTERVAL}" -gt 0 ]; then
    for ((iter = ORACLE_INTERVAL; iter <= TRAIN_ITERS; iter += ORACLE_INTERVAL)); do
        if [ -z "${ORACLE_ITERS}" ]; then
            ORACLE_ITERS="${iter}"
        else
            ORACLE_ITERS="${ORACLE_ITERS},${iter}"
        fi
    done
fi
if [ "${PROPOSAL_ORACLE_ENABLE}" = "1" ] && [ -z "${ORACLE_ITERS}" ]; then
    PROPOSAL_ORACLE_ENABLE=0
fi

export PROPOSAL_INDEPENDENT_ORACLE_BOOTSTRAP=${PROPOSAL_INDEPENDENT_ORACLE_BOOTSTRAP:-0}
export PROPOSAL_INDEPENDENT_ORACLE_AUTOGRAD_GOLD=${PROPOSAL_INDEPENDENT_ORACLE_AUTOGRAD_GOLD:-0}
export PROPOSAL_INDEPENDENT_ORACLE_ONESHOT_HVP_ACTUAL=${PROPOSAL_INDEPENDENT_ORACLE_ONESHOT_HVP_ACTUAL:-0}
export PROPOSAL_INDEPENDENT_ORACLE_SELECTED_HVP=${PROPOSAL_INDEPENDENT_ORACLE_SELECTED_HVP:-0}

PROPOSAL_CP_DIR=${PROPOSAL_CP_DIR:-"${OUTPUT_BASEPATH}/proposal_cp"}
ORACLE_LOG_DIR=${ORACLE_LOG_DIR:-"${OUTPUT_BASEPATH}/oracle"}
FAULT_LOG=${FAULT_LOG:-"${OUTPUT_BASEPATH}/log/proposal_faults.jsonl"}
EXTRA_PROPOSAL_FLAGS=" \
    --proposal-enable \
    --proposal-cp-mode proposal \
    --proposal-cp-target ${PROPOSAL_CP_TARGET:-disk} \
    --proposal-cp-dir ${PROPOSAL_CP_DIR} \
    --proposal-topk-ratio ${PROPOSAL_TOPK:-0.04} \
    --proposal-e-hat-proxy-off \
    --proposal-nonexpert-fc-W ${NONEXPERT_FC_W:-1} \
    --proposal-lfcl-plan-policy ${LFCL_PLAN_POLICY:-lossy_log_size} \
    --proposal-hessian-refresh-warmup-every ${HESSIAN_REFRESH_WARMUP_EVERY:-20} \
    --proposal-hessian-refresh-every ${HESSIAN_REFRESH_EVERY:-100} \
    --proposal-fault-interval ${FAULT_INTERVAL:-100} \
    --proposal-fault-warmup ${FAULT_WARMUP:-0} \
    --proposal-fault-log ${FAULT_LOG}"

if [ "${PROPOSAL_ORACLE_ENABLE}" = "1" ]; then
EXTRA_PROPOSAL_FLAGS="${EXTRA_PROPOSAL_FLAGS} \
    --proposal-independent-oracle-enable \
    --proposal-independent-oracle-iters ${ORACLE_ITERS} \
    --proposal-independent-oracle-log-dir ${ORACLE_LOG_DIR} \
    --proposal-independent-oracle-rank-sample ${ORACLE_RANK_SAMPLE:-262144}"
fi

export EXTRA_MEGATRON_FLAGS="${EXTRA_MEGATRON_FLAGS:-} ${EXTRA_PROPOSAL_FLAGS}"

echo "[10K-proposal] dataset=${DATA_PATH}"
echo "[10K-proposal] dp=${DP_SIZE} pp=${PP_SIZE} ep=${MOE_EXPERT_PARALLEL_SIZE} experts=${EP_SIZE} gpus=${NUM_GPUS}"
echo "[10K-proposal] train_iters=${TRAIN_ITERS} warmup_iters=${WARMUP_ITERS} warmup_tokens=${WARMUP_TOKENS} decay_tokens=${LR_DECAY_TOKENS}"
echo "[10K-proposal] validation_interval=${EVAL_INTERVAL} validation_iters=${EVAL_ITERS}"
echo "[10K-proposal] precision=${TRAIN_PRECISION} proposal=4% cp_target=${PROPOSAL_CP_TARGET:-disk} theta_cp=2byte W=1 LFCL=${LFCL_PLAN_POLICY:-lossy_log_size}_1pct fault_interval=${FAULT_INTERVAL:-100} oracle_enable=${PROPOSAL_ORACLE_ENABLE} oracle_interval=${ORACLE_INTERVAL} oracle_iters=${ORACLE_ITERS}"
if [ "${PROPOSAL_IO_BACKEND}" = "fdp" ]; then
    echo "[10K-proposal] fdp_backend=on block=${PROPOSAL_FDP_BLOCK_DEVICE} device=${PROPOSAL_FDP_DEVICE} readback=${PROPOSAL_FDP_VALIDATE_READBACK:-0}"
fi

exec bash "${BASE_SCRIPT}"
