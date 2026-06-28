#!/bin/bash
MODE=zero0_single
# Valid MODE values:
#   zero0_ep      - 2 GPUs, ZeRO-0 + expert parallelism
#   zero0_pp      - 2 GPUs, ZeRO-0 + pipeline parallelism
#   zero2_ep      - 2 GPUs, ZeRO-2 + expert parallelism
#   zero0_single  - 1 GPU, ZeRO-0

set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT="${DIR}/output/ssd_uring_write_smoke"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_ROOT}/${MODE}_${STAMP}"
MASTER_PORT=$((29500 + $$ % 1000))

case "${MODE}" in
    zero0_ep)
        NUM_GPUS=2
        ZERO_STAGE=0
        PP_SIZE=1
        EP_PARALLEL_SIZE=2
        ;;
    zero0_pp)
        NUM_GPUS=2
        ZERO_STAGE=0
        PP_SIZE=2
        EP_PARALLEL_SIZE=1
        ;;
    zero2_ep)
        NUM_GPUS=2
        ZERO_STAGE=2
        PP_SIZE=1
        EP_PARALLEL_SIZE=2
        ;;
    zero0_single)
        NUM_GPUS=1
        ZERO_STAGE=0
        PP_SIZE=1
        EP_PARALLEL_SIZE=1
        ;;
    *)
        echo "Unsupported MODE='${MODE}'" >&2
        echo "Use one of: zero0_ep, zero0_pp, zero2_ep, zero0_single" >&2
        exit 2
        ;;
esac

if command -v nvidia-smi >/dev/null 2>&1; then
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "${AVAILABLE_GPUS}" -lt "${NUM_GPUS}" ]; then
        echo "MODE=${MODE} needs ${NUM_GPUS} GPU(s), but nvidia-smi sees ${AVAILABLE_GPUS}." >&2
        exit 1
    fi
fi

mkdir -p "${OUT}"

echo "[ssd-uring-write-smoke] MODE=${MODE}"
echo "[ssd-uring-write-smoke] OUT=${OUT}"
echo "[ssd-uring-write-smoke] ZeRO=${ZERO_STAGE} GPUS=${NUM_GPUS} PP=${PP_SIZE} EP=${EP_PARALLEL_SIZE}"

COMMON_ENV=(
    VARIANT=iter_time_debug
    OUTPUT_BASEPATH="${OUT}"
    MASTER_PORT="${MASTER_PORT}"
    CP_TARGET=disk
    SELECTION_MODE=hessian
    PROPOSAL_TOPK=0.04
    NONEXPERT_FC_W=1
    ASYNC_QUEUE=5
    FAULT_INTERVAL=0
    HESSIAN_REFRESH_EVERY=100000
    TRAIN_ITERS=5
    SEED=2024
    NUM_LAYERS=12
    HIDDEN_SIZE=768
    NUM_ATTN_HEADS=12
    SEQ_LEN=512
    NUM_EXPERTS=64
    MOE_TOPK=1
    MOE_TRAIN_CAP=1.25
    MOE_EVAL_CAP=1.25
    GLOBAL_BATCH_SIZE=4
    BATCH_SIZE=1
    LR=4.5e-4
    MIN_LR=4.5e-6
    LR_WARMUP_ITERS=1
    LOG_INTERVAL=1
    EVAL_INTERVAL=999999
    EVAL_ITERS=0
    EXTRA_MEGATRON_FLAGS="--exit-interval 5"
    EXTRA_PROPOSAL_FLAGS=""
    NSYS_PROFILE=""
)

env "${COMMON_ENV[@]}" \
    NUM_GPUS="${NUM_GPUS}" \
    ZERO_STAGE="${ZERO_STAGE}" \
    PP_SIZE="${PP_SIZE}" \
    EP_PARALLEL_SIZE="${EP_PARALLEL_SIZE}" \
    bash "${DIR}/proposal_test_template.sh"

echo "[ssd-uring-write-smoke] done"
echo "[ssd-uring-write-smoke] log dir: ${OUT}/log"
echo "[ssd-uring-write-smoke] checkpoint dir: ${OUT}/cp"
