#!/bin/bash
MODE=${MODE:-zero0_ep}
# Valid MODE values:
#   zero0_ep      - 2 GPUs, ZeRO-0 + expert parallelism
#   zero0_dp      - 2 GPUs, ZeRO-0 + data parallelism
#   zero0_dp_tiny - 2 GPUs, ZeRO-0 + data parallelism, reduced model
#   zero0_pp      - 2 GPUs, ZeRO-0 + pipeline parallelism
#   zero2_ep      - 2 GPUs, ZeRO-2 + expert parallelism
#   zero0_single  - 1 GPU, ZeRO-0

set -euo pipefail

if [ "${PROPOSAL_IO_BACKEND:-fdp}" = "fdp" ] && [ "${EUID}" -ne 0 ]; then
    exec sudo -E env \
        PATH="/home/elice/.local/bin:${PATH}" \
        LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        PYTHONPATH="${PYTHONPATH:-}" \
        PROPOSAL_IO_BACKEND="${PROPOSAL_IO_BACKEND:-fdp}" \
        MODE="${MODE}" \
        bash "$0" "$@"
fi

DIR=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT="${DIR}/output/ssd_uring_recovery_smoke"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_ROOT}/${MODE}_${STAMP}"
MASTER_PORT=$((29500 + $$ % 1000))

cleanup_output_owner() {
    if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ] && [ -d "${OUT}" ]; then
        chown -R "${SUDO_USER}:${SUDO_USER}" "${OUT}" || true
    fi
}
trap cleanup_output_owner EXIT

case "${MODE}" in
    zero0_ep)
        NUM_GPUS=2
        ZERO_STAGE=0
        PP_SIZE=1
        EP_PARALLEL_SIZE=2
        ;;
    zero0_dp)
        NUM_GPUS=2
        ZERO_STAGE=0
        PP_SIZE=1
        EP_PARALLEL_SIZE=1
        ;;
    zero0_dp_tiny)
        NUM_GPUS=2
        ZERO_STAGE=0
        PP_SIZE=1
        EP_PARALLEL_SIZE=1
        TINY_MODEL=1
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
        echo "Use one of: zero0_ep, zero0_dp, zero0_dp_tiny, zero0_pp, zero2_ep, zero0_single" >&2
        exit 2
        ;;
esac

TRAIN_ITERS_VALUE=6
EXIT_INTERVAL_VALUE=6
NUM_LAYERS_VALUE=12
HIDDEN_SIZE_VALUE=768
NUM_ATTN_HEADS_VALUE=12
SEQ_LEN_VALUE=512
NUM_EXPERTS_VALUE=64
GLOBAL_BATCH_SIZE_VALUE=4
BATCH_SIZE_VALUE=1

if [ "${TINY_MODEL:-0}" = "1" ]; then
    NUM_LAYERS_VALUE=2
    HIDDEN_SIZE_VALUE=256
    NUM_ATTN_HEADS_VALUE=4
    SEQ_LEN_VALUE=128
    NUM_EXPERTS_VALUE=8
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "${AVAILABLE_GPUS}" -lt "${NUM_GPUS}" ]; then
        echo "MODE=${MODE} needs ${NUM_GPUS} GPU(s), but nvidia-smi sees ${AVAILABLE_GPUS}." >&2
        exit 1
    fi
fi

mkdir -p "${OUT}"

if [ "${PROPOSAL_IO_BACKEND:-fdp}" = "fdp" ]; then
    FDP_BLOCK_DEVICE="${PROPOSAL_FDP_BLOCK_DEVICE:-/dev/nvme0n1}"
    FDP_DEVICE="${PROPOSAL_FDP_DEVICE:-/dev/ng0n1}"
    FDP_BLOCK_NAME=$(basename "${FDP_BLOCK_DEVICE}")
    FDP_SYSFS="/sys/block/${FDP_BLOCK_NAME}"
    if [ ! -e "${FDP_SYSFS}/queue/logical_block_size" ] || [ ! -e "${FDP_SYSFS}/size" ]; then
        echo "FDP block geometry not found for ${FDP_BLOCK_DEVICE} (${FDP_SYSFS})" >&2
        exit 1
    fi
    FDP_LBA_SIZE="${PROPOSAL_FDP_LBA_SIZE:-$(cat "${FDP_SYSFS}/queue/logical_block_size")}"
    FDP_NAMESPACE_NLB="${PROPOSAL_FDP_NAMESPACE_NLB:-$(( $(cat "${FDP_SYSFS}/size") * 512 / FDP_LBA_SIZE ))}"
    FDP_METADATA_START_LBA="${PROPOSAL_FDP_METADATA_START_LBA:-0}"
    FDP_METADATA_NLB="${PROPOSAL_FDP_METADATA_NLB:-$(( 64 * 1024 * 1024 / FDP_LBA_SIZE ))}"
    FDP_DATA_START_LBA="${PROPOSAL_FDP_DATA_START_LBA:-$(( FDP_METADATA_START_LBA + FDP_METADATA_NLB ))}"
    FDP_DATA_NLB="${PROPOSAL_FDP_DATA_NLB:-$(( FDP_NAMESPACE_NLB - FDP_DATA_START_LBA ))}"
    FDP_MAX_IO_KB="${PROPOSAL_FDP_MAX_IO_KB:-$(cat "${FDP_SYSFS}/queue/max_sectors_kb")}"
    if [ ! -e /usr/include/liburing.h ]; then
        echo "liburing development headers are missing. Install with: sudo apt-get install -y liburing-dev" >&2
        exit 1
    fi
fi

echo "[ssd-uring-recovery-smoke] MODE=${MODE}"
echo "[ssd-uring-recovery-smoke] OUT=${OUT}"
echo "[ssd-uring-recovery-smoke] ZeRO=${ZERO_STAGE} GPUS=${NUM_GPUS} PP=${PP_SIZE} EP=${EP_PARALLEL_SIZE}"
if [ "${PROPOSAL_IO_BACKEND:-fdp}" = "fdp" ]; then
    echo "[ssd-uring-recovery-smoke] FDP device=${FDP_DEVICE} metadata=[${FDP_METADATA_START_LBA}, $((FDP_METADATA_START_LBA + FDP_METADATA_NLB))) data=[${FDP_DATA_START_LBA}, $((FDP_DATA_START_LBA + FDP_DATA_NLB)))"
    echo "[ssd-uring-recovery-smoke] FDP readback validation enabled; force_format=${PROPOSAL_FDP_FORCE_FORMAT:-1}"
fi

COMMON_ENV=(
    VARIANT=iter_time_debug
    OUTPUT_BASEPATH="${OUT}"
    MASTER_PORT="${MASTER_PORT}"
    CP_TARGET=disk
    SELECTION_MODE=hessian
    PROPOSAL_TOPK=0.04
    NONEXPERT_FC_W=1
    ASYNC_QUEUE=5
    FAULT_INTERVAL=4
    FAULT_WARMUP=0
    HESSIAN_REFRESH_EVERY=100000
    TRAIN_ITERS="${TRAIN_ITERS_VALUE}"
    SEED=2024
    NUM_LAYERS="${NUM_LAYERS_VALUE}"
    HIDDEN_SIZE="${HIDDEN_SIZE_VALUE}"
    NUM_ATTN_HEADS="${NUM_ATTN_HEADS_VALUE}"
    SEQ_LEN="${SEQ_LEN_VALUE}"
    NUM_EXPERTS="${NUM_EXPERTS_VALUE}"
    MOE_TOPK=1
    MOE_TRAIN_CAP=1.25
    MOE_EVAL_CAP=1.25
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE_VALUE}"
    BATCH_SIZE="${BATCH_SIZE_VALUE}"
    LR=4.5e-4
    MIN_LR=4.5e-6
    LR_WARMUP_ITERS=1
    LOG_INTERVAL=1
    EVAL_INTERVAL=999999
    EVAL_ITERS=0
    EXTRA_MEGATRON_FLAGS="--exit-interval ${EXIT_INTERVAL_VALUE}"
    EXTRA_PROPOSAL_FLAGS=""
    NSYS_PROFILE=""
    TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${OUT}/torch_extensions}"
    PROPOSAL_FDP_VALIDATE_READBACK=1
)

if [ "${PROPOSAL_IO_BACKEND:-fdp}" = "fdp" ]; then
    COMMON_ENV+=(
        PROPOSAL_IO_BACKEND=fdp
        PROPOSAL_FDP_DEVICE="${FDP_DEVICE}"
        PROPOSAL_FDP_NSID="${PROPOSAL_FDP_NSID:-1}"
        PROPOSAL_FDP_LBA_SIZE="${FDP_LBA_SIZE}"
        PROPOSAL_FDP_NAMESPACE_NLB="${FDP_NAMESPACE_NLB}"
        PROPOSAL_FDP_ASSUME_FDP_SUPPORTED="${PROPOSAL_FDP_ASSUME_FDP_SUPPORTED:-1}"
        PROPOSAL_FDP_METADATA_MMAP="${PROPOSAL_FDP_METADATA_MMAP:-1}"
        PROPOSAL_FDP_METADATA_START_LBA="${FDP_METADATA_START_LBA}"
        PROPOSAL_FDP_METADATA_NLB="${FDP_METADATA_NLB}"
        PROPOSAL_FDP_DATA_START_LBA="${FDP_DATA_START_LBA}"
        PROPOSAL_FDP_DATA_NLB="${FDP_DATA_NLB}"
        PROPOSAL_FDP_CHUNK_MB="${PROPOSAL_FDP_CHUNK_MB:-64}"
        PROPOSAL_FDP_MAX_IO_KB="${FDP_MAX_IO_KB}"
        PROPOSAL_FDP_NUM_MOE_LAYERS="${PROPOSAL_FDP_NUM_MOE_LAYERS:-${NUM_LAYERS_VALUE}}"
        PROPOSAL_FDP_EXPERTS_PER_LAYER="${PROPOSAL_FDP_EXPERTS_PER_LAYER:-${NUM_EXPERTS_VALUE}}"
        PROPOSAL_FDP_ALLOW_FORMAT="${PROPOSAL_FDP_ALLOW_FORMAT:-1}"
        PROPOSAL_FDP_FORCE_FORMAT="${PROPOSAL_FDP_FORCE_FORMAT:-1}"
    )
fi

env "${COMMON_ENV[@]}" \
    NUM_GPUS="${NUM_GPUS}" \
    ZERO_STAGE="${ZERO_STAGE}" \
    PP_SIZE="${PP_SIZE}" \
    EP_PARALLEL_SIZE="${EP_PARALLEL_SIZE}" \
    bash "${DIR}/proposal_test_template.sh"

LOG_FILE="${OUT}/log/iter_time_debug_seed2024.log"
if ! grep -q "FDP readback validation OK" "${LOG_FILE}"; then
    echo "[ssd-uring-recovery-smoke] missing FDP readback validation marker in ${LOG_FILE}" >&2
    exit 1
fi
if ! grep -q "\\[proposal\\] FAULT@4" "${LOG_FILE}"; then
    echo "[ssd-uring-recovery-smoke] missing recovery fault marker in ${LOG_FILE}" >&2
    exit 1
fi

echo "[ssd-uring-recovery-smoke] done"
echo "[ssd-uring-recovery-smoke] log dir: ${OUT}/log"
echo "[ssd-uring-recovery-smoke] checkpoint dir: ${OUT}/cp"
