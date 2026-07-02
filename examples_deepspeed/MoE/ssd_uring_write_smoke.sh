#!/bin/bash
MODE=zero0_ep
# Valid MODE values:
#   zero0_ep      - 2 GPUs, ZeRO-0 + expert parallelism
#   zero0_dp      - 2 GPUs, ZeRO-0 + data parallelism
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
        bash "$0" "$@"
fi

DIR=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT="${DIR}/output/ssd_uring_write_smoke"
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
        echo "Use one of: zero0_ep, zero0_dp, zero0_pp, zero2_ep, zero0_single" >&2
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

echo "[ssd-uring-write-smoke] MODE=${MODE}"
echo "[ssd-uring-write-smoke] OUT=${OUT}"
echo "[ssd-uring-write-smoke] ZeRO=${ZERO_STAGE} GPUS=${NUM_GPUS} PP=${PP_SIZE} EP=${EP_PARALLEL_SIZE}"
if [ "${PROPOSAL_IO_BACKEND:-fdp}" = "fdp" ]; then
    echo "[ssd-uring-write-smoke] FDP device=${FDP_DEVICE} metadata=[${FDP_METADATA_START_LBA}, $((FDP_METADATA_START_LBA + FDP_METADATA_NLB))) data=[${FDP_DATA_START_LBA}, $((FDP_DATA_START_LBA + FDP_DATA_NLB)))"
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
    TORCH_EXTENSIONS_DIR="${OUT}/torch_extensions"
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
        PROPOSAL_FDP_NUM_MOE_LAYERS="${PROPOSAL_FDP_NUM_MOE_LAYERS:-12}"
        PROPOSAL_FDP_EXPERTS_PER_LAYER="${PROPOSAL_FDP_EXPERTS_PER_LAYER:-64}"
        PROPOSAL_FDP_ALLOW_FORMAT="${PROPOSAL_FDP_ALLOW_FORMAT:-1}"
        PROPOSAL_FDP_FORCE_FORMAT="${PROPOSAL_FDP_FORCE_FORMAT:-0}"
    )
fi

env "${COMMON_ENV[@]}" \
    NUM_GPUS="${NUM_GPUS}" \
    ZERO_STAGE="${ZERO_STAGE}" \
    PP_SIZE="${PP_SIZE}" \
    EP_PARALLEL_SIZE="${EP_PARALLEL_SIZE}" \
    bash "${DIR}/proposal_test_template.sh"

echo "[ssd-uring-write-smoke] done"
echo "[ssd-uring-write-smoke] log dir: ${OUT}/log"
echo "[ssd-uring-write-smoke] checkpoint dir: ${OUT}/cp"
