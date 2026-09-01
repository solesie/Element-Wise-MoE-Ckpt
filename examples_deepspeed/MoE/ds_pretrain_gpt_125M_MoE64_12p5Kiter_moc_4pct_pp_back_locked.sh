#!/usr/bin/env bash
set -euo pipefail

# 125M base / MoE64 (~2B total) counterpart of the 350M/MoE128 back-locked
# MoC run.  The first 10K iterations use the canonical cosine decay and the
# final 2.5K remain at min LR.  LR warmup is fixed at 80 iterations.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)

export NUM_GPUS=8
export NUM_GPUS_PERNODE=8
export MP_SIZE=1
export PP_SIZE=2
export MOE_EXPERT_PARALLEL_SIZE=4
export EP_SIZE=64
export NUM_LAYERS=12
export EXPERT_INTERVAL=2
export GLOBAL_BATCH_SIZE=256
export BATCH_SIZE=8
export SEQ_LEN=2048

export TRAIN_ITERS=12500
export TRAIN_TOKENS=6553600000
export WARMUP_ITERS=80
export WARMUP_TOKENS=41943040
export LR_DECAY_TOKENS=5242880000
export SAVE_INTERVAL=1000000

export MOC_SEQUENTIAL_POLICY=expert_rr_pp_back_locked
export MOC_LFCL_INTERVAL_ITERS=1
export MOC_LFCL_FRACTION_PCT=4.0
export FAULT_INTERVAL=100
export FAULT_WARMUP=0

export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_125M_MoE64_12p5Kiter_moc_4pct_pp2_ep4_back_locked"}
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH-"${OUTPUT_BASEPATH}/bf16_model"}

NVME_ROOT=${NVME_ROOT:-/mnt/nvme0}
NVME_FULL_CKPT_PATH=${NVME_FULL_CKPT_PATH:-"${NVME_ROOT}/moc125m_moe64_12p5k_4pct_full_checkpoint"}
if ! mountpoint -q "${NVME_ROOT}"; then
    echo "NVMe root is not a mount point: ${NVME_ROOT}" >&2
    exit 2
fi
if [ ! -w "${NVME_ROOT}" ]; then
    echo "NVMe root is not writable: ${NVME_ROOT}" >&2
    exit 2
fi
if [ -e "${NVME_FULL_CKPT_PATH}" ]; then
    echo "Refusing to reuse existing full-checkpoint path: ${NVME_FULL_CKPT_PATH}" >&2
    exit 2
fi

# NO_MEGATRON_CKPT remains enabled in the common launcher.  Supplying only
# --save activates training.py's normal final DeepSpeed checkpoint at 12.5K;
# the huge SAVE_INTERVAL prevents an earlier periodic checkpoint.
export MOC_EXTRA_MEGATRON_FLAGS="--save ${NVME_FULL_CKPT_PATH}"

echo "[125M-12p5K-moc-4pct] warmup=80 decay_end=10000 min_lr_tail=2500"
echo "[125M-12p5K-moc-4pct] final_full_checkpoint=${NVME_FULL_CKPT_PATH}"

exec bash "${DIR}/ds_pretrain_gpt_125M_MoE64_10Kiter_moc.sh"
