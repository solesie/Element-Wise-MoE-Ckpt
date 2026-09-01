#!/usr/bin/env bash
set -euo pipefail

# ~522M-total Mixtral-style 8x125M PEC/MoC 4% back-locked run.
# Model, data, optimizer, LR schedule, topology, and batch settings match the
# corresponding baseline launcher.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)

export NUM_GPUS=4
export NUM_GPUS_PERNODE=4
export MP_SIZE=1
export PP_SIZE=1
export MOE_EXPERT_PARALLEL_SIZE=4

# Model: 125M dense backbone, eight experts, two active experts per token.
export MODEL_SIZE=0.522
export EP_SIZE=8
export NUM_LAYERS=12
export HIDDEN_SIZE=768
export NUM_ATTN_HEADS=12
export EXPERT_INTERVAL=1
export ACTIVATION_CHECKPOINT=false
export MOC_EXTRA_MEGATRON_FLAGS="--topk 2 --ffn-hidden-size 3072"

export GLOBAL_BATCH_SIZE=24
export BATCH_SIZE=6
export SEQ_LEN=2048

export TRAIN_ITERS=12500
export TRAIN_TOKENS=614400000
export WARMUP_ITERS=80
export WARMUP_TOKENS=3932160
export LR_DECAY_TOKENS=491520000
export SAVE_INTERVAL=1000000

# Exact global 4% expert budget: 96 slots / 25 iterations, deepest layer first.
export MOC_SEQUENTIAL_POLICY=expert_rr_pp_back_locked
export MOC_LFCL_INTERVAL_ITERS=1
export MOC_LFCL_FRACTION_PCT=4.0
export FAULT_INTERVAL=100
export FAULT_WARMUP=0
export PROPOSAL_CP_TARGET=test_dram

export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_mixtral_8x125M_12p5Kiter_pec_4pct_pp1_ep4_mbs6_gbs24_back_locked"}
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH-"${OUTPUT_BASEPATH}/bf16_model"}

echo "[mixtral-8x125M-12p5K-pec-4pct] layers=12 hidden=768 ffn=3072 experts=8 topk=2 expert_interval=1"
echo "[mixtral-8x125M-12p5K-pec-4pct] warmup=80 decay_end=10000 min_lr_tail=2500"
echo "[mixtral-8x125M-12p5K-pec-4pct] policy=${MOC_SEQUENTIAL_POLICY} budget=${MOC_LFCL_FRACTION_PCT}% cp_target=${PROPOSAL_CP_TARGET}"

exec bash "${DIR}/ds_pretrain_gpt_125M_MoE64_10Kiter_moc.sh"
