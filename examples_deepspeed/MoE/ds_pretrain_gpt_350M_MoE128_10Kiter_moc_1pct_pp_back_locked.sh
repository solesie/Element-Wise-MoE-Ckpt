#!/usr/bin/env bash
set -euo pipefail

# Topology-derived MoC stress policy: globally checkpoint 1% of physical
# expert slots per iteration, deepest layers first. Faults at every 100 steps
# repeat the same PP/layer age skew without a learned full-state bootstrap.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)

export MOC_SEQUENTIAL_POLICY=expert_rr_pp_back_locked
export MOC_LFCL_INTERVAL_ITERS=1
export MOC_LFCL_FRACTION_PCT=1.0
export FAULT_INTERVAL=${FAULT_INTERVAL:-100}
export FAULT_WARMUP=${FAULT_WARMUP:-0}
export WARMUP_ITERS=${WARMUP_ITERS:-80}
export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_350M_MoE128_10Kiter_moc_1pct_pp_back_locked"}
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH-"${OUTPUT_BASEPATH}/bf16_model"}

exec bash "${DIR}/ds_pretrain_gpt_350M_MoE128_10Kiter_moc.sh"
