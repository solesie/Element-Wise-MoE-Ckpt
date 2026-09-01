#!/usr/bin/env bash
set -euo pipefail

# Same deterministic PP/layer stress order as the 1% launcher, with an exact
# cumulative 4% global expert-slot budget per iteration.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)

export MOC_SEQUENTIAL_POLICY=expert_rr_pp_back_locked
export MOC_LFCL_INTERVAL_ITERS=1
export MOC_LFCL_FRACTION_PCT=4.0
export FAULT_INTERVAL=${FAULT_INTERVAL:-100}
export FAULT_WARMUP=${FAULT_WARMUP:-0}
export WARMUP_ITERS=${WARMUP_ITERS:-80}
export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_350M_MoE128_10Kiter_moc_4pct_pp_back_locked"}
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH-"${OUTPUT_BASEPATH}/bf16_model"}

exec bash "${DIR}/ds_pretrain_gpt_350M_MoE128_10Kiter_moc.sh"
