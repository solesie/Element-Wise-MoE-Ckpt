#!/usr/bin/env bash
set -euo pipefail

# MoC 1% PP-phase-locked stress profile for the 125M-active / ~2B-total
# MoE64 model. A global layer-major RR writes four complete expert p/m/v
# states per iteration for 96 iterations, then idles for four iterations.
# One 100-iteration frame therefore writes all 384 expert slots exactly once.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)

export MOC_SEQUENTIAL_POLICY=expert_rr_pp_locked
export MOC_LFCL_INTERVAL_ITERS=1
export MOC_LFCL_FRACTION_PCT=1.0
export FAULT_INTERVAL=${FAULT_INTERVAL:-100}
export FAULT_WARMUP=${FAULT_WARMUP:-0}
export WARMUP_ITERS=${WARMUP_ITERS:-80}
export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_125M_MoE64_10Kiter_moc_1pct_expert_rr_pp_locked"}
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH-"${OUTPUT_BASEPATH}/bf16_model"}

exec bash "${DIR}/ds_pretrain_gpt_125M_MoE64_10Kiter_moc.sh"
