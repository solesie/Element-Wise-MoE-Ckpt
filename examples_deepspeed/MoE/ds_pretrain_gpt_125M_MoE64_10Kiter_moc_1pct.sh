#!/usr/bin/env bash
set -euo pipefail

# The canonical MoC event covers 2/6 = 33.3333% of expert state under PP2.
# Firing every 32 iterations averages 1.0417%/iter.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)

export MOC_LFCL_INTERVAL_ITERS=${MOC_LFCL_INTERVAL_ITERS:-32}
export WARMUP_ITERS=${WARMUP_ITERS:-80}
export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_125M_MoE64_10Kiter_moc_1pct"}
# Also make short smoke runs save-verifiable; an explicit empty value disables
# the final BF16 model-only save.
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH-"${OUTPUT_BASEPATH}/bf16_model"}

exec bash "${DIR}/ds_pretrain_gpt_125M_MoE64_10Kiter_moc.sh"
