#!/usr/bin/env bash
set -euo pipefail

# One layer-wise event covers 256 / 1536 = 16.6667% of global expert slots
# under PP2/EP4. Firing every 16 iterations therefore averages 1.0417%/iter,
# matching the existing every-4-iterations 4.1667% convention.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)

export MOC_LFCL_INTERVAL_ITERS=${MOC_LFCL_INTERVAL_ITERS:-16}
export WARMUP_ITERS=${WARMUP_ITERS:-80}
export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_350M_MoE128_10Kiter_moc_1pct"}
# Keep short one-shot runs save-verifiable without changing the canonical 4%
# script. An explicitly empty value remains the opt-out.
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH-"${OUTPUT_BASEPATH}/bf16_model"}

exec bash "${DIR}/ds_pretrain_gpt_350M_MoE128_10Kiter_moc.sh"
