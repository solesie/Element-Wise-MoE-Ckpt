#!/usr/bin/env bash
set -euo pipefail

# 1% proposal profile. Model, optimizer, topology, recovery, oracle, and final
# save behavior remain owned by the canonical 4% proposal script.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)

export PROPOSAL_TOPK=${PROPOSAL_TOPK:-0.01}
export PROPOSAL_LFCL_FRACTION_PCT=${PROPOSAL_LFCL_FRACTION_PCT:-0.3}
export PROPOSAL_AGE_FORCE_PERIOD=${PROPOSAL_AGE_FORCE_PERIOD:-512}
export WARMUP_ITERS=${WARMUP_ITERS:-80}
export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal_1pct"}
# Save BF16 for both the 10K experiment and explicit short validation runs.
# FINAL_BF16_MODEL_PATH='' remains the opt-out.
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH-"${OUTPUT_BASEPATH}/bf16_model"}

# A short smoke would otherwise schedule no canonical 1K oracle sample.
PROFILE_TRAIN_ITERS=${TRAIN_ITERS:-10000}
PROFILE_ORACLE_INTERVAL=${ORACLE_INTERVAL:-1000}
if [ "${PROFILE_TRAIN_ITERS}" -le 200 ]; then
    export PROPOSAL_APPROX_PROBE_TELEMETRY_LOG=${PROPOSAL_APPROX_PROBE_TELEMETRY_LOG-"${OUTPUT_BASEPATH}/log/approx_probe_rank0.jsonl"}
    export PROPOSAL_APPROX_PROBE_TELEMETRY_DETAIL=${PROPOSAL_APPROX_PROBE_TELEMETRY_DETAIL:-1}
    export PROPOSAL_LFCL_FIRE_LOG=${PROPOSAL_LFCL_FIRE_LOG-"${OUTPUT_BASEPATH}/log/proposal_lfcl_rank{rank}.jsonl"}
fi
if [ -z "${ORACLE_ITERS+x}" ] && \
   [ "${PROFILE_TRAIN_ITERS}" -lt "${PROFILE_ORACLE_INTERVAL}" ]; then
    PROFILE_ORACLE_SMOKE_ITER=${ORACLE_SMOKE_ITER:-100}
    if [ "${PROFILE_ORACLE_SMOKE_ITER}" -gt 0 ] && \
       [ "${PROFILE_ORACLE_SMOKE_ITER}" -le "${PROFILE_TRAIN_ITERS}" ]; then
        export ORACLE_ITERS=${PROFILE_ORACLE_SMOKE_ITER}
    fi
fi

exec bash "${DIR}/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal.sh"
