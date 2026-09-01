#!/usr/bin/env bash
set -euo pipefail

# 1% experiment profile. Model, optimizer, parallelism, recovery, oracle, and
# final-save behavior remain owned by the canonical proposal script below.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)

export PROPOSAL_TOPK=${PROPOSAL_TOPK:-0.01}
export PROPOSAL_LFCL_FRACTION_PCT=${PROPOSAL_LFCL_FRACTION_PCT:-0.3}
export PROPOSAL_AGE_FORCE_PERIOD=${PROPOSAL_AGE_FORCE_PERIOD:-512}
export WARMUP_ITERS=${WARMUP_ITERS:-80}
export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_350M_MoE128_10Kiter_proposal_1pct"}
# The 1% profile is also the entry point for the 150-iteration one-shot. Save
# BF16 there by default as well; FINAL_BF16_MODEL_PATH='' still disables it.
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH-"${OUTPUT_BASEPATH}/bf16_model"}

# The canonical 10K profile samples the oracle every 1K iterations. A short
# smoke (for example TRAIN_ITERS=150) would otherwise schedule no oracle at all.
# Respect an explicitly supplied ORACLE_ITERS, including an explicit empty value.
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

exec bash "${DIR}/ds_pretrain_gpt_350M_MoE128_10Kiter_proposal.sh"
