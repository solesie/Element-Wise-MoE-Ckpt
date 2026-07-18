#!/bin/bash
# Long, fail-closed R1-R4 allocator audit.
#
# The baseline and proposal use the same 125M model/data configuration and
# write separate JSONL files.  Proposal runs long enough to provide at least
# 100 non-anchor steady samples and six warm H refreshes (1 is the cold
# anchor; 20,40,...,120 are repeated warm anchors).  Exact allocator history
# is restricted to proposal checkpoint dispatches 111-113 and each H refresh;
# ordinary iterations only sample counters and do not synchronize the GPU.
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$DIR/../.." && pwd)
OUT=${OUTPUT_BASEPATH:-"$REPO_ROOT/results/exp_VRAM_VERIFY"}
rm -rf "$OUT"
mkdir -p "$OUT/baseline" "$OUT/proposal"

COMMON_ENV=(
    NUM_LAYERS=12
    HIDDEN_SIZE=768
    NUM_ATTN_HEADS=12
    SEQ_LEN=2048
    GLOBAL_BATCH_SIZE=8
    BATCH_SIZE=2
    NUM_GPUS=${NUM_GPUS:-1}
    TP_SIZE=${TP_SIZE:-1}
    PP_SIZE=${PP_SIZE:-1}
    EP_PARALLEL_SIZE=${EP_PARALLEL_SIZE:-1}
    ZERO_STAGE=${ZERO_STAGE:-0}
    MOE_TOPK=1
    NUM_EXPERTS=48
    MOE_TRAIN_CAP=1.25
    MOE_EVAL_CAP=1.25
    LR_WARMUP_ITERS=130
    TRAIN_ITERS=130
    FAULT_INTERVAL=0
    HESSIAN_REFRESH_EVERY=20
    HESSIAN_REFRESH_WARMUP_EVERY=20
    BLOCK_QUANT_G=8
    NONEXPERT_FC_W=1
    CP_TARGET=disk
    PROPOSAL_ROUTING_FREEZE_ENABLE=1
    EXTRA_MEGATRON_FLAGS=--eval-interval\ 999999\ --eval-iters\ 0\ --log-interval\ 10\ --proposal-test-skip-expert-commit\ --proposal-test-skip-nonexpert-commit
    SEED=2024
)

BASELINE_AUDIT="$OUT/baseline/vram_audit.jsonl"
PROPOSAL_AUDIT="$OUT/proposal/vram_audit.jsonl"

echo "============== baseline: 130 iterations =============="
env "${COMMON_ENV[@]}" \
    PROPOSAL_VRAM_AUDIT=1 \
    PROPOSAL_VRAM_AUDIT_LOG="$BASELINE_AUDIT" \
    PROPOSAL_AUDIT_RUN_KIND=baseline \
    PROPOSAL_PER_ITER_ALLOC_TRACE=0 \
    OUTPUT_BASEPATH="$OUT/baseline" \
    VARIANT=baseline \
    bash "$DIR/proposal_test_template.sh" \
    >"$OUT/baseline/run.log" 2>&1

echo "============== proposal: 130 iterations =============="
env "${COMMON_ENV[@]}" \
    PROPOSAL_VRAM_AUDIT=1 \
    PROPOSAL_VRAM_AUDIT_LOG="$PROPOSAL_AUDIT" \
    PROPOSAL_AUDIT_RUN_KIND=proposal \
    PROPOSAL_PER_ITER_ALLOC_TRACE=1 \
    PROPOSAL_ALLOC_TRACE_ITERS=111,112,113 \
    PROPOSAL_ALLOC_TRACE_MAX_ENTRIES=${PROPOSAL_ALLOC_TRACE_MAX_ENTRIES:-100000} \
    PROPOSAL_H_REFRESH_TRACE_MAX_ENTRIES=${PROPOSAL_H_REFRESH_TRACE_MAX_ENTRIES:-10000} \
    PROPOSAL_ANCHOR_FULL_DIAG=0 \
    PROPOSAL_MB_TIMING=0 \
    EXTRA_PROPOSAL_FLAGS= \
    OUTPUT_BASEPATH="$OUT/proposal" \
    VARIANT=verify_vram \
    bash "$DIR/proposal_test_template.sh" \
    >"$OUT/proposal/run.log" 2>&1

echo "============== R1-R4 fail-closed evaluation =============="
cd "$REPO_ROOT"
python3 megatron/proposal/tests/vram_audit_eval.py \
    "$PROPOSAL_AUDIT" --baseline "$BASELINE_AUDIT"
