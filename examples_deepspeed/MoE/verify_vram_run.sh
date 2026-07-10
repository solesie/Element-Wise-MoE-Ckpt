#!/bin/bash
# Run proposal test verification: VRAM audit + R1-R4 PASS/FAIL evaluation.
# - test model: 125M MoE-48E, refresh_every=5
# - PROPOSAL_VRAM_AUDIT=1 captures per-phase memory snapshots
# - PROPOSAL_PER_ITER_ALLOC_TRACE=1 traces tensor allocs in proposal hot path
set -e
# Repo-relative paths (portable across servers): this script lives in
# <repo>/examples_deepspeed/MoE, so repo root = two dirs up.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$DIR/../.." && pwd)
OUT="$REPO_ROOT/results/exp_VRAM_VERIFY"
rm -rf "$OUT"; mkdir -p "$OUT"/log "$OUT"/cp
AUDIT_LOG="$OUT/vram_audit.jsonl"

echo "============== run with VRAM audit =============="
env NUM_LAYERS=12 HIDDEN_SIZE=768 NUM_ATTN_HEADS=12 \
    GLOBAL_BATCH_SIZE=8 BATCH_SIZE=2 SEQ_LEN=2048 \
    MOE_TOPK=1 NUM_EXPERTS=48 MOE_TRAIN_CAP=1.25 MOE_EVAL_CAP=1.25 \
    LR_WARMUP_ITERS=50 TRAIN_ITERS=2000 \
    FAULT_INTERVAL=0 \
    HESSIAN_REFRESH_EVERY=5 \
    BLOCK_QUANT_G=8 NONEXPERT_FC_W=1 CP_TARGET=${CP_TARGET:-disk} \
    PROPOSAL_VRAM_AUDIT=1 PROPOSAL_VRAM_AUDIT_LOG="$AUDIT_LOG" \
    PROPOSAL_PER_ITER_ALLOC_TRACE=1 \
    PROPOSAL_ROUTING_FREEZE_ENABLE=1 \
    EXTRA_MEGATRON_FLAGS="--exit-interval 26 --eval-interval 999999 --eval-iters 0 --log-interval 5 --proposal-test-skip-expert-commit --proposal-test-skip-nonexpert-commit" \
    OUTPUT_BASEPATH=$OUT SEED=2024 \
    VARIANT=verify_vram \
    bash "$DIR/proposal_test_template.sh" > "$OUT/run.log" 2>&1 || true

echo "--- run errors (filtered) ---"
grep -E "Error|Traceback|FAIL" "$OUT/run.log" | \
    grep -v "_dynamo\|sharded_moe\|F\.one_hot\|FAIL on tensor" | head -5

echo ""
echo "============== R1-R4 PASS/FAIL evaluation =============="
cd "$REPO_ROOT"
python3 megatron/proposal/tests/vram_audit_eval.py "$AUDIT_LOG"
EVAL_RC=$?

rm -rf "$OUT/cp"
exit $EVAL_RC
