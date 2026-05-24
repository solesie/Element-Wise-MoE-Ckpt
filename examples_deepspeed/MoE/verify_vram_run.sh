#!/bin/bash
# Run verification: VRAM audit + R1-R4 PASS/FAIL evaluation.
# - exp_etc (medium, 226M MoE48), refresh_every=20 (5+ anchors in 130 iter)
# - PROPOSAL_VRAM_AUDIT=1 captures per-phase memory snapshots
# - PROPOSAL_PER_ITER_ALLOC_TRACE=1 traces tensor allocs in proposal hot path
set -e
DIR=/home/asdf1234/Megatron-DeepSpeed/examples_deepspeed/MoE
OUT=/home/asdf1234/Megatron-DeepSpeed/results/exp_VRAM_VERIFY
rm -rf "$OUT"; mkdir -p "$OUT"/log "$OUT"/cp
AUDIT_LOG="$OUT/vram_audit.jsonl"

echo "============== run with VRAM audit =============="
env NUM_LAYERS=12 HIDDEN_SIZE=768 NUM_ATTN_HEADS=12 \
    GLOBAL_BATCH_SIZE=24 BATCH_SIZE=2 SEQ_LEN=2048 \
    MOE_TOPK=1 PRMOE_SPEC=48 MOE_TRAIN_CAP=1.25 MOE_EVAL_CAP=1.25 \
    LR_WARMUP_ITERS=50 TRAIN_ITERS=2000 \
    FAULT_INTERVAL=0 \
    HESSIAN_REFRESH_EVERY=5 HESSIAN_NUM_SAMPLES=1 \
    BLOCK_QUANT_G=8 NONEXPERT_FC_W=1 CP_TARGET=dram \
    PROPOSAL_SKIP_EXPERT_WRITES=1 PROPOSAL_DISK_SKIP_WRITE=1 \
    PROPOSAL_DISPATCH_DBG=1 PROPOSAL_DISPATCH_DBG_EVERY=1 \
    PROPOSAL_VRAM_AUDIT=1 PROPOSAL_VRAM_AUDIT_LOG="$AUDIT_LOG" \
    PROPOSAL_PER_ITER_ALLOC_TRACE=1 \
    PROPOSAL_ROUTING_FREEZE_ENABLE=1 \
    PROPOSAL_FD_NO_OFFLOAD=1 \
    EXTRA_MEGATRON_FLAGS="--exit-interval 30 --eval-interval 999999 --eval-iters 0 --log-interval 5" \
    OUTPUT_BASEPATH=$OUT SEED=2024 \
    VARIANT=proposal_hutch_k1pct_finite_diff_F1 \
    bash "$DIR/proposal_etc_template.sh" > "$OUT/run.log" 2>&1 || true

echo "--- run errors (filtered) ---"
grep -E "Error|Traceback|FAIL" "$OUT/run.log" | \
    grep -v "_dynamo\|sharded_moe\|F\.one_hot\|FAIL on tensor" | head -5

echo ""
echo "============== R1-R4 PASS/FAIL evaluation =============="
cd /home/asdf1234/Megatron-DeepSpeed
python3 megatron/proposal/tests/vram_audit_eval.py "$AUDIT_LOG"
EVAL_RC=$?

rm -rf "$OUT/cp"
exit $EVAL_RC
