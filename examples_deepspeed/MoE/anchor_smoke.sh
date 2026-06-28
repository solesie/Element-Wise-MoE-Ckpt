#!/bin/bash
# anchor_smoke.sh — fast anchor-path smoke test (NOT a contract gate).
#
# Tiny MoE (2 layers / 8 experts / short seq) that fires the H-quantile anchor
# at iter 2 so the anchor accumulation + h_q build can be iterated in ~1 min
# instead of the full verify_vram_run.sh (~11 min). Exits after a few iters.
#
# Pass ZERO_STAGE / GRAD_ACCUM_DTYPE via env to exercise fp16_zero0 / bf16_zero0
# / zero12 optimizer layouts. Defaults to ZeRO-0 (FP16_Optimizer / fp16_zero0).
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$DIR/../.." && pwd)
OUT="$REPO_ROOT/results/exp_ANCHOR_SMOKE"
rm -rf "$OUT"; mkdir -p "$OUT"/log "$OUT"/cp

env NUM_LAYERS=2 HIDDEN_SIZE=256 NUM_ATTN_HEADS=4 \
    GLOBAL_BATCH_SIZE=4 BATCH_SIZE=2 SEQ_LEN=512 \
    MOE_TOPK=1 NUM_EXPERTS=8 MOE_TRAIN_CAP=1.25 MOE_EVAL_CAP=1.25 \
    LR_WARMUP_ITERS=2 TRAIN_ITERS=2000 \
    FAULT_INTERVAL=0 \
    HESSIAN_REFRESH_EVERY=2 \
    BLOCK_QUANT_G=8 NONEXPERT_FC_W=1 CP_TARGET=test_dram \
    EXTRA_MEGATRON_FLAGS="--exit-interval 5 --eval-interval 999999 --eval-iters 0 --log-interval 1" \
    OUTPUT_BASEPATH=$OUT SEED=2024 \
    VARIANT=verify_vram \
    bash "$DIR/proposal_test_template.sh" > "$OUT/run.log" 2>&1 || true

echo "=== anchor fire / crash signal ==="
grep -aE "HRefresh|h_q|anchor|Fp32StateMapper mode|RuntimeError|Traceback|Error" "$OUT/run.log" \
    | grep -aivE "_dynamo|sharded_moe|one_hot|no error" | tail -25
echo "=== reached iters ==="
grep -aE "iteration +[0-9]+/" "$OUT/run.log" | tail -3
rm -rf "$OUT/cp"
