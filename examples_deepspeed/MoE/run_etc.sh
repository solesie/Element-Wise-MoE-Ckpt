#!/bin/bash
# ETC 2-way DRAM 2000-iter @ seed 2024
# PR-MoE 4/4/8/8/8/8 + topk=2 + cap=1.25 + fault every 100 iter
# Eval every 100 iter (128 eval iters) → final validation loss recorded.
#
# Each variant writes to its own ``exp_etc_*`` dir (existing naming
# convention; see exp_etc_g8_2000_dram, exp_etc_hutch_k4pct, etc.).
set -e
DIR=$(cd "$(dirname "$0")" && pwd)

run_variant () {
    local V="$1"; local OUT="$2"; shift 2
    mkdir -p "$OUT"/log "$OUT"/cp
    local TAG="${V}_seed2024"
    local LOG="$OUT/log/${TAG}.log"
    if [ -f "$LOG" ] && grep -q "after training is done" "$LOG"; then
        echo "[etc] === Skipping $V → $OUT (already complete) ==="
        return
    fi
    echo "[etc] === Running $V → $OUT ==="
    rm -rf "$OUT/cp/${TAG}" 2>/dev/null
    env "$@" \
        OUTPUT_BASEPATH="$OUT" VARIANT="$V" SEED=2024 TRAIN_ITERS=2000 \
        FAULT_INTERVAL=100 FAULT_WARMUP=300 \
        bash "$DIR/proposal_etc_template.sh"
    echo "[etc] Cleaning CP for ${TAG}..."
    rm -rf "$OUT/cp/${TAG}"
    echo "[etc] === Finished $V ==="
    df -h /home | tail -1
}

# Run 1: production proposal (DRAM, g=8, refresh=100, K=1, W=1, topk=4%).
# DRAM mode → no expert lossless FC (EXPERT_FC_D=999999).
run_variant proposal_hutch_k1pct_finite_diff_F1 \
    /home/asdf1234/Megatron-DeepSpeed/results/exp_etc_g8_2000_dram \
    PRMOE_SPEC="4 4 8 8 8 8" \
    CP_TARGET=dram \
    BLOCK_QUANT_G=8 \
    HESSIAN_REFRESH_EVERY=100 \
    HESSIAN_NUM_SAMPLES=1 \
    PROPOSAL_TOPK=0.04 \
    NONEXPERT_FC_W=1 \
    EXPERT_FC_D=999999

# Run 2: sequential MoC layer-RR (DRAM, D=4, W=1).
run_variant moc_dram_d6 \
    /home/asdf1234/Megatron-DeepSpeed/results/exp_etc_moc_d4_2000_dram \
    PRMOE_SPEC="4 4 8 8 8 8" \
    CP_TARGET=dram \
    NONEXPERT_FC_W=1 \
    EXPERT_FC_D=4

echo "[etc] ALL DONE."
