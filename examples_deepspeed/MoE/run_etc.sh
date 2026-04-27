#!/bin/bash
# ETC 3-way: baseline + moc_m2 + proposal_s4_k01 @ seed 2024
# PR-MoE 4/4/8/8/8/8 + topk=2 + cap=1.25 + fault every 100 iter
# Rolling lossy-FC state + score distribution diagnostic
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
OUT=/home/asdf1234/Megatron-DeepSpeed/results/exp_etc
mkdir -p "$OUT"/log "$OUT"/cp
echo "[etc] OUTPUT_BASEPATH=$OUT" | tee -a "$OUT/SUMMARY.txt"

for V in baseline moc_m2 proposal_s4_k01 element_wise; do
    TAG="${V}_seed2024"
    LOG="$OUT/log/${TAG}.log"
    if [ -f "$LOG" ] && grep -q "after training is done" "$LOG"; then
        echo "[etc] === Skipping $V (already complete) ==="
        continue
    fi
    echo "[etc] === Running $V ==="
    rm -rf "$OUT/cp/${TAG}" 2>/dev/null
    OUTPUT_BASEPATH="$OUT" VARIANT="$V" SEED=2024 TRAIN_ITERS=2000 FAULT_INTERVAL=100 \
        bash "$DIR/proposal_etc_template.sh"
    # Cleanup CP dir after run (keep only logs for analysis)
    if [ "$V" != "baseline" ]; then
        echo "[etc] Cleaning CP for ${TAG}..."
        rm -rf "$OUT/cp/${TAG}"
    fi
    echo "[etc] === Finished $V ==="
    df -h /home | tail -1
done

echo "[etc] ALL DONE."
