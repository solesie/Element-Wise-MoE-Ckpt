#!/bin/bash
# Independent long-interval oracle telemetry for DS MoE proposal policies.
# Test-only: each policy x K is a separate run with its own checkpoint history.
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-"$DIR/../../results/INDEPENDENT_ORACLE_MATRIX_EP2_${STAMP}"}
ORACLE_ITERS=${ORACLE_ITERS:-"50,550,1050,1550,2050,2550,3050,3550,4050,4550"}
TRAIN_ITERS_FULL=${TRAIN_ITERS_FULL:-5000}
ACTUAL_BATCHES=${ACTUAL_BATCHES:-4}
HVP_BATCHES=${HVP_BATCHES:-1}
RANK_SAMPLE=${RANK_SAMPLE:-4194304}
SEED_BASE=${SEED_BASE:-2024}

POLICIES=${POLICIES:-"production_agebit true_e_m_t_diag_abs_h true_e_g_diag_abs_h"}
K_RATIOS=${K_RATIOS:-"0.02 0.04 0.08"}

mkdir -p "$OUT/log"

run_one() {
    local model="$1"
    local policy="$2"
    local k_ratio="$3"
    local train_iters="$4"
    local out_dir="$OUT/${model}/${policy}/K${k_ratio//./p}"
    mkdir -p "$out_dir/independent_oracle" "$out_dir/log"

    local num_layers hidden heads experts lr min_lr batch global_batch
    case "$model" in
        125M)
            num_layers=12
            hidden=768
            heads=12
            experts=64
            lr=4.5e-4
            min_lr=4.5e-6
            batch=2
            global_batch=8
            ;;
        350M)
            num_layers=24
            hidden=1024
            heads=16
            experts=128
            lr=2.0e-4
            min_lr=2.0e-6
            batch=2
            global_batch=8
            ;;
        *)
            echo "unknown model=$model" >&2
            exit 2
            ;;
    esac

    echo "[independent-oracle] start model=$model policy=$policy K=$k_ratio iters=$train_iters out=$out_dir"
    env \
        NUM_GPUS=2 \
        EP_PARALLEL_SIZE=2 \
        PP_SIZE=1 \
        NUM_LAYERS="$num_layers" HIDDEN_SIZE="$hidden" NUM_ATTN_HEADS="$heads" \
        GLOBAL_BATCH_SIZE="$global_batch" BATCH_SIZE="$batch" SEQ_LEN=2048 \
        MOE_TOPK=1 NUM_EXPERTS="$experts" MOE_TRAIN_CAP=1.25 MOE_EVAL_CAP=1.25 \
        LR="$lr" MIN_LR="$min_lr" LR_WARMUP_ITERS=50 TRAIN_ITERS="$train_iters" \
        CP_TARGET=test_dram \
        HESSIAN_REFRESH_EVERY=10000 \
        FAULT_INTERVAL=0 \
        PROPOSAL_TOPK="$k_ratio" \
        EXTRA_MEGATRON_FLAGS="--exit-interval ${train_iters} --eval-interval 999999 --eval-iters 0 --log-interval 50" \
        EXTRA_PROPOSAL_FLAGS="--proposal-experiment-score-policy ${policy} \
            --proposal-independent-oracle-enable \
            --proposal-independent-oracle-iters ${ORACLE_ITERS} \
            --proposal-independent-oracle-log-dir ${out_dir}/independent_oracle \
            --proposal-independent-oracle-rank-sample ${RANK_SAMPLE} \
            --proposal-independent-oracle-actual-batches ${ACTUAL_BATCHES} \
            --proposal-independent-oracle-hvp-batches ${HVP_BATCHES}" \
        OUTPUT_BASEPATH="$out_dir" \
        SEED="$SEED_BASE" \
        VARIANT=verify_vram \
        bash "$DIR/proposal_test_template.sh" 2>&1 | tee "$out_dir/log/run.log"
    echo "[independent-oracle] done model=$model policy=$policy K=$k_ratio"
}

if [ "${SMOKE:-0}" = "1" ]; then
    run_one "${SMOKE_MODEL:-125M}" "${SMOKE_POLICY:-production_agebit}" "${SMOKE_K:-0.04}" "${SMOKE_ITERS:-55}"
    exit 0
fi

for model in 125M 350M; do
    for policy in $POLICIES; do
        for k_ratio in $K_RATIOS; do
            run_one "$model" "$policy" "$k_ratio" "$TRAIN_ITERS_FULL"
        done
    done
done
