#!/bin/bash
# Benchmark recovery time by replaying existing CP data via
# --proposal-replay-only. Does NOT clean the CP dir. Intended for
# measuring recover_s after recovery.py optimizations.
#
# Usage:
#   CP_DIR=/path/to/cp/proposal_ema_hutch_seed2024 \
#   T_FAULT=834 \
#   bash examples_deepspeed/MoE/bench_recovery.sh
#
# Defaults point at results/exp_etc_new/cp/proposal_ema_hutch_seed2024
# with t_fault=834 (latest DC iter).
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}

# Model config MUST match the original training run that produced the CP.
SEQ_LEN=2048
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTN_HEADS=12
GLOBAL_BATCH_SIZE=24
BATCH_SIZE=8
EXPERT_INTERVAL=2
PRMOE_SPEC="4 4 8 8 8 8"

CP_DIR=${CP_DIR:-${PWD}/results/exp_etc_new/cp/proposal_ema_hutch_seed2024}
T_FAULT=${T_FAULT:-834}
SEED=${SEED:-2024}
TAG="bench_recovery_t${T_FAULT}_$(date +%H%M%S)"
LOG_DIR=${LOG_DIR:-${PWD}/results/exp_etc_new/log}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${TAG}.log"

# Generate a stub DS config matching the original training (ZeRO-0, BF16).
DS_CONFIG=${LOG_DIR}/ds_config_${TAG}.json
cat > "${DS_CONFIG}" <<JSON
{
  "train_batch_size": ${GLOBAL_BATCH_SIZE},
  "train_micro_batch_size_per_gpu": ${BATCH_SIZE},
  "steps_per_print": 10,
  "gradient_clipping": 1.0,
  "zero_optimization": {"stage": 0},
  "bf16": {"enabled": true},
  "fp16": {"enabled": false},
  "wall_clock_breakdown": false
}
JSON

VOCAB_PATH=${DIR}/../rebase/gpt2-vocab.json
MERGE_PATH=${DIR}/../rebase/gpt2-merges.txt
DATA_PATH=${DATA_PATH:-/home/asdf1234/data/slimpajama_6b/slimpajama_6b_train_text_document}
VALID_DATA_PATH=${VALID_DATA_PATH:-/home/asdf1234/data/slimpajama_6b/slimpajama_6b_valid_text_document}

MEGATRON_OPTS="--override-opt_param-scheduler --adam-beta1 0.9 --adam-beta2 0.95 \
    --tensor-model-parallel-size 1 --moe-expert-parallel-size 1 \
    --num-experts ${PRMOE_SPEC} --expert-interval ${EXPERT_INTERVAL} \
    --moe-loss-coeff 0.01 \
    --moe-train-capacity-factor 1.25 --moe-eval-capacity-factor 1.25 --moe-min-capacity 4 \
    --topk 2 --init-method-std 0.014 \
    --micro-batch-size ${BATCH_SIZE} --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --num-layers ${NUM_LAYERS} --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --seq-length ${SEQ_LEN} --max-position-embeddings ${SEQ_LEN} \
    --train-iters ${T_FAULT} \
    --lr 1.5e-4 --min-lr 1.5e-5 --lr-decay-style cosine --lr-warmup-iters 50 \
    --split 98,2,0 \
    --log-interval 10 --eval-interval 1000000 --eval-iters 2 \
    --save-interval 100000 \
    --weight-decay 0.1 --clip-grad 1.0 --hysteresis 2 \
    --num-workers 2 --bf16 --use-flash-attn-v2 --ds_fused_adam \
    --seed ${SEED} \
    --proposal-enable --proposal-cp-dir ${CP_DIR} \
    --proposal-cp-mode proposal \
    --proposal-selection-mode hessian \
    --proposal-hessian-refresh-every 40 \
    --proposal-hessian-num-samples 4 \
    --proposal-hessian-ema-alpha 0.9 \
    --proposal-topk-ratio 0.03 \
    --proposal-expert-fc-D 2 \
    --proposal-lossless-fc-fraction 2.0 \
    --proposal-nonexpert-fc-W 10 \
    --proposal-async-queue 5 \
    --proposal-replay-only"

DATA_OPTS="--vocab-file ${VOCAB_PATH} --merge-file ${MERGE_PATH} \
    --data-path ${DATA_PATH} --valid-data-path ${VALID_DATA_PATH} --data-impl mmap"

DS_OPTS="--deepspeed --deepspeed_config ${DS_CONFIG} \
    --pipeline-model-parallel-size 1 --no-pipeline-parallel --create-moe-param-group"

echo "[bench-recovery] CP=${CP_DIR} t_fault=${T_FAULT} LOG=${LOG_FILE}"
deepspeed --num_gpus 1 --master_port 29500 ${DIR}/../../pretrain_gpt.py \
    ${MEGATRON_OPTS} ${DATA_OPTS} ${DS_OPTS} 2>&1 | tee "${LOG_FILE}"

# Extract recover_s from the log (printed by proposal_recover path).
echo ""
echo "=== recovery timing ==="
grep -E "recovery diag|recover.*time|\[proposal\] recover" "${LOG_FILE}" | tail -5
