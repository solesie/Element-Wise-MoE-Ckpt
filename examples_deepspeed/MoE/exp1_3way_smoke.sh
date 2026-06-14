#!/bin/bash
# 3-way comparison (quick smoke, 50 iters each).
#
# VARIANT (env):
#   prmoe     — PR-MoE "4 4 8 8 8 8", interval=2, proposal D=1 F=2%
#   uniform   — uniform "8", interval=2, proposal D=1 F=2%
#   baseline  — uniform "8", interval=2, NO proposal (pure training)
#
# All three share: NUM_LAYERS=12, micro=8, global=24, seq=2048, seed=2024.
# With interval=2 → 6 MoE layers. prmoe=40 experts, uniform=48 experts.
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}

VARIANT=${VARIANT:?must set VARIANT=prmoe|uniform|baseline}

SEQ_LEN=2048
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTN_HEADS=12
GLOBAL_BATCH_SIZE=24
BATCH_SIZE=8
EXPERT_INTERVAL=2

case "$VARIANT" in
    prmoe)    PRMOE_SPEC="4 4 8 8 8 8" ;;
    uniform)  PRMOE_SPEC="8" ;;
    baseline) PRMOE_SPEC="8" ;;
    *) echo "unknown VARIANT=$VARIANT"; exit 1 ;;
esac

TRAIN_ITERS=${TRAIN_ITERS:-100}
SEED=2024
LR=1.5e-4
MIN_LR=1.5e-5
LR_WARMUP_ITERS=10
INIT_STD=0.014
MLC=0.01

OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-/home/asdf1234/Megatron-DeepSpeed/results/exp1_3way_${VARIANT}}
TAG="${VARIANT}_seed${SEED}"
mkdir -p "${OUTPUT_BASEPATH}/log" "${OUTPUT_BASEPATH}/cp"
LOG_FILE="${OUTPUT_BASEPATH}/log/${TAG}.log"
CP_DIR="${OUTPUT_BASEPATH}/cp/${TAG}"
rm -rf "${CP_DIR}"; mkdir -p "${CP_DIR}/non_expert" "${CP_DIR}/expert"

PROPOSAL_FLAGS=""
if [ "$VARIANT" != "baseline" ]; then
    PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
        --proposal-cp-mode proposal \
        --proposal-selection-mode v_hat \
        --proposal-hessian-refresh-every 0 \
        --proposal-topk-ratio 0.04 \
        --proposal-nonexpert-fc-W 10 \
        --proposal-async-queue 5"
fi

VOCAB_PATH=${DIR}/../rebase/gpt2-vocab.json
MERGE_PATH=${DIR}/../rebase/gpt2-merges.txt
DATA_PATH=/home/asdf1234/data/slimpajama_6b/slimpajama_6b_train_text_document
VALID_DATA_PATH=/home/asdf1234/data/slimpajama_6b/slimpajama_6b_valid_text_document

data_options=" \
    --vocab-file ${VOCAB_PATH} --merge-file ${MERGE_PATH} \
    --data-path ${DATA_PATH} --valid-data-path ${VALID_DATA_PATH} \
    --data-impl mmap"

megatron_options=" \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 --adam-beta2 0.95 \
    --tensor-model-parallel-size 1 --moe-expert-parallel-size 1 \
    --num-experts ${PRMOE_SPEC} --expert-interval ${EXPERT_INTERVAL} \
    --moe-loss-coeff ${MLC} \
    --moe-train-capacity-factor 1.25 --moe-eval-capacity-factor 1.25 --moe-min-capacity 4 \
    --topk 2 \
    --init-method-std ${INIT_STD} \
    --micro-batch-size ${BATCH_SIZE} --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --num-layers ${NUM_LAYERS} --hidden-size ${HIDDEN_SIZE} --num-attention-heads ${NUM_ATTN_HEADS} \
    --seq-length ${SEQ_LEN} --max-position-embeddings ${SEQ_LEN} \
    --train-iters ${TRAIN_ITERS} \
    --lr ${LR} --min-lr ${MIN_LR} --lr-decay-style cosine --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --split 98,2,0 \
    --log-interval 10 --eval-interval 100000 --eval-iters 2 \
    --save-interval 100000 \
    --weight-decay 0.1 --clip-grad 1.0 --hysteresis 2 \
    --num-workers 2 --bf16 --use-flash-attn-v2 --ds_fused_adam \
    --seed ${SEED} \
    ${PROPOSAL_FLAGS}"

template_json="${DIR}/ds_config_gpt_TEMPLATE.json"
config_json="${OUTPUT_BASEPATH}/ds_config_${TAG}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/10/" \
    | sed "s/ZERO_STAGE/0/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/false/" \
    | sed "s/CONFIG_BF16_ENABLED/true/" \
    | sed "s/CONFIG_CL_ENABLED/false/" \
    | sed "s/CONFIG_CL_MIN/80/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/0/" > ${config_json}

deepspeed_options=" \
    --deepspeed --deepspeed_config ${config_json} \
    --pipeline-model-parallel-size 1 --no-pipeline-parallel \
    --create-moe-param-group"

run_cmd="deepspeed --num_gpus 1 --master_port 29513 ${DIR}/../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options}"
echo "[3way-$VARIANT] TAG=${TAG}  LOG=${LOG_FILE}"
echo "[3way-$VARIANT] spec=${PRMOE_SPEC} interval=${EXPERT_INTERVAL} iters=${TRAIN_ITERS} cp=$([ -z "$PROPOSAL_FLAGS" ] && echo off || echo on)"
eval ${run_cmd} 2>&1 | tee "${LOG_FILE}"
