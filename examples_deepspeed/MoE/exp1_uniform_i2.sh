#!/bin/bash
# Isolation test A: uniform 4 experts per MoE layer, interval=2 (6 MoE layers).
# Compares directly with matched_ref (PR-MoE "4 4 8 8 8 8", 6 MoE layers, 40 experts).
#
# Goal: isolate PR-MoE heterogeneity from MoE-layer-count effect.
#   matched_ref: 6 MoE layers, 40 experts, heterogeneous (4,4,8,8,8,8)
#   this run:    6 MoE layers, 24 experts, uniform (4 per layer)
# Same everything else (proposal code, micro_batch=8, etc.).
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}

SEQ_LEN=2048
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTN_HEADS=12
GLOBAL_BATCH_SIZE=24
BATCH_SIZE=8
EXPERT_INTERVAL=2                   # matches matched_ref
PRMOE_SPEC="4"                      # uniform 4 (vs matched_ref's "4 4 8 8 8 8")

TRAIN_ITERS=${TRAIN_ITERS:-2000}
SEED=${SEED:-2024}
LR=1.5e-4
MIN_LR=1.5e-5
LR_WARMUP_ITERS=50
INIT_STD=0.014
MLC=0.01

FAULT_INTERVAL=${FAULT_INTERVAL:-100}
FAULT_WARMUP=${FAULT_WARMUP:-300}

OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-/home/asdf1234/Megatron-DeepSpeed/results/exp1_uniform_i2_seed${SEED}}
TAG="uniform_i2_seed${SEED}"
mkdir -p "${OUTPUT_BASEPATH}/log" "${OUTPUT_BASEPATH}/cp"
LOG_FILE="${OUTPUT_BASEPATH}/log/${TAG}.log"
FAULT_LOG="${OUTPUT_BASEPATH}/log/${TAG}.faults.jsonl"
CP_DIR="${OUTPUT_BASEPATH}/cp/${TAG}"
if [ -d "${CP_DIR}" ]; then
    find "${CP_DIR}" -mindepth 1 -type f -delete 2>/dev/null || true
    find "${CP_DIR}" -mindepth 1 -type d -empty -delete 2>/dev/null || true
else
    mkdir -p "${CP_DIR}"
fi
mkdir -p "${CP_DIR}/non_expert" "${CP_DIR}/expert"
rm -f "${FAULT_LOG}"

PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
    --proposal-cp-mode proposal \
    --proposal-selection-mode hessian \
    --proposal-hessian-estimator hutchinson \
    --proposal-hessian-refresh-every 40 \
    --proposal-hessian-num-samples 4 \
    --proposal-topk-ratio 0.04 \
    --proposal-expert-fc-D 1 \
    --proposal-lossless-fc-fraction 2.0 \
    --proposal-nonexpert-fc-W 10 \
    --proposal-async-queue 5 \
    --proposal-fault-interval ${FAULT_INTERVAL} \
    --proposal-fault-warmup ${FAULT_WARMUP} \
    --proposal-fault-log ${FAULT_LOG}"

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
    --log-interval 10 --eval-interval 100 --eval-iters 128 \
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

run_cmd="deepspeed --num_gpus 1 --master_port 29512 ${DIR}/../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options}"
echo "[exp1-uniform-i2] TAG=${TAG}  LOG=${LOG_FILE}"
echo "[exp1-uniform-i2] uniform 4 experts / MoE layer, interval=2 → 6 MoE layers, 24 experts"
eval ${run_cmd} 2>&1 | tee "${LOG_FILE}"
