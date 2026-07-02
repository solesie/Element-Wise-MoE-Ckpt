#!/bin/bash
# proposal_test_template.sh - proposal test/debug launcher only.
#
# Supported callers:
#   - iter_time_debug.sh
#   - verify_vram_run.sh
#
# This file intentionally does not carry old experiment variants. It is a
# small shared launcher for proposal timing and VRAM tests.
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$DIR/../.." && pwd)
export LD_LIBRARY_PATH=$(python3 -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null):${LD_LIBRARY_PATH:-}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:${PATH}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

SEQ_LEN=${SEQ_LEN:-2048}
NUM_LAYERS=${NUM_LAYERS:-12}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
NUM_ATTN_HEADS=${NUM_ATTN_HEADS:-12}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_GPUS=${NUM_GPUS:-1}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
EP_PARALLEL_SIZE=${EP_PARALLEL_SIZE:-1}
NUM_EXPERTS=${NUM_EXPERTS:-48}
MOE_TOPK=${MOE_TOPK:-1}
ZERO_STAGE=${ZERO_STAGE:-0}

TRAIN_ITERS=${TRAIN_ITERS:-2000}
SEED=${SEED:-2024}
LR=${LR:-1.5e-4}
MIN_LR=${MIN_LR:-1.5e-5}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-50}
INIT_STD=${INIT_STD:-0.014}
MLC=${MLC:-0.01}

VARIANT=${VARIANT:?must set VARIANT=baseline|iter_time_debug|verify_vram}
OUTPUT_BASEPATH=${OUTPUT_BASEPATH:?must set OUTPUT_BASEPATH}
TAG="${VARIANT}_seed${SEED}"
FAULT_INTERVAL=${FAULT_INTERVAL:-0}
FAULT_WARMUP=${FAULT_WARMUP:-300}
mkdir -p "${OUTPUT_BASEPATH}/log" "${OUTPUT_BASEPATH}/cp"
LOG_FILE="${OUTPUT_BASEPATH}/log/${TAG}.log"
FAULT_LOG="${OUTPUT_BASEPATH}/log/${TAG}.faults.jsonl"
CP_DIR="${CP_DIR_OVERRIDE:-${OUTPUT_BASEPATH}/cp/${TAG}}"

if [ -d "${CP_DIR}" ]; then
    find "${CP_DIR}" -mindepth 1 -type f -delete 2>/dev/null || true
else
    mkdir -p "${CP_DIR}"
fi
if [ "${NUM_GPUS}" = "1" ] && [ "${PP_SIZE}" = "1" ] && [ "${EP_PARALLEL_SIZE}" = "1" ]; then
    mkdir -p "${CP_DIR}/non_expert" "${CP_DIR}/expert"
fi
rm -f "${FAULT_LOG}"

PROPOSAL_FLAGS=""
case "$VARIANT" in
    baseline)
        # Hard no-proposal run, even if the parent shell has leftover flags.
        PROPOSAL_FLAGS=""
        EXTRA_PROPOSAL_FLAGS=""
        ;;
    iter_time_debug|verify_vram)
        PROPOSAL_FLAGS="--proposal-enable --proposal-cp-dir ${CP_DIR} \
            --proposal-cp-mode proposal \
            --proposal-cp-target ${CP_TARGET:-disk} \
            --proposal-block-quant-g ${BLOCK_QUANT_G:-8} \
            --proposal-selection-mode ${SELECTION_MODE:-hessian} \
            --proposal-hessian-refresh-every ${HESSIAN_REFRESH_EVERY:-50} \
            --proposal-hessian-refresh-warmup-every ${HESSIAN_REFRESH_WARMUP_EVERY:-20} \
            --proposal-topk-ratio ${PROPOSAL_TOPK:-0.04} \
            --proposal-nonexpert-fc-W ${NONEXPERT_FC_W:-1} \
            --proposal-async-queue ${ASYNC_QUEUE:-5} \
            --proposal-fault-interval ${FAULT_INTERVAL} \
            --proposal-fault-warmup ${FAULT_WARMUP} \
            --proposal-fault-log ${FAULT_LOG}"
        ;;
    *)
        echo "Unsupported proposal test VARIANT=$VARIANT" >&2
        echo "Supported: baseline, iter_time_debug, verify_vram" >&2
        exit 2
        ;;
esac

VOCAB_PATH=${DIR}/../rebase/gpt2-vocab.json
MERGE_PATH=${DIR}/../rebase/gpt2-merges.txt
DEFAULT_DATA_ROOT="/home/elice/data/slimpajama_6b"
if [ ! -e "${DEFAULT_DATA_ROOT}/slimpajama_6b_train_text_document.idx" ]; then
    DEFAULT_DATA_ROOT="/home/asdf1234/data/slimpajama_6b"
fi
DATA_PATH=${DATA_PATH:-${DEFAULT_DATA_ROOT}/slimpajama_6b_train_text_document}
VALID_DATA_PATH=${VALID_DATA_PATH:-${DEFAULT_DATA_ROOT}/slimpajama_6b_valid_text_document}

data_options=" \
    --vocab-file ${VOCAB_PATH} \
    --merge-file ${MERGE_PATH} \
    --data-path ${DATA_PATH} \
    --valid-data-path ${VALID_DATA_PATH} \
    --data-impl mmap"

megatron_options=" \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 --adam-beta2 0.95 \
    --tensor-model-parallel-size ${TP_SIZE} --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
    --num-experts ${NUM_EXPERTS} \
    --moe-loss-coeff ${MLC} \
    --moe-train-capacity-factor ${MOE_TRAIN_CAP:-1.25} --moe-eval-capacity-factor ${MOE_EVAL_CAP:-1.25} --moe-min-capacity 4 \
    --topk ${MOE_TOPK} \
    --init-method-std ${INIT_STD} \
    --micro-batch-size ${BATCH_SIZE} --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --num-layers ${NUM_LAYERS} --hidden-size ${HIDDEN_SIZE} --num-attention-heads ${NUM_ATTN_HEADS} \
    --seq-length ${SEQ_LEN} --max-position-embeddings ${SEQ_LEN} \
    --train-iters ${TRAIN_ITERS} \
    --lr ${LR} --min-lr ${MIN_LR} --lr-decay-style cosine --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --split 98,2,0 \
    --log-interval ${LOG_INTERVAL:-10} --eval-interval ${EVAL_INTERVAL:-100} --eval-iters ${EVAL_ITERS:-128} \
    --save-interval 100000 \
    --weight-decay 0.1 --clip-grad 1.0 --hysteresis 2 \
    --num-workers 2 --bf16 --use-flash-attn-v2 --ds_fused_adam \
    --seed ${SEED} \
    ${ACTIVATION_CHECKPOINT:+--checkpoint-activations --deepspeed-activation-checkpointing} \
    ${EXTRA_MEGATRON_FLAGS:-} \
    ${PROPOSAL_FLAGS} \
    ${EXTRA_PROPOSAL_FLAGS:-}"

template_json="${DIR}/ds_config_gpt_TEMPLATE.json"
config_json="${OUTPUT_BASEPATH}/ds_config_${TAG}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" "${template_json}" \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/10/" \
    | sed "s/ZERO_STAGE/${ZERO_STAGE}/" \
    | sed "s/PRESCALE_GRAD/$([ "${ZERO_STAGE}" -gt 0 ] && echo false || echo true)/" \
    | sed "s/CONFIG_FP16_ENABLED/false/" \
    | sed "s/CONFIG_BF16_ENABLED/true/" \
    | sed "s/CONFIG_CL_ENABLED/false/" \
    | sed "s/CONFIG_CL_MIN/80/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/0/" > "${config_json}"

PIPELINE_FLAG=""
if [ "${PP_SIZE}" -eq 1 ]; then
    PIPELINE_FLAG="--no-pipeline-parallel"
fi

deepspeed_options=" \
    --deepspeed --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --pipeline-model-parallel-size ${PP_SIZE} ${PIPELINE_FLAG} \
    --create-moe-param-group"

if [ -n "${NSYS_PROFILE:-}" ]; then
    NSYS_TRACE=${NSYS_TRACE:-cuda,osrt,nvtx}
    NSYS_PREFIX="nsys profile -o ${NSYS_OUT:-/tmp/nsys_run} --force-overwrite true --trace=${NSYS_TRACE} -s none --cudabacktrace=none ${NSYS_EXTRA_ARGS:-}"
else
    NSYS_PREFIX=""
fi

run_cmd="${NSYS_PREFIX} deepspeed --num_gpus ${NUM_GPUS} --master_port ${MASTER_PORT:-29500} ${DIR}/../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options}"
echo "[proposal-test] TAG=${TAG} NUM_EXPERTS=${NUM_EXPERTS} GPUS=${NUM_GPUS} EP=${EP_PARALLEL_SIZE} PP=${PP_SIZE} LOG=${LOG_FILE}"
eval "${run_cmd}" 2>&1 | tee "${LOG_FILE}"
