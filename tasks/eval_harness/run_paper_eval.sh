#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TRAIN_VENV=${TRAIN_VENV:-"${REPO_ROOT}/.venv"}
DEPS_DIR=${EVAL_DEPS_DIR:-"${HOME}/.cache/element-wise-moe-eval/lm_eval_0_2"}
DATA_ROOT=${LM_EVAL_PAPER_DATA:-"${HOME}/data/lm_eval_paper_6tasks"}
EXPECTED_DEPS_STAMP=lm-eval-0.2.0-paper-overlay-v1

CHECKPOINT_PATH=${1:-${CHECKPOINT_PATH:-}}
if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "usage: $0 BF16_MODEL_CHECKPOINT_ROOT_OR_TAG [RESULT_JSON]" >&2
    exit 2
fi
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "checkpoint directory not found: ${CHECKPOINT_PATH}" >&2
    exit 2
fi
if compgen -G "${CHECKPOINT_PATH}/mp_rank_*_model_states.pt" >/dev/null; then
    CHECKPOINT_TAG_PATH=${CHECKPOINT_PATH}
elif [ -f "${CHECKPOINT_PATH}/latest" ]; then
    CHECKPOINT_TAG=$(<"${CHECKPOINT_PATH}/latest")
    if [[ -z "${CHECKPOINT_TAG}" || "${CHECKPOINT_TAG}" == */* || \
          "${CHECKPOINT_TAG}" == "." || "${CHECKPOINT_TAG}" == ".." ]]; then
        echo "invalid latest tag under: ${CHECKPOINT_PATH}" >&2
        exit 2
    fi
    CHECKPOINT_TAG_PATH="${CHECKPOINT_PATH}/${CHECKPOINT_TAG}"
else
    echo "no model shard or latest pointer under: ${CHECKPOINT_PATH}" >&2
    exit 2
fi
if ! compgen -G "${CHECKPOINT_TAG_PATH}/mp_rank_*_model_states.pt" >/dev/null; then
    echo "no model shards under: ${CHECKPOINT_TAG_PATH}" >&2
    exit 2
fi
CONFIG_PATH="${CHECKPOINT_TAG_PATH}/deepspeed_config.json"
VOCAB_PATH="${CHECKPOINT_TAG_PATH}/tokenizer/gpt2-vocab.json"
MERGE_PATH="${CHECKPOINT_TAG_PATH}/tokenizer/gpt2-merges.txt"
RESULT_PATH=${2:-${RESULT_PATH:-"${REPO_ROOT}/results/downstream_7tasks.json"}}
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "bundled DeepSpeed config not found: ${CONFIG_PATH}" >&2
    exit 2
fi
if [ ! -f "${VOCAB_PATH}" ] || [ ! -f "${MERGE_PATH}" ]; then
    echo "bundled GPT-2 tokenizer not found under: ${CHECKPOINT_TAG_PATH}/tokenizer" >&2
    exit 2
fi
if [ ! -f "${DATA_ROOT}/manifest.json" ]; then
    echo "paper task data not ready: ${DATA_ROOT}" >&2
    echo "run: ${TRAIN_VENV}/bin/python ${SCRIPT_DIR}/prepare_paper_tasks.py" >&2
    exit 2
fi
"${TRAIN_VENV}/bin/python" "${SCRIPT_DIR}/prepare_paper_tasks.py" \
    --data-root "${DATA_ROOT}" --verify-only
if [ ! -f "${DEPS_DIR}/.stamp" ] || \
   [ "$(<"${DEPS_DIR}/.stamp")" != "${EXPECTED_DEPS_STAMP}" ]; then
    "${SCRIPT_DIR}/setup_paper_eval_env.sh"
fi

mkdir -p "$(dirname "${RESULT_PATH}")"

PP_SIZE=${PP_SIZE:-2}
TP_SIZE=${TP_SIZE:-1}
EP_SIZE=${MOE_EXPERT_PARALLEL_SIZE:-4}
NUM_GPUS=${NUM_GPUS:-8}
EVAL_MICRO_BATCH_SIZE=${EVAL_MICRO_BATCH_SIZE:-8}
TASKS=${TASKS:-piqa,hellaswag,race,mathqa,boolq,arc_challenge,winogrande}
GENERATION_MAX_TOKENS=${GENERATION_MAX_TOKENS:-64}

export LM_EVAL_PAPER_DATA="${DATA_ROOT}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="${TRAIN_VENV}/bin:${PATH}"
source "${REPO_ROOT}/tools/cuda_arch_env.sh"
configure_cuda_arch_env "${REPO_ROOT}" "${TRAIN_VENV}/bin/python"
export PYTHONPATH="${DEPS_DIR}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${TRAIN_VENV}/bin/python" -c \
    'from lm_eval import evaluator; from tasks.eval_harness.paper_tasks import PAPER_TASK_REGISTRY, TASK_NAMES; assert set(PAPER_TASK_REGISTRY) == set(TASK_NAMES)'

MEGATRON_REQUIRED_ARGS=(
    --num-layers -1
    --hidden-size -1
    --num-attention-heads -1
    --seq-length -1
    --max-position-embeddings -1
)

CMD=(
    "${TRAIN_VENV}/bin/deepspeed"
    --num_nodes 1
    --num_gpus "${NUM_GPUS}"
    "${SCRIPT_DIR}/evaluate.py"
    --load "${CHECKPOINT_PATH}"
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --moe-expert-parallel-size "${EP_SIZE}"
    --micro-batch-size "${EVAL_MICRO_BATCH_SIZE}"
    --no-load-optim
    --no-load-rng
    --inference
    --disable-moe-token-dropping
    --no-masked-softmax-fusion
    --tokenizer-type GPT2BPETokenizer
    --vocab-file "${VOCAB_PATH}"
    --merge-file "${MERGE_PATH}"
    --adaptive_seq_len
    --task_list "${TASKS}"
    --num_fewshot 0
    --generation-max-tokens "${GENERATION_MAX_TOKENS}"
    --paper-data-dir "${DATA_ROOT}"
    --results_path "${RESULT_PATH}"
    --deepspeed
    --deepspeed_config "${CONFIG_PATH}"
    "${MEGATRON_REQUIRED_ARGS[@]}"
)
if [ "${PP_SIZE}" -le 1 ]; then
    CMD+=(--no-pipeline-parallel)
fi
if [ -n "${EVAL_LIMIT:-}" ]; then
    CMD+=(--limit "${EVAL_LIMIT}")
fi

echo "[paper-eval] checkpoint=${CHECKPOINT_PATH}"
echo "[paper-eval] tasks=${TASKS} precision=bf16 tp=${TP_SIZE} pp=${PP_SIZE} ep=${EP_SIZE} gpus=${NUM_GPUS} micro=${EVAL_MICRO_BATCH_SIZE} limit=${EVAL_LIMIT:-full} gen_max=${GENERATION_MAX_TOKENS}"
if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '[paper-eval] command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi
"${CMD[@]}"
"${TRAIN_VENV}/bin/python" "${SCRIPT_DIR}/summarize_paper_results.py" "${RESULT_PATH}"
