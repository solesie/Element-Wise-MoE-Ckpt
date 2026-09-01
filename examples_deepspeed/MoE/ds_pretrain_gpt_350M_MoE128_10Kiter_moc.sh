#!/usr/bin/env bash
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
if [ ! -x "${REPO_ROOT}/.venv/bin/python" ] || [ ! -x "${REPO_ROOT}/.venv/bin/deepspeed" ]; then
    echo "Training environment is missing under ${REPO_ROOT}/.venv" >&2
    exit 2
fi
export CUDA_HOME
export DS_ACCELERATOR=${DS_ACCELERATOR:-cuda}
export PATH="${REPO_ROOT}/.venv/bin:${CUDA_HOME}/bin:${PATH}"
TORCH_LIB=$("${REPO_ROOT}/.venv/bin/python" -c \
    'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH="${TORCH_LIB}:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
BASE_SCRIPT="${DIR}/ds_pretrain_gpt_350M_MoE128.sh"

DEFAULT_DATA_ROOT=${SLIMPAJAMA_DATA_ROOT:-"${HOME}/data/slimpajama_6b"}
DEFAULT_TRAIN_PREFIX="${DEFAULT_DATA_ROOT}/slimpajama_6b_train_text_document"
DEFAULT_VALID_PREFIX="${DEFAULT_DATA_ROOT}/slimpajama_6b_valid_text_document"
export VOCAB_PATH=${VOCAB_PATH:-"${DIR}/../rebase/gpt2-vocab.json"}
export MERGE_PATH=${MERGE_PATH:-"${DIR}/../rebase/gpt2-merges.txt"}
export DATA_PATH=${DATA_PATH:-"${DEFAULT_TRAIN_PREFIX}"}
export VALID_DATA_PATH=${VALID_DATA_PATH:-"${DEFAULT_VALID_PREFIX}"}
export DATA_CACHE_PATH=${DATA_CACHE_PATH:-"${REPO_ROOT}/results/data-cache"}
if [ ! -e "${DATA_PATH}.idx" ] || [ ! -e "${DATA_PATH}.bin" ] || \
   [ ! -e "${VALID_DATA_PATH}.idx" ] || [ ! -e "${VALID_DATA_PATH}.bin" ]; then
    echo "SlimPajama 6B mmap dataset is not ready; run dataset/download_slimpajama_6b.sh." >&2
    exit 2
fi

export TRAIN_ITERS=${TRAIN_ITERS:-10000}
export SEQ_LEN=${SEQ_LEN:-2048}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
export WARMUP_ITERS=${WARMUP_ITERS:-80}
export TRAIN_TOKENS=${TRAIN_TOKENS:-$((TRAIN_ITERS * GLOBAL_BATCH_SIZE * SEQ_LEN))}
export WARMUP_TOKENS=${WARMUP_TOKENS:-$((WARMUP_ITERS * GLOBAL_BATCH_SIZE * SEQ_LEN))}
export LR_DECAY_TOKENS=${LR_DECAY_TOKENS:-${TRAIN_TOKENS}}
EXPECTED_WARMUP_TOKENS=$((WARMUP_ITERS * GLOBAL_BATCH_SIZE * SEQ_LEN))
if [ "${WARMUP_ITERS}" -le 0 ] && [ "${ALLOW_MOC_ZERO_LR_WARMUP:-0}" != "1" ]; then
    echo "MoC LR warmup must be positive; set ALLOW_MOC_ZERO_LR_WARMUP=1 only for an explicit ablation." >&2
    exit 2
fi
if [ "${WARMUP_TOKENS}" -ne "${EXPECTED_WARMUP_TOKENS}" ]; then
    echo "MoC warmup mismatch: WARMUP_ITERS=${WARMUP_ITERS} implies ${EXPECTED_WARMUP_TOKENS} tokens, got WARMUP_TOKENS=${WARMUP_TOKENS}." >&2
    exit 2
fi

export NUM_GPUS=${NUM_GPUS:-8}
export NUM_GPUS_PERNODE=${NUM_GPUS_PERNODE:-8}
export MP_SIZE=${MP_SIZE:-1}
export PP_SIZE=${PP_SIZE:-2}
export EP_SIZE=${EP_SIZE:-128}
export MOE_EXPERT_PARALLEL_SIZE=${MOE_EXPERT_PARALLEL_SIZE:-4}
export ZERO_STAGE=${ZERO_STAGE:-1}
export BATCH_SIZE=${BATCH_SIZE:-8}
export ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-false}
export TRAIN_PRECISION=${TRAIN_PRECISION:-bf16}
export LR=${LR:-2.0e-4}
export MIN_LR=${MIN_LR:-2e-06}
export NUM_LAYERS=${NUM_LAYERS:-24}
export EXPERT_INTERVAL=${EXPERT_INTERVAL:-2}

DENOM=$((MP_SIZE * PP_SIZE * MOE_EXPERT_PARALLEL_SIZE))
if [ "${DENOM}" -le 0 ] || [ $((NUM_GPUS % DENOM)) -ne 0 ]; then
    echo "Invalid parallelism: NUM_GPUS=${NUM_GPUS}, TP=${MP_SIZE}, PP=${PP_SIZE}, EP=${MOE_EXPERT_PARALLEL_SIZE}" >&2
    exit 2
fi
EXPERT_DP_SIZE=$((NUM_GPUS / DENOM))
DENSE_DP_SIZE=$((NUM_GPUS / (MP_SIZE * PP_SIZE)))
if [ "${EXPERT_DP_SIZE}" -ne 1 ]; then
    echo "Expected expert DP=1, got expert DP=${EXPERT_DP_SIZE}" >&2
    exit 2
fi
if [ "${EXPERT_INTERVAL}" -le 0 ] || [ $((NUM_LAYERS % EXPERT_INTERVAL)) -ne 0 ]; then
    echo "Invalid MoE layout: NUM_LAYERS=${NUM_LAYERS}, EXPERT_INTERVAL=${EXPERT_INTERVAL}" >&2
    exit 2
fi
MOC_MOE_LAYERS=$((NUM_LAYERS / EXPERT_INTERVAL))
if [ $((MOC_MOE_LAYERS % PP_SIZE)) -ne 0 ]; then
    echo "MoE layers must divide evenly across PP stages: layers=${MOC_MOE_LAYERS}, PP=${PP_SIZE}" >&2
    exit 2
fi

export MOC_LFCL_INTERVAL_ITERS=${MOC_LFCL_INTERVAL_ITERS:-4}
export MOC_SEQUENTIAL_POLICY=${MOC_SEQUENTIAL_POLICY:-layer_rr}
export MOC_LFCL_FRACTION_PCT=${MOC_LFCL_FRACTION_PCT:-1.0}
if [ "${MOC_LFCL_INTERVAL_ITERS}" -le 0 ]; then
    echo "MOC_LFCL_INTERVAL_ITERS must be >=1" >&2
    exit 2
fi
if [ "${MOC_SEQUENTIAL_POLICY}" = "expert_rr_pp_back_locked" ]; then
    if [ "${MOC_LFCL_INTERVAL_ITERS}" -ne 1 ]; then
        echo "expert_rr_pp_back_locked requires MOC_LFCL_INTERVAL_ITERS=1" >&2
        exit 2
    fi
    MOC_EVENT_BUDGET_PCT="${MOC_LFCL_FRACTION_PCT}"
    MOC_AVG_BUDGET_PCT="${MOC_LFCL_FRACTION_PCT}"
elif [ "${MOC_SEQUENTIAL_POLICY}" = "layer_rr" ]; then
    MOC_EVENT_BUDGET_PCT=$("${REPO_ROOT}/.venv/bin/python" -c \
        'import sys; print(f"{100.0 * int(sys.argv[1]) / int(sys.argv[2]):.4f}")' \
        "${PP_SIZE}" "${MOC_MOE_LAYERS}")
    MOC_AVG_BUDGET_PCT=$("${REPO_ROOT}/.venv/bin/python" -c \
        'import sys; print(f"{float(sys.argv[1]) / int(sys.argv[2]):.4f}")' \
        "${MOC_EVENT_BUDGET_PCT}" "${MOC_LFCL_INTERVAL_ITERS}")
else
    echo "Unknown MOC_SEQUENTIAL_POLICY=${MOC_SEQUENTIAL_POLICY}" >&2
    exit 2
fi

export EVAL_INTERVAL=${EVAL_INTERVAL:-100}
export EVAL_ITERS=${EVAL_ITERS:-10}
export LOG_INTERVAL=${LOG_INTERVAL:-10}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}
export NO_MEGATRON_CKPT=1
export NO_TENSORBOARD=${NO_TENSORBOARD:-1}
export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/ds_pretrain_gpt_350M_MoE128_10Kiter_moc"}
DEFAULT_FINAL_BF16_MODEL_PATH=""
if [ "${TRAIN_ITERS}" -eq 10000 ]; then
    DEFAULT_FINAL_BF16_MODEL_PATH="${OUTPUT_BASEPATH}/bf16_model"
fi
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH-${DEFAULT_FINAL_BF16_MODEL_PATH}}
source "${REPO_ROOT}/tools/cuda_arch_env.sh"
configure_cuda_arch_env "${REPO_ROOT}" "${REPO_ROOT}/.venv/bin/python"
export DEEPSPEED_LAUNCH_ARGS=${DEEPSPEED_LAUNCH_ARGS:-"--num_gpus ${NUM_GPUS}"}

export PROPOSAL_IO_BACKEND=${PROPOSAL_IO_BACKEND:-file}
export PROPOSAL_CP_TARGET=${PROPOSAL_CP_TARGET:-test_dram}
export PROPOSAL_CP_DIR=${PROPOSAL_CP_DIR:-"${OUTPUT_BASEPATH}/proposal_cp"}
# Empty values explicitly disable these two diagnostics.
export PROPOSAL_IO_BACKLOG_TRACE_DIR=${PROPOSAL_IO_BACKLOG_TRACE_DIR-"${OUTPUT_BASEPATH}/waf_trace"}
export PROPOSAL_LFCL_FIRE_LOG=${PROPOSAL_LFCL_FIRE_LOG-"${OUTPUT_BASEPATH}/log/moc_lfcl_rank{rank}.jsonl"}
export FAULT_INTERVAL=${FAULT_INTERVAL:-100}
export FAULT_WARMUP=${FAULT_WARMUP:-0}
export FAULT_LOG=${FAULT_LOG:-"${OUTPUT_BASEPATH}/log/moc_faults.jsonl"}

EXTRA_MOC_FLAGS=" \
    --proposal-enable \
    --proposal-cp-mode sequential \
    --proposal-cp-target ${PROPOSAL_CP_TARGET} \
    --proposal-cp-dir ${PROPOSAL_CP_DIR} \
    --proposal-moc-sequential-policy ${MOC_SEQUENTIAL_POLICY} \
    --proposal-lfcl-interval-iters ${MOC_LFCL_INTERVAL_ITERS} \
    --proposal-lfcl-fraction-pct ${MOC_LFCL_FRACTION_PCT} \
    --proposal-lfcl-plan-policy round_robin \
    --proposal-nonexpert-fc-W ${NONEXPERT_FC_W:-1} \
    --proposal-hessian-refresh-every 0 \
    --proposal-fault-interval ${FAULT_INTERVAL} \
    --proposal-fault-warmup ${FAULT_WARMUP} \
    --proposal-fault-log ${FAULT_LOG} \
    --expert-interval ${EXPERT_INTERVAL}"

# Do not inherit proposal/oracle flags from a caller's generic override.  MoC
# extras must be opted into through the MoC-specific variable.
export EXTRA_MEGATRON_FLAGS="--use-flash-attn-v2 ${MOC_EXTRA_MEGATRON_FLAGS:-} ${EXTRA_MOC_FLAGS}"

echo "[350M-10K-moc] dataset=${DATA_PATH}"
echo "[350M-10K-moc] dense_dp=${DENSE_DP_SIZE} expert_dp=${EXPERT_DP_SIZE} pp=${PP_SIZE} ep=${MOE_EXPERT_PARALLEL_SIZE} experts=${EP_SIZE} gpus=${NUM_GPUS}"
echo "[350M-10K-moc] iters=${TRAIN_ITERS} tokens=${TRAIN_TOKENS} lr_warmup_iters=${WARMUP_ITERS} lr_warmup_tokens=${WARMUP_TOKENS} decay=${LR_DECAY_TOKENS}"
echo "[350M-10K-moc] micro_batch=${BATCH_SIZE} zero=${ZERO_STAGE} activation_checkpoint=${ACTIVATION_CHECKPOINT} precision=${TRAIN_PRECISION}"
echo "[350M-10K-moc] schedule=${MOC_SEQUENTIAL_POLICY} event=${MOC_EVENT_BUDGET_PCT}% every=${MOC_LFCL_INTERVAL_ITERS}iter avg_dense=${MOC_AVG_BUDGET_PCT}%/iter"
echo "[350M-10K-moc] eval=${EVAL_INTERVAL}x${EVAL_ITERS} cp_target=${PROPOSAL_CP_TARGET} hessian=off oracle=off fault=${FAULT_INTERVAL}"
echo "[350M-10K-moc] fire_log=${PROPOSAL_LFCL_FIRE_LOG:-disabled} waf_trace=${PROPOSAL_IO_BACKLOG_TRACE_DIR:-disabled}"
echo "[350M-10K-moc] final_bf16_model=${FINAL_BF16_MODEL_PATH:-disabled}"

exec bash "${BASE_SCRIPT}"
