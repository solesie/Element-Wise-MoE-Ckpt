#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-matrix}
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/.." && pwd)
MOE_DIR="${REPO_ROOT}/examples_deepspeed/MoE"
STAMP=${STAMP:-$(date "+%Y%m%d_%H%M%S")}
OUT_ROOT=${OUT_ROOT:-"${REPO_ROOT}/results/a100_2gpu_proposal_debug/${STAMP}"}

mkdir -p "${OUT_ROOT}"

COMMON_ENV=(
  PROPOSAL_IO_BACKEND=file
  PROPOSAL_CP_TARGET=test_dram
  NUM_GPUS=2
  NUM_GPUS_PERNODE=2
  MP_SIZE=1
  ZERO_STAGE=${ZERO_STAGE:-0}
  TRAIN_PRECISION=bf16
  TRAIN_ITERS=${TRAIN_ITERS:-300}
  SEQ_LEN=${SEQ_LEN:-512}
  BATCH_SIZE=${BATCH_SIZE:-2}
  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
  WARMUP_ITERS=${WARMUP_ITERS:-20}
  EVAL_INTERVAL=${EVAL_INTERVAL:-100}
  EVAL_ITERS=${EVAL_ITERS:-5}
  LOG_INTERVAL=${LOG_INTERVAL:-10}
  SAVE_INTERVAL=1000000
  NO_MEGATRON_CKPT=1
  NO_TENSORBOARD=1
  ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-false}
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
  MEGATRON_FUSED_KERNEL_ARCH_LIST=${MEGATRON_FUSED_KERNEL_ARCH_LIST:-80}
  DEEPSPEED_LAUNCH_ARGS="--num_gpus 2"
  PROPOSAL_TOPK=${PROPOSAL_TOPK:-0.04}
  NONEXPERT_FC_W=${NONEXPERT_FC_W:-1}
  HESSIAN_REFRESH_WARMUP_EVERY=${HESSIAN_REFRESH_WARMUP_EVERY:-10}
  HESSIAN_REFRESH_EVERY=${HESSIAN_REFRESH_EVERY:-50}
  FAULT_INTERVAL=${FAULT_INTERVAL:-50}
  FAULT_WARMUP=${FAULT_WARMUP:-0}
  PROPOSAL_ORACLE_ENABLE=${PROPOSAL_ORACLE_ENABLE:-0}
  ORACLE_INTERVAL=${ORACLE_INTERVAL:-100}
  ORACLE_RANK_SAMPLE=${ORACLE_RANK_SAMPLE:-32768}
  ORACLE_ACTUAL_BATCHES=${ORACLE_ACTUAL_BATCHES:-2}
  ORACLE_HVP_BATCHES=${ORACLE_HVP_BATCHES:-1}
  PROPOSAL_APPROX_PROBE_TELEMETRY_DETAIL=1
)

# Keep 12 layers so PP2 has multiple MoE-bearing layers per stage, but shrink
# the block size enough for short A100 debug runs.
MODEL_OVERRIDE_FLAGS=${MODEL_OVERRIDE_FLAGS:-"--num-layers 12 --hidden-size 384 --num-attention-heads 6"}

run_proposal() {
  local case_name=$1
  local pp_size=$2
  local ep_parallel=$3
  local num_experts=$4
  local lfcl_policy=$5
  local out="${OUT_ROOT}/${case_name}"
  mkdir -p "${out}/log"
  echo "[a100-debug] proposal case=${case_name} pp=${pp_size} ep_parallel=${ep_parallel} experts=${num_experts} out=${out}"
  env "${COMMON_ENV[@]}" \
    PP_SIZE="${pp_size}" \
    MOE_EXPERT_PARALLEL_SIZE="${ep_parallel}" \
    EP_SIZE="${num_experts}" \
    LFCL_PLAN_POLICY="${lfcl_policy}" \
    OUTPUT_BASEPATH="${out}" \
    PROPOSAL_CP_DIR="${out}/proposal_cp" \
    FAULT_LOG="${out}/log/proposal_faults.jsonl" \
    ORACLE_LOG_DIR="${out}/oracle" \
    PROPOSAL_LFCL_FIRE_LOG="${out}/log/lfcl_fire_rank{rank}.jsonl" \
    PROPOSAL_LFCL_OVERLAP_LOG="${out}/log/lfcl_overlap.jsonl" \
    PROPOSAL_DIRECT_FEEDBACK_LOG="${out}/log/direct_feedback.jsonl" \
    EXTRA_MEGATRON_FLAGS="${MODEL_OVERRIDE_FLAGS} ${EXTRA_MEGATRON_FLAGS:-}" \
    bash "${MOE_DIR}/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal.sh"
}

run_baseline() {
  local case_name=$1
  local pp_size=$2
  local ep_parallel=$3
  local num_experts=$4
  local out="${OUT_ROOT}/${case_name}"
  mkdir -p "${out}/log"
  echo "[a100-debug] baseline case=${case_name} pp=${pp_size} ep_parallel=${ep_parallel} experts=${num_experts} out=${out}"
  env "${COMMON_ENV[@]}" \
    PP_SIZE="${pp_size}" \
    MOE_EXPERT_PARALLEL_SIZE="${ep_parallel}" \
    EP_SIZE="${num_experts}" \
    OUTPUT_BASEPATH="${out}" \
    EXTRA_MEGATRON_FLAGS="${MODEL_OVERRIDE_FLAGS} ${EXTRA_MEGATRON_FLAGS:-}" \
    bash "${MOE_DIR}/ds_pretrain_gpt_125M_MoE64.sh"
}

case "${MODE}" in
  pp2)
    run_proposal pp2_ep1_proposal_size 2 1 "${EP_SIZE_OVERRIDE:-16}" "${LFCL_PLAN_POLICY:-lossy_log_size}"
    ;;
  pp2_rr)
    run_proposal pp2_ep1_proposal_rr 2 1 "${EP_SIZE_OVERRIDE:-16}" round_robin
    ;;
  ep2)
    run_proposal pp1_ep2_proposal_size 1 2 "${EP_SIZE_OVERRIDE:-16}" "${LFCL_PLAN_POLICY:-lossy_log_size}"
    ;;
  dp2)
    run_proposal pp1_ep1_dp2_proposal_size 1 1 "${EP_SIZE_OVERRIDE:-16}" "${LFCL_PLAN_POLICY:-lossy_log_size}"
    ;;
  baseline_pp2)
    run_baseline pp2_ep1_baseline 2 1 "${EP_SIZE_OVERRIDE:-16}"
    ;;
  baseline_ep2)
    run_baseline pp1_ep2_baseline 1 2 "${EP_SIZE_OVERRIDE:-16}"
    ;;
  baseline_dp2)
    run_baseline pp1_ep1_dp2_baseline 1 1 "${EP_SIZE_OVERRIDE:-16}"
    ;;
  matrix)
    run_baseline pp2_ep1_baseline 2 1 "${EP_SIZE_OVERRIDE:-16}"
    run_proposal pp2_ep1_proposal_size 2 1 "${EP_SIZE_OVERRIDE:-16}" lossy_log_size
    run_proposal pp2_ep1_proposal_rr 2 1 "${EP_SIZE_OVERRIDE:-16}" round_robin
    run_baseline pp1_ep2_baseline 1 2 "${EP_SIZE_OVERRIDE:-16}"
    run_proposal pp1_ep2_proposal_size 1 2 "${EP_SIZE_OVERRIDE:-16}" lossy_log_size
    run_baseline pp1_ep1_dp2_baseline 1 1 "${EP_SIZE_OVERRIDE:-16}"
    run_proposal pp1_ep1_dp2_proposal_size 1 1 "${EP_SIZE_OVERRIDE:-16}" lossy_log_size
    ;;
  *)
    cat >&2 <<USAGE
usage: $0 [matrix|pp2|pp2_rr|ep2|dp2|baseline_pp2|baseline_ep2|baseline_dp2]

Useful overrides:
  OUT_ROOT=/path/to/out TRAIN_ITERS=200 SEQ_LEN=512 BATCH_SIZE=2 GLOBAL_BATCH_SIZE=8
  EP_SIZE_OVERRIDE=16 PROPOSAL_TOPK=0.04 PROPOSAL_ORACLE_ENABLE=0
USAGE
    exit 2
    ;;
esac

echo "[a100-debug] done: ${OUT_ROOT}"
