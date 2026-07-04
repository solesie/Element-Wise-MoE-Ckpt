#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/paul/Element-Wise-MoE-Ckpt"
RUN_ROOT="${REPO_ROOT}/results/10k_compare_test_dram_micro8/20260704_011541"
MONITOR_LOG="${RUN_ROOT}/monitor_status.jsonl"
BASELINE_OUT="${RUN_ROOT}/baseline"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"

cd "${REPO_ROOT}" || exit 2

json_quote() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

log_status() {
  local phase="$1"
  local msg="$2"
  local now
  now=$(date --iso-8601=seconds)
  printf '{"ts":%s,"phase":%s,"msg":%s}\n' \
    "$(printf '%s' "${now}" | json_quote)" \
    "$(printf '%s' "${phase}" | json_quote)" \
    "$(printf '%s' "${msg}" | json_quote)" >> "${MONITOR_LOG}"
}

runner_alive() {
  local pid_file="${RUN_ROOT}/runner.pid"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid=$(tr -dc '0-9' < "${pid_file}")
    [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && return 0
  fi
  pgrep -f "tools/run_10k_compare_test_dram_micro8.sh" >/dev/null 2>&1
}

baseline_active_or_done() {
  [[ -d "${BASELINE_OUT}" ]] && return 0
  pgrep -f "ds_pretrain_gpt_125M_MoE64_10Kiter_baseline.sh|${BASELINE_OUT}" >/dev/null 2>&1
}

proposal_success() {
  grep -q "\[runner\] phase=proposal status=0" "${RUN_ROOT}/runner.log" 2>/dev/null && return 0
  grep -q "exits successfully" "${RUN_ROOT}/proposal"/log/*.log 2>/dev/null
}

start_baseline_fallback() {
  mkdir -p "${BASELINE_OUT}/log"
  log_status "fallback" "starting baseline fallback"
  env \
    TRAIN_ITERS=10000 \
    GLOBAL_BATCH_SIZE=256 \
    BATCH_SIZE=8 \
    EVAL_INTERVAL=1000 \
    EVAL_ITERS=20 \
    LOG_INTERVAL=10 \
    NUM_GPUS=8 \
    NUM_GPUS_PERNODE=8 \
    MP_SIZE=1 \
    PP_SIZE=2 \
    EP_SIZE=64 \
    MOE_EXPERT_PARALLEL_SIZE=2 \
    ZERO_STAGE=0 \
    TRAIN_PRECISION=bf16 \
    NO_TENSORBOARD=1 \
    NO_MEGATRON_CKPT=1 \
    DATA_CACHE_PATH="${REPO_ROOT}/results/data-cache" \
    TORCH_EXTENSIONS_DIR="${REPO_ROOT}/.torch_extensions" \
    TORCH_CUDA_ARCH_LIST=9.0 \
    MEGATRON_FUSED_KERNEL_ARCH_LIST=90 \
    OUTPUT_BASEPATH="${BASELINE_OUT}" \
    bash examples_deepspeed/MoE/ds_pretrain_gpt_125M_MoE64_10Kiter_baseline.sh \
      >> "${BASELINE_OUT}/log/baseline_fallback_wrapper.log" 2>&1
  local status=$?
  log_status "fallback" "baseline fallback exited status=${status}"
  exit "${status}"
}

log_status "monitor" "started interval=${INTERVAL_SECONDS}"

while true; do
  if [[ -f "${RUN_ROOT}/status.json" ]]; then
    log_status "monitor" "status_json=$(tr -d '\n' < "${RUN_ROOT}/status.json")"
    exit 0
  fi

  if baseline_active_or_done; then
    log_status "monitor" "baseline active or directory exists"
    sleep "${INTERVAL_SECONDS}"
    continue
  fi

  if runner_alive; then
    log_status "monitor" "runner alive; no action"
    sleep "${INTERVAL_SECONDS}"
    continue
  fi

  if proposal_success; then
    start_baseline_fallback
  fi

  log_status "monitor" "runner dead but proposal success not confirmed; no action"
  sleep "${INTERVAL_SECONDS}"
done
