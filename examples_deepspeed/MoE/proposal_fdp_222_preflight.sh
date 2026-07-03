#!/usr/bin/env bash
# Short preflight for the final 10K comparison:
#   topology (8xH100 + NVLink)
#   proposal FDP write/readback/fault/oracle smoke at (dp,pp,ep)=(2,2,2)
#   iter-time D2H-on/write-off probe against cached baseline
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${DIR}/../.." && pwd)
if [ -x "${REPO_ROOT}/.venv/bin/deepspeed" ]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

export PROPOSAL_IO_BACKEND=${PROPOSAL_IO_BACKEND:-fdp}
export PROPOSAL_FDP_BLOCK_DEVICE=${PROPOSAL_FDP_BLOCK_DEVICE:-/dev/nvme8n1}
export PROPOSAL_FDP_DEVICE=${PROPOSAL_FDP_DEVICE:-/dev/ng8n1}

if [ "${PROPOSAL_IO_BACKEND}" = "fdp" ] && [ "${EUID}" -ne 0 ]; then
    exec sudo -n -E env \
        PATH="${PATH}" \
        LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        PYTHONPATH="${PYTHONPATH:-}" \
        VIRTUAL_ENV="${VIRTUAL_ENV:-}" \
        DATA_PATH="${DATA_PATH:-}" \
        VALID_DATA_PATH="${VALID_DATA_PATH:-}" \
        OUT="${OUT:-}" \
        TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-}" \
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}" \
        MEGATRON_FUSED_KERNEL_ARCH_LIST="${MEGATRON_FUSED_KERNEL_ARCH_LIST:-}" \
        TRAIN_PRECISION="${TRAIN_PRECISION:-}" \
        PROPOSAL_IO_BACKEND="${PROPOSAL_IO_BACKEND}" \
        PROPOSAL_FDP_BLOCK_DEVICE="${PROPOSAL_FDP_BLOCK_DEVICE}" \
        PROPOSAL_FDP_DEVICE="${PROPOSAL_FDP_DEVICE}" \
        PROPOSAL_FDP_VALIDATE_READBACK="${PROPOSAL_FDP_VALIDATE_READBACK:-}" \
        PROPOSAL_FDP_FORCE_FORMAT="${PROPOSAL_FDP_FORCE_FORMAT:-}" \
        bash "$0" "$@"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-"${REPO_ROOT}/results/proposal_fdp_222_preflight/${STAMP}"}
mkdir -p "${OUT}"

cleanup_output_owner() {
    if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ] && [ -d "${OUT}" ]; then
        chown -R "${SUDO_USER}:${SUDO_USER}" "${OUT}" || true
    fi
}
trap cleanup_output_owner EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader > "${OUT}/gpus.txt"
    nvidia-smi topo -m > "${OUT}/topology.txt"
    GPU_COUNT=$(wc -l < "${OUT}/gpus.txt")
    if [ "${GPU_COUNT}" -lt 8 ]; then
        echo "[preflight] need 8 GPUs, found ${GPU_COUNT}" >&2
        exit 1
    fi
    if ! grep -q "NV18" "${OUT}/topology.txt"; then
        echo "[preflight] expected NVLink topology marker NV18 was not found" >&2
        exit 1
    fi
fi

if [ "${PROPOSAL_IO_BACKEND}" = "fdp" ]; then
    FDP_BLOCK_NAME=$(basename "${PROPOSAL_FDP_BLOCK_DEVICE}")
    FDP_SYSFS="/sys/block/${FDP_BLOCK_NAME}"
    if [ ! -e "${FDP_SYSFS}/queue/logical_block_size" ] || [ ! -e "${FDP_SYSFS}/size" ]; then
        echo "[preflight] FDP block geometry not found for ${PROPOSAL_FDP_BLOCK_DEVICE}" >&2
        exit 1
    fi
    if [ ! -e /usr/include/liburing.h ]; then
        echo "[preflight] liburing-dev is required for FDP extension builds" >&2
        exit 1
    fi
    export PROPOSAL_FDP_NSID=${PROPOSAL_FDP_NSID:-1}
    export PROPOSAL_FDP_LBA_SIZE=${PROPOSAL_FDP_LBA_SIZE:-$(cat "${FDP_SYSFS}/queue/logical_block_size")}
    export PROPOSAL_FDP_NAMESPACE_NLB=${PROPOSAL_FDP_NAMESPACE_NLB:-$(( $(cat "${FDP_SYSFS}/size") * 512 / PROPOSAL_FDP_LBA_SIZE ))}
    export PROPOSAL_FDP_METADATA_START_LBA=${PROPOSAL_FDP_METADATA_START_LBA:-0}
    export PROPOSAL_FDP_METADATA_NLB=${PROPOSAL_FDP_METADATA_NLB:-$(( 64 * 1024 * 1024 / PROPOSAL_FDP_LBA_SIZE ))}
    export PROPOSAL_FDP_DATA_START_LBA=${PROPOSAL_FDP_DATA_START_LBA:-$(( PROPOSAL_FDP_METADATA_START_LBA + PROPOSAL_FDP_METADATA_NLB ))}
    export PROPOSAL_FDP_DATA_NLB=${PROPOSAL_FDP_DATA_NLB:-$(( PROPOSAL_FDP_NAMESPACE_NLB - PROPOSAL_FDP_DATA_START_LBA ))}
    export PROPOSAL_FDP_ASSUME_FDP_SUPPORTED=${PROPOSAL_FDP_ASSUME_FDP_SUPPORTED:-1}
    export PROPOSAL_FDP_METADATA_MMAP=${PROPOSAL_FDP_METADATA_MMAP:-1}
    export PROPOSAL_FDP_CHUNK_MB=${PROPOSAL_FDP_CHUNK_MB:-64}
    export PROPOSAL_FDP_MAX_IO_KB=${PROPOSAL_FDP_MAX_IO_KB:-$(cat "${FDP_SYSFS}/queue/max_sectors_kb")}
    export PROPOSAL_FDP_NUM_MOE_LAYERS=${PROPOSAL_FDP_NUM_MOE_LAYERS:-12}
    export PROPOSAL_FDP_EXPERTS_PER_LAYER=${PROPOSAL_FDP_EXPERTS_PER_LAYER:-64}
    export PROPOSAL_FDP_ALLOW_FORMAT=${PROPOSAL_FDP_ALLOW_FORMAT:-1}
fi

echo "[preflight] OUT=${OUT}"
echo "[preflight] topology OK: 8x GPU and NV18 NVLink marker present"
echo "[preflight] FDP block=${PROPOSAL_FDP_BLOCK_DEVICE} char=${PROPOSAL_FDP_DEVICE}"

SMOKE_OUT="${OUT}/functional_smoke"
SMOKE_ITERS=${SMOKE_ITERS:-12}
SMOKE_FAULT_INTERVAL=${SMOKE_FAULT_INTERVAL:-8}
SMOKE_H_REFRESH_EVERY=${SMOKE_H_REFRESH_EVERY:-4}
SMOKE_ORACLE_ITERS=${SMOKE_ORACLE_ITERS:-${SMOKE_H_REFRESH_EVERY},$((SMOKE_H_REFRESH_EVERY * 2))}
mkdir -p "${SMOKE_OUT}"

echo "[preflight] functional smoke: readback=1 fault_interval=${SMOKE_FAULT_INTERVAL} oracle_iters=${SMOKE_ORACLE_ITERS}"
env \
    VARIANT=iter_time_debug \
    OUTPUT_BASEPATH="${SMOKE_OUT}" \
    MASTER_PORT="${MASTER_PORT:-29610}" \
    NUM_GPUS=8 \
    ZERO_STAGE=0 \
    USE_FLASH_ATTN_V2=0 \
    TRAIN_PRECISION=bf16 \
    PP_SIZE=2 \
    EP_PARALLEL_SIZE=2 \
    CP_TARGET=disk \
    SELECTION_MODE=hessian \
    PROPOSAL_TOPK=0.04 \
    NONEXPERT_FC_W=1 \
    ASYNC_QUEUE=5 \
    FAULT_INTERVAL="${SMOKE_FAULT_INTERVAL}" \
    FAULT_WARMUP=0 \
    HESSIAN_REFRESH_EVERY="${SMOKE_H_REFRESH_EVERY}" \
    HESSIAN_REFRESH_WARMUP_EVERY="${SMOKE_H_REFRESH_EVERY}" \
    TRAIN_ITERS="${SMOKE_ITERS}" \
    SEED=2024 \
    NUM_LAYERS=12 \
    HIDDEN_SIZE=768 \
    NUM_ATTN_HEADS=12 \
    SEQ_LEN=512 \
    NUM_EXPERTS=64 \
    MOE_TOPK=1 \
    MOE_TRAIN_CAP=1.25 \
    MOE_EVAL_CAP=1.25 \
    GLOBAL_BATCH_SIZE=8 \
    BATCH_SIZE=1 \
    LR=4.5e-4 \
    MIN_LR=4.5e-6 \
    LR_WARMUP_ITERS=1 \
    LOG_INTERVAL=1 \
    EVAL_INTERVAL=999999 \
    EVAL_ITERS=0 \
    EXTRA_MEGATRON_FLAGS="--exit-interval ${SMOKE_ITERS}" \
    EXTRA_PROPOSAL_FLAGS="--proposal-e-hat-proxy-off --proposal-lfcl-plan-policy lossy_log_size --proposal-independent-oracle-enable --proposal-independent-oracle-iters ${SMOKE_ORACLE_ITERS} --proposal-independent-oracle-log-dir ${SMOKE_OUT}/oracle --proposal-independent-oracle-rank-sample ${ORACLE_RANK_SAMPLE:-65536} --proposal-independent-oracle-actual-batches ${ORACLE_ACTUAL_BATCHES:-1} --proposal-independent-oracle-hvp-batches ${ORACLE_HVP_BATCHES:-1}" \
    PROPOSAL_INDEPENDENT_ORACLE_AUTOGRAD_GOLD=${PROPOSAL_INDEPENDENT_ORACLE_AUTOGRAD_GOLD:-1} \
    PROPOSAL_INDEPENDENT_ORACLE_ONESHOT_HVP_ACTUAL=${PROPOSAL_INDEPENDENT_ORACLE_ONESHOT_HVP_ACTUAL:-1} \
    PROPOSAL_INDEPENDENT_ORACLE_SELECTED_HVP=${PROPOSAL_INDEPENDENT_ORACLE_SELECTED_HVP:-1} \
    PROPOSAL_FDP_VALIDATE_READBACK=1 \
    PROPOSAL_FDP_FORCE_FORMAT=${PROPOSAL_FDP_FORCE_FORMAT_SMOKE:-1} \
    TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_ROOT}/.torch_extensions}" \
    bash "${DIR}/proposal_test_template.sh" 2>&1 | tee "${SMOKE_OUT}/driver.log"

SMOKE_LOG="${SMOKE_OUT}/log/iter_time_debug_seed2024.log"
if ! grep -q "FDP readback validation OK" "${SMOKE_LOG}"; then
    echo "[preflight] missing FDP readback validation marker in ${SMOKE_LOG}" >&2
    exit 1
fi
if ! grep -q "\\[proposal\\] FAULT@${SMOKE_FAULT_INTERVAL}" "${SMOKE_LOG}"; then
    echo "[preflight] missing fault marker in ${SMOKE_LOG}" >&2
    exit 1
fi
if ! find "${SMOKE_OUT}/oracle" -type f -name '*.jsonl' -print -quit | grep -q .; then
    echo "[preflight] missing independent oracle JSONL output" >&2
    exit 1
fi

python3 - "${SMOKE_OUT}/log/iter_time_debug_seed2024.faults.jsonl" "${RECOVERY_TIME_MAX_S:-30}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
limit = float(sys.argv[2])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if not rows:
    raise SystemExit("no fault records")
times = [float(r.get("recover_time_s", 0.0)) for r in rows]
mx = max(times)
print(f"[preflight] recovery_time_s max={mx:.3f} limit={limit:.3f}")
if mx > limit:
    raise SystemExit(f"recovery time exceeded limit: {mx:.3f}s > {limit:.3f}s")
PY

echo "[preflight] iter-time probe: baseline=skip d2h=on write_commit=off readback=0"
ITER_OUT="${OUT}/iter_time_d2h_writeoff"
mkdir -p "${ITER_OUT}"
TIMING_ITERS=${TIMING_ITERS:-80}
TIMING_BASELINE_MS=${TIMING_BASELINE_MS:-488.3}
TIMING_D2H_DELTA_MAX_MS=${TIMING_D2H_DELTA_MAX_MS:-200}
env \
    VARIANT=iter_time_debug \
    OUTPUT_BASEPATH="${ITER_OUT}" \
    MASTER_PORT="${MASTER_PORT_TIMING:-29611}" \
    NUM_GPUS=8 \
    ZERO_STAGE=0 \
    USE_FLASH_ATTN_V2=0 \
    TRAIN_PRECISION=bf16 \
    PP_SIZE=2 \
    EP_PARALLEL_SIZE=2 \
    CP_TARGET=disk \
    SELECTION_MODE=hessian \
    PROPOSAL_TOPK=0.04 \
    NONEXPERT_FC_W=1 \
    ASYNC_QUEUE=5 \
    FAULT_INTERVAL=0 \
    FAULT_WARMUP=0 \
    HESSIAN_REFRESH_EVERY=10000 \
    HESSIAN_REFRESH_WARMUP_EVERY=10000 \
    TRAIN_ITERS="${TIMING_ITERS}" \
    SEED=2024 \
    NUM_LAYERS=12 \
    HIDDEN_SIZE=768 \
    NUM_ATTN_HEADS=12 \
    SEQ_LEN=512 \
    NUM_EXPERTS=64 \
    MOE_TOPK=1 \
    MOE_TRAIN_CAP=1.25 \
    MOE_EVAL_CAP=1.25 \
    GLOBAL_BATCH_SIZE=8 \
    BATCH_SIZE=1 \
    LR=4.5e-4 \
    MIN_LR=4.5e-6 \
    LR_WARMUP_ITERS=1 \
    LOG_INTERVAL=10 \
    EVAL_INTERVAL=999999 \
    EVAL_ITERS=0 \
    EXTRA_MEGATRON_FLAGS="--exit-interval ${TIMING_ITERS}" \
    EXTRA_PROPOSAL_FLAGS="--proposal-e-hat-proxy-off --proposal-lfcl-plan-policy lossy_log_size --proposal-test-skip-expert-commit --proposal-test-skip-nonexpert-commit ${ITER_TIME_EXTRA_PROPOSAL_FLAGS:-}" \
    PROPOSAL_FDP_VALIDATE_READBACK=0 \
    PROPOSAL_FDP_FORCE_FORMAT=${PROPOSAL_FDP_FORCE_FORMAT_TIMING:-1} \
    TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING=false \
    TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_ROOT}/.torch_extensions}" \
    bash "${DIR}/proposal_test_template.sh" 2>&1 | tee "${ITER_OUT}/driver.log"

python3 - "${ITER_OUT}/log/iter_time_debug_seed2024.log" "${TIMING_BASELINE_MS}" "${TIMING_D2H_DELTA_MAX_MS}" <<'PY'
import pathlib
import re
import statistics
import sys

path = pathlib.Path(sys.argv[1])
baseline_ms = float(sys.argv[2])
limit_ms = float(sys.argv[3])
text = path.read_text(errors="replace")
windows = [
    (int(m.group(1)), float(m.group(2)))
    for m in re.finditer(
        r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\): ([0-9.]+)",
        text,
    )
]
refresh = {
    int(m.group(1))
    for m in re.finditer(r"refresh_finite_diff @iter(\d+)", text)
}
steady = [(i, v) for i, v in windows if i >= 30 and i not in refresh]
if not steady:
    raise SystemExit(f"no steady iter-time windows in {path}")
vals = [v for _, v in steady]
mean = statistics.mean(vals)
delta = mean - baseline_ms
print(f"[preflight] iter_windows={windows}")
print(f"[preflight] refresh_iters={sorted(refresh)}")
print(f"[preflight] d2h_writeoff_steady_mean_ms={mean:.2f} "
      f"p50={statistics.median(vals):.2f} min={min(vals):.2f} max={max(vals):.2f} n={len(vals)}")
print(f"[preflight] baseline_cached_steady_ms={baseline_ms:.2f} delta_mean_ms={delta:.2f}")
if delta > limit_ms:
    raise SystemExit(
        f"D2H/write-off overhead exceeded limit: {delta:.2f}ms > {limit_ms:.2f}ms"
    )
PY

echo "[preflight] PASS"
echo "[preflight] outputs: ${OUT}"
