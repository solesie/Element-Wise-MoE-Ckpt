#!/usr/bin/env bash
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
        PROPOSAL_IO_BACKEND="${PROPOSAL_IO_BACKEND}" \
        PROPOSAL_FDP_BLOCK_DEVICE="${PROPOSAL_FDP_BLOCK_DEVICE}" \
        PROPOSAL_FDP_DEVICE="${PROPOSAL_FDP_DEVICE}" \
        bash "$0" "$@"
fi

OUT=${OUT:-"${REPO_ROOT}/results/preflight_222_fdp"}
rm -rf "${OUT}"
mkdir -p "${OUT}"

cleanup_output_owner() {
    if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ] && [ -d "${OUT}" ]; then
        chown -R "${SUDO_USER}:${SUDO_USER}" "${OUT}" || true
    fi
}
trap cleanup_output_owner EXIT

if [ "${PROPOSAL_IO_BACKEND}" = "fdp" ]; then
    FDP_BLOCK_NAME=$(basename "${PROPOSAL_FDP_BLOCK_DEVICE}")
    FDP_SYSFS="/sys/block/${FDP_BLOCK_NAME}"
    if [ ! -e "${FDP_SYSFS}/queue/logical_block_size" ] || [ ! -e "${FDP_SYSFS}/size" ]; then
        echo "FDP block geometry not found for ${PROPOSAL_FDP_BLOCK_DEVICE} (${FDP_SYSFS})" >&2
        exit 1
    fi
    if [ ! -e /usr/include/liburing.h ]; then
        echo "liburing development headers are missing. Install liburing-dev before FDP runs." >&2
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

COMMON_ENV=(
    NUM_GPUS=8
    NUM_GPUS_PERNODE=8
    MP_SIZE=1
    PP_SIZE=2
    MOE_EXPERT_PARALLEL_SIZE=2
    EP_SIZE=64
    ZERO_STAGE=0
    TRAIN_ITERS=${TRAIN_ITERS:-8}
    SEQ_LEN=${SEQ_LEN:-512}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
    BATCH_SIZE=${BATCH_SIZE:-1}
    WARMUP_ITERS=${WARMUP_ITERS:-1}
    LOG_INTERVAL=${LOG_INTERVAL:-1}
    EVAL_INTERVAL=999999
    EVAL_ITERS=0
    SAVE_INTERVAL=1000000
    ACTIVATION_CHECKPOINT=false
    DATA_PATH="${DATA_PATH:-/home/paul/data/slimpajama_6b/slimpajama_6b_train_text_document}"
    VALID_DATA_PATH="${VALID_DATA_PATH:-/home/paul/data/slimpajama_6b/slimpajama_6b_valid_text_document}"
    TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_ROOT}/.torch_extensions}"
)

run_baseline() {
    echo "[preflight] baseline short run"
    env "${COMMON_ENV[@]}" \
        OUTPUT_BASEPATH="${OUT}/baseline" \
        PROPOSAL_ORACLE_ENABLE=0 \
        BASELINE_EXTRA_MEGATRON_FLAGS="--exit-interval ${TRAIN_ITERS:-8}" \
        bash "${DIR}/ds_pretrain_gpt_125M_MoE64_10Kiter_baseline.sh"
}

run_proposal_functional() {
    echo "[preflight] proposal functional run: readback/fault/oracle enabled"
    env "${COMMON_ENV[@]}" \
        OUTPUT_BASEPATH="${OUT}/proposal_functional" \
        PROPOSAL_FDP_FORCE_FORMAT=1 \
        PROPOSAL_FDP_VALIDATE_READBACK=1 \
        FAULT_INTERVAL=${FAULT_INTERVAL:-4} \
        FAULT_WARMUP=0 \
        HESSIAN_REFRESH_EVERY=${HESSIAN_REFRESH_EVERY:-4} \
        HESSIAN_REFRESH_WARMUP_EVERY=${HESSIAN_REFRESH_WARMUP_EVERY:-4} \
        PROPOSAL_ORACLE_ENABLE=1 \
        ORACLE_ITERS=${ORACLE_ITERS:-4} \
        ORACLE_RANK_SAMPLE=${ORACLE_RANK_SAMPLE:-65536} \
        ORACLE_ACTUAL_BATCHES=${ORACLE_ACTUAL_BATCHES:-1} \
        ORACLE_HVP_BATCHES=${ORACLE_HVP_BATCHES:-1} \
        PROPOSAL_INDEPENDENT_ORACLE_AUTOGRAD_GOLD=${PROPOSAL_INDEPENDENT_ORACLE_AUTOGRAD_GOLD:-1} \
        PROPOSAL_INDEPENDENT_ORACLE_ONESHOT_HVP_ACTUAL=${PROPOSAL_INDEPENDENT_ORACLE_ONESHOT_HVP_ACTUAL:-1} \
        PROPOSAL_INDEPENDENT_ORACLE_SELECTED_HVP=${PROPOSAL_INDEPENDENT_ORACLE_SELECTED_HVP:-1} \
        EXTRA_MEGATRON_FLAGS="--exit-interval ${TRAIN_ITERS:-8}" \
        bash "${DIR}/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal.sh"
}

run_proposal_timing() {
    echo "[preflight] proposal timing run: no readback/fault/oracle"
    env "${COMMON_ENV[@]}" \
        OUTPUT_BASEPATH="${OUT}/proposal_timing" \
        PROPOSAL_FDP_FORCE_FORMAT=1 \
        PROPOSAL_FDP_VALIDATE_READBACK=0 \
        FAULT_INTERVAL=0 \
        HESSIAN_REFRESH_EVERY=100000 \
        HESSIAN_REFRESH_WARMUP_EVERY=100000 \
        PROPOSAL_ORACLE_ENABLE=0 \
        EXTRA_MEGATRON_FLAGS="--exit-interval ${TRAIN_ITERS:-8}" \
        bash "${DIR}/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal.sh"
}

run_baseline
run_proposal_functional
run_proposal_timing

python3 - "${OUT}" <<'PY'
import re
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
pat = re.compile(r"elapsed time per iteration \(ms\): ([0-9.]+)")

def times(root: Path):
    vals = []
    for path in sorted((root / "log").glob("*.log")):
        vals.extend(float(m.group(1)) for m in pat.finditer(path.read_text(errors="ignore")))
    return vals

base = times(out / "baseline")
prop = times(out / "proposal_timing")
drop = 1 if min(len(base), len(prop)) > 1 else 0
base, prop = base[drop:], prop[drop:]
n = min(len(base), len(prop))
if n:
    delta = [p - b for b, p in zip(base[:n], prop[:n])]
    text = (
        f"n={n}\n"
        f"baseline_mean_ms={statistics.mean(base[:n]):.3f}\n"
        f"proposal_mean_ms={statistics.mean(prop[:n]):.3f}\n"
        f"delta_mean_ms={statistics.mean(delta):.3f}\n"
        f"gate={'PASS' if statistics.mean(delta) <= 10.0 else 'FAIL'}\n"
    )
else:
    text = "n=0\ngate=FAIL\n"
(out / "iter_time_compare.txt").write_text(text)
print(text, end="")
PY

echo "[preflight] done: ${OUT}"
