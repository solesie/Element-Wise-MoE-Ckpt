#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
MOE_DIR="${SCRIPT_DIR}"
VARIANT=${SMOKE_VARIANT:-}

if [ "$#" -ne 0 ]; then
    echo "usage: $(basename "$0")" >&2
    echo "This runner has one fixed model: 125M-active / ~1.3B-expert-domain MoE." >&2
    exit 2
fi

case "${VARIANT}" in
    baseline|proposal) ;;
    *)
        echo "SMOKE_VARIANT must be baseline or proposal" >&2
        exit 2
        ;;
esac

# Canonical repository smoke family: the 125M GPT backbone uses six MoE
# layers.  48 experts make its expert parameter domain 1,360,060,416 elements
# (about 1.3B), sharded 24/24 across EP2.
MODEL_LABEL=125M-1.3B
export MODEL_SIZE=0.125
export NUM_LAYERS=12
export HIDDEN_SIZE=768
export NUM_ATTN_HEADS=12
export LR=${LR:-4.5e-4}
export MIN_LR=${MIN_LR:-4.5e-6}

export TRAIN_ITERS=${TRAIN_ITERS:-8}
if [ "${TRAIN_ITERS}" -lt 5 ] || [ "${TRAIN_ITERS}" -gt 10 ]; then
    echo "TRAIN_ITERS must stay in the 5-10 smoke-test range" >&2
    exit 2
fi

export NUM_GPUS=2
export NUM_GPUS_PERNODE=2
export MP_SIZE=1
export PP_SIZE=1
export EP_SIZE=48
export MOE_EXPERT_PARALLEL_SIZE=2
export ZERO_STAGE=1
export GLOBAL_BATCH_SIZE=4
export BATCH_SIZE=2
export SEQ_LEN=${SEQ_LEN:-2048}
export TRAIN_PRECISION=bf16
export ACTIVATION_CHECKPOINT=false
export WARMUP_ITERS=${WARMUP_ITERS:-1}
export TRAIN_TOKENS=$((TRAIN_ITERS * GLOBAL_BATCH_SIZE * SEQ_LEN))
export WARMUP_TOKENS=$((WARMUP_ITERS * GLOBAL_BATCH_SIZE * SEQ_LEN))
export LR_DECAY_TOKENS=${TRAIN_TOKENS}
export EVAL_INTERVAL=1000000
export EVAL_ITERS=1
export LOG_INTERVAL=1
export SAVE_INTERVAL=1000000
export NO_MEGATRON_CKPT=1
export NO_TENSORBOARD=1
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"${REPO_ROOT}/.torch_extensions"}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export MEGATRON_FUSED_KERNEL_ARCH_LIST=${MEGATRON_FUSED_KERNEL_ARCH_LIST:-90}
export DEEPSPEED_LAUNCH_ARGS="--num_nodes 1 --num_gpus 2"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export DS_ACCELERATOR=${DS_ACCELERATOR:-cuda}
export PATH="${REPO_ROOT}/.venv/bin:${CUDA_HOME}/bin:${PATH}"
TORCH_LIB=$(
    "${REPO_ROOT}/.venv/bin/python" -c \
        'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")'
)
export LD_LIBRARY_PATH="${TORCH_LIB}:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Keep topology/optimizer choices explicit instead of inheriting an interactive
# shell's experiment flags.
export EXTRA_MEGATRON_FLAGS="--topk 1 --expert-interval 2 --ds_fused_adam"
export BASELINE_EXTRA_MEGATRON_FLAGS="${EXTRA_MEGATRON_FLAGS}"

if [ $((EP_SIZE % MOE_EXPERT_PARALLEL_SIZE)) -ne 0 ]; then
    echo "EP_SIZE=${EP_SIZE} must be divisible by EP parallel size 2" >&2
    exit 2
fi

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
export OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"${REPO_ROOT}/results/h100_2gpu_smoke/${VARIANT}/${MODEL_LABEL}/${RUN_ID}"}
export FINAL_BF16_MODEL_PATH=${FINAL_BF16_MODEL_PATH:-"${OUTPUT_BASEPATH}/bf16_model"}
DOWNSTREAM_RESULT=${DOWNSTREAM_RESULT:-"${OUTPUT_BASEPATH}/downstream_7tasks_smoke.json"}

if [ "${DRY_RUN:-0}" != "1" ] && [ -d "${OUTPUT_BASEPATH}" ] && \
   [ -n "$(find "${OUTPUT_BASEPATH}" -mindepth 1 -print -quit)" ]; then
    echo "refusing to reuse non-empty smoke output: ${OUTPUT_BASEPATH}" >&2
    exit 2
fi

case "${VARIANT}" in
    baseline)
        TRAIN_SCRIPT="${MOE_DIR}/ds_pretrain_gpt_350M_MoE128_10Kiter_baseline.sh"
        unset PROPOSAL_IO_BACKLOG_TRACE_DIR
        ;;
    proposal)
        TRAIN_SCRIPT="${MOE_DIR}/ds_pretrain_gpt_350M_MoE128_10Kiter_proposal.sh"
        export PROPOSAL_IO_BACKEND=file
        export PROPOSAL_CP_TARGET=test_dram
        export PROPOSAL_CP_DIR=${PROPOSAL_CP_DIR:-"${OUTPUT_BASEPATH}/proposal_cp"}
        export PROPOSAL_IO_BACKLOG_TRACE_DIR=${PROPOSAL_IO_BACKLOG_TRACE_DIR:-"${OUTPUT_BASEPATH}/waf_trace"}
        export HESSIAN_REFRESH_WARMUP_EVERY=${HESSIAN_REFRESH_WARMUP_EVERY:-100}
        export HESSIAN_REFRESH_EVERY=${HESSIAN_REFRESH_EVERY:-100}
        export FAULT_INTERVAL=0
        export FAULT_WARMUP=0
        export PROPOSAL_ORACLE_ENABLE=0
        ;;
esac

if [ "${DRY_RUN:-0}" != "1" ]; then
    mapfile -t GPU_NAMES < <(
        "${REPO_ROOT}/.venv/bin/python" -c \
            'import torch; [print(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]'
    )
    if [ "${#GPU_NAMES[@]}" -lt 2 ]; then
        echo "two visible GPUs are required; found ${#GPU_NAMES[@]}" >&2
        exit 1
    fi
    for gpu in 0 1; do
        if [[ "${GPU_NAMES[$gpu]}" != *H100* ]]; then
            echo "GPU ${gpu} is not an H100: ${GPU_NAMES[$gpu]}" >&2
            exit 1
        fi
    done
    if [ "${VARIANT}" = "proposal" ]; then
        # test_dram keeps one pinned dense p/m/v recovery image per local
        # expert: about 7.60 GiB/rank (15.2 GiB total) for this 48E/EP2 model.
        # AOS/LFCL slabs, BF16 theta snapshots, grad snapshots, the dataset,
        # and both rank processes need substantial additional headroom.  The
        # guard is deliberately above the payload-only estimate: a failed
        # pin/allocation must not drive the host back into OOM.
        min_host_gib=${MIN_AVAILABLE_HOST_GIB:-40}
        available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        if [ "${available_kib:-0}" -lt $((min_host_gib * 1024 * 1024)) ]; then
            echo "proposal test_dram needs at least ${min_host_gib} GiB available host RAM; MemAvailable=$((available_kib / 1024 / 1024)) GiB" >&2
            exit 1
        fi
    fi
fi

echo "[h100-smoke] variant=${VARIANT} model=${MODEL_LABEL} layers=${NUM_LAYERS} hidden=${HIDDEN_SIZE} experts=${EP_SIZE}"
echo "[h100-smoke] gpus=2 tp=1 pp=1 ep=2 dp=1 gbs=4 micro=2 seq=${SEQ_LEN} iters=${TRAIN_ITERS} zero=1 bf16 flash_attn=v2"
echo "[h100-smoke] output=${OUTPUT_BASEPATH}"
if [ "${VARIANT}" = "proposal" ]; then
    echo "[h100-smoke] backlog=${PROPOSAL_IO_BACKLOG_TRACE_DIR}"
fi

bash "${TRAIN_SCRIPT}"
if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

"${REPO_ROOT}/.venv/bin/python" - \
    "${OUTPUT_BASEPATH}/log" \
    "${OUTPUT_BASEPATH}/iteration_times_ms.json" \
    "${TRAIN_ITERS}" <<'PY'
import json
import re
import sys
from pathlib import Path

log_dir, output, expected = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
logs = sorted(log_dir.glob("*.log"), key=lambda path: path.stat().st_mtime)
if not logs:
    raise SystemExit(f"no training log under {log_dir}")
pattern = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([0-9.]+)"
)
times = {}
for line in logs[-1].read_text(errors="replace").splitlines():
    match = pattern.search(line)
    if match:
        times[int(match.group(1))] = float(match.group(2))
missing = [iteration for iteration in range(1, expected + 1) if iteration not in times]
if missing:
    raise SystemExit(f"training log is missing iteration timings: {missing}")
payload = {
    "source_log": str(logs[-1]),
    "cold_start_included": True,
    "iterations_ms": [
        {"iteration": iteration, "elapsed_ms": times[iteration]}
        for iteration in range(1, expected + 1)
    ],
}
output.write_text(json.dumps(payload, indent=2) + "\n")
print(f"[h100-smoke] iteration_times={output}")
PY

TAG_DIR="${FINAL_BF16_MODEL_PATH}/global_step${TRAIN_ITERS}"
if [ ! -f "${FINAL_BF16_MODEL_PATH}/latest" ] || \
   [ ! -f "${TAG_DIR}/model_only_manifest.json" ] || \
   ! compgen -G "${TAG_DIR}/mp_rank_*_model_states.pt" >/dev/null; then
    echo "BF16 model-only checkpoint is incomplete: ${TAG_DIR}" >&2
    exit 1
fi
if [ "$(<"${FINAL_BF16_MODEL_PATH}/latest")" != "global_step${TRAIN_ITERS}" ]; then
    echo "BF16 model-only latest pointer does not name global_step${TRAIN_ITERS}" >&2
    exit 1
fi

"${REPO_ROOT}/.venv/bin/python" - "${TAG_DIR}" "${TRAIN_ITERS}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

tag = Path(sys.argv[1])
iteration = int(sys.argv[2])
manifest = json.loads((tag / "model_only_manifest.json").read_text())
expected = {
    "format": "deepspeed_moe_bf16_model_only",
    "complete": True,
    "iteration": iteration,
    "state_semantics": "live_post_optimizer_step",
    "parameter_dtype": "bfloat16",
    "optimizer_state": False,
    "fp32_master_weights": False,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"invalid BF16 manifest {key}: {manifest.get(key)!r} != {value!r}")
expected_topology = {
    "world_size": 2,
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "expert_parallel_size": 2,
    "data_parallel_size": 1,
}
if manifest.get("topology") != expected_topology:
    raise SystemExit(f"invalid BF16 checkpoint topology: {manifest.get('topology')!r}")
if list(tag.glob("*optim_states.pt")):
    raise SystemExit("BF16 model-only checkpoint unexpectedly contains optimizer state")

records = manifest.get("files")
if not isinstance(records, list) or not records:
    raise SystemExit("BF16 manifest contains no payload file records")
listed = set()
total_bytes = 0
for record in records:
    relative = record.get("path")
    expected_bytes = int(record.get("bytes", -1))
    expected_sha256 = str(record.get("sha256", ""))
    if not relative or relative in listed or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise SystemExit(f"invalid BF16 manifest payload path: {relative!r}")
    payload = tag / relative
    if not payload.is_file() or payload.stat().st_size != expected_bytes:
        raise SystemExit(f"BF16 payload size mismatch: {payload}")
    digest = hashlib.sha256()
    with payload.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise SystemExit(f"BF16 payload checksum mismatch: {payload}")
    listed.add(relative)
    total_bytes += expected_bytes
actual = {
    path.relative_to(tag).as_posix()
    for path in tag.rglob("*")
    if path.is_file() and path.name != "model_only_manifest.json"
}
if listed != actual:
    raise SystemExit(
        f"BF16 manifest/file mismatch: missing={sorted(actual - listed)} "
        f"extra={sorted(listed - actual)}"
    )
if total_bytes != int(manifest.get("total_bytes", -1)):
    raise SystemExit("BF16 manifest total_bytes mismatch")
print(f"[h100-smoke] bf16_checkpoint_ok tag={tag}")
PY

if [ "${VARIANT}" = "proposal" ]; then
    PATH="${REPO_ROOT}/.venv/bin:${PATH}" "${REPO_ROOT}/.venv/bin/python" - \
        "${PROPOSAL_IO_BACKLOG_TRACE_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = sorted(root.glob("waf_rank*.jsonl"))
if len(paths) != 2:
    raise SystemExit(f"expected two rank WAF traces under {root}, found {len(paths)}")
writes = 0
kinds = set()
for path in paths:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or rows[0].get("record") != "metadata" or rows[-1].get("record") != "footer":
        raise SystemExit(f"trace has no footer: {path}")
    metadata = rows[0]
    if (metadata.get("world_size"), metadata.get("ep_size"), metadata.get("experts_per_layer")) != (2, 2, 48):
        raise SystemExit(f"invalid trace topology in {path}: {metadata}")
    if (metadata.get("source_cp_target"), metadata.get("source_io_backend")) != ("test_dram", "file"):
        raise SystemExit(f"invalid trace source in {path}: {metadata}")
    for row in rows[1:-1]:
        if row.get("record") != "write_batch" or int(row.get("mono_ns", 0)) <= 0 or int(row.get("batch_seq", -1)) < 0:
            raise SystemExit(f"invalid trace write batch in {path}: {row}")
        for write in row.get("writes", []):
            if len(write) != 6 or int(write[4]) <= 0 or int(write[4]) % 4096:
                raise SystemExit(f"invalid logical write in {path}: {write}")
            kinds.add(write[0])
    footer = rows[-1]
    if footer.get("status") != "ok" or int(footer.get("dropped_batches", 0)):
        raise SystemExit(f"invalid trace footer in {path}: {footer}")
    writes += int(footer.get("writes", 0))
if writes <= 0:
    raise SystemExit("WAF traces contain no logical writes")
if kinds != {"c", "y", "n"}:
    raise SystemExit(f"WAF traces do not cover LFCL/LFSY/non-expert writes: {sorted(kinds)}")
print(f"[h100-smoke] backlog_ok files={len(paths)} logical_writes={writes} kinds={sorted(kinds)}")
PY
fi

if [ "${RUN_DOWNSTREAM:-1}" = "1" ]; then
    PP_SIZE=1 \
    TP_SIZE=1 \
    MOE_EXPERT_PARALLEL_SIZE=2 \
    NUM_GPUS=2 \
    EVAL_MICRO_BATCH_SIZE=2 \
    EVAL_LIMIT=${DOWNSTREAM_LIMIT:-2} \
    GENERATION_MAX_TOKENS=${GENERATION_MAX_TOKENS:-8} \
    TASKS=${TASKS:-piqa,hellaswag,race,mathqa,boolq,arc_challenge,winogrande} \
        "${REPO_ROOT}/tasks/eval_harness/run_paper_eval.sh" \
        "${FINAL_BF16_MODEL_PATH}" \
        "${DOWNSTREAM_RESULT}"
    "${REPO_ROOT}/.venv/bin/python" - "${DOWNSTREAM_RESULT}" <<'PY'
import json
import math
import sys
from pathlib import Path

expected = {
    "piqa",
    "hellaswag",
    "race",
    "mathqa",
    "boolq",
    "arc_challenge",
    "winogrande",
}
raw_path = Path(sys.argv[1])
percent_path = raw_path.with_suffix(".percent.json")
raw = json.loads(raw_path.read_text())
percent = json.loads(percent_path.read_text())
missing_raw = expected - set(raw.get("results", {}))
missing_percent = expected - set(percent)
if missing_raw or missing_percent:
    raise SystemExit(
        f"downstream results incomplete: raw={sorted(missing_raw)} percent={sorted(missing_percent)}"
    )
for task in expected:
    row = percent[task]
    accuracy = float(row["accuracy_percent"])
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 100.0:
        raise SystemExit(f"invalid downstream accuracy for {task}: {accuracy!r}")
print(f"[h100-smoke] downstream_ok tasks={','.join(sorted(expected))}")
PY
fi

echo "[h100-smoke] PASS variant=${VARIANT} model=${MODEL_LABEL}"
