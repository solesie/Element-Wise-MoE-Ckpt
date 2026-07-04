#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-"${REPO_ROOT}/results/10k_compare_test_dram_micro8/${STAMP}"}

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/runner.log") 2>&1

echo "[runner] start=$(date --iso-8601=seconds)"
echo "[runner] repo=${REPO_ROOT}"
echo "[runner] run_root=${RUN_ROOT}"

cd "${REPO_ROOT}" || exit 2

timeout 5s git rev-parse HEAD > "${RUN_ROOT}/git_head.txt" 2>&1 || true
timeout 5s git status --short --untracked-files=no > "${RUN_ROOT}/git_status_short.txt" 2>&1 || true
timeout 10s git diff --binary -- . ':(exclude)results' ':(exclude).torch_extensions' \
  > "${RUN_ROOT}/code_diff.patch" 2>&1 || true
nvidia-smi > "${RUN_ROOT}/nvidia_smi_start.txt" 2>&1 || true
df -h /home/paul /dev/shm > "${RUN_ROOT}/disk_start.txt" 2>&1 || true

COMMON_ENV=(
  TRAIN_ITERS=10000
  GLOBAL_BATCH_SIZE=256
  BATCH_SIZE=8
  EVAL_INTERVAL=1000
  EVAL_ITERS=20
  LOG_INTERVAL=10
  NUM_GPUS=8
  NUM_GPUS_PERNODE=8
  MP_SIZE=1
  PP_SIZE=2
  EP_SIZE=64
  MOE_EXPERT_PARALLEL_SIZE=2
  ZERO_STAGE=0
  TRAIN_PRECISION=bf16
  NO_TENSORBOARD=1
  NO_MEGATRON_CKPT=1
  DATA_CACHE_PATH="${REPO_ROOT}/results/data-cache"
  TORCH_EXTENSIONS_DIR="${REPO_ROOT}/.torch_extensions"
  TORCH_CUDA_ARCH_LIST=9.0
  MEGATRON_FUSED_KERNEL_ARCH_LIST=90
)

summarize_logs() {
  local out_dir="$1"
  local phase="$2"
  python3 - "${out_dir}" "${phase}" <<'PY'
import json
import pathlib
import re
import statistics
import sys

out = pathlib.Path(sys.argv[1])
phase = sys.argv[2]
logs = sorted((out / "log").glob("*.log"))
text = "\n".join(p.read_text(errors="ignore") for p in logs)
iters = []
for m in re.finditer(
    r"iteration\s+(\d+)/\s*(\d+) .*?elapsed time per iteration \(ms\): ([0-9.]+).*?"
    r"loss: ([0-9.E+-]+).*?lm loss: ([0-9.E+-]+).*?"
    r"number of skipped iterations:\s+(\d+) \| number of nan iterations:\s+(\d+)",
    text,
):
    iters.append({
        "iter": int(m.group(1)),
        "total": int(m.group(2)),
        "iter_ms": float(m.group(3)),
        "loss": float(m.group(4)),
        "lm_loss": float(m.group(5)),
        "skipped": int(m.group(6)),
        "nan": int(m.group(7)),
    })
vals = []
for m in re.finditer(r"validation loss at iteration\s+(\d+).*?LM loss: ([0-9.E+-]+)", text):
    vals.append({"iter": int(m.group(1)), "lm_loss": float(m.group(2))})
faults = []
for f in sorted((out / "log").glob("*fault*.jsonl")):
    for line in f.read_text(errors="ignore").splitlines():
        if line.strip():
            try:
                faults.append(json.loads(line))
            except json.JSONDecodeError:
                faults.append({"raw": line})
oracle_counts = {}
oracle_dir = out / "oracle"
if oracle_dir.exists():
    for kind in ("independent_loss", "independent_rank", "independent_restore", "independent_timing"):
        files = sorted(oracle_dir.glob(f"{kind}_rank*.jsonl"))
        oracle_counts[kind] = {
            "files": len(files),
            "lines": [sum(1 for line in f.read_text(errors="ignore").splitlines() if line.strip()) for f in files],
        }
    timing = []
    for f in sorted(oracle_dir.glob("independent_timing_rank*.jsonl")):
        for line in f.read_text(errors="ignore").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "elapsed_ms" in row:
                    timing.append(float(row["elapsed_ms"]))
    if timing:
        oracle_counts["timing_elapsed_ms"] = {
            "mean": statistics.mean(timing),
            "min": min(timing),
            "max": max(timing),
            "rows": len(timing),
        }
summary = {
    "phase": phase,
    "out": str(out),
    "log_files": [str(p) for p in logs],
    "last_iterations": iters[-20:],
    "validation": vals,
    "faults": faults,
    "oracle": oracle_counts,
    "success_lines": len(re.findall(r"exits successfully", text)),
    "failure_lines": re.findall(r"(RuntimeError:.*|OutOfMemoryError:.*|exits with return code\s*=\s*\d+)", text),
}
(out / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"[runner] wrote {out / 'run_summary.json'}")
PY
}

summarize_lfcl_counts() {
  local out_dir="$1"
  python3 - "${out_dir}" <<'PY'
import collections
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
counter = collections.Counter()
by_rank = collections.defaultdict(collections.Counter)
records = 0
for f in sorted((out / "log").glob("lfcl_fire_rank*.jsonl")):
    for line in f.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records += 1
        rank = int(row.get("rank", -1))
        for layer, expert in row.get("experts", []):
            key = f"L{int(layer):02d}_E{int(expert):02d}"
            counter[key] += 1
            by_rank[str(rank)][key] += 1
summary = {
    "records": records,
    "total_fires": int(sum(counter.values())),
    "expert_counts": dict(sorted(counter.items())),
    "by_rank": {rank: dict(sorted(cnt.items())) for rank, cnt in sorted(by_rank.items())},
}
(out / "log" / "lfcl_compaction_counts.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
with (out / "log" / "lfcl_compaction_counts.tsv").open("w", encoding="utf-8") as handle:
    handle.write("expert\tcount\n")
    for key, value in sorted(counter.items()):
        handle.write(f"{key}\t{value}\n")
print(f"[runner] wrote {out / 'log' / 'lfcl_compaction_counts.json'}")
PY
}

run_phase() {
  local phase="$1"
  shift
  local out_dir="${RUN_ROOT}/${phase}"
  mkdir -p "${out_dir}/log"
  echo "[runner] phase=${phase} start=$(date --iso-8601=seconds)"
  env "${COMMON_ENV[@]}" OUTPUT_BASEPATH="${out_dir}" "$@" 2>&1 | tee -a "${out_dir}/log/${phase}_wrapper.log"
  local status=${PIPESTATUS[0]}
  echo "[runner] phase=${phase} status=${status} end=$(date --iso-8601=seconds)"
  summarize_logs "${out_dir}" "${phase}" || true
  return "${status}"
}

PROPOSAL_OUT="${RUN_ROOT}/proposal"
BASELINE_OUT="${RUN_ROOT}/baseline"

run_phase proposal \
  PROPOSAL_IO_BACKEND=file \
  PROPOSAL_CP_TARGET=test_dram \
  PROPOSAL_FDP_VALIDATE_READBACK=0 \
  PROPOSAL_ORACLE_ENABLE=1 \
  ORACLE_INTERVAL=1000 \
  FAULT_INTERVAL=100 \
  LFCL_PLAN_POLICY=lossy_log_size \
  PROPOSAL_TOPK=0.04 \
  PROPOSAL_LFCL_FIRE_LOG="${PROPOSAL_OUT}/log/lfcl_fire_rank{rank}.jsonl" \
  PROPOSAL_LFCL_OVERLAP_LOG="${PROPOSAL_OUT}/log/lfcl_overlap.jsonl" \
  bash examples_deepspeed/MoE/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal.sh
proposal_status=$?
summarize_lfcl_counts "${PROPOSAL_OUT}" || true
if [ "${proposal_status}" -ne 0 ]; then
  echo "{\"status\":\"proposal_failed\",\"code\":${proposal_status}}" > "${RUN_ROOT}/status.json"
  nvidia-smi > "${RUN_ROOT}/nvidia_smi_end.txt" 2>&1 || true
  exit "${proposal_status}"
fi

run_phase baseline \
  bash examples_deepspeed/MoE/ds_pretrain_gpt_125M_MoE64_10Kiter_baseline.sh
baseline_status=$?

nvidia-smi > "${RUN_ROOT}/nvidia_smi_end.txt" 2>&1 || true
df -h /home/paul /dev/shm > "${RUN_ROOT}/disk_end.txt" 2>&1 || true

if [ "${baseline_status}" -ne 0 ]; then
  echo "{\"status\":\"baseline_failed\",\"code\":${baseline_status}}" > "${RUN_ROOT}/status.json"
  exit "${baseline_status}"
fi

echo "{\"status\":\"complete\"}" > "${RUN_ROOT}/status.json"
echo "[runner] complete=$(date --iso-8601=seconds)"
