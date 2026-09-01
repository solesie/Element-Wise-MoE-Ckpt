#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"
PYTHON="${REPO_ROOT}/.venv/bin/python"
DELETE_BF16_AFTER_ACCURACY=${DELETE_BF16_AFTER_ACCURACY:-0}

AUTOMATION_ROOT="${REPO_ROOT}/results/automation_125m_19b_suite"
WORKFLOW_LOG="${AUTOMATION_ROOT}/workflow.log"
mkdir -p "${AUTOMATION_ROOT}"
exec > >(tee -a "${WORKFLOW_LOG}") 2>&1

MOC4_ROOT="${REPO_ROOT}/results/ds_pretrain_gpt_125M_MoE64_10Kiter_moc"
PROP4_ROOT="${REPO_ROOT}/results/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal"
BASE_ROOT="${REPO_ROOT}/results/ds_pretrain_gpt_125M_MoE64_10Kiter_baseline"
MOC1_ROOT="${REPO_ROOT}/results/ds_pretrain_gpt_125M_MoE64_10Kiter_moc_1pct"
PROP1_ROOT="${REPO_ROOT}/results/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal_1pct"

MOC4_SCRIPT="${REPO_ROOT}/examples_deepspeed/MoE/ds_pretrain_gpt_125M_MoE64_10Kiter_moc.sh"
PROP4_SCRIPT="${REPO_ROOT}/examples_deepspeed/MoE/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal.sh"
BASE_SCRIPT="${REPO_ROOT}/examples_deepspeed/MoE/ds_pretrain_gpt_125M_MoE64_10Kiter_baseline.sh"
MOC1_SCRIPT="${REPO_ROOT}/examples_deepspeed/MoE/ds_pretrain_gpt_125M_MoE64_10Kiter_moc_1pct.sh"
PROP1_SCRIPT="${REPO_ROOT}/examples_deepspeed/MoE/ds_pretrain_gpt_125M_MoE64_10Kiter_proposal_1pct.sh"
EVAL_SCRIPT="${REPO_ROOT}/tasks/eval_harness/run_paper_eval.sh"

TASKS_7="piqa,hellaswag,race,mathqa,boolq,arc_challenge,winogrande"

log() {
    printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

fail() {
    log "FATAL: $*"
    exit 1
}

on_error() {
    local status=$?
    log "ABORT status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
    exit "${status}"
}
trap on_error ERR

assert_file() {
    [ -f "$1" ] || fail "required file missing: $1"
}

assert_absent() {
    [ ! -e "$1" ] || fail "refusing to overwrite existing path: $1"
}

wait_for_gpu_idle() {
    local deadline=$((SECONDS + 600))
    local active
    while true; do
        active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {print $1}' | sort -u | paste -sd, -)
        if [ -z "${active}" ]; then
            log "all GPUs are compute-idle"
            return 0
        fi
        if [ "${SECONDS}" -ge "${deadline}" ]; then
            fail "GPU processes did not drain within 10 minutes: ${active}"
        fi
        log "waiting for GPU processes to drain: ${active}"
        sleep 15
    done
}

verify_bf16() {
    local checkpoint_root=$1
    "${REPO_ROOT}/.venv/bin/python" - "${checkpoint_root}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
latest_file = root / "latest"
if not latest_file.is_file():
    raise SystemExit(f"missing latest: {latest_file}")
tag = latest_file.read_text().strip()
if not tag or "/" in tag or tag in {".", ".."}:
    raise SystemExit(f"invalid latest tag: {tag!r}")
tag_dir = root / tag
manifest_path = tag_dir / "model_only_manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"missing manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
records = manifest.get("files")
if not isinstance(records, list) or not records:
    raise SystemExit("manifest has no payload records")
total = 0
for record in records:
    path = tag_dir / record["path"]
    if not path.is_file():
        raise SystemExit(f"missing payload: {path}")
    size = path.stat().st_size
    if size != int(record["bytes"]):
        raise SystemExit(f"size mismatch: {path}: {size} != {record['bytes']}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != record["sha256"]:
        raise SystemExit(f"checksum mismatch: {path}")
    total += size
if total != int(manifest.get("total_bytes", -1)):
    raise SystemExit(f"total byte mismatch: {total} != {manifest.get('total_bytes')}")
if int(manifest.get("iteration", -1)) != 10000:
    raise SystemExit(f"unexpected checkpoint iteration: {manifest.get('iteration')}")
print(f"BF16_OK root={root} tag={tag} files={len(records)} bytes={total}")
PY
}

verify_accuracy() {
    local result=$1
    "${REPO_ROOT}/.venv/bin/python" - "${result}" <<'PY'
import json
import math
import pathlib
import sys

expected = {"piqa", "hellaswag", "race", "mathqa", "boolq", "arc_challenge", "winogrande"}
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
results = data.get("results", {})
if set(results) != expected:
    raise SystemExit(f"task mismatch: got={sorted(results)}")
for task, metrics in results.items():
    metric = "acc_norm" if "acc_norm" in metrics else "acc"
    value = float(metrics[metric])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SystemExit(f"invalid {task}.{metric}: {value}")
print(f"ACCURACY_OK path={path} tasks={len(results)}")
PY
}

run_accuracy() {
    local root=$1
    local result="${root}/accuracy_7tasks.json"
    local eval_log="${root}/log/accuracy_7tasks.log"
    mkdir -p "${root}/log"
    assert_file "${root}/bf16_model/latest"
    [ ! -e "${result}" ] || fail "refusing to overwrite accuracy result: ${result}"
    wait_for_gpu_idle
    log "accuracy start: ${root}"
    if ! env \
        TASKS="${TASKS_7}" \
        EVAL_MICRO_BATCH_SIZE=8 \
        GENERATION_MAX_TOKENS=16 \
        PP_SIZE=2 \
        TP_SIZE=1 \
        MOE_EXPERT_PARALLEL_SIZE=4 \
        NUM_GPUS=8 \
        bash "${EVAL_SCRIPT}" "${root}/bf16_model" "${result}" \
        >"${eval_log}" 2>&1; then
        tail -n 120 "${eval_log}" || true
        fail "accuracy failed: ${root}"
    fi
    verify_accuracy "${result}"
    tail -n 35 "${eval_log}" || true
    log "accuracy complete: ${result}"
}

latest_iteration() {
    local root=$1
    local training_log
    training_log=$(find "${root}/log" -maxdepth 1 -type f -name '*.log' 2>/dev/null \
        | grep -v 'accuracy_7tasks.log' | sort | tail -n 1 || true)
    if [ -z "${training_log}" ]; then
        printf '0\n'
        return
    fi
    rg -o 'iteration +[0-9]+/ +10000' "${training_log}" 2>/dev/null \
        | tail -n 1 | sed -E 's/iteration +([0-9]+).*/\1/' || printf '0\n'
}

verify_moc_initial() {
    local root=$1
    local interval=$2
    "${REPO_ROOT}/.venv/bin/python" - "${root}" "${interval}" <<'PY'
import glob
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
interval = int(sys.argv[2])
paths = sorted(glob.glob(str(root / "log" / "moc_lfcl_rank*.jsonl")))
if len(paths) != 8:
    raise SystemExit(f"expected 8 LFCL fire logs, got {len(paths)}")
events = []
for path_text in paths:
    path = pathlib.Path(path_text)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) < 3:
        raise SystemExit(f"too few LFCL events: {path}: {len(records)}")
    rank = int(records[0]["rank"])
    layers = (1, 3, 5) if rank < 4 else (7, 9, 11)
    for index, record in enumerate(records):
        expected_step = 1 + interval * index
        if int(record["step"]) != expected_step:
            raise SystemExit(f"cadence mismatch {path}: {record['step']} != {expected_step}")
        if int(record["count"]) != 16:
            raise SystemExit(f"count mismatch {path} step={record['step']}: {record['count']}")
        expected_layer = layers[index % 3]
        experts = record["experts"]
        if [int(item[0]) for item in experts] != [expected_layer] * 16:
            raise SystemExit(f"layer mismatch {path} step={record['step']}")
        if [int(item[1]) for item in experts] != list(range(16)):
            raise SystemExit(f"local expert mismatch {path} step={record['step']}")
    events.append(len(records))
waf = sorted(glob.glob(str(root / "waf_trace" / "waf_rank*.jsonl")))
if len(waf) != 8:
    raise SystemExit(f"expected 8 WAF traces, got {len(waf)}")
print(f"MOC_INITIAL_OK interval={interval} per_rank_events={events} fire_count=16 waf_ranks={len(waf)}")
PY
}

verify_proposal_initial() {
    local root=$1
    local target_ratio=$2
    local age_period=$3
    "${REPO_ROOT}/.venv/bin/python" - "${root}" "${target_ratio}" "${age_period}" <<'PY'
import glob
import json
import math
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
target = float(sys.argv[2])
age_period = int(sys.argv[3])
telemetry = root / "log" / "approx_probe_rank0.jsonl"
if not telemetry.is_file():
    raise SystemExit(f"missing cutoff/write telemetry: {telemetry}")
records = [json.loads(line) for line in telemetry.read_text().splitlines() if line.strip()]
records = [r for r in records if 81 <= int(r.get("step", -1)) <= 220]
if len(records) < 40:
    raise SystemExit(f"too few initial telemetry records: {len(records)}")
selected = sum(float(r["actual_selected_ratio"]) for r in records) / len(records)
written = sum(float(r["actual_written_ratio"]) for r in records) / len(records)
written_fraction = written / target
if not math.isfinite(written_fraction) or written_fraction < 0.90:
    raise SystemExit(f"initial written/target below 0.90: {written_fraction}")
training_logs = sorted((root / "log").glob("gpt-*.log"))
if not training_logs:
    raise SystemExit("training log missing")
h_records = {}
pattern = re.compile(r"scale @iter(80|100|200) .*coverage=([0-9.]+)")
for match in pattern.finditer(training_logs[-1].read_text(errors="replace")):
    h_records[int(match.group(1))] = float(match.group(2))
if set(h_records) != {80, 100, 200}:
    raise SystemExit(f"missing H refresh steps: got={sorted(h_records)}")
for step, coverage in h_records.items():
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise SystemExit(f"invalid H coverage at {step}: {coverage}")
waf = sorted(glob.glob(str(root / "waf_trace" / "waf_rank*.jsonl")))
if len(waf) != 8:
    raise SystemExit(f"expected 8 WAF traces, got {len(waf)}")
print(
    f"PROPOSAL_INITIAL_OK target={target:.4%} selected_mean={selected:.4%} "
    f"written_mean={written:.4%} written_over_target={written_fraction:.4f} "
    f"age_force={age_period} h_coverage={h_records} waf_ranks={len(waf)}"
)
PY
}

verify_oracle_1000() {
    local root=$1
    "${REPO_ROOT}/.venv/bin/python" - "${root}" <<'PY'
import glob
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rank_logs = sorted(glob.glob(str(root / "oracle" / "independent_rank_rank*.jsonl")))
if not rank_logs:
    raise SystemExit("oracle rank logs missing")
steps = set()
for path_text in rank_logs:
    for line in pathlib.Path(path_text).read_text().splitlines():
        if line.strip():
            steps.add(int(json.loads(line)["step"]))
if 1000 not in steps:
    raise SystemExit(f"oracle step 1000 missing: {sorted(steps)}")
joint = glob.glob(str(root / "oracle" / "joint_saliency" / "joint_saliency_step0001000_rank*.npz"))
if len(joint) != 8:
    raise SystemExit(f"expected 8 joint-saliency shards at step 1000, got {len(joint)}")
print(f"ORACLE_1000_OK rank_logs={len(rank_logs)} joint_shards={len(joint)}")
PY

    "${PYTHON}" - "${root}" <<'PY'
import glob
import hashlib
import json
import math
import pathlib
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
h_root = root / "oracle" / "h_rank"
logs = sorted(h_root.glob("h_rank_rank*.jsonl"))
snapshots = sorted(h_root.glob("h_rank_step0001000_rank*.npz"))
if len(logs) != 8:
    raise SystemExit(f"expected 8 H-rank logs, got {len(logs)}")
if len(snapshots) != 8:
    raise SystemExit(f"expected 8 H-rank snapshots at step 1000, got {len(snapshots)}")
rows_by_rank = {}
for log_path in logs:
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("mode") == "curvature_progress_h_refresh" and int(row["step"]) == 1000:
            rows_by_rank[int(row["rank"])] = row
if sorted(rows_by_rank) != list(range(8)):
    raise SystemExit(f"H-rank step 1000 rows missing ranks: {sorted(rows_by_rank)}")
sample_total = 0
for snapshot_path in snapshots:
    with np.load(snapshot_path, allow_pickle=False) as payload:
        if payload["schema_version"].tolist() != [2]:
            raise SystemExit(f"bad H-rank schema: {snapshot_path}")
        if payload["step"].tolist() != [1000]:
            raise SystemExit(f"bad H-rank step: {snapshot_path}")
        rank = int(payload["rank"][0])
        abs_h = np.asarray(payload["abs_h"], dtype=np.float32)
        v_t = np.asarray(payload["v_t"], dtype=np.float32)
        group_ids = np.asarray(payload["group_ids"], dtype=np.int64)
        group_size = int(np.asarray(payload["group_size"]).reshape(-1)[0])
        if abs_h.size <= 0 or not (
            abs_h.shape == v_t.shape == group_ids.shape
        ):
            raise SystemExit(f"empty/mismatched H-rank arrays: {snapshot_path}")
        if group_size != 8:
            raise SystemExit(
                f"unexpected H-rank group_size={group_size}: {snapshot_path}"
            )
        _, counts = np.unique(group_ids, return_counts=True)
        if counts.size == 0 or not np.all(counts == group_size):
            raise SystemExit(f"incomplete row-8 H-rank groups: {snapshot_path}")
        if not np.isfinite(abs_h).all() or not np.isfinite(v_t).all():
            raise SystemExit(f"non-finite H-rank arrays: {snapshot_path}")
        row = rows_by_rank[rank]
        digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        if digest != row.get("refresh_snapshot_sha256"):
            raise SystemExit(f"H-rank checksum mismatch: {snapshot_path}")
        for key in ("abs_h_vs_v_spearman", "abs_h_min", "abs_h_p50", "abs_h_p99", "abs_h_max"):
            if key not in row or not math.isfinite(float(row[key])):
                raise SystemExit(f"missing/non-finite {key}: {snapshot_path}")
        sample_total += int(abs_h.size)
print(f"H_RANK_1000_OK logs={len(logs)} snapshots={len(snapshots)} sample_elements={sample_total}")
PY
}

run_training() {
    local label=$1
    local script=$2
    local root=$3
    local mode=$4
    local budget=$5
    local age_period=$6
    local console="${AUTOMATION_ROOT}/${label}.console.log"
    local observed_initial=0
    local observed_oracle=0
    local iteration=0

    assert_file "${script}"
    bash -n "${script}"
    assert_absent "${root}"
    wait_for_gpu_idle
    log "training start: label=${label} script=${script}"

    if [ "${mode}" = "proposal" ]; then
        env \
            PROPOSAL_APPROX_PROBE_TELEMETRY_LOG="${root}/log/approx_probe_rank0.jsonl" \
            PROPOSAL_LFCL_FIRE_LOG="${root}/log/proposal_lfcl_rank{rank}.jsonl" \
            bash "${script}" >"${console}" 2>&1 &
    else
        bash "${script}" >"${console}" 2>&1 &
    fi
    local training_pid=$!
    log "training supervisor pid=${training_pid} console=${console}"

    while [ -d "/proc/${training_pid}" ]; do
        sleep 30
        iteration=$(latest_iteration "${root}")
        log "training progress: label=${label} iteration=${iteration}"
        if [ "${observed_initial}" -eq 0 ] && [ "${iteration}" -ge 220 ]; then
            case "${mode}" in
                moc4) verify_moc_initial "${root}" 8 ;;
                moc1) verify_moc_initial "${root}" 32 ;;
                proposal) verify_proposal_initial "${root}" "${budget}" "${age_period}" ;;
                baseline) log "baseline initial log reached iteration=${iteration}" ;;
                *) fail "unknown training mode: ${mode}" ;;
            esac
            observed_initial=1
        fi
        if [ "${mode}" = "proposal" ] && [ "${observed_oracle}" -eq 0 ] \
            && [ "${iteration}" -ge 1100 ]; then
            verify_oracle_1000 "${root}"
            observed_oracle=1
        fi
    done

    if ! wait "${training_pid}"; then
        tail -n 160 "${console}" || true
        fail "training failed: ${label}"
    fi
    if [ "${observed_initial}" -eq 0 ]; then
        fail "training exited before initial observation: ${label} iteration=${iteration}"
    fi
    if [ "${mode}" = "proposal" ] && [ "${observed_oracle}" -eq 0 ]; then
        fail "proposal exited before oracle observation: ${label} iteration=${iteration}"
    fi
    verify_bf16 "${root}/bf16_model"
    tail -n 80 "${console}" || true
    log "training complete: ${label}"
}

compare_4pct_accuracy() {
    local moc_result="${MOC4_ROOT}/accuracy_7tasks.json"
    local proposal_result="${PROP4_ROOT}/accuracy_7tasks.json"
    local report="${AUTOMATION_ROOT}/accuracy_moc4_vs_proposal4.txt"
    verify_accuracy "${moc_result}"
    verify_accuracy "${proposal_result}"
    "${REPO_ROOT}/.venv/bin/python" - "${moc_result}" "${proposal_result}" >"${report}" <<'PY'
import json
import pathlib
import sys

moc = json.loads(pathlib.Path(sys.argv[1]).read_text())["results"]
proposal = json.loads(pathlib.Path(sys.argv[2]).read_text())["results"]
order = ["piqa", "hellaswag", "race", "mathqa", "boolq", "arc_challenge", "winogrande"]
any_001 = False
print("task\tmetric\tmoc_percent\tproposal_percent\tdelta_proposal_minus_moc_pp")
for task in order:
    metric = "acc_norm" if "acc_norm" in moc[task] else "acc"
    a = 100.0 * float(moc[task][metric])
    b = 100.0 * float(proposal[task][metric])
    delta = b - a
    any_001 |= abs(delta) >= 0.01
    print(f"{task}\t{metric}\t{a:.6f}\t{b:.6f}\t{delta:+.6f}")
print(f"any_abs_delta_ge_0.01_percentage_point={str(any_001).lower()}")
PY
    cat "${report}"
    log "4% accuracy comparison complete: ${report}"
}

delete_verified_bf16() {
    local checkpoint_root=$1
    verify_bf16 "${checkpoint_root}"
    case "${checkpoint_root}" in
        "${MOC4_ROOT}/bf16_model"|"${PROP4_ROOT}/bf16_model"|"${BASE_ROOT}/bf16_model") ;;
        *) fail "refusing unexpected BF16 deletion target: ${checkpoint_root}" ;;
    esac
    rm -rf -- "${checkpoint_root}"
    [ ! -e "${checkpoint_root}" ] || fail "BF16 deletion failed: ${checkpoint_root}"
    log "deleted verified BF16 checkpoint: ${checkpoint_root}"
}

maybe_delete_verified_bf16() {
    local checkpoint_root=$1
    if [ "${DELETE_BF16_AFTER_ACCURACY}" = "1" ]; then
        delete_verified_bf16 "${checkpoint_root}"
    else
        log "preserving BF16 checkpoint: ${checkpoint_root}"
    fi
}

wait_for_current_moc4() {
    log "observing already-running MoC 4% experiment without sending it signals"
    while pgrep -f 'pretrain_gpt.py.*ds_pretrain_gpt_125M_MoE64_10Kiter_moc/bf16_model' >/dev/null; do
        local iteration
        iteration=$(latest_iteration "${MOC4_ROOT}")
        log "current MoC 4% progress: iteration=${iteration}"
        sleep 60
    done
    wait_for_gpu_idle
    verify_bf16 "${MOC4_ROOT}/bf16_model"
    verify_moc_initial "${MOC4_ROOT}" 8
    log "current MoC 4% completed normally"
}

main() {
    local start_stage=${SUITE_START_STAGE:-moc4}
    for file in "${MOC4_SCRIPT}" "${PROP4_SCRIPT}" "${BASE_SCRIPT}" \
        "${MOC1_SCRIPT}" "${PROP1_SCRIPT}" "${EVAL_SCRIPT}"; do
        assert_file "${file}"
        bash -n "${file}"
    done

    case "${start_stage}" in
        moc4)
            wait_for_current_moc4
            run_accuracy "${MOC4_ROOT}"
            ;&
        proposal4)
            verify_accuracy "${MOC4_ROOT}/accuracy_7tasks.json"
            run_training proposal4 "${PROP4_SCRIPT}" "${PROP4_ROOT}" proposal 0.04 128
            run_accuracy "${PROP4_ROOT}"
            compare_4pct_accuracy
            maybe_delete_verified_bf16 "${MOC4_ROOT}/bf16_model"
            maybe_delete_verified_bf16 "${PROP4_ROOT}/bf16_model"
            ;&
        baseline)
            run_training baseline "${BASE_SCRIPT}" "${BASE_ROOT}" baseline 0 0
            run_accuracy "${BASE_ROOT}"
            maybe_delete_verified_bf16 "${BASE_ROOT}/bf16_model"
            ;&
        moc1)
            run_training moc1 "${MOC1_SCRIPT}" "${MOC1_ROOT}" moc1 0.01 0
            run_accuracy "${MOC1_ROOT}"
            ;&
        proposal1)
            run_training proposal1 "${PROP1_SCRIPT}" "${PROP1_ROOT}" proposal 0.01 512
            run_accuracy "${PROP1_ROOT}"
            ;;
        *) fail "invalid SUITE_START_STAGE=${start_stage}" ;;
    esac

    log "SUITE_COMPLETE"
}

main "$@"
