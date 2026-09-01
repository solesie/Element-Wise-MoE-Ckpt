#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TRACE=${TRACE:-$ROOT/results/ds_pretrain_gpt_350M_MoE128_10Kiter_proposal/waf_trace}
WORK=${WORK:-/tmp}
BUDGET=${BUDGET:-900000000}
STEPS=${STEPS:-1001,1002,1003}
SSD_GBPS=${SSD_GBPS:-4.1}
OUT=${OUT:-$ROOT/results/storage_overhead_complete_$(date +%Y%m%d_%H%M%S)}
PY=${PYTHON:-python3}
R=$ROOT/tools/storage_overhead_exact/exact_waf_replay.py
ANALYZE=$ROOT/tools/storage_overhead_exact/analyze_complete.py
BPF=$ROOT/tools/storage_overhead_exact/storage_trace.bt
CACHE=$OUT/plan.pkl
ORIG_PARANOID=$(sysctl -n kernel.perf_event_paranoid)
ORIG_KPTR=$(sysctl -n kernel.kptr_restrict)
BPF_SUDO_PID=
SKIP_CONTROLS=${SKIP_CONTROLS:-0}
SKIP_BPF=${SKIP_BPF:-0}
SKIP_PERF_RECORD=${SKIP_PERF_RECORD:-0}

cleanup() {
  if [[ -n "$BPF_SUDO_PID" ]]; then
    local child
    child=$(pgrep -n -f "^bpftrace .*storage_trace[.]bt$" || true)
    [[ -z "$child" ]] || sudo -n kill -INT "$child" 2>/dev/null || true
    wait "$BPF_SUDO_PID" 2>/dev/null || true
  fi
  sudo -n sysctl -q -w kernel.perf_event_paranoid="$ORIG_PARANOID" >/dev/null || true
  sudo -n sysctl -q -w kernel.kptr_restrict="$ORIG_KPTR" >/dev/null || true
}
trap cleanup EXIT INT TERM

mkdir -p "$OUT" "$OUT/shadow" "$WORK"
sudo -n sysctl -q -w kernel.perf_event_paranoid=1 >/dev/null
sudo -n sysctl -q -w kernel.kptr_restrict=0 >/dev/null

case1() {
  local name=$1 writer=$2 executor=$3 opened=$4 sync_mode=$5
  sync
  "$PY" "$R" --trace "$TRACE" --plan-cache "$CACHE" --work "$WORK" --steps "$STEPS" \
    --budget "$BUDGET" --writer "$writer" --executor "$executor" --open-mode "$opened" \
    --sync "$sync_mode" --sink disk \
    | jq -c --arg case "$name" '. + {case:$case}' | tee -a "$OUT/cases.jsonl"
}

# One uninstrumented headline and five ordered controls; each replays exactly three steps.
if [[ "$SKIP_CONTROLS" != 1 ]]; then
  : > "$OUT/cases.jsonl"
  case1 full_fsync_proc8       torch proc8 persistent     fsync
  case1 torch_flush_proc8      torch proc8 persistent     flush
  case1 torch_none_proc8       torch proc8 persistent     none
  case1 raw_actual_proc8       raw   proc8 persistent     flush
  case1 raw_preopen_proc8      raw   proc8 all-persistent flush
  case1 raw_preopen_proc1      raw   proc1 all-persistent flush
else
  test -s "$OUT/cases.jsonl"
fi

# Shadow 1: aggregate every syscall, off-CPU interval, lock, ext4 sync, journal,
# writeback, and block request. perf is gated to the exact timed interval too.
if [[ "$SKIP_BPF" != 1 ]]; then
  sudo -n bpftrace -f json -B none -o "$OUT/shadow/bpftrace.jsonl" "$BPF" &
  BPF_SUDO_PID=$!
  for _ in $(seq 1 200); do
    grep -q TRACE_READY "$OUT/shadow/bpftrace.jsonl" 2>/dev/null && break
    sleep 0.05
  done
  grep -q TRACE_READY "$OUT/shadow/bpftrace.jsonl"

  [[ ! -p "$OUT/shadow/perf.ctl" ]] || unlink "$OUT/shadow/perf.ctl"
  [[ ! -p "$OUT/shadow/perf.ack" ]] || unlink "$OUT/shadow/perf.ack"
  mkfifo "$OUT/shadow/perf.ctl" "$OUT/shadow/perf.ack"
  perf stat -D -1 --control "fifo:$OUT/shadow/perf.ctl,$OUT/shadow/perf.ack" -x, \
    -o "$OUT/shadow/perf_stat.csv" \
    -e task-clock,cycles,instructions,context-switches,cpu-migrations,page-faults -- \
    "$PY" "$R" --trace "$TRACE" --plan-cache "$CACHE" --work "$WORK" --steps "$STEPS" \
    --budget "$BUDGET" --writer torch --executor proc8 --open-mode persistent --sync fsync \
    --perf-ctl "$OUT/shadow/perf.ctl" --perf-ack "$OUT/shadow/perf.ack" --markers \
    > "$OUT/shadow/bpf_perf_workload.json"
  unlink "$OUT/shadow/perf.ctl"
  unlink "$OUT/shadow/perf.ack"

  BPF_CHILD=$(pgrep -n -f "^bpftrace .*storage_trace[.]bt$")
  sudo -n kill -INT "$BPF_CHILD"
  wait "$BPF_SUDO_PID" || true
  BPF_SUDO_PID=
  sudo -n chown "$(id -u):$(id -g)" "$OUT/shadow/bpftrace.jsonl"
fi

# Shadow 2: native user+kernel sampled stacks. Its elapsed time is never a headline value.
if [[ "$SKIP_PERF_RECORD" != 1 ]]; then
  [[ ! -p "$OUT/shadow/record.ctl" ]] || unlink "$OUT/shadow/record.ctl"
  [[ ! -p "$OUT/shadow/record.ack" ]] || unlink "$OUT/shadow/record.ack"
  mkfifo "$OUT/shadow/record.ctl" "$OUT/shadow/record.ack"
  perf record -D -1 --control "fifo:$OUT/shadow/record.ctl,$OUT/shadow/record.ack" \
    -F 199 -e cycles -g --call-graph dwarf,8192 -o "$OUT/shadow/perf.data" -- \
    env PYTHONPERFSUPPORT=1 "$PY" "$R" --trace "$TRACE" --plan-cache "$CACHE" \
    --work "$WORK" --steps "$STEPS" --budget "$BUDGET" --writer torch --executor proc8 \
    --open-mode persistent --sync fsync --perf-ctl "$OUT/shadow/record.ctl" \
    --perf-ack "$OUT/shadow/record.ack" > "$OUT/shadow/perf_workload.json"
  unlink "$OUT/shadow/record.ctl"
  unlink "$OUT/shadow/record.ack"
fi
sudo -n perf report -f --stdio --no-children --sort dso,symbol --percent-limit 0.5 \
  -i "$OUT/shadow/perf.data" > "$OUT/shadow/perf_report.txt"
sudo -n perf report -f --stdio --no-children --sort dso -g none --percent-limit 0 \
  -i "$OUT/shadow/perf.data" > "$OUT/shadow/perf_dso.txt"

# Shadow 3: Python stacks that actually hold the GIL, including all rank subprocesses.
py-spy record --subprocesses --gil --native --rate 49 --format raw \
  --output "$OUT/shadow/pyspy_gil.raw" -- \
  "$PY" "$R" --trace "$TRACE" --plan-cache "$CACHE" --work "$WORK" --steps "$STEPS" \
  --budget "$BUDGET" --writer torch --executor proc8 --open-mode persistent --sync fsync \
  > "$OUT/shadow/pyspy_workload.json" || true

"$PY" "$ANALYZE" --cases "$OUT/cases.jsonl" --bpf "$OUT/shadow/bpftrace.jsonl" \
  --perf-stat "$OUT/shadow/perf_stat.csv" --perf-dso "$OUT/shadow/perf_dso.txt" \
  --pyspy "$OUT/shadow/pyspy_gil.raw" \
  --ssd-gbps "$SSD_GBPS" --output "$OUT/summary.json"

echo "$OUT"
