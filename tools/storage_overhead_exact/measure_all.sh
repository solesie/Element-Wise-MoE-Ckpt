#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
if [[ -z "${WORK:-}" ]]; then
  echo "WORK must be an actual NVMe mount path (not tmpfs)" >&2
  exit 2
fi

BASE_OUT=${OUT:-$ROOT/results/storage_overhead_all_$(date +%Y%m%d_%H%M%S)}
for workload in log full_checkpoint_10pct; do
  WORKLOAD=$workload BUDGET=full SSD_GBPS=${SSD_GBPS:-6.8} \
    OUT=$BASE_OUT/$workload \
    bash "$ROOT/tools/storage_overhead_exact/measure_complete.sh"
done

echo "$BASE_OUT"
