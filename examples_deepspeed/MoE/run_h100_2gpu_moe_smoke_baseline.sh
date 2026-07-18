#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SMOKE_VARIANT=baseline exec "${SCRIPT_DIR}/run_h100_2gpu_moe_smoke.sh" "$@"
