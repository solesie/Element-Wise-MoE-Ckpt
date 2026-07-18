#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SMOKE_VARIANT=proposal exec "${SCRIPT_DIR}/run_h100_2gpu_moe_smoke.sh" "$@"
