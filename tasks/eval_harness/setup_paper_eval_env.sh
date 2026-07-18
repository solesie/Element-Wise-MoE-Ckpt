#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TRAIN_PYTHON=${TRAIN_PYTHON:-"${REPO_ROOT}/.venv/bin/python"}
DEPS_DIR=${EVAL_DEPS_DIR:-"${HOME}/.cache/element-wise-moe-eval/lm_eval_0_2"}
STAMP="lm-eval-0.2.0-paper-overlay-v1"

if [ ! -x "${TRAIN_PYTHON}" ]; then
    echo "Training Python not found: ${TRAIN_PYTHON}" >&2
    exit 2
fi

if [ -f "${DEPS_DIR}/.stamp" ] && [ "$(<"${DEPS_DIR}/.stamp")" = "${STAMP}" ]; then
    echo "paper eval dependencies already ready: ${DEPS_DIR}"
    exit 0
fi

TMP_DIR="${DEPS_DIR}.tmp.$$"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"
trap 'rm -rf "${TMP_DIR}"' EXIT

"${TRAIN_PYTHON}" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --target "${TMP_DIR}" \
    'lm-eval==0.2.0' \
    'sqlitedict==1.6.0'

cp "${SCRIPT_DIR}/legacy_overlay/lm_eval/metrics.py" "${TMP_DIR}/lm_eval/metrics.py"
cp "${SCRIPT_DIR}/legacy_overlay/lm_eval/models/__init__.py" "${TMP_DIR}/lm_eval/models/__init__.py"
cp "${SCRIPT_DIR}/legacy_overlay/lm_eval/tasks/__init__.py" "${TMP_DIR}/lm_eval/tasks/__init__.py"

PYTHONPATH="${TMP_DIR}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${TRAIN_PYTHON}" -c \
    'from lm_eval import evaluator; from tasks.eval_harness.paper_tasks import PAPER_TASK_REGISTRY, TASK_NAMES; assert set(PAPER_TASK_REGISTRY) == set(TASK_NAMES)'
printf '%s\n' "${STAMP}" > "${TMP_DIR}/.stamp"

rm -rf "${DEPS_DIR}"
mkdir -p "$(dirname "${DEPS_DIR}")"
mv "${TMP_DIR}" "${DEPS_DIR}"
trap - EXIT

echo "paper eval dependencies ready: ${DEPS_DIR}"
