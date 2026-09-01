#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${REPO_ROOT}/.venv/bin/python"}
PIP_BIN=${PIP_BIN:-"${REPO_ROOT}/.venv/bin/pip"}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
APEX_REF=${APEX_REF:-becbb77cea4cb54f2929f7c938a0a6f7dd1fdc39}
APEX_CUDA_ARCH_LIST=${APEX_CUDA_ARCH_LIST:-"8.0;9.0;10.0"}
APEX_WHEEL_ROOT=${APEX_WHEEL_ROOT:-"${REPO_ROOT}/.cuda_wheels/apex"}
APEX_SOURCE_DIR=${APEX_SOURCE_DIR:-}
INSTALL=0

usage() {
    cat <<'EOF'
usage: tools/build_apex_fat_wheel.sh [--install]

Build an NVIDIA Apex wheel containing native SM80, SM90, and SM100 cubins.
The wheel is verified with cuobjdump before --install is allowed.

Environment overrides:
  PYTHON_BIN, PIP_BIN, CUDA_HOME, APEX_REF, APEX_SOURCE_DIR,
  APEX_CUDA_ARCH_LIST, APEX_WHEEL_ROOT, APEX_PARALLEL_BUILD, MAX_JOBS
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --install) INSTALL=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

for executable in "${PYTHON_BIN}" "${PIP_BIN}" "${CUDA_HOME}/bin/nvcc" "${CUDA_HOME}/bin/cuobjdump"; do
    if [ ! -x "${executable}" ]; then
        echo "required executable not found: ${executable}" >&2
        exit 2
    fi
done

for sm in 80 90 100; do
    if ! "${CUDA_HOME}/bin/nvcc" --list-gpu-code | grep -qx "sm_${sm}"; then
        echo "nvcc under ${CUDA_HOME} cannot emit sm_${sm}" >&2
        exit 2
    fi
done

BUILD_TMP=$(mktemp -d /tmp/apex-fat-build.XXXXXX)
cleanup() {
    if [[ "${BUILD_TMP}" == /tmp/apex-fat-build.* ]] && [ -d "${BUILD_TMP}" ]; then
        rm -rf -- "${BUILD_TMP}"
    fi
}
trap cleanup EXIT

if [ -n "${APEX_SOURCE_DIR}" ]; then
    APEX_SOURCE_DIR=$(cd "${APEX_SOURCE_DIR}" && pwd)
else
    APEX_SOURCE_DIR="${BUILD_TMP}/apex"
    git clone --filter=blob:none https://github.com/NVIDIA/apex.git "${APEX_SOURCE_DIR}"
fi
git -C "${APEX_SOURCE_DIR}" checkout --detach "${APEX_REF}"
APEX_COMMIT=$(git -C "${APEX_SOURCE_DIR}" rev-parse HEAD)
WHEEL_DIR="${APEX_WHEEL_ROOT}/${APEX_COMMIT}"
mkdir -p "${WHEEL_DIR}"

export CUDA_HOME
export TORCH_CUDA_ARCH_LIST="${APEX_CUDA_ARCH_LIST}"
export APEX_CPP_EXT=1
export APEX_CUDA_EXT=1
export APEX_PARALLEL_BUILD=${APEX_PARALLEL_BUILD:-8}
export MAX_JOBS=${MAX_JOBS:-8}
export NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-"--threads 4"}

echo "[apex-fat] commit=${APEX_COMMIT}"
echo "[apex-fat] torch_arches=${TORCH_CUDA_ARCH_LIST} cuda_home=${CUDA_HOME}"
"${PIP_BIN}" wheel -v --disable-pip-version-check --no-build-isolation \
    --no-deps --no-cache-dir --wheel-dir "${WHEEL_DIR}" "${APEX_SOURCE_DIR}"

WHEEL=$(find "${WHEEL_DIR}" -maxdepth 1 -type f -name 'apex-*.whl' -print -quit)
if [ -z "${WHEEL}" ]; then
    echo "Apex wheel was not produced under ${WHEEL_DIR}" >&2
    exit 1
fi

VERIFY_DIR="${BUILD_TMP}/verify"
mkdir -p "${VERIFY_DIR}"
"${PYTHON_BIN}" -m zipfile -e "${WHEEL}" "${VERIFY_DIR}"

verify_apex_tree() {
    local root=$1 so listing sm
    local cuda_binary_count=0
    local amp_found=0
    while IFS= read -r -d '' so; do
        listing=$("${CUDA_HOME}/bin/cuobjdump" --list-elf "${so}" 2>/dev/null || true)
        if ! grep -q 'sm_[0-9]' <<<"${listing}"; then
            continue
        fi
        cuda_binary_count=$((cuda_binary_count + 1))
        if [[ "$(basename "${so}")" == amp_C.*.so ]]; then
            amp_found=1
        fi
        for sm in 80 90 100; do
            if ! grep -q "sm_${sm}" <<<"${listing}"; then
                echo "missing sm_${sm}: ${so}" >&2
                return 1
            fi
        done
    done < <(find "${root}" -type f -name '*.so' -print0)
    if [ "${cuda_binary_count}" -eq 0 ] || [ "${amp_found}" -ne 1 ]; then
        echo "verified tree has no CUDA Apex binaries or no amp_C extension: ${root}" >&2
        return 1
    fi
    echo "[apex-fat] verified ${cuda_binary_count} CUDA extensions: sm80, sm90, sm100"
}

verify_apex_tree "${VERIFY_DIR}"
printf 'apex_commit=%s\ntorch_cuda_arch_list=%s\n' \
    "${APEX_COMMIT}" "${TORCH_CUDA_ARCH_LIST}" >"${WHEEL}.build-info"

if [ "${INSTALL}" -eq 1 ]; then
    SITE_PACKAGES=$("${PYTHON_BIN}" -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    RECORD="${SITE_PACKAGES}/apex-0.1.dist-info/RECORD"
    BACKUP_DIR="${HOME}/.cache/apex-backups/apex-before-fat-${APEX_COMMIT:0:7}-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "${BACKUP_DIR}"
    if [ -f "${RECORD}" ]; then
        while IFS=, read -r relative_path _; do
            if [ -n "${relative_path}" ] && [ -e "${SITE_PACKAGES}/${relative_path}" ]; then
                mkdir -p "${BACKUP_DIR}/$(dirname "${relative_path}")"
                cp -a "${SITE_PACKAGES}/${relative_path}" "${BACKUP_DIR}/${relative_path}"
            fi
        done <"${RECORD}"
        echo "[apex-fat] previous install backed up to ${BACKUP_DIR}"
    fi
    "${PIP_BIN}" install --disable-pip-version-check --force-reinstall --no-deps "${WHEEL}"
    verify_apex_tree "${SITE_PACKAGES}"
fi

echo "[apex-fat] wheel=${WHEEL}"
