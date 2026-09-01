#!/usr/bin/env bash

# Shared CUDA build settings for A100 (SM80), H100 (SM90), and B200 (SM100).
# This file is meant to be sourced by launchers.

configure_cuda_arch_env() {
    if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
        echo "usage: configure_cuda_arch_env REPO_ROOT [PYTHON]" >&2
        return 2
    fi

    local repo_root=$1
    local python_bin=${2:-"${repo_root}/.venv/bin/python"}
    local arch_words raw normalized megatron_arches="" arch_tag=""
    local torch_tag cuda_tag

    if [ ! -x "${python_bin}" ]; then
        echo "CUDA architecture setup: Python not found: ${python_bin}" >&2
        return 2
    fi

    # Produce real cubins for all three deployed GPU generations.  Do not use
    # +PTX here: the goal is to fail during the controlled build, not later on
    # a training node with a missing kernel image.
    export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;9.0;10.0"}
    arch_words=$(printf '%s' "${TORCH_CUDA_ARCH_LIST}" | tr ';,' '  ')
    for raw in ${arch_words}; do
        normalized=${raw%%+*}
        normalized=${normalized#sm_}
        normalized=${normalized#compute_}
        normalized=${normalized//./}
        if [[ ! "${normalized}" =~ ^[0-9]+$ ]]; then
            echo "CUDA architecture setup: invalid architecture ${raw@Q}" >&2
            return 2
        fi
        megatron_arches+="${megatron_arches:+;}sm_${normalized}"
        arch_tag+="${arch_tag:+-}sm${normalized}"
    done
    if [ -z "${arch_tag}" ]; then
        echo "CUDA architecture setup: TORCH_CUDA_ARCH_LIST is empty" >&2
        return 2
    fi

    export MEGATRON_FUSED_KERNEL_ARCH_LIST=${MEGATRON_FUSED_KERNEL_ARCH_LIST:-${megatron_arches}}
    if [ -z "${TORCH_EXTENSIONS_DIR:-}" ]; then
        read -r torch_tag cuda_tag < <("${python_bin}" - <<'PY'
import torch

version = torch.__version__.split("+", 1)[0].split(".")
if len(version) < 2 or torch.version.cuda is None:
    raise SystemExit("a CUDA-enabled PyTorch installation is required")
print(".".join(version[:2]), torch.version.cuda.replace(".", ""))
PY
        )
        export TORCH_EXTENSIONS_DIR="${repo_root}/.torch_extensions/torch${torch_tag}-cu${cuda_tag}-${arch_tag}"
    fi

    echo "[cuda-arch] torch=${TORCH_CUDA_ARCH_LIST} megatron=${MEGATRON_FUSED_KERNEL_ARCH_LIST} cache=${TORCH_EXTENSIONS_DIR}"
}
