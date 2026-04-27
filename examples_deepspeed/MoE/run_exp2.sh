#!/bin/bash
# Exp2 multi-run orchestrator: 4 sequential long runs in the exp2 config.
#
# Model (exp2 spec, per CLAUDE.md):
#   24 layers × 8 experts = 192 total experts, topk=1, micro=4, global=48
#
# Runs (in order):
#   1. proposal_exp2_k2   — K_top=2%, D=1, F=2 experts/event (~1%)
#   2. proposal_exp2_k4   — K_top=4%, D=1, F=2 experts/event (~1%)
#   3. moc_m2             — MoC-System baseline, D=2
#   4. baseline           — no CP, no fault injection
#
# CP is written to CP_DIR_OVERRIDE (shared across runs because user has
# device mounts at that specific subdir). The template's cleanup
# (`find -type f -delete`) wipes CP files between runs but preserves the
# mount-point directories.
set -e
DIR=$(cd "$(dirname "$0")" && pwd)

# Use PyTorch's expandable-segments allocator to combat fragmentation —
# the 2nd-order Hutchinson Hv path repeatedly allocates/frees mid-sized
# tensors, which in the default allocator can leave the reserved pool
# fragmented enough to OOM on a 160 MB allocation even with GBs "free".
# expandable_segments:True backs the allocator with virtual memory +
# on-demand physical mapping, effectively zero fragmentation cost.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-${PWD}/results/exp_2}
TRAIN_ITERS=${TRAIN_ITERS:-8000}
FAULT_INTERVAL=${FAULT_INTERVAL:-600}
# No warmup — first fault at iter 600 (= FAULT_INTERVAL).
FAULT_WARMUP=${FAULT_WARMUP:-0}

# Shared CP location (bind mounts live here)
CP_DIR_SHARED=${CP_DIR_SHARED:-${OUTPUT_BASEPATH}/cp/proposal_ema_hutch_seed2024}

# exp2 model dims (per CLAUDE.md § 실험 2)
COMMON_ENV=(
    "OUTPUT_BASEPATH=${OUTPUT_BASEPATH}"
    "TRAIN_ITERS=${TRAIN_ITERS}"
    "FAULT_INTERVAL=${FAULT_INTERVAL}"
    "FAULT_WARMUP=${FAULT_WARMUP}"
    "NUM_LAYERS=24"
    "EXPERT_INTERVAL=1"
    'PRMOE_SPEC=8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8'
    "GLOBAL_BATCH_SIZE=48"
    "BATCH_SIZE=4"
    "MOE_TOPK=1"
)

run_variant() {
    local variant="$1"
    local extra_cp_dir="$2"
    echo ""
    echo "=============================================================="
    echo "  [run_exp2] Starting VARIANT=${variant}  $(date)"
    echo "=============================================================="
    env "${COMMON_ENV[@]}" \
        VARIANT="${variant}" \
        ${extra_cp_dir:+CP_DIR_OVERRIDE="${extra_cp_dir}"} \
        bash "${DIR}/proposal_etc_template.sh"
    echo "=============================================================="
    echo "  [run_exp2] Finished VARIANT=${variant}  $(date)"
    echo "=============================================================="
    # Cleanup CP files immediately after run so disk is freed before the
    # next experiment starts. Preserves mount-point directories via -type f.
    if [ -n "${extra_cp_dir}" ] && [ -d "${extra_cp_dir}" ]; then
        echo "  [run_exp2] Cleaning CP files at ${extra_cp_dir}"
        find "${extra_cp_dir}" -mindepth 1 -type f -delete 2>/dev/null || true
        du -sh "${extra_cp_dir}" 2>/dev/null | awk '{print "  [run_exp2] Post-cleanup size:", $1}'
    fi
}

# 1. Proposal K=2%, D=1, F=1% (2 experts/event of 192)
run_variant "proposal_exp2_k2" "${CP_DIR_SHARED}"

# 2. Proposal K=4%, D=1, F=1%
run_variant "proposal_exp2_k4" "${CP_DIR_SHARED}"

# 3. MoC-System baseline D=2
run_variant "moc_m2" "${CP_DIR_SHARED}"

# 4. Baseline — no CP, no faults. CP_DIR_OVERRIDE irrelevant (proposal off).
run_variant "baseline" ""

echo ""
echo "=============================================================="
echo "  [run_exp2] All 4 runs complete  $(date)"
echo "=============================================================="
