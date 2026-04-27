#!/bin/bash
# Exp3 multi-run orchestrator: 4 sequential runs in scaled-OLMoE config.
#
# Model (per CLAUDE.md § 실험 3):
#   16 layer × 48 expert/layer × topk 6 = scaled-OLMoE-1B (1.38B total)
#   hidden 1024, FFN 512 (SwiGLU), seq 1024, GBS 72 (micro 4 × accum 18)
#   RMSNorm + RoPE + untied embeddings + no biases
#
# Runs (in order):
#   1. exp3_run1          — Proposed: K_top=1%, D=1, F=0.5% (~4 experts/event)
#   2. exp3_run2          — Proposed: K_top=4%, D=1, F=1%  (~8 experts/event)
#   3. exp3_run3_moc_d4   — MoC PEC: D=4, layer-wise round-robin (48 experts/event)
#   4. baseline           — No CP, no fault (perplexity reference)
#
# Per-run logs: ${OUTPUT_BASEPATH}/log/${VARIANT}_seed${SEED}.log
# Diagnostics (Run 1, 2 only):
#   *.score_diag.jsonl   — expert score & |H| percentile distribution
#   *.vhat_vs_h.jsonl    — Pearson/Jaccard between v̂_t and |H_diag|
#
# Disk cleanup after each CP-using run (Runs 1, 2, 3) — frees CP files but
# preserves any bind-mount subdirectories under ${CP_DIR_SHARED}.
#
# Wall-clock estimate (per CLAUDE.md pilot data):
#   Run 1, 2 (with Hutchinson refresh_every=40 K=4): ~19 hr each
#   Run 3 (MoC): ~18 hr
#   Run 4 (baseline): ~18 hr
#   Total: ~74 hr (~3 days)

set -e
DIR=$(cd "$(dirname "$0")" && pwd)

# Pinned + expandable allocator combats fragmentation OOM during Hutchinson
# 2nd-order autograd. Required per pilot debugging.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Hutchinson defaults (CLAUDE.md § 실험 3 Hutchinson 설정):
#   K=4 samples, refresh_every=40, EMA α=0.9, mode=ema, persist=off.
#   CPU offload default ON (HUTCHINSON_NO_CPU_OFFLOAD not set).

OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-${PWD}/results/exp_3}
TRAIN_ITERS=${TRAIN_ITERS:-6700}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-300}
FAULT_INTERVAL=${FAULT_INTERVAL:-600}
FAULT_WARMUP=${FAULT_WARMUP:-0}

# Shared CP dir reused across CP-enabled runs (cleaned between runs).
CP_DIR_SHARED=${CP_DIR_SHARED:-${OUTPUT_BASEPATH}/cp/exp3_shared}

mkdir -p "${OUTPUT_BASEPATH}/log"

# RUN_START: skip runs < this index. Use when resuming from a partial multi-run
# (e.g. after early-terminating Run 1). Default 1 = run all. Valid: 1, 2, 3, 4.
RUN_START=${RUN_START:-1}

# Safety: refuse to overwrite logs from a run we're about to execute. The log
# dir is the canary — if a log for any run ≥ RUN_START already exists, abort.
# Runs prior to RUN_START are skipped, so their logs are preserved untouched.
_RUN_NAMES=(exp3_run1 exp3_run2 exp3_run3_moc_d4 baseline)
_conflicts=""
for ((i=RUN_START-1; i<${#_RUN_NAMES[@]}; i++)); do
    _candidate="${OUTPUT_BASEPATH}/log/${_RUN_NAMES[$i]}_seed2024.log"
    if [ -f "${_candidate}" ]; then
        _conflicts+="${_candidate}\n"
    fi
done
if [ -z "${EXP3_FORCE_OVERWRITE:-}" ] && [ -n "${_conflicts}" ]; then
    echo "[exp3] ABORT: log files for runs ≥ RUN_START=${RUN_START} already exist:" >&2
    echo -e "${_conflicts}" >&2
    echo "       Move/backup or pass EXP3_FORCE_OVERWRITE=1 to proceed." >&2
    exit 1
fi

# scaled-OLMoE config (per CLAUDE.md § 실험 3 모델 사양).
COMMON_ENV=(
    "OUTPUT_BASEPATH=${OUTPUT_BASEPATH}"
    "TRAIN_ITERS=${TRAIN_ITERS}"
    "LR_WARMUP_ITERS=${LR_WARMUP_ITERS}"
    "FAULT_INTERVAL=${FAULT_INTERVAL}"
    "FAULT_WARMUP=${FAULT_WARMUP}"
    "SEQ_LEN=1024"
    "NUM_LAYERS=16"
    "HIDDEN_SIZE=1024"
    "NUM_ATTN_HEADS=16"
    "PRMOE_SPEC=48"
    "EXPERT_INTERVAL=1"
    "GLOBAL_BATCH_SIZE=72"
    "BATCH_SIZE=4"
    "MOE_TOPK=6"
    "MOE_TRAIN_CAP=1.25"
    "MOE_EVAL_CAP=1.25"
    "EXTRA_MEGATRON_FLAGS=--swiglu --ffn-hidden-size 512 --normalization rmsnorm --use-rotary-position-embeddings --untie-embeddings-and-output-weights --disable-bias-linear"
)

run_variant() {
    local variant="$1"
    local extra_cp_dir="$2"   # empty string when no CP (baseline)

    echo ""
    echo "=============================================================="
    echo "  [exp3] Starting VARIANT=${variant}  $(date)"
    echo "=============================================================="

    env "${COMMON_ENV[@]}" \
        VARIANT="${variant}" \
        ${extra_cp_dir:+CP_DIR_OVERRIDE="${extra_cp_dir}"} \
        bash "${DIR}/proposal_etc_template.sh"

    echo "=============================================================="
    echo "  [exp3] Finished VARIANT=${variant}  $(date)"
    echo "=============================================================="

    # Cleanup CP files (preserves any mount-point sub-dirs under extra_cp_dir).
    if [ -n "${extra_cp_dir}" ] && [ -d "${extra_cp_dir}" ]; then
        echo "  [exp3] Cleaning CP files at ${extra_cp_dir}"
        find "${extra_cp_dir}" -mindepth 1 -type f -delete 2>/dev/null || true
        du -sh "${extra_cp_dir}" 2>/dev/null | awk '{print "  [exp3] Post-cleanup size:", $1}'
    fi
}

# ──────────────────────────────────────────────────────────────────────
# Run 1: Proposed system, K_top=1%, D=1, F=0.5% (~4 experts per fire)
# ──────────────────────────────────────────────────────────────────────
if [ "${RUN_START}" -le 1 ]; then
    run_variant "exp3_run1" "${CP_DIR_SHARED}"
fi

# ──────────────────────────────────────────────────────────────────────
# Run 2: Proposed system, K_top=4%, D=1, F=1% (~8 experts per fire)
# ──────────────────────────────────────────────────────────────────────
if [ "${RUN_START}" -le 2 ]; then
    run_variant "exp3_run2" "${CP_DIR_SHARED}"
fi

# ──────────────────────────────────────────────────────────────────────
# Run 3: MoC-System PEC, D=4 (every 4 iter, full layer = 48 experts FCed)
# ──────────────────────────────────────────────────────────────────────
if [ "${RUN_START}" -le 3 ]; then
    run_variant "exp3_run3_moc_d4" "${CP_DIR_SHARED}"
fi

# ──────────────────────────────────────────────────────────────────────
# Run 4: Baseline — no CP, no fault. CP_DIR_OVERRIDE irrelevant since
# proposal disabled. Pass empty extra_cp_dir → no cleanup needed.
# ──────────────────────────────────────────────────────────────────────
if [ "${RUN_START}" -le 4 ]; then
    run_variant "baseline" ""
fi

echo ""
echo "=============================================================="
echo "  [exp3] All 4 runs complete  $(date)"
echo "  Logs at: ${OUTPUT_BASEPATH}/log/"
echo "=============================================================="
