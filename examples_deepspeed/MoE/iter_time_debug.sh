#!/bin/bash
# iter_time_debug.sh — proposal iter time 오버헤드 격리 측정.
#
# ============================== 명세 ==============================
#
# 목적:
#   baseline (no proposal) vs proposal-on 의 iter time 차이를 알고리즘 + stream
#   오버헤드만으로 좁히고, gap 의 발원지를 nsys / py-spy / iter time histogram
#   세 도구로 동시 추적. ARCHITECTURE_V2.md §0.4 검증 게이트 (iter time gap ≤ 10 ms)
#   를 직접 측정.
#
# 기준: verify_vram_run.sh (125M MoE-48E test, TRAIN_ITERS=2000, --exit-interval 20).
#   모델 / 학습 설정 그대로 inheriting, iter time 측정 목적의 차이만 반영.
#
# verify_vram_run.sh 대비 변경점:
#   변경:
#     HESSIAN_REFRESH_EVERY  5 → 10000   anchor 가 iter 0/1 cold start 외 발화 0
#     --log-interval         5 → 10      per-iter stdout sync 회피
#     선택                   ITER_TIME_WITH_NSYS=1 일 때만 NSYS_PROFILE=1
#   제거:
#     Hessian refresh uses K=1; robustness comes from anchor multi-batch.
#   제거 (iter time 측정과 무관, 오버헤드 회피):
#     PROPOSAL_VRAM_AUDIT=1, PROPOSAL_VRAM_AUDIT_LOG
#     PROPOSAL_PER_ITER_ALLOC_TRACE=1
#   추가:
#     baseline run (VARIANT=baseline, PROPOSAL_FLAGS="" 분기)
#     compare 단계 (10-iter window mean delta → §0.4 PASS 게이트)
#
# 측정 창 (TRAIN_ITERS=2000, --exit-interval=20, --log-interval=10):
#   log sample 0 = iter 1~10 window, cold start / first FC / warmup 포함 → drop
#   log sample 1 = iter 11~20 window, stable-ish short window 비교
#
# 산출물:
#   $OUT/baseline/{run.log, iter_times.txt}
#   $OUT/proposal/{run.log, iter_times.txt, py_spy.txt}
#   ITER_TIME_WITH_NSYS=1 일 때만 nsys.qdrep 추가 생성
#   $OUT/compare.txt   ── mean / p50 / p95 / max delta. PASS = mean ≤ 10 ms.
#
# 후속 분석 recipe (script 가 자동 실행하지 않음):
#   nsys-ui $OUT/proposal/nsys.qdrep                # main_stream timeline gap 위치
#   nsys stats $OUT/proposal/nsys.qdrep \
#       --report cuda_kern_exec_sum                 # kernel 별 누적 시간
#   nsys stats $OUT/proposal/nsys.qdrep \
#       --report cuda_api_sum                       # cudaMalloc / Sync 호출 빈도
#   grep -A 30 "Thread.*main\|Thread.*expert\|Thread.*ne" $OUT/proposal/py_spy.txt
#   diff $OUT/baseline/iter_times.txt $OUT/proposal/iter_times.txt
#
# delta > 10 ms 발견 시 의심 순서 (§0):
#   1) nsys main_stream timeline gap → 어떤 host-side sync 가 proposal 을 기다리나
#   2) py-spy stack → cudaStreamSynchronize / cudaHostAlloc / mutex_lock caller
#   3) §0.1 R2 (per-iter alloc) / §0.3 pinned alloc / pageable H2D 위반 확인
#
# ==================================================================

set -euo pipefail
# Repo-relative paths (portable across servers): this script lives in
# <repo>/examples_deepspeed/MoE, so repo root = two dirs up.
DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$DIR/../.." && pwd)
OUT="$REPO_ROOT/results/exp_ITER_TIME_DEBUG"
rm -rf "$OUT"; mkdir -p "$OUT/baseline" "$OUT/proposal"

# 기본 timing gate는 nsys를 끈다. nsys wrapper/export가 short run 종료를
# 오래 붙잡을 수 있어, timeline 분석이 필요할 때만 ITER_TIME_WITH_NSYS=1.
PROFILE_ENV=()
if [ "${ITER_TIME_WITH_NSYS:-0}" = "1" ]; then
    PROFILE_ENV=(NSYS_PROFILE=1 NSYS_TRACE=cuda,osrt,nvtx)
fi

COMMON=(
    NUM_LAYERS=12 HIDDEN_SIZE=768 NUM_ATTN_HEADS=12
    GLOBAL_BATCH_SIZE=8 BATCH_SIZE=2 SEQ_LEN=2048
    MOE_TOPK=1 NUM_EXPERTS=48 MOE_TRAIN_CAP=1.25 MOE_EVAL_CAP=1.25
    LR_WARMUP_ITERS=50 TRAIN_ITERS=2000
    FAULT_INTERVAL=0
    BLOCK_QUANT_G=8 NONEXPERT_FC_W=1 CP_TARGET=${CP_TARGET:-disk}
    EXTRA_MEGATRON_FLAGS="--exit-interval 20 --eval-interval 999999 --eval-iters 0 --log-interval 10 ${ITER_TIME_EXTRA_MEGATRON_FLAGS:-}"
    SEED=2024
)

# 원인 분해용 테스트 플래그는 EXTRA_PROPOSAL_FLAGS 로 주입한다.
#   --proposal-test-skip-expert-commit:
#       expert selection/AOS/D2H 는 그대로 실행하고 CPU scatter/file/LFCL commit 만 생략.
#   --proposal-test-skip-expert-d2h --proposal-test-skip-nonexpert-d2h:
#       D2H 제외 후 proposal compute/AOS producer contention 을 본다.
#   --proposal-test-expert-selection-only:
#       expert selection/cutoff 와 host layout 만 남기고 GPU record emission
#       이후를 생략한다. AOS/scan 전체 skip 이 아니다.

# baseline 은 기준선(no-proposal iter time)이 안정적이라 매번 돌릴 필요 없음.
# 모델/HW/스케줄이 바뀐 경우 등 필요할 때만 측정하면 되고, 불필요하면 이 블록을
# 통째로 주석 처리해도 아래 proposal 측정에는 영향이 없다 (비교만 빠짐).
# ============== baseline (no proposal) ==============
echo "[1/2] baseline (no proposal) — exit @ iter 20"
env "${COMMON[@]}" \
    "${PROFILE_ENV[@]}" \
    EXTRA_PROPOSAL_FLAGS="" \
    NSYS_OUT="$OUT/baseline/nsys" \
    OUTPUT_BASEPATH="$OUT/baseline" \
    VARIANT=baseline \
    bash "$DIR/proposal_test_template.sh" > "$OUT/baseline/run.log" 2>&1 || true

grep "elapsed time per iteration" "$OUT/baseline/run.log" \
    | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9.]+).*/\1/' \
    > "$OUT/baseline/iter_times.txt"

# ============== proposal on (cold-start anchor only, CP_TARGET=disk by default) ==============
echo "[2/2] proposal on (anchor cold-start only, CP_TARGET=${CP_TARGET:-disk}) — py-spy best-effort"
env "${COMMON[@]}" \
    "${PROFILE_ENV[@]}" \
    EXTRA_PROPOSAL_FLAGS="${EXTRA_PROPOSAL_FLAGS:-}" \
    HESSIAN_REFRESH_EVERY=10000 \
    PROPOSAL_ROUTING_FREEZE_ENABLE=1 \
    NSYS_OUT="$OUT/proposal/nsys" \
    OUTPUT_BASEPATH="$OUT/proposal" \
    VARIANT=iter_time_debug \
    bash "$DIR/proposal_test_template.sh" > "$OUT/proposal/run.log" 2>&1 &
PROPOSAL_PID=$!

# py-spy 1 회 best-effort. 20 iter × ~200ms ≈ 4 s 총 길이 라 sleep 1 s 만.
# warmup phase 잡힐 가능성이 큼 — main 분석 도구는 nsys.qdrep.
sleep 1
if ps -p $PROPOSAL_PID > /dev/null; then
    DS_PID=$(pgrep -P "$PROPOSAL_PID" -f pretrain_gpt | head -1 || true)
    [ -z "$DS_PID" ] && DS_PID=$(pgrep -f pretrain_gpt | head -1 || true)
    if [ -n "$DS_PID" ]; then
        py-spy dump --pid "$DS_PID" > "$OUT/proposal/py_spy.txt" 2>&1 || true
        echo "  py-spy dump → $OUT/proposal/py_spy.txt (pid $DS_PID)"
    fi
fi
wait $PROPOSAL_PID || true

grep "elapsed time per iteration" "$OUT/proposal/run.log" \
    | sed -E 's/.*elapsed time per iteration \(ms\): ([0-9.]+).*/\1/' \
    > "$OUT/proposal/iter_times.txt"

# ============== 비교 ==============
echo ""
echo "============== compare =============="
set +e
OUT="$OUT" python3 - <<'PY' | tee "$OUT/compare.txt"
import os, statistics, pathlib
out = pathlib.Path(os.environ["OUT"])
base = [float(x) for x in (out/"baseline/iter_times.txt").read_text().split() if x]
prop = [float(x) for x in (out/"proposal/iter_times.txt").read_text().split() if x]
DROP_SAMPLES = 1 if min(len(base), len(prop)) > 1 else 0
base, prop = base[DROP_SAMPLES:], prop[DROP_SAMPLES:]
n = min(len(base), len(prop))
base, prop = base[:n], prop[:n]
if n == 0:
    print("no overlapping log windows after warmup drop"); raise SystemExit(1)
delta = [p - b for b, p in zip(base, prop)]
def pct(q): return sorted(delta)[int(q*(len(delta)-1))]
print(f"n_iter (stable phase): {n}")
print(f"baseline mean / p50:   {statistics.mean(base):7.2f} / {statistics.median(base):7.2f} ms")
print(f"proposal mean / p50:   {statistics.mean(prop):7.2f} / {statistics.median(prop):7.2f} ms")
print(f"delta mean:            {statistics.mean(delta):7.2f} ms   (§0.4 gate: ≤ 10)")
print(f"delta p50 / p95 / max: {pct(0.5):7.2f} / {pct(0.95):7.2f} / {max(delta):7.2f} ms")
gate = statistics.mean(delta) <= 10.0
print(f"\nGATE: {'PASS' if gate else 'FAIL — investigate via nsys / py-spy (header recipe)'}")
raise SystemExit(0 if gate else 1)
PY
EVAL_RC=${PIPESTATUS[0]}
set -e

echo ""
echo "outputs: $OUT/{baseline,proposal,compare.txt}"
exit $EVAL_RC
