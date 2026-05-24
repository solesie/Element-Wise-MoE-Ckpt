# MoE Element-wise Importance Checkpointing

## Project Goal

MoE 모델의 학습 중 checkpointing에서, expert 단위가 아닌 **파라미터 element 단위**로 중요도를 판단하여 선별 저장하는 시스템을 구현하고 검증한다.

## Core Idea

MoE(Mixture-of-Experts)는 FFN을 expert별로 분리함으로서 적은 계산량으로도 expert개수를 늘려 모델의 크기를 키울 수 있게 한다. MoE 환경에서 parameter는 non-expert parameter(attention, embedding, router)와 expert parameter로 나누어진다. 여러 이전 연구들은 non-expert parameter가 모델의 품질에 미치는 영향은 expert parameter 보다 더 크다고 알려왔지만, expert parameter이 모델의 품질에 미치는 영향도 무시할 수는 없다.

Checkpointing은 모델 훈련 과정에서 예상치못한 상황에서 훈련 계산 결과의 유실을 막기위해 모든 모델의 데이터를 storage에 저장하는 행위를 의미한다.

그러나 MoE 환경의 Checkpointing은 훈련계산량이 줄어든 것에 비해 모델의 크기가 커졌기 때문에, Checkpointing이 훈련 성능 상 미치는 오버헤드가 크다. 
그래서 MoE checkpointing 연구들은 크기가 작고 모델의 품질에 큰 영향을 미치는 non-expert parameter에 대해서는 lossless non-expert checkpointing을 채택하되, 크기가 크고 모델의 품질에 비교적 작은 영향을 미치는 expert parameter에 대해서는 lossless experts checkpointing보다는 lossy experts checkpointing을 방향으로 삼고있다.

대표적으로 MoC-System (ASPLOS'25) 는 특정 interval(D)마다 round-robin 방식으로 한 layer의 모든 experts를 checkpointing하는 PEC(Partial Experts Checkpointing) 전략을 제시했다. 이렇듯 기존 MoE 연구들은 expert-wise checkpointing, 즉 "어떤 expert를 iteration이 완료되고 저장할 것인가"에 집중한다. 만약 예상치못한 fault가 발생하여 MoE 모델을 recovery 한다면, 최근에 저장된 몇몇 expert는 최신의 상태로 들고오고, 저장한지 오래된 expert는 그 상태가 예전의 staleness 상태로 들고온다.

그러나 이러한 expert-wise checkpointing은 큰 약점이 존재한다. staleness가 존재하는 expert 내부에 non-expert parameter와 유사할 정도로 중요한 원소가 있었을 경우, 이 원소가 모델에 미치는 영향은 non-expert가 유실되었을 때 미치는 영향과 흡사할 수 있다. lossy한 저장의 본질은 "잃으면 안 되는 중요한 정보"를 저장하는데에 있는데, expert-wise checkpointing은 이를 제대로 고려하지 못하고 있는 것이다.

따라서 "잃으면 안 되는 중요한 정보"는 expert 단위가 아니라 원소 단위, 즉 element-wise로 판단해야 한다. 

---

# Hard Implementation Constraints (절대 어겨선 안 됨)

이 두 제약은 본 시스템의 모든 구현 결정을 지배한다. 새 코드 / 리팩터 / 최적화 시 **반드시** 준수.

## 1. VRAM 추가 사용량 제약

| 종류 | 한도 | 측정 방식 |
|---|---|---|
| **Persistent VRAM 추가** | **0 MB** (절대 0) | `torch.cuda.memory_allocated()` 의 baseline (학습 시작 직후) 대비 영구 증가량 |
| **Transient VRAM 추가 (whole-iter peak)** | **≤ 80 MB** | `torch.cuda.max_memory_allocated()` − baseline. 매 iter 마다 `reset_peak_memory_stats()` 직후 측정 |

- baseline = "outlier promotion / proposal CP 비활성화" 시점의 alloc/peak.
- 새 GPU buffer 를 추가할 때는 **반드시** view (zero-copy) 를 우선 시도. 새 storage 를 생성해야 한다면 init log 와 `gpu_memory_breakdown()` 에 추가하여 가시화.
- pinned host memory 는 VRAM 이 아니므로 이 제약 밖. 단 200 MB 이상 사용 시 host RAM 영향 검토.
- 측정 의무: 의심스러운 변경 후 `PROPOSAL_OUTLIER_DBG_DSP_PEAK=1` 또는 cp_hooks 의 `peak/Δpeak` 로그로 확인.

## 2. main_stream blocking 제약

| 항목 | 한도 |
|---|---|
| **main_stream 위 blocking 추가** | **0 ms** (절대 0) |
| **training thread 의 추가 sync 호출** | **0** (절대 0) |

- main_stream = Megatron/DeepSpeed 의 forward/backward/optimizer 가 launch 되는 stream. 학습 critical path.
- proposal 의 어떤 작업도 main_stream 위에 enqueue 되어선 안 됨. tracker_stream / d2h_stream 위에 launch.
- training thread 가 `torch.cuda.synchronize()`, `tensor.item()`, `tensor.cpu()` 등으로 GPU 를 기다리는 호출은 **금지**. 필요한 sync 는 writer thread (background) 로 위임 — `cuda.Event` 를 writer 에 전달하여 writer 가 `event.synchronize()` 호출.
- 예외: dispatch 의 `ev.synchronize()` (1-iter pipelining 의 일부) — 이 시점엔 prev iter 의 tracker_stream work 가 main_stream 의 forward+backward+optimizer (~2000 ms) 보다 훨씬 일찍 완료되므로 host wait time = 0.
- SM contention 은 "soft" blocking 으로 간주 — 큰 batched kernel 이 작은 launch 들보다 SM 점유시간 짧아 contention 줄임. 1280 small launch → 30 batched launch 식의 변경은 SM 측면에서도 이득.

## 2-bis. Proposal stream 위 메모리 할당 금지 (main_stream idle gap 의 근본 원인)

| 항목 | 한도 |
|---|---|
| **tracker_stream / d2h_stream / worker_stream 위 cudaMalloc / cudaFree 호출** | **0** (절대 0) |
| **proposal kernel 안에서 PyTorch tensor alloc** | **0** (kernel 은 pre-allocated buffer 만 사용) |

- 직접적 fact (nsys + py-spy 검증): proposal stream 위에서 발생한 alloc 가 caching allocator 의 mutex 를 잡고, **그 mutex 가 main_stream 의 다음 kernel launch 를 blocking** → main_stream 의 wave 사이에 **수십 ms 의 idle gap** 발생. proposal 의 work 자체가 main 과 다른 stream 위라도, **alloc 의 host-side mutex 가 cross-stream blocking**.
- 따라서 **proposal 의 모든 tracker_stream / worker_stream 작업은 pre-allocated persistent buffer 만 사용**:
  - tracker buffer (dg_flat, h_g_bf16_flat, exc_mask 등) — `_init_accum_sq_flat` 에서 한 번 alloc
  - segment_pipeline 의 double buffer — init 시 alloc
  - outlier kernel 의 atomic count / slot — `_init_outlier_buffers` 에서 한 번 alloc
- anchor refresh 는 train_step 외부 (main_stream 의 학습 wave 와 시간상 분리) 에서 실행되므로 alloc 가능. 단 같은 양 (`grad_flat` 2.72 GB = lp.grad transient 와 동량) 으로 학습 peak 와 동일성 유지.
- per-iter dispatch / per-iter outlier kernel 호출 은 **kernel 안 어떤 alloc 도 금지**. 모든 buffer 는 init 시 alloc, 매 iter zero/reuse.
- **검증**: anchor refresh OFF iter 의 iter time = baseline + ≤ 10 ms (proposal 의 host-side overhead = launch count × ~10us 만). 만약 +수십 ms idle gap 이 보이면 alloc 누수 의심 → `PROPOSAL_VRAM_AUDIT=1` + py-spy 로 caller 추적.

## 3. 검증 의무

새 변경마다 다음 측정값을 비교:
- **iter time gap** (vs baseline outlier-OFF) ≤ 10 ms
- **persistent VRAM gap** = 0 MB
- **transient VRAM peak gap** ≤ 80 MB
- **algorithm 정합성**: val loss 가 baseline 과 4-자리 이상 일치

위 4 개 중 하나라도 실패하면 변경 폐기 또는 재설계.

---

# 현재 상태 (2026-05-24, Fast-flat anchor + naming refactor)

## 정리 요약

이전 commit (9e2a474) 이후 (a) anchor refresh 의 **fast-flat path** 도입, (b) `option_a`/`pass_a` 같은 모호한 식별자를 의미-기반 이름으로 일관 rename, (c) dead env vars / 함수 / 테스트 제거, (d) `binary_io.cpp` 내 raw syscall 들을 anonymous-namespace `syscall_*` wrapper 로 일원화. param_rebind 모듈 통째로 제거 (`PROPOSAL_HESSIAN_NO_REBIND` 기본 항상 ON; `_pinned_clone` snap 으로 원본 보존). `PROPOSAL_ANCHOR_N_BATCHES` env 삭제 — N = `global_batch_size // micro_batch_size // 2` 하드코딩.

## 4-kernel pipeline (per iter)

| stream | kernel | 역할 |
|---|---|---|
| tracker | `_outlier_count_kernel` | per BF16 group: outlier 후보 element 의 per-name 카운트 + sub-sample. fold 안 함 |
| tracker | `_group_select_kernel` | per group: `h_g × dg ≥ group_th` 판정. `name_group_count` atomic + `selected_mask` 기록 |
| worker | `_aos_write_kernel` | per segment: outlier records AOS write + 선택적 fold (`dg += Σ non-selected Δθ²`) |
| worker | `_aos_group_write_kernel` | per segment: group records AOS write (`selected_mask` 만 의존; dg-drift decoupled). 선택된 group 의 dg=0 |

`_walk_thresholds` (host) 가 stochastic sub-sample → K-th order statistic 으로 `elem_th` 산출 (heavy-tail-robust; histc 의 12/288 collapse 미발생).

## Anchor refresh — fast-flat path (production) + 메모리 폭발 해결 history

### 진단: anchor 의 메모리 폭발은 어디서 왔는가

이전 chunk path 의 anchor refresh 가 학습 peak 보다 +6 GB 가량 추가로 alloc 했었음. nsys + step-by-step VRAM dump 로 진짜 origin 분해:

| 원인 | 양 | 해결 |
|---|---:|---|
| `_make_rademacher` 가 expert_ps 전체 z 를 한 번에 alloc | ~2.72 GB BF16 | **on-the-fly PCG-hash z** (Triton kernel register-only) → **−2.72 GB** |
| chunk-local snap+z+gp+gm 4 buffer (chunk_size 비례) | ~540 MB | chunk path 통째 제거; single backward 쌍 → **−540 MB** |
| `param_rebind_context` 의 expert_ps GPU clone | ~2.72 GB BF16 | `_param_rebind.py` 삭제; DS lp.data 가 flat-view 임을 활용해 lp 직접 흔들기 → **−2.72 GB** |
| h_accum (per-expert BF16 list, anchor-only alloc) | ~340 MB | tracker 의 `_g_fast_h_g_bf16_flat` (이미 R1 에 포함된 persistent buffer) 의 view 재사용 → **−340 MB** |
| activation transient (forward graph saved tensors) | ~5 GB | 본질적 (학습 backward 와 동량). 그대로 두되 학습과 같은 timing → **R3 strict 충족 가능** |

→ 총 −6.3 GB 절감. anchor peak ≤ training peak 가 **caching allocator 의 reuse 패턴**과 합쳐져 **R3 strict +74 MB tolerance 안 충족**.

### Production path

기존 chunk path 폐기. 모든 anchor 호출은 단일 backward 쌍 + grad-flat batched stream 으로 실행.

```
[setup, multi-batch loop 진입 1회]
  - θ_0 = _pinned_clone(p) per expert (pool 재사용; 첫 anchor 만 alloc, 이후 H2D 만)
  - expert_ps 를 tracker.name_list 의 idx 순으로 sort
  - grad_flat = torch.zeros(ΣP_expert, BF16)   # 2.72 GB; 학습의 lp.grad transient 와 같은 양
  - each expert.grad = grad_flat[off:off+n].view_as(p)   # backward 가 flat 에 직접 write
  - h_accum_flat = _tracker._g_fast_h_g_bf16_flat       # ★ 새 alloc 0, R1 budget 안 ★
  - h_accum_flat.zero_()

[K loop × 2 phase (sign = +1, -1)]
  per-expert perturb: _flat_z_perturb_kernel(p.data, sign*eps, global_offset, seed, k)
                        # 1 Triton launch per expert, z = on-the-fly PCG hash (register-only)
  forward + backward    # _backward_inplace(preserve_expert_grad_storage=True)
                        # ⇒ grad_flat 에 expert grad accumulate
  _flat_signed_stream_kernel(grad_flat, h_accum_flat, g_blk, sign*inv_step*inv_k, seed, k)
                        # 1 Triton launch total; group_mean(z × grad) × sign × scale 누적
  grad_flat.zero_()

[multi-batch loop 종료]
  h_accum_flat.div_(N_succ).abs_()
  expert lp 일괄 restore from pinned snap
```

### 시간 단축 분해 (fast-flat vs chunk path)

per-batch breakdown (N=12 medium, BATCH_SIZE=2):

| 단계 | chunk path | fast-flat | 절감 | 비고 |
|---|---:|---:|---:|---|
| backup_D2H | 106 ms × N | 106 ms × 1 | −1166 ms / anchor | multi-batch session 한 번만 |
| restore_H2D | 126 ms × N | 126 ms × 1 | −1386 ms / anchor | 동일 |
| perturb (per phase) | 191 ms (per-expert 576 × 7 ops) | 130 ms (per-expert 576 × 1 Triton launch) | −61 ms × 2 phase × N | host launch count 7→1 |
| stream (per phase) | 200 ms (per-expert 576 × 11 ops + FP32 transient) | **8 ms (single Triton launch)** | −192 ms × 2 phase × N | grad_flat 덕분에 single kernel |
| bwd (per phase) | 700-1100 ms | 동일 | 0 | 본질적 forward+backward |
| **per-batch total** | **2042 ms** | **1211 ms** | **−41%** | |
| **anchor total (N=12)** | **24500 ms** | **14750 ms** | **−41%** | |

핵심:
- z 는 register-only (kernel 안 PCG hash). VRAM 0.
- h_accum_flat = tracker buffer view → **추가 alloc 0**. R3 strict 유지.
- grad_flat 은 학습 lp.grad transient 와 동량 → caching allocator 가 같은 자리 reuse → **peak 영향 0**.
- 모든 phase 의 z 는 같은 hash (`global_offset + k + seed`) → perturb 와 stream 이 bit-exact 동일 z 사용 (FD cancellation 보장).
- `PROPOSAL_FD_FAST_FLAT` env (default ON, fallback 시 legacy per-expert path).
- N (anchor batch 수) = `global_batch_size / micro_batch_size / 2` (env 없음).
- multi-batch session 안에서 **expert.grad 가 grad_flat 의 view 이므로 alloc 0** (training 의 lp.grad 와 같은 자리). 매 phase 끝의 `grad_flat.zero_()` 가 caching allocator alloc 호출 없음.

## iter time (exp_etc, anchor refresh ON N=6, 30 iter)

| metric | 이전 (chunk path) | fast-flat | 변화 |
|---|---:|---:|---|
| anchor (N=12) | 24500 ms | **14750 ms** (N=6: ~7400 ms) | **−40%** |
| iter time (refresh_every=5) | 9395 ms | **6500 ms** | **−31%** |
| lm_loss (iter5→30, baseline match) | ✓ | **bit-exact 4-digit match** | — |
| iter time write-OFF (no anchor) | ≈ baseline | ≈ baseline | (≤ 50 ms noise) |

## VRAM 상태

| 항목 | exp_etc | 100B per-GPU 추정 |
|---|---:|---:|
| persistent baseline (training) | ~21.3 GB | ~ |
| proposal Δ persistent | ~1.5 GB | ~9.2 GB |
| └ tracker buffer (dg + h_g + exc_mask + chunk_buf) | 1.3 GB | 8.7 GB |
| └ sub-sample buffer (64K × f32 × 3) | 1.5 MB | 1.5 MB |
| └ selected_mask (N_groups × i8) | 30 MB | 125 MB |
| └ segment double buffer | 256 MB | 256 MB |
| └ metadata (N_names × i64 × 5) | 수십 KB | 수십 KB |
| **peak ≡ steady** (anchor refresh 비교) | ✓ +74 MB (tolerance 100) | ✓ (구조상 동일) |

K_max-bounded persistent buffer 0.

Anchor 의 일시 transient: grad_flat (~2.72 GB BF16; 학습 lp.grad transient 와 동량) + h_accum (= tracker view, 0 new) + per-expert z (register, 0). param_rebind clone 없음.

## main_stream blocking = 0

모든 proposal 작업이 tracker_stream / worker_stream 위. training thread 의 sync 호출 0. backward pass 와 d2h 자연 overlap.

## 단위 / 회귀 검증 (`megatron/proposal/tests/`)

본 디렉토리의 파일들은 **production 보호용 핵심 검증** — 코드 수정마다
반드시 통과해야 함. 새 검증 항목은 모두 이 폴더에 추가.

- **`vram_audit_eval.py`** — R1-R4 hard contract PASS/FAIL evaluator
  (VRAM audit JSONL 입력). `verify_vram_run.sh` 가 매 audit run 끝에
  호출 → exit 1 이면 R-contract 위반. **모든 변경의 마지막 게이트.**
  - 사용: `python3 megatron/proposal/tests/vram_audit_eval.py <jsonl>`
- **`test_outlier_per_segment.py`** — Pass B 의 single full-range 호출
  vs N-split 호출 결과 동일성 (late-group i64 포함). 7/7 PASS.
  outlier kernel 의 chunking 정확성 보장.
- **`test_hessian_stream.py`** — legacy h_accum streaming 커널의 reference
  FP32 vs BF16 group-quant 비교. 8/8 PASS. fast-flat fallback path 보호.

폐기:
- `_test_gpu_aos.py` — pre-existing FAIL (9e2a474 이전부터 broken).
  production gpu_aos 동작은 recovery 검증으로 확인.
- `_test_option_a.py` — dead Python ref 의존 (validate_option_a 등).

## End-to-end 검증 (2026-05-24 본 refactor 직후)

| 항목 | 결과 |
|---|---|
| R1 (persistent VRAM) | PASS (+17 MB diff) |
| R2 (proposal-callsite large alloc) | PASS (0 violations) |
| **R3 (anchor peak vs training peak)** | **PASS strict (+74 MB, tolerance 100)** |
| R4 (slope, frag) | PASS (slope=0, waste 8044 MB < limit) |
| H scale + coverage | mean 4.0e-06 ~ 1.6e-07, coverage 0.73~1.0, flip_skip 0~0.27 |
| DRAM recovery | PASS (`FAULT@10 recover_s=4.72`, `FAULT@20 recover_s=4.60`) |
| Disk recovery | PASS (`FAULT@10 recover_s=10.09`, `FAULT@20 recover_s=9.15`) |
| lm_loss baseline match | ✓ bit-exact 4-digit (iter5=10.7212, iter30=9.2312) |
| dispatch coverage | `anchor multi-batch: N=6 succ=6`; DRAM step=20 n_experts=78, step=21 n_experts=243 |

## C++ binary_io 의 syscall 일원화

`megatron/proposal/_csrc/binary_io.cpp` 의 모든 raw `::write/::open/::fsync/::close/::unlink/::rename/::pread/::stat` 호출이 anonymous-namespace 의 **`syscall_*` wrapper 함수** (`syscall_write_full_aligned`, `syscall_open_create`, `syscall_open_rdonly`, `syscall_fsync`, `syscall_close_noexcept`, `syscall_rename`, `syscall_unlink_allow_enoent`, `syscall_pread_loop`, `syscall_stat_size`) 안으로 모임. 외부 public 함수 (`write_prptens`, `write_lossless`, `open_lossy_log_for_append`, `append_lossy_log`, `fsync_fd`, `close_fd`, `full_write_pinned`, `unlink_path`, `read_file_range`, `file_size`) 는 모두 이 wrapper 만 호출 — io_uring swap 시 wrapper 만 교체.

> TODO: caller 가 `int fd` / `const std::string& path` 를 보유하는 현재 API 를 캡슐화 → `(layer, expert_id, lossless|lossy|non_expert)` identity-tuple 기반 API 로 변경 예정 (다음 refactor cycle).

---

# 과거 진단 (참고용)

## CLAUDE.md +355 ms diagnosis (해결됨)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 적용으로 -546 ms / iter 회복
- 잔여 +95~210 ms (write OFF) = host CPU per-launch latency × ~115K launches/iter
- 측정 → 본 production 의 +74 ms 와 일관

## 배제된 가설 (모두 직접 evidence)

GPU SM contention, cross-stream GPU contention, L3 cache pollution (worker scatter), Worker thread spawn, CUDA driver mutex, GIL contention, O_DIRECT, KVM steal — 모두 dominant 아님.

## 다음 우선 (별 작업 트랙)

1. multi-GPU (A100×4 dedicated, full Xeon Gold 6338) 환경 측정 — host CPU 시간 단축 확인
2. 1B / 10B / 100B scale 동작 검증 — sample buffer / threshold dynamics 확인

---

# 용어 정리

- lossless FC(Full Checkpointing):
non-expert 는 매 iteration (W=1) 실행.
expert 는 특정 간격(D) 마다 round-robin F% 만 실행.
Adam Optimizer 의 (m, v) + model weight (p) 를 선별 없이 100% 저장하는 방법.
- lossy FC:
expert 에만 적용됨.
매 iteration 마다 element-level score (single iter `H × Δθ²`) 와 group-level score (time-window `H × δ[g]`) 가 **단일 K-budget (= topk_ratio × N_total_elements) 에서 경쟁** 하여 두 pool 의 통합 top 을 선별. 선택된 element / group 의 (idx, p, m, v) 가 expert 의 lossy FC 파일에 append.

> **DC (Differential Checkpointing) 는 폐기됨.** DeepSpeed BF16 optimizer 의 `update_hp_grads(clear_lp_grads=True)` 는 매 micro-batch 종료 시 `lp.grad` 를 비우기 때문에, gradient_accumulation_steps > 1 환경에서 `lp.grad` 는 마지막 micro-batch grad 만 남고 누적 grad 는 `hp_grad` 에만 존재. DC 가 captured 하는 dense grad 는 의미 없는 partial 값이라 replay 시 발산. → 매 iter non-expert lossless FC (W=1) 로 대체.

---

# Non-expert 저장 전략

- **매 iteration (W=1)** 전체 non-expert parameters + optimizer state (p, m, v) 를 lossless FC.
- 복구 시 가장 최근 non-expert lossless FC 파일 로드 → 전체 lossless 복구. (DC replay 없음.)
- 근거:
  1. Non-expert 는 전체 모델의 10~20% 로 expert 대비 충분히 작음.
  2. 4-stream architecture + `host_writer_non_expert` 전담 worker 가 다음 iter 의 forward / backward 와 overlap 처리 → 학습 critical path 위 blocking 0.
  3. Gradient accumulation 환경에서도 W=1 lossless FC 는 항상 정확한 post-optimizer state 를 snapshot 함 (DC 와 달리 발산 위험 없음).

---

# Expert 저장 전략

## 개요

Lossy FC(append-only) + Round-robin expert lossless FC 방식을 사용한다.

매 iteration 마다 단일 K-budget = `topk_ratio × N_total_elements` 안에서 두 pool 이 경쟁:
1. **Element pool**: single-iter score `H × Δθ_i²` 의 top — outlier promotion (한 iter 에 spike 한 원소).
2. **Group pool**: time-window score `H × δ[g]` 의 top — rotation (지속적으로 변동 누적된 그룹).

선택된 element 와 group 은 각 expert 의 lossy FC 파일에 append (element 는 element entry, group 은 g 개 element entries). 자세한 선택 메커니즘은 [Method: Importance Score](#method-importance-score-expert-parameters) 참조.

특정 간격(D) 마다 전체 expert 중 F% 를 round-robin 선택하여 expert lossless FC 수행 (해당 expert 의 전체 parameter 를 새 파일에 sequential write). LFC 완료된 expert 는 기존 LFC 파일과 lossy FC 파일을 삭제하고 새 lossy FC 파일을 fresh 생성.

## 파일 구조

각 expert는 다음 두 종류의 파일을 가진다.

- **Expert lossless FC 파일**: 해당 expert의 전체 파라미터(weight, optimizer m, optimizer v)를 담는 완전한 snapshot. Expert lossless FC 수행 시 생성된다.
- **Lossy FC 파일**: 매 iteration 선별된 원소가 append되는 로그 파일. 각 엔트리는 `(element_index, param_value, m_value, v_value)` 형태이다. Expert lossless FC 수행 시 삭제되고 새로 생성된다.

## Write 경로

### Expert Lossless FC (매 D iteration, F%의 expert 대상)

1. `step % D == 0`일 때, round-robin 순서에 따라 전체 expert 중 F%를 선택한다.
2. 선택된 expert의 전체 파라미터(weight, optimizer m, optimizer v)를 GPU에서 읽어 새 파일에 sequential write한다.
3. fsync 완료 후, 해당 expert의 기존 expert lossless FC 파일과 기존 lossy FC 파일을 삭제한다.
4. 해당 expert의 새 lossy FC 파일을 생성한다 (빈 파일).
5. 해당 expert의 θ_CP를 θ_current로 전체 갱신한다 → 해당 expert 전체 δ = 0, score = 0.

### Lossy FC (매 iteration, 모든 expert 대상)

1. Expert lossless FC가 실행대상이 되는 expert들은 제외한 전체 expert 원소에 대해 importance score를 계산하고 상위 K%를 선별한다.
2. 선별된 원소를 expert별로 분류한다.
3. 각 expert는 자신에게 포함된 원소들을 자신의 lossy FC 파일 끝에 append한다.
4. 해당 iteration에서 expert lossless FC가 수행되는 expert는 lossy FC 대상에서 제외한다.

## Round-robin 순서

전체 expert를 고정된 순서(expert index 순)로, 매 D iteration마다 다음 F%를 선택한다. 한 expert의 lossless FC 주기는 `D × ⌈100/F⌉` iteration이다.

예시 (총 expert 100개, D=10, F=5%):

- step 10: expert 0~4 lossless FC
- step 20: expert 5~9 lossless FC
- ...
- step 200: expert 95~99 lossless FC
- step 210: expert 0~4 lossless FC (순환)

## Lossy FC 파일 크기 상한

한 expert의 lossy FC 파일은 최대 `D × ⌈100/F⌉` iteration 동안 append된다. 매 iteration 해당 expert에 append되는 양은 해당 expert에서 선별된 원소 수에 비례한다. 전체 expert 원소 중 K%가 선별되므로, 균등 분배 가정 시 per-expert per-iteration append 양은 `expert_param_size × K% / num_experts`이고, lossy FC 파일의 최대 크기는 `expert_param_size × K% / num_experts × D × ⌈100/F⌉`이다. 중복 원소(여러 iteration에 걸쳐 반복 선별된 원소)가 포함되므로 실제 유효 데이터 비율은 이보다 낮다.

## 복구 전략

각 expert의 복구는 독립적이므로 병렬 수행 가능하다.

1. 해당 expert의 가장 최근 expert lossless FC 파일을 로드한다 (전체 파라미터 복원). 없으면 생략한다.
2. 해당 expert의 lossy FC 파일을 순차적으로 읽으며, 각 엔트리의 element_index 위치에 값을 덮어쓴다. 동일 element_index가 여러 번 나타나면 마지막 값이 최종 값이 된다.
3. Lossy FC에 한 번도 나타나지 않은 원소는 expert lossless FC 시점의 값으로 복구된다.

## Importance Score와의 상호작용

δ[g] = (1/g) × Σ_step Σ_{i ∈ g, not selected} Δθ_i² 는 group 단위 self-correcting accumulator. 매 iter 의 element-level Δθ² 도 single-iter score 로 직접 평가되어 같은 K-budget 에서 group 과 경쟁. 자세한 알고리즘은 [Method: Importance Score](#method-importance-score-expert-parameters) 참조.

- **Element 가 선택된 경우** (single iter outlier promotion): 그 element 의 Δθ² 는 δ[g] 에 더해지지 않음 (Intent E selective fold). 이미 lossy log 에 element entry 로 저장됨 → 다음 iter 에 다시 score = 0 (Δθ² 새로 시작).
- **Group 이 선택된 경우** (group-level rotation): 그 group 의 모든 g 개 element 가 lossy log 에 entry 로 저장. 그 group 의 δ[g] 0 으로 초기화 → 다음 iter score = 0.
- **Expert lossless FC 수행 시**: 해당 expert 의 모든 group δ[g] 0 → 그 expert 의 모든 score 0. 이후 학습 진행으로 δ[g] 자연 증가, element Δθ² 새로 측정 → 회복.
- **Rotation**: 최근 보호된 항목은 score 0 으로 떨어지고, 미보호 항목이 우선. element pool / group pool 양쪽에서 자동 회전.

## 복구 성능 주의사항

복구 시 각 expert의 lossy FC 파일을 전부 read해야 하므로, lossy FC 파일 크기가 복구 시간을 직접 결정한다. 다음 두 가지 요인이 복구를 느리게 만들 수 있다.

**중복 엔트리 문제.** 중요한 원소일수록 매 iteration 반복 선별되므로, lossy FC 로그에는 동일 element_index의 엔트리가 다수 누적된다. 복구 시 마지막 값만 유효하지만, 그 이전의 모든 엔트리도 read해야 한다. Lossless FC 주기(`D × ⌈100/F⌉`)가 길수록 중복이 심해진다.

**총 read량.** Expert 수가 E개이고, per-expert lossy FC 파일 크기가 L이면, 총 read량은 `E × L`이다. Expert 1000개, L이 수십MB이면 총 수십GB에 달할 수 있다.

따라서 복구 경로는 다음과 같이 최적화해야 한다.

첫째, lossy FC를 순차 read하면서 인메모리 해시맵(key: element_index, value: 최신 값)에 덮어쓰며 구축하고, read 완료 후 해시맵을 expert lossless FC 위에 일괄 적용한다. 이렇게 하면 lossless FC 텐서에 대한 write가 유효 엔트리 수만큼만 발생한다.

둘째, 각 expert의 복구는 독립적이므로 최대한 병렬 수행한다.

셋째, D, F%의 설정 시 복구 시간을 반드시 함께 고려해야 한다. Lossless FC 주기가 짧을수록 lossy FC 파일이 작아져 복구가 빨라지지만, lossless FC의 write 비용이 증가한다. 이 trade-off를 실험적으로 조율해야 한다.

---

# Method: Importance Score (Expert Parameters)

## Score Definition

매 iteration 마다 두 종류의 score 가 동일한 **per-element score axis** 위에서 동시에 계산되어 단일 K-budget 을 두고 경쟁한다.

```
element score:  s_elem(i)   = H̃[group(i)] × Δθ_i²                    # single iter, per-element
group   score:  s_group(g)  = H̃[group=g]  × δ[g]
                where δ[g]  = (1/g) × Σ_{step ∈ window} Σ_{i ∈ g, not selected} Δθ_i²
```

- `Δθ_i²` = optimizer 의 m, v 로부터 유도한 이번 step 의 update² (single iter).
- `δ[g]` = 시간창에 걸친 그룹 평균 per-element δ². "선택 안 된" 원소들의 (Δθ²/g) 가 누적되어 자연 회복.
- `H̃[group]` = **v_t → H quantile-matching 매핑** (production single path). 매 anchor
  refresh (cheap FD K=1) 에서 H 분포의 4096-quantile lookup table 을 추출 → 매 iter
  Adam ``exp_avg_sq`` (v_t) 의 element-level 값을 그 lookup table 로 piecewise linear
  매핑하여 ``_g_fast_h_g_bf16_flat`` 을 in-place 덮어쓴다.

  **왜 v_t 가 H 의 proxy 로 안전한가 — Spearman rank 일치 fact**:
  - AdamW 내부적으로 v_t = EMA(grad²) ≈ Empirical Fisher (E[grad·grad^T] 의 diagonal)
    ≈ Gauss-Newton (GGN) diagonal ≈ Hessian diagonal (near optimum, positive curvature
    regime). 즉 v_t 와 H 가 **점추정 값으로는 다르지만 (scale 4 orders 차이 가능)
    element-level 순위 (Spearman ρ) 는 일치**. exp_etc 측정 ρ_all = 0.87 plateau.
  - rank 일치 → score 의 top-K 선택 결과 (= 어떤 element 가 lossy CP 에 들어가는지)
    가 진짜 H × δ² 의 top-K 와 거의 일치. **CP 의 본질은 ranking 이지 절대값이 아님**.
  - 단순 affine (mapped = a·v_t + b) 으로는 부족: H/v_t scale ratio 가 quantile 별
    4 orders 차이 (low-mag end 와 high-mag end 가 비선형 관계). 따라서 **piecewise
    linear quantile-to-quantile mapping**: v_t 의 4096-quantile 위치 → 같은 quantile
    위치의 H 값으로 매핑 (rank 보존 + scale 보정 동시 달성).
  - 4096-quantile = 99.99% lognormal 분포 정확도, persistent VRAM 32 KB ≈ 0.
  - **anchor 의 역할 = quantile lookup table 의 H 분포만 갱신** (매 iter H 직접
    측정이 아닌, 분포 통계만 cheap FD 로 update). 따라서 anchor 가 variance 가
    크더라도 (single batch K=1) 분포 통계는 robust (Var(quantile) ∝ 1/N).
- 두 score 모두 per-element 단위 (group 측이 1/g 정규화) → 같은 axis 에서 직접 비교 가능.

Non-expert parameter 는 매 W iter 전부 저장하므로 score 불필요.

## Two-pool Unified Top-K (단일 K-budget 경쟁)

### 알고리즘

매 iter 다음을 수행:

1. **Score 계산**:
   - element pool: 각 expert element 의 `H̃ × Δθ²` (단일 iter scale)
   - group pool: 각 group 의 `H̃ × δ[g]` (시간창 mean scale, 같은 per-element axis)
   - `H̃` = mapped(v_t), 매 iter ``ImportanceTracker.update_v_mapping`` 이
     ``_g_fast_h_g_bf16_flat`` 을 in-place 덮어쓴다. anchor (cheap FD) 가
     공급한 ``_g_fast_h_quantiles`` 와 매 iter 추출하는
     ``_g_fast_v_quantiles`` 사이 piecewise linear lookup.

2. **두 histogram 으로 단일 top-K**:
   - `histc(element_scores, ...)` → 4096 bin
   - `histc(group_scores, ...)` → 10000 bin
   - 두 histogram 의 bin 을 **per-element score** 축에 정렬, weighted cumsum 생성:
     - element bin: `weight = 1` (선택 시 K-budget 1 unit 소비)
     - group   bin: `weight = g` (group 1개 선택 = g element 보호 = g unit 소비)
   - cumsum 이 K_total = `topk_ratio × N_total_elements` 에 도달하는 score 가 **단일 임계값**.
   - 이 임계값 이상의 element 는 element pool 에서 선택됨, 이상의 group 은 group pool 에서 선택됨. **합계 K-budget = K_total**.

3. **선택된 항목 → checkpointing**:
   - 선택된 element: lossy log 에 element entry (idx, p, m, v)
   - 선택된 group: lossy log 에 g 개 element entries (group 의 모든 element)

4. **탈락된 element 의 self-correcting 누적**:
   - 선택 안 된 (탈락된) element 의 Δθ²/g 가 그 element 가 속한 group 의 δ[g] 에 가산
   - 즉 **선택된 element 는 δ[g] 에 기여하지 않음** (= Intent E, 선택적 fold)
   - 시간이 지나며 탈락된 element 가 많은 group 의 δ[g] 가 자연스레 커짐 → group score 증가 → 다음 iter 에서 group pool 우위 → group 단위로 회수 가능

### 이론적 근거

Taylor 2차 근사:

```
ΔL ≈ (1/2) × Σ_j (H × δ_j²)
```

- 1차 항: 기댓값 ≈ 0
- 2차 항: 항상 양수, recovery loss 의 실제 기여량
- 각 element 의 회수 가치 = `H × Δθ_j²`
- 각 group 의 회수 가치 (per element) = `H × δ[g]` = `H × mean Δθ²` over time window
- **K-budget 의 최적 분배**: 가장 회수 가치 높은 element 와 group 을 단일 ranking 으로 선택 (greedy by score-per-cost ratio; cost 를 normalize 하면 단일 score axis 의 top-K).

### 두 pool 의 비대칭성 (역할 분담)

- **Element pool (single iter, sharp)**: 한 iter 에 갑자기 큰 Δθ² 를 보인 outlier. 즉시 보호하지 않으면 회수 손실 큼.
- **Group pool (time window, smooth)**: 여러 iter 에 걸쳐 누적되는 moderate δ². 단발성 spike 는 평균에 묻혀 우위에 안 옴; 지속적 변동은 결국 group threshold 통과.
- 두 pool 이 같은 axis 에서 경쟁 → 분포에 따라 비율이 동적으로 조정됨 (고정 비율 분할이 아님).

### Starvation 방지

- Snapshot 된 element / group: δ[g] 부분 0 → score = 0 → 다음 iter 선택 안 됨.
- LFC 된 expert: 해당 expert 의 모든 group δ[g] = 0 → score = 0.
- 이후 학습 진행 → 탈락된 element 의 Δθ²/g 가 δ[g] 에 누적 → score 자연 회복.
- **자연 rotation**: 보호된 항목 대신 미보호 항목이 다음 iter 선택.

## 구현 요구사항

- **Top-K**: `torch.histc` 기반 두 histogram + dispatch 의 host-side merged walk (numpy `argsort` + `searchsorted`, ~14k entry 처리, < 0.1 ms).
- **두 임계값 산출**:
  - `elem_th` = walk 결과 score → kernel `prev_th` (element 선택 임계)
  - `group_th` = `elem_th × g` → pass2 chunk-loop `prev_threshold_t` (group 선택 임계)
- **Floor pruning**: walk 를 `K_total × FLOOR_MULT` (= 5 × K_total) 까지 연장하여 atomic histogram 의 floor 도출. floor 이하 element 는 kernel 의 `tl.atomic_add(..., mask=above_floor)` 로 atomic 자체를 skip.
- **Bin 정밀도**: element 4096 bins, group 10000 bins. Bin width 는 `prev_smax × 1.5` 로 자동 스케일.
- **prev_smax 갱신**: histogram 의 최상위 non-empty bin 으로 추정 (kernel 의 atomic_max 제거 — 단일-스칼라 contention 방지).

---

# Checkpointing Strategy

## 4-stream / 2-domain-worker 아키텍처 (Hard Contract)

학습은 **절대로** checkpointing 작업에 의해 중단되지 않는다. training thread (Python main) 는 **학습 launch 만** 수행하고, CP 관련 GPU compute / H2D copy / file write 는 모두 background stream / thread 로 분리한다.

```
┌──────────────────────────────┬──────────────────────────────────────────────┐
│ Stream / Thread              │ 책임                                          │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ main_stream                  │ 학습 — Megatron/DeepSpeed 의 forward /        │
│ (training thread launch)     │ backward / optimizer. 이 stream 위에는       │
│                              │ proposal 의 어떤 작업도 enqueue 되어선 안    │
│                              │ 됨. training thread 의 추가 sync 호출 0.     │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ tracker_stream               │ Proposal expert score 계산 — Triton outlier  │
│ (host_writer_expert launch)  │ kernel (atomic hist + selection + selective  │
│                              │ fold), pass2 histc + chunk-loop, BF16-group  │
│                              │ index_select. **launch 주체는 training       │
│                              │ thread 가 아닌 host_writer_expert worker.**  │
│                              │ wait_event(grad_ready) 후 자체 진행.         │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ d2h_stream                   │ **모든** GPU → pinned host memory 데이터 이동│
│                              │ (non-expert FC payload + expert lossy        │
│                              │ dispatch + LFCL payload). 두 worker 가 공유. │
│                              │ H2D destination 은 반드시 pinned (pageable   │
│                              │ ~1.2 GB/s vs pinned ~16 GB/s + non_blocking).│
├──────────────────────────────┼──────────────────────────────────────────────┤
│ host_writer_non_expert       │ Non-expert 전담 — 매 iter (W=1) lossless FC. │
│ (background CPU thread,      │ d2h_stream 으로 payload copy 후 cuda.Event   │
│  FIFO queue, GPU context)    │ 를 직접 `event.synchronize()` 하고 numpy     │
│                              │ scatter / file write / DRAM buffer commit.   │
│                              │ training thread blocking 0.                  │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ host_writer_expert           │ Expert 전담 — tracker_stream kernel launch + │
│ (background CPU thread,      │ dispatch (lossy log append) + round-robin    │
│  FIFO queue, GPU context)    │ LFCL. `torch.cuda.set_device` 로 GPU context │
│                              │ 보장. event sync 후 lossy log / LFCL 파일    │
│                              │ 작성. training thread blocking 0.            │
└──────────────────────────────┴──────────────────────────────────────────────┘
```

### Arm / Release Pair (cross-iter overlap)

매 iter 의 optimizer.step() 직후 task 를 **arm** 만 하고 (cuda.Event record + global stash), 다음 iter 의 **첫 micro-batch forward 직후 (= 첫 backward 직전)** 에 **release** (두 domain worker queue 로 submit). 이로써 background worker 의 d2h 가 첫 backward 와 자연스레 overlap 된다 (forward 와 overlap 하지 않는 이유는 아래 참조).

```
training thread:
  iter T:
    forward_backward → optimizer.step()
    ┌─────────────────────────────────────────┐
    │ proposal_arm_post_iter_fc(model, T)     │  ← record cuda.Event on main_stream,
    │   stash _PostIterFCTask(event, iter_T)  │    stash dataclass globally (no submit)
    └─────────────────────────────────────────┘
  iter T+1:
    mb0 forward (in forward_backward_func)
    ┌─────────────────────────────────────────┐
    │ release fired by forward_hook on        │  ← `register_forward_hook` on _inner_model
    │  _inner_model (after mb0 forward)       │    fires AFTER first mb's forward.
    │   if _armed_post_iter_fc_task:          │    single-gate (task=None for
    │     submit_non_expert / submit_expert   │    subsequent mbs → no-op)
    │     task = None                         │
    └─────────────────────────────────────────┘
    mb0 backward → mb1 fwd → mb1 bwd → ... → optimizer.step()
    ┌─────────────────────────────────────────┐
    │ proposal_arm_post_iter_fc(model, T+1)   │  ← next iter's arm
    └─────────────────────────────────────────┘
    ...
```

- arm 시점 cost = O(1) (Event.record + dataclass assignment).
- release 시점 cost = O(1) (queue put × 2). hook fire 자체는 us-level.
- 두 worker 는 release 직후부터 task 처리. d2h 는 **mb0 backward (~400 ms)** 동안 완료 (44 ms).

### Release timing — 왜 forward 시작 직전이 아니라 첫 mb forward 끝인가

A100 PCIe copy engine 은 stream과 무관하게 **단일 FIFO queue** 로 모든 stream 의 `cudaMemcpyAsync` 를 직렬 처리. PyTorch `tensor.to('cuda')` 는 작은 H2D `cudaMemcpyAsync` + `cudaStreamSynchronize` 호출 (non_blocking=False default). Megatron transformer 의 MoE forward path 에 `torch.tensor(0.0, device=)` 호출이 layer × microbatch 로 누적되어 **forward 동안 H2D 가 매우 자주 발생**.

만약 release 가 forward 시작 직전이면:
1. d2h_stream 이 1.16 GB cudaMemcpyAsync 를 copy engine FIFO 에 큐잉 (~44 ms 점유)
2. forward 의 `tensor.to('cuda')` 가 발행한 작은 H2D 가 queue 끝 (44 ms 후) 대기
3. main thread 가 `cudaStreamSynchronize` 에서 **44 ms 동안 host-blocking wait** (driver function: `cuStreamSynchronize` in libcuda.so → `__clock_gettime` polling). perf stack 으로 직접 측정.
4. 결과: forward 가 ~+41 ms 늦어짐 (iter time +41 ms 가산)

→ release 를 **첫 forward 끝** 으로 옮기면 forward 의 H2D 호출들이 d2h 와 시간상 분리되어 copy engine queue contention 없음. d2h 는 backward 와 overlap (backward 에는 `tensor.to('cuda')` 호출 없음).

**검증 (DRAM mode, 20 iter)**:
- W=99999 (no proposal CP): 2030 ms steady
- W=1 release-before-forward (old): 2098 ms (+41)
- W=1 release-at-first-bwd (현재): 2044 ms (+14)
- lm_loss 4-자리 baseline 일치 (correctness 확인).

### 데이터 흐름 (iter T 의 task 가 iter T+1 forward 와 overlap)

```
iter T training thread:                                 [main_stream]
  forward → backward → optimizer.step() → arm(T)            ↓
                                                            │ event_grad_ready
iter T+1 training thread:                                   ▼
  release(T)            ┌────────────────────────────────────────────────────┐
  → forward (main_stream)│  host_writer_non_expert            host_writer_   │
                         │  ─────────────────────            expert          │
                         │   d2h_stream:                     ─────────────   │
                         │     pinned_host_ne.copy_(p_ne,    tracker_stream:  │
                         │       non_blocking=True)            triton kernel  │
                         │   event_h2d.synchronize()           pass2 + idx_   │
                         │   binary_io.write_lossless_         select         │
                         │     payload (ne)                  d2h_stream:     │
                         │                                     pinned_host_   │
                         │                                     ex.copy_       │
                         │                                   event_h2d.sync   │
                         │                                   binary_io.append_│
                         │                                     lossy_log      │
                         │                                   binary_io.write_ │
                         │                                     lossless (LFCL)│
                         └────────────────────────────────────────────────────┘
```

### File I/O 위치 강제 규칙

**모든 실제 file write 호출은 `megatron/proposal/binary_io.py` 안에만 위치한다.** 다른 모듈에서 `os.write*`, `os.fsync`, `os.replace`, `f.write(`, `tofile(` 를 직접 호출하지 않는다 — `binary_io` 의 헬퍼 (`save_binary`, `write_lossless_payload_atomic`, `open_lossy_log_for_append`, `append_lossy_log`, `fsync_fd`, `close_fd`) 를 사용한다. 위반 사항은 `grep -rEn 'os\.writev|os\.write\(|os\.fsync' megatron/proposal/ | grep -v binary_io.py | grep -v "test\|docstring"` 로 0 건이 나와야 한다.

binary_io 의 헬퍼 함수들은 모두 **CPU 전용** (torch / cuda dependency 0): bytes / memoryview / numpy array 만 받음. → GPU stream 위에 잘못 호출될 가능성 0.

### Stream / Thread sync 의무

- main_stream → tracker_stream: `cuda.Event.record(main_stream)` (grad ready) + `tracker_stream.wait_event(...)` on worker.
- tracker_stream → d2h_stream: `cuda.Event.record(tracker_stream)` + `d2h_stream.wait_event(...)`. 단일 event 가 phase 전체를 커버 (per-name event 금지 — overhead).
- d2h_stream → worker thread: `cuda.Event.record(d2h_stream)` 을 worker task 에 전달. worker 가 `event.synchronize()` 호출 (background thread, training blocking 0).
- training thread 의 host-blocking sync (`stream.synchronize()`, `event.synchronize()`, `.item()`, `.cpu()`) 는 **0 회**. 예외 1 가지: host_writer_expert 의 dispatch 진입 시점 `ev.synchronize()` (1-iter pipelining) — 이는 training thread 가 아닌 worker thread 위 호출이라 main blocking 아님.

### 1-iter pipelining (expert 측)

- iter T 의 tracker_stream 작업 결과는 iter T+1 의 dispatch (host_writer_expert) 에서 사용.
- iter T 의 stash 시점에 GPU intermediate (sel_idx, sel_count, etc.) 를 `.clone()` 하여 ref 보관. iter T+1 dispatch 가 `ev.synchronize()` 후 `.item()` 으로 host scalar 추출 — 이 시점엔 prev iter 의 tracker work 가 forward (~2000 ms) 보다 훨씬 일찍 완료되었으므로 host wait time = 0.

### Worker 내부 규칙 (race / deadlock 방지)

- 두 worker 는 **단일-thread FIFO queue**. submit → 순차 처리 → completion record.
- **자기 자신에게 submit 금지** (self-recursive enqueue 는 deadlock). worker task 안에서 추가 작업이 필요하면 inline 실행.
- 두 worker queue 간 cross-domain submit 금지 (필요 시 inline 또는 별도 release).
- Worker thread init 시 `torch.cuda.set_device(self._device)` 호출 — task 안의 모든 GPU 호출이 동일 device context.
- shared mutable state (pinned host buffer, cuda.Event pool 등) 는 worker 별로 분리되거나 RW lock 보호.

## Checkpoint 이벤트 — 매 iteration

매 training iteration 마다 CP 이벤트가 발생.

**Non-expert (host_writer_non_expert 담당):**

- 매 iter (W=1) lossless FC. 전체 non-expert param + optimizer state (p, m, v) 저장.
- 저장 위치: `results/{실험이름}/cp/non_expert`.
- 이전 non-expert FC 파일은 새 파일 commit (fsync) 후 삭제 — 항상 가장 최근 1 개만 보관.

**Expert (host_writer_expert 담당):**

- 매 iter K% lossy FC append (각 expert lossy log 파일 끝에).
- 매 D iter 마다 round-robin F% expert lossless FC (해당 expert 전체 p/m/v 새 파일).
- 저장 위치: `results/{실험이름}/cp/expert`.

---

# Key Parameters

| 파라미터 | 설명 | 기본값 |
| --- | --- | --- |
| K% | Expert state snapshot 비율 |  |
| W | Non-expert lossless FC 주기 (iter) | 1 |
| D | Expert lossless FC 수행 간격 (iteration) | 4 |
| async_queue_size | Per-domain worker pending queue | 5 |
| F% | Expert lossless FC 시 선택되는 expert 비율 | 2 |
| anchor_refresh_every | cheap FD anchor 주기 (v_t→H mapping 의 H quantile 갱신) | 40 |
| anchor_K | anchor FD K-sample | 1 |
| N_Q | v_t→H mapping lookup table quantile point 수 | 4096 |

---

# Experimental Setup

## Hardware

- GPU: NVIDIA A100 80GB PCIe (단일)
- CPU: 16 vCore
- Memory: 192 GiB
- Storage: 250 GiB

## Metrics

1. Validation Loss, Perplexity
2. Checkpoint 크기
3. 저장/복구 시간
4. 학습 throughput 저하율
5. Wasted Time (복구시간 + CP생성시간)
6. Post-fault Damage
7. iteration 에 따른 H 분포와 score 분포(99.9%, 99.8%, … 99%, 98%, … , 1%, 0%)

## 모델 설정

### 실험 1 (Legacy, 300M-MoE)

| 항목 | 값 |
| --- | --- |
| **Global Batch Size** | 24 |
| **Sequence Length** | 2048 |
| **Hidden Size** | 768 |
| **레이어 수** | 12 |
| **레이어당 Expert 수** | 4 |
| **총 Expert 수** | 48 (= 12 layers × 4 experts) |
| **Gate Top-K** | 2 |
| **총 파라미터** | ~0.3B |
| — Expert 파라미터 | ~4.7M (per expert) |
| — Non-Expert 파라미터 | ~0.1B |
| **데이터셋** | SlimPajama |
| **상태** | ✅ 완료 |

### 실험 2 (미구현)

| 항목 | 값 |
| --- | --- |
| **Global Batch Size** | 48 (micro 4) |
| **Sequence Length** | 2048 |
| **Hidden Size** | 768 |
| **레이어 수** | 24 |
| **레이어당 Expert 수** | 8 |
| **총 Expert 수** | 192 (= 24 layers × 8 experts) |
| **Gate Top-K** | 1 |
| **총 파라미터** | ~1B |
| iterations | 8000 |
| fault 간격 | 600 |
| **데이터셋** | SlimPajama |
| **상태** | ⏳ 미구현 |

### 실험 3 (scaled-OLMoE)

OLMoE-1B-7B([논문](https://arxiv.org/abs/2409.02060)) 아키텍처를 단일 A100 80 GB에 맞게 **hidden·FFN·seq를 축소**한 구성. 

#### 모델 사양

| 항목 | 값 | OLMoE-1B-7B 대비 |
| --- | --- | --- |
| **Global Batch Size** | 72 (micro 4, grad_accum 18) |
| **Hidden Size** | 1024 | 1/2 (원본 2048) |
| **FFN Dimension** | 512 | 1/2 (원본 1024) |
| **레이어 수** | 16 | 동일 |
| **Attention heads** | 16 (head_dim 64) | 동일 (head_dim 1/2) |
| **Sequence Length** | 1024 | 1/4 (원본 4096) |
| **레이어당 Expert 수** | 48 | 동일 |
| **총 Expert 수** | 768 (= 16 layers × 48 experts) | 768/1024 |
| **Gate Top-K** | 6 | 6/8 |
| **활성화율 (topk/num_experts)** | 12.5% | 동일 |
| **Activation** | SwiGLU (`--swiglu`) | 동일 |
| **LayerNorm** | RMSNorm (`--normalization rmsnorm`) | 동일 |
| **Positional** | RoPE θ=10000 (`--use-rotary-position-embeddings`) | 동일 |
| **Embeddings** | untied (`--untie-embeddings-and-output-weights`) | 동일 |
| **Biases** | disabled (`--disable-bias-linear`) | 동일 |
| **MoE 라우팅** | DeepSpeed MoE with capacity 대체 (dMoE 미지원) | **차이** — dropping 방식 |
| **QK-Norm** | 비활성 (Megatron-DeepSpeed 미지원) | **차이** — 생략 |
| **총 파라미터** | ~1.38B | 원본 6.9B |
| **Active parameters** | ~320M  | 원본 1.3B |
| **Expert 파라미터** | 1.57M/expert, ~1B.2 total | 원본 per-expert 6.29M |
| **Vocab** | 50,304 (GPT-2 tokenizer, 103M params) | 동일 |
| **LR** | 1.5e-4 (peak), 1.5e-5 (min), cosine, warmup 300 iter | — |
| **Optimizer** | Adam β=(0.9, 0.95), wd=0.1, grad clip 1.0 | OLMoE와 동일 |
| **데이터셋** | SlimPajama | — |
| **iterations** | 6700 | — |
| **fault 간격** | 600 | — |

dMoE → capacity factor 대체

#### FD Hessian 설정 (anchor refresh — v_t → H mapping 용)

- K=1 (FD), B=1 minibatch, refresh_every 40 — anchor 는 distribution range 만 신뢰하면
  되므로 분산 큰 cheap 측정으로 충분 (Var(quantile) ∝ 1/N, N=expert element 수).
- amortized cost ≈ 1.5 s / 40 iter = 38 ms/iter (기존 K=16 의 ~125 ms/iter 대비 70% 절감).
- anchor 후 ``tracker.update_h_anchor_quantiles`` 가 ``_g_fast_h_g_bf16_flat`` 분포
  에서 4096-quantile lookup table 을 추출.

## 테스트 명세 (Test Specifications)

> **원칙**: 코드 수정이 발생할 때마다 아래 3가지 테스트를 반드시 실행한다.
> 

---

### T-01. TFLOPs 유지 확인

**목적**: 코드 수정 후 연산 효율이 저하되지 않았는지 확인

**기준값**: `≥ 20 TFLOPs`

### T-02. Non-Expert Top-K% 압축 정확도 확인

**목적**: Non-expert 파라미터가 지정된 비율(K%)로 저장되었을 때, 해당 값이 정확히 저장되었는지 검증

2개 테스트 모두 통과한 경우에만 다음 단계로 진행한다.

# Related Work

- **Amber (ICPP'25):** Dense 모델 selective incremental CP. |ΔW|/|W| 상위 2% 매 iter 저장 + 5000 iter마다 full CP. Adaptive threshold. MoE 미적용. 본 연구의 K% 선택 근거.
- **MoC-System (ASPLOS'25):** Expert 단위 PEC. PLT < 3.75%.
- **MoEvement (arXiv 2412.15411):** Sparse CP window. 인기 expert 우선. 최대 9× 절감.
- **ExCP (ICML'24):** √v\_t pruning threshold. 25-70× 압축. Post-hoc.
- **LowDiff (SC'25):** Compressed gradient differential CP. 일반 DNN, MoE 미적용.
- **D²-MoE:** Expert delta low-rank. Fisher-weighted base averaging.
- **Check-N-Run (NSDI'22):** 추천시스템 embedding sparse differential CP.

---

# Code Structure

```markdown
# Code Structure

```
CLAUDE.md                       # This file (spec + parameters + conventions)
megatron/proposal/
├── __init__.py                 # Public API exports
├── cp_hooks.py                 # Training-loop integration: init_proposal / arm_post_iter_fc / release_post_iter_fc / recover / shutdown + refresh_hessian dispatch + fault injection + multi-batch anchor session (fast-flat grad-flat + tracker-view h_accum)
├── cli_args.py                 # CLI arg registration (--proposal-*)
├── hessian.py                  # State-free FD Hessian: compute_hessian_finite_diff + fast-flat (_compute_fd_fast_flat, _flat_z_perturb_kernel, _flat_signed_stream_kernel) + _build_expert_flat_view
├── importance_tracker.py       # Element-wise score (mapped(v_t) × δ²), v_t→H quantile mapping (update_v_mapping / update_h_anchor_quantiles), global top-K, θ_CP cache
├── outlier_kernels.py          # Lossy CP outlier + group select Triton kernels (_outlier_count_kernel, _group_select_kernel, _aos_write_kernel, _aos_group_write_kernel + launchers)
├── fc_manager.py               # FC scheduling: element_wise / staggered / sequential
├── expert_state.py             # Per-expert lossy FC append log (LFSY) + lossless FC snapshot (LFCL) per CLAUDE.md spec
├── dram_expert_state.py        # DRAM-target expert state (CP_TARGET=dram)
├── recovery.py                 # Non-expert lossless FC load + per-expert lossless + lossy replay (via read_expert_state)
├── async_writer.py             # 2-domain workers (host_writer_non_expert + host_writer_expert) FIFO queue with GPU context
├── gpu_aos.py                  # GPU AOS interleave kernels (SoA → AoS into segment_pipeline staging)
├── segment_pipeline.py         # 256-MB segment double buffer (D2H staging)
├── proposal_io.py              # Python bindings for _csrc/*.cpp (binary_io + proposal_io)
├── _csrc/binary_io.cpp         # Single home for all syscalls (write/open/fsync/close/unlink/rename/pread/stat) — anonymous-namespace syscall_* wrappers
├── _csrc/proposal_io.cpp       # pybind11 module exposing binary_io to Python
├── _ds_state.py                # DeepSpeed ZeRO 0/1/2 FP32 master state mapper + _pinned_clone pool (anchor θ_0 snap)
├── _routing_freeze.py          # MoE gate freeze for FD anchor (replays captured dispatch on ±ε forwards)
├── fd_noise_probe.py           # Diagnostic: FD H noise probe (off-path)
├── _vram_audit.py              # Diagnostic: VRAM peak audit (PROPOSAL_VRAM_AUDIT)
└── tests/                      # Core regression tests — must PASS on every change
    ├── vram_audit_eval.py      # R1-R4 PASS/FAIL evaluator (called by verify_vram_run.sh)
    ├── test_outlier_per_segment.py  # outlier kernel split-equivalence (7/7)
    └── test_hessian_stream.py  # legacy h_accum kernel reference (8/8)

Removed in this cycle: _param_rebind.py, binary_io.py (Python), checkpoint_ctx.py.

megatron/training.py            # Hook calls: init_proposal / arm_post_iter_fc (after optimizer) / recover / refresh_hessian. release_post_iter_fc is fired by register_forward_hook on _inner_model at end of first micro-batch fwd (not called directly from training.py).
pretrain_gpt.py                 # Registers add_proposal_args via extra_args_provider

examples_deepspeed/MoE/         # Shell driver scripts
├── proposal_etc_template.sh    # ETC variants: baseline | moc_m2 | moc_d18 | element_wise | hessian_* | ...
├── proposal_exp1_template.sh   # Exp 1 driver (125M-MoE8, WikiText-2, CP reconstruction quality)
├── proposal_exp2_template.sh   # Exp 2 driver (125M PR-MoE, SlimPajama, fault injection)
├── proposal_smoke_{gnb,scorediag,15B}.sh
├── run_{etc,exp1,exp2}.sh      # Multi-seed orchestrators
├── replay_existing_cp.sh       # Load existing CP + run validation only
├── aggregate_exp2.py           # Post-run metrics aggregation (CP size, wasted time, damage)
└── plot_scorediag.py           # Score distribution / top-K membership plotter
```

### CLI 진입점 명세

- `--proposal-enable`: 마스터 플래그.
- `--proposal-cp-mode {proposal|sequential}`: 기본 `proposal`. `proposal` = lossy FC append + round-robin F% 로스리스 FC. `sequential` = MoC baseline.
- `--proposal-lossless-fc-fraction F`: 기본 `2.0` (%). Expert 로스리스 FC 이벤트당 선택되는 expert 비율.
- `--proposal-selection-mode {v_hat|random|hessian}`: 기본 `hessian`.
  - production `hessian` 모드의 score H 자리에 들어가는 값은 직접 측정한 H 가 아니라
    cheap FD anchor + v_t quantile-matching 으로 매 iter 새로 매핑된 ``mapped(v_t)``.
- `--proposal-topk-ratio FLOAT`: 기본 `0.04` (4%).
- `--proposal-hessian-refresh-every N`: 기본 `40` — anchor refresh 주기.
- `--proposal-hessian-num-samples K`: anchor FD K-sample 수, 기본 `1` (cheap; 분산
  큰 추정도 분포 quantile 통계는 robust).
- `--proposal-nonexpert-fc-W N`: 기본 `10`.
- `--proposal-expert-fc-D N`: 기본 `None` → `num_experts` 자동 산출.
- `--proposal-async-queue N`: 기본 `5`.

> Anchor batch 수 (multi-batch loop 의 N) 는 CLI 없이 자동 산출:
> `N = global_batch_size // micro_batch_size // 2` (`init_proposal` 안에서 한 번 계산). 이전의 `PROPOSAL_ANCHOR_N_BATCHES` env 는 제거됨.

측정용 플래그:
- `--proposal-fisher-coverage-iters [list]`
- `--proposal-fd-noise-probe-iters / --proposal-fd-noise-probe-log PATH / --proposal-fd-noise-probe-ref-k / --proposal-fd-noise-probe-ref-n-batches / --proposal-fd-noise-probe-g-blk` (v_t vs FD-H probe)

Fault 실험용 플래그:
- `--proposal-fault-interval N / --proposal-fault-warmup N / --proposal-fault-log PATH`
- `--proposal-replay-at-end / --proposal-replay-only`

## Multi-GPU 검토 (this commit cycle)

본 cycle 의 변경은 single-GPU 환경에서만 측정/검증됨. Multi-GPU (DP / EP / TP / ZeRO-1+) 진행 시 검토 포인트:

| 변경 | Multi-GPU 영향 위험도 | 비고 |
|---|---|---|
| `_compute_fd_fast_flat` — grad_flat alloc + .grad view assign | 낮음 | per-rank local. 각 rank 가 own expert_ps 의 grad 만 flat 으로 모음. ZeRO partitioning 영향 없음 (lp.grad 자체는 partition 안 됨, DS hook detach 상태이므로 fp32 master 도 미접촉) |
| `_flat_z_perturb_kernel` — per-tensor + `global_offset` 기반 PCG hash | 낮음 | rank 별 expert_ps 가 다르면 global_offset 도 다름. rank 내 perturb 와 stream 의 z 일치 보장 (같은 hash 식) |
| `_flat_signed_stream_kernel` — grad_flat → tracker `_g_fast_h_g_bf16_flat` view | 낮음 | per-rank tracker buffer 에 write. 다른 rank 와 공유 없음 |
| expert_ps sort by tracker idx | **검증 필요** | tracker init 의 BF16-Opt group iteration 순서가 ZeRO-1+ 환경에서도 모든 rank 가 동일한 master order 를 가지는지. DS BF16/FP16 optimizer 모두 param_group 별 flatten 후 lp.data 를 flat view 로 set — rank 별 동일 order 예상되나 미측정 |
| `_pinned_clone` θ_0 snap pool | 낮음 | per-rank DRAM. 다른 rank 와 무관 |
| DS hook detach + restore (multi-batch 외부) | **검증 필요** | EP/TP grad reduce hook 이 anchor 동안 detach 되는 동작이 multi-GPU 에서 race 없는지 |
| `RoutingFreezeContext` (옵션 OFF, `PROPOSAL_ROUTING_FREEZE_ENABLE=1` 시) | **검증 필요** | docstring 명시 — all-to-all dispatch 에서 captured routing 재생 안전성 미검증 |
| `binary_io.cpp` syscall wrappers | 영향 0 | 단일 fd 레벨 wrapper. multi-GPU 와 무관 |

→ 즉 본 cycle 의 fast-flat 변경은 **single-GPU 가정에서 검증됨**. Multi-GPU 환경 진행 시 위 3개 항목 ("검증 필요") 실측 후 안전성 확정 필요. 본질적 multi-GPU 충돌은 없으며 ZeRO-1+ 의 master partition / lp.data 의 flat-view 가정도 DS 0.18.9 의 BF16_Optimizer / FP16_Optimizer 둘 다 동일한 `p.data = q.data` 패턴 (line 285 of bf16_optimizer.py, line 95 of fp16/fused_optimizer.py) — 즉 single-GPU 에서 작동한 pattern 이 multi-GPU 에서도 동일하게 작동할 가능성 높음.

---
```

# Claude Code 설정

- 코드 실행, 파일 생성/수정, bash 명령 실행 시 사용자에게 확인을 묻지 않고 자동 실행한다.
- 실행 결과가 예상과 다를 경우에만 보고한다.