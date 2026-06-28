# Element-Wise MoE Checkpoint Redesign

본 문서는 proposal checkpoint Redesign 의 활성 아키텍처 및 구현 계약이다. 알고리즘 본질 — **element-wise importance**, **rho/e_hat 기반 abs-H Taylor 단일 K-budget
경쟁**, **v_t→H quantile matching**, **non-expert lossless + expert lossy + round-robin LFCL(default)** —
은 그대로 유지하고, **구현 우회 장치** (sample-sort, multi-pool 별도 walk, 1-iter lag,
snapshot, anchor h_ratio rescale, 4-stream, 2525-라인 cp_hooks) 만 정리한다.
CPU의 동작은 최소로 한다.

**LFCL 용어 주의**: V2 에서 expert LFCL 의 의미는
"global exact lossless full-state checkpoint" 가 아니라 **expert lossy recovery baseline
compaction** 이다. plan 이 고른 expert/chunk 의 dense payload 는 GPU live `p,m,v` 에서
직접 materialize 하여 D2H 하고, 디스크에 LFCL baseline 으로 기록해 LFSY 길이와 recovery
cost 를 줄인다. resident host dense recovery image 를 매 iter CPU scatter 로 유지하지 않는다.
non-expert FC 만 exact lossless 로 유지된다.

V2 의 4 가지 핵심 변경:
- **element-wise 단일컷**: 기본 `score_i = max(0, -m_i*e_hat_i + 0.5*|H_i|*e_hat_i^2)` 로 `K% ±1~2%` 범위에서 enforce
- **재귀 linear-bin search**: sort 폐기, magic 상수 0
- **단일 proposal compute stream + 1 d2h**: V1 compute split 병합 (host-sync 이미 직렬화)
- **cp_hooks 삭제 완료**: hooks / pipeline / h_refresh / diagnostics 4 모듈로 분할

### Selection mode (`ProposalConfig.selection_mode`)

`hessian` (default, V2 의 모든 §1-§2 서술 대상) 외에 두 ablation mode 유지:
- `v_hat`: H̃ 자리에 v_t 직접 사용 → anchor 자체가 no-op, Stage 3 skip
- `random`: scoring 전체 무시, random τ 산출

V2 의 정확성·성능 contract 는 `hessian` 에만 적용. 두 ablation 은 CLI 호환 유지용.
hessian을 제외하곤 단순 손실률 비교 측정용이므로 성능에 주목하지 않아도 됨.

---

## 0. Hard contract

매 변경마다 `megatron/proposal/tests/vram_audit_eval.py` PASS 필수. 위반 시 commit 금지.

### 0.1 VRAM / allocation contract (R1 ~ R4)

| R | 항목 | 한도 | 검증 방식 |
|---|---|---|---|
| **R1** | Persistent VRAM Δ (steady-state, 학습 안정 phase) | **V2 ≤ V1**, proposal-off 대비 audit limit 내 | V2 는 V1 scale-dependent buffer 를 줄이는 방향이어야 한다. proposal-on iter `torch.cuda.memory_allocated()` 시계열과 proposal-off baseline 을 함께 기록. 100+ iter 평균. |
| **R2** | Per-iter proposal-callsite `cudaMalloc` / `cudaFree` | **0 회** | nsys / py-spy stack 의 cudaMalloc caller 추적; caller module 이 `megatron/proposal/*` 면 위반. scale-independent scratch (≤ tens of KB) 도 init 단계에서만 alloc, per-iter 0. |
| **R3** | Anchor iter peak − training iter peak | **≤ 100 MB** | proposal-on anchor iter `max_memory_allocated` vs proposal-on training iter `max_memory_allocated`. anchor 는 학습이 실제 gradient 를 accumulate 하는 storage/view 를 재사용하므로 별도 `grad_flat` alloc 이 없어야 한다. |
| **R4** | 누적 VRAM slope + fragmentation | slope = 0 MB/iter, `reserved − allocated` waste < limit (audit_eval 참조) | 100+ iter 후 alloc 시계열 linear fit slope, caching allocator waste 측정. |

> 검증 현황: R1·R2·R4 PASS (마지막 full audit). R3 는 anchor refactor (§2.1/§2.2) 후 재검증 필요.

### 0.2 Stream / blocking contract

| 항목 | 한도 | 예외 (anchor halt 중) |
|---|---|---|
| `main_stream` 위 proposal launch (per-iter) | **0** | Anchor Stage 2 의 ±εz forward/backward 가 main_stream 사용 — training halt 중이라 contention 없음 |
| training thread 의 host-blocking sync (`torch.cuda.synchronize` / `event.synchronize` / `.item` / `.cpu` / non-pinned H2D) | **0** | Anchor stage 전환점 (`torch.cuda.synchronize` × 2회 — §2.4), opt 전에만 허용 |
| `compute_stream` / `d2h_stream` 위 per-iter `cudaMalloc` / `cudaFree` | **0** | Anchor halt 중에도 proposal 모듈의 새 GPU buffer alloc 은 금지. Hessian 추정은 기존 학습 grad accumulation storage/view 에 누적한다. |
| Host → Device copy 의 pageable host buffer | **0** (pinned 만) | — |

### 0.3 Host resource contract

| 항목 | 한도 | 비고 |
|---|---|---|
| Host RAM total | segment pinned double buffer + single pinned AOS slab + single per-iteration pinned LFCL batch slab + scratch / metadata (~MB scale + fired payload capacity) | production `disk` target 은 resident host dense expert recovery-image buffer (`Σ_local_expert numel × 12 B`) 를 유지하지 않는다. LFCL 의 직접 write source 는 fired experts 를 한 iter 단위로 pack 한 contiguous pinned LFCL batch slab 이다. 테스트 전용 `test_dram` target 만 pinned dense expert recovery image 를 alloc-once 로 유지한다. |
| **Pinned host memory alloc** (`cudaHostAlloc` / `torch.empty(..., pin_memory=True)` / `tensor.pin_memory()`) | **0 회 per-iter** | (a) init 단계 alloc-once: single pinned AOS slab, segment pinned double buffer, single pinned LFCL batch slab, ne_writer pinned, metadata, `test_dram` pinned dense image. (b) anchor 의 첫 호출에서 `pinned_snapshot` pool 에 key 별 1 회 alloc 후 모든 후속 anchor 가 같은 buffer 재사용. 둘 다 학습 안정 phase 에서 0 회. |
| Pageable host buffer 를 H2D / D2H source/dest 로 사용 | **0** | 위 contract 의 직접 귀결 — 모든 host buffer 가 pinned 이라 pageable 경유 자체 불가 |
| ne_writer / expert_writer 의 per-iter Python heap alloc (numpy scratch 등) | **scale-independent 만 허용** (per-iter 수십 KB 수준) | 큰 array (model-dependent size) 는 init 단계 alloc-once. iter 마다 새 ndarray 만들지 말 것 |

Pinned AOS slab 용량은 cutoff 정확도 tolerance 와 별개의 host-buffer 계약이다.
production 은 `topk_ratio × (1 + pinned_aos_capacity_tolerance)` records 를 기준으로
single pinned slab 를 init 단계에서 한 번만 잡으며, 기본 `pinned_aos_capacity_tolerance=0.75`
는 K-budget 주변의 local burst, per-expert prefix, 4 KiB alignment padding 을 흡수하기 위한
용량 여유다. cutoff controller 는 여전히 strict K-band(`cutoff_K_tolerance`,
기본 0.04) 를 정확도 기준으로 사용한다. host slab 여유를 cutoff feedback band 로
재사용하거나, 반대로 cutoff tolerance 로 pinned AOS capacity 를 축소하면 안 된다.
single slab 의 총 byte capacity 는 고정이며 per-iter pinned allocation 은 없다. GPU layout
state 는 current iter 의 approximate probe 가 산출한 expert별 `capacity_demand_hint` 와
previous iter direct cursor 의 exact selected demand 중 큰 값을 weight 로 사용해
**expert별 contiguous slice offset/capacity** 를 결정한다. Direct-emission
`cursor/overflow` 는 slot reservation, prefix count, overflow hard guard, diagnostics, 그리고
다음 iter capacity 하한 feedback 용도다. CPU 는 layout 을 학습하거나 결정하지 않는다.
GPU 가 결정한 offset table 은 AOS slab 맨 앞의 최소 header 로 함께 D2H 되고, CPU writer 는
그 table 로 expert별 prefix 위치를 찾아 `count/step` 과 payload 를 소비한다.

---

## 1. Per-iteration pipeline (5 stages)

### 1.-1 시작 시점 — `forward` 시작 전 dispatch (old deferral 폐기) — ✅ 완료

**삭제된 V1 deferral 패턴**:
```
iter T  : optimizer.step() → Event record + global stash, submit 안 함
iter T+1: mb0_fwd 끝 (register_forward_hook fire) → worker queue 에 submit
```

**V2 원칙**: V1 의 deferral 은 forward 내부 sync 문제를 우회한 것이다. V2 는 우회가
아니라 **정확한 sync 원인을 제거/격리**하여, proposal 이 iter 의 첫 동작으로 dispatch
되어도 forward 가 proposal stream 을 기다리지 않게 만든다.

**현재 의심 지점**: DeepSpeed timer, `transformer.py` 의 `torch.tensor(0.0, device=...)`,
forward 중 `.item()` / `.cpu()` / pageable H2D, 또는 legacy default-stream semantics 가
`cudaStreamSynchronize` 를 유발할 수 있다. 어느 호출이 실제 원인인지는 nsys + NVTX 로
확정해야 하고, 확정한 후 해결한다.

**V2 iteration sequence**:
```
iter T (T ≥ 1):
  proposal_dispatch  ──► Event.record(main_stream)
                         ne_writer.submit(task(T-1's opt 산출, event_grad_ready))
                         expert_launcher.submit(task(T-1's opt 산출, event_grad_ready))
                         (μs scale, 즉시 반환)
  forward            ─── main_stream; 내부 sync 가 proposal stream 을 기다리면 V2 구현 실패
  backward
  wait               ─── proposal_fence_before_optimizer
                          main_stream.wait_event(d2h_done_event)
                          (workers 의 AOS D2H 까지 optimizer.step 전 완료 보장)
  optimizer.step()

iter 0 (특수 케이스):
  forward → backward → optimizer.step()
  ↑ proposal / wait 모두 skip — 이전 iter 의 opt 산출이 없음
```

**Race / correctness 분석**:
- iter T 의 proposal 은 iter T-1 의 optimizer.step 산출 (p, m, v) 을 D2H — workers 가
  `wait_event(event_grad_ready)` 후 진행하므로 post-step state 를 봄 (T-1 의 step 완료 후
  T 의 dispatch 시점에 event_grad_ready 가 record 됨). ✓
- iter T 의 forward / backward 는 `.grad` 만 mutate. workers 는 `.grad` 미접근. ✓
- iter T 의 optimizer.step 이 p, m, v 를 다시 mutate. 그 전에 workers 의 read/D2H
  완료 필요 → `proposal_fence_before_optimizer` 가
  `main_stream.wait_event(d2h_done_event)` 로 강제. ✓

**V1 deferral 과의 차이**:
| | V1 | V2 |
|---|---|---|
| dispatch 시점 | optimizer.step 직후 stash + mb0_fwd 끝 forward hook 에서 submit | iter 의 **첫 동작** (forward 직전) 에 단일 dispatch |
| stash 변수 | `_armed_post_iter_fc_task` 필요 | 불요 |
| forward hook 등록 | `register_forward_hook` 필요 | 불요 |

**검증 의무**:
- iter time gap (vs baseline no-proposal) ≤ 10 ms.
- nsys trace 에서 forward 구간의 host-side `cudaStreamSynchronize` / `cudaDeviceSynchronize`
  가 proposal stream 완료를 기다리는 구간 **0**.
- 위반 시 deferral 로 되돌리지 말고, sync caller 를 특정해 위 원인 제거 checklist 를
  먼저 적용한다. fwd 직전 dispatch 가 동작하지 않으면 V2 구현 미완성으로 본다.
- `proposal_arm_post_iter_fc` / `proposal_release_post_iter_fc` 호출자 코드 (training.py)
  가 새 단일 entry `proposal_dispatch_post_iter` 로 치환됐는지, 그리고 그 호출이 forward
  **직전** 에 위치하는지 grep + 위치 확인.

### 1.0-prologue 5 stage 의 thread / stream 분포 — ✅ 완료

- **Stage 1–2**: `expert_launcher` 스레드의 단일 `compute_stream` 위에서 순차 실행 (현행 `tracker_stream` / `worker_stream` 가 host-side `.synchronize()` 로 이미 직렬화돼 있어 GPU overlap 이득 0 → 단일 stream 으로 병합).
- **Stage 3**: 같은 `expert_launcher` 스레드가 `d2h_stream` 으로 segment 단위 LFSY AOS D2H 와
  plan-target LFCL dense payload D2H 를 launch 한다. LFCL payload 는 GPU live `p,m,v` 에서
  직접 materialize 하며 CPU dense scatter 를 하지 않는다.
- **Stage 4**: `expert_writer` 스레드의 CPU 측 disk pwrite+fsync (`test_dram` 은 pwrite skip) +
  LFCL tmp write+fsync+rename+parent-fsync + LFSY rotate. GPU work 와 dense scatter 없음.

용어:
- **`delta_i²`** — Adam m,v 로부터
  `(lr/bc1 · m / (√v · √(1/bc2) + eps))²` 로 inline 계산한 현재 update 제곱.
- **`age2bit_i`** — actual written 기준으로 reset되는 2-bit age representative
  (`1, 4, 16, 64`).
- **`geom_lut`** — 매 iteration 작은 GPU LUT 로 갱신되는 Adam 감쇠 기반 누적계수.
  rho 는 LUT 갱신 kernel 안에서만 계산되고, per-element score kernel 은 rho 를 직접 계산하지 않는다.
- **`e_hat_i`** — 현재 Adam update 방향과 `geom_lut[age_code_i, H_lut_idx_i]` 로 만든
  마지막 write 이후 residual proxy.
- **production score policy** — default
  `max(0, -m_i*e_hat_i + 0.5*|H_i|*e_hat_i^2)`, ablation
  `|H_i|*e_hat_i^2`.
- **`τ = total_th`** — `|{i : score_i ≥ τ 또는 force_i}|` 가 `K_total` 의 허용 오차 안에 드는
  단일 임계값. `K_total = topk_ratio × N_total_elements_global`, 목표 오차는
  `±1~2%` (`ProposalConfig.cutoff_K_tolerance`).

### 1.0 Stage 0 — retired: per-iteration `v_q` refresh 없음 — ✅ 완료

| | |
|---|---|
| Stream | — |
| Thread | — |
| ΔVRAM persistent | 0 |
| ΔVRAM transient | 0 |

`v_q` 는 더 이상 매 iteration 또는 별도 주기로 갱신하지 않는다. Production hessian mode 의
고정 순서는 다음이다:

```
training halt
  -> drain proposal workers
  -> H refresh / h_q 갱신 완료
  -> v_q refresh 완료
  -> h_lookup_lut rebuild + refresh-boundary cutoff bridge 적용 완료
training resume
```

즉 `v_t -> H` quantile matching 은 H refresh boundary 에 종속된 후처리이며,
별도 v_q cadence knob 는 존재하지 않는다. 정상 iteration 의
selection/write kernel 은 마지막 anchor boundary 에서 만들어진 cached
`v_q + h_q + h_lookup_lut` 만 읽는다.

기본 알고리즘 (재귀 log-bin, sort 없음):
1. **Coarse log-bin pass**: v_t (양수 분포; Adam exp_avg_sq 정의상 ≥ 0) 의 양의 [v_min, v_max] 를
   N_LOG=1024 log-bin 으로 나눈 atomic histogram. v_min/v_max 는 reduction 한 번으로 데이터
   에서 결정 (v_min 은 양수 최소; 0 은 init 직후 외 거의 없으므로 drop).
2. **Quantile 추출**: cumsum → N_Q=4096 개 quantile 위치별로 cumsum 이 그 값에 닿는
   bin 찾기 → bin 내부 cumulative density 로 lerp → v_q[q].
3. tail 정밀도가 부족하면 (∃ bin 의 count > threshold) **그 bin 만 재귀**적으로
   B-등분 refine. 같은 카운터 buffer reuse, 추가 VRAM 0.

산출물: `v_q[0..N_Q-1]` (FP32, 16 KB persistent, in-place overwrite). N_Q=4096 의 근거는
기존 검증 기준과 동일 — 99.99% lognormal 분포 정확도 + persistent VRAM 32 KB (v_q + h_q
합산) ≈ 0.

`v_q` 또는 `h_q` 가 갱신된 직후에는 `h_lookup_lut` 를 재빌드한다. 이후 Stage 1/2 의
score kernel 은 `v_q` bsearch 를 하지 않고 log-domain LUT index + linear interpolation
으로 H̃ 를 얻는다. LUT 는 scale-independent (`8192 × FP32` = 32 KB) 라 per-element cache
가 아니다.

### 1.1 Stage 1 — `find_selection_cutoff` — ✅ final path

| | |
|---|---|
| Stream | `compute_stream` |
| Thread | `expert_launcher` |
| ΔVRAM persistent | static entry/range metadata + probe aggregate scratch (`sum_q`, `sum_q2`, `n_blocks`, candidate counts) |
| ΔVRAM transient | 0 per-iter allocation |

Production cutoff flow is fixed:

```
AdaptiveCutoffController tau ladder
  -> GPU-generated block-stratified approx probe
  -> GPU between-block variance/CI decision (`total_th`, `needs_exact`)
  -> device-guarded exact candidate fallback
  -> direct emission cursor feedback
```

삭제된 경로: sample table, host/numpy estimator, host τ chooser, recursive host histogram
cutoff, pre-fold validation/retry, per-tname selection count, pre-materialized selection
mask, and writer layout based on selected counts.

#### GPU-generated block-stratified probe

Init 단계에서만 static metadata 를 만든다: per BF16 group 의 entry-block range, stratum id,
stratum total elements. Iter 마다 sample list 를 만들거나 H2D 하지 않는다. Probe kernel 은
`(range_id, sample_id)` 에서 deterministic interval/hash arithmetic 으로 stratum-local
entry-block ordinal 을 생성한다.

For candidate τ and stratum `h`:

```
q_j       = selected_in_sampled_block_j / elements_in_block_j
mean_h    = avg(q_j)
s_h^2     = sample variance across sampled blocks
R_hat     = Σ_h W_h * mean_h
Var(Rhat) = Σ_h W_h^2 * s_h^2 / n_h
accept    iff |R_hat - rho| + 1.96 * sqrt(Var) <= rho * cutoff_K_tolerance
```

이 variance 는 between-block variance 이므로 group/cluster correlation, expert/layer 집중,
post-warmup hot/cold block 차이를 IID 로 가정하지 않고 실제 표본 분산으로 반영한다.

Decision kernel 은 host roundtrip 없이 `total_th` 와 `needs_exact` 를 device 에 쓴다.
Exact candidate fallback kernel 은 항상 enqueue 되며, 첫 줄에서 `needs_exact == 0` 이면
no-op 한다. Production τ 결정에 `.cpu()`, `.item()`, `.tolist()`, `stream.synchronize()` 를
사용하면 contract 위반이다.

#### Correctness vs tau quality

- `capacity-HARD`: direct emission 의 per-expert cursor capacity cap. 이것이 유일한
  overflow/corruption hard guarantee 다.
- `K-band`: controller/probe 가 맞추는 soft τ quality 목표다. CI/exact fallback 은 τ 를
  더 좋게 고르는 보조 수단이지 overflow correctness source 가 아니다.
- overflow 가 발생하면 write 하지 않고 overflow counter 만 증가시키며, 해당 update 는
  `dg` fold 경로로 남아 다음 iter 에 반영된다.

#### Bit identity

Probe, guarded exact fallback, direct emission, controller feedback 은 모두 같은 score helper 를
사용한다:

```
age2bit_i   = {1, 4, 16, 64}[age_code_i]
geom_i      = geom_lut[age_code_i, H_lut_idx(v_i)]
e_hat_i     = -delta_update_i * geom_i
score_i     = score_policy(m_i, |H(v_i)|, e_hat_i)
selected_i  = (score_i >= τ) or force_i
```

같은 τ 에서 test-only exact predicate compare 와 direct-emission cursor count 가 다르면
구현 오류다. Production 은 selection mask 를 materialize 하지 않는다.

### 1.2 Stage 2 — `write_lossy_records` — ✅ final path

| | |
|---|---|
| Stream | `compute_stream` |
| Thread | `expert_launcher` |
| ΔVRAM persistent | per-iter direct write counters (`cursor`, `overflow`; `int64 × n_experts`) + existing segment pipeline |
| ΔVRAM transient | segment staging — 기본 `125 MiB × 2` GPU staging buffer (~250 MB persistent). 단일 expert blob 이 125 MiB 를 넘으면 그 blob 이상으로 init 단계에서 키운다. |

Direct emission is the only production materialization path. Writer input is full
tname-span descriptors plus GPU-owned per-expert layout state; no per-name selected-count
layout is built.

- GPU kernel computes the same element-wise rho/e_hat Taylor score predicate inline.
- Each selected element reserves one cursor slot.
- `slot < capacity` records are written. Overflowing records are skipped and counted.
- Age reset is based on actual write outcome: written element resets to age code `00`;
  skipped/unselected/cap-dropped elements keep aging. There is no production group-dg fold.
- GPU layout state computes expert prefix offsets and capacities from current
  approximate-probe `capacity_demand_hint`. The first bytes of the AOS slab contain the
  minimal layout header: fixed metadata plus `offset[int64][n_experts]` in expert-index
  order.
- LFSY prefix `(count u32, step u32)` is GPU-stamped at each offset. CPU writer consumes
  the slab header, jumps to each prefix, and writes `count × record_bytes` payload. It does
  not recompute prefix/layout.
- Records are self-contained `(idx, p, m, v)`, so emission order is not a correctness
  contract.
- Expert slices are contiguous inside the single pinned slab. The total slab bytes are fixed
  at init. Per-expert record capacities are GPU persistent layout state and are redistributed
  from `max(current capacity_demand_hint, previous exact cursor demand)`; if both are
  invalid/all-zero, the layout fallback is expert-size-proportional. Resetting per-iter write
  counters before emission must never reset or equalize those capacities.

### 1.3 Stage 3 — `copy_records_and_lfcl_payload_to_pinned` (segment/LFCL D2H) — ✅ 완료

| | |
|---|---|
| Stream | `d2h_stream` (compute_stream 과 segment 단위 overlap; LFCL fired payload 도 같은 stream) |
| Thread | `expert_launcher` (launch 만), copy 는 GPU DMA |
| ΔVRAM persistent | `lossy_log_bytes_device[n_experts_or_chunks]` i64 + LFCL fired flag/id list (expert/chunk scale) |
| ΔVRAM transient | 0 (segment buffer 재사용; LFCL payload 는 init 시 잡은 single pinned batch slab 이 목적지) |

`SegmentD2H.run(producer)` 호출 — producer(`compute_stream`) ↔
copy(`d2h_stream`) 가 segment 단위로 overlap, back-pressure 는 `_d2h_done` event 로.
각 segment 의 GPU→pinned AOS copy 는 기본적으로 bounded chunk copy 로 enqueue 한다
(현 구현 기본 16 MiB, segment size 를 넘지 않음). 이는 CPU sleep/pacing 이 아니라
하나의 거대 D2H burst 를 여러 copy 명령으로 나누는 전송 primitive 이며, 정상 hot path 에
추가 allocation 이 없다.
LFSY AOS D2H 를 submit 한 뒤, 같은 expert worker tick 안에서 due LFCL fired targets 를
하나의 contiguous pinned LFCL batch slab 안 4 KiB-aligned slot 들에 pack 하도록 D2H 를
enqueue 한다. expert-ready 신호는 LFCL batch enqueue 이후에만 set 되며, optimizer fence 는
LFCL 이 fired 된 iter 에 LFCL batch D2H event 까지 기다린 뒤 다음 `p,m,v` mutate 를 시작한다. read-done fence 로
D2H 를 다음 forward/backward 와 overlap 시키는 실험은 copy contention 으로 iter time 이
악화되어 production 계약에서 제외한다.

LFCL plan 은 같은 expert worker tick 의 lossy layout 이후 결정한다. `round_robin` 은 host metadata 만으로 충분하고,
`lossy_log_size` 는 파일 시스템 `stat` 를 호출하지 않는다. Stage 2 layout 이 산출한
per-expert/per-chunk padded LFSY byte 수를 `lossy_log_bytes_device` 와 작은 pinned
mirror 에 함께 유지하고, planner 는 pinned mirror 를 읽는다. LFCL commit + LFSY rotate 가 durable 하게 끝난 target 은
다음 dispatch 초기에 device counter 와 host mirror 를 함께 0 으로 reset 한다.

plan-target LFCL payload 는 AOS record 를 CPU에서 decode/scatter 해서 만들지 않는다. GPU 가
해당 expert/chunk 의 live FP32 master/Adam `p,m,v` view 를 LFCL on-disk layout 순서로
single pinned LFCL batch slab 에 직접 D2H 한다. batch slab 내부 각 fired target 은
`HEADER_BLOCK_BYTES + payload` 형태의 4 KiB-aligned slot 을 가지며, writer 는 slot view 를
읽어 LFCL tmp file 로 write/fsync/rename 한다. 이는 plan target 에 대해서는 exact dense
baseline 이며, 이후 iter 의 expert state 는 다시 LFSY lossy records 로 갱신된다.

### 1.4 Stage 4 — `persist_records` (file pwrite+fsync only) — ✅ 완료

| | |
|---|---|
| Stream | — (CPU) |
| Thread | `expert_writer` |
| ΔVRAM persistent | 0 |
| ΔVRAM transient | 0 |

`AsyncWriter.submit_raw(persist_lossy_and_lfcl_from_pinned)` 로 `expert_writer` 큐에 넣는다.
이때 single `pinned_aos` slab 과 single pinned LFCL batch slab generation 은
**writer 완료까지 lease** 된다. `expert_launcher` 는 같은 slab 을 다시 덮어쓰기 전에
lease completion 을 확인한다. `expert_writer` 는 CPU dense scatter 를 수행하지 않고:

1. LFSY blob 을 disk mode 에서 `pwrite+fsync` 한다.
2. 같은 iter 에 LFCL payload 가 있으면 tmp LFCL 파일에 write → `fsync(fd)` → `rename` →
   parent-directory `fsync` 를 수행한다.
3. LFCL commit 이 durable 해진 뒤에만 해당 target 의 LFSY 를 rotate/unlink 하고, reset 대상
   counter 를 기록한다.

### 1.5 Per-iter 흐름 한 줄 요약 — ✅ 완료

```
expert_launcher thread / compute_stream:
  Stage 1  find_selection_cutoff     (cached h_lookup_lut 사용)
  Stage 2  write_lossy_records       (segment staging — _write_lossy_records_for_entries_kernel)
  Stage 3  copy_records_and_lfcl_payload_to_pinned
                                      (LFSY AOS D2H + fired LFCL dense payload D2H)
expert_writer thread:
  Stage 4  persist_records           (disk pwrite+fsync; test_dram 은 pwrite skip)
          + fired LFCL tmp write+fsync+rename+parent-fsync + LFSY rotate
```

순차 의존: anchor boundary 산출 v_q/h_q/h_lookup_lut → Stage 1 의 s_elem 계산 → Stage 1 산출 τ + counts →
Stage 2 의 byte layout → Stage 2 산출 segment → Stage 3 의 D2H → Stage 4 의 commit.
모두 의존적이므로 stream 분리의 GPU overlap 이득 없음 (이미 host-sync 직렬화). 단일
`compute_stream` 으로 충분.

---

## 2. Anchor refresh pipeline (5 stages)

anchor 는 `--proposal-hessian-refresh-every`(**V2 기본 ≥100**) iter 마다 한 번. H 분포는
학습 phase에 걸쳐 천천히 변하므로 (Var(quantile) ∝ 1/N 의 robustness 와 결합) 100+ 간격
이 충분하며, 짧은 간격(40 등)은 학습 overhead를 늘리는 대신 얻는 정확도 이득이 미미.

V2 에서 K-sample loop / `_anchor_h_ratio` rescale / `_g_fast_h_anchor_sample_buf`
(stride sample buffer) 는 **제거**된다. `_flat_signed_stream_kernel` 의 "stride sample 로
별도 sample buffer 에 쓰는 경로" 는 제거하지만, PCG z 를 register-only 로 재생성하는
현행 Triton 구현 패턴은 hook adapter / fallback 의 기준으로 재사용한다.
단, **multi-batch** 는 유지된다 — 한 anchor가 N 개 batch 데이터에 걸쳐
`(+εz, −εz)` 쌍을 반복하고, 각 pair 의 `z·(gp − gm)/(2ε·N)` 가
학습이 실제 gradient 를 accumulate 하는 storage/view 에 hook 으로 **in-place 누적**된다.
anchor 진입 전 기존 grad 는 zero/stash 되어야 하며, 별도 `grad_flat` GPU buffer 는
alloc 하지 않는다. multi-batch 는 routing coverage(≥0.93)와 분포 robustness 를 위해 필요
— single batch는 routing이 ~67% expert만 활성화해 cover되지 않은 expert의 H가 단일
batch noise 에 노출된다. 모든 batch 처리 후 같은 grad storage 에서 바로 h_q 가 build된다.

`N = N_anchor_batches` 는 `init_proposal` 시 `global_batch_size // micro_batch_size // 2`
로 자동 산출 (현행 동작 유지).

### 2.0 Stage 0 — `drain_workers` (training halt 진입점) — ✅ 완료

| | |
|---|---|
| Stream | — |
| Thread | training thread (호출), expert_launcher / expert_writer / ne_writer (drain) |
| ΔVRAM | 0 |

- `_writer.drain_non_expert()`, `_writer.drain_expert()`, `_writer.drain_raw()` 호출.
- training thread 가 anchor 진입점에서 동기적으로 호출 (forward/backward 없음).
- 이후 모든 anchor stage 는 training thread 가 직접 실행.

### 2.1 Stage 1 — `snapshot_training_state` (training state 보존) — ✅ 완료

| | |
|---|---|
| Stream | `d2h_stream` |
| Thread | training thread |
| ΔVRAM | 0 (전부 pinned host) |

**책임 통합 — old 3-갈래(`snapshot_training_state` + `offload_adam_state_to_cpu` +
inline DS hook detach) 분산을 anchor 진입점 하나로 합친다.** `fp32_state.snapshot_training_state`
는 RNG/buffers/grad stash 를 맡고, `h_refresh` anchor body 는 DS hook detach/restore 와
DS attrs snapshot/restore 를 같은 try/finally 안에서 처리한다.

#### 1) bit-exact 보존해야 할 항목만 snapshot (over-snapshot 금지)

| 항목 | anchor가 수정? | snapshot 필요? |
|---|---|---|
| expert `p.data` (BF16 lp) | ✓ (perturb) | **필수** |
| expert FP32 master | ✗ by default | 불필요. perturb 가 master 를 직접 mutate 하는 optional 구현에서만 필수 |
| expert Adam state `(m, v)` | ✗ (optimizer.step 없음) | **불필요** (old path 의 offload 는 VRAM용이었고 V2 정확성에는 불필요) |
| expert `lp.grad` / grad accumulation storage | ✓ (backward) | 기존 값은 anchor 진입 전 zero/stash. Stage 4에서 zero 후 학습 재개 |
| non-expert `p / master / m / v` | ✗ | 불필요 |
| **`fp32_groups_gradients_flat` (DS 내부)** | ✓ (hook fire 시) | **필수** (clone-stash) |
| **`_grad_acc_hooks` (DS)** | ✗ (등록 자체는 그대로지만 fire가 오염원) | **detach + restore handle 반환** |
| CPU + CUDA RNG state | ✓ (forward의 random op) | **필수** |
| Model buffers (BN running stats 등) | ✓ (train-mode forward) | **필수** |
| DS engine attrs (micro_steps, global_steps, loss_scale, cur_iter, cur_scale) | ✓ | **필수** |

#### 2) DS hook detach + `fp32_groups_gradients_flat` clone-stash

old path 에서는 이 책임이 여러 위치에 분산돼 있었다. V2 는 anchor 진입점에서
`fp32_state.snapshot_training_state()` 를 호출해 RNG/buffers/flat grad 를 stash 하고,
곧이어 DS `_grad_acc_hooks` detach 와 DS attrs snapshot 을 수행한다. anchor finally 에서
hook 재등록, DS attrs restore, `fp32_state.restore_training_state()` 를 순서대로 수행한다.

#### 3) VRAM swap (offload) 는 별개 책임 — V2 는 원칙적으로 생략

m, v 는 anchor 가 수정하지 않으므로 정확성 차원의 snapshot 은 불필요.
현행이 m, v 까지 CPU 로 offload 하는 것은 ±εz forward 의 activation + 학습 grad storage 가
billion-scale Adam state 와 같이 못 들어가는 **VRAM 압력 해소용**이다. V2 의 hook trick
은 anchor 의 VRAM peak = 학습 평소 peak (학습 grad storage 재사용) 라 **offload 자체가
불필요**. 단, multi-batch 가 batch 당 큰 activation 을 만드는 모델에서는 재검증 필요 →
optional path 로 유지하되 default off.

#### 4) Fragmentation 회피

snapshot 대상이 작은 것들 (RNG ~수십 KB, model buffers ~MB, DS attrs bytes,
fp32_grads_flat ~수십 MB) 뿐이고 모두 **pinned-resident pool** 에 안정 키로 재사용
(`fp32_state.pinned_snapshot` 의 key-stable pool 방식). m, v offload 자체를 생략하므로 매 anchor
마다 GPU caching allocator 를 거의 안 만지게 됨 → V1 에서 측정된 reserved-allocated waste 누적이 사라짐.

### 2.2 Stage 2 — `accumulate_fd_h_into_grad_flat` (multi-batch ±εz loop) — ✅ 완료 (default fp16_zero0 가 mantissa 손실 가능성 존재하고 확인 필요)

| | |
|---|---|
| Stream | `main_stream` (학습 forward/backward 와 동일) |
| Thread | training thread |
| ΔVRAM persistent | 0 |
| ΔVRAM transient | 학습의 forward 활성화 + lp.grad transient — anchor 추가 **0** |

**핵심 트릭**: `grad_flat` (= 학습용 grad accumulation storage/view) 에 hook 으로
`±z/(2ε·N)` scaling 을 적용해 **새 GPU 버퍼 0** 으로 모든 batch 의
`z·(gp − gm)/(2ε)` 를 누적한다.

**Accumulation 위치 원칙**: anchor 의 FD Hessian 추정은 별도 H buffer 에 쓰지 않고,
해당 training configuration 이 실제 backward gradient 를 accumulate 하는 바로 그 flat
grad storage/view 에 누적한다. gradient accumulation microbatch, ZeRO/BF16 optimizer flat
layout, DDP reducer bucket 등 설정이 달라도 기준은 동일하다: autograd 가 원래 `grad`
를 더할 위치를 anchor 동안 비워 두고, hook/adapter 가 raw grad 를
`±z/(2ε·N)` 로 변환한 값만 그 위치에 더한다. 따라서 multi-batch anchor 는 학습의
gradient accumulation 의미를 그대로 재사용하되, 누적되는 값의 의미만
`Σ grad` 가 아니라 `Σ z·(gp−gm)/(2ε·N)` 가 된다.

전제 조건은 명확하다. anchor 는 학습 forward/backward 와 interleave 하지 않고 training
halt 중 실행하며, 진입 전 기존 grad 는 zero/stash 되어야 한다. loss scaling, no_sync,
ZeRO shard ownership, reducer hook 은 현재 학습 설정의 backward 경로와 같은 규칙을
따르되, optimizer.step 은 실행하지 않는다.

```
# 사전: expert.grad 는 학습이 실제 사용하는 grad accumulation storage/view 를 가리킴.
#       anchor 진입 전 grad_flat.zero_() 또는 동등한 zero/stash 완료.
# z 는 element global_off + i 의 PCG hash (register-only, z 버퍼 0)

for b in range(N_anchor_batches):

    # ─── (+εz phase) ─────────────────────────────────────────────
    restore θ_0 from pinned
    perturb in-place θ ← θ_0 + ε·z
    register per-expert backward hooks:  g  ↦  + z/(2ε·N) · g
    forward(batch_b)  →  backward()
    # hook 이 resident FP32 grad view 에 +z·gp/(2ε·N) 를 side-effect accumulate.
    # raw p.grad 는 hook 제거 직후 버린다.
    remove hooks

    # ─── (−εz phase) ─────────────────────────────────────────────
    restore θ_0 from pinned
    perturb in-place θ ← θ_0 − ε·z
    register per-expert backward hooks:  g  ↦  − z/(2ε·N) · g
    forward(batch_b)  →  backward()
    # hook 이 resident FP32 grad view 에 −z·gm/(2ε·N) 를 side-effect accumulate.
    # raw p.grad 는 hook 제거 직후 버린다.
    remove hooks

# 종료: grad_accum_views[i] = (1/N)·Σ_b  z[i] · (gp_b[i] − gm_b[i]) / (2ε)
#                          = batch-averaged FD H 추정 per element (signed).
```

**왜 hook 인가**: backward 가 raw `gm` 을 resident FP32 grad storage 에 그대로 누적하면
`z` 스케일이 빠진 오염 항이 섞인다. hook 은 raw grad 를 입력으로 받아 custom Triton
kernel 로 `±z/(2ε·N)` 스케일 값을 resident FP32 grad view 에 side-effect 누적한다.
`z` 는 element-index hash 로 register-only 재생성 → **추가 GPU 버퍼 0**.

**구현 조건**: hook 은 scaled grad tensor 를 새로 allocate 하지 않는다. 현재 V2 구현은
`_accumulate_grad_to_fp32_with_pcg_z` 로 resident target 에 누적하고 raw grad tensor 를
그대로 반환한다. 이 raw grad 는 임시 `p.grad` 로만 들어가며 hook 제거 직후 `p.grad = None`
처리되고, 저장해 둔 학습 grad state 는 `restore_training_state()` 로 복원된다. 즉 selection 에
사용되는 H 값의 단일 source 는 `grad_accum_views` 이다. `None` 반환은 raw grad 제어를
명시하지 못하므로 금지한다.

**Routing 일관성**: 한 batch 의 `(+εz, −εz)` 쌍은 **동일한 routing**(같은 expert 활성화)
을 사용해야 H 추정이 의미를 가진다. `RoutingFreezeContext` 로 한 batch 의 첫 forward
에서 routing 을 capture 하고 두 phase 의 forward 가 동일 dispatch 를 재생.

**Multi-batch 필요성**: single batch 는 routing 이 ~67 % expert 만 활성화 → 비활성
expert 의 H 가 단일 batch noise 에 노출. N batch 평균 시 coverage ≥ 0.93, H 분포
quantile 의 robustness 가 추가로 보장.

### 2.3 Stage 3 — `refresh_h_quantile_from_grad_flat` — ✅ 완료

| | |
|---|---|
| Stream | `compute_stream` |
| Thread | training thread (anchor halt 중이라 expert_launcher 도 가능; 단순화를 위해 training thread) |
| ΔVRAM persistent | 0 (덮어쓰기) |
| ΔVRAM transient | hist 카운터 (N_LOG × i64 = 8 KB; Stage 1 의 hist scratch 와 동일 buffer 공유) |

입력: `grad_accum_views` (학습 grad accumulation storage/view 에 누적된 per-element signed FD H 추정).

알고리즘 — anchor-boundary v_q build / Stage 1.1 과 동일 골격:
1. `|grad_flat[i]|` 의 reduction 으로 `(h_min, h_max)` 산출 (`h_min` 은 양수 최소;
   H 는 양 곡률 가정, 0/음수는 FD noise 로 보고 drop_nonpositive).
2. **log-bin coarse pass** (H 분포는 lognormal 에 가까운 heavy-tail 이 학습 전 phase
   에서 안정적 — score 와 달리 log-bin 안전).
3. N_Q=4096 quantile 위치 별로 cumsum → bin → bin 내부 cumulative density 로 lerp →
   `h_q[q]`.
4. tail 정밀도 부족 시 그 bin 범위만 sub-bin 재귀 refine (Stage 1.1 와 동일 카운터
   reuse).

산출물: `h_q` (FP32, `N_Q × 4 bytes` = 16 KB persistent, in-place overwrite).

> V1 의 `_g_fast_h_anchor_sample_buf` 는 **삭제**. h_q 는 grad_flat 전수에서 추출 —
> 학습용 buffer 재사용으로 anchor sample buffer 자체가 불요.
> `_quantile_table_from_flat` 의 sort 경로도 **삭제** (재귀 log-bin 으로 대체).

### 2.4 Stage 4 — `restore_training_state` (학습 state 정확 역복원) — ✅ 완료

| | |
|---|---|
| Stream | `d2h_stream` (pinned → device H2D) |
| Thread | training thread |
| ΔVRAM | 0 |

복원 대상 = Stage 1 snapshot 대상의 정확한 역연산. **m, v 는 미수정이라 복원도 불필요.**

- expert `p.data` ← pinned θ_0 (per-expert `copy_`). FP32 master 는 optional master-perturb 구현에서만 복원.
- `fp32_groups_gradients_flat` ← pinned clone-stash (DS flat 정상화).
- `_grad_acc_hooks` 재등록 (Stage 1 의 restore handle 호출).
- CPU + CUDA RNG state restore.
- Model buffers restore.
- DS engine attrs restore (micro_steps 등).
- `grad_flat.zero_()` 또는 학습 설정의 동등한 grad clear (다음 학습 iter 의 grad 시작).

종료 후 학습 재개. **`_anchor_h_ratio` / `_anchor_just_fired` / `ANCHOR_FIX` 임계값
rescale 로직 전부 삭제** — V2 의 threshold 는 Stage 1.1 `find_selection_cutoff` 가 매 iter 같은 분포에서
다시 뽑으므로 scale shift 가 selection 에 안 보임.

#### Anchor stage 간 cross-stream sync 정책

V2 anchor 는 **각 stage 전환점에서 `torch.cuda.synchronize()` 한 번** 으로 단순화. 사용
가능 이유:

- anchor 동안 training 이 halt — main_stream 위 학습 작업 없음 → host sync 가
  `main_stream blocking = 0` contract 와 무관 (training thread 가 anchor 진입점에서
  의도적 sync 시작).
- 각 stage 가 서로 다른 stream 사용 (Stage 2 = main_stream, Stage 3 = compute_stream,
  Stage 4 = d2h_stream) — explicit cuda.Event + stream.wait_event 시퀀스 대신
  host sync 한 줄로 cross-stream race 차단.
- per-iter lossy pipeline 의 event-based 시퀀스는 training 과 concurrent 라
  필요했으나, anchor 는 training halt 중이라 다름.

요구되는 sync 지점:
1. **Stage 2 끝 → Stage 3 시작**: `torch.cuda.synchronize()` — backward 가 main_stream
   에 enqueue 한 hook-scaled write 완료를 보장한 후 compute_stream 의 hist build.
2. **Stage 3 끝 → Stage 4 시작**: `torch.cuda.synchronize()` — compute_stream 의 hist
   읽기/h_q write 완료 보장 후 d2h_stream 의 H2D + grad clear.

(Stage 1 → Stage 2 는 backward 진입 직전 PyTorch 가 자체 sync.)

### 2.5 Anchor 흐름 한 줄 요약 — ✅ 완료

```
training thread (halt 중):
  Stage 0  drain_workers                    drain all queues
  Stage 1  snapshot_training_state          θ, RNG, buffers, DS attrs,
                                             fp32_grads_flat clone, hook detach
  Stage 2  accumulate_fd_h_into_grad_flat   for b in N: ±εz fwd+bwd + RoutingFreezeContext,
                                             hook trick → grad_flat
                                             grad_flat = avg_b  z·(gp_b − gm_b)/(2ε)  (+0 VRAM)
  Stage 3  refresh_h_quantile_from_grad_flat   h_q ← log-bin (+ sub-bin refine) of |grad_flat|
  Stage 4  restore_training_state           hook 재등록, grad_flat.zero_()
```

---

## 13. File lifecycle + fsync 정책

### 13.2 매 iter lossy log append + 블록 layout — ✅ 완료

```
expert_writer thread (per expert per iter):
  write(ent.lossy_fd, pinned_blob_view, blob_padded_bytes)   # disk mode only
  fsync(ent.lossy_fd)                                        # disk mode default
  # blob 은 [count u32][step u32][N × 20-byte record] padded to 4 KiB

if LFCL plan fired this expert/chunk:
  # pinned_lfcl_payload 는 expert_launcher 가 GPU live p,m,v 에서 D2H 완료한 dense baseline
  write(tmp_lfcl_fd, header + pinned_lfcl_payload)
  fsync(tmp_lfcl_fd)
  rename(tmp_lfcl_path, ent.lossless_path)
  fsync(parent_dir)
  unlink/rotate replayed LFSY only after durable LFCL commit
  mark lossy_log_bytes counter reset for next dispatch
```

fsync 정책: **LFSY append 는 기본적으로 매 append 후 `fsync(lossy_fd)` 한다.**
따라서 disk mode 에서 fsync 가 성공한 LFSY record 는 power loss 이후에도 recovery 입력으로
간주한다. 성능 실험용으로 fsync 를 batch 하거나 끄는 구성은 별도 unsafe 옵션으로만 허용하며,
그 경우 RPO 는 마지막 성공한 fsync 시점으로 후퇴한다.

#### 블록 내부 record 순서 — per-expert flat 채택 (V1 의 per-tname grouping 폐기)

한 expert 블록 안 record 는 tname 그룹 없이 **emission 순서로 일렬 배치**:

```
한 expert 의 한 iter 블록:
  [count u32][step u32]
  [record_0][record_1]...[record_{N-1}]
  각 record 의 idx_lo/idx_hi 가 인코딩한 64-bit 값 = per-expert local element index
  (= 그 element 가 expert 의 dense layout 에서 차지하는 위치, 0 .. n_elements_in_expert − 1)
```

per-expert local 인 이유: recovery 는 expert 별 dense buffer (`size = n_elements_in_expert`)
에 직접 indexing (`dense[record.idx]`, §14.1). global idx 면 expert 별 dense 와 mapping
이 추가 필요. tname 별 within-expert offset (`name_t_off_elem[tname_idx]`) + intra-tname
position 으로 산출 (모두 element 단위; byte 환산은 disk byte offset 이라 별도).

emit kernel 은 `slot_per_expert[e]` atomic_inc 로 슬롯 획득 후
`aos_byte_off_per_expert[e] + PREFIX_BYTES + slot × RECORD_BYTES` 위치에 write. 두 자료만
필요 (per-tname byte/slot 자료 폐기). recovery 와의 호환성은 §14 참조.

#### LFCL plan state — filesystem `stat` 금지

`LossyLogSizeBasedPlan` 은 매 iter filesystem `stat()` 로 LFSY 크기를 읽지 않는다.
Stage 2 layout 이 계산한 실제 padded LFSY byte 수를 `lossy_log_bytes_device[target]` 에
누적한다. target 은 초기 구현에서 expert 단위이며, recovery overhead bound 를 위해 chunk
단위 plan 으로 확장 가능해야 한다. planner 는 해당 counter 의 작은 pinned mirror 를 읽어
선택한다.

필요한 추가 VRAM 은 `int64 × n_targets` + reset/fire flags 정도이며 element 수가 아니라
expert/chunk 수에 비례한다. 이는 R1/R2 위반이 아니지만 audit breakdown 에 별도 항목으로
기록한다. durable LFCL commit 후 LFSY rotate 가 끝난 target 만 다음 dispatch 에서 device
counter 와 mirror 를 함께 reset 한다.

## 14. Recovery 경로

production fault (실제 power loss): VRAM / DRAM 휘발 → **disk mode** 의 fsync 완료된
LFCL baseline + LFSY 만 유효. LFCL baseline 은 plan 이 고른 expert/chunk 의 GPU live
`p,m,v` 에서 materialize 된 dense payload 이다. simulated fault (실험 측정용):
**test_dram** target 은 disk write/fsync 를 생략할 수 있으므로 durability correctness 주장
대상이 아니다. 대신 pinned dense expert recovery image 를 최신 in-memory 상태로 직접
갱신하므로 recovery 시 LFSY replay 를 수행하지 않는다.

아래 §14.1–14.3 은 **disk mode 의 recovery** 만 다룬다.

### 14.1 per-expert 병렬 read + backward early termination — ✅ 완료

`ThreadPoolExecutor(n_workers = min(32, num_local_experts))` 로 expert 단위 GIL-free
병렬. 각 thread 가 자기 expert 의 LFSY 를 **뒤에서부터** 읽어 dedup:

핵심 — dense LFCL baseline 을 먼저 읽고, 해당 baseline 이후의 LFSY 를 **뒤에서부터**
읽어 latest-wins 로 dedup 한다. expert 단위 baseline 에서는
`target = n_elements_in_expert` 이고, chunk 단위 baseline 에서는
`target = n_elements_in_chunk` 이다.

종료 조건이 트리거되지 않는 경우 (일부 element 가 LFCL 이후 한 번도 선택 안 됐을 때):
- 그 element 는 LFCL compaction baseline 값으로 남는다. 이는 최신 실제값이라는 뜻이
  아니라, V2 lossy recovery image 의 baseline 이다.
- read 가 LFSY 끝까지 도달 → fall back. schema / corruption 측면의 문제는 없지만,
  unselected element 의 residual lossy gap 은 설계상 남는다.

전형적 학습 시나리오 (topk_ratio × LFCL_period ≫ 1) 에선 대부분 early termination 트리거.

대규모 모델에서 replay 비용을 bounded 로 유지하려면 full expert 단위만으로는 부족할 수
있다. 권장 확장 방향은 fixed-size chunk LFCL 이다:

- LFSY block header 에 `expert_id`, `chunk_id`, record count, payload byte offset/length 를
  넣어 recovery 가 관련 chunk 의 block 만 읽게 한다.
- `lossy_log_bytes_device` 도 expert 단위가 아니라 chunk 단위 counter 로 확장 가능하게 둔다.
- plan 은 round-robin expert 뿐 아니라 `lossy_log_size` / dirty-rate 기반 chunk 선택을
  지원한다.
- 각 chunk 는 `max_replay_bytes_per_chunk` watermark 를 넘기기 전에 LFCL materialization
  대상이 되어야 한다.

element 별 "최신 log 위치" 를 steady-state 에서 정확히 유지하는 것은 기본 금지한다. 이는
`n_elements × offset` 크기의 dense index 와 selected-record 당 random update 를 요구하므로,
CPU dense scatter 를 dense index scatter 로 되살리거나 VRAM 에서 R1 을 위반할 위험이 크다.
대신 recovery 시점에 chunk-local dedup flags 을 임시로 만들고 backward scan 으로 latest-wins 를
구성한다. steady-state 에 유지하는 것은 chunk/block 수준 metadata 와 byte counter 까지로
제한한다.

### 14.2 GPU apply 단계 — ✅ 완료

per-expert recovered dense buffer → GPU H2D. **per-expert CUDA stream pool** (~8-way) 으로 동시
적용해 PCIe / NVLink bandwidth saturate. V1 의 sequential apply 대비 단순 변경.

## 17. 모듈 책임 분리 상태

`cp_hooks.py` 는 삭제됐다. V2 의 training-loop 접점은 `hooks.py` 의 얇은 wrapper,
runtime/facade 는 `controller.py`, H anchor 는 `refresh/hessian_refresh.py`, 진단은
`telemetry/diagnostics.py`, FP32 optimizer-state mapping 은 `refresh/fp32_state.py` 가 맡는다.
cutoff, emission, refresh, storage 는 subpackage 로 분리하며 외부 training loop 는
`hooks.py` / `controller.py` 이외의 내부 구현 파일을 직접 import 하지 않는다.

| V2 모듈 | 현재 책임 |
|---|---|
| **`hooks.py`** | 외부 training loop 가 부르는 public wrapper. 실제 상태를 소유하지 않음. |
| **`controller.py`** | `ProposalController` facade, worker dispatch/fence, FC/expert lossy owner wiring. |
| **`checkpoint/lossy_checkpoint_engine.py`** | 한 lossy checkpoint iteration 의 순서 배선. cutoff/emission/storage 세부 구현을 직접 소유하지 않도록 축소 대상. |
| **`cutoff/*`** | adaptive τ controller, cold-start cutoff, GPU approx probe kernels. |
| **`emission/*`** | AOS direct emission, GPU-owned layout, segment D2H, record kernels. |
| **`refresh/*`** | multi-batch ±εz H-quantile anchor, v/H quantile table, routing freeze, FP32 state mapper. |
| **`storage/*`** | expert/non-expert writer, LFCL plan/compaction, recovery, binary I/O. |
| **`telemetry/*`** | VRAM audit, fault/loss/FD/GPU-memory diagnostics, removable experiment telemetry. |

허용되는 module-level mutable state 는 facade singleton 과 FP32 mapper singleton 뿐이다:
- `controller.py`: `_CONTROLLER` — proposal runtime owner.
- `refresh/fp32_state.py`: `_MAPPER` — training optimizer layout mapper.
- `refresh/fp32_state.py`: `_PINNED_POOL` — key-stable pinned snapshot pool, anchor 첫 호출 이후 재사용.

현재 public hook:
- `init_proposal`
- `proposal_dispatch_post_iter`
- `proposal_fence_before_optimizer`
- `proposal_record_loss`
- `proposal_refresh_h_quantile`
- `proposal_simulate_fault`
- `proposal_recover`
- `proposal_shutdown`

### 17.6 V2 의 "facade 만 외부 노출" 원칙 — ✅ 완료

`__init__.py` 가 export 하는 public symbol:

`ProposalController` 외의 checkpoint engine / cutoff / emission / storage 구현은 internal — 모듈
경계가 명확해 짐.

### 17.7 expert_launcher / expert_writer 의 책임 변경 (V1 → V2 — 명시) — ✅ 완료

| | V1 | V2 |
|---|---|---|
| **expert_launcher** | scoring 만, segment D2H 는 forward | Stages 0–3 모두 (scoring + SegmentD2H + LFSY AOS D2H + fired LFCL dense payload D2H). LFCL plan target 의 GPU materialization 을 소유한다. |
| **expert_writer** | segment D2H (GPU AOS kernel 호출 포함) + lossy 파일 write | Stage 4 — pinned LFSY / pinned LFCL payload 를 파일에 write+fsync 하고 durable commit 후 LFSY rotate. CPU dense recovery image scatter 없음. GPU work 없음. |

V1 의 `expert_state.lossy_fc_append_batch` 가 GPU AOS emission 을 `submit_raw` →
expert_writer 큐에 넘기는 구조였으나, V2 에서는 **expert_launcher 가 step() 안에서
`AosEmitter.run()` 까지 동기 실행** → `final_event.synchronize()` 로 pinned 도착 보장 →
**그 다음에야** per-expert `append_lossy_log_from_pinned(fd, blob_view)` 를 expert_writer
큐에 submit. LFCL plan 이 발화한 iter 에는 같은 expert_launcher 가 fired target 의 live
`p,m,v` 를 single pinned LFCL batch slab 으로 D2H 한다. 이때 `pinned_aos` slab /
`pinned_lfcl_batch` generation ownership 은 expert_writer task 가 완료될 때까지 writer 에 있다.
expert_launcher 는 같은 slab/generation 을 덮어쓰기 전에 completion 을 확인한다.
expert_writer 는 그 task 안에서:

1. disk mode 에서 selected records blob 을 LFSY 에 pwrite+fsync 한다.
2. iter_T 에 LFCL fired payload 가 있으면 tmp LFCL write → fsync → rename →
   parent-directory fsync 를 수행한다.
3. durable LFCL commit 이후에만 해당 LFSY 를 rotate 하고, `lossy_log_bytes` reset 을 예약한다.

결과:

- expert_writer 가 GPU context 없이 순수 CPU thread 로 단순화 → `torch.cuda.set_device` 불요, GPU stream 관리 불요.
- LFCL 의 planning / GPU materialization 은 expert_launcher 소유, 파일 durable commit 은
  expert_writer 소유로 분리된다.
- 데이터 race 표면 ↓ (GPU 가 launch 되는 thread = expert_launcher 하나뿐).
- expert_writer 는 GPU kernel 을 직접 launch 하지 않는다. 단 `pinned_aos` ring /
  `pinned_lfcl` lease 때문에 writer backlog 가 ring capacity 를 초과하면 expert_launcher 가
  다음 overwrite 전에 대기할 수 있으므로, writer latency 는 진단 지표로 추적한다.

→ non-expert FC 도 같은 패턴: ne_writer 는 D2H 와 CPU 측 write 만, 별도 GPU
kernel launch 없음.

---

## 20. 성능 목표 메모 — adaptive τ, v_q reuse, cp_target

V2 에서는 이전 skip-write 식 실행 구분과 그 경로를 제거한다. checkpoint target 은
오직 `ProposalConfig.cp_target` 으로 구분한다.

| target | 의미 |
|---|---|
| `cp_target=disk` | LFSY / LFCL / non-expert FC 를 disk 에 write+fsync 한다. production durability path. resident host dense recovery image 를 매 iter 갱신하지 않는다. |
| `cp_target=test_dram` | 테스트 전용 target. 동일한 GPU selection/count/AOS/D2H/LFCL-materialization 경로를 수행하되, non-expert FC 와 expert recovery image 를 host RAM 에 보존하고 filesystem durability 를 생략한다. 목적은 빠른 write/recovery 로 proposal 정확도를 확인하는 것이다. |


### 20.1 Final-path-only implementation status

현재 production 은 다음 흐름만 유지한다:

```
AdaptiveCutoffController tau ladder
  -> GPU-generated block-stratified approx probe
  -> GPU between-block variance/CI tau decision
  -> device-guarded exact candidate fallback
  -> direct emission: selection predicate + dg fold + record emit + cursor count
```

삭제된 항목:

- host sampling planner / host estimator / host tau chooser
- production sample-index tables and per-iter sample H2D
- exact pre-fold validation and retry loops
- per-tname selected-count materialization for writer layout
- production selection masks and mask-based writer
- recursive host histogram cutoff as production τ source
- dispatch/selection debug env gates that inserted synchronization into timing paths

남은 hard guarantee 는 direct emission cursor capacity cap 이다. Strict K-band 는 soft τ
quality 목표이며, realized cursor count / overflow diagnostics 로 다음 iter controller 를
교정한다.
CI 는 strict quality certificate 와 diagnostics 이다. Sample size 는 CI shortfall 만으로
증가하지 않고, direct-emission actual cursor count 와 probe expected count 가 tolerance 밖으로
반복해서 어긋날 때만 증가한다. finite positive approximate τ 가 있으면 exact full scan 으로 되돌아가지 않으며, device-guarded exact
fallback 은 NaN / no positive survival / no valid τ 같은 clueless 상태에서만 실행된다.

성능 방향은 두 가지로 고정한다:

1. Approx probe 는 sample work 자체가 아니라 host decision/sync 없이 device 에서 끝나야 한다.
2. Direct emission 이 selection/fold/write 를 하나로 합쳐 writer/layout 재스캔을 제거해야 한다.

### 20.2 Adaptive cutoff 개선 방향

exact cutoff 는 active production score 의 global τ 를 min/max + log/refine
histogram 으로 구한다. 이전 full-scan exact cutoff 는
매 iter 사용하기에는 비싸다. 하지만 현재 adaptive controller 의 regression / `τ *= R/ρ`
계열 보정은 `R(τ)=P(score≥τ or force)` 의 log-log tail slope 를 α=1 로 가정하므로, 실제 α가 크거나
drift 가 있는 warmup/refresh 직후에 ratio 가 크게 흔들린다.
아래 계약이 adaptive cutoff 의 현재 단일 문서 기준이다.

현재 adaptive cutoff 계약:
- AdaptiveCutoffController 는 τ ladder 중심값만 제공한다.
- GPU approximate probe 는 small block-stratified sample 로 τ quality 를 빠르게 판별한다.
- finite positive approximate τ 가 있으면 production 은 exact full count 로 되돌아가지 않는다.
- CI/variance 는 strict certificate 와 diagnostics 이며, overflow correctness source 가 아니다.
- Sample size 증가는 CI shortfall 이 아니라 direct-emission actual cursor count 와 probe expected count 의 반복 mismatch 로 결정한다.
- direct emission cursor cap 이 hard guarantee 이고, realized cursor count / overflow feedback 이 다음 iter τ 를 교정한다.
- device-guarded exact fallback 은 no finite τ / no positive survival / NaN 같은 clueless 상태에서만 active 된다.
- H refresh 는 score remap 이 완전한 scalar 라고 가정하지 않는다. 현재 구현은 refresh 직전
  direct cursor feedback 을 먼저 consume 하고, H/v quantile 과 `h_lookup_lut` 를 갱신한 뒤
  median log-ratio 기반 bridge scale 을 AdaptiveCutoffController 에 적용한다. cold-start
  histogram bootstrap 은 initial step 에만 사용하며, refresh 후마다 강제하지 않는다.
  H refresh body 자체는 approx probe, direct-emission layout, AOS materialization 을 수행하지
  않는다. 따라서 refresh 직후 1~2 iter 의 τ/layout transient 는 가능한 품질 현상으로 보되,
  hard correctness 는 direct emission cursor/capacity guard 가 맡는다.

### 20.3 v_q 재사용

`v_t -> H` quantile matching 에 필요한 `v_q` 는 H refresh 가 끝난 직후에만 갱신한다.
별도 cadence knob 는 없다. H refresh boundary 에서 `h_q` 를 먼저 갱신하고, 이어서 현재
live Adam `v_t` 로 `v_q` 를 갱신한 뒤 `h_lookup_lut` 와 cutoff controller 보정을 끝내고
학습을 재개한다. 정상 iteration 은 cached LUT 를 사용하므로 per-iteration `v_t` scan 이
없다.

### 20.4 남은 overhead 분석 방향

기본 판단은 `cp_target=disk` production path 로 한다. `cp_target=test_dram` 은 같은 GPU
probe/direct-emission/D2H/LFCL-materialization 경로에서 filesystem write+fsync 를 제외하고,
host RAM recovery image 로 빠른 proposal 정확도 검증을 수행하기 위한 테스트 path 이다.

필수 분해 항목:

1. GPU approx probe kernel time and decision kernel time.
2. Guarded exact fallback no-op / active time.
3. Direct emission kernel time, cursor count, overflow count.
4. AOS bounded D2H and LFCL plan-target D2H.
5. Main fence wait before optimizer.

성능 검증은 write-off + D2H on 으로 수행한다:
`--proposal-test-skip-expert-commit --proposal-test-skip-nonexpert-commit` 을 사용하고,
D2H skip 플래그는 사용하지 않는다. 이 모드는 filesystem commit 만 생략하며, GPU probe,
direct emission, AOS materialization, D2H 는 production 과 동일하게 수행한다.

No-proposal baseline 은 반복 측정하지 않는다. 환경/모델/스크립트가 바뀐 경우에만 1회
재측정하고, proposal run 은 같은 설정에서 두 번 성공해야 steady overhead 판정을 내린다.

현재 final-path 측정(125M-48E, iter_time_debug 동일 설정):

| run | window | mean / p50 ms | 해석 |
|---|---|---:|---|
| baseline recheck | iter 20--80 | 1722.1 / 1694.7 | no proposal 기준선 |
| write-off + D2H on, 128 MiB segment | iter 20--40 | 1866.6 / 1898.2 | `segments=9`, AOS enqueue mean/p50 `53.0/53.7 ms`, lossy launch p50 `71.7 ms` |
| write-off + D2H on, 384 MiB segment | iter 20--40 | 1836.6 / 1823.2 | `segments=3`, AOS enqueue mean/p50 `19.7/23.6 ms`, lossy launch p50 `44.7 ms` |
| write-off + D2H on, 512 MiB segment | iter 20--40 | 1851.9 / 1851.9 | `segments=3`, AOS enqueue mean/p50 `22.5/22.8 ms`, VRAM cost higher than 384 MiB |
| write-off + D2H on, 1100 MiB segment | iter 20--40 | 1831.5 / 1851.3 | `segments=1`, AOS enqueue mean/p50 `17.6/17.3 ms`, best enqueue but ~+3 GB post-init VRAM |

따라서 관찰된 steady gap 의 직접 원인은 SM/HBM 포화가 아니라 AOS segment/direct-emission/D2H
command 제출 수가 training launch cadence 를 밀어내는 host/CUDA-driver scheduling 문제다.
128 MiB 계열은 1.09 GB AOS 를 9 segment 로 쪼개고, segment 별 direct emission kernel 과
D2H copy command 를 반복 제출해 `aos enqueue` 가 50 ms 대까지 커졌다. 384 MiB segment 는
같은 payload 를 3 segment 로 줄여 enqueue 를 20 ms 안팎으로 낮추지만 persistent VRAM 이
증가한다. 현재 기본은 R1/R4 audit 여유를 우선해 125 MiB 로 롤백되어 있으며, 384 MiB 는
launch-cadence 완화용 실험값으로만 취급한다.

adaptive per-expert cap 재배분 검증:
- fixed single pinned slab / per-iter pinned allocation 0 / expert별 contiguous slice 유지.
- total AOS D2H capacity 는 1.089 GB 로 증가하지 않았다.
- direct-emission feedback(`cursor`, `overflow`)을 다음 iter 의 capacity hint 로 사용한다.
  기존 layout 은 **모든 expert 의 overflow 가 0일 때만** freeze 한다. 전체 unused capacity 가
  충분해 보여도 expert-local overflow 가 있으면 cold expert 의 남는 slice 를 hot expert 로
  옮길 수 있게 재배분해야 한다.
- cap freeze 전에는 direct layout rebuild 가 반복되어 `layout≈28 ms` 였고, freeze 후
  profiled iter 에서 `layout≈0.1 ms`, `enqueue≈7.6--7.8 ms` 로 내려갔다.
- hot expert overflow 는 기존 size-proportional cap 의 p50 수천만 record 수준에서
  selected-proportional redistribution 후 p50 수십만 record 수준으로 감소했다.
- 1100 MiB segment 는 per-iter segment 수와 enqueue time 을 가장 낮추지만 persistent VRAM 이
  약 +3.0 GB 로 커진다. 384 MiB segment 는 3 segment 로 같은 scheduling 문제를 크게 줄이는
  실험값이다. 현재 production 기본은 VRAM contract 를 우선해 125 MiB 로 둔다.

selection quality 결론:
- 기존 group-dg score / group-forced emission / global element priority 는 production hot path 에서 폐기됐다.
- 현재 production score 는 rho/e_hat 기반 abs-H Taylor element-wise 단일 score 이며, force 는 τ count 에 포함되고 capacity cap 을 우회하지 않는다.
- steady telemetry 에서 sample fraction 은 1%로 유지됐고, probe/direct score identity 와 K-band 가 확인됐다.

H refresh boundary 검증:
- scalar `κ_h` bridge 를 실제 τ row 에 적용하면 refresh step 에 severe over-select 가 발생했다.
- production 은 refresh 직전 direct feedback consume → τ model rows drop → refresh step
  cold-start cutoff bootstrap 순서를 사용한다.
- h-refresh=50 stress 에서 refresh step 및 이후 1--5 iter 가 모두 K-band 안으로 복귀했고,
  `decision_code=2`, `probe_reject=0`, `retry=0` 으로 관찰됐다.
