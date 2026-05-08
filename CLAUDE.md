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

## 3. 검증 의무

새 변경마다 다음 측정값을 비교:
- **iter time gap** (vs baseline outlier-OFF) ≤ 10 ms
- **persistent VRAM gap** = 0 MB
- **transient VRAM peak gap** ≤ 80 MB
- **algorithm 정합성**: val loss 가 baseline 과 4-자리 이상 일치

위 4 개 중 하나라도 실패하면 변경 폐기 또는 재설계.

---

# 용어 정리

- loseless FC(Full Checkpointing):
non-expert는 특정 간격(W)마다 실행함. 
expert는 특정 간격(D)마다 실행함.
Adam Optimizer, model weight 등을 선별 없이 100% 저장하는 가장 일반적인 checkpointing 방법.
- lossy FC:
expert에만 적용됨. 
매 iteration 마다 element-level score (single iter `H × Δθ²`) 와 group-level score (time-window `H × δ[g]`) 가 **단일 K-budget (= topk_ratio × N_total_elements) 에서 경쟁** 하여 두 pool 의 통합 top 을 선별. 선택된 element / group 의 (idx, p, m, v) 가 expert 의 lossy FC 파일에 append.
- DC(Differential Checkpointing):
non-expert에만 적용됨. 
매 iteration 마다 non-expert의 전체 gradient만을 저장함. 복구 시, non-expert loseless FC 와 W간격동안의 DC를 통해 모델을 lossless하게 복구함.

---

# Non-expert 저장 전략

- 특정 간격(W)마다 전체 non-expert parameters를 loseless FC(Full Checkpointing) 함.
- 매 iteration 전체 gradient를 저장함 (선별 없음)
- 복구 시, non-expert loseless FC 와 W간격동안의 DC를 통해 모델을 lossless하게 복구함.
- 근거: (1) 크기가 전체 모델의 10~20%로 작아 매 iteration 마다 저장해도 부담이 적고, 전체 모델에 미치는 영향이 높아 매 iteration 마다 저장함, (2) gradient 는 optimizer 자체에 영향을 미치기 때문에 중간의 어느 iteration이라도 유실되면 안됨.

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
element score:  s_elem(i)   = H[group(i)] × Δθ_i²                    # single iter, per-element
group   score:  s_group(g)  = H[group=g]  × δ[g]
                where δ[g]  = (1/g) × Σ_{step ∈ window} Σ_{i ∈ g, not selected} Δθ_i²
```

- `Δθ_i²` = optimizer 의 m, v 로부터 유도한 이번 step 의 update² (single iter).
- `δ[g]` = 시간창에 걸친 그룹 평균 per-element δ². "선택 안 된" 원소들의 (Δθ²/g) 가 누적되어 자연 회복.
- `H[group]` = Finite-Difference 기반 group-level diagonal Hessian (production). Hutchinson 은 FD vs Hutch cross-check 에서만 사용.
- 두 score 모두 per-element 단위 (group 측이 1/g 정규화) → 같은 axis 에서 직접 비교 가능.

Non-expert parameter 는 매 W iter 전부 저장하므로 score 불필요.

## Two-pool Unified Top-K (단일 K-budget 경쟁)

### 알고리즘

매 iter 다음을 수행:

1. **Score 계산**:
   - element pool: 각 expert element 의 `H × Δθ²` (단일 iter scale)
   - group pool: 각 group 의 `H × δ[g]` (시간창 mean scale, 같은 per-element axis)

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

## 4-stream / writer-thread 아키텍처 (Hard Contract)

학습은 **절대로** checkpointing 작업에 의해 중단되지 않는다. 이를 위해 4 계층의 책임 분리를 강제한다.

```
┌─────────────────┬──────────────────────────────────────────────────────────┐
│ Stream / Thread │ 책임                                                      │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ main_stream     │ 학습 — Megatron/DeepSpeed 의 forward / backward /         │
│                 │ optimizer 가 launch. 이 stream 위에는 proposal 의 어떤   │
│                 │ 작업도 enqueue 되어선 안 됨 (critical path).              │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ tracker_stream  │ Proposal 의 expert 관련 GPU compute — Triton outlier      │
│                 │ kernel, group score chunk-loop, index_select, dg index_   │
│                 │ fill 등. tracker.step 이 진입 시 main_stream 의 grad-     │
│                 │ ready event 만 wait_event 하고 자체적으로 진행.           │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ d2h_stream      │ **모든** GPU → pinned host memory 데이터 이동 (non-       │
│                 │ expert DC + expert lossy dispatch + LFCL payload). H2D    │
│                 │ destination 은 반드시 pinned (pageable 은 ~1.2 GB/s,     │
│                 │ pinned 는 ~16 GB/s + non_blocking 가능). tracker_stream  │
│                 │ 의 결과를 cuda.Event 로 받아 wait_event 후 copy.          │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ writer thread   │ Host DRAM 의 데이터를 DRAM buffer (DRAM mode) /          │
│ (host CPU)      │ disk file (disk mode) 에 저장. 모든 numpy 조립 / scatter │
│                 │ / file write 는 여기서. d2h_stream 의 cuda.Event 를      │
│                 │ writer 가 직접 `event.synchronize()` 호출하여 H2D 완료   │
│                 │ 후 read. training thread 는 stream sync 호출 안 함.      │
└─────────────────┴──────────────────────────────────────────────────────────┘
```

### 데이터 흐름 (per iter)

```
forward ─backward─ optimizer.step()                  [main_stream]
   │                       ↑
   │                       │ wait_event(grad_ready)
   ▼                  ┌────┴─────┐
   (next iter)        │  tracker  │  Triton kernel (atomic histogram +
                      │  _stream  │  selection + selective fold), pass2
                      │           │  histc + chunk-loop selection,
                      │           │  index_select per BF16-group
                      └────┬──────┘
                           │ event_compute
                           ▼
                      ┌──────────┐
                      │d2h_stream│  pinned_host.copy_(gpu_tensor,
                      │          │  non_blocking=True)
                      └────┬─────┘
                           │ event_h2d
                           ▼
                      ┌──────────┐
                      │  writer  │  event_h2d.synchronize()
                      │  thread  │  → numpy assembly / scatter
                      │          │  → binary_io.write_lossless_payload
                      │          │    or binary_io.append_lossy_log
                      └──────────┘
```

### File I/O 위치 강제 규칙

**모든 실제 file write 호출은 `megatron/proposal/binary_io.py` 안에만 위치한다.** 다른 모듈에서 `os.write*`, `os.fsync`, `os.replace`, `f.write(`, `tofile(` 를 직접 호출하지 않는다 — `binary_io` 의 헬퍼 (`save_binary`, `write_lossless_payload_atomic`, `open_lossy_log_for_append`, `append_lossy_log`, `fsync_fd`, `close_fd`) 를 사용한다. 위반 사항은 `grep -rEn 'os\.writev|os\.write\(|os\.fsync' megatron/proposal/ | grep -v binary_io.py | grep -v "test\|docstring"` 로 0 건이 나와야 한다.

binary_io 의 헬퍼 함수들은 모두 **CPU 전용** (torch / cuda dependency 0): bytes / memoryview / numpy array 만 받음. → GPU stream 위에 잘못 호출될 가능성 0.

### Stream sync 의무

- tracker_stream → d2h_stream: `cuda.Event.record(tracker_stream)` + `d2h_stream.wait_event(...)`. 단일 event 가 phase 전체를 커버 (per-name event 금지 — overhead).
- d2h_stream → writer thread: `cuda.Event.record(d2h_stream)` 을 writer 호출에 전달. writer 가 `event.synchronize()` 호출 (background thread 라 main_thread blocking 0).
- main_stream / tracker_stream / d2h_stream 사이의 host-blocking sync (`stream.synchronize()`, `event.synchronize()` on training thread) 는 dispatch 진입 시점의 `ev.synchronize()` (1-iter pipelining) **단 1 회만** 허용. 이는 prev iter 의 tracker work 가 main_stream forward (~2000 ms) 보다 일찍 끝나므로 host wait time = 0.

### 1-iter pipelining

- iter T 의 tracker_stream 작업 결과는 iter T+1 의 dispatch 에서 사용 (1-iter 지연).
- iter T 의 stash 시점에 GPU intermediate (sel_idx, sel_count, etc.) 를 `.clone()` 하여 ref 보관. iter T+1 dispatch 가 `ev.synchronize()` 후 `.item()` 으로 host scalar 추출 (sync 비용 0 — 이미 GPU 작업 완료).
- 이 방식으로 stash 단계의 어떤 host sync 도 main_thread 를 block 하지 않음.

## Checkpoint 이벤트 — 매 iteration

매 training iteration마다 CP 이벤트가 발생. 

**Non-expert:**

- `step % W == 0` → FC (W=10 default). 전체 param + optimizer state 저장.
- 그 외 → DC. Lossless (전체 gradient dense 저장).
- results/{실험이름}/cp/non_expert 에 저장

**각 Expert:**

- 항상 K%만 저장
- results/{실험이름}/cp/expert 에 저장

## Differential Checkpoint (DC)

매 iter `dc_iter{step}_dense_ne.pt` 생성 (non-expert dense gradient only;).

- Non-expert: FC하지 않은 iter의 dense gradient
- Metadata: `step`

Retention: 최근 `max(W, 3)`개만 보관 (구현 정책: `CheckpointManager(retention=W+2)`).

---

# Key Parameters

| 파라미터 | 설명 | 기본값 |
| --- | --- | --- |
| K% | Expert state snapshot 비율 |  |
| W | Non-expert FC 주기 (iter) | 10 |
| D | Expert lossless FC 수행 간격 (iteration) | 4 |
| async_queue_size | AsyncWriter pending queue | 5 |
| F% | Expert lossless FC 시 선택되는 expert 비율 | 2 |

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

#### Hutchinson 설정

- K=4 유지 (K=1 + EMA 조합은 품질 저하 이력으로 금지)
- refresh_every 40
- `HUTCHINSON_CPU_OFFLOAD` 기본 활성 — peak VRAM 예산 안에서 안전 동작

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
├── cp_hooks.py                 # Training-loop integration (init/pre/post/recover/shutdown) + refresh_hessian dispatch + fault injection + Fisher coverage
├── cli_args.py                 # CLI arg registration (--proposal-*)
├── hessian.py                  # State-free Hessian estimators: compute_hessian_hutchinson / compute_hessian_gnb / compare_hutchinson_vs_gnb
├── importance_tracker.py       # Element-wise score (H × δ²), global top-K, θ_CP cache
├── fc_manager.py               # FC scheduling: element_wise / staggered / sequential
├── checkpoint_manager.py       # Non-expert DC (dense gradient bundle) manager
├── expert_state.py             # Per-expert lossy FC append log (LFSY) + lossless FC snapshot (LFCL) per CLAUDE.md spec
├── recovery.py                 # Non-expert FC load + DC replay + per-expert lossless + lossy replay (via read_expert_state)
├── async_writer.py             # Async disk I/O queue (backpressure-bounded)
├── _ds_state.py                # DeepSpeed ZeRO 0/1/2 FP32 master state mapper
└── _exp_tools.py               # (measurement) log parsing + plotting utilities

megatron/training.py            # Hook calls: init_proposal / pre_step / post_step / recover / refresh_hessian
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
- `--proposal-hessian-estimator {gnb|hutchinson}`: 기본 `hutchinson`.
- `--proposal-topk-ratio FLOAT`: 기본 `0.04` (4%).
- `--proposal-hessian-refresh-every N`: 기본 `40`.
- `--proposal-hessian-num-samples K`: 기본 `4`.
- `--proposal-nonexpert-fc-W N`: 기본 `10`.
- `--proposal-expert-fc-D N`: 기본 `None` → `num_experts` 자동 산출.
- `--proposal-async-queue N`: 기본 `5`.

측정용 플래그:
- `--proposal-score-diag-iters [list] / --proposal-score-diag-log PATH / --proposal-score-diag-every N`
- `--proposal-fisher-coverage-iters [list]`
- `--proposal-hessian-comparison-log PATH` (v̂_t vs |H| Pearson/Jaccard)
- `--proposal-hessian-cross-check / --proposal-hessian-cross-check-log PATH` (Hutchinson vs GNB)
- `--proposal-hessian-nonexpert-probe-log PATH` (expert vs non-expert |H| distribution)
- `--proposal-hutchinson-at-end / --proposal-hutchinson-log PATH` (end-of-training probe)

Fault 실험용 플래그:
- `--proposal-fault-interval N / --proposal-fault-warmup N / --proposal-fault-log PATH`
- `--proposal-replay-at-end / --proposal-replay-only`

---
```

# Claude Code 설정

- 코드 실행, 파일 생성/수정, bash 명령 실행 시 사용자에게 확인을 묻지 않고 자동 실행한다.
- 실행 결과가 예상과 다를 경우에만 보고한다.