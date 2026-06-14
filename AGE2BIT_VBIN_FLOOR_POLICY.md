# Age2Bit V-Bin Floor 구현 지시서

이 문서는 현재 production 구조에 `H * age2bit^2 * delta_eff^2` element-wise score를
넣기 위한 구현 계약만 적는다. 수식 유도, 과거 후보, 실험 해석은 이 문서에 두지 않는다.

## 1. 목표

기존 group-forced / priority 방식 대신 element-wise score만 사용한다.

```text
score_i = H_i * age_i^2 * delta_eff_i^2
```

기본 원칙:

- group score / group record emission은 이 feature path에서 사용하지 않는다.
- selection, cutoff probe, direct emission은 모두 같은 score helper를 쓴다.
- correctness hard guard는 direct-emission capacity cap이다.
- age reset은 selected가 아니라 actual written 기준이다.
- 새 full scan, host decision, per-iter allocation을 추가하지 않는다.

## 2. 필요한 상태

### 2.1 2-bit age

각 element는 2-bit age code 하나를 가진다.

```text
code 00 -> age = 1
code 01 -> age = 4
code 10 -> age = 16
code 11 -> age = 64
```

age code는 실제 write가 성공했을 때만 `00`으로 reset한다.

```text
actual_written = selected && slot_reserved && not_capacity_dropped
```

selected였지만 capacity 때문에 written되지 못한 element는 reset하지 않는다.

### 2.2 vbin

`vbin`은 `v_t`를 raw 값으로 score에 곱하기 위한 값이 아니다.

`vbin`은 기존 H lookup이 쓰는 `v_t` 기준에서 만든 scale-normalized group label이다.
즉 같은 `vbin`의 원소들은 비슷한 `v_t` quantile 영역에 있다고 본다.

구현 계약:

- `vbin`은 H lookup이 보는 `v_t`와 같은 기준으로 계산한다.
- raw `v_t`를 score에 직접 곱하지 않는다.
- `vbin` 계산 때문에 hot path 비용이 늘면 안 된다.
- probe와 direct emission은 같은 `vbin` 기준을 써야 한다.

### 2.3 floor snapshot

old age bucket에서 현재 `delta^2`가 너무 작게 보이는 문제를 막기 위해 `vbin`별 floor를 둔다.

```text
floor_mid[vbin]  # code 10용
floor_old[vbin]  # code 11용
```

floor는 correctness source가 아니다. old bucket 점수를 보정하는 prior일 뿐이다.

## 3. Score Helper

모든 경로는 아래 helper와 동등한 계산을 사용해야 한다.

```text
age = age_rep(code_i)
delta2 = adam_delta_squared(i)
floor = 0

if code_i == 10:
    floor = floor_mid[vbin_i]
elif code_i == 11:
    floor = floor_old[vbin_i]

delta_eff2 = max(delta2, floor)
score = H_i * age * age * delta_eff2
```

force가 켜진 element는 selection 후보로 강하게 올릴 수 있다. 단, force는 capacity cap을
우회하면 안 된다.

## 4. Floor 갱신 순서

같은 iteration에서 `floor`를 만들고 그 `floor`로 같은 iteration의 tau를 결정하면 순서가
꼬인다. 따라서 production flow는 snapshot 방식으로 한다.

```text
iter T:
  floor_snapshot_T 를 사용해 cutoff/probe/direct emission 수행
  같은 sample/probe 경로에서 delta2를 vbin별로 관측
  관측 결과로 floor_snapshot_{T+1} 갱신
```

이 1-iteration lag는 허용한다. floor는 correctness source가 아니라 activity-scale prior이기
때문이다.

floor 갱신은 기존 sample/probe 경로에 붙인다.

```text
sample element i:
  delta2 = adam_delta_squared(i)
  vbin = vbin(i)
  vbin별 delta2 통계에 반영
```

그 후:

```text
floor_mid[vbin] = 같은 vbin delta2 분포의 중간 old-bucket upper floor
floor_old[vbin] = 같은 vbin delta2 분포의 더 높은 old-bucket upper floor
```

현재 후보의 의미:

```text
code 00/01: floor 없음
code 10   : same-vbin p75 계열 floor
code 11   : same-vbin p90 계열 floor
```

p75/p90은 수학 상수가 아니다. 구현에서는 "old code일수록 더 높은 same-vbin upper floor"라는
계약만 지킨다.

## 5. Production 흐름

### 5.1 Init / refresh

```text
1. packed 2-bit age state를 init 단계에서 할당한다.
2. floor snapshot device state를 init 단계에서 할당한다.
3. H refresh / vq refresh 후 H lookup 기준과 같은 기준으로 vbin 기준을 갱신한다.
4. per-iter allocation은 금지한다.
```

### 5.2 Cutoff / probe

```text
1. 이미 준비된 floor_snapshot을 읽는다.
2. sample/probe에서 score helper로 candidate tau count를 계산한다.
3. 같은 sample/probe 관측으로 다음 floor_snapshot을 갱신한다.
4. tau decision은 device에서 끝낸다.
```

금지:

```text
floor를 host로 내려 decision
floor를 host에서 만든 뒤 H2D
floor 계산용 full scan
```

### 5.3 Direct emission

```text
1. cutoff/probe와 같은 floor_snapshot을 읽는다.
2. cutoff/probe와 같은 score helper를 사용한다.
3. score >= tau 또는 force인 element만 slot 예약을 시도한다.
4. capacity cap 안에 들어간 record만 write한다.
5. actual written element만 age code를 00으로 reset한다.
6. written되지 않은 element는 age promotion 규칙을 적용한다.
```

## 6. 128-step stale force

`code 11` 원소는 deterministic phase가 맞으면 force candidate가 된다.

계약:

- force는 starvation 방지 장치다.
- force는 capacity cap을 우회하지 않는다.
- force됐지만 cap-drop되면 age를 reset하지 않는다.
- force share가 K-budget을 과도하게 먹는지 debug에서 확인한다.

## 7. 반드시 지킬 금지사항

```text
group score 재도입 금지
group record emission 재도입 금지
global element priority를 main score로 사용 금지
raw v_t multiplication 금지
floor용 full scan 금지
floor host decision 금지
per-iter allocation 금지
selected 기준 age reset 금지
probe/direct emission score 불일치 금지
```

## 8. 검증

feature gate로 켠 뒤 최소 검증:

```text
1. probe selected count와 direct emission predicate가 같은 score 기준인지 확인
2. age reset 수 == actual written 수 확인
3. overflow/cap-drop element가 age reset되지 않는지 확인
4. floor snapshot lag, tau가 K-band를 깨지 않는지 확인
5. write-off + D2H-on iter time이 증가하지 않는지 확인
6. VRAM audit R1-R4 통과
```
