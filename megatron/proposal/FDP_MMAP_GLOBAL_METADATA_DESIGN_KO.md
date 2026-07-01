# FDP mmap Global Metadata Write 설계

## 1. 결론

구현 가능하다. 목표 구조는 다음이다.

```text
FDP session open
  -> SSD metadata load/format
  -> shared mmap metadata 생성/attach
  -> mmap metadata를 runtime source of truth로 사용

iteration N checkpoint
  -> 각 rank process의 non-expert writer / expert writer가 동시에 시작
  -> metadata lock 안에서 필요한 chunk를 할당하고 metadata entry만 갱신
  -> metadata lock 밖에서 uring SQE prepare/submit/CQE wait
  -> 모든 rank process의 write가 끝나면 마지막 완료 process가 SSD metadata flush 1회 수행
```

핵심 효과:

- 매 metadata 접근마다 SSD metadata image를 다시 읽지 않는다.
- bitmap/entry 수정만 lock으로 보호한다.
- data write 준비/submit/CQE wait는 metadata lock 밖에서 진행한다.
- metadata SSD flush는 iteration당 1회만 수행한다.

단, 이를 정확히 보장하려면 현재 `fdp_store.cpp`의 batch API를 조금 더 강하게 바꿔야 한다. 지금의 `write_*()` 개별 호출 방식만으로는 "chunk 전부 할당 후 모든 요청 prepare"를 완전히 보장하기 어렵다. 새 구조에서는 writer batch가 먼저 metadata reservation을 끝내고, 그 결과로 만든 IO plan을 lock 밖에서 uring에 prepare해야 한다.

## 2. 현재 구조의 한계

현재 FDP write path는 각 rank process마다 C++ `FdpStore`가 있고, 각 process의 DRAM에 metadata image copy를 가진다. process 간 정합성은 `flock`과 SSD metadata reload로 맞춘다.

현재 batch 흐름은 대략 다음이다.

```text
begin_nefc_batch / begin_expert_batch
  -> global metadata lock
  -> SSD metadata read

write_nefc / write_lfcs / open_lfsy / append_lfsy
  -> in-memory metadata 수정
  -> uring SQE prepare

end_nefc_batch / end_expert_batch
  -> metadata SSD flush
  -> global metadata lock release
  -> CQE wait
```

이는 이전보다는 좋아졌지만, 다음 한계가 있다.

- batch 시작마다 SSD metadata read가 필요하다.
- NEFC writer와 expert writer의 metadata reservation phase가 `flock`에 묶인다.
- `fsync=True`, `fsync_fd()` 같은 기존 file-backend 의미가 FDP metadata flush를 불필요하게 여러 번 부를 수 있다.
- metadata flush가 writer별로 일어나며, "모든 process 통틀어 iteration당 1회"는 아직 아니다.

## 3. 새 구조의 전제

지원 범위:

- `DP=1`, `TP=1`, `PP>=1`, `EP>=1`
- process 수 = `world_size = pp_size * ep_size`
- 각 rank process는 local GPU 1개를 담당한다.
- 각 process 내부에는 최소 두 writer 흐름이 있다.
  - non-expert writer: NEFC shard write
  - expert writer: LFCS/LFSY write

fault/crash consistency:

- 이번 설계에서는 fault 중간 상태 복구는 고려하지 않는다.
- 정상 실행에서 중복 chunk 할당, metadata race, flush 누락이 없어야 한다.
- crash recovery를 다시 고려하려면 pending/commit 상태 또는 journal이 필요하다.

## 4. mmap 생성 시점

mmap metadata는 반드시 다음 시점에 만들어져야 한다.

```text
FDP device open 후
metadata layout 계산 후
SSD metadata load/format 직후
첫 write 시작 전
```

현재 코드 기준 위치:

```text
FdpStore::open_from_env()
  -> load_config_from_env()
  -> device open / namespace identify
  -> metadata layout 계산
  -> allocate_metadata_image()
  -> format_or_load()
  -> open_or_create_shared_metadata_mmap()
```

`format_or_load()`가 끝난 뒤에는 `metadata_image_`에 SSD metadata의 최신 상태 또는 fresh metadata가 들어 있다. 이 이미지를 shared mmap metadata area로 복사한다.

## 5. mmap 파일과 control header

shared mmap은 `/dev/shm` 또는 `shm_open()` 기반 POSIX shared memory로 만든다.

권장 이름:

```text
/proposal_fdp_<device_sanitized>_m<metadata_start_lba>_v<format_version>
```

전체 mmap layout:

```text
+------------------------------------------------+
| SharedControlHeader                            |
+------------------------------------------------+
| Metadata image bytes                           |
|  - MetadataHeader                              |
|  - global bitmap                               |
|  - NEFC shard table                            |
|  - LFCS table                                  |
|  - LFSY table                                  |
+------------------------------------------------+
```

`SharedControlHeader` 필드:

```cpp
struct SharedControlHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t header_bytes;

    uint64_t layout_fingerprint;
    uint64_t metadata_image_bytes;
    uint32_t world_size;
    uint32_t rank_count;

    pthread_mutex_t metadata_mutex;  // PTHREAD_PROCESS_SHARED

    uint64_t active_iter;
    uint32_t expected_processes;
    uint32_t completed_processes;
    uint32_t flush_started;
    uint32_t flush_done;
    uint32_t flush_owner_rank;
    uint32_t error_rank;

    uint64_t metadata_flush_count;
};
```

`layout_fingerprint`에는 다음 값들을 hash한다.

- format version
- namespace NLB / LBA size
- metadata range
- data range
- chunk size
- data chunk count
- expert slot count
- NEFC shard count
- DP/PP/EP sizes

이 fingerprint가 맞지 않으면 stale mmap으로 보고 재초기화한다.

## 6. mmap 초기화 프로토콜

여러 rank가 동시에 시작할 수 있으므로 초기화는 별도 init lock으로 보호한다.

```text
open_or_create_shared_metadata_mmap()
  -> init lock 획득
  -> shm_open/create
  -> ftruncate(control + metadata_image_bytes)
  -> mmap

  if control.magic/layout mismatch or force_format:
       pthread_mutexattr_init(PTHREAD_PROCESS_SHARED)
       pthread_mutex_init(metadata_mutex)
       control fields 초기화
       SSD-loaded metadata_image_ -> mmap metadata image 복사
  else:
       기존 mmap attach

  -> init lock 해제
```

초기화 후 `FdpStore`는 더 이상 per-process `metadata_image_`를 runtime source로 쓰지 않는다. `metadata_bytes()`는 shared mmap metadata image pointer를 반환한다.

`metadata_image_` local buffer는 SSD metadata read/write scratch로만 남긴다.

## 7. Metadata lock 범위

새 구조에서 metadata lock은 `SharedControlHeader::metadata_mutex` 하나로 통일한다.

lock 안에서만 가능한 작업:

- global bitmap chunk allocate/free
- NEFC shard entry 수정
- LFCS entry 수정
- LFSY entry 수정
- header seq/last_flush_iter/allocated_chunks 갱신
- iteration completion state 수정
- SSD metadata flush snapshot 보호

lock 밖에서 해야 하는 작업:

- pinned buffer header stamp
- uring SQE prepare
- io_uring_submit
- CQE wait
- payload checksum/debug readback

즉 목표는 다음이다.

```text
metadata lock
  chunk allocation
  metadata entry update
  IO plan 생성
unlock

uring prepare/submit/wait
```

## 8. Writer batch reservation 구조

현재 `write_*()`가 metadata update와 ring prepare를 한 함수에서 같이 수행한다. 새 구조에서는 이 둘을 분리한다.

### 8.1 새 내부 자료구조

```cpp
enum class IoPlanKind {
    NefcWrite,
    LfcsWrite,
    LfsyOpenHeader,
    LfsyAppend,
};

struct IoPlan {
    IoPlanKind kind;
    UcmdRing* ring;
    int64_t pinned_addr;
    uint64_t total_padded_bytes;
    uint64_t logical_offset;
    std::vector<FdpChunkId> chunks;
};

struct ActiveBatch {
    uint64_t iter;
    bool active;
    bool metadata_sealed;
    std::vector<IoPlan> plans;
};
```

### 8.2 Batch 흐름

NEFC writer:

```text
begin_nefc_batch(iter)
  -> ActiveBatch 생성

write_nefc(...)
  -> metadata_mutex lock
     - chunk allocate
     - current NEFC shard entry 갱신
     - IoPlan 생성
  -> metadata_mutex unlock
  -> ring prepare는 아직 하지 않음

end_nefc_batch(iter)
  -> 모든 IoPlan을 uring에 prepare
  -> submit/CQE wait
  -> mark_writer_done(iter, NON_EXPERT)
```

Expert writer:

```text
begin_expert_batch(iter)
  -> ActiveBatch 생성

write_lfcs / open_lfsy / append_lfsy
  -> metadata_mutex lock
     - 필요한 chunk allocate/free
     - LFCS/LFSY entry 갱신
     - IoPlan 생성
  -> metadata_mutex unlock
  -> ring prepare는 아직 하지 않음

end_expert_batch(iter)
  -> 모든 IoPlan을 uring에 prepare
  -> submit/CQE wait
  -> mark_writer_done(iter, EXPERT)
```

더 최적화하려면 `write_*()`마다 lock/unlock하지 않고, `begin_*_batch(iter)`에서 metadata reservation lock을 잡고 writer batch의 모든 operation descriptor를 한 번에 넘기는 API를 만들 수 있다. 그러나 현재 Python writer 구조와의 변경 폭을 줄이려면 우선 `write_*()` 단위 short lock으로 시작해도 된다.

중요한 점은 `write_*()`가 더 이상 ring prepare를 하지 않는다는 것이다. chunk 할당이 끝난 뒤 `end_*_batch()`에서 모든 plan을 모아 prepare한다.

## 9. uring prepare/submit 의미

사용자가 말한 "모아서 submit"은 logical batch로 보장한다.

```text
metadata reservation 완료
  -> 모든 IoPlan에 대해 SQE prepare
  -> ring batch submit
  -> CQE wait
```

단, command 수가 queue depth보다 크면 물리적으로는 중간 submit이 필요할 수 있다. `io_uring` SQ size보다 많은 SQE를 한 번에 들고 있을 수 없기 때문이다. 따라서 구현상 의미는 다음이어야 한다.

```text
metadata lock 밖에서 writer batch 전체를 streaming submit
```

즉 metadata lock을 잡은 채 submit하지 않으며, batch 전체의 CQE 완료를 `end_*_batch()`에서 확인한다.

## 10. Iteration coordinator

iteration당 metadata SSD flush 1회를 보장하려면 rank/process 단위 완료 프로토콜이 필요하다.

### 10.1 Python 호출 위치

`proposal_dispatch_post_iter(iter_T)`에서 writer submit 직전에 호출:

```text
proposal_io.fdp_begin_checkpoint_iter(iter_T)
```

각 writer 완료 후:

```text
non-expert writer 완료:
  proposal_io.fdp_mark_writer_done(iter_T, "nonexpert")

expert writer 완료:
  proposal_io.fdp_mark_writer_done(iter_T, "expert")
```

process 내부에서 두 writer가 모두 완료되면 C++가 자동으로 process 완료를 shared mmap state에 반영한다.

### 10.2 C++ local state

각 process의 local state:

```cpp
struct LocalIterState {
    uint64_t iter;
    bool nonexpert_done;
    bool expert_done;
    bool process_done_reported;
};
```

`mark_writer_done()`:

```text
local nonexpert/expert done flag set

if both done and not reported:
    reported = true
    mark_process_done(iter)
```

### 10.3 Shared process completion

`mark_process_done(iter)`는 shared mmap의 `metadata_mutex` 안에서 실행한다.

```text
metadata_mutex lock
  assert active_iter == iter
  completed_processes += 1

  if completed_processes == expected_processes and flush_started == 0:
      flush_started = 1
      flush_owner_rank = my_rank
      flush mmap metadata image -> SSD metadata region
      NVMe flush
      metadata_flush_count += 1
      flush_done = 1
metadata_mutex unlock
```

이렇게 하면 모든 process 통틀어 마지막 완료 process만 SSD metadata flush를 수행한다.

## 11. Iteration 시작/종료 ordering

iteration N metadata flush가 끝나기 전에 iteration N+1 metadata update가 mmap에 섞이면 안 된다. 따라서 다음 규칙이 필요하다.

```text
fdp_begin_checkpoint_iter(N)
  -> shared active_iter가 비어 있거나 N-1 flush_done이어야 함
  -> active_iter = N
  -> completed_processes = 0
  -> flush_started = 0
  -> flush_done = 0
```

즉 FDP metadata 관점에서는 active checkpoint iteration을 하나만 허용한다.

현재 training 구조는 N+1 forward 시작과 동시에 N checkpoint를 시작하지만, 같은 rank에서 다음 checkpoint가 같은 pinned lease를 재사용하기 전에 이전 writer completion을 확인해야 한다. 이 조건을 유지하면 active iteration 하나 제한과 맞는다.

향후 여러 iteration의 disk write를 동시에 outstanding시키려면 shared state를 ring buffer로 바꿔야 한다. 지금 목표에서는 필요 없다.

## 12. 기존 fsync/flush API 변경

기존 file backend 의미 때문에 다음 함수들이 FDP metadata flush를 부를 수 있다.

- `write_prptens_from_pinned(..., fsync=True)`
- `write_prptens_from_pinned_keep_open()`
- `fsync_fd(fd)`
- `flush_metadata(iter)`

mmap iteration coordinator를 켠 경우 이 함수들은 metadata SSD flush를 직접 하면 안 된다.

새 규칙:

```text
PROPOSAL_FDP_METADATA_MMAP=1 인 경우:
  write_prptens_from_pinned(... fsync=True) -> data write만 수행, metadata flush 안 함
  fsync_fd(FDP LFSY handle) -> no-op 또는 handle state check만 수행
  flush_metadata(iter) -> explicit debug/manual flush API로만 사용
  정상 iteration flush -> mark_process_done()의 last process만 수행
```

file backend와 FDP backend의 fsync 의미를 분리해야 한다.

## 13. NEFC/LFCS/LFSY별 metadata update

### 13.1 NEFC

```text
metadata lock
  chunks = allocate_chunks(needed)
  old NEFC shard chunks free
  NEFC[pp_rank * ep_size + ep_rank] = new entry
  IoPlan(chunks, pinned_addr, total_padded)
unlock

uring prepare/write chunks
```

### 13.2 LFCS

```text
metadata lock
  chunks = allocate_chunks(needed)
  old LFCS chunks free
  old LFSY chunks free
  LFCS[global slot] = new dense baseline
  LFSY[global slot] = invalid
  IoPlan(chunks, pinned_addr, total_padded)
unlock

uring prepare/write chunks
```

### 13.3 LFSY open

```text
metadata lock
  header_chunk = allocate_chunks(1)
  old LFSY chunks free
  LFSY[global slot] = valid, logical_tail=HEADER_BLOCK_BYTES
  handle 생성
  IoPlan(header_chunk, header_buf)
unlock

uring prepare/write header block
```

### 13.4 LFSY append

```text
metadata lock
  old_tail = entry.logical_tail
  new_tail = old_tail + append_bytes
  추가 chunk 필요 시 allocate
  entry.logical_tail = new_tail
  IoPlan(chunks snapshot, pinned_addr, old_tail, append_bytes)
unlock

uring prepare/write append range
```

## 14. Lock granularity 평가

최소 lock 단위:

```text
single metadata entry update
```

더 좋은 lock 단위:

```text
writer batch reservation phase
```

권장 초기 구현:

- `write_*()` 호출마다 metadata lock을 짧게 잡는다.
- ring prepare는 반드시 lock 밖으로 분리한다.
- 기능 검증 후 expert writer에서 LFSY append 100개 이상을 한 번에 reserve하는 batch descriptor API를 추가한다.

이 순서가 안전하다. 처음부터 batch descriptor API까지 크게 바꾸면 기존 training path와 smoke path를 동시에 깨뜨릴 가능성이 크다.

## 15. Recovery 관점

정상 종료/정상 iteration에서는 recovery가 SSD metadata를 읽으면 마지막 flush 완료 iteration의 global metadata를 본다.

fault를 고려하지 않는 이번 설계에서는 다음을 허용한다.

- mmap에는 최신 iteration metadata가 있지만 SSD에는 아직 flush되지 않은 상태
- process crash 시 해당 iteration metadata 유실

그러나 정상 training 중 recovery를 수행하지 않는다면 문제 없다.

향후 fault consistency까지 보려면:

- entry pending/valid 상태
- metadata journal
- data write 완료 후 commit

중 하나가 필요하다.

## 16. 구현 단계

### Phase A: mmap runtime metadata 도입

- `SharedControlHeader` 추가
- `shm_open`/`mmap` 초기화 추가
- `metadata_bytes()`가 shared mmap image를 반환하도록 변경
- init lock으로 mmap 생성 race 제거
- 기존 SSD load/format 결과를 mmap으로 복사

### Phase B: metadata lock 교체

- `flock` 기반 per-access SSD reload 제거
- `pthread_mutex_t(PTHREAD_PROCESS_SHARED)` 기반 metadata mutex 사용
- `allocate_chunks/free_chunks`는 mmap bitmap을 직접 수정
- SSD metadata read는 open/load 시점과 explicit recovery/debug에만 수행

### Phase C: write path 분리

- `write_*()`에서 metadata reservation과 ring prepare 분리
- `IoPlan` 저장 구조 추가
- `end_*_batch()`에서 plan을 모아 uring prepare/submit/CQE wait
- FDP backend에서 per-write metadata flush 제거

### Phase D: iteration coordinator

- `fdp_begin_checkpoint_iter(iter)` 추가
- `fdp_mark_writer_done(iter, writer_kind)` 추가
- process-local writer done state 추가
- shared completion counter 추가
- 마지막 process가 metadata SSD flush 1회 수행

### Phase E: Python training path 연결

- `proposal_dispatch_post_iter(iter_T)`에서 `fdp_begin_checkpoint_iter(iter_T)` 호출
- `_execute_non_expert_post_iter` 완료 시 `fdp_mark_writer_done(iter_T, "nonexpert")`
- `_execute_expert_post_iter` 완료 시 `fdp_mark_writer_done(iter_T, "expert")`
- FDP mmap mode에서는 `fsync_fd`/`flush_metadata`의 자동 flush 경로 비활성화

### Phase F: smoke/test

- single process smoke:
  - mmap metadata mode open
  - 1 iter write
  - metadata_flush_count == 1 확인

- multi-process CPU smoke:
  - world_size=4, pp=2, ep=2를 process 4개로 시뮬레이션
  - 각 process가 자기 NEFC shard와 expert slots write
  - 마지막 process만 flush 수행 확인
  - SSD metadata readback으로 global bitmap 중복 할당 없음 확인

- training-like smoke:
  - 12 layer, 16 expert, 1% LFCS
  - NEFC + LFCS + LFSY 동시 write
  - `submit_lifetime_ms`, `metadata_lock_ms`, `metadata_flush_ms` 출력

## 17. 반드시 지켜야 할 invariant

- mmap metadata는 runtime 중 global source of truth다.
- metadata lock 없이 bitmap을 읽고 chunk를 할당하면 안 된다.
- metadata lock을 잡은 상태에서 uring submit/CQE wait를 하면 안 된다.
- iteration N metadata SSD flush는 모든 rank process의 N write CQE 완료 이후에 한 번만 수행한다.
- iteration N+1 metadata update는 N metadata flush 완료 전에는 시작하면 안 된다.
- FDP mmap mode에서 기존 file fsync API가 metadata flush를 중복 수행하면 안 된다.
- pinned buffer는 해당 IO CQE 완료 전까지 재사용하면 안 된다.

## 18. 최종 write flow

```text
N+1 forward 시작
  -> proposal_dispatch_post_iter(N)
     -> fdp_begin_checkpoint_iter(N)
     -> nonexpert writer submit
     -> expert writer submit

rank process nonexpert writer:
  -> begin_nefc_batch(N)
  -> metadata lock: NEFC chunk reserve + entry update + IoPlan 생성
  -> metadata unlock
  -> end_nefc_batch(N): uring prepare/submit/CQE wait
  -> fdp_mark_writer_done(N, nonexpert)

rank process expert writer:
  -> begin_expert_batch(N)
  -> metadata lock: LFCS/LFSY chunk reserve + entry update + IoPlan 생성
  -> metadata unlock
  -> end_expert_batch(N): uring prepare/submit/CQE wait
  -> fdp_mark_writer_done(N, expert)

process local:
  if nonexpert_done && expert_done:
      shared completed_processes += 1

last process:
  if completed_processes == world_size:
      mmap metadata image -> SSD metadata region write
      NVMe flush
      flush_done = 1
```

