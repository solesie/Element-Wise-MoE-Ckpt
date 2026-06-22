# FDP Phase 0-3 Raw NVMe Pre-Recovery 구현 설계

이 문서는 `FDP_C_MODULE_BINARY_IO_DESIGN_KO.md`를 바탕으로 recovery 구현 전까지의
단계를 구현 작업 단위로 쪼갠 설계다. 이 버전의 전제는 명확하다.

```text
FDP backend는 Phase 1부터 raw NVMe io_uring_cmd passthru를 필수 경로로 구현한다.
blockfile simulator는 필수 구현 경로가 아니라 선택적 unit/debug fallback이다.
```

범위:

```text
포함:
  Phase 0 - 현재 코드/문서 이름 정리
  Phase 1 - proposal_io backend switch + raw NVMe FDP session bootstrap
  Phase 2 - NEFC/LFCS write 경로 raw NVMe FDP 분기
  Phase 3 - LFSY fd 기반 상태를 append handle 기반으로 일반화

제외:
  Phase 4 - recovery.py FDP metadata read branch
  Phase 5 - performance hardening / advanced passthru tuning
  Phase 6 - crash-safe metadata A/B commit
```

이 단계의 목표는 "checkpoint write가 raw NVMe namespace의 FDP object store
metadata/allocator 구조를 통해 정상적으로 기록되고, metadata full flush까지 가능한
상태"다. recovery는 아직 기존 file backend 경로만 신뢰한다.

## 공통 전제

### 현재 코드 구조

```text
proposal_io.py
  - Python I/O facade
  - file backend C++ extension load
  - pinned allocation
  - write/read wrapper

_csrc/binary_io.cpp
  - 현재 POSIX file backend 구현

_csrc/proposal_io.cpp
  - file backend pybind

fc_manager.py
  - non-expert PRPTENS write 호출

expert_state.py
  - expert LFCL write
  - expert LFSY open/append/flush/close 상태 관리
```

### MVP 제약

```text
rank/world:
  - 기본은 single rank
  - world_size != 1이면 FDP backend는 fail-fast
  - multi-rank는 static rank partition 설계만 남겨두고 구현하지 않음

storage:
  - raw NVMe namespace에 직접 LBA write/read
  - io_uring_cmd passthru 필수
  - blockfile은 선택적 debug backend일 뿐 기본 경로가 아님

completion:
  - Python wrapper 호출 안에서 C++ write completion까지 기다림
  - C++ background completion과 Python buffer refcount protocol은 아직 없음

metadata:
  - chunk id array 기반
  - LFCS/LFSY packed entry table
  - dirty page tracking 없음
  - metadata image 전체 flush

crash safety:
  - 중간 crash recovery 미지원
  - 정상 종료 또는 명시적 flush 이후 상태만 유효
```

### 장치 안전 전제

raw NVMe passthru는 filesystem을 우회해 namespace LBA에 직접 쓴다. 잘못된 offset을
쓰면 장치 데이터를 즉시 손상시킬 수 있다. 따라서 Phase 1부터 fail-fast guard를
넣는다.

필수 환경변수:

```text
PROPOSAL_IO_BACKEND=fdp
PROPOSAL_FDP_DEVICE=/dev/ngXnY
PROPOSAL_FDP_NSID=...
PROPOSAL_FDP_METADATA_START_LBA=...
PROPOSAL_FDP_DATA_START_LBA=...
PROPOSAL_FDP_DATA_NLB=...
PROPOSAL_FDP_ALLOW_FORMAT=1   # 새 superblock/metadata 초기화 시에만 필요
PROPOSAL_FDP_WORLD_SIZE=1
PROPOSAL_FDP_RANK=0
```

원칙:

```text
- device path가 명시되지 않으면 FDP backend 시작 금지
- metadata/data LBA range가 명시되지 않으면 시작 금지
- 기존 superblock magic이 없고 ALLOW_FORMAT=1도 없으면 초기화 금지
- metadata range와 data range가 겹치면 시작 금지
- namespace capacity를 벗어나면 시작 금지
- world_size != 1이면 현재 단계에서는 시작 금지
```

### 3-ring topology

Phase 0-3부터 `proposal_fdp_store` session은 세 ring을 가진다.

```text
proposal_fdp_store session
  ├─ nefc_data_ring
  │    - NEFC / PRPTENS write
  │    - io_uring_cmd NVMe write submit
  │
  ├─ expert_data_ring
  │    - LFCS / LFCL write
  │    - LFSY append write
  │    - io_uring_cmd NVMe write submit
  │
  └─ metadata_ring
       - metadata image full flush
       - optional NVMe flush
       - io_uring_cmd NVMe write/flush submit
```

Python worker thread가 ring을 소유하지 않는다. `proposal_fdp_store` C++ session이
request type에 따라 내부 ring을 선택한다.

```text
ne_writer      -> proposal_io.write_prptens_from_pinned()
               -> C++ nefc_data_ring

expert_writer  -> proposal_io.write_lossless_from_pinned()
               -> C++ expert_data_ring

expert_writer  -> proposal_io.append_lossy_log_from_pinned()
               -> C++ expert_data_ring

flush path     -> proposal_io.flush_metadata() or fsync_fd(handle)
               -> C++ metadata_ring
```

## Phase 0 - 이름과 기준 정리

### 목적

구버전 `binary_io.py` 전제를 제거하고, 현재 실제 facade인 `proposal_io.py` 기준으로
문서와 테스트 이름을 맞춘다. 기능 변경은 하지 않는다.

### 작업

1. `FDP_C_MODULE_BINARY_IO_DESIGN_KO.md`가 현재 구조와 raw NVMe 필수 경로 기준인지
   확인한다.
2. `CHECKPOINT_PINNED_MEMORY_LAYOUT_KO.md`에서 `binary_io.*`라고 적힌 표현을
   `proposal_io.*` 또는 "proposal I/O facade"로 정리한다.
3. stale test import가 있으면 현재 module명 기준으로 정리한다.

정리 대상이었던 오래된 import:

```text
tests/unit_tests/proposal/test_binary_io.py
  from megatron.proposal.binary_io import ...
```

현재 repo에는 `megatron/proposal/binary_io.py`가 없고 `proposal_io.py`가 있다.
따라서 unit test는 `tests/unit_tests/proposal/test_proposal_io.py`로 옮기고,
`megatron.proposal.proposal_io` 기준으로 round-trip helper를 사용한다. production
API에 구버전 `save_binary()` 호환 함수를 추가하지 않는다.

### 완료 기준

```text
- 문서에서 구현 기준 facade가 proposal_io.py로 통일됨
- 구버전 binary_io.py 전제를 새 FDP 설계에서 사용하지 않음
- 기존 file backend 동작 변화 없음
```

## Phase 1 - Raw NVMe FDP session bootstrap

### 목적

`proposal_io.py`가 file backend와 FDP backend를 선택할 수 있게 하고, FDP backend가
raw NVMe device를 열어 identify/probe를 수행한 뒤 3개 io_uring_cmd ring을 생성한다.

이 단계의 산출물은 아직 checkpoint object write가 아니라, **raw namespace에
superblock/metadata image를 안전하게 쓰고 읽는 최소 passthru session**이다.

### Python 설계

`proposal_io.py`에 backend 선택 helper를 추가한다.

```python
def backend_kind() -> str:
    return os.environ.get("PROPOSAL_IO_BACKEND", "file").lower()
```

허용값:

```text
file
fdp
```

FDP extension은 lazy load한다.

```python
_fdp_cext = None

def _get_fdp():
    global _fdp_cext
    if _fdp_cext is None:
        _fdp_cext = load(
            name="proposal_fdp_cext",
            sources=[
                str(_src_dir / "fdp_store.cpp"),
                str(_src_dir / "fdp_passthru.cpp"),
                str(_src_dir / "proposal_fdp_io.cpp"),
            ],
            ...
        )
        _fdp_cext.open_session_from_env()
    return _fdp_cext
```

중요한 점:

```text
- file backend import는 기존처럼 동작
- FDP 장비나 권한이 없어도 file backend import가 깨지면 안 됨
- PROPOSAL_IO_BACKEND=fdp일 때만 FDP extension/session 초기화
```

### C++ 파일 구조

Phase 1에서 추가할 파일:

```text
megatron/proposal/_csrc/fdp_metadata.h
megatron/proposal/_csrc/fdp_store.h
megatron/proposal/_csrc/fdp_store.cpp
megatron/proposal/_csrc/fdp_passthru.h
megatron/proposal/_csrc/fdp_passthru.cpp
megatron/proposal/_csrc/proposal_fdp_io.cpp
```

역할:

```text
fdp_metadata.h
  - superblock
  - metadata image layout
  - NEFC/LFCS/LFSY entry structs
  - constants/capacity checks

fdp_passthru.*
  - NVMe generic char device open
  - identify namespace/controller helper
  - FDP capability probe helper
  - io_uring queue setup
  - NVMe passthru read/write/flush command builder
  - submit/wait completion wrapper

fdp_store.*
  - session lifecycle
  - namespace layout validation
  - metadata image allocation
  - bitmap allocator
  - chunk id -> LBA 변환
  - request type -> ring selection

proposal_fdp_io.cpp
  - pybind11 binding
```

### Device open/probe

초기 session open:

```text
1. read env
2. open PROPOSAL_FDP_DEVICE
3. identify namespace
4. identify controller if needed
5. verify nsid
6. derive lba_size
7. verify metadata/data LBA ranges
8. probe FDP support / placement ids
9. create 3 io_uring_cmd rings
10. load existing superblock or format if allowed
11. allocate metadata image in aligned host memory
12. read metadata image if existing, otherwise initialize
```

장치 probe 실패 시 FDP backend는 즉시 fail-fast한다. file backend fallback은 자동으로
하지 않는다. 사용자가 `PROPOSAL_IO_BACKEND=fdp`를 명시했으면 실패를 숨기지 않는다.

### io_uring_cmd ring setup

각 ring은 독립 queue depth를 가진다.

```text
PROPOSAL_FDP_NEFC_QD=...
PROPOSAL_FDP_EXPERT_QD=...
PROPOSAL_FDP_METADATA_QD=...
```

초기값 권장:

```text
nefc_data_ring:    32
expert_data_ring:  64
metadata_ring:      8
```

Phase 1에서는 submit 후 즉시 completion을 기다려도 된다.

```text
submit_cmd()
wait_all()
return status
```

즉 ring은 존재하지만 C++ background completion thread는 없다.

### NVMe passthru commands

Phase 1 최소 command:

```text
nvme_write_lba(buffer, slba, nlb, pid_hint)
nvme_read_lba(buffer, slba, nlb)
nvme_flush()
```

요구 조건:

```text
buffer address aligned to lba_size or 4KiB
byte length multiple of lba_size
slba within allowed metadata/data range
nlb within allowed metadata/data range
```

FDP pid hint는 Phase 1에서 probe만 하고, 실제 placement hint 적용이 장치별로 복잡하면
다음처럼 단계화한다.

```text
Phase 1 mandatory:
  raw NVMe io_uring_cmd read/write/flush works
  FDP support probe result exposed in stats()

Phase 1 optional:
  write command에 FDP placement directive/pid 적용
```

단, backend 이름은 FDP로 유지하되, `stats()`에 `fdp_pid_enabled=false`를 명확히
노출한다. 장치별 FDP directive encoding은 raw write skeleton이 안정된 뒤 확정한다.

### Namespace layout

raw namespace는 고정 LBA range로 사용한다.

```text
metadata_start_lba = PROPOSAL_FDP_METADATA_START_LBA
data_start_lba     = PROPOSAL_FDP_DATA_START_LBA
data_nlb           = PROPOSAL_FDP_DATA_NLB

LBA metadata_start_lba
│
├─ superblock
├─ metadata image

LBA data_start_lba
│
└─ data pool
     ├─ chunk 0
     ├─ chunk 1
     └─ ...
```

chunk mapping:

```c
chunk_lba_count = CHUNK_BYTES / lba_size;
chunk_lba = data_start_lba + chunk_id * chunk_lba_count;
```

### Metadata image

Phase 1 metadata는 fixed flat image다.

```c
typedef uint32_t FdpChunkId;

struct NefcEntry {
    uint8_t valid;
    uint32_t rank;
    uint64_t iter;
    uint64_t logical_bytes;
    uint32_t chunk_count;
    FdpChunkId chunks[MAX_CHUNKS_NEFC];
};

struct LfcsEntry {
    uint8_t valid;
    uint32_t rank;
    uint32_t layer_id;
    uint32_t expert_id;
    uint64_t generation;
    uint64_t iter;
    uint64_t logical_bytes;
    uint32_t chunk_count;
    FdpChunkId chunks[MAX_CHUNKS_LFCS];
};

struct LfsyEntry {
    uint8_t valid;
    uint32_t rank;
    uint32_t layer_id;
    uint32_t expert_id;
    uint64_t generation;
    uint64_t start_iter;
    uint64_t logical_tail;
    uint32_t chunk_count;
    FdpChunkId chunks[MAX_CHUNKS_LFSY];
};
```

LFCS/LFSY table은 실제 `num_moe_layers * experts_per_layer`만큼 할당한다. compile
time max는 capacity check 용도다.

### Bitmap allocator

Phase 1 allocator는 single-rank 전용이다.

```text
allocate_chunk():
  scan bitmap for 0 bit
  set bit to 1
  return chunk_id

free_chunk(chunk_id):
  assert bit is 1
  set bit to 0
```

object write는 chunk를 독점한다. chunk 하나를 여러 object가 공유하지 않는다.

### Metadata flush

Phase 1에서는 dirty page tracking 없이 전체 image를 metadata ring으로 flush한다.

```text
flush_metadata(iter):
  update metadata_image header counters
  nvme_write_lba(metadata_image, metadata_start_lba + superblock_nlb, metadata_image_nlb)
  optional nvme_flush()
```

### Phase 1 C++ pybind API

초기 API:

```cpp
open_session_from_env() -> void
close_session() -> void
format_or_load() -> void
flush_metadata(uint64_t iter) -> void
stats() -> dict
debug_read_metadata_summary() -> dict
debug_raw_write_read_smoke() -> bool
```

`debug_raw_write_read_smoke()`는 허용된 metadata scratch LBA 또는 data scratch chunk에
test pattern을 쓰고 읽어 검증한다. 운영 데이터와 겹치지 않는 reserved area를 사용해야
한다.

아직 `write_nefc/write_lfcs/open_lfsy/append_lfsy`는 Phase 2/3에서 붙인다.

### 완료 기준

```text
- PROPOSAL_IO_BACKEND=file이면 기존 경로 그대로 동작
- PROPOSAL_IO_BACKEND=fdp이면 FDP extension이 lazy load됨
- raw NVMe device open/probe 가능
- nsid/lba_size/capacity/range validation 수행
- 3개 io_uring_cmd ring 생성 가능
- metadata image allocation 가능
- bitmap allocate/free unit test 가능
- debug raw write/read smoke test 통과
- flush_metadata()가 raw metadata LBA range에 전체 image를 씀
```

## Phase 2 - NEFC/LFCS raw write 분기

### 목적

현재 pinned write wrapper를 유지하면서 NEFC와 LFCS write를 raw NVMe FDP backend로
분기한다. LFSY append는 아직 file backend 상태여도 된다. 하지만 Phase 3 전환을 위해
generation 필드는 이 단계에서 추가한다.

### proposal_io.py 변경

`write_prptens_from_pinned()`:

```python
def write_prptens_from_pinned(..., step: int, ...) -> int:
    total_padded = _round_up_sector(HEADER_BLOCK_BYTES + int(payload_bytes))
    if backend_kind() == "file":
        return int(_cext.write_prptens(...))
    if backend_kind() == "fdp":
        return int(_get_fdp().write_nefc(
            str(Path(final_path).name),
            int(pinned_tensor.data_ptr()),
            header_bytes,
            int(payload_bytes),
            int(total_padded),
            int(step),
        ))
```

`write_lossless_from_pinned()`:

```python
def write_lossless_from_pinned(
    final_path,
    pinned_tensor,
    header_bytes,
    payload_bytes,
    *,
    layer: int,
    expert: int,
    step: int,
    generation: int = 0,
    mode: int = 0o644,
) -> int:
    total_padded = _round_up_sector(HEADER_BLOCK_BYTES + int(payload_bytes))
    if backend_kind() == "file":
        return int(_cext.write_lossless(...))
    if backend_kind() == "fdp":
        return int(_get_fdp().write_lfcs(
            str(Path(final_path).name),
            int(pinned_tensor.data_ptr()),
            header_bytes,
            int(payload_bytes),
            int(total_padded),
            int(layer),
            int(expert),
            int(step),
            int(generation),
        ))
```

file backend는 `generation`을 무시한다.

### C++ FDP write_nefc

입력:

```text
object_name
pinned_addr
header_bytes
payload_bytes
total_padded_bytes
iter
```

동작:

```text
1. logical_bytes = HEADER_BLOCK_BYTES + payload_bytes
2. physical_bytes = round_up_to_lba(logical_bytes)
3. needed_chunks = ceil(physical_bytes / CHUNK_BYTES)
4. chunks[] allocate
5. pinned buffer 앞에 header stamp
6. chunks[] 순서대로 nefc_data_ring에 NVMe write submit
7. completion 확인
8. old NEFC chunks free
9. nefc.latest 갱신
10. return total_padded_bytes
```

chunk write는 pinned buffer logical offset을 chunk 순서에 맞춰 split한다.

```text
chunk i:
  src = pinned_addr + i * CHUNK_BYTES
  len = min(CHUNK_BYTES, total_padded_bytes - i * CHUNK_BYTES)
  len_lba_aligned = round_up_to_lba(len)
  slba = data_start_lba + chunks[i] * chunk_lba_count
```

### C++ FDP write_lfcs

입력:

```text
object_name
pinned_addr
header_bytes
payload_bytes
total_padded_bytes
layer
expert
iter
generation
```

동작:

```text
1. expert_slot 계산
2. logical_bytes = HEADER_BLOCK_BYTES + payload_bytes
3. needed_chunks 계산
4. chunks[] allocate
5. header stamp
6. chunks[] 순서대로 expert_data_ring에 NVMe write submit
7. completion 확인
8. old LFCS chunks free
9. old LFSY chunks free/reset
10. lfcs_entries[expert_slot] 갱신
11. lfsy_entries[expert_slot] invalid/reset
12. return total_padded_bytes
```

주의:

```text
- write_lfcs()가 old LFSY reset/free를 담당한다.
- file backend에서는 이 의미가 "old lossy file unlink + fresh lossy open"으로 구현되어 있다.
- Phase 3 전에는 Python이 아직 _open_fresh_lossy()를 호출할 수 있으므로 FDP backend에서도 open_lfsy가 곧 필요하다.
```

### fc_manager.py 변경

현재 non-expert write 후 `path.stat().st_size`를 읽는다. FDP backend에서는 POSIX file이
없으므로 반환값을 사용한다.

권장 변경:

```python
written = proposal_io.write_prptens_from_pinned(...)
_rec.file_bytes = int(written)
```

file backend에서도 이 값은 기존 file size와 같은 padded write size로 쓸 수 있다.

### expert_state.py 변경

`_ExpertEntry`에 generation 필드를 추가한다.

```python
lossless_generation: int = 0
```

`_do_lossless_fc()`에서 LFCS write 전에 generation을 증가시킬지, 성공 후 증가시킬지
정해야 한다.

권장:

```text
new_generation = ent.lossless_generation + 1
write_lossless_from_pinned(..., generation=new_generation)
write 성공 후 ent.lossless_generation = new_generation
```

file backend는 이 값이 파일 포맷에 들어가지 않아도 된다. FDP backend는 stale LFSY
append 방지에 사용한다.

### 완료 기준

```text
- file backend regression 없음
- FDP raw backend에서 NEFC object write 가능
- FDP raw backend에서 LFCS object write 가능
- metadata nefc.latest / lfcs_entries[slot] 갱신 확인 가능
- old NEFC/LFCS chunks free 확인 가능
- fc_manager.py가 FDP backend에서 path.stat()에 의존하지 않음
- debug raw read로 written chunk의 magic/header probe 가능
```

## Phase 3 - LFSY handle 전환

### 목적

현재 LFSY 상태는 OS fd인 `_ExpertEntry.lossy_fd`를 중심으로 동작한다. FDP backend에는
fd가 없으므로 "append handle" 의미로 일반화한다.

### Python 상태 변경

단계적 변경을 위해 필드명을 바로 삭제하지 않는다.

```python
@dataclass
class _ExpertEntry:
    ...
    lossy_fd: Optional[int] = None
    lossless_generation: int = 0
```

문서/주석 의미를 바꾼다.

```text
file backend:
  lossy_fd == OS file descriptor

FDP backend:
  lossy_fd == C++ FDP append handle
```

나중에 별도 cleanup phase에서 `lossy_fd`를 `lossy_handle`로 rename한다.

### proposal_io.py wrapper signature

`open_lossy_log_for_append()`:

```python
def open_lossy_log_for_append(
    path: Path,
    header_bytes: bytes,
    *,
    layer: Optional[int] = None,
    expert: Optional[int] = None,
    step: Optional[int] = None,
    generation: Optional[int] = None,
    mode: int = 0o644,
) -> int:
    if backend_kind() == "file":
        return int(_cext.open_lossy_log_for_append(...))
    if backend_kind() == "fdp":
        if layer is None or expert is None or generation is None:
            raise RuntimeError("FDP LFSY open requires layer/expert/generation")
        return int(_get_fdp().open_lfsy(
            str(Path(path).name),
            header_bytes,
            int(layer),
            int(expert),
            int(step if step is not None else -1),
            int(generation),
        ))
```

`append_lossy_log_from_pinned()`:

```python
def append_lossy_log_from_pinned(
    fd: int,
    pinned_tensor: torch.Tensor,
    total_bytes: int,
    *,
    layer: int,
    expert: int,
    step: int,
    generation: Optional[int] = None,
) -> int:
    total_padded = _round_up_sector(total_bytes)
    if backend_kind() == "file":
        return int(_cext.append_lossy_log(...))
    if backend_kind() == "fdp":
        if generation is None:
            raise RuntimeError("FDP LFSY append requires generation")
        return int(_get_fdp().append_lfsy(
            int(fd),
            int(pinned_tensor.data_ptr()),
            int(total_bytes),
            int(total_padded),
            int(layer),
            int(expert),
            int(step),
            int(generation),
        ))
```

file backend는 추가 kwargs를 무시한다.

### expert_state.py callsite 변경

`_open_fresh_lossy()`:

```python
L, E = ent.expert_id
ent.lossy_fd = open_lossy_log_for_append(
    ent.lossy_path,
    header,
    layer=int(L),
    expert=int(E),
    step=int(step),
    generation=int(ent.lossless_generation),
)
```

`_do_lossy_fc_gpu_aos()`:

```python
append_lossy_log_from_pinned(
    int(ent.lossy_fd),
    blob,
    int(spec.blob_padded_bytes),
    layer=int(L),
    expert=int(E),
    step=int(step),
    generation=int(ent.lossless_generation),
)
```

`_do_lossless_fc()` ordering:

```text
1. close current lossy handle if exists
2. new_generation = ent.lossless_generation + 1
3. write LFCS with new_generation
4. on success ent.lossless_generation = new_generation
5. unlink old lossy path compatibility call
6. open fresh LFSY with current generation
```

### C++ FDP open_lfsy

입력:

```text
object_name
header_bytes
layer
expert
start_iter
generation
```

동작:

```text
1. expert_slot 계산
2. lfsy_entries[slot]가 valid이면 old chunks free
3. first chunk allocate
4. 64KiB header를 staging/aligned buffer에 stamp
5. expert_data_ring으로 header write
6. lfsy_entries[slot] 초기화
     valid = 1
     generation = generation
     start_iter = start_iter
     logical_tail = HEADER_BLOCK_BYTES
     chunk_count = 1
     chunks[0] = first_chunk
7. append handle 생성
8. return handle
```

append handle 구현은 단순 map이면 충분하다.

```text
handle -> {expert_slot, generation}
```

single-rank MVP에서는 process-local integer handle로 충분하다.

### C++ FDP append_lfsy

입력:

```text
handle
pinned_addr
total_bytes
total_padded
layer
expert
step
generation
```

동작:

```text
1. handle lookup
2. expert_slot/generation 일치 확인
3. lfsy_entries[slot] valid 확인
4. append logical_tail 위치 계산
5. 현재 chunk 남은 공간 확인
6. 부족하면 새 chunk allocate 후 chunks[]에 append
7. expert_data_ring으로 total_padded bytes write
8. logical_tail += total_padded
9. metadata image update
10. return total_padded
```

`logical_tail` 정책은 file backend와 recovery parser를 고려해 padded append size를
따른다. 현재 file append path도 `total_bytes`를 4KiB padding해 write하고, reader는
padding 구간에서 `count==0`을 만나면 멈추는 구조다.

```text
logical_tail += total_padded
return total_padded
```

실제 유효 record byte 수는 block prefix의 `count`로 parser가 판단한다.

### fsync_fd / close_fd / unlink_path

`fsync_fd(handle)`:

```text
file backend:
  POSIX fsync

FDP backend:
  flush_metadata(current_iter or 0)
```

`close_fd(handle)`:

```text
file backend:
  POSIX close

FDP backend:
  close/remove append handle
  LFSY metadata와 chunks는 유지
```

`unlink_path(path)`:

```text
file backend:
  POSIX unlink

FDP backend:
  compatibility no-op
```

FDP chunk free는 path unlink가 아니라 `write_lfcs()`, `open_lfsy()`, NEFC replacement
같은 metadata transition에서만 한다.

### Metadata full flush 연결

Phase 3 말에는 end-of-training과 fault injection 직전 flush가 metadata image를 raw
metadata LBA range에 반영해야 한다.

연결 지점:

```text
ExpertState.flush()
  -> 기존 raw queue drain
  -> open lossy handles fsync_fd()

proposal_shutdown / proposal trigger fault recovery 직전
  -> 기존 _fc.expert_state.flush()
```

FDP backend에서 `fsync_fd()`가 metadata flush를 수행하면 이 기존 call path를 재사용할
수 있다. 다만 장기적으로는 명시적인 `proposal_io.flush_metadata(iter)` 호출이 더
명확하다.

### 완료 기준

```text
- file backend LFSY open/append/flush/close regression 없음
- FDP raw backend open_lfsy가 handle을 반환함
- FDP raw backend append_lfsy가 logical_tail과 chunks[]를 갱신함
- LFCS write 후 old LFSY chunks가 reset/free됨
- fsync_fd(handle)가 metadata_ring으로 full metadata flush를 수행함
- unlink_path(path)는 FDP에서 잘못된 chunk free를 하지 않음
```

## Pre-Recovery 종료 상태

Phase 3 완료 시점의 상태는 다음과 같다.

```text
가능:
  - file backend 기존 동작
  - FDP raw NVMe session open/probe
  - 3개 io_uring_cmd ring 생성
  - FDP metadata image 생성/flush
  - FDP bitmap allocate/free
  - NEFC raw write
  - LFCS raw write
  - LFSY raw open/append
  - LFCS 교체 시 old LFCS/LFSY chunks free/reset
  - NEFC 교체 시 old NEFC chunks free

아직 불가능:
  - recovery.py가 FDP metadata에서 checkpoint를 읽어 복구
  - crash 중간 recovery
  - multi-rank shared namespace 안전성
  - advanced FDP PID placement tuning
```

즉 Phase 3까지는 "raw NVMe write-side object store" 완성이 목표다. 다음 문서인
recovery phase 설계에서는 `recovery.py`가 file scan 대신 FDP metadata query와 object
range read를 사용하도록 확장한다.

## 권장 테스트

### Unit

```text
fdp_metadata layout:
  - configured expert slots 계산
  - LFCS/LFSY entry offset 계산
  - metadata image size 4KiB alignment

bitmap:
  - allocate/free
  - double free detect
  - capacity exceeded

chunk mapping:
  - chunk_id -> raw namespace LBA
  - logical_bytes -> needed_chunks
  - LFSY logical_tail boundary case

passthru guards:
  - range overlap reject
  - namespace capacity overflow reject
  - unaligned addr/length reject
  - missing ALLOW_FORMAT reject
```

### Raw smoke

```text
PROPOSAL_IO_BACKEND=fdp
PROPOSAL_FDP_DEVICE=/dev/ngXnY
PROPOSAL_FDP_NSID=...
PROPOSAL_FDP_METADATA_START_LBA=...
PROPOSAL_FDP_DATA_START_LBA=...
PROPOSAL_FDP_DATA_NLB=...
PROPOSAL_FDP_ALLOW_FORMAT=1

1. open_session_from_env()
2. identify/probe 결과 stats() 확인
3. debug_raw_write_read_smoke()
4. flush_metadata()
5. debug_read_metadata_summary()
```

### Integration without recovery

```text
PROPOSAL_IO_BACKEND=fdp

1. write_prptens_from_pinned() 호출
2. write_lossless_from_pinned() 호출
3. open_lossy_log_for_append() 호출
4. append_lossy_log_from_pinned() 여러 번 호출
5. flush_metadata()
6. debug_read_metadata_summary()로 metadata 상태 확인
```

확인 항목:

```text
- NEFC latest valid
- LFCS slot valid
- LFSY slot valid
- LFSY logical_tail 증가
- chunk bitmap allocated count 예상값과 일치
- LFCS 재작성 후 이전 LFCS/LFSY chunk가 free됨
- debug raw read로 object header magic 확인 가능
```

### Regression

```text
PROPOSAL_IO_BACKEND=file
  - 기존 proposal_io file write tests
  - expert_state LFSY append/read tests
  - non-expert PRPTENS load tests
```
