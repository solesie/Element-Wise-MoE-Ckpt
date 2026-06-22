# FDP C Module - 현재 proposal_io 구조 연동 설계

이 문서는 현재 repo의 proposal checkpoint write/read 구조에 FDP SSD raw
namespace backend를 붙이기 위한 구현 설계다. 이전 설계의 기능 목표는 유지한다.
즉 POSIX filesystem을 범용으로 대체하지 않고, proposal checkpoint 형식
`PRPTENS`, `LFCL`, `LFSY`만 아는 user-level object store를 만든다.

현재 코드 기준 핵심 진입점은 `binary_io.py`가 아니라 다음 파일들이다.

```text
megatron/proposal/proposal_io.py
  - Python facade
  - torch C++ extension JIT load
  - pinned allocation
  - PRPTENS header build / read parser
  - write_prptens_from_pinned()
  - write_lossless_from_pinned()
  - open_lossy_log_for_append()
  - append_lossy_log_from_pinned()
  - fsync_fd() / close_fd() / unlink_path()
  - read_file_range() / file_size()

megatron/proposal/_csrc/binary_io.cpp
  - 현재 모든 POSIX open/write/fsync/rename/unlink/pread/stat 구현 위치

megatron/proposal/_csrc/proposal_io.cpp
  - pybind11 binding

megatron/proposal/fc_manager.py
  - non-expert PRPTENS writer 호출

megatron/proposal/expert_state.py
  - expert LFCL writer
  - expert LFSY open/append/flush/close 상태 관리

megatron/proposal/recovery.py
  - 현재 POSIX cp_dir scan + read_file_range 기반 recovery
```

따라서 FDP 구현은 단순히 `_csrc/binary_io.cpp`의 `::write()`만 바꾸는 작업이
아니다. `proposal_io.py`에 file/FDP backend switch를 두고, `ExpertState`의
fd 중심 LFSY 상태와 `recovery.py`의 file scan 경로를 FDP metadata 기반 경로로
확장해야 한다.

## 목표와 비목표

### 목표

```text
Python proposal code
  -> proposal_io.py backend switch
  -> file backend 또는 proposal_fdp_store C++ backend
  -> io_uring_cmd + NVMe passthru
  -> FDP SSD raw namespace
```

FDP backend는 다음 object type만 지원한다.

| Object | 현재 파일 형식 | 의미 |
|---|---|---|
| `NEFC` | `PRPTENS` | non-expert full checkpoint |
| `LFCS` | `LFCL` | expert lossless full snapshot |
| `LFSY` | `LFSY` | expert lossy append log |

물리 data section은 object type별로 고정 분할하지 않는다. 하나의 data pool을
64MiB chunk 단위로 관리하고, global bitmap이 free/allocated만 기록한다. 어떤
chunk가 어떤 checkpoint object에 속하는지는 metadata descriptor의 `chunks[]`가
기록한다.

```text
single data pool:
  chunk 0  chunk 1  chunk 2  chunk 3  chunk 4  ...
   LFCS     free     LFSY     NEFC     LFCS

bitmap:
     1       0        1        1        1
```

### 비목표

| 포함 | 제외 |
|---|---|
| `proposal_io.py` backend switch | mount 가능한 POSIX filesystem |
| raw namespace LBA write/read | 일반 파일 API 전체 호환 |
| single data pool + global bitmap | directory/permission/rename semantics 일반화 |
| `NEFC/LFCS/LFSY` metadata 관리 | 기본 모드의 power-loss/torn-write 자동 복구 |
| `LFSY` per-expert append write pointer | FDP PID 자동 최적화 |
| pinned memory direct write 시도 | 모든 payload의 완전 zero-copy 강제 |
| recovery용 metadata lookup/read API | 기존 `cp_dir` scan을 FDP에서 그대로 흉내내기 |

Crash-safe metadata commit은 선택 확장으로 다룬다. 기본 연구 구현은 정상
종료/명시적 metadata flush 이후의 recovery를 목표로 한다.

## 현재 write path와 FDP hook 위치

### PRPTENS / NEFC

현재 non-expert FC는 `fc_manager.py`에서 pinned buffer payload를 준비한 뒤
`proposal_io.write_prptens_from_pinned()`를 호출한다.

```text
fc_manager._fire_nonexpert_fc()
  -> proposal_io.build_prptens_header_block(...)
  -> proposal_io.write_prptens_from_pinned(path, pinned_tensor, header, payload_bytes, step)
  -> _cext.write_prptens(...)
  -> binary_io.cpp::write_prptens(...)
  -> POSIX open/write/close
```

FDP backend에서는 같은 Python wrapper가 다음으로 분기한다.

```text
proposal_io.write_prptens_from_pinned(...)
  if backend == "file":
      _cext.write_prptens(...)
  if backend == "fdp":
      _fdp.write_nefc(
          object_name=basename(final_path),
          rank=session.rank,
          iter=step,
          format=PRPTENS,
          pinned_tensor,
          header_bytes,
          payload_bytes,
      )
```

`final_path`는 FDP에서 실제 파일 경로가 아니라 debug/name-index key로만 쓴다.
기존 `fc_manager.py`가 write 후 `path.stat()`으로 file size를 읽는 부분은 FDP
backend에서 동작하지 않으므로, 구현 시에는 `write_prptens_from_pinned()`의 반환
byte 수를 `WriteRecord.file_bytes`에 사용하도록 바꿔야 한다.

### LFCL / LFCS

현재 expert lossless FC는 `expert_state.py`의 raw worker에서 CUDA event를 기다린
뒤 `proposal_io.write_lossless_from_pinned()`를 호출한다.

```text
ExpertState._do_lossless_fc()
  -> close current lossy fd
  -> write_lossless_from_pinned(path, pinned_tensor, header, payload_bytes, layer, expert, step)
  -> unlink old lossy file
  -> _open_fresh_lossy()
```

FDP backend에서는 `write_lossless_from_pinned()`가 `write_lfcs()`로 분기한다.

```text
proposal_io.write_lossless_from_pinned(...)
  if backend == "file":
      _cext.write_lossless(...)
  if backend == "fdp":
      _fdp.write_lfcs(
          object_name=basename(final_path),
          rank=session.rank,
          layer_id=layer,
          expert_id=expert,
          generation=entry.lossless_generation,
          iter=step,
          format=LFCL,
          pinned_tensor,
          header_bytes,
          payload_bytes,
      )
```

현재 wrapper signature에는 `generation`이 없다. FDP 구현 시 `_ExpertEntry`에
`lossless_generation`을 추가하고 `write_lossless_from_pinned(..., generation=...)`
optional kwarg를 추가한다. file backend는 이 값을 무시한다.

LFCS write가 성공하면 FDP metadata는 해당 expert slot의 LFCS descriptor를 새
chunk list로 교체하고, 같은 expert의 이전 LFSY generation도 reset/free한다. 이
동작은 현재 file backend의 "LFCL atomic rename 후 old LFSY unlink 후 fresh LFSY
open"과 같은 의미다.

### LFSY open/append

현재 LFSY는 OS fd 중심이다.

```text
_ExpertEntry.lossy_fd: Optional[int]

_open_fresh_lossy()
  -> proposal_io.open_lossy_log_for_append(path, header)
  -> returns OS fd

_do_lossy_fc_gpu_aos()
  -> append_lossy_log_from_pinned(fd, blob, bytes, layer, expert, step)

flush()
  -> fsync_fd(fd)

shutdown()
  -> close_fd(fd)
```

FDP backend에서는 OS fd가 없다. 하지만 Python callsite churn을 줄이기 위해
"fd" 이름만 "append handle" 의미로 확장한다.

구현 변경:

```python
@dataclass
class _ExpertEntry:
    ...
    lossy_handle: Optional[int] = None   # file backend: fd, FDP backend: handle id
    lossless_generation: int = 0
```

호환을 위해 단계적으로는 필드명을 `lossy_fd`에서 바로 바꾸지 않고,
`lossy_handle` alias를 도입한 뒤 callsite를 정리한다.

새 wrapper signature:

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
    ...

def append_lossy_log_from_pinned(
    handle: int,
    pinned_tensor: torch.Tensor,
    total_bytes: int,
    *,
    layer: int,
    expert: int,
    step: int,
    generation: Optional[int] = None,
) -> int:
    ...
```

file backend는 `handle`을 OS fd로 사용한다. FDP backend는 `handle`을 C module
내부 LFSY session id로 사용하거나 `(expert_slot, generation)`을 packing한 정수로
사용한다.

FDP open:

```text
open_lossy_log_for_append(..., layer, expert, generation, header)
  -> expert_slot = layer * num_experts_per_layer + expert
  -> allocate first data chunk
  -> write 64KiB LFSY header object bytes
  -> initialize lfsy_entries[expert_slot]
       valid = 1
       generation = generation
       logical_tail = HEADER_BLOCK_BYTES
       chunks[0] = first chunk
       chunk_count = 1
  -> return append_handle
```

FDP append:

```text
append_lossy_log_from_pinned(handle, blob, total_bytes, layer, expert, step, generation)
  -> lookup handle 또는 expert_slot
  -> stale generation check
  -> write [count u32][step u32][records...] block
  -> if current chunk lacks space, allocate new chunk and append to chunks[]
  -> update logical_tail
  -> metadata image update
```

`fsync_fd(handle)` and `close_fd(handle)`:

| Backend | `fsync_fd` | `close_fd` |
|---|---|---|
| file | POSIX fsync | POSIX close |
| FDP 기본 | optional metadata flush 또는 no-op | close handle/no-op |
| FDP crash-safe | full metadata image commit + optional NVMe flush | close handle/no-op |

FDP backend에서 metadata snapshot을 iteration마다 보장하려면 `fsync_fd()`에
의존하기보다 `proposal_io.flush_metadata(iter)`를 명시 API로 추가하는 편이
낫다. `ExpertState.flush()`와 proposal shutdown path가 이 API를 호출하게 만든다.

## Backend 선택과 초기화

`--proposal-cp-target`은 현재 `disk|dram`으로 expert state 구현을 고르는 옵션이다.
FDP는 "disk target의 storage backend"이므로 이 옵션을 바로 오버로드하지 않는
편이 안전하다.

권장:

```text
--proposal-cp-target disk
--proposal-storage-backend file|fdp
```

초기 구현에서는 CLI를 추가하기 전 환경변수로도 충분하다.

```text
PROPOSAL_IO_BACKEND=file  # default
PROPOSAL_IO_BACKEND=fdp
PROPOSAL_FDP_DEVICE=/dev/ngXnY
PROPOSAL_FDP_NAMESPACE_LBA_OFFSET=...
PROPOSAL_FDP_RANK=...
PROPOSAL_FDP_WORLD_SIZE=...
```

`proposal_io.py` import 시점에는 torch extension을 JIT load하고 있다. FDP backend도
같은 `_cext`에 붙이면 import-time dependency가 커진다. 권장 구조는 file backend
extension과 FDP extension을 분리해 lazy load하는 것이다.

```text
proposal_io.py
  _file_cext = load(... binary_io.cpp, proposal_io.cpp ...)
  _fdp_cext = None

  def _get_fdp():
      if _fdp_cext is None:
          load(... proposal_fdp_store.cpp ...)
          _fdp_cext.open_session(...)
      return _fdp_cext
```

이렇게 해야 FDP device/kernel 권한이 없는 환경에서도 기존 tests와 file backend가
깨지지 않는다.

## C/C++ module 구조

새 backend는 기존 `_csrc/binary_io.cpp`에 무리하게 넣지 말고 별도 파일로 둔다.

```text
megatron/proposal/_csrc/
  binary_io.cpp          # 기존 POSIX file backend 유지
  binary_io.h
  proposal_io.cpp        # 기존 file backend pybind
  fdp_store.h            # FDP public C++ API
  fdp_store.cpp          # session, metadata, allocator
  fdp_metadata.h         # packed/POD metadata structs
  fdp_passthru.cpp       # io_uring_cmd + NVMe passthru
  proposal_fdp_io.cpp    # pybind module for FDP backend
```

`proposal_fdp_store` 역할:

```text
proposal_fdp_store
  ├─ device/session manager
  ├─ global bitmap allocator
  ├─ metadata image manager
  ├─ NEFC/LFCS/LFSY descriptor manager
  ├─ pinned/page buffer adapter
  ├─ NVMe passthru command builder
  ├─ io_uring_cmd submit/completion
  ├─ recovery metadata reader
  └─ stats/error reporting
```

## io_uring topology

현재 repo의 `AsyncWriter`는 세 worker를 가진다.

```text
AsyncWriter
  ├─ ne_writer
  │    - non-expert PRPTENS write 담당
  │
  ├─ expert_launcher
  │    - expert GPU work launch 담당
  │    - storage write는 직접 하지 않음
  │
  └─ expert_writer
       - expert LFCL write
       - expert LFSY append 담당
```

현재 구현에는 io_uring이 아직 없고, `ne_writer`와 `expert_writer` 모두
`_csrc/binary_io.cpp`의 synchronous POSIX `write()` 경로를 호출한다. FDP backend는
rank/process당 다음 세 개의 ring을 두는 구조를 권장한다.

```text
rank process
  proposal_fdp_store session
    ├─ nefc_data_ring
    │    - PRPTENS / NEFC 큰 sequential write 전용
    │    - 주 caller: ne_writer
    │
    ├─ expert_data_ring
    │    - LFCL / LFCS write
    │    - LFSY append write
    │    - 주 caller: expert_writer
    │
    └─ metadata_ring
         - metadata image full flush
         - optional NVMe flush
         - superblock/global metadata write
```

세 ring을 분리하는 이유는 write 특성이 다르기 때문이다.

| Ring | I/O 특성 | 분리 이유 |
|---|---|---|
| `nefc_data_ring` | 수백 MiB ~ 수 GiB sequential write | 큰 NEFC write가 expert append를 밀어내지 않게 함 |
| `expert_data_ring` | LFCS 중간 크기 write + LFSY 작은 append | expert checkpoint latency를 NEFC와 분리 |
| `metadata_ring` | 작고 빈번한 metadata image write / flush | metadata commit이 data queue 뒤에 갇히지 않게 함 |

MVP부터 C++ 내부 구조는 세 ring 이름으로 나누고, 각 ring은 raw NVMe
`io_uring_cmd` passthru submit 경로를 가진다. 초기 구현에서는 submit 직후 completion을
기다리는 synchronous contract를 유지해도 되지만, I/O backend 자체는 raw NVMe
passthru가 필수다.

```text
MVP raw NVMe mode:
  nefc_data_ring.submit(write)    -> io_uring_cmd NVMe write
  expert_data_ring.submit(write)  -> io_uring_cmd NVMe write
  metadata_ring.submit(write)     -> io_uring_cmd NVMe metadata write / flush
```

ring ownership은 `proposal_fdp_store` session이 가진다. Python worker thread가 ring을
직접 소유하지 않는다. Python은 `proposal_io.*_from_pinned()`를 호출하고, C++ session이
해당 request type에 맞는 ring으로 submit/wait한다.

초기 synchronous completion contract:

```text
Python worker
  -> proposal_io.write_prptens_from_pinned(...)
  -> C++ nefc_data_ring submit
  -> C++ completion wait
  -> metadata update
  -> return
```

나중에 C++ 내부 background completion으로 확장할 수 있지만, 그 경우에는 Python
pinned tensor object refcount를 completion까지 잡는 별도 lifetime protocol이 필요하다.

blockfile backend는 필수 phase가 아니다. 일반 CI나 metadata unit test를 위해 선택적
debug fallback으로 둘 수는 있지만, FDP backend의 기준 구현은 raw NVMe
`io_uring_cmd` passthru다. 따라서 Phase 1부터 `fdp_passthru.cpp`가 device
open/probe, ring setup, NVMe read/write/flush command builder를 제공해야 한다.

## Namespace layout

FDP raw namespace는 다음 logical layout을 가진다.

```text
LBA 0
│
├─ [ superblock ]
│      - magic/version
│      - lba_size
│      - chunk_size = 64MiB
│      - metadata_start_lba
│      - metadata_pages
│      - data_pool_start_lba
│      - data_pool_nlb
│      - rank/world/rank_partition info
│
├─ [ metadata image ]
│      - global config / counters
│      - global data bitmap
│      - NEFC latest descriptor
│      - LFCS packed entries[num_moe_layers * num_experts_per_layer]
│      - LFSY packed entries[num_moe_layers * num_experts_per_layer]
│
└─ [ data pool ]
       ├─ 64MiB chunk 0
       ├─ 64MiB chunk 1
       ├─ 64MiB chunk 2
       └─ ...
```

Multi-rank에서 하나의 raw namespace를 공유하면 bitmap double allocation이 가장 큰
데이터 손상 위험이다. MVP는 다음 중 하나로 제한한다.

| Mode | 설명 |
|---|---|
| single-rank | 한 process만 allocator 사용 |
| static rank partition | rank별 data_pool range를 superblock/config로 분리 |
| allocator daemon | 별도 process가 chunk allocation 전담 |

초기 구현 권장은 static rank partition이다. 각 rank가 자기 partition의 bitmap만
갱신하면 interprocess lock 없이도 중복 할당을 피할 수 있다.

## Metadata 구조

### Chunk list

모든 checkpoint object는 하나 이상의 64MiB chunk로 표현한다. metadata에는 LBA,
NLB, logical offset 같은 상세 범위 구조를 저장하지 않고 chunk id 배열만 저장한다.

```c
typedef uint32_t FdpChunkId;
```

physical LBA는 chunk id로 계산한다.

```c
chunk_lba = data_pool_start_lba + chunk_id * chunk_nlb;
```

object는 logical offset 0부터 chunk 순서대로 빈틈 없이 저장한다. 따라서 chunk별
used bytes는 저장하지 않는다. entry의 `logical_bytes` 또는 `logical_tail` 하나로
마지막 chunk의 유효 범위를 계산한다.

```text
chunks[0] -> logical offset 0
chunks[1] -> logical offset CHUNK_BYTES
chunks[2] -> logical offset 2 * CHUNK_BYTES

valid bytes in chunk i:
  min(CHUNK_BYTES, logical_bytes - i * CHUNK_BYTES)
```

이 계산이 성립하려면 chunk 하나를 여러 object가 공유하지 않아야 한다. MVP에서는
object가 할당받은 chunk를 독점한다.

### NEFC

NEFC는 최신 non-expert PRPTENS 하나만 recovery에 필요하다.

```c
#define FDP_CHUNK_INVALID UINT32_MAX

struct NefcEntry {
    uint8_t valid;
    uint32_t rank;
    uint64_t iter;
    uint64_t logical_bytes;
    uint32_t chunk_count;
    FdpChunkId chunks[MAX_CHUNKS_NEFC];
};

struct NefcMetadata {
    uint8_t has_latest;
    struct NefcEntry latest;
};
```

새 NEFC write가 완료되면 `latest`를 교체한다. 기본 모드에서는 이전 NEFC chunks를
즉시 free한다. Crash-safe mode에서는 metadata commit 이후 pending-free로 넘긴다.

### LFCS

LFCS entry는 `(layer_id, expert_id)` slot별 최신 LFCL descriptor다.

```text
expert_slot = layer_id * num_experts_per_layer + expert_id
NUM_EXPERT_SLOTS = num_moe_layers * num_experts_per_layer
```

```c
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
```

LFCS entries는 4KiB page 하나에 slot 하나를 고정하지 않는다. fixed-size packed
array로 둔다.

```text
lfcs_entry_offset = lfcs_table_off + expert_slot * sizeof(struct LfcsEntry)
```

### LFSY

LFSY entry는 `(layer_id, expert_id)` slot별 현재 append log 상태다.

```c
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

`logical_tail`은 LFSY object 안의 다음 logical byte offset이다. open 직후에는
`HEADER_BLOCK_BYTES`다. append는 `[count u32][step u32][records...]` block을
`logical_tail` 위치에 추가하고 tail을 증가시킨다.

`wp_lba`와 `wp_chunk_id`는 저장하지 않는다. append 시점에 `logical_tail`과
`chunks[]`로 계산한다.

```c
chunk_index = logical_tail / CHUNK_BYTES;
offset_in_chunk = logical_tail % CHUNK_BYTES;

if (chunk_index == chunk_count) {
    allocate_new_chunk();
}

chunk_id = chunks[chunk_index];
wp_lba = data_pool_start_lba
       + chunk_id * chunk_nlb
       + offset_in_chunk / lba_size;
```

LFSY entries도 fixed-size packed array로 둔다.

```text
lfsy_entry_offset = lfsy_table_off + expert_slot * sizeof(struct LfsyEntry)
```

## Metadata DRAM image와 full flush

metadata는 runtime에 하나의 flat aligned image로 resident 상태를 유지한다.
heap pointer를 SSD에 쓰지 않는다.

```text
metadata_image_base  = 4KiB aligned
metadata_image_size  = round_up(metadata_physical_bytes, 4KiB)
section_base         = 4KiB aligned
flush granularity    = whole metadata image
```

```text
DRAM metadata image

4KiB aligned base
│
├─ page 0    [ superblock mirror / global header ]
├─ page ...  [ global config / counters ]
├─ page ...  [ global data bitmap ]
├─ page ...  [ NEFC metadata ]
├─ page ...  [ LFCS packed entry table ]
├─ page ...  [ LFSY packed entry table ]
└─ page N    [ padding / reserved ]
```

LFCS/LFSY는 packed fixed-size entry table이다. section base는 4KiB aligned로
두지만, entry 하나를 4KiB page 하나에 고정하지 않는다.

기본 flush 정책:

```text
payload write completion
  -> metadata image update

iteration end or explicit flush
  -> write entire metadata_image to metadata region
  -> optional NVMe flush
```

초기 API:

```python
proposal_io.flush_metadata(iter: int) -> None
proposal_io.fdp_stats() -> Dict[str, Any]
```

## Static capacity

동적 metadata allocator는 recovery와 crash handling을 복잡하게 만든다. 초기 구현은
정적 상한으로 fail-fast한다.

```c
#define MAX_EXPERT_SLOTS  65536
#define MAX_CHUNKS_NEFC   4096
#define MAX_CHUNKS_LFCS   32
#define MAX_CHUNKS_LFSY   96
```

의미:

| 상한 | 의미 |
|---|---|
| `MAX_EXPERT_SLOTS=65536` | `num_moe_layers * num_experts_per_layer` 최대값 |
| `MAX_CHUNKS_NEFC=4096` | NEFC 최대 256GiB 표현 가능, 64MiB chunk 기준 |
| `MAX_CHUNKS_LFCS=32` | expert LFCL 하나 최대 2GiB |
| `MAX_CHUNKS_LFSY=96` | expert LFSY generation 하나 최대 6GiB |

chunk id는 4B이므로 metadata는 이전 상세 범위 구조 설계보다 훨씬 작다. 예를 들어
`MAX_CHUNKS_LFCS=32`, `MAX_CHUNKS_LFSY=96`이면 slot당 chunk id 배열은
`(32 + 96) * 4B = 512B`다. fixed fields를 더해도 slot당 대략 1KiB 안쪽으로
잡을 수 있다.

실험 모델이 `num_moe_layers=16`, `experts_per_layer=48`이면 slot은 768개이고,
LFCS/LFSY packed table은 대략 1MiB 수준이다. 이 정도면 매 iteration metadata
전체 flush가 합리적이다.

`MAX_EXPERT_SLOTS=65536` 같은 상한을 그대로 쓰면 metadata image가 수십 MiB까지
커질 수 있다. MVP는 실제 model config로 `num_moe_layers * experts_per_layer`만큼
metadata table을 잡고, compile-time max는 capacity check 용도로만 사용한다.
Crash-safe double-buffer metadata mode는 physical metadata region이 대략 2배
필요하다.

## Pinned memory contract

현재 `proposal_io.alloc_pinned_uint8_sector_aligned()`는 torch pinned tensor를
over-allocate 후 4KiB aligned slice로 반환한다. C++ file backend는
`int(pinned_tensor.data_ptr())`만 받아 동기 `write()`를 끝낸 뒤 반환한다. 이
동기 계약에서는 Python stack에 `pinned_tensor` 참조가 살아 있으므로 안전하다.

FDP backend도 MVP에서는 같은 synchronous completion contract를 유지한다.

```text
Python worker
  -> cuda_event.synchronize()
  -> proposal_io.*_from_pinned(...)
  -> C++ submit passthru writes
  -> C++ wait completion
  -> return to Python
```

이 경우 C++가 Python object refcount를 별도로 잡지 않아도 된다. 단, C++ 내부
background thread로 완전히 비동기화하려면 wrapper signature를 pointer int가 아니라
`py::object` 또는 buffer protocol 기반으로 바꾸고 completion 전까지 ref를 잡아야 한다.

```text
async C++ mode requirement:
  PyObject_GetBuffer()
  Py_INCREF(original object)
  io completion 이후 PyBuffer_Release(), Py_DECREF()
```

alignment 정책:

| 구간 | 처리 |
|---|---|
| large aligned body | pinned buffer direct write |
| 64KiB header | pinned 앞에 stamp 후 direct write 가능 |
| LBA-unaligned tail | C staging/bounce buffer 또는 padded write |
| metadata image | posix_memalign + mlock/io_uring registered buffer |

## Recovery 설계

현재 recovery는 `cp_dir`를 scan해서 최신 non-expert file과 expert pair를 찾는다.
FDP backend에는 directory가 없으므로 별도 metadata query/read path가 필요하다.

새 API:

```python
proposal_io.backend_kind() -> "file" | "fdp"

proposal_io.fdp_get_latest_nefc(t_fault: int) -> FdpObjectRef
proposal_io.fdp_list_expert_slots() -> List[Tuple[int, int]]
proposal_io.fdp_get_lfcs(layer: int, expert: int, t_fault: int) -> FdpObjectRef
proposal_io.fdp_get_lfsy(layer: int, expert: int, generation: int) -> FdpObjectRef
proposal_io.fdp_read_object_range(ref: FdpObjectRef, offset: int, nbytes: int) -> bytes
```

`recovery.recover()` 분기:

```text
if proposal_io.backend_kind() == "file":
    existing cp_dir scan path

if proposal_io.backend_kind() == "fdp":
    1. get latest NEFC <= t_fault from metadata
    2. read PRPTENS header/payload by object range
    3. for every model expert slot:
         read LFCS object
         read matching LFSY object
         replay LFSY blocks exactly like current read_expert_state()
    4. apply payload dict to model/optimizer
```

파서 로직은 최대한 재사용한다. 현재 `proposal_io.load_binary(path)`는 path 기반이다.
FDP용으로는 bytes/range reader를 주입할 수 있게 분리한다.

```python
def parse_prptens_from_reader(read_range, logical_size): ...
def read_expert_state_from_reader(lfcs_reader, lfsy_reader): ...
```

file backend는 `read_range = proposal_io.read_file_range(path, off, n)`로 감싸고,
FDP backend는 `fdp_read_object_range(ref, off, n)`으로 감싼다.

## unlink/name semantics

현재 file backend는 old NEFC/LFSY를 `unlink_path()`로 지운다. FDP backend에서는
object lifetime을 metadata replacement가 관리한다.

정책:

| 호출 | file backend | FDP backend |
|---|---|---|
| old NEFC unlink | POSIX unlink | no-op 또는 exact object name index delete |
| old LFSY unlink after LFCS | POSIX unlink | `write_lfcs()` 성공 시 old LFSY chunks free/reset |
| fresh LFSY open | create/truncate file | initialize LFSY entry generation |

FDP backend의 `unlink_path()`는 compatibility no-op으로 시작한다. 실제 free는
NEFC/LFCS/LFSY metadata transition에서 처리한다. 이렇게 해야 path string과 현재
metadata owner가 불일치할 때 잘못된 chunk를 free하지 않는다.

## Crash-safe metadata commit 선택 확장

기본 구현은 crash 중간 recovery를 보장하지 않는다. 확장 시에는 data-before-metadata
ordering과 metadata image double-buffering을 결합한다.

원칙:

```text
metadata에 publish되지 않은 data chunk는 존재하지 않는 것으로 본다.
metadata에 publish된 data chunk만 valid checkpoint로 본다.
```

metadata image 전체는 physical copy A/B를 가진다. MVP의 full metadata flush
정책을 유지하면 page별 dirty tracking 없이 image 단위 seq/crc만으로 시작할 수
있다.

```text
metadata image
  copy A: seq=100, crc valid, old metadata image
  copy B: seq=99,  crc valid, older metadata image

commit new metadata:
  write inactive copy B with seq=101 + crc(new metadata image)
  wait completion
  recovery chooses valid copy with highest seq
```

old chunk free 순서:

```text
1. allocate new chunks
2. write new payload
3. wait payload completion
4. update DRAM metadata image
5. write inactive metadata image copy
6. wait metadata completion
7. move old chunks to free/pending-free
```

Recovery에서 bitmap은 persisted bitmap을 신뢰하기보다 valid metadata descriptors를
순회해 재구성하는 것이 단순하다.

```text
1. bitmap all free
2. mark NEFC.latest chunks
3. mark all valid LFCS chunks
4. mark all valid LFSY chunks
5. unmarked data chunks are free/orphan
```

## 구현 단계

### Phase 0: 문서/이름 정리

- `binary_io.py` 전제 제거. 실제 facade는 `proposal_io.py`로 통일한다.
- stale test import가 있으면 `megatron.proposal.proposal_io` 기준으로 정리한다.
- `CHECKPOINT_PINNED_MEMORY_LAYOUT_KO.md`의 `binary_io.*` 표현도 같은 기준으로 맞춘다.

### Phase 1: backend switch와 raw NVMe FDP session bootstrap

- `proposal_io.py`에 `_backend_kind()` 추가.
- file backend는 현재 `_cext` 호출 그대로 유지.
- FDP backend lazy loader 골격 추가.
- raw NVMe device open/probe 구현.
- `nefc_data_ring`, `expert_data_ring`, `metadata_ring` 생성.
- NVMe passthru read/write/flush command builder 구현.
- metadata image, bitmap allocator, chunk id -> LBA mapping 검증.

### Phase 2: NEFC/LFCS write 분기

- `write_prptens_from_pinned()` FDP 분기 추가.
- `write_lossless_from_pinned(..., generation=...)` optional kwarg 추가.
- `_ExpertEntry.lossless_generation` 추가.
- `fc_manager.py`는 write return bytes를 stats에 반영하게 수정.
- `path.stat()` 의존을 FDP에서 제거.

### Phase 3: LFSY handle 전환

- `_ExpertEntry.lossy_fd`를 `lossy_handle` 의미로 일반화.
- `open_lossy_log_for_append()`에 `layer/expert/step/generation` optional kwargs 추가.
- file backend는 기존 fd 동작 유지.
- FDP backend는 handle/session id 반환.
- append에서 stale generation check를 C module에 추가.
- `fsync_fd/close_fd/unlink_path` FDP semantics 구현.

### Phase 4: recovery 분기

- PRPTENS parser와 expert LFCL/LFSY replay를 path 기반에서 reader 기반으로 분리.
- `recovery.recover()`에 FDP metadata query branch 추가.
- raw NVMe object range read로 end-to-end checkpoint/recovery test 작성.

### Phase 5: raw NVMe passthru hardening

- startup probe 강화:
  - device open 권한
  - namespace LBA size
  - FDP support/PID list
  - `io_uring_cmd` support
- FDP placement directive/PID 적용 검증.
- queue depth, batching, registered buffer 최적화.
- metadata flush와 data write completion ordering 검증.

### Phase 6: crash-safe option

- metadata A/B copy layout 추가.
- metadata image seq/crc validation 구현.
- recovery bitmap rebuild 구현.
- old chunk pending-free/GC 구현.

## 예상 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| global bitmap multi-process race | 같은 chunk 중복 할당, 데이터 손상 | MVP는 single-rank 또는 static rank partition |
| path 기반 callsite 잔존 | FDP backend에서 stat/scan 실패 | write return bytes 사용, recovery metadata branch 추가 |
| LFSY fd 의미 누수 | FDP append 식별 불가 | `lossy_handle`로 추상화, open에 layer/expert/generation 전달 |
| pinned lifetime | async C++ mode에서 buffer 재사용 | MVP는 sync completion, async 전환 시 py object ref 보유 |
| D2H 완료 전 write | 깨진 payload 저장 | 현재 worker의 `cuda_event.synchronize()` contract 유지 |
| chunk cap 초과 | 큰 checkpoint write 실패 | fail-fast error, 이후 overflow chunk table 검토 |
| FDP 환경 의존성 | 장비/커널/권한 없으면 import 실패 | FDP lazy load + file backend fallback |
| crash 중간 metadata 불일치 | recovery 실패 | 기본 범위에서 미지원 명시, A/B commit option 제공 |

## 결론

현재 구조에서도 동일 기능의 FDP object store 구현은 가능하다. 다만 작업 단위는
기존 `binary_io.cpp`의 write syscall swap이 아니라, `proposal_io.py`에 backend
switch를 두고 `ExpertState`의 LFSY fd 상태와 `recovery.py`의 path scan을 object
metadata 기반으로 확장하는 것이다.

가장 현실적인 MVP는 다음이다.

```text
file backend 유지
  +
single-rank 또는 static rank partition
  +
rank/process당 3-ring raw NVMe io_uring_cmd topology
  - nefc_data_ring
  - expert_data_ring
  - metadata_ring
  +
synchronous FDP write completion
  +
crash-safe commit 제외
```

blockfile은 선택적 debug fallback일 수 있지만 필수 구현 경로는 아니다. 이후 단계에서는
recovery, raw NVMe passthru hardening, crash-safe metadata commit을 붙인다.
