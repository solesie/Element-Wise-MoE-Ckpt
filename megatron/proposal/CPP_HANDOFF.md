# C++ Zero-Copy SSD Passthrough — 인터페이스 핸드오프 설계

Python (Megatron-DeepSpeed proposal CP) ↔ C++ (io_uring + O_DIRECT NVMe writer)

이 문서는 Python 의 `os.write` syscall 을 io_uring submit 으로 교체할 때 참고할 **명세** 다. 함께 읽어야 할 파일: [`binary_io.py`](binary_io.py), [`expert_state.py`](expert_state.py) 의 docstring.

---

## 1. 전체 아키텍처 개요

```
                          training thread (main_stream)
                          │ forward / backward / optimizer launch 만 수행
                          ▼
              ┌──────────┴──────────┐
              │  arm post-iter FC   │  (cuda.Event record + global stash)
              └──────────┬──────────┘
                         │ 다음 iter 의 mb0 forward 끝에서 release
                         ▼
       ┌─────────────────┼─────────────────┐
       │                 │                 │
       ▼                 ▼                 ▼
  host_writer_      host_writer_       (tracker_stream
  non_expert        expert              kernel launches)
  worker thread     worker thread
       │                 │
       │ pinned + meta   │ pinned + meta
       ▼                 ▼
    [C++ submit ring NE] [C++ submit ring EXP]
       │                 │
       └────────┬────────┘
                ▼
       C++ io_uring + O_DIRECT
                │
                ▼          (ack)
       [ack ring] ─────
                            │
                            ▼
                       reaper (Python)
                       pinned 메모리 해제
```

C++ 는 **in-process** (shared library, ctypes/pybind11 로 load) 로 동작하는 것이 기본. Torch 가 pinned 를 관리하므로 in-process 가 단순. (별도 프로세스 격리가 필요하다면 pinned allocator 자체를 별도 구현해야 함...)

대충 지금 python thread는 non-expert worker thread, expert worker thread 있는데, uring 쓸때 c쪽에서 더 생성해도 된다고 봄.

---

## 2. 세 가지 파일 종류

| Magic | 종류 | 빈도 | Atomic? | Filename |
|---|---|---|---|---|
| `PRPTENS` | Non-expert FC | 매 W iter | 덮어쓰기 | `fc_nonexpert_iter{N:08d}.bin` |
| `LFCL` | Lossless expert FC | 매 D iter × F% expert | `.tmp` + fsync + rename, 이건 direct 적용되면 user level file system에서 직접 처리 가능 | `fc_expert_L{L:02d}_E{E:02d}.bin` |
| `LFSY` | Lossy log (append-only) | 매 iter × ~30 expert | O_APPEND | `lossy_expert_L{L:02d}_E{E:02d}.bin` |

`HEADER_BLOCK_BYTES = 65536(64KiB)` 세 종류 모두 동일 (파일 선두에 zero-padded header sector). On-disk 파일 크기 = `HEADER_BLOCK_BYTES + payload_bytes` (pin memory 단위를 chunk와 일치하게 64~128MB 단위로 설정 가능하게 잡고 뒤는 padding으로 파일에 쓰이지 않음).

### 2.1 PRPTENS layout (Non-expert FC)

```
0                                                             
├─ HEADER_BLOCK_BYTES = 65536
├── MAGIC 8B ── "PRPTENS\0"
├── VERSION 4B (u32 LE = 2)
├── HEADER_LEN 4B (JSON byte 길이)
├── JSON body — { entries:[{name,dtype,shape,offset,nbytes}, ...], scalars:{step:..} }
├── HEADER_BLOCK_BYTES 까지 zero pad
│
└─ payload  ── entries 선언 순서대로 fp32 tensor 들의 연속

(pinned alloc 만, 파일에는 없음):
└─ chunk pad ── 다음 64 MiB 까지 zero 채움
```

### 2.2 LFCL layout (per-expert lossless)

```
0                                                              ├─ ├─ HEADER_BLOCK_BYTES = 65536
├── MAGIC 4B ── "LFCL"
├── VERSION 4B (u32 LE)
├── PICKLE_LEN 4B
├── pickle blob ── { expert_id:[L,E], step, tensors:[{name,shape,numel},...],
│                   global_offsets:{name→fp32_offset}, total_numel,
│                   layout:"param_m_v_concat_fp32" }
├── HEADER_BLOCK_BYTES 까지 zero pad
│
└─ payload  ── schema 순서대로 각 tensor 의:
                  param      (numel × 4B fp32)
                  exp_avg    (numel × 4B fp32)
                  exp_avg_sq (numel × 4B fp32)
```

해당 expert 가 아직 한 번도 token 을 받지 않아 Adam state 가 미초기화일 때, `exp_avg` / `exp_avg_sq` 는 zero 로 채움. on-disk byte 길이는 schema 와 항상 일치하므로 복구는 무조건 tensor 당 `numel × 12 B` 를 읽으면 됨.

### 2.3 LFSY layout (per-expert lossy log)

파일 생성 시 한 번 `open_lossy_log_for_append` 호출로 64 KiB 헤더 블록 기록. 이후 append 는 다음 블록 단위:

```
0                                          BLOCK_PREFIX_BYTES + N×RECORD_BYTES
├─ count u32 LE  — 이 block 의 record 수
├─ step  u32 LE  — 이 record 들을 만든 iteration
├─ records[0..N] — 각 (idx i64, p f32, m f32, v f32) packed = 20 byte
```

`idx` 는 per-expert combined offset (`tname_base + idx_local`). i32 는 per-expert (param+m+v) numel 이 ~2.1 B 를 넘는 큰 MoE 에서 silently overflow 하므로 i64 로 고정.

복구는 block 의 `count` 를 사용해 다음 block boundary 를 찾으며 순차 walk. `step` 은 rolling-step marker (informational + 진단 로그 용도) — record 적용에는 사용되지 않음 (file 순서 = chronological 순서).

---

## 3. Pinned-buffer 계약

### 3.1 Alignment 보장 (Python 측)

| Buffer | Size class | `data_ptr() % 4096` | 비고 |
|---|---|---|---|
| PRPTENS pinned (수 GB) | ≥ page | **0** ✓ | torch caching allocator 는 큰 alloc 에서 page-aligned 반환 |
| LFCL pinned (per expert, 수 MB) | ≥ page | **0** ✓ | 동일 |
| LFSY blob pinned (가변, KB ~ MB) | 4 KiB 로 round-up | **0** ✓ | `alloc_pinned_uint8_sector_aligned()` 가 alloc 크기를 다음 sector 로 padding + tail zero-fill |

### 3.2 C++ 에 전달되는 pinned layout

```
[ data_ptr() ]
       │
       ▼
+──────────────────────────+
│ HEADER (64 KiB 고정)      │   ← magic + ver + meta + zero pad
+──────────────────────────+
│                          │
│   payload (write data)   │   ← C++ 가 [HEADER + payload] 만 파일에 기록
│                          │
+──────────────────────────+   ← total_bytes (실제 파일에 쓸 크기)
│                          │
│  chunk pad (zero)        │   ← pinned-only, io_uring chunk submit 정렬용
│                          │
+──────────────────────────+   ← chunk_padded (pinned alloc 전체 크기)
```

- `total_bytes` = 파일에 flush 할 byte 수 (= `HEADER + payload`)
- `chunk_padded` = pinned alloc 크기. PRPTENS/LFCL 의 경우 64 MiB 배수
- 항상: `data_ptr() % 4096 == 0`, `total_bytes` 는 4 byte 배수

LFSY append blob 의 경우:
- `chunk_padded` = blob 크기를 다음 4 KiB 까지 round-up (작은 값)
- `total_bytes` = `BLOCK_PREFIX_BYTES + count × RECORD_BYTES` (4 KiB-align 보장 안 됨)
- O_DIRECT 를 위해 C++ 는 per-expert chunk buffer 에 append 를 누적해야 함 (file offset alignment 가 중요한 부분이지 buffer 크기가 아님)

### 3.3 Pinned buffer 의 소유권 / 수명 — 누가 free 하나?

**Python 이 소유. C++ 는 read-only 로 잠시 빌릴 뿐.**

```
Python worker                                C++
─────────────                                ───
1. pinned alloc (torch caching allocator)
2. pinned 채우기 (D2H + header stamp)
3. submit({buf_id, ptr, size, meta})  ──►   enqueue
                                            io_uring_submit
   (Python 은 _inflight_refs[buf_id]
    = pinned tensor 의 ref 유지)            ...
                                            io_uring_wait_cqe
4. ack ring drain        ◄──── ack(buf_id, status)
5. _inflight_refs.pop(buf_id)
   → Python 의 마지막 ref drop
   → torch caching pinned allocator 가
     해당 block 을 free-list 에 회수
```

**근거**:
1. `cudaHostAlloc` ↔ `cudaFreeHost` 는 CUDA driver 내부. Torch 의 caching pinned allocator 가 그 위에 자체 free-list 를 유지. C++ 는 두 layer 모두에 visibility 가 없어 절대 free 호출 금지.
2. Buffer 재사용은 자동 — Python 이 ref 를 drop 하면 torch allocator 가 block 을 회수해 다음 `torch.empty(..., pin_memory=True)` 가 그 메모리를 재활용. 명시적 free 불필요.
3. Cross-thread 안전성: submit (step 3) 과 ack (step 4) 사이엔 **오직 C++ 만** buffer 를 (read-only) 접근. Python 은 이 구간에 ref 를 유지함으로써 torch allocator 가 회수하지 못하게 보장 — torch 의 caching allocator 는 ref 가 살아있는 pinned block 을 절대 회수하지 않음.

**실패 모드**:
- C++ crash → ack 없음: Python 의 `_inflight_refs` 가 stale entry 를 영원히 보유. 완화책 = timeout sweep (예: 30 s) 으로 ref drop + leak 로그. Pinned host RAM 은 크고 (~200 GB on test box) 약간의 leak 으로 학습이 죽진 않지만 진단 기록 필요.
- Python crash → C++ 는 ring 포인터를 보유 중이지만, OS 가 process exit 시 pinned region 회수. C++ 는 SIGPIPE / heartbeat timeout 으로 감지 후 종료.

---

## 4. Helper API (Python → C++ swap-in 지점)

`binary_io.py` 의 helper 들은 `megatron/proposal/` 전체에서 **유일한** file-write 호출 지점이다. C++ 측이 필요로 할 routing metadata 가 모두 명시되어 있음:

```python
write_prptens_from_pinned(
    final_path,         # 대상 file path
    pinned_mv,          # pinned host buffer 의 memoryview (data_ptr 가 4 KiB-aligned)
    header_bytes,       # ≤ HEADER_BLOCK_BYTES — pinned[:HEADER_BLOCK_BYTES] 에 stamp 됨
    payload_bytes,      # tensor payload 크기; 파일 크기 = HEADER + payload
    *,
    step,               # iteration 번호 — routing / 로깅
    mode=0o644,
    fsync=False,
)

write_lossless_from_pinned(
    final_path, pinned_mv, header_bytes, payload_bytes,
    *,
    layer, expert, step,   # (L, E, iter) — LBA 관리 + per-expert atomic
    mode=0o644,
)
# 구현: `<final_path>.tmp` 에 write → fsync → renameat2() 로 최종 path 교체

append_lossy_log_from_pinned(
    fd, pinned_mv, total_bytes,
    *,
    layer, expert, step,   # (L, E, iter) — per-expert chunk buffer 라우팅
)
```

C++ 로 swap 시 각 helper 안의 `os.write` 호출이 다음과 같이 교체됨:

```c
ProposalSubmitSlot slot = {
    .buf_id      = next_id(),
    .pinned_base = pinned_mv.obj,
    .total_bytes = total,
    .file_type   = LFCL_TYPE,
    .layer       = layer,
    .expert      = expert,
    .step        = step,
    .path        = ...,
    .flags       = ATOMIC_RENAME,
};
proposal_cp_submit_expert(&slot);
```

Python helper 는 worker 의 caller chain (`fc_manager._collect_param_payload` → `_last_collect_pinned_info` → expert_state worker closure) 을 통해 pinned ref 를 계속 들고 있음. C 측 ack 가 ref drop 트리거.

---

## 5. Submit / completion ring (권장)

Lock-free SPSC ring buffer, head/tail 은 lockless atomic:

```c
// Submit
struct ProposalSubmitSlot {
    uint64_t buf_id;
    void*    pinned_base;
    uint64_t total_bytes;     // 파일에 write 할 byte 수
    uint64_t chunk_padded;    // pinned alloc 크기 (mem 힌트)
    uint32_t file_type;       // 0=PRPTENS, 1=LFCL, 2=LFSY_OPEN, 3=LFSY_APPEND
    int32_t  layer;           // 해당 없음 시 -1
    int32_t  expert;          // 해당 없음 시 -1
    uint32_t step;
    int32_t  fd;              // LFSY_APPEND 용; 그 외 -1
    uint32_t flags;           // bit0=fsync, bit1=atomic_rename
    char     path[256];       // null-terminated; atomic_rename 시 tmp_path
};

// Completion
struct ProposalCompleteSlot {
    uint64_t buf_id;
    int32_t  status;          // 0=ok, <0=-errno
    uint32_t bytes_written;
};
```

Submit ring 은 2 개 (NE + Expert, Python 의 두 worker domain 과 매칭) — SPSC 유지. Ack ring 은 통합 1 개 (Python reaper 가 `buf_id` 로 분기) 면 충분.

---

## 6. Read 경로 (recovery)

현재 Python 아직 그대로 file system 유지:
- `recovery.recover()` 가 `os.pread` + `np.frombuffer` + `ThreadPoolExecutor(max_workers=8)` 로 per-expert 병렬 로드.

```python
proposal_cp_submit_read({
    .buf_id, .pinned_dst, .total_bytes, .path
})
# ack 까지 block
```

C++ 가 `IORING_OP_READ` 로 Python alloc 한 pinned destination 에 직접 read (numpy view 로 이어 사용). 

---

---

## 7. 볼만한 파일

| 파일 | 봐야 할 것 |
|---|---|
| [binary_io.py](binary_io.py) | 세 개의 `_from_pinned` helper, `HEADER_BLOCK_BYTES`, `PIN_CHUNK_BYTES`, `SECTOR_BYTES`, `alloc_pinned_uint8_sector_aligned`, `build_prptens_header_block` |
| [expert_state.py](expert_state.py) | 최상단의 LFCL/LFSY 포맷 docstring, `_pack_header`, `_RECORD_DTYPE`, `_do_lossless_fc`, `_do_lossy_fc_batch_write` (pinned blob 이 만들어지는 곳) |
| [fc_manager.py](fc_manager.py) | `_collect_param_payload` (`header_block_bytes` + `chunk_pad_bytes` + `preserve_order` 옵션으로 pinned alloc), `_last_collect_pinned_info` |
| [importance_tracker.py](importance_tracker.py) | `_dispatch_pending_payload` (lossy data 경로의 SoA planar pinned) |
| [cp_hooks.py](cp_hooks.py) | `_collect_helper` (LFCL collect 에 header_block_bytes / chunk_pad_bytes / preserve_order 를 wire) |
| [recovery.py](recovery.py) | Read path — fault recovery 에서만 사용, write hot path 와 무관 |

**세 개의 `_from_pinned` helper 를 거치지 않은 byte 는 절대 file 에 쓰지 않을 것.** 
