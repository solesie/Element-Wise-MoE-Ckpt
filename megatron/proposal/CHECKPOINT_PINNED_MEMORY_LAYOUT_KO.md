# Checkpoint Pinned Memory Layout

이 문서는 proposal checkpoint writer가 사용하는 CPU pinned memory 구조를
정리한다. 기준 코드는 `proposal_io.py`, `fc_manager.py`, `expert_state.py`다.

공통 상수:

```text
HEADER_BLOCK_BYTES = 65536        # 64 KiB fixed header block
PIN_CHUNK_BYTES    = 64 MiB 기본값 # PRPTENS/LFCL pinned allocation padding 단위
SECTOR_BYTES       = 4096         # O_DIRECT alignment 기준
```

공통 규칙:

```text
파일에 쓰는 byte 범위 = [header block][payload]
pinned allocation 끝의 chunk/sector padding은 파일에 쓰지 않는다.
```

## 1. Non-Expert FC: PRPTENS

Non-expert FC는 rank의 non-expert parameter와 Adam state를 하나의
`PRPTENS` 파일로 저장한다.

파일명 예:

```text
fc_nonexpert_iter00000010.bin
```

### 1.1 GPU to CPU Pinned Copy 직후

`fc_manager._collect_nonexpert_payload()`가 pinned buffer 앞 64 KiB를
비워두고, 그 뒤에 non-expert tensor payload를 연속으로 채운다.

```text
GPU non-expert tensors

  param / exp_avg / exp_avg_sq ...
        |
        | d2h_stream, non_blocking GPU -> CPU pinned copy
        v
```

```text
CPU pinned buffer
GPU -> CPU copy 완료 직후

data_ptr()
  |
  v
+==============================================================+ offset 0
| reserved header block                                        |
| 64 KiB = HEADER_BLOCK_BYTES                                  |
|                                                              |
| 아직 header 없음                                             |
| writer thread에서 PRPTENS header를 stamp 예정                |
+==============================================================+ offset 65536
| tensor[0] raw fp32 bytes                                     |
| entry[0] payload offset = 0                                  |
+--------------------------------------------------------------+
| tensor[1] raw fp32 bytes                                     |
| entry[1] payload offset = tensor[0] nbytes                   |
+--------------------------------------------------------------+
| tensor[2] raw fp32 bytes                                     |
+--------------------------------------------------------------+
| ...                                                          |
+--------------------------------------------------------------+
| tensor[N-1] raw fp32 bytes                                   |
+==============================================================+ offset 65536 + payload_bytes
| chunk pad                                                    |
| 64 MiB chunk alignment용                                     |
| pinned allocation에만 존재, 파일에는 쓰지 않음               |
+==============================================================+ total_pinned_bytes
```

### 1.2 Header Stamp 이후

`proposal_io.write_prptens_from_pinned()`가 `pinned[0:HEADER_BLOCK_BYTES]`에
header를 넣고, `pinned[0:HEADER_BLOCK_BYTES + payload_bytes]`만 파일에 쓴다.

```text
CPU pinned buffer / PRPTENS file layout

data_ptr()
  |
  v
+==============================================================+ offset 0
| MAGIC                                                        |
| 8B: b"PRPTENS\0"                                             |
+--------------------------------------------------------------+ offset 8
| VERSION                                                      |
| 4B uint32 little-endian, 현재 2                              |
+--------------------------------------------------------------+ offset 12
| HEADER_LEN                                                   |
| 4B uint32 little-endian, JSON header 길이                    |
+--------------------------------------------------------------+ offset 16
| HEADER_JSON                                                  |
| {                                                            |
|   "entries": [                                               |
|     {                                                        |
|       "name": "...",                                         |
|       "dtype": "float32",                                    |
|       "shape": [...],                                        |
|       "offset": 0,                                           |
|       "nbytes": ...                                          |
|     },                                                       |
|     ...                                                      |
|   ],                                                         |
|   "scalars": {"step": ...}                                   |
| }                                                            |
+--------------------------------------------------------------+
| zero padding                                                 |
| 64 KiB header block 끝까지 0으로 채움                        |
+==============================================================+ offset 65536
| tensor[0] raw fp32 bytes                                     |
+--------------------------------------------------------------+
| tensor[1] raw fp32 bytes                                     |
+--------------------------------------------------------------+
| tensor[2] raw fp32 bytes                                     |
+--------------------------------------------------------------+
| ...                                                          |
+--------------------------------------------------------------+
| tensor[N-1] raw fp32 bytes                                   |
+==============================================================+ offset 65536 + payload_bytes
| chunk pad                                                    |
| pinned-only, file write 범위 밖                              |
+==============================================================+
```

실제 파일:

```text
+--------------------+---------------------------+
| 64 KiB PRPTENS hdr | non-expert FC tensor data |
+--------------------+---------------------------+
```

### 1.3 파라미터 의미

| 이름 | 의미 |
|---|---|
| `MAGIC` | 파일 형식 식별자. `b"PRPTENS\0"`이면 PRPTENS raw binary 파일이다. |
| `VERSION` | PRPTENS format version. 현재 `2`다. |
| `HEADER_LEN` | 뒤따르는 JSON header의 byte 길이. |
| `HEADER_JSON` | tensor metadata와 scalar metadata를 담는 UTF-8 JSON. |
| `entries` | payload 안에 들어 있는 tensor 목록. |
| `entries[i].name` | tensor 이름. 예: `module.weight:param`, `module.weight:exp_avg`. |
| `entries[i].dtype` | 저장 dtype. 현재 collect 경로는 fp32 payload 기준이다. |
| `entries[i].shape` | tensor 복구 시 사용할 원래 shape. |
| `entries[i].offset` | payload 시작점 기준 상대 byte offset. 파일 절대 offset이 아니다. |
| `entries[i].nbytes` | 해당 tensor의 raw byte 크기. |
| `scalars.step` | checkpoint를 만든 training iteration. |
| `payload_bytes` | 모든 non-expert tensor raw bytes의 합. |
| `total_pinned_bytes` | header + payload + chunk pad까지 포함한 pinned allocation 크기. |
| `chunk pad` | C++/O_DIRECT chunk submit 정렬용 padding. 현재 파일에는 쓰지 않는다. |

Offset 관계:

```text
실제 tensor 파일 위치 = HEADER_BLOCK_BYTES + entries[i].offset
                    = 65536 + entries[i].offset
```

## 2. Expert Lossless FC: LFCL

Expert lossless FC는 특정 `(layer, expert)` 하나의 full snapshot을 저장한다.
payload는 schema 순서대로 각 tensor의 `param | exp_avg | exp_avg_sq`를 가진다.

파일명 예:

```text
fc_expert_L03_E07.bin
```

### 2.1 GPU to CPU Pinned Copy 직후

```text
GPU tensors for one expert

  tensor A param
  tensor A exp_avg
  tensor A exp_avg_sq

  tensor B param
  tensor B exp_avg
  tensor B exp_avg_sq

  ...
        |
        | d2h_stream, GPU -> CPU pinned copy
        v
```

```text
CPU pinned buffer for LFCL
one expert lossless snapshot

data_ptr()
  |
  v
+==============================================================+ offset 0
| reserved header block                                        |
| 64 KiB = HEADER_BLOCK_BYTES                                  |
|                                                              |
| 아직 header 없음                                             |
| writer thread에서 LFCL header를 stamp 예정                   |
+==============================================================+ offset 65536
| tensor A : param raw fp32 bytes                              |
+--------------------------------------------------------------+
| tensor A : exp_avg raw fp32 bytes                            |
| Adam state가 없으면 zero placeholder                         |
+--------------------------------------------------------------+
| tensor A : exp_avg_sq raw fp32 bytes                         |
| Adam state가 없으면 zero placeholder                         |
+--------------------------------------------------------------+
| tensor B : param raw fp32 bytes                              |
+--------------------------------------------------------------+
| tensor B : exp_avg raw fp32 bytes                            |
+--------------------------------------------------------------+
| tensor B : exp_avg_sq raw fp32 bytes                         |
+--------------------------------------------------------------+
| ...                                                          |
+--------------------------------------------------------------+
| tensor N : param raw fp32 bytes                              |
+--------------------------------------------------------------+
| tensor N : exp_avg raw fp32 bytes                            |
+--------------------------------------------------------------+
| tensor N : exp_avg_sq raw fp32 bytes                         |
+==============================================================+ offset 65536 + payload_bytes
| chunk pad                                                    |
| pinned allocation에만 존재, 파일에는 쓰지 않음               |
+==============================================================+ total_pinned_bytes
```

### 2.2 Header Stamp 이후

`expert_state._pack_header(MAGIC_LFCL, header_meta)`가 LFCL header를 만들고,
`proposal_io.write_lossless_from_pinned()`가 header를 pinned buffer 앞에 stamp한다.

```text
CPU pinned buffer / LFCL file layout

data_ptr()
  |
  v
+==============================================================+ offset 0
| MAGIC                                                        |
| 4B: b"LFCL"                                                  |
+--------------------------------------------------------------+ offset 4
| VERSION                                                      |
| 4B uint32 little-endian                                      |
+--------------------------------------------------------------+ offset 8
| PICKLE_LEN                                                   |
| 4B uint32 little-endian                                      |
+--------------------------------------------------------------+ offset 12
| pickle metadata blob                                         |
| {                                                            |
|   "expert_id": [layer_id, expert_id],                        |
|   "step": ...,                                               |
|   "tensors": [                                               |
|     {"name": ..., "shape": [...], "numel": ...},             |
|     ...                                                      |
|   ],                                                         |
|   "global_offsets": {name -> fp32_offset},                   |
|   "total_numel": ...,                                        |
|   "layout": "param_m_v_concat_fp32"                          |
| }                                                            |
+--------------------------------------------------------------+
| zero padding                                                 |
| 64 KiB header block 끝까지 0으로 채움                        |
+==============================================================+ offset 65536
| tensor A : param                                             |
+--------------------------------------------------------------+
| tensor A : exp_avg                                           |
+--------------------------------------------------------------+
| tensor A : exp_avg_sq                                        |
+--------------------------------------------------------------+
| tensor B : param                                             |
+--------------------------------------------------------------+
| tensor B : exp_avg                                           |
+--------------------------------------------------------------+
| tensor B : exp_avg_sq                                        |
+--------------------------------------------------------------+
| ...                                                          |
+==============================================================+ offset 65536 + payload_bytes
| chunk pad                                                    |
| pinned-only, file write 범위 밖                              |
+==============================================================+
```

실제 파일:

```text
+-------------------+---------------------------------------+
| 64 KiB LFCL header | one expert lossless FC tensor payload |
+-------------------+---------------------------------------+
```

### 2.3 파라미터 의미

| 이름 | 의미 |
|---|---|
| `MAGIC` | 파일 형식 식별자. `b"LFCL"`이면 expert lossless FC 파일이다. |
| `VERSION` | expert state format version. 현재 `FORMAT_VERSION = 1`이다. |
| `PICKLE_LEN` | 뒤따르는 pickle metadata blob의 byte 길이. |
| `expert_id` | `[layer_id, expert_id]` 쌍. 어떤 expert의 snapshot인지 나타낸다. |
| `step` | lossless snapshot을 만든 training iteration. |
| `tensors` | 이 expert snapshot에 포함되는 tensor schema 목록. |
| `tensors[i].name` | tensor 이름. |
| `tensors[i].shape` | 복구 시 사용할 원래 tensor shape. |
| `tensors[i].numel` | 해당 tensor의 element 개수. |
| `global_offsets` | tensor 이름별 fp32 element offset. LFSY record의 global index와 같은 좌표계다. |
| `total_numel` | 이 expert schema의 전체 fp32 element 수. |
| `layout` | payload 배치 방식. 현재 `"param_m_v_concat_fp32"`. |
| `payload_bytes` | `total_numel * 3 * 4`에 해당하는 param/m/v raw bytes. 코드상 실제 값은 collect 결과를 따른다. |
| `zero placeholder` | Adam state가 아직 없는 expert도 schema 크기를 유지하기 위해 `exp_avg`, `exp_avg_sq` 자리에 쓰는 0 값. |
| `chunk pad` | pinned allocation 정렬용 padding. 파일에는 쓰지 않는다. |

## 3. Expert Lossy FC: LFSY

Expert lossy FC는 특정 `(layer, expert)`의 sparse update append log다.
처음 파일을 만들 때 64 KiB LFSY header를 쓰고, 이후 iteration마다
`[count, step, records...]` block을 append한다.

파일명 예:

```text
lossy_expert_L03_E07.bin
```

### 3.1 파일 생성 시 Header

`open_lossy_log_for_append()`가 새 LFSY 파일을 만들고 header block을 먼저 쓴다.

```text
LFSY file after open_fresh_lossy

+==============================================================+ offset 0
| MAGIC                                                        |
| 4B: b"LFSY"                                                  |
+--------------------------------------------------------------+ offset 4
| VERSION                                                      |
| 4B uint32 little-endian                                      |
+--------------------------------------------------------------+ offset 8
| PICKLE_LEN                                                   |
| 4B uint32 little-endian                                      |
+--------------------------------------------------------------+ offset 12
| pickle metadata blob                                         |
| expert id, tensor schema, global offsets, etc.               |
+--------------------------------------------------------------+
| zero padding                                                 |
| 64 KiB header block 끝까지 0으로 채움                        |
+==============================================================+ offset 65536
| append block들이 이후부터 순서대로 붙음                      |
+==============================================================+
```

### 3.2 Per-Iteration Append Blob Assembly

lossy update는 tensor별 SoA 형태의 `idx_local`, `p`, `m`, `v`를 받아서 writer
thread에서 하나의 pinned uint8 blob으로 AoS packing한다.

```text
Per-expert lossy updates for one iteration

  tensor A:
    idx_local[]  p[]  m[]  v[]

  tensor B:
    idx_local[]  p[]  m[]  v[]

  ...
        |
        | writer thread
        | SoA -> AoS packing into pinned uint8 blob
        v
```

```text
Pinned append blob for one expert, one iteration

data_ptr()
  |
  v
+==============================================================+ offset 0
| count                                                        |
| 4B uint32 little-endian                                      |
| 이 append block에 들어 있는 record 수                        |
+--------------------------------------------------------------+ offset 4
| step                                                         |
| 4B uint32 little-endian                                      |
| 이 block을 만든 training iteration                           |
+==============================================================+ offset 8
| record[0]                                                    |
| idx int32 | p fp32 | m fp32 | v fp32                         |
| 16B                                                          |
+--------------------------------------------------------------+
| record[1]                                                    |
| idx int32 | p fp32 | m fp32 | v fp32                         |
+--------------------------------------------------------------+
| ...                                                          |
+--------------------------------------------------------------+
| record[count-1]                                              |
| idx int32 | p fp32 | m fp32 | v fp32                         |
+==============================================================+ offset 8 + count * 16
| sector pad                                                   |
| 4 KiB alignment용                                            |
| pinned allocation에만 존재, append write에는 쓰지 않음       |
+==============================================================+ sector-aligned allocation size
```

실제 append write:

```text
append_lossy_log_from_pinned()
  writes pinned[0 : 8 + count * 16]
```

LFSY 파일은 시간이 지나면서 다음처럼 커진다.

```text
LFSY file layout

+------------------+------------------+------------------+------------------+
| 64 KiB header    | append block t=10 | append block t=11 | append block t=12 |
+------------------+------------------+------------------+------------------+

append block:

+----------+--------+------------------------------+
| count 4B | step 4B| count x 16B AoS records      |
+----------+--------+------------------------------+
```

### 3.3 파라미터 의미

| 이름 | 의미 |
|---|---|
| `MAGIC` | 파일 형식 식별자. `b"LFSY"`이면 expert lossy FC append log다. |
| `VERSION` | expert state format version. 현재 `FORMAT_VERSION = 1`이다. |
| `PICKLE_LEN` | 뒤따르는 pickle metadata blob의 byte 길이. |
| `count` | 해당 append block에 들어 있는 sparse record 수. |
| `step` | 해당 append block을 만든 training iteration. 복구 시 순서 진단과 chronological marker로 쓰인다. |
| `idx_local` | tensor 내부 local element index. append blob에는 그대로 저장하지 않고 `global_offsets[tensor_name]`를 더한다. |
| `global_offsets[tensor_name]` | expert 전체 flat 좌표계에서 해당 tensor가 시작하는 fp32 element offset. |
| `record.idx` | expert 전체 flat 좌표계의 element index. `idx_local + global_offsets[tensor_name]`. |
| `record.p` | 해당 index의 parameter value 또는 delta 적용에 필요한 parameter 값. fp32. |
| `record.m` | 해당 index의 Adam `exp_avg` 값. fp32. |
| `record.v` | 해당 index의 Adam `exp_avg_sq` 값. fp32. |
| `RECORD_BYTES` | record 하나의 크기. `idx i32 + p f32 + m f32 + v f32 = 16B`. |
| `BLOCK_PREFIX_BYTES` | append block prefix 크기. `count u32 + step u32 = 8B`. |
| `blob_bytes` | 실제 append할 byte 수. `8 + count * 16`. |
| `sector pad` | small pinned uint8 allocation을 4 KiB aligned로 만들기 위한 padding. append write에는 포함하지 않는다. |

복구 관점:

```text
1. LFCL full snapshot을 먼저 읽는다.
2. LFSY append block을 파일 순서대로 순차 replay한다.
3. 같은 record.idx가 여러 번 나오면 뒤에 replay된 값이 최종값이다.
```
