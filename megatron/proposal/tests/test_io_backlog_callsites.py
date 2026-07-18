import json
import struct
from pathlib import Path
from types import SimpleNamespace

import torch

from megatron.proposal.storage.expert_writer import (
    BLOCK_PREFIX_BYTES,
    HEADER_BLOCK_BYTES,
    ExpertState,
    _ExpertEntry,
)
from megatron.proposal.storage.fc_manager import FCManager
from megatron.proposal.telemetry.io_backlog import (
    _reset_io_backlog_trace_for_tests,
    close_io_backlog_trace,
)


def _write_dicts(metadata, batch):
    return [
        dict(zip(metadata["write_fields"], values))
        for values in batch["writes"]
    ]


def test_checkpoint_callsites_emit_exact_object_sizes(monkeypatch, tmp_path):
    monkeypatch.setenv("PROPOSAL_IO_BACKLOG_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("PROPOSAL_CP_TARGET", "test_dram")
    monkeypatch.setenv("RANK", "6")
    monkeypatch.setenv("PROPOSAL_FDP_EP_SIZE", "4")
    monkeypatch.setenv("PROPOSAL_FDP_EXPERTS_PER_LAYER", "128")
    monkeypatch.setenv("PROPOSAL_FDP_EXPERT_ID_BASE", "64")
    _reset_io_backlog_trace_for_tests()

    expert_id = (2, 3)
    entry = _ExpertEntry(
        expert_id=expert_id,
        tensor_schema=[],
        global_offsets={},
        total_numel=100,
        lossless_path=Path("lfcl.bin"),
        lossy_path=Path("lfsy.bin"),
        lossy_start_step=0,
        lossless_step=7,
    )
    state = ExpertState.__new__(ExpertState)
    state.cp_target = "test_dram"
    state._entries = {expert_id: entry}
    state._lossy_log_counter_index = {}
    state._lossy_log_bytes_by_expert = {expert_id: 8192}

    state._record_lfcl_writes(
        [(entry, 1200)], step=9, timestamp=222,
    )

    count = 10
    pinned = torch.zeros(4096, dtype=torch.uint8)
    struct.pack_into("<II", pinned.numpy(), 0, count, 9)
    row = SimpleNamespace(
        aos_byte_off=0,
        expert_id=expert_id,
        record_capacity=100,
    )
    mirror = SimpleNamespace(rows=lambda: [row])
    state._record_lfsy_writes(
        step=9,
        layout_mirror=mirror,
        pinned_aos=pinned,
        timestamp=444,
    )
    applied = []
    state._scatter_records_to_dense_with_cpp = (
        lambda ent, _pinned, byte_offset, records: applied.append(
            (ent.expert_id, byte_offset, records)
        )
    )
    state._add_lossy_log_bytes_from_writer = lambda _eid, nbytes: applied.append(
        ("bytes", nbytes)
    )
    state._scatter_lossy_aos_to_dense(
        9, mirror, pinned, counted_rows=[(row, entry, count)],
    )
    assert applied == [
        (expert_id, BLOCK_PREFIX_BYTES, count),
        ("bytes", 4096),
    ]
    manager = FCManager.__new__(FCManager)
    manager._record_nonexpert_write(step=9, payload_bytes=12345)

    close_io_backlog_trace()
    paths = list(tmp_path.glob("waf_rank00006_pid*_*.jsonl"))
    assert len(paths) == 1
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    metadata = rows[0]
    batches = [row for row in rows if row["record"] == "write_batch"]
    assert len(batches) == 3

    lfcl_writes = _write_dicts(metadata, batches[0])
    assert [(row["kind"], row["op"]) for row in lfcl_writes] == [
        ("c", "r"), ("y", "c"),
    ]
    assert lfcl_writes[0]["io_bytes"] == HEADER_BLOCK_BYTES + 4096
    assert lfcl_writes[0]["global_expert"] == 67
    assert lfcl_writes[1]["io_bytes"] == HEADER_BLOCK_BYTES

    lfsy_write = _write_dicts(metadata, batches[1])[0]
    assert lfsy_write["kind"] == "y"
    assert lfsy_write["op"] == "a"
    assert lfsy_write["io_bytes"] == 4096
    assert lfsy_write["offset"] == HEADER_BLOCK_BYTES + 8192

    nefc_write = _write_dicts(metadata, batches[2])[0]
    assert (nefc_write["kind"], nefc_write["op"]) == ("n", "r")
    assert nefc_write["io_bytes"] == 81920

    _reset_io_backlog_trace_for_tests()
