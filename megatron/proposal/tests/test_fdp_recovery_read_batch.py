import weakref

import pytest
import torch

from megatron.proposal.storage.recovery import (
    _GpuReplayLanePool,
    _nefc_shards_to_read,
    _read_fdp_expert_states_batched,
)


def test_partitioned_nefc_reads_only_current_owner(monkeypatch):
    monkeypatch.setenv("PROPOSAL_FDP_DP_RANK", "1")
    monkeypatch.setenv("PROPOSAL_FDP_EP_RANK", "2")
    shards = [
        {"dp_rank": dp, "ep_rank": ep, "shard_index": dp * 4 + ep}
        for dp in range(2)
        for ep in range(4)
    ]

    assert _nefc_shards_to_read(shards, partitioned=True) == [shards[6]]
    assert _nefc_shards_to_read(shards, partitioned=False) == shards


def test_next_read_window_overlaps_prior_replay_lane():
    events = []

    class _Stream:
        def __init__(self, index):
            self.index = index

        def synchronize(self):
            events.append(f"stream_sync:{self.index}")

    class _Event:
        def __init__(self, index):
            self.index = index

        def record(self, stream):
            events.append(f"record:{stream.index}")

        def synchronize(self):
            events.append(f"event_sync:{self.index}")

    streams = (_Stream(0), _Stream(1))
    done = (_Event(0), _Event(1))
    pool = object.__new__(_GpuReplayLanePool)
    pool._lanes = ((None, streams[0], done[0]), (None, streams[1], done[1]))
    pool._busy = [False, False]
    pool._recorded = [False, False]
    pool._keepalive = [None, None]

    def consume(decoded):
        for lane, item in enumerate(decoded.items()):
            pool.acquire(lane, item)
            events.append(f"enqueue:{item[0][1]}")
            pool.record(lane)

    _read_fdp_expert_states_batched(
        [(0, expert) for expert in range(4)],
        lfcs_by_eid={},
        lfsy_by_eid={(0, expert): {"logical_tail": 1} for expert in range(4)},
        t_fault=10,
        begin_batch=lambda: events.append("begin"),
        end_batch=lambda: events.append("end") or {},
        read_lfcs=lambda *_args, **_kwargs: None,
        read_lfsy=lambda _layer, expert, _nbytes, **_kwargs: (
            events.append(f"read:{expert}") or torch.zeros(1, dtype=torch.uint8)
        ),
        decode=lambda _lfcl, _lfsy, *, layer, expert, source: expert,
        consume_batch=consume,
    )
    pool.drain()

    assert events.index("read:2") < events.index("event_sync:0")
    assert events.index("read:3") < events.index("event_sync:1")


def test_replay_source_lives_until_lane_event_completes():
    class _Source:
        pass

    source = _Source()
    source_ref = weakref.ref(source)

    class _Stream:
        def synchronize(self):
            raise AssertionError("recorded lane must wait on its event")

    class _Event:
        def record(self, _stream):
            pass

        def synchronize(self):
            assert source_ref() is not None

    pool = object.__new__(_GpuReplayLanePool)
    pool._lanes = (
        (None, _Stream(), _Event()),
        (None, _Stream(), _Event()),
    )
    pool._busy = [False, False]
    pool._recorded = [False, False]
    pool._keepalive = [None, None]

    pool.acquire(0, source)
    pool.record(0)
    del source
    assert source_ref() is not None

    pool.wait(0)
    assert source_ref() is None


def test_lane_drain_attempts_both_lanes_after_first_sync_failure():
    synchronized = []

    class _Stream:
        def synchronize(self):
            raise AssertionError("recorded lane must wait on its event")

    class _Event:
        def __init__(self, index):
            self.index = index

        def record(self, _stream):
            pass

        def synchronize(self):
            synchronized.append(self.index)
            if self.index == 0:
                raise RuntimeError("lane 0 failure")

    pool = object.__new__(_GpuReplayLanePool)
    pool._lanes = (
        (None, _Stream(), _Event(0)),
        (None, _Stream(), _Event(1)),
    )
    pool._busy = [True, True]
    pool._recorded = [True, True]
    pool._keepalive = [object(), object()]

    with pytest.raises(RuntimeError, match="lane 0 failure"):
        pool.drain()

    assert synchronized == [0, 1]
    assert pool._keepalive[0] is not None
    assert pool._keepalive[1] is None


def test_fdp_expert_reads_finish_batch_before_decode(monkeypatch):
    monkeypatch.setenv("PROPOSAL_FDP_EP_SIZE", "1")
    events = []
    live_buffers = []

    def begin():
        events.append("begin")

    def read_lfcs(layer, expert, nbytes, *, _batch_keepalive):
        events.append(f"lfcs:{layer}:{expert}:{nbytes}")
        blob = torch.full((nbytes,), expert + 1, dtype=torch.uint8)
        _batch_keepalive.append(blob)
        live_buffers.append(weakref.ref(blob))
        return blob

    def read_lfsy(layer, expert, nbytes, *, _batch_keepalive):
        events.append(f"lfsy:{layer}:{expert}:{nbytes}")
        blob = torch.full((nbytes,), expert + 10, dtype=torch.uint8)
        _batch_keepalive.append(blob)
        live_buffers.append(weakref.ref(blob))
        return blob

    def end():
        assert live_buffers
        assert all(ref() is not None for ref in live_buffers)
        events.append("end")
        return {"commands": 7, "submitted": 7, "submit_calls": 1}

    def decode(lfcl, lfsy, *, layer, expert, source):
        assert events[-1] in ("end", f"decode:{layer}:{expert - 1}")
        assert source == "fdp"
        events.append(f"decode:{layer}:{expert}")
        return {"has_lfcl": lfcl is not None, "has_lfsy": lfsy is not None}

    states, stats = _read_fdp_expert_states_batched(
        [(0, 0), (0, 1)],
        lfcs_by_eid={(0, 0): {"iter": 8, "logical_bytes": 16}},
        lfsy_by_eid={
            (0, 0): {"logical_tail": 8},
            (0, 1): {"logical_tail": 12},
            # State not listed in expert_ids belongs to another owner/rank.
            (0, 2): {"logical_tail": 99},
        },
        t_fault=10,
        begin_batch=begin,
        end_batch=end,
        read_lfcs=read_lfcs,
        read_lfsy=read_lfsy,
        decode=decode,
    )

    assert events == [
        "begin",
        "lfcs:0:0:16",
        "lfsy:0:0:8",
        "lfsy:0:1:12",
        "end",
        "decode:0:0",
        "decode:0:1",
    ]
    assert states == {
        (0, 0): {"has_lfcl": True, "has_lfsy": True},
        (0, 1): {"has_lfcl": False, "has_lfsy": True},
    }
    assert stats["fdp_expert_read_batches"] == 1.0
    assert stats["fdp_expert_read_commands"] == 7.0
    assert stats["fdp_expert_read_submitted"] == 7.0
    assert stats["fdp_expert_read_bytes"] == 36.0


def test_fdp_expert_read_batches_are_bounded(monkeypatch):
    monkeypatch.setenv("PROPOSAL_FDP_EP_SIZE", "1")
    events = []

    states, stats = _read_fdp_expert_states_batched(
        [(0, 0), (0, 1), (0, 2)],
        lfcs_by_eid={},
        lfsy_by_eid={
            (0, 0): {"logical_tail": 1},
            (0, 1): {"logical_tail": 1},
            (0, 2): {"logical_tail": 1},
        },
        t_fault=10,
        begin_batch=lambda: events.append("begin"),
        end_batch=lambda: (
            events.append("end")
            or {"commands": 1, "submitted": 1, "submit_calls": 1}
        ),
        read_lfcs=lambda *_: pytest.fail("unexpected LFCS read"),
        read_lfsy=lambda layer, expert, nbytes, **_kwargs: (
            events.append(f"read:{expert}") or torch.zeros(nbytes, dtype=torch.uint8)
        ),
        decode=lambda _lfcl, _lfsy, *, layer, expert, source: (
            events.append(f"decode:{expert}") or {"expert": expert}
        ),
        batch_size=2,
    )

    assert list(states) == [(0, 0), (0, 1), (0, 2)]
    assert events == [
        "begin", "read:0", "read:1", "end", "decode:0", "decode:1",
        "begin", "read:2", "end", "decode:2",
    ]
    assert stats["fdp_expert_read_batches"] == 2.0


def test_fdp_expert_read_consumer_releases_each_bounded_batch(monkeypatch):
    monkeypatch.setenv("PROPOSAL_FDP_EP_SIZE", "1")
    consumed = []

    states, stats = _read_fdp_expert_states_batched(
        [(0, 0), (0, 1), (0, 2)],
        lfcs_by_eid={},
        lfsy_by_eid={
            (0, 0): {"logical_tail": 1},
            (0, 1): {"logical_tail": 1},
            (0, 2): {"logical_tail": 1},
        },
        t_fault=10,
        begin_batch=lambda: None,
        end_batch=lambda: {
            "commands": 1,
            "submitted": 1,
            "submit_calls": 1,
        },
        read_lfcs=lambda *_: pytest.fail("unexpected LFCS read"),
        read_lfsy=lambda layer, expert, nbytes, **_kwargs: torch.zeros(
            nbytes, dtype=torch.uint8,
        ),
        decode=lambda _lfcl, _lfsy, *, layer, expert, source: expert,
        consume_batch=lambda batch: consumed.append(tuple(batch)),
        batch_size=2,
    )

    assert states == {}
    assert consumed == [((0, 0), (0, 1)), ((0, 2),)]
    assert stats["fdp_expert_read_batches"] == 2.0


def test_fdp_expert_read_failure_drains_open_batch(monkeypatch):
    monkeypatch.setenv("PROPOSAL_FDP_EP_SIZE", "1")
    events = []
    submitted_buffer = None

    def fail_after_submit(_layer, _expert, _nbytes, *, _batch_keepalive):
        nonlocal submitted_buffer
        blob = torch.zeros(8, dtype=torch.uint8)
        submitted_buffer = weakref.ref(blob)
        _batch_keepalive.append(blob)
        events.append("read")
        raise RuntimeError("read failed")

    def drain():
        assert submitted_buffer is not None
        assert submitted_buffer() is not None
        events.append("end")
        return {}

    with pytest.raises(RuntimeError, match="read failed"):
        _read_fdp_expert_states_batched(
            [(0, 0)],
            lfcs_by_eid={(0, 0): {"iter": 1, "logical_bytes": 8}},
            lfsy_by_eid={},
            t_fault=10,
            begin_batch=lambda: events.append("begin"),
            end_batch=drain,
            read_lfcs=fail_after_submit,
            read_lfsy=lambda *_: pytest.fail("unexpected LFSY read"),
            decode=lambda *_args, **_kwargs: pytest.fail("decode must not run"),
        )

    assert events == ["begin", "read", "end"]


def test_empty_fdp_expert_read_does_not_open_batch():
    states, stats = _read_fdp_expert_states_batched(
        [],
        lfcs_by_eid={},
        lfsy_by_eid={},
        t_fault=0,
        begin_batch=lambda: pytest.fail("batch must remain closed"),
        end_batch=lambda: pytest.fail("batch must remain closed"),
        read_lfcs=None,
        read_lfsy=None,
        decode=None,
    )
    assert states == {}
    assert stats == {}
