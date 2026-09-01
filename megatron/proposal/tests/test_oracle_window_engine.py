from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from megatron.proposal.telemetry.oracle_window.engine import (
    DOMAIN_ORDER,
    FullCount,
    LocalOracleWindowCoordinator,
    OracleWindowEngine,
    OracleWindowSettings,
    OracleWindowTopology,
    RankSamplePacket,
    SampleBatch,
    _fdp_global_expert_base,
    _deduplicate_packet,
    _global_cutoff,
)
from megatron.proposal.telemetry.oracle_window.fragments import (
    DOMAIN_EXPERT,
    DOMAIN_NONEXPERT,
    PackedFragmentLayout,
)


class _Layouts:
    def build_layouts(self, *, model):
        del model
        return {
            DOMAIN_EXPERT: PackedFragmentLayout(DOMAIN_EXPERT, (), 90),
            DOMAIN_NONEXPERT: PackedFragmentLayout(DOMAIN_NONEXPERT, (), 10),
        }


class _Curvature:
    def __init__(self) -> None:
        self.device_group = object()

    def prepare_window(self, **kwargs):
        self.device_group = kwargs["device_process_group"]
        assert kwargs["forward_step_func"] == "forward"
        return {DOMAIN_EXPERT: "expert-lut", DOMAIN_NONEXPERT: "ne-lut"}

    def provenance(self):
        return {
            DOMAIN_EXPERT: {"estimator": "expert-fd", "epsilon": 1.0e-3},
            DOMAIN_NONEXPERT: {"estimator": "ne-fd", "epsilon": 1.0e-2},
        }


class _Compute:
    def __init__(self, *, bad_count: bool = False, bad_update: bool = False):
        self.bad_count = bad_count
        self.bad_update = bad_update
        self.events = []

    def initialize(self, **kwargs):
        self.events.append(("initialize", tuple(kwargs["layouts"])))

    def begin(self, **kwargs):
        self.events.append(("begin", tuple(kwargs["layouts"])))

    def sample_round(self, *, domain, lut, round_index, base_salt):
        del lut, base_salt
        domain_base = 0 if domain == DOMAIN_EXPERT else 1_000_000
        ids = domain_base + round_index * 10 + np.asarray([0, 1], dtype=np.int64)
        raw = (
            np.asarray([9.0, -1.0], dtype=np.float32)
            if domain == DOMAIN_EXPERT
            else np.asarray([8.0, 1.0], dtype=np.float32)
        )
        weights = (
            np.asarray([80.0, 10.0], dtype=np.float64)
            if domain == DOMAIN_EXPERT
            else np.asarray([9.0, 1.0], dtype=np.float64)
        )
        return SampleBatch(raw, np.maximum(raw, 0.0), ids, weights)

    def full_count(self, *, domain, lut, tau):
        del lut, tau
        if self.bad_count:
            return FullCount(20, 0, 0, 0, 0, 0)
        return (
            FullCount(7, 2, 20, 10, 60, 0)
            if domain == DOMAIN_EXPERT
            else FullCount(1, 2, 2, 1, 7, 0)
        )

    def update_selected(self, *, domain, lut, tau, tie_hash_limit, tie_salt):
        del lut, tau, tie_hash_limit, tie_salt
        self.events.append(("update", domain))
        if self.bad_update:
            return 20
        return 8 if domain == DOMAIN_EXPERT else 2


class _Writer:
    def __init__(self, compute: _Compute):
        self.compute = compute
        self.records = []

    def write(self, **kwargs):
        assert self.compute.events[-2:] == [
            ("update", DOMAIN_EXPERT),
            ("update", DOMAIN_NONEXPERT),
        ]
        self.records.append(kwargs)


class _RemoteCurvatureFailureCoordinator:
    """CPU-only two-rank collective mock with a remote preflight failure."""

    rank = 0
    world = 2
    cpu_process_group = object()

    def gather_samples(self, packet):
        raise AssertionError("sample gathering is not expected in this test")

    def gather_objects(self, value):
        if isinstance(value, tuple) and len(value) == 4:
            rank, stage, ok, detail = value
            assert (rank, ok, detail) == (0, True, "")
            return (
                value,
                (1, stage, False, "RuntimeError: rank-one invalid FD inputs"),
            )
        if isinstance(value, tuple) and len(value) == 3:
            return (value, (1, True, ""))
        return (value, ())

    def sum_ints(self, values):
        return np.asarray(values, dtype=np.int64) * 2


class _StageAwareCurvature(_Curvature):
    def __init__(self) -> None:
        super().__init__()
        self.entered_device_collective = False

    def prepare_window(self, **kwargs):
        kwargs["stage_consensus"]("nonexpert_fd_inputs", None)
        self.entered_device_collective = True
        return super().prepare_window(**kwargs)


def _engine(tmp_path, *, compute=None, curvature=None):
    compute = compute or _Compute()
    curvature = curvature or _Curvature()
    writer = _Writer(compute)
    engine = OracleWindowEngine(
        settings=OracleWindowSettings(
            output_dir=tmp_path,
            topk_ratio=0.10,
            window_updates=2,
            snapshot_offsets=(1, 2),
            sample_rounds=4,
            max_sample_rounds=4,
            sample_blocks=2,
            cutoff_tolerance=0.0,
            min_effective_tail_count=0.0,
        ),
        topology=OracleWindowTopology(rank=0, world=1),
        layout_provider=_Layouts(),
        curvature_provider=curvature,
        compute=compute,
        coordinator=LocalOracleWindowCoordinator(),
        device_process_group="device-pg",
        writer=writer,
    )
    return engine, compute, curvature, writer


def test_remote_curvature_preflight_failure_stops_before_device_collective(tmp_path):
    compute = _Compute()
    curvature = _StageAwareCurvature()
    engine = OracleWindowEngine(
        settings=OracleWindowSettings(
            output_dir=tmp_path,
            topk_ratio=0.10,
            window_updates=2,
            snapshot_offsets=(1, 2),
            sample_rounds=4,
            max_sample_rounds=4,
            sample_blocks=2,
        ),
        topology=OracleWindowTopology(rank=0, world=2),
        layout_provider=_Layouts(),
        curvature_provider=curvature,
        compute=compute,
        coordinator=_RemoteCurvatureFailureCoordinator(),
        device_process_group="device-pg",
        writer=_Writer(compute),
    )

    with pytest.raises(
        RuntimeError,
        match="rank1=RuntimeError: rank-one invalid FD inputs",
    ):
        engine.prepare_window(
            model=object(),
            forward_step_func="forward",
            data_iterator=object(),
            start_iteration=100,
        )
    assert curvature.entered_device_collective is False


def test_engine_prepares_then_writes_actual_iteration_and_weighted_samples(tmp_path):
    engine, compute, curvature, writer = _engine(tmp_path)
    engine.prepare_window(
        model=object(),
        forward_step_func="forward",
        data_iterator=object(),
        start_iteration=100,
    )
    assert curvature.device_group == "device-pg"
    assert compute.events == [
        ("initialize", DOMAIN_ORDER),
        ("begin", DOMAIN_ORDER),
    ]
    engine.observe_successful_update(
        model=object(), start_iteration=100, iteration=103,
        offset=1, snapshot_due=True,
    )
    engine.observe_successful_update(
        model=object(), start_iteration=100, iteration=104,
        offset=2, snapshot_due=True,
    )
    engine.finish_window(model=object(), start_iteration=100, end_iteration=104)

    assert [record["iteration"] for record in writer.records] == [103, 104]
    first = writer.records[0]
    arrays = first["arrays"]
    assert np.any(arrays["sample_raw"] < 0.0)
    assert np.all(arrays["sample_clipped"] >= 0.0)
    for domain, population in enumerate((90.0, 10.0)):
        mask = arrays["sample_domain"] == domain
        assert arrays["sample_weight"][mask].sum() == pytest.approx(population)
    assert first["manifest"]["global_actual_selected_count"] == 10
    assert first["manifest"]["curvature_provenance"][DOMAIN_EXPERT]["epsilon"] == 1.0e-3
    assert first["manifest"]["curvature_provenance"][DOMAIN_NONEXPERT]["epsilon"] == 1.0e-2


def test_cutoff_refinement_fails_before_shadow_update(tmp_path):
    engine, compute, _curvature, _writer = _engine(
        tmp_path, compute=_Compute(bad_count=True),
    )
    engine.prepare_window(
        model=object(), forward_step_func="forward", data_iterator=object(),
        start_iteration=100,
    )
    with pytest.raises(RuntimeError, match="refinement exhausted"):
        engine.observe_successful_update(
            model=object(), start_iteration=100, iteration=101,
            offset=1, snapshot_due=True,
        )
    assert not any(event[0] == "update" for event in compute.events)


def test_actual_selected_count_is_fail_closed(tmp_path):
    engine, _compute, _curvature, writer = _engine(
        tmp_path, compute=_Compute(bad_update=True),
    )
    engine.prepare_window(
        model=object(), forward_step_func="forward", data_iterator=object(),
        start_iteration=100,
    )
    with pytest.raises(RuntimeError, match="missed K-band"):
        engine.observe_successful_update(
            model=object(), start_iteration=100, iteration=101,
            offset=1, snapshot_due=True,
        )
    assert writer.records == []


def test_global_cutoff_uses_fragment_aware_weights():
    packet = RankSamplePacket(
        rank=0,
        population_count=np.asarray([90, 10], dtype=np.int64),
        sample_domain=np.asarray([0, 0, 1, 1], dtype=np.int16),
        sample_round=np.asarray([0, 0, 0, 0], dtype=np.int16),
        sample_global_id=np.asarray([0, 1, 2, 3], dtype=np.int64),
        sample_raw=np.asarray([10.0, 1.0, 9.0, 0.0], dtype=np.float32),
        sample_clipped=np.asarray([10.0, 1.0, 9.0, 0.0], dtype=np.float32),
        sample_weight=np.asarray([80.0, 10.0, 9.0, 1.0]),
    )
    cutoff, population = _global_cutoff((packet,), topk_ratio=0.10)
    assert population == 100
    assert cutoff["tau"] == pytest.approx(10.0)


def test_expert_base_honors_global_id_contract(monkeypatch):
    monkeypatch.setenv("PROPOSAL_FDP_EXPERT_ID_BASE", "32")
    monkeypatch.delenv("PROPOSAL_FDP_EXPERT_IDS_GLOBAL", raising=False)
    assert _fdp_global_expert_base() == 32
    monkeypatch.setenv("PROPOSAL_FDP_EXPERT_IDS_GLOBAL", "true")
    assert _fdp_global_expert_base() == 0


def test_tiny_stratum_duplicates_aggregate_mass_without_fake_round_evidence():
    domains = []
    rounds = []
    ids = []
    raw = []
    weights = []
    for domain in range(2):
        for round_id in range(4):
            domains.extend((domain, domain))
            rounds.extend((round_id, round_id))
            # One width-1 stratum repeats; the other stratum remains distinct.
            ids.extend((domain * 1000, domain * 1000 + 10 + round_id))
            raw.extend((5.0 + domain, 1.0 + round_id))
            weights.extend((1.0, 1.5))  # 4 + 6 = population 10
    raw_array = np.asarray(raw, dtype=np.float32)
    packet = RankSamplePacket(
        rank=0,
        population_count=np.asarray([10, 10], dtype=np.int64),
        sample_domain=np.asarray(domains, dtype=np.int16),
        sample_round=np.asarray(rounds, dtype=np.int16),
        sample_global_id=np.asarray(ids, dtype=np.int64),
        sample_raw=raw_array,
        sample_clipped=np.maximum(raw_array, 0.0),
        sample_weight=np.asarray(weights, dtype=np.float64),
    )
    deduped = _deduplicate_packet(packet)
    for domain in range(2):
        domain_mask = deduped.sample_domain == domain
        assert deduped.sample_weight[domain_mask].sum() == pytest.approx(10.0)
        sticky = domain_mask & (deduped.sample_global_id == domain * 1000)
        assert sticky.sum() == 1
        assert deduped.sample_weight[sticky][0] == pytest.approx(4.0)
        assert set(deduped.sample_round[domain_mask]) == {0, 1, 2, 3}
