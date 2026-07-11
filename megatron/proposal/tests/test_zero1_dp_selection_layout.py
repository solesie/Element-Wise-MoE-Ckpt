import pytest
import torch

from megatron.proposal.checkpoint.lossy_checkpoint_engine import (
    LossyCheckpointPipeline,
)
from megatron.proposal.cutoff.cutoff_orchestrator import (
    CutoffOrchestrationMixin,
)
from megatron.proposal.storage.fc_manager import FCManager
from megatron.proposal.storage.recovery import (
    _partition_owned_expert_params,
)


class _CountCacheHarness(CutoffOrchestrationMixin):
    pass


def test_partition_layout_invalidates_prepartition_global_count_cache():
    state = _CountCacheHarness()
    state._global_selection_element_count_host = 200
    state._local_selection_element_count_host = 100
    state._selection_global_n_elements = torch.zeros((), dtype=torch.int64)

    state._invalidate_global_selection_element_count_cache()

    assert state._global_selection_element_count_host is None
    assert state._local_selection_element_count_host is None
    assert state._global_selection_element_count_cached(
        50,
        process_group=None,
        world_size=1,
    ) == 50
    assert state._local_selection_element_count_host == 50
    assert state._global_selection_element_count_host == 50


def test_finalized_selection_domain_log_uses_partition_count(capsys):
    state = _CountCacheHarness()
    state._global_selection_element_count_host = None
    state._local_selection_element_count_host = None
    state._selection_domain_count_logged = False
    state._selection_global_n_elements = torch.zeros((), dtype=torch.int64)
    state._topology = type("Topology", (), {"is_global_rank0": True})()

    assert state._global_selection_element_count_cached(
        50,
        process_group=None,
        world_size=1,
    ) == 50

    out = capsys.readouterr().out
    assert "finalized partitioned selection domain" in out
    assert "rank0_local=50 global=50" in out


def _partitioned_pipeline_for_static_layout():
    pipeline = LossyCheckpointPipeline.__new__(LossyCheckpointPipeline)
    pipeline.group_size = 1
    pipeline._direct_write_static_inputs = None
    # Original model names deliberately remain in this mapping.  Rank-local
    # ZeRO fragments cover only experts (15, 10) and (15, 12).
    pipeline._tname_to_eid = {
        "layer15.experts.0.weight": (15, 0),
        "layer15.experts.1.weight": (15, 1),
        "layer15.experts.10.weight": (15, 10),
        "layer15.experts.12.weight": (15, 12),
        "frag_e12": (15, 12),
        "frag_e10": (15, 10),
    }
    pipeline._tname_list = ["frag_e12", "frag_e10"]
    pipeline._group_offsets_cpu = [0, 4, 10]
    flat = torch.zeros(10, dtype=torch.float32)
    pipeline._groups = [{
        "start_name_in_grp": 0,
        "end_name_in_grp": 2,
        "m_flat": flat,
        "v_flat": flat,
        "p_flat": flat,
        "theta_cp_flat": None,
        "master_off": 0,
        "entry_elem_start_host": [0, 4],
        "entry_age_elem_start_host": [0, 4],
        "zero_copy": True,
    }]
    return pipeline


def test_partition_probe_expert_ids_exclude_unowned_original_experts():
    pipeline = _partitioned_pipeline_for_static_layout()

    active = pipeline._active_expert_ids_for_current_layout()

    assert active == ((15, 10), (15, 12))


def test_capacity_hint_order_matches_sorted_partition_write_slots():
    pipeline = _partitioned_pipeline_for_static_layout()
    pipeline._local_expert_ids = (
        pipeline._active_expert_ids_for_current_layout()
    )

    static = pipeline._static_direct_write_inputs()

    assert tuple(static["per_expert_batch"]) == ((15, 10), (15, 12))
    assert static["capacity_hint_expert_ids"] == ((15, 10), (15, 12))


def test_capacity_hint_order_mismatch_is_rejected_before_gpu_emission():
    pipeline = _partitioned_pipeline_for_static_layout()
    pipeline._local_expert_ids = ((15, 0), (15, 1), (15, 10), (15, 12))

    with pytest.raises(RuntimeError, match="capacity-demand hint expert order"):
        pipeline._static_direct_write_inputs()


def test_fc_manager_lfcl_slots_follow_partition_owned_experts():
    manager = FCManager.__new__(FCManager)
    manager.lfcl_fraction_pct = 1.0
    # Model enumeration can contain 384 experts, but this rank owns 291.
    owned = [(layer, expert) for layer in range(3) for expert in range(97)]

    manager._install_active_expert_ids(reversed(owned))

    assert manager._all_expert_ids == sorted(owned)
    assert manager._num_expert_slots == 291
    assert manager._experts_per_lfcl_event == 3
    assert manager._sorted_layers == [0, 1, 2]
    assert manager._num_experts_by_layer == {0: 97, 1: 97, 2: 97}
    assert manager._slot_index[owned[-1]] == 290


def test_recovery_expected_experts_follow_partition_ownership():
    by_expert = {
        (0, expert): {"weight": torch.nn.Parameter(torch.ones(1))}
        for expert in range(4)
    }

    owned = _partition_owned_expert_params(
        by_expert,
        expected_expert_ids=[(0, 1), (0, 3)],
    )

    assert tuple(owned) == ((0, 1), (0, 3))
    assert tuple(_partition_owned_expert_params(by_expert, None)) == tuple(by_expert)
