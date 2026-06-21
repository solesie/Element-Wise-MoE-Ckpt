"""V2 ExpertState LFCL/LFSY behavior."""

from pathlib import Path
import struct
import threading
from typing import List

import pytest
import torch

from megatron.proposal.storage.expert_writer import (
    BLOCK_PREFIX_BYTES,
    HEADER_BLOCK_BYTES,
    RECORD_BYTES,
    ExpertState,
    read_expert_state,
)
from megatron.proposal.emission.aos_layout import AOS_LAYOUT_HEADER_WORDS_FIXED
from megatron.proposal.emission.record_kernels import LossyRecordWriteInputs
from megatron.proposal.emission.record_kernels import AdamHParams
from megatron.proposal.cutoff.age2bit import allocate_age2bit_score_state
from megatron.proposal.storage.proposal_io import (
    SECTOR_BYTES,
    alloc_pinned_uint8_sector_aligned,
)


def _tiny_expert_map():
    return {
        (0, 0): [
            ("mlp.w1", torch.Size([8]), 8),
            ("mlp.w2", torch.Size([16]), 16),
        ],
    }


def _round_up_sector(nbytes: int) -> int:
    return ((int(nbytes) + SECTOR_BYTES - 1) // SECTOR_BYTES) * SECTOR_BYTES


def _first_aos_prefix_offset(pinned_aos: torch.Tensor) -> int:
    header = pinned_aos.view(torch.int64)
    return int(header[AOS_LAYOUT_HEADER_WORDS_FIXED].item())


def _attach_age2bit(lossy_state: LossyRecordWriteInputs, n_elem: int, device):
    h_bins = int(lossy_state.h_lut_values.numel())
    age = allocate_age2bit_score_state(
        n_elements=n_elem,
        device=device,
        h_lut_bins=h_bins,
    )
    lossy_state.age2bit_words = age.age_words
    lossy_state.ehat_geom_lut = age.ehat_geom_lut
    lossy_state.step_device = torch.zeros(1, dtype=torch.int32, device=device)
    return age


def _make_lfcl_slot(pinned: torch.Tensor, byte_off: int, payload_values: List[float]):
    payload_bytes = len(payload_values) * 4
    slot_bytes = _round_up_sector(HEADER_BLOCK_BYTES + payload_bytes)
    slot = pinned.narrow(0, byte_off // 4, slot_bytes // 4)
    payload = slot.narrow(0, HEADER_BLOCK_BYTES // 4, len(payload_values))
    payload.copy_(torch.tensor(payload_values, dtype=torch.float32))
    return slot, payload_bytes, slot_bytes


def test_lfcl_compaction_uses_pinned_gpu_payload_and_rotates_lfsy(
    tmp_path: Path,
):
    state = ExpertState(_tiny_expert_map(), output_dir=tmp_path, fresh=True)
    state.append_lossy_records_for_test(
        (0, 0), step=1,
        tensor_updates=[
            ("mlp.w1", torch.tensor([3], dtype=torch.int32),
             torch.tensor([11.0]), torch.tensor([1.0]), torch.tensor([0.1])),
            ("mlp.w2", torch.tensor([5], dtype=torch.int32),
             torch.tensor([55.0]), torch.tensor([5.0]), torch.tensor([0.5])),
        ],
    )
    lossy = tmp_path / "lossy_expert_L00_E00.bin"
    assert lossy.stat().st_size > HEADER_BLOCK_BYTES

    payload_elems = 8 + 8 + 8 + 16 + 16 + 16
    payload_bytes = payload_elems * 4
    pinned = alloc_pinned_uint8_sector_aligned(HEADER_BLOCK_BYTES + payload_bytes).view(
        torch.float32,
    )
    payload = pinned.narrow(0, HEADER_BLOCK_BYTES // 4, payload_elems)
    payload.zero_()
    payload[3] = 11.0
    payload[8 + 3] = 1.0
    payload[16 + 3] = 0.1
    w2_base = 24
    payload[w2_base + 5] = 55.0
    payload[w2_base + 16 + 5] = 5.0
    payload[w2_base + 32 + 5] = 0.5
    state.compact_lossy_recovery_image_from_pinned(
        expert_id=(0, 0),
        step=4,
        pinned_info={
            "pinned": pinned,
            "payload_offset_bytes": HEADER_BLOCK_BYTES,
            "payload_bytes": int(payload_bytes),
        },
        cuda_event=None,
    )
    state.shutdown()

    lfcl = tmp_path / "fc_expert_L00_E00.bin"
    assert lfcl.exists()
    assert lfcl.stat().st_size == _round_up_sector(HEADER_BLOCK_BYTES + payload_bytes)
    assert lfcl.stat().st_size % SECTOR_BYTES == 0
    assert lossy.stat().st_size == HEADER_BLOCK_BYTES

    payload = read_expert_state(tmp_path, layer=0, expert=0)
    assert payload["_rolling_step"] == 4
    assert payload["mlp.w1:param"].reshape(-1)[3].item() == pytest.approx(11.0)
    assert payload["mlp.w1:exp_avg"].reshape(-1)[3].item() == pytest.approx(1.0)
    assert payload["mlp.w1:exp_avg_sq"].reshape(-1)[3].item() == pytest.approx(0.1)
    assert payload["mlp.w2:param"].reshape(-1)[5].item() == pytest.approx(55.0)
    assert payload["mlp.w2:exp_avg"].reshape(-1)[5].item() == pytest.approx(5.0)
    assert payload["mlp.w2:exp_avg_sq"].reshape(-1)[5].item() == pytest.approx(0.5)


def test_lfcl_batch_uses_one_pinned_slab_and_releases_once(tmp_path: Path):
    expert_map = {
        (0, 0): [("e0.w", torch.Size([2]), 2)],
        (0, 1): [("e1.w", torch.Size([3]), 3)],
    }
    state = ExpertState(expert_map, output_dir=tmp_path, fresh=True)
    ready = threading.Event()
    try:
        state.append_lossy_records_for_test(
            (0, 0), step=1,
            tensor_updates=[
                ("e0.w", torch.tensor([0], dtype=torch.int32),
                 torch.tensor([10.0]), torch.tensor([1.0]), torch.tensor([0.1])),
            ],
        )
        state.append_lossy_records_for_test(
            (0, 1), step=1,
            tensor_updates=[
                ("e1.w", torch.tensor([1], dtype=torch.int32),
                 torch.tensor([20.0]), torch.tensor([2.0]), torch.tensor([0.2])),
            ],
        )
        payload0 = [10.0, 11.0, 1.0, 1.1, 0.1, 0.11]
        payload1 = [20.0, 21.0, 22.0, 2.0, 2.1, 2.2, 0.2, 0.21, 0.22]
        slot0_bytes = _round_up_sector(HEADER_BLOCK_BYTES + len(payload0) * 4)
        slot1_bytes = _round_up_sector(HEADER_BLOCK_BYTES + len(payload1) * 4)
        pinned = alloc_pinned_uint8_sector_aligned(slot0_bytes + slot1_bytes).view(
            torch.float32,
        )
        slot0, payload0_bytes, _ = _make_lfcl_slot(pinned, 0, payload0)
        slot1, payload1_bytes, _ = _make_lfcl_slot(pinned, slot0_bytes, payload1)
        state.compact_lossy_recovery_images_from_pinned_batch(
            step=7,
            batch_info={
                "pinned": pinned,
                "ready_event": ready,
                "entries": [
                    {
                        "expert_id": (0, 0),
                        "pinned_info": {
                            "pinned": slot0,
                            "payload_offset_bytes": HEADER_BLOCK_BYTES,
                            "payload_bytes": payload0_bytes,
                        },
                    },
                    {
                        "expert_id": (0, 1),
                        "pinned_info": {
                            "pinned": slot1,
                            "payload_offset_bytes": HEADER_BLOCK_BYTES,
                            "payload_bytes": payload1_bytes,
                        },
                    },
                ],
            },
            cuda_event=None,
            commit_payload=True,
        )
        assert ready.is_set()
    finally:
        state.shutdown()

    assert (tmp_path / "lossy_expert_L00_E00.bin").stat().st_size == HEADER_BLOCK_BYTES
    assert (tmp_path / "lossy_expert_L00_E01.bin").stat().st_size == HEADER_BLOCK_BYTES
    e0 = read_expert_state(tmp_path, layer=0, expert=0)
    e1 = read_expert_state(tmp_path, layer=0, expert=1)
    assert e0["_rolling_step"] == 7
    assert e1["_rolling_step"] == 7
    assert e0["e0.w:param"].reshape(-1).tolist() == pytest.approx([10.0, 11.0])
    assert e0["e0.w:exp_avg"].reshape(-1).tolist() == pytest.approx([1.0, 1.1])
    assert e0["e0.w:exp_avg_sq"].reshape(-1).tolist() == pytest.approx([0.1, 0.11])
    assert e1["e1.w:param"].reshape(-1).tolist() == pytest.approx([20.0, 21.0, 22.0])


def test_lfsy_recovery_keeps_latest_record_when_indices_repeat(tmp_path: Path):
    state = ExpertState(
        {(0, 0): [("e.w", torch.Size([4]), 4)]},
        output_dir=tmp_path,
        fresh=True,
    )
    try:
        state.append_lossy_records_for_test(
            (0, 0), step=1,
            tensor_updates=[
                ("e.w", torch.tensor([1, 2], dtype=torch.int32),
                 torch.tensor([11.0, 12.0]),
                 torch.tensor([1.1, 1.2]),
                 torch.tensor([0.11, 0.12])),
            ],
        )
        state.append_lossy_records_for_test(
            (0, 0), step=2,
            tensor_updates=[
                ("e.w", torch.tensor([1, 3], dtype=torch.int32),
                 torch.tensor([21.0, 23.0]),
                 torch.tensor([2.1, 2.3]),
                 torch.tensor([0.21, 0.23])),
            ],
        )
        state.append_lossy_records_for_test(
            (0, 0), step=3,
            tensor_updates=[
                ("e.w", torch.tensor([2], dtype=torch.int32),
                 torch.tensor([32.0]),
                 torch.tensor([3.2]),
                 torch.tensor([0.32])),
            ],
        )
    finally:
        state.shutdown()

    payload = read_expert_state(tmp_path, layer=0, expert=0)
    assert payload["_rolling_step"] == 3
    assert payload["e.w:param"].reshape(-1).tolist() == pytest.approx(
        [0.0, 21.0, 32.0, 23.0],
    )
    assert payload["e.w:exp_avg"].reshape(-1).tolist() == pytest.approx(
        [0.0, 2.1, 3.2, 2.3],
    )
    assert payload["e.w:exp_avg_sq"].reshape(-1).tolist() == pytest.approx(
        [0.0, 0.21, 0.32, 0.23],
    )


def test_pinned_aos_capacity_uses_local_worst_case(tmp_path: Path):
    expert_map = {
        (0, 0): [("expert0.w", torch.Size([1000]), 1000)],
        (0, 1): [("expert1.w", torch.Size([1000]), 1000)],
    }
    state = ExpertState(
        expert_map,
        output_dir=tmp_path,
        fresh=True,
    )
    expected = _round_up_sector(
        2000 * RECORD_BYTES
        + len(expert_map) * (BLOCK_PREFIX_BYTES + SECTOR_BYTES)
    )

    assert int(state._pinned_aos.numel()) == expected
    state.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_direct_emission_cursor_cap_prevents_pinned_aos_overflow(tmp_path: Path):
    expert_map = {(0, 0): [("mlp.w", torch.Size([1024]), 1024)]}
    state = ExpertState(
        expert_map,
        output_dir=tmp_path,
        fresh=True,
        cp_target="test_dram",
        pinned_aos_record_capacity=4,
    )
    device = torch.device("cuda")
    group_size = 4
    n_elem = 1024
    m = torch.ones(n_elem, dtype=torch.float32, device=device)
    v = torch.ones(n_elem, dtype=torch.float32, device=device)
    p = torch.arange(n_elem, dtype=torch.float32, device=device) + 10.0
    h_lut_values = torch.ones(8, dtype=torch.float32, device=device)
    h_lut_log_v_min = torch.zeros((), dtype=torch.float32, device=device)
    h_lut_inv_log_v_width = torch.ones((), dtype=torch.float32, device=device)
    total_th = torch.zeros((), dtype=torch.float32, device=device)
    lossy_state = LossyRecordWriteInputs(
        bf16_groups_mvph=[(m, v, p, 0)],
        name_list=["mlp.w"],
        group_size=group_size,
        within_off_per_name=[0],
        group_count_per_name=[n_elem // group_size],
        h_lut_values=h_lut_values,
        h_lut_log_v_min=h_lut_log_v_min,
        h_lut_inv_log_v_width=h_lut_inv_log_v_width,
        total_th=total_th,
        adam_hparams=AdamHParams(
            lr_over_bc1=1.0,
            sqrt_bc2_inv=1.0,
            eps=1.0e-8,
        ),
        direct_emission_enabled=True,
    )
    age = _attach_age2bit(lossy_state, n_elem, device)
    age.age_words.fill_(-1)
    lossy_state.step_device.fill_(5)
    try:
        n_requested = state.write_for_iter(
            step=5,
            per_expert_updates={(0, 0): [("mlp.w", 0, 0, n_elem)]},
            compute_event=None,
            gpu_refs=[
                m, v, p, h_lut_values, total_th,
                age.age_words, age.ehat_geom_lut, lossy_state.step_device,
            ],
            lossy_state=lossy_state,
        )
        torch.cuda.synchronize()
        assert n_requested == n_elem
        assert state._direct_record_cursor_device is not None
        assert state._direct_overflow_count_device is not None
        assert int(state._direct_record_cursor_device[0].item()) == n_elem
        assert int(state._direct_overflow_count_device[0].item()) > 0
        feedback = state.consume_direct_emission_feedback()
        assert feedback is not None
        assert int(feedback["step"]) == 5
        assert int(feedback["selected"]) == n_elem
        assert int(feedback["overflow"]) > 0
        assert int(feedback["written"]) > 0
        assert float(feedback["tau"]) == pytest.approx(0.0)

        dense_p = state._dense_buffers[(0, 0)]["mlp.w"]["p"]
        actual_count = (
            int(state._direct_record_cursor_device[0].item())
            - int(state._direct_overflow_count_device[0].item())
        )
        prefix_count, prefix_step = struct.unpack_from(
            "<II",
            state._pinned_aos.numpy(),
            _first_aos_prefix_offset(state._pinned_aos),
        )
        assert int(prefix_step) == 5
        assert int(prefix_count) == actual_count
        assert 0 < actual_count < n_elem
        nonzero = dense_p[dense_p != 0]
        assert int(nonzero.numel()) == actual_count
        assert float(nonzero.min().item()) >= 10.0
        assert float(nonzero.max().item()) < 10.0 + n_elem
        assert bool(torch.any(age.age_words != -1).item())
        assert bool(torch.any(age.age_words == -1).item())
    finally:
        state.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_direct_emission_count_matches_selection_predicate(tmp_path: Path):
    expert_map = {(0, 0): [("mlp.w", torch.Size([16]), 16)]}
    state = ExpertState(
        expert_map,
        output_dir=tmp_path,
        fresh=True,
        cp_target="test_dram",
    )
    device = torch.device("cuda")
    group_size = 4
    m = torch.zeros(16, dtype=torch.float32, device=device)
    m[0] = 1.0
    m[8] = 1.0
    v = torch.ones(16, dtype=torch.float32, device=device)
    p = torch.arange(16, dtype=torch.float32, device=device) + 20.0
    h_lut_values = torch.ones(8, dtype=torch.float32, device=device)
    h_lut_log_v_min = torch.zeros((), dtype=torch.float32, device=device)
    h_lut_inv_log_v_width = torch.ones((), dtype=torch.float32, device=device)
    total_th = torch.tensor(0.5, dtype=torch.float32, device=device)
    hp = AdamHParams(lr_over_bc1=1.0, sqrt_bc2_inv=1.0, eps=1.0e-8)

    expected_records = 2

    lossy_state = LossyRecordWriteInputs(
        bf16_groups_mvph=[(m, v, p, 0)],
        name_list=["mlp.w"],
        group_size=group_size,
        within_off_per_name=[0],
        group_count_per_name=[4],
        h_lut_values=h_lut_values,
        h_lut_log_v_min=h_lut_log_v_min,
        h_lut_inv_log_v_width=h_lut_inv_log_v_width,
        total_th=total_th,
        adam_hparams=hp,
        direct_emission_enabled=True,
    )
    age = _attach_age2bit(lossy_state, 16, device)
    lossy_state.step_device.fill_(7)
    try:
        state.write_for_iter(
            step=7,
            per_expert_updates={(0, 0): [("mlp.w", 0, 0, 16)]},
            compute_event=None,
            gpu_refs=[
                m, v, p, h_lut_values, total_th,
                age.age_words, age.ehat_geom_lut, lossy_state.step_device,
            ],
            lossy_state=lossy_state,
        )
        torch.cuda.synchronize()
        assert state._direct_record_cursor_device is not None
        assert state._direct_overflow_count_device is not None
        assert int(state._direct_record_cursor_device[0].item()) == expected_records
        assert int(state._direct_overflow_count_device[0].item()) == 0
        prefix_count, prefix_step = struct.unpack_from(
            "<II",
            state._pinned_aos.numpy(),
            _first_aos_prefix_offset(state._pinned_aos),
        )
        assert int(prefix_count) == expected_records
        assert int(prefix_step) == 7
        dense_p = state._dense_buffers[(0, 0)]["mlp.w"]["p"]
        assert int(torch.count_nonzero(dense_p).item()) == expected_records
    finally:
        state.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_direct_emission_resets_age_only_for_actual_written(tmp_path: Path):
    expert_map = {(0, 0): [("mlp.w", torch.Size([16]), 16)]}
    state = ExpertState(
        expert_map,
        output_dir=tmp_path,
        fresh=True,
        cp_target="test_dram",
    )
    device = torch.device("cuda")
    group_size = 4
    m = torch.zeros(16, dtype=torch.float32, device=device)
    m[0] = 4.0
    v = torch.ones(16, dtype=torch.float32, device=device)
    p = torch.arange(16, dtype=torch.float32, device=device) + 30.0
    h_lut_values = torch.ones(8, dtype=torch.float32, device=device)
    h_lut_log_v_min = torch.zeros((), dtype=torch.float32, device=device)
    h_lut_inv_log_v_width = torch.ones((), dtype=torch.float32, device=device)
    total_th = torch.tensor(0.0, dtype=torch.float32, device=device)
    hp = AdamHParams(lr_over_bc1=1.0, sqrt_bc2_inv=1.0, eps=1.0e-8)

    lossy_state = LossyRecordWriteInputs(
        bf16_groups_mvph=[(m, v, p, 0)],
        name_list=["mlp.w"],
        group_size=group_size,
        within_off_per_name=[0],
        group_count_per_name=[4],
        h_lut_values=h_lut_values,
        h_lut_log_v_min=h_lut_log_v_min,
        h_lut_inv_log_v_width=h_lut_inv_log_v_width,
        total_th=total_th,
        adam_hparams=hp,
        direct_emission_enabled=True,
    )
    age = _attach_age2bit(lossy_state, 16, device)
    age.age_words.fill_(-1)
    lossy_state.step_device.fill_(8)
    try:
        state.write_for_iter(
            step=8,
            per_expert_updates={(0, 0): [("mlp.w", 0, 0, 16)]},
            compute_event=None,
            gpu_refs=[
                m, v, p, h_lut_values, total_th,
                age.age_words, age.ehat_geom_lut, lossy_state.step_device,
            ],
            lossy_state=lossy_state,
        )
        torch.cuda.synchronize()
        assert state._direct_record_cursor_device is not None
        assert state._direct_overflow_count_device is not None
        assert int(state._direct_record_cursor_device[0].item()) == 16
        assert int(state._direct_overflow_count_device[0].item()) == 0
        assert int(age.age_words[0].item()) == 0
    finally:
        state.shutdown()


def test_lfcl_compaction_does_not_touch_selection_state(tmp_path: Path):
    state = ExpertState(
        _tiny_expert_map(),
        output_dir=tmp_path,
        fresh=True,
        cp_target="test_dram",
    )
    selection_sentinel = torch.tensor([1.0, 2.0, 3.0])
    proposal_sentinel = torch.tensor([11], dtype=torch.int32)
    try:
        state.compact_lossy_recovery_image((0, 0), step=9)
        assert state.lossless_fc_step((0, 0)) == 9
        assert torch.equal(selection_sentinel, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(proposal_sentinel, torch.tensor([11], dtype=torch.int32))
    finally:
        state.shutdown()
