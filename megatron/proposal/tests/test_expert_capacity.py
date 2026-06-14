from megatron.proposal.emission.expert_capacity import (
    fit_record_capacities_to_bytes,
    redistribute_expert_record_capacities,
)


def _total_bytes(caps):
    total = 0
    for cap in caps:
        raw = 8 + int(cap) * 16
        total += ((raw + 4095) // 4096) * 4096
    return total


def test_fit_record_capacities_respects_fixed_slab_bytes():
    caps = fit_record_capacities_to_bytes(
        [1000, 1000, 1000, 1000],
        max_total_bytes=4 * 4096 + 2000 * 16,
        prefix_bytes=8,
        record_bytes=16,
        sector_bytes=4096,
    )
    assert _total_bytes(caps) <= 4 * 4096 + 2000 * 16


def test_redistribute_freezes_only_when_every_expert_has_zero_overflow():
    previous = [500, 700, 100]
    caps = redistribute_expert_record_capacities(
        selected_by_expert=[100, 900, 0],
        overflow_by_expert=[0, 0, 0],
        max_records_by_expert=[1000, 1000, 1000],
        target_record_capacity=1300,
        max_total_bytes=3 * 4096 + 1300 * 16,
        prefix_bytes=8,
        record_bytes=16,
        sector_bytes=4096,
        previous_capacities=previous,
    )

    assert caps == fit_record_capacities_to_bytes(
        previous,
        max_total_bytes=3 * 4096 + 1300 * 16,
        prefix_bytes=8,
        record_bytes=16,
        sector_bytes=4096,
    )


def test_redistribute_moves_cold_spare_to_overflowing_hot_expert():
    caps = redistribute_expert_record_capacities(
        selected_by_expert=[100, 900, 0],
        overflow_by_expert=[0, 100, 0],
        max_records_by_expert=[1000, 1000, 1000],
        target_record_capacity=1300,
        max_total_bytes=3 * 4096 + 1300 * 16,
        prefix_bytes=8,
        record_bytes=16,
        sector_bytes=4096,
        previous_capacities=[500, 500, 300],
    )

    assert _total_bytes(caps) <= 3 * 4096 + 1300 * 16
    assert caps[1] > caps[0]
    assert caps[1] > 500
