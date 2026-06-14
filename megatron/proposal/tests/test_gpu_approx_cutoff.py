import pytest
import torch

from megatron.proposal.cutoff.approx_probe_kernels import (
    launch_probe_decision,
    launch_retry_candidates_if_code,
    launch_zero_probe_accumulators_if_code,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_probe_decision_accepts_and_shrinks_sample_when_ci_is_tight():
    device = torch.device("cuda")
    candidate_taus = torch.tensor([0.5, 1.0, 2.0], device=device)
    sum_q = torch.tensor([[0.40 * 16, 0.25 * 16, 0.10 * 16]], device=device)
    sum_q2 = torch.tensor(
        [[0.40 * 0.40 * 16, 0.25 * 0.25 * 16, 0.10 * 0.10 * 16]],
        device=device,
    )
    n_blocks = torch.full((1, 3), 16, dtype=torch.int32, device=device)
    total_th = torch.zeros((), dtype=torch.float32, device=device)
    probe_reject = torch.ones((), dtype=torch.int32, device=device)
    decision_code = torch.zeros((), dtype=torch.int32, device=device)
    sample_blocks_current = torch.full((), 16, dtype=torch.int32, device=device)
    ci_shortfall_streak = torch.full((), 2, dtype=torch.int32, device=device)
    diagnostics = torch.zeros(10, dtype=torch.float32, device=device)
    stratum_total = torch.tensor([1024], dtype=torch.int64, device=device)

    launch_probe_decision(
        candidate_taus=candidate_taus,
        sum_q=sum_q,
        sum_q2=sum_q2,
        n_blocks=n_blocks,
        stratum_total_elems=stratum_total,
        total_th=total_th,
        probe_reject=probe_reject,
        decision_code=decision_code,
        sample_blocks_current=sample_blocks_current,
        ci_shortfall_streak=ci_shortfall_streak,
        diagnostics=diagnostics,
        n_candidates=3,
        total_elems_global=1024,
        target_ratio=0.25,
        cutoff_tolerance=0.04,
        sample_blocks_min=4,
        sample_blocks_cap=64,
    )
    torch.cuda.synchronize()

    assert int(probe_reject.item()) == 0
    assert float(total_th.item()) == pytest.approx(1.0)
    assert int(decision_code.item()) == 2
    assert int(sample_blocks_current.item()) == 14
    assert int(ci_shortfall_streak.item()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_probe_decision_grows_sample_after_final_ci_shortfall():
    device = torch.device("cuda")
    candidate_taus = torch.tensor([0.5, 2.0], device=device)
    sum_q = torch.tensor([[0.8, 0.7]], device=device)
    sum_q2 = torch.tensor([[0.64, 0.49]], device=device)
    n_blocks = torch.full((1, 2), 2, dtype=torch.int32, device=device)
    total_th = torch.zeros((), dtype=torch.float32, device=device)
    probe_reject = torch.ones((), dtype=torch.int32, device=device)
    decision_code = torch.zeros((), dtype=torch.int32, device=device)
    sample_blocks_current = torch.full((), 16, dtype=torch.int32, device=device)
    ci_shortfall_streak = torch.zeros((), dtype=torch.int32, device=device)
    diagnostics = torch.zeros(10, dtype=torch.float32, device=device)
    stratum_total = torch.tensor([1024], dtype=torch.int64, device=device)

    for _ in range(3):
        launch_probe_decision(
            candidate_taus=candidate_taus,
            sum_q=sum_q,
            sum_q2=sum_q2,
            n_blocks=n_blocks,
            stratum_total_elems=stratum_total,
            total_th=total_th,
            probe_reject=probe_reject,
            decision_code=decision_code,
            sample_blocks_current=sample_blocks_current,
            ci_shortfall_streak=ci_shortfall_streak,
            diagnostics=diagnostics,
            n_candidates=2,
            total_elems_global=1024,
            target_ratio=0.25,
            cutoff_tolerance=0.04,
            sample_blocks_min=4,
            sample_blocks_cap=64,
        )
        torch.cuda.synchronize()
        assert int(decision_code.item()) == 3
        assert int(sample_blocks_current.item()) == 16
        assert int(ci_shortfall_streak.item()) == 0

    for expected_sample, expected_streak in ((16, 1), (16, 2), (32, 0)):
        decision_code.fill_(3)
        launch_probe_decision(
            candidate_taus=candidate_taus,
            sum_q=sum_q,
            sum_q2=sum_q2,
            n_blocks=n_blocks,
            stratum_total_elems=stratum_total,
            total_th=total_th,
            probe_reject=probe_reject,
            decision_code=decision_code,
            sample_blocks_current=sample_blocks_current,
            ci_shortfall_streak=ci_shortfall_streak,
            diagnostics=diagnostics,
            n_candidates=2,
            total_elems_global=1024,
            target_ratio=0.25,
            cutoff_tolerance=0.04,
            sample_blocks_min=4,
            sample_blocks_cap=64,
            run_if_decision_code=3,
        )
        torch.cuda.synchronize()
        assert int(decision_code.item()) == 4
        assert int(sample_blocks_current.item()) == expected_sample
        assert int(ci_shortfall_streak.item()) == expected_streak

    assert int(probe_reject.item()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_probe_retry_helpers_are_device_guarded():
    device = torch.device("cuda")
    candidate_taus = torch.tensor([1.0, 2.0, 4.0, 8.0], device=device)
    diagnostics = torch.zeros(8, dtype=torch.float32, device=device)
    diagnostics[0] = 0.50
    diagnostics[3] = 2.0
    decision_code = torch.full((), 3, dtype=torch.int32, device=device)
    sum_q = torch.ones((1, 4), dtype=torch.float32, device=device)
    sum_q2 = torch.ones((1, 4), dtype=torch.float32, device=device)
    n_blocks = torch.ones((1, 4), dtype=torch.int32, device=device)

    launch_retry_candidates_if_code(
        candidate_taus=candidate_taus,
        diagnostics=diagnostics,
        decision_code=decision_code,
        n_candidates=4,
        target_ratio=0.25,
        run_if_decision_code=3,
    )
    launch_zero_probe_accumulators_if_code(
        sum_q=sum_q,
        sum_q2=sum_q2,
        n_blocks=n_blocks,
        decision_code=decision_code,
        run_if_decision_code=3,
    )
    torch.cuda.synchronize()

    retry_taus = candidate_taus.detach().cpu().tolist()
    assert retry_taus[0] == pytest.approx(2.0)
    assert retry_taus[1] > retry_taus[0]
    assert retry_taus[2] > retry_taus[1]
    assert float(sum_q.sum().item()) == 0.0
    assert float(sum_q2.sum().item()) == 0.0
    assert int(n_blocks.sum().item()) == 0

    decision_code.fill_(2)
    candidate_taus.copy_(torch.tensor([1.0, 2.0, 4.0, 8.0], device=device))
    sum_q.fill_(1.0)
    launch_retry_candidates_if_code(
        candidate_taus=candidate_taus,
        diagnostics=diagnostics,
        decision_code=decision_code,
        n_candidates=4,
        target_ratio=0.25,
        run_if_decision_code=3,
    )
    launch_zero_probe_accumulators_if_code(
        sum_q=sum_q,
        sum_q2=sum_q2,
        n_blocks=n_blocks,
        decision_code=decision_code,
        run_if_decision_code=3,
    )
    torch.cuda.synchronize()

    assert candidate_taus.detach().cpu().tolist() == pytest.approx([1.0, 2.0, 4.0, 8.0])
    assert float(sum_q.sum().item()) == pytest.approx(4.0)
