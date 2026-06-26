import pytest
import torch

from megatron.proposal.grad_lease import GradScoreLease


class _Group:
    def __init__(self, grad):
        self.grad = grad


class _FakeZeroOptimizer:
    def __init__(self, grad):
        self.single_partition_of_fp32_groups = [_Group(grad)]
        self.called = False

    def _optimizer_step(self, group_no, marker=None):
        self.called = (int(group_no) == 0 and marker == "ok")
        return "done"


class _FakeBF16Optimizer:
    def __init__(self, grad):
        self.fp32_groups_gradients_flat = [grad]
        self.fp32_groups_gradients = [[grad.narrow(0, 0, grad.numel())]]
        self.fp32_groups_has_gradients = [[True]]
        self.clear_called = False
        self.update_called = False

    def clear_hp_grads(self):
        self.clear_called = True
        for grad in self.fp32_groups_gradients_flat:
            grad.zero_()
        for i, group in enumerate(self.fp32_groups_gradients):
            self.fp32_groups_has_gradients[i] = [False] * len(group)

    def update_hp_grads(self, marker=None):
        self.update_called = (marker == "ok")
        return "updated"


def test_grad_score_lease_ignores_non_zero_optimizer():
    lease = GradScoreLease(object())

    assert not lease.install()
    assert not lease.enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grad_score_lease_keeps_existing_cuda_grad_reference():
    grad = torch.arange(16, device="cuda", dtype=torch.float32)
    opt = _FakeZeroOptimizer(grad)
    lease = GradScoreLease(opt)

    assert lease.install()
    assert opt._optimizer_step(0, marker="ok") == "done"
    assert opt.called
    assert lease.has_leases

    view = lease.view_for_group(0, 4, 5)
    assert view is not None
    assert view.data_ptr() == grad.narrow(0, 4, 5).data_ptr()
    torch.testing.assert_close(view, grad[4:9])

    lease.release()
    assert not lease.has_leases
    assert lease.view_for_group(0, 0, 1) is None

    lease.uninstall()
    assert opt._optimizer_step(0, marker="ok") == "done"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grad_score_lease_defers_bf16_flat_clear_until_release():
    grad = torch.arange(16, device="cuda", dtype=torch.float32)
    original = grad.clone()
    opt = _FakeBF16Optimizer(grad)
    lease = GradScoreLease(opt)

    assert lease.install()
    opt.clear_hp_grads()
    assert not opt.clear_called
    assert lease.has_leases
    torch.testing.assert_close(grad, original)
    assert opt.fp32_groups_has_gradients == [[False]]

    opt.clear_hp_grads()
    torch.testing.assert_close(grad, original)

    view = lease.view_for_group(0, 3, 4)
    assert view is not None
    assert view.data_ptr() == grad.narrow(0, 3, 4).data_ptr()

    lease.release()
    assert not lease.has_leases
    torch.testing.assert_close(grad, torch.zeros_like(grad))

    lease.uninstall()
    grad.copy_(original)
    opt.fp32_groups_has_gradients = [[True]]
    opt.clear_hp_grads()
    assert opt.clear_called
    torch.testing.assert_close(grad, torch.zeros_like(grad))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grad_score_lease_clears_bf16_flat_before_next_update():
    grad = torch.arange(8, device="cuda", dtype=torch.float32)
    opt = _FakeBF16Optimizer(grad)
    lease = GradScoreLease(opt)

    assert lease.install()
    opt.clear_hp_grads()
    assert lease.has_leases

    assert opt.update_hp_grads(marker="ok") == "updated"
    assert opt.update_called
    assert not lease.has_leases
    torch.testing.assert_close(grad, torch.zeros_like(grad))

    lease.uninstall()
