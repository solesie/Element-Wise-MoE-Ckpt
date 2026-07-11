import pytest

import megatron.proposal.controller as controller
from deepspeed.runtime.pipe.schedule import TrainSchedule
from megatron.proposal.controller import (
    PostStepCheckpointTask,
    _install_pipeline_optimizer_fence,
    _stage0_bubble_deferral_enabled,
)
from megatron.proposal.distributed import ProposalTopology


def _topology(*, pp_size: int, pp_rank: int) -> ProposalTopology:
    return ProposalTopology(
        global_rank=pp_rank,
        world_size=pp_size,
        dp_rank=0,
        dp_size=1,
        pp_rank=pp_rank,
        pp_size=pp_size,
        ep_rank=0,
        ep_size=1,
        tp_rank=0,
        tp_size=1,
    )


def test_stage0_deferral_is_generic_across_pipeline_sizes():
    for pp_size in (2, 3, 4, 5, 8):
        assert _stage0_bubble_deferral_enabled(
            _topology(pp_size=pp_size, pp_rank=0),
            enable_zbh1_pipeline=False,
        )
        for pp_rank in range(1, pp_size):
            assert not _stage0_bubble_deferral_enabled(
                _topology(pp_size=pp_size, pp_rank=pp_rank),
                enable_zbh1_pipeline=False,
            )

    assert not _stage0_bubble_deferral_enabled(
        _topology(pp_size=1, pp_rank=0),
        enable_zbh1_pipeline=False,
    )
    assert not _stage0_bubble_deferral_enabled(
        _topology(pp_size=4, pp_rank=0),
        enable_zbh1_pipeline=True,
    )


@pytest.mark.parametrize("stages", (2, 3, 4, 5, 8))
@pytest.mark.parametrize("micro_batches", (1, 2, 4, 8, 32))
def test_stage0_release_boundary_is_first_recv_grad(
    stages, micro_batches,
):
    steps = list(TrainSchedule(
        micro_batches=micro_batches,
        stages=stages,
        stage_id=0,
    ))
    names_by_step = [
        [type(command).__name__ for command in commands]
        for commands in steps
    ]
    first_recv_grad = next(
        step_id for step_id, names in enumerate(names_by_step)
        if "RecvGrad" in names
    )
    first_backward = next(
        step_id for step_id, names in enumerate(names_by_step)
        if "BackwardPass" in names
    )
    assert first_recv_grad == first_backward
    assert first_recv_grad == 2 * stages - 1


class ForwardPass:
    pass


class BackwardPass:
    pass


class RecvGrad:
    pass


class OptimizerStep:
    pass


class _FakeEvent:
    def __init__(self, calls):
        self.calls = calls

    def record(self, _stream):
        self.calls.append("record")


class _FailingEvent(_FakeEvent):
    def record(self, _stream):
        super().record(_stream)
        raise RuntimeError("event record failed")


class _FakeHostEvent:
    def __init__(self, calls):
        self.calls = calls
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self.calls.append("release")
        self._set = True

    def wait(self, timeout=None):
        del timeout
        return self._set


class _FakeEngine:
    def __init__(
        self,
        calls,
        *,
        micro_batches=4,
        num_stages=2,
        include_recv=True,
        recv_raises=False,
    ):
        def forward(_engine, **_kwargs):
            calls.append("forward")

        def optimizer(_engine, **_kwargs):
            calls.append("optimizer")

        def backward(_engine, **_kwargs):
            calls.append("backward")

        def recv_grad(_engine, **_kwargs):
            calls.append("recv_grad")
            if recv_raises:
                raise RuntimeError("recv failed")

        self._INSTRUCTION_MAP = {
            ForwardPass: forward,
            BackwardPass: backward,
            OptimizerStep: optimizer,
        }
        if include_recv:
            self._INSTRUCTION_MAP[RecvGrad] = recv_grad
        self.micro_batches = int(micro_batches)
        self.num_stages = int(num_stages)

    def train_batch(self):
        return None


def test_first_recv_is_posted_before_worker_release(monkeypatch):
    calls = []
    engine = _FakeEngine(calls)
    ctrl = controller.get_controller()
    old_task = ctrl._latest_post_step_checkpoint_task
    old_enabled = ctrl._enabled
    task = PostStepCheckpointTask(
        cuda_event=None,
        iter_T=7,
        defer_stage0_expert_emission=True,
        expert_emission_cuda_event=_FakeEvent(calls),
    )
    task.expert_emission_release = _FakeHostEvent(calls)
    ctrl._latest_post_step_checkpoint_task = task
    ctrl._enabled = False
    monkeypatch.setattr(controller.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(controller.torch.cuda, "current_stream", lambda: object())
    try:
        _install_pipeline_optimizer_fence([engine])
        engine._INSTRUCTION_MAP[ForwardPass](engine, buffer_id=0)
        assert not task.expert_emission_release.is_set()
        engine._INSTRUCTION_MAP[RecvGrad](engine, buffer_id=0)
        assert calls == ["forward", "record", "recv_grad", "release"]
        assert task.expert_emission_release.is_set()
        engine._INSTRUCTION_MAP[RecvGrad](engine, buffer_id=0)
        assert calls == [
            "forward", "record", "recv_grad", "release", "recv_grad",
        ]
    finally:
        ctrl._latest_post_step_checkpoint_task = old_task
        ctrl._enabled = old_enabled


def test_release_after_first_recv_is_independent_of_pipeline_depth(monkeypatch):
    calls = []
    engine = _FakeEngine(calls, micro_batches=2, num_stages=4)
    ctrl = controller.get_controller()
    old_task = ctrl._latest_post_step_checkpoint_task
    old_enabled = ctrl._enabled
    task = PostStepCheckpointTask(
        cuda_event=None,
        iter_T=9,
        defer_stage0_expert_emission=True,
    )
    ctrl._latest_post_step_checkpoint_task = task
    ctrl._enabled = False
    monkeypatch.setattr(controller.torch.cuda, "is_available", lambda: False)
    try:
        _install_pipeline_optimizer_fence([engine])
        engine._INSTRUCTION_MAP[ForwardPass](engine, buffer_id=0)
        assert not task.expert_emission_release.is_set()
        engine._INSTRUCTION_MAP[RecvGrad](engine, buffer_id=0)
        assert task.expert_emission_release.is_set()
    finally:
        ctrl._latest_post_step_checkpoint_task = old_task
        ctrl._enabled = old_enabled


@pytest.mark.parametrize("failure_site", ("record", "recv"))
def test_recv_failure_still_releases_deferred_worker(monkeypatch, failure_site):
    calls = []
    engine = _FakeEngine(calls, recv_raises=(failure_site == "recv"))
    ctrl = controller.get_controller()
    old_task = ctrl._latest_post_step_checkpoint_task
    old_enabled = ctrl._enabled
    task = PostStepCheckpointTask(
        cuda_event=None,
        iter_T=11,
        defer_stage0_expert_emission=True,
        expert_emission_cuda_event=(
            _FailingEvent(calls) if failure_site == "record" else _FakeEvent(calls)
        ),
    )
    task.expert_emission_release = _FakeHostEvent(calls)
    ctrl._latest_post_step_checkpoint_task = task
    ctrl._enabled = False
    monkeypatch.setattr(controller.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(controller.torch.cuda, "current_stream", lambda: object())
    try:
        _install_pipeline_optimizer_fence([engine])
        with pytest.raises(RuntimeError):
            engine._INSTRUCTION_MAP[RecvGrad](engine, buffer_id=0)
        assert task.expert_emission_release.is_set()
        expected = ["record", "release"]
        if failure_site == "recv":
            expected = ["record", "recv_grad", "release"]
        assert calls == expected
    finally:
        ctrl._latest_post_step_checkpoint_task = old_task
        ctrl._enabled = old_enabled


def test_optimizer_remains_custom_schedule_release_fallback(monkeypatch):
    calls = []
    engine = _FakeEngine(calls, include_recv=False)
    ctrl = controller.get_controller()
    old_task = ctrl._latest_post_step_checkpoint_task
    old_enabled = ctrl._enabled
    task = PostStepCheckpointTask(
        cuda_event=None,
        iter_T=10,
        defer_stage0_expert_emission=True,
    )
    ctrl._latest_post_step_checkpoint_task = task
    ctrl._enabled = False
    monkeypatch.setattr(controller.torch.cuda, "is_available", lambda: False)
    try:
        _install_pipeline_optimizer_fence([engine])
        engine._INSTRUCTION_MAP[OptimizerStep](engine)
        assert task.expert_emission_release.is_set()
    finally:
        ctrl._latest_post_step_checkpoint_task = old_task
        ctrl._enabled = old_enabled


def test_optimizer_hook_is_deadlock_safe_release_fallback(monkeypatch):
    calls = []
    engine = _FakeEngine(calls)
    ctrl = controller.get_controller()
    old_task = ctrl._latest_post_step_checkpoint_task
    old_enabled = ctrl._enabled
    task = PostStepCheckpointTask(
        cuda_event=None,
        iter_T=8,
        defer_stage0_expert_emission=True,
    )
    ctrl._latest_post_step_checkpoint_task = task
    ctrl._enabled = False
    monkeypatch.setattr(controller.torch.cuda, "is_available", lambda: False)
    try:
        _install_pipeline_optimizer_fence([engine])
        engine._INSTRUCTION_MAP[OptimizerStep](engine)
        assert task.expert_emission_release.is_set()
        assert calls == ["optimizer"]
    finally:
        ctrl._latest_post_step_checkpoint_task = old_task
        ctrl._enabled = old_enabled
