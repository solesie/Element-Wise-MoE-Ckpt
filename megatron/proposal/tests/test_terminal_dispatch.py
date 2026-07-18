import pytest

import megatron.proposal.controller as controller
from megatron.proposal.controller import PostStepCheckpointTask


class _Workers:
    def __init__(self, calls):
        self.calls = calls

    def drain(self):
        self.calls.append("drain")


class _Event:
    def __init__(self, calls, name):
        self.calls = calls
        self.name = name

    def synchronize(self):
        self.calls.append(f"sync:{self.name}")


def _save_controller_state(ctrl):
    return {
        "enabled": ctrl._enabled,
        "active": ctrl._active,
        "warmup_end": ctrl._warmup_end_iter,
        "workers": ctrl._checkpoint_workers,
        "latest": ctrl._latest_post_step_checkpoint_task,
        "last": ctrl._last_dispatched_iter_T,
    }


def _restore_controller_state(ctrl, old):
    ctrl._enabled = old["enabled"]
    ctrl._active = old["active"]
    ctrl._warmup_end_iter = old["warmup_end"]
    ctrl._checkpoint_workers = old["workers"]
    ctrl._latest_post_step_checkpoint_task = old["latest"]
    ctrl._last_dispatched_iter_T = old["last"]


def test_terminal_finalize_dispatches_final_step_and_drains(monkeypatch):
    calls = []
    ctrl = controller.get_controller()
    old = _save_controller_state(ctrl)
    ctrl._enabled = True
    ctrl._active = True
    ctrl._warmup_end_iter = 0
    ctrl._last_dispatched_iter_T = 7
    ctrl._checkpoint_workers = _Workers(calls)
    ctrl._latest_post_step_checkpoint_task = None
    old_world_size = ctrl._proposal_world_size
    ctrl._proposal_world_size = 1

    def dispatch(model, iter_T, **kwargs):
        calls.append(("dispatch", model, iter_T, kwargs))
        task = PostStepCheckpointTask(cuda_event=None, iter_T=iter_T)
        task.non_expert_d2h_event = _Event(calls, "nonexpert")
        task.expert_d2h_event = _Event(calls, "expert")
        ctrl._last_dispatched_iter_T = iter_T
        ctrl._latest_post_step_checkpoint_task = task

    monkeypatch.setattr(controller, "proposal_dispatch_post_iter", dispatch)
    try:
        model = object()
        controller.proposal_finalize_post_iter(model, 8)
        assert calls == [
            ("dispatch", model, 8, {
                "sync_processes": True,
                "defer_stage0_emission": False,
            }),
            "drain",
            "sync:nonexpert",
            "sync:expert",
        ]
        assert ctrl._latest_post_step_checkpoint_task is None
    finally:
        ctrl._proposal_world_size = old_world_size
        _restore_controller_state(ctrl, old)


def test_terminal_finalize_is_safe_when_step_was_already_dispatched(monkeypatch):
    calls = []
    ctrl = controller.get_controller()
    old = _save_controller_state(ctrl)
    ctrl._enabled = True
    ctrl._active = True
    ctrl._warmup_end_iter = 0
    ctrl._last_dispatched_iter_T = 8
    ctrl._checkpoint_workers = _Workers(calls)
    old_world_size = ctrl._proposal_world_size
    ctrl._proposal_world_size = 1
    task = PostStepCheckpointTask(cuda_event=None, iter_T=8)
    ctrl._latest_post_step_checkpoint_task = task

    def guarded_dispatch(_model, iter_T, **_kwargs):
        if int(iter_T) > ctrl._last_dispatched_iter_T:
            calls.append("unexpected-dispatch")

    monkeypatch.setattr(controller, "proposal_dispatch_post_iter", guarded_dispatch)
    try:
        controller.proposal_finalize_post_iter(object(), 8)
        assert calls == ["drain"]
        assert ctrl._latest_post_step_checkpoint_task is None
    finally:
        ctrl._proposal_world_size = old_world_size
        _restore_controller_state(ctrl, old)


@pytest.mark.parametrize("domain", ("non_expert", "expert"))
def test_terminal_finalize_propagates_worker_error(monkeypatch, domain):
    ctrl = controller.get_controller()
    old = _save_controller_state(ctrl)
    ctrl._enabled = True
    ctrl._active = True
    ctrl._warmup_end_iter = 0
    ctrl._last_dispatched_iter_T = 8
    ctrl._checkpoint_workers = _Workers([])
    old_world_size = ctrl._proposal_world_size
    ctrl._proposal_world_size = 1
    task = PostStepCheckpointTask(cuda_event=None, iter_T=8)
    setattr(task, f"{domain}_error", ValueError("worker failed"))
    ctrl._latest_post_step_checkpoint_task = task
    monkeypatch.setattr(controller, "proposal_dispatch_post_iter", lambda *_a, **_k: None)
    try:
        with pytest.raises(RuntimeError, match="terminal checkpoint failed"):
            controller.proposal_finalize_post_iter(object(), 8)
    finally:
        ctrl._proposal_world_size = old_world_size
        _restore_controller_state(ctrl, old)
