"""LFSY append durability behavior."""

from pathlib import Path

import torch

from megatron.proposal.storage import expert_writer as expert_state_mod
from megatron.proposal.storage.expert_writer import ExpertState


def test_lfsy_append_fsyncs_by_default(tmp_path: Path, monkeypatch) -> None:
    state = ExpertState(
        {
            (0, 0): [
                ("mlp.w1", torch.Size([8]), 8),
            ],
        },
        output_dir=tmp_path,
        fresh=True,
    )
    ent = state._entries[(0, 0)]
    state._open_fresh_lossy(ent, step=0)

    fsynced_fds = []

    def _record_fsync(fd: int) -> None:
        fsynced_fds.append(int(fd))

    monkeypatch.setattr(expert_state_mod, "fsync_fd", _record_fsync)

    state.append_lossy_records_for_test(
        (0, 0),
        step=1,
        tensor_updates=[
            (
                "mlp.w1",
                torch.tensor([3], dtype=torch.int32),
                torch.tensor([11.0]),
                torch.tensor([1.0]),
                torch.tensor([0.1]),
            ),
        ],
    )

    assert fsynced_fds == [int(ent.lossy_fd)]

    state.shutdown()
