"""Rank topology helpers for proposal checkpoint ownership.

The proposal path only needs a small, explicit view of the training topology:
which ranks participate in the global K-budget, which rank owns durable writes,
and where each rank's checkpoint shard lives.  Keep this separate from the
checkpoint engines so EP/PP ownership does not leak into selection or storage
logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import torch


def _dist_ready() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )


@dataclass(frozen=True)
class ProposalTopology:
    global_rank: int
    world_size: int
    dp_rank: int
    dp_size: int
    pp_rank: int
    pp_size: int
    ep_rank: int
    ep_size: int
    tp_rank: int
    tp_size: int

    @property
    def is_global_rank0(self) -> bool:
        return self.global_rank == 0

    @property
    def writes_expert(self) -> bool:
        # EP/PP shards are unique on each rank.  DP replicas are identical, so
        # only the canonical DP rank persists them.
        return self.dp_rank == 0

    @property
    def writes_nonexpert(self) -> bool:
        # Non-expert tensors are replicated across EP ranks within a PP stage.
        # Disk mode shards the non-expert FC tensor set across EP replicas to
        # avoid making EP0 copy/write the whole replicated payload.  DP replicas
        # are still duplicates, so only the canonical DP rank participates.
        return self.dp_rank == 0

    @property
    def namespace(self) -> str:
        return f"dp{self.dp_rank}/pp{self.pp_rank}/ep{self.ep_rank}"

    @property
    def canonical_nonexpert_namespace(self) -> str:
        return f"dp{self.dp_rank}/pp{self.pp_rank}/ep0"

    @property
    def selection_ranks(self) -> List[int]:
        # Current implementation supports EP/PP with one DP replica.  Keeping
        # this list explicit makes the future DP extension local to this file.
        return list(range(self.world_size))

    def rank_root(self, cp_root: Path) -> Path:
        root = Path(cp_root)
        if self.world_size <= 1:
            return root
        return root / self.namespace

    def expert_dir(self, cp_root: Path) -> Path:
        return self.rank_root(cp_root) / "expert"

    def nonexpert_dir(self, cp_root: Path) -> Path:
        root = Path(cp_root)
        if self.world_size <= 1:
            return root / "non_expert"
        return root / self.canonical_nonexpert_namespace / "non_expert"


def topology_from_args(args: Any) -> ProposalTopology:
    world = int(torch.distributed.get_world_size()) if _dist_ready() else 1
    rank = int(torch.distributed.get_rank()) if _dist_ready() else 0
    tp_size = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    pp_size = int(getattr(args, "pipeline_model_parallel_size", 1) or 1)
    ep_size = int(
        getattr(args, "moe_expert_parallel_size", getattr(args, "ep_size", 1))
        or 1
    )
    if tp_size != 1:
        raise NotImplementedError(
            f"proposal: tensor parallel is not supported yet (tp={tp_size})."
        )
    denom = max(1, tp_size * pp_size * ep_size)
    if world % denom != 0:
        raise RuntimeError(
            f"proposal: world_size={world} is not divisible by "
            f"tp×pp×ep={tp_size}×{pp_size}×{ep_size}."
        )
    dp_size = world // denom
    if dp_size != 1:
        raise NotImplementedError(
            "proposal: DP replicas need a dp0-only proposal process group and "
            "cross-DP recovery policy; EP/PP support is enabled first."
        )

    # Match Megatron's PP group layout:
    #   pipeline group i = range(i, world, world / pp_size)
    num_pp_groups = max(1, world // max(1, pp_size))
    pp_rank = rank // num_pp_groups
    rank_inside_pp_slice = rank % num_pp_groups
    ep_rank = rank_inside_pp_slice % max(1, ep_size)
    dp_rank = rank_inside_pp_slice // max(1, ep_size)
    return ProposalTopology(
        global_rank=rank,
        world_size=world,
        dp_rank=dp_rank,
        dp_size=dp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        ep_rank=ep_rank,
        ep_size=ep_size,
        tp_rank=0,
        tp_size=tp_size,
    )


__all__ = ["ProposalTopology", "topology_from_args"]
