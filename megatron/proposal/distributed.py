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
        # Expert optimizer state follows DeepSpeed's local DP ownership.  With
        # BF16 optimizer even ZeRO-0 exposes fp32 master/state as local DP
        # partitions, so every DP rank owns its expert shard.
        return True

    @property
    def writes_nonexpert(self) -> bool:
        # ZeRO-0 non-expert tensors are replicated, so every DP×EP replica can
        # own one disjoint tensor-name shard.  ZeRO-1/2 uses the mapper's local
        # partition fragments on the same per-rank writer path.
        return True

    @property
    def namespace(self) -> str:
        return f"dp{self.dp_rank}/pp{self.pp_rank}/ep{self.ep_rank}"

    @property
    def expert_namespace(self) -> str:
        return f"dp{self.dp_rank}/pp{self.pp_rank}/ep{self.ep_rank}"

    @property
    def nonexpert_namespace(self) -> str:
        if self.dp_size > 1:
            return f"pp{self.pp_rank}"
        return f"dp{self.dp_rank}/pp{self.pp_rank}"

    @property
    def selection_ranks(self) -> List[int]:
        # Proposal communication is a participation contract, not a write
        # ownership contract.  Every rank joins the same reductions/barriers;
        # local DP shard size decides each rank's value contribution.
        return list(range(max(1, int(self.world_size))))

    @property
    def nonexpert_shard_rank(self) -> int:
        return int(self.dp_rank) * max(1, int(self.ep_size)) + int(self.ep_rank)

    @property
    def nonexpert_shard_count(self) -> int:
        return max(1, int(self.dp_size) * max(1, int(self.ep_size)))

    def rank_root(self, cp_root: Path) -> Path:
        root = Path(cp_root)
        if self.world_size <= 1:
            return root
        return root / self.namespace

    def expert_dir(self, cp_root: Path) -> Path:
        root = Path(cp_root)
        if self.world_size <= 1:
            return root / "expert"
        return root / self.expert_namespace / "expert"

    def nonexpert_dir(self, cp_root: Path) -> Path:
        root = Path(cp_root)
        if self.world_size <= 1:
            return root / "non_expert"
        return root / self.nonexpert_namespace / "non_expert"


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
