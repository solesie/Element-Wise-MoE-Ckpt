import torch

from megatron.proposal.refresh.fp32_state import Fp32StateMapper, set_mapper
from megatron.proposal.storage.recovery import _apply_zero12_partition_payload


class _Engine:
    def __init__(self, optimizer):
        self.optimizer = optimizer


class _EmptyFp16Candidate:
    def __init__(self, next_optimizer):
        self.fp32_groups_flat = [torch.empty(0, dtype=torch.float32)]
        self.fp16_groups = []
        self.optimizer = next_optimizer


class _Bf16ParamsInFp16Candidate:
    def __init__(self, params, next_optimizer):
        self.fp32_groups_flat = [torch.empty(sum(p.numel() for p in params))]
        self.fp16_groups = [params]
        self.optimizer = next_optimizer


class _Bf16Candidate:
    def __init__(self, params):
        self.bf16_groups = [params]
        total = sum(int(p.numel()) for p in params)
        self.fp32_groups_flat_partition = [torch.arange(total, dtype=torch.float32)]
        self.fp32_groups_gradients_flat = [torch.zeros(total, dtype=torch.float32)]


class _HpMapping:
    def __init__(self, grad_views):
        self._grad_views = grad_views

    def get_lp_grad_fragment(self, index_in_param_group):
        return self._grad_views[index_in_param_group]


class _Zero12Candidate:
    def __init__(self, params, partition_flat, params_in_partition, first_offset):
        self.bit16_groups = [params]
        self.single_partition_of_fp32_groups = [partition_flat]
        self.params_in_partition = [params_in_partition]
        self.first_offset = [first_offset]
        self.partition_size = [int(partition_flat.numel())]
        self.fp32_groups_gradients_flat = [torch.zeros_like(partition_flat)]
        self.optimizer = torch.optim.SGD([partition_flat], lr=0.1)

    def update_lp_params(self):
        self.update_lp_called = True


def test_mapper_skips_empty_fp16_candidate_and_uses_bf16_grad_flat():
    p0 = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    p1 = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    torch_opt = torch.optim.SGD([p0, p1], lr=0.1)
    bf16_candidate = _Bf16Candidate([p0, p1])
    engine = _Engine(_EmptyFp16Candidate(bf16_candidate))

    mapper = Fp32StateMapper(engine, torch_opt, torch.nn.Module())

    assert mapper.mode == "bf16_zero0"
    p1_grad = mapper.get_grad_accum_view(p1)
    p1_grad.add_(7.0)
    assert torch.equal(
        bf16_candidate.fp32_groups_gradients_flat[0],
        torch.tensor([0.0, 0.0, 7.0, 7.0, 7.0]),
    )


def test_mapper_uses_deepspeed_hp_mapping_grad_fragment_fallback():
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    grad = torch.zeros(4, dtype=torch.float32)
    p._hp_mapping = _HpMapping([grad])
    p._index_in_param_group = 0
    torch_opt = torch.optim.SGD([p], lr=0.1)

    mapper = Fp32StateMapper(None, torch_opt, torch.nn.Module())

    assert mapper.mode == "unrecognised"
    view = mapper.get_grad_accum_view(p)
    view.add_(3.0)
    assert torch.equal(grad, torch.full((4,), 3.0))


def test_mapper_searches_training_optimizer_chain_when_engine_has_no_optimizer():
    p0 = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    torch_opt = torch.optim.SGD([p0], lr=0.1)
    bf16_candidate = _Bf16Candidate([p0])
    training_optimizer = _EmptyFp16Candidate(bf16_candidate)

    mapper = Fp32StateMapper(_Engine(None), training_optimizer, torch.nn.Module())

    assert mapper.mode == "bf16_zero0"
    p0_grad = mapper.get_grad_accum_view(p0)
    p0_grad.fill_(3.0)
    assert torch.equal(
        bf16_candidate.fp32_groups_gradients_flat[0],
        torch.tensor([3.0, 3.0, 3.0, 3.0]),
    )


def test_mapper_accepts_deepspeed_fp16_optimizer_with_bf16_params():
    p0 = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    p1 = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    torch_opt = torch.optim.SGD([p0, p1], lr=0.1)
    bf16_candidate = _Bf16Candidate([p0, p1])
    engine = _Engine(_Bf16ParamsInFp16Candidate([p0, p1], bf16_candidate))

    mapper = Fp32StateMapper(engine, torch_opt, torch.nn.Module())

    assert mapper.mode == "fp16_zero0"
    view = mapper.get_grad_accum_view(p1)
    assert view.dtype == torch.bfloat16
    view.fill_(2.0)
    assert torch.equal(p1.grad, torch.full_like(p1, 2.0))


def test_zero12_local_fragments_follow_partition_not_tensor_boundary():
    p0 = torch.nn.Parameter(torch.ones(4, dtype=torch.float16))
    p1 = torch.nn.Parameter(torch.ones(6, dtype=torch.float16))
    part = torch.arange(5, dtype=torch.float32)
    candidate = _Zero12Candidate([p0, p1], part, [p1], first_offset=1)
    mapper = Fp32StateMapper(candidate, candidate.optimizer, torch.nn.Module())

    frags = mapper.zero12_local_fragments([("p0", p0), ("p1", p1)])

    assert mapper.mode == "zero12"
    assert len(frags) == 1
    name, param, group_idx, local_offset, param_offset, numel = frags[0]
    assert name == "p1"
    assert param is p1
    assert (group_idx, local_offset, param_offset, numel) == (0, 0, 1, 5)


def test_zero12_partition_payload_recovery_updates_local_master_and_state():
    p0 = torch.nn.Parameter(torch.ones(4, dtype=torch.float16))
    p1 = torch.nn.Parameter(torch.ones(6, dtype=torch.float16))
    part = torch.zeros(5, dtype=torch.float32)
    candidate = _Zero12Candidate([p0, p1], part, [p1], first_offset=1)
    candidate.optimizer.state[part]["exp_avg"] = torch.zeros_like(part)
    candidate.optimizer.state[part]["exp_avg_sq"] = torch.zeros_like(part)
    mapper = Fp32StateMapper(candidate, candidate.optimizer, torch.nn.Module())
    set_mapper(mapper)
    try:
        _apply_zero12_partition_payload(
            {
                "frag0:param": torch.arange(10, 15, dtype=torch.float32),
                "frag0:exp_avg": torch.arange(20, 25, dtype=torch.float32),
                "frag0:exp_avg_sq": torch.arange(30, 35, dtype=torch.float32),
            },
            {
                "payload_kind": "zero12_nonexpert_partition",
                "owner_global_rank": 0,
                "step": 7,
                "fragments": [{
                    "index": 0,
                    "name": "p1",
                    "group_idx": 0,
                    "local_offset": 0,
                    "param_offset": 1,
                    "numel": 5,
                    "shape": [6],
                }],
            },
        )
    finally:
        set_mapper(None)

    torch.testing.assert_close(part, torch.arange(10, 15, dtype=torch.float32))
    torch.testing.assert_close(
        candidate.optimizer.state[part]["exp_avg"],
        torch.arange(20, 25, dtype=torch.float32),
    )
    torch.testing.assert_close(
        candidate.optimizer.state[part]["exp_avg_sq"],
        torch.arange(30, 35, dtype=torch.float32),
    )
    assert getattr(candidate, "update_lp_called", False)
