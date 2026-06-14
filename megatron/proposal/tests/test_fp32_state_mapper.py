import torch

from megatron.proposal.refresh.fp32_state import Fp32StateMapper


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
