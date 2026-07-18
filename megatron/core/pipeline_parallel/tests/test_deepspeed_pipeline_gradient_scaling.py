import torch
from torch import nn

from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.utils import register_output_backward_hooks

from megatron.core.pipeline_parallel.deepspeed_compat import (
    configure_pipeline_gradient_scaling,
)


class _Stage:
    _backward_prologue_per_tensor = DeepSpeedEngine._backward_prologue_per_tensor

    def __init__(self, stage_id, num_stages, gas=8):
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.gas = gas
        self._scale_wrt_gas = True

    def is_last_stage(self):
        return self.stage_id == self.num_stages - 1

    def gradient_accumulation_steps(self):
        return self.gas


def test_only_nonfinal_pipeline_stages_disable_output_gradient_scaling():
    stages = [_Stage(stage_id=i, num_stages=3) for i in range(3)]

    for stage in stages:
        configure_pipeline_gradient_scaling(stage)

    assert [stage._scale_wrt_gas for stage in stages] == [False, False, True]


def test_nonfinal_stage_does_not_rescale_received_pipeline_gradient():
    stage = _Stage(stage_id=0, num_stages=2, gas=8)
    configure_pipeline_gradient_scaling(stage)

    weight = torch.tensor(3.0, requires_grad=True)
    output = weight * 2.0
    register_output_backward_hooks(
        output,
        preprocess_once_fn=lambda: None,
        preprocess_per_tensor_fn=stage._backward_prologue_per_tensor,
    )

    # The final stage has already divided this activation gradient by GAS.
    output.backward(torch.tensor(1.0 / stage.gas))

    assert weight.grad.item() == 2.0 / stage.gas


def _attach_deepspeed_output_hook(output, stage):
    register_output_backward_hooks(
        output,
        preprocess_once_fn=lambda: None,
        preprocess_per_tensor_fn=stage._backward_prologue_per_tensor,
    )


def test_pp2_raw_parameter_gradients_match_pp1_layer_by_layer():
    """Compare pre-optimizer gradients after a complete GAS window."""
    torch.manual_seed(1234)
    gas = 8
    pp1_layers = nn.ModuleList((nn.Linear(5, 7), nn.Linear(7, 3)))
    pp2_layers = nn.ModuleList((nn.Linear(5, 7), nn.Linear(7, 3)))
    pp2_layers.load_state_dict(pp1_layers.state_dict())

    pp1_last = _Stage(stage_id=0, num_stages=1, gas=gas)
    pp2_first = _Stage(stage_id=0, num_stages=2, gas=gas)
    pp2_last = _Stage(stage_id=1, num_stages=2, gas=gas)
    configure_pipeline_gradient_scaling(pp1_last)
    configure_pipeline_gradient_scaling(pp2_first)
    configure_pipeline_gradient_scaling(pp2_last)

    batches = [
        (torch.randn(4, 5), torch.randn(4, 3))
        for _ in range(gas)
    ]
    for inputs, targets in batches:
        pp1_output = pp1_layers[1](torch.tanh(pp1_layers[0](inputs)))
        _attach_deepspeed_output_hook(pp1_output, pp1_last)
        torch.nn.functional.mse_loss(pp1_output, targets).backward()

        pp2_hidden = torch.tanh(pp2_layers[0](inputs))
        _attach_deepspeed_output_hook(pp2_hidden, pp2_first)
        pp2_output = pp2_layers[1](pp2_hidden)
        _attach_deepspeed_output_hook(pp2_output, pp2_last)
        torch.nn.functional.mse_loss(pp2_output, targets).backward()

    for pp1_layer, pp2_layer in zip(pp1_layers, pp2_layers):
        assert torch.equal(pp2_layer.weight.grad, pp1_layer.weight.grad)
        assert torch.equal(pp2_layer.bias.grad, pp1_layer.bias.grad)
