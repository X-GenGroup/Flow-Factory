# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from flow_factory.hparams import Arguments
from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter
from flow_factory.scheduler.flow_match_euler_discrete import FlowMatchEulerDiscreteSDEScheduler
from flow_factory.trainers.registry import get_trainer_class
from flow_factory.trainers.rl.drtune import spaced_training_steps, unroll_sparse_sampling


class NonlinearDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.29))
        self.inputs = []

    def forward(self, *, hidden_states, encoder_hidden_states, **kwargs):
        self.inputs.append((hidden_states.requires_grad, torch.is_grad_enabled()))
        bias = encoder_hidden_states.view(-1, 1, 1, 1)
        return (self.weight * hidden_states.sin() + self.weight.square() + bias * self.weight,)


@pytest.mark.parametrize("selected", [range(1), range(0, 4, 2), range(4)])
@pytest.mark.parametrize("guidance", [1.0, 2.5])
def test_sparse_gradients_through_native_euler_forward(selected, guidance):
    times = torch.tensor([1000.0, 700.0, 460.0, 190.0])
    initial = torch.tensor([0.6, -0.4]).reshape(1, 2, 1, 1).requires_grad_()
    adapter = SimpleNamespace(
        transformer=NonlinearDenoiser(),
        scheduler=FlowMatchEulerDiscreteSDEScheduler(dynamics_type="ODE"),
    )
    weight = adapter.transformer.weight
    positive = torch.ones(1, 1)
    negative = torch.zeros(1, 1)

    def advance(state, current, following, *, prediction_gradients):
        return SD3_5Adapter.forward(
            adapter,
            t=current,
            t_next=following,
            latents=state,
            prompt_embeds=positive,
            pooled_prompt_embeds=positive,
            negative_prompt_embeds=negative,
            negative_pooled_prompt_embeds=negative,
            guidance_scale=guidance,
            return_kwargs=["next_latents"],
            compute_log_prob=False,
            prediction_gradients=prediction_gradients,
        ).next_latents

    endpoint = unroll_sparse_sampling(initial, times, selected, advance)
    gradient, initial_gradient = torch.autograd.grad(
        endpoint.square().sum(), (weight, initial), allow_unused=True
    )
    assert initial_gradient is None
    assert adapter.transformer.inputs == [(False, i in selected) for i in range(4)]

    z = initial.detach().double()
    w = weight.detach().double()
    sensitivity = torch.zeros_like(z)
    for index, step in enumerate((-0.3, -0.24, -0.27, -0.19)):
        if index in selected:
            sensitivity += step * (z.sin() + 2 * w + guidance)
        z = z + step * (w * z.sin() + w.square() + guidance * w)
    expected_gradient = (2 * z * sensitivity).sum()
    torch.testing.assert_close(endpoint, z.float())
    torch.testing.assert_close(gradient, expected_gradient.float())
    assert gradient.abs() > 1e-3

    with torch.no_grad():
        native = initial.detach()
        for index, current in enumerate(times):
            following = times[index + 1] if index + 1 < len(times) else torch.zeros_like(current)
            native = advance(native, current, following, prediction_gradients="full")
    torch.testing.assert_close(endpoint, native, rtol=0, atol=0)

    def drop_state(state, current, following, *, prediction_gradients):
        return advance(
            state.detach(), current, following, prediction_gradients=prediction_gradients
        )

    broken = unroll_sparse_sampling(initial, times, selected, drop_state)
    if 3 not in selected:
        assert not broken.requires_grad
    else:
        dropped = torch.autograd.grad(broken.square().sum(), weight)[0]
        assert not torch.isclose(dropped, gradient, rtol=1e-3)

    if len(selected) > 1:

        def keep_input_jacobian(state, current, following, *, prediction_gradients):
            mode = "full" if prediction_gradients == "parameters" else "none"
            return advance(state, current, following, prediction_gradients=mode)

        wrong = unroll_sparse_sampling(initial, times, selected, keep_input_jacobian)
        wrong_gradient = torch.autograd.grad(wrong.square().sum(), weight)[0]
        assert not torch.isclose(wrong_gradient, gradient, rtol=1e-3)


@pytest.mark.parametrize(("total", "count"), [(1, 1), (6, 1), (6, 3), (7, 3), (6, 6)])
def test_spaced_mask_is_seeded_and_within_bounds(total, count):
    masks = set()
    for seed in range(64):
        generator = torch.Generator().manual_seed(seed)
        selected = spaced_training_steps(total, count, generator)
        assert selected == spaced_training_steps(total, count, generator.manual_seed(seed))
        assert len(selected) == count
        assert 0 <= selected[0] <= selected[-1] < total
        assert selected.step == total // count
        masks.add(selected[0])
    assert len(masks) == total - (count - 1) * (total // count)


_EXAMPLE = Path("examples/drtune/lora/sd3_5/pickscore.yaml")


def test_drtune_recipe_and_registry():
    config = Arguments.load_from_yaml(_EXAMPLE)
    trainer = get_trainer_class("drtune")
    assert trainer.execution_contract == config.training_args.execution_contract
    assert config.training_args.drtune_backprop_steps == 2
    assert config.training_args.get_num_train_timesteps(config) == 1


@pytest.mark.parametrize("invalid", [0, True, 41])
def test_drtune_rejects_invalid_step_counts(invalid):
    raw = yaml.safe_load(_EXAMPLE.read_text())
    raw["train"]["drtune_backprop_steps"] = invalid
    with pytest.raises(ValueError, match="drtune_backprop_steps"):
        Arguments.from_dict(raw)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("scheduler", "dynamics_type", "Flow-SDE", "requires scheduler.dynamics_type"),
        ("model", "model_type", "flux1", "supports the SD3.5 adapter"),
        ("model", "target_components", ["transformer", "vae"], "requires a frozen VAE"),
    ],
)
def test_drtune_admits_only_supported_paths(section, key, value, message):
    raw = yaml.safe_load(_EXAMPLE.read_text())
    raw[section][key] = value
    with pytest.raises(ValueError, match=message):
        Arguments.from_dict(raw)
