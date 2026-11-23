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
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter
from flow_factory.samples import ComponentTimes, LatentState
from flow_factory.scheduler.flow_match_euler_discrete import FlowMatchEulerDiscreteSDEScheduler
from flow_factory.trainers.registry import get_trainer_class
from flow_factory.trainers.rl.drtune import (
    spaced_training_steps,
    sparse_training_schedule,
    unroll_sparse_sampling,
)


class NonlinearDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.29))
        self.inputs = []

    def forward(self, *, hidden_states, encoder_hidden_states, **kwargs):
        self.inputs.append((hidden_states.requires_grad, torch.is_grad_enabled()))
        bias = encoder_hidden_states.view(-1, 1, 1, 1)
        return (self.weight * hidden_states.sin() + self.weight.square() + bias * self.weight,)


@pytest.mark.parametrize(
    ("selected", "stop"),
    [
        (range(1), None),
        (range(0, 4, 2), None),
        (range(4), None),
        (range(1), 1),
        (range(1), 3),
        (range(0, 3, 2), 3),
        (range(3), 3),
    ],
)
@pytest.mark.parametrize("guidance", [1.0, 2.5])
def test_sparse_gradients_through_native_euler_forward(selected, stop, guidance):
    times = torch.tensor([1000.0, 700.0, 460.0, 190.0])
    initial = torch.tensor([0.6, -0.4]).reshape(1, 2, 1, 1).requires_grad_()
    adapter = SimpleNamespace(
        transformer=NonlinearDenoiser(),
        scheduler=FlowMatchEulerDiscreteSDEScheduler(dynamics_type="ODE"),
        trajectory_component_order=("latent",),
        flow_velocity_direction="noise",
    )
    weight = adapter.transformer.weight
    positive = torch.ones(1, 1)
    negative = torch.zeros(1, 1)

    def advance(state, current, following, *, prediction_gradients, project_clean):
        output = SD3_5Adapter.forward(
            adapter,
            t=current,
            t_next=following,
            latents=state,
            prompt_embeds=positive,
            pooled_prompt_embeds=positive,
            negative_prompt_embeds=negative,
            negative_pooled_prompt_embeds=negative,
            guidance_scale=guidance,
            return_kwargs=["velocity" if project_clean else "next_latents"],
            compute_log_prob=False,
            prediction_gradients=prediction_gradients,
        )
        if project_clean:
            times = BaseAdapter.build_training_component_times(
                adapter, current.reshape(1), batch=None
            )
            return BaseAdapter.project_velocity_to_clean_state(
                adapter,
                LatentState({"latent": state}),
                times,
                LatentState({"latent": output.velocity}),
            ).components["latent"]
        return output.next_latents

    endpoint = unroll_sparse_sampling(initial, times, selected, advance, stop_after=stop)
    gradient, initial_gradient = torch.autograd.grad(
        endpoint.square().sum(), (weight, initial), allow_unused=True
    )
    assert initial_gradient is None
    visited = stop or len(times)
    assert adapter.transformer.inputs == [(False, i in selected) for i in range(visited)]

    z = initial.detach().double()
    w = weight.detach().double()
    sensitivity = torch.zeros_like(z)
    sigmas = (1.0, 0.7, 0.46, 0.19)[:visited] + (0.0,)
    for index, (sigma, following) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
        step = following - sigma
        if index in selected:
            sensitivity += step * (z.sin() + 2 * w + guidance)
        z = z + step * (w * z.sin() + w.square() + guidance * w)
    expected_gradient = (2 * z * sensitivity).sum()
    torch.testing.assert_close(endpoint, z.float())
    torch.testing.assert_close(gradient, expected_gradient.float())
    assert gradient.abs() > 1e-3

    with torch.no_grad():
        native = initial.detach()
        for index, current in enumerate(times[:visited]):
            terminal = index + 1 == visited
            following = torch.zeros_like(current) if terminal else times[index + 1]
            native = advance(
                native,
                current,
                following,
                prediction_gradients="full",
                project_clean=terminal and stop is not None,
            )
    torch.testing.assert_close(endpoint, native, rtol=0, atol=0)

    def drop_state(state, current, following, **kwargs):
        return advance(state.detach(), current, following, **kwargs)

    broken = unroll_sparse_sampling(initial, times, selected, drop_state, stop_after=stop)
    if visited - 1 not in selected:
        assert not broken.requires_grad
    elif len(selected) > 1:
        dropped = torch.autograd.grad(broken.square().sum(), weight)[0]
        assert not torch.isclose(dropped, gradient, rtol=1e-3)

    if len(selected) > 1:

        def keep_input_jacobian(state, current, following, *, prediction_gradients, project_clean):
            mode = "full" if prediction_gradients == "parameters" else "none"
            return advance(
                state, current, following, prediction_gradients=mode, project_clean=project_clean
            )

        wrong = unroll_sparse_sampling(
            initial, times, selected, keep_input_jacobian, stop_after=stop
        )
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


@pytest.mark.parametrize("stop_range", [None, (2, 2), (2, 5), (5, 5)])
def test_stop_precedes_mask_and_preserves_default(stop_range):
    seen = set()
    for seed in range(32):
        generator = torch.Generator().manual_seed(seed)
        stop, mask = sparse_training_schedule(5, 2, stop_range, generator)
        assert (stop, mask) == sparse_training_schedule(
            5, 2, stop_range, generator.manual_seed(seed)
        )
        if stop_range is None:
            assert (stop, mask) == (None, spaced_training_steps(5, 2, generator.manual_seed(seed)))
        else:
            assert stop_range[0] <= stop <= stop_range[1]
            assert len(mask) == 2
            assert 0 <= mask[0] <= mask[-1] < stop
            seen.add(stop)
    if stop_range is not None:
        assert seen == set(range(stop_range[0], stop_range[1] + 1))


_EXAMPLE = Path("examples/drtune/lora/sd3_5/pickscore.yaml")


def test_drtune_recipe_and_registry():
    config = Arguments.load_from_yaml(_EXAMPLE)
    trainer = get_trainer_class("drtune")
    assert trainer.execution_contract == config.training_args.execution_contract
    assert config.training_args.drtune_backprop_steps == 2
    assert tuple(config.training_args.drtune_stop_step_range) == (32, 36)
    assert config.training_args.get_num_train_timesteps(config) == 1


@pytest.mark.parametrize("invalid", [0, True, 41])
def test_drtune_rejects_invalid_step_counts(invalid):
    raw = yaml.safe_load(_EXAMPLE.read_text())
    raw["train"]["drtune_backprop_steps"] = invalid
    with pytest.raises(ValueError, match="drtune_backprop_steps"):
        Arguments.from_dict(raw)


@pytest.mark.parametrize("invalid", [[1, 3], [3, 2], [2, 41], [2], [2, True], "2,3"])
def test_drtune_rejects_invalid_stop_ranges(invalid):
    raw = yaml.safe_load(_EXAMPLE.read_text())
    raw["train"]["drtune_stop_step_range"] = invalid
    with pytest.raises(ValueError, match="drtune_stop_step_range"):
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
