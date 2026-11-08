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

"""Packed-latent CPU composition proof using DMD2's real objective paths.

This is not a native Flux multi-role run and does not modify the Wan native oracle.
"""

import pytest
import torch

from flow_factory.samples import BaseSample
from flow_factory.trainers.common import context_parallel
from tests.trainers.test_multirole_production_paths import _real_objective_trainer


@pytest.mark.parametrize("role", ["fake", "generator"])
@pytest.mark.parametrize("cp_size", [1, 2])
def test_packed_dmd2_real_objectives_and_independent_derivatives(monkeypatch, role, cp_size):
    trainer, adapter, bundle = _real_objective_trainer()
    starts = torch.linspace(-0.3, 1.1, 128).reshape(2, 4, 16)
    times = torch.tensor([1000.0, 500.0])
    batch = BaseSample.stack(
        [
            BaseSample(
                timesteps=times[index : index + 1],
                all_latents=torch.stack((value, value + bundle.generator.detach())),
                latent_index_map=torch.tensor([0, 1]),
                sampling_sample_id=index,
            )
            for index, value in enumerate(starts)
        ]
    )
    perturbation = torch.tensor([250.0, 750.0])
    trainer._sample_perturbation_times = (
        lambda state, batch: adapter.build_training_component_times(perturbation, batch=batch)
    )
    observations = {}
    replay = trainer._replay_generator_clean_prediction
    noise = adapter.add_forward_process_noise

    def observed_replay(*args, **kwargs):
        result = replay(*args, **kwargs)
        tensor = result.components["latent"]
        observations["clean"] = tensor.detach().clone()
        if tensor.requires_grad:
            tensor.register_hook(
                lambda gradient: observations.__setitem__("upstream", gradient.clone())
            )
        return result

    def observed_noise(*args, **kwargs):
        result = noise(*args, **kwargs)
        observations["noised"] = result.state.components["latent"].detach().clone()
        observations["noise"] = result.noise.components["latent"].detach().clone()
        return result

    trainer._replay_generator_clean_prediction = observed_replay
    adapter.add_forward_process_noise = observed_noise
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (cp_size, 0))
    # The backend owns GAS; the production full-output boundary owns CP exactly once.
    trainer.accelerator.backward = lambda loss: (loss / 2).backward()
    torch.manual_seed(19)
    loss = getattr(trainer, f"_{role}_replay_loss")(batch)
    clean = (
        starts.double()
        - (times.double() / 1000).reshape(2, 1, 1) * bundle.generator.detach().double()
    )
    torch.testing.assert_close(observations["clean"].double(), clean, rtol=2e-7, atol=1e-7)
    assert observations["clean"].shape == observations["noised"].shape == (2, 4, 16)
    if role == "fake":
        target = observations["noise"].double() - clean
        residual = bundle.fake.detach().double() - target
        expected_loss = residual.square().mean()
        derivative = 2 * residual.mean()
        expected_upstream = None
    else:
        sigma = (perturbation.double() / 1000).reshape(2, 1, 1)
        real = observations["noised"].double()
        fake = real - sigma * bundle.fake.detach().double()
        normalizer = (clean - real).abs().flatten(1).mean(1).clamp_min(1e-6).reshape(2, 1, 1)
        assert normalizer.shape == (2, 1, 1) and normalizer[0] != normalizer[1]
        direction = (fake - real) / normalizer
        expected_loss = 0.5 * direction.square().mean()
        expected_upstream = direction / clean.numel() * cp_size / 2
        derivative = (direction / clean.numel() * -(times.double() / 1000).reshape(2, 1, 1)).sum()
    torch.testing.assert_close(loss.detach().double(), expected_loss, rtol=2e-6, atol=1e-7)
    trainer._backward_full_output_loss(loss)
    parameter = getattr(bundle, role)
    inactive = getattr(bundle, "generator" if role == "fake" else "fake")
    assert inactive.grad is None
    expected_gradient = derivative * cp_size / 2
    torch.testing.assert_close(parameter.grad.double(), expected_gradient, rtol=2e-6, atol=1e-7)
    if expected_upstream is not None:
        assert observations["upstream"].shape == expected_upstream.shape == (2, 4, 16)
        torch.testing.assert_close(
            observations["upstream"].double(), expected_upstream, rtol=2e-6, atol=1e-7
        )
    assert expected_gradient != 0
    for factor in (0.0, 0.5, 2.0, -1.0):
        assert not torch.allclose(
            parameter.grad.double() * factor, expected_gradient, rtol=2e-6, atol=1e-7
        )
