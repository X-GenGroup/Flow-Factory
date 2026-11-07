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

"""Independent OPD endpoint gradients and frozen two-pass ownership under CP."""

import copy

import pytest
import torch
from test_opd_production_paths import _samples, _snapshot_adapter, _two_teacher_trainer

from flow_factory.trainers.common import context_parallel


@pytest.mark.parametrize("target,sde", [("xt", False), ("v", False), ("x0", False), ("xt", True)])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("degree", [1, 2])
def test_opd_cp_partitioned_gradient_matches_analytic_teacher_oracle(
    monkeypatch, target, sde, normalize, degree
):
    """Sum token-slice gradients, then emulate FSDP's mean across CP peers."""
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (degree, 0))
    student, teachers = 0.75, {"teacher_a": 1.25, "teacher_b": -0.5}
    peer_gradients = []
    expected = 0.0
    for peer in range(degree):
        adapter = _snapshot_adapter(student, teachers)
        velocity = adapter.velocity

        def partitioned(latents, scope):
            value = velocity(latents, scope)
            if torch.is_grad_enabled():
                mask = torch.arange(value.shape[-1]) % degree == peer
                value = torch.where(mask, value, value.detach())
            return value

        adapter.velocity = partitioned
        trainer = _two_teacher_trainer(
            adapter,
            loss_target=target,
            self_normalize=normalize,
            is_sde=sde,
            num_inference_steps=1,
            per_device_batch_size=2,
        )
        if sde:
            adapter.scheduler_group.primary.dynamics_type = "Flow-SDE"
        samples = _samples(["dataset_a", "dataset_b"], [1.25, -0.75], 1)
        # These fixtures use the same recorded source names as the production helper.
        trainer._source_to_teacher = {"dataset_a": 0, "dataset_b": 1}
        trainer._available_sources = set(trainer._source_to_teacher)
        trainer.optimize(samples)
        peer_gradients.append(adapter.weight.grad.detach().double())
        if peer == 0:
            # For the affine fixture, xt=x+v, v=v and x0=x-sigma*v with sigma=1.
            # SDE denominator is std^2*(-dt)=0.125, independently of helpers.
            denominator = 0.125 if sde else 1.0
            coefficient = -1.0 if target == "x0" else 1.0
            for sample, teacher in zip(samples, teachers.values()):
                x = sample.all_latents[0].double()
                error = coefficient * (student - teacher) * x
                scale = float(error.abs().mean()) + 1e-8 if normalize else 1.0
                expected += float((2 * error * coefficient * x).mean()) / (2 * scale * denominator)
        assert adapter.weight.item() == student
        for name, value in teachers.items():
            with adapter.use_named_parameters(name):
                assert adapter.weight.item() == value
    torch.testing.assert_close(
        torch.stack(peer_gradients).mean(),
        torch.tensor(expected, dtype=torch.float64),
        rtol=2e-6,
        atol=1e-7,
    )
    if expected:
        assert not torch.isclose(
            torch.stack(peer_gradients).mean(), torch.tensor(expected / 2, dtype=torch.float64)
        )


def test_two_pass_teacher_cache_is_owned_and_replay_is_immutable():
    adapter = _snapshot_adapter(0.75, {"teacher_a": 1.25, "teacher_b": -0.5})
    trainer = _two_teacher_trainer(adapter, num_inference_steps=2)
    trainer._source_to_teacher = {"dataset_a": 0, "dataset_b": 1}
    trainer._available_sources = set(trainer._source_to_teacher)
    samples = _samples(["dataset_a", "dataset_b"], [1.0, 1.0], 2)
    before = copy.deepcopy(samples)
    steps = trainer._select_train_step_indices(2, (0.0, 1.0))
    trainer._precompute_teacher_targets(samples, steps)
    caches = [sample.extra_kwargs["teacher_target"]["latent"] for sample in samples]
    assert not torch.equal(*caches)
    assert all(not value.requires_grad and value.grad_fn is None for value in caches)
    expected = [value.clone() for value in caches]
    trainer._distill(samples, steps)
    for index, sample in enumerate(samples):
        assert torch.equal(caches[index], expected[index])
        for key in ("all_latents", "timesteps", "prompt_embeds", "latent_index_map"):
            assert torch.equal(getattr(sample, key), getattr(before[index], key))
    caches[0].add_(10)
    assert torch.equal(caches[1], expected[1])
    assert torch.equal(samples[0].all_latents, before[0].all_latents)
