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


"""Production Guard/DPPO objectives against independent dense Flow-SDE derivatives."""

import json
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from coupled_objective_reference import assert_objective, objective_reference
from coupled_observations import assert_objectives, observe_coupled_objectives

from flow_factory.samples import LatentState, MultiModalStepOutput
from flow_factory.trainers.common import context_parallel
from flow_factory.trainers.rl.dppo import DPPOTrainer
from flow_factory.trainers.rl.grpo import GRPOGuardTrainer, GRPOTrainer
from tests.trainers.test_coupled_trainer_parity import _adapter, _replay, _trainer


def _objective_fixture(
    algorithm, mask_space, reference_space, beta, cp_size, gas, latent_shape=(3,)
):
    """Controlled outputs exercise the actual optimize/reference/backward methods."""
    dtype = torch.float64
    ratios = torch.tensor([1.6, 0.5, 1.6, 0.5, 1.0, 1.6, 0.5, 1.4], dtype=dtype)
    advantage = torch.tensor([2, -2, -2, 2, 1, 0, 2, 2], dtype=dtype)
    displacement = torch.tensor([1, 1, 1, 1, 1, 1, 0.25, 0.5], dtype=dtype).reshape(
        (8,) + (1,) * len(latent_shape)
    )
    parameter = torch.nn.Parameter(torch.zeros((8, *latent_shape), dtype=dtype))
    std, dt, sigma = [
        torch.full((8,) + (1,) * len(latent_shape), value, dtype=dtype)
        for value in (1.0, -0.25, 0.6)
    ]
    scale = std * (-dt).sqrt()
    coefficient = (1 + std.square() * (1 - sigma) / (2 * sigma)) * dt
    following = torch.linspace(-0.6, 0.9, parameter.numel(), dtype=dtype).reshape_as(parameter)
    old_mean = (
        (-displacement if algorithm == "dppo" else displacement * 0 - 0.1)
        .expand_as(parameter)
        .clone()
    )
    old_velocity = -displacement.expand_as(parameter).clone()
    initial_log_prob = (-following.square() / (2 * scale.square()) - scale.log()).flatten(1).mean(1)
    old_log_prob = initial_log_prob - ratios.log()
    replay = _replay({"latent": torch.zeros_like(parameter)}, log_prob=old_log_prob)
    replay.next_state.components["latent"] = following
    replay.times.timestep["latent"] = torch.full((8,), 600.0, dtype=dtype)
    adapter = _adapter()
    trainer = _trainer(
        {"grpo": GRPOTrainer, "grpo-guard": GRPOGuardTrainer, "dppo": DPPOTrainer}[algorithm],
        adapter,
        trainer_type=algorithm,
        per_device_batch_size=8,
        num_inner_epochs=1,
        adv_clip_range=(-3.0, 3.0),
        clip_range=(-0.2, 0.2),
        kl_mask_type=mask_space,
        kl_mask_threshold=0.5 if mask_space == "x-based" else 0.25,
        kl_type=reference_space,
        kl_beta=beta,
        guidance_scale=1.0,
        kl_guidance_scale=3.0,
        gradient_accumulation_steps=gas,
    )
    trainer.epoch = trainer.step = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.logger = None
    trainer.config = SimpleNamespace(parallelism_args=SimpleNamespace(cp_size=cp_size))
    trainer.autocast = trainer.accumulate_gradients = nullcontext
    trainer.accelerator = SimpleNamespace(
        sync_gradients=False, backward=lambda loss: (loss / gas).backward()
    )
    trainer._order_samples_for_optimize = lambda samples, inner_epoch: samples
    batch = {
        "advantage": advantage,
        "negative_prompt_embeds": torch.ones(8, 2, dtype=dtype),
        "timesteps": torch.full((8, 1), 600.0, dtype=dtype),
    }
    trainer._iter_prefetched_batches = lambda samples, size: iter([batch])
    adapter.train = lambda: None
    adapter.get_train_step_indices = lambda: [0]
    adapter.get_replay_step = lambda batch, index: replay
    adapter.get_replay_callback = lambda batch, index, field: LatentState(
        {"latent": old_velocity if field == "velocity" else old_mean}
    )
    reference_active = False
    rows = []
    reference_calls = []

    @contextmanager
    def reference_scope():
        nonlocal reference_active
        reference_active = True
        try:
            yield
        finally:
            reference_active = False

    adapter.use_ref_parameters = reference_scope

    def forward_state(**kwargs):
        if reference_active:
            assert not torch.is_grad_enabled() and kwargs["batch"] is batch
            guidance = kwargs["guidance_scale"]
            reference_calls.append(guidance)
            velocity = torch.full_like(parameter, 0.3 * guidance)
            mean = coefficient * velocity
            rows[-1]["reference"] = (velocity if reference_space == "v-based" else mean).detach()
        else:
            velocity = parameter * 1.0
            mean = coefficient * velocity
        log_prob = (
            (-(following - mean).square() / (2 * scale.square()) - scale.log()).flatten(1).mean(1)
        )
        if not reference_active:
            row = {
                **trainer.training_args,
                "advantage": advantage,
                "mean": mean.detach(),
                "old_mean": old_mean,
                "velocity": velocity.detach(),
                "old_velocity": old_velocity,
                "following": following,
                "std": std,
                "dt": dt,
                "sigma": sigma,
                "log_prob": log_prob.detach(),
                "old_log_prob": old_log_prob,
                "algorithm": algorithm,
                "cp_size": cp_size,
                "gas": gas,
            }
            rows.append(row)
            for key, value in (("mean", mean), ("velocity", velocity)):
                value.register_hook(
                    lambda gradient, key=key, row=row: row.__setitem__(
                        key + "_gradient", gradient.detach().clone()
                    )
                )
        return MultiModalStepOutput(
            velocity=LatentState({"latent": velocity}),
            next_state_mean=LatentState({"latent": mean}),
            log_prob=log_prob,
            component_log_probs={"latent": log_prob},
            std_dev_t={"latent": std},
            dt={"latent": dt},
        )

    adapter.forward_state = forward_state
    backward = trainer._backward_full_output_loss

    def observed_backward(loss):
        rows[-1]["loss"] = loss.detach().clone()
        backward(loss)

    trainer._backward_full_output_loss = observed_backward
    return trainer, parameter, rows, reference_calls


@pytest.mark.parametrize("trainer_type", ["grpo", "grpo-guard", "dppo"])
@pytest.mark.parametrize(
    "mask_space,reference_space",
    [
        ("x-based", "x-based"),
        ("x-based", "v-based"),
        ("v-based", "x-based"),
        ("v-based", "v-based"),
    ],
)
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("beta", [0.0, 0.03])
def test_production_objective_matches_independent_upstreams(
    monkeypatch, tmp_path, trainer_type, mask_space, reference_space, cp_size, beta
):
    trainer, parameter, rows, reference_calls = _objective_fixture(
        trainer_type, mask_space, reference_space, beta, cp_size, gas=2
    )
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (cp_size, 0))
    for _ in range(2):
        trainer.optimize([object()] * 8)
    references = [assert_objective(row) for row in rows]
    torch.testing.assert_close(
        parameter.grad, sum(row["velocity_gradient"] for row in references), rtol=2e-12, atol=2e-12
    )
    assert reference_calls == ([3.0 if trainer_type == "dppo" else 1.0] * 2 if beta else [])
    expected_keep = torch.tensor([False, False, True, True, True, True, True, False])
    if trainer_type == "dppo":
        assert torch.equal(references[0]["keep"], expected_keep)
        if not beta:
            assert torch.count_nonzero(parameter.grad[~expected_keep]) == 0
        else:
            assert torch.count_nonzero(parameter.grad[~expected_keep]) > 0
    elif trainer_type == "grpo-guard":
        assert not torch.allclose(references[0]["ratio"], references[0]["plain_ratio"])
        assert references[0]["keep"].any() and not references[0]["keep"].all()
    (tmp_path / "rank-0.json").write_text(
        json.dumps(
            {
                "rank": 0,
                "algorithm": trainer_type,
                "runtime": {"world_size": 1, "rank": 0, "device": "cpu", "distributed_type": "NO"},
                "elementwise_comparison": True,
                "nonzero_gradients": bool(parameter.grad.count_nonzero()),
                "cp_geometry": "simulated",
                "cp_size": cp_size,
                "gas": 2,
                "maximum_gradient_error": float(
                    (parameter.grad - sum(row["velocity_gradient"] for row in references))
                    .abs()
                    .max()
                ),
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("trainer_type", ["grpo", "grpo-guard", "dppo"])
def test_reference_rejects_zero_or_wrongly_scaled_production_derivatives(monkeypatch, trainer_type):
    trainer, _, rows, _ = _objective_fixture(trainer_type, "x-based", "v-based", 0.03, 2, 2)
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    trainer.optimize([object()] * 8)
    row = rows[0]
    assert_objective(row)
    for factor in (0.0, 0.5, 2.0, -1.0):
        corrupted = {**row, "velocity_gradient": row["velocity_gradient"] * factor}
        with pytest.raises(AssertionError):
            assert_objective(corrupted)
    changed = {**row, "cp_size": 1}
    with pytest.raises(AssertionError):
        assert_objective(changed)


def test_guard_on_policy_value_retains_variance_scaled_derivative(monkeypatch):
    trainer, _, rows, _ = _objective_fixture("grpo-guard", "x-based", "v-based", 0.0, 1, 1)
    # Replace the stored rollout callbacks and log probabilities with the current
    # policy's values, while retaining a nonunit transition standard deviation.
    replay = trainer.adapter.get_replay_step(None, 0)
    following = replay.next_state.components["latent"]
    replay.log_prob = (
        (-following.square() / 0.5 - torch.tensor(0.5, dtype=torch.float64).log())
        .flatten(1)
        .mean(1)
    )
    replay.component_log_probs = {"latent": replay.log_prob}
    trainer.adapter.get_replay_callback = lambda *args: LatentState(
        {"latent": torch.zeros_like(following)}
    )
    captured = trainer._backward_full_output_loss

    def backward(loss):
        rows[-1].update(old_mean=torch.zeros_like(following), old_log_prob=replay.log_prob)
        captured(loss)

    trainer._backward_full_output_loss = backward
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (1, 0))
    trainer.optimize([object()] * 8)
    ref = assert_objective(rows[0])
    torch.testing.assert_close(ref["ratio"], torch.ones(8, dtype=torch.float64), atol=1e-8, rtol=0)
    plain = objective_reference({**rows[0], "algorithm": "dppo", "kl_mask_threshold": 1e6})
    torch.testing.assert_close(ref["velocity_gradient"], plain["velocity_gradient"] * 0.5)


@pytest.mark.parametrize("trainer_type", ["grpo", "grpo-guard", "dppo"])
def test_native_objective_observer_checks_real_backward_without_extra_forwards(
    monkeypatch, trainer_type
):
    trainer, _, rows, reference_calls = _objective_fixture(
        trainer_type, "x-based", "v-based", 0.03, 2, 2
    )
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    originals = (
        trainer._replay_forward,
        trainer._reference_forward,
        trainer._backward_full_output_loss,
    )
    with observe_coupled_objectives(trainer) as observed:
        trainer.optimize([object()] * 8)
    assert len(observed) == len(rows) == len(reference_calls) == 1
    assert_objectives(observed, 2)
    assert observed[0]["oracle"]["velocity_gradient"].count_nonzero()
    assert originals == (
        trainer._replay_forward,
        trainer._reference_forward,
        trainer._backward_full_output_loss,
    )


@pytest.mark.parametrize("trainer_type", ["grpo", "grpo-guard", "dppo"])
def test_packed_latent_oracle_preserves_batch_and_upstream_shapes(monkeypatch, trainer_type):
    trainer, parameter, rows, _ = _objective_fixture(
        trainer_type, "x-based", "v-based", 0.03, 2, 2, latent_shape=(4, 16)
    )
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    with observe_coupled_objectives(trainer) as observed:
        trainer.optimize([object()] * 8)
    assert_objectives(observed, 2)
    assert rows[0]["mean"].shape == parameter.shape == (8, 4, 16)
    for key in ("mean_gradient", "velocity_gradient"):
        assert observed[0][key].shape == observed[0]["oracle"][key].shape == parameter.shape
    torch.testing.assert_close(
        parameter.grad, observed[0]["oracle"]["velocity_gradient"], rtol=2e-12, atol=2e-12
    )


@pytest.mark.parametrize("mask_space", ["x-based", "v-based"])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_dppo_inclusive_mask_boundary(monkeypatch, mask_space, direction):
    trainer, parameter, rows, _ = _objective_fixture("dppo", mask_space, "v-based", 0.0, 1, 1)
    boundary = 0.5 if mask_space == "x-based" else 0.25
    trainer.training_args["kl_mask_threshold"] = boundary + direction * 1e-12
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (1, 0))
    trainer.optimize([object()] * 8)
    reference = assert_objective(rows[0])
    assert reference["divergence"][7] == boundary
    assert bool(reference["keep"][7]) == (direction > 0)
    assert bool(parameter.grad[7].count_nonzero()) == (direction > 0)


@pytest.mark.parametrize("trainer_type", ["grpo", "grpo-guard"])
@pytest.mark.parametrize("offset", [-1e-6, 0.0, 1e-6])
def test_exact_on_policy_clip_boundary_and_neighbours(monkeypatch, trainer_type, offset):
    trainer, parameter, rows, _ = _objective_fixture(trainer_type, "x-based", "v-based", 0.0, 1, 1)
    trainer.training_args["clip_range"] = (offset, offset)
    replay = trainer.adapter.get_replay_step(None, 0)
    following = replay.next_state.components["latent"]
    replay.log_prob = (
        (-following.square() / 0.5 - torch.tensor(0.5, dtype=torch.float64).log())
        .flatten(1)
        .mean(1)
    )
    replay.component_log_probs = {"latent": replay.log_prob}
    trainer.adapter.get_replay_callback = lambda *args: LatentState(
        {"latent": torch.zeros_like(following)}
    )
    backward = trainer._backward_full_output_loss

    def observed_backward(loss):
        rows[-1].update(old_mean=torch.zeros_like(following), old_log_prob=replay.log_prob)
        backward(loss)

    trainer._backward_full_output_loss = observed_backward
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (1, 0))
    trainer.optimize([object()] * 8)
    reference = assert_objective(rows[0])
    torch.testing.assert_close(
        reference["ratio"], torch.ones(8, dtype=torch.float64), rtol=0, atol=0
    )
    # Positive and negative advantages select opposite directed clipping sides.
    assert bool(reference["keep"][0]) == (offset >= 0)
    assert bool(reference["keep"][1]) == (offset <= 0)
    assert bool(parameter.grad[0].count_nonzero()) == (offset >= 0)
    assert bool(parameter.grad[1].count_nonzero()) == (offset <= 0)


@pytest.mark.parametrize("trainer_type", ["grpo-guard", "dppo"])
@pytest.mark.parametrize("beta", [0.0, 0.03])
def test_all_removed_policy_terms_preserve_reference_gradient(monkeypatch, trainer_type, beta):
    trainer, parameter, rows, _ = _objective_fixture(trainer_type, "x-based", "v-based", beta, 1, 1)
    trainer.training_args["clip_range"] = (0.0, 0.0)
    trainer.training_args["kl_mask_threshold"] = 0.0
    batch = next(trainer._iter_prefetched_batches([], 8))
    batch["advantage"].copy_(torch.tensor([1, -1, 1, -1, 0, 1, -1, 1]))
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (1, 0))
    trainer.optimize([object()] * 8)
    reference = assert_objective(rows[0])
    assert not reference["keep"][rows[0]["advantage"] != 0].any()
    assert bool(parameter.grad.count_nonzero()) == bool(beta)
    assert bool(reference["velocity_gradient"].count_nonzero()) == bool(beta)
