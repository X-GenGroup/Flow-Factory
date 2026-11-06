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


"""Actual NFT optimization against independent full-output loss and derivatives."""

import copy
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from nft_objective_reference import (
    _quantized_affine_interval,
    assert_objective,
    objective_reference,
)
from nft_observations import observe_objectives

from flow_factory.samples import BaseSample, LatentState, MultiModalStepOutput
from flow_factory.trainers.common import context_parallel
from tests.trainers.test_decoupled_trainer_parity import (
    _adapter,
    _nft_trainer,
    _noised,
    _structured_adapter,
)


def _fixture(
    cp_size,
    gas,
    off_policy,
    kl_beta,
    latent_shape,
    shared_noise=False,
    velocity_dtype=torch.float64,
    state_dtype=torch.float64,
    kl_compute_dtype=None,
):
    batch_size = 5
    shape = (batch_size, *latent_shape)
    count = torch.empty(shape).numel()
    parameter = torch.nn.Parameter(
        torch.linspace(-0.4, 0.6, count, dtype=torch.float64).reshape(shape)
    )
    clean = torch.linspace(-1.2, 1.6, count, dtype=state_dtype).reshape(shape)
    advantages = [-8.0, -3.0, 0.0, 3.0, 8.0]
    samples = [
        BaseSample(
            timesteps=torch.tensor([1000.0, 500.0]),
            all_latents=value[None],
            latent_index_map=torch.tensor([-1, -1, 0]),
            prompt_embeds=torch.tensor([i + 1.0]),
            sampling_sample_id=i,
            sampling_group_id=0,
            sampling_group_member_id=i,
            extra_kwargs={"advantage": torch.tensor(adv, dtype=state_dtype)},
        )
        for i, (value, adv) in enumerate(zip(clean, advantages))
    ]
    batch = BaseSample.stack(samples)
    adapter = _adapter()
    trainer = _nft_trainer(adapter)
    trainer.off_policy = off_policy
    trainer.training_args.update(
        trainer_type="nft",
        per_device_batch_size=batch_size,
        num_inner_epochs=1,
        kl_beta=kl_beta,
        gradient_accumulation_steps=gas,
        use_shared_noise=shared_noise,
    )
    trainer.epoch = trainer.step = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.logger = None
    trainer.config = SimpleNamespace(parallelism_args=SimpleNamespace(cp_size=cp_size))
    trainer.autocast = trainer.accumulate_gradients = nullcontext
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        sync_gradients=False,
        gradient_accumulation_steps=gas,
        backward=lambda loss: (loss / gas).backward(),
    )
    trainer._order_samples_for_optimize = lambda samples, inner_epoch: samples
    trainer._iter_prefetched_batches = lambda samples, size: iter([batch])
    scopes, forwards = [], []
    active = "live"

    @contextmanager
    def scope(name):
        nonlocal active
        before = active
        active = name
        scopes.append(name)
        try:
            yield
        finally:
            active = before

    adapter.use_ema_parameters = lambda: scope("ema")
    adapter.use_ref_parameters = lambda: scope("reference")

    def forward_state(**kwargs):
        assert kwargs["compute_log_prob"] is False
        state = kwargs["state"]
        theta = (
            parameter
            if active == "live"
            else parameter.detach() + (0.3 if active == "ema" else -0.2)
        )
        velocity = (theta + 0.13 * state.components["latent"]).to(velocity_dtype)
        forwards.append(
            {
                "scope": active,
                "grad": torch.is_grad_enabled(),
                "state": state,
                "velocity": velocity.detach().clone(),
            }
        )
        return MultiModalStepOutput(velocity=LatentState({"latent": velocity}))

    adapter.forward_state = forward_state
    if kl_compute_dtype is not None:
        # CPU diagnostic for CUDA autocast's FP32 pow dispatch, with subtraction
        # retained in the public velocity dtype. Native observations check the
        # real CUDA path; this fixture makes that precision boundary reproducible.
        def autocast_kl(new, reference, noised):
            values = {
                name: (value - reference.components[name]).to(kl_compute_dtype).square()
                for name, value in new.components.items()
            }
            return adapter.reduce_latent_values(values, state=noised.state)

        trainer._velocity_kl = autocast_kl
    return trainer, parameter, samples, scopes, forwards


@pytest.mark.parametrize("cp_size,gas", [(1, 1), (2, 4)])
@pytest.mark.parametrize("off_policy", [False, True])
@pytest.mark.parametrize("kl_beta", [0.0, 0.03])
@pytest.mark.parametrize("latent_shape", [(3, 4), (2, 2, 2, 2)])
def test_real_optimize_matches_detached_scale_oracle(
    monkeypatch, cp_size, gas, off_policy, kl_beta, latent_shape
):
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (cp_size, 0))
    trainer, parameter, samples, scopes, forwards = _fixture(
        cp_size, gas, off_policy, kl_beta, latent_shape
    )
    with observe_objectives(trainer) as rows:
        trainer.optimize(samples)
    assert len(rows) == 2
    torch.testing.assert_close(
        parameter.grad,
        sum(row["oracle"]["gradients"]["latent"] for row in rows),
        rtol=2e-12,
        atol=2e-12,
    )
    assert scopes == (["ema"] if off_policy else []) + (["reference"] * 2 if kl_beta else [])
    assert [row["scope"] for row in forwards[:2]] == ["ema" if off_policy else "live"] * 2
    assert all(not row["grad"] for row in forwards[:2])
    assert all(row["same_noised_inputs"] for row in rows)
    assert torch.equal(
        rows[0]["oracle"]["reward"], torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0], dtype=torch.float64)
    )
    assert not torch.equal(rows[0]["state"]["latent"], rows[1]["state"]["latent"])
    for row in rows:
        assert_objective(row)


def test_oracle_rejects_corrupted_gradient_and_missing_cp_factor(monkeypatch):
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    trainer, _, samples, _, _ = _fixture(2, 4, True, 0.03, (3, 4))
    with observe_objectives(trainer) as rows:
        trainer.optimize(samples)
    for factor in (0.0, 0.5, 2.0, -1.0):
        bad = copy.deepcopy(rows[0])
        bad["gradients"]["latent"] *= factor
        with pytest.raises(AssertionError):
            assert_objective(bad)


def test_each_component_owns_its_detached_normalization_and_active_mask():
    adapter = _structured_adapter()
    trainer = _nft_trainer(adapter, nft_beta=0.4)
    clean = {
        "video": torch.tensor([[1.0, 2.0, 9.0]], dtype=torch.float64),
        "audio": torch.tensor([[3.0, 4.0]], dtype=torch.float64),
    }
    noise = {name: value + 0.3 for name, value in clean.items()}
    noised, times = _noised(
        clean,
        noise,
        {
            "video": torch.tensor([0.7], dtype=torch.float64),
            "audio": torch.tensor([0.3], dtype=torch.float64),
        },
    )
    masks = {"video": torch.tensor([[True, True, False]]), "audio": torch.tensor([[True, True]])}
    noised.state.active_masks = masks
    new = {
        "video": torch.tensor([[0.1, 0.2, 100.0]], dtype=torch.float64, requires_grad=True),
        "audio": torch.tensor([[0.6, -0.4]], dtype=torch.float64, requires_grad=True),
    }
    old = {name: torch.zeros_like(value) for name, value in new.items()}
    pos, neg = trainer._matching_losses(
        LatentState(clean), noised, times, LatentState(new), LatentState(old)
    )
    loss = (0.8 * pos + 0.2 * neg).mean() * 5 / 0.4
    loss.backward()
    row = {
        "clean": clean,
        "state": noised.state.components,
        "sigma": times.sigma,
        "new": new,
        "old": old,
        "masks": masks,
        "velocity_direction": "noise",
        "advantage": torch.tensor([3.0], dtype=torch.float64),
        "adv_clip_range": (-5.0, 5.0),
        "nft_beta": 0.4,
        "kl_beta": 0.0,
        "cp_size": 1,
        "gas": 1,
        "loss": loss.detach(),
        "gradients": {name: value.grad for name, value in new.items()},
    }
    expected = assert_objective(row)
    assert new["video"].grad[0, 2] == 0
    assert not torch.equal(*[expected["scales"][name][0] for name in clean])
    # Differentiating through the scale would halve the homogeneous error term.
    bad = copy.deepcopy(row)
    bad["gradients"] = {name: value * 0.5 for name, value in row["gradients"].items()}
    with pytest.raises(AssertionError):
        assert_objective(bad)


@pytest.mark.parametrize(
    "module,tp,mode,off_policy",
    [("wan_nft_fixture", 2, "full", True), ("flux_nft_fixture", 1, "lora", False)],
)
def test_native_recipe_keeps_nft_and_overlap_contracts(
    monkeypatch, tmp_path, module, tp, mode, off_policy
):
    import importlib

    from context_parallel_overlap_runner import make_overlap_config

    from flow_factory.models.registry import get_model_adapter_class
    from flow_factory.trainers.common.context_parallel import validate_context_parallel_config
    from flow_factory.trainers.rl.nft import DiffusionNFTTrainer

    monkeypatch.setenv("WORLD_SIZE", str(4 * tp))
    fixture = importlib.import_module(module)
    options = {
        "dp_size": 2,
        "cp_size": 2,
        "tp_size": tp,
        "training_mode": mode,
        "algorithm": "nft",
        "off_policy": off_policy,
        "scheduling_mode": "ready" if tp == 2 else "ordered",
    }
    config = make_overlap_config(
        fixture.make_config(tmp_path, tmp_path / "results", options),
        tmp_path,
        tmp_path / "results",
        options,
    )
    DiffusionNFTTrainer.validate_reward_optimization_overlap(config)
    validate_context_parallel_config(
        config, get_model_adapter_class(config.model_args.model_type), DiffusionNFTTrainer
    )
    assert (
        config.training_args.trainer_type == "nft" and config.training_args.off_policy is off_policy
    )
    assert (
        config.training_args.num_train_timesteps == 2
        and config.training_args.gradient_accumulation_steps == 4
    )
    assert config.scheduler_args.dynamics_type == "ODE"
    assert fixture.audit_kind == "terminal_and_shared_noised_inputs" and fixture.audit_steps == 1


def test_precomputed_group_noise_is_reused_but_advances_between_timesteps(monkeypatch):
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    trainer, _, samples, _, _ = _fixture(2, 4, True, 0.03, (3, 4), shared_noise=True)
    with observe_objectives(trainer) as rows:
        trainer.optimize(samples)
    for row in rows:
        noise = row["noise"]["latent"]
        for sample_noise in noise[1:]:
            torch.testing.assert_close(sample_noise, noise[0], rtol=0, atol=0)
    assert not torch.equal(rows[0]["noise"]["latent"], rows[1]["noise"]["latent"])


def test_objective_observer_restores_methods_after_failure(monkeypatch):
    from flow_factory.trainers.rl import nft

    trainer, _, _, _, _ = _fixture(1, 1, False, 0.0, (3, 4))
    names = (
        "_precompute_sampling_policy_steps",
        "_matching_losses",
        "_velocity_kl",
        "_backward_full_output_loss",
    )
    originals = tuple(getattr(trainer, name) for name in names)
    forward = nft.forward_velocity_state
    with pytest.raises(RuntimeError, match="controlled failure"):
        with observe_objectives(trainer):
            raise RuntimeError("controlled failure")
    assert originals == tuple(getattr(trainer, name) for name in names)
    assert nft.forward_velocity_state is forward


@pytest.mark.parametrize("off_policy,kl_beta", [(False, 0.0), (True, 0.03)])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float64])
def test_bf16_velocity_affine_rounding_is_part_of_the_nft_objective(
    monkeypatch, off_policy, kl_beta, state_dtype
):
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    trainer, parameter, samples, _, _ = _fixture(
        2, 4, off_policy, kl_beta, (3, 4), velocity_dtype=torch.bfloat16, state_dtype=state_dtype
    )
    with observe_objectives(trainer) as rows:
        trainer.optimize(samples)
    assert len(rows) == 2 and all(row["new"]["latent"].dtype == torch.bfloat16 for row in rows)
    torch.testing.assert_close(
        parameter.grad, sum(row["gradients"]["latent"].double() for row in rows), rtol=0, atol=0
    )
    for row in rows:
        assert_objective(row)


@pytest.mark.parametrize("coefficient", [0.5, -0.5])
def test_fp32_error_interval_only_crosses_adjacent_bf16_quantization_boundary(coefficient):
    lower = torch.tensor(1.0, dtype=torch.bfloat16)
    upper = torch.nextafter(lower, torch.tensor(float("inf"), dtype=torch.bfloat16))
    midpoint = (lower.double() + upper.double()) / 2
    bound = torch.tensor(torch.finfo(torch.float32).eps, dtype=torch.float64)
    low, high = _quantized_affine_interval(
        midpoint, bound, coefficient, torch.bfloat16, torch.bfloat16
    )
    endpoints = torch.tensor([float(lower), float(upper)], dtype=torch.float64) * coefficient
    assert low == endpoints.min() and high == endpoints.max()
    # Away from a midpoint, the same FP32 error bound admits only one BF16 value.
    fixed_low, fixed_high = _quantized_affine_interval(
        lower.double(), bound, coefficient, torch.bfloat16, torch.bfloat16
    )
    assert fixed_low == fixed_high == float(lower) * coefficient


@pytest.mark.parametrize("off_policy", [False, True])
def test_bf16_kl_subtraction_preserves_fp32_autocast_square_boundary(monkeypatch, off_policy):
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    trainer, _, samples, _, _ = _fixture(
        2,
        4,
        off_policy,
        0.01,
        (3, 4),
        velocity_dtype=torch.bfloat16,
        state_dtype=torch.float32,
        kl_compute_dtype=torch.float32,
    )
    with observe_objectives(trainer) as rows:
        trainer.optimize(samples)
    assert all(row["kl_dtype"] == torch.float32 for row in rows)
    for row in rows:
        expected = assert_objective(row)
        wrong = copy.deepcopy(row)
        wrong["kl_dtype"] = torch.bfloat16
        assert not torch.equal(objective_reference(wrong)["loss"], expected["loss"])
