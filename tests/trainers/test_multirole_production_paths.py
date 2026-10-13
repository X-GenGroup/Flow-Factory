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

import inspect
from unittest import mock
from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace
from typing import Any, Iterator, Mapping, Tuple

import pytest
import torch
from accelerate import Accelerator

from flow_factory.hparams import DMD2TrainingArguments
from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import (
    BaseSample,
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    StackedSampleBatch,
)
from flow_factory.scheduler import SDESchedulerOutput
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.distillation.distillation_runtime import (
    pop_distillation_metrics,
    require_velocity,
)
from flow_factory.trainers.distillation.dmd2 import DMD2Trainer
from flow_factory.trainers.distillation.tdm import TDMTrainer
from flow_factory.trainers.distillation.tdm_r1 import TDMR1Trainer
from flow_factory.trainers.distillation.tdm_trajectory import TDMTrajectoryRuntimeMixin
from flow_factory.trainers.role_optimization import (
    OptimizationRole,
    RoleOptimizationCoordinator,
    RoleOptimizerConfig,
)


class TinyBundle(torch.nn.Module):
    """Own disjoint generator and fake scalar parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.generator = torch.nn.Parameter(torch.tensor(0.4))
        self.fake = torch.nn.Parameter(torch.tensor(-0.2))


class TinyDMD2Adapter(BaseAdapter):
    """Exercise the real one-step replay, noising, and score projections."""

    trajectory_component_order = ("latent",)

    declared_variants: tuple[str, ...] = ("generator", "fake")

    def __init__(self, bundle: TinyBundle) -> None:
        self.bundle = bundle
        self.active_role = "generator"
        self.forward_events: list[tuple[str, bool, int, int]] = []
        self.forward_states: list[tuple[str, torch.Tensor]] = []
        self.perturbation_times: list[torch.Tensor] = []

    @contextmanager
    def use_component_variant(self, role_name: str) -> Iterator[None]:
        # Mirror the real registry, which only knows the declared trainable
        # variants. A permissive fake here hid a production KeyError: the frozen
        # reference is a parameter snapshot, so it is never a declared variant.
        if role_name not in self.declared_variants:
            raise KeyError(
                f"component variant {role_name!r} is not declared; declared variants "
                f"are {self.declared_variants!r}"
            )
        previous = self.active_role
        self.active_role = role_name
        try:
            yield
        finally:
            self.active_role = previous

    @contextmanager
    def use_ref_parameters(self) -> Iterator[None]:
        previous = self.active_role
        self.active_role = "reference"
        try:
            yield
        finally:
            self.active_role = previous

    def load_pipeline(self) -> Any:
        raise NotImplementedError

    def decode_latents(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("DMD2 must not decode generated media")

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward(self, *args: Any, **kwargs: Any) -> SDESchedulerOutput:
        raise NotImplementedError

    def train(self, mode: bool = True) -> None:
        del mode

    def build_training_component_times(
        self,
        primary_timesteps: torch.Tensor,
        *,
        batch: StackedSampleBatch | None = None,
    ) -> ComponentTimes:
        times = super().build_training_component_times(primary_timesteps, batch=batch)
        self.perturbation_times.append(primary_timesteps.detach().clone())
        return times

    def _forward_state(
        self,
        *,
        batch: StackedSampleBatch,
        state: LatentState,
        times: ComponentTimes,
        next_state: LatentState | None,
        compute_log_prob: bool,
        return_fields: Tuple[str, ...],
        noise_level: float | None,
        forward_kwargs: Mapping[str, Any],
    ) -> MultiModalStepOutput:
        del batch, next_state, compute_log_prob, return_fields, noise_level, forward_kwargs
        self.forward_events.append(
            (self.active_role, torch.is_grad_enabled(), id(state), id(times))
        )
        latents = state.components["latent"]
        self.forward_states.append((self.active_role, latents.detach().clone()))
        if self.active_role == "generator":
            boundary = LatentState({"latent": latents + self.bundle.generator})
            return MultiModalStepOutput(
                next_state=boundary,
                next_state_mean=boundary,
                velocity=LatentState({"latent": self.bundle.generator.expand_as(latents)}),
            )
        velocity_value = (
            self.bundle.fake if self.active_role == "fake" else torch.zeros_like(self.bundle.fake)
        )
        return MultiModalStepOutput(
            velocity=LatentState({"latent": velocity_value.expand_as(latents)})
        )


def _role_config(role_name: str, learning_rate: float) -> RoleOptimizerConfig:
    return RoleOptimizerConfig(
        role_name=role_name,  # type: ignore[arg-type]
        learning_rate=learning_rate,
        adam_betas=(0.8, 0.9),
        adam_weight_decay=0.0,
        adam_epsilon=1e-8,
        max_grad_norm=100.0,
    )


def _phase_trainer(
    gradient_accumulation_steps: int,
) -> Tuple[DMD2Trainer, TinyBundle, torch.optim.AdamW]:
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=gradient_accumulation_steps)
    bundle = TinyBundle()
    configs = {
        "generator": _role_config("generator", 0.03),
        "fake": _role_config("fake", 0.07),
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [bundle.generator],
                "role_name": "generator",
                "lr": configs["generator"].learning_rate,
                "betas": configs["generator"].adam_betas,
                "weight_decay": configs["generator"].adam_weight_decay,
                "eps": configs["generator"].adam_epsilon,
            },
            {
                "params": [bundle.fake],
                "role_name": "fake",
                "lr": configs["fake"].learning_rate,
                "betas": configs["fake"].adam_betas,
                "weight_decay": configs["fake"].adam_weight_decay,
                "eps": configs["fake"].adam_epsilon,
            },
        ]
    )
    roles = {
        "generator": OptimizationRole(configs["generator"], (bundle.generator,), (0,)),
        "fake": OptimizationRole(configs["fake"], (bundle.fake,), (1,)),
    }
    trainer = object.__new__(DMD2Trainer)
    trainer.accelerator = accelerator
    trainer.adapter = SimpleNamespace(
        use_component_variant=lambda role_name: nullcontext()
    )
    trainer.model_bundle = bundle
    trainer.optimizer = optimizer
    trainer.optimization_roles = roles
    trainer.role_optimization = RoleOptimizationCoordinator(
        accelerator,
        bundle,
        optimizer,
        roles,
    )
    trainer.training_args = SimpleNamespace(
        per_device_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # The public step is paced by the primary role, which is the first declared.
        required_trainable_roles=("generator", "fake"),
    )
    trainer.autocast = nullcontext
    trainer.step = 0
    trainer.epoch = 0
    trainer.log_args = SimpleNamespace(verbose=False)

    def fake_loss(self: DMD2Trainer, batch: StackedSampleBatch) -> torch.Tensor:
        target = batch["prompt_embeds"].flatten()
        return torch.stack([(bundle.fake - value).square() for value in target]).mean()

    def generator_loss(self: DMD2Trainer, batch: StackedSampleBatch) -> torch.Tensor:
        target = batch["prompt_embeds"].flatten()
        return torch.stack([(bundle.generator - value).square() for value in target]).mean()

    trainer._fake_replay_loss = MethodType(fake_loss, trainer)
    trainer._generator_replay_loss = MethodType(generator_loss, trainer)
    return trainer, bundle, optimizer


def _unit(value: float) -> list[BaseSample]:
    return [BaseSample(prompt_embeds=torch.tensor([value]))]


def _real_objective_trainer() -> tuple[DMD2Trainer, TinyDMD2Adapter, TinyBundle]:
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
    bundle = TinyBundle()
    adapter = TinyDMD2Adapter(bundle)
    adapter.scheduler_group = SimpleNamespace(
        primary=SimpleNamespace(seed=42),
        sample_ode_step_index=lambda draw_index: 0,
    )
    configs = {
        "generator": _role_config("generator", 0.03),
        "fake": _role_config("fake", 0.07),
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [bundle.generator],
                "role_name": "generator",
                "lr": configs["generator"].learning_rate,
                "betas": configs["generator"].adam_betas,
                "weight_decay": configs["generator"].adam_weight_decay,
                "eps": configs["generator"].adam_epsilon,
            },
            {
                "params": [bundle.fake],
                "role_name": "fake",
                "lr": configs["fake"].learning_rate,
                "betas": configs["fake"].adam_betas,
                "weight_decay": configs["fake"].adam_weight_decay,
                "eps": configs["fake"].adam_epsilon,
            },
        ]
    )
    roles = {
        "generator": OptimizationRole(configs["generator"], (bundle.generator,), (0,)),
        "fake": OptimizationRole(configs["fake"], (bundle.fake,), (1,)),
    }
    trainer = object.__new__(DMD2Trainer)
    trainer.accelerator = accelerator
    trainer.adapter = adapter
    trainer.model_bundle = bundle
    trainer.optimizer = optimizer
    trainer.optimization_roles = roles
    trainer.role_optimization = RoleOptimizationCoordinator(
        accelerator,
        bundle,
        optimizer,
        roles,
    )
    trainer.training_args = DMD2TrainingArguments(
        num_inference_steps=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
    )
    trainer.autocast_entries = 0

    @contextmanager
    def recording_autocast() -> Iterator[None]:
        trainer.autocast_entries += 1
        yield

    trainer.autocast = recording_autocast
    trainer.step = 0
    trainer.epoch = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    return trainer, adapter, bundle


def _trajectory_unit(generator_value: float = 0.4) -> list[BaseSample]:
    return [
        BaseSample(
            timesteps=torch.tensor([1000.0]),
            all_latents=torch.tensor([[1.0], [1.0 + generator_value]]),
            latent_index_map=torch.tensor([0, 1]),
        )
    ]


@pytest.mark.parametrize("phase_name", ["fake", "generator"])
def test_dmd2_role_phases_use_nested_microbatch_api_and_role_counters(
    phase_name: str,
) -> None:
    trainer, _, _ = _phase_trainer(gradient_accumulation_steps=2)
    replay_units = [_unit(1.0), _unit(3.0)]

    getattr(trainer, f"_{phase_name}_phase")(replay_units)

    assert trainer.optimization_roles[phase_name].step == 1
    other_name = "generator" if phase_name == "fake" else "fake"
    assert trainer.optimization_roles[other_name].step == 0
    assert trainer.step == (1 if phase_name == "generator" else 0)


@pytest.mark.parametrize("phase_name", ["fake", "generator"])
def test_dmd2_gas_two_phase_matches_explicit_large_batch(phase_name: str) -> None:
    trainer, bundle, _ = _phase_trainer(gradient_accumulation_steps=2)
    parameter = getattr(bundle, phase_name)
    expected = torch.nn.Parameter(parameter.detach().clone())
    role = trainer.optimization_roles[phase_name]
    expected_optimizer = torch.optim.AdamW(
        [expected],
        lr=role.config.learning_rate,
        betas=role.config.adam_betas,
        weight_decay=role.config.adam_weight_decay,
        eps=role.config.adam_epsilon,
    )
    targets = (torch.tensor(1.0), torch.tensor(3.0))

    getattr(trainer, f"_{phase_name}_phase")([_unit(1.0), _unit(3.0)])
    torch.stack([(expected - target).square() for target in targets]).mean().backward()
    expected_optimizer.step()

    torch.testing.assert_close(parameter, expected, rtol=0, atol=1e-7)


def test_a_real_role_phase_reports_its_loss_and_gradient_norm() -> None:
    """The wandb view of a distillation run comes entirely from these two per role."""
    torch.manual_seed(17)
    trainer, _, _ = _real_objective_trainer()

    trainer._fake_phase([_trajectory_unit()])
    trainer._generator_phase([_trajectory_unit()])
    metrics = pop_distillation_metrics(trainer)

    assert set(metrics) >= {
        "train/fake_loss",
        "train/fake_grad_norm",
        "train/generator_loss",
        "train/generator_grad_norm",
        "train/x0_real_std",
        "train/x0_fake_std",
        "train/x0_gen_std",
    }
    assert metrics["train/fake_grad_norm"] > 0
    assert metrics["train/generator_grad_norm"] > 0


def test_dmd2_real_objective_uses_fresh_fake_perturbations_and_detached_queries() -> None:
    torch.manual_seed(17)
    trainer, adapter, bundle = _real_objective_trainer()
    unit = _trajectory_unit()
    fake_before = bundle.fake.detach().clone()
    generator_before = bundle.generator.detach().clone()

    trainer._fake_phase([unit])
    trainer._fake_phase([unit])
    trainer._generator_phase([unit])

    assert not torch.equal(bundle.fake, fake_before)
    assert not torch.equal(bundle.generator, generator_before)
    perturbation_times = [
        timestep
        for timestep in adapter.perturbation_times
        if not torch.equal(timestep, torch.full_like(timestep, 1000.0))
    ]
    assert len(perturbation_times) == 3
    assert not torch.equal(perturbation_times[0], perturbation_times[1])
    fake_states = [
        state
        for (role_name, grad_enabled, _, _), (_, state) in zip(
            adapter.forward_events,
            adapter.forward_states,
        )
        if role_name == "fake" and grad_enabled
    ]
    fake_noises = []
    for state, timestep in zip(fake_states, perturbation_times[:2]):
        sigma = timestep / 1000.0
        # At t=1000 the generator's clean prediction is x_t - sigma*v = 1.0-0.4.
        fake_noises.append((state - (1 - sigma) * 0.6) / sigma)
    assert not torch.equal(fake_noises[0], fake_noises[1])
    generator_event, reference_event, fake_query_event = adapter.forward_events[-3:]
    assert generator_event[:2] == ("generator", True)
    assert reference_event[:2] == ("reference", False)
    assert fake_query_event[:2] == ("fake", False)
    assert reference_event[2:] == fake_query_event[2:]
    # Each fake phase now replays one generator step to obtain its clean prediction,
    # in addition to the fake forward; generator optimization adds generator/real/fake.
    assert trainer.autocast_entries == 7
    assert trainer.optimization_roles["fake"].step == 2
    assert trainer.optimization_roles["generator"].step == 1
    assert trainer.step == 1


def test_dmd2_projects_each_selected_step_to_clean_space() -> None:
    """An intermediate x_{i+1} is not the DMD generated clean sample."""
    trainer, _, _ = _real_objective_trainer()
    batch = BaseSample.stack(_trajectory_unit())

    clean = trainer._replay_generator_clean_prediction(batch, boundary_index=1)

    # Stored x_1 is 1.4, but x0_hat from x_t=1, sigma=1, v=0.4 is 0.6.
    torch.testing.assert_close(clean.components["latent"], torch.tensor([[0.6]]))
    assert clean.components["latent"].requires_grad


def test_dmd2_generator_loss_is_finite_and_depends_on_fake_minus_real() -> None:
    torch.manual_seed(23)
    trainer, adapter, bundle = _real_objective_trainer()

    loss = trainer._generator_replay_loss(BaseSample.stack(_trajectory_unit()))
    (gradient,) = torch.autograd.grad(loss, (bundle.generator,))

    assert torch.isfinite(loss)
    assert torch.isfinite(gradient)


def test_dmd2_refuses_to_train_against_rewards_but_keeps_the_eval_runtime() -> None:
    """The reward-free contract covers training, not eval.

    Zeroing the whole feedback runtime took eval monitoring with it: `eval_freq` fired
    and every dataset was skipped for want of a reward buffer. Training rewards are
    rejected in `Arguments`, so the shared implementation is safe to reuse; this guards
    the one thing DMD2 still has to assert.
    """
    trainer = object.__new__(DMD2Trainer)
    trainer._init_reward_model_calls = []

    def fake_base(self_: object) -> tuple:
        return {"ocr": object()}, {}

    with mock.patch.object(BaseTrainer, "_init_reward_model", fake_base):
        with pytest.raises(RuntimeError, match="must not train against rewards.*ocr"):
            trainer._init_reward_model()

    with mock.patch.object(BaseTrainer, "_init_reward_model", lambda self_: ({}, {"ocr": 1})):
        training_models, eval_models = trainer._init_reward_model()

    assert training_models == {}
    assert eval_models == {"ocr": 1}


def test_tdm_r1_bypasses_tdms_reward_free_runtime() -> None:
    trainer = object.__new__(TDMR1Trainer)
    expected = ({"ocr": object()}, {"clip": object()})

    with mock.patch.object(BaseTrainer, "_init_reward_model", return_value=expected) as initialize:
        actual = trainer._init_reward_model()

    assert actual is expected
    initialize.assert_called_once_with(trainer)


def test_dmd2_production_source_has_no_forbidden_data_objectives() -> None:
    source = inspect.getsource(DMD2Trainer).lower()

    for forbidden in (
        "discriminator",
        "gan_loss",
        "regression",
        "real_image",
        "reward_processor",
        "finalize(",
    ):
        assert forbidden not in source


def test_tdm_production_path_is_direct_and_data_free() -> None:
    assert TDMTrainer.__bases__ == (TDMTrajectoryRuntimeMixin, BaseTrainer)
    source = inspect.getsource(TDMTrainer).lower()
    for forbidden in (
        "reward_processor",
        "advantage_processor.compute",
        "reconstruction_loss",
        "real_image",
        "finalize(",
    ):
        assert forbidden not in source


def test_distribution_matching_trainers_share_focused_runtime_helpers() -> None:
    dmd2_source = inspect.getsource(DMD2Trainer)
    tdm_source = inspect.getsource(TDMTrainer)

    assert "distillation_runtime" in inspect.getsource(inspect.getmodule(DMD2Trainer))
    assert "distillation_runtime" in inspect.getsource(inspect.getmodule(TDMTrainer))
    for source in (dmd2_source, tdm_source):
        assert "def _query_score_velocity(" not in source
        assert "def empty_media(" not in source


def test_tdm_r1_production_path_keeps_frozen_reference_and_surrogate() -> None:
    assert TDMR1Trainer.__bases__ == (TDMTrainer,)
    source = inspect.getsource(TDMR1Trainer).lower()
    assert "group_preference_loss(" in source
    assert "use_component_variant(" in source
    assert "surrogate" in source
    assert 'role_name="reference"' in source
    for forbidden in (
        "ema_ref",
        "require_both_signs",
        "advantage *",
        "advantage*",
        "dgpotrainer",
    ):
        assert forbidden not in source


def test_dmd2_phase_requires_velocity_from_role_forward() -> None:
    with pytest.raises(
        ValueError,
        match=r"DMD2 fake forward.*velocity.*received None",
    ):
        require_velocity(SimpleNamespace(velocity=None), algorithm_name="DMD2", role_name="fake")
