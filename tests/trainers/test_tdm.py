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

from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace
from typing import Any, Iterator, Mapping, Optional, Tuple

import pytest
import torch
from accelerate import Accelerator

from flow_factory.hparams import Arguments, TDMQuerySamplingPolicy, TDMTrainingArguments
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.minimax_h3 import MiniMaxH3T2VAAdapter
from flow_factory.samples import (
    BaseSample,
    ComponentTimes,
    ComponentTrajectory,
    LatentState,
    MultiModalStepOutput,
    StackedSampleBatch,
    StructuredTrajectory,
)
from flow_factory.scheduler import MiniMaxH3SDEScheduler, SDESchedulerOutput
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.distillation.tdm import TDMBoundaryUnit, TDMTrainer
from flow_factory.trainers.distillation.tdm_trajectory import (
    TDMQueryBatchContext,
    TDMQueryBounds,
    TDMTrajectoryRuntimeMixin,
)
from flow_factory.trainers.role_optimization import (
    OptimizationRole,
    RoleOptimizationCoordinator,
    RoleOptimizerConfig,
)
from flow_factory.utils.noise_schedule import flow_match_sigma


class TinyTDMAdapter(BaseAdapter):
    """Expose a two-component aligned schedule for TDM unit tests."""

    trajectory_component_order = ("video", "audio")

    def __init__(self) -> None:
        self.scheduler_group = {
            "video": SimpleNamespace(dynamics_type="ODE"),
            "audio": SimpleNamespace(dynamics_type="ODE"),
        }

    def load_pipeline(self) -> Any:
        raise NotImplementedError

    def decode_latents(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("TDM must not decode generated media")

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
        batch: Optional[StackedSampleBatch] = None,
    ) -> ComponentTimes:
        del batch
        audio_timesteps = primary_timesteps * 0.5
        return ComponentTimes(
            timestep={"video": primary_timesteps, "audio": audio_timesteps},
            next_timestep={
                "video": torch.zeros_like(primary_timesteps),
                "audio": torch.zeros_like(audio_timesteps),
            },
            sigma={
                "video": primary_timesteps / 1000,
                "audio": audio_timesteps / 1000,
            },
            next_sigma={
                "video": torch.zeros_like(primary_timesteps),
                "audio": torch.zeros_like(audio_timesteps),
            },
        )

    def _forward_state(
        self,
        *,
        batch: StackedSampleBatch,
        state: LatentState,
        times: ComponentTimes,
        next_state: Optional[LatentState],
        compute_log_prob: bool,
        return_fields: Tuple[str, ...],
        noise_level: Optional[float],
        forward_kwargs: Mapping[str, Any],
    ) -> MultiModalStepOutput:
        del batch, times, next_state, compute_log_prob, return_fields, noise_level, forward_kwargs
        return MultiModalStepOutput(next_state=state, velocity=state)


class ObjectiveBundle(torch.nn.Module):
    """Own generator, fake, and forbidden auxiliary parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.generator = torch.nn.Parameter(torch.tensor(1.0))
        self.fake = torch.nn.Parameter(torch.tensor(-0.2))
        self.auxiliary = torch.nn.Parameter(torch.tensor(0.7))


class ObjectiveTDMAdapter(BaseAdapter):
    """Exercise real TDM replay, score-query, and component RNG paths."""

    trajectory_component_order = ("video", "audio")

    def __init__(self, bundle: ObjectiveBundle) -> None:
        self.bundle = bundle
        self.active_role = "generator"
        self.events: list[tuple[str, bool, int, int]] = []
        self.noise_draws: list[LatentState] = []
        self.mapped_primary_times: list[torch.Tensor] = []
        self.mapped_component_times: list[ComponentTimes] = []
        self.scheduler_group = {
            "video": SimpleNamespace(dynamics_type="ODE"),
            "audio": SimpleNamespace(dynamics_type="ODE"),
        }

    def load_pipeline(self) -> Any:
        raise NotImplementedError

    def decode_latents(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("TDM must not decode generated media")

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
        batch: Optional[StackedSampleBatch] = None,
    ) -> ComponentTimes:
        del batch
        audio_timesteps = primary_timesteps * 0.5
        times = ComponentTimes(
            timestep={"video": primary_timesteps, "audio": audio_timesteps},
            next_timestep={
                "video": torch.zeros_like(primary_timesteps),
                "audio": torch.zeros_like(audio_timesteps),
            },
            sigma={
                "video": primary_timesteps / 1000,
                "audio": audio_timesteps / 1000,
            },
            next_sigma={
                "video": torch.zeros_like(primary_timesteps),
                "audio": torch.zeros_like(audio_timesteps),
            },
        )
        self.mapped_primary_times.append(primary_timesteps.detach().clone())
        self.mapped_component_times.append(times)
        return times

    @contextmanager
    def use_component_variant(self, role_name: str) -> Iterator[None]:
        # Mirror the real registry: only declared trainable variants resolve. The
        # frozen reference is a parameter snapshot and never a declared variant.
        if role_name not in ("generator", "fake"):
            raise KeyError(f"component variant {role_name!r} is not declared")
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

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        *,
        generator: torch.Generator | None = None,
    ) -> Any:
        del generator
        noise = LatentState(
            {
                name: torch.randn_like(clean_state.components[name])
                for name in self.trajectory_component_order
            }
        )
        self.noise_draws.append(
            LatentState(
                {name: values.detach().clone() for name, values in noise.components.items()}
            )
        )
        return self.apply_forward_process_noise(clean_state, times, noise)

    def _project_clean_to_score_state(
        self,
        state: LatentState,
        times: ComponentTimes,
        clean_state: LatentState,
    ) -> LatentState:
        return self._project_flow_match_clean_to_score_state(state, times, clean_state)

    def _forward_state(
        self,
        *,
        batch: StackedSampleBatch,
        state: LatentState,
        times: ComponentTimes,
        next_state: Optional[LatentState],
        compute_log_prob: bool,
        return_fields: Tuple[str, ...],
        noise_level: Optional[float],
        forward_kwargs: Mapping[str, Any],
    ) -> MultiModalStepOutput:
        del batch, next_state, compute_log_prob, return_fields, noise_level, forward_kwargs
        self.events.append((self.active_role, torch.is_grad_enabled(), id(state), id(times)))
        if self.active_role == "generator":
            generated = LatentState(
                {name: values + self.bundle.generator for name, values in state.components.items()},
                active_masks=state.active_masks,
            )
            velocity = LatentState(
                {
                    name: self.bundle.generator.expand_as(values)
                    for name, values in state.components.items()
                },
                active_masks=state.active_masks,
            )
            return MultiModalStepOutput(
                next_state=generated,
                next_state_mean=generated,
                velocity=velocity,
            )
        role_value = (
            self.bundle.fake if self.active_role == "fake" else torch.zeros_like(self.bundle.fake)
        )
        return MultiModalStepOutput(
            velocity=LatentState(
                {name: role_value.expand_as(values) for name, values in state.components.items()},
                active_masks=state.active_masks,
            )
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


def _sample(
    *,
    state_index_map: torch.Tensor | None = None,
    audio_state_index_map: torch.Tensor | None = None,
    video_states: torch.Tensor | None = None,
    audio_states: torch.Tensor | None = None,
    video_times: torch.Tensor | None = None,
    audio_times: torch.Tensor | None = None,
    video_sigmas: torch.Tensor | None = None,
    audio_sigmas: torch.Tensor | None = None,
) -> BaseSample:
    index_map = (
        torch.tensor([0, 1, 2], dtype=torch.int64) if state_index_map is None else state_index_map
    )
    video_schedule = torch.tensor([1000.0, 500.0, 0.0]) if video_times is None else video_times
    audio_schedule = torch.tensor([500.0, 250.0, 0.0]) if audio_times is None else audio_times
    return BaseSample(
        prompt_embeds=torch.tensor([1.0]),
        trajectory=StructuredTrajectory(
            components={
                "video": ComponentTrajectory(
                    states=(
                        torch.tensor([[0.0], [1.0], [2.0]])
                        if video_states is None
                        else video_states
                    ),
                    timesteps=video_schedule,
                    sigmas=video_schedule / 1000 if video_sigmas is None else video_sigmas,
                    state_index_map=index_map,
                ),
                "audio": ComponentTrajectory(
                    states=(
                        torch.tensor([[10.0], [11.0], [12.0]])
                        if audio_states is None
                        else audio_states
                    ),
                    timesteps=audio_schedule,
                    sigmas=audio_schedule / 1000 if audio_sigmas is None else audio_sigmas,
                    state_index_map=(
                        index_map.clone()
                        if audio_state_index_map is None
                        else audio_state_index_map
                    ),
                ),
            }
        ),
    )


def _trainer(**overrides: Any) -> TDMTrainer:
    trainer = object.__new__(TDMTrainer)
    defaults = {
        "num_inference_steps": 2,
        "num_inner_epochs": 1,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "ttur_fake_updates": 1,
        "use_huber": False,
        "huber_c": 1e-3,
        "tdm_snr_gamma": 5.0,
        "tdm_query_distribution": "conditional_logit_normal",
        "tdm_query_logit_mean": 0.0,
        "tdm_query_logit_std": 1.0,
        # Keep original trajectory tests on stored intervals; reverse is tested separately.
        "tdm_query_interval": "trajectory",
        "tdm_query_max_sigma": 1.0,
        "replay_rtol": 1e-6,
        "replay_atol": 1e-6,
    }
    defaults.update(overrides)
    trainer.training_args = SimpleNamespace(**defaults)
    trainer.training_args.get_num_train_timesteps = (
        lambda config: trainer.training_args.num_inference_steps
    )
    trainer.config = SimpleNamespace()
    trainer.adapter = TinyTDMAdapter()
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), is_local_main_process=True)
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.autocast = nullcontext
    trainer.step = 0
    trainer.epoch = 0
    trainer._tdm_generation_provenance = {}
    return trainer


def _boundary_times(
    adapter: BaseAdapter,
    interval_start: torch.Tensor,
    interval_end: torch.Tensor,
) -> tuple[ComponentTimes, ComponentTimes]:
    current = adapter.build_training_component_times(interval_end)
    following = adapter.build_training_component_times(interval_start)
    return (
        ComponentTimes(
            timestep=current.timestep,
            next_timestep=following.timestep,
            sigma=current.sigma,
            next_sigma=following.sigma,
        ),
        following,
    )


def _boundary_unit(
    trainer: TDMTrainer,
    times: ComponentTimes,
    mid_times: ComponentTimes,
) -> TDMBoundaryUnit:
    """Build one explicit-bound test unit outside normal trajectory construction."""
    context = TDMQueryBatchContext(
        policy=TDMQuerySamplingPolicy.from_training_args(trainer.training_args),
        primary_name="video",
        source_transform=None,
        reverse_upper_times=None,
    )
    return TDMBoundaryUnit(
        samples=(_sample(),),
        boundary_index=1,
        times=times,
        mid_times=mid_times,
        query_context=context,
        query_bounds=TDMQueryBounds(lower=mid_times, upper=times),
    )


def _objective_trainer() -> tuple[TDMTrainer, ObjectiveTDMAdapter, ObjectiveBundle]:
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    bundle = ObjectiveBundle()
    adapter = ObjectiveTDMAdapter(bundle)
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
    trainer = object.__new__(TDMTrainer)
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
    trainer.training_args = TDMTrainingArguments(
        num_inference_steps=2,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,
        ttur_fake_updates=1,
        use_huber=False,
        replay_rtol=0,
        replay_atol=0,
    )
    trainer.config = SimpleNamespace()
    trainer.autocast = nullcontext
    trainer.step = 0
    trainer.epoch = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer._tdm_generation_provenance = {}
    return trainer, adapter, bundle


def test_tdm_is_direct_deterministic_distillation_trainer() -> None:
    assert TDMTrainer.__bases__ == (TDMTrajectoryRuntimeMixin, BaseTrainer)
    assert TDMTrainer.paradigm == "distillation"


def test_tdm_config_counts_every_boundary_in_one_generator_window() -> None:
    training_args = TDMTrainingArguments()

    assert training_args.gradient_step_per_epoch == 1
    assert training_args.get_num_train_timesteps(SimpleNamespace()) == 4

    resolved = Arguments.from_dict(
        {
            "train": {
                "trainer_type": "tdm",
                "num_inference_steps": 4,
                "unique_sample_num_per_epoch": 8,
                "per_device_batch_size": 1,
            },
            "scheduler": {"dynamics_type": "ODE"},
        }
    ).training_args
    assert resolved.num_batches_per_epoch == 8
    assert resolved.gradient_accumulation_steps == 32


@pytest.mark.parametrize("manual_gas", [8, 16, 64])
def test_tdm_config_accepts_manual_gas(manual_gas: int) -> None:
    training_args = Arguments.from_dict(
        {
            "train": {
                "trainer_type": "tdm",
                "num_inference_steps": 4,
                "unique_sample_num_per_epoch": 8,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": manual_gas,
            },
            "scheduler": {"dynamics_type": "ODE"},
        }
    ).training_args

    assert training_args.gradient_accumulation_steps == manual_gas


def test_tdm_builds_every_boundary_with_half_open_non_overlapping_intervals() -> None:
    trainer = _trainer()
    sample = _sample()

    units = trainer._build_boundary_units([sample])

    assert [unit.boundary_index for unit in units] == [1, 2]
    torch.testing.assert_close(units[0].interval_start, torch.tensor([500.0]))
    torch.testing.assert_close(units[0].interval_end, torch.tensor([1000.0]))
    torch.testing.assert_close(units[1].interval_start, torch.tensor([0.0]))
    torch.testing.assert_close(units[1].interval_end, torch.tensor([500.0]))
    assert all(isinstance(unit, TDMBoundaryUnit) for unit in units)
    assert all(len(unit.samples) == 1 and unit.samples[0] is sample for unit in units)


def test_tdm_keeps_stored_component_coordinates_at_discrete_boundaries() -> None:
    trainer = _trainer()
    stored_audio_timestep = torch.nextafter(torch.tensor(250.0), torch.tensor(float("inf")))
    stored_audio_sigma = torch.nextafter(torch.tensor(0.25), torch.tensor(float("inf")))
    sample = _sample(
        audio_times=torch.tensor([500.0, stored_audio_timestep.item(), 0.0]),
        audio_sigmas=torch.tensor([0.5, stored_audio_sigma.item(), 0.0]),
    )

    first, second = trainer._build_boundary_units([sample])

    assert torch.equal(first.mid_times.timestep["audio"], stored_audio_timestep.reshape(1))
    assert torch.equal(first.mid_times.sigma["audio"], stored_audio_sigma.reshape(1))
    assert torch.equal(second.mid_times.timestep["audio"], torch.zeros(1))
    assert torch.equal(second.mid_times.sigma["audio"], torch.zeros(1))


def test_tdm_preserves_each_stored_coordinate_native_dtype() -> None:
    trainer = _trainer()
    video_timesteps = torch.tensor([1000.0, 500.0, 0.0], dtype=torch.float16)
    video_sigmas = torch.tensor([1.0, 0.5002, 0.0], dtype=torch.float32)

    first, second = trainer._build_boundary_units(
        [_sample(video_times=video_timesteps, video_sigmas=video_sigmas)]
    )

    assert first.times.timestep["video"].dtype == torch.float16
    assert first.mid_times.sigma["video"].dtype == torch.float32
    assert torch.equal(first.mid_times.sigma["video"], video_sigmas[1:2])
    assert torch.equal(second.times.sigma["video"], video_sigmas[1:2])


def _h3_trainer_and_sample(
    num_steps: int,
) -> tuple[TDMTrainer, BaseSample, MiniMaxH3T2VAAdapter, torch.Tensor, torch.Tensor]:
    adapter = object.__new__(MiniMaxH3T2VAAdapter)
    adapter.pipeline = SimpleNamespace()
    adapter.scheduler = MiniMaxH3SDEScheduler(shift=12.0, dynamics_type="ODE")
    adapter.audio_scheduler = MiniMaxH3SDEScheduler(shift=3.0, dynamics_type="ODE")
    adapter.scheduler.set_timesteps(num_steps)
    adapter.audio_scheduler.set_timesteps(num_steps)
    adapter.scheduler_group = {
        "video": adapter.scheduler,
        "audio": adapter.audio_scheduler,
    }
    video_timesteps = adapter.scheduler.sigmas * 1000
    audio_timesteps = adapter.audio_scheduler.sigmas * 1000
    index_map = torch.arange(num_steps + 1, dtype=torch.int64)
    sample = BaseSample(
        prompt_embeds=torch.tensor([1.0]),
        trajectory=StructuredTrajectory(
            components={
                "video": ComponentTrajectory(
                    states=torch.arange(num_steps + 1, dtype=torch.float32).reshape(-1, 1),
                    timesteps=video_timesteps,
                    sigmas=adapter.scheduler.sigmas,
                    state_index_map=index_map,
                ),
                "audio": ComponentTrajectory(
                    states=torch.arange(num_steps + 1, dtype=torch.float32).reshape(-1, 1),
                    timesteps=audio_timesteps,
                    sigmas=adapter.audio_scheduler.sigmas,
                    state_index_map=index_map.clone(),
                ),
            }
        ),
    )
    trainer = _trainer(num_inference_steps=num_steps)
    trainer.adapter = adapter
    return trainer, sample, adapter, video_timesteps, audio_timesteps


@pytest.mark.parametrize("num_steps", [4, 10, 50, 1000])
def test_tdm_uses_authoritative_h3_boundaries_without_grid_snapping(num_steps: int) -> None:
    trainer, sample, adapter, video_timesteps, audio_timesteps = _h3_trainer_and_sample(num_steps)
    analytic = adapter.build_training_component_times(video_timesteps)
    assert not torch.equal(analytic.timestep["audio"], audio_timesteps)

    units = trainer._build_boundary_units([sample])

    assert len(units) == num_steps
    for index, unit in enumerate(units):
        assert torch.equal(unit.times.timestep["audio"], audio_timesteps[index : index + 1])
        assert torch.equal(
            unit.mid_times.timestep["audio"],
            audio_timesteps[index + 1 : index + 2],
        )
        assert torch.equal(
            unit.mid_times.sigma["audio"],
            adapter.audio_scheduler.sigmas[index + 1 : index + 2],
        )


def test_tdm_resamples_h3_secondary_endpoint_rounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, sample, _, _, _ = _h3_trainer_and_sample(50)
    unit = trainer._build_boundary_units([sample])[0]
    batch = trainer._stack_replay_unit(unit.samples)
    fractions = iter((torch.finfo(torch.float32).eps, 0.5))
    draw_count = 0

    def boundary_then_midpoint(*shape: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal draw_count
        del shape
        draw_count += 1
        return torch.full(
            unit.interval_start.shape,
            next(fractions),
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )

    monkeypatch.setattr(torch, "rand", boundary_then_midpoint)

    times = trainer._sample_score_query_times(unit, batch)

    assert draw_count == 2
    for name in trainer.adapter.trajectory_component_order:
        assert bool((times.timestep[name] > unit.times.next_timestep[name]).all())
        assert bool((times.timestep[name] < unit.times.timestep[name]).all())
        assert bool((times.sigma[name] > unit.times.next_sigma[name]).all())
        assert bool((times.sigma[name] < unit.times.sigma[name]).all())


def test_tdm_fails_when_no_jointly_representable_mapped_interior_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    unit = trainer._build_boundary_units([_sample()])[0]
    batch = trainer._stack_replay_unit(unit.samples)
    original_mapping = trainer.adapter.build_training_component_times
    mapping_calls = 0

    def map_audio_to_lower_boundary(
        primary_timesteps: torch.Tensor,
        *,
        batch: StackedSampleBatch | None = None,
    ) -> ComponentTimes:
        nonlocal mapping_calls
        mapping_calls += 1
        mapped = original_mapping(primary_timesteps, batch=batch)
        lower_timestep = unit.times.next_timestep["audio"]
        lower_sigma = unit.times.next_sigma["audio"]
        mapped.timestep["audio"] = lower_timestep
        mapped.sigma["audio"] = lower_sigma
        return mapped

    trainer.adapter.build_training_component_times = map_audio_to_lower_boundary
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *shape, **kwargs: torch.full(
            unit.interval_start.shape,
            0.5,
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"jointly representable.*after 8 attempts.*boundary_index=1",
    ):
        trainer._sample_score_query_times(unit, batch)

    assert mapping_calls == 8


def test_tdm_samples_inside_each_boundary_half_open_interval() -> None:
    trainer = _trainer()
    units = trainer._build_boundary_units([_sample()])
    torch.manual_seed(7)

    for unit in units:
        sampled = torch.stack([trainer._sample_perturbation_times(unit) for _ in range(32)])
        assert bool((sampled >= unit.interval_start).all())
        assert bool((sampled < unit.interval_end).all())


@pytest.mark.parametrize(
    "random_value", [0.0, torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item()]
)
def test_tdm_sampling_uses_representable_open_interval_bounds(
    monkeypatch: pytest.MonkeyPatch,
    random_value: float,
) -> None:
    trainer = _trainer()
    terminal_unit = trainer._build_boundary_units([_sample()])[-1]

    def deterministic_rand(*shape: Any, **kwargs: Any) -> torch.Tensor:
        del shape
        return torch.full(
            terminal_unit.interval_start.shape,
            random_value,
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )

    monkeypatch.setattr(torch, "rand", deterministic_rand)
    sampled = trainer._sample_perturbation_times(terminal_unit)

    assert bool((sampled > terminal_unit.interval_start).all())
    assert bool((sampled < terminal_unit.interval_end).all())
    assert bool((sampled > 0).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tdm_upper_interior_maps_to_sigma_strictly_below_one_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    interval_start = torch.tensor([900.0], device="cuda", dtype=torch.float32)
    interval_end = torch.tensor([1000.0], device="cuda", dtype=torch.float32)
    times, mid_times = _boundary_times(trainer.adapter, interval_start, interval_end)
    unit = _boundary_unit(trainer, times, mid_times)

    def upper_rand(*shape: Any, **kwargs: Any) -> torch.Tensor:
        del shape
        return torch.full(
            interval_start.shape,
            torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item(),
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )

    monkeypatch.setattr(torch, "rand", upper_rand)
    sampled = trainer._sample_perturbation_times(unit)
    sigma_mid = flow_match_sigma(interval_start)
    sigma_t = flow_match_sigma(sampled)

    assert bool((sigma_mid < sigma_t).all())
    assert bool((sigma_t < 1).all())


def test_tdm_terminal_generator_score_path_keeps_mapped_sigmas_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, adapter, _ = _objective_trainer()
    terminal_unit = trainer._build_boundary_units([_sample()])[-1]

    def zero_rand(*shape: Any, **kwargs: Any) -> torch.Tensor:
        del shape
        return torch.zeros(
            terminal_unit.interval_start.shape,
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )

    monkeypatch.setattr(torch, "rand", zero_rand)
    loss = trainer._generator_boundary_loss(terminal_unit)

    sampled_tau = adapter.mapped_primary_times[-1]
    mapped_times = adapter.mapped_component_times[-1]
    assert bool((sampled_tau > 0).all())
    assert mapped_times.sigma is not None
    assert all(
        bool((mapped_times.sigma[name] > 0).all()) for name in adapter.trajectory_component_order
    )
    assert bool(torch.isfinite(loss).item())
    assert [event[0] for event in adapter.events] == ["generator", "reference", "fake"]


def test_tdm_rejects_zero_custom_mapped_sigma_before_score_queries() -> None:
    trainer, adapter, _ = _objective_trainer()
    terminal_unit = trainer._build_boundary_units([_sample()])[-1]
    original_mapping = adapter.build_training_component_times

    def zero_audio_sigma(
        primary_timesteps: torch.Tensor,
        *,
        batch: StackedSampleBatch | None = None,
    ) -> ComponentTimes:
        mapped = original_mapping(primary_timesteps, batch=batch)
        assert mapped.sigma is not None
        return ComponentTimes(
            timestep=mapped.timestep,
            next_timestep=mapped.next_timestep,
            sigma={
                "video": mapped.sigma["video"],
                "audio": torch.zeros_like(mapped.sigma["audio"]),
            },
            next_sigma=mapped.next_sigma,
        )

    adapter.build_training_component_times = zero_audio_sigma
    with pytest.raises(
        ValueError,
        match=r"mapped continuous.*component 'audio'.*boundary_index=2.*sigma \* 1000.*ULP",
    ):
        trainer._generator_boundary_loss(terminal_unit)

    assert [event[0] for event in adapter.events] == ["generator"]


def test_tdm_rejects_interval_without_representable_interior() -> None:
    trainer = _trainer()
    start = torch.tensor([1.0])
    end = torch.nextafter(start, torch.tensor([float("inf")]))
    times, mid_times = _boundary_times(trainer.adapter, start, end)
    unit = _boundary_unit(trainer, times, mid_times)

    with pytest.raises(ValueError, match=r"no representable floating interior"):
        trainer._sample_perturbation_times(unit)


@pytest.mark.parametrize(
    ("video_times", "expected_type"),
    [
        (torch.tensor([True, True, False]), "bool"),
        (torch.tensor([1000 + 0j, 500 + 0j, 0 + 0j]), "complex"),
    ],
)
def test_tdm_rejects_nonreal_primary_coordinates(
    video_times: torch.Tensor,
    expected_type: str,
) -> None:
    trainer = _trainer()

    with pytest.raises(
        TypeError,
        match=rf"stored timestep.*component='video'.*real numeric.*{expected_type}",
    ):
        trainer._build_boundary_units([_sample(video_times=video_times)])


def test_tdm_accepts_integral_primary_coordinates() -> None:
    trainer = _trainer()
    video_times = torch.tensor([1000, 500, 0], dtype=torch.int64)

    units = trainer._build_boundary_units([_sample(video_times=video_times)])
    sampled = trainer._sample_perturbation_times(units[0])

    assert len(units) == 2
    assert units[0].times.timestep["video"].dtype == torch.int64
    assert sampled.is_floating_point()
    assert bool(((sampled < 1000) & (sampled > 500)).all())


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_tdm_rejects_nonfinite_primary_coordinates(bad_value: float) -> None:
    trainer = _trainer()
    video_times = torch.tensor([1000.0, 500.0, bad_value])

    with pytest.raises(
        ValueError,
        match=r"stored next_timestep.*component='video'.*boundary_index=2.*finite",
    ):
        trainer._build_boundary_units([_sample(video_times=video_times)])


@pytest.mark.parametrize(
    ("bad_value", "error_type", "match"),
    [
        (torch.tensor([True]), TypeError, r"real numeric.*bool"),
        (torch.tensor([500 + 0j]), TypeError, r"real numeric.*complex"),
        (torch.tensor([float("nan")]), ValueError, r"finite"),
        (torch.tensor([float("inf")]), ValueError, r"finite"),
    ],
)
def test_tdm_rejects_invalid_component_coordinates(
    bad_value: torch.Tensor,
    error_type: type[Exception],
    match: str,
) -> None:
    trainer = _trainer()

    def invalid_component_times(
        primary_timesteps: torch.Tensor,
        *,
        batch: StackedSampleBatch | None = None,
    ) -> ComponentTimes:
        del batch
        invalid = bad_value.to(primary_timesteps.device).expand_as(primary_timesteps)
        return ComponentTimes(
            timestep={"video": primary_timesteps, "audio": invalid},
            next_timestep={
                "video": torch.zeros_like(primary_timesteps),
                "audio": torch.zeros_like(invalid),
            },
            sigma={
                "video": primary_timesteps / 1000,
                "audio": primary_timesteps / 2000,
            },
            next_sigma={
                "video": torch.zeros_like(primary_timesteps),
                "audio": torch.zeros_like(primary_timesteps),
            },
        )

    unit = trainer._build_boundary_units([_sample()])[0]
    primary_times = (unit.interval_start + unit.interval_end) / 2
    times = invalid_component_times(primary_times)
    with pytest.raises(
        error_type,
        match=rf"mapped continuous timestep/sigma.*component 'audio'.*{match}",
    ):
        trainer._validate_score_query_coordinates(times, primary_times, unit=unit)


def test_tdm_rejects_a_mapped_component_time_outside_the_stored_interval() -> None:
    trainer = _trainer()
    unit = trainer._build_boundary_units([_sample()])[0]
    primary_times = (unit.interval_start + unit.interval_end) / 2
    times = trainer.adapter.build_training_component_times(primary_times)
    times.timestep["audio"] = unit.times.next_timestep["audio"].clone()
    times.sigma["audio"] = unit.times.next_sigma["audio"].clone()

    with pytest.raises(
        ValueError,
        match=r"continuous timestep.*strictly inside.*component='audio'",
    ):
        trainer._validate_score_query_coordinates(times, primary_times, unit=unit)


def test_tdm_resamples_legacy_sigma_that_rounds_to_one() -> None:
    trainer = _trainer()
    lower_primary = torch.tensor([900.0])
    upper_primary = torch.tensor([1000.0])
    current = trainer.adapter.build_training_component_times(upper_primary)
    following = trainer.adapter.build_training_component_times(lower_primary)
    stored_times = ComponentTimes(
        timestep=current.timestep,
        next_timestep=following.timestep,
    )
    unit = _boundary_unit(trainer, stored_times, following)
    primary_times = torch.nextafter(upper_primary, lower_primary)
    times = trainer.adapter.build_training_component_times(primary_times)
    times.sigma["video"] = torch.ones_like(times.sigma["video"])

    with pytest.raises(
        ValueError,
        match=r"continuous sigma.*lower boundary.*strictly below one.*component='video'",
    ):
        trainer._validate_score_query_coordinates(times, primary_times, unit=unit)


@pytest.mark.parametrize(
    ("current_end", "previous_start", "kind"),
    [
        (500.0 + 1e-4, 500.0, "overlap"),
        (500.0 - 1e-4, 500.0, "gap"),
    ],
)
def test_tdm_rejects_tiny_nonzero_interval_gap_or_overlap(
    current_end: float,
    previous_start: float,
    kind: str,
) -> None:
    trainer = _trainer()
    batch = trainer._stack_replay_unit((_sample(),))
    times, _ = _boundary_times(
        trainer.adapter,
        torch.tensor([0.0]),
        torch.tensor([current_end]),
    )
    previous_times, _ = _boundary_times(
        trainer.adapter,
        torch.tensor([previous_start]),
        torch.tensor([1000.0]),
    )

    with pytest.raises(ValueError, match=rf"exact.*{kind}"):
        trainer._validate_interval(
            batch,
            times,
            boundary_index=2,
            previous_times=previous_times,
        )


def test_tdm_rejects_tiny_nonzero_terminal_coordinate() -> None:
    trainer = _trainer()
    tiny_terminal = torch.tensor([1e-7])
    batch = trainer._stack_replay_unit((_sample(),))
    end_times = trainer.adapter.build_training_component_times(torch.tensor([500.0]), batch=batch)
    start_times = trainer.adapter.build_training_component_times(tiny_terminal, batch=batch)
    times = ComponentTimes(
        timestep=end_times.timestep,
        next_timestep=start_times.timestep,
        sigma=end_times.sigma,
        next_sigma=start_times.sigma,
    )
    previous_times, _ = _boundary_times(
        trainer.adapter,
        torch.tensor([500.0]),
        torch.tensor([1000.0]),
    )

    with pytest.raises(ValueError, match=r"terminal.*exact.*zero"):
        trainer._validate_interval(
            batch,
            times,
            boundary_index=2,
            previous_times=previous_times,
        )


def test_tdm_rejects_tiny_component_topology_overlap() -> None:
    trainer = _trainer()
    batch = trainer._stack_replay_unit((_sample(),))
    previous_times, _ = _boundary_times(
        trainer.adapter,
        torch.tensor([500.0]),
        torch.tensor([1000.0]),
    )
    times, _ = _boundary_times(
        trainer.adapter,
        torch.tensor([0.0]),
        torch.tensor([500.0]),
    )
    times.timestep["audio"] = times.timestep["audio"] + 1e-4
    times.sigma["audio"] = times.timestep["audio"] / 1000

    with pytest.raises(ValueError, match=r"component='audio'.*timestep.*exact.*overlap"):
        trainer._validate_interval(
            batch,
            times,
            boundary_index=2,
            previous_times=previous_times,
        )


@pytest.mark.parametrize(
    ("sample_kwargs", "match"),
    [
        ({"state_index_map": torch.tensor([0, 0, 1])}, "one-to-one"),
        ({"state_index_map": torch.tensor([0, 2, 1])}, "aligned.*arange"),
        (
            {"audio_state_index_map": torch.tensor([0, 2, 1])},
            "component='audio'.*match.*component='video'",
        ),
        (
            {"video_states": torch.tensor([[0.0], [1.0], [2.0], [3.0]])},
            "stored states.*K \\+ 1=3",
        ),
    ],
)
def test_tdm_rejects_non_dense_or_misaligned_state_maps(
    sample_kwargs: dict[str, torch.Tensor],
    match: str,
) -> None:
    trainer = _trainer()

    with pytest.raises(ValueError, match=match):
        trainer._build_boundary_units([_sample(**sample_kwargs)])


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        # A hole in the map is now caught by the replay accessor, which reports which
        # transition could not be read rather than which map entry was -1.
        ({"state_index_map": torch.tensor([0, -1, 2])}, "reading transition .* failed"),
        ({"video_times": torch.tensor([1000.0, 500.0, 100.0])}, "gap|terminal"),
        ({"video_times": torch.tensor([1000.0, 1100.0, 0.0])}, "reversed"),
    ],
)
def test_tdm_rejects_invalid_boundary_geometry(
    overrides: dict[str, torch.Tensor],
    match: str,
) -> None:
    trainer = _trainer()

    with pytest.raises(ValueError, match=match):
        trainer._build_boundary_units([_sample(**overrides)])


def test_tdm_rejects_non_ode_before_building_units() -> None:
    non_ode = _trainer()
    non_ode.adapter.scheduler_group["audio"].dynamics_type = "Flow-SDE"
    with pytest.raises(ValueError, match=r"deterministic ODE.*audio.*Flow-SDE"):
        non_ode._build_boundary_units([_sample()])


def test_tdm_optimize_runs_fake_ttur_then_generator_over_identical_ordered_units() -> None:
    trainer = _trainer(ttur_fake_updates=3, gradient_accumulation_steps=4)
    events: list[tuple[str, tuple[tuple[int, float], ...]]] = []

    def record(self: TDMTrainer, units: list, name: str) -> None:
        ordered = []
        for unit in units:
            trajectory = unit.samples[0].trajectory
            assert trajectory is not None
            ordered.append(
                (
                    unit.boundary_index,
                    float(trajectory.components["video"].states[0].item()),
                )
            )
        events.append((name, tuple(ordered)))

    trainer._fake_phase = MethodType(lambda self, units: record(self, units, "fake"), trainer)
    trainer._generator_phase = MethodType(
        lambda self, units: record(self, units, "generator"), trainer
    )

    trainer.optimize(
        [
            [_sample()],
            [
                _sample(
                    video_states=torch.tensor([[10.0], [11.0], [12.0]]),
                    audio_states=torch.tensor([[20.0], [21.0], [22.0]]),
                )
            ],
        ]
    )

    ordered = ((1, 0.0), (2, 0.0), (1, 10.0), (2, 10.0))
    assert events == [
        ("fake", ordered),
        ("fake", ordered),
        ("fake", ordered),
        ("generator", ordered),
    ]


def test_tdm_optimize_rejects_rollout_batch_count_mismatch() -> None:
    trainer = _trainer(gradient_accumulation_steps=4)

    with pytest.raises(ValueError, match=r"TDM optimize expected 2 role microbatches.*received 1"):
        trainer.optimize([_sample()])


def test_tdm_generator_loss_is_finite_and_queries_reference_and_fake() -> None:
    trainer, adapter, bundle = _objective_trainer()
    unit = trainer._build_boundary_units([_sample()])[0]
    trainer._sample_perturbation_times = MethodType(
        lambda self, selected: (selected.interval_start + selected.interval_end) / 2,
        trainer,
    )

    loss = trainer._generator_boundary_loss(unit)
    (gradient,) = torch.autograd.grad(loss, (bundle.generator,))

    assert torch.isfinite(loss)
    assert torch.isfinite(gradient)
    assert adapter.events[0][:2] == ("generator", True)


def test_tdm_remaps_only_the_sampled_continuous_interior_during_training() -> None:
    trainer, adapter, _ = _objective_trainer()
    unit = trainer._build_boundary_units([_sample()])[0]
    adapter.mapped_primary_times.clear()
    adapter.mapped_component_times.clear()

    trainer._generator_boundary_loss(unit)

    assert len(adapter.mapped_primary_times) == 1
    sampled = adapter.mapped_primary_times[0]
    assert bool((sampled > unit.interval_start).all())
    assert bool((sampled < unit.interval_end).all())


def test_tdm_replay_mismatch_fails_before_score_queries() -> None:
    trainer, adapter, _ = _objective_trainer()
    mismatched = _sample(
        video_states=torch.tensor([[0.0], [9.0], [10.0]]),
    )
    unit = trainer._build_boundary_units([mismatched])[0]

    with pytest.raises(ValueError, match=r"replay_generator_boundary.*mismatch|replay.*mismatch"):
        trainer._generator_boundary_loss(unit)

    assert [event[0] for event in adapter.events] == ["generator"]


def test_tdm_reference_and_fake_receive_identical_detached_state_and_times_objects() -> None:
    trainer, adapter, _ = _objective_trainer()
    unit = trainer._build_boundary_units([_sample()])[0]

    trainer._generator_boundary_loss(unit)

    generator_event, reference_event, fake_event = adapter.events
    assert generator_event[:2] == ("generator", True)
    assert reference_event[:2] == ("reference", False)
    assert fake_event[:2] == ("fake", False)
    assert reference_event[2:] == fake_event[2:]


def test_tdm_generator_graph_owns_one_transition_and_no_auxiliary_parameters() -> None:
    trainer, adapter, bundle = _objective_trainer()
    unit = trainer._build_boundary_units([_sample()])[0]

    trainer._generator_boundary_loss(unit).backward()

    assert bundle.generator.grad is not None
    assert bundle.fake.grad is None
    assert bundle.auxiliary.grad is None
    assert sum(role == "generator" and grad for role, grad, _, _ in adapter.events) == 1


def test_tdm_fake_units_draw_fresh_noise_for_every_component_and_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, adapter, _ = _objective_trainer()
    first_unit, second_unit = trainer._build_boundary_units([_sample()])
    torch.manual_seed(31)
    original_randn_like = torch.randn_like
    draws: list[torch.Tensor] = []

    def recording_randn_like(value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        draw = original_randn_like(value, *args, **kwargs)
        draws.append(draw.detach().clone())
        return draw

    monkeypatch.setattr(torch, "randn_like", recording_randn_like)

    trainer._fake_boundary_loss(first_unit)
    trainer._fake_boundary_loss(second_unit)
    trainer._fake_boundary_loss(first_unit)

    component_count = len(adapter.trajectory_component_order)
    assert len(draws) == 3 * component_count
    grouped = [
        draws[offset : offset + component_count] for offset in range(0, len(draws), component_count)
    ]
    for draw in grouped:
        assert not torch.equal(draw[0], draw[1])
    for component_index in range(component_count):
        component_draws = [draw[component_index] for draw in grouped]
        assert all(
            not torch.equal(component_draws[left], component_draws[right])
            for left in range(len(component_draws))
            for right in range(left + 1, len(component_draws))
        )


def test_tdm_media_suppression_restores_decoder_after_rollout_scope() -> None:
    trainer = _trainer()

    def decoder(video_latents: torch.Tensor, audio_latents: torch.Tensor) -> Any:
        del video_latents, audio_latents
        raise AssertionError("TDM must not decode generated media")

    trainer.adapter.decode_latents = decoder

    with trainer._without_media_decoding():
        assert trainer.adapter.decode_latents(
            torch.zeros(2, 3),
            torch.zeros(2, 4),
        ) == ([None, None], [None, None])

    assert trainer.adapter.decode_latents is decoder
    with pytest.raises(AssertionError, match="must not decode"):
        trainer.adapter.decode_latents(torch.zeros(2, 3), torch.zeros(2, 4))


def test_tdm_real_role_phases_advance_exact_gas_counters() -> None:
    trainer, _, _ = _objective_trainer()
    boundary_units = trainer._flatten_boundary_units([[_sample()]])

    trainer._fake_phase(boundary_units)
    trainer._generator_phase(boundary_units)

    assert trainer.optimization_roles["fake"].step == 1
    assert trainer.optimization_roles["generator"].step == 1
    assert trainer.step == 1
