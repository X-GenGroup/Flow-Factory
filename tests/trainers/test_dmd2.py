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

import re
from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace
from typing import Any, Iterator, List, Tuple

import pytest
import torch

from flow_factory.hparams import (
    AdamWOptimizerArguments,
    Arguments,
    DMD2TrainingArguments,
    MultiOptimizerArguments,
)
from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import BaseSample, ComponentTimes, LatentState, MultiModalStepOutput
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.distillation import dmd2 as dmd2_module
from flow_factory.trainers.distillation.distillation_runtime import (
    _completed_rollout_batch_count,
    resolve_rollout_sets_per_cycle,
)
from flow_factory.trainers.distillation.dmd2 import DMD2Trainer
from flow_factory.trainers.execution import TrainingProgress


def _sample(value: float = 1.0) -> BaseSample:
    return BaseSample(
        timesteps=torch.tensor([1000.0]),
        all_latents=torch.tensor([[value], [value + 1.0]]),
        latent_index_map=torch.tensor([0, 1]),
    )


class _FakeCoordinator:
    """Record sequential role phases."""

    def __init__(self) -> None:
        self.entered: List[str] = []


def _trainer(*, fake_update_frequency: int | None = None, **training_overrides: Any) -> DMD2Trainer:
    trainer = object.__new__(DMD2Trainer)
    defaults = {
        "num_inference_steps": 1,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "get_num_train_timesteps": lambda _config: 1,
        "required_trainable_roles": ("generator", "fake"),
        "role_update_plan": DMD2TrainingArguments().role_update_plan,
        "num_inner_epochs": 1,
    }
    defaults.update(training_overrides)
    trainer.training_args = SimpleNamespace(**defaults)
    optimizer_configs = []
    if fake_update_frequency is not None:
        optimizer_configs.append(
            AdamWOptimizerArguments(name="fake", update_frequency=fake_update_frequency)
        )
    trainer.config = SimpleNamespace(
        optimizer_args=MultiOptimizerArguments(optimizer_configs=optimizer_configs)
    )
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"), is_local_main_process=True)
    trainer.log_args = SimpleNamespace(verbose=False)

    class SchedulerGroupFake:
        primary = SimpleNamespace(seed=42)

        def sample_ode_step_index(self, draw_index: int) -> int:
            generator = torch.Generator().manual_seed(self.primary.seed + draw_index)
            return int(torch.randperm(defaults["num_inference_steps"], generator=generator)[0])

    trainer.adapter = SimpleNamespace(train=lambda: None, scheduler_group=SchedulerGroupFake())
    trainer.autocast = nullcontext
    trainer.role_optimization = _FakeCoordinator()
    trainer.dataloader = None
    trainer._rollout_data_iter = None
    trainer._rollout_batches_consumed = None
    trainer.step = 0
    trainer.epoch = 0
    return trainer


def test_dmd2_is_direct_data_free_distillation_trainer() -> None:
    assert DMD2Trainer.__bases__ == (BaseTrainer,)
    assert DMD2Trainer.paradigm == "distillation"


@pytest.mark.parametrize("num_inference_steps", [0, -1, 1.5])
def test_dmd2_rejects_a_schedule_with_no_boundary_to_match_on(
    num_inference_steps: object,
) -> None:
    trainer = _trainer(num_inference_steps=num_inference_steps)

    with pytest.raises(
        ValueError,
        match=rf"num_inference_steps as an int >= 1.*received {re.escape(repr(num_inference_steps))}",
    ):
        trainer._validate_generation_schedule()


@pytest.mark.parametrize("num_inference_steps", [1, 2, 4])
def test_dmd2_accepts_a_multi_step_generator(num_inference_steps: int) -> None:
    trainer = _trainer(num_inference_steps=num_inference_steps)

    trainer._validate_generation_schedule()

    drawn = {trainer._draw_boundary_index() for _ in range(200)}
    assert drawn == set(range(1, num_inference_steps + 1))


def test_dmd2_clean_projection_reuses_authoritative_structured_sigmas() -> None:
    trainer = _trainer()
    trainer.training_args.replay_rtol = 0.0
    trainer.training_args.replay_atol = 0.0
    state = LatentState(
        {
            "video": torch.tensor([[1.0]]),
            "audio": torch.tensor([[2.0]]),
        }
    )
    velocity = LatentState(
        {
            "video": torch.tensor([[3.0]]),
            "audio": torch.tensor([[4.0]]),
        }
    )
    stored_times = ComponentTimes(
        timestep={"video": torch.tensor([750.0]), "audio": torch.tensor([600.0])},
        next_timestep={"video": torch.tensor([500.0]), "audio": torch.tensor([300.0])},
        sigma={"video": torch.tensor([0.75]), "audio": torch.tensor([0.6])},
        next_sigma={"video": torch.tensor([0.5]), "audio": torch.tensor([0.3])},
    )
    mapping_calls: list[torch.Tensor] = []
    projected_times: list[ComponentTimes] = []

    def project(
        replay_state: LatentState,
        times: ComponentTimes,
        replay_velocity: LatentState,
    ) -> LatentState:
        assert replay_state is state
        assert replay_velocity is velocity
        projected_times.append(times)
        return state

    trainer.adapter = SimpleNamespace(
        trajectory_component_order=("video", "audio"),
        get_replay_step=lambda batch, index: SimpleNamespace(state=state, times=stored_times),
        use_component_variant=lambda name: nullcontext(),
        replay_generator_boundary=lambda *args, **kwargs: MultiModalStepOutput(velocity=velocity),
        build_training_component_times=lambda timesteps, batch=None: mapping_calls.append(
            timesteps
        ),
        project_velocity_to_clean_state=project,
    )
    trainer._replay_forward_kwargs = lambda batch: {}

    result = trainer._replay_generator_clean_prediction(SimpleNamespace(), boundary_index=1)

    assert result is state
    assert projected_times == [stored_times]
    assert mapping_calls == []


def test_dmd2_rejects_inconsistent_stored_projection_sigma_before_replay() -> None:
    trainer = _trainer()
    state = LatentState({"latent": torch.tensor([[1.0]])})
    stored_times = ComponentTimes(
        timestep={"latent": torch.tensor([750.0])},
        next_timestep={"latent": torch.tensor([500.0])},
        sigma={"latent": torch.tensor([0.5])},
        next_sigma={"latent": torch.tensor([0.25])},
    )
    replay_calls: list[int] = []
    trainer.adapter = SimpleNamespace(
        trajectory_component_order=("latent",),
        get_replay_step=lambda batch, index: SimpleNamespace(state=state, times=stored_times),
        replay_generator_boundary=lambda *args, **kwargs: replay_calls.append(1),
    )

    with pytest.raises(ValueError, match=r"DMD2.*latent.*timestep.*sigma.*one native ULP"):
        trainer._replay_generator_clean_prediction(SimpleNamespace(), boundary_index=1)

    assert replay_calls == []


def test_dmd2_clean_projection_maps_legacy_timestep_only_replay() -> None:
    trainer = _trainer()
    trainer.training_args.replay_rtol = 0.0
    trainer.training_args.replay_atol = 0.0
    state = LatentState({"latent": torch.tensor([[1.0]])})
    velocity = LatentState({"latent": torch.tensor([[2.0]])})
    replay_times = ComponentTimes(
        timestep={"latent": torch.tensor([750.0])},
        next_timestep={"latent": torch.tensor([500.0])},
    )
    mapped_times = ComponentTimes(
        timestep=replay_times.timestep,
        next_timestep=replay_times.next_timestep,
        sigma={"latent": torch.tensor([0.75])},
        next_sigma={"latent": torch.tensor([0.5])},
    )
    mapping_calls: list[torch.Tensor] = []
    projected_times: list[ComponentTimes] = []

    def map_times(
        timesteps: torch.Tensor,
        *,
        batch: Any = None,
    ) -> ComponentTimes:
        del batch
        mapping_calls.append(timesteps)
        return mapped_times

    def project(
        replay_state: LatentState,
        times: ComponentTimes,
        replay_velocity: LatentState,
    ) -> LatentState:
        del replay_state, replay_velocity
        projected_times.append(times)
        return state

    trainer.adapter = SimpleNamespace(
        trajectory_component_order=("latent",),
        get_replay_step=lambda batch, index: SimpleNamespace(state=state, times=replay_times),
        use_component_variant=lambda name: nullcontext(),
        replay_generator_boundary=lambda *args, **kwargs: MultiModalStepOutput(velocity=velocity),
        build_training_component_times=map_times,
        project_velocity_to_clean_state=project,
    )
    trainer._replay_forward_kwargs = lambda batch: {}

    trainer._replay_generator_clean_prediction(SimpleNamespace(), boundary_index=1)

    assert mapping_calls == [replay_times.timestep["latent"]]
    assert projected_times == [mapped_times]


def test_dmd2_boundary_draw_restarts_reproducibly_with_scheduler_seed() -> None:
    """Epoch reseeding resets the deterministic draw sequence on every rank."""
    trainer = _trainer(num_inference_steps=4)

    first = [trainer._draw_boundary_index() for _ in range(8)]
    trainer.adapter.scheduler_group.primary.seed += 1
    second = [trainer._draw_boundary_index() for _ in range(8)]
    trainer.adapter.scheduler_group.primary.seed -= 1
    repeated = [trainer._draw_boundary_index() for _ in range(8)]

    assert first == repeated
    assert first != second


def test_dmd2_rejects_multiple_generator_inner_epochs_per_rollout() -> None:
    trainer = _trainer(num_inner_epochs=2)

    with pytest.raises(
        ValueError,
        match=r"DMD2.*num_inner_epochs=1.*received train.num_inner_epochs=2.*fresh rollout",
    ):
        trainer._validate_generation_schedule()


def test_dmd2_sample_collects_all_rollout_boundaries() -> None:
    trainer = _trainer(num_inference_steps=4)
    calls = []

    def generate_samples(**kwargs: Any) -> List[BaseSample]:
        calls.append(kwargs)
        assert trainer.adapter.decode_latents(torch.zeros(2, 1)) == [None, None]
        return [_sample()]

    def forbidden_decode(latents: torch.Tensor) -> Any:
        del latents
        raise AssertionError("DMD2 invoked the real media decoder")

    trainer.adapter.trajectory_component_order = ("latent",)
    trainer.adapter.decode_latents = forbidden_decode
    trainer.generate_samples = generate_samples

    generated = trainer.sample()
    assert len(generated) == 1
    torch.testing.assert_close(generated[0].all_latents, _sample().all_latents)
    assert trainer.adapter.decode_latents is forbidden_decode
    assert calls == [
        {
            "reward_buffer": None,
            "compute_log_prob": False,
            "trajectory_indices": [0, 1, 2, 3, 4],
        }
    ]


def test_dmd2_media_suppression_accepts_keyword_latents_and_restores_decoder() -> None:
    trainer = _trainer()

    class KeywordDecoderAdapter:
        trajectory_component_order = ("latent",)

        def decode_latents(
            self,
            *,
            latents: torch.Tensor,
            output_type: str = "pt",
        ) -> Any:
            del latents, output_type
            raise AssertionError("DMD2 invoked the real keyword media decoder")

    adapter = KeywordDecoderAdapter()
    trainer.adapter = adapter

    with trainer._without_media_decoding():
        assert adapter.decode_latents(
            latents=torch.zeros(3, 4),
            output_type="pt",
        ) == [None, None, None]

    with pytest.raises(AssertionError, match="real keyword media decoder"):
        adapter.decode_latents(latents=torch.zeros(3, 4))


def test_dmd2_media_suppression_rejects_ambiguous_batch_tensors() -> None:
    trainer = _trainer()

    class MultiComponentDecoderAdapter:
        trajectory_component_order = ("video", "audio")

        def decode_latents(
            self,
            video_latents: torch.Tensor,
            audio_latents: torch.Tensor,
        ) -> Any:
            del video_latents, audio_latents
            raise AssertionError("decoder must remain suppressed")

    adapter = MultiComponentDecoderAdapter()
    trainer.adapter = adapter

    with trainer._without_media_decoding():
        with pytest.raises(
            ValueError,
            match=(
                r"DMD2 media-free decoder adapter='MultiComponentDecoderAdapter'.*"
                r"signature=.*video_latents.*audio_latents.*"
                r"ambiguous batch sizes.*video_latents=2.*audio_latents=3"
            ),
        ):
            adapter.decode_latents(torch.zeros(2, 4), torch.zeros(3, 4))


def test_dmd2_media_suppression_rejects_missing_batch_tensor() -> None:
    trainer = _trainer()

    class KeywordDecoderAdapter:
        trajectory_component_order = ("latent",)

        def decode_latents(self, *, latents: torch.Tensor) -> Any:
            del latents
            raise AssertionError("decoder must remain suppressed")

    adapter = KeywordDecoderAdapter()
    trainer.adapter = adapter

    with trainer._without_media_decoding():
        with pytest.raises(
            TypeError,
            match=(
                r"DMD2 media-free decoder adapter='KeywordDecoderAdapter'.*"
                r"signature=.*latents.*expected tensor or LatentState argument named.*"
                r"received latents=str"
            ),
        ):
            adapter.decode_latents(latents="not-a-tensor")


def test_dmd2_feedback_is_noop_without_reward_access() -> None:
    trainer = _trainer()

    class ForbiddenReward:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"DMD2 accessed forbidden reward surface {name!r}")

    trainer.reward_buffer = ForbiddenReward()
    samples = [_sample()]

    assert trainer.prepare_feedback(samples) is None
    assert samples[0].applicable_rewards == set()


def test_dmd2_rejects_adapter_that_bypasses_media_decode_hook() -> None:
    trainer = _trainer()

    class DirectDecoderAdapter:
        decode_latents = BaseAdapter.decode_latents

    trainer.adapter = DirectDecoderAdapter()

    with pytest.raises(
        ValueError,
        match=r"DMD2 media-free rollout.*DirectDecoderAdapter.*decode_latents.*unsupported",
    ):
        trainer._validate_media_free_rollout()


def test_dmd2_accepts_minimax_h3_when_decode_routes_through_adapter() -> None:
    trainer = _trainer()

    class MiniMaxBypassAdapter:
        __module__ = "flow_factory.models.minimax_h3.adapters"

        def decode_latents(self, *args: Any, **kwargs: Any) -> Any:
            return []

    trainer.adapter = MiniMaxBypassAdapter()

    trainer._validate_media_free_rollout()


def test_dmd2_suppresses_structured_h3_media_through_adapter_owned_shape() -> None:
    trainer = _trainer()

    class MiniMaxAdapter:
        __module__ = "flow_factory.models.minimax_h3.adapters"
        trajectory_component_order = ("video", "audio")

        def decode_latents(self, latents: LatentState, **kwargs: Any) -> Any:
            del latents, kwargs
            raise AssertionError("real H3 decoder must remain suppressed")

        def empty_decoded_media(self, batch_size: int) -> Any:
            return ([None] * batch_size, [None] * batch_size, None)

    adapter = MiniMaxAdapter()
    trainer.adapter = adapter
    state = LatentState(
        {
            "video": torch.zeros(2, 3, 4),
            "audio": torch.zeros(2, 5),
        }
    )

    with trainer._without_media_decoding():
        assert adapter.decode_latents(state, geometry={}) == (
            [None, None],
            [None, None],
            None,
        )


def test_dmd2_accepts_bagel_batched_decode_contract() -> None:
    trainer = _trainer()

    class BagelDecodeAdapter:
        __module__ = "flow_factory.models.bagel.bagel"
        trajectory_component_order = ("latent",)

        def decode_latents(self, latents: torch.Tensor, **kwargs: Any) -> Any:
            del latents, kwargs
            raise AssertionError("real Bagel decoder must remain suppressed")

    trainer.adapter = BagelDecodeAdapter()

    trainer._validate_media_free_rollout()
    with trainer._without_media_decoding():
        assert trainer.adapter.decode_latents(
            torch.zeros(2, 16, 64),
            image_shape=(512, 512),
        ) == [None, None]


def _record_phases(trainer: DMD2Trainer) -> List[Tuple[str, Any]]:
    """Replace both role phases with recorders of (role, replay window)."""
    events: List[Tuple[str, Any]] = []

    def fake_phase(self: DMD2Trainer, replay_units: Any) -> None:
        events.append(("fake", tuple(tuple(unit) for unit in replay_units)))

    def generator_phase(self: DMD2Trainer, replay_units: Any) -> None:
        events.append(("generator", tuple(tuple(unit) for unit in replay_units)))

    trainer._fake_phase = MethodType(fake_phase, trainer)
    trainer._generator_phase = MethodType(generator_phase, trainer)
    return events


def _fresh_rollouts(trainer: DMD2Trainer) -> List[BaseSample]:
    """Make ``sample()`` return a new one-sample rollout batch on every call."""
    drawn: List[BaseSample] = []

    def sample(self: DMD2Trainer) -> List[BaseSample]:
        drawn.append(_sample(float(len(drawn))))
        return [drawn[-1]]

    trainer.sample = MethodType(sample, trainer)
    trainer.sampling_context = nullcontext
    trainer.accelerator.reduce = lambda tensor, reduction: tensor
    trainer.log_data = lambda data, step: None
    return drawn


def test_dmd2_optimize_fits_fake_before_generator_on_one_set() -> None:
    trainer = _trainer(fake_update_frequency=3)
    events = _record_phases(trainer)
    samples = [_sample()]

    trainer.optimize(samples)

    assert events == [("fake", (tuple(samples),)), ("generator", (tuple(samples),))]


def test_dmd2_optimize_can_fit_only_the_fake_score() -> None:
    trainer = _trainer(fake_update_frequency=3)
    events = _record_phases(trainer)
    samples = [_sample()]

    trainer.optimize(samples, update_generator=False)

    assert events == [("fake", (tuple(samples),))]


def test_dmd2_cycle_fits_every_fake_step_on_a_fresh_rollout() -> None:
    trainer = _trainer(fake_update_frequency=3)
    events = _record_phases(trainer)
    drawn = _fresh_rollouts(trainer)

    trainer._run_training_step()

    assert trainer.rollout_sets_per_cycle() == 3
    assert len(drawn) == 3
    # Every fake step sees samples no earlier step has seen; the generator then
    # queries the fake score on the set it has just fit.
    assert events == [
        ("fake", ((drawn[0],),)),
        ("fake", ((drawn[1],),)),
        ("fake", ((drawn[2],),)),
        ("generator", ((drawn[2],),)),
    ]


def test_dmd2_default_config_geometry_draws_one_rollout_set_per_fake_step() -> None:
    config = Arguments.from_dict(
        {
            "train": {
                "trainer_type": "dmd2",
                "num_inference_steps": 1,
                "unique_sample_num_per_epoch": 8,
                "per_device_batch_size": 1,
            },
            "scheduler": {"dynamics_type": "ODE"},
        }
    )
    trainer = _trainer()
    trainer.config = config
    trainer.training_args = config.training_args
    events = _record_phases(trainer)
    drawn = _fresh_rollouts(trainer)

    trainer._run_training_step()

    # Five fake TTUR steps per generator step is DMD2's default cadence, and each
    # of them owns a fresh one-unit rollout set.
    assert [(name, len(window)) for name, window in events] == [("fake", 1)] * 5 + [
        ("generator", 1)
    ]
    assert len(drawn) == 5


def test_dmd2_identity_locks_the_fresh_rollout_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _trainer(fake_update_frequency=3)
    monkeypatch.setattr(
        dmd2_module, "build_default_execution_identity_payload", lambda _trainer: {"training": {}}
    )

    payload = trainer.runtime_execution_identity_payload()

    # A checkpoint from the shared-rollout schedule lacks this entry, so its exact
    # resume fails instead of applying a cursor it never consumed.
    assert payload["dmd2_acquisition"] == {
        "schedule": "fresh_rollout_per_fake_step",
        "rollout_sets_per_cycle": 3,
    }


def test_dmd2_rollout_cursor_counts_every_fresh_rollout_set() -> None:
    trainer = _trainer(fake_update_frequency=5, gradient_accumulation_steps=2)
    trainer.progress = TrainingProgress(rollout_iteration=3)

    # Exact resume must skip all five GAS-sized sets each completed cycle consumed.
    assert _completed_rollout_batch_count(trainer) == 3 * 2 * 5

    single_set = SimpleNamespace(progress=trainer.progress, training_args=trainer.training_args)
    assert _completed_rollout_batch_count(single_set) == 3 * 2


def _cursor_cycle_trainer(progress: TrainingProgress) -> Tuple[DMD2Trainer, List[List[int]]]:
    """Build a DMD2 trainer whose rollouts come from the real exact-resume cursor."""
    trainer = _trainer(fake_update_frequency=3)
    trainer.progress = progress
    trainer.dataloader = [{"index": index} for index in range(7)]
    trainer.adapter.rollout = lambda: None
    trainer._rollout_acceleration = nullcontext
    trainer.sample_batch = lambda batch, **kwargs: [_sample(float(batch["index"]))]
    trainer.sample = MethodType(
        lambda self: self.generate_samples(reward_buffer=None, trajectory_indices=[0, 1]),
        trainer,
    )
    trainer.sampling_context = nullcontext
    trainer.accelerator.reduce = lambda tensor, reduction: tensor
    trainer.log_data = lambda data, step: None
    trainer._stack_replay_unit = MethodType(lambda self, unit: unit, trainer)
    phase_prompts: List[List[int]] = []

    def record(self: DMD2Trainer, replay_units: Any) -> None:
        phase_prompts.append(
            [int(sample.all_latents[0, 0]) for unit in replay_units for sample in unit]
        )

    trainer._fake_phase = MethodType(record, trainer)
    trainer._generator_phase = MethodType(record, trainer)
    return trainer, phase_prompts


def test_dmd2_exact_resume_continues_after_every_consumed_rollout_set() -> None:
    uninterrupted, uninterrupted_prompts = _cursor_cycle_trainer(TrainingProgress())
    uninterrupted._run_training_step()
    uninterrupted.progress = TrainingProgress(rollout_iteration=1)
    uninterrupted._run_training_step()

    resumed, resumed_prompts = _cursor_cycle_trainer(TrainingProgress(rollout_iteration=1))
    resumed._run_training_step()

    # Phases: three fake steps on fresh sets, then the generator on the last set.
    assert uninterrupted_prompts == [[0], [1], [2], [2], [3], [4], [5], [5]]
    assert resumed_prompts == uninterrupted_prompts[4:]


@pytest.mark.parametrize("count", [0, 1.0, True])
def test_rollout_sets_per_cycle_rejects_non_positive_int(count: object) -> None:
    trainer = SimpleNamespace(rollout_sets_per_cycle=lambda: count)

    with pytest.raises(ValueError, match=r"rollout_sets_per_cycle\(\) >= 1 as an int"):
        resolve_rollout_sets_per_cycle(trainer)


def test_dmd2_optimize_rejects_multiple_samples_in_one_flat_microbatch() -> None:
    trainer = _trainer(per_device_batch_size=1)

    with pytest.raises(
        ValueError,
        match=r"DMD2 optimize expected microbatch index 0 to have per_device_batch_size=1 samples, received 2",
    ):
        trainer.optimize([_sample(1.0), _sample(2.0)])


def test_dmd2_optimize_rejects_gas_microbatch_count_mismatch() -> None:
    trainer = _trainer(gradient_accumulation_steps=2)

    with pytest.raises(
        ValueError,
        match=r"DMD2 optimize expected 2 role microbatches.*received 1",
    ):
        trainer.optimize([_sample()])


def test_dmd2_generate_samples_draws_one_fresh_batch_per_outer_iteration() -> None:
    trainer = _trainer()
    trainer.dataloader = [{"prompt": ["p0"]}, {"prompt": ["p1"]}]
    trainer.adapter = SimpleNamespace(train=lambda: None, rollout=lambda: None)
    trainer._rollout_acceleration = nullcontext
    calls = []

    def sample_batch(batch: Any, **kwargs: Any) -> List[BaseSample]:
        calls.append((batch, kwargs))
        return [_sample(float(len(calls)))]

    trainer.sample_batch = sample_batch

    first = trainer.generate_samples(
        reward_buffer=None,
        compute_log_prob=False,
        trajectory_indices=[0, 1],
    )
    second = trainer.generate_samples(
        reward_buffer=None,
        compute_log_prob=False,
        trajectory_indices=[0, 1],
    )
    third = trainer.generate_samples(
        reward_buffer=None,
        compute_log_prob=False,
        trajectory_indices=[0, 1],
    )

    assert len(first) == len(second) == len(third) == 1
    assert [call[0] for call in calls] == [
        {"prompt": ["p0"]},
        {"prompt": ["p1"]},
        {"prompt": ["p0"]},
    ]
    assert all(
        call[1]
        == {
            "reward_buffer": None,
            "compute_log_prob": False,
            "trajectory_indices": [0, 1],
        }
        for call in calls
    )
