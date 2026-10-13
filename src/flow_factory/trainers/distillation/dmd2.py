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

"""Data-free DMD2 distribution-matching trainer for few-step generators."""

from __future__ import annotations

from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import torch
from accelerate import Accelerator

from ...contracts.execution import (
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    ExecutionContract,
)
from ...hparams import Arguments, DMD2TrainingArguments
from ...hparams.optimizer_args import resolve_role_optimizer_args
from ...hparams.training_args.dmd2 import DMD2_DEFAULT_OPTIMIZERS
from ...models.abc import BaseAdapter
from ...models.trajectory_bridge import resolve_replay_projection_times
from ...rewards import BaseRewardModel
from ...samples import BaseSample, ComponentTimes, LatentState, StackedSampleBatch
from ...utils.noise_schedule import TIMESTEP_MAX, validate_flow_match_coordinates
from ..abc import BaseTrainer
from .distillation_runtime import (
    as_role_microbatches,
    detach_state,
    generate_one_rollout_batch,
    query_score_velocity,
    record_distillation_metric,
    record_state_statistics,
    reference_forward_kwargs,
    reject_training_rewards,
    replay_forward_kwargs,
    require_velocity,
    role_repeat_progress,
    run_distillation_training_step,
    run_role_phase,
    validate_media_free_rollout,
    without_media_decoding,
)
from .distribution_matching import dmd_generator_loss, flow_matching_loss

if TYPE_CHECKING:
    from ..rewards import RewardBuffer


class DMD2Trainer(BaseTrainer):
    """DMD2 trainer for a deterministic few-step generator without real training data."""

    paradigm: ClassVar[Literal["distillation"]] = "distillation"
    execution_contract: ClassVar[ExecutionContract] = ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT

    def _optimizer_args_for_role(self, role_name: str):
        """Resolve this role's optimizer, falling back to DMD2's published defaults.

        An ``optimizers`` list in the config file wins, which is also how a run puts
        one of these roles on Muon. Without one, the algorithm supplies the learning
        rates and update cadence it was published with, because those numbers belong
        to the algorithm rather than to the framework.
        """
        return resolve_role_optimizer_args(
            role_name,
            self.config.optimizer_args.get_by_name(role_name),
            DMD2_DEFAULT_OPTIMIZERS,
            algorithm_name="DMD2",
        )

    def __init__(
        self,
        accelerator: Accelerator,
        config: Arguments,
        adapter: BaseAdapter,
    ) -> None:
        super().__init__(accelerator=accelerator, config=config, adapter=adapter)
        self.training_args: DMD2TrainingArguments
        self._validate_generation_schedule()
        self._rollout_data_iter: Optional[Iterator[Any]] = None
        self._rollout_batches_consumed: Optional[int] = None
        non_ode = {
            name: scheduler.dynamics_type
            for name, scheduler in self.adapter.scheduler_group.items()
            if scheduler.dynamics_type != "ODE"
        }
        if non_ode:
            raise ValueError(
                "DMD2 requires deterministic ODE generation for boundary replay; "
                f"received scheduler dynamics {non_ode!r}"
            )

    def _validate_generation_schedule(self) -> None:
        """Reject generation schedules the boundary objective cannot be defined on."""
        num_inference_steps = self.training_args.num_inference_steps
        if (
            not isinstance(num_inference_steps, int)
            or isinstance(num_inference_steps, bool)
            or num_inference_steps < 1
        ):
            raise ValueError(
                "expected train.num_inference_steps as an int >= 1 for the DMD2 generator, "
                f"received {num_inference_steps!r}"
            )
        if self.training_args.num_inner_epochs != 1:
            raise ValueError(
                "DMD2 requires train.num_inner_epochs=1 so every generator update owns a "
                "fresh rollout boundary; "
                f"received train.num_inner_epochs={self.training_args.num_inner_epochs}. "
                "Start a fresh rollout before another generator update."
            )

    def _init_reward_model(self) -> Tuple[Dict[str, BaseRewardModel], Dict[str, BaseRewardModel]]:
        """Build the shared feedback runtime, which for this algorithm is eval-only.

        The reward-free training contract is enforced where it belongs: `Arguments`
        rejects any `rewards` entry for this trainer, so the training side of the shared
        implementation builds empty on its own. Zeroing the whole runtime here as well
        would take eval reward monitoring down with it.

        Returns:
            Training and eval reward models; the training mapping is always empty.
        """
        return reject_training_rewards(self, algorithm_name="DMD2")

    def _run_training_step(self) -> None:
        """Run GAS distinct rollouts, then fake TTUR updates, then one generator step.

        Overriding only this keeps the shared epoch loop, so checkpointing and
        eval-time reward monitoring behave exactly as they do for every other
        trainer.
        """
        run_distillation_training_step(self)

    def sample(self) -> List[BaseSample]:
        """Collect every boundary of one fresh rollout."""
        self._validate_generation_schedule()
        self._validate_media_free_rollout()
        with self._without_media_decoding():
            return self.generate_samples(
                reward_buffer=None,
                compute_log_prob=False,
                # Every boundary is kept, because the objective matches distributions at
                # one step drawn per replay unit rather than at a fixed terminal one.
                trajectory_indices=list(range(int(self.training_args.num_inference_steps) + 1)),
            )

    def generate_samples(
        self,
        reward_buffer: Optional[RewardBuffer] = None,
        compute_log_prob: bool = False,
        trajectory_indices: Optional[List[int]] = None,
        **extra_inference_kwargs: Any,
    ) -> List[BaseSample]:
        """Generate exactly one fresh rollout batch for one outer iteration.

        Unlike the epoch-sized base implementation, DMD2 consumes one
        dataloader batch per outer iteration; the iterator rolls over with a
        reseeded epoch only after a full pass through the prompt dataloader.
        """
        del extra_inference_kwargs
        if reward_buffer is not None:
            raise ValueError(
                "DMD2 is data-free and computes no rewards; expected "
                f"reward_buffer=None, received {type(reward_buffer).__name__}"
            )
        return generate_one_rollout_batch(
            self,
            reward_buffer=None,
            compute_log_prob=compute_log_prob,
            trajectory_indices=trajectory_indices,
            algorithm_name="DMD2",
        )

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Perform no feedback work for the data-free objective."""
        del samples

    def optimize(self, samples: Sequence[Any]) -> None:
        """Run fake TTUR updates, then one generator step, over GAS replay units."""
        if not samples:
            return
        self._validate_generation_schedule()
        replay_units = as_role_microbatches(
            samples,
            batch_size=self.training_args.per_device_batch_size,
            accumulation_steps=self.training_args.gradient_accumulation_steps,
            algorithm_name="DMD2",
        )
        self.adapter.train()
        for _ in role_repeat_progress(
            self, role_name="fake", repeats=self._role_update_repeats("fake")
        ):
            self._fake_phase(replay_units)
        self._generator_phase(replay_units)

    def _fake_phase(self, replay_units: Sequence[Sequence[BaseSample]]) -> None:
        """Fit the fake score on fresh perturbations of generated boundaries."""
        run_role_phase(
            self,
            "fake",
            replay_units,
            lambda unit: self._fake_replay_loss(self._stack_replay_unit(list(unit))),
        )

    def _generator_phase(self, replay_units: Sequence[Sequence[BaseSample]]) -> None:
        """Update the generator from detached reference/fake score differences."""
        run_role_phase(
            self,
            "generator",
            replay_units,
            lambda unit: self._generator_replay_loss(self._stack_replay_unit(list(unit))),
        )

    def _stack_replay_unit(self, replay_unit: Sequence[BaseSample]) -> StackedSampleBatch:
        """Move and stack one replay unit without touching generated media."""
        if not replay_unit:
            raise ValueError("expected a non-empty DMD2 replay unit, received no samples")
        return BaseSample.stack([sample.to(self.accelerator.device) for sample in replay_unit])

    def _validate_media_free_rollout(self) -> None:
        """Require inference to expose the adapter media reconstruction seam."""
        validate_media_free_rollout(self.adapter, algorithm_name="DMD2")

    @contextmanager
    def _without_media_decoding(self) -> Iterator[None]:
        """Replace rollout media reconstruction with shape-preserving empty outputs."""
        with without_media_decoding(self.adapter, algorithm_name="DMD2"):
            yield

    def _draw_boundary_index(self) -> int:
        """Pick a reproducible rank-shared rollout boundary.

        A multi-step generator is supervised at one step drawn uniformly from its
        schedule, so over training every step is covered. The scheduler seed is reset
        from ``training_args.seed + epoch`` by the shared loop; a local draw counter
        then gives each replay unit a deterministic sample without touching Python's
        global RNG. All ranks execute the same role plan and therefore draw the same
        boundary.

        Returns:
            Boundary index in ``[1, num_inference_steps]``.
        """
        scheduler_seed = int(self.adapter.scheduler_group.primary.seed)
        if getattr(self, "_boundary_draw_seed", None) != scheduler_seed:
            self._boundary_draw_seed = scheduler_seed
            self._boundary_draw_count = 0
        step_index = self.adapter.scheduler_group.sample_ode_step_index(self._boundary_draw_count)
        self._boundary_draw_count += 1
        num_steps = int(self.training_args.num_inference_steps)
        if step_index >= num_steps:
            raise ValueError(
                "scheduler selected ODE replay step outside the configured generator "
                f"schedule: step_index={step_index}, num_inference_steps={num_steps}, "
                f"scheduler_seed={scheduler_seed}, draw_index={self._boundary_draw_count - 1}"
            )
        return step_index + 1

    def _fake_replay_loss(self, batch: StackedSampleBatch) -> torch.Tensor:
        """Compute fake clean-state denoising loss for one replay unit."""
        boundary_index = self._draw_boundary_index()
        # DMD matches a clean prediction, not the still-noisy x_{i+1} boundary.
        # Recompute v_i and project (x_i, t_i, v_i) into clean space for every
        # selected denoising step; only the terminal transition has x_{i+1} == x0.
        with torch.no_grad():
            clean_state = detach_state(
                self._replay_generator_clean_prediction(batch, boundary_index)
            )
        times = self._sample_perturbation_times(clean_state, batch)
        noised = self.adapter.add_forward_process_noise(clean_state, times)
        with self.adapter.use_component_variant("fake"):
            with self.autocast():
                output = self.adapter.forward_state(
                    batch=batch,
                    state=noised.state,
                    times=times,
                    compute_log_prob=False,
                    return_fields=("velocity",),
                    **self._replay_forward_kwargs(batch),
                )
        velocity = require_velocity(output, algorithm_name="DMD2", role_name="fake")
        return flow_matching_loss(
            self.adapter,
            velocity,
            noised.target_velocity,
            state=noised.state,
        )

    def _generator_replay_loss(self, batch: StackedSampleBatch) -> torch.Tensor:
        """Compute the generator pseudo-loss from one step's live clean prediction."""
        boundary_index = self._draw_boundary_index()
        clean_state = self._replay_generator_clean_prediction(batch, boundary_index)
        detached_clean = detach_state(clean_state)
        times = self._sample_perturbation_times(detached_clean, batch)
        noised = self.adapter.add_forward_process_noise(detached_clean, times)
        reference_velocity = query_score_velocity(
            self.adapter,
            batch,
            noised.state,
            times,
            role_name="reference",
            autocast=self.autocast,
            forward_kwargs=self._reference_forward_kwargs(batch),
            algorithm_name="DMD2",
        )
        fake_velocity = query_score_velocity(
            self.adapter,
            batch,
            noised.state,
            times,
            role_name="fake",
            autocast=self.autocast,
            forward_kwargs=self._replay_forward_kwargs(batch),
            algorithm_name="DMD2",
        )
        x0_real = self.adapter.project_velocity_to_clean_state(
            noised.state,
            times,
            reference_velocity,
        )
        x0_fake = self.adapter.project_velocity_to_clean_state(
            noised.state,
            times,
            fake_velocity,
        )
        record_state_statistics(self, "train/x0_gen", clean_state)
        record_state_statistics(self, "train/x0_real", x0_real)
        record_state_statistics(self, "train/x0_fake", x0_fake)
        record_distillation_metric(self, "train/boundary_index", boundary_index)
        return dmd_generator_loss(
            self.adapter,
            clean_state,
            detach_state(x0_real),
            detach_state(x0_fake),
        )

    def _replay_generator_clean_prediction(
        self,
        batch: StackedSampleBatch,
        boundary_index: int,
    ) -> LatentState:
        """Replay one ODE step and project its live velocity to clean x0.

        ``boundary_index`` names ``x_i -> x_{i+1}``, while DMD's generated sample
        is ``x0_hat_i = project(x_i, t_i, v_i)``. The transition replay remains
        valuable: it verifies the live generator still lands exactly on the stored
        trajectory before its velocity is used to form the clean prediction.
        """
        replay_step = self.adapter.get_replay_step(batch, boundary_index - 1)
        if replay_step.times.sigma is not None:
            expected_order = self.adapter.trajectory_component_order
            if (
                tuple(replay_step.times.timestep) != expected_order
                or tuple(replay_step.times.sigma) != expected_order
            ):
                raise ValueError(
                    "DMD2 stored projection coordinates expected component order "
                    f"{expected_order}, received timestep={tuple(replay_step.times.timestep)} "
                    f"and sigma={tuple(replay_step.times.sigma)}"
                )
            for name in expected_order:
                validate_flow_match_coordinates(
                    replay_step.times.timestep[name],
                    replay_step.times.sigma[name],
                    identifier=f"DMD2 stored projection coordinates for component {name!r}",
                )
        with self.adapter.use_component_variant("generator"):
            with self.autocast():
                output = self.adapter.replay_generator_boundary(
                    batch,
                    boundary_index,
                    return_fields=("velocity", "next_latents", "next_latents_mean"),
                    rtol=self.training_args.replay_rtol,
                    atol=self.training_args.replay_atol,
                    **self._replay_forward_kwargs(batch),
                )
        velocity = require_velocity(output, algorithm_name="DMD2", role_name="generator")
        projection_times = resolve_replay_projection_times(
            self.adapter,
            replay_step.times,
            batch=batch,
        )
        return self.adapter.project_velocity_to_clean_state(
            replay_step.state,
            projection_times,
            velocity,
        )

    def _sample_perturbation_times(
        self,
        boundary_state: LatentState,
        batch: StackedSampleBatch,
    ) -> ComponentTimes:
        """Draw one fresh forward-process coordinate per sample."""
        primary_name = boundary_state.component_names[0]
        component = boundary_state.components[primary_name]
        lower, upper = self.training_args.perturbation_timestep_range
        sigma = torch.rand(
            component.shape[0],
            device=component.device,
            dtype=torch.float32,
        )
        sigma = lower + (upper - lower) * sigma
        primary_timesteps = sigma * TIMESTEP_MAX
        return self.adapter.build_training_component_times(primary_timesteps, batch=batch)

    def _replay_forward_kwargs(self, batch: StackedSampleBatch) -> Dict[str, object]:
        """Return allow-listed adapter arguments not already owned by the batch."""
        return replay_forward_kwargs(self.training_args, batch)

    def _reference_forward_kwargs(self, batch: StackedSampleBatch) -> Dict[str, object]:
        """Return forward arguments for the real score, which alone may be guided."""
        return reference_forward_kwargs(self.adapter, self.training_args, batch)
