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

"""Train a deterministic few-step generator with trajectory distribution matching."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterator, List, Literal, Sequence, Tuple

import torch
from accelerate import Accelerator

from ...contracts.execution import (
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    ExecutionContract,
)
from ...hparams import Arguments, TDMQuerySamplingPolicy, TDMTrainingArguments
from ...hparams.training_args.dmd2 import DMD2_DEFAULT_OPTIMIZERS
from ...models.abc import BaseAdapter
from ...models.trajectory_bridge import resolve_replay_projection_times
from ...rewards import RewardBuffer
from ...samples import (
    BaseSample,
    ComponentTimes,
    LatentState,
    NoisedState,
    StackedSampleBatch,
)
from ..abc import BaseTrainer
from ..common.runtime_identity import build_default_execution_identity_payload
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
    resolve_rollout_accumulation_steps,
    role_repeat_progress,
    run_distillation_training_step,
    run_role_phase,
    validate_media_free_rollout,
    without_media_decoding,
)
from .distribution_matching import (
    tdm_conditional_renoise,
    tdm_fake_loss,
    tdm_generator_loss,
)
from .dmd2 import DMD2Trainer
from .tdm_time_sampling import TDMGenerationProvenance, capture_generation_shift
from .tdm_trajectory import TDMBoundaryUnit, TDMTrajectoryRuntimeMixin


@dataclass(frozen=True)
class TDMGeneratorScoreTerms:
    """Share one live boundary, noised state, and frozen score queries."""

    loss: torch.Tensor
    boundary_state: LatentState
    times: ComponentTimes
    noised: NoisedState
    reference_velocity: LatentState
    fake_velocity: LatentState
    x0_real: LatentState
    x0_fake: LatentState


class TDMTrainer(TDMTrajectoryRuntimeMixin, BaseTrainer):
    """Optimize every boundary of a deterministic few-step generator trajectory."""

    paradigm: ClassVar[Literal["distillation"]] = "distillation"
    execution_contract: ClassVar[ExecutionContract] = ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT

    def runtime_execution_identity_payload(self) -> Dict[str, Any]:
        """Lock only active query semantics and effective static shifts for exact resume.

        Returns:
            Default identity plus TDM sampling semantics. Dynamic shift configuration
            and input geometry are already locked by the shared execution/data schema.
        """
        payload = build_default_execution_identity_payload(self)
        policy = TDMQuerySamplingPolicy.from_training_args(self.training_args)
        for field_name in (
            "tdm_query_interval",
            "tdm_query_distribution",
            "tdm_query_max_sigma",
            "tdm_query_logit_mean",
            "tdm_query_logit_std",
        ):
            payload["training"].pop(field_name, None)
        query_identity = policy.resume_identity()
        if policy.requires_generation_shift:
            query_identity["static_shifts"] = {
                name: (
                    None
                    if getattr(scheduler, "config", {}).get("use_dynamic_shifting", False)
                    else getattr(scheduler, "shift", None)
                )
                for name, scheduler in self.adapter.scheduler_group.items()
            }
        payload["tdm_query_sampling"] = query_identity
        return payload

    def sample_batch(
        self, batch: Dict[str, Any], reward_buffer: RewardBuffer | None = None, **kwargs: Any
    ) -> List[BaseSample]:
        """Collect samples and snapshot the generation shift before another rollout.

        Args:
            batch: Input conditions for one generation batch.
            reward_buffer: Optional TDM-R1 reward buffer.
            **kwargs: Generation options forwarded to the shared sampling pipeline.

        Returns:
            Samples whose rollout-only provenance is registered before reward submission.
        """
        policy = TDMQuerySamplingPolicy.from_training_args(self.training_args)
        if not policy.requires_generation_shift:
            return super().sample_batch(batch, reward_buffer=reward_buffer, **kwargs)
        primary = self.adapter.trajectory_component_order[0]
        with capture_generation_shift(self.adapter.scheduler_group[primary]) as shifts:
            samples = super().sample_batch(batch, reward_buffer=None, **kwargs)
        self._record_tdm_generation_provenance(samples, shifts[0])
        if reward_buffer is not None:
            reward_buffer.add_samples(samples)
        return samples

    def _record_tdm_generation_provenance(
        self,
        samples: Sequence[BaseSample],
        source_shift: float,
    ) -> None:
        """Register one rollout's private source-coordinate provenance."""
        provenance = TDMGenerationProvenance(source_shift=source_shift)
        for sample in samples:
            self._tdm_generation_provenance[id(sample)] = provenance

    def _tdm_generation_provenance_for(self, sample: BaseSample) -> TDMGenerationProvenance:
        """Return private rollout provenance for one unchanged local sample object."""
        provenance = self._tdm_generation_provenance.get(id(sample))
        if provenance is None:
            raise ValueError(
                "source_uniform requires rollout-owned generation provenance; collect "
                "trajectories through TDM.sample_batch() before replay"
            )
        return provenance

    def _clear_tdm_generation_provenance(self) -> None:
        """Drop rollout-private provenance at an acquisition-cycle boundary."""
        self._tdm_generation_provenance = {}

    def _optimizer_args_for_role(self, role_name: str):
        """Resolve this role's optimizer, falling back to TDM's published defaults.

        An ``optimizers`` list in the config file wins, which is also how a run puts
        one of these roles on Muon. Without one, the algorithm supplies the learning
        rates it was published with, because those numbers belong to the algorithm
        rather than to the framework. TDM reuses DMD2's generator and fake-score rates.
        """
        configured = self.config.optimizer_args.get_by_name(role_name)
        if configured is not None:
            return configured
        for default in DMD2_DEFAULT_OPTIMIZERS:
            if default.name == role_name:
                return default
        raise ValueError(
            f"expected an optimizer configuration for role {role_name!r}: no "
            f"`optimizers` entry and no TDM default"
        )

    def __init__(
        self,
        accelerator: Accelerator,
        config: Arguments,
        adapter: BaseAdapter,
    ) -> None:
        super().__init__(accelerator=accelerator, config=config, adapter=adapter)
        self.training_args: TDMTrainingArguments
        self._rollout_data_iter: Iterator[Any] | None = None
        self._rollout_batches_consumed: int | None = None
        self._tdm_generation_provenance: Dict[int, TDMGenerationProvenance] = {}
        self._validate_trajectory_configuration()

    def _init_reward_model(self) -> Tuple[Dict[str, object], Dict[str, object]]:
        """Build the shared feedback runtime, which for this algorithm is eval-only.

        See :meth:`DMD2Trainer._init_reward_model`: the reward-free training contract is
        enforced in `Arguments`, and zeroing the runtime here removed eval monitoring
        along with it.

        Returns:
            Training and eval reward models; the training mapping is always empty.
        """
        return reject_training_rewards(self, algorithm_name="TDM")

    def _run_training_step(self) -> None:
        """Run timestep-aligned trajectory rollouts and one fake/generator phase pair.

        Overriding only this keeps the shared epoch loop, so checkpointing and
        eval-time reward monitoring behave exactly as they do for every other
        trainer.
        """
        self._clear_tdm_generation_provenance()
        try:
            run_distillation_training_step(self)
        finally:
            self._clear_tdm_generation_provenance()

    def sample(self) -> List[BaseSample]:
        """Collect the initial state and every generated ODE boundary."""
        self._validate_trajectory_configuration()
        self._validate_media_free_rollout()
        trajectory_indices = list(range(self.training_args.num_inference_steps + 1))
        with self._without_media_decoding():
            return generate_one_rollout_batch(
                self,
                reward_buffer=None,
                compute_log_prob=False,
                trajectory_indices=trajectory_indices,
                algorithm_name="TDM",
            )

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Perform no feedback work for the data-free objective."""
        del samples

    def optimize(self, samples: Sequence[Any]) -> None:
        """Run fake TTUR updates, then one generator step, over all boundaries."""
        if not samples:
            return
        rollout_accumulation_steps = resolve_rollout_accumulation_steps(
            self.training_args,
        )
        microbatches = as_role_microbatches(
            samples,
            batch_size=self.training_args.per_device_batch_size,
            accumulation_steps=rollout_accumulation_steps,
            algorithm_name="TDM",
        )
        boundary_units = self._flatten_boundary_units(microbatches)
        self.adapter.train()
        for _ in role_repeat_progress(
            self, role_name="fake", repeats=self.training_args.ttur_fake_updates
        ):
            self._fake_phase(boundary_units)
        self._generator_phase(boundary_units)

    def _flatten_boundary_units(
        self,
        microbatches: Sequence[Sequence[BaseSample]],
    ) -> List[TDMBoundaryUnit]:
        """Flatten rollout-major boundaries into backend auto-GAS work items."""
        units = [
            unit for microbatch in microbatches for unit in self._build_boundary_units(microbatch)
        ]
        expected = self.training_args.gradient_accumulation_steps
        if len(units) != expected:
            raise RuntimeError(
                f"TDM expected {expected} boundary work items from "
                f"{len(microbatches)} rollout batches, received {len(units)}"
            )
        return units

    def _fake_phase(self, boundary_units: Sequence[TDMBoundaryUnit]) -> None:
        """Fit the fake score over timestep-aligned boundary work items."""
        run_role_phase(
            self,
            "fake",
            boundary_units,
            self._fake_boundary_loss,
        )

    def _generator_phase(self, boundary_units: Sequence[TDMBoundaryUnit]) -> None:
        """Update the generator over the identical ordered boundary work items."""
        run_role_phase(
            self,
            "generator",
            boundary_units,
            self._generator_boundary_loss,
        )

    def _fake_boundary_loss(self, unit: TDMBoundaryUnit) -> torch.Tensor:
        """Compute official conditionally-renoised fake DSM loss for one stage."""
        with torch.no_grad():
            batch, clean_state, model_noise = self._replay_generator_prediction(unit)
        detached_clean = detach_state(clean_state)
        times = self._sample_score_query_times(unit, batch)
        noised, importance = tdm_conditional_renoise(
            self.adapter,
            detached_clean,
            detach_state(model_noise),
            mid_times=unit.mid_times,
            target_times=times,
            importance_clip=self.training_args.tdm_importance_clip,
        )
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
        velocity = require_velocity(output, algorithm_name="TDM", role_name="fake")
        predicted_clean = self.adapter.project_velocity_to_clean_state(
            noised.state,
            times,
            velocity,
        )
        primary_sigma = times.sigma[self.adapter.trajectory_component_order[0]]
        return tdm_fake_loss(
            self.adapter,
            predicted_clean,
            detached_clean,
            sigma=primary_sigma,
            importance=importance,
            snr_gamma=self.training_args.tdm_snr_gamma,
        )

    def _generator_boundary_loss(self, unit: TDMBoundaryUnit) -> torch.Tensor:
        """Replay one preceding transition with gradient before detached score queries."""
        return self._generator_score_terms(unit).loss

    def _generator_score_terms(self, unit: TDMBoundaryUnit) -> TDMGeneratorScoreTerms:
        """Replay a live clean prediction and query scores on conditional stage noise."""
        batch, clean_state, model_noise = self._replay_generator_prediction(unit)
        times = self._sample_score_query_times(unit, batch)
        noised, _ = tdm_conditional_renoise(
            self.adapter,
            detach_state(clean_state),
            detach_state(model_noise),
            mid_times=unit.mid_times,
            target_times=times,
            importance_clip=self.training_args.tdm_importance_clip,
        )
        reference_velocity = query_score_velocity(
            self.adapter,
            batch,
            noised.state,
            times,
            role_name="reference",
            autocast=self.autocast,
            forward_kwargs=self._reference_forward_kwargs(batch),
            algorithm_name="TDM",
        )
        fake_velocity = query_score_velocity(
            self.adapter,
            batch,
            noised.state,
            times,
            role_name="fake",
            autocast=self.autocast,
            forward_kwargs=self._replay_forward_kwargs(batch),
            algorithm_name="TDM",
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
        record_distillation_metric(self, "train/boundary_index", unit.boundary_index)
        loss = tdm_generator_loss(
            self.adapter,
            clean_state,
            detach_state(x0_real),
            detach_state(x0_fake),
            use_huber=self.training_args.use_huber,
            huber_c=self.training_args.huber_c,
        )
        return TDMGeneratorScoreTerms(
            loss=loss,
            boundary_state=clean_state,
            times=times,
            noised=noised,
            reference_velocity=reference_velocity,
            fake_velocity=fake_velocity,
            # Detached: these are frozen score queries, and every consumer uses them as
            # a target or a scale rather than a path for the generator's gradient.
            x0_real=detach_state(x0_real),
            x0_fake=detach_state(x0_fake),
        )

    def _replay_generator_prediction(
        self,
        unit: TDMBoundaryUnit,
    ) -> Tuple[StackedSampleBatch, LatentState, LatentState]:
        """Replay a stage and return its live clean prediction and implied noise."""
        batch = self._stack_replay_unit(unit.samples)
        replay_step = self.adapter.get_replay_step(batch, unit.boundary_index - 1)
        with self.adapter.use_component_variant("generator"):
            with self.autocast():
                output = self.adapter.replay_generator_boundary(
                    batch,
                    unit.boundary_index,
                    return_fields=("velocity", "next_latents", "next_latents_mean"),
                    rtol=self.training_args.replay_rtol,
                    atol=self.training_args.replay_atol,
                    **self._replay_forward_kwargs(batch),
                )
        velocity = require_velocity(output, algorithm_name="TDM", role_name="generator")
        projection_times = self._normalize_replay_times(replay_step.times, len(unit.samples))
        projection_times = resolve_replay_projection_times(
            self.adapter,
            projection_times,
            batch=batch,
        )
        clean_state = self.adapter.project_velocity_to_clean_state(
            replay_step.state,
            projection_times,
            velocity,
        )
        direction = self.adapter.flow_velocity_direction
        if direction not in ("noise", "data"):
            raise ValueError(
                "TDM expected adapter.flow_velocity_direction in ('noise', 'data'), "
                f"received {direction!r}"
            )
        sign = 1.0 if direction == "noise" else -1.0
        model_noise = LatentState(
            {
                name: (
                    clean_state.components[name].to(torch.float32)
                    + sign * velocity.components[name].to(torch.float32)
                ).detach()
                for name in self.adapter.trajectory_component_order
            },
            active_masks=clean_state.active_masks,
        )
        return batch, clean_state, model_noise

    def _stack_replay_unit(
        self,
        replay_samples: Sequence[BaseSample],
    ) -> StackedSampleBatch:
        """Move and stack one boundary unit without generated media."""
        if not replay_samples:
            raise ValueError("expected a non-empty TDM boundary unit, received no samples")
        return BaseSample.stack([sample.to(self.accelerator.device) for sample in replay_samples])

    def _replay_forward_kwargs(self, batch: StackedSampleBatch) -> Dict[str, object]:
        """Return allow-listed adapter arguments not already owned by the batch."""
        return replay_forward_kwargs(self.training_args, batch)

    def _reference_forward_kwargs(self, batch: StackedSampleBatch) -> Dict[str, object]:
        """Return forward arguments for the real score, which alone may be guided."""
        return reference_forward_kwargs(self.adapter, self.training_args, batch)

    def _validate_media_free_rollout(self) -> None:
        """Require inference to expose a suppressible media reconstruction seam."""
        validate_media_free_rollout(self.adapter, algorithm_name="TDM")

    @contextmanager
    def _without_media_decoding(self) -> Iterator[None]:
        """Replace media reconstruction with shape-preserving empty outputs."""
        with without_media_decoding(self.adapter, algorithm_name="TDM"):
            yield
