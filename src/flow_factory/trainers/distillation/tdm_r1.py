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

"""Train TDM with a learned surrogate reward and frozen reference."""

from __future__ import annotations

import time
from contextlib import ExitStack
from numbers import Real
from typing import Any, ClassVar, Dict, List, Literal, Optional, Sequence

import torch
from accelerate import Accelerator

from ...contracts.execution import ONLINE_EXECUTION_CONTRACT, ExecutionContract
from ...contracts.reward_overlap import WHOLE_GROUP_BATCH_REWARD_OPTIMIZATION_OVERLAP
from ...contracts.sampler import get_sampler_layout_contract
from ...hparams import Arguments, TDMR1TrainingArguments
from ...hparams.training_args.tdm_r1 import TDM_R1_DEFAULT_OPTIMIZERS
from ...models.abc import BaseAdapter
from ...rewards import RewardTile, RewardTileGeometry, RewardTilePlan
from ...samples import AcquisitionManifest, BaseSample, LatentState, group_identity_rows
from ..abc import BaseTrainer
from .distillation_runtime import (
    as_role_microbatches,
    detach_state,
    generate_one_rollout_batch,
    pop_distillation_metrics,
    query_score_velocity,
    record_distillation_metric,
    require_velocity,
    resolve_rollout_accumulation_steps,
    role_repeat_progress,
    run_role_phase,
)
from .distribution_matching import revised_x0_loss
from .group_preference import GroupPreferenceBatch, group_preference_loss
from .tdm import TDMBoundaryUnit, TDMTrainer

SLOW_SURROGATE_SNAPSHOT = "tdm_r1_slow_surrogate"
"""Name of the lagging surrogate copy the PPO trust region is measured against."""


class TDMR1Trainer(TDMTrainer):
    """Reinforce deterministic TDM trajectories through a frozen-reference surrogate."""

    paradigm: ClassVar[Literal["decoupled"]] = "decoupled"
    execution_contract: ClassVar[ExecutionContract] = ONLINE_EXECUTION_CONTRACT
    reward_optimization_overlap_contract = WHOLE_GROUP_BATCH_REWARD_OPTIMIZATION_OVERLAP

    @classmethod
    def reward_optimization_overlap_geometry(cls, config: Arguments) -> RewardTileGeometry:
        """Stream one rollout microbatch while accumulating one whole role phase."""
        sampler_layout = get_sampler_layout_contract(config.data_args.sampler_type)
        layout = {
            "rank_local": "rank_local",
            "global_batch": "cross_rank_sharded",
        }[sampler_layout.group_placement]
        return RewardTileGeometry(
            group_layout=layout,
            optimizer_terms_per_batch=config.training_args.get_num_train_timesteps(config),
            accumulation_scope="acquisition",
        )

    def _optimizer_args_for_role(self, role_name: str):
        """Resolve this role's optimizer, falling back to TDM-R1's published defaults.

        An ``optimizers`` list in the config file wins, which is also how a run puts
        one of these roles on Muon. Without one, the algorithm supplies the learning
        rates it was published with, because those numbers belong to the algorithm
        rather than to the framework.
        """
        configured = self.config.optimizer_args.get_by_name(role_name)
        if configured is not None:
            return configured
        for default in TDM_R1_DEFAULT_OPTIMIZERS:
            if default.name == role_name:
                return default
        raise ValueError(
            f"expected an optimizer configuration for role {role_name!r}: no "
            f"`optimizers` entry and no TDM-R1 default"
        )

    def __init__(
        self,
        accelerator: Accelerator,
        config: Arguments,
        adapter: BaseAdapter,
    ) -> None:
        super().__init__(accelerator=accelerator, config=config, adapter=adapter)
        self.training_args: TDMR1TrainingArguments

    def _initialize_snapshots(self) -> None:
        """Declare the slow surrogate before exact-state compatibility preflight."""
        super()._initialize_snapshots()
        self.adapter.declare_variant_snapshot("surrogate", SLOW_SURROGATE_SNAPSHOT)

    def _init_reward_model(self):
        """Use train-time rewards instead of TDM's reward-free runtime."""
        return BaseTrainer._init_reward_model(self)

    def sample(self) -> List[BaseSample]:
        """Generate complete reward groups with every deterministic boundary stored."""
        self._validate_trajectory_configuration()
        trajectory_indices = list(range(self.training_args.num_inference_steps + 1))
        return generate_one_rollout_batch(
            self,
            reward_buffer=self.reward_buffer,
            compute_log_prob=False,
            trajectory_indices=trajectory_indices,
            algorithm_name="TDM-R1",
        )

    def _run_training_step(self) -> None:
        """Collect role microbatches, then stream reward-ready surrogate work."""
        if not getattr(self.training_args, "reward_optimization_overlap", False):
            super()._run_training_step()
            return

        self._clear_tdm_generation_provenance()
        try:
            self._run_overlap_training_step()
        finally:
            self._clear_tdm_generation_provenance()

    def _run_overlap_training_step(self) -> None:
        """Run one TDM-R1 acquisition with reward-ready surrogate streaming."""

        cycle_started = time.monotonic()
        rollout_started = cycle_started
        rollout_steps = resolve_rollout_accumulation_steps(self.training_args)
        self.reward_buffer.clear()
        microbatches: List[List[BaseSample]] = []
        for _ in range(rollout_steps):
            with self.sampling_context():
                microbatches.append(self.sample())
        samples = [sample for microbatch in microbatches for sample in microbatch]
        self._reward_overlap_acquisition_manifest = AcquisitionManifest.from_samples(
            samples,
            rollout_batch_object_ids=(
                tuple(id(sample) for sample in microbatch) for microbatch in microbatches
            ),
        )
        self._run_reward_optimization_overlap(
            samples,
            cycle_started=cycle_started,
            rollout_seconds=time.monotonic() - rollout_started,
        )

    def _prepare_reward_optimization_overlap(
        self,
        samples: List[BaseSample],
        plan: RewardTilePlan,
    ) -> Dict[str, Any]:
        """Run reward-independent fake TTUR and open one streamed surrogate phase."""
        batch_size = self.training_args.per_device_batch_size
        rollout_steps = resolve_rollout_accumulation_steps(self.training_args)
        microbatches = [
            samples[start : start + batch_size] for start in range(0, len(samples), batch_size)
        ]
        microbatches = as_role_microbatches(
            microbatches,
            batch_size=batch_size,
            accumulation_steps=rollout_steps,
            algorithm_name="TDM-R1",
        )
        units_by_tile = {
            tile.tile_id: list(self._build_boundary_units(microbatches[tile.tile_id]))
            for tile in plan.tiles
        }
        boundary_units = [unit for tile in plan.tiles for unit in units_by_tile[tile.tile_id]]
        expected_units = self.training_args.gradient_accumulation_steps
        if len(boundary_units) != expected_units:
            raise RuntimeError(
                "TDM-R1 overlap expected one acquisition-wide surrogate accumulation "
                f"window with {expected_units} boundary units, received {len(boundary_units)}"
            )

        self.adapter.train()
        for _ in role_repeat_progress(
            self,
            role_name="fake",
            repeats=self.training_args.ttur_fake_updates,
        ):
            self._fake_phase(boundary_units)

        self._ensure_slow_surrogate()
        stack = ExitStack()
        try:
            stack.enter_context(self.role_optimization.phase("surrogate"))
            stack.enter_context(self.adapter.use_component_variant("surrogate"))
        except Exception as error:
            stack.__exit__(type(error), error, error.__traceback__)
            raise
        return {
            "stack": stack,
            "closed": False,
            "units_by_tile": units_by_tile,
            "boundary_units": boundary_units,
        }

    def _optimize_reward_overlap_tile(
        self,
        tile: RewardTile,
        samples: List[BaseSample],
        context: Dict[str, Any],
    ) -> None:
        """Accumulate one reward-ready rollout's surrogate boundary losses."""
        del samples
        group_infos = ()
        if not get_sampler_layout_contract(
            self.config.data_args.sampler_type
        ).groups_are_rank_local:
            group_infos = self._reward_overlap_group_infos_for_tile(tile.tile_id)
            if len(group_infos) != 1:
                raise RuntimeError(
                    "TDM-R1 reward overlap expected exactly one cached group mapping "
                    f"for tile_id={tile.tile_id}, received {len(group_infos)}"
                )
            self._tdm_r1_reward_overlap_group_info = group_infos[0]
        try:
            for unit in context["units_by_tile"][tile.tile_id]:
                with self.role_optimization.microbatch():
                    loss = self._surrogate_boundary_loss(unit)
                    record_distillation_metric(self, "train/surrogate_loss", loss)
                    with self.adapter.use_component_variant("surrogate"):
                        self.role_optimization.backward(loss)
                    self._finish_role_microbatch()
        finally:
            if group_infos:
                del self._tdm_r1_reward_overlap_group_info

    def _finalize_reward_optimization_overlap(
        self,
        context: Dict[str, Any],
        samples: List[BaseSample],
    ) -> Dict[str, Any]:
        """Step the surrogate, advance its snapshot, then update the generator."""
        del samples
        context["closed"] = True
        context["stack"].close()
        grad_norm = self.role_optimization.roles["surrogate"].last_grad_norm
        if grad_norm is not None:
            record_distillation_metric(self, "train/surrogate_grad_norm", grad_norm)
        decay = min(
            self.training_args.surrogate_slow_decay_max,
            self.training_args.surrogate_slow_decay_min + 0.001 * self.step,
        )
        self.adapter.update_variant_snapshot(SLOW_SURROGATE_SNAPSHOT, decay)
        record_distillation_metric(self, "train/surrogate_slow_decay", decay)
        self._generator_phase(context["boundary_units"])
        return pop_distillation_metrics(self)

    def _abort_reward_optimization_overlap(self, context: Any) -> None:
        """Unwind an open surrogate role without demanding a completed step."""
        if not isinstance(context, dict) or context.get("closed", True):
            return
        context["closed"] = True
        error = RuntimeError("aborting incomplete TDM-R1 reward overlap phase")
        context["stack"].__exit__(type(error), error, error.__traceback__)

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize endpoint rewards and store group-normalized advantages."""
        rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
        self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=True,
            aggregation_func=self.training_args.advantage_aggregation,
        )
        advantage_metrics = self.advantage_processor.pop_advantage_metrics()
        if advantage_metrics:
            self.log_data(advantage_metrics, step=self.step)

    def optimize(self, samples: Sequence[Any]) -> None:
        """Run fake TTUR, one surrogate step, then one generator step."""
        if not samples:
            return
        rollout_accumulation_steps = resolve_rollout_accumulation_steps(
            self.training_args,
        )
        microbatches = as_role_microbatches(
            samples,
            batch_size=self.training_args.per_device_batch_size,
            accumulation_steps=rollout_accumulation_steps,
            algorithm_name="TDM-R1",
        )
        boundary_units = self._flatten_boundary_units(microbatches)
        self.adapter.train()
        for _ in role_repeat_progress(
            self, role_name="fake", repeats=self.training_args.ttur_fake_updates
        ):
            self._fake_phase(boundary_units)
        self._surrogate_phase(boundary_units)
        self._generator_phase(boundary_units)

    def _surrogate_phase(self, boundary_units: Sequence[TDMBoundaryUnit]) -> None:
        """Update the surrogate with group preference on endpoint advantages."""
        self._ensure_slow_surrogate()
        run_role_phase(
            self,
            "surrogate",
            boundary_units,
            self._surrogate_boundary_loss,
        )
        # Increase snapshot lag gradually so the trust-region clip strengthens over time.
        decay = min(
            self.training_args.surrogate_slow_decay_max,
            self.training_args.surrogate_slow_decay_min + 0.001 * self.step,
        )
        self.adapter.update_variant_snapshot(SLOW_SURROGATE_SNAPSHOT, decay)
        record_distillation_metric(self, "train/surrogate_slow_decay", decay)

    def _ensure_slow_surrogate(self) -> None:
        """Retain lazy compatibility for lightweight, non-constructor test hosts."""
        if not self.adapter.has_variant_snapshot(SLOW_SURROGATE_SNAPSHOT):
            self.adapter.declare_variant_snapshot("surrogate", SLOW_SURROGATE_SNAPSHOT)

    def _surrogate_boundary_loss(self, unit: TDMBoundaryUnit) -> torch.Tensor:
        """Score one stored boundary with the surrogate and apply group preference."""
        # Share one perturbation so the ratio measures parameter drift only.
        context = self._boundary_score_context(unit)
        trainable_values, reference_values = self._boundary_preference_values(
            unit,
            trainable_role="surrogate",
            context=context,
        )
        preference_batch = self._group_preference_batch(unit, trainable_values)
        # Track density drift that the preference loss alone cannot distinguish.
        record_distillation_metric(
            self,
            "train/surrogate_value_delta",
            (trainable_values - reference_values).mean(),
        )
        clipped_values = self._clip_outside_trust_region(
            unit, trainable_values, preference_batch, context=context
        )
        preference_loss = group_preference_loss(
            self.accelerator,
            preference_batch,
            self._clip_runaway_surrogate(clipped_values, reference_values, preference_batch),
            reference_values,
            self.training_args.surrogate_preference_beta,
        )
        # Anchor the surrogate so globally inflated scores cannot mimic progress.
        reference_penalty = (trainable_values - reference_values).square().mean()
        record_distillation_metric(self, "train/surrogate_reference_penalty", reference_penalty)
        loss = preference_loss + self.training_args.surrogate_reference_beta * reference_penalty
        return self._time_weight(unit) * loss

    def _clip_outside_trust_region(
        self,
        unit: TDMBoundaryUnit,
        trainable_values: torch.Tensor,
        preference_batch: GroupPreferenceBatch,
        *,
        context: Optional[tuple[Any, Any, Any]] = None,
    ) -> torch.Tensor:
        """Freeze samples that have moved too far since the slow copy was taken.

        This is the PPO trust region, measured on the same quantity the preference loss
        uses: the ratio ``exp(old_dsm - dsm)`` is above one exactly when the surrogate
        now denoises the sample better than its slow copy did. A sample that has already
        moved further than ``surrogate_clip_range`` in the direction its advantage asks
        for keeps the update it earned but stops accumulating more, which is what bounds
        how far one batch of noisy rewards can drag the surrogate.
        """
        clip_range = self.training_args.surrogate_clip_range
        if clip_range == 0.0:
            return trainable_values
        with self.adapter.use_variant_snapshot(SLOW_SURROGATE_SNAPSHOT):
            with torch.no_grad():
                slow_values, _ = self._boundary_preference_values(
                    unit,
                    trainable_role="surrogate",
                    context=context,
                )
        ratio = torch.exp(slow_values.detach() - trainable_values.detach())
        clipped = torch.where(
            preference_batch.advantages > 0,
            ratio > 1.0 + clip_range,
            ratio < 1.0 - clip_range,
        )
        record_distillation_metric(self, "train/surrogate_trust_clip_ratio", clipped.float().mean())
        return torch.where(clipped, trainable_values.detach(), trainable_values)

    def _clip_runaway_surrogate(
        self,
        trainable_values: torch.Tensor,
        reference_values: torch.Tensor,
        preference_batch: GroupPreferenceBatch,
    ) -> torch.Tensor:
        """Stop the gradient on samples already moved far enough past the reference.

        A sample whose surrogate density has improved in the direction its advantage
        asks for, while sitting further from the frozen reference than the threshold
        allows, has nothing left to gain from another step and everything to lose in
        drift. Detaching it leaves the rest of the batch training normally.
        """
        threshold = self.training_args.surrogate_reference_threshold
        deviation = (trainable_values - reference_values).square()
        improved = torch.where(
            preference_batch.advantages > 0,
            trainable_values < reference_values,
            trainable_values > reference_values,
        )
        clipped = improved & (deviation > threshold)
        record_distillation_metric(self, "train/surrogate_clip_ratio", clipped.float().mean())
        return torch.where(clipped, trainable_values.detach(), trainable_values)

    def _time_weight(self, unit: TDMBoundaryUnit) -> float:
        """Weight a boundary by how far along the trajectory it sits.

        Later boundaries are closer to the image the reward actually scored, so TDM-R1
        ramps their contribution linearly rather than treating every step alike.
        """
        if not self.training_args.use_time_weighting:
            return 1.0
        return (unit.boundary_index + 1) / self.training_args.num_inference_steps

    def _generator_boundary_loss(self, unit: TDMBoundaryUnit) -> torch.Tensor:
        """Combine the distribution-matching term with two rewards, as TDM-R1 defines it.

        The generator follows three detached directions in clean-prediction space: the
        teacher against the fake score, which is the TDM reverse-KL term and carries no
        coefficient; the teacher's guidance direction, treated as a reward; and the
        surrogate's guided direction against its frozen reference, which is the learned
        reward. ``tdm_weight`` mixes the two rewards convexly, so raising it trades
        learned reward for guidance and never weakens the distribution-matching anchor.
        """
        terms = self._generator_score_terms(unit)
        batch = self._stack_replay_unit(unit.samples)
        tdm_weight = self.training_args.tdm_weight

        guidance_direction = self._guidance_reward_direction(batch, terms)
        if guidance_direction is None:
            guidance_loss = torch.zeros((), device=self.accelerator.device)
        else:
            guidance_loss = revised_x0_loss(
                self.adapter,
                terms.boundary_state,
                guidance_direction,
                terms.x0_real,
                use_huber=self.training_args.use_huber,
                huber_c=self.training_args.huber_c,
            )
        surrogate_direction, surrogate_target = self._surrogate_reward_direction(batch, terms)
        surrogate_loss = revised_x0_loss(
            self.adapter,
            terms.boundary_state,
            surrogate_direction,
            surrogate_target,
            use_huber=self.training_args.use_huber,
            huber_c=self.training_args.huber_c,
        )

        record_distillation_metric(self, "train/generator_tdm_loss", terms.loss)
        record_distillation_metric(self, "train/generator_guidance_reward_loss", guidance_loss)
        record_distillation_metric(self, "train/generator_surrogate_reward_loss", surrogate_loss)
        return terms.loss + tdm_weight * guidance_loss + (1.0 - tdm_weight) * surrogate_loss

    def _guidance_reward_direction(self, batch: Any, terms: Any) -> Optional[LatentState]:
        """Recover the teacher's conditional-minus-unconditional direction.

        The adapter combines guidance internally and returns only the guided velocity,
        so the two ends are recovered from two queries rather than by reaching into it:
        a query at scale 1 is the conditional prediction, and the guided query at scale
        ``s`` is ``uncond + s * (cond - uncond)``, which leaves
        ``cond - uncond = (guided - cond) / (s - 1)``.

        Returns:
            The direction, or ``None`` when this run has no guidance to reward.
        """
        scale = float(self.training_args.get_reference_guidance_scale())
        # A CFG-free reference has no conditional-minus-unconditional direction to
        # follow, so this reward simply does not exist for that run rather than being a
        # misconfiguration: the generator then trains on the distribution match and the
        # learned reward alone.
        if scale <= 1.0 or self.training_args.cfg_reward_scale == 0.0:
            return None
        conditional_velocity = query_score_velocity(
            self.adapter,
            batch,
            terms.noised.state,
            terms.times,
            role_name="reference",
            autocast=self.autocast,
            forward_kwargs=self._replay_forward_kwargs(batch),
            algorithm_name="TDM-R1",
        )
        x0_conditional = self.adapter.project_velocity_to_clean_state(
            terms.noised.state,
            terms.times,
            conditional_velocity,
        )
        reward_scale = self.training_args.cfg_reward_scale / (scale - 1.0)
        return LatentState(
            {
                name: (
                    reward_scale
                    * (
                        terms.x0_real.components[name].to(torch.float32)
                        - x0_conditional.components[name].to(torch.float32)
                    )
                ).detach()
                for name in self.adapter.trajectory_component_order
            },
            active_masks=terms.boundary_state.active_masks,
        )

    def _surrogate_reward_direction(
        self, batch: Any, terms: Any
    ) -> tuple[LatentState, LatentState]:
        """Return the surrogate's guided direction against its frozen reference.

        The surrogate is queried with guidance so its learned preference is amplified the
        same way the teacher's is, and the frozen reference subtracts off what the base
        model would have predicted, leaving only what the surrogate learned.
        """
        surrogate_velocity = query_score_velocity(
            self.adapter,
            batch,
            terms.noised.state,
            terms.times,
            role_name="surrogate",
            autocast=self.autocast,
            forward_kwargs=self._reference_forward_kwargs(batch),
            algorithm_name="TDM-R1",
        )
        x0_surrogate = self.adapter.project_velocity_to_clean_state(
            terms.noised.state,
            terms.times,
            surrogate_velocity,
        )
        component_names = self.adapter.trajectory_component_order
        direction = LatentState(
            {
                name: (
                    x0_surrogate.components[name].to(torch.float32)
                    - terms.x0_real.components[name].to(torch.float32)
                ).detach()
                for name in component_names
            },
            active_masks=terms.boundary_state.active_masks,
        )
        normalizer_reference = LatentState(
            {name: x0_surrogate.components[name].detach() for name in component_names},
            active_masks=terms.boundary_state.active_masks,
        )
        return direction, normalizer_reference

    def _live_generator_preference_values(
        self,
        unit: TDMBoundaryUnit,
        terms: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score the live generated boundary, reusing the TDM reference query."""
        batch = self._stack_replay_unit(unit.samples)
        with self.adapter.use_component_variant("generator"):
            with self.autocast():
                trainable_output = self.adapter.forward_state(
                    batch=batch,
                    state=terms.noised.state,
                    times=terms.times,
                    compute_log_prob=False,
                    return_fields=("velocity",),
                    **self._replay_forward_kwargs(batch),
                )
        trainable_velocity = require_velocity(
            trainable_output,
            algorithm_name="TDM-R1",
            role_name="generator",
        )
        trainable_values = self._diffusion_density_values(
            terms.noised.target_velocity,
            trainable_velocity,
            state=terms.noised.state,
        )
        reference_values = self._diffusion_density_values(
            terms.noised.target_velocity,
            terms.reference_velocity,
            state=terms.noised.state,
        )
        return trainable_values, reference_values.detach()

    def _boundary_score_context(self, unit: TDMBoundaryUnit) -> tuple[Any, Any, Any]:
        """Draw one noised boundary for every role scored against it.

        Returned rather than redrawn per role because two roles compared on different
        perturbations differ by the draw as much as by their parameters, which would
        turn the trust-region ratio into a measure of sampling noise.

        Args:
            unit: Boundary unit to perturb.

        Returns:
            The stacked batch, the noised state, and the component times.
        """
        batch = self._stack_replay_unit(unit.samples)
        replay_step = self.adapter.get_replay_step(batch, unit.boundary_index - 1)
        boundary_state = detach_state(replay_step.next_state)
        times = self._sample_score_query_times(unit, batch)
        return batch, self.adapter.add_forward_process_noise(boundary_state, times), times

    def _boundary_preference_values(
        self,
        unit: TDMBoundaryUnit,
        *,
        trainable_role: str,
        context: Optional[tuple[Any, Any, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-sample DSM values for a trainable role and frozen reference."""
        if context is not None:
            batch, noised, times = context
            return self._score_boundary_values(batch, noised, times, trainable_role)
        batch = self._stack_replay_unit(unit.samples)
        replay_step = self.adapter.get_replay_step(batch, unit.boundary_index - 1)
        boundary_state = detach_state(replay_step.next_state)
        times = self._sample_score_query_times(unit, batch)
        noised = self.adapter.add_forward_process_noise(boundary_state, times)
        return self._score_boundary_values(batch, noised, times, trainable_role)

    def _score_boundary_values(
        self,
        batch: Any,
        noised: Any,
        times: Any,
        trainable_role: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score one already-noised boundary with a trainable role and the reference."""
        with self.adapter.use_component_variant(trainable_role):
            with self.autocast():
                trainable_output = self.adapter.forward_state(
                    batch=batch,
                    state=noised.state,
                    times=times,
                    compute_log_prob=False,
                    return_fields=("velocity",),
                    **self._replay_forward_kwargs(batch),
                )
        trainable_velocity = require_velocity(
            trainable_output,
            algorithm_name="TDM-R1",
            role_name=trainable_role,
        )
        reference_velocity = query_score_velocity(
            self.adapter,
            batch,
            noised.state,
            times,
            role_name="reference",
            autocast=self.autocast,
            forward_kwargs=self._reference_forward_kwargs(batch),
            algorithm_name="TDM-R1",
        )
        trainable_values = self._diffusion_density_values(
            noised.target_velocity,
            trainable_velocity,
            state=noised.state,
        )
        reference_values = self._diffusion_density_values(
            noised.target_velocity,
            reference_velocity,
            state=noised.state,
        )
        return trainable_values, reference_values.detach()

    def _diffusion_density_values(
        self,
        target_velocity: LatentState,
        predicted_velocity: LatentState,
        *,
        state: LatentState,
    ) -> torch.Tensor:
        """Return per-sample squared velocity error used as a preference logit."""
        expected_names = self.adapter.trajectory_component_order
        if target_velocity.component_names != expected_names:
            raise ValueError(
                "TDM-R1 expected target velocity component order "
                f"{expected_names}, received {target_velocity.component_names}"
            )
        if predicted_velocity.component_names != expected_names:
            raise ValueError(
                "TDM-R1 expected predicted velocity component order "
                f"{expected_names}, received {predicted_velocity.component_names}"
            )
        errors = {
            name: (target_velocity.components[name] - predicted_velocity.components[name]).square()
            for name in expected_names
        }
        return self.adapter.reduce_latent_values(errors, state=state)

    def _group_preference_batch(
        self,
        unit: TDMBoundaryUnit,
        values: torch.Tensor,
    ) -> GroupPreferenceBatch:
        """Build the dense group indices for one replay unit under either layout.

        ``group_contiguous`` puts a whole group on one rank, so the group logit is a
        local sum. ``group_distributed`` deals every rank an equal share of every
        group, so each rank holds ``group_size // num_replicas`` members and the logits
        are summed across ranks. Both give the preference loss whole groups; they
        differ only in where the members live.
        """
        group_identities = torch.as_tensor(
            group_identity_rows(unit.samples),
            device=values.device,
            dtype=torch.int64,
        )
        group_size = self.training_args.group_size
        rank_local_groups = get_sampler_layout_contract(
            self.config.data_args.sampler_type
        ).groups_are_rank_local
        packed_cross_rank_groups = (
            not rank_local_groups and group_size % self.accelerator.num_processes != 0
        )
        cached_group_info = getattr(
            self,
            "_tdm_r1_reward_overlap_group_info",
            None,
        )
        if rank_local_groups:
            sorted_group_identities, local_group_indices = torch.unique(
                group_identities,
                dim=0,
                sorted=True,
                return_inverse=True,
            )
            counts = torch.bincount(
                local_group_indices,
                minlength=int(sorted_group_identities.shape[0]),
            )
            expected_members = group_size
        elif cached_group_info is not None:
            cached_group_info.validate_samples(
                unit.samples,
                context="the TDM-R1 boundary samples",
            )
            local_group_indices = cached_group_info.local_group_indices
            num_groups = cached_group_info.num_groups
            counts = None
            expected_members = None
        elif packed_cross_rank_groups:
            global_group_identities = self.accelerator.gather(group_identities)
            sorted_group_identities, counts = torch.unique(
                global_group_identities,
                dim=0,
                sorted=True,
                return_counts=True,
            )
            local_matches = torch.all(
                group_identities[:, None, :] == sorted_group_identities[None, :, :],
                dim=-1,
            )
            if not torch.all(local_matches.sum(dim=1) == 1):
                raise RuntimeError(
                    "TDM-R1 could not map local samples into the global group identity"
                )
            local_group_indices = local_matches.to(torch.int64).argmax(dim=1)
            expected_members = group_size
        else:
            sorted_group_identities, local_group_indices = torch.unique(
                group_identities,
                dim=0,
                sorted=True,
                return_inverse=True,
            )
            counts = torch.bincount(
                local_group_indices,
                minlength=int(sorted_group_identities.shape[0]),
            )
            expected_members = group_size // self.accelerator.num_processes
            self._validate_shared_group_identity(sorted_group_identities)
        if cached_group_info is None or rank_local_groups:
            num_groups = int(sorted_group_identities.shape[0])
        expected_counts = None if counts is None else torch.full_like(counts, expected_members)
        if counts is not None and not torch.equal(counts, expected_counts):
            raise ValueError(
                "TDM-R1 expected every group to contribute exactly "
                f"{expected_members} members to one "
                f"{'rank-local' if rank_local_groups else 'global'} microbatch under "
                f"sampler_type={self.config.data_args.sampler_type!r} with group_size={group_size} "
                f"and num_replicas={self.accelerator.num_processes}, received "
                f"counts={counts.tolist()} for identities={group_identities.tolist()}"
            )

        advantages = self._advantages_for_samples(unit.samples, values)
        return GroupPreferenceBatch(
            local_group_indices=local_group_indices,
            num_groups=num_groups,
            group_size=group_size,
            advantages=advantages,
            reduce_across_ranks=not rank_local_groups,
            allow_sparse_local_groups=packed_cross_rank_groups,
        )

    def _validate_shared_group_identity(self, sorted_group_identities: torch.Tensor) -> None:
        """Validate the legacy equal-share layout's rank-local group identity."""
        signature = sorted_group_identities
        highest = self.accelerator.reduce(signature.clone(), reduction="max")
        if not torch.equal(highest, signature):
            raise RuntimeError(
                "TDM-R1 expected every rank to hold the same reward groups in a microbatch "
                f"under sampler_type={self.config.data_args.sampler_type!r}; this rank holds "
                f"identities={sorted_group_identities.tolist()} while the group maximum is "
                f"{highest.tolist()}. Cross-rank group logits are summed by index, so the "
                "ranks must agree on what each index names."
            )

    def _advantages_for_samples(
        self,
        samples: Sequence[BaseSample],
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Return validated endpoint advantages aligned with diffusion values."""
        advantage_rows = []
        for sample_index, sample in enumerate(samples):
            advantage = sample.extra_kwargs.get("advantage")
            if not isinstance(advantage, (Real, torch.Tensor)) or isinstance(advantage, bool):
                raise TypeError(
                    "TDM-R1 expected scalar numeric endpoint advantage in "
                    f"sample.extra_kwargs for sample_index={sample_index}, received "
                    f"{type(advantage).__name__}: {advantage!r}"
                )
            if isinstance(advantage, torch.Tensor) and not advantage.is_floating_point():
                raise TypeError(
                    "TDM-R1 expected floating endpoint advantage tensor for "
                    f"sample_index={sample_index}, received dtype={advantage.dtype}"
                )
            advantage_row = torch.as_tensor(
                advantage,
                device=values.device,
                dtype=values.dtype,
            )
            if advantage_row.ndim != 0:
                raise ValueError(
                    "TDM-R1 expected scalar endpoint advantage for "
                    f"sample_index={sample_index}, received shape "
                    f"{tuple(advantage_row.shape)}"
                )
            advantage_rows.append(advantage_row)
        advantages = torch.stack(advantage_rows)
        if advantages.shape != values.shape:
            raise ValueError(
                "TDM-R1 expected one endpoint advantage per diffusion value, received "
                f"advantages_shape={tuple(advantages.shape)} and "
                f"values_shape={tuple(values.shape)}"
            )
        clip_range = self.training_args.advantage_clip_range
        return advantages.clamp(-clip_range, clip_range)
