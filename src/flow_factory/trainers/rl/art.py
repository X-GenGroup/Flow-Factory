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

"""DiffusionART on complete rank-local groups and stored ODE trajectories."""

from collections import defaultdict
from dataclasses import replace
from typing import ClassVar

import torch

from ...contracts.reward_overlap import RANK_LOCAL_REWARD_OPTIMIZATION_OVERLAP
from ...samples import BaseSample
from ...samples.identity import GroupKey, sample_group_key
from ...scheduler.flow_match_euler_discrete import FlowMatchEulerDiscreteSDEScheduler
from ...scheduler.minimax_h3 import MiniMaxH3SDEScheduler
from ...utils.noise_schedule import flow_match_sigma
from ...utils.trajectory_collector import compute_trajectory_indices
from ..abc import BaseTrainer
from ..common.context_parallel import ContextParallelTrainingCapability
from ..common.forward_kwargs import replay_forward_kwargs
from .art_objective import art_objective, endpoint_guidance


def _group_key(sample: BaseSample) -> GroupKey:
    """Use the sampler's identity, never prompt text or rank-local batch position."""
    if sample.sampling_group_id is None:
        raise ValueError("ART requires a planned sampling_group_id on every sample")
    return sample_group_key(sample)


class DiffusionARTTrainer(BaseTrainer):
    """Configurable replay passes against one frozen rollout anchor.

    This is ODE velocity regression, not a transition likelihood objective. The
    existing decoupled execution category is used, with stricter rollout parity
    checks because ART consumes actual rollout states and their saved velocities.
    Only acquisition-boundary checkpoints are resumable; guidance is ephemeral.
    """

    reward_optimization_overlap_contract = RANK_LOCAL_REWARD_OPTIMIZATION_OVERLAP
    _art_velocity_sign = 1.0
    paradigm = "decoupled"
    context_parallel_training_capability: ClassVar[ContextParallelTrainingCapability] = (
        ContextParallelTrainingCapability()
    )

    def _validate_paradigm_dynamics(self):
        """The trust metric is defined for this concrete Euler ODE convention."""
        scheduler = self.adapter.scheduler
        if (
            not isinstance(scheduler, (FlowMatchEulerDiscreteSDEScheduler, MiniMaxH3SDEScheduler))
            or scheduler.dynamics_type != "ODE"
        ):
            raise ValueError(
                f"ART requires a supported Euler flow scheduler with ODE dynamics, received {type(scheduler).__name__} / {scheduler.dynamics_type}"
            )

        # Stored H3 velocity points toward data; normalize both policy and anchor
        # to dx/dsigma before applying the shared loss and virtual transition metric.
        self._art_velocity_sign = -1.0 if isinstance(scheduler, MiniMaxH3SDEScheduler) else 1.0

    def sample(self) -> list[BaseSample]:
        """Store selected states, terminal endpoints and the actual rollout velocity."""
        indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.get_train_step_indices(),
            num_inference_steps=self.training_args.num_inference_steps,
            include_terminal=True,
        )
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=False,
            trajectory_indices=indices,
            extra_call_back_kwargs=["velocity"],
        )

    def _group_guidance(
        self, samples: list[BaseSample], component: str = "latent"
    ) -> dict[GroupKey, torch.Tensor]:
        """Aggregate complete groups before shuffling or optimizer microbatching."""
        groups = defaultdict(list)
        for sample in samples:
            groups[_group_key(sample)].append(sample)
        result = {}
        for key, members in groups.items():
            if len(members) != self.training_args.group_size or sorted(
                s.sampling_group_member_id for s in members
            ) != list(range(self.training_args.group_size)):
                raise ValueError(
                    f"ART requires exactly one member 0..K-1 for group {key}; received {[s.sampling_group_member_id for s in members]}"
                )
            # A terminal view may retain its entire stacked CPU trajectory.
            # Copy each endpoint before stacking the next member, bounding that
            # temporary residency to one trajectory plus the group endpoints.
            endpoints, masks = [], []
            for sample in members:
                state = self.adapter.get_terminal_state(BaseSample.stack([sample]))
                latent = state.components[component]
                endpoints.append(latent.detach().to("cpu", copy=True))
                masks.append(
                    None
                    if state.active_masks is None
                    else torch.broadcast_to(state.active_masks[component], latent.shape).to(
                        "cpu", copy=True
                    )
                )
                del state, latent
            if any(m is not None for m in masks) and not all(m is not None for m in masks):
                raise ValueError(f"ART group {key} mixes masked and unmasked endpoints")
            mask = None if masks[0] is None else torch.cat(masks)
            endpoints = torch.cat(endpoints)
            rewards = torch.stack(
                [
                    torch.as_tensor(s.extra_kwargs["advantage"]).detach().cpu().reshape(())
                    for s in members
                ]
            )
            result[key] = endpoint_guidance(
                endpoints, rewards, epsilon=self.training_args.art_epsilon, active_mask=mask
            )
        return result

    def optimize(self, samples: list[BaseSample]) -> None:
        """Replay each configured inner epoch using the shared optimizer schedule."""
        expected_rows = (
            self.training_args.num_batches_per_epoch * self.training_args.per_device_batch_size
        )
        if len(samples) != expected_rows:
            raise ValueError(
                f"ART requires {expected_rows} acquired samples, received {len(samples)}"
            )
        loss_info = defaultdict(list)
        for inner_epoch in range(self.training_args.num_inner_epochs):
            ordered = self._order_samples_for_optimize(samples, inner_epoch)
            _, loss_info = self._optimize_partition(ordered, loss_info)

    @classmethod
    def reward_optimization_overlap_geometry(cls, config):
        """Let group tiles share an acquisition-wide accumulation window."""
        return replace(
            super().reward_optimization_overlap_geometry(config), accumulation_scope="acquisition"
        )

    def _prepare_reward_optimization_overlap(self, samples, plan):
        """Keep one update counter across complete-group reward tiles."""
        return {"updates": 0, "rows": 0, "loss_info": defaultdict(list)}

    def _optimize_reward_overlap_tile(self, tile, samples, context):
        """Accumulate a ready group tile without closing an artificial update window."""
        updates, metrics = self._optimize_partition(samples, context["loss_info"])
        context["updates"] += updates
        context["rows"] += len(samples)
        context["loss_info"] = metrics

    def _finalize_reward_optimization_overlap(self, context, samples):
        """Check the resolved accumulation schedule and one visit per acquired row."""
        expected_updates = (
            self.training_args.num_batches_per_epoch * len(self.adapter.get_train_step_indices())
        ) // self.training_args.gradient_accumulation_steps
        if context["updates"] != expected_updates or context["rows"] != len(samples):
            raise RuntimeError(
                f"ART incomplete tile stream: {context['updates']} updates, {context['rows']}/{len(samples)} rows"
            )
        return {}

    def _optimize_partition(self, ordered, loss_info):
        """Consume complete groups using saved anchors and the shared accumulator."""
        components = self.adapter.trajectory_component_order
        guidance = {name: self._group_guidance(ordered, name) for name in components}
        self.adapter.train()
        updates = 0
        for batch in self._iter_prefetched_batches(
            ordered, self.training_args.per_device_batch_size
        ):
            directions = {
                name: torch.cat([guidance[name][_group_key(s)] for s in batch.samples]).to(
                    self.accelerator.device
                )
                for name in components
            }
            for step_index in self.adapter.get_train_step_indices().tolist():
                with self.accumulate_gradients():
                    replay = self.adapter.get_replay_step(batch, step_index)
                    old = self.adapter.get_replay_callback(batch, step_index, "velocity")
                    with self.autocast():
                        output = self.adapter.forward_state(
                            batch=batch,
                            state=replay.state,
                            times=replay.times,
                            next_state=replay.next_state,
                            compute_log_prob=False,
                            return_fields=("velocity", "dt"),
                            **replay_forward_kwargs(self, batch),
                        )
                    component_losses = []
                    for name in components:
                        terms = art_objective(
                            self._art_velocity_sign * output.velocity.components[name],
                            self._art_velocity_sign * old.components[name],
                            directions[name],
                            flow_match_sigma(replay.times.timestep[name]),
                            output.dt[name],
                            guidance_scale=self.training_args.art_guidance_scale,
                            kl_threshold=self.training_args.art_kl_threshold,
                            reference_noise_level=self.training_args.art_reference_noise_level,
                            epsilon=self.training_args.art_epsilon,
                            active_mask=(
                                None
                                if replay.state.active_masks is None
                                else replay.state.active_masks[name]
                            ),
                        )
                        component_losses.append(terms["per_sample_loss"])
                        loss_info[f"art_keep_fraction/{name}"].append(terms["keep"].float().mean())
                        loss_info[f"art_reference_kl/{name}"].append(terms["kl"].mean())
                    # Equal component means match the upstream AV objective and prevent
                    # a larger video latent from implicitly overwhelming audio.
                    loss = torch.stack(component_losses).mean()
                    loss_info["loss"].append(loss.detach())
                    self._backward_full_output_loss(loss)
                    if self.accelerator.sync_gradients:
                        loss_info = self._apply_optimizer_step(loss_info)
                        updates += 1
        return updates, loss_info
