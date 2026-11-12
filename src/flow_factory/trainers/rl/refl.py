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

"""ReFL reward gradients through a clean estimate and frozen native decoder."""

from collections import defaultdict

import torch
import torch.nn.functional as F

from ...contracts.execution import DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT
from ...samples import BaseSample
from ...scheduler.flow_match_euler_discrete import FlowMatchEulerDiscreteSDEScheduler
from ...utils.trajectory_collector import compute_trajectory_indices
from ..abc import BaseTrainer
from ..common.context_parallel import ContextParallelTrainingCapability
from ..common.forward_kwargs import replay_forward_kwargs
from ..common.group_noise import identity_timestep_generator, sample_noise_ids


class ReFLTrainer(BaseTrainer):
    """Differentiate one selected prediction per saved-state microbatch.

    Prefix sampling is detached. Clean-state projection, native decode and scorer
    input derivatives remain connected until the policy backward completes.
    """

    execution_contract = DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT
    paradigm = "decoupled"
    context_parallel_training_capability = ContextParallelTrainingCapability()

    @classmethod
    def validate_adapter_class_execution_contract(cls, adapter_cls: type) -> None:
        """Require an explicitly qualified image decoder before loading weights."""
        super().validate_adapter_class_execution_contract(adapter_cls)
        if adapter_cls.__dict__.get("differentiable_decode_fields") != ("image",):
            raise ValueError(
                f"ReFL requires declared image input-gradient decoding, received {adapter_cls.__name__}"
            )

    def _validate_paradigm_dynamics(self) -> None:
        scheduler = self.adapter.scheduler
        if (
            not isinstance(scheduler, FlowMatchEulerDiscreteSDEScheduler)
            or scheduler.dynamics_type != "ODE"
        ):
            raise ValueError(
                f"ReFL requires the Euler flow ODE scheduler, received {type(scheduler).__name__}"
            )

    def _initialize_snapshots(self) -> None:
        """Keep decoder parameters frozen while allowing gradients to its input."""
        super()._initialize_snapshots()
        self.adapter._require_component("vae").eval().requires_grad_(False)

    def runtime_execution_identity_payload(self) -> dict:
        """Bind resolved scorer revisions and preprocessing to drained continuation."""
        identity = super().runtime_execution_identity_payload()
        scorers = {}
        for name, scorer in self.reward_models.items():
            config = getattr(scorer.model, "config", None)
            processor = getattr(scorer, "processor", None)
            image_processor = getattr(processor, "image_processor", None)
            scorers[name] = {
                "class": f"{type(scorer).__module__}.{type(scorer).__qualname__}",
                "gradient_fields": scorer.gradient_capability.input_fields,
                "model_revision": getattr(
                    scorer, "resolved_model_revision", getattr(config, "_commit_hash", None)
                ),
                "processor_revision": getattr(scorer, "resolved_processor_revision", None),
                "image_preprocessing": (
                    image_processor.to_dict() if image_processor is not None else None
                ),
            }
        identity["differentiable_scorers"] = scorers
        identity["differentiable_decode_fields"] = self.adapter.differentiable_decode_fields
        return identity

    def sample(self) -> list[BaseSample]:
        """Retain shared scheduler-selected states for late-stage prediction."""
        indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.get_train_step_indices(),
            num_inference_steps=self.training_args.num_inference_steps,
            include_terminal=True,
        )
        return self.generate_samples(
            reward_buffer=None, compute_log_prob=False, trajectory_indices=indices
        )

    def optimize(self, samples: list[BaseSample]) -> None:
        """Build and release one reward-to-policy graph per accumulated microstep."""
        loss_info = defaultdict(list)
        self.adapter.train()
        candidates = self.adapter.get_train_step_indices()
        for inner_epoch in range(self.training_args.num_inner_epochs):
            ordered = self._order_samples_for_optimize(samples, inner_epoch)
            for batch in self._iter_prefetched_batches(
                ordered, self.training_args.per_device_batch_size
            ):
                generator = identity_timestep_generator(
                    self, sample_noise_ids(batch.samples), inner_epoch
                )
                step = int(candidates[torch.randint(len(candidates), (), generator=generator)])
                replay = self.adapter.get_replay_step(batch, step)
                with self.accumulate_gradients():
                    with self.autocast():
                        output = self.adapter.forward_state(
                            batch=batch,
                            state=replay.state,
                            times=replay.times,
                            next_state=replay.next_state,
                            compute_log_prob=False,
                            return_fields=("velocity",),
                            **replay_forward_kwargs(self, batch),
                        )
                        projection_times = self.adapter.build_training_component_times(
                            replay.times.timestep["latent"], batch=batch
                        )
                        clean = self.adapter.project_velocity_to_clean_state(
                            replay.state, projection_times, output.velocity
                        )
                        media = self.adapter.decode_differentiable_state(clean, batch=batch)
                    source_names = self._reward_source_names(batch.samples)
                    feedback = self.differentiable_feedback(
                        prompt=list(batch["prompt"]), source_names=source_names, **media
                    )
                    terms = []
                    for name, scores in feedback.scores.items():
                        target = self.training_args.refl_reward_target
                        per_sample = -scores if target is None else F.relu(target - scores)
                        terms.append((per_sample * feedback.weights[name]).mean())
                        loss_info[f"reward/{name}"].append(scores.detach().mean())
                    loss = self.training_args.refl_reward_scale * torch.stack(terms).sum()
                    loss_info["loss"].append(loss.detach())
                    self._backward_full_output_loss(loss)
                    if self.accelerator.sync_gradients:
                        loss_info = self._apply_optimizer_step(loss_info)

    def _reward_source_names(self, samples) -> list[str]:
        """Resolve source-less single-source batches without guessing mixed routing."""
        sources = self.config.data_args.source_id_to_name
        names = []
        for sample in samples:
            if sample.source_id is None:
                if len(sources) != 1:
                    raise ValueError(
                        f"ReFL sample lacks source identity; configured sources={sources!r}"
                    )
                names.append(sources[0])
            else:
                names.append(sources[int(sample.source_id)])
        return names
