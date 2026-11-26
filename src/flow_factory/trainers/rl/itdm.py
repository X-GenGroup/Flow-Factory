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

"""Reward-driven velocity revision on terminal samples of the current policy."""

from dataclasses import replace
from functools import partial
from typing import ClassVar

import torch
import torch.nn.functional as F

from ...contracts.execution import ONLINE_EXECUTION_CONTRACT
from ...hparams.optimizer_args import resolve_role_optimizer_args
from ...samples import BaseSample
from ...utils.base import create_generator
from ...utils.dist import data_parallel_geometry
from ..abc import BaseTrainer
from ..common.context_parallel import ContextParallelTrainingCapability
from ..common.group_noise import training_timestep_generator
from ..common.replay_batching import move_and_stack_samples
from ..distillation.distillation_runtime import (
    log_distillation_metrics,
    query_score_velocity,
    record_distillation_metric,
    require_velocity,
    run_role_phase,
)


def _policy_parameter_name(parameter_name: str, role_adapter: str, policy_variant: str) -> str:
    """Return the policy counterpart of one role-owned LoRA parameter name.

    PEFT names a tuner weight ``<module>.lora_X.<adapter>.<suffix>``, so only the
    segment that follows a ``lora_`` tuner segment names the adapter. The base
    variant owns PEFT's ``default`` adapter.
    """
    parts = parameter_name.split(".")
    for index in range(1, len(parts)):
        if parts[index] == role_adapter and parts[index - 1].startswith("lora_"):
            return ".".join(parts[:index] + ["default"] + parts[index + 1 :])
    raise ValueError(
        f"expected a PEFT tuner segment before adapter {role_adapter!r} in parameter "
        f"{parameter_name!r} while seeding it from policy variant {policy_variant!r}"
    )


def revised_velocity_loss(
    policy: torch.Tensor,
    dsm: torch.Tensor,
    target: torch.Tensor,
    advantage: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    """Match the detached iTDM revision with one signed reward per sample."""
    weight = advantage.float().clamp(-clip, clip).reshape(-1, 1, 1, 1) / 5
    revised = (policy - weight * (dsm - target)).detach()
    return F.mse_loss(policy, revised)


class ITDMTrainer(BaseTrainer):
    """Fit an online DSM before revising the rollout policy on the same perturbations."""

    paradigm = "decoupled"
    execution_contract = ONLINE_EXECUTION_CONTRACT
    context_parallel_training_capability: ClassVar[ContextParallelTrainingCapability] = (
        ContextParallelTrainingCapability()
    )

    def _declare_model_variants(self) -> None:
        """Seed the DSM role from the policy LoRA once the roles hold final weights.

        Upstream iTDM initializes ``dsm`` from ``default`` *after* placing the policy
        adapter, so a warm start fits the score to the policy it will revise. The base
        declaration restores whichever roles the checkpoint provides first; only a role
        the checkpoint omits still needs seeding.
        """
        super()._declare_model_variants()
        if "fake" in self.restored_training_roles:
            return
        registry = self.adapter.component_variant_registry
        policy = registry.base_variant
        base = {
            (record.component_name, record.parameter_name): record.parameter
            for record in registry.parameter_records(policy)
        }
        with torch.no_grad():
            for record in registry.parameter_records("fake"):
                key = (
                    record.component_name,
                    _policy_parameter_name(record.parameter_name, "fake", policy),
                )
                source = base.get(key)
                if source is None:
                    raise ValueError(
                        f"iTDM expected policy parameter {key[0]}.{key[1]} as the counterpart "
                        f"for fake parameter {record.component_name}.{record.parameter_name}, "
                        f"received none among {len(base)} policy parameters"
                    )
                record.parameter.copy_(source)

    def _optimizer_args_for_role(self, role_name):
        return resolve_role_optimizer_args(
            role_name,
            self.config.optimizer_args.get_by_name(role_name),
            self.training_args.default_role_optimizers(),
            algorithm_name="iTDM",
        )

    def sample(self) -> list[BaseSample]:
        return self.generate_samples(
            reward_buffer=self.reward_buffer, compute_log_prob=False, trajectory_indices=[-1]
        )

    def optimize(self, samples: list[BaseSample]) -> None:
        args = self.training_args
        size = args.per_device_batch_size
        gas = args.gradient_accumulation_steps
        if (
            not samples
            or len(samples) % size
            or (len(samples) // size * args.num_train_timesteps) % gas
        ):
            raise ValueError(
                f"iTDM cannot partition {len(samples)} samples into batch={size}, GAS={gas}"
            )
        self.adapter.train()
        for inner in range(args.num_inner_epochs):
            ordered_samples = self._order_samples_for_optimize(samples, inner)
            # All ranks use the same time draw. Each logical data replica replays
            # its own keyed noise in both roles without caching GPU tensors.
            indices = torch.randperm(
                args.max_timestep_index + 1, generator=training_timestep_generator(self, inner, 0)
            )[: args.num_train_timesteps].tolist()
            work = [(start, index) for start in range(0, len(samples), size) for index in indices]
            roles = self._role_update_plan().phases
            for phase in roles:
                if phase.role_name == "generator" and (self.epoch + 1) % args.ttur:
                    continue
                loss_fn = partial(self._loss, ordered_samples, inner, phase.role_name)
                for offset in range(0, len(work), gas):
                    run_role_phase(self, phase.role_name, work[offset : offset + gas], loss_fn)
                record_distillation_metric(
                    self,
                    f"train/{phase.role_name}_step",
                    self.optimization_roles[phase.role_name].step,
                )
        log_distillation_metrics(self)

    def _loss(self, samples, inner, role, item):
        start, index = item
        args = self.training_args
        device = self.accelerator.device
        # Replay consumes conditioning and the terminal latent. Moving copies keeps
        # offloaded rollout storage on its original device throughout both phases.
        replay = [
            replace(sample, image=None)
            for sample in samples[start : start + args.per_device_batch_size]
        ]
        batch = move_and_stack_samples(replay, device)
        clean = self.adapter.get_terminal_state(batch)
        times = self.adapter.build_training_component_times(
            batch["timesteps"][:, index], batch=batch
        )
        _, replica_rank = data_parallel_geometry(self.accelerator)
        rng = create_generator(
            args.seed,
            self.epoch,
            inner,
            start,
            index,
            replica_rank,
            device=device,
        )
        noised = self.adapter.add_forward_process_noise(clean, times, generator=rng)
        target = noised.target_velocity.components["latent"].float()
        guidance = args.guidance_scale
        if role == "generator":
            query = partial(
                query_score_velocity,
                self.adapter,
                batch,
                noised.state,
                times,
                autocast=self.autocast,
                algorithm_name="iTDM",
            )
            dsm = (
                query(role_name="fake", forward_kwargs={"guidance_scale": 1.0})
                .components["latent"]
                .float()
            )
            if args.kl_beta:
                ref = (
                    query(role_name="reference", forward_kwargs={"guidance_scale": guidance})
                    .components["latent"]
                    .float()
                )

        with self.autocast():
            output = self.adapter.forward_state(
                batch=batch,
                state=noised.state,
                times=times,
                compute_log_prob=False,
                return_fields=("velocity",),
                guidance_scale=1.0 if role == "fake" else guidance,
            )
        velocity = (
            require_velocity(output, algorithm_name="iTDM", role_name=role)
            .components["latent"]
            .float()
        )
        if role == "fake":
            return F.mse_loss(velocity, target)
        loss = revised_velocity_loss(
            velocity, dsm, target, batch["advantage"], args.advantage_clip_range
        )
        if args.kl_beta:
            reference_loss = F.mse_loss(velocity, ref)
            record_distillation_metric(self, "train/reference_velocity_mse", reference_loss)
            loss = loss + args.kl_beta * reference_loss
        return loss
