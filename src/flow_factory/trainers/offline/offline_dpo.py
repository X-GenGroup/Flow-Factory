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

"""Finite-dataset diffusion DPO with on-the-fly output encoding."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, Dict, List, Literal, Tuple

import torch
from torch.utils.data import DataLoader

from ...contracts import OFFLINE_EXECUTION_CONTRACT
from ...data_utils.offline_dataset import OfflineBatch, PreferenceOutputBatch
from ...data_utils.offline_train_data import build_offline_train_dataloader
from ..abc import BaseTrainer
from ..common import pairwise_policy_activation_context
from ..common.dpo_objective import dpo_objective
from ..common.flow_matching import (
    build_noised_output_state,
    flow_matching_per_sample_loss,
    sample_offline_timesteps,
    validate_preference_component_times,
    validate_preference_output_states,
)
from ..common.offline_batch import bind_prepared_condition_output, move_condition_to_device
from ..forward_process import forward_velocity_state

MetricAccumulator = Dict[str, List[torch.Tensor]]


class OfflineDPOTrainer(BaseTrainer):
    """Optimize chosen/rejected dataset pairs against a frozen reference policy."""

    paradigm: ClassVar[Literal["decoupled"]] = "decoupled"
    execution_contract = OFFLINE_EXECUTION_CONTRACT

    def _build_train_dataloader(self) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        """Build the finite preference loader without Accelerator reshaping it."""
        dataloader = build_offline_train_dataloader(
            config=self.config,
            accelerator=self.accelerator,
            preprocess_func=self.adapter.preprocess_func,
            supervision_type="preference",
            pipeline_io_contract=self.adapter.effective_pipeline_io_contract,
        )
        return dataloader, {}

    def optimize_batch(self, batch: Any) -> None:
        """Apply one gradient-accumulation microstep from a preference batch."""
        preference = _require_preference_batch(batch)
        self.adapter.train()

        condition = move_condition_to_device(batch.condition, self.accelerator.device)
        prepared_condition = self.adapter.prepare_condition_state(condition)
        chosen = self.adapter.encode_output_state(preference.chosen_media, prepared_condition)
        rejected = self.adapter.encode_output_state(preference.rejected_media, prepared_condition)
        validate_preference_output_states(chosen, rejected)

        chosen_batch = bind_prepared_condition_output(
            prepared_condition,
            chosen.forward_context,
        )
        rejected_batch = bind_prepared_condition_output(
            prepared_condition,
            rejected.forward_context,
        )
        all_timesteps = sample_offline_timesteps(
            self.training_args,
            batch_size=len(preference.chosen_media),
            device=self.accelerator.device,
        )

        timestep_losses: List[torch.Tensor] = []
        timestep_metrics: Dict[str, List[torch.Tensor]] = defaultdict(list)
        with self.accumulate_gradients():
            for primary_timesteps in all_timesteps:
                chosen_times, chosen_noised = build_noised_output_state(
                    self.adapter,
                    chosen.clean_state,
                    primary_timesteps,
                    batch=chosen_batch,
                )
                rejected_times, rejected_noised = build_noised_output_state(
                    self.adapter,
                    rejected.clean_state,
                    primary_timesteps,
                    batch=rejected_batch,
                    noise=chosen_noised.noise,
                )
                validate_preference_component_times(chosen_times, rejected_times)

                with self.autocast(), pairwise_policy_activation_context(self):
                    policy_chosen = forward_velocity_state(
                        self,
                        chosen_batch,
                        chosen_noised.state,
                        chosen_times,
                        source="offline DPO policy chosen",
                        **self.adapter.offline_training_forward_overrides,
                    )
                with self.autocast(), pairwise_policy_activation_context(self):
                    policy_rejected = forward_velocity_state(
                        self,
                        rejected_batch,
                        rejected_noised.state,
                        rejected_times,
                        source="offline DPO policy rejected",
                        **self.adapter.offline_training_forward_overrides,
                    )

                # A full-parameter snapshot is installed once for both arms.
                # LoRA adapters use the same scope to disable trainable adapters.
                with torch.no_grad(), self.adapter.use_ref_parameters(), self.autocast():
                    reference_chosen = forward_velocity_state(
                        self,
                        chosen_batch,
                        chosen_noised.state,
                        chosen_times,
                        source="offline DPO reference chosen",
                        **self.adapter.offline_training_forward_overrides,
                    )
                    reference_rejected = forward_velocity_state(
                        self,
                        rejected_batch,
                        rejected_noised.state,
                        rejected_times,
                        source="offline DPO reference rejected",
                        **self.adapter.offline_training_forward_overrides,
                    )

                policy_chosen_loss = flow_matching_per_sample_loss(
                    self.adapter,
                    policy_chosen,
                    chosen_noised,
                )
                policy_rejected_loss = flow_matching_per_sample_loss(
                    self.adapter,
                    policy_rejected,
                    rejected_noised,
                )
                reference_chosen_loss = flow_matching_per_sample_loss(
                    self.adapter,
                    reference_chosen,
                    chosen_noised,
                )
                reference_rejected_loss = flow_matching_per_sample_loss(
                    self.adapter,
                    reference_rejected,
                    rejected_noised,
                )
                loss, metrics = dpo_objective(
                    policy_chosen_loss=policy_chosen_loss,
                    policy_rejected_loss=policy_rejected_loss,
                    reference_chosen_loss=reference_chosen_loss,
                    reference_rejected_loss=reference_rejected_loss,
                    beta=self.training_args.beta,
                )
                timestep_losses.append(loss)
                timestep_metrics["theta_w_err"].append(policy_chosen_loss.mean())
                timestep_metrics["theta_l_err"].append(policy_rejected_loss.mean())
                timestep_metrics["ref_w_err"].append(reference_chosen_loss.mean())
                timestep_metrics["ref_l_err"].append(reference_rejected_loss.mean())
                timestep_metrics["implicit_accuracy"].append(metrics["implicit_accuracy"])
                timestep_metrics["implicit_reward_chosen"].append(
                    metrics["implicit_reward_chosen"].mean()
                )
                timestep_metrics["implicit_reward_rejected"].append(
                    metrics["implicit_reward_rejected"].mean()
                )

            loss = torch.stack(timestep_losses).mean()
            self.accelerator.backward(loss)

            loss_info = self._offline_loss_info()
            loss_info["loss"].append(loss.detach())
            for name, values in timestep_metrics.items():
                loss_info[name].append(torch.stack(values).mean().detach())
            if self.accelerator.sync_gradients:
                self._offline_dpo_loss_info = self._apply_optimizer_step(loss_info)

    def _offline_loss_info(self) -> MetricAccumulator:
        """Return metrics accumulated across the current gradient window."""
        loss_info = getattr(self, "_offline_dpo_loss_info", None)
        if loss_info is None:
            loss_info = defaultdict(list)
            self._offline_dpo_loss_info = loss_info
        return loss_info


def _require_preference_batch(batch: Any) -> PreferenceOutputBatch:
    """Validate the algorithm-owned portion of one collated offline batch."""
    if type(batch) is not OfflineBatch:
        raise TypeError(
            "OfflineDPOTrainer requires an exact OfflineBatch, "
            f"received {type(batch).__name__}: {batch!r}"
        )
    if batch.supervision_type != "preference":
        raise ValueError(
            "OfflineDPOTrainer requires supervision_type='preference', "
            f"received {batch.supervision_type!r}"
        )
    if type(batch.output) is not PreferenceOutputBatch:
        raise TypeError(
            "OfflineDPOTrainer requires PreferenceOutputBatch output, "
            f"received {type(batch.output).__name__}: {batch.output!r}"
        )
    output = batch.output
    if type(output.chosen_media) is not tuple or type(output.rejected_media) is not tuple:
        raise TypeError("offline preference media arms must be tuples")
    if not output.chosen_media:
        raise ValueError("offline preference batch must contain at least one pair")
    if len(output.chosen_media) != len(output.rejected_media):
        raise ValueError(
            "offline preference arms must contain the same batch size, "
            f"received chosen={len(output.chosen_media)} and "
            f"rejected={len(output.rejected_media)}"
        )
    return output


__all__ = ["OfflineDPOTrainer"]
