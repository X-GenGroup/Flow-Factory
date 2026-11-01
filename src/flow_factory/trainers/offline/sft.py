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

"""Offline supervised flow-matching trainer."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, Dict, List, Literal, Tuple

import torch
from torch.utils.data import DataLoader

from ...contracts import OFFLINE_EXECUTION_CONTRACT
from ...data_utils.offline_dataset import DemonstrationOutputBatch, OfflineBatch
from ...data_utils.offline_train_data import build_offline_train_dataloader
from ..abc import BaseTrainer
from ..common.flow_matching import (
    build_noised_output_state,
    flow_matching_per_sample_loss,
    sample_offline_timesteps,
)
from ..common.group_noise import (
    STREAM_FORWARD_NOISE,
    STREAM_OUTPUT_ENCODING,
    identity_timestep_generator,
    independent_batch_generator,
    record_noise_ids,
)
from ..common.offline_batch import bind_prepared_condition_output, move_condition_to_device
from ..forward_process import forward_velocity_state


class SFTTrainer(BaseTrainer):
    """Train a flow-matching policy from a finite demonstration dataset."""

    paradigm: ClassVar[Literal["decoupled"]] = "decoupled"
    execution_contract = OFFLINE_EXECUTION_CONTRACT

    def _build_train_dataloader(self) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        """Build the finite loader owned by its official distributed sampler."""
        dataloader = build_offline_train_dataloader(
            config=self.config,
            accelerator=self.accelerator,
            preprocess_func=self.adapter.preprocess_func,
            supervision_type="demonstration",
            pipeline_io_contract=self.adapter.effective_pipeline_io_contract,
        )
        return dataloader, {}

    def optimize_batch(self, batch: Any) -> None:
        """Apply one gradient-accumulation microstep to a demonstration batch.

        Target media is VAE-encoded on demand. Independently sampled time terms
        are averaged inside this microstep, so they do not alter dataloader epoch,
        gradient-accumulation, or optimizer-step cadence.

        Args:
            batch: One exact :class:`OfflineBatch` with demonstration output.
        """
        output = self._require_demonstration_batch(batch)
        condition = move_condition_to_device(batch.condition, self.accelerator.device)

        # Evaluation leaves trainable components in eval mode. Every finite-data
        # microstep explicitly restores training mode before policy execution.
        self.adapter.train()

        prepared_condition = self.adapter.prepare_condition_state(condition)
        identity = record_noise_ids(batch.record_ids)
        # A stochastic output codec (a VAE posterior `sample()`) would otherwise
        # draw from the process RNG, which `set_seed(device_specific=True)` keys
        # per rank: the supervised target would then depend on which rank encoded
        # it and on how many ranks were launched. Tensor-parallel peers hold the
        # same record and must encode it to the same latent.
        encoded = self.adapter.encode_output_state(
            output.target_media,
            prepared_condition,
            generator=independent_batch_generator(self, identity, 0, STREAM_OUTPUT_ENCODING),
        )
        model_batch = bind_prepared_condition_output(
            prepared_condition,
            encoded.forward_context,
        )
        all_timesteps = sample_offline_timesteps(
            self.training_args,
            batch_size=len(output.target_media),
            device=self.accelerator.device,
            generator=identity_timestep_generator(self, identity),
        )

        with self.accumulate_gradients():
            time_losses = []
            for timestep_index, primary_timesteps in enumerate(all_timesteps):
                times, noised = build_noised_output_state(
                    self.adapter,
                    encoded.clean_state,
                    primary_timesteps,
                    batch=model_batch,
                    generator=independent_batch_generator(
                        self, identity, timestep_index, STREAM_FORWARD_NOISE
                    ),
                )
                with self.autocast():
                    predicted_velocity = forward_velocity_state(
                        self,
                        model_batch,
                        noised.state,
                        times,
                        source="SFT policy",
                        **self.adapter.offline_training_forward_overrides,
                    )
                time_losses.append(
                    flow_matching_per_sample_loss(
                        self.adapter,
                        predicted_velocity,
                        noised,
                    )
                )

            per_sample_loss = torch.stack(time_losses, dim=0).mean(dim=0)
            loss = per_sample_loss.mean()
            self.accelerator.backward(loss)

            # Only a completed backward enters the persistent accumulation window.
            loss_info = self._get_loss_info()
            loss_info["loss"].append(loss.detach())
            loss_info["flow_matching_loss"].append(per_sample_loss.mean().detach())
            if self.accelerator.sync_gradients:
                self._loss_info = self._apply_optimizer_step(loss_info)

    def _get_loss_info(self) -> Dict[str, List[torch.Tensor]]:
        """Return the metric window, including for lightweight test instances."""
        loss_info = getattr(self, "_loss_info", None)
        if loss_info is None:
            loss_info = defaultdict(list)
            self._loss_info = loss_info
        return loss_info

    @staticmethod
    def _require_demonstration_batch(batch: Any) -> DemonstrationOutputBatch:
        """Return a demonstration output after strict algorithm-boundary checks."""
        if type(batch) is not OfflineBatch:
            raise TypeError(
                "SFTTrainer requires an exact OfflineBatch, "
                f"received {type(batch).__name__}: {batch!r}"
            )
        if batch.supervision_type != "demonstration":
            raise ValueError(
                "SFTTrainer requires supervision_type='demonstration', "
                f"received {batch.supervision_type!r}"
            )
        if type(batch.output) is not DemonstrationOutputBatch:
            raise TypeError(
                "SFTTrainer requires DemonstrationOutputBatch output, "
                f"received {type(batch.output).__name__}: {batch.output!r}"
            )
        return batch.output


__all__ = ["SFTTrainer"]
