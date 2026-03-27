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

# src/flow_factory/data_utils/sampler_loader.py
from torch.utils.data import Sampler, Dataset
from accelerate import Accelerator

from .sampler import DistributedKRepeatSampler, GroupContiguousSampler
from ..hparams import Arguments


def get_data_sampler(
    dataset: Dataset,
    config: Arguments,
    accelerator: Accelerator,
) -> Sampler:
    """
    Factory function to create the appropriate distributed sampler.

    Returns:
        - GroupContiguousSampler when any reward model uses async_reward
          (keeps each group's samples on the same rank)
        - DistributedKRepeatSampler otherwise (default behavior)
    """
    training_args = config.training_args
    sampler_cls = (
        GroupContiguousSampler
        if config._has_async_rewards
        else DistributedKRepeatSampler
    )
    return sampler_cls(
        dataset=dataset,
        batch_size=training_args.per_device_batch_size,
        group_size=training_args.group_size,
        unique_sample_num=training_args.unique_sample_num_per_epoch,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=training_args.seed,
    )
