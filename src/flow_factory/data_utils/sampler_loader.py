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

    The sampler strategy is determined by ``config.data_args.sampler_type``,
    which is resolved in ``Arguments._resolve_sampler_type()`` and aligned in
    ``Arguments._align_batch_geometry()`` during ``__post_init__``.

    Returns:
        - GroupContiguousSampler when resolved type is ``"group_contiguous"``
          (keeps each group's samples on the same rank)
        - DistributedKRepeatSampler when resolved type is ``"distributed_k_repeat"``
          (default behavior)
    """
    training_args = config.training_args
    sampler_cls = (
        GroupContiguousSampler
        if config.data_args.sampler_type == "group_contiguous"
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
