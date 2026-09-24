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
from torch.utils.data import Dataset, Sampler

from .sampler import (
    DistributedKRepeatSampler,
    GroupContiguousSampler,
    GroupDistributedSampler,
    GroupTiledSampler,
)

SAMPLER_REGISTRY = {
    "distributed_k_repeat": DistributedKRepeatSampler,
    "group_contiguous": GroupContiguousSampler,
    "group_distributed": GroupDistributedSampler,
    "group_tiled": GroupTiledSampler,
}


def get_data_sampler(
    dataset: Dataset,
    *,
    sampler_type: str,
    batch_size: int,
    group_size: int,
    unique_sample_num: int,
    num_replicas: int,
    rank: int,
    seed: int,
) -> Sampler:
    """Factory function to create the appropriate distributed sampler.

    All parameters are passed as explicit primitives so this function
    has no dependency on the ``Arguments`` config object.  The resolved
    per-source ``unique_sample_num`` flows naturally from
    ``DatasetTrainSpec.unique_sample_num_per_epoch`` (written by
    ``Arguments._align_batch_geometry``).

    Returns:
        - GroupContiguousSampler when ``sampler_type == "group_contiguous"``
        - GroupDistributedSampler when ``sampler_type == "group_distributed"``
        - DistributedKRepeatSampler when ``sampler_type == "distributed_k_repeat"``
        - GroupTiledSampler when ``sampler_type == "group_tiled"``
    """
    sampler_cls = SAMPLER_REGISTRY.get(sampler_type)
    if sampler_cls is None:
        raise ValueError(
            f"Unknown sampler_type={sampler_type!r}. Expected one of {sorted(SAMPLER_REGISTRY)}."
        )
    return sampler_cls(
        dataset=dataset,
        batch_size=batch_size,
        group_size=group_size,
        unique_sample_num=unique_sample_num,
        num_replicas=num_replicas,
        rank=rank,
        seed=seed,
    )
