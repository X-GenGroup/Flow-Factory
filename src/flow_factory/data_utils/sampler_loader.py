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
    ContiguousShardSampler,
    DistributedKRepeatSampler,
    GroupContiguousSampler,
    GroupDistributedSampler,
    GroupTiledSampler,
    SubgroupTiledSampler,
)

SAMPLER_REGISTRY = {
    "global_random": DistributedKRepeatSampler,
    "contiguous_shard": ContiguousShardSampler,
    "rank_local": GroupContiguousSampler,
    "global_batch": GroupDistributedSampler,
    "global_tile": GroupTiledSampler,
    "subgroup_tile": SubgroupTiledSampler,
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
    subgroup_size: int | None = None,
) -> Sampler:
    """Factory function to create the appropriate distributed sampler.

    All parameters are passed as explicit primitives so this function
    has no dependency on the ``Arguments`` config object.  The resolved
    per-source ``unique_sample_num`` flows naturally from
    ``DatasetTrainSpec.unique_sample_num_per_epoch`` (written by
    ``Arguments._align_batch_geometry``).

    Returns:
        A planned batch sampler registered under ``sampler_type``. Both the
        semantic names and the four legacy aliases are accepted.
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
        subgroup_size=subgroup_size,
    )
