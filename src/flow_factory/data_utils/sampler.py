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

# src/flow_factory/data_utils/sampler.py
import math
from typing import Sized, cast

import torch
from torch.utils.data import Dataset, Sampler

from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__, rank_zero_only=True)


def _dataset_size(dataset: Dataset) -> int:
    if not hasattr(dataset, "__len__"):
        raise TypeError("Sampler requires dataset with __len__, " f"got {type(dataset).__name__}.")
    return len(cast(Sized, dataset))


class DistributedKRepeatSampler(Sampler):
    """ """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        group_size: int,
        unique_sample_num: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = group_size  # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas, process num, gpu num
        self.rank = rank  # Current replica rank
        self.seed = seed  # Random seed for synchronization
        self.m = unique_sample_num  # `Least` number of unique sample per epoch

        dataset_size = _dataset_size(self.dataset)
        if unique_sample_num > dataset_size:
            raise ValueError(
                f"`unique_sample_num` ({unique_sample_num}) must be <= dataset size ({dataset_size})."
            )

        # Compute the number of samples for each batch iteration
        self.sample_num_per_iteration = self.num_replicas * self.batch_size
        step = self.sample_num_per_iteration // math.gcd(self.k, self.sample_num_per_iteration)
        new_m = (self.m + step - 1) // step * step  # Round up m to be multiple of step
        if new_m != self.m:
            logger.warning(
                f"Adjusted `unique_sample_num` from {self.m} to {new_m} to make sure `unique_sample_num`*`group_size` is multiple of `batch_size`*`num_replicas` for even distribution."
            )
            self.m = new_m

        self.num_batches_per_epoch = (self.m * self.k) // self.sample_num_per_iteration

        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # Randomly select m unique samples, less if dataset is smaller than m
            indices = torch.randperm(_dataset_size(self.dataset), generator=g)[: self.m].tolist()

            # Repeat each sample k times to generate m*k total samples.
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            for i in range(self.num_batches_per_epoch):
                # Offset for current iteration
                offset = i * self.sample_num_per_iteration
                # Compute start and end indices for current replica
                start = offset + self.rank * self.batch_size
                end = start + self.batch_size
                yield shuffled_samples[start:end]

            # Increment epoch for next iteration
            self.epoch += 1

    def set_epoch(self, epoch: int):
        self.epoch = epoch  # Used to synchronize random state across epochs


class GroupContiguousSampler(Sampler):
    """
    Distributed sampler that keeps each group's k repeated samples
    contiguously on the SAME rank. Enables local groupwise reward
    computation without cross-rank communication.

    Constraint: m must be divisible by num_replicas (auto-enforced
    when any reward model has async_reward=True).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        group_size: int,
        unique_sample_num: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = group_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.m = unique_sample_num

        dataset_size = _dataset_size(self.dataset)
        if unique_sample_num > dataset_size:
            raise ValueError(
                f"`unique_sample_num` ({unique_sample_num}) must be <= dataset size ({dataset_size})."
            )

        if self.m % self.num_replicas != 0:
            raise ValueError(
                f"unique_sample_num ({self.m}) must be divisible by "
                f"num_replicas ({self.num_replicas}) for GroupContiguousSampler. "
                f"Set async_reward=True on a reward model config to auto-adjust."
            )

        self.groups_per_rank = self.m // self.num_replicas
        samples_per_rank = self.groups_per_rank * self.k
        if samples_per_rank % self.batch_size != 0:
            raise ValueError(
                f"groups_per_rank * group_size ({samples_per_rank}) must be "
                f"divisible by batch_size ({self.batch_size})"
            )

        self.num_batches_per_epoch = samples_per_rank // self.batch_size
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(_dataset_size(self.dataset), generator=g)[: self.m].tolist()

            # Shuffle group order (all ranks see the same permutation)
            group_perm = torch.randperm(self.m, generator=g).tolist()
            shuffled_groups = [indices[i] for i in group_perm]

            # Each rank gets a contiguous block of complete groups
            start_g = self.rank * self.groups_per_rank
            my_groups = shuffled_groups[start_g : start_g + self.groups_per_rank]

            # Expand: each group index repeated k times, groups stay contiguous
            my_samples = [gidx for gidx in my_groups for _ in range(self.k)]

            for i in range(self.num_batches_per_epoch):
                yield my_samples[i * self.batch_size : (i + 1) * self.batch_size]

            self.epoch += 1

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class GroupDistributedSampler(Sampler):
    """Distributed sampler that packs complete groups into global microbatches.

    Unlike :class:`GroupContiguousSampler` (all ``K`` copies on one rank),
    The concatenation of one local micro-batch from every rank is
    **group-complete**: every selected group appears exactly ``K`` times
    across the ``W * B`` samples of the global microbatch. When ``K`` is a
    multiple of ``W`` the legacy equal-share layout is retained. Otherwise
    whole groups are packed into the global batch and rank-major slices are
    dealt to workers; a rank may then contain only a subset of its global
    microbatch's groups.

    Rank contract (public invariant — DGPO depends on this)
    -------------------------------------------------------
    In the equal-share layout every rank yields the same prompt sequence.
    In the packed layout, dense group identities must instead be derived
    from the gathered global microbatch. Both layouts guarantee exactly K
    members per group globally.

    DGPO and TDM-R1 gather only the integer prompt ids to derive a shared
    dense group-id space; full samples remain rank-local.

    Constraints
    -----------
    - ``(num_replicas * batch_size) % group_size == 0``: exact global
      micro-batch tiling.

    Both are pre-aligned by
    :meth:`flow_factory.hparams.Arguments._align_for_group_distributed` so
    end users don't need to hand-tune them.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        group_size: int,
        unique_sample_num: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = group_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.m = unique_sample_num

        dataset_size = _dataset_size(self.dataset)
        if unique_sample_num > dataset_size:
            raise ValueError(
                f"`unique_sample_num` ({unique_sample_num}) must be <= dataset size ({dataset_size})."
            )

        # Geometric constraints — pre-aligned by
        # ``Arguments._align_for_group_distributed``; assert here as
        # belt-and-suspenders.
        sample_num_per_iteration = self.num_replicas * self.batch_size
        assert sample_num_per_iteration % self.k == 0, (
            "GroupDistributedSampler requires `(num_replicas * batch_size) % group_size == 0`, "
            f"got {self.num_replicas} * {self.batch_size} = {sample_num_per_iteration}, "
            f"group_size={self.k}. "
            "Arguments._align_for_group_distributed should have enforced this."
        )

        self._equal_share_layout = self.k % self.num_replicas == 0
        self.copies_per_rank = self.k // self.num_replicas if self._equal_share_layout else None
        total_samples = self.m * self.k
        assert total_samples % self.num_replicas == 0, (
            "GroupDistributedSampler requires total samples divisible by num_replicas, "
            f"got unique_sample_num * group_size={total_samples}, "
            f"num_replicas={self.num_replicas}. "
            "Arguments._align_for_group_distributed should have enforced this."
        )
        samples_per_rank = total_samples // self.num_replicas
        assert samples_per_rank % self.batch_size == 0, (
            "GroupDistributedSampler requires local samples per rank divisible by batch_size, "
            f"got samples_per_rank={samples_per_rank}, batch_size={self.batch_size}. "
            "Arguments._align_for_group_distributed should have enforced this."
        )

        self.num_batches_per_epoch = samples_per_rank // self.batch_size
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(_dataset_size(self.dataset), generator=g)[: self.m].tolist()
            group_perm = torch.randperm(self.m, generator=g).tolist()
            shuffled_groups = [indices[i] for i in group_perm]

            if self._equal_share_layout:
                # Preserve the historic byte-for-byte local sequence when K/W
                # is integral.
                my_samples = [
                    group_idx for group_idx in shuffled_groups for _ in range(self.copies_per_rank)
                ]
            else:
                groups_per_global_batch = self.num_replicas * self.batch_size // self.k
                my_samples = []
                for start in range(0, self.m, groups_per_global_batch):
                    global_batch = [
                        group_idx
                        for group_idx in shuffled_groups[start : start + groups_per_global_batch]
                        for _ in range(self.k)
                    ]
                    rank_start = self.rank * self.batch_size
                    my_samples.extend(global_batch[rank_start : rank_start + self.batch_size])
            for i in range(self.num_batches_per_epoch):
                yield my_samples[i * self.batch_size : (i + 1) * self.batch_size]

            self.epoch += 1

    def set_epoch(self, epoch: int):
        self.epoch = epoch
