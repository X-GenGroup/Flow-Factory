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

"""Logical evaluation rows and bounded distributed reward statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from ...utils.dist import data_parallel_geometry, data_parallel_process_group


@dataclass(frozen=True)
class EvaluationBatchPlan:
    """Identify padding from the existing sequential, evenly sharded loader."""

    dataset_size: int
    batch_size: int
    replicas: int
    rank: int

    @classmethod
    def from_loader(cls, loader: DataLoader, accelerator: Any) -> EvaluationBatchPlan:
        replicas, rank = data_parallel_geometry(accelerator)
        sampler = loader.batch_sampler
        if isinstance(sampler, BatchSamplerShard):
            if (
                sampler.num_processes != replicas
                or sampler.process_index != rank
                or sampler.split_batches
                or not sampler.even_batches
            ):
                raise ValueError(
                    "evaluation requires evenly sharded complete batches over logical DP; "
                    f"expected replicas/rank={replicas}/{rank}, received "
                    f"{sampler.num_processes}/{sampler.process_index}, "
                    f"split_batches={sampler.split_batches}, even_batches={sampler.even_batches}"
                )
            sampler = sampler.batch_sampler
        elif replicas != 1:
            raise ValueError("evaluation loader must be prepared over logical DP before generation")
        if (
            not isinstance(sampler, BatchSampler)
            or not isinstance(sampler.sampler, SequentialSampler)
            or sampler.drop_last
        ):
            raise ValueError(
                "evaluation requires a finite sequential BatchSampler with drop_last=false; "
                f"received {type(sampler).__name__}"
            )
        return cls(len(loader.dataset), sampler.batch_size, replicas, rank)

    def valid_count(self, batch_index: int) -> int:
        """Keep real rows while every model peer still executes padded forwards."""
        if batch_index < 0:
            raise ValueError(f"evaluation batch_index must be nonnegative, received {batch_index}")
        first = (batch_index * self.replicas + self.rank) * self.batch_size
        return min(self.batch_size, max(0, self.dataset_size - first))


def summarize_evaluation_rewards(
    accelerator: Any, rewards: Mapping[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    """Gather five FP64 values per reward/replica and merge in fixed rank order.

    Each local vector contains only real dataset rows. Welford's merge avoids
    subtracting two large squared sums, and empty replicas contribute no weight.
    NaNs from an inapplicable real reward retain the existing undefined-statistic
    behavior; loader padding never enters these vectors.
    """
    if not rewards:
        return {}
    names = sorted(rewards)
    local = torch.empty((len(names), 5), dtype=torch.float64, device=accelerator.device)
    for index, name in enumerate(names):
        values = torch.as_tensor(rewards[name], device=accelerator.device).detach().double()
        if values.ndim != 1:
            raise ValueError(
                f"evaluation reward {name!r} must be a vector, received {values.shape}"
            )
        count = values.numel()
        local[index, 0] = count
        if count:
            variance, mean = torch.var_mean(values, correction=0)
            local[index, 1] = mean
            local[index, 2] = variance * count
            local[index, 3] = values.min()
            local[index, 4] = values.max()
        else:
            local[index, 1:3] = 0
            local[index, 3] = math.inf
            local[index, 4] = -math.inf
    replicas, _ = data_parallel_geometry(accelerator)
    if replicas == 1:
        gathered = [local]
    else:
        gathered = [torch.empty_like(local) for _ in range(replicas)]
        dist.all_gather(gathered, local, group=data_parallel_process_group(accelerator))
    rows = torch.stack(gathered).cpu().tolist()
    result = {}
    for index, name in enumerate(names):
        count, mean, m2, minimum, maximum = 0.0, 0.0, 0.0, math.inf, -math.inf
        undefined = False
        for replica in rows:
            n, other_mean, other_m2, low, high = replica[index]
            if not n:
                continue
            undefined |= any(math.isnan(value) for value in (other_mean, other_m2, low, high))
            total = count + n
            delta = other_mean - mean
            m2 += other_m2 + delta * delta * count * n / total
            mean += delta * n / total
            count = total
            minimum, maximum = min(minimum, low), max(maximum, high)
        stats = {"count": count}
        if not count or undefined:
            stats.update({key: float("nan") for key in ("mean", "std", "min", "max")})
        else:
            stats.update(mean=mean, std=math.sqrt(max(0.0, m2 / count)), min=minimum, max=maximum)
        result[name] = stats
    return result
