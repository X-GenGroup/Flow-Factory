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

"""Collective scope selected by a sampler layout."""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.distributed as dist
from accelerate import Accelerator

from ..contracts.sampler import SamplerLayoutContract, get_sampler_layout_contract
from .dist import data_parallel_geometry, data_parallel_process_group


class GroupCoordinator:
    """Route group-relative collectives to the smallest correct rank scope.

    The optimizer remains globally data parallel. Only reward payloads,
    identities, and group statistics use this scope, because ``subgroup_tile``
    guarantees that no comparison group crosses a subgroup boundary.

    Every scope here is counted in *data-parallel replicas*, never in launched
    ranks. The ranks of one tensor-parallel group hold the same sample, so
    counting them separately would gather each sample once per tensor-parallel
    rank and silently average duplicated rows into every group statistic.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        *,
        sampler_type: str,
        subgroup_size: Optional[int] = None,
    ) -> None:
        self.accelerator = accelerator
        self.layout: SamplerLayoutContract = get_sampler_layout_contract(sampler_type)
        self.dp_world_size, self.dp_rank = data_parallel_geometry(accelerator)
        self.process_group = None
        # None when the launched world is the replica set, which is every run
        # without a `parallelism` block; set only when ranks are reserved for
        # tensor parallelism, where Accelerate's world collectives are too wide.
        self.data_parallel_group = data_parallel_process_group(accelerator)

        if self.layout.group_placement == "rank_local":
            self.group_world_size = 1
            self.group_rank = 0
        elif self.layout.group_placement == "subgroup_tile":
            if self.data_parallel_group is not None:
                raise ValueError(
                    f"sampler layout {self.layout.name!r} builds its subgroups from "
                    f"contiguous launched ranks, which stops matching the replica set "
                    f"once ranks are reserved for tensor parallelism: "
                    f"data_parallel_size={self.dp_world_size}, "
                    f"num_processes={getattr(accelerator, 'num_processes', 1)}. Use the "
                    f"'rank_local', 'global_batch' or 'global_tile' layout, or disable "
                    f"tensor parallelism (parallelism.tp_size=1)."
                )
            self.group_world_size = self.layout.resolve_subgroup_size(
                num_replicas=self.dp_world_size,
                subgroup_size=subgroup_size,
            )
            self.group_rank = self.dp_rank % self.group_world_size
            if self.group_world_size < self.dp_world_size:
                self.process_group = self._create_subgroup()
        else:
            if subgroup_size is not None:
                raise ValueError(
                    f"sampler {self.layout.name!r} does not accept sampler_subgroup_size"
                )
            self.group_world_size = self.dp_world_size
            self.group_rank = self.dp_rank

    @property
    def groups_are_rank_local(self) -> bool:
        return self.layout.groups_are_rank_local

    @property
    def uses_global_collective(self) -> bool:
        return self.group_world_size == self.dp_world_size

    def _all_gather_data_parallel(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather along dim 0 across data-parallel replicas only.

        Mirrors ``accelerator.gather`` but over the data-parallel group, for runs
        where that is narrower than the launched world.
        """
        tensor = tensor.contiguous()
        gathered = torch.empty(
            (self.dp_world_size * tensor.shape[0], *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_gather_into_tensor(gathered, tensor, group=self.data_parallel_group)
        return gathered

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a fixed-shape tensor inside the group ownership scope."""

        if self.groups_are_rank_local:
            return tensor
        if self.uses_global_collective:
            if self.data_parallel_group is None:
                return self.accelerator.gather(tensor)
            return self._all_gather_data_parallel(tensor)
        if self.process_group is None or not dist.is_initialized():
            raise RuntimeError("subgroup gather requires initialized torch.distributed")
        tensor = tensor.contiguous()
        gathered = torch.empty(
            (self.group_world_size * tensor.shape[0], *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_gather_into_tensor(gathered, tensor, group=self.process_group)
        return gathered

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum a tensor inside the group ownership scope."""

        if self.groups_are_rank_local:
            return tensor
        if self.uses_global_collective:
            if self.data_parallel_group is None:
                return self.accelerator.reduce(tensor, reduction="sum")
            reduced = tensor.clone()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=self.data_parallel_group)
            return reduced
        if self.process_group is None or not dist.is_initialized():
            raise RuntimeError("subgroup reduction requires initialized torch.distributed")
        reduced = tensor.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=self.process_group)
        return reduced

    def gather_object(self, values: List[Any]) -> List[Any]:
        """Gather one rank-local object list inside the ownership scope."""

        if self.groups_are_rank_local:
            return list(values)
        if self.uses_global_collective:
            raise RuntimeError(
                "global object gathering remains owned by accelerate; "
                "gather_object is only for a strict subgroup"
            )
        if self.process_group is None or not dist.is_initialized():
            raise RuntimeError("subgroup object gather requires initialized torch.distributed")
        gathered: List[Optional[List[Any]]] = [None] * self.group_world_size
        dist.all_gather_object(gathered, list(values), group=self.process_group)
        return [
            value for rank_values in gathered if rank_values is not None for value in rank_values
        ]

    def _create_subgroup(self):
        if self.dp_world_size == 1:
            return None
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("subgroup_tile requires initialized torch.distributed")
        own_group = None
        for start in range(0, self.dp_world_size, self.group_world_size):
            ranks = list(range(start, start + self.group_world_size))
            process_group = dist.new_group(ranks=ranks)
            if self.dp_rank in ranks:
                own_group = process_group
        if own_group is None:  # pragma: no cover - rank geometry validated above
            raise RuntimeError(f"rank {self.dp_rank} was not assigned to a subgroup")
        return own_group


__all__ = ["GroupCoordinator"]
