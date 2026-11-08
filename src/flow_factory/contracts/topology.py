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

"""Physical rank dimensions and logical sample ownership, without backend state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

CollectiveScope = Literal["data", "tensor", "context", "sample"]


@dataclass(frozen=True)
class ParallelTopology:
    """Describe one canonical ``(dp_replicate, dp_shard, cp, tp)`` rank layout.

    Each data coordinate owns distinct samples. All TP x CP peers at that data
    coordinate own the same samples; sequence parallelism reuses TP and does
    not add a dimension. Sample counts and feedback statistics reduce across
    the data scope, while backend gradient synchronization may use a different
    scope (for example FSDP's joint ``dp_shard_cp`` mesh).

    This contract describes geometry. A valid geometry does not imply that a
    backend or adapter implements every dimension; those capability gates run
    before model loading.
    """

    dp_replicate_size: int = 1
    dp_shard_size: int = 1
    cp_size: int = 1
    tp_size: int = 1

    def __post_init__(self) -> None:
        for name, value in self.dimensions.items():
            _positive_integer(value, f"parallelism.{name}")

    @classmethod
    def resolve(
        cls,
        world_size: int,
        *,
        dp_replicate_size: Optional[int] = None,
        dp_shard_size: Optional[int] = None,
        tp_size: Optional[int] = None,
        cp_size: Optional[int] = None,
    ) -> ParallelTopology:
        """Resolve omitted degrees and validate the complete launched product.

        Only ``dp_shard_size`` is inferred from the remaining ranks. Other
        omitted dimensions mean one, independent of launcher environment
        defaults. This is the same rule used by the configuration and runtime
        translation paths.
        """
        _positive_integer(world_size, "world_size")
        replicate = 1 if dp_replicate_size is None else dp_replicate_size
        tensor = 1 if tp_size is None else tp_size
        context = 1 if cp_size is None else cp_size
        for name, value in (
            ("dp_replicate_size", replicate),
            ("tp_size", tensor),
            ("cp_size", context),
        ):
            _positive_integer(value, f"parallelism.{name}")
        reserved = replicate * tensor * context
        if dp_shard_size is None:
            if world_size % reserved:
                raise ValueError(
                    f"parallelism dp_replicate_size({replicate}) * tp_size({tensor}) * "
                    f"cp_size({context}) must divide the launched world size {world_size} "
                    f"to derive dp_shard_size; received remainder {world_size % reserved}."
                )
            dp_shard_size = world_size // reserved
        topology = cls(replicate, dp_shard_size, context, tensor)
        if topology.world_size != world_size:
            raise ValueError(
                f"parallelism dimensions {topology.dimensions} have product "
                f"{topology.world_size}, but the launched world size is {world_size}. "
                "Explicit sizes are never overridden. Adjust the launch process count or "
                "parallelism dimensions; omit dp_shard_size or set it to null to infer it "
                "when dp_replicate_size * tp_size * cp_size divides the launched world."
            )
        return topology

    @property
    def dimensions(self) -> dict[str, int]:
        """Return explicit degrees in the checkpoint schema's stable order."""
        return {
            "dp_replicate_size": self.dp_replicate_size,
            "dp_shard_size": self.dp_shard_size,
            "tp_size": self.tp_size,
            "cp_size": self.cp_size,
        }

    @property
    def world_size(self) -> int:
        """Return the number of physical ranks."""
        return self.data_parallel_size * self.model_parallel_size

    @property
    def data_parallel_size(self) -> int:
        """Return the number of distinct logical sample replicas."""
        return self.dp_replicate_size * self.dp_shard_size

    @property
    def model_parallel_size(self) -> int:
        """Return the number of TP x CP peers sharing each logical sample."""
        return self.tp_size * self.cp_size

    def coordinates(self, rank: int) -> tuple[int, int, int, int]:
        """Return ``(dp_replicate, dp_shard, cp, tp)`` for a physical rank."""
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ValueError(f"expected rank in [0, {self.world_size}), received {rank!r}.")
        remaining, tensor = divmod(rank, self.tp_size)
        remaining, context = divmod(remaining, self.cp_size)
        replicate, shard = divmod(remaining, self.dp_shard_size)
        return replicate, shard, context, tensor

    def data_parallel_rank(self, rank: int) -> int:
        """Return the logical data rank used for sampling and replica RNG."""
        replicate, shard, _, _ = self.coordinates(rank)
        return replicate * self.dp_shard_size + shard

    def group_ranks(self, rank: int, scope: CollectiveScope) -> tuple[int, ...]:
        """Return physical members of a semantic scope in collective order.

        ``data`` holds one representative of each sample replica at a fixed
        TP/CP coordinate. ``sample`` holds all peers of one sample replica.
        ``tensor`` and ``context`` vary only their named coordinate.
        """
        _, _, context, tensor = self.coordinates(rank)
        data_rank = self.data_parallel_rank(rank)
        sample_start = data_rank * self.model_parallel_size
        if scope == "data":
            offset = context * self.tp_size + tensor
            return tuple(
                replica * self.model_parallel_size + offset
                for replica in range(self.data_parallel_size)
            )
        if scope == "sample":
            return tuple(range(sample_start, sample_start + self.model_parallel_size))
        if scope == "tensor":
            start = sample_start + context * self.tp_size
            return tuple(range(start, start + self.tp_size))
        if scope == "context":
            return tuple(
                sample_start + peer * self.tp_size + tensor for peer in range(self.cp_size)
            )
        raise ValueError(
            f"unknown collective scope {scope!r}; expected data/tensor/context/sample."
        )


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, received {value!r}.")
