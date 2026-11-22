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

"""Typed requests for the placement of frozen auxiliary components."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..contracts.topology import ParallelTopology
from .abc import ArgABC


@dataclass(frozen=True)
class ComponentExecutionPolicy:
    """Describe one auxiliary's weight placement and execution preferences."""

    placement: Literal["replicated", "fsdp2"] = "replicated"
    shard_group: Literal["data"] | None = None
    offload: Literal["inherit", "cpu"] = "inherit"
    loading: Literal["inherit", "sharded"] = "inherit"
    precision: Literal["inherit", "fp32", "bf16", "fp16"] = "inherit"

    def __post_init__(self) -> None:
        choices = {
            "placement": ("replicated", "fsdp2"),
            "shard_group": (None, "data"),
            "offload": ("inherit", "cpu"),
            "loading": ("inherit", "sharded"),
            "precision": ("inherit", "fp32", "bf16", "fp16"),
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"component execution {name}={value!r}, expected one of {allowed}")
        if (self.placement == "fsdp2") != (self.shard_group is not None):
            raise ValueError(
                f"placement={self.placement!r} and shard_group={self.shard_group!r} are incompatible. "
                "FSDP2 requires an explicit shard_group and replication uses no group."
            )
        if self.loading == "sharded" and self.placement != "fsdp2":
            raise ValueError("loading='sharded' requires placement='fsdp2'")

    def group_ranks(self, topology: ParallelTopology, rank: int) -> tuple[int, ...] | None:
        """Resolve an existing logical scope without constructing a process group."""
        if self.shard_group is None:
            return None
        ranks = topology.group_ranks(rank, self.shard_group)
        if len(ranks) < 2:
            raise ValueError(
                f"shard_group={self.shard_group!r} requires at least two ranks, got {ranks} "
                f"from topology {topology.dimensions}"
            )
        return ranks


@dataclass
class ComponentExecutionArguments(ArgABC):
    """Select model components and named training or evaluation rewards."""

    model: dict[str, ComponentExecutionPolicy] = field(default_factory=dict)
    rewards: dict[str, ComponentExecutionPolicy] = field(default_factory=dict)
    eval_rewards: dict[str, ComponentExecutionPolicy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.extra_kwargs:
            raise ValueError(f"unknown component_execution sections: {sorted(self.extra_kwargs)}")
        for section in ("model", "rewards", "eval_rewards"):
            raw = getattr(self, section)
            if not isinstance(raw, Mapping):
                raise TypeError(f"component_execution.{section} must be a mapping, got {raw!r}")
            policies = {}
            for name, policy in raw.items():
                label = f"component_execution.{section}.{name}"
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"component_execution.{section} has invalid selector {name!r}")
                try:
                    policies[name] = (
                        policy
                        if isinstance(policy, ComponentExecutionPolicy)
                        else ComponentExecutionPolicy(**policy)
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{label}: {error}") from error
            setattr(self, section, policies)

    @classmethod
    def from_dict(cls, args_dict: Mapping[str, Any]) -> "ComponentExecutionArguments":
        """Parse the optional policy block with strict nested keys."""
        if not isinstance(args_dict, Mapping):
            raise TypeError(f"component_execution must be a mapping, got {args_dict!r}")
        return cls(**args_dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize only the declared sections."""
        return {
            section: {name: asdict(policy) for name, policy in policies.items()}
            for section in ("model", "rewards", "eval_rewards")
            if (policies := getattr(self, section))
        }
