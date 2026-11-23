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

"""Resolve auxiliary execution requests against the existing loading identities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeVar

from ..contracts.execution import FeedbackMode
from ..hparams.component_execution import ComponentExecutionArguments, ComponentExecutionPolicy
from ..loading import ComponentRole, build_adapter_load_plan
from ..utils.dist import parallel_topology

if TYPE_CHECKING:
    from accelerate import Accelerator

    from ..hparams.args import Arguments
    from .abc import BaseAdapter


class OptimizationGradients(str, Enum):
    """Derivatives required while a component participates in optimization."""

    NONE = "none"
    INPUT = "input"
    PARAMETERS = "parameters"


_REPLICATED = ComponentExecutionPolicy()
_Root = TypeVar("_Root", str, tuple)


@dataclass(frozen=True)
class ComponentExecution:
    """Resolved policy for a physical root.

    A null policy delegates placement to the trainable model backend. The
    gradient requirement applies to optimization. Preprocessing, rollout and
    evaluation retain their existing no-grad paths.
    """

    policy: ComponentExecutionPolicy | None
    optimization_gradients: OptimizationGradients


@dataclass(frozen=True)
class ComponentExecutionPlan:
    """Model load roots and reward deduplication keys for one configuration."""

    model: Mapping[str, ComponentExecution]
    rewards: Mapping[tuple, ComponentExecution]


def _bind_policy(
    policies: dict[_Root, tuple[str, ComponentExecutionPolicy]],
    root: _Root,
    label: str,
    policy: ComponentExecutionPolicy,
) -> None:
    if root in policies:
        previous_label, previous_policy = policies[root]
        if policy != previous_policy:
            raise ValueError(
                f"{label} and {previous_label} refer to one physical component and request "
                f"conflicting execution policies: {policy!r} and {previous_policy!r}"
            )
    policies[root] = (label, policy)


def resolve_reward_execution(
    config: Arguments, requests: ComponentExecutionArguments
) -> Mapping[tuple, ComponentExecution]:
    """Bind train and eval selectors using the same keys as MultiRewardLoader."""
    execution = config._get_execution_contract()
    binding: dict[tuple, tuple[str, ComponentExecutionPolicy]] = {}
    gradients: dict[tuple, OptimizationGradients] = {}
    for section, rewards, requested in (
        ("rewards", config.reward_args, requests.rewards),
        ("eval_rewards", config.eval_reward_args or config.reward_args, requests.eval_rewards),
    ):
        identities = {}
        for reward in rewards:
            if reward.reward_model is None:
                continue
            key = reward.get_identity_key()
            identities[reward.name] = key
            gradients.setdefault(key, OptimizationGradients.NONE)
            if section == "rewards" and execution.feedback is FeedbackMode.DIFFERENTIABLE_REWARD:
                gradients[key] = OptimizationGradients.INPUT
        for name, policy in requested.items():
            label = f"component_execution.{section}.{name}"
            if name not in identities:
                raise ValueError(f"{label} is unknown, available rewards: {sorted(identities)}")
            _bind_policy(binding, identities[name], label, policy)
    return MappingProxyType(
        {
            key: ComponentExecution(binding[key][1] if key in binding else _REPLICATED, gradient)
            for key, gradient in gradients.items()
        }
    )


def resolve_component_execution(
    config: Arguments,
    adapter: BaseAdapter,
    requests: ComponentExecutionArguments,
    rewards: Mapping[tuple, ComponentExecution],
) -> ComponentExecutionPlan:
    """Bind model selectors and decoder gradient requirements to physical roots."""
    load_plan = build_adapter_load_plan(adapter)
    binding: dict[str, tuple[str, ComponentExecutionPolicy]] = {}
    for selector, policy in requests.model.items():
        names = adapter._resolve_component_names(selector)
        if not names:
            raise ValueError(f"component_execution.model.{selector} matches no components")
        for name in names:
            request = load_plan.request_for_component(name)
            if request.role is not ComponentRole.AUXILIARY:
                raise ValueError(
                    f"component_execution.model.{selector} resolves to {request.role.value} "
                    f"root {request.root!r}; auxiliary policies require a frozen auxiliary root"
                )
            _bind_policy(binding, request.root, f"component_execution.model.{selector}", policy)
    decode_roots = set()
    if config._get_execution_contract().feedback is FeedbackMode.DIFFERENTIABLE_REWARD:
        for name in adapter.differentiable_decode_modules:
            root = load_plan.request_for_component(name)
            if root.role is not ComponentRole.AUXILIARY:
                raise ValueError(
                    f"differentiable decoder {name!r} requires a frozen auxiliary root"
                )
            decode_roots.add(root.root)

    model = {}
    for root, request in load_plan.requests.items():
        if request.role is ComponentRole.TARGET:
            model[root] = ComponentExecution(None, OptimizationGradients.PARAMETERS)
        elif request.role is ComponentRole.AUXILIARY:
            policy = binding[root][1] if root in binding else _REPLICATED
            gradient = (
                OptimizationGradients.INPUT if root in decode_roots else OptimizationGradients.NONE
            )
            model[root] = ComponentExecution(policy, gradient)
    return ComponentExecutionPlan(MappingProxyType(model), rewards)


def validate_component_execution(
    config: Arguments, requests: ComponentExecutionArguments, accelerator: Accelerator
) -> None:
    """Admit explicit auxiliary placements before model construction."""
    for section in ("model", "rewards", "eval_rewards"):
        for name, policy in getattr(requests, section).items():
            label = f"component_execution.{section}.{name}"
            if policy.placement == "fsdp2":
                topology = parallel_topology(accelerator)
                try:
                    policy.group_ranks(topology, accelerator.process_index)
                except ValueError as error:
                    raise ValueError(f"{label}: {error}") from error
                _validate_sharded_auxiliary(
                    config, accelerator, topology.model_parallel_size, section, name
                )
            if (
                policy.offload != "inherit"
                or policy.loading != "inherit"
                or policy.precision != "inherit"
            ):
                raise ValueError(
                    f"{label} requests {policy!r}. Use inherited loading, offload, and precision; "
                    "execution overrides require a supported auxiliary preparation path."
                )


def _validate_sharded_auxiliary(
    config: Arguments, accelerator: Accelerator, model_parallel_size: int, section: str, name: str
) -> None:
    label = f"component_execution.{section}.{name}"
    if (
        not getattr(accelerator, "is_fsdp2", False)
        or model_parallel_size != 1
        or config.training_args.trainer_type != "refl"
        or config.model_args.model_type.lower() != "sd3-5"
    ):
        raise ValueError(
            f"{label} has no supported auxiliary preparation path. "
            "FSDP2 auxiliary placement currently requires SD3.5 ReFL with a data-only FSDP2 launch."
        )
    if section == "model" and name == "vae":
        return
    if section != "rewards":
        raise ValueError(f"{label} requires an SD3.5 VAE or a synchronous training PickScore")
    reward = config.reward_args.get_by_name(name)
    if (
        reward.reward_model.lower() != "pickscore"
        or reward.async_reward
        or str(reward.device) != accelerator.device.type
        or len(config.data_args.training_datasets) != 1
    ):
        raise ValueError(
            f"{label} requires synchronous PickScore on the launch device and one training source"
        )
    eval_rewards = config.eval_reward_args or config.reward_args
    if config.data_args.eval_datasets and any(
        candidate.get_identity_key() == reward.get_identity_key() for candidate in eval_rewards
    ):
        raise ValueError(
            f"{label} shares its sharded scorer with evaluation. "
            "Configure a distinct replicated evaluation scorer."
        )
