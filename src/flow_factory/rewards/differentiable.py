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

"""Optimization-stage scoring with frozen networks and live input derivatives."""

from dataclasses import dataclass
from typing import Mapping

import torch

from ..contracts.reward import RewardGradientCapability
from ..hparams.reward_args import RewardArguments
from .abc import BaseRewardModel, PointwiseRewardModel
from .registry import get_reward_model_class


def validate_differentiable_rewards(
    configs, input_fields: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    """Validate model declarations and scheduling before constructing networks."""
    if not input_fields or not configs:
        raise ValueError("differentiable feedback requires input fields and training rewards")
    selected = {}
    for config in configs:
        try:
            cls = get_reward_model_class(config.reward_model)
        except ImportError as error:
            raise ValueError(
                f"reward {config.name!r} ({config.reward_model}) cannot establish required "
                f"input gradients for {input_fields!r}: {error}"
            ) from error
        capability = getattr(cls, "gradient_capability", None)
        if (
            not isinstance(capability, RewardGradientCapability)
            or not capability.select_input_fields(input_fields)
            or cls.score_differentiable is BaseRewardModel.score_differentiable
        ):
            raise ValueError(
                f"reward {config.name!r} ({cls.__name__}) requires input gradients for "
                f"{input_fields!r}, declared capability={capability!r}; use eval_rewards for monitoring"
            )
        if not issubclass(cls, PointwiseRewardModel):
            raise ValueError(f"reward {config.name!r} needs complete-group differentiable dispatch")
        if type(config.batch_size) is not int or config.batch_size < 1:
            raise ValueError(
                f"reward {config.name!r} requires positive batch_size, received {config.batch_size!r}"
            )
        if config.async_reward:
            raise ValueError(
                f"reward {config.name!r} requires synchronous optimization-stage scoring; async_reward=True"
            )
        selected[config.name] = capability.select_input_fields(input_fields)
    return selected


@dataclass(frozen=True)
class DifferentiableRewardOutput:
    """Per-sample scores and source-specific weights, zero outside applicability."""

    scores: dict[str, torch.Tensor]
    weights: dict[str, torch.Tensor]


class DifferentiableRewardProcessor:
    """Route pointwise differentiable calls on the optimization thread.

    Configuration owns source applicability and weights. Models own scoring.
    Algorithms own objective transforms and reduction. Frozen train/eval-shared
    networks are prepared once and have no optimizer state.
    """

    def __init__(
        self,
        models: Mapping[str, BaseRewardModel],
        configs: Mapping[str, RewardArguments],
        input_fields: tuple[str, ...],
    ):
        self.reward_input_fields = validate_differentiable_rewards(
            list(configs.values()), input_fields
        )
        self.models = dict(models)
        self.configs = dict(configs)
        self.input_fields = input_fields
        for model in self.models.values():
            if model.model is not None:
                model.model.eval().requires_grad_(False)

    def __call__(
        self, *, prompt: list[str], source_names: list[str], **media: torch.Tensor
    ) -> DifferentiableRewardOutput:
        """Return live scores without crossing the detached reward buffer."""
        if len(prompt) != len(source_names) or not prompt:
            raise ValueError("prompt and source_names must have equal nonzero batch length")
        if set(media) != set(self.input_fields):
            raise ValueError(
                f"expected differentiable media {self.input_fields!r}, received {tuple(media)!r}"
            )
        device = next(iter(media.values())).device
        required_fields = {
            field for fields in self.reward_input_fields.values() for field in fields
        }
        for name, value in media.items():
            if (
                value.shape[0] != len(prompt)
                or value.device != device
                or (name in required_fields and not value.requires_grad)
            ):
                raise ValueError(
                    f"{name} must carry input gradients, batch={len(prompt)}, device={device}"
                )
        scores, weights = {}, {}
        for name, model in self.models.items():
            config = self.configs[name]
            indices = [
                i
                for i, source in enumerate(source_names)
                if config.applicable_datasets is None or source in config.applicable_datasets
            ]
            values = []
            for start in range(0, len(indices), config.batch_size):
                chunk = indices[start : start + config.batch_size]
                index = torch.tensor(chunk, device=device)
                inputs = {
                    key: media[key].index_select(0, index) for key in self.reward_input_fields[name]
                }
                with (
                    torch.enable_grad(),
                    torch.autocast(device_type=model.device.type, enabled=False),
                ):
                    result = model.score_differentiable(
                        prompt=[prompt[i] for i in chunk], **inputs
                    ).rewards
                if not isinstance(result, torch.Tensor) or result.shape != (len(chunk),):
                    raise ValueError(
                        f"reward {name!r} must return a Tensor of shape {(len(chunk),)}"
                    )
                if not result.requires_grad or not torch.isfinite(result).all():
                    raise ValueError(
                        f"reward {name!r} returned detached or non-finite differentiable scores"
                    )
                values.append(result.to(device=device, dtype=torch.float32))
            index = torch.tensor(indices, device=device, dtype=torch.long)
            base = torch.zeros(len(prompt), device=device)
            scores[name] = base.index_copy(0, index, torch.cat(values)) if values else base
            weights[name] = base.new_tensor(
                [
                    (
                        (
                            config.weight[source]
                            if isinstance(config.weight, dict)
                            else config.weight
                        )
                        if i in indices
                        else 0.0
                    )
                    for i, source in enumerate(source_names)
                ]
            )
        return DifferentiableRewardOutput(scores=scores, weights=weights)
