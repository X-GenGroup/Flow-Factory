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

# src/flow_factory/rewards/__init__.py
"""
Reward models module for evaluating generated content.

Provides interfaces for single and multi-reward model loading and evaluation.
"""

from .abc import (
    BaseRewardModel,
    GroupwiseRewardModel,
    PointwiseRewardModel,
    RewardModelOutput,
)
from .loader import MultiRewardLoader, RewardModelHandle, load_reward_model
from .registry import get_reward_model_class, list_registered_reward_models
from .reward_processor import (
    RewardBuffer,
    RewardProcessor,
)
from .tile_plan import (
    RewardAccumulationScope,
    RewardGroupLayout,
    RewardTile,
    RewardTileGeometry,
    RewardTilePlan,
    build_reward_tile_plan,
    resolve_reward_tile_size,
)

__all__ = [
    # Base classes
    "BaseRewardModel",
    "PointwiseRewardModel",
    "GroupwiseRewardModel",
    "RewardModelOutput",
    "RewardProcessor",
    "RewardBuffer",
    "RewardAccumulationScope",
    "RewardGroupLayout",
    "RewardTile",
    "RewardTileGeometry",
    "RewardTilePlan",
    "build_reward_tile_plan",
    "resolve_reward_tile_size",
    # Registry
    "get_reward_model_class",
    "list_registered_reward_models",
    # Loaders
    "load_reward_model",
    "MultiRewardLoader",
    "RewardModelHandle",
]
