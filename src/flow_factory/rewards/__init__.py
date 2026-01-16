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
    RewardModelOutput,
    BaseRewardModel,
    PointwiseRewardModel,
    GroupwiseRewardModel,
)
from .reward_processor import (
    RewardProcessor,
)
from .registry import get_reward_model_class, list_registered_reward_models
from .loader import load_reward_model, MultiRewardLoader, RewardModelHandle


__all__ = [
    # Base classes
    'BaseRewardModel',
    'PointwiseRewardModel',
    'GroupwiseRewardModel',
    'RewardModelOutput',
    'RewardProcessor',
    # Registry
    'get_reward_model_class',
    'list_registered_reward_models',
    # Loaders
    'load_reward_model',
    'MultiRewardLoader',
    'RewardModelHandle',
]