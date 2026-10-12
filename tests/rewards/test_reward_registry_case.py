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

"""Reward registry keys stay case-insensitive for decorator registrations."""

import pytest

from flow_factory.rewards import registry
from flow_factory.rewards.my_reward import MyPointwiseRewardModel


def test_register_reward_model_resolves_mixed_case_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store decorator names lowercased so case-insensitive lookups find them."""
    monkeypatch.setattr(registry, "_REWARD_MODEL_REGISTRY", dict(registry._REWARD_MODEL_REGISTRY))

    registry.register_reward_model("MyPointwise")(MyPointwiseRewardModel)

    assert registry._REWARD_MODEL_REGISTRY["mypointwise"] == (
        "flow_factory.rewards.my_reward.MyPointwiseRewardModel"
    )
    assert registry.get_reward_model_class("MyPointwise") is MyPointwiseRewardModel


def test_register_reward_model_rejects_non_string_names() -> None:
    """Fail at registration time instead of storing an unreachable key."""
    with pytest.raises(TypeError, match="expected str reward model name"):
        registry.register_reward_model(3)
