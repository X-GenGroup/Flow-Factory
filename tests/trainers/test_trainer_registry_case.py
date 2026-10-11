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

"""Trainer registry keys stay case-insensitive for decorator registrations."""

import pytest

from flow_factory.trainers import registry
from flow_factory.trainers.rl.grpo import GRPOTrainer


def test_register_trainer_resolves_mixed_case_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Store decorator names lowercased so case-insensitive lookups find them."""
    monkeypatch.setattr(registry, "_TRAINER_REGISTRY", dict(registry._TRAINER_REGISTRY))

    registry.register_trainer("MyGRPO")(GRPOTrainer)

    assert registry._TRAINER_REGISTRY["mygrpo"] == "flow_factory.trainers.rl.grpo.GRPOTrainer"
    assert registry.get_trainer_class("MyGRPO") is GRPOTrainer
    assert registry.get_trainer_class("mygrpo") is GRPOTrainer


def test_register_trainer_rejects_non_string_names() -> None:
    """Fail at registration time instead of storing an unreachable key."""
    with pytest.raises(TypeError, match="expected str trainer name"):
        registry.register_trainer(None)
