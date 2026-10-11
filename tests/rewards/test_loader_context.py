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

from contextlib import contextmanager
from types import SimpleNamespace

from flow_factory.hparams import MultiRewardArguments, RewardArguments
from flow_factory.rewards.loader import MultiRewardLoader


def test_reward_loader_enters_backend_context_once_per_unique_model(
    monkeypatch,
) -> None:
    events = []

    @contextmanager
    def load_context():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch.setattr(
        "flow_factory.rewards.loader.load_reward_model",
        lambda config, accelerator: SimpleNamespace(config=config),
    )
    config = RewardArguments(name="score", reward_model="clip")

    loader = MultiRewardLoader(
        reward_args=MultiRewardArguments(reward_configs=[config]),
        accelerator=SimpleNamespace(),
        load_context=load_context,
    ).load()

    assert loader.get_unique_model_count() == 1
    assert events == ["enter", "exit"]
