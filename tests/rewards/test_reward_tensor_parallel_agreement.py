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

"""Every path that hands rewards to a trainer passes them through peer agreement.

The agreement itself runs on real process groups in
``tests/distributed/test_tensor_parallel_reward_agreement.py``; here it is
replaced by a marker so each exit -- a streamed tile, the streamed cycle's
reassembly, and the full-acquisition finalize -- can be checked to return and
store the agreed values rather than the locally computed ones.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch

from flow_factory.hparams import RewardArguments
from flow_factory.rewards import PointwiseRewardModel, RewardBuffer, RewardModelOutput
from flow_factory.rewards import RewardProcessor
from flow_factory.rewards import reward_processor as reward_processor_module
from flow_factory.samples import BaseSample

AGREED_OFFSET = 100.0


class IndexReward(PointwiseRewardModel):
    """Scores ``prompt-<n>`` as ``n``."""

    required_fields = ("prompt",)

    def __call__(self, prompt: List[str]) -> RewardModelOutput:
        return RewardModelOutput(
            rewards=torch.tensor([float(text.rsplit("-", 1)[1]) for text in prompt])
        )


def _accelerator() -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        process_index=0,
        num_processes=1,
        is_local_main_process=False,
        wait_for_everyone=lambda: None,
    )


def _processor(*, async_reward: bool) -> RewardProcessor:
    accelerator = _accelerator()
    config = RewardArguments(
        name="index",
        reward_model="test.IndexReward",
        device="cpu",
        batch_size=2,
        async_reward=async_reward,
        num_workers=1,
    )
    return RewardProcessor(
        accelerator=accelerator,
        reward_models={"index": IndexReward(config, accelerator)},
        reward_configs={"index": config},
        group_on_same_rank=True,
        verbose=False,
    )


def _samples() -> List[BaseSample]:
    return [BaseSample(prompt=f"prompt-{index}") for index in range(4)]


@pytest.fixture(autouse=True)
def marked_agreement(monkeypatch: pytest.MonkeyPatch) -> None:
    def agree(accelerator: object, values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        del accelerator
        return {name: value + AGREED_OFFSET for name, value in values.items()}

    monkeypatch.setattr(reward_processor_module, "agree_across_tensor_parallel_peers", agree)


def _stored(samples: List[BaseSample]) -> List[float]:
    return [float(sample.extra_kwargs["rewards"]["index"]) for sample in samples]


def test_compute_rewards_returns_and_stores_agreed_values() -> None:
    samples = _samples()

    rewards = _processor(async_reward=False).compute_rewards(samples, split="pointwise")

    expected = [index + AGREED_OFFSET for index in range(4)]
    assert rewards["index"].tolist() == expected
    assert _stored(samples) == expected


def test_finalize_returns_and_stores_agreed_values() -> None:
    samples = _samples()
    buffer = RewardBuffer(_processor(async_reward=False), group_size=2)
    buffer.add_samples(samples)

    rewards = buffer.finalize(store_to_samples=True, split="all")

    expected = [index + AGREED_OFFSET for index in range(4)]
    assert rewards["index"].tolist() == expected
    assert _stored(samples) == expected


def test_streamed_tiles_and_their_reassembly_use_agreed_values() -> None:
    samples = _samples()
    buffer = RewardBuffer(_processor(async_reward=True), group_size=2)
    buffer.add_samples(samples)
    buffer.seal_for_streaming()
    tiles = {0: (0, 1), 1: (2, 3)}
    try:
        for tile_id, indices in tiles.items():
            deadline = time.monotonic() + 5.0
            while tile_id not in buffer.poll_ready_tiles({tile_id: indices}):
                assert time.monotonic() < deadline, f"reward tile {tile_id} never became ready"
                time.sleep(0.01)
            tile_rewards = buffer.resolve_streaming_tile(indices)
            assert tile_rewards["index"].tolist() == [i + AGREED_OFFSET for i in indices]

        full = buffer.finish_streaming()
    finally:
        buffer.shutdown(wait=True)

    expected = [index + AGREED_OFFSET for index in range(4)]
    assert full["index"].tolist() == expected
    assert _stored(samples) == expected
