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

"""Model-parallel peers split the scoring, and every exit combines their shares.

The combine itself runs on real process groups in
``tests/distributed/test_tensor_parallel_reward_agreement.py``. Here it is
replaced by a marker so each exit -- a streamed tile, the streamed cycle's
reassembly, and the full-acquisition finalize -- can be checked to return and
store the combined values; and the split is checked on one process standing in
for the second peer of a pair: which samples it scores, and what it holds for
the ones its partner scores.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch

from flow_factory.hparams import RewardArguments
from flow_factory.rewards import (
    GroupwiseRewardModel,
    PointwiseRewardModel,
    RewardBuffer,
    RewardModelOutput,
    RewardProcessor,
)
from flow_factory.rewards import reward_processor as reward_processor_module
from flow_factory.samples import BaseSample

COMBINED_OFFSET = 100.0


class IndexReward(PointwiseRewardModel):
    """Scores ``prompt-<n>`` as ``n`` and records what it was asked to score."""

    required_fields = ("prompt",)
    scored: List[str] = []

    def __call__(self, prompt: List[str]) -> RewardModelOutput:
        IndexReward.scored.extend(prompt)
        return RewardModelOutput(
            rewards=torch.tensor([float(text.rsplit("-", 1)[1]) for text in prompt])
        )


class GroupIndexReward(GroupwiseRewardModel):
    """Scores a group by its members' indices and records the groups it saw."""

    required_fields = ("prompt",)
    scored: List[List[str]] = []

    def __call__(self, prompt: List[str]) -> RewardModelOutput:
        GroupIndexReward.scored.append(list(prompt))
        return RewardModelOutput(
            rewards=torch.tensor([float(text.rsplit("-", 1)[1]) for text in prompt])
        )


@pytest.fixture(autouse=True)
def fresh_records() -> None:
    IndexReward.scored = []
    GroupIndexReward.scored = []


def _accelerator() -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        process_index=0,
        num_processes=1,
        is_local_main_process=False,
        wait_for_everyone=lambda: None,
    )


def _processor(*, async_reward: bool, groupwise: bool = False) -> RewardProcessor:
    accelerator = _accelerator()
    model_class = GroupIndexReward if groupwise else IndexReward
    config = RewardArguments(
        name="index",
        reward_model=f"test.{model_class.__name__}",
        device="cpu",
        batch_size=2,
        async_reward=async_reward,
        num_workers=1,
    )
    return RewardProcessor(
        accelerator=accelerator,
        reward_models={"index": model_class(config, accelerator)},
        reward_configs={"index": config},
        group_on_same_rank=True,
        verbose=False,
    )


def _samples(count: int = 4) -> List[BaseSample]:
    return [BaseSample(prompt=f"prompt-{index}") for index in range(count)]


def _stored(samples: List[BaseSample]) -> List[float]:
    return [float(sample.extra_kwargs["rewards"]["index"]) for sample in samples]


@pytest.fixture
def marked_combine(monkeypatch: pytest.MonkeyPatch) -> None:
    def combine(accelerator: object, values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        del accelerator
        return {name: value + COMBINED_OFFSET for name, value in values.items()}

    monkeypatch.setattr(reward_processor_module, "combine_model_parallel_shares", combine)


@pytest.fixture
def second_of_two_peers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand this process in for peer 1 of a tensor-parallel pair; nothing combined."""
    monkeypatch.setattr(RewardProcessor, "_model_parallel_share", lambda self: (2, 1))
    monkeypatch.setattr(
        reward_processor_module,
        "combine_model_parallel_shares",
        lambda accelerator, values: dict(values),
    )


def _as_list(values: torch.Tensor) -> List[object]:
    return ["nan" if math.isnan(v) else v for v in values.tolist()]


@pytest.mark.usefixtures("marked_combine")
def test_compute_rewards_returns_and_stores_combined_values() -> None:
    samples = _samples()

    rewards = _processor(async_reward=False).compute_rewards(samples, split="pointwise")

    expected = [index + COMBINED_OFFSET for index in range(4)]
    assert rewards["index"].tolist() == expected
    assert _stored(samples) == expected


@pytest.mark.usefixtures("marked_combine")
def test_finalize_returns_and_stores_combined_values() -> None:
    samples = _samples()
    buffer = RewardBuffer(_processor(async_reward=False), group_size=2)
    buffer.add_samples(samples)

    rewards = buffer.finalize(store_to_samples=True, split="all")

    expected = [index + COMBINED_OFFSET for index in range(4)]
    assert rewards["index"].tolist() == expected
    assert _stored(samples) == expected


@pytest.mark.usefixtures("marked_combine")
def test_streamed_tiles_and_their_reassembly_use_combined_values() -> None:
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
            assert tile_rewards["index"].tolist() == [i + COMBINED_OFFSET for i in indices]

        full = buffer.finish_streaming()
    finally:
        buffer.shutdown(wait=True)

    expected = [index + COMBINED_OFFSET for index in range(4)]
    assert full["index"].tolist() == expected
    assert _stored(samples) == expected


@pytest.mark.usefixtures("second_of_two_peers")
def test_a_peer_scores_only_its_share_and_holds_zero_for_its_partners() -> None:
    samples = _samples(5)

    rewards = _processor(async_reward=False).compute_rewards(samples, split="pointwise")

    # Peer 1 of 2 owns the odd indices; its partner's rows are 0, ready to sum.
    assert IndexReward.scored == ["prompt-1", "prompt-3"]
    assert rewards["index"].tolist() == [0.0, 1.0, 0.0, 3.0, 0.0]
    # Aggregation reads applicability on every peer, scored here or not.
    assert all("index" in sample.applicable_rewards for sample in samples)


@pytest.mark.usefixtures("second_of_two_peers")
def test_a_partners_rows_are_nan_where_the_reward_does_not_apply() -> None:
    processor = _processor(async_reward=False)
    processor.reward_configs["index"].applicable_datasets = ["kept"]
    processor.reward_configs["index"]._datasets_resolved = None
    samples = _samples(4)
    for index, sample in enumerate(samples):
        sample.source = "kept" if index < 2 else "dropped"

    rewards = processor.compute_rewards(samples, split="pointwise")

    # Index 0 is the partner's and applies (0); 2 and 3 do not apply (NaN on
    # every peer, so the sum keeps NaN); 1 is scored here.
    assert _as_list(rewards["index"]) == [0.0, 1.0, "nan", "nan"]
    assert ["index" in s.applicable_rewards for s in samples] == [True, True, False, False]


@pytest.mark.usefixtures("second_of_two_peers")
def test_groups_are_split_whole_between_peers() -> None:
    samples = _samples(6)
    for index, sample in enumerate(samples):
        sample.sampling_group_id = index // 2

    rewards = _processor(async_reward=False, groupwise=True).compute_rewards(
        samples, split="groupwise"
    )

    # Three groups of two; peer 1 owns the second.
    assert GroupIndexReward.scored == [["prompt-2", "prompt-3"]]
    assert rewards["index"].tolist() == [0.0, 0.0, 2.0, 3.0, 0.0, 0.0]


@pytest.mark.usefixtures("second_of_two_peers")
def test_async_rewards_split_the_same_way_and_stream_with_placeholders() -> None:
    samples = _samples(4)
    buffer = RewardBuffer(_processor(async_reward=True), group_size=2)
    buffer.add_samples(samples)
    buffer.seal_for_streaming()
    try:
        deadline = time.monotonic() + 5.0
        while 0 not in buffer.poll_ready_tiles({0: (0, 1, 2, 3)}):
            assert time.monotonic() < deadline, "reward tile never became ready"
            time.sleep(0.01)
        tile = buffer.resolve_streaming_tile((0, 1, 2, 3))
        full = buffer.finish_streaming()
    finally:
        buffer.shutdown(wait=True)

    assert sorted(IndexReward.scored) == ["prompt-1", "prompt-3"]
    assert tile["index"].tolist() == [0.0, 1.0, 0.0, 3.0]
    assert full["index"].tolist() == [0.0, 1.0, 0.0, 3.0]


@pytest.mark.usefixtures("second_of_two_peers")
def test_a_peer_that_owns_no_sample_still_assembles_its_rows() -> None:
    """With one sample per rank, peer 1 of 2 scores nothing and holds no future."""
    samples = _samples(1)
    buffer = RewardBuffer(_processor(async_reward=True), group_size=1)
    buffer.add_samples(samples)
    try:
        rewards = buffer.finalize(store_to_samples=True)
    finally:
        buffer.shutdown(wait=True)

    assert IndexReward.scored == []
    assert rewards["index"].tolist() == [0.0]
