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

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from flow_factory.hparams import RewardArguments
from flow_factory.rewards import (
    PointwiseRewardModel,
    RewardBuffer,
    RewardModelOutput,
    RewardProcessor,
    build_reward_tile_plan,
    resolve_reward_tile_size,
)
from flow_factory.samples import BaseSample


class GatedPointwiseReward(PointwiseRewardModel):
    """Pointwise test reward whose prompt groups can complete independently."""

    required_fields = ("prompt",)

    def __init__(self, config: RewardArguments, accelerator: SimpleNamespace) -> None:
        super().__init__(config, accelerator)
        self.gates = {"group-0": threading.Event(), "group-1": threading.Event()}

    def __call__(self, prompt: list[str]) -> RewardModelOutput:
        group = prompt[0]
        if not self.gates[group].wait(timeout=2.0):
            raise TimeoutError(f"test reward gate did not open for {group}")
        value = float(group.rsplit("-", 1)[1])
        return RewardModelOutput(rewards=torch.full((len(prompt),), value))


def _accelerator() -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        process_index=0,
        num_processes=1,
        is_local_main_process=False,
        wait_for_everyone=lambda: None,
    )


def _grouped_samples() -> list[BaseSample]:
    return [
        BaseSample(prompt="group-0"),
        BaseSample(prompt="group-0"),
        BaseSample(prompt="group-1"),
        BaseSample(prompt="group-1"),
    ]


def _wait_until_ready(buffer: RewardBuffer, tile_indices: dict[int, tuple[int, ...]]) -> set[int]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        ready = buffer.poll_ready_tiles(tile_indices)
        if ready:
            return ready
        time.sleep(0.01)
    raise AssertionError("reward tile did not become ready")


def test_tile_plan_closes_groups_and_optimizer_windows() -> None:
    samples = [BaseSample(prompt=f"group-{group}") for group in range(6) for _ in range(4)]

    plan = build_reward_tile_plan(
        samples,
        group_size=4,
        per_device_batch_size=3,
        gradient_accumulation_steps=4,
        optimizer_terms_per_batch=2,
    )

    assert plan.samples_per_tile == 12
    assert plan.batches_per_tile == 4
    assert [tile.sample_count for tile in plan.tiles] == [12, 12]
    assert [len(tile.group_ids) for tile in plan.tiles] == [3, 3]


def test_tile_plan_rejects_non_contiguous_groups() -> None:
    samples = [
        BaseSample(prompt="group-0"),
        BaseSample(prompt="group-1"),
        BaseSample(prompt="group-0"),
        BaseSample(prompt="group-1"),
    ]

    with pytest.raises(ValueError, match="group_contiguous"):
        build_reward_tile_plan(
            samples,
            group_size=2,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            optimizer_terms_per_batch=1,
        )


def test_online_dpo_tile_geometry_counts_pairs_instead_of_raw_samples() -> None:
    samples_per_tile, batches_per_tile = resolve_reward_tile_size(
        group_size=16,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,
        optimizer_terms_per_batch=2,
        optimizer_examples_per_group=1,
    )

    assert samples_per_tile == 32
    assert batches_per_tile == 2


def test_cross_rank_tile_geometry_supports_groups_smaller_than_world_size() -> None:
    samples = [BaseSample(prompt=f"rank-local-{index}") for index in range(2)]

    plan = build_reward_tile_plan(
        samples,
        group_size=16,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,
        optimizer_terms_per_batch=2,
        group_layout="cross_rank_sharded",
    )

    assert plan.samples_per_tile == 2
    assert plan.batches_per_tile == 2
    assert len(plan.tiles) == 1


def test_acquisition_scoped_geometry_streams_one_local_microbatch_per_tile() -> None:
    assert resolve_reward_tile_size(
        group_size=16,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        optimizer_terms_per_batch=4,
        group_layout="cross_rank_sharded",
        accumulation_scope="acquisition",
    ) == (1, 1)


def test_reward_buffer_resolves_tiles_in_completion_order_without_global_finalize() -> None:
    accelerator = _accelerator()
    config = RewardArguments(
        name="gated",
        reward_model="test.GatedPointwiseReward",
        device="cpu",
        batch_size=2,
        async_reward=True,
        num_workers=2,
    )
    model = GatedPointwiseReward(config, accelerator)
    processor = RewardProcessor(
        accelerator=accelerator,
        reward_models={"gated": model},
        reward_configs={"gated": config},
        group_on_same_rank=True,
        verbose=False,
    )
    buffer = RewardBuffer(processor, group_size=2)
    samples = _grouped_samples()
    buffer.add_samples(samples)
    buffer.seal_for_streaming()
    tile_indices = {0: (0, 1), 1: (2, 3)}

    # Complete the later tile first: readiness and resolution must not imply
    # rollout order. Trainer scheduling decides whether this tile may bypass 0.
    model.gates["group-1"].set()
    assert _wait_until_ready(buffer, tile_indices) == {1}
    second = buffer.resolve_streaming_tile(tile_indices[1])
    torch.testing.assert_close(second["gated"], torch.ones(2))
    assert "rewards" in samples[2].extra_kwargs
    with pytest.raises(RuntimeError, match="already consumed"):
        buffer.resolve_streaming_tile(tile_indices[1])
    assert buffer.poll_ready_tiles({0: tile_indices[0]}) == set()

    model.gates["group-0"].set()
    assert _wait_until_ready(buffer, {0: tile_indices[0]}) == {0}
    first = buffer.resolve_streaming_tile(tile_indices[0])
    torch.testing.assert_close(first["gated"], torch.zeros(2))

    full = buffer.finish_streaming()
    torch.testing.assert_close(full["gated"], torch.tensor([0.0, 0.0, 1.0, 1.0]))
    buffer.shutdown(wait=True)


def test_async_reward_components_have_independent_executor_lanes() -> None:
    accelerator = _accelerator()
    slow_config = RewardArguments(
        name="slow",
        reward_model="test.GatedPointwiseReward",
        device="cpu",
        batch_size=2,
        async_reward=True,
        num_workers=1,
    )
    fast_config = RewardArguments(
        name="fast",
        reward_model="test.GatedPointwiseReward",
        device="cpu",
        batch_size=2,
        async_reward=True,
        num_workers=1,
    )
    slow = GatedPointwiseReward(slow_config, accelerator)
    fast = GatedPointwiseReward(fast_config, accelerator)
    fast.gates["group-0"].set()
    fast.gates["group-1"].set()
    processor = RewardProcessor(
        accelerator=accelerator,
        reward_models={"slow": slow, "fast": fast},
        reward_configs={"slow": slow_config, "fast": fast_config},
        group_on_same_rank=True,
        verbose=False,
    )
    buffer = RewardBuffer(processor, group_size=2)
    samples = _grouped_samples()
    buffer.add_samples(samples)
    buffer.seal_for_streaming()
    tile_indices = {0: (0, 1), 1: (2, 3)}

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and any(reward is None for reward in buffer._rewards["fast"]):
        assert buffer.poll_ready_tiles(tile_indices) == set()
        time.sleep(0.01)
    assert all(reward is not None for reward in buffer._rewards["fast"])
    assert all(reward is None for reward in buffer._rewards["slow"])

    slow.gates["group-0"].set()
    slow.gates["group-1"].set()
    assert _wait_until_ready(buffer, tile_indices) == {0, 1}
    buffer.resolve_streaming_tile(tile_indices[0])
    buffer.resolve_streaming_tile(tile_indices[1])
    buffer.finish_streaming()
    buffer.shutdown(wait=True)


def test_pointwise_reward_requests_never_cross_streaming_tile_boundaries() -> None:
    accelerator = _accelerator()
    config = RewardArguments(
        name="gated",
        reward_model="test.GatedPointwiseReward",
        device="cpu",
        batch_size=3,
        async_reward=True,
        num_workers=1,
    )
    model = GatedPointwiseReward(config, accelerator)
    model.gates["group-0"].set()
    model.gates["group-1"].set()
    processor = RewardProcessor(
        accelerator=accelerator,
        reward_models={"gated": model},
        reward_configs={"gated": config},
        group_on_same_rank=True,
        verbose=False,
    )
    buffer = RewardBuffer(processor, group_size=2)
    buffer.configure_streaming_tiles(samples_per_tile=2)
    buffer.add_samples(_grouped_samples())
    buffer.seal_for_streaming()

    submitted_indices = [indices for name, indices, _future in buffer._futures if name == "gated"]
    assert submitted_indices == [[0, 1], [2, 3]]

    tile_indices = {0: (0, 1), 1: (2, 3)}
    assert _wait_until_ready(buffer, tile_indices) == {0, 1}
    buffer.resolve_streaming_tile(tile_indices[0])
    buffer.resolve_streaming_tile(tile_indices[1])
    buffer.finish_streaming()
    buffer.shutdown(wait=True)


def test_streaming_rejects_a_synchronous_reward_lane() -> None:
    accelerator = _accelerator()
    config = RewardArguments(
        name="sync",
        reward_model="test.GatedPointwiseReward",
        device="cpu",
        batch_size=2,
        async_reward=False,
    )
    model = GatedPointwiseReward(config, accelerator)
    processor = RewardProcessor(
        accelerator=accelerator,
        reward_models={"sync": model},
        reward_configs={"sync": config},
        group_on_same_rank=True,
        verbose=False,
    )
    buffer = RewardBuffer(processor, group_size=2)
    buffer.add_samples(_grouped_samples())

    with pytest.raises(RuntimeError, match="every training reward"):
        buffer.seal_for_streaming()


def test_standard_async_finalize_remains_compatible() -> None:
    accelerator = _accelerator()
    config = RewardArguments(
        name="gated",
        reward_model="test.GatedPointwiseReward",
        device="cpu",
        batch_size=2,
        async_reward=True,
        num_workers=2,
    )
    model = GatedPointwiseReward(config, accelerator)
    model.gates["group-0"].set()
    model.gates["group-1"].set()
    processor = RewardProcessor(
        accelerator=accelerator,
        reward_models={"gated": model},
        reward_configs={"gated": config},
        group_on_same_rank=True,
        verbose=False,
    )
    buffer = RewardBuffer(processor, group_size=2)
    samples = _grouped_samples()
    buffer.add_samples(samples)

    full = buffer.finalize()

    torch.testing.assert_close(full["gated"], torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert all(sample.extra_kwargs["rewards"]["gated"] is not None for sample in samples)
    buffer.shutdown(wait=True)
