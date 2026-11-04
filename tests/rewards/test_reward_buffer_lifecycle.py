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

"""Bound async requests without moving failures ahead of peer agreement."""

import gc
import threading
import weakref
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from flow_factory.hparams import RewardArguments
from flow_factory.hparams.reward_args import MultiRewardArguments
from flow_factory.rewards import (
    GroupwiseRewardModel,
    PointwiseRewardModel,
    RewardBuffer,
    RewardModelOutput,
    RewardProcessor,
)
from flow_factory.rewards import reward_processor as reward_processor_module
from flow_factory.samples import BaseSample
from flow_factory.trainers.common.runtime_identity import _schema_digest


class _Pointwise(PointwiseRewardModel):
    required_fields = ("prompt",)

    def __call__(self, prompt):
        return self.compute(prompt)


class _Groupwise(GroupwiseRewardModel):
    required_fields = ("prompt",)

    def __call__(self, prompt):
        return self.compute(prompt)


def _buffer(*, groupwise=False, workers=1, limit=2, asynchronous=True):
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        process_index=0,
        num_processes=1,
        is_local_main_process=False,
        wait_for_everyone=lambda: None,
    )
    config = RewardArguments(
        name="score",
        device="cpu",
        async_reward=asynchronous,
        num_workers=workers,
        max_pending_requests=limit,
        batch_size=1,
    )
    model = (_Groupwise if groupwise else _Pointwise)(config, accelerator)
    model.compute = lambda prompt: RewardModelOutput(
        rewards=torch.tensor([float(text) for text in prompt])
    )
    processor = RewardProcessor(
        accelerator,
        {"score": model},
        {"score": config},
        group_on_same_rank=True,
        verbose=False,
    )
    return RewardBuffer(processor, group_size=1), model


@pytest.mark.parametrize("groupwise", [False, True])
def test_admission_bounds_running_and_queued_requests_without_consumer_deadlock(
    monkeypatch, groupwise
):
    buffer, model = _buffer(groupwise=groupwise)
    release = threading.Event()
    waiting_for_capacity = threading.Event()
    real_wait = reward_processor_module.wait

    def observed_wait(futures, **kwargs):
        if kwargs.get("return_when") == FIRST_COMPLETED:
            assert len(futures) == 2
            assert sum(not future.done() for _, _, future in buffer._futures) == 2
            waiting_for_capacity.set()
        return real_wait(futures, **kwargs)

    def compute(prompt):
        assert release.wait(30), "test did not release the reward worker"
        return RewardModelOutput(rewards=torch.tensor([float(text) for text in prompt]))

    monkeypatch.setattr(reward_processor_module, "wait", observed_wait)
    model.compute = compute
    producer = ThreadPoolExecutor(max_workers=1)
    try:
        produced = producer.submit(
            buffer.add_samples, [BaseSample(prompt=str(index)) for index in range(5)]
        )
        assert waiting_for_capacity.wait(30)
        assert not produced.done()
        # Reward workers alone release capacity. No main-thread feedback consumer
        # or optimizer step is needed to let the acquisition finish.
        release.set()
        produced.result(timeout=30)
        result = buffer.finalize()
        torch.testing.assert_close(result["score"], torch.arange(5, dtype=torch.float))
    finally:
        release.set()
        producer.shutdown(wait=True)
        buffer.shutdown(wait=True)


@pytest.mark.parametrize("completion", ["success", "failure", "cancelled"])
def test_completed_requests_release_capacity_without_raising_during_rollout(completion):
    buffer, _ = _buffer(limit=1)
    completed = Future()
    if completion == "success":
        completed.set_result(torch.tensor([7.0]))
    elif completion == "failure":
        completed.set_exception(RuntimeError("remote reward failed"))
    else:
        completed.cancel()
    buffer.all_samples = [BaseSample(prompt="7")]
    buffer._rewards["score"] = [None]
    buffer._futures.append(("score", [0], completed))
    buffer._inflight["score"].add(completed)
    try:
        buffer.add_samples([BaseSample(prompt="8")])
        assert len(buffer._futures) == 2
        buffer.seal_for_streaming()
        with pytest.raises(RuntimeError, match="sealed"):
            buffer.add_samples([BaseSample(prompt="9")])
        buffer._futures[-1][2].result(timeout=30)
        if completion == "success":
            assert buffer.poll_ready_tiles({0: (0, 1)}) == {0}
            buffer.resolve_streaming_tile((0, 1))
            torch.testing.assert_close(buffer.finish_streaming()["score"], torch.tensor([7.0, 8.0]))
        else:
            expected = RuntimeError if completion == "failure" else CancelledError
            with pytest.raises(expected):
                buffer.poll_ready_tiles({0: (0, 1)})
            buffer.abort_streaming()
    finally:
        buffer.shutdown(wait=True)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_repeated_cycles_reuse_workers_and_release_samples_and_futures(asynchronous):
    buffer, _ = _buffer(asynchronous=asynchronous)
    executors = dict(buffer._executors)
    previous_threads = None
    try:
        for cycle in range(6):
            sample = BaseSample(prompt=str(cycle), source_id=cycle % 2)
            sample_ref = weakref.ref(sample)
            buffer.add_samples([sample])
            result = buffer.finalize()
            torch.testing.assert_close(result["score"], torch.tensor([float(cycle)]))
            del sample, result
            if asynchronous:
                threads = tuple(executors["score"]._threads)
                assert len(threads) == 1
                if previous_threads is not None:
                    assert threads == previous_threads
                previous_threads = threads
            buffer.clear()
            gc.collect()
            assert sample_ref() is None
            assert buffer.all_samples == []
            assert buffer._executors == executors
            if asynchronous:
                assert not buffer._futures
                assert not buffer._collected_futures
                assert not any(buffer._inflight.values())
    finally:
        buffer.shutdown(wait=True)
    assert all(not thread.is_alive() for pool in executors.values() for thread in pool._threads)


def test_abort_cancels_queued_requests_and_clear_reuses_drained_workers(monkeypatch):
    buffer, model = _buffer(limit=3)
    started = threading.Event()
    release = threading.Event()
    draining = threading.Event()
    clearer = ThreadPoolExecutor(max_workers=1)
    real_wait = reward_processor_module.wait

    def observed_wait(futures, **kwargs):
        draining.set()
        return real_wait(futures, **kwargs)

    def compute(prompt):
        started.set()
        assert release.wait(30)
        return RewardModelOutput(rewards=torch.tensor([float(text) for text in prompt]))

    model.compute = compute
    monkeypatch.setattr(reward_processor_module, "wait", observed_wait)
    try:
        buffer.add_samples([BaseSample(prompt=str(index)) for index in range(3)])
        assert started.wait(30)
        futures = [future for _, _, future in buffer._futures]
        buffer.abort_streaming()
        assert not futures[0].done()
        assert all(future.cancelled() for future in futures[1:])
        with pytest.raises(RuntimeError, match="aborted"):
            buffer.add_samples([BaseSample(prompt="9")])
        cleared = clearer.submit(buffer.clear)
        assert draining.wait(30)
        assert not cleared.done()
        assert len(buffer.all_samples) == 3
        release.set()
        cleared.result(timeout=30)
        assert all(future.done() for future in futures)
        buffer.add_samples([BaseSample(prompt="9")])
        torch.testing.assert_close(buffer.finalize()["score"], torch.tensor([9.0]))
    finally:
        release.set()
        clearer.shutdown(wait=True)
        buffer.shutdown(wait=True)


def test_terminal_close_can_join_prior_interrupt_shutdown_and_rejects_reuse():
    buffer, model = _buffer()
    started = threading.Event()
    release = threading.Event()

    def compute(prompt):
        started.set()
        assert release.wait(30)
        return RewardModelOutput(rewards=torch.ones(len(prompt)))

    model.compute = compute
    try:
        buffer.add_samples([BaseSample(prompt="0")])
        assert started.wait(30)
        buffer.shutdown(wait=False)
        assert any(thread.is_alive() for thread in buffer._executors["score"]._threads)
        with pytest.raises(RuntimeError, match="closed"):
            buffer.clear()
        with pytest.raises(RuntimeError, match="closed"):
            buffer.add_samples([])
    finally:
        release.set()
        buffer.shutdown(wait=True)
        buffer.shutdown(wait=True)
    assert not buffer.all_samples and not buffer._futures
    assert all(not thread.is_alive() for thread in buffer._executors["score"]._threads)


@pytest.mark.parametrize("name", ["num_workers", "max_pending_requests"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_request_capacity_rejects_invalid_configuration(name, value):
    with pytest.raises(ValueError, match=name):
        RewardArguments(**{name: value})


def test_default_capacity_and_config_round_trip():
    buffer, _ = _buffer(workers=3, limit=None)
    try:
        assert buffer._request_limits == {"score": None}
        assert "max_pending_requests" not in RewardArguments().to_dict()
        assert (
            "max_pending_requests"
            not in RewardArguments.from_dict({"max_pending_requests": None}).to_dict()
        )
        configured = RewardArguments.from_dict({"max_pending_requests": 5})
        assert configured.max_pending_requests == 5
        assert configured.to_dict()["max_pending_requests"] == 5
        assert "max_pending_requests" not in configured.extra_kwargs
    finally:
        buffer.shutdown(wait=True)


def test_default_admission_preserves_submission_ahead_of_slow_rewards():
    buffer, model = _buffer(limit=None)
    release = threading.Event()
    producer = ThreadPoolExecutor(max_workers=1)

    def compute(prompt):
        assert release.wait(30)
        return RewardModelOutput(rewards=torch.ones(len(prompt)))

    model.compute = compute
    try:
        produced = producer.submit(
            buffer.add_samples, [BaseSample(prompt=str(index)) for index in range(8)]
        )
        produced.result(timeout=10)
        assert sum(not future.done() for _, _, future in buffer._futures) == 8
        release.set()
        torch.testing.assert_close(buffer.finalize()["score"], torch.ones(8))
    finally:
        release.set()
        producer.shutdown(wait=True)
        buffer.shutdown(wait=True)


def test_default_reward_export_preserves_existing_checkpoint_digest():
    # Captured from main 27a2ab15 before the optional admission field existed.
    legacy = {
        "reward_0": {
            "applicable_datasets": None,
            "async_reward": False,
            "batch_size": 16,
            "device": "cuda",
            "dtype": "bfloat16",
            "name": "default",
            "num_workers": 1,
            "reward_model": None,
            "weight": 1.0,
        }
    }
    current = MultiRewardArguments(reward_configs=[RewardArguments()]).to_dict()
    assert current == legacy
    digest = "42388c09afd0417d7440951d812cb1fef6aed491f683e090d1cfc1eca9302986"
    assert _schema_digest(current) == digest
    bounded = MultiRewardArguments(
        reward_configs=[RewardArguments(max_pending_requests=3)]
    ).to_dict()
    assert bounded["reward_0"]["max_pending_requests"] == 3
    assert _schema_digest(bounded) != digest


def test_mixed_sync_async_components_retain_values_across_reused_cycles():
    original, asynchronous_model = _buffer(limit=1)
    accelerator = original.rp.accelerator
    async_config = original.rp.reward_configs["score"]
    sync_config = RewardArguments(name="sync", device="cpu", batch_size=2)
    synchronous_model = _Pointwise(sync_config, accelerator)
    synchronous_model.compute = lambda prompt: RewardModelOutput(
        rewards=torch.tensor([float(text) * 10 for text in prompt])
    )
    original.shutdown(wait=True)
    buffer = RewardBuffer(
        RewardProcessor(
            accelerator,
            {"score": asynchronous_model, "sync": synchronous_model},
            {"score": async_config, "sync": sync_config},
            group_on_same_rank=True,
            verbose=False,
        ),
        group_size=1,
    )
    try:
        for cycle in range(3):
            buffer.add_samples([BaseSample(prompt=str(index + cycle)) for index in range(3)])
            assert not buffer.supports_streaming
            rewards = buffer.finalize()
            expected = torch.arange(cycle, cycle + 3, dtype=torch.float)
            torch.testing.assert_close(rewards["score"], expected)
            torch.testing.assert_close(rewards["sync"], expected * 10)
            buffer.clear()
    finally:
        buffer.shutdown(wait=True)
