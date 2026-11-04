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

"""Exercise the real HTTP reward client with a tiny local protocol service."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from flow_factory.hparams import RewardArguments
from flow_factory.rewards import RewardBuffer, RewardProcessor
from flow_factory.rewards.my_reward_remote import RemotePointwiseRewardModel, RemoteRewardClient
from flow_factory.samples import T2ISample


def test_remote_reward_full_and_tail_batches_preserve_varying_group_scores(
    reward_service, monkeypatch
):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    accelerator = SimpleNamespace(
        device=torch.device("cpu"), process_index=0, num_processes=1, is_local_main_process=False
    )
    config = RewardArguments.from_dict(
        dict(
            name="remote",
            reward_model="flow_factory.rewards.my_reward_remote.RemotePointwiseRewardModel",
            device="cpu",
            batch_size=3,
            async_reward=True,
            num_workers=1,
            server_url=reward_service.url,
        )
    )
    model = RemotePointwiseRewardModel(config, accelerator)
    # Requests to the local fixture must never be sent through an ambient corporate proxy.
    model.client._session.trust_env = False
    processor = RewardProcessor(
        accelerator, {"remote": model}, {"remote": config}, group_on_same_rank=True, verbose=False
    )
    samples = [
        T2ISample(prompt="one group", image=Image.new("RGB", (2, 2), (value, 0, 0)))
        for value in (0, 64, 128, 255)
    ]
    buffer = RewardBuffer(processor, group_size=4)
    try:
        buffer.add_samples(samples)
        result = buffer.finalize()["remote"]
        torch.testing.assert_close(result, torch.tensor([0, 64 / 255, 128 / 255, 1]))
        assert result.unique().numel() == 4
        assert [len(batch) for batch in reward_service.batches] == [3, 1]
        assert all(sample.extra_kwargs["rewards"]["remote"] is not None for sample in samples)
    finally:
        buffer.shutdown(wait=True)
        model.client._session.close()
        assert all(
            not thread.is_alive()
            for executor in buffer._executors.values()
            for thread in executor._threads
        )


def test_remote_protocol_rejects_malformed_requests(reward_service):
    client = RemoteRewardClient(reward_service.url, timeout=5, retries=1)
    client._session.trust_env = False
    try:
        assert client.health_check()
        with pytest.raises(RuntimeError, match="400"):
            client.compute(["missing image"])
    finally:
        client._session.close()
