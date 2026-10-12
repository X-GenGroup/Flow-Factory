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

from types import SimpleNamespace

import pytest
import torch

from flow_factory.samples import BaseSample, StackedSampleBatch
from flow_factory.trainers.common.forward_kwargs import (
    reference_forward_kwargs,
    replay_forward_kwargs,
    training_forward_kwargs,
)
from flow_factory.trainers.common.replay_batching import move_and_stack_samples
from flow_factory.trainers.common.sample_prefetch import iter_prefetched_batches


def test_forward_kwargs_preserve_batch_precedence_and_explicit_reference_overrides():
    trainer = SimpleNamespace(training_args={"guidance_scale": 3.0, "resolution": 512, "seed": 42})
    batch = StackedSampleBatch(
        [
            BaseSample(
                prompt="prompt",
                extra_kwargs={"guidance_scale": 1.0},
            )
        ]
    )

    expected = {"resolution": 512, "seed": 42}
    assert training_forward_kwargs(trainer, batch) == expected
    assert replay_forward_kwargs(trainer, batch) == expected
    assert reference_forward_kwargs(trainer, batch, guidance_scale=4.0) == {
        **expected,
        "guidance_scale": 4.0,
    }


def test_move_and_stack_samples_keeps_moved_sources_on_batch():
    samples = [
        BaseSample(prompt="a", prompt_embeds=torch.tensor([1.0])),
        BaseSample(prompt="b", prompt_embeds=torch.tensor([2.0])),
    ]

    batch = move_and_stack_samples(samples, torch.device("cpu"))

    assert isinstance(batch, StackedSampleBatch)
    assert batch.samples == samples
    assert batch["prompt"] == ["a", "b"]
    torch.testing.assert_close(batch["prompt_embeds"], torch.tensor([[1.0], [2.0]]))


def test_cpu_prefetch_path_chunks_and_stacks_once_per_micro_batch():
    samples = [
        BaseSample(prompt=str(index), prompt_embeds=torch.tensor([float(index)]))
        for index in range(3)
    ]

    batches = list(
        iter_prefetched_batches(
            samples,
            2,
            device=torch.device("cpu"),
            offload_samples_to_cpu=False,
        )
    )

    assert [len(batch.samples) for batch in batches] == [2, 1]
    assert batches[0]["prompt"] == ["0", "1"]
    assert batches[1]["prompt"] == ["2"]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_prefetch_rejects_non_positive_batch_size(batch_size):
    with pytest.raises(ValueError, match="expected per_device_batch_size >= 1"):
        list(
            iter_prefetched_batches(
                [],
                batch_size,
                device=torch.device("cpu"),
                offload_samples_to_cpu=False,
            )
        )
