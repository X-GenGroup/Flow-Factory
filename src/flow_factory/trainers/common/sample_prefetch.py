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

"""Device-transfer pipeline for replay micro-batches."""

from collections.abc import Iterator
from typing import Any

import torch

from ...samples import BaseSample, StackedSampleBatch
from ...utils.base import visit_tensor_leaves
from .replay_batching import move_and_stack_samples


def _record_stream_on_batch(value: Any, stream: "torch.cuda.Stream") -> None:
    """Keep copy-stream tensors alive until the consuming stream finishes."""
    visit_tensor_leaves(
        value, lambda tensor: tensor.record_stream(stream) if tensor.is_cuda else None
    )


def iter_prefetched_batches(
    samples: list[BaseSample],
    per_device_batch_size: int,
    *,
    device: str | torch.device,
    offload_samples_to_cpu: bool,
) -> Iterator[StackedSampleBatch]:
    """Yield device-resident stacked replay micro-batches.

    With pinned CPU samples and more than one micro-batch, the next H2D copy is
    issued on a dedicated stream. All other cases use the same blocking
    move-and-stack path.
    """
    if not isinstance(per_device_batch_size, int):
        raise TypeError(
            "expected int for per_device_batch_size in replay prefetch, "
            f"received {type(per_device_batch_size).__name__}: {per_device_batch_size!r}"
        )
    if per_device_batch_size < 1:
        raise ValueError(
            "expected per_device_batch_size >= 1 in replay prefetch, "
            f"received {per_device_batch_size}"
        )
    starts = list(range(0, len(samples), per_device_batch_size))
    use_prefetch = torch.cuda.is_available() and offload_samples_to_cpu and len(starts) > 1
    if not use_prefetch:
        for start in starts:
            yield move_and_stack_samples(
                samples[start : start + per_device_batch_size],
                device,
            )
        return

    copy_stream = torch.cuda.Stream(device)
    compute_stream = torch.cuda.current_stream(device)

    def _load(start: int) -> StackedSampleBatch:
        with torch.cuda.stream(copy_stream):
            return move_and_stack_samples(
                samples[start : start + per_device_batch_size],
                device,
                non_blocking=True,
            )

    next_batch = _load(starts[0])
    for index, _ in enumerate(starts):
        batch = next_batch
        compute_stream.wait_stream(copy_stream)
        _record_stream_on_batch(batch, compute_stream)
        if index + 1 < len(starts):
            next_batch = _load(starts[index + 1])
        yield batch
