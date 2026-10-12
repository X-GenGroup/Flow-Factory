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

"""Sample movement and collation for replay-oriented trainer loops."""

from collections.abc import Sequence

import torch

from ...samples import BaseSample, StackedSampleBatch


def move_and_stack_samples(
    samples: Sequence[BaseSample],
    device: str | torch.device,
    *,
    non_blocking: bool = False,
) -> StackedSampleBatch:
    """Move one micro-batch to ``device`` and collate it exactly once."""
    moved = [sample.to(device, non_blocking=non_blocking) for sample in samples]
    return BaseSample.stack(moved)
