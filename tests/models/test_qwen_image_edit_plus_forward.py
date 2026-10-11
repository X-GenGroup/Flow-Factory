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

"""Qwen-Image-Edit-Plus per-sample I2I forward receives per-sample timesteps."""

from typing import Any, List, Tuple

import pytest
import torch

from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter
from flow_factory.scheduler import FlowMatchEulerDiscreteSDESchedulerOutput


def _adapter_recording_times(calls: List[Tuple[torch.Tensor, Any]]) -> QwenImageEditPlusAdapter:
    """Build an adapter whose single-sample forward records the timesteps it receives."""
    adapter = object.__new__(QwenImageEditPlusAdapter)

    def fake_forward(**kwargs: Any) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        calls.append((kwargs["t"], kwargs["t_next"]))
        return FlowMatchEulerDiscreteSDESchedulerOutput(next_latents=kwargs["latents"])

    adapter._forward = fake_forward
    return adapter


def _i2i_inputs(batch_size: int) -> dict:
    """Return equal-shape I2I inputs for ``batch_size`` samples."""
    return {
        "latents": torch.zeros(batch_size, 4, 8),
        "prompt_embeds": torch.zeros(batch_size, 3, 16),
        "prompt_embeds_mask": torch.ones(batch_size, 3),
        "image_latents": [torch.zeros(4, 8) for _ in range(batch_size)],
        "img_shapes": [[(1, 2, 2)] for _ in range(batch_size)],
    }


def test_i2i_forward_slices_per_sample_timesteps() -> None:
    """Give each sample its own timestep instead of the whole batch tensor."""
    calls: List[Tuple[torch.Tensor, Any]] = []
    adapter = _adapter_recording_times(calls)

    output = adapter.forward(
        t=torch.tensor([900.0, 500.0]),
        t_next=torch.tensor([800.0, 400.0]),
        **_i2i_inputs(batch_size=2),
    )

    assert [call[0].tolist() for call in calls] == [[900.0], [500.0]]
    assert [call[1].tolist() for call in calls] == [[800.0], [400.0]]
    assert output.next_latents.shape == (2, 4, 8)


def test_i2i_forward_shares_scalar_timesteps() -> None:
    """Pass a shared scalar timestep and a missing next timestep through unchanged."""
    calls: List[Tuple[torch.Tensor, Any]] = []
    adapter = _adapter_recording_times(calls)
    timestep = torch.tensor(700.0)

    adapter.forward(t=timestep, **_i2i_inputs(batch_size=2))

    assert all(call[0] is timestep and call[1] is None for call in calls)


def test_i2i_forward_rejects_timesteps_not_sized_by_the_batch() -> None:
    """Report a timestep tensor whose leading size does not match the batch."""
    adapter = _adapter_recording_times([])

    with pytest.raises(ValueError, match="expected t to hold one value or 2 per-sample values"):
        adapter.forward(t=torch.tensor([900.0, 500.0, 100.0]), **_i2i_inputs(batch_size=2))
