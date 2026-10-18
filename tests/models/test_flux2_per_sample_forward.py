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

from typing import Any, Dict, List

import pytest
import torch

from flow_factory.models.flux.flux2 import Flux2Adapter
from flow_factory.models.flux.flux2_klein import Flux2KleinAdapter
from flow_factory.scheduler.flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSDESchedulerOutput,
)
from flow_factory.utils.base import expand_per_sample


@pytest.mark.parametrize("value", [torch.tensor(0.7), torch.tensor([0.7])])
def test_expand_per_sample_broadcasts_a_shared_value(value: torch.Tensor) -> None:
    assert expand_per_sample(value, 3, "t").tolist() == pytest.approx([0.7] * 3)


def test_expand_per_sample_keeps_per_sample_values_and_none() -> None:
    values = torch.tensor([0.1, 0.2, 0.3])

    assert expand_per_sample(values, 3, "t") is values
    assert expand_per_sample(None, 3, "t_next") is None


@pytest.mark.parametrize(
    ("value", "error_type"),
    [(torch.zeros(2), ValueError), (torch.zeros(3, 1), ValueError), (0.5, TypeError)],
)
def test_expand_per_sample_rejects_mismatched_values(value: Any, error_type: type) -> None:
    with pytest.raises(error_type, match="t"):
        expand_per_sample(value, 3, "t")


@pytest.mark.parametrize("adapter_cls", [Flux2Adapter, Flux2KleinAdapter])
@pytest.mark.parametrize(
    ("t", "t_next"),
    [
        (torch.tensor(700.0), torch.tensor(400.0)),
        (torch.tensor([700.0, 650.0]), torch.tensor([400.0, 350.0])),
    ],
    ids=["shared", "per_sample"],
)
def test_flux2_per_sample_forward_accepts_shared_and_per_sample_times(
    adapter_cls: type, t: torch.Tensor, t_next: torch.Tensor
) -> None:
    """I2I runs one forward per sample; a shared replay time reaches every sample."""
    adapter = object.__new__(adapter_cls)
    adapter._has_warned_forward_fallback = True
    calls: List[Dict[str, Any]] = []

    def fake_forward(**kwargs: Any) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        calls.append(kwargs)
        return FlowMatchEulerDiscreteSDESchedulerOutput(next_latents=kwargs["latents"] + 1.0)

    adapter._forward = fake_forward
    batch_size = 2
    output = adapter.forward(
        t=t,
        latents=torch.zeros(batch_size, 3, 4),
        latent_ids=torch.zeros(batch_size, 3, 4),
        prompt_embeds=torch.zeros(batch_size, 2, 5),
        text_ids=torch.zeros(batch_size, 2, 4),
        image_latents=[torch.zeros(3, 4), torch.zeros(5, 4)],
        image_latent_ids=[torch.zeros(3, 4), torch.zeros(5, 4)],
        t_next=t_next,
        compute_log_prob=False,
        return_kwargs=["next_latents"],
    )

    assert [tuple(call["t"].shape) for call in calls] == [(1,), (1,)]
    assert torch.equal(torch.cat([call["t"] for call in calls]), t.expand(batch_size))
    assert torch.equal(torch.cat([call["t_next"] for call in calls]), t_next.expand(batch_size))
    assert output.next_latents.shape == (batch_size, 3, 4)
