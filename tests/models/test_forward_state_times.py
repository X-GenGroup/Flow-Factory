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
from flow_factory.samples import ComponentTimes, LatentState
from flow_factory.scheduler.flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSDESchedulerOutput,
)


def _i2i_batch(batch_size: int) -> Dict[str, Any]:
    return {
        "latent_ids": torch.zeros(batch_size, 3, 4),
        "prompt_embeds": torch.zeros(batch_size, 2, 5),
        "text_ids": torch.zeros(batch_size, 2, 4),
        "image_latents": [torch.zeros(3, 4), torch.zeros(5, 4)],
        "image_latent_ids": [torch.zeros(3, 4), torch.zeros(5, 4)],
    }


def test_forward_state_hands_every_adapter_per_sample_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared boundary expands batch-shared replay times before any model hook runs."""
    adapter = object.__new__(Flux2KleinAdapter)
    received: List[ComponentTimes] = []

    def capture(**kwargs: Any) -> Any:
        received.append(kwargs["times"])
        return None

    monkeypatch.setattr(adapter, "_forward_state", capture)
    adapter.forward_state(
        batch={},
        state=LatentState({"latent": torch.zeros(3, 4, 2)}),
        times=ComponentTimes(
            timestep={"latent": torch.tensor(700.0)},
            next_timestep={"latent": torch.tensor([400.0])},
        ),
    )

    assert received[0].timestep["latent"].tolist() == [700.0, 700.0, 700.0]
    assert received[0].next_timestep["latent"].tolist() == [400.0, 400.0, 400.0]


@pytest.mark.parametrize("adapter_cls", [Flux2Adapter, Flux2KleinAdapter])
def test_distillation_replay_time_reaches_each_i2i_sample(adapter_cls: type) -> None:
    """A 0-dim replay time travels through forward_state into the per-sample I2I loop."""
    adapter = object.__new__(adapter_cls)
    adapter._has_warned_forward_fallback = True
    calls: List[Dict[str, Any]] = []

    def fake_forward(**kwargs: Any) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        calls.append(kwargs)
        return FlowMatchEulerDiscreteSDESchedulerOutput(next_latents=kwargs["latents"])

    adapter._forward = fake_forward
    output = adapter.forward_state(
        batch=_i2i_batch(2),
        state=LatentState({"latent": torch.zeros(2, 3, 4)}),
        times=ComponentTimes(
            timestep={"latent": torch.tensor(700.0)},
            next_timestep={"latent": torch.tensor(400.0)},
        ),
        return_fields=("next_latents",),
    )

    assert [call["t"].tolist() for call in calls] == [[700.0], [700.0]]
    assert [call["t_next"].tolist() for call in calls] == [[400.0], [400.0]]
    assert output.next_state.components["latent"].shape == (2, 3, 4)
