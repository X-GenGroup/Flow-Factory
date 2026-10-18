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

import pytest
import torch

from flow_factory.samples import ComponentTimes, LatentState


def test_latent_state_reports_its_batch_size() -> None:
    state = LatentState({"video": torch.zeros(3, 4, 2), "audio": torch.zeros(3, 5, 2)})

    assert state.batch_size == 3


def test_per_sample_times_broadcast_shared_times_for_every_component() -> None:
    times = ComponentTimes(
        timestep={"video": torch.tensor(700.0), "audio": torch.tensor([600.0])},
        next_timestep={"video": torch.tensor(400.0), "audio": torch.tensor([300.0])},
        sigma={"video": torch.tensor(0.7), "audio": torch.tensor(0.6)},
        next_sigma=None,
    )

    expanded = times.per_sample(2)

    assert expanded.timestep["video"].tolist() == [700.0, 700.0]
    assert expanded.timestep["audio"].tolist() == [600.0, 600.0]
    assert expanded.next_timestep["audio"].tolist() == [300.0, 300.0]
    assert expanded.sigma["video"].tolist() == pytest.approx([0.7, 0.7])
    assert expanded.next_sigma is None
    assert tuple(expanded.timestep) == ("video", "audio")


def test_per_sample_times_keep_per_sample_values() -> None:
    timestep = torch.tensor([700.0, 650.0])
    times = ComponentTimes(
        timestep={"latent": timestep},
        next_timestep={"latent": torch.tensor([400.0, 350.0])},
    )

    expanded = times.per_sample(2)

    assert expanded.timestep["latent"] is timestep
    assert expanded.next_timestep["latent"].tolist() == [400.0, 350.0]


def test_per_sample_times_reject_a_mismatched_batch() -> None:
    times = ComponentTimes(
        timestep={"latent": torch.tensor([700.0, 650.0])},
        next_timestep={"latent": torch.tensor([400.0, 350.0])},
    )

    with pytest.raises(ValueError, match=r"times\.timestep\['latent'\].*\(3,\).*\(2,\)"):
        times.per_sample(3)
