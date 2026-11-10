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

from pathlib import Path

import torch
import torchaudio

from flow_factory.utils.audio import save_audio


def test_audio_codec_roundtrip_and_feature_gradients(tmp_path: Path):
    """Torch compatibility policy is device/version bounded and never runs on import."""


from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flow_factory.utils.torch_runtime import configure_torch_runtime


@pytest.mark.parametrize("version", ["2.10.0+cu128", "2.11.0+cu128", "2.12.0"])
@pytest.mark.parametrize(
    "device,capability", [("cpu", None), ("cuda:3", (9, 0)), ("cuda:3", (8, 9))]
)
def test_runtime_policy_boundaries(monkeypatch, version, device, capability):
    monkeypatch.setattr(torch, "__version__", version)
    capability_query = Mock(return_value=capability)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability_query)
    enabled = [True]
    monkeypatch.setattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: enabled[0])
    setter = Mock(side_effect=lambda value: enabled.__setitem__(0, value))
    monkeypatch.setattr(torch.backends.cuda, "enable_cudnn_sdp", setter)
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 91900)
    if version.startswith("2.11.") and capability == (9, 0):
        with pytest.warns(RuntimeWarning, match="disabling cuDNN SDPA"):
            configure_torch_runtime(torch.device(device))
        configure_torch_runtime(torch.device(device))
        setter.assert_called_once_with(False)
    else:
        configure_torch_runtime(torch.device(device))
        setter.assert_not_called()
    if version.startswith("2.11.") and device.startswith("cuda"):
        capability_query.assert_called_with(torch.device(device))
    else:
        capability_query.assert_not_called()


def test_accelerator_applies_runtime_policy_after_device_selection(monkeypatch):
    from flow_factory.trainers import loader

    device = torch.device("cuda:3")

    def initialize(accelerator, **kwargs):
        assert kwargs == {"mixed_precision": "bf16"}
        accelerator.state = SimpleNamespace(device=device)

    monkeypatch.setattr(loader._UpstreamAccelerator, "__init__", initialize)
    policy = Mock()
    monkeypatch.setattr(loader, "configure_torch_runtime", policy)
    loader.Accelerator(mixed_precision="bf16")
    policy.assert_called_once_with(device)
