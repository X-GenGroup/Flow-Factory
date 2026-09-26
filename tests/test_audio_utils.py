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

import pytest
import torch

import flow_factory.utils.audio as audio_utils
from flow_factory.utils.audio import require_decoded_audio_waveform


def test_require_decoded_audio_waveform_accepts_canonical_unclipped_amplitudes() -> None:
    """The canonical waveform is finite float32; amplitudes are not silently clipped."""
    waveform = torch.tensor([[-2.0, 0.0, 3.0]], dtype=torch.float32)

    assert require_decoded_audio_waveform(waveform, source="test audio") is waveform


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        (object(), TypeError, "decoded torch.Tensor"),
        (torch.zeros(1, 4, dtype=torch.float64), TypeError, "dtype float32"),
        (torch.zeros(4), ValueError, "channels, samples"),
        (torch.zeros(0, 4), ValueError, "channels, samples"),
        (torch.zeros(1, 8)[:, ::2], ValueError, "contiguous waveform"),
        (torch.tensor([[float("nan")]]), ValueError, "non-finite samples"),
    ],
)
def test_require_decoded_audio_waveform_rejects_noncanonical_payloads(
    payload: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        require_decoded_audio_waveform(payload, source="test audio")


def test_require_decoded_audio_waveform_rejects_attached_tensor() -> None:
    waveform = torch.zeros(1, 4, requires_grad=True)

    with pytest.raises(ValueError, match="detached no-grad waveform"):
        require_decoded_audio_waveform(waveform, source="test audio")


def test_load_audio_canonicalizes_backend_waveform(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every backend leaves load_audio at the same detached CPU float32 boundary."""
    path = tmp_path / "audio.fake"
    path.write_bytes(b"stub")
    backend_waveform = torch.arange(16.0).reshape(8, 2).t().requires_grad_()
    assert not backend_waveform.is_contiguous()
    monkeypatch.setattr(
        audio_utils,
        "_load_audio_backend",
        lambda _: (backend_waveform, 16_000),
    )

    waveform = audio_utils.load_audio(path)

    torch.testing.assert_close(waveform, backend_waveform.detach())
    assert waveform.dtype is torch.float32
    assert waveform.device.type == "cpu"
    assert waveform.is_contiguous()
    assert waveform.requires_grad is False
