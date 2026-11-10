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
    """Exercise native codec I/O and the differentiable features used by AV owners."""
    rate = 16000
    time = torch.arange(rate, dtype=torch.float32) / rate
    waveform = (0.3 * torch.sin(2 * torch.pi * 440 * time)).unsqueeze(0)
    path = tmp_path / "tone.wav"
    save_audio(waveform, path, rate)
    # Direct TorchAudio calls prevent Flow-Factory's alternate decoders from
    # concealing a broken TorchCodec installation.
    decoded, actual_rate = torchaudio.load(path)
    assert actual_rate == rate
    torch.testing.assert_close(decoded, waveform, rtol=0, atol=4e-5)
    differentiable = decoded.detach().requires_grad_()
    resampled = torchaudio.functional.resample(differentiable, rate, 24000)
    assert resampled.shape == (1, 24000)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_mels=32)(resampled)
    assert mel.shape[1] == 32 and torch.isfinite(mel).all()
    mel.square().mean().backward()
    assert torch.isfinite(differentiable.grad).all()
    assert differentiable.grad.abs().sum() > 0
    fbank = torchaudio.compliance.kaldi.fbank(decoded, sample_frequency=rate, num_mel_bins=32)
    assert fbank.shape[1] == 32 and fbank.shape[0] > 1
    assert torch.isfinite(fbank).all() and fbank.var() > 0
