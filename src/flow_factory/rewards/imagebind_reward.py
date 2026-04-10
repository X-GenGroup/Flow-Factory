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

# src/flow_factory/rewards/imagebind_reward.py
"""Audio-video semantic alignment reward using Meta ImageBind.

IMPORTANT: ImageBind is licensed under CC-BY-NC-SA 4.0 (NonCommercial).
This module is only loaded when ``reward_model: "imagebind"`` is configured
in the YAML, via the registry's ``importlib.import_module()`` call.
"""
from __future__ import annotations
import warnings
from typing import Optional, List

import torch
import torch.nn.functional as F
import numpy as np
import torchaudio.functional as AF
import torchaudio.compliance.kaldi as kaldi

from accelerate import Accelerator

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments

_IMAGEBIND_INSTALL_MSG = (
    "ImageBind is not installed. Install with:\n"
    "  pip install git+https://github.com/facebookresearch/ImageBind.git\n"
    "  pip install git+https://github.com/facebookresearch/pytorchvideo.git\n"
    "Note: ImageBind is CC-BY-NC-SA 4.0 (NonCommercial only)."
)

try:
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
    from imagebind.data import load_and_transform_text
except ImportError as e:
    raise ImportError(_IMAGEBIND_INSTALL_MSG) from e

_IMAGEBIND_LICENSE_WARNING = (
    "ImageBind is licensed under CC-BY-NC-SA 4.0 (NonCommercial). "
    "Using it in commercial applications may violate the license. "
    "See: https://github.com/facebookresearch/ImageBind/blob/main/LICENSE"
)

_IB_AUDIO_SAMPLE_RATE = 16_000
_IB_AUDIO_NUM_MEL_BINS = 128
_IB_AUDIO_TARGET_LENGTH = 204
_IB_AUDIO_CLIP_DURATION = 2
_IB_AUDIO_CLIPS_PER_SAMPLE = 3
_IB_AUDIO_MEAN = -4.268
_IB_AUDIO_STD = 9.138

_IB_VISION_SIZE = 224
_IB_VISION_MEAN = (0.48145466, 0.4578275, 0.40821073)
_IB_VISION_STD = (0.26862954, 0.26130258, 0.27577711)


class ImageBindRewardModel(PointwiseRewardModel):
    """Audio-video semantic alignment reward using Meta ImageBind.

    Supports multiple scoring modes via extra_kwargs["mode"]:
        - "audio_video" (default): cos_sim(audio_embed, video_embed)
        - "text_audio":            cos_sim(text_embed, audio_embed)
        - "text_video":            cos_sim(text_embed, video_embed)
        - "all":                   weighted sum of all three

    IMPORTANT: ImageBind is CC-BY-NC-SA 4.0 (NonCommercial).
    """
    required_fields = ("prompt", "audio", "video")
    use_tensor_inputs = True
    DEFAULT_MODE = "audio_video"

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        warnings.warn(_IMAGEBIND_LICENSE_WARNING, stacklevel=2)

        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.to(self.device).eval()

        self.mode: str = config.extra_kwargs.get("mode", self.DEFAULT_MODE)
        self.weights: dict = config.extra_kwargs.get(
            "weights", {"audio_video": 0.5, "text_audio": 0.25, "text_video": 0.25}
        )

    def _preprocess_audio_to_melspec(
        self,
        audio_list: List[torch.Tensor],
        src_sample_rate: int,
    ) -> torch.Tensor:
        """Convert List[Tensor(C, T)] to batched mel-spectrograms.

        Per-sample pipeline:
            (C, T) @ src_rate -> mono (1, T) -> resample 16 kHz (1, T')
            -> split into 3 clips of 2 s = 32000 samples -> waveform2melspec
            -> (3, 1, 128, 204) -> normalize

        Returns:
            (B, 3, 1, 128, 204)
        """
        batch_clips = []
        samples_per_clip = _IB_AUDIO_CLIP_DURATION * _IB_AUDIO_SAMPLE_RATE

        for waveform in audio_list:
            if waveform.ndim == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            if src_sample_rate != _IB_AUDIO_SAMPLE_RATE:
                waveform = AF.resample(
                    waveform, src_sample_rate, _IB_AUDIO_SAMPLE_RATE
                )

            total_samples = waveform.shape[1]
            duration_s = total_samples / _IB_AUDIO_SAMPLE_RATE
            clip_starts = self._compute_clip_starts(
                duration_s, _IB_AUDIO_CLIP_DURATION, _IB_AUDIO_CLIPS_PER_SAMPLE
            )

            mel_clips = []
            for start_s in clip_starts:
                start_idx = int(start_s * _IB_AUDIO_SAMPLE_RATE)
                end_idx = start_idx + samples_per_clip
                clip = waveform[:, start_idx:end_idx]
                if clip.shape[1] < samples_per_clip:
                    clip = F.pad(clip, (0, samples_per_clip - clip.shape[1]))

                mel = self._waveform_to_melspec(clip)
                mel = (mel - _IB_AUDIO_MEAN) / _IB_AUDIO_STD
                mel_clips.append(mel)

            batch_clips.append(torch.stack(mel_clips, dim=0))

        return torch.stack(batch_clips, dim=0).to(self.device)

    @staticmethod
    def _waveform_to_melspec(waveform: torch.Tensor) -> torch.Tensor:
        """Replicate ImageBind waveform2melspec.

        Args:
            waveform: (1, T) mono 16 kHz

        Returns:
            (1, 128, 204) mel-spectrogram tensor
        """
        waveform = waveform - waveform.mean()
        fbank = kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=_IB_AUDIO_SAMPLE_RATE,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=_IB_AUDIO_NUM_MEL_BINS,
            dither=0.0,
            frame_length=25,
            frame_shift=10,
        )  # (num_frames, 128)

        fbank = fbank.transpose(0, 1)  # (128, num_frames)

        n_frames = fbank.shape[1]
        if n_frames < _IB_AUDIO_TARGET_LENGTH:
            fbank = F.pad(fbank, (0, _IB_AUDIO_TARGET_LENGTH - n_frames))
        else:
            fbank = fbank[:, :_IB_AUDIO_TARGET_LENGTH]

        return fbank.unsqueeze(0)  # (1, 128, 204)

    @staticmethod
    def _compute_clip_starts(
        duration_s: float, clip_duration: float, num_clips: int
    ) -> List[float]:
        """Evenly space clip start times, matching ConstantClipsPerVideoSampler."""
        if duration_s <= clip_duration:
            return [0.0] * num_clips
        spacing = (duration_s - clip_duration) / max(num_clips - 1, 1)
        return [i * spacing for i in range(num_clips)]

    def _preprocess_video(
        self, video_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Convert List[Tensor(T, C, H, W)] to ImageBind video format.

        Per-sample pipeline:
            (T, C, H, W) -> 5 clips x 2 frames -> resize short side 224
            -> CLIP normalize -> 3 spatial crops -> (15, C, 2, 224, 224)

        Returns:
            (B, 15, C, 2, 224, 224)
        """
        batch_result = []
        for video in video_list:
            T, C, H, W = video.shape
            video_f = video.float() / 255.0 if video.dtype == torch.uint8 else video.float()

            clips = self._temporal_subsample_clips(
                video_f, num_clips=5, frames_per_clip=2
            )

            all_crops = []
            for clip in clips:
                clip = self._resize_short_side(clip, _IB_VISION_SIZE)
                clip = self._normalize_video_clip(clip)
                crops = self._spatial_crop(clip, _IB_VISION_SIZE)
                all_crops.extend(crops)

            batch_result.append(torch.stack(all_crops, dim=0))

        return torch.stack(batch_result, dim=0).to(self.device)

    def _preprocess_text(self, prompts: List[str]) -> torch.Tensor:
        """BPE tokenize prompts using ImageBind's SimpleTokenizer.

        Returns:
            (B, 77)
        """
        return load_and_transform_text(prompts, self.device)

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        audio: Optional[List[torch.Tensor]] = None,
        video: Optional[List[torch.Tensor]] = None,
        audio_sample_rate: Optional[List[int]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        src_rate = audio_sample_rate[0] if audio_sample_rate else _IB_AUDIO_SAMPLE_RATE

        inputs = {}

        need_text = self.mode in ("text_audio", "text_video", "all")
        need_audio = self.mode in ("audio_video", "text_audio", "all")
        need_video = self.mode in ("audio_video", "text_video", "all")

        if need_text:
            inputs[ModalityType.TEXT] = self._preprocess_text(prompt)
        if need_audio and audio is not None:
            inputs[ModalityType.AUDIO] = self._preprocess_audio_to_melspec(
                audio, src_rate
            )
        if need_video and video is not None:
            inputs[ModalityType.VISION] = self._preprocess_video(video)

        embeddings = self.model(inputs)

        rewards = self._compute_similarity(embeddings)
        return RewardModelOutput(rewards=rewards.float().cpu())

    def _compute_similarity(self, embeddings: dict) -> torch.Tensor:
        """Compute cosine similarity based on self.mode."""
        def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)

        if self.mode == "audio_video":
            return cos_sim(embeddings[ModalityType.AUDIO], embeddings[ModalityType.VISION])
        elif self.mode == "text_audio":
            return cos_sim(embeddings[ModalityType.TEXT], embeddings[ModalityType.AUDIO])
        elif self.mode == "text_video":
            return cos_sim(embeddings[ModalityType.TEXT], embeddings[ModalityType.VISION])
        elif self.mode == "all":
            w = self.weights
            av = cos_sim(embeddings[ModalityType.AUDIO], embeddings[ModalityType.VISION])
            ta = cos_sim(embeddings[ModalityType.TEXT], embeddings[ModalityType.AUDIO])
            tv = cos_sim(embeddings[ModalityType.TEXT], embeddings[ModalityType.VISION])
            return w["audio_video"] * av + w["text_audio"] * ta + w["text_video"] * tv
        else:
            raise ValueError(
                f"Unknown ImageBind mode {self.mode!r}, "
                f"expected one of: audio_video, text_audio, text_video, all"
            )

    @staticmethod
    def _temporal_subsample_clips(
        video: torch.Tensor, num_clips: int, frames_per_clip: int
    ) -> List[torch.Tensor]:
        """Sample evenly spaced temporal clips.

        Args:
            video: (T, C, H, W)
        Returns:
            List of (C, frames_per_clip, H, W) clips
        """
        T = video.shape[0]
        clips = []
        for i in range(num_clips):
            center = int((i + 0.5) * T / num_clips)
            indices = torch.linspace(
                max(0, center - frames_per_clip // 2),
                min(T - 1, center + frames_per_clip // 2 - 1),
                frames_per_clip,
            ).long()
            clip = video[indices]  # (frames_per_clip, C, H, W)
            clips.append(clip.permute(1, 0, 2, 3))  # (C, frames_per_clip, H, W)
        return clips

    @staticmethod
    def _resize_short_side(clip: torch.Tensor, size: int) -> torch.Tensor:
        """Resize so the shorter spatial side == size.

        Args:
            clip: (C, T, H, W)
        Returns:
            (C, T, H', W') with min(H', W') == size
        """
        C, T, H, W = clip.shape
        if W <= H:
            new_w, new_h = size, int(H / W * size)
        else:
            new_w, new_h = int(W / H * size), size
        # Reshape to (C*T, 1, H, W) for F.interpolate
        clip_flat = clip.reshape(C * T, 1, H, W)
        clip_resized = F.interpolate(
            clip_flat, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        return clip_resized.reshape(C, T, new_h, new_w)

    @staticmethod
    def _normalize_video_clip(clip: torch.Tensor) -> torch.Tensor:
        """Apply CLIP normalization to (C, T, H, W)."""
        mean = torch.tensor(_IB_VISION_MEAN, device=clip.device).view(3, 1, 1, 1)
        std = torch.tensor(_IB_VISION_STD, device=clip.device).view(3, 1, 1, 1)
        return (clip - mean) / std

    @staticmethod
    def _spatial_crop(
        clip: torch.Tensor, crop_size: int
    ) -> List[torch.Tensor]:
        """3 spatial crops (left/center/right or top/center/bottom).

        Args:
            clip: (C, T, H, W) where min(H, W) == crop_size
        Returns:
            List of 3 tensors each (C, T, crop_size, crop_size)
        """
        C, T, H, W = clip.shape
        crops = []
        if H > W:
            offsets = [0, (H - crop_size) // 2, H - crop_size]
            for y in offsets:
                crops.append(clip[:, :, y:y + crop_size, :])
        else:
            offsets = [0, (W - crop_size) // 2, W - crop_size]
            for x in offsets:
                crops.append(clip[:, :, :, x:x + crop_size])
        return crops
