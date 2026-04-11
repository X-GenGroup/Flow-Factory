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

# src/flow_factory/rewards/clap.py
"""Audio-text alignment reward using LAION CLAP via HuggingFace transformers."""
from __future__ import annotations
from typing import Optional, List

import torch
import torch.nn.functional as F
import numpy as np
import torchaudio.functional as AF

from accelerate import Accelerator
from transformers import ClapModel, ClapProcessor

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments


class CLAPRewardModel(PointwiseRewardModel):
    """Audio-text alignment reward using LAION CLAP (via HuggingFace transformers).

    Computes cosine similarity between audio and text embeddings.
    Zero additional dependencies -- uses transformers.ClapModel already in dep tree.
    """
    required_fields = ("prompt", "audio")
    use_tensor_inputs = True
    DEFAULT_MODEL = "laion/larger_clap_general"
    CLAP_SAMPLE_RATE = 48_000

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        model_name = config.extra_kwargs.get(
            "model_name_or_path", self.DEFAULT_MODEL
        )
        # float32 required: CLAP audio encoder uses BatchNorm which doesn't support bf16
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(model_name)

    def _preprocess_audio(
        self,
        audio_list: List[torch.Tensor],
        src_sample_rate: int,
    ) -> List[np.ndarray]:
        """Convert List[Tensor(C, T)] to List[ndarray(T',)] mono 48 kHz.

        Steps per waveform:
            1. Downmix to mono  : (C, T) -> (1, T) -> (T,)
            2. Resample          : (T,)   -> (T',)  via torchaudio.functional.resample
            3. To numpy float32  : ndarray (T',)
        """
        processed: List[np.ndarray] = []
        for waveform in audio_list:
            if waveform.ndim == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            elif waveform.ndim == 2:
                waveform = waveform.squeeze(0)

            if src_sample_rate != self.CLAP_SAMPLE_RATE:
                waveform = AF.resample(
                    waveform.unsqueeze(0),
                    orig_freq=src_sample_rate,
                    new_freq=self.CLAP_SAMPLE_RATE,
                ).squeeze(0)

            processed.append(waveform.detach().cpu().float().numpy())

        return processed

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        audio: List[torch.Tensor],
        audio_sample_rate: Optional[List[int]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """Compute audio-text cosine similarity.

        Args:
            prompt:            (B,) text descriptions
            audio:             (B,) tensors each (C, T) at source sample rate
            audio_sample_rate: (B,) ints, all identical -- source Hz
        """
        if not isinstance(audio_sample_rate, list) or len(audio_sample_rate) == 0:
            raise ValueError(
                "audio_sample_rate is required for CLAP reward; "
                f"got {audio_sample_rate!r}"
            )
        src_rate = audio_sample_rate[0]

        waveforms_np = self._preprocess_audio(audio, src_rate)

        inputs = self.processor(
            text=prompt,
            audio=waveforms_np,
            sampling_rate=self.CLAP_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        audio_embeds = F.normalize(outputs.audio_embeds, p=2, dim=-1)
        text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)
        rewards = (audio_embeds * text_embeds).sum(dim=-1)

        return RewardModelOutput(
            rewards=rewards.float().cpu(),
            extra_info={"src_sample_rate": src_rate},
        )
