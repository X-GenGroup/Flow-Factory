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

# src/flow_factory/models/ltx2/ltx2_t2av.py
from __future__ import annotations

from typing import Union, List, Dict, Any, Optional, ClassVar
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline, rescale_noise_cfg
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

from ..abc import BaseAdapter
from ...samples import T2AVSample
from ...hparams import *
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    SDESchedulerOutput,
    set_scheduler_timesteps,
)
from ...scheduler.flow_match_euler_discrete import calculate_shift
from ...utils.trajectory_collector import (
    TrajectoryIndicesType,
    create_trajectory_collector,
    create_callback_collector,
)
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


# ================================== Sample Dataclass ==================================

@dataclass
class LTX2Sample(T2AVSample):
    """Output class for LTX2 text-to-audio-video adapter."""
    _shared_fields: ClassVar[frozenset[str]] = frozenset({
        'height', 'width',
        'latent_index_map', 'log_prob_index_map',
        'audio_latent_index_map',
    })

    # Audio latent trajectory (for training reconstruction only, NOT RL-optimized)
    audio_all_latents: Optional[torch.Tensor] = None      # (stored_steps, audio_seq, C)
    audio_latent_index_map: Optional[torch.Tensor] = None  # (num_steps+1,) LongTensor

    # Connector outputs (video/audio text embeddings, cached from preprocessing)
    connector_prompt_embeds: Optional[torch.Tensor] = None        # (seq, D_video)
    connector_audio_prompt_embeds: Optional[torch.Tensor] = None  # (seq, D_audio)
    connector_attention_mask: Optional[torch.Tensor] = None       # (seq,)

    # Negative prompt connector outputs (for CFG during training forward)
    negative_connector_prompt_embeds: Optional[torch.Tensor] = None
    negative_connector_audio_prompt_embeds: Optional[torch.Tensor] = None
    negative_connector_attention_mask: Optional[torch.Tensor] = None


# ================================== Adapter ==================================

class LTX2_T2AV_Adapter(BaseAdapter):
    """
    Adapter for LTX2 text-to-audio-video generation with video-only SDE optimization.

    Audio is generated jointly via the transformer's cross-modal attention but uses
    deterministic ODE sampling (no log_prob, no RL gradient). Only the video pathway
    receives stochastic SDE treatment for policy gradient training.
    """

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: LTX2Pipeline
        self.audio_scheduler: FlowMatchEulerDiscreteSDEScheduler = self._create_audio_scheduler()

    # ============================== Pipeline Loading ==============================

    def load_pipeline(self) -> LTX2Pipeline:
        return LTX2Pipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False,  # Required for FSDP compatibility
        )

    def _create_audio_scheduler(self) -> FlowMatchEulerDiscreteSDEScheduler:
        """Create a separate ODE-only scheduler for audio.

        A dedicated instance is needed because scheduler.step() mutates internal
        state (step_index), which would conflict if shared with the video scheduler.
        """
        base_config = {
            k: v for k, v in dict(self.pipeline.scheduler.config).items()
            if k not in ('_class_name', '_diffusers_version')
        }
        return FlowMatchEulerDiscreteSDEScheduler(
            dynamics_type='ODE', noise_level=0.0, **base_config,
        )

    # ============================== Module Properties ==============================

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for LTX2 transformer.

        Verified against LTX2VideoTransformerBlock.named_modules():
        28 Linear layers per block (6 attention groups x 4 projections + 2 FFN groups x 2 layers).
        """
        return [
            # Video self-attention
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            # Video cross-attention (text)
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
            # Audio self-attention
            "audio_attn1.to_q", "audio_attn1.to_k", "audio_attn1.to_v", "audio_attn1.to_out.0",
            # Audio cross-attention (text)
            "audio_attn2.to_q", "audio_attn2.to_k", "audio_attn2.to_v", "audio_attn2.to_out.0",
            # Cross-modal attention (audio→video, video→audio)
            "audio_to_video_attn.to_q", "audio_to_video_attn.to_k",
            "audio_to_video_attn.to_v", "audio_to_video_attn.to_out.0",
            "video_to_audio_attn.to_q", "video_to_audio_attn.to_k",
            "video_to_audio_attn.to_v", "video_to_audio_attn.to_out.0",
            # Video FFN
            "ff.net.0.proj", "ff.net.2",
            # Audio FFN
            "audio_ff.net.0.proj", "audio_ff.net.2",
        ]

    @property
    def preprocessing_modules(self) -> List[str]:
        """Components needed for offline preprocessing (text encoding + connectors)."""
        return ['text_encoders', 'connectors']

    @property
    def inference_modules(self) -> List[str]:
        """Components needed during inference and training forward."""
        return ['transformer', 'vae', 'audio_vae', 'connectors', 'vocoder']

    # ============================== Encoding ==============================

    def encode_prompt(self, prompt, **kwargs):
        raise NotImplementedError("encode_prompt will be implemented in sub-step 6b.")

    def encode_image(self, images, **kwargs):
        """LTX2 T2AV does not use image conditioning."""
        return None

    def encode_video(self, videos, **kwargs):
        """LTX2 T2AV does not use video conditioning."""
        return None

    # ============================== Decoding ==============================

    def decode_latents(self, latents, **kwargs):
        raise NotImplementedError("decode_latents will be implemented in sub-step 6b.")

    # ============================== Forward / Inference ==============================

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward will be implemented in sub-step 6c.")

    def inference(self, *args, **kwargs):
        raise NotImplementedError("inference will be implemented in sub-step 6d.")
