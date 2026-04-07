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

    # ============================== Text Encoding ==============================

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        max_sequence_length: int = 1024,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode text prompts into connector embeddings for video and audio streams.

        Delegates to pipeline.encode_prompt() for Gemma3 + _pack_text_embeds, then
        passes through connectors with additive mask. Matches pipeline L941-966.

        Returns:
            Dict with keys: prompt_ids, connector_prompt_embeds, connector_audio_prompt_embeds,
            connector_attention_mask, and their negative_* counterparts if CFG is enabled.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = device or self.pipeline.text_encoder.device

        # 1. Pipeline handles Gemma3 encoding + _pack_text_embeds (pipeline L941-958)
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            self.pipeline.encode_prompt(
                prompt=prompt, negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                max_sequence_length=max_sequence_length, device=device,
            )
        )

        # Tokenize for prompt_ids (needed for unique_id hashing in reward grouping)
        prompt_ids = self.pipeline.tokenizer(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)

        # 2. Concat [negative, positive] for connectors if CFG (pipeline L959-961)
        if do_classifier_free_guidance:
            combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            combined_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        else:
            combined_embeds = prompt_embeds
            combined_mask = prompt_attention_mask

        # 3. Connectors with additive mask (pipeline L963-966)
        additive_mask = (1 - combined_mask.to(combined_embeds.dtype)) * -1000000.0
        connector_out, connector_audio_out, connector_mask = self.pipeline.connectors(
            combined_embeds, additive_mask, additive_mask=True,
        )

        # 4. Split neg/pos if CFG
        if do_classifier_free_guidance:
            neg_conn, pos_conn = connector_out.chunk(2)
            neg_audio_conn, pos_audio_conn = connector_audio_out.chunk(2)
            neg_conn_mask, pos_conn_mask = connector_mask.chunk(2)
        else:
            pos_conn = connector_out
            pos_audio_conn = connector_audio_out
            pos_conn_mask = connector_mask
            neg_conn = neg_audio_conn = neg_conn_mask = None

        return {
            'prompt_ids': prompt_ids,
            'connector_prompt_embeds': pos_conn,
            'connector_audio_prompt_embeds': pos_audio_conn,
            'connector_attention_mask': pos_conn_mask,
            'negative_connector_prompt_embeds': neg_conn,
            'negative_connector_audio_prompt_embeds': neg_audio_conn,
            'negative_connector_attention_mask': neg_conn_mask,
        }

    # ============================== Image / Video Encoding ==============================

    def encode_image(self, images, **kwargs):
        """LTX2 T2AV does not use image conditioning."""
        return None

    def encode_video(self, videos, **kwargs):
        """LTX2 T2AV does not use video conditioning."""
        return None

    # ============================== Decoding ==============================

    def decode_latents(
        self,
        video_latents: torch.Tensor,
        audio_latents: Optional[torch.Tensor] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        decode_timestep: float = 0.0,
        decode_noise_scale: Optional[float] = None,
        output_type: str = 'pt',
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        """Decode packed latents to video frames and audio waveform.

        Video: unpack → denormalize → VAE decode (pipeline L1172-1214).
        Audio: denormalize → unpack → audio_vae decode → vocoder (pipeline L1184-1218).
        Note the different operation order for video vs audio.
        """
        device = video_latents.device
        batch_size = video_latents.shape[0]
        vae = self.pipeline.vae
        patch_size = self.pipeline.transformer_spatial_patch_size     # 1
        patch_size_t = self.pipeline.transformer_temporal_patch_size  # 1
        vae_spatial = self.pipeline.vae_spatial_compression_ratio     # 32
        vae_temporal = self.pipeline.vae_temporal_compression_ratio   # 8

        # --- Video decode (pipeline L1172-1214) ---
        latent_h = height // vae_spatial
        latent_w = width // vae_spatial
        latent_f = (num_frames - 1) // vae_temporal + 1

        # 1. Unpack: (B, seq, C) → (B, C, F, H, W)
        vid = self.pipeline._unpack_latents(video_latents, latent_f, latent_h, latent_w, patch_size, patch_size_t)
        # 2. Denormalize: latents * std / scaling_factor + mean
        vid = self.pipeline._denormalize_latents(vid, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)
        # 3. Timestep conditioning + decode noise injection (pipeline L1195-1210)
        if not vae.config.timestep_conditioning:
            vae_timestep = None
        else:
            noise = torch.randn_like(vid)
            _dt = [decode_timestep] * batch_size if not isinstance(decode_timestep, list) else decode_timestep
            _dns = _dt if decode_noise_scale is None else (
                [decode_noise_scale] * batch_size if not isinstance(decode_noise_scale, list) else decode_noise_scale
            )
            vae_timestep = torch.tensor(_dt, device=device, dtype=vid.dtype)
            _dns_t = torch.tensor(_dns, device=device, dtype=vid.dtype)[:, None, None, None, None]
            vid = (1 - _dns_t) * vid + _dns_t * noise
        # 4. VAE decode
        vid = vid.to(vae.dtype)
        video = vae.decode(vid, vae_timestep, return_dict=False)[0]
        video = self.pipeline.video_processor.postprocess_video(video, output_type=output_type)

        # --- Audio decode (pipeline L1184-1218) ---
        audio = None
        if audio_latents is not None:
            audio_vae = self.pipeline.audio_vae
            mel_compression = self.pipeline.audio_vae_mel_compression_ratio
            temporal_compression = self.pipeline.audio_vae_temporal_compression_ratio
            num_mel_bins = audio_vae.config.mel_bins if getattr(self.pipeline, 'audio_vae', None) is not None else 64
            latent_mel_bins = num_mel_bins // mel_compression

            duration_s = num_frames / frame_rate
            sr = self.pipeline.audio_sampling_rate
            hop = self.pipeline.audio_hop_length
            audio_num_frames = round(duration_s * sr / hop / temporal_compression)

            # 1. Denormalize FIRST, then unpack (order differs from video!)
            aud = self.pipeline._denormalize_audio_latents(
                audio_latents, audio_vae.latents_mean, audio_vae.latents_std
            )
            # 2. Unpack: (B, seq, C) → (B, C, T, mel_bins)
            aud = self.pipeline._unpack_audio_latents(aud, audio_num_frames, num_mel_bins=latent_mel_bins)
            # 3. Audio VAE decode → mel spectrogram
            aud = aud.to(audio_vae.dtype)
            mel = audio_vae.decode(aud, return_dict=False)[0]
            # 4. Vocoder → waveform
            audio = self.pipeline.vocoder(mel)

        return video, audio

    # ============================== Forward / Inference ==============================

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward will be implemented in sub-step 6c.")

    def inference(self, *args, **kwargs):
        raise NotImplementedError("inference will be implemented in sub-step 6d.")
