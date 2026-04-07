# Step 6 — LTX2 Adapter Implementation Sub-Plan (v6-final)

## Environment

diffusers **0.38.0.dev0** — local editable install at `Flow-Factory/diffusers/`.
Source verified line-by-line against `diffusers/src/diffusers/pipelines/ltx2/pipeline_ltx2.py` (1226 lines).

---

## Sub-step 6a — Scaffold

**New files**: `models/ltx2/__init__.py`, `models/ltx2/ltx2_t2av.py`

### Imports

```python
from __future__ import annotations

import copy
from typing import Union, List, Dict, Any, Optional, Literal, ClassVar
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
from accelerate import Accelerator
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline, rescale_noise_cfg
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

from ..abc import BaseAdapter
from ...samples import T2AVSample
from ...hparams import *
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
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
```

### LTX2Sample

```python
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
```

### Adapter skeleton

```python
class LTX2_T2AV_Adapter(BaseAdapter):
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
        """Separate ODE-only scheduler for audio (avoids step_index collision with video)."""
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
        Verified against LTX2VideoTransformerBlock.named_modules() output.
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
            "audio_to_video_attn.to_q", "audio_to_video_attn.to_k", "audio_to_video_attn.to_v", "audio_to_video_attn.to_out.0",
            "video_to_audio_attn.to_q", "video_to_audio_attn.to_k", "video_to_audio_attn.to_v", "video_to_audio_attn.to_out.0",
            # Video FFN
            "ff.net.0.proj", "ff.net.2",
            # Audio FFN
            "audio_ff.net.0.proj", "audio_ff.net.2",
        ]

    @property
    def preprocessing_modules(self) -> List[str]:
        return ['text_encoders', 'connectors']

    @property
    def inference_modules(self) -> List[str]:
        return ['transformer', 'vae', 'audio_vae', 'connectors', 'vocoder']

    # ============================== Encoding (stubs for 6a) ==============================
    def encode_prompt(self, prompt, **kwargs):
        raise NotImplementedError

    def encode_image(self, images, **kwargs):
        return None

    def encode_video(self, videos, **kwargs):
        return None

    # ============================== Decoding (stub for 6a) ==============================
    def decode_latents(self, latents, **kwargs):
        raise NotImplementedError

    # ============================== Forward / Inference (stubs for 6a) ==============================
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def inference(self, *args, **kwargs):
        raise NotImplementedError
```

**Commit**: `[models/ltx2] feat: scaffold LTX2 adapter with sample dataclass and pipeline loading`
**Verify**: `from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter, LTX2Sample` succeeds.

---

## Sub-step 6b — Encoding & Decoding

### `encode_prompt()`

Delegates to `self.pipeline.encode_prompt()` for Gemma3 + `_pack_text_embeds`, then passes through connectors with additive mask. Follows the same `encode → concat [neg, pos] → connectors → split` flow as pipeline L941-966.

```python
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
        """Encode text prompts into connector embeddings for video and audio streams."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = device or self.pipeline.text_encoder.device

        # 1. Pipeline handles Gemma3 encoding + _pack_text_embeds (source L941-958)
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

        # 2. Concat [negative, positive] for connectors if CFG (source L959-961)
        if do_classifier_free_guidance:
            combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            combined_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        else:
            combined_embeds = prompt_embeds
            combined_mask = prompt_attention_mask

        # 3. Connectors with additive mask (source L963-966)
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

    # ============================== Image / Video / Audio Encoding ==============================
    def encode_image(self, images, **kwargs):
        """LTX2 T2AV does not use image conditioning."""
        return None

    def encode_video(self, videos, **kwargs):
        """LTX2 T2AV does not use video conditioning."""
        return None
```

### `decode_latents()`

Matches pipeline L1172-1218 exactly. Video: unpack → denorm → decode. Audio: denorm → unpack → decode → vocoder.

```python
    # ============================== Decoding ==============================
    def decode_latents(
        self,
        video_latents: torch.Tensor,
        audio_latents: Optional[torch.Tensor] = None,
        height: int = 512, width: int = 768,
        num_frames: int = 121, frame_rate: float = 24.0,
        decode_timestep: float = 0.0,
        decode_noise_scale: Optional[float] = None,
        output_type: str = 'pt',
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        """Decode packed latents to video frames and audio waveform."""
        device = video_latents.device
        batch_size = video_latents.shape[0]
        vae = self.pipeline.vae
        patch_size = self.pipeline.transformer_spatial_patch_size     # 1
        patch_size_t = self.pipeline.transformer_temporal_patch_size  # 1
        vae_spatial = self.pipeline.vae_spatial_compression_ratio     # 32
        vae_temporal = self.pipeline.vae_temporal_compression_ratio   # 8

        # --- Video decode (source L1172-1214) ---
        latent_h = height // vae_spatial
        latent_w = width // vae_spatial
        latent_f = (num_frames - 1) // vae_temporal + 1

        vid = self.pipeline._unpack_latents(video_latents, latent_f, latent_h, latent_w, patch_size, patch_size_t)
        vid = self.pipeline._denormalize_latents(vid, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)

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

        vid = vid.to(vae.dtype)
        video = vae.decode(vid, vae_timestep, return_dict=False)[0]
        video = self.pipeline.video_processor.postprocess_video(video, output_type=output_type)

        # --- Audio decode (source L1184-1218) ---
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

            # Denormalize FIRST, then unpack (order differs from video!)
            aud = self.pipeline._denormalize_audio_latents(audio_latents, audio_vae.latents_mean, audio_vae.latents_std)
            aud = self.pipeline._unpack_audio_latents(aud, audio_num_frames, num_mel_bins=latent_mel_bins)
            aud = aud.to(audio_vae.dtype)
            mel = audio_vae.decode(aud, return_dict=False)[0]
            audio = self.pipeline.vocoder(mel)

        return video, audio
```

**Commit**: `[models/ltx2] feat: implement encode_prompt and decode_latents for LTX2`

---

## Sub-step 6c — Forward (single denoising step)

Matches pipeline L1097-1154 denoising loop body. CFG in velocity-space with `[uncond, cond]` chunk order.

```python
    # ============================== Forward ==============================
    def forward(
        self,
        t: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
        latents: torch.Tensor,
        next_latents: Optional[torch.Tensor] = None,
        audio_latents: Optional[torch.Tensor] = None,
        # Text embeddings (from connectors)
        connector_prompt_embeds: torch.Tensor = None,
        connector_audio_prompt_embeds: torch.Tensor = None,
        connector_attention_mask: torch.Tensor = None,
        negative_connector_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_attention_mask: Optional[torch.Tensor] = None,
        # Guidance
        guidance_scale: float = 4.0,
        guidance_rescale: float = 0.0,
        # Generation shape (pixel-space, for computing latent dims + RoPE)
        height: int = 512, width: int = 768,
        num_frames: int = 121, frame_rate: float = 24.0,
        audio_num_frames: Optional[int] = None,
        # Positional coords (cached from inference loop)
        video_coords: Optional[torch.Tensor] = None,
        audio_coords: Optional[torch.Tensor] = None,
        # SDE control
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = ['next_latents', 'log_prob', 'noise_pred'],
        **kwargs,
    ) -> SDESchedulerOutput:
        """Single denoising step: joint transformer forward + video SDE / audio ODE scheduler steps."""
        batch_size = latents.shape[0]
        device = latents.device
        do_cfg = guidance_scale > 1.0 and negative_connector_prompt_embeds is not None

        # --- Compute latent dims (for RoPE if coords not cached) ---
        vae_spatial = self.pipeline.vae_spatial_compression_ratio
        vae_temporal = self.pipeline.vae_temporal_compression_ratio
        latent_h = height // vae_spatial
        latent_w = width // vae_spatial
        latent_f = (num_frames - 1) // vae_temporal + 1
        if audio_num_frames is None:
            duration_s = num_frames / frame_rate
            sr = self.pipeline.audio_sampling_rate
            hop = self.pipeline.audio_hop_length
            tc = self.pipeline.audio_vae_temporal_compression_ratio
            audio_num_frames = round(duration_s * sr / hop / tc)

        # --- Prepare RoPE coords if not cached ---
        if video_coords is None:
            video_coords = self.pipeline.transformer.rope.prepare_video_coords(
                batch_size, latent_f, latent_h, latent_w, device, fps=frame_rate)
        if audio_coords is None:
            audio_coords = self.pipeline.transformer.audio_rope.prepare_audio_coords(
                batch_size, audio_num_frames, device)

        # --- 1. Prepare CFG inputs (source L1097-1105) ---
        if do_cfg:
            lat_in = torch.cat([latents, latents])
            aud_in = torch.cat([audio_latents, audio_latents])
            text_in = torch.cat([negative_connector_prompt_embeds, connector_prompt_embeds])
            audio_text_in = torch.cat([negative_connector_audio_prompt_embeds, connector_audio_prompt_embeds])
            mask_in = torch.cat([negative_connector_attention_mask, connector_attention_mask])
            vid_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            aud_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))
            ts = t.expand(batch_size * 2)
        else:
            lat_in, aud_in = latents, audio_latents
            text_in = connector_prompt_embeds
            audio_text_in = connector_audio_prompt_embeds
            mask_in = connector_attention_mask
            vid_coords, aud_coords = video_coords, audio_coords
            ts = t.expand(batch_size)

        # --- 2. Transformer forward (source L1107-1128) ---
        lat_in = lat_in.to(connector_prompt_embeds.dtype)
        aud_in = aud_in.to(connector_prompt_embeds.dtype)

        with self.pipeline.transformer.cache_context("cond_uncond"):
            video_pred, audio_pred = self.transformer(
                hidden_states=lat_in,
                audio_hidden_states=aud_in,
                encoder_hidden_states=text_in,
                audio_encoder_hidden_states=audio_text_in,
                timestep=ts,
                encoder_attention_mask=mask_in,
                audio_encoder_attention_mask=mask_in,
                num_frames=latent_f,
                height=latent_h,
                width=latent_w,
                fps=frame_rate,
                audio_num_frames=audio_num_frames,
                video_coords=vid_coords,
                audio_coords=aud_coords,
                attention_kwargs=None,
                return_dict=False,
            )
        video_pred = video_pred.float()
        audio_pred = audio_pred.float()

        # --- 3. CFG in velocity-space (source L1130-1148) ---
        if do_cfg:
            v_uncond, v_cond = video_pred.chunk(2)
            video_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
            a_uncond, a_cond = audio_pred.chunk(2)
            audio_pred = a_uncond + guidance_scale * (a_cond - a_uncond)
            if guidance_rescale > 0:
                video_pred = rescale_noise_cfg(video_pred, v_cond, guidance_rescale=guidance_rescale)
                audio_pred = rescale_noise_cfg(audio_pred, a_cond, guidance_rescale=guidance_rescale)

        # --- 4. Video: SDE scheduler step (with log_prob) ---
        video_output = self.scheduler.step(
            noise_pred=video_pred, timestep=t, latents=latents,
            timestep_next=t_next, next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True, return_kwargs=return_kwargs,
            noise_level=noise_level,
        )

        # --- 5. Audio: ODE scheduler step (deterministic, no log_prob) ---
        audio_output = self.audio_scheduler.step(
            noise_pred=audio_pred, timestep=t, latents=audio_latents,
            timestep_next=t_next, compute_log_prob=False,
            return_dict=True, return_kwargs=['next_latents'],
            dynamics_type='ODE',
        )

        video_output.audio_next_latents = audio_output.next_latents
        return video_output
```

**Commit**: `[models/ltx2] feat: implement forward() with CFG and dual scheduler steps`

---

## Sub-step 6d — Inference loop + Registry

### `inference()`

Follows the standard Flow-Factory inference pattern (encode → prepare latents → set timesteps → trajectory collectors → denoising loop → decode → construct samples). Source reference: pipeline L908-1226.

```python
    # ============================== Inference ==============================
    @torch.no_grad()
    def inference(
        self,
        # Raw inputs
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # Generation shape
        height: int = 512, width: int = 768,
        num_frames: int = 121, frame_rate: float = 24.0,
        # Scheduling
        num_inference_steps: int = 40,
        sigmas: Optional[List[float]] = None,
        # Guidance
        guidance_scale: float = 4.0,
        guidance_rescale: float = 0.0,
        noise_scale: float = 0.0,
        # Generator
        generator: Optional[torch.Generator] = None,
        # Pre-encoded inputs
        prompt_ids: Optional[torch.Tensor] = None,
        connector_prompt_embeds: Optional[torch.Tensor] = None,
        connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
        connector_attention_mask: Optional[torch.Tensor] = None,
        negative_connector_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_attention_mask: Optional[torch.Tensor] = None,
        # Decode options
        decode_timestep: float = 0.0,
        decode_noise_scale: Optional[float] = None,
        max_sequence_length: int = 1024,
        # RL-specific
        compute_log_prob: bool = True,
        trajectory_indices: TrajectoryIndicesType = 'all',
        extra_call_back_kwargs: List[str] = [],
        **kwargs,
    ) -> List[LTX2Sample]:
        """Full denoising inference loop for LTX2 text-to-audio-video generation."""
        device = self.device

        # ========== 1. Encode prompts ==========
        if connector_prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt=prompt, negative_prompt=negative_prompt,
                do_classifier_free_guidance=(guidance_scale > 1.0),
                max_sequence_length=max_sequence_length, device=device,
            )
            prompt_ids = encoded['prompt_ids']
            connector_prompt_embeds = encoded['connector_prompt_embeds']
            connector_audio_prompt_embeds = encoded['connector_audio_prompt_embeds']
            connector_attention_mask = encoded['connector_attention_mask']
            negative_connector_prompt_embeds = encoded.get('negative_connector_prompt_embeds')
            negative_connector_audio_prompt_embeds = encoded.get('negative_connector_audio_prompt_embeds')
            negative_connector_attention_mask = encoded.get('negative_connector_attention_mask')
        else:
            connector_prompt_embeds = connector_prompt_embeds.to(device)
            connector_audio_prompt_embeds = connector_audio_prompt_embeds.to(device)
            connector_attention_mask = connector_attention_mask.to(device)
            if negative_connector_prompt_embeds is not None:
                negative_connector_prompt_embeds = negative_connector_prompt_embeds.to(device)
                negative_connector_audio_prompt_embeds = negative_connector_audio_prompt_embeds.to(device)
                negative_connector_attention_mask = negative_connector_attention_mask.to(device)

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = connector_prompt_embeds.shape[0]

        # ========== 2. Compute dimensions (source L968-1007) ==========
        vae_spatial = self.pipeline.vae_spatial_compression_ratio
        vae_temporal = self.pipeline.vae_temporal_compression_ratio
        latent_h = height // vae_spatial
        latent_w = width // vae_spatial
        latent_f = (num_frames - 1) // vae_temporal + 1

        duration_s = num_frames / frame_rate
        sr = self.pipeline.audio_sampling_rate
        hop = self.pipeline.audio_hop_length
        audio_temporal_compression = self.pipeline.audio_vae_temporal_compression_ratio
        audio_mel_compression = self.pipeline.audio_vae_mel_compression_ratio
        audio_num_frames = round(duration_s * sr / hop / audio_temporal_compression)
        num_mel_bins = self.pipeline.audio_vae.config.mel_bins if getattr(self.pipeline, 'audio_vae', None) is not None else 64

        # ========== 3. Prepare latents (source L989-1039) ==========
        video_latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=self.transformer_config.in_channels,
            height=height, width=width, num_frames=num_frames,
            noise_scale=noise_scale, dtype=torch.float32,
            device=device, generator=generator,
        )
        audio_latents = self.pipeline.prepare_audio_latents(
            batch_size=batch_size,
            num_channels_latents=(
                self.pipeline.audio_vae.config.latent_channels
                if getattr(self.pipeline, 'audio_vae', None) is not None else 8
            ),
            audio_latent_length=audio_num_frames,
            num_mel_bins=num_mel_bins,
            noise_scale=noise_scale, dtype=torch.float32,
            device=device, generator=generator,
        )

        # ========== 4. Set timesteps (source L1041-1069) ==========
        video_seq_len = latent_f * latent_h * latent_w
        mu = calculate_shift(
            video_seq_len,
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        timesteps = set_scheduler_timesteps(
            self.scheduler, num_inference_steps, device=device, sigmas=sigmas, mu=mu,
        )
        set_scheduler_timesteps(
            self.audio_scheduler, num_inference_steps, device=device, sigmas=sigmas, mu=mu,
        )

        # ========== 5. Prepare positional coords (source L1078-1087) ==========
        video_coords = self.pipeline.transformer.rope.prepare_video_coords(
            batch_size, latent_f, latent_h, latent_w, device, fps=frame_rate,
        )
        audio_coords = self.pipeline.transformer.audio_rope.prepare_audio_coords(
            batch_size, audio_num_frames, device,
        )

        # ========== 6. Setup trajectory collectors ==========
        video_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        audio_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        video_latents = self.cast_latents(video_latents)
        video_collector.collect(video_latents, step_idx=0)
        audio_collector.collect(audio_latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        # ========== 7. Denoising loop ==========
        for i, t in enumerate(timesteps):
            noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kw = list(set(['next_latents', 'log_prob', 'noise_pred'] + extra_call_back_kwargs))
            current_compute_lp = compute_log_prob and noise_level > 0

            output = self.forward(
                t=t, t_next=t_next,
                latents=video_latents, audio_latents=audio_latents,
                connector_prompt_embeds=connector_prompt_embeds,
                connector_audio_prompt_embeds=connector_audio_prompt_embeds,
                connector_attention_mask=connector_attention_mask,
                negative_connector_prompt_embeds=negative_connector_prompt_embeds,
                negative_connector_audio_prompt_embeds=negative_connector_audio_prompt_embeds,
                negative_connector_attention_mask=negative_connector_attention_mask,
                guidance_scale=guidance_scale, guidance_rescale=guidance_rescale,
                height=height, width=width, num_frames=num_frames, frame_rate=frame_rate,
                audio_num_frames=audio_num_frames,
                video_coords=video_coords, audio_coords=audio_coords,
                noise_level=noise_level, compute_log_prob=current_compute_lp,
                return_kwargs=return_kw,
            )

            video_latents = self.cast_latents(output.next_latents)
            audio_latents = output.audio_next_latents
            video_collector.collect(video_latents, i + 1)
            audio_collector.collect(audio_latents, i + 1)
            if current_compute_lp:
                log_prob_collector.collect(output.log_prob, i)
            callback_collector.collect_step(i, output, extra_call_back_kwargs, capturable={'noise_level': noise_level})

        # ========== 8. Decode ==========
        video, audio_waveform = self.decode_latents(
            video_latents, audio_latents,
            height=height, width=width, num_frames=num_frames, frame_rate=frame_rate,
            decode_timestep=decode_timestep, decode_noise_scale=decode_noise_scale,
            output_type='pt', generator=generator,
        )

        # ========== 9. Construct samples (per-batch, NO batch dimension) ==========
        all_vid_lats = video_collector.get_result()
        vid_lat_map = video_collector.get_index_map()
        all_aud_lats = audio_collector.get_result()
        aud_lat_map = audio_collector.get_index_map()
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        lp_map = log_prob_collector.get_index_map() if compute_log_prob else None
        cb_res = callback_collector.get_result()
        cb_map = callback_collector.get_index_map()

        prompt_list = prompt if isinstance(prompt, list) else [prompt] * batch_size

        samples = [
            LTX2Sample(
                # Video trajectory (for RL training)
                timesteps=timesteps,
                all_latents=torch.stack([l[b] for l in all_vid_lats], dim=0) if all_vid_lats else None,
                log_probs=torch.stack([l[b] for l in all_log_probs], dim=0) if all_log_probs else None,
                latent_index_map=vid_lat_map,
                log_prob_index_map=lp_map,
                # Audio trajectory (for reconstruction, NOT RL)
                audio_all_latents=torch.stack([l[b] for l in all_aud_lats], dim=0) if all_aud_lats else None,
                audio_latent_index_map=aud_lat_map,
                # Generated media
                video=video[b],
                audio=audio_waveform[b] if audio_waveform is not None else None,
                # Metadata
                height=height, width=width,
                # Prompt
                prompt=prompt_list[b],
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                # Connector embeddings (for training forward)
                connector_prompt_embeds=connector_prompt_embeds[b],
                connector_audio_prompt_embeds=connector_audio_prompt_embeds[b],
                connector_attention_mask=connector_attention_mask[b],
                negative_connector_prompt_embeds=(
                    negative_connector_prompt_embeds[b] if negative_connector_prompt_embeds is not None else None
                ),
                negative_connector_audio_prompt_embeds=(
                    negative_connector_audio_prompt_embeds[b] if negative_connector_audio_prompt_embeds is not None else None
                ),
                negative_connector_attention_mask=(
                    negative_connector_attention_mask[b] if negative_connector_attention_mask is not None else None
                ),
                extra_kwargs={
                    **{k: v[b] for k, v in cb_res.items()},
                    'callback_index_map': cb_map,
                    'num_frames': num_frames,
                    'frame_rate': frame_rate,
                    'duration_s': duration_s,
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()
        return samples
```

### Registry

```python
# models/registry.py — add one line:
'ltx2_t2av': 'flow_factory.models.ltx2.ltx2_t2av.LTX2_T2AV_Adapter',
```

**Commit**: `[models/ltx2] feat: implement inference loop and register ltx2_t2av adapter`

---

## Verification Checklist

Per Flow-Factory `guidance/new_model.md`:

- [ ] `load_pipeline()` uses `low_cpu_mem_usage=False`
- [ ] `default_target_modules` verified against `LTX2VideoTransformerBlock.named_modules()` output
- [ ] `preprocessing_modules` = `['text_encoders', 'connectors']` (no VAE needed for T2AV)
- [ ] `inference_modules` = `['transformer', 'vae', 'audio_vae', 'connectors', 'vocoder']`
- [ ] `encode_prompt()` returns dict with `prompt_ids` + connector embeddings
- [ ] `encode_image()` / `encode_video()` return `None` (pure T2AV)
- [ ] `inference()` accepts both raw and pre-encoded inputs
- [ ] `inference()` returns `List[LTX2Sample]` with NO batch dimension on fields
- [ ] `forward()` ends with `self.scheduler.step()` returning `SDESchedulerOutput`
- [ ] `LTX2Sample._shared_fields` correctly set
- [ ] Registry entry added
- [ ] License header: Apache 2.0 with `Copyright 2026 Jayce-Ping`
- [ ] All code comments and docstrings in English
