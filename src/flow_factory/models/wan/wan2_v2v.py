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

# src/flow_factory/models/wan/wan2_v2v.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple, Literal, ClassVar
from dataclasses import dataclass
import logging
from collections import defaultdict

import numpy as np
import torch
from diffusers.pipelines.wan.pipeline_wan_video2video import WanVideoToVideoPipeline, prompt_clean, retrieve_timesteps
from PIL import Image
from accelerate import Accelerator
from peft import PeftModel

from ..abc import BaseAdapter
from ..samples import V2VSample
from ...hparams import *
from ...scheduler import UniPCMultistepSDESchedulerOutput, UniPCMultistepSDEScheduler
from ...utils.base import filter_kwargs
from ...utils.video import (
    VideoSingle,
    VideoBatch,
    MultiVideoBatch,
    is_video,
    is_video_frame_list,
    is_video_batch,
    is_multi_video_batch,
    standardize_video_batch,
)
from ...utils.trajectory_collector import (
    TrajectoryCollector, 
    TrajectoryIndicesType, 
    create_trajectory_collector,
)
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


WanPipelineVideoInput = Union[
    List[Image.Image], # One video as list of PIL images
    torch.Tensor, # One video as tensor (T, C, H, W) or a batch of videos (B, T, C, H, W)
    np.ndarray, # One video as numpy array (T, H, W, C) or a batch of videos (B, T, H, W, C)
    List[Union[torch.Tensor, np.ndarray, List[Image.Image]]] # A list of videos with various sizes
]


@dataclass
class WanV2VSample(V2VSample):
    """Sample dataclass for Wan V2V outputs."""
    _shared_fields: ClassVar[frozenset[str]] = frozenset({'strength', 'num_inference_steps'})
    strength: Optional[float] = None
    num_inference_steps: Optional[int] = None

class Wan2_V2V_Adapter(BaseAdapter):
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
        self._has_warned_multi_video_input = False
        self.pipeline: WanVideoToVideoPipeline
        self.scheduler: UniPCMultistepSDEScheduler
    
    def load_pipeline(self) -> WanVideoToVideoPipeline:
        return WanVideoToVideoPipeline.from_pretrained(
            self.model_args.model_name_or_path,
        )
    
    def apply_lora(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]] = ['transformer', 'transformer_2'],
    ) -> Union[PeftModel, Dict[str, PeftModel]]:
        return super().apply_lora(target_modules=target_modules, components=components)
    
    # ============================ Module Management ============================
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Wan transformer."""
        return [
            # --- Self Attention ---
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            
            # --- Cross Attention ---
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",

            # --- Feed Forward Network ---
            "ffn.net.0.proj", "ffn.net.2"
        ]
    
    @property
    def inference_modules(self) -> List[str]:
        """Modules that are required for inference and forward"""
        if self.pipeline.config.boundary_ratio is None or self.pipeline.config.boundary_ratio <= 0:
            return ['transformer', 'vae']

        if self.pipeline.config.boundary_ratio >= 1:
            return ['transformer_2', 'vae']

        return ['transformer', 'transformer_2', 'vae']
    
    # ======================== Component Getters & Setters ========================
    @property
    def transformer_2(self) -> torch.nn.Module:
        return self.get_component('transformer_2')

    @transformer_2.setter
    def transformer_2(self, module: torch.nn.Module):
        self.set_prepared('transformer_2', module)

    # ============================ Encoding & Decoding ============================
    # --------------------------- Prompt Encoding --------------------------
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.pipeline.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        return text_input_ids, prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_ids, prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        results = {
            'prompt_ids': prompt_ids,
            'prompt_embeds': prompt_embeds,
        }

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_ids, negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            results.update({
                "negative_prompt_ids": negative_prompt_ids,
                "negative_prompt_embeds": negative_prompt_embeds
            })

        return results
    # --------------------------- Image Encoding --------------------------
    def encode_image(self, images: Union[Image.Image, torch.Tensor, List[torch.Tensor]]) -> None:
        """Skip this for Wan V2V as the pipeline handles encoding internally."""
        pass
    # --------------------------- Video Encoding --------------------------
    def encode_video(self, videos: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        """Skip this for Wan V2V as the pipeline handles encoding internally."""
        pass

    def _standardize_video_input(
        self,
        videos: Union[VideoSingle, VideoBatch, MultiVideoBatch],
        output_type: Literal['np', 'pt', 'pil'] = 'pt',
    ) -> VideoBatch:
        """Convert a batch/list of videos into the target format."""
        if is_video_frame_list(videos):
            # One video as list of PIL images
            videos = [videos]
        if is_multi_video_batch(videos):
            # A list of video batches
            if any(len(batch) > 1 for batch in videos) and not self._has_warned_multi_video_input:
                self._has_warned_multi_video_input = True
                logger.warning(
                    "Multiple condition videos are not supported for Wan2 V2V. Only the first video of each batch will be used."
                )
            videos = [batch[0] for batch in videos]
        # To a batch of videos
        standardized_videos = standardize_video_batch(
            videos,
            output_type=output_type,
        )
        return standardized_videos

    # --------------------------- Video Decoding --------------------------
    def decode_latents(self, latents: torch.Tensor, output_type: Literal['pt', 'pil', 'np'] = 'pil') -> torch.Tensor:
        """Decode the latents using the VAE decoder."""
        latents = latents.float()
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(1, self.pipeline.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.pipeline.vae.decode(latents, return_dict=False)[0]

        video = self.pipeline.video_processor.postprocess_video(video, output_type=output_type)
        return video
    

    # ============================ Inference ============================
    def inference(
        self,
        # Ordinary inputs
        videos: Union[VideoSingle, VideoBatch],
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        strength: float = 0.8,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # Encoded Prompt
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # Other kwargs
        compute_log_prob: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = 'all',
    ) -> List[WanV2VSample]:
        # 1. Setup args
        device = self.device
        dtype = self.pipeline.transformer.dtype
        do_classifier_free_guidance = guidance_scale > 1.0
        height = height or self.pipeline.transformer.config.sample_height * self.pipeline.vae_scale_factor_spatial
        width = width or self.pipeline.transformer.config.sample_width * self.pipeline.vae_scale_factor_spatial

        # 2. Encode prompt
        if prompt_embeds is None or negative_prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_ids = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
            negative_prompt_ids = encoded.get("negative_prompt_ids", None)
            negative_prompt_embeds = encoded.get("negative_prompt_embeds", None)
        else:
            prompt_embeds = prompt_embeds.to(device)
            negative_prompt_embeds = negative_prompt_embeds.to(device)

        batch_size = prompt_embeds.shape[0]

        # 3. Set timesteps
        input_inference_steps = num_inference_steps # 50
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device) # [1000, ..., 0], 50
        timesteps, num_inference_steps = self.pipeline.get_timesteps(num_inference_steps, timesteps, strength, device) # strength=0.8, [800, ..., 0], 40
        latent_timestep = timesteps[:1].repeat(batch_size)
        self.pipeline._num_timesteps = len(timesteps)

        # 4. Prepare latents
        videos = self._standardize_video_input(videos, output_type='pt')
        videos = self.pipeline.video_processor.preprocess_video(videos, height=height, width=width).to(
            device, dtype=dtype
        )

        num_channels_latents = self.pipeline.transformer.config.in_channels
        latents = self.pipeline.prepare_latents(
            video=videos,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=None,
            timestep=latent_timestep,
        )

        # 5. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.pipeline._num_timesteps = len(timesteps)

        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        extra_call_back_res = defaultdict(list)

        for i, t in enumerate(timesteps):
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kwargs = list(set(['next_latents', 'log_prob', 'noise_pred'] + extra_call_back_kwargs))

            output = self.forward(
                t=t,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                attention_kwargs=attention_kwargs,
                compute_log_prob=compute_log_prob and current_noise_level > 0,
                return_kwargs=return_kwargs,
                noise_level=current_noise_level,
            )

            latents = output.next_latents
            latent_collector.collect(latents, i + 1)
            if compute_log_prob:
                log_prob_collector.collect(output.log_prob, i)

            if extra_call_back_kwargs:
                capturable = {'noise_level': current_noise_level}
                for key in extra_call_back_kwargs:
                    if key in capturable and capturable[key] is not None:
                        extra_call_back_res[key].append(capturable[key])
                    elif hasattr(output, key):
                        val = getattr(output, key)
                        if val is not None:
                            extra_call_back_res[key].append(val)

        
        self._current_timestep = None

        # 7. Decode latents to videos (list of pil images)
        decoded_videos = self.decode_latents(latents, output_type='pt')

        # 8. Prepare output samples

        # Transpose `extra_call_back_res` lists to have batch dimension first
        # (T, B, ...) -> (B, T, ...)
        extra_call_back_res = {
            k: torch.stack(v, dim=1)
            if isinstance(v[0], torch.Tensor) else v
            for k, v in extra_call_back_res.items()
        }
        all_latents = latent_collector.get_result()
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        samples = [
            WanV2VSample(
                # Denoising trajectory
                timesteps=timesteps,
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0) if all_latents is not None else None,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if all_log_probs is not None else None,
                # Generated video & metadata
                video=decoded_videos[b],
                height=height,
                width=width,
                # Prompt info
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b],
                prompt_embeds=prompt_embeds[b],
                # Negative prompt info
                negative_prompt=negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt,
                negative_prompt_ids=negative_prompt_ids[b] if negative_prompt_ids is not None else None,
                negative_prompt_embeds=negative_prompt_embeds[b] if negative_prompt_embeds is not None else None,
                # Condition Video
                condition_videos=videos[b],
                # Extra kwargs
                strength=strength,
                num_inference_steps=input_inference_steps,
                extra_kwargs={
                    **{k: v[b] for k, v in extra_call_back_res.items()}
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()

        return samples
    
    # =========================== Forward ===========================
    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        # Optional for CFG
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 5.0,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        noise_level: Optional[float] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = ['noise_pred', 'next_latents', 'next_latents_mean', 'std_dev_t', 'dt', 'log_prob'],
    ) -> UniPCMultistepSDESchedulerOutput:
        """
        Core forward pass for V2V generation.

        Args:
            t: Current timestep tensor.
            t_next: Next timestep tensor.
            latents: Current latent representations (B, C, T, H, W).
            prompt_embeds: Text prompt embeddings.
            negative_prompt_embeds: Optional negative prompt embeddings (for CFG).
            guidance_scale: CFG scale factor.
            next_latents: Optional target latents for log-prob computation.
            noise_level: Current noise level for SDE sampling.
            attention_kwargs: Optional kwargs for attention layers.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.

        Returns:
            UniPCMultistepSDESchedulerOutput containing requested outputs.
        """
        # 1. Prepare variables
        batch_size = latents.shape[0]
        device = latents.device
        dtype = self.pipeline.transformer.dtype

        do_classifier_free_guidance = (
            negative_prompt_embeds is not None and guidance_scale > 1.0
        )

        # 2. Prepare timestep
        timestep = t.expand(batch_size)
        latent_model_input = latents.to(dtype)

        # 3. Transformer forward pass
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

        # 4. Apply CFG
        if do_classifier_free_guidance:
            noise_uncond = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        # 5. Scheduler step
        output = self.scheduler.step(
            noise_pred=noise_pred,
            timestep=t,
            latents=latents,
            timestep_next=t_next,
            next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            return_kwargs=return_kwargs,
            noise_level=noise_level,
        )

        return output