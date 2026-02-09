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

# src/flow_factory/models/z_image/z_image.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple, ClassVar, Literal
from dataclasses import dataclass
from PIL import Image
from collections import defaultdict
import logging

import torch
from accelerate import Accelerator
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

from ..abc import BaseAdapter
from ...samples import T2ISample
from ...hparams import *
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SDESchedulerOutput,
    set_scheduler_timesteps
)
from ...utils.trajectory_collector import (
    TrajectoryCollector,
    CallbackCollector,
    TrajectoryIndicesType,
    create_trajectory_collector,
    create_callback_collector,
)
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class ZImageSample(T2ISample):
    # Class var
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    # Obj var - no extra

class ZImageAdapter(BaseAdapter):
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: ZImagePipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

    def load_pipeline(self) -> ZImagePipeline:
        return ZImagePipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Z-Image transformer."""
        return [
            "attention.to_k", "attention.to_q", "attention.to_v", "attention.to_out.0",
            "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
        ]

    # ======================== Encoding / Decoding ======================== 
    # ----------------------- Prompt Encoding -----------------------   
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
    ) -> Tuple[List[torch.FloatTensor], torch.Tensor]:
        device = device or self.text_encoder.device

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.pipeline.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list, text_input_ids

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        max_sequence_length: int = 512,
    ) -> Dict[str, Union[List[torch.FloatTensor], torch.Tensor]]:
        device = device or self.text_encoder.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds, prompt_ids = self._encode_prompt(
            prompt=prompt,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        results = {
            "prompt_embeds": prompt_embeds,
            "prompt_ids": prompt_ids,
        }

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            assert len(prompt) == len(negative_prompt), "The number of negative prompts must match the number of prompts."
            negative_prompt_embeds, negative_prompt_ids = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                max_sequence_length=max_sequence_length,
            )
            results.update({
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_ids": negative_prompt_ids,
            })

        return results
    
    # ----------------------- Image / Video Encoding & Decoding -----------------------
    def encode_image(
        self,
        images: Union[Image.Image, torch.Tensor, List[torch.Tensor]],
    ):
        """Not needed for Z-Image models."""
        pass

    def encode_video(
        self,
        videos: Union[torch.Tensor, List[torch.Tensor]],
    ):
        """Not needed for Z-Image models."""
        pass

    def decode_latents(
        self,
        latents: torch.Tensor,
        output_type: Literal['pil', 'pt', 'np'] = 'pil',
    ) -> torch.Tensor:
        latents = latents.to(self.pipeline.vae.dtype)
        latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor

        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type=output_type)

        return images
    
    # ======================== Inference ========================

    @torch.no_grad()
    def inference(
        self,
        # Generation parameters
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        height: int = 1024,
        width: int = 1024,
        # Prompt
        prompt: Union[str, List[str]] = None,
        prompt_ids : Optional[torch.Tensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        # Negative prompt
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        # CFG options
        cfg_normalization: bool = False,
        cfg_truncation: Optional[float] = 1.0,
        # Other parameters
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        max_sequence_length: int = 512,
        compute_log_prob: bool = True,
        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = 'all',
    ):
        """Generate images from text prompts using the Z-Image model."""

        # 1. Setup
        device = self.device
        dtype = self.pipeline.transformer.dtype
        do_classifier_free_guidance = guidance_scale > 0.0

        # 2. Encode prompts if not provided
        if prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt=prompt, 
                negative_prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=device
            )
            prompt_ids = encoded['prompt_ids']
            prompt_embeds = encoded['prompt_embeds']
            negative_prompt_ids = encoded['negative_prompt_ids'] if do_classifier_free_guidance else None
            negative_prompt_embeds = encoded['negative_prompt_embeds'] if do_classifier_free_guidance else None
        else:
            prompt_embeds = [pe.to(device) for pe in prompt_embeds]
            negative_prompt_embeds = [npe.to(device) for npe in negative_prompt_embeds]

        batch_size = len(prompt_embeds)
        num_channels_latents = self.pipeline.transformer.in_channels

        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

        timesteps = set_scheduler_timesteps(
            self.scheduler,
            num_inference_steps,
            seq_len=image_seq_len,
            device=device,
        )

        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)
        
        for i, t in enumerate(timesteps):
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kwargs = list(set(['next_latents', 'log_prob', 'noise_pred'] + extra_call_back_kwargs))
            current_compute_log_prob = compute_log_prob and current_noise_level > 0

            output = self.forward(
                t=t,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                cfg_normalization=cfg_normalization,
                cfg_truncation=cfg_truncation,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=current_noise_level,
            )

            latents = output.next_latents.to(dtype)
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob:
                log_prob_collector.collect(output.log_prob, i)

            callback_collector.collect_step(
                step_idx=i,
                output=output,
                keys=extra_call_back_kwargs,
                capturable={'noise_level': current_noise_level},
            )

        # Decode latents to images
        images = self.decode_latents(latents, output_type='pt')

        # Create samples
        extra_call_back_res = callback_collector.get_result()          # (B, len(trajectory_indices), ...)
        callback_index_map = callback_collector.get_index_map()        # (T,) LongTensor
        all_latents = latent_collector.get_result()                    # List[torch.Tensor(B, ...)]
        latent_index_map = latent_collector.get_index_map()            # (T+1,) LongTensor
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
        samples = [
            ZImageSample(
                # Denoising trajectory
                timesteps=timesteps,
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0) if all_latents is not None else None,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if all_log_probs is not None else None,
                latent_index_map=latent_index_map,
                log_prob_index_map=log_prob_index_map,
                # Generated image & metadata
                height=height,
                width=width,
                image=images[b],
                # Encoded prompt
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b] if prompt_embeds is not None else None,
                # Encoded negative prompt
                negative_prompt=negative_prompt[b] if negative_prompt is not None else None,
                negative_prompt_ids=negative_prompt_ids[b] if negative_prompt_ids is not None else None,
                negative_prompt_embeds=negative_prompt_embeds[b] if negative_prompt_embeds is not None else None,
                # Extra kwargs
                extra_kwargs={
                    **{k: v[b] for k, v in extra_call_back_res.items()},
                    'callback_index_map': callback_index_map,
                },
            )
            for b in range(batch_size)
        ]
        
        self.pipeline.maybe_free_model_hooks()

        return samples
    
    # ======================== Forward (Training) ========================
    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: List[torch.FloatTensor],
        # Optional for CFG
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: Optional[float] = 1.0,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = ['noise_pred', 'next_latents', 'next_latents_mean', 'std_dev_t', 'dt', 'log_prob'],
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """
        Core forward pass for T2I generation.

        Args:
            t: Current timestep tensor.
            t_next: Next timestep tensor.
            latents: Current latent representations (B, C, H, W).
            prompt_embeds: List of text prompt embeddings (ragged).
            negative_prompt_embeds: Optional list of negative prompt embeddings.
            guidance_scale: CFG scale factor.
            cfg_normalization: Whether to apply CFG normalization.
            cfg_truncation: CFG truncation threshold.
            next_latents: Optional target latents for log-prob computation.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            noise_level: Current noise level for SDE sampling.

        Returns:
            FlowMatchEulerDiscreteSDESchedulerOutput containing requested outputs.
        """
        # 1. Prepare variables
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        # Convert prompt_embeds to `list of tensors`` on correct device
        prompt_embeds = [pe.to(device) for pe in prompt_embeds]
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = [npe.to(device) for npe in negative_prompt_embeds]
        
        # Z-Image uses reversed timesteps
        timestep = t.expand(batch_size).to(latents.dtype)
        t_reversed = (1000 - timestep) / 1000
        t_norm = t_reversed[0].item()

        # Auto-detect CFG
        do_classifier_free_guidance = (
            negative_prompt_embeds is not None
            and guidance_scale > 0.0
        )

        # 2. Determine if CFG should be applied at this timestep
        if (
            do_classifier_free_guidance
            and cfg_truncation is not None
            and float(cfg_truncation) <= 1
            and t_norm > cfg_truncation
        ):
            current_guidance_scale = 0.0
        else:
            current_guidance_scale = guidance_scale

        apply_cfg = do_classifier_free_guidance and current_guidance_scale > 0

        # 3. Prepare inputs
        if apply_cfg:
            latents_typed = latents.to(self.pipeline.transformer.dtype)
            latent_model_input = latents_typed.repeat(2, 1, 1, 1)
            prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds  # List concatenation
            timestep_model_input = t_reversed.repeat(2)
        else:
            latent_model_input = latents.to(self.pipeline.transformer.dtype)
            prompt_embeds_model_input = prompt_embeds
            timestep_model_input = t_reversed

        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        # 4. Transformer forward pass
        model_out_list = self.transformer(
            latent_model_input_list,
            timestep_model_input,
            prompt_embeds_model_input,
            return_dict=False,
        )[0]

        # 5. Apply CFG
        if apply_cfg:
            pos_out = model_out_list[:batch_size]
            neg_out = model_out_list[batch_size:]
            noise_pred = []
            
            for j in range(batch_size):
                pos = pos_out[j].float()
                neg = neg_out[j].float()
                pred = pos + current_guidance_scale * (pos - neg)
                
                # CFG normalization
                if cfg_normalization and float(cfg_normalization) > 0.0:
                    ori_pos_norm = torch.linalg.vector_norm(pos)
                    new_pos_norm = torch.linalg.vector_norm(pred)
                    max_new_norm = ori_pos_norm * float(cfg_normalization)
                    if new_pos_norm > max_new_norm:
                        pred = pred * (max_new_norm / new_pos_norm)
                
                noise_pred.append(pred)
            
            noise_pred = torch.stack(noise_pred, dim=0)
        else:
            noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)

        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred  # Z-Image specific: negate noise prediction

        # 6. Scheduler step
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
