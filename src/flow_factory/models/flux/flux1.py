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

# src/flow_factory/models/flux/flux1.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import logging
from collections import defaultdict

from accelerate import Accelerator
import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

from ..samples import T2ISample
from ..adapter import BaseAdapter
from ...hparams import *
from ...scheduler import FlowMatchEulerDiscreteSDEScheduler, SDESchedulerOutput, set_scheduler_timesteps
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class Flux1Sample(T2ISample):
    """Output class for Flux Adapter models."""
    pooled_prompt_embeds : Optional[torch.FloatTensor] = None
    image_ids : Optional[torch.Tensor] = None


class Flux1Adapter(BaseAdapter):
    """Concrete implementation for Flow Matching models (FLUX.1)."""
    
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: FluxPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler
    
    def load_pipeline(self) -> FluxPipeline:
        return FluxPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )

    @property
    def default_target_modules(self) -> List[str]:
        return [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2",
        ]

    # ========================== Tokenizer & Text Encoder ==========================
    @property
    def tokenizer(self) -> Any:
        """Use T5 for longer context length."""
        return self.pipeline.tokenizer_2

    @property
    def text_encoder(self) -> Any:
        """Use T5 text encoder."""
        return self.pipeline.text_encoder_2

    # ======================== Encoding & Decoding ========================
    
    def encode_prompt(self, prompt: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Encode text prompts using the pipeline's text encoder."""

        execution_device = self.pipeline.text_encoder.device
        
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=execution_device,
        )
        
        prompt_ids = self.pipeline.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(execution_device)
                
        return {
            'prompt_ids': prompt_ids,
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
        }
    
    def encode_image(self, images: Union[Image.Image, List[Optional[Image.Image]]], **kwargs) -> None:
        """
        Encode input images into latent representations using the VAE encoder.
         Args:
            images:
                - Single Image.Image
                - List[Image.Image]: list of images
        """
        pass

    def encode_video(self, videos: Union[torch.Tensor, List[torch.Tensor]], **kwargs) -> None:
        """Not needed for FLUX text-to-image models."""
        pass

    def decode_latents(self, latents: torch.Tensor, height: int, width: int, **kwargs) -> List[Image.Image]:
        """Decode latents to images using VAE."""
        
        latents = self.pipeline._unpack_latents(latents, height, width, self.pipeline.vae_scale_factor)
        latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
        latents = latents.to(dtype=self.pipeline.vae.dtype)
        
        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type="pil")
        
        return images

    # ======================== Inference ========================
    
    @torch.no_grad()
    def inference(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_ids : Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        generator: Optional[torch.Generator] = None,
        compute_log_prob: bool = True,
        extra_call_back_kwargs: List[str] = [],
    ) -> List[Flux1Sample]:
        """Execute generation and return FluxSample objects."""
        
        # 1. Setup
        device = self.device
        
        # 2. Encode prompts if not provided
        if prompt_embeds is None:
            encoded = self.encode_prompt(prompt)
            prompt_embeds = encoded['prompt_embeds']
            pooled_prompt_embeds = encoded['pooled_prompt_embeds']
            prompt_ids = encoded['prompt_ids']
        else:
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)

        batch_size = len(prompt_embeds)
        dtype = prompt_embeds.dtype

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=device, dtype=dtype
        )
        
        # 3. Prepare latents
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, latent_image_ids = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        
        # 4. Set timesteps with scheduler
        timesteps = set_scheduler_timesteps(
            scheduler=self.pipeline.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device,
        )

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        
        # 5. Denoising loop
        all_latents = [latents]
        all_log_probs = [] if compute_log_prob else None
        extra_call_back_res = defaultdict(list)
        
        for i, t in enumerate(timesteps):
            timestep = t.expand(batch_size).to(latents.dtype)
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            
            # Predict noise
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance.expand(batch_size),
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            
            # Scheduler step
            output = self.scheduler.step(
                noise_pred=noise_pred,
                timestep=t,
                latents=latents,
                compute_log_prob=compute_log_prob and current_noise_level > 0,
            )
            
            latents = output.next_latents.to(dtype)
            all_latents.append(latents)
            
            if compute_log_prob:
                all_log_probs.append(output.log_prob)

            if extra_call_back_kwargs:
                capturable = {'noise_pred': noise_pred, 'noise_levels': current_noise_level}
                for key in extra_call_back_kwargs:
                    if key in capturable and capturable[key] is not None:
                        # First check in capturable dict
                        extra_call_back_res[key].append(capturable[key])
                    elif hasattr(output, key):
                        # Then check in output
                        val = getattr(output, key)
                        if val is not None:
                            extra_call_back_res[key].append(val)

        
        # 6. Decode images
        images = self.decode_latents(latents, height, width)
        
        # 7. Create samples
        # Transpose `extra_call_back_res` tensors to have batch dimension first
        # (T, B, ...) -> (B, T, ...)
        extra_call_back_res = {
            k: torch.stack(v, dim=1)
            if isinstance(v[0], torch.Tensor) else v
            for k, v in extra_call_back_res.items()
        }
        samples = [
            Flux1Sample(
                # Denoising trajectory
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_prob else None,

                # Prompt
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b],
                pooled_prompt_embeds=pooled_prompt_embeds[b],

                # Image & metadata
                height=height,
                width=width,
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                image=images[b],
                image_ids=latent_image_ids,

                # Extra kwargs
                extra_kwargs={
                    'guidance_scale': guidance_scale,
                    **{k: v[b] for k, v in extra_call_back_res.items()}
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()
        
        return samples

    # ======================== Forward (Training) ========================
    
    def forward(
        self,
        samples: List[Flux1Sample],
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> SDESchedulerOutput:
        """Compute log-probabilities for training."""
        
        batch_size = len(samples)
        device = self.device
        guidance_scale = [
            s.extra_kwargs.get('guidance_scale', self.training_args.guidance_scale)
            for s in samples
        ]
        guidance = torch.as_tensor(guidance_scale, device=device, dtype=torch.float32)
        
        # 1. Extract data from samples
        latents = torch.stack([s.all_latents[timestep_index] for s in samples], dim=0).to(device)
        next_latents = torch.stack([s.all_latents[timestep_index + 1] for s in samples], dim=0).to(device)
        timestep = torch.stack([s.timesteps[timestep_index] for s in samples], dim=0).to(device)
        num_inference_steps = len(samples[0].timesteps)
        t = timestep[0]
        
        prompt_embeds = torch.stack([s.prompt_embeds for s in samples], dim=0).to(device)
        pooled_prompt_embeds = torch.stack([s.pooled_prompt_embeds for s in samples], dim=0).to(device)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device)
        latent_image_ids = samples[0].image_ids.to(device) # No batch dimension needed
        
        # 2. Set scheduler timesteps
        _ = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device
        )
    
        
        # 3. Forward pass
        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance.expand(batch_size),
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        
        # 4. Compute log prob with given next_latents
        step_kwargs = filter_kwargs(self.scheduler.step, **kwargs)
        output = self.scheduler.step(
            noise_pred=noise_pred,
            timestep=timestep,
            latents=latents,
            next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            **step_kwargs,
        )
        
        return output
