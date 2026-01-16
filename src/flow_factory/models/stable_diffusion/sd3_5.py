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

# src/flow_factory/models/sd3.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from PIL import Image
import logging

from accelerate import Accelerator

from ...hparams import *
from ..adapter import BaseAdapter, BaseSample
from ...scheduler import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput, set_scheduler_timesteps
from ...utils.base import filter_kwargs


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SD3_5Sample(BaseSample):
    pooled_prompt_embeds : Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds : Optional[torch.Tensor] = None

class SD3_5Adapter(BaseAdapter):
    """Concrete implementation for Stable Diffusion 3 medium."""
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: StableDiffusion3Pipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

    def load_pipeline(self) -> StableDiffusion3Pipeline:
        return StableDiffusion3Pipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False,
        )

    # ============================ Modules & Components ============================
    @property
    def default_target_modules(self) -> List[str]:
        return [
            # Attention modules
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
        ]
    
    @property
    def tokenizer(self) -> Any:
        return self.pipeline.tokenizer_3
    
    # ============================ Encoding & Decoding ============================
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = False,
        max_sequence_length: Optional[int] = 256,
        device: Optional[torch.device] = None,
    ):
        device = device if device is not None else self.device
        (
            prompt_embeds, 
            negative_prompt_embeds, 
            pooled_prompt_embeds, 
            negative_pooled_prompt_embeds
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt
        )
        result = {
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
        }

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # Token ids for downstream bookkeeping (used as `prompt_ids` in samples)
        result['prompt_ids'] = text_inputs.input_ids

        if negative_prompt is not None and do_classifier_free_guidance:
            result["negative_prompt_embeds"] = negative_prompt_embeds
            result["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

            negative_text_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            result['negative_prompt_ids'] = negative_text_inputs.input_ids

        return result

    def encode_image(self, image: Union[Image.Image, torch.Tensor, List[torch.Tensor]], **kwargs):
        """Not needed for SD3 text-to-image models."""
        pass

    def encode_video(self, video: Union[torch.Tensor, List[torch.Tensor]], **kwargs) -> torch.Tensor:
        """Not needed for SD3 text-to-image models."""
        pass

    def decode_latents(
        self,
        latents: torch.Tensor,
        output_type: str = "pil",
        **kwargs
    ) -> torch.Tensor:
        latents = latents.to(self.pipeline.vae.dtype)
        latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor

        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type=output_type)

        return images

    # ============================ Inference ============================
    @torch.no_grad()
    def inference(
        self,
        # Oridinary args
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        do_classifier_free_guidance: bool = False,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        # Encoded Prompt
        prompt_ids : Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # Encoded Negative Prompt
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,

        # Other args
        compute_log_prob: bool = True,
        extra_call_back_kwargs: List[str] = [],
    ) -> List[SD3_5Sample]:
        # 1. Setup
        device = self.device
        dtype = self.transformer.dtype

        # 2. Encode prompt
        if prompt_embeds is None or pooled_prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt,
                negative_prompt,
                do_classifier_free_guidance,
                device=device,
            )
            prompt_embeds = encoded['prompt_embeds']
            pooled_prompt_embeds = encoded['pooled_prompt_embeds']
            prompt_ids = encoded['prompt_ids']
            if do_classifier_free_guidance:
                negative_prompt_embeds = encoded['negative_prompt_embeds']
                negative_prompt_ids = encoded['negative_prompt_ids']
                negative_pooled_prompt_embeds = encoded['negative_pooled_prompt_embeds']
        else:
            if do_classifier_free_guidance:
                if negative_prompt_embeds is None:
                    raise ValueError(
                        "When using CFG with provided `prompt_embeds`, "
                        "you must also provide `negative_prompt_embeds`"
                    )
                negative_prompt_embeds = negative_prompt_embeds.to(device)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)
            else:
                negative_prompt_embeds = None
                negative_pooled_prompt_embeds = None
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)

        batch_size = len(prompt_embeds)
        num_channels_latents = self.pipeline.transformer.config.in_channels
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3. Prepare latent variables
        latents = self.pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
        )
        # latents : torch.Tensor of shape (B, C, H/8, W/8), not packed

        # 5. Prepare noise schedule
        image_seq_len = (
            (latents.shape[2] // self.pipeline.transformer.config.patch_size) * 
            (latents.shape[3] // self.pipeline.transformer.config.patch_size)
        )
        timesteps = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=image_seq_len,
            device=device,
        )

        # 6. Denosing loop
        all_latents = [latents]
        all_log_probs = [] if compute_log_prob else None
        extra_call_back_res = defaultdict(list)

        for i, t in enumerate(timesteps):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # x_t -> x_t-1
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
                # Everything that might be needed for callbacks - must have batch dimension
                capturable = {'noise_pred': noise_pred, 'noise_levels': [current_noise_level] * batch_size}
                for key in extra_call_back_kwargs:
                    if key in capturable and capturable[key] is not None:
                        # First check in capturable dict
                        extra_call_back_res[key].append(capturable[key])
                    elif hasattr(output, key):
                        # Then check in output
                        val = getattr(output, key)
                        if val is not None:
                            extra_call_back_res[key].append(val)

        # 7. Decode latents
        images = self.decode_latents(latents=latents)

        # 8. Create samples
        samples = [
            SD3_5Sample(
                # Denoising trajectory
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_prob else None,
                # Prompt
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b] if prompt_embeds is not None else None,
                pooled_prompt_embeds=pooled_prompt_embeds[b] if pooled_prompt_embeds is not None else None,
                # Negative Prompt
                negative_prompt=negative_prompt[b] if negative_prompt is not None else None,
                negative_prompt_ids=negative_prompt_ids[b] if negative_prompt_ids is not None else None,
                negative_prompt_embeds=negative_prompt_embeds[b] if negative_prompt_embeds is not None else None,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds[b] if negative_pooled_prompt_embeds is not None else None,
                # Image & metadata
                height=height,
                width=width,
                image=images[b],
                # Extra kwargs
                extra_kwargs={
                    'guidance_scale': guidance_scale,
                    'do_classifier_free_guidance': do_classifier_free_guidance,
                    **{k: v[b] for k, v in extra_call_back_res.items()}
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()

        return samples
    
    # ============================ Training Forward ============================
    def forward(
        self,
        samples : List[SD3_5Sample],
        timestep_index : int,
        compute_log_prob : bool = True,
        **kwargs
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """Compute log-probabilities for training."""
        # 1. Prepare inputs
        batch_size = len(samples)
        device = self.device
        guidance_scale = [
            s.extra_kwargs.get('guidance_scale', self.training_args.guidance_scale)
            for s in samples
        ]
        guidance = torch.as_tensor(guidance_scale, device=device, dtype=torch.float32)

        latents = torch.stack([s.all_latents[timestep_index] for s in samples], dim=0).to(device)
        next_latents = torch.stack([s.all_latents[timestep_index + 1] for s in samples], dim=0).to(device)
        timestep = torch.stack([s.timesteps[timestep_index] for s in samples], dim=0).to(device)  
        do_classifier_free_guidance = any(
            s.extra_kwargs.get('do_classifier_free_guidance', False)
            for s in samples
        )

        prompt_embeds = torch.stack([s.prompt_embeds for s in samples], dim=0).to(device)
        pooled_prompt_embeds = torch.stack([s.pooled_prompt_embeds for s in samples], dim=0).to(device)    
        negative_prompt_embeds = torch.stack([s.negative_prompt_embeds for s in samples], dim=0).to(device) if do_classifier_free_guidance else []
        negative_pooled_prompt_embeds = torch.stack([s.negative_pooled_prompt_embeds for s in samples], dim=0).to(device) if do_classifier_free_guidance else []

        # 2. Set scheduler timesteps
        num_inference_steps = len(samples[0].timesteps)
        image_seq_len = (
            (latents.shape[2] // self.pipeline.transformer.config.patch_size) * 
            (latents.shape[3] // self.pipeline.transformer.config.patch_size)
        )
        _ = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=image_seq_len,
            device=device
        )

        # 3. Forward pass
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            latents_input = torch.cat([latents, latents], dim=0)
        else:
            latents_input = latents
        
        # Forward pass
        noise_pred = self.transformer(
            hidden_states=latents_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

        # 4. Compute scheduler step
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
