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

# src/flow_factory/models/stable_diffusion/sd3_5.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple, ClassVar, Literal
from dataclasses import dataclass
from collections import defaultdict

import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from PIL import Image
import logging

from accelerate import Accelerator

from ...hparams import *
from ..abc import BaseAdapter, BaseSample
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SDESchedulerOutput,
    set_scheduler_timesteps
)
from ...utils.trajectory_collector import (
    TrajectoryCollector, 
    TrajectoryIndicesType, 
    create_trajectory_collector,
)
from ...utils.base import filter_kwargs


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SD3_5Sample(BaseSample):
    # Class var
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    # Obj var
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None
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
        max_sequence_length: Optional[int] = 512,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode the prompt(s) into embeddings using the pipeline's text encoder."""
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

    def encode_image(self, images: Union[Image.Image, torch.Tensor, List[torch.Tensor]]):
        """Not needed for SD3 text-to-image models."""
        pass

    def encode_video(self, videos: Union[torch.Tensor, List[torch.Tensor]]):
        """Not needed for SD3 text-to-image models."""
        pass

    def decode_latents(
        self,
        latents: torch.Tensor,
        output_type: Literal['pil', 'pt', 'np'] = "pil",
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
        trajectory_indices: TrajectoryIndicesType = 'all',
    ) -> List[SD3_5Sample]:
        # 1. Setup
        device = self.device
        dtype = self.pipeline.transformer.dtype

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
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                joint_attention_kwargs=joint_attention_kwargs,
                compute_log_prob=compute_log_prob and current_noise_level > 0,
                return_kwargs=return_kwargs,
                noise_level=current_noise_level,
            )

            latents = output.next_latents.to(dtype)
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

        # 7. Decode latents
        images = self.decode_latents(latents=latents, output_type='pt')

        # 8. Create samples
        all_latents = latent_collector.get_result()
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        samples = [
            SD3_5Sample(
                # Denoising trajectory
                timesteps=timesteps,
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0) if all_latents is not None else None,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if all_log_probs is not None else None,
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
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        # Optional for CFG
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        noise_level: Optional[float] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = ['noise_pred', 'next_latents', 'next_latents_mean', 'std_dev_t', 'dt', 'log_prob'],
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """
        Core forward pass for T2I generation.

        Args:
            t: Current timestep tensor.
            t_next: Next timestep tensor.
            latents: Current latent representations (B, C, H, W).
            prompt_embeds: Text prompt embeddings.
            pooled_prompt_embeds: Pooled text prompt embeddings.
            negative_prompt_embeds: Optional negative prompt embeddings (for CFG).
            negative_pooled_prompt_embeds: Optional negative pooled prompt embeddings.
            guidance_scale: CFG scale factor.
            next_latents: Optional target latents for log-prob computation.
            joint_attention_kwargs: Optional kwargs for attention layers.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            noise_level: Current noise level for SDE sampling.

        Returns:
            FlowMatchEulerDiscreteSDESchedulerOutput containing requested outputs.
        """
        # 1. Prepare variables
        batch_size = latents.shape[0]
        timestep = t.expand(batch_size).to(latents.dtype)
        
        # Auto-detect CFG
        do_classifier_free_guidance = (
            negative_prompt_embeds is not None
            and negative_pooled_prompt_embeds is not None
            and guidance_scale > 1.0
        )

        # 2. Prepare inputs for CFG
        if do_classifier_free_guidance:
            prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            latents_input = torch.cat([latents, latents], dim=0)
            timestep_input = timestep.repeat(2)
        else:
            prompt_embeds_input = prompt_embeds
            pooled_prompt_embeds_input = pooled_prompt_embeds
            latents_input = latents
            timestep_input = timestep

        # 3. Transformer forward pass
        noise_pred = self.transformer(
            hidden_states=latents_input,
            timestep=timestep_input,
            encoder_hidden_states=prompt_embeds_input,
            pooled_projections=pooled_prompt_embeds_input,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

        # 4. Apply CFG
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

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