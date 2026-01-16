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
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
from collections import defaultdict
import logging

import torch
from accelerate import Accelerator
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

from ..adapter import BaseAdapter
from ..samples import T2ISample
from ...hparams import *
from ...scheduler import SDESchedulerOutput, set_scheduler_timesteps, FlowMatchEulerDiscreteSDEScheduler
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class ZImageSample(T2ISample):
    pass

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
        return [
            "attention.to_k", "attention.to_q", "attention.to_v", "attention.to_out.0",
            "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
        ]
    
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
    ) -> Tuple[List[torch.FloatTensor], torch.LongTensor]:
        device = device or self.device

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
    ) -> Dict[str, Union[List[torch.FloatTensor], torch.LongTensor]]:
        device = device or self.device
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
    
    def encode_image(
        self,
        images: Union[Image.Image, torch.Tensor, List[torch.Tensor]],
        **kwargs   
    ):
        """Not needed for Z-Image models."""
        pass

    def encode_video(
        self,
        videos: Union[torch.Tensor, List[torch.Tensor]],
        **kwargs
    ):
        """Not needed for Z-Image models."""
        pass

    def decode_latents(
        self,
        latents: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        latents = latents.to(self.pipeline.vae.dtype)
        latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor

        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type="pil")

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
        **kwargs
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

        all_latents = [latents]
        all_log_probs = [] if compute_log_prob else None
        extra_call_back_res = defaultdict(list)
        
        for i, t in enumerate(timesteps):
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            timestep = t.expand(batch_size).to(latents.dtype)
            timestep = (1000 - timestep) / 1000 # Z-Image uses reversed timesteps?
            # Normalized time for time-aware config (0 at start, 1 at end)
            t_norm = timestep[0].item()

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

            if apply_cfg:
                latents_typed = latents.to(self.pipeline.transformer.dtype)
                latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds # List concatenation
                timestep_model_input = timestep.repeat(2)
            else:
                latent_model_input = latents.to(self.pipeline.transformer.dtype)
                prompt_embeds_model_input = prompt_embeds
                timestep_model_input = timestep

            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))

            model_out_list = self.transformer(
                latent_model_input_list,
                timestep_model_input,
                prompt_embeds_model_input,
                return_dict=False,
            )[0]

            if apply_cfg:
                # Perform CFG
                pos_out = model_out_list[:batch_size]
                neg_out = model_out_list[batch_size:]

                noise_pred = []
                for j in range(batch_size):
                    pos = pos_out[j].float()
                    neg = neg_out[j].float()

                    pred = pos + current_guidance_scale * (pos - neg)

                    # Renormalization
                    if cfg_normalization and float(cfg_normalization) > 0.0:
                        ori_pos_norm = torch.linalg.vector_norm(pos)
                        new_pos_norm = torch.linalg.vector_norm(pred)
                        max_new_norm = ori_pos_norm * float(cfg_normalization)
                        if new_pos_norm > max_new_norm:
                            pred = pred * (max_new_norm / new_pos_norm)

                    noise_pred.append(pred)

                noise_pred = torch.stack(noise_pred, dim=0)
            else:
                noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred

            # compute the previous noisy sample x_t -> x_t-1
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

        # Decode latents to images
        images = self.decode_latents(latents)

        # Create samples

        # Transpose `extra_call_back_res` lists to have batch dimension first
        # (T, B, ...) -> (B, T, ...)
        extra_call_back_res = {
            k: torch.stack(v, dim=1)
            if isinstance(v[0], torch.Tensor) else v
            for k, v in extra_call_back_res.items()
        }
        samples = [
            ZImageSample(
                # Denoising trajectory
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_prob else None,

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
                    'guidance_scale': guidance_scale,
                    'cfg_truncation': cfg_truncation,
                    'cfg_normalization': cfg_normalization,
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
        samples : List[ZImageSample],
        timestep_index : int,
        compute_log_prob : bool = True,
        **kwargs
    ) -> SDESchedulerOutput:
        # 1. Extract data from samples
        batch_size = len(samples)
        device = self.device
        guidance_scale = [s.extra_kwargs.get('guidance_scale', self.training_args.guidance_scale) for s in samples]
        do_classifier_free_guidance = guidance_scale[0] > 0.0
        cfg_truncation = samples[0].extra_kwargs.get('cfg_truncation', 1.0)
        cfg_normalization = samples[0].extra_kwargs.get('cfg_normalization', False)

        latents = torch.stack([s.all_latents[timestep_index] for s in samples], dim=0).to(device)
        next_latents = torch.stack([s.all_latents[timestep_index + 1] for s in samples], dim=0).to(device)
        timestep = torch.stack([s.timesteps[timestep_index] for s in samples], dim=0).to(device)
        num_inference_steps = len(samples[0].timesteps)
        t = (1000 - timestep) / 1000 # Z-Image uses reversed timesteps
        t_norm = t[0].item()

        prompt_embeds = [s.prompt_embeds.to(device) for s in samples]
        negative_prompt_embeds = [s.negative_prompt_embeds.to(device) for s in samples] if do_classifier_free_guidance else []

        # 2. Set scheduler timesteps        
        _ = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device
        )

        # 3. Forward pass
        if (
            do_classifier_free_guidance
            and cfg_truncation
            and float(cfg_truncation) <= 1
            and t_norm > cfg_truncation
        ):
            current_guidance_scale = 0.0
        else:
            current_guidance_scale = guidance_scale[0]

        apply_cfg = do_classifier_free_guidance and current_guidance_scale > 0

        if apply_cfg:
            latents_typed = latents.to(self.pipeline.transformer.dtype)
            latent_model_input = latents_typed.repeat(2, 1, 1, 1)
            prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds # List concatenation
            timestep_model_input = t.repeat(2)
        else:
            latent_model_input = latents.to(self.pipeline.transformer.dtype)
            prompt_embeds_model_input = prompt_embeds
            timestep_model_input = t

        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        model_out_list = self.transformer(
            latent_model_input_list,
            timestep_model_input,
            prompt_embeds_model_input,
            return_dict=False,
        )[0]

        if apply_cfg:
            pos_out = model_out_list[:batch_size]
            neg_out = model_out_list[batch_size:]
            noise_pred = []
            
            for j in range(batch_size):
                pos = pos_out[j].float()
                neg = neg_out[j].float()
                pred = pos + current_guidance_scale * (pos - neg)
                
                if cfg_normalization and float(cfg_normalization) > 0.0:
                    ori_pos_norm = torch.linalg.vector_norm(pos)
                    new_pos_norm = torch.linalg.vector_norm(pred)
                    max_new_norm = ori_pos_norm * float(cfg_normalization)
                    if new_pos_norm > max_new_norm:
                        pred = pred * (max_new_norm / new_pos_norm)
                
                noise_pred.append(pred)
            
            noise_pred = torch.stack(noise_pred, dim=0)
        else:
            noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred

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