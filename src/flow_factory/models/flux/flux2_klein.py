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

# src/flow_factory/models/flux/flux2_klein.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
from PIL import Image
from collections import defaultdict
import numpy as np
from accelerate import Accelerator
import torch
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline, compute_empirical_mu
import logging

from ..abc import BaseAdapter
from ..samples import I2ISample
from ...hparams import *
from ...scheduler import FlowMatchEulerDiscreteSDEScheduler, SDESchedulerOutput, set_scheduler_timesteps
from ...utils.base import (
    filter_kwargs,
    is_pil_image_batch_list,
    is_pil_image_list,
    tensor_to_pil_image,
    tensor_list_to_pil_image,
    numpy_list_to_pil_image,
    numpy_to_pil_image,
    pil_image_to_tensor,
    standardize_image_batch,
)
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class Flux2KleinSample(I2ISample):
    """Output class for Flux2Adapter models."""
    latent_ids : Optional[torch.Tensor] = None
    text_ids : Optional[torch.Tensor] = None
    negative_text_ids : Optional[torch.Tensor] = None
    image_latents : Optional[torch.Tensor] = None
    image_latent_ids : Optional[torch.Tensor] = None


CONDITION_IMAGE_SIZE = 1024 * 1024

Flux2KleinImageInput = Union[
    Image.Image,
    np.ndarray,
    torch.Tensor,
    List[Image.Image],
    List[np.ndarray],
    List[torch.Tensor]
]

class Flux2Adapter(BaseAdapter):
    """Concrete implementation for Flow Matching models (FLUX.2)."""
    
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: Flux2KleinPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler
        
        self._has_warned_inference_fallback = False
        self._has_warned_forward_fallback = False
    
    def load_pipeline(self) -> Flux2KleinPipeline:
        return Flux2KleinPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default Trainable target modules for FLUX.2 Klein model."""
        return [
            # --- Double Stream Block ---
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.linear_in", "ff.linear_out", 
            "ff_context.linear_in", "ff_context.linear_out",
            
            # --- Single Stream Block ---
            "attn.to_qkv_mlp_proj", 
            "attn.to_out.0",
        ]
    
    # ======================== Encoding & Decoding ========================
    @staticmethod
    def _get_qwen3_prompt_embeds(
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        hidden_states_layers: Tuple[int, ...] = (9, 18, 27),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )

            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        # Forward pass through the model
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return input_ids, prompt_embeds
    
    # ======================== Prompt Encoding ========================
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = False,
        device: Optional[torch.device] = None,        
        max_sequence_length: int = 512,
        hidden_states_layers: Tuple[int, ...] = (9, 18, 27),
    ) -> Dict[str, torch.Tensor]:
        """Preprocess the prompt(s) into embeddings using the Qwen3 text encoder."""
        device = self.pipeline.text_encoder.device if device is None else device
        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt_ids, prompt_embeds = self._get_qwen3_prompt_embeds(
            text_encoder=self.pipeline.text_encoder,
            tokenizer=self.pipeline.tokenizer,
            prompt=prompt,
            dtype=self.pipeline.text_encoder.dtype,
            device=device,
            max_sequence_length=max_sequence_length,
            hidden_states_layers=hidden_states_layers,
        )
        text_ids = self.pipeline._prepare_text_ids(prompt_embeds)
        results = {
            "prompt_ids": prompt_ids,
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
        }
        if do_classifier_free_guidance:
            negative_prompt = "" if negative_prompt is None else negative_prompt
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt = negative_prompt * (len(prompt) // len(negative_prompt)) # Expand to match batch size
            assert len(negative_prompt) == len(prompt), "The number of negative prompts must match the number of prompts."

            negative_prompt_ids, negative_prompt_embeds = self._get_qwen3_prompt_embeds(
                text_encoder=self.pipeline.text_encoder,
                tokenizer=self.pipeline.tokenizer,
                prompt=negative_prompt,
                dtype=self.pipeline.text_encoder.dtype,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=hidden_states_layers,
            )
            negative_text_ids = self.pipeline._prepare_text_ids(negative_prompt_embeds)
            results.update({
                "negative_prompt_ids": negative_prompt_ids,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_text_ids": negative_text_ids,
            })

        return results
    
    # ======================== Image Encoding ========================
    def encode_image(
        self,
        images: Union[Flux2KleinImageInput, List[Flux2KleinImageInput]],
        condition_image_size : Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Union[List[List[torch.Tensor]], torch.Tensor]]:
        """Preprocess the image(s) into latents using the FLUX.2 Klein VAE encoder."""
        device = self.pipeline.vae.device if device is None else device
        dtype = self.pipeline.vae.dtype if dtype is None else dtype
        # A simple check to see if input is a batch of condition image lists
        is_nested_batch = (
            (isinstance(images, list) and len(images) > 0 and isinstance(images[0], list))
            or (isinstance(images, torch.Tensor) and images.ndim == 5)
            or (isinstance(images, np.ndarray) and images.ndim == 5)
        )
        if not is_nested_batch:
            images = [images] # Wrap into a batch

        images = [self._standardize_image_input(imgs, output_type='pil') for imgs in images]

        condition_image_tensors : List[List[torch.Tensor]] = [
            self._resize_condition_images(
                condition_images=imgs,
                condition_image_size=condition_image_size,
            ) for imgs in images
        ]
        image_latents_list = []
        image_latent_ids_list = []
        for cond_img_tensors in condition_image_tensors:
            image_latents, image_latent_ids =  self.pipeline.prepare_image_latents(
                images=cond_img_tensors,
                batch_size=1,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            image_latents_list.append(image_latents)
            image_latent_ids_list.append(image_latent_ids)
        
        condition_image_tensors : List[List[torch.Tensor]] = [
            [
                self.pipeline.image_processor.postprocess(img, output_type='pt')[0]
                for img in cond_img_tensors
            ]
            for cond_img_tensors in condition_image_tensors
        ]

        return {
            "condition_images": condition_image_tensors, # List[List[torch.Tensor (3, H, W)]]
            "image_latents": image_latents_list, # List[torch.Tensor (1, seq_len, C)]
            "image_latent_ids": image_latent_ids_list, # List[torch.Tensor (1, seq_len)]
        }

    def _standardize_image_input(
        self,
        images: Flux2KleinImageInput,
        output_type: Literal['pil', 'pt', 'np'] = 'pil',
    ):
        """
        Standardize image input to desired output type.
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        return standardize_image_batch(
            images,
            output_type=output_type,
        )
    
    def _resize_condition_images(
        self,
        condition_images: Union[Image.Image, List[Image.Image]],
        condition_image_size : Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
    ) -> List[torch.Tensor]:
        """Resize condition images for Flux.2 model."""
        if isinstance(condition_images, Image.Image):
            condition_images = [condition_images]

        for img in condition_images:
            self.pipeline.image_processor.check_image_input(img)

        if isinstance(condition_image_size, int):
            condition_image_size = (condition_image_size, condition_image_size)

        max_area = condition_image_size[0] * condition_image_size[1]

        condition_image_tensors = []
        for img in condition_images:
            image_width, image_height = img.size
            if image_width * image_height > max_area:
                img = self.pipeline.image_processor._resize_to_target_area(img, max_area)
                image_width, image_height = img.size

            multiple_of = self.pipeline.vae_scale_factor * 2
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of
            img = self.pipeline.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
            condition_image_tensors.append(img)

        return condition_image_tensors
    
    def decode_latents(self, latents: torch.Tensor, latent_ids, output_type: Literal['pil', 'pt', 'np'] = 'pil') -> Union[List[Image.Image], torch.Tensor, np.ndarray]:
        latents = self.pipeline._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.pipeline.vae.bn.running_var.view(1, -1, 1, 1) + self.pipeline.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self.pipeline._unpatchify_latents(latents)

        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type=output_type)

        return images
    
    # ======================== Inference ========================
    # Since Flux.2 does not support ragged batches of condition images, we implement a single-sample inference method.
    @torch.no_grad()
    def _inference(
        self,
        # Ordinary arguments
        images: Optional[Flux2KleinImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        do_classifier_free_guidance: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # Prompt encoding arguments
        prompt_ids: Optional[torch.LongTensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        # Negative prompt encoding arguments
        negative_prompt_ids: Optional[torch.LongTensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_text_ids: Optional[torch.Tensor] = None,
        # Image encoding arguments
        condition_images: Optional[Union[Flux2KleinImageInput, List[Flux2KleinImageInput]]] = None,
        image_latents: Optional[torch.Tensor] = None,
        image_latent_ids: Optional[torch.Tensor] = None,
        # Other arguments
        condition_image_size : Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        hidden_states_layers: Tuple[int, ...] = (9, 18, 27),
        compute_log_prob: bool = False,
        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
    ) -> List[Flux2KleinSample]:
        
        device = self.device
        dtype = self.pipeline.transformer.dtype

        # 1. Encode prompt        
        if isinstance(prompt, str):
            prompt = [prompt]

        if (prompt_embeds is None or prompt_ids is None or text_ids is None) or (
            do_classifier_free_guidance and (negative_prompt_embeds is None or negative_prompt_ids is None or negative_text_ids is None)
        ):
            prompt_encoding = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=hidden_states_layers,
            )
            prompt_ids = prompt_encoding["prompt_ids"]
            prompt_embeds = prompt_encoding["prompt_embeds"]
            text_ids = prompt_encoding["text_ids"]
            if do_classifier_free_guidance:
                negative_prompt_ids = prompt_encoding["negative_prompt_ids"]
                negative_prompt_embeds = prompt_encoding["negative_prompt_embeds"]
                negative_text_ids = prompt_encoding["negative_text_ids"]
        
        batch_size = prompt_embeds.shape[0]

        # 2. Encode image
        if images is not None and (condition_images is None and image_latents is None and image_latent_ids is None):
            image_encoding = self.encode_image(
                images=images,
                condition_image_size=condition_image_size,
                device=device,
                dtype=dtype,
                generator=generator if isinstance(generator, torch.Generator) else None,
            )
            condition_images = image_encoding["condition_images"][0] # List[torch.Tensor (3, H, W)]
            image_latents = image_encoding["image_latents"][0] # torch.Tensor (1, seq_len, C)
            image_latent_ids = image_encoding["image_latent_ids"][0] # torch.Tensor (1, seq_len)

        # 3. Prepare initial latents
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 4. Set timesteps
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
        timesteps = set_scheduler_timesteps(
            scheduler=self.pipeline.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            mu=mu,
        )

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        # 5. Denoising loop
        all_latents = [latents]
        all_log_probs = [] if compute_log_prob else None
        extra_call_back_res = defaultdict(list)

        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)

            latent_model_input = latents.to(torch.float32)
            latent_image_ids = latent_ids

            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1).to(torch.float32)
                latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

            with self.pipeline.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,  # (B, image_seq_len, C)
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,  # B, text_seq_len, 4
                    img_ids=latent_image_ids,  # B, image_seq_len, 4
                    joint_attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

            noise_pred = noise_pred[:, : latents.size(1) :]
            
            if do_classifier_free_guidance and guidance_scale > 1.0:
                with self.pipeline.transformer.cache_context("uncond"):
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

            # Compute the previous noisy sample x_t -> x_t-1
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

        # 6. Decode latents to images
        decoded_images = self.decode_latents(latents, latent_ids)

        # 7. Prepare samples

        # Transpose `extra_call_back_res` tensors to have batch dimension first
        # (T, B, ...) -> (B, T, ...)
        extra_call_back_res = {
            k: torch.stack(v, dim=1)
            if isinstance(v[0], torch.Tensor) else v
            for k, v in extra_call_back_res.items()
        }

        samples = [
            Flux2KleinSample(
                # Denoising trajectory
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_prob else None,

                # Generated image & metadata
                height=height,
                width=width,
                image=decoded_images[b],
                latent_ids=latent_ids[b],

                # Prompt & condition info
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b],
                prompt_embeds=prompt_embeds[b],
                text_ids=text_ids[b],

                # Negative prompt info
                negative_prompt=negative_prompt[b],
                negative_prompt_ids=negative_prompt_ids[b] if negative_prompt_ids is not None else None,
                negative_prompt_embeds=negative_prompt_embeds[b] if negative_prompt_embeds is not None else None,
                negative_text_ids=negative_text_ids[b] if negative_text_ids is not None else None,

                # Condition images & latents
                condition_images=condition_images[b] if condition_images is not None else None,
                image_latents=image_latents[b] if image_latents is not None else None,
                image_latent_ids=image_latent_ids[b] if image_latent_ids is not None else None,
                extra_kwargs={
                    'do_classifier_free_guidance': do_classifier_free_guidance,
                    'guidance_scale': guidance_scale,
                    **{k: v[b] for k, v in extra_call_back_res.items()}
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()
        
        return samples
    
    # Bacth inference
    @torch.no_grad()
    def inference(
        self,
        # Ordinary arguments
        images: Optional[Union[Flux2KleinImageInput, List[Flux2KleinImageInput]]] = None,
        prompt: Optional[List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        hidden_states_layers: Tuple[int, ...] = (9, 18, 27),
        # Encoded prompt
        prompt_ids: Optional[torch.LongTensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        # Encoded negative prompt
        negative_prompt_ids: Optional[torch.LongTensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_text_ids: Optional[torch.Tensor] = None,
        # Encoded images
        condition_images: Optional[Union[Flux2KleinImageInput, List[Flux2KleinImageInput]]] = None,
        image_latents: Optional[Union[torch.Tensor, List[Union[None, torch.Tensor]]]] = None,
        image_latent_ids: Optional[Union[torch.Tensor, List[Union[None, torch.Tensor]]]] = None,
        # Other arguments
        compute_log_prob: bool = False,
        extra_call_back_kwargs: List[str] = []
    ) -> List[Flux2KleinSample]:
        if isinstance(prompt, str):
            prompt = [prompt]
        
        # Check for ragged inputs that require per-sample processing
        is_ragged_images = (
            ( isinstance(images, list) and len(images) > 0 and isinstance(images[0], list) ) # List[List[Image]]
            or
            ( isinstance(images, list) and len(images) > 0 and isinstance(images[0], torch.Tensor) and images[0].ndim == 4 ) # List[torch.Tensor : ndim=4]
            or
            ( isinstance(images, torch.Tensor) and images.ndim == 5 ) # torch.Tensor : ndim=5
        )
        is_ragged_image_latents = (
            (
                isinstance(image_latents, list) and len(image_latents) > 0
                and isinstance(image_latents[0], torch.Tensor) and image_latents[0].ndim == 3
            ) # List[torch.Tensor : ndim=3]
            or
            (
                isinstance(image_latents, torch.Tensor) and image_latents.ndim == 4
            ) # torch.Tensor : ndim=4
        )
        if not (is_ragged_images or is_ragged_image_latents):
            # T2I or Shared condition images across the batch
            return self._inference(
                # Ordinary args
                images=images,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                # Prompt encoding args
                prompt_ids=prompt_ids,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,                
                # Negative prompt encoding args
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
                # Image encoding args
                condition_images=condition_images,
                image_latents=image_latents,
                image_latent_ids=image_latent_ids,
                # Other args
                attention_kwargs=attention_kwargs,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=hidden_states_layers,
                compute_log_prob=compute_log_prob,
                extra_call_back_kwargs=extra_call_back_kwargs,
            )
        
        # Ragged case: per-sample fallback
        if not self._has_warned_inference_fallback:
            logger.warning(
                "FLUX.2 does not support batch inference with varying condition images per sample. "
                "Falling back to single-sample inference. This warning will only appear once."
            )
            self._has_warned_inference_fallback = True
        # Process each sample individually by calling _inference
        batch_size = len(images) if is_ragged_images else len(image_latents)

        samples = []
        for idx in range(batch_size):
            sample = self._inference(
                # Ordinary args
                images=images[idx] if is_ragged_images else images,
                prompt=prompt[idx] if prompt is not None else None,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator[idx] if isinstance(generator, list) else generator,
                # Prompt encoding args
                prompt_ids=prompt_ids[idx:idx+1] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[idx:idx+1] if prompt_embeds is not None else None,
                text_ids=text_ids[idx:idx+1] if text_ids is not None else None,
                # Negative prompt encoding args
                negative_prompt_ids=negative_prompt_ids[idx:idx+1] if negative_prompt_ids is not None else None,
                negative_prompt_embeds=negative_prompt_embeds[idx:idx+1] if negative_prompt_embeds is not None else None,
                negative_text_ids=negative_text_ids[idx:idx+1] if negative_text_ids is not None else None,
                # Image encoding args
                condition_images=condition_images[idx] if is_ragged_images else condition_images,
                image_latents=image_latents[idx] if is_ragged_image_latents else image_latents,
                image_latent_ids=image_latent_ids[idx] if is_ragged_image_latents else image_latent_ids,
                # Other args
                attention_kwargs=attention_kwargs,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=hidden_states_layers,
                compute_log_prob=compute_log_prob,
                extra_call_back_kwargs=extra_call_back_kwargs,
            )
            samples.extend(sample)

        return samples
    
    # ======================== Forward ========================
    def forward(
        self,
        samples: List[Flux2KleinSample],
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> SDESchedulerOutput:
        """Compute log-probabilities for training."""
        # Determine T2I / I2I
        is_i2i = any(
            s.image_latents is not None
            for s in samples
        )

        if is_i2i:
            if not self._has_warned_forward_fallback:
                logger.warning(
                    "Flux.2: Batched I2I training unsupported. Falling back to single-sample forward (warning shown once)."
                )
                self._has_warned_forward_fallback = True
            # Fallback to single-sample forward
            outputs = []
            for s in samples:
                out = self._i2i_forward(
                    sample=s,
                    timestep_index=timestep_index,
                    compute_log_prob=compute_log_prob,
                    **kwargs,
                )
                outputs.append(out)

            outputs = [o.to_dict() for o in outputs]
            # Concatenate outputs
            output = SDESchedulerOutput.from_dict({
                k: torch.cat([o[k] for o in outputs], dim=0) if outputs[0][k] is not None else None
                for k in outputs[0].keys()
            })

        else:
            # T2I, can be batched
            output = self._t2i_forward(
                samples=samples,
                timestep_index=timestep_index,
                compute_log_prob=compute_log_prob,
                **kwargs,
            )

        return output

    def _t2i_forward(
        self,
        samples: List[Flux2KleinSample],
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> SDESchedulerOutput:
        if not isinstance(samples, list):
            samples = [samples]

        batch_size = len(samples)
        device = self.device
        do_classifier_free_guidance = any(
            s.extra_kwargs.get('do_classifier_free_guidance', False)
            for s in samples
        )
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
        latent_ids = torch.stack([s.latent_ids for s in samples], dim=0).to(device)
        text_ids = torch.stack([s.text_ids for s in samples], dim=0).to(device)
        image_latents =  None # Hard code for T2I
        image_latent_ids =  None
        attention_kwargs = samples[0].extra_kwargs.get('attention_kwargs', None)

        # Catenate condition latents if given
        latent_model_input = latents.to(torch.float32)
        latent_image_ids = latent_ids
                
        # 2. Set scheduler timesteps
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
        timesteps = set_scheduler_timesteps(
            scheduler=self.pipeline.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            mu=mu,
        )

        # 3. Predict noise
        noise_pred = self.transformer(
            hidden_states=latent_model_input,  # (B, image_seq_len, C)
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,  # B, text_seq_len, 4
            img_ids=latent_image_ids,  # B, image_seq_len, 4
            joint_attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

        noise_pred = noise_pred[:, : latents.size(1) :]

        if do_classifier_free_guidance and guidance_scale[0] > 1.0:
            neg_prompt_embeds = torch.stack([s.negative_prompt_embeds for s in samples], dim=0).to(device)
            neg_text_ids = torch.stack([s.negative_text_ids for s in samples], dim=0).to(device)

            neg_noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=neg_prompt_embeds,
                txt_ids=neg_text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
            noise_pred = neg_noise_pred + guidance * (noise_pred - neg_noise_pred)

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
    
    def _i2i_forward(
        self,
        sample: Flux2KleinSample,
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> SDESchedulerOutput:
        device = self.device
        batch_size = 1 # Single-sample only
        do_classifier_free_guidance = sample.extra_kwargs.get('do_classifier_free_guidance', False)
        guidance_scale = sample.extra_kwargs.get('guidance_scale', self.training_args.guidance_scale)
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        # 1. Extract data from sample
        latents = sample.all_latents[timestep_index].unsqueeze(0).to(device)
        next_latents = sample.all_latents[timestep_index + 1].unsqueeze(0).to(device)
        timestep = sample.timesteps[timestep_index].unsqueeze(0).to(device)
        num_inference_steps = len(sample.timesteps)
        t = timestep[0]
        prompt_embeds = sample.prompt_embeds.unsqueeze(0).to(device)
        latent_ids = sample.latent_ids.unsqueeze(0).to(device)
        text_ids = sample.text_ids.unsqueeze(0).to(device)
        image_latents = sample.image_latents.unsqueeze(0).to(device) if sample.image_latents is not None else None
        image_latent_ids = sample.image_latent_ids.unsqueeze(0).to(device) if sample.image_latent_ids is not None else None
        attention_kwargs = sample.extra_kwargs.get('attention_kwargs', None)

        latent_model_input = latents.to(torch.float32)
        latent_image_ids = latent_ids

        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1).to(torch.float32)
            latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

        # 2. Set scheduler timesteps
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
        timesteps = set_scheduler_timesteps(
            scheduler=self.pipeline.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            mu=mu,
        )
        # 3. Predict noise
        noise_pred = self.transformer(
            hidden_states=latent_model_input,  # (B, image_seq_len, C)
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,  # B, text_seq_len, 4
            img_ids=latent_image_ids,  # B, image_seq_len, 4
            joint_attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, : latents.size(1) :]

        if do_classifier_free_guidance and guidance_scale > 1.0:
            neg_prompt_embeds = sample.negative_prompt_embeds.unsqueeze(0).to(device)
            neg_text_ids = sample.negative_text_ids.unsqueeze(0).to(device)

            neg_noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=neg_prompt_embeds,
                txt_ids=neg_text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
            noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

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