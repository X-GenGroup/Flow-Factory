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

# src/flow_factory/models/flux/flux2.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
from PIL import Image
from collections import defaultdict
import numpy as np
from accelerate import Accelerator
import torch
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline, format_input, compute_empirical_mu
from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE, SYSTEM_MESSAGE_UPSAMPLING_T2I, SYSTEM_MESSAGE_UPSAMPLING_I2I
import logging

from ..adapter import BaseAdapter
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
class Flux2Sample(I2ISample):
    """Output class for Flux2Adapter models."""
    latent_ids : Optional[torch.Tensor] = None
    text_ids : Optional[torch.Tensor] = None
    image_latents : Optional[torch.Tensor] = None
    image_latent_ids : Optional[torch.Tensor] = None


CONDITION_IMAGE_SIZE = 1024 * 1024

Flux2ImageInput = Union[
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
        self.pipeline: Flux2Pipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler
        
        self._has_warned_inference_fallback = False
        self._has_warned_forward_fallback = False
        self._has_warned_preprocess_fallback = False
    
    def load_pipeline(self) -> Flux2Pipeline:
        return Flux2Pipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Flux.2 DiT."""
        return [
            # --- Double Stream Block Targets ---
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.linear_in", "ff.linear_out", 
            "ff_context.linear_in", "ff_context.linear_out",
            
            # --- Single Stream Block Targets ---
            "attn.to_qkv_mlp_proj", 
            # "attn.to_out.0"
        ]

    # ======================== Encoding & Decoding ========================

    # ------------------------- Text Encoding ------------------------
    def _get_mistral_3_small_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        system_message: str = SYSTEM_MESSAGE,
        hidden_states_layers: List[int] = (10, 20, 30),
    ):
        dtype = self.pipeline.text_encoder.dtype if dtype is None else dtype
        device = self.pipeline.text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Format input messages
        messages_batch = format_input(prompts=prompt, system_message=system_message)

        # Process all messages at once
        inputs = self.pipeline.tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass through the model
        output = self.pipeline.text_encoder(
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
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: List[int] = (10, 20, 30),
    ) -> Dict[str, torch.Tensor]:
        """Encode prompt(s) into embeddings using the Flux.2 text encoder."""
        device = device or self.pipeline.text_encoder.device

        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt_ids, prompt_embeds = self._get_mistral_3_small_prompt_embeds(
            prompt=prompt,
            device=device,
            max_sequence_length=max_sequence_length,
            system_message=self.pipeline.system_message,
            hidden_states_layers=text_encoder_out_layers,
        )

        text_ids = self.pipeline._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return {
            'prompt_ids': prompt_ids,
            'prompt_embeds': prompt_embeds,
            'text_ids': text_ids,
        }

    # ------------------------- Image Encoding ------------------------
    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        condition_image_size : Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Encode input condition_image(s) into latent representations using the Flux.2 image encoder."""
        device = device or self.pipeline.vae.device
        dtype = dtype or self.pipeline.vae.dtype
        condition_image_tensors : List[torch.Tensor] = self._resize_condition_images(
            condition_images=images,
            condition_image_size=condition_image_size,
            **filter_kwargs(self._resize_condition_images, **kwargs)
        )
        image_latents, image_latent_ids =  self.pipeline.prepare_image_latents(
            images=condition_image_tensors,
            batch_size=1,
            device=device,
            dtype=dtype,
            generator=kwargs.get('generator', None),
            # `generator` is not used, since eventually it calls `vae.encode` -> `retrieve_latents` with `argmax` mode, which is deterministic
        )
        return {
            'image_latents': image_latents,
            'image_latent_ids': image_latent_ids,
        }
    
    # ------------------------- Video Encoding ------------------------
    def encode_video(self, videos: Any, **kwargs) -> None:
        """Flux.2 does not support video encoding."""
        pass

    # ------------------------- Latent Decoding ------------------------
    def decode_latents(self, latents: torch.Tensor, latent_ids, **kwargs) -> List[Image.Image]:
        latents = self.pipeline._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.pipeline.vae.bn.running_var.view(1, -1, 1, 1) + self.pipeline.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self.pipeline._unpatchify_latents(latents)

        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type='pil')

        return images

    def _resize_condition_images(
        self,
        condition_images: Union[Image.Image, List[Image.Image]],
        condition_image_size : Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        **kwargs
    ) -> List[torch.Tensor]:
        """Preprocess condition images for Flux.2 model."""
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

    def _standardize_image_input(
        self,
        images: Flux2ImageInput,
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

    # ========================Preprocessing ========================
    def preprocess_func(
        self,
        prompt: List[str],
        images: Optional[Union[List[Optional[Image.Image]], List[List[Optional[Image.Image]]]]] = None,
        caption_upsample_temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """Preprocess inputs for Flux.2 model. The inputs are expected to be batches."""
        if images is not None:
            assert len(prompt) == len(images), "Prompts and images must have same batch size"
            # Normalize images to List[List[Image | None]]
            if isinstance(images, list) and all(isinstance(imgs, Image.Image) or imgs is None for imgs in images):
                images = [[imgs] for imgs in images]

            has_images = any(img is not None for img_list in images for img in img_list)
        else:
            has_images = False


        # Case 1: batch process when no images
        if not has_images:
            final_prompts = (
                self.pipeline.upsample_prompt(
                    prompt=prompt,
                    images=None,
                    temperature=caption_upsample_temperature,
                ) if caption_upsample_temperature else prompt
            )
            batch = self.encode_prompt(
                prompt=final_prompts,
                **filter_kwargs(self.encode_prompt, **kwargs)
            )
            return batch
        
        # Case 2: process each sample individually

        if not self._has_warned_preprocess_fallback:
            logger.warning(
                "Flux.2: Batched image processing unsupported. Processing individually (warning shown once)."
            )
            self._has_warned_preprocess_fallback = True

        batch = []
        for p, imgs in zip(prompt, images):
            final_p = (
                self.pipeline.upsample_prompt(
                    prompt=p,
                    images=imgs,
                    temperature=caption_upsample_temperature,
                )
                if caption_upsample_temperature else p
            )
            prompt_encode_dict = self.encode_prompt(
                prompt=final_p,
                **filter_kwargs(self.encode_prompt, **kwargs)
            )

            if len(imgs) == 0 or imgs[0] is None:
                image_dict = {
                    'image_latents': None,
                    'image_latent_ids': None,
                }
            else:
                image_encode_dict = self.encode_image(
                    images=imgs,
                    **filter_kwargs(self.encode_image, **kwargs)
                )
                sample = {**prompt_encode_dict, **image_encode_dict}
                batch.append(sample)
    
        # Collate batch
        collated_batch ={
            k: [sample[k] for sample in batch]
            for k in batch[0].keys()
        }
        return collated_batch
        
    
    # ======================== Sampling / Inference ========================

    # Since Flux.2 does not support ragged batches of condition images, we implement a single-sample inference method.
    @torch.no_grad()
    def _inference(
        self,
        # Ordinary arguments
        images: Optional[Flux2ImageInput] = None,
        prompt: Union[str, List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,

        # Prompt encoding arguments
        prompt_ids: Optional[torch.LongTensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,

        # Image encoding arguments
        image_latents: Optional[Union[torch.Tensor, List[Union[None, torch.Tensor]]]] = None,
        image_latent_ids: Optional[Union[torch.Tensor, List[Union[None, torch.Tensor]]]] = None,

        # Other arguments
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
        caption_upsample_temperature: Optional[float] = None,
        compute_log_prob: bool = False,

        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
    ) -> List[Flux2Sample]:
        """
        Inference method for Flux.2 model for a single sample.
        The condition images can be a list of images or a single image, shared across the batch.
        """

        # 1. Preprocess inputs
        device = self.device
        if isinstance(prompt, str):
            prompt = [prompt]
        
        if images is not None:
            images = self._standardize_image_input(images, output_type='pt')

        if (
            (prompt is not None and (prompt_embeds is None or text_ids is None))
            or (images is not None and (image_latents is None or image_latent_ids is None))
        ):
            encode_dict = self.preprocess_func(
                prompt=prompt,
                images=images,
                device=device,
                text_encoder_out_layers=text_encoder_out_layers,
                max_sequence_length=max_sequence_length,
                caption_upsample_temperature=caption_upsample_temperature,
            )
            prompt_ids = encode_dict['prompt_ids'][0]
            prompt_embeds = encode_dict['prompt_embeds'][0]
            text_ids = encode_dict['text_ids'][0]
            image_latents = encode_dict['image_latents'][0] if encode_dict['image_latents'] is not None else None
            image_latent_ids = encode_dict['image_latent_ids'][0] if encode_dict['image_latent_ids'] is not None else None
        else:
            prompt_ids = prompt_ids.to(device)
            prompt_embeds = prompt_embeds.to(device)
            text_ids = text_ids.to(device)
            image_latents = image_latents.to(device) if image_latents is not None else None
            image_latent_ids = image_latent_ids.to(device) if image_latent_ids is not None else None

        batch_size = prompt_embeds.shape[0]
        dtype = prompt_embeds.dtype

        # 2. Prepare initial noise
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, latent_ids = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # 3. Prepare timesteps
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
        timesteps = set_scheduler_timesteps(
            scheduler=self.pipeline.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            mu=mu,
        )

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])


        # 4. Run diffusion process
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

        # 5. Decode latents to images
        decoded_images = self.decode_latents(latents, latent_ids)
        # decoded_condition_images = self.decode_latents(image_latents, image_latent_ids) if image_latents is not None else None

        # 6. Create samples

        # Transpose `extra_call_back_res` tensors to have batch dimension first
        # (T, B, ...) -> (B, T, ...)
        extra_call_back_res = {
            k: torch.stack(v, dim=1)
            if isinstance(v[0], torch.Tensor) else v
            for k, v in extra_call_back_res.items()
        }

        samples = [
            Flux2Sample(
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

                # Condition images & latents
                # condition_images=decoded_condition_images if decoded_condition_images is not None else None
                condition_images=images if images is not None else None,
                image_latents=image_latents[b] if image_latents is not None else None,
                image_latent_ids=image_latent_ids[b] if image_latent_ids is not None else None,
                extra_kwargs={
                    'guidance_scale': guidance_scale,
                    **{k: v[b] for k, v in extra_call_back_res.items()}
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()
        
        return samples

    @torch.no_grad()
    def inference(
        self,
        images: Optional[Union[Flux2ImageInput, List[Flux2ImageInput]]] = None,
        prompt: Optional[List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_ids: Optional[torch.LongTensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        image_latents: Optional[Union[torch.Tensor, List[Union[None, torch.Tensor]]]] = None,
        image_latent_ids: Optional[Union[torch.Tensor, List[Union[None, torch.Tensor]]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
        caption_upsample_temperature: Optional[float] = None,
        compute_log_prob: bool = False,
        extra_call_back_kwargs: List[str] = []
    ) -> List[Flux2Sample]:
        """Batch inference for Flux2"""
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
            # Images are shared across the batch
            return self._inference(
                # Ordinary args
                images=images,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                # Encoded prompt
                prompt_ids=prompt_ids,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                # Encoded images
                image_latents=image_latents,
                image_latent_ids=image_latent_ids,
                # Other args
                attention_kwargs=attention_kwargs,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
                caption_upsample_temperature=caption_upsample_temperature,
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
                images=images[idx] if images is not None else None,
                prompt=prompt[idx] if prompt is not None else None,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                # Encoded prompt
                prompt_ids=prompt_ids[idx] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[idx] if prompt_embeds is not None else None,
                text_ids=text_ids[idx] if text_ids is not None else None,
                # Encoded image
                image_latents=image_latents[idx] if image_latents is not None else None,
                image_latent_ids=image_latent_ids[idx] if image_latent_ids is not None else None,

                # Other args
                attention_kwargs=attention_kwargs,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
                caption_upsample_temperature=caption_upsample_temperature,
                compute_log_prob=compute_log_prob,
                extra_call_back_kwargs=extra_call_back_kwargs,
            )
            samples.extend(sample)
        return samples

    # ======================== Forward (Training) ========================

    def _i2i_forward(
        self,
        sample : Flux2Sample,
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> SDESchedulerOutput:
        """Forward method wrapper for single sample."""
        batch_size = 1
        device = self.device

        # 1. Extract data from sample
        guidance_scale = sample.extra_kwargs.get('guidance_scale', self.training_args.guidance_scale)
        guidance = torch.as_tensor([guidance_scale], device=device, dtype=torch.float32)

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

    def _t2i_forward(
        self,
        samples: Union[Flux2Sample, List[Flux2Sample]],
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,        
    ) -> SDESchedulerOutput:
        if not isinstance(samples, list):
            samples = [samples]

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

    def forward(
        self,
        samples: List[Flux2Sample],
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
