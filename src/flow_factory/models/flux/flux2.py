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
from typing import Union, List, Dict, Any, Optional, Tuple, Literal, ClassVar
from dataclasses import dataclass
from PIL import Image
from collections import defaultdict
import numpy as np
from accelerate import Accelerator
import torch
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline, format_input, compute_empirical_mu
from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE, SYSTEM_MESSAGE_UPSAMPLING_T2I, SYSTEM_MESSAGE_UPSAMPLING_I2I
import logging

from ..abc import BaseAdapter
from ...samples import I2ISample
from ...hparams import *
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SDESchedulerOutput,
    set_scheduler_timesteps
)
from ...utils.base import filter_kwargs
from ...utils.image import (
    ImageSingle,
    ImageBatch,
    MultiImageBatch,
    is_image,
    is_image_batch,
    is_multi_image_batch,
    standardize_image_batch,
)
from ...utils.trajectory_collector import (
    TrajectoryCollector,
    CallbackCollector,
    TrajectoryIndicesType, 
    create_trajectory_collector,
    create_callback_collector,
)
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class Flux2Sample(I2ISample):
    """Output class for Flux2Adapter models."""
    # Class vars
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    # Obj vars
    latent_ids : Optional[torch.Tensor] = None
    text_ids : Optional[torch.Tensor] = None
    image_latents : Optional[torch.Tensor] = None
    image_latent_ids : Optional[torch.Tensor] = None


CONDITION_IMAGE_SIZE = (1024, 1024)

class Flux2Adapter(BaseAdapter):    
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
            "attn.to_out.0"
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
        hidden_states_layers: Tuple[int, ...] = (10, 20, 30),
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
        text_encoder_out_layers: Tuple[int, ...] = (10, 20, 30),
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
        images: Union[ImageSingle, ImageBatch, MultiImageBatch],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Union[List[List[torch.Tensor]], List[torch.Tensor]]]:
        """
        Encode input condition image(s) into latent representations using the Flux.2 image encoder.
        
        Supports both single batch and nested batch inputs:
        - Single batch: Image.Image, List[Image.Image], torch.Tensor, np.ndarray
        - Nested batch: List[List[Image.Image]], List[torch.Tensor(4D)], torch.Tensor(5D)
        
        Args:
            images: Input images in various formats
            condition_image_size: Target size for condition images
            device: Device to place tensors on
            dtype: Data type for tensors
            generator: Random generator for encoding (not used, kept for API consistency)
        
        Returns:
            Dictionary containing:
            - condition_images: List[List[torch.Tensor(3, H, W)]]
            - image_latents: List[torch.Tensor(1, seq_len, C)]
            - image_latent_ids: List[torch.Tensor(1, seq_len)]
        """
        device = device or self.pipeline.vae.device
        dtype = dtype or self.pipeline.vae.dtype

        # Check if input is a batch of condition image lists (nested batch)
        # Wrap single batch into nested structure
        images = [images] if not self._is_multi_images_batch(images) else images 
        
        # Standardize each batch to PIL format
        images = [self._standardize_image_input(imgs, output_type='pil') for imgs in images]
        
        # Resize all condition images
        condition_image_tensors: List[List[torch.Tensor]] = [
            self._resize_condition_images(
                condition_images=imgs,
                condition_image_size=condition_image_size,
            ) for imgs in images
        ]
        
        # Encode each batch separately
        image_latents_list = []
        image_latent_ids_list = []
        for cond_img_tensors in condition_image_tensors:
            image_latents, image_latent_ids = self.pipeline.prepare_image_latents(
                images=cond_img_tensors,
                batch_size=1,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            image_latents_list.append(image_latents.squeeze(0))
            image_latent_ids_list.append(image_latent_ids.squeeze(0))
        
        # Convert back to [0, 1] range tensors for storage
        condition_image_tensors: List[List[torch.Tensor]] = [
            [
                self.pipeline.image_processor.postprocess(img, output_type='pt')[0]
                for img in cond_img_tensors
            ]
            for cond_img_tensors in condition_image_tensors
        ]
        
        return {
            'condition_images': condition_image_tensors,  # List[List[torch.Tensor(3, H, W)]]
            'image_latents': image_latents_list,          # List[torch.Tensor(seq_len, C)]
            'image_latent_ids': image_latent_ids_list,    # List[torch.Tensor(seq_len, 3)]
        }
    
    def _resize_condition_images(
        self,
        condition_images: Union[Image.Image, List[Image.Image]],
        condition_image_size : Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
    ) -> List[torch.Tensor]:
        """Preprocess condition images for Flux.2 model."""
        if isinstance(condition_image_size, int):
            condition_image_size = (condition_image_size, condition_image_size)

        if isinstance(condition_images, Image.Image):
            condition_images = [condition_images]

        condition_images = self._standardize_image_input(
            condition_images,
            output_type='pil',
        )

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

    @staticmethod
    def _is_multi_images_batch(images : Union[ImageBatch, MultiImageBatch]):
        return is_multi_image_batch(images)

    @staticmethod
    def _is_ragged_multi_image_batch(images : Union[ImageBatch, MultiImageBatch]):
        is_ragged_batch = (
            ( isinstance(images, list) and len(images) > 0 and isinstance(images[0], list) ) # List[List[Image]]
            or
            ( isinstance(images, list) and len(images) > 0 and isinstance(images[0], (np.ndarray, torch.Tensor)) and images[0].ndim == 4 ) # List[torch.Tensor : ndim=4]
        )
        return is_ragged_batch
    
    @staticmethod
    def _is_multi_image_latents(image_latents: Union[torch.Tensor, List[torch.Tensor]]):
        is_ragged_image_latents = (
            (
                isinstance(image_latents, list) and len(image_latents) > 0
                and isinstance(image_latents[0], torch.Tensor) and image_latents[0].ndim == 2
            ) # List[torch.Tensor : ndim=2 (seq_len, C)]
            or (
                isinstance(image_latents, torch.Tensor) and image_latents.ndim == 3
             ) # torch.Tensor : ndim=3 (B, seq_len, C)
        )
        return is_ragged_image_latents

    @staticmethod
    def _is_ragged_multi_image_latents(image_latents: Union[torch.Tensor, List[torch.Tensor]]):
        is_ragged_image_latents = (
            isinstance(image_latents, list) and len(image_latents) > 0
            and isinstance(image_latents[0], torch.Tensor) and image_latents[0].ndim == 2
        ) # List[torch.Tensor : ndim=2 (seq_len, C)]
        return is_ragged_image_latents
    
    def _standardize_image_input(
        self,
        images: Union[ImageSingle, ImageBatch],
        output_type: Literal['pil', 'pt', 'np'] = 'pil',
    ) -> ImageBatch:
        """
        Standardize image input to desired output type.
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        return standardize_image_batch(
            images,
            output_type=output_type,
        )
    # ------------------------- Video Encoding ------------------------
    def encode_video(self, videos: Any) -> None:
        """Flux.2 does not support video encoding."""
        pass

    # ------------------------- Latent Decoding ------------------------
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

    # ======================== Preprocessing ========================
    def preprocess_func(
        self,
        prompt: List[str],
        images: Optional[MultiImageBatch] = None,
        caption_upsample_temperature: Optional[float] = None,
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int, ...] = (10, 20, 30),
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """
        Preprocess inputs for Flux.2 model (batched processing).
        
        Args:
            prompt: List of text prompts
            images: Optional images in various formats (MultiImageBatch)
            caption_upsample_temperature: Temperature for prompt upsampling
            max_sequence_length: Max sequence length for text encoder
            text_encoder_out_layers: Layers to extract from text encoder
            generator: Random generator for encoding (not used, kept for API consistency)
        
        Returns:
            Dictionary with all encoded data in list format for consistency
        """
        # 1. Normalize images to List[List[Image | None]]
        if images is not None:
            assert len(prompt) == len(images), "Prompts and images must have same batch size"
            if isinstance(images, list) and all(isinstance(img, Image.Image) or img is None for img in images):
                images = [[img] for img in images]
            
            has_images = any(img is not None for img_list in images for img in img_list)
        else:
            has_images = False

        # 2: Handle caption upsampling
        if caption_upsample_temperature is not None:
            final_prompts = []
            for i, p in enumerate(prompt):
                imgs = images[i] if images is not None else None
                upsampled = self.pipeline.upsample_prompt(
                    prompt=p,
                    images=imgs,
                    temperature=caption_upsample_temperature,
                )
                final_prompts.append(upsampled)
        else:
            final_prompts = prompt

        # 3: Batch encode prompts
        batch = self.encode_prompt(
            prompt=final_prompts,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        
        # 4: Batch encode images if present
        if has_images:
            image_dict = self.encode_image(
                images=images,
                condition_image_size=condition_image_size,
                device=self.device,
                generator=generator,
            )
            # image_dict already returns lists, so directly merge
            batch.update(image_dict)
        
        return batch

    # ======================== Sampling / Inference ========================

    # Since Flux.2 does not support ragged batches of condition images, we implement a single-sample inference method.
    @torch.no_grad()
    def _inference(
        self,
        # Ordinary arguments
        images: Optional[Union[ImageBatch, MultiImageBatch]] = None, # A batch of condition images
        prompt: Union[str, List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # Prompt encoding arguments
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        # Image encoding arguments
        condition_images: Optional[MultiImageBatch] = None,
        image_latents: Optional[torch.Tensor] = None,
        image_latent_ids: Optional[torch.Tensor] = None,
        # Other arguments
        condition_image_size : Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int, ...] = (10, 20, 30),
        caption_upsample_temperature: Optional[float] = None,
        compute_log_prob: bool = False,
        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = 'all',
    ) -> List[Flux2Sample]:
        """
        Inference method for Flux.2 model for a single sample.
        The condition images can be a list of images or a single image, shared across the batch.
        """
        # 1. Preprocess inputs
        device = self.device
        if isinstance(prompt, str):
            prompt = [prompt]
        
        images = [images] if images is not None and not self._is_multi_images_batch(images) else images

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
                condition_image_size=condition_image_size,
            )
            prompt_ids = encode_dict['prompt_ids']
            prompt_embeds = encode_dict['prompt_embeds']
            text_ids = encode_dict['text_ids']
            # Potential issue: the following stack relies on uniform size of input condition images
            condition_images = (
                encode_dict['condition_images'] # List[List[torch.Tensor(3, H, W)]] with len B
                if encode_dict.get('condition_images', None) is not None
                else None
            )
            image_latents = (
                torch.stack(encode_dict['image_latents'], dim=0) # torch.Tensor(B, seq_len, C)
                if encode_dict.get('image_latents', None) is not None
                else None
            )
            image_latent_ids = (
                torch.stack(encode_dict['image_latent_ids'], dim=0) # torch.Tensor(B, seq_len, 3)
                if encode_dict.get('image_latent_ids', None) is not None
                else None
            )
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
        
        # 4. Run diffusion process
        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        # Inside denoising loop in _inference, replace the inline transformer call with:
        for i, t in enumerate(timesteps):
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kwargs = list(set(['next_latents', 'log_prob', 'noise_pred'] + extra_call_back_kwargs))
            current_compute_log_prob = compute_log_prob and current_noise_level > 0

            output = self._forward(
                t=t,
                latents=latents,
                latent_ids=latent_ids,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                image_latents=image_latents,
                image_latent_ids=image_latent_ids,
                guidance_scale=guidance_scale,
                joint_attention_kwargs=joint_attention_kwargs,
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

        # 5. Decode latents to images
        decoded_images = self.decode_latents(latents, latent_ids, output_type='pt')
        # decoded_condition_images = self.decode_latents(image_latents, image_latent_ids, output_type='pt') if image_latents is not None else None

        # 6. Create samples
        extra_call_back_res = callback_collector.get_result()          # (B, len(trajectory_indices), ...)
        callback_index_map = callback_collector.get_index_map()        # (T,) LongTensor
        all_latents = latent_collector.get_result()                    # List[torch.Tensor(B, ...)]
        latent_index_map = latent_collector.get_index_map()            # (T+1,) LongTensor
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
        samples = [
            Flux2Sample(
                # Denoising trajectory
                timesteps=timesteps,
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0) if all_latents is not None else None,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if all_log_probs is not None else None,
                latent_index_map=latent_index_map,
                log_prob_index_map=log_prob_index_map,
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
                condition_images=condition_images[b] if condition_images is not None else None, # The condition images are shared and without batch dimension
                image_latents=image_latents[b] if image_latents is not None else None, # If not None, it has batch dim 1
                image_latent_ids=image_latent_ids[b] if image_latent_ids is not None else None, # If not None, it has batch dim 1
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

    @torch.no_grad()
    def inference(
        self,
        # Ordinary arguments
        images: Optional[MultiImageBatch] = None,
        prompt: Optional[List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
        caption_upsample_temperature: Optional[float] = None,
        # Encoded prompt
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        # Encoded images
        condition_images: Optional[MultiImageBatch] = None,
        image_latents: Optional[Union[torch.Tensor, List[Union[None, torch.Tensor]]]] = None,
        image_latent_ids: Optional[Union[torch.Tensor, List[Union[None, torch.Tensor]]]] = None,
        # Other arguments
        compute_log_prob: bool = False,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = 'all',
    ) -> List[Flux2Sample]:
        """Batch inference for Flux2"""
        if isinstance(prompt, str):
            prompt = [prompt]
        # # Approach 1: Fallback for ragged I2I - unstable asynchronization among processes
        # is_ragged_images = self._is_ragged_multi_image_batch(images)
        # is_ragged_image_latents = self._is_ragged_multi_image_latents(images)
        # fall_back = (is_ragged_images or is_ragged_image_latents)

        # Approach 2: Fallback for all I2I, this is good for asynchronization among processes
        is_nested_images = self._is_multi_images_batch(images)
        is_nested_image_latents = self._is_multi_image_latents(image_latents)
        fall_back = (is_nested_images or is_nested_image_latents)
        if not fall_back:
            # batched T2I or uniformed I2I
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
                condition_images=condition_images,
                image_latents=image_latents,
                image_latent_ids=image_latent_ids,
                # Other args
                joint_attention_kwargs=joint_attention_kwargs,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
                caption_upsample_temperature=caption_upsample_temperature,
                compute_log_prob=compute_log_prob,
                extra_call_back_kwargs=extra_call_back_kwargs,
                trajectory_indices=trajectory_indices,
            )
    
        # Ragged case: per-sample fallback
        if not self._has_warned_inference_fallback:
            logger.warning(
                "FLUX.2 does not support batch inference with varying condition images per sample. "
                "Falling back to single-sample inference. This warning will only appear once."
            )
            self._has_warned_inference_fallback = True
        # Process each sample individually by calling _inference
        batch_size = len(images) if images is not None else len(image_latents)

        samples = []
        for idx in range(batch_size):
            # Extract single sample tensors -  keep batch dimension as 1
            # Prompt
            this_prompt = prompt[idx] if prompt is not None else None
            this_prompt_ids = prompt_ids[idx].unsqueeze(0) if prompt_ids is not None else None
            this_prompt_embeds = prompt_embeds[idx].unsqueeze(0) if prompt_embeds is not None else None
            this_text_ids = text_ids[idx].unsqueeze(0) if text_ids is not None else None
            # Image
            this_images=images[idx] if images is not None else None # No batch dimension for `images`
            this_condition_images=condition_images[idx:idx+1] if condition_images is not None else None
            this_image_latents=image_latents[idx].unsqueeze(0) if image_latents is not None else None
            this_image_latent_ids=image_latent_ids[idx].unsqueeze(0) if image_latent_ids is not None else None
            sample = self._inference(
                # Ordinary args
                images=this_images,
                prompt=this_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator[idx] if isinstance(generator, list) else generator,
                # Encoded prompt
                prompt_ids=this_prompt_ids, # Keep batch dim as 1
                prompt_embeds=this_prompt_embeds,
                text_ids=this_text_ids,
                # Encoded image
                condition_images=this_condition_images,
                image_latents=this_image_latents,
                image_latent_ids=this_image_latent_ids,
                # Other args
                joint_attention_kwargs=joint_attention_kwargs,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
                caption_upsample_temperature=caption_upsample_temperature,
                compute_log_prob=compute_log_prob,
                extra_call_back_kwargs=extra_call_back_kwargs,
                trajectory_indices=trajectory_indices,
            )
            samples.extend(sample)
        return samples

    # ======================== Forward (Training) ========================
    def _forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        # Optional for I2I
        image_latents: Optional[torch.Tensor] = None,
        image_latent_ids: Optional[torch.Tensor] = None,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        guidance_scale: float = 4.0,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = ['noise_pred', 'next_latents', 'next_latents_mean', 'std_dev_t', 'dt', 'log_prob'],
        noise_level: Optional[float] = None,
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """
        Core forward pass handling both T2I and I2I.

        Args:
            t: Current timestep tensor.
            t_next: Next timestep tensor.
            latents: Current latent representations (B, seq_len, C).
            latent_ids: Latent position IDs (B, seq_len, 4).
            prompt_embeds: Text prompt embeddings.
            text_ids: Text position IDs.
            image_latents: Optional condition image latents (for I2I).
            image_latent_ids: Optional condition image position IDs.
            guidance_scale: Guidance scale factor.
            next_latents: Optional target latents for log-prob computation.
            joint_attention_kwargs: Optional kwargs for attention layers.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            noise_level: Current noise level for SDE sampling.

        Returns:
            SDESchedulerOutput containing requested outputs.
        """
        # 1. Prepare variables        
        batch_size = latents.shape[0]

        guidance = torch.full([batch_size], guidance_scale, device=latents.device, dtype=torch.float32)

        # Prepare model input (concatenate condition latents for I2I)
        latent_model_input = latents.to(torch.float32)
        latent_image_ids = latent_ids

        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1).to(torch.float32)
            latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

        # Forward pass
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=t.expand(batch_size) / 1000,  # Scale timestep
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

        # Extract only target latent predictions (exclude condition image part)
        noise_pred = noise_pred[:, :latents.shape[1]]

        # Scheduler step
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

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: Union[torch.Tensor, List[torch.Tensor]],
        prompt_embeds: torch.Tensor,
        text_ids: Union[torch.Tensor, List[torch.Tensor]],
        # Optional for I2I (can be List for ragged batches)
        image_latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        image_latent_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        guidance_scale: float = 4.0,
        noise_level: Optional[float] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = ['noise_pred', 'next_latents', 'next_latents_mean', 'std_dev_t', 'dt', 'log_prob'],
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """
        General forward method handling both T2I and I2I, including ragged I2I batches.
        """
        # # Approach 1: Fallback only when ragged I2I
        # is_ragged_multi_image_latents = self._is_ragged_image_latents(image_latents)
        # fall_back = is_ragged_multi_image_latents

        # Approach 2: Fallback for all I2I, this is good for asynchronization among processes
        fall_back = image_latents is not None

        if not fall_back:
            # T2I or uniform I2I, call _forward() directly
            return self._forward(
                t=t,
                latents=latents,
                latent_ids=latent_ids,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                image_latents=image_latents,
                image_latent_ids=image_latent_ids,
                guidance_scale=guidance_scale,
                t_next=t_next,
                next_latents=next_latents,
                joint_attention_kwargs=joint_attention_kwargs,
                compute_log_prob=compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=noise_level,
            )

        # Ragged I2I: process one by one
        if not self._has_warned_forward_fallback:
            logger.warning(
                "Flux.2: Ragged I2I detected (varying condition image sizes). "
                "Processing samples individually (warning shown once)."
            )
            self._has_warned_forward_fallback = True

        batch_size = latents.shape[0]
        outputs = []

        for idx in range(batch_size):
            # Extract single sample tensors -  keep batch dimension as 1
            # Timestep
            single_t = t[idx].unsqueeze(0)
            single_t_next = t_next[idx].unsqueeze(0)
            # Latents
            single_latents = latents[idx].unsqueeze(0)
            single_latent_ids = latent_ids[idx].unsqueeze(0)
            single_next_latents = next_latents[idx].unsqueeze(0) if next_latents is not None else None
            # Prompt
            single_prompt_embeds = prompt_embeds[idx].unsqueeze(0)
            single_text_ids = text_ids[idx].unsqueeze(0)
            # Condtion Images
            single_image_latents = image_latents[idx].unsqueeze(0) if image_latents[idx] is not None else None
            single_image_latent_ids = image_latent_ids[idx].unsqueeze(0) if image_latent_ids is not None and image_latent_ids[idx] is not None else None

            out = self._forward(
                t=single_t,
                latents=single_latents,
                latent_ids=single_latent_ids,
                prompt_embeds=single_prompt_embeds,
                text_ids=single_text_ids,
                image_latents=single_image_latents,
                image_latent_ids=single_image_latent_ids,
                guidance_scale=guidance_scale,
                t_next=single_t_next,
                next_latents=single_next_latents,
                joint_attention_kwargs=joint_attention_kwargs,
                compute_log_prob=compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=noise_level,
            )
            outputs.append(out)

        # Concatenate outputs along batch dimension
        outputs_dict = [o.to_dict() for o in outputs]
        return FlowMatchEulerDiscreteSDESchedulerOutput.from_dict({
            k: torch.cat([o[k] for o in outputs_dict], dim=0) if outputs_dict[0][k] is not None else None
            for k in outputs_dict[0].keys()
        })