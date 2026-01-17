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

# src/flow_factory/models/qwen_image/qwen_image_edit_plus.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
import logging
import math
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from diffusers.utils.torch_utils import randn_tensor

from ..abc import BaseAdapter
from ..samples import I2ISample
from ...hparams import *
from ...scheduler import FlowMatchEulerDiscreteSDEScheduler, SDESchedulerOutput, set_scheduler_timesteps
from ...utils.logger_utils import setup_logger
from ...utils.base import (
    filter_kwargs,
    is_valid_image,
    is_valid_image_batch,
    is_pil_image_batch_list,
    is_pil_image_list,
    tensor_to_pil_image,
    tensor_list_to_pil_image,
    numpy_list_to_pil_image,
    numpy_to_pil_image,
    pil_image_to_tensor,
    standardize_image_batch
)

logger = setup_logger(__name__)

CONDITION_IMAGE_SIZE = (1024, 1024)
CONDITION_IMAGE_SIZE_FOR_ENCODE = (384, 384)

QwenImageEditPlusImageInput = Union[
    Image.Image,
    np.ndarray,
    torch.Tensor,
    List[Image.Image],
    List[np.ndarray],
    List[torch.Tensor]
]

@dataclass
class QwenImageEditPlusSample(I2ISample):
    """Output class for Qwen-Image-Edit Plus model"""
    prompt_embeds_mask : Optional[torch.FloatTensor] = None
    negative_prompt_embeds_mask : Optional[torch.FloatTensor] = None
    img_shapes : Optional[List[Tuple[int, int, int]]] = None
    image_latents : Optional[torch.Tensor] = None

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def calculate_dimensions(target_area, ratio):
    # Calculate width and height based on target area and aspect ratio (height / width)
    height = math.sqrt(target_area * ratio)
    width = height / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height

class QwenImageEditPlusAdapter(BaseAdapter):
    """Adapter for Qwen-Image-Edit Plus text-to-image models."""
    
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: QwenImageEditPlusPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

        self._warned_cfg_no_neg_prompt = False
        self._warned_no_cfg = False
        self._has_warned_inference_fallback = False
        self._has_warned_forward_fallback = False
        self._has_warned_preprocess_fallback = False
        self._has_warned_inference_auto_resize = False
    
    def load_pipeline(self) -> QwenImageEditPlusPipeline:
        return QwenImageEditPlusPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Qwen-Image transformer."""
        return [
            # Attention
            "to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
            # MLP
            "net.0.proj", "net.2"
        ]
    
    # ================================= Encoding and Decoding Methods ================================= #

    # ---------------------------------- Text Encoding ---------------------------------- #

    def _standardize_image_input(
        self,
        images: QwenImageEditPlusImageInput,
        output_type: Literal['pil', 'pt', 'np'] = 'pil',
    ):
        """
        Standardize image input to desired output type.
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        images = standardize_image_batch(
            images,
            output_type=output_type,
        )
        return images

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        images: Optional[QwenImageEditPlusImageInput] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 1024,
    ):
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        images = self._standardize_image_input(images, output_type='pil') if images is not None else None
        if isinstance(images, list):
            base_img_prompt = ""
            for i, img in enumerate(images):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif images is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = self.pipeline.prompt_template_encode

        drop_idx = self.pipeline.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]

        model_inputs = self.pipeline.processor(
            text=txt,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(device)
        input_ids = model_inputs.input_ids

        outputs = self.pipeline.text_encoder(
            input_ids=input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self.pipeline._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        input_ids = input_ids[:, drop_idx:] # Extract only user input ids

        return input_ids, prompt_embeds, encoder_attention_mask
    

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        images : Optional[QwenImageEditPlusImageInput] = None,
        max_sequence_length: int = 1024,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Union[torch.LongTensor, torch.Tensor]]:
        """Encode text prompts using the pipeline's text encoder."""

        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        # Encode positive prompt
        prompt_ids, prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
            prompt=prompt,
            images=images,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length
        )
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        results = {
            "prompt_ids": prompt_ids,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
        }
        # Encode negative prompt
        if negative_prompt:
            negative_prompt_ids, negative_prompt_embeds, negative_prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt=negative_prompt,
                images=images,
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length
            )
            results.update({
                "negative_prompt_ids": negative_prompt_ids,
                "negative_prompt_embeds": negative_prompt_embeds[:, :max_sequence_length],
                "negative_prompt_embeds_mask": negative_prompt_embeds_mask[:, :max_sequence_length],
            })

        return results
    
    # ---------------------------------------- Image Encoding ---------------------------------- #

    def _preprocess_condition_images(
        self,
        images: QwenImageEditPlusImageInput,
        condition_image_size : Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        generator : Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], List[Tuple[int, int]]]]: 
        """
        Preprocess condition images and prepare image latents.
        The input requires `multiple condition images` used to generate one image.
        Args:
            images: QwenImageEditPlusImageInput
                - A list of conditioning images.
                - Each element can be: PIL Image, np.ndarray, torch.Tensor, or a list of these types.
                - Each image will be resized to fit within `condition_image_size` while maintaining aspect ratio.
            condition_image_size: `Union[int, Tuple[int, int]]`
                - Maximum size for conditioning images. If int, will be used for both height and width.
        Returns:
            Dictionary containing:
                - "condition_images": List of resized conditioning images.
                - "condition_image_sizes": List of sizes for conditioning images.
                - "vae_images": List of preprocessed images for VAE encoding.
                - "vae_image_sizes": List of sizes for VAE images.
                - "image_latents": batch of packed image latents
        """
        dtype = dtype or self.pipeline.vae.dtype
        device = device or self.pipeline.vae.device
        if isinstance(condition_image_size, int):
            condition_image_size = (condition_image_size, condition_image_size)

        condition_image_max_area = condition_image_size[0] * condition_image_size[1]
        condition_image_for_encode_max_area = CONDITION_IMAGE_SIZE_FOR_ENCODE[0] * CONDITION_IMAGE_SIZE_FOR_ENCODE[1]
        images = self._standardize_image_input(images, output_type='pil')

        condition_image_sizes = []
        condition_images = []
        vae_image_sizes = []
        vae_images = []
        for img in images:
            image_width, image_height = img.size
            # Maintain the original aspect ratio and fit within the maximum area.
            # The original Diffusers pipeline uses a hard-coded 384x384 resolution for `condition_images` (prompt encoding)
            # and 1024x1024 for `vae_images` (the actual conditioning input for image generation).
            # Here, `condition_image_size` is exposed to allow control over the overall training resolution.
            condition_width, condition_height = calculate_dimensions(
                condition_image_for_encode_max_area, image_height / image_width
            )
            vae_width, vae_height = calculate_dimensions(
                condition_image_max_area, image_height / image_width
            )
            condition_image_sizes.append((condition_width, condition_height))
            vae_image_sizes.append((vae_width, vae_height))
            condition_image = self.pipeline.image_processor.resize(img, condition_height, condition_width)
            condition_image = self._standardize_image_input(condition_image, output_type='pt')[0] # Convert to tensor (C, H, W)
            condition_images.append(condition_image)
            vae_images.append(self.pipeline.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2)) # (1, C, 1, H, W)

        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        image_latents = self.prepare_image_latents(
            images=vae_images,
            batch_size=1,
            num_channels_latents=num_channels_latents,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        return {
            "condition_images": condition_images, # List[tensor(C, H, W)]
            "condition_image_sizes": condition_image_sizes, # List[Tuple[int, int]]
            "vae_images": vae_images, # List[tensor(1, C, 1, H, W)]
            "vae_image_sizes": vae_image_sizes, # List[Tuple[int, int]]
            'image_latents': image_latents, # tensor(1, seq_len_total, C)
        }
        
    def encode_image(
        self,
        images: List[QwenImageEditPlusImageInput],
        condition_image_size : Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        generator : Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, List[Union[torch.Tensor, List[torch.Tensor], List[Tuple[int, int]]]]]:
        """
        Encode input images into latent representations using the VAE encoder.
        Args:
            images: List[`QwenImageEditPlusImageInput`]
                - A batch of conditioning image lists.
                - Each element can be: PIL Image, np.ndarray, torch.Tensor, or a list of these types.
                - Each image will be resized to fit within `condition_image_size` while maintaining aspect ratio.
            condition_image_size: `Union[int, Tuple[int, int]]`
                - Maximum size for conditioning images. If int, will be used for both height and width.
        Returns:
            Dictionary containing:
                - "condition_images": Nested list of resized conditioning images.
                - "condition_image_sizes": Nested list of sizes for conditioning images.
                - "vae_images": Nested list of VAE image tensors.
                - "vae_image_sizes": List of sizes for VAE images.
                - "image_latents": batch of packed image latents
        """
        # Check if input is a batch of condition image lists (nested batch)
        is_nested_batch = (
            (isinstance(images, list) and len(images) > 0 and isinstance(images[0], list))
            or (isinstance(images, list) and len(images) > 0 and isinstance(images[0], (np.ndarray, torch.Tensor)) and images[0].ndim == 4)
            or (isinstance(images, torch.Tensor) and images.ndim == 5)
            or (isinstance(images, np.ndarray) and images.ndim == 5)
        )
        if not is_nested_batch:
            images = [images] # Wrap into a batch of one

        results = defaultdict(list)
        for cond_images in images:
            encoded = self._preprocess_condition_images(
                cond_images,
                condition_image_size=condition_image_size,
                generator=generator,
                dtype=dtype,
                device=device,
            )
            for k, v in encoded.items():
                results[k].append(v)

        return results

    def prepare_image_latents(
        self,
        images : QwenImageEditPlusImageInput,
        batch_size : int,
        num_channels_latents : int,
        dtype : torch.dtype,
        device : torch.device,
        generator : Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        images = self._standardize_image_input(images, 'pt')

        all_image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.pipeline.latent_channels:
                image_latents = self.pipeline._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[3:]
            image_latents = self.pipeline._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )
            all_image_latents.append(image_latents)
        image_latents = torch.cat(all_image_latents, dim=1)
        return image_latents


    def prepare_latents(
        self,
        batch_size : int,
        num_channels_latents : int,
        height : int,
        width : int,
        dtype : torch.dtype,
        device : torch.device,
        generator : Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents : Optional[torch.Tensor] = None,
        images : Optional[QwenImageEditPlusImageInput] = None,
        image_latents : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.pipeline.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.pipeline.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        if image_latents is None and images is not None:
            image_latents = self.prepare_image_latents(
                images=images,
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                dtype=dtype,
                device=device,
                generator=generator,
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.pipeline._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    # ---------------------------------------- Video Encoding ---------------------------------- #
    def encode_video(self, videos: Union[torch.Tensor, List[torch.Tensor]], **kwargs) -> torch.Tensor:
        """Not needed for Qwen-Image-Edit models."""
        pass

    # ---------------------------------------- Image Decoding ---------------------------------- #
    def decode_latents(self, latents: torch.Tensor, height: int, width: int, **kwargs) -> List[Image.Image]:
        """Decode latents to images using VAE."""
        
        latents = self.pipeline._unpack_latents(latents, height, width, self.pipeline.vae_scale_factor)
        latents = latents.to(self.pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(1, self.pipeline.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        images = self.pipeline.vae.decode(latents, return_dict=False)[0][:, :, 0]
        images = self.pipeline.image_processor.postprocess(images, output_type='pil')

        return images

    # ========================Preprocessing ========================
    def preprocess_func(
        self,
        prompt: List[str],
        images: List[QwenImageEditPlusImageInput] = None,
        negative_prompt: Optional[List[str]] = None,
        condition_image_size : Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        max_sequence_length: int = 1024,
        generator : Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Dict[str, List[Any]]:
        """Preprocess data samples for Qwen-Image-Edit Plus model training or evaluation.

        Args:
            prompt (List[str]): A Batch of text prompts.
            images (Optional[List[QwenImageEditPlusImageInput]]): A batch of conditioning image lists.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        is_nested_batch = (
            (isinstance(images, list) and len(images) > 0 and isinstance(images[0], list))
            or (isinstance(images, list) and len(images) > 0 and isinstance(images[0], (np.ndarray, torch.Tensor)) and images[0].ndim == 4)
            or (isinstance(images, (torch.Tensor, np.ndarray)) and images.ndim == 5)
        )
        if not is_nested_batch:
            images = [images] * batch_size # Duplicate for each prompt

        results = defaultdict(list)
        encoded_images = self.encode_image(
            images=images,
            condition_image_size=condition_image_size,
            generator=generator,
        )
        for k, v in encoded_images.items():
            results[k] = v

        for i in range(batch_size):
            encoded_prompts = self.encode_prompt(
                prompt=prompt[i],
                negative_prompt=negative_prompt[i] if isinstance(negative_prompt, list) else negative_prompt,
                images=results['condition_images'][i],
                max_sequence_length=max_sequence_length,
            )
            for k, v in encoded_prompts.items():
                if isinstance(v, torch.Tensor):
                    v = list(v.unbind(0)) # Convert to a list for ragged case
                results[k].extend(v)

        return results


    # ======================== Padding Utilities ========================
    def _standardize_data(
        self,
        data : Union[None, torch.Tensor, List[torch.Tensor]],
        padding_value : Union[int, float],
        device: Optional[torch.device] = None,
        max_len: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if data is None: 
            return None
        
        # If data is a list (ragged), pad it into a batch tensor first
        if isinstance(data, list):
            # Ensure data is on the correct device before padding
            if len(data) > 0 and data[0].device != device:
                data = [t.to(device) for t in data]
            data = pad_sequence(data, batch_first=True, padding_value=padding_value)
        else:
            data = data.to(device)
        
        return data[:, :max_len] if data.shape[1] > max_len else data

    def _pad_batch_prompt(
        self,
        prompt_embeds_mask: Union[List[torch.LongTensor], torch.LongTensor],
        prompt_ids: Optional[Union[List[torch.LongTensor], torch.LongTensor]] = None,
        prompt_embeds: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        device : Optional[torch.device] = None,
    ) -> Tuple[List[int], torch.LongTensor, Optional[torch.Tensor], Optional[torch.LongTensor]]:
        if isinstance(prompt_embeds_mask, list):
            device = device or prompt_embeds_mask[0].device
            txt_seq_lens = [mask.sum() for mask in prompt_embeds_mask]
        else:
            device = device or prompt_embeds_mask.device
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

        max_pos_len = max(txt_seq_lens)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        padded_prompt_ids = self._standardize_data(
            prompt_ids,
            padding_value=pad_token_id,
            device=device,
            max_len=max_pos_len,
        )
        padded_prompt_embeds = self._standardize_data(
            prompt_embeds,
            padding_value=0.0,
            device=device,
            max_len=max_pos_len,
        )
        padded_prompt_embeds_mask = self._standardize_data(
            prompt_embeds_mask,
            padding_value=0,
            device=device,
            max_len=max_pos_len,
        )

        return (
            txt_seq_lens,
            padded_prompt_ids,
            padded_prompt_embeds,
            padded_prompt_embeds_mask,
        )
    
        # ======================== Sampling / Inference ========================

    # ================================ Inference ================================ #

    # Handle one sample
    @torch.no_grad()
    def _inference(
        self,
        # Ordinary arguments
        images: Optional[QwenImageEditPlusImageInput] = None,
        prompt: Optional[Union[List[str], str]] = None,
        negative_prompt: Optional[Union[List[str], str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0, # Corresponds to `true_cfg_scale` in Qwen-Image-Edit-Plus-Pipeline.
        height: int = 1024,
        width: int = 1024,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,

        # Prompt encoding arguments
        prompt_ids: Optional[torch.LongTensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.LongTensor] = None,
        negative_prompt_ids: Optional[torch.LongTensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.LongTensor] = None,
        
        # Image encoding arguments
        condition_images: Optional[QwenImageEditPlusImageInput] = None,
        condition_image_sizes: Optional[List[Tuple[int, int]]] = None,
        vae_images: Optional[QwenImageEditPlusImageInput] = None,
        vae_image_sizes: Optional[List[Tuple[int, int]]] = None,
        image_latents: Optional[torch.Tensor] = None,

        # Other arguments
        condition_image_size : Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        attention_kwargs: Optional[Dict[str, Any]] = {},
        max_sequence_length: int = 1024,
        compute_log_prob: bool = False,
        auto_resize : bool = True,

        # Callback arguments
        extra_call_back_kwargs: List[str] = [],
    ):
        """Generate images using Qwen-Image-Edit Plus model."""
        # 1. Set up

        # Determine height and width. Encoded `condition_images` is prioritized than raw input `images`
        detemine_size_images = condition_images if condition_images is not None else images
        detemine_size_images = self._standardize_image_input(detemine_size_images, output_type='pil') if detemine_size_images is not None else None
        if detemine_size_images is not None and auto_resize:
            # Auto resize the output image to fit the input image's aspect ratio (use the last condition image)
            image_size = detemine_size_images[-1].size
            calculated_width, calculated_height = calculate_dimensions(height * width, image_size[1] / image_size[0])
            if (calculated_height != height or calculated_width != width) and not self._has_warned_inference_auto_resize:
                self._has_warned_inference_auto_resize = True
                logger.warning(
                    f"Auto-resizing output from ({height}, {width}) to ({calculated_height}, {calculated_width}) "
                    f"to match input aspect ratio {image_size[1] / image_size[0]:.2f}. This message appears only once. "
                    f"To disable auto-resizing and enforce given resolution ({height}, {width}), set `auto_resize` to `false`."
                )

            height = calculated_height
            width = calculated_width

        multiple_of = self.pipeline.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # cfg and others
        true_cfg_scale = guidance_scale
        device = self.device
        dtype = self.pipeline.transformer.dtype
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1.0 and has_neg_prompt

        if true_cfg_scale > 1 and not has_neg_prompt and not self._warned_cfg_no_neg_prompt:
            self._warned_cfg_no_neg_prompt = True
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided. Warning will only be shown once."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt and not self._warned_no_cfg:
            self._warned_no_cfg = True
            logger.warning(
                " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1. Warning will only be shown once."
            )

        # 2. Preprocess images
        if (
            images is not None
            and (condition_images is None or vae_images is None or condition_image_sizes is None or vae_image_sizes is None)
        ):
            # Process multiple condition images
            encoded_images = self._preprocess_condition_images(
                images=images,
                condition_image_size=condition_image_size,
                generator=generator,
                dtype=dtype,
                device=device,
            )
            condition_images = encoded_images["condition_images"]
            condition_image_sizes = encoded_images["condition_image_sizes"]
            vae_images = encoded_images["vae_images"]
            vae_image_sizes = encoded_images["vae_image_sizes"]
            image_latents = encoded_images['image_latents']
        else:
            condition_images = self._standardize_image_input(condition_images, output_type='pt') if condition_images is not None else None
            if isinstance(vae_images, torch.Tensor):
                vae_images = list(vae_images.unbind(0))
            vae_images = [img.to(device) for img in vae_images]
            image_latents = image_latents.to(device) if image_latents is not None else None
        
        # 3. Encode prompts
        if (
            (prompt is not None and prompt_embeds is None and prompt_embeds_mask is None)
            or (negative_prompt is not None and negative_prompt_embeds is None and negative_prompt_embeds_mask is None)
        ):
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                images=condition_images,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            prompt_ids = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
            prompt_embeds_mask = encoded["prompt_embeds_mask"]
            negative_prompt_ids = encoded.get("negative_prompt_ids", None)
            negative_prompt_embeds = encoded.get("negative_prompt_embeds", None)
            negative_prompt_embeds_mask = encoded.get("negative_prompt_embeds_mask", None)
        else:
            prompt_embeds = prompt_embeds.to(device)
            prompt_embeds_mask = prompt_embeds_mask.to(device)
            negative_prompt_embeds = negative_prompt_embeds.to(device) if negative_prompt_embeds is not None else None
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device) if negative_prompt_embeds_mask is not None else None


        batch_size = prompt_embeds.shape[0]

        # Truncate prompt embeddings and masks to max valid lengths in the batch
        txt_seq_lens, prompt_ids, prompt_embeds, prompt_embeds_mask = self._pad_batch_prompt(
            prompt_embeds_mask,
            prompt_ids,
            prompt_embeds,
            device=device
        )

        if do_true_cfg:
            negative_txt_seq_lens, negative_prompt_ids, negative_prompt_embeds, negative_prompt_embeds_mask = self._pad_batch_prompt(
                negative_prompt_embeds_mask,
                negative_prompt_ids,
                negative_prompt_embeds,
                device=device
            )

        
        # 4. Prepare latents
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            images=vae_images,
            image_latents=image_latents,
        )
        img_shapes = [
            [
                (1, height // self.pipeline.vae_scale_factor // 2, width // self.pipeline.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.pipeline.vae_scale_factor // 2, vae_width // self.pipeline.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size


        # 5. Set scheduler timesteps
        timesteps = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device,
        )

        guidance = None # Always None for Qwen-Image-Edit Plus


        # 6. Denoising loop
        all_latents = [latents]
        all_log_probs = [] if compute_log_prob else None
        extra_call_back_res = defaultdict(list)

        for i, t in enumerate(timesteps):
            timestep = t.expand(batch_size).to(latents.dtype)
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)

            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1) # Concatenate along sequence dimension

            # Conditioned prediction
            with self.pipeline.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

            # Negative conditioned prediction
            if do_true_cfg:
                with self.pipeline.transformer.cache_context("uncond"):
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        encoder_hidden_states=negative_prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=negative_txt_seq_lens,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

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

        # 7. Post-process results
        generated_images = self.decode_latents(latents, height, width)

        # Transpose `extra_call_back_res` tensors to have batch dimension first
        # (T, B, ...) -> (B, T, ...)
        extra_call_back_res = {
            k: torch.stack(v, dim=1)
            if isinstance(v[0], torch.Tensor) else v
            for k, v in extra_call_back_res.items()
        }

        samples = [
            QwenImageEditPlusSample(
                # Denoising trajectory
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_prob else None,
                
                # Generated image
                image=generated_images[b],

                # Condition images
                image_latents=image_latents[b] if image_latents is not None else None,
                condition_images=condition_images if condition_images is not None else None,
                img_shapes=img_shapes[b],

                # Prompt
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b] if prompt_embeds is not None else None,
                prompt_embeds_mask=prompt_embeds_mask[b] if prompt_embeds_mask is not None else None,

                # Negative Prompt
                negative_prompt=negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt,
                negative_prompt_ids=negative_prompt_ids[b] if negative_prompt_ids is not None else None,
                negative_prompt_embeds=negative_prompt_embeds[b] if negative_prompt_embeds is not None else None,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask[b] if negative_prompt_embeds_mask is not None else None,

                # Extra kwargs
                extra_kwargs={
                    'guidance_scale': true_cfg_scale,
                    'attention_kwargs': attention_kwargs,
                }
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()

        return samples

    @torch.no_grad()
    def inference(
        self,
        # Ordinary arguments
        images: Optional[List[QwenImageEditPlusImageInput]] = None,
        prompt: Optional[List[str]] = None,
        negative_prompt: Optional[List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0, # Corresponds to `true_cfg_scale` in Qwen-Image-Edit-Plus-Pipeline.
        height: int = 1024,
        width: int = 1024,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # Prompt encoding arguments
        prompt_ids: Optional[Union[List[torch.LongTensor], torch.LongTensor]] = None,
        prompt_embeds: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        prompt_embeds_mask: Optional[Union[List[torch.LongTensor], torch.LongTensor]] = None,
        negative_prompt_ids: Optional[Union[List[torch.LongTensor], torch.LongTensor]] = None,
        negative_prompt_embeds: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        negative_prompt_embeds_mask: Optional[Union[List[torch.LongTensor], torch.LongTensor]] = None,        
        # Image encoding arguments
        condition_images: Optional[List[QwenImageEditPlusImageInput]] = None, # A batch of condition image lists
        condition_image_sizes: Optional[List[List[Tuple[int, int]]]] = None, # A batch of condition image size lists
        vae_images: Optional[List[QwenImageEditPlusImageInput]] = None, # A batch of VAE image lists
        vae_image_sizes: Optional[List[List[Tuple[int, int]]]] = None, # A batch of VAE image size lists
        image_latents: Optional[List[torch.Tensor]] = None, # A batch of image latents
        # Other arguments
        extra_call_back_kwargs: List[str] = [],
        attention_kwargs: Optional[Dict[str, Any]] = {},
        max_sequence_length: int = 1024,
        compute_log_prob: bool = False,
    ):
        """
        Batch inference, the input must be in the batch format
        """
        if not self._has_warned_inference_fallback:
            logger.warning(
                "Qwen-Image-Edit-Plus does not support batch inference with varying condition images per sample. "
                "Falling back to single-sample inference. This warning will only appear once."
            )
            self._has_warned_inference_fallback = True
        # Process each sample individually by calling _inference
        all_samples = []

        batch_size = (
            len(images) if images is not None else
            len(prompt) if prompt is not None else
            len(prompt_ids) if prompt_ids is not None else 
            len(prompt_embeds) if prompt_embeds is not None else
            len(condition_images) if condition_images is not None else
            len(vae_images) if vae_images is not None else 1
        )

        for b in range(batch_size):
            sample = self._inference(
                # Extract b-th element from each parameter - be careful to tensors shape.
                # Ordinary Args
                images=images[b],
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                negative_prompt=negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                attention_kwargs=attention_kwargs,
                max_sequence_length=max_sequence_length,
                # Encoded prompt
                prompt_ids=prompt_ids[b:b+1] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b:b+1] if prompt_embeds is not None else None,
                prompt_embeds_mask=prompt_embeds_mask[b:b+1] if prompt_embeds_mask is not None else None,
                # Encoded negative prompt
                negative_prompt_ids=negative_prompt_ids[b:b+1] if negative_prompt_ids is not None else None,
                negative_prompt_embeds=negative_prompt_embeds[b:b+1] if negative_prompt_embeds is not None else None,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask[b:b+1] if negative_prompt_embeds_mask is not None else None,
                # Encoded images
                condition_images=condition_images[b] if condition_images is not None else None,
                condition_image_sizes=condition_image_sizes[b] if condition_image_sizes is not None else None,
                vae_images=vae_images[b] if vae_images is not None else None,
                vae_image_sizes=vae_image_sizes[b] if vae_image_sizes is not None else None,
                image_latents=image_latents[b] if image_latents is not None else None,
                # Shared parameters
                compute_log_prob=compute_log_prob,
                extra_call_back_kwargs=extra_call_back_kwargs,
            )
            all_samples.extend(sample)

        return all_samples

    # ======================== Forward for training ========================

    def _t2i_forward(
        self,
        samples: Union[QwenImageEditPlusSample, List[QwenImageEditPlusSample]],
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> SDESchedulerOutput:
        """Forward method for text-to-image generation with Qwen-Image-Edit Plus model."""
        """
            TODO in the future if needed.
        """
        pass

    def _i2i_forward(
        self,
        sample: QwenImageEditPlusSample,
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> SDESchedulerOutput:
        """Forward method for image-to-image editing with Qwen-Image-Edit Plus model."""
        # 1. Extract data from samples
        batch_size = 1
        device = self.device
        dtype = self.pipeline.transformer.dtype
        # Assume all samples have the same guidance scale
        true_cfg_scale = sample.extra_kwargs.get('guidance_scale', self.training_args.guidance_scale)
        # Assume all samples have the same attention kwargs
        attention_kwargs = sample.extra_kwargs.get('attention_kwargs', {})

        latents = sample.all_latents[timestep_index].unsqueeze(0).to(device)
        next_latents = sample.all_latents[timestep_index + 1].unsqueeze(0).to(device)
        timestep = sample.timesteps[timestep_index].unsqueeze(0).to(device)
        t = timestep[0]
        num_inference_steps = len(sample.timesteps)

        prompt_embeds = [sample.prompt_embeds]
        prompt_embeds_mask = [sample.prompt_embeds_mask]
        image_latents = sample.image_latents.unsqueeze(0) if sample.image_latents is not None else None

        negative_prompt_embeds = sample.negative_prompt_embeds
        negative_prompt_embeds_mask = sample.negative_prompt_embeds_mask

        # Assume all samples have negative prompt embeds if any sample has it
        has_neg_prompt = (
            negative_prompt_embeds is not None
            and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1.0 and has_neg_prompt

        if do_true_cfg:
            negative_prompt_embeds = [sample.negative_prompt_embeds]
            negative_prompt_embeds_mask = [sample.negative_prompt_embeds_mask]
    
        img_shapes = [sample.img_shapes]

        # Get txt_seq_lens
        txt_seq_lens, _, prompt_embeds, prompt_embeds_mask = self._pad_batch_prompt(
            prompt_embeds_mask=prompt_embeds_mask,
            prompt_ids=None, # prompt_ids are not needed for forward
            prompt_embeds=prompt_embeds,
            device=device
        )
        
        if do_true_cfg:
            negative_txt_seq_lens, _, negative_prompt_embeds, negative_prompt_embeds_mask = self._pad_batch_prompt(
                prompt_embeds_mask=negative_prompt_embeds_mask,
                prompt_ids=None,
                prompt_embeds=negative_prompt_embeds,
                device=device
            )

        guidance = None # Always None for Qwen-Image, use `true_cfg_scale` instead

        # 2. Set scheduler timesteps
        _ = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device
        )

        # 3. Compute model output
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1) # Concatenate along sequence dimension

        # Conditioned prediction
        with self.pipeline.transformer.cache_context("cond"):
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]

        # Negative conditioned prediction
        if do_true_cfg:
            with self.pipeline.transformer.cache_context("uncond"):
                neg_noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=negative_txt_seq_lens,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

        # Scheduler step
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
        samples: List[QwenImageEditPlusSample],
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> SDESchedulerOutput:
        is_i2i = any(
            s.image_latents is not None
            for s in samples
        )
        if is_i2i:
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
            # It is weird that a Edit model can inference without condition images, but we allow it for flexibility.
            output = self._t2i_forward(
                samples=samples,
                timestep_index=timestep_index,
                compute_log_prob=compute_log_prob,
                **kwargs,
            )

        return output