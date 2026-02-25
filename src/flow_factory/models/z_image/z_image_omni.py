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
from diffusers.pipelines.z_image.pipeline_z_image_omni import ZImageOmniPipeline

from ..abc import BaseAdapter
from ..samples import T2ISample, I2ISample
from ...hparams import *
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SDESchedulerOutput,
    set_scheduler_timesteps
)
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
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)

CONDITION_IMAGE_SIZE = (1024, 1024)

@dataclass
class ZImageOmniSample(I2ISample):
    # Class var
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    # Condition image encodings (per-sample list, may contain None sentinel at end)
    condition_latents: Optional[List[torch.Tensor]] = None
    negative_condition_latents: Optional[List[torch.Tensor]] = None
    condition_siglip_embeds: Optional[List[Optional[torch.Tensor]]] = None
    negative_condition_siglip_embeds: Optional[List[Optional[torch.Tensor]]] = None



class ZImageOmniAdapter(BaseAdapter):
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: ZImageOmniPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

        self._has_warned_inference_fallback = False
        self._has_warned_forward_fallback = False

    def load_pipeline(self) -> ZImageOmniPipeline:
        return ZImageOmniPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Z-Image-Omni transformer."""
        return [
            # TODO
        ]

    # ======================== Encoding / Decoding ======================== 
    # ----------------------- Prompt Encoding -----------------------   
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        num_condition_images: Union[int, List[int]] = 0,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        device = device or self.pipeline.text_encoder.device

        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)

        # Normalize num_condition_images to per-sample list
        if isinstance(num_condition_images, int):
            num_condition_images = [num_condition_images] * batch_size

        # Format prompts based on per-sample condition image count
        formatted_prompts = []
        for prompt_item, n_cond in zip(prompt, num_condition_images):
            if n_cond == 0:
                formatted_prompts.append(
                    ["<|im_start|>user\n" + prompt_item + "<|im_end|>\n<|im_start|>assistant\n"]
                )
            else:
                prompt_list = ["<|im_start|>user\n<|vision_start|>"]
                prompt_list += ["<|vision_end|><|vision_start|>"] * (n_cond - 1)
                prompt_list += ["<|vision_end|>" + prompt_item + "<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"]
                prompt_list += ["<|vision_end|><|im_end|>"]
                formatted_prompts.append(prompt_list)

        # Flatten for batch tokenization
        flattened_prompt = []
        prompt_list_lengths = []
        for p in formatted_prompts:
            prompt_list_lengths.append(len(p))
            flattened_prompt.extend(p)

        text_inputs = self.tokenizer(
            flattened_prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        # Reconstruct nested structure
        embeddings_list = []
        start_idx = 0
        for length in prompt_list_lengths:
            batch_embeddings = []
            for j in range(start_idx, start_idx + length):
                batch_embeddings.append(prompt_embeds[j][prompt_masks[j]])
            embeddings_list.append(batch_embeddings)
            start_idx += length

        return text_input_ids, embeddings_list

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        max_sequence_length: int = 512,
        num_condition_images: Union[int, List[int]] = 0,
    ) -> Dict[str, Any]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_ids, prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            max_sequence_length=max_sequence_length,
            num_condition_images=num_condition_images,
        )

        negative_prompt_ids = None
        negative_prompt_embeds = None

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = [""] * len(prompt)
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            assert len(prompt) == len(negative_prompt), "The length of `prompt` and `negative_prompt` must be the same."
            
            negative_prompt_ids, negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                num_condition_images=num_condition_images,
            )
        
        return {
            'prompt_ids': prompt_ids,
            'prompt_embeds': prompt_embeds,
            'negative_prompt_ids': negative_prompt_ids,
            'negative_prompt_embeds': negative_prompt_embeds,
        }
    
    # ----------------------- Image Encoding -----------------------
    @staticmethod
    def _is_multi_images_batch(images: Union[ImageBatch, MultiImageBatch]) -> bool:
        return is_multi_image_batch(images)

    def _standardize_image_input(
        self,
        images: Union[ImageSingle, ImageBatch],
        output_type: Literal['pil', 'pt', 'np'] = 'pil',
    ) -> ImageBatch:
        """Standardize image input to PIL format."""
        if isinstance(images, Image.Image):
            images = [images]
        return standardize_image_batch(images, output_type=output_type)

    def _preprocess_condition_images(
        self,
        images: Union[ImageSingle, ImageBatch],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
    ) -> Tuple[List[Image.Image], List[torch.Tensor]]:
        """
        Preprocess a batch of condition images.
        
        Args:
            images: Single image or list of images
            condition_image_size: Max size constraint.
                - int: 
                - Tuple[int, int]: (max_height, max_width)
        
        Returns:
            resized_images: List[PIL.Image] for siglip encoding
            image_tensors: List[torch.Tensor(1, C, H, W)] for VAE encoding
        """
        if isinstance(condition_image_size, int):
            condition_image_size = (condition_image_size, condition_image_size)

        images_pil : List[Image.Image] = self._standardize_image_input(
            images,
            output_type='pil',
        )

        max_area = condition_image_size[0] * condition_image_size[1]

        condition_images_resized = []
        condition_image_tensors = []
        for img in images_pil:
            image_width, image_height = img.size
            if image_width * image_height > max_area:
                img = self.pipeline.image_processor._resize_to_target_area(img, max_area)
                image_width, image_height = img.size

            condition_images_resized.append(img)
            multiple_of = self.pipeline.vae_scale_factor * 2
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of
            img = self.pipeline.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
            condition_image_tensors.append(img)

        return condition_images_resized, condition_image_tensors
    
    def _prepare_image_latents(
        self,
        images: List[List[torch.Tensor]],
        device : torch.device,
        dtype : torch.dtype,
    ) -> List[List[torch.Tensor]]:
        """
        Encode condition images into latent space using VAE.
        Args:
            images: List of List[torch.Tensor(1, C, H, W)] or torch.Tensor(B, C, H, W)
            device: Target device
            dtype: Target data type
        Returns:
            image_latents: List of List[torch.Tensor(latent_channels, 1, H//vae_scale, W//vae_scale)]
        """

        image_latents = []
        for cond_images in images:
            img_latents = []
            for img in cond_images:
                img = img.to(device=device, dtype=dtype)
                image_latent = (
                    self.pipeline.vae.encode(img.bfloat16()).latent_dist.mode()[0] - self.pipeline.vae.config.shift_factor
                ) * self.pipeline.vae.config.scaling_factor # (latent_channels, H//vae_scale, W//vae_scale)
                image_latent = image_latent.unsqueeze(1).to(dtype) # (latent_channels, 1, H//vae_scale, W//vae_scale)
                img_latents.append(image_latent)

            image_latents.append(img_latents)

        return image_latents # List[List[torch.Tensor(latent_channels, 1, H//vae_scale, W//vae_scale)]]
    
    def _prepare_siglip_embeds(
        self,
        images: List[List[Image.Image]],
        device : torch.device,
        dtype : torch.dtype,
    ) -> List[List[torch.Tensor]]:
        """
        Encode condition images into SigLiP embeddings.
        Args:
            images: List of List[PIL.Image]
            device: Target device
            dtype: Target data type
        Returns:
            siglip_embeds: List of List[torch.Tensor(H_patches, W_patches, hidden_dim)]
        """
        siglip_embeds = []
        for cond_images in images:
            embeds = []
            for img in cond_images:
                siglip_inputs = self.pipeline.siglip_processor(images=[img], return_tensors="pt").to(device)
                shape = siglip_inputs.spatial_shapes[0]
                hidden_state = self.pipeline.siglip(**siglip_inputs).last_hidden_state
                B, N, C = hidden_state.shape # (1, num_patches, hidden_dim)
                hidden_state = hidden_state[:, : shape[0] * shape[1]]
                hidden_state = hidden_state.view(shape[0], shape[1], C) # (H_patches, W_patches, hidden_dim)
                embeds.append(hidden_state.to(dtype))

            siglip_embeds.append(embeds)

        return siglip_embeds # List[List[torch.Tensor(H_patches, W_patches, hidden_dim)]]

    def encode_image(
        self,
        images: Union[ImageBatch, MultiImageBatch],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        device : Optional[torch.device] = None,
        dtype : Optional[torch.dtype] = None,
        do_classifier_free_guidance: bool = True,
    ) -> Dict[str, List[List[torch.Tensor]]]:
        """Encode condition images for Z-Image-Omni model."""
        device = device or self.pipeline.device
        dtype = dtype or self.pipeline.text_encoder.dtype
        images = [images] if not self._is_multi_images_batch(images) else images
        condition_image_resized = []
        condition_image_tensors = []
        for img_batch in images:
            condition_image_resized_batch, condition_image_tensors_batch = self._preprocess_condition_images(
                images=img_batch,
                condition_image_size=condition_image_size,
            )
            condition_image_resized.append(condition_image_resized_batch)
            condition_image_tensors.append(condition_image_tensors_batch)

        image_latents = self._prepare_image_latents(
            images=condition_image_tensors,
            device=device,
            dtype=dtype,
        )
        siglip_embeds = self._prepare_siglip_embeds(
            images=condition_image_resized,
            device=device,
            dtype=dtype,
        )

        # Convert back to [0, 1] range tensors for storage
        condition_image_tensors: List[List[torch.Tensor]] = [
            [
                self.pipeline.image_processor.postprocess(img, output_type='pt')[0]
                for img in cond_img_tensors
            ]
            for cond_img_tensors in condition_image_tensors
        ]

        res = {
            'condition_images': condition_image_tensors, # List[List[torch.Tensor(C, H, W)]]
            'condition_latents': image_latents, # List[List[torch.Tensor(latent_channels, 1, H//vae_scale, W//vae_scale)]]
            'condition_siglip_embeds': siglip_embeds, # List[List[torch.Tensor(H_patches, W_patches, hidden_dim)]]
        }

        if do_classifier_free_guidance:
            # Duplicate for negative guidance
            res.update({
                'negative_condition_latents': [[lat.clone() for lat in batch] for batch in res['condition_latents']],
                'negative_condition_siglip_embeds': [[se.clone() for se in batch] for batch in res['condition_siglip_embeds']]
            })
        
        return res
    
    # ----------------------- Video Encoding -----------------------
    def encode_video(
        self,
        videos: Union[torch.Tensor, List[torch.Tensor]],
    ):
        """Not needed for Z-Image-Omni models."""
        pass

    # ----------------------- Decoding -----------------------
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
    
    # ======================== Preprocessing ========================
    def preprocess_func(
        self,
        prompt: List[str],
        images: Optional[MultiImageBatch] = None,
        negative_prompt: Optional[List[str]] = None,
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        max_sequence_length: int = 512,
        do_classifier_free_guidance: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """
        Preprocess inputs for Flux.2 model (batched processing).
        
        Args:
            prompt: List of text prompts
            images: Optional images in various formats (MultiImageBatch)
            condition_image_size: Size constraint for condition images
            max_sequence_length: Max token length for text encoding
            do_classifier_free_guidance: Whether to prepare negative prompts
        
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

        batch = {}
        # 2. Batch encode prompts
        num_condition_images = [
            len(imgs) for imgs in images
        ] if has_images else 0
        prompt_dict = self.encode_prompt(
            prompt=prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
            num_condition_images=num_condition_images,
        )
        batch.update(prompt_dict)
        
        # 3. Encode condition images if provided
        if has_images:
            image_dict = self.encode_image(
                images=images,
                condition_image_size=condition_image_size,
                device=device,
                dtype=dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            batch.update(image_dict)
        
        return batch
    
    # ======================== Inference ========================
    @torch.no_grad()
    def inference(
        self,
        # Generation parameters
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        height: int = 1024,
        width: int = 1024,
        # Conditioning inputs (raw)
        images: Optional[Union[ImageBatch, MultiImageBatch]] = None,
        # Prompt
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[List[List[torch.Tensor]]] = None,
        # Negative prompt
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[List[List[torch.Tensor]]] = None,
        # Encoded condition images
        condition_images: Optional[List[List[torch.Tensor]]] = None,
        condition_latents: Optional[List[List[torch.Tensor]]] = None,
        condition_siglip_embeds: Optional[List[List[Optional[torch.Tensor]]]] = None,
        negative_condition_latents: Optional[List[List[torch.Tensor]]] = None,
        negative_condition_siglip_embeds: Optional[List[List[Optional[torch.Tensor]]]] = None,
        # CFG options
        cfg_normalization: bool = False,
        cfg_truncation: Optional[float] = 1.0,
        # Other parameters
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        max_sequence_length: int = 512,
        compute_log_prob: bool = True,
        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = 'all',
    ) -> List[ZImageOmniSample]:
        """
        Generate images using Z-Image-Omni model with optional condition images.

        Args:
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale (0 disables CFG).
            height: Output image height.
            width: Output image width.
            images: Raw condition images (will be encoded if latents not provided).
            prompt: Text prompt(s) for generation.
            prompt_ids: Pre-tokenized prompt IDs.
            prompt_embeds: Pre-encoded prompt embeddings.
            negative_prompt: Negative prompt(s) for CFG.
            negative_prompt_ids: Pre-tokenized negative prompt IDs.
            negative_prompt_embeds: Pre-encoded negative prompt embeddings.
            condition_images: Preprocessed condition image tensors.
            condition_latents: Pre-encoded condition image latents.
            condition_siglip_embeds: Pre-encoded SigLIP embeddings.
            negative_condition_latents: Negative condition latents for CFG.
            negative_condition_siglip_embeds: Negative SigLIP embeddings for CFG.
            cfg_normalization: Whether to apply CFG normalization.
            cfg_truncation: Disable CFG when t_norm > threshold.
            condition_image_size: Max size for condition images.
            generator: Random generator for reproducibility.
            max_sequence_length: Max token length for prompts.
            compute_log_prob: Whether to compute log probabilities.
            extra_call_back_kwargs: Additional outputs to capture per step.
            trajectory_indices: Which trajectory steps to collect.

        Returns:
            List of ZImageOmniSample objects containing generated images and metadata.
        """
        # 1. Setup device and dtype
        device = self.device
        dtype = self.transformer.dtype
        do_classifier_free_guidance = guidance_scale > 0.0

        # Adjust dimensions to be multiples of required factor
        multiple_of = self.pipeline.vae_scale_factor * 2
        calc_height = (height // multiple_of) * multiple_of
        calc_width = (width // multiple_of) * multiple_of
        if calc_height != height or calc_width != width:
            logger.warning(
                f"Adjusting dimensions to multiples of {multiple_of}: "
                f"({height}, {width}) -> ({calc_height}, {calc_width})"
            )
            height, width = calc_height, calc_width

        # Normalize prompt inputs to list format
        if isinstance(prompt, str):
            prompt = [prompt]
        if negative_prompt is not None and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        # 2. Determine what needs encoding
        is_prompt_encoded = prompt_embeds is not None
        is_negative_encoded = negative_prompt_embeds is not None
        has_condition_images = images is not None or condition_latents is not None
        is_condition_encoded = (
            condition_latents is not None and condition_siglip_embeds is not None
        )

        # 3. Preprocess if needed
        needs_preprocessing = not (
            is_prompt_encoded
            and (not do_classifier_free_guidance or is_negative_encoded)
            and (not has_condition_images or is_condition_encoded)
        )

        if needs_preprocessing:
            encoded = self.preprocess_func(
                prompt=prompt,
                images=images,
                negative_prompt=negative_prompt,
                condition_image_size=condition_image_size,
                max_sequence_length=max_sequence_length,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=device,
                dtype=dtype,
            )
            prompt_ids = encoded['prompt_ids']
            prompt_embeds = encoded['prompt_embeds']
            if do_classifier_free_guidance:
                negative_prompt_ids = encoded.get('negative_prompt_ids')
                negative_prompt_embeds = encoded.get('negative_prompt_embeds')
            if 'condition_latents' in encoded:
                condition_images = encoded['condition_images']
                condition_latents = encoded['condition_latents']
                condition_siglip_embeds = encoded['condition_siglip_embeds']
                if do_classifier_free_guidance:
                    negative_condition_latents = encoded.get('negative_condition_latents')
                    negative_condition_siglip_embeds = encoded.get('negative_condition_siglip_embeds')
        else:
            # Move pre-encoded inputs to correct device/dtype
            prompt_embeds = [
                [pe.to(device=device, dtype=dtype) for pe in batch]
                for batch in prompt_embeds
            ]
            if do_classifier_free_guidance and negative_prompt_embeds is not None:
                negative_prompt_embeds = [
                    [npe.to(device=device, dtype=dtype) for npe in batch]
                    for batch in negative_prompt_embeds
                ]
            if condition_latents is not None:
                condition_latents = [
                    [cl.to(device=device, dtype=dtype) for cl in batch]
                    for batch in condition_latents
                ]
                condition_siglip_embeds = [
                    [cse.to(device=device, dtype=dtype) for cse in batch]
                    for batch in condition_siglip_embeds
                ]
                if do_classifier_free_guidance:
                    if negative_condition_latents is not None:
                        negative_condition_latents = [
                            [ncl.to(device=device, dtype=dtype) for ncl in batch]
                            for batch in negative_condition_latents
                        ]
                    if negative_condition_siglip_embeds is not None:
                        negative_condition_siglip_embeds = [
                            [ncse.to(device=device, dtype=dtype) for ncse in batch]
                            for batch in negative_condition_siglip_embeds
                        ]

        batch_size = len(prompt_embeds)

        # 4. Initialize condition inputs if not provided
        if condition_latents is None:
            condition_latents = [[] for _ in range(batch_size)]
            condition_siglip_embeds = [None for _ in range(batch_size)]
            if do_classifier_free_guidance:
                negative_condition_latents = [[] for _ in range(batch_size)]
                negative_condition_siglip_embeds = [None for _ in range(batch_size)]
        else:
            # Append None sentinel to siglip embeds (required by transformer)
            condition_siglip_embeds = [
                None if sels == [] else sels + [None]
                for sels in condition_siglip_embeds
            ]
            if do_classifier_free_guidance and negative_condition_siglip_embeds is not None:
                negative_condition_siglip_embeds = [
                    None if sels == [] else sels + [None]
                    for sels in negative_condition_siglip_embeds
                ]

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        # 6. Set scheduler timesteps
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        timesteps = set_scheduler_timesteps(
            self.scheduler,
            num_inference_steps,
            seq_len=image_seq_len,
            device=device,
        )

        # 7. Initialize collectors
        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        # 8. Denoising loop
        for i, t in enumerate(timesteps):
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kwargs = list(set(['next_latents', 'log_prob', 'noise_pred'] + extra_call_back_kwargs))
            current_compute_log_prob = compute_log_prob and current_noise_level > 0
            
            output = self.forward(
                t=t,
                latents=latents,
                prompt_embeds=prompt_embeds,
                condition_latents=condition_latents,
                condition_siglip_embeds=condition_siglip_embeds,
                negative_prompt_embeds=negative_prompt_embeds if do_classifier_free_guidance else None,
                negative_condition_latents=negative_condition_latents if do_classifier_free_guidance else None,
                negative_condition_siglip_embeds=negative_condition_siglip_embeds if do_classifier_free_guidance else None,
                guidance_scale=guidance_scale,
                cfg_normalization=cfg_normalization,
                cfg_truncation=cfg_truncation,
                t_next=t_next,
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

        # 9. Decode latents to images
        images_out = self.decode_latents(latents, output_type='pt')

        # 10. Create output samples
        extra_call_back_res = callback_collector.get_result()          # (B, len(trajectory_indices), ...)
        callback_index_map = callback_collector.get_index_map()        # (T,) LongTensor
        all_latents = latent_collector.get_result()                    # List[torch.Tensor(B, ...)]
        latent_index_map = latent_collector.get_index_map()            # (T+1,) LongTensor
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
        samples = [
            ZImageOmniSample(
                # Denoising trajectory
                timesteps=timesteps,
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0) if all_latents is not None else None,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if all_log_probs is not None else None,
                latent_index_map=latent_index_map,
                log_prob_index_map=log_prob_index_map,
                # Generated image & metadata
                height=height,
                width=width,
                image=images_out[b],
                # Encoded prompt
                prompt=prompt[b] if prompt is not None else None,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b] if prompt_embeds is not None else None,
                # Encoded negative prompt
                negative_prompt=(
                    negative_prompt[b] if negative_prompt is not None else None
                ),
                negative_prompt_ids=(
                    negative_prompt_ids[b] if negative_prompt_ids is not None else None
                ),
                negative_prompt_embeds=(
                    negative_prompt_embeds[b] if negative_prompt_embeds is not None else None
                ),
                # Condition image encodings
                condition_images=(
                    condition_images[b] if condition_images and condition_images[b] else None
                ),
                condition_latents=(
                    condition_latents[b] if condition_latents and condition_latents[b] else None
                ),
                negative_condition_latents=(
                    negative_condition_latents[b] if negative_condition_latents and negative_condition_latents[b] else None
                ),
                condition_siglip_embeds=(
                    condition_siglip_embeds[b] if condition_siglip_embeds and condition_siglip_embeds[b] else None
                ),
                negative_condition_siglip_embeds=(
                    negative_condition_siglip_embeds[b] if negative_condition_siglip_embeds and negative_condition_siglip_embeds[b] else None
                ),
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
        prompt_embeds: List[List[torch.Tensor]],
        # Condition image inputs
        condition_latents: Optional[List[List[torch.Tensor]]] = None,
        condition_siglip_embeds: Optional[List[Optional[List[torch.Tensor]]]] = None,
        # Optional for CFG
        negative_prompt_embeds: Optional[List[List[torch.Tensor]]] = None,
        negative_condition_latents: Optional[List[List[torch.Tensor]]] = None,
        negative_condition_siglip_embeds: Optional[List[Optional[List[torch.Tensor]]]] = None,
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
        Forward pass for Z-Image-Omni with condition image support.

        Args:
            t: Current timestep tensor.
            latents: Current latent representations (B, C, H, W).
            prompt_embeds: Nested list of text embeddings [batch][segment].
            condition_latents: Condition image latents [batch][num_cond_images].
            condition_siglip_embeds: SigLIP embeddings [batch][num_cond_images + 1(None)].
            negative_prompt_embeds: Negative text embeddings for CFG.
            negative_condition_latents: Negative condition latents for CFG.
            negative_condition_siglip_embeds: Negative SigLIP embeddings for CFG.
            guidance_scale: CFG scale factor.
            cfg_normalization: Whether to apply CFG normalization.
            cfg_truncation: CFG truncation threshold (disable CFG when t_norm > threshold).
            t_next: Next timestep tensor for SDE sampling.
            next_latents: Target latents for log-prob computation.
            noise_level: Current noise level for SDE sampling.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.

        Returns:
            FlowMatchEulerDiscreteSDESchedulerOutput with requested outputs.
        """
        # 1. Prepare variables
        device = latents.device
        dtype = self.transformer.dtype
        batch_size = latents.shape[0]

        # Initialize empty condition inputs if not provided
        if condition_latents is None:
            condition_latents = [[] for _ in range(batch_size)]
        if condition_siglip_embeds is None:
            condition_siglip_embeds = [None for _ in range(batch_size)]

        # Z-Image uses reversed timesteps: t_reversed = (1000 - t) / 1000
        timestep = t.expand(batch_size).to(latents.dtype)
        t_reversed = (1000 - timestep) / 1000
        t_norm = t_reversed[0].item()

        # Auto-detect CFG
        apply_cfg = (
            negative_prompt_embeds is not None
            and guidance_scale > 0.0
        )

        # 2. Determine if CFG should be applied at this timestep
        current_guidance_scale = guidance_scale
        if apply_cfg and cfg_truncation is not None and cfg_truncation <= 1.0:
            if t_norm > cfg_truncation:
                current_guidance_scale = 0.0

        apply_cfg = apply_cfg and current_guidance_scale > 0

        # 3. Prepare model inputs
        if apply_cfg:
            # Initialize negative conditions if not provided
            if negative_condition_latents is None:
                negative_condition_latents = [[cl.clone() for cl in batch] for batch in condition_latents]
            if negative_condition_siglip_embeds is None:
                negative_condition_siglip_embeds = [
                    [se.clone() for se in batch] if batch is not None else None
                    for batch in condition_siglip_embeds
                ]

            latents_typed = latents.to(dtype)
            latent_model_input = latents_typed.repeat(2, 1, 1, 1)
            prompt_embeds_input = prompt_embeds + negative_prompt_embeds
            condition_latents_input = condition_latents + negative_condition_latents
            condition_siglip_input = condition_siglip_embeds + negative_condition_siglip_embeds
            timestep_input = t_reversed.repeat(2)
        else:
            latent_model_input = latents.to(dtype)
            prompt_embeds_input = prompt_embeds
            condition_latents_input = condition_latents
            condition_siglip_input = condition_siglip_embeds
            timestep_input = t_reversed

        # 4. Prepare combined latent input with noise mask
        latent_model_input = latent_model_input.unsqueeze(2)  # (B, C, 1, H, W)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        current_batch_size = len(latent_model_input_list)
        # Combine: [cond_lat_1, cond_lat_2, ..., target_lat]
        x_combined = [
            condition_latents_input[i] + [latent_model_input_list[i]]
            for i in range(current_batch_size)
        ]
        # Noise mask: 0 for condition (clean), 1 for target (noisy)
        image_noise_mask = [
            [0] * len(condition_latents_input[i]) + [1]
            for i in range(current_batch_size)
        ]

        # 5. Transformer forward pass
        model_out_list = self.transformer(
            x=x_combined,
            t=timestep_input,
            cap_feats=prompt_embeds_input,
            siglip_feats=condition_siglip_input,
            image_noise_mask=image_noise_mask,
            return_dict=False,
        )[0]

        # 6. Apply CFG
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
                    ori_norm = torch.linalg.vector_norm(pos)
                    new_norm = torch.linalg.vector_norm(pred)
                    max_norm = ori_norm * float(cfg_normalization)
                    if new_norm > max_norm:
                        pred = pred * (max_norm / new_norm)

                noise_pred.append(pred)

            noise_pred = torch.stack(noise_pred, dim=0)
        else:
            noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)

        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred  # Z-Image specific: negate prediction

        # 7. Scheduler step
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