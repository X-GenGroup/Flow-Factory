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

# src/flow_factory/models/flux/flux1_kontext.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple, Literal, ClassVar
from dataclasses import dataclass
import logging
from collections import defaultdict
from PIL import Image
import numpy as np

from accelerate import Accelerator
import torch
from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
from diffusers.utils.torch_utils import randn_tensor

from ..abc import BaseAdapter
from ..samples import I2ISample
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
    TrajectoryIndicesType, 
    create_trajectory_collector,
)
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

CONDITION_IMAGE_SIZE = (1024, 1024)

@dataclass
class Flux1KontextSample(I2ISample):
    """Output class for Flux Adapter models."""
    # Class variables
    _shared_fields: ClassVar[frozenset[str]] = frozenset({'latent_ids'})
    # object varibales
    pooled_prompt_embeds : Optional[torch.FloatTensor] = None
    image_latents : Optional[torch.FloatTensor] = None
    condition_image_size: Optional[Tuple[int, int]] = None
    latent_ids : Optional[torch.Tensor] = None

def adjust_image_dimension(
        height: int,
        width: int,
        max_area: int,
        vae_scale_factor: int,
    ) -> Tuple[int, int]:
    """
    Logic of adjusting image dimensions to fit model requirements.
    """
    original_height, original_width = height, width
    original_area = height * width

    if original_area > max_area:
        # Resize if area is larger than max
        aspect_ratio = width / height
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)

    multiple_of = vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    if height != original_height or width != original_width:
        logger.warning(
            f"Generation `height` and `width` have been adjusted from ({original_height, original_width}) to ({height}, {width}) to fit the model requirements."
        )

    return height, width


class Flux1KontextAdapter(BaseAdapter):
    """Concrete implementation for Flow Matching models (FLUX.1)."""
    
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: FluxKontextPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

        self._has_warned_multi_image = False
    
    def load_pipeline(self) -> FluxKontextPipeline:
        return FluxKontextPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )

    @property
    def default_target_modules(self) -> List[str]:
        """Default Trainable target modules for FLUX.1-Kontext-dev model."""
        return [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff_context.net.0.proj", "ff.net.2", "ff_context.net.2",
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
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """Encode text prompts using the pipeline's text encoder."""

        execution_device = self.pipeline.text_encoder.device
        
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompt,
            device=execution_device,
            max_sequence_length=max_sequence_length,
        )
        
        prompt_ids = self.pipeline.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(execution_device)
                
        return {
            'prompt_ids': prompt_ids,
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
        }

    def _standardize_image_input(
        self,
        images: Union[ImageSingle, ImageBatch, MultiImageBatch],
        output_type: Literal['pil', 'np', 'pt'] = 'pil',
    ):
        """
        Standardize image input to desired output type.
        """
        if isinstance(images, Image.Image):
            images = [images]
        elif is_multi_image_batch(images):
            images = [batch[0] for batch in images]
            # A list of list of images
            if not self._has_warned_multi_image and any(len(batch) > 1 for batch in images):
                self._has_warned_multi_image = True
                logger.warning(
                    "Multiple condition images are not supported for Flux1-Kontext-dev. Only the first image of each batch will be used."
                )
        
        images = standardize_image_batch(
            images,
            output_type=output_type,
        )
        
        return images

    def encode_image(
        self,
        images: Union[ImageSingle, ImageBatch],
        condition_image_size : Optional[Union[int, Tuple[int, int]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode input images into latent representations using the VAE encoder.
        Args:
            images: Single condition image or a batch of images (PIL.Image).
            condition_image_size: Desired size for condition images (int or (width, height)).
            auto_resize: Whether to automatically resize images to preferred resolutions.
            generator: Optional random generator(s) for encoding.
        Returns:
            Dictionary containing resized 'condition_images', 'image_latents' and 'image_ids'.
        """
        device = self.pipeline.vae.device
        dtype = self.pipeline.vae.dtype
        images = self._standardize_image_input(
            images,
            output_type='pil',
        )
        
        if not is_image_batch(images):
            raise ValueError(f"Invalid image input type: {type(images)}. Must be a PIL Image, numpy array, torch tensor, or a list of these types.")

        batch_size = len(images)
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4

        if condition_image_size is None:
            first_image = images[0] # Use the first image to determine size
            image_height, image_width = self.pipeline.image_processor.get_default_height_width(first_image)
            aspect_ratio = image_width / image_height
            # Auto resize to preferred kontext resolution
            _, image_width, image_height = min(
                (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
            )
        elif isinstance(condition_image_size, int):
            image_height, image_width = condition_image_size, condition_image_size
        else:
            image_height, image_width = condition_image_size

        condition_max_area = image_height * image_width

        # resize to integer multiple of vae_scale_factor
        image_height, image_width = adjust_image_dimension(
            image_height,
            image_width,
            condition_max_area,
            self.pipeline.vae_scale_factor,
        )
        images = self.pipeline.image_processor.resize(images, image_height, image_width)
        image_tensors = self.pipeline.image_processor.preprocess(images, image_height, image_width)
        # 2. Prepare `image_latents` and `image_ids`
        image_tensors = image_tensors.to(device=device, dtype=dtype)
        image_latents = self.pipeline._encode_vae_image(image=image_tensors, generator=generator)
        image_latent_height, image_latent_width = image_latents.shape[2:]
        image_latents = self.pipeline._pack_latents(
            image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
        )
        image_ids = self.pipeline._prepare_latent_image_ids(
            batch_size, image_latent_height // 2, image_latent_width // 2, device, dtype
        )
        # image ids are the same as latent ids with the first dimension set to 1 instead of 0
        image_ids[..., 0] = 1

        return {
            'condition_images': self.pipeline.image_processor.postprocess(image_tensors, output_type='pt'), # convert numerical range to [0, 1]
            'image_latents': image_latents,
            'image_ids': image_ids.unsqueeze(0).expand(batch_size, *[-1] * (image_ids.ndim)),  # Expand to batch size
        }
    
    def encode_video(self, videos: Any) -> None:
        """Flux.2 does not support video encoding."""
        pass

    def decode_latents(self, latents: torch.Tensor, height, width, output_type="pil") -> List[Union[Image.Image, torch.Tensor, np.ndarray]]:
        latents = latents.to(dtype=self.pipeline.vae.dtype)
        latents = self.pipeline._unpack_latents(latents, height, width, self.pipeline.vae_scale_factor)
        latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
        image = self.pipeline.vae.decode(latents, return_dict=False)[0]
        image = self.pipeline.image_processor.postprocess(image, output_type=output_type)
        return image

    # ======================== Prepare Latents =============================
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.pipeline.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.pipeline.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        latent_ids = self.pipeline._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.pipeline._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, latent_ids

    # ======================== Inference =============================
    @torch.no_grad()
    def inference(
        self,
        # Oridinary inputs
        images: Optional[ImageBatch] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        condition_image_size : Optional[Union[int, Tuple[int, int]]] = None,
        num_inference_steps: int = 50,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.5,
        generator: Optional[torch.Generator] = None,
        max_sequence_length: int = 512,
        # Encodede prompt
        prompt_ids : Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # Encoded images
        condition_images: Optional[ImageBatch] = None,
        image_latents: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        # Extra kwargs
        joint_attention_kwargs : Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = 'all',
    ):
        # 1. Setup
        device = self.device
        # 2. Encode prompt if not encoded
        if prompt_embeds is None or pooled_prompt_embeds is None:
            encoded = self.encode_prompt(prompt=prompt, max_sequence_length=max_sequence_length)
            prompt_embeds = encoded['prompt_embeds']
            pooled_prompt_embeds = encoded['pooled_prompt_embeds']
            prompt_ids = encoded['prompt_ids']
        else:
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)

        # 3. Encode images if not encoded
        if condition_images is None or image_latents is None or image_ids is None:
            encoded_image = self.encode_image(
                images=images, 
                condition_image_size=condition_image_size,
                generator=generator,
            )
            condition_images = encoded_image['condition_images']
            image_latents = encoded_image['image_latents']
            image_ids = encoded_image['image_ids']
        else:
            # Convert to pt if needed
            condition_images = self._standardize_image_input(
                condition_images,
                output_type='pt',
            )
            image_latents = image_latents.to(device)
            image_ids = image_ids.to(device)

        if image_ids.dim() == 3:
            # Remove batch dimension if exists
            image_ids = image_ids[0]

        batch_size = len(prompt_embeds)
        dtype = prompt_embeds.dtype

        # 4. Prepare initial latents
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        latent_ids = torch.cat([latent_ids, image_ids], dim=0) # Catenate at the sequence dimension

        # 5. Set scheduler timesteps
        timesteps = set_scheduler_timesteps(
            scheduler=self.pipeline.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device,
        )

        # 6. Denoising loop
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
                image_latents=image_latents,
                latent_ids=latent_ids,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                compute_log_prob=compute_log_prob and current_noise_level > 0,
                joint_attention_kwargs=joint_attention_kwargs,
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
                        # First check in capturable dict
                        extra_call_back_res[key].append(capturable[key])
                    elif hasattr(output, key):
                        # Then check in output
                        val = getattr(output, key)
                        if val is not None:
                            extra_call_back_res[key].append(val)


        # 7. Prepare output images
        generated_images = self.decode_latents(latents, height, width)

        # 8. Create samples
        # Transpose `extra_call_back_res` tensors to have batch dimension first
        # (T, B, ...) -> (B, T, ...)
        extra_call_back_res = {
            k: torch.stack(v, dim=1)
            if isinstance(v[0], torch.Tensor) else v
            for k, v in extra_call_back_res.items()
        }
        all_latents = latent_collector.get_result()
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        samples = [
            Flux1KontextSample(
                # Denoising trajectory
                timesteps=timesteps,
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0) if all_latents is not None else None,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if all_log_probs is not None else None,
                # Generated image & metadata
                image=generated_images[b],
                height=height,
                width=width,
                latent_ids=latent_ids, # Store latent ids (after catenation, no batch dimension)
                # Prompt
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b],
                prompt_embeds=prompt_embeds[b],
                pooled_prompt_embeds=pooled_prompt_embeds[b],
                # Condition image
                image_latents=image_latents[b] if image_latents is not None else None,
                condition_images=condition_images[b] if condition_images is not None else None,
                # Extra callback results
                extra_kwargs={
                    'guidance_scale': guidance_scale,
                    **{k: v[b] for k, v in extra_call_back_res.items()}
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()
        
        return samples
    
    # =====================================  Forward =====================================

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        # Condtion image
        image_latents: torch.Tensor,
        latent_ids: torch.Tensor,
        # Prompt Info
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        guidance_scale : float = 3.5,
        noise_level: Optional[float] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = ['noise_pred', 'next_latents', 'next_latents_mean', 'std_dev_t', 'dt', 'log_prob'],
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """
        Forward pass with given timestep, next timestep, and latents.

        Args:
            t: Current timestep tensor.
            t_next: Next timestep tensor.
            latents: Current latent representations.
            image_latents: Encoded condition image latents.
            prompt_embeds: Text prompt embeddings.
            pooled_prompt_embeds: Pooled text embeddings.
            guidance: Guidance scale tensor.
            txt_ids: Text position IDs.
            latent_ids: Combined latent + image position IDs.
            next_latents: Optional target latents for log-prob computation.
            joint_attention_kwargs: Optional kwargs for joint attention.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            noise_level: Current noise level for SDE sampling.

        Returns:
            SDESchedulerOutput containing requested outputs.
        """
        # 1. Prepare variables
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]
        guidance = torch.as_tensor(guidance_scale, device=device, dtype=dtype)
        guidance = guidance.expand(batch_size) # Assume List[float] has len `batch_size`
        # Concatenate latents with condition image latents
        latent_model_input = torch.cat([latents, image_latents], dim=1)

        # 2. Transformer foward pass
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=t.expand(batch_size) / 1000,  # Scale timestep to [0, 1]
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype),
            img_ids=latent_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

        # Extract only the target latent predictions (exclude condition image part)
        noise_pred = noise_pred[:, :latents.shape[1]]

        # 3. Scheduler step
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