# src/flow_factory/models/wan/wan2_i2v.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple, Literal
import logging
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from accelerate import Accelerator
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline, prompt_clean
from peft import PeftModel

from ..adapter import BaseAdapter
from ..samples import I2VSample
from ...hparams import *
from ...scheduler import UniPCMultistepSDESchedulerOutput, set_scheduler_timesteps, UniPCMultistepSDEScheduler
from ...utils.base import (
    filter_kwargs,
    pil_image_to_tensor,
    tensor_to_pil_image,
    tensor_list_to_pil_image,
    numpy_to_pil_image,
    numpy_list_to_pil_image,
    is_valid_image,
    is_valid_image_batch,
    is_valid_image_list,
    is_valid_image_batch_list,
    standardize_image_batch,
)
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)

WanPipelineImageInput = Union[
    Image.Image,
    np.ndarray,
    torch.Tensor,
    List[Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]

@dataclass
class WanI2VSample(I2VSample):
    video : Optional[Union[np.ndarray, torch.Tensor, List[Image.Image]]] = None
    image_embeds : Optional[torch.FloatTensor] = None
    condition_latents : Optional[torch.FloatTensor] = None
    first_frame_mask : Optional[torch.FloatTensor] = None


class Wan2_I2V_Adapter(BaseAdapter):
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
    
    def load_pipeline(self) -> WanImageToVideoPipeline:
        return WanImageToVideoPipeline.from_pretrained(
            self.model_args.model_name_or_path,
        )
    
    def load_scheduler(self) -> UniPCMultistepSDEScheduler:
        """Load and return the scheduler."""
        sde_config_keys = ['noise_level', 'train_steps', 'num_train_steps', 'seed', 'dynamics_type']
        # Check keys:
        for k in sde_config_keys:
            if not hasattr(self.training_args, k):
                logger.warning(f"Missing SDE config key '{k}' in training_args, using default value")

        sde_config = {
            k: getattr(self.training_args, k)
            for k in sde_config_keys
            if hasattr(self.training_args, k)
        }
        scheduler_config = self.pipeline.scheduler.config.__dict__.copy()
        scheduler_config.update(sde_config)
        return UniPCMultistepSDEScheduler(**scheduler_config)
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Wan transformer."""
        return [
            # --- Self Attention ---
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            
            # --- Cross Attention ---
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",

            # --- Feed Forward Network ---
            "ffn.net.0.proj", "ffn.net.2"
        ]
    
    @property
    def inference_modules(self) -> List[str]:
        """Modules taht are requires for inference and forward"""
        if self.pipeline.config.boundary_ratio is None or self.pipeline.config.boundary_ratio <= 0:
            return ['transformer', 'vae']

        if self.pipeline.config.boundary_ratio >= 1:
            return ['transformer_2', 'vae']

        return ['transformer', 'transformer_2', 'vae']

    def apply_lora(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]] = ['transformer', 'transformer_2'],
    ) -> Union[PeftModel, Dict[str, PeftModel]]:
        return super().apply_lora(target_modules=target_modules, components=components)
    
    @property
    def transformer_2(self) -> torch.nn.Module:
        return self.get_component('transformer_2')

    @transformer_2.setter
    def transformer_2(self, module: torch.nn.Module):
        self.set_prepared('transformer_2', module)

    # ======================== Encoding & Decoding ========================
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.pipeline._execution_device
        dtype = dtype or self.pipeline.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.pipeline.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        return text_input_ids, prompt_embeds
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self.pipeline._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_ids, prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        results = {
            'prompt_ids': prompt_ids,
            'prompt_embeds': prompt_embeds,
        }

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_ids, negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            results.update({
                "negative_prompt_ids": negative_prompt_ids,
                "negative_prompt_embeds": negative_prompt_embeds
            })

        return results
    
    def encode_image(
        self,
        images: WanPipelineImageInput, # A batch of images or a single image
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(images, Image.Image):
            images = [images]
        
        if not is_valid_image_batch(images):
            raise ValueError(f'Invalid image input for encoding: {type(images)}. Expected a single image or a batch of images.')
        
        input_images = self._standardize_image_input(images, output_type='pt')

        device = device or self.device
        images = self.pipeline.image_processor(images=input_images, return_tensors="pt").to(device)
        image_embeds = self.pipeline.image_encoder(**images, output_hidden_states=True)
        return {
            'image_embeds': image_embeds.hidden_states[-2],
            'condition_images': input_images, # Keep the standardized input images, numerically normalized to [0, 1]
        }
    
    def _standardize_image_input(
        self,
        images: WanPipelineImageInput,
        output_type: Literal['pil', 'pt', 'np'] = 'pil',
    ):
        """
        Standardize image input to desired output type.
        """
        if isinstance(images, Image.Image):
            images = [images]
        elif isinstance(images, list) and all(isinstance(batch, list) for batch in images):
            # A list of list of images
            if any(len(batch) > 1 for batch in images):
                logger.warning(
                    "Multiple condition images are not supported for Wan2_I2V. Only the first image of each batch will be used."
                )
            
            images = [batch[0] for batch in images]

        images = standardize_image_batch(
            images,
            output_type=output_type
        )
        return images

    def encode_video(
        self,
        video: Union[np.ndarray, torch.Tensor, List[Image.Image]],
        **kwargs
    ):
        pass

    # ========================Preprocessing ========================
    def preprocess_func(
        self,
        prompt: List[str],
        images: Union[List[WanPipelineImageInput], torch.Tensor, np.ndarray],
        **kwargs
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        r"""
        Preprocess the batched inputs for the model.

        Args:
            prompt (`List[str]`):
                The prompt to guide the image generation.
            images (`List[WanPipelineImageInput]`):
                The image or images to be used as conditioning input.
        """
        device = self.device
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        assert len(images) == batch_size, f"Number of images ({len(images)}) must match batch size ({batch_size})"

        # 1. Encode prompt
        prompt_encode_kwargs = filter_kwargs(self.encode_prompt, **kwargs)
        prompt_encode_kwargs.update({
            'prompt': prompt,
            'device': device,
        })
        prompt_results = self.encode_prompt(**prompt_encode_kwargs)

        # 2. Encode images - process each conditioning image(s) input separately
        images_results = defaultdict(list)
        for img in images:
            image_encode_kwargs = filter_kwargs(self.encode_image, **kwargs)
            image_encode_kwargs.update({
                'images': img,
                'device': device,
            })
            img_result = self.encode_image(**image_encode_kwargs)
            for k, v in img_result.items():
                images_results[k].append(v)

        # 3. Combine results
        return {**prompt_results, **images_results}


    def decode_latents(self, latents: torch.Tensor, output_type: Literal['pt', 'pil', 'np'] = 'pil', **kwargs) -> torch.Tensor:
        """Decode the latents using the VAE decoder."""
        latents = latents.float()
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(1, self.pipeline.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.pipeline.vae.decode(latents, return_dict=False)[0]

        video = self.pipeline.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    # ======================== Inference ========================
    def inference(
        self,
        # Oridinary arguments
        images: WanPipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # Encoded Prompt
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # Encoded Image
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None, # Not supported yet
        # Other args
        compute_log_prob: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
    ) -> List[WanI2VSample]:
        # 1. Setup args
        device = self.device
        do_classifier_free_guidance = guidance_scale > 1.0

        if self.pipeline.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        if (num_frames - 1) % self.pipeline.vae_scale_factor_temporal != 0:
            logger.warning(f"`num_frames - 1` has to be divisible by {self.pipeline.vae_scale_factor_temporal}. Rounding to the nearest number.")
            num_frames = num_frames // self.pipeline.vae_scale_factor_temporal * self.pipeline.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        images = self._standardize_image_input(images, output_type='pt')

        # 2. Encode prompt
        if prompt_embeds is None or negative_prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_ids = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
            negative_prompt_ids = encoded.get("negative_prompt_ids", None)
            negative_prompt_embeds = encoded.get("negative_prompt_embeds", None)
        else:
            prompt_embeds = prompt_embeds.to(device)
            negative_prompt_embeds = negative_prompt_embeds.to(device)

        batch_size = prompt_embeds.shape[0]
        transformer_dtype = self.pipeline.transformer.dtype if self.pipeline.transformer is not None else self.pipeline.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 3. Set scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Encode image
        # Only wan 2.1 i2v transformer accepts image_embeds
        if self.pipeline.transformer is not None and self.pipeline.transformer.config.image_dim is not None:
            if image_embeds is None:
                image_to_encode = images if last_image is None else [images, last_image]
                image_encoded = self.encode_image(image_to_encode, device)
                image_embeds = image_encoded['image_embeds']
                # condition_images = image_encoded['condition_images']

            image_embeds = image_embeds.to(transformer_dtype)

        # 5. Prepare latent variables
        num_channels_latents = self.pipeline.vae.config.z_dim
        images = self.pipeline.video_processor.preprocess(images, height=height, width=width).to(device, dtype=torch.float32)
        if last_image is not None:
            last_image = self.pipeline.video_processor.preprocess(last_image, height=height, width=width).to(
                device, dtype=torch.float32
            )

        # Inside the following function, preparing `latents_condition` requires `latents_mean` and `latents_std`,
        # which depend on `latents` initialized at runtime. Therefore, this part is kept inside inference function and not moved to preprocess_func.
        latents_outputs = self.pipeline.prepare_latents(
            image=images,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=None,
            last_image=last_image,
        )
        if self.config.expand_timesteps:
            # wan 2.2 5b i2v use firt_frame_mask to mask timesteps
            latents, condition, first_frame_mask = latents_outputs
        else:
            latents, condition = latents_outputs
            first_frame_mask = None

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.pipeline.config.boundary_ratio is not None:
            boundary_timestep = self.pipeline.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None
        
        all_latents = [latents]
        all_log_probs = [] if compute_log_prob else None
        extra_call_back_res = defaultdict(list)

        for i, t in enumerate(timesteps):

            self.pipeline._current_timestep = t
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)

            if boundary_timestep is None or t >= boundary_timestep:
                # wan2.1 or high-noise stage in wan2.2
                current_pipeline_model = self.pipeline.transformer
                current_model = self.transformer
                current_guidance_scale = guidance_scale
            else:
                # low-noise stage in wan2.2
                current_pipeline_model = self.pipeline.transformer_2
                current_model = self.transformer_2
                current_guidance_scale = guidance_scale_2

            if self.pipeline.config.expand_timesteps:
                latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                latent_model_input = latent_model_input.to(transformer_dtype)

                # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
                temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                # batch_size, seq_len
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

            with current_pipeline_model.cache_context("cond"):
                noise_pred = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

            if do_classifier_free_guidance:
                with current_pipeline_model.cache_context("uncond"):
                    noise_uncond = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            output = self.scheduler.step(
                noise_pred=noise_pred,
                timestep=t,
                latents=latents,
                compute_log_prob=compute_log_prob and current_noise_level > 0,
            )
            latents = output.next_latents
            all_latents.append(latents)
            
            if compute_log_prob:
                all_log_probs.append(output.log_prob)

            # call extra callbacks
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


        self.pipeline._current_timestep = None

        # 7. Decode latents to videos (list of pil images)
        if self.pipeline.config.expand_timesteps:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
        decoded_videos = self.decode_latents(latents, output_type='pil')

        # 8. Prepare output samples

        # Transpose `extra_call_back_res` lists to have batch dimension first
        # (T, B, ...) -> (B, T, ...)
        extra_call_back_res = {
            k: torch.stack(v, dim=1)
            if isinstance(v[0], torch.Tensor) else v
            for k, v in extra_call_back_res.items()
        }

        samples = [
            WanI2VSample(
                # Denoising trajectory
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_prob else None,

                # Generated video & metadata
                video=decoded_videos[b],
                height=height,
                width=width,

                # Conditions
                condition_images=images[b],
                condition_latents=condition[b],
                first_frame_mask=first_frame_mask, # Possibly None
                image_embeds=image_embeds[b] if image_embeds is not None else None,

                # Prompt info
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b],
                prompt_embeds=prompt_embeds[b],

                # Negative prompt info
                negative_prompt=negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt,
                negative_prompt_ids=negative_prompt_ids[b] if negative_prompt_ids is not None else None,
                negative_prompt_embeds=negative_prompt_embeds[b] if negative_prompt_embeds is not None else None,

                # Extra kwargs
                extra_kwargs={
                    'guidance_scale': guidance_scale,
                    'guidance_scale_2': guidance_scale_2,
                    'boundary_timestep': boundary_timestep,
                    'attention_kwargs': attention_kwargs,
                    **{k: v[b] for k, v in extra_call_back_res.items()}
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()

        return samples
    

    # ======================== Forward ========================

    def forward(
        self,
        samples: List[WanI2VSample],
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> UniPCMultistepSDESchedulerOutput:
        # 1. Extract data from samples
        batch_size = len(samples)
        device = self.device
        dtype = self.pipeline.transformer.dtype if self.pipeline.transformer is not None else self.pipeline.transformer_2.dtype
        # Assume all samples have the same guidance scale
        guidance_scale = samples[0].extra_kwargs.get('guidance_scale', self.training_args.guidance_scale)
        guidance_scale_2 = samples[0].extra_kwargs.get('guidance_scale_2', guidance_scale)
        do_classifier_free_guidance = guidance_scale > 1.0
        # Assume all samples have the same attention kwargs
        attention_kwargs = samples[0].extra_kwargs.get('attention_kwargs', {})
        boundary_timestep = samples[0].extra_kwargs.get('boundary_timestep', None)

        # Stack latents and timesteps
        latents = torch.stack([s.all_latents[timestep_index] for s in samples], dim=0).to(device)
        next_latents = torch.stack([s.all_latents[timestep_index + 1] for s in samples], dim=0).to(device)
        timestep = torch.stack([s.timesteps[timestep_index] for s in samples], dim=0).to(device)
        t = timestep[0]
        num_inference_steps = len(samples[0].timesteps)

        # Get prompt embeddings        
        prompt_embeds = torch.stack([s.prompt_embeds for s in samples], dim=0).to(device=device,dtype=dtype)
        negative_prompt_embeds = (
            torch.stack([s.negative_prompt_embeds for s in samples], dim=0).to(device=device,dtype=dtype)
            if samples[0].negative_prompt_embeds is not None else None
        )
        # Get conditions
        condition = torch.stack([s.condition_latents for s in samples], dim=0).to(device=device,dtype=dtype)
        image_embeds = (
            torch.stack([s.image_embeds for s in samples], dim=0).to(device=device,dtype=dtype)
            if samples[0].image_embeds is not None else None
        )
        first_frame_mask = samples[0].first_frame_mask.to(device=device,dtype=dtype) if samples[0].first_frame_mask is not None else None


        # 2. Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # 3. Determine which transformer to use
        if boundary_timestep is None or t >= boundary_timestep:
            # wan2.1 or high-noise stage in wan2.2
            current_pipeline_model = self.pipeline.transformer
            current_model = self.transformer
            current_guidance_scale = guidance_scale
        else:
            # low-noise stage in wan2.2
            current_pipeline_model = self.pipeline.transformer_2
            current_model = self.transformer_2
            current_guidance_scale = guidance_scale_2


        # 4. Determine latent model input
        if self.pipeline.config.expand_timesteps:
            latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
            latent_model_input = latent_model_input.to(dtype)

            # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
            temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
            # batch_size, seq_len
            timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
        else:
            latent_model_input = torch.cat([latents, condition], dim=1).to(dtype)
            timestep = t.expand(latents.shape[0])


        # 5. Predict noise
        with current_pipeline_model.cache_context("cond"):
            noise_pred = current_model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

        if do_classifier_free_guidance:
            with current_pipeline_model.cache_context("uncond"):
                noise_uncond = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

        # 6. Step the scheduler
        step_kwargs = filter_kwargs(self.scheduler.step, **kwargs)
        output = self.scheduler.step(
            noise_pred=noise_pred,
            timestep=t,
            latents=latents,
            next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            **step_kwargs,
        )

        return output