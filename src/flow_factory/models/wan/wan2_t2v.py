# src/flow_factory/models/wan/wan2_t2v.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple, Literal
import logging
from dataclasses import dataclass
from collections import defaultdict

from PIL import Image
import torch
from accelerate import Accelerator
from diffusers.pipelines.wan.pipeline_wan import WanPipeline, prompt_clean
from peft import PeftModel

from ..adapter import BaseAdapter, BaseSample
from ...hparams import *
from ...scheduler import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput, set_scheduler_timesteps
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class WanT2VSample(BaseSample):
    video : Optional[List[List[Image.Image]]] = None



class Wan2_T2V_Adapter(BaseAdapter):
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
    
    def load_pipeline(self) -> WanPipeline:
        return WanPipeline.from_pretrained(
            self.model_args.model_name_or_path,
        )
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Wan transformer."""
        return [
            # --- Self Attention ---
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            
            # --- Cross Attention ---
            "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
            
            # --- Feed Forward Network ---
            "ffn.0", "ffn.2"
        ]
    
    def apply_lora(self, components=['transformer', 'transformer_2'], target_modules=None) -> Union[PeftModel, Dict[str, PeftModel]]:
        return super().apply_lora(components, target_modules)
    
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
    ):
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
    
    def encode_image(self, image: Union[Image.Image, torch.Tensor, List[torch.Tensor]], **kwargs) -> torch.Tensor:
        """Not needed for Wan text-to-video models."""
        pass

    def encode_video(self, video: Union[torch.Tensor, List[torch.Tensor]], **kwargs) -> torch.Tensor:
        """Not needed for Wan text-to-video models."""
        pass

    def decode_latents(self, latents: torch.Tensor, output_type: Literal['pt', 'pil', 'np'] = 'pil', **kwargs) -> torch.Tensor:
        """Decode the latents using the VAE decoder."""
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
        video = self.pipeline.vae.decode(latents, return_dict=False)[0]

        video = self.pipeline.video_processor.postprocess_video(video, output_type=output_type)

    # ======================== Inference ========================

    @torch.no_grad()
    def inference(
        self,
        # Ordinary args
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,

        # Prompt encoding args
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,

        # Other args
        compute_log_prob: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,

        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
        **kwargs,
    ):
        # 1. Setup args
        height = height or (self.eval_args.resolution[0] if self.mode == 'eval' else self.training_args.resolution[0])
        width = width or (self.eval_args.resolution[1] if self.mode == 'eval' else self.training_args.resolution[1])
        num_inference_steps = num_inference_steps or (self.eval_args.num_inference_steps if self.mode == 'eval' else self.training_args.num_inference_steps)
        guidance_scale = guidance_scale or (self.eval_args.guidance_scale if self.mode == 'eval' else self.training_args.guidance_scale)
        device = self.pipeline._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        if self.pipeline.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        if num_frames % self.pipeline.vae_scale_factor_temporal != 1:
            logger.warning(f"`num_frames - 1` has to be divisible by {self.pipeline.vae_scale_factor_temporal}. Rounding to the nearest number.")
            num_frames = num_frames // self.pipeline.vae_scale_factor_temporal * self.pipeline.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

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
            
        batch_size =prompt_embeds.shape[0]
        transformer_dtype = self.pipeline.transformer.dtype if self.pipeline.transformer is not None else self.pipeline.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)


        # 3. Set scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps


        # 5. Prepare latent variables
        num_channels_latents = (
            self.pipeline.transformer.config.in_channels
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.config.in_channels
        )
        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        if self.pipeline.config.boundary_ratio is not None:
            boundary_timestep = self.pipeline.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None        

        all_latents = [latents]
        all_log_probs = [] if compute_log_prob else None
        extra_call_back_res = defaultdict(list)

        for i, t in enumerate(timesteps):
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)

            if boundary_timestep is None or t >= boundary_timestep:
                # wan2.1 or high-noise stage in wan2.2
                current_model_pipeline = self.pipeline.transformer
                current_model = self.transformer
                current_guidance_scale = guidance_scale
            else:
                # low-noise stage in wan2.2
                current_model_pipeline = self.pipeline.transformer_2
                current_model = self.transformer_2
                current_guidance_scale = guidance_scale_2

            latent_model_input = latents.to(transformer_dtype)
            if self.pipeline.config.expand_timesteps:
                # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                # batch_size, seq_len
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                timestep = t.expand(latents.shape[0])

            with current_model_pipeline.cache_context("cond"):
                noise_pred = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

            if do_classifier_free_guidance:
                with current_model_pipeline.cache_context("uncond"):
                    noise_uncond = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
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
                    if hasattr(output, key):
                        val = getattr(output, key)
                        if val is not None:
                            extra_call_back_res[key].append(val)
                    elif key in capturable and capturable[key] is not None:
                        extra_call_back_res[key].append(capturable[key])

        # 7. Decode latents to videos (list of pil images)
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
            WanT2VSample(
                # Denoising trajectory
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_prob else None,

                # Generated video & metadata
                video=decoded_videos[b],
                height=height,
                width=width,

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

    # ======================== Forward (Training) ========================

    def forward(
        self,
        samples: List[WanT2VSample],
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
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


        # 2. Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)


        # 3. Determine which transformer to use
        if boundary_timestep is None or t >= boundary_timestep:
            # wan2.1 or high-noise stage in wan2.2
            current_model_pipeline = self.pipeline.transformer
            current_model = self.transformer
            current_guidance_scale = guidance_scale
        else:
            # low-noise stage in wan2.2
            current_model_pipeline = self.pipeline.transformer_2
            current_model = self.transformer_2
            current_guidance_scale = guidance_scale_2

        # 4. Predict noise
        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)
        latent_model_input = latents.to(dtype)
        if self.pipeline.config.expand_timesteps:
            # seq_len: num_latent_frames * latent_height//2 * latent_width//2
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            # batch_size, seq_len
            timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
        else:
            timestep = t.expand(latents.shape[0])

        with current_model_pipeline.cache_context("cond"):
            noise_pred = current_model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

        if do_classifier_free_guidance:
            with current_model_pipeline.cache_context("uncond"):
                noise_uncond = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

        # compute the previous noisy sample x_t -> x_t-1
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