# src/flow_factory/models/flux/flux2.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline, format_input, compute_empirical_mu
from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE, SYSTEM_MESSAGE_UPSAMPLING_T2I, SYSTEM_MESSAGE_UPSAMPLING_I2I
from PIL import Image
import logging

from ..adapter import BaseAdapter, BaseSample
from ...hparams import *
from ...scheduler import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput, set_scheduler_timesteps
from ...utils.base import filter_kwargs, pil_image_to_tensor

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Flux2Sample(BaseSample):
    """Output class for Flux2Adapter models."""
    text_ids : Optional[torch.Tensor] = None
    condition_images : Optional[Union[List[Image.Image], Image.Image]] = None
    image_latents : Optional[torch.Tensor] = None
    image_latent_ids : Optional[torch.Tensor] = None


class Flux2Adapter(BaseAdapter):
    """Concrete implementation for Flow Matching models (FLUX.2)."""
    
    def __init__(self, config: Arguments):
        super().__init__(config)
        self._has_warned_inference_fallback = False
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
            "attn.to_out"
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
    def encode_image(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> Dict[str, torch.Tensor]:
        """Encode input condition_image(s) into latent representations using the Flux.2 image encoder."""
        device = kwargs.get('device', self.pipeline.vae.device)
        dtype = kwargs.get('dtype', self.pipeline.vae.dtype)
        condition_image_tensors : List[torch.Tensor] = self._resize_condition_images(
            condition_images=images,
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
    def encode_video(self, video: Any, **kwargs) -> None:
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

    def _resize_condition_images(self, condition_images: Union[Image.Image, List[Image.Image]], **kwargs) -> List[torch.Tensor]:
        """Preprocess condition images for Flux.2 model."""
        if isinstance(condition_images, Image.Image):
            condition_images = [condition_images]

        for img in condition_images:
            self.pipeline.image_processor.check_image_input(img)

        condition_image_tensors = []
        for img in condition_images:
            image_width, image_height = img.size
            if image_width * image_height > 1024 * 1024:
                img = self.pipeline.image_processor._resize_to_target_area(img, 1024 * 1024)
                image_width, image_height = img.size

            multiple_of = self.pipeline.vae_scale_factor * 2
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of
            img = self.pipeline.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
            condition_image_tensors.append(img)

        return condition_image_tensors


    # ========================Preprocessing ========================
    def preprocess_func(
            self,
            prompt: List[str],
            images: Optional[Union[List[Optional[Image.Image]], List[List[Optional[Image.Image]]]]] = None,
            caption_upsample_temperature: Optional[float] = None,
            **kwargs
        ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """Preprocess inputs for Flux.2 model. The inputs are expected to be batches."""
        assert isinstance(prompt, list), "Prompt must be a batch of strings."
        if images is not None:
            assert len(prompt) == len(images), "Number of prompts must match number of condition image sets."
            # A batch of image lists, convert each image to a singleton list if not already
            images = [
                [img] if not isinstance(img, list) else img
                for img in images
            ]
        
        if images is None:
            # No condition images provided, process only prompts in batch
            if caption_upsample_temperature:
                final_prompt = self.pipeline.upsample_prompt(
                    prompt=prompt,
                    images=None,
                    temperature=caption_upsample_temperature,
                    device=self.pipeline.text_encoder.device
                )
            else:
                final_prompt = prompt
            
            # 1. Process prompt
            prompt_encode_dict = self.encode_prompt(
                prompt=final_prompt,
                device=self.pipeline.text_encoder.device,
                **filter_kwargs(self.encode_prompt, **kwargs)
            )
            # No condition images
            batch = {**prompt_encode_dict}
        else:
            # Condition images provided, process each sample individually
            if not self._has_warned_preprocess_fallback:
                logger.warning(
                    "Flux.2 does not support processing with batches of condition images."
                    "Processing each sample individually. This warning will only appear once."
                )
                self._has_warned_preprocess_fallback = True

            # images: List[List[Optional[Image.Image]]]
            batch = []
            for p, imgs in zip(
                prompt,
                images,
            ):
                # imgs: List[Optional[Image.Image]]
                if caption_upsample_temperature:
                    final_prompt = self.pipeline.upsample_prompt(
                        prompt=p,
                        images=imgs,
                        temperature=caption_upsample_temperature,
                        device=self.pipeline.text_encoder.device
                    )
                else:
                    final_prompt = p
                
                # 1. Process prompt
                prompt_encode_dict = self.encode_prompt(
                    prompt=final_prompt,
                    device=self.pipeline.text_encoder.device,
                    **filter_kwargs(self.encode_prompt, **kwargs)
                )

                # 2. Process condition images
                if len(imgs) == 0 or imgs[0] is None:
                    image_encode_dict = {
                        'image_latents': None,
                        'image_latent_ids': None,
                    }
                else:
                    image_encode_dict = self.encode_image(
                        images=imgs,
                        device=self.pipeline.vae.device,
                        **filter_kwargs(self.encode_image, **kwargs)
                    )
                sample = {**prompt_encode_dict, **image_encode_dict}
                batch.append(sample)

            # Collate batch, convert List[Dict] to Dict[List]
            batch = {
                k : [sample[k] for sample in batch]
                for k in batch[0].keys()
            }

        return batch
        
    
    # ======================== Sampling / Inference ========================

    # Since Flux.2 does not support ragged batches of condition images, we implement a single-sample inference method.
    def _inference(
        self,
        condition_images: Optional[Union[List[Image.Image], Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 4.0,
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
    ) -> List[Flux2Sample]:
        """
        Inference method for Flux.2 model for a single sample.
        The condition images can be a list of images or a single image, shared across the batch.
        """

        # 1. Setup
        height = height or (self.training_args.resolution[0] if self.training else self.training_args.eval_args.resolution[0])
        width = width or (self.training_args.resolution[1] if self.training else self.training_args.eval_args.resolution[1])
        num_inference_steps = num_inference_steps or (self.training_args.num_inference_steps if self.training else self.training_args.eval_args.num_inference_steps)
        guidance_scale = guidance_scale or (self.training_args.guidance_scale if self.training else self.training_args.eval_args.guidance_scale)
        device = self.device
        dtype = self.transformer.dtype

        # 2. Preprocess inputs
        if (
            prompt_embeds is None or text_ids is None
            or image_latents is None or image_latent_ids is None
        ):
            if isinstance(prompt, str):
                prompt = [prompt]
            if isinstance(condition_images, Image.Image):
                condition_images = [condition_images]
            encode_dict = self.preprocess_func(
                prompt=prompt,
                images=condition_images,
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

        # 3. Prepare initial noise
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, latent_ids = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )

        # 4. Prepare timesteps
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
        timesteps = set_scheduler_timesteps(
            scheduler=self.pipeline.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            mu=mu,
        )

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])


        # 5. Run diffusion process
        all_latents = [latents]
        all_log_probs = [] if compute_log_prob else None
        for i, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                current_noise_level = self.scheduler.get_noise_level_for_timestep(t)

                latent_model_input = latents.to(self.pipeline.transformer.dtype)
                latent_image_ids = latent_ids

                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.pipeline.transformer.dtype)
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
                latents_dtype = latents.dtype
                output = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents,
                    compute_log_prob=compute_log_prob and current_noise_level > 0,
                )

                latents = output.prev_sample.to(dtype)
                all_latents.append(latents)
                
                if compute_log_prob:
                    all_log_probs.append(output.log_prob)

        images = self.decode_latents(latents, latent_ids)
        samples = [
            Flux2Sample(
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps,
                height=height,
                width=width,
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b],
                prompt_embeds=prompt_embeds[b],
                text_ids=text_ids[b] if text_ids is not None else None,
                condition_images=condition_images if condition_images is not None else None,
                image_latents=image_latents[b] if image_latents is not None else None,
                image_latent_ids=image_latent_ids[b] if image_latent_ids is not None else None,
                image=images[b],
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_prob else None,
                extra_kwargs={'guidance_scale': guidance_scale},
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()
        
        return samples

    def inference(
        self,
        condition_images: Optional[Union[List[Image.Image], List[List[Image.Image]]]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 4.0,
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
    ) -> List[Flux2Sample]:
        # Check if images are given as a batch of image lists
        if (
            condition_images is not None 
            and isinstance(condition_images, list)
            and any(isinstance(i, list) for i in condition_images) # A batch of image lists
        ):
            if not self._has_warned_inference_fallback:
                logger.warning(
                    "FLUX.2 does not support batch inference with varying condition images per sample. "
                    "Falling back to single-sample inference. This warning will only appear once."
                )
                self._has_warned_inference_fallback = True
            # Process each sample individually by calling _inference_single
            samples = []
            # Expand prompt if needed
            if not isinstance(prompt, list):
                prompt = [prompt] * len(condition_images)
            for idx in range(len(condition_images)):
                sample = self._inference(
                    condition_images=condition_images[idx],
                    prompt=prompt[idx],
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    prompt_ids=prompt_ids,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    image_latents=image_latents[idx] if image_latents is not None else None,
                    image_latent_ids=image_latent_ids[idx] if image_latent_ids is not None else None,
                    attention_kwargs=attention_kwargs,
                    max_sequence_length=max_sequence_length,
                    text_encoder_out_layers=text_encoder_out_layers,
                    caption_upsample_temperature=caption_upsample_temperature,
                    compute_log_prob=compute_log_prob,
                )
                samples.append(sample[0])
            return samples

        else:
            # Images are shared across the batch
            return self._inference(
                condition_images=condition_images,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                prompt_ids=prompt_ids,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                image_latents=image_latents,
                image_latent_ids=image_latent_ids,
                attention_kwargs=attention_kwargs,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
                caption_upsample_temperature=caption_upsample_temperature,
                compute_log_prob=compute_log_prob,
            )
        

    # ======================== Forward (Training) ========================
    def forward(
        self,
        samples: List[Flux2Sample],
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """Compute log-probabilities for training."""
        
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
        prompt_embeds = torch.stack([s.prompt_embeds for s in samples], dim=0).to(device)
        text_ids = torch.stack([s.text_ids for s in samples], dim=0).to(device)
        image_latents = torch.stack(
            [s.image_latents for s in samples],
            dim=0
        ) if samples[0].image_latents is not None else None
        image_latent_ids = torch.stack(
            [s.image_latent_ids for s in samples],
            dim=0
        ) if samples[0].image_latent_ids is not None else None

        latent_ids = self.pipeline._prepare_latent_ids(latents)

        # Catenate condition latents if given
        latent_model_input = latents.to(self.pipeline.transformer.dtype)
        latent_image_ids = latent_ids

        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.pipeline.transformer.dtype)
            latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
                
        # 2. Set scheduler timesteps
        _ = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=self.training_args.num_inference_steps,
            seq_len=latents.shape[1],
            device=device
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
            model_output=noise_pred,
            timestep=timestep,
            sample=latents,
            prev_sample=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            **step_kwargs,
        )
        
        return output
