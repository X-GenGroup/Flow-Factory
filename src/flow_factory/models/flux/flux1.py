# src/flow_factory/models/flux/flux1.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from PIL import Image
import logging

from ..adapter import BaseAdapter, BaseSample
from ...hparams import *
from ...scheduler import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput, set_scheduler_timesteps
from ...utils.base import filter_kwargs

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Flux1Sample(BaseSample):
    """Output class for Flux Adapter models."""
    pooled_prompt_embeds : Optional[torch.FloatTensor] = None


class Flux1Adapter(BaseAdapter):
    """Concrete implementation for Flow Matching models (FLUX.1)."""
    
    def __init__(self, config: Arguments):
        super().__init__(config)
    
    def load_pipeline(self) -> FluxPipeline:
        return FluxPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )

    @property
    def default_target_modules(self) -> List[str]:
        return [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2",
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
    
    def encode_prompt(self, prompt: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Encode text prompts using the pipeline's text encoder."""

        execution_device = self.pipeline.text_encoder.device
        
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=execution_device,
        )
        
        prompt_ids = self.pipeline.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(execution_device)
                
        return {
            'prompt_ids': prompt_ids,
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
        }
    
    def encode_image(self, images: Union[Image.Image, List[Optional[Image.Image]]], **kwargs) -> None:
        """
        Encode input images into latent representations using the VAE encoder.
         Args:
            images:
                - Single Image.Image
                - List[Image.Image]: list of images
        """
        pass

    def encode_video(self, videos: Union[torch.Tensor, List[torch.Tensor]], **kwargs) -> None:
        """Not needed for FLUX text-to-image models."""
        pass

    def decode_latents(self, latents: torch.Tensor, height: int, width: int, **kwargs) -> List[Image.Image]:
        """Decode latents to images using VAE."""
        
        latents = self.pipeline._unpack_latents(latents, height, width, self.pipeline.vae_scale_factor)
        latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
        latents = latents.to(dtype=self.pipeline.vae.dtype)
        
        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type="pil")
        
        return images

    # ======================== Inference ========================
    
    @torch.no_grad()
    def inference(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_ids : Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        compute_log_prob: bool = True,
    ) -> List[Flux1Sample]:
        """Execute generation and return FluxSample objects."""
        
        # 1. Setup
        height = height or (self.training_args.resolution[0] if self.training else self.eval_args.resolution[0])
        width = width or (self.training_args.resolution[1] if self.training else self.eval_args.resolution[1])
        num_inference_steps = num_inference_steps or (self.training_args.num_inference_steps if self.training else self.eval_args.num_inference_steps)
        guidance_scale = guidance_scale or (self.training_args.guidance_scale if self.training else self.eval_args.guidance_scale)
        device = self.device
        dtype = self.transformer.dtype
        
        # 2. Encode prompts if not provided
        if prompt_embeds is None:
            encoded = self.encode_prompt(prompt)
            prompt_embeds = encoded['prompt_embeds']
            pooled_prompt_embeds = encoded['pooled_prompt_embeds']
            prompt_ids = encoded['prompt_ids']
        else:
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)

        batch_size = len(prompt_embeds)

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=device, dtype=dtype
        )
        
        # 3. Prepare latents
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, latent_image_ids = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        
        # 4. Set timesteps with scheduler
        timesteps = set_scheduler_timesteps(
            scheduler=self.pipeline.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device,
        )

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        
        # 5. Denoising loop
        all_latents = [latents]
        all_log_probs = [] if compute_log_prob else None
        
        for i, t in enumerate(timesteps):
            timestep = t.expand(batch_size).to(latents.dtype)
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            
            # Predict noise
            noise_pred = self.pipeline.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance.expand(batch_size),
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            
            # Scheduler step
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
        
        # 6. Decode images
        images = self.decode_latents(latents, height, width)
        
        # 7. Create samples
        samples = [
            Flux1Sample(
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                height=height,
                width=width,
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                image=images[b],
                prompt_embeds=prompt_embeds[b],
                pooled_prompt_embeds=pooled_prompt_embeds[b],
                image_ids=latent_image_ids,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_prob else None,
                extra_kwargs={'guidance_scale': guidance_scale},
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()
        
        return samples

    # ======================== Forward (Training) ========================
    
    def forward(
        self,
        samples: List[Flux1Sample],
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
        t = timestep[0]
        
        prompt_embeds = torch.stack([s.prompt_embeds for s in samples], dim=0).to(device)
        pooled_prompt_embeds = torch.stack([s.pooled_prompt_embeds for s in samples], dim=0).to(device)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device)
        latent_image_ids = samples[0].image_ids.to(device) # No batch dimension needed
        
        # 2. Set scheduler timesteps
        _ = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=self.training_args.num_inference_steps,
            seq_len=latents.shape[1],
            device=device
        )
    
        
        # 3. Forward pass
        noise_pred = self.pipeline.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance.expand(batch_size),
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        
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
