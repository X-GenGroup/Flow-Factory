# src/flow_factory/models/flux.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional
from dataclasses import dataclass
import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from PIL import Image
import logging

from ..hparams import *
from .adapter import BaseAdapter, BaseSample
from ..scheduler.flow_matching import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput, set_scheduler_timesteps

@dataclass
class Flux1Sample(BaseSample):
    """Output class for Flux Adapter models."""
    pass


class Flux1Adapter(BaseAdapter):
    """Concrete implementation for Flow Matching models (FLUX.1)."""
    
    def __init__(self, config: Arguments):
        super().__init__(config)

        # Load pipeline
        self.pipeline = FluxPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=torch.bfloat16 if self.training_args.mixed_precision == "bf16" else torch.float16,
        )
        
        
        # Initialize Scheduler
        self.pipeline.scheduler = FlowMatchEulerDiscreteSDEScheduler(
            noise_level=self.training_args.noise_level,
            noise_steps=self.training_args.noise_steps,
            num_noise_steps=self.training_args.num_noise_steps,
        )

        if self.config.model_args.resume_path:
            self.load_checkpoint(self.config.model_args.resume_path)
        elif self.config.model_args.finetune_type == 'lora':
            self.apply_lora()

        # Freeze non-trainable components
        self._freeze_components()
        self._mix_precision()

        if self.training_args.enable_gradient_checkpointing:
            self.enable_gradient_checkpointing()

    # ======================== Component Management ========================

    @property
    def transformer(self) -> torch.nn.Module:
        return self.pipeline.transformer

    @property
    def scheduler(self) -> FlowMatchEulerDiscreteSDEScheduler:
        return self.pipeline.scheduler

    def _mix_precision(self):
        """Handle mixed precision settings."""
        if self.training_args.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.training_args.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float32
        
        self.pipeline.text_encoder.to(dtype=inference_dtype)
        self.pipeline.text_encoder_2.to(dtype=inference_dtype)
        self.pipeline.vae.to(dtype=inference_dtype)
        if self.config.model_args.finetune_type == 'full':
            self.pipeline.transformer.to(dtype=torch.float32)
        else:
            self.pipeline.transformer.to(dtype=inference_dtype)

    def _freeze_components(self):
        """Encapsulate freezing logic for cleanliness."""
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder_2.requires_grad_(False)
        self.pipeline.transformer.requires_grad_(self.model_args.finetune_type == 'full')

    def off_load_text_encoder(self):
        self.pipeline.text_encoder.to("cpu")
        self.pipeline.text_encoder_2.to("cpu")

    def off_load_vae(self):
        self.pipeline.vae.to("cpu")

    def off_load_transformer(self):
        self.pipeline.transformer.to("cpu")

    def on_load_text_encoder(self, device: Union[torch.device, str] = None):
        device = device or self.device
        self.pipeline.text_encoder.to(device)
        self.pipeline.text_encoder_2.to(device)

    def on_load_vae(self, device: Union[torch.device, str] = None):
        device = device or self.device
        self.pipeline.vae.to(device)

    def on_load_transformer(self, device: Union[torch.device, str] = None):
        device = device or self.device
        self.pipeline.transformer.to(device)

    # ======================== Encoding & Decoding ========================
    
    def encode_prompts(self, prompt: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
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
    
    def encode_images(self, images: Union[torch.Tensor, List[torch.Tensor]], **kwargs) -> torch.Tensor:
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
        compute_log_probs: bool = True,
        **kwargs,
    ) -> List[Flux1Sample]:
        """Execute generation and return FluxSample objects."""
        
        # Setup
        height = height or self.training_args.resolution[0]
        width = width or self.training_args.resolution[1]
        num_inference_steps = num_inference_steps or self.training_args.num_timesteps
        guidance_scale = guidance_scale or self.training_args.guidance_scale
        batch_size = prompt_embeds.shape[0] if prompt_embeds is not None else 1
        device = self.device
        dtype = prompt_embeds.dtype if prompt_embeds is not None else torch.float32
        # Encode prompts if not provided
        if prompt_embeds is None:
            encoded = self.encode_prompts(prompt)
            prompt_embeds = encoded['prompt_embeds']
            pooled_prompt_embeds = encoded['pooled_prompt_embeds']
            prompt_ids = encoded['prompt_ids']
        else:
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
            device=device, dtype=prompt_embeds.dtype
        )
        
        # Prepare latents
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
        
        # Set timesteps with scheduler
        timesteps = set_scheduler_timesteps(
            scheduler=self.pipeline.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device,
        )

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        
        # Denoising loop
        all_latents = [latents]
        all_log_probs = [] if compute_log_probs else None
        
        for i, t in enumerate(timesteps):
            timestep = t.expand(batch_size).to(latents.dtype)
            
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
                timestep=timestep,
                sample=latents,
                return_log_prob=compute_log_probs,
                sde_type=self.training_args.sde_type
            )
            
            latents = output.prev_sample.to(prompt_embeds.dtype)
            all_latents.append(latents)
            
            if compute_log_probs:
                all_log_probs.append(output.log_prob)
        
        # Decode images
        images = self.decode_latents(latents, height, width)
        timesteps_tensor = timesteps.unsqueeze(0).expand(batch_size, -1)
        
        # Create samples
        samples = [
            Flux1Sample(
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps_tensor[b],
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                height=height,
                width=width,
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                image=images[b],
                prompt_embeds=prompt_embeds[b],
                pooled_prompt_embeds=pooled_prompt_embeds[b],
                image_ids=latent_image_ids,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_probs else None,
            )
            for b in range(batch_size)
        ]
        
        return samples

    # ======================== Forward (Training) ========================
    
    def forward(
        self,
        samples: List[Flux1Sample],
        timestep_index : int,
        return_log_prob: bool = True,
        **kwargs,
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """Compute log-probabilities for training."""
        
        batch_size = len(samples)
        device = self.device
        
        # Extract data from samples
        latents = torch.stack([s.all_latents[timestep_index] for s in samples], dim=0).to(device)
        next_latents = torch.stack([s.all_latents[timestep_index + 1] for s in samples], dim=0).to(device)
        timestep = torch.stack([s.timesteps[timestep_index] for s in samples], dim=0).to(device)
        
        prompt_embeds = torch.stack([s.prompt_embeds for s in samples], dim=0).to(device)
        pooled_prompt_embeds = torch.stack([s.pooled_prompt_embeds for s in samples], dim=0).to(device)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device)
        latent_image_ids = samples[0].image_ids.to(device) # No batch dimension needed
        
        # Set scheduler timesteps
        _ = set_scheduler_timesteps(
            self.scheduler, self.training_args.num_timesteps, latents.shape[1], device
        )
        
        guidance = torch.full([batch_size], 3.5, device=device, dtype=torch.float32)
        
        # Forward pass
        noise_pred = self.pipeline.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        
        # Compute log prob with ground truth next_latents
        output = self.scheduler.step(
            model_output=noise_pred,
            timestep=timestep,
            sample=latents,
            prev_sample=next_latents,
            return_log_prob=return_log_prob,
            sde_type=self.training_args.sde_type,
        )
        
        return output

    # ======================== Utilities ========================

    def train(self, mode: bool = True) -> "Flux1Adapter":
        super().train(mode)
        self.pipeline.transformer.train(mode)
        return self

    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        self.pipeline.transformer.eval()

    @property
    def default_lora_target_modules(self) -> List[str]:
        return [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2",
        ]

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.pipeline.transformer, 'enable_gradient_checkpointing'):
            self.pipeline.transformer.enable_gradient_checkpointing()
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable parameters for optimizer."""
        return [p for p in self.pipeline.transformer.parameters() if p.requires_grad]

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return self.pipeline.transformer.device