# src/flow_factory/models/flux.py
from typing import Union, List, Dict, Any, Optional
from dataclasses import dataclass
import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from PIL import Image

from .adapter import BaseAdapter, BaseSample
from ..scheduler.flow_matching import FlowMatchEulerDiscreteSDESchedulerOutput, set_scheduler_timesteps

@dataclass
class FluxSample(BaseSample):
    """Output class for Flux Adapter models."""
    pass


class FluxAdapter(BaseAdapter):
    """Concrete implementation for Flow Matching models (FLUX.1)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Load pipeline
        self.pipeline = FluxPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=torch.bfloat16 if self.training_args.mixed_precision == "bf16" else torch.float16,
        )
        
        # Freeze non-trainable components
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder_2.requires_grad_(False)
        
        # Use custom SDE scheduler
        self.pipeline.scheduler = self.scheduler

    # ======================== Component Management ========================
    
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
    
    def encode_prompts(self, prompts: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Encode text prompts using the pipeline's text encoder."""
        
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            device=self.device,
        )
        
        prompt_ids = self.pipeline.tokenizer_2(
            prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
                
        return {
            'prompt_ids': prompt_ids,
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
            'text_ids': text_ids,
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
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        compute_log_probs: bool = True,
        **kwargs,
    ) -> List[FluxSample]:
        """Execute generation and return FluxSample objects."""
        
        # Setup
        height = height or self.training_args.resolution[0]
        width = width or self.training_args.resolution[1]
        num_inference_steps = num_inference_steps or self.training_args.num_timesteps
        guidance_scale = guidance_scale or self.training_args.guidance_scale
        batch_size = prompt_embeds.shape[0] if prompt_embeds is not None else 1
        device = self.device
        
        # Encode prompts if not provided
        if prompt_embeds is None:
            encoded = self.encode_prompts(prompt)
            prompt_embeds = encoded['prompt_embeds']
            pooled_prompt_embeds = encoded['pooled_prompt_embeds']
            prompt_ids = encoded['prompt_ids']
            text_ids = encoded['text_ids']
        else:
            text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(
                device=device, dtype=prompt_embeds.dtype
            )
        
        # Prepare latents
        num_channels = self.pipeline.transformer.config.in_channels // 4
        latents = self.pipeline.prepare_latents(
            batch_size, num_channels, height, width,
            prompt_embeds.dtype, device, generator,
        )
        
        latent_image_ids = self.pipeline._prepare_latent_image_ids(
            batch_size, height, width, device, prompt_embeds.dtype
        )
        latents = self.pipeline._pack_latents(latents, batch_size, num_channels, height, width)
        
        # Set timesteps with scheduler
        timesteps = set_scheduler_timesteps(
            self.scheduler, num_inference_steps, latents.shape[1], device
        )
        self.scheduler.set_begin_index(0)
        
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
        samples = []
        for b in range(batch_size):
            sample = FluxSample(
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps_tensor[b],
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                height=height,
                width=width,
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                image=images[b],
                prompt_embeds=prompt_embeds[b],
                pooled_prompt_embeds=pooled_prompt_embeds[b],
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_probs else None,
            )
            samples.append(sample)
        
        return samples

    # ======================== Forward (Training) ========================
    
    def forward(
        self,
        samples: List[FluxSample],
        timestep_index : int,
        return_log_prob: bool = True,
        **kwargs,
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """Compute log-probabilities for training."""
        self.on_load_transformer()
        
        batch_size = len(samples)
        device = self.device
        
        # Extract data from samples
        latents = torch.stack([s.all_latents[timestep_index] for s in samples], dim=0).to(device)
        next_latents = torch.stack([s.all_latents[timestep_index + 1] for s in samples], dim=0).to(device)
        timesteps = torch.stack([s.timesteps[timestep_index] for s in samples], dim=0).to(device)
        
        prompt_embeds = torch.stack([s.prompt_embeds for s in samples], dim=0).to(device)
        pooled_prompt_embeds = torch.stack([s.pooled_prompt_embeds for s in samples], dim=0).to(device)
        text_ids = torch.stack([s.extra_kwargs['text_ids'] for s in samples], dim=0).to(device)
        latent_image_ids = torch.stack([s.extra_kwargs['latent_image_ids'] for s in samples], dim=0).to(device)
        
        # Set scheduler timesteps
        _ = set_scheduler_timesteps(
            self.scheduler, self.training_args.num_timesteps, latents.shape[1], device
        )
        
        guidance = torch.full([batch_size], 3.5, device=device, dtype=torch.float32)
        
        # Forward pass
        noise_pred = self.pipeline.transformer(
            hidden_states=latents,
            timestep=timesteps / 1000,
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
            timestep=timesteps,
            sample=latents,
            prev_sample=next_latents,
            return_log_prob=return_log_prob,
            sde_type=self.training_args.sde_type,
        )
        
        self.off_load_transformer()
        return output

    # ======================== Utilities ========================
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.pipeline.transformer, 'enable_gradient_checkpointing'):
            self.pipeline.transformer.enable_gradient_checkpointing()
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable parameters for optimizer."""
        return list(self.pipeline.transformer.parameters())
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        state_dict = torch.load(path, map_location=self.device)
        self.pipeline.transformer.load_state_dict(state_dict, strict=False)
    
    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        torch.save(self.pipeline.transformer.state_dict(), path)

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.pipeline.transformer.parameters()).device