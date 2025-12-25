# src/flow_factory/models/qwen_image/qwen_image.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from PIL import Image
import logging

from ..adapter import BaseAdapter, BaseSample
from ...hparams import *
from ...scheduler import FlowMatchEulerDiscreteSDEScheduler, FlowMatchEulerDiscreteSDESchedulerOutput, set_scheduler_timesteps
from ...utils.base import filter_kwargs

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QwenImageSample(BaseSample):
    """Output class for Qwen-Image models."""
    prompt_embeds_mask : Optional[torch.FloatTensor] = None
    negative_prompt_embeds_mask : Optional[torch.FloatTensor] = None
    img_shapes : Optional[List[Tuple[int, int, int]]] = None

class QwenImageAdapter(BaseAdapter):
    """Adapter for Qwen-Image text-to-image models."""
    
    def __init__(self, config: Arguments):
        super().__init__(config)
        self._warned_cfg_no_neg_prompt = False
        self._warned_no_cfg = False
    
    def load_pipeline(self) -> QwenImagePipeline:
        return QwenImagePipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Qwen-Image transformer."""
        return [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
            "img_mlp.net.0.proj", "img_mlp.net.2.proj",
            "txt_mlp.net.0.proj", "txt_mlp.net.2.proj"
        ]
    
    # ======================== Encoding & Decoding ========================
    
    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.pipeline.prompt_template_encode
        drop_idx = self.pipeline.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.pipeline.tokenizer(
            txt, max_length=self.pipeline.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self.pipeline._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
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

        return txt_tokens.input_ids, prompt_embeds, encoder_attention_mask
    
    def encode_prompt(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: Optional[Union[str, List[str]]] = None,
            max_sequence_length: int = 1024,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            **kwargs
        ) -> Dict[str, Union[torch.LongTensor, torch.Tensor]]:
        """Encode text prompts using the pipeline's text encoder."""

        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        # Encode positive prompt
        prompt_ids, prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, device, dtype)
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
                negative_prompt, device, dtype
            )
            results.update({
                "negative_prompt_ids": negative_prompt_ids,
                "negative_prompt_embeds": negative_prompt_embeds[:, :max_sequence_length],
                "negative_prompt_embeds_mask": negative_prompt_embeds_mask[:, :max_sequence_length],
            })

        return results
    
    def encode_image(self, image: Union[Image.Image, torch.Tensor, List[torch.Tensor]], **kwargs) -> torch.Tensor:
        """Not needed for Qwen-Image text-to-image models."""
        pass

    def encode_video(self, video: Union[torch.Tensor, List[torch.Tensor]], **kwargs) -> torch.Tensor:
        """Not needed for Qwen-Image text-to-image models."""
        pass

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
    

    # ======================== Inference ========================
    def inference(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0, # Corresponds to `true_cfg_scale` in Qwen-Image-Pipeline.
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_ids: Optional[torch.LongTensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_ids: Optional[torch.LongTensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = {},
        max_sequence_length: int = 1024,
        compute_log_prob: bool = False,
        **kwargs,
    ):
        height = height or (self.training_args.resolution[0] if self.training else self.training_args.eval_args.resolution[0])
        width = width or (self.training_args.resolution[1] if self.training else self.training_args.eval_args.resolution[1])
        num_inference_steps = num_inference_steps or (self.training_args.num_inference_steps if self.training else self.training_args.eval_args.num_inference_steps)
        # Qwen-Image uses `true_cfg_scale` since it is not a guidance-distilled model.
        true_cfg_scale = guidance_scale or (self.training_args.guidance_scale if self.training else self.training_args.eval_args.guidance_scale)
        device = self.device
        dtype = self.transformer.dtype
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1.0 and has_neg_prompt

        if prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
            )
            prompt_ids = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
            prompt_embeds_mask = encoded["prompt_embeds_mask"]
            negative_prompt_ids = encoded.get("negative_prompt_ids", None)
            negative_prompt_embeds = encoded.get("negative_prompt_embeds", None)
            negative_prompt_embeds_mask = encoded.get("negative_prompt_embeds_mask", None)
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            prompt_embeds_mask = prompt_embeds_mask.to(device=device)
            if do_true_cfg:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
                negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device=device)

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
        batch_size = prompt_embeds.shape[0]
        
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        img_shapes = [[(1, height // self.pipeline.vae_scale_factor // 2, width // self.pipeline.vae_scale_factor // 2)]] * batch_size


        timesteps = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device,
        )

        # Truncate prompt embeddings and masks to max valid lengths in the batch
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        if txt_seq_lens:
            max_pos_len = max(txt_seq_lens)
            if prompt_embeds.shape[1] > max_pos_len:
                prompt_embeds = prompt_embeds[:, :max_pos_len]
                prompt_embeds_mask = prompt_embeds_mask[:, :max_pos_len]

        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if do_true_cfg and negative_prompt_embeds_mask is not None else None
        )
        if negative_txt_seq_lens:
            negative_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).long().tolist()
            max_neg_len = max(negative_txt_seq_lens)
            if negative_prompt_embeds.shape[1] > max_neg_len:
                negative_prompt_embeds = negative_prompt_embeds[:, :max_neg_len]
                negative_prompt_embeds_mask = negative_prompt_embeds_mask[:, :max_neg_len]

        guidance = None # Always None for Qwen-Image

        # Denoising loop
        all_latents = [latents]
        all_log_probs = [] if compute_log_prob else None
        
        for i, t in enumerate(timesteps):
            timestep = t.expand(batch_size).to(latents.dtype)
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)

            # Conditioned prediction
            with self.pipeline.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

            # Negative conditioned prediction
            if do_true_cfg:
                with self.pipeline.transformer.cache_context("uncond"):
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        encoder_hidden_states=negative_prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=negative_txt_seq_lens,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

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

        images = self.decode_latents(latents, height, width)

        samples = [
            QwenImageSample(
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps,
                height=height,
                width=width,
                image=images[b],
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b],
                negative_prompt=negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt,
                negative_prompt_ids=negative_prompt_ids[b] if negative_prompt_ids is not None else None,
                negative_prompt_embeds=negative_prompt_embeds[b] if negative_prompt_embeds is not None else None,
                img_shapes=img_shapes[b],
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_prob else None,
                extra_kwargs={
                    'guidance_scale': guidance_scale,
                    'attention_kwargs': attention_kwargs,
                },
            )
            for b in range(batch_size)
        ]


        self.pipeline.maybe_free_model_hooks()

        return samples
    

    # ======================== Forward (Training) ========================
    def forward(
        self,
        samples: List[QwenImageSample],
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        
        # 1. Extract data from samples
        batch_size = len(samples)
        device = self.device
        dtype = self.transformer.dtype
        # Assume all samples have the same guidance scale
        true_cfg_scale = [
            s.extra_kwargs.get('guidance_scale', self.training_args.guidance_scale)
            for s in samples
        ]
        # Assume all samples have the same attention kwargs
        attention_kwargs = samples[0].extra_kwargs.get('attention_kwargs', {})

        latents = torch.stack([s.all_latents[timestep_index] for s in samples], dim=0).to(device)
        next_latents = torch.stack([s.all_latents[timestep_index + 1] for s in samples], dim=0).to(device)
        timestep = torch.stack([s.timesteps[timestep_index] for s in samples], dim=0).to(device)

        prompt_embeds = torch.stack([s.prompt_embeds for s in samples], dim=0).to(device)
        prompt_embeds_mask = torch.stack([s.prompt_embeds_mask for s in samples], dim=0).to(device)
        # Assume all samples have negative prompt embeds if any sample has it
        has_neg_prompt = all(
            s.negative_prompt_embeds is not None and s.negative_prompt_embeds_mask is not None
            for s in samples
        )
        do_true_cfg = any(gs > 1.0 for gs in true_cfg_scale) and has_neg_prompt

        if do_true_cfg:
            negative_prompt_embeds = torch.stack([s.negative_prompt_embeds for s in samples], dim=0).to(device)
            negative_prompt_embeds_mask = torch.stack([s.negative_prompt_embeds_mask for s in samples], dim=0).to(device)    
    
        img_shapes = [s.img_shapes for s in samples]

        # Truncate prompt embeddings and masks to max valid lengths in the batch
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        if txt_seq_lens:
            max_pos_len = max(txt_seq_lens)
            if prompt_embeds.shape[1] > max_pos_len:
                prompt_embeds = prompt_embeds[:, :max_pos_len]
                prompt_embeds_mask = prompt_embeds_mask[:, :max_pos_len]

        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if do_true_cfg and negative_prompt_embeds_mask is not None else None
        )
        if negative_txt_seq_lens:
            negative_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).long().tolist()
            max_neg_len = max(negative_txt_seq_lens)
            if negative_prompt_embeds.shape[1] > max_neg_len:
                negative_prompt_embeds = negative_prompt_embeds[:, :max_neg_len]
                negative_prompt_embeds_mask = negative_prompt_embeds_mask[:, :max_neg_len]

        guidance = None # Always None for Qwen-Image

        # 2. Set scheduler timesteps
        _ = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=self.training_args.num_inference_steps,
            seq_len=latents.shape[1],
            device=device
        )

        # 3. Predict noise
        with self.transformer.cache_context("cond"):
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

        if do_true_cfg:
            with self.pipeline.transformer.cache_context("uncond"):
                neg_noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=negative_txt_seq_lens,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

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

