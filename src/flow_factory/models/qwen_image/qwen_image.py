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

# src/flow_factory/models/qwen_image/qwen_image.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from collections import defaultdict

from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from accelerate import Accelerator

from ..adapter import BaseAdapter
from ..samples import T2ISample
from ...hparams import *
from ...scheduler import SDESchedulerOutput, set_scheduler_timesteps, FlowMatchEulerDiscreteSDEScheduler
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class QwenImageSample(T2ISample):
    """Output class for Qwen-Image models."""
    prompt_embeds_mask : Optional[torch.FloatTensor] = None
    negative_prompt_embeds_mask : Optional[torch.FloatTensor] = None
    img_shapes : Optional[List[Tuple[int, int, int]]] = None

class QwenImageAdapter(BaseAdapter):
    """Adapter for Qwen-Image text-to-image models."""
    
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: QwenImagePipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

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
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0", # Image Stream / Main Stream
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out", # Text Stream
            # FFNs
            "img_mlp.net.0.proj", "img_mlp.net.2.proj",
            "txt_mlp.net.0.proj", "txt_mlp.net.2.proj"
        ]
    
    # ======================== Encoding & Decoding ========================
    
    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        max_sequence_length = max_sequence_length or self.pipeline.tokenizer_max_length

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.pipeline.prompt_template_encode
        drop_idx = self.pipeline.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.pipeline.tokenizer(
            txt, max_length=max_sequence_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        input_ids = txt_tokens.input_ids
        encoder_hidden_states = self.text_encoder(
            input_ids=input_ids,
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

        input_ids = input_ids[:, drop_idx:] # Extract only user input ids
        return input_ids, prompt_embeds, encoder_attention_mask
    
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
        prompt_ids, prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
            prompt=prompt,
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
    
    # ======================== Padding Utilities ========================
    def _standardize_data(
        self,
        data : Union[None, torch.Tensor, List[torch.Tensor]],
        padding_value : Union[int, float],
        device: Optional[torch.device] = None,
        max_len: Optional[int] = None,
    ):
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
    ) -> Tuple[List[int], Optional[torch.LongTensor], Optional[torch.Tensor], Optional[torch.LongTensor]]:
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

    # ======================== Inference ========================
    @torch.no_grad()
    def inference(
        self,
        # Ordinary arguments
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0, # Corresponds to `true_cfg_scale` in Qwen-Image-Pipeline.
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

        # Other arguments
        attention_kwargs: Optional[Dict[str, Any]] = {},
        max_sequence_length: int = 1024,
        compute_log_prob: bool = False,

        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
        **kwargs,
    ):
        # 1. Prepare inputs
        # Qwen-Image uses `true_cfg_scale` since it is not a guidance-distilled model.
        true_cfg_scale = guidance_scale or (self.eval_args.guidance_scale if self.mode == 'eval' else self.training_args.guidance_scale)
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

        # 2. Get prompt embeddings
        if prompt_embeds is None or prompt_embeds_mask is None:
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
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
            pass

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

        # 3. Prepare latents
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

        # 4. Set scheduler timesteps
        timesteps = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device,
        )

        guidance = None # Always None for Qwen-Image

        # 5. Denoising loop
        all_latents = [latents]
        all_log_probs = [] if compute_log_prob else None
        extra_call_back_res = defaultdict(list)
        
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

        # 6. Decode latents to images
        decoded_images = self.decode_latents(latents, height, width)

        # 7. Prepare output samples

        # Transpose `extra_call_back_res` tensors to have batch dimension first
        # (T, B, ...) -> (B, T, ...)
        extra_call_back_res = {
            k: torch.stack(v, dim=1)
            if isinstance(v[0], torch.Tensor) else v
            for k, v in extra_call_back_res.items()
        }

        samples = [
            QwenImageSample(
                # Denoising trajectory
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_prob else None,
                
                # Generated image & metadata
                height=height,
                width=width,
                image=decoded_images[b],
                img_shapes=img_shapes[b],

                # Prompt info
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b],
                prompt_embeds=prompt_embeds[b],
                prompt_embeds_mask=prompt_embeds_mask[b],
                
                # Negative prompt info
                negative_prompt=negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt,
                negative_prompt_ids=negative_prompt_ids[b] if negative_prompt_ids is not None else None,
                negative_prompt_embeds=negative_prompt_embeds[b] if negative_prompt_embeds is not None else None,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask[b] if negative_prompt_embeds_mask is not None else None,

                # Extra kwargs
                extra_kwargs={
                    'guidance_scale': true_cfg_scale,
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
        samples: List[QwenImageSample],
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> SDESchedulerOutput:
        
        # 1. Extract data from samples
        batch_size = len(samples)
        device = self.device
        dtype = self.pipeline.transformer.dtype
        # Assume all samples have the same guidance scale
        true_cfg_scale = samples[0].extra_kwargs.get('guidance_scale', self.training_args.guidance_scale)
        # Assume all samples have the same attention kwargs
        attention_kwargs = samples[0].extra_kwargs.get('attention_kwargs', {})

        latents = torch.stack([s.all_latents[timestep_index] for s in samples], dim=0).to(device)
        next_latents = torch.stack([s.all_latents[timestep_index + 1] for s in samples], dim=0).to(device)
        timestep = torch.stack([s.timesteps[timestep_index] for s in samples], dim=0).to(device)
        num_inference_steps = len(samples[0].timesteps)

        prompt_embeds = [s.prompt_embeds for s in samples]
        prompt_embeds_mask = [s.prompt_embeds_mask for s in samples]
        # Assume all samples have negative prompt embeds if any sample has it
        has_neg_prompt = all(
            s.negative_prompt_embeds is not None and s.negative_prompt_embeds_mask is not None
            for s in samples
        )
        do_true_cfg = true_cfg_scale > 1.0 and has_neg_prompt

        if do_true_cfg:
            negative_prompt_embeds = [s.negative_prompt_embeds for s in samples]
            negative_prompt_embeds_mask = [s.negative_prompt_embeds_mask for s in samples]
    
        img_shapes = [s.img_shapes for s in samples]

        # Truncate prompt embeddings and masks to max valid lengths in the batch
        txt_seq_lens, _, prompt_embeds, prompt_embeds_mask = self._pad_batch_prompt(
            prompt_embeds_mask=prompt_embeds_mask,
            prompt_ids=None,
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
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        guidance = None # Always None for Qwen-Image

        # 2. Set scheduler timesteps
        _ = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device
        )

        # 3. Predict noise
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
            noise_pred=noise_pred,
            timestep=timestep,
            latents=latents,
            next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            **step_kwargs,
        )

        return output

