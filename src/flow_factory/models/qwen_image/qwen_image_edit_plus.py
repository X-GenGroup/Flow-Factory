# src/flow_factory/models/qwen_image/qwen_image_edit_plus.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
import logging
import math
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

from ..adapter import BaseAdapter, BaseSample
from ...hparams import *
from ...scheduler import SDESchedulerOutput, set_scheduler_timesteps
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger
from ...utils.base import (
    filter_kwargs,
    is_pil_image_batch_list,
    is_pil_image_list,
    tensor_to_pil_image,
    tensor_list_to_pil_image,
    numpy_list_to_pil_image,
    numpy_to_pil_image,
    pil_image_to_tensor
)

logger = setup_logger(__name__)

CONDITION_IMAGE_SIZE = (1024, 1024)

QwenImageEditPlusImageInput = Union[
    Image.Image,
    np.ndarray,
    torch.Tensor,
    List[Image.Image],
    List[np.ndarray],
    List[torch.Tensor]
]

@dataclass
class QwenImageEditPlusSample(BaseSample):
    """Output class for Qwen-Image-Edit Plus model"""
    condition_images : Optional[Union[List[Image.Image], Image.Image]] = None
    prompt_embeds_mask : Optional[torch.FloatTensor] = None
    negative_prompt_embeds_mask : Optional[torch.FloatTensor] = None
    img_shapes : Optional[List[Tuple[int, int, int]]] = None
    image_latents : Optional[torch.Tensor] = None

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height

class QwenImageEditPlusAdapter(BaseAdapter):
    """Adapter for Qwen-Image-Edit Plus text-to-image models."""
    
    def __init__(self, config: Arguments, accelerator : Accelerator):
        super().__init__(config, accelerator)
        self._warned_cfg_no_neg_prompt = False
        self._warned_no_cfg = False

        self._has_warned_inference_fallback = False
        self._has_warned_forward_fallback = False
        self._has_warned_preprocess_fallback = False
    
    def load_pipeline(self) -> QwenImageEditPlusPipeline:
        return QwenImageEditPlusPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False
        )
    
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Qwen-Image transformer."""
        return [
            # Attention
            "to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
            # MLP
            "net.0.proj", "net.2"
        ]
    
    # ================================= Encoding and Decoding Methods ================================= #

    # ---------------------------------- Text Encoding ---------------------------------- #

    def _standardize_image_input(
        self,
        images: QwenImageEditPlusImageInput,
        output_type: Literal['pil', 'pt', 'np'] = 'pil',
    ):
        """
        Standardize image input to desired output type.
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(images, torch.Tensor):
            if output_type == 'pil':
                images = tensor_to_pil_image(images)
            elif output_type == 'np':
                images = images.cpu().numpy()
        elif isinstance(images, np.ndarray):
            if output_type == 'pil':
                images = numpy_to_pil_image(images)
            elif output_type == 'pt':
                images = torch.from_numpy(images)
        elif isinstance(images, list):
            if isinstance(images[0], torch.Tensor):
                if output_type == 'pil':
                    images = tensor_list_to_pil_image(images)
                elif output_type == 'np':
                    # From tensor's [0, 1] to numpy's [0, 255]
                    if images[0].max() <= 1.0:
                        images = [ (img.cpu().numpy() * 255).astype(np.uint8) for img in images ]
                    else:
                        images = [ img.cpu().numpy().astype(np.uint8) for img in images ]
            elif isinstance(images[0], np.ndarray):
                if output_type == 'pil':
                    images = numpy_list_to_pil_image(images)
                elif output_type == 'pt':
                    # From numpy's [0, 255] to tensor's [0, 1]
                    if images.max() > 1.0:
                        images = images.astype(np.float32) / 255.0
                    images = torch.from_numpy(images)
            elif isinstance(images[0], Image.Image):
                if output_type == 'np':
                    images = [np.array(img) for img in images]
                elif output_type == 'pt':
                    images = [pil_image_to_tensor(img)[0] for img in images]
            else:
                raise ValueError(f'Unsupported image type in list: {type(images[0])}.')
        else:
            raise ValueError(f'Unsupported image input type: {type(images)}.')
        return images

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        images: Optional[QwenImageEditPlusImageInput] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 1024,
    ):
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        images = self._standardize_image_input(images, output_type='pil') if images is not None else None
        if isinstance(images, list):
            base_img_prompt = ""
            for i, img in enumerate(images):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif images is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = self.pipeline.prompt_template_encode

        drop_idx = self.pipeline.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]

        model_inputs = self.pipeline.processor(
            text=txt,
            images=images,
            padding=True,
            max_length=max_sequence_length + drop_idx,
            return_tensors="pt",
        ).to(device)
        input_ids = model_inputs.input_ids

        outputs = self.pipeline.text_encoder(
            input_ids=input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self.pipeline._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
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
        images : Optional[QwenImageEditPlusImageInput] = None,
        max_sequence_length: int = 1024,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict[str, Union[torch.LongTensor, torch.Tensor]]:
        """Encode text prompts using the pipeline's text encoder."""

        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        # Encode positive prompt
        prompt_ids, prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
            prompt=prompt,
            images=images,
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
                images=images,
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
    
    # ---------------------------------------- Image Encoding ---------------------------------- #
    def encode_image(
            self,
            images: QwenImageEditPlusImageInput,
            condition_image_size : Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
            **kwargs
        ) -> Dict[str, List[Union[torch.Tensor, Tuple[int, int]]]]:
        """
        Encode input images into latent representations using the VAE encoder.
        Args:
            images: `QwenImageEditPlusImageInput`
                - Single conditioning image (PIL Image) or a list of conditioning images.
                - Each image will be resized to fit within `condition_image_size` while maintaining aspect ratio.
            resolution: `Union[int, Tuple[int, int]]`
                - Maximum resolution for VAE images. If int, will be used for both height and width.
            condition_image_size: `Union[int, Tuple[int, int]]`
                - Maximum size for conditioning images. If int, will be used for both height and width.
        Returns:
            Dictionary containing:
                - "condition_images": List of resized conditioning images.
                - "condition_image_sizes": List of sizes for conditioning images.
                - "vae_images": List of preprocessed images for VAE encoding.
                - "vae_image_sizes": List of sizes for VAE images.
        """

        if isinstance(condition_image_size, int):
            condition_image_size = (condition_image_size, condition_image_size)

        condition_image_max_area = condition_image_size[0] * condition_image_size[1]

        condition_image_sizes = []
        condition_images = []
        vae_image_sizes = []
        vae_images = []

        if not isinstance(images, list):
            images = [images]

        for img in images:
            image_width, image_height = img.size
            condition_width, condition_height = calculate_dimensions(
                condition_image_max_area, image_width / image_height
            )
            vae_width, vae_height = calculate_dimensions(condition_image_max_area, image_width / image_height)
            condition_image_sizes.append((condition_width, condition_height))
            vae_image_sizes.append((vae_width, vae_height))
            condition_image = self.pipeline.image_processor.resize(img, condition_height, condition_width)
            condition_image = self._standardize_image_input(condition_image, output_type='pt')[0] # Convert to tensor
            condition_images.append(condition_image)
            vae_images.append(self.pipeline.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))


        return {
            "condition_images": condition_images,
            "condition_image_sizes": condition_image_sizes,
            "vae_images": vae_images,
            "vae_image_sizes": vae_image_sizes,
        }

    # ---------------------------------------- Video Encoding ---------------------------------- #
    def encode_video(self, videos: Union[torch.Tensor, List[torch.Tensor]], **kwargs) -> torch.Tensor:
        """Not needed for Qwen-Image-Edit models."""
        pass

    # ---------------------------------------- Image Decoding ---------------------------------- #
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

    # ========================Preprocessing ========================
    def preprocess_func(
        self,
        prompt: List[str],
        images: Optional[Union[List[Optional[Image.Image]], List[List[Optional[Image.Image]]]]] = None,
        negative_prompt: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, List[Any]]:
        """Preprocess data samples for Qwen-Image-Edit Plus model training or evaluation.

        Args:
            prompt (List[str]): A Batch of text prompts.
            images (Optional[Union[List[Optional[Image.Image]], List[List[Optional[Image.Image]]]]]): 
                A batch of conditioning images. Each element can be a PIL Image or a list of PIL Images.                
        """
        batch_size = len(prompt)
        if images is None:
            images = [None] * batch_size
        if negative_prompt is None:
            negative_prompt = [None] * batch_size

        # assert all(isinstance(imgs, list) or imgs is None for imgs in images), \
        #     "Each element in images must be a list of PIL Images or None."

        dtype = self.pipeline.vae.dtype
        device = self.pipeline.vae.device

        results = defaultdict(list)

        for p, neg_p, imgs in zip(prompt, negative_prompt, images):
            input_kwargs = kwargs.copy()
            input_kwargs = {'device': device, 'dtype': dtype}
            encoded_prompt = self.encode_prompt(
                prompt=p,
                negative_prompt=neg_p,
                images=imgs,
                **filter_kwargs(self.encode_prompt, **input_kwargs)
            )
            encoded_images = self.encode_image(
                images=imgs,
                **filter_kwargs(self.encode_image, **input_kwargs)
            )
            for k, v in encoded_prompt.items():
                results[k].append(v)

            for k, v in encoded_images.items():
                results[k].append(v)

        return results


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
    
        # ======================== Sampling / Inference ========================

    # ================================ Inference ================================ #

    # Handle one sample
    @torch.no_grad()
    def _inference(
        self,
        # Ordinary arguments
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0, # Corresponds to `true_cfg_scale` in Qwen-Image-Edit-Plus-Pipeline.
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
        
        # Image encoding arguments
        condition_images: Optional[Union[QwenImageEditPlusImageInput, List[QwenImageEditPlusImageInput]]] = None,
        condition_image_sizes: Optional[List[Tuple[int, int]]] = None,
        vae_images: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        vae_image_sizes: Optional[List[Tuple[int, int]]] = None,

        # Other arguments
        attention_kwargs: Optional[Dict[str, Any]] = {},
        max_sequence_length: int = 1024,
        compute_log_prob: bool = False,

        # Callback arguments
        extra_call_back_kwargs: List[str] = [],
        **kwargs,
    ):
        """Generate images using Qwen-Image-Edit Plus model."""
        # 1. Set up

        # Determine height and width
        image_size = images[-1].size if isinstance(images, list) else images.size
        calculated_width, calculated_height = calculate_dimensions(height * width, image_size[0] / image_size[1])
        if calculated_height != height or calculated_width != width:
            logger.warning(
                f"Input image has aspect ratio {image_size[0] / image_size[1]:.2f}, "
                f"which differs from target aspect ratio {width / height:.2f}. "
                f"Adjusting output dimensions to ({calculated_width}, {calculated_height})."
            )

        height = calculated_height
        width = calculated_width
        multiple_of = self.pipeline.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # cfg and others
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

        # 2. Preprocess images
        if (
            images is not None
            and (condition_images is None or vae_images is None or condition_image_sizes is None or vae_image_sizes is None)
        ):
            encoded_images = self.encode_image(
                images=images,
                resolution=(height, width),
                condition_image_size=self.config.training_args.extra_kwargs.get('condition_image_size', CONDITION_IMAGE_SIZE),
                device=device,
                dtype=dtype,
            )
            condition_images = encoded_images["condition_images"]
            condition_image_sizes = encoded_images["condition_image_sizes"]
            vae_images = encoded_images["vae_images"]
            vae_image_sizes = encoded_images["vae_image_sizes"]
        else:
            condition_images = self._standardize_image_input(condition_images, output_type='pt') if condition_images is not None else None
            vae_images = vae_images.to(device)
        
        # 3. Encode prompts
        if prompt_embeds is None or prompt_embeds_mask is None:
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                images=images,
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

        batch_size = prompt_embeds.shape[0]

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

        
        # 4. Prepare latents
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, image_latents = self.pipeline.prepare_latents(
            images=vae_images,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )
        img_shapes = [
            [
                (1, height // self.pipeline.vae_scale_factor // 2, width // self.pipeline.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.pipeline.vae_scale_factor // 2, vae_width // self.pipeline.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size


        # 5. Set scheduler timesteps
        timesteps = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device,
        )

        guidance = None # Always None for Qwen-Image-Edit Plus


        # 6. Denoising loop
        all_latents = [latents]
        all_log_probs = [] if compute_log_prob else None
        extra_call_back_res = defaultdict(list)

        for i, t in enumerate(timesteps):
            timestep = t.expand(batch_size).to(latents.dtype)
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)

            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1) # Concatenate along sequence dimension

            # Conditioned prediction
            with self.pipeline.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

            # Negative conditioned prediction
            if do_true_cfg:
                with self.pipeline.transformer.cache_context("uncond"):
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        encoder_hidden_states=negative_prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=negative_txt_seq_lens,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
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

        # 7. Post-process results
        generated_images = self.decode_latents(latents, height, width)

        # Transpose `extra_call_back_res` tensors to have batch dimension first
        # (T, B, ...) -> (B, T, ...)
        extra_call_back_res = {
            k: torch.stack(v, dim=1)
            if isinstance(v[0], torch.Tensor) else v
            for k, v in extra_call_back_res.items()
        }

        samples = [
            QwenImageEditPlusSample(
                # Denoising trajectory
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
                timesteps=timesteps,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if compute_log_prob else None,
                
                # Generated image
                image=generated_images[b],

                # Condition images
                image_latents=image_latents[b] if image_latents is not None else None,
                condition_images=condition_images[b] if condition_images is not None else None,
                img_shapes=img_shapes[b],

                # Prompt
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b] if prompt_embeds is not None else None,
                prompt_embeds_mask=prompt_embeds_mask[b] if prompt_embeds_mask is not None else None,

                # Negative Prompt
                negative_prompt=negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt,
                negative_prompt_ids=negative_prompt_ids[b] if negative_prompt_ids is not None else None,
                negative_prompt_embeds=negative_prompt_embeds[b] if negative_prompt_embeds is not None else None,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask[b] if negative_prompt_embeds_mask is not None else None,

                # Extra kwargs
                extra_kwargs={
                    'guidance_scale': true_cfg_scale,
                    'attention_kwargs': attention_kwargs,
                }
            )
            for b in range(batch_size)
        ]

        return samples

    @torch.no_grad()
    def inference(
        self,
        # Ordinary arguments
        images: Optional[Union[List[Image.Image], List[List[Image.Image]]]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0, # Corresponds to `true_cfg_scale` in Qwen-Image-Edit-Plus-Pipeline.
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
        
        # Image encoding arguments
        condition_images: Optional[Union[List[Image.Image], List[List[Image.Image]]]] = None, # A batch of condition image lists
        condition_image_sizes: Optional[Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]]] = None, # A batch of condition image size lists
        vae_images: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]] = None, # A batch of VAE image lists
        vae_image_sizes: Optional[Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]]] = None, # A batch of VAE image size lists

        # Other arguments
        attention_kwargs: Optional[Dict[str, Any]] = {},
        max_sequence_length: int = 1024,
        compute_log_prob: bool = False,
        **kwargs,
    ):
        # Check if images are given as a batch of image lists
        if (
            images is not None 
            and isinstance(images, list)
            and any(isinstance(i, list) for i in images) # A batch of image lists
        ):
            if not self._has_warned_inference_fallback:
                logger.warning(
                    "Qwen-Image-Edit-Plus does not support batch inference with varying condition images per sample. "
                    "Falling back to single-sample inference. This warning will only appear once."
                )
                self._has_warned_inference_fallback = True
            # Process each sample individually by calling _inference
            all_samples = []
            if not isinstance(prompt, list):
                prompt = [prompt] * len(images)

            if not isinstance(negative_prompt, list):
                negative_prompt = [negative_prompt] * len(images)
            
            prompt_ids = prompt_ids or [None] * len(images)
            prompt_embeds = prompt_embeds or [None] * len(images)
            prompt_embeds_mask = prompt_embeds_mask or [None] * len(images)
            negative_prompt_ids = negative_prompt_ids or [None] * len(images)
            negative_prompt_embeds = negative_prompt_embeds or [None] * len(images)
            negative_prompt_embeds_mask = negative_prompt_embeds_mask or [None] * len(images)
            
            condition_images = condition_images or [None] * len(images)
            condition_image_sizes = condition_image_sizes or [None] * len(images)
            vae_images = vae_images or [None] * len(images)
            vae_image_sizes = vae_image_sizes or [None] * len(images)

            for (
                img_list, p, neg_p,
                p_ids, p_embeds, p_embeds_mask,
                neg_p_ids, neg_p_embeds, neg_p_embeds_mask,
                cond_imgs, cond_img_sizes, vae_imgs, vae_img_sizes
            ) in zip(
                images, prompt, negative_prompt,
                prompt_ids, prompt_embeds, prompt_embeds_mask,
                negative_prompt_ids, negative_prompt_embeds, negative_prompt_embeds_mask,
                condition_images, condition_image_sizes, vae_images, vae_image_sizes,
            ):
                sample = self._inference(
                    # Ordinary arguments
                    images=img_list,
                    prompt=p,
                    negative_prompt=neg_p,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                    # Prompt encoding arguments
                    prompt_ids=p_ids,
                    prompt_embeds=p_embeds,
                    prompt_embeds_mask=p_embeds_mask,
                    # Negative prompt encoding arguments
                    negative_prompt_ids=neg_p_ids,
                    negative_prompt_embeds=neg_p_embeds,
                    negative_prompt_embeds_mask=neg_p_embeds_mask,
                    # Conditioning images encoding arguments
                    condition_images=cond_imgs,
                    condition_image_sizes=cond_img_sizes,
                    vae_images=vae_imgs,
                    vae_image_sizes=vae_img_sizes,
                    # Other arguments
                    attention_kwargs=attention_kwargs,
                    max_sequence_length=max_sequence_length,
                    compute_log_prob=compute_log_prob,
                    **kwargs,
                )
                all_samples.extend(sample)
            return all_samples
        else:
            # Directly call _inference for batch processing, the condition images are shared across the batch
            return self._inference(
                # Ordinary arguments
                images=images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                # Prompt encoding arguments
                prompt_ids=prompt_ids,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                # Negative prompt encoding arguments
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                # Conditioning images encoding arguments
                condition_images=condition_images,
                vae_images=vae_images,
                condition_image_sizes=condition_image_sizes,
                vae_image_sizes=vae_image_sizes,
                # Other arguments
                attention_kwargs=attention_kwargs,
                max_sequence_length=max_sequence_length,
                compute_log_prob=compute_log_prob,
                **kwargs,
            )



    # ======================== Forward for training ========================

    def _t2i_forward(
        self,
        samples: Union[QwenImageEditPlusSample, List[QwenImageEditPlusSample]],
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> SDESchedulerOutput:
        """Forward method for text-to-image generation with Qwen-Image-Edit Plus model."""
        """
            TODO in the future if needed.
        """
        pass

    def _i2i_forward(
        self,
        sample: QwenImageEditPlusSample,
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> SDESchedulerOutput:
        """Forward method for image-to-image editing with Qwen-Image-Edit Plus model."""
        # 1. Extract data from samples
        batch_size = 1
        device = self.device
        dtype = self.pipeline.transformer.dtype
        # Assume all samples have the same guidance scale
        true_cfg_scale = sample.extra_kwargs.get('guidance_scale', self.training_args.guidance_scale)
        # Assume all samples have the same attention kwargs
        attention_kwargs = sample.extra_kwargs.get('attention_kwargs', {})

        latents = sample.all_latents[timestep_index].unsqueeze(0).to(device)
        next_latents = sample.all_latents[timestep_index + 1].unsqueeze(0).to(device)
        timestep = sample.timesteps[timestep_index].unsqueeze(0).to(device)
        t = timestep[0]
        num_inference_steps = len(sample.timesteps)

        prompt_embeds = [sample.prompt_embeds]
        prompt_embeds_mask = [sample.prompt_embeds_mask]
        image_latents = sample.image_latents.unsqueeze(0) if sample.image_latents is not None else None

        negative_prompt_embeds = sample.negative_prompt_embeds
        negative_prompt_embeds_mask = sample.negative_prompt_embeds_mask

        # Assume all samples have negative prompt embeds if any sample has it
        has_neg_prompt = (
            negative_prompt_embeds is not None
            and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1.0 and has_neg_prompt

        if do_true_cfg:
            negative_prompt_embeds = [sample.negative_prompt_embeds]
            negative_prompt_embeds_mask = [sample.negative_prompt_embeds_mask]
    
        img_shapes = [sample.img_shapes]

        # Get txt_seq_lens
        txt_seq_lens, _, prompt_embeds, prompt_embeds_mask = self._pad_batch_prompt(
            prompt_embeds_mask=prompt_embeds_mask,
            prompt_ids=None, # prompt_ids are not needed for forward
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

        guidance = None # Always None for Qwen-Image, use `true_cfg_scale` instead

        # 2. Set scheduler timesteps
        _ = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device
        )

        # 3. Compute model output
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1) # Concatenate along sequence dimension

        # Conditioned prediction
        with self.pipeline.transformer.cache_context("cond"):
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]

        # Negative conditioned prediction
        if do_true_cfg:
            with self.pipeline.transformer.cache_context("uncond"):
                neg_noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=negative_txt_seq_lens,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

        # Scheduler step
        step_kwargs = filter_kwargs(self.scheduler.step, **kwargs)
        output = self.scheduler.step(
            noise_pred=noise_pred,
            timestep=timestep,
            latents=latents,
            compute_log_prob=compute_log_prob,
            **step_kwargs,
        )

        return output

    def forward(
        self,
        samples: List[QwenImageEditPlusSample],
        timestep_index : int,
        compute_log_prob: bool = True,
        **kwargs,
    ) -> SDESchedulerOutput:
        is_i2i = any(
            s.image_latents is not None
            for s in samples
        )
        if is_i2i:
            outputs = []
            for s in samples:
                out = self._i2i_forward(
                    sample=s,
                    timestep_index=timestep_index,
                    compute_log_prob=compute_log_prob,
                    **kwargs,
                )
                outputs.append(out)
            
            outputs = [o.to_dict() for o in outputs]
            # Concatenate outputs
            output = SDESchedulerOutput.from_dict({
                k: torch.cat([o[k] for o in outputs], dim=0)
                for k in outputs[0].keys()
            })
        else:
            # It is weird that a Edit model can inference without condition images, but we allow it for flexibility.
            output = self._t2i_forward(
                samples=samples,
                timestep_index=timestep_index,
                compute_log_prob=compute_log_prob,
                **kwargs,
            )

        return output