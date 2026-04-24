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

# src/flow_factory/models/sit_dit/sit_dit.py
"""
SiT / DiT adapter for Flow-Factory.

Architecture: class-conditional image generation on 256x256 (ImageNet-style).
The transformer is the SiT/DiT backbone; the VAE is an SD-VAE compatible encoder.

Timestep conventions
--------------------
- Flow-Factory scheduler: t ∈ [0, 1000], t=1000 ↔ pure noise
- SiT model:              t ∈ [0, 1],    t=0    ↔ noise,  t=1 ↔ data
- Conversion:             t_sit = 1 - t_ff / 1000

Sign convention for scheduler
------------------------------
Flow-Factory: next = x + noise_pred * dt,  dt = sigma_next - sigma_curr < 0
SiT velocity points toward data  →  noise_pred = -v_sit

CFG for class conditioning
--------------------------
  v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
  null label index = num_classes (extra embedding row)
"""
from __future__ import annotations

import os
import json
from typing import Union, List, Dict, Any, Optional, Tuple, Literal, ClassVar

import numpy as np
from dataclasses import dataclass
from PIL import Image

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from accelerate import Accelerator

from ...samples import T2ISample
from ..abc import BaseAdapter
from ...hparams import *
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SDESchedulerOutput,
    set_scheduler_timesteps,
)
from ...utils.base import filter_kwargs
from ...utils.trajectory_collector import (
    TrajectoryCollector,
    CallbackCollector,
    TrajectoryIndicesType,
    create_trajectory_collector,
    create_callback_collector,
)
from ...utils.logger_utils import setup_logger
from .models import SiT_models

logger = setup_logger(__name__)

# SD-VAE scale factor (same as Stable Diffusion)
_SCALE_FACTOR = 0.18215


# ---------------------------------------------------------------------------
#  Pipeline container
# ---------------------------------------------------------------------------

class SiTDiTPipeline:
    """
    Minimal pipeline container for SiT / DiT models.

    Attributes
    ----------
    transformer : SiT
        The SiT / DiT transformer.
    vae : AutoencoderKL
        Diffusers VAE for encoding / decoding latents.
    scheduler : FlowMatchEulerDiscreteScheduler
        Initial scheduler (replaced by the SDE variant during BaseAdapter init).
    new_mean : Optional[float]
        ReaLS latent normalization mean. None for standard SD-VAE latents.
    new_std : Optional[float]
        ReaLS latent normalization std. None for standard SD-VAE latents.
    """

    def __init__(
        self,
        transformer: nn.Module,
        vae: AutoencoderKL,
        scheduler: FlowMatchEulerDiscreteScheduler,
        new_mean: Optional[float] = None,
        new_std: Optional[float] = None,
    ):
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
        self.new_mean = new_mean
        self.new_std = new_std

    def maybe_free_model_hooks(self):
        """No-op — kept for API compatibility with BaseAdapter."""
        pass


# ---------------------------------------------------------------------------
#  Sample dataclass
# ---------------------------------------------------------------------------

@dataclass
class SiTDiTSample(T2ISample):
    """Output sample for SiT / DiT generation."""
    # Instance variables
    class_labels: Optional[torch.Tensor] = None   # () scalar integer class label (per sample)


# ---------------------------------------------------------------------------
#  Adapter
# ---------------------------------------------------------------------------

class SiTDiTAdapter(BaseAdapter):
    """
    Flow-Factory adapter for SiT (Scalable Interpolant Transformer) and
    DiT (Diffusion Transformer) class-conditional image generation.

    Config JSON format (``model_name_or_path/config.json``)
    --------------------------------------------------------
    {
        "model_name":  "SiT-XL/2",       // key in SiT_models dict
        "num_classes": 1000,              // number of classes (ImageNet: 1000)
        "learn_sigma": false,
        "input_size":  32,                // spatial size of latent (image/8)
        "in_channels": 4,                 // VAE latent channels
        "vae_path":    "stabilityai/sd-vae-ft-ema",  // HF id or local path
        "reals_vae":   false,             // true → use mean_std.json normalization
        "mean_std_path": null             // path to mean_std.json (if reals_vae=true)
    }

    Weights are loaded from ``model_name_or_path/pytorch_model.bin`` or
    ``model_name_or_path/model.safetensors``.
    """

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: SiTDiTPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

    # ========================== Pipeline Loading ==========================

    def load_pipeline(self) -> SiTDiTPipeline:
        path = self.model_args.model_name_or_path

        # -- Load model config ------------------------------------------------
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)

        model_name  = cfg.get("model_name", "SiT-XL/2")
        num_classes = cfg.get("num_classes", 1000)
        learn_sigma = cfg.get("learn_sigma", False)
        input_size  = cfg.get("input_size", 32)
        in_channels = cfg.get("in_channels", 4)
        vae_path    = cfg.get("vae_path", "stabilityai/sd-vae-ft-ema")
        reals_vae   = cfg.get("reals_vae", False)
        mean_std_path = cfg.get("mean_std_path", None)

        # Cache for decode_latents
        self._num_classes = num_classes
        self._reals_vae   = reals_vae

        # -- Load SiT / DiT transformer ---------------------------------------
        if model_name not in SiT_models:
            raise ValueError(
                f"Unknown model_name '{model_name}'. "
                f"Available: {list(SiT_models.keys())}"
            )
        transformer = SiT_models[model_name](
            input_size=input_size,
            in_channels=in_channels,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
        )

        # Try safetensors first, then pytorch_model.bin
        weights_path = None
        for fname in ("model.safetensors", "pytorch_model.bin"):
            candidate = os.path.join(path, fname)
            if os.path.isfile(candidate):
                weights_path = candidate
                break

        if weights_path is not None:
            if weights_path.endswith(".safetensors"):
                from safetensors.torch import load_file as load_safetensors
                state_dict = load_safetensors(weights_path)
            else:
                state_dict = torch.load(weights_path, map_location="cpu")

            # Support bare state dict or nested {"ema": ..., "model": ...}
            if isinstance(state_dict, dict):
                if "ema" in state_dict:
                    state_dict = state_dict["ema"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]

            missing, unexpected = transformer.load_state_dict(state_dict, strict=True)
            if missing:
                logger.warning(f"Missing keys when loading transformer: {missing[:5]}...")
            if unexpected:
                logger.warning(f"Unexpected keys when loading transformer: {unexpected[:5]}...")
            logger.info(f"Loaded transformer weights from {weights_path}")
        else:
            logger.warning(
                f"No transformer weights found in {path}. "
                "Starting from random initialization."
            )

        # -- Load VAE ---------------------------------------------------------
        vae = AutoencoderKL.from_pretrained(vae_path)

        # -- Load ReaLS latent normalization stats ----------------------------
        new_mean = None
        new_std  = None
        if reals_vae:
            if mean_std_path is None:
                mean_std_path = os.path.join(path, "mean_std.json")
            if os.path.isfile(mean_std_path):
                with open(mean_std_path, "r") as f:
                    stats = json.load(f)
                new_mean = float(stats["mean"])
                new_std  = float(stats["std"])
                logger.info(
                    f"Loaded ReaLS latent stats: mean={new_mean:.4f}, std={new_std:.4f}"
                )
            else:
                raise FileNotFoundError(
                    f"reals_vae=True but mean_std.json not found at {mean_std_path}. "
                    "Set 'mean_std_path' in config.json or place mean_std.json in "
                    "model_name_or_path/."
                )

        # -- Build scheduler (placeholder; replaced by load_scheduler) --------
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.0,
        )

        return SiTDiTPipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            new_mean=new_mean,
            new_std=new_std,
        )

    # ========================== Default Modules ==========================

    @property
    def default_target_modules(self) -> List[str]:
        """Default trainable modules for SiT / DiT (attention projections)."""
        return [
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "attn.qkv",
        ]

    # ======================== Encoding & Decoding ========================

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Parse class label(s) from string prompt(s).

        Each prompt is interpreted as an integer class index (e.g. "0" or "985").
        Returns dummy ``prompt_embeds`` / ``prompt_ids`` tensors so that the
        BaseAdapter machinery that expects these fields does not crash.
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        device = self.device
        labels = []
        for p in prompt:
            try:
                labels.append(int(p.strip()))
            except ValueError:
                logger.warning(
                    f"Could not parse class label from prompt '{p}'. Using 0."
                )
                labels.append(0)

        class_labels = torch.tensor(labels, dtype=torch.long, device=device)
        batch_size = len(labels)

        # Dummy tensors — kept for API compatibility; not used by forward()
        dummy_embeds = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
        dummy_ids    = torch.zeros(batch_size, 1, dtype=torch.long,    device=device)

        return {
            "class_labels":   class_labels,
            "prompt_embeds":  dummy_embeds,
            "prompt_ids":     dummy_ids,
        }

    def encode_image(self, images) -> None:
        """Not used for class-conditional generation."""
        pass

    def encode_video(self, videos) -> None:
        """Not used for SiT / DiT."""
        pass

    def decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: Literal["pil", "pt", "np"] = "pil",
    ) -> Union[List[Image.Image], torch.Tensor, np.ndarray]:
        """Decode latents to pixel images using the VAE."""
        vae   = self.pipeline.vae
        dtype = vae.dtype if hasattr(vae, 'dtype') else latents.dtype
        latents = latents.to(dtype=dtype)

        if self._reals_vae and self.pipeline.new_mean is not None:
            # ReaLS latent space: denormalize to raw VAE latents
            new_mean = self.pipeline.new_mean
            new_std  = self.pipeline.new_std
            z_vae = latents * new_std + new_mean
        else:
            # Standard SD-VAE: latents are scaled by SCALE_FACTOR during training
            z_vae = latents / _SCALE_FACTOR

        images = vae.decode(z_vae, return_dict=False)[0]

        # Map from [-1, 1] to [0, 1]
        images = (images / 2 + 0.5).clamp(0, 1)

        if output_type == "pt":
            return images  # (B, C, H, W)

        images_np = images.permute(0, 2, 3, 1).float().cpu().numpy()
        if output_type == "np":
            return images_np

        # PIL output
        images_uint8 = (images_np * 255).round().astype(np.uint8)
        return [Image.fromarray(img) for img in images_uint8]

    # ======================== Inference ========================

    @torch.no_grad()
    def inference(
        self,
        # Ordinary args
        prompt: Optional[Union[str, List[str]]] = None,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        # Pre-encoded prompt
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        # Other args
        compute_log_prob: bool = True,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
    ) -> List[SiTDiTSample]:
        """Execute generation and return SiTDiTSample objects."""

        device = self.device
        transformer = self.transformer
        dtype = self._inference_dtype

        # 1. Encode prompts if not provided
        if class_labels is None:
            encoded      = self.encode_prompt(prompt)
            class_labels = encoded["class_labels"]
            prompt_ids   = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
        else:
            class_labels = class_labels.to(device)

        batch_size = class_labels.shape[0]

        # 2. Prepare latents
        vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
        latent_height = height // vae_scale_factor
        latent_width  = width  // vae_scale_factor
        in_channels   = transformer.in_channels

        latents = torch.randn(
            batch_size, in_channels, latent_height, latent_width,
            dtype=dtype, device=device, generator=generator,
        )

        # 3. Set timesteps
        timesteps = set_scheduler_timesteps(
            scheduler=self.pipeline.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latent_height * latent_width,  # spatial token count (H*W)
            device=device,
        )

        # 4. Denoising loop
        latent_collector   = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latents            = self.cast_latents(latents, default_dtype=dtype)
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        for i, t in enumerate(timesteps):
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = (
                timesteps[i + 1]
                if i + 1 < len(timesteps)
                else torch.tensor(0, device=device)
            )
            return_kwargs = list(
                set(["next_latents", "log_prob", "noise_pred"] + extra_call_back_kwargs)
            )
            current_compute_log_prob = compute_log_prob and current_noise_level > 0

            output = self.forward(
                t=t,
                t_next=t_next,
                latents=latents,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=current_noise_level,
            )

            latents = self.cast_latents(output.next_latents, default_dtype=dtype)
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob:
                log_prob_collector.collect(output.log_prob, i)

            callback_collector.collect_step(
                step_idx=i,
                output=output,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": current_noise_level},
            )

        # 5. Decode images
        images = self.decode_latents(latents, height, width, output_type="pt")

        # 6. Build sample list
        extra_call_back_res = callback_collector.get_result()
        callback_index_map  = callback_collector.get_index_map()
        all_latents         = latent_collector.get_result()
        latent_index_map    = latent_collector.get_index_map()
        all_log_probs       = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map  = log_prob_collector.get_index_map() if compute_log_prob else None

        samples = [
            SiTDiTSample(
                # Denoising trajectory
                timesteps=timesteps,
                all_latents=(
                    torch.stack([lat[b] for lat in all_latents], dim=0)
                    if all_latents else None
                ),
                log_probs=(
                    torch.stack([lp[b] for lp in all_log_probs], dim=0)
                    if all_log_probs else None
                ),
                latent_index_map=latent_index_map,
                log_prob_index_map=log_prob_index_map,
                # Prompt / class label
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b] if prompt_embeds is not None else None,
                class_labels=class_labels[b],  # scalar per-sample label
                # Image & metadata
                height=height,
                width=width,
                image=images[b],
                # Extra kwargs
                extra_kwargs={
                    **{k: v[b] for k, v in extra_call_back_res.items()},
                    "callback_index_map": callback_index_map,
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()
        return samples

    # ======================== Forward (Training) ========================

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        # Class conditioning (used instead of prompt_embeds)
        class_labels: torch.Tensor,
        # Next timestep
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # CFG
        guidance_scale: Union[float, List[float]] = 4.0,
        noise_level: Optional[float] = None,
        # Other
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "noise_pred", "next_latents", "next_latents_mean", "std_dev_t", "dt", "log_prob"
        ],
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """
        One denoising step.

        Converts the Flow-Factory timestep to SiT convention, runs the
        transformer with optional CFG, negates the velocity to match the
        Flow-Factory scheduler sign convention, then calls scheduler.step().
        """
        device     = latents.device
        dtype      = latents.dtype
        batch_size = latents.shape[0]
        transformer = self.transformer

        # ------------------------------------------------------------------
        # 1.  Timestep conversion:  t_ff ∈ [0,1000]  →  t_sit ∈ [0,1]
        # ------------------------------------------------------------------
        sigma_t = t.float() / 1000.0          # sigma: 1 = pure noise
        t_sit   = 1.0 - sigma_t               # SiT:   0 = noise, 1 = data
        t_sit_batch = t_sit.expand(batch_size).to(dtype=dtype, device=device)

        # ------------------------------------------------------------------
        # 2.  Transformer forward pass (with optional CFG)
        # ------------------------------------------------------------------
        do_cfg = (guidance_scale > 1.0) and (class_labels is not None)

        if do_cfg:
            # Concatenate conditional and unconditional batches
            null_labels   = torch.full_like(class_labels, transformer.y_embedder.num_classes)
            labels_double = torch.cat([class_labels, null_labels], dim=0)
            latents_double = torch.cat([latents, latents], dim=0)
            t_sit_double   = t_sit_batch.repeat(2)

            v_double = transformer(latents_double, t_sit_double, labels_double)
            v_cond, v_uncond = v_double.chunk(2, dim=0)
            v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v_pred = transformer(latents, t_sit_batch, class_labels)

        # ------------------------------------------------------------------
        # 3.  Sign flip: Flow-Factory uses next = x + noise_pred * dt
        #                with dt < 0 (sigma decreasing).
        #                SiT velocity points toward data, so we negate.
        # ------------------------------------------------------------------
        noise_pred = -v_pred

        # ------------------------------------------------------------------
        # 4.  Scheduler step
        # ------------------------------------------------------------------
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
