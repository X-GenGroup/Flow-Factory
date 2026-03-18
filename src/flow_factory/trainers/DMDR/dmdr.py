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

# src/flow_factory/trainers/dmdr.py
"""
DMDR (Distribution Matching Distillation Meets RL) Trainer for existing T2I adapters.
Uses adapter.forward(t, latents, prompt_embeds, ..., return_kwargs=['noise_pred']) and
adapter.preprocess_func(prompt). Dual model: generator = adapter, guidance = deepcopy(adapter).
"""
import copy
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from functools import partial

from ...models.abc import BaseAdapter
from ...trainers.abc import BaseTrainer
from .dmdr_utils import (
    sample_continue,
    sample_discrete,
    get_sample,
    v2x0_sampler_adapter,
    DINOv2ProcessorWithGrad,
)
from ...utils.base import filter_kwargs, create_generator_by_prompt
from ...samples import BaseSample
from ...logger.formatting import LogImage
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)

try:
    import tqdm as tqdm_
    tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)
except Exception:
    tqdm = lambda x, **kw: x


class DMDRTrainer(BaseTrainer):
    """
    DMDR Trainer for Flow-Factory T2I adapters (Flux, SD3, etc.).
    - Generator = main adapter; guidance = deep copy of adapter (same forward interface).
    - Data: dataloader yields batch with pre-encoded embeddings (when preprocessing enabled)
      or raw prompts (encoded on-the-fly via adapter.preprocess_func).
    - Backward sampler and velocity prediction use adapter.forward (noise_pred as velocity).
    """

    def __init__(self, **kwargs):
        self._dmdr_validate_adapter(kwargs["adapter"])
        super().__init__(**kwargs)
        self.train_batch_size = self.training_args.per_device_batch_size
        self.num_steps = getattr(self.training_args, "dmdr_num_steps", 4)
        self.shift = getattr(self.training_args, "dmdr_shift", 1.0)
        self.ratio_update = getattr(self.training_args, "ratio_update", 1)
        self.cold_start_iter = getattr(self.training_args, "cold_start_iter", 0)
        self.guidance_scale = getattr(self.training_args, "guidance_scale", 0.0)
        self.dynamic_step = getattr(self.training_args, "dynamic_step", 1000)
        self.image_log_steps = 1

    # =========================== Initialization ============================
    def _dmdr_validate_adapter(self, adapter):
        if not hasattr(adapter, "forward") or not callable(adapter.forward):
            raise NotImplementedError("DMDR requires adapter.forward(t, latents, prompt_embeds, ...).")
        if not hasattr(adapter, "scheduler"):
            raise NotImplementedError("DMDR requires adapter.scheduler for timesteps.")

    def _initialization(self):
        if self.adapter._is_fsdp_cpu_efficient_loading():
            logger.info("FSDP CPU Efficient Loading detected. Synchronizing frozen components...")
            self._synchronize_frozen_components()

        self.dataloader, self.test_dataloader = self._init_dataloader()

        guidance_adapter = copy.deepcopy(self.adapter)
        lr_gui = getattr(self.training_args, "learning_rate_gui", None) or self.training_args.learning_rate
        self.optimizer = torch.optim.AdamW(
            self.adapter.get_trainable_parameters(),
            lr=self.training_args.learning_rate,
            betas=self.training_args.adam_betas,
            weight_decay=self.training_args.adam_weight_decay,
            eps=self.training_args.adam_epsilon,
        )
        self.optimizer_gui = torch.optim.AdamW(
            guidance_adapter.get_trainable_parameters(),
            lr=lr_gui,
            betas=self.training_args.adam_betas,
            weight_decay=0.0,
            eps=self.training_args.adam_epsilon,
        )

        trainable_module_names = list(self.adapter.target_module_map.keys())
        trainable_modules = [
            getattr(self.adapter, name)
            for name in trainable_module_names
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None
        ]
        guidance_modules = [
            getattr(guidance_adapter, name)
            for name in trainable_module_names
            if hasattr(guidance_adapter, name) and getattr(guidance_adapter, name) is not None
        ]
        to_prepare = trainable_modules + guidance_modules + [self.optimizer, self.optimizer_gui]
        if self.test_dataloader is not None:
            to_prepare.append(self.test_dataloader)

        prepared = self.accelerator.prepare(*to_prepare)
        n_gen = len(trainable_modules)
        n_gui = len(guidance_modules)
        for i, name in enumerate(trainable_module_names):
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None:
                self.adapter.set_component(name, prepared[i])
        for i, name in enumerate(trainable_module_names):
            if hasattr(guidance_adapter, name) and getattr(guidance_adapter, name) is not None:
                guidance_adapter.set_component(name, prepared[n_gen + i])
        self.optimizer = prepared[n_gen + n_gui]
        self.optimizer_gui = prepared[n_gen + n_gui + 1]
        if self.test_dataloader is not None:
            self.test_dataloader = prepared[-1]
        self.guidance_adapter = guidance_adapter

        self._load_inference_components(trainable_module_names)
        self._init_reward_model()
        self._init_dino_model()

    def _init_dino_model(self):
        if self.training_args.encoder_type == "dinov2":
            self.rep_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc').to(self.accelerator.device, dtype=self.adapter._inference_dtype)
            self.z_size = self.rep_model.backbone.embed_dim
            if self.accelerator.is_main_process:
                logger.info(f"Using DINOv2 model with {self.z_size} feature dimension")
            self.transform_rep = DINOv2ProcessorWithGrad(res=224)
            self.rep_model.requires_grad_(False)
        elif self.training_args.encoder_type is None:
            self.rep_model = None
            self.transform_rep = None
        else:
            raise ValueError(f"Unsupported encoder type: {self.training_args.encoder_type}. Supported: None, 'dinov2'")

    # =========================== Helper Functions ============================
    def _prepare_inputs(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        Extract or compute embeddings from batch.
        - If preprocessing is enabled, batch already contains encoded embeddings.
        - If disabled, use adapter.preprocess_func to encode on-the-fly.
        Returns a dict suitable for passing to adapter.forward via **kwargs.
        """
        # Check if batch already has pre-encoded embeddings
        if self.config.data_args.enable_preprocess:
            return {k: v for k, v in batch.items()}
        else:
            preprocess_kwargs = filter_kwargs(self.adapter.preprocess_func, **batch)
            reprocessed_batch = self.adapter.preprocess_func(**preprocess_kwargs)
            return {
                **batch,
                **{
                    k: v.to(device=self.accelerator.device)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in reprocessed_batch.items()
                }
            }

    # =========================== Image Visualization ============================
    @torch.no_grad()
    def _decode_and_build_log_images(
        self,
        latents_dict: Dict[str, torch.Tensor],
        prompts: Optional[List[str]] = None,
        max_images: int = 4,
    ) -> Dict[str, Any]:
        """
        Decode named latent tensors, then build per-sample comparison grids.

        For each sample index i, produces a single row image:
            [x0_gen | x0_real | x0_fake | backward_sample]
        with a label strip on top of each column. All rows are logged as a list
        under "vis/comparison" so WandB / TensorBoard show them side-by-side.

        Args:
            latents_dict: {"name": (B, C, H, W) latent tensor}.
                          Ordered dict recommended; column order follows insertion order.
            prompts: Optional prompt strings for captions.
            max_images: Max samples to decode (to limit cost).

        Returns:
            Dict with "vis/comparison" -> list[LogImage], one per sample.
        """
        from PIL import Image, ImageDraw, ImageFont

        # 1. Decode all categories
        decoded: Dict[str, List] = {}  # name -> list of PIL images
        names_order = []
        for name, latents in latents_dict.items():
            if latents is None:
                continue
            latents_subset = latents[:max_images]
            try:
                pil_images = self.adapter.decode_latents(latents_subset.detach(), output_type="pil")
                if not isinstance(pil_images, list):
                    pil_images = [pil_images]
                decoded[name] = pil_images
                names_order.append(name)
            except Exception as e:
                logger.warning(f"Failed to decode latents for '{name}': {e}")

        if not decoded:
            return {}

        # 2. Build per-sample comparison strips
        n_samples = min(max_images, *(len(imgs) for imgs in decoded.values()))
        label_h = 20  # height of the label strip

        comparison_images: List[LogImage] = []
        for i in range(n_samples):
            panels = []
            for name in names_order:
                imgs = decoded[name]
                if i >= len(imgs):
                    continue
                img = imgs[i].convert("RGB")
                w, h = img.size

                # Create label strip
                label = Image.new("RGB", (w, label_h), color=(0, 0, 0))
                draw = ImageDraw.Draw(label)
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 12)
                except Exception:
                    font = ImageFont.load_default()
                draw.text((4, 2), name, fill=(255, 255, 255), font=font)

                # Stack label on top of image
                panel = Image.new("RGB", (w, h + label_h))
                panel.paste(label, (0, 0))
                panel.paste(img, (0, label_h))
                panels.append(panel)

            if not panels:
                continue

            # Resize all panels to the same height (use first panel's height)
            target_h = panels[0].size[1]
            resized = []
            for p in panels:
                if p.size[1] != target_h:
                    ratio = target_h / p.size[1]
                    p = p.resize((int(p.size[0] * ratio), target_h), Image.Resampling.LANCZOS)
                resized.append(p)

            # Horizontally concatenate
            total_w = sum(p.size[0] for p in resized)
            strip = Image.new("RGB", (total_w, target_h))
            x_offset = 0
            for p in resized:
                strip.paste(p, (x_offset, 0))
                x_offset += p.size[0]

            caption = f"step {self.step} | sample {i}"
            if prompts and i < len(prompts):
                caption += f" | {prompts[i][:80]}"
            comparison_images.append(LogImage(strip, caption=caption))

        return {"vis/comparison": comparison_images} if comparison_images else {}

    # =========================== Velocity Prediction ============================
    def pred_velocity(
        self,
        adapter: BaseAdapter,
        latents: torch.Tensor,
        t: torch.Tensor,
        embeddings_batch: Dict[str, Any],
        guidance_scale: float = 1.0,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Predict velocity (noise_pred) using adapter.forward.
        t: (B,) or (B,1,1,1) in [0,1]; will be scaled to scheduler scale if needed.
        joint_attention_kwargs: Optional kwargs for attention layers (e.g., LoRA scale).
        """
        batch_size = latents.shape[0]
        device, dtype = latents.device, latents.dtype
        if t.ndim > 1:
            t_flat = t.flatten()
        else:
            t_flat = t
        scheduler = adapter.scheduler
        t_max = 1000.0
        if hasattr(scheduler, "timesteps") and scheduler.timesteps is not None:
            tt = scheduler.timesteps
            if isinstance(tt, torch.Tensor):
                t_max = float(tt.max().item()) or 1000.0
            else:
                t_max = float(max(tt)) or 1000.0
        t_scaled = t_flat.to(device=device, dtype=dtype) * t_max
        forward_kwargs = {
            "t": t_scaled,
            "latents": latents,
            "t_next": 0.0,
            "next_latents": None,
            "compute_log_prob": False,
            "return_kwargs": ["noise_pred"],
            "guidance_scale": guidance_scale,
            **embeddings_batch,
        }
        if joint_attention_kwargs is not None:
            forward_kwargs["joint_attention_kwargs"] = joint_attention_kwargs
        forward_kwargs = filter_kwargs(adapter.forward, **forward_kwargs)
        out = adapter.forward(**forward_kwargs)
        return out.noise_pred.to(latents.dtype)

    # =========================== Evaluation Loop ============================
    def evaluate(self) -> None:
        """Evaluation loop."""
        if self.test_dataloader is None:
            return
        
        self.adapter.eval()
        with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
            all_samples : List[BaseSample] = []
            
            for batch in tqdm(
                self.test_dataloader,
                desc='Evaluating',
                disable=not self.show_progress_bar,
            ):
                generator = create_generator_by_prompt(batch['prompt'], self.training_args.seed)
                inference_kwargs = {
                    'compute_log_prob': False,
                    'generator': generator,
                    'trajectory_indices': None, # No need to store all trajectories during evaluation
                    **self.eval_args,
                }
                inference_kwargs.update(**batch)
                inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
                samples = self.adapter.inference(**inference_kwargs)
                all_samples.extend(samples)
            
            # Compute rewards with eval reward models
            rewards = self.eval_reward_processor.compute_rewards(
                samples=all_samples,
                store_to_samples=False,
                epoch=self.epoch,
                split='pointwise',  # Only `pointwise` reward can be compute when evaluation, since there is no `group` here.
            )
            # Gather and log rewards
            rewards = {key: torch.as_tensor(value).to(self.accelerator.device) for key, value in rewards.items()}
            gathered_rewards = {
                key: self.accelerator.gather(value).cpu().numpy()
                for key, value in rewards.items()
            }
            
            # Log statistics
            if self.accelerator.is_main_process:
                _log_data = {f'eval/reward_{key}_mean': np.mean(value) for key, value in gathered_rewards.items()}
                _log_data.update({f'eval/reward_{key}_std': np.std(value) for key, value in gathered_rewards.items()})
                _log_data['eval_samples'] = all_samples
                self.log_data(_log_data, step=self.step)
            self.accelerator.wait_for_everyone()

    # =========================== Training Loop ============================
    def start(self):
        while True:
            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.config.run_name),
                    "checkpoints",
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()

            self.optimize()
            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def sample(self):
        return []

    def optimize(self):
        self.adapter.train()
        self.guidance_adapter.train()
        device = self.accelerator.device
        train_args = self.training_args

        data_iter = iter(self.dataloader)
        loss_info = defaultdict(list)
        inner_step = 0
        global_step = self.step

        lora_scale_f_kwargs = {"scale": train_args.lora_scale_f}
        res = self.training_args.resolution
        if isinstance(res, int):
            height, width = res, res
        else:
            height, width = res

        vis_latents = None
        vis_prompts = None

        for inner_step in range(1, self.ratio_update + 1):
            # ---- Sample new batch and new random latents each inner step ----
            batch = next(data_iter)
            with torch.no_grad():
                input_batch = self._prepare_inputs(batch, device)

            dtype = next(self.adapter.transformer.parameters()).dtype

            latent = self.adapter.pipeline.prepare_latents(
                batch_size=self.train_batch_size,
                num_channels_latents=self.adapter.pipeline.transformer.config.in_channels,
                height=height,
                width=width,
                dtype=dtype,
                device=device,
                generator=None,
            )

            # ---- Backward sampling ----
            with torch.no_grad(), self.autocast():
                xT, all_x0, t_steps = v2x0_sampler_adapter(
                    self.adapter,
                    latent,
                    input_batch,
                    num_steps=self.num_steps,
                    shift=self.shift,
                    guidance_scale=self.guidance_scale,
                )
            bsz = xT.shape[0]
            dtype = xT.dtype
            t_steps = t_steps.to(device=device, dtype=dtype)

            # ---- Sample timesteps for generator and guidance ----
            ts = sample_discrete(
                bsz,
                t_steps.flip(0),
                alpha=train_args.gen_a,
                beta=train_args.gen_b,
                s_type=train_args.s_type_gen,
                step=global_step,
                dynamic_step=self.dynamic_step,
                device=device,
                dtype=dtype,
            )
            input_latent_clean = get_sample(all_x0, ts, t_steps).to(device=device, dtype=dtype)
            noise_i = torch.randn_like(input_latent_clean, device=device, dtype=dtype)
            input_latent_gen = (1 - ts) * input_latent_clean + ts * noise_i
            input_latent_gen = input_latent_gen.to(device=device, dtype=dtype)

            ts_gui = sample_continue(
                bsz,
                alpha=train_args.gui_a,
                beta=train_args.gui_b,
                s_type=train_args.s_type_gui,
                step=global_step,
                dynamic_step=self.dynamic_step,
                device=device,
                dtype=dtype,
            )
            noise_gui = torch.randn_like(input_latent_clean, device=device, dtype=dtype)
            input_latent_gui = (1 - ts_gui) * input_latent_clean + ts_gui * noise_gui
            gt_velocity = noise_gui - input_latent_clean

            # ================ Guidance model update (every inner step) ================
            self.adapter.eval()
            self.guidance_adapter.train()

            with self.accelerator.accumulate(*self.guidance_adapter.trainable_components):
                with self.autocast():
                    v_pred_gui = self.pred_velocity(
                        self.guidance_adapter,
                        input_latent_gui,
                        ts_gui,
                        input_batch,
                        guidance_scale=1.0,
                        joint_attention_kwargs=lora_scale_f_kwargs,
                    )
                    diffusion_loss = ((v_pred_gui - gt_velocity) ** 2).mean(dim=tuple(range(1, v_pred_gui.ndim))).mean()
                self.accelerator.backward(diffusion_loss)
                if self.accelerator.sync_gradients:
                    grad_norm_gui = torch.nn.utils.clip_grad_norm_(
                        self.guidance_adapter.get_trainable_parameters(),
                        train_args.max_grad_norm,
                    )
                    loss_info["grad_norm_gui"].append(grad_norm_gui.detach().float())
                self.optimizer_gui.step()
                self.optimizer_gui.zero_grad(set_to_none=True)
                self.optimizer.zero_grad(set_to_none=True)
                loss_info["diff"].append(diffusion_loss.detach().float())

            # ================ Generator update (only on last inner step) ================
            if inner_step == self.ratio_update:
                self.adapter.train()
                self.guidance_adapter.eval()

                # Predict x0 from generator (LoRA enabled, no use_ref_parameters)
                with self.autocast():
                    v_pred_gen = self.pred_velocity(
                        self.adapter,
                        input_latent_gen,
                        ts,
                        input_batch,
                        guidance_scale=1.0,
                    )
                x0 = input_latent_gen + (0.0 - ts) * v_pred_gen

                # ---- Reward loss (DINOv2) ----
                reward_loss = torch.zeros(1, device=device, dtype=torch.float32)
                if global_step >= self.cold_start_iter and self.training_args.encoder_type is not None:
                    if self.training_args.encoder_type == "dinov2":
                        with self.autocast():
                            # Decode latents to pixel space
                            vae = self.adapter.pipeline.vae
                            scaling_factor = vae.config.scaling_factor
                            shift_factor = getattr(vae.config, "shift_factor", 0.0)
                            latent_x0 = (x0 / scaling_factor) + shift_factor
                            samples = vae.decode(latent_x0.to(vae.dtype), return_dict=False)[0]
                            samples = ((samples + 1) / 2.0).clamp(0, 1)
                            samples = self.transform_rep.preprocess(samples)
                            logistics = self.rep_model(samples)
                            # For T2I (no class labels), keep reward_loss=0.0
                            if "class_labels" in batch and batch["class_labels"] is not None:
                                criterion = torch.nn.CrossEntropyLoss()
                                dino_loss = criterion(logistics, batch["class_labels"].to(device))
                                reward_loss = dino_loss
                    else:
                        reward_loss = torch.zeros(1, device=device, dtype=torch.float32)

                # ---- DMD loss ----
                # Compute lora_scale_r with cosine decay
                if global_step < self.dynamic_step:
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * global_step / self.dynamic_step))
                    lora_scale_r = train_args.lora_scale_r * cosine_factor
                else:
                    lora_scale_r = 0.0

                lora_scale_r_kwargs = {"scale": lora_scale_r}

                ts_dmd = sample_continue(
                    bsz,
                    alpha=train_args.gui_a,
                    beta=train_args.gui_b,
                    s_type=train_args.s_type_gui,
                    step=global_step,
                    dynamic_step=self.dynamic_step,
                    device=device,
                    dtype=dtype,
                )
                noise_dmd = torch.randn_like(input_latent_gen, device=device, dtype=dtype)
                input_latent_dmd = (1 - ts_dmd) * x0 + ts_dmd * noise_dmd

                with self.autocast():
                    with torch.no_grad():
                        v_fake_dmd = self.pred_velocity(
                            self.guidance_adapter,
                            input_latent_dmd,
                            ts_dmd,
                            input_batch,
                            guidance_scale=1.0,
                            joint_attention_kwargs=lora_scale_f_kwargs,
                        )
                        v_real_dmd = self.pred_velocity(
                            self.guidance_adapter,
                            input_latent_dmd,
                            ts_dmd,
                            input_batch,
                            guidance_scale=train_args.cfg_r,
                            joint_attention_kwargs=lora_scale_r_kwargs,
                        )

                    x0_r = input_latent_dmd + (0.0 - ts_dmd) * v_real_dmd
                    x0_f = input_latent_dmd + (0.0 - ts_dmd) * v_fake_dmd
                    p_real = x0 - x0_r
                    p_fake = x0 - x0_f
                    grad = (p_real - p_fake) / (
                        torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True) + 1e-8
                    )
                    grad = torch.nan_to_num(grad)

                    dmd_loss = 0.5 * torch.nn.functional.mse_loss(
                        x0.float(),
                        (x0 - grad).detach().float(),
                        reduction="mean",
                    )
                    loss_gen = dmd_loss + train_args.dino_loss_weight * reward_loss

                self.accelerator.backward(loss_gen)

                if self.accelerator.sync_gradients:
                    grad_norm_gen = torch.nn.utils.clip_grad_norm_(
                        self.adapter.get_trainable_parameters(),
                        train_args.max_grad_norm,
                    )
                    loss_info["grad_norm_gen"].append(grad_norm_gen.detach().float())
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_gui.zero_grad(set_to_none=True)

                # ---- Collect generator metrics (no_grad to avoid graph issues) ----
                with torch.no_grad():
                    loss_info["dmd"].append(dmd_loss.detach().float())
                    loss_info["reward"].append(
                        reward_loss.detach().float()
                        if isinstance(reward_loss, torch.Tensor)
                        else torch.tensor(reward_loss, device=device)
                    )
                    loss_info["loss_gen_total"].append(loss_gen.detach().float())
                    loss_info["lora_scale_r"].append(torch.tensor(lora_scale_r, device=device))
                    # DMD signal diagnostics
                    loss_info["grad_direction_norm"].append(grad.detach().float().norm())
                    loss_info["x0_pred_norm"].append(x0.detach().float().norm())
                    loss_info["x0_real_norm"].append(x0_r.detach().float().norm())
                    loss_info["x0_fake_norm"].append(x0_f.detach().float().norm())

                # ---- Stash latents for image visualization ----
                vis_latents = {
                    "x0_gen": x0.detach(),
                    "x0_real": x0_r.detach(),
                    "x0_fake": x0_f.detach(),
                    "backward_sample": xT.detach(),
                }
                vis_prompts = batch.get("prompt", None)

        # ---- Logging (once per optimize() = once per global_step) ----
        if self.accelerator.sync_gradients:
            global_step += 1
            log_data = {}

            # --- Guidance losses (averaged over all inner steps) ---
            if loss_info["diff"]:
                diff_losses = torch.stack(loss_info["diff"])
                log_data["train/diff_loss_mean"] = diff_losses.mean().item()
                log_data["train/diff_loss_last"] = diff_losses[-1].item()
                # Track if guidance is improving within the inner loop
                if len(diff_losses) > 1:
                    log_data["train/diff_loss_first"] = diff_losses[0].item()

            # --- Generator losses ---
            if loss_info["dmd"]:
                log_data["train/dmd_loss"] = torch.stack(loss_info["dmd"]).mean().item()
            if loss_info["reward"]:
                log_data["train/reward_loss"] = torch.stack(loss_info["reward"]).mean().item()
            if loss_info["loss_gen_total"]:
                log_data["train/loss_gen_total"] = torch.stack(loss_info["loss_gen_total"]).mean().item()

            # --- Gradient norms ---
            if loss_info["grad_norm_gui"]:
                log_data["train/grad_norm_gui"] = torch.stack(loss_info["grad_norm_gui"]).mean().item()
            if loss_info["grad_norm_gen"]:
                log_data["train/grad_norm_gen"] = torch.stack(loss_info["grad_norm_gen"]).mean().item()

            # --- LoRA scale schedule ---
            if loss_info["lora_scale_r"]:
                log_data["train/lora_scale_r"] = torch.stack(loss_info["lora_scale_r"]).mean().item()

            # --- DMD signal diagnostics ---
            if loss_info["grad_direction_norm"]:
                log_data["train/dmd_grad_norm"] = torch.stack(loss_info["grad_direction_norm"]).mean().item()
            if loss_info["x0_pred_norm"]:
                log_data["train/x0_pred_norm"] = torch.stack(loss_info["x0_pred_norm"]).mean().item()
            if loss_info["x0_real_norm"]:
                log_data["train/x0_real_norm"] = torch.stack(loss_info["x0_real_norm"]).mean().item()
            if loss_info["x0_fake_norm"]:
                log_data["train/x0_fake_norm"] = torch.stack(loss_info["x0_fake_norm"]).mean().item()

            # --- Image visualization (periodic) ---
            if (
                self.image_log_steps > 0
                and global_step % self.image_log_steps == 0
                and self.accelerator.is_main_process
                and vis_latents is not None
            ):
                try:
                    vis_data = self._decode_and_build_log_images(
                        vis_latents,
                        prompts=vis_prompts,
                        max_images=4,
                    )
                    log_data.update(vis_data)
                except Exception as e:
                    logger.warning(f"Image visualization failed at step {global_step}: {e}")

            # --- Step counters ---
            log_data["train/global_step"] = global_step
            log_data["train/epoch"] = self.epoch

            self.log_data(log_data, step=self.step)
            self.step = global_step