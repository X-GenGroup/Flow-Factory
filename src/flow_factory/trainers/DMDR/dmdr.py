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
                    torch.nn.utils.clip_grad_norm_(
                        self.guidance_adapter.get_trainable_parameters(),
                        train_args.max_grad_norm,
                    )
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
                    torch.nn.utils.clip_grad_norm_(
                        self.adapter.get_trainable_parameters(),
                        train_args.max_grad_norm,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_gui.zero_grad(set_to_none=True)
                loss_info["dmd"].append(dmd_loss.detach().float())
                loss_info["reward"].append(reward_loss.detach().float() if isinstance(reward_loss, torch.Tensor) else torch.tensor(reward_loss))

        # ---- Logging ----
        if self.accelerator.sync_gradients:
            global_step += 1
            log_data = {}
            if "diff" in loss_info and loss_info["diff"]:
                log_data["train/diff"] = torch.stack(loss_info["diff"]).mean().item()
            if "dmd" in loss_info and loss_info["dmd"]:
                log_data["train/dmd"] = torch.stack(loss_info["dmd"]).mean().item()
            if "reward" in loss_info and loss_info["reward"]:
                log_data["train/reward"] = torch.stack(loss_info["reward"]).mean().item()
            log_data["train/inner_steps"] = self.ratio_update
            log_data["train/global_step"] = global_step
            self.log_data(log_data, step=self.step)
            self.step = global_step