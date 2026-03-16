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
from typing import Any, Dict, List

import torch
import numpy as np
from functools import partial

from ...models.abc import BaseAdapter
from ...trainers.abc import BaseTrainer
from .dmdr_utils import (
    mean_flat,
    sample_continue,
    sample_discrete,
    get_sample,
    v2x0_sampler_adapter,
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

    # =========================== Helper Functions ============================
    def _prepare_embeddings(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        Extract or compute embeddings from batch.
        - If preprocessing is enabled, batch already contains encoded embeddings.
        - If disabled, use adapter.preprocess_func to encode on-the-fly.
        Returns a dict suitable for passing to adapter.forward via **kwargs.
        """
        # Check if batch already has pre-encoded embeddings
        if "prompt_embeds" in batch:
            embeddings = {k: v for k, v in batch.items()}
        else:
            # Preprocessing disabled: encode on-the-fly using the standard interface
            prompts = batch.get("prompt") or batch.get("prompts")
            if isinstance(prompts, str):
                prompts = [prompts]
            embeddings = self.adapter.preprocess_func(prompt=prompts)
            embeddings.update({k: v for k, v in batch.items() if k not in ("prompt", "prompts")})

        # Move tensors to device
        for k, v in embeddings.items():
            if isinstance(v, torch.Tensor):
                embeddings[k] = v.to(device)
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                embeddings[k] = [x.to(device) for x in v]
        return embeddings

    def _dmdr_latent_shape(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        res = self.training_args.resolution
        if isinstance(res, (list, tuple)):
            h, w = res[0], res[1] if len(res) > 1 else res[0]
        else:
            h = w = res
        latent_h, latent_w = h // 8, w // 8
        pipeline = self.adapter.pipeline
        channels = getattr(
            getattr(pipeline, "transformer", None),
            "in_channels",
            4,
        ) or 4
        return (batch_size, channels, latent_h, latent_w)


    # =========================== Velocity Prediction ============================
    def pred_velocity_adapter(
        self,
        adapter: BaseAdapter,
        latents: torch.Tensor,
        t: torch.Tensor,
        embeddings_batch: Dict[str, Any],
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        """
        Predict velocity (noise_pred) using adapter.forward.
        t: (B,) or (B,1,1,1) in [0,1]; will be scaled to scheduler scale if needed.
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
        forward_kwargs = filter_kwargs(adapter.forward, **forward_kwargs)
        out = adapter.forward(**forward_kwargs)
        return out.noise_pred.to(latents.dtype)

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
        ta = self.training_args
        batch_size = ta.per_device_batch_size
        num_steps = getattr(ta, "dmdr_num_steps", 4)
        shift = getattr(ta, "dmdr_shift", 1.0)
        ratio_update = getattr(ta, "ratio_update", 1)
        cold_start = getattr(ta, "cold_start_iter", 0)  # Reserved: gates reward/DINO loss only (not DMD)
        guidance_scale = getattr(ta, "guidance_scale", 0.0)

        num_batches = getattr(ta, "num_batches_per_epoch", 1)

        data_iter = iter(self.dataloader)
        loss_info = defaultdict(list)
        inner_step = 0
        global_step = self.step

        for batch_idx in range(num_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            with torch.no_grad():
                embeddings_batch = self._prepare_embeddings(batch, device)

            # Infer batch size from any tensor in the embeddings (model-agnostic)
            b_actual = min(
                next(v.shape[0] for v in embeddings_batch.values() if isinstance(v, torch.Tensor)),
                batch_size,
            )
            try:
                dtype = next(self.adapter.transformer.parameters()).dtype
            except StopIteration:
                dtype = torch.float32
            latent_shape = self._dmdr_latent_shape(b_actual, device, dtype)
            latent = torch.randn(*latent_shape, device=device, dtype=dtype)

            with torch.no_grad(), self.autocast():
                xT, all_x0, t_steps = v2x0_sampler_adapter(
                    self.adapter,
                    latent,
                    embeddings_batch,
                    num_steps=num_steps,
                    shift=shift,
                    guidance_scale=guidance_scale,
                )
            bsz = xT.shape[0]
            dtype = xT.dtype
            t_steps = t_steps.to(device=device, dtype=dtype)

            t_steps_schedule = t_steps if t_steps.dim() > 0 else t_steps.unsqueeze(0)
            ts = sample_discrete(
                bsz,
                t_steps_schedule,
                alpha=ta.gen_a,
                beta=ta.gen_b,
                s_type=ta.s_type_gen,
                step=global_step,
                dynamic_step=getattr(ta, "dynamic_step", 1000),
                device=device,
                dtype=dtype,
            )
            input_latent_clean = get_sample(all_x0, ts, t_steps).to(device=device, dtype=dtype)
            noise_i = torch.randn_like(input_latent_clean, device=device, dtype=dtype)
            input_latent_gen = (1 - ts) * input_latent_clean + ts * noise_i

            ts_gui = sample_continue(
                bsz,
                alpha=ta.gui_a,
                beta=ta.gui_b,
                s_type=ta.s_type_gui,
                step=global_step,
                dynamic_step=getattr(ta, "dynamic_step", 1000),
                device=device,
                dtype=dtype,
            )
            noise_gui = torch.randn_like(input_latent_clean, device=device, dtype=dtype)
            input_latent_gui = (1 - ts_gui) * input_latent_clean + ts_gui * noise_gui
            gt_velocity = noise_gui - input_latent_clean

            with self.accelerator.accumulate(*self.guidance_adapter.trainable_components):
                with self.autocast():
                    v_pred_gui = self.pred_velocity_adapter(
                        self.guidance_adapter,
                        input_latent_gui,
                        ts_gui,
                        embeddings_batch,
                        guidance_scale=0.0,
                    )
                    diffusion_loss = ((v_pred_gui - gt_velocity) ** 2).mean(dim=tuple(range(1, v_pred_gui.ndim))).mean()
                self.accelerator.backward(diffusion_loss)
                if self.accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(
                        self.guidance_adapter.get_trainable_parameters(),
                        ta.max_grad_norm,
                    )
                    self.optimizer_gui.step()
                    self.optimizer_gui.zero_grad(set_to_none=True)
                self.optimizer.zero_grad(set_to_none=True)
                loss_info["diff"].append(diffusion_loss.detach().float())

            if (inner_step % ratio_update == 0) and inner_step > 0:
                self.adapter.train()
                self.guidance_adapter.eval()
                with self.autocast():
                    v_pred_gen = self.pred_velocity_adapter(
                        self.adapter,
                        input_latent_gen,
                        ts,
                        embeddings_batch,
                        guidance_scale=0.0,
                    )
                x0 = input_latent_gen + (0.0 - ts) * v_pred_gen

                lora_scale_r = getattr(ta, "lora_scale_r", 0.25)
                if getattr(ta, "dynamic_step", 1000) and global_step < getattr(ta, "dynamic_step", 1000):
                    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * global_step / ta.dynamic_step))
                    lora_scale_r = lora_scale_r * cosine_factor
                else:
                    lora_scale_r = 0.0

                ts_dmd = sample_continue(
                    bsz,
                    alpha=ta.gui_a,
                    beta=ta.gui_b,
                    s_type=ta.s_type_gui,
                    step=global_step,
                    dynamic_step=getattr(ta, "dynamic_step", 1000),
                    device=device,
                    dtype=dtype,
                )
                noise_dmd = torch.randn_like(input_latent_gen, device=device, dtype=dtype)
                input_latent_dmd = (1 - ts_dmd) * x0 + ts_dmd * noise_dmd

                with torch.no_grad(), self.autocast():
                    v_fake_dmd = self.pred_velocity_adapter(
                        self.guidance_adapter,
                        input_latent_dmd,
                        ts_dmd,
                        embeddings_batch,
                        guidance_scale=0.0,
                    )
                    v_real_dmd = self.pred_velocity_adapter(
                        self.guidance_adapter,
                        input_latent_dmd,
                        ts_dmd,
                        embeddings_batch,
                        guidance_scale=getattr(ta, "cfg_r", 1.0),
                    )
                x0_r = input_latent_dmd + (0.0 - ts_dmd) * v_real_dmd
                x0_f = input_latent_dmd + (0.0 - ts_dmd) * v_fake_dmd
                p_real = x0 - x0_r
                p_fake = x0 - x0_f
                grad = (p_real - p_fake) / (
                    torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True) + 1e-8
                )
                grad = torch.nan_to_num(grad)

                with self.accelerator.accumulate(*self.adapter.trainable_components):
                    with self.autocast():
                        dmd_loss = 0.5 * torch.nn.functional.mse_loss(
                            x0.float(),
                            (x0 - grad).detach().float(),
                            reduction="mean",
                        )
                    self.accelerator.backward(dmd_loss)
                    if self.accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(
                            self.adapter.get_trainable_parameters(),
                            ta.max_grad_norm,
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.optimizer_gui.zero_grad(set_to_none=True)
                    loss_info["dmd"].append(dmd_loss.detach().float())

            if self.accelerator.sync_gradients:
                if (inner_step % ratio_update == 0) and inner_step > 0:
                    global_step += 1
                inner_step += 1
            if self.accelerator.sync_gradients and loss_info:
                log_data = {}
                if "diff" in loss_info and loss_info["diff"]:
                    log_data["train/diff"] = torch.stack(loss_info["diff"]).mean().item()
                if "dmd" in loss_info and loss_info["dmd"]:
                    log_data["train/dmd"] = torch.stack(loss_info["dmd"]).mean().item()
                log_data["train/inner_step"] = inner_step
                log_data["train/global_step"] = global_step
                self.log_data(log_data, step=self.step)
                self.step += 1
                loss_info = defaultdict(list)

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
