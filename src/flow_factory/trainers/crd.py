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

# src/flow_factory/trainers/crd.py
"""
Centered Reward Distillation (CRD) Trainer.
Reference:
[1] Diffusion Reinforcement Learning via Centered Reward Distillation
    - https://arxiv.org/abs/2603.14128
"""
import os
from typing import List, Dict, Any, Union, Optional
from functools import partial
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from ..hparams import CRDTrainingArguments
from ..samples import BaseSample
from ..rewards import RewardBuffer
from ..utils.base import filter_kwargs, create_generator, create_generator_by_prompt, to_broadcast_tensor
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import TimeSampler, flow_match_sigma
from ..utils.dist import reduce_loss_info

logger = setup_logger(__name__)


# ========================= Decay Utilities =========================

# Predefined decay presets: (start_step, start_value, slope, end_value)
_DECAY_PRESETS = {
    0: (0, 0.0, 0.0, 0.0),
    1: (0, 0.0, 0.001, 0.5),
    2: (75, 0.0, 0.0075, 0.999),
    3: (0, 1.0, 0.0, 1.0),
    4: (0, 0.0, 0.02, 0.99),
    5: (0, 0.0, 0.01, 0.5),
    6: (0, 0.0, 0.0075, 0.999),
    'none': (0, 0.0, 0.0, 0.0),
    'slow': (0, 0.0, 0.001, 0.5),
    'medium': (75, 0.0, 0.0075, 0.999),
    'offline': (0, 1.0, 0.0, 1.0),
    'fast': (0, 0.0, 0.02, 0.99),
    'moderate': (0, 0.0, 0.01, 0.5),
}


def compute_decay(step: int, decay_type) -> float:
    """
    Compute a decay value at the given step.

    Args:
        step: Current training step.
        decay_type: An int/str preset key, or a string ``"start_step-start_value-slope-end_value"``.

    Returns:
        Decay value (float in [0, 1]).
    """
    # Try int conversion for string digits like "0", "1", etc.
    if isinstance(decay_type, str):
        try:
            decay_type = int(decay_type)
        except ValueError:
            pass

    if decay_type in _DECAY_PRESETS:
        start_step, start_value, slope, end_value = _DECAY_PRESETS[decay_type]
    elif isinstance(decay_type, str) and '-' in decay_type:
        parts = decay_type.split('-')
        assert len(parts) == 4, (
            f"Decay string format must be 'start_step-start_value-slope-end_value', got: {decay_type}"
        )
        start_step, start_value, slope, end_value = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        start_step = int(start_step)
    else:
        raise ValueError(
            f"Invalid decay_type: {decay_type}. "
            f"Valid options: {list(_DECAY_PRESETS.keys())} or 'start_step-start_value-slope-end_value'"
        )

    if step < start_step:
        return start_value
    return min(start_value + (step - start_step) * slope, end_value)


# ============================ CRD Trainer ============================

class CRDTrainer(BaseTrainer):
    """
    Centered Reward Distillation (CRD) Trainer.

    Core algorithm: match centered external rewards with implicit model rewards
    estimated from prediction error in velocity space.

    Key features (matching the original CRD implementation):
    - Loss is based on centered reward distillation (not contrastive positive/negative).
    - Maintains an "old" model snapshot for implicit reward estimation (decay_type).
    - Maintains a "sampling" model snapshot for off-policy rollouts (decay_type2).
    - Supports dual-direction centering with temperature-weighted softmax.
    - Supports adaptive KL based on reward signals.

    Model snapshots:
    - Current model: trainable parameters (LoRA "default" in original CRD).
    - Old model: named parameter snapshot for implicit reward estimation.
    - Sampling model: named parameter snapshot for rollout generation.
    - Reference model: original pre-trained weights (LoRA disabled / base model).

    Reference: https://arxiv.org/abs/2603.14128
    """

    _OLD_PARAMS_NAME = '_crd_old'
    _SAMPLING_PARAMS_NAME = '_crd_sampling'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.training_args: CRDTrainingArguments

        # CRD-specific config
        self.crd_beta = self.training_args.crd_beta
        self.crd_loss_type = self.training_args.crd_loss_type
        self.use_old_for_loss = self.training_args.use_old_for_loss
        self.adaptive_logp = self.training_args.adaptive_logp
        self.weight_temp = self.training_args.weight_temp

        # Decay schedules
        self.old_model_decay = self.training_args.old_model_decay
        self.sampling_model_decay = self.training_args.sampling_model_decay

        # KL
        self.kl_beta = self.training_args.kl_beta
        self.kl_cfg = self.training_args.kl_cfg
        self.reward_adaptive_kl = self.training_args.reward_adaptive_kl

        # Timestep sampling
        self.time_sampling_strategy = self.training_args.time_sampling_strategy
        self.time_shift = self.training_args.time_shift
        self.num_train_timesteps = self.training_args.num_train_timesteps
        self.timestep_range = self.training_args.timestep_range

        self.kl_type = self.training_args.kl_type
        if self.kl_type != 'v-based':
            logger.warning(
                f"CRD-Trainer only supports 'v-based' KL loss, got {self.kl_type}, switching to 'v-based'."
            )
            self.kl_type = 'v-based'

        # Initialize model snapshots: "old" (for implicit reward) and "sampling" (for rollout)
        self._init_model_snapshots()

    # ========================= Initialization =========================

    def _init_model_snapshots(self):
        """
        Initialize both model snapshots by storing copies of current trainable parameters.

        In the original CRD, this corresponds to:
        - ``transformer.add_adapter("old", ...)``  +  copy from "default"
        - ``transformer.add_adapter("sampling", ...)``  +  copy from "default"
        """
        ref_device = self.training_args.ref_param_device

        # Old model snapshot (for implicit reward estimation)
        self.adapter.add_named_parameters(
            name=self._OLD_PARAMS_NAME,
            device=ref_device,
        )
        logger.info("CRD: Initialized 'old' model snapshot for implicit reward estimation.")

        # Sampling model snapshot (for off-policy rollout generation)
        self.adapter.add_named_parameters(
            name=self._SAMPLING_PARAMS_NAME,
            device=ref_device,
        )
        logger.info("CRD: Initialized 'sampling' model snapshot for rollout generation.")

    @property
    def enable_kl_loss(self) -> bool:
        return self.kl_beta > 0.0

    @contextmanager
    def sampling_context(self):
        """
        Use the sampling model snapshot for rollout generation.

        In the original CRD, this corresponds to ``transformer_ddp.module.set_adapter("sampling")``.
        The sampling model is a separate snapshot blended towards current weights with
        ``sampling_model_decay`` (decay_type2 in the original).
        """
        with self.adapter.use_named_parameters(self._SAMPLING_PARAMS_NAME):
            yield

    # ========================= Timestep Sampling =========================

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample continuous or discrete timesteps based on configured `time_sampling_strategy`.

        Returns:
            Tensor of shape ``(num_train_timesteps, batch_size)`` with scheduler-scale ``t`` in ``[0, 1000]``.
        """
        device = self.accelerator.device
        time_sampling_strategy = self.time_sampling_strategy.lower()
        available = ['logit_normal', 'uniform', 'discrete', 'discrete_with_init', 'discrete_wo_init']

        if time_sampling_strategy == 'logit_normal':
            return TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
                stratified=True,
            )
        elif time_sampling_strategy == 'uniform':
            return TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
            )
        elif time_sampling_strategy.startswith('discrete'):
            # Map time_sampling_strategy to (include_init, force_init)
            discrete_config = {
                'discrete':           (True, False),
                'discrete_with_init': (True, True),
                'discrete_wo_init':   (False, False),
            }
            if time_sampling_strategy not in discrete_config:
                raise ValueError(f"Unknown time_sampling_strategy: {time_sampling_strategy}. Available: {available}")
            include_init, force_init = discrete_config[time_sampling_strategy]
            return TimeSampler.discrete(
                batch_size=batch_size,
                num_train_timesteps=self.num_train_timesteps,
                scheduler_timesteps=self.adapter.scheduler.timesteps,
                timestep_range=self.timestep_range,
                include_init=include_init,
                force_init=force_init,
            )
        else:
            raise ValueError(f"Unknown time_sampling_strategy: {time_sampling_strategy}. Available: {available}")

    # ========================= Evaluation Loop =========================

    def evaluate(self) -> None:
        """Evaluation loop."""
        if self.test_dataloader is None:
            return

        self.adapter.eval()
        self.eval_reward_buffer.clear()

        with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
            all_samples: List[BaseSample] = []

            for batch in tqdm(
                self.test_dataloader,
                desc='Evaluating',
                disable=not self.show_progress_bar,
            ):
                generator = create_generator_by_prompt(batch['prompt'], self.training_args.seed)
                inference_kwargs = {
                    'compute_log_prob': False,
                    'generator': generator,
                    'trajectory_indices': None,  # No need to store trajectories during evaluation
                    **self.eval_args,
                }
                inference_kwargs.update(**batch)
                inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
                samples = self.adapter.inference(**inference_kwargs)
                all_samples.extend(samples)
                self.eval_reward_buffer.add_samples(samples)

            rewards = self.eval_reward_buffer.finalize(store_to_samples=True, split='pointwise')

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

    # ========================= Advantage Computation =========================

    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func=None,
    ) -> torch.Tensor:
        """Compute advantages — delegates to AdvantageProcessor.

        Args:
            samples: List of BaseSample instances
            rewards: Dict of reward_name to reward tensors aligned with samples
            store_to_samples: Whether to store computed advantages back to samples' extra_kwargs
            aggregation_func: Method to aggregate advantages within each group.
                Options: 'sum' (default GRPO), 'gdpo' (GDPO-style), or a custom callable.
        Returns:
            advantages: Tensor of shape (num_samples, ) with computed advantages
        """
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )

    # ========================= Main Training Loop =========================

    def start(self):
        """Main training loop."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            # Save checkpoint
            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    'checkpoints',
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            # Evaluation
            if (
                self.eval_args.eval_freq > 0
                and self.epoch % self.eval_args.eval_freq == 0
            ):
                self.evaluate()

            # Sampling: always use the "sampling" model snapshot
            with self.sampling_context():
                samples = self.sample()

            self.prepare_feedback(samples)
            self.optimize(samples)

            # Update EMA (if enabled), old model, and sampling model
            self.adapter.ema_step(step=self.epoch)
            self._update_old_model()
            self._update_sampling_model()

            self.epoch += 1

    def _blend_named_params(self, name: str, decay: float):
        """
        Blend a named parameter snapshot towards the current trainable parameters.

        Formula: ``snapshot = decay * snapshot + (1 - decay) * current``

        Args:
            name: Name of the parameter snapshot.
            decay: Blending coefficient. 0.0 = full copy, 1.0 = no change.
        """
        if decay <= 0.0:
            # Full copy from current params (no blending)
            self.adapter.update_named_parameters(name)
        elif decay >= 1.0:
            # Keep snapshot unchanged (fully offline)
            pass
        else:
            # Exponential blending: snapshot = decay * snapshot + (1 - decay) * current
            info = self.adapter._named_parameters[name]
            current_params = self.adapter._get_component_parameters(info.target_components)
            with torch.no_grad():
                for ema_param, param in zip(info.ema_wrapper.ema_parameters, current_params, strict=True):
                    ema_param.data.mul_(decay).add_(
                        param.detach().to(ema_param.device), alpha=(1.0 - decay)
                    )

    def _update_old_model(self):
        """
        Blend the old model snapshot towards the current trainable parameters.

        In the original CRD, controlled by ``decay_type`` (default: ``"0-0.25-0.001-0.5"``).
        """
        decay = compute_decay(self.step, self.old_model_decay)
        self._blend_named_params(self._OLD_PARAMS_NAME, decay)

        # Log decay value
        if self.accelerator.is_main_process:
            self.log_data({'train/old_model_decay': decay}, step=self.step)

    def _update_sampling_model(self):
        """
        Blend the sampling model snapshot towards the current trainable parameters.

        In the original CRD, controlled by ``decay_type2`` (default: preset 1 = ``(0, 0.0, 0.001, 0.5)``).
        """
        decay = compute_decay(self.step, self.sampling_model_decay)
        self._blend_named_params(self._SAMPLING_PARAMS_NAME, decay)

        # Log decay value
        if self.accelerator.is_main_process:
            self.log_data({'train/sampling_model_decay': decay}, step=self.step)

    # ========================= Sampling =========================

    def sample(self) -> List[BaseSample]:
        """Generate rollouts. Only keeps final latents (like NFT)."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []
        data_iter = iter(self.dataloader)

        with torch.no_grad(), self.autocast():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': False,
                    'trajectory_indices': [-1],  # Only keep final latents
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                # Deterministic D2H so reward_buffer sees CPU-resident samples
                # (no-op when offload_samples_to_cpu is False).
                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)

        return samples

    # ========================= Forward Pass Helpers =========================

    def _compute_crd_output(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
        guidance_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CRD forward pass for a single timestep.

        Args:
            batch: Batch dict with prompt embeddings etc.
            timestep: (B,) tensor in scheduler scale ``[0, 1000]``.
            noised_latents: Interpolated latents ``x_t = (1-σ) x_1 + σ noise`` with ``σ = t/1000``.
            guidance_scale: Override CFG scale. If None, uses the value from training_args
                (typically 1.0 for student training). Pass ``self.kl_cfg`` for teacher
                CFG inference — the model adapter will automatically do the double forward
                pass using ``negative_prompt_embeds`` / ``negative_pooled_prompt_embeds``
                from the batch if ``guidance_scale > 1.0``.

        Returns:
            Dict with ``noise_pred`` (velocity prediction), shape ``(B, C, H, W)``.
        """
        t_b = timestep.view(-1)  # Scheduler scale [0, 1000]
        device = self.accelerator.device

        forward_kwargs = {
            **self.training_args,
            't': t_b,
            't_next': torch.zeros_like(t_b),
            'latents': noised_latents,
            'compute_log_prob': False,
            'return_kwargs': ['noise_pred'],
            'noise_level': 0.0,
            **{
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
                if k not in ['all_latents', 'timesteps', 'advantage']
            },
        }
        if guidance_scale is not None:
            forward_kwargs['guidance_scale'] = guidance_scale
        forward_kwargs = filter_kwargs(self.adapter.forward, **forward_kwargs)
        output = self.adapter.forward(**forward_kwargs)
        return {'noise_pred': output.noise_pred}

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize rewards, compute advantages, and log advantage metrics."""
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')
        self.compute_advantages(samples, rewards, store_to_samples=True)
        adv_metrics = self.advantage_processor.pop_advantage_metrics()
        if adv_metrics:
            self.log_data(adv_metrics, step=self.step)

    # ========================= CRD Centering Loss =========================

    def _compute_crd_loss(
        self,
        adv_cur: torch.Tensor,
        adv_cur_rank: torch.Tensor,
        r_theta_gathered: torch.Tensor,
        r_theta_local: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the centered reward distillation (CRD) loss.

        Supports three modes depending on ``weight_temp``:
        - **Uniform** (``weight_temp < 0`` -> inf): Simple mean centering (single direction).
        - **Hard selection** (``weight_temp == 0``): Separate positive/negative sample pools.
        - **Softmax temperature** (``weight_temp > 0``): Dual-direction centering with
          ``softmax(adv/T)`` for positive direction and ``softmax(-adv/T)`` for negative direction.

        In the non-uniform case (``weight_temp >= 0``), the loss is the average of two
        directions: one centered on high-reward samples, one centered on low-reward samples.

        Args:
            adv_cur: Gathered advantages across all GPUs, shape ``(N,)``.
            adv_cur_rank: Local advantages for this rank, shape ``(B,)``.
            r_theta_gathered: Gathered implicit rewards across all GPUs, shape ``(N,)``.
            r_theta_local: Local implicit rewards for this rank, shape ``(B,)``.

        Returns:
            Unscaled CRD policy loss (scalar).
        """
        device = adv_cur.device
        weight_temp = torch.inf if self.weight_temp < 0 else self.weight_temp

        if weight_temp == torch.inf:
            # ---- Uniform weighting (single-direction centering) ----
            softmax_p = torch.softmax(adv_cur / weight_temp, dim=0)  # uniform
            adv_cur_avg = (adv_cur * softmax_p).sum(dim=0, keepdim=True)
            r_theta_avg = (r_theta_gathered * softmax_p).sum(dim=0, keepdim=True)

            Rc = adv_cur_rank - adv_cur_avg
            R_theta_c = r_theta_local - r_theta_avg.detach()

            if self.crd_loss_type == 'bce':
                ori_policy_loss = F.binary_cross_entropy_with_logits(
                    self.crd_beta * R_theta_c,
                    torch.sigmoid(Rc.detach()),
                    reduction='mean',
                )
            else:
                diff = self.crd_beta * R_theta_c - Rc
                ori_policy_loss = (diff ** 2).mean()

        else:
            # ---- Non-uniform: Dual-direction centering ----
            # Positive direction: weight towards higher-reward samples
            if weight_temp == 0:
                # Hard selection: only positive-advantage samples
                adv_plus_mask = (adv_cur > 0.0)
                if adv_plus_mask.sum() == 0:
                    softmax_p = torch.ones_like(adv_cur) / adv_cur.shape[0]
                else:
                    masked_adv = adv_cur.where(
                        adv_plus_mask, torch.tensor(float('-inf'), device=device)
                    )
                    softmax_p = torch.softmax(masked_adv, dim=0)
            else:
                softmax_p = torch.softmax(adv_cur / weight_temp, dim=0)

            # Negative direction: weight towards lower-reward samples
            if weight_temp == 0:
                # Hard selection: only negative-advantage samples
                adv_minus_mask = (adv_cur < 0.0)
                if adv_minus_mask.sum() == 0:
                    softmax_p_minus = torch.ones_like(adv_cur) / adv_cur.shape[0]
                else:
                    masked_adv = adv_cur.where(
                        adv_minus_mask, torch.tensor(float('-inf'), device=device)
                    )
                    softmax_p_minus = torch.softmax(masked_adv, dim=0)
            else:
                softmax_p_minus = torch.softmax(-adv_cur / weight_temp, dim=0)

            # Positive direction centering
            adv_cur_avg = (adv_cur * softmax_p).sum(dim=0, keepdim=True)
            r_theta_avg = (r_theta_gathered * softmax_p).sum(dim=0, keepdim=True)
            Rc = adv_cur_rank - adv_cur_avg
            R_theta_c = r_theta_local - r_theta_avg.detach()

            # Negative direction centering
            adv_cur_avg_minus = (adv_cur * softmax_p_minus).sum(dim=0, keepdim=True)
            r_theta_avg_minus = (r_theta_gathered * softmax_p_minus).sum(dim=0, keepdim=True)
            Rc_minus = adv_cur_rank - adv_cur_avg_minus
            R_theta_c_minus = r_theta_local - r_theta_avg_minus.detach()

            if self.crd_loss_type == 'bce':
                ori_policy_loss = 0.5 * F.binary_cross_entropy_with_logits(
                    self.crd_beta * R_theta_c,
                    torch.sigmoid(Rc.detach()),
                    reduction='mean',
                ) + 0.5 * F.binary_cross_entropy_with_logits(
                    self.crd_beta * R_theta_c_minus,
                    torch.sigmoid(Rc_minus.detach()),
                    reduction='mean',
                )
            else:
                diff = self.crd_beta * R_theta_c - Rc
                diff_minus = self.crd_beta * R_theta_c_minus - Rc_minus
                ori_policy_loss = 0.5 * (diff ** 2).mean() + 0.5 * (diff_minus ** 2).mean()

        return ori_policy_loss

    # ========================= Optimization =========================

    def optimize(self, samples: List[BaseSample]) -> None:
        """
        CRD optimization loop.

        For each timestep:
        1. Compute velocity predictions from current model, old model, and reference model.
        2. Estimate implicit reward r_theta from prediction errors.
        3. Center both external and implicit rewards (with optional dual-direction centering).
        4. Compute CRD loss matching centered rewards.
        5. Add KL regularization (with optional reward-adaptive scaling).

        Note on batching strategy:
            Unlike GRPO/NFT/AWM which use a per-batch interleaved pattern (lazy
            ``sample.to(device)`` reload to support ``offload_samples_to_cpu``),
            CRD uses a two-pass design:
              Pass 1: Pre-compute old model predictions for ALL batches.
              Pass 2: Train all batches using the pre-computed predictions.
            This may be refactored to the per-batch interleave pattern in the future.
        """
        for inner_epoch in range(self.training_args.num_inner_epochs):
            # CRD does not shuffle samples (needs same-prompt grouping for centering)
            # Re-group samples into batches
            sample_batches: List[Dict[str, Union[torch.Tensor, Any, List[Any]]]] = [
                BaseSample.stack(samples[i:i + self.training_args.per_device_batch_size])
                for i in range(0, len(samples), self.training_args.per_device_batch_size)
            ]

            # ==================== Pre-compute: Old V Predictions ====================
            self.adapter.rollout()
            with torch.no_grad(), self.autocast():
                for batch in tqdm(
                    sample_batches,
                    total=len(sample_batches),
                    desc=f'Epoch {self.epoch} Pre-computing Old V Predictions',
                    position=0,
                    disable=not self.show_progress_bar,
                ):
                    batch_size = batch['all_latents'].shape[0]
                    clean_latents = batch['all_latents'][:, -1]

                    # Sample timesteps: (T, B) in scheduler scale [0, 1000]
                    all_timesteps = self._sample_timesteps(batch_size)
                    batch['_all_timesteps'] = all_timesteps
                    batch['_all_random_noise'] = []

                    # Pre-compute old model predictions
                    old_v_pred_list = []
                    for t_idx in range(self.num_train_timesteps):
                        t_flat = all_timesteps[t_idx]  # (B,) scheduler scale [0, 1000]
                        sigma_broadcast = to_broadcast_tensor(flow_match_sigma(t_flat), clean_latents)
                        noise = randn_tensor(
                            clean_latents.shape,
                            device=clean_latents.device,
                            dtype=clean_latents.dtype,
                        )
                        batch['_all_random_noise'].append(noise)
                        noised_latents = (1 - sigma_broadcast) * clean_latents + sigma_broadcast * noise

                        if self.use_old_for_loss:
                            # Use old model snapshot
                            with self.adapter.use_named_parameters(self._OLD_PARAMS_NAME):
                                old_output = self._compute_crd_output(batch, t_flat, noised_latents)
                        else:
                            # Use reference model (original weights)
                            with self.adapter.use_ref_parameters():
                                old_output = self._compute_crd_output(batch, t_flat, noised_latents)
                        old_v_pred_list.append(old_output['noise_pred'].detach())

                    batch['_old_v_pred_list'] = old_v_pred_list

            # ==================== Training Loop ====================
            self.adapter.train()
            loss_info = defaultdict(list)

            with self.autocast():
                for batch in tqdm(
                    sample_batches,
                    total=len(sample_batches),
                    desc=f'Epoch {self.epoch} Training',
                    position=0,
                    disable=not self.show_progress_bar,
                ):
                    # Retrieve pre-computed data
                    batch_size = batch['all_latents'].shape[0]
                    clean_latents = batch['all_latents'][:, -1]
                    all_timesteps = batch['_all_timesteps']
                    all_random_noise = batch['_all_random_noise']
                    old_v_pred_list = batch['_old_v_pred_list']
                    # Iterate through timesteps
                    for t_idx in tqdm(
                        range(self.num_train_timesteps),
                        desc=f'Epoch {self.epoch} Timestep',
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    ):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            # 1. Prepare inputs
                            t_flat = all_timesteps[t_idx]  # (B,) scheduler scale [0, 1000]
                            sigma_broadcast = to_broadcast_tensor(flow_match_sigma(t_flat), clean_latents)
                            noise = all_random_noise[t_idx]
                            noised_latents = (1 - sigma_broadcast) * clean_latents + sigma_broadcast * noise
                            old_v_pred = old_v_pred_list[t_idx]
                            v_target = noise - clean_latents

                            # 2. Current model forward pass
                            output = self._compute_crd_output(batch, t_flat, noised_latents)
                            forward_pred = output['noise_pred']

                            # 3. Reference model forward pass (for KL)
                            # If kl_cfg > 1.0, the adapter's forward() will do CFG automatically:
                            # it concatenates [neg_embeds, pos_embeds] and computes:
                            #   noise_pred = uncond + kl_cfg * (cond - uncond)
                            # The negative embeddings come from the batch (negative_prompt_embeds,
                            # negative_pooled_prompt_embeds stored by SD3_5Sample during rollout).
                            with torch.no_grad(), self.adapter.use_ref_parameters():
                                cfg = self.kl_cfg if self.kl_cfg > 1.0 else None
                                ref_output = self._compute_crd_output(batch, t_flat, noised_latents, guidance_scale=cfg)
                                ref_pred = ref_output['noise_pred']

                            # 4. Compute implicit reward: r_theta = -(||pred_theta - v_target||^2 - ||pred_old - v_target||^2)
                            if self.adaptive_logp:
                                with torch.no_grad():
                                    weight_theta = (
                                        torch.abs(forward_pred.double() - v_target.double())
                                        .mean(dim=tuple(range(1, forward_pred.ndim)), keepdim=True)
                                        .clip(min=1e-5)
                                    )
                                    weight_old = (
                                        torch.abs(old_v_pred.double() - v_target.double())
                                        .mean(dim=tuple(range(1, old_v_pred.ndim)), keepdim=True)
                                        .clip(min=1e-5)
                                    )
                                r_theta = -(
                                    (forward_pred - v_target) ** 2 / weight_theta
                                    - (old_v_pred - v_target) ** 2 / weight_old
                                )
                            else:
                                r_theta = -(
                                    (forward_pred - v_target) ** 2
                                    - (old_v_pred - v_target) ** 2
                                )

                            # Reduce spatial dims to per-sample scalar
                            r_theta_local = r_theta.mean(dim=tuple(range(1, r_theta.ndim)))

                            # Gather r_theta across all GPUs for centering
                            r_theta_gathered = self.accelerator.gather(r_theta_local.detach()).to(
                                self.accelerator.device
                            )

                            # 5. Compute advantages for CRD centering
                            adv = batch['advantage']
                            adv_clip_range = self.training_args.adv_clip_range
                            adv_clipped = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])

                            # Normalize to [0, 1]
                            normalized_adv = (adv_clipped / max(adv_clip_range)) / 2.0 + 0.5
                            adv_cur_rank = torch.clamp(normalized_adv, 0, 1)

                            # Gather advantages across all GPUs
                            adv_cur = self.accelerator.gather(adv_cur_rank.detach()).to(
                                self.accelerator.device
                            )

                            # 6. Centered Reward Distillation loss (supports dual-direction centering)
                            ori_policy_loss = self._compute_crd_loss(
                                adv_cur=adv_cur,
                                adv_cur_rank=adv_cur_rank,
                                r_theta_gathered=r_theta_gathered,
                                r_theta_local=r_theta_local,
                            )

                            # Scale by adv_clip_max / beta for gradient magnitude normalization
                            policy_loss = (ori_policy_loss * adv_clip_range[1] / max(self.crd_beta, 1e-8)).mean()
                            loss = policy_loss

                            # 7. KL regularization against reference model
                            kl_div = ((forward_pred - ref_pred) ** 2).mean(
                                dim=tuple(range(1, forward_pred.ndim))
                            )

                            if self.reward_adaptive_kl:
                                # Linearly scale KL based on reward value
                                raw_reward = adv_cur_rank  # Already in [0, 1]
                                base_beta = 1e-4
                                min_coef = base_beta / max(self.kl_beta, 1e-8)
                                kl_loss = self.kl_beta * torch.mean((min_coef + raw_reward * (1 - min_coef)) * kl_div)
                            else:
                                kl_loss = self.kl_beta * kl_div.mean()

                            loss = loss + kl_loss

                            # 8. Logging
                            loss_info['policy_loss'].append(policy_loss.detach())
                            loss_info['unweighted_policy_loss'].append(ori_policy_loss.mean().detach())
                            loss_info['kl_div'].append(kl_div.mean().detach())
                            loss_info['kl_loss'].append(kl_loss.detach())
                            loss_info['r_theta_mean'].append(r_theta_local.mean().detach())
                            loss_info['loss'].append(loss.detach())

                            if self.use_old_for_loss:
                                old_kl = ((old_v_pred - ref_pred) ** 2).mean(
                                    dim=tuple(range(1, old_v_pred.ndim))
                                ).mean()
                                loss_info['old_kl_div'].append(old_kl.detach())
                                old_deviate = ((forward_pred - old_v_pred) ** 2).mean()
                                loss_info['old_deviate'].append(old_deviate.detach())

                            # 9. Backward and optimizer step
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                # Log accumulated loss info
                                loss_info = reduce_loss_info(self.accelerator, loss_info)
                                loss_info['grad_norm'] = grad_norm
                                self.log_data(
                                    {f'train/{k}': v for k, v in loss_info.items()},
                                    step=self.step,
                                )
                                self.step += 1
                                loss_info = defaultdict(list)
