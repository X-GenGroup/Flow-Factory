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

# src/flow_factory/trainers/rl/crd.py
"""
Centered Reward Distillation (CRD) Trainer.
Reference:
[1] Diffusion Reinforcement Learning via Centered Reward Distillation
    - https://arxiv.org/abs/2603.14128
"""

import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ...contracts.reward_overlap import GROUP_RELATIVE_REWARD_OPTIMIZATION_OVERLAP
from ...hparams import CRDTrainingArguments
from ...rewards import RewardBuffer, RewardTile, RewardTilePlan
from ...samples import (
    BaseSample,
    ComponentTimes,
    LatentState,
    NoisedState,
    StackedSampleBatch,
)
from ...utils.base import create_generator, create_generator_by_prompt
from ...utils.dist import gather_aligned_floating_tensors
from ...utils.logger_utils import setup_logger
from ..abc import BaseTrainer
from ..common.state_validation import require_latent_state, state_batch_size
from ..forward_process import forward_velocity_state

logger = setup_logger(__name__)


@dataclass
class _CRDStep:
    """Pass-1 state for one training timestep, replayed verbatim by pass 2.

    Only the forward-process *inputs* are kept: pass 2 reapplies ``noise`` to the
    reloaded terminal state through the adapter's RNG-free application hook, so
    the noised state and the target velocity are never stored twice.
    """

    times: ComponentTimes
    noise: LatentState
    old_velocity: LatentState


@dataclass
class _CRDBatch:
    """A stacked micro-batch plus its pass-1 pre-computed old-model targets.

    CRD uses a two-pass design: pass 1 pre-computes the old-model velocity
    predictions for every micro-batch (against a frozen snapshot), pass 2 trains
    on them. This bundles the device-resident batch with those per-timestep
    values so pass 2 has typed, named access (``prepared.steps``) instead of
    keying scratch onto the shared batch type.
    """

    batch: StackedSampleBatch
    steps: List[_CRDStep]


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
    "none": (0, 0.0, 0.0, 0.0),
    "slow": (0, 0.0, 0.001, 0.5),
    "medium": (75, 0.0, 0.0075, 0.999),
    "offline": (0, 1.0, 0.0, 1.0),
    "fast": (0, 0.0, 0.02, 0.99),
    "moderate": (0, 0.0, 0.01, 0.5),
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
    elif isinstance(decay_type, str) and "-" in decay_type:
        parts = decay_type.split("-")
        assert (
            len(parts) == 4
        ), f"Decay string format must be 'start_step-start_value-slope-end_value', got: {decay_type}"
        start_step, start_value, slope, end_value = (
            float(parts[0]),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
        )
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

    # Decoupled paradigm: lossy rollout acceleration is permitted (constraints.md #7).
    paradigm = "decoupled"
    reward_optimization_overlap_contract = GROUP_RELATIVE_REWARD_OPTIMIZATION_OVERLAP

    _OLD_PARAMS_NAME = "_crd_old"
    _SAMPLING_PARAMS_NAME = "_crd_sampling"
    runtime_child_names = (_OLD_PARAMS_NAME, _SAMPLING_PARAMS_NAME)

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
        if self.kl_type != "v-based":
            logger.warning(
                f"CRD-Trainer only supports 'v-based' KL loss, got {self.kl_type}, switching to 'v-based'."
            )
            self.kl_type = "v-based"

    # ========================= Initialization =========================

    def _initialize_snapshots(self) -> None:
        """
        Initialize and register both model snapshots before exact state resume.

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
        self._register_named_parameter_runtime_child(self._OLD_PARAMS_NAME)
        logger.info("CRD: Initialized 'old' model snapshot for implicit reward estimation.")

        # Sampling model snapshot (for off-policy rollout generation)
        self.adapter.add_named_parameters(
            name=self._SAMPLING_PARAMS_NAME,
            device=ref_device,
        )
        self._register_named_parameter_runtime_child(self._SAMPLING_PARAMS_NAME)
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

    # ========================= Advantage Computation =========================

    # ========================= Main Training Loop =========================

    def _after_acquisition_cycle(self) -> None:
        """Advance CRD's two auxiliary snapshots after each rollout iteration."""
        self._update_old_model()
        self._update_sampling_model()

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
                for ema_param, param in zip(
                    info.ema_wrapper.ema_parameters, current_params, strict=True
                ):
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
            self.log_data({"train/old_model_decay": decay}, step=self.step)

    def _update_sampling_model(self):
        """
        Blend the sampling model snapshot towards the current trainable parameters.

        In the original CRD, controlled by ``decay_type2`` (default: preset 1 = ``(0, 0.0, 0.001, 0.5)``).
        """
        decay = compute_decay(self.step, self.sampling_model_decay)
        self._blend_named_params(self._SAMPLING_PARAMS_NAME, decay)

        # Log decay value
        if self.accelerator.is_main_process:
            self.log_data({"train/sampling_model_decay": decay}, step=self.step)

    # ========================= Sampling =========================

    def sample(self) -> List[BaseSample]:
        """Generate rollouts for CRD (final latents only)."""
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=False,
            trajectory_indices=[-1],
        )

    # ========================= Forward Pass Helpers =========================

    def _reference_cfg_overrides(self) -> Dict[str, Any]:
        """CFG override for the teacher forward, empty below the CFG threshold.

        With ``kl_cfg > 1.0`` the adapter runs its own double forward using the
        batch's negative embeddings; otherwise the guidance scale keeps coming
        from ``training_args``, exactly as before the migration.

        Returns:
            Forward-argument overrides for the reference pass.
        """
        if self.kl_cfg > 1.0:
            return {"guidance_scale": self.kl_cfg}
        return {}

    def _rebuild_noised_state(
        self,
        clean_state: LatentState,
        step: _CRDStep,
        *,
        timestep_index: Optional[int] = None,
    ) -> NoisedState:
        """Reapply a pass-1 noise draw to the terminal state, consuming no RNG.

        Pass 1 and pass 2 resolve the terminal state independently, so a stored
        noise that no longer matches the reloaded geometry would silently
        broadcast into a different forward process. Validate both states against
        the declared component order first — a missing or extra component must
        report the timestep it belongs to instead of surfacing a bare
        ``KeyError`` from the lookup below — then check the per-component
        geometry.

        Args:
            clean_state: Terminal clean state reloaded for pass 2.
            step: Pass-1 state carrying the component times and the drawn noise.
            timestep_index: Training timestep this step replays, reported by
                validation errors.

        Returns:
            The same noised state and target velocity pass 1 evaluated.
        """
        clean_state = require_latent_state(
            self,
            clean_state,
            f"pass-2 terminal clean state at timestep_index={timestep_index}",
        )
        stored_noise = require_latent_state(
            self, step.noise, f"pass-1 stored noise at timestep_index={timestep_index}"
        )
        for name in self.adapter.trajectory_component_order:
            clean = clean_state.components[name]
            noise = stored_noise.components[name]
            if (
                noise.shape != clean.shape
                or noise.dtype != clean.dtype
                or noise.device != clean.device
            ):
                raise ValueError(
                    f"expected {type(self).__name__} pass-1 noise at "
                    f"timestep_index={timestep_index} component {name!r} to match the "
                    f"reloaded terminal state ({tuple(clean.shape)}, {clean.dtype}, "
                    f"{clean.device}), received ({tuple(noise.shape)}, {noise.dtype}, "
                    f"{noise.device})"
                )
        return self.adapter.apply_forward_process_noise(clean_state, step.times, stored_noise)

    def _precompute_old_velocities(self, batch: StackedSampleBatch) -> _CRDBatch:
        """Pass 1: sample times, draw one noise state per time, record old velocities.

        The old model is a frozen snapshot, so its predictions are independent of
        the pass-2 weight updates; only the times, the noise and the detached old
        velocity survive into pass 2.

        Args:
            batch: Collated sample batch supplying conditioning arguments.

        Returns:
            The batch bundled with one precomputed step per training timestep.
        """
        clean_state = self.adapter.get_terminal_state(batch)
        batch_size = state_batch_size(self, clean_state, "terminal clean state")
        steps: List[_CRDStep] = []
        with torch.no_grad(), self.autocast():
            all_timesteps = self._sample_timesteps(batch_size)  # (T, B)
            for t_idx in range(self.num_train_timesteps):
                times = self.adapter.build_training_component_times(
                    all_timesteps[t_idx], batch=batch
                )
                noised = self.adapter.add_forward_process_noise(clean_state, times)
                old_parameters = (
                    self.adapter.use_named_parameters(self._OLD_PARAMS_NAME)
                    if self.use_old_for_loss
                    else self.adapter.use_ref_parameters()
                )
                with old_parameters:
                    velocity = forward_velocity_state(
                        self, batch, noised.state, times, source="old policy"
                    )
                steps.append(
                    _CRDStep(
                        times=times,
                        noise=noised.noise,
                        old_velocity=LatentState(
                            {name: value.detach() for name, value in velocity.components.items()}
                        ),
                    )
                )
        return _CRDBatch(batch=batch, steps=steps)

    # ========================= Reward and KL Helpers =========================

    def _implicit_reward(
        self,
        velocity: LatentState,
        old_velocity: LatentState,
        noised: NoisedState,
    ) -> torch.Tensor:
        """Estimate the per-sample implicit reward from velocity prediction errors.

        ``r = -((current - target)^2 - (old - target)^2)`` per element. Under
        ``adaptive_logp`` each component's squared error is divided by that
        component's own detached mean absolute deviation (clipped at ``1e-5``)
        *before* the global reduction, which keeps the legacy
        elementwise-divide-then-average order.

        Args:
            velocity: Current-policy velocity per component.
            old_velocity: Old-snapshot velocity per component.
            noised: Forward-noised state carrying the per-component target velocity.

        Returns:
            Per-sample implicit reward of shape ``(B,)``.
        """
        component_names = self.adapter.trajectory_component_order
        target = noised.target_velocity
        rewards: Dict[str, torch.Tensor] = {}
        if self.adaptive_logp:
            with torch.no_grad():
                current_weights = self.adapter.reduce_component_latent_values(
                    {
                        name: torch.abs(
                            velocity.components[name].double() - target.components[name].double()
                        )
                        for name in component_names
                    },
                    state=noised.state,
                )
                old_weights = self.adapter.reduce_component_latent_values(
                    {
                        name: torch.abs(
                            old_velocity.components[name].double()
                            - target.components[name].double()
                        )
                        for name in component_names
                    },
                    state=noised.state,
                )
            for name in component_names:
                component_target = target.components[name]
                broadcast_shape = (-1,) + (1,) * (component_target.ndim - 1)
                current_weight = current_weights[name].clip(min=1e-5).reshape(broadcast_shape)
                old_weight = old_weights[name].clip(min=1e-5).reshape(broadcast_shape)
                rewards[name] = -(
                    (velocity.components[name] - component_target) ** 2 / current_weight
                    - (old_velocity.components[name] - component_target) ** 2 / old_weight
                )
        else:
            for name in component_names:
                component_target = target.components[name]
                rewards[name] = -(
                    (velocity.components[name] - component_target) ** 2
                    - (old_velocity.components[name] - component_target) ** 2
                )
        return self.adapter.reduce_latent_values(rewards, state=noised.state)

    def _kl_loss(self, kl_div: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """Scale the per-sample KL surrogate into the loss term.

        Args:
            kl_div: Per-sample KL surrogate of shape ``(B,)``.
            reward: Per-sample normalized reward in ``[0, 1]``, shape ``(B,)``.

        Returns:
            Scalar KL loss contribution.
        """
        if self.reward_adaptive_kl:
            # Linearly scale KL based on reward value
            base_beta = 1e-4
            min_coef = base_beta / max(self.kl_beta, 1e-8)
            return self.kl_beta * torch.mean((min_coef + reward * (1 - min_coef)) * kl_div)
        return self.kl_beta * kl_div.mean()

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

            if self.crd_loss_type == "bce":
                ori_policy_loss = F.binary_cross_entropy_with_logits(
                    self.crd_beta * R_theta_c,
                    torch.sigmoid(Rc.detach()),
                    reduction="mean",
                )
            else:
                diff = self.crd_beta * R_theta_c - Rc
                ori_policy_loss = (diff**2).mean()

        else:
            # ---- Non-uniform: Dual-direction centering ----
            # Positive direction: weight towards higher-reward samples
            if weight_temp == 0:
                # Hard selection: only positive-advantage samples
                adv_plus_mask = adv_cur > 0.0
                if adv_plus_mask.sum() == 0:
                    softmax_p = torch.ones_like(adv_cur) / adv_cur.shape[0]
                else:
                    masked_adv = adv_cur.where(
                        adv_plus_mask, torch.tensor(float("-inf"), device=device)
                    )
                    softmax_p = torch.softmax(masked_adv, dim=0)
            else:
                softmax_p = torch.softmax(adv_cur / weight_temp, dim=0)

            # Negative direction: weight towards lower-reward samples
            if weight_temp == 0:
                # Hard selection: only negative-advantage samples
                adv_minus_mask = adv_cur < 0.0
                if adv_minus_mask.sum() == 0:
                    softmax_p_minus = torch.ones_like(adv_cur) / adv_cur.shape[0]
                else:
                    masked_adv = adv_cur.where(
                        adv_minus_mask, torch.tensor(float("-inf"), device=device)
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

            if self.crd_loss_type == "bce":
                ori_policy_loss = 0.5 * F.binary_cross_entropy_with_logits(
                    self.crd_beta * R_theta_c,
                    torch.sigmoid(Rc.detach()),
                    reduction="mean",
                ) + 0.5 * F.binary_cross_entropy_with_logits(
                    self.crd_beta * R_theta_c_minus,
                    torch.sigmoid(Rc_minus.detach()),
                    reduction="mean",
                )
            else:
                diff = self.crd_beta * R_theta_c - Rc
                diff_minus = self.crd_beta * R_theta_c_minus - Rc_minus
                ori_policy_loss = 0.5 * (diff**2).mean() + 0.5 * (diff_minus**2).mean()

        return ori_policy_loss

    # ========================= Optimization =========================

    def _precompute_crd_batches(self, samples: List[BaseSample]) -> List[_CRDBatch]:
        """Run CRD pass 1 for a sample span without changing trainable weights."""
        sample_batches: List[_CRDBatch] = []
        num_batches = (
            len(samples) + self.training_args.per_device_batch_size - 1
        ) // self.training_args.per_device_batch_size
        self.adapter.rollout()
        for batch in tqdm(
            self._iter_prefetched_batches(samples, self.training_args.per_device_batch_size),
            total=num_batches,
            desc=f"Epoch {self.epoch} Pre-computing Old V Predictions",
            position=0,
            disable=not self.show_progress_bar,
        ):
            sample_batches.append(self._precompute_old_velocities(batch))
        return sample_batches

    def _prepare_reward_optimization_overlap(
        self,
        samples: List[BaseSample],
        plan: RewardTilePlan,
    ) -> Dict[int, List[_CRDBatch]]:
        """Finish CRD's immutable old-policy pass while rewards are pending."""
        prepared = self._precompute_crd_batches(samples)
        expected = len(plan.tiles) * plan.batches_per_tile
        if len(prepared) != expected:
            raise RuntimeError(
                "CRD overlap precompute did not match tile geometry: "
                f"prepared_batches={len(prepared)}, expected={expected}"
            )
        return {
            tile.tile_id: prepared[
                tile.tile_id * plan.batches_per_tile : (tile.tile_id + 1) * plan.batches_per_tile
            ]
            for tile in plan.tiles
        }

    def _optimize_reward_overlap_tile(
        self,
        tile: RewardTile,
        samples: List[BaseSample],
        context: Dict[int, List[_CRDBatch]],
    ) -> None:
        """Train CRD pass 2 from the acquisition-wide immutable pass-1 state."""
        if hasattr(self, "_crd_overlap_precomputed_batches"):
            raise RuntimeError("CRD overlap precomputed-batch override is already active")
        prepared_batches = context[tile.tile_id]
        sample_offset = 0
        for prepared in prepared_batches:
            batch_size = len(prepared.batch.samples)
            sample_slice = samples[sample_offset : sample_offset + batch_size]
            if len(sample_slice) != batch_size:
                raise RuntimeError(
                    "CRD overlap advantage attachment did not match the prepared batch: "
                    f"needed={batch_size}, remaining={len(sample_slice)}"
                )
            advantages = []
            for sample in sample_slice:
                if "advantage" not in sample.extra_kwargs:
                    raise RuntimeError(
                        "CRD overlap expected tile feedback to store an advantage before pass 2"
                    )
                advantage = torch.as_tensor(
                    sample.extra_kwargs["advantage"],
                    device=self.accelerator.device,
                )
                if advantage.ndim != 0:
                    raise ValueError(
                        "CRD overlap expected scalar advantages, received "
                        f"shape={tuple(advantage.shape)}"
                    )
                advantages.append(advantage)
            prepared.batch["advantage"] = torch.stack(advantages)
            sample_offset += batch_size
        if sample_offset != len(samples):
            raise RuntimeError(
                "CRD overlap prepared batches did not consume every tile sample: "
                f"consumed={sample_offset}, samples={len(samples)}"
            )
        self._crd_overlap_precomputed_batches = prepared_batches
        try:
            self.optimize(samples)
        finally:
            del self._crd_overlap_precomputed_batches

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
            # CRD does not shuffle samples (needs same-prompt grouping for centering).
            # ==================== Pre-compute: Old V Predictions ====================
            # Prefetch each micro-batch here so its H2D overlaps the heavy old-V
            # forward, then keep the device-resident batch for pass 2 (it is
            # reused, not reloaded). The old model is a frozen snapshot
            # (_OLD_PARAMS_NAME), so per-batch old-V is independent of pass-2
            # weight updates.
            sample_batches = getattr(self, "_crd_overlap_precomputed_batches", None)
            if sample_batches is None:
                sample_batches = self._precompute_crd_batches(samples)

            # ==================== Training Loop ====================
            self.adapter.train()
            loss_info = defaultdict(list)

            for prepared in tqdm(
                sample_batches,
                total=len(sample_batches),
                desc=f"Epoch {self.epoch} Training",
                position=0,
                disable=not self.show_progress_bar,
            ):
                # Retrieve pre-computed data
                batch = prepared.batch
                clean_state = self.adapter.get_terminal_state(batch)
                # Iterate through timesteps
                for t_idx in tqdm(
                    range(self.num_train_timesteps),
                    desc=f"Epoch {self.epoch} Timestep",
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                ):
                    with self.accumulate_gradients():
                        # 1. Replay the pass-1 forward process without drawing noise
                        step = prepared.steps[t_idx]
                        noised = self._rebuild_noised_state(clean_state, step, timestep_index=t_idx)
                        old_velocity = step.old_velocity

                        # 2. Current model forward pass
                        with self.autocast():
                            velocity = forward_velocity_state(
                                self, batch, noised.state, step.times, source="policy"
                            )

                        # 3. Reference model forward pass (for KL)
                        # If kl_cfg > 1.0, the adapter's forward() will do CFG automatically:
                        # it concatenates [neg_embeds, pos_embeds] and computes:
                        #   velocity = uncond + kl_cfg * (cond - uncond)
                        # The negative embeddings come from the batch (negative_prompt_embeds,
                        # negative_pooled_prompt_embeds stored by SD3_5Sample during rollout).
                        with torch.no_grad(), self.adapter.use_ref_parameters(), self.autocast():
                            ref_velocity = forward_velocity_state(
                                self,
                                batch,
                                noised.state,
                                step.times,
                                source="reference",
                                **self._reference_cfg_overrides(),
                            )

                        # 4. Compute implicit reward: r_theta = -(||pred_theta - v_target||^2 - ||pred_old - v_target||^2)
                        r_theta_local = self._implicit_reward(velocity, old_velocity, noised)

                        # 5. Compute advantages for CRD centering
                        adv = batch["advantage"]
                        adv_clip_range = self.training_args.adv_clip_range
                        adv_clipped = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])

                        # Normalize to [0, 1]
                        normalized_adv = (adv_clipped / max(adv_clip_range)) / 2.0 + 0.5
                        adv_cur_rank = torch.clamp(normalized_adv, 0, 1)

                        # Centering needs both global vectors. Pack them into one
                        # aligned gather instead of launching two collectives.
                        gathered_centering = gather_aligned_floating_tensors(
                            self.accelerator,
                            {
                                "advantage": adv_cur_rank.detach(),
                                "implicit_reward": r_theta_local.detach(),
                            },
                        )
                        adv_cur = gathered_centering["advantage"]
                        r_theta_gathered = gathered_centering["implicit_reward"]

                        # 6. Centered Reward Distillation loss (supports dual-direction centering)
                        ori_policy_loss = self._compute_crd_loss(
                            adv_cur=adv_cur,
                            adv_cur_rank=adv_cur_rank,
                            r_theta_gathered=r_theta_gathered,
                            r_theta_local=r_theta_local,
                        )

                        # Scale by adv_clip_max / beta for gradient magnitude normalization
                        policy_loss = (
                            ori_policy_loss * adv_clip_range[1] / max(self.crd_beta, 1e-8)
                        ).mean()
                        loss = policy_loss

                        # 7. KL regularization against reference model
                        with self.autocast():
                            kl_div = self._velocity_kl(velocity, ref_velocity, noised)
                            kl_loss = self._kl_loss(kl_div, adv_cur_rank)
                            loss = loss + kl_loss

                        # 8. Logging
                        loss_info["policy_loss"].append(policy_loss.detach())
                        loss_info["unweighted_policy_loss"].append(ori_policy_loss.mean().detach())
                        loss_info["kl_div"].append(kl_div.mean().detach())
                        loss_info["kl_loss"].append(kl_loss.detach())
                        loss_info["r_theta_mean"].append(r_theta_local.mean().detach())
                        loss_info["loss"].append(loss.detach())

                        if self.use_old_for_loss:
                            old_kl = self._velocity_kl(old_velocity, ref_velocity, noised).mean()
                            loss_info["old_kl_div"].append(old_kl.detach())
                            old_deviate = self._velocity_kl(velocity, old_velocity, noised).mean()
                            loss_info["old_deviate"].append(old_deviate.detach())

                        # 9. Backward and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            loss_info = self._apply_optimizer_step(loss_info)
