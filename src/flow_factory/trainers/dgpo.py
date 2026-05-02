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

# src/flow_factory/trainers/dgpo.py
"""
DGPO (Direct Group Preference Optimization) Trainer.

Reference:
[1] DGPO: Reinforcing Diffusion Models by Direct Group Preference Optimization
    - ICLR 2026
"""

import os
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
import tqdm as tqdm_

from diffusers.utils.torch_utils import randn_tensor

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ..hparams import DGPOTrainingArguments
from ..samples import BaseSample
from ..utils.base import (
    create_generator,
    create_generator_by_prompt,
    filter_kwargs,
    to_broadcast_tensor,
)
from ..utils.dist import reduce_loss_info
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import TimeSampler, flow_match_sigma
from .abc import BaseTrainer

logger = setup_logger(__name__)

# Seed-namespace tags — appended to ``create_generator(...)`` integer keys so
# independent RNG streams (shared timesteps / shared per-group noise) never
# collide even if the other keys happen to coincide.  Independent (non-shared)
# training noise uses the global default RNG and is not seeded.
_SEED_TAG_SHARED_TIMESTEPS = 1
_SEED_TAG_SHARED_NOISE = 2


class DGPOGroupInfo(TypedDict):
    """Per-minibatch group labels for scatter-add in :meth:`_compute_group_dgpo_loss`."""

    local_group_indices: torch.Tensor
    num_groups: int


class _PreppedBatch(TypedDict):
    """Unpacked view of one entry from ``training_batches``.

    Created once per ``tb`` by :meth:`DGPOTrainer._prep_training_batch` to
    avoid repeating the same six-field unpack on every timestep inside
    :meth:`DGPOTrainer._optimize_step`.
    """

    batch: Dict[str, Any]
    clean_latents: torch.Tensor
    adv: torch.Tensor
    group_info: DGPOGroupInfo
    timesteps: torch.Tensor
    samples_slice: List[BaseSample]
    inner_epoch: int


class _NoisedInputs(TypedDict):
    """``(t_flat, noised, target_v)`` for a single ``(prepped_batch, t_idx)`` pair."""

    t_flat: torch.Tensor
    noised: torch.Tensor
    target_v: torch.Tensor


class _VelocityPredictions(TypedDict):
    """Output bundle of :meth:`DGPOTrainer._forward_velocities`.

    ``model_v`` carries autograd; ``old_v`` / ``ref_v`` / ``ref_dgpo_v`` are
    detached.  ``old_v`` and ``ref_v`` are computed on demand (``None`` when
    the corresponding feature — clipping / KL / ``use_ema_ref`` — is off);
    ``ref_dgpo_v`` is always set (aliased to ``old_v`` if
    ``use_ema_ref=True`` and to ``ref_v`` otherwise).
    """

    model_v: torch.Tensor
    old_v: Optional[torch.Tensor]
    ref_v: Optional[torch.Tensor]
    ref_dgpo_v: torch.Tensor


class DGPOTrainer(BaseTrainer):
    """DGPO Trainer: Direct Group Preference Optimization for diffusion models.

    Uses a group-level DPO loss instead of per-sample PPO ratio loss.
    Partitions samples into groups by prompt, computes DSM losses vs a frozen
    reference model, aggregates group-level preference signals via sigmoid,
    and applies PPO-style DSM clipping using an EMA "old policy".

    Cross-rank determinism is achieved via ``create_generator`` (``utils.base``)
    — every random draw is seeded from an explicit integer tuple so all ranks
    produce byte-identical shared timesteps and per-group noise without any
    ``dist.broadcast``/``torch.random.fork_rng`` side effects.

    Reference: [1] DGPO: Reinforcing Diffusion Models by Direct Group Preference Optimization (ICLR 2026).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ta: DGPOTrainingArguments = self.training_args  # type: ignore[assignment]
        self.training_args = ta

        # DGPO is only valid under GroupDistributedSampler — `hparams.Arguments.
        # _resolve_sampler_type` hard-forces this.  This assert is a
        # belt-and-suspenders guard against future code paths bypassing hparams.
        assert self.config.data_args.sampler_type == "group_distributed", (
            "DGPOTrainer requires sampler_type='group_distributed'; "
            "hparams.Arguments._resolve_sampler_type should have enforced this, "
            f"got sampler_type={self.config.data_args.sampler_type!r}."
        )

        # DGPO core
        self.dpo_beta = ta.dpo_beta
        self.use_shared_noise = ta.use_shared_noise
        self.clip_dsm = ta.clip_dsm
        self.clip_kl = ta.clip_kl
        self.switch_ema_ref = ta.switch_ema_ref
        self.kl_cfg = ta.kl_cfg
        self.use_ema_ref = ta.use_ema_ref

        # Timestep sampling
        self.off_policy = ta.off_policy
        self.time_sampling_strategy = ta.time_sampling_strategy
        self.time_shift = ta.time_shift
        self.num_train_timesteps = ta.num_train_timesteps
        self.timestep_range = ta.timestep_range

        # KL regularisation
        self.kl_beta = ta.kl_beta
        self.kl_type = ta.kl_type
        if self.kl_type != "v-based":
            logger.warning(
                f"DGPOTrainer only supports 'v-based' KL loss (got {self.kl_type!r}); "
                "switching to 'v-based'."
            )
            self.kl_type = "v-based"

        # Old-policy EMA ref (fast-tracking EMA separate from sampling EMA)
        self.ema_ref_max_decay = ta.ema_ref_max_decay
        self.ema_ref_ramp_rate = ta.ema_ref_ramp_rate
        self._requires_ema_ref = self.clip_dsm or self.clip_kl or self.use_ema_ref
        if self._requires_ema_ref:
            ema_ref_device = (
                self.accelerator.device if ta.ema_ref_device == "cuda" else torch.device("cpu")
            )
            self.adapter.add_named_parameters(
                "ema_ref",
                device=ema_ref_device,
                overwrite=True,
            )
            logger.info(
                f"Initialized old-policy EMA ref on {ema_ref_device} "
                f"(max_decay={self.ema_ref_max_decay}, ramp_rate={self.ema_ref_ramp_rate})."
            )

    # =========================== Properties ============================
    @property
    def enable_kl_loss(self) -> bool:
        """Whether the v-based KL penalty is active."""
        return self.kl_beta > 0.0

    # =========================== Parameter-swap Contexts ============================
    @contextmanager
    def sampling_context(self):
        """Swap to the appropriate parameters during sampling.

        Mirrors the reference DGPO (``global_step > switch_ema_ref``): once
        warmup is done we sample under the fast-tracking ``ema_ref``; before
        that either keep current parameters or use the slow sampling EMA if
        ``off_policy=True``.
        """
        if self.step > self.switch_ema_ref and self._requires_ema_ref:
            with self.adapter.use_named_parameters("ema_ref"):
                yield
        elif self.off_policy:
            with self.adapter.use_ema_parameters():
                yield
        else:
            yield

    @contextmanager
    def _ema_ref_forward_context(self):
        """Swap to ``ema_ref`` for the old-policy forward pass.

        Only called when :attr:`_requires_ema_ref` is ``True``; there is no
        fallback — the caller must gate the call itself.
        """
        with self.adapter.use_named_parameters("ema_ref"):
            yield

    # =========================== EMA-ref Update ============================
    def _update_ema_ref(self, step: int) -> None:
        """Update the old-policy EMA ref with adaptive decay.

        Reproduces the reference DGPO per-step update::

            decay       = min(max_decay, ramp_rate * step)
            ema_ref_new = decay * ema_ref_old + (1 - decay) * current
        """
        if not self._requires_ema_ref:
            return

        decay = min(self.ema_ref_max_decay, self.ema_ref_ramp_rate * step)
        one_minus_decay = 1.0 - decay

        ema_params = self.adapter.get_named_parameters("ema_ref")
        current_params = self.adapter.get_trainable_parameters()

        with torch.no_grad():
            for ema_p, cur_p in zip(ema_params, current_params, strict=True):
                ema_p.mul_(decay).add_(cur_p.detach().to(ema_p.device), alpha=one_minus_decay)

    # =========================== Timestep Sampling ============================
    def _sample_timesteps(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample scheduler-scale training timesteps ``[0, 1000]``.

        Args:
            batch_size: Size of the broadcast batch dimension.
            generator: Optional ``torch.Generator``. When supplied, the draw is
                deterministic and cross-rank-reproducible for any strategy;
                threaded straight through to :class:`TimeSampler`.

        Returns:
            Tensor of shape ``(num_train_timesteps, batch_size)``.
        """
        device = self.accelerator.device
        strategy = self.time_sampling_strategy.lower()
        available = [
            "logit_normal",
            "uniform",
            "discrete",
            "discrete_with_init",
            "discrete_wo_init",
        ]

        if strategy == "logit_normal":
            return TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
                stratified=True,
                generator=generator,
            )
        if strategy == "uniform":
            return TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
                generator=generator,
            )
        if strategy.startswith("discrete"):
            discrete_config = {
                "discrete": (True, False),
                "discrete_with_init": (True, True),
                "discrete_wo_init": (False, False),
            }
            if strategy not in discrete_config:
                raise ValueError(
                    f"Unknown time_sampling_strategy: {strategy!r}. Available: {available}"
                )
            include_init, force_init = discrete_config[strategy]
            return TimeSampler.discrete(
                batch_size=batch_size,
                num_train_timesteps=self.num_train_timesteps,
                scheduler_timesteps=self.adapter.scheduler.timesteps,
                timestep_range=self.timestep_range,
                include_init=include_init,
                force_init=force_init,
                generator=generator,
            )

        raise ValueError(f"Unknown time_sampling_strategy: {strategy!r}. Available: {available}")

    def _sample_shared_timesteps(self, inner_epoch: int) -> torch.Tensor:
        """Sample ``num_train_timesteps`` scheduler-scale timesteps, identical on all ranks.

        All ranks call :func:`create_generator` with the same integer tuple
        ``(seed, epoch, inner_epoch, tag)`` and hand the resulting generator
        to :class:`TimeSampler` — no broadcast, no global-RNG fork, and any
        configured ``time_sampling_strategy`` (continuous or discrete) works.

        Returns:
            Tensor of shape ``(num_train_timesteps,)`` in scheduler scale
            ``[0, 1000]``.
        """
        gen = create_generator(
            self.training_args.seed,
            self.epoch,
            inner_epoch,
            _SEED_TAG_SHARED_TIMESTEPS,
        )
        return self._sample_timesteps(batch_size=1, generator=gen).squeeze(-1)

    # =========================== Forward Pass ============================
    def _compute_dgpo_output(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run the adapter for DGPO at a single timestep.

        Args:
            batch: Stacked batch dict (must contain at least prompt embeddings).
            timestep: ``(B,)`` timestep in scheduler scale ``[0, 1000]``.
            noised_latents: ``x_t`` for the current sample.
            guidance_scale: CFG scale; use ``1.0`` for the uncond branch.

        Returns:
            Tensor ``(B, C, ...)``: the velocity prediction ``noise_pred``.
        """
        t = timestep.view(-1)
        forward_kwargs = {
            **self.training_args,
            "t": t,
            "t_next": torch.zeros_like(t),
            "latents": noised_latents,
            "compute_log_prob": False,
            "return_kwargs": ["noise_pred"],
            "noise_level": 0.0,
            "guidance_scale": guidance_scale,
            **{
                k: v for k, v in batch.items() if k not in ("all_latents", "timesteps", "advantage")
            },
        }
        forward_kwargs = filter_kwargs(self.adapter.forward, **forward_kwargs)
        noise_pred = self.adapter.forward(**forward_kwargs).noise_pred
        assert noise_pred is not None, (
            "adapter.forward must return `noise_pred` for DGPO "
            "(ensure 'noise_pred' is in `return_kwargs`)"
        )
        return noise_pred

    # =========================== Group Bookkeeping ============================
    def _precompute_group_info(
        self,
        samples: List[BaseSample],
    ) -> DGPOGroupInfo:
        """Return ``local_group_indices`` + ``num_groups`` for a micro-batch.

        Derives a dense group id space from ``torch.unique`` on the
        micro-batch's ``unique_id`` values.

        Cross-rank consistency relies on the
        :class:`GroupDistributedSampler` contract: every rank yields the
        same prompt-index sequence per micro-batch, so ``local_uids`` is
        byte-identical on every rank and ``torch.unique(sorted=True)``
        produces the same dense ``0..L-1`` mapping.  That in turn makes
        the ``scatter_add`` + ``accelerator.reduce`` in
        :meth:`_compute_group_dgpo_loss` operate on a consistent group-id
        space without any cross-rank coordination on the id assignment.
        """
        device = self.accelerator.device
        local_uids = torch.as_tensor(
            [int(s.unique_id) for s in samples],
            dtype=torch.int64,
            device=device,
        )
        _, inverse = torch.unique(local_uids, return_inverse=True)
        return {
            "local_group_indices": inverse,
            "num_groups": int(inverse.max().item()) + 1,
        }

    # =========================== Noise Construction ============================
    def _make_shared_noise(
        self,
        x0: torch.Tensor,
        samples: List[BaseSample],
        inner_epoch: int,
    ) -> torch.Tensor:
        """Per-``unique_id`` shared noise on the same device as ``x0``.

        Uses one ``torch.Generator`` per unique group; because the generator
        is seeded deterministically from ``(seed, epoch, inner_epoch, uid)``
        and lives on ``x0.device``, we avoid any CPU→GPU copy and produce
        byte-identical noise across ranks for the same ``unique_id``.

        The noise is **timestep-invariant** — all training timesteps within
        an epoch share the same per-group noise, matching the reference
        DGPO implementation.
        """
        device, dtype = x0.device, x0.dtype
        per_sample_shape = x0.shape[1:]

        group_cache: Dict[int, torch.Tensor] = {}
        noises: List[torch.Tensor] = []
        for sample in samples:
            uid = int(sample.unique_id)
            noise = group_cache.get(uid)
            if noise is None:
                gen = create_generator(
                    self.training_args.seed,
                    self.epoch,
                    inner_epoch,
                    int(uid),
                    _SEED_TAG_SHARED_NOISE,
                    device=device,
                )
                noise = randn_tensor(
                    per_sample_shape,
                    generator=gen,
                    device=device,
                    dtype=dtype,
                )
                group_cache[uid] = noise
            noises.append(noise)
        return torch.stack(noises, dim=0)

    # =========================== Group DGPO Loss ============================
    def _compute_per_sample_preference(
        self,
        dsm_loss: torch.Tensor,
        ref_dgpo_v: torch.Tensor,
        target_v: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample contribution to a group's sigmoid argument.

        ``per_sample = advantage * dpo_beta * (dsm - ref_dsm) / group_size``

        We always detach ``dsm_loss`` internally because the sigmoid arm
        must be a constant w.r.t. the loss gradient — otherwise the
        reweighting is no longer a DGPO group preference but a
        second-order correction on ``dsm_loss`` itself.
        """
        batch_size = ref_dgpo_v.shape[0]
        with torch.no_grad():
            ref_dsm = (target_v - ref_dgpo_v).square().reshape(batch_size, -1).mean(dim=1)
        delta = dsm_loss.detach() - ref_dsm
        return advantages * self.dpo_beta * delta / self.training_args.group_size

    def _reduce_group_sums(
        self,
        local_sums: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-rank-sum partial per-group contributions.

        ``local_sums[g]`` is **this** rank's partial sum for group ``g``;
        after reduction every rank holds the full per-group sum, indexed
        by the same dense ``0..L-1`` id space that every rank derived
        locally via :meth:`_precompute_group_info` under the
        :class:`GroupDistributedSampler` contract.

        Wraps :meth:`accelerator.reduce`; a no-op in single-process /
        uninitialised-dist contexts.  Always ``.detach()`` the input so
        autograd does not try to flow through the collective.
        """
        if self.accelerator.num_processes > 1:
            return self.accelerator.reduce(local_sums.detach(), reduction="sum")  # type: ignore[return-value]
        return local_sums.detach()

    def _compute_group_dgpo_loss(
        self,
        ref_v: torch.Tensor,
        target_v: torch.Tensor,
        advantages: torch.Tensor,
        group_info: DGPOGroupInfo,
        dsm_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Group-level DGPO loss.

        Under the :class:`GroupDistributedSampler` contract every global
        micro-batch (``num_processes * per_device_batch_size`` samples, seen
        by all ranks in lockstep) holds an integer number of complete groups
        and every rank sees the same ``local_group_indices`` (via
        :meth:`_precompute_group_info`'s local ``torch.unique``).  A
        single complete group's ``group_size`` copies are split across
        ranks — one ``group_size / num_processes`` chunk per rank — so we
        ``scatter_add`` the local
        per-sample contributions, ``accelerator.reduce`` across ranks
        to recover the full-group sum, then apply ``sigmoid``.  This is
        the only group-level collective in the entire DGPO optimize
        loop.
        """
        device = dsm_loss.device
        num_groups = int(group_info["num_groups"])
        local_group_indices = group_info["local_group_indices"]

        per_sample = self._compute_per_sample_preference(
            dsm_loss=dsm_loss,
            ref_dgpo_v=ref_v,
            target_v=target_v,
            advantages=advantages,
        )

        local_sums = torch.zeros(num_groups, device=device, dtype=per_sample.dtype)
        local_sums.scatter_add_(0, local_group_indices, per_sample)
        global_sums = self._reduce_group_sums(local_sums)
        group_weights = torch.sigmoid(global_sums)[local_group_indices].detach()
        return (group_weights * advantages * dsm_loss).mean()

    # =========================== Per-Micro-batch Helpers ============================
    def _prep_training_batch(self, tb: Dict[str, Any]) -> _PreppedBatch:
        """Unpack one ``training_batches`` entry once.

        Avoids repeating the same six-field unpack and the
        ``adv_clip_range`` + ``clean_latents`` derivation on every
        timestep of :meth:`_optimize_step`.
        """
        batch = tb["batch"]
        all_latents: torch.Tensor = batch["all_latents"]
        clean_latents = all_latents[:, -1]
        adv_clip_range = self.training_args.adv_clip_range
        adv = torch.clamp(batch["advantage"], adv_clip_range[0], adv_clip_range[1])
        return {
            "batch": batch,
            "clean_latents": clean_latents,
            "adv": adv,
            "group_info": tb["group_info"],
            "timesteps": tb["timesteps"],
            "samples_slice": tb["samples_slice"],
            "inner_epoch": tb["inner_epoch"],
        }

    def _build_noised_inputs(
        self, p: _PreppedBatch, t_idx: int,
    ) -> _NoisedInputs:
        """Compute ``(t_flat, noised_latents, target_velocity)`` for a
        ``(prepped_batch, t_idx)`` pair.

        The per-group shared noise is **timestep-invariant** — all timesteps
        within an epoch receive the same noise for a given ``unique_id``,
        matching the reference DGPO implementation.
        """
        clean_latents = p["clean_latents"]
        t_flat = p["timesteps"][t_idx]
        sigma_bcast = to_broadcast_tensor(flow_match_sigma(t_flat), clean_latents)
        if self.use_shared_noise:
            noise = self._make_shared_noise(
                clean_latents, p["samples_slice"], p["inner_epoch"],
            )
        else:
            noise = randn_tensor(clean_latents.shape, device=clean_latents.device, dtype=clean_latents.dtype)
        noised = (1 - sigma_bcast) * clean_latents + sigma_bcast * noise
        target_v = noise - clean_latents
        return {"t_flat": t_flat, "noised": noised, "target_v": target_v}

    def _forward_velocities(
        self,
        batch: Dict[str, Any],
        t_flat: torch.Tensor,
        noised: torch.Tensor,
    ) -> _VelocityPredictions:
        """Run the per-optimizer-step velocity forwards.

        - ``model_v`` — **always** computed with gradient (this is the forward
          that backprop flows through).
        - ``old_v`` (``ema_ref``) — computed only when needed for DSM/KL
          clipping **or** as the DGPO reference under ``use_ema_ref=True``.
          Always detached.
        - ``ref_v`` (frozen pretrained) — computed only when needed for the
          KL penalty **or** as the DGPO reference when ``use_ema_ref=False``.
          Always detached.
        - ``ref_dgpo_v`` — alias of ``old_v`` if ``use_ema_ref=True``,
          otherwise ``ref_v``.

        This "compute on demand" pattern avoids unconditional extra forwards
        (two per step) when the corresponding feature is disabled.
        """
        need_old_v_for_clip = self._requires_ema_ref and (self.clip_dsm or self.clip_kl)
        need_old_v_for_dgpo_ref = self._requires_ema_ref and self.use_ema_ref
        compute_old_v = need_old_v_for_clip or need_old_v_for_dgpo_ref

        old_v: Optional[torch.Tensor] = None
        if compute_old_v:
            with torch.no_grad(), self._ema_ref_forward_context():
                old_v = self._compute_dgpo_output(
                    batch, t_flat, noised, guidance_scale=1.0
                ).detach()

        model_v = self._compute_dgpo_output(batch, t_flat, noised, guidance_scale=1.0)

        ref_v: Optional[torch.Tensor] = None
        if self.enable_kl_loss or (not self.use_ema_ref):
            ref_cfg = self.kl_cfg if self.kl_cfg > 1.0 else 1.0
            with torch.no_grad(), self.adapter.use_ref_parameters():
                ref_v = self._compute_dgpo_output(batch, t_flat, noised, guidance_scale=ref_cfg)

        if self.use_ema_ref:
            # Guaranteed non-None: use_ema_ref → compute_old_v → old_v assigned.
            assert old_v is not None
            ref_dgpo_v = old_v
        else:
            # Guaranteed non-None: not use_ema_ref → ref_v branch taken.
            assert ref_v is not None
            ref_dgpo_v = ref_v

        return {
            "model_v": model_v,
            "old_v": old_v,
            "ref_v": ref_v,
            "ref_dgpo_v": ref_dgpo_v,
        }

    def _compute_dsm_loss(
        self,
        target_v: torch.Tensor,
        pred_v: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample DSM MSE reduced over non-batch dimensions."""
        batch_size = target_v.shape[0]
        return (target_v - pred_v).square().reshape(batch_size, -1).mean(dim=1)

    def _maybe_clip_dsm(
        self,
        dsm_loss: torch.Tensor,
        old_v: Optional[torch.Tensor],
        target_v: torch.Tensor,
        adv: torch.Tensor,
        loss_info: Dict[str, List[torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Return ``(should_clip_mask, possibly_clipped_dsm_loss)``.

        PPO-ratio clip on the DSM loss against the old policy (``ema_ref``).
        ``should_clip`` is reused by the KL-clip path when ``clip_kl`` is
        set, hence its return.
        """
        if not (self.clip_dsm or self.clip_kl) or old_v is None:
            return None, dsm_loss

        batch_size = old_v.shape[0]
        clip_range = self.training_args.clip_range
        old_dsm = (target_v - old_v).square().reshape(batch_size, -1).mean(dim=1)
        ratio = torch.exp(-dsm_loss.detach() + old_dsm)
        should_clip = torch.where(
            adv > 0,
            ratio > 1.0 + clip_range[1],
            ratio < 1.0 + clip_range[0],
        )
        if self.clip_dsm:
            dsm_loss = torch.where(should_clip, dsm_loss.detach(), dsm_loss)
        loss_info["clip_ratio"].append(should_clip.float().mean().detach())
        return should_clip, dsm_loss

    def _apply_total_loss_and_backward(
        self,
        *,
        dgpo_loss: torch.Tensor,
        model_v: torch.Tensor,
        ref_v: Optional[torch.Tensor],
        should_clip: Optional[torch.Tensor],
        loss_info: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, List[torch.Tensor]]:
        """Assemble ``dgpo_loss + optional kl``, log components, backward,
        and optionally finalize the optimizer step.

        Returns the (possibly-reset) ``loss_info`` dict so the caller can
        keep accumulating into the same handle across micro-batches.
        """
        loss = dgpo_loss
        if self.enable_kl_loss:
            if ref_v is None:
                raise RuntimeError(
                    "DGPOTrainer._apply_total_loss_and_backward expected ref_v when KL is "
                    "enabled, but got None."
                )
            batch_size = model_v.shape[0]
            kl_div = (model_v - ref_v).square().reshape(batch_size, -1).mean(dim=1)
            if self.clip_kl and should_clip is not None:
                kl_div = torch.where(should_clip, kl_div.detach(), kl_div)
            kl_loss = self.kl_beta * kl_div.mean()
            loss = loss + kl_loss
            loss_info["kl_div"].append(kl_div.mean().detach())
            loss_info["kl_loss"].append(kl_loss.detach())

        loss_info["dgpo_loss"].append(dgpo_loss.detach())
        loss_info["loss"].append(loss.detach())

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            loss_info = self._finalize_step(loss_info)
        return loss_info

    # =========================== Advantage Processor Dispatch ============================
    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func: Optional[Union[Literal["sum", "gdpo"], Callable]] = None,
    ) -> torch.Tensor:
        """Compute advantages via the shared ``AdvantageProcessor``."""
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )

    # =========================== Training Batch Builder ============================
    def _build_training_batches(
        self,
        sample_slices: List[List[BaseSample]],
        shared_timesteps: torch.Tensor,
        inner_epoch: int,
    ) -> List[Dict[str, Any]]:
        """Materialise per-micro-batch inputs for the training loop.

        Noise is **not** pre-allocated: each timestep tensor is created
        inside the optimize loop to cap peak memory (``T`` × latent
        tensors).

        ``local_group_indices`` are derived per-micro-batch by
        :meth:`_precompute_group_info` via local ``torch.unique`` —
        cross-rank consistency is guaranteed by the
        :class:`GroupDistributedSampler` contract (identical prompt
        sequence on every rank).
        """
        training_batches: List[Dict[str, Any]] = []
        self.adapter.rollout()

        with torch.no_grad(), self.autocast():
            for samples_slice in tqdm(
                sample_slices,
                desc=f"Epoch {self.epoch} Pre-computing",
                position=0,
                disable=not self.show_progress_bar,
            ):
                batch = BaseSample.stack(samples_slice)
                all_latents: torch.Tensor = batch["all_latents"]  # type: ignore[assignment]
                clean_latents = all_latents[:, -1]
                batch_size = clean_latents.shape[0]

                group_info = self._precompute_group_info(samples_slice)
                timesteps = shared_timesteps.unsqueeze(1).expand(-1, batch_size)  # (T, B)

                training_batches.append(
                    {
                        "batch": batch,
                        "group_info": group_info,
                        "timesteps": timesteps,
                        "samples_slice": samples_slice,
                        "inner_epoch": inner_epoch,
                    }
                )

        return training_batches

    # =========================== Main Loop ============================
    def start(self):
        """Main training loop."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    "checkpoints",
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()

            with self.sampling_context():
                samples = self.sample()

            self.prepare_feedback(samples)
            self.optimize(samples)
            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    # =========================== Evaluation ============================
    def evaluate(self) -> None:
        """Evaluation loop (kept GRPO-equivalent: always use sampling EMA)."""
        if self.test_dataloader is None:
            return

        self.adapter.eval()
        self.eval_reward_buffer.clear()

        with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
            all_samples: List[BaseSample] = []

            for batch in tqdm(
                self.test_dataloader,
                desc="Evaluating",
                disable=not self.show_progress_bar,
            ):
                generator = create_generator_by_prompt(batch["prompt"], self.training_args.seed)
                inference_kwargs = {
                    "compute_log_prob": False,
                    "generator": generator,
                    "trajectory_indices": None,
                    **self.eval_args,
                }
                inference_kwargs.update(**batch)
                inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
                samples = self.adapter.inference(**inference_kwargs)
                all_samples.extend(samples)
                self.eval_reward_buffer.add_samples(samples)

            rewards = self.eval_reward_buffer.finalize(store_to_samples=True, split="pointwise")
            rewards = {
                key: torch.as_tensor(value).to(self.accelerator.device)
                for key, value in rewards.items()
            }
            gathered_rewards: Dict[str, np.ndarray] = {}
            for key, value in rewards.items():
                gathered: torch.Tensor = self.accelerator.gather(value)  # type: ignore[assignment]
                gathered_rewards[key] = gathered.cpu().numpy()

            if self.accelerator.is_main_process:
                log_data: Dict[str, Any] = {
                    f"eval/reward_{key}_mean": np.mean(value)
                    for key, value in gathered_rewards.items()
                }
                log_data.update(
                    {
                        f"eval/reward_{key}_std": np.std(value)
                        for key, value in gathered_rewards.items()
                    }
                )
                log_data["eval_samples"] = all_samples
                self.log_data(log_data, step=self.step)
            self.accelerator.wait_for_everyone()

    # =========================== Sampling (Stages 2-3) ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for DGPO (ODE sampling, final latents only)."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []
        data_iter = iter(self.dataloader)

        with torch.no_grad(), self.autocast():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f"Epoch {self.epoch} Sampling",
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    "compute_log_prob": False,
                    "trajectory_indices": [-1],
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)

        return samples

    # =========================== Reward / Advantage (Stages 4-5) ============================
    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalise rewards, compute advantages, and log advantage metrics."""
        rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
        self.compute_advantages(samples, rewards, store_to_samples=True)
        adv_metrics = self.advantage_processor.pop_advantage_metrics()
        if adv_metrics:
            self.log_data(adv_metrics, step=self.step)

    # =========================== Optimization (Stage 6) ============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimisation (Stage 6): group DGPO loss + optional PPO clip + KL.

        Rewards and advantages are finalised in :meth:`prepare_feedback`; this
        method performs policy gradients and the per-step ``ema_ref`` update.

        Under the :class:`GroupDistributedSampler` contract (enforced by
        ``hparams._resolve_sampler_type`` + ``_align_for_group_distributed``),
        every local micro-batch holds the same prompt sequence on every rank
        and a whole number of complete groups is present in every global
        micro-batch (``(num_processes * per_device_batch_size) % group_size == 0``).  This means we never need to
        gather full samples across ranks — the single ``accelerator.reduce``
        inside :meth:`_compute_group_dgpo_loss` is the only group-level
        collective required to recover the full-group sigmoid weights.
        """
        bsz = self.training_args.per_device_batch_size
        assert len(samples) % bsz == 0, (
            "DGPOTrainer.optimize expects len(samples) to be a multiple of "
            f"per_device_batch_size, got len(samples)={len(samples)} and bsz={bsz}."
        )

        for inner_epoch in range(self.training_args.num_inner_epochs):
            sample_slices = [
                samples[i : i + bsz] for i in range(0, len(samples), bsz)
            ]
            shared_timesteps = self._sample_shared_timesteps(inner_epoch)  # (T,)
            training_batches = self._build_training_batches(
                sample_slices,
                shared_timesteps,
                inner_epoch,
            )

            self.adapter.train()
            self._optimize_step(training_batches)

    def _optimize_step(
        self,
        training_batches: List[Dict[str, Any]],
    ) -> None:
        """Per-optimizer-step DGPO loss over every timestep of every micro-batch.

        ``(num_processes * per_device_batch_size) % group_size == 0`` holds
        by the sampler contract, so each global micro-batch is
        group-complete and a single forward per ``(micro_batch, t_idx)``
        pair is sufficient.  ``accelerator.reduce`` inside
        :meth:`_compute_group_dgpo_loss` aggregates partial per-rank
        per-group sums into the full-group sum before the sigmoid.
        """
        loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)

        with self.autocast():
            for tb in tqdm(
                training_batches,
                desc=f"Epoch {self.epoch} Training",
                position=0,
                disable=not self.show_progress_bar,
            ):
                p = self._prep_training_batch(tb)
                batch = p["batch"]
                adv = p["adv"]
                group_info = p["group_info"]

                for t_idx in tqdm(
                    range(self.num_train_timesteps),
                    desc=f"Epoch {self.epoch} Timestep",
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                ):
                    with self.accelerator.accumulate(*self.adapter.trainable_components):
                        noised = self._build_noised_inputs(p, t_idx)
                        vels = self._forward_velocities(
                            batch, noised["t_flat"], noised["noised"]
                        )
                        dsm_loss = self._compute_dsm_loss(noised["target_v"], vels["model_v"])
                        should_clip, dsm_loss = self._maybe_clip_dsm(
                            dsm_loss=dsm_loss,
                            old_v=vels["old_v"],
                            target_v=noised["target_v"],
                            adv=adv,
                            loss_info=loss_info,
                        )
                        ref_dgpo_v = vels["ref_dgpo_v"]
                        dgpo_loss = self._compute_group_dgpo_loss(
                            ref_v=ref_dgpo_v,
                            target_v=noised["target_v"],
                            advantages=adv,
                            group_info=group_info,
                            dsm_loss=dsm_loss,
                        )
                        loss_info["dsm_loss"].append(dsm_loss.mean().detach())
                        loss_info = self._apply_total_loss_and_backward(
                            dgpo_loss=dgpo_loss,
                            model_v=vels["model_v"],
                            ref_v=vels["ref_v"],
                            should_clip=should_clip,
                            loss_info=loss_info,
                        )

    # =========================== Optimizer Step Finalization ============================
    def _finalize_step(
        self,
        loss_info: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, List[torch.Tensor]]:
        """Optimizer step + ema_ref update + loss reduction/logging."""
        grad_norm = self.accelerator.clip_grad_norm_(
            self.adapter.get_trainable_parameters(),
            self.training_args.max_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        # ema_ref advances once per optimiser step (reference DGPO);
        # sampling EMA advances once per epoch in ``start()``.
        self._update_ema_ref(step=self.step)

        reduced = reduce_loss_info(self.accelerator, loss_info)
        reduced["grad_norm"] = grad_norm
        self.log_data(
            {f"train/{k}": v for k, v in reduced.items()},
            step=self.step,
        )
        self.step += 1
        return defaultdict(list)

