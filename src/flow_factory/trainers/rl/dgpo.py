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

# src/flow_factory/trainers/rl/dgpo.py
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
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import torch
import tqdm as tqdm_
from diffusers.utils.torch_utils import randn_tensor

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ...contracts.reward_overlap import GLOBAL_BATCH_REWARD_OPTIMIZATION_OVERLAP
from ...contracts.sampler import get_sampler_layout_contract
from ...hparams import DGPOTrainingArguments
from ...rewards import RewardTile, RewardTileGeometry
from ...samples import (
    LEGACY_SOURCE_ID,
    BaseSample,
    ComponentTimes,
    GroupKey,
    LatentState,
    NoisedState,
    StackedSampleBatch,
    group_identity_rows,
    sample_group_key,
)
from ...utils.base import create_generator, create_generator_by_prompt
from ...utils.logger_utils import setup_logger
from ..abc import BaseTrainer
from ..common.state_validation import require_latent_state, state_batch_size
from ..forward_process import forward_velocity_state

logger = setup_logger(__name__)

# Seed-namespace tags — appended to ``create_generator(...)`` integer keys so
# independent RNG streams (shared timesteps / shared per-group noise) never
# collide even if the other keys happen to coincide.  Independent (non-shared)
# training noise uses the global default RNG and is not seeded.
_SEED_TAG_SHARED_TIMESTEPS = 1
_SEED_TAG_SHARED_NOISE = 2
_SEED_TAG_SOURCE_NAMESPACE = 3


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

    batch: StackedSampleBatch
    clean_state: LatentState
    adv: torch.Tensor
    group_info: DGPOGroupInfo
    timesteps: torch.Tensor
    samples_slice: List[BaseSample]
    inner_epoch: int


class _NoisedInputs(NamedTuple):
    """Component times and the forward-noised state for one training timestep."""

    times: ComponentTimes
    noised: NoisedState


class _VelocityPredictions(TypedDict):
    """Output bundle of :meth:`DGPOTrainer._forward_velocities`.

    ``model_v`` carries autograd; ``old_v`` / ``ref_v`` / ``ref_dgpo_v`` are
    detached.  ``old_v`` and ``ref_v`` are computed on demand (``None`` when
    the corresponding feature — clipping / KL / ``use_ema_ref`` — is off);
    ``ref_dgpo_v`` is always set (aliased to ``old_v`` if
    ``use_ema_ref=True`` and to ``ref_v`` otherwise).
    """

    model_v: LatentState
    old_v: Optional[LatentState]
    ref_v: Optional[LatentState]
    ref_dgpo_v: LatentState


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

    # Decoupled paradigm: lossy rollout acceleration is permitted (constraints.md #7).
    paradigm = "decoupled"
    reward_optimization_overlap_contract = GLOBAL_BATCH_REWARD_OPTIMIZATION_OVERLAP

    @classmethod
    def reward_optimization_overlap_geometry(cls, config):
        """Consume rank-local shards whose groups close in the global batch."""
        return RewardTileGeometry(
            group_layout="cross_rank_sharded",
            optimizer_terms_per_batch=config.training_args.get_num_train_timesteps(config),
        )

    runtime_child_names = ("ema_ref",)

    def _algorithm_runtime_child_names(self) -> Tuple[str, ...]:
        """Declare the old-policy snapshot only when DGPO consumes it."""
        training_args: DGPOTrainingArguments = self.training_args  # type: ignore[assignment]
        requires_ema_ref = (
            training_args.clip_dsm or training_args.clip_kl or training_args.use_ema_ref
        )
        return type(self).runtime_child_names if requires_ema_ref else ()

    def _initialize_snapshots(self) -> None:
        """Initialize the optional old-policy snapshot before exact state resume."""
        if not self._algorithm_runtime_child_names():
            return
        training_args: DGPOTrainingArguments = self.training_args  # type: ignore[assignment]
        ema_ref_device = (
            self.accelerator.device
            if training_args.ema_ref_device == "cuda"
            else torch.device("cpu")
        )
        self.adapter.add_named_parameters(
            "ema_ref",
            device=ema_ref_device,
            overwrite=True,
        )
        self._register_named_parameter_runtime_child("ema_ref")
        logger.info(
            f"Initialized old-policy EMA ref on {ema_ref_device} "
            f"(max_decay={training_args.ema_ref_max_decay}, "
            f"ramp_rate={training_args.ema_ref_ramp_rate})."
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ta: DGPOTrainingArguments = self.training_args  # type: ignore[assignment]
        self.training_args = ta

        # DGPO is only valid under GroupDistributedSampler — `hparams.Arguments.
        # _resolve_sampler_type` hard-forces this.  This assert is a
        # belt-and-suspenders guard against future code paths bypassing hparams.
        sampler_layout = get_sampler_layout_contract(self.config.data_args.sampler_type)
        if sampler_layout.group_placement != "global_batch":
            raise ValueError(
                "DGPOTrainer requires a global-batch group layout, got "
                f"sampler_type={self.config.data_args.sampler_type!r}, "
                f"group_placement={sampler_layout.group_placement!r}"
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

    # =========================== Group Bookkeeping ============================
    def _precompute_group_info(
        self,
        samples: List[BaseSample],
    ) -> DGPOGroupInfo:
        """Return ``local_group_indices`` + ``num_groups`` for a micro-batch.

        Derives one dense id space from the gathered global microbatch. This
        supports both the legacy equal-share layout and packed groups smaller
        than the world size, where a rank may not contain every group.
        """
        device = self.accelerator.device
        local_group_identities = torch.as_tensor(
            group_identity_rows(samples),
            dtype=torch.int64,
            device=device,
        )
        cached_group_infos = getattr(
            self,
            "_dgpo_reward_overlap_group_infos",
            None,
        )
        if cached_group_infos is not None:
            if not cached_group_infos:
                raise RuntimeError("DGPO reward overlap exhausted cached cross-rank group metadata")
            cached = cached_group_infos.pop(0)
            cached.validate_samples(samples, context="the DGPO optimizer microbatch")
            return {
                "local_group_indices": cached.local_group_indices,
                "num_groups": cached.num_groups,
            }
        if self.training_args.group_size % self.accelerator.num_processes == 0:
            # GroupDistributedSampler preserves its historical equal-share
            # layout for this geometry: every rank observes the same ordered
            # groups and K / W members of each. Keep that common path free of
            # an otherwise redundant UID gather.
            sorted_identities, inverse = torch.unique(
                local_group_identities,
                dim=0,
                sorted=True,
                return_inverse=True,
            )
            return {
                "local_group_indices": inverse,
                "num_groups": int(sorted_identities.shape[0]),
            }
        global_identities = self.accelerator.gather(local_group_identities)
        sorted_identities, counts = torch.unique(
            global_identities,
            dim=0,
            sorted=True,
            return_counts=True,
        )
        expected_counts = torch.full_like(counts, self.training_args.group_size)
        if not torch.equal(counts, expected_counts):
            raise ValueError(
                "DGPO expected every global microbatch group to contain exactly "
                f"group_size={self.training_args.group_size} members, received "
                f"identities={sorted_identities.tolist()} with counts={counts.tolist()}"
            )
        local_matches = torch.all(
            local_group_identities[:, None, :] == sorted_identities[None, :, :],
            dim=-1,
        )
        if not torch.all(local_matches.sum(dim=1) == 1):
            raise RuntimeError("DGPO could not map local samples into the global group identity")
        local_inverse = local_matches.to(torch.int64).argmax(dim=1)
        return {
            "local_group_indices": local_inverse,
            "num_groups": int(sorted_identities.shape[0]),
        }

    # =========================== Noise Construction ============================
    def _draw_group_component_noise(
        self,
        *,
        unique_id: int,
        source_id: int = LEGACY_SOURCE_ID,
        component_name: str,
        component_index: int,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        inner_epoch: int,
    ) -> torch.Tensor:
        """Draw one group's noise for one component from an explicit seed tuple.

        The primary component keeps the legacy namespace when ``source_id`` is
        unavailable; source-tagged samples append a source namespace so equal
        prompt hashes from different datasets cannot share noise. Every further
        component extends the key by its position in
        ``trajectory_component_order``, which never depends on mapping iteration
        order.

        Args:
            unique_id: Group identifier shared by every sample of the group.
            source_id: Dataset-source namespace for the group identity.
            component_name: Component the noise belongs to. The default namespace
                keys off ``component_index`` because the index, not the name, is
                the authoritative order; the name stays part of the keyword
                contract so overrides can extend the draw per component and so
                the caller's mismatch errors can name the offending component.
            component_index: Component position in ``trajectory_component_order``.
            shape: Per-sample noise shape for this component.
            dtype: Per-sample noise dtype for this component.
            device: Device the generator and the noise live on.
            inner_epoch: Inner epoch index, part of the seed namespace.

        Returns:
            One group's noise for one component, without a batch dimension.
        """
        namespace = () if component_index == 0 else (component_index,)
        source_namespace = (
            () if source_id == LEGACY_SOURCE_ID else (_SEED_TAG_SOURCE_NAMESPACE, source_id)
        )
        generator = create_generator(
            self.training_args.seed,
            self.epoch,
            inner_epoch,
            int(unique_id),
            _SEED_TAG_SHARED_NOISE,
            *source_namespace,
            *namespace,
            device=device,
        )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def _shared_group_noise(
        self,
        clean_state: LatentState,
        samples: List[BaseSample],
        inner_epoch: int,
        *,
        timestep_index: Optional[int] = None,
    ) -> LatentState:
        """Build source-aware per-group noise for every trajectory component.

        Every sample of a group receives the same component noise, drawn once per
        group in ``trajectory_component_order``. Because each draw carries its own
        deterministically seeded generator, the result is byte-identical across
        ranks and independent of how many groups a rank happens to hold.

        The noise is **timestep-invariant** — all training timesteps within an
        inner epoch share one group noise, matching the reference DGPO
        implementation.

        Args:
            clean_state: Terminal clean state supplying per-component geometry.
            samples: Micro-batch samples, aligned with the state batch dimension.
            inner_epoch: Inner epoch index, part of the seed namespace.
            timestep_index: Training timestep the caller is building, reported by
                validation errors. The noise itself does not depend on it.

        Returns:
            Batched noise state in ``trajectory_component_order``.
        """
        component_names = self.adapter.trajectory_component_order
        context = (
            f"inner_epoch={inner_epoch}, timestep_index={timestep_index}, "
            f"component order {component_names}"
        )
        batch_size = state_batch_size(self, clean_state, "terminal clean state")
        if len(samples) != batch_size:
            raise ValueError(
                f"expected {type(self).__name__} shared noise ({context}) to receive one "
                f"sample per terminal state row, i.e. {batch_size} samples, received "
                f"{len(samples)}"
            )
        references: Dict[str, torch.Tensor] = {}
        for name in component_names:
            component = clean_state.components[name]
            if component.ndim < 2 or component.shape[0] != batch_size:
                raise ValueError(
                    f"expected {type(self).__name__} terminal clean component {name!r} "
                    f"({context}) to be batched with shape (B, ...) and batch size "
                    f"{batch_size}, received {tuple(component.shape)}"
                )
            references[name] = component

        group_cache: Dict[GroupKey, Dict[str, torch.Tensor]] = {}
        rows: Dict[str, List[torch.Tensor]] = {name: [] for name in component_names}
        for sample in samples:
            group_key = sample_group_key(sample)
            group_noise = group_cache.get(group_key)
            if group_noise is None:
                group_noise = {
                    name: self._draw_group_component_noise(
                        unique_id=group_key.unique_id,
                        source_id=group_key.source_id,
                        component_name=name,
                        component_index=component_index,
                        shape=references[name].shape[1:],
                        dtype=references[name].dtype,
                        device=references[name].device,
                        inner_epoch=inner_epoch,
                    )
                    for component_index, name in enumerate(component_names)
                }
                group_cache[group_key] = group_noise
            for name in component_names:
                noise = group_noise[name]
                reference = references[name]
                if (
                    noise.shape != reference.shape[1:]
                    or noise.dtype != reference.dtype
                    or noise.device != reference.device
                ):
                    raise ValueError(
                        f"expected {type(self).__name__} shared noise for unique_id="
                        f"{group_key.unique_id}, source_id={group_key.source_id} ({context}) "
                        f"component {name!r} to match the clean "
                        f"per-sample shape/dtype/device ({tuple(reference.shape[1:])}, "
                        f"{reference.dtype}, {reference.device}), received "
                        f"({tuple(noise.shape)}, {noise.dtype}, {noise.device})"
                    )
                rows[name].append(noise)
        return require_latent_state(
            self,
            LatentState({name: torch.stack(rows[name], dim=0) for name in component_names}),
            f"shared noise ({context})",
        )

    # =========================== Group DGPO Loss ============================
    def _compute_per_sample_preference(
        self,
        dsm_loss: torch.Tensor,
        ref_dgpo_v: LatentState,
        target_v: LatentState,
        advantages: torch.Tensor,
        noised: NoisedState,
    ) -> torch.Tensor:
        """Per-sample contribution to a group's sigmoid argument.

        ``per_sample = advantage * dpo_beta * (dsm - ref_dsm) / group_size``

        We always detach ``dsm_loss`` internally because the sigmoid arm
        must be a constant w.r.t. the loss gradient — otherwise the
        reweighting is no longer a DGPO group preference but a
        second-order correction on ``dsm_loss`` itself.
        """
        with torch.no_grad():
            ref_dsm = self._compute_dsm_loss(target_v, ref_dgpo_v, noised)
        delta = dsm_loss.detach() - ref_dsm
        return advantages * self.dpo_beta * delta / self.training_args.group_size

    def _reduce_group_sums(
        self,
        local_sums: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-rank-sum partial per-group contributions.

        ``local_sums[g]`` is **this** rank's partial sum for group ``g``;
        after reduction every rank holds the full per-group sum, indexed
        by the same dense ``0..L-1`` id space established by
        :meth:`_precompute_group_info` under the
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
        ref_v: LatentState,
        target_v: LatentState,
        advantages: torch.Tensor,
        group_info: DGPOGroupInfo,
        dsm_loss: torch.Tensor,
        noised: NoisedState,
    ) -> torch.Tensor:
        """Group-level DGPO loss.

        Under the :class:`GroupDistributedSampler` contract every global
        micro-batch (``num_processes * per_device_batch_size`` samples, seen
        by all ranks in lockstep) holds an integer number of complete groups
        and every rank maps its local rows into one shared dense group-id
        space. Equal-share geometry derives that space locally; packed geometry
        gathers only integer identities and may leave a rank with no member of
        some groups. We ``scatter_add`` local per-sample contributions,
        ``accelerator.reduce`` across ranks to recover full-group sums, then
        apply ``sigmoid``.
        """
        device = dsm_loss.device
        num_groups = int(group_info["num_groups"])
        local_group_indices = group_info["local_group_indices"]

        per_sample = self._compute_per_sample_preference(
            dsm_loss=dsm_loss,
            ref_dgpo_v=ref_v,
            target_v=target_v,
            advantages=advantages,
            noised=noised,
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
        ``adv_clip_range`` + terminal-state derivation on every
        timestep of :meth:`_optimize_step`.
        """
        batch = tb["batch"]
        clean_state = require_latent_state(
            self, self.adapter.get_terminal_state(batch), "terminal clean state"
        )
        adv_clip_range = self.training_args.adv_clip_range
        adv = torch.clamp(batch["advantage"], adv_clip_range[0], adv_clip_range[1])
        return {
            "batch": batch,
            "clean_state": clean_state,
            "adv": adv,
            "group_info": tb["group_info"],
            "timesteps": tb["timesteps"],
            "samples_slice": tb["samples_slice"],
            "inner_epoch": tb["inner_epoch"],
        }

    def _build_noised_inputs(self, p: _PreppedBatch, t_idx: int) -> _NoisedInputs:
        """Compute the component times and forward-noised state for a
        ``(prepped_batch, t_idx)`` pair.

        Shared noise is predetermined per canonical group identity and only applied here, so
        the application consumes no randomness; independent noise delegates the
        draw to the adapter's ordered noising hook.

        The per-group shared noise is **timestep-invariant** — all timesteps
        within an epoch receive the same noise for a given group identity,
        matching the reference DGPO implementation.
        """
        clean_state = p["clean_state"]
        times = self.adapter.build_training_component_times(p["timesteps"][t_idx], batch=p["batch"])
        if self.use_shared_noise:
            noise = self._shared_group_noise(
                clean_state,
                p["samples_slice"],
                p["inner_epoch"],
                timestep_index=t_idx,
            )
            noised = self.adapter.apply_forward_process_noise(clean_state, times, noise)
        else:
            noised = self.adapter.add_forward_process_noise(clean_state, times)
        return _NoisedInputs(times=times, noised=noised)

    def _forward_velocities(
        self,
        batch: StackedSampleBatch,
        times: ComponentTimes,
        noised: NoisedState,
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

        old_v: Optional[LatentState] = None
        if compute_old_v:
            with torch.no_grad(), self._ema_ref_forward_context(), self.autocast():
                old_velocity = forward_velocity_state(
                    self,
                    batch,
                    noised.state,
                    times,
                    source="old policy",
                    guidance_scale=1.0,
                )
            old_v = LatentState(
                {name: value.detach() for name, value in old_velocity.components.items()}
            )

        with self.autocast():
            model_v = forward_velocity_state(
                self, batch, noised.state, times, source="policy", guidance_scale=1.0
            )

        ref_v: Optional[LatentState] = None
        if self.enable_kl_loss or (not self.use_ema_ref):
            ref_cfg = self.kl_cfg if self.kl_cfg > 1.0 else 1.0
            with torch.no_grad(), self.adapter.use_ref_parameters(), self.autocast():
                ref_v = forward_velocity_state(
                    self,
                    batch,
                    noised.state,
                    times,
                    source="reference",
                    guidance_scale=ref_cfg,
                )

        return {
            "model_v": model_v,
            "old_v": old_v,
            "ref_v": ref_v,
            "ref_dgpo_v": self._select_dgpo_reference(old_v, ref_v),
        }

    def _select_dgpo_reference(
        self,
        old_v: Optional[LatentState],
        ref_v: Optional[LatentState],
    ) -> LatentState:
        """Pick the velocity the group preference measures the policy against.

        Both branches are reachable only when the configuration flags agree with
        the forwards :meth:`_forward_velocities` actually ran, so a ``None`` here
        means the gating flags drifted apart — report them instead of failing on
        a bare truth check.

        Args:
            old_v: Old-policy (``ema_ref``) velocity, or ``None`` if not computed.
            ref_v: Frozen-reference velocity, or ``None`` if not computed.

        Returns:
            The velocity state to use as the DGPO reference.
        """
        if self.use_ema_ref:
            if old_v is None:
                raise ValueError(
                    f"expected {type(self).__name__} to hold an old policy velocity as the "
                    "DGPO reference when use_ema_ref=True, received None; the old-policy "
                    f"forward is gated by _requires_ema_ref={self._requires_ema_ref} "
                    f"(clip_dsm={self.clip_dsm}, clip_kl={self.clip_kl}, "
                    f"use_ema_ref={self.use_ema_ref}), so no 'ema_ref' forward ran"
                )
            return old_v
        if ref_v is None:
            raise ValueError(
                f"expected {type(self).__name__} to hold a reference velocity as the DGPO "
                "reference when use_ema_ref=False, received None; the reference forward is "
                f"gated by enable_kl_loss={self.enable_kl_loss} (kl_beta={self.kl_beta}) or "
                f"use_ema_ref={self.use_ema_ref}"
            )
        return ref_v

    def _compute_dsm_loss(
        self,
        target_v: LatentState,
        pred_v: LatentState,
        noised: NoisedState,
    ) -> torch.Tensor:
        """Per-sample DSM error, reduced with the noised state's active elements.

        Args:
            target_v: Flow-matching target velocity per component.
            pred_v: Predicted velocity per component.
            noised: Forward-noised state supplying per-sample reduction context.

        Returns:
            Per-sample DSM loss of shape ``(B,)``.
        """
        errors = {
            name: (target_v.components[name] - pred_v.components[name]).square()
            for name in self.adapter.trajectory_component_order
        }
        return self.adapter.reduce_latent_values(errors, state=noised.state)

    def _maybe_clip_dsm(
        self,
        dsm_loss: torch.Tensor,
        old_v: Optional[LatentState],
        target_v: LatentState,
        adv: torch.Tensor,
        loss_info: Dict[str, List[torch.Tensor]],
        noised: NoisedState,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Return ``(should_clip_mask, possibly_clipped_dsm_loss)``.

        PPO-ratio clip on the DSM loss against the old policy (``ema_ref``).
        ``should_clip`` is reused by the KL-clip path when ``clip_kl`` is
        set, hence its return.
        """
        if not (self.clip_dsm or self.clip_kl) or old_v is None:
            return None, dsm_loss

        clip_range = self.training_args.clip_range
        old_dsm = self._compute_dsm_loss(target_v, old_v, noised)
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
        model_v: LatentState,
        ref_v: Optional[LatentState],
        should_clip: Optional[torch.Tensor],
        loss_info: Dict[str, List[torch.Tensor]],
        noised: NoisedState,
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
            with self.autocast():
                kl_div = self._velocity_kl(model_v, ref_v, noised)
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
            loss_info = self._apply_optimizer_step(loss_info)
        return loss_info

    # =========================== Advantage Processor Dispatch ============================
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
        device = self.accelerator.device
        self.adapter.rollout()

        with torch.no_grad(), self.autocast():
            for samples_slice in tqdm(
                sample_slices,
                desc=f"Epoch {self.epoch} Pre-computing",
                position=0,
                disable=not self.show_progress_bar,
            ):
                # Blocking H2D reload (no-op when GPU-resident). This two-phase
                # builder has no per-batch compute to overlap, so prefetch is N/A
                # here; the prefetch dividend is realised inside the single-pass
                # trainers' optimize loops, not this builder. DGPO samples are
                # final-latent-only, so the H2D is tiny regardless.
                batch: StackedSampleBatch = BaseSample.stack([s.to(device) for s in samples_slice])
                clean_state = self.adapter.get_terminal_state(batch)
                batch_size = state_batch_size(self, clean_state, "terminal clean state")

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

    def _optimize_reward_overlap_tile(
        self,
        tile: RewardTile,
        samples: List[BaseSample],
        context: Any,
    ) -> None:
        """Reuse acquisition-level group mappings instead of gathering per batch."""
        group_infos = list(self._reward_overlap_group_infos_for_tile(tile.tile_id))
        self._dgpo_reward_overlap_group_infos = group_infos
        try:
            super()._optimize_reward_overlap_tile(tile, samples, context)
            if group_infos:
                raise RuntimeError(
                    "DGPO reward overlap did not consume all cached cross-rank group "
                    f"metadata for tile_id={tile.tile_id}: remaining={len(group_infos)}"
                )
        finally:
            del self._dgpo_reward_overlap_group_infos

    # =========================== Main Loop ============================
    # =========================== Sampling (Stages 2-3) ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for DGPO (final latents only)."""
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=False,
            trajectory_indices=[-1],
        )

    # =========================== Reward / Advantage (Stages 4-5) ============================
    # =========================== Optimization (Stage 6) ============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimisation (Stage 6): group DGPO loss + optional PPO clip + KL.

        Rewards and advantages are finalised in :meth:`prepare_feedback`; this
        method performs policy gradients and the per-step ``ema_ref`` update.

        Under the :class:`GroupDistributedSampler` contract (enforced by
        ``hparams._resolve_sampler_type`` + ``_align_for_group_distributed``),
        every global microbatch contains a whole number of complete groups
        (``(num_processes * per_device_batch_size) % group_size == 0``). The
        equal-share layout gives every rank the same ordered groups; the packed
        layout gathers only exact integer group identities. Full samples remain
        local, and the ``accelerator.reduce`` inside
        :meth:`_compute_group_dgpo_loss` recovers the full-group sigmoid weights.
        """
        bsz = self.training_args.per_device_batch_size
        assert len(samples) % bsz == 0, (
            "DGPOTrainer.optimize expects len(samples) to be a multiple of "
            f"per_device_batch_size, got len(samples)={len(samples)} and bsz={bsz}."
        )

        for inner_epoch in range(self.training_args.num_inner_epochs):
            sample_slices = [samples[i : i + bsz] for i in range(0, len(samples), bsz)]
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

        # No loop-level autocast: forwards are wrapped in `_forward_velocities` (#20a).
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
                with self.accumulate_gradients():
                    times, noised = self._build_noised_inputs(p, t_idx)
                    vels = self._forward_velocities(batch, times, noised)
                    target_v = noised.target_velocity
                    dsm_loss = self._compute_dsm_loss(target_v, vels["model_v"], noised)
                    should_clip, dsm_loss = self._maybe_clip_dsm(
                        dsm_loss=dsm_loss,
                        old_v=vels["old_v"],
                        target_v=target_v,
                        adv=adv,
                        loss_info=loss_info,
                        noised=noised,
                    )
                    ref_dgpo_v = vels["ref_dgpo_v"]
                    dgpo_loss = self._compute_group_dgpo_loss(
                        ref_v=ref_dgpo_v,
                        target_v=target_v,
                        advantages=adv,
                        group_info=group_info,
                        dsm_loss=dsm_loss,
                        noised=noised,
                    )
                    loss_info["dsm_loss"].append(dsm_loss.mean().detach())
                    loss_info = self._apply_total_loss_and_backward(
                        dgpo_loss=dgpo_loss,
                        model_v=vels["model_v"],
                        ref_v=vels["ref_v"],
                        should_clip=should_clip,
                        loss_info=loss_info,
                        noised=noised,
                    )

    def _after_gradient_step(self) -> None:
        """Advance the fast reference EMA once per optimizer step.

        Matches the reference DGPO cadence; the slow sampling EMA advances once
        per epoch through the shared training loop instead.
        """
        self._update_ema_ref(step=self.step)
