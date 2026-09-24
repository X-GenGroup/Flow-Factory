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

# src/flow_factory/rewards/reward_processor.py
"""
Unified Reward Processor for handling multiple reward models.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm

from ..hparams import RewardArguments
from ..samples import BaseSample, GroupKey, sample_group_key
from ..utils.audio import standardize_audio_batch
from ..utils.base import filter_kwargs, move_tensors_to_device
from ..utils.dist import gather_samples
from ..utils.image import standardize_image_batch
from ..utils.video import standardize_video_batch
from .abc import (
    BaseRewardModel,
    GroupwiseRewardModel,
    PointwiseRewardModel,
    RewardModelOutput,
)


# ============================ Reward Processor ============================
class RewardProcessor:
    """
    Unified reward processor bound to specific reward models.

    Handles both PointwiseRewardModel and GroupwiseRewardModel seamlessly.
    """

    MEDIA_FIELDS = {
        "image",
        "video",
        "audio",
        "condition_images",
        "condition_videos",
    }  # Fields that may contain media data, requiring format conversion

    def __init__(
        self,
        accelerator: Accelerator,
        reward_models: Dict[str, BaseRewardModel],
        reward_configs: Optional[Dict[str, RewardArguments]] = None,
        tokenizer: Optional[Any] = None,
        group_on_same_rank: bool = False,
        verbose: bool = True,
    ):
        self.accelerator = accelerator
        self.reward_models = reward_models
        self.reward_configs = reward_configs or {}
        self.tokenizer = tokenizer
        self.group_on_same_rank = group_on_same_rank
        self.verbose = verbose

        # Pre-categorize models by type
        self._pointwise_models: Dict[str, PointwiseRewardModel] = {
            k: v for k, v in reward_models.items() if isinstance(v, PointwiseRewardModel)
        }
        self._groupwise_models: Dict[str, GroupwiseRewardModel] = {
            k: v for k, v in reward_models.items() if isinstance(v, GroupwiseRewardModel)
        }

    @property
    def show_progress_bar(self) -> bool:
        """Whether to show tqdm progress bars."""
        return self.verbose and self.accelerator.is_local_main_process

    # ============================ Source-Aware Reward Gating ============================
    def _reward_applies(self, name: str, sample: BaseSample) -> bool:
        """Authoritative per-sample applicability for reward ``name``.

        Single source of truth for the gate consumed by both compute
        (skip the model call when no sample matches) AND aggregation
        (`AdvantageProcessor` reads `sample.applicable_rewards`, never
        `np.isnan`, so an in-model NaN bug is NOT silently masked).

        Routing rules:

        - ``RewardArguments.applicable_datasets`` is normally a concrete
          ``List[str]`` here (resolved by
          ``Arguments._resolve_reward_dataset_routing``).  As a defensive
          fallback for callers that construct a ``RewardProcessor`` outside
          the ``Arguments`` flow, ``None`` is also treated as "applies to
          every sample".
        - When the sample has no source (``source is None`` AND
          ``source_id is None``), absence is treated as "applies" -
          preserves byte-identical behavior for legacy single-source mode.
        - Otherwise we prefer the small-int form
          (``sample.source_id in cfg._datasets_resolved``) when available,
          falling back to the string form
          (``sample.source in cfg.applicable_datasets``) when the id
          resolver hasn't run (direct test instantiation of
          ``RewardProcessor``).
        """
        cfg = self.reward_configs.get(name)
        if cfg is None or cfg.applicable_datasets is None:
            return True
        # Legacy single-source: no source bookkeeping at all -> every reward
        # applies (this is what the legacy GRPO behavior expected).
        if sample.source_id is None and sample.source is None:
            return True
        # Hot path: int-set membership when both sides have ids.
        if sample.source_id is not None and cfg._datasets_resolved is not None:
            return sample.source_id in cfg._datasets_resolved
        # Fallback: string-list membership.
        return sample.source in cfg.applicable_datasets

    @staticmethod
    def _scatter_with_nan_padding(
        sub_scores: torch.Tensor,
        mask: List[bool],
        reward_name: str,
    ) -> torch.Tensor:
        """Build a full-length NaN-padded result from a sub-batch model output.

        ``sub_scores`` has shape ``(sum(mask),)`` -- the model output for
        only the applicable positions.  Returns a tensor of shape
        ``(len(mask),)`` with NaN at non-applicable positions and
        ``sub_scores`` scattered into the applicable ones.

        Asserts that the model's output is finite at every applicable
        position.  A NaN there means the reward model itself returned
        NaN (overflow / bug) -- surface immediately rather than silently
        mask, per plan §6.3.
        """
        sub_scores = sub_scores.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        if sub_scores.shape[0] != sum(mask):
            raise RuntimeError(
                f"Reward '{reward_name}' returned {sub_scores.shape[0]} scores "
                f"but {sum(mask)} samples were applicable. Shape mismatch."
            )
        if not torch.isfinite(sub_scores).all():
            bad = (~torch.isfinite(sub_scores)).nonzero(as_tuple=True)[0].tolist()
            raise RuntimeError(
                f"Reward '{reward_name}' produced NaN/Inf at applicable "
                f"sub-batch positions {bad}. This is a model bug, not a "
                "routing miss — the gate has already filtered non-applicable "
                "samples out of the input."
            )
        full = torch.full((len(mask),), float("nan"), dtype=torch.float32)
        if any(mask):
            idx = [i for i, m in enumerate(mask) if m]
            full[idx] = sub_scores
        return full

    def _mark_applicable(
        self,
        samples: List[BaseSample],
        mask: List[bool],
        reward_name: str,
    ) -> None:
        """Mark each applicable sample's ``applicable_rewards`` set with ``reward_name``.

        Aggregation in :class:`AdvantageProcessor` reads this set as the
        authoritative truth (no NaN sniffing), so we MUST update it for
        every applicable position regardless of whether the model was
        actually invoked (e.g., even when ``not any(mask)`` we still
        write NaN; in that case mask is all False and the loop is a no-op).
        """
        if not any(mask):
            return
        for s, m in zip(samples, mask):
            if m:
                s.applicable_rewards.add(reward_name)

    def _is_async_reward(self, name: str) -> bool:
        """Check if a named reward model is configured for async computation."""
        config = self.reward_configs.get(name)
        return getattr(config, "async_reward", False) if config else False

    def _resolve_num_workers(self, name: str) -> int:
        """Resolve the number of concurrent workers for an async reward model."""
        config = self.reward_configs.get(name)
        return max(1, getattr(config, "num_workers", 1)) if config else 1

    def _resolve_batch_size(self, name: str, model: BaseRewardModel) -> int:
        """
        Resolve runtime batch size for a pointwise reward model.

        Priority:
            1) Explicit config in `self.reward_configs` for this reward name.
            2) Fallback to shared model config (`model.config.batch_size`).
        """
        batch_size = None
        if name in self.reward_configs:
            batch_size = getattr(self.reward_configs[name], "batch_size", None)
        if batch_size is None:
            batch_size = getattr(model.config, "batch_size", None)

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"Invalid batch_size for reward '{name}': {batch_size}. "
                "batch_size must be a positive integer."
            )

        return batch_size

    # ============================ Media Format Conversion ============================
    def _convert_media_format(
        self, batch_input: Dict[str, Any], model: BaseRewardModel
    ) -> Dict[str, Any]:
        """Convert tensor media fields to PIL format (unless model opts out)."""
        if getattr(model, "use_tensor_inputs", False):
            output_type = "pt"
        else:
            output_type = "pil"

        result = {}
        for k, v in batch_input.items():
            if k not in self.MEDIA_FIELDS or v is None:
                result[k] = v
                continue
            if k == "image":
                result[k] = standardize_image_batch(v, output_type=output_type)
            elif k == "video":
                result[k] = standardize_video_batch(v, output_type=output_type)
            elif k == "audio":
                # Audio has no PIL representation; map 'pil' -> 'np'
                audio_output = "pt" if output_type == "pt" else "np"
                result[k] = standardize_audio_batch(v, output_type=audio_output)
            elif k == "condition_images":
                result[k] = [standardize_image_batch(imgs, output_type=output_type) for imgs in v]
            elif k == "condition_videos":
                result[k] = [
                    standardize_video_batch(videos, output_type=output_type) for videos in v
                ]

        return result

    # ============================ Single-batch / Single-group Helpers ============================
    def _gated_compute(
        self, name: str, model: BaseRewardModel, samples: List[BaseSample]
    ) -> torch.Tensor:
        """Shared gate + compute + NaN-pad for both pointwise and groupwise paths.

        Returns a ``(len(samples),)`` tensor with NaN at non-applicable
        positions and finite model scores at applicable positions.
        Updates ``sample.applicable_rewards`` for every applicable sample.
        """
        mask = [self._reward_applies(name, s) for s in samples]
        self._mark_applicable(samples, mask, name)

        if not any(mask):
            return torch.full((len(samples),), float("nan"), dtype=torch.float32)

        sub_samples = [s for s, m in zip(samples, mask) if m]
        filtered_fields = filter_kwargs(model.__call__, **sub_samples[0])
        sub_input: Dict[str, List[Any]] = {
            k: [getattr(s, k) for s in sub_samples]
            for k in filtered_fields
            if all(getattr(s, k) is not None for s in sub_samples)
        }
        sub_input = self._convert_media_format(sub_input, model)
        sub_input = move_tensors_to_device(sub_input, model.device)
        output = model(**sub_input)
        sub_scores = torch.as_tensor(
            output.rewards if hasattr(output, "rewards") else output,
            dtype=torch.float32,
        )
        return self._scatter_with_nan_padding(sub_scores, mask, reward_name=name)

    def _compute_pointwise_batch(
        self, name: str, model: PointwiseRewardModel, batch_samples: List[BaseSample]
    ) -> torch.Tensor:
        """Compute pointwise rewards for a single batch with source-aware gating."""
        return self._gated_compute(name, model, batch_samples)

    def _compute_groupwise_group(
        self, name: str, model: GroupwiseRewardModel, group_samples: List[BaseSample]
    ) -> torch.Tensor:
        """Compute groupwise rewards for one complete group with source-aware gating."""
        return self._gated_compute(name, model, group_samples)

    # ============================ Public API ============================
    def compute_rewards(
        self,
        samples: List[BaseSample],
        store_to_samples: bool = True,
        epoch: Optional[int] = None,
        split: Literal["pointwise", "groupwise", "all"] = "all",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rewards using bound reward models.

        Args:
            samples: Local samples on this rank
            store_to_samples: Whether to store rewards in sample.extra_kwargs
            epoch: Current epoch for progress bar display
            split: Which reward models to use
                - 'pointwise': Only pointwise models (no cross-rank communication)
                - 'groupwise': Only groupwise models (requires gather/scatter)
                - 'all': Both pointwise and groupwise models

        Returns:
            Dict mapping reward_name -> rewards tensor aligned with local samples
        """
        results: Dict[str, torch.Tensor] = {}

        # Pointwise: local computation
        if split in ("pointwise", "all") and self._pointwise_models:
            results.update(self._compute_pointwise_rewards(samples, epoch))

        # Groupwise: gather -> compute -> scatter
        if split in ("groupwise", "all") and self._groupwise_models:
            results.update(self._compute_groupwise_rewards(samples, epoch))

        self.accelerator.wait_for_everyone()
        # Store to samples
        if store_to_samples:
            for i, sample in enumerate(samples):
                sample.extra_kwargs["rewards"] = {k: v[i] for k, v in results.items()}

        return results

    # ============================ Pointwise Computation ============================
    def _compute_pointwise_rewards(
        self,
        samples: List[BaseSample],
        epoch: Optional[int] = None,
        models: Optional[Dict[str, PointwiseRewardModel]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards for PointwiseRewardModels."""
        models = models if models is not None else self._pointwise_models
        results: Dict[str, torch.Tensor] = {}

        for name, model in models.items():
            rewards = []
            batch_size = self._resolve_batch_size(name, model)

            desc = (
                f"Epoch {epoch} Pointwise Rewards: {name}"
                if epoch is not None
                else f"Pointwise Rewards: {name}"
            )
            pbar = tqdm(
                range(0, len(samples), batch_size),
                desc=desc,
                disable=not self.show_progress_bar,
            )
            for i in pbar:
                batch_samples = samples[i : i + batch_size]
                reward_tensor = self._compute_pointwise_batch(name, model, batch_samples)
                rewards.append(reward_tensor)

            results[name] = torch.cat(rewards, dim=0)

        return results

    # ============================ Groupwise Computation ============================
    def _compute_groupwise_rewards(
        self,
        samples: List[BaseSample],
        epoch: Optional[int] = None,
        models: Optional[Dict[str, GroupwiseRewardModel]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rewards for GroupwiseRewardModels.

        Dispatches to local or distributed path based on ``group_on_same_rank``:
        - **Local**: all K copies already on this rank → compute directly, no communication.
        - **Distributed**: gather samples across ranks → stride-partition groups →
          compute → all_reduce → scatter back.
        """
        models = models if models is not None else self._groupwise_models
        if self.group_on_same_rank:
            return self._compute_groupwise_local(samples, models, epoch)
        else:
            return self._compute_groupwise_distributed(samples, models, epoch)

    def _compute_groupwise_local(
        self,
        samples: List[BaseSample],
        models: Dict[str, GroupwiseRewardModel],
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Local groupwise computation — no cross-rank communication.

        Used for a rank-local sampler layout (currently ``group_contiguous``):
        all K copies of each comparison group reside on the same rank, so we
        group and compute entirely locally.

        Each per-group call routes through :meth:`_compute_groupwise_group`,
        which applies the source-aware gate + NaN-pad uniformly with the
        sync pointwise / async paths.
        """
        groups, inverse = self.group_samples(samples, key=None, return_inverse=True)
        group_keys = list(groups.keys())

        # Sanity check: all groups must have the same size (= K)
        group_sizes = {uid: len(g) for uid, g in groups.items()}
        bad = {uid: sz for uid, sz in group_sizes.items() if sz != next(iter(group_sizes.values()))}
        if bad:
            raise RuntimeError(
                f"group_on_same_rank=True requires uniform group sizes on each rank, "
                f"but found mismatched groups: {bad}. Please check your sampler configuration."
            )

        results: Dict[str, torch.Tensor] = {}
        for name, model in models.items():
            all_rewards = torch.full((len(samples),), float("nan"), dtype=torch.float32)
            desc = (
                f"Epoch {epoch} Groupwise Rewards: {name}"
                if epoch is not None
                else f"Groupwise Rewards: {name}"
            )
            pbar = tqdm(
                range(len(group_keys)),
                desc=desc,
                disable=not self.show_progress_bar,
            )
            for group_idx in pbar:
                uid = group_keys[group_idx]
                group_list = groups[uid]
                # Gated + NaN-padded compute. all_rewards is already NaN-init,
                # so non-applicable groups land as NaN automatically.
                group_rewards = self._compute_groupwise_group(name, model, group_list)
                all_rewards[inverse == group_idx] = group_rewards

            results[name] = all_rewards

        return results

    def _compute_groupwise_distributed(
        self,
        samples: List[BaseSample],
        models: Dict[str, GroupwiseRewardModel],
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Distributed groupwise computation with gather → stride → all_reduce → scatter.

        Used for every cross-rank sampler layout. K copies may be scattered
        across ranks, so we gather all samples, partition groups by stride,
        compute, all-reduce, and scatter back.

        Source-aware gating: each group is owned by exactly one rank
        under the stride partition.  That rank computes its rewards via
        :meth:`_compute_groupwise_group`, which writes NaN at
        non-applicable positions (and finite scores otherwise).  Other
        ranks contribute zeros for groups they don't own; the
        ``reduce(sum)`` therefore produces NaN at non-applicable group
        positions and finite scores at applicable ones — gather/reduce
        participants stay shape-uniform across ranks (no deadlock).
        """
        device = self.accelerator.device
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes

        # 1. Collect required fields from all groupwise models
        required_fields: Set[str] = set()
        for model in models.values():
            required_fields.update(model.required_fields)

        # Always include the typed source bookkeeping — the gate needs
        # `source` (and ideally `source_id`) on the gathered side.  Now
        # that they're real dataclass fields on `BaseSample`, gathering
        # them directly is cheaper than dragging the whole `extra_kwargs`
        # dict across ranks (the latter forces every extra key to be
        # packed/unpacked, which is wasteful when we only need one or
        # two of them).
        required_fields.add("source")
        required_fields.add("source_id")
        # Preserve the exact signed 64-bit identity rather than relying on the
        # gathered subset of conditioning fields to reproduce its hash.
        for sample in samples:
            sample.unique_id
        required_fields.add("_unique_id")

        # Optimize: use prompt_ids instead of prompt strings for communication
        needs_decode = False
        if "prompt" in required_fields:
            if hasattr(samples[0], "prompt_ids") and samples[0].prompt_ids is not None:
                required_fields.discard("prompt")
                required_fields.add("prompt_ids")
                needs_decode = True

        # 2. Sync and gather samples from all ranks
        self.accelerator.wait_for_everyone()
        gathered = gather_samples(
            accelerator=self.accelerator,
            samples=samples,
            field_names=list(required_fields),
            device=device,
        )

        # Decode prompts if needed
        if needs_decode:
            prompts = self._decode_prompts([s.prompt_ids for s in gathered])
            for i, s in enumerate(gathered):
                s.prompt = prompts[i]

        # 3. Group by the source-aware reward identity and build inverse mapping.
        groups, inverse = self.group_samples(gathered, key=None, return_inverse=True)
        group_keys = list(groups.keys())
        num_gathered = len(gathered)

        # 4. Stride distribution: rank i handles groups [i, i+W, i+2W, ...]
        local_group_indices = list(range(rank, len(group_keys), world_size))

        # 4b. Pre-compute per-group applicability (deterministic across all
        # ranks because they all see the same `gathered` and the same
        # reward configs).  Used to:
        #   - mark applicable_rewards on every rank's gathered view (so
        #     the bookkeeping is consistent before the post-reduce scatter);
        #   - post-process the reduced tensor to write NaN at
        #     non-applicable group positions (the owning rank produced 0
        #     there during compute; NaN-replacement happens once after
        #     the all-reduce).
        per_group_applicable: Dict[str, np.ndarray] = {}
        for name in models.keys():
            applicable = np.zeros(len(group_keys), dtype=bool)
            for g_idx, uid in enumerate(group_keys):
                # All K samples in a group share __source__ (asserted in
                # _compute_groupwise_group), so checking sample[0] suffices.
                applicable[g_idx] = self._reward_applies(name, groups[uid][0])
            per_group_applicable[name] = applicable

        # 5. Compute rewards per model (accumulate for batched reduce).
        model_names = list(models.keys())
        reward_columns: List[torch.Tensor] = []

        for name, model in models.items():
            all_rewards = torch.zeros(num_gathered, dtype=torch.float32, device=device)
            applicable_groups = per_group_applicable[name]

            desc = (
                f"Epoch {epoch} Groupwise Rewards: {name}"
                if epoch is not None
                else f"Groupwise Rewards: {name}"
            )
            pbar = tqdm(
                local_group_indices,
                desc=desc,
                disable=not self.show_progress_bar,
            )
            for group_idx in pbar:
                uid = group_keys[group_idx]
                group_list = groups[uid]

                if not applicable_groups[group_idx]:
                    continue

                fields = filter_kwargs(model.__call__, **group_list[0])
                group_input = {
                    k: [getattr(s, k) for s in group_list]
                    for k in fields
                    if all(getattr(s, k) is not None for s in group_list)
                }
                group_input = self._convert_media_format(group_input, model)

                output = model(**group_input)
                group_rewards = torch.as_tensor(
                    output.rewards if hasattr(output, "rewards") else output,
                    device=device,
                    dtype=torch.float32,
                )

                if not torch.isfinite(group_rewards).all():
                    bad = (~torch.isfinite(group_rewards)).nonzero(as_tuple=True)[0].tolist()
                    raise RuntimeError(
                        f"Groupwise reward '{name}' produced NaN/Inf at applicable "
                        f"positions {bad} of group uid={uid}. This is a model bug, "
                        "not a routing miss."
                    )

                mask_t = inverse == group_idx
                all_rewards[mask_t] = group_rewards

            reward_columns.append(all_rewards)

        # 6. Batched all-reduce: pack M reward vectors into (W*B, M),
        # reduce once, then unpack. M sequential NCCL calls -> 1.
        packed_rewards = torch.stack(reward_columns, dim=1)  # (W*B, M)
        packed_rewards = self.accelerator.reduce(packed_rewards, reduction="sum")

        # 6b. NaN-fill non-applicable group positions + unpack.
        results: Dict[str, torch.Tensor] = {}
        for m_idx, name in enumerate(model_names):
            cpu = packed_rewards[:, m_idx].cpu()
            applicable_groups = per_group_applicable[name]
            nan_mask = ~applicable_groups[inverse]
            cpu[nan_mask] = float("nan")
            results[name] = cpu

        # 7. Scatter back to local rank
        results = {k: v.chunk(world_size)[rank] for k, v in results.items()}

        return results

    # ============================ Prompt Encoding/Decoding ============================
    def _decode_prompts(self, prompt_ids_list: List[torch.Tensor]) -> List[str]:
        """Decode prompt_ids to strings."""
        if self.tokenizer is None:
            raise ValueError("Cannot decode prompts: tokenizer not provided")

        return [
            self.tokenizer.decode(
                ids.cpu().tolist() if isinstance(ids, torch.Tensor) else ids,
                skip_special_tokens=True,
            )
            for ids in prompt_ids_list
        ]

    # ============================ Helper Functions ============================
    @staticmethod
    def compute_group_zero_std_ratio(
        rewards: np.ndarray, group_indices: np.ndarray, eps: float = 1e-6
    ) -> float:
        """
        Compute the fraction of groups with near-zero standard deviation.

        Args:
            rewards: Array of reward values
            group_indices: Array mapping each sample to its group
            eps: Threshold for considering std as zero

        Returns:
            Fraction of groups with std < eps
        """
        unique_groups = np.unique(group_indices)
        zero_std_count = sum(
            1 for gid in unique_groups if np.std(rewards[group_indices == gid]) < eps
        )
        return zero_std_count / len(unique_groups)

    @staticmethod
    def compute_group_reward_stats(
        rewards: np.ndarray,
        group_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-group reward statistics.

        Args:
            rewards: Array of reward values, shape (N,)
            group_indices: Array mapping each sample to its group index, shape (N,)

        Returns:
            group_means: Per-group mean rewards, shape (num_groups,)
            group_stds:  Per-group std of rewards, shape (num_groups,)
        """
        unique_groups = np.unique(group_indices)
        group_stds = np.array([np.std(rewards[group_indices == gid]) for gid in unique_groups])
        group_means = np.array([np.mean(rewards[group_indices == gid]) for gid in unique_groups])
        return group_means, group_stds

    @staticmethod
    def group_samples(
        samples: List[BaseSample],
        key: Optional[str] = "unique_id",
        return_inverse: bool = False,
    ) -> Union[Dict[Any, List[BaseSample]], Tuple[Dict[Any, List[BaseSample]], np.ndarray]]:
        """
        Group samples by a key field, similar to np.unique.

        Args:
            samples: List of BaseSample instances
            key: Field name to group by. ``None`` selects the canonical
                source-aware :class:`GroupKey` used by reward algorithms.
            return_inverse: If True, return indices to reconstruct original order
        Returns:
            groups: Dict mapping key_value -> List[BaseSample]
            inverse: (optional) Array where inverse[i] gives group index for samples[i]
        """
        keys = [
            sample_group_key(sample) if key is None else getattr(sample, key) for sample in samples
        ]
        groups: Dict[Any, List[BaseSample]] = {}
        group_indices: Dict[Any, int] = {}
        inverse = np.empty(len(samples), dtype=np.int64)
        for index, (sample, group_key) in enumerate(zip(samples, keys)):
            if group_key not in groups:
                group_indices[group_key] = len(groups)
                groups[group_key] = []
            groups[group_key].append(sample)
            inverse[index] = group_indices[group_key]

        return (groups, inverse) if return_inverse else groups


# ============================ Reward Buffer ============================
class RewardBuffer:
    """
    Unified reward computation buffer with per-model async/sync control.

    Each reward model's ``RewardArguments.async_reward`` determines its mode:
        - **async** models: rewards are computed concurrently via a
          ``ThreadPoolExecutor`` as samples arrive. The pool size is
          determined by each model's ``num_workers`` config, enabling
          true IO-level parallelism for API-based reward models.
        - **sync** models: samples are accumulated and rewards are computed
          in ``finalize()`` using the standard ``RewardProcessor`` path.

    Usage (inside trainer.sample()):
        buffer.clear()
        for batch in dataloader:
            new_samples = adapter.inference(...)
            buffer.add_samples(new_samples)
        rewards = buffer.finalize()
    """

    def __init__(self, reward_processor: RewardProcessor, group_size: int):
        self.rp = reward_processor
        self.group_size = group_size
        self.all_samples: List[BaseSample] = []

        # Partition all reward models into async / sync groups based on
        # each model's RewardArguments.async_reward setting.
        self._async_pointwise = {
            n: m for n, m in self.rp._pointwise_models.items() if self.rp._is_async_reward(n)
        }
        self._sync_pointwise = {
            n: m for n, m in self.rp._pointwise_models.items() if not self.rp._is_async_reward(n)
        }
        self._async_groupwise = {
            n: m for n, m in self.rp._groupwise_models.items() if self.rp._is_async_reward(n)
        }
        self._sync_groupwise = {
            n: m for n, m in self.rp._groupwise_models.items() if not self.rp._is_async_reward(n)
        }
        self._has_async = bool(self._async_pointwise or self._async_groupwise)

        # Pre-create one CUDA stream per unique device among async models
        # (used inside _execute_task for CUDA-based reward models).
        self._reward_streams: Dict[torch.device, torch.cuda.Stream] = {}
        if self._has_async:
            for m in list(self._async_pointwise.values()) + list(self._async_groupwise.values()):
                if m.device.type == "cuda" and m.device not in self._reward_streams:
                    self._reward_streams[m.device] = torch.cuda.Stream(device=m.device)

        self._init_async_state()

    def _init_async_state(self) -> None:
        """Initialize (or reset) all mutable state used by the async path.

        Sets up:
        - ``_rewards``: per-model list of reward scalars (None until filled by futures).
        - ``_pointwise_pending``: per-model list of sample indices awaiting batch dispatch.
        - ``_groupwise_pending``: maps GroupKey -> list of sample indices; dispatched
          when a group reaches ``group_size``.
        - ``_executors``: one ``ThreadPoolExecutor`` lane per async reward, each
          sized by that reward's ``num_workers``.
        - ``_futures``: list of ``(name, indices, Future)`` tuples for result collection.
        """
        self._streaming_sealed = False
        self._streaming_consumed_indices: Set[int] = set()
        if not self._has_async:
            return
        async_names = list(self._async_pointwise) + list(self._async_groupwise)
        self._rewards: Dict[str, List[Optional[torch.Tensor]]] = {n: [] for n in async_names}
        self._pointwise_pending: Dict[str, List[int]] = {n: [] for n in self._async_pointwise}
        self._groupwise_pending: Dict[GroupKey, List[int]] = defaultdict(list)
        self._any_cuda_reward = bool(self._reward_streams)
        self._executors: Dict[str, ThreadPoolExecutor] = {
            name: ThreadPoolExecutor(max_workers=self.rp._resolve_num_workers(name))
            for name in async_names
        }
        self._futures: List[Tuple[str, List[int], Future]] = []
        self._collected_futures: Set[Future] = set()

    # ---- Main thread API ----

    def clear(self) -> None:
        """Reset buffer to initial state for reuse across epochs.

        Shuts down the thread pool (waiting for in-flight tasks) and
        reinitializes all async tracking structures.
        """
        self.all_samples = []
        if self._has_async:
            for executor in self._executors.values():
                executor.shutdown(wait=True)
            self._init_async_state()

    def shutdown(self, wait: bool = False, cancel_futures: bool = True) -> None:
        """Terminate the async executor without reinitializing.

        Unlike ``clear()`` (which waits for tasks and resets for reuse), this
        method is intended for final teardown — e.g. on KeyboardInterrupt —
        where speed matters more than task completion.
        """
        if self._has_async and hasattr(self, "_executors"):
            for executor in self._executors.values():
                executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def add_samples(self, samples: List[BaseSample]) -> None:
        """Accumulate new samples and submit ready async reward tasks.

        For sync models, samples are only accumulated (no computation here).
        For async models, this method:
        1. Extends per-model pending lists with new sample indices.
        2. Records a CUDA event on the current (sampling) stream for
           cross-stream synchronization (only if any async model uses CUDA).
        3. Calls ``_submit_ready_tasks()`` to submit any batches / groups
           that have reached their trigger threshold to the thread pool.

        This method is non-blocking -- tasks are submitted to the executor
        and return ``Future`` objects immediately.
        """
        self.all_samples.extend(samples)
        if not self._has_async:
            return
        # Register new sample indices in per-model pending lists
        start = len(self.all_samples) - len(samples)
        new_indices = list(range(start, start + len(samples)))
        for name in self._async_pointwise:
            self._rewards[name].extend([None] * len(samples))
            self._pointwise_pending[name].extend(new_indices)
        if self._async_groupwise:
            for name in self._async_groupwise:
                self._rewards[name].extend([None] * len(samples))
            for idx, s in zip(new_indices, samples):
                self._groupwise_pending[sample_group_key(s)].append(idx)
        # Record CUDA event so pool workers can wait for sample data readiness
        sync_event = None
        if self._any_cuda_reward:
            sync_event = torch.cuda.Event()
            sync_event.record()
        self._submit_ready_tasks(sync_event)

    @property
    def supports_streaming(self) -> bool:
        """Return whether every configured training reward uses the async lane."""
        return self._has_async and not self._sync_pointwise and not self._sync_groupwise

    @property
    def has_async_groupwise_rewards(self) -> bool:
        """Return whether streaming includes a rank-local groupwise model."""
        return bool(self._async_groupwise)

    def seal_for_streaming(self) -> None:
        """Close async submission and make tile readiness observable.

        Streaming cannot include a synchronous reward because a sync model would
        recreate the full-acquisition barrier that this API is designed to remove.
        """
        if not self.supports_streaming:
            raise RuntimeError(
                "streamed reward optimization requires at least one reward and every "
                "training reward to use async_reward=true"
            )
        if self._streaming_sealed:
            raise RuntimeError("reward buffer is already sealed for streaming")
        self._flush_async_tail_tasks()
        if self._groupwise_pending:
            raise RuntimeError(
                "cannot seal streamed rewards with incomplete groups: "
                f"{tuple(self._groupwise_pending)!r}"
            )
        self._streaming_sealed = True

    def poll_ready_tiles(
        self,
        tile_indices: Dict[int, Tuple[int, ...]],
    ) -> Set[int]:
        """Collect completed futures and return locally ready tile identifiers.

        Args:
            tile_indices: Mapping from tile id to rank-local sample indices.

        Returns:
            Tile ids for which every async reward component has materialized.

        Raises:
            RuntimeError: If the buffer has not been sealed.
        """
        if not self._streaming_sealed:
            raise RuntimeError("seal_for_streaming() must run before polling reward tiles")
        self._collect_async_futures(block=False)
        ready: Set[int] = set()
        reward_lists = tuple(self._rewards.values())
        for tile_id, indices in tile_indices.items():
            self._validate_streaming_indices(indices)
            if all(
                reward_list[index] is not None for reward_list in reward_lists for index in indices
            ):
                ready.add(tile_id)
        return ready

    def resolve_streaming_tile(
        self,
        indices: Tuple[int, ...],
        *,
        store_to_samples: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return one ready tile's rewards in tile-local sample order.

        Args:
            indices: Stable rank-local indices belonging to one tile.
            store_to_samples: Whether to attach rewards to the source samples.

        Returns:
            Reward tensors keyed by configured reward name.

        Raises:
            RuntimeError: If the tile is unsealed, incomplete, or already consumed.
        """
        if not self._streaming_sealed:
            raise RuntimeError("seal_for_streaming() must run before resolving a reward tile")
        self._validate_streaming_indices(indices)
        overlap = self._streaming_consumed_indices.intersection(indices)
        if overlap:
            raise RuntimeError(
                f"reward tile sample indices were already consumed: {sorted(overlap)!r}"
            )
        self._collect_async_futures(block=False)
        results: Dict[str, torch.Tensor] = {}
        for name, reward_list in self._rewards.items():
            missing = [index for index in indices if reward_list[index] is None]
            if missing:
                raise RuntimeError(
                    f"reward tile is not ready for model {name!r}; missing indices={missing!r}"
                )
            results[name] = torch.stack([reward_list[index] for index in indices])
        if store_to_samples:
            for local_index, sample_index in enumerate(indices):
                self.all_samples[sample_index].extra_kwargs["rewards"] = {
                    name: values[local_index] for name, values in results.items()
                }
        self._streaming_consumed_indices.update(indices)
        return results

    def finish_streaming(self) -> Dict[str, torch.Tensor]:
        """Finish a streamed cycle and return full rewards in rollout order."""
        if not self._streaming_sealed:
            raise RuntimeError("seal_for_streaming() must run before finishing streamed rewards")
        self._collect_async_futures(block=True)
        expected = set(range(len(self.all_samples)))
        if self._streaming_consumed_indices != expected:
            missing = sorted(expected - self._streaming_consumed_indices)
            unexpected = sorted(self._streaming_consumed_indices - expected)
            raise RuntimeError(
                "streamed reward cycle did not consume every sample exactly once: "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )
        self._synchronize_reward_streams()
        results = self._assemble_async_results()
        self._streaming_sealed = False
        return results

    def abort_streaming(self) -> None:
        """Cancel futures that have not started after a streamed-cycle failure."""
        if self._has_async:
            for _name, _indices, future in self._futures:
                if future not in self._collected_futures:
                    future.cancel()
        self._streaming_sealed = False

    def _validate_streaming_indices(self, indices: Tuple[int, ...]) -> None:
        """Validate one tile's rank-local index identity."""
        if not indices:
            raise ValueError("reward tile indices cannot be empty")
        if len(set(indices)) != len(indices):
            raise ValueError(f"reward tile indices contain duplicates: {indices!r}")
        bad = [
            index
            for index in indices
            if type(index) is not int or not 0 <= index < len(self.all_samples)
        ]
        if bad:
            raise IndexError(
                f"reward tile indices are outside [0, {len(self.all_samples)}): {bad!r}"
            )

    def finalize(
        self,
        store_to_samples: bool = True,
        split: Literal["pointwise", "groupwise", "all"] = "all",
    ) -> Dict[str, torch.Tensor]:
        """Complete all reward computation and return the merged result dict.

        Execution order:
        1. Compute **sync** rewards on the main thread (pointwise then
           groupwise with cross-rank gather/scatter).
        2. Flush remaining async tasks (tail samples < batch_size),
           then collect all ``Future`` results with a progress bar.
        3. Merge sync + async results into a single dict.
        4. Optionally store per-sample rewards into ``sample.extra_kwargs``.
        """
        results: Dict[str, torch.Tensor] = {}

        # 1. Compute sync rewards (blocking, on main thread)
        if split in ("pointwise", "all") and self._sync_pointwise:
            results.update(
                self.rp._compute_pointwise_rewards(self.all_samples, models=self._sync_pointwise)
            )
        if split in ("groupwise", "all") and self._sync_groupwise:
            results.update(
                self.rp._compute_groupwise_rewards(self.all_samples, models=self._sync_groupwise)
            )

        # 2. Flush and collect async rewards
        if self._has_async:
            async_results = self._finalize_async()
            results.update(async_results)

        self.rp.accelerator.wait_for_everyone()

        # 3. Store to samples
        if store_to_samples:
            for i, sample in enumerate(self.all_samples):
                sample.extra_kwargs["rewards"] = {k: v[i] for k, v in results.items()}

        return results

    # ---- Async internals ----

    def _execute_task(self, task_type: str, name: str, model, samples, sync_event) -> torch.Tensor:
        """Execute a single reward computation task (runs in thread pool worker).

        Handles CUDA stream context for GPU-based models and waits on
        the sync_event to ensure sample tensors are ready before reading.
        For CPU / API-based models, runs directly without stream context.
        """
        stream = self._reward_streams.get(model.device)
        ctx = torch.cuda.stream(stream) if stream else nullcontext()
        with ctx:
            if sync_event is not None and stream is not None:
                stream.wait_event(sync_event)
            if task_type == "pointwise":
                return self.rp._compute_pointwise_batch(name, model, samples)
            else:
                return self.rp._compute_groupwise_group(name, model, samples)

    def _submit_ready_tasks(self, sync_event) -> None:
        """Check pending lists and submit tasks that meet their trigger condition.

        - Pointwise: submitted when a model's pending count >= its batch_size.
          Each model has its own pending list so different batch_sizes are
          handled independently.
        - Groupwise: submitted when a source-aware group accumulates
          ``group_size`` samples. One task is created per async groupwise model
          for the completed group.

        Each task is submitted to the ``ThreadPoolExecutor`` and the resulting
        ``Future`` is stored in ``_futures`` for collection in ``_finalize_async``.
        """
        # Pointwise: dispatch full batches per model
        for name, model in self._async_pointwise.items():
            bs = self.rp._resolve_batch_size(name, model)
            while True:
                batch_idx = self._pop_ready_pointwise_indices(name, bs)
                if batch_idx is None:
                    break
                self._submit_pointwise_task(name, model, batch_idx, sync_event)
        # Groupwise: dispatch complete groups
        for group_key, indices in list(self._groupwise_pending.items()):
            if len(indices) >= self.group_size:
                group_samples = [self.all_samples[i] for i in indices]
                for name, model in self._async_groupwise.items():
                    future = self._executors[name].submit(
                        self._execute_task,
                        "groupwise",
                        name,
                        model,
                        group_samples,
                        sync_event,
                    )
                    self._futures.append((name, list(indices), future))
                del self._groupwise_pending[group_key]

    def _pop_ready_pointwise_indices(
        self,
        name: str,
        batch_size: int,
    ) -> Optional[List[int]]:
        """Pop one full reward-model batch without optimizer-boundary coupling."""
        pending = self._pointwise_pending[name]
        if not pending:
            return None
        count = batch_size if len(pending) >= batch_size else 0
        if count == 0:
            return None
        batch_idx = pending[:count]
        self._pointwise_pending[name] = pending[count:]
        return batch_idx

    def _submit_pointwise_task(
        self,
        name: str,
        model: PointwiseRewardModel,
        indices: List[int],
        sync_event: Optional[torch.cuda.Event],
    ) -> None:
        """Submit one pointwise request on its independent component lane."""
        batch_samples = [self.all_samples[index] for index in indices]
        future = self._executors[name].submit(
            self._execute_task,
            "pointwise",
            name,
            model,
            batch_samples,
            sync_event,
        )
        self._futures.append((name, indices, future))

    def _flush_async_tail_tasks(self) -> None:
        """Submit pointwise tail batches that did not reach their configured size."""
        sync_event = None
        if self._any_cuda_reward:
            sync_event = torch.cuda.Event()
            sync_event.record()
        for name, model in self._async_pointwise.items():
            pending = self._pointwise_pending.get(name, [])
            while pending:
                count = min(len(pending), self.rp._resolve_batch_size(name, model))
                batch_idx = pending[:count]
                self._pointwise_pending[name] = pending[count:]
                self._submit_pointwise_task(name, model, batch_idx, sync_event)
                pending = self._pointwise_pending[name]

    def _collect_async_futures(self, *, block: bool) -> None:
        """Materialize newly completed async results into their stable sample rows."""
        for name, indices, future in self._futures:
            if future in self._collected_futures or (not block and not future.done()):
                continue
            self._collect_async_future(name, indices, future)

    def _collect_async_future(self, name: str, indices: List[int], future: Future) -> None:
        """Collect one future exactly once into its stable sample rows."""
        if future in self._collected_futures:
            return
        rewards = future.result()
        if len(rewards) != len(indices):
            raise RuntimeError(
                f"async reward {name!r} returned {len(rewards)} rows for sample indices {indices!r}"
            )
        for offset, sample_index in enumerate(indices):
            if self._rewards[name][sample_index] is not None:
                raise RuntimeError(
                    f"async reward {name!r} produced duplicate row for sample index {sample_index}"
                )
            self._rewards[name][sample_index] = rewards[offset]
        self._collected_futures.add(future)

    def _synchronize_reward_streams(self) -> None:
        """Wait for CUDA reward streams before exposing a completed cycle."""
        for stream in self._reward_streams.values():
            stream.synchronize()

    def _assemble_async_results(self) -> Dict[str, torch.Tensor]:
        """Stack every async component in stable rank-local rollout order."""
        results: Dict[str, torch.Tensor] = {}
        for name, reward_list in self._rewards.items():
            missing = [index for index, reward in enumerate(reward_list) if reward is None]
            if missing:
                raise RuntimeError(f"missing rows for async reward {name!r}: {missing!r}")
            results[name] = torch.stack(reward_list)
        return results

    def _finalize_async(self) -> Dict[str, torch.Tensor]:
        """Flush tail tasks, collect all futures, and assemble results.

        Steps:
        1. Submit tail pointwise samples (< batch_size) that weren't
           dispatched during ``add_samples()``.
        2. Iterate over all ``Future`` objects, calling ``.result()`` to
           block until each completes. A tqdm progress bar tracks completion.
        3. Verify all groupwise groups were completed.
        4. Synchronize any CUDA streams used by async models.
        5. Stack per-model reward lists into tensors and return.
        """
        # 1. Flush remaining pointwise pending (tail < batch_size)
        self._flush_async_tail_tasks()
        if self._groupwise_pending:
            raise RuntimeError(
                "incomplete async reward groups remain at finalize: "
                f"{tuple(self._groupwise_pending)!r}"
            )
        if not self._futures:
            return {}
        # 2. Collect all futures with progress bar
        num_async = len(self._async_pointwise) + len(self._async_groupwise)
        total = len(self.all_samples) * num_async
        completed = 0
        with tqdm(
            total=total,
            desc="Async Rewards",
            disable=not self.rp.show_progress_bar,
        ) as pbar:
            for name, indices, future in self._futures:
                self._collect_async_future(name, indices, future)
                completed += len(indices)
                pbar.n = completed
                pbar.refresh()
        # 3. Synchronize CUDA streams
        self._synchronize_reward_streams()
        # 4. Assemble results
        return self._assemble_async_results()
