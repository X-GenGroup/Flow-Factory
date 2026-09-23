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

# src/flow_factory/advantage/advantage_processor.py
"""
Communication-aware Advantage Processor.

Extracts advantage computation logic from GRPOTrainer into a standalone,
reusable component.  Automatically selects the communication strategy based
on the resolved sampler type:

- ``distributed_k_repeat``: gather rewards + unique_ids across ranks →
  global grouping → scatter back to local rank.
- ``group_contiguous``: all K copies already reside on the same rank →
  skip all cross-rank communication for advantage computation.  Training log
  metrics are computed via mode-aware ``_metric_*`` helpers that transparently
  select between plain NumPy (post-gather global arrays) and ``utils.dist``
  reductions (local shards) so logging always reflects global statistics.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator

from ..rewards import RewardProcessor
from ..samples import BaseSample
from ..utils.dist import global_tensor_stats_batch
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class _PreparedGroupRewardCollection:
    """Rank-local tensor payload prepared before a distributed reward gather."""

    reward_keys: Tuple[str, ...]
    packed: torch.Tensor


class AdvantageProcessor:
    """Communication-aware advantage computation processor.

    Parameters
    ----------
    accelerator : Accelerator
        HuggingFace Accelerator instance for distributed ops.
    reward_weights : dict[str, dict[str, float]]
        Mapping from reward name to per-dataset weights
        (``{reward_name: {dataset_name: weight}}``).  Resolved by
        ``Arguments._resolve_reward_weights`` from scalar or dict form.
    group_size : int
        Number of repeated samples per unique prompt (K).
    global_std : bool
        If ``True``, normalise advantages using the global std across all
        groups; otherwise use per-group std.
    sampler_type : str
        One of ``"distributed_k_repeat"`` or ``"group_contiguous"``.
        Determines whether cross-rank communication is needed.
    verbose : bool
        Whether to emit progress information.

    Notes
    -----
    After :meth:`compute_advantages` with ``'sum'`` or ``'gdpo'``, call
    :meth:`pop_advantage_metrics` once to retrieve training metrics (including
    ``train_samples``) for ``log_data``. Custom callables leave an empty metrics
    snapshot. This class does not perform logging itself.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        reward_weights: Dict[str, Dict[str, float]],
        group_size: int,
        global_std: bool = True,
        sampler_type: str = "distributed_k_repeat",
        verbose: bool = True,
        source_id_to_name: Optional[List[str]] = None,
    ):
        self.accelerator = accelerator
        self.reward_weights = reward_weights
        self.group_size = group_size
        self.global_std = global_std
        self.sampler_type = sampler_type
        self.verbose = verbose
        self._source_id_to_name = source_id_to_name or []

        self.group_on_same_rank = sampler_type == "group_contiguous"
        self._pending_advantage_metrics: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pop_advantage_metrics(self) -> Dict[str, Any]:
        """Return and clear metrics from the last ``sum`` / ``gdpo`` advantage pass.

        Call once per :meth:`compute_advantages` when using built-in aggregation.
        Returns an empty dict if nothing was produced (e.g. custom callable only,
        or no prior computation).
        """
        out = dict(self._pending_advantage_metrics or {})
        self._pending_advantage_metrics = None
        return out

    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func: Optional[Union[Literal["sum", "gdpo"], Callable]] = None,
    ) -> torch.Tensor:
        """Compute advantages and retain acquisition metrics for the caller."""
        return self._compute_advantages(
            samples,
            rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
            build_metrics=True,
        )

    def _compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func: Optional[Union[Literal["sum", "gdpo"], Callable]] = None,
        build_metrics: bool = True,
        prepared_collection: Optional[_PreparedGroupRewardCollection] = None,
    ) -> torch.Tensor:
        """Compute per-sample advantages.

        Parameters
        ----------
        samples : list[BaseSample]
            Samples on the current rank.
        rewards : dict[str, Tensor]
            Per-reward-model reward tensors aligned with *samples*.
        store_to_samples : bool
            Write computed advantages into ``sample.extra_kwargs['advantage']``.
        aggregation_func : str or callable
            ``'sum'`` for weighted-sum GRPO, ``'gdpo'`` for GDPO-style, or a
            custom ``callable(processor, samples, rewards, store_to_samples)``.
        build_metrics : bool
            Whether to build acquisition metrics for this internal pass.
        prepared_collection : _PreparedGroupRewardCollection, optional
            Cross-rank payload whose rank-local construction already passed a
            synchronized error guard.

        Returns
        -------
        Tensor
            Advantages for the local rank, shape ``(len(samples),)``.
        """
        self._pending_advantage_metrics = None
        aggregation_func = aggregation_func or "gdpo"
        if aggregation_func == "sum":
            return self.compute_weighted_sum(
                samples,
                rewards,
                store_to_samples,
                build_metrics=build_metrics,
                prepared_collection=prepared_collection,
            )
        elif aggregation_func == "gdpo":
            return self.compute_gdpo(
                samples,
                rewards,
                store_to_samples,
                build_metrics=build_metrics,
                prepared_collection=prepared_collection,
            )
        elif callable(aggregation_func):
            if not build_metrics:
                raise ValueError(
                    "custom advantage aggregation cannot disable metrics because its "
                    "metric ownership is not declared"
                )
            adv = aggregation_func(self, samples, rewards, store_to_samples)
            if self._pending_advantage_metrics is None:
                self._pending_advantage_metrics = {}
            return adv
        else:
            raise ValueError(
                f"Unsupported advantage aggregation method: {aggregation_func}. "
                "Supported: ['sum', 'gdpo'] "
                "or a callable function that takes (processor, samples, rewards, store_to_samples) as inputs."
            )

    # ------------------------------------------------------------------
    # Communication layer
    # ------------------------------------------------------------------

    def prepare_group_reward_collection(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        *,
        require_all_rewards: bool = False,
    ) -> Optional[_PreparedGroupRewardCollection]:
        """Build the local cross-rank payload without entering a collective.

        Overlap callers run this method behind a distributed error guard. Once
        it succeeds on every rank, :meth:`collect_group_rewards` can enter its
        gather without any remaining rank-local conversion or shape work.
        Rank-local group layouts need no preparation because they do not gather.
        """
        if self.group_on_same_rank:
            return None

        reward_keys = tuple(sorted(rewards))
        if not reward_keys:
            raise ValueError("distributed advantage computation requires at least one reward")
        if require_all_rewards:
            expected_keys = tuple(sorted(self.reward_weights))
            if reward_keys != expected_keys:
                raise ValueError(
                    "distributed advantage reward keys do not match configured weights: "
                    f"expected={expected_keys!r}, received={reward_keys!r}"
                )

        sample_count = len(samples)
        if sample_count == 0:
            raise ValueError("distributed advantage computation requires at least one sample")
        device = self.accelerator.device
        columns: List[torch.Tensor] = []
        for key in reward_keys:
            values = torch.as_tensor(rewards[key])
            if values.numel() != sample_count:
                raise ValueError(
                    f"reward {key!r} has {values.numel()} values for "
                    f"{sample_count} local samples"
                )
            columns.append(values.reshape(-1).to(device=device, dtype=torch.float32))

        unique_ids = torch.tensor(
            [int(sample.unique_id) for sample in samples],
            dtype=torch.int64,
            device=device,
        )
        source_ids = torch.tensor(
            [int(sample.source_id) if sample.source_id is not None else -1 for sample in samples],
            dtype=torch.int64,
            device=device,
        )
        columns.extend((unique_ids.float(), source_ids.float()))
        packed = torch.stack(columns, dim=1)
        expected_shape = (sample_count, len(reward_keys) + 2)
        if tuple(packed.shape) != expected_shape:
            raise RuntimeError(
                "distributed advantage payload has an invalid shape: "
                f"expected={expected_shape}, received={tuple(packed.shape)}"
            )
        return _PreparedGroupRewardCollection(reward_keys=reward_keys, packed=packed)

    def collect_group_rewards(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        *,
        prepared_collection: Optional[_PreparedGroupRewardCollection] = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Collect rewards, group indices, and source IDs in one gather.

        ``group_contiguous``: no communication; arrays are local ``(B,)``.
        ``distributed_k_repeat``: rewards + ``unique_id`` + ``source_id``
        are packed into a single ``(B, N+2)`` tensor and gathered with
        one ``accelerator.gather()`` call. Arrays are global ``(W*B,)``.

        Returns:
            collected_rewards: ``{reward_name: np.ndarray}``
            group_indices: integer array mapping each sample to its group
            gathered_source_ids: integer array of source IDs (``-1`` = legacy)
        """
        if self.group_on_same_rank:
            if prepared_collection is not None:
                raise ValueError("rank-local advantage computation cannot use a gathered payload")
            collected_rewards = {
                key: torch.as_tensor(value).cpu().numpy() for key, value in rewards.items()
            }
            unique_ids = np.array([s.unique_id for s in samples], dtype=np.int64)
            _unique_ids, group_indices = np.unique(unique_ids, return_inverse=True)
            source_ids = np.array(
                [s.source_id if s.source_id is not None else -1 for s in samples],
                dtype=np.int64,
            )
            return collected_rewards, group_indices, source_ids
        else:
            prepared_collection = prepared_collection or self.prepare_group_reward_collection(
                samples,
                rewards,
            )
            if prepared_collection is None:  # pragma: no cover - guarded by layout above
                raise RuntimeError("distributed reward collection did not build a payload")
            reward_keys = list(prepared_collection.reward_keys)
            gathered = (
                self.accelerator.gather(prepared_collection.packed).cpu().numpy()
            )  # (W*B, N+2)

            collected_rewards = {key: gathered[:, i] for i, key in enumerate(reward_keys)}
            gathered_ids = gathered[:, -2].astype(np.int64)
            _unique_ids, group_indices = np.unique(gathered_ids, return_inverse=True)
            source_ids = gathered[:, -1].astype(np.int64)
            return collected_rewards, group_indices, source_ids

    def build_source_aware_matrices(
        self,
        samples: List[BaseSample],
        reward_keys: List[str],
        gathered_source_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build ``(R, S)`` applicability mask and weight matrix locally.

        Uses ``applicable_rewards`` from local samples (``group_contiguous``)
        or derives applicability from ``gathered_source_ids`` + config-level
        ``_datasets_resolved`` (``distributed_k_repeat``). Weight matrix
        is computed from ``gathered_source_ids`` + ``reward_weights`` with
        zero communication.

        Args:
            samples: Local samples (used in ``group_contiguous`` path).
            reward_keys: Ordered list of reward names.
            gathered_source_ids: Source IDs from ``collect_group_rewards``.

        Returns:
            Tuple of ``(applicable, weight_matrix)`` both shape ``(R, S)``.
        """
        R = len(reward_keys)
        S = len(gathered_source_ids)

        if self.group_on_same_rank:
            local_mask = np.zeros((R, len(samples)), dtype=bool)
            for j, s in enumerate(samples):
                applicable = s.applicable_rewards
                has_source = s.source is not None or s.source_id is not None
                if not applicable and not has_source:
                    local_mask[:, j] = True
                else:
                    for i, name in enumerate(reward_keys):
                        local_mask[i, j] = name in applicable
            sources = [s.source for s in samples]
            weight_matrix = self._weights_from_sources(reward_keys, sources)
            return local_mask, weight_matrix

        # Distributed: derive applicability from gathered source_ids +
        # config-level reward routing (no communication needed).
        source_names = [
            self._source_id_to_name[sid] if 0 <= sid < len(self._source_id_to_name) else None
            for sid in gathered_source_ids
        ]
        applicable = np.zeros((R, S), dtype=bool)
        for j, src in enumerate(source_names):
            if src is None:
                applicable[:, j] = True
            else:
                for i, key in enumerate(reward_keys):
                    per_ds = self.reward_weights[key]
                    applicable[i, j] = src in per_ds

        weight_matrix = self._weights_from_sources(reward_keys, source_names)
        return applicable, weight_matrix

    def _weights_from_sources(
        self,
        reward_keys: List[str],
        sources: List[Optional[str]],
    ) -> np.ndarray:
        """Build ``(R, S)`` weight matrix from source names (no communication)."""
        R = len(reward_keys)
        S = len(sources)
        matrix = np.ones((R, S), dtype=np.float64)
        for r_idx, key in enumerate(reward_keys):
            per_ds = self.reward_weights[key]
            default_w = next(iter(per_ds.values()))
            for s_idx, src in enumerate(sources):
                if src is not None and src in per_ds:
                    matrix[r_idx, s_idx] = per_ds[src]
                else:
                    matrix[r_idx, s_idx] = default_w
        return matrix

    def _to_local(
        self,
        values: np.ndarray,
    ) -> torch.Tensor:
        """Convert collected values back to a local-rank tensor.

        When ``group_on_same_rank`` is ``True`` the array is already local and
        is simply converted.  Otherwise the array spans all ranks and is sliced
        to this rank's portion.
        """
        if not self.group_on_same_rank:
            values = (
                torch.as_tensor(values)
                .reshape(self.accelerator.num_processes, -1, *values.shape[1:])[
                    self.accelerator.process_index
                ]
                .to(self.accelerator.device)
            )
        else:
            values = torch.as_tensor(values).to(self.accelerator.device)
        return values

    def _global_mean_std(self, values: np.ndarray) -> tuple:
        """Compute global mean and std for *values*.

        When ``group_on_same_rank`` is ``True`` the array only contains
        local-rank data, so we all-reduce ``(count, sum, sum_sq)`` in a
        single call to obtain the true global statistics.  Otherwise the
        array already spans all ranks (post-gather) and we compute
        directly with NumPy — no communication needed.
        """
        if self.group_on_same_rank:
            t = torch.tensor(
                [float(len(values)), float(np.sum(values)), float(np.sum(values**2))],
                device=self.accelerator.device,
            )
            t = self.accelerator.reduce(t, reduction="sum")  # 1 call, 3 scalars
            n, s, ss = t[0].item(), t[1].item(), t[2].item()
            mean = s / n
            std = max((ss / n - mean**2) ** 0.5, 1e-6)
        else:
            mean = float(np.mean(values))
            std = max(float(np.std(values)), 1e-6)
        return mean, std

    # ------------------------------------------------------------------
    # Batched metric reduction (mode-aware)
    # ------------------------------------------------------------------

    def _batch_reduce_stats(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compute global ``{min, max, mean, std}`` for each named array.

        When ``group_on_same_rank`` the arrays are local shards and require
        cross-rank reduction via :func:`dm.global_tensor_stats_batch` (2
        all-reduce calls total, regardless of the number of arrays).

        Otherwise the arrays already span all ranks (post-gather) and stats
        are computed locally with plain NumPy.
        """
        if self.group_on_same_rank:
            tensors = {
                k: torch.from_numpy(np.asarray(v, dtype=np.float64)) for k, v in arrays.items()
            }
            return global_tensor_stats_batch(self.accelerator, tensors)

        out: Dict[str, Dict[str, float]] = {}
        for k, v in arrays.items():
            v = np.asarray(v, dtype=np.float64)
            if len(v) == 0:
                out[k] = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
            else:
                out[k] = {
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "mean": float(np.mean(v)),
                    "std": max(float(np.std(v)), 1e-8),
                }
        return out

    @staticmethod
    def _group_normalize(
        values: np.ndarray,
        group_indices: np.ndarray,
        mask: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Per-group zero-mean unit-variance normalization (vectorized).

        Args:
            values: ``(S,)`` array of values to normalize.
            group_indices: ``(S,)`` integer group assignments.
            mask: ``(S,)`` boolean; only masked-in positions participate.
                ``None`` means all positions participate.
            eps: Minimum std to avoid division by zero.

        Returns:
            ``(S,)`` normalized values (0 at non-participating positions).
        """
        S = len(values)
        num_groups = group_indices.max() + 1
        if mask is None:
            mask = np.ones(S, dtype=bool)

        masked_vals = np.where(mask, values, 0.0)
        counts = np.bincount(group_indices, weights=mask.astype(np.float64), minlength=num_groups)
        sums = np.bincount(group_indices, weights=masked_vals, minlength=num_groups)
        safe_counts = np.maximum(counts, 1.0)
        means = sums / safe_counts

        residuals = np.where(mask, values - means[group_indices], 0.0)
        sq_sums = np.bincount(group_indices, weights=residuals**2, minlength=num_groups)
        stds = np.sqrt(sq_sums / safe_counts)
        stds = np.maximum(stds, eps)

        result = np.zeros(S, dtype=np.float64)
        result[mask] = residuals[mask] / stds[group_indices[mask]]
        return result

    # ------------------------------------------------------------------
    # Strategy: weighted sum (default GRPO)
    # ------------------------------------------------------------------

    def compute_weighted_sum(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool,
        *,
        build_metrics: bool = True,
        prepared_collection: Optional[_PreparedGroupRewardCollection] = None,
    ) -> torch.Tensor:
        """Compute advantages using the weighted-sum GRPO strategy.

        This is the standard GRPO advantage computation.  Each reward model's
        scores are multiplied by its configured weight and summed into a single
        aggregated reward per sample.  Advantages are then group-normalised
        (subtract per-group mean, divide by std).

        **Source-aware aggregation** (plan §6.4): the per-sample
        applicability matrix from :meth:`build_source_aware_matrices` is
        the authoritative source of truth.  NaN at applicable positions
        is asserted to be a model bug (loud failure); NaN at
        non-applicable positions is honored as "this reward doesn't
        contribute to this sample".  Samples with NO applicable reward
        raise -- a misconfigured `RewardArguments.applicable_datasets` shouldn't
        silently produce zero advantages.

        **Algorithm**:

        1. **Collect** — call :meth:`collect_group_rewards` to obtain
           reward arrays and group assignments.
        2. **Aggregate** — compute
           ``r_agg[i] = sum_k(reward_k[i] * weight_k * applicable_k_i)``.
           NaN values at non-applicable positions are zero-weighted; NaN
           at applicable positions raises.
        3. **Group-normalise** — for each group *g*:
           ``advantage[i] = (r_agg[i] - mean(r_agg[g])) / std``
           where *std* is either the global std across all samples (when
           ``global_std=True``) or the per-group std (when ``global_std=False``).
        4. **To-local** — convert back to local-rank tensor via
           :meth:`_to_local`.
        5. **Store** — optionally write advantages into each sample's
           ``extra_kwargs['advantage']``.
        """
        gathered_rewards, group_indices, source_ids = self.collect_group_rewards(
            samples,
            rewards,
            prepared_collection=prepared_collection,
        )
        reward_keys = list(gathered_rewards.keys())
        applicable, weight_matrix = self.build_source_aware_matrices(
            samples, reward_keys, source_ids
        )

        # Bug-detection: NaN at applicable position == reward-model bug.
        stack = np.stack(
            [gathered_rewards[k].astype(np.float64) for k in reward_keys], axis=0
        )  # (R, S)
        nan_mask = ~np.isfinite(stack)
        bug_positions = nan_mask & applicable
        if bug_positions.any():
            r_idx, s_idx = np.where(bug_positions)
            offenders = sorted({reward_keys[i] for i in r_idx})
            raise RuntimeError(
                f"NaN/Inf reward at APPLICABLE positions for reward(s) "
                f"{offenders} (sample indices {sorted(set(s_idx.tolist()))[:10]}{'...' if len(s_idx) > 10 else ''}). "
                "This is a reward-model bug, not a routing miss; "
                "aggregation refuses to silently mask it."
            )

        # Aggregate: weighted sum over applicable rewards only.
        contrib = np.where(applicable, stack, 0.0) * weight_matrix
        aggregated_rewards = contrib.sum(axis=0)  # (S,)

        # Per-sample applicable weight sum -> sanity check.
        weight_per_s = (applicable * weight_matrix).sum(axis=0)  # (S,)
        if (weight_per_s == 0).any():
            bad = np.where(weight_per_s == 0)[0].tolist()
            raise RuntimeError(
                "AdvantageProcessor: samples at indices "
                f"{bad[:10]}{'...' if len(bad) > 10 else ''} have NO applicable "
                "reward (weight_sum == 0). Check that "
                "`RewardArguments.applicable_datasets` covers every training source — "
                "at least one reward must apply to every source."
            )

        # Group-normalise (vectorized via bincount)
        if self.global_std:
            _, std = self._global_mean_std(aggregated_rewards)
            num_groups = group_indices.max() + 1
            sums = np.bincount(group_indices, weights=aggregated_rewards, minlength=num_groups)
            counts = np.bincount(group_indices, minlength=num_groups)
            means = sums / np.maximum(counts, 1)
            advantages = (aggregated_rewards - means[group_indices]) / std
        else:
            advantages = self._group_normalize(aggregated_rewards, group_indices)

        if build_metrics:
            self._pending_advantage_metrics = self._build_weighted_sum_log_data(
                gathered_rewards,
                group_indices,
                aggregated_rewards,
                advantages,
                samples,
                applicable=applicable,
                reward_keys=reward_keys,
            )

        # Scatter & store
        advantages = self._to_local(advantages)
        if store_to_samples:
            for sample, adv in zip(samples, advantages):
                sample.extra_kwargs["advantage"] = adv
        return advantages

    # ------------------------------------------------------------------
    # Strategy: GDPO
    # ------------------------------------------------------------------

    def compute_gdpo(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool,
        *,
        build_metrics: bool = True,
        prepared_collection: Optional[_PreparedGroupRewardCollection] = None,
    ) -> torch.Tensor:
        """Compute advantages using the GDPO (Group-wise DPO) strategy.

        Unlike :meth:`compute_weighted_sum`, which first aggregates all
        rewards into a single scalar then normalises, GDPO normalises each
        reward **independently** within its group before combining.  This
        prevents a single high-variance reward from dominating the advantage
        signal.

        **Source-aware aggregation**: per-reward group statistics are
        computed only over applicable group members.  Under the
        homogeneous-batch design (plan §6.7) a reward is either
        applicable to ALL K samples of a group or to NONE — so GDPO's
        per-(reward, group) normalisation either fires or is skipped
        entirely for that pair.  Mixed applicability within a group is
        an asserted error (caught upstream in
        ``_compute_groupwise_group``).

        **Algorithm**:

        1. **Collect** — call :meth:`collect_group_rewards` to obtain
           reward arrays and group assignments; also gather the
           per-(reward, sample) applicability matrix.
        2. **Per-reward, per-group, per-applicable normalisation**.
        3. **Combine** — sum per-reward normalised contributions.
        4. **Batch normalisation** — compute global mean and std and
           normalise.
        5. **To-local** — convert back to local-rank tensor.
        6. **Store** — optionally write advantages into each sample's
           ``extra_kwargs['advantage']``.
        """
        gathered_rewards, group_indices, source_ids = self.collect_group_rewards(
            samples,
            rewards,
            prepared_collection=prepared_collection,
        )
        reward_keys = list(gathered_rewards.keys())
        applicable, weight_matrix = self.build_source_aware_matrices(
            samples, reward_keys, source_ids
        )

        # Bug-detection: NaN at applicable position == reward-model bug.
        stack = np.stack([gathered_rewards[k].astype(np.float64) for k in reward_keys], axis=0)
        nan_mask = ~np.isfinite(stack)
        bug_positions = nan_mask & applicable
        if bug_positions.any():
            r_idx, _s_idx = np.where(bug_positions)
            offenders = sorted({reward_keys[i] for i in r_idx})
            raise RuntimeError(
                f"GDPO: NaN/Inf reward at APPLICABLE positions for reward(s) "
                f"{offenders}. This is a reward-model bug, not a routing miss."
            )

        # Per-reward group-wise normalisation, restricted to applicable samples.
        all_reward_advantages = []
        for r_idx, key in enumerate(reward_keys):
            reward_array = gathered_rewards[key].astype(np.float64)
            r_applicable = applicable[r_idx]
            reward_adv = self._group_normalize(reward_array, group_indices, mask=r_applicable)
            all_reward_advantages.append(reward_adv * weight_matrix[r_idx])

        # Combine and batch normalise.
        weight_per_s = (applicable * weight_matrix).sum(axis=0)
        if (weight_per_s == 0).any():
            bad = np.where(weight_per_s == 0)[0].tolist()
            raise RuntimeError(
                "GDPO: samples at indices "
                f"{bad[:10]}{'...' if len(bad) > 10 else ''} have NO applicable "
                "reward. Check `RewardArguments.applicable_datasets` coverage."
            )

        combined_advantages = np.sum(all_reward_advantages, axis=0)
        bn_mean, bn_std = self._global_mean_std(combined_advantages)
        advantages = (combined_advantages - bn_mean) / bn_std

        if build_metrics:
            self._pending_advantage_metrics = self._build_gdpo_log_data(
                gathered_rewards,
                group_indices,
                advantages,
                bn_mean,
                bn_std,
                samples,
                applicable=applicable,
                reward_keys=reward_keys,
            )

        # Scatter & store
        advantages = self._to_local(advantages)
        if store_to_samples:
            for sample, adv in zip(samples, advantages):
                sample.extra_kwargs["advantage"] = adv
        return advantages

    # ------------------------------------------------------------------
    # Log payloads (trainers pass to ``log_data``)
    # ------------------------------------------------------------------

    def _build_base_log_stats(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        applicable: Optional[np.ndarray],
        reward_keys: Optional[List[str]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, bool]]]:
        """Shared boilerplate for both log-data builders.

        Returns (stat_arrays, r_applicable) where stat_arrays is ready
        for ``_batch_reduce_stats`` and r_applicable maps each reward
        key to its boolean mask over gathered samples.
        """
        keys_sorted = sorted(gathered_rewards.keys())
        if applicable is not None and reward_keys is not None:
            r_applicable = {k: applicable[reward_keys.index(k)] for k in keys_sorted}
        else:
            r_applicable = {k: np.ones(len(gathered_rewards[k]), dtype=bool) for k in keys_sorted}

        stat_arrays: Dict[str, np.ndarray] = {}
        for key in keys_sorted:
            mask_k = r_applicable[key]
            stat_arrays[f"reward_{key}"] = gathered_rewards[key][mask_k]

        for key in keys_sorted:
            mask_k = r_applicable[key]
            group_means, group_stds = RewardProcessor.compute_group_reward_stats(
                gathered_rewards[key][mask_k], group_indices[mask_k]
            )
            stat_arrays[f"reward_{key}_g_stds"] = group_stds
            stat_arrays[f"reward_{key}_g_means"] = group_means

        return stat_arrays, r_applicable

    def _unpack_per_reward_log_data(
        self,
        all_stats: Dict[str, Dict[str, float]],
        gathered_rewards: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Unpack per-reward stats common to both log-data builders."""
        _log_data: Dict[str, Any] = {}
        keys_sorted = sorted(gathered_rewards.keys())
        for key in keys_sorted:
            reward_stats = all_stats[f"reward_{key}"]
            _log_data[f"train/reward_{key}_mean"] = reward_stats["mean"]
            _log_data[f"train/reward_{key}_std"] = reward_stats["std"]

        for key in keys_sorted:
            group_std_stats = all_stats[f"reward_{key}_g_stds"]
            group_mean_stats = all_stats[f"reward_{key}_g_means"]
            _log_data[f"train/reward_{key}_group_std_mean"] = group_std_stats["mean"]
            _log_data[f"train/reward_{key}_group_std_max"] = group_std_stats["max"]
            _log_data[f"train/reward_{key}_group_std_min"] = group_std_stats["min"]
            _log_data[f"train/reward_{key}_group_mean_std"] = group_mean_stats["std"]
        return _log_data

    def _build_weighted_sum_log_data(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        aggregated_rewards: np.ndarray,
        advantages: np.ndarray,
        samples: List[BaseSample],
        applicable: Optional[np.ndarray] = None,
        reward_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        stat_arrays, _ = self._build_base_log_stats(
            gathered_rewards, group_indices, applicable, reward_keys
        )

        stat_arrays["reward_agg"] = aggregated_rewards
        agg_group_means, agg_group_stds = RewardProcessor.compute_group_reward_stats(
            aggregated_rewards, group_indices
        )
        stat_arrays["reward_agg_g_stds"] = agg_group_stds
        stat_arrays["reward_agg_g_means"] = agg_group_means
        stat_arrays["reward_agg_zero_std_flags"] = (agg_group_stds < 1e-6).astype(np.float64)
        stat_arrays["adv"] = advantages
        stat_arrays["adv_abs"] = np.abs(advantages)

        all_stats = self._batch_reduce_stats(stat_arrays)

        _log_data = self._unpack_per_reward_log_data(all_stats, gathered_rewards)
        _log_data["train/reward_mean"] = all_stats["reward_agg"]["mean"]
        _log_data["train/reward_std"] = all_stats["reward_agg"]["std"]

        agg_group_std_stats = all_stats["reward_agg_g_stds"]
        agg_group_mean_stats = all_stats["reward_agg_g_means"]
        _log_data["train/reward_group_std_mean"] = agg_group_std_stats["mean"]
        _log_data["train/reward_group_std_max"] = agg_group_std_stats["max"]
        _log_data["train/reward_group_mean_std"] = agg_group_mean_stats["std"]

        _log_data["train/reward_zero_std_ratio"] = all_stats[
            "reward_agg_zero_std_flags"
        ]["mean"]

        # Unpack advantage stats
        adv_stats = all_stats["adv"]
        _log_data["train/adv_min"] = adv_stats["min"]
        _log_data["train/adv_max"] = adv_stats["max"]
        _log_data["train/adv_abs_mean"] = all_stats["adv_abs"]["mean"]

        _log_data["train_samples"] = samples[:30]
        return _log_data

    def _build_gdpo_log_data(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        advantages: np.ndarray,
        bn_mean: float,
        bn_std: float,
        samples: List[BaseSample],
        applicable: Optional[np.ndarray] = None,
        reward_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        stat_arrays, _ = self._build_base_log_stats(
            gathered_rewards, group_indices, applicable, reward_keys
        )

        stat_arrays["adv"] = advantages
        stat_arrays["adv_abs"] = np.abs(advantages)
        keys_sorted = sorted(gathered_rewards.keys())
        for key in keys_sorted:
            group_stds = stat_arrays[f"reward_{key}_g_stds"]
            stat_arrays[f"reward_{key}_zero_std_flags"] = (group_stds < 1e-6).astype(
                np.float64
            )

        all_stats = self._batch_reduce_stats(stat_arrays)

        _log_data = self._unpack_per_reward_log_data(all_stats, gathered_rewards)

        for key in keys_sorted:
            _log_data[f"train/reward_{key}_zero_std_ratio"] = all_stats[
                f"reward_{key}_zero_std_flags"
            ]["mean"]

        adv_stats = all_stats["adv"]
        _log_data.update(
            {
                "train/batch_norm_mean": bn_mean,
                "train/batch_norm_std": bn_std,
                "train/adv_min": adv_stats["min"],
                "train/adv_max": adv_stats["max"],
                "train/adv_abs_mean": all_stats["adv_abs"]["mean"],
                "train_samples": samples[:30],
            }
        )
        return _log_data
