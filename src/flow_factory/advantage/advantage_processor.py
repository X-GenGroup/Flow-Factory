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
  skip all cross-rank communication and compute locally.
"""
from typing import List, Dict, Optional, Union, Literal, Callable, Tuple, Any
import numpy as np
import torch
from accelerate import Accelerator

from ..samples import BaseSample
from ..rewards import RewardProcessor
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class AdvantageProcessor:
    """Communication-aware advantage computation processor.

    Parameters
    ----------
    accelerator : Accelerator
        HuggingFace Accelerator instance for distributed ops.
    reward_weights : dict[str, float]
        Mapping from reward name to its aggregation weight.
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
        reward_weights: Dict[str, float],
        group_size: int,
        global_std: bool = True,
        sampler_type: str = "distributed_k_repeat",
        verbose: bool = True,
    ):
        self.accelerator = accelerator
        self.reward_weights = reward_weights
        self.group_size = group_size
        self.global_std = global_std
        self.sampler_type = sampler_type
        self.verbose = verbose

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

        Returns
        -------
        Tensor
            Advantages for the local rank, shape ``(len(samples),)``.
        """
        self._pending_advantage_metrics = None
        aggregation_func = aggregation_func or "gdpo"
        if aggregation_func == "sum":
            return self.compute_weighted_sum(samples, rewards, store_to_samples)
        elif aggregation_func == "gdpo":
            return self.compute_gdpo(samples, rewards, store_to_samples)
        elif callable(aggregation_func):
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

    def collect_group_rewards(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Collect rewards and group indices, respecting sampler topology.

        Automatically selects between two code paths based on the sampler type:

        - ``group_contiguous``: no cross-rank communication.  Rewards are
          converted to NumPy locally and group indices are derived from
          ``sample.unique_id``.  Returned arrays have shape ``(B,)`` (local).
        - ``distributed_k_repeat``: all per-reward tensors and the
          ``unique_id`` vector are packed into a single ``(B, N+1)`` tensor
          and gathered with one ``accelerator.gather()`` call.  Returned
          arrays have shape ``(W*B,)`` (global, ordered by rank index).

        Whether the returned arrays are local or global is an internal detail
        handled by :meth:`_to_local`.  Callers should not branch on it.

        Parameters
        ----------
        samples : list[BaseSample]
            Samples on the current rank.  Only ``sample.unique_id`` is read.
        rewards : dict[str, Tensor]
            Mapping from reward name to a 1-D tensor of reward values,
            aligned with *samples*.

        Returns
        -------
        collected_rewards : dict[str, np.ndarray]
            Mapping from reward name to a NumPy array of reward values.
        group_indices : np.ndarray
            Integer array mapping each element to its prompt group
            (contiguous integers starting from 0).
        """
        rewards = {
            key: torch.as_tensor(value).to(self.accelerator.device)
            for key, value in rewards.items()
        }

        if self.group_on_same_rank:
            # group_contiguous: all K copies on same rank — no communication
            collected_rewards = {
                key: value.cpu().numpy() for key, value in rewards.items()
            }
            unique_ids = np.array([s.unique_id for s in samples], dtype=np.int64)
            _unique_ids, group_indices = np.unique(unique_ids, return_inverse=True)
            return collected_rewards, group_indices
        else:
            # distributed_k_repeat: pack rewards + ids into one tensor → single gather
            reward_keys = list(rewards.keys())
            unique_ids = torch.tensor(
                [s.unique_id for s in samples],
                dtype=torch.int64,
                device=self.accelerator.device,
            )
            columns = [rewards[k].view(-1).float() for k in reward_keys]
            columns.append(unique_ids.float())
            packed = torch.stack(columns, dim=1)  # (B, N+1)

            gathered = self.accelerator.gather(packed).cpu().numpy()  # (W*B, N+1)

            collected_rewards = {
                key: gathered[:, i] for i, key in enumerate(reward_keys)
            }
            gathered_ids = gathered[:, -1].astype(np.int64)
            _unique_ids, group_indices = np.unique(gathered_ids, return_inverse=True)
            return collected_rewards, group_indices

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
            values = torch.as_tensor(values).reshape(
                self.accelerator.num_processes, -1, *values.shape[1:]
            )[self.accelerator.process_index].to(self.accelerator.device)
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
                [float(len(values)), float(np.sum(values)), float(np.sum(values ** 2))],
                device=self.accelerator.device,
            )
            t = self.accelerator.reduce(t, reduction="sum")  # 1 call, 3 scalars
            n, s, ss = t[0].item(), t[1].item(), t[2].item()
            mean = s / n
            std = max((ss / n - mean ** 2) ** 0.5, 1e-6)
        else:
            mean = float(np.mean(values))
            std = max(float(np.std(values)), 1e-6)
        return mean, std

    # ------------------------------------------------------------------
    # Strategy: weighted sum (default GRPO)
    # ------------------------------------------------------------------

    def compute_weighted_sum(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool,
    ) -> torch.Tensor:
        """Compute advantages using the weighted-sum GRPO strategy.

        This is the standard GRPO advantage computation.  Each reward model's
        scores are multiplied by its configured weight and summed into a single
        aggregated reward per sample.  Advantages are then group-normalised
        (subtract per-group mean, divide by std).

        **Algorithm**:

        1. **Collect** — call :meth:`collect_group_rewards` to obtain
           reward arrays and group assignments.
        2. **Aggregate** — compute
           ``r_agg[i] = sum_k(reward_k[i] * weight_k)`` for each sample.
        3. **Group-normalise** — for each group *g*:
           ``advantage[i] = (r_agg[i] - mean(r_agg[g])) / std``
           where *std* is either the global std across all samples (when
           ``global_std=True``) or the per-group std (when ``global_std=False``).
        4. **To-local** — convert back to local-rank tensor via
           :meth:`_to_local`.
        5. **Store** — optionally write advantages into each sample's
           ``extra_kwargs['advantage']``.

        Parameters
        ----------
        samples : list[BaseSample]
            Samples on the current rank.
        rewards : dict[str, Tensor]
            Per-reward-model reward tensors aligned with *samples*.
        store_to_samples : bool
            If ``True``, write the computed advantage into each sample's
            ``extra_kwargs['advantage']`` field.

        Returns
        -------
        torch.Tensor
            Advantages for the local rank, shape ``(len(samples),)``.
        """
        gathered_rewards, group_indices = self.collect_group_rewards(
            samples, rewards
        )

        # Aggregate rewards with weights
        aggregated_rewards = np.zeros_like(
            next(iter(gathered_rewards.values())), dtype=np.float64
        )
        for key, reward_array in gathered_rewards.items():
            aggregated_rewards += reward_array * self.reward_weights[key]

        # Group-normalise
        _unique_ids, _counts = np.unique(group_indices, return_counts=True)
        advantages = np.zeros_like(aggregated_rewards, dtype=np.float64)

        if self.global_std:
            _, std = self._global_mean_std(aggregated_rewards)

        for group_id in np.unique(group_indices):
            mask = group_indices == group_id
            group_rewards = aggregated_rewards[mask]
            assert len(group_rewards) == self.group_size, (
                f"Group size mismatch: expected {self.group_size}, got {len(group_rewards)}"
            )
            mean = np.mean(group_rewards, axis=0, keepdims=True)
            if not self.global_std:
                std = max(np.std(group_rewards, axis=0, keepdims=True), 1e-6)
            advantages[mask] = (group_rewards - mean) / std

        self._pending_advantage_metrics = self._build_weighted_sum_log_data(
            gathered_rewards, group_indices, aggregated_rewards, advantages, samples
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
    ) -> torch.Tensor:
        """Compute advantages using the GDPO (Group-wise DPO) strategy.

        Unlike :meth:`compute_weighted_sum`, which first aggregates all
        rewards into a single scalar then normalises, GDPO normalises each
        reward **independently** within its group before combining.  This
        prevents a single high-variance reward from dominating the advantage
        signal.

        **Algorithm**:

        1. **Collect** — call :meth:`collect_group_rewards` to obtain
           reward arrays and group assignments.
        2. **Per-reward group normalisation** — for each reward *k* and
           each group *g*:
           ``norm_k[i] = (reward_k[i] - mean(reward_k[g])) / std(reward_k[g])``
           then scale by the reward weight:
           ``adv_k[i] = norm_k[i] * weight_k``.
        3. **Combine** — sum per-reward advantages:
           ``combined[i] = sum_k(adv_k[i])``.
        4. **Batch normalisation** — compute global mean and std of the
           combined advantages and normalise:
           ``advantage[i] = (combined[i] - global_mean) / global_std``.
        5. **To-local** — convert back to local-rank tensor via
           :meth:`_to_local`.
        6. **Store** — optionally write advantages into each sample's
           ``extra_kwargs['advantage']``.

        Parameters
        ----------
        samples : list[BaseSample]
            Samples on the current rank.
        rewards : dict[str, Tensor]
            Per-reward-model reward tensors aligned with *samples*.
        store_to_samples : bool
            If ``True``, write the computed advantage into each sample's
            ``extra_kwargs['advantage']`` field.

        Returns
        -------
        torch.Tensor
            Advantages for the local rank, shape ``(len(samples),)``.
        """
        gathered_rewards, group_indices = self.collect_group_rewards(
            samples, rewards
        )

        # Per-reward group-wise normalisation
        all_reward_advantages = []
        for key, reward_array in gathered_rewards.items():
            reward_adv = np.zeros_like(reward_array, dtype=np.float64)
            for group_id in np.unique(group_indices):
                mask = group_indices == group_id
                group_rewards = reward_array[mask]
                mean = np.mean(group_rewards)
                std = max(np.std(group_rewards), 1e-6)
                reward_adv[mask] = (group_rewards - mean) / std
            all_reward_advantages.append(reward_adv * self.reward_weights[key])

        # Combine and batch normalise
        combined_advantages = np.sum(all_reward_advantages, axis=0)
        bn_mean, bn_std = self._global_mean_std(combined_advantages)
        advantages = (combined_advantages - bn_mean) / bn_std

        self._pending_advantage_metrics = self._build_gdpo_log_data(
            gathered_rewards, group_indices, advantages, bn_mean, bn_std, samples
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

    def _build_weighted_sum_log_data(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        aggregated_rewards: np.ndarray,
        advantages: np.ndarray,
        samples: List[BaseSample],
    ) -> Dict[str, Any]:
        _log_data: Dict[str, Any] = {}
        # Per-reward mean / std
        for key, value in gathered_rewards.items():
            _log_data[f"train/reward_{key}_mean"] = np.mean(value)
            _log_data[f"train/reward_{key}_std"] = np.std(value)
        # Per-reward group stats
        for key, reward_array in gathered_rewards.items():
            g_means, g_stds = RewardProcessor.compute_group_reward_stats(
                reward_array, group_indices
            )
            _log_data.update(
                {
                    f"train/reward_{key}_group_std_mean": float(np.mean(g_stds)),
                    f"train/reward_{key}_group_std_max": float(np.max(g_stds)),
                    f"train/reward_{key}_group_std_min": float(np.min(g_stds)),
                    f"train/reward_{key}_group_mean_std": float(np.std(g_means)),
                }
            )
        # Aggregated reward stats
        zero_std_ratio = RewardProcessor.compute_group_zero_std_ratio(
            aggregated_rewards, group_indices
        )
        _log_data["train/reward_zero_std_ratio"] = zero_std_ratio
        _log_data["train/reward_mean"] = np.mean(aggregated_rewards)
        _log_data["train/reward_std"] = np.std(aggregated_rewards)
        g_means, g_stds = RewardProcessor.compute_group_reward_stats(
            aggregated_rewards, group_indices
        )
        _log_data.update(
            {
                "train/reward_group_std_mean": float(np.mean(g_stds)),
                "train/reward_group_std_max": float(np.max(g_stds)),
                "train/reward_group_mean_std": float(np.std(g_means)),
            }
        )
        # Advantage stats
        _log_data.update(
            {
                "train/adv_max": np.max(advantages),
                "train/adv_min": np.min(advantages),
                "train/adv_abs_mean": np.mean(np.abs(advantages)),
            }
        )
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
    ) -> Dict[str, Any]:
        _log_data: Dict[str, Any] = {}
        # Per-reward mean / std
        for key, value in gathered_rewards.items():
            _log_data[f"train/reward_{key}_mean"] = np.mean(value)
            _log_data[f"train/reward_{key}_std"] = np.std(value)
        # Per-reward zero std ratio
        for key, arr in gathered_rewards.items():
            _log_data[f"train/reward_{key}_zero_std_ratio"] = (
                RewardProcessor.compute_group_zero_std_ratio(arr, group_indices)
            )
        # Per-reward group stats
        for key, reward_array in gathered_rewards.items():
            g_means, g_stds = RewardProcessor.compute_group_reward_stats(
                reward_array, group_indices
            )
            _log_data.update(
                {
                    f"train/reward_{key}_group_std_mean": float(np.mean(g_stds)),
                    f"train/reward_{key}_group_std_max": float(np.max(g_stds)),
                    f"train/reward_{key}_group_std_min": float(np.min(g_stds)),
                    f"train/reward_{key}_group_mean_std": float(np.std(g_means)),
                }
            )
        # Combined stats
        _log_data.update(
            {
                "train/batch_norm_mean": bn_mean,
                "train/batch_norm_std": bn_std,
                "train/adv_max": np.max(advantages),
                "train/adv_min": np.min(advantages),
                "train/adv_abs_mean": np.mean(np.abs(advantages)),
                "train_samples": samples[:30],
            }
        )
        return _log_data
