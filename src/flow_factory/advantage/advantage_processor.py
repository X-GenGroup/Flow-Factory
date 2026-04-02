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
from typing import List, Dict, Optional, Union, Literal, Callable
import time
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
    log_func : callable, optional
        Logging callback (typically ``trainer.log_data``).
    verbose : bool
        Whether to emit progress information.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        reward_weights: Dict[str, float],
        group_size: int,
        global_std: bool = True,
        sampler_type: str = "distributed_k_repeat",
        log_func: Optional[Callable] = None,
        verbose: bool = True,
    ):
        self.accelerator = accelerator
        self.reward_weights = reward_weights
        self.group_size = group_size
        self.global_std = global_std
        self.sampler_type = sampler_type
        self.log_func = log_func
        self.verbose = verbose

        self.group_on_same_rank = sampler_type == "group_contiguous"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func: Optional[Union[Literal["sum", "gdpo"], Callable]] = None,
        step: int = 0,
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
        step : int
            Current training step (used for logging).

        Returns
        -------
        Tensor  – advantages for the local rank, shape ``(len(samples),)``.
        """
        aggregation_func = aggregation_func or "gdpo"
        if aggregation_func == "sum":
            fn = self._compute_weighted_sum
        elif aggregation_func == "gdpo":
            fn = self._compute_gdpo
        elif callable(aggregation_func):
            return aggregation_func(self, samples, rewards, store_to_samples)
        else:
            raise ValueError(
                f"Unsupported advantage aggregation method: {aggregation_func}. "
                "Supported: ['sum', 'gdpo'] "
                "or a callable function that takes (processor, samples, rewards, store_to_samples) as inputs."
            )

        # Timed execution
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        result = fn(samples, rewards, store_to_samples, step)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        if self.log_func is not None:
            self.log_func(
                {
                    "train/advantage_compute_ms": elapsed_ms,
                    "train/advantage_sampler_type": 1.0 if self.group_on_same_rank else 0.0,
                },
                step=step,
            )
        logger.info(
            f"[AdvantageProcessor] compute_advantages ({self.sampler_type}): "
            f"{elapsed_ms:.1f} ms | samples={len(samples)} | strategy={aggregation_func}"
        )
        return result

    # ------------------------------------------------------------------
    # Communication layer
    # ------------------------------------------------------------------

    def _gather_rewards_and_ids(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
    ):
        """Gather rewards and unique_ids, respecting the sampler topology.

        Returns
        -------
        gathered_rewards : dict[str, np.ndarray]
        group_indices : np.ndarray
        needs_scatter : bool
            ``True`` when the returned arrays span all ranks and must be
            scattered back; ``False`` when they are already local.
        """
        rewards = {
            key: torch.as_tensor(value).to(self.accelerator.device)
            for key, value in rewards.items()
        }

        if self.group_on_same_rank:
            # group_contiguous: all K copies on same rank — no communication
            gathered_rewards = {
                key: value.cpu().numpy() for key, value in rewards.items()
            }
            unique_ids = np.array([s.unique_id for s in samples], dtype=np.int64)
            _unique_ids, group_indices = np.unique(unique_ids, return_inverse=True)
            return gathered_rewards, group_indices, False
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

            gathered_rewards = {
                key: gathered[:, i] for i, key in enumerate(reward_keys)
            }
            gathered_ids = gathered[:, -1].astype(np.int64)
            _unique_ids, group_indices = np.unique(gathered_ids, return_inverse=True)
            return gathered_rewards, group_indices, True

    def _scatter_to_local(
        self,
        advantages: np.ndarray,
        needs_scatter: bool,
    ) -> torch.Tensor:
        """Convert global advantages back to local-rank tensor.

        When ``needs_scatter`` is ``False`` the array is already local and is
        simply converted to a device tensor.
        """
        if needs_scatter:
            advantages = torch.as_tensor(advantages).reshape(
                self.accelerator.num_processes, -1, *advantages.shape[1:]
            )[self.accelerator.process_index].to(self.accelerator.device)
        else:
            advantages = torch.as_tensor(advantages).to(self.accelerator.device)
        return advantages

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

    def _compute_weighted_sum(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool,
        step: int,
    ) -> torch.Tensor:
        gathered_rewards, group_indices, needs_scatter = self._gather_rewards_and_ids(
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

        # Log
        self._log_weighted_sum_stats(
            gathered_rewards, group_indices, aggregated_rewards, advantages, samples, step
        )

        # Scatter & store
        advantages = self._scatter_to_local(advantages, needs_scatter)
        if store_to_samples:
            for sample, adv in zip(samples, advantages):
                sample.extra_kwargs["advantage"] = adv
        return advantages

    # ------------------------------------------------------------------
    # Strategy: GDPO
    # ------------------------------------------------------------------

    def _compute_gdpo(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool,
        step: int,
    ) -> torch.Tensor:
        gathered_rewards, group_indices, needs_scatter = self._gather_rewards_and_ids(
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

        # Log
        self._log_gdpo_stats(
            gathered_rewards, group_indices, advantages, bn_mean, bn_std, samples, step
        )

        # Scatter & store
        advantages = self._scatter_to_local(advantages, needs_scatter)
        if store_to_samples:
            for sample, adv in zip(samples, advantages):
                sample.extra_kwargs["advantage"] = adv
        return advantages

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_weighted_sum_stats(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        aggregated_rewards: np.ndarray,
        advantages: np.ndarray,
        samples: List[BaseSample],
        step: int,
    ) -> None:
        if self.log_func is None:
            return
        _log_data: Dict = {}
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
        self.log_func(_log_data, step=step)

    def _log_gdpo_stats(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        group_indices: np.ndarray,
        advantages: np.ndarray,
        bn_mean: float,
        bn_std: float,
        samples: List[BaseSample],
        step: int,
    ) -> None:
        if self.log_func is None:
            return
        _log_data: Dict = {}
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
        self.log_func(_log_data, step=step)
