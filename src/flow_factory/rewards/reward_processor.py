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
from typing import Dict, Any, Optional, List, Tuple, Set, Union, Literal
from collections import defaultdict
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, Future
import torch
import numpy as np
from tqdm import tqdm

from accelerate import Accelerator

from .abc import (
    BaseRewardModel,
    PointwiseRewardModel,
    GroupwiseRewardModel,
    RewardModelOutput,
)
from ..hparams import RewardArguments
from ..samples import BaseSample
from ..utils.dist import gather_samples
from ..utils.base import filter_kwargs
from ..utils.image import standardize_image_batch
from ..utils.video import standardize_video_batch

# ============================ Reward Processor ============================
class RewardProcessor:
    """
    Unified reward processor bound to specific reward models.
    
    Handles both PointwiseRewardModel and GroupwiseRewardModel seamlessly.
    """
    MEDIA_FIELDS = {'image', 'video', 'condition_images', 'condition_videos'} # Fields that may contain media data, requiring format conversion

    def __init__(
        self,
        accelerator: Accelerator,
        reward_models: Dict[str, BaseRewardModel],
        reward_configs: Optional[Dict[str, RewardArguments]] = None,
        tokenizer: Optional[Any] = None,
        verbose: bool = True,
    ):
        self.accelerator = accelerator
        self.reward_models = reward_models
        self.reward_configs = reward_configs or {}
        self.tokenizer = tokenizer
        self.verbose = verbose
        
        # Pre-categorize models by type
        self._pointwise_models : Dict[str, PointwiseRewardModel] = {
            k: v for k, v in reward_models.items()
            if isinstance(v, PointwiseRewardModel)
        }
        self._groupwise_models : Dict[str, GroupwiseRewardModel] = {
            k: v for k, v in reward_models.items()
            if isinstance(v, GroupwiseRewardModel)
        }

    @property
    def show_progress_bar(self) -> bool:
        """Whether to show tqdm progress bars."""
        return self.verbose and self.accelerator.is_local_main_process

    def _is_async_reward(self, name: str) -> bool:
        """Check if a named reward model is configured for async computation."""
        config = self.reward_configs.get(name)
        return getattr(config, 'async_reward', False) if config else False

    def _resolve_num_workers(self, name: str) -> int:
        """Resolve the number of concurrent workers for an async reward model."""
        config = self.reward_configs.get(name)
        return max(1, getattr(config, 'num_workers', 1)) if config else 1

    def _resolve_batch_size(self, name: str, model: BaseRewardModel) -> int:
        """
        Resolve runtime batch size for a pointwise reward model.
        
        Priority:
            1) Explicit config in `self.reward_configs` for this reward name.
            2) Fallback to shared model config (`model.config.batch_size`).
        """
        batch_size = None
        if name in self.reward_configs:
            batch_size = getattr(self.reward_configs[name], 'batch_size', None)
        if batch_size is None:
            batch_size = getattr(model.config, 'batch_size', None)

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"Invalid batch_size for reward '{name}': {batch_size}. "
                "batch_size must be a positive integer."
            )

        return batch_size

    # ============================ Media Format Conversion ============================
    def _convert_media_format(self, batch_input: Dict[str, Any], model: BaseRewardModel) -> Dict[str, Any]:
        """Convert tensor media fields to PIL format (unless model opts out)."""
        if getattr(model, 'use_tensor_inputs', False):
            output_type = 'pt'
        else:
            output_type = 'pil'
        
        result = {}
        for k, v in batch_input.items():
            if k not in self.MEDIA_FIELDS or v is None:
                result[k] = v
                continue
            if k == 'image':
                result[k] = standardize_image_batch(v, output_type=output_type)
            elif k == 'video':
                result[k] = standardize_video_batch(v, output_type=output_type)
            elif k == 'condition_images':
                result[k] = [
                    standardize_image_batch(imgs, output_type=output_type)
                    for imgs in v
                ]
            elif k == 'condition_videos':
                result[k] = [
                    standardize_video_batch(videos, output_type=output_type)
                    for videos in v
                ]

        return result
    
    # ============================ Single-batch / Single-group Helpers ============================
    def _compute_pointwise_batch(
        self, name: str, model: PointwiseRewardModel, batch_samples: List[BaseSample]
    ) -> torch.Tensor:
        """Compute pointwise rewards for a single batch. Returns (batch_size,) tensor."""
        filtered_fields = filter_kwargs(model.__call__, **batch_samples[0])
        batch_input: Dict[str, List[Any]] = {
            k: [getattr(s, k) for s in batch_samples]
            for k in filtered_fields
            if all(getattr(s, k) is not None for s in batch_samples)
        }
        batch_input = self._convert_media_format(batch_input, model)
        output = model(**batch_input)
        return torch.as_tensor(
            output.rewards if hasattr(output, 'rewards') else output,
            device='cpu', dtype=torch.float32,
        )

    def _compute_groupwise_group(
        self, name: str, model: GroupwiseRewardModel, group_samples: List[BaseSample]
    ) -> torch.Tensor:
        """Compute groupwise rewards for one complete group. Returns (group_size,) tensor."""
        fields = filter_kwargs(model.__call__, **group_samples[0])
        group_input: Dict[str, List[Any]] = {
            k: [getattr(s, k) for s in group_samples]
            for k in fields
            if all(getattr(s, k) is not None for s in group_samples)
        }
        group_input = self._convert_media_format(group_input, model)
        output = model(**group_input)
        return torch.as_tensor(
            output.rewards if hasattr(output, 'rewards') else output,
            device='cpu', dtype=torch.float32,
        )

    # ============================ Public API ============================
    def compute_rewards(
        self,
        samples: List[BaseSample],
        store_to_samples: bool = True,
        epoch: Optional[int] = None,
        split: Literal['pointwise', 'groupwise', 'all'] = 'all',
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
        if split in ('pointwise', 'all') and self._pointwise_models:
            results.update(self._compute_pointwise_rewards(samples, epoch))
        
        # Groupwise: gather -> compute -> scatter
        if split in ('groupwise', 'all') and self._groupwise_models:
            results.update(self._compute_groupwise_rewards(samples, epoch))

        self.accelerator.wait_for_everyone()
        # Store to samples
        if store_to_samples:
            for i, sample in enumerate(samples):
                sample.extra_kwargs['rewards'] = {
                    k: v[i] for k, v in results.items()
                }
        
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

            desc = f'Epoch {epoch} Pointwise Rewards: {name}' if epoch is not None else f'Pointwise Rewards: {name}'
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
        Compute rewards for GroupwiseRewardModels with distributed workload.
        
        Each rank computes a subset of groups (stride distribution), then results
        are aggregated via all_reduce to restore the complete reward tensor.
        """
        models = models if models is not None else self._groupwise_models
        device = self.accelerator.device
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes
        
        # 1. Collect required fields from all groupwise models
        required_fields: Set[str] = set()
        for model in models.values():
            required_fields.update(model.required_fields)
        
        # Optimize: use prompt_ids instead of prompt strings for communication
        needs_decode = False
        if 'prompt' in required_fields:
            if hasattr(samples[0], 'prompt_ids') and samples[0].prompt_ids is not None:
                required_fields.discard('prompt')
                required_fields.add('prompt_ids')
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
        
        # 3. Group by unique_id and build inverse mapping
        groups, inverse = self.group_samples(gathered, key='unique_id', return_inverse=True)
        group_keys = list(groups.keys())
        num_gathered = len(gathered)
        
        # 4. Stride distribution: rank i handles groups [i, i+W, i+2W, ...]
        local_group_indices = list(range(rank, len(group_keys), world_size))
        
        # 5. Compute rewards per model
        results: Dict[str, torch.Tensor] = {}
        
        for name, model in models.items():
            # Initialize with zeros - only fill positions this rank computes
            all_rewards = torch.zeros(num_gathered, dtype=torch.float32, device=device)
            desc = f'Epoch {epoch} Groupwise Rewards: {name}' if epoch is not None else f'Groupwise Rewards: {name}'
            pbar = tqdm(
                local_group_indices,
                desc=desc,
                disable=not self.show_progress_bar,
            )
            for group_idx in pbar:
                uid = group_keys[group_idx]
                group_list = groups[uid]
                
                # Prepare group input
                fields = filter_kwargs(model.__call__, **group_list[0])
                group_input = {
                    k: [getattr(s, k) for s in group_list]
                    for k in fields
                    if all(getattr(s, k) is not None for s in group_list)
                }
                group_input = self._convert_media_format(group_input, model)
                
                # Compute rewards
                output = model(**group_input)
                group_rewards = torch.as_tensor(
                    output.rewards if hasattr(output, 'rewards') else output,
                    device=device, dtype=torch.float32,
                )
                
                # Fill positions belonging to this group
                mask = (inverse == group_idx)
                all_rewards[mask] = group_rewards
            
            # 6. All-reduce SUM: each position has value from exactly one rank
            all_rewards = self.accelerator.reduce(all_rewards, reduction='sum')
            results[name] = all_rewards.cpu()
        
        # 7. Scatter back to local rank
        results = {
            k: v.chunk(world_size)[rank]
            for k, v in results.items()
        }
        
        return results

    # ============================ Prompt Encoding/Decoding ============================
    def _decode_prompts(self, prompt_ids_list: List[torch.Tensor]) -> List[str]:
        """Decode prompt_ids to strings."""
        if self.tokenizer is None:
            raise ValueError("Cannot decode prompts: tokenizer not provided")
        
        return [
            self.tokenizer.decode(
                ids.cpu().tolist() if isinstance(ids, torch.Tensor) else ids,
                skip_special_tokens=True
            )
            for ids in prompt_ids_list
        ]

    def _encode_prompts(self, prompts: List[str]) -> List[torch.Tensor]:
        """Encode strings to prompt_ids."""
        if self.tokenizer is None:
            raise ValueError("Cannot encode prompts: tokenizer not provided")
        
        return [
            self.tokenizer(text, return_tensors='pt', padding=False, truncation=True)
            .input_ids.squeeze(0)
            for text in prompts
        ]
    
    # ============================ Helper Functions ============================
    @staticmethod
    def compute_group_zero_std_ratio(
        rewards: np.ndarray, 
        group_indices: np.ndarray, 
        eps: float = 1e-6
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
            1 for gid in unique_groups 
            if np.std(rewards[group_indices == gid]) < eps
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
        group_stds  = np.array([np.std(rewards[group_indices == gid])  for gid in unique_groups])
        group_means = np.array([np.mean(rewards[group_indices == gid]) for gid in unique_groups])
        return group_means, group_stds

    @staticmethod
    def group_samples(
        samples: List[BaseSample],
        key: str = 'unique_id',
        return_inverse: bool = False,
    ) -> Union[Dict[Any, List[BaseSample]], Tuple[Dict[Any, List[BaseSample]], np.ndarray]]:
        """
        Group samples by a key field, similar to np.unique.
        
        Args:
            samples: List of BaseSample instances
            key: Field name to group by (default: 'unique_id')
            return_inverse: If True, return indices to reconstruct original order
            return_index: If True, return first occurrence index for each group
        
        Returns:
            groups: Dict mapping key_value -> List[BaseSample]
            inverse: (optional) Array where inverse[i] gives group index for samples[i]
            index: (optional) Array of first occurrence indices for each unique key
        """
        keys = np.array([getattr(s, key) for s in samples])
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        
        groups: Dict[Any, List[BaseSample]] = {k: [] for k in unique_keys}
        for sample, k in zip(samples, keys):
            groups[k].append(sample)
        
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
        self._async_pointwise = {n: m for n, m in self.rp._pointwise_models.items() if self.rp._is_async_reward(n)}
        self._sync_pointwise  = {n: m for n, m in self.rp._pointwise_models.items() if not self.rp._is_async_reward(n)}
        self._async_groupwise = {n: m for n, m in self.rp._groupwise_models.items() if self.rp._is_async_reward(n)}
        self._sync_groupwise  = {n: m for n, m in self.rp._groupwise_models.items() if not self.rp._is_async_reward(n)}
        self._has_async = bool(self._async_pointwise or self._async_groupwise)

        # Pre-create one CUDA stream per unique device among async models
        # (used inside _execute_task for CUDA-based reward models).
        self._reward_streams: Dict[torch.device, torch.cuda.Stream] = {}
        if self._has_async:
            for m in list(self._async_pointwise.values()) + list(self._async_groupwise.values()):
                if m.device.type == 'cuda' and m.device not in self._reward_streams:
                    self._reward_streams[m.device] = torch.cuda.Stream(device=m.device)

        self._init_async_state()

    def _init_async_state(self) -> None:
        """Initialize (or reset) all mutable state used by the async path.

        Sets up:
        - ``_rewards``: per-model list of reward scalars (None until filled by futures).
        - ``_pointwise_pending``: per-model list of sample indices awaiting batch dispatch.
        - ``_groupwise_pending``: maps unique_id -> list of sample indices; dispatched
          when a group reaches ``group_size``.
        - ``_executor``: ``ThreadPoolExecutor`` whose pool size is the sum of all
          async models' ``num_workers``.
        - ``_futures``: list of ``(name, indices, Future)`` tuples for result collection.
        """
        if not self._has_async:
            return
        async_names = list(self._async_pointwise) + list(self._async_groupwise)
        self._rewards: Dict[str, List[Optional[torch.Tensor]]] = {n: [] for n in async_names}
        self._pointwise_pending: Dict[str, List[int]] = {n: [] for n in self._async_pointwise}
        self._groupwise_pending: Dict[int, List[int]] = defaultdict(list)
        self._any_cuda_reward = bool(self._reward_streams)
        total_workers = sum(
            self.rp._resolve_num_workers(n)
            for n in async_names
        )
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max(1, total_workers))
        self._futures: List[Tuple[str, List[int], Future]] = []

    # ---- Main thread API ----

    def clear(self) -> None:
        """Reset buffer to initial state for reuse across epochs.

        Shuts down the thread pool (waiting for in-flight tasks) and
        reinitializes all async tracking structures.
        """
        self.all_samples = []
        if self._has_async:
            self._executor.shutdown(wait=True)
            self._init_async_state()

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
                self._groupwise_pending[s.unique_id].append(idx)
        # Record CUDA event so pool workers can wait for sample data readiness
        sync_event = None
        if self._any_cuda_reward:
            sync_event = torch.cuda.Event()
            sync_event.record()
        self._submit_ready_tasks(sync_event)

    def finalize(
        self,
        store_to_samples: bool = True,
        split: Literal['pointwise', 'groupwise', 'all'] = 'all',
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
        if split in ('pointwise', 'all') and self._sync_pointwise:
            results.update(self.rp._compute_pointwise_rewards(self.all_samples, models=self._sync_pointwise))
        if split in ('groupwise', 'all') and self._sync_groupwise:
            results.update(self.rp._compute_groupwise_rewards(self.all_samples, models=self._sync_groupwise))

        # 2. Flush and collect async rewards
        if self._has_async:
            async_results = self._finalize_async()
            results.update(async_results)

        self.rp.accelerator.wait_for_everyone()

        # 3. Store to samples
        if store_to_samples:
            for i, sample in enumerate(self.all_samples):
                sample.extra_kwargs['rewards'] = {k: v[i] for k, v in results.items()}

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
            if task_type == 'pointwise':
                return self.rp._compute_pointwise_batch(name, model, samples)
            else:
                return self.rp._compute_groupwise_group(name, model, samples)

    def _submit_ready_tasks(self, sync_event) -> None:
        """Check pending lists and submit tasks that meet their trigger condition.

        - Pointwise: submitted when a model's pending count >= its batch_size.
          Each model has its own pending list so different batch_sizes are
          handled independently.
        - Groupwise: submitted when a group (identified by unique_id) accumulates
          ``group_size`` samples. One task is created per async groupwise model
          for the completed group.

        Each task is submitted to the ``ThreadPoolExecutor`` and the resulting
        ``Future`` is stored in ``_futures`` for collection in ``_finalize_async``.
        """
        # Pointwise: dispatch full batches per model
        for name, model in self._async_pointwise.items():
            bs = self.rp._resolve_batch_size(name, model)
            pending = self._pointwise_pending[name]
            while len(pending) >= bs:
                batch_idx = pending[:bs]
                self._pointwise_pending[name] = pending[bs:]
                pending = self._pointwise_pending[name]
                batch_samples = [self.all_samples[i] for i in batch_idx]
                future = self._executor.submit(
                    self._execute_task, 'pointwise', name, model, batch_samples, sync_event,
                )
                self._futures.append((name, batch_idx, future))
        # Groupwise: dispatch complete groups
        for uid, indices in list(self._groupwise_pending.items()):
            if len(indices) >= self.group_size:
                group_samples = [self.all_samples[i] for i in indices]
                for name, model in self._async_groupwise.items():
                    future = self._executor.submit(
                        self._execute_task, 'groupwise', name, model, group_samples, sync_event,
                    )
                    self._futures.append((name, list(indices), future))
                del self._groupwise_pending[uid]

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
        if not self._futures and not any(self._pointwise_pending.values()):
            return {}
        # 1. Flush remaining pointwise pending (tail < batch_size)
        sync_event = None
        if self._any_cuda_reward:
            sync_event = torch.cuda.Event()
            sync_event.record()
        for name, model in self._async_pointwise.items():
            pending = self._pointwise_pending.get(name, [])
            if pending:
                batch_samples = [self.all_samples[i] for i in pending]
                future = self._executor.submit(
                    self._execute_task, 'pointwise', name, model, batch_samples, sync_event,
                )
                self._futures.append((name, list(pending), future))
                self._pointwise_pending[name] = []
        # 2. Collect all futures with progress bar
        num_async = len(self._async_pointwise) + len(self._async_groupwise)
        total = len(self.all_samples) * num_async
        completed = 0
        with tqdm(
            total=total,
            desc='Async Rewards',
            disable=not self.rp.show_progress_bar,
        ) as pbar:
            for name, indices, future in self._futures:
                rewards = future.result()
                for i, idx in enumerate(indices):
                    self._rewards[name][idx] = rewards[i]
                completed += len(indices)
                pbar.n = completed
                pbar.refresh()
        # 3. Verify all groupwise groups completed
        assert len(self._groupwise_pending) == 0, (
            f"Incomplete groups remaining: {list(self._groupwise_pending.keys())}"
        )
        # 4. Synchronize CUDA streams
        for stream in self._reward_streams.values():
            stream.synchronize()
        # 5. Assemble results
        results: Dict[str, torch.Tensor] = {}
        for name, reward_list in self._rewards.items():
            assert all(r is not None for r in reward_list), (
                f"Missing rewards for async model '{name}'"
            )
            results[name] = torch.stack(reward_list)
        return results
