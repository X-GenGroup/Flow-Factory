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
import threading
from queue import Queue
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
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards for all PointwiseRewardModels."""
        results: Dict[str, torch.Tensor] = {}
        
        for name, model in self._pointwise_models.items():
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
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rewards for all GroupwiseRewardModels with distributed workload.
        
        Each rank computes a subset of groups (stride distribution), then results
        are aggregated via all_reduce to restore the complete reward tensor.
        """
        device = self.accelerator.device
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes
        
        # 1. Collect required fields from all groupwise models
        required_fields: Set[str] = set()
        for model in self._groupwise_models.values():
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
        
        for name, model in self._groupwise_models.items():
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
    Unified reward computation buffer.

    When ``async_reward=True`` (async mode):
        - Pointwise models: triggered when pending samples >= model batch_size.
        - Groupwise models: triggered when a complete group arrives.
        - Reward computation runs on a background thread with a dedicated
          CUDA stream, so the main sampling loop is not blocked.

    When ``async_reward=False`` (sync mode):
        - Samples are accumulated without any reward computation.
        - ``finalize()`` delegates to ``RewardProcessor.compute_rewards()``
          to compute all rewards at once (identical to the original flow).

    Usage (inside trainer.sample()):
        buffer = RewardBuffer(reward_processor, group_size, async_reward)
        for batch in dataloader:
            new_samples = adapter.inference(...)
            buffer.add_samples(new_samples)
        rewards = buffer.finalize()
    """

    _SHUTDOWN = None

    def __init__(self, reward_processor: RewardProcessor, group_size: int,
                 async_reward: bool = False):
        self.rp = reward_processor
        self.group_size = group_size
        self.async_reward = async_reward
        self.all_samples: List[BaseSample] = []

        if self.async_reward:
            self._rewards: Dict[str, List[Optional[torch.Tensor]]] = {
                name: [] for name in reward_processor.reward_models
            }
            self._pw_pending: List[int] = []
            self._gw_pending: Dict[int, List[int]] = defaultdict(list)
            self._any_cuda_reward = any(
                m.device.type == 'cuda' for m in reward_processor.reward_models.values()
            )
            self._task_queue: Queue = Queue()
            self._worker_error: Optional[BaseException] = None
            self._worker_started = False

    # ---- Main thread API ----

    def add_samples(self, samples: List[BaseSample]) -> None:
        """Add samples. In async mode, enqueues ready reward tasks (non-blocking)."""
        self.all_samples.extend(samples)
        if not self.async_reward:
            return
        if not self._worker_started:
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()
            self._worker_started = True
        self._check_worker_error()
        start = len(self.all_samples) - len(samples)
        new_indices = list(range(start, start + len(samples)))
        for name in self._rewards:
            self._rewards[name].extend([None] * len(samples))
        self._pw_pending.extend(new_indices)
        for idx, s in zip(new_indices, samples):
            self._gw_pending[s.unique_id].append(idx)
        sync_event = None
        if self._any_cuda_reward:
            sync_event = torch.cuda.Event()
            sync_event.record()
        self._enqueue_ready_tasks(sync_event)

    def finalize(
        self,
        store_to_samples: bool = True,
        split: Literal['pointwise', 'groupwise', 'all'] = 'all',
    ) -> Dict[str, torch.Tensor]:
        """Compute / collect all rewards and return the result dict."""
        if not self.async_reward:
            return self.rp.compute_rewards(
                self.all_samples, store_to_samples=store_to_samples, split=split,
            )
        return self._finalize_async(store_to_samples=store_to_samples)

    # ---- Async internals ----

    def _finalize_async(self, store_to_samples: bool = True) -> Dict[str, torch.Tensor]:
        """Flush remaining async tasks, join worker, build result."""
        if not self._worker_started:
            return {}
        sync_event = None
        if self._any_cuda_reward:
            sync_event = torch.cuda.Event()
            sync_event.record()
        for name, model in self.rp._pointwise_models.items():
            if self._pw_pending:
                self._task_queue.put((
                    'pointwise', name, model,
                    list(self._pw_pending),
                    [self.all_samples[i] for i in self._pw_pending],
                    sync_event,
                ))
        self._pw_pending = []
        self._task_queue.put(self._SHUTDOWN)
        self._worker.join()
        self._check_worker_error()
        assert len(self._gw_pending) == 0, (
            f"Incomplete groups remaining: {list(self._gw_pending.keys())}"
        )
        results: Dict[str, torch.Tensor] = {}
        for name, reward_list in self._rewards.items():
            assert all(r is not None for r in reward_list), (
                f"Missing rewards for model '{name}'"
            )
            results[name] = torch.stack(reward_list)
        if store_to_samples:
            for i, sample in enumerate(self.all_samples):
                sample.extra_kwargs['rewards'] = {k: v[i] for k, v in results.items()}
        return results

    def _enqueue_ready_tasks(self, sync_event) -> None:
        for name, model in self.rp._pointwise_models.items():
            bs = self.rp._resolve_batch_size(name, model)
            while len(self._pw_pending) >= bs:
                batch_idx = self._pw_pending[:bs]
                self._pw_pending = self._pw_pending[bs:]
                batch_samples = [self.all_samples[i] for i in batch_idx]
                self._task_queue.put((
                    'pointwise', name, model, batch_idx, batch_samples, sync_event,
                ))
        for uid, indices in list(self._gw_pending.items()):
            if len(indices) >= self.group_size:
                group_samples = [self.all_samples[i] for i in indices]
                for name, model in self.rp._groupwise_models.items():
                    self._task_queue.put((
                        'groupwise', name, model, list(indices), group_samples, sync_event,
                    ))
                del self._gw_pending[uid]

    # ---- Worker thread ----

    def _worker_loop(self) -> None:
        try:
            reward_streams: Dict[torch.device, torch.cuda.Stream] = {}
            for m in self.rp.reward_models.values():
                if m.device.type == 'cuda' and m.device not in reward_streams:
                    reward_streams[m.device] = torch.cuda.Stream(device=m.device)

            while True:
                task = self._task_queue.get()
                if task is self._SHUTDOWN:
                    break
                task_type, name, model, indices, samples, sync_event = task
                stream = reward_streams.get(model.device)
                ctx = torch.cuda.stream(stream) if stream else nullcontext()
                with ctx:
                    if sync_event is not None and stream is not None:
                        stream.wait_event(sync_event)
                    if task_type == 'pointwise':
                        rewards = self.rp._compute_pointwise_batch(name, model, samples)
                    else:
                        rewards = self.rp._compute_groupwise_group(name, model, samples)
                    for i, idx in enumerate(indices):
                        self._rewards[name][idx] = rewards[i]

            for stream in reward_streams.values():
                stream.synchronize()
        except Exception as e:
            self._worker_error = e

    def _check_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError(
                "RewardBuffer worker thread failed"
            ) from self._worker_error
