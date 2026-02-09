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

# src/flow_factory/utils/trajectory_collector.py
"""
Trajectory & Callback Collectors for Inference

Memory-efficient recording utilities for denoising trajectories.
- ``TrajectoryCollector``: records tensors (latents, log_probs) at specified steps.
- ``CallbackCollector``: records named callback values (noise_level, noise_pred, …)
  at specified steps. Drop-in replacement for the ``defaultdict(list)`` pattern.

Both produce compact storage + lightweight index maps to eliminate redundant
data in multi-GPU gather operations.
"""
from typing import Union, List, Optional, Literal, Set, Dict, Any, TypeVar
from collections import defaultdict
import torch


T = TypeVar('T')
TrajectoryIndicesType = Union[Literal['all'], List[int], None]


# =============================================================================
# TrajectoryCollector — single-sequence tensor recording
# =============================================================================

class TrajectoryCollector:
    """
    Collects tensors at specified indices during denoising trajectory.
    
    Produces compact storage + index map for O(1) position lookup,
    avoiding redundant storage and multi-GPU communication overhead.
    
    Args:
        indices: Controls which steps to record:
            - 'all': Record all steps (default behavior)
            - None: Don't record any steps (returns None)
            - List[int]: Record only at specified indices
                - Supports negative indexing like Python lists
        total_steps: Total number of denoising steps (T)
    
    Examples:
        >>> collector = TrajectoryCollector('all', total_steps=20)
        >>> collector = TrajectoryCollector(None, total_steps=20)     # No recording
        >>> collector = TrajectoryCollector([0, -1], total_steps=20)  # Initial + final only
    
    Usage:
        >>> collector = TrajectoryCollector([0, -1], total_steps=20)
        >>> collector.collect(initial_latents, step_idx=0)
        >>> for i in range(20):
        ...     latents = denoise_step(latents)
        ...     collector.collect(latents, step_idx=i + 1)
        >>> trajectory = collector.get_result()       # [initial, final]
        >>> index_map = collector.get_index_map()     # [0, -1, ..., -1, 1]
    """
    
    def __init__(
        self,
        indices: TrajectoryIndicesType = 'all',
        total_steps: int = 0,
    ):
        self.indices = indices
        self.total_steps = total_steps
        self._collected: List[torch.Tensor] = []
        self._collected_indices: List[int] = []
        
        # Precompute normalized indices for O(1) lookup
        self._target_indices: Optional[Set[int]] = self._normalize_indices()
    
    def _normalize_indices(self) -> Optional[Set[int]]:
        """
        Convert user indices to normalized positive indices.
        
        Returns:
            - set(): Empty set when indices=None (disabled, collect nothing)
            - None: When indices='all' (collect all steps)
            - Set[int]: Specific normalized indices to collect
        """
        if self.indices is None:
            return set()  # Empty set signals "collect nothing"
        if self.indices == 'all':
            return None   # None signals "collect everything"

        # Total positions = total_steps + 1 (initial + each step result)
        total_positions = self.total_steps + 1
        normalized = set()
        
        for idx in self.indices:
            # Handle negative indices
            if idx < 0:
                idx = total_positions + idx
            # Clamp to valid range
            if 0 <= idx < total_positions:
                normalized.add(idx)
        
        return normalized
    
    @property
    def is_disabled(self) -> bool:
        """Check if collection is disabled."""
        return self._target_indices is not None and len(self._target_indices) == 0 # Empty set means "collect nothing"
    
    @property
    def collect_all(self) -> bool:
        """Check if collecting all steps."""
        return self._target_indices is None  # `None` means collect all
    
    def should_collect(self, step_idx: int) -> bool:
        """Check if value should be collected at this step."""
        if self.is_disabled:
            return False
        if self.collect_all:
            return True
        return step_idx in self._target_indices
    
    def collect(self, value: torch.Tensor, step_idx: int) -> None:
        """Conditionally collect tensor at given step."""
        if self.should_collect(step_idx):
            self._collected.append(value)
            self._collected_indices.append(step_idx)
    
    def get_result(self) -> Optional[List[torch.Tensor]]:
        """Get collected tensors, or None if disabled."""
        if self.is_disabled:
            return None
        return self._collected
    
    @property
    def collected_indices(self) -> List[int]:
        """Get list of indices at which values were collected."""
        return self._collected_indices
    
    def get_index_map(self) -> Optional[torch.Tensor]:
        """
        Build dense index map: original_position → compact_index.
        
        Returns a 1D LongTensor of size (total_steps + 1), where entry ``i``
        gives the index into compact ``all_latents`` for original position ``i``,
        or -1 if that position was not collected.
        
        When ``collect_all=True``, returns identity ``[0, 1, ..., T]``.
        Cost is negligible (<1KB for typical step counts).
        
        Returns:
            LongTensor of shape (total_steps + 1), or None if disabled.
        """
        if self.is_disabled:
            return None
        
        total_positions = self.total_steps + 1
        
        if self.collect_all:
            return torch.arange(total_positions, dtype=torch.long)
        
        index_map = torch.full((total_positions,), -1, dtype=torch.long)
        for compact_idx, original_idx in enumerate(self._collected_indices):
            index_map[original_idx] = compact_idx
        
        return index_map
    
    def reset(self) -> None:
        """Clear collected values for reuse."""
        self._collected = []
        self._collected_indices = []
    
    def __len__(self) -> int:
        return len(self._collected)


# =============================================================================
# CallbackCollector — multi-key dict recording with step-gating
# =============================================================================

class CallbackCollector:
    """
    Selectively collects named callback values during denoising.
    
    Drop-in replacement for the repeated ``extra_call_back_res = defaultdict(list)``
    pattern found in every model adapter. Encapsulates:
    
    1. **Step gating** — only records at specified trajectory indices.
    2. **Value resolution** — checks ``capturable`` dict first, then ``output`` attrs.
    3. **Index map** — provides ``get_index_map()`` for O(1) trainer-side lookup.
    
    This eliminates both the duplicated collection boilerplate across adapters
    and the redundant storage/communication of unneeded callback values.
    
    Args:
        indices: Which steps to record ('all', None, or List[int]).
            Uses same convention as TrajectoryCollector but with step indices
            (0 to T-1) rather than position indices (0 to T).
        total_steps: Total number of denoising steps (T).
    
    Usage in model adapter::
    
        # ── Before the loop ──
        callback_collector = create_callback_collector(callback_indices, num_inference_steps)
        
        # ── Inside the loop (replaces 6+ lines of boilerplate) ──
        callback_collector.collect_step(
            step_idx=i,
            output=output,
            keys=extra_call_back_kwargs,
            capturable={'noise_level': current_noise_level},
        )
        
        # ── After the loop ──
        extra_call_back_res = callback_collector.get_result()          # Dict[str, Tensor (B,T',...)]
        callback_index_map  = callback_collector.get_index_map()       # (T,) LongTensor or None
    """
    
    def __init__(
        self,
        indices: TrajectoryIndicesType = 'all',
        total_steps: int = 0,
    ):
        self._gate = TrajectoryCollector(indices=indices, total_steps=total_steps)
        self._data: Dict[str, List] = defaultdict(list)
        self._collected_indices: List[int] = []
        self._collected_set: Set[int] = set()
    
    @property
    def is_disabled(self) -> bool:
        return self._gate.is_disabled
    
    def should_collect(self, step_idx: int) -> bool:
        """Check if callbacks should be collected at this step."""
        return self._gate.should_collect(step_idx)
    
    def collect_step(
        self,
        step_idx: int,
        output: Any,
        keys: List[str],
        capturable: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Collect all requested callback keys for a denoising step.
        
        Resolves each key by checking ``capturable`` dict first, then
        falling back to ``getattr(output, key)``. Skips None values.
        No-op if ``keys`` is empty or this step is not in the target set.
        
        Args:
            step_idx: Current denoising step index (0-based, the loop var ``i``).
            output: Step output object (scheduler output / forward output).
            keys: List of callback keys to capture (e.g., extra_call_back_kwargs).
            capturable: Optional dict of pre-computed values checked before output.
                        Typical: ``{'noise_level': current_noise_level}``.
        """
        if not keys or not self.should_collect(step_idx):
            return
        
        # Track unique step indices (ordered)
        if step_idx not in self._collected_set:
            self._collected_indices.append(step_idx)
            self._collected_set.add(step_idx)
        
        for key in keys:
            val = None
            if capturable and key in capturable and capturable[key] is not None:
                val = capturable[key]
            elif hasattr(output, key):
                val = getattr(output, key)
            if val is not None:
                self._data[key].append(val)
    
    def get_result(self) -> Dict[str, Any]:
        """
        Get collected values, with tensors transposed to batch-first layout.
        
        Tensor lists are stacked along a new dim=1: list of (B, ...) →  (B, T', ...).
        Non-tensor values are returned as plain lists.
        
        Returns:
            Dict mapping key → stacked tensor or list of values.
        """
        result = {}
        for k, v in self._data.items():
            if v and isinstance(v[0], torch.Tensor):
                result[k] = torch.stack(v, dim=1)  # list[(B,...)] → (B, T', ...)
            else:
                result[k] = v
        return result
    
    def get_index_map(self) -> Optional[torch.Tensor]:
        """
        Build dense step index map: step_index → compact_index.
        
        Returns a 1D LongTensor of size ``total_steps``, where entry ``i``
        gives the compact index for denoising step ``i``, or -1 if not collected.
        
        Returns:
            LongTensor of shape (total_steps,), or None if disabled.
        """
        if self.is_disabled:
            return None
        
        total_steps = self._gate.total_steps
        
        if self._gate.collect_all:
            return torch.arange(total_steps, dtype=torch.long)
        
        index_map = torch.full((total_steps,), -1, dtype=torch.long)
        for compact_idx, original_idx in enumerate(self._collected_indices):
            if 0 <= original_idx < total_steps:
                index_map[original_idx] = compact_idx
        
        return index_map
    
    @property
    def collected_indices(self) -> List[int]:
        """Get ordered list of step indices at which values were collected."""
        return self._collected_indices
    
    def reset(self) -> None:
        """Clear all collected data for reuse."""
        self._data = defaultdict(list)
        self._collected_indices = []
        self._collected_set = set()
    
    def __len__(self) -> int:
        """Number of unique steps collected."""
        return len(self._collected_indices)


# =============================================================================
# Utility functions
# =============================================================================

def compute_trajectory_indices(
    train_timestep_indices: Union[List[int], torch.Tensor],
    num_inference_steps: int,
    include_initial: bool = False,
) -> List[int]:
    """
    Compute the minimal set of trajectory positions needed for training.
    
    For each training timestep index ``i``, the trainer needs positions
    ``i`` (current latents) and ``i + 1`` (next latents). Returns the
    deduplicated union. Consecutive training steps share boundaries,
    further reducing the set.
    
    Args:
        train_timestep_indices: Step indices used during training
            (e.g., scheduler.train_timesteps). 0-based trajectory indices.
        num_inference_steps: Total denoising steps T.
        include_initial: Always include position 0 (initial noise).
    
    Returns:
        Sorted list of unique trajectory positions to collect.
    
    Examples:
        >>> compute_trajectory_indices([2, 5, 8], num_inference_steps=20)
        [0, 2, 3, 5, 6, 8, 9]   # 7 positions instead of 21
        
        >>> compute_trajectory_indices([0, 1, 2], num_inference_steps=20)
        [0, 1, 2, 3]            # Consecutive steps share boundaries
    """
    if isinstance(train_timestep_indices, torch.Tensor):
        train_timestep_indices = train_timestep_indices.tolist()
    
    total_positions = num_inference_steps + 1
    positions = set()
    
    if include_initial:
        positions.add(0)
    
    for idx in train_timestep_indices:
        if 0 <= idx < total_positions:
            positions.add(idx)
        if 0 <= idx + 1 < total_positions:
            positions.add(idx + 1)
    
    return sorted(positions)


def create_trajectory_collector(
    indices: TrajectoryIndicesType,
    num_steps: int,
) -> TrajectoryCollector:
    """Factory function to create a TrajectoryCollector."""
    return TrajectoryCollector(indices=indices, total_steps=num_steps)


def create_callback_collector(
    indices: TrajectoryIndicesType,
    num_steps: int,
) -> CallbackCollector:
    """Factory function to create a CallbackCollector."""
    return CallbackCollector(indices=indices, total_steps=num_steps)