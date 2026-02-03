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
Trajectory Collector for Inference

Generic utility for memory-efficient tensor recording during denoising.
Supports collecting all, none, or specific timestep indices.
"""
from typing import Union, List, Optional, Literal, Set, TypeVar
import torch


T = TypeVar('T')
TrajectoryIndicesType = Union[Literal['all'], List[int], None]


class TrajectoryCollector:
    """
    Collects tensors at specified indices during denoising trajectory.
    
    Memory-efficient alternative to storing all intermediate values.
    Useful for AWM/NFT algorithms that only need initial/final states.
    
    Args:
        indices: Controls which steps to record:
            - 'all': Record all steps (default behavior)
            - None: Don't record any steps (returns None)
            - List[int]: Record only at specified indices
                - Index 0: Initial state (before denoising)
                - Index i (1 to T): State after i-th denoising step  
                - Index -1: Final state (same as index T)
                - Supports negative indexing like Python lists
        total_steps: Total number of denoising steps (T)
    
    Examples:
        >>> collector = TrajectoryCollector('all', total_steps=20)
        >>> collector = TrajectoryCollector(None, total_steps=20)     # No recording
        >>> collector = TrajectoryCollector([0, -1], total_steps=20)  # Initial + final only
        >>> collector = TrajectoryCollector([0, 10, -1], total_steps=20)  # Specific checkpoints
    
    Usage:
        >>> collector = TrajectoryCollector([0, -1], total_steps=20)
        >>> collector.collect(initial_latents, step_idx=0)
        >>> for i in range(20):
        ...     latents = denoise_step(latents)
        ...     collector.collect(latents, step_idx=i + 1)
        >>> trajectory = collector.get_result()  # [initial, final]
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
        """Convert user indices to normalized positive indices."""
        if self.indices is None:
            return None
        if self.indices == 'all':
            return None  # Signal to collect all
        
        # Total positions = total_steps + 1 (initial + each step result)
        total_positions = self.total_steps + 1
        normalized = set()
        
        for idx in self.indices:
            # Handle negative indices (Python-style)
            if idx < 0:
                idx = total_positions + idx
            # Clamp to valid range
            if 0 <= idx < total_positions:
                normalized.add(idx)
        
        return normalized
    
    @property
    def is_disabled(self) -> bool:
        """Check if collection is disabled."""
        return self.indices is None
    
    @property
    def collect_all(self) -> bool:
        """Check if collecting all steps."""
        return self.indices == 'all'
    
    def should_collect(self, step_idx: int) -> bool:
        """
        Check if value should be collected at this step.
        
        Args:
            step_idx: Current position (0=initial, 1..T=after each step)
        
        Returns:
            True if value should be recorded at this position
        """
        if self.is_disabled:
            return False
        if self.collect_all:
            return True
        return step_idx in self._target_indices
    
    def collect(self, value: torch.Tensor, step_idx: int) -> None:
        """
        Conditionally collect tensor at given step.
        
        Args:
            value: Tensor to potentially store
            step_idx: Current position index
        """
        if self.should_collect(step_idx):
            self._collected.append(value)
            self._collected_indices.append(step_idx)
    
    def get_result(self) -> Optional[List[torch.Tensor]]:
        """
        Get collected tensors.
        
        Returns:
            List of collected tensors, or None if disabled
        """
        if self.is_disabled:
            return None
        return self._collected
    
    @property
    def collected_indices(self) -> List[int]:
        """Get list of indices at which values were collected."""
        return self._collected_indices
    
    def reset(self) -> None:
        """Clear collected values for reuse."""
        self._collected = []
        self._collected_indices = []
    
    def __len__(self) -> int:
        """Number of collected values."""
        return len(self._collected)


def create_trajectory_collector(
    indices: TrajectoryIndicesType,
    num_steps: int,
) -> TrajectoryCollector:
    """
    Factory function to create a TrajectoryCollector.
    
    Args:
        indices: Which steps to collect ('all', None, or List[int])
        num_steps: Number of denoising steps
    
    Returns:
        Configured TrajectoryCollector instance
    """
    return TrajectoryCollector(
        indices=indices,
        total_steps=num_steps,
    )