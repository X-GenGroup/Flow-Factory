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

# src/flow_factory/ema/ema.py
"""
EMA Module Wrapper with functional decay scheduling.
"""

from typing import Optional, Literal
from collections.abc import Iterable
from contextlib import contextmanager
import torch

from ..utils.base import filter_kwargs
from .ema_utils import DecayFn, create_decay_fn
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class EMAModuleWrapper:
    """
    Exponential Moving Average wrapper with configurable decay function.
    
    Example:
        >>> # Simple usage with schedule type
        >>> ema = EMAModuleWrapper(model.parameters(), decay=0.999, decay_schedule="power")
        >>> 
        >>> # Custom decay function
        >>> from ema_utils import piecewise_linear_decay
        >>> ema = EMAModuleWrapper(model.parameters(), decay_fn=piecewise_linear_decay(0, 0.001, 0.5))
    """
    
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.999,
        update_step_interval: int = 1,
        device: Optional[torch.device] = None,
        # Decay function (direct or via config)
        decay_fn: Optional[DecayFn] = None,
        decay_schedule: Literal["constant", "power", "linear", "piecewise_linear", "cosine", "warmup_cosine"] = "power",
        # Schedule params passed to create_decay_fn
        **schedule_params
    ):
        """
        Args:
            parameters: Model parameters to track
            decay: Target EMA decay rate
            update_step_interval: Update EMA every N steps (0 = disabled)
            device: Device for EMA parameters
            decay_fn: Custom decay function (step -> decay). Overrides schedule params.
            decay_schedule: Schedule type if decay_fn not provided
            **schedule_params: Additional params for create_decay_fn:
                - initial_decay: Starting decay (linear/cosine)
                - warmup_steps: Warmup steps (power/linear/warmup_cosine)
                - total_steps: Total steps (cosine schedules)
                - flat_steps: piecewise_linear flat phase steps
                - ramp_rate: piecewise_linear ramp rate
        """
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]
        self.temp_stored_parameters = None
        
        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device
        self.num_updates = 0
        self._schedule_params = filter_kwargs(create_decay_fn, **schedule_params)
        
        # Set decay function
        if decay_fn is not None:
            self.decay_fn = decay_fn
            self._decay_schedule = "custom"
        else:
            self.decay_fn = create_decay_fn(
                schedule_type=decay_schedule,
                decay=decay,
                **self._schedule_params
            )
            self._decay_schedule = decay_schedule
        
        # Validation
        assert 0.0 <= decay <= 1.0, f"Decay must be in [0, 1], got {decay}"
        assert update_step_interval >= 0, f"update_step_interval must be >= 0, got {update_step_interval}"

    def get_current_decay(self, step: int) -> float:
        """Get decay value at given step."""
        return self.decay_fn(step)

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step: int) -> None:
        """Update EMA parameters."""
        if self.update_step_interval <= 0:
            return
        if (optimization_step + 1) % self.update_step_interval != 0:
            return
            
        parameters = list(parameters)
        assert len(parameters) == len(self.ema_parameters), "Parameter count mismatch"
        
        current_decay = self.decay_fn(optimization_step)
        one_minus_decay = 1 - current_decay
        
        for ema_param, param in zip(self.ema_parameters, parameters, strict=True):
            if not param.requires_grad:
                continue
            
            if ema_param.device == param.device:
                # In-place update: ema = ema * decay + param * (1 - decay)
                ema_param.mul_(current_decay).add_(param, alpha=one_minus_decay)
            else:
                # Cross-device update (memory efficient)
                param_copy = param.detach().to(ema_param.device)
                ema_param.mul_(current_decay).add_(param_copy, alpha=one_minus_decay)
                del param_copy
        
        self.num_updates += 1

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "EMAModuleWrapper":
        """Move EMA parameters to device/dtype."""
        if device is not None:
            self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]
        return self

    def copy_ema_to(self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True) -> None:
        """Copy EMA parameters to model (optionally storing originals)."""
        parameters = list(parameters)
        if store_temp:
            self.temp_stored_parameters = [p.detach().cpu().clone() for p in parameters]
        for ema_param, param in zip(self.ema_parameters, parameters, strict=True):
            if param.numel() > 0:
                param.data.copy_(ema_param.to(param.device).data)

    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Restore temporarily stored parameters."""
        assert self.temp_stored_parameters is not None, "No temp parameters stored"
        for temp_param, param in zip(self.temp_stored_parameters, parameters, strict=True):
            param.data.copy_(temp_param.to(param.device).data)
        self.temp_stored_parameters = None

    @contextmanager
    def use_ema_parameters(self, parameters: Iterable[torch.nn.Parameter]):
        """
        Context manager for temporary EMA swap.
        
        Usage:
            with ema.use_ema_parameters(model.parameters()):
                evaluate(model)  # Uses EMA weights
            # Original weights restored
        """
        self.copy_ema_to(parameters, store_temp=True)
        try:
            yield
        finally:
            self.copy_temp_to(parameters)

    def state_dict(self) -> dict:
        """Save state for checkpointing."""
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
            "num_updates": self.num_updates,
            "decay_schedule": self._decay_schedule,
            "schedule_params": self._schedule_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.decay = state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict["ema_parameters"]
        self.num_updates = state_dict.get("num_updates", 0)
        self.to(self.device)

    @staticmethod
    def get_decay_for_impact(impact: float, num_steps: int) -> float:
        """Calculate decay to achieve specific impact after num_steps."""
        assert 0 < impact < 1 and num_steps > 0
        return (1 - impact) ** (1 / num_steps)

    @staticmethod
    def get_steps_for_impact(impact: float, decay: float) -> int:
        """Calculate steps needed to achieve specific impact."""
        assert 0 < impact < 1 and 0 < decay < 1
        import math
        return int(math.log(1 - impact) / math.log(decay))

    def __repr__(self) -> str:
        return (
            f"EMAModuleWrapper(decay={self.decay}, schedule={self._decay_schedule}, "
            f"num_params={len(self.ema_parameters)}, updates={self.num_updates})"
        )