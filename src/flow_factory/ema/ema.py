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
from typing import Optional
from collections.abc import Iterable
from contextlib import contextmanager
import torch
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class EMAModuleWrapper:
    """
    Exponential Moving Average (EMA) wrapper for model parameters.
    
    Maintains shadow copies of parameters with exponentially decayed updates.
    Useful for stabilizing training and improving generalization.
    """
    
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.995,
        update_step_interval: int = 10,
        device: Optional[torch.device] = None,
        warmup_steps: int = 10,
        use_warmup: bool = True,
    ):
        """
        Args:
            parameters: Model parameters to track
            decay: EMA decay rate (higher = slower updates)
            update_step_interval: Update EMA every N steps
            device: Device to store EMA parameters
            warmup_steps: Steps for decay warmup
            use_warmup: Whether to use warmup schedule
        """
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters]
        self.temp_stored_parameters = None
        
        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device
        self.warmup_steps = warmup_steps
        self.use_warmup = use_warmup
        self.num_updates = 0
        
        # Validation
        assert 0.0 <= decay <= 1.0, f"Decay must be in [0, 1], got {decay}"
        assert update_step_interval >= 0, f"""Update interval must be >= 0. Got {update_step_interval}. (0 indicates no updates, to maintain initial params.)"""

    def get_current_decay(self, optimization_step: int) -> float:
        """Calculate current decay with optional warmup."""
        if not self.use_warmup:
            return self.decay
            
        warmup_decay = min(
            (1 + optimization_step) / (self.warmup_steps + optimization_step),
            self.decay
        )
        return warmup_decay

    @torch.no_grad()
    def step(
        self, 
        parameters: Iterable[torch.nn.Parameter], 
        optimization_step: int
    ) -> None:
        """Update EMA parameters."""
        if self.update_step_interval <= 0 or (optimization_step + 1) % self.update_step_interval != 0:
            return
            
        parameters = list(parameters)
        assert len(parameters) == len(self.ema_parameters), \
            "Parameter count mismatch"
        
        one_minus_decay = 1 - self.get_current_decay(optimization_step)
        
        for ema_param, param in zip(self.ema_parameters, parameters, strict=True):
            if not param.requires_grad:
                continue
                
            if ema_param.device == param.device:
                # In-place update: ema = ema * decay + param * (1 - decay)
                ema_param.mul_(1 - one_minus_decay).add_(param, alpha=one_minus_decay)
            else:
                # Cross-device update (memory efficient)
                param_copy = param.detach().to(ema_param.device)
                ema_param.mul_(1 - one_minus_decay).add_(param_copy, alpha=one_minus_decay)
                del param_copy
        
        self.num_updates += 1

    def to(
        self, 
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None
    ) -> None:
        """Move EMA parameters to device/dtype."""
        if device is not None:
            self.device = device
            
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]

    def copy_ema_to(
        self, 
        parameters: Iterable[torch.nn.Parameter], 
        store_temp: bool = True
    ) -> None:
        """Copy EMA parameters to model (optionally storing originals)."""
        parameters = list(parameters)
        
        if store_temp:
            self.temp_stored_parameters = [p.detach().cpu().clone() for p in parameters]
        
        for ema_param, param in zip(self.ema_parameters, parameters, strict=True):
            if param.numel() == 0:
                continue

            param.data.copy_(ema_param.to(param.device).data)

    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Restore temporarily stored parameters."""
        assert self.temp_stored_parameters is not None, \
            "No temp parameters stored. Call copy_ema_to with store_temp=True first"
            
        for temp_param, param in zip(self.temp_stored_parameters, parameters, strict=True):
            param.data.copy_(temp_param.to(param.device).data)
        
        self.temp_stored_parameters = None

    @contextmanager
    def use_ema_parameters(self, parameters: Iterable[torch.nn.Parameter]):
        """
        Context manager for temporary EMA swap.
        
        Usage:
            with ema.average_parameters(model.parameters()):
                evaluate(model)  # Uses EMA weights
            # Original weights restored
        """
        self.copy_ema_to(parameters, store_temp=True)
        try:
            yield
        finally:
            self.copy_temp_to(parameters)

    def state_dict(self) -> dict:
        """Save EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
            "num_updates": self.num_updates,
            "warmup_steps": self.warmup_steps,
            "use_warmup": self.use_warmup,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict["ema_parameters"]
        self.num_updates = state_dict.get("num_updates", 0)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.use_warmup = state_dict.get("use_warmup", self.use_warmup)
        self.to(self.device)

    @staticmethod
    def get_decay_for_impact(impact: float, num_steps: int) -> float:
        """
        Calculate decay to achieve specific impact after num_steps.
        
        Args:
            impact: Desired impact in [0, 1] (e.g., 0.9 = 90% contribution)
            num_steps: Number of steps
            
        Returns:
            Required decay value
        """
        assert 0 < impact < 1, "Impact must be in (0, 1)"
        assert num_steps > 0, "num_steps must be positive"
        return (1 - impact) ** (1 / num_steps)

    @staticmethod
    def get_steps_for_impact(impact: float, decay: float) -> int:
        """
        Calculate steps needed to achieve specific impact.
        
        Args:
            impact: Desired impact in [0, 1]
            decay: Decay rate
            
        Returns:
            Number of steps required
        """
        assert 0 < impact < 1, "Impact must be in (0, 1)"
        assert 0 < decay < 1, "Decay must be in (0, 1)"
        import math
        return int(math.log(1 - impact) / math.log(decay))

    def __repr__(self) -> str:
        return (
            f"EMAModuleWrapper(decay={self.decay}, "
            f"num_params={len(self.ema_parameters)}, "
            f"num_updates={self.num_updates}, "
            f"device={self.device})"
        )