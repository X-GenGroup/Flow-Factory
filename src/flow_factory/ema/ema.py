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
"""EMA module wrapper with functional decay scheduling."""

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Any, Literal, Optional

import torch

from ..utils.base import filter_kwargs
from ..utils.logger_utils import setup_logger
from .ema_utils import DecayFn, create_decay_fn

logger = setup_logger(__name__)

EMA_STATE_VERSION = 1
_EMA_STATE_KEYS = frozenset(
    {
        "version",
        "decay",
        "update_step_interval",
        "ema_parameters",
        "num_updates",
        "decay_schedule",
        "schedule_params",
    }
)


class EMAModuleWrapper:
    """Exponential Moving Average wrapper with configurable decay function.

    Example:
        >>> # Simple usage with schedule type
        >>> ema = EMAModuleWrapper(model.parameters(), decay=0.999, decay_schedule="power")
        >>>
        >>> # Custom decay function
        >>> from flow_factory.ema import piecewise_linear_decay
        >>> decay_fn = piecewise_linear_decay(0, 0.001, 0.5)
        >>> ema = EMAModuleWrapper(model.parameters(), decay_fn=decay_fn)
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.999,
        update_step_interval: int = 1,
        device: Optional[torch.device] = None,
        # Decay function (direct or via config)
        decay_fn: Optional[DecayFn] = None,
        decay_schedule: Literal[
            "constant", "power", "linear", "piecewise_linear", "cosine", "warmup_cosine"
        ] = "power",
        # Schedule params passed to create_decay_fn
        **schedule_params,
    ):
        """Snapshot ``parameters`` as the initial EMA weights and build the decay schedule.

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
                schedule_type=decay_schedule, decay=decay, **self._schedule_params
            )
            self._decay_schedule = decay_schedule

        # Validation
        assert 0.0 <= decay <= 1.0, f"Decay must be in [0, 1], got {decay}"
        assert (
            update_step_interval >= 0
        ), f"update_step_interval must be >= 0, got {update_step_interval}"

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

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> "EMAModuleWrapper":
        """Move EMA parameters to device/dtype."""
        if device is not None:
            self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]
        return self

    def copy_ema_to(
        self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True
    ) -> None:
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
        """Temporarily swap the EMA weights into ``parameters``.

        ``parameters`` is iterated twice (swap in, then restore), so pass a list rather than
        a generator.

        Example:
            params = list(model.parameters())
            with ema.use_ema_parameters(params):
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
        for index, parameter in enumerate(self.ema_parameters):
            if type(parameter) is not torch.Tensor:
                raise TypeError(
                    "EMA runtime checkpointing supports only replicated plain tensors; "
                    f"parameter {index} is {type(parameter).__name__}. Sharded/DTensor EMA "
                    "state requires a distributed-aware gather implementation."
                )
        state = {
            "version": EMA_STATE_VERSION,
            "decay": self.decay,
            "update_step_interval": self.update_step_interval,
            "ema_parameters": self.ema_parameters,
            "num_updates": self.num_updates,
            "decay_schedule": self._decay_schedule,
            "schedule_params": dict(self._schedule_params),
        }
        # Saving while EMA weights are temporarily installed would pair the EMA
        # policy with the pre-swap runtime payload and make resume permanently use
        # the wrong live parameters. Reuse the strict, non-mutating load validator
        # so every state we emit is also one this wrapper can restore.
        self._validated_state_dict(state)
        return state

    def validate_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Validate a checkpoint payload without mutating the live EMA state."""
        self._validated_state_dict(state_dict)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load a compatible replicated EMA state after complete validation."""
        parameters, num_updates = self._validated_state_dict(state_dict)
        self.ema_parameters = [parameter.detach().clone() for parameter in parameters]
        self.num_updates = num_updates
        self.to(self.device)

    def _validated_state_dict(
        self,
        state_dict: Mapping[str, Any],
    ) -> tuple[list[torch.Tensor], int]:
        """Return validated tensor and counter values without changing this wrapper."""
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                "expected EMA state as a mapping, "
                f"received {type(state_dict).__name__}: {state_dict!r}"
            )
        non_string_keys = tuple(key for key in state_dict if type(key) is not str)
        if non_string_keys:
            raise TypeError(f"expected EMA state keys to be str, received {non_string_keys!r}")
        received_keys = frozenset(state_dict)
        if received_keys != _EMA_STATE_KEYS:
            raise ValueError(
                f"EMA state keys mismatch: expected {tuple(sorted(_EMA_STATE_KEYS))!r}, "
                f"received {tuple(sorted(received_keys))!r}"
            )

        version = state_dict["version"]
        if type(version) is not int:
            raise TypeError(
                "expected EMA state version to be int, "
                f"received {type(version).__name__}: {version!r}"
            )
        if version != EMA_STATE_VERSION:
            raise ValueError(
                f"EMA state version mismatch: expected {EMA_STATE_VERSION}, received {version}"
            )

        decay = state_dict["decay"]
        if type(decay) not in (int, float):
            raise TypeError(
                "expected EMA state decay to be int or float, "
                f"received {type(decay).__name__}: {decay!r}"
            )
        if float(decay) != float(self.decay):
            raise ValueError(f"EMA state decay mismatch: expected {self.decay}, received {decay}")

        update_step_interval = state_dict["update_step_interval"]
        if type(update_step_interval) is not int:
            raise TypeError(
                "expected EMA state update_step_interval to be int, "
                f"received {type(update_step_interval).__name__}: "
                f"{update_step_interval!r}"
            )
        if update_step_interval < 0:
            raise ValueError(
                "expected EMA state update_step_interval as a non-negative int, "
                f"received {update_step_interval!r}"
            )
        if update_step_interval != self.update_step_interval:
            raise ValueError(
                "EMA state update_step_interval mismatch: expected "
                f"{self.update_step_interval}, received {update_step_interval}"
            )

        decay_schedule = state_dict["decay_schedule"]
        if type(decay_schedule) is not str:
            raise TypeError(
                "expected EMA state decay_schedule to be str, "
                f"received {type(decay_schedule).__name__}: {decay_schedule!r}"
            )
        if decay_schedule != self._decay_schedule:
            raise ValueError(
                "EMA state decay_schedule mismatch: expected "
                f"{self._decay_schedule!r}, received {decay_schedule!r}"
            )

        schedule_params = state_dict["schedule_params"]
        if not isinstance(schedule_params, Mapping):
            raise TypeError(
                "expected EMA state schedule_params as a mapping, "
                f"received {type(schedule_params).__name__}: {schedule_params!r}"
            )
        if dict(schedule_params) != self._schedule_params:
            raise ValueError(
                "EMA state schedule_params mismatch: expected "
                f"{self._schedule_params!r}, received {dict(schedule_params)!r}"
            )

        num_updates = state_dict["num_updates"]
        if type(num_updates) is not int:
            raise TypeError(
                "expected EMA state num_updates to be int, "
                f"received {type(num_updates).__name__}: {num_updates!r}"
            )
        if num_updates < 0:
            raise ValueError(
                "expected EMA state num_updates as a non-negative int, " f"received {num_updates!r}"
            )

        parameters = state_dict["ema_parameters"]
        if not isinstance(parameters, list):
            raise TypeError(
                "expected EMA state ema_parameters as a list, "
                f"received {type(parameters).__name__}: {parameters!r}"
            )
        if len(parameters) != len(self.ema_parameters):
            raise ValueError(
                "EMA state parameter count mismatch: expected "
                f"{len(self.ema_parameters)}, received {len(parameters)}"
            )
        for index, (received, expected) in enumerate(
            zip(parameters, self.ema_parameters, strict=True)
        ):
            if type(received) is not torch.Tensor:
                raise TypeError(
                    "expected EMA state parameter "
                    f"{index} to be a plain torch.Tensor, received "
                    f"{type(received).__name__}: {received!r}"
                )
            if received.shape != expected.shape:
                raise ValueError(
                    f"EMA state parameter {index} shape mismatch: expected "
                    f"{tuple(expected.shape)}, received {tuple(received.shape)}"
                )
            if received.dtype != expected.dtype:
                raise ValueError(
                    f"EMA state parameter {index} dtype mismatch: expected "
                    f"{expected.dtype}, received {received.dtype}"
                )

        if self.temp_stored_parameters is not None:
            raise RuntimeError("cannot load EMA state while EMA parameters are installed")
        return parameters, num_updates

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
