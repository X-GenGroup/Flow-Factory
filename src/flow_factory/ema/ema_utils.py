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

# src/flow_factory/ema/ema_utils.py
"""
Functional EMA Decay Utilities for Flow-Factory.

Factory functions that return step -> decay callables.
"""

import math
from typing import Callable, Literal

DecayFn = Callable[[int], float]


def constant_decay(decay: float = 0.999) -> DecayFn:
    """
    Constant decay rate.
    
    decay(t) = d
    """
    assert 0.0 <= decay <= 1.0
    return lambda step: decay


def power_warmup_decay(decay: float = 0.999, warmup_steps: int = 10) -> DecayFn:
    """
    Power-law warmup (diffusers/timm style).
    
    decay(t) = min(d, (1 + t) / (warmup_steps + t))
    """
    assert 0.0 <= decay <= 1.0
    assert warmup_steps >= 0
    
    if warmup_steps == 0:
        return constant_decay(decay)
    
    def _decay(step: int) -> float:
        return min(decay, (1 + step) / (warmup_steps + step))
    return _decay


def linear_warmup_decay(
    decay: float = 0.999,
    initial_decay: float = 0.0,
    warmup_steps: int = 100
) -> DecayFn:
    """
    Linear warmup from initial to target.
    
    decay(t) = d0 + (d - d0) * min(t / warmup_steps, 1)
    """
    assert 0.0 <= decay <= 1.0
    assert 0.0 <= initial_decay <= 1.0
    assert warmup_steps > 0
    
    delta = decay - initial_decay
    
    def _decay(step: int) -> float:
        if step >= warmup_steps:
            return decay
        return initial_decay + delta * (step / warmup_steps)
    return _decay


def piecewise_linear_decay(
    flat_steps: int = 0,
    ramp_rate: float = 0.001,
    max_decay: float = 0.5
) -> DecayFn:
    """
    Piecewise linear: flat → ramp → hold.
    
    decay(t) = 
        0, if t < flat_steps
        min((t - flat_steps) * ramp_rate, max_decay), otherwise
        
    Common presets:
        - Fast warmup: flat_steps=0, ramp_rate=0.001, max_decay=0.5
        - Slow warmup: flat_steps=75, ramp_rate=0.0075, max_decay=0.999
    """
    assert flat_steps >= 0
    assert ramp_rate >= 0
    assert 0.0 <= max_decay <= 1.0
    
    def _decay(step: int) -> float:
        if step < flat_steps:
            return 0.0
        return min((step - flat_steps) * ramp_rate, max_decay)
    return _decay


def cosine_decay(
    decay: float = 0.999,
    initial_decay: float = 0.9,
    total_steps: int = 1000
) -> DecayFn:
    """
    Cosine annealing decay.
    
    decay(t) = d0 + (d - d0) * (1 - cos(π * min(t/T, 1))) / 2
    """
    assert 0.0 <= decay <= 1.0
    assert 0.0 <= initial_decay <= 1.0
    assert total_steps > 0
    
    delta = decay - initial_decay
    
    def _decay(step: int) -> float:
        progress = min(step / total_steps, 1.0)
        return initial_decay + delta * (1 - math.cos(math.pi * progress)) / 2
    return _decay


def warmup_cosine_decay(
    decay: float = 0.999,
    initial_decay: float = 0.9,
    warmup_steps: int = 100,
    total_steps: int = 1000
) -> DecayFn:
    """
    Linear warmup + cosine annealing.
    
    Phase 1: Linear 0 → d0 over warmup_steps
    Phase 2: Cosine d0 → d over remaining steps
    """
    assert 0.0 <= decay <= 1.0
    assert 0.0 <= initial_decay <= 1.0
    assert 0 < warmup_steps < total_steps
    
    cosine_steps = total_steps - warmup_steps
    delta = decay - initial_decay
    
    def _decay(step: int) -> float:
        if step < warmup_steps:
            return initial_decay * (step / warmup_steps)
        cosine_progress = min((step - warmup_steps) / cosine_steps, 1.0)
        return initial_decay + delta * (1 - math.cos(math.pi * cosine_progress)) / 2
    return _decay


def create_decay_fn(
    schedule_type: Literal["constant", "power", "linear", "piecewise_linear", "cosine", "warmup_cosine"] = "power",
    decay: float = 0.999,
    initial_decay: float = 0.0,
    warmup_steps: int = 10,
    total_steps: int = 1000,
    flat_steps: int = 0,
    ramp_rate: float = 0.001,
) -> DecayFn:
    """
    Factory function to create decay callable from config.
    
    Args:
        schedule_type: "constant", "power", "linear", "piecewise_linear", "cosine", "warmup_cosine"
        decay: Target decay rate (also used as max_decay for piecewise_linear)
        initial_decay: Starting decay (linear/cosine)
        warmup_steps: Warmup duration (power/linear/warmup_cosine)
        total_steps: Total steps (cosine schedules)
        flat_steps: piecewise_linear flat phase steps
        ramp_rate: piecewise_linear ramp rate
        
    Returns:
        Callable[[int], float]: step -> decay function
    """
    if schedule_type == "constant":
        return constant_decay(decay)
    
    elif schedule_type == "power":
        return power_warmup_decay(decay, warmup_steps)
    
    elif schedule_type == "linear":
        return linear_warmup_decay(decay, initial_decay, warmup_steps)
    
    elif schedule_type == "piecewise_linear":
        return piecewise_linear_decay(flat_steps, ramp_rate, decay)
    
    elif schedule_type == "cosine":
        return cosine_decay(decay, initial_decay, total_steps)
    
    elif schedule_type == "warmup_cosine":
        return warmup_cosine_decay(decay, initial_decay, warmup_steps, total_steps)
    
    else:
        raise ValueError(
            f"Unknown schedule_type: {schedule_type}. "
            f"Choose from: constant, power, linear, piecewise_linear, cosine, warmup_cosine"
        )


def visualize_ema_schedules(total_steps=1000):    
    import matplotlib.pyplot as plt
    import numpy as np
    steps = np.arange(total_steps)
    
    configs = [
        ("Constant (0.999)", {"schedule_type": "constant", "decay": 0.999}),
        ("Power Warmup", {"schedule_type": "power", "decay": 0.999, "warmup_steps": 100}),
        ("Linear Warmup", {"schedule_type": "linear", "decay": 0.999, "initial_decay": 0.9, "warmup_steps": 500}),
        ("piecewise_linear (NFT Style)", {"schedule_type": "piecewise_linear", "flat_steps": 200, "ramp_rate": 0.0005, "decay": 0.999}),
        ("Cosine Decay", {"schedule_type": "cosine", "decay": 0.999, "initial_decay": 0.9, "total_steps": 1500}),
        ("Warmup Cosine", {"schedule_type": "warmup_cosine", "decay": 0.999, "initial_decay": 0.9, "warmup_steps": 400, "total_steps": 1500}),
    ]

    plt.figure(figsize=(12, 7))
    
    for label, config in configs:
        decay_fn = create_decay_fn(**config)
        y = [decay_fn(int(s)) for s in steps]
        plt.plot(steps, y, label=label, linewidth=2)

    plt.title("Comparison of EMA Decay Schedules", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Decay Rate (α)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.ylim(0.0, 1.05)
    
    plt.axes([0.2, 0.25, 0.3, 0.3])
    for label, config in configs:
        decay_fn = create_decay_fn(**config)
        y = [decay_fn(int(s)) for s in steps[:300]]
        plt.plot(steps[:300], y)
    plt.title("Early Steps Detail")
    plt.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_ema_schedules()