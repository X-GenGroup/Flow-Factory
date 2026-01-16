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

# src/flow_factory/scheduler/registry.py
"""
Scheduler Registry System
Maps diffusers scheduler classes to custom SDE scheduler implementations.
"""
from typing import Type, Dict, Optional
import importlib

from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

# Maps diffusers scheduler class names to custom SDE scheduler paths
_SCHEDULER_REGISTRY: Dict[str, str] = {
    'FlowMatchEulerDiscreteScheduler': 'flow_factory.scheduler.flow_match_euler_discrete.FlowMatchEulerDiscreteSDEScheduler',
    'UniPCMultistepScheduler': 'flow_factory.scheduler.unipc_multistep.UniPCMultistepSDEScheduler',
}


def register_scheduler(diffusers_class_name: str, sde_class_path: str) -> None:
    """Register a custom SDE scheduler for a diffusers scheduler class."""
    _SCHEDULER_REGISTRY[diffusers_class_name] = sde_class_path
    logger.debug(f"Registered scheduler: {diffusers_class_name} -> {sde_class_path}")


def get_sde_scheduler_class(scheduler) -> Type:
    """
    Get the SDE scheduler class for a given diffusers scheduler.
    
    Args:
        scheduler: A diffusers scheduler instance or class
    
    Returns:
        Corresponding SDE scheduler class
    
    Raises:
        ImportError: If no matching SDE scheduler is found
    """
    class_name = scheduler.__class__.__name__ if not isinstance(scheduler, type) else scheduler.__name__
    
    if class_name not in _SCHEDULER_REGISTRY:
        raise ImportError(
            f"No SDE scheduler registered for '{class_name}'. "
            f"Registered schedulers: {list(_SCHEDULER_REGISTRY.keys())}"
        )
    
    class_path = _SCHEDULER_REGISTRY[class_name]
    module_path, cls_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def list_registered_schedulers() -> Dict[str, str]:
    """Get all registered scheduler mappings."""
    return _SCHEDULER_REGISTRY.copy()