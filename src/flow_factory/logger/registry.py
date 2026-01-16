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

# src/flow_factory/logger/registry.py
"""
Logger Registry System
Provides a centralized registry for logging backends with dynamic loading.
"""
from typing import Type, Dict, Optional
import importlib
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

# Logger Backend Registry Storage
_LOGGER_REGISTRY = {
    'wandb': 'flow_factory.logger.wandb.WandbLogger',
    'swanlab': 'flow_factory.logger.swanlab.SwanlabLogger',
    'tensorboard': 'flow_factory.logger.tensorboard.TensorboardLogger',
    'none': None,
}

def get_logger_class(identifier: str) -> Type:
    """
    Resolve and import a logger class from registry or python path.
    
    Supports three modes:
    1. Registry lookup: 'wandb' -> WandbLogger
    2. Direct import: 'my_package.loggers.CustomLogger' -> CustomLogger
    3. None/disable: 'none' -> None (no logging)
    
    Args:
        identifier: Logger backend name or fully qualified class path
    
    Returns:
        Logger class or None if logging is disabled
    
    Raises:
        ImportError: If the logger backend cannot be loaded
    
    Examples:
        >>> cls = get_logger_class('wandb')
        >>> logger = cls(config)
        
        >>> cls = get_logger_class('my_lib.loggers.CustomLogger')
        >>> logger = cls(config)
        
        >>> cls = get_logger_class('none')
        >>> # cls is None, logging disabled
    """
    if identifier is None or identifier.lower() == 'none':
        logger.info("Logging disabled (backend='none')")
        return None
    
    # Normalize identifier to lowercase for registry lookup
    identifier_lower = identifier.lower()
    
    # Check registry first
    if identifier_lower in _LOGGER_REGISTRY:
        class_path = _LOGGER_REGISTRY[identifier_lower]
        
        # Handle 'none' case
        if class_path is None:
            logger.info("Logging disabled (backend='none')")
            return None
    else:
        # Assume it's a direct python path
        class_path = identifier
    
    # Dynamic import
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        logger_class = getattr(module, class_name)
        
        logger.debug(f"Loaded logger backend: {identifier} -> {class_name}")
        return logger_class
        
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(
            f"Could not load logger backend '{identifier}'. "
            f"Ensure it is either:\n"
            f"  1. A registered backend: {list(_LOGGER_REGISTRY.keys())}\n"
            f"  2. A valid python path (e.g., 'my_package.loggers.CustomLogger')\n"
            f"  3. 'none' to disable logging\n"
            f"Error: {e}"
        ) from e


def list_registered_loggers() -> Dict[str, str]:
    """
    Get all registered logger backends.
    
    Returns:
        Dictionary mapping backend names to their class paths
    """
    return _LOGGER_REGISTRY.copy()