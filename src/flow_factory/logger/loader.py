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

# src/flow_factory/logger/loader.py
"""
Logger Loader
Factory function using registry pattern for extensibility.
"""
from typing import Optional
from .abc import Logger
from .registry import get_logger_class, list_registered_loggers


def load_logger(config) -> Optional[Logger]:
    """
    Load and initialize the appropriate logger backend based on configuration.
    
    Uses a registry pattern for automatic logger discovery and loading.
    Supports both built-in loggers and custom backends via python paths.
    
    Args:
        config: Arguments object containing logging_backend
    
    Returns:
        Logger instance or None if logging is disabled
    
    Raises:
        ImportError: If the logger backend is not registered or cannot be imported
    
    Examples:
        # Using built-in logger
        config.logging_backend = "wandb"
        logger = load_logger(config)
        
        # Using custom logger
        config.logging_backend = "my_package.loggers.CustomLogger"
        logger = load_logger(config)
        
        # Disabling logging
        config.logging_backend = "none"
        logger = load_logger(config)  # Returns None
    """
    logging_backend = config.logging_backend
    
    try:
        # Get logger class from registry or direct import
        logger_class = get_logger_class(logging_backend)
        
        # Return None if logging is disabled
        if logger_class is None:
            return None
        
        # Instantiate logger
        logger_instance = logger_class(config=config)
        
        return logger_instance
        
    except ImportError as e:
        registered_loggers = list(list_registered_loggers().keys())
        raise ImportError(
            f"Failed to load logger backend '{logging_backend}'. "
            f"Available backends: {registered_loggers}"
        ) from e