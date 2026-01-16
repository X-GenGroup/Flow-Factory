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

# src/flow_factory/logger/__init__.py
"""
Logging Module

Provides logging backends for experiment tracking with a registry-based
loading system for easy extensibility.

Supported backends:
- WandB (Weights & Biases)
- SwanLab
"""

from .abc import Logger, LogImage, LogVideo, LogTable
from .registry import (
    get_logger_class,
    list_registered_loggers,
)
from .loader import load_logger

__all__ = [
    # Core classes
    "Logger",
    "LogImage",
    "LogVideo",
    "LogTable",
    
    # Registry functions
    "get_logger_class",
    "list_registered_loggers",
    
    # Factory function
    "load_logger",
]