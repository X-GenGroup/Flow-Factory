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

# src/flow_factory/models/__init__.py
"""
Model Adapters Module

Provides model adapters for different diffusion/flow-matching architectures
with a registry-based loading system for easy extensibility.
"""

from .adapter import BaseAdapter
from .samples import BaseSample, ImageConditionSample, VideoConditionSample
from .registry import (
    get_model_adapter_class,
    list_registered_models,
)
from .loader import load_model

__all__ = [
    # Core classes
    "BaseAdapter",

    # Sample classes
    "BaseSample",
    "ImageConditionSample",
    "VideoConditionSample",
    
    # Registry functions
    "get_model_adapter_class",
    "list_registered_models",
    
    # Factory function
    "load_model",
]