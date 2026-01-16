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

# src/flow_factory/models/registry.py
"""
Model Adapter Registry System
Provides a centralized registry for model adapters with dynamic loading.
"""
from typing import Type, Dict
import importlib
import logging

from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

# Model Adapter Registry Storage
_MODEL_ADAPTER_REGISTRY: Dict[str, str] = {
    'sd3-5': 'flow_factory.models.stable_diffusion.sd3_5.SD3_5Adapter',
    'flux1': 'flow_factory.models.flux.flux1.Flux1Adapter',
    'flux1-kontext': 'flow_factory.models.flux.flux1_kontext.Flux1KontextAdapter',
    'flux2': 'flow_factory.models.flux.flux2.Flux2Adapter',
    'qwen-image': 'flow_factory.models.qwen_image.qwen_image.QwenImageAdapter',
    'qwen-image-edit-plus': 'flow_factory.models.qwen_image.qwen_image_edit_plus.QwenImageEditPlusAdapter',
    'z-image': 'flow_factory.models.z_image.z_image.ZImageAdapter',
    'wan2_i2v': 'flow_factory.models.wan.wan2_i2v.Wan2_I2V_Adapter',
    'wan2_t2v': 'flow_factory.models.wan.wan2_t2v.Wan2_T2V_Adapter',
    'wan2_v2v': 'flow_factory.models.wan.wan2_v2v.Wan2_V2V_Adapter',
}

def get_model_adapter_class(identifier: str) -> Type:
    """
    Resolve and import a model adapter class from registry or python path.
    
    Supports two modes:
    1. Registry lookup: 'flux1' -> Flux1Adapter
    2. Direct import: 'my_package.models.CustomAdapter' -> CustomAdapter
    
    Args:
        identifier: Model type name or fully qualified class path
    
    Returns:
        Model adapter class
    
    Raises:
        ImportError: If the model adapter cannot be loaded
    
    Examples:
        >>> cls = get_model_adapter_class('flux1')
        >>> adapter = cls(config)
        
        >>> cls = get_model_adapter_class('my_lib.models.CustomAdapter')
        >>> adapter = cls(config)
    """
    # Normalize identifier to lowercase for registry lookup
    identifier_lower = identifier.lower()
    
    # Check registry first
    class_path = _MODEL_ADAPTER_REGISTRY.get(identifier_lower, identifier)
    
    # Dynamic import
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        adapter_class = getattr(module, class_name)
        
        logger.debug(f"Loaded model adapter: {identifier} -> {class_name}")
        return adapter_class
        
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(
            f"Could not load model adapter '{identifier}'. "
            f"Ensure it is either:\n"
            f"  1. A registered model type: {list(_MODEL_ADAPTER_REGISTRY.keys())}\n"
            f"  2. A valid python path (e.g., 'my_package.models.CustomAdapter')\n"
            f"Error: {e}"
        ) from e


def list_registered_models() -> Dict[str, str]:
    """
    Get all registered model adapters.
    
    Returns:
        Dictionary mapping model types to their class paths
    """
    return _MODEL_ADAPTER_REGISTRY.copy()