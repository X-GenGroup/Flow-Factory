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

# src/flow_factory/models/loader.py
"""
Model Adapter Loader
Factory function using registry pattern for extensibility.
"""
import logging
from accelerate import Accelerator

from .adapter import BaseAdapter
from .registry import get_model_adapter_class, list_registered_models
from ..hparams import Arguments
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

def load_model(config: Arguments, accelerator : Accelerator) -> BaseAdapter:
    """
    Factory function to instantiate the correct model adapter based on configuration.
    
    Uses a registry pattern for automatic model discovery and loading.
    Supports both built-in models and custom adapters via python paths.
    
    Args:
        config: Arguments object containing model_args with 'model_type'
    
    Returns:
        An instance of a subclass of BaseAdapter
    
    Raises:
        ImportError: If the model type is not registered or cannot be imported
    
    Examples:
        # Using built-in model
        config.model_args.model_type = "flux1"
        adapter = load_model(config)
        
        # Using custom model adapter
        config.model_args.model_type = "my_package.models.CustomAdapter"
        adapter = load_model(config)
    """

    model_type = config.model_args.model_type
    
    logger.info(f"Loading model architecture: {model_type}...")
    
    try:
        # Get adapter class from registry or direct import
        adapter_class = get_model_adapter_class(model_type)
        
        # Instantiate adapter
        adapter = adapter_class(config=config, accelerator=accelerator)
        
        logger.info(f"Successfully loaded {adapter_class.__name__}")
        return adapter
        
    except ImportError as e:
        registered_models = list(list_registered_models().keys())
        logger.error(
            f"Failed to load model adapter '{model_type}'. "
            f"Available models: {registered_models}"
        )
        raise

