# src/flow_factory/models/registry.py
"""
Model Adapter Registry System
Provides a centralized registry for model adapters with dynamic loading.
"""
from typing import Type, Dict
import importlib
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

# Model Adapter Registry Storage
_MODEL_ADAPTER_REGISTRY: Dict[str, str] = {
    'flux1': 'flow_factory.models.flux1.Flux1Adapter',
    'flux1_init': 'flow_factory.models.flux1_init.Flux1InitAdapter',
    'z-image': 'flow_factory.models.z_image.ZImageAdapter',
    'qwenimage': 'flow_factory.models.qwenimage.QwenImageAdapter',
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