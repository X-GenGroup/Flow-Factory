# src/flow_factory/models/loader.py
"""
Model Adapter Loader
Factory function using registry pattern for extensibility.
"""
import logging

from .adapter import BaseAdapter
from .registry import get_model_adapter_class, list_registered_models
from ..hparams import Arguments

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)


def load_model(config: Arguments) -> BaseAdapter:

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
        adapter = adapter_class(config=config)
        
        logger.info(f"Successfully loaded {adapter_class.__name__}")
        return adapter
        
    except ImportError as e:
        registered_models = list(list_registered_models().keys())
        logger.error(
            f"Failed to load model adapter '{model_type}'. "
            f"Available models: {registered_models}"
        )
        raise

