# src/flow_factory/model/loader.py
import logging
import importlib
from typing import Tuple
from accelerate import Accelerator
from .adapter import BaseAdapter
from ..hparams import *

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

# register map
_MODEL_REGISTRY = {
    "flux1": ("flux1", "Flux1Adapter"),
    "z-image": ("z_image", "ZImageAdapter"),
}

def load_model(config : Arguments) -> BaseAdapter:
    """
    Factory function to instantiate the correct model adapter based on configuration.
    
    Args:
        model_args: DataClass containing 'model_type', 'model_name_or_path', etc.
        training_args: DataClass containing bf16/fp16 settings.
    
    Returns:
        An instance of a subclass of BaseAdapter.
    """
    model_args = config.model_args
    model_type = model_args.model_type.lower()
    
    logger.info(f"Loading model architecture: {model_type}...")
    
    if model_type not in _MODEL_REGISTRY:
        raise NotImplementedError(f"Model type '{model_type}' is not supported yet.")

    module_name, class_name = _MODEL_REGISTRY[model_type]
    module = importlib.import_module(f'.{module_name}', package=__package__)
    adapter_class = getattr(module, class_name)


    
    return adapter_class(config=config)