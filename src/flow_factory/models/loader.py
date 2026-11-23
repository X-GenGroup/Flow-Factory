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
"""Factory that instantiates model adapters through the adapter registry."""

from accelerate import Accelerator

from ..hparams import Arguments
from ..utils.logger_utils import setup_logger
from .abc import BaseAdapter
from .component_execution import (
    resolve_component_execution,
    resolve_reward_execution,
    validate_component_execution,
)
from .registry import get_model_adapter_class, list_registered_models

logger = setup_logger(__name__)


def load_model(config: Arguments, accelerator: Accelerator) -> BaseAdapter:
    """Instantiate the model adapter selected by ``config.model_args.model_type``.

    ``model_type`` is either a registered model name or a python path to a custom adapter class.

    Args:
        config: Full arguments; ``config.model_args.model_type`` selects the adapter.
        accelerator: Accelerator passed to the adapter constructor.

    Returns:
        An instance of a subclass of BaseAdapter.

    Raises:
        ImportError: If the model type is not registered or cannot be imported.
        ValueError: If an explicit auxiliary policy cannot be applied.

    Example:
        # Using built-in model
        config.model_args.model_type = "flux1"
        adapter = load_model(config, accelerator)

        # Using custom model adapter
        config.model_args.model_type = "my_package.models.CustomAdapter"
        adapter = load_model(config, accelerator)
    """

    model_type = config.model_args.model_type
    requests = config.component_execution_args
    if requests is not None:
        reward_execution = resolve_reward_execution(config, requests)
        validate_component_execution(requests, accelerator)

    logger.info(f"Loading model architecture: {model_type}...")

    try:
        # Get adapter class from registry or direct import
        adapter_class = get_model_adapter_class(model_type)

        # Instantiate adapter
        adapter = adapter_class(config=config, accelerator=accelerator)
        if requests is not None:
            adapter.component_execution_plan = resolve_component_execution(
                config, adapter, requests, reward_execution
            )

        logger.info(f"Successfully loaded {adapter_class.__name__}")
        return adapter

    except ImportError as e:
        registered_models = list(list_registered_models().keys())
        logger.error(
            f"Failed to load model adapter '{model_type}'. "
            f"Available models: {registered_models}"
        )
        raise
