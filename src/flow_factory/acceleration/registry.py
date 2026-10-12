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

# src/flow_factory/acceleration/registry.py
"""Accelerator registry: maps string identifiers to accelerator class paths.

Mirrors the trainer / model / reward registries (see ``constraints.md`` #1-#3):
case-insensitive keys, lazy ``importlib`` resolution, and a direct-python-path
fallback so users can plug in a custom accelerator without editing this file.
"""

import importlib
from typing import Any, Dict, Type

from ..utils.logger_utils import setup_logger
from .abc import BaseAccelerator

logger = setup_logger(__name__)


# Accelerator Registry Storage
_ACCELERATOR_REGISTRY: Dict[str, str] = {
    # Persistent stage='both' accelerators applied to rollout and training.
    # `attention_backend` is the single code path for attention-backend selection;
    # add it as a `shared` entry (before `torch_compile`) in the acceleration block.
    "attention_backend": "flow_factory.acceleration.attention_backend.AttentionBackendAccelerator",
    "torch_compile": "flow_factory.acceleration.torch_compile.CompileAccelerator",
    # Lossy (rollout-only; validator restricts to decoupled / distillation algos).
    "diffusers_cache": "flow_factory.acceleration.diffusers_cache.DiffusersCacheAccelerator",
}
_ACCELERATOR_REGISTRY = {k.lower(): v for k, v in _ACCELERATOR_REGISTRY.items()}

# Accelerators that were removed; mapped to an actionable migration message so a
# config still naming them fails fast with guidance instead of a generic error.
_REMOVED_ACCELERATORS: Dict[str, str] = {
    "cache_dit": (
        "The 'cache_dit' accelerator was removed. cache-dit only caches inside a "
        "pipeline `__call__` it patches, but Flow-Factory's rollout drives the "
        "transformer directly (adapter.inference()), so it was a silent no-op; its "
        "transformer-only path also assumes enable-once (incompatible with the "
        "per-epoch rollout lifecycle) and conflicts with the adapters' native "
        "diffusers `cache_context`. Use the `diffusers_cache` accelerator instead "
        "(diffusers-native FirstBlockCache / TaylorSeer / FasterCache; no extra "
        "dependency): `{ name: diffusers_cache, params: { policy: first_block, "
        "threshold: 0.08 } }`."
    ),
}


def get_accelerator_class(identifier: str) -> Type[BaseAccelerator]:
    """Resolve and import an accelerator class from the registry or a python path.

    Supports two modes:
    1. Registry lookup: ``'torch_compile'`` -> ``CompileAccelerator``.
    2. Direct import: ``'my_pkg.accel.CustomAccelerator'`` -> ``CustomAccelerator``.

    Args:
        identifier: Accelerator name or fully qualified class path.

    Returns:
        The accelerator class (a ``BaseAccelerator`` subclass).

    Raises:
        ValueError: If ``identifier`` names a removed accelerator.
        ImportError: If the accelerator cannot be loaded.
    """
    identifier_lower = identifier.lower()
    if identifier_lower in _REMOVED_ACCELERATORS:
        raise ValueError(_REMOVED_ACCELERATORS[identifier_lower])
    if identifier_lower in _ACCELERATOR_REGISTRY:
        class_path = _ACCELERATOR_REGISTRY[identifier_lower]
    else:
        class_path = identifier

    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        accelerator_class = getattr(module, class_name)
        logger.debug(f"Loaded accelerator: {identifier} -> {class_name}")
        return accelerator_class
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(
            f"Could not load accelerator '{identifier}'. "
            f"Ensure it is either:\n"
            f"  1. A registered accelerator: {list(_ACCELERATOR_REGISTRY.keys())}\n"
            f"  2. A valid python path (e.g., 'my_package.accel.CustomAccelerator')\n"
            f"Error: {e}"
        ) from e


def build_accelerator(identifier: str, params: Dict[str, Any]) -> BaseAccelerator:
    """Instantiate an accelerator from its identifier and parameters.

    Args:
        identifier: Accelerator name or fully qualified class path.
        params: Keyword parameters forwarded to the accelerator constructor.

    Returns:
        A constructed ``BaseAccelerator`` instance.

    Raises:
        TypeError: If the resolved class is not a ``BaseAccelerator`` subclass.
    """
    accelerator_class = get_accelerator_class(identifier)
    if not (isinstance(accelerator_class, type) and issubclass(accelerator_class, BaseAccelerator)):
        raise TypeError(
            f"Accelerator '{identifier}' resolved to {accelerator_class!r}, which is not a "
            "BaseAccelerator subclass."
        )
    return accelerator_class(**(params or {}))


def list_registered_accelerators() -> Dict[str, str]:
    """Return a copy of the accelerator name -> class-path mapping."""
    return _ACCELERATOR_REGISTRY.copy()
