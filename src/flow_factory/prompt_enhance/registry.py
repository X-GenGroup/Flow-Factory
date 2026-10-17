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

"""Prompt-enhancer registry with lazy class loading."""

from __future__ import annotations

import importlib
from typing import Dict, Type

from .abc import BasePromptEnhancer

_PROMPT_ENHANCER_REGISTRY: Dict[str, str] = {
    "flux2": "flow_factory.prompt_enhance.enhancers.flux2.Flux2PromptEnhancer",
    "ltx2": "flow_factory.prompt_enhance.enhancers.ltx2.LTX2PromptEnhancer",
    "flux3": "flow_factory.prompt_enhance.enhancers.flux3.Flux3PromptEnhancer",
    "openai-compatible": (
        "flow_factory.prompt_enhance.enhancers.openai_compatible.OpenAICompatiblePromptEnhancer"
    ),
}


def get_prompt_enhancer_class(identifier: str) -> Type[BasePromptEnhancer]:
    """Resolve a registered enhancer key or a ``package.module.Class`` python path.

    Args:
        identifier: Registry key (case-insensitive) or fully qualified class path.

    Returns:
        Prompt-enhancer class.

    Raises:
        ImportError: If the class cannot be imported.
        TypeError: If the resolved object is not a ``BasePromptEnhancer`` subclass.
    """
    class_path = _PROMPT_ENHANCER_REGISTRY.get(identifier.lower(), identifier)
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        enhancer_class = getattr(importlib.import_module(module_path), class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(
            f"Could not load prompt enhancer {identifier!r}. Expected a registered key "
            f"{sorted(_PROMPT_ENHANCER_REGISTRY)} or a python path such as "
            f"'my_package.enhancers.MyEnhancer'. Error: {e}"
        ) from e
    if not isinstance(enhancer_class, type) or not issubclass(enhancer_class, BasePromptEnhancer):
        raise TypeError(
            f"expected prompt enhancer {identifier!r} to resolve to a BasePromptEnhancer "
            f"subclass, received {enhancer_class!r}"
        )
    return enhancer_class


def list_registered_prompt_enhancers() -> Dict[str, str]:
    """Return a copy of the registry mapping enhancer keys to class paths."""
    return dict(_PROMPT_ENHANCER_REGISTRY)
