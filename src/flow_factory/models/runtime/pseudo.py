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

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn

from .abc import ComponentRuntime


class PseudoPipelineRuntime(ComponentRuntime):
    """Manage an explicit component mapping for a pseudo-pipeline container.

    Args:
        pipeline: Compatibility container retained as ``adapter.pipeline``.
        components: Explicit canonical component mapping.
        aliases: Addressable component aliases excluded from device lifecycle
            enumeration.
        alias_routes: Alias ownership as ``alias -> (physical root, path)``.
    """

    def __init__(
        self,
        pipeline: Any,
        components: Mapping[str, Any],
        aliases: Optional[Mapping[str, Any]] = None,
        alias_routes: Optional[Mapping[str, tuple[str, Sequence[str]]]] = None,
    ) -> None:
        """Initialize an explicit pseudo-pipeline runtime.

        Args:
            pipeline: Compatibility pipeline/container.
            components: Canonical modules managed by stage lifecycle.
            aliases: Addressable module aliases excluded from stage lifecycle.
            alias_routes: Alias ownership as ``alias -> (physical root, path)``.

        Raises:
            TypeError: If a component or alias is not a ``torch.nn.Module``.
            ValueError: If an alias duplicates a canonical component name.
        """
        super().__init__(pipeline)
        aliases = aliases or {}
        duplicate_names = set(components).intersection(aliases)
        if duplicate_names:
            raise ValueError(
                "Pseudo-pipeline aliases must not duplicate canonical component names; "
                f"received duplicates={sorted(duplicate_names)}."
            )
        invalid = {
            name: type(component).__name__
            for name, component in {**components, **aliases}.items()
            if not isinstance(component, nn.Module)
        }
        if invalid:
            raise TypeError(
                "Pseudo-pipeline components must all be torch.nn.Module instances; "
                f"received invalid entries={invalid}."
            )
        self._canonical_components: Dict[str, Any] = dict(components)
        self._alias_components: Dict[str, Any] = dict(aliases)
        alias_routes = alias_routes or {}
        self._alias_routes: Dict[str, tuple[str, tuple[str, ...]]] = {}
        for alias_name, route in alias_routes.items():
            if alias_name not in self._alias_components:
                raise ValueError(
                    f"Pseudo-pipeline alias route {alias_name!r} has no matching alias; "
                    f"expected one of {sorted(self._alias_components)}"
                )
            root, path = route
            if root not in self._canonical_components:
                raise ValueError(
                    f"Pseudo-pipeline alias {alias_name!r} references unknown root={root!r}; "
                    f"expected one of {sorted(self._canonical_components)}"
                )
            self._alias_routes[alias_name] = (root, tuple(path))

    @property
    def canonical_components(self) -> Mapping[str, Any]:
        """Return the explicit canonical component mapping."""
        return self._canonical_components

    @property
    def alias_components(self) -> Mapping[str, Any]:
        """Return explicit addressable aliases excluded from device lifecycle."""
        return self._alias_components

    def physical_route(self, name: str) -> tuple[str, tuple[str, ...]]:
        if name in self._alias_routes:
            return self._alias_routes[name]
        return super().physical_route(name)

    def load_root_remainder(
        self,
        root: str,
        *,
        excluded_paths: Sequence[tuple[str, ...]],
        device: Union[torch.device, str],
    ) -> None:
        """Move auxiliary state inside a target-owned pseudo-pipeline root."""
        component = self._canonical_components[root]
        self._move_remainder(component, tuple(excluded_paths), device)

    @classmethod
    def _move_remainder(
        cls,
        module: nn.Module,
        excluded_paths: tuple[tuple[str, ...], ...],
        device: Union[torch.device, str],
    ) -> None:
        """Recursively move a module tree while leaving excluded routes untouched."""
        if () in excluded_paths:
            return
        for parameter in module.parameters(recurse=False):
            parameter.data = parameter.data.to(device)
        for buffer in module.buffers(recurse=False):
            buffer.data = buffer.data.to(device)

        excluded_by_child: Dict[str, List[tuple[str, ...]]] = {}
        for path in excluded_paths:
            if path:
                excluded_by_child.setdefault(path[0], []).append(path[1:])
        for child_name, child in module.named_children():
            child_exclusions = excluded_by_child.get(child_name)
            if child_exclusions is None:
                child.to(device)
            elif () not in child_exclusions:
                cls._move_remainder(child, tuple(child_exclusions), device)

    def _get_materialized_component(self, name: str) -> Any:
        if name in self._alias_components:
            return self._alias_components[name]
        return self._canonical_components.get(name)

    def _materialize_components(self, names: List[str]) -> None:
        missing = [name for name in names if name not in self.declared_components]
        if missing:
            raise RuntimeError(
                "Pseudo-pipeline is missing required explicit components; "
                f"expected={missing}, received={self.declared_component_names}."
            )
