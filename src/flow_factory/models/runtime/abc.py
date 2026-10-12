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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch


class ComponentRuntime(ABC):
    """Manage canonical components and device-excluded runtime overrides.

    Args:
        pipeline: Backend pipeline or explicit component container.
    """

    def __init__(self, pipeline: Any) -> None:
        """Initialize a component runtime.

        Args:
            pipeline: Backend pipeline or explicit component container.

        Raises:
            ValueError: If ``pipeline`` is ``None``.
        """
        if pipeline is None:
            raise ValueError("ComponentRuntime requires a pipeline/container instance, got None.")
        self.pipeline = pipeline
        self.override_components: Dict[str, Any] = {}

    @property
    @abstractmethod
    def canonical_components(self) -> Mapping[str, Any]:
        """Return canonical component specifications keyed by component name."""
        pass

    @property
    def alias_components(self) -> Mapping[str, Any]:
        """Return addressable aliases excluded from module lifecycle enumeration."""
        return {}

    @property
    def declared_components(self) -> Mapping[str, Any]:
        """Return every declared canonical component/spec and addressable alias."""
        return {**self.canonical_components, **self.alias_components}

    @property
    def declared_component_names(self) -> List[str]:
        """Return all declared component/spec and alias names."""
        return sorted(self.declared_components)

    def physical_route(self, name: str) -> tuple[str, tuple[str, ...]]:
        """Return the physical ownership root and root-relative logical path."""
        self._validate_declared_names([name])
        return name, ()

    def load_root_remainder(
        self,
        root: str,
        *,
        excluded_paths: Sequence[tuple[str, ...]],
        device: Union[torch.device, str],
    ) -> None:
        """Move a physical root except prepared logical routes."""
        raise RuntimeError(
            f"{type(self).__name__} cannot partially move physical root={root!r}; "
            f"excluded_paths={list(excluded_paths)!r}"
        )

    @property
    def materialized_component_names(self) -> List[str]:
        """Return materialized canonical module names, excluding aliases."""
        return [
            name
            for name in sorted(self.canonical_components)
            if isinstance(self._get_materialized_component(name), torch.nn.Module)
        ]

    @property
    def component_names(self) -> List[str]:
        """Return materialized canonical module names for compatibility."""
        return self.materialized_component_names

    @property
    def text_encoder_names(self) -> List[str]:
        """Return declared text encoder component and alias names."""
        return self._role_component_names("text_encoder")

    @property
    def transformer_names(self) -> List[str]:
        """Return declared transformer component and alias names."""
        return self._role_component_names("transformer")

    @property
    def prepared_components(self) -> Dict[str, Any]:
        """Return the component override mapping under its compatibility name."""
        return self.override_components

    def set_component_override(self, name: str, module: Any) -> None:
        """Install a component override excluded from runtime device management.

        Overrides include accelerator-prepared modules, routed proxies, LoRA wrappers,
        and checkpoint replacement modules.

        Args:
            name: Component name to override.
            module: Replacement module or routed proxy.

        Raises:
            ValueError: If the name is empty, undeclared, or the replacement is ``None``.
        """
        if not name:
            raise ValueError("Component override name must be non-empty.")
        if module is None:
            raise ValueError(
                f"Component override '{name}' must be a module or routed proxy, got None."
            )
        # An override under an undeclared name is unreachable: every reader resolves
        # through the declared set, so the module would be installed and never used.
        self._validate_declared_names([name])
        self.override_components[name] = module

    def has_component_override(self, name: str) -> bool:
        """Return whether a component has a runtime override.

        Args:
            name: Component name.

        Returns:
            True when an override is installed.
        """
        return name in self.override_components

    def set_prepared_component(self, name: str, module: Any) -> None:
        """Install a component override under the prepared compatibility API.

        Args:
            name: Component name to override.
            module: Replacement module or routed proxy.

        Raises:
            ValueError: If the name is empty, undeclared, or the replacement is ``None``.
        """
        self.set_component_override(name, module)

    def is_prepared(self, name: str) -> bool:
        """Return whether a component has an override under the compatibility API.

        Args:
            name: Component name.

        Returns:
            True when an override is installed.
        """
        return self.has_component_override(name)

    def get_component(self, name: str) -> Any:
        """Return a component with override-over-canonical precedence.

        Args:
            name: Canonical component name.

        Returns:
            Runtime override when present, otherwise the canonical component.

        Raises:
            ValueError: If the component name is unknown.
            RuntimeError: If a required canonical component remains unavailable.
        """
        if self.has_component_override(name):
            return self.override_components[name]
        return self.get_canonical_component(name)

    def get_canonical_component(self, name: str) -> Any:
        """Return the canonical backend component, materializing it if needed.

        Args:
            name: Canonical component name.

        Returns:
            Canonical backend component.

        Raises:
            ValueError: If the component name is unknown.
            RuntimeError: If a required component remains unavailable.
        """
        component = self._get_materialized_component(name)
        if component is not None:
            return component

        self._validate_declared_names([name])
        self._materialize_components([name])
        component = self._get_materialized_component(name)
        if component is None:
            if self._allows_none_component(name):
                return None
            raise RuntimeError(
                f"Canonical component '{name}' remained unavailable after materialization; "
                f"expected={[name]}, received={self._materialized_component_names()}."
            )
        return component

    def resolve_component_names(
        self,
        components: Optional[Union[str, List[str]]] = None,
    ) -> List[str]:
        """Resolve concrete and group component names.

        Args:
            components: Component name, names, or ``None`` for the materialized canonical
                modules. The groups ``text_encoders`` and ``transformers`` expand to their
                declared members.

        Returns:
            Deduplicated concrete component names in request order.

        Raises:
            ValueError: If an explicit component name is unknown.
        """
        if components is None:
            return self.materialized_component_names
        requested = [components] if isinstance(components, str) else components
        resolved: List[str] = []
        for name in requested:
            if name == "text_encoders":
                resolved.extend(self.text_encoder_names)
            elif name == "transformers":
                resolved.extend(self.transformer_names)
            else:
                resolved.append(name)
        resolved = list(dict.fromkeys(resolved))
        self._validate_declared_names(resolved)
        return resolved

    def materialize_components(
        self,
        components: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Materialize requested non-overridden canonical components.

        Args:
            components: Component name, names, groups, or ``None`` for only the
                canonical modules that are already materialized.

        Raises:
            ValueError: If an explicit component name is unknown.
            RuntimeError: If a required component cannot be materialized.
        """
        names = [
            name
            for name in self.resolve_component_names(components)
            if not self.has_component_override(name)
        ]
        if not names:
            return
        self._materialize_components(names)

    def load_stage_components(
        self,
        components: Optional[Union[str, List[str]]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Materialize and move non-prepared components to a stage device.

        Args:
            components: Component name, names, groups, or ``None`` for the materialized
                canonical modules.
            device: Target device.

        Raises:
            ValueError: If a component name is unknown or ``device`` is ``None``.
            RuntimeError: If a required component cannot be materialized.
        """
        names = [
            name
            for name in self.resolve_component_names(components)
            if self._owns_device_lifecycle(name)
        ]
        if names and device is None:
            raise ValueError(
                f"Component runtime device must not be None when loading components {names}."
            )
        self.materialize_components(names)
        for name in names:
            component = self._get_materialized_component(name)
            if component is None:
                if self._allows_none_component(name):
                    continue
                raise RuntimeError(
                    f"Cannot load component '{name}' because it is not materialized; "
                    f"expected={[name]}, received={self._materialized_component_names()}."
                )
            if hasattr(component, "to"):
                component.to(device)

    def unload_stage_components(
        self,
        components: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Move materialized non-prepared components to CPU.

        Args:
            components: Component name, names, groups, or ``None`` for the materialized
                canonical modules.

        Raises:
            ValueError: If an explicit component name is unknown.
        """
        names = [
            name
            for name in self.resolve_component_names(components)
            if self._owns_device_lifecycle(name)
        ]
        for name in names:
            component = self._get_materialized_component(name)
            if component is not None and hasattr(component, "to"):
                component.to("cpu")

    def load_components(
        self,
        components: Optional[Union[str, List[str]]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Load stage components; compatibility alias of ``load_stage_components``.

        Args:
            components: Component name, names, groups, or ``None`` for the materialized
                canonical modules.
            device: Target device.

        Raises:
            ValueError: If a component name is unknown or ``device`` is ``None``.
            RuntimeError: If a required component cannot be materialized.
        """
        self.load_stage_components(components, device)

    def unload_components(
        self,
        components: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Unload stage components; compatibility alias of ``unload_stage_components``.

        Args:
            components: Component name, names, groups, or ``None`` for the materialized
                canonical modules.

        Raises:
            ValueError: If an explicit component name is unknown.
        """
        self.unload_stage_components(components)

    @abstractmethod
    def _get_materialized_component(self, name: str) -> Any:
        """Return a canonical component without causing materialization."""
        pass

    @abstractmethod
    def _materialize_components(self, names: List[str]) -> None:
        """Materialize canonical components for the backend."""
        pass

    def _materialized_component_names(self) -> List[str]:
        """Return names that currently resolve to materialized components."""
        return self.materialized_component_names

    def _allows_none_component(self, name: str) -> bool:
        """Return whether an explicitly requested declared component may be ``None``."""
        return False

    def _role_component_names(self, role: str) -> List[str]:
        """Return non-``None`` declared names matching a component role."""
        components = self.declared_components
        return [
            name for name in sorted(components) if role in name and components[name] is not None
        ]

    def _owns_device_lifecycle(self, name: str) -> bool:
        """Return whether runtime stage lifecycle owns a component's device."""
        return not self.has_component_override(name) and name not in self.alias_components

    def _validate_declared_names(self, names: List[str]) -> None:
        """Fail fast when explicitly requested names are not declared."""
        unknown = [name for name in names if name not in self.declared_components]
        if unknown:
            raise ValueError(
                "Cannot resolve unknown components; "
                f"expected={unknown}, received={self.declared_component_names}."
            )
