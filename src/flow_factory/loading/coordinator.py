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

"""Compile adapter declarations and coordinate backend-owned loading."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Iterable, List, Optional, Union

from accelerate import Accelerator

from .backend import BackendLoadRuntime, build_backend_load_runtime
from .domain import (
    ComponentDescriptor,
    ComponentRole,
    LoadPlan,
    LoadPlanner,
)

_HOST_MARKERS = ("tokenizer", "processor", "scheduler")


def _expanded_names(adapter: Any, declarations: Iterable[str]) -> set[str]:
    names: set[str] = set()
    for declaration in declarations:
        names.update(adapter._resolve_component_names(declaration))
    return names


def build_adapter_load_plan(adapter: Any) -> LoadPlan:
    """Build one physical-root plan from an adapter's public lifecycle declarations."""
    runtime = adapter.component_runtime
    target_names = _expanded_names(adapter, adapter.model_args.target_components)

    descriptors = []
    for name in runtime.declared_component_names:
        root, path = runtime.physical_route(name)
        if name in target_names:
            role = ComponentRole.TARGET
        elif any(marker in name for marker in _HOST_MARKERS):
            role = ComponentRole.HOST
        else:
            role = ComponentRole.AUXILIARY

        descriptors.append(
            ComponentDescriptor(
                name=name,
                root=root,
                path=path,
                role=role,
            )
        )
    return LoadPlanner().build(descriptors)


class ModelLoadCoordinator:
    """Single trainer-facing facade for component and backend loading."""

    def __init__(self, adapter: Any, accelerator: Accelerator) -> None:
        self.adapter = adapter
        self.accelerator = accelerator
        self.plan = build_adapter_load_plan(adapter)
        self.backend: BackendLoadRuntime = build_backend_load_runtime(
            accelerator,
            self.plan,
            adapter,
        )

    def load_scope(self, role: ComponentRole) -> AbstractContextManager[None]:
        return self.backend.load_scope(role)

    def bootstrap_targets(self) -> None:
        self.backend.bootstrap_targets()

    def components_loaded(
        self,
        components: Optional[Union[str, List[str]]],
    ) -> None:
        self.backend.components_loaded(components)

    def load_components(
        self,
        components: Optional[Union[str, List[str]]],
        *,
        device: Any,
    ) -> None:
        """Load replicas and target-owned auxiliary remainders without moving targets.

        Only replicas observed as materialized are finalized through the backend.

        Args:
            components: Logical components requested by the adapter lifecycle call.
            device: Destination device forwarded to component materialization.
        """
        requested = self.adapter._resolve_component_names(components)
        replicated = [
            name
            for name in requested
            if self.plan.request_for_component(name).role is not ComponentRole.TARGET
        ]
        remainder_roots = list(
            dict.fromkeys(
                self.plan.descriptors[name].root
                for name in requested
                if self.plan.request_for_component(name).role is ComponentRole.TARGET
                and self.plan.descriptors[name].role is ComponentRole.AUXILIARY
            )
        )
        if not replicated and not remainder_roots:
            return
        loaded_replicated: List[str] = []
        with self.load_scope(ComponentRole.AUXILIARY):
            if replicated:
                self.adapter.on_load_components(components=replicated, device=device)
                materialized_roots = set(
                    self.adapter.component_runtime.materialized_component_names
                )
                loaded_replicated = [
                    name
                    for name in replicated
                    if self.plan.descriptors[name].root in materialized_roots
                ]
            for root in remainder_roots:
                request = self.plan.request_for_root(root)
                excluded_paths = [
                    descriptor.path
                    for descriptor in request.descriptors.values()
                    if descriptor.role is ComponentRole.TARGET
                ]
                self.adapter.component_runtime.load_root_remainder(
                    root,
                    excluded_paths=excluded_paths,
                    device=device,
                )
        if loaded_replicated:
            self.components_loaded(loaded_replicated)

    def prepare(self, *objects: Any) -> Any:
        return self.backend.prepare(*objects)
