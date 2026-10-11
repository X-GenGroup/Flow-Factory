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

"""Immutable model-variant declarations and parameter ownership."""

import copy
from collections.abc import Iterable, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Literal, Mapping, Optional, Tuple, cast

import torch
from peft import PeftModel

# A variant is a live trainable copy of the canonical components that coexists
# with the other variants: it owns parameters, gradients and at least one
# optimizer group at the same time as its siblings. Weights of the *same* copy at
# another point in time (a frozen reference, an EMA, a rollout snapshot) are not
# variants; they are value-only snapshots (on `BaseAdapter`, or the per-variant
# parameter EMAs below) installed into the live parameters on demand.
#
# Variant names are caller-chosen. The model layer holds no opinion about what a
# variant means: an algorithm that trains two copies against each other names them
# in its own vocabulary, and this module never reads those names.
VariantName = str
VariantStorageMode = Literal["lora", "full"]

# The base variant is positional, not named: whichever variant is declared first
# owns the adapter's canonical components and every later variant is layered on
# it. The model layer therefore never needs to recognise a particular name. This
# default is only what a single-policy caller passes when it has nothing to say.
DEFAULT_BASE_VARIANT: VariantName = "base"
_STORAGE_MODES: Tuple[VariantStorageMode, ...] = ("lora", "full")


@dataclass(frozen=True)
class ComponentVariantSpec:
    """Declare one immutable component variant."""

    name: VariantName
    storage_mode: VariantStorageMode
    component_routes: Mapping[str, str]
    adapter_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and detach declaration values."""
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(
                "expected a non-empty string component variant name, received "
                f"{type(self.name).__name__}: {self.name!r}"
            )
        if self.storage_mode not in _STORAGE_MODES:
            raise ValueError(
                f"expected storage_mode for variant {self.name!r} to be one of "
                f"{_STORAGE_MODES}, received {self.storage_mode!r}"
            )
        if not isinstance(self.component_routes, Mapping):
            raise TypeError(
                f"expected component_routes for variant {self.name!r} to be a mapping, "
                f"received {type(self.component_routes).__name__}: {self.component_routes!r}"
            )

        detached_routes: Dict[str, str] = {}
        for component_name, route_name in self.component_routes.items():
            if not isinstance(component_name, str):
                raise TypeError(
                    f"expected string component name for variant {self.name!r}, received "
                    f"{type(component_name).__name__}: {component_name!r}"
                )
            if not component_name:
                raise ValueError(
                    f"expected a non-empty string component name for variant {self.name!r}, "
                    f"received {component_name!r}"
                )
            if not isinstance(route_name, str):
                raise TypeError(
                    f"expected string route for variant {self.name!r} component "
                    f"{component_name!r}, received {type(route_name).__name__}: "
                    f"{route_name!r}"
                )
            if not route_name:
                raise ValueError(
                    f"expected a non-empty string route for variant {self.name!r} component "
                    f"{component_name!r}, received {route_name!r}"
                )
            detached_routes[component_name] = route_name
        if self.adapter_name is not None and not isinstance(self.adapter_name, str):
            raise TypeError(
                f"expected adapter_name for variant {self.name!r} to be None or a string, "
                f"received {type(self.adapter_name).__name__}: {self.adapter_name!r}"
            )
        if self.adapter_name == "":
            raise ValueError(
                f"expected adapter_name for variant {self.name!r} to be None or a non-empty "
                f"string, received {self.adapter_name!r}"
            )
        object.__setattr__(self, "component_routes", MappingProxyType(detached_routes))


@dataclass(frozen=True)
class VariantParameter:
    """Record one parameter owned by a component variant."""

    variant_name: VariantName
    component_name: str
    parameter_name: str
    parameter: torch.nn.Parameter


class ComponentVariantRegistry:
    """Store immutable variant declarations and identity-based parameter ownership.

    Args:
        adapter: Model adapter exposing canonical ``trainable_component_names``.
    """

    def __init__(self, adapter: "BaseAdapter") -> None:
        component_names = getattr(adapter, "trainable_component_names", None)
        if component_names is None:
            raise TypeError(
                "expected adapter to expose trainable_component_names, "
                f"received {type(adapter).__name__}"
            )
        if isinstance(component_names, (str, bytes)) or not isinstance(component_names, Iterable):
            raise TypeError(
                "expected adapter trainable_component_names to be a non-string iterable, "
                f"received {type(component_names).__name__}: {component_names!r} "
                f"from adapter type {type(adapter).__name__}"
            )
        canonical_components = tuple(component_names)
        if any(not isinstance(name, str) or not name for name in canonical_components):
            raise ValueError(
                "expected adapter trainable_component_names to contain non-empty strings, "
                f"received {canonical_components!r}"
            )
        if len(set(canonical_components)) != len(canonical_components):
            raise ValueError(
                "expected unique adapter trainable_component_names, "
                f"received {canonical_components!r}"
            )

        self._adapter = adapter
        self._canonical_components = canonical_components
        self._specs: Dict[VariantName, ComponentVariantSpec] = {}
        self._parameter_records: Dict[VariantName, List[VariantParameter]] = {}
        self._parameter_owners: Dict[int, VariantParameter] = {}
        self._route_owners: Dict[str, Tuple[VariantName, str]] = {}
        self._bundle_members: Dict[str, torch.nn.Module] = {}
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._active_variant: Optional[VariantName] = None
        self._active_context: Optional[ExitStack] = None
        self._is_frozen = False

    def declare(self, spec: ComponentVariantSpec) -> None:
        """Declare a component variant.

        Args:
            spec: Immutable variant declaration.
        """
        attempted_variant = getattr(spec, "name", "<unknown>")
        self._require_mutable(attempted_variant, "declare variant")
        if not isinstance(spec, ComponentVariantSpec):
            raise TypeError(
                "expected ComponentVariantSpec for variant declaration, "
                f"received {type(spec).__name__}: {spec!r}"
            )
        if spec.name in self._specs:
            raise ValueError(f"component variant is already declared: {spec.name!r}")

        declared_components = tuple(spec.component_routes)
        unknown_components = tuple(
            name for name in declared_components if name not in self._canonical_components
        )
        if unknown_components:
            raise ValueError(
                f"expected variant {spec.name!r} component routes to use canonical trainable "
                f"components {self._canonical_components!r}, received unknown components "
                f"{unknown_components!r}"
            )
        if not self._specs:
            expected_routes = {name: name for name in self._canonical_components}
            received_routes = dict(spec.component_routes)
            if received_routes != expected_routes:
                raise ValueError(
                    f"expected the base variant {spec.name!r} to map every canonical trainable "
                    f"component to its canonical route {expected_routes!r}, received "
                    f"{received_routes!r}"
                )

        pending_routes: Dict[str, str] = {}
        for component_name, route_name in spec.component_routes.items():
            if route_name in pending_routes:
                raise ValueError(
                    f"route collision for route {route_name!r} in variant {spec.name!r}: "
                    f"components {pending_routes[route_name]!r} and {component_name!r}"
                )
            if route_name in self._route_owners:
                owner_variant, owner_component = self._route_owners[route_name]
                if owner_component != component_name:
                    raise ValueError(
                        f"route collision for route {route_name!r}: variant {owner_variant!r} "
                        f"component {owner_component!r} already owns it; variant {spec.name!r} "
                        f"component {component_name!r} attempted to reuse it"
                    )
                if not (spec.storage_mode == "lora" and spec.adapter_name is not None):
                    raise ValueError(
                        f"shared route {route_name!r} for canonical component "
                        f"{component_name!r} is invalid for variant {spec.name!r} with "
                        f"storage_mode={spec.storage_mode!r}; route is already owned by variant "
                        f"{owner_variant!r}. Only a named LoRA adapter may share a route, "
                        "because it layers on the base weights instead of copying them."
                    )
            pending_routes[route_name] = component_name

        self._specs[spec.name] = spec
        self._parameter_records[spec.name] = []
        for route_name, component_name in pending_routes.items():
            self._route_owners.setdefault(route_name, (spec.name, component_name))

    def register_parameter(
        self,
        variant_name: VariantName,
        component_name: str,
        parameter_name: str,
        parameter: torch.nn.Parameter,
    ) -> None:
        """Register one optimizer parameter with its owning variant.

        Args:
            variant_name: Declared trainable variant.
            component_name: Canonical component containing the parameter.
            parameter_name: Name relative to the component.
            parameter: Parameter object whose identity is being registered.
        """
        self._require_mutable(variant_name, "register parameter ownership")
        spec = self._get_declared_spec(variant_name)
        if component_name not in spec.component_routes:
            raise KeyError(
                f"variant {variant_name!r} has no route for component {component_name!r}; "
                f"declared components are {tuple(spec.component_routes)!r}"
            )
        if not isinstance(parameter_name, str) or not parameter_name:
            raise ValueError(
                f"expected a non-empty parameter_name for variant {variant_name!r} component "
                f"{component_name!r}, received {parameter_name!r}"
            )
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError(
                f"expected torch.nn.Parameter for variant {variant_name!r} component "
                f"{component_name!r} parameter {parameter_name!r}, received "
                f"{type(parameter).__name__}: {parameter!r}"
            )

        existing_owner = self._parameter_owners.get(id(parameter))
        if existing_owner is not None:
            raise ValueError(
                f"parameter identity is already owned by variant {existing_owner.variant_name!r} "
                f"component {existing_owner.component_name!r} parameter "
                f"{existing_owner.parameter_name!r}; variant {variant_name!r} component "
                f"{component_name!r} parameter {parameter_name!r} attempted to register it"
            )
        record = VariantParameter(
            variant_name=variant_name,
            component_name=component_name,
            parameter_name=parameter_name,
            parameter=parameter,
        )
        self._parameter_records[variant_name].append(record)
        self._parameter_owners[id(parameter)] = record

    def materialize(self, required_variants: Sequence[VariantName]) -> None:
        """Materialize and register all requested trainable component variants.

        Args:
            required_variants: Trainable variants to materialize, base variant first.
        """
        required_variants = self._validate_required_variants(required_variants)
        self._require_mutable(required_variants, "materialize component variants")
        declared_variants = tuple(self._specs)
        if required_variants != declared_variants:
            raise ValueError(
                "expected required_variants to exactly match declared variants "
                f"{declared_variants!r}, received {required_variants!r}"
            )

        self._bundle_members = self._canonical_bundle_members()
        base_spec = self._specs[self.base_variant]
        if base_spec.storage_mode == "lora":
            self._materialize_lora_variants(required_variants)
        elif base_spec.storage_mode == "full":
            self._materialize_full_variants(required_variants)
        else:
            raise ValueError(
                f"expected base variant {self.base_variant!r} storage_mode to be "
                "'lora' or 'full', "
                f"received {base_spec.storage_mode!r}"
            )
        self.activate(self.base_variant)
        self.freeze()

    @staticmethod
    def _validate_required_variants(
        required_variants: object,
    ) -> Tuple[VariantName, ...]:
        """Validate and detach a requested trainable-variant sequence."""
        if isinstance(required_variants, (str, bytes)) or not isinstance(
            required_variants, Sequence
        ):
            raise TypeError(
                "expected required_variants to be a non-string sequence, "
                f"received {type(required_variants).__name__}: "
                f"{required_variants!r}"
            )
        for variant_index, variant_name in enumerate(required_variants):
            if not isinstance(variant_name, str):
                raise TypeError(
                    "expected string variant name in required_variants at "
                    f"index {variant_index}, received {type(variant_name).__name__}: {variant_name!r}"
                )

        required_variants = tuple(required_variants)
        if not required_variants:
            raise ValueError(
                "expected at least the base variant in required_variants, received an "
                "empty sequence"
            )
        if len(set(required_variants)) != len(required_variants):
            raise ValueError(
                "expected unique required_variants, "
                f"received duplicates in {required_variants!r}"
            )
        return cast(Tuple[VariantName, ...], required_variants)

    def bundle_members(self) -> Dict[str, torch.nn.Module]:
        """Return detached bundle-route to module mappings.

        Returns:
            Materialized modules keyed by stable bundle route.
        """
        return dict(self._bundle_members)

    def activate(self, variant_name: VariantName) -> None:
        """Activate a materialized variant.

        Args:
            variant_name: Declared variant to activate.
        """
        spec = self._get_declared_spec(variant_name)
        if not self._bundle_members:
            raise RuntimeError(
                f"cannot activate variant {variant_name!r}: component variants are not materialized"
            )

        previous_context = self._active_context
        if previous_context is not None:
            previous_context.close()
        self._active_context = None

        next_context = ExitStack()
        if self._specs[self.base_variant].storage_mode == "lora":
            if spec.adapter_name is None:
                raise ValueError(
                    f"expected named LoRA adapter for variant {variant_name!r}, "
                    f"received adapter_name={spec.adapter_name!r}"
                )
            for component_name in self._canonical_components:
                self._get_peft_component(component_name).set_adapter(spec.adapter_name)

        self._restore_trainable_variant_parameters()
        self._active_context = next_context
        self._active_variant = variant_name

    @property
    def active_variant(self) -> VariantName:
        """Return the currently active materialized variant."""
        if self._active_variant is None:
            raise RuntimeError("component variants are not materialized and no variant is active")
        return self._active_variant

    @contextmanager
    def use(self, variant_name: VariantName) -> Iterator[None]:
        """Temporarily activate a variant and restore the exact previous variant.

        Args:
            variant_name: Declared variant to use inside the context.

        Yields:
            Control while the requested variant is active.
        """
        previous_variant = self.active_variant
        self.activate(variant_name)
        try:
            yield
        finally:
            self.activate(previous_variant)

    @contextmanager
    def use_base_variant(self) -> Iterator[None]:
        """Activate the canonical base route, which is what an export writes."""
        with self.use(self.base_variant):
            yield

    def freeze(self) -> None:
        """Validate all declarations and prevent further mutation."""
        if self._is_frozen:
            return
        if not self._specs:
            raise ValueError("expected a declared base variant before freezing, received none")
        for variant_name in self._specs:
            if not self._parameter_records[variant_name]:
                raise ValueError(
                    f"variant {variant_name!r} must own at least one parameter before freeze"
                )
        self._is_frozen = True

    @property
    def is_frozen(self) -> bool:
        """Return whether the registry rejects further mutation."""
        return self._is_frozen

    @property
    def variant_names(self) -> Tuple[VariantName, ...]:
        """Return declared variant names in declaration order."""
        return tuple(self._specs)

    @property
    def base_variant(self) -> VariantName:
        """Return the variant that owns the canonical components.

        The base is whichever variant was declared first; this registry attaches no
        meaning to its name.

        Raises:
            RuntimeError: If no variant has been declared yet.
        """
        if not self._specs:
            raise RuntimeError("no component variant is declared, so there is no base variant yet")
        return next(iter(self._specs))

    def get_spec(self, variant_name: VariantName) -> ComponentVariantSpec:
        """Return an immutable variant declaration.

        Args:
            variant_name: Declared variant name.

        Returns:
            Immutable declaration for the requested variant.
        """
        return self._get_declared_spec(variant_name)

    def parameters(self, variant_name: VariantName) -> Tuple[torch.nn.Parameter, ...]:
        """Return parameters owned by a variant.

        Args:
            variant_name: Declared variant name.

        Returns:
            Parameters in registration order.
        """
        records = self.parameter_records(variant_name)
        return tuple(record.parameter for record in records)

    def parameter_records(self, variant_name: VariantName) -> Tuple[VariantParameter, ...]:
        """Return immutable parameter ownership records.

        Args:
            variant_name: Declared variant name.

        Returns:
            Ownership records in registration order.
        """
        self._get_declared_spec(variant_name)
        return tuple(self._parameter_records[variant_name])

    def rebind_parameters(self, bundle_members: Mapping[str, torch.nn.Module]) -> None:
        """Rebind ownership records after a distributed backend replaces parameters.

        FSDP2 preserves module and parameter names but swaps each original Parameter
        for a DTensor-backed Parameter during ``prepare``. Variant activation must
        restore trainability on those live objects, not on the stale pre-prepare
        identities recorded during materialization.

        Args:
            bundle_members: Prepared, unwrapped route-to-module mapping.

        Raises:
            ValueError: If a route/name is missing, duplicated, or changes shape.
            RuntimeError: If snapshots already exist and would retain stale identities.
        """
        if self._snapshots:
            raise RuntimeError(
                "cannot rebind component variant parameters after snapshots were created; "
                f"received snapshots={tuple(self._snapshots)!r}"
            )
        expected_routes = set(self._bundle_members)
        received_routes = set(bundle_members)
        if received_routes != expected_routes:
            raise ValueError(
                "expected prepared bundle routes to match materialized variant routes, "
                f"missing={tuple(sorted(expected_routes - received_routes))!r}, "
                f"unexpected={tuple(sorted(received_routes - expected_routes))!r}"
            )

        named_by_route: Dict[str, Dict[str, torch.nn.Parameter]] = {}
        for route_name, module in bundle_members.items():
            canonical_parameters: Dict[str, torch.nn.Parameter] = {}
            for parameter_name, parameter in module.named_parameters():
                canonical_name = self._canonical_prepared_parameter_name(parameter_name)
                if canonical_name in canonical_parameters:
                    raise ValueError(
                        f"prepared route {route_name!r} has duplicate canonical parameter "
                        f"name {canonical_name!r} after removing distributed wrappers"
                    )
                canonical_parameters[canonical_name] = parameter
            named_by_route[route_name] = canonical_parameters
        rebound_records: Dict[VariantName, List[VariantParameter]] = {
            variant_name: [] for variant_name in self._specs
        }
        rebound_owners: Dict[int, VariantParameter] = {}
        for variant_name, records in self._parameter_records.items():
            for record in records:
                route_name = self.resolve_route(variant_name, record.component_name)
                named_parameters = named_by_route[route_name]
                if record.parameter_name not in named_parameters:
                    lora_candidates = tuple(name for name in named_parameters if "lora_" in name)[:5]
                    raise ValueError(
                        f"prepared route {route_name!r} for variant {variant_name!r} "
                        f"component {record.component_name!r} is missing parameter "
                        f"{record.parameter_name!r}; first parameter names="
                        f"{tuple(named_parameters)[:5]!r}, first LoRA names="
                        f"{lora_candidates!r}"
                    )
                parameter = named_parameters[record.parameter_name]
                if not isinstance(parameter, torch.nn.Parameter):
                    raise TypeError(
                        f"expected prepared parameter for {variant_name!r}/"
                        f"{record.component_name!r}/{record.parameter_name!r} to be "
                        f"torch.nn.Parameter, received {type(parameter).__name__}"
                    )
                if parameter.shape != record.parameter.shape:
                    raise ValueError(
                        f"prepared parameter shape changed for {variant_name!r}/"
                        f"{record.component_name!r}/{record.parameter_name!r}: expected "
                        f"{tuple(record.parameter.shape)}, received {tuple(parameter.shape)}"
                    )
                rebound = VariantParameter(
                    variant_name=variant_name,
                    component_name=record.component_name,
                    parameter_name=record.parameter_name,
                    parameter=parameter,
                )
                existing = rebound_owners.get(id(parameter))
                if existing is not None:
                    raise ValueError(
                        "prepared parameter identity is shared by component variants: "
                        f"{existing.variant_name!r}/{existing.parameter_name!r} and "
                        f"{variant_name!r}/{record.parameter_name!r}"
                    )
                rebound_records[variant_name].append(rebound)
                rebound_owners[id(parameter)] = rebound

        self._bundle_members = dict(bundle_members)
        self._parameter_records = rebound_records
        self._parameter_owners = rebound_owners
        self.restore_trainable_parameters()

    @staticmethod
    def _canonical_prepared_parameter_name(parameter_name: str) -> str:
        """Remove transparent wrapper path segments introduced during prepare."""
        if not isinstance(parameter_name, str) or not parameter_name:
            raise ValueError(
                "expected non-empty string prepared parameter_name, "
                f"received {parameter_name!r}"
            )
        canonical = parameter_name
        for wrapper_segment in (
            "._checkpoint_wrapped_module",
            "._fsdp_wrapped_module",
        ):
            canonical = canonical.replace(wrapper_segment, "")
        return canonical

    def add_snapshot(self, variant_name: VariantName, snapshot_name: str) -> None:
        """Create a detached named parameter snapshot for one trainable variant.

        Args:
            variant_name: Trainable variant whose parameters initialize the snapshot.
            snapshot_name: Unique non-empty snapshot identifier.
        """
        self._get_declared_spec(variant_name)
        if not isinstance(snapshot_name, str) or not snapshot_name:
            raise ValueError(
                "expected non-empty string snapshot_name for parameter EMA, "
                f"received {snapshot_name!r}"
            )
        if snapshot_name in self._snapshots:
            raise ValueError(f"parameter EMA snapshot already exists: {snapshot_name!r}")
        records = self.parameter_records(variant_name)
        self._snapshots[snapshot_name] = {
            "variant_name": variant_name,
            "parameters": {
                f"{record.component_name}.{record.parameter_name}": record.parameter.detach().clone()
                for record in records
            },
            "update_count": 0,
        }

    @contextmanager
    def use_snapshot(self, snapshot_name: str) -> Iterator[None]:
        """Temporarily swap one parameter EMA into its existing component variant.

        Args:
            snapshot_name: Existing snapshot identifier.

        Yields:
            Control while the snapshot tensors are installed.
        """
        snapshot = self._get_snapshot(snapshot_name)
        variant_name = cast(VariantName, snapshot["variant_name"])
        records = self.parameter_records(variant_name)
        live_parameters = [record.parameter.detach().clone() for record in records]
        with self.use(variant_name):
            try:
                for record in records:
                    key = f"{record.component_name}.{record.parameter_name}"
                    record.parameter.data.copy_(snapshot["parameters"][key])
                yield
            finally:
                for record, live in zip(records, live_parameters):
                    record.parameter.data.copy_(live)

    def update_snapshot(self, snapshot_name: str, decay: float) -> None:
        """Update one parameter EMA from its live variant parameters.

        Args:
            snapshot_name: Existing snapshot identifier.
            decay: Finite EMA decay in ``[0, 1]``.
        """
        if isinstance(decay, bool) or not isinstance(decay, (int, float)):
            raise TypeError(
                "expected numeric decay for parameter EMA, "
                f"received {type(decay).__name__}: {decay!r}"
            )
        decay_value = float(decay)
        if not 0 <= decay_value <= 1:
            raise ValueError(f"expected parameter EMA decay in [0, 1], received {decay_value!r}")
        snapshot = self._get_snapshot(snapshot_name)
        variant_name = cast(VariantName, snapshot["variant_name"])
        for record in self.parameter_records(variant_name):
            key = f"{record.component_name}.{record.parameter_name}"
            stored = snapshot["parameters"][key]
            live = record.parameter.detach().to(device=stored.device, dtype=stored.dtype)
            stored.lerp_(live, 1 - decay_value)
        snapshot["update_count"] += 1

    def snapshot_tensors(self, snapshot_name: str) -> Tuple[torch.Tensor, ...]:
        """Return one snapshot in the owning variant's parameter registration order."""
        snapshot = self._get_snapshot(snapshot_name)
        variant_name = cast(VariantName, snapshot["variant_name"])
        return tuple(
            snapshot["parameters"][f"{record.component_name}.{record.parameter_name}"]
            .detach()
            .clone()
            for record in self.parameter_records(variant_name)
        )

    def snapshot_state_dict(self) -> Dict[str, Any]:
        """Return exact checkpoint state for parameter EMA snapshots."""
        snapshots = {
            snapshot_name: {
                "variant_name": snapshot["variant_name"],
                "parameters": {
                    name: tensor.detach().clone() for name, tensor in snapshot["parameters"].items()
                },
            }
            for snapshot_name, snapshot in self._snapshots.items()
        }
        return {
            "version": 1,
            "snapshots": snapshots,
            "update_counts": {
                snapshot_name: snapshot["update_count"]
                for snapshot_name, snapshot in self._snapshots.items()
            },
        }

    def load_snapshot_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore exact checkpoint state for declared parameter EMA snapshots.

        Args:
            state: State produced by :meth:`snapshot_state_dict`.
        """
        if not isinstance(state, Mapping):
            raise TypeError(
                "expected parameter EMA state as a mapping, "
                f"received {type(state).__name__}: {state!r}"
            )
        if set(state) != {"version", "snapshots", "update_counts"} or state["version"] != 1:
            raise ValueError(
                "expected parameter EMA state keys ('snapshots', 'update_counts', 'version') "
                f"and version 1, received keys={tuple(sorted(state))!r}, "
                f"version={state.get('version')!r}"
            )
        snapshots = state["snapshots"]
        update_counts = state["update_counts"]
        if not isinstance(snapshots, Mapping) or not isinstance(update_counts, Mapping):
            raise TypeError(
                "expected parameter EMA snapshots and update_counts as mappings, received "
                f"{type(snapshots).__name__} and {type(update_counts).__name__}"
            )
        expected_names = tuple(self._snapshots)
        if tuple(snapshots) != expected_names or tuple(update_counts) != expected_names:
            raise ValueError(
                f"expected parameter EMA snapshots {expected_names!r}, received "
                f"snapshots={tuple(snapshots)!r}, update_counts={tuple(update_counts)!r}"
            )
        for snapshot_name in expected_names:
            target = self._snapshots[snapshot_name]
            source = snapshots[snapshot_name]
            if (
                not isinstance(source, Mapping)
                or source.get("variant_name") != target["variant_name"]
            ):
                raise ValueError(
                    f"expected parameter EMA variant {target['variant_name']!r} for "
                    f"snapshot {snapshot_name!r}, received {source!r}"
                )
            source_parameters = source.get("parameters")
            if not isinstance(source_parameters, Mapping) or tuple(source_parameters) != tuple(
                target["parameters"]
            ):
                received = (
                    tuple(source_parameters)
                    if isinstance(source_parameters, Mapping)
                    else source_parameters
                )
                raise ValueError(
                    f"expected parameter names {tuple(target['parameters'])!r} for "
                    f"snapshot {snapshot_name!r}, received {received!r}"
                )
            for name, target_tensor in target["parameters"].items():
                source_tensor = source_parameters[name]
                if (
                    not isinstance(source_tensor, torch.Tensor)
                    or source_tensor.shape != target_tensor.shape
                    or source_tensor.dtype != target_tensor.dtype
                ):
                    raise ValueError(
                        f"expected tensor shape={tuple(target_tensor.shape)}, "
                        f"dtype={target_tensor.dtype} for parameter EMA "
                        f"{snapshot_name!r}/{name!r}, received {source_tensor!r}"
                    )
                target_tensor.copy_(source_tensor)
            update_count = update_counts[snapshot_name]
            if (
                not isinstance(update_count, int)
                or isinstance(update_count, bool)
                or update_count < 0
            ):
                raise ValueError(
                    f"expected non-negative int update count for parameter EMA "
                    f"{snapshot_name!r}, received {update_count!r}"
                )
            target["update_count"] = update_count

    def has_snapshot(self, snapshot_name: str) -> bool:
        """Report whether a parameter EMA snapshot has been created.

        Args:
            snapshot_name: Identifier to look up.

        Returns:
            Whether the snapshot exists.
        """
        return snapshot_name in self._snapshots

    def _get_snapshot(self, snapshot_name: str) -> Dict[str, Any]:
        """Return one parameter EMA snapshot or fail with available names."""
        if snapshot_name not in self._snapshots:
            raise KeyError(
                f"parameter EMA snapshot {snapshot_name!r} is not declared; "
                f"available snapshots are {tuple(self._snapshots)!r}"
            )
        return self._snapshots[snapshot_name]

    def resolve_route(self, variant_name: VariantName, component_name: str) -> str:
        """Resolve a canonical component to its variant-specific bundle route.

        Args:
            variant_name: Declared variant name.
            component_name: Canonical component name.

        Returns:
            Bundle route declared for the variant and component.
        """
        spec = self._get_declared_spec(variant_name)
        if component_name not in spec.component_routes:
            raise KeyError(
                f"variant {variant_name!r} has no route for component {component_name!r}; "
                f"declared components are {tuple(spec.component_routes)!r}"
            )
        return spec.component_routes[component_name]

    def metadata(self) -> Dict[str, Any]:
        """Return detached serializable variant metadata.

        Returns:
            Variant declarations and parameter names without parameter objects.
        """
        variants = []
        for variant_name, spec in self._specs.items():
            records = self._parameter_records[variant_name]
            variants.append(
                {
                    "name": spec.name,
                    "storage_mode": spec.storage_mode,
                    "component_routes": dict(spec.component_routes),
                    "adapter_name": spec.adapter_name,
                    "parameters": [
                        {
                            "component_name": record.component_name,
                            "parameter_name": record.parameter_name,
                        }
                        for record in records
                    ],
                }
            )
        return {"is_frozen": self._is_frozen, "variants": variants}

    def training_state_dict(self) -> Dict[str, Any]:
        """Return versioned variant metadata for training-state compatibility."""
        variants = []
        for variant_name, spec in self._specs.items():
            variants.append(
                {
                    "name": variant_name,
                    "storage_mode": spec.storage_mode,
                    "component_routes": dict(spec.component_routes),
                    "adapter_name": spec.adapter_name,
                    "parameters": [
                        {
                            "component_name": record.component_name,
                            "parameter_name": record.parameter_name,
                            "shape": list(record.parameter.shape),
                        }
                        for record in self._parameter_records[variant_name]
                    ],
                }
            )
        return {"version": 1, "variants": variants}

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate checkpoint variant metadata without mutating model parameters.

        Args:
            state: Versioned variant metadata produced by :meth:`training_state_dict`.
        """
        if not isinstance(state, Mapping):
            raise TypeError(
                "expected multi-variant metadata state as a mapping, "
                f"received {type(state).__name__}: {state!r}"
            )
        expected_keys = {"version", "variants"}
        received_keys = set(state)
        if received_keys != expected_keys:
            raise ValueError(
                "multi-variant metadata state keys mismatch: expected "
                f"{tuple(sorted(expected_keys))!r}, "
                f"received {tuple(sorted(received_keys))!r}"
            )
        received_version = state.get("version")
        if (
            not isinstance(received_version, int)
            or isinstance(received_version, bool)
            or received_version != 1
        ):
            raise ValueError(
                "multi-variant metadata version mismatch: expected 1, "
                f"received {received_version!r}"
            )
        received_variants = state.get("variants")
        if not isinstance(received_variants, list):
            raise TypeError(
                "expected multi-variant metadata variants as a list, "
                f"received {type(received_variants).__name__}: {received_variants!r}"
            )

        expected = self.training_state_dict()
        expected_variants = expected["variants"]
        expected_variant_order = tuple(variant["name"] for variant in expected_variants)
        received_variant_order = tuple(
            variant.get("name") if isinstance(variant, Mapping) else variant
            for variant in received_variants
        )
        if received_variant_order != expected_variant_order:
            raise ValueError(
                "multi-variant metadata variant order mismatch: expected "
                f"{expected_variant_order!r}, received {received_variant_order!r}"
            )
        for expected_variant, received_variant in zip(expected_variants, received_variants):
            variant_name = expected_variant["name"]
            if not isinstance(received_variant, Mapping):
                raise TypeError(
                    f"expected multi-variant metadata for variant {variant_name!r} as a mapping, "
                    f"received {type(received_variant).__name__}: {received_variant!r}"
                )
            expected_variant_keys = set(expected_variant)
            received_variant_keys = set(received_variant)
            if received_variant_keys != expected_variant_keys:
                raise ValueError(
                    f"multi-variant metadata keys mismatch for variant {variant_name!r}: "
                    f"expected {tuple(sorted(expected_variant_keys))!r}, "
                    f"received {tuple(sorted(received_variant_keys))!r}"
                )
            for field_name in (
                "storage_mode",
                "component_routes",
                "adapter_name",
                "parameters",
            ):
                expected_value = expected_variant[field_name]
                received_value = received_variant.get(field_name)
                if received_value != expected_value:
                    raise ValueError(
                        f"multi-variant metadata {field_name} mismatch for variant "
                        f"{variant_name!r}: expected {expected_value!r}, "
                        f"received {received_value!r}"
                    )

    def _canonical_bundle_members(self) -> Dict[str, torch.nn.Module]:
        """Collect canonical bundle members without forcing unrelated components."""
        target_module_map = getattr(self._adapter, "target_module_map", None)
        component_names = (
            tuple(target_module_map)
            if isinstance(target_module_map, Mapping)
            else self._canonical_components
        )
        members: Dict[str, torch.nn.Module] = {}
        for component_name in component_names:
            component = self._adapter.get_component(component_name)
            if not isinstance(component, torch.nn.Module):
                raise TypeError(
                    f"expected torch.nn.Module for canonical component {component_name!r}, "
                    f"received {type(component).__name__}: {component!r}"
                )
            members[component_name] = component
        return members

    def _materialize_lora_variants(self, required_variants: Tuple[VariantName, ...]) -> None:
        """Add named PEFT adapters and register exact newly created parameters."""
        base_spec = self._specs[self.base_variant]
        if base_spec.adapter_name != "default":
            raise ValueError(
                f"expected base variant {self.base_variant!r} LoRA adapter_name to be "
                "'default', "
                f"received {base_spec.adapter_name!r}"
            )
        for component_name in self._canonical_components:
            component = self._get_peft_component(component_name)
            if "default" not in component.peft_config:
                raise ValueError(
                    f"expected canonical PEFT component {component_name!r} to contain "
                    f"the existing 'default' adapter, received {tuple(component.peft_config)!r}"
                )
            component.set_adapter("default")
            for parameter_name, parameter in component.named_parameters():
                if parameter.requires_grad:
                    self.register_parameter(
                        self.base_variant, component_name, parameter_name, parameter
                    )

            for variant_name in required_variants[1:]:
                spec = self._specs[variant_name]
                if spec.adapter_name != variant_name:
                    raise ValueError(
                        f"expected LoRA adapter_name for variant {variant_name!r} to equal the variant "
                        f"name, received {spec.adapter_name!r}"
                    )
                parameter_ids_before = {id(parameter) for parameter in component.parameters()}
                component.add_adapter(
                    spec.adapter_name,
                    copy.deepcopy(component.peft_config["default"]),
                )
                new_parameters = [
                    (parameter_name, parameter)
                    for parameter_name, parameter in component.named_parameters()
                    if id(parameter) not in parameter_ids_before
                ]
                if not new_parameters:
                    raise RuntimeError(
                        f"expected add_adapter({spec.adapter_name!r}) to create parameters "
                        f"for component {component_name!r}, received none"
                    )
                for parameter_name, parameter in new_parameters:
                    self.register_parameter(variant_name, component_name, parameter_name, parameter)

    def _materialize_full_variants(self, required_variants: Tuple[VariantName, ...]) -> None:
        """Deep-copy full trainable variants and preserve target-module freezing."""
        for component_name in self._canonical_components:
            base_module = self._bundle_members[component_name]
            self._apply_full_freezing(component_name, base_module)
            self._register_trainable_parameters(self.base_variant, component_name, base_module)
            for variant_name in required_variants[1:]:
                replica = copy.deepcopy(base_module)
                self._apply_full_freezing(component_name, replica)
                route_name = self.resolve_route(variant_name, component_name)
                if route_name == component_name:
                    raise ValueError(
                        f"expected distinct full-model route for variant {variant_name!r} component "
                        f"{component_name!r}, received shared route {route_name!r}"
                    )
                self._bundle_members[route_name] = replica
                self._register_trainable_parameters(variant_name, component_name, replica)

    def _apply_full_freezing(self, component_name: str, component: torch.nn.Module) -> None:
        """Apply the adapter's configured full-model target-module map."""
        target_module_map = getattr(self._adapter, "target_module_map", None)
        if not isinstance(target_module_map, Mapping) or component_name not in target_module_map:
            raise ValueError(
                f"expected target_module_map entry for component {component_name!r}, "
                f"received {target_module_map!r}"
            )
        target_modules = target_module_map[component_name]
        component.requires_grad_(False)
        if target_modules == "all":
            component.requires_grad_(True)
            return
        if not target_modules:
            return
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        for parameter_name, parameter in component.named_parameters():
            if any(target_module in parameter_name for target_module in target_modules):
                parameter.requires_grad = True

    def _register_trainable_parameters(
        self,
        variant_name: VariantName,
        component_name: str,
        component: torch.nn.Module,
    ) -> None:
        """Register every trainable parameter of one full-component variant component."""
        for parameter_name, parameter in component.named_parameters():
            if parameter.requires_grad:
                self.register_parameter(variant_name, component_name, parameter_name, parameter)

    def restore_trainable_parameters(self) -> None:
        """Re-mark every identity-owned variant parameter as trainable.

        Public because anything that changes adapter state behind the registry's back has
        to hand this invariant back. PEFT keeps only the active adapter trainable, so a
        run with several trainable variants relies on this being reasserted after every
        such change; a variant left frozen between its forward and its backward loses
        every gradient and reports nothing, because autograd reads ``requires_grad`` when
        backward executes rather than when the graph was built.
        """
        for variant_name in self._specs:
            for record in self._parameter_records[variant_name]:
                record.parameter.requires_grad = True

    def _restore_trainable_variant_parameters(self) -> None:
        """Keep every identity-owned variant parameter trainable."""
        self.restore_trainable_parameters()

    def _get_peft_component(self, component_name: str) -> PeftModel:
        """Return one canonical PEFT component with detailed validation."""
        component = self._bundle_members[component_name]
        if not isinstance(component, PeftModel):
            raise TypeError(
                f"expected PeftModel for LoRA component {component_name!r}, "
                f"received {type(component).__name__}: {component!r}"
            )
        return component

    def _get_declared_spec(self, variant_name: VariantName) -> ComponentVariantSpec:
        """Return a declaration or fail with available variant context."""
        if variant_name not in self._specs:
            raise KeyError(
                f"component variant {variant_name!r} is not declared; available variants are {self.variant_names!r}"
            )
        return self._specs[variant_name]

    def _require_mutable(self, variant_name: object, operation: str) -> None:
        """Reject an attempted variant mutation after freeze."""
        if self._is_frozen:
            raise RuntimeError(
                f"cannot {operation} for variant {variant_name!r}: component variant registry is frozen"
            )
