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

import copy
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch
from accelerate import FullyShardedDataParallelPlugin
from peft import LoraConfig, get_peft_model

from flow_factory.models.abc import BaseAdapter
from flow_factory.models.model_bundle import ModelBundle, RoutedComponentProxy
from flow_factory.models.variants import DEFAULT_BASE_VARIANT as BASE_VARIANT
from flow_factory.models.variants import (
    ComponentVariantRegistry,
    ComponentVariantSpec,
    VariantParameter,
)


def test_model_bundle_exposes_diffusers_repeated_blocks_to_fsdp() -> None:
    class RepeatedBlock(torch.nn.Module):
        pass

    class RepeatedBlockModel(torch.nn.Module):
        _repeated_blocks = ["RepeatedBlock"]

        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([RepeatedBlock(), RepeatedBlock()])

    bundle = ModelBundle({"transformer": RepeatedBlockModel()})
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
    )

    plugin.set_auto_wrap_policy(bundle)

    assert bundle._no_split_modules == ["RepeatedBlock"]
    assert plugin.auto_wrap_policy.keywords["transformer_layer_cls"] == {RepeatedBlock}


def _registry() -> ComponentVariantRegistry:
    return ComponentVariantRegistry(
        SimpleNamespace(trainable_component_names=["transformer", "transformer_2"])
    )


def _base_spec() -> ComponentVariantSpec:
    return ComponentVariantSpec(
        name=BASE_VARIANT,
        storage_mode="lora",
        component_routes={
            "transformer": "transformer",
            "transformer_2": "transformer_2",
        },
        adapter_name=BASE_VARIANT,
    )


def _declare_base(registry: ComponentVariantRegistry) -> torch.nn.Parameter:
    registry.declare(_base_spec())
    parameter = torch.nn.Parameter(torch.ones(1))
    registry.register_parameter(BASE_VARIANT, "transformer", "weight", parameter)
    return parameter


def test_declares_roles_in_exact_order_and_resolves_routes() -> None:
    registry = _registry()
    _declare_base(registry)
    registry.declare(
        ComponentVariantSpec(
            name="fake",
            storage_mode="full",
            component_routes={
                "transformer": "fake__transformer",
                "transformer_2": "fake__transformer_2",
            },
        )
    )
    fake_parameter = torch.nn.Parameter(torch.zeros(1))
    registry.register_parameter("fake", "transformer", "weight", fake_parameter)

    assert registry.variant_names == (BASE_VARIANT, "fake")
    assert registry.resolve_route(BASE_VARIANT, "transformer") == "transformer"
    assert registry.resolve_route("fake", "transformer_2") == "fake__transformer_2"
    assert registry.parameters("fake") == (fake_parameter,)
    assert registry.parameter_records("fake")[0].parameter_name == "weight"


def test_the_first_declared_variant_must_own_the_canonical_routes() -> None:
    """The base is positional, so whatever is declared first must be the base."""
    registry = _registry()

    with pytest.raises(ValueError, match="base variant 'fake'.*canonical route"):
        registry.declare(
            ComponentVariantSpec(
                name="fake",
                storage_mode="full",
                component_routes={"transformer": "fake__transformer"},
            )
        )


def test_rejects_duplicate_and_illegal_variant_names_with_details() -> None:
    registry = _registry()
    registry.declare(_base_spec())

    with pytest.raises(ValueError, match="already declared.*base"):
        registry.declare(_base_spec())

    # Variant names are the caller's. The model layer knows nothing about what an
    # algorithm calls its copies, so an unfamiliar name is simply accepted.
    registry.declare(
        ComponentVariantSpec(
            name="critic",
            storage_mode="full",
            component_routes={"transformer": "critic__transformer"},
        )
    )
    assert "critic" in registry.variant_names

    with pytest.raises(TypeError, match="non-empty string component variant name"):
        ComponentVariantSpec(
            name="",
            storage_mode="full",
            component_routes={"transformer": "empty__transformer"},
        )


def test_base_maps_every_trainable_component_to_canonical_route() -> None:
    registry = _registry()

    with pytest.raises(
        ValueError,
        match="base.*canonical.*transformer_2.*received",
    ):
        registry.declare(
            ComponentVariantSpec(
                name=BASE_VARIANT,
                storage_mode="lora",
                component_routes={"transformer": "renamed_transformer"},
            )
        )


def test_rejects_route_collisions_with_role_and_component_details() -> None:
    registry = _registry()
    registry.declare(_base_spec())

    with pytest.raises(
        ValueError,
        match="route collision.*transformer.*base.*transformer.*fake.*transformer_2",
    ):
        registry.declare(
            ComponentVariantSpec(
                name="fake",
                storage_mode="full",
                component_routes={"transformer_2": "transformer"},
            )
        )


def test_lora_roles_can_share_a_route_for_the_same_canonical_component() -> None:
    registry = _registry()
    registry.declare(_base_spec())

    registry.declare(
        ComponentVariantSpec(
            name="fake",
            storage_mode="lora",
            component_routes={"transformer": "transformer"},
            adapter_name="fake",
        )
    )

    assert registry.resolve_route(BASE_VARIANT, "transformer") == "transformer"
    assert registry.resolve_route("fake", "transformer") == "transformer"


@pytest.mark.parametrize("variant_name", ["fake", "surrogate"])
def test_full_trainable_roles_cannot_alias_the_base_route(variant_name: str) -> None:
    registry = _registry()
    registry.declare(_base_spec())

    with pytest.raises(
        ValueError,
        match=f"shared route.*transformer.*{variant_name}.*storage_mode.*full.*base",
    ):
        registry.declare(
            ComponentVariantSpec(
                name=variant_name,  # type: ignore[arg-type]
                storage_mode="full",
                component_routes={"transformer": "transformer"},
            )
        )


def test_only_a_named_lora_adapter_may_share_the_base_route() -> None:
    """A full copy sharing a route would silently alias the base weights."""
    registry = _registry()
    registry.declare(_base_spec())

    with pytest.raises(ValueError, match="shared route.*Only a named LoRA adapter"):
        registry.declare(
            ComponentVariantSpec(
                name="fake",
                storage_mode="full",
                component_routes={"transformer": "transformer"},
            )
        )

    lora_registry = _registry()
    lora_registry.declare(
        ComponentVariantSpec(
            name=BASE_VARIANT,
            storage_mode="lora",
            component_routes={"transformer": "transformer", "transformer_2": "transformer_2"},
            adapter_name="default",
        )
    )
    lora_registry.declare(
        ComponentVariantSpec(
            name="fake",
            storage_mode="lora",
            component_routes={"transformer": "transformer"},
            adapter_name="fake",
        )
    )

    assert lora_registry.resolve_route("fake", "transformer") == "transformer"


def test_parameter_identity_belongs_to_only_one_trainable_role() -> None:
    registry = _registry()
    base_parameter = _declare_base(registry)
    registry.declare(
        ComponentVariantSpec(
            name="fake",
            storage_mode="full",
            component_routes={"transformer": "fake__transformer"},
        )
    )

    with pytest.raises(
        ValueError,
        match="parameter identity.*base.*fake.*transformer.*weight",
    ):
        registry.register_parameter("fake", "transformer", "weight", base_parameter)


def test_a_variant_is_always_a_live_trainable_copy() -> None:
    """A frozen reference is a snapshot on the adapter, never a declared variant."""
    with pytest.raises(ValueError, match="storage_mode.*'lora', 'full'.*received 'frozen'"):
        ComponentVariantSpec(
            name="reference",
            storage_mode="frozen",  # type: ignore[arg-type]
            component_routes={"transformer": "reference__transformer"},
        )


def test_freeze_rejects_empty_trainable_role_with_variant_name() -> None:
    registry = _registry()
    registry.declare(_base_spec())

    with pytest.raises(ValueError, match="variant.*base.*at least one parameter"):
        registry.freeze()


def test_freeze_blocks_declaration_and_ownership_mutation_with_details() -> None:
    registry = _registry()
    _declare_base(registry)
    registry.freeze()

    assert registry.is_frozen
    with pytest.raises(RuntimeError, match="declare.*fake.*frozen"):
        registry.declare(
            ComponentVariantSpec(
                name="fake",
                storage_mode="full",
                component_routes={"transformer": "fake__transformer"},
            )
        )
    with pytest.raises(RuntimeError, match="register parameter.*base.*frozen"):
        registry.register_parameter(
            BASE_VARIANT,
            "transformer",
            "bias",
            torch.nn.Parameter(torch.zeros(1)),
        )


def test_spec_routes_and_metadata_are_immutable_snapshots() -> None:
    component_routes = {
        "transformer": "transformer",
        "transformer_2": "transformer_2",
    }
    spec = ComponentVariantSpec(
        name=BASE_VARIANT,
        storage_mode="lora",
        component_routes=component_routes,
        adapter_name=BASE_VARIANT,
    )
    component_routes["transformer"] = "mutated"
    with pytest.raises(TypeError):
        spec.component_routes["transformer"] = "mutated"  # type: ignore[index]

    registry = _registry()
    registry.declare(spec)
    _declare_parameter = torch.nn.Parameter(torch.ones(1))
    registry.register_parameter(BASE_VARIANT, "transformer", "weight", _declare_parameter)
    metadata = registry.metadata()
    metadata["variants"][0]["component_routes"]["transformer"] = "mutated"
    metadata["variants"].append({"name": "fake"})

    assert spec.component_routes["transformer"] == "transformer"
    assert registry.resolve_route(BASE_VARIANT, "transformer") == "transformer"
    assert registry.variant_names == (BASE_VARIANT,)
    assert registry.metadata()["variants"][0]["component_routes"]["transformer"] == "transformer"


def test_registration_validates_declared_route_and_parameter_type() -> None:
    registry = _registry()
    registry.declare(_base_spec())

    with pytest.raises(KeyError, match="base.*vae.*declared components"):
        registry.register_parameter(
            BASE_VARIANT, "vae", "weight", torch.nn.Parameter(torch.ones(1))
        )
    with pytest.raises(TypeError, match="torch.nn.Parameter.*base.*weight.*Tensor"):
        registry.register_parameter(
            BASE_VARIANT,
            "transformer",
            "weight",
            torch.ones(1),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("component_names", [7, "transformer"])
def test_registry_rejects_malformed_adapter_component_collections(component_names: object) -> None:
    adapter = SimpleNamespace(trainable_component_names=component_names)

    with pytest.raises(
        TypeError,
        match=f"non-string iterable.*{type(component_names).__name__}.*{component_names!r}",
    ):
        ComponentVariantRegistry(adapter)


@pytest.mark.parametrize(
    ("spec_kwargs", "expected_detail"),
    [
        (
            {"component_routes": {1: "transformer"}},
            "component name.*int",
        ),
        (
            {"component_routes": {"transformer": 1}},
            "route.*transformer.*int",
        ),
        (
            {"adapter_name": 1},
            "adapter_name.*int",
        ),
    ],
)
def test_spec_rejects_wrong_value_types(
    spec_kwargs: dict[str, object], expected_detail: str
) -> None:
    kwargs = {
        "name": "fake",
        "storage_mode": "full",
        "component_routes": {"transformer": "fake__transformer"},
        **spec_kwargs,
    }

    with pytest.raises(TypeError, match=expected_detail):
        ComponentVariantSpec(**kwargs)  # type: ignore[arg-type]


def test_declaration_order_is_the_callers_and_route_collisions_are_rejected() -> None:
    registry = _registry()
    registry.declare(_base_spec())
    registry.declare(
        ComponentVariantSpec(
            name="surrogate",
            storage_mode="full",
            component_routes={"transformer": "surrogate__transformer"},
        )
    )

    # Declaration order after the base is the caller's; the model layer imposes no
    # canonical sequence over names it does not understand.
    registry.declare(
        ComponentVariantSpec(
            name="fake",
            storage_mode="full",
            component_routes={"transformer": "fake__transformer"},
        )
    )
    assert registry.variant_names == (BASE_VARIANT, "surrogate", "fake")

    collision_registry = _registry()
    collision_registry.declare(_base_spec())
    with pytest.raises(ValueError, match="route collision.*shared.*transformer.*transformer_2"):
        collision_registry.declare(
            ComponentVariantSpec(
                name="fake",
                storage_mode="full",
                component_routes={
                    "transformer": "shared",
                    "transformer_2": "shared",
                },
            )
        )


def test_parameter_records_are_frozen_and_metadata_excludes_parameter_objects() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    record = VariantParameter(
        variant_name=BASE_VARIANT,
        component_name="transformer",
        parameter_name="weight",
        parameter=parameter,
    )

    with pytest.raises(FrozenInstanceError):
        record.parameter_name = "bias"  # type: ignore[misc]

    registry = _registry()
    registry.declare(_base_spec())
    registry.register_parameter(BASE_VARIANT, "transformer", "weight", parameter)
    metadata_text = repr(registry.metadata())

    assert "Parameter containing" not in metadata_text
    assert metadata_text.count("weight") == 1


class _TinyRoleModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target = torch.nn.Linear(2, 2, bias=False)
        self.frozen = torch.nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.target(inputs) + self.frozen(inputs)


class _TinyRoleAdapter(BaseAdapter):
    def __init__(self, finetune_type: str) -> None:
        self.model_args = SimpleNamespace(
            finetune_type=finetune_type, trainable_parameters_dtype=torch.float32
        )
        self.target_module_map = {"transformer": ["target"]}
        component = _TinyRoleModule()
        component.requires_grad_(False)
        if finetune_type == "lora":
            component = get_peft_model(
                component,
                LoraConfig(
                    r=1,
                    lora_alpha=1,
                    init_lora_weights="gaussian",
                    target_modules=["target"],
                ),
            )
        else:
            for parameter_name, parameter in component.named_parameters():
                parameter.requires_grad = "target" in parameter_name
        self._role_components = {"transformer": component}
        self.reference_is_active = False
        self.accelerator = SimpleNamespace(unwrap_model=lambda module: module)

    @property
    def trainable_component_names(self) -> list[str]:
        return ["transformer"]

    def get_component(self, name: str) -> torch.nn.Module:
        return self._role_components[name]

    def has_component(self, name: str) -> bool:
        return name in self._role_components

    @contextmanager
    def use_ref_parameters(self) -> Iterator[None]:
        previous_state = self.reference_is_active
        self.reference_is_active = True
        try:
            yield
        finally:
            self.reference_is_active = previous_state

    def load_pipeline(self) -> object:
        raise NotImplementedError

    def decode_latents(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def inference(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def forward(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError


def _adapter_parameter_ids(adapter: _TinyRoleAdapter) -> set[int]:
    return {id(parameter) for parameter in adapter.get_component("transformer").parameters()}


def _assert_all_owned_role_parameters_trainable(registry: ComponentVariantRegistry) -> None:
    for variant_name in registry.variant_names:
        assert registry.parameters(variant_name)
        assert all(parameter.requires_grad for parameter in registry.parameters(variant_name))


def test_lora_materialization_uses_named_adapters_and_exact_new_parameter_identity() -> None:
    adapter = _TinyRoleAdapter("lora")
    ids_before_materialization = _adapter_parameter_ids(adapter)

    adapter.declare_component_variants([BASE_VARIANT, "fake", "surrogate"])

    registry = adapter.component_variant_registry
    component = adapter.get_component("transformer")
    assert set(component.peft_config) == {"default", "fake", "surrogate"}
    assert registry.variant_names == (BASE_VARIANT, "fake", "surrogate")
    assert registry.get_spec(BASE_VARIANT).adapter_name == "default"
    assert registry.get_spec("fake").adapter_name == "fake"
    assert registry.get_spec("surrogate").adapter_name == "surrogate"
    assert registry.bundle_members() == {"transformer": component}

    base_ids = {id(parameter) for parameter in registry.parameters(BASE_VARIANT)}
    fake_ids = {id(parameter) for parameter in registry.parameters("fake")}
    surrogate_ids = {id(parameter) for parameter in registry.parameters("surrogate")}
    ids_after_materialization = _adapter_parameter_ids(adapter)
    assert base_ids <= ids_before_materialization
    assert fake_ids | surrogate_ids == ids_after_materialization - ids_before_materialization
    assert base_ids
    assert fake_ids
    assert surrogate_ids
    assert base_ids.isdisjoint(fake_ids)
    assert base_ids.isdisjoint(surrogate_ids)
    assert fake_ids.isdisjoint(surrogate_ids)
    _assert_all_owned_role_parameters_trainable(registry)


def test_rebind_parameters_tracks_backend_replacements_by_stable_name() -> None:
    """FSDP2-style replacement must not leave trainability writes on stale tensors."""
    adapter = _TinyRoleAdapter("lora")
    adapter.declare_component_variants([BASE_VARIANT, "fake", "surrogate"])
    registry = adapter.component_variant_registry
    stale_ids = {
        id(parameter)
        for variant_name in registry.variant_names
        for parameter in registry.parameters(variant_name)
    }
    prepared_component = copy.deepcopy(registry.bundle_members()["transformer"])
    prepared_component.set_adapter("default")

    registry.rebind_parameters({"transformer": prepared_component})

    live_ids = {
        id(parameter)
        for variant_name in registry.variant_names
        for parameter in registry.parameters(variant_name)
    }
    assert live_ids.isdisjoint(stale_ids)
    assert live_ids <= {id(parameter) for parameter in prepared_component.parameters()}
    _assert_all_owned_role_parameters_trainable(registry)
    registry.activate("fake")
    registry.activate(BASE_VARIANT)
    _assert_all_owned_role_parameters_trainable(registry)


def test_rebind_canonicalizes_fully_shard_and_checkpoint_wrapper_names() -> None:
    """Transparent prepare wrappers must not change logical parameter ownership."""
    canonical = "base_model.model.blocks.0.attn.to_q.lora_A.fake.weight"
    wrapped = (
        "base_model.model.blocks.0._fsdp_wrapped_module.attn."
        "_checkpoint_wrapped_module.to_q.lora_A.fake.weight"
    )

    assert ComponentVariantRegistry._canonical_prepared_parameter_name(wrapped) == canonical


def test_lora_role_contexts_activate_named_adapters_and_restore_after_exception() -> None:
    adapter = _TinyRoleAdapter("lora")
    adapter.declare_component_variants([BASE_VARIANT, "fake", "surrogate"])
    registry = adapter.component_variant_registry
    component = adapter.get_component("transformer")

    assert registry.active_variant == BASE_VARIANT
    assert component.active_adapter == "default"
    _assert_all_owned_role_parameters_trainable(registry)
    with adapter.use_component_variant("fake"):
        assert registry.active_variant == "fake"
        assert component.active_adapter == "fake"
        _assert_all_owned_role_parameters_trainable(registry)
        with pytest.raises(RuntimeError, match="inner failure"):
            with adapter.use_component_variant("surrogate"):
                assert registry.active_variant == "surrogate"
                assert component.active_adapter == "surrogate"
                _assert_all_owned_role_parameters_trainable(registry)
                with adapter.use_component_variant("surrogate"):
                    assert component.active_adapter == "surrogate"
                    _assert_all_owned_role_parameters_trainable(registry)
                _assert_all_owned_role_parameters_trainable(registry)
                raise RuntimeError("inner failure")
        assert registry.active_variant == "fake"
        assert component.active_adapter == "fake"
        _assert_all_owned_role_parameters_trainable(registry)
    assert registry.active_variant == BASE_VARIANT
    assert component.active_adapter == "default"
    _assert_all_owned_role_parameters_trainable(registry)


def test_a_frozen_reference_comes_from_the_adapter_snapshot_not_a_variant() -> None:
    """Under LoRA the pre-finetune weights are reached by disabling adapters.

    Calls the real ``BaseAdapter`` implementation rather than the fake's stub, so
    the assertion covers the path a trainer actually takes.
    """
    adapter = _TinyRoleAdapter("lora")
    adapter.declare_component_variants([BASE_VARIANT, "fake"])
    component = adapter.get_component("transformer")

    with BaseAdapter.use_ref_parameters(adapter):
        assert all(
            module.disable_adapters
            for module in component.modules()
            if hasattr(module, "disable_adapters")
        )
    assert not any(
        module.disable_adapters
        for module in component.modules()
        if hasattr(module, "disable_adapters")
    )
    with adapter.use_component_variant("fake"):
        with BaseAdapter.use_ref_parameters(adapter):
            pass
        assert adapter.component_variant_registry.active_variant == "fake"
        assert component.active_adapter == "fake"
        _assert_all_owned_role_parameters_trainable(adapter.component_variant_registry)


@pytest.mark.parametrize("required_roles", [None, BASE_VARIANT, b"base", 7, {BASE_VARIANT}])
def test_declare_component_variants_rejects_non_sequence_inputs(required_roles: object) -> None:
    adapter = _TinyRoleAdapter("full")

    with pytest.raises(
        TypeError,
        match=(
            "expected required_variants to be a non-string sequence.*"
            f"received {type(required_roles).__name__}: {required_roles!r}"
        ),
    ):
        adapter.declare_component_variants(required_roles)  # type: ignore[arg-type]


@pytest.mark.parametrize("required_roles", [None, BASE_VARIANT, b"base", 7, {BASE_VARIANT}])
def test_registry_materialize_rejects_non_sequence_inputs(required_roles: object) -> None:
    registry = _registry()

    with pytest.raises(
        TypeError,
        match=(
            "expected required_variants to be a non-string sequence.*"
            f"received {type(required_roles).__name__}: {required_roles!r}"
        ),
    ):
        registry.materialize(required_roles)  # type: ignore[arg-type]


def test_declare_component_variants_rejects_non_string_unhashable_entries() -> None:
    adapter = _TinyRoleAdapter("full")
    required_roles = [BASE_VARIANT, ["fake"]]

    with pytest.raises(
        TypeError,
        match="expected string variant name.*index 1.*received list.*fake",
    ):
        adapter.declare_component_variants(required_roles)  # type: ignore[arg-type]


def test_registry_materialize_rejects_non_string_unhashable_entries() -> None:
    registry = _registry()
    required_roles = [BASE_VARIANT, ["fake"]]

    with pytest.raises(
        TypeError,
        match="expected string variant name.*index 1.*received list.*fake",
    ):
        registry.materialize(required_roles)  # type: ignore[arg-type]


def test_full_materialization_copies_routes_and_preserves_target_module_freezing() -> None:
    adapter = _TinyRoleAdapter("full")
    base = adapter.get_component("transformer")
    base_state = {
        parameter_name: parameter.detach().clone()
        for parameter_name, parameter in base.named_parameters()
    }

    adapter.declare_component_variants([BASE_VARIANT, "fake", "surrogate"])

    registry = adapter.component_variant_registry
    members = registry.bundle_members()
    assert tuple(members) == (
        "transformer",
        "fake__transformer",
        "surrogate__transformer",
    )
    for variant_name in ("fake", "surrogate"):
        route_name = f"{variant_name}__transformer"
        replica = members[route_name]
        assert replica is not base
        assert registry.resolve_route(variant_name, "transformer") == route_name
        assert {
            parameter_name: parameter.detach()
            for parameter_name, parameter in replica.named_parameters()
        }.keys() == base_state.keys()
        for parameter_name, parameter in replica.named_parameters():
            assert torch.equal(parameter, base_state[parameter_name])
            assert parameter.requires_grad is ("target" in parameter_name)

    role_parameter_ids = [
        {id(parameter) for parameter in registry.parameters(variant_name)}
        for variant_name in (BASE_VARIANT, "fake", "surrogate")
    ]
    assert all(role_parameter_ids)
    assert role_parameter_ids[0].isdisjoint(role_parameter_ids[1])
    assert role_parameter_ids[0].isdisjoint(role_parameter_ids[2])
    assert role_parameter_ids[1].isdisjoint(role_parameter_ids[2])


def test_full_reference_is_a_snapshot_that_composes_with_variant_routing() -> None:
    """The snapshot swaps values in place, so it nests inside any active variant."""
    adapter = _TinyRoleAdapter("full")
    adapter.declare_component_variants([BASE_VARIANT, "fake"])
    registry = adapter.component_variant_registry

    with registry.use("fake"):
        with adapter.use_ref_parameters():
            assert registry.active_variant == "fake"
            assert adapter.reference_is_active
        assert registry.active_variant == "fake"
        assert not adapter.reference_is_active
    assert registry.active_variant == BASE_VARIANT


def test_role_materialization_is_one_shot_and_rejected_after_registry_freeze() -> None:
    adapter = _TinyRoleAdapter("full")
    adapter.declare_component_variants([BASE_VARIANT, "fake"])

    assert adapter.component_variant_registry.is_frozen
    with pytest.raises(RuntimeError, match="materialize.*frozen"):
        adapter.component_variant_registry.materialize([BASE_VARIANT])
    with pytest.raises(RuntimeError, match="declare.*frozen"):
        adapter.declare_component_variants([BASE_VARIANT])


def _fill_role_module(component: _TinyRoleModule, target: float, frozen: float = 0.0) -> None:
    component.target.weight.data.fill_(target)
    component.frozen.weight.data.fill_(frozen)


def test_routed_proxy_dispatches_full_roles_and_attributes_through_one_bundle() -> None:
    adapter = _TinyRoleAdapter("full")
    adapter.declare_component_variants([BASE_VARIANT, "fake", "surrogate"])
    registry = adapter.component_variant_registry
    members = registry.bundle_members()
    _fill_role_module(members["transformer"], 1.0)
    _fill_role_module(members["fake__transformer"], 2.0)
    _fill_role_module(members["surrogate__transformer"], 3.0)
    bundle = ModelBundle(members)
    proxy = RoutedComponentProxy(bundle, "transformer", registry, bundle.members)
    inputs = torch.ones(1, 2)

    assert not isinstance(proxy, torch.nn.Module)
    assert proxy.inner is members["transformer"]
    assert torch.equal(proxy(inputs), torch.full((1, 2), 2.0))
    assert proxy.target is members["transformer"].target
    with adapter.use_component_variant("fake"):
        assert proxy.inner is members["fake__transformer"]
        assert proxy.target is members["fake__transformer"].target
        assert torch.equal(proxy(inputs), torch.full((1, 2), 4.0))
        with adapter.use_component_variant("surrogate"):
            assert proxy.inner is members["surrogate__transformer"]
            assert torch.equal(proxy(inputs), torch.full((1, 2), 6.0))
        assert proxy.inner is members["fake__transformer"]
    assert proxy.inner is members["transformer"]
    assert torch.equal(proxy(inputs), torch.full((1, 2), 2.0))


def test_routed_proxy_uses_active_lora_adapter_on_canonical_bundle_route() -> None:
    adapter = _TinyRoleAdapter("lora")
    adapter.declare_component_variants([BASE_VARIANT, "fake"])
    registry = adapter.component_variant_registry
    members = registry.bundle_members()
    bundle = ModelBundle(members)
    proxy = RoutedComponentProxy(bundle, "transformer", registry, bundle.members)
    inputs = torch.ones(1, 2)
    for parameter in registry.parameters(BASE_VARIANT):
        parameter.data.fill_(1.0)
    for parameter in registry.parameters("fake"):
        parameter.data.fill_(2.0)

    assert proxy.active_adapter == "default"
    assert registry.resolve_route(BASE_VARIANT, "transformer") == "transformer"
    base_output = proxy(inputs)
    with adapter.use_component_variant("fake"):
        assert proxy.active_adapter == "fake"
        assert registry.resolve_route("fake", "transformer") == "transformer"
        assert proxy.inner is members["transformer"]
        fake_output = proxy(inputs)
        assert not torch.equal(fake_output, base_output)
    assert proxy.active_adapter == "default"
    assert torch.equal(proxy(inputs), base_output)


def test_routed_proxy_preserves_default_role_after_unknown_role_error() -> None:
    adapter = _TinyRoleAdapter("full")
    adapter.declare_component_variants([BASE_VARIANT, "fake"])
    registry = adapter.component_variant_registry
    bundle = ModelBundle(registry.bundle_members())
    proxy = RoutedComponentProxy(bundle, "transformer", registry, bundle.members)

    with pytest.raises(KeyError, match="critic.*available variants"):
        with adapter.use_component_variant("critic"):  # type: ignore[arg-type]
            pass

    assert registry.active_variant == BASE_VARIANT
    assert proxy.inner is bundle.members["transformer"]


def test_base_adapter_keeps_four_method_abstract_contract() -> None:
    assert BaseAdapter.__abstractmethods__ == {
        "load_pipeline",
        "decode_latents",
        "inference",
        "forward",
    }
