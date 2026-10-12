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

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import torch

from flow_factory.loading import ComponentDescriptor, ComponentRole, LoadPlanner
from flow_factory.loading.coordinator import ModelLoadCoordinator, build_adapter_load_plan
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.runtime import ClassicPipelineRuntime, PseudoPipelineRuntime


def test_coordinator_promotes_pseudo_alias_physical_root_to_target() -> None:
    bagel = torch.nn.Linear(2, 2)
    vae = torch.nn.Linear(2, 2)
    runtime = PseudoPipelineRuntime(
        SimpleNamespace(),
        {"bagel": bagel, "vae": vae},
        aliases={"transformer": bagel},
        alias_routes={"transformer": ("bagel", ("language_model",))},
    )
    adapter = SimpleNamespace(
        component_runtime=runtime,
        model_args=SimpleNamespace(target_components=["transformer"]),
        preprocessing_modules=["vae"],
        inference_modules=["transformer", "vae"],
        _resolve_component_names=runtime.resolve_component_names,
    )

    plan = build_adapter_load_plan(adapter)

    assert plan.request_for_component("transformer").root == "bagel"
    assert plan.request_for_component("bagel").role is ComponentRole.TARGET
    assert plan.request_for_component("vae").role is ComponentRole.AUXILIARY


def test_classic_runtime_collapses_same_object_aliases() -> None:
    encoder = torch.nn.Linear(2, 2)
    pipeline = SimpleNamespace(
        components={"text_encoder_2": encoder},
        text_encoder_2=encoder,
        text_encoder=encoder,
    )
    runtime = ClassicPipelineRuntime(pipeline)

    assert runtime.physical_route("text_encoder") == ("text_encoder_2", ())


def test_classic_runtime_keeps_absent_optional_components_as_distinct_roots() -> None:
    pipeline = SimpleNamespace(
        components={"image_encoder": None, "image_processor": None},
        image_encoder=None,
        image_processor=None,
    )
    runtime = ClassicPipelineRuntime(pipeline)
    adapter = SimpleNamespace(
        component_runtime=runtime,
        model_args=SimpleNamespace(target_components=[]),
        _resolve_component_names=runtime.resolve_component_names,
    )

    plan = build_adapter_load_plan(adapter)

    assert runtime.physical_route("image_encoder") == ("image_encoder", ())
    assert runtime.physical_route("image_processor") == ("image_processor", ())
    assert plan.request_for_component("image_encoder").role is ComponentRole.AUXILIARY
    assert plan.request_for_component("image_processor").role is ComponentRole.HOST


def test_coordinator_expands_target_component_groups() -> None:
    transformer = torch.nn.Linear(2, 2)
    runtime = PseudoPipelineRuntime(
        SimpleNamespace(),
        {"transformer": transformer},
    )
    adapter = SimpleNamespace(
        component_runtime=runtime,
        model_args=SimpleNamespace(target_components=["transformers"]),
        preprocessing_modules=[],
        inference_modules=["transformer"],
        _resolve_component_names=runtime.resolve_component_names,
    )

    plan = build_adapter_load_plan(adapter)

    assert plan.request_for_component("transformer").role is ComponentRole.TARGET


def test_base_freezing_uses_materialized_physical_roots_then_logical_target() -> None:
    root = torch.nn.Module()
    root.language_model = torch.nn.Linear(2, 2)
    root.vision_model = torch.nn.Linear(2, 2)
    vae = torch.nn.Linear(2, 2)
    runtime = PseudoPipelineRuntime(
        SimpleNamespace(),
        {"bagel": root, "vae": vae},
        aliases={"transformer": root.language_model},
        alias_routes={"transformer": ("bagel", ("language_model",))},
    )
    adapter = SimpleNamespace(
        component_runtime=runtime,
        model_args=SimpleNamespace(
            target_components=["transformer"],
            finetune_type="full",
        ),
        target_module_map={"transformer": "all"},
        has_component=lambda name: name in runtime.declared_component_names,
        _require_component=lambda name: runtime.get_component(name),
    )
    adapter._freeze_component = MethodType(BaseAdapter._freeze_component, adapter)

    BaseAdapter._freeze_components(adapter)

    assert all(parameter.requires_grad for parameter in root.language_model.parameters())
    assert not any(parameter.requires_grad for parameter in root.vision_model.parameters())
    assert not any(parameter.requires_grad for parameter in vae.parameters())


def test_coordinator_moves_only_auxiliary_remainder_of_target_owned_root() -> None:
    plan = LoadPlanner().build(
        [
            ComponentDescriptor(
                name="bagel",
                root="bagel",
                role=ComponentRole.AUXILIARY,
            ),
            ComponentDescriptor(
                name="transformer",
                root="bagel",
                path=("language_model",),
                role=ComponentRole.TARGET,
            ),
            ComponentDescriptor(
                name="vae",
                root="vae",
                role=ComponentRole.AUXILIARY,
            ),
        ]
    )
    calls = []
    coordinator = object.__new__(ModelLoadCoordinator)
    coordinator.plan = plan
    coordinator.adapter = SimpleNamespace(
        _resolve_component_names=lambda components: list(components),
        on_load_components=lambda components, device: calls.append(("load", components, device)),
        component_runtime=SimpleNamespace(
            materialized_component_names=["bagel", "vae"],
            load_root_remainder=lambda root, excluded_paths, device: calls.append(
                ("remainder", root, excluded_paths, device)
            ),
        ),
    )
    coordinator.load_scope = lambda role: nullcontext()
    coordinator.components_loaded = lambda components: calls.append(("finalize", components))

    coordinator.load_components(
        ["bagel", "transformer", "vae"],
        device=torch.device("cpu"),
    )

    assert calls == [
        ("load", ["vae"], torch.device("cpu")),
        (
            "remainder",
            "bagel",
            [("language_model",)],
            torch.device("cpu"),
        ),
        ("finalize", ["vae"]),
    ]


def test_coordinator_finalizes_only_materialized_replicas() -> None:
    plan = LoadPlanner().build(
        [
            ComponentDescriptor(
                name="image_encoder",
                root="image_encoder",
                role=ComponentRole.AUXILIARY,
            ),
            ComponentDescriptor(
                name="vae",
                root="vae",
                role=ComponentRole.AUXILIARY,
            ),
        ]
    )
    calls = []
    coordinator = object.__new__(ModelLoadCoordinator)
    coordinator.plan = plan
    coordinator.adapter = SimpleNamespace(
        _resolve_component_names=lambda components: list(components),
        on_load_components=lambda components, device: calls.append(("load", components, device)),
        component_runtime=SimpleNamespace(materialized_component_names=["vae"]),
    )
    coordinator.load_scope = lambda role: nullcontext()
    coordinator.components_loaded = lambda components: calls.append(("finalize", components))

    coordinator.load_components(
        ["image_encoder", "vae"],
        device=torch.device("cpu"),
    )

    assert calls == [
        ("load", ["image_encoder", "vae"], torch.device("cpu")),
        ("finalize", ["vae"]),
    ]


def test_pseudo_runtime_does_not_move_excluded_target_submodule() -> None:
    class TrackingModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moves = []

        def to(self, device):
            self.moves.append(device)
            return self

    root = torch.nn.Module()
    root.language_model = TrackingModule()
    root.latent_pos_embed = TrackingModule()
    runtime = PseudoPipelineRuntime(
        SimpleNamespace(),
        {"bagel": root},
        aliases={"transformer": root.language_model},
        alias_routes={"transformer": ("bagel", ("language_model",))},
    )

    runtime.load_root_remainder(
        "bagel",
        excluded_paths=[("language_model",)],
        device="cuda",
    )

    assert root.language_model.moves == []
    assert root.latent_pos_embed.moves == ["cuda"]

    root.latent_pos_embed.moves.clear()
    runtime.load_root_remainder(
        "bagel",
        excluded_paths=[()],
        device="cpu",
    )
    assert root.latent_pos_embed.moves == []
