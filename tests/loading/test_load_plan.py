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

from dataclasses import FrozenInstanceError

import pytest

from flow_factory.loading import ComponentDescriptor, ComponentRole, LoadPlanner


def _component(
    name: str,
    root: str,
    *,
    path: str | tuple[str, ...] = (),
    role: ComponentRole = ComponentRole.AUXILIARY,
) -> ComponentDescriptor:
    return ComponentDescriptor(name=name, root=root, path=path, role=role)


def test_planner_emits_each_physical_root_exactly_once() -> None:
    plan = LoadPlanner().build(
        [
            _component("encoder", "model", path="encoder"),
            _component("decoder", "model", path="decoder"),
            _component("vae", "vae"),
        ]
    )

    assert plan.roots == ("model", "vae")
    assert plan.requests["model"].logical_names == ("encoder", "decoder")


def test_bagel_transformer_alias_promotes_physical_root_to_target() -> None:
    plan = LoadPlanner().build(
        [
            _component("bagel", "bagel"),
            _component(
                "transformer",
                "bagel",
                path="language_model",
                role=ComponentRole.TARGET,
            ),
        ]
    )

    request = plan.request_for_root("bagel")

    assert request.role is ComponentRole.TARGET
    assert request.routes == {
        "bagel": (),
        "transformer": ("language_model",),
    }
    assert plan.routes["transformer"] == "bagel.language_model"
    assert plan.request_for_component("transformer") is request


def test_planner_rejects_conflicting_logical_routes() -> None:
    with pytest.raises(ValueError, match=r"logical component='transformer'.*conflicting"):
        LoadPlanner().build(
            [
                _component("transformer", "first"),
                _component("transformer", "second"),
            ]
        )


def test_planner_rejects_incompatible_roles_on_one_root() -> None:
    with pytest.raises(ValueError, match=r"root='shared'.*incompatible roles"):
        LoadPlanner().build(
            [
                _component("model", "shared"),
                _component(
                    "reward",
                    "shared",
                    path="scorer",
                    role=ComponentRole.REWARD,
                ),
            ]
        )


def test_plan_is_immutable() -> None:
    descriptor = _component("vae", "vae")
    plan = LoadPlanner().build([descriptor])

    with pytest.raises(FrozenInstanceError):
        descriptor.root = "other"
    with pytest.raises(TypeError):
        plan.requests["other"] = plan.requests["vae"]
