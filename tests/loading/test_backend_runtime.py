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

import os
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType

from flow_factory.loading.backend import (
    FSDPBackendLoadRuntime,
    configure_backend_loading,
)
from flow_factory.loading.domain import (
    ComponentDescriptor,
    ComponentRole,
    LoadPlanner,
)


class _AcceleratorFake:
    distributed_type = DistributedType.FSDP
    device = torch.device("cpu")
    num_processes = 2
    process_index = 0

    def __init__(self, *, efficient: bool, fsdp_version: int = 2) -> None:
        self.state = SimpleNamespace(
            fsdp_plugin=SimpleNamespace(
                fsdp_version=fsdp_version,
                cpu_ram_efficient_loading=efficient,
                transformer_cls_names_to_wrap=None,
            )
        )

    def prepare(self, *objects):
        return list(objects) if len(objects) > 1 else objects[0]

    def wait_for_everyone(self) -> None:
        return None


def _plan():
    return LoadPlanner().build(
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


def test_backend_capability_disables_global_efficient_loading_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _AcceleratorFake(efficient=True)
    monkeypatch.setenv("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")
    adapter_class = type(
        "EagerAdapter",
        (),
        {"supports_fsdp2_cpu_efficient_loading": False},
    )

    configure_backend_loading(accelerator, adapter_class)

    assert not accelerator.state.fsdp_plugin.cpu_ram_efficient_loading
    assert os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] == "false"


def test_backend_capability_preserves_supported_efficient_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _AcceleratorFake(efficient=True)
    monkeypatch.setenv("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")
    adapter_class = type(
        "ModularAdapter",
        (),
        {"supports_fsdp2_cpu_efficient_loading": True},
    )

    configure_backend_loading(accelerator, adapter_class)

    assert accelerator.state.fsdp_plugin.cpu_ram_efficient_loading
    assert os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] == "true"


def test_reward_scope_restores_target_loading_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _AcceleratorFake(efficient=True)
    adapter = SimpleNamespace()
    runtime = FSDPBackendLoadRuntime(accelerator, _plan(), adapter)
    monkeypatch.setenv("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")

    with runtime.load_scope(ComponentRole.REWARD):
        assert os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] == "false"

    assert os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] == "true"


class _FSDPModuleFake(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 2)
        self.unshard_async_op_calls = []
        self.backward_prefetch_calls = []
        self.replay_hook_events = []

    def _set_unshard_async_op(self, enabled: bool) -> None:
        self.unshard_async_op_calls.append(enabled)

    def set_modules_to_backward_prefetch(self, modules) -> None:
        self.backward_prefetch_calls.append(modules)

    def unshard(self) -> None:
        self.replay_hook_events.append("unshard")


def test_prepare_enables_adapter_requested_fsdp2_default_stream_unshard() -> None:
    module = _FSDPModuleFake()
    optimizer = object()
    runtime = FSDPBackendLoadRuntime(
        _AcceleratorFake(efficient=True),
        _plan(),
        SimpleNamespace(fsdp2_use_default_stream_unshard=True),
    )

    prepared = runtime.prepare(module, optimizer)

    assert prepared == [module, optimizer]
    assert module.unshard_async_op_calls == [True]


def test_prepare_disables_default_fsdp2_backward_prefetch_recursively() -> None:
    module = _FSDPModuleFake()
    module.child = _FSDPModuleFake()
    runtime = FSDPBackendLoadRuntime(
        _AcceleratorFake(efficient=True),
        _plan(),
        SimpleNamespace(fsdp2_disable_backward_prefetch=True),
    )

    assert runtime.prepare(module) is module
    assert module.backward_prefetch_calls == [[module]]
    assert module.child.backward_prefetch_calls == [[module.child]]


def test_prepare_extends_fsdp2_wrap_policy_before_distributed_preparation() -> None:
    module = _FSDPModuleFake()
    module._no_split_modules = ["TransformerBlock"]
    accelerator = _AcceleratorFake(efficient=True)
    runtime = FSDPBackendLoadRuntime(
        accelerator,
        _plan(),
        SimpleNamespace(
            fsdp2_additional_wrap_module_names=("ChunkedFeedForward",),
        ),
    )

    assert runtime.prepare(module) is module
    assert accelerator.state.fsdp_plugin.transformer_cls_names_to_wrap == [
        "TransformerBlock",
        "ChunkedFeedForward",
    ]


def test_prepare_delegates_fsdp2_activation_checkpointing_during_prepare() -> None:
    module = _FSDPModuleFake()
    module.child = _FSDPModuleFake()
    module.register_forward_pre_hook(
        lambda _module, _inputs: module.replay_hook_events.append("fsdp-pre-forward")
    )
    module.child.register_forward_pre_hook(
        lambda _module, _inputs: module.child.replay_hook_events.append("fsdp-pre-forward")
    )
    module._no_split_modules = ["TransformerBlock"]
    accelerator = _AcceleratorFake(efficient=True)
    plugin = accelerator.state.fsdp_plugin
    plugin.activation_checkpointing = True
    prepare_observations = []
    accelerator.prepare = (
        lambda *objects: prepare_observations.append(
            (
                plugin.activation_checkpointing,
                tuple(plugin.transformer_cls_names_to_wrap),
            )
        )
        or objects[0]
    )
    configured_roots = []
    runtime = FSDPBackendLoadRuntime(
        accelerator,
        _plan(),
        SimpleNamespace(
            fsdp2_additional_wrap_module_names=("ChunkedFeedForward",),
            fsdp2_use_in_forward_activation_checkpointing=True,
            configure_fsdp2_in_forward_activation_checkpointing=lambda root: (
                configured_roots.append(root) or 2
            ),
        ),
    )

    assert runtime.prepare(module) is module
    assert configured_roots == [module]
    assert prepare_observations == [
        (False, ("TransformerBlock", "ChunkedFeedForward")),
    ]
    assert plugin.activation_checkpointing is True
    module(torch.ones(1, 2))
    assert module.replay_hook_events == ["fsdp-pre-forward", "unshard"]
    module.child(torch.ones(1, 2))
    assert module.child.replay_hook_events == ["fsdp-pre-forward", "unshard"]


def test_prepare_replay_unshard_hook_is_idempotent() -> None:
    module = _FSDPModuleFake()
    accelerator = _AcceleratorFake(efficient=True)
    accelerator.state.fsdp_plugin.activation_checkpointing = True
    runtime = FSDPBackendLoadRuntime(
        accelerator,
        _plan(),
        SimpleNamespace(
            fsdp2_use_in_forward_activation_checkpointing=True,
            configure_fsdp2_in_forward_activation_checkpointing=lambda _root: 1,
        ),
    )

    assert runtime.prepare(module) is module
    assert runtime.prepare(module) is module
    module(torch.ones(1, 2))

    assert module.replay_hook_events == ["unshard"]


@pytest.mark.parametrize(
    ("fsdp_version", "requested"),
    [(1, True), (2, False)],
)
def test_prepare_preserves_default_unshard_without_fsdp2_adapter_opt_in(
    fsdp_version: int,
    requested: bool,
) -> None:
    module = _FSDPModuleFake()
    runtime = FSDPBackendLoadRuntime(
        _AcceleratorFake(efficient=True, fsdp_version=fsdp_version),
        _plan(),
        SimpleNamespace(fsdp2_use_default_stream_unshard=requested),
    )

    assert runtime.prepare(module) is module
    assert module.unshard_async_op_calls == []


def test_prepare_rejects_missing_fsdp2_default_stream_unshard_api() -> None:
    runtime = FSDPBackendLoadRuntime(
        _AcceleratorFake(efficient=True),
        _plan(),
        SimpleNamespace(fsdp2_use_default_stream_unshard=True),
    )

    with pytest.raises(TypeError, match="requires prepared modules.*_set_unshard_async_op"):
        runtime.prepare(torch.nn.Linear(2, 2))


def test_physical_target_root_is_not_selected_as_auxiliary() -> None:
    adapter = SimpleNamespace(
        _resolve_component_names=lambda components: list(components),
    )
    runtime = FSDPBackendLoadRuntime(
        _AcceleratorFake(efficient=True),
        _plan(),
        adapter,
    )

    assert runtime._logical_names(
        ["bagel", "vae"],
        role=ComponentRole.AUXILIARY,
    ) == ["vae"]


def test_target_bootstrap_broadcasts_buffers_not_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.nn.Linear(2, 2)
    target.register_buffer("rope", torch.ones(2))
    vae = torch.nn.Linear(2, 2)
    broadcasts = []
    monkeypatch.setattr(
        "flow_factory.loading.backend.dist.broadcast",
        lambda tensor, src: broadcasts.append(tensor.clone()),
    )
    monkeypatch.setattr(
        "flow_factory.loading.backend.dist.all_reduce",
        lambda tensor, op: None,
    )
    adapter = SimpleNamespace(
        get_component=lambda name: {"bagel": target, "vae": vae}[name],
    )
    runtime = FSDPBackendLoadRuntime(
        _AcceleratorFake(efficient=True),
        _plan(),
        adapter,
    )

    runtime.bootstrap_targets()

    assert len(broadcasts) == 1
    assert torch.equal(broadcasts[0], target.rope)


def test_target_bootstrap_raises_before_buffer_broadcast_on_meta_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.nn.Module()
    target.register_buffer("rope", torch.empty(2, device="meta"))
    monkeypatch.setattr(
        "flow_factory.loading.backend.dist.all_reduce",
        lambda tensor, op: None,
    )
    monkeypatch.setattr(
        "flow_factory.loading.backend.dist.broadcast",
        lambda tensor, src: pytest.fail("invalid ranks must not enter buffer broadcast"),
    )
    runtime = FSDPBackendLoadRuntime(
        _AcceleratorFake(efficient=True),
        _plan(),
        SimpleNamespace(get_component=lambda name: target),
    )

    with pytest.raises(RuntimeError, match=r"target root='bagel'.*meta buffer"):
        runtime.bootstrap_targets()


def test_replicated_root_fingerprint_is_cached_after_first_stage() -> None:
    adapter = SimpleNamespace(
        _resolve_component_names=lambda components: list(components),
    )
    runtime = FSDPBackendLoadRuntime(
        _AcceleratorFake(efficient=False),
        _plan(),
        adapter,
    )
    calls = []
    runtime._check_replica_fingerprints = lambda roots: calls.append(list(roots))

    runtime.components_loaded(["vae"])
    runtime.components_loaded(["vae"])

    assert calls == [["vae"]]
