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


@pytest.mark.parametrize("torch_version", ["2.10.0", "2.11.0+cu128"])
def test_prepare_delegates_fsdp2_activation_checkpointing_during_prepare(
    monkeypatch, torch_version
) -> None:
    monkeypatch.setattr(torch, "__version__", torch_version)
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
    expected = ["fsdp-pre-forward"] + (["unshard"] if torch_version == "2.10.0" else [])
    assert module.replay_hook_events == expected
    module.child(torch.ones(1, 2))
    assert module.child.replay_hook_events == expected


def test_prepare_replay_unshard_hook_is_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(torch, "__version__", "2.10.0")
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


@pytest.mark.parametrize(
    ("fsdp_version", "device", "expected_releases"),
    [(2, "cuda", 1), (1, "cuda", 0), (2, "cpu", 0)],
)
def test_prepare_releases_fsdp2_preparation_cache_on_cuda_only(
    monkeypatch: pytest.MonkeyPatch,
    fsdp_version: int,
    device: str,
    expected_releases: int,
) -> None:
    releases = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: releases.append("released"))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device=None: 0)
    accelerator = _AcceleratorFake(efficient=True, fsdp_version=fsdp_version)
    accelerator.device = torch.device(device)
    module = _FSDPModuleFake()
    runtime = FSDPBackendLoadRuntime(accelerator, _plan(), SimpleNamespace())

    assert runtime.prepare(module) is module
    assert releases == ["released"] * expected_releases


def test_prepare_releases_fsdp2_cache_before_adapter_memory_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("released"))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device=None: 0)
    accelerator = _AcceleratorFake(efficient=True)
    accelerator.device = torch.device("cuda")
    accelerator.prepare = lambda module: events.append("prepared") or module
    module = _FSDPModuleFake()
    module._set_unshard_async_op = lambda enabled: events.append(f"default-stream={enabled}")
    runtime = FSDPBackendLoadRuntime(
        accelerator,
        _plan(),
        SimpleNamespace(fsdp2_use_default_stream_unshard=True),
    )

    assert runtime.prepare(module) is module
    assert events == ["prepared", "released", "default-stream=True"]


class _MetaPrepareAccelerator(_AcceleratorFake):
    """Emulates Accelerate's FSDP2 prepare on a meta root: every remapped optimizer
    parameter collides on ``data_ptr() == 0`` and lands on the same parameter."""

    def __init__(self) -> None:
        super().__init__(efficient=True)
        self.efficient_during_prepare = None

    def prepare(self, module, optimizer):
        self.efficient_during_prepare = self.state.fsdp_plugin.cpu_ram_efficient_loading
        collided = next(module.parameters())
        for group in optimizer.param_groups:
            group["params"] = [collided for _ in group["params"]]
        return module, optimizer

    def unwrap_model(self, module):
        return module


def test_sharded_prepare_materializes_targets_and_rebuilds_optimizer_groups() -> None:
    torch.manual_seed(0)
    bundle = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    loaded = {name: value.detach().clone() for name, value in bundle.state_dict().items()}
    accelerator = _MetaPrepareAccelerator()
    accelerator.num_processes = 1
    runtime = FSDPBackendLoadRuntime(accelerator, _plan(), SimpleNamespace())
    runtime.detach_targets(bundle, lambda name: name)
    assert all(parameter.is_meta for parameter in bundle.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [bundle[0].weight, bundle[1].weight], "role_name": "policy"},
            {"params": [bundle[0].bias, bundle[1].bias], "role_name": "policy"},
        ]
    )

    module, prepared = runtime.prepare(bundle, optimizer)

    assert accelerator.efficient_during_prepare is False
    assert accelerator.state.fsdp_plugin.cpu_ram_efficient_loading is True
    assert all(torch.equal(module.state_dict()[name], value) for name, value in loaded.items())
    assert [group["params"] for group in prepared.param_groups] == [
        [module[0].weight, module[1].weight],
        [module[0].bias, module[1].bias],
    ]


def test_sharded_prepare_rejects_optimizer_parameters_outside_the_bundle() -> None:
    bundle = torch.nn.Linear(3, 2)
    runtime = FSDPBackendLoadRuntime(_MetaPrepareAccelerator(), _plan(), SimpleNamespace())
    runtime.detach_targets(bundle, lambda name: name)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1)), bundle.weight])

    with pytest.raises(ValueError, match="1 parameters outside the detached target bundle"):
        runtime.prepare(bundle, optimizer)


def test_detach_targets_refuses_a_pending_source() -> None:
    runtime = FSDPBackendLoadRuntime(_MetaPrepareAccelerator(), _plan(), SimpleNamespace())
    runtime.detach_targets(torch.nn.Linear(3, 2), lambda name: name)

    with pytest.raises(RuntimeError, match="already detached"):
        runtime.detach_targets(torch.nn.Linear(3, 2), lambda name: name)


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
