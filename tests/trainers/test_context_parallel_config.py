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

"""Public CP gates reject unqualified compositions before external allocation."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flow_factory.hparams import Arguments
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter
from flow_factory.trainers.common.context_parallel import (
    ContextParallelTrainingCapability,
    validate_context_parallel_backend,
    validate_context_parallel_config,
)
from flow_factory.trainers.loader import Accelerator
from flow_factory.trainers.offline.sft import SFTTrainer


@pytest.mark.parametrize(
    "override",
    [
        {"loss_gradient": "token_local"},
        {"checkpoint_boundary": "in_flight"},
        {"version": 2},
        {"version": True},
    ],
)
def test_unimplemented_training_capability_semantics_are_rejected(override):
    with pytest.raises(ValueError, match="unsupported context-parallel training capability"):
        ContextParallelTrainingCapability(**override)


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    return Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "parallelism": {"cp_size": 2},
            "data": {
                "datasets": [{"name": "fixture", "dataset_dir": "unused", "train": {"weight": 1}}]
            },
            "model": {"model_type": "wan2_t2v", "target_components": ["transformer"]},
            "train": {"trainer_type": "sft", "gradient_accumulation_steps": 1, "ema_decay": 0},
            "log": {"save_model_only": True},
            "eval": {"eval_freq": 0},
            "acceleration": {
                "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
            },
        }
    )


def test_cp_parsing_uses_logical_replicas(config):
    assert config.data_parallel_world_size() == 2
    assert config.parallelism_args.resolve_topology(4).cp_size == 2
    validate_context_parallel_config(config, Wan2_T2V_Adapter, SFTTrainer)


@pytest.mark.parametrize(
    "filename,world,tp_size", [("wan21_cp.yaml", 4, 1), ("wan21_tp_cp.yaml", 8, 2)]
)
def test_documented_wan_example_satisfies_public_capability(monkeypatch, filename, world, tp_size):
    monkeypatch.setenv("WORLD_SIZE", str(world))
    example = Path(__file__).resolve().parents[2] / "examples/sft/lora/wan2_t2v" / filename
    config = Arguments.load_from_yaml(str(example))
    topology = config.parallelism_args.resolve_topology(world)
    assert (topology.cp_size, topology.dp_shard_size, topology.tp_size) == (2, 2, tp_size)
    assert config.data_parallel_world_size() == 2
    validate_context_parallel_config(config, Wan2_T2V_Adapter, SFTTrainer)


def test_tp_cp_requires_an_explicit_model_capability(config, monkeypatch):
    from flow_factory.models.context_parallel import ContextParallelCapability

    monkeypatch.setenv("WORLD_SIZE", "8")
    config.parallelism_args.tp_size = 2
    config.model_args.compute_precision_policy = "cross_attention_fp32_v1"
    validate_context_parallel_config(config, Wan2_T2V_Adapter, SFTTrainer)
    cp_only = SimpleNamespace(
        __name__="CPOnlyAdapter",
        context_parallel_capability=ContextParallelCapability(layout="wan_dense_video"),
    )
    with pytest.raises(ValueError, match="does not declare TP\\+CP"):
        validate_context_parallel_config(config, cp_only, SFTTrainer)


def test_tp_cp_requires_the_declared_precision_policy(config, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    config.parallelism_args.tp_size = 2
    with pytest.raises(ValueError, match="compute_precision_policy=cross_attention_fp32_v1"):
        validate_context_parallel_config(config, Wan2_T2V_Adapter, SFTTrainer)


def test_wan_heads_must_divide_by_the_combined_tp_cp_degree():
    config = {"num_attention_heads": 6, "in_channels": 4, "out_channels": 4}
    Wan2_T2V_Adapter._validate_context_parallel_config(config, 2)
    with pytest.raises(ValueError, match="cp_size=2 \\* tp_size=2"):
        Wan2_T2V_Adapter._validate_context_parallel_config(config, 2, 2)


def test_wan_tokens_must_divide_by_cp_and_enabled_sp(monkeypatch):
    monkeypatch.setattr(
        "flow_factory.models.wan.wan2_t2v.context_parallel_geometry", lambda _: (2, 0)
    )
    monkeypatch.setattr(
        "flow_factory.models.wan.wan2_t2v.tensor_parallel_geometry", lambda _: (2, 0)
    )
    adapter = SimpleNamespace(
        accelerator=object(),
        config=SimpleNamespace(parallelism_args=SimpleNamespace(sequence_parallel=True)),
        pipeline=SimpleNamespace(
            config=SimpleNamespace(expand_timesteps=False),
            transformer_2=None,
            transformer=SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2))),
        ),
    )
    with pytest.raises(ValueError, match="combined CP/SP degree=4"):
        Wan2_T2V_Adapter._validate_context_parallel_inputs(
            adapter, torch.zeros(1, 4, 1, 2, 4), torch.zeros(1, 4, 8), None
        )
    adapter.config.parallelism_args.sequence_parallel = False
    Wan2_T2V_Adapter._validate_context_parallel_inputs(
        adapter, torch.zeros(1, 4, 1, 2, 4), torch.zeros(1, 4, 8), None
    )


def test_cp_boundaries_surround_sp_and_remain_owned():
    from diffusers.hooks.context_parallel import ContextParallelGatherHook, ContextParallelSplitHook
    from diffusers.hooks.hooks import HookRegistry

    from flow_factory.models.context_parallel import (
        PreparedContextParallel,
        compose_wan_sequence_boundaries,
        validate_prepared_context_parallel,
    )
    from flow_factory.models.model_bundle import ModelBundle

    events = []

    class Split(ContextParallelSplitHook):
        def pre_forward(self, module, *args, **kwargs):
            events.append("cp_split")
            return args, kwargs

    class Gather(ContextParallelGatherHook):
        def post_forward(self, module, output):
            events.append("cp_gather")
            return output

    root = torch.nn.Module()
    root.blocks = torch.nn.ModuleList([torch.nn.Identity()])
    root.proj_out = torch.nn.Identity()
    root._parallel_config = SimpleNamespace(
        context_parallel_config=SimpleNamespace(ulysses_degree=2, ring_degree=1)
    )
    HookRegistry.check_if_exists_or_initialize(root.blocks[0]).register_hook(
        Split({}, None), "cp_input---blocks.0"
    )
    HookRegistry.check_if_exists_or_initialize(root.proj_out).register_hook(
        Gather([], None), "cp_output---proj_out"
    )
    root.blocks[0].register_forward_pre_hook(lambda *_: events.append("sp_scatter"))
    root.proj_out.register_forward_hook(lambda *_: events.append("sp_gather"))
    boundaries = compose_wan_sequence_boundaries(root)
    bundle = ModelBundle({"transformer": root})
    bundle.context_parallel_contract = PreparedContextParallel(
        Wan2_T2V_Adapter.context_parallel_capability, 2, (root,), boundaries
    )
    for _ in range(2):
        root.proj_out(root.blocks[0](torch.ones(1)))
    assert events == ["cp_split", "sp_scatter", "sp_gather", "cp_gather"] * 2
    validate_prepared_context_parallel((bundle,), 2)
    with pytest.raises(ValueError, match="qualified native input/output hooks"):
        compose_wan_sequence_boundaries(root)
    boundaries[0].handle.remove()
    with pytest.raises(ValueError, match="lost an owned sequence boundary"):
        validate_prepared_context_parallel((bundle,), 2)


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        ("parallelism_args", "dp_replicate_size", 2, "requires dp_replicate_size=1"),
        ("eval_args", "eval_freq", 1, "eval.eval_freq=0"),
        ("training_args", "reward_optimization_overlap", True, "reward_optimization_overlap=false"),
        ("model_args", "target_components", ["transformer_2"], "target_components"),
        ("training_args", "per_device_batch_size", 2, "per_device_batch_size=1"),
        ("acceleration_args", "shared", [], "exactly one shared attention_backend"),
    ],
)
def test_unsupported_compositions_fail_before_loading(config, section, field, value, message):
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match=message):
        validate_context_parallel_config(config, Wan2_T2V_Adapter, SFTTrainer)


def test_no_implicit_model_or_trainer_capability(config):
    with pytest.raises(ValueError, match="does not declare"):
        validate_context_parallel_config(config, BaseAdapter, SFTTrainer)
    with pytest.raises(ValueError, match="does not explicitly declare"):
        validate_context_parallel_config(config, Wan2_T2V_Adapter, object)


def test_cp_state_and_ema_use_the_existing_drained_boundary(config):
    config.training_args.ema_decay = 0.99
    config.model_args.resume_type = "state"
    config.log_args.save_model_only = False
    validate_context_parallel_config(config, Wan2_T2V_Adapter, SFTTrainer)


def test_concrete_registry_capabilities_do_not_leak_to_subclasses(config):
    from flow_factory.trainers.registry import _TRAINER_REGISTRY, get_trainer_class

    supported = {"sft", "grpo", "dmd2"}
    for name in _TRAINER_REGISTRY:
        cls = get_trainer_class(name)
        if name in supported:
            validate_context_parallel_config(config, Wan2_T2V_Adapter, cls)
        else:
            with pytest.raises(ValueError, match="does not explicitly declare"):
                validate_context_parallel_config(config, Wan2_T2V_Adapter, cls)

    class CustomSFT(SFTTrainer):
        pass

    with pytest.raises(ValueError, match="does not explicitly declare"):
        validate_context_parallel_config(config, Wan2_T2V_Adapter, CustomSFT)


def test_unqualified_dependency_is_rejected(config, monkeypatch):
    monkeypatch.setattr(
        "flow_factory.trainers.common.context_parallel.version",
        lambda name: (
            "0.42.0.dev0"
            if name == "diffusers"
            else {"torch": "2.10.0", "accelerate": "1.15.0"}[name]
        ),
    )
    with pytest.raises(ValueError, match="qualified diffusers"):
        validate_context_parallel_config(config, Wan2_T2V_Adapter, SFTTrainer)


def test_external_cp_injection_and_non_fsdp2_backend_are_rejected():
    accelerator = SimpleNamespace(
        num_processes=4,
        parallelism_config=SimpleNamespace(
            cp_size=2, tp_size=1, dp_shard_size=2, dp_replicate_size=1
        ),
        is_fsdp2=False,
    )
    with pytest.raises(ValueError, match="remove the launcher's CP override"):
        validate_context_parallel_backend(accelerator, 1)
    with pytest.raises(ValueError, match="requires FSDP2"):
        validate_context_parallel_backend(accelerator, 2)


def test_bridge_cannot_skip_upstream_cp_without_owned_hooks():
    accelerator = SimpleNamespace(parallelism_config=SimpleNamespace(cp_size=2))
    with pytest.raises(ValueError, match="one Flow-Factory ModelBundle"):
        Accelerator._prepare_cp(accelerator, torch.nn.Linear(2, 2))


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"num_attention_heads": 3}, "divisible by cp_size"),
        ({"image_dim": 8}, "image KV"),
        ({"added_kv_proj_dim": 8}, "image KV"),
        ({"in_channels": 8}, "equal input/output"),
    ],
)
def test_wan_metadata_rejects_unsupported_layouts_before_weights(overrides, message):
    metadata = {"num_attention_heads": 2, "in_channels": 4, "out_channels": 4}
    metadata.update(overrides)
    with pytest.raises(ValueError, match=message):
        Wan2_T2V_Adapter._validate_context_parallel_config(metadata, 2)


@pytest.mark.parametrize(
    "shape,attention,message",
    [
        ((2, 4, 2, 4, 4), None, "one dense video"),
        ((1, 4, 2, 3, 4), None, "divisible by patch_size"),
        ((1, 4, 1, 2, 2), None, "token count 1"),
        ((1, 4, 2, 4, 4), {"mask": True}, "special masks"),
    ],
)
def test_wan_forward_rejects_unsupported_inputs_before_collectives(
    monkeypatch, shape, attention, message
):
    monkeypatch.setattr(
        "flow_factory.models.wan.wan2_t2v.context_parallel_geometry", lambda _: (2, 0)
    )
    adapter = SimpleNamespace(
        accelerator=object(),
        pipeline=SimpleNamespace(
            config=SimpleNamespace(expand_timesteps=False),
            transformer_2=None,
            transformer=SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2))),
        ),
    )
    with pytest.raises(ValueError, match=message):
        Wan2_T2V_Adapter._validate_context_parallel_inputs(
            adapter, torch.zeros(shape), torch.zeros(1, 4, 8), attention
        )


def test_wan_manifest_rejection_does_not_load_weights(config, monkeypatch):
    from diffusers import WanPipeline

    monkeypatch.setattr(WanPipeline, "load_config", lambda *_: {"expand_timesteps": True})
    adapter = SimpleNamespace(config=config, model_args=config.model_args)
    with pytest.raises(ValueError, match="scalar sample timesteps"):
        Wan2_T2V_Adapter.load_pipeline(adapter)


def test_unqualified_precision_and_optimizer_are_rejected(config):
    config.mixed_precision = "no"
    with pytest.raises(ValueError, match="mixed_precision=bf16"):
        validate_context_parallel_config(config, Wan2_T2V_Adapter, SFTTrainer)
    config.mixed_precision = "bf16"
    config.optimizer_args[0].optimizer = "muon"
    with pytest.raises(ValueError, match="requires AdamW"):
        validate_context_parallel_config(config, Wan2_T2V_Adapter, SFTTrainer)
