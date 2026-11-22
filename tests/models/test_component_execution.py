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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from accelerate.utils import DistributedType

from flow_factory.contracts.topology import ParallelTopology
from flow_factory.hparams import Arguments
from flow_factory.hparams.component_execution import (
    ComponentExecutionArguments,
    ComponentExecutionPolicy,
)
from flow_factory.hparams.training_args import TrainingArguments
from flow_factory.models import loader
from flow_factory.models.component_execution import (
    OptimizationGradients,
    resolve_component_execution,
    resolve_reward_execution,
)
from flow_factory.models.runtime import ClassicPipelineRuntime, PseudoPipelineRuntime
from flow_factory.rewards.loader import MultiRewardLoader

_REFL_EXAMPLE = Path("examples/refl/lora/sd3_5/pickscore.yaml")


def _config(requests=None):
    values = yaml.safe_load(_REFL_EXAMPLE.read_text())
    values["log"]["run_name"] = "component-execution"
    if requests is not None:
        values["component_execution"] = requests
    return Arguments.from_dict(values)


def _adapter(config):
    encoder = torch.nn.Linear(2, 2)
    pipeline = SimpleNamespace(
        transformer=torch.nn.Linear(2, 2),
        vae=torch.nn.Linear(2, 2),
        text_encoder=encoder,
        text_encoder_2=encoder,
        tokenizer=object(),
    )
    pipeline.components = {
        name: getattr(pipeline, name)
        for name in ("transformer", "vae", "text_encoder_2", "tokenizer")
    }
    runtime = ClassicPipelineRuntime(pipeline)
    return SimpleNamespace(
        component_runtime=runtime,
        model_args=config.model_args,
        differentiable_decode_modules=("vae",),
        _resolve_component_names=runtime.resolve_component_names,
    )


@pytest.mark.parametrize(
    "backend", [DistributedType.MULTI_GPU, DistributedType.DEEPSPEED, DistributedType.FSDP]
)
def test_model_factory_defaults_preserve_native_construction(monkeypatch, backend):
    config = _config()
    accelerator = SimpleNamespace(distributed_type=backend)
    adapter = object()

    def construct(**kwargs):
        assert kwargs == {"config": config, "accelerator": accelerator}
        return adapter

    monkeypatch.setattr(loader, "get_model_adapter_class", lambda _: construct)
    assert config.component_execution_args is None
    assert "component_execution" not in config.to_dict()
    assert loader.load_model(config, accelerator) is adapter


def test_explicit_replication_binds_physical_model_and_reward_once(monkeypatch):
    config = _config(
        {
            "model": {"text_encoders": {}, "text_encoder": {"placement": "replicated"}, "vae": {}},
            "rewards": {"alignment": {}},
            "eval_rewards": {"alignment": {}},
        }
    )
    default = _config()
    payload = config.to_dict()
    exported_policy = payload.pop("component_execution")
    assert _config(exported_policy).component_execution_args == config.component_execution_args
    assert payload == default.to_dict()
    assert config.reward_args[0].get_identity_key() == default.reward_args[0].get_identity_key()

    adapter = _adapter(config)
    monkeypatch.setattr(loader, "get_model_adapter_class", lambda _: lambda **_: adapter)
    assert loader.load_model(config, SimpleNamespace()) is adapter
    plan = adapter.component_execution_plan
    assert set(plan.model) == {"transformer", "vae", "text_encoder_2"}
    assert plan.model["transformer"].policy is None
    assert plan.model["transformer"].optimization_gradients is OptimizationGradients.PARAMETERS
    assert plan.model["vae"].optimization_gradients is OptimizationGradients.INPUT
    assert plan.model["text_encoder_2"].optimization_gradients is OptimizationGradients.NONE
    assert plan.model["vae"].policy == plan.model["text_encoder_2"].policy

    instances = []

    def construct_reward(config, accelerator):
        instance = object()
        instances.append(instance)
        return instance

    monkeypatch.setattr("flow_factory.rewards.loader.load_reward_model", construct_reward)
    reward_loader = MultiRewardLoader(config.reward_args, SimpleNamespace()).load()
    assert reward_loader.get_training_reward_models() == reward_loader.get_eval_reward_models()
    assert len(instances) == 1
    assert len(plan.rewards) == 1
    assert (
        plan.rewards[config.reward_args[0].get_identity_key()].optimization_gradients
        is OptimizationGradients.INPUT
    )


def test_conflicting_train_eval_policies_fail_before_model_construction(monkeypatch):
    config = _config(
        {
            "rewards": {"alignment": {}},
            "eval_rewards": {"alignment": {"placement": "fsdp2", "shard_group": "data"}},
        }
    )

    def unexpected(_):
        pytest.fail("adapter resolution must follow policy admission")

    monkeypatch.setattr(loader, "get_model_adapter_class", unexpected)
    with pytest.raises(
        ValueError, match="eval_rewards.alignment.*rewards.alignment.*one physical component"
    ):
        loader.load_model(config, SimpleNamespace())


def test_separate_eval_alias_inherits_same_physical_reward():
    config = _config({"rewards": {"alignment": {}}, "eval_rewards": {"quality": {}}})
    config.eval_reward_args = type(config.reward_args).from_dict(
        [{**config.reward_args[0].to_dict(), "name": "quality"}]
    )
    plan = resolve_reward_execution(config, config.component_execution_args)
    assert config.reward_args[0].get_identity_key() == config.eval_reward_args[0].get_identity_key()
    assert len(plan) == 1
    assert next(iter(plan.values())).optimization_gradients is OptimizationGradients.INPUT


def test_runtime_reward_and_auxiliary_decoder_use_no_grad():
    config = _config({"model": {"vae": {}}, "rewards": {"alignment": {}}})
    config.training_args = TrainingArguments()
    requests = config.component_execution_args
    rewards = resolve_reward_execution(config, requests)
    plan = resolve_component_execution(config, _adapter(config), requests, rewards)
    assert plan.model["vae"].optimization_gradients is OptimizationGradients.NONE
    assert next(iter(plan.rewards.values())).optimization_gradients is OptimizationGradients.NONE


def test_model_aliases_share_one_policy_and_detect_conflict():
    config = _config(
        {
            "model": {
                "text_encoder": {},
                "text_encoder_2": {"placement": "fsdp2", "shard_group": "data"},
            }
        }
    )
    with pytest.raises(ValueError, match="text_encoder_2.*text_encoder.*one physical component"):
        resolve_component_execution(config, _adapter(config), config.component_execution_args, {})


def test_pseudo_auxiliary_under_target_root_cannot_change_placement():
    config = _config({"model": {"text_encoder": {}}})
    root = torch.nn.ModuleDict(
        {"denoiser": torch.nn.Linear(2, 2), "encoder": torch.nn.Linear(2, 2)}
    )
    runtime = PseudoPipelineRuntime(
        object(),
        {"root": root, "vae": torch.nn.Linear(2, 2)},
        {"transformer": root["denoiser"], "text_encoder": root["encoder"]},
        {"transformer": ("root", ("denoiser",)), "text_encoder": ("root", ("encoder",))},
    )
    adapter = SimpleNamespace(
        model_args=config.model_args,
        component_runtime=runtime,
        _resolve_component_names=runtime.resolve_component_names,
        differentiable_decode_modules=("vae",),
    )
    with pytest.raises(ValueError, match="text_encoder resolves to target root 'root'"):
        resolve_component_execution(config, adapter, config.component_execution_args, {})


def test_shard_group_reuses_data_scope_and_launch_fails_before_loading(monkeypatch):
    config = _config({"model": {"vae": {"placement": "fsdp2", "shard_group": "data"}}})
    topology = ParallelTopology(dp_shard_size=2, tp_size=2, cp_size=2)
    policy = config.component_execution_args.model["vae"]
    assert policy.group_ranks(topology, 3) == (3, 7)
    with pytest.raises(ValueError, match="requires at least two ranks"):
        policy.group_ranks(ParallelTopology(), 0)

    monkeypatch.setattr(
        "flow_factory.models.component_execution.parallel_topology", lambda _: topology
    )
    monkeypatch.setattr(loader, "get_model_adapter_class", lambda _: pytest.fail("unexpected load"))
    with pytest.raises(
        ValueError, match="component_execution.model.vae.*supported auxiliary preparation"
    ):
        loader.load_model(config, SimpleNamespace(process_index=3))


@pytest.mark.parametrize("selector", ["missing", "tokenizer", "transformer"])
def test_invalid_model_selectors_fail_before_training_preparation(selector):
    config = _config({"model": {selector: {}}})
    with pytest.raises(ValueError, match=selector):
        resolve_component_execution(config, _adapter(config), config.component_execution_args, {})


def test_unknown_reward_name_fails_at_plan_resolution():
    config = _config({"rewards": {"other": {}}})
    with pytest.raises(ValueError, match="other.*unknown.*alignment"):
        resolve_reward_execution(config, config.component_execution_args)


@pytest.mark.parametrize(
    "policy, message",
    [
        ({"placment": "replicated"}, "placment"),
        ({"placement": "zero2"}, "placement='zero2'"),
        ({"placement": "fsdp2"}, "requires an explicit shard_group"),
        ({"shard_group": "data"}, "replication uses no group"),
        ({"loading": "sharded"}, "requires placement='fsdp2'"),
        ({"precision": "int8"}, "precision='int8'"),
    ],
)
def test_policy_parser_rejects_invalid_values(policy, message):
    with pytest.raises(ValueError, match=message):
        ComponentExecutionArguments.from_dict({"model": {"vae": policy}})


@pytest.mark.parametrize(
    "policy",
    [
        ComponentExecutionPolicy(precision="bf16"),
        ComponentExecutionPolicy(offload="cpu"),
    ],
)
def test_pending_execution_preference_does_not_silently_fall_back(monkeypatch, policy):
    config = _config({"model": {"vae": policy}})
    monkeypatch.setattr(loader, "get_model_adapter_class", lambda _: pytest.fail("unexpected load"))
    with pytest.raises(ValueError, match="component_execution.model.vae.*inherited loading"):
        loader.load_model(config, SimpleNamespace())
