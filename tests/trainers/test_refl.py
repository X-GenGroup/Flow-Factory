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

"""ReFL configuration, independent derivative program and decoder admission."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from flow_factory.hparams import Arguments, ReFLTrainingArguments
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.flux.flux1 import Flux1Adapter
from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter
from flow_factory.rewards.differentiable import DifferentiableRewardProcessor
from flow_factory.samples import ComponentTimes, LatentState
from flow_factory.trainers.registry import get_trainer_class
from flow_factory.trainers.rl.refl import ReFLTrainer


def values():
    return {
        "model": {"model_type": "sd3-5", "model_name_or_path": "/tmp/refl-model"},
        "train": {
            "trainer_type": "refl",
            "unique_sample_num_per_epoch": 2,
            "num_inference_steps": 4,
        },
        "scheduler": {"dynamics_type": "ODE", "replay_steps": [2, 3]},
        "rewards": [{"reward_model": "pickscore"}],
        "log": {"save_dir": ".scratch/p19/refl-saves"},
    }


def test_registry_recipe_and_single_backward_geometry():
    args = Arguments.from_dict(values())
    assert isinstance(args.training_args, ReFLTrainingArguments)
    assert get_trainer_class("refl") is ReFLTrainer
    assert args.training_args.get_num_train_timesteps(args) == 1
    assert args.training_args.group_size == 1
    ReFLTrainer.validate_training_arguments_contract(args.training_args)
    for adapter in (SD3_5Adapter, Flux1Adapter):
        ReFLTrainer.validate_adapter_class_execution_contract(adapter)
    with pytest.raises(ValueError, match="decoder|decoding"):
        ReFLTrainer.validate_adapter_class_execution_contract(BaseAdapter)


@pytest.mark.parametrize("change", ["no_states", "sde", "ocr", "async", "target_vae"])
def test_invalid_compositions_fail_before_loading(change):
    raw = values()
    if change == "no_states":
        raw["scheduler"]["replay_steps"] = None
    if change == "sde":
        raw["scheduler"]["dynamics_type"] = "Flow-SDE"
    if change == "ocr":
        raw["rewards"][0]["reward_model"] = "pickscore_rank"
    if change == "async":
        raw["rewards"][0]["async_reward"] = True
    if change == "target_vae":
        raw["model"]["target_components"] = ["vae"]
    with pytest.raises(ValueError):
        Arguments.from_dict(raw)


def test_native_clean_projection_gradient_matches_independent_formula():
    # Native SD3 state and velocity use the shared flow coordinate convention.
    adapter = object.__new__(SD3_5Adapter)
    x = torch.tensor([[[[0.4]]]], dtype=torch.float64)
    weight = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    velocity = x * weight
    state = LatentState(components={"latent": x})
    times = ComponentTimes(
        timestep={"latent": torch.tensor([500.0])},
        next_timestep={"latent": torch.tensor([250.0])},
        sigma={"latent": torch.tensor([0.5])},
    )
    clean = adapter.project_velocity_to_clean_state(
        state, times, LatentState(components={"latent": velocity})
    )
    loss = -(clean.components["latent"].sigmoid().square()).mean()
    expected_clean = x - 0.5 * x * weight
    expected = -(expected_clean.sigmoid().square()).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, weight, retain_graph=True)[0],
        torch.autograd.grad(expected, weight)[0],
    )


def test_production_optimize_matches_independent_reward_gradient_two_cycles():
    from contextlib import nullcontext

    from flow_factory.rewards.differentiable import DifferentiableRewardOutput
    from flow_factory.trainers.common.group_noise import identity_timestep_generator

    weight = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
    oracle_weight = weight.detach().clone().requires_grad_()
    x = torch.tensor([0.4, 0.6, 0.8], dtype=torch.float64).reshape(1, 3, 1, 1)
    times = ComponentTimes(
        timestep={"latent": torch.tensor([500.0])}, next_timestep={"latent": torch.tensor([250.0])}
    )
    state = LatentState({"latent": x})
    sample = SimpleNamespace(source_id=None, sampling_sample_id=7)

    class Batch(dict):
        samples = [sample]

    batch = Batch(prompt=["a bird"])
    native = object.__new__(SD3_5Adapter)
    adapter = SimpleNamespace(
        trajectory_component_order=("latent",),
        train=lambda: None,
        get_train_step_indices=lambda: torch.tensor([2]),
        get_replay_step=lambda batch, step: SimpleNamespace(
            state=state, next_state=state, times=times
        ),
        forward_state=lambda **kwargs: SimpleNamespace(
            velocity=LatentState({"latent": x * weight})
        ),
        build_training_component_times=lambda *a, **kw: replace(
            times, sigma={"latent": torch.tensor([0.5])}
        ),
        project_velocity_to_clean_state=native.project_velocity_to_clean_state,
        decode_differentiable_state=lambda state, **kw: {
            "image": state.components["latent"].sigmoid()
        },
    )
    trainer = object.__new__(ReFLTrainer)
    trainer.training_args = ReFLTrainingArguments(num_inner_epochs=1, seed=17)
    trainer.config = SimpleNamespace(
        data_args=SimpleNamespace(
            source_id_to_name=["tiny"], training_datasets=[SimpleNamespace(name="tiny")]
        )
    )
    trainer.adapter = adapter
    trainer.accelerator = SimpleNamespace(sync_gradients=True)
    trainer.autocast = nullcontext
    trainer.accumulate_gradients = nullcontext
    trainer._order_samples_for_optimize = lambda samples, inner: samples
    trainer._iter_prefetched_batches = lambda samples, size: iter([batch])
    trainer.differentiable_feedback = lambda **kw: DifferentiableRewardOutput(
        scores={"pixel": kw["image"].square().mean((1, 2, 3))}, weights={"pixel": torch.ones(1)}
    )
    gradients = []
    trainer._backward_full_output_loss = lambda loss: loss.backward()

    def update(info):
        gradients.append(weight.grad.clone())
        with torch.no_grad():
            weight.add_(weight.grad, alpha=-0.1)
        weight.grad = None
        from collections import defaultdict

        return defaultdict(list)

    trainer._apply_optimizer_step = update
    for cycle in range(2):
        trainer.epoch = cycle
        trainer.optimize([sample])
        expected = -((x - 0.5 * x * oracle_weight).sigmoid().square()).mean()
        gradient = torch.autograd.grad(expected, oracle_weight)[0]
        torch.testing.assert_close(gradients[-1], gradient)
        with torch.no_grad():
            oracle_weight.add_(gradient, alpha=-0.1)
        torch.testing.assert_close(weight, oracle_weight)


def test_resume_identity_binds_reward_revision_and_preprocessing(monkeypatch):
    from flow_factory.contracts.reward import RewardGradientCapability
    from flow_factory.trainers.abc import BaseTrainer

    monkeypatch.setattr(BaseTrainer, "runtime_execution_identity_payload", lambda self: {})
    trainer = object.__new__(ReFLTrainer)
    model_config = SimpleNamespace(_commit_hash="revision-a")
    scorer = SimpleNamespace(
        model=SimpleNamespace(config=model_config),
        gradient_capability=RewardGradientCapability(("image",)),
        processor=SimpleNamespace(image_processor=SimpleNamespace(to_dict=lambda: {"size": 224})),
    )
    trainer.reward_models = {"pixel": scorer}
    trainer.adapter = SimpleNamespace(differentiable_decode_fields=("image",))
    first = trainer.runtime_execution_identity_payload()
    model_config._commit_hash = "revision-b"
    second = trainer.runtime_execution_identity_payload()
    assert first != second
    assert first["differentiable_scorers"]["pixel"]["image_preprocessing"] == {"size": 224}


def test_shipped_pickscore_recipe_parses_through_public_arguments():
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(Path("examples/refl/lora/sd3_5/pickscore.yaml").read_text())
    args = Arguments.from_dict(raw)
    assert args.training_args.trainer_type == "refl"
    assert args.scheduler_args.replay_steps == list(range(30, 40))
    assert args.training_args.get_num_train_timesteps(args) == 1
    assert args.model_args.model_type == "sd3-5"


def test_single_training_source_with_independent_evaluation_source():
    trainer = object.__new__(ReFLTrainer)
    trainer.config = SimpleNamespace(
        data_args=SimpleNamespace(
            source_id_to_name=["evaluation", "alignment"],
            training_datasets=[SimpleNamespace(name="alignment")],
        )
    )
    assert trainer._reward_source_names([SimpleNamespace(source_id=None)]) == ["alignment"]
    assert trainer._reward_source_names([SimpleNamespace(source_id=1)]) == ["alignment"]
    trainer.config.data_args.training_datasets.append(SimpleNamespace(name="other"))
    with pytest.raises(ValueError, match="lacks source identity"):
        trainer._reward_source_names([SimpleNamespace(source_id=None)])
