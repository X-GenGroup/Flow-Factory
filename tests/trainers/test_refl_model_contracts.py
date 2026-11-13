# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""ReFL modality admission, decoder graph and structured projection contracts."""

import ast
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flow_factory.hparams import Arguments, ReFLTrainingArguments
from flow_factory.models.registry import get_model_adapter_class, list_registered_models
from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter
from flow_factory.samples import ComponentTimes, LatentState
from flow_factory.trainers.rl.refl import ReFLTrainer


@pytest.mark.parametrize("model", list_registered_models())
def test_registry_refl_admission_uses_model_media(model):
    if model == "bagel":
        if importlib.util.find_spec("flash_attn") is None:
            # CPU CI validates the declaration without importing CUDA-only native
            # kernels. The registry-matched GPU case imports and trains Bagel.
            path = Path(__file__).parents[2] / "src/flow_factory/models/bagel/bagel.py"
            cls = next(
                node
                for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == "BagelAdapter"
            )
            assert any(
                isinstance(node, ast.FunctionDef) and node.name == "_decode_differentiable_state"
                for node in cls.body
            )
            assert any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "differentiable_decode_fields"
                    for target in node.targets
                )
                and ast.literal_eval(node.value) == ("image",)
                for node in cls.body
            )
            return
    cls = get_model_adapter_class(model)
    ReFLTrainer.validate_adapter_class_execution_contract(cls)
    assert cls.differentiable_decode_fields in (("image",), ("video",), ("video", "audio"))
    config = SimpleNamespace(model_args=SimpleNamespace(model_type=model))
    assert (
        ReFLTrainingArguments().get_reward_gradient_fields(config)
        == cls.differentiable_decode_fields
    )


@pytest.mark.parametrize(
    "model", ["qwen-image-edit-plus", "wan2_t2v", "ltx2_i2av", "minimax-h3-ref2va"]
)
def test_pickscore_configuration_accepts_image_video_and_joint_output(model):
    args = Arguments.from_dict(
        {
            "model": {"model_type": model, "model_name_or_path": "/tmp/refl-native"},
            "train": {"trainer_type": "refl", "unique_sample_num_per_epoch": 1},
            "scheduler": {"dynamics_type": "ODE", "replay_steps": [1]},
            "rewards": [{"reward_model": "pickscore"}],
            "log": {"save_dir": ".scratch/refl-all/config"},
        }
    )
    assert args.training_args.trainer_type == "refl"


@pytest.mark.parametrize("field,shape", [("image", (2, 3, 2, 2)), ("video", (2, 3, 3, 2, 2))])
def test_native_decode_boundary_retains_graph_and_generation_autocast(field, shape):
    class Adapter(SD3_5Adapter):
        differentiable_decode_fields = (field,)

    adapter = object.__new__(Adapter)
    values = torch.full(shape, 0.3, requires_grad=True)
    seen = []

    def decode(latents, output_type):
        seen.append((output_type, torch.is_autocast_enabled("cpu")))
        return latents.sigmoid()

    adapter.decode_latents = decode
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = adapter.decode_differentiable_state(LatentState({"latent": values}), batch={})
    result[field].sum().backward()
    torch.testing.assert_close(
        values.grad, values.detach().sigmoid() * (1 - values.detach().sigmoid())
    )
    assert seen == [("pt", True)]
    adapter.decode_latents = lambda **kw: values.detach().flatten()
    with pytest.raises(ValueError, match="batched tensors"):
        adapter.decode_differentiable_state(LatentState({"latent": values}), batch={})


def test_structured_refl_uses_stored_component_sigmas_direction_and_fixed_tokens():
    from flow_factory.models.minimax_h3.adapters import MiniMaxH3T2VAAdapter
    from flow_factory.rewards.differentiable import DifferentiableRewardOutput

    native = object.__new__(MiniMaxH3T2VAAdapter)
    weights = torch.nn.Parameter(torch.tensor([0.2, 0.4], dtype=torch.float64))
    x = {
        "video": torch.tensor([[[0.3], [0.7]]], dtype=torch.float64),
        "audio": torch.tensor([[[0.5], [0.8]]], dtype=torch.float64),
    }
    masks = {
        "video": torch.tensor([[[False], [True]]]),
        "audio": torch.ones(1, 2, 1, dtype=torch.bool),
    }
    state = LatentState(x, active_masks=masks)
    times = ComponentTimes(
        timestep={"video": torch.tensor([123.0]), "audio": torch.tensor([456.0])},
        next_timestep={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
        sigma={"video": torch.tensor([0.25]), "audio": torch.tensor([0.75])},
    )
    sample = SimpleNamespace(source_id=None, sampling_sample_id=7)

    class Batch(dict):
        samples = [sample]

    batch = Batch(prompt=["a bird"])
    clean_values = []

    def decode(clean, **kw):
        clean_values.append(clean)
        return {name: value for name, value in clean.components.items()}

    trainer = object.__new__(ReFLTrainer)
    trainer.model_bundle = torch.nn.ParameterList([weights])
    trainer.training_args = ReFLTrainingArguments(seed=17)
    trainer.config = SimpleNamespace(
        data_args=SimpleNamespace(
            source_id_to_name=["tiny"], training_datasets=[SimpleNamespace(name="tiny")]
        )
    )
    trainer.adapter = SimpleNamespace(
        train=lambda: None,
        get_train_step_indices=lambda: torch.tensor([1]),
        get_replay_step=lambda *a: SimpleNamespace(state=state, times=times, next_state=state),
        forward_state=lambda **kw: SimpleNamespace(
            velocity=LatentState(
                {name: value * weights[i] for i, (name, value) in enumerate(x.items())}
            )
        ),
        project_velocity_to_clean_state=native.project_velocity_to_clean_state,
        decode_differentiable_state=decode,
    )
    trainer.autocast = trainer.accumulate_gradients = nullcontext
    trainer.accelerator = SimpleNamespace(sync_gradients=True)
    trainer._order_samples_for_optimize = lambda samples, inner: samples
    trainer._iter_prefetched_batches = lambda *a: iter([batch])
    trainer.differentiable_feedback = lambda **kw: DifferentiableRewardOutput(
        {"joint": sum(kw[name].square().flatten(1).mean(1) for name in x)},
        {"joint": torch.ones(1)},
    )
    trainer._backward_full_output_loss = lambda loss: loss.backward()
    trainer._apply_optimizer_step = lambda info: info
    trainer.epoch = 0
    trainer.optimize([sample])
    expected = {}
    for i, (name, value) in enumerate(x.items()):
        # H3 velocities point toward data. Audio/video have distinct stored clocks.
        expected[name] = torch.where(
            masks[name], value + times.sigma[name] * value * weights[i], value
        )
        torch.testing.assert_close(clean_values[0].components[name], expected[name])
    oracle = -sum(value.square().mean() for value in expected.values())
    torch.testing.assert_close(weights.grad, torch.autograd.grad(oracle, weights)[0])


@pytest.mark.parametrize(
    "model,expected", [("sd3-5", False), ("ltx2_i2av", True), ("minimax-h3-ref2va", True)]
)
def test_refl_ddp_preparation_accounts_for_unscored_output_modalities(model, expected):
    from flow_factory.trainers.loader import _requires_ddp_unused_parameter_detection

    config = SimpleNamespace(training_args=ReFLTrainingArguments())
    assert (
        _requires_ddp_unused_parameter_detection(config, get_model_adapter_class(model)) is expected
    )


def test_refl_coordinates_follow_global_microbatch_for_conditional_sharded_routes():
    trainer = object.__new__(ReFLTrainer)
    trainer.training_args = ReFLTrainingArguments(seed=17)
    trainer.epoch = 0
    candidates = torch.tensor([1, 2])
    first = [trainer._select_replay_step(candidates, 0, batch) for batch in range(16)]
    torch.manual_seed(890)
    restored = object.__new__(ReFLTrainer)
    restored.training_args = ReFLTrainingArguments(seed=17)
    restored.epoch = 0
    assert first == [restored._select_replay_step(candidates, 0, batch) for batch in range(16)]
    assert set(first) == {1, 2}
    restored.epoch = 1
    assert restored._select_replay_step(candidates, 0, 0) != first[0]
