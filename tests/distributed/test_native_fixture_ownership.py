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

"""CPU contracts for extracted native fixture recipes and passive observations."""

import copy
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from context_parallel_evaluation_runner import _config as evaluation_config
from context_parallel_online_runner import _config as online_config
from native_observation_utils import cpu
from online_optimizer_observations import observe_optimizer_steps
from test_context_parallel_state import _cpu_tree
from test_context_parallel_training import _check_adam_trajectory, _compare
from wan_evaluation_fixture import build_fixture as build_evaluation
from wan_online_fixture import build_fixture as build_online


def _trainer(gas):
    model = torch.nn.ParameterDict(
        {
            "policy": torch.nn.Parameter(torch.linspace(-0.4, 0.6, 12)),
            "inactive": torch.nn.Parameter(torch.linspace(0.3, 0.8, 12)),
        }
    )
    optimizer = torch.optim.AdamW(
        [{"params": [model["policy"]]}, {"params": [model["inactive"]]}],
        lr=0.002,
        eps=1e-4,
        weight_decay=0.01,
        foreach=False,
    )
    return SimpleNamespace(
        model_bundle=model,
        optimizer=SimpleNamespace(optimizer=optimizer, step=optimizer.step),
        accelerator=SimpleNamespace(
            backward=lambda loss: (loss / gas).backward(),
            clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
        ),
        role_optimization=SimpleNamespace(_active_role_name=None),
        _primary_role=lambda: "default",
        optimization_roles={
            "default": SimpleNamespace(
                parameters=[model["policy"]],
                optimizer_group_ids=[0],
                config=SimpleNamespace(max_grad_norm=0.3),
            ),
            "inactive": SimpleNamespace(
                parameters=[model["inactive"]],
                optimizer_group_ids=[1],
                config=SimpleNamespace(max_grad_norm=0.3),
            ),
        },
    )


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("gas", [1, 2])
def test_optimizer_observer_is_passive_and_preserves_independent_adam(cp_size, gas):
    observed, control = _trainer(gas), _trainer(gas)
    records = {
        "initial_parameters": _cpu_tree(dict(observed.model_bundle.items())),
        "updates": [],
        "backward_losses": [],
    }
    initial_inactive = observed.model_bundle["inactive"].detach().clone()
    originals = (
        observed.accelerator.backward,
        observed.accelerator.clip_grad_norm_,
        observed.optimizer.step,
    )
    for trainer in (observed, control):
        context = (
            observe_optimizer_steps(trainer, records, cp_size=cp_size, is_saving=lambda: False)
            if trainer is observed
            else nullcontext()
        )
        with context:
            for update in range(2):
                for microbatch in range(gas):
                    slope = torch.linspace(-0.9, 1.2, 12) * (update + 1 + microbatch / 4)
                    loss = (trainer.model_bundle["policy"] * slope).sum()
                    trainer.accelerator.backward(loss * cp_size)
                trainer.accelerator.clip_grad_norm_(trainer.model_bundle.parameters(), 0.3)
                trainer.optimizer.step()
                trainer.optimizer.optimizer.zero_grad(set_to_none=True)
    assert originals == (
        observed.accelerator.backward,
        observed.accelerator.clip_grad_norm_,
        observed.optimizer.step,
    )
    _compare(
        _cpu_tree(dict(observed.model_bundle.items())),
        _cpu_tree(dict(control.model_bundle.items())),
        exact=True,
    )
    _compare(
        observed.optimizer.optimizer.state_dict(),
        control.optimizer.optimizer.state_dict(),
        exact=True,
    )
    assert not records["backward_losses"] and len(records["updates"]) == 2
    for index, update in enumerate(records["updates"]):
        # CPU degree injection checks trace accounting, not native CP execution.
        expected = torch.linspace(-0.9, 1.2, 12) * (index + 1 + (gas - 1) / 8) * cp_size
        torch.testing.assert_close(update["gradients"]["policy"], expected)
        assert len(update["loss"]) == gas
        assert set(update["moments"]) == {"policy"}
        torch.testing.assert_close(
            update["parameters"]["inactive"], initial_inactive, rtol=0, atol=0
        )
    assert len(_check_adam_trajectory(records)) == 2
    corrupted = copy.deepcopy(records)
    corrupted["updates"][0]["clipped_gradients"]["policy"] *= 0.5
    with pytest.raises(AssertionError):
        _check_adam_trajectory(corrupted)


def test_optimizer_observer_restores_hooks_after_failure():
    trainer = _trainer(1)
    originals = (
        trainer.accelerator.backward,
        trainer.accelerator.clip_grad_norm_,
        trainer.optimizer.step,
    )
    records = {"updates": [], "backward_losses": []}
    with pytest.raises(RuntimeError, match="objective failure"):
        with observe_optimizer_steps(trainer, records, cp_size=1, is_saving=lambda: False):
            raise RuntimeError("objective failure")
    assert originals == (
        trainer.accelerator.backward,
        trainer.accelerator.clip_grad_norm_,
        trainer.optimizer.step,
    )
    assert records == {"updates": [], "backward_losses": []}


@pytest.mark.parametrize("factory", [online_config, evaluation_config])
def test_runner_resolves_the_explicit_recipe_module(factory, monkeypatch, tmp_path):
    seen = []
    recipe = SimpleNamespace(make_config=lambda *args: seen.append(args) or "selected recipe")
    monkeypatch.setitem(sys.modules, "explicit_native_recipe", recipe)
    options = {"case_module": "explicit_native_recipe"}
    assert factory(tmp_path, "outputs", options) == "selected recipe"
    assert seen == [(tmp_path, "outputs", options)]


@pytest.mark.parametrize(
    "builder,expected_counts",
    [
        (build_online, {"prompts": 8}),
        (build_evaluation, {"prompts": 8, "odd": 3, "single": 1, "empty": 0, "divisible": 4}),
    ],
)
def test_recipe_preserves_subprocess_construction_and_prompt_rows(
    builder, expected_counts, monkeypatch, tmp_path
):
    calls = []

    def construct(command, *, check):
        assert check is True
        calls.append(command)
        Path(command[2]).mkdir()

    monkeypatch.setattr("subprocess.run", construct)
    root = tmp_path / "fixture"
    builder(root, 2)
    assert len(calls) == 1
    assert calls[0][0] == sys.executable
    assert Path(calls[0][1]) == Path(__file__).with_name("wan_sft_fixture.py")
    assert calls[0][2:] == [str(root), "4"]
    for name, count in expected_counts.items():
        files = list((root / name).iterdir())
        assert len(files) == 1 and len(files[0].read_text().splitlines()) == count


@pytest.mark.parametrize("trainer_type", ["grpo", "dmd2"])
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_online_recipe_preserves_training_and_resume_inputs(
    trainer_type, mode, tmp_path, monkeypatch
):
    monkeypatch.setattr("flow_factory.hparams.args.get_world_size", lambda: 8)
    monkeypatch.setattr("flow_factory.hparams.training_args._base.get_world_size", lambda: 8)
    options = {
        "algorithm": trainer_type,
        "mode": mode,
        "cp_size": 2,
        "tp_size": 2,
        "guidance_scale": 3.0,
        "resume": str(tmp_path / "checkpoint-1"),
    }
    config = online_config(tmp_path, tmp_path / "output", options)
    assert config.training_args.trainer_type == trainer_type
    assert config.training_args.guidance_scale == 3.0
    assert config.training_args.max_epochs == 2
    assert config.model_args.finetune_type == mode
    assert config.model_args.resume_path == options["resume"]
    assert config.model_args.compute_precision_policy == "cross_attention_fp32_v1"
    assert config.parallelism_args.cp_size == config.parallelism_args.tp_size == 2
    assert config.log_args.run_name == "wan-online"


def test_local_observation_copy_preserves_packed_and_video_shapes_without_aliasing():
    @dataclass
    class State:
        components: dict

    value = State(
        {
            "packed": torch.arange(24.0).reshape(2, 3, 4),
            "video": torch.arange(24.0).reshape(1, 2, 3, 2, 2),
        }
    )
    copied = cpu(value)
    for name, original in value.components.items():
        torch.testing.assert_close(copied["components"][name], original, rtol=0, atol=0)
        assert copied["components"][name].data_ptr() != original.data_ptr()
        original.add_(1)
        assert not torch.equal(copied["components"][name], original)


def test_optimizer_observer_records_zero_gradients_and_weight_decay():
    trainer = _trainer(1)
    initial = _cpu_tree(dict(trainer.model_bundle.items()))
    records = {"initial_parameters": initial, "updates": [], "backward_losses": []}
    with observe_optimizer_steps(trainer, records, cp_size=1, is_saving=lambda: False):
        for index in range(2):
            trainer.accelerator.backward((trainer.model_bundle["policy"] * 0).sum())
            trainer.accelerator.clip_grad_norm_(trainer.model_bundle.parameters(), 0.3)
            trainer.optimizer.step()
            trainer.optimizer.optimizer.zero_grad(set_to_none=True)
    assert len(_check_adam_trajectory(records)) == 2
    for index, update in enumerate(records["updates"], 1):
        assert torch.count_nonzero(update["gradients"]["policy"]) == 0
        assert torch.count_nonzero(update["clipped_gradients"]["policy"]) == 0
        assert torch.count_nonzero(update["moments"]["policy"]["exp_avg"]) == 0
        assert torch.count_nonzero(update["moments"]["policy"]["exp_avg_sq"]) == 0
        assert update["moments"]["policy"]["step"] == index
        assert torch.count_nonzero(update["deltas"]["policy"]) > 0
        torch.testing.assert_close(
            update["parameters"]["policy"], initial["policy"] * (1 - 0.002 * 0.01) ** index
        )


def _trainer_with_unused_role_output():
    trainer = _trainer(1)
    trainer.optimization_roles["default"].parameters.append(trainer.model_bundle["inactive"])
    return trainer


def test_optimizer_observer_declared_unused_role_output_is_unchanged():
    trainer = _trainer_with_unused_role_output()
    initial = _cpu_tree(dict(trainer.model_bundle.items()))
    records = {"initial_parameters": initial, "updates": [], "backward_losses": []}
    with observe_optimizer_steps(
        trainer,
        records,
        cp_size=1,
        is_saving=lambda: False,
        inactive_parameter_names=frozenset({"inactive"}),
    ):
        trainer.accelerator.backward(trainer.model_bundle["policy"].square().mean())
        trainer.accelerator.clip_grad_norm_(trainer.model_bundle.parameters(), 0.3)
        trainer.optimizer.step()
    assert records["updates"][0]["gradients"].keys() == {"policy"}
    assert trainer.model_bundle["inactive"].grad is None
    assert trainer.model_bundle["inactive"] not in trainer.optimizer.optimizer.state
    _compare(records["updates"][0]["parameters"]["inactive"], initial["inactive"], exact=True)
    assert len(_check_adam_trajectory(records)) == 1


@pytest.mark.parametrize("missing_policy", [True, False])
def test_optimizer_observer_unused_declaration_rejects_unexpected_gradient_membership(
    missing_policy,
):
    trainer = _trainer_with_unused_role_output()
    records = {"updates": [], "backward_losses": []}
    with observe_optimizer_steps(
        trainer,
        records,
        cp_size=1,
        is_saving=lambda: False,
        inactive_parameter_names=frozenset({"inactive"}),
    ):
        loss = (
            torch.zeros((), requires_grad=True)
            if missing_policy
            else trainer.model_bundle["policy"].sum() + trainer.model_bundle["inactive"].sum()
        )
        trainer.accelerator.backward(loss)
        with pytest.raises(AssertionError):
            trainer.accelerator.clip_grad_norm_(trainer.model_bundle.parameters(), 0.3)


@pytest.mark.parametrize("corrupt_state", [True, False])
def test_optimizer_observer_rejects_unused_parameter_or_optimizer_mutation(corrupt_state):
    trainer = _trainer_with_unused_role_output()
    records = {"updates": [], "backward_losses": []}
    step = trainer.optimizer.step

    def corrupt_step():
        step()
        parameter = trainer.model_bundle["inactive"]
        if corrupt_state:
            trainer.optimizer.optimizer.state[parameter]["unexpected"] = torch.ones_like(parameter)
        else:
            with torch.no_grad():
                parameter.add_(0.1)

    trainer.optimizer.step = corrupt_step
    with observe_optimizer_steps(
        trainer,
        records,
        cp_size=1,
        is_saving=lambda: False,
        inactive_parameter_names=frozenset({"inactive"}),
    ):
        trainer.accelerator.backward(trainer.model_bundle["policy"].square().mean())
        trainer.accelerator.clip_grad_norm_(trainer.model_bundle.parameters(), 0.3)
        with pytest.raises(AssertionError):
            trainer.optimizer.step()
