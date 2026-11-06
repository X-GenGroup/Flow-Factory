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

"""Native DCP sparse AdamW state and Accelerate lifecycle ownership on CPU."""

import copy
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from accelerate.utils import DistributedType
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.fsdp import StateDictType

from flow_factory.trainers.common import fsdp_optimizer_checkpoint as checkpoint
from flow_factory.trainers.loader import Accelerator


def _pair(*, amsgrad=False):
    torch.manual_seed(73)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, amsgrad=amsgrad)
    return model, optimizer


def _step(model, optimizer, *, activate_bias):
    optimizer.zero_grad(set_to_none=True)
    loss = model.weight.square().sum()
    if activate_bias:
        loss = loss + model.bias.square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _save(tmp_path, model, optimizer, mutate=None):
    state = get_optimizer_state_dict(model, optimizer)
    if mutate is not None:
        mutate(state)
    dcp.save({"optimizer": state}, checkpoint_id=tmp_path / "optimizer_0")


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("amsgrad", [False, True])
def test_sparse_resume_matches_continuation_and_first_activation(tmp_path, warm, amsgrad):
    model, optimizer = _pair(amsgrad=amsgrad)
    _step(model, optimizer, activate_bias=False)
    _step(model, optimizer, activate_bias=False)
    assert model.bias not in optimizer.state
    _save(tmp_path, model, optimizer)
    restored, target = _pair(amsgrad=amsgrad)
    if warm:
        _step(restored, target, activate_bias=True)
    restored.load_state_dict(model.state_dict())
    before = copy.deepcopy(restored.state_dict())
    loader = checkpoint.inspect_sparse_adamw_checkpoint(restored, target, tmp_path, 0)
    assert loader is not None
    loader.load()
    _assert_nested_equal(before, restored.state_dict())
    _assert_nested_equal(optimizer.state_dict(), target.state_dict())
    assert restored.bias not in target.state
    _step(model, optimizer, activate_bias=False)
    _step(restored, target, activate_bias=False)
    _assert_nested_equal(model.state_dict(), restored.state_dict())
    _assert_nested_equal(optimizer.state_dict(), target.state_dict())
    _step(model, optimizer, activate_bias=True)
    _step(restored, target, activate_bias=True)
    _assert_nested_equal(model.state_dict(), restored.state_dict())
    _assert_nested_equal(optimizer.state_dict(), target.state_dict())
    assert target.state[restored.bias]["step"].item() == 1


def test_dense_checkpoint_uses_upstream_unchanged(tmp_path):
    model, optimizer = _pair()
    _step(model, optimizer, activate_bias=True)
    _save(tmp_path, model, optimizer)
    target = torch.optim.AdamW(model.parameters(), lr=0.03)
    assert checkpoint.inspect_sparse_adamw_checkpoint(model, target, tmp_path, 0) is None
    assert not target.state
    assert all(parameter.grad is None for parameter in model.parameters())


def test_non_adamw_checkpoint_is_not_intercepted(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    assert checkpoint.inspect_sparse_adamw_checkpoint(model, optimizer, tmp_path, 0) is None
    assert not (tmp_path / "optimizer_0").exists()


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda state: state["state"]["weight"].pop("exp_avg"), "incomplete or unexpected"),
        (
            lambda state: state["state"]["weight"].update(unexpected=torch.zeros(2, 2)),
            "incomplete or unexpected",
        ),
        (
            lambda state: state["state"].update(unknown=state["state"].pop("weight")),
            "unknown parameter",
        ),
        (lambda state: state["param_groups"][0]["params"].reverse(), "ownership mismatch"),
        (lambda state: state["param_groups"][0]["params"].append("extra"), "ownership mismatch"),
    ],
)
def test_saved_schema_rejects_corruption_before_optimizer_mutation(tmp_path, mutation, error):
    model, optimizer = _pair()
    _step(model, optimizer, activate_bias=False)
    _save(tmp_path, model, optimizer, mutation)
    target = torch.optim.AdamW(model.parameters(), lr=0.03)
    state, groups = target.state, target.param_groups
    with pytest.raises(ValueError, match=error):
        checkpoint.inspect_sparse_adamw_checkpoint(model, target, tmp_path, 0)
    assert target.state is state and not state
    assert target.param_groups is groups
    assert all(parameter.grad is None for parameter in model.parameters())


def test_wrong_moment_shape_rejected_without_mutation(tmp_path):
    model, optimizer = _pair()
    _step(model, optimizer, activate_bias=False)
    _save(
        tmp_path,
        model,
        optimizer,
        lambda state: state["state"]["weight"].update(exp_avg=torch.zeros(3)),
    )
    state, groups = optimizer.state, optimizer.param_groups
    before = copy.deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError, match="tensor schema mismatch"):
        checkpoint.inspect_sparse_adamw_checkpoint(model, optimizer, tmp_path, 0)
    assert optimizer.state is state and optimizer.param_groups is groups
    _assert_nested_equal(before, optimizer.state_dict())


@pytest.mark.parametrize("phase", ["initialize", "load", "apply"])
@pytest.mark.parametrize("warm", [False, True])
def test_failure_restores_state_groups_gradients_and_parameters(tmp_path, monkeypatch, phase, warm):
    source, source_optimizer = _pair()
    _step(source, source_optimizer, activate_bias=False)
    _save(tmp_path, source, source_optimizer)
    model, optimizer = _pair()
    if warm:
        _step(model, optimizer, activate_bias=True)
    loader = (
        None
        if phase == "initialize"
        else checkpoint.inspect_sparse_adamw_checkpoint(model, optimizer, tmp_path, 0)
    )
    model.weight.grad = torch.ones_like(model.weight)
    gradient = model.weight.grad
    state, groups = optimizer.state, optimizer.param_groups
    before_state = copy.deepcopy(optimizer.state_dict())
    before_model = copy.deepcopy(model.state_dict())
    if phase == "initialize":

        def fail_step(*args, **kwargs):
            optimizer.state[model.weight]["partial"] = torch.ones(1)
            optimizer.param_groups[0]["lr"] = 99
            raise RuntimeError("injected")

        monkeypatch.setattr(optimizer, "step", fail_step)
    elif phase == "load":
        monkeypatch.setattr(
            checkpoint.dcp,
            "load",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
        )
    else:

        def fail_apply(*args, **kwargs):
            optimizer.state[model.bias] = {"partial": torch.ones(1)}
            optimizer.param_groups[0]["lr"] = 99
            raise RuntimeError("injected")

        monkeypatch.setattr(checkpoint, "set_optimizer_state_dict", fail_apply)
    with pytest.raises(RuntimeError, match="injected"):
        if phase == "initialize":
            checkpoint.inspect_sparse_adamw_checkpoint(model, optimizer, tmp_path, 0)
        else:
            loader.load()
    assert optimizer.state is state and optimizer.param_groups is groups
    assert model.weight.grad is gradient and model.bias.grad is None
    _assert_nested_equal(before_state, optimizer.state_dict())
    _assert_nested_equal(before_model, model.state_dict())


def _accelerator(model, optimizer):
    accelerator = object.__new__(Accelerator)
    accelerator.state = SimpleNamespace(
        distributed_type=DistributedType.FSDP,
        is_fsdp2=True,
        fsdp_plugin=SimpleNamespace(
            fsdp_version=2, state_dict_type=StateDictType.SHARDED_STATE_DICT
        ),
        process_index=0,
        num_processes=1,
    )
    accelerator._models, accelerator._optimizers = [model], [optimizer]
    accelerator._schedulers, accelerator._dataloaders, accelerator._custom_objects = [], [], []
    accelerator._load_model_state_pre_hook = OrderedDict()
    accelerator.scaler = None
    return accelerator


@pytest.mark.parametrize("tensor", [torch.zeros(3), torch.zeros(2, 2, dtype=torch.float64)])
def test_sparse_tensor_schema_fails_before_native_model_restore(tmp_path, monkeypatch, tensor):
    import accelerate.accelerator as upstream

    model, optimizer = _pair()
    _step(model, optimizer, activate_bias=False)
    _save(tmp_path, model, optimizer, lambda state: state["state"]["weight"].update(exp_avg=tensor))
    accelerator = _accelerator(model, torch.optim.AdamW(model.parameters(), lr=0.03))
    registered = accelerator._optimizers
    monkeypatch.setattr(
        upstream,
        "load_fsdp_model",
        lambda *args, **kwargs: pytest.fail("model mutated before preflight"),
    )
    with pytest.raises(ValueError, match="tensor schema mismatch"):
        accelerator.load_state(str(tmp_path))
    assert accelerator._optimizers is registered and not registered[0].state
    assert not accelerator._load_model_state_pre_hook


@pytest.mark.parametrize("fail", [False, True])
def test_native_accelerator_order_and_registration_cleanup(tmp_path, monkeypatch, fail):
    import accelerate.accelerator as upstream

    model, optimizer = _pair()
    _step(model, optimizer, activate_bias=False)
    _save(tmp_path, model, optimizer)
    target = torch.optim.AdamW(model.parameters(), lr=0.03)
    accelerator = _accelerator(model, target)
    registered = accelerator._optimizers
    events = []
    accelerator.register_load_state_pre_hook(lambda *args: events.append("existing_hook"))
    existing_hooks = dict(accelerator._load_model_state_pre_hook)
    accelerator.scaler = SimpleNamespace(
        _device="cpu",
        load_state_dict=lambda state: events.append("scaler_load"),
        _lazy_init_scale_growth_tracker=lambda device: events.append("scaler_initialize"),
    )
    torch.save({}, tmp_path / upstream.SCALER_NAME)
    monkeypatch.setattr(upstream.logger, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(upstream, "load_fsdp_model", lambda *args, **kwargs: events.append("model"))
    monkeypatch.setattr(
        upstream,
        "load_fsdp_optimizer",
        lambda *args, **kwargs: pytest.fail("duplicate native optimizer load"),
    )

    def generic_load(*args, **kwargs):
        assert args[2] == [] and args[6] is None
        events.append("scheduler_rng")
        return {}

    monkeypatch.setattr(upstream, "load_accelerator_state", generic_load)
    original_load = checkpoint.SparseAdamWCheckpoint.load

    def restore(loader):
        events.append("optimizer")
        if fail:
            raise RuntimeError("injected")
        original_load(loader)

    monkeypatch.setattr(checkpoint.SparseAdamWCheckpoint, "load", restore)
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            accelerator.load_state(str(tmp_path))
    else:
        accelerator.load_state(str(tmp_path))
    assert events == ["model", "scaler_load", "scaler_initialize", "existing_hook", "optimizer"] + (
        [] if fail else ["scheduler_rng"]
    )
    assert accelerator._optimizers is registered
    assert dict(accelerator._load_model_state_pre_hook) == existing_hooks


@pytest.mark.parametrize(
    "backend,version,state_type",
    [
        (DistributedType.NO, 2, StateDictType.SHARDED_STATE_DICT),
        (DistributedType.FSDP, 1, StateDictType.SHARDED_STATE_DICT),
        (DistributedType.FSDP, 2, StateDictType.FULL_STATE_DICT),
    ],
)
def test_unrelated_accelerator_modes_pass_through(
    tmp_path, monkeypatch, backend, version, state_type
):
    import flow_factory.trainers.loader as loader

    model, optimizer = _pair()
    accelerator = _accelerator(model, optimizer)
    accelerator.state.distributed_type = backend
    accelerator.state.fsdp_plugin.fsdp_version = version
    accelerator.state.fsdp_plugin.state_dict_type = state_type
    marker = object()
    monkeypatch.setattr(loader._UpstreamAccelerator, "load_state", lambda *args, **kwargs: marker)
    assert accelerator.load_state(str(tmp_path)) is marker
