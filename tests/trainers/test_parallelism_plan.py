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

from types import SimpleNamespace

import pytest

from flow_factory.hparams import AdamWOptimizerArguments, ParallelismArguments
from flow_factory.trainers import loader


def _config(parallelism_args: ParallelismArguments) -> SimpleNamespace:
    return SimpleNamespace(
        mixed_precision="bf16",
        optimizer_args=(AdamWOptimizerArguments(name="base"),),
        parallelism_args=parallelism_args,
        model_args=SimpleNamespace(model_type="test"),
        log_args=SimpleNamespace(save_dir="/tmp", run_name="parallelism-plan-test"),
        training_args=SimpleNamespace(
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            seed=42,
            trainer_type="grpo",
            required_trainable_roles=None,
        ),
    )


def test_unconfigured_block_builds_no_plan() -> None:
    """Without an explicit dimension the launched world stays purely data parallel."""
    assert loader._build_parallelism_config(_config(ParallelismArguments())) is None


def test_configured_block_carries_the_requested_sizes() -> None:
    """Each configured dimension reaches Accelerate unchanged."""
    plan = loader._build_parallelism_config(_config(ParallelismArguments(tp_size=2)))

    assert plan is not None
    assert plan.tp_size == 2
    assert plan.dp_shard_size == 1
    assert plan.dp_replicate_size == 1


def test_load_trainer_forwards_the_plan_to_the_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mesh plan must reach the Accelerator constructor, not a later hook."""
    observed: dict = {}

    class Adapter:
        ddp_find_unused_parameters = False

    class AcceleratorReached(Exception):
        """Raised in place of a real Accelerator to stop the factory right after it."""

    def fake_accelerator(**kwargs):
        observed.update(kwargs)
        raise AcceleratorReached

    monkeypatch.setattr(loader, "get_model_adapter_class", lambda model_type: Adapter)
    monkeypatch.setattr(loader, "Accelerator", fake_accelerator)
    monkeypatch.setattr(loader, "load_model", lambda **kwargs: SimpleNamespace())

    with pytest.raises(AcceleratorReached):
        loader.load_trainer(_config(ParallelismArguments(dp_shard_size=2, tp_size=2)))

    passed = observed["parallelism_config"]
    assert passed is not None
    assert (passed.dp_shard_size, passed.tp_size) == (2, 2)
