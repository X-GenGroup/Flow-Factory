# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""A late future failure unwinds R1's real partially accumulated role phase."""

import time
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from test_role_optimization import DeterministicAccelerator, _config

from flow_factory.hparams import (
    AdamWOptimizerArguments,
    MultiOptimizerArguments,
    TDMR1TrainingArguments,
)
from flow_factory.trainers.distillation.tdm_r1 import TDMR1Trainer
from flow_factory.trainers.execution import TrainingProgress
from flow_factory.trainers.role_optimization import OptimizationRole, RoleOptimizationCoordinator


def test_late_reward_failure_preserves_completed_fake_updates_and_rejects_partial_resume(tmp_path):
    trainer = object.__new__(TDMR1Trainer)
    trainer.progress = TrainingProgress()
    args = TDMR1TrainingArguments()
    trainer.training_args = SimpleNamespace(
        seed=17,
        required_trainable_roles=args.required_trainable_roles,
        role_update_plan=args.role_update_plan,
        gradient_accumulation_steps=4,
        per_device_batch_size=1,
        get_num_train_timesteps=lambda config: 2,
        reward_optimization_overlap=True,
        reward_optimization_overlap_mode="ordered",
        reward_optimization_overlap_poll_interval=0.001,
    )
    trainer.config = SimpleNamespace(
        data_args=SimpleNamespace(sampler_type="rank_local"),
        optimizer_args=MultiOptimizerArguments(
            optimizer_configs=[AdamWOptimizerArguments(name="fake", update_frequency=2)]
        ),
    )
    trainer.log_args = SimpleNamespace(verbose=False)
    accelerator = DeterministicAccelerator((False, False, False, True) * 2 + (False, False))
    accelerator.device, accelerator.num_processes, accelerator.process_index = (
        torch.device("cpu"),
        1,
        0,
    )
    accelerator.reduce = lambda tensor, reduction: tensor
    trainer.accelerator = accelerator
    bundle = torch.nn.ParameterDict(
        {
            name: torch.nn.Parameter(torch.tensor(float(i + 1)))
            for i, name in enumerate(("generator", "fake", "surrogate"))
        }
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": [bundle[name]], "role_name": name, "lr": 0.01}
            for name in ("generator", "fake", "surrogate")
        ]
    )
    roles = {
        name: OptimizationRole(
            config=_config(name), parameters=(bundle[name],), optimizer_group_ids=(i,)
        )
        for i, name in enumerate(("generator", "fake", "surrogate"))
    }
    trainer.optimization_roles = roles
    trainer.role_optimization = RoleOptimizationCoordinator(accelerator, bundle, optimizer, roles)
    scopes, snapshots, cleanup = [], [], []

    @contextmanager
    def variant(name):
        scopes.append(name)
        try:
            yield
        finally:
            assert scopes.pop() == name

    trainer.adapter = SimpleNamespace(
        train=lambda: None,
        use_component_variant=variant,
        has_variant_snapshot=lambda name: True,
        update_variant_snapshot=lambda *args: snapshots.append(args),
        ema_step=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("partial cycle may not update EMA")
        ),
    )
    trainer._build_boundary_units = lambda samples: [
        SimpleNamespace(samples=samples, boundary_index=i) for i in (1, 2)
    ]
    trainer._fake_boundary_loss = lambda unit: bundle["fake"].square()
    trainer._surrogate_boundary_loss = lambda unit: bundle["surrogate"].square()
    trainer.should_continue_training = lambda: trainer.cycle_index == 0
    trainer._run_periodic_cycle_boundaries = lambda: None
    trainer.cleanup = lambda *, wait=False: cleanup.append(wait)
    future = Future()
    samples = [object(), object()]
    tiles = [SimpleNamespace(tile_id=i, sample_indices=(i,)) for i in range(2)]
    plan = SimpleNamespace(tiles=tiles, samples_for=lambda tile, values: [values[tile.tile_id]])
    trainer._build_and_seal_reward_tile_plan = lambda samples: plan
    trainer._resolve_reward_overlap_tile_feedback = lambda tile, samples: None
    aborted = []

    def poll(pending):
        if future.done():
            future.result()
        return {0} & set(pending)

    trainer.reward_buffer = SimpleNamespace(
        poll_ready_tiles=poll, abort_streaming=lambda: aborted.append(True)
    )
    optimize = trainer._optimize_reward_overlap_tile

    def first_tile(tile, samples, context):
        optimize(tile, samples, context)
        assert tile.tile_id == 0 and roles["surrogate"].step == 0
        assert bundle["surrogate"].grad is not None
        future.set_exception(ValueError("late endpoint reward failure"))

    trainer._optimize_reward_overlap_tile = first_tile
    trainer._run_overlap_training_step = lambda: trainer._run_reward_optimization_overlap(
        samples, cycle_started=time.monotonic(), rollout_seconds=0.0
    )
    trainer.acquisition_driver = SimpleNamespace(
        prepare_cycle=lambda *args, **kwargs: None,
        run_cycle=lambda host, progress: host._run_training_step(),
    )
    before = {name: value.detach().clone() for name, value in bundle.items()}
    with pytest.raises(ValueError, match="late endpoint reward failure"):
        trainer.start()
    assert roles["fake"].step == 2
    assert roles["generator"].step == roles["surrogate"].step == 0
    assert not torch.equal(bundle["fake"], before["fake"])
    for name in ("generator", "surrogate"):
        assert torch.equal(bundle[name], before[name]) and bundle[name] not in optimizer.state
    assert trainer.step == trainer.cycle_index == 0
    assert trainer._acquisition_cycle_incomplete and not trainer._acquisition_cycle_active
    assert trainer.role_optimization._active_role_name is None and not scopes
    assert not snapshots and aborted == [True] and cleanup == [True]
    assert not trainer._tdm_generation_provenance
    # Abort preserves diagnostic partial gradients. They cannot silently leak into a retry.
    assert bundle["surrogate"].grad is not None
    with pytest.raises(RuntimeError, match="grad to be None"):
        with trainer.role_optimization.phase("surrogate"):
            pass
    with pytest.raises(RuntimeError, match="complete acquisition boundary"):
        trainer._save_exact_training_state(str(tmp_path / "partial"))
    assert not (tmp_path / "partial").exists()
