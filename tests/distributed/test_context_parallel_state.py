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

"""Native CP training continues from complete-epoch checkpoints with EMA state."""

import copy
import json
import os
import subprocess
import sys
import traceback
from functools import wraps
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from test_context_parallel_training import _compare, _compare_bf16_updates, _config
from test_model_preparation import full_value, initialize_cuda, runtime_observation


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return full_value(value).cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(item) for item in value)
    return value


def _state(trainer):
    parameters = dict(trainer.model_bundle.named_parameters())
    names = {id(parameter): name for name, parameter in parameters.items()}
    optimizer = trainer.optimizer.optimizer
    return {
        "parameters": _cpu_tree(parameters),
        "moments": {name: _cpu_tree(optimizer.state.get(p, {})) for name, p in parameters.items()},
        "runtime": _cpu_tree(trainer.runtime_state.state_dict()),
        "ema_names": [names[id(p)] for p in trainer.adapter._ema_tracked_parameters()],
        "cpu_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }


def _assert_ema_trajectory(initial, final, updates):
    state = final["runtime"]["children"]["adapter_ema"]
    assert state["decay_schedule"] == "constant" and state["decay"] == 0.9
    assert state["num_updates"] == len(updates)
    initial_values = initial["runtime"]["children"]["adapter_ema"]["ema_parameters"]
    eps = torch.finfo(torch.float32).eps
    gamma = 4 * eps / (1 - 4 * eps)
    for name, start, actual in zip(
        initial["ema_names"], initial_values, state["ema_parameters"], strict=True
    ):
        expected, error = start.double(), torch.zeros_like(start, dtype=torch.float64)
        for update in updates:
            parameter = update["parameters"][name].double()
            error = 0.9 * error + gamma * (0.9 * expected.abs() + 0.1 * parameter.abs())
            expected = 0.9 * expected + 0.1 * parameter
        assert torch.all((actual.double() - expected).abs() <= error + 1e-30), name
    assert any(not torch.equal(a, b) for a, b in zip(initial_values, state["ema_parameters"]))


def _worker(rank, store_path, result_dir, options):
    os.environ.update(
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP="WanTransformerBlock",
        FSDP_CPU_RAM_EFFICIENT_LOADING="false",
        FSDP_RESHARD_AFTER_FORWARD="true",
        FSDP_ACTIVATION_CHECKPOINTING=str(options["checkpointing"]).lower(),
    )
    world = 4 * options["tp_size"]
    initialize_cuda(rank, store_path, world_size=world)
    result = {"rank": rank, **options}
    try:
        from flow_factory.trainers.loader import load_trainer

        trainer = load_trainer(
            _config(
                Path(options["fixture"]),
                result_dir,
                {
                    **options,
                    "ema_decay": 0.9,
                    "ema_update_interval": 1,
                    "save_model_only": False,
                    "save_freq": 1,
                },
            )
        )
        result["runtime"] = runtime_observation(trainer.accelerator)
        loaded = _state(trainer)
        assert "adapter_ema" in loaded["runtime"]["children"]
        records = {
            "initial_parameters": {
                name: full_value(p).cpu().clone()
                for name, p in trainer.model_bundle.named_parameters()
            },
            "batches": [],
            "forwards": [],
            "updates": [],
        }
        optimize = trainer.optimize_batch
        forward = trainer.adapter.forward
        step = trainer._apply_optimizer_step
        optimizer_step = trainer.optimizer.step
        clipped = []

        def observed_optimizer_step(*args, **kwargs):
            clipped.append(
                {
                    name: full_value(p.grad).cpu().clone()
                    for name, p in trainer.model_bundle.named_parameters()
                    if p.grad is not None
                }
            )
            return optimizer_step(*args, **kwargs)

        def observed_batch(batch):
            records["batches"].append(batch.record_ids)
            return optimize(batch)

        @wraps(forward)
        def observed_forward(*args, **kwargs):
            output = forward(*args, **kwargs)
            records["forwards"].append(
                {
                    "t": kwargs["t"].detach().cpu().clone(),
                    "latents": kwargs["latents"].detach().cpu().clone(),
                    "prompt_embeds": kwargs["prompt_embeds"].detach().cpu().clone(),
                    "velocity": output.velocity.detach().cpu().clone(),
                }
            )
            return output

        def observed_step(loss_info):
            optimizer = trainer.optimizer.optimizer
            assert len(optimizer.param_groups) == 1
            group = optimizer.param_groups[0]
            assert not group["amsgrad"] and not group["maximize"]
            settings = {key: group[key] for key in ("lr", "betas", "eps", "weight_decay")}
            settings["max_grad_norm"] = trainer.training_args.max_grad_norm
            parameters = dict(trainer.model_bundle.named_parameters())
            before = {name: full_value(p).cpu().clone() for name, p in parameters.items()}
            gradients = {
                name: full_value(p.grad).cpu().clone()
                for name, p in parameters.items()
                if p.grad is not None
            }
            expected_names = {name for name, p in parameters.items() if p.requires_grad}
            assert set(gradients) == expected_names
            assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
            assert any(torch.count_nonzero(g) for g in gradients.values())
            losses = torch.stack(loss_info["loss"]).cpu().clone()
            returned = step(loss_info)
            after = {name: full_value(p).cpu().clone() for name, p in parameters.items()}
            for name, p in parameters.items():
                if not p.requires_grad:
                    torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)
            moments = {
                name: {
                    key: full_value(value).cpu().clone()
                    for key, value in optimizer.state[p].items()
                }
                for name, p in parameters.items()
                if p.requires_grad
            }
            records["updates"].append(
                {
                    "gradients": gradients,
                    "parameters": after,
                    "moments": moments,
                    "loss": losses,
                    "deltas": {name: after[name] - before[name] for name in expected_names},
                    "clipped_gradients": clipped.pop(),
                    "optimizer": settings,
                }
            )
            return returned

        trainer.optimize_batch = observed_batch
        trainer.adapter.forward = observed_forward
        trainer._apply_optimizer_step = observed_step
        trainer.optimizer.step = observed_optimizer_step
        saved = {}
        save = trainer._save_exact_training_state

        def observed_save(path):
            saved[trainer.cycle_index] = _state(trainer)
            return save(path)

        trainer._save_exact_training_state = observed_save
        trainer.start()
        final = _state(trainer)
        assert (
            final["runtime"]["children"]["adapter_ema"]["num_updates"]
            > loaded["runtime"]["children"]["adapter_ema"]["num_updates"]
        )
        assert trainer.step == options["epochs"] * 2
        assert trainer.progress.data_epoch == options["epochs"]
        assert trainer.progress.rollout_iteration == 0
        assert (
            saved
            and len(records["updates"]) == (options["epochs"] - bool(options.get("resume"))) * 2
        )
        assert any(
            not torch.equal(final["parameters"][name], loaded["parameters"][name])
            for name in final["parameters"]
        )
        torch.save(
            {"loaded": loaded, "saved": saved, "final": final, "records": records},
            Path(result_dir, f"observations-{rank}.pt"),
        )
        result.update(
            status="pass",
            optimizer_steps=trainer.step,
            data_epochs=trainer.progress.data_epoch,
            execution_capability="wan-cp-state-v1",
            elementwise_comparison=True,
            nonzero_gradients=True,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_cuda_wan_sft_cp_state_resume(tmp_path, mode, tp_size):
    world = 4 * tp_size
    if torch.cuda.device_count() < world:
        pytest.skip(f"native CP state continuation requires {world} CUDA ranks")
    root = tmp_path.resolve() / "fixture"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("wan_sft_fixture.py")),
            str(root),
            str(2 * tp_size),
        ],
        check=True,
    )
    common = {
        "fixture": str(root),
        "cp_size": 2,
        "tp_size": tp_size,
        "sequence_parallel": tp_size > 1,
        "mode": mode,
        "checkpointing": True,
    }
    groups = {}
    for leg, epochs in (("continuous", 2), ("first", 1), ("resumed", 2)):
        directory = tmp_path / leg
        options = {**common, "epochs": epochs}
        if leg == "resumed":
            options["resume"] = str(tmp_path / "first/saves/wan-sft/checkpoints/checkpoint-1")
        rows = run_gloo_ranks(_worker, directory, world, options)
        for row in rows:
            assert "error" not in row, row.get("error")
            assert row["status"] == "pass"
        groups[leg] = [
            torch.load(directory / f"observations-{rank}.pt", weights_only=True)
            for rank in range(world)
        ]
    residuals = []
    for rank in range(world):
        continuous, first, resumed = (
            groups[name][rank] for name in ("continuous", "first", "resumed")
        )
        # Loaded model, every optimizer moment, runtime child and device RNG are
        # checkpoint state: all must match exactly before the resumed update.
        _compare(resumed["loaded"], first["saved"][1], exact=True)
        joined = copy.deepcopy(first["records"])
        for key in ("batches", "forwards", "updates"):
            joined[key] += resumed["records"][key]
        expected = continuous["records"]
        _compare(joined["initial_parameters"], expected["initial_parameters"], exact=True)
        _compare(joined["batches"], expected["batches"], exact=True)
        for left, right in zip(joined["forwards"], expected["forwards"]):
            for key in ("t", "latents", "prompt_embeds"):
                _compare(left[key], right[key], exact=True)
        # Reuse the independently qualified clipping/Adam and unchanged global
        # gradient/update bounds over the entire four-update trajectory.
        residuals.append(_compare_bf16_updates(joined, expected))
        _assert_ema_trajectory(first["loaded"], resumed["final"], joined["updates"])
        _assert_ema_trajectory(continuous["loaded"], continuous["final"], expected["updates"])
        _compare(resumed["final"]["runtime"], continuous["final"]["runtime"])
    print(
        json.dumps(
            {
                "mode": mode,
                "tp_size": tp_size,
                "optimizer_steps": 4,
                "exact_loaded_state": True,
                "residuals": residuals,
            }
        )
    )
