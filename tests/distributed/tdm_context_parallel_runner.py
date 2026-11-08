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

"""Bounded recipe-driven native trajectory/multirole CP continuation runner."""

import copy
import importlib
import json
import os
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from online_optimizer_observations import observe_optimizer_steps
from test_context_parallel_state import _assert_ema_trajectory, _cpu_tree, _state
from test_context_parallel_training import _compare, _compare_bf16_updates
from test_model_preparation import initialize_cuda, runtime_observation


def _worker(rank, store_path, result_dir, options):
    recipe = importlib.import_module(options["recipe_module"])
    os.environ.update(
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP=",".join(recipe.wrap_classes),
        FSDP_CPU_RAM_EFFICIENT_LOADING="false",
        FSDP_RESHARD_AFTER_FORWARD="true",
        FSDP_ACTIVATION_CHECKPOINTING="true",
    )
    world = options["dp_size"] * options["cp_size"] * options["tp_size"]
    initialize_cuda(rank, store_path, world_size=world)
    result = {"rank": rank, **options, "model_family": "wan"}
    try:
        from flow_factory.trainers.loader import load_trainer

        trainer = load_trainer(recipe.make_config(Path(options["fixture"]), result_dir, options))
        result["runtime"] = runtime_observation(trainer.accelerator)
        loaded = _state(trainer)
        records = {
            "initial_parameters": _cpu_tree(dict(trainer.model_bundle.named_parameters())),
            "updates": [],
            "backward_losses": [],
            "objectives": [],
            "samples": [],
            "evaluations": [],
        }
        saving, saved = False, {}
        save, evaluate, sample = (
            trainer._save_exact_training_state,
            trainer.evaluate,
            trainer.sample_batch,
        )

        def observed_save(path):
            nonlocal saving
            assert not trainer._tdm_generation_provenance
            saved[trainer.cycle_index] = _state(trainer)
            saving = True
            try:
                return save(path)
            finally:
                saving = False

        def observed_evaluate(*args, **kwargs):
            before = _state(trainer)
            returned = evaluate(*args, **kwargs)
            after = _state(trainer)
            for field in ("parameters", "moments"):
                _compare(after[field], before[field], exact=True)
            records["evaluations"].append(
                {"cycle": trainer.cycle_index, "parameters_preserved": True}
            )
            return returned

        def observed_sample(*args, **kwargs):
            samples = sample(*args, **kwargs)
            identities = [s.sampling_sample_id for s in samples]
            assert all(value is None for value in identities) or all(
                value is not None for value in identities
            )
            if identities and identities[0] is not None:
                records["samples"].append(
                    {
                        "cycle": trainer.cycle_index,
                        "ids": tuple((s.source, int(s.sampling_sample_id)) for s in samples),
                    }
                )
            return samples

        trainer._save_exact_training_state, trainer.evaluate, trainer.sample_batch = (
            observed_save,
            observed_evaluate,
            observed_sample,
        )
        with (
            observe_optimizer_steps(
                trainer, records, cp_size=options["cp_size"], is_saving=lambda: saving
            ),
            recipe.observe_training(trainer, records),
        ):
            trainer.start()
        final = _state(trainer)
        assert trainer.step == 2 and trainer.progress.rollout_iteration == 2
        assert not records["backward_losses"] and not trainer._tdm_generation_provenance
        assert len(records["evaluations"]) == (1 if options.get("resume") else 2)
        assert all(
            any(torch.count_nonzero(g) for g in row["gradients"].values())
            for row in records["updates"]
        )
        recipe.assert_worker(trainer, records, loaded, final, options)
        torch.save(
            {"records": records, "loaded": loaded, "final": final, "saved": saved},
            Path(result_dir, f"observations-{rank}.pt"),
        )
        result.update(
            status="pass",
            optimizer_steps=trainer.step,
            nonzero_gradients=True,
            elementwise_comparison=True,
            numerical_comparison=True,
            evaluation_calls=len(records["evaluations"]),
            execution_capability=recipe.execution_capability,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


def qualify_lifecycle(tmp_path, *, recipe_module, options):
    """Run two acquisition cycles and resume cycle two in fresh native processes."""
    recipe = importlib.import_module(recipe_module)
    options = {**options, "recipe_module": recipe_module}
    world = options["dp_size"] * options["cp_size"] * options["tp_size"]
    available = int(os.environ.get("FF_TEST_CUDA_RANKS", torch.cuda.device_count()))
    if available < world:
        pytest.skip(f"native trajectory CP lifecycle needs {world} CUDA ranks")
    root = tmp_path.resolve() / "fixture"
    recipe.build_fixture(root, options["tp_size"])
    options["fixture"] = str(root)
    groups = {}
    for leg in ("continuous", "resumed"):
        leg_options = dict(options)
        if leg == "resumed":
            leg_options["resume"] = str(
                tmp_path
                / "continuous/saves"
                / recipe.checkpoint_run_name
                / "checkpoints/checkpoint-1"
            )
        directory = tmp_path / leg
        rows = run_gloo_ranks(_worker, directory, world, leg_options)
        for row in rows:
            assert "error" not in row, row.get("error")
            assert row["status"] == "pass"
        groups[leg] = [
            torch.load(directory / f"observations-{rank}.pt", weights_only=True)
            for rank in range(world)
        ]
    continuation = []
    for rank, (continuous, resumed) in enumerate(
        zip(groups["continuous"], groups["resumed"], strict=True)
    ):
        _compare(resumed["loaded"], continuous["saved"][1], exact=True)
        expected = continuous["records"]
        joined = copy.deepcopy(expected)
        for key, value in expected.items():
            if isinstance(value, list):
                # The explicit recipe has equal phase geometry in both complete cycles.
                assert len(value) % 2 == 0, key
                joined[key] = value[: len(value) // 2] + resumed["records"][key]
                assert len(joined[key]) == len(value), key
        role_metrics = {}
        for role in sorted({row["role"] for row in expected["updates"]}):
            a = {
                "initial_parameters": joined["initial_parameters"],
                "updates": [u for u in joined["updates"] if u["role"] == role],
            }
            b = {
                "initial_parameters": expected["initial_parameters"],
                "updates": [u for u in expected["updates"] if u["role"] == role],
            }
            role_metrics[role] = _compare_bf16_updates(a, b)
        _assert_ema_trajectory(
            continuous["loaded"],
            resumed["final"],
            [u for u in joined["updates"] if u["role"] == "generator"],
        )
        for field in ("runtime", "cpu_rng", "cuda_rng"):
            _compare(resumed["final"][field], continuous["final"][field], exact=True)
        # An algorithm-specific observer owns stronger or additional state checks.
        recipe.assert_continuation(continuous, resumed, joined, options)
        for state, trajectory in ((continuous, expected), (resumed, joined)):
            _compare(
                state["final"]["parameters"], trajectory["updates"][-1]["parameters"], exact=True
            )
            latest = {}
            for update in trajectory["updates"]:
                latest.update(update["moments"])
            for name, value in state["final"]["moments"].items():
                _compare(value, latest.get(name, {}), exact=True)
        continuation.append({"rank": rank, "roles": role_metrics})
        for leg in ("continuous", "resumed"):
            path = tmp_path / leg / f"rank-{rank}.json"
            row = json.loads(path.read_text())
            row["exact_resume"] = True
            path.write_text(json.dumps(row, indent=2))
    print(
        json.dumps(
            {
                "algorithm": options["algorithm"],
                "exact_loaded_state": True,
                "continuation": continuation,
            }
        )
    )
    return groups
