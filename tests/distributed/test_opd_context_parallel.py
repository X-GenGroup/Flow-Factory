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

"""Native multi-teacher OPD CP updates, evaluation and exact-state continuation."""

import copy
import json
import os
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from online_optimizer_observations import observe_optimizer_steps
from opd_observations import observe_objectives
from test_context_parallel_state import _assert_ema_trajectory, _cpu_tree
from test_context_parallel_training import _compare, _compare_bf16_updates
from test_model_preparation import initialize_cuda, runtime_observation
from wan_opd_fixture import build_fixture, checkpoint_run_name, make_config, teacher_names


def _state(trainer):
    """Capture the shared state fields, gathering offloaded snapshots on their CUDA mesh.

    CPU DTensor snapshots retain an NCCL mesh. Only the observation copy moves to
    the accelerator before gathering; the production snapshot remains offloaded.
    """
    from torch.distributed.tensor import DTensor

    def runtime_cpu(value):
        if isinstance(value, DTensor) and value.device.type == "cpu":
            return value.detach().to(trainer.accelerator.device).full_tensor().cpu().clone()
        if isinstance(value, dict):
            return {key: runtime_cpu(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(runtime_cpu(item) for item in value)
        return _cpu_tree(value)

    parameters = dict(trainer.model_bundle.named_parameters())
    names = {id(parameter): name for name, parameter in parameters.items()}
    optimizer = trainer.optimizer.optimizer
    return {
        "parameters": _cpu_tree(parameters),
        "moments": {name: _cpu_tree(optimizer.state.get(p, {})) for name, p in parameters.items()},
        "runtime": runtime_cpu(trainer.runtime_state.state_dict()),
        "ema_names": [names[id(p)] for p in trainer.adapter._ema_tracked_parameters()],
        "cpu_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }


def _assert_teachers(before, after):
    for name in teacher_names:
        _compare(
            before["runtime"]["children"][name], after["runtime"]["children"][name], exact=True
        )


def _worker(rank, store_path, result_dir, options):
    os.environ.update(
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP="WanTransformerBlock",
        FSDP_CPU_RAM_EFFICIENT_LOADING="false",
        FSDP_RESHARD_AFTER_FORWARD="true",
        FSDP_ACTIVATION_CHECKPOINTING="true",
    )
    world = options["dp_size"] * options["cp_size"] * options["tp_size"]
    initialize_cuda(rank, store_path, world_size=world)
    result = {"rank": rank, **options, "model_family": "wan", "algorithm": "diffusion-opd"}
    try:
        from flow_factory.trainers.loader import load_trainer

        trainer = load_trainer(make_config(Path(options["fixture"]), result_dir, options))
        result["runtime"] = runtime_observation(trainer.accelerator)
        loaded = _state(trainer)
        for name in teacher_names:
            expected = torch.load(
                Path(options["fixture"]) / (name + "-expected.pt"), weights_only=True
            )
            actual = loaded["runtime"]["children"][name]["ema_parameters"]
            assert len(actual) == len(expected)
            for value, reference in zip(actual, expected, strict=True):
                _compare(value, reference.to(value.dtype), exact=True)
        records = {
            "initial_parameters": _cpu_tree(dict(trainer.model_bundle.named_parameters())),
            "updates": [],
            "backward_losses": [],
            "objectives": [],
            "replay_inputs": [],
            "cache_events": [],
            "metrics": [],
            "packed_metrics": [],
            "samples": [],
            "evaluations": [],
        }
        saving, saved = False, {}
        save, evaluate, log_data = (
            trainer._save_exact_training_state,
            trainer.evaluate,
            trainer.log_data,
        )

        def observed_save(path):
            nonlocal saving
            saved[trainer.cycle_index] = _state(trainer)
            saving = True
            try:
                return save(path)
            finally:
                saving = False

        def observed_evaluate(*args, **kwargs):
            before = _state(trainer)
            result = evaluate(*args, **kwargs)
            after = _state(trainer)
            _compare(after["parameters"], before["parameters"], exact=True)
            _compare(after["moments"], before["moments"], exact=True)
            _assert_teachers(before, after)
            records["evaluations"].append(
                {"cycle": trainer.cycle_index, "parameters_preserved": True}
            )
            return result

        def observed_metrics(data, **kwargs):
            selected = {
                key: _cpu_tree(value)
                for key, value in data.items()
                if key.startswith("train/distill_loss")
            }
            if selected:
                records["metrics"].append(selected)
            return log_data(data, **kwargs)

        def forbidden_feedback(*args, **kwargs):
            raise AssertionError("ONLINE_NO_FEEDBACK must not invoke training feedback")

        generate = trainer.generate_samples

        def observed_generate(**kwargs):
            assert kwargs["reward_buffer"] is None and kwargs["compute_log_prob"] is False
            return generate(**kwargs)

        assert not trainer.reward_models and trainer.reward_buffer is None
        trainer.prepare_feedback = forbidden_feedback
        trainer.generate_samples = observed_generate
        trainer._save_exact_training_state = observed_save
        trainer.evaluate = observed_evaluate
        trainer.log_data = observed_metrics
        with (
            observe_optimizer_steps(
                trainer, records, cp_size=options["cp_size"], is_saving=lambda: saving
            ),
            observe_objectives(trainer, records),
        ):
            trainer.start()
        final = _state(trainer)
        _assert_teachers(loaded, final)
        assert trainer.step == 2 and trainer.progress.rollout_iteration == 2
        assert trainer.progress.data_epoch == 0 and not records["backward_losses"]
        assert {row["source"] for row in records["objectives"]} == {"red", "blue"}
        assert all(
            any(torch.count_nonzero(value) for value in update["gradients"].values())
            for update in records["updates"]
        )
        assert len(records["evaluations"]) == (1 if options.get("resume") else 2)
        torch.save(
            {"records": records, "loaded": loaded, "final": final, "saved": saved},
            Path(result_dir, f"observations-{rank}.pt"),
        )
        result.update(
            status="pass",
            optimizer_steps=trainer.step,
            reward_calls=0,
            teacher_names=list(teacher_names),
            elementwise_comparison=True,
            numerical_comparison=True,
            nonzero_gradients=True,
            exact_teacher_snapshots=True,
            evaluation_calls=len(records["evaluations"]),
            execution_capability="wan-opd-cp-v1",
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


def _assert_logical_metrics(groups, options):
    """Count each logical sample once, despite its TP/CP peer copies."""
    degree = options["tp_size"] * options["cp_size"]
    for cycle in range(2):
        values = {name: [] for name in ("red", "blue")}
        for rank in range(0, len(groups), degree):
            rows = groups[rank]["records"]["objectives"]
            width = len(rows) // 2
            for row in rows[cycle * width : (cycle + 1) * width]:
                values[row["source"]].append(float(row["loss"]))
        expected = {
            f"train/distill_loss_teacher_{source}": sum(items) / len(items)
            for source, items in values.items()
        }
        all_values = [value for items in values.values() for value in items]
        expected["train/distill_loss"] = sum(all_values) / len(all_values)
        for rank in groups:
            metric = rank["records"]["metrics"][cycle]
            assert metric.keys() == expected.keys()
            for key, value in expected.items():
                assert float(metric[key]) == pytest.approx(value, rel=2e-6, abs=1e-8)
            packed = rank["records"]["packed_metrics"][cycle]["global"]
            for index, source in enumerate(("red", "blue")):
                assert float(packed[1, index]) == len(values[source])
                assert float(packed[0, index]) == pytest.approx(
                    sum(values[source]), rel=2e-6, abs=1e-8
                )


def test_cuda_wan_opd_lora_tp_cp_lifecycle(tmp_path):
    options = {
        "dp_size": 2,
        "cp_size": 2,
        "tp_size": 2,
        "mode": "lora",
        "algorithm": "diffusion-opd",
    }
    world = 8
    available = int(os.environ.get("FF_TEST_CUDA_RANKS", torch.cuda.device_count()))
    if available < world:
        pytest.skip("native OPD TP/SP/CP lifecycle qualification requires 8 CUDA ranks")
    root = tmp_path.resolve() / "fixture"
    build_fixture(root, options["tp_size"])
    options["fixture"] = str(root)
    groups = {}
    for leg in ("continuous", "resumed"):
        leg_options = dict(options)
        if leg == "resumed":
            # Exact restore must use the saved runtime children, without external
            # teacher loading. The configured teacher path identities stay unchanged.
            for name in teacher_names:
                (root / name).rename(root / (name + "-unavailable"))
            leg_options["resume"] = str(
                tmp_path / "continuous/saves" / checkpoint_run_name / "checkpoints/checkpoint-1"
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
    _assert_logical_metrics(groups["continuous"], options)
    for leg in ("continuous", "resumed"):
        expected_queries = [
            (row["teacher_name"], row["identity"][0][0], row["step"])
            for row in groups[leg][0]["records"]["cache_events"]
        ]
        for peer in groups[leg]:
            assert [
                (row["teacher_name"], row["identity"][0][0], row["step"])
                for row in peer["records"]["cache_events"]
            ] == expected_queries
    continuation = []
    for rank, (continuous, resumed) in enumerate(
        zip(groups["continuous"], groups["resumed"], strict=True)
    ):
        _compare(resumed["loaded"], continuous["saved"][1], exact=True)
        _assert_teachers(continuous["loaded"], resumed["final"])
        expected = continuous["records"]
        joined = copy.deepcopy(expected)
        for key in (
            "samples",
            "replay_inputs",
            "cache_events",
            "metrics",
            "packed_metrics",
            "objectives",
            "updates",
            "evaluations",
        ):
            cutoff = len(expected[key]) // 2
            joined[key] = expected[key][:cutoff] + resumed["records"][key]
            if key != "updates":
                _compare(joined[key], expected[key], exact=True)
        continuation.append({"rank": rank, "updates": _compare_bf16_updates(joined, expected)})
        _assert_ema_trajectory(continuous["loaded"], resumed["final"], joined["updates"])
        _compare(resumed["final"]["runtime"], continuous["final"]["runtime"], exact=True)
        for field in ("cpu_rng", "cuda_rng"):
            _compare(resumed["final"][field], continuous["final"][field], exact=True)
        # Continued arithmetic is checked by the shared Adam/CP bounds above.
        # Link the final checkpoint view to those actual observed endpoints exactly.
        for state, trajectory in ((continuous, expected), (resumed, joined)):
            last = trajectory["updates"][-1]
            _compare(state["final"]["parameters"], last["parameters"], exact=True)
            for name, value in state["final"]["moments"].items():
                _compare(value, last["moments"].get(name, {}), exact=True)
        # Mark exact resume only after the independent parent-side state checks.
        for leg in ("continuous", "resumed"):
            path = tmp_path / leg / f"rank-{rank}.json"
            row = json.loads(path.read_text())
            row["exact_resume"] = True
            path.write_text(json.dumps(row, indent=2))
    print(
        json.dumps(
            {
                "algorithm": "diffusion-opd",
                "model_family": "wan",
                "mode": "lora",
                "dp_size": 2,
                "cp_size": 2,
                "tp_size": 2,
                "exact_loaded_state": True,
                "frozen_teacher_snapshots": True,
                "no_training_reward_work": True,
                "logical_dp_metrics": True,
                "continuation": continuation,
            }
        )
    )
