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

"""CPU admission and negative controls for the exact native overlap recipes."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import context_parallel_overlap_runner as runner
import pytest
import torch
from context_parallel_overlap_runner import (
    GatedReward,
    RewardGates,
    _assert_final_optimizer_state,
    _assert_ordered_continuation,
    _expected_advantages,
    make_overlap_config,
)
from test_context_parallel_training import _compare, _optimizer_fixture
from wan_overlap_fixture import make_config

from flow_factory.hparams import RewardArguments
from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter
from flow_factory.rewards import RewardProcessor
from flow_factory.samples import BaseSample
from flow_factory.trainers.common.context_parallel import validate_context_parallel_config
from flow_factory.trainers.rl.grpo import GRPOTrainer


def _config(monkeypatch, tmp_path, *, tp_size=1, subgroup=False, scheduling_mode="ordered"):
    options = {
        "dp_size": 4 if subgroup else 2,
        "cp_size": 2,
        "tp_size": tp_size,
        "training_mode": "lora" if tp_size > 1 else "full",
        "scheduling_mode": scheduling_mode,
        "sampler_type": "subgroup_tile" if subgroup else "rank_local",
        "tiles": 4 if subgroup else 2,
    }
    monkeypatch.setenv("WORLD_SIZE", str(options["dp_size"] * 2 * tp_size))
    root = tmp_path / "native-overlap"
    return make_overlap_config(
        make_config(root, root / "results", options), root, root / "results", options
    )


@pytest.mark.parametrize("tp_size,subgroup", [(1, False), (2, False), (1, True)])
@pytest.mark.parametrize("scheduling_mode", ["ordered", "ready"])
def test_native_recipe_preserves_overlap_and_cp_contracts(
    monkeypatch, tmp_path, tp_size, subgroup, scheduling_mode
):
    config = _config(
        monkeypatch, tmp_path, tp_size=tp_size, subgroup=subgroup, scheduling_mode=scheduling_mode
    )
    GRPOTrainer.validate_reward_optimization_overlap(config)
    validate_context_parallel_config(config, Wan2_T2V_Adapter, GRPOTrainer)
    assert config.training_args.gradient_accumulation_steps == 4
    assert config.training_args.num_batches_per_epoch == (8 if subgroup else 4)
    assert config.training_args.max_epochs == 2
    assert config.eval_args.resolution == (32, 32)
    assert config.eval_args.num_frames == 5 and config.eval_args.eval_freq == 1
    assert len(list(config.optimizer_args)) == 1
    assert next(iter(config.optimizer_args)).learning_rate == 1e-3
    assert config.log_args.save_model_only is False


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("num_inner_epochs", 2, "num_inner_epochs=1"),
        ("shuffle_samples", True, "shuffle_samples=false"),
        ("global_std", True, "without acquisition-wide statistics"),
        ("gradient_accumulation_steps", 3, "complete gradient accumulation windows"),
    ],
)
def test_cp_overlap_keeps_structural_guards(monkeypatch, tmp_path, field, value, message):
    config = _config(monkeypatch, tmp_path)
    setattr(config.training_args, field, value)
    with pytest.raises(ValueError, match=message):
        GRPOTrainer.validate_reward_optimization_overlap(config)


def test_cp_overlap_requires_async_cpu_reward_clients(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    reward = next(iter(config.reward_args))
    reward.async_reward = False
    with pytest.raises(ValueError, match="async_reward=true"):
        GRPOTrainer.validate_reward_optimization_overlap(config)
    reward.async_reward = True
    reward.device = torch.device("cuda")
    with pytest.raises(ValueError, match="device='cpu'"):
        GRPOTrainer.validate_reward_optimization_overlap(config)


def test_cp_overlap_admission_does_not_inherit_concrete_capabilities(monkeypatch, tmp_path):
    from flow_factory.trainers.registry import _TRAINER_REGISTRY, get_trainer_class

    config = _config(monkeypatch, tmp_path)
    reached = []
    for name in _TRAINER_REGISTRY:
        trainer = get_trainer_class(name)
        if "context_parallel_training_capability" not in trainer.__dict__:
            with pytest.raises(ValueError, match="does not explicitly declare"):
                validate_context_parallel_config(config, Wan2_T2V_Adapter, trainer)
        elif trainer.reward_optimization_overlap_contract.supported:
            trainer.validate_reward_optimization_overlap(config)
            validate_context_parallel_config(config, Wan2_T2V_Adapter, trainer)
            reached.append(name)
    assert "grpo" in reached


def test_subgroup_oracle_rejects_cross_subgroup_members_and_peer_drift():
    gathered = []
    for rank in range(4):
        rows = [
            {
                "source": "a",
                "group": rank // 2,
                "sample": rank * 2 + member,
                "member": rank % 2 * 2 + member,
                "score": float(rank % 2 * 2 + member),
            }
            for member in range(2)
        ]
        gathered.extend([rows, deepcopy(rows)])
    kwargs = dict(model_degree=2, dp_size=4, group_size=4, subgroup_size=2)
    assert len(_expected_advantages(gathered, **kwargs)) == 8
    bad = deepcopy(gathered)
    bad[1][0]["score"] += 1
    with pytest.raises(AssertionError):
        _expected_advantages(bad, **kwargs)
    bad = deepcopy(gathered)
    bad[2:4], bad[4:6] = bad[4:6], bad[2:4]
    with pytest.raises(AssertionError):
        _expected_advantages(bad, **kwargs)


@pytest.mark.parametrize("source", [None, "a"])
@pytest.mark.parametrize("media_field,shape", [("video", (5, 3, 32, 32)), ("image", (3, 64, 64))])
def test_gated_reward_actual_processor_call_preserves_optional_source_and_tensor_media(
    source, media_field, shape
):
    """Exercise the exact filtered call that failed before the first native tile."""
    media = torch.full(shape, 0.25)
    fixture = SimpleNamespace(score_media=lambda value: float(value.double().mean()))
    gates = RewardGates({"tiles": 2}, 0, fixture)
    gates.row_to_tile[(source, 19)] = 0
    gates.plan_ready.set()
    gates.gates[0].set()
    accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=1, process_index=0)
    config = RewardArguments(name="score", device="cpu", batch_size=1)
    model = GatedReward(config, accelerator)
    model.controller, model.media_field = gates, media_field
    model.required_fields = (
        "source",
        "sampling_sample_id",
        "sampling_group_member_id",
        media_field,
    )
    processor = RewardProcessor(accelerator, {"score": model}, {"score": config}, verbose=False)
    sample = BaseSample(
        source=source, sampling_sample_id=19, sampling_group_member_id=2, **{media_field: media}
    )
    actual = processor._compute_pointwise_batch("score", model, [sample])
    torch.testing.assert_close(actual, torch.tensor([4.25]), rtol=0, atol=0)
    assert gates.scored == [(-1, source, 19, 4.25)]


def test_subgroup_recipe_rejects_a_dropped_weighted_source():
    gathered = []
    for rank in range(4):
        rows = [
            {
                "source": source,
                "group": rank // 2,
                "sample": rank * 2 + member,
                "member": rank % 2 * 2 + member,
                "score": float(rank % 2 * 2 + member),
            }
            for source in ("a", "b")
            for member in range(2)
        ]
        gathered.extend([rows, deepcopy(rows)])
    kwargs = dict(
        model_degree=2,
        dp_size=4,
        group_size=4,
        subgroup_size=2,
        source_rows_per_replica={"a": 2, "b": 2},
    )
    expected = _expected_advantages(gathered, **kwargs)
    assert expected[("a", 0)] == -expected[("b", 0)]
    for rows in gathered:
        for row in rows:
            if row["source"] == "b":
                row["source"] = "a"
                row["sample"] += 100
                row["group"] += 100
    with pytest.raises(AssertionError):
        _expected_advantages(gathered, **kwargs)


def _cpu_rank_pass_then_fail(rank, store_path, result_dir, options):
    del store_path, options
    torch.save({}, Path(result_dir, f"observations-{rank}.pt"))
    runner._publish_rank_result(Path(result_dir), rank, {"rank": rank, "status": "pass"})
    raise SystemExit(7)


def test_native_launcher_rejects_terminal_failure_after_pass_receipt(monkeypatch, tmp_path):
    """A real CPU child completes before the first liveness poll to force the race."""
    context = runner.mp.get_context("spawn")

    class CompletedBeforePolling:
        def __init__(self, **kwargs):
            self.process = context.Process(**kwargs)

        def start(self):
            self.process.start()
            self.process.join(timeout=30)
            assert not self.process.is_alive()

        def __getattr__(self, name):
            return getattr(self.process, name)

    monkeypatch.setattr(runner, "_worker", _cpu_rank_pass_then_fail)
    monkeypatch.setattr(
        runner.mp, "get_context", lambda _: SimpleNamespace(Process=CompletedBeforePolling)
    )
    with pytest.raises(AssertionError, match="terminal cleanup"):
        runner._run_native(tmp_path / "terminal", {"dp_size": 1, "cp_size": 1, "tp_size": 1})


def test_native_rank_receipt_is_published_atomically(monkeypatch, tmp_path):
    write_text = Path.write_text
    observed = []

    def before_publish(path, data, *args, **kwargs):
        assert path.suffix == ".tmp"
        assert not list(tmp_path.glob("rank-*.json"))
        observed.append(path.name)
        return write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", before_publish)
    runner._publish_rank_result(tmp_path, 0, {"status": "pass"})
    assert observed == ["rank-0.json.tmp"]
    assert (tmp_path / "rank-0.json").read_text() == '{\n  "status": "pass"\n}'
    assert not list(tmp_path.glob("*.tmp"))


def _ordered_continuation_fixture(*, roundoff=False, gradient_scale=1.0):
    """Use actual CPU AdamW steps, with a changed gradient only after the checkpoint."""
    gradients = [torch.linspace(-0.3, 0.4, 8) * scale for scale in (1.0, -0.9, 0.7, -0.5)]
    changed = [gradient.clone() for gradient in gradients]
    if roundoff:
        changed[2][0] = torch.nextafter(changed[2][0], torch.tensor(float("inf")))
    changed[2] *= gradient_scale
    expected, actual = _optimizer_fixture(gradients), _optimizer_fixture(changed)

    def cycle(index):
        rows = [
            {
                "source": "a",
                "sample": index * 4 + row,
                "group": index * 2 + row // 2,
                "member": row % 2,
                "score": float(row % 2),
            }
            for row in range(4)
        ]
        keys = [(row["source"], row["sample"]) for row in rows]
        return {
            "cycle": index,
            "rows": rows,
            "audit": [(key, transition) for key in keys for transition in (0, 1)],
            "expected_advantages": {key: float(2 * (row % 2) - 1) for row, key in enumerate(keys)},
            "drained": True,
            "tiles": [
                {
                    "tile": tile,
                    "samples": keys[tile * 2 : tile * 2 + 2],
                    "advantages": [-1.0, 1.0],
                    "pending": 2 if tile == 0 else 0,
                    "step": index * 2 + tile + 1,
                    "gas": 4,
                    "terms": [
                        (key, time, 4)
                        for key in keys[tile * 2 : tile * 2 + 2]
                        for time in (1000.0, 500.5)
                    ],
                }
                for tile in range(2)
            ],
        }

    def snapshot(records, steps):
        start = records["initial_parameters"]
        update = records["updates"][steps - 1] if steps else None
        ema = start["p"].clone()
        for completed in records["updates"][1:steps:2]:
            ema.mul_(0.9).add_(completed["parameters"]["p"], alpha=0.1)
        return deepcopy(
            {
                "parameters": update["parameters"] if update else start,
                "moments": {"p": update["moments"]["p"] if update else {}, "frozen": {}},
                "ema_names": ["p"],
                "runtime": {
                    "version": 1,
                    "progress": {
                        "optimizer_step": steps,
                        "rollout_iteration": steps // 2,
                        "data_epoch": 0,
                    },
                    "children": {
                        "adapter_ema": {
                            "version": 1,
                            "decay": 0.9,
                            "update_step_interval": 1,
                            "ema_parameters": [ema],
                            "num_updates": steps // 2,
                            "decay_schedule": "constant",
                            "schedule_params": {},
                        },
                        "adapter_reference": {"ema_parameters": [start["p"]], "num_updates": 0},
                    },
                },
                "cpu_rng": torch.tensor([1, 2, steps], dtype=torch.uint8),
                "cuda_rng": torch.tensor([3, 4, steps], dtype=torch.uint8),
            }
        )

    for records in (expected, actual):
        records["initial_parameters"]["frozen"] = torch.tensor([0.5])
        previous = records["initial_parameters"]
        for update in records["updates"]:
            update["parameters"]["frozen"] = previous["frozen"].clone()
            update["role"], update["before"] = "policy", deepcopy(previous)
            previous = update["parameters"]
        records["cycles"] = [cycle(0), cycle(1)]
    initial = {
        "loaded": snapshot(expected, 0),
        "saved": {1: snapshot(expected, 2)},
        "final": snapshot(expected, 4),
        "records": expected,
    }
    resumed = {
        "loaded": deepcopy(initial["saved"][1]),
        "final": snapshot(actual, 4),
        "records": {
            "initial_parameters": deepcopy(initial["saved"][1]["parameters"]),
            "updates": actual["updates"][2:],
            "cycles": actual["cycles"][1:],
        },
    }
    return initial, resumed


def test_ordered_continuation_accepts_real_adam_roundoff_and_pending_timing():
    initial, resumed = _ordered_continuation_fixture(roundoff=True)
    resumed["records"]["cycles"][0]["tiles"][0]["pending"] += 1
    with pytest.raises(AssertionError):
        _compare(resumed["final"], initial["final"], exact=True)
    residuals = _assert_ordered_continuation(initial, resumed, tiles=2)
    assert len(residuals) == 4
    assert residuals[2]["gradient_relative_l2"] > 0
    assert all(row["update_norm_enforced"] for row in residuals)


@pytest.mark.parametrize("field", ["parameters", "moments", "runtime", "cpu_rng", "cuda_rng"])
def test_ordered_continuation_rejects_loaded_state_corruption(field):
    initial, resumed = _ordered_continuation_fixture()
    loaded = resumed["loaded"]
    if field == "parameters":
        loaded[field]["p"][0] += 1e-5
    elif field == "moments":
        loaded[field]["p"]["exp_avg"][0] += 1e-5
    elif field == "runtime":
        loaded[field]["progress"]["optimizer_step"] += 1
    else:
        loaded[field][0] += 1
    with pytest.raises(AssertionError):
        _assert_ordered_continuation(initial, resumed, tiles=2)


@pytest.mark.parametrize(
    "corruption",
    [
        "cycle",
        "sample",
        "source",
        "group",
        "member",
        "score",
        "advantage",
        "expected_advantage",
        "order",
        "membership",
        "timestep",
        "term_gas",
        "gas",
        "step",
        "audit",
        "drained",
    ],
)
def test_ordered_continuation_rejects_work_identity_corruption(corruption):
    initial, resumed = _ordered_continuation_fixture()
    cycle = resumed["records"]["cycles"][0]
    tile = cycle["tiles"][0]
    if corruption in ("sample", "group", "member", "score"):
        cycle["rows"][0][corruption] += 1
    elif corruption == "source":
        cycle["rows"][0]["source"] = "b"
    elif corruption == "cycle":
        cycle["cycle"] += 1
    elif corruption == "advantage":
        tile["advantages"][0] *= -1
    elif corruption == "expected_advantage":
        cycle["expected_advantages"][tile["samples"][0]] *= -1
    elif corruption == "order":
        cycle["tiles"].reverse()
    elif corruption == "membership":
        tile["samples"].reverse()
    elif corruption in ("timestep", "term_gas"):
        key, time, gas = tile["terms"][0]
        tile["terms"][0] = (
            key,
            time + (corruption == "timestep"),
            gas + (corruption == "term_gas"),
        )
    elif corruption in ("gas", "step"):
        tile[corruption] += 1
    elif corruption == "audit":
        cycle["audit"].reverse()
    else:
        cycle["drained"] = False
    with pytest.raises(AssertionError):
        _assert_ordered_continuation(initial, resumed, tiles=2)


@pytest.mark.parametrize(
    "corruption",
    [
        "gradient",
        "clipped_gradient",
        "first_moment",
        "second_moment",
        "delta",
        "parameter",
        "missing_update",
        "role",
        "loss",
    ],
)
def test_ordered_continuation_rejects_update_corruption(corruption):
    initial, resumed = _ordered_continuation_fixture()
    update = resumed["records"]["updates"][0]
    if corruption in ("gradient", "clipped_gradient", "delta", "parameter"):
        field = {
            "gradient": "gradients",
            "clipped_gradient": "clipped_gradients",
            "delta": "deltas",
            "parameter": "parameters",
        }[corruption]
        update[field]["p"][0] += 0.01
    elif corruption in ("first_moment", "second_moment"):
        field = "exp_avg" if corruption == "first_moment" else "exp_avg_sq"
        update["moments"]["p"][field][0] += 1e-5
    elif corruption == "missing_update":
        resumed["records"]["updates"].pop()
    elif corruption == "role":
        update["role"] = "wrong"
    else:
        update["loss"] *= 2
    with pytest.raises(AssertionError):
        _assert_ordered_continuation(initial, resumed, tiles=2)


def test_ordered_continuation_rejects_valid_adam_outside_existing_gradient_bounds():
    initial, resumed = _ordered_continuation_fixture(gradient_scale=1.1)
    with pytest.raises(AssertionError):
        _assert_ordered_continuation(initial, resumed, tiles=2)


@pytest.mark.parametrize("leg", ["initial", "resumed"])
@pytest.mark.parametrize("field", ["parameter", "first_moment", "second_moment", "frozen_moment"])
def test_final_optimizer_state_rejects_snapshot_corruption_for_either_schedule(leg, field):
    initial, resumed = _ordered_continuation_fixture()
    worker = initial if leg == "initial" else resumed
    _assert_final_optimizer_state(worker)
    final = worker["final"]
    if field == "parameter":
        final["parameters"]["p"][0] += 1e-5
    elif field == "frozen_moment":
        final["moments"]["frozen"] = {"step": torch.tensor(1.0)}
    else:
        key = "exp_avg" if field == "first_moment" else "exp_avg_sq"
        final["moments"]["p"][key][0] += 1e-5
    with pytest.raises(AssertionError):
        _assert_final_optimizer_state(worker)
    with pytest.raises(AssertionError):
        _assert_ordered_continuation(initial, resumed, tiles=2)


@pytest.mark.parametrize(
    "field",
    [
        "cpu_rng",
        "cuda_rng",
        "optimizer_step",
        "rollout_iteration",
        "data_epoch",
        "reference",
        "ema_value",
        "ema_metadata",
        "ema_names",
    ],
)
def test_ordered_continuation_rejects_final_runtime_corruption(field):
    initial, resumed = _ordered_continuation_fixture()
    final = resumed["final"]
    if field in ("cpu_rng", "cuda_rng"):
        final[field][0] += 1
    elif field in ("optimizer_step", "rollout_iteration", "data_epoch"):
        final["runtime"]["progress"][field] += 1
    elif field == "reference":
        final["runtime"]["children"]["adapter_reference"]["ema_parameters"][0][0] += 1e-5
    elif field == "ema_value":
        final["runtime"]["children"]["adapter_ema"]["ema_parameters"][0][0] += 1e-5
    elif field == "ema_metadata":
        final["runtime"]["children"]["adapter_ema"]["update_step_interval"] += 1
    else:
        final["ema_names"] = ["frozen"]
    with pytest.raises(AssertionError):
        _assert_ordered_continuation(initial, resumed, tiles=2)
