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

"""Native overlap lifecycle shared by explicitly selected fixture modules.

The fixture owns model construction, media scoring and the independent objective
oracle. This runner owns delayed futures, membership, optimizer observations and
drained continuation. Ready runs are checked independently, not against another
completion schedule's parameter trajectory.
"""

import copy
import importlib
import json
import math
import os
import threading
import time
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from online_optimizer_observations import observe_optimizer_steps
from test_context_parallel_state import _assert_ema_trajectory, _cpu_tree, _state
from test_context_parallel_training import _check_adam_trajectory, _compare, _compare_bf16_updates
from test_model_preparation import initialize_cuda, runtime_observation
from wan_eval_reward import PromptReward

from flow_factory.hparams import Arguments
from flow_factory.rewards import PointwiseRewardModel, RewardModelOutput


def make_overlap_config(base, root, result_dir, options):
    """Resolve geometry once, after the leaf supplies model/algorithm defaults."""
    tiles = options.get("tiles", 2)
    sampler_type = options.get("sampler_type", "rank_local")
    if sampler_type not in {"rank_local", "global_batch", "subgroup_tile"}:
        raise ValueError(f"Unsupported native sampler_type: {sampler_type!r}")
    subgroup = sampler_type == "subgroup_tile"
    group_size = 4 if subgroup else 2
    init_fields = {field.name for field in fields(base.training_args) if field.init}
    train = {
        key: value for key, value in base.training_args.to_dict().items() if key in init_fields
    }
    train["extra_kwargs"] = copy.deepcopy(base.training_args.extra_kwargs)
    train.update(
        reward_optimization_overlap=True,
        reward_optimization_overlap_mode=options["scheduling_mode"],
        reward_optimization_overlap_poll_interval=0.002,
        max_epochs=options.get("epochs", 2),
        unique_sample_num_per_epoch=2 * tiles * options["dp_size"] // group_size,
        group_size=group_size,
        gradient_step_per_epoch=tiles,
        gradient_accumulation_steps="auto",
        num_inner_epochs=1,
        shuffle_samples=False,
        global_std=False,
        advantage_aggregation="sum",
    )
    data = base.data_args.to_dict()
    sources = ("a", "b") if subgroup else ("tiny",)
    data.update(
        datasets=[
            {"name": name, "dataset_dir": str(root / name), "train": {"weight": 1}}
            for name in sources
        ]
        + [{"name": "odd", "dataset_dir": str(root / "odd"), "eval": {}}],
        sampler_type=sampler_type,
        sampler_subgroup_size=2 if subgroup else None,
        cache_dir=str(root / "overlap-cache"),
    )
    evaluation = base.eval_args.to_dict()
    evaluation.update(
        eval_freq=1,
        resolution=list(train["resolution"]),
        height=train["resolution"][0],
        width=train["resolution"][1],
        num_frames=train["num_frames"],
        frame_rate=train["frame_rate"],
        num_inference_steps=2,
        guidance_scale=train["guidance_scale"],
        per_device_batch_size=1,
        seed=0,
    )
    model = base.model_args.to_dict()
    if options.get("resume_from"):
        model.update(resume_type="state", resume_path=options["resume_from"])
    return Arguments.from_dict(
        {
            "mixed_precision": base.mixed_precision,
            "parallelism": base.parallelism_args.to_dict(),
            "model": model,
            "data": data,
            "train": train,
            "scheduler": base.scheduler_args.to_dict(),
            "acceleration": base.acceleration_args.to_dict(),
            "optimizers": [optimizer.to_dict() for optimizer in base.optimizer_args],
            "log": {
                "save_model_only": False,
                "save_freq": 1,
                "save_dir": str(Path(result_dir) / "saves"),
                "run_name": "overlap",
                "logging_backend": "none",
                "verbose": False,
            },
            "eval": evaluation,
            "rewards": [
                {
                    "name": "score",
                    "reward_model": "context_parallel_overlap_runner.GatedReward",
                    "device": "cpu",
                    "batch_size": 1,
                    "async_reward": True,
                    "num_workers": 1 if options.get("fail_after_first_tile") else tiles * 2,
                    "weight": {name: -2.0 if name == "b" else 1.0 for name in sources},
                }
            ],
            "eval_rewards": [
                {
                    "name": "score",
                    "reward_model": "wan_eval_reward.PromptReward",
                    "device": "cpu",
                    "batch_size": 1,
                    "async_reward": True,
                }
            ],
        }
    )


class RewardGates:
    """Bounded local thread control; reward workers never call collectives."""

    def __init__(self, options, rank, fixture):
        self.options, self.rank, self.fixture = options, rank, fixture
        self.scored = []
        self.release_all = threading.Event()
        self.reset(-1)

    def reset(self, cycle):
        self.cycle = cycle
        self.plan_ready = threading.Event()
        self.gates = [threading.Event() for _ in range(self.options.get("tiles", 2))]
        self.started = [threading.Event() for _ in self.gates]
        self.row_to_tile = {}

    def release(self):
        self.release_all.set()
        self.plan_ready.set()
        for gate in self.gates:
            gate.set()

    def score(self, source, sample_id, member, media):
        assert self.plan_ready.wait(180), "native tile plan was not published"
        if self.release_all.is_set():
            raise RuntimeError("native reward released during cleanup")
        tile = self.row_to_tile[(source, sample_id)]
        self.started[tile].set()
        assert self.gates[tile].wait(180), "native reward tile was not released"
        if self.options.get("fail_after_first_tile") and self.rank == 0 and tile == 0:
            raise ValueError("native late reward failure after first optimizer tile")
        value = float(member**2 + self.fixture.score_media(media))
        self.scored.append((self.cycle, source, sample_id, value))
        return value


class GatedReward(PointwiseRewardModel):
    """The worker binds one fixture's explicit media field before model loading."""

    use_tensor_inputs = True
    required_fields = ("source", "sampling_sample_id", "sampling_group_member_id")
    controller = None
    media_field = None

    def __call__(self, sampling_sample_id, sampling_group_member_id, source=None, **kwargs):
        # Single-source training intentionally has no source bookkeeping. The
        # processor omits None fields when preparing the actual model call.
        assert len(sampling_sample_id) == 1
        media = kwargs[self.media_field][0]
        value = self.controller.score(
            source[0] if source is not None else None,
            int(sampling_sample_id[0]),
            int(sampling_group_member_id[0]),
            media,
        )
        return RewardModelOutput(rewards=torch.tensor([value], dtype=torch.float64))


def _identity(sample):
    return (sample.source, int(sample.sampling_sample_id))


def _row(sample, fixture):
    media = getattr(sample, fixture.reward_media_field)
    assert isinstance(media, torch.Tensor) and torch.isfinite(media).all()
    return {
        "source": sample.source,
        "sample": int(sample.sampling_sample_id),
        "group": int(sample.sampling_group_id),
        "member": int(sample.sampling_group_member_id),
        "score": float(sample.sampling_group_member_id**2 + fixture.reference_media_score(media)),
    }


def _audit_fresh_replay(trainer, samples):
    """Audit every stored transition before any tile changes the sampled policy."""
    cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    rows = []
    trainer.adapter.train()
    try:
        for batch in trainer._iter_prefetched_batches(samples, 1):
            for index in trainer.adapter.get_train_step_indices():
                replay = trainer.adapter.get_replay_step(batch, int(index))
                with torch.no_grad(), trainer.autocast():
                    output = trainer._replay_forward(
                        batch, replay, ("log_prob", "velocity", "next_latents_mean")
                    )
                torch.testing.assert_close(output.log_prob, replay.log_prob, rtol=0, atol=0)
                for field, actual in (
                    ("velocity", output.velocity),
                    ("next_latents_mean", output.next_state_mean),
                ):
                    expected = trainer.adapter.get_replay_callback(batch, int(index), field)
                    assert actual.component_names == expected.component_names
                    for name in expected.component_names:
                        torch.testing.assert_close(
                            actual.components[name], expected.components[name], rtol=0, atol=0
                        )
                rows.append((_identity(batch.samples[0]), int(index)))
    finally:
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
    return rows


@contextmanager
def observe_coupled_terms(trainer, terms):
    """Observe actual retained-transition terms without owning their objective."""
    original = trainer._replay_forward

    def replay(batch, step, return_fields):
        if "advantage" in batch:
            terms.append(
                (
                    _identity(batch.samples[0]),
                    float(step.times.timestep["latent"].reshape(-1)[0]),
                    trainer.accelerator.gradient_accumulation_steps,
                )
            )
        return original(batch, step, return_fields)

    trainer._replay_forward = replay
    try:
        yield
    finally:
        trainer._replay_forward = original


def _expected_advantages(
    gathered,
    *,
    model_degree,
    dp_size,
    group_size,
    subgroup_size,
    source_rows_per_replica=None,
    sampler_type="rank_local",
):
    """Compute statistics from original generated rows, independent of packed reductions."""
    groups = defaultdict(list)
    for data_rank in range(dp_size):
        rows = gathered[data_rank * model_degree]
        if source_rows_per_replica is not None:
            assert Counter(row["source"] for row in rows) == source_rows_per_replica
        for peer in range(1, model_degree):
            _compare(gathered[data_rank * model_degree + peer], rows, exact=True)
        for row in rows:
            groups[(row["source"], row["group"])].append((data_rank, row))
    expected = {}
    for members in groups.values():
        assert len(members) == group_size
        assert sorted(row["member"] for _, row in members) == list(range(group_size))
        ranks = {rank for rank, _ in members}
        if subgroup_size:
            assert len(ranks) == subgroup_size
            assert len({rank // subgroup_size for rank in ranks}) == 1
        elif sampler_type == "global_batch":
            # This native recipe covers complete evenly spread groups, not arbitrary packing.
            assert group_size % dp_size == 0
            assert Counter(rank for rank, _ in members) == {
                rank: group_size // dp_size for rank in range(dp_size)
            }
        else:
            assert sampler_type == "rank_local" and len(ranks) == 1
        values = [(-2.0 if row["source"] == "b" else 1.0) * row["score"] for _, row in members]
        mean = math.fsum(values) / group_size
        std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / group_size)
        assert std > 1e-6
        for (_, row), value in zip(members, values, strict=True):
            key = (row["source"], row["sample"])
            assert key not in expected
            expected[key] = (value - mean) / std
    return expected


def assert_tile_training_terms(terms, *, identities, gas, losses):
    """Count two observed training times per independently expected optimizer example."""
    assert len(set(identities)) == len(identities)
    expected = 2 * len(identities)
    assert expected > 0 and gas == expected
    assert losses.numel() == expected and len(terms) == expected
    assert Counter(term[0] for term in terms) == Counter({key: 2 for key in identities})
    assert all(term[2] == gas for term in terms)
    assert all(len({term[1] for term in terms if term[0] == key}) == 2 for key in identities)


def _worker(rank, store_path, result_dir, options):
    fixture = importlib.import_module(options["case_module"])
    os.environ.update(
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP=",".join(fixture.wrap_classes),
        FSDP_CPU_RAM_EFFICIENT_LOADING="false",
        FSDP_RESHARD_AFTER_FORWARD="true",
        FSDP_ACTIVATION_CHECKPOINTING="true",
    )
    model_degree = options["cp_size"] * options["tp_size"]
    world = options["dp_size"] * model_degree
    initialize_cuda(rank, store_path, world_size=world)
    result = {"rank": rank, **options}
    gates = RewardGates(options, rank, fixture)
    GatedReward.controller, GatedReward.media_field = gates, fixture.reward_media_field
    GatedReward.required_fields = (
        "source",
        "sampling_sample_id",
        "sampling_group_member_id",
        fixture.reward_media_field,
    )
    trainer = None
    records = {}
    try:
        from flow_factory.trainers.loader import load_trainer

        root = Path(options["fixture"])
        config = make_overlap_config(
            fixture.make_config(root, result_dir, options), root, result_dir, options
        )
        trainer = load_trainer(config)
        result["runtime"] = runtime_observation(trainer.accelerator)
        from torch.distributed.fsdp import FSDPModule

        wrapped = {
            cls.__name__
            for module in trainer.model_bundle.modules()
            if isinstance(module, FSDPModule)
            for cls in type(module).__mro__
            if cls.__name__ in fixture.wrap_classes
        }
        assert wrapped == set(fixture.wrap_classes)
        result["wrapped_blocks"] = sorted(wrapped)
        loaded = _state(trainer)
        records = {
            "initial_parameters": loaded["parameters"],
            "updates": [],
            "backward_losses": [],
            "cycles": [],
            "evaluations": [],
            "cleanup": [],
            "aborts": [],
            "boundaries": [],
        }
        saved, saving, evaluating = {}, False, False
        sample, sample_batch = trainer.sample, trainer.sample_batch
        prepare = trainer._prepare_reward_optimization_overlap
        optimize = trainer._optimize_reward_overlap_tile
        finalize = trainer._finalize_reward_optimization_overlap
        abort = trainer._abort_reward_optimization_overlap
        cleanup = trainer.cleanup
        current = None
        optimization_terms = []

        def observed_samples(*args, **kwargs):
            if evaluating:
                _compare(
                    _cpu_tree(trainer.adapter._ema_tracked_parameters()),
                    _cpu_tree(trainer.adapter.ema_wrapper.ema_parameters),
                    exact=True,
                )
            else:
                kwargs["extra_call_back_kwargs"] = list(
                    dict.fromkeys(
                        [*kwargs.get("extra_call_back_kwargs", []), "velocity", "next_latents_mean"]
                    )
                )
            return sample_batch(*args, **kwargs)

        def observed_sample():
            gates.reset(trainer.cycle_index)
            return sample()

        def observed_prepare(samples, plan):
            nonlocal current
            rows = [_row(sample, fixture) for sample in samples]
            gathered = [None] * world
            dist.all_gather_object(gathered, rows)
            expected = _expected_advantages(
                gathered,
                model_degree=model_degree,
                dp_size=options["dp_size"],
                group_size=config.training_args.group_size,
                subgroup_size=2 if options.get("sampler_type") == "subgroup_tile" else None,
                sampler_type=options.get("sampler_type", "rank_local"),
                source_rows_per_replica=(
                    {"a": options["tiles"], "b": options["tiles"]}
                    if options.get("sampler_type") == "subgroup_tile"
                    else None
                ),
            )
            assert len(plan.tiles) == options.get("tiles", 2)
            assert (
                plan.samples_per_tile == 2
                and config.training_args.gradient_accumulation_steps
                == 2 * getattr(fixture, "training_examples_per_tile", 2)
            )
            current = {
                "cycle": trainer.cycle_index,
                "rows": rows,
                "audit": fixture.audit_samples(trainer, samples),
                "tiles": [],
                "expected_advantages": expected,
                "drained": False,
            }
            records["cycles"].append(current)
            gates.row_to_tile = {
                _identity(samples[index]): tile.tile_id
                for tile in plan.tiles
                for index in tile.sample_indices
            }
            gates.plan_ready.set()
            first = 1 if options["scheduling_mode"] == "ready" else 0
            gates.gates[first].set()
            return prepare(samples, plan)

        def observed_optimize(tile, samples, context):
            pending = sum(not future.done() for _, _, future in trainer.reward_buffer._futures)
            before = len(records["updates"])
            terms_start = len(optimization_terms)
            identities = [_identity(sample) for sample in samples]
            advantages = [float(sample["advantage"]) for sample in samples]
            for key, actual in zip(identities, advantages, strict=True):
                assert actual == pytest.approx(
                    current["expected_advantages"][key], rel=2e-6, abs=2e-6
                )
            optimize(tile, samples, context)
            assert len(records["updates"]) == before + 1
            terms = optimization_terms[terms_start:]
            expected_ids = (
                fixture.training_term_identities(samples)
                if hasattr(fixture, "training_term_identities")
                else identities
            )
            assert len(expected_ids) == getattr(fixture, "training_examples_per_tile", 2)
            assert_tile_training_terms(
                terms,
                identities=expected_ids,
                gas=trainer.accelerator.gradient_accumulation_steps,
                losses=records["updates"][-1]["loss"],
            )
            current["tiles"].append(
                {
                    "tile": tile.tile_id,
                    "samples": identities,
                    "advantages": advantages,
                    "pending": pending,
                    "step": trainer.step,
                    "gas": trainer.accelerator.gradient_accumulation_steps,
                    "terms": terms,
                }
            )
            if len(current["tiles"]) == 1:
                assert tile.tile_id == (1 if options["scheduling_mode"] == "ready" else 0)
                if options.get("fail_after_first_tile"):
                    if rank == 0:
                        gates.gates[0].set()
                        failed = next(
                            f for _, indices, f in trainer.reward_buffer._futures if indices == [0]
                        )
                        assert isinstance(failed.exception(timeout=30), ValueError)
                        assert gates.started[2].wait(30)
                else:
                    for gate in gates.gates:
                        gate.set()

        def observed_finalize(context, samples):
            assert not trainer.reward_buffer._streaming_sealed
            assert trainer.reward_buffer._streaming_consumed_indices == set(range(len(samples)))
            assert all(f.done() for _, _, f in trainer.reward_buffer._futures)
            assert Counter(key for tile in current["tiles"] for key in tile["samples"]) == Counter(
                _identity(s) for s in samples
            )
            current["drained"] = True
            return finalize(context, samples)

        def observed_abort(context):
            records["aborts"].append(trainer.cycle_index)
            return abort(context)

        def observed_cleanup(*, wait=False):
            records["cleanup"].append(
                {
                    "cancelled": sum(f.cancelled() for _, _, f in trainer.reward_buffer._futures),
                    "incomplete": trainer._acquisition_cycle_incomplete,
                    "active": trainer._acquisition_cycle_active,
                }
            )
            gates.release()
            started = time.monotonic()
            returned = cleanup(wait=wait)
            assert time.monotonic() - started < 30
            assert all(
                not thread.is_alive()
                for executor in trainer.reward_buffer._executors.values()
                for thread in executor._threads
            )
            return returned

        trainer.sample, trainer.sample_batch = observed_sample, observed_samples
        trainer._prepare_reward_optimization_overlap = observed_prepare
        trainer._optimize_reward_overlap_tile = observed_optimize
        trainer._finalize_reward_optimization_overlap = observed_finalize
        trainer._abort_reward_optimization_overlap = observed_abort
        trainer.cleanup = observed_cleanup

        evaluate, dataset = trainer.evaluate, trainer._evaluate_dataset

        def observed_dataset(source, loader):
            row = dataset(source, loader)
            expected = [f"a red video {value}" for value in (1, 4, 9)]
            assert [sample.prompt for sample in row["eval/odd/samples"]] == expected[
                rank // model_degree :: options["dp_size"]
            ]
            prefix = "eval/odd/reward_score_"
            assert row[prefix + "count"] == 3
            for field, value in (
                ("mean", 14 / 3),
                ("std", math.sqrt(98 / 9)),
                ("min", 1),
                ("max", 9),
            ):
                assert row[prefix + field] == pytest.approx(value, rel=1e-12)
            return row

        def observed_evaluate():
            nonlocal evaluating
            assert not trainer._acquisition_cycle_active
            assert not trainer.reward_buffer._streaming_sealed
            before = _cpu_tree(dict(trainer.model_bundle.named_parameters()))
            evaluating = True
            try:
                evaluate()
            finally:
                evaluating = False
            _compare(_cpu_tree(dict(trainer.model_bundle.named_parameters())), before, exact=True)
            assert all(
                not buffer.all_samples for buffer in trainer.eval_dataset_reward_buffers.values()
            )
            records["evaluations"].append((trainer.cycle_index, trainer.step))

        save = trainer._save_exact_training_state

        def observed_save(path):
            nonlocal saving
            assert (
                not trainer._acquisition_cycle_active and not trainer._acquisition_cycle_incomplete
            )
            assert not trainer.reward_buffer._streaming_sealed
            saved[trainer.cycle_index] = _state(trainer)
            records["boundaries"].append((trainer.cycle_index, trainer.step))
            saving = True
            try:
                return save(path)
            finally:
                saving = False

        trainer.evaluate, trainer._evaluate_dataset = observed_evaluate, observed_dataset
        trainer._save_exact_training_state = observed_save
        failure = None
        with observe_optimizer_steps(
            trainer,
            records,
            cp_size=options["cp_size"],
            is_saving=lambda: saving,
            inactive_parameter_names=(
                fixture.inactive_parameter_names(trainer.model_bundle)
                if hasattr(fixture, "inactive_parameter_names")
                else frozenset()
            ),
        ):
            with (
                fixture.observe_training_terms(trainer, optimization_terms),
                fixture.observe_objectives(trainer) as objectives,
            ):
                records["objectives"] = objectives
                try:
                    trainer.start()
                except RuntimeError as error:
                    if not options.get("fail_after_first_tile"):
                        raise
                    failure = str(error)
                    assert "native late reward failure after first optimizer tile" in failure
        records["objectives"] = objectives
        fixture.assert_objectives(objectives, options["cp_size"])
        if options.get("fail_after_first_tile"):
            assert failure is not None and trainer.step == 1
            assert records["aborts"] == [0] and len(records["cycles"]) == 1
            assert records["cleanup"][0]["incomplete"] and not records["cleanup"][0]["active"]
            assert all(cycle == 0 for cycle in saved)
            assert _state(trainer)["runtime"]["children"]["adapter_ema"]["num_updates"] == 0
        else:
            assert failure is None and not records["aborts"]
            assert trainer.step == options.get("epochs", 2) * options.get("tiles", 2)
            assert all(cycle["drained"] for cycle in records["cycles"])
            assert not records["backward_losses"]
        final = _state(trainer)
        nonzero = any(
            torch.count_nonzero(g)
            for update in records["updates"]
            for g in update["gradients"].values()
        )
        assert nonzero
        torch.save(
            {
                "loaded": loaded,
                "saved": saved,
                "final": final,
                "records": records,
                "scored": gates.scored,
                "eval_scored": PromptReward.scored,
            },
            Path(result_dir, f"observations-{rank}.pt"),
        )
        result.update(
            status="pass",
            optimizer_steps=trainer.step,
            failure=failure,
            strict_replay=fixture.audit_kind == "coupled_replay",
            input_audit=fixture.audit_kind,
            reward_overlap=True,
            numerical_comparison=True,
            exact_resume=bool(options.get("resume_from")),
            elementwise_comparison=True,
            nonzero_gradients=True,
            varying_rewards=True,
            execution_capability=f"{options.get('model_family', 'wan')}-{options['algorithm']}-cp-v1",
        )
    except Exception:
        result["error"] = traceback.format_exc()
        try:
            torch.save(records, Path(result_dir, f"failed-observations-{rank}.pt"))
        except Exception:
            result["observation_error"] = traceback.format_exc()
    finally:
        # No final barrier: another rank may have failed before a collective.
        _publish_rank_result(Path(result_dir), rank, result)
        gates.release()
        if trainer is not None:
            trainer.cleanup(wait=True)
        dist.destroy_process_group()


def _publish_rank_result(directory, rank, result):
    """Publish a complete result while the parent may be polling for failures."""
    path = directory / f"rank-{rank}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(path)


def _run_native(directory, options, *, timeout=900):
    """Watch rank failures before the deadline and reap only this launch's ranks."""
    directory.mkdir(parents=True)
    world = options["dp_size"] * options["cp_size"] * options["tp_size"]
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_worker, args=(rank, str(directory / "store"), str(directory), options)
        )
        for rank in range(world)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + timeout
        while any(process.is_alive() for process in processes):
            for path in directory.glob("rank-*.json"):
                row = json.loads(path.read_text())
                assert "error" not in row, f"{path}: {row.get('error')}"
            failed = [process for process in processes if process.exitcode not in (None, 0)]
            assert (
                not failed
            ), f"native rank exited without successful completion: {[p.exitcode for p in failed]}"
            assert (
                time.monotonic() < deadline
            ), f"native overlap exceeded {timeout}s; inspect {directory}"
            time.sleep(0.2)
        for process in processes:
            process.join()
        assert all(
            process.exitcode == 0 for process in processes
        ), f"native ranks failed during terminal cleanup: {[p.exitcode for p in processes]}"
        rows = [json.loads((directory / f"rank-{rank}.json").read_text()) for rank in range(world)]
        assert all(row.get("status") == "pass" and "error" not in row for row in rows), rows
        return [
            torch.load(directory / f"observations-{rank}.pt", weights_only=True)
            for rank in range(world)
        ]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)


def _assert_run(group, options, *, failure=False):
    model_degree = options["cp_size"] * options["tp_size"]
    fixture = importlib.import_module(options["case_module"])
    cycles = len(group[0]["records"]["cycles"])
    for cycle_index in range(cycles):
        expected_order = [
            tile["tile"] for tile in group[0]["records"]["cycles"][cycle_index]["tiles"]
        ]
        assert (
            sum(worker["records"]["cycles"][cycle_index]["tiles"][0]["pending"] for worker in group)
            > 0
        )
        for worker in group:
            cycle = worker["records"]["cycles"][cycle_index]
            assert [tile["tile"] for tile in cycle["tiles"]] == expected_order
            assert len(cycle["audit"]) == fixture.audit_steps * len(cycle["rows"])
            assert Counter(key for key, _ in cycle["audit"]) == Counter(
                {(row["source"], row["sample"]): fixture.audit_steps for row in cycle["rows"]}
            )
            if not failure:
                assert sorted(expected_order) == list(range(options.get("tiles", 2)))
    for rank, worker in enumerate(group):
        records = worker["records"]
        assert not records["backward_losses"]
        if not failure:
            expected = {}
            for cycle in records["cycles"]:
                for index, row in enumerate(cycle["rows"]):
                    if index % model_degree == rank % model_degree:
                        expected[(cycle["cycle"], row["source"], row["sample"])] = row["score"]
            assert len(worker["scored"]) == len(expected)
            for cycle, source, sample, score in worker["scored"]:
                assert score == pytest.approx(
                    expected.pop((cycle, source, sample)), rel=1e-12, abs=1e-12
                )
            assert not expected
        assert all(not event["active"] for event in records["cleanup"])
    if failure:
        assert sum(worker["records"]["cleanup"][0]["cancelled"] for worker in group) > 0
    else:
        scored = Counter(tuple(row) for worker in group for row in worker["eval_scored"])
        assert scored == Counter({("odd", f"a red video {value}"): cycles for value in (1, 4, 9)})


def _assert_final_optimizer_state(worker):
    """Bind the final snapshot to this run's observed optimizer trajectory."""
    _compare(worker["records"]["initial_parameters"], worker["loaded"]["parameters"], exact=True)
    final, last = worker["final"], worker["records"]["updates"][-1]
    _compare(final["parameters"], last["parameters"], exact=True)
    _compare(final["moments"], {**worker["loaded"]["moments"], **last["moments"]}, exact=True)


def _assert_ordered_continuation(initial, continuation, *, tiles, check_objectives):
    """Check exact restore/work and the first update from identical parameters.

    Later BF16 trajectories may enter different clipped-objective branches.
    Each retains its own objective, clipping, Adam and EMA proofs; cross-run
    statistics after the first resumed update are diagnostic. Independent model
    parameter-gradient equivalence remains the native model probe's owner.
    """
    _compare(continuation["loaded"], initial["saved"][1], exact=True)
    expected, resumed = initial["records"], continuation["records"]
    assert len(expected["cycles"]) == 2 and len(resumed["cycles"]) == 1
    assert len(expected["updates"]) == 2 * tiles and len(resumed["updates"]) == tiles

    def canonical_cycles(cycles):
        return [
            {
                **cycle,
                "tiles": [
                    {key: value for key, value in tile.items() if key != "pending"}
                    for tile in cycle["tiles"]
                ],
            }
            for cycle in cycles
        ]

    # Pending-future counts observe thread timing, not optimizer membership.
    # Every other row, score, advantage, transition, GAS and ordering field stays exact.
    _compare(
        canonical_cycles(resumed["cycles"]), canonical_cycles(expected["cycles"][1:]), exact=True
    )
    joined = {
        "initial_parameters": expected["initial_parameters"],
        "updates": expected["updates"][:tiles] + resumed["updates"],
    }
    _compare(
        [update["role"] for update in joined["updates"]],
        [update["role"] for update in expected["updates"]],
        exact=True,
    )
    assert resumed["objectives"] and len(expected["objectives"]) == 2 * len(resumed["objectives"])
    for left, right in zip(joined["updates"], expected["updates"], strict=True):
        assert left["optimizer"] == right["optimizer"]
    _compare(resumed["updates"][0]["before"], continuation["loaded"]["parameters"], exact=True)
    _compare(resumed["updates"][0]["before"], expected["updates"][tiles]["before"], exact=True)
    for worker, records in ((initial, expected), (continuation, joined)):
        check_objectives(worker["records"]["objectives"])
        previous = records["initial_parameters"]
        for update in records["updates"]:
            _compare(update["before"], previous, exact=True)
            inactive = set(previous) - set(update["gradients"])
            _compare(
                {name: update["parameters"][name] for name in inactive},
                {name: previous[name] for name in inactive},
                exact=True,
            )
            previous = update["parameters"]
        _check_adam_trajectory(records)
    first = _compare_bf16_updates(
        {**joined, "updates": joined["updates"][: tiles + 1]},
        {**expected, "updates": expected["updates"][: tiles + 1]},
    )[-1]

    def diagnostic(actual, reference):
        assert set(actual) == set(reference) and reference
        squared_error = squared_signal = 0.0
        outliers = 0
        for name, value in reference.items():
            delta = actual[name].double() - value.double()
            assert torch.isfinite(delta).all() and torch.isfinite(value).all()
            squared_error += float(delta.square().sum())
            squared_signal += float(value.double().square().sum())
            outliers += int((delta.abs() > 1e-3 + 0.02 * value.double().abs()).sum())
        return {
            "relative_l2": math.sqrt(squared_error / max(squared_signal, 1e-24)),
            "element_outliers_at_original_gradient_budget": outliers,
        }

    later = [
        {
            "global_update": index + 1,
            "comparison_scope": "autonomous_diagnostic_only",
            "gradients": diagnostic(left["gradients"], right["gradients"]),
            "parameters": diagnostic(left["parameters"], right["parameters"]),
            "deltas": diagnostic(left["deltas"], right["deltas"]),
        }
        for index, (left, right) in enumerate(
            zip(joined["updates"], expected["updates"], strict=True)
        )
        if index > tiles
    ]
    identities = []
    for worker, records in ((initial, expected), (continuation, joined)):
        _assert_final_optimizer_state(worker)
        final, last = worker["final"], records["updates"][-1]
        _assert_ema_trajectory(initial["loaded"], final, [records["updates"][tiles - 1], last])
        identity = copy.deepcopy(
            {key: value for key, value in final.items() if key not in ("parameters", "moments")}
        )
        # EMA values follow each independently checked trajectory. Its metadata,
        # all other runtime children, progress and both device RNG states stay exact.
        del identity["runtime"]["children"]["adapter_ema"]["ema_parameters"]
        identities.append(identity)
    _compare(*identities, exact=True)
    return {
        "contract": "exact_restore_first_update_own_continuation_v1",
        "first_resumed_update": tiles + 1,
        "same_state_update": first,
        "own_objectives_checked": True,
        "objective_rows": {
            "continuous": len(expected["objectives"]),
            "resumed": len(resumed["objectives"]),
        },
        "own_clipping_adam_checked": True,
        "own_ema_final_state_checked": True,
        "exact_work_runtime_rng": True,
        "optimizer_settings_equal_all_steps": True,
        "model_gradient_owner": "independent_native_model_probe",
        "later_diagnostics": later,
    }


def write_overlap_datasets(root: Path, fixture) -> None:
    """Let model recipes own input conditions while retaining common prompt identities."""
    factory = getattr(fixture, "dataset_record", None)

    def record(prompt: str, index: int) -> dict:
        return {"prompt": prompt} if factory is None else factory(root, prompt, index)

    for name in ("tiny", "a", "b"):
        directory = root / name
        directory.mkdir(exist_ok=True)
        (directory / "train.jsonl").write_text(
            "".join(
                json.dumps(record(f"a {'red' if i % 2 else 'blue'} video", i)) + "\n"
                for i in range(16)
            )
        )
    (root / "odd").mkdir()
    (root / "odd/test.jsonl").write_text(
        "".join(json.dumps(record(f"a red video {value}", value)) + "\n" for value in (1, 4, 9))
    )


def qualify_overlap(tmp_path, options):
    """Execute one leaf's real updates, evaluation, exact load and continuation."""
    world = options["dp_size"] * options["cp_size"] * options["tp_size"]
    available = int(os.environ.get("FF_TEST_CUDA_RANKS", torch.cuda.device_count()))
    if available < world:
        pytest.skip(f"native overlap requires {world} CUDA ranks")
    fixture = importlib.import_module(options["case_module"])
    root = tmp_path.resolve() / "fixture"
    fixture.build_fixture(root, options["tp_size"])
    write_overlap_datasets(root, fixture)
    options = {"epochs": 2, "tiles": 2, **options, "fixture": str(root)}
    if hasattr(fixture, "qualify_model_control"):
        fixture.qualify_model_control(tmp_path / "model-control", root, options)
    continuous = _run_native(tmp_path / "continuous", options)
    _assert_run(continuous, options)
    for worker in continuous:
        _assert_final_optimizer_state(worker)
        if options["scheduling_mode"] != "ordered":
            _check_adam_trajectory(worker["records"])
        cycle_updates = [
            worker["records"]["updates"][(index + 1) * options["tiles"] - 1] for index in range(2)
        ]
        _assert_ema_trajectory(worker["loaded"], worker["final"], cycle_updates)
    resumed_options = {
        **options,
        "resume_from": str(tmp_path / "continuous/saves/overlap/checkpoints/checkpoint-1"),
    }
    resumed = _run_native(tmp_path / "resumed", resumed_options)
    _assert_run(resumed, resumed_options)
    continuation_residuals = []
    for rank, (initial, continuation) in enumerate(zip(continuous, resumed, strict=True)):
        _assert_final_optimizer_state(continuation)
        _compare(continuation["loaded"], initial["saved"][1], exact=True)
        assert len(continuation["records"]["cycles"]) == 1
        joined = copy.deepcopy(initial["records"])
        joined["updates"] = (
            joined["updates"][: options["tiles"]] + continuation["records"]["updates"]
        )
        # The prefix is the state actually loaded. Future ready ordering may
        # differ; every new moment/update still follows this exact prefix.
        if options["scheduling_mode"] != "ordered":
            _check_adam_trajectory(joined)
        _assert_ema_trajectory(
            initial["loaded"],
            continuation["final"],
            [joined["updates"][options["tiles"] - 1], joined["updates"][-1]],
        )
        if options["scheduling_mode"] == "ordered":
            continuation_residuals.append(
                {
                    "rank": rank,
                    "updates": _assert_ordered_continuation(
                        initial,
                        continuation,
                        tiles=options["tiles"],
                        check_objectives=lambda rows: fixture.assert_objectives(
                            rows, options["cp_size"]
                        ),
                    ),
                }
            )
        assert continuation["final"]["runtime"]["children"]["adapter_ema"]["num_updates"] == 2
    if options.get("failure_leg"):
        failed_options = {**options, "tiles": 4, "fail_after_first_tile": True}
        failed = _run_native(tmp_path / "failed", failed_options)
        _assert_run(failed, failed_options, failure=True)
        for worker in failed:
            _check_adam_trajectory(worker["records"])
        rows = [json.loads(path.read_text()) for path in (tmp_path / "failed").glob("rank-*.json")]
        assert len({row["failure"] for row in rows}) == 1
    print(
        json.dumps(
            {
                "scenario": options,
                "elementwise_comparison": True,
                "nonzero_gradients": True,
                "varying_rewards": True,
                "strict_replay": fixture.audit_kind == "coupled_replay",
                "input_audit": fixture.audit_kind,
                "drained_evaluation": True,
                "exact_loaded_state": True,
                "valid_continuation": True,
                "ordered_continuation": continuation_residuals,
                "pending_during_optimizer": True,
            }
        )
    )
