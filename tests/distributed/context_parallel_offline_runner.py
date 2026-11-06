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

"""Finite preference acquisition with native optimizer and fresh-process state proof."""

import copy
import json
import os
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from offline_dpo_cp_fixture import build_fixture, make_config
from offline_dpo_observations import assert_preference_rows, observe_preferences
from online_optimizer_observations import observe_optimizer_steps
from test_context_parallel_state import _assert_ema_trajectory, _state
from test_context_parallel_training import _compare, _compare_bf16_updates
from test_model_preparation import initialize_cuda, runtime_observation


def _worker(rank, store_path, result_dir, options):
    wrap = (
        "FluxTransformerBlock,FluxSingleTransformerBlock"
        if options["model"] == "flux1"
        else "WanTransformerBlock"
    )
    os.environ.update(
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP=wrap,
        FSDP_CPU_RAM_EFFICIENT_LOADING="false",
        FSDP_RESHARD_AFTER_FORWARD="true",
        FSDP_ACTIVATION_CHECKPOINTING="true",
    )
    degree = options["tp_size"] * options["cp_size"]
    initialize_cuda(rank, store_path, world_size=2 * degree)
    result = {"rank": rank, **options}
    try:
        from flow_factory.trainers.loader import load_trainer

        trainer = load_trainer(make_config(Path(options["fixture"]), result_dir, options))
        result["runtime"] = runtime_observation(trainer.accelerator)
        assert trainer.dataloader.sampler.num_replicas == 2
        assert trainer.dataloader.sampler.rank == rank // degree
        loaded = _state(trainer)
        records = {
            "initial_parameters": loaded["parameters"],
            "updates": [],
            "backward_losses": [],
            "batches": [],
        }
        saved = {}
        saving = False
        save = trainer._save_exact_training_state
        optimize = trainer.optimize_batch

        def observed_save(path):
            nonlocal saving
            saved[trainer.cycle_index] = _state(trainer)
            saving = True
            try:
                return save(path)
            finally:
                saving = False

        def observed_batch(batch):
            records["batches"].append(batch.record_ids)
            return optimize(batch)

        trainer._save_exact_training_state = observed_save
        trainer.optimize_batch = observed_batch
        with (
            observe_preferences(trainer) as preferences,
            observe_optimizer_steps(
                trainer, records, cp_size=options["cp_size"], is_saving=lambda: saving
            ),
        ):
            trainer.start()
        final = _state(trainer)
        # Full tuning has a checkpointed frozen reference; LoRA disables adapters
        # over frozen base weights, whose immutability the optimizer observer checks.
        if options["mode"] == "full":
            _compare(
                final["runtime"]["children"]["adapter_reference"],
                loaded["runtime"]["children"]["adapter_reference"],
                exact=True,
            )
        epochs = options["epochs"] - bool(options.get("resume"))
        assert trainer.step == options["epochs"] * 2
        assert trainer.progress.data_epoch == options["epochs"]
        assert trainer.progress.rollout_iteration == 0
        assert len(records["updates"]) == 2 * epochs
        assert len(records["batches"]) == 4 * epochs
        assert len(preferences) == 8 * epochs
        assert_preference_rows(preferences, cp_size=options["cp_size"])
        assert any(
            torch.count_nonzero(delta)
            for step in records["updates"]
            for delta in step["deltas"].values()
        )
        torch.save(
            {
                "loaded": loaded,
                "saved": saved,
                "final": final,
                "records": records,
                "preferences": preferences,
            },
            Path(result_dir, f"observations-{rank}.pt"),
        )
        result.update(
            status="pass",
            optimizer_steps=trainer.step,
            pairs=len(records["batches"]),
            execution_capability=("flux" if options["model"] == "flux1" else "wan")
            + "-offline-dpo-cp-v1",
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


def _run(directory, options):
    world = 2 * options["cp_size"] * options["tp_size"]
    rows = run_gloo_ranks(_worker, directory, world, options, timeout=900)
    for row in rows:
        assert "error" not in row, row.get("error")
        assert row["status"] == "pass"
    workers = [
        torch.load(directory / f"observations-{rank}.pt", weights_only=True)
        for rank in range(world)
    ]
    for worker in workers:
        last = worker["records"]["updates"][-1]
        _compare(worker["final"]["parameters"], last["parameters"], exact=True)
        expected_moments = {**worker["loaded"]["moments"], **last["moments"]}
        _compare(worker["final"]["moments"], expected_moments, exact=True)
    degree = options["cp_size"] * options["tp_size"]
    for replica in range(2):
        first = workers[replica * degree]
        for peer in workers[replica * degree + 1 : (replica + 1) * degree]:
            _compare(peer["records"], first["records"], exact=True)
            _compare(peer["preferences"], first["preferences"], exact=True)
    for epoch in range(options["epochs"] - bool(options.get("resume"))):
        batches = [
            worker["records"]["batches"][epoch * 4 : (epoch + 1) * 4]
            for worker in workers[::degree]
        ]
        identities = [identity for replica in batches for batch in replica for identity in batch]
        assert len(identities) == len(set(identities)) == 8
    return workers


def qualify_offline(tmp_path, *, model, mode, tp_size):
    """CP1/CP2 share samples; loaded state is exact and subsequent updates use BF16 bounds."""
    root = tmp_path.resolve() / "fixture"
    build_fixture(root, model, tp_size)
    common = {"fixture": str(root), "model": model, "mode": mode, "tp_size": tp_size}
    groups = {}
    for leg, cp, epochs in (
        ("reference", 1, 2),
        ("continuous", 2, 2),
        ("first", 2, 1),
        ("resumed", 2, 2),
    ):
        options = {**common, "cp_size": cp, "epochs": epochs}
        if leg == "resumed":
            options["resume"] = str(tmp_path / "first/saves/offline-dpo/checkpoints/checkpoint-1")
        groups[leg] = _run(tmp_path / leg, options)
    residuals = []
    for rank, continuous in enumerate(groups["continuous"]):
        reference = groups["reference"][rank // (2 * tp_size) * tp_size]
        first, resumed = groups["first"][rank], groups["resumed"][rank]
        _compare(continuous["records"]["batches"], reference["records"]["batches"], exact=True)
        for actual, expected in zip(continuous["preferences"], reference["preferences"]):
            _compare(actual["noising"], expected["noising"], exact=True)
        residuals.append(_compare_bf16_updates(continuous["records"], reference["records"]))
        _compare(resumed["loaded"], first["saved"][1], exact=True)
        joined = copy.deepcopy(first["records"])
        for key in ("batches", "updates"):
            joined[key] += resumed["records"][key]
        _compare(joined["batches"], continuous["records"]["batches"], exact=True)
        residuals.append(_compare_bf16_updates(joined, continuous["records"]))
        for actual, expected in zip(
            first["preferences"] + resumed["preferences"], continuous["preferences"]
        ):
            _compare(actual["noising"], expected["noising"], exact=True)
        _assert_ema_trajectory(first["loaded"], resumed["final"], joined["updates"])
        _compare(resumed["final"]["runtime"], continuous["final"]["runtime"])
    print(json.dumps({"model": model, "mode": mode, "tp_size": tp_size, "residuals": residuals}))
