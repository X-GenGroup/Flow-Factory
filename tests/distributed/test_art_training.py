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

"""Native ART lifecycle: frozen anchors, configurable replay, CP and exact resume."""

import json
import os
import subprocess
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from online_optimizer_observations import observe_optimizer_steps
from test_context_parallel_state import _state
from test_context_parallel_training import _check_adam_trajectory, _compare, _compare_bf16_updates
from test_model_preparation import runtime_observation


def _build_fixture(root, model):
    script = "flux_sft_fixture.py" if model == "flux" else "wan_sft_fixture.py"
    command = [sys.executable, str(Path(__file__).with_name(script)), str(root)]
    if model == "wan":
        command.append("4")  # Four native heads admit TP2 x CP2.
    subprocess.run(command, check=True)
    prompts = root / "prompts"
    prompts.mkdir()
    noun = "image" if model == "flux" else "video"
    (prompts / "train.jsonl").write_text(
        "".join(
            json.dumps({"prompt": f"a {'red' if i % 2 else 'blue'} {noun}"}) + "\n"
            for i in range(8)
        )
    )


def _config(root, output, options):
    from flow_factory.hparams import Arguments

    flux = options["model"] == "flux"
    return Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "parallelism": {
                "dp_shard_size": 2 if not flux else 1,
                "dp_replicate_size": 2 if flux else 1,
                "tp_size": options["tp"],
                "cp_size": options["cp"],
                "sequence_parallel": options["tp"] > 1,
            },
            "model": {
                "model_type": "flux1" if flux else "wan2_t2v",
                "model_name_or_path": str(root / "model"),
                "target_components": ["transformer"],
                "finetune_type": "lora" if flux else "full",
                "target_modules": "default" if flux else "all",
                "lora_rank": 2,
                "lora_alpha": 2,
                "component_load_dtypes": "float32",
                "trainable_parameters_dtype": "float32",
                **({} if flux else {"compute_precision_policy": "cross_attention_fp32_v1"}),
                **(
                    {"resume_type": "state", "resume_path": options["resume"]}
                    if options.get("resume")
                    else {}
                ),
            },
            "data": {
                "datasets": [
                    {"name": "tiny", "dataset_dir": str(root / "prompts"), "train": {"weight": 1}}
                ],
                "cache_dir": str(root / "art-cache"),
                "dataloader_num_workers": 0,
                "preprocessing_batch_size": 1,
                "sampler_type": "rank_local",
            },
            "train": {
                "trainer_type": "art",
                "max_epochs": 2,
                "per_device_batch_size": 1,
                "resolution": [64, 64] if flux else [32, 32],
                **({} if flux else {"num_frames": 5, "frame_rate": 4}),
                "num_inference_steps": 3,
                "guidance_scale": 3.5 if flux else 1.0,
                "group_size": 2,
                "unique_sample_num_per_epoch": 4,
                **options.get("train", {}),
                "enable_gradient_checkpointing": True,
                "seed": 17,
                "shuffle_samples": True,
                "advantage_aggregation": "sum",
            },
            "scheduler": {"dynamics_type": "ODE", "seed": 17},
            "rewards": [
                {
                    "name": "score",
                    "reward_model": (
                        "flux_online_reward.ImageReward"
                        if flux
                        else "wan_online_reward.MemberReward"
                    ),
                    "device": "cpu",
                    "async_reward": True,
                }
            ],
            "acceleration": {
                "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
            },
            "optimizers": [
                {"name": "default", "learning_rate": 1e-4, "eps": 1e-4, "max_grad_norm": 1.0}
            ],
            "log": {
                "run_name": "art-native",
                "save_model_only": False,
                "save_freq": 1,
                "save_dir": str(Path(output) / "saves"),
                "logging_backend": "none",
                "verbose": False,
            },
            "eval": {"eval_freq": 0},
        }
    )


def _worker(rank, store_path, result_dir, options):
    world = 2 * options["tp"] * options["cp"]
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        ACCELERATE_USE_FSDP=str(options["model"] != "flux").lower(),
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP="WanTransformerBlock",
        FSDP_CPU_RAM_EFFICIENT_LOADING="false",
        FSDP_RESHARD_AFTER_FORWARD="true",
        FSDP_ACTIVATION_CHECKPOINTING="true",
    )
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {"rank": rank, **options}
    trainer = None
    try:
        dist.init_process_group(
            "nccl",
            store=dist.FileStore(store_path, world),
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=120),
        )
        from flow_factory.trainers.common.forward_kwargs import replay_forward_kwargs
        from flow_factory.trainers.loader import load_trainer

        trainer = load_trainer(_config(Path(options["fixture"]), result_dir, options))
        result["runtime"] = runtime_observation(trainer.accelerator)
        loaded = _state(trainer)
        records = {
            "initial_parameters": loaded["parameters"],
            "updates": [],
            "backward_losses": [],
            "cycles": [],
            "saved": {},
        }
        optimize, forward, save = (
            trainer.optimize,
            trainer.adapter.forward_state,
            trainer.save_checkpoint,
        )
        saving = False
        cycle = None

        def observed_forward(**kwargs):
            output = forward(**kwargs)
            if cycle is not None and torch.is_grad_enabled():
                batch = kwargs["batch"]
                cycle["training"].append(
                    (
                        int(batch.samples[0].sampling_sample_id),
                        float(kwargs["times"].timestep["latent"].reshape(-1)[0]),
                        len(records["updates"]),
                    )
                )
            return output

        def observed_optimize(samples):
            nonlocal cycle
            # Use the actual trainer's old callbacks; audit before any new update.
            cycle = {
                "samples": [int(s.sampling_sample_id) for s in samples],
                "training": [],
                "anchors": [],
            }
            rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            trainer.adapter.train()
            try:
                for batch in trainer._iter_prefetched_batches(samples, 1):
                    for i in trainer.adapter.get_train_step_indices().tolist():
                        replay = trainer.adapter.get_replay_step(batch, i)
                        old = trainer.adapter.get_replay_callback(batch, i, "velocity").components[
                            "latent"
                        ]
                        with torch.no_grad(), trainer.autocast():
                            out = forward(
                                batch=batch,
                                state=replay.state,
                                times=replay.times,
                                next_state=replay.next_state,
                                compute_log_prob=False,
                                return_fields=("velocity",),
                                **replay_forward_kwargs(trainer, batch),
                            )
                        torch.testing.assert_close(
                            out.velocity.components["latent"], old, rtol=0, atol=0
                        )
                        cycle["anchors"].append(old.detach().cpu().clone())
            finally:
                torch.random.set_rng_state(rng[0])
                torch.cuda.set_rng_state(rng[1])
            start = len(records["updates"])
            optimize(samples)
            args = trainer.training_args
            updates_per_pass = (
                args.num_batches_per_epoch
                * len(trainer.adapter.get_train_step_indices())
                // args.gradient_accumulation_steps
            )
            assert len(records["updates"]) == start + updates_per_pass * args.num_inner_epochs
            rows = cycle["training"]
            terms = len(samples) * len(trainer.adapter.get_train_step_indices())
            assert len(rows) == terms * args.num_inner_epochs
            for inner in range(args.num_inner_epochs):
                pass_rows = rows[inner * terms : (inner + 1) * terms]
                visited = set()
                for update in range(updates_per_pass):
                    members = {
                        sample
                        for sample, _, step in pass_rows
                        if step == start + inner * updates_per_pass + update
                    }
                    assert not visited.intersection(members)
                    visited.update(members)
                assert visited == set(cycle["samples"])
            # Stored callbacks must stay frozen even after the second update.
            k = 0
            for batch in trainer._iter_prefetched_batches(samples, 1):
                for i in trainer.adapter.get_train_step_indices().tolist():
                    old = (
                        trainer.adapter.get_replay_callback(batch, i, "velocity")
                        .components["latent"]
                        .cpu()
                    )
                    torch.testing.assert_close(old, cycle["anchors"][k], rtol=0, atol=0)
                    k += 1
            records["cycles"].append(cycle)
            cycle = None

        def observed_save(directory, epoch=None):
            nonlocal saving
            records["saved"][int(epoch)] = _state(trainer)
            saving = True
            try:
                return save(directory, epoch=epoch)
            finally:
                saving = False

        trainer.optimize = observed_optimize
        trainer.adapter.forward_state = observed_forward
        trainer.save_checkpoint = observed_save
        common_state = (
            torch.load(options["common_state"], weights_only=False)
            if options.get("common_state")
            else None
        )
        with observe_optimizer_steps(
            trainer,
            records,
            cp_size=options["cp"],
            is_saving=lambda: saving,
            common_state=common_state,
        ):
            trainer.start()
        records["loaded"], records["final"] = loaded, _state(trainer)
        args = trainer.training_args
        per_cycle = (
            args.num_batches_per_epoch
            * len(trainer.adapter.get_train_step_indices())
            // args.gradient_accumulation_steps
            * args.num_inner_epochs
        )
        expected = per_cycle if options.get("resume") else 2 * per_cycle
        assert len(records["updates"]) == expected
        for update in records["updates"]:
            assert all(torch.isfinite(v).all() for v in update["gradients"].values())
            assert any(torch.count_nonzero(v) for v in update["gradients"].values())
            assert any(torch.count_nonzero(v) for v in update["deltas"].values())
        torch.save(records, Path(result_dir) / f"observations-{rank}.pt")
        result["updates"] = expected
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if trainer is not None:
            trainer.cleanup(wait=True)
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(path, fixture, model, tp=1, cp=1, resume=None, common_state=None, train=None):
    options = {
        "fixture": str(fixture),
        "model": model,
        "tp": tp,
        "cp": cp,
        "resume": resume,
        "common_state": common_state,
        "train": train or {},
    }
    results = run_gloo_ranks(_worker, path, 2 * tp * cp, options, timeout=600)
    assert all("error" not in r for r in results), results
    records = [
        torch.load(path / f"observations-{r}.pt", weights_only=False) for r in range(2 * tp * cp)
    ]
    for record in records[1:]:
        _compare(record["final"]["parameters"], records[0]["final"]["parameters"], exact=True)
    if resume is None and common_state is None:
        for record in records:
            _check_adam_trajectory(record)
    return records


def _compare_float32_continuation(actual, expected):
    """Require exact structure/non-floating state and bounded FP32 continuation.

    A fresh DDP process can reduce accumulated gradients in a different order.
    Checkpoint restoration remains bitwise checked separately; no generic BF16
    tolerance is used for this FP32 optimizer-state comparison.
    """
    from torch.utils._pytree import tree_flatten

    left, left_spec = tree_flatten(actual)
    right, right_spec = tree_flatten(expected)
    assert left_spec == right_spec
    for observed, wanted in zip(left, right, strict=True):
        if isinstance(wanted, torch.Tensor):
            floating = wanted.is_floating_point()
            torch.testing.assert_close(
                observed, wanted, rtol=1e-6 if floating else 0, atol=1e-9 if floating else 0
            )
        else:
            assert observed == wanted


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("updates,inner_epochs", [(2, 1), (3, 2)])
def test_art_flux_ddp_resume(tmp_path, updates, inner_epochs):
    fixture = tmp_path / "fixture"
    _build_fixture(fixture, "flux")
    train = {"gradient_step_per_epoch": updates, "num_inner_epochs": inner_epochs}
    if inner_epochs > 1:
        train["art_kl_threshold"] = 1e6  # Isolate scheduling from separately tested mask branches.
    baseline = _run(tmp_path / "baseline", fixture, "flux", train=train)
    checkpoints = list((tmp_path / "baseline" / "saves").rglob("checkpoint-1"))
    assert len(checkpoints) == 1
    resumed = _run(tmp_path / "resumed", fixture, "flux", resume=str(checkpoints[0]), train=train)
    for original, continued in zip(baseline, resumed, strict=True):
        _compare(continued["loaded"], original["saved"][1], exact=True)
        if inner_epochs == 1:
            _compare(
                continued["updates"], original["updates"][updates * inner_epochs :], exact=True
            )
            _compare(continued["final"], original["final"], exact=True)
        else:
            _compare_float32_continuation(
                continued["updates"], original["updates"][updates * inner_epochs :]
            )
            _compare_float32_continuation(continued["final"], original["final"])


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires eight CUDA devices")
def test_art_wan_tp_sp_cp(tmp_path):
    fixture = tmp_path / "fixture"
    _build_fixture(fixture, "wan")
    plain = _run(tmp_path / "tp", fixture, "wan", tp=2)
    cp = _run(tmp_path / "tp-cp", fixture, "wan", tp=2, cp=2)
    # Discrete trust decisions may diverge after BF16 update roundoff. First
    # retain the uninterrupted two-cycle runs above, then isolate derivatives
    # at identical parameters. The existing observer labels/replaces weights
    # only in this diagnostic leg; it does not certify equal full trajectories.
    diagnostic = _run(
        tmp_path / "common-state",
        fixture,
        "wan",
        tp=2,
        cp=2,
        common_state=str(tmp_path / "tp" / "observations-0.pt"),
    )
    from test_context_parallel_training import _bf16_tensors

    for index in range(8):
        reference = plain[(index // 4) * 2 + index % 2]
        _compare_bf16_updates(
            {**cp[index], "updates": cp[index]["updates"][:2]},
            {**reference, "updates": reference["updates"][:2]},
        )
        for observed, expected in zip(
            diagnostic[index]["updates"], reference["updates"], strict=True
        ):
            _compare(observed["before"], expected["before"], exact=True)
            _bf16_tensors(observed["gradients"], expected["gradients"], atol=1e-3)
            _bf16_tensors({"loss": observed["loss"]}, {"loss": expected["loss"]}, atol=2e-3)
