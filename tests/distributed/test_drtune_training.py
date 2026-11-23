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

"""Native SD3.5 sampling, exact resume and distributed updates for DRTune."""

import json
import os
import traceback
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import yaml
from conftest import run_gloo_ranks
from test_draft_training import _weights
from test_model_preparation import initialize_cuda, runtime_observation
from test_refl_training import build_fixture
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel

from flow_factory.hparams import Arguments
from flow_factory.samples import ComponentTimes, LatentState
from flow_factory.trainers.common.forward_kwargs import replay_forward_kwargs
from flow_factory.trainers.loader import load_trainer


def _config(
    fixture, saves, *, resume=None, distributed=False, cpu=True, stop_range=None, fsdp2=False
):
    raw = yaml.safe_load(Path("examples/drtune/lora/sd3_5/pickscore.yaml").read_text())
    raw.update(
        num_processes=2 if distributed else 1,
        mixed_precision="no" if cpu else "bf16",
        config_file=None,
    )
    raw["model"].update(
        model_name_or_path=str(fixture / "model"), finetune_type="full", target_modules="all"
    )
    if resume:
        raw["model"].update(resume_type="state", resume_path=str(resume))
    raw["train"].update(
        max_epochs=2,
        resolution=64,
        num_inference_steps=4,
        unique_sample_num_per_epoch=4,
        drtune_backprop_steps=2,
        drtune_stop_step_range=stop_range,
        seed=57,
    )
    if fsdp2:
        raw["parallelism"] = {"dp_shard_size": 2}
    raw["data"].update(
        datasets=[
            {"name": "tiny", "dataset_dir": str(fixture / "prompts"), "train": {"weight": 1}}
        ],
        cache_dir=str(fixture / "cache"),
    )
    raw["rewards"][0].update(
        reward_model="refl_reward.PixelReward", device="cpu" if cpu else "cuda"
    )
    raw["log"].update(
        save_dir=str(saves), run_name="drtune-native", save_freq=1, logging_backend="none"
    )
    raw["eval"]["eval_freq"] = 0
    return Arguments.from_dict(raw)


def _train_with_masks(trainer, *, capture_gradients=False):
    calls = []
    masks = []
    scores = []
    optimizer_gradients = []
    forward = trainer.adapter.forward_state
    sample_endpoint = trainer._sample_endpoint
    step = trainer._apply_optimizer_step
    stop_range = trainer.training_args.drtune_stop_step_range

    def observed_forward(**kwargs):
        calls.append((kwargs["prediction_gradients"], kwargs["return_fields"]))
        return forward(**kwargs)

    def observed_endpoint(*args):
        endpoint, metrics = sample_endpoint(*args)
        count = len(calls)
        assert count == metrics["drtune/visited_steps"]
        assert metrics["drtune/backprop_steps"] == 2
        if stop_range is None:
            assert count == 4
        else:
            assert stop_range[0] <= count <= stop_range[1]
        assert [fields for _, fields in calls] == [("next_latents",)] * (count - 1) + [
            ("next_latents",) if stop_range is None else ("velocity",)
        ]
        selected = [i for i, (mode, _) in enumerate(calls) if mode == "parameters"]
        assert len(selected) == 2 and selected[1] - selected[0] == count // 2
        assert all(mode in ("parameters", "none") for mode, _ in calls)
        masks.append([count, *selected])
        calls.clear()
        return endpoint, metrics

    def observed_step(loss_info):
        gradients = [
            parameter.grad
            for parameter in trainer.adapter.get_trainable_parameters()
            if parameter.grad is not None
        ]
        observed = [
            gradient.to_local() if isinstance(gradient, DTensor) else gradient
            for gradient in gradients
        ]
        assert all(gradient.isfinite().all() for gradient in observed)
        nonzero = torch.stack([gradient.count_nonzero() for gradient in observed]).sum()
        if trainer.accelerator.is_fsdp2:
            dist.all_reduce(nonzero)
        assert nonzero > 0
        if capture_gradients:
            optimizer_gradients.append([gradient.detach().clone().cpu() for gradient in observed])
        assert all(count == 2 for count in loss_info["drtune/backprop_steps"])
        scores.extend(float(score) for score in loss_info["reward/alignment"])
        return step(loss_info)

    with (
        patch.object(trainer.adapter, "forward_state", side_effect=observed_forward),
        patch.object(trainer, "_sample_endpoint", side_effect=observed_endpoint),
        patch.object(trainer, "_apply_optimizer_step", side_effect=observed_step),
    ):
        trainer.start()
    assert len(set(scores)) > 1
    assert masks and not calls
    return masks, optimizer_gradients


def _check_frozen_auxiliaries(trainer):
    vae = trainer.adapter._require_component("vae")
    scorer = trainer.reward_models["alignment"].model
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in (*vae.parameters(), *scorer.parameters())
    )


@pytest.mark.parametrize("stop_range", [None, (2, 3)])
def test_native_sd35_drtune_endpoint_and_resume_cpu(tmp_path, monkeypatch, stop_range):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    fixture = tmp_path / "fixture"
    build_fixture(fixture, "sd3")
    original = tmp_path / "original"
    trainer = load_trainer(_config(fixture, original, stop_range=stop_range))
    trainer.autocast = nullcontext
    try:
        batch = next(iter(trainer.dataloader))
        trainer.adapter.rollout()
        inputs = trainer.sample_batch(
            batch,
            trajectory_indices=[0],
            initial_state_only=True,
            compute_log_prob=False,
            generator=torch.Generator().manual_seed(13),
        )
        expected = trainer.sample_batch(
            batch,
            trajectory_indices=[-1] if stop_range is None else [0, 1, 2, 3],
            compute_log_prob=False,
            generator=torch.Generator().manual_seed(13),
        )
        packed = next(trainer._iter_prefetched_batches(inputs, len(inputs)))
        state, metrics = trainer._sample_endpoint(packed)
        endpoint = state.components["latent"]
        assert endpoint.requires_grad
        if stop_range is None:
            reference = torch.stack([sample.all_latents[-1] for sample in expected])
        else:
            index = metrics["drtune/visited_steps"] - 1
            live = torch.stack([sample.all_latents[index] for sample in expected])
            current = packed["timesteps"][:, index]
            with torch.no_grad():
                velocity = trainer.adapter.forward_state(
                    batch=packed,
                    state=LatentState({"latent": live}),
                    times=ComponentTimes(
                        timestep={"latent": current},
                        next_timestep={"latent": torch.zeros_like(current)},
                    ),
                    return_fields=("velocity",),
                    **replay_forward_kwargs(trainer, packed),
                ).velocity.components["latent"]
            reference = live - (current / 1000.0).reshape(-1, 1, 1, 1) * velocity
        torch.testing.assert_close(endpoint, reference)
        runtime_identity = trainer.runtime_execution_identity_payload()
        assert ("drtune_stop_rule" in runtime_identity) == (stop_range is not None)
        assert ("drtune_stop_step_range" in runtime_identity["training"]) == (
            stop_range is not None
        )
        del endpoint
        before = _weights(trainer)
        masks, gradients = _train_with_masks(trainer, capture_gradients=True)
        weights = _weights(trainer)
        assert trainer.progress.optimizer_step == 2
        assert any(not torch.equal(a, b) for a, b in zip(before, weights, strict=True))
        _check_frozen_auxiliaries(trainer)
    finally:
        trainer.cleanup(wait=True)

    (checkpoint,) = original.rglob("checkpoint-1")
    resumed = load_trainer(
        _config(fixture, tmp_path / "resumed", resume=checkpoint, stop_range=stop_range)
    )
    resumed.autocast = nullcontext
    try:
        assert resumed.progress.optimizer_step == 1
        resumed_masks, resumed_gradients = _train_with_masks(resumed, capture_gradients=True)
        assert resumed_masks == masks[len(masks) // 2 :]
        assert len(resumed_gradients) == 1
        for actual, expected_gradient in zip(resumed_gradients[0], gradients[-1], strict=True):
            torch.testing.assert_close(actual, expected_gradient, rtol=0, atol=0)
        assert resumed.progress.optimizer_step == 2
        for expected, actual in zip(weights, _weights(resumed), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        resumed.cleanup(wait=True)


def _distributed_worker(rank, store, output, options):
    cpu = options["cpu"]
    fsdp2 = options.get("fsdp2", False)
    resume = options.get("resume")
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        ACCELERATE_USE_CPU=str(cpu).lower(),
        ACCELERATE_USE_FSDP=str(fsdp2).lower(),
    )
    record = {"rank": rank}
    trainer = None
    output = Path(output)
    try:
        if fsdp2:
            os.environ.update(
                FSDP_VERSION="2",
                FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
                FSDP_TRANSFORMER_CLS_TO_WRAP="JointTransformerBlock",
                FSDP_CPU_RAM_EFFICIENT_LOADING="false",
                FSDP_RESHARD_AFTER_FORWARD="true",
                FSDP_ACTIVATION_CHECKPOINTING="true",
            )
            initialize_cuda(rank, store, world_size=2)
        else:
            if not cpu:
                torch.cuda.set_device(rank)
            dist.init_process_group(
                "gloo" if cpu else "nccl",
                store=dist.FileStore(store, 2),
                rank=rank,
                world_size=2,
                timeout=timedelta(seconds=90),
            )
        trainer = load_trainer(
            _config(
                Path(options["fixture"]),
                output / "saves",
                distributed=True,
                cpu=cpu,
                stop_range=options["stop_range"],
                fsdp2=fsdp2,
                resume=resume,
            )
        )
        if cpu:
            trainer.autocast = nullcontext
        if fsdp2:
            assert trainer.accelerator.is_fsdp2
            assert trainer.accelerator.state.fsdp_plugin.activation_checkpointing
            record["activation_checkpointing"] = True
            assert all(isinstance(p, DTensor) for p in trainer.adapter.get_trainable_parameters())
        else:
            assert isinstance(trainer.model_bundle, DistributedDataParallel)
        record["runtime"] = runtime_observation(trainer.accelerator)
        assert trainer.progress.optimizer_step == (1 if resume else 0)

        def full_weights():
            return [
                (weight.full_tensor() if isinstance(weight, DTensor) else weight).cpu()
                for weight in _weights(trainer)
            ]

        before = full_weights()
        record["masks"], gradients = _train_with_masks(trainer, capture_gradients=fsdp2)
        weights = full_weights()
        assert trainer.progress.optimizer_step == 2
        assert any(not torch.equal(a, b) for a, b in zip(before, weights, strict=True))
        assert all(weight.isfinite().all() for weight in weights)
        _check_frozen_auxiliaries(trainer)
        torch.save({"weights": weights, "gradients": gradients}, output / f"weights-{rank}.pt")
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        (output / f"rank-{rank}.json").write_text(json.dumps(record))
        if trainer is not None:
            trainer.cleanup(wait="error" not in record)
        if dist.is_initialized():
            dist.destroy_process_group()


def _assert_distributed_update(fixture, output, *, cpu, stop_range, fsdp2=False, resume=None):
    left, right = run_gloo_ranks(
        _distributed_worker,
        output,
        2,
        {
            "fixture": str(fixture),
            "cpu": cpu,
            "stop_range": stop_range,
            "fsdp2": fsdp2,
            "resume": resume,
        },
    )
    assert "error" not in left and "error" not in right, (left, right)
    assert left["masks"] == right["masks"]
    print(left, right)
    a, b = [torch.load(output / f"weights-{rank}.pt", weights_only=True) for rank in range(2)]
    for expected, actual in zip(a["weights"], b["weights"], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return (left["masks"], a, b)


@pytest.mark.parametrize("stop_range", [None, (2, 3)])
def test_native_sd35_drtune_cpu_ddp(tmp_path, stop_range):
    fixture = tmp_path / "fixture"
    build_fixture(fixture, "sd3")
    _assert_distributed_update(fixture, tmp_path / "ddp", cpu=True, stop_range=stop_range)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("stop_range", [None, (2, 3)])
def test_native_sd35_drtune_cuda_ddp(tmp_path, stop_range):
    fixture = tmp_path / "fixture"
    build_fixture(fixture, "sd3")
    _assert_distributed_update(fixture, tmp_path / "ddp", cpu=False, stop_range=stop_range)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_native_sd35_drtune_cuda_fsdp2(tmp_path):
    fixture = tmp_path / "fixture"
    build_fixture(fixture, "sd3")
    original = tmp_path / "original"
    options = {"cpu": False, "stop_range": (3, 3), "fsdp2": True}
    masks, *ranks = _assert_distributed_update(fixture, original, **options)
    assert {last == count - 1 for count, _, last in masks} == {False, True}
    (checkpoint,) = original.rglob("checkpoint-1")
    resumed_masks, *resumed = _assert_distributed_update(
        fixture, tmp_path / "resumed", resume=str(checkpoint), **options
    )
    assert resumed_masks == masks[len(masks) // 2 :]
    for expected, actual in zip(ranks, resumed, strict=True):
        assert len(expected["gradients"]) == 2 and len(actual["gradients"]) == 1
        for before, after in zip(expected["gradients"][-1], actual["gradients"][0], strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        for before, after in zip(expected["weights"], actual["weights"], strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
