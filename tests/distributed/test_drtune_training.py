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

"""Native SD3.5 endpoint, exact resume and DDP checks for DRTune."""

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
from test_model_preparation import runtime_observation
from test_refl_training import build_fixture
from torch.nn.parallel import DistributedDataParallel

from flow_factory.hparams import Arguments
from flow_factory.trainers.loader import load_trainer


def _config(fixture, saves, *, resume=None, distributed=False, cpu=True):
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
        seed=57,
    )
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


def _train_with_masks(trainer):
    calls = []
    scores = []
    forward = trainer.adapter.forward_state
    step = trainer._apply_optimizer_step

    def observed_forward(**kwargs):
        calls.append(kwargs["prediction_gradients"])
        return forward(**kwargs)

    def observed_step(loss_info):
        gradients = [
            parameter.grad
            for parameter in trainer.adapter.get_trainable_parameters()
            if parameter.grad is not None
        ]
        assert all(gradient.isfinite().all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
        scores.extend(float(score) for score in loss_info["reward/alignment"])
        return step(loss_info)

    with (
        patch.object(trainer.adapter, "forward_state", side_effect=observed_forward),
        patch.object(trainer, "_apply_optimizer_step", side_effect=observed_step),
    ):
        trainer.start()
    assert len(set(scores)) > 1
    masks = [calls[begin : begin + 4] for begin in range(0, len(calls), 4)]
    assert masks
    assert all(mask in (["parameters", "none"] * 2, ["none", "parameters"] * 2) for mask in masks)
    return masks


def _check_frozen_auxiliaries(trainer):
    vae = trainer.adapter._require_component("vae")
    scorer = trainer.reward_models["alignment"].model
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in (*vae.parameters(), *scorer.parameters())
    )


def test_native_sd35_drtune_endpoint_and_resume_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    fixture = tmp_path / "fixture"
    build_fixture(fixture, "sd3")
    original = tmp_path / "original"
    trainer = load_trainer(_config(fixture, original))
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
            trajectory_indices=[-1],
            compute_log_prob=False,
            generator=torch.Generator().manual_seed(13),
        )
        packed = next(trainer._iter_prefetched_batches(inputs, len(inputs)))
        endpoint = trainer._sample_endpoint(packed).components["latent"]
        assert endpoint.requires_grad
        torch.testing.assert_close(
            endpoint, torch.stack([sample.all_latents[-1] for sample in expected])
        )
        del endpoint
        before = _weights(trainer)
        masks = _train_with_masks(trainer)
        weights = _weights(trainer)
        assert trainer.progress.optimizer_step == 2
        assert any(not torch.equal(a, b) for a, b in zip(before, weights, strict=True))
        _check_frozen_auxiliaries(trainer)
    finally:
        trainer.cleanup(wait=True)

    (checkpoint,) = original.rglob("checkpoint-1")
    resumed = load_trainer(_config(fixture, tmp_path / "resumed", resume=checkpoint))
    resumed.autocast = nullcontext
    try:
        assert resumed.progress.optimizer_step == 1
        resumed_masks = _train_with_masks(resumed)
        assert resumed_masks == masks[len(masks) // 2 :]
        assert resumed.progress.optimizer_step == 2
        for expected, actual in zip(weights, _weights(resumed), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        resumed.cleanup(wait=True)


def _ddp_worker(rank, store, output, options):
    cpu = options["cpu"]
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        ACCELERATE_USE_CPU=str(cpu).lower(),
        ACCELERATE_USE_FSDP="false",
    )
    record = {"rank": rank}
    trainer = None
    output = Path(output)
    try:
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
            _config(Path(options["fixture"]), output / "saves", distributed=True, cpu=cpu)
        )
        if cpu:
            trainer.autocast = nullcontext
        assert isinstance(trainer.model_bundle, DistributedDataParallel)
        record["runtime"] = runtime_observation(trainer.accelerator)
        before = _weights(trainer)
        record["masks"] = _train_with_masks(trainer)
        weights = _weights(trainer)
        assert trainer.progress.optimizer_step == 2
        assert any(not torch.equal(a, b) for a, b in zip(before, weights, strict=True))
        assert all(weight.isfinite().all() for weight in weights)
        _check_frozen_auxiliaries(trainer)
        torch.save([weight.cpu() for weight in weights], output / f"weights-{rank}.pt")
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        (output / f"rank-{rank}.json").write_text(json.dumps(record))
        if trainer is not None:
            trainer.cleanup(wait="error" not in record)
        if dist.is_initialized():
            dist.destroy_process_group()


def _assert_ddp_update(tmp_path, *, cpu):
    fixture = tmp_path / "fixture"
    build_fixture(fixture, "sd3")
    output = tmp_path / "ddp"
    left, right = run_gloo_ranks(_ddp_worker, output, 2, {"fixture": str(fixture), "cpu": cpu})
    assert "error" not in left and "error" not in right, (left, right)
    assert left["masks"] == right["masks"]
    print(left, right)
    a, b = [torch.load(output / f"weights-{rank}.pt", weights_only=True) for rank in range(2)]
    for expected, actual in zip(a, b, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_native_sd35_drtune_cpu_ddp(tmp_path):
    _assert_ddp_update(tmp_path, cpu=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_native_sd35_drtune_cuda_ddp(tmp_path):
    _assert_ddp_update(tmp_path, cpu=False)
