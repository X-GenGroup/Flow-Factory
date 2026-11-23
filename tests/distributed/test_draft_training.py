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
from test_refl_training import build_fixture
from torch.nn.parallel import DistributedDataParallel

from flow_factory.hparams import Arguments
from flow_factory.trainers.loader import load_trainer


def _config(fixture, saves, resume=None, *, distributed=False, cpu=True):
    raw = yaml.safe_load(Path("examples/draft/lora/sd3_5/pickscore.yaml").read_text())
    raw.update(num_processes=1, mixed_precision="no", config_file=None)
    raw["model"].update(
        model_name_or_path=str(fixture / "model"), finetune_type="full", target_modules="all"
    )
    if distributed:
        raw["num_processes"] = 2
    if resume is not None:
        raw["model"].update(resume_type="state", resume_path=str(resume))
    raw["train"].update(
        max_epochs=2,
        resolution=64,
        num_inference_steps=3,
        unique_sample_num_per_epoch=2,
        enable_gradient_checkpointing=True,
        draft_backprop_steps=2,
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
        save_dir=str(saves), run_name="draft-native", save_freq=1, logging_backend="none"
    )
    raw["eval"]["eval_freq"] = 0
    return Arguments.from_dict(raw)


def _weights(trainer):
    return [p.detach().clone() for p in trainer.adapter.get_trainable_parameters()]


def test_native_sd35_endpoint_and_next_update_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    fixture = tmp_path / "fixture"
    build_fixture(fixture, "sd3")
    trainer = load_trainer(_config(fixture, tmp_path / "original"))
    trainer.autocast = nullcontext
    try:
        batch = next(iter(trainer.dataloader))
        trainer.adapter.rollout()
        with patch.object(
            trainer.adapter, "forward", side_effect=AssertionError("unexpected denoising")
        ):
            initial = trainer.sample_batch(
                batch,
                trajectory_indices=[0],
                initial_state_only=True,
                compute_log_prob=False,
                generator=torch.Generator().manual_seed(13),
            )
        assert all(s.image is None and s.all_latents.shape[0] == 1 for s in initial)
        expected = trainer.sample_batch(
            batch,
            trajectory_indices=[-1],
            compute_log_prob=False,
            generator=torch.Generator().manual_seed(13),
        )
        packed = next(trainer._iter_prefetched_batches(initial, len(initial)))
        with torch.no_grad():
            terminal = trainer._sample_endpoint(packed).components["latent"]
        torch.testing.assert_close(
            terminal, torch.stack([item.all_latents[-1] for item in expected])
        )

        before = _weights(trainer)
        trainer.start()
        final = _weights(trainer)
        assert trainer.progress.optimizer_step == 2
        assert any(not torch.equal(a, b) for a, b in zip(before, final, strict=True))
        vae = trainer.adapter._require_component("vae")
        reward = trainer.reward_models["alignment"].model
        assert all(
            not p.requires_grad and p.grad is None
            for p in (*vae.parameters(), *reward.parameters())
        )
    finally:
        trainer.cleanup(wait=True)

    checkpoint = next((tmp_path / "original").rglob("checkpoint-1"))
    resumed = load_trainer(_config(fixture, tmp_path / "resumed", checkpoint))
    resumed.autocast = nullcontext
    try:
        assert resumed.progress.optimizer_step == 1
        resumed.start()
        assert resumed.progress.optimizer_step == 2
        for expected, actual in zip(final, _weights(resumed), strict=True):
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
    if not cpu:
        torch.cuda.set_device(rank)
    trainer = None
    record = {"rank": rank}
    try:
        dist.init_process_group(
            "gloo" if cpu else "nccl",
            store=dist.FileStore(store, 2),
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=90),
        )
        config = _config(
            Path(options["fixture"]), Path(output) / "saves", distributed=True, cpu=cpu
        )
        trainer = load_trainer(config)
        if cpu:
            trainer.autocast = nullcontext
        assert isinstance(trainer.model_bundle, DistributedDataParallel)
        before = _weights(trainer)
        trainer.start()
        assert trainer.progress.optimizer_step == 2
        weights = _weights(trainer)
        assert any(not torch.equal(a, b) for a, b in zip(before, weights, strict=True))
        assert all(p.isfinite().all() for p in weights)
        torch.save([p.cpu() for p in weights], Path(output) / f"weights-{rank}.pt")
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        Path(output, f"rank-{rank}.json").write_text(json.dumps(record))
        if trainer is not None:
            trainer.cleanup(wait="error" not in record)
        if dist.is_initialized():
            dist.destroy_process_group()


def _assert_ddp_update(tmp_path, *, cpu):
    fixture = tmp_path / "fixture"
    build_fixture(fixture, "sd3")
    output = tmp_path / "ddp"
    records = run_gloo_ranks(_ddp_worker, output, 2, {"fixture": str(fixture), "cpu": cpu})
    assert all("error" not in record for record in records), records
    left, right = [
        torch.load(output / f"weights-{rank}.pt", weights_only=True) for rank in range(2)
    ]
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_native_sd35_draft_cpu_ddp(tmp_path):
    _assert_ddp_update(tmp_path, cpu=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_native_sd35_draft_cuda_ddp(tmp_path):
    _assert_ddp_update(tmp_path, cpu=False)
