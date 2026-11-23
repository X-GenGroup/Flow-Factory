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

"""Two-rank FSDP2 ReFL with real PickScore and optional encoder sharding."""

import json
import os
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import yaml
from conftest import run_gloo_ranks
from online_optimizer_observations import observe_optimizer_steps
from test_context_parallel_state import _state
from test_context_parallel_training import _compare
from test_model_preparation import initialize_cuda, runtime_observation
from test_refl_training import build_fixture
from torch.distributed.fsdp import FSDPModule

from flow_factory.hparams import Arguments
from flow_factory.trainers.loader import load_trainer

_POLICY = {"placement": "fsdp2", "shard_group": "data"}
_PICK_SCORE_REVISION = "a4e4367c6dfa7288a00c550414478f865b875800"


def _options(root, output, options):
    recipe = Path("examples/refl/lora/sd3_5/pickscore.yaml")
    raw = yaml.safe_load(recipe.read_text())
    raw["parallelism"] = {"dp_shard_size": 2}
    raw["model"].update(
        model_name_or_path=str(root / "model"), finetune_type="full", target_modules="all"
    )
    if options.get("resume"):
        raw["model"].update(resume_type="state", resume_path=options["resume"])
    raw["data"].update(
        datasets=[
            {"name": "alignment", "dataset_dir": str(root / "prompts"), "train": {"weight": 1}}
        ],
        cache_dir=options["cache"],
        preprocessing_batch_size=3,
        preprocess_parallelism="global",
    )
    raw["train"].update(
        max_epochs=2,
        resolution=64,
        num_inference_steps=3,
        unique_sample_num_per_epoch=4,
        refl_reward_target=None,
        refl_reward_scale=1.0,
        shuffle_samples=False,
    )
    raw["scheduler"]["replay_steps"] = [1, 2]
    raw["log"].update(
        run_name="refl-sharded-auxiliary",
        logging_backend="none",
        save_dir=str(output / "saves"),
        save_freq=1,
    )
    raw["eval"]["eval_freq"] = 0
    if options["sharded"]:
        raw["component_execution"] = {
            "model": {name: _POLICY for name in ("vae", *options["encoders"])},
            "rewards": {"alignment": _POLICY},
        }
    return Arguments.from_dict(raw)


def _worker(rank, store, directory, options):
    os.environ.update(
        ACCELERATE_USE_CPU="false",
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP="JointTransformerBlock",
        FSDP_CPU_RAM_EFFICIENT_LOADING="false",
        FSDP_RESHARD_AFTER_FORWARD="true",
        FSDP_ACTIVATION_CHECKPOINTING="true",
    )
    output, root = Path(directory), Path(options["fixture"])
    record = {"rank": rank}
    trainer = None
    try:
        initialize_cuda(rank, store, world_size=2)
        torch.cuda.reset_peak_memory_stats()
        trainer = load_trainer(_options(root, output, options))
        record["runtime"] = runtime_observation(trainer.accelerator)
        assert record["runtime"]["fsdp_version"] == 2
        assert record["runtime"]["process_group_backend"] == "nccl"
        adapter = trainer.adapter
        vae = adapter._require_component("vae")
        scorer = trainer.reward_models["alignment"]
        assert scorer.resolved_model_revision == _PICK_SCORE_REVISION
        assert isinstance(vae, FSDPModule) == options["sharded"]
        for tower in (scorer.model.text_model, scorer.model.vision_model):
            assert isinstance(tower, FSDPModule) == options["sharded"]
        for name in options["encoders"]:
            if options["sharded"] and not options.get("resume"):
                encoder = adapter.component_runtime.get_component(name)
                assert isinstance(encoder, FSDPModule)
                assert {p.device.type for p in encoder.parameters()} == {"cpu"}
            elif options.get("resume"):
                assert not adapter.component_runtime.has_component_override(name)
                assert name not in adapter.component_runtime.materialized_component_names

        observations = {
            "updates": [],
            "backward_losses": [],
            "images": [],
            "media_gradients": [],
            "latent_gradients": [],
            "scores": [],
            "step_peaks": [],
            "initial": _state(trainer)["parameters"],
        }
        decode, feedback, step = (
            adapter.decode_differentiable_state,
            trainer.differentiable_feedback,
            trainer._apply_optimizer_step,
        )

        def capture(target):
            def store(gradient):
                assert torch.isfinite(gradient).all() and gradient.norm() > 0
                observations[target].append(gradient.detach().float().cpu())

            return store

        def observed_decode(state, **kwargs):
            state.components["latent"].register_hook(capture("latent_gradients"))
            media = decode(state, **kwargs)
            image = media["image"]
            image.register_hook(capture("media_gradients"))
            observations["images"].append(image.detach().float().cpu())
            return media

        def observed_feedback(**kwargs):
            result = feedback(**kwargs)
            observations["scores"].append(result.scores["alignment"].detach().float().cpu())
            return result

        def observed_step(loss_info):
            result = step(loss_info)
            observations["step_peaks"].append(torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            return result

        saving = False
        save = trainer.save_checkpoint

        def observed_save(*args, **kwargs):
            nonlocal saving
            saving = True
            try:
                return save(*args, **kwargs)
            finally:
                saving = False

        trainer.save_checkpoint = observed_save
        adapter.decode_differentiable_state = observed_decode
        trainer.differentiable_feedback = observed_feedback
        trainer._apply_optimizer_step = observed_step
        with observe_optimizer_steps(trainer, observations, cp_size=1, is_saving=lambda: saving):
            trainer.start()
        assert trainer.progress.optimizer_step == 2
        expected_steps = 1 if options.get("resume") else 2
        assert len(observations["updates"]) == expected_steps
        assert observations["media_gradients"] and observations["latent_gradients"]
        assert all(
            any(torch.count_nonzero(value) for value in update["deltas"].values())
            for update in observations["updates"]
        )
        for parameter in (*vae.parameters(), *scorer.model.parameters()):
            assert not parameter.requires_grad and parameter.grad is None
        observations["final"] = _state(trainer)["parameters"]
        torch.save(observations, output / f"rank-{rank}.pt")
        record.update(
            reward_revision=scorer.resolved_model_revision, step_peaks=observations["step_peaks"]
        )
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        (output / f"rank-{rank}.json").write_text(json.dumps(record))
        if trainer is not None:
            trainer.cleanup(wait="error" not in record)
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(root, output, *, sharded, encoders, cache, resume=None):
    options = dict(fixture=str(root), sharded=sharded, encoders=encoders, cache=str(cache))
    if resume:
        options["resume"] = str(resume)
    ranks = run_gloo_ranks(_worker, output, 2, options, timeout=600)
    assert all("error" not in rank for rank in ranks), ranks
    print(ranks)
    return [torch.load(output / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]


def qualify_sharded_refl(tmp_path, *, encoders=()):
    """Compare two uninterrupted CUDA updates and restore the sharded next update."""
    fixture = tmp_path / "fixture"
    build_fixture(fixture, "sd3")
    prompts = fixture / "prompts" / "train.jsonl"
    prompts.write_text("".join(prompts.read_text().splitlines(keepends=True)[:7]))
    reference = _run(
        fixture,
        tmp_path / "replicated",
        sharded=False,
        encoders=encoders,
        cache=tmp_path / "replicated-cache",
    )
    original = _run(
        fixture,
        tmp_path / "sharded",
        sharded=True,
        encoders=encoders,
        cache=tmp_path / "sharded-cache",
    )
    (checkpoint,) = (tmp_path / "sharded").rglob("checkpoint-1")
    resumed = _run(
        fixture,
        tmp_path / "resumed",
        sharded=True,
        encoders=encoders,
        cache=tmp_path / "sharded-cache",
        resume=checkpoint,
    )
    assert torch.cat([score for rank in original for score in rank["scores"]]).unique().numel() > 1
    for dense, sharded, restored in zip(reference, original, resumed, strict=True):
        _compare(sharded["initial"], dense["initial"], exact=True)
        for field in ("images", "media_gradients", "latent_gradients", "scores", "updates"):
            _compare(sharded[field], dense[field])
            _compare(restored[field], sharded[field][-len(restored[field]) :], exact=True)
        _compare(restored["final"], sharded["final"], exact=True)


requires_real_pickscore_cuda = pytest.mark.skipif(
    os.environ.get("FF_TEST_REAL_REWARD") != "1" or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices and selected pretrained PickScore",
)


@requires_real_pickscore_cuda
def test_cuda_sd35_refl_sharded_reward_and_decoder(tmp_path):
    qualify_sharded_refl(tmp_path)
