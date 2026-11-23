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

"""Native SD3.5 DRaFT-K suffix and continuation on two FSDP2 CUDA ranks."""

import json
import os
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from online_optimizer_observations import observe_optimizer_steps
from test_context_parallel_state import _state
from test_context_parallel_training import _compare
from test_draft_training import _config
from test_model_preparation import initialize_cuda, runtime_observation
from test_refl_training import build_fixture

from flow_factory.hparams import ParallelismArguments
from flow_factory.trainers.loader import load_trainer


def _fsdp_config(fixture, saves, resume):
    config = _config(fixture, saves, resume, distributed=True, cpu=False)
    config.mixed_precision = "bf16"
    config.parallelism_args = ParallelismArguments(dp_shard_size=2)
    return config


def _check_endpoint(trainer):
    batch = next(iter(trainer.dataloader))
    trainer.adapter.rollout()
    with torch.no_grad(), trainer.autocast():
        initial = trainer.sample_batch(
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
        packed = next(trainer._iter_prefetched_batches(initial, len(initial)))
        actual = trainer._sample_endpoint(packed).components["latent"]
    expected = torch.stack([item.all_latents[-1] for item in expected]).to(actual)
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=2e-3)


def _worker(rank, store, output, options):
    os.environ.update(
        ACCELERATE_USE_CPU="false",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP="JointTransformerBlock",
        FSDP_CPU_RAM_EFFICIENT_LOADING="false",
        FSDP_RESHARD_AFTER_FORWARD="true",
        FSDP_ACTIVATION_CHECKPOINTING="true",
    )
    record = {"rank": rank}
    output = Path(output)
    trainer = None
    try:
        torch.use_deterministic_algorithms(True)
        initialize_cuda(rank, store, world_size=2)
        trainer = load_trainer(
            _fsdp_config(Path(options["fixture"]), output / "saves", options.get("resume"))
        )
        record["runtime"] = runtime_observation(trainer.accelerator)
        assert record["runtime"]["fsdp_version"] == 2
        assert record["runtime"]["process_group_backend"] == "nccl"
        assert trainer.accelerator.state.fsdp_plugin.activation_checkpointing
        if not options.get("resume"):
            _check_endpoint(trainer)
            # The inference probe may leave unsharded views on the prepared root.
            trainer.adapter.component_variant_registry.prepare_parameter_write()
        observations = {"updates": [], "backward_losses": [], "latent_gradients": [], "scores": []}
        decode, save = trainer.adapter.decode_differentiable_state, trainer.save_checkpoint
        saving = False

        def capture_gradient(gradient):
            assert gradient.isfinite().all() and gradient.norm() > 0
            observations["latent_gradients"].append(gradient.detach().float().cpu())

        def observed_decode(state, **kwargs):
            state.components["latent"].register_hook(capture_gradient)
            return decode(state, **kwargs)

        def observed_save(*args, **kwargs):
            nonlocal saving
            saving = True
            try:
                return save(*args, **kwargs)
            finally:
                saving = False

        feedback = trainer.differentiable_feedback

        def observed_feedback(**kwargs):
            result = feedback(**kwargs)
            observations["scores"].append(result.scores["alignment"].detach().cpu())
            return result

        trainer.differentiable_feedback = observed_feedback
        trainer.adapter.decode_differentiable_state = observed_decode
        trainer.save_checkpoint = observed_save
        with observe_optimizer_steps(trainer, observations, cp_size=1, is_saving=lambda: saving):
            trainer.start()
        assert trainer.progress.optimizer_step == 2
        updates = observations["updates"]
        assert len(updates) == (1 if options.get("resume") else 2)
        for update in updates:
            assert all(value.isfinite().all() for value in update["gradients"].values())
            assert any(torch.count_nonzero(value) for value in update["gradients"].values())
            assert any(torch.count_nonzero(value) for value in update["deltas"].values())
        assert observations["latent_gradients"]
        vae = trainer.adapter._require_component("vae")
        scorer = trainer.reward_models["alignment"].model
        assert all(
            not p.requires_grad and p.grad is None
            for p in (*vae.parameters(), *scorer.parameters())
        )
        observations["final"] = _state(trainer)["parameters"]
        torch.save(observations, output / f"rank-{rank}.pt")
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        (output / f"rank-{rank}.json").write_text(json.dumps(record))
        if trainer is not None:
            trainer.cleanup(wait="error" not in record)
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(fixture, output, *, resume=None):
    options = {"fixture": str(fixture), "resume": str(resume) if resume else None}
    records = run_gloo_ranks(_worker, output, 2, options)
    assert all("error" not in record for record in records), records
    print(records)
    return [torch.load(output / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]


def test_draft_fsdp_config_admission(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    config = _fsdp_config(tmp_path / "fixture", tmp_path / "saves", None)
    assert config.training_args.trainer_type == "draft"
    assert config.training_args.draft_backprop_steps == 2
    assert config.parallelism_args.resolve_topology(2).dp_shard_size == 2
    assert config.training_args.gradient_checkpointing_enabled
    assert config.component_execution_args is None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cuda_sd35_draft_fsdp2_suffix_and_resume(tmp_path):
    fixture = tmp_path / "fixture"
    build_fixture(fixture, "sd3")
    original = _run(fixture, tmp_path / "original")
    (checkpoint,) = (tmp_path / "original").rglob("checkpoint-1")
    resumed = _run(fixture, tmp_path / "resumed", resume=checkpoint)
    assert torch.cat([score for rank in original for score in rank["scores"]]).unique().numel() > 1
    for full, continued in zip(original, resumed, strict=True):
        _compare(continued["updates"], full["updates"][1:], exact=True)
        _compare(
            continued["latent_gradients"],
            full["latent_gradients"][-len(continued["latent_gradients"]) :],
            exact=True,
        )
        _compare(continued["final"], full["final"], exact=True)
    _compare(original[0]["final"], original[1]["final"], exact=True)
