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

"""Native image ReFL gradients, updates and drained continuation on selected meshes."""

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
from test_context_parallel_training import _compare
from test_model_preparation import runtime_observation


def build_fixture(root, family):
    script = "sd35_cp_fixture.py" if family == "sd3" else "flux_sft_fixture.py"
    command = [sys.executable, str(Path(__file__).with_name(script)), str(root)]
    if family == "sd3":
        command.append("medium")
    subprocess.run(command, check=True)
    prompts = root / "prompts"
    prompts.mkdir()
    (prompts / "train.jsonl").write_text(
        "".join(
            json.dumps({"prompt": f"a {'red' if i % 2 else 'blue'} image"}) + "\n" for i in range(8)
        )
    )


def config(root, output, options):
    from flow_factory.hparams import Arguments

    return Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "parallelism": {
                "dp_shard_size": 2 if options["fsdp"] else 1,
                "dp_replicate_size": 1 if options["fsdp"] else 2,
                "tp_size": options["tp"],
                "cp_size": options["cp"],
                "sequence_parallel": options["tp"] > 1,
            },
            "model": {
                "model_type": "sd3-5" if options["family"] == "sd3" else "flux1",
                "model_name_or_path": str(root / "model"),
                "target_components": ["transformer"],
                "finetune_type": "full",
                "target_modules": "all",
                "component_load_dtypes": "float32",
                "trainable_parameters_dtype": "float32",
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
                "cache_dir": str(root / "cache"),
                "dataloader_num_workers": 0,
                "preprocessing_batch_size": 1,
            },
            "train": {
                "trainer_type": "refl",
                "max_epochs": 2,
                "per_device_batch_size": 1,
                "resolution": [64, 64],
                "num_inference_steps": 3,
                "guidance_scale": 1.0 if options["family"] == "sd3" else 3.5,
                "group_size": 1,
                "unique_sample_num_per_epoch": 4,
                "gradient_step_per_epoch": 1,
                "enable_gradient_checkpointing": True,
                "seed": 17,
                "shuffle_samples": False,
            },
            "scheduler": {"dynamics_type": "ODE", "replay_steps": [1, 2], "seed": 17},
            "rewards": [
                {"name": "pixel", "reward_model": "refl_reward.PixelReward", "device": "cuda"}
            ],
            "acceleration": {
                "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
            },
            "optimizers": [
                {"name": "default", "learning_rate": 1e-4, "eps": 1e-4, "max_grad_norm": 1.0}
            ],
            "log": {
                "run_name": "refl-native",
                "save_model_only": False,
                "save_freq": 1,
                "save_dir": str(Path(output) / "saves"),
                "logging_backend": "none",
                "verbose": False,
            },
            "eval": {"eval_freq": 0},
        }
    )


def worker(rank, store_path, result_dir, options):
    world = 2 * options["tp"] * options["cp"]
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        ACCELERATE_USE_FSDP=str(options["fsdp"]).lower(),
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP="FluxTransformerBlock,FluxSingleTransformerBlock",
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
            timeout=timedelta(seconds=180),
        )
        from flow_factory.trainers import loader

        if options.get("fp32_control"):
            from contextlib import nullcontext

            original_accelerator = loader.Accelerator

            class FP32ControlAccelerator(original_accelerator):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.state.fsdp_plugin.set_mixed_precision("fp32", override=True)

            loader.Accelerator = FP32ControlAccelerator
        trainer = loader.load_trainer(config(Path(options["fixture"]), result_dir, options))
        if options.get("fp32_control"):
            trainer.autocast = nullcontext
        result["runtime"] = runtime_observation(trainer.accelerator)
        loaded = _state(trainer)
        records = {
            "initial_parameters": loaded["parameters"],
            "updates": [],
            "backward_losses": [],
            "saved": {},
        }
        optimize, save = trainer.optimize, trainer.save_checkpoint
        decoder = trainer.adapter.decode_differentiable_state
        saving = False
        decoded_gradients = []

        def observed_decode(state, **kwargs):
            assert not state.components["latent"].is_leaf
            media = decoder(state, **kwargs)
            media["image"].register_hook(
                lambda g: decoded_gradients.append(g.detach().float().norm().item())
            )
            return media

        def observed_optimize(samples):
            cycle = len(records.get("sample_files", []))
            path = Path(result_dir) / f"samples-{cycle}-rank-{rank}.pt"
            torch.save(samples, path)
            records.setdefault("sample_files", []).append(str(path))
            if options.get("common_samples"):
                replica = rank // (options["tp"] * options["cp"])
                samples = torch.load(
                    Path(options["common_samples"]) / f"samples-{cycle}-rank-{replica}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            assert all(not s.all_latents.requires_grad for s in samples)
            return optimize(samples)

        def observed_save(directory, epoch=None):
            nonlocal saving
            records["saved"][int(epoch)] = _state(trainer)
            saving = True
            try:
                return save(directory, epoch=epoch)
            finally:
                saving = False

        trainer.adapter.decode_differentiable_state = observed_decode
        trainer.optimize = observed_optimize
        trainer.save_checkpoint = observed_save
        common_state = (
            torch.load(Path(options["common_samples"]) / "observations-0.pt", weights_only=False)
            if options.get("common_samples")
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
        assert len(records["updates"]) == (1 if options.get("resume") else 2)
        assert decoded_gradients and all(g > 0 for g in decoded_gradients)
        assert all(
            p.grad is None and not p.requires_grad
            for p in trainer.adapter._require_component("vae").parameters()
        )
        assert all(
            p.grad is None and not p.requires_grad
            for p in trainer.reward_models["pixel"].model.parameters()
        )
        for update in records["updates"]:
            assert all(torch.isfinite(v).all() for v in update["gradients"].values())
            assert any(torch.count_nonzero(v) for v in update["gradients"].values())
            assert any(torch.count_nonzero(v) for v in update["deltas"].values())
        records["decoded_gradient_norms"] = decoded_gradients
        torch.save(records, Path(result_dir) / f"observations-{rank}.pt")
        result["updates"] = len(records["updates"])
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if trainer is not None:
            trainer.cleanup(wait=True)
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        if dist.is_initialized():
            dist.destroy_process_group()


def run(
    root,
    fixture,
    family,
    *,
    fsdp=False,
    tp=1,
    cp=1,
    resume=None,
    common_samples=None,
    fp32_control=False,
):
    options = dict(
        fixture=str(fixture),
        family=family,
        fsdp=fsdp,
        tp=tp,
        cp=cp,
        resume=resume,
        common_samples=str(common_samples) if common_samples else None,
        fp32_control=fp32_control,
    )
    results = run_gloo_ranks(worker, root, 2 * tp * cp, options, timeout=600)
    assert all("error" not in r for r in results), results
    records = [
        torch.load(root / f"observations-{r}.pt", weights_only=False) for r in range(2 * tp * cp)
    ]
    for record in records[1:]:
        _compare(record["final"]["parameters"], records[0]["final"]["parameters"], exact=True)
    return records


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cuda_sd35_refl_ddp_resume(tmp_path):
    fixture = tmp_path / "fixture"
    build_fixture(fixture, "sd3")
    original = run(tmp_path / "original", fixture, "sd3")
    checkpoints = list((tmp_path / "original" / "saves").rglob("checkpoint-1"))
    assert len(checkpoints) == 1
    resumed = run(tmp_path / "resumed", fixture, "sd3", resume=str(checkpoints[0]))
    for a, b in zip(original, resumed, strict=True):
        _compare(b["loaded"], a["saved"][1], exact=True)
        _compare(b["updates"], a["updates"][1:], exact=True)
        _compare(b["final"], a["final"], exact=True)


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires eight CUDA devices")
def test_cuda_flux_refl_tp_sp_cp(tmp_path):
    fixture = tmp_path / "fixture"
    build_fixture(fixture, "flux")
    plain = run(tmp_path / "plain", fixture, "flux", fsdp=True)
    composed = run(tmp_path / "composed", fixture, "flux", fsdp=True, tp=2, cp=2)
    reference = run(tmp_path / "fp32-reference", fixture, "flux", fsdp=True, fp32_control=True)
    diagnostic = run(
        tmp_path / "common-state",
        fixture,
        "flux",
        fsdp=True,
        tp=2,
        cp=2,
        common_samples=tmp_path / "fp32-reference",
        fp32_control=True,
    )
    # Ordinary runs above retain their independent BF16 rollouts. The diagnostic
    # reuses saved operands and reference parameters to isolate derivative parity.
    for record in diagnostic:
        for actual, expected in zip(record["updates"], reference[0]["updates"], strict=True):
            assert set(actual["gradients"]) == set(expected["gradients"])
            for name in expected["gradients"]:
                torch.testing.assert_close(
                    actual["gradients"][name], expected["gradients"][name], atol=2e-6, rtol=2e-4
                )
            for name in expected["deltas"]:
                torch.testing.assert_close(
                    actual["deltas"][name], expected["deltas"][name], atol=2e-6, rtol=2e-4
                )
