# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Native ReFL closure across registry models and generation modalities."""

import importlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import fields, replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

MODELS = {
    "sd3-5": ("sd35_coupled_fixture", {}),
    "flux1": ("flux_coupled_fixture", {}),
    "flux1-kontext": ("flux_kontext_sp_coupled_fixture", {}),
    "flux2": ("flux2_dev_coupled_fixture", {"variant": "dev"}),
    "flux2-klein": ("flux2_klein4b_coupled_fixture", {"variant": "klein4b"}),
    "flux3": ("flux3_base_coupled_fixture", {"variant": "base"}),
    "qwen-image": ("qwen_coupled_fixture", {}),
    "qwen-image-edit-plus": ("qwen_edit_coupled_fixture", {}),
    "qwen-image-2.1": ("qwen21_coupled_fixture", {"use_kv_cache": True}),
    "z-image": ("z_image_coupled_fixture", {}),
    "wan2_t2v": ("wan_overlap_fixture", {}),
    "wan2_i2v": ("wan_i2v_expanded_coupled_fixture", {}),
    "bagel": ("bagel_coupled_fixture", {}),
    "sensenova": ("sensenova_u15_coupled_fixture", {}),
    "ltx2_t2av": ("ltx20_t2av_coupled_fixture", {}),
    "ltx2_i2av": ("ltx23_i2av_coupled_fixture", {}),
    "minimax-h3-t2va": ("h3_coupled_fixture", {}),
    "minimax-h3-fl2va": ("h3_fl2va_coupled_fixture", {}),
    "minimax-h3-ref2va": ("h3_ref2va_coupled_fixture", {}),
}


def build_model_fixture(root, model):
    module = importlib.import_module(MODELS[model][0])
    if model == "minimax-h3-t2va":
        module.build_fixture(root, 2, "t2va")
    elif model == "qwen-image":
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("qwen_cp_fixture.py")), str(root)],
            check=True,
        )
    else:
        module.build_fixture(root, 2)
    # Share native reference builders while owning a small ReFL source here.
    for name in ("prompts", "tiny"):
        dataset = root / name
        dataset.mkdir(exist_ok=True)
        rows = []
        for index in range(8):
            prompt = f"a {'red' if index % 2 else 'blue'} image"
            record = {"prompt": prompt}
            if model != "minimax-h3-t2va" and hasattr(module, "dataset_record"):
                record = module.dataset_record(root, prompt, index)
            rows.append(json.dumps(record))
        (dataset / "train.jsonl").write_text("\n".join(rows) + "\n")


def model_config(root, output, model, *, tp=1, cp=1, dp=1, real_reward=False):
    from flow_factory.hparams import ReFLTrainingArguments
    from flow_factory.hparams.parallelism_args import ParallelismArguments
    from flow_factory.hparams.reward_args import MultiRewardArguments
    from flow_factory.hparams.scheduler_args import SchedulerArguments

    module = importlib.import_module(MODELS[model][0])
    options = {
        "algorithm": "grpo-guard",
        "training_mode": "full",
        "tp_size": 2,
        "cp_size": 1,
        "dp_size": 1,
        **MODELS[model][1],
    }
    if model.startswith("minimax-h3") or model.startswith("ltx2"):
        options["algorithm"] = "grpo"
    if model == "sd3-5":
        options["algorithm"] = "dppo"
    # Existing fixture recipes describe TP2. Parse that source recipe using its
    # matching geometry, then validate the final ReFL config in the actual world.
    with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
        if model == "minimax-h3-t2va":
            base = module.make_config(root, output, options, "t2va")
        else:
            base = module.make_config(root, output, options)
    train = {
        field.name: getattr(base.training_args, field.name)
        for field in fields(ReFLTrainingArguments)
        if field.init and hasattr(base.training_args, field.name)
    }
    train.update(
        trainer_type="refl",
        max_epochs=2,
        group_size=1,
        ema_decay=0.0,
        unique_sample_num_per_epoch=dp,
        gradient_step_per_epoch=1,
        gradient_accumulation_steps="auto",
        num_inner_epochs=1,
    )
    base.log_args.save_freq = 0
    base.eval_args.eval_freq = 0
    return replace(
        base,
        num_processes=tp * cp * dp,
        training_args=ReFLTrainingArguments(**train),
        parallelism_args=ParallelismArguments(
            dp_shard_size=dp,
            tp_size=tp,
            cp_size=cp,
            sequence_parallel=tp > 1,
        ),
        scheduler_args=SchedulerArguments(dynamics_type="ODE", replay_steps=[1, 2], seed=17),
        reward_args=MultiRewardArguments.from_dict(
            [
                {
                    "name": "media",
                    "reward_model": "pickscore" if real_reward else "refl_reward.MediaReward",
                    "device": "cuda",
                }
            ]
        ),
        eval_reward_args=None,
    )


def train_model(root, model, *, tp=1, cp=1, dp=1, real_reward=False):
    from flow_factory.trainers.loader import load_trainer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fixture = root / "fixture"
    trainer = load_trainer(
        model_config(fixture, root / "run", model, tp=tp, cp=cp, dp=dp, real_reward=real_reward)
    )
    gradients = []
    decoded_gradients = {}
    latent_gradients = {}
    original_decode = trainer.adapter.decode_differentiable_state
    original_step = trainer._apply_optimizer_step
    parameters = trainer.adapter.get_trainable_parameters()

    def local(tensor):
        return tensor.to_local() if hasattr(tensor, "to_local") else tensor

    initial = [local(p).detach().clone() for p in parameters]

    def observed_decode(state, **kwargs):
        for name, value in state.components.items():
            value.register_hook(
                lambda grad, field=name: latent_gradients.setdefault(field, []).append(
                    grad.detach().float().norm().item()
                )
            )
        # Decoder stochasticity, when configured, must see the same draw on both
        # sides. Native inference and optimization share the decoder operations.
        rng = torch.cuda.get_rng_state()
        media = original_decode(state, **kwargs)
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.cuda.set_rng_state(rng)
            if model.startswith("minimax-h3"):
                from flow_factory.models.minimax_h3.workflow import _normalize_geometry

                video, audio, _ = trainer.adapter.decode_latents(
                    state, geometry=_normalize_geometry(kwargs["batch"]), output_type="pt"
                )
                reference = {"video": video, "audio": audio}
            else:
                reference = trainer.adapter._decode_differentiable_state(state, **kwargs)
        for name in media:
            torch.testing.assert_close(media[name], reference[name], atol=0, rtol=0)
        for name, value in media.items():
            value.register_hook(
                lambda grad, field=name: decoded_gradients.setdefault(field, []).append(
                    grad.detach().float().norm().item()
                )
            )
        return media

    def observed_step(info):
        values = [local(p.grad).float().norm() for p in parameters if p.grad is not None]
        norm = torch.stack(values).norm()
        assert torch.isfinite(norm) and norm > 0
        gradients.append(norm.item())
        return original_step(info)

    trainer.adapter.decode_differentiable_state = observed_decode
    trainer._apply_optimizer_step = observed_step
    try:
        trainer.start()
        assert len(gradients) == 2, gradients
        assert any(
            not torch.equal(before, local(after)) for before, after in zip(initial, parameters)
        )
        assert set(decoded_gradients) == set(trainer.adapter.differentiable_decode_fields)
        assert all(len(norms) == 2 and min(norms) > 0 for norms in decoded_gradients.values())
        assert set(latent_gradients) == set(trainer.adapter.trajectory_component_order)
        assert all(len(norms) == 2 and min(norms) > 0 for norms in latent_gradients.values())
        for name in trainer.adapter.differentiable_decode_modules:
            decoder = trainer.adapter._require_component(name)
            assert not decoder.training
            assert all(not p.requires_grad and p.grad is None for p in decoder.parameters())
        for scorer in trainer.reward_models.values():
            assert all(not p.requires_grad and p.grad is None for p in scorer.model.parameters())
        result = {
            "model": model,
            "policy_gradient_norms": gradients,
            "media_gradient_norms": decoded_gradients,
            "latent_gradient_norms": latent_gradients,
        }
        if real_reward:
            result["reward_revision"] = trainer.reward_models["media"].resolved_model_revision
        if torch.distributed.is_initialized():
            from test_model_preparation import runtime_observation

            result["runtime"] = runtime_observation(trainer.accelerator)
        print(json.dumps(result), flush=True)
        return result
    finally:
        trainer.cleanup(wait=True)


def worker(root, model):
    torch.cuda.set_device(0)
    build_model_fixture(root / "fixture", model)
    result = train_model(root, model)
    (root / "result.json").write_text(json.dumps(result))


def distributed_worker(rank, store_path, result_dir, options):
    import torch.distributed as dist

    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="8",
        LOCAL_WORLD_SIZE="8",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29876",
        ACCELERATE_USE_FSDP="true",
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP="LTX2VideoTransformerBlock",
        FSDP_CPU_RAM_EFFICIENT_LOADING="false",
        FSDP_RESHARD_AFTER_FORWARD="true",
        FSDP_ACTIVATION_CHECKPOINTING="true",
    )
    torch.cuda.set_device(rank)
    result = {"rank": rank}
    try:
        dist.init_process_group(
            "nccl",
            store=dist.FileStore(store_path, 8),
            rank=rank,
            world_size=8,
            timeout=timedelta(seconds=180),
        )
        result.update(train_model(Path(options["root"]), "ltx2_i2av", tp=2, cp=2, dp=2))
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_refl_model_closure(tmp_path, model):
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), model],
        check=True,
        timeout=600,
    )
    assert json.loads((tmp_path / "result.json").read_text())["model"] == model


def test_refl_fixture_covers_complete_model_registry():
    from flow_factory.models.registry import list_registered_models

    assert set(MODELS) == set(list_registered_models())


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires eight CUDA devices")
def test_cuda_refl_joint_tp_sp_cp(tmp_path):
    from conftest import run_gloo_ranks

    build_model_fixture(tmp_path / "fixture", "ltx2_i2av")
    results = run_gloo_ranks(
        distributed_worker, tmp_path / "ranks", 8, {"root": str(tmp_path)}, timeout=600
    )
    assert all("error" not in result for result in results), results
    assert all(set(result["media_gradient_norms"]) == {"video", "audio"} for result in results)


if __name__ == "__main__":
    worker(Path(sys.argv[1]), sys.argv[2])
