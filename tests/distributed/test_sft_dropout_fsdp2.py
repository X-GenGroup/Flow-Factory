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

"""Compare raw and cached SFT editing through acquisition, conditioning and an FSDP2 update."""

import json
import os
import traceback
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from flux2_cp_fixture import build_fixture
from online_optimizer_observations import observe_optimizer_steps
from test_context_parallel_state import _cpu_tree
from test_context_parallel_training import _compare
from test_model_preparation import initialize_cuda, runtime_observation
from torch.distributed.tensor import DTensor

from flow_factory.hparams import Arguments
from flow_factory.trainers.loader import load_trainer
from flow_factory.trainers.offline import sft


def _config(fixture, output, *, cached):
    return Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "num_processes": 2,
            "parallelism": {"dp_shard_size": 2},
            "model": {
                "model_type": "flux2-klein",
                "model_name_or_path": str(fixture / "model"),
                "finetune_type": "lora",
                "target_modules": "default",
                "component_load_dtypes": "float32",
                "lora_rank": 2,
                "lora_alpha": 2,
            },
            "data": {
                "datasets": [
                    {"name": "edits", "dataset_dir": str(fixture / "data"), "train": {"weight": 1}}
                ],
                "enable_preprocess": cached,
                "preprocessing_batch_size": 1,
                "dataloader_num_workers": 0,
                "cache_dir": str(fixture / "cache"),
            },
            "train": {
                "trainer_type": "sft",
                "max_epochs": 1,
                "condition_dropout": 1.0,
                "resolution": [64, 64],
                "condition_image_size": [32, 64],
                "max_sequence_length": 8,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "num_train_timesteps": 1,
                "weighting_scheme": "uniform",
                "enable_gradient_checkpointing": False,
                "ema_decay": 0,
                "seed": 17,
            },
            "scheduler": {"dynamics_type": "ODE"},
            "eval": {"eval_freq": 0},
            "log": {
                "save_dir": str(output),
                "run_name": "sft-edits",
                "logging_backend": "none",
                "save_freq": 0,
            },
            "optimizers": [
                {"name": "default", "learning_rate": 1.0e-3, "max_grad_norm": 1.0, "eps": 1.0e-4}
            ],
        }
    )


def _worker(rank, store, output, options):
    os.environ.update(
        ACCELERATE_USE_CPU="false",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP="Flux2TransformerBlock,Flux2SingleTransformerBlock",
        FSDP_CPU_RAM_EFFICIENT_LOADING="false",
        FSDP_RESHARD_AFTER_FORWARD="true",
        FSDP_ACTIVATION_CHECKPOINTING="false",
    )
    record = {"rank": rank}
    output = Path(output)
    trainer = None
    try:
        initialize_cuda(rank, store, world_size=2)
        trainer = load_trainer(
            _config(Path(options["fixture"]), output / "saves", cached=options["cached"])
        )
        record["runtime"] = runtime_observation(trainer.accelerator)
        assert trainer.accelerator.is_fsdp2
        assert all(isinstance(p, DTensor) for p in trainer.adapter.get_trainable_parameters())
        observations = {"updates": [], "backward_losses": []}
        terms = {}
        forward_state = trainer.adapter.forward_state
        objective = sft.flow_matching_per_sample_loss

        def forward(*args, **kwargs):
            batch = kwargs["batch"]
            for key in (
                "prompt_ids",
                "prompt_embeds",
                "text_ids",
                "image_latents",
                "image_latent_ids",
            ):
                terms[key] = _cpu_tree(batch[key])
            terms["timestep"] = kwargs["times"].timestep["latent"].detach().cpu()
            return forward_state(*args, **kwargs)

        def loss(adapter, predicted_velocity, noised):
            for name, operand in [
                ("latent", noised.state),
                ("velocity", predicted_velocity),
                ("target", noised.target_velocity),
            ]:
                terms[name] = operand.components["latent"].detach().cpu()
            return objective(adapter, predicted_velocity, noised)

        with (
            patch.object(trainer.adapter, "forward_state", side_effect=forward),
            patch.object(sft, "flow_matching_per_sample_loss", side_effect=loss),
            observe_optimizer_steps(trainer, observations, cp_size=1, is_saving=lambda: False),
        ):
            trainer.start()
        assert trainer.progress.optimizer_step == 1
        (update,) = observations["updates"]
        assert torch.isfinite(update["loss"]).all()
        assert any(torch.count_nonzero(grad) for grad in update["gradients"].values())
        assert any(torch.count_nonzero(delta) for delta in update["deltas"].values())
        # The tiny Qwen template emits only EOS plus padding when a caption is dropped.
        ids = terms["prompt_ids"]
        assert torch.as_tensor(ids).count_nonzero().item() == 1
        result = {key: update[key] for key in ("loss", "gradients", "deltas")}
        torch.save({"terms": terms, "update": result}, output / f"rank-{rank}.pt")
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        (output / f"rank-{rank}.json").write_text(json.dumps(record))
        if trainer is not None:
            trainer.cleanup(wait="error" not in record)
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA ranks")
def test_klein_sft_raw_and_cached_fsdp2_edit_update(tmp_path):
    root = tmp_path / "fixture"
    build_fixture(root, "klein4b")
    rows = [
        {
            "schema_version": 2,
            "input": {
                "prompt": "a red image",
                "negative_prompt": "blue",
                "media": [{"type": "image", "path": f"{index}.png"} for index in order],
            },
            "supervision": {
                "type": "demonstration",
                "target": {"media": [{"type": "image", "path": f"{target}.png"}]},
            },
        }
        for order, target in [((0, 1), 2), ((1, 0), 3)]
    ]
    (root / "data" / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    by_mode = {}
    for mode in ("cached", "raw"):
        output = tmp_path / mode
        options = {"fixture": str(root), "cached": mode == "cached"}
        results = run_gloo_ranks(_worker, output, 2, options)
        assert all("error" not in worker for worker in results), results
        by_mode[mode] = [
            torch.load(output / f"rank-{rank}.pt", weights_only=True) for rank in range(2)
        ]
    for raw, cached in zip(by_mode["raw"], by_mode["cached"], strict=True):
        _compare(raw, cached, exact=True)
