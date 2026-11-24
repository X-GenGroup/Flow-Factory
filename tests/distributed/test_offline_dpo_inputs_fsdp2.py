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

"""Check raw and cached offline DPO edits from ordered input to the second FSDP2 update."""

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
from native_observation_utils import cpu
from offline_dpo_observations import assert_preference_rows, observe_preferences
from online_optimizer_observations import observe_optimizer_steps
from test_context_parallel_training import _compare
from test_model_preparation import initialize_cuda, runtime_observation
from torch.distributed.tensor import DTensor

from flow_factory.hparams import Arguments
from flow_factory.trainers.loader import load_trainer


def _config(fixture, output, cached):
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
                    {"name": "pairs", "dataset_dir": str(fixture / "data"), "train": {"weight": 1}}
                ],
                "enable_preprocess": cached,
                "preprocessing_batch_size": 1,
                "dataloader_num_workers": 0,
                "cache_dir": str(output / "cache"),
            },
            "train": {
                "trainer_type": "offline-dpo",
                "beta": 2.0,
                "max_epochs": 1,
                "resolution": [64, 64],
                "condition_image_size": [32, 64],
                "guidance_scale": 2.0,
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
            "log": {"save_dir": str(output), "logging_backend": "none", "save_freq": 0},
            "optimizers": [{"name": "default", "learning_rate": 1e-3, "eps": 1e-4}],
        }
    )


def _worker(rank, store, directory, options):
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
    output = Path(directory)
    record = {"rank": rank}
    trainer = None
    try:
        initialize_cuda(rank, store, world_size=2)
        trainer = load_trainer(_config(Path(options["fixture"]), output, options["cached"]))
        record["runtime"] = runtime_observation(trainer.accelerator)
        assert trainer.accelerator.is_fsdp2
        assert all(isinstance(p, DTensor) for p in trainer.adapter.get_trainable_parameters())

        arms, conditions = [], []
        observations = {"updates": [], "backward_losses": []}
        encode = trainer.adapter.encode_output_state
        forward = trainer.adapter.forward_state

        def record_output(media_batch, *args, **kwargs):
            value = encode(media_batch, *args, **kwargs)
            ((media,),) = media_batch
            arms.append((Path(media.path).name, cpu(value.clean_state.components["latent"])))
            return value

        def record_forward(*args, **kwargs):
            batch = kwargs["batch"]
            keys = (
                "prompt_ids",
                "prompt_embeds",
                "negative_prompt_ids",
                "negative_prompt_embeds",
                "text_ids",
                "image_latents",
                "image_latent_ids",
            )
            conditions.append(cpu({key: batch[key] for key in keys}))
            return forward(*args, **kwargs)

        with (
            patch.object(trainer.adapter, "encode_output_state", side_effect=record_output),
            patch.object(trainer.adapter, "forward_state", side_effect=record_forward),
            observe_preferences(trainer) as pairs,
            observe_optimizer_steps(trainer, observations, cp_size=1, is_saving=lambda: False),
        ):
            trainer.start()
        assert trainer.progress.optimizer_step == len(pairs) == 2
        assert len(arms) == 2 * len(pairs) and len(conditions) == 4 * len(pairs)
        assert_preference_rows(pairs, cp_size=1, gas=1, timesteps=1)
        for policy, reference in zip(pairs[0]["terms"][:2], pairs[0]["terms"][2:], strict=True):
            _compare(policy["prediction"], reference["prediction"], exact=True)
        assert any(
            not torch.equal(policy["prediction"], reference["prediction"])
            for policy, reference in zip(pairs[1]["terms"][:2], pairs[1]["terms"][2:], strict=True)
        ), "the second update must evaluate the evolved policy against the frozen base"
        for i in range(len(pairs)):
            first, *others = conditions[4 * i : 4 * i + 4]
            for condition in others:
                _compare(first, condition, exact=True)
            assert torch.as_tensor(first["prompt_ids"]).count_nonzero().item() == 4
            assert torch.as_tensor(first["negative_prompt_ids"]).count_nonzero().item() == 2
        updates = []
        for update in observations["updates"]:
            assert torch.isfinite(update["loss"]).all()
            for key in ("gradients", "deltas"):
                assert any(torch.count_nonzero(value) for value in update[key].values())
            updates.append({key: update[key] for key in ("loss", "gradients", "deltas", "moments")})
        torch.save(
            {"arms": arms, "conditions": conditions[::4], "pairs": pairs, "updates": updates},
            output / f"rank-{rank}.pt",
        )
        record["optimizer_steps"] = trainer.progress.optimizer_step
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        (output / f"rank-{rank}.json").write_text(json.dumps(record))
        if trainer is not None:
            trainer.cleanup(wait="error" not in record)
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA ranks")
def test_klein_offline_dpo_raw_and_cached_fsdp2_edit_update(tmp_path):
    root = tmp_path / "fixture"
    build_fixture(root, "klein4b")

    def image(index):
        return {"type": "image", "path": f"{index}.png"}

    rows = [
        {
            "schema_version": 2,
            "input": {
                "prompt": f"a {color} image",
                "negative_prompt": "small",
                "media": [image(i) for i in order],
            },
            "supervision": {
                "type": "preference",
                "chosen": {"media": [image(chosen)]},
                "rejected": {"media": [image(rejected)]},
            },
        }
        for chosen, rejected in [(2, 3), (4, 5)]
        for color, order in [("red", (0, 1)), ("blue", (1, 0))]
    ]
    (root / "data" / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    by_mode = {}
    for mode in ("raw", "cached"):
        output = tmp_path / mode
        options = {"fixture": str(root), "cached": mode == "cached"}
        results = run_gloo_ranks(_worker, output, 2, options)
        assert all("error" not in worker for worker in results), results
        by_mode[mode] = [
            torch.load(output / f"rank-{rank}.pt", weights_only=True) for rank in range(2)
        ]
    _compare(by_mode["raw"], by_mode["cached"], exact=True)
    observed_pairs = []
    references = {}
    for rank in by_mode["raw"]:
        for i, condition in enumerate(rank["conditions"]):
            (chosen, left), (rejected, right) = rank["arms"][2 * i : 2 * i + 2]
            assert not torch.equal(left, right)
            observed_pairs.append((chosen, rejected))
            color = int(torch.as_tensor(condition["prompt_ids"]).flatten()[1])
            latents = torch.as_tensor(condition["image_latents"]).squeeze(0)
            if color in references:
                _compare(references[color], latents, exact=True)
            references[color] = latents
    assert sorted(observed_pairs) == [("2.png", "3.png")] * 2 + [("4.png", "5.png")] * 2
    red, blue = references[4].chunk(2), references[5].chunk(2)
    _compare(red, tuple(reversed(blue)), exact=True)
    assert not torch.equal(*red)
