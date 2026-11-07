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

"""Explicit native Flux recipe for the coupled overlap lifecycle runner."""

import json
import math
import subprocess
import sys
from pathlib import Path

from context_parallel_overlap_runner import _audit_fresh_replay as audit_samples
from context_parallel_overlap_runner import observe_coupled_terms as observe_training_terms
from coupled_observations import assert_objectives, observe_coupled_objectives
from flux_cp_model_control import qualify_model_control

audit_kind = "coupled_replay"
audit_steps = 2

wrap_classes = ("FluxTransformerBlock", "FluxSingleTransformerBlock")
observe_objectives = observe_coupled_objectives


def build_fixture(root, tp_size):
    """Build real components in a separate process before launching CUDA workers."""
    assert tp_size in (1, 2), "The four-head fixture covers TP1/TP2 with CP2"
    root = Path(root)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("flux_sft_fixture.py")), str(root)],
        check=True,
    )
    prompts = root / "prompts"
    prompts.mkdir()
    (prompts / "train.jsonl").write_text(
        "".join(
            json.dumps({"prompt": f"a {'red' if i % 2 else 'blue'} image"}) + "\n" for i in range(8)
        )
    )


def make_config(root, result_dir, options):
    """Bind model and objective defaults; the common runner owns overlap controls."""
    from flow_factory.hparams import Arguments

    algorithm = options["algorithm"]
    assert algorithm in ("grpo-guard", "dppo"), algorithm
    assert options["tp_size"] in (1, 2), "The four-head fixture covers TP1/TP2 with CP2"
    root = Path(root)
    mode = options["training_mode"]
    train = {
        "trainer_type": algorithm,
        "max_epochs": options.get("epochs", 2),
        "per_device_batch_size": 1,
        "resolution": [64, 64],
        "num_inference_steps": 3,
        "guidance_scale": 3.5,
        "group_size": 2,
        "unique_sample_num_per_epoch": 2 * options["dp_size"],
        "gradient_step_per_epoch": 2,
        "gradient_accumulation_steps": "auto",
        "num_inner_epochs": 1,
        "ema_decay": 0.9,
        "ema_update_interval": 1,
        "ema_decay_schedule": "constant",
        "enable_gradient_checkpointing": True,
        "seed": 17,
        "shuffle_samples": False,
        "latent_storage_dtype": "float32",
        "kl_beta": 0.01,
        "kl_type": "v-based",
        "global_std": False,
        "advantage_aggregation": "sum",
    }
    if algorithm == "grpo-guard":
        # Keep the historical recipe; deterministic CPU controls own both branches.
        train["clip_range"] = (-1.5e-3, 1.5e-3)
    else:
        train.update(kl_mask_type="x-based", kl_mask_threshold=0.01)
    return Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "parallelism": {
                "dp_shard_size": options["dp_size"],
                "cp_size": options["cp_size"],
                "tp_size": options["tp_size"],
                "sequence_parallel": False,
            },
            "model": {
                "model_type": "flux1",
                "model_name_or_path": str(root / "model"),
                "target_components": ["transformer"],
                "finetune_type": mode,
                "target_modules": "all" if mode == "full" else "default",
                "component_load_dtypes": "float32",
                "lora_rank": 2,
                "lora_alpha": 2,
                **(
                    {"resume_type": "state", "resume_path": options["resume_from"]}
                    if options.get("resume_from")
                    else {}
                ),
            },
            "data": {
                "datasets": [
                    {"name": "tiny", "dataset_dir": str(root / "prompts"), "train": {"weight": 1}}
                ],
                "cache_dir": str(root / "online-cache"),
                "dataloader_num_workers": 0,
                "preprocessing_batch_size": 1,
                "sampler_type": {"ranklocal": "rank_local", "subgroup": "subgroup_tile"}.get(
                    options.get("sampler_type"), options.get("sampler_type", "rank_local")
                ),
                "sampler_subgroup_size": options.get("subgroup_size"),
                "enable_preprocess": True,
            },
            "train": train,
            "log": {
                "save_model_only": False,
                "save_freq": 1,
                "save_dir": str(Path(result_dir) / "saves"),
                "run_name": "flux-coupled-overlap",
                "logging_backend": "none",
                "verbose": False,
            },
            "eval": {"eval_freq": 0},
            "scheduler": {
                "dynamics_type": "Flow-SDE",
                "sde_steps": [0, 1],
                "num_sde_steps": 2,
                "seed": 17,
            },
            "rewards": [
                {
                    "name": "image",
                    "reward_model": "flux_online_reward.ImageReward",
                    "device": "cpu",
                    "batch_size": 1,
                    "async_reward": True,
                }
            ],
            "acceleration": {
                "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
            },
            "optimizers": [
                {
                    "name": "default",
                    "learning_rate": 1e-3,
                    "eps": 1e-4,
                    "max_grad_norm": 1.0,
                }
            ],
        }
    )


reward_media_field = "image"


def score_media(media):
    """Score actual decoded pixels in the gated training reward."""
    import torch

    assert media.shape == (3, 64, 64) and torch.isfinite(media).all()
    return float(media[0, :32].double().mean() - media[2, 32:].double().mean())


def reference_media_score(media):
    """Independently sum original pixel values for the feedback oracle."""
    red = media[0, :32].reshape(-1).tolist()
    blue = media[2, 32:].reshape(-1).tolist()
    return math.fsum(red) / len(red) - math.fsum(blue) / len(blue)
