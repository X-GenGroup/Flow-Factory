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

"""Qwen-Image Guard recipe for masked conditioning and shared overlap lifecycle."""

import math
import subprocess
import sys
from pathlib import Path

from context_parallel_overlap_runner import _audit_fresh_replay as audit_samples
from context_parallel_overlap_runner import observe_coupled_terms as observe_training_terms
from coupled_observations import assert_objectives, observe_coupled_objectives
from qwen_cp_fixture import inactive_parameter_names, wrap_classes

observe_objectives = observe_coupled_objectives
reward_media_field = "image"
audit_kind = "coupled_replay"
audit_steps = 2


def build_fixture(root, tp_size):
    """Build the serialized native Qwen pipeline before CUDA workers start."""
    assert tp_size == 1, "Qwen TP+CP remains a separate layout proof"
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("qwen_cp_fixture.py")), str(root)],
        check=True,
    )


def make_config(root, result_dir, options):
    """Use active CFG, frozen reference, EMA and real paired positive/negative conditions."""
    from flow_factory.hparams import Arguments

    assert options["algorithm"] == "grpo-guard"
    assert options["tp_size"] == 1
    return Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "parallelism": {
                "dp_shard_size": options["dp_size"],
                "cp_size": options["cp_size"],
                "tp_size": 1,
                "sequence_parallel": False,
            },
            "model": {
                "model_type": "qwen-image",
                "model_name_or_path": str(Path(root) / "model"),
                "target_components": ["transformer"],
                "finetune_type": options["training_mode"],
                "target_modules": "all" if options["training_mode"] == "full" else "default",
                "component_load_dtypes": "float32",
                "lora_rank": 2,
                "lora_alpha": 2,
            },
            "data": {
                "datasets": [
                    {
                        "name": "tiny",
                        "dataset_dir": str(Path(root) / "prompts"),
                        "train": {"weight": 1},
                    }
                ],
                "cache_dir": str(Path(root) / "cache"),
                "dataloader_num_workers": 0,
                "preprocessing_batch_size": 1,
                "sampler_type": "rank_local",
                "enable_preprocess": True,
            },
            "train": {
                "trainer_type": "grpo-guard",
                "max_epochs": options.get("epochs", 2),
                "per_device_batch_size": 1,
                "resolution": [64, 64],
                "num_inference_steps": 3,
                "guidance_scale": 2.0,
                "negative_prompt": "blue",
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
                "clip_range": (-1.5e-3, 1.5e-3),
            },
            "log": {
                "save_model_only": False,
                "save_freq": 1,
                "save_dir": str(Path(result_dir) / "saves"),
                "run_name": "qwen-guard-overlap",
                "logging_backend": "none",
                "verbose": False,
            },
            "eval": {
                "eval_freq": 0,
                "resolution": [64, 64],
                "num_inference_steps": 2,
                "guidance_scale": 2.0,
            },
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
                {"name": "default", "learning_rate": 1e-3, "eps": 1e-4, "max_grad_norm": 1.0}
            ],
        }
    )


def score_media(media):
    """Score decoded pixels with a deterministic nonconstant image reward."""
    import torch

    assert media.shape == (3, 64, 64) and torch.isfinite(media).all()
    return float(media[0, :32].double().mean() - media[2, 32:].double().mean())


def reference_media_score(media):
    """Check the reward by an independent scalar summation."""
    red, blue = media[0, :32].reshape(-1).tolist(), media[2, 32:].reshape(-1).tolist()
    return math.fsum(red) / len(red) - math.fsum(blue) / len(blue)


def qualify_model_control(tmp_path, root, options):
    """Cover full and LoRA gradients with the same fixed mask/position operands."""
    from context_parallel_model_control import qualify_model_control as qualify

    return [
        qualify(
            tmp_path / mode,
            root,
            {**options, "training_mode": mode, "fixture_module": "qwen_cp_fixture"},
        )
        for mode in ("full", "lora")
    ]
