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

"""Wan leaf recipe for native GRPO overlap; no central model dispatch."""

import math
import subprocess
import sys
from pathlib import Path

from flow_factory.hparams import Arguments

wrap_classes = ("WanTransformerBlock",)
reward_media_field = "video"


def build_fixture(root, tp_size):
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("wan_sft_fixture.py")),
            str(root),
            str(2 * tp_size),
        ],
        check=True,
    )


def score_media(media):
    assert media.shape == (5, 3, 32, 32)
    return float(media.double().mean())


def reference_media_score(media):
    values = media.reshape(-1).tolist()
    return math.fsum(values) / len(values)


def observe_objectives(trainer):
    from coupled_observations import observe_coupled_objectives

    return observe_coupled_objectives(trainer)


def assert_objectives(records, cp_size):
    from coupled_observations import assert_objectives as check

    return check(records, cp_size)


def make_config(root, result_dir, options):
    mode = options["training_mode"]
    return Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "parallelism": {
                "dp_shard_size": options["dp_size"],
                "cp_size": options["cp_size"],
                "tp_size": options["tp_size"],
                "sequence_parallel": options["tp_size"] > 1,
            },
            "model": {
                "model_type": "wan2_t2v",
                "model_name_or_path": str(root / "model"),
                "target_components": ["transformer"],
                "finetune_type": mode,
                "target_modules": "all" if mode == "full" else "default",
                "component_load_dtypes": "float32",
                "lora_rank": 2,
                "lora_alpha": 2,
                "compute_precision_policy": "cross_attention_fp32_v1",
            },
            "data": {
                "datasets": [
                    {"name": "tiny", "dataset_dir": str(root / "tiny"), "train": {"weight": 1}}
                ],
                "cache_dir": str(root / "overlap-cache"),
                "dataloader_num_workers": 0,
                "preprocessing_batch_size": 1,
                "sampler_type": "rank_local",
                "enable_preprocess": True,
            },
            "train": {
                "trainer_type": "grpo",
                "max_epochs": 2,
                "per_device_batch_size": 1,
                "resolution": [32, 32],
                "num_frames": 5,
                "frame_rate": 4.0,
                "num_inference_steps": 3,
                "guidance_scale": 1.0,
                "group_size": 2,
                "unique_sample_num_per_epoch": 2 * options["dp_size"],
                "gradient_step_per_epoch": 2,
                "gradient_accumulation_steps": "auto",
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
            },
            "log": {
                "save_model_only": False,
                "save_dir": str(result_dir),
                "logging_backend": "none",
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
                    "name": "score",
                    "reward_model": "wan_online_reward.MemberReward",
                    "device": "cpu",
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
