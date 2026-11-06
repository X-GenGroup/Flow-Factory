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

"""Explicit tiny-Wan recipe for the existing online qualification cases."""

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

wrap_classes = ("WanTransformerBlock",)
checkpoint_run_name = "wan-online"


def build_fixture(root, tp_size):
    """Build native components in a child process and write fixed rollout rows."""
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("wan_sft_fixture.py")),
            str(root),
            str(2 * tp_size),
        ],
        check=True,
    )
    prompts = root / "prompts"
    prompts.mkdir()
    (prompts / "train.jsonl").write_text(
        "".join(
            json.dumps({"prompt": "a red video" if i % 2 else "a blue video"}) + "\n"
            for i in range(8)
        )
    )


def make_config(root, result_dir, options):
    from flow_factory.hparams import Arguments

    algorithm = options["algorithm"]
    roles = ("generator", "fake") if algorithm == "dmd2" else ("default",)
    train = {
        "trainer_type": algorithm,
        "max_epochs": options.get("epochs", 2),
        "per_device_batch_size": 1,
        "resolution": [32, 32],
        "num_frames": 5,
        "frame_rate": 4.0,
        "num_inference_steps": 3,
        "guidance_scale": options.get("guidance_scale", 1.0),
        "group_size": 1 if algorithm == "dmd2" else 2,
        "unique_sample_num_per_epoch": (4 if algorithm == "dmd2" else options.get("dp_size", 2)),
        "gradient_step_per_epoch": 1,
        "gradient_accumulation_steps": 2 if algorithm == "dmd2" else "auto",
        "ema_decay": 0.9,
        "ema_update_interval": 1,
        "ema_decay_schedule": "constant",
        "enable_gradient_checkpointing": True,
        "seed": 17,
        "shuffle_samples": False,
        "latent_storage_dtype": "float32",
    }
    if algorithm == "dmd2":
        train.update(real_guidance_scale=1.0, replay_rtol=0.0, replay_atol=0.0)
    else:
        train.update(kl_beta=0.01, kl_type="v-based", global_std=False, advantage_aggregation="sum")
    return Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "parallelism": {
                "dp_shard_size": options.get("dp_size", 2),
                "cp_size": options["cp_size"],
                "tp_size": options["tp_size"],
                "sequence_parallel": options["tp_size"] > 1,
            },
            "model": {
                "model_type": "wan2_t2v",
                "model_name_or_path": str(root / "model"),
                "target_components": ["transformer"],
                "finetune_type": options["mode"],
                "target_modules": "all" if options["mode"] == "full" else "default",
                "component_load_dtypes": "float32",
                "lora_rank": 2,
                "lora_alpha": 2,
                "compute_precision_policy": "cross_attention_fp32_v1",
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
                "cache_dir": str(root / "online-cache"),
                "dataloader_num_workers": 0,
                "preprocessing_batch_size": 1,
                "sampler_type": options.get("sampler_type", "auto"),
                "sampler_subgroup_size": options.get("subgroup_size"),
                "enable_preprocess": True,
            },
            "train": train,
            "log": {
                "save_model_only": False,
                "save_freq": 1,
                "save_dir": str(Path(result_dir) / "saves"),
                "run_name": "wan-online",
                "logging_backend": "none",
                "verbose": False,
            },
            "eval": {"eval_freq": 0},
            "scheduler": {
                "dynamics_type": "ODE" if algorithm == "dmd2" else "Flow-SDE",
                "sde_steps": [0, 1],
                "num_sde_steps": 2,
                "seed": 17,
            },
            "rewards": (
                []
                if algorithm == "dmd2"
                else [
                    {
                        "name": "member",
                        "reward_model": "wan_online_reward.MemberReward",
                        "device": "cpu",
                        "batch_size": 2,
                        "async_reward": True,
                    }
                ]
            ),
            "acceleration": {
                "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
            },
            "optimizers": [
                {
                    "name": role,
                    "learning_rate": 1e-3,
                    "eps": 1e-4,
                    "max_grad_norm": 1.0,
                    "update_frequency": 2 if role == "fake" else 1,
                }
                for role in roles
            ],
        }
    )


@contextmanager
def observe_objectives(trainer):
    """Select only the objective observer used by this concrete Wan recipe."""
    if trainer.training_args.trainer_type == "dmd2":
        from online_observations import observe_dmd2_objectives

        with observe_dmd2_objectives(trainer) as records:
            yield records
        assert records, "DMD2 must retain actual objective derivative observations"
    else:
        yield []


def assert_objectives(records, cp_size):
    if records:
        from dmd2_objective_reference import objective_negative_controls, objective_reference

        for event in records:
            check = objective_reference(event, cp_size)
            assert check["latent_pass"] and check["root_pass"]
            assert all(objective_negative_controls(event, cp_size).values())
