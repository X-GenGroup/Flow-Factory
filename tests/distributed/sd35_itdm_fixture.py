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

"""Explicit tiny-SD3.5 iTDM recipe for the shared online qualification cases."""

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

wrap_classes = ("JointTransformerBlock",)
checkpoint_run_name = "sd35-itdm"


def build_fixture(root, tp_size):
    """Build native SD3.5 components in a child process and write fixed rollout rows."""
    assert tp_size in (1, 2), "The four-head fixture covers TP1/TP2 with CP2"
    root = Path(root)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("sd35_cp_fixture.py")), str(root), "large"],
        check=True,
    )
    prompts = root / "prompts"
    prompts.mkdir()
    (prompts / "train.jsonl").write_text(
        "".join(
            json.dumps({"prompt": "a red image" if i % 2 else "a blue image"}) + "\n"
            for i in range(8)
        )
    )


def make_config(root, result_dir, options):
    """Bind the iTDM objective to SD3.5 LoRA; the common runner owns CP controls."""
    from flow_factory.hparams import Arguments

    assert options["algorithm"] == "itdm", options["algorithm"]
    assert options["mode"] == "lora", "iTDM requires an sd3-5 LoRA model"
    root = Path(root)
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
                "model_type": "sd3-5",
                "model_name_or_path": str(root / "model"),
                "target_components": ["transformer"],
                "finetune_type": "lora",
                "target_modules": "default",
                "component_load_dtypes": "float32",
                "lora_rank": 2,
                "lora_alpha": 2,
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
            "train": {
                "trainer_type": "itdm",
                "max_epochs": options.get("epochs", 2),
                "per_device_batch_size": 1,
                "resolution": [64, 64],
                "num_inference_steps": 3,
                "guidance_scale": options.get("guidance_scale", 1.0),
                "group_size": 2,
                "unique_sample_num_per_epoch": options.get("dp_size", 2),
                "gradient_step_per_epoch": 1,
                "gradient_accumulation_steps": "auto",
                "num_inner_epochs": 1,
                "num_train_timesteps": 2,
                "max_timestep_index": 1,
                "ttur": options.get("ttur", 1),
                "kl_beta": 0.01,
                "advantage_clip_range": 5.0,
                "advantage_aggregation": "sum",
                "global_std": False,
                "ema_decay": 0.9,
                "ema_update_interval": 1,
                "ema_decay_schedule": "constant",
                "enable_gradient_checkpointing": True,
                "seed": 17,
                "shuffle_samples": False,
                "latent_storage_dtype": "float32",
            },
            "log": {
                "save_model_only": False,
                "save_freq": 1,
                "save_dir": str(Path(result_dir) / "saves"),
                "run_name": checkpoint_run_name,
                "logging_backend": "none",
                "verbose": False,
            },
            "eval": {"eval_freq": 0},
            # The shipped recipes rely on DPM++ 2M; the rollout identity it
            # establishes is exactly what CP has to preserve.
            "scheduler": {"dynamics_type": "ODE", "solver": "dpmpp_2m", "seed": 17},
            "rewards": [
                {
                    "name": "member",
                    "reward_model": "wan_online_reward.MemberReward",
                    "device": "cpu",
                    "batch_size": 2,
                    "async_reward": True,
                }
            ],
            "acceleration": {
                "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
            },
            "optimizers": [
                {
                    "name": role,
                    "learning_rate": 1e-3,
                    "eps": 1e-4,
                    "max_grad_norm": 1.0,
                    "update_frequency": 1,
                }
                for role in ("generator", "fake")
            ],
        }
    )


@contextmanager
def observe_objectives(trainer):
    """iTDM's revision target is verified by the CPU oracle; CP observes updates."""
    del trainer
    yield []


def assert_objectives(records, cp_size):
    del cp_size
    assert not records, "iTDM declares no CP-specific objective observations"


def assert_feedback(trainer, records, rank_local_feedback, cycles):
    """iTDM scores terminal samples per cycle and replays no log probabilities."""
    from collections import Counter

    from wan_online_reward import MemberReward

    assert trainer.reward_models, "iTDM requires a training reward"
    assert Counter(MemberReward.scored) == Counter(rank_local_feedback)
    assert len(records["feedback"]) == cycles
    # `trajectory_indices=[-1]` with `compute_log_prob=False` keeps the rollout
    # endpoint-only, so no transition carries a replayed log probability.
    assert not records["replay_log_probs"]
    assert Counter(update["role"] for update in records["updates"]) == {
        "generator": cycles // trainer.training_args.ttur,
        "fake": cycles,
    }
