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

"""Tiny-Wan evaluation inputs and configuration, independent of online runners."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from test_context_parallel_training import _config as sft_config
from wan_online_fixture import make_config as online_config

wrap_classes = ("WanTransformerBlock",)

EVALUATION_DATASETS = (
    ("odd", 3, "red"),
    ("single", 1, "blue"),
    ("empty", 0, "gray"),
    ("divisible", 4, "green"),
)


def _evaluation_prompts(root):
    return {
        name: (
            [
                json.loads(line)["prompt"]
                for line in (root / name / "test.jsonl").read_text().splitlines()
            ]
            if count
            else (root / name / "test.txt").read_text().splitlines()
        )
        for name, count, _ in EVALUATION_DATASETS
    }


def make_config(root, result_dir, options):
    from flow_factory.hparams import DataArguments, MultiRewardArguments
    from flow_factory.hparams.training_args._base import EvaluationArguments

    factory = sft_config if options["algorithm"] == "sft" else online_config
    config = factory(
        root,
        result_dir,
        {
            **options,
            "checkpointing": True,
            "ema_decay": 0.9,
            "ema_update_interval": 1,
            "sequence_parallel": options["tp_size"] > 1,
        },
    )
    data = config.data_args.to_dict()
    data["datasets"].extend(
        [
            {
                "name": "odd",
                "dataset_dir": str(root / "odd"),
                "eval": {"guidance_scale": 3.0, "num_inference_steps": 2},
            },
            {
                "name": "single",
                "dataset_dir": str(root / "single"),
                "eval": {"resolution": [32, 48]},
            },
            {"name": "empty", "dataset_dir": str(root / "empty"), "eval": {}},
            {"name": "divisible", "dataset_dir": str(root / "divisible"), "eval": {}},
        ]
    )
    return replace(
        config,
        data_args=DataArguments.from_dict(data),
        eval_args=EvaluationArguments(
            eval_freq=1,
            resolution=[32, 32],
            num_frames=5,
            frame_rate=4.0,
            num_inference_steps=3,
            guidance_scale=1.0,
            seed=0,
            per_device_batch_size=1,
        ),
        eval_reward_args=MultiRewardArguments.from_dict(
            [
                {
                    "name": "score",
                    "reward_model": "wan_eval_reward.PromptReward",
                    "device": "cpu",
                    "batch_size": 2,
                    "async_reward": True,
                }
            ]
        ),
        log_args=replace(
            config.log_args,
            save_model_only=not options.get("state_resume", False),
            save_freq=int(options.get("state_resume", False)),
        ),
    )


def build_fixture(root, tp_size):
    """Construct native components separately, preserving every evaluation row."""
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("wan_sft_fixture.py")),
            str(root),
            str(2 * tp_size),
        ],
        check=True,
    )
    for name, count, color in (("prompts", 8, "red"), *EVALUATION_DATASETS):
        folder = root / name
        folder.mkdir()
        split = "train" if name == "prompts" else "test"
        (folder / f"{split}.{'jsonl' if count else 'txt'}").write_text(
            "".join(json.dumps({"prompt": f"a {color} video {i}"}) + "\n" for i in range(count))
        )
