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

"""Native image/video preference recipes sharing the existing pipeline fixtures."""

import json
import subprocess
import sys
from pathlib import Path


def build_fixture(root, model, tp_size):
    """Pair distinct native demonstration records without changing model construction."""
    script = "flux_sft_fixture.py" if model == "flux1" else "wan_sft_fixture.py"
    command = [sys.executable, str(Path(__file__).with_name(script)), str(root)]
    if model != "flux1":
        command.append(str(2 * tp_size))
    subprocess.run(command, check=True)
    path = root / "data/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    targets = [row["supervision"]["target"] for row in rows]
    for i, row in enumerate(rows):
        row["supervision"] = {
            "type": "preference",
            "chosen": targets[i],
            "rejected": targets[(i + 1) % len(rows)],
        }
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_config(root, directory, options):
    """Exercise actual offline acquisition with two Monte Carlo terms and GAS=2."""
    from flow_factory.hparams import Arguments

    flux = options["model"] == "flux1"
    mode = options["mode"]
    return Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "parallelism": {
                "dp_shard_size": 2,
                "cp_size": options["cp_size"],
                "tp_size": options["tp_size"],
                "sequence_parallel": options["tp_size"] > 1,
            },
            "model": {
                "model_type": options["model"],
                "model_name_or_path": str(root / "model"),
                "target_components": ["transformer"],
                "target_modules": "all" if mode == "full" else "default",
                "finetune_type": mode,
                "component_load_dtypes": "float32",
                "lora_rank": 2,
                "lora_alpha": 2,
                **({} if flux else {"compute_precision_policy": "cross_attention_fp32_v1"}),
                **(
                    {"resume_path": options["resume"], "resume_type": "state"}
                    if options.get("resume")
                    else {}
                ),
            },
            "data": {
                "datasets": [
                    {"name": "pairs", "dataset_dir": str(root / "data"), "train": {"weight": 1}}
                ],
                "cache_dir": str(root / "cache"),
                "dataloader_num_workers": 0,
                "preprocessing_batch_size": 1,
                "enable_preprocess": True,
            },
            "train": {
                "trainer_type": "offline-dpo",
                "beta": 2.0,
                "max_epochs": options["epochs"],
                "gradient_accumulation_steps": 2,
                "num_train_timesteps": 2,
                "weighting_scheme": "uniform",
                "ema_decay": 0.9,
                "ema_update_interval": 1,
                "ema_decay_schedule": "constant",
                "resolution": [64, 64] if flux else [32, 32],
                **({} if flux else {"num_frames": 5, "frame_rate": 4.0}),
                "guidance_scale": 3.5 if flux else 1.0,
                "enable_gradient_checkpointing": True,
                "seed": 17,
            },
            "log": {
                "save_model_only": False,
                "save_freq": 1,
                "save_dir": str(Path(directory) / "saves"),
                "run_name": "offline-dpo",
                "logging_backend": "none",
                "verbose": False,
            },
            "eval": {"eval_freq": 0},
            "acceleration": {
                "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
            },
            "scheduler": {"dynamics_type": "ODE"},
            "optimizers": [
                {"name": "default", "learning_rate": 1e-3, "max_grad_norm": 1.0, "eps": 1e-4}
            ],
        }
    )
