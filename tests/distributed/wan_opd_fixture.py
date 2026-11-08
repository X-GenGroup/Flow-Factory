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

"""Native two-teacher Wan LoRA recipe for reward-free CP lifecycle evidence."""

import json
import subprocess
import sys
from pathlib import Path

from flow_factory.hparams import Arguments

wrap_classes = ("WanTransformerBlock",)
checkpoint_run_name = "wan-opd"
teacher_names = ("teacher_red", "teacher_blue")


def build_fixture(root, tp_size):
    """Build real model components and serialized, distinct compatible teacher LoRAs."""
    subprocess.run([sys.executable, str(Path(__file__)), str(root), str(tp_size)], check=True)


def _build(root, tp_size):
    import torch
    from peft import LoraConfig, get_peft_model
    from wan_sft_fixture import build_wan_sft_fixture

    from diffusers import WanTransformer3DModel
    from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter

    build_wan_sft_fixture(root, 2 * tp_size)
    targets = Wan2_T2V_Adapter.default_target_modules.fget(None)
    transformer = get_peft_model(
        WanTransformer3DModel.from_pretrained(str(root / "model/transformer")),
        LoraConfig(r=2, lora_alpha=2, target_modules=targets),
    )
    for index, name in enumerate(teacher_names):
        torch.manual_seed(701 + index)
        with torch.no_grad():
            for parameter_name, parameter in transformer.named_parameters():
                if "lora_" in parameter_name:
                    parameter.copy_(torch.randn_like(parameter) * (0.15 + 0.1 * index))
        transformer.save_pretrained(root / name)
        torch.save(
            [
                parameter.detach().clone()
                for parameter in transformer.parameters()
                if parameter.requires_grad
            ],
            root / (name + "-expected.pt"),
        )
        source = root / ("red" if index == 0 else "blue")
        source.mkdir()
        prompt = "a red video" if index == 0 else "a blue video"
        (source / "train.jsonl").write_text(
            "".join(json.dumps({"prompt": prompt}) + "\n" for _ in range(8))
        )
        (source / "test.jsonl").write_text(json.dumps({"prompt": prompt}) + "\n")


def make_config(root, result_dir, options):
    """Declare the complete reward-free recipe without another algorithm's defaults."""
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
                "finetune_type": "lora",
                "target_modules": "default",
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
                    {
                        "name": source,
                        "dataset_dir": str(root / source),
                        "train": {"weight": 1},
                        "eval": {},
                    }
                    for source in ("red", "blue")
                ],
                "cache_dir": str(root / "opd-cache"),
                "dataloader_num_workers": 0,
                "preprocessing_batch_size": 1,
                "sampler_type": "rank_local",
                "enable_preprocess": True,
            },
            "train": {
                "trainer_type": "diffusion-opd",
                "max_epochs": options.get("epochs", 2),
                "per_device_batch_size": 1,
                "resolution": [32, 32],
                "num_frames": 5,
                "frame_rate": 4.0,
                "num_inference_steps": 3,
                "guidance_scale": 1.0,
                "group_size": 1,
                "unique_sample_num_per_epoch": 4,
                "gradient_step_per_epoch": 1,
                "gradient_accumulation_steps": "auto",
                "ema_decay": 0.9,
                "ema_update_interval": 1,
                "ema_decay_schedule": "constant",
                "enable_gradient_checkpointing": True,
                "seed": 17,
                "shuffle_samples": False,
                "latent_storage_dtype": "float32",
                "loss_target": "xt",
                "self_normalize": True,
                "timestep_range": (0.0, 0.99),
                "teacher_param_device": "cpu",
                "teachers": [
                    {
                        "name": name,
                        "path": str(root / name),
                        "applicable_datasets": [source],
                        "guidance_scale": scale,
                    }
                    for name, source, scale in zip(teacher_names, ("red", "blue"), (2.0, 3.0))
                ],
            },
            "scheduler": {"dynamics_type": "ODE", "noise_level": 0.0},
            "log": {
                "save_model_only": False,
                "save_freq": 1,
                "save_dir": str(Path(result_dir) / "saves"),
                "run_name": checkpoint_run_name,
                "logging_backend": "none",
                "verbose": False,
            },
            "eval": {
                "eval_freq": 1,
                "resolution": [32, 32],
                "num_frames": 5,
                "frame_rate": 4.0,
                "num_inference_steps": 3,
                "guidance_scale": 1.0,
                "per_device_batch_size": 1,
                "seed": 0,
            },
            "acceleration": {
                "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
            },
            "optimizers": [
                {"name": "default", "learning_rate": 1e-3, "eps": 1e-4, "max_grad_norm": 1.0}
            ],
        }
    )


if __name__ == "__main__":
    _build(Path(sys.argv[1]), int(sys.argv[2]))
