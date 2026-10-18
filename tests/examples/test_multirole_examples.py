# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import yaml


def test_only_gpu_validated_multirole_examples_are_published() -> None:
    repository_root = Path(__file__).parents[2]

    dmd2_path = repository_root / "examples" / "dmd2" / "lora" / "sd3_5" / "ocr.yaml"
    dmd2_config = yaml.safe_load(dmd2_path.read_text())
    assert dmd2_config["data"]["datasets"][0]["dataset_dir"] == "dataset/ocr"
    assert dmd2_config["train"]["trainer_type"] == "dmd2"
    assert dmd2_config["train"]["num_inference_steps"] == 4
    assert dmd2_config["train"]["per_device_batch_size"] == 32
    assert dmd2_config["train"]["unique_sample_num_per_epoch"] == 512
    assert {
        optimizer["name"]: optimizer.get("update_frequency")
        for optimizer in dmd2_config["optimizers"]
    } == {"generator": None, "fake": 5}
    assert dmd2_config["train"]["replay_rtol"] == dmd2_config["train"]["replay_atol"] == 0
    assert dmd2_config["scheduler"] == {"dynamics_type": "ODE", "seed": 42}
    assert [optimizer["name"] for optimizer in dmd2_config["optimizers"]] == [
        "generator",
        "fake",
    ]

    tdm_path = repository_root / "examples" / "tdm" / "lora" / "sd3_5" / "ocr.yaml"
    config = yaml.safe_load(tdm_path.read_text())
    assert config["data"]["datasets"][0]["dataset_dir"] == "dataset/ocr"
    assert config["train"]["trainer_type"] == "tdm"
    assert config["train"]["num_inference_steps"] == 4
    assert "trajectory_steps" not in config["train"]
    assert config["train"]["tdm_importance_clip"] == 20.0
    assert config["train"]["replay_rtol"] == config["train"]["replay_atol"] == 0
    assert [optimizer["name"] for optimizer in config["optimizers"]] == [
        "generator",
        "fake",
    ]
    assert config["scheduler"] == {"dynamics_type": "ODE"}

    bagel_tdm_path = repository_root / "examples" / "tdm" / "lora" / "bagel" / "default.yaml"
    bagel_tdm_config = yaml.safe_load(bagel_tdm_path.read_text())
    assert bagel_tdm_config["train"]["trainer_type"] == "tdm"
    assert bagel_tdm_config["train"]["shuffle_samples"] is False
    assert bagel_tdm_config["train"]["per_device_batch_size"] == 2
    assert bagel_tdm_config["train"]["real_guidance_scale"] == 4.0
    assert bagel_tdm_config["scheduler"]["dynamics_type"] == "ODE"
    assert [optimizer["name"] for optimizer in bagel_tdm_config["optimizers"]] == [
        "generator",
        "fake",
    ]

    h3_tdm_path = repository_root / "examples" / "tdm" / "lora" / "minimax_h3_t2va" / "default.yaml"
    h3_tdm_text = h3_tdm_path.read_text()
    h3_tdm_config = yaml.safe_load(h3_tdm_text)
    assert "over 200 steps" in h3_tdm_text
    assert h3_tdm_config["log"]["project"] == "Flow-Factory-TDM"
    assert h3_tdm_config["log"]["save_dir"] == "saves/"
    assert h3_tdm_config["log"]["run_name"] is None
    assert h3_tdm_config["model"]["model_type"] == "minimax-h3-t2va"
    assert h3_tdm_config["model"]["model_name_or_path"] == "MiniMaxAI/MiniMax-H3"
    assert h3_tdm_config["train"]["trainer_type"] == "tdm"
    assert h3_tdm_config["train"]["num_inference_steps"] == 6
    assert h3_tdm_config["train"]["resolution"] == [512, 768]
    assert h3_tdm_config["train"]["num_frames"] == 124
    assert h3_tdm_config["train"]["per_device_batch_size"] == 1
    assert h3_tdm_config["train"]["unique_sample_num_per_epoch"] == 64
    assert h3_tdm_config["train"]["enable_gradient_checkpointing"] == {"mode": "full"}
    assert h3_tdm_config["scheduler"]["dynamics_type"] == "ODE"
    assert [optimizer["name"] for optimizer in h3_tdm_config["optimizers"]] == [
        "generator",
        "fake",
    ]

    tdm_r1_path = repository_root / "examples" / "tdm_r1" / "lora" / "sd3_5" / "ocr.yaml"
    tdm_r1_config = yaml.safe_load(tdm_r1_path.read_text())
    assert tdm_r1_config["data"]["datasets"][0]["dataset_dir"] == "dataset/ocr"
    assert tdm_r1_config["train"]["trainer_type"] == "tdm-r1"
    assert tdm_r1_config["model"]["lora_rank"] == 32
    assert tdm_r1_config["model"]["lora_alpha"] == 64
    assert tdm_r1_config["train"]["num_inference_steps"] == 4
    assert tdm_r1_config["train"]["per_device_batch_size"] == 24
    assert tdm_r1_config["train"]["group_size"] == 24
    assert tdm_r1_config["train"]["unique_sample_num_per_epoch"] == 48
    assert tdm_r1_config["train"]["gradient_accumulation_steps"] == 12
    assert tdm_r1_config["train"]["tdm_weight"] == 0.3
    assert tdm_r1_config["train"]["surrogate_preference_beta"] == 10.0
    assert tdm_r1_config["train"]["cfg_reward_scale"] == 4.5
    assert not tdm_r1_config["rewards"][0]["async_reward"]
    assert [optimizer["name"] for optimizer in tdm_r1_config["optimizers"]] == [
        "generator",
        "fake",
        "surrogate",
    ]
    assert [optimizer["learning_rate"] for optimizer in tdm_r1_config["optimizers"]] == [
        7.5e-5,
        3.0e-4,
        3.0e-4,
    ]
    assert tdm_r1_config["scheduler"] == {"dynamics_type": "ODE", "seed": 42}

    algorithms = (repository_root / "docs" / "algorithms.md").read_text()
    assert "update_frequency" in algorithms
    assert "fake first" in algorithms.lower()
    assert not (repository_root / "scripts" / "convert_ocr_prompts.py").exists()


def test_examples_set_role_cadence_only_through_optimizer_entries() -> None:
    """The removed `train.ttur_fake_updates` field now fails at parse time."""
    examples = Path(__file__).parents[2] / "examples"
    leftovers = [
        str(path.relative_to(examples))
        for path in sorted(examples.rglob("*.yaml"))
        if "ttur_fake_updates" in path.read_text()
    ]
    assert leftovers == []
