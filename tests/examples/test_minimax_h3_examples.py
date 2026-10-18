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

import json
import wave
from pathlib import Path

import av
import pytest
from PIL import Image

from flow_factory.hparams import Arguments
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.registry import get_model_adapter_class
from flow_factory.samples.references import canonicalize_reference_manifest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    "minimax_h3_t2va": {
        "model_type": "minimax-h3-t2va",
        "target": "transformer",
        "dataset": Path("dataset/vid_prompt"),
    },
    "minimax_h3_fl2va": {
        "model_type": "minimax-h3-fl2va",
        "target": "transformer",
        "dataset": Path("dataset/minimax_h3_fl2va"),
    },
    "minimax_h3_ref2va": {
        "model_type": "minimax-h3-ref2va",
        "target": "transformer_ref",
        "dataset": Path("dataset/minimax_h3_ref2va"),
    },
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(("directory", "expected"), EXAMPLES.items())
def test_examples_parse_through_production_config_and_registry(
    directory: str, expected: dict
) -> None:
    path = ROOT / "examples/grpo/lora" / directory / "default.yaml"
    config = Arguments.load_from_yaml(str(path))
    adapter_class = get_model_adapter_class(config.model_args.model_type)
    yaml_text = path.read_text(encoding="utf-8")

    assert config.training_args.trainer_type == "grpo"
    assert config.model_args.finetune_type == "lora"
    assert config.model_args.model_type == expected["model_type"]
    assert config.model_args.target_components == [expected["target"]]
    assert issubclass(adapter_class, BaseAdapter)
    assert BaseAdapter in adapter_class.__bases__
    assert not [
        base
        for base in adapter_class.__mro__[1:]
        if issubclass(base, BaseAdapter) and base is not BaseAdapter
    ]
    assert config.data_args.preprocessing_batch_size == 1
    assert config.training_args.per_device_batch_size == 1
    assert config.eval_args.per_device_batch_size == 1
    assert config.eval_args.num_frames == config.training_args.num_frames == 124
    assert config.eval_args.frame_rate == config.training_args.frame_rate == 24.0
    assert config.training_args.guidance_scale == 1.0
    assert config.eval_args.guidance_scale == 1.0
    assert config.scheduler_args.dynamics_type in {"Flow-SDE", "Dance-SDE", "CPS"}
    assert config.scheduler_args.sde_steps
    assert config.scheduler_args.num_sde_steps > 0
    assert max(config.scheduler_args.sde_steps) < config.training_args.num_inference_steps
    assert config.training_args.offload_samples_to_cpu is True
    assert len(config.data_args.datasets) == 1
    assert Path(config.data_args.datasets[0].dataset_dir) == expected["dataset"]
    assert len(config.optimizer_args) == 1
    assert config.optimizer_args[0].name == "default"
    if directory == "minimax_h3_fl2va":
        assert Path(config.data_args.datasets[0].image_dir) == expected["dataset"]
    assert (ROOT / expected["dataset"]).is_dir()
    assert "N + 1 states and exactly N trainable transitions" in yaml_text
    assert "B=1" in yaml_text
    assert "no CFG" in yaml_text
    if directory == "minimax_h3_t2va":
        assert "not been run with the 61 GB checkpoint" not in yaml_text
        assert [reward.reward_model for reward in config.reward_args] == [
            "clap",
            "imagebind",
        ]
        assert all(reward.applicable_datasets == ["vid_prompt"] for reward in config.reward_args)
    else:
        assert "not been run with the 61 GB checkpoint" in yaml_text
    assert "stg_scale" not in yaml_text
    assert "modality_scale" not in yaml_text
    assert "negative_prompt" not in yaml_text


@pytest.mark.parametrize(
    ("filename", "num_processes", "lora_rank", "learning_rate"),
    (
        ("debug.yaml", 1, 8, 1.0e-4),
        ("quality_720p_fsdp2.yaml", 16, 64, 3.0e-4),
    ),
)
def test_t2va_validated_variants_parse(
    filename: str,
    num_processes: int,
    lora_rank: int,
    learning_rate: float,
) -> None:
    path = ROOT / "examples/grpo/lora/minimax_h3_t2va" / filename
    config = Arguments.load_from_yaml(str(path))

    assert config.model_args.model_type == "minimax-h3-t2va"
    assert config.model_args.lora_rank == lora_rank
    assert config.num_processes == num_processes
    assert config.optimizer_args[0].name == "default"
    assert config.optimizer_args[0].learning_rate == learning_rate
    assert config.training_args.per_device_batch_size == 1
    assert config.training_args.guidance_scale == 1.0


def test_t2va_manifests_contain_prompt_only() -> None:
    """Keep the dedicated JSONL fixture used by the validated debug recipe strict."""
    for split in ("train", "test"):
        rows = _read_jsonl(ROOT / "dataset/minimax_h3_t2va" / f"{split}.jsonl")
        assert rows
        assert all(set(row) == {"prompt"} and row["prompt"] for row in rows)


def test_t2va_default_shared_text_manifests_contain_prompts() -> None:
    """The aligned default recipe uses the shared prompt-only TXT dataset."""
    for split in ("train", "test"):
        prompts = (ROOT / "dataset/vid_prompt" / f"{split}.txt").read_text(encoding="utf-8")
        lines = prompts.splitlines()
        assert lines
        assert all(prompt.strip() for prompt in lines)


def test_fl2va_manifests_preserve_one_or_two_ordered_images() -> None:
    dataset_dir = ROOT / "dataset/minimax_h3_fl2va"
    for split in ("train", "test"):
        rows = _read_jsonl(dataset_dir / f"{split}.jsonl")
        assert {len(row["images"]) for row in rows} == {1, 2}
        for row in rows:
            assert set(row) == {"prompt", "images"}
            assert all(not Path(image_path).is_absolute() for image_path in row["images"])
            assert all((dataset_dir / image_path).is_file() for image_path in row["images"])
        assert rows[1]["images"] == ["images/first.png", "images/last.png"]


def test_ref2va_manifests_are_ordered_valid_and_dataset_relative() -> None:
    dataset_dir = ROOT / "dataset/minimax_h3_ref2va"
    for split in ("train", "test"):
        rows = _read_jsonl(dataset_dir / f"{split}.jsonl")
        assert rows
        for row_index, row in enumerate(rows):
            references = row["references"]
            canonical = json.loads(canonicalize_reference_manifest(references, row_index=row_index))
            assert [entry["type"] for entry in canonical] == [entry["type"] for entry in references]
            for reference in references:
                path = Path(reference["path"])
                assert not path.is_absolute()
                assert (dataset_dir / path).is_file()
                if "audio_path" in reference:
                    audio_path = Path(reference["audio_path"])
                    assert not audio_path.is_absolute()
                    assert (dataset_dir / audio_path).is_file()
        assert [entry["type"] for entry in rows[0]["references"]] == [
            "image",
            "video",
            "audio",
        ]


def test_media_fixtures_are_real_png_mp4_and_wav() -> None:
    image_paths = [
        ROOT / "dataset/minimax_h3_fl2va/images/first.png",
        ROOT / "dataset/minimax_h3_fl2va/images/last.png",
        ROOT / "dataset/minimax_h3_ref2va/references/style.png",
    ]
    for image_path in image_paths:
        with Image.open(image_path) as image:
            assert image.format == "PNG"
            image.verify()

    video_path = ROOT / "dataset/minimax_h3_ref2va/references/motion.mp4"
    with av.open(str(video_path)) as container:
        assert container.streams.video
        assert next(container.decode(video=0)) is not None

    for audio_path in (
        ROOT / "dataset/minimax_h3_ref2va/references/ambience.wav",
        ROOT / "dataset/minimax_h3_ref2va/references/soundtrack.wav",
    ):
        with wave.open(str(audio_path), "rb") as audio:
            assert audio.getnframes() > 0
            assert audio.getframerate() > 0
