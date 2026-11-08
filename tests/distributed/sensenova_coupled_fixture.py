# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""SenseNova Guard lifecycle with native ordered-reference prefixes and both flow heads."""

import json
import subprocess
import sys
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import qwen_coupled_fixture as qwen
from context_parallel_overlap_runner import _audit_fresh_replay

wrap_classes = ("Qwen3DecoderLayer", "NEOVisionModel")
reward_media_field = "image"
audit_kind = "coupled_replay"
audit_steps = 2
observe_objectives = qwen.observe_objectives
assert_objectives = qwen.assert_objectives
score_media = qwen.score_media
reference_media_score = qwen.reference_media_score


def build_native_fixture(root, tp_size, *, pixel_head):
    assert tp_size == 2
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("sensenova_cp_fixture.py")),
            str(root),
            "u15" if pixel_head else "u10",
        ],
        check=True,
    )


def dataset_record(root, prompt, index):
    return {
        "prompt": prompt,
        "images": [
            str((Path(root) / "data" / f"{index%8}-{reference}.png").resolve())
            for reference in (0, 1)
        ],
    }


def make_config(root, result_dir, options):
    config = qwen.make_config(root, result_dir, options)
    config.model_args.model_type = "sensenova"
    config.training_args.extra_kwargs.pop("negative_prompt", None)
    config.training_args.guidance_scale = 3.0
    config.eval_args.guidance_scale = 3.0
    config.training_args.extra_kwargs.update(
        image_guidance_scale=1.5,
        cfg_norm="global",
        cfg_interval=(0.0, 1.0),
        timestep_shift=1.0,
        enable_timestep_shift=False,
    )
    config.eval_args.extra_kwargs.update(config.training_args.extra_kwargs)
    assert config.training_args.per_device_batch_size == 1
    return config


def audit_samples(trainer, samples):
    for sample in samples:
        assert [image.size for image in sample.condition_images] == [(32, 32), (64, 32)]
    return _audit_fresh_replay(trainer, samples)


@contextmanager
def observe_training_terms(trainer, terms):
    original = trainer.adapter._build_context
    observations = []

    @wraps(original)
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        values = []
        for prefix in ("", "img_", "uncond_"):
            cache = result[prefix + "past_key_values"]
            assert cache is not None
            assert all(
                not layer.keys.requires_grad and not layer.values.requires_grad
                for layer in cache.layers
            )
            values.append(
                {
                    "prefix_length": cache.get_seq_length(),
                    "kv_heads": cache.layers[0].keys.shape[1],
                    "head_dim": cache.layers[0].keys.shape[-1],
                }
            )
        assert values[0]["prefix_length"] > values[1]["prefix_length"] > values[2]["prefix_length"]
        observations.append(values)
        return result

    trainer.adapter._build_context = observed
    try:
        with qwen.observe_training_terms(trainer, terms):
            yield
    finally:
        trainer.adapter._build_context = original
        assert observations
        print(json.dumps({"sensenova_native_prefixes": observations}), flush=True)
