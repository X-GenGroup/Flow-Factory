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

"""Actual Wan image conditions with native GRPO overlap, evaluation and recovery."""

import math
from dataclasses import fields
from pathlib import Path

import torch
import wan_overlap_fixture as wan

from flow_factory.hparams import Arguments

audit_kind = "coupled_replay"
audit_steps = 2
wrap_classes = wan.wrap_classes
reward_media_field = "video"
observe_objectives = wan.observe_objectives
assert_objectives = wan.assert_objectives
observe_training_terms = wan.observe_training_terms


def score_media(media):
    assert media.shape[:2] == (9, 3) and media.shape[2:] in ((32, 32), (64, 64))
    return float(media.double().mean())


def reference_media_score(media):
    values = media.reshape(-1).tolist()
    return math.fsum(values) / len(values)


def dataset_record(root, prompt, index, variant):
    images = [str(Path(root) / "data" / f"{index}-first.png")]
    if variant != "i2v_expanded":
        images.append(str(Path(root) / "data" / f"{index}-last.png"))
    return {"prompt": prompt, "images": images}


def build_fixture(root, tp_size, variant):
    from wan_remaining_cp_fixture import build_fixture as build

    assert tp_size == 2
    build(root, variant)


def make_config(root, result_dir, options, variant):
    base = wan.make_config(root, result_dir, options)
    model = base.model_args.to_dict()
    model.update(
        model_type="wan2_i2v",
        target_components=(
            ["transformer", "transformer_2"] if variant.endswith("dual") else ["transformer"]
        ),
    )
    init_fields = {field.name for field in fields(base.training_args) if field.init}
    train = {
        key: value for key, value in base.training_args.to_dict().items() if key in init_fields
    }
    size = 64 if variant == "i2v_expanded" else 32
    train.update(num_frames=9, guidance_scale=2.0, resolution=[size, size], height=size, width=size)
    train["extra_kwargs"] = {
        "guidance_scale_2": 3.0,
        "negative_prompt": "blue",
        "max_sequence_length": 512,
    }
    evaluation = base.eval_args.to_dict()
    evaluation.update(
        num_frames=9,
        guidance_scale=2.0,
        guidance_scale_2=3.0,
        negative_prompt="blue",
        max_sequence_length=512,
    )
    return Arguments.from_dict(
        {
            "mixed_precision": base.mixed_precision,
            "parallelism": base.parallelism_args.to_dict(),
            "model": model,
            "data": base.data_args.to_dict(),
            "train": train,
            "eval": evaluation,
            "scheduler": base.scheduler_args.to_dict(),
            "log": base.log_args.to_dict(),
            "acceleration": base.acceleration_args.to_dict(),
            "optimizers": [item.to_dict() for item in base.optimizer_args],
            "rewards": [item.to_dict() for item in base.reward_args],
        }
    )


def audit_samples(trainer, samples, variant):
    assert samples
    for sample in samples:
        assert sample.prompt_embeds.shape == (512, 16)
        assert sample.negative_prompt_embeds.shape == (512, 16)
        assert len(sample.condition_images) == (1 if variant == "i2v_expanded" else 2)
        assert torch.isfinite(sample.latent_condition).all() and torch.count_nonzero(
            sample.latent_condition
        )
        if variant == "i2v_expanded":
            assert sample.latent_condition.shape == (48, 1, 4, 4)
            assert sample.first_frame_mask.shape == (1, 3, 4, 4)
            assert not torch.count_nonzero(sample.first_frame_mask[:, :1])
            assert torch.all(sample.first_frame_mask[:, 1:] == 1)
        else:
            assert sample.latent_condition.shape == (20, 3, 4, 4)
            assert not torch.equal(sample.condition_images[0], sample.condition_images[1])
        if variant == "flf2v":
            assert sample.image_embeds.shape == (2, 257, 16)
            assert not torch.equal(sample.image_embeds[0], sample.image_embeds[1])
        if variant == "i2v_dual":
            boundary = (
                trainer.adapter.pipeline.config.boundary_ratio
                * trainer.adapter.scheduler.config.num_train_timesteps
            )
            times = sample.timesteps[trainer.adapter.get_train_step_indices()]
            assert torch.any(times >= boundary) and torch.any(times < boundary), times
    return wan.audit_samples(trainer, samples)
