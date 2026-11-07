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

"""Edit2511 ordered references with Guard, VL/VAE preprocessing and exact continuation."""

from dataclasses import fields
from pathlib import Path

import qwen_coupled_fixture as qwen
from qwen_edit_cp_fixture import dataset_record, inactive_parameter_names, wrap_classes

from flow_factory.hparams import Arguments

observe_objectives = qwen.observe_objectives
assert_objectives = qwen.assert_objectives
observe_training_terms = qwen.observe_training_terms
reward_media_field = "image"
audit_kind = "coupled_replay"
audit_steps = 2
score_media = qwen.score_media
reference_media_score = qwen.reference_media_score


def build_fixture(root, tp_size):
    """Create a genuine editing pipeline and both ordered input images."""
    from qwen_edit_cp_fixture import build_fixture as build

    assert tp_size == 2
    build(root, zero_cond_t=True)


def make_config(root, result_dir, options):
    """Use the shared objective with native edit conditions in training and evaluation."""
    assert options["algorithm"] == "grpo-guard" and options["tp_size"] == 2
    base = qwen.make_config(root, result_dir, {**options, "sequence_parallel": True})
    parallelism = base.parallelism_args.to_dict()
    parallelism.update(tp_size=2, sequence_parallel=True)
    model = base.model_args.to_dict()
    model.update(model_type="qwen-image-edit-plus", model_name_or_path=str(Path(root) / "model"))
    init_fields = {field.name for field in fields(base.training_args) if field.init}
    train = {
        key: value for key, value in base.training_args.to_dict().items() if key in init_fields
    }
    train["extra_kwargs"] = {
        **base.training_args.extra_kwargs,
        "condition_image_size": 64,
        "max_sequence_length": 512,
        "negative_prompt": "blue",
    }
    evaluation = base.eval_args.to_dict()
    evaluation.update(condition_image_size=64, max_sequence_length=512, negative_prompt="blue")
    return Arguments.from_dict(
        {
            "mixed_precision": base.mixed_precision,
            "parallelism": parallelism,
            "model": model,
            "data": base.data_args.to_dict(),
            "train": train,
            "scheduler": base.scheduler_args.to_dict(),
            "log": base.log_args.to_dict(),
            "eval": evaluation,
            "acceleration": base.acceleration_args.to_dict(),
            "optimizers": [item.to_dict() for item in base.optimizer_args],
            "rewards": [item.to_dict() for item in base.reward_args],
        }
    )


def audit_samples(trainer, samples):
    """Require actual reference latents/positions before the independent replay audit."""
    import torch

    assert samples
    for sample in samples:
        assert len(sample.condition_images) == 2
        assert sample.img_shapes == [(1, 4, 4), (1, 4, 4), (1, 4, 4)]
        assert sample.image_latents.shape == (32, 16)
        assert torch.isfinite(sample.image_latents).all()
        assert not torch.equal(sample.image_latents[:16], sample.image_latents[16:])
        assert sample.prompt_embeds.shape[0] > 64
    return qwen.audit_samples(trainer, samples)
