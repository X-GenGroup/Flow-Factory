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

"""Flux3 native guidance/CFG and real reference-image overlap recipes."""

import hashlib
import subprocess
import sys
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import flux_coupled_fixture as flux
import torch
from context_parallel_overlap_runner import _audit_fresh_replay
from flux3_cp_fixture import wrap_classes

reward_media_field = flux.reward_media_field
score_media = flux.score_media
reference_media_score = flux.reference_media_score
observe_objectives = flux.observe_objectives
assert_objectives = flux.assert_objectives
audit_kind = "coupled_replay"
audit_steps = 2


def build_variant_fixture(root, tp_size, variant):
    assert tp_size == 2
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("flux3_cp_fixture.py")), str(root), variant],
        check=True,
    )


def dataset_record(root, prompt, index):
    return {"prompt": prompt, "images": [str((Path(root) / "data" / f"{index%8}.png").resolve())]}


def make_config(root, result_dir, options):
    assert options["algorithm"] in ("grpo-guard", "dppo") and options["tp_size"] == 2
    config = flux.make_config(root, result_dir, options)
    config.model_args.model_type = "flux3"
    config.parallelism_args.sequence_parallel = True
    config.training_args.guidance_scale = 2.0 if options["variant"] == "base" else 1.0
    conditions = {"condition_image_size": [32, 64], "max_sequence_length": 80}
    if options["variant"] == "base":
        conditions["negative_prompt"] = "blue"
    if options["variant"] == "distilled":
        config.training_args.extra_kwargs.pop("negative_prompt", None)
        config.eval_args.extra_kwargs.pop("negative_prompt", None)
    config.training_args.extra_kwargs.update(conditions)
    config.eval_args.extra_kwargs.update(conditions)
    return config


def audit_samples(trainer, samples):
    for sample in samples:
        assert sample.image_latents.shape == (8, 128)
        assert sample.latent_ids.shape == (16, 4) and sample.image_latent_ids.shape == (8, 4)
        assert sample.prompt_embeds.shape == (80, 64) and sample.text_ids.shape == (80, 4)
        assert torch.all(sample.latent_ids[:, 0] == 0)
        assert torch.all(sample.image_latent_ids[:, 3] < 0)
        assert torch.isfinite(sample.image_latents).all() and torch.count_nonzero(
            sample.image_latents
        )
        assert len(sample.condition_images) == 1 and sample.condition_images[0].shape == (3, 32, 64)
        if not trainer.adapter.is_distilled:
            assert sample.negative_prompt_embeds.shape == (80, 64)
    return _audit_fresh_replay(trainer, samples)


@contextmanager
def observe_training_terms(trainer, terms):
    """Check reference operands on the actual optimizer forward, preserving its signature."""
    original = trainer.adapter.forward
    references = set()

    @wraps(original)
    def forward(*args, **kwargs):
        target, reference = kwargs["latents"], kwargs["image_latents"]
        assert target.shape == (1, 16, 128) and reference.shape == (1, 8, 128)
        assert kwargs["latent_ids"].shape == (1, 16, 4)
        assert kwargs["image_latent_ids"].shape == (1, 8, 4)
        assert torch.isfinite(reference).all() and torch.count_nonzero(reference)
        references.add(
            hashlib.sha256(reference.detach().float().cpu().numpy().tobytes()).hexdigest()
        )
        return original(*args, **kwargs)

    trainer.adapter.forward = forward
    try:
        with flux.observe_training_terms(trainer, terms):
            yield
    finally:
        trainer.adapter.forward = original
        assert references, "Expected real reference-image optimizer forwards"
