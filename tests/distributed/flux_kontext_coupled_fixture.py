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

"""Kontext reference-image recipe for the shared ordered Guard lifecycle."""

import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import flux_coupled_fixture as flux
import torch
from context_parallel_overlap_runner import _audit_fresh_replay

wrap_classes = flux.wrap_classes
reward_media_field = flux.reward_media_field
score_media = flux.score_media
reference_media_score = flux.reference_media_score
observe_objectives = flux.observe_objectives
assert_objectives = flux.assert_objectives
audit_kind = "coupled_replay"
audit_steps = 2


def build_fixture(root, tp_size):
    assert tp_size == 2
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("flux_kontext_cp_fixture.py")), str(root)],
        check=True,
    )


def dataset_record(root, prompt, index):
    """Use a genuine encoded reference image for every training and evaluation row."""
    return {"prompt": prompt, "images": [str((Path(root) / "data" / f"{index % 8}.png").resolve())]}


def make_config(root, result_dir, options):
    assert options["algorithm"] == "grpo-guard" and options["tp_size"] == 2
    config = flux.make_config(root, result_dir, options)
    config.model_args.model_type = "flux1-kontext"
    config.training_args.extra_kwargs["condition_image_size"] = [32, 64]
    config.eval_args.extra_kwargs["condition_image_size"] = [32, 64]
    return config


def audit_samples(trainer, samples):
    for sample in samples:
        assert sample.image_latents.shape == (8, 16)
        assert sample.latent_ids.shape == (24, 3)
        assert torch.all(sample.latent_ids[:16, 0] == 0)
        assert torch.all(sample.latent_ids[16:, 0] == 1)
        assert torch.isfinite(sample.image_latents).all() and torch.count_nonzero(
            sample.image_latents
        )
        assert len(sample.condition_images) == 1
        assert sample.condition_images[0].shape == (3, 32, 64)
    return _audit_fresh_replay(trainer, samples)


@contextmanager
def observe_training_terms(trainer, terms):
    """Retain signature filtering and inspect real reference operands on every forward."""
    original = trainer.adapter.forward
    counts = {}

    @wraps(original)
    def forward(*args, **kwargs):
        target, reference, ids = kwargs["latents"], kwargs["image_latents"], kwargs["latent_ids"]
        assert target.shape == (1, 16, 16) and reference.shape == (1, 8, 16)
        assert ids.shape == (24, 3)
        assert torch.all(ids[:16, 0] == 0) and torch.all(ids[16:, 0] == 1)
        assert torch.isfinite(reference).all() and torch.count_nonzero(reference)
        digest = hashlib.sha256(reference.detach().float().cpu().numpy().tobytes()).hexdigest()
        counts[digest] = counts.get(digest, 0) + 1
        return original(*args, **kwargs)

    trainer.adapter.forward = forward
    try:
        with flux.observe_training_terms(trainer, terms):
            yield
    finally:
        trainer.adapter.forward = original
        assert counts, "Native lifecycle must execute encoded reference-image forwards"
        print(
            json.dumps(
                {
                    "kontext_reference_rank": trainer.accelerator.process_index,
                    "forward_counts_by_reference_sha256": counts,
                }
            )
        )
