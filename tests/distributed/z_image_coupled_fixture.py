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

"""Z-Image Guard lifecycle through native CFG, overlap and exact continuation."""

import json
import subprocess
import sys
from pathlib import Path

import flux_coupled_fixture as flux
import torch
from context_parallel_overlap_runner import _audit_fresh_replay

wrap_classes = ("ZImageTransformerBlock",)
reward_media_field = "image"
score_media = flux.score_media
reference_media_score = flux.reference_media_score
observe_objectives = flux.observe_objectives
assert_objectives = flux.assert_objectives
observe_training_terms = flux.observe_training_terms
audit_kind = "coupled_replay"
audit_steps = 2


def build_fixture(root, tp_size):
    assert tp_size == 2
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("z_image_cp_fixture.py")), str(root)],
        check=True,
    )
    prompts = Path(root) / "prompts"
    prompts.mkdir()
    (prompts / "train.jsonl").write_text(
        "".join(
            json.dumps({"prompt": f"a {'red' if i % 2 else 'blue'} image"}) + "\n" for i in range(8)
        )
    )


def make_config(root, result_dir, options):
    assert options["algorithm"] == "grpo-guard" and options["tp_size"] == 2
    config = flux.make_config(root, result_dir, options)
    config.model_args.model_type = "z-image"
    config.parallelism_args.sequence_parallel = True
    return config


def audit_samples(trainer, samples):
    for sample in samples:
        assert sample.prompt_embeds.ndim == 2 and sample.prompt_embeds.shape[-1] == 32
        assert torch.isfinite(sample.prompt_embeds).all()
        assert sample.negative_prompt_embeds is not None
    return _audit_fresh_replay(trainer, samples)
