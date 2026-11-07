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

"""Native Qwen-2.1 ordered-reference Guard lifecycle and cache/replay observations."""

import hashlib
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import qwen_coupled_fixture as qwen
import torch
from context_parallel_overlap_runner import _audit_fresh_replay

wrap_classes = ("QwenImage21TransformerBlock",)
reward_media_field = "image"
observe_objectives = qwen.observe_objectives
assert_objectives = qwen.assert_objectives
audit_kind = "coupled_replay"
audit_steps = 2


def build_fixture(root, tp_size):
    assert tp_size == 2
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("qwen21_cp_fixture.py")), str(root)],
        check=True,
    )


def dataset_record(root, prompt, index):
    return {
        "prompt": prompt,
        "images": [
            str((Path(root) / "data" / f"{index % 8}-{reference}.png").resolve())
            for reference in (0, 1)
        ],
    }


def make_config(root, result_dir, options):
    config = qwen.make_config(root, result_dir, options)
    config.model_args.model_type = "qwen-image-2.1"
    config.training_args.extra_kwargs.update(
        condition_image_size=[32, 64], use_kv_cache=options["use_kv_cache"]
    )
    config.eval_args.extra_kwargs.update(
        condition_image_size=[32, 64], use_kv_cache=options["use_kv_cache"]
    )
    return config


def score_media(media):
    assert media.shape[0] in (3, 4) and media.shape[1:] == (64, 64)
    assert torch.isfinite(media).all()
    return float(media[0, :32].double().mean() - media[2, 32:].double().mean())


def reference_media_score(media):
    red, blue = media[0, :32].reshape(-1).tolist(), media[2, 32:].reshape(-1).tolist()
    return math.fsum(red) / len(red) - math.fsum(blue) / len(blue)


def audit_samples(trainer, samples):
    for sample in samples:
        assert len(sample.condition_images) == 2
        assert len(sample.condition_img_shapes) == 2
        assert sample.condition_img_shapes[0] != sample.condition_img_shapes[1]
        assert torch.isfinite(sample.condition_image_latents).all()
        assert torch.count_nonzero(sample.condition_image_latents)
        assert sample.use_kv_cache is trainer.training_args.extra_kwargs["use_kv_cache"]
    return _audit_fresh_replay(trainer, samples)


@contextmanager
def observe_training_terms(trainer, terms):
    original = trainer.adapter.forward
    calls = {}

    @wraps(original)
    def forward(*args, **kwargs):
        references = kwargs["condition_image_latents"]
        if isinstance(references, list):
            references = torch.cat(references, dim=0)
        assert references.shape[-1] == 64 and torch.isfinite(references).all()
        digest = hashlib.sha256(references.detach().float().cpu().numpy().tobytes()).hexdigest()
        mode = kwargs.get("use_kv_cache", True)
        calls[(digest, mode)] = calls.get((digest, mode), 0) + 1
        return original(*args, **kwargs)

    trainer.adapter.forward = forward
    try:
        with qwen.observe_training_terms(trainer, terms):
            yield
    finally:
        trainer.adapter.forward = original
        assert calls
        print(
            json.dumps(
                {
                    "qwen21_reference_forwards": [
                        {"sha256": digest, "use_kv_cache": mode, "calls": count}
                        for (digest, mode), count in sorted(calls.items())
                    ]
                }
            ),
            flush=True,
        )
