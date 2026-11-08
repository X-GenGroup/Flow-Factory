# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Native Bagel ordered-reference Guard lifecycle with its existing FSDP count guard."""

import json
import subprocess
import sys
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import qwen_coupled_fixture as qwen
from context_parallel_overlap_runner import _audit_fresh_replay

wrap_classes = ("Qwen2MoTDecoderLayer",)
reward_media_field = "image"
audit_kind = "coupled_replay"
audit_steps = 2
observe_objectives = qwen.observe_objectives
assert_objectives = qwen.assert_objectives
score_media = qwen.score_media
reference_media_score = qwen.reference_media_score


def build_fixture(root, tp_size):
    assert tp_size == 2
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("bagel_cp_fixture.py")), str(root)],
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
    config.model_args.model_type = "bagel"
    config.training_args.extra_kwargs.pop("negative_prompt", None)
    config.training_args.extra_kwargs.update(
        cfg_text_scale=2.0,
        cfg_img_scale=1.5,
        cfg_interval=(0.0, 1.0),
        cfg_renorm_min=0.1,
        cfg_renorm_type="global",
    )
    config.eval_args.extra_kwargs.update(config.training_args.extra_kwargs)
    # Both generation and replay retain one logical sample per native forward.
    # The separate model probe covers true packed B2 under TP/SP/CP.
    assert config.training_args.per_device_batch_size == 1
    return config


def audit_samples(trainer, samples):
    for sample in samples:
        assert len(sample.condition_images) == 2
        assert [image.size for image in sample.condition_images] == [(32, 32), (64, 32)]
    return _audit_fresh_replay(trainer, samples)


@contextmanager
def observe_training_terms(trainer, terms):
    original = trainer.adapter._build_gen_context
    contexts = []

    @wraps(original)
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        row = []
        for context in result:
            cache = context["past_key_values"]
            assert all(
                not value.requires_grad
                for value in (*cache.key_cache.values(), *cache.value_cache.values())
            )
            row.append(
                {
                    "prefix_lengths": context["kv_lens"],
                    "position_offsets": context["ropes"],
                    "kv_heads": next(iter(cache.key_cache.values())).shape[1],
                }
            )
        contexts.append(row)
        return result

    trainer.adapter._build_gen_context = observed
    try:
        with qwen.observe_training_terms(trainer, terms):
            yield
    finally:
        trainer.adapter._build_gen_context = original
        assert contexts and all(len(row) == 3 for row in contexts)
        print(json.dumps({"bagel_native_prefix_contexts": contexts}), flush=True)
