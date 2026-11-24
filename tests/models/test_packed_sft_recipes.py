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

"""Check per-record dropout mass and the released multimodal recipes."""

import json
import random
from collections import Counter

import pytest

from flow_factory.data_utils.schema import MediaAsset, NormalizedModelInput
from flow_factory.models.sft_recipes import BagelDropout, SenseNovaDropout


def record(count, prompt="caption"):
    return NormalizedModelInput(
        prompt, "negative", tuple(MediaAsset("image", str(i)) for i in range(count))
    )


@pytest.mark.parametrize("mode", ["native", 0.1, 1.0])
def test_bagel_drop_distribution_preserves_independent_segment_law(mode):
    rng = random.Random(72)
    policy = BagelDropout()
    item = record(2)
    n = 12000
    counts = Counter()
    for _ in range(n):
        mask = policy.sample(mode, item, lambda _: rng.random())
        counts.update(mask)
        counts["any"] += bool(mask)
        counts["vae_only"] += "vae:0" in mask and "vit:0" not in mask
    nonempty = 1 - 0.9 * 0.7**4
    p = nonempty if mode == "native" else mode
    assert counts["any"] / n == pytest.approx(p, abs=0.014)
    for key, q in {"text": 0.1, "vae:0": 0.3, "vit:1": 0.3, "vae_only": 0.21}.items():
        assert counts[key] / n == pytest.approx(p * q / nonempty, abs=0.014)


@pytest.mark.parametrize("reference_count", [0, 2])
@pytest.mark.parametrize("mode", ["native", 0.1, 1.0])
def test_sensenova_exclusive_task_recipe(reference_count, mode):
    policy = SenseNovaDropout()
    rng = random.Random(40)
    counts = Counter(
        policy.sample(mode, record(reference_count), lambda _: rng.random()) for _ in range(8000)
    )
    native_rate = 0.2 if reference_count else 0.1
    rate = native_rate if mode == "native" else mode
    assert counts[frozenset()] / 8000 == pytest.approx(1 - rate, abs=0.015)
    joint = rate / 2 if reference_count else 0
    assert counts[frozenset(("text", "images"))] / 8000 == pytest.approx(joint, abs=0.015)
    assert counts[frozenset(("text",))] / 8000 == pytest.approx(rate - joint, abs=0.015)


@pytest.mark.parametrize("policy", [BagelDropout(), SenseNovaDropout()])
def test_native_admission_and_no_condition(policy, tmp_path):
    model_id = policy.checkpoints[0]
    policy.validate_native(model_id)
    snapshot = tmp_path / ("models--" + model_id.replace("/", "--")) / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    policy.validate_native(str(snapshot))
    alias = tmp_path / "model-link"
    alias.symlink_to(snapshot, target_is_directory=True)
    policy.validate_native(str(alias))
    copy = tmp_path / "renamed"
    copy.mkdir()
    (copy / "config.json").write_text(json.dumps({"_name_or_path": model_id}))
    policy.validate_native(str(copy))
    with pytest.raises(ValueError, match="numeric condition_dropout"):
        policy.validate_native("custom/unknown")
    assert not policy.sample(1, record(0, prompt=""), lambda _: 0)
