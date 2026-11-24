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

"""Published condition-dropout recipes for multimodal packed-prefix models."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_utils.schema import NormalizedModelInput


class _PackedRecipe:
    supports_native = True
    cached_empty_caption = False
    checkpoints: tuple[str, ...]

    def validate_native(self, checkpoint: str) -> None:
        """Identify a qualified HF release or a local copy bearing its release name."""
        path = Path(checkpoint)
        if checkpoint in self.checkpoints:
            return
        if path.is_dir():
            for local in (path, path.resolve()):
                for model_id in self.checkpoints:
                    if local.name == model_id.split("/")[-1] or (
                        local.parent.name == "snapshots"
                        and local.parent.parent.name == "models--" + model_id.replace("/", "--")
                    ):
                        return
            for name in ("config.json", "llm_config.json", "model_index.json"):
                config = path / name
                if (
                    config.is_file()
                    and json.loads(config.read_text()).get("_name_or_path") in self.checkpoints
                ):
                    return
        raise ValueError(
            f"condition_dropout='native' for T2I/I2I is qualified for {self.checkpoints}, "
            f"received {checkpoint!r}. For a renamed local copy, keep its original HF "
            "release name or _name_or_path metadata. Other checkpoints support numeric condition_dropout."
        )


class BagelDropout(_PackedRecipe):
    """BAGEL pretraining draws independently per caption and per input-image representation."""

    revision = "bagel-pretrain-text10-vit30-vae30-v1"
    checkpoints = ("ByteDance-Seed/BAGEL-7B-MoT",)

    def sample(
        self, mode: float | str, model_input: NormalizedModelInput, draw: Callable[[str], float]
    ) -> frozenset[str]:
        segments = []
        references = [asset for asset in model_input.media if asset.type == "image"]
        for index in range(len(references)):
            segments.extend(((f"vae:{index}", 0.3), (f"vit:{index}", 0.3)))
        if model_input.prompt:
            segments.append(("text", 0.1))
        if mode == "native":
            return frozenset(name for name, q in segments if draw(name) < q)
        if not segments or draw("record") >= mode:
            return frozenset()

        # Select the first dropped segment conditional on at least one drop, then
        # retain the original independent Bernoulli law for all subsequent segments.
        total = 1 - math.prod(1 - q for _, q in segments)
        threshold = draw("first") * total
        remaining = 1.0
        for index, (name, q) in enumerate(segments):
            threshold -= remaining * q
            if threshold <= 0 or index == len(segments) - 1:
                return frozenset(
                    [name] + [later for later, rate in segments[index + 1 :] if draw(later) < rate]
                )
            remaining *= 1 - q
        raise AssertionError("Nonempty condition distribution has no eligible segment")


class SenseNovaDropout(_PackedRecipe):
    """U1/U1.5 launchers select mutually exclusive task-specific dropout branches."""

    revision = "sensenova-u1-t2i10-it2i10-0-10-v1"
    checkpoints = ("sensenova/SenseNova-U1-8B-MoT", "sensenova/SenseNova-U1.5-8B-MoT-SFT")

    def sample(
        self, mode: float | str, model_input: NormalizedModelInput, draw: Callable[[str], float]
    ) -> frozenset[str]:
        has_images = any(asset.type == "image" for asset in model_input.media)
        if not model_input.prompt and not has_images:
            return frozenset()
        rate = (0.2 if has_images else 0.1) if mode == "native" else mode
        if draw("record") >= rate:
            return frozenset()
        if has_images and draw("branch") >= 0.5:
            return frozenset(("text", "images"))
        return frozenset(("text",))
