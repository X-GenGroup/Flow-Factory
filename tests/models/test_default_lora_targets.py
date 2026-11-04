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

"""Every default LoRA target must reach a ``Linear`` of the architecture it names.

PEFT accepts a target that matches no module without a word, so a default list
can lose a whole projection family and still train. These tests build each
transformer on the meta device and resolve the defaults with PEFT's own matcher.
"""

from __future__ import annotations

import inspect
from typing import Iterable, List, Set

import pytest
import torch
from peft import LoraConfig
from peft.tuners.tuners_utils import check_target_module_exists


def _resolve(model: torch.nn.Module, targets: Iterable[str]) -> Set[str]:
    config = LoraConfig(r=1, target_modules=list(targets))
    return {
        name
        for name, _ in model.named_modules()
        if name and check_target_module_exists(config, name)
    }


def _unreached(model: torch.nn.Module, targets: Iterable[str]) -> List[str]:
    return [target for target in targets if not _resolve(model, [target])]


def test_qwen_image_defaults_reach_both_projections_of_each_feed_forward() -> None:
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter

    with torch.device("meta"):
        model = QwenImageTransformer2DModel(num_layers=1)
    targets = QwenImageAdapter.default_target_modules.fget(QwenImageAdapter)
    resolved = _resolve(model, targets)

    assert _unreached(model, targets) == []
    for stream in ("img_mlp", "txt_mlp"):
        assert f"transformer_blocks.0.{stream}.net.0.proj" in resolved
        assert f"transformer_blocks.0.{stream}.net.2" in resolved
    assert all(isinstance(model.get_submodule(name), torch.nn.Linear) for name in resolved)


@pytest.mark.parametrize("variant", ["dev", "klein"])
def test_flux2_defaults_reach_the_single_stream_output_projection(variant: str) -> None:
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

    from flow_factory.models.flux.flux2 import MAX_SINGLE_BLOCKS, Flux2Adapter
    from flow_factory.models.flux.flux2_klein import Flux2KleinAdapter

    published_depth = (
        inspect.signature(Flux2Transformer2DModel.__init__).parameters["num_single_layers"].default
    )
    assert MAX_SINGLE_BLOCKS > published_depth

    adapter = Flux2Adapter if variant == "dev" else Flux2KleinAdapter
    depth = 2
    with torch.device("meta"):
        model = Flux2Transformer2DModel(num_layers=1, num_single_layers=depth)
    targets = adapter.default_target_modules.fget(adapter)
    resolved = _resolve(model, targets)

    assert {f"single_transformer_blocks.{index}.attn.to_out" for index in range(depth)} <= resolved
    assert "transformer_blocks.0.attn.to_out.0" in resolved
    assert _unreached(model, targets) == [
        f"single_transformer_blocks.{index}.attn.to_out"
        for index in range(depth, MAX_SINGLE_BLOCKS)
    ]
    assert all(isinstance(model.get_submodule(name), torch.nn.Linear) for name in resolved)
