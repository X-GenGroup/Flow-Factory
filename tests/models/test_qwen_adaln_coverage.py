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

"""Count actual Qwen TP weight ownership without allocating released weights."""

import pytest
import torch
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from diffusers import QwenImageTransformer2DModel
from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter
from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter
from flow_factory.models.tensor_parallel import SharedInputColwiseParallel, _expand_globs


@pytest.mark.parametrize("adapter", [QwenImageAdapter, QwenImageEditPlusAdapter])
def test_native_qwen_adaln_parameter_coverage(adapter):
    with torch.device("meta"):
        model = QwenImageTransformer2DModel()
    plan = _expand_globs(model, adapter.tensor_parallel_plan())
    total = sharded = added = 0
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        owner, leaf = name.rsplit(".", 1)
        style = plan.get(owner)
        partitions = isinstance(style, (ColwiseParallel, SharedInputColwiseParallel)) or (
            isinstance(style, RowwiseParallel) and leaf == "weight"
        )
        if partitions:
            sharded += parameter.numel()
            if owner.endswith(("img_mod.1", "txt_mod.1")) or owner == "norm_out.linear":
                added += parameter.numel()
    assert total == 20_430_401_088
    assert sharded == 20_405_772_288
    assert added == 6_813_646_848
    assert sharded < total  # Biases and boundaries are deliberately replicated.
