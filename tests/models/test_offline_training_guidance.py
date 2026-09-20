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

import pytest

from flow_factory.models.abc import BaseAdapter
from flow_factory.models.flux.flux1 import Flux1Adapter
from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter
from flow_factory.models.flux.flux2 import Flux2Adapter
from flow_factory.models.flux.flux2_klein import Flux2KleinAdapter
from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter
from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter
from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter
from flow_factory.models.sensenova.sensenova import SenseNovaAdapter
from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter
from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter
from flow_factory.models.z_image.z_image import ZImageAdapter


def test_offline_training_forward_overrides_are_immutable() -> None:
    with pytest.raises(TypeError):
        BaseAdapter.offline_training_forward_overrides["guidance_scale"] = 8.0  # type: ignore[index]


@pytest.mark.parametrize(
    "adapter_cls",
    [
        BaseAdapter,
        SD3_5Adapter,
        Flux2KleinAdapter,
        QwenImageAdapter,
        QwenImageEditPlusAdapter,
        QwenImage21Adapter,
    ],
)
def test_cfg_adapters_default_to_a_non_composite_offline_velocity(
    adapter_cls: type[BaseAdapter],
) -> None:
    assert dict(adapter_cls.offline_training_forward_overrides) == {"guidance_scale": 1.0}


@pytest.mark.parametrize(
    "adapter_cls",
    [Flux1Adapter, Flux1KontextAdapter, Flux2Adapter],
)
def test_guidance_distilled_adapters_declare_their_official_training_condition(
    adapter_cls: type[BaseAdapter],
) -> None:
    assert dict(adapter_cls.offline_training_forward_overrides) == {"guidance_scale": 3.5}


def test_z_image_declares_its_model_specific_cfg_off_value() -> None:
    assert dict(ZImageAdapter.offline_training_forward_overrides) == {
        "guidance_scale": 0.0,
        "cfg_normalization": False,
        "cfg_truncation": 1.0,
    }


def test_wan_t2v_disables_both_transformer_cfg_scales() -> None:
    assert dict(Wan2_T2V_Adapter.offline_training_forward_overrides) == {
        "guidance_scale": 1.0,
        "guidance_scale_2": 1.0,
    }


def test_sensenova_disables_text_and_image_guidance() -> None:
    assert dict(SenseNovaAdapter.offline_training_forward_overrides) == {
        "guidance_scale": 1.0,
        "image_guidance_scale": 1.0,
        "cfg_norm": "none",
        "cfg_interval": (0.0, 1.0),
    }
