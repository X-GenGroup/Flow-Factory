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

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from flow_factory.contracts import (
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    MediaType,
    NegativePromptPolicy,
)
from flow_factory.data_utils.offline_dataset import DecodedMedia
from flow_factory.models.sensenova._output import SenseNovaPixelOutputCodec
from flow_factory.models.sensenova.sensenova import SenseNovaAdapter


class _Adapter:
    _image_shape = SenseNovaAdapter._image_shape
    _configured_output_image_geometry = SenseNovaAdapter._configured_output_image_geometry

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.transformer = SimpleNamespace(dtype=torch.float32)
        self.training_args = SimpleNamespace(height=8, width=16)
        self.model = SimpleNamespace(patch_size=4, downsample_ratio=0.5)

    def _base_model(self):
        return self.model


def _media(image: Image.Image):
    return (
        (
            DecodedMedia(
                type="image",
                path="target.png",
                payload=image,
            ),
        ),
    )


def test_sensenova_declares_ordered_images_and_pixel_output() -> None:
    contract = SenseNovaAdapter.pipeline_io_contract

    assert SenseNovaAdapter.supports_ordered_references is False
    assert contract.input_media.binding is InputMediaBinding.GROUPED_BY_TYPE
    assert contract.input_media.order is InputMediaOrder.WITHIN_TYPE
    assert contract.input_media.rules[0].min_count == 0
    assert contract.input_media.rules[0].max_count is None
    assert contract.negative_prompt is NegativePromptPolicy.UNSUPPORTED
    assert contract.geometry_source is GeometrySource.CONFIGURED
    assert contract.output_media.items[0].type is MediaType.IMAGE
    SenseNovaAdapter.validate_offline_output_capability()

    codec = SenseNovaAdapter.build_output_state_codec(object.__new__(SenseNovaAdapter))
    assert isinstance(codec, SenseNovaPixelOutputCodec)
    assert codec.required_components == ("transformer",)


def test_sensenova_codec_maps_configured_rgb_targets_to_pixel_state() -> None:
    adapter = _Adapter()
    target = Image.new("RGB", (3, 5), color=(0, 127, 255))

    encoded = SenseNovaPixelOutputCodec(adapter).encode_output_state(
        _media(target),
        {"prompt": ["mixed endpoints"]},
        torch.Generator().manual_seed(3),
    )

    pixels = encoded.clean_state.components["latent"]
    assert pixels.shape == (1, 3, 8, 16)
    expected_channels = torch.tensor(
        [-1.0, 127.0 / 127.5 - 1.0, 1.0],
        dtype=torch.float32,
    )
    expected = expected_channels.view(1, 3, 1, 1).expand(1, 3, 8, 16)
    torch.testing.assert_close(pixels, expected, rtol=0, atol=1e-7)
    assert pixels.dtype is torch.float32
    assert encoded.forward_context == {}
    assert dict(encoded.decode_context) == {"height": 8, "width": 16}
    geometry = encoded.geometry_signatures[0].media[0]
    assert (geometry.type, geometry.height, geometry.width) == (MediaType.IMAGE, 8, 16)


def test_sensenova_codec_requires_canonical_decoded_rgb_targets() -> None:
    adapter = _Adapter()

    with pytest.raises(ValueError, match="RGB mode"):
        SenseNovaPixelOutputCodec(adapter).encode_output_state(
            _media(Image.new("L", (3, 5))),
            {},
        )


def test_sensenova_configured_geometry_uses_model_patch_merge_factor() -> None:
    adapter = _Adapter()
    assert adapter._configured_output_image_geometry() == (8, 16)

    adapter.training_args.width = 12
    with pytest.raises(ValueError, match="patch merge factor 8"):
        adapter._configured_output_image_geometry()


def test_sensenova_geometry_validator_rejects_decode_context_drift() -> None:
    adapter = _Adapter()
    media = _media(Image.new("RGB", (8, 16)))
    encoded = SenseNovaPixelOutputCodec(adapter).encode_output_state(media, {})

    SenseNovaAdapter._validate_encoded_output_geometry(adapter, media, {}, encoded)

    drifted = type(encoded)(
        clean_state=encoded.clean_state,
        forward_context=encoded.forward_context,
        decode_context={"height": 16, "width": 16},
        geometry_signatures=encoded.geometry_signatures,
    )
    with pytest.raises(ValueError, match="decode_context 'height'"):
        SenseNovaAdapter._validate_encoded_output_geometry(adapter, media, {}, drifted)
