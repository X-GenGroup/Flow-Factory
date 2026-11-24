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

"""Verify raw encoding against actual Arrow cache formatting for mixed offline conditions."""

import torch
from datasets import Dataset
from PIL import Image

from flow_factory.data_utils.dataset import (
    _apply_torch_format,
    format_preprocessed_columns,
)
from flow_factory.data_utils.offline_dataset import _collate_condition_mappings
from flow_factory.data_utils.sft_conditions import encode_raw_sft_conditions


class _Encoder:
    python_format_columns = frozenset({"condition_images"})

    def preprocess(self, prompt, images):
        i = int(prompt[0])
        return {
            "prompt_embeds": [torch.full((i + 1, 2), i + 0.25, dtype=torch.bfloat16)],
            "mask": torch.tensor([[i, 0]], dtype=torch.int32),
            "flag": torch.tensor([i == 1]),
            "sizes": [[(32, 64)] * i],
            "empty_ids": [[] if i == 0 else [42, 43]],
            "weight": [0.375],
            "image_latent": [None if not i else torch.ones(2, dtype=torch.float16)],
            "equal_refs": [[torch.tensor([i]), torch.tensor([i + 1])]],
            "ragged_refs": [[torch.ones(2), torch.ones(3)]],
            "condition_images": images,
        }


def test_raw_sft_matches_cache_dtypes_and_ragged_fields_without_serializing_images(monkeypatch):
    image = Image.new("RGB", (3, 5), (23, 87, 111))
    encoder = _Encoder()
    rows = [
        (
            {"prompt": [str(i)], "images": [[image]], "negative_prompt": ["protected"]},
            {"prompt": [str(i)], "images": [[image]]},
        )
        for i in range(2)
    ]
    encoded = [
        {
            key: column[0]
            for key, column in {
                **batch,
                **format_preprocessed_columns(encoder.preprocess, encoder.preprocess(**inputs)),
            }.items()
        }
        for batch, inputs in rows
    ]
    cached = Dataset.from_list(encoded)
    _apply_torch_format(cached)
    expected = _collate_condition_mappings([cached[0], cached[1]])

    def unexpected_image_save(*args, **kwargs):
        raise AssertionError("raw encoding serialized an image through Arrow")

    monkeypatch.setattr(Image.Image, "save", unexpected_image_save)
    actual = encode_raw_sft_conditions(rows, encoder.preprocess, {})
    for field in ("images", "condition_images"):
        for raw_refs, cached_refs in zip(actual.pop(field), expected.pop(field), strict=True):
            assert [ref.tobytes() for ref in raw_refs] == [ref.tobytes() for ref in cached_refs]
    assert actual.pop("negative_prompt") == expected.pop("negative_prompt") == ["protected"] * 2
    assert actual.pop("prompt") == expected.pop("prompt") == ["0", "1"]
    assert actual["prompt_embeds"][0].dtype == torch.float32
    assert actual["mask"].dtype == torch.int64
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
