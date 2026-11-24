"""Input-reference geometry is independent of SFT cache batch size."""

import pytest
import torch
from PIL import Image

from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter
from flow_factory.models.qwen_image_21.output_codec import normalize_condition_img_shapes


@pytest.mark.parametrize("refs", [1, 2, 3, 4])
def test_qwen_21_keeps_reference_count_and_target_slot(refs):
    shapes = [(1, 2, index + 1) for index in range(refs)]
    for value in (shapes, [shapes], torch.tensor([shapes])):
        assert normalize_condition_img_shapes(value, batch_size=1) == [shapes]


def test_kontext_auto_geometry_retains_the_selected_records_aspect(monkeypatch):
    adapter = object.__new__(Flux1KontextAdapter)
    images = [Image.new("RGB", size) for size in ((32, 64), (64, 32))]
    monkeypatch.setattr(
        adapter, "_standardize_image_input", lambda rows, **kwargs: [row[0] for row in rows]
    )
    monkeypatch.setattr(
        adapter, "encode_prompt", lambda prompt: {"prompt_ids": [[len(s)] for s in prompt]}
    )
    monkeypatch.setattr(
        adapter,
        "encode_image",
        lambda rows: {"image_ids": torch.tensor([rows[0].size] * len(rows))},
    )
    result = adapter.preprocess_func(prompt=["", ""], images=[[image] for image in images])
    assert [tuple(ids.tolist()) for ids in result["image_ids"]] == [image.size for image in images]
