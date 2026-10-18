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

import numpy as np
import pytest
import torch
from PIL import Image

from flow_factory.samples import I2ISample, T2ISample, T2VSample, V2VSample


def test_sample_media_read_bytes_and_pil_as_unit_floats() -> None:
    """Divide PIL and integer pixels by 255 at construction, whatever their peak."""
    pil = T2ISample(image=Image.new("RGB", (4, 2), color=(255, 0, 51)))
    dark = T2ISample(image=torch.ones(3, 2, 4, dtype=torch.uint8))
    frames = T2VSample(video=[Image.new("RGB", (4, 2), color=(51, 51, 51))] * 2)

    assert pil.image.dtype == torch.float32
    assert pil.image.shape == (3, 2, 4)
    torch.testing.assert_close(pil.image[:, 0, 0], torch.tensor([1.0, 0.0, 0.2]))
    torch.testing.assert_close(dark.image, torch.full((3, 2, 4), 1.0 / 255.0))
    torch.testing.assert_close(frames.video, torch.full((2, 3, 2, 4), 0.2))


def test_sample_keeps_canonical_unit_media_untouched() -> None:
    """Hand in-range producer tensors through without copying or changing their dtype."""
    generator = torch.Generator().manual_seed(0)
    image = torch.rand(3, 4, 4, generator=generator).bfloat16()
    video = torch.rand(2, 3, 4, 4, generator=generator)

    for stored, source in (
        (T2ISample(image=image).image, image),
        (T2VSample(video=video).video, video),
    ):
        assert stored.data_ptr() == source.data_ptr()
        assert stored.dtype == source.dtype
        assert torch.equal(stored, source)


def test_sample_clamps_edge_drift_but_rejects_other_intervals() -> None:
    """Clamp bfloat16-sized drift and name the field that received [-1, 1] or [0, 255] floats."""
    drifted = torch.full((3, 4, 4), 0.6)
    drifted[0, 0, 0] = 1.0078125

    assert float(T2ISample(image=drifted).image.max()) == 1.0
    with pytest.raises(ValueError, match=r"^T2ISample\.image expected unit pixels"):
        T2ISample(image=torch.full((3, 4, 4), -0.5))
    with pytest.raises(ValueError, match=r"^T2VSample\.video expected unit pixels"):
        T2VSample(video=torch.full((2, 3, 4, 4), 255.0))


def test_condition_media_canonicalize_and_validate_each_entry() -> None:
    """Read byte conditions as unit floats and name the condition entry that breaks the range."""
    sample = I2ISample(condition_images=[np.full((4, 4, 3), 51, dtype=np.uint8)])

    torch.testing.assert_close(sample.condition_images[0], torch.full((3, 4, 4), 0.2))
    with pytest.raises(ValueError, match=r"^I2ISample\.condition_images\[1\] expected unit"):
        I2ISample(condition_images=[torch.rand(3, 4, 4), torch.full((3, 4, 4), -0.5)])
    with pytest.raises(ValueError, match=r"^V2VSample\.condition_videos\[0\] expected unit"):
        V2VSample(condition_videos=[torch.full((2, 3, 4, 4), -0.5)])


def test_condition_identity_matches_between_pil_and_unit_tensor_inputs() -> None:
    """Fingerprint a condition image identically whether it arrived as PIL or unit floats."""
    pil = Image.new("RGB", (8, 8), color=(255, 128, 0))
    from_pil = I2ISample(prompt="edit", condition_images=[pil])
    from_tensor = I2ISample(
        prompt="edit",
        condition_images=[from_pil.condition_images[0].clone()],
    )

    assert from_pil.compute_unique_id() == from_tensor.compute_unique_id()
