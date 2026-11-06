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

"""Image-derived native Flux reward with retained request identities."""

import torch

from flow_factory.rewards import PointwiseRewardModel, RewardModelOutput


class ImageReward(PointwiseRewardModel):
    """Read actual decoded images; planned member IDs do not determine reward."""

    required_fields = ("image", "sampling_sample_id")
    use_tensor_inputs = True
    scored = []
    values = []

    def __call__(self, image, sampling_sample_id):
        type(self).scored.extend(int(value) for value in sampling_sample_id)
        rewards = []
        for pixels in image:
            assert pixels.ndim == 3 and pixels.shape == (3, 64, 64)
            assert torch.isfinite(pixels).all()
            # A spatial/color contrast is sensitive to the actual generated image.
            score = (pixels[0, :32].float().mean() - pixels[2, 32:].float().mean()).cpu()
            rewards.append(score)
        values = torch.stack(rewards)
        type(self).values.extend(values.tolist())
        return RewardModelOutput(rewards=values)
