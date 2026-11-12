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

"""Independent differentiable image signal for native ReFL lifecycle evidence."""

import torch
from torch import nn

from flow_factory.contracts.reward import RewardGradientCapability
from flow_factory.rewards.abc import PointwiseRewardModel, RewardModelOutput


class PixelReward(PointwiseRewardModel):
    gradient_capability = RewardGradientCapability(input_fields=("image",))
    required_fields = ("prompt", "image")
    use_tensor_inputs = True

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self.model = nn.Conv2d(3, 1, 1, bias=False).to(self.device)
        with torch.no_grad():
            self.model.weight.copy_(
                torch.tensor([0.2, 0.3, 0.5], device=self.device).reshape(1, 3, 1, 1)
            )

    def score_differentiable(self, *, prompt, image):
        return RewardModelOutput(rewards=self.model(image.float()).mean((1, 2, 3)))

    @torch.no_grad()
    def __call__(self, *, prompt, image, **kwargs):
        return self.score_differentiable(
            prompt=prompt, image=torch.stack(image) if isinstance(image, list) else image
        )
