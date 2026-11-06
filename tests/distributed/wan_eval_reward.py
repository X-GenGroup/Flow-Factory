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

"""Pointwise evaluation probe with visible, dataset-local request identities."""

import torch

from flow_factory.rewards import PointwiseRewardModel, RewardModelOutput


class PromptReward(PointwiseRewardModel):
    required_fields = ("prompt", "source")
    scored = []

    def __call__(self, prompt, source):
        type(self).scored.extend(zip(source, prompt))
        return RewardModelOutput(
            rewards=torch.tensor([float(p.rsplit(" ", 1)[-1]) for p in prompt])
        )
