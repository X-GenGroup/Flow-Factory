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

"""Deterministic varying rewards for logical-sample CP qualification."""

import torch

from flow_factory.rewards import PointwiseRewardModel, RewardModelOutput


class MemberReward(PointwiseRewardModel):
    """Score planned members differently and retain actual request identities."""

    required_fields = ("sampling_sample_id", "sampling_group_member_id")
    scored = []

    def __call__(self, sampling_sample_id, sampling_group_member_id):
        type(self).scored.extend(int(value) for value in sampling_sample_id)
        return RewardModelOutput(
            rewards=torch.tensor(
                [-1.0 if int(value) == 0 else 1.0 for value in sampling_group_member_id]
            )
        )
