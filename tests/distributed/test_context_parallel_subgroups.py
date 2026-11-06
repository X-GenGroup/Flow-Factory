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

"""Native subgroup GRPO uses logical DP for feedback and exact resumed replay."""

import pytest
from context_parallel_online_runner import _qualify_online


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("tp_size,dp_size", [(1, 4), (2, 2)], ids=["tp1-dp4", "tp2-dp2"])
def test_cuda_subgroup_grpo_cp_equivalence(tmp_path, mode, tp_size, dp_size):
    """Separate two R2 subgroups from TP+CP composition within eight CUDA ranks."""
    _qualify_online(
        tmp_path,
        "grpo",
        mode,
        tp_size,
        dp_size=dp_size,
        sampler_type="subgroup_tile",
        subgroup_size=2,
    )
