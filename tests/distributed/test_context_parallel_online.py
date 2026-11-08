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

"""Collected Wan online qualification leaves; shared execution lives in its runner."""

import pytest
from context_parallel_online_runner import _config, _qualify_online, _worker


@pytest.mark.parametrize("trainer_type", ["grpo", "dmd2"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_cuda_wan_online_cp_equivalence(tmp_path, trainer_type, mode, tp_size):
    _qualify_online(tmp_path, trainer_type, mode, tp_size)


@pytest.mark.parametrize("tp_size,mode", [(1, "full"), (2, "lora")])
def test_cuda_wan_grpo_cfg_cp_equivalence(tmp_path, tp_size, mode):
    _qualify_online(tmp_path, "grpo", mode, tp_size, guidance_scale=3.0)
