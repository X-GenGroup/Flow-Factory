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

"""Native TDM TP/SP/CP, role cadence, objectives and drained continuation."""

from tdm_context_parallel_runner import qualify_lifecycle


def test_cuda_wan_tdm_full_tp_cp_lifecycle(tmp_path):
    qualify_lifecycle(
        tmp_path,
        recipe_module="wan_tdm_fixture",
        options={"dp_size": 2, "tp_size": 2, "cp_size": 2, "mode": "full", "algorithm": "tdm"},
    )
