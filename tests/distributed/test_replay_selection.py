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

"""A selected stochastic subset through real FSDP2 training and continuation."""


def test_cuda_selected_sde_replay(tmp_path):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        {
            "case_module": "flux_replay_selection_fixture",
            "model_family": "flux",
            "algorithm": "dppo",
            "training_mode": "lora",
            "scheduling_mode": "ordered",
            "dp_size": 2,
            "cp_size": 1,
            "tp_size": 1,
        },
    )
