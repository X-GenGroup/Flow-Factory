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

"""Three bounded native Wan overlap selectors, each with drained continuation."""

from context_parallel_overlap_runner import qualify_overlap


def test_cuda_wan_overlap_ordered_full(tmp_path):
    qualify_overlap(
        tmp_path,
        {
            "case_module": "wan_overlap_fixture",
            "algorithm": "grpo",
            "training_mode": "full",
            "scheduling_mode": "ordered",
            "dp_size": 2,
            "cp_size": 2,
            "tp_size": 1,
        },
    )


def test_cuda_wan_overlap_ready_tp_lora(tmp_path):
    qualify_overlap(
        tmp_path,
        {
            "case_module": "wan_overlap_fixture",
            "algorithm": "grpo",
            "training_mode": "lora",
            "scheduling_mode": "ready",
            "dp_size": 2,
            "cp_size": 2,
            "tp_size": 2,
            "failure_leg": True,
        },
    )


def test_cuda_wan_overlap_ready_subgroup(tmp_path):
    qualify_overlap(
        tmp_path,
        {
            "case_module": "wan_overlap_fixture",
            "algorithm": "grpo",
            "training_mode": "full",
            "scheduling_mode": "ready",
            "dp_size": 4,
            "cp_size": 2,
            "tp_size": 1,
            "sampler_type": "subgroup_tile",
            "subgroup_size": 2,
            "tiles": 4,
        },
    )
