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


"""Selected NFT native intersections; not a Cartesian model/algorithm matrix."""

from context_parallel_overlap_runner import qualify_overlap


def test_cuda_wan_nft_ready_tp_full(tmp_path):
    qualify_overlap(
        tmp_path,
        {
            "case_module": "wan_nft_fixture",
            "algorithm": "nft",
            "model_family": "wan",
            "training_mode": "full",
            "scheduling_mode": "ready",
            "dp_size": 2,
            "cp_size": 2,
            "tp_size": 2,
            "off_policy": True,
        },
    )


def test_cuda_flux_nft_ordered_lora(tmp_path):
    qualify_overlap(
        tmp_path,
        {
            "case_module": "flux_nft_fixture",
            "algorithm": "nft",
            "model_family": "flux",
            "training_mode": "lora",
            "scheduling_mode": "ordered",
            "dp_size": 2,
            "cp_size": 2,
            "tp_size": 1,
            "off_policy": False,
        },
    )
