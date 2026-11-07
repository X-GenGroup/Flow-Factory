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


"""One CRD/model/topology intersection exercises the full shared lifecycle."""


def test_cuda_qwen_crd_ordered_full(tmp_path):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        {
            "case_module": "qwen_crd_fixture",
            "model_family": "qwen-image",
            "algorithm": "crd",
            "training_mode": "full",
            "scheduling_mode": "ordered",
            "sampler_type": "global_batch",
            "dp_size": 2,
            "cp_size": 2,
            "tp_size": 1,
        },
    )
