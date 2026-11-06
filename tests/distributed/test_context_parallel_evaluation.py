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

"""Collected Wan evaluation leaves; configuration and workers have explicit owners."""

import pytest
from context_parallel_evaluation_runner import _config, _qualify_evaluation, _worker


@pytest.mark.parametrize("trainer_type", ["sft", "grpo", "dmd2"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_cuda_cp_evaluation_preserves_ema_and_training(tmp_path, trainer_type, mode, tp_size):
    _qualify_evaluation(tmp_path, trainer_type, mode, tp_size, cp_size=2)


@pytest.mark.parametrize("trainer_type,mode,tp_size", [("sft", "full", 1), ("grpo", "lora", 2)])
def test_cuda_cp_off_evaluation_regression(tmp_path, trainer_type, mode, tp_size):
    _qualify_evaluation(tmp_path, trainer_type, mode, tp_size, cp_size=1)
