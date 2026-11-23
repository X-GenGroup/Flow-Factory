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

"""Combined encoder, frozen VAE and real reward sharding through ReFL."""

from test_refl_sharded_auxiliary import qualify_sharded_refl, requires_real_pickscore_cuda


@requires_real_pickscore_cuda
def test_cuda_sd35_refl_sharded_preprocessing_and_reward(tmp_path):
    qualify_sharded_refl(tmp_path, encoders=("text_encoder", "text_encoder_2", "text_encoder_3"))
