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


"""Native packed Flux NFT recipe exercising LoRA CP ordered overlap."""

import flux_coupled_fixture as flux
from nft_overlap_fixture import (
    assert_objectives,
    audit_kind,
    audit_samples,
    audit_steps,
    nft_config,
    observe_objectives,
    observe_training_terms,
)

build_fixture = flux.build_fixture
reference_media_score = flux.reference_media_score
reward_media_field = flux.reward_media_field
score_media = flux.score_media
wrap_classes = flux.wrap_classes


def make_config(root, result_dir, options):
    return nft_config(
        flux.make_config(root, result_dir, {**options, "algorithm": "grpo-guard"}), options
    )
