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

"""Bind the native distilled encoder/denoiser to the common Flux3 lifecycle recipe."""

from functools import partial

from flux3_coupled_fixture import (
    assert_objectives,
    audit_kind,
    audit_samples,
    audit_steps,
    build_variant_fixture,
    dataset_record,
    make_config,
    observe_objectives,
    observe_training_terms,
    reference_media_score,
    reward_media_field,
    score_media,
    wrap_classes,
)

build_fixture = partial(build_variant_fixture, variant="distilled")
