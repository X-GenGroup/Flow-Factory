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

"""Bound native ref2va structured AV lifecycle fixture."""

import h3_coupled_fixture as native
from h3_coupled_fixture import (
    assert_objectives,
    audit_kind,
    audit_steps,
    observe_objectives,
    observe_training_terms,
    reference_media_score,
    reward_media_field,
    score_media,
    wrap_classes,
)


def build_fixture(root, tp_size):
    return native.build_fixture(root, tp_size, "ref2va")


def dataset_record(root, prompt, index):
    return native.dataset_record(root, prompt, index, "ref2va")


def make_config(root, result_dir, options):
    return native.make_config(root, result_dir, options, "ref2va")


def audit_samples(trainer, samples):
    return native.audit_samples(trainer, samples, "ref2va")
