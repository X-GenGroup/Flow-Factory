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

"""Native flf2v ordered acquisition/evaluation/recovery recipe."""

from functools import partial

import wan_remaining_coupled_fixture as wan
from wan_remaining_coupled_fixture import (
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

build_fixture = partial(wan.build_fixture, variant="flf2v")
make_config = partial(wan.make_config, variant="flf2v")
dataset_record = partial(wan.dataset_record, variant="flf2v")
audit_samples = partial(wan.audit_samples, variant="flf2v")
