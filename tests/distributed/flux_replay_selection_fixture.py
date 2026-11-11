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

"""Reuse the native coupled lifecycle with three noisy steps and two replay steps."""

from flux_coupled_fixture import (
    assert_objectives,
    audit_kind,
    audit_samples,
    audit_steps,
    build_fixture,
)
from flux_coupled_fixture import make_config as make_base_config
from flux_coupled_fixture import (
    observe_objectives,
    observe_training_terms,
    reference_media_score,
    reward_media_field,
    score_media,
    wrap_classes,
)


def make_config(root, result_dir, options):
    """Keep the two-term geometry, independently expand stochastic exploration."""
    from flow_factory.hparams import SchedulerArguments

    config = make_base_config(root, result_dir, options)
    config.scheduler_args = SchedulerArguments(
        dynamics_type="Flow-SDE",
        sde_steps=[0, 1, 2],
        num_sde_steps=None,
        replay_steps=[1, 0],
        seed=17,
    )
    assert config.training_args.get_num_train_timesteps(config) == 2
    return config
