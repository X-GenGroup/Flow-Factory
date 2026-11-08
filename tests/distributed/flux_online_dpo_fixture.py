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

"""Flux online DPO keeps model construction and shared lifecycle in their existing owners."""

from dataclasses import fields

import flux_coupled_fixture as flux
from online_dpo_observations import (
    assert_objectives,
    audit_samples,
    observe_objectives,
    observe_training_terms,
    training_term_identities,
)

from flow_factory.hparams import Arguments, DPOTrainingArguments

build_fixture = flux.build_fixture
reference_media_score = flux.reference_media_score
reward_media_field = flux.reward_media_field
score_media = flux.score_media
wrap_classes = flux.wrap_classes
audit_kind = "terminal_and_shared_preference_inputs"
audit_steps = 1
training_examples_per_tile = 1


def make_config(root, result_dir, options):
    base = flux.make_config(root, result_dir, {**options, "algorithm": "grpo-guard"})
    allowed = {field.name for field in fields(DPOTrainingArguments) if field.init}
    train = {key: value for key, value in base.training_args.to_dict().items() if key in allowed}
    train.update(
        trainer_type="dpo",
        beta=3.0,
        num_train_timesteps=2,
        weighting_scheme="uniform",
        time_shift=1.0,
        timestep_range=(0.1, 0.8),
    )
    scheduler = base.scheduler_args.to_dict()
    scheduler.update(dynamics_type="ODE", sde_steps=None, num_sde_steps=0)
    return Arguments.from_dict(
        {
            "mixed_precision": base.mixed_precision,
            "parallelism": base.parallelism_args.to_dict(),
            "model": base.model_args.to_dict(),
            "data": base.data_args.to_dict(),
            "train": train,
            "scheduler": scheduler,
            "log": base.log_args.to_dict(),
            "eval": base.eval_args.to_dict(),
            "acceleration": base.acceleration_args.to_dict(),
            "optimizers": [item.to_dict() for item in base.optimizer_args],
            "rewards": [item.to_dict() for item in base.reward_args],
        }
    )
