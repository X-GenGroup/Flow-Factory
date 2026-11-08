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


"""Qwen Image CRD full training with independent matching and snapshot proofs."""

from dataclasses import fields

import qwen_coupled_fixture as qwen
from crd_observations import (
    assert_objectives,
    audit_samples,
    observe_objectives,
    observe_training_terms,
)
from qwen_cp_fixture import inactive_parameter_names

from flow_factory.hparams import Arguments, CRDTrainingArguments

build_fixture = qwen.build_fixture
reference_media_score = qwen.reference_media_score
reward_media_field = qwen.reward_media_field
score_media = qwen.score_media
wrap_classes = qwen.wrap_classes
audit_kind = "terminal_and_shared_noised_inputs"
audit_steps = 1


def make_config(root, result_dir, options):
    base = qwen.make_config(root, result_dir, {**options, "algorithm": "grpo-guard"})
    allowed = {field.name for field in fields(CRDTrainingArguments) if field.init}
    train = {key: value for key, value in base.training_args.to_dict().items() if key in allowed}
    train.update(
        trainer_type="crd",
        crd_beta=0.6,
        crd_loss_type="mse",
        weight_temp=0.7,
        adaptive_logp=True,
        use_old_for_loss=True,
        old_model_decay="0-0.25-0.05-0.6",
        sampling_model_decay="0-0.1-0.03-0.5",
        kl_cfg=3.0,
        reward_adaptive_kl=True,
        num_train_timesteps=2,
        time_sampling_strategy="uniform",
        time_shift=1.0,
        timestep_range=(0.1, 0.8),
        kl_beta=0.01,
        kl_type="v-based",
        global_std=False,
        gradient_accumulation_steps="auto",
        advantage_aggregation="sum",
    )
    data = base.data_args.to_dict()
    data.update(sampler_type="global_batch")
    scheduler = base.scheduler_args.to_dict()
    scheduler.update(dynamics_type="ODE", sde_steps=None, num_sde_steps=0)
    return Arguments.from_dict(
        {
            "mixed_precision": base.mixed_precision,
            "parallelism": base.parallelism_args.to_dict(),
            "model": base.model_args.to_dict(),
            "data": data,
            "train": train,
            "scheduler": scheduler,
            "log": base.log_args.to_dict(),
            "eval": base.eval_args.to_dict(),
            "acceleration": base.acceleration_args.to_dict(),
            "optimizers": [item.to_dict() for item in base.optimizer_args],
            "rewards": [item.to_dict() for item in base.reward_args],
        }
    )
