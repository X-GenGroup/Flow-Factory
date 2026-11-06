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


"""Wan full DGPO with logical-DP groups and TP/SP/CP ordered overlap."""

from dataclasses import fields

from dgpo_observations import (
    assert_objectives,
    audit_samples,
    observe_objectives,
    observe_training_terms,
)
from wan_overlap_fixture import build_fixture
from wan_overlap_fixture import make_config as wan_config
from wan_overlap_fixture import reference_media_score, reward_media_field, score_media, wrap_classes

from flow_factory.hparams import Arguments, DGPOTrainingArguments

audit_kind = "terminal_and_shared_noised_inputs"
audit_steps = 1


def make_config(root, result_dir, options):
    base = wan_config(root, result_dir, options)
    allowed = {field.name for field in fields(DGPOTrainingArguments) if field.init}
    train = {key: value for key, value in base.training_args.to_dict().items() if key in allowed}
    train.update(
        trainer_type="dgpo",
        num_train_timesteps=2,
        time_sampling_strategy="uniform",
        time_shift=1.0,
        timestep_range=(0.1, 0.8),
        dpo_beta=4.0,
        off_policy=True,
        switch_ema_ref=0,
        use_shared_noise=True,
        use_ema_ref=True,
        clip_dsm=True,
        clip_kl=True,
        clip_range=(-0.2, 0.2),
        ema_ref_max_decay=0.5,
        ema_ref_ramp_rate=0.25,
        global_std=False,
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
