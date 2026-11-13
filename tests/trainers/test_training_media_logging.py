# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Decoded feedback media have a shared, detached acquisition logging budget."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from flow_factory.contracts.execution import AcquisitionMode, FeedbackMode
from flow_factory.logger.formatting import LogFormatter
from flow_factory.samples import T2AVSample, T2ISample
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.registry import get_trainer_class, list_registered_trainers
from flow_factory.trainers.rl.refl import ReFLTrainer


def host(limit=2, cls=ReFLTrainer):
    trainer = object.__new__(cls)
    trainer.log_args = SimpleNamespace(max_logged_train_samples=limit)
    trainer.logger = object()
    trainer.epoch = 0
    trainer.step = 7
    trainer.records = []
    trainer.log_data = lambda data, step: trainer.records.append((step, data))
    return trainer


def test_budget_skips_latent_only_rows_and_resets_per_acquisition():
    trainer = host()
    hidden = T2ISample(prompt="latent only")
    samples = [T2ISample(prompt=str(i), image=torch.ones(3, 2, 2)) for i in range(3)]
    trainer.log_train_samples([hidden, *samples])
    trainer.log_train_samples(samples)
    assert len(trainer.records) == 1
    assert [s.prompt for s in trainer.records[0][1]["train_samples"]] == ["0", "1"]
    trainer.epoch = 1
    trainer.log_train_samples(samples)
    assert len(trainer.records) == 2


@pytest.mark.parametrize("limit,enabled", [(0, True), (2, False)])
def test_disabled_logging_never_accesses_media(limit, enabled):
    trainer = host(limit)
    if not enabled:
        trainer.logger = None
    trainer.log_train_samples([object()], media={"image": object()})
    assert not trainer.records


def test_differentiable_av_logs_scored_media_without_mutating_rollout_or_graph():
    trainer = host(1)
    original = T2AVSample(prompt="a bird", extra_kwargs={"frame_rate": 24}, audio_sample_rate=24000)
    video = torch.full((1, 2, 3, 2, 2), 0.2, requires_grad=True)
    audio = torch.ones(1, 2, 12, requires_grad=True)
    scores = video.square().mean((1, 2, 3, 4))
    trainer.log_train_samples(
        [original], media={"video": video, "audio": audio}, rewards={"pick": scores}
    )
    logged = trainer.records[0][1]["train_samples"][0]
    assert logged.video.grad_fn is None and not logged.video.requires_grad
    assert logged.audio.grad_fn is None and not logged.audio.requires_grad
    assert logged.extra_kwargs["rewards"]["pick"] == pytest.approx(0.04)
    assert (
        original.video is None
        and original.audio is None
        and original.extra_kwargs == {"frame_rate": 24}
    )
    scores.sum().backward()
    torch.testing.assert_close(video.grad, 2 * video.detach() / video.numel())
    ir = LogFormatter.format_dict({"train_samples": [logged]})["train_samples"][0]
    assert ir.fps == 24 and ir.audio_sample_rate == 24000
    assert "pick" not in ir.caption or "0.04" in ir.caption


@pytest.mark.parametrize(
    "algorithm",
    [
        name
        for name in list_registered_trainers()
        if get_trainer_class(name).execution_contract.feedback is FeedbackMode.RUNTIME_REWARD
    ],
)
def test_runtime_feedback_logs_after_custom_feedback_before_optimization(algorithm):
    trainer = host(1, get_trainer_class(algorithm))
    sample = T2ISample(prompt="a bird", image=torch.ones(3, 2, 2))
    trainer.training_args = SimpleNamespace(reward_optimization_overlap=False)
    trainer.sampling_context = nullcontext
    trainer.sample = lambda: [sample]
    trainer.prepare_feedback = lambda rows: rows[0].extra_kwargs.update(rewards={"pick": 0.8})
    trainer.optimize = lambda rows: None
    trainer._critical_path_timing_metrics = lambda metrics: metrics
    BaseTrainer._run_training_step(trainer)
    entries = [data["train_samples"] for _, data in trainer.records if "train_samples" in data]
    assert len(entries) == 1 and entries[0][0].extra_kwargs["rewards"] == {"pick": 0.8}


@pytest.mark.parametrize(
    "algorithm",
    [
        name
        for name in list_registered_trainers()
        if get_trainer_class(name).execution_contract.feedback is FeedbackMode.NONE
        and get_trainer_class(name).execution_contract.acquisition is AcquisitionMode.GENERATION
    ],
)
def test_reward_free_generation_adds_no_media_logging_or_decoding(algorithm):
    trainer = host(1, get_trainer_class(algorithm))
    trainer.training_args = SimpleNamespace(reward_optimization_overlap=False)
    trainer.sampling_context = nullcontext
    trainer.sample = lambda: [T2ISample(prompt="latent only")]
    trainer.optimize = lambda rows: None
    trainer._critical_path_timing_metrics = lambda metrics: metrics
    BaseTrainer._run_training_step(trainer)
    assert all("train_samples" not in data for _, data in trainer.records)


def test_tdm_r1_distillation_driver_logs_its_rewarded_microbatch_collection():
    from flow_factory.trainers.distillation.distillation_runtime import (
        run_distillation_training_step,
    )

    trainer = host(1, get_trainer_class("tdm-r1"))
    trainer.log_args.verbose = False
    trainer.accelerator = SimpleNamespace(
        is_local_main_process=False,
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value,
    )
    trainer.training_args = SimpleNamespace(
        gradient_accumulation_steps=2,
        get_num_train_timesteps=lambda config: 1,
        advantage_aggregation="sum",
    )
    trainer.config = SimpleNamespace()
    sample = T2ISample(prompt="a bird", image=torch.ones(3, 2, 2))
    trainer.sample = lambda: [sample]
    trainer.sampling_context = nullcontext
    trainer.reward_buffer = SimpleNamespace(
        clear=lambda: None, finalize=lambda **kwargs: {"pick": torch.tensor([0.8, 0.8])}
    )

    def advantages(**kwargs):
        for row in kwargs["samples"]:
            row.extra_kwargs["rewards"] = {"pick": 0.8}

    trainer.advantage_processor = SimpleNamespace(
        compute_advantages=advantages, pop_advantage_metrics=lambda: {}
    )
    trainer.optimize = lambda rows: None
    run_distillation_training_step(trainer)
    entries = [data["train_samples"] for _, data in trainer.records if "train_samples" in data]
    assert len(entries) == 1 and len(entries[0]) == 1
    assert entries[0][0].extra_kwargs["rewards"] == {"pick": 0.8}
