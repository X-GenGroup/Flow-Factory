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

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from flow_factory.samples import T2ISample
from flow_factory.trainers.abc import BaseTrainer


class EvaluationTrainer(BaseTrainer):
    def optimize(self, samples) -> None:
        del samples


class RewardBufferFake:
    def __init__(self) -> None:
        self.clears = 0
        self.samples = []

    def clear(self) -> None:
        self.clears += 1
        self.samples.clear()

    def add_samples(self, samples) -> None:
        self.samples.extend(samples)

    def finalize(self, *, store_to_samples: bool, split: str):
        assert store_to_samples is True
        assert split == "pointwise"
        return {"score": [1.0] * len(self.samples)}


def _trainer(buffer):
    logged = []
    reward_arguments = []
    waited = []
    trainer = object.__new__(EvaluationTrainer)
    trainer.eval_dataloaders = {"bench": [{"prompt": ["one"]}]}
    trainer.eval_dataset_reward_buffers = {"bench": buffer} if buffer is not None else {}
    trainer._eval_dataset_configs = {
        "bench": SimpleNamespace(
            source_id=0,
            eval=SimpleNamespace(get_merged_eval_kwargs=lambda eval_args: {}),
        )
    }
    trainer.eval_args = SimpleNamespace()
    trainer.training_args = SimpleNamespace(seed=42)
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.adapter = SimpleNamespace(
        eval=lambda: None,
        use_ema_parameters=nullcontext,
    )
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        is_main_process=True,
        is_local_main_process=True,
        gather=lambda tensor: tensor,
        wait_for_everyone=lambda: waited.append(True),
    )
    trainer.autocast = nullcontext

    def sample_batch(batch, reward_buffer=None, **kwargs):
        del batch, kwargs
        reward_arguments.append(reward_buffer)
        samples = [
            T2ISample(
                prompt="one",
                image=torch.zeros(3, 8, 8),
                height=8,
                width=8,
            )
        ]
        if reward_buffer is not None:
            reward_buffer.add_samples(samples)
        return samples

    trainer.sample_batch = sample_batch
    trainer.log_data = lambda data, step: logged.append((data, step))
    trainer.step = 3
    return trainer, logged, reward_arguments, waited


def test_evaluation_without_rewards_still_generates_and_logs_media() -> None:
    trainer, logged, reward_arguments, waited = _trainer(buffer=None)

    trainer.evaluate()

    assert reward_arguments == [None]
    assert len(logged) == 1
    assert list(logged[0][0]) == ["eval/bench/samples"]
    assert len(logged[0][0]["eval/bench/samples"]) == 1
    assert logged[0][1] == 3
    assert waited == [True]


def test_evaluation_with_rewards_keeps_metrics_and_media() -> None:
    buffer = RewardBufferFake()
    trainer, logged, reward_arguments, waited = _trainer(buffer=buffer)

    trainer.evaluate()

    assert reward_arguments == [buffer]
    assert buffer.clears == 1
    assert len(logged) == 1
    assert logged[0][0]["eval/bench/reward_score_mean"] == 1.0
    assert logged[0][0]["eval/bench/reward_score_std"] == 0.0
    assert len(logged[0][0]["eval/bench/samples"]) == 1
    assert waited == [True]
