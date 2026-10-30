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

"""Group-shared rollout noise and replay noise."""

from types import SimpleNamespace

import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor

from flow_factory.data_utils.sampling_plan import (
    SAMPLING_GROUP_ID_COLUMN,
    SAMPLING_SAMPLE_ID_COLUMN,
)
from flow_factory.hparams import (
    AWMTrainingArguments,
    DPPOTrainingArguments,
    GRPOTrainingArguments,
    NFTTrainingArguments,
)
from flow_factory.samples import BaseSample, LatentState
from flow_factory.scheduler.sde_noise import (
    bind_sde_sample,
    rollout_sde_sample_ids,
    sde_step_generators,
    seeded_sde_transitions,
)
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.common.group_noise import (
    STREAM_FORWARD_NOISE,
    identity_timestep_generator,
    independent_batch_generator,
    init_same_noise_generators,
    record_noise_ids,
    require_planned_group_ids,
    sequenced_draw_generator,
    shared_group_noise,
    training_timestep_generator,
)
from flow_factory.trainers.rl.grpo import GRPOTrainer
from flow_factory.utils.noise_schedule import TimeSampler


def _trainer(*, init_same_noise: bool = False, seed: int = 7, epoch: int = 2):
    return SimpleNamespace(
        epoch=epoch,
        training_args=SimpleNamespace(init_same_noise=init_same_noise, seed=seed),
        adapter=SimpleNamespace(trajectory_component_order=("latent",)),
    )


def _batch(group_ids, source_ids=None, sample_ids=None):
    batch = {
        "prompt": [f"prompt-{index}" for index in range(len(group_ids))],
        SAMPLING_GROUP_ID_COLUMN: list(group_ids),
    }
    if sample_ids is not None:
        batch[SAMPLING_SAMPLE_ID_COLUMN] = list(sample_ids)
    if source_ids is not None:
        batch["__source_id__"] = list(source_ids)
    return batch


def _draw(generator: torch.Generator) -> torch.Tensor:
    return randn_tensor((4, 3), generator=generator, device="cpu", dtype=torch.float32)


def test_independent_initial_noise_follows_sample_id() -> None:
    trainer = _trainer()
    del trainer.training_args.init_same_noise
    planned = _batch([4, 4, 9], sample_ids=[40, 41, 90])
    missing_flag = init_same_noise_generators(trainer, planned)
    trainer.training_args.init_same_noise = False
    flagged = init_same_noise_generators(trainer, planned)
    swapped = init_same_noise_generators(trainer, _batch([9, 4, 4], sample_ids=[90, 40, 41]))
    planned_draws = [_draw(generator) for generator in flagged]
    swapped_draws = [_draw(generator) for generator in swapped]
    assert torch.equal(_draw(missing_flag[0]), planned_draws[0])
    assert torch.equal(planned_draws[0], swapped_draws[1])
    assert torch.equal(planned_draws[2], swapped_draws[0])
    assert not torch.equal(planned_draws[0], planned_draws[1])
    trainer.epoch = trainer.epoch + 1
    later = init_same_noise_generators(trainer, planned)
    assert not torch.equal(planned_draws[0], _draw(later[0]))


def test_init_same_noise_matches_within_a_group_and_across_calls() -> None:
    trainer = _trainer(init_same_noise=True)
    first = init_same_noise_generators(
        trainer,
        _batch([4, 4, 9], source_ids=[1, 1, 1], sample_ids=[1, 2, 3]),
    )
    second = init_same_noise_generators(trainer, _batch([4], source_ids=[1]))
    assert first is not None and second is not None

    same_group = _draw(first[0])
    assert torch.equal(same_group, _draw(first[1]))
    assert torch.equal(same_group, _draw(second[0]))
    assert not torch.equal(same_group, _draw(first[2]))


def test_init_same_noise_separates_equal_group_ids_from_different_sources() -> None:
    trainer = _trainer(init_same_noise=True)
    generators = init_same_noise_generators(trainer, _batch([4, 4], source_ids=[0, 1]))
    assert generators is not None
    assert not torch.equal(_draw(generators[0]), _draw(generators[1]))


def test_independent_initial_noise_requires_both_planned_ids() -> None:
    trainer = _trainer(init_same_noise=False)
    with pytest.raises(ValueError, match="both planned group ids and sample ids"):
        init_same_noise_generators(trainer, _batch([1, 1]))


def test_init_same_noise_requires_planned_group_ids() -> None:
    trainer = _trainer(init_same_noise=True)
    batch = _batch([1])
    del batch[SAMPLING_GROUP_ID_COLUMN]
    with pytest.raises(ValueError, match="planned sampling group id"):
        init_same_noise_generators(trainer, batch)


def test_grpo_sample_batch_shares_only_the_initial_latent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def record_batch(self, batch, reward_buffer=None, **kwargs):
        del self, batch, reward_buffer
        captured.update(kwargs)
        return []

    monkeypatch.setattr(BaseTrainer, "sample_batch", record_batch)
    trainer = object.__new__(GRPOTrainer)
    trainer.epoch = 0
    trainer.training_args = SimpleNamespace(init_same_noise=True, seed=3)
    GRPOTrainer.sample_batch(trainer, _batch([8, 8], source_ids=[2, 2]))

    generators = captured["generator"]
    assert "share_init_noise_only" not in captured
    assert torch.equal(_draw(generators[0]), _draw(generators[1]))

    captured.clear()
    GRPOTrainer.sample_batch(
        trainer,
        _batch([8, 8]),
        generator="caller",
    )
    assert captured["generator"] == "caller"
    assert "share_init_noise_only" not in captured


def test_grpo_sample_batch_seeds_independent_initial_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def record_batch(self, batch, reward_buffer=None, **kwargs):
        del self, batch, reward_buffer
        captured.update(kwargs)
        return []

    monkeypatch.setattr(BaseTrainer, "sample_batch", record_batch)
    trainer = object.__new__(GRPOTrainer)
    trainer.epoch = 0
    trainer.training_args = SimpleNamespace(init_same_noise=False, seed=3)
    GRPOTrainer.sample_batch(trainer, _batch([8, 8], sample_ids=[1, 2]))
    generators = captured["generator"]
    assert not torch.equal(_draw(generators[0]), _draw(generators[1]))


def test_init_same_noise_rejects_a_non_bool_flag() -> None:
    trainer = _trainer()
    trainer.training_args.init_same_noise = "true"
    with pytest.raises(TypeError, match="train.init_same_noise"):
        init_same_noise_generators(trainer, _batch([1]))


def test_coupled_arguments_accept_init_same_noise_and_reject_strings() -> None:
    assert GRPOTrainingArguments().init_same_noise is False
    assert DPPOTrainingArguments().init_same_noise is False
    with pytest.raises(TypeError, match="train.init_same_noise"):
        GRPOTrainingArguments(init_same_noise="false")
    with pytest.raises(TypeError, match="train.init_same_noise"):
        DPPOTrainingArguments(init_same_noise=1)


def test_rollout_sde_sample_ids_use_planned_ids_or_the_prompt() -> None:
    planned = rollout_sde_sample_ids(
        {"prompt": ["a red cube", "a blue sphere"], SAMPLING_SAMPLE_ID_COLUMN: [3, 4]}
    )
    assert planned == [3, 4]
    hashed = rollout_sde_sample_ids({"prompt": ["a red cube", "a blue sphere"]})
    assert hashed[0] != hashed[1]
    assert hashed == rollout_sde_sample_ids({"prompt": ["a red cube", "a blue sphere"]})


def test_decoupled_arguments_accept_use_shared_noise_and_reject_strings() -> None:
    assert NFTTrainingArguments().use_shared_noise is False
    assert AWMTrainingArguments().use_shared_noise is False
    with pytest.raises(TypeError, match="train.use_shared_noise"):
        NFTTrainingArguments(use_shared_noise="false")
    with pytest.raises(TypeError, match="train.use_shared_noise"):
        AWMTrainingArguments(use_shared_noise=1)


def _sample(group_id: int) -> BaseSample:
    return BaseSample(
        timesteps=torch.tensor([1000.0, 0.0]),
        all_latents=torch.zeros(1, 3),
        latent_index_map=torch.tensor([-1, -1, 0]),
        prompt_embeds=torch.tensor([float(group_id)]),
        sampling_group_id=group_id,
    )


def test_shared_replay_noise_is_redrawn_at_each_timestep() -> None:
    trainer = _trainer()
    samples = [_sample(4), _sample(4), _sample(9)]
    clean = LatentState({"latent": torch.zeros(3, 5)})

    first = shared_group_noise(trainer, clean, samples, inner_epoch=1, timestep_index=0)
    again = shared_group_noise(trainer, clean, samples, inner_epoch=1, timestep_index=0)
    later = shared_group_noise(trainer, clean, samples, inner_epoch=1, timestep_index=6)
    other_epoch = shared_group_noise(trainer, clean, samples, inner_epoch=2, timestep_index=0)

    noise = first.components["latent"]
    assert torch.equal(noise, again.components["latent"])
    assert not torch.equal(noise, later.components["latent"])
    assert torch.equal(noise[0], noise[1])
    assert not torch.equal(noise[0], noise[2])
    assert not torch.equal(noise, other_epoch.components["latent"])


def test_sde_step_generators_depend_on_the_sample_and_the_step() -> None:
    first = sde_step_generators(7, 0, [10, 11])
    again = sde_step_generators(7, 0, [10])
    nxt = sde_step_generators(7, 1, [10])
    assert torch.equal(_draw(first[0]), _draw(again[0]))
    assert not torch.equal(_draw(first[0]), _draw(first[1]))
    assert not torch.equal(_draw(first[0]), _draw(nxt[0]))
    legacy = _draw(sde_step_generators(7, 0, [10])[0])
    assert torch.equal(legacy, _draw(sde_step_generators(7, 0, [10], source_ids=[-1])[0]))
    assert not torch.equal(legacy, _draw(sde_step_generators(7, 0, [10], source_ids=[1])[0]))


def _scalar(generator: torch.Generator) -> float:
    return float(torch.randn((), generator=generator))


def test_seeded_sde_transitions_follow_batch_order_and_ignore_replay() -> None:
    seen: list[float] = []

    class Scheduler:
        def step(self, velocity, timestep, latents, next_latents=None, generator=None):
            del velocity, timestep
            if generator is not None:
                seen.extend(_scalar(gen) for gen in generator)
            return next_latents

    scheduler = Scheduler()
    adapter = SimpleNamespace(scheduler_group={"latent": scheduler})
    with seeded_sde_transitions(adapter, base_seed=3, sample_ids=[4, 9]):
        scheduler.step(None, None, torch.zeros(2, 1))
        scheduler.step(None, None, torch.zeros(2, 1), next_latents=torch.zeros(2, 1))
        scheduler.step(None, None, torch.zeros(2, 1))
    assert len(seen) == 4
    assert seen[0] != seen[1]
    assert seen[0] != seen[2]

    per_sample: list[float] = []

    class One:
        def step(self, velocity, timestep, latents, next_latents=None, generator=None):
            del velocity, timestep, next_latents, latents
            per_sample.append(_scalar(generator[0]))

    one = One()
    adapter = SimpleNamespace(scheduler_group={"latent": one})
    with seeded_sde_transitions(adapter, base_seed=3, sample_ids=[4, 9]):
        with bind_sde_sample(0):
            one.step(None, None, torch.zeros(1, 1))
            one.step(None, None, torch.zeros(1, 1))
        with bind_sde_sample(1):
            one.step(None, None, torch.zeros(1, 1))
    assert per_sample[0] != per_sample[1]
    assert per_sample[0] != per_sample[2]
    assert one.step.__func__ is One.step

    calls: list[str] = []

    class Video:
        def step(self, velocity, timestep, latents, next_latents=None, generator=None):
            del velocity, timestep, next_latents, latents
            calls.append("video")
            assert generator is not None

    class Audio:
        def step(self, velocity, timestep, latents, next_latents=None, generator=None):
            del velocity, timestep, next_latents, latents
            calls.append("audio")
            assert generator is not None

    pair = SimpleNamespace(scheduler_group={"video": Video(), "audio": Audio()})
    with seeded_sde_transitions(pair, base_seed=3, sample_ids=[4]):
        pair.scheduler_group["video"].step(None, None, torch.zeros(1, 1))
        pair.scheduler_group["audio"].step(None, None, torch.zeros(1, 1))
    assert calls == ["video", "audio"]

    with pytest.raises(RuntimeError, match="constraint #12b"):
        with seeded_sde_transitions(adapter, base_seed=3, sample_ids=[4, 9]):
            one.step(None, None, torch.zeros(2, 1), generator=torch.Generator())

    with pytest.raises(RuntimeError, match="middle of the inference batch"):
        with seeded_sde_transitions(adapter, base_seed=3, sample_ids=[4, 9]):
            one.step(None, None, torch.zeros(1, 1))


def test_training_timestep_generator_is_stable_and_changes_with_the_batch() -> None:
    trainer = SimpleNamespace(
        epoch=1,
        training_args=SimpleNamespace(seed=3),
        accelerator=SimpleNamespace(device=torch.device("cpu")),
        num_train_timesteps=4,
        time_shift=1.0,
        timestep_range=(0.0, 1.0),
        adapter=SimpleNamespace(scheduler=SimpleNamespace(timesteps=torch.linspace(1000, 0, 20))),
    )

    def draw(strategy: str, batch_index: int) -> torch.Tensor:
        trainer.time_sampling_strategy = strategy
        return BaseTrainer._sample_timesteps(
            trainer,
            2,
            generator=training_timestep_generator(trainer, inner_epoch=0, batch_index=batch_index),
        )

    for strategy in (
        "logit_normal",
        "uniform",
        "discrete",
        "discrete_with_init",
        "discrete_wo_init",
    ):
        first = draw(strategy, 0)
        again = draw(strategy, 0)
        other = draw(strategy, 1)
        assert torch.equal(first, again)
        assert not torch.equal(first, other)
        assert torch.unique(first[:, 0]).numel() > 1


def test_independent_noise_follows_the_sample_and_the_timestep() -> None:
    trainer = SimpleNamespace(epoch=1, training_args=SimpleNamespace(seed=3))
    first = independent_batch_generator(trainer, (10, 11), 0, STREAM_FORWARD_NOISE)
    again = independent_batch_generator(trainer, (10, 11), 0, STREAM_FORWARD_NOISE)
    other_sample = independent_batch_generator(trainer, (10, 12), 0, STREAM_FORWARD_NOISE)
    other_step = independent_batch_generator(trainer, (10, 11), 1, STREAM_FORWARD_NOISE)
    left = torch.randn(4, generator=first)
    assert torch.equal(left, torch.randn(4, generator=again))
    assert not torch.equal(left, torch.randn(4, generator=other_sample))
    assert not torch.equal(left, torch.randn(4, generator=other_step))
    assert not torch.equal(left[:2], left[2:])

    times = identity_timestep_generator(trainer, record_noise_ids(("a", "b")))
    repeat = identity_timestep_generator(trainer, record_noise_ids(("a", "b")))
    other_records = identity_timestep_generator(trainer, record_noise_ids(("a", "c")))
    drawn = torch.rand(4, generator=times)
    assert torch.equal(drawn, torch.rand(4, generator=repeat))
    assert not torch.equal(drawn, torch.rand(4, generator=other_records))

    sequenced = sequenced_draw_generator(trainer, STREAM_FORWARD_NOISE, (4,), 1)
    sequenced_again = sequenced_draw_generator(trainer, STREAM_FORWARD_NOISE, (4,), 1)
    assert not torch.equal(
        torch.randn(2, generator=sequenced),
        torch.randn(2, generator=sequenced_again),
    )


def test_shared_replay_noise_requires_planned_group_ids() -> None:
    sample = _sample(1)
    sample.sampling_group_id = None
    with pytest.raises(ValueError, match="sampling_group_id"):
        require_planned_group_ids([sample], context="DiffusionNFTTrainer use_shared_noise")
