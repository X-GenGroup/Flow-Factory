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

"""CUDA checks for seeded noise: a rerun matches, and distinct keys stay distinct.

Run the single-GPU section directly, then the two-rank section under torchrun.
Both sections exit non-zero on the first failed check.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
from diffusers.utils.torch_utils import randn_tensor

from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler, MiniMaxH3SDEScheduler
from flow_factory.scheduler.sde_noise import seeded_sde_transitions
from flow_factory.trainers.common.group_noise import (
    SEED_TAG_SHARED_NOISE,
    STREAM_CONDITIONAL_FRESH,
    STREAM_FORWARD_NOISE,
    STREAM_QUERY_TIME,
    draw_group_component_noise,
    identity_timestep_generator,
    independent_batch_generator,
    init_same_noise_generators,
    record_noise_ids,
    sample_noise_ids,
    sequenced_draw_generator,
)
from flow_factory.utils.base import create_generator
from flow_factory.utils.noise_schedule import TimeSampler


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("noise contract GPU checks require CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _check(name: str, condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}", flush=True)


def _flow_scheduler() -> FlowMatchEulerDiscreteSDEScheduler:
    scheduler = FlowMatchEulerDiscreteSDEScheduler(noise_level=0.7, dynamics_type="Flow-SDE")
    scheduler.set_timesteps(4, device="cuda")
    scheduler.train()
    return scheduler


def _sde_noise(
    scheduler: FlowMatchEulerDiscreteSDEScheduler, latents: torch.Tensor
) -> torch.Tensor:
    velocity = torch.zeros_like(latents)
    output = scheduler.step(velocity, scheduler.timesteps[0], latents, compute_log_prob=False)
    if output.next_latents is None:
        raise RuntimeError("SDE step did not return next_latents")
    return output.next_latents


def _trainer(seed: int = 42, epoch: int = 0) -> SimpleNamespace:
    return SimpleNamespace(epoch=epoch, training_args=SimpleNamespace(seed=seed))


def _pairwise_std(rows: torch.Tensor) -> float:
    if rows.shape[0] < 2:
        raise ValueError(f"expected at least 2 rows, received {rows.shape[0]}")
    return float((rows[0] - rows[1]).float().pow(2).mean().sqrt())


def single_gpu() -> None:
    device = _require_cuda()
    trainer = _trainer()

    class _Adapter:
        scheduler_group = {"latent": _flow_scheduler()}

    adapter = _Adapter()
    latents = torch.zeros(2, 4, device=device)
    with seeded_sde_transitions(adapter, base_seed=42, sample_ids=[10, 11]):
        first = _sde_noise(adapter.scheduler_group["latent"], latents).cpu()
    adapter.scheduler_group["latent"] = _flow_scheduler()
    with seeded_sde_transitions(adapter, base_seed=42, sample_ids=[10, 11]):
        second = _sde_noise(adapter.scheduler_group["latent"], latents).cpu()
    _check(
        "grpo-sde-rerun",
        torch.equal(first, second),
        "same sample ids and seed produced different SDE latents",
    )
    _check(
        "grpo-sde-members-differ",
        not torch.equal(first[0], first[1]),
        "two sample ids drew the same SDE latent",
    )

    adapter.scheduler_group["latent"] = _flow_scheduler()
    with seeded_sde_transitions(adapter, base_seed=42, sample_ids=[10]):
        step0 = _sde_noise(adapter.scheduler_group["latent"], latents[:1]).cpu()
        step1 = _sde_noise(adapter.scheduler_group["latent"], latents[:1]).cpu()
    _check(
        "grpo-sde-steps-differ",
        not torch.equal(step0, step1),
        "adjacent SDE steps reused one latent",
    )

    video = MiniMaxH3SDEScheduler(shift=3.0, dynamics_type="Flow-SDE", noise_level=0.7)
    audio = MiniMaxH3SDEScheduler(shift=3.0, dynamics_type="Flow-SDE", noise_level=0.7)
    for scheduler in (video, audio):
        scheduler.set_timesteps(sigmas=[1.0, 0.6, 0.2, 0.0])
        scheduler.train()

    class _H3:
        scheduler_group = {"video": video, "audio": audio}

    h3_latent = torch.zeros(1, 2, device=device)
    sigma = torch.tensor(0.6)
    sigma_next = torch.tensor(0.2)

    def _h3_pair(adapter_h3: _H3) -> tuple[torch.Tensor, torch.Tensor]:
        with seeded_sde_transitions(adapter_h3, base_seed=42, sample_ids=[7]):
            video_out = adapter_h3.scheduler_group["video"].step(
                torch.zeros_like(h3_latent),
                adapter_h3.scheduler_group["video"].timesteps[1],
                h3_latent,
                sigma=sigma,
                sigma_next=sigma_next,
                compute_log_prob=False,
            )
            audio_out = adapter_h3.scheduler_group["audio"].step(
                torch.zeros_like(h3_latent),
                adapter_h3.scheduler_group["audio"].timesteps[1],
                h3_latent,
                sigma=sigma,
                sigma_next=sigma_next,
                compute_log_prob=False,
            )
        return video_out.next_latents.cpu(), audio_out.next_latents.cpu()

    h3_first = _h3_pair(_H3())
    h3_second = _h3_pair(_H3())
    _check("h3-rerun-video", torch.equal(h3_first[0], h3_second[0]), "video SDE rerun diverged")
    _check("h3-rerun-audio", torch.equal(h3_first[1], h3_second[1]), "audio SDE rerun diverged")
    _check(
        "h3-modalities-differ",
        not torch.equal(h3_first[0], h3_first[1]),
        "video and audio SDE steps reused one noise",
    )

    shape = torch.Size([8])
    feat_t0 = draw_group_component_noise(
        trainer,
        unique_id=3,
        component_name="latent",
        component_index=0,
        shape=shape,
        dtype=torch.float32,
        device=device,
        inner_epoch=0,
        timestep_index=0,
    )
    feat_t0_again = draw_group_component_noise(
        trainer,
        unique_id=3,
        component_name="latent",
        component_index=0,
        shape=shape,
        dtype=torch.float32,
        device=device,
        inner_epoch=0,
        timestep_index=0,
    )
    feat_t1 = draw_group_component_noise(
        trainer,
        unique_id=3,
        component_name="latent",
        component_index=0,
        shape=shape,
        dtype=torch.float32,
        device=device,
        inner_epoch=0,
        timestep_index=1,
    )
    other_group = draw_group_component_noise(
        trainer,
        unique_id=4,
        component_name="latent",
        component_index=0,
        shape=shape,
        dtype=torch.float32,
        device=device,
        inner_epoch=0,
        timestep_index=0,
    )
    _check("dgpo-rerun", torch.equal(feat_t0, feat_t0_again), "shared group noise rerun diverged")
    _check(
        "dgpo-timesteps-differ",
        not torch.equal(feat_t0, feat_t1),
        "feat reused one group noise across timesteps",
    )
    _check("dgpo-groups-differ", not torch.equal(feat_t0, other_group), "two groups shared noise")

    # origin/main DGPO omitted timestep_index, so every training step reused one tensor.
    main_generator = create_generator(
        trainer.training_args.seed,
        trainer.epoch,
        0,
        3,
        SEED_TAG_SHARED_NOISE,
        device=device,
    )
    main_noise = randn_tensor(shape, generator=main_generator, device=device, dtype=torch.float32)
    main_again = randn_tensor(
        shape,
        generator=create_generator(
            trainer.training_args.seed,
            trainer.epoch,
            0,
            3,
            SEED_TAG_SHARED_NOISE,
            device=device,
        ),
        device=device,
        dtype=torch.float32,
    )
    _check(
        "main-dgpo-is-timestep-invariant",
        torch.equal(main_noise, main_again),
        "the reconstructed main seed did not repeat",
    )
    _check(
        "feat-dgpo-is-more-diverse-across-time",
        float((feat_t0 - feat_t1).float().pow(2).mean().sqrt()) > 0,
        "feat cross-timestep distance was zero",
    )

    ids = sample_noise_ids(
        [
            SimpleNamespace(sampling_sample_id=10),
            SimpleNamespace(sampling_sample_id=11),
        ]
    )
    noise_a = randn_tensor(
        (2, 4),
        generator=independent_batch_generator(trainer, ids, 0, STREAM_FORWARD_NOISE),
        device=device,
    )
    noise_b = randn_tensor(
        (2, 4),
        generator=independent_batch_generator(trainer, ids, 0, STREAM_FORWARD_NOISE),
        device=device,
    )
    noise_step = randn_tensor(
        (2, 4),
        generator=independent_batch_generator(trainer, ids, 1, STREAM_FORWARD_NOISE),
        device=device,
    )
    noise_other = randn_tensor(
        (2, 4),
        generator=independent_batch_generator(trainer, (10, 12), 0, STREAM_FORWARD_NOISE),
        device=device,
    )
    _check("independent-rerun", torch.equal(noise_a, noise_b), "independent noise rerun diverged")
    _check(
        "independent-timesteps-differ",
        not torch.equal(noise_a, noise_step),
        "independent noise repeated across timesteps",
    )
    _check(
        "independent-samples-differ",
        not torch.equal(noise_a, noise_other),
        "a different sample id reused the noise",
    )
    _check(
        "independent-rows-differ",
        _pairwise_std(noise_a) > 0,
        "one batch draw collapsed its rows",
    )

    record_ids = record_noise_ids(("record-a", "record-b"))
    times = TimeSampler.independent_uniform(
        batch_size=2,
        num_timesteps=2,
        timestep_range=(0.0, 0.99),
        device=device,
        generator=identity_timestep_generator(trainer, record_ids),
    )
    times_again = TimeSampler.independent_uniform(
        batch_size=2,
        num_timesteps=2,
        timestep_range=(0.0, 0.99),
        device=device,
        generator=identity_timestep_generator(trainer, record_ids),
    )
    times_other = TimeSampler.independent_uniform(
        batch_size=2,
        num_timesteps=2,
        timestep_range=(0.0, 0.99),
        device=device,
        generator=identity_timestep_generator(trainer, record_noise_ids(("record-a", "record-c"))),
    )
    _check("sft-time-rerun", torch.equal(times, times_again), "offline times rerun diverged")
    _check(
        "sft-time-records-differ",
        not torch.equal(times, times_other),
        "different record ids reused coordinates",
    )
    _check(
        "sft-time-matrix-diverse",
        not torch.equal(times[0], times[1]) and not torch.equal(times[:, 0], times[:, 1]),
        "offline time matrix collapsed",
    )

    init_batch = {
        "prompt": ["a", "a", "b"],
        "__sampling_group_id__": [4, 4, 9],
    }
    init_trainer = SimpleNamespace(
        epoch=0,
        training_args=SimpleNamespace(seed=42, init_same_noise=True),
    )
    generators = init_same_noise_generators(init_trainer, init_batch)
    if generators is None:
        raise RuntimeError("init_same_noise did not build generators")
    x0 = randn_tensor((4,), generator=generators[0], device=device)
    x1 = randn_tensor((4,), generator=generators[1], device=device)
    x2 = randn_tensor((4,), generator=generators[2], device=device)
    _check("init-same-noise-group", torch.equal(x0, x1), "group members did not share x_T")
    _check("init-same-noise-other-group", not torch.equal(x0, x2), "two groups shared x_T")

    query_a = torch.rand(4, generator=sequenced_draw_generator(trainer, STREAM_QUERY_TIME, ids, 1))
    query_b = torch.rand(4, generator=sequenced_draw_generator(trainer, STREAM_QUERY_TIME, ids, 1))
    fresh_a = torch.rand(
        4, generator=sequenced_draw_generator(trainer, STREAM_CONDITIONAL_FRESH, ids, 1)
    )
    trainer._independent_draw_counts = {}
    query_restart = torch.rand(
        4, generator=sequenced_draw_generator(trainer, STREAM_QUERY_TIME, ids, 1)
    )
    _check(
        "tdm-repeated-query-differs",
        not torch.equal(query_a, query_b),
        "a second TDM query reused the first draw",
    )
    _check(
        "tdm-streams-differ",
        not torch.equal(query_a, fresh_a),
        "query time and fresh noise shared a stream",
    )
    _check(
        "tdm-restart-matches",
        torch.equal(query_a, query_restart),
        "resetting the draw counter did not repeat the first query",
    )


def distributed() -> None:
    import torch.distributed as dist

    device = _require_cuda()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    trainer = _trainer()
    shape = torch.Size([16])
    local_shared = draw_group_component_noise(
        trainer,
        unique_id=3,
        component_name="latent",
        component_index=0,
        shape=shape,
        dtype=torch.float32,
        device=device,
        inner_epoch=0,
        timestep_index=0,
    )
    gathered = [torch.empty_like(local_shared) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_shared)
    if rank == 0:
        shared_cpu = [value.cpu() for value in gathered]
        _check(
            "cross-rank-shared-noise",
            all(torch.equal(shared_cpu[0], other) for other in shared_cpu[1:]),
            "ranks with the same group key drew different noise",
        )
    local_private = randn_tensor(
        (8,),
        generator=independent_batch_generator(
            trainer,
            (1000 + rank,),
            0,
            STREAM_FORWARD_NOISE,
        ),
        device=device,
    )
    gathered_private = [torch.empty_like(local_private) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_private, local_private)
    if rank == 0:
        private_cpu = [value.cpu() for value in gathered_private]
        _check(
            "cross-rank-independent-noise",
            not torch.equal(private_cpu[0], private_cpu[1]),
            "different ranks reused one independent noise tensor",
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        distributed()
    else:
        single_gpu()
        print("SINGLE_GPU_OK", flush=True)
