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

"""Real async workers must not split peers between share and error collectives."""

import json
import threading
import traceback
from datetime import timedelta
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks


def _worker(rank, store_path, result_dir, options):
    degree = options["tp_size"] * options["cp_size"]
    dist.init_process_group(
        "gloo",
        store=dist.FileStore(store_path, degree),
        rank=rank,
        world_size=degree,
        timeout=timedelta(seconds=10),
    )
    from accelerate.parallelism_config import ParallelismConfig
    from torch.distributed.device_mesh import DeviceMesh

    from flow_factory.hparams import RewardArguments
    from flow_factory.rewards import (
        PointwiseRewardModel,
        RewardBuffer,
        RewardModelOutput,
        RewardProcessor,
    )
    from flow_factory.samples import BaseSample
    from flow_factory.trainers import abc as trainer_module
    from flow_factory.trainers.abc import BaseTrainer

    release, started = threading.Event(), threading.Event()
    events = []
    result = {"rank": rank}
    buffer = None
    real_reduce = dist.all_reduce
    original_gather = trainer_module.gather_object
    mode = options["mode"]
    try:
        config = ParallelismConfig(tp_size=options["tp_size"], cp_size=options["cp_size"])
        if degree == 2:
            # A one-dimensional DeviceMesh normally reuses WORLD. Keep a distinct
            # communicator so a regression produces a bounded timeout, not a
            # mismatched dtype on the same communicator.
            group = dist.new_group(list(range(degree)), timeout=timedelta(seconds=10))
            axis = "tp" if options["tp_size"] > 1 else "cp"
            mesh = DeviceMesh.from_group(group, "cpu", mesh_dim_names=(axis,))
        else:
            mesh = config.build_device_mesh("cpu")
        groups = {
            axis: mesh[axis].get_group()
            for axis in ("tp", "cp")
            if getattr(config, f"{axis}_size") > 1
        }
        result["runtime"] = {
            "world_size": dist.get_world_size(),
            "process_group_backend": dist.get_backend(),
            "device": "cpu",
            "tp_size": config.tp_size,
            "cp_size": config.cp_size,
            "mesh": mesh.mesh.tolist(),
            "groups": {axis: dist.get_process_group_ranks(group) for axis, group in groups.items()},
        }

        def traced_reduce(tensor, *args, **kwargs):
            group = kwargs.get("group")
            label = next((axis for axis, value in groups.items() if group is value), "world")
            events.append(["reduce", label, str(tensor.dtype), list(tensor.shape)])
            return real_reduce(tensor, *args, **kwargs)

        dist.all_reduce = traced_reduce

        def reduce(tensor, reduction):
            assert reduction == "sum"
            reduced = tensor.clone()
            dist.all_reduce(reduced)
            return reduced

        def gather_object(payload):
            events.append(["error_details"])
            gathered = [None] * degree
            dist.all_gather_object(gathered, payload)
            return [item for peer in gathered for item in peer]

        trainer_module.gather_object = gather_object
        accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            process_index=rank,
            num_processes=degree,
            is_local_main_process=False,
            wait_for_everyone=lambda: None,
            reduce=reduce,
            parallelism_config=config,
            torch_device_mesh=mesh,
        )

        class ControlledReward(PointwiseRewardModel):
            required_fields = ("prompt",)

            def __call__(self, prompt):
                indices = [int(text) for text in prompt]
                if max(indices) >= degree:
                    started.set()
                    assert release.wait(30), "reward worker was not released"
                    if rank == 0 and mode in {"before_poll", "after_vote", "after_prepare"}:
                        raise ValueError("controlled unrelated reward failure")
                return RewardModelOutput(rewards=torch.tensor([float(i + 1) for i in indices]))

        reward_config = RewardArguments(
            name="score",
            device="cpu",
            async_reward=True,
            batch_size=1,
            num_workers=2,
            max_pending_requests=None,
        )
        processor = RewardProcessor(
            accelerator,
            {"score": ControlledReward(reward_config, accelerator)},
            {"score": reward_config},
            group_on_same_rank=True,
            verbose=False,
        )
        buffer = RewardBuffer(processor, group_size=degree)
        samples = [
            BaseSample(prompt=str(i), sampling_group_id=i // degree) for i in range(2 * degree)
        ]
        buffer.add_samples(samples)
        buffer.seal_for_streaming()
        assert [indices for _, indices, _ in buffer._futures] == [[rank], [rank + degree]]
        buffer._futures[0][2].result(timeout=30)
        assert started.wait(30)
        tail = buffer._futures[1][2]
        assert not tail.done()

        def fail_tail():
            release.set()
            assert isinstance(tail.exception(timeout=30), ValueError)

        trainer = SimpleNamespace(
            accelerator=accelerator,
            reward_buffer=buffer,
            advantage_processor=SimpleNamespace(
                prepare_group_reward_collection=lambda *a, **k: events.append(["prepare_advantage"])
            ),
            _reward_overlap_group_layouts_by_tile={},
            _compute_reward_overlap_advantages=lambda *a, **k: events.append(["advantage"]),
        )
        trainer._raise_reward_overlap_errors = MethodType(
            BaseTrainer._raise_reward_overlap_errors, trainer
        )

        def synchronize(phase, error):
            events.append(["guard", phase, error is not None])
            if phase == "tile reward preparation" and mode == "after_prepare" and rank == 0:
                # The selected payload is already stacked. A future completed
                # here must wait for the next poll, not diverge share collectives.
                fail_tail()
            BaseTrainer._synchronize_reward_overlap_error(trainer, phase, error)

        trainer._synchronize_reward_overlap_error = synchronize

        def poll(indices):
            error, ready = None, set()
            try:
                ready = buffer.poll_ready_tiles(indices)
            except Exception as caught:
                error = caught
            flags = torch.tensor(
                [int(0 in ready), int(set(indices) <= ready), int(error is not None)],
                dtype=torch.int32,
            )
            counts = accelerator.reduce(flags, reduction="sum")
            if int(counts[-1]):
                trainer._raise_reward_overlap_errors("reward polling", error)
            return counts

        if mode == "before_poll" and rank == 0:
            fail_tail()
        indices = tuple(range(degree))
        counts = poll({0: indices, 1: tuple(range(degree, 2 * degree))})
        assert counts.tolist() == [degree, 0, 0]
        if mode == "after_vote" and rank == 0:
            fail_tail()
        if mode == "stack" and rank == 0:
            buffer._rewards["score"][0] = torch.ones(2)
        if mode == "storage" and rank == 0:

            class FailedStorage(list):
                def __setitem__(self, key, value):
                    raise ValueError("controlled reward storage failure")

            buffer._rewards["score"] = FailedStorage(buffer._rewards["score"])
        tile = SimpleNamespace(tile_id=0, sample_indices=indices)
        BaseTrainer._resolve_reward_overlap_tile_feedback(trainer, tile, samples[:degree])
        result["rewards"] = [
            float(sample.extra_kwargs["rewards"]["score"]) for sample in samples[:degree]
        ]
        result["tail_pending"] = not tail.done()
        if mode == "after_prepare":
            poll({1: tuple(range(degree, 2 * degree))})
        result["status"] = "resolved"
    except Exception as error:
        result.update(status="error", error=str(error), traceback=traceback.format_exc())
    finally:
        release.set()
        if buffer is not None:
            buffer.abort_streaming()
            buffer.shutdown(wait=True)
            result["workers_stopped"] = all(
                not thread.is_alive()
                for executor in buffer._executors.values()
                for thread in executor._threads
            )
        dist.all_reduce = real_reduce
        trainer_module.gather_object = original_gather
        result["events"] = events
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize(
    "tp_size,cp_size,mode",
    [
        (1, 2, "healthy"),
        (1, 2, "before_poll"),
        (1, 2, "after_vote"),
        (1, 2, "stack"),
        (1, 2, "storage"),
        (1, 2, "after_prepare"),
        (2, 1, "after_vote"),
        (2, 2, "healthy"),
        (2, 2, "after_vote"),
    ],
)
def test_streaming_failures_preserve_collective_order(tmp_path, tp_size, cp_size, mode):
    degree = tp_size * cp_size
    rows = run_gloo_ranks(
        _worker,
        tmp_path,
        degree,
        {"tp_size": tp_size, "cp_size": cp_size, "mode": mode},
        timeout=60,
    )
    readiness = ["reduce", "world", "torch.int32", [3]]
    guard = ["reduce", "world", "torch.int32", [1]]
    shares = [
        ["reduce", axis, "torch.float64", [degree]]
        for axis, size in (("tp", tp_size), ("cp", cp_size))
        if size > 1
    ]
    if mode == "before_poll":
        expected = [readiness, ["error_details"]]
    elif mode in {"after_vote", "stack"}:
        expected = [readiness, guard, ["error_details"]]
    elif mode == "storage":
        expected = [readiness, guard, *shares, guard, ["error_details"]]
    else:
        expected = [readiness, guard, *shares, guard, guard]
        if mode == "after_prepare":
            expected += [readiness, ["error_details"]]
    for row in rows:
        assert row["workers_stopped"], row
        assert row["runtime"]["world_size"] == degree
        assert row["runtime"]["process_group_backend"] == "gloo"
        assert row["runtime"]["device"] == "cpu"
        assert (row["runtime"]["tp_size"], row["runtime"]["cp_size"]) == (tp_size, cp_size)
        assert torch.tensor(row["runtime"]["mesh"]).flatten().tolist() == list(range(degree))
        assert all(
            len(row["runtime"]["groups"][axis]) == size
            for axis, size in (("tp", tp_size), ("cp", cp_size))
            if size > 1
        )
        communication = [
            event for event in row["events"] if event[0] in {"reduce", "error_details"}
        ]
        assert communication == expected, row
        if mode == "healthy":
            assert row["status"] == "resolved", row
            assert row["rewards"] == list(range(1, degree + 1))
            assert row["tail_pending"]
        else:
            message = {
                "stack": "stack expects each tensor",
                "storage": "controlled reward storage failure",
            }.get(mode, "controlled unrelated reward failure")
            assert row["status"] == "error" and message in row["error"], row
        assert sum(event == ["advantage"] for event in row["events"]) == int(
            mode in {"healthy", "after_prepare"}
        )
    if mode != "healthy":
        assert len({row["error"] for row in rows}) == 1
