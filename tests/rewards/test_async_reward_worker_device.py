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

"""Async reward workers stay on their reward model's device."""

import subprocess
import sys
import textwrap

import pytest
import torch

# Runs in a fresh interpreter: any earlier CUDA use in the pytest process may already
# hold a context on device 0, which would hide the one the worker creates.
_PROBE = textwrap.dedent("""
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import gc
    import weakref

    import torch

    torch.cuda.set_device(1)
    from flow_factory.rewards.reward_processor import RewardBuffer

    device = torch.device("cuda", 1)
    stream = torch.cuda.Stream(device=device)
    on_reward_stream = []

    def compute(name, model, samples):
        on_reward_stream.append(torch.cuda.current_stream() == stream)
        return torch.ones(len(samples), device=device)

    buffer = SimpleNamespace(
        _reward_streams={device: stream},
        rp=SimpleNamespace(_compute_pointwise_batch=compute),
    )
    model = SimpleNamespace(device=device)
    with ThreadPoolExecutor(max_workers=1) as pool:
        rewards = pool.submit(
            RewardBuffer._execute_task, buffer, "pointwise", "probe", model, [0, 1], None
        ).result()
    torch.cuda.synchronize(device)

    if rewards.tolist() != [1.0, 1.0] or on_reward_stream != [True]:
        raise AssertionError(
            f"expected rewards [1.0, 1.0] computed on the reward stream, got "
            f"{rewards.tolist()} with on_reward_stream={on_reward_stream}"
        )
    if torch._C._cuda_hasPrimaryContext(0):
        raise AssertionError(
            "the async reward worker created a CUDA context on device 0 while its reward "
            "model is on cuda:1"
        )

    # Exercise the owner too: clearing a finite acquisition must reuse the same
    # CUDA worker and release samples, rather than accumulate threads or tensors.
    from flow_factory.hparams import RewardArguments
    from flow_factory.rewards import PointwiseRewardModel, RewardModelOutput, RewardProcessor
    from flow_factory.samples import BaseSample

    class TensorReward(PointwiseRewardModel):
        required_fields = ("prompt",)

        def __call__(self, prompt):
            assert torch.cuda.current_device() == 1
            return RewardModelOutput(rewards=torch.ones(len(prompt), device=device))

    accelerator = SimpleNamespace(
        device=device, process_index=0, num_processes=1,
        is_local_main_process=False, wait_for_everyone=lambda: None,
    )
    args = RewardArguments(
        name="score", device=device, async_reward=True, batch_size=1,
        num_workers=1, max_pending_requests=1,
    )
    model = TensorReward(args, accelerator)
    owner = RewardBuffer(RewardProcessor(
        accelerator, {"score": model}, {"score": args}, verbose=False,
    ), group_size=1)
    allocated = []
    worker_threads = None
    try:
        for cycle in range(5):
            sample = BaseSample(prompt=str(cycle), all_latents=torch.ones(4096, device=device))
            sample_ref = weakref.ref(sample)
            owner.add_samples([sample])
            result = owner.finalize()
            assert result["score"].tolist() == [1.0]
            del sample, result
            owner.clear()
            gc.collect()
            torch.cuda.synchronize(device)
            assert sample_ref() is None
            threads = frozenset(owner._executors["score"]._threads)
            assert len(threads) == 1
            if worker_threads is not None:
                assert threads == worker_threads
            worker_threads = threads
            assert not torch._C._cuda_hasPrimaryContext(0)
            allocated.append(torch.cuda.memory_allocated(device))
        assert max(allocated) - min(allocated) <= 4096, allocated
    finally:
        owner.shutdown(wait=True)
    assert all(not thread.is_alive() for thread in worker_threads)
    assert not owner._reward_streams
    """)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_async_reward_worker_creates_no_context_on_device_zero() -> None:
    """A worker serving a cuda:1 reward must not initialize device 0.

    Pool threads start on device 0, so each rank whose reward model is elsewhere used to
    leave an idle CUDA context (~500 MiB) on GPU 0.
    """
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-4000:]
