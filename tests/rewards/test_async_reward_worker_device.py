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
