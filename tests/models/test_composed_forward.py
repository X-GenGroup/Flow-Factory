# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""One prepared forward for composed predictions, with root-local proxy routing."""

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint

from flow_factory.models.model_bundle import ModelBundle, RoutedComponentProxy, composed_forward


class CountingBundle(ModelBundle):
    def __init__(self, members):
        super().__init__(members)
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return super().forward(*args, **kwargs)


def test_composition_uses_one_root_and_restores_proxy_dispatch_after_errors():
    member = torch.nn.Linear(3, 2)
    bundle = CountingBundle({"policy": member})
    proxy = RoutedComponentProxy(bundle, "policy", member)
    x = torch.ones(1, 3)
    actual = composed_forward(bundle, lambda: proxy(x) + proxy(x * 2))
    torch.testing.assert_close(actual, member(x) + member(x * 2))
    assert bundle.calls == 1

    def broken():
        proxy(x)
        raise ValueError("test exception")

    with pytest.raises(ValueError, match="test exception"):
        composed_forward(bundle, broken)
    proxy(x)
    assert bundle.calls == 3


def test_nested_compositions_restore_the_outer_root_and_preserve_foreign_routing():
    first = CountingBundle({"policy": torch.nn.Linear(3, 2)})
    second = CountingBundle({"policy": torch.nn.Linear(3, 2)})
    a = RoutedComponentProxy(first, "policy", first.members["policy"])
    b = RoutedComponentProxy(second, "policy", second.members["policy"])
    x = torch.ones(1, 3)

    def predict():
        initial = a(x)
        nested = composed_forward(second, lambda: b(x) + b(x))
        foreign = b(x)
        return initial + nested + foreign + a(x)

    composed_forward(first, predict)
    assert first.calls == 1
    assert second.calls == 2


class GuidedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Linear(3, 4)
        self.mix = torch.nn.Linear(4, 4)
        self.video = torch.nn.Linear(4, 2)
        self.audio = torch.nn.Linear(4, 2)

    def forward(self, x, *, isolate=False):
        hidden = checkpoint(self.stem, x, use_reentrant=False).tanh()
        if not isolate:
            hidden = hidden + checkpoint(self.mix, hidden, use_reentrant=False).tanh()
        return self.video(hidden), self.audio(hidden)


def _prediction(policy, x):
    video, audio = policy(x, isolate=False)
    isolated_video, isolated_audio = policy(x * 0.7, isolate=True)
    return video + 0.2 * isolated_video, audio + 0.2 * isolated_audio


def _worker(rank, store_path, output):
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 2), rank=rank, world_size=2)
    result = {"rank": rank}
    try:
        torch.manual_seed(17)
        member = GuidedPolicy().double()
        reference = copy.deepcopy(member)
        root = DistributedDataParallel(ModelBundle({"policy": member}), find_unused_parameters=True)
        proxy = RoutedComponentProxy(root, "policy", member)
        optimizer = torch.optim.SGD(root.parameters(), lr=0.03, weight_decay=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03, weight_decay=0.1)
        x = torch.arange(6, dtype=torch.float64).reshape(2, 3) / 10 + rank
        combined = torch.cat([x - rank, x - rank + 1])
        for objective in ("video", "audio", "mixed"):
            selected = {"video": 0, "audio": 1, "mixed": rank}[objective]
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            outputs = composed_forward(root, lambda: _prediction(proxy, x))
            outputs[selected].square().mean().backward()
            reference_outputs = _prediction(reference, combined)
            if objective == "mixed":
                reference_loss = (
                    reference_outputs[0][:2].square().mean()
                    + reference_outputs[1][2:].square().mean()
                ) / 2
            else:
                reference_loss = reference_outputs[selected].square().mean()
            reference_loss.backward()
            unused = {"video": "audio", "audio": "video", "mixed": None}[objective]
            unchanged = {
                name: value.detach().clone()
                for name, value in member.named_parameters()
                if unused is not None and name.startswith(unused)
            }
            for (name, actual), (_, expected) in zip(
                member.named_parameters(), reference.named_parameters(), strict=True
            ):
                assert (actual.grad is None) == (expected.grad is None), name
                if actual.grad is not None:
                    torch.testing.assert_close(actual.grad, expected.grad, atol=1e-12, rtol=1e-12)
                if unused is not None and name.startswith(unused):
                    assert actual.grad is None
            optimizer.step()
            reference_optimizer.step()
            for (name, actual), (_, expected) in zip(
                member.named_parameters(), reference.named_parameters(), strict=True
            ):
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                if name in unchanged:
                    torch.testing.assert_close(actual, unchanged[name], atol=0, rtol=0)
        result["status"] = "pass"
    except Exception:
        import traceback

        result["error"] = traceback.format_exc()
    finally:
        Path(output, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


def test_gloo_composed_guidance_matches_unpartitioned_gradients_with_dynamic_unused_heads(tmp_path):
    context = mp.get_context("spawn")
    workers = [
        context.Process(target=_worker, args=(rank, str(tmp_path / "store"), str(tmp_path)))
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(120)
    alive = [worker for worker in workers if worker.is_alive()]
    for worker in alive:
        worker.terminate()
    assert not alive
    assert all(worker.exitcode == 0 for worker in workers)
    rows = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert all(row.get("status") == "pass" for row in rows), rows


def test_separate_ddp_guidance_forwards_reproduce_premature_unused_marking(tmp_path):
    dist.init_process_group(
        "gloo", store=dist.FileStore(str(tmp_path / "store"), 1), rank=0, world_size=1
    )
    try:
        root = DistributedDataParallel(
            ModelBundle({"policy": GuidedPolicy()}), find_unused_parameters=True
        )
        proxy = RoutedComponentProxy(root, "policy", root.module.members["policy"])
        with pytest.raises(RuntimeError, match="mark a variable ready only once"):
            _prediction(proxy, torch.ones(2, 3))[0].square().mean().backward()
    finally:
        dist.destroy_process_group()
