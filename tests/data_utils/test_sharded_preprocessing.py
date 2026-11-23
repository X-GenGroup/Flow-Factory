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

from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FSDPModule

from flow_factory.data_utils.loader import (
    _create_or_load_dataset,
    coordinated_sharded_preprocessing,
    deferred_preprocessing_modules,
)
from flow_factory.hparams.component_execution import ComponentExecutionPolicy
from flow_factory.models.auxiliary_fsdp import (
    prepare_sharded_preprocessing,
    sharded_preprocessing_roots,
)
from flow_factory.models.component_execution import (
    ComponentExecution,
    ComponentExecutionPlan,
    OptimizationGradients,
)


class _Encoder:
    def __init__(self):
        self.module = torch.nn.Linear(1, 2).eval().requires_grad_(False)
        self.calls = []

    @torch.no_grad()
    def preprocess(self, prompt):
        self.calls.append(tuple(prompt))
        x = torch.tensor([[float(len(item))] for item in prompt])
        return {"embedding": list(self.module(x))}


def _ids(dataset):
    return [row["prompt"] for row in dataset.processed_dataset]


def _worker(rank, root, store):
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2)
    try:

        def gather(flags):
            if simulate_miss and rank == 1:
                flags.zero_()
            peers = [torch.empty_like(flags) for _ in range(2)]
            dist.all_gather(peers, flags)
            return torch.cat(peers)

        simulate_miss = False
        accelerator = SimpleNamespace(
            num_processes=2,
            process_index=rank,
            device=torch.device("cpu"),
            is_main_process=rank == 0,
            is_local_main_process=rank == 0,
            gather=gather,
            wait_for_everyone=dist.barrier,
        )
        torch.manual_seed(11)
        encoder = _Encoder()
        weight = encoder.module.weight.detach().clone()
        bias = encoder.module.bias.detach().clone()
        overrides = {}
        adapter = SimpleNamespace(
            accelerator=accelerator,
            component_execution_plan=ComponentExecutionPlan(
                {
                    "text_encoder": ComponentExecution(
                        ComponentExecutionPolicy(placement="fsdp2", shard_group="data"),
                        OptimizationGradients.NONE,
                    )
                },
                {},
            ),
            _require_component=lambda _: encoder.module,
            set_component=overrides.__setitem__,
        )
        roots = sharded_preprocessing_roots(adapter)
        assert roots == ("text_encoder",)

        def load():
            prepare_sharded_preprocessing(adapter, roots)

        def read(split):
            return _create_or_load_dataset(
                split,
                accelerator,
                {
                    "dataset_dir": root,
                    "cache_dir": f"{root}/cache",
                    "preprocess_func": encoder.preprocess,
                    "preprocessing_batch_size": 2,
                },
                enable_distributed=True,
                preprocess_parallelism="global",
            )

        with coordinated_sharded_preprocessing(load), deferred_preprocessing_modules(lambda: None):
            tiny = read("tiny")
            uneven = read("uneven")
            empty = read("empty")
            assert isinstance(overrides["text_encoder"], FSDPModule)
            assert _ids(tiny) == ["a"]
            assert _ids(uneven) == ["a", "bb", "ccc", "dddd", "eeeee"]
            for row in uneven.processed_dataset:
                x = torch.tensor([[float(len(row["prompt"]))]])
                torch.testing.assert_close(row["embedding"], (x @ weight.T + bias)[0])
            assert not _ids(empty)
            assert len(encoder.calls) == 5
            # A real partial batch reaches the encoder at its original width.
            if rank == 1:
                assert encoder.calls[-1] == ("eeeee",)
            encoder.calls.clear()
            assert _ids(read("uneven")) == _ids(uneven)
            assert encoder.calls == []

            # Simulate a peer that did not see the shared merged cache.
            simulate_miss = True
            assert _ids(read("uneven")) == _ids(uneven)
            assert len(encoder.calls) == 3

        assert all(not p.requires_grad and p.grad is None for p in encoder.module.parameters())
    finally:
        dist.destroy_process_group()


def test_cpu_sharded_preprocessing_cache_agreement_and_tails(tmp_path):
    (tmp_path / "tiny.txt").write_text(chr(10).join(("a", "")))
    (tmp_path / "uneven.txt").write_text(chr(10).join(("a", "bb", "ccc", "dddd", "eeeee", "")))
    (tmp_path / "empty.txt").write_text("")
    mp.spawn(_worker, args=(str(tmp_path), str(tmp_path / "store")), nprocs=2, join=True)
