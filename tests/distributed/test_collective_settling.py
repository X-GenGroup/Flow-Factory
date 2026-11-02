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

"""No collective may leave the plan undischarged.

A style that computes one placement and returns another ends in a collective,
and torch issues it with ``async_op=True``. ``DTensor.to_local()`` then hands
back ``local_tensor.view_as(local_tensor)`` *without* unwrapping the
``AsyncCollectiveTensor``, so what leaves the module is a promise whose buffer
is owned by the functional-collectives work registry -- not by the tensor's
refcount.

That is fine while the next op in the same forward consumes it, which is the
usage torch's default is written for. It stops being fine the moment a forward
can end without that consumption, and tensor parallelism puts us there:
``torch.utils.checkpoint``'s non-reentrant recompute runs with
``early_stop=True`` and aborts through ``_StopRecomputationError`` as soon as
backward's demands are met, leaving anything issued-and-unconsumed in flight for
the process's lifetime. A raised forward and a discarded branch open the same
hole.

Measured before the fix existed: GRPO on SD3.5-Medium at ``tp_size: 2`` grew
~4.5 GB per epoch and died in the third evaluation's VAE decode. The retained
buffers had **no Python referrer at all**, which is what named the registry as
the holder, and torch's own exit warning counted unwaited collectives in exact
multiples of the layer count.

The invariant asserted here is deliberately about the *obligation*, not the
value: a module may still hand out an ``AsyncCollectiveTensor``, because
replacing it would strip the autograd metadata its output carries. It must hand
out a **discharged** one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

DIM = 16
TOKENS = 8
BATCH = 2
ITERATIONS = 6


class _Block(nn.Module):
    """Column-parallel in, row-parallel out -- the shape that ends in a reduce."""

    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(DIM, DIM, bias=False)
        self.to_out = nn.Linear(DIM, DIM, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.to_out(F.gelu(self.to_q(hidden_states)))


class _Model(nn.Module):
    def __init__(self, blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(blocks)])

    def forward(self, hidden_states: torch.Tensor, checkpointed: bool) -> torch.Tensor:
        for block in self.blocks:
            if checkpointed:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block, hidden_states, use_reentrant=False
                )
            else:
                hidden_states = block(hidden_states)
        return hidden_states


def _plan(blocks: int) -> Dict[str, Any]:
    plan: Dict[str, Any] = {}
    for index in range(blocks):
        plan[f"blocks.{index}.to_q"] = ColwiseParallel()
        plan[f"blocks.{index}.to_out"] = RowwiseParallel()
    return plan


def _worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    from torch.distributed._functional_collectives import AsyncCollectiveTensor

    from flow_factory.models.tensor_parallel import apply_tensor_parallel

    tp_size = options["tp_size"]
    dist.init_process_group(
        "gloo", store=dist.FileStore(store_path, tp_size), rank=rank, world_size=tp_size
    )
    result: Dict[str, Any] = {"rank": rank}
    try:
        mesh = init_device_mesh("cpu", (tp_size,), mesh_dim_names=("tp",))
        torch.manual_seed(1234)
        model = _Model()
        applied = apply_tensor_parallel(model, mesh["tp"], _plan(len(model.blocks)))
        result["plan_entries"] = len(applied)

        # Read the obligation at the boundary. A hook registered now runs after
        # the one the applier installed, so it sees what downstream sees.
        seen: List[Dict[str, Any]] = []

        def observe(_module, _inputs, outputs):
            values = outputs if isinstance(outputs, tuple) else (outputs,)
            for value in values:
                if isinstance(value, AsyncCollectiveTensor):
                    seen.append({"async": True, "completed": bool(value.completed)})
                else:
                    seen.append({"async": False, "completed": True})

        for index in range(len(model.blocks)):
            model.get_submodule(f"blocks.{index}.to_out").register_forward_hook(observe)

        hidden = torch.randn(BATCH, TOKENS, DIM, generator=torch.Generator().manual_seed(7))
        hidden.requires_grad_(True)
        for _ in range(ITERATIONS):
            output = model(hidden, options["checkpointed"])
            output.float().pow(2).mean().backward()
            model.zero_grad(set_to_none=True)

        result["observations"] = len(seen)
        result["undischarged"] = [entry for entry in seen if not entry["completed"]]
        # The gradient has to survive: the first attempt at this fix returned
        # `wait()`'s value from the hook, which is the *unwrapped* tensor without
        # the output's autograd metadata, and backward died on it.
        result["input_grad_is_finite"] = bool(
            hidden.grad is not None and torch.isfinite(hidden.grad).all()
        )
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {str(error)[:300]}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


def _run(tmp_path: Path, **options: Any) -> List[Dict[str, Any]]:
    options.setdefault("checkpointed", True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = str(tmp_path / "store")
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(rank, store, str(tmp_path), options))
        for rank in range(options["tp_size"])
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=300)
    assert not [p for p in processes if p.is_alive()], "settling run exceeded 300 seconds"
    return [
        json.loads(Path(tmp_path, f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(options["tp_size"])
    ]


pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo backend unavailable",
)


@pytest.mark.parametrize("checkpointed", [True, False], ids=["checkpointed", "plain"])
def test_no_collective_leaves_a_module_boundary_undischarged(
    tmp_path: Path, checkpointed: bool
) -> None:
    """Every in-flight output is waited before it can escape the plan.

    Parameterized over activation checkpointing because that is what turned the
    latent hazard into a live leak, but the invariant is not conditional on it:
    the obligation has to be discharged either way, since a raised forward or a
    discarded branch leaves the same hole.
    """
    results = _run(tmp_path, tp_size=2, checkpointed=checkpointed)

    for result in results:
        assert "error" not in result, result["error"]
        # Two row-parallel boundaries x ITERATIONS, plus whatever recompute adds.
        assert result["observations"] >= 2 * ITERATIONS, result
        assert result["undischarged"] == [], result
        assert result["input_grad_is_finite"], result


def test_settling_does_not_replace_the_output_object(tmp_path: Path) -> None:
    """The fix is a side effect, and the gradient is how we know.

    Returning the waited value from the hook would strip the autograd metadata
    the module's output carries -- both `wait` and `trigger_wait` hand back the
    unwrapped `elem` -- and backward fails with "element 0 of tensors does not
    require grad and does not have a grad_fn". Asserting a finite input gradient
    pins the side-effect-only form.
    """
    results = _run(tmp_path, tp_size=2, checkpointed=True)

    for result in results:
        assert "error" not in result, result["error"]
        assert result["input_grad_is_finite"], result
