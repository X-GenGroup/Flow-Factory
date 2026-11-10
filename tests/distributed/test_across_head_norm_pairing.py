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

"""An attention's q and k across-head norms share one all-reduce.

Wan and LTX-2 normalize q and k over every head, so under tensor parallelism
each norm all-reduces a ``[batch, seq, 1]`` statistic, forward and backward.
The pairing computes both statistics in ``norm_q``'s call and hands ``norm_k``
its result. These tests pin the collective count, gradient parity against the
unsharded model -- for a self-attention and for a cross-attention, whose q and
k lengths differ -- the fallback when a processor orders the calls otherwise,
and that nothing is left held after a raised forward or a checkpointed
recompute.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from conftest import requires_gloo, run_gloo_ranks
from torch import nn

WORLD_SIZE = 2
DIM = 16
HEAD_DIM = 4
SEQ = 8
TEXT = 6


class _AcrossHeadAttention(nn.Module):
    """q/k normalized over the whole inner width before the head split, as Wan does."""

    def __init__(self, *, reverse: bool = False) -> None:
        super().__init__()
        self.heads = DIM // HEAD_DIM
        self.to_q = nn.Linear(DIM, DIM)
        self.to_k = nn.Linear(DIM, DIM)
        self.to_v = nn.Linear(DIM, DIM)
        self.norm_q = nn.RMSNorm(DIM, eps=1e-6)
        self.norm_k = nn.RMSNorm(DIM, eps=1e-6)
        self.to_out = nn.Linear(DIM, DIM)
        self.reverse = reverse
        self.fail_after_query = False

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query, key, value = self.to_q(hidden_states), self.to_k(context), self.to_v(context)
        if self.reverse:
            key = self.norm_k(key)
            query = self.norm_q(query)
        else:
            query = self.norm_q(query)
            if self.fail_after_query:
                raise RuntimeError("raised between the paired norms")
            key = self.norm_k(key)

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.unflatten(-1, (-1, HEAD_DIM)).transpose(1, 2)

        attended = F.scaled_dot_product_attention(heads(query), heads(key), heads(value))
        return self.to_out(attended.transpose(1, 2).flatten(-2))


class _Model(nn.Module):
    def __init__(self, *, reverse: bool = False, checkpoint: bool = False) -> None:
        super().__init__()
        self.embed = nn.Linear(DIM, DIM)
        self.text_embed = nn.Linear(DIM, DIM)
        self.self_attn = _AcrossHeadAttention(reverse=reverse)
        self.cross_attn = _AcrossHeadAttention(reverse=reverse)
        self.head = nn.Linear(DIM, DIM)
        self.checkpoint = checkpoint

    def forward(self, hidden_states: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(hidden_states)
        context = self.text_embed(text)
        hidden = hidden + self.self_attn(hidden)
        if self.checkpoint:
            # Non-reentrant, so the recompute can stop early once backward has
            # what it asked for -- the exit the pairing has to survive.
            hidden = hidden + torch.utils.checkpoint.checkpoint(
                self.cross_attn, hidden, context, use_reentrant=False
            )
        else:
            hidden = hidden + self.cross_attn(hidden, context)
        return self.head(hidden)


def _plan() -> Dict[str, Any]:
    from torch.distributed.tensor.parallel import RowwiseParallel

    from flow_factory.models.tensor_parallel import (
        ColumnParallelRegion,
        ShardedAcrossHeadRMSNorm,
        SharedInputColwiseParallel,
    )

    plan: Dict[str, Any] = {}
    for attention in ("self_attn", "cross_attn"):
        plan[attention] = ColumnParallelRegion(inputs=("hidden_states", "encoder_hidden_states"))
        for name in ("to_q", "to_k", "to_v"):
            plan[f"{attention}.{name}"] = SharedInputColwiseParallel()
        plan[f"{attention}.norm_q"] = ShardedAcrossHeadRMSNorm()
        plan[f"{attention}.norm_k"] = ShardedAcrossHeadRMSNorm()
        plan[f"{attention}.to_out"] = RowwiseParallel()
    return plan


def _build(**options: bool) -> nn.Module:
    torch.manual_seed(0)
    model = _Model(**options).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.3)
    return model


def _run(
    model: nn.Module, count: bool = True
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, int]]]:
    """Forward and backward, counting collectives unless ``count`` is off.

    ``CommDebugMode``'s module tracker cannot attribute a collective issued
    inside a checkpointed recompute, so the checkpointed case is not counted.
    """
    from contextlib import nullcontext

    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.debug import CommDebugMode

    generator = torch.Generator().manual_seed(5)
    hidden = torch.randn(2, SEQ, DIM, generator=generator, dtype=torch.float64)
    text = torch.randn(2, TEXT, DIM, generator=generator, dtype=torch.float64)
    forward_mode = CommDebugMode() if count else nullcontext()
    backward_mode = CommDebugMode() if count else nullcontext()
    with forward_mode:
        output = model(hidden, text)
    with backward_mode:
        output.pow(2).sum().backward()
    counts = (
        {
            phase: dict(
                Counter(
                    {str(op).split(".")[-1]: n for op, n in mode.get_comm_counts().items() if n}
                )
            )
            for phase, mode in (("forward", forward_mode), ("backward", backward_mode))
        }
        if count
        else {}
    )
    values = {"output": output.detach()}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        values[name] = (grad.full_tensor() if isinstance(grad, DTensor) else grad).detach()
    return values, counts


def _worst(expected: Dict[str, torch.Tensor], received: Dict[str, torch.Tensor]) -> float:
    assert set(expected) == set(received), sorted(set(expected) ^ set(received))
    return max(
        ((received[k] - expected[k]).abs().max() / expected[k].abs().max()).item()
        for k in expected
        if expected[k].abs().max() > 0
    )


def _held(model: nn.Module) -> list:
    """Whatever each attention's pairing still holds."""
    return [
        name
        for name, module in model.named_modules()
        if getattr(module, "_ff_norm_pair", None) is not None
        and (module._ff_norm_pair.key is not None or module._ff_norm_pair.result is not None)
    ]


def _pairing_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from flow_factory.models.parallelism import runtime as tensor_parallel
        from flow_factory.models.tensor_parallel import apply_tensor_parallel

        mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
        pair = tensor_parallel._pair_across_head_norms

        def case(paired: bool, **model_options: bool) -> Dict[str, Any]:
            tensor_parallel._pair_across_head_norms = pair if paired else (lambda *a: 0)
            count = not model_options.get("checkpoint", False)
            reference = _build(**model_options)
            expected, _ = _run(copy.deepcopy(reference), count=count)
            sharded = copy.deepcopy(reference)
            apply_tensor_parallel(sharded, mesh, _plan())
            assert (getattr(sharded.self_attn, "_ff_norm_pair", None) is not None) == paired
            received, counts = _run(sharded, count=count)
            return {"worst": _worst(expected, received), "counts": counts, "held": _held(sharded)}

        result["paired"] = case(True)
        result["unpaired"] = case(False)
        result["reversed"] = case(True, reverse=True)
        result["checkpointed"] = case(True, checkpoint=True)

        tensor_parallel._pair_across_head_norms = pair
        failing = _build()
        apply_tensor_parallel(failing, mesh, _plan())
        failing.self_attn.fail_after_query = True
        try:
            _run(failing)
            result["raised_held"] = "did not raise"
        except RuntimeError:
            result["raised_held"] = _held(failing)
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_paired_norms_share_one_all_reduce_and_match_the_unsharded_model(tmp_path: Path) -> None:
    results = run_gloo_ranks(_pairing_worker, tmp_path, WORLD_SIZE, {})
    for result in results:
        assert "error" not in result, result["error"]
        for name in ("paired", "unpaired", "reversed", "checkpointed"):
            assert result[name]["worst"] < 1e-12, (name, result[name])
            assert result[name]["held"] == [], (name, result[name])
        # The norms' statistics are c10d `allreduce_`s, two per attention
        # unpaired and one paired, in each direction. The functional
        # `all_reduce`s are untouched: `to_out`'s forward and, backward, the
        # region entry's lifts -- one stream for the self-attention, two for
        # the cross-attention.
        assert result["unpaired"]["counts"] == {
            "forward": {"allreduce_": 4, "all_reduce": 2},
            "backward": {"allreduce_": 4, "all_reduce": 3},
        }
        assert result["paired"]["counts"] == {
            "forward": {"allreduce_": 2, "all_reduce": 2},
            "backward": {"allreduce_": 2, "all_reduce": 3},
        }
        # Calling `norm_k` first finds no stored result and computes alone.
        assert result["reversed"]["counts"] == result["unpaired"]["counts"]
        assert result["raised_held"] == [], result["raised_held"]


def _wan_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from test_wan_sequence_parallel_parity import _build as build_wan
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.debug import CommDebugMode

        from flow_factory.models.parallelism import runtime as tensor_parallel
        from flow_factory.models.tensor_parallel import apply_tensor_parallel, resolve_parallel_plan
        from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter

        mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
        pair = tensor_parallel._pair_across_head_norms
        for paired in (True, False):
            tensor_parallel._pair_across_head_norms = pair if paired else (lambda *a: 0)
            model = build_wan()
            apply_tensor_parallel(
                model, mesh, resolve_parallel_plan(Wan2_T2V_Adapter, sequence_parallel=False)
            )
            assert sum(
                getattr(module, "_ff_norm_pair", None) is not None for module in model.modules()
            ) == (4 if paired else 0)
            generator = torch.Generator().manual_seed(11)
            latents = torch.randn(1, 4, 2, 4, 4, generator=generator).double()
            text = torch.randn(1, 5, 16, generator=generator).double()
            with CommDebugMode() as mode:
                output = model(
                    hidden_states=latents,
                    timestep=torch.tensor([600.0], dtype=torch.float64),
                    encoder_hidden_states=text,
                    return_dict=False,
                )[0]
                output.pow(2).sum().backward()
            counts = mode.get_comm_counts()
            result["paired" if paired else "unpaired"] = sum(
                n for op, n in counts.items() if str(op).endswith(("all_reduce", "allreduce_"))
            )
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_wan_pairs_both_attentions_of_every_block(tmp_path: Path) -> None:
    """Two blocks of self- and cross-attention: four norm pairs, two directions."""
    results = run_gloo_ranks(_wan_worker, tmp_path, WORLD_SIZE, {})
    for result in results:
        assert "error" not in result, result["error"]
        assert result["unpaired"] - result["paired"] == 4 * 2, result


class _NormBoundary(nn.Module):
    """Expose the same q/key call ordering without projection rounding."""

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.to_k = nn.Identity()
        self.norm_q = nn.RMSNorm(32, eps=1e-6, dtype=dtype)
        self.norm_k = nn.RMSNorm(32, eps=1e-6, dtype=dtype)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple:
        key = self.to_k(key)
        return self.norm_q(query), self.norm_k(key)


def _low_precision_worker(rank: int, store_path: str, result_dir: str, options: dict) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result = {"rank": rank, "cases": []}
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from flow_factory.models.tensor_parallel import (
            ShardedAcrossHeadRMSNorm,
            _install_norm_pair,
        )

        mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
        for dtype in (torch.float16, torch.bfloat16):
            for paired in (False, True):
                torch.manual_seed(481)
                model = _NormBoundary(dtype)
                weights = []
                for norm in (model.norm_q, model.norm_k):
                    with torch.no_grad():
                        norm.weight.uniform_(0.5, 1.5)
                    weights.append(norm.weight.detach().double().clone().requires_grad_())
                    ShardedAcrossHeadRMSNorm()._apply(norm, mesh)
                if paired:
                    _install_norm_pair(model, model.to_k, mesh.get_group())
                inputs, upstreams, references = [], [], []
                for length, weight in zip((8, 6), weights):
                    value = torch.randn(2, length, 32).to(dtype)
                    # Nearly radial derivatives subtract two large terms. Both
                    # branches must meet in FP32 before one low-precision cast.
                    upstream = (value.double() / weight + 0.01 * torch.randn_like(value)).to(dtype)
                    reference = value.double().requires_grad_()
                    output = F.rms_norm(reference, (32,), weight, eps=1e-6)
                    output.backward(upstream.double())
                    references.append((output.detach(), reference.grad, weight.grad))
                    inputs.append(value.chunk(WORLD_SIZE, -1)[rank].detach().requires_grad_())
                    upstreams.append(upstream.chunk(WORLD_SIZE, -1)[rank])
                outputs = model(*inputs)
                torch.autograd.backward(outputs, upstreams)
                residuals = []
                for value, output, (reference, gradient, weight_gradient), norm in zip(
                    inputs, outputs, references, (model.norm_q, model.norm_k)
                ):
                    reference = reference.chunk(WORLD_SIZE, -1)[rank]
                    gradient = gradient.chunk(WORLD_SIZE, -1)[rank]
                    torch.testing.assert_close(output.double(), reference, rtol=1 / 128, atol=1e-6)
                    torch.testing.assert_close(
                        value.grad.double(), gradient, rtol=1 / 128, atol=2e-6
                    )
                    residual = float((value.grad.double() - gradient).norm() / gradient.norm())
                    assert residual < 1 / 128, residual
                    torch.testing.assert_close(
                        norm.weight.grad.to_local().double(),
                        weight_gradient.chunk(WORLD_SIZE)[rank],
                        rtol=1 / 128,
                        atol=2e-6,
                    )
                    residuals.append(residual)
                assert _held(model) == []
                result["cases"].append(
                    {"dtype": str(dtype), "paired": paired, "residuals": residuals}
                )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_low_precision_norm_gradients_round_after_branch_accumulation(tmp_path: Path) -> None:
    results = run_gloo_ranks(_low_precision_worker, tmp_path, WORLD_SIZE, {})
    for result in results:
        assert "error" not in result, result.get("error")
        assert len(result["cases"]) == 4
