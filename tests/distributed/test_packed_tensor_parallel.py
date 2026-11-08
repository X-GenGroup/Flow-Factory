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

"""Canonical packed TP, nonlinear derivatives, optimizer state and fresh replay."""

from __future__ import annotations

import copy
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from conftest import requires_gloo, run_gloo_ranks
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, PrepareModuleOutput, RowwiseParallel
from torch.utils.checkpoint import checkpoint

from flow_factory.models.packed_tensor_parallel import (
    PackedColwiseParallel,
    PackedRowwiseParallel,
    _PackedExchange,
)
from flow_factory.models.tensor_parallel import apply_tensor_parallel, peel_peft_wrapper


class _PackedBlock(nn.Module):
    """Real attention and unequal SwiGLU outputs share the native linear packs."""

    def __init__(self, checkpointed: bool) -> None:
        super().__init__()
        self.start = nn.Identity()
        self.gather = nn.Identity()
        self.up = nn.Linear(24, 264)
        self.down = nn.Linear(120, 24)
        self.up._tp_packed_col_blocks = [24, 24, 24, 96, 96]
        self.down._tp_packed_row_blocks = [24, 96]
        self.end = nn.Identity()
        self.checkpointed = checkpointed

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.up(self.gather(x))
        width = projected.shape[-1] // 11
        query, key, value, gate, up = projected.split([width] * 3 + [width * 4] * 2, -1)
        query, key, value = [
            operand.unflatten(-1, (-1, 6)).transpose(1, 2) for operand in (query, key, value)
        ]
        attended = F.scaled_dot_product_attention(query, key, value).transpose(1, 2).flatten(-2)
        return x + self.down(torch.cat((attended, F.silu(gate) * up), -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.start(x)
        output = (
            checkpoint(self._compute, x, use_reentrant=False)
            if self.checkpointed
            else self._compute(x)
        )
        return self.end(output)


def _build(lora: bool, checkpointed: bool) -> nn.Module:
    torch.manual_seed(142)
    model = _PackedBlock(checkpointed).double()
    if lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, target_modules=["up", "down"]))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    parameter.uniform_(-0.08, 0.08)
    return model


def _plan(sp: bool) -> dict:
    plan = {
        "up": PackedColwiseParallel(None if sp else (1, 1, 1, 4, 4)),
        "down": PackedRowwiseParallel(
            None if sp else (1, 4), output_layouts=Shard(1) if sp else None
        ),
    }
    if sp:
        plan.update(
            start=PrepareModuleOutput(output_layouts=Replicate(), desired_output_layouts=Shard(1)),
            gather=PrepareModuleOutput(output_layouts=Shard(1), desired_output_layouts=Replicate()),
            end=PrepareModuleOutput(output_layouts=Shard(1), desired_output_layouts=Replicate()),
        )
    return plan


def _whole(value: torch.Tensor) -> torch.Tensor:
    return value.full_tensor() if isinstance(value, DTensor) else value


def _operand(step: int, replica: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(29 + 17 * step + replica)
    x = torch.randn(2, 8, 24, dtype=torch.float64, generator=generator)
    target = torch.randn(x.shape, dtype=x.dtype, generator=generator)
    return x, target


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-9, msg=label)


def _assert_parameters(
    model: nn.Module, reference: nn.Module, optimizer, reference_optimizer
) -> int:
    compared = 0
    for name, parameter in model.named_parameters():
        expected = dict(reference.named_parameters())[name]
        _assert_close(_whole(parameter), expected, name)
        if parameter.requires_grad:
            for key in ("exp_avg", "exp_avg_sq"):
                _assert_close(
                    _whole(optimizer.state[parameter][key]),
                    reference_optimizer.state[expected][key],
                    name + key,
                )
            compared += 1
    return compared


def _state(model: nn.Module, optimizer) -> dict:
    options = StateDictOptions(full_state_dict=False)
    return {
        "model": get_model_state_dict(model, options=options),
        "optimizer": get_optimizer_state_dict(model, optimizer, options=options),
    }


def _worker(rank: int, store_path: str, result_dir: str, options: dict) -> None:
    dp_size = options["dp_size"]
    cuda = options.get("cuda", False)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        store=dist.FileStore(store_path, dp_size * 2),
        rank=rank,
        world_size=dp_size * 2,
    )
    result = {"rank": rank, **options}
    try:
        mesh = init_device_mesh(device.type, (dp_size, 2), mesh_dim_names=("dp_shard", "tp"))
        assert mesh["tp"].get_local_rank() == rank % 2
        assert mesh["dp_shard"].get_local_rank() == rank // 2
        result["runtime"] = {
            "backend": dist.get_backend(),
            "tp_size": 2,
            "tp_rank": rank % 2,
            "dp_shard": dp_size,
            "dp_rank": rank // 2,
            "device": device.type,
        }
        reference = _build(options["lora"], options["checkpointed"]).to(device)
        model = copy.deepcopy(reference)
        control = copy.deepcopy(reference)
        apply_tensor_parallel(
            peel_peft_wrapper(control),
            mesh["tp"],
            {"up": ColwiseParallel(), "down": RowwiseParallel()},
        )
        with torch.no_grad():
            operand = _operand(0, rank // 2)[0].to(device)
            negative_delta = float((control(operand) - reference(operand)).abs().max())
            assert negative_delta > 1e-3, negative_delta
        result["unexchanged_activation_max_error"] = negative_delta
        del control
        if cuda:
            values = (
                torch.arange(528, device=device)
                .remainder(127)
                .to(torch.bfloat16)
                .reshape(1, 2, 264)
            )
            local = values.chunk(2, -1)[rank % 2].clone().requires_grad_()
            exchange = _PackedExchange.build(264, (1, 1, 1, 4, 4), mesh["tp"], inverse=False)
            inverse = _PackedExchange.build(264, (1, 1, 1, 4, 4), mesh["tp"], inverse=True)
            packed = exchange(local)
            expected_packed = torch.cat(
                [part.chunk(2, -1)[rank % 2] for part in values.split([24, 24, 24, 96, 96], -1)], -1
            )
            torch.testing.assert_close(packed, expected_packed, atol=0, rtol=0)
            restored = inverse(packed)
            torch.testing.assert_close(restored, local, atol=0, rtol=0)
            restored.sum().backward()
            torch.testing.assert_close(local.grad, torch.ones_like(local), atol=0, rtol=0)
            result["bf16_transport_exact"] = True
        apply_tensor_parallel(peel_peft_wrapper(model), mesh["tp"], _plan(options["sp"]))
        if dp_size > 1:
            fully_shard(model, mesh=mesh["dp_shard"])
        shapes = {}
        native = peel_peft_wrapper(model)
        for path in ("start", "gather", "up", "down"):
            native.get_submodule(path).register_forward_hook(
                lambda _module, _inputs, output, path=path: shapes.setdefault(path, set()).add(
                    tuple(output.shape)
                )
            )
        result["placements"] = sorted(
            {str(parameter.placements) for parameter in model.parameters()}
        )
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        reference_optimizer = torch.optim.AdamW(
            [p for p in reference.parameters() if p.requires_grad], lr=0.001
        )
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                _whole(parameter), dict(reference.named_parameters())[name], atol=0, rtol=0
            )
        canonical = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))
        assert canonical.keys() == reference.state_dict().keys()
        for name, value in canonical.items():
            torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)
        checkpoint_dir = options["checkpoint_dir"]
        resume = options["resume"]
        if resume:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.3)
            state = _state(model, optimizer)
            dist.checkpoint.load(state, checkpoint_id=checkpoint_dir)
            set_model_state_dict(model, state["model"])
            set_optimizer_state_dict(model, optimizer, state["optimizer"])
            reference_state = torch.load(
                Path(checkpoint_dir).parent / "reference.pt", weights_only=True, map_location=device
            )
            reference.load_state_dict(reference_state["model"])
            reference_optimizer.load_state_dict(reference_state["optimizer"])
            _assert_parameters(model, reference, optimizer, reference_optimizer)
        result["updates"] = []
        for step in range(2 if resume else 0, 3):
            operands = [
                tuple(value.to(device) for value in _operand(step, replica))
                for replica in range(dp_size)
            ]
            x = operands[rank // 2][0].clone().requires_grad_()
            target = operands[rank // 2][1]
            reference_x = torch.cat([pair[0] for pair in operands]).requires_grad_()
            reference_target = torch.cat([pair[1] for pair in operands])
            output, expected = model(x), reference(reference_x)
            _assert_close(output, expected.chunk(dp_size)[rank // 2], "output")
            loss = (output - target).square().mean()
            reference_loss = (expected - reference_target).square().mean()
            loss.backward()
            reference_loss.backward()
            _assert_close(
                x.grad, reference_x.grad.chunk(dp_size)[rank // 2] * dp_size, "input gradient"
            )
            compared = 0
            for name, parameter in model.named_parameters():
                expected_parameter = dict(reference.named_parameters())[name]
                assert (parameter.grad is None) == (expected_parameter.grad is None), name
                if parameter.grad is not None:
                    gradient = _whole(parameter.grad)
                    assert torch.count_nonzero(gradient), name
                    _assert_close(gradient, expected_parameter.grad, name + " gradient")
                    compared += 1
            optimizer.step()
            reference_optimizer.step()
            assert _assert_parameters(model, reference, optimizer, reference_optimizer) == compared
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            result["updates"].append(
                {"step": step, "gradient_count": compared, "loss": float(loss.detach())}
            )
            if step == 1:
                dist.checkpoint.save(_state(model, optimizer), checkpoint_id=checkpoint_dir)
                if rank == 0:
                    torch.save(
                        {
                            "model": reference.state_dict(),
                            "optimizer": reference_optimizer.state_dict(),
                        },
                        Path(checkpoint_dir).parent / "reference.pt",
                    )
        local_tokens = 4 if options["sp"] else 8
        assert shapes == {
            "start": {(2, local_tokens, 24)},
            "gather": {(2, 8, 24)},
            "up": {(2, 8, 132)},
            "down": {(2, local_tokens, 24)},
        }, shapes
        result["stream_shapes"] = {
            name: [list(shape) for shape in values] for name, values in shapes.items()
        }
        final = {
            name: _whole(parameter).detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                for key in ("exp_avg", "exp_avg_sq"):
                    final[f"{name}:{key}"] = (
                        _whole(optimizer.state[parameter][key]).detach().cpu().clone()
                    )
        final_path = Path(checkpoint_dir).parent / "continuous.pt"
        if resume:
            continuous = torch.load(final_path, weights_only=True)
            for name, value in final.items():
                torch.testing.assert_close(value, continuous[name], atol=0, rtol=0)
            result["exact_continuation"] = True
        elif rank == 0:
            torch.save(final, final_path)
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
@pytest.mark.parametrize(
    "sp,checkpointed", [(False, False), (True, True)], ids=["tp", "sp-checkpointed"]
)
@requires_gloo
def test_packed_native_math_and_fresh_checkpoint(
    tmp_path: Path, lora: bool, sp: bool, checkpointed: bool
) -> None:
    options = dict(
        lora=lora,
        sp=sp,
        checkpointed=checkpointed,
        dp_size=1,
        checkpoint_dir=str(tmp_path / "checkpoint"),
    )
    for resume in (False, True):
        records = run_gloo_ranks(
            _worker,
            tmp_path / ("resume" if resume else "continuous"),
            2,
            {**options, "resume": resume},
        )
        for record in records:
            assert "error" not in record, record.get("error")
            assert record["status"] == "pass"
            assert len(record["updates"]) == (1 if resume else 3)
            assert all(update["gradient_count"] == 4 for update in record["updates"])
            if resume:
                assert record["exact_continuation"]


@pytest.mark.parametrize("blocks", [(), (1,), (0, 1), (True, 1), (1.0, 2), (-1, 2)])
def test_invalid_packed_blocks_rejected(blocks: tuple) -> None:
    for style in (PackedColwiseParallel, PackedRowwiseParallel):
        with pytest.raises(ValueError, match="positive integers"):
            style(blocks)


@pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
@requires_gloo
def test_packed_fsdp_composed_checkpoint(tmp_path: Path, lora: bool) -> None:
    options = dict(
        lora=lora,
        sp=True,
        checkpointed=True,
        dp_size=2,
        checkpoint_dir=str(tmp_path / "checkpoint"),
    )
    for resume in (False, True):
        records = run_gloo_ranks(
            _worker,
            tmp_path / ("resume" if resume else "continuous"),
            4,
            {**options, "resume": resume},
            timeout=600,
        )
        for record in records:
            assert "error" not in record, record.get("error")
            assert record["status"] == "pass"
            if resume:
                assert record["exact_continuation"]


def _exchange_worker(rank: int, store_path: str, result_dir: str, options: dict) -> None:
    degree = options["degree"]
    dist.init_process_group(
        "gloo", store=dist.FileStore(store_path, degree), rank=rank, world_size=degree
    )
    record = {"rank": rank, "degree": degree}
    try:
        mesh = init_device_mesh("cpu", (degree,), mesh_dim_names=("tp",))
        record["packs"] = []
        for blocks in ((1, 1), (1, 1, 1), (1, 4), (1, 1, 1, 4, 4), (2, 4)):
            unit = 12 if blocks != (2, 4) else degree // 2
            width = unit * sum(blocks)
            values = torch.arange(2 * 3 * width, dtype=torch.float64).reshape(2, 3, width)
            local = values.chunk(degree, dim=-1)[rank].detach().clone().requires_grad_()
            exchange = _PackedExchange.build(width, blocks, mesh, inverse=False)
            inverse = _PackedExchange.build(width, blocks, mesh, inverse=True)
            actual = exchange(local)
            expected = torch.cat(
                [
                    part.chunk(degree, dim=-1)[rank]
                    for part in values.split([unit * block for block in blocks], dim=-1)
                ],
                dim=-1,
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            restored = inverse(actual)
            torch.testing.assert_close(restored, local, atol=0, rtol=0)
            # A rank-dependent operand makes a wrong reverse peer order visible.
            (actual * (rank + 1)).sum().backward()
            expected_gradient = torch.cat(
                [
                    torch.cat(
                        [
                            torch.full((unit * block // degree,), peer + 1, dtype=values.dtype)
                            for peer in range(degree)
                        ]
                    )
                    for block in blocks
                ]
            ).chunk(degree)[rank]
            torch.testing.assert_close(
                local.grad, expected_gradient.expand_as(local), atol=0, rtol=0
            )
            record["packs"].append(list(blocks))
        record["status"] = "pass"
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(record))
        dist.destroy_process_group()


@pytest.mark.parametrize("degree", [2, 4])
@requires_gloo
def test_activation_exchange_and_inverse_gradient(tmp_path: Path, degree: int) -> None:
    for record in run_gloo_ranks(_exchange_worker, tmp_path, degree, {"degree": degree}):
        assert "error" not in record, record.get("error")
        assert record["status"] == "pass"
        assert len(record["packs"]) == 5


@pytest.mark.parametrize("style,width", [(PackedColwiseParallel, 9), (PackedRowwiseParallel, 15)])
def test_packed_divisibility_rejected_before_mutation(style, width: int) -> None:
    from types import SimpleNamespace

    module = nn.Linear(width, width)
    original = dict(module.named_parameters())
    with pytest.raises(ValueError, match="tp_size=2"):
        style((1, 1, 1))._apply(module, SimpleNamespace(size=lambda: 2))
    assert all(parameter is original[name] for name, parameter in module.named_parameters())
    assert not module._forward_pre_hooks and not module._forward_hooks


@pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four leased CUDA devices")
def test_cuda_packed_fsdp_state(tmp_path: Path, lora: bool) -> None:
    """Native NCCL/FSDP2 composes canonical packed weights with token sharding."""
    options = dict(
        lora=lora,
        sp=True,
        checkpointed=True,
        dp_size=2,
        cuda=True,
        checkpoint_dir=str(tmp_path / "checkpoint"),
    )
    for resume in (False, True):
        records = run_gloo_ranks(
            _worker,
            tmp_path / ("resume" if resume else "continuous"),
            4,
            {**options, "resume": resume},
            timeout=600,
        )
        for record in records:
            assert "error" not in record, record.get("error")
            assert record["status"] == "pass"
            assert record["bf16_transport_exact"]
            assert record["unexchanged_activation_max_error"] > 1e-3
            assert record["runtime"]["backend"] == "nccl"
            assert record["runtime"]["tp_size"] == record["runtime"]["dp_shard"] == 2
            assert record["runtime"]["tp_rank"] == record["rank"] % 2
            if resume:
                assert record["exact_continuation"]


@pytest.mark.parametrize(
    "style,attribute",
    [
        (PackedColwiseParallel, "_tp_packed_col_blocks"),
        (PackedRowwiseParallel, "_tp_packed_row_blocks"),
    ],
)
@pytest.mark.parametrize("metadata", [None, [True, 23], [0, 24], [6, 6], [12.0, 12]])
def test_native_packed_metadata_fails_before_mutation(style, attribute, metadata):
    from types import SimpleNamespace

    module = nn.Linear(24, 24)
    if metadata is not None:
        setattr(module, attribute, metadata)
    original = dict(module.named_parameters())
    with pytest.raises(ValueError, match="packed|positive integers"):
        style()._apply(module, SimpleNamespace(size=lambda: 2))
    assert all(parameter is original[name] for name, parameter in module.named_parameters())
    assert not module._forward_pre_hooks and not module._forward_hooks


def test_lora_styles_freeze_concrete_native_packing():
    from flow_factory.models.tensor_parallel import build_total_plan

    model = peel_peft_wrapper(_build(True, False))
    plan = build_total_plan(model, _plan(True))
    assert plan["up.lora_B.default"].block_sizes == (24, 24, 24, 96, 96)
    assert plan["down.lora_A.default"].block_sizes == (24, 96)
    model.up.base_layer._tp_packed_col_blocks[0] = 12
    model.down.base_layer._tp_packed_row_blocks[0] = 12
    assert plan["up.lora_B.default"].block_sizes == (24, 24, 24, 96, 96)
    assert plan["down.lora_A.default"].block_sizes == (24, 96)
