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

"""Qwen-Image under tensor and sequence parallelism matches the unsharded model.

Including ``zero_cond_t`` (Qwen-Image-Edit-2511), whose blocks take a per-token
``modulate_index`` the overlay has to narrow with the image sequence, and with
the blocks called both by keyword and positionally (gradient checkpointing).
"""

from __future__ import annotations

import copy
import json
import traceback
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks
from tp_qualification import assert_optimizer_step_parity, full_tensor, parameters_by_name

WORLD_SIZE = 2


def _build(zero_cond_t: bool) -> torch.nn.Module:
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    torch.manual_seed(0)
    model = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=4,
        joint_attention_dim=32,
        axes_dims_rope=(4, 6, 6),
        zero_cond_t=zero_cond_t,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.2)
    return model.double()


def _forward_backward(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    from torch.distributed.tensor import DTensor

    generator = torch.Generator().manual_seed(1)
    target, reference = (1, 4, 4), (1, 4, 4)
    hidden = torch.randn(1, 32, 16, generator=generator).double()
    inputs = dict(
        hidden_states=hidden,
        encoder_hidden_states=torch.randn(1, 7, 32, generator=generator).double(),
        encoder_hidden_states_mask=torch.ones(1, 7, dtype=torch.long),
        timestep=torch.tensor([0.5], dtype=torch.float64),
        img_shapes=[[target, reference]],
        return_dict=False,
    )
    parameter = next(model.parameters())
    inputs = {
        name: (
            value.to(
                device=parameter.device,
                dtype=parameter.dtype if value.is_floating_point() else value.dtype,
            )
            if isinstance(value, torch.Tensor)
            else value
        )
        for name, value in inputs.items()
    }
    output = model(**inputs)[0]
    velocity = torch.randn(output.shape, generator=generator).to(output)
    # Fixed supervised flow target isolates placement from rollout/reward semantics.
    torch.nn.functional.mse_loss(output, velocity).backward()
    values = {"output": output.detach()}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is not None:
            values[name] = (grad.full_tensor() if isinstance(grad, DTensor) else grad).detach()
    return values


def _sgd_rounding_check(before, gradient, after, learning_rate):
    """Bound an FP32 multiply/add from its operands, evaluated independently in FP64."""
    expected = before.double() - learning_rate * gradient.double()
    eps = torch.finfo(before.dtype).eps
    # One scalar multiply and one addition, including scalar representation error.
    budget = 3 * eps * (before.double().abs() + abs(learning_rate) * gradient.double().abs())
    assert torch.all((after.double() - expected).abs() <= budget), "incorrect SGD recurrence"
    return budget


def _assert_cuda_sgd_parity(reference, sharded):
    """Keep gradient gates; account for parameter-ULP rounding in FP32 update deltas."""
    expected, actual = parameters_by_name(reference), parameters_by_name(sharded)
    assert expected.keys() == actual.keys()
    before, gradients = {}, {}
    for name, parameter in expected.items():
        other = actual[name]
        assert parameter.requires_grad == other.requires_grad, name
        assert (parameter.grad is None) == (other.grad is None), name
        before[name] = full_tensor(parameter), full_tensor(other)
        torch.testing.assert_close(before[name][0], before[name][1], rtol=0, atol=0, msg=name)
        if parameter.grad is not None:
            gradients[name] = full_tensor(parameter.grad), full_tensor(other.grad)
            torch.testing.assert_close(
                gradients[name][1], gradients[name][0], rtol=3e-4, atol=2e-6, msg=name
            )
    assert gradients
    learning_rate = 0.01
    for model in (reference, sharded):
        torch.optim.SGD(model.parameters(), lr=learning_rate).step()
    changed = 0
    for name in expected:
        left, right = before[name]
        left_after, right_after = full_tensor(expected[name]), full_tensor(actual[name])
        if name not in gradients:
            torch.testing.assert_close(left_after, left, rtol=0, atol=0, msg=name)
            torch.testing.assert_close(right_after, right, rtol=0, atol=0, msg=name)
            continue
        left_grad, right_grad = gradients[name]
        left_budget = _sgd_rounding_check(left, left_grad, left_after, learning_rate)
        right_budget = _sgd_rounding_check(right, right_grad, right_after, learning_rate)
        delta_difference = (right_after.double() - right.double()) - (
            left_after.double() - left.double()
        )
        budget = (
            learning_rate * (right_grad.double() - left_grad.double()).abs()
            + left_budget
            + right_budget
        )
        assert torch.all(delta_difference.abs() <= budget), name
        changed += int(bool(torch.count_nonzero(right_after - right)))
    assert changed


@pytest.mark.parametrize("corruption", ["missing", "wrong_sign", "wrong_rate"])
def test_sgd_rounding_check_rejects_incorrect_updates(corruption):
    before = torch.tensor([0.4, -0.3], dtype=torch.float32)
    gradient = torch.tensor([0.5, -0.25], dtype=torch.float32)
    rate = {"missing": 0.0, "wrong_sign": -0.01, "wrong_rate": 0.02}[corruption]
    after = before - rate * gradient
    with pytest.raises(AssertionError, match="incorrect SGD recurrence"):
        _sgd_rounding_check(before, gradient, after, 0.01)


def test_sgd_rounding_check_accepts_native_fp32_update():
    before = torch.tensor([0.4, -0.3], dtype=torch.float32)
    gradient = torch.tensor([0.5, -0.25], dtype=torch.float32)
    _sgd_rounding_check(before, gradient, before.add(gradient, alpha=-0.01), 0.01)


def _parity_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    cuda = options.get("cuda", False)
    if cuda:
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    dtype = torch.float32 if cuda else torch.float64
    dist.init_process_group(
        "nccl" if cuda else "gloo", store=store, rank=rank, world_size=WORLD_SIZE
    )
    result: Dict[str, Any] = {"rank": rank}
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter
        from flow_factory.models.tensor_parallel import apply_tensor_parallel, resolve_parallel_plan

        mesh = init_device_mesh(device.type, (WORLD_SIZE,), mesh_dim_names=("tp",))
        for zero_cond_t in (False, True):
            for checkpointing in ((True,) if cuda else (False, True)):
                reference = _build(zero_cond_t).to(device=device, dtype=dtype)
                sharded = copy.deepcopy(reference)
                plan = resolve_parallel_plan(QwenImageEditPlusAdapter, sequence_parallel=True)
                apply_tensor_parallel(sharded, mesh, plan)
                if checkpointing:
                    sharded.enable_gradient_checkpointing()
                expected = _forward_backward(reference)
                received = _forward_backward(sharded)
                if cuda:
                    _assert_cuda_sgd_parity(reference, sharded)
                else:
                    assert_optimizer_step_parity(reference, sharded, rtol=1e-8, atol=1e-10)
                assert expected.keys() == received.keys(), sorted(expected.keys() ^ received.keys())
                result[f"zero_cond_t={zero_cond_t},checkpointing={checkpointing}"] = max(
                    (
                        (received[name] - expected[name]).abs().max() / expected[name].abs().max()
                    ).item()
                    for name in expected
                    if expected[name].abs().max() > 0
                )
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = traceback.format_exc()
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_qwen_image_with_and_without_zero_cond_t_matches_the_unsharded_model(
    tmp_path: Path,
) -> None:
    results = run_gloo_ranks(_parity_worker, tmp_path, WORLD_SIZE, {})

    for result in results:
        assert "error" not in result, result["error"]
        for case, relative in result.items():
            if case != "rank":
                assert relative < 1e-9, (case, result)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cuda_qwen_adaln_tp2_supervised_update(tmp_path: Path) -> None:
    """One TP2/DP1 update checks ordinary and Edit-2511 modulation with image SP.

    This component-level supervised flow test checks every gradient and parameter
    update against the dense model; it does not run the dataset or SFTTrainer loop.
    """
    results = run_gloo_ranks(_parity_worker, tmp_path, WORLD_SIZE, {"cuda": True})
    for result in results:
        assert "error" not in result, result.get("error")
        for case, relative in result.items():
            if case != "rank":
                assert relative < 3e-4, (case, result)
