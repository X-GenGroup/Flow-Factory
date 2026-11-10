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

"""Diffusers CP's native SDPA replay uses its own forward precision context."""

import pytest
import torch
from diffusers.models import attention_dispatch as dispatch
from torch.nn import functional as F

from flow_factory.models.parallelism.autocast import install_native_cp_autocast_compat


@pytest.fixture(autouse=True)
def install_compat(monkeypatch):
    monkeypatch.setattr(
        dispatch, "_native_attention_forward_op", dispatch._native_attention_forward_op
    )
    monkeypatch.setattr(
        dispatch, "_native_attention_backward_op", dispatch._native_attention_backward_op
    )
    assert install_native_cp_autocast_compat()


class NativeReplay(torch.autograd.Function):
    """Exercise the real native CP operations without transport or model noise."""

    @staticmethod
    def forward(ctx, query, key, value):
        return dispatch._native_attention_forward_op(ctx, query, key, value)

    @staticmethod
    def backward(ctx, gradient):
        return dispatch._native_attention_backward_op(ctx, gradient)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("enabled", [False, True])
def test_backward_restores_each_forward_context(enabled, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(27)
    operands = [torch.randn(1, 7, 2, 8, device=device, requires_grad=True) for _ in range(3)]
    if enabled:
        operands[2] = operands[2].detach().bfloat16().requires_grad_()
    reference = [x.detach().clone().requires_grad_() for x in operands]
    with torch.autocast(device, dtype=torch.bfloat16, enabled=enabled):
        output = NativeReplay.apply(*operands)
        expected = F.scaled_dot_product_attention(*(x.transpose(1, 2) for x in reference))
        expected = expected.transpose(1, 2)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    upstream = torch.randn_like(output)
    # Deliberately reverse the ambient context: both enabled and disabled
    # forward scopes must survive a later backward under a different scope.
    with torch.autocast(device, dtype=torch.bfloat16, enabled=not enabled):
        gradients = torch.autograd.grad(output, operands, upstream)
    expected_gradients = torch.autograd.grad(expected, reference, upstream)
    for actual, target in zip(gradients, expected_gradients):
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    assert not torch.is_autocast_enabled(device)


def test_install_is_idempotent_and_does_not_replace_sdpa(monkeypatch):
    import diffusers

    forward, backward = (
        dispatch._native_attention_forward_op,
        dispatch._native_attention_backward_op,
    )
    native_sdpa = F.scaled_dot_product_attention
    assert install_native_cp_autocast_compat()
    assert (dispatch._native_attention_forward_op, dispatch._native_attention_backward_op) == (
        forward,
        backward,
    )
    assert F.scaled_dot_product_attention is native_sdpa
    monkeypatch.setattr(diffusers, "__version__", "0.43.0")
    assert not install_native_cp_autocast_compat()
    assert (dispatch._native_attention_forward_op, dispatch._native_attention_backward_op) == (
        forward,
        backward,
    )


def test_pending_forwards_replay_independent_contexts():
    torch.manual_seed(42)
    pending = []
    for enabled in (True, False):
        operands = [torch.randn(1, 7, 2, 8, requires_grad=True) for _ in range(3)]
        if enabled:
            operands[2] = operands[2].detach().bfloat16().requires_grad_()
        reference = [x.detach().clone().requires_grad_() for x in operands]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled, cache_enabled=enabled):
            actual = NativeReplay.apply(*operands)
            expected = F.scaled_dot_product_attention(
                *(x.transpose(1, 2) for x in reference)
            ).transpose(1, 2)
        pending.append((actual, expected, operands, reference))
    # Replay the first graph after another forward has recorded a different state.
    for actual, expected, operands, reference in pending:
        gradient = torch.randn_like(actual)
        for left, right in zip(
            torch.autograd.grad(actual, operands, gradient),
            torch.autograd.grad(expected, reference, gradient),
        ):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert not torch.is_autocast_enabled("cpu")
