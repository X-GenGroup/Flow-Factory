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

"""Operation policies preserve autograd, ownership and surrounding precision."""

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.overrides import TorchFunctionMode
from torch.utils.checkpoint import checkpoint

from flow_factory.hparams.model_args import ModelArguments
from flow_factory.models.compute_precision import (
    ComputePrecisionPolicy,
    ComputePrecisionRule,
    apply_compute_precision_policy,
    resolve_compute_precision_policy,
)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, query, key, value, **kwargs):
        return F.scaled_dot_product_attention(
            query * self.weight.to(query.dtype), key, value, **kwargs
        )


class Capture(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        result = func(*args, **kwargs)
        if func is F.scaled_dot_product_attention:
            query = args[0] if args else kwargs["query"]
            self.calls.append(
                (query.dtype, result.dtype, torch.is_autocast_enabled(query.device.type))
            )
        return result


def model():
    root = nn.ModuleDict({"selected": Attention(), "native": Attention()})
    policy = ComputePrecisionPolicy("attention_fp32_v1", (ComputePrecisionRule("selected"),))
    return root, policy


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("mask_kind", ["none", "bool", "float", "causal"])
def test_scoped_sdpa_matches_explicit_fp32_forward_and_backward(dtype, mask_kind):
    torch.manual_seed(52)
    root, policy = model()
    apply_compute_precision_policy(root, policy)
    values = [torch.randn(1, 2, 5, 8).to(dtype).requires_grad_() for _ in range(3)]
    expected_values = [value.detach().clone().requires_grad_() for value in values]
    mask = None
    if mask_kind == "bool":
        mask = torch.ones(5, 5, dtype=torch.bool).tril()
    elif mask_kind == "float":
        mask = torch.randn(5, 5).to(dtype)
    options = dict(attn_mask=mask, is_causal=mask_kind == "causal", scale=0.37)
    with Capture() as capture, torch.autocast("cpu", dtype=torch.bfloat16):
        output = root["selected"](*values, **options)
        assert torch.is_autocast_enabled("cpu")
        root["native"](*values)
    assert capture.calls[0] == (torch.float32, torch.float32, False)
    assert capture.calls[1][2] is True
    if mask is not None and mask.is_floating_point():
        options["attn_mask"] = mask.float()
    expected = F.scaled_dot_product_attention(
        *(value.float() for value in expected_values), **options
    ).to(dtype)
    upstream = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, values, upstream)
    expected_grads = torch.autograd.grad(expected, expected_values, upstream)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_checkpoint_clone_and_state_ownership():
    from flow_factory.models.tensor_parallel import _argument_positions

    root, policy = model()
    original_signature = inspect.signature(root["selected"].forward)
    positions = _argument_positions(root["selected"], ("query", "key", "value"))
    parameter_ids = tuple(map(id, root.parameters()))
    keys = tuple(root.state_dict())
    assert apply_compute_precision_policy(root, policy) == ("selected",)
    assert apply_compute_precision_policy(root, policy) == ("selected",)
    assert tuple(map(id, root.parameters())) == parameter_ids
    assert tuple(root.state_dict()) == keys
    clone = copy.deepcopy(root)
    for current in (root, clone):
        assert inspect.signature(current["selected"].forward) == original_signature
        assert _argument_positions(current["selected"], ("query", "key", "value")) == positions
    assert clone["selected"]._ff_precision_original_forward.__self__ is clone["selected"]
    values = [torch.randn(1, 2, 5, 8, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = checkpoint(clone["selected"], *values, use_reentrant=False)
    output.float().sum().backward()
    assert clone["selected"].weight.grad is not None
    assert root["selected"].weight.grad is None
    assert not torch.is_autocast_enabled("cpu")
    clone.load_state_dict(root.state_dict(), strict=True)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(
            root["selected"](*values), clone["selected"](*values), rtol=0, atol=0
        )


def test_positional_options_dropout_and_grouped_query_attention():
    values = [
        torch.randn(1, heads, 5, 8, dtype=torch.bfloat16, requires_grad=True) for heads in (4, 2, 2)
    ]
    copies = [value.detach().clone().requires_grad_() for value in values]

    # Exercise the complete SDPA call signature inside the selected module.
    class PositionalAttention(nn.Module):
        def forward(self, query, key, value):
            return F.scaled_dot_product_attention(
                query, key, value, None, 0.2, False, enable_gqa=True
            )

    selected = nn.Sequential(PositionalAttention())
    apply_compute_precision_policy(
        selected, ComputePrecisionPolicy("gqa", (ComputePrecisionRule("0"),))
    )
    torch.manual_seed(69)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = selected[0](*values)
    torch.manual_seed(69)
    expected = F.scaled_dot_product_attention(
        *(value.float() for value in copies), None, 0.2, False, enable_gqa=True
    ).bfloat16()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    for left, right in zip(
        torch.autograd.grad(output.sum(), values), torch.autograd.grad(expected.sum(), copies)
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_exception_and_missing_operation_leave_no_active_mode():
    class Raises(nn.Module):
        def forward(self, value):
            raise RuntimeError("injected failure")

    root = nn.ModuleDict({"raises": Raises(), "identity": nn.Identity()})
    for name in root:
        apply_compute_precision_policy(
            root, ComputePrecisionPolicy(name, (ComputePrecisionRule(name),))
        )
    with pytest.raises(RuntimeError, match="injected failure"):
        root["raises"](torch.ones(1))
    with pytest.raises(RuntimeError, match="matched no native SDPA"):
        root["identity"](torch.ones(1))
    x = torch.randn(1, 1, 3, 4)
    with Capture() as capture, torch.autocast("cpu", dtype=torch.bfloat16):
        F.scaled_dot_product_attention(x, x, x)
    assert capture.calls == [(torch.float32, torch.bfloat16, True)]


def test_invalid_patterns_are_rejected_before_any_mutation():
    root, _ = model()
    for rules in (
        (ComputePrecisionRule("*"),),
        (ComputePrecisionRule("selected"), ComputePrecisionRule("missing")),
        (ComputePrecisionRule("*"), ComputePrecisionRule("selected")),
    ):
        with pytest.raises(ValueError):
            apply_compute_precision_policy(root, ComputePrecisionPolicy("invalid", rules))
        assert not hasattr(root["selected"], "_ff_precision_original_forward")


def test_policy_selection_is_explicit_and_rejects_unsupported_backends():
    from flow_factory.models.abc import BaseAdapter
    from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter

    config = SimpleNamespace(
        model_args=ModelArguments(), acceleration_args=SimpleNamespace(shared=[], rollout=[])
    )
    assert resolve_compute_precision_policy(BaseAdapter, config) is None
    config.model_args.compute_precision_policy = "cross_attention_fp32_v1"
    with pytest.raises(ValueError, match="available=\\(\\)"):
        resolve_compute_precision_policy(BaseAdapter, config)
    assert (
        resolve_compute_precision_policy(Wan2_T2V_Adapter, config)
        == Wan2_T2V_Adapter.compute_precision_policies[0]
    )
    config.acceleration_args.shared = [
        SimpleNamespace(name="attention_backend", params={"backend": "flash"})
    ]
    with pytest.raises(ValueError, match="only native attention"):
        resolve_compute_precision_policy(Wan2_T2V_Adapter, config)


@pytest.mark.parametrize("value", ["", 123, {}, []])
def test_invalid_policy_argument(value):
    with pytest.raises(ValueError, match="compute_precision_policy"):
        ModelArguments(compute_precision_policy=value)
