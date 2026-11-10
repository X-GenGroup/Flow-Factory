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

"""Native autocast contracts for sharded normalization and CP replay."""

from functools import wraps
from typing import Callable

import torch
from packaging.version import Version

_CUDA_RMSNORM_AUTOCAST_FP32 = Version(torch.__version__).release[:2] >= (2, 12)


def _native_norm_output_dtype(value: torch.Tensor) -> torch.dtype:
    """Match native RMSNorm's version-specific CUDA autocast output contract.

    Torch 2.12 registers RMSNorm under the CUDA autocast FP32 policy. A TP
    implementation replaces that operator and must honor the same promotion.
    Outside that policy, preserve the input dtype, including FP64. This does
    not change native norms or impose a cross-version numerical policy.
    """
    if (
        _CUDA_RMSNORM_AUTOCAST_FP32
        and value.device.type == "cuda"
        and value.dtype != torch.float64
        and torch.is_autocast_enabled("cuda")
    ):
        return torch.float32
    return value.dtype


def _preserve_autocast(forward: Callable, backward: Callable) -> tuple[Callable, Callable]:
    """Capture precision per autograd context, including explicitly disabled AMP."""

    @wraps(forward)
    def forward_with_context(ctx, query, *args, **kwargs):
        device = query.device.type
        ctx._ff_native_attention_autocast = (
            device,
            torch.is_autocast_enabled(device),
            torch.get_autocast_dtype(device),
            torch.is_autocast_cache_enabled(),
        )
        return forward(ctx, query, *args, **kwargs)

    @wraps(backward)
    def backward_with_context(ctx, *args, **kwargs):
        device, enabled, dtype, cache_enabled = ctx._ff_native_attention_autocast
        with torch.autocast(device, enabled=enabled, dtype=dtype, cache_enabled=cache_enabled):
            return backward(ctx, *args, **kwargs)

    forward_with_context._ff_autocast_replay = True
    return forward_with_context, backward_with_context


def install_native_cp_autocast_compat() -> bool:
    """Backport native CP replay context preservation for the qualified 0.42 line.

    Diffusers saves Q/K/V before SDPA autocasts them, then replays SDPA inside
    its custom backward without restoring that context. Torch >=2.12 exposes
    this with FP32 RMSNorm Q/K and low-precision V. Preserve the actual forward
    state rather than guessing a dtype or changing normalization semantics.

    Install once per worker, before CP forwards. Only the two native CP
    operation callbacks change: ordinary SDPA, RMSNorm, Ring/Flash kernels and
    transport implementations remain owned by Torch/Diffusers. Remove this
    version-bounded backport when an upstream release passes the same replay
    regressions. Later unqualified Diffusers lines retain their native behavior.
    """
    from diffusers.models import attention_dispatch

    import diffusers

    if Version(diffusers.__version__).release[:2] != (0, 42):
        return False
    forward = attention_dispatch._native_attention_forward_op
    if getattr(forward, "_ff_autocast_replay", False):
        return True
    forward, backward = _preserve_autocast(
        forward, attention_dispatch._native_attention_backward_op
    )
    attention_dispatch._native_attention_forward_op = forward
    attention_dispatch._native_attention_backward_op = backward
    return True
