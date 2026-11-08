# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""SenseNova-owned prefix attention and spatial flow-head CP boundaries."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from ..context_parallel import ContextParallelBoundary
from ..tensor_parallel import _GatherTokens

if TYPE_CHECKING:
    from diffusers.models._modeling_parallel import ContextParallelConfig


def _pad(tensor: torch.Tensor, count: int, dim: int) -> torch.Tensor:
    shape = list(tensor.shape)
    shape[dim] = count - shape[dim]
    if shape[dim] < 0:
        raise ValueError("SenseNova CP padding cannot remove native tokens")
    return torch.cat([tensor, tensor.new_zeros(shape)], dim=dim)


def _generation(kwargs: dict[str, Any]) -> bool:
    return (
        kwargs.get("exist_image_gen_tokens") is True
        and kwargs.get("exist_non_image_gen_tokens") is False
    )


def configure_module(module: torch.nn.Module) -> None:
    """Declare one physical denoiser's attention owners before backend setup."""
    if getattr(module, "_parallel_config", None) is not None:
        raise ValueError("SenseNova CP configuration cannot reset installed native attention")
    for layer in module.model.language_model.model.layers:
        layer.self_attn._ff_sensenova_parallel_config = None


def install_context_parallel(module: torch.nn.Module, config: ContextParallelConfig) -> None:
    """Compose native CP slicing around the existing generation-only TP/SP plan."""
    import torch.distributed as dist
    from diffusers.hooks.context_parallel import EquipartitionSharder
    from diffusers.models._modeling_parallel import ParallelConfig

    if config.ring_degree != 1:
        raise ValueError("SenseNova prefixes require native Ulysses transport")
    if getattr(module, "_parallel_config", None) is not None:
        raise ValueError("SenseNova context parallelism may be installed only once")
    native = ParallelConfig(context_parallel_config=config)
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if config.mesh.device_type == "cuda"
        else torch.device("cpu")
    )
    native.setup(dist.get_rank(), dist.get_world_size(), device, mesh=config.mesh)
    backbone = module.model.language_model.model
    sp_size = getattr(backbone.layers[0], "_ff_sensenova_sp_size", 1)
    degree = config.ulysses_degree
    mesh = config._flattened_mesh

    def padded(length):
        return math.ceil(length / (degree * sp_size)) * degree * sp_size

    def split(_owner, args, kwargs):
        if not _generation(kwargs):
            return None
        if kwargs.get("update_cache", True) or kwargs.get("attention_mask") is not None:
            raise ValueError("SenseNova CP requires noncausal generation with a read-only prefix")
        hidden = args[0]
        indexes = kwargs["indexes"]
        if hidden.ndim != 3 or hidden.shape[0] != 1 or indexes.shape != (3, hidden.shape[1]):
            raise ValueError(
                "SenseNova CP requires one generated sample and three native rotary axes"
            )
        local = EquipartitionSharder.shard(_pad(hidden, padded(hidden.shape[1]), 1), 1, mesh)
        return (local, *args[1:]), kwargs

    def positions(_owner, args, kwargs):
        if not _generation(kwargs):
            return None
        indexes = kwargs["indexes"]
        count = indexes.shape[1]
        local = EquipartitionSharder.shard(_pad(indexes, padded(count), 1), 1, mesh)
        return args, {**kwargs, "indexes": local, "_ff_cp_target_length": count}

    def before_head(_owner, args, kwargs, output):
        indicators = kwargs.get("image_gen_indicators")
        if indicators is None or not bool(indicators.all()):
            return output
        count = kwargs["indexes"].shape[1]
        # The trainable spatial head consumes the complete image. Its different
        # CP ranks own different output tokens, so their input derivatives are
        # partial contributions that must sum before returning to local tokens.
        output.last_hidden_state = _GatherTokens.apply(output.last_hidden_state, mesh, 1)[:, :count]
        return output

    def finish(_owner, args, output):
        if not isinstance(output, torch.Tensor):
            return output
        count = output.shape[1]
        local = EquipartitionSharder.shard(_pad(output, padded(count), 1), 1, mesh)
        return EquipartitionSharder.unshard(local, 1, mesh)[:, :count]

    first = backbone.layers[0]
    before = first.register_forward_pre_hook(split, with_kwargs=True, prepend=True)
    for layer in backbone.layers:
        layer.self_attn._ff_sensenova_parallel_config = native
        layer.self_attn.register_forward_pre_hook(positions, with_kwargs=True)
    backbone.register_forward_hook(before_head, with_kwargs=True)
    after = module.register_forward_hook(finish)
    module._parallel_config = native
    module._ff_context_parallel_boundaries = (
        ContextParallelBoundary(first, "input", split, before),
        ContextParallelBoundary(module, "output", finish, after),
    )


def prefix_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: Any,
    target_length: int | None,
) -> torch.Tensor:
    """Attend local native Q/K/V to immutable prefix plus current image tokens.

    Inputs use native [batch, heads, tokens, width] layout after all native
    split-half qk normalization and temporal/spatial RoPE operations.
    """
    from diffusers.hooks.context_parallel import EquipartitionSharder
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    native = module._ff_sensenova_parallel_config
    cp = native.context_parallel_config
    mesh, degree = cp._flattened_mesh, cp.ulysses_degree
    query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    local_length = query.shape[1]
    if target_length is None or not 0 < target_length <= local_length * degree:
        raise ValueError("SenseNova CP attention lost its invocation's original target length")
    prefix_length = 0
    if cache is not None:
        layer = cache.layers[module.layer_idx]
        if layer.keys is not None:
            prefix_key, prefix_value = layer.keys, layer.values
            if prefix_key.requires_grad or prefix_value.requires_grad:
                raise ValueError("SenseNova CP requires a detached native prefix cache")
            prefix_length = prefix_key.shape[2]
            padded_prefix = math.ceil(prefix_length / degree) * degree
            prefixes = [
                EquipartitionSharder.shard(_pad(tensor.transpose(1, 2), padded_prefix, 1), 1, mesh)
                for tensor in (prefix_key, prefix_value)
            ]
            key, value = (
                torch.cat([prefix, current], dim=1)
                for prefix, current in zip(prefixes, (key, value))
            )
    padded_prefix = math.ceil(prefix_length / degree) * degree
    prefix_mask = torch.arange(padded_prefix, device=query.device) < prefix_length
    target_mask = torch.arange(local_length * degree, device=query.device) < target_length
    mask = torch.cat(
        [
            torch.cat(
                [prefix_mask.reshape(degree, -1)[rank], target_mask.reshape(degree, -1)[rank]]
            )
            for rank in range(degree)
        ]
    )[None, None, None]
    if query.shape[2] % key.shape[2]:
        raise ValueError("SenseNova native query heads must be a multiple of cached KV heads")
    repeats = query.shape[2] // key.shape[2]
    key, value = (tensor.repeat_interleave(repeats, dim=2) for tensor in (key, value))
    # Native Ulysses' rectangular backward currently expects equal local Q/KV
    # lengths. Independent dummy queries add no loss and never change real Q.
    output = dispatch_attention_fn(
        _pad(query, key.shape[1], 1),
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=module.scaling,
        backend="native",
        parallel_config=native,
    )
    return output[:, :local_length]
