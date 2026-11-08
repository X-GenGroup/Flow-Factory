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

"""Bagel-owned packed sample layouts for native TP/SP/context parallelism."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedLayout:
    """Keep each logical sample isolated while every CP rank receives equal slices.

    Padding is internal to the denoiser. Indices always refer to the original
    packed samples; native prefix caches and reference ordering stay unchanged.
    """

    lengths: tuple[int, ...]
    cp_size: int
    sp_size: int = 1

    def __post_init__(self) -> None:
        if not self.lengths or any(type(n) is not int or n < 0 for n in self.lengths):
            raise ValueError(f"Bagel packed lengths must be nonnegative integers: {self.lengths}")
        if any(type(n) is not int or n < 1 for n in (self.cp_size, self.sp_size)):
            raise ValueError(f"Bagel requires positive CP/SP sizes: {self.cp_size}, {self.sp_size}")

    @property
    def padded_lengths(self) -> tuple[int, ...]:
        degree = self.cp_size * self.sp_size
        return tuple(math.ceil(n / degree) * degree for n in self.lengths)

    @property
    def local_lengths(self) -> tuple[int, ...]:
        return tuple(n // self.cp_size for n in self.padded_lengths)

    def ids(self, rank: int | None, *, device: torch.device | str) -> torch.Tensor:
        """Map rank-major padded tokens to original packed offsets; -1 means padding."""
        if rank is not None and not 0 <= rank < self.cp_size:
            raise ValueError(f"Bagel CP rank{rank} is outside degree{self.cp_size}")
        samples = []
        offset = 0
        for length, padded in zip(self.lengths, self.padded_lengths):
            samples.append(
                torch.cat(
                    [
                        torch.arange(offset, offset + length, device=device),
                        torch.full((padded - length,), -1, device=device, dtype=torch.long),
                    ]
                ).reshape(self.cp_size, padded // self.cp_size)
            )
            offset += length
        rows = [torch.cat([sample[r] for sample in samples]) for r in range(self.cp_size)]
        return torch.cat(rows) if rank is None else rows[rank]

    def reorder(self, tensor: torch.Tensor, *, fill: float = 0) -> torch.Tensor:
        """Arrange padded sample chunks for one native contiguous CP split."""
        if tensor.shape[0] != sum(self.lengths):
            raise ValueError(
                f"Bagel packed axis{tensor.shape[0]} differs from lengths{self.lengths}"
            )
        ids = self.ids(None, device=tensor.device)
        if not tensor.shape[0]:
            return tensor
        result = tensor.index_select(0, ids.clamp_min(0))
        mask = (ids >= 0).reshape(-1, *([1] * (tensor.ndim - 1)))
        return torch.where(mask, result, tensor.new_tensor(fill))

    def restore(self, gathered: torch.Tensor) -> torch.Tensor:
        """Undo rank-major native gather and remove padding without changing sample order."""
        ids = self.ids(None, device=gathered.device)
        if gathered.shape[0] != ids.numel():
            raise ValueError("Bagel gathered packed axis differs from its owning layout")
        valid = torch.nonzero(ids >= 0, as_tuple=True)[0]
        return gathered.index_select(0, valid[ids[valid].argsort()])

    def roles(self, text_indexes: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return local text and generation indexes, assigning loss-free padding to generation."""
        if text_indexes.ndim != 1 or text_indexes.dtype != torch.long:
            raise ValueError("Bagel text token indexes must be a one-dimensional int64 tensor")
        if text_indexes.numel() and (
            int(text_indexes.min()) < 0 or int(text_indexes.max()) >= sum(self.lengths)
        ):
            raise ValueError("Bagel text token indexes are outside the original packed sequence")
        if text_indexes.unique().numel() != text_indexes.numel():
            raise ValueError("Bagel text token indexes must be unique")
        original = torch.zeros(sum(self.lengths), dtype=torch.bool, device=text_indexes.device)
        original[text_indexes] = True
        ids = self.ids(rank, device=text_indexes.device)
        text = (ids >= 0) & original[ids.clamp_min(0)] if original.numel() else ids.bool()
        return torch.nonzero(text, as_tuple=True)[0], torch.nonzero(~text, as_tuple=True)[0]

    def sample_key_mask(self, sample: int, prefix_length: int, *, device) -> torch.Tensor:
        """Match native Ulysses' rank-major [prefix chunk, target chunk] key order."""
        if prefix_length < 0:
            raise ValueError("Bagel cached prefix length must be nonnegative")
        prefix_padded = math.ceil(prefix_length / self.cp_size) * self.cp_size
        target_padded = self.padded_lengths[sample]
        prefix = torch.arange(prefix_padded, device=device) < prefix_length
        target = torch.arange(target_padded, device=device) < self.lengths[sample]
        return torch.cat(
            [
                torch.cat(
                    [prefix.reshape(self.cp_size, -1)[rank], target.reshape(self.cp_size, -1)[rank]]
                )
                for rank in range(self.cp_size)
            ]
        )[None, None, None]


def _pad_rows(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if length < tensor.shape[0]:
        raise ValueError("Bagel parallel padding cannot remove original tokens")
    return torch.cat([tensor, tensor.new_zeros((length - tensor.shape[0], *tensor.shape[1:]))])


def _select_rows(tensor: torch.Tensor, ids: torch.Tensor, fill: float = 0) -> torch.Tensor:
    if not tensor.shape[0]:
        return tensor.new_full((ids.numel(), *tensor.shape[1:]), fill)
    mask = (ids >= 0).reshape(-1, *([1] * (tensor.ndim - 1)))
    return torch.where(mask, tensor.index_select(0, ids.clamp_min(0)), tensor.new_tensor(fill))


def _call_arguments(module, args, kwargs):
    import inspect

    return dict(
        inspect.signature(type(module).forward_inference)
        .bind_partial(module, *args, **kwargs)
        .arguments
    )


@dataclass(frozen=True)
class _Call:
    layout: PackedLayout
    text_indexes: torch.Tensor
    active: bool


class _State:
    def __init__(self):
        from contextvars import ContextVar

        self.call = ContextVar("bagel_parallel_call", default=None)
        self.tp_mesh = None
        self.cp = None
        self.parallel_config = None
        self.boundaries = ()

    @property
    def cp_rank(self):
        return 0 if self.cp is None else self.cp._ulysses_local_rank

    @property
    def active(self):
        call = self.call.get()
        return call is not None and call.active

    def describe(self, arguments):
        cp_size = 1 if self.cp is None else self.cp.ulysses_degree
        sp_size = 1 if self.tp_mesh is None else self.tp_mesh.size()
        active = arguments.get("mode", "und") == "gen" and not arguments.get(
            "update_past_key_values", True
        )
        if active and arguments.get("is_causal", True):
            raise ValueError("Bagel generation parallelism requires noncausal target attention")
        lengths = tuple(int(n) for n in arguments["query_lens"].tolist())
        text = arguments.get("packed_text_indexes")
        if text is None:
            text = torch.empty(0, dtype=torch.long, device=arguments["query_lens"].device)
        if active:
            generation = arguments.get("packed_vae_token_indexes")
            if generation is None or generation.ndim != 1 or generation.dtype != torch.long:
                raise ValueError("Bagel generation requires one-dimensional int64 latent indexes")
            roles = torch.cat([text, generation]).sort().values
            if not torch.equal(roles, torch.arange(sum(lengths), device=roles.device)):
                raise ValueError("Bagel text and latent indexes must partition the packed query")
            prefix = arguments.get("key_values_lens")
            prefix_lengths = (
                (0,) * len(lengths) if prefix is None else tuple(int(n) for n in prefix.tolist())
            )
            if len(prefix_lengths) != len(lengths) or any(n < 0 for n in prefix_lengths):
                raise ValueError("Bagel prefix lengths must match every packed query sample")
            offset = 0
            expected = []
            expected_prefix = []
            for length, prefix_length in zip(lengths, prefix_lengths):
                expected_prefix.append(
                    torch.arange(offset, offset + prefix_length, device=text.device)
                )
                expected.append(
                    torch.arange(
                        offset + prefix_length, offset + prefix_length + length, device=text.device
                    )
                )
                offset += prefix_length + length
            if not torch.equal(arguments["packed_query_indexes"], torch.cat(expected)):
                raise ValueError(
                    "Bagel parallel generation requires ordered prefix-then-query sample indexes"
                )
            expected_prefix = torch.cat(expected_prefix)
            prefix_indexes = arguments.get("packed_key_value_indexes")
            if prefix_indexes is None and not expected_prefix.numel():
                prefix_indexes = expected_prefix
            if (
                not isinstance(prefix_indexes, torch.Tensor)
                or prefix_indexes.ndim != 1
                or prefix_indexes.dtype != torch.long
                or not torch.equal(prefix_indexes, expected_prefix)
            ):
                raise ValueError(
                    "Bagel parallel generation requires canonical packed prefix cache indexes; "
                    f"query lengths={lengths}, prefix lengths={prefix_lengths}"
                )
        return _Call(PackedLayout(lengths, cp_size, sp_size), text, active)


def configure_module(module):
    """Own one physical language-model scope; prefix/cache construction remains whole."""
    from functools import wraps

    from torch.distributed.tensor import DTensor, Replicate, Shard

    existing = getattr(module, "_ff_bagel_parallel_state", None)
    if existing is not None:
        return existing
    state = _State()
    for child in module.modules():
        child._ff_bagel_parallel_state = state
    original = module.forward

    @wraps(original)
    def forward(*args, **kwargs):
        call = state.describe(_call_arguments(module, args, kwargs))
        token = state.call.set(call)
        try:
            return original(*args, **kwargs)
        finally:
            state.call.reset(token)

    module.forward = forward

    def prepare(owner, args, kwargs):
        call = state.call.get()
        if not call.active:
            return None
        if getattr(owner, "enable_taylorseer", False):
            raise ValueError("Bagel parallel generation cannot use TaylorSeer activation caches")
        kwargs = dict(kwargs)
        value = call.layout.reorder(kwargs["packed_query_sequence"])
        if state.cp is not None:
            from diffusers.hooks.context_parallel import EquipartitionSharder

            value = EquipartitionSharder.shard(value, 0, state.cp._flattened_mesh)
        if state.tp_mesh is not None:
            value = (
                DTensor.from_local(value, state.tp_mesh, [Replicate()], run_check=False)
                .redistribute(placements=[Shard(0)])
                .to_local()
            )
        kwargs["packed_query_sequence"] = value
        return args, kwargs

    def restore(owner, args, output):
        call = state.call.get()
        if not call.active:
            return output
        value = output.packed_query_sequence
        if state.tp_mesh is not None:
            value = (
                DTensor.from_local(value, state.tp_mesh, [Shard(0)], run_check=False)
                .redistribute(placements=[Replicate()])
                .to_local()
            )
        if state.cp is not None:
            from diffusers.hooks.context_parallel import EquipartitionSharder

            value = EquipartitionSharder.unshard(value, 0, state.cp._flattened_mesh)
        output.packed_query_sequence = call.layout.restore(value)
        return output

    # The native sharder supplies both directions' autograd. Per-sample packing,
    # scopes and ordering belong to this non-ModelMixin family boundary.
    before = module.model.register_forward_pre_hook(prepare, with_kwargs=True)
    after = module.model.register_forward_hook(restore)
    from ..context_parallel import ContextParallelBoundary

    state.boundaries = (
        ContextParallelBoundary(module.model, "input", prepare, before),
        ContextParallelBoundary(module.model, "output", restore, after),
    )

    return state


def local_final_indexes(module, text_indexes, generation_indexes):
    state = getattr(module, "_ff_bagel_parallel_state", None)
    if state is None or not state.active:
        return text_indexes, generation_indexes
    layout = state.call.get().layout
    roles = layout.roles(text_indexes, state.cp_rank)
    if state.tp_mesh is None:
        return roles
    length = sum(layout.local_lengths) // state.tp_mesh.size()
    start = state.tp_mesh.get_local_rank() * length
    return tuple(index[(index >= start) & (index < start + length)] - start for index in roles)


def install_context_parallel(module, config):
    """Install the exact native ParallelConfig and native sharder on a Bagel root."""
    import torch.distributed as dist
    from diffusers.models._modeling_parallel import ParallelConfig

    if config.ring_degree != 1:
        raise ValueError("Bagel packed prefixes require native Ulysses transport")
    if getattr(module, "_parallel_config", None) is not None:
        raise ValueError("Bagel context parallelism can be installed only once")
    state = configure_module(module)
    native = ParallelConfig(context_parallel_config=config)
    device = (
        torch.device(config.mesh.device_type, torch.cuda.current_device())
        if config.mesh.device_type == "cuda"
        else torch.device("cpu")
    )
    native.setup(dist.get_rank(), dist.get_world_size(), device, mesh=config.mesh)
    state.cp = config
    state.parallel_config = native
    module._parallel_config = native


def parallel_block_forward(block, arguments):
    """Run native MoT projections with local residuals and one gather per TP region."""
    from torch.distributed.tensor import DTensor, Replicate, Shard

    from ..tensor_parallel import _GatherTokens

    state = block._ff_bagel_parallel_state
    call = state.describe(arguments)
    token = state.call.set(call)
    try:
        value = arguments["packed_query_sequence"]
        text, gen = call.layout.roles(call.text_indexes, state.cp_rank)
        cp_text, cp_gen = text, gen
        if state.tp_mesh is not None:
            rank = state.tp_mesh.get_local_rank()
            start = rank * value.shape[0]
            text = text[(text >= start) & (text < start + value.shape[0])] - start
            gen = gen[(gen >= start) & (gen < start + value.shape[0])] - start

        def normalize(hidden, text_norm, gen_norm):
            result = torch.zeros_like(hidden)
            result[text] = text_norm(hidden[text])
            result[gen] = gen_norm(hidden[gen])
            return result

        def gather(hidden):
            return (
                _GatherTokens.apply(hidden, state.tp_mesh, 0)
                if state.tp_mesh is not None
                else hidden
            )

        def scatter(hidden):
            if state.tp_mesh is None:
                return hidden
            return (
                DTensor.from_local(hidden, state.tp_mesh, [Replicate()], run_check=False)
                .redistribute(placements=[Shard(0)])
                .to_local()
            )

        residual = value
        normalized = normalize(value, block.input_layernorm, block.input_layernorm_moe_gen)
        attention = packed_attention(block.self_attn, gather(normalized), arguments, call, state)
        value = residual + scatter(attention)
        residual = value
        normalized = normalize(
            value, block.post_attention_layernorm, block.post_attention_layernorm_moe_gen
        ).to(torch.bfloat16)
        normalized = gather(normalized)
        mlp = torch.zeros_like(normalized)
        mlp[cp_text] = block.mlp(normalized[cp_text])
        mlp[cp_gen] = block.mlp_moe_gen(normalized[cp_gen])
        return residual + scatter(mlp), arguments["past_key_values"]
    finally:
        state.call.reset(token)


def packed_attention(attention, hidden, arguments, call, state):
    """Preserve native MoT QKV/norm/RoPE and isolate each packed sample through Ulysses."""
    from .modeling.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    hidden = hidden.to(torch.bfloat16)
    text, gen = call.layout.roles(call.text_indexes, state.cp_rank)
    query = hidden.new_zeros(hidden.shape[0], attention.num_heads, attention.head_dim)
    key = hidden.new_zeros(hidden.shape[0], attention.num_key_value_heads, attention.head_dim)
    value = torch.zeros_like(key)
    for indexes, suffix in ((text, ""), (gen, "_moe_gen")):
        query[indexes] = getattr(attention, "q_proj" + suffix)(hidden[indexes]).reshape(
            -1, attention.num_heads, attention.head_dim
        )
        key[indexes] = getattr(attention, "k_proj" + suffix)(hidden[indexes]).reshape(
            -1, attention.num_key_value_heads, attention.head_dim
        )
        value[indexes] = getattr(attention, "v_proj" + suffix)(hidden[indexes]).reshape(
            -1, attention.num_key_value_heads, attention.head_dim
        )
    query, key = query.float(), key.float()
    for indexes, suffix in ((text, ""), (gen, "_moe_gen")):
        query[indexes] = getattr(attention, "q_norm" + suffix)(query[indexes])
        key[indexes] = getattr(attention, "k_norm" + suffix)(key[indexes])
    ids = call.layout.ids(state.cp_rank, device=hidden.device)
    cos, sin = arguments["packed_query_position_embeddings"]
    cos, sin = _select_rows(cos, ids, 1), _select_rows(sin, ids)
    query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)
    query, key, value = query.bfloat16(), key.bfloat16(), value.bfloat16()
    cache = arguments["past_key_values"]
    lengths = arguments.get("key_values_lens")
    prefix_lengths = (
        (0,) * len(call.layout.lengths)
        if lengths is None
        else tuple(int(n) for n in lengths.tolist())
    )
    if cache is None or cache.key_cache[attention.layer_idx] is None:
        prefix_k = key.new_empty((0, attention.num_key_value_heads, attention.head_dim))
        prefix_v = torch.empty_like(prefix_k)
    else:
        prefix_k, prefix_v = (
            cache.key_cache[attention.layer_idx],
            cache.value_cache[attention.layer_idx],
        )
    if prefix_k.shape[0] != sum(prefix_lengths) or prefix_v.shape != prefix_k.shape:
        raise ValueError("Bagel prefix cache does not match its packed sample lengths")
    attended = packed_attention_kernel(
        query,
        key,
        value,
        prefix_k,
        prefix_v,
        prefix_lengths,
        call.layout,
        state.cp,
        state.parallel_config,
    ).reshape(hidden.shape[0], attention.num_heads * attention.head_dim)
    output = hidden.new_zeros(hidden.shape[0], attention.hidden_size)
    output[text] = attention.o_proj(attended[text])
    output[gen] = attention.o_proj_moe_gen(attended[gen])
    return output


def packed_attention_kernel(
    query, key, value, prefix_k, prefix_v, prefix_lengths, layout, cp, parallel_config
):
    """Attend each packed sample with native collectives and masked internal padding.

    The prefix is detached by native Bagel prefill. Query padding only satisfies
    native Ulysses' equal local Q/K backward geometry; those rows never reach
    output projections. GQA head repetition preserves its native sum in backward.
    """
    from diffusers.hooks.context_parallel import EquipartitionSharder
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    outputs = []
    for sample, (q, k, v, pk, pv) in enumerate(
        zip(
            query.split(layout.local_lengths),
            key.split(layout.local_lengths),
            value.split(layout.local_lengths),
            prefix_k.split(prefix_lengths),
            prefix_v.split(prefix_lengths),
        )
    ):
        local_query_length = q.shape[0]
        if local_query_length == 0:
            outputs.append(q)
            continue
        padded_prefix = math.ceil(pk.shape[0] / layout.cp_size) * layout.cp_size
        pk, pv = _pad_rows(pk, padded_prefix), _pad_rows(pv, padded_prefix)
        if cp is not None:
            pk, pv = (EquipartitionSharder.shard(x, 0, cp._flattened_mesh) for x in (pk, pv))
        k, v = torch.cat([pk, k]), torch.cat([pv, v])
        k = k.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
        v = v.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
        if cp is not None:
            q = _pad_rows(q, k.shape[0])
        mask = layout.sample_key_mask(sample, prefix_lengths[sample], device=query.device)
        output = dispatch_attention_fn(
            q[None],
            k[None],
            v[None],
            attn_mask=mask,
            backend="native",
            parallel_config=parallel_config,
        )[0, :local_query_length]
        outputs.append(output)
    return torch.cat(outputs)


def sequence_parallel_plan():
    """Add attention TP only for the new explicit packed-residual SP contract."""
    from torch.distributed.tensor.parallel import ParallelStyle, RowwiseParallel

    from ..tensor_parallel import (
        SequenceBroadcastParameter,
        SequenceParallelRegion,
        ShardedInputReplicatedWeight,
        SharedInputColwiseParallel,
        _replicated_view,
    )

    class Root(ParallelStyle):
        def _apply(self, module, mesh):
            if mesh.size() not in (2, 4) or module.config.num_key_value_heads % mesh.size():
                raise ValueError("Bagel explicit SP requires TP2/4 dividing its native KV heads")
            state = configure_module(module)
            state.tp_mesh = mesh
            return module

    class Region(SequenceParallelRegion):
        def __init__(self, name):
            super().__init__(gather=(name,))
            self.owner = None

        def _apply(self, module, mesh):
            fresh = Region(self.gather[0])
            fresh.owner = module
            return super(Region, fresh)._apply(module, mesh)

        def _lift(self, name, value, device_mesh, call):
            if self.owner._ff_bagel_parallel_state.active:
                # The model gathers the complete packed residual before routing
                # unequal text/generation subsets through their TP projections.
                return value
            from ..tensor_parallel import _RegionAllReduceGradient

            return _RegionAllReduceGradient.apply(value, device_mesh)

    class Norm(SequenceBroadcastParameter):
        def __init__(self, owner=None):
            super().__init__()
            self.owner = owner

        def _apply(self, module, mesh):
            return super(Norm, Norm(module))._apply(module, mesh)

        def _local_view(self, parameter, mesh):
            if self.owner._ff_bagel_parallel_state.active:
                return super()._local_view(parameter, mesh)
            return _replicated_view(parameter, mesh)

    plan = {"": Root()}
    for branch in ("", "_moe_gen"):
        plan[f"model.layers.*.mlp{branch}"] = Region("hidden_state")
        for norm in ("input_layernorm", "post_attention_layernorm"):
            plan[f"model.layers.*.{norm}{branch}"] = Norm()
        for projection in ("q_proj", "k_proj", "v_proj"):
            plan[f"model.layers.*.self_attn.{projection}{branch}"] = SharedInputColwiseParallel()
        plan[f"model.layers.*.self_attn.o_proj{branch}"] = RowwiseParallel()
        for norm in ("q_norm", "k_norm"):
            plan[f"model.layers.*.self_attn.{norm}{branch}"] = ShardedInputReplicatedWeight()
    plan["model.layers.*.self_attn"] = Region("packed_query_sequence")
    plan["model.norm"] = Norm()
    plan["model.norm_moe_gen"] = Norm()
    return plan
