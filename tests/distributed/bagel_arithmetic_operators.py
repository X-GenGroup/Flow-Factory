# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Independent virtual TP2 arithmetic over ordinary canonical full parameters."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _parts(module):
    """Expose only the fixture's unmerged linear + ordinary warm LoRA contract."""
    if not hasattr(module, "base_layer"):
        assert isinstance(module, torch.nn.Linear)
        return module, []
    assert isinstance(module.base_layer, torch.nn.Linear)
    assert not module.merged and not module.disable_adapters
    adapters = []
    for name in module.active_adapters:
        assert name in module.lora_A and name in module.lora_B
        assert not module.use_dora.get(name, False)
        assert getattr(module.lora_dropout[name], "p", 0) == 0
        a, b = module.lora_A[name], module.lora_B[name]
        assert isinstance(a, torch.nn.Linear) and isinstance(b, torch.nn.Linear)
        assert a.bias is None and b.bias is None
        adapters.append((a.weight, b.weight, module.scaling[name]))
    assert adapters
    return module.base_layer, adapters


def column(module, value: torch.Tensor, rank: int, degree: int = 2) -> torch.Tensor:
    """Compute one output-feature block; A stays whole and B follows output rows."""
    assert degree in (1, 2) and 0 <= rank < degree
    base, adapters = _parts(module)
    assert base.weight.shape[0] % degree == 0
    rows = base.weight.shape[0] // degree
    section = slice(rank * rows, (rank + 1) * rows)
    bias = None if base.bias is None else base.bias[section]
    result = F.linear(value, base.weight[section].contiguous(), bias)
    output_dtype = result.dtype
    for a, b, scale in adapters:
        adapted = value.to(a.dtype)
        result = result + F.linear(F.linear(adapted, a), b[section].contiguous()) * scale
    return result.to(output_dtype)


def row(module, values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Reduce two input contractions; rowwise LoRA reduces A before replicated B."""
    assert len(values) in (1, 2)
    base, adapters = _parts(module)
    assert base.weight.shape[1] % len(values) == 0
    width = base.weight.shape[1] // len(values)
    assert all(value.shape[-1] == width for value in values)
    partials = [
        F.linear(value, base.weight[:, rank * width : (rank + 1) * width].contiguous())
        for rank, value in enumerate(values)
    ]
    result = partials[0] if len(partials) == 1 else partials[0] + partials[1]
    # Bias belongs to the complete output once, independently of TP degree.
    if base.bias is not None:
        result = result + base.bias.to(result.dtype)
    output_dtype = result.dtype
    for a, b, scale in adapters:
        pieces = [
            F.linear(value.to(a.dtype), a[:, rank * width : (rank + 1) * width].contiguous())
            for rank, value in enumerate(values)
        ]
        low_rank = pieces[0] if len(pieces) == 1 else pieces[0] + pieces[1]
        result = result + F.linear(low_rank, b) * scale
    return result.to(output_dtype)


def mlp(module, value: torch.Tensor, degree: int = 2) -> torch.Tensor:
    hidden = [
        module.act_fn(column(module.gate_proj, value, rank, degree))
        * column(module.up_proj, value, rank, degree)
        for rank in range(degree)
    ]
    return row(module.down_proj, hidden)


def padded_geometry(lengths: Sequence[int], cp: int, sp: int):
    """Construct canonical padded samples and independent CP membership arrays."""
    assert cp == 2 and sp in (1, 2)
    assert lengths and all(type(n) is int and n >= 0 for n in lengths)
    padded = [((n + cp * sp - 1) // (cp * sp)) * cp * sp for n in lengths]
    original_to_padded, padded_to_original = [], []
    cp_rows = [[] for _ in range(cp)]
    original_offset = padded_offset = 0
    for length, size in zip(lengths, padded):
        original_to_padded.extend(range(padded_offset, padded_offset + length))
        padded_to_original.extend(range(original_offset, original_offset + length))
        padded_to_original.extend([-1] * (size - length))
        chunk = size // cp
        for rank in range(cp):
            cp_rows[rank].extend(
                range(padded_offset + rank * chunk, padded_offset + (rank + 1) * chunk)
            )
        original_offset += length
        padded_offset += size
    return padded, original_to_padded, padded_to_original, cp_rows


def attention(
    query_parts,
    key_parts,
    value_parts,
    prefix_k,
    prefix_v,
    lengths,
    padded_lengths,
    *,
    sp: bool,
    mask_padding: bool = True,
    dense_fp64: bool = False,
):
    """One TP head block: virtual CP2 all-to-all geometry with native SDPA.

    Each part contains the complete CP-local sequence (all packed samples) at
    that rank. Prefix tensors are independently constructed canonical caches.
    Query/key/value are post-native-norm/RoPE operands. No distributed helper,
    production packing function or actual tensor is used by this reference.
    """
    assert len(query_parts) == len(key_parts) == len(value_parts) == 2
    heads, width = query_parts[0].shape[-2:]
    kv_heads = key_parts[0].shape[-2]
    assert heads == (14 if sp else 28) and heads // kv_heads == 7 and width == 128
    assert len(lengths) == len(padded_lengths) == len(prefix_k) == len(prefix_v)
    outputs = [[], []]
    offsets = [0, 0]
    for sample, (length, padded) in enumerate(zip(lengths, padded_lengths)):
        local = padded // 2
        if not local:
            for rank in range(2):
                outputs[rank].append(query_parts[rank].new_empty((0, heads, width)))
            continue
        pk, pv = prefix_k[sample], prefix_v[sample]
        prefix_length = pk.shape[0]
        prefix_local = (prefix_length + 1) // 2
        missing = 2 * prefix_local - prefix_length
        pk = torch.cat([pk, pk.new_zeros((missing, kv_heads, width))])
        pv = torch.cat([pv, pv.new_zeros((missing, kv_heads, width))])
        queries, keys, values, masks = [], [], [], []
        for rank in range(2):
            selection = slice(offsets[rank], offsets[rank] + local)
            q, k, v = (parts[rank][selection] for parts in (query_parts, key_parts, value_parts))
            offsets[rank] += local
            queries.append(torch.cat([q, q.new_zeros((prefix_local, heads, width))]))
            keys.append(
                torch.cat(
                    [pk[rank * prefix_local : (rank + 1) * prefix_local], k]
                ).repeat_interleave(7, 1)
            )
            values.append(
                torch.cat(
                    [pv[rank * prefix_local : (rank + 1) * prefix_local], v]
                ).repeat_interleave(7, 1)
            )
            masks.extend([rank * prefix_local + j < prefix_length for j in range(prefix_local)])
            masks.extend([rank * local + j < length for j in range(local)])
        query, key, value = (torch.cat(parts) for parts in (queries, keys, values))
        mask = torch.tensor(masks, device=query.device, dtype=torch.bool)[None, None, None]
        if not mask_padding:
            mask = torch.ones_like(mask)
        head_outputs = []
        for owner in range(2):
            hs = slice(owner * (heads // 2), (owner + 1) * (heads // 2))
            q, k, v = [x[:, hs].contiguous()[None].permute(0, 2, 1, 3) for x in (query, key, value)]
            if dense_fp64:
                assert q.dtype == k.dtype == v.dtype == torch.float64
                logits = (q @ k.transpose(-2, -1)) / (width**0.5)
                attended = torch.softmax(logits.masked_fill(~mask, -torch.inf), -1) @ v
            else:
                attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            head_outputs.append(attended.permute(0, 2, 1, 3)[0])
        attended = torch.cat(head_outputs, dim=1)
        stride = local + prefix_local
        for rank in range(2):
            outputs[rank].append(attended[rank * stride : rank * stride + local])
    return [torch.cat(parts) for parts in outputs]
