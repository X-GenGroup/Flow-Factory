# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Reference-instance overrides; canonical full parameters and independent packing.

No production TP/CP/layout/kernel helper is imported. Native model, conditioning,
normalization, rotary application and Flash prefix owners remain explicit.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
from dataclasses import dataclass
from types import MethodType

import torch
from bagel_arithmetic_operators import attention, column, mlp, padded_geometry, row


@dataclass
class Call:
    generation: bool
    lengths: list[int]
    padded: list[int]
    original_rows: list[int]
    cp_rows: list[list[int]]
    text: torch.Tensor
    gen: torch.Tensor


class ArithmeticReference:
    """A serial virtual TP2/CP2 shadow, never installed on the actual model."""

    def __init__(self, model, *, sequence_parallel):
        # The caller supplies the raw canonical native root, not the PEFT proxy.
        assert not hasattr(model, "_ff_bagel_parallel_state")
        assert model.config.num_attention_heads == 28
        assert model.config.num_key_value_heads == 4
        assert model.config.hidden_size == 3584
        assert not model.model.gradient_checkpointing
        self.model = model
        self.sp = bool(sequence_parallel)
        self.context = contextvars.ContextVar("independent_bagel_arithmetic_call", default=None)
        self.observations = {"prefix_calls": 0, "generation_calls": 0}

    def _install_role_norm(self, module, suffix, *, residual_sp):
        """Keep native RMSNorm arithmetic on each physical residual role group.

        Blocks normalize CP-local residuals, split across TP when SP is enabled.
        The final root norms run after the last block gathers TP, so they use CP
        groups only. Prefix calls keep the original whole-pack native function.
        """
        native = module.forward

        @functools.wraps(native)
        def grouped_norm(owner, hidden_state):
            call = self.context.get()
            assert call is not None
            if not call.generation:
                return native(hidden_state)
            global_roles = call.text if suffix == "" else call.gen
            assert hidden_state.shape[0] == global_roles.numel()
            reverse = torch.full(
                (sum(call.padded),), -1, device=hidden_state.device, dtype=torch.long
            )
            reverse[global_roles] = torch.arange(global_roles.numel(), device=hidden_state.device)
            output = None
            degree = 2 if residual_sp and self.sp else 1
            for cp_rows in call.cp_rows:
                assert len(cp_rows) % degree == 0
                size = len(cp_rows) // degree
                for rank in range(degree):
                    # Split the complete CP residual before selecting a role.
                    selected = reverse[cp_rows[rank * size : (rank + 1) * size]]
                    selected = selected[selected >= 0]
                    normalized = native(hidden_state[selected])
                    if output is None:
                        output = normalized.new_zeros(hidden_state.shape)
                    output = output.index_copy(0, selected, normalized)
            assert output is not None
            return output

        module.forward = MethodType(grouped_norm, module)

    def install(self):
        root = self.model.model
        original_root = root.forward_inference

        @functools.wraps(original_root)
        def root_forward(owner, *args, **kwargs):
            arguments = dict(inspect.signature(original_root).bind(*args, **kwargs).arguments)
            arguments = {
                **{
                    name: p.default
                    for name, p in inspect.signature(original_root).parameters.items()
                    if p.default is not inspect.Parameter.empty
                },
                **arguments,
            }
            value = arguments["packed_query_sequence"]
            generation = arguments.get("mode", "und") == "gen" and not arguments.get(
                "update_past_key_values", True
            )
            lengths = [int(n) for n in arguments["query_lens"].tolist()]
            if generation:
                padded, original_rows, _, membership = padded_geometry(
                    lengths, 2, 2 if self.sp else 1
                )
                original_rows_tensor = torch.tensor(
                    original_rows, device=value.device, dtype=torch.long
                )
                expanded = value.new_zeros((sum(padded), value.shape[-1])).index_copy(
                    0, original_rows_tensor, value
                )
                positions = arguments["packed_query_position_ids"]
                positions = positions.new_zeros(sum(padded)).index_copy(
                    0, original_rows_tensor, positions
                )
                text = original_rows_tensor[arguments["packed_text_indexes"]]
                is_gen = torch.ones(sum(padded), device=value.device, dtype=torch.bool)
                is_gen[text] = False
                gen = torch.nonzero(is_gen, as_tuple=True)[0]
                arguments = {
                    **arguments,
                    "packed_query_sequence": expanded,
                    "query_lens": arguments["query_lens"].new_tensor(padded),
                    "packed_query_position_ids": positions,
                    "packed_text_indexes": text,
                    "packed_vae_token_indexes": gen,
                }
                self.observations["generation_calls"] += 1
            else:
                padded, original_rows, membership = lengths, list(range(sum(lengths))), []
                text, gen = arguments.get("packed_text_indexes"), arguments.get(
                    "packed_vae_token_indexes"
                )
                self.observations["prefix_calls"] += 1
            token = self.context.set(
                Call(generation, lengths, padded, original_rows, membership, text, gen)
            )
            try:
                output = original_root(**arguments)
                if generation:
                    output.packed_query_sequence = output.packed_query_sequence[
                        original_rows_tensor
                    ]
                return output
            finally:
                self.context.reset(token)

        root.forward_inference = MethodType(root_forward, root)
        for suffix in ("", "_moe_gen"):
            self._install_role_norm(getattr(root, "norm" + suffix), suffix, residual_sp=False)
        for block in root.layers:
            for suffix in ("", "_moe_gen"):
                for name in ("input_layernorm", "post_attention_layernorm"):
                    self._install_role_norm(getattr(block, name + suffix), suffix, residual_sp=True)
                module = getattr(block, "mlp" + suffix)

                def grouped_mlp(owner, hidden_state, _suffix=suffix):
                    call = self.context.get()
                    assert call is not None
                    if not call.generation:
                        return mlp(owner, hidden_state)
                    global_roles = call.text if _suffix == "" else call.gen
                    reverse = torch.full(
                        (sum(call.padded),), -1, device=hidden_state.device, dtype=torch.long
                    )
                    reverse[global_roles] = torch.arange(
                        global_roles.numel(), device=hidden_state.device
                    )
                    output = torch.zeros_like(hidden_state)
                    for cp_rows in call.cp_rows:
                        selected = reverse[cp_rows]
                        selected = selected[selected >= 0]
                        output = output.index_copy(0, selected, mlp(owner, hidden_state[selected]))
                    return output

                module.forward = MethodType(grouped_mlp, module)
            native_attention = block.self_attn.forward_inference

            @functools.wraps(native_attention)
            def attention_forward(owner, *args, _native=native_attention, **kwargs):
                arguments = dict(inspect.signature(_native).bind(*args, **kwargs).arguments)
                arguments = {
                    **{
                        name: p.default
                        for name, p in inspect.signature(_native).parameters.items()
                        if p.default is not inspect.Parameter.empty
                    },
                    **arguments,
                }
                call = self.context.get()
                assert call is not None
                if call.generation:
                    return self.generation_attention(owner, arguments, call)
                if self.sp:
                    return self.prefix_attention(owner, arguments)
                return _native(**arguments)

            block.self_attn.forward_inference = MethodType(attention_forward, block.self_attn)
        return self

    def projections(self, owner, hidden, text, gen, *, tp_rank, tp_degree, generation):
        heads, kv = 28 // tp_degree, 4 // tp_degree
        if generation:
            hidden = hidden.to(torch.bfloat16)
            q = hidden.new_zeros((hidden.shape[0], heads, 128))
            k, v = [hidden.new_zeros((hidden.shape[0], kv, 128)) for _ in range(2)]
            for indices, suffix in ((text, ""), (gen, "_moe_gen")):
                for kind, target in (("q", q), ("k", k), ("v", v)):
                    projected = column(
                        getattr(owner, kind + "_proj" + suffix), hidden[indices], tp_rank, tp_degree
                    )
                    target[indices] = projected.reshape(-1, target.shape[1], 128)
            q, k = q.float(), k.float()
            for indices, suffix in ((text, ""), (gen, "_moe_gen")):
                q[indices] = getattr(owner, "q_norm" + suffix)(q[indices])
                k[indices] = getattr(owner, "k_norm" + suffix)(k[indices])
        else:
            q = column(owner.q_proj, hidden, tp_rank, tp_degree).reshape(-1, heads, 128)
            k = column(owner.k_proj, hidden, tp_rank, tp_degree).reshape(-1, kv, 128)
            v = column(owner.v_proj, hidden, tp_rank, tp_degree).reshape(-1, kv, 128)
            q, k = owner.q_norm(q), owner.k_norm(k)
        return q, k, v

    def generation_attention(self, owner, arguments, call):
        from flow_factory.models.bagel.modeling.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        hidden = arguments["packed_query_sequence"]
        cos, sin = arguments["packed_query_position_embeddings"]
        cache = arguments["past_key_values"]
        prefix_lengths = [int(n) for n in arguments["key_values_lens"].tolist()]
        degree = 2 if self.sp else 1
        all_attended = []
        local_roles = []
        is_text = torch.zeros(sum(call.padded), device=hidden.device, dtype=torch.bool)
        is_text[call.text] = True
        for membership in call.cp_rows:
            roles = is_text[membership]
            local_roles.append(
                (torch.nonzero(roles, as_tuple=True)[0], torch.nonzero(~roles, as_tuple=True)[0])
            )
        for tp_rank in range(degree):
            queries, keys, values = [], [], []
            for cp_rank, membership in enumerate(call.cp_rows):
                q, k, v = self.projections(
                    owner,
                    hidden[membership],
                    *local_roles[cp_rank],
                    tp_rank=tp_rank,
                    tp_degree=degree,
                    generation=True,
                )
                q, k = apply_rotary_pos_emb(q, k, cos[membership], sin[membership], unsqueeze_dim=1)
                queries.append(q.bfloat16())
                keys.append(k.bfloat16())
                values.append(v.bfloat16())
            kv = 4 // degree
            pk = cache.key_cache[owner.layer_idx][:, tp_rank * kv : (tp_rank + 1) * kv].contiguous()
            pv = cache.value_cache[owner.layer_idx][
                :, tp_rank * kv : (tp_rank + 1) * kv
            ].contiguous()
            all_attended.append(
                attention(
                    queries,
                    keys,
                    values,
                    pk.split(prefix_lengths),
                    pv.split(prefix_lengths),
                    call.lengths,
                    call.padded,
                    sp=self.sp,
                )
            )
        output = hidden.new_zeros(hidden.shape, dtype=torch.bfloat16)
        for cp_rank, membership in enumerate(call.cp_rows):
            local = hidden.new_zeros((len(membership), 3584), dtype=torch.bfloat16)
            for indices, suffix in zip(local_roles[cp_rank], ("", "_moe_gen")):
                inputs = [
                    parts[cp_rank][indices].reshape(indices.numel(), (28 // degree) * 128)
                    for parts in all_attended
                ]
                projected = row(getattr(owner, "o_proj" + suffix), inputs)
                local = local.index_copy(0, indices, projected)
            output = output.index_copy(0, torch.tensor(membership, device=hidden.device), local)
        return output, cache

    def prefix_attention(self, owner, arguments):
        from flow_factory.models.bagel.modeling.bagel import qwen2_navit as native
        from flow_factory.models.bagel.modeling.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        hidden = arguments["packed_query_sequence"]
        cos, sin = arguments["packed_query_position_embeddings"]
        mode = arguments.get("mode", "und")
        query_lens = arguments["query_lens"]
        cache = arguments["past_key_values"]
        prefix_lengths = arguments.get("key_values_lens")
        attended_parts, cache_keys, cache_values = [], [], []
        for tp_rank in range(2):
            q, k, v = self.projections(
                owner,
                hidden,
                arguments.get("packed_text_indexes"),
                arguments.get("packed_vae_token_indexes"),
                tp_rank=tp_rank,
                tp_degree=2,
                generation=mode == "gen",
            )
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
            if cache is not None and cache.key_cache[owner.layer_idx] is not None:
                old_k = cache.key_cache[owner.layer_idx][:, tp_rank * 2 : (tp_rank + 1) * 2]
                old_v = cache.value_cache[owner.layer_idx][:, tp_rank * 2 : (tp_rank + 1) * 2]
                total = int(query_lens.sum() + prefix_lengths.sum())
                merged_k = old_k.new_zeros((total, 2, 128))
                merged_v = old_v.new_zeros((total, 2, 128))
                merged_k[arguments["packed_query_indexes"]] = k
                merged_v[arguments["packed_query_indexes"]] = v
                merged_k[arguments["packed_key_value_indexes"]] = old_k
                merged_v[arguments["packed_key_value_indexes"]] = old_v
                lengths = prefix_lengths + query_lens
            else:
                merged_k, merged_v, lengths = k, v, query_lens
            output = native.flash_attn_varlen_func(
                q=q,
                k=merged_k,
                v=merged_v,
                cu_seqlens_q=torch.nn.functional.pad(query_lens.cumsum(0), (1, 0)).to(torch.int32),
                cu_seqlens_k=torch.nn.functional.pad(lengths.cumsum(0), (1, 0)).to(torch.int32),
                max_seqlen_q=int(query_lens.max()),
                max_seqlen_k=int(lengths.max()),
                causal=arguments.get("is_causal", True),
            ).reshape(hidden.shape[0], 1792)
            attended_parts.append(output)
            cache_keys.append(merged_k)
            cache_values.append(merged_v)
        if mode == "und":
            output = row(owner.o_proj, attended_parts)
        else:
            output = attended_parts[0].new_zeros((hidden.shape[0], 3584))
            for indices, suffix in (
                (arguments["packed_text_indexes"], ""),
                (arguments["packed_vae_token_indexes"], "_moe_gen"),
            ):
                output = output.index_copy(
                    0,
                    indices,
                    row(
                        getattr(owner, "o_proj" + suffix),
                        [part[indices] for part in attended_parts],
                    ),
                )
        if arguments.get("update_past_key_values", True):
            cache.key_cache[owner.layer_idx] = torch.cat(cache_keys, 1)
            cache.value_cache[owner.layer_idx] = torch.cat(cache_values, 1)
        return output, cache
