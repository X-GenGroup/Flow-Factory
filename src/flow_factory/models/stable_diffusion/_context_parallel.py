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

"""SD3 joint-attention geometry around native Ulysses and image-only SP."""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import JointAttnProcessor2_0

from ..context_parallel import ContextParallelBoundary


class SD35ContextAttnProcessor(JointAttnProcessor2_0):
    """Retain SD3 projections and norms, dispatching only attention through native CP.

    Medium keeps its second image-only attention. Large's 38 real heads remain
    19 per TP2 rank; independent zero heads pad only the communication operand
    to 20 and are discarded before the native output projection. No parameter,
    attention-head declaration, or checkpoint tensor changes shape.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        cp_text_tokens=None,
        *args,
        **kwargs,
    ):
        if self._parallel_config is None:
            return super().__call__(
                attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs
            )
        if attention_mask is not None:
            raise ValueError(
                "SD3 CP owns its padding mask; external attention masks are unsupported"
            )
        batch, image_tokens = hidden_states.shape[:2]

        def project(states, q_proj, k_proj, v_proj, q_norm, k_norm):
            q, k, v = [
                projection(states)
                .view(batch, -1, attn.heads, attn.inner_dim // attn.heads)
                .transpose(1, 2)
                for projection in (q_proj, k_proj, v_proj)
            ]
            # The existing SD3 TP plan declares the normalization head axis as 1.
            q = q_norm(q) if q_norm is not None else q
            k = k_norm(k) if k_norm is not None else k
            return q, k, v

        q, k, v = project(hidden_states, attn.to_q, attn.to_k, attn.to_v, attn.norm_q, attn.norm_k)
        cp = self._parallel_config.context_parallel_config
        mask = None
        if encoder_hidden_states is not None:
            if cp_text_tokens is None:
                raise ValueError(
                    "SD3 CP requires original text length from its model input boundary"
                )
            tq, tk, tv = project(
                encoder_hidden_states,
                attn.add_q_proj,
                attn.add_k_proj,
                attn.add_v_proj,
                attn.norm_added_q,
                attn.norm_added_k,
            )
            q, k, v = [
                torch.cat((image, text), dim=2) for image, text in ((q, tq), (k, tk), (v, tv))
            ]
            text_tokens = encoder_hidden_states.shape[1]
            valid_text = (
                torch.arange(text_tokens, device=k.device) + cp._ulysses_local_rank * text_tokens
                < cp_text_tokens
            )
            # Native Ulysses gathers this local [image, text] mask in exactly
            # the same rank-major order as its Q/K/V all-to-all.
            mask = torch.cat(
                (torch.ones(image_tokens, dtype=torch.bool, device=k.device), valid_text)
            )[None, None, None, :]
        q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]
        heads = q.shape[2]
        padding = (-heads) % cp.ulysses_degree
        if padding:
            q, k, v = [F.pad(tensor, (0, 0, 0, padding)) for tensor in (q, k, v)]
        output = dispatch_attention_fn(
            q,
            k,
            v,
            attn_mask=mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        output = output[:, :, :heads].reshape(batch, -1, heads * q.shape[-1]).to(q.dtype)
        image = output[:, :image_tokens]
        image = attn.to_out[1](attn.to_out[0](image))
        if encoder_hidden_states is None:
            return image
        text = output[:, image_tokens:]
        if not attn.context_pre_only:
            text = attn.to_add_out(text)
        return image, text


def install_sd35_context_parallel(module, config):
    """Pad only masked text positions, then delegate sequence slicing to Diffusers."""
    from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

    signature = inspect.signature(type(module).forward)

    def prepare_inputs(target, args, kwargs):
        bound = signature.bind_partial(target, *args, **kwargs)
        values = bound.arguments
        text, image = values.get("encoder_hidden_states"), values.get("hidden_states")
        if not isinstance(text, torch.Tensor) or text.ndim != 3 or text.shape[1] < 1:
            raise ValueError("SD3 CP requires nonempty rank-3 text embeddings")
        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 4
            or image.shape[0] != text.shape[0]
        ):
            raise ValueError("SD3 CP requires image latents and matching text batch")
        patch = target.config.patch_size
        if (
            any(size % patch for size in image.shape[-2:])
            or (image.shape[-2] // patch) * (image.shape[-1] // patch) % config.ulysses_degree
        ):
            raise ValueError(
                f"SD3 CP requires patch-aligned image tokens divisible by cp_size={config.ulysses_degree}; received {tuple(image.shape)}"
            )
        if (
            values.get("joint_attention_kwargs")
            or values.get("block_controlnet_hidden_states") is not None
            or values.get("skip_layers")
        ):
            raise ValueError(
                "SD3 CP does not support custom attention, ControlNet residuals or skipped blocks"
            )
        length = text.shape[1]
        values["encoder_hidden_states"] = F.pad(text, (0, 0, 0, (-length) % config.ulysses_degree))
        values["joint_attention_kwargs"] = {"cp_text_tokens": length}
        return bound.args[1:], bound.kwargs

    module.register_forward_pre_hook(prepare_inputs, with_kwargs=True)
    module.enable_parallelism(
        config=config,
        cp_plan={
            "pos_embed": {0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True)},
            "context_embedder": {
                0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True)
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        },
    )


def compose_sd35_sequence_boundaries(module):
    """Complete image SP before native CP output gathering and unpatchification."""
    from diffusers.hooks.context_parallel import ContextParallelGatherHook
    from diffusers.models._modeling_parallel import ContextParallelOutput

    output = module.proj_out
    registry = output._diffusers_hook
    name = "cp_output---proj_out"
    gather = registry.get_hook(name)
    if not isinstance(gather, ContextParallelGatherHook) or gather.metadata != [
        ContextParallelOutput(gather_dim=1, expected_dims=3)
    ]:
        raise ValueError("SD3 CP/SP requires its native image output gather")
    registry.remove_hook(name, recurse=False)

    def gather_image(target, args, result):
        return gather.post_forward(target, result)

    handle = output.register_forward_hook(gather_image)
    return (ContextParallelBoundary(output, "output", gather_image, handle),)
