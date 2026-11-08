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

"""LTX-owned joint audio/video boundaries around native Diffusers CP autograd."""

from collections.abc import Mapping
from typing import Any

import torch
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2Attention,
    LTX2AudioVideoAttnProcessor,
    LTX2PerturbedAttnProcessor,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
)
from torch import nn

from ...utils.dist import context_parallel_geometry, tensor_parallel_geometry
from ..parallelism.context import ContextParallelBoundary, ContextParallelLayout
from ..parallelism.hooks import require_native_hook
from ..parallelism.plan import sequence_parallel_enabled


class LTX2ContextParallelLayout(ContextParallelLayout):
    """Stable identity for the joint video/audio native transformer layout."""

    JOINT_AUDIO_VIDEO = "ltx2_joint_audio_video"


class LTX2CrossModalCPProcessor(LTX2AudioVideoAttnProcessor):
    """Preserve native rectangular cross-attention through square Ulysses backward.

    Diffusers0.41 reshapes Q and KV backward with the query length. Only padded
    attention operands have equal lengths here; false key-mask entries remove
    padded KV, and discarded query rows contribute no gradient. Trim before the
    output projection so its TP/SP scatter still sees the actual query geometry.
    """

    def __call__(
        self,
        attn: LTX2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None or attention_mask is not None:
            raise ValueError(
                "LTX2 CP cross-modal attention requires a source stream without a custom mask"
            )
        query = attn.norm_q(attn.to_q(hidden_states))
        key = attn.norm_k(attn.to_k(encoder_hidden_states))
        value = attn.to_v(encoder_hidden_states)
        if query_rotary_emb is not None:
            rotary = (
                apply_split_rotary_emb
                if attn.rope_type == "split"
                else apply_interleaved_rotary_emb
            )
            query = rotary(query, query_rotary_emb)
            key = rotary(key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        query_tokens, key_tokens = query.shape[1], key.shape[1]
        if self._parallel_config is not None and query_tokens != key_tokens:
            tokens = max(query_tokens, key_tokens)

            def pad(tensor: torch.Tensor) -> torch.Tensor:
                return torch.cat(
                    (
                        tensor,
                        tensor.new_zeros(
                            tensor.shape[0], tokens - tensor.shape[1], *tensor.shape[2:]
                        ),
                    ),
                    dim=1,
                )

            query, key, value = pad(query), pad(key), pad(value)
            if key_tokens < tokens:
                # Native Ulysses gathers this local mask in the same rank-major
                # order as the padded keys after all-to-all.
                attention_mask = (torch.arange(tokens, device=key.device) < key_tokens)[
                    None, None, None, :
                ]
        output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )[:, :query_tokens].to(query.dtype)
        if attn.to_gate_logits is not None:
            gates = 2.0 * torch.sigmoid(attn.to_gate_logits(hidden_states))
            output = output * gates.unsqueeze(-1)
        output = output.flatten(2, 3)
        return attn.to_out[1](attn.to_out[0](output))


def _validate_native_ltx_processors(module: nn.Module) -> None:
    for index, block in enumerate(module.transformer_blocks):
        # Native 2.3 constructs every attention processor from this flag; only
        # self-attention receives perturbation arguments during STG execution.
        expected = (
            LTX2PerturbedAttnProcessor if block.perturbed_attn else LTX2AudioVideoAttnProcessor
        )
        for name, attention in block.named_modules():
            if isinstance(attention, LTX2Attention) and type(attention.processor) is not expected:
                raise ValueError(
                    f"LTX2 CP requires the native {expected.__name__} at transformer_blocks.{index}.{name}, "
                    f"received {type(attention.processor).__name__}"
                )


def configure_ltx_context_parallel(module: nn.Module) -> None:
    """Install the family adaptation only on native cross-modal attention stacks."""
    _validate_native_ltx_processors(module)
    for block in module.transformer_blocks:
        for name in ("audio_to_video_attn", "video_to_audio_attn"):
            getattr(block, name).set_processor(LTX2CrossModalCPProcessor())


def ltx_context_parallel_plan(module: nn.Module) -> dict[str, Any]:
    """Split both modalities, their actual rotary axes and per-token timesteps."""
    plan = {
        "": {
            "timestep": ContextParallelInput(split_dim=1, expected_dims=2),
            "audio_timestep": ContextParallelInput(split_dim=1, expected_dims=2),
        },
        "proj_in": {0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True)},
        "audio_proj_in": {0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True)},
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "audio_proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    for name in ("rope", "audio_rope", "cross_attn_rope", "cross_attn_audio_rope"):
        split = getattr(module, name).rope_type == "split"
        plan[name] = {
            index: ContextParallelInput(
                split_dim=2 if split else 1, expected_dims=4 if split else 3, split_output=True
            )
            for index in (0, 1)
        }
    return plan


def install_ltx_context_parallel(module: nn.Module, config: Any) -> None:
    """Retain native Ulysses for generated streams and native replicated text KV."""
    module.enable_parallelism(config=config, cp_plan=ltx_context_parallel_plan(module))
    for block in module.transformer_blocks:
        for name in ("attn2", "audio_attn2"):
            getattr(block, name).processor._parallel_config = None


def compose_ltx_sequence_boundaries(module: nn.Module) -> tuple[ContextParallelBoundary, ...]:
    """Gather each modality over SP before its native CP output gather."""
    from diffusers.hooks.context_parallel import ContextParallelGatherHook

    gathers = [
        require_native_hook(
            getattr(module, name),
            f"cp_output---{name}",
            ContextParallelGatherHook,
            error=f"LTX2 TP/SP+CP requires the native {name} output gather",
        )
        for name in ("proj_out", "audio_proj_out")
    ]
    return tuple(binding.move_output() for binding in gathers)


def validate_ltx_context_parallel_config(
    config: Mapping[str, Any], degree: int, tp_size: int = 1
) -> None:
    """Validate both native attention streams before weight loading or hooks."""
    for name in ("num_attention_heads", "audio_num_attention_heads"):
        heads = config[name]
        if (
            isinstance(heads, bool)
            or not isinstance(heads, int)
            or heads < 1
            or heads % (degree * tp_size)
        ):
            raise ValueError(
                f"LTX2 CP requires {name}={heads} divisible by cp_size={degree} * tp_size={tp_size}"
            )
    if config.get("num_layers", 0) < 1:
        raise ValueError("LTX2 CP requires native joint audio/video transformer blocks")
    if config.get("rope_type") != "split":
        raise ValueError(
            f"LTX2 CP requires the published split rotary layout, received {config.get('rope_type')!r}"
        )
    if (config.get("patch_size"), config.get("patch_size_t")) != (1, 1):
        raise ValueError("LTX2 CP requires native unit spatial and temporal transformer patches")
    channels = tuple(
        config.get(name)
        for name in ("in_channels", "out_channels", "audio_in_channels", "audio_out_channels")
    )
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in channels
        )
        or len(set(channels)) != 1
    ):
        raise ValueError(
            f"LTX2 CP requires equal positive video/audio latent channels, received {channels}"
        )


def validate_ltx_context_parallel_module(module: nn.Module, degree: int, tp_size: int = 1) -> None:
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    if not isinstance(module, LTX2VideoTransformer3DModel):
        raise ValueError(
            f"LTX2 CP requires LTX2VideoTransformer3DModel, received {type(module).__name__}"
        )
    validate_ltx_context_parallel_config(module.config, degree, tp_size)
    _validate_native_ltx_processors(module)


def validate_ltx_context_parallel_inputs(
    adapter: Any,
    latents: torch.Tensor,
    video_tokens: int,
    audio_tokens: int,
    video_shape: tuple[int, int, int],
    *,
    video_coords: torch.Tensor | None = None,
    audio_coords: torch.Tensor | None = None,
    conditioning_mask: torch.Tensor | None = None,
) -> None:
    """Reject inconsistent packed streams before any prepared model collective."""
    degree, _ = context_parallel_geometry(adapter.accelerator)
    if degree == 1:
        return
    tp_size, _ = tensor_parallel_geometry(adapter.accelerator)
    divisor = degree * (
        tp_size
        if sequence_parallel_enabled(adapter, adapter.config.parallelism_args.sequence_parallel)
        else 1
    )
    if latents.ndim != 3 or latents.shape[0] != 1:
        raise ValueError(
            f"LTX2 CP requires one packed video/audio sample, received {tuple(latents.shape)}"
        )
    for name, tokens in (("video", video_tokens), ("audio", audio_tokens)):
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens < 1
            or tokens % divisor
        ):
            raise ValueError(
                f"LTX2 CP {name} tokens={tokens} must be positive and divisible by CP/SP degree={divisor}"
            )
    frames, height, width = video_shape
    if video_tokens != frames * height * width or latents.shape[1] != video_tokens + audio_tokens:
        raise ValueError(
            f"LTX2 CP packed geometry {tuple(latents.shape)} disagrees with video_shape={video_shape}, video_tokens={video_tokens}, audio_tokens={audio_tokens}"
        )
    for name, coords, axes, tokens in (
        ("video", video_coords, 3, video_tokens),
        ("audio", audio_coords, 1, audio_tokens),
    ):
        if coords is not None and tuple(coords.shape) != (1, axes, tokens, 2):
            raise ValueError(
                f"LTX2 CP {name} coordinates must have shape {(1, axes, tokens, 2)}, received {tuple(coords.shape)}"
            )
    if conditioning_mask is not None:
        expected = torch.zeros(
            (1, video_tokens), device=conditioning_mask.device, dtype=conditioning_mask.dtype
        )
        expected[:, : height * width] = 1
        if conditioning_mask.shape != expected.shape or not torch.equal(
            conditioning_mask, expected
        ):
            raise ValueError(
                f"LTX2 I2AV CP requires the first-frame mask at {height * width}/{video_tokens} video tokens, received {tuple(conditioning_mask.shape)}"
            )
