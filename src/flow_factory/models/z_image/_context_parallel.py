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

"""Z-Image unified-token CP around the existing main-stack sequence overlay."""

import inspect

import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor

from ..parallelism.context import ContextParallelBoundary
from ..parallelism.hooks import require_native_hook


class ZImageContextAttnProcessor(ZSingleStreamAttnProcessor):
    """Keep real head geometry and pad only independent attention operands for Ulysses."""

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, freqs_cis=None
    ):
        if self._parallel_config is None:
            return super().__call__(
                attn, hidden_states, encoder_hidden_states, attention_mask, freqs_cis
            )
        q, k, v = [
            projection(hidden_states).unflatten(-1, (attn.heads, -1))
            for projection in (attn.to_q, attn.to_k, attn.to_v)
        ]
        q = attn.norm_q(q) if attn.norm_q is not None else q
        k = attn.norm_k(k) if attn.norm_k is not None else k
        if freqs_cis is not None:

            def rotate(value):
                with torch.amp.autocast(value.device.type, enabled=False):
                    complex_value = torch.view_as_complex(
                        value.float().reshape(*value.shape[:-1], -1, 2)
                    )
                    return (
                        torch.view_as_real(complex_value * freqs_cis.unsqueeze(2))
                        .flatten(3)
                        .to(value.dtype)
                    )

            q, k = rotate(q), rotate(k)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        heads = q.shape[2]
        padding = (-heads) % self._parallel_config.context_parallel_config.ulysses_degree
        if padding:
            q, k, v = [F.pad(value, (0, 0, 0, padding)) for value in (q, k, v)]
        output = dispatch_attention_fn(
            q,
            k,
            v,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        output = output[:, :, :heads].flatten(2, 3).to(q.dtype)
        return attn.to_out[1](attn.to_out[0](output))


def install_z_image_context_parallel(module, config):
    """Split the unified main stack; native image/text refiners remain CP-replicated.

    The refiners run before native image/text concatenation. Keeping their
    outputs replicated avoids an intermediate local-slice gather across a
    changed token partition. Their partial gradients still belong to FSDP.
    """
    from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

    signature = inspect.signature(type(module).forward)

    def validate_inputs(target, args, kwargs):
        bound = signature.bind_partial(target, *args, **kwargs)
        values = bound.arguments
        images, texts = values.get("x"), values.get("cap_feats")
        if (
            not isinstance(images, list)
            or not images
            or not all(isinstance(x, torch.Tensor) and x.ndim == 4 for x in images)
        ):
            raise ValueError(
                "Z-Image CP requires ordinary T2I latent tensors; omni inputs are unsupported"
            )
        if (
            not isinstance(texts, list)
            or len(images) != len(texts)
            or not all(
                isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[0] > 0 for x in texts
            )
        ):
            raise ValueError("Z-Image CP requires one nonempty rank-2 text tensor per image")
        if any(
            values.get(name) is not None
            for name in ("controlnet_block_samples", "siglip_feats", "image_noise_mask")
        ):
            raise ValueError("Z-Image CP does not support ControlNet or omni conditioning")
        if values.get("patch_size", 2) != 2 or values.get("f_patch_size", 1) != 1:
            raise ValueError("Z-Image CP requires patch_size=2 and f_patch_size=1")
        lengths = []
        for image, text in zip(images, texts):
            if (
                image.shape[0] != target.config.in_channels
                or image.shape[1] != 1
                or any(v < 2 or v % 2 for v in image.shape[-2:])
            ):
                raise ValueError(
                    f"Z-Image CP requires one frame and patch-aligned latents; received {tuple(image.shape)}"
                )
            image_length = image.shape[-2] * image.shape[-1] // 4
            lengths.append(((image_length + 31) // 32 + (text.shape[0] + 31) // 32) * 32)
        if max(lengths) % config.ulysses_degree:
            raise ValueError(
                f"Z-Image CP requires padded tokens={max(lengths)} divisible by cp_size={config.ulysses_degree}"
            )
        return args, kwargs

    module.register_forward_pre_hook(validate_inputs, with_kwargs=True)
    plan = {
        "layers.0": {"x": ContextParallelInput(split_dim=1, expected_dims=3)},
        "layers.*": {"freqs_cis": ContextParallelInput(split_dim=1, expected_dims=3)},
        **{
            f"all_final_layer.{name}": ContextParallelOutput(gather_dim=1, expected_dims=3)
            for name in module.all_final_layer
        },
    }
    module.enable_parallelism(config=config, cp_plan=plan)
    for block in (*module.noise_refiner, *module.context_refiner):
        block.attention.processor._parallel_config = None


def compose_z_image_sequence_boundaries(module):
    """Apply CP before the main-stack SP scatter, and gather them in reverse order."""
    from diffusers.hooks.context_parallel import ContextParallelGatherHook, ContextParallelSplitHook
    from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

    split = require_native_hook(
        module.layers[0],
        "cp_input---layers.0",
        ContextParallelSplitHook,
        metadata={"x": ContextParallelInput(split_dim=1, expected_dims=3)},
        error="Z-Image CP/SP requires its native main-stack input split",
    )
    outputs = [
        require_native_hook(
            target,
            f"cp_output---all_final_layer.{name}",
            ContextParallelGatherHook,
            metadata=[ContextParallelOutput(gather_dim=1, expected_dims=3)],
            error="Z-Image CP/SP requires native per-patch output gathers",
        )
        for name, target in module.all_final_layer.items()
    ]
    return (split.move_input(), *(binding.move_output() for binding in outputs))
