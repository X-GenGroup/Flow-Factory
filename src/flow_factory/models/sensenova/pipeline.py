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

"""SenseNova's explicit pseudo-pipeline and differentiable denoiser wrapper.

SenseNova-U1 1.0/1.5 are distributed as custom Transformers composite models
rather than diffusers pipelines. The wrapper keeps the official NEO-Unify model
intact while making its image-generation path an explicit Flow-Factory component.
Tokenization and vision encoding live inside the composite model; there is no
standalone Flow-Factory text encoder or VAE. The checkpoint config selects the
U1.0 MLP flow head or U1.5 pixel head automatically.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .modeling.neo_unify.modeling_neo_chat import NEOChatModel


class SenseNovaTextPrefix(NamedTuple):
    """Native text-only prefix operands routed through the prepared component."""

    input_ids: torch.Tensor
    indexes: torch.Tensor
    attention_mask: Dict[str, Any]


class SenseNovaImagePrefix(NamedTuple):
    """Native ordered-reference prefix construction inside the prepared component."""

    tokenizer: Any
    query: str
    pixel_values: Optional[torch.Tensor] = None
    grid_hw: Optional[torch.Tensor] = None


class SenseNovaPrefixOutput(NamedTuple):
    past_key_values: Any
    indexes: torch.Tensor
    hidden_states: torch.Tensor


class SenseNovaDenoiser(nn.Module):
    """Expose SenseNova's image flow-matching prediction as one trainable module."""

    def __init__(self, model: NEOChatModel):
        super().__init__()
        self.model = model

    @property
    def config(self):
        """Delegate the checkpoint config for adapter/runtime introspection."""
        return self.model.config

    @property
    def device(self) -> torch.device:
        """Expose the underlying model device like a diffusers component."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Expose the underlying model dtype like a diffusers component."""
        return next(self.parameters()).dtype

    def set_attention_backend(self, backend: str) -> None:
        """Declare native ownership for the model's prefix and generation attention."""
        if backend != "native":
            raise ValueError(f"SenseNova's native attention contract does not support {backend!r}")
        self._attention_backend = backend

    def enable_gradient_checkpointing(self) -> None:
        """Checkpoint read-only generation blocks, preserving native prefix writers."""
        from functools import partial

        from torch.utils.checkpoint import checkpoint

        from .modeling.neo_unify.modeling_qwen3 import Qwen3DecoderLayer

        if any(
            type(block) is not Qwen3DecoderLayer for block in self.model.language_model.model.layers
        ):
            raise ValueError(
                "SenseNova cache-preserving checkpointing requires dense U1.0/U1.5 blocks"
            )
        for block in self.model.language_model.model.layers:
            block.gradient_checkpointing = True
            block._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)

    def disable_gradient_checkpointing(self) -> None:
        """Leave backend checkpoint wrappers as the sole owner when they are selected."""
        for module in self.model.language_model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False

    def forward(
        self,
        *,
        latents: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        indexes_image: Optional[torch.Tensor] = None,
        attention_mask: Optional[Dict[str, Any]] = None,
        image_size: Optional[tuple[int, int]] = None,
        noise_scale: Optional[float] = None,
        prefix: Optional[SenseNovaTextPrefix | SenseNovaImagePrefix] = None,
    ) -> torch.Tensor | SenseNovaPrefixOutput:
        """Predict the native SenseNova clean-minus-noise velocity.

        The Flow-Factory adapter calls this for one generated sample at a time.
        That sample may carry multiple ordered reference images in its prefix.
        Keeping independent samples separate preserves each variable-length KV
        cache and avoids padding-dependent changes to the RoPE indexes.
        """
        if prefix is not None:
            if any(
                value is not None
                for value in (
                    latents,
                    timestep,
                    past_key_values,
                    indexes_image,
                    attention_mask,
                    image_size,
                    noise_scale,
                )
            ):
                raise ValueError(
                    "SenseNova prefix and denoising operands cannot share one invocation"
                )
            with torch.no_grad():
                if isinstance(prefix, SenseNovaTextPrefix):
                    indexes = prefix.indexes
                    cache, hidden = self.model._t2i_prefix_forward(*prefix)
                elif isinstance(prefix, SenseNovaImagePrefix):
                    embeds, indexes, mask = self.model._build_it2i_inputs(*prefix)
                    cache, hidden = self.model._it2i_prefix_forward(embeds, indexes, mask)
                else:
                    raise TypeError(f"Unsupported SenseNova prefix input: {type(prefix).__name__}")
            return SenseNovaPrefixOutput(cache, indexes, hidden)
        missing = [
            name
            for name, value in (
                ("latents", latents),
                ("timestep", timestep),
                ("indexes_image", indexes_image),
                ("attention_mask", attention_mask),
                ("image_size", image_size),
                ("noise_scale", noise_scale),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"SenseNova denoising is missing required operands: {missing}")
        if latents is None or latents.ndim != 4 or latents.shape[0] != 1:
            raise ValueError(
                "SenseNovaDenoiser expects latents with shape (1, 3, H, W); "
                f"received {None if latents is None else tuple(latents.shape)}."
            )

        model = self.model
        merge_size = int(1 / model.downsample_ratio)
        patch_size = model.patch_size
        token_h = image_size[1] // (patch_size * merge_size)
        token_w = image_size[0] // (patch_size * merge_size)
        grid_h = image_size[1] // patch_size
        grid_w = image_size[0] // patch_size
        image_token_num = token_h * token_w

        z = model.patchify(latents, patch_size * merge_size)
        image_input = model.patchify(latents, patch_size, channel_first=True)
        grid_hw = torch.tensor([[grid_h, grid_w]], device=latents.device, dtype=torch.long)
        image_embeds = model.extract_feature(
            image_input.view(grid_h * grid_w, -1),
            gen_model=True,
            grid_hw=grid_hw,
        ).view(1, image_token_num, -1)

        native_t = timestep.reshape(-1)
        if native_t.numel() != 1:
            raise ValueError(
                "SenseNovaDenoiser expects one native timestep per call; "
                f"received shape {tuple(timestep.shape)}."
            )
        t_expanded = native_t.expand(image_token_num)
        timestep_embeddings = model.fm_modules["timestep_embedder"](t_expanded).view(
            1, image_token_num, -1
        )
        if model.add_noise_scale_embedding:
            noise_scale_tensor = torch.full_like(
                t_expanded, noise_scale / model.noise_scale_max_value
            )
            timestep_embeddings = timestep_embeddings + model.fm_modules["noise_scale_embedder"](
                noise_scale_tensor
            ).view(1, image_token_num, -1)
        image_embeds = image_embeds + timestep_embeddings

        outputs = model.language_model.model(
            inputs_embeds=image_embeds,
            image_gen_indicators=torch.ones(
                (1, image_token_num), dtype=torch.bool, device=latents.device
            ),
            indexes=indexes_image,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            update_cache=False,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state[:, -image_token_num:]

        if model.use_pixel_head:
            hidden_2d = hidden_states.view(1, token_h, token_w, -1)
            hidden_2d = torch.einsum("b h w c -> b c h w", hidden_2d).contiguous()
            smoothed = model.fm_modules["fm_head"](hidden_2d)
            smoothed = smoothed.view(
                1,
                3,
                token_h,
                patch_size * merge_size,
                token_w,
                patch_size * merge_size,
            )
            smoothed = torch.einsum("b c h p w q -> b h w p q c", smoothed)
            x_pred = smoothed.contiguous().view(1, image_token_num, -1)
        elif model.use_deep_fm_head:
            x_pred = model.fm_modules["fm_head"](
                hidden_states.reshape(image_token_num, -1),
                native_t.expand(image_token_num),
            ).view(1, image_token_num, -1)
        else:
            from ._sequence_parallel import gather_pointwise_output

            x_pred = gather_pointwise_output(
                self, model.fm_modules["fm_head"](hidden_states), image_token_num
            ).view(1, image_token_num, -1)

        return (x_pred - z) / (1 - native_t).clamp_min(model.config.t_eps)


class SenseNovaPseudoPipeline:
    """Component container for a custom SenseNova-U1 1.0/1.5 checkpoint."""

    def __init__(
        self,
        model: NEOChatModel,
        tokenizer: Any,
        scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.transformer = SenseNovaDenoiser(model)
        self.tokenizer = tokenizer
        self.scheduler = scheduler

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    @property
    def components(self) -> Dict[str, nn.Module]:
        """Return the one canonical trainable component."""
        return {"transformer": self.transformer}

    def maybe_free_model_hooks(self) -> None:
        """Compatibility no-op; SenseNova has no diffusers hooks."""

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        low_cpu_mem_usage: bool = False,
        **kwargs: Any,
    ) -> "SenseNovaPseudoPipeline":
        """Load a SenseNova-U1 1.0/1.5 checkpoint from the Hub or a local directory."""
        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {}) or {}
        model = NEOChatModel.from_pretrained(
            model_path,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=True,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, **tokenizer_kwargs
        )
        return cls(model=model, tokenizer=tokenizer)
