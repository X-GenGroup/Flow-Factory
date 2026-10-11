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

# src/flow_factory/models/bagel/pipeline.py
"""Bagel pseudo-pipeline.

Lightweight container that mimics the diffusers DiffusionPipeline interface. ``BagelAdapter``
keeps it as ``adapter.pipeline`` and registers its modules with a ``PseudoPipelineRuntime``
for component management (get_component, set_component, freeze, LoRA, offload).

Bagel differs from diffusers pipelines in that:
  - Text encoding is internal to the Bagel model (no separate text_encoder)
  - The VAE is a custom autoencoder (not diffusers' AutoencoderKL)
  - Image understanding uses a ViT (SiglipVisionModel) inside the Bagel model
  - Context is built via KV-cache, not via separate encoder embeddings

Component Mapping:
  pipeline.bagel        →  Bagel model (LLM + ViT + generation heads)
  pipeline.transformer  →  Bagel's language model (``bagel.language_model``)
  pipeline.vae          →  Custom autoencoder (encode/decode images)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from .modeling.autoencoder import load_ae
from .modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)

logger = logging.getLogger(__name__)


def _resolve_model_path(model_path: str, **kwargs) -> str:
    """Resolve *model_path* to a local directory.

    If *model_path* is already an existing local directory it is returned
    as-is.  Otherwise it is treated as a HuggingFace Hub repo-id
    (e.g. ``"ByteDance-Seed/BAGEL-7B-MoT"``) and downloaded via
    ``huggingface_hub.snapshot_download``.

    Args:
        model_path: Local directory or HuggingFace Hub repo-id.
        **kwargs: Only ``revision``, ``cache_dir``, ``token``, ``local_dir``,
            ``allow_patterns``, ``ignore_patterns``, ``force_download``,
            ``resume_download``, and ``local_files_only`` are forwarded to
            ``snapshot_download``; other keys are ignored.
    """
    if os.path.isdir(model_path):
        return model_path

    # Filter kwargs that snapshot_download accepts
    _SNAPSHOT_KEYS = {
        "revision",
        "cache_dir",
        "token",
        "local_dir",
        "allow_patterns",
        "ignore_patterns",
        "force_download",
        "resume_download",
        "local_files_only",
    }
    dl_kwargs = {k: v for k, v in kwargs.items() if k in _SNAPSHOT_KEYS}

    local_dir = snapshot_download(repo_id=model_path, **dl_kwargs)
    return local_dir


class BagelPseudoPipeline:
    """Pseudo-pipeline holding Bagel components under diffusers-compatible names.

    This is NOT a real DiffusionPipeline; it is the compatibility container kept as
    ``adapter.pipeline``, while ``PseudoPipelineRuntime`` owns the explicit component
    mapping built by ``BagelAdapter.build_component_runtime``.
    """

    def __init__(
        self,
        bagel: Bagel,
        vae: nn.Module,
        scheduler: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        self.bagel = bagel
        self.transformer = bagel.language_model
        self.vae = vae
        self.scheduler = scheduler

        # Store the original BagelConfig for reference
        self._bagel_config = config or getattr(self.bagel, "config", None)

    # ---- DiffusionPipeline-like interface stubs ----
    def maybe_free_model_hooks(self):
        """No-op: Bagel doesn't use diffusers model hooks."""
        pass

    @property
    def device(self) -> torch.device:
        """Infer device from transformer parameters."""
        try:
            return next(self.transformer.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Infer dtype from transformer parameters."""
        try:
            return next(self.transformer.parameters()).dtype
        except StopIteration:
            return torch.bfloat16

    @property
    def components(self) -> Dict[str, nn.Module]:
        """Return default modules managed like DiffusionPipeline components."""
        return {
            "bagel": self.bagel,
            "vae": self.vae,
        }

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        vae_path: Optional[str] = None,
        low_cpu_mem_usage: bool = False,
        component_dtypes: Optional[Mapping[str, torch.dtype]] = None,
        **kwargs,
    ) -> "BagelPseudoPipeline":
        """Construct Bagel components from a pretrained checkpoint.

        ``model_path`` can be either:
          - A **local directory** containing Bagel checkpoint files, or
          - A **HuggingFace Hub repo-id** (e.g. ``"ByteDance-Seed/BAGEL-7B-MoT"``),
            which will be automatically downloaded and cached.

        Expected directory layout (BAGEL-7B-MoT style)::

            model_path/
            ├── llm_config.json
            ├── vit_config.json
            ├── ae.safetensors        # VAE weights
            ├── ema.safetensors       # Bagel model weights
            ├── tokenizer files …
            └── …

        Args:
            model_path: Local path **or** HuggingFace repo-id.
            vae_path: Optional separate path for VAE weights.
                Defaults to ``<model_path>/ae.safetensors``.
            low_cpu_mem_usage: If True, build the Bagel model under ``init_empty_weights``
                and skip loading ``ema.safetensors``, deferring weight materialization
                to the caller (for multi-GPU dispatch). The VAE is always loaded.
            component_dtypes: Optional resolved dtype per Bagel runtime component.
            **kwargs: Extra arguments.  HuggingFace download keys
                (``revision``, ``cache_dir``, ``token``, …) are forwarded to
                ``snapshot_download``; model-building keys (``layer_module``,
                ``latent_patch_size``, …) are used directly.
        """
        # 1. Resolve to a local directory (download if needed)
        model_path = _resolve_model_path(model_path, **kwargs)

        # 2. LLM config
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.pad_token_id = kwargs.get("pad_token_id", llm_config.eos_token_id)
        llm_config.layer_module = kwargs.get("layer_module", "Qwen2MoTDecoderLayer")

        # 3. ViT config
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = kwargs.get("vit_rope", False)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1  # Default for inference

        # 4. VAE
        ae_path = vae_path or os.path.join(model_path, "ae.safetensors")
        vae_model, vae_config = load_ae(local_path=ae_path)

        # 5. Bagel config
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=kwargs.get("vit_max_num_patch_per_side", 70),
            connector_act=kwargs.get("connector_act", "gelu_pytorch_tanh"),
            latent_patch_size=kwargs.get("latent_patch_size", 2),
            max_latent_size=kwargs.get("max_latent_size", 64),
        )

        # 6. Build models
        if low_cpu_mem_usage:
            with init_empty_weights():
                language_model = Qwen2ForCausalLM(llm_config)
                vit_model = SiglipVisionModel(vit_config)
                model = Bagel(language_model, vit_model, config)
                model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                    vit_config, meta=True
                )
        else:
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

            # Load weights
            ema_path = os.path.join(model_path, "ema.safetensors")
            if os.path.exists(ema_path):
                state_dict = load_file(ema_path)
                model.load_state_dict(state_dict, strict=False)

        # 7. Cast components to their resolved dtypes
        component_dtypes = component_dtypes or {}
        bagel_dtype = component_dtypes.get("bagel")
        transformer_dtype = component_dtypes.get("transformer")
        if bagel_dtype is not None:
            model.to(dtype=bagel_dtype)
        if transformer_dtype is not None and transformer_dtype != bagel_dtype:
            model.language_model.to(dtype=transformer_dtype)
        if "vae" in component_dtypes:
            vae_model.to(dtype=component_dtypes["vae"])

        return cls(
            bagel=model,
            vae=vae_model,
            config=config,
        )
