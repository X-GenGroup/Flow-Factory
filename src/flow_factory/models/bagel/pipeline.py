"""
Bagel Pseudo-Pipeline

Lightweight wrapper that mimics the diffusers DiffusionPipeline interface,
allowing BaseAdapter's component management (get_component, set_component,
freeze, LoRA, offload) to work unchanged with Bagel's non-diffusers
architecture.

Bagel differs from standard diffusers pipelines in several ways:

1. **No separate text_encoder** — text encoding is internal to the Bagel
   model via ``language_model.embed_tokens`` + KV-cache prefill.
2. **Custom AutoEncoder** — not diffusers' ``AutoencoderKL``, loaded from
   ``ae.safetensors`` via ``load_ae()``.
3. **ViT for image understanding** — ``SiglipVisionModel`` inside the
   Bagel model processes condition images for I2I tasks.
4. **KV-cache context** — prompt/image conditioning is built via
   incremental KV-cache updates, not separate encoder embeddings.
5. **Mixture-of-Transformer (MoT)** — optional MoE routing with separate
   expert weights for understanding vs generation tokens.

Component Mapping:
    pipeline.bagel        →  Bagel model (full: LLM + ViT + gen heads)
    pipeline.transformer  →  Qwen2ForCausalLM (the LLM backbone)
    pipeline.vae          →  Custom AutoEncoder (encode/decode images)
    pipeline.tokenizer    →  Qwen2Tokenizer (with Bagel special tokens)
    pipeline.new_token_ids → Dict of special token IDs (bos, eos, soi, eoi)
    pipeline.vae_transform → ImageTransform(1024, 512, 16) for VAE path
    pipeline.vit_transform → ImageTransform(980, 224, 14) for ViT path

Loading:
    Supports both local directories and HuggingFace Hub repo-ids
    (e.g. ``"ByteDance-Seed/BAGEL-7B-MoT"``).  Hub models are
    automatically downloaded via ``huggingface_hub.snapshot_download``.
"""
from __future__ import annotations

import os
import logging
from typing import Optional, Any

import torch
import torch.nn as nn
from .modeling.bagel import Bagel, BagelConfig
from .modeling.qwen2 import Qwen2Tokenizer
from .modeling.autoencoder import AutoEncoder
from .data.data_utils import add_special_tokens
from .data.transforms import ImageTransform

logger = logging.getLogger(__name__)


def _resolve_model_path(model_path: str, **kwargs) -> str:
    """Resolve *model_path* to a local directory.

    If *model_path* is already an existing local directory it is returned
    as-is.  Otherwise it is treated as a HuggingFace Hub repo-id
    (e.g. ``"ByteDance-Seed/BAGEL-7B-MoT"``) and downloaded via
    ``huggingface_hub.snapshot_download``.

    Accepted ``kwargs`` forwarded to ``snapshot_download``:
        revision, cache_dir, token, local_dir, allow_patterns,
        ignore_patterns, force_download, resume_download …
    """
    if os.path.isdir(model_path):
        return model_path

    from huggingface_hub import snapshot_download

    # Filter kwargs that snapshot_download accepts
    _SNAPSHOT_KEYS = {
        "revision", "cache_dir", "token", "local_dir",
        "allow_patterns", "ignore_patterns",
        "force_download", "resume_download", "local_files_only",
    }
    dl_kwargs = {k: v for k, v in kwargs.items() if k in _SNAPSHOT_KEYS}

    local_dir = snapshot_download(repo_id=model_path, **dl_kwargs)
    return local_dir


class BagelPseudoPipeline:
    """
    Pseudo-pipeline holding Bagel components under diffusers-compatible names.

    This is **not** a real ``DiffusionPipeline``; it is a thin namespace
    that the ``BaseAdapter`` can query via ``getattr(self.pipeline, name)``
    for component management (freeze, LoRA, device offload, etc.).

    The pipeline holds:
      - ``bagel``: The full ``Bagel`` model (LLM + ViT + generation heads).
      - ``transformer``: Alias to ``bagel.language_model`` (``Qwen2ForCausalLM``),
        exposed for BaseAdapter's ``self.transformer`` property.
      - ``vae``: Custom ``AutoEncoder`` for image encode/decode.
      - ``tokenizer``: ``Qwen2Tokenizer`` with Bagel's special tokens.
      - ``new_token_ids``: Dict mapping special token names to their IDs
        (``bos_token_id``, ``eos_token_id``, ``start_of_image``,
        ``end_of_image``).
      - ``vae_transform`` / ``vit_transform``: ``ImageTransform`` instances
        for resizing images to VAE and ViT input sizes respectively.

    Attributes:
        _bagel_config: The original ``BagelConfig`` for reference.
    """

    def __init__(
        self,
        bagel: Bagel,
        vae: AutoEncoder,
        tokenizer: Qwen2Tokenizer,
        scheduler: Optional[Any] = None,
        config: Optional[BagelConfig] = None,
        new_token_ids: Optional[Any] = None,
    ):
        self.bagel = bagel
        self.transformer = bagel.language_model
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids

        # Store the original BagelConfig for reference
        self._bagel_config = config or getattr(self.bagel, "config", None)

        # VAE transform: max_size=1024, min_size=512, patch=16
        self.vae_transform = ImageTransform(1024, 512, 16)
        # ViT transform: max_size=980, min_size=224, patch=14
        self.vit_transform = ImageTransform(980, 224, 14)

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

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        vae_path: Optional[str] = None,
        low_cpu_mem_usage: bool = False,
        **kwargs,
    ) -> "BagelPseudoPipeline":
        """
        Construct Bagel components from a pretrained checkpoint.

        ``model_path`` can be either:
          - A **local directory** containing Bagel checkpoint files, or
          - A **HuggingFace Hub repo-id** (e.g. ``"ByteDance-Seed/BAGEL-7B-MoT"``),
            which will be automatically downloaded and cached via
            ``huggingface_hub.snapshot_download``.

        Expected directory layout (BAGEL-7B-MoT style)::

            model_path/
            ├── llm_config.json       # Qwen2Config
            ├── vit_config.json       # SiglipVisionConfig
            ├── ae.safetensors        # AutoEncoder weights
            ├── ema.safetensors       # Bagel model weights (LLM + ViT + heads)
            ├── tokenizer files …     # Qwen2Tokenizer assets
            └── …

        Build sequence:
            1. Resolve ``model_path`` to a local directory.
            2. Load LLM config (``Qwen2Config``) and ViT config
               (``SiglipVisionConfig``) from JSON files.
            3. Load VAE via ``load_ae()``.
            4. Construct ``BagelConfig`` and build the ``Bagel`` model.
            5. Load weights from ``ema.safetensors``.
            6. Initialise tokenizer and add Bagel's special tokens.
            7. Resize embeddings if new tokens were added.

        Args:
            model_path: Local path **or** HuggingFace repo-id.
            vae_path: Optional separate path for VAE weights.
                      Defaults to ``<model_path>/ae.safetensors``.
            low_cpu_mem_usage: If True, use ``init_empty_weights`` to defer
                               weight materialization (for multi-GPU dispatch).
            **kwargs: Extra arguments.  HuggingFace download keys
                      (``revision``, ``cache_dir``, ``token``, …) are
                      forwarded to ``snapshot_download``; model-building
                      keys (``layer_module``, ``latent_patch_size``, …)
                      are used directly.

        Returns:
            Fully initialised ``BagelPseudoPipeline`` instance.
        """
        from .modeling.bagel import (
            BagelConfig, Bagel,
            Qwen2Config, Qwen2ForCausalLM,
            SiglipVisionConfig, SiglipVisionModel,
        )
        from .modeling.autoencoder import load_ae
        from safetensors.torch import load_file

        # ── Resolve to local directory (download if needed) ──────────
        model_path = _resolve_model_path(model_path, **kwargs)

        # ---- LLM Config ----
        llm_config = Qwen2Config.from_json_file(
            os.path.join(model_path, "llm_config.json")
        )
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = kwargs.get("layer_module", "Qwen2MoTDecoderLayer")

        # ---- ViT Config ----
        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(model_path, "vit_config.json")
        )
        vit_config.rope = kwargs.get("vit_rope", False)
        vit_config.num_hidden_layers = (
            vit_config.num_hidden_layers - 1
        )  # Default for inference

        # ---- VAE ----
        ae_path = vae_path or os.path.join(model_path, "ae.safetensors")
        vae_model, vae_config = load_ae(local_path=ae_path)

        # ---- Bagel Config ----
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

        # ---- Build Models ----
        if low_cpu_mem_usage:
            from accelerate import init_empty_weights

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

        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        if num_new_tokens > 0:
            model.language_model.resize_token_embeddings(len(tokenizer))
            model.config.llm_config.vocab_size = len(tokenizer)
            model.language_model.config.vocab_size = len(tokenizer)

        return cls(
            bagel=model,
            vae=vae_model,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            config=config,
        )