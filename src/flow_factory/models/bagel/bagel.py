"""
Bagel Model Adapter for Flow-Factory

Integrates ByteDance's Bagel (unified multimodal model with Mixture-of-Transformer
architecture) into the Flow-Factory RL fine-tuning framework.

Architecture Mapping:
    ┌──────────────────────────────┬──────────────────────────────────┐
    │ Flow-Factory Interface       │  Bagel Component                 │
    ├──────────────────────────────┼──────────────────────────────────┤
    │ self.transformer             │  Qwen2ForCausalLM (MoT decoder)  │
    │ self.pipeline.bagel          │  Bagel (LLM + ViT + gen heads)   │
    │ self.vae                     │  Custom AutoEncoder              │
    │ self.tokenizer               │  Qwen2Tokenizer (+ special toks) │
    │ encode_prompt()              │  Deferred — returns raw strings   │
    │ encode_image()               │  Standardize PIL → tensor batch   │
    │ _build_gen_context()         │  KV-cache prefill (text/ViT/VAE) │
    │ _forward_flow()              │  LLM velocity prediction + CFG   │
    │ forward()                    │  Single denoise step + scheduler  │
    │ inference()                  │  Full denoising loop              │
    │ decode_latents()             │  VAE decode (packed → PIL)        │
    └──────────────────────────────┴──────────────────────────────────┘

Supported Tasks:
    - Text-to-Image (T2I): prompt → image  (BagelSample)
    - Image(s)-to-Image (I2I): images + prompt → image  (BagelI2ISample)

Training-mode Caveats:
    Bagel's Qwen2Model.forward() dispatches to ``forward_train()`` or
    ``forward_inference()`` based on ``self.training``.  During RL training
    we always need the inference-path signatures (packed_query_sequence,
    KV-caches …), so we **temporarily switch the model to eval mode**
    via ``_eval_mode()`` for every LLM forward call.  Gradients still
    flow because we do NOT wrap with ``@torch.no_grad`` — autograd is
    orthogonal to train/eval mode.  The only behavioural difference is
    that dropout is disabled, which is desirable for generation anyway.

CFG (Classifier-Free Guidance):
    Bagel uses **dual CFG** with separate text-CFG and image-CFG branches,
    each with its own KV-cache context.  CFG is gated by a sigma interval
    and re-normalised (global / channel / text_channel) to stabilise
    high guidance scales.

Timestep Convention:
    - Bagel natively uses sigmas in [0, 1].
    - The scheduler and GRPO trainer operate in [0, 1000].
    - ``forward()`` converts [0, 1000] → [0, 1] before calling
      ``_forward_flow()``, and passes [0, 1000] to ``scheduler.step()``.
"""
from __future__ import annotations

import os
import random
from contextlib import contextmanager
from copy import deepcopy
from typing import Union, List, Dict, Any, Optional, Tuple, Literal, ClassVar
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

from accelerate import Accelerator

from ...samples import T2ISample, I2ISample
from ..abc import BaseAdapter
from ...hparams import Arguments
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    SDESchedulerOutput,
)
from ...utils.base import filter_kwargs
from ...utils.image import (
    ImageSingle,
    ImageBatch,
    MultiImageBatch,   
    standardize_image_batch
)
from ...utils.trajectory_collector import (
    TrajectoryCollector,
    CallbackCollector,
    TrajectoryIndicesType,
    create_trajectory_collector,
    create_callback_collector,
)
from ...utils.logger_utils import setup_logger

from .data.data_utils import pil_img2rgb
from .pipeline import BagelPseudoPipeline
from .modeling.bagel import Bagel
from .modeling.bagel.qwen2_navit import NaiveCache

logger = setup_logger(__name__)

VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

# ============================================================================
# Sample Dataclasses
# ============================================================================

@dataclass
class BagelSample(T2ISample):
    """
    Sample class for Bagel T2I generation.

    Extends T2ISample with ``image_shape`` for latent-to-pixel unpacking.
    Stores denoising trajectory (all_latents, log_probs) plus packed
    tensor metadata needed to reconstruct KV-cache context during training.
    """

    _shared_fields: ClassVar[frozenset[str]] = frozenset({
        "image_shape",
    })
    # Image shape (H, W) before downsampling — required for VAE decode
    image_shape: Optional[Tuple[int, int]] = None


@dataclass
class BagelI2ISample(I2ISample):
    """
    Sample class for Bagel Image(s)-to-Image generation.

    Identical to BagelSample but inherits from I2ISample to carry
    ``condition_images`` for the image-conditioned generation path.
    """

    _shared_fields: ClassVar[frozenset[str]] = frozenset({
        "image_shape",
    })
    image_shape: Optional[Tuple[int, int]] = None


# ============================================================================
# BagelAdapter
# ============================================================================

class BagelAdapter(BaseAdapter):
    """
    Flow-Factory adapter for ByteDance's Bagel unified multimodal model.

    Wraps ``BagelPseudoPipeline`` (which holds the Bagel model, VAE, and
    tokenizer) behind the standard BaseAdapter interface so that existing
    RL trainers (GRPO, DiffusionNFT, AWM, …) work without modification.

    Key architectural differences from diffusers-based adapters:

    1. **No separate text_encoder** — text encoding is internal to the
       Bagel model via ``language_model.embed_tokens`` + KV-cache prefill.
       ``encode_prompt()`` therefore returns raw strings; actual encoding
       is deferred to ``_build_gen_context()``.

    2. **ViT for image understanding** — ``SiglipVisionModel`` inside
       Bagel processes condition images, producing tokens that are injected
       into the KV-cache alongside text tokens.

    3. **Packed latent sequences** — denoising operates on variable-length
       packed token sequences with position-aware indexing, not fixed-shape
       spatial tensors.  Batch size is effectively 1 per forward call; the
       trainer handles outer batching.

    4. **Dual CFG** — separate pre-computed KV caches for text-only
       (``cfg_text``) and image-only (``cfg_img``) conditions, with
       configurable renormalisation (global / channel / text_channel).

    5. **Eval-mode dispatch** — the LLM must be in ``eval()`` for correct
       ``forward_inference()`` dispatch, even during training.  The
       ``_eval_mode()`` context manager handles this transparently.

    6. **Mixture-of-Transformer (MoT)** — when ``use_moe=True``, separate
       expert weights (``*_moe_gen``) are used for generation tokens,
       which are the default LoRA targets.

    Attributes:
        pipeline: ``BagelPseudoPipeline`` holding all model components.
        scheduler: ``FlowMatchEulerDiscreteSDEScheduler`` with shifted
                   timestep schedule (default shift=3.0).
    """

    def __init__(self, config: Arguments, accelerator: Accelerator):
        # Load tokenizer and transforms before super().__init__
        # because load_pipeline may need them, and base __init__ calls load_pipeline
        self._model_path = config.model_args.model_name_or_path
        super().__init__(config, accelerator)
        self.pipeline: BagelPseudoPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler
    
    # ======================== Pipeline & Scheduler ========================

    def load_pipeline(self) -> BagelPseudoPipeline:
        """Load the Bagel model and VAE into a pseudo-pipeline.

        Delegates to ``BagelPseudoPipeline.from_pretrained()`` which
        supports both local directories and HuggingFace Hub repo-ids.
        """
        pipeline = BagelPseudoPipeline.from_pretrained(
            self._model_path,
            low_cpu_mem_usage=False,
            **self.model_args.extra_kwargs,
        )
        return pipeline

    def load_scheduler(self) -> FlowMatchEulerDiscreteSDEScheduler:
        """
        Create a FlowMatchEulerDiscreteScheduler for Bagel's flow matching.

        Bagel uses a shifted timestep schedule:
            σ_shifted = shift × σ / (1 + (shift − 1) × σ)
        with default shift=3.0.  The scheduler operates in [0, 1000];
        the adapter converts to/from Bagel's native [0, 1] sigma space.
        """
        scheduler_kwargs = {
            "num_train_timesteps": 1000,
            "shift": 3.0
        }
        if hasattr(self.config, "scheduler_args") and self.config.scheduler_args:
            scheduler_kwargs.update(self.config.scheduler_args.to_dict())

        scheduler = FlowMatchEulerDiscreteSDEScheduler(**scheduler_kwargs)

        return scheduler

    # ======================== Module Management ========================

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Bagel's MoT generation experts.

        Targets the ``*_moe_gen`` projections in self-attention and MLP
        of the Qwen2 decoder layers, which handle generation-mode tokens.
        """
        return [
            "self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen",
            "self_attn.v_proj_moe_gen", "self_attn.o_proj_moe_gen",
            "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
        ]

    @property
    def tokenizer(self) -> Any:
        """Qwen2Tokenizer with Bagel's added special tokens."""
        return self.pipeline.tokenizer

    @property
    def text_encoder_names(self) -> List[str]:
        """Bagel has no separate text encoder; encoding is inside the transformer."""
        return []

    @property
    def text_encoders(self) -> List[nn.Module]:
        """Empty — text encoding is internal to the Bagel LLM."""
        return []

    @property
    def text_encoder(self) -> Optional[nn.Module]:
        """None — text encoding is internal to the Bagel LLM."""
        return None

    @property
    def preprocessing_modules(self) -> List[str]:
        """Modules needed for preprocessing (VAE for decode)."""
        return ["vae"]

    @property
    def inference_modules(self) -> List[str]:
        """Modules needed for inference: the full Bagel model + VAE."""
        return ["bagel", "transformer", "vae"]

    # ─────────────── Convenience accessors ───────────────

    @property
    def bagel_model(self) -> nn.Module:
        """The underlying Bagel nn.Module (alias for transformer)."""
        return self.get_component("transformer")

    @property
    def bagel_config(self):
        """The BagelConfig from the loaded model."""
        return self.pipeline._bagel_config

    # ======================== Eval-mode context manager ========================

    @property
    def mode(self) -> str:
        """Get current mode."""
        return self._mode

    def eval(self):
        """Set all components to evaluation mode."""
        super().eval()  # Set base adapter mode
        self.transformer.eval()
        self.pipeline.bagel.eval()
        self.pipeline.vae.eval()

    def rollout(self, *args, **kwargs):
        """Set model to rollout mode (uses eval behaviour for all components)."""
        self.eval()
        if hasattr(self.scheduler, 'rollout'):
            self.scheduler.rollout(*args, **kwargs)

    def train(self, mode: bool = True):
        """Set trainable components to training mode."""
        super().train(mode)  # Set base adapter mode
        if mode:
            self.transformer.train()
            self.pipeline.bagel.train()
    
    @contextmanager
    def _eval_mode(self, module: nn.Module):
        """
        Temporarily switch a module to eval mode, restoring afterwards.

        Required because Bagel's Qwen2Model.forward() dispatches to
        ``forward_train()`` vs ``forward_inference()`` based on
        ``self.training``.  We always need the inference dispatch
        (packed_query_sequence / KV-cache API), even during RL training.

        Note: eval mode only affects dropout / batchnorm; autograd is
        **not** affected, so gradients still flow normally.
        """
        was_training = module.training
        module.eval()
        try:
            yield
        finally:
            if was_training:
                module.train()

    # ======================== Encoding ========================

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
    ) -> Dict[str, Any]:
        """
        Tokenize text prompts for Bagel.

        Unlike diffusers adapters, Bagel's prompt encoding is **deferred**
        to ``_build_gen_context()`` where it becomes part of KV-cache
        prefill.  This method only normalises the input to a list.

        Returns:
            Dict with ``prompt`` key mapping to ``List[str]``.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        return {"prompt": prompt}

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image], List[List[Image.Image]]],
    ) -> Optional[Dict[str, Any]]:
        """
        Pre-process condition images for Bagel I2I tasks.

        Converts PIL images to RGB tensors and normalises to
        ``List[List[Tensor]]``.  The actual ViT/VAE encoding happens
        later in ``_build_gen_context()``.

        Returns:
            Dict with ``condition_images`` key, or None if no images.
        """
        if images is None:
            return None

        # Normalize to List[List[Image.Image]]
        if isinstance(images, Image.Image):
            images = [[images]]
        elif isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
            images = [[img] for img in images]

        # Convert to RGB
        processed = [
            standardize_image_batch(img_list, output_type='pt')
            for img_list in images
        ]
        return {"condition_images": processed}

    def encode_video(self, videos: Any) -> None:
        """Bagel does not support video input."""
        return None

    # ======================== Decoding ========================

    def decode_latents(
        self,
        latents: torch.Tensor,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Decode packed latent tokens back into PIL images.

        Reshapes from packed ``(seq_len, patch_dim)`` representation to
        spatial ``(C, H, W)`` layout, then runs the VAE decoder.

        Args:
            latents: Packed latent tensor of shape ``(seq_len, patch_dim)``
                     or ``(B, seq_len, patch_dim)`` for a batch.
            image_shape: ``(H, W)`` of the target image (pre-downsampling).

        Returns:
            Single PIL Image or list of PIL Images.
        """
        bagel = self.pipeline.bagel
        vae = self.pipeline.vae

        p = bagel.latent_patch_size
        ch = bagel.latent_channel
        ds = bagel.latent_downsample

        single = latents.dim() == 2
        if single:
            latents = latents.unsqueeze(0)

        images = []
        for lat in latents:
            H, W = image_shape
            h, w = H // ds, W // ds
            # (seq, patch_dim) → (1, C, H_lat, W_lat)
            lat = lat.reshape(1, h, w, p, p, ch)
            lat = torch.einsum("nhwpqc->nchpwq", lat)
            lat = lat.reshape(1, ch, h * p, w * p)
            decoded = vae.decode(lat.to(vae.dtype if hasattr(vae, 'dtype') else torch.bfloat16))
            decoded = (decoded * 0.5 + 0.5).clamp(0, 1)[0].float()
            images.append(decoded)

        if single:
            return images[0]
        return images

    # ======================== Context Building ========================

    def _build_gen_context(
        self,
        prompt: str,
        condition_images: Optional[ImageBatch] = None,
        think: bool = False,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Build KV-cache contexts for generation with dual CFG.

        Constructs three separate KV-cache contexts by prefilling
        different subsets of the input:

        - **gen_context**: full context (text + condition images)
        - **cfg_text_context**: images only (text dropped → text-CFG)
        - **cfg_img_context**: text only (images dropped → image-CFG)

        For I2I tasks, condition images are processed first (VAE + ViT
        encoding into the KV-cache), then the text prompt is appended.

        The model is temporarily switched to eval mode via
        ``_eval_mode()`` so that Qwen2's ``forward_inference()`` path
        is used for all KV-cache updates.

        Args:
            prompt: Text prompt string.
            condition_images: Optional list of PIL images for I2I.
            think: If True, prepend a chain-of-thought system prompt.

        Returns:
            Tuple of (gen_context, cfg_text_context, cfg_img_context),
            each a dict with keys ``kv_lens``, ``ropes``,
            ``past_key_values`` (NaiveCache).
        """

        bagel = self.pipeline.bagel
        num_layers = bagel.config.llm_config.num_hidden_layers

        def _init_ctx():
            return {
                "kv_lens": [0],
                "ropes": [0],
                "past_key_values": NaiveCache(num_layers),
            }

        gen_context = _init_ctx()
        cfg_text_context = _init_ctx()
        cfg_img_context = _init_ctx()

        # ── Must use eval mode for Qwen2 dispatch (forward_inference) ──
        with self._eval_mode(bagel):
            # --- Optional thinking prompt ---
            if think:
                system_prompt = GEN_THINK_SYSTEM_PROMPT # Here, only for generation tasks.
                gen_context = self._update_context_text(system_prompt, gen_context)
                cfg_img_context = self._update_context_text(system_prompt, cfg_img_context)

            # --- Process interleaved inputs ---
            # For I2I: images go first, then text
            if condition_images is not None:
                condition_images = standardize_image_batch(condition_images, output_type='pil')
                for img in condition_images:
                    img_tensor = self.vae_transform.resize_transform(pil_img2rgb(img))
                    gen_context = self._update_context_image(img_tensor, gen_context)
                    cfg_text_context = deepcopy(gen_context)

            # Text always comes last (before generation)
            cfg_text_context = deepcopy(gen_context)
            gen_context = self._update_context_text(prompt, gen_context)
            cfg_img_context = self._update_context_text(prompt, cfg_img_context)

        return gen_context, cfg_text_context, cfg_img_context

    # ─── _update_context_text ───
    @torch.no_grad()
    def _update_context_text(self, text: str, gen_context: Dict) -> Dict:
        """Add text tokens to the KV-cache context.
        
        Tokenizes ``text``, prepends/appends BOS/EOS, runs a causal
        LLM forward to update the NaiveCache.

        IMPORTANT: Caller must ensure the model is in eval mode
        (via ``self._eval_mode``) for correct Qwen2 dispatch.
        """
        bagel = self.pipeline.bagel
        device = self.device
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]

        generation_input, kv_lens, ropes = bagel.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.pipeline.tokenizer,
            new_token_ids=self.pipeline.new_token_ids,
        )
        # Move all tensors to model device before forward
        generation_input = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in generation_input.items()
        }
        past_key_values = bagel.forward_cache_update_text(
            gen_context["past_key_values"], **generation_input
        )
        return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past_key_values}

    # ─── _update_context_image ───
    @torch.no_grad()
    def _update_context_image(
        self,
        image_tensor,
        gen_context: Dict,
        vae: bool = True,
        vit: bool = True,
    ) -> Dict:
        """Add image tokens (VAE + ViT) to the KV-cache context.

        Processes the image through both the VAE path (latent tokens)
        and the ViT path (understanding tokens), updating the
        NaiveCache with both sets of embeddings.

        Args:
            image_tensor: Pre-processed image tensor.
            gen_context: Current context dict to update.
            vae: Whether to add VAE-encoded latent tokens.
            vit: Whether to add ViT-encoded understanding tokens.
        """
        bagel = self.pipeline.bagel
        vae_model = self.pipeline.vae
        device = self.device
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        past_key_values = gen_context["past_key_values"]

        if vae:
            gen_input, kv_lens, ropes = bagel.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image_tensor],
                transforms=self.pipeline.vae_transform,
                new_token_ids=self.pipeline.new_token_ids,
            )
            gen_input = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in gen_input.items()
            }
            past_key_values = bagel.forward_cache_update_vae(
                vae_model, past_key_values, **gen_input
            )

        if vit:
            gen_input, kv_lens, ropes = bagel.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image_tensor],
                transforms=self.pipeline.vit_transform,
                new_token_ids=self.pipeline.new_token_ids,
            )
            gen_input = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in gen_input.items()
            }
            past_key_values = bagel.forward_cache_update_vit(
                past_key_values, **gen_input
            )

        return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past_key_values}

    # ======================== Flow Forward (grad-safe) ========================

    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
    ):
        """
        Gradient-safe flow velocity prediction with dual CFG.

        Constructs the packed sequence (text embeddings + noised VAE
        latents), runs the LLM, and extracts the velocity prediction
        ``v_t`` at the VAE token positions.  When CFG scales > 1.0,
        additionally runs text-CFG and/or image-CFG branches and
        applies renormalisation.

        This method is called by ``forward()`` and does **not** use
        ``@torch.no_grad``, so gradients flow through for RL training.

        Args:
            x_t: Noised latent tokens ``(seq_len, patch_dim)``.
            timestep: Current sigma in [0, 1], shape ``(B,)``.
            cfg_renorm_type: One of ``"global"``, ``"channel"``,
                ``"text_channel"``.
            cfg_type: Currently only ``"parallel"`` is supported.

        Returns:
            ``v_t``: Flow velocity prediction ``(num_vae_tokens, patch_dim)``.
        """
        packed_text_embedding = self.pipeline.transformer.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.pipeline.bagel.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.pipeline.bagel.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.pipeline.bagel.time_embedder(timestep)
        x_t = self.pipeline.bagel.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.pipeline.bagel.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }
        output = self.transformer(
            packed_query_sequence=packed_sequence, # packed_sequence.shape=[1026, 3584]
            query_lens=packed_seqlens,   # packed_seqlens=tensor([1026])
            packed_query_position_ids=packed_position_ids,  # packed_position_ids.shape=[1026]
            packed_query_indexes=packed_indexes,    # packed_indexes.shape=[1026]
            past_key_values=past_key_values,     
            key_values_lens=key_values_lens,    # key_values_lens=tensor([9])
            packed_key_value_indexes=packed_key_value_indexes,   # packed_key_value_indexes.shape=[9]
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.pipeline.bagel.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]
        if cfg_text_scale > 1.0:
            cfg_text_output = self.transformer(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,   # cfg_text_packed_position_ids.shape=[1026]
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.pipeline.bagel.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]
        if cfg_img_scale > 1.0:
            cfg_img_output = self.transformer(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.pipeline.bagel.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    # ======================== Inference ========================

    @torch.no_grad()
    def inference(
        self,
        # Generation params
        num_inference_steps: int = 50,
        height: int = 1024,
        width: int = 1024,
        # Prompt
        prompt: Union[str, List[str]] = None,
        # Condition images for I2I
        condition_images: Optional[MultiImageBatch] = None,
        # CFG params
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # SDE params
        compute_log_prob: bool = True,
        # Trajectory
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
        # Other
        generator: Optional[torch.Generator] = None,
        think: bool = False,
    ) -> List[BagelSample]:
        """
        Full generation loop: context build → denoise → decode → samples.

        Processes one sample at a time (batch_size=1 per call) due to
        Bagel's variable-length KV-cache architecture.  The trainer
        handles outer batching across prompts.

        Pipeline:
            1. Build three KV-cache contexts (full, text-CFG, img-CFG)
               via ``_build_gen_context()``.
            2. Prepare packed latent generation inputs (initial noise,
               position IDs, token indexes).
            3. Run ``_denoise_loop()`` for ``num_inference_steps`` steps.
            4. Decode final latent via ``decode_latents()``.
            5. Package into ``BagelSample`` or ``BagelI2ISample``.

        Args:
            num_inference_steps: Number of denoising steps.
            height, width: Target image dimensions.
            prompt: Text prompt(s).
            condition_images: Optional condition images for I2I tasks,
                as ``List[List[Image]]`` (per-sample, per-image).
            cfg_text_scale: Text classifier-free guidance scale.
            cfg_img_scale: Image classifier-free guidance scale.
            cfg_interval: Sigma range ``(lo, hi)`` where CFG is active.
            cfg_renorm_min: Minimum clamp for CFG renormalisation.
            cfg_renorm_type: Renorm strategy (``"global"``/``"channel"``/
                ``"text_channel"``).
            compute_log_prob: Whether to compute SDE log-probabilities.
            trajectory_indices: Which denoising steps to store.
            generator: Optional torch.Generator for reproducibility.
            think: If True, prepend chain-of-thought system prompt.

        Returns:
            List of ``BagelSample`` (T2I) or ``BagelI2ISample`` (I2I).
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        device = self.device
        bagel = self.pipeline.bagel
        image_shape = (height, width)

        samples = []

        for b in range(batch_size):
            cur_prompt = prompt[b]
            cur_cond_images = (
                condition_images[b] if condition_images is not None else None
            )

            # 1. Build KV-cache contexts
            gen_ctx, cfg_text_ctx, cfg_img_ctx = self._build_gen_context(
                prompt=cur_prompt,
                condition_images=cur_cond_images,
                think=think,
            )

            # 2. Prepare latent generation inputs
            gen_input = bagel.prepare_vae_latent(
                curr_kvlens=gen_ctx["kv_lens"],
                curr_rope=gen_ctx["ropes"],
                image_sizes=[image_shape],
                new_token_ids=self.new_token_ids,
                device=device,
                generator=generator
            )

            cfg_text_gen_input = bagel.prepare_vae_latent_cfg(
                curr_kvlens=cfg_text_ctx["kv_lens"],
                curr_rope=cfg_text_ctx["ropes"],
                image_sizes=[image_shape],
                device=device,
            )
            cfg_img_gen_input = bagel.prepare_vae_latent_cfg(
                curr_kvlens=cfg_img_ctx["kv_lens"],
                curr_rope=cfg_img_ctx["ropes"],
                image_sizes=[image_shape],
                device=device,
            )

            # 3. Run denoising loop
            result = self._denoise_loop(
                generation_input=gen_input,
                cfg_text_generation_input=cfg_text_gen_input,
                cfg_img_generation_input=cfg_img_gen_input,
                past_key_values=gen_ctx["past_key_values"],
                cfg_text_past_kv=cfg_text_ctx["past_key_values"],
                cfg_img_past_kv=cfg_img_ctx["past_key_values"],
                num_inference_steps=num_inference_steps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                compute_log_prob=compute_log_prob,
                trajectory_indices=trajectory_indices,
                extra_call_back_kwargs=extra_call_back_kwargs,
                device=device,
            )

            # 4. Decode final latent
            final_latent = result["unpacked_latent"]
            image = self.decode_latents(final_latent, image_shape=image_shape)

            # 5. Build sample
            is_i2i = cur_cond_images is not None and len(cur_cond_images) > 0
            SampleCls = BagelI2ISample if is_i2i else BagelSample

            # Collect trajectory from collectors
            all_latents = result["all_latents"]        # List[Tensor] or None
            all_log_probs = result["all_log_probs"]    # List[Tensor] or None

            sample = SampleCls(
                # Trajectory — timesteps stored in [0, 1000] for scheduler
                timesteps=result["timesteps"],
                all_latents=(
                    torch.stack(all_latents, dim=0)
                    if all_latents is not None else None
                ),
                log_probs=(
                    torch.stack(all_log_probs, dim=0)
                    if all_log_probs is not None else None
                ),
                latent_index_map=result.get("latent_index_map"),
                log_prob_index_map=result.get("log_prob_index_map"),
                # Prompt
                prompt=cur_prompt,
                # Image
                height=height,
                width=width,
                image=image,
                image_shape=image_shape,
                # Condition images (for I2I)
                **(
                    {"condition_images": cur_cond_images}
                    if is_i2i and hasattr(SampleCls, "condition_images")
                    else {}
                ),
                extra_kwargs={
                    **result.get("callback_results", {}),
                    "callback_index_map": result.get("callback_index_map"),
                },
            )
            samples.append(sample)

        return samples

    # ======================== Denoising Loop ========================

    def _denoise_loop(
        self,
        generation_input: Dict[str, torch.Tensor],
        cfg_text_generation_input: Dict[str, torch.Tensor],
        cfg_img_generation_input: Dict[str, torch.Tensor],
        past_key_values : NaiveCache,
        cfg_text_past_kv: NaiveCache,
        cfg_img_past_kv: NaiveCache,
        num_inference_steps: int,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval: Tuple[float, float],
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        compute_log_prob: bool,
        trajectory_indices: TrajectoryIndicesType,
        extra_call_back_kwargs: List[str],
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        Core denoising loop using Bagel's flow matching with SDE scheduling.

        **Timestep convention**:
          1. Computes Bagel's shifted sigma schedule in [0, 1].
          2. Passes sigmas to the scheduler (stored as
             ``timesteps = sigmas * 1000``).
          3. ``forward()`` converts back to [0, 1] for the Bagel LLM.
          4. Trajectory stores timesteps in [0, 1000].

        At each step, calls ``forward()`` which handles velocity prediction
        + scheduler step, then collects latents and log-probs via
        trajectory collectors.

        Returns:
            Dict with keys:
              - ``unpacked_latent``: Final denoised latent (float32).
              - ``all_latents``: List of trajectory latents or None.
              - ``all_log_probs``: List of trajectory log-probs or None.
              - ``timesteps``: Scheduler timesteps in [0, 1000].
              - ``latent_index_map``, ``log_prob_index_map``: Index maps.
              - ``callback_results``, ``callback_index_map``: Extra outputs.
        """
        # ── 1. Build Bagel's shifted sigma schedule & configure scheduler ──
        #
        # Bagel's schedule:  σ_shifted = shift * σ / (1 + (shift - 1) * σ)
        # where σ goes linearly from 1 → 0.
        #
        # We pass these sigmas to the scheduler, which converts them to
        # timesteps in [0, 1000] and sets up SDE noise level machinery.
        linear_sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)[:-1]
        # Configure scheduler with Bagel's schedule, shift is applied inside scheduler's set_timesteps
        self.scheduler.set_timesteps(sigmas=linear_sigmas.tolist(), device=device)
        timesteps = self.scheduler.timesteps  # (T,) in [0, 1000]

        # ── 2. Initial noise ──
        x_t = generation_input["packed_init_noises"].to(device)

        # Move all packed tensors to device once
        generation_input = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in generation_input.items()
        }

        # ── 3. Trajectory & callback collectors ──
        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latent_collector.collect(x_t, step_idx=0)

        log_prob_collector = (
            create_trajectory_collector(trajectory_indices, num_inference_steps)
            if compute_log_prob else None
        )
        callback_collector = create_callback_collector(
            trajectory_indices, num_inference_steps
        )

        # ── 4. Denoising loop ──
        for i, t in enumerate(timesteps):
            t_next = (
                timesteps[i + 1]
                if i + 1 < len(timesteps)
                else torch.tensor(0.0, device=device)
            )
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            current_compute_log_prob = compute_log_prob and current_noise_level > 0
            return_kwargs = list(set(
                ['next_latents', 'log_prob', 'noise_pred'] + extra_call_back_kwargs
            ))

            # Single forward step: flow prediction + scheduler step
            output = self.forward(
                t=t.unsqueeze(0),
                latents=x_t,
                # KV caches (pre-built)
                past_key_values=past_key_values,
                cfg_text_past_kv=cfg_text_past_kv,
                cfg_img_past_kv=cfg_img_past_kv,
                # Packed generation inputs
                generation_input=generation_input,
                # CFG inputs
                cfg_text_generation_input=cfg_text_generation_input,
                cfg_img_generation_input=cfg_img_generation_input,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # Scheduler
                t_next=t_next.unsqueeze(0),
                noise_level=current_noise_level,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
            )

            # Advance latents
            x_t = output.next_latents

            # Collect trajectory
            latent_collector.collect(x_t, step_idx=i + 1)
            if current_compute_log_prob and log_prob_collector is not None:
                log_prob_collector.collect(output.log_prob, step_idx=i)

            callback_collector.collect_step(
                step_idx=i,
                output=output,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": current_noise_level},
            )

        # ── 5. Unpack final latent ──
        packed_seqlens = generation_input["packed_seqlens"]
        unpacked = x_t.split((packed_seqlens - 2).tolist())

        # ── 6. Assemble results ──
        return {
            "unpacked_latent": unpacked[0].float(),
            "all_latents": latent_collector.get_result(),
            "all_log_probs": (
                log_prob_collector.get_result() if log_prob_collector else None
            ),
            # Store timesteps in [0, 1000] — same convention as all other adapters
            "timesteps": timesteps,
            "latent_index_map": latent_collector.get_index_map(),
            "log_prob_index_map": (
                log_prob_collector.get_index_map() if log_prob_collector else None
            ),
            "callback_results": callback_collector.get_result(),
            "callback_index_map": callback_collector.get_index_map(),
        }

    # ======================== Forward (Training & Inference) ========================

    def forward(
        self,
        # ── Core (always required) ──
        t: torch.Tensor,
        latents: torch.Tensor,
        # ── Packed generation inputs ──
        generation_input: Optional[Dict[str, torch.Tensor]] = None,
        # ── CFG generation inputs ──
        cfg_text_generation_input: Optional[Dict[str, torch.Tensor]] = None,
        cfg_img_generation_input: Optional[Dict[str, torch.Tensor]] = None,
        # ── KV caches (inference: provided; training: rebuilt from prompt) ──
        past_key_values : Optional[NaiveCache] = None,
        cfg_text_past_kv : Optional[NaiveCache] = None,
        cfg_img_past_kv : Optional[NaiveCache]=None,
        # ── CFG params ──
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # ── Scheduler / SDE ──
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "noise_pred", "next_latents", "next_latents_mean",
            "std_dev_t", "dt", "log_prob",
        ],
        # ── Context rebuild (training path) ──
        prompt: Optional[Union[str, List[str]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> SDESchedulerOutput:
        """
        Single denoising step: flow velocity prediction → scheduler step.

        Supports two calling modes:

        **Inference mode** (called from ``_denoise_loop``):
            ``past_key_values`` and ``generation_input`` are pre-built.
            No context rebuild needed — just run flow forward + scheduler.

        **Training mode** (called from RL trainers):
            ``past_key_values=None``, ``prompt`` provided.  KV-cache
            contexts and packed generation inputs are rebuilt from scratch
            via ``_build_gen_context()`` and ``prepare_vae_latent()``.
            Only batch_size=1 is supported.

        **Timestep convention**:
            ``t`` and ``t_next`` are in [0, 1000].  They are converted to
            [0, 1] sigmas for ``_forward_flow()``, then passed as-is to
            ``scheduler.step()`` which expects [0, 1000].

        Args:
            t: Current timestep in [0, 1000], shape ``(1,)`` or ``(B,)``.
            latents: Current noised latent tokens.
            generation_input: Pre-built packed tensor dict (inference).
            cfg_text_generation_input: Pre-built text-CFG tensors.
            cfg_img_generation_input: Pre-built image-CFG tensors.
            past_key_values: Pre-built KV cache (inference) or None (training).
            cfg_text_past_kv: Text-CFG KV cache.
            cfg_img_past_kv: Image-CFG KV cache.
            cfg_text_scale: Text guidance scale.
            cfg_img_scale: Image guidance scale.
            cfg_interval: Sigma range where CFG is active.
            t_next: Next timestep for scheduler step.
            next_latents: Ground-truth next latents (for training loss).
            noise_level: Current SDE noise level.
            compute_log_prob: Whether to compute log-probability.
            return_kwargs: Which fields to include in output.
            prompt: Text prompt (training mode context rebuild).
            condition_images: Condition images (training mode).
            image_shape: Target ``(H, W)`` (training mode).

        Returns:
            ``SDESchedulerOutput`` with ``next_latents``, ``log_prob``,
            ``noise_pred``, etc. depending on ``return_kwargs``.
        """
        bagel = self.pipeline.bagel
        device = latents.device

        # ── 1. Rebuild KV-cache contexts if not provided (training path) ──
        rebuild_context = (
            past_key_values is None or
            generation_input is None
        )
        if rebuild_context:
            if prompt is None:
                raise ValueError(
                    "BagelAdapter.forward() requires either `past_key_values` "
                    "(inference) or `prompt` (training) to build KV caches."
                )
            if isinstance(prompt, str):
                prompt = [prompt]

            assert len(prompt) == 1, "Batch size > 1 not supported for Bagel training."
            prompt = prompt[0]
            condition_images = condition_images[0] if condition_images is not None else None
            _image_shape = image_shape or (
                kwargs.get("height", 1024), kwargs.get("width", 1024)
            )

            # Context building is always @torch.no_grad + eval mode
            with torch.no_grad():
                gen_ctx, cfg_text_ctx, cfg_img_ctx = self._build_gen_context(
                    prompt=prompt,
                    condition_images=condition_images,
                )

                # Prepare packed latent generation inputs
                generation_input = bagel.prepare_vae_latent(
                    curr_kvlens=gen_ctx["kv_lens"],
                    curr_rope=gen_ctx["ropes"],
                    image_sizes=[_image_shape],
                    new_token_ids=self.new_token_ids,
                    device=device,
                )

            past_key_values = gen_ctx["past_key_values"]
            cfg_text_past_kv = cfg_text_ctx["past_key_values"]
            cfg_img_past_kv = cfg_img_ctx["past_key_values"]

            with torch.no_grad():
                cfg_text_generation_input = bagel.prepare_vae_latent_cfg(
                    curr_kvlens=cfg_text_ctx["kv_lens"],
                    curr_rope=cfg_text_ctx["ropes"],
                    image_sizes=[_image_shape],
                    device=device,
                )
                cfg_img_generation_input = bagel.prepare_vae_latent_cfg(
                    curr_kvlens=cfg_img_ctx["kv_lens"],
                    curr_rope=cfg_img_ctx["ropes"],
                    image_sizes=[_image_shape],
                    device=device,
                )

        # Override packed tensors from rebuilt context
        packed_text_ids = generation_input["packed_text_ids"]
        packed_text_indexes = generation_input["packed_text_indexes"]
        packed_vae_position_ids = generation_input["packed_vae_position_ids"]
        packed_vae_token_indexes = generation_input["packed_vae_token_indexes"]
        packed_seqlens = generation_input["packed_seqlens"]
        packed_position_ids = generation_input["packed_position_ids"]
        packed_indexes = generation_input["packed_indexes"]
        packed_key_value_indexes = generation_input["packed_key_value_indexes"]
        key_values_lens = generation_input["key_values_lens"]

        # ── 2. Convert [0, 1000] → [0, 1] sigma for Bagel ──
        # Bagel's flow forward expects timesteps as sigmas in [0, 1].
        # The scheduler and GRPO trainer work in [0, 1000].
        sigma = t.float() / 1000.0                                 # (B,) or (1,)
        timestep_for_bagel = sigma.expand(latents.shape[0])         # (B,) in [0, 1]

        # ── 3. CFG gating based on sigma (not timestep/1000) ──
        sigma_val = sigma.flatten()[0].item()
        if sigma_val > cfg_interval[0] and sigma_val <= cfg_interval[1]:
            cfg_text_s = cfg_text_scale
            cfg_img_s = cfg_img_scale
        else:
            cfg_text_s = 1.0
            cfg_img_s = 1.0

        # Helper: safely extract CFG tensor, fallback to empty tensor
        def _cfg(d: Optional[Dict], key: str) -> Optional[torch.Tensor]:
            if d is None:
                return None
            v = d.get(key)
            if isinstance(v, torch.Tensor):
                return v.to(device)
            return None

        # ── 4. Flow velocity prediction (gradient-safe) ──
        v_t = self._forward_flow(
            x_t=latents,
            timestep=timestep_for_bagel,
            packed_vae_token_indexes=packed_vae_token_indexes.to(device),
            packed_vae_position_ids=packed_vae_position_ids.to(device),
            packed_text_ids=packed_text_ids.to(device),
            packed_text_indexes=packed_text_indexes.to(device),
            packed_position_ids=packed_position_ids.to(device),
            packed_indexes=packed_indexes.to(device),
            packed_seqlens=packed_seqlens.to(device),
            key_values_lens=key_values_lens.to(device),
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes.to(device),
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            # Text CFG
            cfg_text_scale=cfg_text_s,
            cfg_text_packed_position_ids=_cfg(cfg_text_generation_input, "cfg_packed_position_ids"),
            cfg_text_packed_query_indexes=_cfg(cfg_text_generation_input, "cfg_packed_query_indexes"),
            cfg_text_key_values_lens=_cfg(cfg_text_generation_input, "cfg_key_values_lens"),
            cfg_text_past_key_values=cfg_text_past_kv,
            cfg_text_packed_key_value_indexes=_cfg(cfg_text_generation_input, "cfg_packed_key_value_indexes"),
            # Image CFG
            cfg_img_scale=cfg_img_s,
            cfg_img_packed_position_ids=_cfg(cfg_img_generation_input, "cfg_packed_position_ids"),
            cfg_img_packed_query_indexes=_cfg(cfg_img_generation_input, "cfg_packed_query_indexes"),
            cfg_img_key_values_lens=_cfg(cfg_img_generation_input, "cfg_key_values_lens"),
            cfg_img_past_key_values=cfg_img_past_kv,
            cfg_img_packed_key_value_indexes=_cfg(cfg_img_generation_input, "cfg_packed_key_value_indexes"),
            cfg_type="parallel",
        )

        # ── 5. Scheduler step (timesteps stay in [0, 1000]) ──
        output = self.scheduler.step(
            noise_pred=v_t,
            timestep=t,
            latents=latents,
            timestep_next=t_next,
            next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            return_kwargs=return_kwargs,
            noise_level=noise_level,
        )
        return output