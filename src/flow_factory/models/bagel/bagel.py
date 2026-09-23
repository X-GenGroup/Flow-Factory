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

# src/flow_factory/models/bagel/bagel.py
"""
Bagel Model Adapter for Flow-Factory

Integrates ByteDance's Bagel (unified multimodal model) into the
Flow-Factory RL fine-tuning framework.

Architecture Mapping:
    ┌─────────────────────────────────────────────────────┐
    │ Flow-Factory Interface     │  Bagel Component        │
    ├───────────────────────────┼─────────────────────────┤
    │ self.transformer          │  Bagel (LLM + gen heads) │
    │ self.vae                  │  Custom Autoencoder      │
    │ self.tokenizer            │  Qwen2Tokenizer          │
    │ encode_prompt()           │  Build KV-cache context  │
    │ encode_image()            │  ViT + VAE transforms    │
    │ forward()                 │  _forward_flow + sched   │
    │ inference()               │  Full denoising loop     │
    │ decode_latents()          │  VAE decode              │
    └───────────────────────────┴─────────────────────────┘

Supported Tasks:
    - Text-to-Image (T2I): prompt → image
    - Image(s)-to-Image (I2I): images + prompt → image

Training-mode Caveats:
    Bagel's Qwen2Model.forward() dispatches to ``forward_train()`` or
    ``forward_inference()`` based on ``self.training``.  During RL training
    we always need the inference-path signatures (packed_query_sequence,
    KV-caches …), so we **temporarily switch the model to eval mode**
    for every LLM forward call.  Gradients still flow because we do NOT
    wrap with ``@torch.no_grad`` (autograd is orthogonal to train/eval
    mode).  The only behavioural difference is that dropout is disabled,
    which is desirable for generation modules anyway.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from ...contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaOrder,
    NegativePromptPolicy,
)
from ...hparams import Arguments
from ...samples import I2ISample, T2ISample
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    SDESchedulerOutput,
)
from ...utils.base import move_tensors_to_device
from ...utils.image import (
    ImageBatch,
    MultiImageBatch,
    is_multi_image_batch,
    standardize_image_batch,
)
from ...utils.imports import get_flash_attn_version, is_flash_attn_available
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import (
    CallbackCollector,
    TrajectoryCollector,
    TrajectoryIndicesType,
    create_callback_collector,
    create_trajectory_collector,
)
from ..abc import BaseAdapter
from ..output_state import DecodedMediaBatch, EncodedOutputState, OutputStateCodec
from ..pipeline_contracts import image_output_contract
from ..runtime import ComponentRuntime, PseudoPipelineRuntime
from ._output import (
    BagelOutputStateCodec,
    validate_bagel_encoded_output_geometry,
)

# Bagel's LLM attention (qwen2_navit) hard-requires flash-attn's varlen kernel,
# imported transitively by the `.modeling` imports below. Fail fast here with
# install guidance instead of a deep ModuleNotFoundError from the vendored
# modeling code when the adapter is loaded.
_FLASH_ATTN_MIN_VERSION = "2.5.8"
if not is_flash_attn_available(_FLASH_ATTN_MIN_VERSION):
    _installed_flash_attn = get_flash_attn_version()
    _flash_attn_detail = (
        f"found flash-attn {_installed_flash_attn}, but >= {_FLASH_ATTN_MIN_VERSION} is required"
        if _installed_flash_attn is not None
        else "flash-attn is not installed"
    )
    raise ImportError(
        f"The Bagel model adapter requires flash-attn >= {_FLASH_ATTN_MIN_VERSION} "
        f'({_flash_attn_detail}). Install it with `pip install -e ".[bagel]"`, or a '
        f"prebuilt wheel matching your torch/CUDA/Python from "
        f"https://github.com/Dao-AILab/flash-attention/releases."
    )

from .data.data_utils import add_special_tokens, pil_img2rgb
from .data.transforms import ImageTransform
from .modeling.bagel import Bagel
from .modeling.bagel.qwen2_navit import NaiveCache
from .modeling.qwen2 import Qwen2Tokenizer
from .pipeline import BagelPseudoPipeline

VLM_THINK_SYSTEM_PROMPT = """You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"""

GEN_THINK_SYSTEM_PROMPT = """You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here"""

logger = setup_logger(__name__, rank_zero_only=True)

# ============================================================================
# Sample Dataclasses
# ============================================================================


@dataclass
class BagelSample(T2ISample):
    """
    Sample class for Bagel T2I generation.

    Stores denoising trajectory plus Bagel-specific packed tensor info
    needed to reconstruct the KV-cache context during training.
    """

    _shared_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "image_shape",
        }
    )
    # Image shape for latent unpacking
    image_shape: Optional[Tuple[int, int]] = None


@dataclass
class BagelI2ISample(I2ISample):
    """Sample class for Bagel Image(s)-to-Image generation."""

    _shared_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "image_shape",
        }
    )
    # Keep condition_images as PIL on the sample (Bagel persists them via
    # ``python_format_columns`` and re-normalizes from PIL at forward time), avoiding
    # a PIL -> float tensor -> PIL round-trip and ~4x buffer memory. Sync with that ClassVar.
    condition_images_as_pil: ClassVar[bool] = True
    image_shape: Optional[Tuple[int, int]] = None


# ============================================================================
# BagelAdapter
# ============================================================================


class BagelAdapter(BaseAdapter):
    """
    Flow-Factory adapter for Bagel multimodal models.

    Key differences from diffusers-based adapters:
      1. No separate text_encoder; text encoding is internal to the Bagel model
         via its language_model.embed_tokens + KV-cache prefill.
      2. Image understanding uses ViT (SiglipVisionModel) inside the Bagel model.
      3. Denoising operates on packed latent sequences with position-aware indexing.
      4. CFG uses separate pre-computed KV caches for text-only and image-only conditions.
    """

    requires_preserved_replay_batch_composition = True

    offline_training_forward_overrides = MappingProxyType(
        {
            "cfg_text_scale": 1.0,
            "cfg_img_scale": 1.0,
            "cfg_interval": (0.0, 1.0),
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "global",
        }
    )

    # Bagel stores raw, variable-size condition images (no fixed resize) and
    # re-encodes them at rollout/training. Persist them via the HF Image feature
    # so ragged multi-reference batches serialize; they read back as PIL and are
    # re-normalized by ``_normalize_condition_images``.
    python_format_columns: ClassVar[frozenset[str]] = frozenset({"condition_images"})
    pipeline_io_contract = image_output_contract(
        negative_prompt=NegativePromptPolicy.UNSUPPORTED,
        input_image_min_count=0,
        input_image_max_count=None,
        input_order=InputMediaOrder.WITHIN_TYPE,
        geometry_source=GeometrySource.OUTPUT_MEDIA,
        batch_capability=BatchCapability.UNIFORM,
    )

    # Bagel is a mixture-of-transformer-experts model: the generation path uses
    # *_moe_gen experts while the understanding/ViT path is unused during RL
    # generation, and NaViT packing varies which sub-modules run per batch. Some
    # trainable params can therefore receive no gradient in a given step, so DDP
    # must scan for unused parameters. Ignored under DeepSpeed/FSDP.
    ddp_find_unused_parameters = True

    def __init__(self, config: Arguments, accelerator: Accelerator):
        # Load tokenizer and transforms before super().__init__
        # because load_pipeline may need them, and base __init__ calls load_pipeline
        self._model_path = config.model_args.model_name_or_path
        self._init_tokenizer_and_transforms()

        super().__init__(config, accelerator)
        self.pipeline: BagelPseudoPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

        # Bagel's batched forward is pack-composition-dependent (NaViT packing), so the
        # optimize-time sample shuffle must be off to keep the on-policy ratio == 1.
        if getattr(self.training_args, "shuffle_samples", True):
            logger.warning(
                "Bagel is not batch-invariant (NaViT packing) but training_args.shuffle_samples "
                "is True: this reorders optimize micro-batches vs rollout packs and breaks "
                "train-inference consistency (on-policy ratio != 1). Set `train.shuffle_samples: "
                "false` (see the Bagel example configs / train_inference_consistency.md)."
            )

    # ─────────────────── Tokenizer & Transforms ───────────────────

    def _init_tokenizer_and_transforms(self):
        """Initialize tokenizer, special tokens, and image transforms."""
        self._tokenizer = Qwen2Tokenizer.from_pretrained(self._model_path)
        self._tokenizer, self.new_token_ids, _ = add_special_tokens(self._tokenizer)

        # VAE transform: max_size=1024, min_size=512, patch=16
        self.vae_transform = ImageTransform(1024, 512, 16)
        # ViT transform: max_size=980, min_size=224, patch=14
        self.vit_transform = ImageTransform(980, 224, 14)

    # ======================== Pipeline & Scheduler ========================

    def load_pipeline(self) -> BagelPseudoPipeline:
        """Load the Bagel model and VAE into a pseudo-pipeline."""
        component_dtypes = self._resolve_component_load_dtype_mapping(
            component_names=["bagel", "transformer", "vae"],
            transformer_names=["transformer"],
            text_encoder_names=[],
        )
        pipeline = BagelPseudoPipeline.from_pretrained(
            self._model_path,
            low_cpu_mem_usage=False,
            component_dtypes=component_dtypes,
            pad_token_id=self._tokenizer.pad_token_id,
            **self.model_args.extra_kwargs,
        )
        # Train-inference consistency (I2I): the condition-image VAE encode is rebuilt
        # on every training forward() but only once during rollout. Stochastic sampling
        # (mean + std*randn) would then differ between the two and break the on-policy
        # ratio (== 1). Use the posterior mean so condition encoding is deterministic.
        # vae.encode is only used for condition images here (generation uses init noise;
        # vae.decode is unaffected), so this is safe.
        pipeline.vae.reg.sample = False
        return pipeline

    def build_component_runtime(self) -> ComponentRuntime:
        """Build Bagel's explicit pseudo-pipeline component runtime.

        Returns:
            Runtime exposing Bagel's model, transformer alias, and VAE.
        """
        pipeline = self.load_pipeline()
        return PseudoPipelineRuntime(
            pipeline,
            {
                "bagel": pipeline.bagel,
                "vae": pipeline.vae,
            },
            aliases={"transformer": pipeline.transformer},
            alias_routes={"transformer": ("bagel", ("language_model",))},
        )

    def build_output_state_codec(self) -> OutputStateCodec:
        """Declare Bagel's on-the-fly stochastic target-image codec.

        Returns:
            Codec requiring the logical Bagel and VAE components at encode time.
        """
        return BagelOutputStateCodec(self)

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Verify target geometry against Bagel's output-media transform.

        Args:
            media_batch: Decoded target images validated by the shared boundary.
            condition: Input condition paired with the targets.
            encoded: Codec result after generic validation.
        """
        del condition
        validate_bagel_encoded_output_geometry(self, media_batch, encoded)

    def load_scheduler(self) -> FlowMatchEulerDiscreteSDEScheduler:
        """
        Create a FlowMatchEulerDiscreteSDEScheduler for Bagel.

        Bagel uses flow matching with a shifted timestep schedule:
            t_shifted = shift * t / (1 + (shift - 1) * t)
        The scheduler operates in [0, 1000] units; the adapter handles
        conversion to/from Bagel's native [0, 1] sigma space.
        """
        scheduler_kwargs = {"num_train_timesteps": 1000, "shift": 3.0}
        if hasattr(self.config, "scheduler_args") and self.config.scheduler_args:
            scheduler_kwargs.update(self.config.scheduler_args.to_dict())

        scheduler = FlowMatchEulerDiscreteSDEScheduler(**scheduler_kwargs)

        return scheduler

    # ======================== Module Management ========================

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Bagel's Qwen2 decoder layers."""
        return [
            "self_attn.q_proj_moe_gen",
            "self_attn.k_proj_moe_gen",
            "self_attn.v_proj_moe_gen",
            "self_attn.o_proj_moe_gen",
            "mlp_moe_gen.gate_proj",
            "mlp_moe_gen.up_proj",
            "mlp_moe_gen.down_proj",
        ]

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def text_encoder_names(self) -> List[str]:
        """Bagel has no separate text encoder; encoding is inside the transformer."""
        return []

    @property
    def text_encoders(self) -> List[nn.Module]:
        return []

    @property
    def text_encoder(self) -> Optional[nn.Module]:
        return None

    @property
    def preprocessing_modules(self) -> List[str]:
        """Modules needed for preprocessing (tokenization uses CPU, VAE for decode)."""
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
        """Set all target components to evaluation mode."""
        super().eval()  # Set base adapter mode
        self.transformer.eval()
        self.pipeline.bagel.eval()
        self.pipeline.vae.eval()

    def rollout(self, *args, **kwargs):
        """Set model to rollout mode."""
        self.eval()  # Rollout mode uses eval behaviour for all components
        # If the scheduler has a rollout method, call it (e.g. for noise sampling adjustments)
        if hasattr(self.scheduler, "rollout"):
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

        This is required because Bagel's Qwen2Model.forward() dispatches to
        ``forward_train()`` vs ``forward_inference()`` based on ``self.training``.
        We always need the inference dispatch (packed_query_sequence / KV-cache
        API), even during RL training.

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

        Unlike diffusers adapters, Bagel's prompt encoding is deferred to
        ``inference()`` / ``forward()`` where it becomes part of KV-cache
        context building. Here we just return the raw prompt strings.

        Returns:
            Dict with ``prompt`` key mapping to the list of prompts.
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

        Converts PIL images to RGB and stores them for later context building.
        The actual ViT/VAE encoding happens in ``inference()`` / ``forward()``.

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
            standardize_image_batch(img_list, output_type="pt") if img_list else []
            for img_list in images
        ]
        return {"condition_images": processed}

    def encode_video(self, videos: Any) -> None:
        """No-op: Bagel consumes no video modality, so encoding returns None.

        Returning ``None`` signals ``preprocess_func`` to skip video
        integration (see constraint #12).
        """
        return None

    # ======================== Batch-mode dispatch ========================

    @staticmethod
    def _is_i2i(condition_images: Optional[Any]) -> bool:
        """Return whether the batch carries condition image(s) (Image-to-Image).

        This is the dispatch flag for batch handling: both T2I (no condition
        images) and I2I run the same NaViT-packed batched path (I2I adds the
        condition images in per-image subset-rounds). The decision depends only on
        the presence of condition images, which is fixed by the run's task/config
        and is therefore identical across distributed ranks (avoiding collective/RNG
        desync — see ``adapter_conventions.md``).

        Args:
            condition_images: ``None`` (T2I) or a per-sample batch of condition
                images (``MultiImageBatch`` / ``List[List[...]]``). Empty
                per-sample lists count as no condition.

        Returns:
            True if at least one sample has a non-empty condition image list.
        """
        if condition_images is None:
            return False
        if isinstance(condition_images, (list, tuple)):
            return any(
                img is not None and (not isinstance(img, (list, tuple)) or len(img) > 0)
                for img in condition_images
            )
        # Any non-list, non-None value (e.g. a single image / tensor) is a condition.
        return True

    @staticmethod
    def _normalize_condition_images(
        condition_images: Optional[MultiImageBatch],
    ) -> Optional[List[List[Image.Image]]]:
        """Normalize condition images to a per-sample ``List[List[PIL]]`` (or None).

        Reuses ``is_multi_image_batch`` + ``standardize_image_batch`` (mirroring the
        multi-condition pattern in ``flux2.py``): a non-nested input is treated as a
        single sample's images; each sample's images are standardized to PIL. Empty
        per-sample entries become ``[]``. Accepts PIL (rollout) or tensor (training,
        after ``BaseSample`` canonicalization) inputs uniformly.
        """
        if condition_images is None:
            return None
        per_sample = (
            condition_images if is_multi_image_batch(condition_images) else [condition_images]
        )
        return [
            (
                standardize_image_batch(imgs, output_type="pil")
                if imgs is not None and len(imgs) > 0
                else []
            )
            for imgs in per_sample
        ]

    def _resolve_condition_images_for_packing(
        self, condition_images: Optional[MultiImageBatch]
    ) -> Optional[List[List[Image.Image]]]:
        """Resolve condition images for the packed dispatch shared by ``inference()``
        and ``forward()``: returns per-sample ``List[List[PIL]]`` for I2I (after the
        FSDP variable-count guard), or ``None`` for T2I (no / all-empty conditions).
        Both T2I and I2I then take the same NaViT-packed path.
        """
        cond = self._normalize_condition_images(condition_images)
        if cond is None or not self._is_i2i(cond):
            return None
        self._assert_variable_count_supported(cond)
        return cond

    def _assert_variable_count_supported(self, condition_images: List[List[Image.Image]]) -> None:
        """Fail fast on condition-image counts that desync collectives under a
        parameter-sharded ``language_model`` (FSDP FULL/HYBRID or FSDP2).

        The prefill (``_build_gen_context``) issues a *data-dependent* number of
        ``language_model.forward_inference`` calls -- ``2*num_rounds + 2`` where
        ``num_rounds = max(per-sample count)`` (2 per image round for the VAE and ViT
        cache updates, plus the final gen / cfg_img text passes). ``language_model`` is
        the only sharded module (frozen ViT/VAE are not). When params are sharded each
        call must ``AllGather`` its shard, so a per-forward call count that differs
        ACROSS ranks mismatches the collective and deadlocks. Two ways this happens:
          (a) variable counts WITHIN a rank's batch, and
          (b) locally-uniform counts that differ ACROSS ranks (different ``num_rounds``).
        DDP / DeepSpeed ZeRO-1/2 replicate params (forward is collective-free; gradients
        sync a fixed number of times at backward), so any count is safe -- early-return.

        The cross-rank check (b) is itself a collective; all ranks reach it because I2I
        vs T2I is fixed by the run's task/config (see ``_is_i2i``) and every sample in an
        I2I run carries condition images. The robust fix to *support* variable counts
        under sharding is to gather ``language_model`` once (``summon_full_params`` /
        ``reshard_after_forward=False``); until then we fail fast (a clear error beats a
        silent hang).
        """
        if not self._is_param_sharded():
            return
        counts = {len(imgs) for imgs in condition_images}
        if len(counts) > 1:
            raise RuntimeError(
                "Bagel batched I2I with a variable per-sample condition-image count "
                f"(counts={sorted(counts)}) is unsupported under a parameter-sharded "
                "language_model (FSDP FULL/HYBRID or FSDP2): the prefill "
                "makes a data-dependent number of all-gathers (2*max_count+2) that "
                "mismatches across ranks and deadlocks. Use DDP or ZeRO-1/2 for "
                "variable-count I2I, pad to a uniform count, or gather language_model "
                "once (summon_full_params / reshard_after_forward=False)."
            )
        if self.accelerator.num_processes > 1:
            local_rounds = max(counts) if counts else 0
            global_max = torch.tensor([local_rounds], device=self.accelerator.device)
            global_min = global_max.clone()
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(global_min, op=dist.ReduceOp.MIN)
            if int(global_max.item()) != int(global_min.item()):
                raise RuntimeError(
                    "Bagel batched I2I has a per-sample condition-image count that is "
                    "uniform within each rank but differs ACROSS ranks (num_rounds in "
                    f"[{int(global_min.item())}, {int(global_max.item())}]) under a "
                    "parameter-sharded language_model: the prefill issues a different "
                    "number of all-gathers per rank and deadlocks. Use DDP or ZeRO-1/2, "
                    "or pad all ranks to a uniform reference-image count."
                )

    # ======================== Decoding ========================

    def reference_guidance_kwargs(self, guidance_scale: float) -> Dict[str, object]:
        """Map canonical reference guidance to Bagel's text-CFG forward argument."""
        return {"cfg_text_scale": guidance_scale}

    def decode_latents(
        self,
        latents: torch.Tensor,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Decode packed latent tokens back into PIL images.

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
            decoded = vae.decode(lat.to(vae.dtype if hasattr(vae, "dtype") else torch.bfloat16))
            decoded = (decoded * 0.5 + 0.5).clamp(0, 1)[0].float()
            images.append(decoded)

        if single:
            return images[0]
        return images

    # ======================== Context Building ========================

    def _build_gen_context(
        self,
        prompt: Union[str, List[str]],
        condition_images: Optional[List[List[Image.Image]]] = None,
        think: bool = False,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Build KV-cache contexts for generation over B samples (NaViT packing).

        A single ``str`` prompt is treated as ``B == 1``; a ``List[str]`` of length
        B is packed into one block-diagonal context by threading per-sample
        ``kv_lens`` / ``ropes`` lists through the model's ``prepare_*`` /
        ``forward_cache_update_*`` methods (which already iterate them).

        ``condition_images`` (I2I) is per-sample: ``List[List[PIL]]`` of length B,
        each a sample's reference images (already normalized to PIL). Counts may
        differ across samples (and sizes too — the model pads VAE and uses varlen
        ViT). Images are appended in per-image rounds: round ``r`` adds the r-th
        image of every sample that still has one (``num_rounds = max count``). A
        sample without an r-th image is passed as ``None`` for that round, which
        keeps its cached KV but adds no query tokens (a zero-length query segment).

        Constructs three contexts:
          - gen_context: full context (images + text)
          - cfg_text_context: context without text (for text-CFG; images-only)
          - cfg_img_context: context without images (for image-CFG; text-only)

        The model is temporarily switched to eval mode so that
        ``Qwen2Model.forward()`` dispatches to ``forward_inference()``.
        """
        prompts = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompts)
        if condition_images is not None:
            if len(condition_images) != batch_size:
                raise ValueError(
                    f"condition_images length ({len(condition_images)}) must match the number "
                    f"of prompts ({batch_size})."
                )

        bagel = self.pipeline.bagel
        num_layers = bagel.config.llm_config.num_hidden_layers

        def _init_ctx():
            return {
                "kv_lens": [0] * batch_size,
                "ropes": [0] * batch_size,
                "past_key_values": NaiveCache(num_layers),
            }

        gen_context = _init_ctx()
        cfg_img_context = _init_ctx()
        # cfg_text_context is captured below as a snapshot of gen_context after the
        # condition images are added, so it needs no separate init here.

        # ── Must use eval mode for Qwen2 dispatch (forward_inference) ──
        with self._eval_mode(bagel):
            # --- Optional thinking prompt ---
            if think:
                system_prompts = [GEN_THINK_SYSTEM_PROMPT] * batch_size
                gen_context = self._update_context_text(system_prompts, gen_context)
                cfg_img_context = self._update_context_text(system_prompts, cfg_img_context)

            # --- Interleaved condition images (I2I): images first, then text ---
            # One per-image round per reference index. ``num_rounds`` is the max
            # per-sample count; round ``r`` adds the r-th image of every sample that
            # has one (others pass ``None`` -> their cache is kept, no query tokens).
            # The max-count sample is active in every round, so each round has >=1
            # active image (no empty VAE/ViT encode).
            if condition_images is not None:
                num_rounds = max(len(imgs) for imgs in condition_images)
                for r in range(num_rounds):
                    round_tensors: List[Optional[torch.Tensor]] = []
                    for imgs in condition_images:
                        if r < len(imgs):
                            round_tensors.append(
                                self.vae_transform.resize_transform(pil_img2rgb(imgs[r]))
                            )
                        else:
                            round_tensors.append(None)
                    gen_context = self._update_context_image(round_tensors, gen_context)

            # cfg_text drops the text conditioning: snapshot state after images, before
            # the text append below. A shallow snapshot (shared KV-tensor refs) suffices
            # because cache updates reassign ``key_cache[layer]`` to a freshly-allocated
            # tensor and never mutate cached tensors in place (see ``_snapshot_context``).
            cfg_text_context = self._snapshot_context(gen_context)

            # Text always comes last (before generation).
            gen_context = self._update_context_text(prompts, gen_context)
            cfg_img_context = self._update_context_text(prompts, cfg_img_context)

        return gen_context, cfg_text_context, cfg_img_context

    @staticmethod
    def _snapshot_context(gen_context: Dict) -> Dict:
        """Cheap, behavior-preserving snapshot of a KV-cache context.

        Used for ``cfg_text_context``, which only *reads* its cache. The qwen2_navit
        cache update reassigns ``past_key_values.key_cache[layer]`` to a newly
        allocated (cat'd) tensor and never mutates cached tensors in place, so copying
        only the per-layer dicts (with shared tensor references) isolates this snapshot
        from later text appends to ``gen_context`` -- without the full ``deepcopy`` of
        every KV tensor (material for I2I with long condition KV, rebuilt per training
        forward).
        """
        src = gen_context["past_key_values"]
        snap = NaiveCache(len(src.key_cache))
        snap.key_cache = dict(src.key_cache)
        snap.value_cache = dict(src.value_cache)
        return {
            "kv_lens": list(gen_context["kv_lens"]),
            "ropes": list(gen_context["ropes"]),
            "past_key_values": snap,
        }

    # ─── _update_context_text ───
    @torch.no_grad()
    def _update_context_text(self, text: Union[str, List[str]], gen_context: Dict) -> Dict:
        """Append text tokens to the KV-cache context for one or B samples.

        A single ``str`` is treated as ``B == 1``; a ``List[str]`` of length B is
        packed in one update (``prepare_prompts`` / ``forward_cache_update_text``
        already iterate the per-sample ``kv_lens`` / ``ropes`` lists).

        IMPORTANT: Caller must ensure the model is in eval mode
        (via ``self._eval_mode``) for correct Qwen2 dispatch.
        """
        prompts = [text] if isinstance(text, str) else text
        bagel = self.pipeline.bagel
        device = self.device

        generation_input, kv_lens, ropes = bagel.prepare_prompts(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            prompts=prompts,
            tokenizer=self._tokenizer,
            new_token_ids=self.new_token_ids,
        )
        generation_input = move_tensors_to_device(generation_input, device, max_depth=1)
        past_key_values = bagel.forward_cache_update_text(
            gen_context["past_key_values"],
            language_model_forward=self.transformer,
            **generation_input,
        )
        return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past_key_values}

    # ─── _update_context_image ───
    @torch.no_grad()
    def _update_context_image(
        self,
        image_tensors: List[Optional[torch.Tensor]],
        gen_context: Dict,
        vae: bool = True,
        vit: bool = True,
    ) -> Dict:
        """Add one image per sample (a single packed round) to the KV-cache context.

        ``image_tensors`` is one pre-transformed image tensor per sample (length B),
        aligned with the context's per-sample ``kv_lens`` / ``ropes``. The model's
        ``prepare_vae_images`` pads varying sizes for a single batched ``vae.encode``
        and ``prepare_vit_images`` uses varlen concat, so sizes may differ across
        samples. An entry may be ``None`` for a sample that has no image this round
        (variable condition-image count): its cached KV is kept and it contributes a
        zero-length query segment. At least one entry per call must be non-``None``.
        """
        if not any(img is not None for img in image_tensors):
            raise ValueError(
                "Bagel _update_context_image received an all-None image round (no active "
                "sample). Each packed round must have >=1 non-None image; an all-None round "
                "indicates a packing bug (num_rounds must equal the max per-sample "
                "condition-image count) and would crash opaquely in the vendored encoder."
            )
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
                images=image_tensors,
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            gen_input = move_tensors_to_device(gen_input, device, max_depth=1)
            past_key_values = bagel.forward_cache_update_vae(
                vae_model,
                past_key_values,
                language_model_forward=self.transformer,
                **gen_input,
            )

        if vit:
            gen_input, kv_lens, ropes = bagel.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=image_tensors,
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            gen_input = move_tensors_to_device(gen_input, device, max_depth=1)
            past_key_values = bagel.forward_cache_update_vit(
                past_key_values,
                language_model_forward=self.transformer,
                **gen_input,
            )

        return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past_key_values}

    def _prepare_gen_inputs(
        self,
        gen_context: Dict,
        cfg_text_context: Dict,
        cfg_img_context: Dict,
        image_sizes: List[Tuple[int, int]],
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Tuple[Dict, Dict, Dict]:
        """Build packed VAE-latent generation inputs (+ CFG inputs) for B samples.

        ``image_sizes`` carries one ``(H, W)`` per sample. The model's
        ``prepare_vae_latent`` / ``prepare_vae_latent_cfg`` already iterate the
        per-sample ``kv_lens`` / ``ropes`` lists, so this works for B >= 1;
        ``generator`` seeds the per-sample init noise. Returns
        ``(generation_input, cfg_text_generation_input, cfg_img_generation_input)``.
        """
        bagel = self.pipeline.bagel
        generation_input = bagel.prepare_vae_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            image_sizes=image_sizes,
            new_token_ids=self.new_token_ids,
            device=device,
            generator=generator,
        )
        cfg_text_generation_input = bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=cfg_text_context["ropes"],
            image_sizes=image_sizes,
            device=device,
        )
        cfg_img_generation_input = bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_context["kv_lens"],
            curr_rope=cfg_img_context["ropes"],
            image_sizes=image_sizes,
            device=device,
        )
        return generation_input, cfg_text_generation_input, cfg_img_generation_input

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
        # ``x_t`` is the packed VAE-token tensor (sum_vae_tokens, patch_dim). A stray
        # leading batch dim of 1 (callers passing (1, tokens, dim)) is squeezed off.
        # ``timestep`` is one sigma per VAE token (expanded per sample upstream), so it
        # may carry different values across packed samples — no single-timestep assert.
        if x_t.ndim == 3 and x_t.shape[0] == 1:
            x_t = x_t.squeeze(0)
        if x_t.ndim != 2:
            raise ValueError(
                f"BagelAdapter._forward_flow expects packed `x_t` of rank 2 "
                f"(sum_vae_tokens, patch_dim); got shape {tuple(x_t.shape)}."
            )
        packed_pos_embed = self.pipeline.bagel.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.pipeline.bagel.time_embedder(timestep)
        x_t = self.pipeline.bagel.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        packed_sequence = x_t.new_zeros(
            (sum(packed_seqlens), self.pipeline.bagel.hidden_size), dtype=torch.float32
        )
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.pipeline.bagel.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
            }
        output = self.transformer(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            packed_text_ids=packed_text_ids,
            packed_text_indexes=packed_text_indexes,
            **extra_inputs,
        )
        v_t = self.pipeline.bagel.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]
        if cfg_text_scale > 1.0:
            cfg_text_output = self.transformer(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
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
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
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

                if cfg_renorm_type == "global":
                    # Per-sample global renorm over each packed sample's VAE tokens, so
                    # B>1 NaViT packing reproduces the B=1 result (a single global norm
                    # per sample). ``packed_seqlens - 2`` is each sample's VAE-token count
                    # (the 2 extra tokens are the start/end-of-image markers).
                    vae_token_counts = (packed_seqlens - 2).tolist()
                    v_t = torch.cat(
                        [
                            seg_
                            * (torch.norm(seg) / (torch.norm(seg_) + 1e-8)).clamp(
                                min=cfg_renorm_min, max=1.0
                            )
                            for seg, seg_ in zip(
                                v_t.split(vae_token_counts), v_t_.split(vae_token_counts)
                            )
                        ],
                        dim=0,
                    )
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                    scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    v_t = v_t_ * scale
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not supported")
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
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        think: bool = False,
    ) -> List[BagelSample]:
        """
        Full generation loop: build context -> denoise -> decode -> return samples.

        Dispatches by task:
          - **T2I** (no condition images): all B prompts are packed into one
            block-diagonal denoising loop (NaViT) via a single ``_inference`` call.
          - **I2I**: all B samples are packed together in one ``_inference``. The
            per-sample reference-image count may vary (subset-round packing) and
            sizes may vary (the model pads VAE / varlen ViT).

        Returns:
            List of B ``BagelSample`` / ``BagelI2ISample`` (one per prompt).
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        common_kwargs = dict(
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            compute_log_prob=compute_log_prob,
            extra_call_back_kwargs=extra_call_back_kwargs,
            trajectory_indices=trajectory_indices,
            think=think,
        )

        # Per-sample List[List[PIL]] for I2I (after the FSDP guard), or None for T2I.
        # Both paths run the same NaViT-packed batched generation over all B prompts.
        cond = self._resolve_condition_images_for_packing(condition_images)
        return self._inference(
            prompts=prompt, condition_images=cond, generator=generator, **common_kwargs
        )

    def _inference(
        self,
        prompts: List[str],
        condition_images: Optional[List[List[Image.Image]]],
        num_inference_steps: int,
        height: int,
        width: int,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval: Tuple[float, float],
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        compute_log_prob: bool,
        extra_call_back_kwargs: List[str],
        trajectory_indices: TrajectoryIndicesType,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        think: bool,
    ) -> List[BagelSample]:
        """Core generation: build context -> denoise once -> split per sample.

        Handles T2I (``condition_images=None``) and batched I2I, where
        ``condition_images`` is per-sample ``List[List[PIL]]`` (length B). Per-sample
        counts may vary (subset-round packing); B==1 is just a length-1 list.
        """
        device = self.device
        image_shape = (height, width)

        gen_ctx, cfg_text_ctx, cfg_img_ctx = self._build_gen_context(
            prompts, condition_images=condition_images, think=think
        )
        generation_input, cfg_text_gen_input, cfg_img_gen_input = self._prepare_gen_inputs(
            gen_ctx,
            cfg_text_ctx,
            cfg_img_ctx,
            image_sizes=[image_shape] * len(prompts),
            device=device,
            generator=generator,
        )
        result = self._denoise_loop(
            generation_input=generation_input,
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
        # ``condition_images`` is already per-sample (or None for T2I).
        return self._assemble_samples(
            result,
            prompts=prompts,
            condition_images_list=condition_images,
            height=height,
            width=width,
        )

    def _assemble_samples(
        self,
        result: Dict[str, Any],
        prompts: List[str],
        condition_images_list: Optional[List[Optional[ImageBatch]]],
        height: int,
        width: int,
    ) -> List[BagelSample]:
        """Split a batched denoise result into per-sample ``BagelSample`` objects.

        The denoise loop produces batch-first trajectories (``(B, ...)`` per step);
        here we slice sample ``b`` out of each, decode its final latent, and build
        the appropriate sample class (T2I vs I2I).
        """
        image_shape = (height, width)
        final_latents = result["final_latents"]  # (B, n, d)
        batch_size = final_latents.shape[0]

        all_latents = result["all_latents"]  # List[(B, n, d)] or None
        all_log_probs = result["all_log_probs"]  # List[(B,)] or None
        # Stack along a leading step dim, then slice per sample.
        latents_stacked = torch.stack(all_latents, dim=0) if all_latents is not None else None
        log_probs_stacked = torch.stack(all_log_probs, dim=0) if all_log_probs is not None else None
        callback_results = result.get("callback_results") or {}

        images = self.decode_latents(final_latents, image_shape=image_shape)
        if not isinstance(images, list):
            raise TypeError(
                "expected Bagel batched decode to return a list, "
                f"received {type(images).__name__} for final_latents shape="
                f"{tuple(final_latents.shape)}"
            )
        if len(images) != batch_size:
            raise ValueError(
                "expected one Bagel decoded image per sample, "
                f"received {len(images)} images for batch_size={batch_size}"
            )

        samples: List[BagelSample] = []
        for b in range(batch_size):
            cur_cond_images = (
                condition_images_list[b] if condition_images_list is not None else None
            )
            is_i2i = cur_cond_images is not None and len(cur_cond_images) > 0
            SampleCls = BagelI2ISample if is_i2i else BagelSample

            # Per-sample callbacks: slice tensor callbacks along batch dim; keep
            # non-tensor values (e.g. lists) as-is.
            cb_b = {
                k: (v[b] if isinstance(v, torch.Tensor) else v) for k, v in callback_results.items()
            }

            sample = SampleCls(
                # Trajectory — timesteps stored in [0, 1000] for scheduler
                timesteps=result["timesteps"],
                all_latents=(latents_stacked[:, b] if latents_stacked is not None else None),
                log_probs=(log_probs_stacked[:, b] if log_probs_stacked is not None else None),
                latent_index_map=result.get("latent_index_map"),
                log_prob_index_map=result.get("log_prob_index_map"),
                # Prompt
                prompt=prompts[b],
                # Image
                height=height,
                width=width,
                image=images[b],
                image_shape=image_shape,
                # Condition images (for I2I)
                **(
                    {"condition_images": cur_cond_images}
                    if is_i2i and hasattr(SampleCls, "condition_images")
                    else {}
                ),
                extra_kwargs={
                    **cb_b,
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
        past_key_values: NaiveCache,
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
        Core denoising loop using Bagel's flow matching, batched over B samples.

        ``generation_input`` carries B packed samples (``packed_seqlens`` has B
        entries). The latents are kept as ``(B, num_tokens, dim)`` and each step
        runs one packed ``_forward_packed`` call; trajectories are collected
        batch-first and split per sample by ``_assemble_samples``.

        **Timestep convention**: Bagel natively works with sigmas in [0, 1],
        but the scheduler operates in [0, 1000].  This method:
          1. Computes Bagel's shifted sigma schedule in [0, 1]
          2. Passes sigmas to the scheduler (which stores them as
             ``timesteps = sigmas * 1000``)
          3. Uses ``scheduler.timesteps`` (in [0, 1000]) for the sample's
             timestep storage and for ``scheduler.step()``
          4. ``_forward_packed`` converts back to [0, 1] for the Bagel LLM

        Returns:
            Dict with keys: ``final_latents`` ``(B, n, d)``, ``all_latents``
            (list of ``(B, n, d)``), ``all_log_probs`` (list of ``(B,)``),
            ``timesteps``, ``latent_index_map``, ``log_prob_index_map``,
            ``callback_results`` (batch-first), ``callback_index_map``.
        """
        # ── 1. Build Bagel's shifted sigma schedule & configure scheduler ──
        #
        # Bagel's schedule:  sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
        # where sigma goes linearly from 1 -> 0.
        #
        # We pass these sigmas to the scheduler, which converts them to
        # timesteps in [0, 1000] and sets up SDE noise level machinery.
        linear_sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)[:-1]
        # Configure scheduler with Bagel's schedule, shift is applied inside scheduler's set_timesteps
        self.scheduler.set_timesteps(sigmas=linear_sigmas.tolist(), device=device)
        timesteps = self.scheduler.timesteps  # (T,) in [0, 1000]

        # Move all packed tensors to device once.
        generation_input = move_tensors_to_device(generation_input, device, max_depth=1)

        # ── 2. Initial noise, reshaped to (B, num_tokens, dim) ──
        # ``packed_init_noises`` is (sum_vae_tokens, dim); with uniform target
        # resolution each sample contributes ``n = packed_seqlens - 2`` tokens.
        packed_seqlens = generation_input["packed_seqlens"]
        batch_size = len(packed_seqlens)
        vae_token_counts = packed_seqlens - 2
        num_tokens = int(vae_token_counts[0].item())
        if not bool(torch.all(vae_token_counts == num_tokens)):
            raise ValueError(
                "Bagel batched denoising requires a uniform target resolution across the "
                f"batch; got per-sample VAE token counts {vae_token_counts.tolist()}."
            )
        init_noises = generation_input["packed_init_noises"]  # already on device (moved above)
        patch_dim = init_noises.shape[-1]
        # Cast the init noise to the trajectory storage dtype up front (every later
        # step does this via ``cast_latents``). Otherwise step 0 runs at the raw
        # float32 noise dtype while its ``next_latents`` is stored/replayed at
        # ``latent_storage_dtype`` -> the on-policy log_prob (and PPO ratio) would
        # diverge between rollout and training at step 0.
        x_t = self.cast_latents(init_noises.reshape(batch_size, num_tokens, patch_dim))

        # ── 3. Trajectory & callback collectors ──
        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latent_collector.collect(x_t, step_idx=0)

        log_prob_collector = (
            create_trajectory_collector(trajectory_indices, num_inference_steps)
            if compute_log_prob
            else None
        )
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        # ── 4. Denoising loop (one packed forward per step) ──
        for i, t in enumerate(timesteps):
            t_next = (
                timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0, device=device)
            )
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            current_compute_log_prob = compute_log_prob and current_noise_level > 0
            return_kwargs = list(
                set(["next_latents", "log_prob", "velocity"] + extra_call_back_kwargs)
            )

            # Single forward step: flow prediction + scheduler step. Context is
            # pre-built (packed for B samples), so call _forward_packed directly.
            output = self._forward_packed(
                t=t,
                latents=x_t,
                generation_input=generation_input,
                cfg_text_generation_input=cfg_text_generation_input,
                cfg_img_generation_input=cfg_img_generation_input,
                past_key_values=past_key_values,
                cfg_text_past_kv=cfg_text_past_kv,
                cfg_img_past_kv=cfg_img_past_kv,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                t_next=t_next,
                next_latents=None,
                noise_level=current_noise_level,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
            )

            # Advance latents
            x_t = self.cast_latents(output.next_latents)

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

        # ── 5. Assemble results (per-sample split happens in _assemble_samples) ──
        return {
            "final_latents": x_t,  # (B, n, d)
            "all_latents": latent_collector.get_result(),
            "all_log_probs": (log_prob_collector.get_result() if log_prob_collector else None),
            # Store timesteps in [0, 1000] — same convention as all other adapters
            "timesteps": timesteps,
            "latent_index_map": latent_collector.get_index_map(),
            "log_prob_index_map": (
                log_prob_collector.get_index_map() if log_prob_collector else None
            ),
            "callback_results": callback_collector.get_result(),
            "callback_index_map": callback_collector.get_index_map(),
        }

    def _normalize_batch(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        t_next: Optional[torch.Tensor],
        next_latents: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Canonical batch layout for packed Bagel denoising (``batch_size >= 1``).

        - **Latents** / **next_latents**: promoted to ``(B, num_tokens, dim)`` (a
          bare ``(num_tokens, dim)`` is treated as ``B == 1``). ``num_tokens`` is
          uniform by construction (a single stacked tensor).
        - **Timesteps** ``t`` / ``t_next``: float32 ``(B,)`` on the latents' device.
          A scalar (or single-element) value is broadcast to all B samples; any
          other count must match ``B`` exactly.

        Raises:
            TypeError: if ``t`` / ``t_next`` are not tensors.
            ValueError: on unsupported tensor ranks or a timestep count that is
                neither 1 nor ``B``.
        """

        def _to_bnd(x: torch.Tensor, name: str) -> torch.Tensor:
            if x.dim() == 2:
                return x.unsqueeze(0)
            if x.dim() == 3:
                return x
            raise ValueError(
                f"BagelAdapter.forward expects {name} of rank 2 (packed tokens, dim) or 3 "
                f"(batch, tokens, dim); got shape {tuple(x.shape)}."
            )

        latents = _to_bnd(latents, "latents")
        batch_size = latents.shape[0]
        device = latents.device

        def _to_batch_timestep(x: torch.Tensor, name: str) -> torch.Tensor:
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"`{name}` must be a torch.Tensor, got {type(x)!r}.")
            xf = x.float().reshape(-1).to(device=device)
            if xf.numel() == 1:
                return xf.expand(batch_size)
            if xf.numel() != batch_size:
                raise ValueError(
                    f"BagelAdapter.forward `{name}` has {xf.numel()} elements; expected 1 "
                    f"(broadcast) or batch_size={batch_size}."
                )
            return xf

        t = _to_batch_timestep(t, "t")
        if t_next is not None:
            t_next = _to_batch_timestep(t_next, "t_next")
        if next_latents is not None:
            next_latents = _to_bnd(next_latents, "next_latents")

        return latents, t, t_next, next_latents

    # ======================== Forward (Training & Inference) ========================

    def _forward_packed(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        generation_input: Dict[str, torch.Tensor],
        cfg_text_generation_input: Optional[Dict[str, torch.Tensor]],
        cfg_img_generation_input: Optional[Dict[str, torch.Tensor]],
        past_key_values: NaiveCache,
        cfg_text_past_kv: Optional[NaiveCache],
        cfg_img_past_kv: Optional[NaiveCache],
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval: Tuple[float, float],
        cfg_renorm_min: float,
        cfg_renorm_type: str,
        t_next: Optional[torch.Tensor],
        next_latents: Optional[torch.Tensor],
        noise_level: Optional[float],
        compute_log_prob: bool,
        return_kwargs: List[str],
    ) -> SDESchedulerOutput:
        """
        Pure single-step compute over a pre-built packed context (``batch_size >= 1``).

        Flattens ``latents (B, n, d)`` to the packed ``(B*n, d)`` order expected by
        ``packed_vae_token_indexes``, runs the (grad-safe) flow prediction, reshapes
        the velocity back to ``(B, n, d)``, and takes one ``scheduler.step`` so the
        scheduler returns per-sample ``(B,)`` ``log_prob``.

        **Timestep convention**: ``t`` / ``t_next`` are in [0, 1000]; a per-token
        sigma in [0, 1] is built for the Bagel LLM via ``repeat_interleave`` (so
        samples may carry different ``t`` values, as decoupled trainers do), while
        the scheduler receives the original ``t`` / ``t_next``. When CFG is active,
        all samples must fall on the same side of ``cfg_interval`` (batched CFG
        gating is shared across the pack); this is enforced below.
        """
        device = latents.device
        latents, t, t_next, next_latents = self._normalize_batch(latents, t, t_next, next_latents)
        batch_size, num_tokens, patch_dim = latents.shape
        # (B, n, d) -> packed (B*n, d): samples are sequential and each sample's VAE
        # tokens are contiguous, matching ``packed_vae_token_indexes`` ordering.
        packed_latents = latents.reshape(batch_size * num_tokens, patch_dim)

        packed_text_ids = generation_input["packed_text_ids"]
        packed_text_indexes = generation_input["packed_text_indexes"]
        packed_vae_position_ids = generation_input["packed_vae_position_ids"]
        packed_vae_token_indexes = generation_input["packed_vae_token_indexes"]
        packed_seqlens = generation_input["packed_seqlens"]
        packed_position_ids = generation_input["packed_position_ids"]
        packed_indexes = generation_input["packed_indexes"]
        packed_key_value_indexes = generation_input["packed_key_value_indexes"]
        key_values_lens = generation_input["key_values_lens"]

        # ── Per-token sigma in [0, 1] for the Bagel LLM ──
        # Each sample contributes ``packed_seqlens - 2`` VAE tokens; expand its
        # sigma over exactly those tokens (supports per-sample-varying t).
        sigma = t / 1000.0  # (B,)
        vae_token_counts = (packed_seqlens.to(device) - 2).to(torch.long)
        # The caller-provided context must describe exactly ``num_tokens`` VAE tokens
        # per sample, or the ``v_t.reshape(B, n, d)`` below would silently misalign
        # sample boundaries (corrupting per-sample log-probs) or fail opaquely.
        if not bool((vae_token_counts == num_tokens).all()):
            raise ValueError(
                f"Packed forward requires uniform per-sample VAE token counts matching "
                f"latents (n={num_tokens}); got {vae_token_counts.tolist()}."
            )
        timestep_for_bagel = torch.repeat_interleave(sigma.to(device), vae_token_counts)

        # ── CFG gating based on sigma ──
        # Gating is shared across the pack, so when CFG is active all samples must be on
        # the same side of cfg_interval. Fail loudly on straddling rather than silently
        # applying the wrong CFG scale / renorm to some samples (a hidden train-inference
        # inconsistency). A schedule that draws one shared t for the pack never straddles.
        # When CFG is disabled (scales <= 1.0) gating is a no-op, so skip the check entirely.
        if cfg_text_scale > 1.0 or cfg_img_scale > 1.0:
            sigma_vals = sigma.flatten()
            in_interval = (sigma_vals > cfg_interval[0]) & (sigma_vals <= cfg_interval[1])
            if not (bool(in_interval.all()) or bool((~in_interval).all())):
                raise ValueError(
                    "Per-sample timesteps straddle cfg_interval; batched CFG gating requires "
                    f"all samples on the same side. Got sigmas {sigma_vals.tolist()} vs "
                    f"interval {cfg_interval}."
                )
            use_cfg = bool(in_interval[0])
        else:
            use_cfg = False
        cfg_text_s = cfg_text_scale if use_cfg else 1.0
        cfg_img_s = cfg_img_scale if use_cfg else 1.0

        # Helper: safely extract a CFG tensor onto the compute device.
        def _cfg(d: Optional[Dict], key: str) -> Optional[torch.Tensor]:
            if d is None:
                return None
            v = d.get(key)
            if isinstance(v, torch.Tensor):
                return v.to(device)
            return None

        # ── Flow velocity prediction (gradient-safe) ──
        v_t = self._forward_flow(
            x_t=packed_latents,
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
            cfg_text_scale=cfg_text_s,
            cfg_text_packed_position_ids=_cfg(cfg_text_generation_input, "cfg_packed_position_ids"),
            cfg_text_packed_query_indexes=_cfg(
                cfg_text_generation_input, "cfg_packed_query_indexes"
            ),
            cfg_text_key_values_lens=_cfg(cfg_text_generation_input, "cfg_key_values_lens"),
            cfg_text_past_key_values=cfg_text_past_kv,
            cfg_text_packed_key_value_indexes=_cfg(
                cfg_text_generation_input, "cfg_packed_key_value_indexes"
            ),
            cfg_img_scale=cfg_img_s,
            cfg_img_packed_position_ids=_cfg(cfg_img_generation_input, "cfg_packed_position_ids"),
            cfg_img_packed_query_indexes=_cfg(cfg_img_generation_input, "cfg_packed_query_indexes"),
            cfg_img_key_values_lens=_cfg(cfg_img_generation_input, "cfg_key_values_lens"),
            cfg_img_past_key_values=cfg_img_past_kv,
            cfg_img_packed_key_value_indexes=_cfg(
                cfg_img_generation_input, "cfg_packed_key_value_indexes"
            ),
            cfg_type="parallel",
        )

        # Packed (B*n, d) -> (B, n, d) so the scheduler reduces per sample -> (B,).
        v_t = v_t.reshape(batch_size, num_tokens, patch_dim)

        # ── Scheduler step (timesteps stay in [0, 1000]) ──
        output = self.scheduler.step(
            velocity=v_t,
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
        past_key_values: Optional[NaiveCache] = None,
        cfg_text_past_kv: Optional[NaiveCache] = None,
        cfg_img_past_kv: Optional[NaiveCache] = None,
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
        return_kwargs: Optional[List[str]] = None,
        # ── Context rebuild (training path) ──
        prompt: Optional[Union[str, List[str]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> SDESchedulerOutput:
        """
        Single denoising step: flow prediction -> scheduler step.

        Three calling modes:
          - **Inference**: ``generation_input`` + ``past_key_values`` provided
            (pre-built, possibly packed for B samples). Computed directly.
          - **Training, T2I**: context rebuilt from ``prompt`` (``List[str]`` of
            length B) and packed into one block-diagonal forward (NaViT).
          - **Training, I2I**: context rebuilt from ``prompt`` + per-sample
            ``condition_images`` and packed into one block-diagonal forward. Counts
            may vary per sample (subset-round packing); sizes may vary (model pads).

        **Timestep convention**: ``t`` / ``t_next`` are in [0, 1000]; converted to
        [0, 1] sigmas internally for the Bagel LLM, passed as-is to ``scheduler.step``.

        **Batch layout**: ``latents`` is ``(B, num_tokens, dim)`` (a bare
        ``(num_tokens, dim)`` is treated as ``B == 1``); ``t`` / ``t_next`` are
        ``(B,)`` (or scalar, broadcast). Returns per-sample ``(B,)`` ``log_prob``.

        Returns:
            ``SDESchedulerOutput`` with ``next_latents``, ``log_prob``,
            ``velocity``, etc. depending on ``return_kwargs``.
        """
        if return_kwargs is None:
            return_kwargs = [
                "velocity",
                "next_latents",
                "next_latents_mean",
                "std_dev_t",
                "dt",
                "log_prob",
            ]
        # Invariant CFG + scheduler params threaded unchanged through the forward
        # chain (forward -> _forward_rebuild -> _forward_packed). ``t`` / ``latents`` /
        # ``t_next`` / ``next_latents`` stay explicit; ``_forward_packed`` keeps these
        # as explicit params, so a malformed key surfaces as a TypeError at that boundary.
        step_kwargs: Dict[str, Any] = dict(
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            noise_level=noise_level,
            compute_log_prob=compute_log_prob,
            return_kwargs=return_kwargs,
        )

        # ── Inference path: context already built (single or packed) ──
        if generation_input is not None and past_key_values is not None:
            return self._forward_packed(
                t=t,
                latents=latents,
                generation_input=generation_input,
                cfg_text_generation_input=cfg_text_generation_input,
                cfg_img_generation_input=cfg_img_generation_input,
                past_key_values=past_key_values,
                cfg_text_past_kv=cfg_text_past_kv,
                cfg_img_past_kv=cfg_img_past_kv,
                t_next=t_next,
                next_latents=next_latents,
                **step_kwargs,
            )

        # ── Training path: rebuild KV-cache context from prompt ──
        if prompt is None:
            raise ValueError(
                "BagelAdapter.forward() requires either prebuilt `past_key_values` + "
                "`generation_input` (inference) or `prompt` (training) to build KV caches."
            )
        if isinstance(prompt, str):
            prompt = [prompt]
        # Resolution must come from the batch (BagelSample stores ``image_shape`` /
        # ``height`` / ``width`` as shared fields). Refuse to silently default to
        # 1024x1024, which would mismatch the stored latents and surface as an
        # opaque reshape failure downstream.
        if image_shape is None and "height" not in kwargs:
            raise ValueError(
                "Bagel training forward requires `image_shape` (or `height`/`width`) "
                "from the batch; refusing to default to 1024x1024."
            )
        _image_shape = image_shape or (kwargs["height"], kwargs["width"])

        # Per-sample List[List[PIL]] for I2I (after the FSDP guard), or None for T2I.
        # Both paths run the same NaViT-packed rebuild + forward.
        cond = self._resolve_condition_images_for_packing(condition_images)
        return self._forward_rebuild(
            t=t,
            latents=latents,
            prompts=prompt,
            condition_images=cond,
            image_shape=_image_shape,
            t_next=t_next,
            next_latents=next_latents,
            **step_kwargs,
        )

    def _forward_rebuild(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompts: List[str],
        condition_images: Optional[List[List[Image.Image]]],
        image_shape: Tuple[int, int],
        t_next: Optional[torch.Tensor],
        next_latents: Optional[torch.Tensor],
        **step_kwargs: Any,
    ) -> SDESchedulerOutput:
        """Rebuild a KV-cache context from prompts and run one ``_forward_packed``.

        Used by T2I training (``condition_images=None``) and batched I2I (per-sample
        ``condition_images`` of length B, counts may vary across samples — see
        ``_build_gen_context`` subset-round packing). ``condition_images`` is
        per-sample ``List[List[PIL]]``.

        ``step_kwargs`` carries the invariant CFG + scheduler params (``cfg_text_scale``,
        ``cfg_img_scale``, ``cfg_interval``, ``cfg_renorm_min``, ``cfg_renorm_type``,
        ``noise_level``, ``compute_log_prob``, ``return_kwargs``) forwarded as-is to
        ``_forward_packed``.
        """
        device = latents.device
        with torch.no_grad():
            gen_ctx, cfg_text_ctx, cfg_img_ctx = self._build_gen_context(
                prompts, condition_images=condition_images
            )
            generation_input, cfg_text_generation_input, cfg_img_generation_input = (
                self._prepare_gen_inputs(
                    gen_ctx,
                    cfg_text_ctx,
                    cfg_img_ctx,
                    image_sizes=[image_shape] * len(prompts),
                    device=device,
                )
            )
        return self._forward_packed(
            t=t,
            latents=latents,
            generation_input=generation_input,
            cfg_text_generation_input=cfg_text_generation_input,
            cfg_img_generation_input=cfg_img_generation_input,
            past_key_values=gen_ctx["past_key_values"],
            cfg_text_past_kv=cfg_text_ctx["past_key_values"],
            cfg_img_past_kv=cfg_img_ctx["past_key_values"],
            t_next=t_next,
            next_latents=next_latents,
            **step_kwargs,
        )
