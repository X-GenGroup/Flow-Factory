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

# src/flow_factory/models/qwen_image/qwen_image_edit_plus.py
from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

import diffusers

from ...contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaOrder,
    MediaGeometry,
    MediaType,
    NegativePromptPolicy,
)
from ...hparams import *
from ...samples import I2ISample, LatentState
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SDESchedulerOutput,
    set_scheduler_timesteps,
)
from ...utils.base import filter_kwargs
from ...utils.dist import context_parallel_geometry, tensor_parallel_geometry
from ...utils.image import (
    ImageBatch,
    ImageSingle,
    MultiImageBatch,
    is_image,
    is_image_batch,
    is_multi_image_batch,
    standardize_image_batch,
)
from ...utils.imports import is_version_at_least
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import (
    CallbackCollector,
    TrajectoryCollector,
    TrajectoryIndicesType,
    create_callback_collector,
    create_trajectory_collector,
)
from ..abc import BaseAdapter
from ..configured_image_output import ConfiguredImageOutputCodec
from ..output_state import (
    DecodedMediaBatch,
    EncodedOutputState,
    GeometrySignature,
    OutputStateCodec,
)
from ..parallelism.context import (
    ContextParallelBoundary,
    ContextParallelCapability,
    ContextParallelLayout,
)
from ..parallelism.plan import sequence_parallel_enabled
from ..pipeline_contracts import image_output_contract
from ._output import (
    encode_qwen_output_images,
    encode_qwen_vae_image,
    parse_qwen_condition_sizes,
)
from ._utils import _pad_seq_dim
from .qwen_image import QwenImageAdapter

logger = setup_logger(__name__)

CONDITION_IMAGE_SIZE = (1024, 1024)
CONDITION_IMAGE_SIZE_FOR_ENCODE = (384, 384)


class QwenEditContextParallelLayout(ContextParallelLayout):
    """Identify ordered target/reference tokens and native VL text conditioning."""

    ORDERED_REFERENCE_IMAGE_TEXT = "qwen_edit_ordered_reference_image_text"


@dataclass
class QwenImageEditPlusSample(I2ISample):
    """Output class for Qwen-Image-Edit Plus model."""

    # Class vars
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    # Obj vars
    prompt_embeds_mask: Optional[torch.FloatTensor] = None
    negative_prompt_embeds_mask: Optional[torch.FloatTensor] = None
    img_shapes: Optional[List[Tuple[int, int, int]]] = None
    image_latents: Optional[torch.Tensor] = None


def calculate_dimensions(target_area, ratio):
    # Calculate width and height based on target area and aspect ratio (height / width)
    height = math.sqrt(target_area * ratio)
    width = height / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


@dataclass(frozen=True, slots=True)
class _QwenImageEditOutputStateCodec:
    """Encode targets at the input-derived geometry used by Qwen Edit Plus."""

    adapter: "QwenImageEditPlusAdapter"
    required_components: ClassVar[Tuple[str, ...]] = ("vae",)

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        """Resolve output aspect ratio from the last reference, then VAE encode."""
        if len(media_batch) != 1:
            raise ValueError(
                "Qwen-Image-Edit-Plus output codec supports exactly one sample per batch, "
                f"received {len(media_batch)}"
            )
        images = ConfiguredImageOutputCodec._extract_images(media_batch)
        height, width = self.adapter._condition_derived_output_geometry(condition)
        pixel_values = self.adapter.pipeline.image_processor.preprocess(
            images,
            height=height,
            width=width,
        )
        ConfiguredImageOutputCodec._validate_pixel_values(
            pixel_values,
            1,
            height,
            width,
        )
        vae_dtype = getattr(self.adapter.vae, "dtype", None)
        if not isinstance(vae_dtype, torch.dtype) or not vae_dtype.is_floating_point:
            raise TypeError(
                "Qwen-Image-Edit-Plus output codec expected VAE floating dtype, "
                f"received {vae_dtype!r}"
            )
        pixel_values = pixel_values.to(device=self.adapter.device, dtype=vae_dtype)
        encoded_image = encode_qwen_output_images(
            self.adapter,
            pixel_values,
            condition,
            generator,
            condition_sizes_key="vae_image_sizes",
        )
        signature = GeometrySignature(
            media=(
                MediaGeometry(
                    type=MediaType.IMAGE,
                    height=height,
                    width=width,
                ),
            )
        )
        return EncodedOutputState(
            clean_state=LatentState({"latent": encoded_image.latents}),
            forward_context=encoded_image.forward_context,
            decode_context={"height": height, "width": width},
            geometry_signatures=(signature,),
        )


class QwenImageEditPlusAdapter(BaseAdapter):
    """Adapter for Qwen-Image-Edit Plus image editing models."""

    supports_sharded_target_materialization = True
    tensor_parallel_head_config = ("transformer", ("num_attention_heads",))
    context_parallel_capability = ContextParallelCapability(
        layout=QwenEditContextParallelLayout.ORDERED_REFERENCE_IMAGE_TEXT,
        tensor_parallel=True,
    )

    # Qwen-Image-Edit Plus runs with guidance=None, so the transformer's guidance
    # embedder receives no gradient and DDP must scan for unused parameters.
    ddp_find_unused_parameters = True
    supports_diffusers_cache = True
    pipeline_io_contract = image_output_contract(
        negative_prompt=NegativePromptPolicy.OPTIONAL,
        input_image_min_count=1,
        input_image_max_count=None,
        input_order=InputMediaOrder.WITHIN_TYPE,
        geometry_source=GeometrySource.INPUT_MEDIA,
        batch_capability=BatchCapability.SINGLE_SAMPLE,
    )

    def __init__(self, config: Arguments, accelerator: Accelerator):
        if not is_version_at_least("diffusers", "0.37.0"):
            raise ImportError(
                "QwenImageEditPlusAdapter requires diffusers>=0.37.0 (the "
                "transformer derives the text sequence length from "
                "encoder_hidden_states_mask; txt_seq_lens is no longer passed). "
                f"Found diffusers {diffusers.__version__}. "
                "Upgrade with `pip install -U 'diffusers>=0.37.0'`."
            )
        super().__init__(config, accelerator)
        self.pipeline: QwenImageEditPlusPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

        self._warned_cfg_no_neg_prompt = False
        self._warned_no_cfg = False
        self._has_warned_inference_fallback = False
        self._has_warned_forward_fallback = False
        self._has_warned_preprocess_fallback = False
        self._has_warned_inference_auto_resize = False

    def load_pipeline(self) -> QwenImageEditPlusPipeline:
        if self.config.parallelism_args.effective_context_parallel_size > 1:
            from diffusers import QwenImageTransformer2DModel

            config = QwenImageTransformer2DModel.load_config(
                self.model_args.model_name_or_path, subfolder="transformer"
            )
            self._validate_context_parallel_config(
                config,
                self.config.parallelism_args.effective_context_parallel_size,
                self.config.parallelism_args.effective_tensor_parallel_size,
            )
        return self._load_diffusers_pipeline(
            QwenImageEditPlusPipeline, self.model_args.model_name_or_path, low_cpu_mem_usage=False
        )

    @classmethod
    def compose_context_parallel_sequence_boundaries(
        cls, module: torch.nn.Module
    ) -> Tuple[ContextParallelBoundary, ...]:
        """Preserve CP-before-SP modulation and complete output before target cropping."""
        from ._context_parallel import compose_qwen_sequence_boundaries

        return compose_qwen_sequence_boundaries(module)

    @staticmethod
    def _validate_context_parallel_config(
        config: Mapping[str, Any], degree: int, tp_size: int = 1
    ) -> None:
        """Admit both native 2509 and 2511 layouts before loading model weights."""
        heads = config.get("num_attention_heads")
        if type(heads) is not int or heads < 1 or heads % (degree * tp_size):
            raise ValueError(
                "Qwen Edit Plus CP requires num_attention_heads divisible by tp_size*cp_size; "
                f"received heads={heads!r}, tp_size={tp_size}, cp_size={degree}"
            )
        if type(config.get("num_layers")) is not int or config["num_layers"] < 1:
            raise ValueError("Qwen Edit Plus CP requires native transformer blocks")
        for option in ("guidance_embeds", "use_additional_t_cond", "use_layer3d_rope"):
            if config.get(option, False) is not False:
                raise ValueError(f"Qwen Edit Plus CP requires {option}=False")
        if type(config.get("zero_cond_t", False)) is not bool:
            raise ValueError("Qwen Edit Plus CP requires zero_cond_t to be a boolean")
        channels = config.get("in_channels")
        output = config.get("out_channels") or channels
        patch = config.get("patch_size", 2)
        if (
            type(channels) is not int
            or channels < 4
            or channels % 4
            or type(output) is not int
            or patch != 2
            or output * patch**2 != channels
        ):
            raise ValueError(
                "Qwen Edit Plus CP requires patch_size=2 and matching packed input/output "
                f"channels; received in_channels={channels}, out_channels={output}, patch_size={patch}"
            )
        axes = config.get("axes_dims_rope", (16, 56, 56))
        head_dim = config.get("attention_head_dim")
        if (
            not isinstance(axes, (list, tuple))
            or len(axes) != 3
            or any(type(value) is not int or value <= 0 or value % 2 for value in axes)
            or sum(axes) != head_dim
        ):
            raise ValueError(
                "Qwen Edit Plus CP requires three positive even rotary axes matching the head "
                f"width; received axes_dims_rope={axes!r}, attention_head_dim={head_dim!r}"
            )

    @classmethod
    def validate_context_parallel_module(
        cls, module: torch.nn.Module, degree: int, tp_size: int = 1
    ) -> None:
        """Require native ordered rotary positions, modulation and joint attention."""
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenDoubleStreamAttnProcessor2_0,
            QwenEmbedRope,
        )

        from diffusers import QwenImageTransformer2DModel

        if not isinstance(module, QwenImageTransformer2DModel):
            raise ValueError(
                "Qwen Edit Plus CP requires QwenImageTransformer2DModel; "
                f"received {type(module).__name__}"
            )
        cls._validate_context_parallel_config(module.config, degree, tp_size)
        if not isinstance(module.pos_embed, QwenEmbedRope) or not module.transformer_blocks:
            raise ValueError("Qwen Edit Plus CP requires native ordered rotary transformer blocks")
        zero_cond_t = module.config.get("zero_cond_t", False)
        if module.zero_cond_t is not zero_cond_t or any(
            block.zero_cond_t is not zero_cond_t for block in module.transformer_blocks
        ):
            raise ValueError("Qwen Edit Plus CP zero_cond_t must match every native block")
        for name, child in module.named_modules():
            if hasattr(child, "processor") and not isinstance(
                child.processor, QwenDoubleStreamAttnProcessor2_0
            ):
                raise ValueError(
                    "Qwen Edit Plus CP requires native QwenDoubleStreamAttnProcessor2_0 "
                    f"at {name}; received {type(child.processor).__name__}"
                )

    @staticmethod
    def _validate_context_parallel_prompt(
        prompt_embeds: Union[List[torch.Tensor], torch.Tensor],
        prompt_embeds_mask: Union[List[torch.Tensor], torch.Tensor],
    ) -> None:
        """Reject malformed original rows before padding can conceal their mismatch."""
        if isinstance(prompt_embeds, torch.Tensor):
            if prompt_embeds.ndim != 3:
                raise ValueError("Qwen Edit CP embeddings require [batch, tokens, width]")
            embeddings = prompt_embeds.unbind(0)
        else:
            embeddings = prompt_embeds
        if isinstance(prompt_embeds_mask, torch.Tensor):
            if prompt_embeds_mask.ndim != 2:
                raise ValueError("Qwen Edit CP mask requires [batch, tokens]")
            masks = prompt_embeds_mask.unbind(0)
        else:
            masks = prompt_embeds_mask
        if not embeddings or not masks or len(embeddings) != len(masks):
            raise ValueError("Qwen Edit CP mask must match the original embedding batch")
        for index, (embedding, mask) in enumerate(zip(embeddings, masks)):
            if embedding.ndim != 2 or mask.ndim != 1 or embedding.shape[0] != mask.shape[0]:
                raise ValueError(
                    f"Qwen Edit CP mask must match original embeddings at row {index}: "
                    f"embedding shape={tuple(embedding.shape)}, mask shape={tuple(mask.shape)}"
                )

    def _validate_context_parallel_inputs(
        self,
        latents: torch.Tensor,
        image_latents: Optional[torch.Tensor],
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        img_shapes: Optional[List[List[Tuple[int, int, int]]]],
        attention_kwargs: Optional[Dict[str, Any]],
    ) -> None:
        """Validate ordered target/reference spans before any native CP collective."""
        degree, _ = context_parallel_geometry(self.accelerator)
        if degree == 1:
            return
        if attention_kwargs:
            raise ValueError("Qwen Edit Plus CP does not support attention_kwargs or IP adapters")
        config = self.pipeline.transformer.config
        for name, value, width in (
            ("latents", latents, config.in_channels),
            ("image_latents", image_latents, config.in_channels),
            ("prompt_embeds", prompt_embeds, config.joint_attention_dim),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 3
                or value.shape[0] != 1
                or value.shape[1] < 1
                or value.shape[-1] != width
            ):
                raise ValueError(
                    f"Qwen Edit Plus CP requires {name} with batch=1, positive tokens, width={width}; "
                    f"received shape={getattr(value, 'shape', None)}"
                )
        if prompt_embeds_mask.shape != prompt_embeds.shape[:2]:
            raise ValueError("Qwen Edit Plus CP text mask must match prompt embeddings")
        if prompt_embeds.shape[1] % degree:
            raise ValueError(
                "Qwen Edit Plus CP text tokens must be divisible by cp_size; "
                f"received text_tokens={prompt_embeds.shape[1]}, cp_size={degree}"
            )
        if (
            not isinstance(img_shapes, list)
            or len(img_shapes) != 1
            or not isinstance(img_shapes[0], (list, tuple))
            or len(img_shapes[0]) < 2
            or any(
                not isinstance(shape, (list, tuple))
                or len(shape) != 3
                or any(type(value) is not int or value < 1 for value in shape)
                or shape[0] != 1
                for shape in img_shapes[0]
            )
        ):
            raise ValueError(
                "Qwen Edit Plus CP requires target-first ordered (1, height, width) image shapes "
                f"and at least one reference; received img_shapes={img_shapes!r}"
            )
        spans = [math.prod(shape) for shape in img_shapes[0]]
        if spans[0] != latents.shape[1] or sum(spans[1:]) != image_latents.shape[1]:
            raise ValueError(
                "Qwen Edit Plus CP ordered shapes must match target and reference tokens; "
                f"received spans={spans}, target={latents.shape[1]}, references={image_latents.shape[1]}"
            )
        tp_size, _ = tensor_parallel_geometry(self.accelerator)
        sp_size = (
            tp_size
            if sequence_parallel_enabled(self, self.config.parallelism_args.sequence_parallel)
            else 1
        )
        if sum(spans) % (degree * sp_size):
            raise ValueError(
                "Qwen Edit Plus CP concatenated image tokens must be divisible by cp_size*sp_size; "
                f"received tokens={sum(spans)}, cp_size={degree}, sp_size={sp_size}"
            )

    def build_output_state_codec(self) -> OutputStateCodec:
        """Declare the condition-geometry-aware Qwen target image codec."""
        return _QwenImageEditOutputStateCodec(self)

    def _condition_derived_output_geometry(
        self,
        condition: Mapping[str, Any],
    ) -> Tuple[int, int]:
        """Match inference auto-resize using the last reference aspect ratio."""
        sizes = parse_qwen_condition_sizes(
            condition.get("condition_image_sizes"),
            batch_size=1,
            source="Qwen-Image-Edit-Plus condition_image_sizes",
        )[0]
        if not sizes:
            raise ValueError(
                "Qwen-Image-Edit-Plus offline output geometry requires at least one "
                "condition image size"
            )
        image_width, image_height = sizes[-1]
        configured_height = getattr(self.training_args, "height", None)
        configured_width = getattr(self.training_args, "width", None)
        if type(configured_height) is not int or type(configured_width) is not int:
            raise TypeError(
                "Qwen-Image-Edit-Plus offline output geometry requires configured integer "
                f"height/width, received {(configured_height, configured_width)!r}"
            )
        width, height = calculate_dimensions(
            configured_height * configured_width,
            image_height / image_width,
        )
        multiple = self.pipeline.vae_scale_factor * 2
        width = width // multiple * multiple
        height = height // multiple * multiple
        if width <= 0 or height <= 0:
            raise ValueError(
                "Qwen-Image-Edit-Plus derived non-positive output geometry " f"{(height, width)}"
            )
        return height, width

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Verify signatures, decode fields, and target-first ordered image shapes."""
        height, width = self._condition_derived_output_geometry(condition)
        expected_signature = GeometrySignature(
            media=(
                MediaGeometry(
                    type=MediaType.IMAGE,
                    height=height,
                    width=width,
                ),
            )
        )
        if len(media_batch) != 1 or encoded.geometry_signatures != (expected_signature,):
            raise ValueError(
                "Qwen-Image-Edit-Plus encoded geometry must match its single input-derived "
                f"target geometry {(height, width)}"
            )
        if dict(encoded.decode_context) != {"height": height, "width": width}:
            raise ValueError(
                "Qwen-Image-Edit-Plus decode_context must exactly match input-derived "
                f"geometry {(height, width)}, received {dict(encoded.decode_context)!r}"
            )
        expected_img_shapes = self._offline_img_shapes(height, width, condition)
        img_shapes = encoded.forward_context.get("img_shapes")
        if img_shapes != expected_img_shapes:
            raise ValueError(
                "Qwen-Image-Edit-Plus img_shapes must contain target geometry first and "
                "preserve ordered reference geometry: "
                f"expected {expected_img_shapes!r}, received {img_shapes!r}"
            )

    def _offline_img_shapes(
        self,
        height: int,
        width: int,
        condition: Mapping[str, Any],
    ) -> List[List[Tuple[int, int, int]]]:
        """Build the exact target-first shape sequence consumed by the transformer."""
        sizes = parse_qwen_condition_sizes(
            condition.get("vae_image_sizes"),
            batch_size=1,
            source="Qwen-Image-Edit-Plus vae_image_sizes",
        )[0]
        if not sizes:
            raise ValueError(
                "Qwen-Image-Edit-Plus offline img_shapes requires at least one VAE image size"
            )
        scale = self.pipeline.vae_scale_factor * 2
        shapes = [(1, height // scale, width // scale)]
        for size_index, (image_width, image_height) in enumerate(sizes):
            if image_width % scale or image_height % scale:
                raise ValueError(
                    "Qwen-Image-Edit-Plus VAE image geometry must be divisible by "
                    f"{scale}, received {(image_height, image_width)} at index {size_index}"
                )
            shapes.append((1, image_height // scale, image_width // scale))
        return [shapes]

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Qwen-Image-Edit-Plus transformer."""
        return [
            # Attention
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            # MLP
            "net.0.proj",
            "net.2",
        ]

    @classmethod
    def tensor_parallel_plan(cls) -> Dict[str, Any]:
        """Reuse the Qwen-Image plan: the same transformer class, sharded the same way.

        Edit-Plus is a different *pipeline* -- extra reference-image latents
        concatenated onto the image sequence, a different prompt template -- but
        the denoiser is literally ``QwenImageTransformer2DModel``, same 24 heads
        of 128 and the same unfused double-stream blocks. Delegating keeps one
        definition: a plan that drifted from Qwen-Image's would mean the two
        adapters disagreed about a module tree they share.

        This adapter does not subclass :class:`QwenImageAdapter` (the two derive
        from :class:`BaseAdapter` independently, with different samples, codecs
        and IO contracts), so the reuse has to be an explicit call rather than
        inheritance. The extra reference tokens only lengthen the sequence,
        which is orthogonal to a head-parallel split.
        """
        return QwenImageAdapter.tensor_parallel_plan()

    @classmethod
    def sequence_parallel_plan(cls) -> Dict[str, Any]:
        """Reuse Qwen's image/text SP on the shared native transformer.

        Reference images remain in their original image-sequence order. Text
        padding keeps the native mask, and Edit-2511 modulation indices follow
        their local image rows. The same wrapper boundaries preserve LoRA.
        """
        return QwenImageAdapter.sequence_parallel_plan()

    # ============================== Encoding and Decoding Methods ============================== #

    # ---------------------------------- Text Encoding ---------------------------------- #

    def _standardize_image_input(
        self,
        images: Union[ImageSingle, ImageBatch],
        output_type: Literal["pil", "pt", "np"] = "pil",
    ):
        """Standardize image input to desired output type."""
        if isinstance(images, Image.Image):
            images = [images]

        images = standardize_image_batch(
            images,
            output_type=output_type,
        )
        return images

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        images: Optional[Union[ImageSingle, ImageBatch]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 1024,
    ):
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        images = (
            self._standardize_image_input(images, output_type="pil") if images is not None else None
        )
        if isinstance(images, list):
            base_img_prompt = ""
            for i, img in enumerate(images):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif images is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = self.pipeline.prompt_template_encode

        drop_idx = self.pipeline.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]

        model_inputs = self.pipeline.processor(
            text=txt,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(device)
        input_ids = model_inputs.input_ids

        outputs = self.pipeline.text_encoder(
            input_ids=input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self.pipeline._extract_masked_hidden(
            hidden_states, model_inputs.attention_mask
        )
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states
        ]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
                for u in split_hidden_states
            ]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        input_ids = input_ids[:, drop_idx:]  # Extract only user input ids

        return input_ids, prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 4.0,
        images: Optional[Union[ImageSingle, ImageBatch]] = None,
        max_sequence_length: int = 1024,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Union[torch.Tensor, torch.Tensor]]:
        """Encode text prompts using the pipeline's text encoder."""

        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Encode positive prompt
        prompt_ids, prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
            prompt=prompt,
            images=images,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        results = {
            "prompt_ids": prompt_ids,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
        }
        # Encode negative prompt
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            )
            negative_prompt = negative_prompt * (
                len(prompt) // len(negative_prompt)
            )  # Expand to match batch size
            assert len(negative_prompt) == len(
                prompt
            ), "The number of negative prompts must match the number of prompts."
            negative_prompt_ids, negative_prompt_embeds, negative_prompt_embeds_mask = (
                self._get_qwen_prompt_embeds(
                    prompt=negative_prompt,
                    images=images,
                    device=device,
                    dtype=dtype,
                    max_sequence_length=max_sequence_length,
                )
            )
            results.update(
                {
                    "negative_prompt_ids": negative_prompt_ids,
                    "negative_prompt_embeds": negative_prompt_embeds[:, :max_sequence_length],
                    "negative_prompt_embeds_mask": negative_prompt_embeds_mask[
                        :, :max_sequence_length
                    ],
                }
            )

        return results

    # ---------------------------------------- Image Encoding ---------------------------------- #

    def _preprocess_condition_images(
        self,
        images: Union[ImageBatch, List[ImageBatch]],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], List[Tuple[int, int]]]]:
        """Preprocess the condition images of one sample and prepare image latents.

        Each image keeps its aspect ratio and is resized (to multiples of 32) to a 384x384 area
        for prompt encoding and to the ``condition_image_size`` area for VAE encoding.

        Args:
            images: Condition images (PIL, array, or tensor) used to generate one image.
            condition_image_size: Size whose area sets the VAE image resolution; an int means a
                square size.
            generator: Unused; condition latents take the VAE posterior mode.
            dtype: VAE input dtype; defaults to the VAE dtype.
            device: VAE input device; defaults to the VAE device.

        Returns:
            Dictionary containing:
                - "condition_images": Resized ``(C, H, W)`` images for prompt encoding.
                - "condition_image_sizes": ``(width, height)`` of each condition image.
                - "vae_images": Preprocessed ``(1, C, 1, H, W)`` images for VAE encoding.
                - "vae_image_sizes": ``(width, height)`` of each VAE image.
                - "image_latents": Packed latents of all images, ``(1, seq_len_total, C)``.
        """
        dtype = dtype or self.pipeline.vae.dtype
        device = device or self.pipeline.vae.device
        if isinstance(condition_image_size, int):
            condition_image_size = (condition_image_size, condition_image_size)

        condition_image_max_area = condition_image_size[0] * condition_image_size[1]
        condition_image_for_encode_max_area = (
            CONDITION_IMAGE_SIZE_FOR_ENCODE[0] * CONDITION_IMAGE_SIZE_FOR_ENCODE[1]
        )
        images = self._standardize_image_input(images, output_type="pil")

        condition_image_sizes = []
        condition_images = []
        vae_image_sizes = []
        vae_images = []
        for img in images:
            image_width, image_height = img.size
            # Keep the original aspect ratio at a fixed target area. Diffusers hard-codes 384x384
            # for `condition_images` (prompt encoding) and 1024x1024 for `vae_images` (the actual
            # conditioning input for generation); `condition_image_size` exposes the latter to
            # control the overall training resolution.
            condition_width, condition_height = calculate_dimensions(
                condition_image_for_encode_max_area, image_height / image_width
            )
            vae_width, vae_height = calculate_dimensions(
                condition_image_max_area, image_height / image_width
            )
            condition_image_sizes.append((condition_width, condition_height))
            vae_image_sizes.append((vae_width, vae_height))
            condition_image = self.pipeline.image_processor.resize(
                img, condition_height, condition_width
            )
            condition_image = self._standardize_image_input(condition_image, output_type="pt")[
                0
            ]  # Convert to tensor (C, H, W)
            condition_images.append(condition_image)
            vae_images.append(
                self.pipeline.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2)
            )  # (1, C, 1, H, W)

        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        image_latents = self.prepare_image_latents(
            images=vae_images,
            batch_size=1,
            num_channels_latents=num_channels_latents,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        return {
            "condition_images": condition_images,  # List[tensor(C, H, W)]
            "condition_image_sizes": condition_image_sizes,  # List[Tuple[int, int]]
            "vae_images": vae_images,  # List[tensor(1, C, 1, H, W)]
            "vae_image_sizes": vae_image_sizes,  # List[Tuple[int, int]]
            "image_latents": image_latents,  # tensor(1, seq_len_total, C)
        }

    def encode_image(
        self,
        images: Union[ImageBatch, List[ImageBatch]],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, List[Union[torch.Tensor, List[torch.Tensor], List[Tuple[int, int]]]]]:
        """Encode input images into latent representations using the VAE encoder.

        Args:
            images: A batch of condition-image lists; a non-nested list is treated as one sample.
            condition_image_size: Size whose area sets the VAE image resolution; an int means a
                square size.
            generator: Unused; condition latents take the VAE posterior mode.
            dtype: VAE input dtype; defaults to the VAE dtype.
            device: VAE input device; defaults to the VAE device.

        Returns:
            Dictionary mapping each ``_preprocess_condition_images`` key to a per-sample list.
        """
        # Check if input is a batch of condition image lists (nested batch)
        images = [images] if not is_multi_image_batch(images) else images

        results = defaultdict(list)
        for cond_images in images:
            encoded = self._preprocess_condition_images(
                cond_images,
                condition_image_size=condition_image_size,
                generator=generator,
                dtype=dtype,
                device=device,
            )
            for k, v in encoded.items():
                results[k].append(v)

        return results

    def prepare_image_latents(
        self,
        images: Union[ImageSingle, ImageBatch],
        batch_size: int,
        num_channels_latents: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        """Encode condition images with posterior argmax, then pack in input order."""
        del generator
        images = self._standardize_image_input(images, "pt")

        all_image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.pipeline.latent_channels:
                image_latents = encode_qwen_vae_image(
                    self,
                    image,
                    sample_mode="argmax",
                )
            else:
                image_latents = image
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # Expand image_latents to batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[3:]
            image_latents = self.pipeline._pack_latents(
                image_latents,
                batch_size,
                num_channels_latents,
                image_latent_height,
                image_latent_width,
            )
            all_image_latents.append(image_latents)
        image_latents = torch.cat(all_image_latents, dim=1)
        return image_latents

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        images: Optional[Union[ImageSingle, ImageBatch]] = None,
        image_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.pipeline.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.pipeline.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        if image_latents is None and images is not None:
            image_latents = self.prepare_image_latents(
                images=images,
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                dtype=dtype,
                device=device,
                generator=generator,
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.pipeline._pack_latents(
                latents, batch_size, num_channels_latents, height, width
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    # ---------------------------------------- Video Encoding ---------------------------------- #
    def encode_video(self, videos: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        """Not needed for Qwen-Image-Edit models."""
        return None

    # ---------------------------------------- Image Decoding ---------------------------------- #
    def decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: Literal["pil", "pt", "np"] = "pil",
    ) -> List[Image.Image]:
        """Decode latents to images using VAE."""

        latents = self.pipeline._unpack_latents(
            latents, height, width, self.pipeline.vae_scale_factor
        )
        latents = latents.to(self.pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(
            1, self.pipeline.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        images = self.pipeline.vae.decode(latents, return_dict=False)[0][:, :, 0]
        images = self.pipeline.image_processor.postprocess(images, output_type=output_type)

        return images

    # ======================== Preprocessing ========================
    def preprocess_func(
        self,
        prompt: List[str],
        images: MultiImageBatch,
        negative_prompt: Optional[List[str]] = None,
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        max_sequence_length: int = 1024,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Dict[str, List[Any]]:
        """Preprocess data samples for Qwen-Image-Edit Plus model training or evaluation.

        Args:
            prompt: A batch of text prompts.
            images: A batch of condition-image lists; a non-nested list is shared by all prompts.
            negative_prompt: Optional negative prompt, or one per prompt when given as a list.
            condition_image_size: Size whose area sets the VAE image resolution.
            max_sequence_length: Max sequence length for the text encoder.
            generator: Unused; condition latents take the VAE posterior mode.

        Returns:
            Dictionary mapping each encoded image and prompt field to a per-sample list.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        is_nested_batch = is_multi_image_batch(images)
        if not is_nested_batch:
            images = [images] * batch_size  # Duplicate for each prompt

        results = defaultdict(list)
        encoded_images = self.encode_image(
            images=images,
            condition_image_size=condition_image_size,
            generator=generator,
        )
        for k, v in encoded_images.items():
            results[k] = v

        for i in range(batch_size):
            encoded_prompts = self.encode_prompt(
                prompt=prompt[i],
                negative_prompt=(
                    negative_prompt[i] if isinstance(negative_prompt, list) else negative_prompt
                ),
                images=results["condition_images"][i],
                max_sequence_length=max_sequence_length,
            )
            for k, v in encoded_prompts.items():
                if isinstance(v, torch.Tensor):
                    v = list(v.unbind(0))  # Convert to a list for ragged case
                results[k].extend(v)

        return results

    # ======================== Padding Utilities ========================
    def _standardize_data(
        self,
        data: Union[None, torch.Tensor, List[torch.Tensor]],
        padding_value: Union[int, float],
        device: Optional[torch.device] = None,
        max_len: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if data is None:
            return None

        # If data is a list (ragged), pad it into a batch tensor first
        if isinstance(data, list):
            # Ensure data is on the correct device before padding
            if len(data) > 0 and data[0].device != device:
                data = [t.to(device) for t in data]
            data = pad_sequence(data, batch_first=True, padding_value=padding_value)
        else:
            data = data.to(device)

        return data[:, :max_len] if data.shape[1] > max_len else data

    def _pad_batch_prompt(
        self,
        prompt_embeds_mask: Union[List[torch.Tensor], torch.Tensor],
        prompt_embeds: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        prompt_ids: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if isinstance(prompt_embeds_mask, list):
            device = device or prompt_embeds_mask[0].device
            masks = prompt_embeds_mask
        else:
            device = device or prompt_embeds_mask.device
            masks = prompt_embeds_mask.unbind(0)
        # Native attention accepts holes. Keep the last valid token and its
        # rotary position rather than cropping by the number of active tokens.
        max_pos_len = max(
            1,
            max(
                (
                    int(
                        torch.where(
                            mask.bool(), torch.arange(1, mask.numel() + 1, device=mask.device), 0
                        ).max()
                    )
                    if mask.numel()
                    else 0
                )
                for mask in masks
            ),
        )

        if prompt_ids is not None:
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            padded_prompt_ids = self._standardize_data(
                prompt_ids,
                padding_value=pad_token_id,
                device=device,
                max_len=max_pos_len,
            )
        else:
            padded_prompt_ids = None

        if prompt_embeds is not None:
            padded_prompt_embeds = self._standardize_data(
                prompt_embeds,
                padding_value=0.0,
                device=device,
                max_len=max_pos_len,
            )
        else:
            padded_prompt_embeds = None

        padded_prompt_embeds_mask = self._standardize_data(
            prompt_embeds_mask,
            padding_value=0,
            device=device,
            max_len=max_pos_len,
        )
        return padded_prompt_embeds_mask, padded_prompt_embeds, padded_prompt_ids

    # ======================== Sampling / Inference ========================
    # Handle one sample
    @torch.no_grad()
    def _inference(
        self,
        # Ordinary arguments
        images: Optional[Union[ImageSingle, ImageBatch]] = None,
        prompt: Optional[Union[List[str], str]] = None,
        negative_prompt: Optional[Union[List[str], str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,  # Same as `true_cfg_scale` in QwenImageEditPlusPipeline.
        height: int = 1024,
        width: int = 1024,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # Prompt encoding arguments
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        # Image encoding arguments
        condition_images: Optional[Union[ImageSingle, ImageBatch]] = None,
        condition_image_sizes: Optional[List[Tuple[int, int]]] = None,
        vae_images: Optional[ImageBatch] = None,
        vae_image_sizes: Optional[List[Tuple[int, int]]] = None,
        image_latents: Optional[torch.Tensor] = None,
        # Other arguments
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        attention_kwargs: Optional[Dict[str, Any]] = {},
        max_sequence_length: int = 1024,
        compute_log_prob: bool = False,
        auto_resize: bool = True,
        # Callback arguments
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
    ) -> List[QwenImageEditPlusSample]:
        """Generate images using Qwen-Image-Edit Plus model."""
        # 1. Set up
        # Determine height and width, preferring encoded `condition_images` over raw `images`
        detemine_size_images = condition_images if condition_images is not None else images
        detemine_size_images = (
            self._standardize_image_input(detemine_size_images, output_type="pil")
            if detemine_size_images is not None
            else None
        )
        if detemine_size_images is not None and auto_resize:
            # Auto resize the output to the last condition image's aspect ratio at the same area
            image_size = detemine_size_images[-1].size
            calculated_width, calculated_height = calculate_dimensions(
                height * width, image_size[1] / image_size[0]
            )
            if (
                calculated_height != height or calculated_width != width
            ) and not self._has_warned_inference_auto_resize:
                self._has_warned_inference_auto_resize = True
                logger.warning(
                    f"Auto-resizing output from ({height}, {width}) to ({calculated_height}, {calculated_width}) "
                    f"to match input aspect ratio {image_size[1] / image_size[0]:.2f}. This message appears only once. "
                    f"To disable auto-resizing and enforce given resolution ({height}, {width}), set `auto_resize` to `false`."
                )

            height = calculated_height
            width = calculated_width

        multiple_of = self.pipeline.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # cfg and others
        guidance_scale = guidance_scale
        device = self.device
        dtype = self.pipeline.transformer.dtype
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        if guidance_scale > 1 and not has_neg_prompt and not self._warned_cfg_no_neg_prompt:
            self._warned_cfg_no_neg_prompt = True
            logger.warning(
                f"guidance_scale is passed as {guidance_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided. Warning will only be shown once."
            )
        elif guidance_scale <= 1 and has_neg_prompt and not self._warned_no_cfg:
            self._warned_no_cfg = True
            logger.warning(
                " negative_prompt is passed but classifier-free guidance is not enabled since guidance_scale <= 1. Warning will only be shown once."
            )

        # 2. Preprocess images
        if images is not None and (
            condition_images is None
            or vae_images is None
            or condition_image_sizes is None
            or vae_image_sizes is None
        ):
            # Process multiple condition images
            encoded_images = self._preprocess_condition_images(
                images=images,
                condition_image_size=condition_image_size,
                generator=generator,
                dtype=dtype,
                device=device,
            )
            condition_images = encoded_images["condition_images"]
            condition_image_sizes = encoded_images["condition_image_sizes"]
            vae_images = encoded_images["vae_images"]
            vae_image_sizes = encoded_images["vae_image_sizes"]
            image_latents = encoded_images["image_latents"]
        else:
            condition_images = (
                self._standardize_image_input(condition_images, output_type="pt")
                if condition_images is not None
                else None
            )
            if isinstance(vae_images, torch.Tensor):
                vae_images = list(vae_images.unbind(0))
            vae_images = [img.to(device) for img in vae_images]
            image_latents = image_latents.to(device) if image_latents is not None else None

        # 3. Encode prompts
        if (prompt is not None and (prompt_embeds is None or prompt_embeds_mask is None)) or (
            negative_prompt is not None
            and (negative_prompt_embeds is None or negative_prompt_embeds_mask is None)
        ):
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                images=condition_images,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            prompt_ids = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
            prompt_embeds_mask = encoded["prompt_embeds_mask"]
            negative_prompt_ids = encoded.get("negative_prompt_ids", None)
            negative_prompt_embeds = encoded.get("negative_prompt_embeds", None)
            negative_prompt_embeds_mask = encoded.get("negative_prompt_embeds_mask", None)
        else:
            prompt_embeds = (
                [p.to(device) for p in prompt_embeds]
                if isinstance(prompt_embeds, list)
                else prompt_embeds.to(device)
            )
            prompt_embeds_mask = (
                [p.to(device) for p in prompt_embeds_mask]
                if isinstance(prompt_embeds_mask, list)
                else prompt_embeds_mask.to(device)
            )
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = (
                    [p.to(device) for p in negative_prompt_embeds]
                    if isinstance(negative_prompt_embeds, list)
                    else negative_prompt_embeds.to(device)
                )
            if negative_prompt_embeds_mask is not None:
                negative_prompt_embeds_mask = (
                    [p.to(device) for p in negative_prompt_embeds_mask]
                    if isinstance(negative_prompt_embeds_mask, list)
                    else negative_prompt_embeds_mask.to(device)
                )

        batch_size = len(prompt_embeds)
        # Cached evaluation batches may tensorize reference dimensions. Normalize
        # their representation before constructing native rotary-position shapes.
        vae_image_sizes = parse_qwen_condition_sizes(
            [vae_image_sizes], batch_size=1, source="Qwen-Image-Edit-Plus vae_image_sizes"
        )[0]

        # 4. Prepare latents
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            images=vae_images,
            image_latents=image_latents,
        )
        img_shapes = [
            [
                (
                    1,
                    height // self.pipeline.vae_scale_factor // 2,
                    width // self.pipeline.vae_scale_factor // 2,
                ),
                *[
                    (
                        1,
                        vae_height // self.pipeline.vae_scale_factor // 2,
                        vae_width // self.pipeline.vae_scale_factor // 2,
                    )
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        # 5. Set scheduler timesteps
        timesteps = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device,
        )

        # 6. Denoising loop
        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latents = self.cast_latents(latents, default_dtype=dtype)
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(
                trajectory_indices, num_inference_steps
            )
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        for i, t in enumerate(timesteps):
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kwargs = list(
                set(["next_latents", "log_prob", "velocity"] + extra_call_back_kwargs)
            )
            current_compute_log_prob = compute_log_prob and current_noise_level > 0

            output = self._forward(
                t=t,
                t_next=t_next,
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                img_shapes=img_shapes,
                image_latents=image_latents,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                guidance_scale=guidance_scale,
                attention_kwargs=attention_kwargs,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=current_noise_level,
            )

            latents = self.cast_latents(output.next_latents, default_dtype=dtype)
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob:
                log_prob_collector.collect(output.log_prob, i)

            callback_collector.collect_step(
                step_idx=i,
                output=output,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": current_noise_level},
            )

        # 7. Post-process results
        generated_images = self.decode_latents(latents, height, width, output_type="pt")

        # 8. Collect results for each sample in the batch
        extra_call_back_res = callback_collector.get_result()  # (B, len(trajectory_indices), ...)
        callback_index_map = callback_collector.get_index_map()  # (T,) LongTensor
        all_latents = latent_collector.get_result()  # List[torch.Tensor(B, ...)]
        latent_index_map = latent_collector.get_index_map()  # (T+1,) LongTensor
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
        samples = [
            QwenImageEditPlusSample(
                # Denoising trajectory
                timesteps=timesteps,
                all_latents=(
                    torch.stack([lat[b] for lat in all_latents], dim=0)
                    if all_latents is not None
                    else None
                ),
                log_probs=(
                    torch.stack([lp[b] for lp in all_log_probs], dim=0)
                    if all_log_probs is not None
                    else None
                ),
                latent_index_map=latent_index_map,
                log_prob_index_map=log_prob_index_map,
                # Generated image
                image=generated_images[b],
                # Condition images
                image_latents=image_latents[b] if image_latents is not None else None,
                condition_images=(
                    condition_images if condition_images is not None else None
                ),  # No Batch dimension yet
                img_shapes=img_shapes[b],
                # Prompt
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b],
                prompt_embeds=prompt_embeds[b],
                prompt_embeds_mask=prompt_embeds_mask[b],
                # Negative Prompt
                negative_prompt=(
                    negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt
                ),
                negative_prompt_ids=(
                    negative_prompt_ids[b] if negative_prompt_ids is not None else None
                ),
                negative_prompt_embeds=(
                    negative_prompt_embeds[b] if negative_prompt_embeds is not None else None
                ),
                negative_prompt_embeds_mask=(
                    negative_prompt_embeds_mask[b]
                    if negative_prompt_embeds_mask is not None
                    else None
                ),
                # Extra kwargs
                extra_kwargs={
                    **{k: v[b] for k, v in extra_call_back_res.items()},
                    "callback_index_map": callback_index_map,
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()

        return samples

    @torch.no_grad()
    def inference(
        self,
        # Ordinary arguments
        images: Optional[MultiImageBatch] = None,
        prompt: Optional[List[str]] = None,
        negative_prompt: Optional[List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,  # Same as `true_cfg_scale` in QwenImageEditPlusPipeline.
        height: int = 1024,
        width: int = 1024,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # Prompt encoding arguments
        prompt_ids: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        prompt_embeds: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        prompt_embeds_mask: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        negative_prompt_ids: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        negative_prompt_embeds: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        negative_prompt_embeds_mask: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        # Image encoding arguments
        condition_images: Optional[MultiImageBatch] = None,  # A batch of condition image lists
        condition_image_sizes: Optional[
            List[List[Tuple[int, int]]]
        ] = None,  # A batch of condition image size lists
        vae_images: Optional[MultiImageBatch] = None,  # A batch of VAE image lists
        vae_image_sizes: Optional[
            List[List[Tuple[int, int]]]
        ] = None,  # A batch of VAE image size lists
        image_latents: Optional[List[torch.Tensor]] = None,  # A batch of image latents
        # Other arguments
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        attention_kwargs: Optional[Dict[str, Any]] = {},
        max_sequence_length: int = 1024,
        compute_log_prob: bool = False,
        auto_resize: bool = True,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
    ):
        """Run per-sample inference on batch-format inputs; only batch size 1 is supported."""
        batch_size = (
            len(images)
            if images is not None
            else (
                len(prompt)
                if prompt is not None
                else (
                    len(prompt_ids)
                    if prompt_ids is not None
                    else (
                        len(prompt_embeds)
                        if prompt_embeds is not None
                        else (
                            len(condition_images)
                            if condition_images is not None
                            else len(vae_images) if vae_images is not None else 1
                        )
                    )
                )
            )
        )

        if batch_size > 1:
            raise ValueError(
                f"Qwen-Image-Edit-Plus does not support batch_size > 1 for image-to-image tasks! "
                f"Unexpected error may occur, please set `per_device_batch_size` to `1` for both training and evaluation!"
            )

        # Process each sample individually by calling _inference
        all_samples = []
        for b in range(batch_size):
            sample = self._inference(
                # Extract b-th element from each parameter - be careful to tensors shape.
                # Ordinary Args
                images=images[b],
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                negative_prompt=(
                    negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt
                ),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator[b] if isinstance(generator, List) else generator,
                # Encoded prompt
                prompt_ids=prompt_ids[b : b + 1] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b : b + 1] if prompt_embeds is not None else None,
                prompt_embeds_mask=(
                    prompt_embeds_mask[b : b + 1] if prompt_embeds_mask is not None else None
                ),
                # Encoded negative prompt
                negative_prompt_ids=(
                    negative_prompt_ids[b : b + 1] if negative_prompt_ids is not None else None
                ),
                negative_prompt_embeds=(
                    negative_prompt_embeds[b : b + 1]
                    if negative_prompt_embeds is not None
                    else None
                ),
                negative_prompt_embeds_mask=(
                    negative_prompt_embeds_mask[b : b + 1]
                    if negative_prompt_embeds_mask is not None
                    else None
                ),
                # Encoded images
                condition_images=condition_images[b] if condition_images is not None else None,
                condition_image_sizes=(
                    condition_image_sizes[b] if condition_image_sizes is not None else None
                ),
                vae_images=vae_images[b] if vae_images is not None else None,
                vae_image_sizes=vae_image_sizes[b] if vae_image_sizes is not None else None,
                image_latents=image_latents[b] if image_latents is not None else None,
                # Shared parameters
                condition_image_size=condition_image_size,
                attention_kwargs=attention_kwargs,
                max_sequence_length=max_sequence_length,
                compute_log_prob=compute_log_prob,
                extra_call_back_kwargs=extra_call_back_kwargs,
                auto_resize=auto_resize,
                trajectory_indices=trajectory_indices,
            )
            all_samples.extend(sample)

        return all_samples

    # ======================== Forward for training ========================
    def _forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        # Args for I2I
        image_latents: Optional[torch.Tensor] = None,
        img_shapes: Optional[List[List[Tuple[int, int, int]]]] = None,
        # Args for CFG
        negative_prompt_embeds: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        negative_prompt_embeds_mask: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        guidance_scale: float = 4.0,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        noise_level: Optional[float] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "velocity",
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
        ],
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """Core forward pass handling both T2I and I2I.

        Args:
            t: Current timestep tensor.
            t_next: Next timestep tensor.
            latents: Current latent representations (B, seq_len, C).
            prompt_embeds: Text prompt embeddings.
            prompt_embeds_mask: Attention mask for prompt embeddings.
            img_shapes: List of image shapes per sample.
            image_latents: Optional condition image latents (for I2I).
            negative_prompt_embeds: Optional negative prompt embeddings (for CFG).
            negative_prompt_embeds_mask: Optional negative prompt attention mask.
            guidance_scale: CFG scale factor.
            next_latents: Optional target latents for log-prob computation.
            attention_kwargs: Optional kwargs for attention layers.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            noise_level: Current noise level for SDE sampling.

        Returns:
            SDESchedulerOutput containing requested outputs.
        """
        # 1. Prepare inputs
        device = latents.device
        batch_size = latents.shape[0]
        timestep = t.expand(batch_size).to(latents.dtype)
        has_negative_prompt = (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        if guidance_scale > 1.0 and not has_negative_prompt:
            logger.warning(
                "Passed `guidance_scale` > 1.0, but no `negative_prompt_embeds` provided. "
                "Classifier-free guidance will be disabled."
            )
        do_classifier_free_guidance = guidance_scale > 1.0 and has_negative_prompt
        guidance = None  # Always None for Qwen-Image-Edit Plus

        cp_degree, _ = context_parallel_geometry(self.accelerator)
        if cp_degree > 1:
            self._validate_context_parallel_prompt(prompt_embeds, prompt_embeds_mask)
            if do_classifier_free_guidance:
                self._validate_context_parallel_prompt(
                    negative_prompt_embeds, negative_prompt_embeds_mask
                )

        # Truncate prompt embeddings and masks to the max valid length in the
        # batch. diffusers (>=0.37) derives the per-sample text length from
        # encoder_hidden_states_mask, so the deprecated txt_seq_lens is not passed.
        prompt_embeds_mask, prompt_embeds, _ = self._pad_batch_prompt(
            prompt_embeds_mask=prompt_embeds_mask, prompt_embeds=prompt_embeds, device=device
        )

        if do_classifier_free_guidance:
            negative_prompt_embeds_mask, negative_prompt_embeds, _ = self._pad_batch_prompt(
                prompt_embeds_mask=negative_prompt_embeds_mask,
                prompt_embeds=negative_prompt_embeds,
                device=device,
            )

        if cp_degree > 1:
            seq_len = prompt_embeds.shape[1]
            if do_classifier_free_guidance:
                seq_len = max(seq_len, negative_prompt_embeds.shape[1])
            seq_len = ((seq_len + cp_degree - 1) // cp_degree) * cp_degree
            prompt_embeds = _pad_seq_dim(prompt_embeds, seq_len, 0.0)
            prompt_embeds_mask = _pad_seq_dim(prompt_embeds_mask, seq_len, 0)
            self._validate_context_parallel_inputs(
                latents,
                image_latents,
                prompt_embeds,
                prompt_embeds_mask,
                img_shapes,
                attention_kwargs,
            )
            if do_classifier_free_guidance:
                negative_prompt_embeds = _pad_seq_dim(negative_prompt_embeds, seq_len, 0.0)
                negative_prompt_embeds_mask = _pad_seq_dim(negative_prompt_embeds_mask, seq_len, 0)
                self._validate_context_parallel_inputs(
                    latents,
                    image_latents,
                    negative_prompt_embeds,
                    negative_prompt_embeds_mask,
                    img_shapes,
                    attention_kwargs,
                )

        # Prepare model input (concatenate condition latents for I2I)
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)

        # 2. Transformer forward pass
        if do_classifier_free_guidance:
            # Merge cond/uncond into one batched forward (halves transformer
            # calls). cond/uncond share latent_model_input (image latents are
            # CFG-invariant); pad both text streams to a common length and mask
            # via encoder_hidden_states_mask (diffusers derives each sample's
            # length from it). Output is sliced to the generated-latent tokens.
            # The merged cond/uncond batch shares one cache_context.
            # Tradeoff: ~2x peak activation memory vs two serial forwards
            # (lower batch/resolution if it OOMs).
            seq_len = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
            prompt_embeds = _pad_seq_dim(prompt_embeds, seq_len, 0.0)
            prompt_embeds_mask = _pad_seq_dim(prompt_embeds_mask, seq_len, 0)
            negative_prompt_embeds = _pad_seq_dim(negative_prompt_embeds, seq_len, 0.0)
            negative_prompt_embeds_mask = _pad_seq_dim(negative_prompt_embeds_mask, seq_len, 0)

            with self.pipeline.transformer.cache_context("cond_uncond"):
                both_pred = self.transformer(
                    hidden_states=torch.cat([latent_model_input, latent_model_input], dim=0),
                    timestep=torch.cat([timestep, timestep], dim=0) / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=torch.cat(
                        [prompt_embeds_mask, negative_prompt_embeds_mask], dim=0
                    ),
                    encoder_hidden_states=torch.cat([prompt_embeds, negative_prompt_embeds], dim=0),
                    img_shapes=img_shapes * 2,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            both_pred = both_pred[:, : latents.size(1)]
            velocity, neg_velocity = both_pred.chunk(2, dim=0)

            comb_pred = neg_velocity + guidance_scale * (velocity - neg_velocity)

            # Rescale norm (Qwen-Image-Edit Plus specific)
            cond_norm = torch.norm(velocity, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            velocity = comb_pred * (cond_norm / noise_norm)
        else:
            # Single conditional forward pass (no CFG).
            with self.pipeline.transformer.cache_context("cond"):
                velocity = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            velocity = velocity[:, : latents.size(1)]

        # 3. Scheduler step
        output = self.scheduler.step(
            velocity=velocity,
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
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: Union[torch.Tensor, List[torch.Tensor]],
        # Optional for I2I (can be List for ragged batches)
        image_latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        img_shapes: Optional[List[List[Tuple[int, int, int]]]] = None,
        # Optional for CFG
        negative_prompt_embeds: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        negative_prompt_embeds_mask: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        guidance_scale: float = 4.0,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        noise_level: Optional[float] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "velocity",
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
        ],
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """General forward method handling both T2I and I2I, including ragged I2I batches.

        Args:
            t: Current timestep tensor (scalar or batch).
            t_next: Next timestep tensor (scalar or batch).
            latents: Current latent representations (B, seq_len, C).
            prompt_embeds: Text prompt embeddings (B, seq_len, D).
            prompt_embeds_mask: Attention mask for prompts.
            img_shapes: Image shapes per sample for transformer.
            image_latents: Optional condition image latents (for I2I).
            negative_prompt_embeds: Optional negative prompt embeddings.
            negative_prompt_embeds_mask: Optional negative prompt mask.
            guidance_scale: CFG scale factor.
            next_latents: Optional target latents for log-prob computation.
            attention_kwargs: Optional kwargs for attention layers.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            noise_level: Current noise level for SDE sampling.

        Returns:
            SDESchedulerOutput containing requested outputs.
        """
        # Check whether the batch carries condition image latents (I2I)
        has_images = image_latents is not None
        if not has_images:
            # T2I: call _forward() directly
            return self._forward(
                t=t,
                t_next=t_next,
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                img_shapes=img_shapes,
                image_latents=image_latents,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                guidance_scale=guidance_scale,
                next_latents=next_latents,
                attention_kwargs=attention_kwargs,
                compute_log_prob=compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=noise_level,
            )

        # I2I: process one by one
        batch_size = len(latents)
        if batch_size > 1:
            first_latent_shape = latents[0].shape
            if not all(first_latent_shape == lat.shape for lat in latents[1:]):
                raise ValueError(
                    f"Qwen-Image-Edit-Plus does not support batch_size > 1 for image-to-image tasks! "
                    f"Unexpected error may occur, please set `per_device_batch_size` to `1` for both training and evaluation!"
                )

        outputs = []

        for idx in range(batch_size):
            # Extract single sample tensors - keep batch dimension as 1
            single_latents = latents[idx].unsqueeze(0)
            single_prompt_embeds = prompt_embeds[idx].unsqueeze(0)
            single_prompt_embeds_mask = prompt_embeds_mask[idx].unsqueeze(0)

            single_negative_prompt_embeds = (
                negative_prompt_embeds[idx].unsqueeze(0)
                if negative_prompt_embeds is not None
                else None
            )
            single_negative_prompt_embeds_mask = (
                negative_prompt_embeds_mask[idx].unsqueeze(0)
                if negative_prompt_embeds_mask is not None
                else None
            )

            single_img_shapes = [img_shapes[idx]] if img_shapes is not None else None
            single_image_latents = (
                image_latents[idx].unsqueeze(0) if image_latents[idx] is not None else None
            )
            single_next_latents = (
                next_latents[idx].unsqueeze(0) if next_latents is not None else None
            )

            out = self._forward(
                t=self._select_sample_time(t, idx, batch_size, "t"),
                t_next=self._select_sample_time(t_next, idx, batch_size, "t_next"),
                latents=single_latents,
                prompt_embeds=single_prompt_embeds,
                prompt_embeds_mask=single_prompt_embeds_mask,
                img_shapes=single_img_shapes,
                image_latents=single_image_latents,
                negative_prompt_embeds=single_negative_prompt_embeds,
                negative_prompt_embeds_mask=single_negative_prompt_embeds_mask,
                guidance_scale=guidance_scale,
                next_latents=single_next_latents,
                attention_kwargs=attention_kwargs,
                compute_log_prob=compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=noise_level,
            )
            outputs.append(out)

        # Preserve the native tensor graph and public container for one sample;
        # separately concatenated mean/velocity tensors are siblings of log_prob.
        if len(outputs) == 1:
            return FlowMatchEulerDiscreteSDESchedulerOutput.from_dict(outputs[0].to_dict())

        # Concatenate outputs along batch dimension
        outputs_dict = [o.to_dict() for o in outputs]
        return FlowMatchEulerDiscreteSDESchedulerOutput.from_dict(
            {
                k: (
                    torch.cat([o[k] for o in outputs_dict], dim=0)
                    if outputs_dict[0][k] is not None
                    else None
                )
                for k in outputs_dict[0].keys()
            }
        )

    @staticmethod
    def _select_sample_time(
        value: Optional[torch.Tensor],
        index: int,
        batch_size: int,
        name: str,
    ) -> Optional[torch.Tensor]:
        """Return one sample's timestep for the per-sample I2I forward.

        Args:
            value: Shared scalar timestep or a per-sample tensor of leading size ``batch_size``.
            index: Sample index within the batch.
            batch_size: Number of samples in the batch.
            name: Argument name reported by validation errors.

        Returns:
            ``value`` unchanged when it holds one element, otherwise ``value[index:index + 1]``.

        Raises:
            ValueError: If a multi-element ``value`` is not sized by the batch.
        """
        if value is None or value.numel() == 1:
            return value
        if value.ndim == 0 or value.shape[0] != batch_size:
            raise ValueError(
                f"expected {name} to hold one value or {batch_size} per-sample values for the "
                f"Qwen-Image-Edit-Plus I2I forward, received shape {tuple(value.shape)}"
            )
        return value[index : index + 1]
