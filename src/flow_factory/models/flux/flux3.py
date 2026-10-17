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

# src/flow_factory/models/flux/flux3.py
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from PIL import Image

import diffusers
from diffusers.utils.torch_utils import randn_tensor

try:
    from diffusers import Flux3Transformer2DModel, ModularPipeline
    from diffusers.modular_pipelines.flux3 import (
        Flux3AutoBlocks,
        Flux3DistilledAutoBlocks,
        Flux3DistilledModularPipeline,
        Flux3ModularPipeline,
    )
    from diffusers.modular_pipelines.flux3.image_processor import Flux3ImageProcessor

    _FLUX3_IMPORT_ERROR: Optional[Exception] = None
except (ImportError, RuntimeError) as exc:  # pragma: no cover - exercised with older diffusers
    Flux3Transformer2DModel = Any  # type: ignore[misc,assignment]
    ModularPipeline = Any  # type: ignore[misc,assignment]
    Flux3AutoBlocks = Any  # type: ignore[misc,assignment]
    Flux3DistilledAutoBlocks = Any  # type: ignore[misc,assignment]
    Flux3DistilledModularPipeline = Any  # type: ignore[misc,assignment]
    Flux3ModularPipeline = Any  # type: ignore[misc,assignment]
    Flux3ImageProcessor = Any  # type: ignore[misc,assignment]
    _FLUX3_IMPORT_ERROR = exc

from ...contracts import InputMediaOrder, NegativePromptPolicy, PipelineIOContract
from ...hparams import Arguments
from ...samples import I2ISample
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    set_scheduler_timesteps,
)
from ...utils.image import (
    ImageBatch,
    ImageSingle,
    MultiImageBatch,
    is_multi_image_batch,
    standardize_image_batch,
)
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import (
    TrajectoryIndicesType,
    create_callback_collector,
    create_trajectory_collector,
)
from ..abc import BaseAdapter
from ..checkpointing import CheckpointUnit
from ..configured_image_output import (
    ConfiguredImageOutputAdapterMixin,
    EncodedImageTensor,
)
from ..pipeline_contracts import image_output_contract
from ..precision import DTypePolicy
from ..runtime import ModularPipelineRuntime
from ._output import (
    encode_flux3_output_images,
    pack_flux3_latents,
    prepare_flux3_condition_latents,
    prepare_flux3_image_ids,
    unpack_flux3_latents,
    unpatchify_flux2_latents,
)

logger = setup_logger(__name__)

FLUX3_DIFFUSERS_COMMIT = "7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941"
FLUX3_INSTALL = "git submodule update --init && pip install -e ./diffusers"
CONDITION_IMAGE_SIZE = (1024, 1024)
IMAGE_ROPE_SCALE = 0.5
TEXT_PAD_MULTIPLE = 80
# Matches the `Flux3TextEncoderStep` inference default, so long captions are not truncated.
DEFAULT_MAX_SEQUENCE_LENGTH = 8192
DEFAULT_TEXT_ENCODER_OUT_LAYERS: Tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32)


@dataclass
class Flux3Sample(I2ISample):
    """Output sample class for Flux3Adapter models."""

    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    latent_ids: Optional[torch.Tensor] = None
    text_ids: Optional[torch.Tensor] = None
    negative_text_ids: Optional[torch.Tensor] = None
    image_latents: Optional[torch.Tensor] = None
    image_latent_ids: Optional[torch.Tensor] = None


class Flux3Adapter(ConfiguredImageOutputAdapterMixin, BaseAdapter):
    """Adapter for FLUX.3 modular text-to-image and multi-reference image-to-image models."""

    supports_diffusers_cache: ClassVar[bool] = False
    supports_fsdp2_cpu_efficient_loading: ClassVar[bool] = True
    ddp_find_unused_parameters: ClassVar[bool] = True
    component_load_dtype_defaults: ClassVar[DTypePolicy] = MappingProxyType(
        {
            "transformers": torch.bfloat16,
            "text_encoders": torch.bfloat16,
            "vae": torch.float32,
        }
    )
    offline_training_forward_overrides = MappingProxyType({"guidance_scale": 1.0})
    pipeline_io_contract = image_output_contract(
        negative_prompt=NegativePromptPolicy.OPTIONAL,
        input_image_min_count=0,
        input_image_max_count=None,
        input_order=InputMediaOrder.WITHIN_TYPE,
    )

    def __init__(self, config: Arguments, accelerator: Accelerator):
        if _FLUX3_IMPORT_ERROR is not None:
            raise ImportError(
                "Flux3Adapter requires the bundled Diffusers submodule at "
                f"{FLUX3_DIFFUSERS_COMMIT}. Install it with `{FLUX3_INSTALL}`. "
                f"The active diffusers version is {getattr(diffusers, '__version__', 'unknown')}."
            ) from _FLUX3_IMPORT_ERROR
        super().__init__(config, accelerator)
        self.pipeline: Flux3ModularPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler
        self._has_warned_inference_fallback = False
        self._has_warned_forward_fallback = False

    def load_pipeline(self) -> Flux3ModularPipeline:
        """Load the FLUX.3 modular pipeline specification from a local path or Hub repo."""
        if _FLUX3_IMPORT_ERROR is not None:
            raise ImportError(
                "Flux3Adapter requires the bundled Diffusers submodule at "
                f"{FLUX3_DIFFUSERS_COMMIT}. Install it with `{FLUX3_INSTALL}`. "
                f"The active diffusers version is {getattr(diffusers, '__version__', 'unknown')}."
            ) from _FLUX3_IMPORT_ERROR
        model_name_or_path = self.model_args.model_name_or_path
        pipeline = ModularPipeline.from_pretrained(model_name_or_path)
        if Path(model_name_or_path).is_dir():
            for name in pipeline.pretrained_component_names:
                spec = pipeline._component_specs[name]
                spec.pretrained_model_name_or_path = model_name_or_path
                spec.revision = None
        return pipeline

    def build_component_runtime(self) -> ModularPipelineRuntime:
        """Build the modular runtime and materialize the target transformer."""
        runtime = ModularPipelineRuntime.from_adapter(self, self.load_pipeline())
        target_names = runtime.resolve_component_names(self.model_args.target_components)
        runtime.materialize_components(target_names)
        return runtime

    @property
    def is_distilled(self) -> bool:
        """Return whether the loaded checkpoint uses the step-distilled flow-map contract."""
        transformer = (
            self.get_component("transformer") if self.has_component("transformer") else None
        )
        transformer_config = getattr(transformer, "config", None)
        if getattr(transformer_config, "flow_map", None) is not None:
            return bool(transformer_config.flow_map)
        if Flux3DistilledModularPipeline is not Any and isinstance(
            self.pipeline, Flux3DistilledModularPipeline
        ):
            return True
        return "guider" not in self.component_runtime.declared_component_names

    def _resolve_pipeline_io_contract(self) -> Optional[PipelineIOContract]:
        """Narrow negative-prompt support when a step-distilled flow-map checkpoint is loaded."""
        if self.is_distilled:
            return image_output_contract(
                negative_prompt=NegativePromptPolicy.UNSUPPORTED,
                input_image_min_count=0,
                input_image_max_count=None,
                input_order=InputMediaOrder.WITHIN_TYPE,
            )
        return type(self).pipeline_io_contract

    @property
    def preprocessing_modules(self) -> List[str]:
        return ["text_encoder", "tokenizer", "vae", "image_processor"]

    @property
    def inference_modules(self) -> List[str]:
        return ["transformer", "vae", "image_processor"]

    @property
    def tokenizer(self) -> Any:
        """Expose the tokenizer owned by the Qwen3-VL processor."""
        if self.get_component("tokenizer") is None and self.has_component("tokenizer"):
            self.component_runtime.materialize_components(["tokenizer"])
        processor = self.get_component("tokenizer")
        if processor is None:
            raise ValueError("No tokenizer found in the FLUX.3 pipeline.")
        return getattr(processor, "tokenizer", processor)

    @property
    def processor(self) -> Any:
        """Return the Qwen3-VL processor declared under the modular ``tokenizer`` slot."""
        if self.get_component("tokenizer") is None and self.has_component("tokenizer"):
            self.component_runtime.materialize_components(["tokenizer"])
        processor = self.get_component("tokenizer")
        if processor is None:
            raise ValueError("No processor/tokenizer component found in the FLUX.3 pipeline.")
        return processor

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules across all FLUX.3 mode and single-stream blocks."""
        return [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "ff.linear_in",
            "ff.linear_out",
        ]

    def _gradient_checkpointing_units(
        self,
        component_name: str,
        component: torch.nn.Module,
    ) -> List[CheckpointUnit]:
        """Return FLUX.3 transformer blocks in forward execution order."""
        txt_blocks = getattr(component, "txt_mode_blocks", None)
        mode_blocks = getattr(component, "mode_blocks", None)
        single_blocks = getattr(component, "single_transformer_blocks", None)
        image_blocks = (
            mode_blocks["image"]
            if isinstance(mode_blocks, nn.ModuleDict) and "image" in mode_blocks
            else None
        )
        cond_blocks = (
            mode_blocks["image_cond"]
            if isinstance(mode_blocks, nn.ModuleDict) and "image_cond" in mode_blocks
            else None
        )
        if (
            isinstance(txt_blocks, nn.ModuleList)
            and isinstance(image_blocks, nn.ModuleList)
            and isinstance(cond_blocks, nn.ModuleList)
            and isinstance(single_blocks, nn.ModuleList)
        ):
            return [
                *[(f"txt_mode_blocks.{idx}", block) for idx, block in enumerate(txt_blocks)],
                *[(f"mode_blocks.image.{idx}", block) for idx, block in enumerate(image_blocks)],
                *[
                    (f"mode_blocks.image_cond.{idx}", block)
                    for idx, block in enumerate(cond_blocks)
                ],
                *[
                    (f"single_transformer_blocks.{idx}", block)
                    for idx, block in enumerate(single_blocks)
                ],
            ]
        return super()._gradient_checkpointing_units(component_name, component)

    # ======================== Prompt Encoding ========================

    @staticmethod
    def _prepare_text_ids(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Prepare 4D position IDs ``(T, H, W, L)`` of shape ``(seq_len, 4)`` for text tokens."""
        if seq_len <= 0:
            raise ValueError(f"expected positive text sequence length, received {seq_len}")
        ids = torch.cartesian_prod(
            torch.arange(1),
            torch.arange(1),
            torch.arange(1),
            torch.arange(seq_len),
        ).to(dtype=torch.float32)
        if device is not None:
            ids = ids.to(device)
        return ids

    @staticmethod
    def _encode_single_qwen3_vl_prompt(
        text_encoder: Any,
        processor: Any,
        prompt: str,
        *,
        dtype: torch.dtype,
        device: torch.device,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        pad_multiple: int = TEXT_PAD_MULTIPLE,
        text_encoder_out_layers: Sequence[int] = DEFAULT_TEXT_ENCODER_OUT_LAYERS,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode one prompt with the FLUX.3 80-token right-padded Qwen3-VL recipe."""
        if not isinstance(prompt, str):
            raise TypeError(
                f"expected prompt to be str, received {type(prompt).__name__}: {prompt!r}"
            )
        if max_sequence_length <= 0:
            raise ValueError(
                f"expected positive max_sequence_length, received {max_sequence_length}"
            )
        if pad_multiple <= 0:
            raise ValueError(f"expected positive pad_multiple, received {pad_multiple}")
        if not text_encoder_out_layers:
            raise ValueError("expected non-empty text_encoder_out_layers")

        inner_tokenizer = getattr(processor, "tokenizer", processor)
        chat_text = processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        unpadded_ids = inner_tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_sequence_length,
        )["input_ids"]
        real_length = max(int(unpadded_ids.shape[1]), 1)
        padded_length = min(
            math.ceil(real_length / pad_multiple) * pad_multiple,
            max_sequence_length,
        )
        inputs = inner_tokenizer(
            chat_text,
            return_tensors="pt",
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=padded_length,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        # Qwen3-VL computes padding position IDs as 1 in transformers 4.57 (the FLUX.3 training
        # environment) and 0 in 5.x. Pass explicit position IDs so padding tokens match upstream.
        position_ids = torch.arange(padded_length, device=device)[None].masked_fill(
            attention_mask == 0, 1
        )

        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )

        stacked = torch.stack([output.hidden_states[k] for k in text_encoder_out_layers], dim=1)
        stacked = stacked.to(dtype=dtype, device=device)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_layers * hidden_dim
        )
        return input_ids.squeeze(0), prompt_embeds.squeeze(0)

    @classmethod
    def _get_qwen3_vl_prompt_embeds(
        cls,
        text_encoder: Any,
        processor: Any,
        prompt: Union[str, Sequence[str]],
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        pad_multiple: int = TEXT_PAD_MULTIPLE,
        text_encoder_out_layers: Sequence[int] = DEFAULT_TEXT_ENCODER_OUT_LAYERS,
    ) -> Tuple[
        Union[torch.Tensor, List[torch.Tensor]],
        Union[torch.Tensor, List[torch.Tensor]],
        Union[torch.Tensor, List[torch.Tensor]],
    ]:
        """Encode one or more prompts while preserving each prompt's exact 80-token window."""
        resolved_dtype = text_encoder.dtype if dtype is None else dtype
        resolved_device = text_encoder.device if device is None else device
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts:
            raise ValueError("expected at least one prompt to encode")

        ids_list: List[torch.Tensor] = []
        embeds_list: List[torch.Tensor] = []
        text_ids_list: List[torch.Tensor] = []
        for single_prompt in prompts:
            single_ids, single_embeds = cls._encode_single_qwen3_vl_prompt(
                text_encoder=text_encoder,
                processor=processor,
                prompt=single_prompt,
                dtype=resolved_dtype,
                device=resolved_device,
                max_sequence_length=max_sequence_length,
                pad_multiple=pad_multiple,
                text_encoder_out_layers=text_encoder_out_layers,
            )
            ids_list.append(single_ids)
            embeds_list.append(single_embeds)
            text_ids_list.append(
                cls._prepare_text_ids(single_embeds.shape[0], device=resolved_device)
            )

        lengths = {item.shape[0] for item in embeds_list}
        if len(lengths) == 1:
            return (
                torch.stack(ids_list, dim=0),
                torch.stack(embeds_list, dim=0),
                torch.stack(text_ids_list, dim=0),
            )
        return ids_list, embeds_list, text_ids_list

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 3.5,
        device: Optional[torch.device] = None,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        text_encoder_out_layers: Sequence[int] = DEFAULT_TEXT_ENCODER_OUT_LAYERS,
        pad_multiple: int = TEXT_PAD_MULTIPLE,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """Encode prompt(s) and optional negative prompt(s) with Qwen3-VL."""
        if (
            self.get_component("text_encoder") is None or self.get_component("tokenizer") is None
        ) and self.has_component("text_encoder"):
            self.on_load_components(["text_encoder", "tokenizer"], device=device)

        text_encoder = self._require_component("text_encoder")
        processor = self.processor
        resolved_device = text_encoder.device if device is None else device

        if prompt is None:
            prompt = ""
        prompt_list = [prompt] if isinstance(prompt, str) else list(prompt)

        if self.is_distilled and negative_prompt is not None:
            neg_values = (
                [negative_prompt] if isinstance(negative_prompt, str) else list(negative_prompt)
            )
            if any(item is not None and item != "" for item in neg_values):
                raise ValueError(
                    "Step-distilled FLUX.3 (flow_map=True) does not support negative_prompt; "
                    f"received {negative_prompt!r}."
                )

        do_classifier_free_guidance = (not self.is_distilled) and guidance_scale > 1.0

        prompt_ids, prompt_embeds, text_ids = self._get_qwen3_vl_prompt_embeds(
            text_encoder=text_encoder,
            processor=processor,
            prompt=prompt_list,
            dtype=text_encoder.dtype,
            device=resolved_device,
            max_sequence_length=max_sequence_length,
            pad_multiple=pad_multiple,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        results: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {
            "prompt_ids": prompt_ids,
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
        }

        if do_classifier_free_guidance:
            negative_prompt = "" if negative_prompt is None else negative_prompt
            neg_list = (
                [negative_prompt] if isinstance(negative_prompt, str) else list(negative_prompt)
            )
            if len(neg_list) == 1 and len(prompt_list) > 1:
                neg_list = neg_list * len(prompt_list)
            if len(neg_list) != len(prompt_list):
                raise ValueError(
                    f"The number of negative prompts ({len(neg_list)}) must match "
                    f"the number of prompts ({len(prompt_list)})."
                )
            neg_ids, neg_embeds, neg_text_ids = self._get_qwen3_vl_prompt_embeds(
                text_encoder=text_encoder,
                processor=processor,
                prompt=neg_list,
                dtype=text_encoder.dtype,
                device=resolved_device,
                max_sequence_length=max_sequence_length,
                pad_multiple=pad_multiple,
                text_encoder_out_layers=text_encoder_out_layers,
            )
            results.update(
                {
                    "negative_prompt_ids": neg_ids,
                    "negative_prompt_embeds": neg_embeds,
                    "negative_text_ids": neg_text_ids,
                }
            )

        return results

    # ======================== Image Encoding ========================

    def encode_image(
        self,
        images: Union[ImageSingle, ImageBatch, MultiImageBatch],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Union[List[List[torch.Tensor]], List[Optional[torch.Tensor]]]]:
        """Encode reference image(s) into normalized packed FLUX.3 conditioning tokens."""
        del generator
        if self.get_component("vae") is None and self.has_component("vae"):
            self.on_load_components(["vae", "image_processor"], device=device)

        vae = self._require_component("vae")
        resolved_device = vae.device if device is None else device
        transformer = (
            self.get_component("transformer") if self.has_component("transformer") else None
        )
        resolved_dtype = (
            dtype if dtype is not None else getattr(transformer, "dtype", None) or vae.dtype
        )

        if not self._is_multi_images_batch(images):
            images = [images]  # type: ignore[list-item]

        standardized_batches = [
            (
                []
                if isinstance(imgs, list) and not imgs
                else self._standardize_image_input(imgs, output_type="pil")
            )
            for imgs in images  # type: ignore[union-attr]
        ]

        condition_image_tensors: List[List[torch.Tensor]] = [
            (
                self._resize_condition_images(
                    condition_images=imgs,
                    condition_image_size=condition_image_size,
                )
                if imgs
                else []
            )
            for imgs in standardized_batches
        ]

        image_latents_list: List[Optional[torch.Tensor]] = []
        image_latent_ids_list: List[Optional[torch.Tensor]] = []
        for cond_img_tensors in condition_image_tensors:
            if not cond_img_tensors:
                image_latents_list.append(None)
                image_latent_ids_list.append(None)
                continue
            image_latents, image_latent_ids = prepare_flux3_condition_latents(
                self,
                cond_img_tensors,
                batch_size=1,
                device=resolved_device,
                dtype=resolved_dtype,
                image_rope_scale=IMAGE_ROPE_SCALE,
            )
            image_latents_list.append(image_latents.squeeze(0))
            image_latent_ids_list.append(image_latent_ids.squeeze(0))

        postprocessed_condition_images: List[List[torch.Tensor]] = [
            [
                self.pipeline.image_processor.postprocess(img, output_type="pt")[0].to(
                    resolved_device
                )
                for img in cond_img_tensors
            ]
            for cond_img_tensors in condition_image_tensors
        ]

        return {
            "condition_images": postprocessed_condition_images,
            "image_latents": image_latents_list,
            "image_latent_ids": image_latent_ids_list,
        }

    def _resize_condition_images(
        self,
        condition_images: Union[Image.Image, List[Image.Image]],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
    ) -> List[torch.Tensor]:
        """Resize reference images when exceeding the area budget and center-crop to 16px grid."""
        if isinstance(condition_images, Image.Image):
            condition_images = [condition_images]

        condition_images = self._standardize_image_input(
            condition_images,
            output_type="pil",
        )

        if isinstance(condition_image_size, int):
            condition_image_size = (condition_image_size, condition_image_size)
        elif not isinstance(condition_image_size, (tuple, list)) or len(condition_image_size) != 2:
            raise ValueError(
                f"expected condition_image_size as int or (height, width), "
                f"received {condition_image_size!r}"
            )
        max_area = int(condition_image_size[0]) * int(condition_image_size[1])
        if max_area <= 0:
            raise ValueError(
                f"expected positive condition_image_size area, received {condition_image_size!r}"
            )

        multiple_of = self.pipeline.vae_scale_factor * 2
        condition_image_tensors: List[torch.Tensor] = []
        for img in condition_images:
            image_width, image_height = img.size
            if image_width * image_height > max_area:
                img = self.pipeline.image_processor._resize_to_target_area(img, max_area)
                image_width, image_height = img.size

            cropped_width = max(multiple_of, (image_width // multiple_of) * multiple_of)
            cropped_height = max(multiple_of, (image_height // multiple_of) * multiple_of)
            tensor_img = self.pipeline.image_processor.preprocess(
                img,
                height=cropped_height,
                width=cropped_width,
                resize_mode="crop",
            )
            condition_image_tensors.append(tensor_img)

        return condition_image_tensors

    @staticmethod
    def _is_multi_images_batch(images: Optional[Union[ImageBatch, MultiImageBatch]]) -> bool:
        return images is not None and is_multi_image_batch(images)

    @staticmethod
    def _is_multi_image_latents(
        image_latents: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]],
    ) -> bool:
        return (
            isinstance(image_latents, list)
            and len(image_latents) > 0
            and all(
                latent is None or (isinstance(latent, torch.Tensor) and latent.ndim == 2)
                for latent in image_latents
            )
        ) or (isinstance(image_latents, torch.Tensor) and image_latents.ndim == 3)

    def _standardize_image_input(
        self,
        images: Union[ImageSingle, ImageBatch],
        output_type: Literal["pil", "pt", "np"] = "pil",
    ) -> ImageBatch:
        """Standardize image input to the requested container format."""
        if isinstance(images, Image.Image):
            images = [images]
        return standardize_image_batch(images, output_type=output_type)

    # ======================== Output Image Encoding & Decoding ========================

    def _output_geometry_multiple(self) -> int:
        """Require the 16-pixel grid imposed by the 8x VAE and 2x2 latent patchification."""
        return self.pipeline.vae_scale_factor * 2

    def _encode_output_images(
        self,
        pixel_values: torch.Tensor,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator],
    ) -> EncodedImageTensor:
        """Sample and pack target images with the FLUX.3 latent recipe."""
        del condition
        return encode_flux3_output_images(
            self,
            pixel_values,
            generator,
            image_rope_scale=IMAGE_ROPE_SCALE,
        )

    def _resolve_latent_grid(
        self,
        latents: torch.Tensor,
        *,
        latent_ids: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Resolve ``(latent_height, latent_width)`` from explicit dimensions or position IDs."""
        scale = self.pipeline.vae_scale_factor * 2
        seq_len = latents.shape[1]
        if height is not None and width is not None:
            latent_height = int(height) // scale
            latent_width = int(width) // scale
            if latent_height * latent_width != seq_len:
                raise ValueError(
                    f"FLUX.3 latents has {seq_len} tokens, which does not match "
                    f"height x width = {height} x {width} ({latent_height * latent_width} tokens)"
                )
            return latent_height, latent_width

        if latent_ids is not None:
            ids_2d = self._as_2d_position_ids(latent_ids, "latent_ids")
            max_h = float(ids_2d[:, 1].max().item())
            max_w = float(ids_2d[:, 2].max().item())
            latent_height = int(round(max_h / IMAGE_ROPE_SCALE)) + 1
            latent_width = int(round(max_w / IMAGE_ROPE_SCALE)) + 1
            if latent_height * latent_width == seq_len:
                return latent_height, latent_width

        side = int(math.isqrt(seq_len))
        if side * side == seq_len:
            return side, side
        raise ValueError(
            f"Cannot infer FLUX.3 2D latent grid for seq_len={seq_len}; "
            "pass explicit height/width or valid latent_ids."
        )

    def decode_latents(
        self,
        latents: torch.Tensor,
        latent_ids: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        output_type: Literal["pil", "pt", "np"] = "pil",
    ) -> Union[List[Image.Image], torch.Tensor, np.ndarray]:
        """Unpack, BatchNorm-denormalize, unpatchify, and decode FLUX.3 image latents."""
        if self.get_component("vae") is None and self.has_component("vae"):
            self.on_load_components(["vae", "image_processor"], device=latents.device)

        vae = self._require_component("vae")
        latent_height, latent_width = self._resolve_latent_grid(
            latents,
            latent_ids=latent_ids,
            height=height,
            width=width,
        )
        unpacked = unpack_flux3_latents(latents, latent_height, latent_width)

        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        latents_bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(unpacked.device, unpacked.dtype)
        unpacked = unpacked * latents_bn_std + latents_bn_mean
        unpatched = unpatchify_flux2_latents(unpacked)

        decoded = vae.decode(unpatched.to(device=vae.device, dtype=vae.dtype), return_dict=False)[0]
        return self.pipeline.image_processor.postprocess(decoded, output_type=output_type)

    # ======================== Inference ========================

    @staticmethod
    def _prepare_latents(
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        *,
        vae_scale_factor: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw initial float32 noise on the 2D grid, cast, pack to tokens, and build RoPE IDs."""
        image_scale_factor = vae_scale_factor * 2
        if height % image_scale_factor != 0 or width % image_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {image_scale_factor}, "
                f"received {height} and {width}."
            )
        latent_height = height // image_scale_factor
        latent_width = width // image_scale_factor
        if latents is None:
            shape = (batch_size, num_channels_latents, latent_height, latent_width)
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"Passed a list of {len(generator)} generators for batch_size={batch_size}."
                )
            latents = randn_tensor(
                shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            ).to(dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        if latents.ndim == 4:
            latents = pack_flux3_latents(latents)
        latent_ids = (
            prepare_flux3_image_ids(
                latent_height,
                latent_width,
                image_rope_scale=IMAGE_ROPE_SCALE,
                l_coord=0,
                device=device,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        return latents, latent_ids

    @torch.no_grad()
    def _inference(
        self,
        images: Optional[Union[ImageBatch, MultiImageBatch]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_text_ids: Optional[torch.Tensor] = None,
        condition_images: Optional[MultiImageBatch] = None,
        image_latents: Optional[torch.Tensor] = None,
        image_latent_ids: Optional[torch.Tensor] = None,
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        text_encoder_out_layers: Sequence[int] = DEFAULT_TEXT_ENCODER_OUT_LAYERS,
        pad_multiple: int = TEXT_PAD_MULTIPLE,
        compute_log_prob: bool = False,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
    ) -> List[Flux3Sample]:
        """Run FLUX.3 rollout on a uniform-window T2I batch or a single sample."""
        device = self.device
        transformer = self._require_component("transformer")
        dtype = transformer.dtype
        do_classifier_free_guidance = (not self.is_distilled) and guidance_scale > 1.0
        resolved_attention_kwargs = (
            attention_kwargs if attention_kwargs is not None else joint_attention_kwargs
        )

        # 1. Encode prompt
        if isinstance(prompt, str):
            prompt = [prompt]

        if (prompt_embeds is None or prompt_ids is None or text_ids is None) or (
            do_classifier_free_guidance
            and (
                negative_prompt_embeds is None
                or negative_prompt_ids is None
                or negative_text_ids is None
            )
        ):
            prompt_encoding = self.encode_prompt(
                prompt=prompt or [""],
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                device=device,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
                pad_multiple=pad_multiple,
            )
            prompt_ids = prompt_encoding["prompt_ids"]  # type: ignore[assignment]
            prompt_embeds = prompt_encoding["prompt_embeds"]  # type: ignore[assignment]
            text_ids = prompt_encoding["text_ids"]  # type: ignore[assignment]
            if do_classifier_free_guidance:
                negative_prompt_ids = prompt_encoding["negative_prompt_ids"]  # type: ignore[assignment]
                negative_prompt_embeds = prompt_encoding["negative_prompt_embeds"]  # type: ignore[assignment]
                negative_text_ids = prompt_encoding["negative_text_ids"]  # type: ignore[assignment]
        else:
            prompt_ids = prompt_ids.to(device)
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            text_ids = text_ids.to(device)
            if negative_prompt_ids is not None:
                negative_prompt_ids = negative_prompt_ids.to(device)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
            if negative_text_ids is not None:
                negative_text_ids = negative_text_ids.to(device)

        if (
            not isinstance(prompt_embeds, torch.Tensor)
            or not isinstance(prompt_ids, torch.Tensor)
            or not isinstance(text_ids, torch.Tensor)
        ):
            raise ValueError(
                "Flux3Adapter._inference requires uniform stacked prompt tensors; "
                "use Flux3Adapter.inference() for variable-window batches."
            )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        if isinstance(negative_prompt_embeds, torch.Tensor):
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        batch_size = prompt_embeds.shape[0]

        # 2. Encode condition images
        images = (
            [images] if images is not None and not self._is_multi_images_batch(images) else images
        )
        if images is not None and (
            condition_images is None or image_latents is None or image_latent_ids is None
        ):
            image_encoding = self.encode_image(
                images=images,
                condition_image_size=condition_image_size,
                device=device,
                dtype=dtype,
                generator=generator[0] if isinstance(generator, list) else generator,
            )
            condition_images = image_encoding["condition_images"]  # type: ignore[assignment]
            raw_latents = image_encoding["image_latents"]
            raw_ids = image_encoding["image_latent_ids"]
            if any(item is None for item in raw_latents):
                image_latents = None
                image_latent_ids = None
            else:
                image_latents = torch.stack(
                    [item for item in raw_latents if item is not None], dim=0
                ).to(device=device, dtype=dtype)
                image_latent_ids = torch.stack(
                    [item for item in raw_ids if item is not None], dim=0
                ).to(device=device)
        else:
            image_latents = (
                image_latents.to(device=device, dtype=dtype) if image_latents is not None else None
            )
            image_latent_ids = (
                image_latent_ids.to(device=device) if image_latent_ids is not None else None
            )

        # 3. Prepare initial latents
        num_channels_latents = int(getattr(transformer.config, "in_channels", 128))
        latents, latent_ids = self._prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            vae_scale_factor=self.pipeline.vae_scale_factor,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # 4. Prepare timesteps (constant shift 6.93 on the scheduler)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            seq_len=latents.shape[1],
            device=device,
        )

        # 5. Denoising loop
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
            t_next = (
                timesteps[i + 1]
                if i + 1 < len(timesteps)
                else torch.tensor(0, device=device, dtype=timesteps.dtype)
            )
            return_kwargs = list(
                set(["next_latents", "log_prob", "velocity"] + extra_call_back_kwargs)
            )
            current_compute_log_prob = compute_log_prob and current_noise_level > 0

            output = self._forward(
                t=t,
                t_next=t_next,
                latents=latents,
                latent_ids=latent_ids,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                image_latents=image_latents,
                image_latent_ids=image_latent_ids,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
                guidance_scale=guidance_scale,
                attention_kwargs=resolved_attention_kwargs,
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

        # 6. Decode latents to images
        decoded_images = self.decode_latents(
            latents,
            latent_ids=latent_ids,
            height=height,
            width=width,
            output_type="pt",
        )

        # 7. Assemble Flux3Sample instances
        extra_call_back_res = callback_collector.get_result()
        callback_index_map = callback_collector.get_index_map()
        all_latents = latent_collector.get_result()
        latent_index_map = latent_collector.get_index_map()
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None

        samples = [
            Flux3Sample(
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
                height=height,
                width=width,
                image=decoded_images[b],
                latent_ids=latent_ids[b],
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b],
                prompt_embeds=prompt_embeds[b],
                text_ids=text_ids[b],
                negative_prompt=(
                    negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt
                ),
                negative_prompt_ids=(
                    negative_prompt_ids[b] if negative_prompt_ids is not None else None
                ),
                negative_prompt_embeds=(
                    negative_prompt_embeds[b] if negative_prompt_embeds is not None else None
                ),
                negative_text_ids=(negative_text_ids[b] if negative_text_ids is not None else None),
                condition_images=condition_images[b] if condition_images is not None else None,
                image_latents=image_latents[b] if image_latents is not None else None,
                image_latent_ids=image_latent_ids[b] if image_latent_ids is not None else None,
                extra_kwargs={
                    **{k: v[b] for k, v in extra_call_back_res.items()},
                    "callback_index_map": callback_index_map,
                },
            )
            for b in range(batch_size)
        ]

        if hasattr(self.pipeline, "maybe_free_model_hooks"):
            self.pipeline.maybe_free_model_hooks()

        return samples

    @torch.no_grad()
    def inference(
        self,
        images: Optional[MultiImageBatch] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        text_encoder_out_layers: Sequence[int] = DEFAULT_TEXT_ENCODER_OUT_LAYERS,
        pad_multiple: int = TEXT_PAD_MULTIPLE,
        prompt_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        text_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_prompt_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_text_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        condition_images: Optional[MultiImageBatch] = None,
        image_latents: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
        image_latent_ids: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
        compute_log_prob: bool = False,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
    ) -> List[Flux3Sample]:
        """Run FLUX.3 inference; falls back to per-sample execution for I2I or ragged windows."""
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * (len(prompt) if prompt is not None else 1)

        is_nested_images = self._is_multi_images_batch(images)
        is_nested_image_latents = self._is_multi_image_latents(image_latents)
        is_ragged_prompts = isinstance(prompt_embeds, list) or isinstance(
            negative_prompt_embeds, list
        )

        # If raw prompts are passed with batch_size > 1 and no pre-encoded embeds, check whether
        # their 80-token windows differ by encoding once upfront.
        if (
            prompt is not None
            and len(prompt) > 1
            and prompt_embeds is None
            and not (is_nested_images or is_nested_image_latents)
        ):
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                device=self.device,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
                pad_multiple=pad_multiple,
            )
            prompt_ids = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
            text_ids = encoded["text_ids"]
            negative_prompt_ids = encoded.get("negative_prompt_ids")
            negative_prompt_embeds = encoded.get("negative_prompt_embeds")
            negative_text_ids = encoded.get("negative_text_ids")
            is_ragged_prompts = isinstance(prompt_embeds, list) or isinstance(
                negative_prompt_embeds, list
            )

        fall_back = is_nested_images or is_nested_image_latents or is_ragged_prompts
        if not fall_back:
            return self._inference(
                images=images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                condition_image_size=condition_image_size,
                attention_kwargs=attention_kwargs,
                joint_attention_kwargs=joint_attention_kwargs,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
                pad_multiple=pad_multiple,
                prompt_ids=prompt_ids,  # type: ignore[arg-type]
                prompt_embeds=prompt_embeds,  # type: ignore[arg-type]
                text_ids=text_ids,  # type: ignore[arg-type]
                negative_prompt_ids=negative_prompt_ids,  # type: ignore[arg-type]
                negative_prompt_embeds=negative_prompt_embeds,  # type: ignore[arg-type]
                negative_text_ids=negative_text_ids,  # type: ignore[arg-type]
                condition_images=condition_images,
                image_latents=image_latents,  # type: ignore[arg-type]
                image_latent_ids=image_latent_ids,  # type: ignore[arg-type]
                compute_log_prob=compute_log_prob,
                extra_call_back_kwargs=extra_call_back_kwargs,
                trajectory_indices=trajectory_indices,
            )

        if not self._has_warned_inference_fallback:
            logger.warning(
                "FLUX.3 per-sample inference fallback activated (image conditioning or "
                "variable 80-token prompt windows). This warning will only appear once."
            )
            self._has_warned_inference_fallback = True

        if images is not None:
            batch_size = len(images)
        elif image_latents is not None:
            batch_size = len(image_latents)
        elif prompt_embeds is not None:
            batch_size = len(prompt_embeds)
        elif prompt is not None:
            batch_size = len(prompt)
        else:
            raise ValueError("Flux3Adapter.inference requires prompt, prompt_embeds, or images.")

        samples: List[Flux3Sample] = []
        for idx in range(batch_size):
            this_prompt = [prompt[idx]] if prompt is not None else None
            this_neg_prompt = (
                [negative_prompt[idx]] if isinstance(negative_prompt, list) else negative_prompt
            )
            this_prompt_ids = (
                prompt_ids[idx].unsqueeze(0)
                if prompt_ids is not None and prompt_ids[idx] is not None
                else None
            )
            this_prompt_embeds = (
                prompt_embeds[idx].unsqueeze(0)
                if prompt_embeds is not None and prompt_embeds[idx] is not None
                else None
            )
            this_text_ids = (
                text_ids[idx].unsqueeze(0)
                if text_ids is not None and text_ids[idx] is not None
                else None
            )
            this_neg_prompt_ids = (
                negative_prompt_ids[idx].unsqueeze(0)
                if negative_prompt_ids is not None and negative_prompt_ids[idx] is not None
                else None
            )
            this_neg_prompt_embeds = (
                negative_prompt_embeds[idx].unsqueeze(0)
                if negative_prompt_embeds is not None and negative_prompt_embeds[idx] is not None
                else None
            )
            this_neg_text_ids = (
                negative_text_ids[idx].unsqueeze(0)
                if negative_text_ids is not None and negative_text_ids[idx] is not None
                else None
            )
            this_images = images[idx] if images is not None else None
            this_condition_images = (
                condition_images[idx : idx + 1] if condition_images is not None else None
            )
            this_image_latents = (
                image_latents[idx].unsqueeze(0)
                if image_latents is not None and image_latents[idx] is not None
                else None
            )
            this_image_latent_ids = (
                image_latent_ids[idx].unsqueeze(0)
                if image_latent_ids is not None and image_latent_ids[idx] is not None
                else None
            )
            sample = self._inference(
                images=this_images,
                prompt=this_prompt,
                negative_prompt=this_neg_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator[idx] if isinstance(generator, list) else generator,
                condition_image_size=condition_image_size,
                attention_kwargs=attention_kwargs,
                joint_attention_kwargs=joint_attention_kwargs,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
                pad_multiple=pad_multiple,
                prompt_ids=this_prompt_ids,
                prompt_embeds=this_prompt_embeds,
                text_ids=this_text_ids,
                negative_prompt_ids=this_neg_prompt_ids,
                negative_prompt_embeds=this_neg_prompt_embeds,
                negative_text_ids=this_neg_text_ids,
                condition_images=this_condition_images,
                image_latents=this_image_latents,
                image_latent_ids=this_image_latent_ids,
                compute_log_prob=compute_log_prob,
                extra_call_back_kwargs=extra_call_back_kwargs,
                trajectory_indices=trajectory_indices,
            )
            samples.extend(sample)

        return samples

    # ======================== Forward ========================

    @staticmethod
    def _as_2d_position_ids(ids: torch.Tensor, name: str) -> torch.Tensor:
        """Normalize ``(seq_len, 4)`` or batched ``(B, seq_len, 4)`` position IDs to 2D."""
        if not isinstance(ids, torch.Tensor):
            raise TypeError(f"expected {name} to be torch.Tensor, received {type(ids).__name__}")
        if ids.ndim == 2 and ids.shape[-1] == 4:
            return ids
        if ids.ndim == 3 and ids.shape[-1] == 4:
            return ids[0]
        raise ValueError(
            f"expected {name} to have shape (seq_len, 4) or (B, seq_len, 4), "
            f"received {tuple(ids.shape)}"
        )

    def _forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_latents: Optional[torch.Tensor] = None,
        image_latent_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_text_ids: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        noise_level: Optional[float] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: Sequence[str] = (
            "velocity",
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
        ),
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """Core single-step FLUX.3 forward handling T2I, I2I, CFG, and step-distilled flow maps."""
        batch_size = latents.shape[0]
        transformer = self.transformer
        transformer_dtype = getattr(transformer, "dtype", latents.dtype)
        resolved_attention_kwargs = (
            attention_kwargs if attention_kwargs is not None else joint_attention_kwargs
        )

        is_distilled = self.is_distilled
        if not is_distilled and guidance_scale > 1.0 and negative_prompt_embeds is None:
            logger.warning(
                "Passed `guidance_scale=%s` > 1.0 to Flux3Adapter, but `negative_prompt_embeds` "
                "is None. Falling back to no classifier-free guidance.",
                guidance_scale,
            )
        do_classifier_free_guidance = (
            not is_distilled and guidance_scale > 1.0 and negative_prompt_embeds is not None
        )

        # 1. Prepare inputs in transformer compute dtype and 2D position IDs
        latent_model_input = latents.to(dtype=transformer_dtype)
        cond_prompt_embeds = prompt_embeds.to(dtype=transformer_dtype)
        img_ids_2d = self._as_2d_position_ids(latent_ids, "latent_ids").to(latents.device)
        txt_ids_2d = self._as_2d_position_ids(text_ids, "text_ids").to(latents.device)

        image_cond_hidden_states: Optional[torch.Tensor] = None
        image_cond_ids_2d: Optional[torch.Tensor] = None
        if image_latents is not None:
            if image_latent_ids is None:
                raise ValueError("`image_latents` requires `image_latent_ids` in Flux3Adapter.")
            image_cond_hidden_states = image_latents.to(
                device=latents.device, dtype=transformer_dtype
            )
            image_cond_ids_2d = self._as_2d_position_ids(image_latent_ids, "image_latent_ids").to(
                latents.device
            )

        timestep = (t.expand(batch_size) / 1000.0).to(
            device=latents.device, dtype=transformer_dtype
        )
        target_timestep: Optional[torch.Tensor] = None
        if is_distilled:
            target_t = (
                torch.zeros(batch_size, device=latents.device, dtype=t.dtype)
                if t_next is None
                else t_next.expand(batch_size)
            )
            target_timestep = (target_t / 1000.0).to(device=latents.device, dtype=transformer_dtype)

        # 2. Conditional forward pass
        velocity = transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=cond_prompt_embeds,
            timestep=timestep,
            img_ids=img_ids_2d,
            txt_ids=txt_ids_2d,
            image_cond_hidden_states=image_cond_hidden_states,
            image_cond_ids=image_cond_ids_2d,
            target_timestep=target_timestep,
            attention_kwargs=resolved_attention_kwargs,
            return_dict=False,
        )[0]

        # 3. Unconditional forward pass for true CFG
        if do_classifier_free_guidance:
            uncond_prompt_embeds = negative_prompt_embeds.to(dtype=transformer_dtype)
            uncond_txt_ids_2d = (
                self._as_2d_position_ids(negative_text_ids, "negative_text_ids").to(latents.device)
                if negative_text_ids is not None
                else self._prepare_text_ids(uncond_prompt_embeds.shape[1], device=latents.device)
            )
            neg_velocity = transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=uncond_prompt_embeds,
                timestep=timestep,
                img_ids=img_ids_2d,
                txt_ids=uncond_txt_ids_2d,
                image_cond_hidden_states=image_cond_hidden_states,
                image_cond_ids=image_cond_ids_2d,
                target_timestep=target_timestep,
                attention_kwargs=resolved_attention_kwargs,
                return_dict=False,
            )[0]
            velocity = neg_velocity + guidance_scale * (velocity - neg_velocity)

        # 4. Scheduler step
        output = self.scheduler.step(
            velocity=velocity,
            timestep=t,
            latents=latents,
            timestep_next=t_next,
            next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            return_kwargs=list(return_kwargs),
            noise_level=noise_level,
        )
        return output

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: Union[torch.Tensor, List[torch.Tensor]],
        prompt_embeds: Union[torch.Tensor, List[torch.Tensor]],
        text_ids: Union[torch.Tensor, List[torch.Tensor]],
        image_latents: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
        image_latent_ids: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
        negative_prompt_embeds: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
        negative_text_ids: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
        guidance_scale: float = 3.5,
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        noise_level: Optional[float] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: Sequence[str] = (
            "velocity",
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
        ),
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """General forward pass supporting T2I, multi-reference I2I, and ragged prompt windows."""
        has_image_cond = image_latents is not None and (
            isinstance(image_latents, torch.Tensor)
            or any(item is not None for item in image_latents)
        )
        has_ragged_prompts = (
            isinstance(prompt_embeds, list)
            or isinstance(text_ids, list)
            or isinstance(negative_prompt_embeds, list)
            or isinstance(negative_text_ids, list)
        )
        fall_back = has_image_cond or has_ragged_prompts

        if not fall_back:
            return self._forward(
                t=t,
                latents=latents,
                latent_ids=latent_ids,  # type: ignore[arg-type]
                prompt_embeds=prompt_embeds,  # type: ignore[arg-type]
                text_ids=text_ids,  # type: ignore[arg-type]
                image_latents=None,
                image_latent_ids=None,
                negative_prompt_embeds=negative_prompt_embeds,  # type: ignore[arg-type]
                negative_text_ids=negative_text_ids,  # type: ignore[arg-type]
                guidance_scale=guidance_scale,
                t_next=t_next,
                next_latents=next_latents,
                noise_level=noise_level,
                attention_kwargs=attention_kwargs,
                joint_attention_kwargs=joint_attention_kwargs,
                compute_log_prob=compute_log_prob,
                return_kwargs=return_kwargs,
            )

        if not self._has_warned_forward_fallback:
            logger.warning(
                "FLUX.3: Image conditioning or variable prompt window detected in forward(). "
                "Processing samples individually (warning shown once)."
            )
            self._has_warned_forward_fallback = True

        batch_size = latents.shape[0]
        outputs: List[FlowMatchEulerDiscreteSDESchedulerOutput] = []

        for idx in range(batch_size):
            single_t = t[idx].unsqueeze(0) if t.ndim > 0 and t.shape[0] == batch_size else t
            single_t_next = (
                t_next[idx].unsqueeze(0)
                if t_next is not None and t_next.ndim > 0 and t_next.shape[0] == batch_size
                else t_next
            )
            single_latents = latents[idx].unsqueeze(0)
            single_latent_ids = (
                latent_ids[idx].unsqueeze(0)
                if isinstance(latent_ids, list) or latent_ids.ndim == 3
                else latent_ids
            )
            single_next_latents = (
                next_latents[idx].unsqueeze(0) if next_latents is not None else None
            )
            single_prompt_embeds = prompt_embeds[idx].unsqueeze(0)
            single_text_ids = (
                text_ids[idx].unsqueeze(0)
                if isinstance(text_ids, list) or text_ids.ndim == 3
                else text_ids
            )
            single_image_latents = (
                image_latents[idx].unsqueeze(0)
                if image_latents is not None and image_latents[idx] is not None
                else None
            )
            single_image_latent_ids = (
                image_latent_ids[idx].unsqueeze(0)
                if image_latent_ids is not None and image_latent_ids[idx] is not None
                else None
            )
            single_neg_prompt_embeds = (
                negative_prompt_embeds[idx].unsqueeze(0)
                if negative_prompt_embeds is not None and negative_prompt_embeds[idx] is not None
                else None
            )
            single_neg_text_ids = (
                negative_text_ids[idx].unsqueeze(0)
                if negative_text_ids is not None and negative_text_ids[idx] is not None
                else None
            )

            out = self._forward(
                t=single_t,
                latents=single_latents,
                latent_ids=single_latent_ids,
                prompt_embeds=single_prompt_embeds,
                text_ids=single_text_ids,
                image_latents=single_image_latents,
                image_latent_ids=single_image_latent_ids,
                negative_prompt_embeds=single_neg_prompt_embeds,
                negative_text_ids=single_neg_text_ids,
                guidance_scale=guidance_scale,
                t_next=single_t_next,
                next_latents=single_next_latents,
                noise_level=noise_level,
                attention_kwargs=attention_kwargs,
                joint_attention_kwargs=joint_attention_kwargs,
                compute_log_prob=compute_log_prob,
                return_kwargs=return_kwargs,
            )
            outputs.append(out)

        outputs_dict = [o.to_dict() for o in outputs]
        return FlowMatchEulerDiscreteSDESchedulerOutput.from_dict(
            {
                key: (
                    torch.cat([item[key] for item in outputs_dict], dim=0)
                    if outputs_dict[0][key] is not None
                    else None
                )
                for key in outputs_dict[0].keys()
            }
        )
