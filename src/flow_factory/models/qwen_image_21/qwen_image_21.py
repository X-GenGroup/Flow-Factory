# Copyright 2026 Jayce-Ping
# Copyright 2026 Qwen-Image Team, The HuggingFace Team.
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

"""Flow-Factory adapter for Qwen-Image 2.1.

The adapter intentionally owns an independent copy of the Qwen-Image 2.1
conditioning and latent-layout logic. It does not import or inherit from the
Qwen-Image 1.x adapters because the two generations use incompatible prompt,
packing, CFG, and transformer contracts.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import torch
from accelerate import Accelerator
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

import diffusers
from diffusers.utils.torch_utils import randn_tensor

try:
    from diffusers import QwenImage21Pipeline

    _QWEN_IMAGE_21_IMPORT_ERROR: Optional[Exception] = None
except (ImportError, RuntimeError) as exc:  # pragma: no cover - exercised with released diffusers
    QwenImage21Pipeline = Any  # type: ignore[misc,assignment]
    _QWEN_IMAGE_21_IMPORT_ERROR = exc

from ...contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaOrder,
    NegativePromptPolicy,
)
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
from ..configured_image_output import ConfiguredImageOutputAdapterMixin, EncodedImageTensor
from ..pipeline_contracts import image_output_contract
from .output_codec import (
    QWEN_IMAGE_21_DIFFUSERS_COMMIT,
    encode_qwen_image_21_output,
    encode_qwen_image_21_vae,
)

logger = setup_logger(__name__)

QWEN_IMAGE_21_INSTALL = "git submodule update --init && pip install -e ./diffusers"
DEFAULT_CONDITION_IMAGE_SIZE = (1024, 1024)


@dataclass
class QwenImage21Sample(I2ISample):
    """Per-sample Qwen-Image 2.1 rollout state."""

    condition_images_as_pil: ClassVar[bool] = True
    _shared_fields: ClassVar[frozenset[str]] = frozenset({"height", "width"})

    prompt_embeds_mask: Optional[torch.Tensor] = None
    image_pad_mask: Optional[torch.Tensor] = None
    negative_prompt_embeds_mask: Optional[torch.Tensor] = None
    negative_image_pad_mask: Optional[torch.Tensor] = None
    condition_image_latents: Optional[torch.Tensor] = None
    condition_img_shapes: Optional[List[Tuple[int, int, int]]] = None
    img_shapes: Optional[List[Tuple[int, int, int]]] = None


# Modified from diffusers.pipelines.qwenimage21.pipeline_qwenimage21.calculate_dimensions
# at QWEN_IMAGE_21_DIFFUSERS_COMMIT. The third, unused return value is removed.
def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int]:
    """Return 32-aligned width/height preserving ``width / height``."""
    if target_area <= 0 or ratio <= 0:
        raise ValueError(
            f"target_area and ratio must be positive, received {target_area=} and {ratio=}"
        )
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


class QwenImage21Adapter(ConfiguredImageOutputAdapterMixin, BaseAdapter):
    """Adapter for unified Qwen-Image 2.1 text-to-image and image editing."""

    supports_diffusers_cache = False
    python_format_columns: ClassVar[frozenset[str]] = frozenset({"condition_images"})
    offline_training_forward_overrides = MappingProxyType({"guidance_scale": 1.0})
    pipeline_io_contract = image_output_contract(
        negative_prompt=NegativePromptPolicy.OPTIONAL,
        input_image_min_count=0,
        input_image_max_count=None,
        input_order=InputMediaOrder.WITHIN_TYPE,
        geometry_source=GeometrySource.CONFIGURED,
        batch_capability=BatchCapability.SINGLE_SAMPLE,
    )

    def __init__(self, config: Arguments, accelerator: Accelerator):
        if _QWEN_IMAGE_21_IMPORT_ERROR is not None:
            raise ImportError(
                "QwenImage21Adapter requires the bundled Diffusers submodule at "
                f"{QWEN_IMAGE_21_DIFFUSERS_COMMIT}. Install it with `{QWEN_IMAGE_21_INSTALL}`. "
                f"The active diffusers version is {getattr(diffusers, '__version__', 'unknown')}."
            ) from _QWEN_IMAGE_21_IMPORT_ERROR
        super().__init__(config, accelerator)
        self.pipeline: QwenImage21Pipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler
        self._warned_cfg_without_negative = False
        self._warned_negative_without_cfg = False

    def load_pipeline(self) -> QwenImage21Pipeline:
        """Load the official eager pipeline for the classic component runtime."""
        return self._load_diffusers_pipeline(
            QwenImage21Pipeline,
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False,
        )

    @property
    def tokenizer(self) -> Any:
        """Expose the processor-owned tokenizer to distributed reward decoding."""
        return self.pipeline.processor.tokenizer

    @property
    def preprocessing_modules(self) -> List[str]:
        return ["text_encoders", "vae"]

    @property
    def inference_modules(self) -> List[str]:
        return ["transformer", "vae"]

    @property
    def default_target_modules(self) -> List[str]:
        """Target all projections inside each repeated single-stream block."""
        return [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "img_mlp.proj",
            "img_mlp.gate_layer",
            "img_mlp.out",
        ]

    # ============================== Prompt encoding ==============================

    @staticmethod
    def _extract_masked_hidden(
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return list(torch.split(selected, valid_lengths.tolist(), dim=0))

    @staticmethod
    def _pad_token_sequences(
        values: Sequence[torch.Tensor],
        *,
        padding_value: int,
    ) -> torch.Tensor:
        return pad_sequence(list(values), batch_first=True, padding_value=padding_value)

    def _encode_qwen_image_21_condition(
        self,
        prompt: Union[str, List[str]],
        images: Optional[List[Image.Image]] = None,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Jointly encode prompt/images and return IDs without a second tokenize pass.

        Modified from
        ``QwenImage21Pipeline._get_qwen_prompt_embeds`` at
        :data:`QWEN_IMAGE_21_DIFFUSERS_COMMIT`. Changes are limited to routing
        the text encoder through the adapter and returning the already-tokenized
        post-system ``prompt_ids`` required by Flow-Factory.
        """
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.transformer.dtype
        prompt = [prompt] if isinstance(prompt, str) else list(prompt)
        prompt = [" " if not value else value for value in prompt]
        is_t2i = not images

        if is_t2i:
            formatted_prompts = [self.pipeline.prompt_template_t2i.format(text) for text in prompt]
            processor_images = None
        else:
            formatted_prompts = []
            image_marker = "<image1><|vision_start|><|image_pad|><|vision_end|>"
            replacement = image_marker
            for index in range(2, len(images) + 1):
                replacement += f" <image{index}><|vision_start|><|image_pad|><|vision_end|>"
            template = self.pipeline.prompt_template_ti2i.replace(image_marker, replacement)
            formatted_prompts = [template.format(text) for text in prompt]

            processor_images = []
            for _ in prompt:
                for image in images:
                    vlm_image = image
                    if vlm_image.mode == "RGBA":
                        white = Image.new("RGB", vlm_image.size, (255, 255, 255))
                        white.paste(vlm_image, mask=vlm_image.getchannel("A"))
                        vlm_image = white
                    processor_images.append(vlm_image)

        processor_kwargs: Dict[str, Any] = {
            "text": formatted_prompts,
            "padding": True,
            "padding_side": "left",
            "return_tensors": "pt",
        }
        if processor_images is not None:
            processor_kwargs["images"] = processor_images
        model_inputs = self.pipeline.processor(**processor_kwargs).to(device)

        forward_kwargs: Dict[str, Any] = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": True,
        }
        if not is_t2i and hasattr(model_inputs, "pixel_values"):
            forward_kwargs.update(
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
            )
        if hasattr(model_inputs, "mm_token_type_ids"):
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

        # Qwen-Image 2.1 was trained on the last decoder block output before
        # final RMSNorm. Transformers 5 aliases hidden_states[-1] to the
        # normalized output, so neutralize that norm for this call exactly as
        # the pinned Diffusers pipeline does.
        text_encoder = self.text_encoder
        text_model = getattr(text_encoder.model, "language_model", text_encoder.model)
        handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            outputs = text_encoder(**forward_kwargs)
        finally:
            handle.remove()

        split_hidden = self._extract_masked_hidden(
            outputs.hidden_states[-1],
            model_inputs.attention_mask,
        )
        valid_ids = [
            ids[mask.bool()]
            for ids, mask in zip(model_inputs.input_ids, model_inputs.attention_mask)
        ]
        drop_idx = self.pipeline._drop_idx
        split_hidden = [value[drop_idx:] for value in split_hidden]
        valid_ids = [value[drop_idx:] for value in valid_ids]
        image_pad_masks = [value.eq(self.pipeline._img_token_id) for value in valid_ids]

        prompt_embeds = pad_sequence(split_hidden, batch_first=True, padding_value=0.0).to(
            device=device,
            dtype=dtype,
        )
        prompt_embeds_mask = pad_sequence(
            [torch.ones(value.shape[0], dtype=torch.long, device=device) for value in split_hidden],
            batch_first=True,
            padding_value=0,
        )
        image_pad_mask = pad_sequence(
            image_pad_masks,
            batch_first=True,
            padding_value=False,
        )
        tokenizer = self.pipeline.processor.tokenizer
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        prompt_ids = self._pad_token_sequences(valid_ids, padding_value=pad_token_id)

        return {
            "prompt_ids": prompt_ids,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "image_pad_mask": image_pad_mask,
        }

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 1.0,
        images: Optional[List[Image.Image]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode positive and optional true-CFG conditions independently."""
        prompt_batch = [prompt] if isinstance(prompt, str) else list(prompt)
        encoded = self._encode_qwen_image_21_condition(
            prompt_batch,
            images,
            device=device,
            dtype=dtype,
        )
        if guidance_scale <= 1.0 or negative_prompt is None:
            return encoded

        if isinstance(negative_prompt, str):
            negative_batch = [negative_prompt] * len(prompt_batch)
        else:
            negative_batch = list(negative_prompt)
        if len(negative_batch) != len(prompt_batch):
            raise ValueError(
                "negative_prompt batch size must match prompt batch size: "
                f"{len(negative_batch)} != {len(prompt_batch)}"
            )
        negative = self._encode_qwen_image_21_condition(
            negative_batch,
            images,
            device=device,
            dtype=dtype,
        )
        encoded.update(
            {
                "negative_prompt_ids": negative["prompt_ids"],
                "negative_prompt_embeds": negative["prompt_embeds"],
                "negative_prompt_embeds_mask": negative["prompt_embeds_mask"],
                "negative_image_pad_mask": negative["image_pad_mask"],
            }
        )
        return encoded

    # ============================== Image and latent encoding ==============================

    # Copied from diffusers.pipelines.qwenimage21.pipeline_qwenimage21.
    # QwenImage21Pipeline._pack_latents at QWEN_IMAGE_21_DIFFUSERS_COMMIT.
    @staticmethod
    def _pack_latents(
        latents: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Plain spatial flatten used by Qwen-Image 2.1.

        Copied from ``QwenImage21Pipeline._pack_latents`` at the pinned
        Diffusers commit.
        """
        return latents.view(batch_size, num_channels_latents, height * width).transpose(1, 2)

    # Copied from diffusers.pipelines.qwenimage21.pipeline_qwenimage21.
    # QwenImage21Pipeline._unpack_latents at QWEN_IMAGE_21_DIFFUSERS_COMMIT.
    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> torch.Tensor:
        """Invert :meth:`_pack_latents` using the official target geometry."""
        batch_size, _, channels = latents.shape
        latent_height = 2 * (int(height) // (vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (vae_scale_factor * 2))
        return latents.transpose(1, 2).reshape(
            batch_size,
            channels,
            1,
            latent_height,
            latent_width,
        )

    @staticmethod
    def _standardize_condition_images(images: Union[ImageSingle, ImageBatch]) -> List[Image.Image]:
        if isinstance(images, Image.Image):
            images = [images]
        standardized = standardize_image_batch(images, output_type="pil")
        return [image.convert("RGBA") if image.mode != "RGBA" else image for image in standardized]

    def _prepare_condition_images(
        self,
        images: Optional[Union[ImageSingle, ImageBatch]],
        *,
        condition_image_size: Union[int, Tuple[int, int]] = DEFAULT_CONDITION_IMAGE_SIZE,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Resize one ordered condition set for both Qwen3-VL and the VAE.

        Modified from the condition-image preprocessing and ``prepare_latents``
        branches in ``QwenImage21Pipeline.__call__`` at the pinned Diffusers
        commit. The packed posterior-mode latents are exposed for replay.
        """
        if images is None or (isinstance(images, list) and not images):
            return {
                "condition_images": [],
                "condition_image_sizes": [],
                "condition_img_shapes": [],
                "condition_image_latents": None,
            }
        device = device or self.pipeline.vae.device
        dtype = dtype or self.pipeline.vae.dtype
        output_resolution = (
            condition_image_size
            if isinstance(condition_image_size, int)
            else max(condition_image_size)
        )
        rgba_images = self._standardize_condition_images(images)
        resized_images: List[Image.Image] = []
        image_sizes: List[Tuple[int, int]] = []
        packed_latents: List[torch.Tensor] = []
        img_shapes: List[Tuple[int, int, int]] = []

        for image in rgba_images:
            input_width, input_height = calculate_dimensions(
                output_resolution * output_resolution,
                image.size[0] / image.size[1],
            )
            resized = self.pipeline.image_processor.resize(
                image,
                width=input_width,
                height=input_height,
            )
            vae_values = self.pipeline.image_processor.preprocess(
                image,
                width=input_width,
                height=input_height,
            ).unsqueeze(2)
            vae_values = vae_values.to(device=device, dtype=dtype)
            encoded = encode_qwen_image_21_vae(
                self,
                vae_values,
                sample_mode="argmax",
            )
            latent_height, latent_width = encoded.shape[-2:]
            packed_latents.append(
                self._pack_latents(
                    encoded,
                    batch_size=1,
                    num_channels_latents=encoded.shape[1],
                    height=latent_height,
                    width=latent_width,
                )
            )
            resized_images.append(resized)
            image_sizes.append((input_width, input_height))
            img_shapes.append((1, latent_height, latent_width))

        return {
            "condition_images": resized_images,
            "condition_image_sizes": image_sizes,
            "condition_img_shapes": img_shapes,
            "condition_image_latents": torch.cat(packed_latents, dim=1),
        }

    def encode_image(
        self,
        images: Optional[MultiImageBatch],
        condition_image_size: Union[int, Tuple[int, int]] = DEFAULT_CONDITION_IMAGE_SIZE,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **_: Any,
    ) -> Dict[str, List[Any]]:
        """Encode each sample's ordered condition images independently."""
        if images is None:
            image_batches: List[Any] = [[]]
        elif is_multi_image_batch(images):
            image_batches = list(images)
        else:
            image_batches = [images]
        results: Dict[str, List[Any]] = defaultdict(list)
        for image_batch in image_batches:
            encoded = self._prepare_condition_images(
                image_batch,
                condition_image_size=condition_image_size,
                device=device,
                dtype=dtype,
            )
            for key, value in encoded.items():
                results[key].append(value)
        return dict(results)

    def encode_video(self, video: Any) -> None:
        return None

    def prepare_latents(
        self,
        *,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create target noise with the official 2.1 unpatched layout."""
        multiple = self.pipeline.vae_scale_factor * 2
        if height % multiple or width % multiple:
            raise ValueError(
                f"Qwen-Image 2.1 height/width must be divisible by {multiple}, "
                f"received {(height, width)}"
            )
        latent_height = height // self.pipeline.vae_scale_factor
        latent_width = width // self.pipeline.vae_scale_factor
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"generator list length {len(generator)} does not match batch size {batch_size}"
            )
        if latents is None:
            latents = randn_tensor(
                (batch_size, 1, num_channels_latents, latent_height, latent_width),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            return self._pack_latents(
                latents,
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=latent_height,
                width=latent_width,
            )
        return latents.to(device=device, dtype=dtype)

    # ============================== Offline output state ==============================

    def _output_geometry_multiple(self) -> int:
        return self.pipeline.vae_scale_factor * 2

    def _preprocess_output_images(
        self,
        images: List[Image.Image],
        height: int,
        width: int,
    ) -> torch.Tensor:
        rgba_images = [image.convert("RGBA") if image.mode != "RGBA" else image for image in images]
        return self.pipeline.image_processor.preprocess(
            rgba_images,
            height=height,
            width=width,
        )

    def _encode_output_images(
        self,
        pixel_values: torch.Tensor,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator],
    ) -> EncodedImageTensor:
        return encode_qwen_image_21_output(self, pixel_values, condition, generator)

    # ============================== Decode and preprocess ==============================

    def decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: Literal["pil", "pt", "np"] = "pil",
    ) -> Any:
        """Decode packed 64-channel latents through the RGBA VAE."""
        latents = self._unpack_latents(
            latents,
            height,
            width,
            self.pipeline.vae_scale_factor,
        ).to(self.vae.dtype)
        channels = latents.shape[1]
        means = torch.as_tensor(
            self.vae.config.latents_mean,
            device=latents.device,
            dtype=latents.dtype,
        ).reshape(1, channels, 1, 1, 1)
        stds = torch.as_tensor(
            self.vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype,
        ).reshape(1, channels, 1, 1, 1)
        decoded = self.vae.decode(latents * stds + means, return_dict=False)[0][:, :, 0]
        return self.pipeline.image_processor.postprocess(decoded, output_type=output_type)

    def preprocess_func(
        self,
        prompt: List[str],
        images: Optional[MultiImageBatch] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 1.0,
        condition_image_size: Union[int, Tuple[int, int]] = DEFAULT_CONDITION_IMAGE_SIZE,
        device: Optional[torch.device] = None,
        **_: Any,
    ) -> Dict[str, List[Any]]:
        """Cache exact per-sample multimodal conditions for rollout and replay."""
        prompt_batch = [prompt] if isinstance(prompt, str) else list(prompt)
        batch_size = len(prompt_batch)
        if images is None:
            image_batches: List[Any] = [[] for _ in range(batch_size)]
        elif is_multi_image_batch(images):
            image_batches = list(images)
        elif batch_size == 1:
            image_batches = [images]
        elif isinstance(images, list) and len(images) == batch_size:
            image_batches = [[image] for image in images]
        else:
            raise ValueError("Qwen-Image 2.1 images must preserve one ordered list per prompt")
        if len(image_batches) != batch_size:
            raise ValueError(
                f"prompt/image batch mismatch: {batch_size} prompts and {len(image_batches)} image sets"
            )

        results: Dict[str, List[Any]] = defaultdict(list)
        for index, text in enumerate(prompt_batch):
            image_state = self._prepare_condition_images(
                image_batches[index],
                condition_image_size=condition_image_size,
                device=device,
            )
            negative_text = (
                negative_prompt[index] if isinstance(negative_prompt, list) else negative_prompt
            )
            prompt_state = self.encode_prompt(
                text,
                negative_prompt=negative_text,
                guidance_scale=guidance_scale,
                images=image_state["condition_images"] or None,
                device=device,
            )

            results["condition_images"].append(image_state["condition_images"])
            results["condition_image_sizes"].append(image_state["condition_image_sizes"])
            results["condition_img_shapes"].append(image_state["condition_img_shapes"])
            condition_latents = image_state["condition_image_latents"]
            results["condition_image_latents"].append(
                condition_latents[0] if condition_latents is not None else None
            )
            for key, value in prompt_state.items():
                results[key].append(value[0])
        return dict(results)

    # ============================== Model prediction ==============================

    @staticmethod
    def _single_tensor(
        value: Any,
        *,
        index: int,
        batch_size: int,
        unbatched_ndim: int,
        name: str,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        came_from_list = isinstance(value, list)
        if isinstance(value, list):
            if len(value) != batch_size:
                raise ValueError(
                    f"{name} list length {len(value)} does not match batch size {batch_size}"
                )
            value = value[index]
            if value is None:
                return None
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must contain tensors, received {type(value).__name__}")
        value = value.to(device)
        if value.ndim == unbatched_ndim:
            if batch_size != 1 and not came_from_list:
                raise ValueError(
                    f"unbatched {name} is only valid for batch size 1, received {batch_size}"
                )
            return value.unsqueeze(0)
        if value.ndim != unbatched_ndim + 1:
            raise ValueError(
                f"{name} expected rank {unbatched_ndim} or {unbatched_ndim + 1}, "
                f"received shape {tuple(value.shape)}"
            )
        if value.shape[0] != batch_size:
            raise ValueError(f"{name} batch dimension {value.shape[0]} does not match {batch_size}")
        return value[index : index + 1]

    @staticmethod
    def _single_img_shapes(
        img_shapes: Any,
        *,
        index: int,
        batch_size: int,
    ) -> List[Tuple[int, int, int]]:
        if isinstance(img_shapes, torch.Tensor):
            img_shapes = img_shapes.detach().cpu().tolist()
        if (
            batch_size == 1
            and img_shapes
            and len(img_shapes[0]) == 3
            and isinstance(img_shapes[0][0], (int, float))
        ):
            img_shapes = [img_shapes]
        if not isinstance(img_shapes, Sequence) or len(img_shapes) != batch_size:
            raise ValueError(
                f"img_shapes must contain one layout per sample, received {img_shapes!r}"
            )
        shapes = img_shapes[index]
        normalized = []
        for shape in shapes:
            if isinstance(shape, torch.Tensor):
                shape = shape.detach().cpu().tolist()
            if not isinstance(shape, Sequence) or len(shape) != 3:
                raise ValueError(f"invalid Qwen-Image 2.1 image shape {shape!r}")
            normalized.append(tuple(int(value) for value in shape))
        if not normalized:
            raise ValueError("img_shapes must include the target image as the final block")
        return normalized

    def _predict_velocity_one(
        self,
        *,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        image_pad_mask: torch.Tensor,
        img_shapes: List[Tuple[int, int, int]],
        condition_image_latents: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        negative_image_pad_mask: Optional[torch.Tensor],
        guidance_scale: float,
        attention_kwargs: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Run the exact cache-free transformer prediction for one sample."""
        valid_length = int(prompt_embeds_mask.sum().item())
        if valid_length < 1:
            raise ValueError("Qwen-Image 2.1 prompt mask must contain at least one valid token")
        prompt_embeds = prompt_embeds[:, :valid_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :valid_length]
        image_pad_mask = image_pad_mask[:, :valid_length]
        model_prompt_mask = None if prompt_embeds_mask.bool().all() else prompt_embeds_mask

        target_tokens = latents.shape[1]
        if target_tokens % 4:
            raise ValueError(
                "Qwen-Image 2.1 target token count must be divisible by 4, "
                f"received {target_tokens}"
            )
        target_slots = image_pad_mask.new_ones((1, target_tokens // 4))
        model_img_mask = torch.cat([image_pad_mask.bool(), target_slots], dim=1)
        model_input = latents
        if condition_image_latents is not None:
            model_input = torch.cat([condition_image_latents, latents], dim=1)

        timestep = t.reshape(-1)[:1].to(device=latents.device, dtype=latents.dtype)
        velocity = self.transformer(
            hidden_states=model_input,
            timestep=timestep / 1000,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=model_prompt_mask,
            img_shapes=[img_shapes],
            img_mask=model_img_mask,
            attention_kwargs=attention_kwargs,
            kv_cache=None,
            kv_cache_mode=None,
            return_dict=False,
        )[0][:, -target_tokens:]

        has_negative = (
            negative_prompt_embeds is not None
            and negative_prompt_embeds_mask is not None
            and negative_image_pad_mask is not None
        )
        if guidance_scale > 1.0 and not has_negative:
            if not self._warned_cfg_without_negative:
                self._warned_cfg_without_negative = True
                logger.warning(
                    "Qwen-Image 2.1 guidance_scale > 1 requires a negative prompt; "
                    "classifier-free guidance is disabled."
                )
            return velocity
        if guidance_scale <= 1.0 or not has_negative:
            if has_negative and not self._warned_negative_without_cfg:
                self._warned_negative_without_cfg = True
                logger.warning(
                    "Qwen-Image 2.1 negative prompt is ignored because guidance_scale <= 1."
                )
            return velocity

        negative_valid_length = int(negative_prompt_embeds_mask.sum().item())
        if negative_valid_length < 1:
            raise ValueError(
                "Qwen-Image 2.1 negative prompt mask must contain at least one valid token"
            )
        negative_prompt_embeds = negative_prompt_embeds[:, :negative_valid_length]
        negative_prompt_embeds_mask = negative_prompt_embeds_mask[:, :negative_valid_length]
        negative_image_pad_mask = negative_image_pad_mask[:, :negative_valid_length]
        negative_model_prompt_mask = (
            None if negative_prompt_embeds_mask.bool().all() else negative_prompt_embeds_mask
        )
        negative_model_img_mask = torch.cat(
            [negative_image_pad_mask.bool(), target_slots],
            dim=1,
        )
        negative_velocity = self.transformer(
            hidden_states=model_input,
            timestep=timestep / 1000,
            encoder_hidden_states=negative_prompt_embeds,
            encoder_hidden_states_mask=negative_model_prompt_mask,
            img_shapes=[img_shapes],
            img_mask=negative_model_img_mask,
            attention_kwargs=attention_kwargs,
            kv_cache=None,
            kv_cache_mode=None,
            return_dict=False,
        )[0][:, -target_tokens:]
        return negative_velocity + guidance_scale * (velocity - negative_velocity)

    def _predict_velocity(
        self,
        *,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: Union[torch.Tensor, List[torch.Tensor]],
        prompt_embeds_mask: Union[torch.Tensor, List[torch.Tensor]],
        image_pad_mask: Union[torch.Tensor, List[torch.Tensor]],
        img_shapes: Any,
        condition_image_latents: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]],
        negative_prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        negative_prompt_embeds_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        negative_image_pad_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        guidance_scale: float,
        attention_kwargs: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Predict samples independently so ragged multimodal layouts never mix."""
        batch_size = latents.shape[0]
        device = latents.device
        predictions = []
        for index in range(batch_size):
            single_t = t
            if t.ndim > 0 and t.numel() == batch_size:
                single_t = t[index : index + 1]
            predictions.append(
                self._predict_velocity_one(
                    t=single_t,
                    latents=latents[index : index + 1],
                    prompt_embeds=self._single_tensor(
                        prompt_embeds,
                        index=index,
                        batch_size=batch_size,
                        unbatched_ndim=2,
                        name="prompt_embeds",
                        device=device,
                    ),
                    prompt_embeds_mask=self._single_tensor(
                        prompt_embeds_mask,
                        index=index,
                        batch_size=batch_size,
                        unbatched_ndim=1,
                        name="prompt_embeds_mask",
                        device=device,
                    ),
                    image_pad_mask=self._single_tensor(
                        image_pad_mask,
                        index=index,
                        batch_size=batch_size,
                        unbatched_ndim=1,
                        name="image_pad_mask",
                        device=device,
                    ),
                    img_shapes=self._single_img_shapes(
                        img_shapes,
                        index=index,
                        batch_size=batch_size,
                    ),
                    condition_image_latents=self._single_tensor(
                        condition_image_latents,
                        index=index,
                        batch_size=batch_size,
                        unbatched_ndim=2,
                        name="condition_image_latents",
                        device=device,
                    ),
                    negative_prompt_embeds=self._single_tensor(
                        negative_prompt_embeds,
                        index=index,
                        batch_size=batch_size,
                        unbatched_ndim=2,
                        name="negative_prompt_embeds",
                        device=device,
                    ),
                    negative_prompt_embeds_mask=self._single_tensor(
                        negative_prompt_embeds_mask,
                        index=index,
                        batch_size=batch_size,
                        unbatched_ndim=1,
                        name="negative_prompt_embeds_mask",
                        device=device,
                    ),
                    negative_image_pad_mask=self._single_tensor(
                        negative_image_pad_mask,
                        index=index,
                        batch_size=batch_size,
                        unbatched_ndim=1,
                        name="negative_image_pad_mask",
                        device=device,
                    ),
                    guidance_scale=guidance_scale,
                    attention_kwargs=attention_kwargs,
                )
            )
        return torch.cat(predictions, dim=0)

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: Union[torch.Tensor, List[torch.Tensor]],
        prompt_embeds_mask: Union[torch.Tensor, List[torch.Tensor]],
        image_pad_mask: Union[torch.Tensor, List[torch.Tensor]],
        img_shapes: Any,
        condition_image_latents: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
        negative_prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_prompt_embeds_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_image_pad_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        guidance_scale: float = 1.0,
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        noise_level: Optional[float] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
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
        """Run one cache-free rollout/replay step and one batched scheduler step."""
        velocity = self._predict_velocity(
            t=t,
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            image_pad_mask=image_pad_mask,
            img_shapes=img_shapes,
            condition_image_latents=condition_image_latents,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            negative_image_pad_mask=negative_image_pad_mask,
            guidance_scale=guidance_scale,
            attention_kwargs=attention_kwargs,
        )
        return self.scheduler.step(
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

    # ============================== Inference ==============================

    @staticmethod
    def _batch_one(value: Any, *, unbatched_ndim: int, name: str) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"{name} must contain exactly one sample")
            value = value[0]
            if value is None:
                return None
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, received {type(value).__name__}")
        if value.ndim == unbatched_ndim:
            return value.unsqueeze(0)
        if value.ndim == unbatched_ndim + 1 and value.shape[0] == 1:
            return value
        raise ValueError(f"{name} has invalid shape {tuple(value.shape)}")

    @staticmethod
    def _one_layout(value: Any, *, name: str) -> List[Tuple[int, int, int]]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(f"{name} must be a sequence, received {type(value).__name__}")
        if (
            len(value) == 1
            and isinstance(value[0], Sequence)
            and not isinstance(value[0], (str, bytes))
            and (not value[0] or isinstance(value[0][0], Sequence))
        ):
            value = value[0]
        result = []
        for shape in value:
            if isinstance(shape, torch.Tensor):
                shape = shape.detach().cpu().tolist()
            if (
                not isinstance(shape, Sequence)
                or isinstance(shape, (str, bytes))
                or len(shape) != 3
            ):
                raise ValueError(f"{name} contains invalid shape {shape!r}")
            result.append(tuple(int(item) for item in shape))
        return result

    @torch.no_grad()
    def inference(
        self,
        images: Optional[MultiImageBatch] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 1.0,
        height: int = 1024,
        width: int = 1024,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        prompt_embeds_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        image_pad_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_prompt_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_prompt_embeds_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_image_pad_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        condition_images: Optional[Any] = None,
        condition_image_sizes: Optional[Any] = None,
        condition_img_shapes: Optional[Any] = None,
        condition_image_latents: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
        condition_image_size: Union[int, Tuple[int, int]] = DEFAULT_CONDITION_IMAGE_SIZE,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        latents: Optional[torch.Tensor] = None,
        compute_log_prob: bool = False,
        use_kv_cache: bool = False,
        extra_call_back_kwargs: Sequence[str] = (),
        trajectory_indices: TrajectoryIndicesType = "all",
        **_: Any,
    ) -> List[QwenImage21Sample]:
        """Generate one sample with an exact replayable cache-free trajectory."""
        if use_kv_cache:
            raise ValueError(
                "Qwen-Image 2.1 KV cache is disabled in Flow-Factory because a no-grad "
                "rollout cache cannot be replayed through trainable prefix projections."
            )
        prompt_batch = [prompt] if isinstance(prompt, str) else prompt
        if prompt_batch is not None and len(prompt_batch) != 1:
            raise ValueError(
                "Qwen-Image 2.1 currently supports per_device_batch_size=1 to preserve "
                "ragged multimodal layout and RoPE parity."
            )
        if isinstance(generator, list):
            if len(generator) != 1:
                raise ValueError("Qwen-Image 2.1 expects exactly one generator")
            generator = generator[0]

        missing_condition = any(
            value is None for value in (prompt_embeds, prompt_embeds_mask, image_pad_mask)
        )
        if missing_condition:
            if prompt_batch is None:
                raise ValueError("prompt is required when encoded prompt state is incomplete")
            encoded = self.preprocess_func(
                prompt=list(prompt_batch),
                images=images,
                negative_prompt=(
                    [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                ),
                guidance_scale=guidance_scale,
                condition_image_size=condition_image_size,
                device=self.device,
            )
            prompt_ids = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
            prompt_embeds_mask = encoded["prompt_embeds_mask"]
            image_pad_mask = encoded["image_pad_mask"]
            negative_prompt_ids = encoded.get("negative_prompt_ids")
            negative_prompt_embeds = encoded.get("negative_prompt_embeds")
            negative_prompt_embeds_mask = encoded.get("negative_prompt_embeds_mask")
            negative_image_pad_mask = encoded.get("negative_image_pad_mask")
            condition_images = encoded["condition_images"]
            condition_image_sizes = encoded["condition_image_sizes"]
            condition_img_shapes = encoded["condition_img_shapes"]
            condition_image_latents = encoded["condition_image_latents"]

        device = self.device
        dtype = self.transformer.dtype
        prompt_ids_b = self._batch_one(prompt_ids, unbatched_ndim=1, name="prompt_ids")
        prompt_embeds_b = self._batch_one(
            prompt_embeds,
            unbatched_ndim=2,
            name="prompt_embeds",
        ).to(device=device, dtype=dtype)
        prompt_mask_b = self._batch_one(
            prompt_embeds_mask,
            unbatched_ndim=1,
            name="prompt_embeds_mask",
        ).to(device)
        image_mask_b = self._batch_one(
            image_pad_mask,
            unbatched_ndim=1,
            name="image_pad_mask",
        ).to(device)
        negative_prompt_ids_b = self._batch_one(
            negative_prompt_ids,
            unbatched_ndim=1,
            name="negative_prompt_ids",
        )
        negative_prompt_embeds_b = self._batch_one(
            negative_prompt_embeds,
            unbatched_ndim=2,
            name="negative_prompt_embeds",
        )
        if negative_prompt_embeds_b is not None:
            negative_prompt_embeds_b = negative_prompt_embeds_b.to(device=device, dtype=dtype)
        negative_prompt_mask_b = self._batch_one(
            negative_prompt_embeds_mask,
            unbatched_ndim=1,
            name="negative_prompt_embeds_mask",
        )
        if negative_prompt_mask_b is not None:
            negative_prompt_mask_b = negative_prompt_mask_b.to(device)
        negative_image_mask_b = self._batch_one(
            negative_image_pad_mask,
            unbatched_ndim=1,
            name="negative_image_pad_mask",
        )
        if negative_image_mask_b is not None:
            negative_image_mask_b = negative_image_mask_b.to(device)
        condition_latents_b = self._batch_one(
            condition_image_latents,
            unbatched_ndim=2,
            name="condition_image_latents",
        )
        if condition_latents_b is not None:
            condition_latents_b = condition_latents_b.to(device=device, dtype=dtype)

        target_latents = self.prepare_latents(
            batch_size=1,
            num_channels_latents=self.transformer.config.in_channels,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        condition_shapes = self._one_layout(
            condition_img_shapes,
            name="condition_img_shapes",
        )
        target_shape = (
            1,
            height // self.pipeline.vae_scale_factor,
            width // self.pipeline.vae_scale_factor,
        )
        img_shapes = [condition_shapes + [target_shape]]

        timesteps = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=target_latents.shape[1],
            device=device,
        )
        latent_collector = create_trajectory_collector(
            trajectory_indices,
            num_inference_steps,
        )
        target_latents = self.cast_latents(target_latents, default_dtype=dtype)
        latent_collector.collect(target_latents, step_idx=0)
        log_prob_collector = (
            create_trajectory_collector(trajectory_indices, num_inference_steps)
            if compute_log_prob
            else None
        )
        callback_collector = create_callback_collector(
            trajectory_indices,
            num_inference_steps,
        )

        for index, timestep in enumerate(timesteps):
            current_noise_level = self.scheduler.get_noise_level_for_timestep(timestep)
            timestep_next = (
                timesteps[index + 1]
                if index + 1 < len(timesteps)
                else torch.tensor(0, device=device)
            )
            current_compute_log_prob = compute_log_prob and current_noise_level > 0
            return_fields = tuple(
                set(("next_latents", "log_prob", "velocity", *extra_call_back_kwargs))
            )
            output = self.forward(
                t=timestep,
                t_next=timestep_next,
                latents=target_latents,
                prompt_embeds=prompt_embeds_b,
                prompt_embeds_mask=prompt_mask_b,
                image_pad_mask=image_mask_b,
                img_shapes=img_shapes,
                condition_image_latents=condition_latents_b,
                negative_prompt_embeds=negative_prompt_embeds_b,
                negative_prompt_embeds_mask=negative_prompt_mask_b,
                negative_image_pad_mask=negative_image_mask_b,
                guidance_scale=guidance_scale,
                attention_kwargs=attention_kwargs,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_fields,
                noise_level=current_noise_level,
            )
            target_latents = self.cast_latents(output.next_latents, default_dtype=dtype)
            latent_collector.collect(target_latents, index + 1)
            if current_compute_log_prob and log_prob_collector is not None:
                log_prob_collector.collect(output.log_prob, index)
            callback_collector.collect_step(
                step_idx=index,
                output=output,
                keys=list(extra_call_back_kwargs),
                capturable={"noise_level": current_noise_level},
            )

        generated_images = self.decode_latents(
            target_latents,
            height,
            width,
            output_type="pt",
        )
        all_latents = latent_collector.get_result()
        all_log_probs = log_prob_collector.get_result() if log_prob_collector else None
        callback_values = callback_collector.get_result()
        callback_index_map = callback_collector.get_index_map()

        stored_condition_images = condition_images
        if (
            isinstance(stored_condition_images, list)
            and len(stored_condition_images) == 1
            and isinstance(stored_condition_images[0], list)
        ):
            stored_condition_images = stored_condition_images[0]
        sample = QwenImage21Sample(
            timesteps=timesteps,
            all_latents=(
                torch.stack([value[0] for value in all_latents], dim=0)
                if all_latents is not None
                else None
            ),
            latent_index_map=latent_collector.get_index_map(),
            log_probs=(
                torch.stack([value[0] for value in all_log_probs], dim=0)
                if all_log_probs is not None
                else None
            ),
            log_prob_index_map=(log_prob_collector.get_index_map() if log_prob_collector else None),
            height=height,
            width=width,
            image=generated_images[0],
            condition_images=stored_condition_images or None,
            condition_image_latents=(
                condition_latents_b[0] if condition_latents_b is not None else None
            ),
            condition_img_shapes=condition_shapes,
            img_shapes=img_shapes[0],
            prompt=prompt_batch[0] if prompt_batch is not None else None,
            prompt_ids=prompt_ids_b[0] if prompt_ids_b is not None else None,
            prompt_embeds=prompt_embeds_b[0],
            prompt_embeds_mask=prompt_mask_b[0],
            image_pad_mask=image_mask_b[0],
            negative_prompt=(
                negative_prompt[0] if isinstance(negative_prompt, list) else negative_prompt
            ),
            negative_prompt_ids=(
                negative_prompt_ids_b[0] if negative_prompt_ids_b is not None else None
            ),
            negative_prompt_embeds=(
                negative_prompt_embeds_b[0] if negative_prompt_embeds_b is not None else None
            ),
            negative_prompt_embeds_mask=(
                negative_prompt_mask_b[0] if negative_prompt_mask_b is not None else None
            ),
            negative_image_pad_mask=(
                negative_image_mask_b[0] if negative_image_mask_b is not None else None
            ),
            extra_kwargs={
                **{key: value[0] for key, value in callback_values.items()},
                "callback_index_map": callback_index_map,
                "condition_image_sizes": (
                    condition_image_sizes[0]
                    if isinstance(condition_image_sizes, list)
                    and len(condition_image_sizes) == 1
                    and isinstance(condition_image_sizes[0], list)
                    else condition_image_sizes
                ),
            },
        )
        self.pipeline.maybe_free_model_hooks()
        return [sample]


__all__ = ["QwenImage21Adapter", "QwenImage21Sample", "calculate_dimensions"]
