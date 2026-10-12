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

"""SenseNova-U1 1.0/1.5 adapter for T2I and ordered multi-reference I2I."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from ...contracts import (
    GeometrySource,
    InputMediaOrder,
    NegativePromptPolicy,
)
from ...samples import I2ISample, T2ISample
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
)
from ...utils.image import MultiImageBatch, is_multi_image_batch, standardize_image_batch
from ...utils.trajectory_collector import (
    TrajectoryIndicesType,
    create_callback_collector,
    create_trajectory_collector,
)
from ..abc import BaseAdapter
from ..output_state import DecodedMediaBatch, EncodedOutputState, OutputStateCodec
from ..pipeline_contracts import image_output_contract
from ..runtime import ComponentRuntime, PseudoPipelineRuntime
from ._output import SenseNovaPixelOutputCodec
from .modeling.neo_unify.modeling_neo_chat import (
    SYSTEM_MESSAGE_FOR_GEN,
    clear_flash_kv_cache,
    load_image_native,
    optimized_scale,
    prepare_flash_kv_cache,
)
from .pipeline import SenseNovaPseudoPipeline


@dataclass
class SenseNovaSample(T2ISample):
    """T2I rollout sample shared by SenseNova-U1 1.0 and 1.5."""

    _shared_fields: ClassVar[frozenset[str]] = frozenset()


@dataclass
class SenseNovaI2ISample(I2ISample):
    """Image-to-image sample with ordered, variable-count reference images."""

    _shared_fields: ClassVar[frozenset[str]] = frozenset()
    # Reference images are persisted by the HF Image feature and must stay PIL on
    # the sample so ragged image sizes/counts survive the rollout/replay boundary.
    condition_images_as_pil: ClassVar[bool] = True


class SenseNovaAdapter(BaseAdapter):
    """Adapter for SenseNova-U1 1.0/1.5 T2I and ordered multi-reference I2I.

    The two checkpoints use the same NEO-Unify backbone and differ primarily in
    their flow-matching output head.  The vendored model reads ``use_pixel_head``
    from the checkpoint config, so no model-specific branch is needed here.

    The adapter follows the official NEO-Unify image-prefill contract for I2I:
    each ordered reference image is inserted into the prompt as a visual-token
    block, then the generated image is denoised against a text+image KV cache plus
    the image-only and unconditional caches its CFG scales require. Ragged
    reference-image batches remain nested PIL lists and are evaluated one generated
    sample at a time rather than using Bagel-style packed attention across
    independent samples.

    NEO-Unify owns text tokenization, vision encoding, and pixel-space flow
    matching inside one composite model. Consequently the component runtime
    declares only ``transformer`` (the ``SenseNovaDenoiser`` wrapper), with no
    standalone Flow-Factory VAE or text encoder.
    """

    offline_training_forward_overrides = MappingProxyType(
        {
            "guidance_scale": 1.0,
            "image_guidance_scale": 1.0,
            "cfg_norm": "none",
            "cfg_interval": (0.0, 1.0),
        }
    )

    # Reference images have variable spatial sizes/counts and are re-encoded at
    # rollout/replay time. Persist them through the HF Image feature as PIL.
    python_format_columns: ClassVar[frozenset[str]] = frozenset({"condition_images"})
    ddp_find_unused_parameters = True
    flow_velocity_direction: ClassVar[Literal["noise", "data"]] = "noise"
    pipeline_io_contract = image_output_contract(
        negative_prompt=NegativePromptPolicy.UNSUPPORTED,
        input_image_min_count=0,
        input_image_max_count=None,
        input_order=InputMediaOrder.WITHIN_TYPE,
        geometry_source=GeometrySource.CONFIGURED,
    )

    def load_pipeline(self) -> SenseNovaPseudoPipeline:
        """Load the custom Transformers checkpoint and tokenizer."""
        load_kwargs = dict(self.model_args.extra_kwargs)
        component_dtypes = self._resolve_component_load_dtype_mapping(
            component_names=["transformer"],
            transformer_names=["transformer"],
            text_encoder_names=[],
        )
        if "transformer" in component_dtypes:
            load_kwargs.setdefault("dtype", component_dtypes["transformer"])
        return SenseNovaPseudoPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False,
            **load_kwargs,
        )

    def build_component_runtime(self) -> ComponentRuntime:
        """Expose the differentiable generation wrapper as the sole component."""
        pipeline = self.load_pipeline()
        return PseudoPipelineRuntime(pipeline, {"transformer": pipeline.transformer})

    def load_scheduler(self) -> FlowMatchEulerDiscreteSDEScheduler:
        """Create Flow-Factory's SDE scheduler for the native clean-noise flow."""
        scheduler_kwargs = {"num_train_timesteps": 1000, "shift": 1.0}
        scheduler_args = getattr(self.config, "scheduler_args", None)
        if scheduler_args:
            scheduler_kwargs.update(scheduler_args.to_dict())
        return FlowMatchEulerDiscreteSDEScheduler(**scheduler_kwargs)

    @property
    def default_target_modules(self) -> List[str]:
        """Return the union of U1.0/U1.5 generation LoRA targets.

        PEFT matches only modules present in the loaded checkpoint: U1.0 uses the
        MLP ``fm_head`` targets while U1.5 uses the pixel-head convolution targets.
        """
        return [
            "q_proj_mot_gen",
            "k_proj_mot_gen",
            "v_proj_mot_gen",
            "o_proj_mot_gen",
            "mlp_mot_gen.gate_proj",
            "mlp_mot_gen.up_proj",
            "mlp_mot_gen.down_proj",
            # U1.0's MLP flow head.
            "fm_head.0",
            "fm_head.2",
            # U1.5's pixel flow head.
            "fm_head.conv1",
            "fm_head.conv2",
            "timestep_embedder.mlp.0",
            "timestep_embedder.mlp.2",
            "noise_scale_embedder.mlp.0",
            "noise_scale_embedder.mlp.2",
        ]

    @property
    def preprocessing_modules(self) -> List[str]:
        """Tokenize prompts and encode reference images lazily during KV-cache building."""
        return []

    @property
    def inference_modules(self) -> List[str]:
        return ["transformer"]

    def encode_prompt(self, prompt: Union[str, List[str]], **kwargs: Any) -> Dict[str, Any]:
        """Persist raw prompts; official NEO tokenization is cache-dependent."""
        return {"prompt": [prompt] if isinstance(prompt, str) else prompt}

    def encode_image(
        self, images: MultiImageBatch, **kwargs: Any
    ) -> Optional[Dict[str, List[List[Image.Image]]]]:
        """Normalize one or more reference images per sample to RGB PIL images.

        Vision preprocessing is intentionally deferred to context construction so
        the same exact reference bytes are used by rollout and replay. Returning
        a nested list is required for ragged multi-reference Arrow columns.
        """
        if images is None:
            return None
        return {"condition_images": self._normalize_condition_images(images)}

    def encode_video(self, videos: Any, **kwargs: Any) -> None:
        """SenseNova-U1 has no video input in this adapter."""

    def build_output_state_codec(self) -> OutputStateCodec:
        """Declare direct pixel-state target encoding without model materialization."""
        return SenseNovaPixelOutputCodec(self)

    def _configured_output_image_geometry(self) -> Tuple[int, int]:
        """Return configured H/W after validating the official patch merge grid."""
        return self._image_shape(
            getattr(self.training_args, "height", None),
            getattr(self.training_args, "width", None),
            None,
        )

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Require target signatures and decode metadata to match configured H/W."""
        del condition
        height, width = self._configured_output_image_geometry()
        if len(encoded.geometry_signatures) != len(media_batch):
            raise ValueError(
                "SenseNova output codec must return one geometry signature per sample, "
                f"received {len(encoded.geometry_signatures)} for {len(media_batch)}"
            )
        for sample_index, signature in enumerate(encoded.geometry_signatures):
            geometry = signature.media[0]
            received = (geometry.height, geometry.width)
            if received != (height, width):
                raise ValueError(
                    "SenseNova encoded output geometry disagrees with configured H/W for "
                    f"sample {sample_index}: expected {(height, width)}, received {received}"
                )
        for name, value in (("height", height), ("width", width)):
            if encoded.decode_context.get(name) != value:
                raise ValueError(
                    f"SenseNova decode_context {name!r} must equal {value}, "
                    f"received {encoded.decode_context.get(name)!r}"
                )

    def decode_latents(
        self,
        latents: torch.Tensor,
        output_type: Literal["pil", "pt", "np"] = "pil",
    ) -> Union[torch.Tensor, np.ndarray, List[Image.Image]]:
        """Map SenseNova's normalized pixel output to the requested image type."""
        single = latents.ndim == 3
        if single:
            latents = latents.unsqueeze(0)
        images = (latents.float() * 0.5 + 0.5).clamp(0, 1)
        if output_type == "pt":
            return images
        if output_type == "np":
            return images.permute(0, 2, 3, 1).cpu().numpy()
        if output_type != "pil":
            raise ValueError(f"unsupported output_type={output_type!r}")
        return [
            Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            for image in images
        ]

    # ============================== Context and schedule ==============================

    def _base_model(self):
        """Return the unwrapped official NEO model for no-grad prefix work."""
        component = self._unwrap(self.transformer)
        if hasattr(component, "get_base_model"):
            component = component.get_base_model()
        return component.model

    @staticmethod
    def _as_batch_values(value: Any, batch_size: int, name: str) -> List[Any]:
        """Normalize a scalar or per-sample sequence to a batch list."""
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return list(value) * batch_size
            if len(value) != batch_size:
                raise ValueError(
                    f"SenseNova forward `{name}` has {len(value)} values; expected 1 or "
                    f"batch_size={batch_size}."
                )
            return list(value)
        return [value] * batch_size

    @staticmethod
    def _normalize_batch_tensor(
        value: torch.Tensor,
        batch_size: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        """Normalize scalar or one-dimensional timesteps to ``(B,)``."""
        values = value.float().reshape(-1).to(device=device)
        if values.numel() == 1:
            return values.expand(batch_size)
        if values.numel() != batch_size:
            raise ValueError(
                f"SenseNova forward `{name}` has {values.numel()} values; expected 1 or "
                f"batch_size={batch_size}."
            )
        return values

    def _image_shape(
        self,
        height: Optional[int],
        width: Optional[int],
        image_shape: Optional[Tuple[int, int]],
    ) -> Tuple[int, int]:
        if image_shape is not None:
            height, width = image_shape
        if height is None or width is None:
            raise ValueError(
                "SenseNova requires `height` and `width` in the sample batch; refusing to "
                "silently choose a resolution that may not match the stored trajectory."
            )
        height, width = int(height), int(width)
        model = self._base_model()
        divisor = model.patch_size * int(1 / model.downsample_ratio)
        if height % divisor or width % divisor:
            raise ValueError(
                f"SenseNova image dimensions must be divisible by patch merge factor {divisor}; "
                f"received height={height}, width={width}."
            )
        return height, width

    def _noise_scale(self, image_size: Tuple[int, int]) -> float:
        """Match the official resolution-dependent initialization scale."""
        model = self._base_model()
        height, width = image_size
        merge_size = int(1 / model.downsample_ratio)
        grid_h = height // model.patch_size
        grid_w = width // model.patch_size
        noise_scale = float(model.noise_scale)
        if model.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
            base = float(model.noise_scale_base_image_seq_len)
            noise_scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / base) * noise_scale
            if model.noise_scale_mode == "dynamic_sqrt":
                noise_scale = math.sqrt(noise_scale)
        return min(noise_scale, float(model.noise_scale_max_value))

    @staticmethod
    def _normalize_condition_images(
        condition_images: Optional[MultiImageBatch],
        batch_size: Optional[int] = None,
    ) -> List[List[Image.Image]]:
        """Normalize condition images to ordered per-sample RGB PIL lists.

        The dataset contract uses ``List[List[Image]]`` for ragged multi-reference
        batches, while direct callers often pass ``List[Image]`` for one sample.
        Empty inner lists are preserved so a batch can mix I2I and T2I samples.
        """
        if condition_images is None:
            return [[] for _ in range(batch_size or 0)]

        if isinstance(condition_images, list) and (
            not condition_images
            or any(isinstance(value, (list, tuple)) for value in condition_images)
        ):
            per_sample: List[Any] = list(condition_images)
        elif is_multi_image_batch(condition_images):
            if isinstance(condition_images, torch.Tensor):
                per_sample = list(condition_images.unbind(0))
            else:
                per_sample = list(condition_images)
        else:
            per_sample = [condition_images]

        if batch_size is not None and len(per_sample) != batch_size:
            raise ValueError(
                "SenseNova `condition_images` must contain one image list per sample; "
                f"received {len(per_sample)} lists for batch_size={batch_size}."
            )

        normalized: List[List[Image.Image]] = []
        for images in per_sample:
            if images is None:
                normalized.append([])
                continue
            if isinstance(images, (list, tuple)) and len(images) == 0:
                normalized.append([])
                continue
            normalized.append(list(standardize_image_batch(images, output_type="pil")))
        return normalized

    def _prepare_reference_images(
        self, condition_images: List[Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the official NEO-Unify image preprocessing to all references."""
        if not condition_images:
            raise ValueError("SenseNova I2I requires at least one condition image.")
        model = self._base_model()
        pixel_values: List[torch.Tensor] = []
        grid_hw: List[torch.Tensor] = []
        max_pixels = min(2048 * 2048, (4096 * 4096) // len(condition_images))
        for image in condition_images:
            current_pixels, current_grid = load_image_native(
                image,
                patch_size=model.patch_size,
                downsample_ratio=model.downsample_ratio,
                min_pixels=512 * 512,
                max_pixels=max_pixels,
                upscale=False,
            )
            pixel_values.append(current_pixels.to(device=self.device, dtype=self.transformer.dtype))
            grid_hw.append(current_grid.to(device=self.device))
        return torch.cat(pixel_values, dim=0), torch.cat(grid_hw, dim=0)

    @staticmethod
    def _insert_missing_image_placeholders(prompt: str, image_count: int) -> str:
        """Match the official prompt convention for omitted ``<image>`` markers."""
        image_token_count = prompt.count("<image>")
        if image_count < image_token_count:
            raise ValueError(
                f"SenseNova prompt contains {image_token_count} `<image>` placeholders but "
                f"only {image_count} condition images were provided."
            )
        if image_count > image_token_count:
            if image_token_count == 0 and image_count > 1:
                prefix = "".join(f"Image-{index + 1}:<image>\n" for index in range(image_count))
                prompt = prefix + prompt
            else:
                prompt = "<image>\n" * (image_count - image_token_count) + prompt
        return prompt

    def _replace_image_placeholders(self, query: str, grid_hw: torch.Tensor) -> str:
        """Expand each logical image marker to the official visual-token block."""
        model = self._base_model()
        for grid in grid_hw:
            num_patch_tokens = int(grid[0].item() * grid[1].item() * model.downsample_ratio**2)
            image_tokens = "<img>" + "<IMG_CONTEXT>" * num_patch_tokens + "</img>"
            if "<image>" not in query:
                raise ValueError("SenseNova image-token expansion ran out of `<image>` markers.")
            query = query.replace("<image>", image_tokens, 1)
        return query

    def _build_it2i_branch(
        self,
        query: str,
        pixel_values: torch.Tensor,
        grid_hw: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Build one official image-prefilled generation branch."""
        model = self._base_model()
        model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        model.img_start_token_id = self.tokenizer.convert_tokens_to_ids("<img>")
        query = self._replace_image_placeholders(query, grid_hw)
        with torch.no_grad():
            input_embeds, indexes, attention_mask = model._build_it2i_inputs(
                self.tokenizer, query, pixel_values, grid_hw
            )
            cache, _ = model._it2i_prefix_forward(input_embeds, indexes, attention_mask)

        merge_size = int(1 / model.downsample_ratio)
        token_h = image_size[1] // (model.patch_size * merge_size)
        token_w = image_size[0] // (model.patch_size * merge_size)
        indexes_image = model._build_t2i_image_indexes(
            token_h,
            token_w,
            int(indexes[0].max().item()) + 1,
            device=self.device,
        )
        prepare_flash_kv_cache(cache, current_len=token_h * token_w, batch_size=1)
        return {
            "past_key_values": cache,
            "indexes_image": indexes_image,
            "attention_mask": {"full_attention": None},
        }

    def _build_i2i_context(
        self,
        prompt: str,
        image_size: Tuple[int, int],
        guidance_scale: float,
        image_guidance_scale: float,
        condition_images: List[Image.Image],
    ) -> Dict[str, Any]:
        """Build the text+image cache and whichever image-only/unconditional caches CFG needs."""
        if not condition_images:
            raise ValueError("SenseNova I2I context construction needs condition images.")
        if guidance_scale < 0 or image_guidance_scale < 0:
            raise ValueError("SenseNova CFG scales must be non-negative.")
        pixel_values, grid_hw = self._prepare_reference_images(condition_images)
        prompt = self._insert_missing_image_placeholders(prompt, len(condition_images))
        model = self._base_model()
        condition_query = model._build_t2i_query(
            prompt,
            system_message=SYSTEM_MESSAGE_FOR_GEN,
            append_text="<think>\n\n</think>\n\n<img>",
        )
        image_query = model._build_t2i_query("<image>" * len(condition_images), append_text="<img>")
        unconditional_query = model._build_t2i_query("", append_text="<img>")

        condition_branch = self._build_it2i_branch(
            condition_query, pixel_values, grid_hw, image_size
        )
        needs_guidance = not (guidance_scale == 1 and image_guidance_scale == 1)
        needs_image_branch = needs_guidance and (
            image_guidance_scale == 1 or guidance_scale != image_guidance_scale
        )
        needs_unconditional = needs_guidance and image_guidance_scale != 1
        image_branch = (
            self._build_it2i_branch(image_query, pixel_values, grid_hw, image_size)
            if needs_image_branch
            else {"past_key_values": None, "indexes_image": None, "attention_mask": None}
        )
        unconditional_branch = {
            "past_key_values": None,
            "indexes_image": None,
            "attention_mask": None,
        }
        # The unconditional branch has no reference pixels, but it still uses the
        # official image-aware index builder so the terminal ``<img>`` token keeps
        # the same positional contract as the image-prefilled branches.
        if needs_unconditional:
            model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            model.img_start_token_id = self.tokenizer.convert_tokens_to_ids("<img>")
            input_embeds, indexes, attention_mask = model._build_it2i_inputs(
                self.tokenizer, unconditional_query
            )
            with torch.no_grad():
                cache, _ = model._it2i_prefix_forward(input_embeds, indexes, attention_mask)
            merge_size = int(1 / model.downsample_ratio)
            token_h = image_size[1] // (model.patch_size * merge_size)
            token_w = image_size[0] // (model.patch_size * merge_size)
            unconditional_branch = {
                "past_key_values": cache,
                "indexes_image": model._build_t2i_image_indexes(
                    token_h, token_w, indexes.shape[1], device=self.device
                ),
                "attention_mask": {"full_attention": None},
            }
            prepare_flash_kv_cache(cache, current_len=token_h * token_w, batch_size=1)

        return {
            **condition_branch,
            "img_past_key_values": image_branch["past_key_values"],
            "img_indexes_image": image_branch["indexes_image"],
            "img_attention_mask": image_branch["attention_mask"],
            "uncond_past_key_values": unconditional_branch["past_key_values"],
            "uncond_indexes_image": unconditional_branch["indexes_image"],
            "uncond_attention_mask": unconditional_branch["attention_mask"],
            "is_i2i": True,
        }

    def _build_context(
        self,
        prompt: str,
        image_size: Tuple[int, int],
        guidance_scale: float,
        image_guidance_scale: float = 1.0,
        condition_images: Optional[List[Image.Image]] = None,
    ) -> Dict[str, Any]:
        """Build and flash-prepare T2I or I2I prefix caches."""
        if condition_images:
            return self._build_i2i_context(
                prompt,
                image_size,
                guidance_scale,
                image_guidance_scale,
                condition_images,
            )
        model = self._base_model()
        device = self.device
        merge_size = int(1 / model.downsample_ratio)
        token_h = image_size[1] // (model.patch_size * merge_size)
        token_w = image_size[0] // (model.patch_size * merge_size)
        append_text = "<think>\n\n</think>\n\n<img>"
        query = model._build_t2i_query(
            prompt,
            system_message=SYSTEM_MESSAGE_FOR_GEN,
            append_text=append_text,
        )
        input_ids, indexes, attention_mask = model._build_t2i_text_inputs(self.tokenizer, query)
        uncond_ids = uncond_indexes = uncond_attention_mask = uncond_cache = None
        if guidance_scale > 1:
            uncond_query = model._build_t2i_query("", append_text="<img>")
            uncond_ids, uncond_indexes, uncond_attention_mask = model._build_t2i_text_inputs(
                self.tokenizer, uncond_query
            )

        with torch.no_grad():
            cache, _ = model._t2i_prefix_forward(input_ids, indexes, attention_mask)
            if uncond_ids is not None:
                uncond_cache, _ = model._t2i_prefix_forward(
                    uncond_ids, uncond_indexes, uncond_attention_mask
                )

        indexes_image = model._build_t2i_image_indexes(
            token_h, token_w, indexes.shape[1], device=device
        )
        uncond_indexes_image = (
            model._build_t2i_image_indexes(token_h, token_w, uncond_indexes.shape[1], device=device)
            if uncond_indexes is not None
            else None
        )
        prepare_flash_kv_cache(cache, current_len=token_h * token_w, batch_size=1)
        if uncond_cache is not None:
            prepare_flash_kv_cache(uncond_cache, current_len=token_h * token_w, batch_size=1)

        return {
            "past_key_values": cache,
            "indexes_image": indexes_image,
            "attention_mask": {"full_attention": None},
            "img_past_key_values": None,
            "img_indexes_image": None,
            "img_attention_mask": None,
            "uncond_past_key_values": uncond_cache,
            "uncond_indexes_image": uncond_indexes_image,
            "uncond_attention_mask": {"full_attention": None},
            "is_i2i": False,
        }

    def _clear_context(self, context: Dict[str, Any]) -> None:
        cleared = set()
        for key in (
            "past_key_values",
            "img_past_key_values",
            "uncond_past_key_values",
        ):
            cache = context.get(key)
            if cache is not None and id(cache) not in cleared:
                clear_flash_kv_cache(cache)
                cleared.add(id(cache))

    @staticmethod
    def _cfg_velocity(
        condition: torch.Tensor,
        uncondition: torch.Tensor,
        scale: float,
        cfg_norm: str,
        step_index: Optional[int],
    ) -> torch.Tensor:
        """Apply official SenseNova classifier-free guidance and renorm."""
        if cfg_norm == "cfg_zero_star":
            alpha = optimized_scale(condition.flatten(1), uncondition.flatten(1))
            alpha = alpha.view(condition.shape[0], *([1] * (condition.ndim - 1)))
            alpha = alpha.to(condition.dtype)
            if step_index is not None and step_index <= 0:
                return condition * 0
            return uncondition * alpha + scale * (condition - uncondition * alpha)

        velocity = uncondition + scale * (condition - uncondition)
        if cfg_norm == "global":
            norm_condition = torch.norm(condition, dim=(1, 2), keepdim=True)
            norm_cfg = torch.norm(velocity, dim=(1, 2), keepdim=True)
            velocity = velocity * (norm_condition / (norm_cfg + 1e-8)).clamp(min=0, max=1.0)
        elif cfg_norm == "channel":
            norm_condition = torch.norm(condition, dim=-1, keepdim=True)
            norm_cfg = torch.norm(velocity, dim=-1, keepdim=True)
            velocity = velocity * (norm_condition / (norm_cfg + 1e-8)).clamp(min=0, max=1.0)
        elif cfg_norm != "none":
            raise ValueError(
                f"unsupported cfg_norm={cfg_norm!r}; expected 'none', 'global', 'channel', "
                "or 'cfg_zero_star'."
            )
        return velocity

    @staticmethod
    def _i2i_cfg_velocity(
        condition: torch.Tensor,
        image_condition: Optional[torch.Tensor],
        uncondition: Optional[torch.Tensor],
        guidance_scale: float,
        image_guidance_scale: float,
        cfg_norm: str,
        step_index: Optional[int],
    ) -> torch.Tensor:
        """Apply official dual CFG using text and image guidance scales."""
        if image_condition is None and uncondition is None:
            return condition
        if image_condition is None:
            return SenseNovaAdapter._cfg_velocity(
                condition, uncondition, guidance_scale, cfg_norm, step_index
            )
        if cfg_norm == "cfg_zero_star":
            raise ValueError("SenseNova I2I supports cfg_norm='none', 'global', or 'channel'.")

        if guidance_scale == 1 and image_guidance_scale == 1:
            velocity = condition
        elif image_guidance_scale == 1:
            velocity = image_condition + guidance_scale * (condition - image_condition)
        elif guidance_scale == image_guidance_scale:
            if uncondition is None:
                raise ValueError("SenseNova I2I image CFG requires an unconditional cache.")
            velocity = uncondition + guidance_scale * (condition - uncondition)
        else:
            if uncondition is None:
                raise ValueError("SenseNova I2I dual CFG requires an unconditional cache.")
            velocity = (
                uncondition
                + guidance_scale * (condition - image_condition)
                + image_guidance_scale * (image_condition - uncondition)
            )

        if cfg_norm == "global":
            norm_condition = torch.norm(condition, dim=(1, 2), keepdim=True)
            norm_cfg = torch.norm(velocity, dim=(1, 2), keepdim=True)
            velocity = velocity * (norm_condition / (norm_cfg + 1e-8)).clamp(min=0, max=1.0)
        elif cfg_norm == "channel":
            norm_condition = torch.norm(condition, dim=-1, keepdim=True)
            norm_cfg = torch.norm(velocity, dim=-1, keepdim=True)
            velocity = velocity * (norm_condition / (norm_cfg + 1e-8)).clamp(min=0, max=1.0)
        elif cfg_norm != "none":
            raise ValueError(
                f"unsupported I2I cfg_norm={cfg_norm!r}; expected 'none', 'global', or 'channel'."
            )
        return velocity

    def _native_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert scheduler's descending [0, 1000] time to native [0, 1] time."""
        return (1.0 - timestep.float() / 1000.0).clamp(0, 1)

    def _patch_velocity_to_pixels(
        self, velocity: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Convert NEO's patch velocity to the pixel-space scheduler state."""
        model = self._base_model()
        return model.unpatchify(
            velocity,
            model.patch_size * int(1 / model.downsample_ratio),
            image_size[1],
            image_size[0],
        )

    # ============================== Forward ==============================

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        guidance_scale: float = 4.0,
        cfg_norm: str = "none",
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        timestep_shift: float = 3.0,
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        indexes_image: Optional[torch.Tensor] = None,
        attention_mask: Optional[Dict[str, Any]] = None,
        uncond_past_key_values: Optional[Any] = None,
        uncond_indexes_image: Optional[torch.Tensor] = None,
        uncond_attention_mask: Optional[Dict[str, Any]] = None,
        noise_level: Optional[Union[float, torch.Tensor]] = None,
        compute_log_prob: bool = True,
        return_kwargs: Optional[List[str]] = None,
        step_index: Optional[int] = None,
        image_guidance_scale: float = 1.0,
        img_past_key_values: Optional[Any] = None,
        img_indexes_image: Optional[torch.Tensor] = None,
        img_attention_mask: Optional[Dict[str, Any]] = None,
        condition_images: Optional[MultiImageBatch] = None,
        **kwargs: Any,
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """Predict one transition and optionally evaluate its SDE log probability.

        A leading batch is accepted, but independent samples are processed
        sequentially with separate variable-length prefix caches and B=1 denoiser
        calls. Prebuilt ``past_key_values`` and companion cache arguments describe
        one sample and are therefore intended for B=1 inference replay.

        ``guidance_scale`` controls text CFG. ``image_guidance_scale`` controls the
        image branch for I2I and is ignored for T2I. Both are non-negative; ``1.0``
        disables the corresponding guidance delta.
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
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)
        if latents.ndim != 4:
            raise ValueError(
                f"SenseNovaAdapter.forward expects latents of rank 4, got {tuple(latents.shape)}"
            )
        batch_size = latents.shape[0]
        prompts = self._as_batch_values(prompt, batch_size, "prompt") if prompt is not None else []
        condition_batch = (
            self._normalize_condition_images(condition_images, batch_size)
            if condition_images is not None
            else [[] for _ in range(batch_size)]
        )
        shape = self._image_shape(height, width, image_shape)
        times = self._normalize_batch_tensor(t, batch_size, latents.device, "t")
        next_times = (
            self._normalize_batch_tensor(t_next, batch_size, latents.device, "t_next")
            if t_next is not None
            else None
        )
        if next_latents is not None and next_latents.ndim == 3:
            next_latents = next_latents.unsqueeze(0)
        if next_latents is not None and next_latents.shape != latents.shape:
            raise ValueError(
                "SenseNovaAdapter.forward requires `next_latents` to match `latents`; "
                f"received {tuple(next_latents.shape)} vs {tuple(latents.shape)}."
            )

        outputs: List[FlowMatchEulerDiscreteSDESchedulerOutput] = []
        for batch_index in range(batch_size):
            owns_context = past_key_values is None
            if owns_context:
                if not prompts:
                    raise ValueError(
                        "SenseNovaAdapter.forward requires raw `prompt` when no prebuilt "
                        "prefix cache is supplied."
                    )
                context = self._build_context(
                    prompts[batch_index],
                    shape,
                    guidance_scale,
                    image_guidance_scale,
                    condition_batch[batch_index],
                )
            else:
                context = {
                    "past_key_values": past_key_values,
                    "indexes_image": indexes_image,
                    "attention_mask": attention_mask or {"full_attention": None},
                    "uncond_past_key_values": uncond_past_key_values,
                    "uncond_indexes_image": uncond_indexes_image,
                    "uncond_attention_mask": uncond_attention_mask or {"full_attention": None},
                    "img_past_key_values": img_past_key_values,
                    "img_indexes_image": img_indexes_image,
                    "img_attention_mask": img_attention_mask,
                    "is_i2i": img_past_key_values is not None,
                }

            try:
                native_t = self._native_timestep(times[batch_index])
                noise_scale = self._noise_scale(shape)
                velocity_native = self.transformer(
                    latents=latents[batch_index : batch_index + 1],
                    timestep=native_t,
                    past_key_values=context["past_key_values"],
                    indexes_image=context["indexes_image"],
                    attention_mask=context["attention_mask"],
                    image_size=shape,
                    noise_scale=noise_scale,
                )
                guidance_requested = (
                    (guidance_scale != 1 or image_guidance_scale != 1)
                    if context["is_i2i"]
                    else guidance_scale > 1
                )
                use_cfg = (
                    guidance_requested
                    and native_t >= cfg_interval[0]
                    and native_t <= cfg_interval[1]
                )
                if use_cfg and context["is_i2i"]:
                    velocity_img = None
                    velocity_uncond = None
                    if context["img_past_key_values"] is not None:
                        velocity_img = self.transformer(
                            latents=latents[batch_index : batch_index + 1],
                            timestep=native_t,
                            past_key_values=context["img_past_key_values"],
                            indexes_image=context["img_indexes_image"],
                            attention_mask=context["img_attention_mask"],
                            image_size=shape,
                            noise_scale=noise_scale,
                        )
                    if context["uncond_past_key_values"] is not None:
                        velocity_uncond = self.transformer(
                            latents=latents[batch_index : batch_index + 1],
                            timestep=native_t,
                            past_key_values=context["uncond_past_key_values"],
                            indexes_image=context["uncond_indexes_image"],
                            attention_mask=context["uncond_attention_mask"],
                            image_size=shape,
                            noise_scale=noise_scale,
                        )
                    velocity_native = self._i2i_cfg_velocity(
                        velocity_native,
                        velocity_img,
                        velocity_uncond,
                        guidance_scale,
                        image_guidance_scale,
                        cfg_norm,
                        step_index,
                    )
                elif use_cfg and context["uncond_past_key_values"] is not None:
                    velocity_uncond = self.transformer(
                        latents=latents[batch_index : batch_index + 1],
                        timestep=native_t,
                        past_key_values=context["uncond_past_key_values"],
                        indexes_image=context["uncond_indexes_image"],
                        attention_mask=context["uncond_attention_mask"],
                        image_size=shape,
                        noise_scale=noise_scale,
                    )
                    velocity_native = self._cfg_velocity(
                        velocity_native,
                        velocity_uncond,
                        guidance_scale,
                        cfg_norm,
                        step_index,
                    )

                # The official model predicts clean-minus-noise while the
                # Flow-Factory scheduler integrates descending sigma and expects
                # noise-minus-clean.
                velocity_flow = -self._patch_velocity_to_pixels(velocity_native, shape)
                current_noise_level = noise_level
                if isinstance(noise_level, torch.Tensor) and noise_level.ndim > 0:
                    current_noise_level = noise_level.reshape(-1)[batch_index]
                output = self.scheduler.step(
                    velocity=velocity_flow,
                    timestep=times[batch_index],
                    latents=latents[batch_index : batch_index + 1],
                    timestep_next=None if next_times is None else next_times[batch_index],
                    next_latents=(
                        None
                        if next_latents is None
                        else next_latents[batch_index : batch_index + 1]
                    ),
                    compute_log_prob=compute_log_prob,
                    return_dict=True,
                    return_kwargs=return_kwargs,
                    noise_level=current_noise_level,
                )
                outputs.append(output)
            finally:
                if owns_context:
                    self._clear_context(context)

        merged: Dict[str, torch.Tensor] = {}
        for field in (
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
            "velocity",
        ):
            values = [getattr(output, field) for output in outputs]
            if all(value is not None for value in values):
                merged[field] = torch.cat(values, dim=0)
        return FlowMatchEulerDiscreteSDESchedulerOutput.from_dict(merged)

    # ============================== Inference ==============================

    @torch.no_grad()
    def inference(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        cfg_norm: str = "none",
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        timestep_shift: float = 3.0,
        enable_timestep_shift: bool = True,
        compute_log_prob: bool = True,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        extra_call_back_kwargs: Optional[List[str]] = None,
        trajectory_indices: TrajectoryIndicesType = "all",
        condition_images: Optional[MultiImageBatch] = None,
        image_guidance_scale: float = 1.0,
        **kwargs: Any,
    ) -> List[Union[SenseNovaSample, SenseNovaI2ISample]]:
        """Generate T2I or ordered multi-reference I2I samples.

        ``condition_images`` is a nested per-sample batch, and each inner list
        may contain one or more reference images. Prompts are generated
        sequentially with one prefix cache and denoising loop per sample; output
        order matches prompt order.

        ``guidance_scale`` controls text CFG. ``image_guidance_scale`` is I2I-only
        image guidance; ``1.0`` disables the corresponding guidance delta.
        """
        if prompt is None:
            raise ValueError("SenseNovaAdapter.inference requires `prompt`.")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        condition_batch = (
            self._normalize_condition_images(condition_images, len(prompts))
            if condition_images is not None
            else [[] for _ in prompts]
        )
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        shape = self._image_shape(height, width, None)
        device = self.device
        dtype = self.transformer.dtype
        callbacks = extra_call_back_kwargs or []
        model = self._base_model()
        merge_size = int(1 / model.downsample_ratio)
        token_h = height // (model.patch_size * merge_size)
        token_w = width // (model.patch_size * merge_size)
        image_token_num = token_h * token_w

        native_timesteps = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
        if enable_timestep_shift:
            native_timesteps = model._apply_time_schedule(
                native_timesteps, image_token_num, timestep_shift
            )
        flow_timesteps = 1000.0 * (1.0 - native_timesteps[:-1])
        # diffusers scheduler converts timesteps via np.array(); must be host-side.
        self.scheduler.set_timesteps(
            timesteps=flow_timesteps.detach().cpu().tolist(), device=device
        )
        timesteps = self.scheduler.timesteps

        samples: List[Union[SenseNovaSample, SenseNovaI2ISample]] = []
        for sample_index, sample_prompt in enumerate(prompts):
            sample_condition_images = condition_batch[sample_index]
            context = self._build_context(
                sample_prompt,
                shape,
                guidance_scale,
                image_guidance_scale,
                sample_condition_images,
            )
            try:
                sample_generator = (
                    generator[sample_index] if isinstance(generator, list) else generator
                )
                noise = randn_tensor(
                    (1, 3, height, width),
                    generator=sample_generator,
                    device=device,
                    dtype=dtype,
                )
                latents = self.cast_latents(noise * self._noise_scale(shape), default_dtype=dtype)
                latent_collector = create_trajectory_collector(
                    trajectory_indices, num_inference_steps
                )
                latent_collector.collect(latents, step_idx=0)
                log_prob_collector = (
                    create_trajectory_collector(trajectory_indices, num_inference_steps)
                    if compute_log_prob
                    else None
                )
                callback_collector = create_callback_collector(
                    trajectory_indices, num_inference_steps
                )

                for step_index, timestep in enumerate(timesteps):
                    timestep_next = (
                        timesteps[step_index + 1]
                        if step_index + 1 < len(timesteps)
                        else torch.tensor(0.0, device=device)
                    )
                    current_noise_level = self.scheduler.get_noise_level_for_timestep(timestep)
                    current_compute_log_prob = compute_log_prob and current_noise_level > 0
                    output = self.forward(
                        t=timestep,
                        t_next=timestep_next,
                        latents=latents,
                        prompt=sample_prompt,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale,
                        image_guidance_scale=image_guidance_scale,
                        cfg_norm=cfg_norm,
                        cfg_interval=cfg_interval,
                        timestep_shift=timestep_shift,
                        past_key_values=context["past_key_values"],
                        indexes_image=context["indexes_image"],
                        attention_mask=context["attention_mask"],
                        uncond_past_key_values=context["uncond_past_key_values"],
                        uncond_indexes_image=context["uncond_indexes_image"],
                        uncond_attention_mask=context["uncond_attention_mask"],
                        img_past_key_values=context["img_past_key_values"],
                        img_indexes_image=context["img_indexes_image"],
                        img_attention_mask=context["img_attention_mask"],
                        noise_level=current_noise_level,
                        compute_log_prob=current_compute_log_prob,
                        return_kwargs=list(
                            set(["velocity", "next_latents", "log_prob"] + callbacks)
                        ),
                        step_index=step_index,
                    )
                    latents = self.cast_latents(output.next_latents, default_dtype=dtype)
                    latent_collector.collect(latents, step_idx=step_index + 1)
                    if current_compute_log_prob and log_prob_collector is not None:
                        log_prob_collector.collect(output.log_prob, step_idx=step_index)
                    callback_collector.collect_step(
                        step_idx=step_index,
                        output=output,
                        keys=callbacks,
                        capturable={"noise_level": current_noise_level},
                    )

                images = self.decode_latents(latents, output_type="pt")
                all_latents = latent_collector.get_result()
                all_log_probs = log_prob_collector.get_result() if log_prob_collector else None
                callback_results = callback_collector.get_result() or {}
                sample_cls = SenseNovaI2ISample if sample_condition_images else SenseNovaSample
                sample_kwargs: Dict[str, Any] = {
                    "timesteps": timesteps,
                    "all_latents": (
                        torch.stack([value[0] for value in all_latents], dim=0)
                        if all_latents is not None
                        else None
                    ),
                    "latent_index_map": latent_collector.get_index_map(),
                    "log_probs": (
                        torch.stack([value[0] for value in all_log_probs], dim=0)
                        if all_log_probs is not None
                        else None
                    ),
                    "log_prob_index_map": (
                        log_prob_collector.get_index_map() if log_prob_collector else None
                    ),
                    "height": height,
                    "width": width,
                    "image": images[0],
                    "prompt": sample_prompt,
                    "extra_kwargs": {
                        **{key: value for key, value in callback_results.items()},
                        "callback_index_map": callback_collector.get_index_map(),
                    },
                }
                if sample_condition_images:
                    sample_kwargs["condition_images"] = sample_condition_images
                samples.append(sample_cls(**sample_kwargs))
            finally:
                self._clear_context(context)
        self.pipeline.maybe_free_model_hooks()
        return samples
