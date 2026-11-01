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

# src/flow_factory/models/wan/wan2_i2v.py
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from peft import PeftModel
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline, prompt_clean
from diffusers.utils.torch_utils import randn_tensor

from ...contracts import (
    BatchCapability,
    GeometrySource,
    NegativePromptPolicy,
    PipelineIOContract,
    RateRequirement,
)
from ...hparams import *
from ...samples import I2VSample
from ...scheduler import UniPCMultistepSDEScheduler, UniPCMultistepSDESchedulerOutput
from ...utils.image import (
    ImageBatch,
    ImageSingle,
    MultiImageBatch,
)
from ...utils.logger_utils import setup_logger
from ...utils.media import signed_unit_to_unit_pixels
from ...utils.trajectory_collector import (
    TrajectoryIndicesType,
    create_callback_collector,
    create_trajectory_collector,
)
from ..abc import BaseAdapter
from ..condition_state import ConditionStatePreparer
from ..output_state import DecodedMediaBatch, EncodedOutputState, OutputStateCodec
from ..pipeline_contracts import video_output_contract
from ._conditioning import (
    WanI2VConditionStatePreparer,
    append_wan_i2v_last_images,
    normalize_wan_i2v_image_rows,
    normalize_wan_image_embeds,
    prepare_wan_i2v_condition_tensors,
    preprocess_wan_i2v_image_rows,
    restore_wan_i2v_condition_pixels,
    split_wan_image_embeds,
)
from ._output import (
    WanVideoOutputCodec,
    configured_wan_video_output_geometry,
    normalize_wan_video_latents,
    resample_wan_output_video,
    validate_wan_encoded_output_geometry,
)

logger = setup_logger(__name__)


@dataclass
class WanI2VSample(I2VSample):
    # Class var
    _shared_fields: ClassVar[frozenset[str]] = frozenset()
    # Obj var
    image_embeds: Optional[torch.FloatTensor] = None
    latent_condition: Optional[torch.FloatTensor] = None
    first_frame_mask: Optional[torch.FloatTensor] = None


class Wan2_I2V_Adapter(BaseAdapter):
    preprocess_cache_fields = frozenset({"height", "width"})
    preprocess_cache_version = "wan-i2v-condition-pixels-v2"
    offline_training_forward_overrides = MappingProxyType(
        {"guidance_scale": 1.0, "guidance_scale_2": 1.0}
    )
    # Wan2.2 trains both transformer and transformer_2 but uses only one per
    # timestep (boundary_ratio), so under DDP the other's trainable params get no
    # gradient in a given step. Ignored under DeepSpeed/FSDP.
    ddp_find_unused_parameters = True
    supports_diffusers_cache = True
    component_load_dtype_defaults = {
        "transformers": torch.bfloat16,
        "text_encoders": torch.bfloat16,
        "vae": torch.float32,
        "image_encoder": torch.float32,
    }
    pipeline_io_contract = video_output_contract(
        negative_prompt=NegativePromptPolicy.OPTIONAL,
        input_image_min_count=1,
        input_image_max_count=2,
        input_image_slots=("first_frame", "last_frame"),
        required_input_image_slots=("first_frame",),
        output_fps=RateRequirement.REQUIRED,
        geometry_source=GeometrySource.CONFIGURED,
        batch_capability=BatchCapability.SINGLE_SAMPLE,
    )

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: WanImageToVideoPipeline
        self.scheduler: UniPCMultistepSDEScheduler

    def load_pipeline(self) -> WanImageToVideoPipeline:
        return self._load_diffusers_pipeline(
            WanImageToVideoPipeline,
            self.model_args.model_name_or_path,
        )

    def _resolve_pipeline_io_contract(self) -> PipelineIOContract:
        """Resolve exact-one for ordinary/expanded paths and exact-two for FLF2V weights.

        Wan2.2's VAE-only path retains the class-level one-or-two input superset.
        """
        supports_endpoint_pair = False
        if not self.pipeline.config.expand_timesteps:
            transformer_configs = tuple(
                transformer.config
                for transformer in (
                    getattr(self.pipeline, "transformer", None),
                    getattr(self.pipeline, "transformer_2", None),
                )
                if transformer is not None
            )
            clip_configs = tuple(
                config
                for config in transformer_configs
                if getattr(config, "image_dim", None) is not None
            )
            if not clip_configs:
                return type(self).pipeline_io_contract
            supports_endpoint_pair = any(
                getattr(config, "pos_embed_seq_len", None) is not None for config in clip_configs
            )
        if supports_endpoint_pair:
            return video_output_contract(
                negative_prompt=NegativePromptPolicy.OPTIONAL,
                input_image_min_count=2,
                input_image_max_count=2,
                input_image_slots=("first_frame", "last_frame"),
                required_input_image_slots=("first_frame", "last_frame"),
                output_fps=RateRequirement.REQUIRED,
                geometry_source=GeometrySource.CONFIGURED,
                batch_capability=BatchCapability.SINGLE_SAMPLE,
            )
        return video_output_contract(
            negative_prompt=NegativePromptPolicy.OPTIONAL,
            input_image_min_count=1,
            input_image_max_count=1,
            input_image_slots=("first_frame",),
            required_input_image_slots=("first_frame",),
            output_fps=RateRequirement.REQUIRED,
            geometry_source=GeometrySource.CONFIGURED,
            batch_capability=BatchCapability.SINGLE_SAMPLE,
        )

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Wan transformer."""
        return [
            # Self Attention
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            # Cross Attention
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            # Feed Forward Network
            "ffn.net.0.proj",
            "ffn.net.2",
        ]

    @classmethod
    def tensor_parallel_plan(cls) -> Dict[str, Any]:
        """Shard each block's feed-forward; attention is blocked by across-head qk-norm.

        Wan's qk-norm is ``rms_norm_across_heads`` and ``WanAttnProcessor``
        applies it *before* it splits heads, so every
        ``blocks.*.attn{1,2}.norm_{q,k}`` is ``RMSNorm(num_heads *
        attention_head_dim)`` -- ``[5120]`` on the 14B config, the whole inner
        dimension rather than ``[attention_head_dim]``. Column-parallel q/k
        would leave each rank taking the mean square over only its own
        ``inner_dim / tp`` slice, so the *forward* would be wrong, not merely the
        weight gradient that ``ShardedInputReplicatedWeight`` exists to repair.
        It fails loudly rather than silently -- measured at tp=2, ``Given
        normalized_shape=[5120] ... got input of size [1, 320, 2560]`` -- so
        attention stays out of the plan until a style that all-reduces the mean
        square exists. That costs the other 59.7% of a block, and with it the
        head-sharding of the attention activations.

        What is left is the feed-forward: 40.3% of a block
        (``ffn.net.0.proj`` 20.1% + ``ffn.net.2`` 20.1%), which is 19.8% of the
        14B model at tp=2 and 34.7% at tp=8. ``ffn`` is a plain GELU MLP rather
        than a packed SwiGLU, so its two halves column/row-shard without the
        ``[hidden, gate]`` interleaving that keeps H3's fused projection
        replicated. Sharding it also halves the ``ffn_dim``-wide intermediate,
        the largest single activation per token.

        The modulation is unshardable by a plan of this shape, and it is where
        this architecture actually spends its activation memory:
        ``condition_embedder.time_proj`` (1.10% of the model) emits six
        full-width vectors which each block's bare ``scale_shift_table``
        (0.009%) offsets, and they are applied in fp32 to hidden states that
        stay replicated. Measured on 75,600 video tokens in bf16, that fp32
        modulation chain -- not attention, which SDPA never materializes --
        dominates the high-water mark, so tp=2 saves only 7.8% of peak memory.
        Sharding it would take sequence parallelism.

        Applying it also needs two completions ``tensor_parallel.py`` does not
        make yet, both reported separately: a style for the bare
        ``scale_shift_table`` parameters that the transformer and each block own
        directly (``ReplicateParallel`` hooks a whole block's activations and
        then its residual mixes a DTensor with a plain tensor), and gating
        ``localize_attention_head_counts`` on whether the plan sharded q/k/v --
        unconditional localization rewrites ``attn.heads`` while the projections
        still emit full width, which Wan's RoPE catches but its cross-attention
        would not.
        """
        return {
            "blocks.*.ffn.net.0.proj": ColwiseParallel(),
            "blocks.*.ffn.net.2": RowwiseParallel(),
        }

    @property
    def inference_modules(self) -> List[str]:
        """Modules that are required for inference and forward."""
        if self.pipeline.config.boundary_ratio is None or self.pipeline.config.boundary_ratio <= 0:
            return ["transformer", "vae"]

        if self.pipeline.config.boundary_ratio >= 1:
            return ["transformer_2", "vae"]

        return ["transformer", "transformer_2", "vae"]

    @property
    def preprocessing_modules(self) -> List[str]:
        """Modules that are required for preprocessing."""
        return ["text_encoders", "image_encoder"]

    def apply_lora(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]] = ["transformer", "transformer_2"],
        **kwargs,
    ) -> Union[PeftModel, Dict[str, PeftModel]]:
        return super().apply_lora(target_modules=target_modules, components=components, **kwargs)

    # ======================= Components Getters & Setters =======================
    @property
    def image_encoder(self) -> torch.nn.Module:
        return self.get_component("image_encoder")

    @image_encoder.setter
    def image_encoder(self, module: torch.nn.Module):
        self.set_component("image_encoder", module)

    @property
    def transformer_2(self) -> torch.nn.Module:
        return self.get_component("transformer_2")

    @transformer_2.setter
    def transformer_2(self, module: torch.nn.Module):
        self.set_component("transformer_2", module)

    # ======================== Encoding & Decoding ========================
    # ------------------------ Prompt Encoding ------------------------
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.pipeline.text_encoder(
            text_input_ids.to(device), mask.to(device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        return text_input_ids, prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 5.0,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode prompts, and negative prompts when CFG is enabled, with the T5 encoder.

        Args:
            prompt: Prompt or prompts to encode.
            negative_prompt: Negative prompt or prompts; a string is repeated to the batch
                size and a list must match ``prompt``. Defaults to ``""``; encoded only when
                CFG is enabled.
            guidance_scale: Classifier-free guidance scale; CFG is enabled when it exceeds 1.0.
            max_sequence_length: Token budget; prompts are truncated and embeddings
                zero-padded to it.
            device: Target device; defaults to the text encoder's device.
            dtype: Target dtype; defaults to the text encoder's dtype.

        Returns:
            Dict with ``prompt_ids`` and ``prompt_embeds``, plus ``negative_prompt_ids`` and
            ``negative_prompt_embeds`` when CFG is enabled.
        """
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_ids, prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        results = {
            "prompt_ids": prompt_ids,
            "prompt_embeds": prompt_embeds,
        }

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_ids, negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            results.update(
                {
                    "negative_prompt_ids": negative_prompt_ids,
                    "negative_prompt_embeds": negative_prompt_embeds,
                }
            )

        return results

    # ------------------------ Image Encoding ------------------------
    def encode_image(
        self,
        images: Union[ImageSingle, ImageBatch, MultiImageBatch],
        device: Optional[torch.device] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        rows = normalize_wan_i2v_image_rows(images)
        height = height if height is not None else getattr(self.training_args, "height", None)
        width = width if width is not None else getattr(self.training_args, "width", None)
        if type(height) is not int or height <= 0 or type(width) is not int or width <= 0:
            raise ValueError(
                "Wan I2V image preprocessing requires positive integer train.height/width, "
                f"received {(height, width)}"
            )
        condition_images = []
        for row_index, row in enumerate(rows):
            pixel_values = self.pipeline.video_processor.preprocess(
                list(row),
                height=height,
                width=width,
            )
            expected_shape = (len(row), 3, height, width)
            if (
                not isinstance(pixel_values, torch.Tensor)
                or tuple(pixel_values.shape) != expected_shape
            ):
                raise ValueError(
                    "Wan video_processor.preprocess changed condition-image geometry at "
                    f"sample {row_index}: expected {expected_shape}, received "
                    f"{getattr(pixel_values, 'shape', None)}"
                )
            condition_images.append(
                signed_unit_to_unit_pixels(
                    pixel_values.detach().to(device="cpu", dtype=torch.float32)
                )
            )
        results: Dict[str, Union[List[torch.Tensor], torch.Tensor]] = {
            "condition_images": condition_images
        }

        # Only Wan2.1 I2V transformers (``image_dim`` set) accept image_embeds; others omit them.
        if (
            self.pipeline.transformer is not None
            and self.pipeline.transformer.config.image_dim is not None
        ):
            device = device or self.image_encoder.device
            counts = [len(row) for row in rows]
            flattened = [image for row in rows for image in row]
            processor_output = self.pipeline.image_processor(
                images=flattened,
                return_tensors="pt",
            ).to(device)
            image_embeds = self.pipeline.image_encoder(
                **processor_output,
                output_hidden_states=True,
            ).hidden_states[-2]
            if not isinstance(image_embeds, torch.Tensor) or image_embeds.shape[0] != sum(counts):
                raise ValueError(
                    "Wan image encoder must preserve the flattened condition-image count, "
                    f"expected {sum(counts)}, received "
                    f"{getattr(image_embeds, 'shape', None)}"
                )
            if all(count == 1 for count in counts):
                results["image_embeds"] = image_embeds
                return results
            per_sample = []
            offset = 0
            for count in counts:
                per_sample.append(image_embeds[offset : offset + count])
                offset += count
            results["image_embeds"] = per_sample
        return results

    # ------------------------ Condition & Output-State Encoding ------------------------
    def build_condition_state_preparer(self) -> ConditionStatePreparer:
        """Declare on-the-fly input-frame VAE conditioning."""
        return WanI2VConditionStatePreparer(self)

    def build_output_state_codec(self) -> OutputStateCodec:
        """Declare the shared Wan target-video codec with I2V active-mask binding."""
        return WanVideoOutputCodec(self, bind_condition_active_mask=True)

    def _configured_video_output_geometry(self) -> Tuple[int, int, int, float]:
        """Return configured Wan geometry after exact latent-grid validation."""
        return configured_wan_video_output_geometry(self)

    @staticmethod
    def _resample_output_video(
        video: np.ndarray,
        *,
        source_fps: Optional[float],
        target_frames: int,
        target_fps: float,
    ) -> np.ndarray:
        """Select deterministic nearest-time frames for configured target cadence."""
        return resample_wan_output_video(
            video,
            source_fps=source_fps,
            target_frames=target_frames,
            target_fps=target_fps,
        )

    def _normalize_output_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply Wan's VAE latent normalization."""
        return normalize_wan_video_latents(self, latents)

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Require encoded target metadata to match configured video geometry."""
        validate_wan_encoded_output_geometry(self, media_batch, condition, encoded)

    # ------------------------ Latent Decoding ------------------------
    def decode_latents(
        self, latents: torch.Tensor, output_type: Literal["pt", "pil", "np"] = "pil"
    ) -> torch.Tensor:
        """Decode the latents using the VAE decoder."""
        latents = latents.float()
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(
            1, self.pipeline.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = self.pipeline.vae.decode(latents, return_dict=False)[0]

        video = self.pipeline.video_processor.postprocess_video(video, output_type=output_type)
        return video

    # ======================== Latent Preparation ========================
    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare rollout noise and reuse the offline condition realization helper.

        Returns:
            ``(latents, condition)``, plus ``first_frame_mask`` as a third element when the
            pipeline uses ``expand_timesteps``.
        """
        dtype = dtype or torch.float32
        device = device or self.device
        num_latent_frames = (num_frames - 1) // self.pipeline.vae_scale_factor_temporal + 1
        latent_height = height // self.pipeline.vae_scale_factor_spatial
        latent_width = width // self.pipeline.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        realized = prepare_wan_i2v_condition_tensors(
            self,
            image,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
            last_image=last_image,
        )
        if realized.first_frame_mask is not None:
            return latents, realized.condition, realized.first_frame_mask
        return latents, realized.condition

    # ======================== Inference ========================
    @torch.no_grad()
    def inference(
        self,
        # Ordinary arguments
        images: Union[ImageSingle, ImageBatch, MultiImageBatch],
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # Encoded Prompt
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        # Encoded Negative Prompt
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # Encoded Image
        image_embeds: Optional[torch.Tensor] = None,
        condition_images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        last_image: Optional[Union[ImageSingle, ImageBatch]] = None,
        # Other args
        compute_log_prob: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
    ) -> List[WanI2VSample]:
        # 1. Setup args
        device = self.device
        do_classifier_free_guidance = guidance_scale > 1.0

        if self.pipeline.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale
        # Check `num_frames`
        if (num_frames - 1) % self.pipeline.vae_scale_factor_temporal != 0:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.pipeline.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames
                // self.pipeline.vae_scale_factor_temporal
                * self.pipeline.vae_scale_factor_temporal
                + 1
            )
        num_frames = max(num_frames, 1)
        # Check `height` and `width`
        patch_size = (
            self.pipeline.transformer.config.patch_size
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.config.patch_size
        )
        h_multiple_of = self.pipeline.vae_scale_factor_spatial * patch_size[1]
        w_multiple_of = self.pipeline.vae_scale_factor_spatial * patch_size[2]
        calc_height = height // h_multiple_of * h_multiple_of
        calc_width = width // w_multiple_of * w_multiple_of
        if height != calc_height or width != calc_width:
            logger.warning(
                f"`height` and `width` must be multiples of ({h_multiple_of}, {w_multiple_of}) for proper patchification. "
                f"Adjusting ({height}, {width}) -> ({calc_height}, {calc_width})."
            )
            height, width = calc_height, calc_width

        # 2. Encode prompt
        if prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_ids = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
            negative_prompt_ids = encoded.get("negative_prompt_ids", None)
            negative_prompt_embeds = encoded.get("negative_prompt_embeds", None)
        else:
            prompt_embeds = prompt_embeds.to(device)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device)

        batch_size = prompt_embeds.shape[0]
        transformer_dtype = (
            self.pipeline.transformer.dtype
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.dtype
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        condition_rows = normalize_wan_i2v_image_rows(
            images,
            expected_batch_size=batch_size,
        )
        condition_rows = append_wan_i2v_last_images(condition_rows, last_image)
        if self.pipeline.config.expand_timesteps and any(len(row) == 2 for row in condition_rows):
            raise ValueError(
                "Wan I2V expand_timesteps does not support an optional last-frame image; "
                "Diffusers would otherwise ignore it"
            )

        # 3. Set scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Encode image
        # Only wan 2.1 i2v transformer accepts image_embeds
        if (
            self.pipeline.transformer is not None
            and self.pipeline.transformer.config.image_dim is not None
        ):
            if image_embeds is None:
                image_encoded = self.encode_image(
                    condition_rows,
                    device,
                    height=height,
                    width=width,
                )
                image_embeds = image_encoded["image_embeds"]
                if condition_images is None:
                    condition_images = image_encoded["condition_images"]
        if image_embeds is not None:
            image_embeds = normalize_wan_image_embeds(
                image_embeds,
                batch_size=batch_size,
            ).to(device=device, dtype=transformer_dtype)
            per_sample_image_embeds = split_wan_image_embeds(
                image_embeds,
                [len(row) for row in condition_rows],
            )
        else:
            per_sample_image_embeds = (None,) * batch_size

        # 5. Prepare latent variables
        num_channels_latents = self.pipeline.vae.config.z_dim
        if condition_images is None:
            images, last_image_pixels = preprocess_wan_i2v_image_rows(
                self,
                condition_rows,
                height=height,
                width=width,
            )
        else:
            images, last_image_pixels = restore_wan_i2v_condition_pixels(
                condition_images,
                batch_size=batch_size,
                height=height,
                width=width,
                device=device,
            )
            expected_last = any(len(row) == 2 for row in condition_rows)
            if expected_last != (last_image_pixels is not None):
                raise ValueError(
                    "Wan cached condition_images count disagrees with ordered raw input images"
                )

        latents_outputs = self.prepare_latents(
            image=images,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=None,
            last_image=last_image_pixels,
        )
        if self.pipeline.config.expand_timesteps:
            # Wan 2.2 5B I2V uses first_frame_mask to expand timesteps.
            latents, condition, first_frame_mask = latents_outputs
        else:
            latents, condition = latents_outputs
            first_frame_mask = None

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.pipeline._num_timesteps = len(timesteps)

        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latents = self.cast_latents(latents)
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(
                trajectory_indices, num_inference_steps
            )
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        for i, t in enumerate(timesteps):
            self.pipeline._current_timestep = t
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kwargs = list(
                set(["next_latents", "log_prob", "velocity"] + extra_call_back_kwargs)
            )
            current_compute_log_prob = compute_log_prob and current_noise_level > 0

            output = self.forward(
                t=t,
                t_next=t_next,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                image_embeds=image_embeds,
                latent_condition=condition,
                first_frame_mask=first_frame_mask,
                attention_kwargs=attention_kwargs,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=current_noise_level,
            )

            latents = self.cast_latents(output.next_latents)
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob:
                log_prob_collector.collect(output.log_prob, i)

            callback_collector.collect_step(
                step_idx=i,
                output=output,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": current_noise_level},
            )

        self.pipeline._current_timestep = None

        # 7. Decode latents to videos
        if self.pipeline.config.expand_timesteps:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
        decoded_videos = self.decode_latents(latents, output_type="pt")

        # 8. Prepare output samples
        extra_call_back_res = callback_collector.get_result()  # (B, len(trajectory_indices), ...)
        callback_index_map = callback_collector.get_index_map()  # (T,) LongTensor
        all_latents = latent_collector.get_result()  # List[torch.Tensor(B, ...)]
        latent_index_map = latent_collector.get_index_map()  # (T+1,) LongTensor
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
        samples = [
            WanI2VSample(
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
                # Generated video & metadata
                video=decoded_videos[b],
                height=height,
                width=width,
                # Conditions
                condition_images=list(condition_rows[b]),
                latent_condition=condition[b],
                first_frame_mask=(first_frame_mask[b] if first_frame_mask is not None else None),
                image_embeds=per_sample_image_embeds[b],
                # Prompt info
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b] if prompt_embeds is not None else None,
                # Negative prompt info
                negative_prompt=(
                    negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt
                ),
                negative_prompt_ids=(
                    negative_prompt_ids[b] if negative_prompt_ids is not None else None
                ),
                negative_prompt_embeds=(
                    negative_prompt_embeds[b] if negative_prompt_embeds is not None else None
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

    # ======================== Forward ========================
    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        # Optional for CFG
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        # Optional for I2V
        image_embeds: Optional[torch.Tensor] = None,
        latent_condition: Optional[torch.Tensor] = None,
        first_frame_mask: Optional[torch.Tensor] = None,
        boundary_timestep: Optional[float] = None,
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
    ) -> UniPCMultistepSDESchedulerOutput:
        """Run one I2V denoising step: transformer forward, optional CFG, and scheduler step.

        Args:
            t: Current timestep; a 1-D tensor uses ``t[0]``, so the batch shares one timestep.
            latents: Current latent representations.
            latent_condition: Required condition latents (first/optional-last frames encoded).
            prompt_embeds: Text prompt embeddings.
            negative_prompt_embeds: Optional negative prompt embeddings (for CFG).
            guidance_scale: CFG scale for transformer (wan2.1 / wan2.2 high-noise).
            guidance_scale_2: CFG scale for transformer_2 (wan2.2 low-noise); defaults to
                ``guidance_scale``.
            image_embeds: Optional CLIP image embeddings (wan2.1 only).
            first_frame_mask: Timestep-expansion mask (Wan2.2 TI2V-5B); required exactly when
                the pipeline sets ``expand_timesteps``.
            boundary_timestep: Timestep threshold for switching transformers (wan2.2).
            t_next: Next timestep for the scheduler step; reduced like ``t``.
            next_latents: Optional target latents for log-prob computation.
            noise_level: Current noise level for SDE sampling.
            attention_kwargs: Optional kwargs for attention layers.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.

        Returns:
            UniPCMultistepSDESchedulerOutput with the requested fields. The scheduler step is
            skipped when only ``velocity`` is requested, ``compute_log_prob`` is False, and
            ``next_latents`` is None.
        """
        # 1. Prepare variables
        t = t[0] if t.ndim == 1 else t  # A scalar
        if t_next is not None:
            t_next = t_next[0] if t_next.ndim == 1 else t_next

        batch_size = latents.shape[0]
        dtype = (
            self.pipeline.transformer.dtype
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.dtype
        )
        device = latents.device
        if latent_condition is None:
            raise ValueError("Wan I2V forward requires realized VAE condition channels")
        if self.pipeline.config.expand_timesteps and first_frame_mask is None:
            raise ValueError("Wan I2V expand_timesteps forward requires first_frame_mask")
        if not self.pipeline.config.expand_timesteps and first_frame_mask is not None:
            raise ValueError("Wan I2V non-expanded forward must not receive first_frame_mask")
        if image_embeds is not None:
            image_embeds = normalize_wan_image_embeds(
                image_embeds,
                batch_size=batch_size,
            ).to(device=device, dtype=dtype)

        # Determine boundary timestep
        if boundary_timestep is None and self.pipeline.config.boundary_ratio is not None:
            boundary_timestep = (
                self.pipeline.config.boundary_ratio * self.scheduler.config.num_train_timesteps
            )
        # Determine which transformer to use
        if boundary_timestep is None or t >= boundary_timestep:
            pipeline_transformer = self.pipeline.transformer
            transformer = self.transformer
            current_guidance_scale = guidance_scale
        else:
            pipeline_transformer = self.pipeline.transformer_2
            transformer = self.transformer_2
            current_guidance_scale = (
                guidance_scale_2 if guidance_scale_2 is not None else guidance_scale
            )

        # Auto-detect CFG
        if current_guidance_scale > 1.0 and negative_prompt_embeds is None:
            logger.warning(
                "Passed `guidance_scale` > 1.0, but no `negative_prompt_embeds` provided. "
                "Classifier-free guidance will be disabled."
            )
        do_classifier_free_guidance = (
            negative_prompt_embeds is not None and current_guidance_scale > 1.0
        )

        # Prepare latent model input based on wan version
        if first_frame_mask is not None:
            # Expanded timesteps (Wan2.2 TI2V-5B): blend the condition in with the mask
            latent_model_input = (
                1 - first_frame_mask
            ) * latent_condition + first_frame_mask * latents
            latent_model_input = latent_model_input.to(dtype)
            temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
            timestep = temp_ts.unsqueeze(0).expand(batch_size, -1)
        else:
            # Wan2.1 and Wan2.2 I2V-A14B: concatenate condition channels
            latent_model_input = torch.cat([latents, latent_condition], dim=1).to(dtype)
            timestep = t.expand(batch_size)

        # Conditional forward pass
        with pipeline_transformer.cache_context("cond"):
            velocity = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

        # CFG: unconditional forward pass
        if do_classifier_free_guidance:
            with pipeline_transformer.cache_context("uncond"):
                velocity_uncond = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            velocity = velocity_uncond + current_guidance_scale * (velocity - velocity_uncond)

        if not compute_log_prob and next_latents is None and tuple(return_kwargs) == ("velocity",):
            return UniPCMultistepSDESchedulerOutput(velocity=velocity)

        # Scheduler step
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
