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

# src/flow_factory/models/ltx2/ltx2_i2av.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from diffusers.pipelines.ltx2.pipeline_ltx2_image2video import (
    LTX2ImageToVideoPipeline,
    rescale_noise_cfg,
)
from PIL import Image

from ...contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaRule,
    NegativePromptPolicy,
)
from ...hparams import *
from ...samples import (
    ComponentTimes,
    I2AVSample,
    LatentState,
    MultiModalStepOutput,
    NoisedState,
    StackedSampleBatch,
)
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SchedulerGroup,
    set_scheduler_timesteps,
)
from ...scheduler.flow_match_euler_discrete import calculate_shift
from ...utils.base import filter_kwargs
from ...utils.image import (
    ImageBatch,
    ImageSingle,
    MultiImageBatch,
    is_image_batch,
    is_multi_image_batch,
    standardize_image_batch,
)
from ...utils.logger_utils import setup_logger
from ...utils.media import pixels_to_unit, signed_unit_to_unit_pixels, unit_to_signed_unit_pixels
from ...utils.trajectory_collector import (
    TrajectoryIndicesType,
    create_callback_collector,
    create_trajectory_collector,
)
from ..abc import BaseAdapter
from ..output_state import DecodedMediaBatch, EncodedOutputState
from ..pipeline_contracts import IMAGE_FORMAT, audio_video_output_contract
from ._common import (
    LTX2_COMPONENT_ORDER,
    attach_ltx2_state_masks,
    build_ltx2_component_step_output,
    build_ltx2_full_component_schedule,
    build_ltx2_joint_forward_kwargs,
    build_ltx2_legacy_callback_view,
    build_ltx2_rollout_log_probs,
    build_ltx2_structured_trajectories,
    build_ltx2_training_component_times,
    combine_modality_log_prob,
    draw_ltx2_forward_process_noise,
    split_ltx2_callback_results,
    validate_i2av_forward_state_inputs,
)
from ._output import (
    LTX2_OFFLINE_FORWARD_OVERRIDES,
    LTX2AVOutputCodec,
    LTX2FirstFrameConditionPreparer,
    decode_ltx2_output_state,
    reduce_ltx2_flow_matching_objective_values,
    validate_ltx2_encoded_output_geometry,
)

logger = setup_logger(__name__)

CONDITION_IMAGE_SIZE = (512, 768)


@dataclass
class LTX2I2AVSample(I2AVSample):
    """Output class for LTX2 image-to-audio-video adapter.

    Inherits from I2AVSample (task-level) per constraint #14. All LTX2-specific
    fields are redeclared here (code duplication from LTX2Sample) since
    model-specific samples must not inherit from other model-specific samples.
    """

    _shared_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "height",
            "width",
            "num_frames",
            "frame_rate",
            "video_seq_len",
            "latent_index_map",
            "log_prob_index_map",
        }
    )

    num_frames: Optional[int] = None
    frame_rate: Optional[float] = None

    video_seq_len: Optional[int] = None

    conditioning_mask: Optional[torch.Tensor] = None  # (S_vid,) packed mask

    connector_prompt_embeds: Optional[torch.Tensor] = None  # (seq, D_video)
    connector_audio_prompt_embeds: Optional[torch.Tensor] = None  # (seq, D_audio)
    connector_attention_mask: Optional[torch.Tensor] = None  # (seq,)

    negative_connector_prompt_embeds: Optional[torch.Tensor] = None
    negative_connector_audio_prompt_embeds: Optional[torch.Tensor] = None
    negative_connector_attention_mask: Optional[torch.Tensor] = None


class LTX2_I2AV_Adapter(BaseAdapter):
    """Adapter for LTX2 image-to-audio-video generation.

    Conditions video generation on a first-frame image. Audio is generated
    jointly but is NOT conditioned by the image (pure noise init, scalar timestep).
    Video and audio are stepped by two twin SDE schedulers, so every transition
    produces a per-modality log_prob whose element-weighted combination is the
    joint policy log_prob; the fixed conditioning frame is never stepped and
    therefore carries no stochastic degrees of freedom.

    Per constraint #12, inherits directly from BaseAdapter (flat hierarchy).
    Shared logic with LTX2_T2AV_Adapter lives in the private ``_common`` and
    ``_output`` helpers or is duplicated.
    """

    component_load_dtype_defaults = {"audio_vae": torch.float32}
    pipeline_io_contract = audio_video_output_contract(
        negative_prompt=NegativePromptPolicy.OPTIONAL,
        input_rules=(
            InputMediaRule(
                format=IMAGE_FORMAT,
                min_count=1,
                max_count=1,
                slots=("first_frame",),
                required_slots=("first_frame",),
            ),
        ),
        geometry_source=GeometrySource.CONFIGURED,
        batch_capability=BatchCapability.UNIFORM,
    )
    offline_training_forward_overrides = LTX2_OFFLINE_FORWARD_OVERRIDES

    supports_diffusers_cache = True
    trajectory_component_order: ClassVar[Tuple[str, ...]] = LTX2_COMPONENT_ORDER
    preprocess_cache_version: ClassVar[str] = "ltx2-i2av-condition-unit-pixels-v1"

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: LTX2ImageToVideoPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler
        self.audio_scheduler: FlowMatchEulerDiscreteSDEScheduler
        self._has_warned_multi_image = False

    def load_pipeline(self) -> LTX2ImageToVideoPipeline:
        return self._load_diffusers_pipeline(
            LTX2ImageToVideoPipeline,
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False,
        )

    def build_scheduler_group(self) -> SchedulerGroup:
        """Build the ordered video/audio scheduler group and its audio twin.

        Audio is sampled with the same SDE dynamics as video so that both
        modalities form a single joint policy with one per-step log_prob over
        both. A dedicated instance is still required because
        scheduler.step() mutates internal state (step_index), which would
        conflict if shared with the video scheduler. ``load_scheduler`` rebuilds
        an independent scheduler from the same pipeline scheduler + scheduler
        args used for ``self.scheduler`` (which ``BaseAdapter.__init__`` has
        already installed when it calls this hook).

        Returns:
            Scheduler group whose primary ``"video"`` component is the canonical
            pipeline scheduler.
        """
        self.audio_scheduler = self.load_scheduler()
        return SchedulerGroup(
            {"video": self.scheduler, "audio": self.audio_scheduler},
            primary_name="video",
        )

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for LTX2 transformer.

        Verified against LTX2VideoTransformerBlock.named_modules():
        28 Linear layers per block (6 attention groups x 4 projections + 2 FFN groups x 2 layers).
        """
        return [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            "audio_attn1.to_q",
            "audio_attn1.to_k",
            "audio_attn1.to_v",
            "audio_attn1.to_out.0",
            "audio_attn2.to_q",
            "audio_attn2.to_k",
            "audio_attn2.to_v",
            "audio_attn2.to_out.0",
            "audio_to_video_attn.to_q",
            "audio_to_video_attn.to_k",
            "audio_to_video_attn.to_v",
            "audio_to_video_attn.to_out.0",
            "video_to_audio_attn.to_q",
            "video_to_audio_attn.to_k",
            "video_to_audio_attn.to_v",
            "video_to_audio_attn.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "audio_ff.net.0.proj",
            "audio_ff.net.2",
        ]

    @property
    def preprocessing_modules(self) -> List[str]:
        """Components needed for offline preprocessing (text encoding + connectors)."""
        return ["text_encoders", "connectors"]

    @property
    def inference_modules(self) -> List[str]:
        """Components needed during inference and training forward."""
        return ["transformer", "vae", "audio_vae", "connectors", "vocoder"]

    def build_condition_state_preparer(self) -> LTX2FirstFrameConditionPreparer:
        """Declare one-time deterministic first-frame condition encoding."""
        return LTX2FirstFrameConditionPreparer(self)

    def build_output_state_codec(self) -> LTX2AVOutputCodec:
        """Declare deterministic on-the-fly encoding for paired AV targets."""
        return LTX2AVOutputCodec(self, conditioned=True)

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Validate AV geometry and the pinned first-frame binding."""
        validate_ltx2_encoded_output_geometry(
            self,
            media_batch,
            condition,
            encoded,
            conditioned=True,
        )

    def _decode_output_state(
        self,
        encoded: EncodedOutputState,
        *,
        output_type: Literal["pil", "pt", "np"],
    ) -> Any:
        """Decode both components through the joint AV ``decode_latents``."""
        return decode_ltx2_output_state(self, encoded, output_type=output_type)

    def _reduce_flow_matching_objective_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        state: Optional[LatentState] = None,
    ) -> torch.Tensor:
        """Sum official video/audio means without changing online reducers."""
        return reduce_ltx2_flow_matching_objective_values(
            self,
            values,
            state=state,
        )

    def _check_inputs(
        self,
        height: int,
        width: int,
        num_frames: int,
        images=None,
        condition_images=None,
        prompt: Optional[Union[str, List[str]]] = None,
        connector_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        audio_guidance_scale: Optional[float] = None,
        stg_scale: float = 0.0,
        audio_stg_scale: Optional[float] = None,
        spatio_temporal_guidance_blocks: Optional[List[int]] = None,
    ) -> int:
        """Validate generation parameters and return VAE-compatible num_frames.

        Extends T2AV validation with image presence check.

        Returns:
            num_frames rounded down to the nearest VAE-temporal-compatible value.
        """
        vae_spatial = self.pipeline.vae_spatial_compression_ratio
        vae_temporal = self.pipeline.vae_temporal_compression_ratio

        if images is None and condition_images is None:
            raise ValueError(
                "I2AV requires conditioning images. Provide either `images` "
                "(raw PIL/tensor) or `condition_images` (preprocessed [0, 1] tensor)."
            )

        if prompt is None and connector_prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `connector_prompt_embeds`. "
                "Cannot leave both undefined."
            )

        do_cfg = guidance_scale > 1.0 or (audio_guidance_scale or guidance_scale) > 1.0
        if (
            do_cfg
            and connector_prompt_embeds is not None
            and negative_connector_prompt_embeds is None
        ):
            raise ValueError(
                "guidance_scale > 1.0 requires negative_connector_prompt_embeds "
                "when using pre-encoded embeddings. Either provide negative "
                "embeddings or set guidance_scale <= 1.0."
            )

        if height % vae_spatial != 0 or width % vae_spatial != 0:
            raise ValueError(
                f"height ({height}) and width ({width}) must be divisible by "
                f"vae_spatial_compression_ratio ({vae_spatial})."
            )
        if (
            (stg_scale > 0.0) or ((audio_stg_scale or 0.0) > 0.0)
        ) and not spatio_temporal_guidance_blocks:
            raise ValueError(
                "Spatio-Temporal Guidance (STG) is enabled (stg_scale > 0) but no "
                "spatio_temporal_guidance_blocks specified. Recommended: [29] for LTX-2."
            )

        if (num_frames - 1) % vae_temporal != 0:
            num_frames = (num_frames - 1) // vae_temporal * vae_temporal + 1
            logger.warning(
                f"num_frames rounded to {num_frames} (must satisfy (num_frames - 1) % {vae_temporal} == 0)."
            )
        return max(num_frames, 1)

    def _encode_text(
        self,
        text: List[str],
        max_sequence_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple:
        """Tokenize and encode text via Gemma3, returning (input_ids, embeds, mask).

        Inlines the logic from ``LTX2Pipeline._get_gemma_prompt_embeds`` so that
        ``input_ids`` (used as ``prompt_ids`` / ``negative_prompt_ids`` for reward
        grouping) are obtained from the same tokenization pass as the embeddings.
        """
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        tokenizer = self.pipeline.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text = [t.strip() for t in text]
        tok_out = tokenizer(
            text,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tok_out.input_ids.to(device)
        attention_mask = tok_out.attention_mask.to(device)

        enc_out = self.pipeline.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = torch.stack(enc_out.hidden_states, dim=-1)
        embeds = hidden.flatten(2, 3).to(dtype=dtype)

        return input_ids, embeds, attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 4.0,
        audio_guidance_scale: Optional[float] = None,
        max_sequence_length: int = 1024,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Encode text prompts into connector embeddings for video and audio streams."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        batch_size = len(prompt)
        do_classifier_free_guidance = (
            guidance_scale > 1.0 or (audio_guidance_scale or guidance_scale) > 1.0
        )

        prompt_ids, prompt_embeds, prompt_attention_mask = self._encode_text(
            prompt,
            max_sequence_length,
            device,
            dtype,
        )

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            if len(negative_prompt) != batch_size:
                raise ValueError(
                    f"negative_prompt batch size {len(negative_prompt)} != "
                    f"prompt batch size {batch_size}"
                )
            negative_prompt_ids, negative_prompt_embeds, negative_prompt_attention_mask = (
                self._encode_text(negative_prompt, max_sequence_length, device, dtype)
            )
            combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            combined_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )
        else:
            negative_prompt_ids = None
            combined_embeds = prompt_embeds
            combined_mask = prompt_attention_mask

        connector_out, connector_audio_out, connector_mask = self.pipeline.connectors(
            combined_embeds,
            combined_mask,
        )

        if do_classifier_free_guidance:
            neg_conn, pos_conn = connector_out.chunk(2)
            neg_audio_conn, pos_audio_conn = connector_audio_out.chunk(2)
            neg_conn_mask, pos_conn_mask = connector_mask.chunk(2)
            results: Dict[str, Optional[torch.Tensor]] = {
                "prompt_ids": prompt_ids,
                "negative_prompt_ids": negative_prompt_ids,
                "connector_prompt_embeds": pos_conn,
                "connector_audio_prompt_embeds": pos_audio_conn,
                "connector_attention_mask": pos_conn_mask,
                "negative_connector_prompt_embeds": neg_conn,
                "negative_connector_audio_prompt_embeds": neg_audio_conn,
                "negative_connector_attention_mask": neg_conn_mask,
            }
        else:
            results = {
                "prompt_ids": prompt_ids,
                "connector_prompt_embeds": connector_out,
                "connector_audio_prompt_embeds": connector_audio_out,
                "connector_attention_mask": connector_mask,
            }

        return results

    def _standardize_image_input(
        self,
        images: Union[ImageSingle, ImageBatch, MultiImageBatch],
        output_type: Literal["pil", "pt", "np"] = "pil",
    ) -> ImageBatch:
        """Standardize image input to desired output type.

        Handles MultiImageBatch (nested List[List[Image]]) by extracting the
        first image per sample, since LTX2 I2AV only supports single-image
        conditioning.
        """
        if isinstance(images, Image.Image):
            images = [images]
        elif is_multi_image_batch(images):
            if any(len(batch) > 1 for batch in images) and not self._has_warned_multi_image:
                self._has_warned_multi_image = True
                logger.warning(
                    "Multiple condition images are not supported for LTX2 I2AV. "
                    "Only the first image of each batch will be used."
                )
            images = [batch[0] for batch in images]
        return standardize_image_batch(images, output_type=output_type)

    def encode_image(
        self,
        images: Union[ImageSingle, ImageBatch, MultiImageBatch],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        height: int = 512,
        width: int = 768,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        """Preprocess conditioning images to ``[0, 1]`` pixel tensors at generation resolution.

        VAE encoding is deferred to inference(), which maps these pixels back to the
        processor's ``[-1, 1]`` VAE input for prepare_latents(image=...).
        Uses height/width (generation resolution) for preprocessing, NOT
        condition_image_size. LTX2 I2V encodes the image as the first video frame
        inside prepare_latents, so it must match generation resolution.

        ``condition_image_size`` is kept in the signature for API compatibility
        with other adapters (Flux2, Qwen) but is not used.
        """
        device = device or self.device
        images = self._standardize_image_input(images, output_type="pil")
        processed = self.pipeline.video_processor.preprocess(images, height=height, width=width)
        return {"condition_images": signed_unit_to_unit_pixels(processed).to(device=device)}

    def encode_video(self, videos, **kwargs):
        """LTX2 I2AV does not use video conditioning."""
        return None

    def preprocess_func(
        self,
        prompt: List[str],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        images: Optional[List[Union[Image.Image, List[Image.Image]]]] = None,
        height: int = 512,
        width: int = 768,
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        max_sequence_length: int = 1024,
        guidance_scale: float = 4.0,
        audio_guidance_scale: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """Preprocess inputs for LTX2 I2AV (batched processing).

        Overrides BaseAdapter.preprocess_func because I2AV conditions on exactly one
        image per sample: the nested ``MultiImageBatch`` is flattened to its first
        image before ``encode_image`` prepares it for the VAE.
        """
        if images is not None:
            if isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
                images = [[img] for img in images]
            has_images = True
        else:
            has_images = False

        batch = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            max_sequence_length=max_sequence_length,
        )

        if has_images:
            flat_images = [img_list[0] for img_list in images]
            image_dict = self.encode_image(
                flat_images,
                height=height,
                width=width,
            )
            batch.update(image_dict)

        return batch

    def decode_latents(
        self,
        video_latents: torch.Tensor,
        audio_latents: Optional[torch.Tensor] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        decode_timestep: float = 0.0,
        decode_noise_scale: Optional[float] = None,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        """Decode packed latents to video frames and audio waveform."""
        device = video_latents.device
        batch_size = video_latents.shape[0]
        vae = self.pipeline.vae
        patch_size = self.pipeline.transformer_spatial_patch_size
        patch_size_t = self.pipeline.transformer_temporal_patch_size
        vae_spatial = self.pipeline.vae_spatial_compression_ratio
        vae_temporal = self.pipeline.vae_temporal_compression_ratio

        latent_h = height // vae_spatial
        latent_w = width // vae_spatial
        latent_f = (num_frames - 1) // vae_temporal + 1

        vid = self.pipeline._unpack_latents(
            video_latents, latent_f, latent_h, latent_w, patch_size, patch_size_t
        )
        vid = self.pipeline._denormalize_latents(
            vid, vae.latents_mean, vae.latents_std, vae.config.scaling_factor
        )
        if not vae.config.timestep_conditioning:
            vae_timestep = None
        else:
            noise = torch.randn_like(vid)
            _dt = (
                [decode_timestep] * batch_size
                if not isinstance(decode_timestep, list)
                else decode_timestep
            )
            _dns = (
                _dt
                if decode_noise_scale is None
                else (
                    [decode_noise_scale] * batch_size
                    if not isinstance(decode_noise_scale, list)
                    else decode_noise_scale
                )
            )
            vae_timestep = torch.tensor(_dt, device=device, dtype=vid.dtype)
            _dns_t = torch.tensor(_dns, device=device, dtype=vid.dtype)[:, None, None, None, None]
            vid = (1 - _dns_t) * vid + _dns_t * noise
        vid = vid.to(vae.dtype)
        video = vae.decode(vid, vae_timestep, return_dict=False)[0]
        video = self.pipeline.video_processor.postprocess_video(video, output_type=output_type)

        audio = None
        if audio_latents is not None:
            audio_vae = self.pipeline.audio_vae
            mel_compression = self.pipeline.audio_vae_mel_compression_ratio
            temporal_compression = self.pipeline.audio_vae_temporal_compression_ratio
            num_mel_bins = (
                audio_vae.config.mel_bins
                if getattr(self.pipeline, "audio_vae", None) is not None
                else 64
            )
            latent_mel_bins = num_mel_bins // mel_compression

            duration_s = num_frames / frame_rate
            sr = self.pipeline.audio_sampling_rate
            hop = self.pipeline.audio_hop_length
            audio_num_frames = round(duration_s * sr / hop / temporal_compression)

            aud = self.pipeline._denormalize_audio_latents(
                audio_latents, audio_vae.latents_mean, audio_vae.latents_std
            )
            aud = self.pipeline._unpack_audio_latents(
                aud, audio_num_frames, num_mel_bins=latent_mel_bins
            )
            aud = aud.to(audio_vae.dtype)
            mel = audio_vae.decode(aud, return_dict=False)[0]
            audio = self.pipeline.vocoder(mel)

        return video, audio

    def convert_velocity_to_x0(
        self,
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Convert velocity-space prediction to x0: x_0 = x_t - sigma * v."""
        return sample - velocity * sigma

    def convert_x0_to_velocity(
        self,
        sample: torch.Tensor,
        x0: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Convert x0 prediction back to velocity: v = (x_t - x_0) / sigma."""
        return (sample - x0) / sigma

    def forward(
        self,
        t: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        video_seq_len: Optional[int] = None,
        conditioning_mask: Optional[torch.Tensor] = None,
        connector_prompt_embeds: Optional[torch.Tensor] = None,
        connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
        connector_attention_mask: Optional[torch.Tensor] = None,
        negative_connector_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_attention_mask: Optional[torch.Tensor] = None,
        guidance_scale: float = 4.0,
        audio_guidance_scale: Optional[float] = None,
        guidance_rescale: float = 0.0,
        audio_guidance_rescale: Optional[float] = None,
        stg_scale: float = 0.0,
        audio_stg_scale: Optional[float] = None,
        spatio_temporal_guidance_blocks: Optional[List[int]] = None,
        modality_scale: float = 1.0,
        audio_modality_scale: Optional[float] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        audio_num_frames: Optional[int] = None,
        video_coords: Optional[torch.Tensor] = None,
        audio_coords: Optional[torch.Tensor] = None,
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = ["next_latents", "log_prob", "velocity"],
        use_cross_timestep: bool = False,
        preserve_raw_model_velocity: bool = False,
        # Component-return mode, owned by ``_forward_state``
        _return_components: bool = False,
        **kwargs,
    ) -> Union[FlowMatchEulerDiscreteSDESchedulerOutput, MultiModalStepOutput]:
        """Single denoising step with I2AV conditioning mask and frame-slicing.

        Compared to T2AV forward():
        - CFG-doubles conditioning_mask internally alongside latents/coords
        - Builds per-token video timestep: ts * (1 - conditioning_mask)
        - Scheduler step runs on generated frames only (1:), preserving frame 0
        - next_latents for training is also sliced to exclude frame 0

        The default return concatenates both modalities into one legacy
        ``SDESchedulerOutput``. ``_return_components`` is reserved for
        :meth:`_forward_state`, which needs the per-modality latents, statistics
        and log-probs captured before that concatenation.
        """
        batch_size = latents.shape[0]
        device = latents.device

        # Normalize timestep to a 1-D (B,) tensor.
        # Inference passes a 0-D scalar; training passes (B,) with potentially
        # distinct values per sample (after K-repeat sampling). A singleton (1,)
        # tensor is also accepted and broadcast to (B,).
        if t.ndim == 0 or (t.ndim == 1 and t.shape[0] == 1):
            t = t.expand(batch_size)
        elif t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(
                f"expected `t` to be a scalar, a (1,) tensor, or a 1-D tensor of shape ({batch_size},), "
                f"got shape {tuple(t.shape)}"
            )

        sigma = (t / 1000).view(-1, 1, 1)

        audio_guidance_scale = audio_guidance_scale or guidance_scale
        audio_stg_scale = audio_stg_scale or stg_scale
        audio_modality_scale = audio_modality_scale or modality_scale
        audio_guidance_rescale = audio_guidance_rescale or guidance_rescale

        if (
            guidance_scale > 1.0 or audio_guidance_scale > 1.0
        ) and negative_connector_prompt_embeds is None:
            logger.warning(
                "Passed `guidance_scale` > 1.0, but no `negative_connector_prompt_embeds` provided. "
                "Classifier-free guidance will be disabled."
            )
        do_cfg = (
            guidance_scale > 1.0 or audio_guidance_scale > 1.0
        ) and negative_connector_prompt_embeds is not None
        do_stg = (
            stg_scale > 0.0 or audio_stg_scale > 0.0
        ) and spatio_temporal_guidance_blocks is not None
        do_modality_isolation = modality_scale > 1.0 or audio_modality_scale > 1.0

        vae_spatial = self.pipeline.vae_spatial_compression_ratio
        vae_temporal = self.pipeline.vae_temporal_compression_ratio
        latent_h = height // vae_spatial
        latent_w = width // vae_spatial
        latent_f = (num_frames - 1) // vae_temporal + 1
        patch_size = self.pipeline.transformer_spatial_patch_size
        patch_size_t = self.pipeline.transformer_temporal_patch_size
        if video_seq_len is None:
            video_seq_len = latent_f * latent_h * latent_w
        if audio_num_frames is None:
            duration_s = num_frames / frame_rate
            sr = self.pipeline.audio_sampling_rate
            hop = self.pipeline.audio_hop_length
            tc = self.pipeline.audio_vae_temporal_compression_ratio
            audio_num_frames = round(duration_s * sr / hop / tc)

        video_latents = latents[:, :video_seq_len]
        audio_latents = latents[:, video_seq_len:]
        if next_latents is not None:
            video_next = next_latents[:, :video_seq_len]
            audio_next = next_latents[:, video_seq_len:]
        else:
            video_next = audio_next = None

        if video_coords is None:
            video_coords = self.pipeline.transformer.rope.prepare_video_coords(
                batch_size, latent_f, latent_h, latent_w, device, fps=frame_rate
            )
        if audio_coords is None:
            audio_coords = self.pipeline.transformer.audio_rope.prepare_audio_coords(
                batch_size, audio_num_frames, device
            )

        dtype = self.pipeline.transformer.dtype

        # [I2AV] CFG-double conditioning_mask (per-token timesteps are built below)
        if conditioning_mask is not None:
            cm_single = conditioning_mask
            if do_cfg:
                cm = torch.cat([conditioning_mask, conditioning_mask])
            else:
                cm = conditioning_mask
        else:
            cm_single = None
            cm = None

        transformer_kwargs = dict(
            num_frames=latent_f,
            height=latent_h,
            width=latent_w,
            fps=frame_rate,
            audio_num_frames=audio_num_frames,
            use_cross_timestep=use_cross_timestep,
            attention_kwargs=None,
            return_dict=False,
        )

        # 1. Prepare CFG inputs and run main transformer forward
        if do_cfg:
            lat_in = torch.cat([video_latents, video_latents])
            aud_in = torch.cat([audio_latents, audio_latents])
            text_in = torch.cat([negative_connector_prompt_embeds, connector_prompt_embeds])
            audio_text_in = torch.cat(
                [negative_connector_audio_prompt_embeds, connector_audio_prompt_embeds]
            )
            mask_in = torch.cat([negative_connector_attention_mask, connector_attention_mask])
            vid_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            aud_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))
            # Duplicate timesteps to match torch.cat([lat, lat]) ordering: [t0..tB-1, t0..tB-1]
            ts = torch.cat([t, t])
        else:
            lat_in = video_latents
            aud_in = audio_latents
            text_in = connector_prompt_embeds
            audio_text_in = connector_audio_prompt_embeds
            mask_in = connector_attention_mask
            vid_coords, aud_coords = video_coords, audio_coords
            ts = t

        # [I2AV] Per-token video timestep (conditioned tokens see t=0)
        if cm is not None:
            video_ts = ts.unsqueeze(-1) * (1 - cm)
            audio_ts = ts
        else:
            video_ts = ts
            audio_ts = ts

        with self.pipeline.transformer.cache_context("cond_uncond"):
            video_pred, audio_pred = self.transformer(
                hidden_states=lat_in.to(dtype),
                audio_hidden_states=aud_in.to(dtype),
                encoder_hidden_states=text_in,
                audio_encoder_hidden_states=audio_text_in,
                timestep=video_ts,
                audio_timestep=audio_ts,
                sigma=ts,
                encoder_attention_mask=mask_in,
                audio_encoder_attention_mask=mask_in,
                video_coords=vid_coords,
                audio_coords=aud_coords,
                **transformer_kwargs,
            )
        video_pred = video_pred.float()
        audio_pred = audio_pred.float()
        raw_video_pred = video_pred
        raw_audio_pred = audio_pred
        uses_x0_guidance = (
            do_cfg
            or do_stg
            or do_modality_isolation
            or guidance_rescale > 0
            or audio_guidance_rescale > 0
        )

        # 2. Convert to x0-space and compute guidance deltas
        if do_cfg:
            video_uncond, video_cond = video_pred.chunk(2)
            video_x0 = self.convert_velocity_to_x0(video_latents, video_cond, sigma)
            video_x0_uncond = self.convert_velocity_to_x0(video_latents, video_uncond, sigma)
            video_cfg_delta = (guidance_scale - 1) * (video_x0 - video_x0_uncond)

            audio_uncond, audio_cond = audio_pred.chunk(2)
            audio_x0 = self.convert_velocity_to_x0(audio_latents, audio_cond, sigma)
            audio_x0_uncond = self.convert_velocity_to_x0(audio_latents, audio_uncond, sigma)
            audio_cfg_delta = (audio_guidance_scale - 1) * (audio_x0 - audio_x0_uncond)
        else:
            video_x0 = self.convert_velocity_to_x0(video_latents, video_pred, sigma)
            audio_x0 = self.convert_velocity_to_x0(audio_latents, audio_pred, sigma)
            video_cfg_delta = audio_cfg_delta = 0

        pos_text = connector_prompt_embeds
        pos_audio_text = connector_audio_prompt_embeds
        pos_mask = connector_attention_mask
        pos_ts = t

        # [I2AV] Per-token timestep for STG/modality (single-batch, positive-prompt only)
        if cm_single is not None:
            pos_video_ts = pos_ts.unsqueeze(-1) * (1 - cm_single)
        else:
            pos_video_ts = pos_ts

        # 3. STG: extra transformer forward with perturbed blocks
        video_stg_delta = audio_stg_delta = 0
        if do_stg:
            with self.pipeline.transformer.cache_context("uncond_stg"):
                stg_video, stg_audio = self.transformer(
                    hidden_states=video_latents.to(dtype),
                    audio_hidden_states=audio_latents.to(dtype),
                    encoder_hidden_states=pos_text,
                    audio_encoder_hidden_states=pos_audio_text,
                    timestep=pos_video_ts,
                    audio_timestep=pos_ts,
                    sigma=pos_ts,
                    encoder_attention_mask=pos_mask,
                    audio_encoder_attention_mask=pos_mask,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    isolate_modalities=False,
                    spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
                    perturbation_mask=None,
                    **transformer_kwargs,
                )
            stg_video_x0 = self.convert_velocity_to_x0(video_latents, stg_video.float(), sigma)
            stg_audio_x0 = self.convert_velocity_to_x0(audio_latents, stg_audio.float(), sigma)
            video_stg_delta = stg_scale * (video_x0 - stg_video_x0)
            audio_stg_delta = audio_stg_scale * (audio_x0 - stg_audio_x0)

        # 4. Modality Isolation: extra forward with cross-modal attn disabled
        video_modality_delta = audio_modality_delta = 0
        if do_modality_isolation:
            with self.pipeline.transformer.cache_context("uncond_modality"):
                iso_video, iso_audio = self.transformer(
                    hidden_states=video_latents.to(dtype),
                    audio_hidden_states=audio_latents.to(dtype),
                    encoder_hidden_states=pos_text,
                    audio_encoder_hidden_states=pos_audio_text,
                    timestep=pos_video_ts,
                    audio_timestep=pos_ts,
                    sigma=pos_ts,
                    encoder_attention_mask=pos_mask,
                    audio_encoder_attention_mask=pos_mask,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    isolate_modalities=True,
                    spatio_temporal_guidance_blocks=None,
                    perturbation_mask=None,
                    **transformer_kwargs,
                )
            iso_video_x0 = self.convert_velocity_to_x0(video_latents, iso_video.float(), sigma)
            iso_audio_x0 = self.convert_velocity_to_x0(audio_latents, iso_audio.float(), sigma)
            video_modality_delta = (modality_scale - 1) * (video_x0 - iso_video_x0)
            audio_modality_delta = (audio_modality_scale - 1) * (audio_x0 - iso_audio_x0)

        # 5. Combine all guidance deltas in x0-space
        video_x0_guided = video_x0 + video_cfg_delta + video_stg_delta + video_modality_delta
        audio_x0_guided = audio_x0 + audio_cfg_delta + audio_stg_delta + audio_modality_delta

        if guidance_rescale > 0:
            video_x0_guided = rescale_noise_cfg(
                video_x0_guided, video_x0, guidance_rescale=guidance_rescale
            )
        if audio_guidance_rescale > 0:
            audio_x0_guided = rescale_noise_cfg(
                audio_x0_guided, audio_x0, guidance_rescale=audio_guidance_rescale
            )

        # 6. Convert back to velocity; with `preserve_raw_model_velocity`, keep the raw
        # prediction when no x0-space guidance transform ran.
        # The algebraic round trip is numerically unstable near sigma=0 and is
        # unnecessary for SFT/offline-DPO's neutral-guidance forward.
        if uses_x0_guidance or not preserve_raw_model_velocity:
            video_pred = self.convert_x0_to_velocity(
                video_latents,
                video_x0_guided,
                sigma,
            )
            audio_pred = self.convert_x0_to_velocity(
                audio_latents,
                audio_x0_guided,
                sigma,
            )
        else:
            video_pred = raw_video_pred
            audio_pred = raw_audio_pred

        velocity_only = (
            not compute_log_prob and next_latents is None and tuple(return_kwargs) == ("velocity",)
        )
        if velocity_only:
            if _return_components:
                return MultiModalStepOutput(
                    velocity=LatentState({"video": video_pred, "audio": audio_pred})
                )
            return FlowMatchEulerDiscreteSDESchedulerOutput(
                velocity=torch.cat([video_pred, audio_pred], dim=1)
            )

        # 7. [I2AV] Video scheduler step with frame-slicing
        if conditioning_mask is not None:
            video_pred_5d = self.pipeline._unpack_latents(
                video_pred, latent_f, latent_h, latent_w, patch_size, patch_size_t
            )
            video_latents_5d = self.pipeline._unpack_latents(
                video_latents, latent_f, latent_h, latent_w, patch_size, patch_size_t
            )

            gen_pred = video_pred_5d[:, :, 1:]
            gen_lats = video_latents_5d[:, :, 1:]
            # Only the generated frames (1:) are stepped, so they are the video
            # contribution to the joint log_prob (the conditioning frame is fixed).
            n_video_stepped = gen_lats[0].numel()

            video_next_gen = None
            if video_next is not None:
                video_next_5d = self.pipeline._unpack_latents(
                    video_next, latent_f, latent_h, latent_w, patch_size, patch_size_t
                )
                video_next_gen = video_next_5d[:, :, 1:]

            # Make sure `next_latents` is included in return_kwargs
            return_kwargs = list({"next_latents"} | set(return_kwargs))
            video_output = self.scheduler.step(
                velocity=gen_pred,
                timestep=t,
                latents=gen_lats,
                timestep_next=t_next,
                next_latents=video_next_gen,
                compute_log_prob=compute_log_prob,
                return_dict=True,
                return_kwargs=return_kwargs,
                noise_level=noise_level,
            )

            stepped_5d = video_output.next_latents
            next_5d = torch.cat([video_latents_5d[:, :, :1], stepped_5d], dim=2)
            video_output.next_latents = self.pipeline._pack_latents(
                next_5d, patch_size, patch_size_t
            )
            if video_output.velocity is not None:
                pred_5d = torch.cat([video_pred_5d[:, :, :1], video_output.velocity], dim=2)
                video_output.velocity = self.pipeline._pack_latents(
                    pred_5d, patch_size, patch_size_t
                )
            if video_output.next_latents_mean is not None:
                mean_5d = torch.cat(
                    [video_latents_5d[:, :, :1], video_output.next_latents_mean], dim=2
                )
                video_output.next_latents_mean = self.pipeline._pack_latents(
                    mean_5d, patch_size, patch_size_t
                )
        else:
            video_output = self.scheduler.step(
                velocity=video_pred,
                timestep=t,
                latents=video_latents,
                timestep_next=t_next,
                next_latents=video_next,
                compute_log_prob=compute_log_prob,
                return_dict=True,
                return_kwargs=return_kwargs,
                noise_level=noise_level,
            )
            n_video_stepped = video_latents[0].numel()

        # 8. Audio: SDE scheduler step (twin of video, with log_prob)
        audio_output = self.audio_scheduler.step(
            velocity=audio_pred,
            timestep=t,
            latents=audio_latents,
            timestep_next=t_next,
            next_latents=audio_next,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            return_kwargs=return_kwargs,
            noise_level=noise_level,
        )

        # 9. Component view, captured before the legacy concatenation mutates it
        if _return_components:
            return build_ltx2_component_step_output(
                self,
                video_output=video_output,
                audio_output=audio_output,
                video_velocity=video_pred,
                audio_velocity=audio_pred,
                n_video=n_video_stepped,
                n_audio=audio_latents[0].numel(),
                compute_log_prob=compute_log_prob,
            )

        # 10. Concatenate back into unified latents
        if video_output.next_latents is not None and audio_output.next_latents is not None:
            video_output.next_latents = torch.cat(
                [video_output.next_latents, audio_output.next_latents],
                dim=1,
            )
        if (
            video_output.next_latents_mean is not None
            and getattr(audio_output, "next_latents_mean", None) is not None
        ):
            video_output.next_latents_mean = torch.cat(
                [video_output.next_latents_mean, audio_output.next_latents_mean],
                dim=1,
            )
        if video_output.velocity is not None:
            video_output.velocity = torch.cat(
                [video_output.velocity, audio_pred],
                dim=1,
            )

        # 11. Combine per-step log_prob across modalities
        # Joint transition p(v,a|z_t) = p(v|z_t) p(a|z_t); the element-weighted mean
        # reproduces what a single scheduler over the concatenated [video|audio] latent
        # would return. `n_video_stepped` counts only the generated video frames (the
        # conditioning frame is fixed and not stepped).
        if (
            compute_log_prob
            and video_output.log_prob is not None
            and audio_output.log_prob is not None
        ):
            video_output.log_prob = combine_modality_log_prob(
                video_output.log_prob,
                audio_output.log_prob,
                n_video=n_video_stepped,
                n_audio=audio_latents[0].numel(),
            )

        return video_output

    # ============================== Trajectory Hooks ==============================

    def build_training_component_times(
        self,
        primary_timesteps: torch.Tensor,
        *,
        batch: Optional[StackedSampleBatch] = None,
    ) -> ComponentTimes:
        """Mirror one sampled coordinate onto the twin video/audio schedules.

        Args:
            primary_timesteps: Primary scheduler coordinates of shape ``(B,)``.
            batch: Unused; both components share the primary coordinate.

        Returns:
            Video/audio timesteps and sigmas in authoritative component order.
        """
        return build_ltx2_training_component_times(self, primary_timesteps)

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> NoisedState:
        """Draw video noise then audio noise, then apply both deterministically.

        The fixed conditioning tokens stay clean because the application hook
        honours ``clean_state.active_masks``.

        Args:
            clean_state: Clean latent state in authoritative component order.
            times: Component times including each component's current sigma.
            generator: Optional generator shared by both ordered draws.

        Returns:
            Noised state, target velocity, and the sampled noise.
        """
        return draw_ltx2_forward_process_noise(self, clean_state, times, generator=generator)

    def _forward_state(
        self,
        *,
        batch: StackedSampleBatch,
        state: LatentState,
        times: ComponentTimes,
        next_state: Optional[LatentState],
        compute_log_prob: bool,
        return_fields: Tuple[str, ...],
        noise_level: Optional[float],
        forward_kwargs: Mapping[str, Any],
    ) -> MultiModalStepOutput:
        """Pack the ordered video/audio state for ``forward`` and unpack components.

        The state masks are the single source of truth for which video tokens are
        stochastic, so they are checked against the batch conditioning mask that
        ``forward`` uses for its first-frame slicing.

        Args:
            batch: Collated batch supplying conditioning and ``video_seq_len``.
            state: Current state in authoritative component order.
            times: Current and next times in authoritative component order.
            next_state: Optional stored next state in authoritative component order.
            compute_log_prob: Whether to compute transition log probabilities.
            return_fields: Scheduler output fields requested from ``forward``.
            noise_level: Scheduler noise-level override.
            forward_kwargs: Model-conditioning arguments resolved by the wrapper.

        Returns:
            Ordered component step output carrying the input active masks.
        """
        validate_i2av_forward_state_inputs(
            self, batch=batch, state=state, times=times, next_state=next_state
        )
        output = self.forward(
            **build_ltx2_joint_forward_kwargs(
                self,
                batch=batch,
                state=state,
                times=times,
                next_state=next_state,
                compute_log_prob=compute_log_prob,
                return_fields=return_fields,
                noise_level=noise_level,
                forward_kwargs=forward_kwargs,
            )
        )
        return attach_ltx2_state_masks(state, output)

    @torch.no_grad()
    def inference(
        self,
        images: Optional[MultiImageBatch] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 40,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 4.0,
        audio_guidance_scale: Optional[float] = None,
        guidance_rescale: float = 0.0,
        audio_guidance_rescale: Optional[float] = None,
        noise_scale: float = 0.0,
        stg_scale: float = 0.0,
        audio_stg_scale: Optional[float] = None,
        spatio_temporal_guidance_blocks: Optional[List[int]] = None,
        modality_scale: float = 1.0,
        audio_modality_scale: Optional[float] = None,
        use_cross_timestep: bool = False,
        generator: Optional[torch.Generator] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        connector_prompt_embeds: Optional[torch.Tensor] = None,
        connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
        connector_attention_mask: Optional[torch.Tensor] = None,
        negative_connector_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_attention_mask: Optional[torch.Tensor] = None,
        condition_images: Optional[torch.Tensor] = None,
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        decode_timestep: float = 0.0,
        decode_noise_scale: Optional[float] = None,
        max_sequence_length: int = 1024,
        compute_log_prob: bool = True,
        trajectory_indices: TrajectoryIndicesType = "all",
        extra_call_back_kwargs: List[str] = [],
        **kwargs,
    ) -> List[LTX2I2AVSample]:
        """Full denoising inference loop for LTX2 image-to-audio-video generation.

        Supports two image input paths:
        - Raw ``images`` (PIL/tensor): calls encode_image() inline
        - Pre-encoded ``condition_images`` (from preprocess_func): ``[0, 1]`` pixels
          used directly
        """
        device = self.device

        # 0. Validate inputs
        num_frames = self._check_inputs(
            height,
            width,
            num_frames,
            images=images,
            condition_images=condition_images,
            prompt=prompt,
            connector_prompt_embeds=connector_prompt_embeds,
            negative_connector_prompt_embeds=negative_connector_prompt_embeds,
            guidance_scale=guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            stg_scale=stg_scale,
            audio_stg_scale=audio_stg_scale,
            spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
        )
        if isinstance(prompt, str):
            prompt = [prompt]

        # 1. Encode prompts
        if connector_prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_ids = encoded["prompt_ids"]
            connector_prompt_embeds = encoded["connector_prompt_embeds"]
            connector_audio_prompt_embeds = encoded["connector_audio_prompt_embeds"]
            connector_attention_mask = encoded["connector_attention_mask"]
            negative_connector_prompt_embeds = encoded.get("negative_connector_prompt_embeds")
            negative_connector_audio_prompt_embeds = encoded.get(
                "negative_connector_audio_prompt_embeds"
            )
            negative_connector_attention_mask = encoded.get("negative_connector_attention_mask")
        else:
            connector_prompt_embeds = connector_prompt_embeds.to(device)
            connector_audio_prompt_embeds = connector_audio_prompt_embeds.to(device)
            connector_attention_mask = connector_attention_mask.to(device)
            if negative_connector_prompt_embeds is not None:
                negative_connector_prompt_embeds = negative_connector_prompt_embeds.to(device)
                negative_connector_audio_prompt_embeds = negative_connector_audio_prompt_embeds.to(
                    device
                )
                negative_connector_attention_mask = negative_connector_attention_mask.to(device)

        # 2. [I2AV] Image preprocessing (after enhancement)
        if images is not None and condition_images is None:
            encoded_img = self.encode_image(
                images,
                height=height,
                width=width,
                device=device,
            )
            condition_images = encoded_img["condition_images"]
        condition_images = pixels_to_unit(
            condition_images.to(device=device, dtype=torch.float32),
            source="LTX2 I2AV condition_images",
        )

        batch_size = connector_prompt_embeds.shape[0]

        # 3. Compute dimensions
        vae_spatial = self.pipeline.vae_spatial_compression_ratio
        vae_temporal = self.pipeline.vae_temporal_compression_ratio
        latent_h = height // vae_spatial
        latent_w = width // vae_spatial
        latent_f = (num_frames - 1) // vae_temporal + 1

        duration_s = num_frames / frame_rate
        sr = self.pipeline.audio_sampling_rate
        hop = self.pipeline.audio_hop_length
        audio_temporal_compression = self.pipeline.audio_vae_temporal_compression_ratio
        audio_num_frames = round(duration_s * sr / hop / audio_temporal_compression)
        num_mel_bins = (
            self.pipeline.audio_vae.config.mel_bins
            if getattr(self.pipeline, "audio_vae", None) is not None
            else 64
        )

        # 4. [I2AV] Prepare video latents with image conditioning
        video_latents, conditioning_mask = self.pipeline.prepare_latents(
            image=unit_to_signed_unit_pixels(condition_images),
            batch_size=batch_size,
            num_channels_latents=self.transformer_config.in_channels,
            height=height,
            width=width,
            num_frames=num_frames,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        audio_latents = self.pipeline.prepare_audio_latents(
            batch_size=batch_size,
            num_channels_latents=(
                self.pipeline.audio_vae.config.latent_channels
                if getattr(self.pipeline, "audio_vae", None) is not None
                else 8
            ),
            audio_latent_length=audio_num_frames,
            num_mel_bins=num_mel_bins,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        # 5. Set timesteps
        video_seq_len = latent_f * latent_h * latent_w
        mu = calculate_shift(
            video_seq_len,
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        timesteps = set_scheduler_timesteps(
            self.scheduler,
            num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
        )
        set_scheduler_timesteps(
            self.audio_scheduler,
            num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
        )

        video_coords = self.pipeline.transformer.rope.prepare_video_coords(
            batch_size,
            latent_f,
            latent_h,
            latent_w,
            device,
            fps=frame_rate,
        )
        audio_coords = self.pipeline.transformer.audio_rope.prepare_audio_coords(
            batch_size,
            audio_num_frames,
            device,
        )

        # 6. Setup trajectory collectors + denoising loop
        video_seq_len = video_latents.shape[1]
        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latents = self.cast_latents(torch.cat([video_latents, audio_latents], dim=1))
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(
                trajectory_indices, num_inference_steps
            )
            component_log_prob_collectors = {
                name: create_trajectory_collector(trajectory_indices, num_inference_steps)
                for name in LTX2_COMPONENT_ORDER
            }
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        for i, t in enumerate(timesteps):
            noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kw = list(set(["next_latents", "log_prob", "velocity"] + extra_call_back_kwargs))
            current_compute_log_prob: bool = compute_log_prob and noise_level > 0

            output = self.forward(
                t=t,
                t_next=t_next,
                latents=latents,
                video_seq_len=video_seq_len,
                conditioning_mask=conditioning_mask,
                connector_prompt_embeds=connector_prompt_embeds,
                connector_audio_prompt_embeds=connector_audio_prompt_embeds,
                connector_attention_mask=connector_attention_mask,
                negative_connector_prompt_embeds=negative_connector_prompt_embeds,
                negative_connector_audio_prompt_embeds=negative_connector_audio_prompt_embeds,
                negative_connector_attention_mask=negative_connector_attention_mask,
                guidance_scale=guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                guidance_rescale=guidance_rescale,
                audio_guidance_rescale=audio_guidance_rescale,
                stg_scale=stg_scale,
                audio_stg_scale=audio_stg_scale,
                spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
                modality_scale=modality_scale,
                audio_modality_scale=audio_modality_scale,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                audio_num_frames=audio_num_frames,
                video_coords=video_coords,
                audio_coords=audio_coords,
                noise_level=noise_level,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kw,
                use_cross_timestep=use_cross_timestep,
                _return_components=True,
            )

            # The component output is authoritative; the concatenated view only
            # advances the loop state and feeds the legacy callback collector.
            legacy_view = build_ltx2_legacy_callback_view(self, output)
            latents = self.cast_latents(legacy_view.next_latents)
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob:
                log_prob_collector.collect(output.log_prob, i)
                for name in LTX2_COMPONENT_ORDER:
                    component_log_prob_collectors[name].collect(output.component_log_probs[name], i)
            callback_collector.collect_step(
                step_idx=i,
                output=legacy_view,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": noise_level},
            )

        # 7. Split and Decode
        video_latents = latents[:, :video_seq_len]
        audio_latents = latents[:, video_seq_len:]
        video, audio_waveform = self.decode_latents(
            video_latents,
            audio_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type="pt",
            generator=generator,
        )

        # 8. Split the concatenated rollout into authoritative component trajectories
        collected_states = latent_collector.get_result()
        structured_callbacks, legacy_callbacks = split_ltx2_callback_results(
            self, callback_collector.get_result()
        )
        callback_index_map = callback_collector.get_index_map()
        trajectories = (
            build_ltx2_structured_trajectories(
                self,
                states=torch.stack(collected_states, dim=1),
                state_index_map=latent_collector.get_index_map(),
                video_seq_len=video_seq_len,
                schedule=build_ltx2_full_component_schedule(self, timesteps),
                callbacks=structured_callbacks,
                callback_index_map=callback_index_map if structured_callbacks else None,
                # The conditioning frame is never stepped, so it carries no
                # stochastic degrees of freedom for any downstream reducer.
                video_active_mask=~conditioning_mask.bool(),
                **build_ltx2_rollout_log_probs(
                    self,
                    log_prob_collector=log_prob_collector if compute_log_prob else None,
                    component_log_prob_collectors=(
                        component_log_prob_collectors if compute_log_prob else None
                    ),
                    num_transitions=num_inference_steps,
                ),
            )
            if collected_states
            else None
        )

        # 9. Construct samples
        prompt_list = prompt if isinstance(prompt, list) else [prompt] * batch_size
        condition_images_list = standardize_image_batch(condition_images, "pt")
        if isinstance(condition_images_list, torch.Tensor):
            condition_images_list = list(condition_images_list.unbind(0))

        samples = [
            LTX2I2AVSample(
                # Authoritative video/audio trajectory; legacy fields stay unset.
                trajectory=None if trajectories is None else trajectories[b],
                video=video[b],
                audio=audio_waveform[b] if audio_waveform is not None else None,
                audio_sample_rate=(
                    int(self.pipeline.vocoder.config.output_sampling_rate)
                    if audio_waveform is not None
                    else None
                ),
                condition_images=[condition_images_list[b]],
                conditioning_mask=conditioning_mask[b],
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                video_seq_len=video_seq_len,
                prompt=prompt_list[b],
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                connector_prompt_embeds=connector_prompt_embeds[b],
                connector_audio_prompt_embeds=connector_audio_prompt_embeds[b],
                connector_attention_mask=connector_attention_mask[b],
                negative_connector_prompt_embeds=(
                    negative_connector_prompt_embeds[b]
                    if negative_connector_prompt_embeds is not None
                    else None
                ),
                negative_connector_audio_prompt_embeds=(
                    negative_connector_audio_prompt_embeds[b]
                    if negative_connector_audio_prompt_embeds is not None
                    else None
                ),
                negative_connector_attention_mask=(
                    negative_connector_attention_mask[b]
                    if negative_connector_attention_mask is not None
                    else None
                ),
                extra_kwargs={
                    **{k: v[b] for k, v in legacy_callbacks.items()},
                    **({"callback_index_map": callback_index_map} if legacy_callbacks else {}),
                    "duration_s": duration_s,
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()
        return samples
