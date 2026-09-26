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

"""Small constructors for repeated adapter pipeline I/O declarations."""

from __future__ import annotations

from typing import Optional, Tuple

from ..contracts import (
    DECODED_AUDIO_REPRESENTATION,
    DECODED_IMAGE_REPRESENTATION,
    DECODED_VIDEO_REPRESENTATION,
    BatchCapability,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    InputMediaRule,
    InputMediaSpec,
    MediaFormat,
    MediaType,
    NegativePromptPolicy,
    OutputMediaSequence,
    PipelineIOContract,
    RateRequirement,
)

IMAGE_FORMAT = MediaFormat(
    type=MediaType.IMAGE,
    fps=RateRequirement.NOT_APPLICABLE,
    sample_rate=RateRequirement.NOT_APPLICABLE,
    representation=DECODED_IMAGE_REPRESENTATION,
)
VIDEO_FORMAT_OPTIONAL_FPS = MediaFormat(
    type=MediaType.VIDEO,
    fps=RateRequirement.OPTIONAL,
    sample_rate=RateRequirement.NOT_APPLICABLE,
    representation=DECODED_VIDEO_REPRESENTATION,
)
VIDEO_FORMAT_REQUIRED_FPS = MediaFormat(
    type=MediaType.VIDEO,
    fps=RateRequirement.REQUIRED,
    sample_rate=RateRequirement.NOT_APPLICABLE,
    representation=DECODED_VIDEO_REPRESENTATION,
)
AUDIO_FORMAT_REQUIRED_RATE = MediaFormat(
    type=MediaType.AUDIO,
    fps=RateRequirement.NOT_APPLICABLE,
    sample_rate=RateRequirement.REQUIRED,
    representation=DECODED_AUDIO_REPRESENTATION,
)


def image_output_contract(
    *,
    negative_prompt: NegativePromptPolicy,
    input_image_min_count: Optional[int] = None,
    input_image_max_count: Optional[int] = None,
    input_order: InputMediaOrder = InputMediaOrder.INSENSITIVE,
    input_binding: InputMediaBinding = InputMediaBinding.GROUPED_BY_TYPE,
    geometry_source: GeometrySource = GeometrySource.CONFIGURED,
    batch_capability: BatchCapability = BatchCapability.UNIFORM,
) -> PipelineIOContract:
    """Build an exact one-image output declaration with optional image inputs.

    ``input_image_min_count=None`` means the pipeline accepts no input image field.
    A value of zero creates an optional image rule, while a positive value creates
    a required rule.

    Args:
        negative_prompt: Whether negative prompts are unsupported, optional, or required.
        input_image_min_count: Minimum condition-image count, or ``None`` for no image input.
        input_image_max_count: Maximum condition-image count when an image rule is present.
        input_order: Ordering semantics for condition images.
        input_binding: Whether input media is grouped by type or preserves manifest order.
        geometry_source: Boundary that determines output geometry.
        batch_capability: Whether the adapter accepts uniform batches or one sample only.

    Returns:
        Immutable pipeline I/O contract for one image output.
    """
    rules = ()
    if input_image_min_count is not None:
        rules = (
            InputMediaRule(
                format=IMAGE_FORMAT,
                min_count=input_image_min_count,
                max_count=input_image_max_count,
            ),
        )
    return PipelineIOContract(
        input_media=InputMediaSpec(
            rules=rules,
            binding=input_binding,
            order=input_order,
        ),
        negative_prompt=negative_prompt,
        output_media=OutputMediaSequence(items=(IMAGE_FORMAT,)),
        geometry_source=geometry_source,
        batch_capability=batch_capability,
    )


def video_output_contract(
    *,
    negative_prompt: NegativePromptPolicy,
    input_image_min_count: Optional[int] = None,
    input_image_max_count: Optional[int] = None,
    input_image_slots: Tuple[str, ...] = (),
    required_input_image_slots: Tuple[str, ...] = (),
    output_fps: RateRequirement = RateRequirement.OPTIONAL,
    geometry_source: GeometrySource = GeometrySource.OUTPUT_MEDIA,
    batch_capability: BatchCapability = BatchCapability.UNIFORM,
) -> PipelineIOContract:
    """Build an exact one-video output declaration.

    Args:
        negative_prompt: Whether negative prompts are unsupported, optional, or required.
        input_image_min_count: Minimum condition-image count, or ``None`` for no image input.
        input_image_max_count: Maximum condition-image count when an image rule is present.
        input_image_slots: Semantic image argument slots in positional fallback order.
        required_input_image_slots: Slots that every valid request must fill.
        output_fps: Whether target video frame rate metadata is required.
        geometry_source: Boundary that determines output geometry.
        batch_capability: Whether the adapter accepts uniform batches or one sample only.

    Returns:
        Immutable pipeline I/O contract for one video output.
    """
    rules = ()
    if input_image_min_count is not None:
        rules = (
            InputMediaRule(
                format=IMAGE_FORMAT,
                min_count=input_image_min_count,
                max_count=input_image_max_count,
                slots=input_image_slots,
                required_slots=required_input_image_slots,
            ),
        )
    video_format = MediaFormat(
        type=MediaType.VIDEO,
        fps=output_fps,
        sample_rate=RateRequirement.NOT_APPLICABLE,
        representation=DECODED_VIDEO_REPRESENTATION,
    )
    return PipelineIOContract(
        input_media=InputMediaSpec(
            rules=rules,
            binding=InputMediaBinding.GROUPED_BY_TYPE,
            order=(
                InputMediaOrder.INSENSITIVE
                if input_image_min_count is None or input_image_max_count == 1
                else InputMediaOrder.WITHIN_TYPE
            ),
        ),
        negative_prompt=negative_prompt,
        output_media=OutputMediaSequence(items=(video_format,)),
        geometry_source=geometry_source,
        batch_capability=batch_capability,
    )


def audio_video_output_contract(
    *,
    negative_prompt: NegativePromptPolicy,
    input_rules: Tuple[InputMediaRule, ...] = (),
    input_binding: InputMediaBinding = InputMediaBinding.GROUPED_BY_TYPE,
    input_order: InputMediaOrder = InputMediaOrder.INSENSITIVE,
    min_input_media_count: Optional[int] = None,
    max_input_media_count: Optional[int] = None,
    required_any_input_types: Tuple[MediaType, ...] = (),
    output_fps: RateRequirement = RateRequirement.REQUIRED,
    output_sample_rate: RateRequirement = RateRequirement.REQUIRED,
    geometry_source: GeometrySource = GeometrySource.OUTPUT_MEDIA,
    batch_capability: BatchCapability = BatchCapability.SINGLE_SAMPLE,
) -> PipelineIOContract:
    """Build an exact ordered video-and-audio output declaration.

    Supplying explicit input rules keeps this constructor neutral to how a model
    binds conditions: prompt-only, grouped image conditions, and globally ordered
    heterogeneous references all use the same output contract. Aggregate count
    and required-any-type constraints cover cross-modality request invariants.

    Args:
        negative_prompt: Whether negative prompts are unsupported, optional, or required.
        input_rules: Canonically ordered per-type input-media rules.
        input_binding: How input media are projected into model-facing arguments.
        input_order: Which input-media ordering carries semantic meaning.
        min_input_media_count: Optional minimum count across all input modalities.
        max_input_media_count: Optional maximum count across all input modalities.
        required_any_input_types: Media types of which at least one must be present.
        output_fps: Whether target-video frame-rate metadata is accepted or required.
        output_sample_rate: Whether target-audio sample-rate metadata is accepted or required.
        geometry_source: Boundary that determines aligned output geometry.
        batch_capability: Whether the adapter accepts uniform, ragged, or single-sample batches.

    Returns:
        Immutable pipeline I/O contract with exact output order ``(video, audio)``.
    """
    video_format = MediaFormat(
        type=MediaType.VIDEO,
        fps=output_fps,
        sample_rate=RateRequirement.NOT_APPLICABLE,
        representation=DECODED_VIDEO_REPRESENTATION,
    )
    audio_format = MediaFormat(
        type=MediaType.AUDIO,
        fps=RateRequirement.NOT_APPLICABLE,
        sample_rate=output_sample_rate,
        representation=DECODED_AUDIO_REPRESENTATION,
    )
    return PipelineIOContract(
        input_media=InputMediaSpec(
            rules=input_rules,
            binding=input_binding,
            order=input_order,
            min_total_count=min_input_media_count,
            max_total_count=max_input_media_count,
            required_any_types=required_any_input_types,
        ),
        negative_prompt=negative_prompt,
        output_media=OutputMediaSequence(items=(video_format, audio_format)),
        geometry_source=geometry_source,
        batch_capability=batch_capability,
    )


__all__ = [
    "AUDIO_FORMAT_REQUIRED_RATE",
    "IMAGE_FORMAT",
    "VIDEO_FORMAT_OPTIONAL_FPS",
    "VIDEO_FORMAT_REQUIRED_FPS",
    "audio_video_output_contract",
    "image_output_contract",
    "video_output_contract",
]
