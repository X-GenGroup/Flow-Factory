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

"""Canonical task catalog for the public SFT and offline-DPO smoke datasets.

Profiles reuse Flow-Factory's official pipeline contract rather than defining a
second dataset-only type system. Model identifiers remain strings, so this
module never imports model adapters such as Bagel or their optional runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from flow_factory.contracts import (
    DECODED_AUDIO_REPRESENTATION,
    InputMediaBinding,
    InputMediaOrder,
    InputMediaRule,
    MediaFormat,
    MediaType,
    NegativePromptPolicy,
    PipelineIOContract,
    RateRequirement,
)
from flow_factory.models.pipeline_contracts import (
    IMAGE_FORMAT,
    VIDEO_FORMAT_OPTIONAL_FPS,
    audio_video_output_contract,
    image_output_contract,
    video_output_contract,
)

SFT_REPO_ID = "Jayce-Ping/Flow-Factory-SFT-Smoke"
OFFLINE_DPO_REPO_ID = "Jayce-Ping/Flow-Factory-Offline-DPO-Smoke"
DATASET_REPO_IDS: Mapping[str, str] = MappingProxyType(
    {"sft": SFT_REPO_ID, "offline-dpo": OFFLINE_DPO_REPO_ID}
)


@dataclass(frozen=True, slots=True)
class SmokeGeometry:
    """Small real-weight geometry used by a dataset or GPU variant."""

    height: int
    width: int
    num_frames: int | None = None
    frame_rate: float | None = None
    sample_rate: int | None = None
    num_inference_steps: int = 2

    @property
    def duration_seconds(self) -> float | None:
        """Return the declared video duration when a frame clock is available.

        Returns:
            Duration in seconds, or ``None`` for image-only geometry.
        """
        if self.num_frames is None or self.frame_rate is None:
            return None
        return self.num_frames / self.frame_rate


@dataclass(frozen=True, slots=True)
class GPUSmokeCase:
    """One model-specific alias and geometry in the GPU handoff catalog."""

    alias: str
    model_type: str
    checkpoint: str
    geometry: SmokeGeometry
    main_matrix: bool = True


@dataclass(frozen=True, slots=True)
class OfflineSmokeProfile:
    """One canonical dataset profile shared by SFT and offline DPO."""

    name: str
    compatible_model_types: tuple[str, ...]
    contract: PipelineIOContract
    default_geometry: SmokeGeometry
    gpu_cases: tuple[GPUSmokeCase, ...] = ()

    @property
    def profile_id(self) -> str:
        """Return the stable canonical profile identifier.

        Returns:
            Canonical profile name.
        """
        return self.name

    @property
    def gpu_aliases(self) -> tuple[str, ...]:
        """Return model-specific aliases that use this task profile.

        Returns:
            Ordered runtime alias tuple.
        """
        return tuple(case.alias for case in self.gpu_cases)


def output_media_types(contract: PipelineIOContract) -> tuple[str, ...]:
    """Return the exact output sequence without assuming a modality family.

    Args:
        contract: Pipeline contract whose output sequence is projected.

    Returns:
        Ordered public media type names.
    """
    return tuple(item.type.value for item in contract.output_media.items)


_NO_NEGATIVE = NegativePromptPolicy.UNSUPPORTED
_IMAGE = SmokeGeometry(256, 256)
_WAN = SmokeGeometry(240, 240, 5, 24.0)
_LTX = SmokeGeometry(128, 192, 9, 24.0, 16000)
_H3 = SmokeGeometry(64, 96, 124, 24.0, 32000)
_AUDIO_REFERENCE_FORMAT = MediaFormat(
    type=MediaType.AUDIO,
    fps=RateRequirement.NOT_APPLICABLE,
    sample_rate=RateRequirement.OPTIONAL,
    representation=DECODED_AUDIO_REPRESENTATION,
)


def _image_rule(
    min_count: int,
    max_count: int,
    slots: tuple[str, ...] = (),
    required_slots: tuple[str, ...] = (),
) -> InputMediaRule:
    return InputMediaRule(IMAGE_FORMAT, min_count, max_count, slots, required_slots)


_T2I = image_output_contract(negative_prompt=_NO_NEGATIVE)
_I2I = image_output_contract(
    negative_prompt=_NO_NEGATIVE,
    input_image_min_count=1,
    input_image_max_count=1,
)
_MRI2I = image_output_contract(
    negative_prompt=_NO_NEGATIVE,
    input_image_min_count=2,
    input_image_max_count=2,
    input_order=InputMediaOrder.WITHIN_TYPE,
)
_T2V = video_output_contract(
    negative_prompt=_NO_NEGATIVE,
    output_fps=RateRequirement.REQUIRED,
)
_I2V = video_output_contract(
    negative_prompt=_NO_NEGATIVE,
    input_image_min_count=1,
    input_image_max_count=1,
    input_image_slots=("first_frame",),
    required_input_image_slots=("first_frame",),
    output_fps=RateRequirement.REQUIRED,
)
_FL2V = video_output_contract(
    negative_prompt=_NO_NEGATIVE,
    input_image_min_count=2,
    input_image_max_count=2,
    input_image_slots=("first_frame", "last_frame"),
    required_input_image_slots=("first_frame", "last_frame"),
    output_fps=RateRequirement.REQUIRED,
)
_T2AV = audio_video_output_contract(negative_prompt=_NO_NEGATIVE)
_I2AV = audio_video_output_contract(
    negative_prompt=_NO_NEGATIVE,
    input_rules=(_image_rule(1, 1, ("first_frame",), ("first_frame",)),),
)
_FL2AV = audio_video_output_contract(
    negative_prompt=_NO_NEGATIVE,
    input_rules=(_image_rule(1, 2, ("first_frame", "last_frame")),),
    input_order=InputMediaOrder.WITHIN_TYPE,
)
_REF2AV = audio_video_output_contract(
    negative_prompt=_NO_NEGATIVE,
    input_rules=(
        InputMediaRule(format=IMAGE_FORMAT, min_count=0, max_count=9),
        InputMediaRule(format=VIDEO_FORMAT_OPTIONAL_FPS, min_count=0, max_count=3),
        InputMediaRule(format=_AUDIO_REFERENCE_FORMAT, min_count=0, max_count=3),
    ),
    input_binding=InputMediaBinding.ORDERED_REFERENCES,
    input_order=InputMediaOrder.GLOBAL,
    min_input_media_count=1,
    max_input_media_count=12,
    required_any_input_types=(MediaType.IMAGE, MediaType.VIDEO),
)


_CHECKPOINTS = {
    "sd35-t2i": "stabilityai/stable-diffusion-3.5-medium",
    "image-i2i": "black-forest-labs/FLUX.1-Kontext-dev",
    "bagel-mri2i": "ByteDance-Seed/BAGEL-7B-MoT",
    "wan-t2v": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "wan-i2v-first": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "wan-flf2v": "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
    "ltx2-t2av": "Lightricks/LTX-2",
    "ltx2-i2av": "Lightricks/LTX-2",
    "h3-t2va": "MiniMaxAI/MiniMax-H3",
    "h3-fl2va": "MiniMaxAI/MiniMax-H3",
    "h3-ref2va": "MiniMaxAI/MiniMax-H3",
}


def _case(
    alias: str,
    model: str,
    geometry: SmokeGeometry,
    main: bool = True,
) -> GPUSmokeCase:
    return GPUSmokeCase(alias, model, _CHECKPOINTS[alias], geometry, main)


_MODEL_TYPES = {
    "text_to_image": "sd3-5 flux1 flux2 flux2-klein qwen-image qwen-image-2.1 z-image bagel sensenova",
    "image_to_image": "flux1-kontext flux2 flux2-klein qwen-image-edit-plus bagel sensenova",
    "multi_image_to_image": "flux2 flux2-klein qwen-image-edit-plus bagel sensenova",
    "text_to_video": "wan2_t2v",
    "first_frame_to_video": "wan2_i2v",
    "first_last_frame_to_video": "wan2_i2v",
    "text_to_audio_video": "ltx2_t2av minimax-h3-t2va",
    "first_frame_to_audio_video": "ltx2_i2av minimax-h3-fl2va",
    "first_last_frame_to_audio_video": "minimax-h3-fl2va",
    "ordered_references_to_audio_video": "minimax-h3-ref2va",
}
_PROFILE_SPECS = (
    ("text_to_image", _T2I, _IMAGE, (_case("sd35-t2i", "sd3-5", _IMAGE),)),
    ("image_to_image", _I2I, _IMAGE, (_case("image-i2i", "flux1-kontext", _IMAGE, False),)),
    ("multi_image_to_image", _MRI2I, _IMAGE, (_case("bagel-mri2i", "bagel", _IMAGE),)),
    ("text_to_video", _T2V, _WAN, (_case("wan-t2v", "wan2_t2v", _WAN),)),
    ("first_frame_to_video", _I2V, _WAN, (_case("wan-i2v-first", "wan2_i2v", _WAN),)),
    ("first_last_frame_to_video", _FL2V, _WAN, (_case("wan-flf2v", "wan2_i2v", _WAN),)),
    (
        "text_to_audio_video",
        _T2AV,
        _LTX,
        (_case("ltx2-t2av", "ltx2_t2av", _LTX), _case("h3-t2va", "minimax-h3-t2va", _H3)),
    ),
    (
        "first_frame_to_audio_video",
        _I2AV,
        _LTX,
        (_case("ltx2-i2av", "ltx2_i2av", _LTX),),
    ),
    (
        "first_last_frame_to_audio_video",
        _FL2AV,
        _H3,
        (_case("h3-fl2va", "minimax-h3-fl2va", _H3),),
    ),
    (
        "ordered_references_to_audio_video",
        _REF2AV,
        _H3,
        (_case("h3-ref2va", "minimax-h3-ref2va", _H3),),
    ),
)
_PROFILES = tuple(
    OfflineSmokeProfile(name, tuple(_MODEL_TYPES[name].split()), contract, geometry, cases)
    for name, contract, geometry, cases in _PROFILE_SPECS
)

CANONICAL_PROFILES: Mapping[str, OfflineSmokeProfile] = MappingProxyType(
    {profile.name: profile for profile in _PROFILES}
)
GPU_ALIAS_TO_PROFILE: Mapping[str, OfflineSmokeProfile] = MappingProxyType(
    {case.alias: profile for profile in _PROFILES for case in profile.gpu_cases}
)
MAIN_GPU_ALIASES = tuple(
    "sd35-t2i bagel-mri2i wan-t2v wan-i2v-first wan-flf2v "
    "ltx2-t2av ltx2-i2av h3-t2va h3-fl2va h3-ref2va".split()
)
SUPPLEMENTAL_GPU_ALIASES = ("image-i2i",)


def get_profile(name_or_gpu_alias: str) -> OfflineSmokeProfile:
    """Resolve a canonical profile name or a model-specific GPU alias.

    Args:
        name_or_gpu_alias: Canonical profile name or runtime alias.

    Returns:
        Matching immutable smoke profile.

    Raises:
        KeyError: If no canonical profile or runtime alias matches the value.
    """
    try:
        return CANONICAL_PROFILES[name_or_gpu_alias]
    except KeyError:
        try:
            return GPU_ALIAS_TO_PROFILE[name_or_gpu_alias]
        except KeyError as error:
            raise KeyError(f"unknown offline smoke profile {name_or_gpu_alias!r}") from error


__all__ = tuple(
    "CANONICAL_PROFILES DATASET_REPO_IDS GPUSmokeCase GPU_ALIAS_TO_PROFILE MAIN_GPU_ALIASES "
    "OFFLINE_DPO_REPO_ID OfflineSmokeProfile SFT_REPO_ID SUPPLEMENTAL_GPU_ALIASES "
    "SmokeGeometry get_profile output_media_types".split()
)
