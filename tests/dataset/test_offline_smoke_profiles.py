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

"""Lock the public offline-smoke task catalog and official I/O contracts."""

import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from dataset.offline_smoke import profiles as p
from flow_factory import contracts as c
from flow_factory.models.registry import list_registered_models

PROFILE_IDS = tuple(
    "text_to_image image_to_image multi_image_to_image text_to_video first_frame_to_video "
    "first_last_frame_to_video text_to_audio_video first_frame_to_audio_video "
    "first_last_frame_to_audio_video ordered_references_to_audio_video".split()
)


def _input(profile: str, media_type: c.MediaType) -> c.InputMediaRule:
    return next(
        rule
        for rule in p.CANONICAL_PROFILES[profile].contract.input_media.rules
        if rule.format.type is media_type
    )


def test_catalog_covers_all_actual_profiles_models_repositories_and_aliases() -> None:
    assert tuple(p.CANONICAL_PROFILES) == PROFILE_IDS
    assert {
        model
        for profile in p.CANONICAL_PROFILES.values()
        for model in profile.compatible_model_types
    } == set(list_registered_models())
    assert p.SFT_REPO_ID == "Jayce-Ping/Flow-Factory-SFT-Smoke"
    assert p.OFFLINE_DPO_REPO_ID == "Jayce-Ping/Flow-Factory-Offline-DPO-Smoke"
    assert set(p.MAIN_GPU_ALIASES) == set(p.GPU_ALIAS_TO_PROFILE) - {"image-i2i"}
    assert p.SUPPLEMENTAL_GPU_ALIASES == ("image-i2i",)


def test_actual_profiles_preserve_exact_output_sequences() -> None:
    outputs = tuple(p.output_media_types(item.contract) for item in p.CANONICAL_PROFILES.values())
    assert outputs == (("image",),) * 3 + (("video",),) * 3 + (("video", "audio"),) * 4
    for profile in tuple(p.CANONICAL_PROFILES.values())[6:]:
        contract = profile.contract
        assert contract.output_media.items[0].fps is c.RateRequirement.REQUIRED
        assert contract.output_media.items[1].sample_rate is c.RateRequirement.REQUIRED


def test_image_and_endpoint_profiles_keep_cardinality_order_and_slots() -> None:
    i2i = _input("image_to_image", c.MediaType.IMAGE)
    multi = _input("multi_image_to_image", c.MediaType.IMAGE)
    first_video = _input("first_frame_to_video", c.MediaType.IMAGE)
    fl_video = _input("first_last_frame_to_video", c.MediaType.IMAGE)
    first_av = _input("first_frame_to_audio_video", c.MediaType.IMAGE)
    fl_av = _input("first_last_frame_to_audio_video", c.MediaType.IMAGE)

    assert (i2i.min_count, i2i.max_count) == (1, 1)
    assert (multi.min_count, multi.max_count) == (2, 2)
    assert (
        p.CANONICAL_PROFILES["multi_image_to_image"].contract.input_media.order
        is c.InputMediaOrder.WITHIN_TYPE
    )
    assert (first_video.slots, first_video.required_slots) == (("first_frame",), ("first_frame",))
    assert (fl_video.slots, fl_video.required_slots) == (
        ("first_frame", "last_frame"),
        ("first_frame", "last_frame"),
    )
    assert (fl_video.min_count, fl_video.max_count) == (2, 2)
    assert (first_av.slots, first_av.required_slots) == (("first_frame",), ("first_frame",))
    assert (fl_av.slots, fl_av.required_slots) == (("first_frame", "last_frame"), ())


def test_h3_reference_contract_is_global_heterogeneous_and_bounded() -> None:
    contract = p.CANONICAL_PROFILES["ordered_references_to_audio_video"].contract
    assert contract.input_media.binding is c.InputMediaBinding.ORDERED_REFERENCES
    assert contract.input_media.order is c.InputMediaOrder.GLOBAL
    assert (contract.input_media.min_total_count, contract.input_media.max_total_count) == (1, 12)
    assert contract.input_media.required_any_types == (c.MediaType.IMAGE, c.MediaType.VIDEO)
    counts = tuple(
        (rule.format.type.value, rule.min_count, rule.max_count)
        for rule in contract.input_media.rules
    )
    assert counts == (("image", 0, 9), ("video", 0, 3), ("audio", 0, 3))


def test_gpu_variants_keep_model_specific_clocks() -> None:
    cases = {
        case.alias: case for profile in p.CANONICAL_PROFILES.values() for case in profile.gpu_cases
    }
    wan, ltx, h3 = (cases[name].geometry for name in ("wan-t2v", "ltx2-t2av", "h3-t2va"))
    assert (wan.height, wan.num_frames) == (240, 5)
    assert (ltx.height, ltx.width, ltx.num_frames, ltx.sample_rate) == (128, 192, 9, 16000)
    assert (h3.height, h3.width, h3.num_frames, h3.sample_rate) == (64, 96, 124, 32000)
    assert cases["wan-flf2v"].checkpoint == "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
    assert p.get_profile("image-i2i") is p.CANONICAL_PROFILES["image_to_image"]


def test_official_contract_supports_ordered_heterogeneous_input_and_audio_only_output() -> None:
    rate = c.RateRequirement
    video = c.MediaFormat(
        c.MediaType.VIDEO,
        rate.OPTIONAL,
        rate.NOT_APPLICABLE,
        c.DECODED_VIDEO_REPRESENTATION,
    )
    audio_input = c.MediaFormat(
        c.MediaType.AUDIO,
        rate.NOT_APPLICABLE,
        rate.OPTIONAL,
        c.DECODED_AUDIO_REPRESENTATION,
    )
    audio_output = c.MediaFormat(
        c.MediaType.AUDIO,
        rate.NOT_APPLICABLE,
        rate.REQUIRED,
        c.DECODED_AUDIO_REPRESENTATION,
    )
    rules = (c.InputMediaRule(video, 1, 2), c.InputMediaRule(audio_input, 1, 2))
    inputs = c.InputMediaSpec(
        rules, c.InputMediaBinding.ORDERED_REFERENCES, c.InputMediaOrder.GLOBAL, 2, 4
    )
    contract = c.PipelineIOContract(
        inputs,
        c.NegativePromptPolicy.UNSUPPORTED,
        c.OutputMediaSequence((audio_output,)),
        c.GeometrySource.OUTPUT_MEDIA,
        c.BatchCapability.UNIFORM,
    )
    media = (
        SimpleNamespace(type="video", fps=24.0, sample_rate=None),
        SimpleNamespace(type="audio", fps=None, sample_rate=16000),
    )
    model_input = SimpleNamespace(prompt="Compose a sound.", negative_prompt=None, media=media)
    c.validate_pipeline_model_input(model_input, contract)
    assert p.output_media_types(contract) == ("audio",)
    output = SimpleNamespace(type="audio", fps=None, sample_rate=16000)
    c.validate_pipeline_output_candidate((output,), contract)
    assert "audio_only" not in p.CANONICAL_PROFILES


def test_catalog_is_frozen_and_import_does_not_load_bagel_adapter() -> None:
    with pytest.raises(FrozenInstanceError):
        p.CANONICAL_PROFILES["text_to_image"].name = "changed"
    script = "import sys; import dataset.offline_smoke.profiles; assert 'flow_factory.models.bagel.bagel' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)
