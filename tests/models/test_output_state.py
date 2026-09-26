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

"""Tests for adapter-owned target output-state contracts."""

from dataclasses import FrozenInstanceError, dataclass
from typing import Any, Mapping, Optional, Tuple

import pytest
import torch

from flow_factory.contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    InputMediaSpec,
    MediaFormat,
    MediaType,
    NegativePromptPolicy,
    OutputMediaSequence,
    PipelineIOContract,
    RateRequirement,
)
from flow_factory.models.output_state import (
    DecodedMediaBatch,
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
    OutputStateCodec,
    validate_codec_required_components,
    validate_encoded_output_state,
    validate_output_candidate_batch,
)
from flow_factory.samples import LatentState

IMAGE_FORMAT = MediaFormat(
    type=MediaType.IMAGE,
    fps=RateRequirement.NOT_APPLICABLE,
    sample_rate=RateRequirement.NOT_APPLICABLE,
)
VIDEO_FORMAT = MediaFormat(
    type=MediaType.VIDEO,
    fps=RateRequirement.REQUIRED,
    sample_rate=RateRequirement.NOT_APPLICABLE,
)
AUDIO_FORMAT = MediaFormat(
    type=MediaType.AUDIO,
    fps=RateRequirement.NOT_APPLICABLE,
    sample_rate=RateRequirement.REQUIRED,
)


@dataclass
class _DecodedMedia:
    type: str
    payload: object
    fps: Optional[float] = None
    sample_rate: Optional[int] = None


def _contract(
    *output: MediaFormat,
    batch_capability: BatchCapability = BatchCapability.UNIFORM,
) -> PipelineIOContract:
    return PipelineIOContract(
        input_media=InputMediaSpec(
            rules=(),
            binding=InputMediaBinding.GROUPED_BY_TYPE,
            order=InputMediaOrder.INSENSITIVE,
        ),
        negative_prompt=NegativePromptPolicy.UNSUPPORTED,
        output_media=OutputMediaSequence(items=tuple(output)),
        geometry_source=GeometrySource.OUTPUT_MEDIA,
        batch_capability=batch_capability,
    )


def _image_signature(height: int = 32, width: int = 48) -> GeometrySignature:
    return GeometrySignature(
        media=(
            MediaGeometrySignature(
                type=MediaType.IMAGE,
                height=height,
                width=width,
            ),
        )
    )


def _encoded_image_batch(
    batch_size: int = 2,
    *,
    component: Optional[torch.Tensor] = None,
    signatures: Optional[Tuple[GeometrySignature, ...]] = None,
    forward_context: Optional[Mapping[str, Any]] = None,
    decode_context: Optional[Mapping[str, Any]] = None,
) -> EncodedOutputState:
    if component is None:
        component = torch.zeros(batch_size, 4, 8, 12)
    if signatures is None:
        signatures = tuple(_image_signature() for _ in range(batch_size))
    return EncodedOutputState(
        clean_state=LatentState({"latent": component}),
        forward_context={} if forward_context is None else forward_context,
        decode_context={} if decode_context is None else decode_context,
        geometry_signatures=signatures,
    )


def test_validate_image_candidate_batch_preserves_structural_media_objects() -> None:
    media_batch = (
        (_DecodedMedia(type="image", payload=object()),),
        (_DecodedMedia(type="image", payload=object()),),
    )

    validated = validate_output_candidate_batch(media_batch, _contract(IMAGE_FORMAT))

    assert validated is media_batch


def test_validate_multimodal_candidate_enforces_exact_sequence_and_required_rates() -> None:
    media_batch = (
        (
            _DecodedMedia(type="video", payload=object(), fps=24.0),
            _DecodedMedia(type="audio", payload=object(), sample_rate=48_000),
        ),
    )

    assert (
        validate_output_candidate_batch(
            media_batch,
            _contract(VIDEO_FORMAT, AUDIO_FORMAT),
        )
        is media_batch
    )

    with pytest.raises(ValueError, match="exact sequence length 2"):
        validate_output_candidate_batch(
            (media_batch[0][:1],), _contract(VIDEO_FORMAT, AUDIO_FORMAT)
        )
    with pytest.raises(ValueError, match=r"item 0\.type 'video'"):
        validate_output_candidate_batch(
            (
                (
                    _DecodedMedia(type="audio", payload=object(), sample_rate=48_000),
                    media_batch[0][1],
                ),
            ),
            _contract(VIDEO_FORMAT, AUDIO_FORMAT),
        )
    with pytest.raises(ValueError, match="required.*fps"):
        validate_output_candidate_batch(
            (
                (
                    _DecodedMedia(type="video", payload=object()),
                    media_batch[0][1],
                ),
            ),
            _contract(VIDEO_FORMAT, AUDIO_FORMAT),
        )
    with pytest.raises(ValueError, match="required.*sample_rate"):
        validate_output_candidate_batch(
            (
                (
                    media_batch[0][0],
                    _DecodedMedia(type="audio", payload=object()),
                ),
            ),
            _contract(VIDEO_FORMAT, AUDIO_FORMAT),
        )


@pytest.mark.parametrize(
    "media_batch,match",
    [
        ([], "output media batch to be tuple"),
        (([_DecodedMedia(type="image", payload=object())],), "sample 0 to be tuple"),
        (((_DecodedMedia(type="image", payload=None),),), "decoded payload"),
        (
            ((_DecodedMedia(type="image", payload=object(), fps=1.0),),),
            "fps=None",
        ),
    ],
)
def test_candidate_validation_rejects_mutable_or_incoherent_media(
    media_batch: object,
    match: str,
) -> None:
    error_type = TypeError if "tuple" in match else ValueError
    with pytest.raises(error_type, match=match):
        validate_output_candidate_batch(media_batch, _contract(IMAGE_FORMAT))


def test_single_sample_contract_rejects_larger_candidate_batch() -> None:
    media_batch = tuple((_DecodedMedia(type="image", payload=object()),) for _ in range(2))
    with pytest.raises(ValueError, match="batch size 1"):
        validate_output_candidate_batch(
            media_batch,
            _contract(IMAGE_FORMAT, batch_capability=BatchCapability.SINGLE_SAMPLE),
        )


def test_media_geometry_signature_is_strict_coherent_and_hashable() -> None:
    image = MediaGeometrySignature(type=MediaType.IMAGE, height=32, width=48)
    video = MediaGeometrySignature(
        type=MediaType.VIDEO,
        height=32,
        width=48,
        frames=9,
        fps=24.0,
    )
    audio = MediaGeometrySignature(
        type=MediaType.AUDIO,
        samples=18_000,
        sample_rate=48_000,
    )
    signature = GeometrySignature(media=(video, audio))

    assert image.height == 32
    assert hash(signature)
    with pytest.raises(TypeError, match="type to be MediaType"):
        MediaGeometrySignature(type="image", height=32, width=48)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected image geometry fields"):
        MediaGeometrySignature(type=MediaType.IMAGE, height=32)
    with pytest.raises(ValueError, match="expected video geometry fields"):
        MediaGeometrySignature(type=MediaType.VIDEO, height=32, width=48)
    with pytest.raises(TypeError, match="positive finite float"):
        MediaGeometrySignature(
            type=MediaType.VIDEO,
            height=32,
            width=48,
            frames=9,
            fps=24,  # type: ignore[arg-type]
        )


def test_encoded_output_state_freezes_outer_contexts_without_copying_tensor_leaves() -> None:
    ids = torch.zeros(2, 4, 3)
    forward_context = {"img_ids": ids}
    encoded = _encoded_image_batch(
        forward_context=forward_context,
        decode_context={"height": 32, "width": 48},
    )
    forward_context["late_mutation"] = True

    assert encoded.forward_context["img_ids"] is ids
    assert "late_mutation" not in encoded.forward_context
    with pytest.raises(TypeError):
        encoded.forward_context["other"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        encoded.clean_state = LatentState({"latent": torch.zeros(2, 1)})  # type: ignore[misc]


@pytest.mark.parametrize(
    "key",
    [
        "state",
        "latents",
        "return_fields",
        "record_ids",
        "target_media",
        "clean_state",
    ],
)
def test_encoded_output_state_rejects_reserved_forward_context_keys(key: str) -> None:
    with pytest.raises(ValueError, match=rf"non-model or state-owned.*{key}"):
        _encoded_image_batch(forward_context={key: object()})


def test_validate_encoded_output_state_accepts_detached_uniform_image_state() -> None:
    encoded = _encoded_image_batch(
        forward_context={"img_ids": torch.zeros(2, 4, 3)},
        decode_context={"height": 32, "width": 48},
    )

    validated = validate_encoded_output_state(
        encoded,
        contract=_contract(IMAGE_FORMAT),
        expected_component_order=("latent",),
        expected_batch_size=2,
        device="cpu",
    )

    assert validated is encoded


def test_validate_encoded_output_state_checks_component_order_batch_and_dtype() -> None:
    encoded = _encoded_image_batch()
    with pytest.raises(ValueError, match="component order.*video"):
        validate_encoded_output_state(
            encoded,
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("video",),
            expected_batch_size=2,
            device="cpu",
        )
    with pytest.raises(ValueError, match="batch size 3"):
        validate_encoded_output_state(
            encoded,
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=3,
            device="cpu",
        )
    integer_state = _encoded_image_batch(component=torch.zeros(2, 4, dtype=torch.int64))
    with pytest.raises(TypeError, match="expected float16, bfloat16, or float32"):
        validate_encoded_output_state(
            integer_state,
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )


def test_validate_encoded_output_state_revalidates_mutable_latent_active_masks() -> None:
    clean_state = LatentState(
        {"latent": torch.zeros(2, 4)},
        active_masks={"latent": torch.ones(2, 1, dtype=torch.bool)},
    )
    clean_state.active_masks["latent"] = torch.ones(2, 1)
    encoded = EncodedOutputState(
        clean_state=clean_state,
        forward_context={},
        decode_context={},
        geometry_signatures=(_image_signature(), _image_signature()),
    )

    with pytest.raises(TypeError, match="active mask.*dtype torch.bool"):
        validate_encoded_output_state(
            encoded,
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )


def test_validate_encoded_output_state_checks_device_and_no_grad_tensors() -> None:
    requires_grad = _encoded_image_batch(component=torch.zeros(2, 4, requires_grad=True))
    with pytest.raises(ValueError, match="detached no-grad.*latent"):
        validate_encoded_output_state(
            requires_grad,
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )

    wrong_device = _encoded_image_batch(component=torch.empty(2, 4, device="meta"))
    with pytest.raises(ValueError, match="on device cpu.*meta"):
        validate_encoded_output_state(
            wrong_device,
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )

    context_grad = _encoded_image_batch(
        forward_context={"img_ids": torch.zeros(2, 3, requires_grad=True)}
    )
    with pytest.raises(ValueError, match="detached no-grad.*img_ids"):
        validate_encoded_output_state(
            context_grad,
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )

    decode_context_device = _encoded_image_batch(
        decode_context={"sizes": torch.empty(2, 2, device="meta")}
    )
    assert (
        validate_encoded_output_state(
            decode_context_device,
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )
        is decode_context_device
    )


def test_validate_encoded_output_state_rejects_unsupported_context_containers() -> None:
    encoded = _encoded_image_batch(decode_context={"sizes": {32, 48}})

    with pytest.raises(TypeError, match="unsupported set"):
        validate_encoded_output_state(
            encoded,
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )


def test_validate_encoded_output_state_rejects_cyclic_context_trees() -> None:
    nested: dict[str, Any] = {}
    nested["cycle"] = nested
    encoded = _encoded_image_batch(decode_context={"nested": nested})

    with pytest.raises(ValueError, match="acyclic context tree"):
        validate_encoded_output_state(
            encoded,
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )


def test_validate_encoded_output_state_checks_signatures_and_uniform_geometry() -> None:
    with pytest.raises(ValueError, match="one geometry signature per encoded sample"):
        validate_encoded_output_state(
            _encoded_image_batch(signatures=(_image_signature(),)),
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )

    ragged_signatures = (_image_signature(32, 48), _image_signature(48, 32))
    encoded = _encoded_image_batch(signatures=ragged_signatures)
    with pytest.raises(ValueError, match="identical geometry signatures"):
        validate_encoded_output_state(
            encoded,
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )
    with pytest.raises(ValueError, match="require active masks"):
        validate_encoded_output_state(
            encoded,
            contract=_contract(IMAGE_FORMAT, batch_capability=BatchCapability.RAGGED),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )

    masked = EncodedOutputState(
        clean_state=LatentState(
            {"latent": encoded.clean_state.components["latent"]},
            active_masks={"latent": torch.ones(2, 1, 1, 1, dtype=torch.bool)},
        ),
        forward_context=encoded.forward_context,
        decode_context=encoded.decode_context,
        geometry_signatures=encoded.geometry_signatures,
    )
    assert (
        validate_encoded_output_state(
            masked,
            contract=_contract(IMAGE_FORMAT, batch_capability=BatchCapability.RAGGED),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )
        is masked
    )


def test_validate_encoded_output_state_rejects_float64_components() -> None:
    encoded = _encoded_image_batch(component=torch.zeros(2, 4, dtype=torch.float64))

    with pytest.raises(TypeError, match="expected float16, bfloat16, or float32"):
        validate_encoded_output_state(
            encoded,
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_encoded_output_state_rejects_non_finite_components(value: float) -> None:
    component = torch.zeros(2, 4)
    component[0, 0] = value
    encoded = _encoded_image_batch(component=component)

    with pytest.raises(ValueError, match="clean_state component 'latent'.*non-finite"):
        validate_encoded_output_state(
            encoded,
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )


def test_geometry_signature_must_match_exact_output_types_and_rate_policy() -> None:
    audio_geometry = GeometrySignature(
        media=(
            MediaGeometrySignature(
                type=MediaType.AUDIO,
                samples=18_000,
                sample_rate=48_000,
            ),
        )
    )
    with pytest.raises(ValueError, match=r"item 0\.type 'image'"):
        validate_encoded_output_state(
            _encoded_image_batch(signatures=(audio_geometry, audio_geometry)),
            contract=_contract(IMAGE_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=2,
            device="cpu",
        )

    video_without_fps = GeometrySignature(
        media=(
            MediaGeometrySignature(
                type=MediaType.VIDEO,
                height=32,
                width=48,
                frames=9,
            ),
        )
    )
    video_state = EncodedOutputState(
        clean_state=LatentState({"latent": torch.zeros(1, 4, 9, 4, 6)}),
        forward_context={},
        decode_context={},
        geometry_signatures=(video_without_fps,),
    )
    with pytest.raises(ValueError, match="required.*fps"):
        validate_encoded_output_state(
            video_state,
            contract=_contract(VIDEO_FORMAT),
            expected_component_order=("latent",),
            expected_batch_size=1,
            device="cpu",
        )


class _ImageCodec:
    required_components = ("vae", "image_processor")

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        del media_batch, condition, generator
        return _encoded_image_batch(batch_size=1)


def test_output_state_codec_is_structural_and_required_components_are_validated() -> None:
    codec = _ImageCodec()

    assert isinstance(codec, OutputStateCodec)
    assert validate_codec_required_components(
        codec,
        ("transformer", "vae", "image_processor"),
    ) == ("vae", "image_processor")


@pytest.mark.parametrize(
    "required,available,error_type,match",
    [
        (["vae"], ("vae",), TypeError, "required_components to be tuple"),
        (("vae", "vae"), ("vae",), ValueError, "unique component names"),
        (("decoder",), ("vae",), ValueError, "unavailable adapter components"),
        (("",), ("vae",), ValueError, "non-empty component name"),
    ],
)
def test_codec_required_component_validation_fails_fast(
    required: object,
    available: Tuple[str, ...],
    error_type: type[Exception],
    match: str,
) -> None:
    codec = _ImageCodec()
    codec.required_components = required  # type: ignore[assignment]

    with pytest.raises(error_type, match=match):
        validate_codec_required_components(codec, available)


def test_codec_required_component_validation_rejects_missing_encode_method() -> None:
    class _NotACodec:
        required_components = ("vae",)

    with pytest.raises(TypeError, match="callable encode_output_state"):
        validate_codec_required_components(_NotACodec(), ("vae",))
