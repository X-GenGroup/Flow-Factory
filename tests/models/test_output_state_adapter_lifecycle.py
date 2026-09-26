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

"""Lightweight tests for the BaseAdapter output-state codec lifecycle seam."""

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Tuple

import pytest
import torch
from PIL import Image

from flow_factory.contracts import (
    DECODED_IMAGE_REPRESENTATION,
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
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.condition_state import PreparedConditionState
from flow_factory.models.output_state import (
    DecodedMediaBatch,
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
)
from flow_factory.samples import LatentState

IMAGE_CONTRACT = PipelineIOContract(
    input_media=InputMediaSpec(
        rules=(),
        binding=InputMediaBinding.GROUPED_BY_TYPE,
        order=InputMediaOrder.INSENSITIVE,
    ),
    negative_prompt=NegativePromptPolicy.OPTIONAL,
    output_media=OutputMediaSequence(
        items=(
            MediaFormat(
                type=MediaType.IMAGE,
                fps=RateRequirement.NOT_APPLICABLE,
                sample_rate=RateRequirement.NOT_APPLICABLE,
                representation=DECODED_IMAGE_REPRESENTATION,
            ),
        )
    ),
    geometry_source=GeometrySource.CONFIGURED,
    batch_capability=BatchCapability.UNIFORM,
)


@dataclass(frozen=True)
class _DecodedMedia:
    type: str
    payload: Any
    fps: Optional[float] = None
    sample_rate: Optional[int] = None


class _Scheduler:
    def step(self) -> None:
        """Provide the scheduler-like surface required by SchedulerGroup."""


def _config(latent_storage_dtype: Optional[str] = None) -> SimpleNamespace:
    return SimpleNamespace(
        model_args=SimpleNamespace(
            resume_path=None,
            resume_type=None,
            finetune_type="full",
            target_components=["transformer_alias"],
        ),
        training_args=SimpleNamespace(
            enable_gradient_checkpointing=False,
            latent_storage_dtype=latent_storage_dtype,
        ),
        eval_args=SimpleNamespace(),
    )


def _image_batch(batch_size: int = 2) -> DecodedMediaBatch:
    return tuple(
        (_DecodedMedia(type="image", payload=Image.new("RGB", (8, 8))),) for _ in range(batch_size)
    )


def _encoded_image_batch(
    batch_size: int,
    *,
    component_name: str = "latent",
    tensor: Optional[torch.Tensor] = None,
) -> EncodedOutputState:
    if tensor is None:
        tensor = torch.ones(batch_size, 2, dtype=torch.float32)
    signature = GeometrySignature(
        media=(
            MediaGeometrySignature(
                type=MediaType.IMAGE,
                height=8,
                width=8,
            ),
        )
    )
    return EncodedOutputState(
        clean_state=LatentState({component_name: tensor}),
        forward_context={},
        decode_context={"height": 8, "width": 8},
        geometry_signatures=tuple(signature for _ in range(batch_size)),
    )


class _Codec:
    def __init__(
        self,
        required_components: Tuple[str, ...] = ("vae",),
        result: Optional[EncodedOutputState] = None,
    ) -> None:
        self.required_components = required_components
        self.result = result
        self.calls = 0
        self.grad_enabled: Optional[bool] = None
        self.received: Optional[Tuple[Any, ...]] = None

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        self.calls += 1
        self.grad_enabled = torch.is_grad_enabled()
        self.received = (media_batch, condition, generator)
        return self.result or _encoded_image_batch(len(media_batch))


class _ConditionPreparer:
    required_components = ("vae",)

    def __init__(self) -> None:
        self.calls = 0
        self.grad_enabled: Optional[bool] = None

    def prepare_condition_state(
        self,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> PreparedConditionState:
        del generator
        self.calls += 1
        self.grad_enabled = torch.is_grad_enabled()
        return PreparedConditionState(
            condition=condition,
            forward_context={"condition_prefix": torch.zeros(1, 2)},
            output_context={"codec_condition": torch.ones(1, 2)},
        )


class _StochasticConditionPreparer(_ConditionPreparer):
    def prepare_condition_state(
        self,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> PreparedConditionState:
        torch.rand((), generator=generator)
        return super().prepare_condition_state(condition, generator)


class _LifecycleAdapter(BaseAdapter):
    pipeline_io_contract = IMAGE_CONTRACT

    def __init__(
        self,
        codec: Optional[_Codec],
        *,
        latent_storage_dtype: Optional[str] = None,
    ) -> None:
        self._codec_to_build = codec
        self.codec_build_context: Optional[Tuple[bool, bool, bool, bool]] = None
        self.geometry_validation: Optional[Tuple[Any, ...]] = None
        self.decode_call: Optional[Tuple[Any, ...]] = None
        super().__init__(
            _config(latent_storage_dtype),
            SimpleNamespace(device=torch.device("cpu")),
        )

    def build_component_runtime(self) -> Any:
        scheduler = _Scheduler()
        return SimpleNamespace(
            pipeline=SimpleNamespace(scheduler=scheduler),
            declared_component_names=("scheduler", "vae", "transformer"),
            materialized_component_names=("scheduler", "vae", "transformer"),
            override_components={},
            resolve_component_names=lambda names: (
                ["transformer"] if names == ["transformer_alias"] else list(names or ())
            ),
        )

    def load_pipeline(self) -> Any:
        raise AssertionError("build_component_runtime owns this test pipeline")

    def load_scheduler(self) -> Any:
        return self.pipeline.scheduler

    def build_output_state_codec(self) -> Optional[_Codec]:
        self.codec_build_context = (
            hasattr(self, "component_runtime"),
            hasattr(self, "pipeline"),
            hasattr(self, "scheduler_group"),
            self.model_args.target_components == ["transformer"],
        )
        return self._codec_to_build

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        self.geometry_validation = (media_batch, condition, encoded)

    def _init_target_module_map(self) -> Any:
        return {}

    def _freeze_components(self) -> None:
        pass

    def _mix_precision(self) -> None:
        pass

    def decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        output_type: str = "pil",
    ) -> torch.Tensor:
        self.decode_call = (latents, height, output_type)
        return latents

    def inference(self, **kwargs: Any) -> Any:
        return []

    def forward(self, **kwargs: Any) -> Any:
        return None


class _OnlineOnlyAdapter(_LifecycleAdapter):
    pipeline_io_contract = None


class _DefaultGeometryAdapter(_LifecycleAdapter):
    _validate_encoded_output_geometry = BaseAdapter._validate_encoded_output_geometry


class _PreparedLifecycleAdapter(_LifecycleAdapter):
    def __init__(self, codec: Optional[_Codec], preparer: _ConditionPreparer) -> None:
        self._preparer_to_build = preparer
        super().__init__(codec)

    def build_condition_state_preparer(self) -> _ConditionPreparer:
        return self._preparer_to_build


def test_codec_build_runs_after_component_and_scheduler_lifecycle() -> None:
    codec = _Codec()

    adapter = _LifecycleAdapter(codec)

    assert adapter.codec_build_context == (True, True, True, True)
    assert adapter.output_state_codec is codec
    assert adapter.output_state_encoding_modules == ("vae",)


def test_condition_preparer_is_declaration_only_and_runs_under_no_grad() -> None:
    preparer = _ConditionPreparer()
    adapter = _PreparedLifecycleAdapter(_Codec(), preparer)

    prepared = adapter.prepare_condition_state({"prompt_embeds": torch.ones(1, 2)})

    assert adapter.condition_state_preparer is preparer
    assert adapter.condition_state_encoding_modules == ("vae",)
    assert preparer.calls == 1
    assert preparer.grad_enabled is False
    assert tuple(prepared.forward_context) == ("condition_prefix",)
    assert tuple(prepared.output_context) == ("codec_condition",)


def test_raw_condition_encode_routes_through_declared_preparer() -> None:
    preparer = _ConditionPreparer()
    codec = _Codec()
    adapter = _PreparedLifecycleAdapter(codec, preparer)
    condition = {"prompt_embeds": torch.ones(1, 2)}

    adapter.encode_output_state(_image_batch(1), condition)

    assert preparer.calls == 1
    assert codec.received is not None
    assert tuple(codec.received[1]) == ("prompt_embeds", "codec_condition")
    assert torch.equal(codec.received[1]["prompt_embeds"], condition["prompt_embeds"])
    assert torch.equal(codec.received[1]["codec_condition"], torch.ones(1, 2))


def test_default_condition_preparation_is_identity() -> None:
    adapter = _LifecycleAdapter(_Codec())
    condition = {"prompt_embeds": torch.ones(1, 2)}

    prepared = adapter.prepare_condition_state(condition)

    assert adapter.condition_state_preparer is None
    assert adapter.condition_state_encoding_modules == ()
    assert dict(prepared.condition) == condition
    assert not prepared.forward_context
    assert not prepared.output_context


def test_codec_build_must_remain_declaration_only() -> None:
    class MaterializingCodecAdapter(_LifecycleAdapter):
        def build_output_state_codec(self) -> Optional[_Codec]:
            self.component_runtime.materialized_component_names += ("late_component",)
            return self._codec_to_build

    with pytest.raises(RuntimeError, match=r"declaration-only.*cannot materialize"):
        MaterializingCodecAdapter(_Codec())


def test_contract_without_codec_preserves_online_adapter_construction() -> None:
    adapter = _LifecycleAdapter(None)

    assert adapter.output_state_codec is None
    assert adapter.output_state_encoding_modules == ()
    with pytest.raises(RuntimeError, match=r"does not provide an output-state codec"):
        adapter.encode_output_state(_image_batch(1), {})


def test_effective_pipeline_contract_can_narrow_checkpoint_capability() -> None:
    class SingleSampleAdapter(_LifecycleAdapter):
        def _resolve_pipeline_io_contract(self) -> PipelineIOContract:
            return replace(
                self.pipeline_io_contract,
                batch_capability=BatchCapability.SINGLE_SAMPLE,
            )

    adapter = SingleSampleAdapter(_Codec())

    assert adapter.pipeline_io_contract.batch_capability is BatchCapability.UNIFORM
    assert adapter.effective_pipeline_io_contract.batch_capability is BatchCapability.SINGLE_SAMPLE
    with pytest.raises(ValueError, match=r"single_sample.*batch size 1.*2"):
        adapter.encode_output_state(_image_batch(2), {})


def test_known_codec_blocker_is_actionable_at_direct_encode_boundary() -> None:
    class KnownUnavailableAdapter(_OnlineOnlyAdapter):
        output_state_codec_unavailable_reason = (
            "Source conditioning pixels are not retained; extend the condition contract."
        )

    adapter = KnownUnavailableAdapter(None)

    with pytest.raises(
        NotImplementedError,
        match=r"KnownUnavailableAdapter.*Source conditioning pixels.*extend",
    ):
        adapter.encode_output_state(_image_batch(1), {})


@pytest.mark.parametrize("reason", ["", "   ", 3])
def test_codec_blocker_reason_must_be_a_non_empty_string(reason: Any) -> None:
    class InvalidReasonAdapter(_LifecycleAdapter):
        output_state_codec_unavailable_reason = reason

    with pytest.raises(TypeError, match=r"non-empty string or None"):
        InvalidReasonAdapter(None)


def test_codec_and_unavailable_reason_cannot_be_declared_together() -> None:
    class StaleBlockerAdapter(_LifecycleAdapter):
        output_state_codec_unavailable_reason = "Codec is not implemented."

    with pytest.raises(ValueError, match=r"built an output-state codec.*stale blocker"):
        StaleBlockerAdapter(_Codec())


def test_online_only_adapter_fails_clearly_when_encoding_is_requested() -> None:
    adapter = _OnlineOnlyAdapter(None)

    with pytest.raises(RuntimeError, match=r"does not declare pipeline_io_contract"):
        adapter.encode_output_state(_image_batch(1), {})


def test_codec_without_pipeline_contract_fails_during_init() -> None:
    with pytest.raises(ValueError, match=r"codec without declaring pipeline_io_contract"):
        _OnlineOnlyAdapter(_Codec())


def test_invalid_pipeline_contract_type_fails_during_init() -> None:
    class InvalidContractAdapter(_LifecycleAdapter):
        pipeline_io_contract = "image"  # type: ignore[assignment]

    with pytest.raises(TypeError, match=r"PipelineIOContract or None.*str"):
        InvalidContractAdapter(None)


def test_codec_required_components_must_exist_in_runtime() -> None:
    with pytest.raises(ValueError, match=r"unavailable adapter components.*missing"):
        _LifecycleAdapter(_Codec(required_components=("missing",)))


def test_public_output_state_wrapper_cannot_be_overridden() -> None:
    with pytest.raises(TypeError, match=r"must not override BaseAdapter.prepare_condition_state"):

        class InvalidConditionAdapter(_LifecycleAdapter):
            def prepare_condition_state(self, *args: Any, **kwargs: Any) -> Any:
                return None

    with pytest.raises(TypeError, match=r"must not override BaseAdapter.encode_output_state"):

        class InvalidAdapter(_LifecycleAdapter):
            def encode_output_state(self, *args: Any, **kwargs: Any) -> Any:
                return None

    with pytest.raises(TypeError, match=r"must not override BaseAdapter.decode_output_state"):

        class InvalidDecodeAdapter(_LifecycleAdapter):
            def decode_output_state(self, *args: Any, **kwargs: Any) -> Any:
                return None


def test_multi_component_decoder_extends_the_protected_hook() -> None:
    class MultiComponentDecodeAdapter(_LifecycleAdapter):
        def _decode_output_state(
            self,
            encoded: EncodedOutputState,
            *,
            output_type: str,
        ) -> Any:
            return encoded.clean_state.component_names, output_type

    adapter = MultiComponentDecodeAdapter(_Codec())
    encoded = _encoded_image_batch(1, component_name="video")

    assert adapter.decode_output_state(encoded, output_type="pt") == (("video",), "pt")


def test_encode_output_state_validates_invokes_no_grad_and_applies_storage_dtype() -> None:
    codec = _Codec()
    adapter = _LifecycleAdapter(codec, latent_storage_dtype="fp16")
    media_batch = _image_batch(2)
    condition = {"prompt_embeds": torch.zeros(2, 4)}
    generator = torch.Generator().manual_seed(7)

    encoded = adapter.encode_output_state(media_batch, condition, generator)

    assert codec.calls == 1
    assert codec.grad_enabled is False
    assert codec.received is not None
    assert codec.received[0] is media_batch
    assert dict(codec.received[1]) == condition
    assert codec.received[2] is generator
    assert encoded.clean_state.components["latent"].dtype is torch.float16
    assert adapter.geometry_validation is not None
    assert adapter.geometry_validation[0] is media_batch
    assert dict(adapter.geometry_validation[1]) == condition
    assert adapter.geometry_validation[2] is encoded


def test_encode_output_state_rejects_candidate_before_invoking_codec() -> None:
    codec = _Codec()
    adapter = _LifecycleAdapter(codec)
    wrong_type = ((_DecodedMedia(type="video", payload=torch.zeros(1)),),)

    with pytest.raises(ValueError, match=r"expected.*type 'image'.*'video'"):
        adapter.encode_output_state(wrong_type, {})

    assert codec.calls == 0
    assert adapter.geometry_validation is None


def test_encode_output_state_rejects_candidate_before_stochastic_preparation() -> None:
    preparer = _StochasticConditionPreparer()
    adapter = _PreparedLifecycleAdapter(_Codec(), preparer)
    generator = torch.Generator().manual_seed(7)
    original_state = generator.get_state().clone()
    wrong_type = ((_DecodedMedia(type="video", payload=torch.zeros(1)),),)

    with pytest.raises(ValueError, match=r"expected.*type 'image'.*'video'"):
        adapter.encode_output_state(wrong_type, {}, generator)

    assert preparer.calls == 0
    assert torch.equal(generator.get_state(), original_state)


@pytest.mark.parametrize(
    ("condition", "generator", "message"),
    [
        ([], None, r"condition to be Mapping"),
        ({}, object(), r"generator to be torch.Generator or None"),
    ],
)
def test_encode_output_state_rejects_invalid_wrapper_arguments(
    condition: Any,
    generator: Any,
    message: str,
) -> None:
    codec = _Codec()
    adapter = _LifecycleAdapter(codec)

    with pytest.raises(TypeError, match=message):
        adapter.encode_output_state(_image_batch(1), condition, generator)

    assert codec.calls == 0


def test_encode_output_state_validates_codec_result_before_geometry_hook() -> None:
    codec = _Codec(result=_encoded_image_batch(1, component_name="other"))
    adapter = _LifecycleAdapter(codec)

    with pytest.raises(ValueError, match=r"component order \('latent',\).+\('other',\)"):
        adapter.encode_output_state(_image_batch(1), {})

    assert adapter.geometry_validation is None


def test_encode_output_state_does_not_hide_attached_codec_tensor_during_cast() -> None:
    attached = torch.ones(1, 2, requires_grad=True)
    codec = _Codec(result=_encoded_image_batch(1, tensor=attached))
    adapter = _LifecycleAdapter(codec, latent_storage_dtype="fp16")

    with pytest.raises(ValueError, match=r"detached.*clean_state component 'latent'"):
        adapter.encode_output_state(_image_batch(1), {})


def test_decode_output_state_routes_context_through_existing_decoder_signature() -> None:
    adapter = _LifecycleAdapter(_Codec())
    encoded = _encoded_image_batch(1)

    decoded = adapter.decode_output_state(encoded, output_type="pt")

    latent = encoded.clean_state.components["latent"]
    assert decoded is latent
    assert adapter.decode_call == (latent, 8, "pt")


@pytest.mark.parametrize(
    ("encoded", "output_type", "message"),
    [
        (object(), "pil", r"expected encoded output state"),
        (_encoded_image_batch(1), 3, r"expected output_type to be str"),
        (_encoded_image_batch(1), "latent", r"expected output_type in"),
        (
            _encoded_image_batch(1, component_name="video"),
            "pil",
            r"exactly one 'latent' component",
        ),
    ],
)
def test_decode_output_state_validates_shared_boundary_arguments(
    encoded: Any,
    output_type: Any,
    message: str,
) -> None:
    adapter = _LifecycleAdapter(_Codec())

    with pytest.raises((TypeError, ValueError), match=message):
        adapter.decode_output_state(encoded, output_type=output_type)

    assert adapter.decode_call is None


def test_default_geometry_hook_rejects_self_reported_codec_geometry() -> None:
    adapter = _DefaultGeometryAdapter(_Codec())

    with pytest.raises(
        NotImplementedError,
        match=r"must override _validate_encoded_output_geometry.*configured",
    ):
        adapter.encode_output_state(_image_batch(1), {})
