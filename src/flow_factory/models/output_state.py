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

"""Adapter-owned target-media encoding contracts and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass, is_dataclass
from types import MappingProxyType
from typing import (
    Any,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

import torch

from ..contracts import (
    NON_MODEL_CONDITION_KEYS,
    BatchCapability,
    DecodedMediaLike,
    MediaGeometry,
    PipelineIOContract,
    validate_media_geometry,
    validate_media_metadata,
)
from ..samples import LatentState
from ..utils.media import require_media_payload

DecodedMediaBatch = Tuple[Tuple[DecodedMediaLike, ...], ...]


OUTPUT_STATE_OWNED_KEYS = frozenset(
    {
        "clean_state",
        "decode_context",
        "forward_context",
        "geometry_signatures",
    }
)
OUTPUT_FORWARD_CONTEXT_RESERVED_KEYS = NON_MODEL_CONDITION_KEYS | OUTPUT_STATE_OWNED_KEYS


# Compatibility alias for adapter code that predates the common media contract.
MediaGeometrySignature = MediaGeometry


@dataclass(frozen=True, slots=True)
class GeometrySignature:
    """Describe one sample's exact ordered output-media geometry.

    Args:
        media: Geometry entries in the pipeline contract's exact output order.
    """

    media: Tuple[MediaGeometry, ...]

    def __post_init__(self) -> None:
        """Validate an immutable, non-empty media geometry sequence."""
        _require_exact_tuple(self.media, MediaGeometry, "GeometrySignature.media")
        if not self.media:
            raise ValueError("expected GeometrySignature.media to contain at least one item")


@dataclass(frozen=True, slots=True)
class EncodedOutputState:
    """Bundle a detached clean latent state with adapter-owned contexts.

    Args:
        clean_state: Batched clean target latents in adapter component order.
        forward_context: Output-derived fields that may enter model forward.
        decode_context: Geometry, rate, and model metadata routed by
            ``BaseAdapter.decode_output_state`` into the existing decoder. The wrapper filters
            validation-only fields that ``decode_latents`` does not accept.
        geometry_signatures: One exact output-geometry signature per batch sample.

    Note:
        The result freezes its ownership shell and copies both outer context mappings.
        Tensor leaves and ``LatentState`` are retained without cloning and are revalidated
        immediately before an offline objective consumes them.
    """

    clean_state: LatentState
    forward_context: Mapping[str, Any]
    decode_context: Mapping[str, Any]
    geometry_signatures: Tuple[GeometrySignature, ...]

    def __post_init__(self) -> None:
        """Freeze context mappings and reject malformed result containers."""
        if not isinstance(self.clean_state, LatentState):
            raise TypeError(
                "expected EncodedOutputState.clean_state to be LatentState, "
                f"received {type(self.clean_state).__name__}"
            )
        frozen_forward_context = _freeze_context_mapping(
            self.forward_context,
            "EncodedOutputState.forward_context",
        )
        rejected = tuple(
            sorted(set(frozen_forward_context).intersection(OUTPUT_FORWARD_CONTEXT_RESERVED_KEYS))
        )
        if rejected:
            raise ValueError(
                "EncodedOutputState.forward_context contains non-model or state-owned fields "
                f"{rejected}"
            )
        object.__setattr__(self, "forward_context", frozen_forward_context)
        object.__setattr__(
            self,
            "decode_context",
            _freeze_context_mapping(
                self.decode_context,
                "EncodedOutputState.decode_context",
            ),
        )
        _require_exact_tuple(
            self.geometry_signatures,
            GeometrySignature,
            "EncodedOutputState.geometry_signatures",
        )
        if not self.geometry_signatures:
            raise ValueError(
                "expected EncodedOutputState.geometry_signatures to contain at least one sample"
            )


@runtime_checkable
class OutputStateCodec(Protocol):
    """Define adapter-owned on-the-fly target-media encoding."""

    @property
    def required_components(self) -> Tuple[str, ...]:
        """Return adapter component names required while encoding targets."""
        ...

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        """Encode one validated media batch into detached clean model state."""
        ...


def validate_output_candidate_batch(
    media_batch: object,
    contract: PipelineIOContract,
) -> DecodedMediaBatch:
    """Validate decoded target candidates against exact pipeline output semantics.

    Args:
        media_batch: Tuple of samples, each containing an exact ordered media tuple.
        contract: Neutral pipeline I/O declaration owned by the adapter.

    Returns:
        The validated immutable media batch.

    Note:
        Raw decoded media expose rates but not canonical encoded dimensions. Uniform geometry
        is therefore enforced against codec-produced signatures by
        :func:`validate_encoded_output_state`.
    """
    _require_contract(contract)
    if type(media_batch) is not tuple:
        raise TypeError(
            "expected output media batch to be tuple, "
            f"received {type(media_batch).__name__}: {media_batch!r}"
        )
    if not media_batch:
        raise ValueError("expected output media batch to contain at least one sample")
    if contract.batch_capability is BatchCapability.SINGLE_SAMPLE and len(media_batch) != 1:
        raise ValueError(
            "single_sample pipeline expected output media batch size 1, "
            f"received {len(media_batch)}"
        )

    expected_items = contract.output_media.items
    for sample_index, candidate in enumerate(media_batch):
        if type(candidate) is not tuple:
            raise TypeError(
                f"expected output media sample {sample_index} to be tuple, "
                f"received {type(candidate).__name__}: {candidate!r}"
            )
        if len(candidate) != len(expected_items):
            raise ValueError(
                f"expected output media sample {sample_index} to contain exact sequence length "
                f"{len(expected_items)}, received {len(candidate)}"
            )
        for media_index, (media, expected) in enumerate(zip(candidate, expected_items)):
            identifier = f"output media sample {sample_index} item {media_index}"
            if not isinstance(media, DecodedMediaLike):
                raise TypeError(
                    f"expected DecodedMediaLike for {identifier}, "
                    f"received {type(media).__name__}"
                )
            validate_media_metadata(media, expected, identifier=identifier)
            if media.payload is None:
                raise ValueError(f"expected decoded payload for {identifier}, received None")
            require_media_payload(
                media.payload,
                representation=expected.representation,
                source=f"{identifier} payload",
            )
    return cast(DecodedMediaBatch, media_batch)


def validate_encoded_output_state(
    encoded: object,
    *,
    contract: PipelineIOContract,
    expected_component_order: Tuple[str, ...],
    expected_batch_size: int,
    device: Union[torch.device, str],
) -> EncodedOutputState:
    """Validate an encoded target result before an offline objective consumes it.

    Args:
        encoded: Result returned by an adapter's output-state codec.
        contract: Neutral pipeline I/O declaration owned by the adapter.
        expected_component_order: Adapter trajectory component order.
        expected_batch_size: Number of validated target candidates encoded together.
        device: Device on which model-facing encoded tensors must reside.

    Returns:
        The validated encoded output state.
    """
    if not isinstance(encoded, EncodedOutputState):
        raise TypeError(
            "expected encoded output to be EncodedOutputState, "
            f"received {type(encoded).__name__}"
        )
    _require_contract(contract)
    _validate_component_names(expected_component_order, "expected_component_order")
    _require_positive_int(expected_batch_size, "expected_batch_size")
    target_device = torch.device(device)

    clean_state = encoded.clean_state
    if clean_state.component_names != expected_component_order:
        raise ValueError(
            f"expected encoded clean_state component order {expected_component_order}, "
            f"received {clean_state.component_names}"
        )
    for name in expected_component_order:
        component = clean_state.components.get(name)
        if not isinstance(component, torch.Tensor):
            raise TypeError(
                f"expected clean_state component {name!r} to be torch.Tensor, "
                f"received {type(component).__name__}"
            )
        if component.ndim < 1 or component.shape[0] != expected_batch_size:
            raise ValueError(
                f"expected clean_state component {name!r} to use batch size "
                f"{expected_batch_size}, received shape {tuple(component.shape)}"
            )
        if component.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(
                "expected float16, bfloat16, or float32 clean_state component "
                f"{name!r}, received {component.dtype}"
            )
        _validate_tensor_runtime(
            component,
            target_device,
            f"clean_state component {name!r}",
        )
        if not bool(torch.isfinite(component).all()):
            raise ValueError(f"clean_state component {name!r} contains non-finite values")

    if clean_state.active_masks is not None:
        if tuple(clean_state.active_masks) != expected_component_order:
            raise ValueError(
                f"expected clean_state active mask order {expected_component_order}, "
                f"received {tuple(clean_state.active_masks)}"
            )
        for name, mask in clean_state.active_masks.items():
            if not isinstance(mask, torch.Tensor):
                raise TypeError(
                    f"expected clean_state active mask {name!r} to be torch.Tensor, "
                    f"received {type(mask).__name__}"
                )
            if mask.ndim < 1 or mask.shape[0] != expected_batch_size:
                raise ValueError(
                    f"expected clean_state active mask {name!r} to use batch size "
                    f"{expected_batch_size}, received shape {tuple(mask.shape)}"
                )
            if mask.dtype is not torch.bool:
                raise TypeError(
                    f"expected clean_state active mask {name!r} dtype torch.bool, "
                    f"received {mask.dtype}"
                )
            component_shape = clean_state.components[name].shape
            if mask.ndim != len(component_shape) or any(
                mask_dim not in (1, component_dim)
                for mask_dim, component_dim in zip(mask.shape, component_shape)
            ):
                raise ValueError(
                    f"expected clean_state active mask {name!r} broadcastable to component "
                    f"shape {tuple(component_shape)}, received {tuple(mask.shape)}"
                )
            _validate_tensor_runtime(mask, target_device, f"clean_state active mask {name!r}")

    if len(encoded.geometry_signatures) != expected_batch_size:
        raise ValueError(
            "expected one geometry signature per encoded sample "
            f"({expected_batch_size}), received {len(encoded.geometry_signatures)}"
        )
    for sample_index, signature in enumerate(encoded.geometry_signatures):
        _validate_geometry_signature(signature, contract, sample_index)
    has_different_geometry = any(
        signature != encoded.geometry_signatures[0] for signature in encoded.geometry_signatures[1:]
    )
    if contract.batch_capability is BatchCapability.UNIFORM and has_different_geometry:
        raise ValueError(
            "uniform pipeline expected identical geometry signatures across the encoded batch, "
            f"received {encoded.geometry_signatures}"
        )
    if (
        contract.batch_capability is BatchCapability.RAGGED
        and has_different_geometry
        and clean_state.active_masks is None
    ):
        raise ValueError(
            "ragged encoded batches with different geometry signatures require active masks "
            "so padded latent elements cannot contribute to the objective"
        )
    if contract.batch_capability is BatchCapability.SINGLE_SAMPLE and expected_batch_size != 1:
        raise ValueError(
            "single_sample pipeline expected encoded batch size 1, "
            f"received {expected_batch_size}"
        )

    _validate_context_tensor_tree(
        encoded.forward_context,
        expected_device=target_device,
        identifier="EncodedOutputState.forward_context",
        active_container_ids=set(),
    )
    _validate_context_tensor_tree(
        encoded.decode_context,
        expected_device=None,
        identifier="EncodedOutputState.decode_context",
        active_container_ids=set(),
    )
    return encoded


def validate_codec_required_components(
    codec: object,
    available_components: Sequence[str],
) -> Tuple[str, ...]:
    """Validate codec component requirements against one adapter runtime.

    Args:
        codec: Structural output-state codec instance.
        available_components: Canonical component names exposed by the adapter runtime.

    Returns:
        The codec's validated required component tuple.
    """
    encode = getattr(codec, "encode_output_state", None)
    if not callable(encode):
        raise TypeError(
            "expected output-state codec with callable encode_output_state, "
            f"received {type(codec).__name__}"
        )
    required_components = getattr(codec, "required_components", None)
    _validate_component_names(required_components, "codec.required_components", allow_empty=True)
    if isinstance(available_components, (str, bytes)) or not isinstance(
        available_components, Sequence
    ):
        raise TypeError(
            "expected available_components to be a sequence of strings, "
            f"received {type(available_components).__name__}: {available_components!r}"
        )
    available = tuple(available_components)
    _validate_component_names(available, "available_components", allow_empty=True)
    unknown = tuple(name for name in required_components if name not in available)
    if unknown:
        raise ValueError(
            f"codec requires unavailable adapter components {unknown}; available={available}"
        )
    return required_components


def _require_contract(contract: object) -> None:
    if not isinstance(contract, PipelineIOContract):
        raise TypeError(
            "expected contract to be PipelineIOContract, " f"received {type(contract).__name__}"
        )


def _require_exact_tuple(value: object, item_type: type, identifier: str) -> None:
    if type(value) is not tuple:
        raise TypeError(
            f"expected {identifier} to be tuple, received {type(value).__name__}: {value!r}"
        )
    for index, item in enumerate(value):
        if type(item) is not item_type:
            raise TypeError(
                f"expected {identifier}[{index}] to be {item_type.__name__}, "
                f"received {type(item).__name__}"
            )


def _require_positive_int(value: object, identifier: str) -> None:
    if type(value) is not int:
        raise TypeError(
            f"expected positive int for {identifier}, received {type(value).__name__}: {value!r}"
        )
    if value <= 0:
        raise ValueError(f"expected positive int for {identifier}, received {value}")


def _freeze_context_mapping(value: object, identifier: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"expected Mapping[str, Any] for {identifier}, "
            f"received {type(value).__name__}: {value!r}"
        )
    copied = {}
    for key, item in value.items():
        if type(key) is not str:
            raise TypeError(
                f"expected string keys for {identifier}, received " f"{type(key).__name__}: {key!r}"
            )
        if not key:
            raise ValueError(f"expected non-empty string keys for {identifier}")
        copied[key] = item
    return MappingProxyType(copied)


def _validate_component_names(
    names: object,
    identifier: str,
    *,
    allow_empty: bool = False,
) -> None:
    if type(names) is not tuple:
        raise TypeError(
            f"expected {identifier} to be tuple, received {type(names).__name__}: {names!r}"
        )
    for index, name in enumerate(names):
        if type(name) is not str:
            raise TypeError(
                f"expected {identifier}[{index}] to be str, received "
                f"{type(name).__name__}: {name!r}"
            )
        if not name:
            raise ValueError(f"expected non-empty component name for {identifier}[{index}]")
    if not names and not allow_empty:
        raise ValueError(f"expected {identifier} to contain at least one component")
    if len(set(names)) != len(names):
        raise ValueError(f"expected unique component names for {identifier}, received {names}")


def _validate_geometry_signature(
    signature: GeometrySignature,
    contract: PipelineIOContract,
    sample_index: int,
) -> None:
    expected_items = contract.output_media.items
    if len(signature.media) != len(expected_items):
        raise ValueError(
            f"expected geometry signature {sample_index} exact media sequence length "
            f"{len(expected_items)}, received {len(signature.media)}"
        )
    for media_index, (geometry, expected) in enumerate(zip(signature.media, expected_items)):
        identifier = f"geometry signature {sample_index} item {media_index}"
        validate_media_geometry(geometry, expected, identifier=identifier)


def _validate_tensor_runtime(
    tensor: torch.Tensor,
    device: torch.device,
    identifier: str,
) -> None:
    if tensor.device != device:
        raise ValueError(f"expected {identifier} on device {device}, received {tensor.device}")
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise ValueError(
            f"expected detached no-grad tensor for {identifier}, received "
            f"requires_grad={tensor.requires_grad}, grad_fn={tensor.grad_fn}"
        )


def _validate_context_tensor_tree(
    value: Any,
    expected_device: Optional[torch.device],
    identifier: str,
    active_container_ids: set[int],
) -> None:
    if isinstance(value, torch.Tensor):
        if expected_device is not None and value.device != expected_device:
            raise ValueError(
                f"expected {identifier} on device {expected_device}, received {value.device}"
            )
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError(
                f"expected detached no-grad tensor for {identifier}, received "
                f"requires_grad={value.requires_grad}, grad_fn={value.grad_fn}"
            )
        return
    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in active_container_ids:
            raise ValueError(f"expected acyclic context tree for {identifier}")
        active_container_ids.add(container_id)
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    f"expected string nested context key for {identifier}, "
                    f"received {type(key).__name__}: {key!r}"
                )
            if not key:
                raise ValueError(f"expected non-empty nested context key for {identifier}")
            _validate_context_tensor_tree(
                item,
                expected_device,
                f"{identifier}[{key!r}]",
                active_container_ids,
            )
        active_container_ids.remove(container_id)
        return
    if isinstance(value, (list, tuple)):
        container_id = id(value)
        if container_id in active_container_ids:
            raise ValueError(f"expected acyclic context tree for {identifier}")
        active_container_ids.add(container_id)
        for index, item in enumerate(value):
            _validate_context_tensor_tree(
                item,
                expected_device,
                f"{identifier}[{index}]",
                active_container_ids,
            )
        active_container_ids.remove(container_id)
        return
    if isinstance(value, Set) or is_dataclass(value):
        raise TypeError(
            f"expected tensor context tree for {identifier} to use Mapping, list, tuple, "
            f"or scalar leaves, received unsupported {type(value).__name__}"
        )
    if value is not None and not isinstance(
        value,
        (str, bytes, bool, int, float, torch.dtype, torch.device),
    ):
        raise TypeError(
            f"expected scalar leaf for {identifier}, received unsupported "
            f"{type(value).__name__}: {value!r}"
        )


__all__ = [
    "DecodedMediaBatch",
    "EncodedOutputState",
    "GeometrySignature",
    "MediaGeometrySignature",
    "OUTPUT_FORWARD_CONTEXT_RESERVED_KEYS",
    "OUTPUT_STATE_OWNED_KEYS",
    "OutputStateCodec",
    "validate_codec_required_components",
    "validate_encoded_output_state",
    "validate_output_candidate_batch",
]
