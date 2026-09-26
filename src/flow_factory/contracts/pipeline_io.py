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

"""Dependency-neutral declarations for model pipeline inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .media import (
    MediaFormat,
    MediaMetadataLike,
    MediaType,
    RateRequirement,
    validate_media_metadata,
)


class InputMediaBinding(str, Enum):
    """Describe how input media are bound to model-facing arguments."""

    GROUPED_BY_TYPE = "grouped_by_type"
    ORDERED_REFERENCES = "ordered_references"


class InputMediaOrder(str, Enum):
    """Describe which input-media ordering carries semantic meaning."""

    INSENSITIVE = "insensitive"
    WITHIN_TYPE = "within_type"
    GLOBAL = "global"


class NegativePromptPolicy(str, Enum):
    """Declare whether a pipeline accepts a negative prompt."""

    UNSUPPORTED = "unsupported"
    OPTIONAL = "optional"
    REQUIRED = "required"


class GeometrySource(str, Enum):
    """Declare where output geometry is resolved for one pipeline operation."""

    CONFIGURED = "configured"
    INPUT_MEDIA = "input_media"
    OUTPUT_MEDIA = "output_media"
    PRIMARY_OUTPUT_MEDIA = "primary_output_media"


class BatchCapability(str, Enum):
    """Describe the media-layout uniformity supported by a pipeline operation."""

    UNIFORM = "uniform"
    RAGGED = "ragged"
    SINGLE_SAMPLE = "single_sample"


@runtime_checkable
class DecodedMediaLike(MediaMetadataLike, Protocol):
    """Structural boundary for decoded media without importing a dataset type."""

    @property
    def payload(self) -> Any:
        """Return the decoded CPU-side media payload."""
        ...


@runtime_checkable
class InputMediaLike(MediaMetadataLike, Protocol):
    """Structural media reference accepted by input-contract validation."""


@runtime_checkable
class OutputMediaLike(MediaMetadataLike, Protocol):
    """Structural output-media metadata accepted before payload decoding."""


@runtime_checkable
class ModelInputLike(Protocol):
    """Structural normalized input consumed by a pipeline I/O contract."""

    @property
    def prompt(self) -> str:
        """Return the positive text prompt."""
        ...

    @property
    def negative_prompt(self) -> str | None:
        """Return the optional negative text prompt."""
        ...

    @property
    def media(self) -> tuple[InputMediaLike, ...]:
        """Return input media in public record order."""
        ...


@dataclass(frozen=True, slots=True)
class InputMediaRule:
    """Declare count and optional semantic slots for one input media format."""

    format: MediaFormat
    min_count: int
    max_count: int | None
    slots: tuple[str, ...] = ()
    required_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate strict cardinality types and bounds."""
        _require_instance(self.format, MediaFormat, "format")
        _require_non_negative_int(self.min_count, "min_count")
        _require_string_tuple(self.slots, "slots")
        _require_string_tuple(self.required_slots, "required_slots")
        if self.max_count is not None:
            _require_non_negative_int(self.max_count, "max_count")
            if self.max_count == 0:
                raise ValueError("max_count=0 is not canonical; omit the media rule instead")
            if self.max_count < self.min_count:
                raise ValueError(
                    f"expected max_count >= min_count, received "
                    f"min_count={self.min_count} and max_count={self.max_count}"
                )
        if len(set(self.slots)) != len(self.slots):
            raise ValueError("input media slots must be unique within one media rule")
        if len(set(self.required_slots)) != len(self.required_slots):
            raise ValueError("required input media slots must be unique")
        unknown_required_slots = tuple(
            slot for slot in self.required_slots if slot not in self.slots
        )
        if unknown_required_slots:
            raise ValueError(
                "required input media slots must be declared in slots; "
                f"unknown={unknown_required_slots!r}"
            )
        canonical_required_slots = tuple(slot for slot in self.slots if slot in self.required_slots)
        if self.required_slots != canonical_required_slots:
            raise ValueError(
                "required input media slots must use declared slot order "
                f"{canonical_required_slots!r}, received {self.required_slots!r}"
            )
        if self.slots:
            if self.max_count != len(self.slots):
                raise ValueError(
                    "a slotted input media rule requires max_count to equal the number "
                    f"of slots, received max_count={self.max_count!r}, slots={self.slots!r}"
                )
            if len(self.required_slots) > self.min_count:
                raise ValueError(
                    "required slot count cannot exceed min_count, received "
                    f"required_slots={self.required_slots!r}, min_count={self.min_count}"
                )
        elif self.required_slots:
            raise ValueError("required input media slots cannot be declared without slots")


@dataclass(frozen=True, slots=True)
class InputMediaSpec:
    """Declare input-media types, counts, ordering, and argument binding."""

    rules: tuple[InputMediaRule, ...]
    binding: InputMediaBinding
    order: InputMediaOrder
    min_total_count: int | None = None
    max_total_count: int | None = None
    required_any_types: tuple[MediaType, ...] = ()

    def __post_init__(self) -> None:
        """Validate a canonical and coherent input-media declaration."""
        _require_tuple(self.rules, InputMediaRule, "rules")
        _require_enum(self.binding, InputMediaBinding, "binding")
        _require_enum(self.order, InputMediaOrder, "order")
        if self.min_total_count is not None:
            _require_non_negative_int(self.min_total_count, "min_total_count")
        if self.max_total_count is not None:
            _require_non_negative_int(self.max_total_count, "max_total_count")
            if self.max_total_count == 0:
                raise ValueError(
                    "max_total_count=0 is not canonical; omit all input media rules instead"
                )
        if (
            self.min_total_count is not None
            and self.max_total_count is not None
            and self.max_total_count < self.min_total_count
        ):
            raise ValueError(
                "expected max_total_count >= min_total_count, received "
                f"min_total_count={self.min_total_count} and "
                f"max_total_count={self.max_total_count}"
            )
        _require_enum_tuple(self.required_any_types, MediaType, "required_any_types")
        if len(set(self.required_any_types)) != len(self.required_any_types):
            raise ValueError("required_any_types must contain each media type at most once")
        canonical_required_any_types = tuple(
            media_type for media_type in MediaType if media_type in self.required_any_types
        )
        if self.required_any_types != canonical_required_any_types:
            raise ValueError(
                "required_any_types must use canonical type order "
                f"{canonical_required_any_types}, received {self.required_any_types}"
            )

        media_types = tuple(rule.format.type for rule in self.rules)
        if len(set(media_types)) != len(media_types):
            raise ValueError("input media rules must contain each media type at most once")
        canonical_media_types = tuple(
            media_type for media_type in MediaType if media_type in media_types
        )
        if media_types != canonical_media_types:
            raise ValueError(
                "input media rules must use canonical type order "
                f"{canonical_media_types}, received {media_types}"
            )
        declared_slots = tuple(slot for rule in self.rules for slot in rule.slots)
        if len(set(declared_slots)) != len(declared_slots):
            raise ValueError("input media slot names must be unique across media rules")
        if declared_slots and self.binding is not InputMediaBinding.GROUPED_BY_TYPE:
            raise ValueError("semantic input media slots require grouped_by_type binding")
        if (
            any(len(rule.slots) > 1 for rule in self.rules)
            and self.order is not InputMediaOrder.WITHIN_TYPE
        ):
            raise ValueError(
                "multi-slot input media rules require within_type ordering because "
                "unslotted media uses positional fallback"
            )
        unknown_required_types = tuple(
            media_type for media_type in self.required_any_types if media_type not in media_types
        )
        if unknown_required_types:
            raise ValueError(
                "required_any_types must be declared by input media rules; "
                f"unknown={unknown_required_types!r}"
            )
        minimum_from_rules = sum(rule.min_count for rule in self.rules)
        if (
            self.rules
            and self.max_total_count is not None
            and self.max_total_count < minimum_from_rules
        ):
            raise ValueError(
                "max_total_count cannot be smaller than the sum of per-type minimums, "
                f"received max_total_count={self.max_total_count}, "
                f"per_type_minimum={minimum_from_rules}"
            )
        if (
            self.rules
            and self.min_total_count is not None
            and all(rule.max_count is not None for rule in self.rules)
        ):
            maximum_from_rules = sum(
                rule.max_count for rule in self.rules if rule.max_count is not None
            )
            if self.min_total_count > maximum_from_rules:
                raise ValueError(
                    "min_total_count cannot exceed the sum of finite per-type maximums, "
                    f"received min_total_count={self.min_total_count}, "
                    f"per_type_maximum={maximum_from_rules}"
                )
        if not self.rules:
            if (
                self.min_total_count is not None
                or self.max_total_count is not None
                or self.required_any_types
            ):
                raise ValueError("media-free inputs cannot declare aggregate media constraints")
            if self.binding is not InputMediaBinding.GROUPED_BY_TYPE:
                raise ValueError("media-free inputs must use grouped_by_type binding")
            if self.order is not InputMediaOrder.INSENSITIVE:
                raise ValueError("media-free inputs must use insensitive ordering")
            return
        if self.binding is InputMediaBinding.ORDERED_REFERENCES:
            if self.order is not InputMediaOrder.GLOBAL:
                raise ValueError("ordered_references binding requires global input ordering")
        elif self.order is InputMediaOrder.GLOBAL:
            raise ValueError("global input ordering requires ordered_references binding")


@dataclass(frozen=True, slots=True)
class OutputMediaSequence:
    """Declare the exact ordered media sequence produced by a pipeline."""

    items: tuple[MediaFormat, ...]

    def __post_init__(self) -> None:
        """Validate a non-empty, immutable output-media sequence."""
        _require_tuple(self.items, MediaFormat, "items")
        if not self.items:
            raise ValueError("output media sequence must contain at least one item")


@dataclass(frozen=True, slots=True)
class PipelineIOContract:
    """Declare model-agnostic pipeline input and decoded-output semantics."""

    input_media: InputMediaSpec
    negative_prompt: NegativePromptPolicy
    output_media: OutputMediaSequence
    geometry_source: GeometrySource
    batch_capability: BatchCapability

    def __post_init__(self) -> None:
        """Validate strict types for every pipeline I/O declaration."""
        _require_instance(self.input_media, InputMediaSpec, "input_media")
        _require_enum(self.negative_prompt, NegativePromptPolicy, "negative_prompt")
        _require_instance(self.output_media, OutputMediaSequence, "output_media")
        _require_enum(self.geometry_source, GeometrySource, "geometry_source")
        _require_enum(self.batch_capability, BatchCapability, "batch_capability")
        guarantees_input_media = (
            any(rule.min_count > 0 for rule in self.input_media.rules)
            or bool(self.input_media.min_total_count)
            or bool(self.input_media.required_any_types)
        )
        if self.geometry_source is GeometrySource.INPUT_MEDIA and not guarantees_input_media:
            raise ValueError(
                "input_media geometry requires input constraints that guarantee at least "
                "one media item"
            )


def validate_pipeline_model_input(
    model_input: ModelInputLike,
    contract: PipelineIOContract,
) -> None:
    """Validate one normalized input against model-declared pipeline semantics.

    The function depends only on structural protocols, so the dataset schema and
    model adapter remain independent. Callers must run it before preprocessing;
    otherwise an adapter may silently ignore unsupported conditioning media.

    Args:
        model_input: Structurally normalized prompt and input-media metadata.
        contract: Adapter-owned pipeline input/output declaration.

    Returns:
        None after successful validation.

    Raises:
        TypeError: If the input or any field violates boundary types.
        ValueError: If prompt or input media violates the pipeline contract.
    """
    _require_instance(contract, PipelineIOContract, "contract")
    if not isinstance(model_input, ModelInputLike):
        raise TypeError(
            "expected model_input to implement ModelInputLike, received "
            f"{type(model_input).__name__}: {model_input!r}"
        )
    if type(model_input.prompt) is not str:
        raise TypeError(
            "expected model_input.prompt to be str, received "
            f"{type(model_input.prompt).__name__}: {model_input.prompt!r}"
        )
    negative_prompt = model_input.negative_prompt
    if negative_prompt is not None and type(negative_prompt) is not str:
        raise TypeError(
            "expected model_input.negative_prompt to be str or None, received "
            f"{type(negative_prompt).__name__}: {negative_prompt!r}"
        )
    if contract.negative_prompt is NegativePromptPolicy.UNSUPPORTED and negative_prompt is not None:
        raise ValueError("pipeline does not support negative_prompt")
    if contract.negative_prompt is NegativePromptPolicy.REQUIRED and negative_prompt is None:
        raise ValueError("pipeline requires negative_prompt")

    media = model_input.media
    if type(media) is not tuple:
        raise TypeError(
            "expected model_input.media to be tuple, received " f"{type(media).__name__}: {media!r}"
        )
    rules_by_type = {rule.format.type.value: rule for rule in contract.input_media.rules}
    counts = {media_type: 0 for media_type in rules_by_type}
    for index, item in enumerate(media):
        if not isinstance(item, InputMediaLike):
            raise TypeError(
                f"expected model_input.media[{index}] to implement InputMediaLike, "
                f"received {type(item).__name__}: {item!r}"
            )
        media_type = item.type
        if type(media_type) is not str:
            raise TypeError(
                f"expected model_input.media[{index}].type to be str, received "
                f"{type(media_type).__name__}: {media_type!r}"
            )
        rule = rules_by_type.get(media_type)
        if rule is None:
            raise ValueError(
                f"pipeline does not accept input media type {media_type!r} at index {index}; "
                f"accepted types={tuple(rules_by_type)!r}"
            )
        validate_media_metadata(
            item,
            rule.format,
            identifier=f"pipeline input media[{index}]",
        )
        counts[media_type] += 1

    for media_type, rule in rules_by_type.items():
        count = counts[media_type]
        if count < rule.min_count:
            raise ValueError(
                f"pipeline requires at least {rule.min_count} input {media_type!r} item(s), "
                f"received {count}"
            )
        if rule.max_count is not None and count > rule.max_count:
            raise ValueError(
                f"pipeline accepts at most {rule.max_count} input {media_type!r} item(s), "
                f"received {count}"
            )

    total_count = len(media)
    input_media = contract.input_media
    if input_media.min_total_count is not None and total_count < input_media.min_total_count:
        raise ValueError(
            f"pipeline requires at least {input_media.min_total_count} input media item(s) "
            f"in total, received {total_count}"
        )
    if input_media.max_total_count is not None and total_count > input_media.max_total_count:
        raise ValueError(
            f"pipeline accepts at most {input_media.max_total_count} input media item(s) "
            f"in total, received {total_count}"
        )
    if input_media.required_any_types and not any(
        counts[media_type.value] > 0 for media_type in input_media.required_any_types
    ):
        accepted = tuple(media_type.value for media_type in input_media.required_any_types)
        raise ValueError(
            "pipeline requires at least one input media item whose type is in " f"{accepted!r}"
        )
    _resolve_pipeline_input_media_slots_unchecked(media, contract)


def resolve_pipeline_input_media_slots(
    model_input: ModelInputLike,
    contract: PipelineIOContract,
) -> tuple[str | None, ...]:
    """Resolve explicit and positional input media onto adapter-declared slots.

    Explicit slot values reserve their named positions first. Unslotted media
    then fill the remaining positions in declaration order, preserving the V2
    positional shorthand while making sparse bindings such as a last-frame-only
    request unambiguous.
    """
    validate_pipeline_model_input(model_input, contract)
    return _resolve_pipeline_input_media_slots_unchecked(model_input.media, contract)


def _resolve_pipeline_input_media_slots_unchecked(
    media: tuple[InputMediaLike, ...],
    contract: PipelineIOContract,
) -> tuple[str | None, ...]:
    assignments: list[str | None] = [None] * len(media)
    rules_by_type = {rule.format.type.value: rule for rule in contract.input_media.rules}
    for media_type, rule in rules_by_type.items():
        indices = [index for index, item in enumerate(media) if item.type == media_type]
        if not rule.slots:
            for index in indices:
                slot = getattr(media[index], "slot", None)
                if slot is not None:
                    raise ValueError(
                        f"pipeline input media[{index}] declares slot={slot!r}, but "
                        f"media type {media_type!r} has no semantic slots"
                    )
            continue

        claimed: dict[str, int] = {}
        unassigned_indices = []
        for index in indices:
            slot = getattr(media[index], "slot", None)
            if slot is None:
                unassigned_indices.append(index)
                continue
            if type(slot) is not str:
                raise TypeError(
                    f"expected model_input.media[{index}].slot to be str or None, "
                    f"received {type(slot).__name__}: {slot!r}"
                )
            if slot not in rule.slots:
                raise ValueError(
                    f"pipeline input media[{index}] slot {slot!r} is not accepted for "
                    f"media type {media_type!r}; accepted slots={rule.slots!r}"
                )
            if slot in claimed:
                raise ValueError(
                    f"pipeline input media slot {slot!r} is assigned more than once at "
                    f"indices {claimed[slot]} and {index}"
                )
            assignments[index] = slot
            claimed[slot] = index

        remaining_slots = [slot for slot in rule.slots if slot not in claimed]
        for index, slot in zip(unassigned_indices, remaining_slots):
            assignments[index] = slot
            claimed[slot] = index
        missing_required_slots = tuple(slot for slot in rule.required_slots if slot not in claimed)
        if missing_required_slots:
            raise ValueError(
                f"pipeline requires input media slots {missing_required_slots!r} for "
                f"media type {media_type!r}"
            )
    return tuple(assignments)


def validate_pipeline_output_candidate(
    media: tuple[OutputMediaLike, ...],
    contract: PipelineIOContract,
) -> None:
    """Validate undecoded output metadata against exact pipeline semantics.

    This dependency-neutral boundary lets a dataset reject incompatible target,
    chosen, or rejected media before condition preprocessing or payload decoding.
    Model-specific geometry remains owned by later boundaries.

    Args:
        media: Exact ordered media tuple for one output candidate.
        contract: Adapter-owned pipeline input/output declaration.

    Returns:
        None after successful validation.

    Raises:
        TypeError: If the candidate or its metadata violates boundary types.
        ValueError: If media order, modality, or rate violates the contract.
    """
    _require_instance(contract, PipelineIOContract, "contract")
    if type(media) is not tuple:
        raise TypeError(
            "expected output candidate media to be tuple, received "
            f"{type(media).__name__}: {media!r}"
        )
    expected_items = contract.output_media.items
    if len(media) != len(expected_items):
        raise ValueError(
            "expected output candidate exact media sequence length "
            f"{len(expected_items)}, received {len(media)}"
        )
    for index, (item, expected) in enumerate(zip(media, expected_items)):
        if not isinstance(item, OutputMediaLike):
            raise TypeError(
                f"expected output candidate media[{index}] to implement OutputMediaLike, "
                f"received {type(item).__name__}: {item!r}"
            )
        slot = getattr(item, "slot", None)
        if slot is not None:
            raise ValueError(
                f"output candidate media[{index}] cannot declare input-only slot={slot!r}"
            )
        validate_media_metadata(
            item,
            expected,
            identifier=f"output candidate media[{index}]",
        )


def _require_enum(value: object, enum_type: type[Enum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(
            f"expected {field_name} to be {enum_type.__name__}, received "
            f"{type(value).__name__}: {value!r}"
        )


def _require_instance(value: object, expected_type: type[object], field_name: str) -> None:
    if type(value) is not expected_type:
        raise TypeError(
            f"expected {field_name} to be {expected_type.__name__}, received "
            f"{type(value).__name__}: {value!r}"
        )


def _require_tuple(value: object, item_type: type[object], field_name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(
            f"expected {field_name} to be tuple, received {type(value).__name__}: {value!r}"
        )
    for index, item in enumerate(value):
        _require_instance(item, item_type, f"{field_name}[{index}]")


def _require_string_tuple(value: object, field_name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(
            f"expected {field_name} to be tuple, received {type(value).__name__}: {value!r}"
        )
    for index, item in enumerate(value):
        if type(item) is not str:
            raise TypeError(
                f"expected {field_name}[{index}] to be str, received "
                f"{type(item).__name__}: {item!r}"
            )
        if not item.strip():
            raise ValueError(f"expected {field_name}[{index}] to be a non-empty string")


def _require_enum_tuple(
    value: object,
    enum_type: type[Enum],
    field_name: str,
) -> None:
    if type(value) is not tuple:
        raise TypeError(
            f"expected {field_name} to be tuple, received {type(value).__name__}: {value!r}"
        )
    for index, item in enumerate(value):
        _require_enum(item, enum_type, f"{field_name}[{index}]")


def _require_non_negative_int(value: object, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(
            f"expected {field_name} to be int, received {type(value).__name__}: {value!r}"
        )
    if value < 0:
        raise ValueError(f"expected {field_name} >= 0, received {value}")


__all__ = [
    "BatchCapability",
    "DecodedMediaLike",
    "GeometrySource",
    "InputMediaBinding",
    "InputMediaLike",
    "InputMediaOrder",
    "InputMediaRule",
    "InputMediaSpec",
    "MediaFormat",
    "MediaType",
    "ModelInputLike",
    "NegativePromptPolicy",
    "OutputMediaSequence",
    "OutputMediaLike",
    "PipelineIOContract",
    "RateRequirement",
    "resolve_pipeline_input_media_slots",
    "validate_pipeline_model_input",
    "validate_pipeline_output_candidate",
]
