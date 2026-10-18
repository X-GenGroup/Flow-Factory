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

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from ..utils.base import expand_per_sample


def _validate_component_mapping(
    values: Mapping[str, torch.Tensor], identifier: str
) -> Dict[str, torch.Tensor]:
    if not isinstance(values, Mapping):
        raise TypeError(
            f"expected Mapping[str, torch.Tensor] for {identifier}, "
            f"received {type(values).__name__}"
        )
    result: Dict[str, torch.Tensor] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"expected non-empty string component names for {identifier}, " f"received {name!r}"
            )
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"expected torch.Tensor for {identifier}[{name!r}], "
                f"received {type(value).__name__}"
            )
        result[name] = value
    if not result:
        raise ValueError(f"expected at least one component for {identifier}, received no keys")
    return result


def _require_component_order(
    received: Tuple[str, ...], expected: Tuple[str, ...], identifier: str
) -> None:
    if received != expected:
        raise ValueError(f"expected {identifier} component order {expected}, received {received}")


def _batch_size_of(state: "LatentState", identifier: str) -> int:
    first_name = state.component_names[0]
    first = state.components[first_name]
    if first.ndim < 1:
        raise ValueError(
            f"expected batched tensor for {identifier}[{first_name!r}], "
            f"received shape {tuple(first.shape)}"
        )
    return first.shape[0]


def _require_component_batch_size(
    values: Mapping[str, torch.Tensor], batch_size: int, identifier: str
) -> None:
    for name, value in values.items():
        if value.ndim < 1 or value.shape[0] != batch_size:
            raise ValueError(
                f"expected {identifier}[{name!r}] to use batch size {batch_size}, "
                f"received shape {tuple(value.shape)}"
            )


def _require_scalar_per_sample(
    values: Mapping[str, torch.Tensor], batch_size: int, identifier: str
) -> None:
    for name, value in values.items():
        if value.shape != (batch_size,):
            raise ValueError(
                f"expected {identifier}[{name!r}] shape {(batch_size,)}, "
                f"received {tuple(value.shape)}"
            )


def _require_scalar_like_per_sample(
    values: Mapping[str, torch.Tensor], batch_size: int, identifier: str
) -> None:
    """Require a broadcast statistic carrying exactly one value per sample."""
    _require_component_batch_size(values, batch_size, identifier)
    for name, value in values.items():
        if value.numel() != batch_size:
            raise ValueError(
                f"expected {identifier}[{name!r}] to hold exactly one value per sample, "
                f"received shape {tuple(value.shape)}"
            )


def _require_scalar_or_batched(
    values: Mapping[str, torch.Tensor], batch_size: int, identifier: str
) -> None:
    """Allow a 0-dim shared coordinate or a leading batch dimension.

    Legacy replay stores the terminal ``t_next`` as a 0-dim scalar shared by the
    whole batch, so scheduler coordinates cannot require a batch axis.
    """
    for name, value in values.items():
        if value.ndim == 0:
            continue
        if value.shape[0] != batch_size:
            raise ValueError(
                f"expected {identifier}[{name!r}] to be a 0-dim scalar or use batch size "
                f"{batch_size}, received shape {tuple(value.shape)}"
            )


def _move_tensor(tensor: torch.Tensor, device: Union[torch.device, str]) -> torch.Tensor:
    return tensor.to(device)


def _broadcasts_to(mask: torch.Tensor, target_shape: Tuple[int, ...]) -> bool:
    """Return whether a same-rank boolean mask broadcasts onto ``target_shape``.

    Equal rank is required so a per-sample mask can never be silently reinterpreted
    as a batched one (or the other way round) when the batch axis is missing.
    """
    if mask.ndim != len(target_shape):
        return False
    return all(
        mask_dim in (1, target_dim) for mask_dim, target_dim in zip(mask.shape, target_shape)
    )


def _validate_active_mask(
    mask: torch.Tensor, target_shape: Tuple[int, ...], identifier: str
) -> torch.Tensor:
    """Validate one static active mask against the state shape it must cover."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError(
            f"expected torch.Tensor or None for {identifier}, received {type(mask).__name__}"
        )
    if mask.dtype is not torch.bool:
        raise TypeError(f"expected {identifier} dtype torch.bool, received {mask.dtype}")
    if not _broadcasts_to(mask, target_shape):
        raise ValueError(
            f"expected {identifier} broadcastable to the latent state shape "
            f"{tuple(target_shape)}, received shape {tuple(mask.shape)}"
        )
    return mask


_SIGNED_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


@dataclass
class IndexedTrajectoryTensor:
    """Store one compact per-step tensor plus its sparse rollout-position map.

    Args:
        values: Stored entries shaped ``(stored, *shape)`` per sample or
            ``(B, stored, *shape)`` once ``batched`` is set.
        index_map: Map from rollout positions to stored indices, where ``-1``
            marks an uncollected rollout position.
        batched: Whether ``values`` carries a leading batch axis.
    """

    values: torch.Tensor
    index_map: torch.Tensor
    batched: bool = False

    def __post_init__(self) -> None:
        for name in ("values", "index_map"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"expected torch.Tensor for IndexedTrajectoryTensor.{name}, "
                    f"received {type(value).__name__}"
                )
        if not isinstance(self.batched, bool):
            raise TypeError(
                "expected bool for IndexedTrajectoryTensor.batched, "
                f"received {type(self.batched).__name__}: {self.batched!r}"
            )
        if self.index_map.ndim != 1:
            raise ValueError(
                "expected IndexedTrajectoryTensor.index_map shape (T,), "
                f"received {tuple(self.index_map.shape)}"
            )
        if self.index_map.dtype not in _SIGNED_INTEGER_DTYPES:
            raise TypeError(
                "expected signed integer IndexedTrajectoryTensor.index_map, "
                f"received dtype {self.index_map.dtype}"
            )
        stored_axis = 1 if self.batched else 0
        if self.values.ndim <= stored_axis:
            raise ValueError(
                f"expected IndexedTrajectoryTensor.values with stored axis {stored_axis}, "
                f"received shape {tuple(self.values.shape)} for batched={self.batched}"
            )
        num_stored = self.values.shape[stored_axis]
        if self.index_map.numel() and (
            self.index_map.min().item() < -1 or self.index_map.max().item() >= num_stored
        ):
            raise ValueError(
                f"expected IndexedTrajectoryTensor.index_map values for {num_stored} stored "
                f"entries in range [0, {num_stored - 1}] or uncollected sentinel -1, received "
                f"{self.index_map.tolist()}"
            )

    @property
    def num_stored(self) -> int:
        """Return the number of compact stored entries."""
        return int(self.values.shape[1 if self.batched else 0])

    def at(
        self, rollout_position: int, *, identifier: str = "IndexedTrajectoryTensor"
    ) -> torch.Tensor:
        """Read the stored entry recorded for one rollout position.

        Args:
            rollout_position: Rollout transition index to read.
            identifier: Caller-facing name used in error messages.

        Returns:
            Stored tensor slice for ``rollout_position``.
        """
        if not isinstance(rollout_position, int) or isinstance(rollout_position, bool):
            raise TypeError(
                f"expected int rollout_position for {identifier}, received "
                f"{type(rollout_position).__name__}: {rollout_position!r}"
            )
        if rollout_position < 0 or rollout_position >= self.index_map.shape[0]:
            raise ValueError(
                f"expected {identifier} rollout position in [0, {self.index_map.shape[0] - 1}], "
                f"received {rollout_position} with index map contents {self.index_map.tolist()}"
            )
        stored_position = int(self.index_map[rollout_position].item())
        if stored_position == -1:
            raise ValueError(
                f"{identifier} at rollout position {rollout_position} received uncollected "
                f"sentinel -1; index map contents {self.index_map.tolist()}"
            )
        return self.values[:, stored_position] if self.batched else self.values[stored_position]

    def to(self, device: Union[torch.device, str]) -> "IndexedTrajectoryTensor":
        """Move stored values and the index map to a device in place.

        Args:
            device: Target tensor device.

        Returns:
            This indexed tensor after moving its tensors.
        """
        return self.map_tensors(lambda tensor: _move_tensor(tensor, device))

    def map_tensors(
        self, transform: Callable[[torch.Tensor], torch.Tensor]
    ) -> "IndexedTrajectoryTensor":
        """Transform stored values and the index map in place.

        Args:
            transform: Tensor transformation callback.

        Returns:
            This indexed tensor after transforming its tensors.
        """
        self.values = transform(self.values)
        self.index_map = transform(self.index_map)
        return self

    @classmethod
    def stack(cls, tensors: List["IndexedTrajectoryTensor"]) -> "IndexedTrajectoryTensor":
        """Stack per-sample indexed tensors sharing one index map.

        Args:
            tensors: Non-empty unbatched indexed tensors with identical index maps
                and stored shapes.

        Returns:
            Batched indexed tensor.
        """
        if not tensors:
            raise ValueError("expected non-empty IndexedTrajectoryTensor list to stack, received 0")
        first = tensors[0]
        for sample_index, indexed in enumerate(tensors):
            if not isinstance(indexed, cls):
                raise TypeError(
                    f"expected IndexedTrajectoryTensor at sample index {sample_index}, "
                    f"received {type(indexed).__name__}"
                )
            if indexed.batched:
                raise ValueError(
                    "expected unbatched IndexedTrajectoryTensor entries to stack, received "
                    f"batched=True at sample index {sample_index}"
                )
            if sample_index == 0:
                continue
            if not torch.equal(indexed.index_map, first.index_map):
                raise ValueError(
                    f"expected shared IndexedTrajectoryTensor.index_map "
                    f"{first.index_map.tolist()}, received {indexed.index_map.tolist()} "
                    f"for sample index {sample_index}"
                )
            if indexed.values.shape != first.values.shape:
                raise ValueError(
                    f"expected IndexedTrajectoryTensor.values shape "
                    f"{tuple(first.values.shape)}, received {tuple(indexed.values.shape)} "
                    f"for sample index {sample_index}"
                )
        return cls(
            values=torch.stack([indexed.values for indexed in tensors]),
            index_map=first.index_map,
            batched=True,
        )


@dataclass
class ComponentTrajectory:
    """Store one component's compact latent states and full scheduler trajectory.

    Args:
        states: Per-sample states or batched states. The stored-state axis is zero
            for a per-sample trajectory and one after collation.
        timesteps: Full scheduler timesteps with shape ``(T + 1,)`` per sample or
            ``(B, T + 1)`` after collation.
        state_index_map: Shared map from rollout positions to stored-state indices.
        sigmas: Optional full sigma schedule matching ``timesteps``.
        active_mask: Optional static boolean mask marking the elements that carry
            stochastic degrees of freedom, broadcastable to one stored state.
    """

    states: torch.Tensor
    timesteps: torch.Tensor
    state_index_map: torch.Tensor
    sigmas: Optional[torch.Tensor] = None
    active_mask: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        for name in ("states", "timesteps", "state_index_map"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"expected torch.Tensor for ComponentTrajectory.{name}, "
                    f"received {type(value).__name__}"
                )
        if self.states.ndim < 1:
            raise ValueError(
                f"expected ComponentTrajectory.states with at least 1 dimension, "
                f"received shape {tuple(self.states.shape)}"
            )
        if self.timesteps.ndim not in (1, 2):
            raise ValueError(
                "expected ComponentTrajectory.timesteps shape (T + 1,) or (B, T + 1), "
                f"received {tuple(self.timesteps.shape)}"
            )
        if self.timesteps.shape[-1] < 2:
            raise ValueError(
                "expected ComponentTrajectory.timesteps with at least two rollout positions, "
                f"received shape {tuple(self.timesteps.shape)}"
            )
        if self.state_index_map.ndim != 1:
            raise ValueError(
                "expected ComponentTrajectory.state_index_map shape (T + 1,), "
                f"received {tuple(self.state_index_map.shape)}"
            )
        schedule_length = self.timesteps.shape[-1]
        if self.state_index_map.shape[0] != schedule_length:
            raise ValueError(
                "expected ComponentTrajectory.state_index_map length to match timesteps "
                f"length {schedule_length}, received {self.state_index_map.shape[0]}"
            )
        if self.sigmas is not None:
            if not isinstance(self.sigmas, torch.Tensor):
                raise TypeError(
                    "expected torch.Tensor or None for ComponentTrajectory.sigmas, "
                    f"received {type(self.sigmas).__name__}"
                )
            if self.sigmas.shape != self.timesteps.shape:
                raise ValueError(
                    f"expected timesteps and sigmas with the same shape/length "
                    f"{schedule_length}, received timesteps {tuple(self.timesteps.shape)} "
                    f"and sigmas {tuple(self.sigmas.shape)}"
                )
        if self.state_index_map.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                "expected signed integer ComponentTrajectory.state_index_map, "
                f"received dtype {self.state_index_map.dtype}"
            )
        state_axis = 1 if self.timesteps.ndim == 2 else 0
        if self.states.ndim <= state_axis:
            raise ValueError(
                f"expected states with stored-state axis {state_axis}, "
                f"received shape {tuple(self.states.shape)}"
            )
        num_stored_states = self.states.shape[state_axis]
        if self.state_index_map.numel() and (
            self.state_index_map.min().item() < -1
            or self.state_index_map.max().item() >= num_stored_states
        ):
            raise ValueError(
                "ComponentTrajectory.state_index_map expected values for "
                f"{num_stored_states} stored states in range [0, {num_stored_states - 1}] "
                "or uncollected sentinel -1, "
                f"received {self.state_index_map.tolist()} with maximum "
                f"{self.state_index_map.max().item()}"
            )
        if self.timesteps.ndim == 2 and self.states.shape[0] != self.timesteps.shape[0]:
            raise ValueError(
                "expected batched states and timesteps with equal batch size, received "
                f"states shape {tuple(self.states.shape)} and timesteps shape "
                f"{tuple(self.timesteps.shape)}"
            )
        if self.active_mask is not None:
            # One state keeps the batch axis once collated, so the mask covers
            # ``states[:, stored]`` for a batched trajectory and ``states[stored]`` otherwise.
            state_shape = (
                self.states.shape[:1] + self.states.shape[2:]
                if state_axis == 1
                else self.states.shape[1:]
            )
            _validate_active_mask(
                self.active_mask, tuple(state_shape), "ComponentTrajectory.active_mask"
            )

    def to(self, device: Union[torch.device, str]) -> "ComponentTrajectory":
        """Move all trajectory tensors to a device in place.

        Args:
            device: Target tensor device.

        Returns:
            This trajectory after moving its tensors.
        """
        self.map_tensors(lambda tensor: _move_tensor(tensor, device))
        return self

    def map_tensors(
        self, transform: Callable[[torch.Tensor], torch.Tensor]
    ) -> "ComponentTrajectory":
        """Transform every trajectory tensor in place.

        Args:
            transform: Tensor transformation callback.

        Returns:
            This trajectory after transforming its tensors.
        """
        self.states = transform(self.states)
        self.timesteps = transform(self.timesteps)
        self.state_index_map = transform(self.state_index_map)
        if self.sigmas is not None:
            self.sigmas = transform(self.sigmas)
        if self.active_mask is not None:
            self.active_mask = transform(self.active_mask)
        return self


@dataclass
class StructuredTrajectory:
    """Store ordered component trajectories and an optional joint log probability.

    Args:
        components: Component trajectories in authoritative iteration order.
        log_probs: Optional per-step joint scalar log probabilities.
        log_prob_index_map: Optional shared rollout-position map for ``log_probs``.
        component_log_probs: Optional per-component scalar log probabilities sharing
            ``log_prob_index_map``, keyed in authoritative component order.
        callbacks: Optional named per-component callback trajectories, e.g.
            ``{"velocity": {"latent": IndexedTrajectoryTensor(...)}}``.
    """

    components: Mapping[str, ComponentTrajectory]
    log_probs: Optional[torch.Tensor] = None
    log_prob_index_map: Optional[torch.Tensor] = None
    component_log_probs: Optional[Mapping[str, torch.Tensor]] = None
    callbacks: Optional[Mapping[str, Mapping[str, IndexedTrajectoryTensor]]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.components, Mapping):
            raise TypeError(
                "expected Mapping[str, ComponentTrajectory] for StructuredTrajectory.components, "
                f"received {type(self.components).__name__}"
            )
        copied: Dict[str, ComponentTrajectory] = {}
        for name, trajectory in self.components.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "expected non-empty string StructuredTrajectory component names, "
                    f"received {name!r}"
                )
            if not isinstance(trajectory, ComponentTrajectory):
                raise TypeError(
                    f"expected ComponentTrajectory for component {name!r}, "
                    f"received {type(trajectory).__name__}"
                )
            copied[name] = trajectory
        if not copied:
            raise ValueError("expected at least one StructuredTrajectory component, received none")
        self.components = copied
        if self.log_probs is not None and not isinstance(self.log_probs, torch.Tensor):
            raise TypeError(
                "expected torch.Tensor or None for StructuredTrajectory.log_probs, "
                f"received {type(self.log_probs).__name__}"
            )
        if self.log_probs is not None and self.log_probs.ndim not in (1, 2):
            raise ValueError(
                "expected StructuredTrajectory.log_probs shape (T,) or (B, T), "
                f"received {tuple(self.log_probs.shape)}"
            )
        if self.log_prob_index_map is not None:
            if not isinstance(self.log_prob_index_map, torch.Tensor):
                raise TypeError(
                    "expected torch.Tensor or None for StructuredTrajectory.log_prob_index_map, "
                    f"received {type(self.log_prob_index_map).__name__}"
                )
            if self.log_prob_index_map.ndim != 1:
                raise ValueError(
                    "expected StructuredTrajectory.log_prob_index_map shape (T,), "
                    f"received {tuple(self.log_prob_index_map.shape)}"
                )
            if self.log_prob_index_map.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise TypeError(
                    "expected signed integer StructuredTrajectory.log_prob_index_map, "
                    f"received dtype {self.log_prob_index_map.dtype}"
                )
            if self.log_probs is None:
                raise ValueError(
                    "expected log_probs when log_prob_index_map is provided, received log_probs=None"
                )
            log_prob_length = self.log_probs.shape[-1]
            if self.log_prob_index_map.numel() and (
                self.log_prob_index_map.min().item() < -1
                or self.log_prob_index_map.max().item() >= log_prob_length
            ):
                raise ValueError(
                    "expected StructuredTrajectory.log_prob_index_map values in range "
                    f"[0, {log_prob_length - 1}] or uncollected sentinel -1, received "
                    f"{self.log_prob_index_map.tolist()}"
                )
        self._validate_component_log_probs()
        self._validate_callbacks()

    def _validate_component_log_probs(self) -> None:
        if self.component_log_probs is None:
            return
        validated = _validate_component_mapping(
            self.component_log_probs, "StructuredTrajectory.component_log_probs"
        )
        expected_names = tuple(self.components)
        if tuple(validated) != expected_names:
            raise ValueError(
                "expected StructuredTrajectory.component_log_probs component order "
                f"{expected_names}, received {tuple(validated)}"
            )
        reference_name = expected_names[0]
        reference_shape = validated[reference_name].shape
        for name, values in validated.items():
            if values.ndim not in (1, 2):
                raise ValueError(
                    f"expected StructuredTrajectory.component_log_probs[{name!r}] shape "
                    f"(T,) or (B, T), received {tuple(values.shape)}"
                )
            if self.log_probs is not None and values.shape != self.log_probs.shape:
                raise ValueError(
                    f"expected StructuredTrajectory.component_log_probs[{name!r}] shape "
                    f"{tuple(self.log_probs.shape)} to match log_probs, received "
                    f"{tuple(values.shape)}"
                )
            # Components share one rollout schedule, so their stored lengths must agree
            # even when no joint log probability is stored to compare against.
            if values.shape != reference_shape:
                raise ValueError(
                    f"expected StructuredTrajectory.component_log_probs[{name!r}] shape "
                    f"{tuple(reference_shape)} to match component {reference_name!r}, "
                    f"received {tuple(values.shape)}"
                )
        if self.log_prob_index_map is not None:
            stored_length = reference_shape[-1]
            if self.log_prob_index_map.numel() and (
                self.log_prob_index_map.max().item() >= stored_length
            ):
                raise ValueError(
                    "expected StructuredTrajectory.component_log_probs stored length to cover "
                    f"log_prob_index_map values up to "
                    f"{self.log_prob_index_map.max().item()}, received length {stored_length}"
                )
        self.component_log_probs = validated

    def _validate_callbacks(self) -> None:
        if self.callbacks is None:
            return
        if not isinstance(self.callbacks, Mapping):
            raise TypeError(
                "expected Mapping[str, Mapping[str, IndexedTrajectoryTensor]] for "
                f"StructuredTrajectory.callbacks, received {type(self.callbacks).__name__}"
            )
        expected_names = tuple(self.components)
        validated: Dict[str, Dict[str, IndexedTrajectoryTensor]] = {}
        for field_name, component_values in self.callbacks.items():
            if not isinstance(field_name, str) or not field_name:
                raise ValueError(
                    "expected non-empty string StructuredTrajectory.callbacks field names, "
                    f"received {field_name!r}"
                )
            if not isinstance(component_values, Mapping):
                raise TypeError(
                    f"expected Mapping[str, IndexedTrajectoryTensor] for "
                    f"StructuredTrajectory.callbacks[{field_name!r}], received "
                    f"{type(component_values).__name__}"
                )
            for name, indexed in component_values.items():
                if not isinstance(indexed, IndexedTrajectoryTensor):
                    raise TypeError(
                        f"expected IndexedTrajectoryTensor for "
                        f"StructuredTrajectory.callbacks[{field_name!r}][{name!r}], received "
                        f"{type(indexed).__name__}"
                    )
            if tuple(component_values) != expected_names:
                raise ValueError(
                    f"expected StructuredTrajectory.callbacks[{field_name!r}] component order "
                    f"{expected_names}, received {tuple(component_values)}"
                )
            validated[field_name] = dict(component_values)
        if not validated:
            raise ValueError(
                "expected at least one StructuredTrajectory.callbacks field, received none"
            )
        self.callbacks = validated

    @property
    def callback_fields(self) -> Tuple[str, ...]:
        """Return stored callback field names in declaration order."""
        return () if self.callbacks is None else tuple(self.callbacks)

    @property
    def component_names(self) -> Tuple[str, ...]:
        """Return component names in authoritative trajectory order."""
        return tuple(self.components)

    def to(self, device: Union[torch.device, str]) -> "StructuredTrajectory":
        """Move all nested trajectory tensors to a device in place.

        Args:
            device: Target tensor device.

        Returns:
            This trajectory after moving its tensors.
        """
        self.map_tensors(lambda tensor: _move_tensor(tensor, device))
        return self

    def map_tensors(
        self, transform: Callable[[torch.Tensor], torch.Tensor]
    ) -> "StructuredTrajectory":
        """Transform every nested trajectory tensor in place.

        Args:
            transform: Tensor transformation callback.

        Returns:
            This trajectory after transforming its tensors.
        """
        for trajectory in self.components.values():
            trajectory.map_tensors(transform)
        if self.log_probs is not None:
            self.log_probs = transform(self.log_probs)
        if self.log_prob_index_map is not None:
            self.log_prob_index_map = transform(self.log_prob_index_map)
        if self.component_log_probs is not None:
            self.component_log_probs = {
                name: transform(values) for name, values in self.component_log_probs.items()
            }
        if self.callbacks is not None:
            for component_values in self.callbacks.values():
                for indexed in component_values.values():
                    indexed.map_tensors(transform)
        return self

    @classmethod
    def stack(cls, trajectories: List["StructuredTrajectory"]) -> "StructuredTrajectory":
        """Stack compatible per-sample trajectories into a batched trajectory.

        Args:
            trajectories: Non-empty trajectories with identical component order and
                shared index maps.

        Returns:
            Batched structured trajectory.
        """
        if not trajectories:
            raise ValueError("expected non-empty trajectories to stack, received 0")
        first = trajectories[0]
        expected_names = first.component_names
        for sample_index, trajectory in enumerate(trajectories[1:], start=1):
            if trajectory.component_names != expected_names:
                raise ValueError(
                    "expected identical trajectory component order "
                    f"{expected_names}, received {trajectory.component_names} "
                    f"for sample index {sample_index}"
                )
            for name in expected_names:
                expected_map = first.components[name].state_index_map
                received_map = trajectory.components[name].state_index_map
                if not torch.equal(received_map, expected_map):
                    raise ValueError(
                        f"component {name!r} state_index_map expected "
                        f"{expected_map.tolist()}, received {received_map.tolist()} "
                        f"for sample index {sample_index}"
                    )
            if (first.log_prob_index_map is None) != (trajectory.log_prob_index_map is None):
                raise ValueError(
                    "expected identical log_prob_index_map presence across samples, "
                    f"received mismatch at sample index {sample_index}"
                )
            if first.log_prob_index_map is not None and not torch.equal(
                trajectory.log_prob_index_map, first.log_prob_index_map
            ):
                raise ValueError(
                    "expected shared log_prob_index_map "
                    f"{first.log_prob_index_map.tolist()}, received "
                    f"{trajectory.log_prob_index_map.tolist()} for sample index {sample_index}"
                )
            if (first.component_log_probs is None) != (trajectory.component_log_probs is None):
                raise ValueError(
                    "expected identical component_log_probs presence across samples, "
                    f"received mismatch at sample index {sample_index}"
                )
            if trajectory.callback_fields != first.callback_fields:
                raise ValueError(
                    f"expected identical callback field names {first.callback_fields}, "
                    f"received {trajectory.callback_fields} for sample index {sample_index}"
                )

        components: Dict[str, ComponentTrajectory] = {}
        for name in expected_names:
            component_values = [trajectory.components[name] for trajectory in trajectories]
            expected_shapes = {
                "states": tuple(component_values[0].states.shape),
                "timesteps": tuple(component_values[0].timesteps.shape),
            }
            for sample_index, component in enumerate(component_values[1:], start=1):
                received_shapes = {
                    "states": tuple(component.states.shape),
                    "timesteps": tuple(component.timesteps.shape),
                }
                if received_shapes != expected_shapes:
                    raise ValueError(
                        f"expected component {name!r} shapes {expected_shapes}, received "
                        f"{received_shapes} for sample index {sample_index}"
                    )
            sigma_presence = [component.sigmas is not None for component in component_values]
            if len(set(sigma_presence)) != 1:
                raise ValueError(
                    f"expected identical sigma presence for component {name!r}, "
                    f"received {sigma_presence}"
                )
            if sigma_presence[0]:
                expected_sigma_shape = tuple(component_values[0].sigmas.shape)
                for sample_index, component in enumerate(component_values[1:], start=1):
                    if tuple(component.sigmas.shape) != expected_sigma_shape:
                        raise ValueError(
                            f"expected component {name!r} sigma shape "
                            f"{expected_sigma_shape}, received "
                            f"{tuple(component.sigmas.shape)} for sample index {sample_index}"
                        )
            mask_presence = [component.active_mask is not None for component in component_values]
            if len(set(mask_presence)) != 1:
                raise ValueError(
                    f"expected identical active_mask presence for component {name!r}, "
                    f"received {mask_presence}"
                )
            if mask_presence[0]:
                expected_mask_shape = tuple(component_values[0].active_mask.shape)
                for sample_index, component in enumerate(component_values[1:], start=1):
                    if tuple(component.active_mask.shape) != expected_mask_shape:
                        raise ValueError(
                            f"expected component {name!r} active_mask shape "
                            f"{expected_mask_shape}, received "
                            f"{tuple(component.active_mask.shape)} for sample index {sample_index}"
                        )
            components[name] = ComponentTrajectory(
                states=torch.stack([component.states for component in component_values]),
                timesteps=torch.stack([component.timesteps for component in component_values]),
                sigmas=(
                    torch.stack([component.sigmas for component in component_values])
                    if sigma_presence[0]
                    else None
                ),
                state_index_map=component_values[0].state_index_map,
                active_mask=(
                    torch.stack([component.active_mask for component in component_values])
                    if mask_presence[0]
                    else None
                ),
            )

        log_prob_presence = [trajectory.log_probs is not None for trajectory in trajectories]
        if len(set(log_prob_presence)) != 1:
            raise ValueError(
                "expected identical log_probs presence across samples, "
                f"received {log_prob_presence}"
            )
        if log_prob_presence[0]:
            expected_log_prob_shape = tuple(first.log_probs.shape)
            for sample_index, trajectory in enumerate(trajectories[1:], start=1):
                if tuple(trajectory.log_probs.shape) != expected_log_prob_shape:
                    raise ValueError(
                        f"expected log_probs shape {expected_log_prob_shape}, received "
                        f"{tuple(trajectory.log_probs.shape)} for sample index {sample_index}"
                    )
        component_log_probs: Optional[Dict[str, torch.Tensor]] = None
        if first.component_log_probs is not None:
            for name in expected_names:
                expected_shape = tuple(first.component_log_probs[name].shape)
                for sample_index, trajectory in enumerate(trajectories[1:], start=1):
                    received_shape = tuple(trajectory.component_log_probs[name].shape)
                    if received_shape != expected_shape:
                        raise ValueError(
                            f"expected component_log_probs[{name!r}] shape {expected_shape}, "
                            f"received {received_shape} for sample index {sample_index}"
                        )
            component_log_probs = {
                name: torch.stack(
                    [trajectory.component_log_probs[name] for trajectory in trajectories]
                )
                for name in expected_names
            }

        callbacks: Optional[Dict[str, Dict[str, IndexedTrajectoryTensor]]] = None
        if first.callback_fields:
            callbacks = {
                field_name: {
                    name: IndexedTrajectoryTensor.stack(
                        [trajectory.callbacks[field_name][name] for trajectory in trajectories]
                    )
                    for name in expected_names
                }
                for field_name in first.callback_fields
            }

        return cls(
            components=components,
            log_probs=(
                torch.stack([trajectory.log_probs for trajectory in trajectories])
                if log_prob_presence[0]
                else None
            ),
            log_prob_index_map=first.log_prob_index_map,
            component_log_probs=component_log_probs,
            callbacks=callbacks,
        )


def unstack_structured_trajectories(
    *,
    component_order: Sequence[str],
    states: Mapping[str, torch.Tensor],
    schedule: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
    state_index_maps: Mapping[str, torch.Tensor],
    active_masks: Optional[Mapping[str, torch.Tensor]] = None,
    log_probs: Optional[torch.Tensor] = None,
    log_prob_index_map: Optional[torch.Tensor] = None,
    component_log_probs: Optional[Mapping[str, torch.Tensor]] = None,
    callbacks: Optional[Mapping[str, Mapping[str, torch.Tensor]]] = None,
    callback_index_maps: Optional[Mapping[str, torch.Tensor]] = None,
) -> List["StructuredTrajectory"]:
    """Split batched component rollout results into one trajectory per sample.

    This is the inverse of ``StructuredTrajectory.stack``: every model family that
    collects componentized rollout tensors shares this assembly, so only the
    family-specific validation and any concatenated-tensor splitting stay local.

    Args:
        component_order: Authoritative component order for every mapping.
        states: Batched states by component, each shaped ``(B, stored, ...)``.
        schedule: Full timestep/sigma schedules by component.
        state_index_maps: Rollout-position map by component.
        active_masks: Optional batched stochastic-degree-of-freedom masks by component.
        log_probs: Optional joint log probabilities shaped ``(B, stored)``.
        log_prob_index_map: Rollout-transition map shared by all log probabilities.
        component_log_probs: Optional per-component log probabilities shaped like ``log_probs``.
        callbacks: Optional batched callback tensors keyed by field then component.
        callback_index_maps: Rollout-transition map by component for the callbacks.

    Returns:
        One structured trajectory per batch sample, in batch order.

    Raises:
        ValueError: If any mapping does not follow ``component_order`` or the
            component batch sizes disagree.
    """
    expected = tuple(component_order)
    named_mappings = {
        "states": states,
        "schedule": schedule,
        "state_index_maps": state_index_maps,
        "active_masks": active_masks,
        "component_log_probs": component_log_probs,
        "callback_index_maps": callback_index_maps,
    }
    for field_name, values in named_mappings.items():
        if values is None:
            continue
        if tuple(values) != expected:
            raise ValueError(
                f"expected {field_name} component order {expected}, received {tuple(values)}"
            )
    if callbacks is not None:
        for field_name, component_values in callbacks.items():
            if tuple(component_values) != expected:
                raise ValueError(
                    f"expected callbacks[{field_name!r}] component order {expected}, "
                    f"received {tuple(component_values)}"
                )

    batch_size = states[expected[0]].shape[0]
    for name in expected:
        if states[name].shape[0] != batch_size:
            raise ValueError(
                f"expected states[{name!r}] batch size {batch_size}, "
                f"received {states[name].shape[0]}"
            )

    resolved_log_prob_map = None if log_probs is None else log_prob_index_map
    trajectories: List[StructuredTrajectory] = []
    for sample_index in range(batch_size):
        trajectories.append(
            StructuredTrajectory(
                components={
                    name: ComponentTrajectory(
                        states=states[name][sample_index],
                        timesteps=schedule[name][0],
                        sigmas=schedule[name][1],
                        state_index_map=state_index_maps[name],
                        active_mask=(
                            None if active_masks is None else active_masks[name][sample_index]
                        ),
                    )
                    for name in expected
                },
                log_probs=None if log_probs is None else log_probs[sample_index],
                log_prob_index_map=resolved_log_prob_map,
                component_log_probs=(
                    None
                    if component_log_probs is None
                    else {name: component_log_probs[name][sample_index] for name in expected}
                ),
                callbacks=(
                    None
                    if not callbacks
                    else {
                        field_name: {
                            name: IndexedTrajectoryTensor(
                                values=component_values[name][sample_index],
                                index_map=callback_index_maps[name],
                            )
                            for name in expected
                        }
                        for field_name, component_values in callbacks.items()
                    }
                ),
            )
        )
    return trajectories


@dataclass
class LatentState:
    """Represent a batched latent state keyed by trajectory component.

    Args:
        components: Ordered component-to-latent mapping.
        active_masks: Optional static boolean masks marking each component's
            stochastic degrees of freedom. When present every component needs one.
    """

    components: Mapping[str, torch.Tensor]
    active_masks: Optional[Mapping[str, torch.Tensor]] = None

    def __post_init__(self) -> None:
        self.components = _validate_component_mapping(self.components, "LatentState.components")
        if self.active_masks is None:
            return
        validated = _validate_component_mapping(self.active_masks, "LatentState.active_masks")
        expected_names = tuple(self.components)
        if tuple(validated) != expected_names:
            raise ValueError(
                f"expected LatentState.active_masks component order {expected_names}, "
                f"received {tuple(validated)}"
            )
        batch_size = _batch_size_of(self, "LatentState.components")
        for name, mask in validated.items():
            identifier = f"LatentState.active_masks[{name!r}]"
            if mask.ndim < 1 or mask.shape[0] != batch_size:
                raise ValueError(
                    f"expected {identifier} to use batch size {batch_size}, "
                    f"received shape {tuple(mask.shape)}"
                )
            _validate_active_mask(mask, tuple(self.components[name].shape), identifier)
        self.active_masks = validated

    @property
    def component_names(self) -> Tuple[str, ...]:
        """Return component names in state order."""
        return tuple(self.components)

    @property
    def batch_size(self) -> int:
        """Return the leading batch dimension shared by every component."""
        return _batch_size_of(self, "LatentState.components")


@dataclass
class ComponentTimes:
    """Represent current and next scheduler coordinates for each component.

    Args:
        timestep: Current timestep by component.
        next_timestep: Next timestep by component.
        sigma: Optional current sigma by component.
        next_sigma: Optional next sigma by component.
    """

    timestep: Mapping[str, torch.Tensor]
    next_timestep: Mapping[str, torch.Tensor]
    sigma: Optional[Mapping[str, torch.Tensor]] = None
    next_sigma: Optional[Mapping[str, torch.Tensor]] = None

    def __post_init__(self) -> None:
        self.timestep = _validate_component_mapping(self.timestep, "ComponentTimes.timestep")
        self.next_timestep = _validate_component_mapping(
            self.next_timestep, "ComponentTimes.next_timestep"
        )
        expected_names = tuple(self.timestep)
        if tuple(self.next_timestep) != expected_names:
            raise ValueError(
                f"expected next_timestep component order {expected_names}, "
                f"received {tuple(self.next_timestep)}"
            )
        for field_name in ("sigma", "next_sigma"):
            values = getattr(self, field_name)
            if values is None:
                continue
            validated = _validate_component_mapping(values, f"ComponentTimes.{field_name}")
            if tuple(validated) != expected_names:
                raise ValueError(
                    f"expected {field_name} component order {expected_names}, "
                    f"received {tuple(validated)}"
                )
            setattr(self, field_name, validated)

    def per_sample(self, batch_size: int) -> "ComponentTimes":
        """Return these times with one value per sample for every component.

        Rollout replays carry one time per sample, while distillation replays share
        one time across the batch; adapters receive the per-sample form either way.

        Args:
            batch_size: Leading batch dimension of the state these times describe.

        Returns:
            Times whose tensors all have shape ``(batch_size,)``.

        Raises:
            ValueError: If a time is neither shared nor shaped ``(batch_size,)``.
        """

        def expand(
            values: Optional[Mapping[str, torch.Tensor]], field_name: str
        ) -> Optional[Dict[str, torch.Tensor]]:
            if values is None:
                return None
            return {
                name: expand_per_sample(value, batch_size, f"times.{field_name}[{name!r}]")
                for name, value in values.items()
            }

        return ComponentTimes(
            timestep=expand(self.timestep, "timestep"),
            next_timestep=expand(self.next_timestep, "next_timestep"),
            sigma=expand(self.sigma, "sigma"),
            next_sigma=expand(self.next_sigma, "next_sigma"),
        )


@dataclass
class ReplayStep:
    """Bundle one replay transition and its optional stored joint log probability.

    Args:
        state: Current component latent state.
        next_state: Stored next component latent state.
        times: Current and next scheduler coordinates.
        log_prob: Optional stored joint scalar log probability.
        component_log_probs: Optional stored per-component scalar log probabilities.
    """

    state: LatentState
    next_state: LatentState
    times: ComponentTimes
    log_prob: Optional[torch.Tensor] = None
    component_log_probs: Optional[Mapping[str, torch.Tensor]] = None

    def __post_init__(self) -> None:
        if self.component_log_probs is not None:
            self.component_log_probs = _validate_component_mapping(
                self.component_log_probs, "ReplayStep.component_log_probs"
            )
        expected_names = self.state.component_names
        batch_size = _batch_size_of(self.state, "ReplayStep.state")
        _require_component_batch_size(self.state.components, batch_size, "ReplayStep.state")
        _require_component_order(
            self.next_state.component_names, expected_names, "ReplayStep.next_state"
        )
        _require_component_batch_size(
            self.next_state.components, batch_size, "ReplayStep.next_state"
        )
        for field_name in ("timestep", "next_timestep", "sigma", "next_sigma"):
            times = getattr(self.times, field_name)
            if times is None:
                continue
            identifier = f"ReplayStep.times.{field_name}"
            _require_component_order(tuple(times), expected_names, identifier)
            _require_scalar_or_batched(times, batch_size, identifier)
        if self.log_prob is not None:
            if not isinstance(self.log_prob, torch.Tensor):
                raise TypeError(
                    "expected torch.Tensor or None for ReplayStep.log_prob, "
                    f"received {type(self.log_prob).__name__}"
                )
            if self.log_prob.shape != (batch_size,):
                raise ValueError(
                    f"expected ReplayStep.log_prob shape {(batch_size,)}, "
                    f"received {tuple(self.log_prob.shape)}"
                )
        if self.component_log_probs is not None:
            _require_component_order(
                tuple(self.component_log_probs), expected_names, "ReplayStep.component_log_probs"
            )
            _require_scalar_per_sample(
                self.component_log_probs, batch_size, "ReplayStep.component_log_probs"
            )


@dataclass
class NoisedState:
    """Bundle a forward-noised state, velocity target, and sampled noise.

    Args:
        state: Forward-noised component latent state.
        target_velocity: Flow-matching target velocity by component.
        noise: Sampled noise by component.
    """

    state: LatentState
    target_velocity: LatentState
    noise: LatentState


@dataclass
class MultiModalStepOutput:
    """Represent a scheduler step output over one or more latent components.

    Args:
        next_state: Optional sampled next latent state.
        next_state_mean: Optional transition mean latent state.
        std_dev_t: Optional per-component transition standard deviation.
        dt: Optional per-component scheduler step size.
        log_prob: Optional joint scalar transition log probability.
        component_log_probs: Optional per-component scalar transition log probabilities.
        velocity: Optional predicted velocity state.
    """

    next_state: Optional[LatentState] = None
    next_state_mean: Optional[LatentState] = None
    std_dev_t: Optional[Mapping[str, torch.Tensor]] = None
    dt: Optional[Mapping[str, torch.Tensor]] = None
    log_prob: Optional[torch.Tensor] = None
    component_log_probs: Optional[Mapping[str, torch.Tensor]] = None
    velocity: Optional[LatentState] = None

    def __post_init__(self) -> None:
        for field_name in ("std_dev_t", "dt", "component_log_probs"):
            values = getattr(self, field_name)
            if values is None:
                continue
            setattr(
                self,
                field_name,
                _validate_component_mapping(values, f"MultiModalStepOutput.{field_name}"),
            )
        reference = self._reference_contract()
        if reference is None:
            return
        expected_names, batch_size = reference
        for field_name in ("next_state", "next_state_mean", "velocity"):
            state = getattr(self, field_name)
            if state is None:
                continue
            _require_component_order(
                state.component_names, expected_names, f"MultiModalStepOutput.{field_name}"
            )
            _require_component_batch_size(
                state.components, batch_size, f"MultiModalStepOutput.{field_name}"
            )
        for field_name in ("std_dev_t", "dt"):
            values = getattr(self, field_name)
            if values is None:
                continue
            identifier = f"MultiModalStepOutput.{field_name}"
            _require_component_order(tuple(values), expected_names, identifier)
            _require_scalar_like_per_sample(values, batch_size, identifier)
        if self.component_log_probs is not None:
            _require_component_order(
                tuple(self.component_log_probs),
                expected_names,
                "MultiModalStepOutput.component_log_probs",
            )
            _require_scalar_per_sample(
                self.component_log_probs, batch_size, "MultiModalStepOutput.component_log_probs"
            )

    def _reference_contract(self) -> Optional[Tuple[Tuple[str, ...], int]]:
        """Return the component order and batch size the other fields must match.

        A statistics-only output carries no latent state, so the first available
        component mapping becomes authoritative instead.
        """
        for field_name in ("next_state", "next_state_mean", "velocity"):
            state = getattr(self, field_name)
            if state is not None:
                return (
                    state.component_names,
                    _batch_size_of(state, f"MultiModalStepOutput.{field_name}"),
                )
        for field_name in ("std_dev_t", "dt", "component_log_probs"):
            values = getattr(self, field_name)
            if values is None:
                continue
            names = tuple(values)
            first = values[names[0]]
            if first.ndim < 1:
                raise ValueError(
                    f"expected batched tensor for MultiModalStepOutput.{field_name}"
                    f"[{names[0]!r}], received shape {tuple(first.shape)}"
                )
            return names, first.shape[0]
        return None
