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

"""Trajectory topology, coordinates, and boundary construction for TDM trainers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ...hparams.training_args.tdm import TDMQuerySamplingPolicy
from ...samples import (
    BaseSample,
    ComponentTimes,
    StackedSampleBatch,
    StructuredTrajectory,
)
from ...utils.noise_schedule import TIMESTEP_MAX, flow_match_sigma, validate_flow_match_coordinates
from .tdm_time_sampling import RationalFlowShift, sample_query_timestep

_SCORE_QUERY_SAMPLE_ATTEMPTS = 8


class _TDMCoordinateContainmentError(ValueError):
    """Signal a representational endpoint collision that may be resampled."""


@dataclass(frozen=True)
class TDMQueryBounds:
    """Store explicit component-wise lower and upper query coordinates."""

    lower: ComponentTimes
    upper: ComponentTimes


@dataclass(frozen=True)
class TDMQueryBatchContext:
    """Share immutable query policy and coordinate transforms across K boundaries."""

    policy: TDMQuerySamplingPolicy
    primary_name: str
    source_transform: RationalFlowShift | None
    reverse_upper_times: ComponentTimes | None


@dataclass(frozen=True)
class TDMBoundaryUnit:
    """Store one replay batch and one boundary's explicit score-query bounds."""

    samples: tuple[BaseSample, ...]
    boundary_index: int
    times: ComponentTimes
    mid_times: ComponentTimes
    query_context: TDMQueryBatchContext
    query_bounds: TDMQueryBounds

    @property
    def primary_name(self) -> str:
        """Return the primary component shared by this replay microbatch."""
        return self.query_context.primary_name

    @property
    def interval_start(self) -> torch.Tensor:
        """Return the authoritative primary lower-boundary timestep.

        Returns:
            Stored primary ``next_timestep`` for this transition.
        """
        return self.times.next_timestep[self.primary_name]

    @property
    def interval_end(self) -> torch.Tensor:
        """Return the authoritative primary upper-boundary timestep.

        Returns:
            Stored primary current ``timestep`` for this transition.
        """
        return self.times.timestep[self.primary_name]

    @property
    def query_interval_end(self) -> torch.Tensor:
        """Return the upper timestep of the selected loss sampling interval.

        Returns:
            Mapped maximum for reverse, or the stored upper boundary for trajectory.
        """
        return self.query_bounds.upper.timestep[self.primary_name]


class TDMTrajectoryRuntimeMixin:
    """Validate and construct dense deterministic TDM trajectory boundaries."""

    def _validate_trajectory_configuration(self) -> None:
        """Require one deterministic rollout with exactly K generated transitions."""
        if self.training_args.num_inner_epochs != 1:
            raise ValueError(
                "TDM requires train.num_inner_epochs=1 so every generator update owns a "
                "fresh rollout; "
                f"received train.num_inner_epochs={self.training_args.num_inner_epochs}"
            )
        non_ode = {
            name: scheduler.dynamics_type
            for name, scheduler in self.adapter.scheduler_group.items()
            if scheduler.dynamics_type != "ODE"
        }
        if non_ode:
            raise ValueError(
                "TDM requires deterministic ODE dynamics for every scheduler component; "
                f"received {non_ode!r}"
            )

    def _build_boundary_units(
        self,
        samples: Sequence[BaseSample],
    ) -> list[TDMBoundaryUnit]:
        """Build all K ordered replay boundaries for one dataloader batch."""
        self._validate_trajectory_configuration()
        if not samples:
            return []
        batch_size = self.training_args.per_device_batch_size
        if len(samples) != batch_size:
            raise ValueError(
                "TDM boundary construction requires exactly one replay batch; "
                f"received samples={len(samples)} and per_device_batch_size={batch_size}"
            )

        replay_samples = tuple(samples)
        batch = self._stack_replay_unit(replay_samples)
        self._validate_sample_boundaries(replay_samples, batch)
        boundaries: list[tuple[int, ComponentTimes, ComponentTimes]] = []
        previous_times: ComponentTimes | None = None
        for boundary_index in range(1, self.training_args.num_inference_steps + 1):
            replay_step = self.adapter.get_replay_step(
                batch,
                boundary_index - 1,
            )
            times = TDMTrajectoryRuntimeMixin._normalize_replay_times(
                self,
                replay_step.times,
                len(replay_samples),
            )
            mid_times = self._validate_interval(
                batch,
                times,
                boundary_index=boundary_index,
                previous_times=previous_times,
            )
            stored_times = TDMTrajectoryRuntimeMixin._clone_component_times(times)
            boundaries.append((boundary_index, stored_times, mid_times))
            previous_times = stored_times

        context = self._build_query_batch_context(
            replay_samples,
            batch,
            template_times=boundaries[0][1],
        )
        units: list[TDMBoundaryUnit] = []
        for boundary_index, stored_times, mid_times in boundaries:
            query_bounds = self._build_query_bounds(
                context,
                stored_times,
                mid_times,
                boundary_index=boundary_index,
            )
            units.append(
                TDMBoundaryUnit(
                    samples=replay_samples,
                    boundary_index=boundary_index,
                    times=stored_times,
                    mid_times=mid_times,
                    query_context=context,
                    query_bounds=query_bounds,
                )
            )
        return units

    def _build_query_batch_context(
        self,
        samples: tuple[BaseSample, ...],
        batch: StackedSampleBatch,
        template_times: ComponentTimes,
    ) -> TDMQueryBatchContext:
        """Build one query policy context shared by every boundary in a microbatch."""
        policy = TDMQuerySamplingPolicy.from_training_args(self.training_args)
        primary = self.adapter.trajectory_component_order[0]
        source_transform = None
        if policy.requires_generation_shift:
            shifts = torch.tensor(
                [self._tdm_generation_provenance_for(sample).source_shift for sample in samples],
                device=template_times.timestep[primary].device,
                dtype=torch.float64,
            )
            source_transform = RationalFlowShift(shifts)
        reverse_upper_times = None
        if policy.interval == "reverse":
            upper = torch.full_like(
                template_times.timestep[primary],
                policy.max_sigma * TIMESTEP_MAX,
            )
            mapped = self.adapter.build_training_component_times(upper, batch=batch)
            order = self.adapter.trajectory_component_order
            if (
                mapped.sigma is None
                or tuple(mapped.timestep) != order
                or tuple(mapped.sigma) != order
            ):
                raise ValueError(
                    "TDM reverse upper-bound mapping requires ordered component times and sigmas"
                )
            for name in order:
                validate_flow_match_coordinates(
                    mapped.timestep[name],
                    mapped.sigma[name],
                    identifier=f"TDM reverse upper bound component={name!r}",
                )
            reverse_upper_times = self._clone_component_times(mapped)
        return TDMQueryBatchContext(
            policy=policy,
            primary_name=primary,
            source_transform=source_transform,
            reverse_upper_times=reverse_upper_times,
        )

    def _build_query_bounds(
        self,
        context: TDMQueryBatchContext,
        stored_times: ComponentTimes,
        mid_times: ComponentTimes,
        *,
        boundary_index: int,
    ) -> TDMQueryBounds:
        """Resolve and validate explicit component bounds for one trajectory boundary."""
        upper_times = (
            stored_times if context.policy.interval == "trajectory" else context.reverse_upper_times
        )
        if upper_times is None:
            raise RuntimeError("TDM reverse query context has no mapped upper coordinates")
        primary = context.primary_name
        lower, upper = self._normalize_coordinates(
            mid_times.timestep[primary], upper_times.timestep[primary]
        )
        if not bool((lower < upper).all()):
            raise ValueError(
                "TDM query interval requires its upper sigma above every lower boundary; "
                f"tdm_query_interval={context.policy.interval!r}, "
                f"tdm_query_max_sigma={context.policy.max_sigma}, "
                f"boundary_index={boundary_index}, "
                f"lower_sigma={flow_match_sigma(lower).tolist()}"
            )
        for name in self.adapter.trajectory_component_order:
            for field in ("timestep", "sigma"):
                upper_mapping = getattr(upper_times, field)
                lower_mapping = getattr(mid_times, field)
                if upper_mapping is None or lower_mapping is None:
                    continue
                self._validate_descending_coordinate_pair(
                    upper_mapping[name],
                    lower_mapping[name],
                    field=f"query {field}",
                    component=name,
                    boundary_index=boundary_index,
                )
        return TDMQueryBounds(lower=mid_times, upper=upper_times)

    def _normalize_replay_times(
        self,
        times: ComponentTimes,
        batch_size: int,
    ) -> ComponentTimes:
        """Return one real coordinate per sample for every component of a transition."""

        def widen(
            mapping: Mapping[str, torch.Tensor] | None,
        ) -> dict[str, torch.Tensor] | None:
            if mapping is None:
                return None
            return {
                name: TDMTrajectoryRuntimeMixin._as_per_sample_coordinate(
                    value,
                    batch_size,
                )
                for name, value in mapping.items()
            }

        return ComponentTimes(
            timestep=widen(times.timestep),
            next_timestep=widen(times.next_timestep),
            sigma=widen(times.sigma),
            next_sigma=widen(times.next_sigma),
        )

    @staticmethod
    def _as_per_sample_coordinate(
        coordinate: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Return a scheduler coordinate as one real value per sample."""
        if coordinate.ndim == 0:
            coordinate = coordinate.expand(batch_size)
        elif coordinate.ndim != 1 or coordinate.shape[0] != batch_size:
            raise ValueError(
                f"expected a scalar or ({batch_size},) scheduler coordinate, "
                f"received shape {tuple(coordinate.shape)}"
            )
        return coordinate

    def _validate_sample_boundaries(
        self,
        samples: tuple[BaseSample, ...],
        batch: StackedSampleBatch,
    ) -> None:
        """Require every one of the K + 1 rollout positions to be readable."""
        for boundary_index in range(self.training_args.num_inference_steps):
            try:
                self.adapter.get_replay_step(batch, boundary_index)
            except (KeyError, IndexError, ValueError, TypeError) as error:
                raise ValueError(
                    f"TDM requires all {self.training_args.num_inference_steps} rollout "
                    f"transitions to be stored; reading transition {boundary_index} failed. "
                    "The rollout must collect every boundary, so "
                    "`trajectory_indices` has to span the full schedule."
                ) from error

        TDMTrajectoryRuntimeMixin._validate_structured_boundaries(self, samples)

    def _validate_structured_boundaries(self, samples: tuple[BaseSample, ...]) -> None:
        """Check the stronger invariants an adapter that publishes structure can offer."""
        expected_length = self.training_args.num_inference_steps + 1
        component_order = self.adapter.trajectory_component_order
        expected_map = torch.arange(expected_length, dtype=torch.int64)
        for sample_index, sample in enumerate(samples):
            trajectory = sample.trajectory
            if not isinstance(trajectory, StructuredTrajectory):
                continue
            if trajectory.component_names != component_order:
                raise ValueError(
                    f"TDM expected trajectory component order {component_order}, received "
                    f"{trajectory.component_names} for sample_index={sample_index}"
                )
            primary_name = component_order[0]
            primary_map = trajectory.components[primary_name].state_index_map
            for name in component_order:
                component = trajectory.components[name]
                if component.timesteps.shape[-1] != expected_length:
                    raise ValueError(
                        f"TDM expected K + 1={expected_length} scheduler coordinates for "
                        f"component {name!r}, received {component.timesteps.shape[-1]} "
                        f"for sample_index={sample_index}"
                    )
                if component.states.shape[0] != expected_length:
                    raise ValueError(
                        f"TDM expected stored states length K + 1={expected_length} for "
                        f"component={name!r}, received {component.states.shape[0]} "
                        f"for sample_index={sample_index}"
                    )
                if component.state_index_map.shape[0] != expected_length:
                    raise ValueError(
                        f"TDM expected state_index_map length {expected_length} for component "
                        f"{name!r}, received {component.state_index_map.shape[0]} "
                        f"for sample_index={sample_index}"
                    )
                missing = torch.nonzero(component.state_index_map == -1).flatten().tolist()
                if missing:
                    raise ValueError(
                        f"TDM missing boundary states for component {name!r} at rollout "
                        f"positions={missing}, sample_index={sample_index}; all K + 1 "
                        "boundaries must be collected"
                    )
                if torch.unique(component.state_index_map).numel() != expected_length:
                    raise ValueError(
                        "TDM requires a strictly one-to-one state_index_map for "
                        f"component={name!r}, received {component.state_index_map.tolist()} "
                        f"for sample_index={sample_index}"
                    )
                if name != primary_name and not torch.equal(
                    component.state_index_map.to(
                        device=primary_map.device,
                        dtype=primary_map.dtype,
                    ),
                    primary_map,
                ):
                    raise ValueError(
                        f"TDM requires component={name!r} state_index_map to match "
                        f"component={primary_name!r}; received "
                        f"{component.state_index_map.tolist()} versus {primary_map.tolist()} "
                        f"for sample_index={sample_index}"
                    )
                if not torch.equal(
                    component.state_index_map.to(
                        device=expected_map.device,
                        dtype=expected_map.dtype,
                    ),
                    expected_map,
                ):
                    raise ValueError(
                        "TDM requires dense aligned state_index_map=arange(K + 1); "
                        f"component={name!r}, expected={expected_map.tolist()}, "
                        f"received={component.state_index_map.tolist()}, "
                        f"sample_index={sample_index}"
                    )

    def _validate_interval(
        self,
        batch: StackedSampleBatch,
        stored_times: ComponentTimes,
        *,
        boundary_index: int,
        previous_times: ComponentTimes | None,
    ) -> ComponentTimes:
        """Validate exact stored topology and return authoritative lower-boundary times."""
        expected_order = self.adapter.trajectory_component_order
        for field_name in ("timestep", "next_timestep", "sigma", "next_sigma"):
            coordinates = getattr(stored_times, field_name)
            if coordinates is None:
                continue
            if tuple(coordinates) != expected_order:
                raise ValueError(
                    f"TDM stored {field_name} expected component order {expected_order}, "
                    f"received {tuple(coordinates)} at boundary_index={boundary_index}"
                )
            for name in expected_order:
                self._validate_coordinate(
                    coordinates[name],
                    field=f"stored {field_name}",
                    component=name,
                    boundary_index=boundary_index,
                )
        if (stored_times.sigma is None) != (stored_times.next_sigma is None):
            raise ValueError(
                "TDM stored sigma and next_sigma must be provided together; "
                f"received sigma={stored_times.sigma is not None} and "
                f"next_sigma={stored_times.next_sigma is not None} at "
                f"boundary_index={boundary_index}"
            )

        for name in expected_order:
            current_timestep = stored_times.timestep[name]
            next_timestep = stored_times.next_timestep[name]
            self._validate_descending_coordinate_pair(
                current_timestep,
                next_timestep,
                field="timestep",
                component=name,
                boundary_index=boundary_index,
            )
            if stored_times.sigma is not None:
                validate_flow_match_coordinates(
                    current_timestep,
                    stored_times.sigma[name],
                    identifier=(
                        f"TDM stored timestep/sigma for component {name!r} at "
                        f"boundary_index={boundary_index}"
                    ),
                )
                validate_flow_match_coordinates(
                    next_timestep,
                    stored_times.next_sigma[name],
                    identifier=(
                        f"TDM stored next_timestep/next_sigma for component {name!r} at "
                        f"boundary_index={boundary_index}"
                    ),
                )
                self._validate_descending_coordinate_pair(
                    stored_times.sigma[name],
                    stored_times.next_sigma[name],
                    field="sigma",
                    component=name,
                    boundary_index=boundary_index,
                )
            if previous_times is not None:
                self._validate_adjacent_coordinate(
                    current_timestep,
                    previous_times.next_timestep[name],
                    field="timestep",
                    component=name,
                    boundary_index=boundary_index,
                )
                if stored_times.sigma is not None:
                    if previous_times.next_sigma is None:
                        raise ValueError(
                            "TDM adjacent transitions must preserve stored sigma topology; "
                            f"component={name!r}, boundary_index={boundary_index}"
                        )
                    self._validate_adjacent_coordinate(
                        stored_times.sigma[name],
                        previous_times.next_sigma[name],
                        field="sigma",
                        component=name,
                        boundary_index=boundary_index,
                    )
            if boundary_index == self.training_args.num_inference_steps:
                terminal_coordinates = [("next_timestep", next_timestep)]
                if stored_times.next_sigma is not None:
                    terminal_coordinates.append(("next_sigma", stored_times.next_sigma[name]))
                for field, coordinate in terminal_coordinates:
                    if not self._coordinates_equal(coordinate, torch.zeros_like(coordinate)):
                        raise ValueError(
                            "TDM terminal interval must end at exact coordinate zero; "
                            f"component={name!r}, field={field!r}, "
                            f"received={coordinate.tolist()}"
                        )
        return self._authoritative_mid_times(
            batch,
            stored_times,
            boundary_index=boundary_index,
        )

    def _authoritative_mid_times(
        self,
        batch: StackedSampleBatch,
        stored_times: ComponentTimes,
        *,
        boundary_index: int,
    ) -> ComponentTimes:
        """Use stored lower-boundary coordinates, falling back only when absent."""
        current_sigma = stored_times.next_sigma
        if current_sigma is None:
            primary_name = self.adapter.trajectory_component_order[0]
            mapped = self.adapter.build_training_component_times(
                stored_times.next_timestep[primary_name],
                batch=batch,
            )
            current_sigma = mapped.sigma
        if current_sigma is None:
            raise ValueError(
                "TDM lower-boundary mapping expected component sigmas, received None at "
                f"boundary_index={boundary_index}"
            )
        expected_order = self.adapter.trajectory_component_order
        if tuple(current_sigma) != expected_order:
            raise ValueError(
                f"TDM lower-boundary sigma expected component order {expected_order}, "
                f"received {tuple(current_sigma)} at boundary_index={boundary_index}"
            )
        for name in expected_order:
            self._validate_coordinate(
                current_sigma[name],
                field="lower-boundary sigma",
                component=name,
                boundary_index=boundary_index,
            )

        timestep = self._clone_mapping(stored_times.next_timestep)
        sigma = self._clone_mapping(current_sigma)
        return ComponentTimes(
            timestep=timestep,
            next_timestep={name: torch.zeros_like(value) for name, value in timestep.items()},
            sigma=sigma,
            next_sigma={name: torch.zeros_like(value) for name, value in sigma.items()},
        )

    @classmethod
    def _clone_component_times(cls, times: ComponentTimes) -> ComponentTimes:
        return ComponentTimes(
            timestep=cls._clone_mapping(times.timestep),
            next_timestep=cls._clone_mapping(times.next_timestep),
            sigma=None if times.sigma is None else cls._clone_mapping(times.sigma),
            next_sigma=(None if times.next_sigma is None else cls._clone_mapping(times.next_sigma)),
        )

    @staticmethod
    def _clone_mapping(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in values.items()}

    @classmethod
    def _validate_descending_coordinate_pair(
        cls,
        current: torch.Tensor,
        following: torch.Tensor,
        *,
        field: str,
        component: str,
        boundary_index: int,
    ) -> None:
        if current.shape != following.shape or current.ndim != 1:
            raise ValueError(
                f"TDM component={component!r}, boundary_index={boundary_index} expected "
                f"{field} interval tensors shaped (B,), received "
                f"current={tuple(current.shape)} and next={tuple(following.shape)}"
            )
        normalized_current, normalized_following = cls._normalize_coordinates(
            current,
            following,
        )
        if not bool((normalized_following < normalized_current).all().item()):
            raise ValueError(
                f"TDM component={component!r}, boundary_index={boundary_index} received "
                f"reversed or empty {field} interval "
                f"[{following.tolist()}, {current.tolist()})"
            )

    @classmethod
    def _validate_adjacent_coordinate(
        cls,
        current: torch.Tensor,
        previous_next: torch.Tensor,
        *,
        field: str,
        component: str,
        boundary_index: int,
    ) -> None:
        if cls._coordinates_equal(current, previous_next):
            return
        normalized_current, normalized_previous = cls._normalize_coordinates(
            current,
            previous_next,
        )
        topology = (
            "overlap" if bool((normalized_current > normalized_previous).any().item()) else "gap"
        )
        raise ValueError(
            f"TDM component={component!r}, boundary_index={boundary_index} {field} "
            f"interval has an exact {topology}; previous_next={previous_next.tolist()}, "
            f"current={current.tolist()}"
        )

    def _sample_perturbation_times(self, unit: TDMBoundaryUnit) -> torch.Tensor:
        """Sample strictly inside the selected primary loss interval."""
        self._validate_coordinate(
            unit.interval_start,
            field="primary interval_start timestep",
            boundary_index=unit.boundary_index,
        )
        self._validate_coordinate(
            unit.query_interval_end,
            field="primary query interval_end timestep",
            boundary_index=unit.boundary_index,
        )
        interval_start, interval_end = self._normalize_coordinates(
            unit.interval_start,
            unit.query_interval_end,
        )
        open_start = torch.nextafter(interval_start, interval_end)
        open_end = torch.nextafter(interval_end, interval_start)
        if not bool((open_start <= open_end).all().item()):
            raise ValueError(
                f"TDM boundary_index={unit.boundary_index} interval has no representable "
                f"floating interior between start={unit.interval_start.tolist()} and "
                f"end={unit.query_interval_end.tolist()}"
            )
        sampled = sample_query_timestep(
            interval_start,
            interval_end,
            policy=unit.query_context.policy,
            source_transform=unit.query_context.source_transform,
        )
        return torch.minimum(torch.maximum(sampled, open_start), open_end)

    def _sample_score_query_times(
        self,
        unit: TDMBoundaryUnit,
        batch: StackedSampleBatch,
    ) -> ComponentTimes:
        """Map a random primary time whose every component has an open interior value."""
        last_error: _TDMCoordinateContainmentError | None = None
        for _ in range(_SCORE_QUERY_SAMPLE_ATTEMPTS):
            primary_times = self._sample_perturbation_times(unit)
            times = self.adapter.build_training_component_times(primary_times, batch=batch)
            try:
                self._validate_score_query_coordinates(
                    times,
                    primary_times,
                    unit=unit,
                )
            except _TDMCoordinateContainmentError as error:
                # A mathematically interior primary value can round onto a secondary
                # endpoint after a nonlinear component-time transform. Redraw the
                # primary value instead of mutating the adapter's mapped coordinates.
                last_error = error
                continue
            return times
        raise ValueError(
            "TDM could not sample a jointly representable open component-time interval "
            f"after {_SCORE_QUERY_SAMPLE_ATTEMPTS} attempts at "
            f"boundary_index={unit.boundary_index}"
        ) from last_error

    def _validate_score_query_coordinates(
        self,
        times: ComponentTimes,
        primary_times: torch.Tensor,
        *,
        unit: TDMBoundaryUnit,
    ) -> None:
        """Require every component coordinate inside the selected loss interval."""
        boundary_index = unit.boundary_index
        component_order = self.adapter.trajectory_component_order
        lower_times = unit.query_bounds.lower
        upper_times = unit.query_bounds.upper
        if times.sigma is None:
            raise ValueError(
                "TDM score-query sigma validation expected component sigmas; "
                f"boundary_index={boundary_index}, component={component_order[0]!r}, "
                f"tau={primary_times.tolist()}, received sigma=None"
            )
        if tuple(times.timestep) != component_order or tuple(times.sigma) != component_order:
            raise ValueError(
                "TDM score-query coordinate validation expected component order "
                f"{component_order}; boundary_index={boundary_index}, "
                f"tau={primary_times.tolist()}, received timestep components="
                f"{tuple(times.timestep)} and sigma components={tuple(times.sigma)}"
            )
        for name in component_order:
            timestep = times.timestep[name]
            sigma = times.sigma[name]
            validate_flow_match_coordinates(
                timestep,
                sigma,
                identifier=(
                    f"TDM mapped continuous timestep/sigma for component {name!r} at "
                    f"boundary_index={boundary_index}"
                ),
            )
            self._validate_contained_coordinate(
                timestep,
                lower=lower_times.timestep[name],
                upper=upper_times.timestep[name],
                field="timestep",
                component=name,
                boundary_index=boundary_index,
            )
            if upper_times.sigma is not None:
                self._validate_contained_coordinate(
                    sigma,
                    lower=lower_times.sigma[name],
                    upper=upper_times.sigma[name],
                    field="sigma",
                    component=name,
                    boundary_index=boundary_index,
                )
            else:
                lower_sigma = lower_times.sigma[name]
                normalized_sigma, normalized_lower = self._normalize_coordinates(
                    sigma,
                    lower_sigma,
                )
                if not bool(
                    ((normalized_sigma > normalized_lower) & (normalized_sigma < 1)).all().item()
                ):
                    raise _TDMCoordinateContainmentError(
                        "TDM mapped continuous sigma must be above its lower boundary and "
                        "strictly below one; "
                        f"component={name!r}, boundary_index={boundary_index}, "
                        f"lower={lower_sigma.tolist()}, mapped={sigma.tolist()}"
                    )

    @classmethod
    def _validate_contained_coordinate(
        cls,
        coordinate: torch.Tensor,
        *,
        lower: torch.Tensor,
        upper: torch.Tensor,
        field: str,
        component: str,
        boundary_index: int,
    ) -> None:
        if coordinate.shape != lower.shape or coordinate.shape != upper.shape:
            raise ValueError(
                f"TDM mapped {field} expected stored interval shape {tuple(lower.shape)} for "
                f"component={component!r}, received mapped={tuple(coordinate.shape)} and "
                f"upper={tuple(upper.shape)} at boundary_index={boundary_index}"
            )
        normalized_coordinate, normalized_lower = cls._normalize_coordinates(
            coordinate,
            lower,
        )
        normalized_coordinate, normalized_upper = cls._normalize_coordinates(
            normalized_coordinate,
            upper,
        )
        if not bool(
            (
                (normalized_coordinate > normalized_lower)
                & (normalized_coordinate < normalized_upper)
            )
            .all()
            .item()
        ):
            raise _TDMCoordinateContainmentError(
                f"TDM mapped continuous {field} must lie strictly inside the selected "
                f"component interval; component={component!r}, "
                f"boundary_index={boundary_index}, lower={lower.tolist()}, "
                f"mapped={coordinate.tolist()}, upper={upper.tolist()}"
            )

    @staticmethod
    def _validate_coordinate(
        coordinate: torch.Tensor,
        *,
        field: str,
        boundary_index: int,
        component: str | None = None,
    ) -> None:
        """Require one finite real scheduler coordinate tensor."""
        component_context = "" if component is None else f", component={component!r}"
        if not isinstance(coordinate, torch.Tensor):
            raise TypeError(
                f"TDM {field}{component_context}, boundary_index={boundary_index} expected "
                f"torch.Tensor coordinates, received {type(coordinate).__name__}: "
                f"{coordinate!r}"
            )
        if coordinate.dtype == torch.bool or coordinate.is_complex():
            raise TypeError(
                f"TDM {field}{component_context}, boundary_index={boundary_index} expected "
                f"real numeric coordinates, received dtype={coordinate.dtype}"
            )
        if not bool(torch.isfinite(coordinate).all().item()):
            raise ValueError(
                f"TDM {field}{component_context}, boundary_index={boundary_index} expected "
                f"finite coordinates, received {coordinate}"
            )

    @staticmethod
    def _normalize_coordinates(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize two validated coordinates to one real promoted dtype and device."""
        common_dtype = torch.promote_types(left.dtype, right.dtype)
        if not common_dtype.is_floating_point:
            common_dtype = torch.get_default_dtype()
        return (
            left.to(device=left.device, dtype=common_dtype),
            right.to(device=left.device, dtype=common_dtype),
        )

    @classmethod
    def _coordinates_equal(
        cls,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> bool:
        """Compare scheduler coordinates exactly after safe normalization."""
        if left.shape != right.shape:
            return False
        normalized_left, normalized_right = cls._normalize_coordinates(left, right)
        return torch.equal(normalized_left, normalized_right)
