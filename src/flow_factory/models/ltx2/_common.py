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

# src/flow_factory/models/ltx2/_common.py
"""Shared helpers for the LTX2 (T2AV / I2AV) adapters."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from diffusers.utils.torch_utils import randn_tensor

from ...samples import (
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    NoisedState,
    StackedSampleBatch,
    StructuredTrajectory,
    unstack_structured_trajectories,
)
from ...scheduler import SDESchedulerOutput
from ...utils.base import filter_kwargs
from ...utils.noise_schedule import flow_match_sigma, validate_flow_match_coordinates
from ..component_reduction import reduce_component_log_probs

# Both LTX2 adapters expose one joint video+audio policy, so the authoritative
# component order is fixed here rather than derived from any mapping iteration.
LTX2_COMPONENT_ORDER: Tuple[str, ...] = ("video", "audio")

# Latent-shaped scheduler outputs a trainer replays per component. Every other
# callback result stays a legacy ``extra_kwargs`` entry because it either carries
# no component structure (``noise_level``) or is a per-sample statistic.
LTX2_STRUCTURED_CALLBACK_FIELDS: Tuple[str, ...] = (
    "next_latents",
    "next_latents_mean",
    "velocity",
)

_SIGNED_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


def combine_modality_log_prob(
    video_log_prob: torch.Tensor,
    audio_log_prob: torch.Tensor,
    n_video: int,
    n_audio: int,
) -> torch.Tensor:
    """Element-weighted mean of the per-step video/audio log-probs.

    LTX2 steps video and audio with two scheduler instances, each returning a
    per-sample log_prob already meaned over its own latent dims. Weighting by the
    element counts ``n_video`` / ``n_audio`` reproduces the mean that a single
    scheduler over the concatenated ``[video|audio]`` latent would produce (mean
    over all latent dims), so the joint log_prob keeps the same scale as a
    video-only policy.

    Args:
        video_log_prob: Per-sample video log_prob, shape ``(B,)``.
        audio_log_prob: Per-sample audio log_prob, shape ``(B,)``.
        n_video: Number of video latent elements per sample (the tensor passed to
            the video ``step()``).
        n_audio: Number of audio latent elements per sample.

    Returns:
        Per-sample joint log_prob, shape ``(B,)``.
    """
    return reduce_component_log_probs(
        {"video": video_log_prob, "audio": audio_log_prob},
        {"video": n_video, "audio": n_audio},
    )


def build_ltx2_training_component_times(
    adapter: Any, primary_timesteps: torch.Tensor
) -> ComponentTimes:
    """Map one sampled coordinate onto the video and audio training times.

    LTX2 currently runs twin schedules, so both components receive the same
    numeric timestep and sigma. Decoupled training keeps the next coordinate at
    zero, and the mapping consumes no randomness.

    Args:
        adapter: Adapter declaring ``trajectory_component_order``.
        primary_timesteps: Primary scheduler coordinates of shape ``(B,)``.

    Returns:
        Video/audio timesteps and sigmas in authoritative component order.
    """
    if not isinstance(primary_timesteps, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor primary_timesteps for build_training_component_times on "
            f"{type(adapter).__name__}, received {type(primary_timesteps).__name__}: "
            f"{primary_timesteps!r}"
        )
    if primary_timesteps.ndim != 1:
        raise ValueError(
            "expected primary_timesteps with one scheduler coordinate per sample, shape "
            f"(B,), received {tuple(primary_timesteps.shape)}"
        )
    _require_ltx2_component_order(adapter, adapter.trajectory_component_order, "adapter")
    sigma = flow_match_sigma(primary_timesteps)
    zero_timestep = torch.zeros_like(primary_timesteps)
    zero_sigma = torch.zeros_like(sigma)
    return ComponentTimes(
        timestep={"video": primary_timesteps, "audio": primary_timesteps},
        next_timestep={"video": zero_timestep, "audio": zero_timestep},
        sigma={"video": sigma, "audio": sigma},
        next_sigma={"video": zero_sigma, "audio": zero_sigma},
    )


def draw_ltx2_forward_process_noise(
    adapter: Any,
    clean_state: LatentState,
    times: ComponentTimes,
    *,
    generator: Optional[torch.Generator],
) -> NoisedState:
    """Draw video noise then audio noise, then apply it deterministically.

    Args:
        adapter: Adapter owning the deterministic application hook.
        clean_state: Clean latent state in authoritative component order.
        times: Component times including each component's current sigma.
        generator: Optional generator shared by both ordered draws.

    Returns:
        Noised state, target velocity, and the sampled noise.
    """
    _require_ltx2_component_order(adapter, clean_state.component_names, "clean_state")
    noise: Dict[str, torch.Tensor] = {}
    for name in LTX2_COMPONENT_ORDER:
        component = clean_state.components[name]
        noise[name] = randn_tensor(
            component.shape,
            generator=generator,
            device=component.device,
            dtype=component.dtype,
        )
    return adapter.apply_forward_process_noise(clean_state, times, LatentState(noise))


def build_ltx2_component_step_output(
    adapter: Any,
    *,
    video_output: SDESchedulerOutput,
    audio_output: SDESchedulerOutput,
    video_velocity: torch.Tensor,
    audio_velocity: torch.Tensor,
    n_video: int,
    n_audio: int,
    compute_log_prob: bool,
) -> MultiModalStepOutput:
    """Wrap the two scheduler outputs before the legacy concatenation runs.

    Args:
        adapter: Adapter whose forward produced both outputs.
        video_output: Video scheduler step output.
        audio_output: Audio scheduler step output.
        video_velocity: Guided video velocity fed to the video scheduler.
        audio_velocity: Guided audio velocity fed to the audio scheduler.
        n_video: Video elements per sample that carry stochastic DOF.
        n_audio: Audio elements per sample.
        compute_log_prob: Whether the step computed transition log probabilities.

    Returns:
        Ordered component step output holding the real per-modality statistics.
    """

    def paired(field: str) -> Optional[Dict[str, torch.Tensor]]:
        video_value = getattr(video_output, field)
        audio_value = getattr(audio_output, field)
        if (video_value is None) != (audio_value is None):
            raise ValueError(
                f"expected the {type(adapter).__name__} video and audio scheduler steps to agree "
                f"on {field!r}, received video={type(video_value).__name__} and "
                f"audio={type(audio_value).__name__}"
            )
        if video_value is None:
            return None
        return {"video": video_value, "audio": audio_value}

    def paired_statistic(field: str) -> Optional[Dict[str, torch.Tensor]]:
        values = paired(field)
        if values is None:
            return None
        reference = {"video": video_velocity, "audio": audio_velocity}
        return {
            name: _align_statistic_rank(
                values[name],
                reference[name],
                adapter=adapter,
                component=name,
                field=field,
            )
            for name in LTX2_COMPONENT_ORDER
        }

    next_state = paired("next_latents")
    next_state_mean = paired("next_latents_mean")
    component_log_probs = paired("log_prob")
    log_prob = None
    if compute_log_prob and component_log_probs is not None:
        log_prob = combine_modality_log_prob(
            component_log_probs["video"],
            component_log_probs["audio"],
            n_video=n_video,
            n_audio=n_audio,
        )
    return MultiModalStepOutput(
        next_state=None if next_state is None else LatentState(next_state),
        next_state_mean=None if next_state_mean is None else LatentState(next_state_mean),
        std_dev_t=paired_statistic("std_dev_t"),
        dt=paired_statistic("dt"),
        log_prob=log_prob,
        component_log_probs=component_log_probs,
        velocity=LatentState({"video": video_velocity, "audio": audio_velocity}),
    )


def build_ltx2_legacy_callback_view(
    adapter: Any, output: MultiModalStepOutput
) -> SDESchedulerOutput:
    """Rebuild the legacy concatenated ``[video|audio]`` scheduler output.

    The component forward is authoritative, but the rollout still advances its
    loop state and feeds the callback collector with the single concatenated
    ``[video|audio]`` tensor. Concatenating the component fields here reproduces
    that view without a further model or scheduler call, so no extra dispatch or
    random draw enters the rollout.

    Args:
        adapter: Adapter whose component forward produced ``output``.
        output: Ordered component step output.

    Returns:
        Legacy-shaped scheduler output carrying the concatenated latent fields,
        the video component's per-sample statistics, and the joint log prob.
    """
    name = type(adapter).__name__
    if not isinstance(output, MultiModalStepOutput):
        raise TypeError(
            f"expected MultiModalStepOutput from the {name} component forward, "
            f"received {type(output).__name__}: {output!r}"
        )
    if output.next_state is None:
        raise ValueError(
            f"expected the {name} component forward to return next_state so the rollout can "
            "advance, received None"
        )

    def concatenated(state: Optional[LatentState]) -> Optional[torch.Tensor]:
        if state is None:
            return None
        _require_ltx2_component_order(adapter, state.component_names, "component step output")
        return torch.cat([state.components[key] for key in LTX2_COMPONENT_ORDER], dim=1)

    def video_statistic(values: Optional[Mapping[str, torch.Tensor]]) -> Optional[torch.Tensor]:
        return None if values is None else values["video"]

    return SDESchedulerOutput(
        next_latents=concatenated(output.next_state),
        next_latents_mean=concatenated(output.next_state_mean),
        std_dev_t=video_statistic(output.std_dev_t),
        dt=video_statistic(output.dt),
        log_prob=output.log_prob,
        velocity=concatenated(output.velocity),
    )


def split_ltx2_callback_results(
    adapter: Any, callback_results: Mapping[str, Any]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Partition the collected callbacks into structured and legacy results.

    Args:
        adapter: Adapter that produced the rollout, named in every error.
        callback_results: Collected callbacks keyed by requested field name.

    Returns:
        Latent-shaped fields the structured trajectory owns, and the remaining
        results that stay legacy ``extra_kwargs`` entries. Both keep the
        requested field order.
    """
    if not isinstance(callback_results, Mapping):
        raise TypeError(
            f"expected Mapping callback results for {type(adapter).__name__}, "
            f"received {type(callback_results).__name__}"
        )
    structured: Dict[str, torch.Tensor] = {}
    legacy: Dict[str, Any] = {}
    for field, values in callback_results.items():
        if field not in LTX2_STRUCTURED_CALLBACK_FIELDS:
            legacy[field] = values
            continue
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"expected the {type(adapter).__name__} structured callback {field!r} to be a "
                f"torch.Tensor shaped (B, stored, V + A, C), received "
                f"{type(values).__name__}: {values!r}"
            )
        structured[field] = values
    return structured, legacy


def build_ltx2_rollout_log_probs(
    adapter: Any,
    *,
    log_prob_collector: Optional[Any],
    component_log_prob_collectors: Optional[Mapping[str, Any]],
    num_transitions: int,
) -> Dict[str, Any]:
    """Batch the collected joint/component log probabilities and their shared map.

    Args:
        adapter: Adapter that produced the rollout, named in every error.
        log_prob_collector: Joint log-probability collector, or ``None`` when the
            rollout computed no log probabilities.
        component_log_prob_collectors: Per-component collectors that recorded at
            exactly the joint collection steps.
        num_transitions: Number of rollout transitions ``T``.

    Returns:
        Structured-trajectory builder arguments, empty when the rollout stored no
        transition log probability.
    """
    joint = None if log_prob_collector is None else log_prob_collector.get_result()
    if not joint:
        return {}
    _require_ltx2_component_order(
        adapter, tuple(component_log_prob_collectors), "component log-probability collectors"
    )
    return {
        "log_probs": torch.stack(joint, dim=1),
        "component_log_probs": {
            component: torch.stack(component_log_prob_collectors[component].get_result(), dim=1)
            for component in LTX2_COMPONENT_ORDER
        },
        "log_prob_index_map": build_ltx2_sparse_transition_map(
            adapter, log_prob_collector.collected_indices, num_transitions
        ),
    }


def build_ltx2_sparse_transition_map(
    adapter: Any, collected_indices: List[int], num_transitions: int
) -> torch.Tensor:
    """Build the signed rollout-transition map for a sparsely collected stream.

    The rollout collector publishes a dense identity map whenever it was asked
    for every position, which misdescribes a stream that only some transitions
    contribute to -- log probabilities stop once the schedule leaves the SDE
    window. This rebuilds the map from the transitions that actually stored a
    value, keeping the collector ``-1`` sentinel for the rest.

    Args:
        adapter: Adapter that produced the rollout, named in every error.
        collected_indices: Rollout transition indices that stored a value, in
            storage order.
        num_transitions: Number of rollout transitions ``T``.

    Returns:
        Signed map of length ``T``.
    """
    name = type(adapter).__name__
    if num_transitions < 1:
        raise ValueError(
            f"expected at least one {name} rollout transition to map, received {num_transitions}"
        )
    index_map = torch.full((num_transitions,), -1, dtype=torch.long)
    for stored_position, transition in enumerate(collected_indices):
        if not isinstance(transition, int) or isinstance(transition, bool):
            raise TypeError(
                f"expected int {name} collected transition indices, received "
                f"{type(transition).__name__}: {transition!r}"
            )
        if transition < 0 or transition >= num_transitions:
            raise ValueError(
                f"expected {name} collected transition indices in [0, {num_transitions - 1}], "
                f"received {collected_indices}"
            )
        index_map[transition] = stored_position
    return index_map


def build_ltx2_full_component_schedule(
    adapter: Any, rollout_timesteps: torch.Tensor
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Extend the rollout timesteps into the full per-component schedule.

    A rollout of ``T`` transitions visits ``T + 1`` coordinates, and the last one
    is the terminal ``0`` the loop steps toward but never enumerates. Both LTX2
    schedulers currently run the same numeric schedule, yet each component owns
    its own tensors so a future asymmetric schedule needs no caller change.

    Args:
        adapter: Adapter whose rollout produced the timesteps.
        rollout_timesteps: Enumerated rollout coordinates of shape ``(T,)``.

    Returns:
        Full ``(T + 1,)`` timesteps and sigmas per component, in authoritative order.
    """
    name = type(adapter).__name__
    if not isinstance(rollout_timesteps, torch.Tensor):
        raise TypeError(
            f"expected torch.Tensor rollout timesteps for the {name} full component schedule, "
            f"received {type(rollout_timesteps).__name__}: {rollout_timesteps!r}"
        )
    if rollout_timesteps.ndim != 1 or rollout_timesteps.shape[0] < 1:
        raise ValueError(
            f"expected {name} rollout timesteps shaped (T,) with at least one transition, "
            f"received {tuple(rollout_timesteps.shape)}"
        )
    terminal = torch.zeros(1, dtype=rollout_timesteps.dtype, device=rollout_timesteps.device)
    full_timesteps = torch.cat([rollout_timesteps, terminal])
    full_sigmas = flow_match_sigma(full_timesteps)
    return {
        component: (full_timesteps.clone(), full_sigmas.clone())
        for component in LTX2_COMPONENT_ORDER
    }


def build_ltx2_structured_trajectories(
    adapter: Any,
    *,
    states: torch.Tensor,
    state_index_map: torch.Tensor,
    video_seq_len: int,
    schedule: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
    log_probs: Optional[torch.Tensor] = None,
    component_log_probs: Optional[Mapping[str, torch.Tensor]] = None,
    log_prob_index_map: Optional[torch.Tensor] = None,
    callbacks: Optional[Mapping[str, torch.Tensor]] = None,
    callback_index_map: Optional[torch.Tensor] = None,
    video_active_mask: Optional[torch.Tensor] = None,
) -> List[StructuredTrajectory]:
    """Split batched concatenated rollout results into one trajectory per sample.

    The rollout collects the single concatenated ``[video|audio]`` tensor, which
    preserves the numerics and the RNG stream; this helper performs the component
    split afterwards, so every stored tensor is a view of exactly the same values.

    Args:
        adapter: Adapter that produced the rollout, named in every error.
        states: Collected states shaped ``(B, stored, V + A, C)``.
        state_index_map: Rollout-position map of length ``T + 1`` using ``-1``
            for an uncollected position.
        video_seq_len: Number of leading video tokens in each collected state.
        schedule: Full ``(T + 1,)`` timesteps/sigmas per component.
        log_probs: Optional joint log probabilities shaped ``(B, stored)``.
        component_log_probs: Optional per-component log probabilities sharing the
            joint compact length and map.
        log_prob_index_map: Rollout-transition map of length ``T`` for the log
            probabilities.
        callbacks: Optional latent-shaped callback results, each shaped like
            ``states``.
        callback_index_map: Rollout-transition map of length ``T`` shared by every
            callback field.
        video_active_mask: Optional ``(B, V)`` or ``(B, V, 1)`` boolean mask
            marking the video tokens that carry stochastic degrees of freedom.

    Returns:
        One ``StructuredTrajectory`` per sample, in batch order.
    """
    name = type(adapter).__name__
    _require_ltx2_component_order(adapter, adapter.trajectory_component_order, "adapter")

    if not isinstance(states, torch.Tensor):
        raise TypeError(
            f"expected torch.Tensor states for the {name} structured trajectory builder, "
            f"received {type(states).__name__}: {states!r}"
        )
    if states.ndim != 4:
        raise ValueError(
            f"expected {name} collected states shaped (B, stored, V + A, C), "
            f"received {tuple(states.shape)}"
        )
    batch_size, num_stored, total_seq_len, _ = states.shape
    if not isinstance(video_seq_len, int) or isinstance(video_seq_len, bool):
        raise TypeError(
            f"expected int video_seq_len for the {name} structured trajectory builder, "
            f"received {type(video_seq_len).__name__}: {video_seq_len!r}"
        )
    if video_seq_len < 1 or video_seq_len >= total_seq_len:
        raise ValueError(
            f"expected {name} video_seq_len in [1, {total_seq_len - 1}] to split the collected "
            f"states shaped {tuple(states.shape)}, received {video_seq_len}"
        )

    schedule_length = _validate_ltx2_component_schedule(adapter, schedule)
    num_transitions = schedule_length - 1
    _validate_ltx2_index_map(
        adapter,
        state_index_map,
        field="state_index_map",
        expected_length=schedule_length,
        num_stored=num_stored,
        stored_description=f"{num_stored} stored states",
    )
    _validate_ltx2_rollout_log_probs(
        adapter,
        log_probs=log_probs,
        component_log_probs=component_log_probs,
        log_prob_index_map=log_prob_index_map,
        batch_size=batch_size,
        num_transitions=num_transitions,
    )
    structured_callbacks = _validate_ltx2_rollout_callbacks(
        adapter,
        callbacks=callbacks,
        callback_index_map=callback_index_map,
        batch_size=batch_size,
        total_seq_len=total_seq_len,
        num_transitions=num_transitions,
    )
    component_masks = _validate_ltx2_video_active_mask(
        adapter,
        video_active_mask,
        states=states,
        video_seq_len=video_seq_len,
    )

    def split(values: torch.Tensor, component: str) -> torch.Tensor:
        return (
            values[..., :video_seq_len, :]
            if component == "video"
            else values[..., video_seq_len:, :]
        )

    return unstack_structured_trajectories(
        component_order=LTX2_COMPONENT_ORDER,
        states={component: split(states, component) for component in LTX2_COMPONENT_ORDER},
        schedule=schedule,
        state_index_maps={component: state_index_map for component in LTX2_COMPONENT_ORDER},
        active_masks=component_masks,
        log_probs=log_probs,
        log_prob_index_map=log_prob_index_map,
        component_log_probs=component_log_probs,
        callbacks=(
            None
            if not structured_callbacks
            else {
                field: {component: split(values, component) for component in LTX2_COMPONENT_ORDER}
                for field, values in structured_callbacks.items()
            }
        ),
        callback_index_maps={component: callback_index_map for component in LTX2_COMPONENT_ORDER},
    )


def _validate_ltx2_component_schedule(
    adapter: Any, schedule: Mapping[str, Tuple[torch.Tensor, torch.Tensor]]
) -> int:
    """Validate the full per-component schedule and return its shared length."""
    name = type(adapter).__name__
    if not isinstance(schedule, Mapping):
        raise TypeError(
            f"expected Mapping[str, Tuple[torch.Tensor, torch.Tensor]] schedule for {name}, "
            f"received {type(schedule).__name__}"
        )
    _require_ltx2_component_order(adapter, tuple(schedule), "schedule")
    schedule_length: Optional[int] = None
    for component in LTX2_COMPONENT_ORDER:
        entry = schedule[component]
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(
                f"expected a (timesteps, sigmas) pair for the {name} component {component!r} "
                f"schedule, received {type(entry).__name__}: {entry!r}"
            )
        timesteps, sigmas = entry
        for field, values in (("timesteps", timesteps), ("sigmas", sigmas)):
            if not isinstance(values, torch.Tensor):
                raise TypeError(
                    f"expected torch.Tensor for the {name} component {component!r} {field}, "
                    f"received {type(values).__name__}: {values!r}"
                )
            if values.ndim != 1:
                raise ValueError(
                    f"expected {name} component {component!r} {field} shaped (T + 1,), "
                    f"received {tuple(values.shape)}"
                )
        if timesteps.shape != sigmas.shape:
            raise ValueError(
                f"expected {name} component {component!r} timesteps/sigmas of one shared length, "
                f"received timesteps {tuple(timesteps.shape)} and sigmas {tuple(sigmas.shape)}"
            )
        if timesteps.shape[0] < 2:
            raise ValueError(
                f"expected {name} component {component!r} full schedule to hold at least one "
                f"rollout transition plus the terminal coordinate, shape (T + 1,) with T >= 1, "
                f"received {tuple(timesteps.shape)}"
            )
        validate_flow_match_coordinates(
            timesteps,
            sigmas,
            identifier=f"{name} component {component!r} schedule timesteps/sigmas",
        )
        if float(timesteps[-1]) != 0.0 or float(sigmas[-1]) != 0.0:
            raise ValueError(
                f"expected the {name} component {component!r} full schedule to end at the "
                f"terminal zero coordinate, received timestep {float(timesteps[-1])} and sigma "
                f"{float(sigmas[-1])}"
            )
        if schedule_length is None:
            schedule_length = timesteps.shape[0]
        elif timesteps.shape[0] != schedule_length:
            raise ValueError(
                f"expected every {name} component schedule to share one length "
                f"{schedule_length}, received {timesteps.shape[0]} for component {component!r}"
            )
    return int(schedule_length)


def _validate_ltx2_index_map(
    adapter: Any,
    index_map: torch.Tensor,
    *,
    field: str,
    expected_length: int,
    num_stored: int,
    stored_description: str,
) -> None:
    """Validate one sparse rollout map against the entries it addresses."""
    name = type(adapter).__name__
    if not isinstance(index_map, torch.Tensor):
        raise TypeError(
            f"expected torch.Tensor {field} for {name}, "
            f"received {type(index_map).__name__}: {index_map!r}"
        )
    if index_map.ndim != 1:
        raise ValueError(
            f"expected {name} {field} shaped ({expected_length},), "
            f"received {tuple(index_map.shape)}"
        )
    if index_map.dtype not in _SIGNED_INTEGER_DTYPES:
        raise TypeError(f"expected {name} signed integer {field}, received dtype {index_map.dtype}")
    if index_map.shape[0] != expected_length:
        raise ValueError(
            f"expected {name} {field} length {expected_length} to match the rollout schedule, "
            f"received {index_map.shape[0]}"
        )
    if int(index_map.min().item()) < -1 or int(index_map.max().item()) >= num_stored:
        raise ValueError(
            f"expected {name} {field} values in [-1, {num_stored - 1}] for {stored_description}, "
            f"received {index_map.tolist()}"
        )


def _validate_ltx2_rollout_log_probs(
    adapter: Any,
    *,
    log_probs: Optional[torch.Tensor],
    component_log_probs: Optional[Mapping[str, torch.Tensor]],
    log_prob_index_map: Optional[torch.Tensor],
    batch_size: int,
    num_transitions: int,
) -> None:
    """Validate the joint and per-component log probabilities and their shared map."""
    name = type(adapter).__name__
    if log_probs is None:
        if component_log_probs is not None:
            raise ValueError(
                f"expected {name} component_log_probs to accompany the joint log_probs, received "
                f"components {tuple(component_log_probs)} with log_probs=None"
            )
        if log_prob_index_map is not None:
            raise ValueError(
                f"expected {name} log_probs alongside a log_prob_index_map, received "
                f"log_probs=None with map {log_prob_index_map.tolist()}"
            )
        return
    if not isinstance(log_probs, torch.Tensor):
        raise TypeError(
            f"expected torch.Tensor log_probs for {name}, "
            f"received {type(log_probs).__name__}: {log_probs!r}"
        )
    if log_probs.ndim != 2 or log_probs.shape[0] != batch_size:
        raise ValueError(
            f"expected {name} log_probs shaped ({batch_size}, stored), "
            f"received {tuple(log_probs.shape)}"
        )
    if log_prob_index_map is None:
        raise ValueError(
            f"expected a log_prob_index_map alongside log_probs for {name}, received None with "
            f"log_probs shaped {tuple(log_probs.shape)}"
        )
    _validate_ltx2_index_map(
        adapter,
        log_prob_index_map,
        field="log_prob_index_map",
        expected_length=num_transitions,
        num_stored=log_probs.shape[1],
        stored_description=f"{log_probs.shape[1]} stored log probabilities",
    )
    if component_log_probs is None:
        return
    if not isinstance(component_log_probs, Mapping):
        raise TypeError(
            f"expected Mapping[str, torch.Tensor] component_log_probs for {name}, "
            f"received {type(component_log_probs).__name__}"
        )
    _require_ltx2_component_order(adapter, tuple(component_log_probs), "component_log_probs")
    for component in LTX2_COMPONENT_ORDER:
        values = component_log_probs[component]
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"expected torch.Tensor {name} component_log_probs[{component!r}], "
                f"received {type(values).__name__}: {values!r}"
            )
        if values.shape != log_probs.shape:
            raise ValueError(
                f"expected {name} component_log_probs[{component!r}] shaped "
                f"{tuple(log_probs.shape)} to match the joint log_probs, received "
                f"{tuple(values.shape)}"
            )


def _validate_ltx2_rollout_callbacks(
    adapter: Any,
    *,
    callbacks: Optional[Mapping[str, torch.Tensor]],
    callback_index_map: Optional[torch.Tensor],
    batch_size: int,
    total_seq_len: int,
    num_transitions: int,
) -> Dict[str, torch.Tensor]:
    """Validate the latent-shaped callback results and return them in order."""
    name = type(adapter).__name__
    if callbacks is not None and not isinstance(callbacks, Mapping):
        raise TypeError(
            f"expected Mapping[str, torch.Tensor] callbacks for {name}, "
            f"received {type(callbacks).__name__}"
        )
    structured: Dict[str, torch.Tensor] = {} if callbacks is None else dict(callbacks)
    if not structured:
        return structured
    if callback_index_map is None:
        raise ValueError(
            f"expected a callback_index_map alongside callback fields {tuple(structured)} for "
            f"{name}, received None"
        )
    for field, values in structured.items():
        if not isinstance(field, str) or not field:
            raise ValueError(
                f"expected non-empty string {name} callback field names, received {field!r}"
            )
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"expected the {name} callback {field!r} to be a torch.Tensor, "
                f"received {type(values).__name__}: {values!r}"
            )
        if values.ndim != 4 or values.shape[0] != batch_size or values.shape[2] != total_seq_len:
            raise ValueError(
                f"expected {name} callback {field!r} shaped (B, stored, V + A, C) matching the "
                f"collected states ({batch_size}, stored, {total_seq_len}, C), received "
                f"{tuple(values.shape)}"
            )
        _validate_ltx2_index_map(
            adapter,
            callback_index_map,
            field="callback_index_map",
            expected_length=num_transitions,
            num_stored=values.shape[1],
            stored_description=f"{values.shape[1]} stored callback {field!r} entries",
        )
    return structured


def _validate_ltx2_video_active_mask(
    adapter: Any,
    video_active_mask: Optional[torch.Tensor],
    *,
    states: torch.Tensor,
    video_seq_len: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Return the per-component static masks, or none when nothing is conditioned.

    Audio carries no fixed conditioning, but the trainer bridge reads either every
    component's mask or none, so a full-active audio mask accompanies the video one.
    """
    if video_active_mask is None:
        return None
    name = type(adapter).__name__
    batch_size, _, total_seq_len, _ = states.shape
    if not isinstance(video_active_mask, torch.Tensor):
        raise TypeError(
            f"expected torch.Tensor or None video_active_mask for {name}, "
            f"received {type(video_active_mask).__name__}: {video_active_mask!r}"
        )
    if video_active_mask.dtype is not torch.bool:
        raise TypeError(
            f"expected {name} video_active_mask dtype torch.bool marking the generated video "
            f"tokens, received {video_active_mask.dtype}"
        )
    if tuple(video_active_mask.shape) not in (
        (batch_size, video_seq_len),
        (batch_size, video_seq_len, 1),
    ):
        raise ValueError(
            f"expected {name} video_active_mask shaped ({batch_size}, {video_seq_len}) or "
            f"({batch_size}, {video_seq_len}, 1), received {tuple(video_active_mask.shape)}"
        )
    counts = video_active_mask.reshape(batch_size, -1).sum(dim=1)
    if int(counts.min().item()) <= 0:
        raise ValueError(
            f"expected {name} video_active_mask to leave a positive generated token count for "
            f"every sample, received counts {counts.tolist()}"
        )
    video_mask = (
        video_active_mask if video_active_mask.ndim == 3 else video_active_mask.unsqueeze(-1)
    )
    return {
        "video": video_mask,
        "audio": torch.ones(
            (batch_size, total_seq_len - video_seq_len, 1),
            dtype=torch.bool,
            device=states.device,
        ),
    }


def build_ltx2_joint_forward_kwargs(
    adapter: Any,
    *,
    batch: StackedSampleBatch,
    state: LatentState,
    times: ComponentTimes,
    next_state: Optional[LatentState],
    compute_log_prob: bool,
    return_fields: Tuple[str, ...],
    noise_level: Optional[float],
    forward_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Pack an ordered component state for the legacy concatenated ``forward``.

    The caller runs the full input contract first (``validate_ltx2_forward_state_inputs``
    or its I2AV superset); only the component order is re-checked here, so an unvalidated
    caller still gets the contextual error instead of a ``KeyError``.

    Args:
        adapter: Adapter whose ``forward`` will receive the packed arguments.
        batch: Collated batch supplying conditioning and ``video_seq_len``.
        state: Current state in authoritative component order.
        times: Current and next times in authoritative component order.
        next_state: Optional stored next state in authoritative component order.
        compute_log_prob: Whether to compute transition log probabilities.
        return_fields: Scheduler output fields requested from ``forward``.
        noise_level: Scheduler noise-level override.
        forward_kwargs: Model-conditioning arguments resolved by the wrapper.

    Returns:
        Keyword arguments for ``adapter.forward`` in component-return mode.
    """
    _require_ltx2_component_orders(adapter, state=state, times=times, next_state=next_state)

    timestep = times.timestep["video"]
    next_timestep = times.next_timestep["video"]
    video = state.components["video"]
    audio = state.components["audio"]
    video_seq_len = video.shape[1]
    stored_seq_len = batch.get("video_seq_len")
    if not isinstance(stored_seq_len, int) or isinstance(stored_seq_len, bool):
        raise TypeError(
            f"expected int batch video_seq_len for {type(adapter).__name__} forward_state, "
            f"received {type(stored_seq_len).__name__}: {stored_seq_len!r}"
        )
    if stored_seq_len != video_seq_len:
        raise ValueError(
            f"expected batch video_seq_len {stored_seq_len} to match the video component "
            f"sequence length {video_seq_len} for {type(adapter).__name__} forward_state"
        )

    call_kwargs = filter_kwargs(adapter.forward, **forward_kwargs)
    call_kwargs.pop("video_seq_len", None)
    return {
        "t": timestep,
        "t_next": next_timestep,
        "latents": torch.cat([video, audio], dim=1),
        "next_latents": (
            None
            if next_state is None
            else torch.cat([next_state.components["video"], next_state.components["audio"]], dim=1)
        ),
        "video_seq_len": video_seq_len,
        "compute_log_prob": compute_log_prob,
        "return_kwargs": list(return_fields),
        "noise_level": noise_level,
        "_return_components": True,
        **call_kwargs,
    }


def validate_ltx2_forward_state_inputs(
    adapter: Any,
    *,
    state: LatentState,
    times: ComponentTimes,
    next_state: Optional[LatentState],
) -> None:
    """Check every component input the joint LTX2 forward is about to consume.

    The concatenated ``forward`` cannot see component boundaries, so a mismatch in
    channel count, dtype, device or stored next state would either raise deep inside
    ``torch.cat`` or silently mis-attribute one modality's values to the other.

    Args:
        adapter: Adapter running the component forward.
        state: Current state in authoritative component order.
        times: Current and next times in authoritative component order.
        next_state: Optional stored next state in authoritative component order.

    Raises:
        ValueError: If any order, shape, dtype, device or mask expectation fails.
    """
    _require_ltx2_component_orders(adapter, state=state, times=times, next_state=next_state)

    name = type(adapter).__name__
    video = state.components["video"]
    audio = state.components["audio"]
    for component, values in (("video", video), ("audio", audio)):
        if values.ndim != 3:
            raise ValueError(
                f"expected {name} state component {component!r} packed as (B, S, C), received "
                f"shape {tuple(values.shape)}"
            )
    if audio.shape[0] != video.shape[0]:
        raise ValueError(
            f"expected {name} batch size of state component 'audio' to match 'video' "
            f"({video.shape[0]}), received {audio.shape[0]}"
        )
    if audio.shape[2] != video.shape[2]:
        raise ValueError(
            f"expected {name} channel dimension of state component 'audio' to match 'video' "
            f"({video.shape[2]}), received {audio.shape[2]}"
        )
    if audio.dtype is not video.dtype:
        raise ValueError(
            f"expected {name} dtype of state component 'audio' to match 'video' "
            f"({video.dtype}), received {audio.dtype}"
        )
    if audio.device != video.device:
        raise ValueError(
            f"expected {name} device of state component 'audio' to match 'video' "
            f"({video.device}), received {audio.device}"
        )

    batch_size = video.shape[0]
    for field in ("timestep", "next_timestep"):
        values = getattr(times, field)
        for component in LTX2_COMPONENT_ORDER:
            coordinate = values[component]
            identifier = f"times.{field}[{component!r}]"
            if not isinstance(coordinate, torch.Tensor):
                raise TypeError(
                    f"expected {name} {identifier} as a torch.Tensor, received "
                    f"{type(coordinate).__name__}: {coordinate!r}"
                )
            if coordinate.shape != (batch_size,):
                raise ValueError(
                    f"expected {name} {identifier} with one coordinate per sample, shape "
                    f"({batch_size},), received {tuple(coordinate.shape)}"
                )
            if coordinate.device != video.device:
                raise ValueError(
                    f"expected {name} {identifier} on the state device {video.device}, received "
                    f"{coordinate.device}"
                )
    _require_joint_coordinate(adapter, times.timestep, "timestep")
    _require_joint_coordinate(adapter, times.next_timestep, "next_timestep")

    if next_state is None:
        return
    for component in LTX2_COMPONENT_ORDER:
        current = state.components[component]
        stored = next_state.components[component]
        identifier = f"next_state component {component!r}"
        if stored.shape != current.shape:
            raise ValueError(
                f"expected {name} {identifier} to match the state shape "
                f"{tuple(current.shape)}, received {tuple(stored.shape)}"
            )
        if stored.dtype is not current.dtype:
            raise ValueError(
                f"expected {name} {identifier} dtype {current.dtype}, received {stored.dtype}"
            )
        if stored.device != current.device:
            raise ValueError(
                f"expected {name} {identifier} device {current.device}, received {stored.device}"
            )
    _require_matching_state_masks(adapter, state=state, next_state=next_state)


def _require_matching_state_masks(
    adapter: Any, *, state: LatentState, next_state: LatentState
) -> None:
    """Require the stored next state to carry exactly the current static masks."""
    name = type(adapter).__name__
    if state.active_masks is None:
        if next_state.active_masks is not None:
            raise ValueError(
                f"expected {name} next_state active_masks to be None because the state carries "
                f"none, received {tuple(next_state.active_masks)}"
            )
        return
    if next_state.active_masks is None:
        raise ValueError(
            f"expected {name} next_state active_masks to be present because the state carries "
            "masks, received None"
        )
    for component in LTX2_COMPONENT_ORDER:
        current = state.active_masks[component]
        stored = next_state.active_masks[component]
        identifier = f"next_state active_masks[{component!r}]"
        if stored.shape != current.shape or not bool(torch.equal(stored, current)):
            raise ValueError(
                f"expected {name} {identifier} to equal the state mask of shape "
                f"{tuple(current.shape)} with {int(current.sum().item())} active entries, "
                f"received shape {tuple(stored.shape)} with {int(stored.sum().item())}"
            )


def attach_ltx2_state_masks(
    state: LatentState, output: MultiModalStepOutput
) -> MultiModalStepOutput:
    """Carry the input active masks onto every shape-compatible returned state.

    Args:
        state: Input state supplying the static component masks.
        output: Component step output produced by ``forward``.

    Returns:
        The same output, with masks attached where they broadcast.
    """
    if not isinstance(output, MultiModalStepOutput):
        raise TypeError(
            "expected MultiModalStepOutput from the LTX2 component forward, "
            f"received {type(output).__name__}"
        )
    if state.active_masks is None:
        return output
    for field in ("next_state", "next_state_mean", "velocity"):
        component_state = getattr(output, field)
        if component_state is None or component_state.active_masks is not None:
            continue
        if not all(
            _mask_broadcasts(state.active_masks[name], component_state.components[name])
            for name in component_state.component_names
        ):
            continue
        setattr(
            output,
            field,
            LatentState(dict(component_state.components), active_masks=state.active_masks),
        )
    return output


def validate_i2av_forward_state_inputs(
    adapter: Any,
    *,
    batch: StackedSampleBatch,
    state: LatentState,
    times: ComponentTimes,
    next_state: Optional[LatentState],
) -> None:
    """Check the shared component contract, then the I2AV conditioning contract.

    The shared contract runs first so a missing or reordered component is reported
    with its expected/received context instead of failing where this function reads
    ``active_masks['video']``.

    Args:
        adapter: Adapter running the component forward.
        batch: Collated batch that must carry ``conditioning_mask``.
        state: Current state whose masks describe the stochastic video tokens.
        times: Current and next times in authoritative component order.
        next_state: Optional stored next state in authoritative component order.

    Raises:
        ValueError: If the shared contract fails or the masks disagree with the
            batch conditioning mask.
    """
    validate_ltx2_forward_state_inputs(adapter, state=state, times=times, next_state=next_state)
    conditioning_mask = batch.get("conditioning_mask")
    if conditioning_mask is None:
        raise ValueError(
            f"expected batch conditioning_mask for {type(adapter).__name__} forward_state, "
            f"received None with batch keys {tuple(sorted(batch.keys()))}"
        )
    if not isinstance(conditioning_mask, torch.Tensor):
        raise TypeError(
            f"expected torch.Tensor batch conditioning_mask for {type(adapter).__name__} "
            f"forward_state, received {type(conditioning_mask).__name__}"
        )
    if state.active_masks is None:
        raise ValueError(
            f"expected LatentState.active_masks marking the generated video tokens for "
            f"{type(adapter).__name__} forward_state, received None"
        )
    expected = ~conditioning_mask.bool()
    video_mask = state.active_masks["video"]
    if (
        video_mask.shape[: expected.ndim] != expected.shape
        or video_mask.numel() != expected.numel()
    ):
        raise ValueError(
            f"expected LatentState.active_masks['video'] to cover the conditioning_mask shape "
            f"{tuple(expected.shape)} (a broadcast channel axis is allowed), received "
            f"{tuple(video_mask.shape)}"
        )
    if not bool(torch.equal(video_mask.reshape(expected.shape), expected)):
        raise ValueError(
            "expected LatentState.active_masks['video'] to equal ~conditioning_mask, received "
            f"{int(video_mask.sum().item())} active tokens against "
            f"{int(expected.sum().item())} generated tokens"
        )
    audio_mask = state.active_masks["audio"]
    if not bool(audio_mask.all()):
        raise ValueError(
            "expected LatentState.active_masks['audio'] to be all active because LTX2 audio "
            f"carries no fixed conditioning, received {int(audio_mask.sum().item())} of "
            f"{audio_mask.numel()} active entries"
        )


def _align_statistic_rank(
    statistic: torch.Tensor,
    reference: torch.Tensor,
    *,
    adapter: Any,
    component: str,
    field: str,
) -> torch.Tensor:
    """Re-rank a per-sample scheduler statistic onto the packed component layout.

    I2AV steps the video scheduler on unpacked ``(B, C, F, H, W)`` frames, so its
    statistics carry that rank while the component latents stay packed
    ``(B, S, C)``. Trailing singleton axes are dropped so the same value keeps
    broadcasting against the component it describes instead of silently producing
    a higher-rank result.
    """
    if statistic.shape == reference.shape:
        return statistic
    if statistic.shape[0] != reference.shape[0] or statistic.numel() != statistic.shape[0]:
        raise ValueError(
            f"expected {type(adapter).__name__} {field!r} for component {component!r} to be "
            f"per-sample or shaped like the component {tuple(reference.shape)}, received "
            f"{tuple(statistic.shape)}"
        )
    return statistic.reshape((statistic.shape[0],) + (1,) * (reference.ndim - 1))


def _require_ltx2_component_orders(
    adapter: Any,
    *,
    state: LatentState,
    times: ComponentTimes,
    next_state: Optional[LatentState],
) -> None:
    """Check every component mapping the joint forward indexes by name."""
    _require_ltx2_component_order(adapter, state.component_names, "state")
    _require_ltx2_component_order(adapter, tuple(times.timestep), "times.timestep")
    _require_ltx2_component_order(adapter, tuple(times.next_timestep), "times.next_timestep")
    if next_state is not None:
        _require_ltx2_component_order(adapter, next_state.component_names, "next_state")


def _require_ltx2_component_order(adapter: Any, received: Tuple[str, ...], field: str) -> None:
    if tuple(received) != LTX2_COMPONENT_ORDER:
        raise ValueError(
            f"expected {type(adapter).__name__} {field} component order "
            f"{LTX2_COMPONENT_ORDER}, received {tuple(received)}"
        )


def _require_joint_coordinate(
    adapter: Any, values: Mapping[str, torch.Tensor], field: str
) -> torch.Tensor:
    """Return the shared coordinate the joint LTX2 transformer can accept."""
    video = values["video"]
    audio = values["audio"]
    if video.shape != audio.shape or not bool(torch.equal(video, audio)):
        raise ValueError(
            f"expected {type(adapter).__name__} {field} entries for 'video' and 'audio' to be "
            f"equal because the LTX2 transformer takes one joint time coordinate, received "
            f"video {video.tolist()} and audio {audio.tolist()}"
        )
    return video


def _mask_broadcasts(mask: torch.Tensor, values: torch.Tensor) -> bool:
    if mask.ndim != values.ndim:
        return False
    return all(mask_dim in (1, value_dim) for mask_dim, value_dim in zip(mask.shape, values.shape))
