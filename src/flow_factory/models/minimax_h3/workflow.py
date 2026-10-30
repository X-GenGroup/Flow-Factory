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

"""Own MiniMax H3 workflow loading, setup, and execution contracts."""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch

from ...samples import (
    ComponentTimes,
    LatentState,
    MiniMaxH3FL2VASample,
    MiniMaxH3Ref2VASample,
    MiniMaxH3T2VASample,
    MultiModalStepOutput,
    StructuredTrajectory,
)
from ...scheduler import MiniMaxH3SDEScheduler, SchedulerGroup
from ..runtime import ModularPipelineRuntime
from ._chunking import (
    install_h3_attention_norm_chunking,
    install_h3_feed_forward_chunking,
)
from ._common import (
    build_structured_trajectories,
)
from ._common import build_training_component_times as build_h3_component_times
from ._common import validate_target_state
from ._diffusers_cache import h3_diffusers_cache_context, reset_h3_diffusers_cache
from .blocks import encode_h3_workflow_inputs, prepare_h3_rollout_state
from .decoding import decode_h3_targets
from .denoise import forward_h3_state
from .dependency import require_minimax_h3_support
from .layout import build_h3_schedule_plan

_COMPONENT_ORDER = ("video", "audio")
_LAYOUT_MATRIX_FIELDS = ("position_ids",)
_LAYOUT_INDEX_FIELDS = (
    "token_tags",
    "video_indices",
    "audio_indices",
    "text_indices",
)
_LAYOUT_COUNT_FIELDS = (
    "num_condition_video_rows",
    "num_condition_audio_rows",
)
_GEOMETRY_FIELDS = (
    "height",
    "width",
    "num_frames",
    "num_latent_frames",
    "latent_height",
    "latent_width",
    "num_audio_latents",
)
_CALLBACK_STATE_FIELDS = {
    "next_latents": "next_state",
    "next_latents_mean": "next_state_mean",
    "velocity": "velocity",
}


def load_h3_workflow_pipeline(
    model_name_or_path: str,
    *,
    workflow: str,
) -> Any:
    """Load a workflow-pruned H3 pipeline from a local directory or Hub repo."""
    symbols = require_minimax_h3_support()
    pipeline = symbols.ModularPipeline.from_pretrained(
        model_name_or_path,
        workflow=workflow,
    )
    if Path(model_name_or_path).is_dir():
        # Hub snapshots retain their original repo ID inside each ComponentSpec.
        # Point only those sources at the local snapshot so load_components()
        # reads its subfolders from disk instead of downloading them again.
        for name in pipeline.pretrained_component_names:
            spec = pipeline._component_specs[name]
            spec.pretrained_model_name_or_path = model_name_or_path
            spec.revision = None
    return pipeline


def build_h3_component_runtime(adapter: Any) -> ModularPipelineRuntime:
    """Build the pruned runtime and prepare the H3 transformer for bounded execution.

    Args:
        adapter: H3 adapter that declares the target transformer and loads the modular pipeline.

    Returns:
        Runtime with the training transformer materialized and its feed-forward and attention
        normalization operations configured for bounded token chunks.

    Raises:
        ValueError: If the adapter targets components outside the H3 training contract.
        TypeError: If the materialized transformer structure cannot accept the required chunking.
    """
    validate_h3_target_components(adapter)
    runtime = ModularPipelineRuntime.from_adapter(adapter, adapter.load_pipeline())
    runtime.materialize_components([adapter.transformer_component_name])
    transformer = runtime.get_component(adapter.transformer_component_name)
    install_h3_feed_forward_chunking(transformer)
    install_h3_attention_norm_chunking(transformer)
    return runtime


def build_h3_scheduler(scheduler_args: Any, *, shift: float) -> MiniMaxH3SDEScheduler:
    """Build one independent H3 scheduler from Flow-Factory scheduler arguments."""
    return MiniMaxH3SDEScheduler(
        shift=shift,
        noise_level=scheduler_args.noise_level,
        sde_steps=scheduler_args.sde_steps,
        num_sde_steps=scheduler_args.num_sde_steps,
        seed=scheduler_args.seed,
        dynamics_type=scheduler_args.dynamics_type,
    )


def build_h3_scheduler_group(adapter: Any) -> SchedulerGroup:
    """Build fresh shift-12/shift-3 schedulers in video/audio order."""
    adapter.audio_scheduler = build_h3_scheduler(adapter.config.scheduler_args, shift=3.0)
    group = SchedulerGroup(
        {"video": adapter.scheduler, "audio": adapter.audio_scheduler},
        primary_name="video",
    )
    num_inference_steps = adapter.training_args.num_inference_steps
    for scheduler in group.values():
        scheduler.set_timesteps(num_inference_steps, device="cpu")
    return group


def init_h3_target_module_map(
    adapter: Any,
) -> Dict[str, Union[List[str], None]]:
    """Validate the sole workflow transformer before checkpoint and LoRA setup."""
    validate_h3_target_components(adapter)
    return adapter._parse_target_modules(
        target_modules=adapter.model_args.target_modules,
        components=adapter.model_args.target_components,
    )


def validate_h3_target_components(adapter: Any) -> None:
    """Reject an invalid training partition before loading model weights."""
    expected_targets = [adapter.transformer_component_name]
    received_targets = adapter.model_args.target_components
    if received_targets != expected_targets:
        raise ValueError(
            f"MiniMax H3 workflow={adapter.workflow!r} expected target_components "
            f"{expected_targets!r}, received {received_targets!r}"
        )


def map_h3_training_component_times(
    adapter: Any, primary_timesteps: torch.Tensor
) -> ComponentTimes:
    """Map primary video coordinates onto the independent audio shift."""
    return build_h3_component_times(
        primary_timesteps,
        video_shift=adapter.scheduler.shift,
        audio_shift=adapter.audio_scheduler.shift,
    )


def preprocess_h3_workflow(adapter: Any, **kwargs: Any) -> Dict[str, Any]:
    """Encode one Arrow-safe B=1 workflow input with pinned blocks."""
    _validate_public_no_cfg_inputs(adapter.workflow, kwargs, "preprocess")
    _validate_workflow_media_inputs(adapter.workflow, kwargs, "preprocess")
    prompt = _single_outer_value(kwargs.get("prompt"), "prompt", adapter.workflow)
    values: Dict[str, Any] = {
        "prompt": prompt,
        "height": kwargs["height"],
        "width": kwargs["width"],
        "num_frames": kwargs["num_frames"],
    }
    if adapter.workflow == "fl2va":
        first_image, last_image = _validate_fl2va_condition_images(kwargs, "preprocess")
        if first_image is not None:
            values["image"] = first_image
        if last_image is not None:
            values["last_image"] = last_image
    elif adapter.workflow == "ref2va":
        references = _single_outer_value(kwargs.get("references"), "references", adapter.workflow)
        values["references"] = _build_pinned_references(references)

    with torch.no_grad():
        encoded = encode_h3_workflow_inputs(adapter.pipeline, values, workflow=adapter.workflow)
    result = {}
    for field, value in encoded.items():
        if field == "prompt_embeds":
            if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
                raise ValueError(
                    f"MiniMax H3 workflow={adapter.workflow!r} preprocessing "
                    f"prompt_embeds expected shape (B=1,N,C), received "
                    f"{getattr(value, 'shape', None)}"
                )
            result[field] = value
            continue
        # Upstream blocks operate on one request and return sample-level
        # scalars/tensors. GeneralDataset.map is batched, so every cache column
        # needs an explicit outer B=1 container. Tuples are not Arrow batch
        # columns; canonicalize them to lists before adding that outer axis.
        result[field] = [list(value) if isinstance(value, tuple) else value]
    if adapter.workflow == "ref2va":
        manifest = kwargs.get("reference_manifest")
        _single_outer_value(manifest, "reference_manifest", adapter.workflow)
        result["reference_manifest"] = manifest
    if any(_is_upstream_reference(value) for value in result.values()):
        raise TypeError(
            f"MiniMax H3 workflow={adapter.workflow!r} preprocessing output contains "
            "an upstream reference object"
        )
    return result


def infer_h3_workflow(adapter: Any, **kwargs: Any) -> List[Any]:
    """Run one B=1 target-only rollout and return a structured sample."""
    _validate_public_no_cfg_inputs(adapter.workflow, kwargs, "inference")
    _validate_workflow_media_inputs(adapter.workflow, kwargs, "inference")
    condition_images = None
    condition_image_slots = None
    if adapter.workflow == "fl2va":
        first_image, last_image = _validate_fl2va_condition_images(kwargs, "inference")
        condition_images = tuple(image for image in (first_image, last_image) if image is not None)
        condition_image_slots = tuple(
            slot
            for slot, image in (
                ("first_frame", first_image),
                ("last_frame", last_image),
            )
            if image is not None
        )
    prompt = kwargs.get("prompt")
    prompt_value = _single_outer_value(prompt, "prompt", adapter.workflow)
    prompt_embeds = kwargs["prompt_embeds"]
    if not isinstance(prompt_embeds, torch.Tensor) or prompt_embeds.shape[0] != 1:
        raise ValueError(
            f"MiniMax H3 workflow={adapter.workflow!r} field='prompt_embeds' requires B=1, "
            f"received {getattr(prompt_embeds, 'shape', None)}"
        )
    layout = _normalize_layout(kwargs)
    geometry = _normalize_geometry(kwargs)
    generator = kwargs.get("generator")
    transformer = adapter.get_component(adapter.transformer_component_name)
    reset_h3_diffusers_cache(transformer, workflow=adapter.workflow)
    rollout_values = dict(kwargs)
    rollout_values.update(layout)
    rollout_values.update(geometry)
    state, condition_prefixes = prepare_h3_rollout_state(
        adapter.pipeline,
        rollout_values,
        workflow=adapter.workflow,
        generator=generator,
    )
    # Constraint #12b: this generator seeds the initial latent (and the official
    # condition-prefix draws in that prepare). Denoising steps do not receive it.
    transformer_parameters = getattr(transformer, "parameters", None)
    if not callable(transformer_parameters):
        raise TypeError(
            f"MiniMax H3 workflow={adapter.workflow!r} prepared transformer expected "
            f"callable parameters(), received {type(transformer).__name__}"
        )
    try:
        execution_device = next(transformer_parameters()).device
    except StopIteration as error:
        raise ValueError(
            f"MiniMax H3 workflow={adapter.workflow!r} prepared transformer has no parameters"
        ) from error
    state = LatentState(
        {component: values.to(execution_device) for component, values in state.components.items()},
        active_masks=(
            None
            if state.active_masks is None
            else {
                component: values.to(execution_device)
                for component, values in state.active_masks.items()
            }
        ),
    )
    condition_prefixes = {
        component: values.to(execution_device) for component, values in condition_prefixes.items()
    }
    # Store and consume the same precision, so replay reproduces the rollout inputs.
    state = adapter.cast_latent_state(state)
    plan = build_h3_schedule_plan(
        adapter.scheduler,
        adapter.audio_scheduler,
        kwargs.get("num_inference_steps", 40),
        layout,
        state.components["video"].device,
    )
    num_transitions = len(plan.schedules["video"][0]) - 1
    trajectory_indices = kwargs.get("trajectory_indices", "all")
    state_positions, transition_positions = _resolve_trajectory_positions(
        trajectory_indices, num_transitions
    )
    collected_states: Optional[Dict[str, List[torch.Tensor]]] = None
    collected_log_probs: Optional[List[torch.Tensor]] = None
    collected_component_log_probs: Optional[Dict[str, List[torch.Tensor]]] = None
    callback_fields = tuple(kwargs.get("extra_call_back_kwargs", ()))
    unknown_callback_fields = tuple(
        field for field in callback_fields if field not in _CALLBACK_STATE_FIELDS
    )
    if unknown_callback_fields:
        raise ValueError(
            f"MiniMax H3 workflow={adapter.workflow!r} callback fields expected a subset of "
            f"{tuple(_CALLBACK_STATE_FIELDS)}, received unknown={unknown_callback_fields}"
        )
    collected_callbacks: Optional[Dict[str, Dict[str, List[torch.Tensor]]]] = None
    if trajectory_indices is not None and state_positions:
        collected_states = {"video": [], "audio": []}
        if 0 in state_positions:
            _append_state(collected_states, state)
        if kwargs.get("compute_log_prob", True) and transition_positions:
            collected_log_probs = []
            collected_component_log_probs = {"video": [], "audio": []}
        if callback_fields and transition_positions:
            collected_callbacks = {field: {"video": [], "audio": []} for field in callback_fields}

    for index in range(num_transitions):
        times = _component_times_at(plan.schedules, index)
        return_fields = ["next_latents"]
        if kwargs.get("compute_log_prob", True):
            return_fields.append("log_prob")
        return_fields.extend(callback_fields)
        return_fields = list(dict.fromkeys(return_fields))
        output = adapter.forward(
            state=state,
            times=times,
            condition_prefixes=condition_prefixes,
            prompt_embeds=prompt_embeds,
            layout=layout,
            generator=None,
            compute_log_prob=kwargs.get("compute_log_prob", True),
            return_fields=tuple(return_fields),
        )
        if output.next_state is None:
            raise ValueError(
                f"MiniMax H3 workflow={adapter.workflow!r} transition {index} "
                "expected target next_state, received None"
            )
        state = adapter.cast_latent_state(output.next_state)
        if collected_states is not None and index + 1 in state_positions:
            _append_state(collected_states, state)
        if collected_log_probs is not None and index in transition_positions:
            _validate_rollout_log_output(output, adapter.workflow, index)
            collected_log_probs.append(output.log_prob)
            for component in _COMPONENT_ORDER:
                collected_component_log_probs[component].append(
                    output.component_log_probs[component]
                )
        if collected_callbacks is not None and index in transition_positions:
            for field in callback_fields:
                value = getattr(output, _CALLBACK_STATE_FIELDS[field], None)
                if not isinstance(value, LatentState):
                    raise TypeError(
                        f"MiniMax H3 workflow={adapter.workflow!r} callback field={field!r} "
                        f"expected LatentState, received {type(value).__name__}"
                    )
                for component in _COMPONENT_ORDER:
                    collected_callbacks[field][component].append(
                        value.components[component].detach()
                    )

    video, audio, sample_rate = adapter.decode_latents(
        state,
        geometry=geometry,
        output_type=kwargs.get("output_type", "pt"),
    )
    trajectory: Optional[StructuredTrajectory] = None
    if collected_states is not None:
        state_map = _index_map(num_transitions + 1, state_positions)
        log_map = (
            None
            if collected_log_probs is None
            else _index_map(num_transitions, transition_positions)
        )
        trajectory = build_structured_trajectories(
            states={
                component: torch.stack(values, dim=1)
                for component, values in collected_states.items()
            },
            state_index_map=state_map,
            schedule=plan.schedules,
            log_probs=(
                None if collected_log_probs is None else torch.stack(collected_log_probs, dim=1)
            ),
            component_log_probs=(
                None
                if collected_component_log_probs is None
                else {
                    component: torch.stack(values, dim=1)
                    for component, values in collected_component_log_probs.items()
                }
            ),
            log_prob_index_map=log_map,
            callbacks=(
                None
                if collected_callbacks is None
                else {
                    field: {
                        component: torch.stack(values, dim=1)
                        for component, values in component_values.items()
                    }
                    for field, component_values in collected_callbacks.items()
                }
            ),
            callback_index_map=(
                None
                if collected_callbacks is None
                else _index_map(num_transitions, transition_positions)
            ),
        )[0]
    sample = _build_h3_sample(
        type(adapter),
        prompt=prompt_value,
        prompt_embeds=prompt_embeds[0],
        video=_decoded_video_sample(video),
        audio=audio[0],
        sample_rate=sample_rate,
        trajectory=trajectory,
        condition_images=condition_images,
        reference_manifest=(
            None
            if adapter.workflow != "ref2va"
            else _single_outer_value(
                kwargs.get("reference_manifest"),
                "reference_manifest",
                adapter.workflow,
            )
        ),
        height=geometry.get("height"),
        width=geometry.get("width"),
        extra_kwargs={
            "condition_prefixes": {
                component: values[0] for component, values in condition_prefixes.items()
            },
            "layout": layout,
            "geometry": geometry,
            **(
                {}
                if condition_image_slots is None
                else {"condition_image_slots": condition_image_slots}
            ),
        },
    )
    return [sample]


_H3_FORWARD_CONDITIONING_FIELDS = frozenset(
    {
        "condition_prefixes",
        "prompt_embeds",
        "layout",
        "generator",
        "attention_kwargs",
        "guidance_scale",
    }
)
_H3_FORWARD_FIELDS = _H3_FORWARD_CONDITIONING_FIELDS | {
    "state",
    "times",
    "next_state",
    "compute_log_prob",
    "return_fields",
    "noise_level",
}


def forward_h3_adapter_state(
    adapter: Any,
    *,
    state: LatentState,
    times: ComponentTimes,
    next_state: Optional[LatentState],
    compute_log_prob: bool,
    return_fields: Sequence[str],
    noise_level: Optional[float],
    forward_kwargs: Mapping[str, Any],
) -> Any:
    """Run the prepared workflow transformer through the common H3 path."""
    validate_target_state(state)
    _require_b1_state(state, adapter.workflow, "forward")
    if next_state is not None:
        validate_target_state(next_state)
        _require_b1_state(next_state, adapter.workflow, "forward next_state")
    _validate_neutral_guidance(adapter.workflow, forward_kwargs)
    prompt_embeds = forward_kwargs["prompt_embeds"]
    condition_prefixes = forward_kwargs["condition_prefixes"]
    if not isinstance(prompt_embeds, torch.Tensor) or prompt_embeds.ndim != 3:
        raise ValueError(
            f"MiniMax H3 workflow={adapter.workflow!r} forward prompt_embeds expected "
            f"shape (B=1,N,C), received {getattr(prompt_embeds, 'shape', None)}"
        )
    if prompt_embeds.shape[0] != 1:
        raise ValueError(
            f"MiniMax H3 workflow={adapter.workflow!r} forward requires B=1, "
            f"received prompt B={prompt_embeds.shape[0]}"
        )
    layout = _normalize_layout({"layout": forward_kwargs["layout"]})
    velocity_only = (
        not compute_log_prob and next_state is None and tuple(return_fields) == ("velocity",)
    )
    transformer = adapter.get_component(adapter.transformer_component_name)
    with h3_diffusers_cache_context(transformer, workflow=adapter.workflow):
        result = forward_h3_state(
            transformer,
            state,
            condition_prefixes,
            prompt_embeds,
            times,
            layout,
            adapter.scheduler,
            adapter.audio_scheduler,
            next_state=next_state,
            generator=forward_kwargs.get("generator"),
            noise_level=noise_level,
            compute_log_prob=compute_log_prob,
            velocity_only=velocity_only,
            attention_kwargs=forward_kwargs.get("attention_kwargs"),
            return_kwargs=return_fields,
            workflow=adapter.workflow,
        )
    if velocity_only:
        if not isinstance(result, LatentState):
            raise TypeError(
                "MiniMax H3 velocity-only forward expected LatentState, "
                f"received {type(result).__name__}"
            )
        return MultiModalStepOutput(velocity=result)
    return result


def build_h3_replay_forward_kwargs(
    forward_kwargs: Mapping[str, Any],
    *,
    state: Optional[LatentState] = None,
    workflow: Optional[str] = None,
) -> Dict[str, Any]:
    """Select the conditioning arguments H3 ``forward`` accepts from a replay batch.

    Replay wrappers receive every collated batch field, while ``forward`` is a strict
    public boundary. Offline T2VA cache rows retain their authoritative layout as flat
    input fields and have no condition latents. This binder nests that layout and builds
    empty prefixes from the current state, so storage casts cannot leave prefix dtype or
    device stale. Online replay already supplies both nested fields and keeps that path.

    Args:
        forward_kwargs: Conditioning arguments resolved from the stored batch.
        state: Current target state, required to bind missing T2VA prefixes.
        workflow: H3 workflow identifier, required when prefixes are missing.

    Returns:
        Conditioning arguments accepted by ``forward``.
    """
    selected = {
        name: value
        for name, value in forward_kwargs.items()
        if name in _H3_FORWARD_CONDITIONING_FIELDS
    }
    if "layout" not in selected:
        layout = _normalize_layout(forward_kwargs)
        required_layout_fields = (
            *_LAYOUT_MATRIX_FIELDS,
            *_LAYOUT_INDEX_FIELDS,
            *_LAYOUT_COUNT_FIELDS,
        )
        missing = tuple(field for field in required_layout_fields if field not in layout)
        if missing:
            raise ValueError(
                f"MiniMax H3 workflow={workflow!r} replay flat layout missing fields={missing}"
            )
        selected["layout"] = layout

    if "condition_prefixes" not in selected:
        if workflow != "t2va":
            raise ValueError(
                f"MiniMax H3 workflow={workflow!r} replay requires a shared, reproducible "
                "conditioned-prefix binder"
            )
        if not isinstance(state, LatentState):
            raise TypeError(
                "MiniMax H3 T2VA replay requires LatentState to bind empty prefixes, "
                f"received {type(state).__name__}"
            )
        validate_target_state(state)
        selected["condition_prefixes"] = {
            component: values.new_empty((values.shape[0], 0, values.shape[-1]))
            for component, values in state.components.items()
        }
    return selected


def forward_h3_adapter(adapter: Any, **kwargs: Any) -> Any:
    """Run one rollout or replay transition through the single H3 forward boundary."""
    unknown = tuple(sorted(set(kwargs) - _H3_FORWARD_FIELDS))
    if unknown:
        raise ValueError(
            f"MiniMax H3 workflow={adapter.workflow!r} forward received unsupported "
            f"arguments={unknown}"
        )
    state = kwargs.pop("state")
    times = kwargs.pop("times")
    return forward_h3_adapter_state(
        adapter,
        state=state,
        times=times,
        next_state=kwargs.pop("next_state", None),
        compute_log_prob=kwargs.pop("compute_log_prob", False),
        return_fields=kwargs.pop("return_fields", ()),
        noise_level=kwargs.pop("noise_level", None),
        forward_kwargs=kwargs,
    )


def decode_h3_adapter_latents(adapter: Any, latents: Any, **kwargs: Any) -> Any:
    """Decode one target-only state without condition-prefix concatenation."""
    if not isinstance(latents, LatentState):
        raise TypeError(
            f"MiniMax H3 workflow={adapter.workflow!r} decode expected LatentState, "
            f"received {type(latents).__name__}"
        )
    _require_b1_state(latents, adapter.workflow, "decode")
    adapter.on_load_components(["vae", "video_processor", "audio_vae"])
    return decode_h3_targets(
        adapter.pipeline,
        latents,
        kwargs["geometry"],
        output_type=kwargs.get("output_type", "pt"),
        workflow=adapter.workflow,
    )


def _single_outer_value(value: Any, field: str, workflow: str) -> Any:
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} field={field!r} requires outer B=1, "
            f"received {value!r}"
        )
    return value[0]


def _build_pinned_references(entries: Sequence[Mapping[str, Any]]) -> List[Any]:
    symbols = require_minimax_h3_support()
    references = []
    for entry in entries:
        reference_type = entry["type"]
        if reference_type == "image":
            references.append(symbols.ImageReference(image=entry["media"]))
        elif reference_type == "video":
            reference_kwargs = {"frames": entry["frames"], "fps": entry["fps"]}
            if entry.get("audio") is not None:
                reference_kwargs.update(audio=entry["audio"], sample_rate=entry["sample_rate"])
            references.append(symbols.VideoReference(**reference_kwargs))
        elif reference_type == "audio":
            references.append(
                symbols.AudioReference(audio=entry["media"], sample_rate=entry["sample_rate"])
            )
        else:
            raise ValueError(
                f"expected image/video/audio reference type, received {reference_type!r}"
            )
    return references


def _is_upstream_reference(value: Any) -> bool:
    return type(value).__name__.startswith("MiniMaxH3") and type(value).__name__.endswith(
        "Reference"
    )


def _component_times_at(
    schedules: Mapping[str, Sequence[torch.Tensor]], index: int
) -> ComponentTimes:
    timesteps = {name: values[0][index : index + 1] for name, values in schedules.items()}
    next_timesteps = {name: values[0][index + 1 : index + 2] for name, values in schedules.items()}
    sigmas = {name: values[1][index : index + 1] for name, values in schedules.items()}
    next_sigmas = {name: values[1][index + 1 : index + 2] for name, values in schedules.items()}
    return ComponentTimes(timesteps, next_timesteps, sigmas, next_sigmas)


def _resolve_trajectory_positions(
    indices: Any, num_transitions: int
) -> tuple[List[int], List[int]]:
    state_length = num_transitions + 1
    if indices == "all":
        return list(range(state_length)), list(range(num_transitions))
    if indices is None:
        return [], []
    if not isinstance(indices, list):
        raise TypeError(
            "MiniMax H3 trajectory_indices expected 'all', None, or List[int], "
            f"received {type(indices).__name__}: {indices!r}"
        )
    normalized = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(
                "MiniMax H3 trajectory_indices expected integer entries, "
                f"received {type(index).__name__}: {index!r}"
            )
        normalized_index = index + state_length if index < 0 else index
        if not 0 <= normalized_index < state_length:
            raise ValueError(
                f"MiniMax H3 trajectory_indices entry {index} normalized to "
                f"{normalized_index}, expected range [0,{state_length - 1}]"
            )
        normalized.append(normalized_index)
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "MiniMax H3 trajectory_indices expected unique normalized positions, "
            f"received raw={indices!r}, normalized={normalized!r}"
        )
    state_positions = sorted(normalized)
    return state_positions, [position for position in state_positions if position < num_transitions]


def _index_map(length: int, positions: Sequence[int]) -> torch.Tensor:
    result = torch.full((length,), -1, dtype=torch.long)
    for stored_index, position in enumerate(positions):
        result[position] = stored_index
    return result


def _append_state(storage: Dict[str, List[torch.Tensor]], state: LatentState) -> None:
    for component in _COMPONENT_ORDER:
        storage[component].append(state.components[component].detach())


def _validate_public_no_cfg_inputs(workflow: str, values: Mapping[str, Any], boundary: str) -> None:
    if "guidance_scale" in values:
        guidance_scale = values["guidance_scale"]
        if (
            isinstance(guidance_scale, bool)
            or not isinstance(guidance_scale, (int, float))
            or float(guidance_scale) != 1.0
        ):
            raise ValueError(
                f"MiniMax H3 workflow={workflow!r} public {boundary} guidance_scale "
                f"expected neutral 1.0, received {guidance_scale!r}"
            )
    if "negative_prompt" not in values:
        return
    negative_prompt = values["negative_prompt"]
    if isinstance(negative_prompt, (list, tuple)):
        if len(negative_prompt) != 1:
            raise ValueError(
                f"MiniMax H3 workflow={workflow!r} public {boundary} negative_prompt "
                f"requires outer B=1, received {negative_prompt!r}"
            )
        negative_prompt = negative_prompt[0]
    if negative_prompt is not None and (
        not isinstance(negative_prompt, str) or negative_prompt != ""
    ):
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} public {boundary} negative_prompt "
            f"expected None or empty string, received {negative_prompt!r}; model has no CFG"
        )


def _validate_fl2va_condition_images(
    values: Mapping[str, Any],
    boundary: str,
) -> tuple[Any | None, Any | None]:
    direct_first = values.get("image")
    direct_last = values.get("last_image")
    outer_images = values.get("images")
    if outer_images is None:
        outer_images = values.get("condition_images")
    if (direct_first is not None or direct_last is not None) and outer_images is not None:
        raise ValueError(
            "MiniMax H3 workflow='fl2va' cannot combine direct image/last_image "
            "arguments with grouped images"
        )
    if direct_first is not None or direct_last is not None:
        return direct_first, direct_last

    images = _single_outer_value(outer_images, "images", "fl2va")
    if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= 2:
        raise ValueError(
            f"MiniMax H3 workflow='fl2va' public {boundary} field='images' expected "
            f"one or two ordered images, received {images!r}"
        )
    outer_slots = values.get("image_slots")
    if outer_slots is None:
        slots = ("first_frame", "last_frame")[: len(images)]
    else:
        slots = _single_outer_value(outer_slots, "image_slots", "fl2va")
        if not isinstance(slots, (list, tuple)) or len(slots) != len(images):
            raise ValueError(
                f"MiniMax H3 workflow='fl2va' public {boundary} field='image_slots' "
                f"expected {len(images)} slot(s), received {slots!r}"
            )
        slots = tuple(slots)
    if len(set(slots)) != len(slots) or any(
        slot not in ("first_frame", "last_frame") for slot in slots
    ):
        raise ValueError(
            f"MiniMax H3 workflow='fl2va' public {boundary} field='image_slots' "
            f"expected unique first_frame/last_frame values, received {slots!r}"
        )
    bound = dict(zip(slots, images))
    return bound.get("first_frame"), bound.get("last_frame")


def _validate_workflow_media_inputs(
    workflow: str, values: Mapping[str, Any], boundary: str
) -> None:
    media_fields = (
        "images",
        "condition_images",
        "image",
        "last_image",
        "videos",
        "audios",
        "references",
    )
    present = {
        field for field in media_fields if field in values and _media_value_present(values[field])
    }
    if workflow == "t2va" and present:
        raise ValueError(
            f"MiniMax H3 workflow='t2va' {boundary} rejects media fields={tuple(sorted(present))}"
        )
    if workflow == "ref2va":
        generic = present & {
            "images",
            "condition_images",
            "image",
            "last_image",
            "videos",
            "audios",
        }
        if generic:
            raise ValueError(
                f"MiniMax H3 workflow='ref2va' {boundary} rejects generic media "
                f"fields={tuple(sorted(generic))}"
            )
    if workflow == "fl2va":
        invalid = present & {"videos", "audios", "references"}
        if invalid:
            raise ValueError(
                f"MiniMax H3 workflow='fl2va' {boundary} rejects fields={tuple(sorted(invalid))}"
            )


def _validate_neutral_guidance(workflow: str, forward_kwargs: Mapping[str, Any]) -> None:
    if forward_kwargs.get("negative_prompt") is not None:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} forward rejects negative_prompt; model has no CFG"
        )
    if "guidance_scale" not in forward_kwargs:
        return
    guidance_scale = forward_kwargs["guidance_scale"]
    if (
        isinstance(guidance_scale, bool)
        or not isinstance(guidance_scale, (int, float))
        or float(guidance_scale) != 1.0
    ):
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} forward guidance_scale expected neutral 1.0, "
            f"received {guidance_scale!r}"
        )


def _require_b1_state(state: LatentState, workflow: str, boundary: str) -> None:
    batch_sizes = {component: values.shape[0] for component, values in state.components.items()}
    if any(batch_size != 1 for batch_size in batch_sizes.values()):
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} {boundary} requires B=1, "
            f"received component batch sizes={batch_sizes}"
        )


def _validate_rollout_log_output(output: Any, workflow: str, index: int) -> None:
    if not isinstance(output.log_prob, torch.Tensor):
        raise TypeError(
            f"MiniMax H3 workflow={workflow!r} transition={index} log_prob expected "
            f"torch.Tensor, received {type(output.log_prob).__name__}"
        )
    component_log_probs = output.component_log_probs
    if (
        not isinstance(component_log_probs, Mapping)
        or tuple(component_log_probs) != _COMPONENT_ORDER
    ):
        received = (
            tuple(component_log_probs)
            if isinstance(component_log_probs, Mapping)
            else type(component_log_probs).__name__
        )
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} transition={index} component_log_probs "
            f"expected order {_COMPONENT_ORDER}, received {received}"
        )
    for component, value in component_log_probs.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"MiniMax H3 workflow={workflow!r} transition={index} "
                f"component_log_probs[{component!r}] expected torch.Tensor, "
                f"received {type(value).__name__}"
            )


def _build_h3_sample(
    adapter_class: type,
    *,
    prompt: str,
    prompt_embeds: torch.Tensor,
    video: Any,
    audio: torch.Tensor,
    sample_rate: int,
    trajectory: Optional[StructuredTrajectory],
    condition_images: Optional[Sequence[Any]],
    extra_kwargs: Mapping[str, Any],
    reference_manifest: Optional[str] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Any:
    sample_classes = {
        "t2va": MiniMaxH3T2VASample,
        "fl2va": MiniMaxH3FL2VASample,
        "ref2va": MiniMaxH3Ref2VASample,
    }
    workflow = getattr(adapter_class, "workflow", None)
    if workflow not in sample_classes:
        raise TypeError(f"expected one public MiniMax H3 adapter class, received {adapter_class!r}")
    sample_kwargs = {
        "prompt": prompt,
        "prompt_embeds": prompt_embeds,
        "video": video,
        "audio": audio,
        "audio_sample_rate": sample_rate,
        "height": height,
        "width": width,
        "trajectory": trajectory,
        "extra_kwargs": dict(extra_kwargs),
    }
    if workflow == "fl2va":
        sample_kwargs["condition_images"] = list(condition_images or ())
    if workflow == "ref2va":
        sample_kwargs["reference_manifest"] = reference_manifest
    return sample_classes[workflow](**sample_kwargs)


def _media_value_present(value: Any) -> bool:
    if value is None:
        return False
    return not isinstance(value, (list, tuple)) or len(value) > 0


def _normalize_layout(values: Mapping[str, Any]) -> Dict[str, Any]:
    source = values.get("layout", values)
    if not isinstance(source, Mapping):
        raise TypeError(f"MiniMax H3 layout expected Mapping, received {type(source).__name__}")
    normalized = {}
    for field in _LAYOUT_MATRIX_FIELDS:
        if field not in source:
            continue
        value = source[field]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"MiniMax H3 layout field={field!r} expected torch.Tensor, "
                f"received {type(value).__name__}"
            )
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2:
            raise ValueError(
                f"MiniMax H3 layout field={field!r} expected shape (N,D) or "
                f"collated (B=1,N,D), received {tuple(value.shape)}"
            )
        if value.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                f"MiniMax H3 layout field={field!r} expected dtype float32 or float64, "
                f"received {value.dtype}"
            )
        # HF Dataset's torch formatter downcasts cached float64 coordinates.
        normalized[field] = value.to(dtype=torch.float64)
    for field in _LAYOUT_INDEX_FIELDS:
        if field not in source:
            continue
        value = source[field]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"MiniMax H3 layout field={field!r} expected torch.Tensor, "
                f"received {type(value).__name__}"
            )
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise ValueError(
                f"MiniMax H3 layout field={field!r} expected shape (N,) or "
                f"collated (B=1,N), received {tuple(value.shape)}"
            )
        normalized[field] = value
    for field in _LAYOUT_COUNT_FIELDS:
        if field in source:
            normalized[field] = _normalize_b1_integer(source[field], f"layout.{field}")
    return normalized


def _normalize_geometry(values: Mapping[str, Any]) -> Dict[str, int]:
    source = values.get("geometry", values)
    if not isinstance(source, Mapping):
        raise TypeError(f"MiniMax H3 geometry expected Mapping, received {type(source).__name__}")
    return {
        field: _normalize_b1_integer(source[field], f"geometry.{field}")
        for field in _GEOMETRY_FIELDS
        if field in source
    }


def _normalize_b1_integer(value: Any, field: str) -> int:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                f"MiniMax H3 {field} expected collated B=1 list, received length {len(value)}"
            )
        value = value[0]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"MiniMax H3 {field} expected one scalar for B=1, "
                f"received shape {tuple(value.shape)}"
            )
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"MiniMax H3 {field} expected int, received {type(value).__name__}: {value!r}"
        )
    return value


def _decoded_video_sample(video: Any) -> Any:
    if isinstance(video, torch.Tensor) and video.shape[0] == 1:
        return video[0]
    if (
        isinstance(video, list)
        and len(video) == 1
        and (video[0] is None or isinstance(video[0], list))
    ):
        return video[0]
    return video
