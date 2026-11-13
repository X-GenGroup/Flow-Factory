# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Execute frozen pinned MiniMax H3 modular blocks over shared state."""

from inspect import unwrap
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from ...samples import LatentState
from .dependency import require_minimax_h3_support

WORKFLOWS = ("t2va", "fl2va", "ref2va")
LAYOUT_FIELDS = (
    "position_ids",
    "token_tags",
    "video_indices",
    "audio_indices",
    "text_indices",
    "num_condition_video_rows",
    "num_condition_audio_rows",
)
GEOMETRY_FIELDS = (
    "height",
    "width",
    "num_frames",
    "num_latent_frames",
    "latent_height",
    "latent_width",
    "num_audio_latents",
)
ENCODE_COMMON_FIELDS = (
    "prompt_embeds",
    *GEOMETRY_FIELDS,
    *LAYOUT_FIELDS,
)
ENCODE_WORKFLOW_FIELDS = {
    "t2va": ENCODE_COMMON_FIELDS,
    "fl2va": (*ENCODE_COMMON_FIELDS, "condition_latents"),
    "ref2va": (
        *ENCODE_COMMON_FIELDS,
        "condition_latents",
        "audio_condition_latents",
    ),
}


def run_h3_blocks(
    pipeline: Any,
    blocks: Sequence[Any],
    values: Mapping[str, Any],
    *,
    requested_outputs: Sequence[str],
    workflow: str = "shared",
    input_gradients: bool = False,
) -> Dict[str, Any]:
    """Run ordered upstream blocks and select promised state fields.

    Args:
        pipeline: Materialized adapter-owned modular pipeline.
        blocks: Ordered instantiated upstream blocks.
        values: Initial state values.
        requested_outputs: Fields promised to the caller.
        workflow: Workflow identifier used in diagnostics.
        input_gradients: Execute native block bodies with input derivatives enabled.

    Returns:
        Requested values preserving upstream object identity.
    """
    symbols = require_minimax_h3_support()
    state = symbols.PipelineState(values=dict(values))
    for block in blocks:
        # Native H3 decode blocks wrap __call__ in no_grad. Reward optimization
        # executes the same tensor operations with their input graph retained.
        if input_gradients:
            result = unwrap(type(block).__call__)(block, pipeline, state)
        else:
            result = block(pipeline, state)
        if isinstance(result, tuple) and len(result) == 2:
            pipeline, state = result
        else:
            state = result
    selected = {}
    for field in requested_outputs:
        if field not in state.values:
            raise ValueError(
                f"MiniMax H3 workflow={workflow!r} promised field={field!r}, "
                f"but block sequence produced fields={tuple(state.values)}"
            )
        selected[field] = state.values[field]
    return selected


def encode_h3_workflow_inputs(
    pipeline: Any,
    values: Mapping[str, Any],
    *,
    workflow: str,
) -> Dict[str, Any]:
    """Encode one H3 workflow with pinned frozen upstream blocks.

    Args:
        pipeline: Adapter-owned modular pipeline with frozen components.
        values: Raw workflow inputs.
        workflow: One of ``t2va``, ``fl2va``, or ``ref2va``.

    Returns:
        Cache-safe encoded and layout values.
    """
    symbols = require_minimax_h3_support()
    _validate_workflow(workflow)
    block_types = {
        "t2va": (
            symbols.TextEncoderStep,
            symbols.NoKeyframeAnchorsStep,
            symbols.PrepareLayoutStep,
        ),
        "fl2va": (
            symbols.ResizeStep,
            symbols.FL2VATextEncoderStep,
            symbols.KeyframeEncoderStep,
            symbols.PrepareLayoutStep,
        ),
        "ref2va": (
            symbols.RefSetupStep,
            symbols.Ref2VATextEncoderStep,
            symbols.ReferenceEncoderStep,
            symbols.RefPrepareLayoutStep,
        ),
    }[workflow]
    return run_h3_blocks(
        pipeline,
        [block_type() for block_type in block_types],
        values,
        requested_outputs=ENCODE_WORKFLOW_FIELDS[workflow],
        workflow=workflow,
    )


def prepare_h3_rollout_state(
    pipeline: Any,
    cached_values: Mapping[str, Any],
    *,
    workflow: str,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    audio_latents: Optional[torch.Tensor] = None,
) -> Tuple[LatentState, Dict[str, torch.Tensor]]:
    """Prepare target rows and immutable condition prefixes with pinned blocks.

    Args:
        pipeline: Adapter-owned modular pipeline.
        cached_values: Cached encoding and layout fields.
        workflow: H3 workflow identifier.
        generator: Shared ordered random generator.
        latents: Optional pre-generated video noise.
        audio_latents: Optional pre-generated audio noise.

    Returns:
        Target-only state and immutable video/audio prefixes.
    """
    _validate_workflow(workflow)
    symbols = require_minimax_h3_support()
    values = dict(cached_values)
    values.update(generator=generator, latents=latents, audio_latents=audio_latents)
    video_count = _condition_count(cached_values, workflow, "video")
    audio_count = _condition_count(cached_values, workflow, "audio")
    prefixes: Dict[str, torch.Tensor]
    if workflow == "t2va":
        if video_count or audio_count:
            raise ValueError(
                "MiniMax H3 workflow='t2va' requires zero condition rows, "
                f"received video={video_count}, audio={audio_count}"
            )
        prefixes = {}
    else:
        prefixes = prepare_h3_condition_prefixes(
            pipeline,
            cached_values,
            workflow=workflow,
            generator=generator,
        )
        values.update(
            condition_rows=prefixes["video"][0],
            num_condition_video_rows=video_count,
            num_condition_audio_rows=audio_count,
        )
        if workflow == "ref2va":
            values["audio_condition_latents"] = [] if audio_count == 0 else [prefixes["audio"][0]]

    block_types = [symbols.PrepareLatentsStep]
    if workflow == "fl2va":
        block_types.append(symbols.FL2VAPrepareLatentsStep)
    elif workflow == "ref2va":
        block_types.append(symbols.Ref2VAPrepareLatentsStep)
    output = run_h3_blocks(
        pipeline,
        [block_type() for block_type in block_types],
        values,
        requested_outputs=("latents", "audio_latents"),
        workflow=workflow,
    )
    video_rows = _as_batched_rows(output["latents"], 96, workflow, "latents")
    audio_rows = _as_batched_rows(output["audio_latents"], 32, workflow, "audio_latents")
    _validate_rollout_row_counts(
        cached_values,
        workflow,
        "video",
        full_row_count=video_rows.shape[1],
        condition_row_count=video_count,
    )
    _validate_rollout_row_counts(
        cached_values,
        workflow,
        "audio",
        full_row_count=audio_rows.shape[1],
        condition_row_count=audio_count,
    )
    prefixes = {
        "video": video_rows[:, :video_count],
        "audio": audio_rows[:, :audio_count],
    }
    targets = LatentState(
        {
            "video": video_rows[:, video_count:],
            "audio": audio_rows[:, audio_count:],
        }
    )
    return targets, prefixes


def prepare_h3_condition_prefixes(
    pipeline: Any,
    cached_values: Mapping[str, Any],
    *,
    workflow: str,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Realize one immutable H3 condition prefix from cached clean latents.

    The helper owns the only runtime condition-noise draw. Online rollout calls it
    before generated video/audio noise, while offline SFT and DPO call it once per
    batch and reuse the returned tensors for every target and policy/reference arm.

    Args:
        pipeline: Adapter-owned modular pipeline with the canonical scheduler.
        cached_values: Cached layout and clean visual/audio condition latents.
        workflow: Conditioned H3 workflow, ``fl2va`` or ``ref2va``.
        generator: Optional request generator consumed in packed visual order.

    Returns:
        Batched immutable video/audio prefix rows in model packing order.
    """
    _validate_workflow(workflow)
    if workflow == "t2va":
        raise ValueError("MiniMax H3 T2VA has no runtime condition prefix to realize")

    symbols = require_minimax_h3_support()
    video_count = _condition_count(cached_values, workflow, "video")
    audio_count = _condition_count(cached_values, workflow, "audio")
    condition_latents = _normalize_condition_tensor_list(
        cached_values.get("condition_latents"),
        workflow=workflow,
        field="condition_latents",
        item_rank=5,
    )
    values = dict(cached_values)
    values.update(
        generator=generator,
        num_condition_video_rows=video_count,
        condition_latents=condition_latents,
    )
    output = run_h3_blocks(
        pipeline,
        [symbols.PrepareConditionLatentsStep()],
        values,
        requested_outputs=("condition_rows",),
        workflow=workflow,
    )
    video_rows = _as_batched_rows(
        output["condition_rows"],
        96,
        workflow,
        "condition_rows",
    )
    _validate_condition_prefix_rows(
        cached_values,
        workflow,
        "video",
        prefix_rows=video_rows.shape[1],
        condition_row_count=video_count,
    )

    if workflow == "fl2va":
        if audio_count:
            raise ValueError(
                "MiniMax H3 FL2VA requires zero audio condition rows, " f"received {audio_count}"
            )
        audio_rows = video_rows.new_empty((1, 0, 32))
    else:
        audio_latents = _normalize_condition_tensor_list(
            cached_values.get("audio_condition_latents", []),
            workflow=workflow,
            field="audio_condition_latents",
            item_rank=2,
        )
        if audio_latents:
            audio_rows = _as_batched_rows(
                torch.cat([rows.to(video_rows.device) for rows in audio_latents], dim=0),
                32,
                workflow,
                "audio_condition_latents",
            )
        else:
            audio_rows = video_rows.new_empty((1, 0, 32))
        _validate_condition_prefix_rows(
            cached_values,
            workflow,
            "audio",
            prefix_rows=audio_rows.shape[1],
            condition_row_count=audio_count,
        )
    return {"video": video_rows, "audio": audio_rows}


def _validate_workflow(workflow: str) -> None:
    if workflow not in WORKFLOWS:
        raise ValueError(f"expected MiniMax H3 workflow in {WORKFLOWS}, received {workflow!r}")


def _condition_count(values: Mapping[str, Any], workflow: str, component: str) -> int:
    field = f"num_condition_{component}_rows"
    count = values.get(field, 0)
    if isinstance(count, (list, tuple)):
        if len(count) != 1:
            raise ValueError(
                f"MiniMax H3 workflow={workflow!r} field={field!r} expected B=1, "
                f"received length={len(count)}"
            )
        count = count[0]
    if isinstance(count, torch.Tensor):
        if count.numel() != 1:
            raise ValueError(
                f"MiniMax H3 workflow={workflow!r} field={field!r} expected one scalar, "
                f"received shape={tuple(count.shape)}"
            )
        count = count.item()
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} field={field!r} expected non-negative int, "
            f"received {count!r}"
        )
    return count


def _validate_rollout_row_counts(
    values: Mapping[str, Any],
    workflow: str,
    component: str,
    *,
    full_row_count: int,
    condition_row_count: int,
) -> None:
    index_field = f"{component}_indices"
    indices = _normalize_layout_indices(values.get(index_field), workflow, index_field)
    expected_full_rows = indices.numel()
    if full_row_count != expected_full_rows:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} component={component!r} "
            f"field='full_row_count' actual={full_row_count} expected={expected_full_rows}"
        )
    actual_target_rows = max(full_row_count - condition_row_count, 0)
    expected_target_rows = expected_full_rows - condition_row_count
    if actual_target_rows != expected_target_rows or expected_target_rows <= 0:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} component={component!r} "
            f"field='target_row_count' actual={actual_target_rows} "
            f"expected={expected_target_rows}"
        )


def _validate_condition_prefix_rows(
    values: Mapping[str, Any],
    workflow: str,
    component: str,
    *,
    prefix_rows: int,
    condition_row_count: int,
) -> None:
    index_field = f"{component}_indices"
    indices = _normalize_layout_indices(values.get(index_field), workflow, index_field)
    if prefix_rows != condition_row_count:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} component={component!r} "
            f"field='condition_row_count' actual={prefix_rows} expected={condition_row_count}"
        )
    if condition_row_count > indices.numel():
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} component={component!r} "
            f"field='condition_row_count' actual={condition_row_count} "
            f"expected_at_most={indices.numel()}"
        )


def _normalize_layout_indices(
    value: Any,
    workflow: str,
    field: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} field={field!r} expected Tensor, "
            f"received {type(value).__name__}"
        )
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} field={field!r} expected shape (N,) or "
            f"collated (B=1,N), received {tuple(value.shape)}"
        )
    return value


def _normalize_condition_tensor_list(
    value: Any,
    *,
    workflow: str,
    field: str,
    item_rank: int,
) -> list[torch.Tensor]:
    """Normalize Arrow/HF B=1 list and stacked-tensor representations."""
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])

    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], torch.Tensor):
        first = value[0]
        if first.ndim == item_rank + 1:
            value = first
    if isinstance(value, torch.Tensor):
        if value.ndim == item_rank:
            tensors = [value]
        else:
            if value.ndim == item_rank + 2 and value.shape[0] == 1:
                value = value[0]
            if value.ndim != item_rank + 1:
                raise ValueError(
                    f"MiniMax H3 workflow={workflow!r} field={field!r} expected one "
                    f"rank-{item_rank} tensor or a B=1/stacked container, received "
                    f"shape={tuple(value.shape)}"
                )
            tensors = list(torch.unbind(value, dim=0))
    elif isinstance(value, list):
        tensors = value
    else:
        raise TypeError(
            f"MiniMax H3 workflow={workflow!r} field={field!r} expected tensor/list, "
            f"received {type(value).__name__}"
        )

    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != item_rank:
            raise ValueError(
                f"MiniMax H3 workflow={workflow!r} field={field!r}[{index}] expected "
                f"rank-{item_rank} Tensor, received "
                f"{type(tensor).__name__}/{getattr(tensor, 'shape', None)}"
            )
    return tensors


def _as_batched_rows(rows: torch.Tensor, width: int, workflow: str, field: str) -> torch.Tensor:
    if not isinstance(rows, torch.Tensor):
        raise TypeError(
            f"MiniMax H3 workflow={workflow!r} field={field!r} expected Tensor, "
            f"received {type(rows).__name__}"
        )
    if rows.ndim == 2:
        rows = rows.unsqueeze(0)
    if rows.ndim != 3 or rows.shape[0] != 1 or rows.shape[-1] != width:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} field={field!r} expected shape (B=1,N,{width}), "
            f"received {tuple(rows.shape)}"
        )
    return rows
