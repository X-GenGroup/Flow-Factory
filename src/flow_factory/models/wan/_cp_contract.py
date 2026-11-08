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

"""Wan-owned metadata admission for native scalar and per-token CP layouts."""

from collections.abc import Mapping
from enum import Enum
from typing import Any


class WanConditionLayout(str, Enum):
    DENSE = "dense"
    FIRST_FRAME_CLIP = "first_frame_clip"
    FIRST_LAST_CLIP = "first_last_clip"
    CHANNEL_CONDITION = "channel_condition"
    FIRST_FRAME_BLEND = "first_frame_blend"


def wan_context_parallel_components(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate a native schedule before loading any denoiser weights."""
    secondary = config.get("transformer_2")
    dual = secondary is not None and secondary not in ([None, None], (None, None))
    expanded = config.get("expand_timesteps", False)
    boundary = config.get("boundary_ratio")
    if not isinstance(expanded, bool):
        raise ValueError(f"Wan CP expand_timesteps must be bool, received {expanded!r}")
    if boundary is not None and (
        isinstance(boundary, bool)
        or not isinstance(boundary, (int, float))
        or not 0 <= boundary <= 1
    ):
        raise ValueError(f"Wan CP boundary_ratio must be in [0, 1], received {boundary!r}")
    if boundary not in (None, 0) and not dual:
        raise ValueError(f"Wan CP boundary_ratio={boundary} requires transformer_2")
    if expanded and (dual or boundary not in (None, 0)):
        raise ValueError("Wan CP expanded timesteps require a single TI2V transformer")
    return ("transformer", "transformer_2") if dual else ("transformer",)


def wan_context_parallel_layout(
    config: Mapping[str, Any],
    degree: int,
    tp_size: int,
    *,
    image_conditioned: bool,
    expanded: bool | None = None,
) -> WanConditionLayout:
    """Identify supported native conditioning without inferring a checkpoint name."""
    heads = config["num_attention_heads"]
    if heads % (degree * tp_size):
        raise ValueError(
            f"Wan CP requires num_attention_heads={heads} divisible by "
            f"cp_size={degree} * tp_size={tp_size}"
        )
    if config.get("num_layers", 0) < 1:
        raise ValueError("Wan CP requires native transformer blocks")
    channels, output = config["in_channels"], config.get("out_channels")
    if not output or channels <= 0:
        raise ValueError(
            f"Wan CP requires positive input/output channels, received {channels}/{output}"
        )
    image_dim = config.get("image_dim")
    added = config.get("added_kv_proj_dim")
    positions = config.get("pos_embed_seq_len")
    if image_dim is not None:
        if not image_conditioned or not image_dim or added != heads * config["attention_head_dim"]:
            raise ValueError("Wan CP requires compatible CLIP image KV projections")
        if positions not in (None, 514):
            raise ValueError(f"Wan CP requires 514 FLF2V image positions, received {positions}")
        if channels != 2 * output + 4:
            raise ValueError(
                "Wan I2V CP requires in_channels=2*out_channels+4 for CLIP conditioning"
            )
        layout = (
            WanConditionLayout.FIRST_FRAME_CLIP
            if positions is None
            else WanConditionLayout.FIRST_LAST_CLIP
        )
    else:
        if added is not None or positions is not None:
            raise ValueError(
                "Wan CP requires CLIP image KV for added projections or image positions"
            )
        if image_conditioned and channels == 2 * output + 4:
            layout = WanConditionLayout.CHANNEL_CONDITION
        elif channels == output:
            layout = (
                WanConditionLayout.FIRST_FRAME_BLEND
                if image_conditioned
                else WanConditionLayout.DENSE
            )
        else:
            raise ValueError(
                f"Wan CP unsupported channel geometry: in_channels={channels}, out_channels={output}"
            )
    if expanded is not None and image_conditioned:
        if expanded != (layout is WanConditionLayout.FIRST_FRAME_BLEND):
            raise ValueError(
                f"Wan I2V CP expand_timesteps={expanded} disagrees with conditioning layout={layout.value}"
            )
    if expanded and tuple(config["patch_size"]) != (1, 2, 2):
        raise ValueError("Wan CP expanded timesteps require patch_size=(1, 2, 2)")
    return layout


def validate_wan_context_parallel_configs(
    configs: Mapping[str, Mapping[str, Any]],
    pipeline_config: Mapping[str, Any],
    degree: int,
    tp_size: int,
    *,
    image_conditioned: bool,
) -> None:
    """Require both scheduled denoisers to consume identical condition geometry."""
    names = wan_context_parallel_components(pipeline_config)
    fields = (
        "patch_size",
        "in_channels",
        "out_channels",
        "text_dim",
        "image_dim",
        "added_kv_proj_dim",
        "pos_embed_seq_len",
        "num_attention_heads",
        "attention_head_dim",
    )
    for name in names:
        wan_context_parallel_layout(
            configs[name],
            degree,
            tp_size,
            image_conditioned=image_conditioned,
            expanded=pipeline_config.get("expand_timesteps", False),
        )
        if name != names[0]:
            mismatched = [
                field
                for field in fields
                if configs[name].get(field) != configs[names[0]].get(field)
            ]
            if mismatched:
                raise ValueError(f"Wan CP denoiser geometry differs for {name}: {mismatched}")
