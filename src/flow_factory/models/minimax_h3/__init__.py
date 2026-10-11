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

"""MiniMax H3 workflow adapters and core helpers."""

from ._common import (
    MINIMAX_H3_COMPONENT_ORDER,
    apply_forward_process_noise,
    build_component_step_output,
    build_structured_trajectories,
    build_training_component_times,
    combine_component_log_probs,
    draw_forward_process_noise,
    framework_sigma_to_model_time,
    inverse_shift_sigma,
    model_time_to_framework_sigma,
    pack_audio_latents,
    pack_video_latents,
    shift_sigma,
    unpack_audio_latents,
    unpack_video_latents,
    validate_target_state,
)
from .adapters import (
    MiniMaxH3FL2VAAdapter,
    MiniMaxH3Ref2VAAdapter,
    MiniMaxH3T2VAAdapter,
)
from .blocks import (
    encode_h3_workflow_inputs,
    prepare_h3_condition_prefixes,
    prepare_h3_rollout_state,
    run_h3_blocks,
)
from .decoding import decode_h3_targets
from .denoise import forward_h3_state, run_h3_joint_transformer, step_h3_components
from .dependency import require_minimax_h3_support
from .layout import H3SchedulePlan, build_h3_schedule_plan, build_row_timesteps

__all__ = [
    "H3SchedulePlan",
    "MINIMAX_H3_COMPONENT_ORDER",
    "MiniMaxH3FL2VAAdapter",
    "MiniMaxH3Ref2VAAdapter",
    "MiniMaxH3T2VAAdapter",
    "apply_forward_process_noise",
    "build_component_step_output",
    "build_structured_trajectories",
    "build_training_component_times",
    "build_h3_schedule_plan",
    "build_row_timesteps",
    "combine_component_log_probs",
    "draw_forward_process_noise",
    "decode_h3_targets",
    "encode_h3_workflow_inputs",
    "forward_h3_state",
    "framework_sigma_to_model_time",
    "inverse_shift_sigma",
    "model_time_to_framework_sigma",
    "pack_audio_latents",
    "pack_video_latents",
    "prepare_h3_condition_prefixes",
    "prepare_h3_rollout_state",
    "require_minimax_h3_support",
    "run_h3_blocks",
    "run_h3_joint_transformer",
    "shift_sigma",
    "step_h3_components",
    "unpack_audio_latents",
    "unpack_video_latents",
    "validate_target_state",
]
