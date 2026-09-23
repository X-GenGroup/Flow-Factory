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

"""Dependency-neutral framework contracts."""

from .execution import (
    OFFLINE_EXECUTION_CONTRACT,
    ONLINE_EXECUTION_CONTRACT,
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    AcquisitionMode,
    ExecutionContract,
    FeedbackMode,
)
from .model_condition import (
    FORWARD_STATE_BOUNDARY_KEYS,
    FORWARD_STATE_OWNED_KEYS,
    NON_MODEL_CONDITION_KEYS,
    OFFLINE_PROVENANCE_KEYS,
    ROLLOUT_STORAGE_KEYS,
    TRAINER_METADATA_KEYS,
)
from .pipeline_io import (
    BatchCapability,
    DecodedMediaLike,
    GeometrySource,
    InputMediaBinding,
    InputMediaLike,
    InputMediaOrder,
    InputMediaRule,
    InputMediaSpec,
    MediaFormat,
    MediaType,
    ModelInputLike,
    NegativePromptPolicy,
    OutputMediaLike,
    OutputMediaSequence,
    PipelineIOContract,
    RateRequirement,
    resolve_pipeline_input_media_slots,
    validate_pipeline_model_input,
    validate_pipeline_output_candidate,
)
from .reward_overlap import (
    COUPLED_REWARD_OPTIMIZATION_OVERLAP,
    GROUP_RELATIVE_REWARD_OPTIMIZATION_OVERLAP,
    NO_REWARD_OPTIMIZATION_OVERLAP,
    RewardOptimizationOverlapContract,
    RewardTileScheduling,
)

__all__ = [
    "AcquisitionMode",
    "BatchCapability",
    "COUPLED_REWARD_OPTIMIZATION_OVERLAP",
    "GROUP_RELATIVE_REWARD_OPTIMIZATION_OVERLAP",
    "DecodedMediaLike",
    "ExecutionContract",
    "FeedbackMode",
    "FORWARD_STATE_BOUNDARY_KEYS",
    "FORWARD_STATE_OWNED_KEYS",
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
    "NON_MODEL_CONDITION_KEYS",
    "NO_REWARD_OPTIMIZATION_OVERLAP",
    "OFFLINE_EXECUTION_CONTRACT",
    "OFFLINE_PROVENANCE_KEYS",
    "ONLINE_EXECUTION_CONTRACT",
    "ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT",
    "OutputMediaLike",
    "OutputMediaSequence",
    "PipelineIOContract",
    "RateRequirement",
    "RewardOptimizationOverlapContract",
    "RewardTileScheduling",
    "resolve_pipeline_input_media_slots",
    "ROLLOUT_STORAGE_KEYS",
    "TRAINER_METADATA_KEYS",
    "validate_pipeline_model_input",
    "validate_pipeline_output_candidate",
]
