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

# src/flow_factory/samples/__init__.py

from .identity import (
    LEGACY_SOURCE_ID,
    AcquisitionManifest,
    GroupKey,
    group_identity_rows,
    sample_group_key,
)
from .samples import (
    BaseSample,
    I2AVSample,
    I2ISample,
    I2VSample,
    ImageConditionSample,
    MiniMaxH3FL2VASample,
    MiniMaxH3Ref2VASample,
    MiniMaxH3T2VASample,
    OrderedReferenceConditionSample,
    Ref2AVSample,
    StackedSampleBatch,
    T2AVSample,
    T2ISample,
    T2VSample,
    V2VSample,
    VideoConditionSample,
)
from .trajectory import (
    ComponentTimes,
    ComponentTrajectory,
    IndexedTrajectoryTensor,
    LatentState,
    MultiModalStepOutput,
    NoisedState,
    ReplayStep,
    StructuredTrajectory,
    unstack_structured_trajectories,
)

__all__ = [
    # Sample classes
    "AcquisitionManifest",
    "GroupKey",
    "LEGACY_SOURCE_ID",
    "group_identity_rows",
    "sample_group_key",
    "BaseSample",
    "StackedSampleBatch",
    "ImageConditionSample",
    "VideoConditionSample",
    "T2ISample",
    "T2VSample",
    "T2AVSample",
    "I2ISample",
    "I2VSample",
    "I2AVSample",
    "V2VSample",
    "OrderedReferenceConditionSample",
    "Ref2AVSample",
    "MiniMaxH3T2VASample",
    "MiniMaxH3FL2VASample",
    "MiniMaxH3Ref2VASample",
    "ComponentTrajectory",
    "IndexedTrajectoryTensor",
    "StructuredTrajectory",
    "unstack_structured_trajectories",
    "LatentState",
    "ComponentTimes",
    "ReplayStep",
    "NoisedState",
    "MultiModalStepOutput",
]
