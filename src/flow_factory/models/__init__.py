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

# src/flow_factory/models/__init__.py
"""Model adapters for diffusion and flow-matching architectures.

Provides the adapter base class, shared adapter contracts, and a registry-based loader.
"""

from .abc import BaseAdapter
from .condition_state import (
    ConditionStatePreparer,
    PreparedConditionState,
    validate_condition_preparer_required_components,
)
from .latent_geometry import LatentAxes, LatentLayout, infer_latent_axes
from .loader import load_model
from .model_bundle import ModelBundle, RoutedComponentProxy
from .output_state import (
    DecodedMediaBatch,
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
    OutputStateCodec,
    validate_codec_required_components,
    validate_encoded_output_state,
    validate_output_candidate_batch,
)
from .registry import (
    get_model_adapter_class,
    list_registered_models,
)

__all__ = [
    # Core classes
    "BaseAdapter",
    # Runtime input-condition preparation
    "ConditionStatePreparer",
    "PreparedConditionState",
    "validate_condition_preparer_required_components",
    # Offline target encoding
    "DecodedMediaBatch",
    "EncodedOutputState",
    "GeometrySignature",
    "MediaGeometrySignature",
    "OutputStateCodec",
    "validate_codec_required_components",
    "validate_encoded_output_state",
    "validate_output_candidate_batch",
    # Latent geometry
    "LatentAxes",
    "LatentLayout",
    "infer_latent_axes",
    # Single-root distributed wrapping
    "ModelBundle",
    "RoutedComponentProxy",
    # Registry functions
    "get_model_adapter_class",
    "list_registered_models",
    # Factory function
    "load_model",
]
