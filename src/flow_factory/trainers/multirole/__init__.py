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

"""Prepared-backend and checkpoint contracts for multi-role trainers."""

from .backend import (
    MultiRoleBackendValidationMixin,
    configure_checkpointing_backend_plan,
    configure_deepspeed_micro_batch_size,
    validate_optimizer_backend_plan,
    validate_supported_distributed_plan,
)
from .checkpointing import MULTIROLE_RUNTIME_CHILD_NAME, MultiRoleCheckpointingMixin

__all__ = [
    "MultiRoleBackendValidationMixin",
    "MultiRoleCheckpointingMixin",
    "MULTIROLE_RUNTIME_CHILD_NAME",
    "configure_checkpointing_backend_plan",
    "configure_deepspeed_micro_batch_size",
    "validate_optimizer_backend_plan",
    "validate_supported_distributed_plan",
]
