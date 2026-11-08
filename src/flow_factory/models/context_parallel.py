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


"""Compatibility imports; implementation lives in ``models.parallelism.context``."""

from .parallelism.context import ContextParallelBoundary as ContextParallelBoundary
from .parallelism.context import ContextParallelCapability as ContextParallelCapability
from .parallelism.context import ContextParallelLayout as ContextParallelLayout
from .parallelism.context import PreparedContextParallel as PreparedContextParallel
from .parallelism.context import context_parallel_mesh as context_parallel_mesh
from .parallelism.context import (
    validate_prepared_context_parallel as validate_prepared_context_parallel,
)
