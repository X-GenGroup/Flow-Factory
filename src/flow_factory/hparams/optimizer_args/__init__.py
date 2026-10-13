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

"""Per-variant optimizer configuration."""

from ._base import MultiOptimizerArguments, OptimizerArguments, resolve_role_optimizer_args
from ._registry import (
    build_optimizer_args,
    get_optimizer_args_class,
    register_optimizer_args,
)
from .adamw import AdamWOptimizerArguments
from .muon import MuonOptimizerArguments

__all__ = [
    "AdamWOptimizerArguments",
    "MultiOptimizerArguments",
    "MuonOptimizerArguments",
    "OptimizerArguments",
    "build_optimizer_args",
    "get_optimizer_args_class",
    "register_optimizer_args",
    "resolve_role_optimizer_args",
]
