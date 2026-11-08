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

"""Wan-owned native context-parallel layout and TP/SP boundary composition."""

from __future__ import annotations

from typing import Any

from torch import nn

from ..parallelism.context import ContextParallelBoundary, ContextParallelLayout
from ..parallelism.hooks import require_native_hook


class WanContextParallelLayout(ContextParallelLayout):
    """Stable identities for qualified Wan context-parallel layouts."""

    DENSE_VIDEO = "wan_dense_video"


def compose_wan_sequence_boundaries(module: nn.Module) -> tuple[ContextParallelBoundary, ...]:
    """Place native CP outside Torch SP while retaining native rotary/attention ownership."""
    from diffusers.hooks.context_parallel import ContextParallelGatherHook, ContextParallelSplitHook

    error = "Wan TP/SP+CP requires the qualified native input/output hooks"
    split = require_native_hook(
        module.blocks[0], "cp_input---blocks.0", ContextParallelSplitHook, error=error
    )
    gather = require_native_hook(
        module.proj_out, "cp_output---proj_out", ContextParallelGatherHook, error=error
    )
    return split.move_input(), gather.move_output()
