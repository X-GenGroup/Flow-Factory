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

"""Activation-storage policy shared by pairwise policy objectives."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

from torch.autograd.graph import save_on_cpu


def pairwise_policy_activation_context(trainer: Any) -> AbstractContextManager[None]:
    """Select activation storage for one trainable preference arm.

    Args:
        trainer: Pairwise trainer that owns the adapter and Accelerator runtime.

    Returns:
        A context that offloads saved tensors when the adapter and backend require it.
    """
    if (
        not trainer.adapter.requires_pairwise_policy_activation_offload
        or trainer.adapter._is_fsdp2()
        or trainer.accelerator.device.type != "cuda"
    ):
        return nullcontext()
    return save_on_cpu(pin_memory=True, device_type="cuda")


__all__ = ["pairwise_policy_activation_context"]
