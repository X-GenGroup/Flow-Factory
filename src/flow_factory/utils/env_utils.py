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

# src/flow_factory/utils/env_utils.py
"""Shared environment variable detection for distributed training.

Provides a unified mapping table and lookup utility used by both the CLI
launcher (``cli.py``) and the runtime config reconciliation
(``reconcile_config``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from accelerate import Accelerator

    from ..hparams import Arguments


# ========================================================================
# Multi-node environment variable mapping table.
# Supports multiple cluster scheduler naming conventions (ordered by priority).
# ========================================================================
ENV_VAR_MAPPINGS: dict[str, list[str]] = {
    "master_ip": ["MASTER_ADDR", "MASTER_IP", "CHIEF_IP"],
    "master_port": ["MASTER_PORT"],
    "machine_rank": ["MACHINE_RANK", "NODE_RANK", "INDEX"],
    "num_machines": ["NUM_MACHINES", "NUM_NODES", "HOST_NUM", "NNODES"],
    "gpus_per_node": ["GPUS_PER_NODE", "HOST_GPU_NUM"],
}


def env_lookup(key: str) -> str | None:
    """Look up a value from environment variable mapping by priority."""
    for env_name in ENV_VAR_MAPPINGS.get(key, []):
        val = os.environ.get(env_name)
        if val is not None and val != "":
            return val
    return None


def reconcile_config(config: "Arguments", accelerator: "Accelerator") -> None:
    """Reconcile config with actual runtime distributed state.

    Call ONCE in the trainer loader after accelerator creation, before the
    adapter, trainer, and logger are built.  Mutates ``config`` in-place so all
    downstream consumers (logger, checkpointing, etc.) see consistent values
    without needing direct access to the accelerator.

    Args:
        config: The top-level ``Arguments`` config (mutated in place).
        accelerator: The Accelerate instance (used read-only, not stored).
    """
    # Authoritative values from accelerator
    config.num_processes = accelerator.num_processes
    config.mixed_precision = accelerator.mixed_precision  # type: ignore[assignment]
    config.process_index = accelerator.process_index
    config.local_process_index = accelerator.local_process_index

    # Environment variable lookups (same table as cli.py)
    port_str = env_lookup("master_port")
    if port_str is not None:
        try:
            config.main_process_port = int(port_str)
        except ValueError:
            pass

    ip = env_lookup("master_ip")
    if ip is not None:
        config.main_process_ip = ip

    num_machines_str = env_lookup("num_machines")
    if num_machines_str is not None:
        try:
            num_machines = int(num_machines_str)
            config.num_machines = num_machines
            if num_machines > 0:
                config.gpus_per_node = accelerator.num_processes // num_machines
        except ValueError:
            pass

    rank_str = env_lookup("machine_rank")
    if rank_str is not None:
        try:
            config.machine_rank = int(rank_str)
        except ValueError:
            pass
