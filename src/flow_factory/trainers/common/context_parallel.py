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

"""Fail-fast gates for the first complete context-parallel training composition."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

from packaging.version import Version

from ...acceleration.attention_backend import AttentionBackendAccelerator
from ...acceleration.registry import get_accelerator_class
from ...hparams import Arguments
from ...models.context_parallel import ContextParallelCapability
from ...utils.dist import get_world_size, parallel_topology


def validate_context_parallel_config(
    config: Arguments, adapter_cls: type, trainer_cls: type
) -> None:
    """Validate declared layout, objective and lifecycle before weights are loaded."""
    if config.parallelism_args.effective_context_parallel_size == 1:
        return
    from ..offline.sft import SFTTrainer

    topology = config.parallelism_args.resolve_topology(get_world_size())
    if topology.dp_replicate_size != 1 or topology.dp_shard_size < 2:
        raise ValueError(
            "context parallelism requires dp_replicate_size=1 and dp_shard_size>=2; "
            f"received {topology.dimensions}"
        )
    capability = getattr(adapter_cls, "context_parallel_capability", None)
    if not isinstance(capability, ContextParallelCapability):
        raise ValueError(f"{adapter_cls.__name__} does not declare a context-parallel capability")
    if topology.tp_size > 1 and not capability.tensor_parallel:
        raise ValueError(f"{adapter_cls.__name__} does not declare TP+CP composition")
    required_precision = capability.tensor_parallel_precision_policy
    if (
        topology.tp_size > 1
        and required_precision is not None
        and config.model_args.compute_precision_policy != required_precision
    ):
        raise ValueError(
            f"TP+CP SFT requires model.compute_precision_policy={required_precision}; "
            "native BF16 cross-attention does not satisfy the qualified update bound"
        )
    if trainer_cls is not SFTTrainer:
        raise ValueError(
            f"context parallelism currently supports SFTTrainer, received {trainer_cls.__name__}"
        )
    if config.mixed_precision != "bf16":
        raise ValueError(
            f"context parallel SFT requires mixed_precision=bf16, received {config.mixed_precision!r}"
        )
    if config.training_args.ema_decay != 0 or config.eval_args.eval_freq != 0:
        raise ValueError("context parallel SFT requires ema_decay=0 and eval.eval_freq=0")
    if config.model_args.resume_type == "state" or not config.log_args.save_model_only:
        raise ValueError(
            "context parallel SFT requires log.save_model_only=true; exact state resume is not yet supported"
        )
    if config.model_args.target_components != ["transformer"]:
        raise ValueError("context parallel SFT requires model.target_components=[transformer]")
    if config.training_args.per_device_batch_size != 1:
        raise ValueError("context parallel Wan requires train.per_device_batch_size=1")
    if any(args.optimizer != "adamw" for args in config.optimizer_args):
        raise ValueError("context parallel SFT currently requires AdamW optimizers")
    shared = config.acceleration_args.shared
    if (
        len(shared) != 1
        or get_accelerator_class(shared[0].name) is not AttentionBackendAccelerator
        or shared[0].params != {"backend": "native"}
        or config.acceleration_args.rollout
    ):
        raise ValueError(
            "context parallel SFT requires exactly one shared attention_backend with "
            "params={backend: native}, and no rollout accelerators"
        )
    for package, expected in (("torch", (2, 10)), ("accelerate", (1, 15)), ("diffusers", (0, 41))):
        installed = Version(version(package))
        if installed.release[:2] != expected or installed.is_prerelease or installed.is_devrelease:
            raise ValueError(
                f"context parallelism requires the qualified {package} {expected[0]}.{expected[1]}.x "
                f"release line, received {installed}"
            )


def validate_context_parallel_backend(accelerator: Any, configured_degree: int) -> None:
    """Require one explicit CP degree and the measured FSDP2 reduction contract."""
    topology = parallel_topology(accelerator)
    if topology.cp_size != configured_degree:
        raise ValueError(
            f"runtime cp_size={topology.cp_size} differs from explicit parallelism.cp_size={configured_degree}; "
            "remove the launcher's CP override"
        )
    if configured_degree == 1:
        return
    if getattr(accelerator, "is_fsdp2", False) is not True:
        raise ValueError("context parallelism requires FSDP2 before loading model weights")
    plugin = accelerator.state.fsdp_plugin
    if getattr(plugin, "cpu_ram_efficient_loading", False):
        raise ValueError(
            "context parallelism currently requires fsdp_cpu_ram_efficient_loading=false"
        )
    if accelerator.device.type != "cuda":
        raise ValueError("context parallelism requires CUDA/NCCL")
