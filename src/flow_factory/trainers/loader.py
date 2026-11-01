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

# src/flow_factory/trainers/loader.py
"""Registry-driven trainer factory."""

import logging
import os
from typing import Any, Optional

from accelerate import Accelerator as _UpstreamAccelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.parallelism_config import ParallelismConfig
from accelerate.utils import ProjectConfiguration, set_seed

from ..contracts.execution import ExecutionContract
from ..hparams import Arguments, get_training_args_class
from ..loading.backend import configure_backend_loading
from ..models.loader import load_model
from ..models.registry import get_model_adapter_class
from ..utils.dist import get_world_size
from ..utils.env_utils import reconcile_config
from ..utils.logger_utils import setup_logger
from .abc import BaseTrainer
from .multirole import (
    configure_checkpointing_backend_plan,
    validate_optimizer_backend_plan,
    validate_supported_distributed_plan,
    validate_tensor_parallel_backend_plan,
)
from .registry import get_trainer_class, list_registered_trainers

logger = setup_logger(__name__)


class Accelerator(_UpstreamAccelerator):
    """An Accelerator that leaves tensor parallelism to Flow-Factory.

    Accelerate's ``prepare`` runs ``_prepare_tp`` whenever ``tp_size > 1``, and
    that path unconditionally imports
    ``transformers.integrations.tensor_parallel.ReplicateParallel`` -- a symbol
    the ``transformers`` 5.x line removed, while ``pyproject.toml`` supports both
    4.x and 5.x. It also shards nothing by itself; its only job is to complete a
    partially-sharded model, which
    :func:`flow_factory.models.tensor_parallel.apply_tensor_parallel` has already
    done by the time ``prepare`` runs.

    The parallelism config is still handed to the base class, because it is what
    builds the ``DeviceMesh`` that both the data-parallel geometry and
    ``fully_shard`` read.

    This deliberately shadows the imported name so that the construction site
    below, and the tests that patch ``loader.Accelerator`` to intercept it,
    both keep reading as the one place an Accelerator is built.
    """

    def _prepare_tp(self, *args: Any) -> Any:
        return args


def _requires_ddp_unused_parameter_detection(
    config: Arguments,
    adapter_cls: type,
) -> bool:
    """Resolve DDP unused-parameter detection before Accelerator construction.

    A run that trains several variants leaves some of them ungraded on any given
    backward, which DDP's static buckets cannot express. The decision has to be made
    here because it is a constructor argument to the Accelerator.
    """
    training_args_cls = get_training_args_class(config.training_args.trainer_type)
    required_roles = getattr(config.training_args, "required_trainable_roles", None)
    if required_roles is None:
        required_roles = getattr(training_args_cls, "required_trainable_roles", ())
    return len(tuple(required_roles)) > 1 or adapter_cls.ddp_find_unused_parameters


def _build_parallelism_config(config: Arguments) -> Optional[ParallelismConfig]:
    """Translate the configured parallelism block into an Accelerate mesh plan.

    An unconfigured block returns ``None``, which leaves Accelerate deriving one
    data-parallel dimension from the launched world -- exactly what no block
    means today, so no current run changes behavior.

    A configured block hands Accelerate explicit dimension sizes, and their
    product must equal ``num_processes`` -- Accelerate does not derive the
    remainder, so ``tp_size: 4`` alone on eight ranks is rejected as a plan for
    four. The shard size is therefore filled in here from the launched world,
    which is what asking only for a tensor-parallel degree means: give every
    group its own replica and let FSDP shard across them.

    The resulting ``DeviceMesh`` is what both ``data_parallel_geometry`` and
    ``fully_shard`` read, so it is built here even though Flow-Factory applies
    the tensor-parallel plan itself.

    Args:
        config: Parsed configuration carrying the parallelism block.

    Returns:
        The Accelerate parallelism plan, or ``None`` when no dimension was asked for.

    Raises:
        ValueError: If the plan leaves FSDP nothing to shard across, or does not
            divide the launched world.
    """
    parallelism_args = config.parallelism_args
    if not parallelism_args.is_configured:
        return None

    world_size = get_world_size()
    replicate_size = parallelism_args.dp_replicate_size or 1
    tensor_parallel_size = parallelism_args.effective_tensor_parallel_size
    shard_size = parallelism_args.dp_shard_size
    if shard_size is None:
        reserved = replicate_size * tensor_parallel_size
        if world_size % reserved != 0:
            raise ValueError(
                f"parallelism dp_replicate_size({replicate_size}) * "
                f"tp_size({tensor_parallel_size}) must divide the launched world size "
                f"{world_size} to derive dp_shard_size; received remainder "
                f"{world_size % reserved}"
            )
        shard_size = world_size // reserved

    # FSDP needs a dimension of its own. Accelerate rejects the all-tensor-parallel
    # plan too, but its message blames `DistributedConfig(tp_size=N)` and expert
    # parallelism, neither of which is how this plan was built, so the ceiling is
    # stated here instead: under FSDP2, `tp_size <= world_size / 2`. Pure tensor
    # parallelism without FSDP is a different backend, and
    # `validate_tensor_parallel_backend_plan` rejects it.
    if shard_size < 2 and _fsdp_requested():
        raise ValueError(
            f"parallelism.tp_size={tensor_parallel_size} leaves FSDP nothing to shard "
            f"across: it consumes all {world_size} launched ranks, so dp_shard_size "
            f"would be {shard_size}. Tensor parallelism composes with FSDP2 by splitting "
            f"each data-parallel replica, so at least two replicas are needed -- lower "
            f"tp_size to at most {world_size // 2}, or launch more ranks."
        )

    logger.info(
        "Requested parallelism plan: dp_replicate_size=%s, dp_shard_size=%s, tp_size=%s",
        parallelism_args.dp_replicate_size,
        shard_size,
        parallelism_args.tp_size,
    )
    return ParallelismConfig(
        dp_replicate_size=parallelism_args.dp_replicate_size,
        dp_shard_size=shard_size,
        tp_size=parallelism_args.tp_size,
    )


def _fsdp_requested() -> bool:
    """Report whether the launcher selected FSDP, before any Accelerator exists.

    ``accelerate launch`` exports this from the chosen config file, and the mesh
    has to be built before ``Accelerator.__init__`` can be asked.
    """
    return os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"


def load_trainer(config: Arguments) -> BaseTrainer:
    """Instantiate the configured trainer after validating its execution plan.

    Args:
        config: Parsed configuration containing the trainer, model, optimizer, and backend policy.

    Returns:
        The initialized trainer selected by ``config.training_args.trainer_type``.

    Raises:
        ImportError: If the requested trainer cannot be resolved or imported.
        TypeError: If the registry entry does not resolve to a trainer class.
        ValueError: If the trainer/adapter contract or distributed optimizer/checkpoint plan is
            unsupported.
        RuntimeError: If an FSDP2 checkpoint plan lacks the required plugin state.
    """
    # Resolve and validate algorithm semantics before constructing an Accelerator or
    # loading model weights. A stale trainer/argument registry pairing must fail with
    # no external allocation side effects.
    trainer_type = config.training_args.trainer_type
    try:
        trainer_cls = get_trainer_class(trainer_type)
    except ImportError as e:
        registered_trainers = list(list_registered_trainers().keys())
        raise ImportError(
            f"Failed to load trainer '{trainer_type}'. "
            f"Available trainers: {registered_trainers}"
        ) from e
    if not isinstance(trainer_cls, type):
        raise TypeError(
            f"trainer {trainer_type!r} must resolve to a class, " f"received {trainer_cls!r}"
        )
    uses_execution_kernel = issubclass(trainer_cls, BaseTrainer)
    has_typed_training_contract = isinstance(
        getattr(type(config.training_args), "execution_contract", None),
        ExecutionContract,
    )
    if uses_execution_kernel and has_typed_training_contract:
        trainer_cls.validate_training_arguments_contract(config.training_args)
        trainer_cls.validate_reward_optimization_overlap(config)

    # Resolve DDP find_unused_parameters from the trainable-role count and the
    # adapter class (opt-in per model). Resolving via the registry imports only the
    # class (no instantiation). This kwarg only affects the DDP backend; FSDP/DeepSpeed
    # ignore it. Default False lets DDP use static buckets and overlap gradient
    # all-reduce with backward; adapters that leave trainable params ungraded in
    # some iterations (e.g. Qwen-Image) opt in via ddp_find_unused_parameters.
    adapter_cls = get_model_adapter_class(config.model_args.model_type)
    if uses_execution_kernel:
        trainer_cls.validate_adapter_class_execution_contract(adapter_cls)
    find_unused = _requires_ddp_unused_parameter_detection(config, adapter_cls)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused)

    # DeepSpeed clips inside its engine: `accelerator.clip_grad_norm_` ignores the
    # value it is handed and only reports the resulting norm. The threshold therefore
    # has to reach the DeepSpeed plugin before the Accelerator is built, and accelerate
    # reads it from this environment variable. Without it the generated DeepSpeed
    # config carries an unresolved "auto" and `max_grad_norm` never takes effect.
    os.environ.setdefault("ACCELERATE_GRADIENT_CLIPPING", str(config.training_args.max_grad_norm))

    # Initialize Accelerator
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.log_args.save_dir, config.log_args.run_name),
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.training_args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        parallelism_config=_build_parallelism_config(config),
    )
    configure_backend_loading(accelerator, adapter_cls)
    # Validate the runtime backend before loading model weights. In particular,
    # constructing an adapter under ZeRO-3 can shard parameters immediately, so
    # rejecting it in BaseTrainer.__init__ is too late.
    validate_supported_distributed_plan(accelerator)
    validate_optimizer_backend_plan(accelerator, tuple(config.optimizer_args))
    validate_tensor_parallel_backend_plan(accelerator, adapter_cls, config.optimizer_args)
    configure_checkpointing_backend_plan(accelerator, config.training_args)
    set_seed(config.training_args.seed, device_specific=True)

    # Reconcile config with runtime distributed state (before any consumer reads it)
    reconcile_config(config, accelerator)

    # Initialize model adapter
    adapter = load_model(config=config, accelerator=accelerator)

    return trainer_cls(
        config=config,
        accelerator=accelerator,
        adapter=adapter,
    )
