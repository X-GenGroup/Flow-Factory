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
"""
Trainer loader factory for extensibility.
Supports multiple RL algorithms via registry pattern.
"""

import logging
import os

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed

from ..contracts.execution import ExecutionContract
from ..hparams import Arguments, get_training_args_class
from ..loading.backend import configure_backend_loading
from ..models.loader import load_model
from ..models.registry import get_model_adapter_class
from ..utils.env_utils import reconcile_config
from ..utils.logger_utils import setup_logger
from .abc import BaseTrainer
from .multirole import (
    configure_checkpointing_backend_plan,
    validate_optimizer_backend_plan,
    validate_supported_distributed_plan,
)
from .registry import get_trainer_class, list_registered_trainers

logger = setup_logger(__name__)


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

    # Resolve DDP find_unused_parameters from the adapter class (opt-in per
    # model). Resolving via the registry imports only the class (no
    # instantiation). This kwarg only affects the DDP backend; FSDP/DeepSpeed
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
    )
    configure_backend_loading(accelerator, adapter_cls)
    # Validate the runtime backend before loading model weights. In particular,
    # constructing an adapter under ZeRO-3 can shard parameters immediately, so
    # rejecting it in BaseTrainer.__init__ is too late.
    validate_supported_distributed_plan(accelerator)
    validate_optimizer_backend_plan(accelerator, tuple(config.optimizer_args))
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
