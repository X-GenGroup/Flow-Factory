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
from accelerate.utils import ProjectConfiguration

from ..contracts.execution import ExecutionContract
from ..hparams import Arguments, get_training_args_class
from ..loading.backend import configure_backend_loading
from ..models.compute_precision import resolve_compute_precision_policy
from ..models.loader import load_model
from ..models.parallelism.context import validate_prepared_context_parallel
from ..models.registry import get_model_adapter_class
from ..utils.dist import get_world_size, seed_data_parallel_replica
from ..utils.env_utils import reconcile_config
from ..utils.logger_utils import setup_logger
from .abc import BaseTrainer
from .common.context_parallel import (
    validate_context_parallel_backend,
    validate_context_parallel_config,
)
from .multirole import (
    configure_checkpointing_backend_plan,
    validate_optimizer_backend_plan,
    validate_supported_distributed_plan,
    validate_target_materialization_plan,
    validate_tensor_parallel_backend_plan,
    validate_tensor_parallel_run_config,
)
from .registry import get_trainer_class, list_registered_trainers

logger = setup_logger(__name__)


class Accelerator(_UpstreamAccelerator):
    """An Accelerator that uses Flow-Factory's prepared TP and Diffusers CP plans.

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

    def _prepare_cp(self, *args: Any) -> Any:
        """Use Diffusers hooks only for a root already prepared by Flow-Factory."""
        validate_prepared_context_parallel(args, self.parallelism_config.cp_size)
        return args

    def load_state(self, input_dir=None, load_kwargs=None, **load_model_func_kwargs):
        """Preserve lazy AdamW state in native FSDP2 sharded checkpoints."""
        from accelerate.utils import DistributedType, load_fsdp_optimizer
        from torch.distributed.fsdp import StateDictType

        from .common.fsdp_optimizer_checkpoint import inspect_sparse_adamw_checkpoint

        plugin = getattr(self.state, "fsdp_plugin", None)
        if (
            input_dir is None
            or self.distributed_type != DistributedType.FSDP
            or getattr(plugin, "fsdp_version", None) != 2
            or plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT
        ):
            return super().load_state(input_dir, load_kwargs=load_kwargs, **load_model_func_kwargs)
        input_dir = os.path.expanduser(input_dir)
        original_optimizers = self._optimizers
        sparse = [
            inspect_sparse_adamw_checkpoint(self._models[index], optimizer, input_dir, index)
            for index, optimizer in enumerate(original_optimizers)
        ]
        if not any(sparse):
            return super().load_state(input_dir, load_kwargs=load_kwargs, **load_model_func_kwargs)

        def restore_optimizers(models, checkpoint_dir):
            del models
            # FSDP2's native model/scaler loads precede this public hook. Existing
            # hooks run first; schedulers, RNG and custom objects still load next.
            for index, (optimizer, checkpoint) in enumerate(zip(original_optimizers, sparse)):
                if checkpoint is None:
                    load_fsdp_optimizer(
                        plugin, self, optimizer, self._models[index], checkpoint_dir, index
                    )
                else:
                    checkpoint.load()

        handle = self.register_load_state_pre_hook(restore_optimizers)
        self._optimizers = []
        try:
            return super().load_state(input_dir, load_kwargs=load_kwargs, **load_model_func_kwargs)
        finally:
            self._optimizers = original_optimizers
            handle.remove()


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

    topology = parallelism_args.resolve_topology(get_world_size())
    world_size = topology.world_size
    shard_size = topology.dp_shard_size

    # Each TP/CP peer and replicated-DP coordinate still needs a shard group.
    if shard_size < 2 and _fsdp_requested():
        raise ValueError(
            f"FSDP requires dp_shard_size >= 2; resolved {topology.dimensions} "
            f"from launched world size {world_size}. Reduce tp_size, cp_size or "
            "dp_replicate_size, or launch more ranks. Omitting dp_shard_size "
            "derives the remainder but does not relax this minimum."
        )

    logger.info(
        "Resolved parallelism plan: world_size=%s, dp_replicate_size=%s, "
        "dp_shard_size=%s (%s), tp_size=%s, cp_size=%s, logical_dp_size=%s",
        world_size,
        topology.dp_replicate_size,
        shard_size,
        "inferred" if parallelism_args.dp_shard_size is None else "explicit",
        topology.tp_size,
        topology.cp_size,
        topology.data_parallel_size,
    )
    return ParallelismConfig(
        # The *resolved* degrees, not `parallelism_args.*`. An unset dimension
        # reaches `ParallelismConfig` as `None`, and its `__post_init__` then
        # fills it from `PARALLELISM_CONFIG_{DP_REPLICATE,DP_SHARD,TP}_SIZE` --
        # so a config writing only `dp_shard_size` could come back with a
        # `tp_size` the lines above never saw. That splits the tensor-parallel
        # degree into two disagreeing sources: batch geometry reads
        # `Arguments.data_parallel_world_size()` (config side) while the mesh,
        # the sharding and the samplers read `accelerator.parallelism_config`.
        # Measured with `PARALLELISM_CONFIG_TP_SIZE=2` and a config setting only
        # `dp_shard_size`: the config side says 1 and the runtime says 2, so
        # `num_batches_per_epoch` -- and under `auto`,
        # `gradient_accumulation_steps` -- comes out a factor of 2 wrong, which
        # is exactly what `data_parallel_world_size`'s docstring warns about and
        # nothing would have reported. It also made `shard_size` inconsistent
        # with the config it is returned in, since it was derived from these
        # resolved values.
        **topology.dimensions,
        # Accelerate's independent DeepSpeed SP is not Flow-Factory's TP-bound
        # sequence_parallel overlay. Neither it nor CP can be injected through
        # an environment default into this supported TP/FSDP plan.
        sp_size=1,
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
    resolve_compute_precision_policy(adapter_cls, config)
    validate_context_parallel_config(config, adapter_cls, trainer_cls)
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
    validate_context_parallel_backend(
        accelerator, config.parallelism_args.effective_context_parallel_size
    )
    configure_backend_loading(accelerator, adapter_cls)
    # Validate the runtime backend before loading model weights. In particular,
    # constructing an adapter under ZeRO-3 can shard parameters immediately, so
    # rejecting it in BaseTrainer.__init__ is too late.
    validate_supported_distributed_plan(accelerator)
    validate_optimizer_backend_plan(accelerator, tuple(config.optimizer_args))
    validate_target_materialization_plan(
        accelerator,
        adapter_cls,
        config.parallelism_args.target_materialization,
        config.parallelism_args.effective_context_parallel_size,
    )
    validate_tensor_parallel_backend_plan(
        accelerator,
        adapter_cls,
        config.optimizer_args,
        config.parallelism_args.sequence_parallel,
        config.parallelism_args.effective_tensor_parallel_size,
        config.parallelism_args.target_materialization,
    )
    validate_tensor_parallel_run_config(config, adapter_cls)
    configure_checkpointing_backend_plan(accelerator, config.training_args)
    seed_data_parallel_replica(accelerator, config.training_args.seed)

    # Reconcile config with runtime distributed state (before any consumer reads it)
    reconcile_config(config, accelerator)

    # Initialize model adapter
    adapter = load_model(config=config, accelerator=accelerator)

    return trainer_cls(
        config=config,
        accelerator=accelerator,
        adapter=adapter,
    )
