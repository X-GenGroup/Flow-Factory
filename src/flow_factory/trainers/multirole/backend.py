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

"""Distributed-backend validation for trainer and multi-role invariants."""

from collections.abc import Sequence

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType

from ...hparams.optimizer_args import OptimizerArguments
from ...hparams.training_args import TrainingArguments
from ...optimizer import uses_muon, validate_muon_available
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def validate_supported_distributed_plan(accelerator: Accelerator) -> None:
    """Reject distributed plans this framework cannot train correctly."""
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        return
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    if deepspeed_plugin is None:
        return
    if deepspeed_plugin.zero_stage == 3:
        raise ValueError(
            "DeepSpeed ZeRO-3 is not supported by Flow-Factory: parameter sharding breaks "
            "reward-model loading and frozen-component synchronization. Expected ZeRO-1/2, "
            "FSDP, or DDP; received DeepSpeed stage 3."
        )


def validate_optimizer_backend_plan(
    accelerator: Accelerator,
    optimizer_args: Sequence[OptimizerArguments],
) -> None:
    """Reject optimizer/backend pairings before pretrained weights are loaded.

    Args:
        accelerator: Runtime backend whose optimizer support is being validated.
        optimizer_args: Parsed optimizer configurations for every trainable role.

    Raises:
        ValueError: If Muon is unavailable or paired with DeepSpeed or FSDP1.
    """
    if not uses_muon(optimizer_args):
        return
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        raise ValueError(
            "Muon with DeepSpeed is not verified in this framework: Muon rejects "
            "non-matrix parameters, so it runs inside a CompositeOptimizer, and "
            "DeepSpeed rebuilds its own optimizer wrapper around the object it "
            "receives. Use DDP or FSDP2 with Muon, or select the adamw optimizer."
        )
    if accelerator.distributed_type == DistributedType.FSDP:
        fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
        fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
        if fsdp_version < 2:
            raise ValueError(
                "Muon with FSDP1 does not work: FSDP1 flattens each wrapped unit into a "
                "1D FlatParameter, so Muon is constructed over matrices and then receives "
                "a 1D gradient, failing with 'Param gradient must be a 2D matrix' at the "
                "first optimizer step. Set `fsdp_version: 2` in the accelerate config "
                "(config/accelerate_configs/fsdp2.yaml), use DDP, or select the adamw "
                "optimizer."
            )
    validate_muon_available()


def tensor_parallel_size(accelerator: Accelerator) -> int:
    """Return the requested tensor-parallel degree, or 1 when none is configured."""
    parallelism_config = getattr(accelerator, "parallelism_config", None)
    if parallelism_config is None:
        return 1
    return int(getattr(parallelism_config, "tp_size", 1) or 1)


def validate_tensor_parallel_backend_plan(
    accelerator: Accelerator,
    adapter_cls: type,
    optimizer_args: Sequence[OptimizerArguments],
) -> None:
    """Reject tensor-parallel plans this framework cannot train correctly.

    Flow-Factory shards the denoiser itself and then lets FSDP2 compose over the
    result, so the supported backend is FSDP2 and the adapter must declare a
    plan. The rejections below are each a case that would otherwise fail late,
    or quietly produce wrong numbers:

    * **No adapter plan.** Nothing would be sharded and the run would train a
      fully replicated model while reporting a tensor-parallel world, with the
      data-parallel geometry already divided by ``tp_size`` -- so it would also
      see ``tp_size`` times fewer samples per step.
    * **Not FSDP2.** DDP accepts the mesh and then raises
      ``NotImplementedError`` from ``prepare`` because Accelerate expects an
      already-sharded transformers model; DeepSpeed ignores the dimension
      entirely (its tensor parallelism comes from ``autotp_size``) and would
      train pure data parallel while the samplers behave as if ranks were
      reserved. FSDP1 is rejected by Accelerate's own constructor.
    * **Muon.** Its Newton-Schulz iteration orthogonalizes each 2-D parameter
      through ``grad @ grad.T``. ``.ndim`` and ``.shape`` of a DTensor are
      global, so the guards pass, but whether the iteration redistributes
      correctly on a 2-D ``_StridedShard`` x ``Shard`` spec is unverified -- and
      if the spec degrades to a local tensor anywhere it orthogonalizes a shard
      instead of the matrix, which is wrong and silent.

    Args:
        accelerator: Runtime backend whose plan is being validated.
        adapter_cls: The resolved adapter class, consulted for its plan.
        optimizer_args: Parsed optimizer configurations for every trainable role.

    Raises:
        ValueError: If tensor parallelism is requested in an unsupported pairing.
    """
    tp_size = tensor_parallel_size(accelerator)
    if tp_size <= 1:
        return

    if not getattr(accelerator, "is_fsdp2", False):
        raise ValueError(
            f"tensor parallelism (tp_size={tp_size}) requires FSDP2, but the launch uses "
            f"{accelerator.distributed_type}. Flow-Factory shards the denoiser itself and "
            f"relies on FSDP2 composing over the result; DDP raises from `prepare` because "
            f"Accelerate expects an already-sharded transformers model, and DeepSpeed "
            f"ignores the dimension (its tensor parallelism is `autotp_size`) while the "
            f"samplers would already be sized for reserved ranks. Launch with "
            f"config/accelerate_configs/fsdp2.yaml, or set parallelism.tp_size=1."
        )

    plan = adapter_cls.tensor_parallel_plan()
    if not plan:
        raise ValueError(
            f"tensor parallelism (tp_size={tp_size}) was requested but "
            f"{adapter_cls.__name__} declares no tensor-parallel plan, so nothing would be "
            f"sharded. The run would train a fully replicated model while consuming "
            f"{tp_size}x fewer samples per step, because the samplers are already scoped to "
            f"the data-parallel group. Implement `tensor_parallel_plan()` on the adapter, "
            f"or set parallelism.tp_size=1."
        )

    if uses_muon(optimizer_args):
        raise ValueError(
            f"Muon with tensor parallelism (tp_size={tp_size}) is not verified: its "
            f"Newton-Schulz iteration orthogonalizes each matrix through `grad @ grad.T`, "
            f"and a DTensor reports global `.ndim`/`.shape` so the optimizer's own guards "
            f"cannot tell a sharded gradient from a whole one. Use adamw with tensor "
            f"parallelism, or set parallelism.tp_size=1."
        )


def configure_checkpointing_backend_plan(
    accelerator: Accelerator,
    training_args: TrainingArguments,
) -> bool:
    """Select one checkpointing owner for the active distributed plan.

    Args:
        accelerator: Runtime backend whose checkpointing policy is being configured.
        training_args: Parsed algorithm arguments containing the model checkpoint policy.

    Returns:
        True if model-level checkpointing was disabled and a caller holding an already-realized
        adapter must remove its checkpoint wrappers; otherwise False.

    Raises:
        ValueError: If FSDP2 is paired with a selective model checkpoint policy.
        RuntimeError: If FSDP2 backend checkpointing is selected without an FSDP plugin.
    """
    if accelerator.distributed_type != DistributedType.FSDP:
        return False
    fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
    model_checkpointing = bool(
        getattr(
            training_args,
            "gradient_checkpointing_enabled",
            getattr(training_args, "enable_gradient_checkpointing", False),
        )
    )
    fsdp_checkpointing = bool(getattr(fsdp_plugin, "activation_checkpointing", False))
    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) or 1

    if training_args.trainer_type == "tdm-r1" and fsdp_version < 2:
        if not model_checkpointing and not fsdp_checkpointing:
            return False
        training_args.enable_gradient_checkpointing = False
        if fsdp_plugin is not None:
            fsdp_plugin.activation_checkpointing = False
        logger.warning(
            "Disabled model and FSDP activation checkpointing for TDM-R1 on FSDP1: "
            "the surrogate objective runs reference/snapshot forwards between its live "
            "forward and backward, so FSDP1 recomputation saves a different graph. "
            "FSDP2 does not require this fallback."
        )
        return model_checkpointing

    if fsdp_version >= 2 and model_checkpointing:
        checkpoint_policy = training_args.enable_gradient_checkpointing
        full_checkpointing = checkpoint_policy is True or (
            getattr(checkpoint_policy, "mode", None) == "full"
        )
        if not full_checkpointing:
            raise ValueError(
                "FSDP2 activation checkpointing cannot preserve selective model "
                "checkpointing boundaries. Disable model checkpointing or use "
                "train.enable_gradient_checkpointing=true/mode=full."
            )
        training_args.enable_gradient_checkpointing = False
        if fsdp_plugin is None:
            raise RuntimeError(
                "FSDP2 full activation checkpointing requires an FSDP plugin, received None"
            )
        fsdp_plugin.activation_checkpointing = True
        logger.info(
            "Selected FSDP2 backend activation checkpointing and disabled train-level "
            "model checkpointing so recomputation stays inside the mixed-precision boundary."
        )
        return True

    if fsdp_version < 2 and model_checkpointing and fsdp_checkpointing:
        fsdp_plugin.activation_checkpointing = False
        logger.info(
            "Disabled FSDP activation checkpointing because train-level model "
            "checkpointing is enabled; nested checkpoint boundaries duplicate recompute."
        )
    return False


def configure_deepspeed_micro_batch_size(
    accelerator: Accelerator, per_device_batch_size: int
) -> None:
    """Supply batch geometry when the custom train loader is not prepared."""
    if not isinstance(per_device_batch_size, int):
        raise TypeError(
            "expected int for per_device_batch_size, "
            f"got {type(per_device_batch_size).__name__}: {per_device_batch_size!r}"
        )
    if per_device_batch_size < 1:
        raise ValueError(f"expected per_device_batch_size >= 1, got {per_device_batch_size}")
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        return
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    if deepspeed_plugin is None:
        raise RuntimeError(
            "expected a DeepSpeed plugin for distributed_type=DEEPSPEED, received None"
        )
    key = "train_micro_batch_size_per_gpu"
    configured = deepspeed_plugin.deepspeed_config.get(key)
    if configured not in (None, "auto", per_device_batch_size):
        raise ValueError(
            f"expected DeepSpeed {key} to equal per_device_batch_size "
            f"{per_device_batch_size}, got {configured!r}"
        )
    deepspeed_plugin.deepspeed_config[key] = per_device_batch_size


class MultiRoleBackendValidationMixin:
    """Validate optimizer, prepared-root, and trainability backend contracts."""

    def _validate_optimizer_backend(
        self,
        optimizer_args: Sequence[OptimizerArguments],
    ) -> None:
        """Reject optimizer and distributed-backend pairings that are not verified."""
        validate_optimizer_backend_plan(self.accelerator, optimizer_args)

    def _validate_trainable_parameters_survived_prepare(self) -> None:
        """Reject a prepared root that no rank can train."""
        local = sum(
            parameter.numel()
            for parameter in self.model_bundle.parameters()
            if parameter.requires_grad
        )
        counts = torch.tensor([float(local)], device=self.accelerator.device)
        global_trainable = int(self.accelerator.reduce(counts, reduction="sum").item())
        if global_trainable > 0:
            return
        raise RuntimeError(
            "expected the prepared model to expose trainable parameters under "
            f"{self.accelerator.distributed_type}, received 0 across all "
            f"{self.accelerator.num_processes} rank(s). Every parameter was frozen or "
            "absorbed while wrapping, so an optimizer step would change nothing."
        )

    def _validate_multirole_backend(self) -> None:
        """Validate one-root backend semantics for multi-role trainers."""
        required_roles = self._required_trainable_roles()
        if len(required_roles) <= 1:
            return

        algorithm = self.training_args.trainer_type
        prepared_models = tuple(self.accelerator._models)
        if len(prepared_models) != 1:
            raise RuntimeError(
                f"algorithm {algorithm!r} with roles {required_roles!r} expected exactly "
                f"one prepared model root, received {len(prepared_models)}"
            )
        tracked = self.accelerator.unwrap_model(prepared_models[0])
        driven = self.accelerator.unwrap_model(self.model_bundle)
        if tracked is not driven:
            raise RuntimeError(
                f"algorithm {algorithm!r} with roles {required_roles!r} expected the tracked "
                "prepared model root to be self.model_bundle, received different identities"
            )

        prepared_optimizers = tuple(self.accelerator._optimizers)
        if len(prepared_optimizers) != 1:
            raise RuntimeError(
                f"algorithm {algorithm!r} with roles {required_roles!r} expected exactly "
                f"one prepared optimizer, received {len(prepared_optimizers)}"
            )
        if prepared_optimizers[0] is not self.optimizer:
            raise RuntimeError(
                f"algorithm {algorithm!r} with roles {required_roles!r} expected the tracked "
                "prepared optimizer to be self.optimizer, received different identities"
            )

        prepared_group_roles = tuple(
            group.get("role_name") for group in self.optimizer.param_groups
        )
        if prepared_group_roles != self._unprepared_optimizer_group_roles:
            raise RuntimeError(
                f"algorithm {algorithm!r} optimizer group role mapping expected "
                f"{self._unprepared_optimizer_group_roles!r}, received {prepared_group_roles!r}"
            )

        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if deepspeed_plugin is not None and deepspeed_plugin.zero_stage not in (1, 2):
            raise ValueError(
                f"DeepSpeed multi-role algorithm {algorithm!r} with roles "
                f"{required_roles!r} requires zero_stage in (1, 2), received "
                f"zero_stage={deepspeed_plugin.zero_stage!r}"
            )

        fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
        if fsdp_plugin is None or fsdp_plugin.fsdp_version != 2:
            return
        if fsdp_plugin.use_orig_params is False:
            raise ValueError(
                f"FSDP2 multi-role algorithm {algorithm!r} with roles {required_roles!r} "
                "requires use_orig_params=True, received False"
            )
        prepared_root_parameter_ids = {
            id(parameter)
            for parameter in self.accelerator.unwrap_model(self.model_bundle).parameters()
        }
        foreign_parameters = tuple(
            (group_id, group["role_name"], parameter_index, id(parameter))
            for group_id, group in enumerate(self.optimizer.param_groups)
            for parameter_index, parameter in enumerate(group["params"])
            if id(parameter) not in prepared_root_parameter_ids
        )
        if foreign_parameters:
            raise RuntimeError(
                "FSDP2 optimizer parameter identity must reference the prepared model root "
                f"for algorithm {algorithm!r} with roles {required_roles!r}; received foreign "
                f"(group_id, role_name, parameter_index, parameter_id) entries "
                f"{foreign_parameters!r}"
            )
