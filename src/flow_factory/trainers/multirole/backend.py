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

import json
import os
from collections.abc import Sequence
from typing import Any, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType

from ...contracts.sampler import get_sampler_layout_contract
from ...hparams.optimizer_args import OptimizerArguments
from ...hparams.training_args import TrainingArguments
from ...models.tensor_parallel import resolve_parallel_plan, sequence_parallel_enabled
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
    sequence_parallel: Optional[bool] = None,
    declared_tensor_parallel_size: Optional[int] = None,
) -> None:
    """Reject tensor-parallel plans this framework cannot train correctly.

    Flow-Factory shards the denoiser itself and then lets FSDP2 compose over the
    result, so the supported backend is FSDP2 and the adapter must declare a
    plan. The rejections below are each a case that would otherwise fail late,
    or quietly produce wrong numbers:

    * **The two readings of the degree disagree.** Batch geometry reads the
      config (``Arguments.data_parallel_world_size()``, which has to answer
      before any ``Accelerator`` exists) while the mesh, the sharding and the
      samplers read ``accelerator.parallelism_config``. They are built from one
      field, so agreement is structural rather than enforced -- and a structural
      argument is worth one comparison, because a divergence is silent: the
      global batch comes out a factor of ``tp_size`` wrong and nothing reports
      it. Checked here because no later point has both.
    * **No adapter plan.** Nothing would be sharded and the run would train a
      fully replicated model while reporting a tensor-parallel world, with the
      data-parallel geometry already divided by ``tp_size`` -- so it would also
      see ``tp_size`` times fewer samples per step.
    * **No sequence-parallel plan where one applies.** Spelling
      ``sequence_parallel: true`` and getting tensor parallelism alone is not
      what was asked for, and an adapter enabling the overlay by default must
      declare one. Checked here so it costs nothing instead of after a 14B load.
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
        sequence_parallel: The configured ``parallelism.sequence_parallel``;
            ``None`` defers to the adapter's default.
        declared_tensor_parallel_size: The degree the *config* reads, from
            ``parallelism_args.effective_tensor_parallel_size``. ``None`` skips
            the comparison, for callers that have no parsed config.

    Raises:
        ValueError: If tensor parallelism is requested in an unsupported pairing.
    """
    tp_size = tensor_parallel_size(accelerator)
    # Before the early return, not after: a config asking for 2 against a runtime
    # of 1 is just as silent, and that direction would exit here unchecked.
    if declared_tensor_parallel_size is not None and declared_tensor_parallel_size != tp_size:
        raise ValueError(
            f"the tensor-parallel degree disagrees between its two readers: the config says "
            f"{declared_tensor_parallel_size} (parallelism.tp_size, which is what every batch "
            f"geometry rule divides the world by) and the built runtime says {tp_size} (the "
            f"mesh, the sharding and the samplers). They are built from one field, so this "
            f"means something refilled it -- `ParallelismConfig.__post_init__` reads "
            f"PARALLELISM_CONFIG_TP_SIZE for any dimension left unset. Nothing downstream "
            f"would report it: the global batch would simply come out a factor of "
            f"{max(declared_tensor_parallel_size, tp_size) // max(min(declared_tensor_parallel_size, tp_size), 1)} "
            f"wrong. Set parallelism.tp_size explicitly, or unset that environment variable."
        )
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
    fsdp_plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    if getattr(fsdp_plugin, "cpu_ram_efficient_loading", False):
        raise ValueError(
            f"tensor parallelism (tp_size={tp_size}) cannot use FSDP's cpu_ram_efficient_loading: "
            f"Accelerate's full-state-dict load broadcasts rank 0's tensors into every rank, and "
            f"under tensor parallelism rank 0 holds only its own shard while the others expect the "
            f"whole tensor, so the load fails on a size mismatch. Set "
            f"fsdp_cpu_ram_efficient_loading: false in the accelerate config, or set "
            f"parallelism.tp_size=1."
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

    # Called for its effect, not its value: this is the one place the policy is
    # cheap to refuse, and `_apply_tensor_parallel` calls the same resolver for
    # the merged plan once the weights are loaded. Written twice, the rule drifted
    # into two messages for one condition.
    resolve_parallel_plan(adapter_cls, sequence_parallel=sequence_parallel)

    if uses_muon(optimizer_args):
        raise ValueError(
            f"Muon with tensor parallelism (tp_size={tp_size}) is not verified: its "
            f"Newton-Schulz iteration orthogonalizes each matrix through `grad @ grad.T`, "
            f"and a DTensor reports global `.ndim`/`.shape` so the optimizer's own guards "
            f"cannot tell a sharded gradient from a whole one. Use adamw with tensor "
            f"parallelism, or set parallelism.tp_size=1."
        )


def _read_component_config(model_name_or_path: str, subfolder: str) -> Optional[dict]:
    """Read one component's diffusers ``config.json`` without touching its weights.

    ``None`` for a single-file checkpoint, which carries no separate config.
    """
    if os.path.isfile(model_name_or_path):
        return None
    if os.path.isdir(model_name_or_path):
        path = os.path.join(model_name_or_path, subfolder, "config.json")
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_name_or_path, "config.json", subfolder=subfolder)
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"expected a JSON object in {path}, received {type(config).__name__}")
    return config


def _nested_config_value(config: dict, dotted_key: str, where: str) -> Any:
    """Follow a dotted key into nested config objects; ``None`` when a level is absent."""
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict):
            raise TypeError(
                f"expected a JSON object above {part!r} in {where}:{dotted_key}, "
                f"received {type(value).__name__}: {value!r}"
            )
        value = value.get(part)
        if value is None:
            return None
    return value


def validate_tensor_parallel_run_config(config: Any, adapter_cls: type) -> None:
    """Refuse run settings tensor parallelism cannot honour, before any weights load.

    * **Head counts.** Every head-sharded projection needs ``tp_size`` to divide
      its head count, and SD3.5-Large (38), Z-Image (30) and others do not divide
      by every power of two; SenseNova's 8 kv heads cap it at 8. Read from the
      checkpoint's config, where the adapter declares it in
      ``tensor_parallel_head_config``.
    * **``subgroup_tile`` sampling.** It builds its subgroups from contiguous
      launched ranks, which stop matching the replica set once ranks are reserved
      for tensor parallelism.
    * **``diffusers_cache`` with sequence parallelism.** The cache decides which
      blocks to skip from the residual, which the overlay shards across the
      group, so peers can skip different blocks and post mismatched collectives.

    Args:
        config: The parsed ``Arguments``.
        adapter_cls: The resolved adapter class.

    Raises:
        ValueError: If a setting cannot run under the configured degree.
    """
    tp_size = config.parallelism_args.effective_tensor_parallel_size
    if tp_size <= 1:
        return

    sampler_type = config.data_args.sampler_type
    if get_sampler_layout_contract(sampler_type).group_placement == "subgroup_tile":
        raise ValueError(
            f"sampler_type={sampler_type!r} builds its subgroups from contiguous launched "
            f"ranks, which stop matching the replica set under tensor parallelism "
            f"(tp_size={tp_size}). Use the 'rank_local', 'global_batch' or 'global_tile' "
            f"layout, or set parallelism.tp_size=1."
        )

    if sequence_parallel_enabled(adapter_cls, config.parallelism_args.sequence_parallel):
        acceleration = config.acceleration_args
        cached = [
            spec.name
            for spec in (*acceleration.shared, *acceleration.rollout)
            if spec.name.lower() == "diffusers_cache"
        ]
        if cached:
            raise ValueError(
                "the diffusers_cache accelerator decides which blocks to skip from the "
                "residual stream, which sequence parallelism shards across the "
                f"tensor-parallel group (tp_size={tp_size}); peers can then skip different "
                "blocks and post mismatched collectives. Remove the accelerator, or set "
                "parallelism.sequence_parallel: false."
            )

    declaration = adapter_cls.tensor_parallel_head_config
    if declaration is None:
        return
    subfolder, keys = declaration
    model_name_or_path = config.model_args.model_name_or_path
    component_config = _read_component_config(model_name_or_path, subfolder)
    if component_config is None:
        return
    where = os.path.join(subfolder, "config.json")
    for key in keys:
        heads = _nested_config_value(component_config, key, where)
        if heads is None:
            continue
        if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
            raise TypeError(
                f"expected a positive int for {where}:{key} of "
                f"{model_name_or_path!r}, received {heads!r}"
            )
        if heads % tp_size:
            divisors = [d for d in range(2, min(heads, 64) + 1) if heads % d == 0]
            raise ValueError(
                f"{adapter_cls.__name__}: {model_name_or_path!r} has {key}={heads}, which "
                f"parallelism.tp_size={tp_size} does not divide, so its heads cannot be split "
                f"across the group. Degrees that divide it: {divisors or 'none above 1'}."
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
