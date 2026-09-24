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

# src/flow_factory/trainers/abc.py
import json
import os
import time
from abc import ABC
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DistributedType, ProjectConfiguration, gather_object, set_seed
from diffusers.utils.outputs import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..acceleration import BaseAccelerator, build_accelerator, validate_accelerator
from ..advantage import AdvantageProcessor, CollectedGroupLayout
from ..contracts.execution import (
    ONLINE_EXECUTION_CONTRACT,
    AcquisitionMode,
    ExecutionContract,
    FeedbackMode,
)
from ..contracts.feedback import resolve_feedback_reducer_contract
from ..contracts.reward_overlap import (
    NO_REWARD_OPTIMIZATION_OVERLAP,
    RewardOptimizationOverlapContract,
)
from ..contracts.sampler import get_sampler_layout_contract
from ..data_utils.dataset import METADATA_COLUMN
from ..data_utils.loader import (
    get_eval_dataloaders,
    get_train_dataloader,
)
from ..data_utils.sampling_plan import (
    SAMPLING_GROUP_ID_COLUMN,
    SAMPLING_GROUP_MEMBER_ID_COLUMN,
    SAMPLING_IDENTITY_COLUMNS,
    SAMPLING_SAMPLE_ID_COLUMN,
)
from ..hparams import *
from ..hparams.optimizer_args import OptimizerArguments
from ..loading import ComponentRole, ModelLoadCoordinator
from ..logger import LogFormatter, load_logger
from ..models.abc import BaseAdapter
from ..models.model_bundle import ModelBundle, RoutedComponentProxy
from ..models.variants import DEFAULT_BASE_VARIANT, ComponentVariantRegistry
from ..optimizer import build_optimizer
from ..rewards import (
    BaseRewardModel,
    GroupwiseRewardModel,
    MultiRewardLoader,
    RewardBuffer,
    RewardProcessor,
    RewardTile,
    RewardTileGeometry,
    RewardTilePlan,
    build_reward_tile_plan,
)
from ..samples import (
    AcquisitionManifest,
    BaseSample,
    LatentState,
    NoisedState,
    StackedSampleBatch,
    group_identity_rows,
)
from ..utils.base import (
    create_generator,
    create_generator_by_prompt,
    filter_kwargs,
    json_default,
)
from ..utils.checkpoint import (
    HF_PATH_PREFIX,
    download_hf_checkpoint,
    parse_hf_checkpoint_path,
)
from ..utils.dist import gather_aligned_floating_tensors, get_world_size, reduce_loss_info
from ..utils.group_coordinator import GroupCoordinator
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import TimeSampler
from .common.runtime_identity import (
    build_default_data_identity_payload,
    build_default_execution_identity_payload,
    build_trainer_runtime_identity,
)
from .common.runtime_state import TrainerRuntimeState
from .common.sample_prefetch import iter_prefetched_batches
from .execution import (
    AcquisitionDriver,
    TrainingProgress,
    build_acquisition_driver,
)
from .multirole import (
    MULTIROLE_RUNTIME_CHILD_NAME,
    MultiRoleBackendValidationMixin,
    MultiRoleCheckpointingMixin,
    configure_checkpointing_backend_plan,
    configure_deepspeed_micro_batch_size,
    validate_supported_distributed_plan,
)
from .role_optimization import (
    OptimizationRole,
    RoleOptimizationCoordinator,
    RoleOptimizerConfig,
    RolePhase,
    RoleUpdatePlan,
)

logger = setup_logger(__name__)


@dataclass(frozen=True)
class _RewardOverlapGroupInfo:
    """Acquisition-level cross-rank group metadata reused by optimizers."""

    local_group_identities: torch.Tensor
    local_group_indices: torch.Tensor
    num_groups: int

    @property
    def local_unique_ids(self) -> torch.Tensor:
        """Return the unique-id column for objective-specific seed logic."""

        return self.local_group_identities[:, 1]

    @property
    def local_source_ids(self) -> torch.Tensor:
        """Return the source-id column of the canonical group identity."""

        return self.local_group_identities[:, 0]

    def validate_samples(self, samples: Sequence[BaseSample], *, context: str) -> None:
        """Require the optimizer batch to match the cached identity mapping."""

        identities = torch.tensor(
            group_identity_rows(samples),
            dtype=torch.int64,
            device=self.local_group_identities.device,
        )
        if not torch.equal(identities, self.local_group_identities):
            raise RuntimeError(
                f"reward overlap group metadata does not match {context}: "
                f"cached={self.local_group_identities.tolist()}, "
                f"received={identities.tolist()}"
            )


class BaseTrainer(MultiRoleCheckpointingMixin, MultiRoleBackendValidationMixin, ABC):
    """
    Abstract Base Class for Flow-Factory trainers.
    """

    # RL paradigm of this algorithm (``constraints.md`` #7). Read by the
    # acceleration validator to gate lossy rollout accelerators: only
    # 'decoupled' / 'distillation' trainers may use them. Concrete trainers
    # MUST override this; leaving it None disables lossy acceleration.
    paradigm: ClassVar[Optional[Literal["coupled", "decoupled", "distillation"]]] = None
    execution_contract: ClassVar[ExecutionContract] = ONLINE_EXECUTION_CONTRACT
    reward_optimization_overlap_contract: ClassVar[RewardOptimizationOverlapContract] = (
        NO_REWARD_OPTIMIZATION_OVERLAP
    )
    runtime_child_names: ClassVar[Tuple[str, ...]] = ()

    _ADAPTER_EMA_RUNTIME_CHILD = "adapter_ema"
    _ADAPTER_REFERENCE_RUNTIME_CHILD = "adapter_reference"

    def __init__(
        self,
        accelerator: Accelerator,
        config: Arguments,
        adapter: BaseAdapter,
    ):
        validate_supported_distributed_plan(accelerator)

        self.accelerator = accelerator
        self.config = config
        self.log_args = config.log_args
        self.model_args = config.model_args

        self.training_args = config.training_args
        self.eval_args = config.eval_args
        self._validate_execution_contract()

        self.reward_args = config.reward_args
        self.eval_reward_args = (
            config.eval_reward_args or config.reward_args
        )  # If `eval_reward_args` is not given, use `reward_args`
        type(self).validate_reward_optimization_overlap(config)

        self.adapter = adapter
        self._validate_adapter_execution_contract()
        self.load_coordinator = ModelLoadCoordinator(adapter, accelerator)
        self.runtime_state = TrainerRuntimeState(child_names=self._declared_runtime_child_names())
        self._runtime_children_attached = False
        self._exact_resume_source_checkpoint: Optional[str] = None
        self._exact_resume_boundary_pending = False
        self._acquisition_cycle_active = False
        self._acquisition_cycle_incomplete = False
        self._reward_overlap_acquisition_manifest: Optional[AcquisitionManifest] = None
        self.acquisition_driver: AcquisitionDriver = build_acquisition_driver(
            type(self).execution_contract
        )
        self._validate_execution_hooks()

        self._initialization()
        self._realize_runtime_child_declarations()
        self._initialize_adapter_runtime()
        self._initialize_snapshots()
        self._register_multirole_checkpointing()
        self.runtime_state.configure_identity(build_trainer_runtime_identity(self))
        self._finalize_adapter_runtime()
        # Apply persistent stage='both' accelerators last: after prepare, state-resume,
        # EMA, and reference-parameter setup, so e.g. torch.compile wraps the final
        # weights and keeps state_dict keys / parameter identity stable.
        self._apply_shared_acceleration()
        self._init_logging_backend()

        self._patch_deepspeed_autocast(accelerator)
        self.autocast = partial(
            torch.autocast,
            device_type=accelerator.device.type,
            dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16,
        )

        if self.accelerator.is_local_main_process:
            self.adapter.log_trainable_parameters()

    @property
    def show_progress_bar(self) -> bool:
        """Whether to show tqdm progress bars."""
        return self.log_args.verbose and self.accelerator.is_local_main_process

    @property
    def cycle_index(self) -> int:
        """Return the completed acquisition-cycle count for this trainer."""
        return self._get_progress().cycle_index(type(self).execution_contract.acquisition)

    @property
    def progress(self) -> TrainingProgress:
        """Return the single runtime-owned progress value.

        Lightweight structural tests that construct a trainer without running
        ``BaseTrainer.__init__`` retain a private fallback, but initialized trainers
        never duplicate counters outside :class:`TrainerRuntimeState`.
        """
        runtime_state = self.__dict__.get("runtime_state")
        if runtime_state is not None:
            if not isinstance(runtime_state, TrainerRuntimeState):
                raise TypeError(
                    "expected runtime_state to be TrainerRuntimeState, received "
                    f"{type(runtime_state).__name__}: {runtime_state!r}"
                )
            return runtime_state.progress
        progress = self.__dict__.get("_lightweight_progress")
        if progress is None:
            progress = TrainingProgress()
            self.__dict__["_lightweight_progress"] = progress
        return progress

    @progress.setter
    def progress(self, progress: TrainingProgress) -> None:
        """Replace runtime progress without maintaining a second counter copy."""
        if not isinstance(progress, TrainingProgress):
            raise TypeError(
                "expected trainer progress to be TrainingProgress, received "
                f"{type(progress).__name__}: {progress!r}"
            )
        runtime_state = self.__dict__.get("runtime_state")
        if runtime_state is None:
            self.__dict__["_lightweight_progress"] = progress
        else:
            runtime_state.progress = progress

    def _get_progress(self) -> TrainingProgress:
        """Return typed progress for initialized and lightweight test trainers."""
        progress = self.progress
        if not isinstance(progress, TrainingProgress):
            raise TypeError(
                "expected trainer progress to be TrainingProgress, received "
                f"{type(progress).__name__}: {progress!r}"
            )
        return progress

    @property
    def epoch(self) -> int:
        """Return the compatibility alias for the active acquisition cycle.

        Generated acquisition maps this alias to rollout iterations. Dataset
        acquisition maps it to complete dataloader traversals.
        """
        return self.cycle_index

    @epoch.setter
    def epoch(self, value: int) -> None:
        """Set the compatibility acquisition-cycle alias.

        Args:
            value: Non-negative completed-cycle count.
        """
        progress = self._get_progress()
        if type(self).execution_contract.acquisition is AcquisitionMode.GENERATION:
            self.progress = replace(progress, rollout_iteration=value)
        else:
            self.progress = replace(progress, data_epoch=value)

    @property
    def step(self) -> int:
        """Return the completed primary optimizer-step count."""
        return self._get_progress().optimizer_step

    @step.setter
    def step(self, value: int) -> None:
        """Set the completed primary optimizer-step count.

        Args:
            value: Non-negative number of completed optimizer updates.
        """
        self.progress = replace(self._get_progress(), optimizer_step=value)

    def _validate_execution_hooks(self) -> None:
        """Require the optimization hook selected by acquisition mode."""
        acquisition = type(self).execution_contract.acquisition
        if (
            acquisition is AcquisitionMode.GENERATION
            and type(self).optimize is BaseTrainer.optimize
        ):
            raise TypeError(
                f"generation trainer {type(self).__name__} must override optimize(samples)"
            )
        if (
            acquisition is AcquisitionMode.DATASET
            and type(self).optimize_batch is BaseTrainer.optimize_batch
        ):
            raise TypeError(
                f"dataset trainer {type(self).__name__} must override optimize_batch(batch)"
            )

    def runtime_execution_identity_payload(self) -> Dict[str, Any]:
        """Return exact-resume objective semantics, extensible by trainers."""
        return build_default_execution_identity_payload(self)

    def runtime_data_identity_payload(self) -> Dict[str, Any]:
        """Return exact-resume loader semantics, extensible by trainers."""
        return build_default_data_identity_payload(self)

    def _validate_execution_contract(self) -> None:
        """Require trainer runtime and arguments to declare equal semantics."""
        type(self).validate_training_arguments_contract(self.training_args)

    @classmethod
    def validate_training_arguments_contract(cls, training_args: Any) -> None:
        """Validate algorithm arguments before heavyweight initialization.

        Args:
            training_args: Resolved algorithm-specific training arguments.
        """
        trainer_contract = cls.execution_contract
        arguments_contract = getattr(type(training_args), "execution_contract", None)
        if not isinstance(trainer_contract, ExecutionContract):
            raise TypeError(
                f"trainer {cls.__name__}.execution_contract must be ExecutionContract, "
                f"got {type(trainer_contract).__name__}: {trainer_contract!r}"
            )
        if not isinstance(arguments_contract, ExecutionContract):
            raise TypeError(
                f"training arguments {type(training_args).__name__}.execution_contract "
                f"must be ExecutionContract, got {type(arguments_contract).__name__}: "
                f"{arguments_contract!r}"
            )
        if trainer_contract != arguments_contract:
            raise ValueError(
                f"execution contract mismatch for trainer {cls.__name__} and "
                f"training arguments {type(training_args).__name__}: "
                f"trainer={trainer_contract!r}, arguments={arguments_contract!r}"
            )

    @classmethod
    def validate_reward_optimization_overlap(cls, config: Arguments) -> None:
        """Validate streamed reward optimization before heavyweight allocation."""
        training_args = config.training_args
        if not getattr(training_args, "reward_optimization_overlap", False):
            return

        capability = cls.reward_optimization_overlap_contract
        if not isinstance(capability, RewardOptimizationOverlapContract):
            raise TypeError(
                f"trainer {cls.__name__}.reward_optimization_overlap_contract must be "
                "RewardOptimizationOverlapContract, received "
                f"{type(capability).__name__}: {capability!r}"
            )
        if not capability.supported:
            raise ValueError(
                f"trainer {cls.__name__} does not support train.reward_optimization_overlap"
            )
        if cls.execution_contract.feedback is not FeedbackMode.RUNTIME_REWARD:
            raise ValueError(
                "reward/optimization overlap requires execution feedback='runtime_reward', "
                f"received {cls.execution_contract.feedback.value!r} for {cls.__name__}"
            )

        mode = training_args.reward_optimization_overlap_mode
        if not capability.supports(mode):
            raise ValueError(
                f"trainer {cls.__name__} does not support reward overlap mode {mode!r}; "
                f"supported={capability.scheduling_modes!r}"
            )
        sampler_contract = get_sampler_layout_contract(config.data_args.sampler_type)
        if not capability.supports_sampler_group_placement(sampler_contract.group_placement):
            raise ValueError(
                f"trainer {cls.__name__} cannot use sampler "
                f"{config.data_args.sampler_type!r} for reward/optimization overlap: "
                f"group_placement={sampler_contract.group_placement!r}, "
                f"supported={capability.sampler_group_placements!r}"
            )
        geometry = cls.reward_optimization_overlap_geometry(config)
        expected_group_layout = {
            "rank_local": "rank_local",
            "global_batch": "cross_rank_sharded",
            "global_tile": "cross_rank_tiled",
            "subgroup_tile": "cross_rank_tiled",
        }[sampler_contract.group_placement]
        if geometry.group_layout != expected_group_layout:
            raise ValueError(
                f"trainer {cls.__name__} produced reward group layout "
                f"{geometry.group_layout!r} for sampler placement "
                f"{sampler_contract.group_placement!r}; expected {expected_group_layout!r}"
            )

        reward_configs = list(config.reward_args or [])
        if not reward_configs:
            raise ValueError("reward/optimization overlap requires at least one training reward")
        synchronous = tuple(cfg.name for cfg in reward_configs if not cfg.async_reward)
        if synchronous:
            raise ValueError(
                "reward/optimization overlap requires async_reward=true for every training "
                f"reward; synchronous rewards={synchronous!r}"
            )
        non_cpu = tuple(
            (cfg.name, str(cfg.device))
            for cfg in reward_configs
            if getattr(cfg.device, "type", cfg.device) != "cpu"
        )
        if non_cpu:
            raise ValueError(
                "reward/optimization overlap requires device='cpu' for every training reward "
                "client so reward CUDA work cannot contend with policy optimization; "
                f"non-CPU rewards={non_cpu!r}"
            )
        reducer_contract = resolve_feedback_reducer_contract(
            getattr(training_args, "advantage_aggregation", None),
            global_std=getattr(training_args, "global_std", True),
        )
        if not reducer_contract.supports_group_complete_streaming:
            raise ValueError(
                "reward/optimization overlap requires a group-relative feedback reducer "
                "without acquisition-wide statistics; "
                f"reducer={reducer_contract.name!r}, "
                f"combination_order={reducer_contract.combination_order!r}, "
                f"requires_acquisition_statistics="
                f"{reducer_contract.requires_acquisition_statistics}"
            )
        if training_args.num_inner_epochs != 1:
            raise ValueError("reward/optimization overlap requires train.num_inner_epochs=1")
        if training_args.shuffle_samples:
            raise ValueError("reward/optimization overlap requires train.shuffle_samples=false")

        accumulation_steps = training_args.gradient_accumulation_steps
        if type(accumulation_steps) is not int or accumulation_steps < 1:
            raise ValueError(
                "reward/optimization overlap requires resolved integer "
                f"gradient_accumulation_steps, received {accumulation_steps!r}"
            )
        optimizer_terms_per_batch = geometry.optimizer_terms_per_batch
        optimizer_batches_per_acquisition = training_args.num_batches_per_epoch
        if geometry.group_layout == "rank_local":
            local_samples = (
                training_args.num_batches_per_epoch * training_args.per_device_batch_size
            )
            if local_samples % training_args.group_size:
                raise ValueError(
                    "reward/optimization overlap requires rank-local acquisitions to contain "
                    "complete groups: "
                    f"samples={local_samples}, group_size={training_args.group_size}"
                )
            group_count = local_samples // training_args.group_size
            examples_per_group = geometry.optimizer_examples_per_group or training_args.group_size
            optimizer_examples = group_count * examples_per_group
            if optimizer_examples % training_args.per_device_batch_size:
                raise ValueError(
                    "reward/optimization overlap requires acquisition-wide optimizer "
                    "examples to form complete microbatches: "
                    f"optimizer_examples={optimizer_examples}, "
                    f"per_device_batch_size={training_args.per_device_batch_size}"
                )
            optimizer_batches_per_acquisition = (
                optimizer_examples // training_args.per_device_batch_size
            )
        if (
            optimizer_batches_per_acquisition * optimizer_terms_per_batch
        ) % accumulation_steps != 0:
            raise ValueError(
                "reward/optimization overlap requires every acquisition to close complete "
                "gradient accumulation windows: "
                f"optimizer_batches_per_acquisition={optimizer_batches_per_acquisition}, "
                f"optimizer_terms_per_batch={optimizer_terms_per_batch}, "
                f"gradient_accumulation_steps={accumulation_steps}"
            )

    @classmethod
    def reward_optimization_overlap_geometry(
        cls,
        config: Arguments,
    ) -> RewardTileGeometry:
        """Describe this objective's reward-group to optimizer mapping."""
        sampler_contract = get_sampler_layout_contract(config.data_args.sampler_type)
        group_layout = {
            "rank_local": "rank_local",
            "global_batch": "cross_rank_sharded",
            "global_tile": "cross_rank_tiled",
            "subgroup_tile": "cross_rank_tiled",
        }.get(sampler_contract.group_placement)
        if group_layout is None:
            raise ValueError(
                f"sampler {sampler_contract.name!r} does not expose bounded group-complete "
                "work units for reward/optimization overlap"
            )
        group_window_batches = None
        if group_layout != "rank_local":
            group_window_batches = sampler_contract.global_group_window_batches(
                num_replicas=get_world_size(),
                per_device_batch_size=config.training_args.per_device_batch_size,
                group_size=config.training_args.group_size,
                subgroup_size=(
                    config.data_args.sampler_subgroup_size
                    if sampler_contract.group_placement == "subgroup_tile"
                    else None
                ),
            )
        return RewardTileGeometry(
            group_layout=group_layout,
            optimizer_terms_per_batch=config.training_args.get_num_train_timesteps(config),
            group_window_batches=group_window_batches,
        )

    @classmethod
    def validate_adapter_class_execution_contract(cls, adapter_cls: type) -> None:
        """Reject statically unsupported dataset acquisition before model loading.

        Args:
            adapter_cls: Resolved model adapter class.
        """
        if cls.execution_contract.acquisition is not AcquisitionMode.DATASET:
            return
        if not isinstance(adapter_cls, type) or not issubclass(adapter_cls, BaseAdapter):
            raise TypeError(
                f"dataset trainer {cls.__name__} requires a BaseAdapter subclass, "
                f"received {adapter_cls!r}"
            )
        validator = getattr(adapter_cls, "validate_offline_output_capability", None)
        if not callable(validator):
            raise TypeError(
                f"adapter {adapter_cls.__name__} must define "
                "validate_offline_output_capability() for dataset acquisition"
            )
        validator()

    def _validate_adapter_execution_contract(self) -> None:
        """Validate the realized adapter on the selected acquisition path."""
        type(self).validate_adapter_class_execution_contract(type(self.adapter))

    def _initialize_snapshots(self) -> None:
        """Initialize optional trainer-owned parameter snapshots before state resume."""

    def _declared_runtime_child_names(self) -> Tuple[str, ...]:
        """Declare every child whose state must participate in exact resume.

        The declaration is configuration-derived and therefore available before
        heavyweight initialization. Concrete trainers may add class-level names and
        register matching objects during :meth:`_initialize_snapshots`.
        """
        algorithm_names = self._algorithm_runtime_child_names()
        reserved_names = {
            self._ADAPTER_EMA_RUNTIME_CHILD,
            self._ADAPTER_REFERENCE_RUNTIME_CHILD,
            MULTIROLE_RUNTIME_CHILD_NAME,
        }
        collisions = tuple(name for name in algorithm_names if name in reserved_names)
        if collisions:
            raise ValueError(
                "algorithm runtime child names collide with framework-reserved names: "
                f"collisions={collisions!r}, reserved={tuple(sorted(reserved_names))!r}"
            )

        names = []
        if self.training_args.ema_decay > 0:
            names.append(self._ADAPTER_EMA_RUNTIME_CHILD)
        if self.training_args.requires_ref_model and self.model_args.finetune_type == "full":
            names.append(self._ADAPTER_REFERENCE_RUNTIME_CHILD)
        if len(self._required_trainable_roles()) > 1:
            names.append(MULTIROLE_RUNTIME_CHILD_NAME)
        names.extend(algorithm_names)
        return tuple(names)

    def _algorithm_runtime_child_names(self) -> Tuple[str, ...]:
        """Return configuration-active trainer-owned checkpoint children.

        Most algorithms declare a fixed class-level tuple. Algorithms whose
        snapshots are conditional or data-driven may override this hook, while
        retaining a declaration that is computable before heavyweight setup.
        """
        return type(self).runtime_child_names

    def _realize_runtime_child_declarations(self) -> None:
        """Refresh declarations after algorithms materialize their concrete roles."""
        child_names = self._declared_runtime_child_names()
        if child_names == self.runtime_state.child_names:
            return
        self.runtime_state = TrainerRuntimeState(
            progress=self.runtime_state.progress,
            child_names=child_names,
        )

    def register_runtime_child(self, name: str, child: Any) -> None:
        """Register one trainer-owned child declared by ``runtime_child_names``.

        This hook lets an algorithm build a reference/EMA snapshot after distributed
        preparation while still attaching it only after an exact-resume payload has
        been fully preflighted and committed.

        Args:
            name: Configuration-declared runtime child name.
            child: Object implementing state, validation, and load methods.
        """
        declared_names = self._algorithm_runtime_child_names()
        if name not in declared_names:
            raise KeyError(
                f"trainer runtime child {name!r} was not declared for this "
                f"configuration; expected one of {declared_names!r}"
            )
        children = self.__dict__.setdefault("_trainer_runtime_children", {})
        if name in children:
            raise RuntimeError(f"trainer runtime child {name!r} is already registered")
        children[name] = child

    def _register_named_parameter_runtime_child(self, name: str) -> None:
        """Register one realized adapter named-parameter snapshot for exact resume."""
        named_parameters = getattr(self.adapter, "_named_parameters", None)
        if not isinstance(named_parameters, dict):
            raise TypeError(
                "adapter named-parameter snapshots must be stored as a dict, received "
                f"{type(named_parameters).__name__}"
            )
        if name not in named_parameters:
            raise KeyError(
                f"adapter named-parameter snapshot {name!r} was not initialized; "
                f"available snapshots={tuple(named_parameters)!r}"
            )
        child = getattr(named_parameters[name], "ema_wrapper", None)
        if child is None:
            raise TypeError(
                f"adapter named-parameter snapshot {name!r} has no checkpointable " "ema_wrapper"
            )
        self.register_runtime_child(name, child)

    @contextmanager
    def _suspend_adapter_state_resume(self) -> Iterator[None]:
        """Let adapter post-init realize children without loading state itself."""
        resume_path = self.model_args.resume_path
        resume_type = self.model_args.resume_type
        self.model_args.resume_path = None
        self.model_args.resume_type = None
        try:
            yield
        finally:
            self.model_args.resume_path = resume_path
            self.model_args.resume_type = resume_type

    def _runtime_checkpoint_children(self) -> Dict[str, Any]:
        """Return realized children in the immutable declaration order."""
        children: Dict[str, Any] = {}
        declared_names = self.runtime_state.child_names
        if self._ADAPTER_EMA_RUNTIME_CHILD in declared_names:
            children[self._ADAPTER_EMA_RUNTIME_CHILD] = self.adapter.ema_wrapper
        if self._ADAPTER_REFERENCE_RUNTIME_CHILD in declared_names:
            children[self._ADAPTER_REFERENCE_RUNTIME_CHILD] = self.adapter._ref_ema
        if MULTIROLE_RUNTIME_CHILD_NAME in declared_names:
            children[MULTIROLE_RUNTIME_CHILD_NAME] = self._multirole_checkpoint_state
        children.update(self.__dict__.get("_trainer_runtime_children", {}))

        missing = tuple(name for name in declared_names if children.get(name) is None)
        unexpected = tuple(name for name in children if name not in declared_names)
        if missing or unexpected:
            raise RuntimeError(
                "trainer runtime children do not match their declaration: "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )
        return {name: children[name] for name in declared_names}

    def _attach_runtime_children(self, children: Dict[str, Any]) -> None:
        """Attach every prevalidated child exactly once after state restoration."""
        if self._runtime_children_attached:
            raise RuntimeError("trainer runtime children are already attached")
        for name in self.runtime_state.child_names:
            self.runtime_state.attach_child(name, children[name])
        self._runtime_children_attached = True

    def _validate_runtime_checkpoint_invariants(
        self,
        progress: TrainingProgress,
        child_states: Dict[str, Any],
    ) -> None:
        """Validate cross-child counter invariants before prepared-state mutation."""
        if MULTIROLE_RUNTIME_CHILD_NAME in child_states:
            self._multirole_checkpoint_state.validate_runtime_progress(
                progress,
                child_states[MULTIROLE_RUNTIME_CHILD_NAME],
            )

    def _validate_distributed_runtime_children(self) -> None:
        """Reject auxiliary state that lacks a distributed-aware gather contract."""
        distributed_type = getattr(getattr(self, "accelerator", None), "distributed_type", None)
        if distributed_type is not DistributedType.FSDP:
            return
        unsafe_children = [
            name
            for name in (
                self._ADAPTER_EMA_RUNTIME_CHILD,
                self._ADAPTER_REFERENCE_RUNTIME_CHILD,
            )
            if name in self.runtime_state.child_names
        ]
        unsafe_children.extend(
            name
            for name in self._algorithm_runtime_child_names()
            if name in self.runtime_state.child_names
        )
        registry = getattr(self.adapter, "component_variant_registry", None)
        snapshots = getattr(registry, "_snapshots", {})
        if snapshots:
            unsafe_children.append("multirole_variant_snapshots")
        if unsafe_children:
            raise RuntimeError(
                "exact state checkpointing under FSDP requires a "
                "distributed-aware gather/restore implementation for auxiliary tensors; "
                f"unsupported runtime children={tuple(unsafe_children)!r}. Model-only "
                "checkpoints remain supported."
            )

    def _validate_runtime_child_coverage(self) -> None:
        """Reject adapter snapshots that an exact checkpoint would silently omit."""
        named_parameters = getattr(self.adapter, "_named_parameters", {})
        if not named_parameters:
            return
        if not isinstance(named_parameters, dict):
            raise TypeError(
                "adapter named-parameter snapshots must be stored as a dict, received "
                f"{type(named_parameters).__name__}"
            )
        tracked_children = self._runtime_checkpoint_children()
        tracked_identities = {id(child) for child in tracked_children.values()}
        untracked_names = tuple(
            name
            for name, info in named_parameters.items()
            if id(getattr(info, "ema_wrapper", None)) not in tracked_identities
        )
        if untracked_names:
            raise RuntimeError(
                "exact state checkpointing would omit adapter named-parameter snapshots "
                f"{untracked_names!r}; declare runtime_child_names and register each wrapper "
                "during _initialize_snapshots(), or save model weights only"
            )

    def _initialize_adapter_runtime(self) -> None:
        """Run adapter post-init while deferring only exact prepared-state loading."""
        state_resume = bool(self.model_args.resume_path and self.model_args.resume_type == "state")
        if state_resume:
            # Reject declaration-known FSDP auxiliary state before allocating late
            # EMA/reference snapshots. Variant snapshots are checked again after
            # their algorithm hook materializes them.
            self._validate_distributed_runtime_children()
            # BaseAdapter.post_init historically performs load-before-EMA. Temporarily
            # suppress only that load so the same hook still realizes all late children;
            # the trainer then owns the preflight/load/commit boundary below.
            with self._suspend_adapter_state_resume():
                self.adapter.post_init()
        else:
            self.adapter.post_init()

    @staticmethod
    def _resolve_exact_state_checkpoint_path(path: str) -> str:
        """Resolve exact-state input without an internal distributed barrier.

        Rank-local resolution failures are synchronized by the caller before any
        prepared state mutates. The adapter's general resolver intentionally ends in
        a barrier after an HF download, which is useful for ordinary weight loading
        but would hide an asymmetric failure from that preflight error gather.
        """
        path = os.path.expanduser(path)
        force_hf = path.startswith(HF_PATH_PREFIX)
        if not force_hf and os.path.exists(path):
            return path
        repo_id, subfolder, revision = parse_hf_checkpoint_path(path)
        return download_hf_checkpoint(repo_id, subfolder, revision)

    def _finalize_adapter_runtime(self) -> None:
        """Execute exact-resume preflight/load/commit after all children exist."""
        state_resume = bool(self.model_args.resume_path and self.model_args.resume_type == "state")
        if state_resume:
            children: Dict[str, Any] = {}
            resume_path = ""
            preflight_error = None
            try:
                children = self._runtime_checkpoint_children()
                self._validate_distributed_runtime_children()
                resume_path = self._resolve_exact_state_checkpoint_path(self.model_args.resume_path)
                self.runtime_state.validate_load(
                    resume_path,
                    children=children,
                    invariant_validator=self._validate_runtime_checkpoint_invariants,
                    expected_process_index=self.accelerator.process_index,
                    expected_device_type=self.accelerator.device.type,
                )
            except Exception as error:
                preflight_error = error
            self._synchronize_checkpoint_phase_error("resume preflight", preflight_error)

            core_load_error = None
            try:
                # The public adapter wrapper adds an unconditional trailing barrier.
                # Invoke the already-resolved prepared-state primitive directly so a
                # rank-local exception can reach the error gather below instead of
                # leaving successful peers blocked at that barrier.
                self.adapter._load_training_state(resume_path)
            except Exception as error:
                core_load_error = error
            self._synchronize_checkpoint_phase_error(
                "Accelerator artifact load",
                core_load_error,
            )
            runtime_commit_error = None
            try:
                self.runtime_state.commit_validated_load()
                self._attach_runtime_children(children)
            except Exception as error:
                runtime_commit_error = error
            self._synchronize_checkpoint_phase_error(
                "runtime child commit",
                runtime_commit_error,
            )
            self._exact_resume_source_checkpoint = self._canonical_checkpoint_path(resume_path)
            self._exact_resume_boundary_pending = (
                type(self).execution_contract.acquisition is AcquisitionMode.GENERATION
            )
        else:
            children = self._runtime_checkpoint_children()
            self._attach_runtime_children(children)

    @staticmethod
    def _canonical_checkpoint_path(path: str) -> str:
        """Return a symlink-resolved absolute checkpoint identity."""
        return os.path.realpath(os.path.abspath(os.path.expanduser(os.fspath(path))))

    def should_continue_training(self) -> bool:
        """Continue until the active acquisition cycle reaches ``max_epochs``."""
        m = self.training_args.max_epochs
        if m is None or m < 0:
            return True
        return self.cycle_index < m

    def accumulate_gradients(self):
        """Context manager for gradient accumulation over the single prepared root.

        Centralizes ``accelerator.accumulate(self.model_bundle)`` so trainers do
        not couple to the prepared-root identity: ``self.model_bundle`` is the one
        object DDP/FSDP/DeepSpeed wraps, and accumulation must always target it.

        Usage::

            with self.accumulate_gradients():
                ...  # forward / loss / backward / step
        """
        return self.accelerator.accumulate(self.model_bundle)

    def log_data(self, data: Dict[str, Any], step: int):
        """Log data using the initialized logger."""
        if self.logger is not None:
            self.logger.log_data(data, step=step)

        # Print summary to console
        if self.accelerator.is_local_main_process:
            metrics = {
                k: v
                for k, v in ((k, LogFormatter.to_scalar(v)) for k, v in data.items())
                if v is not None
            }
            if metrics:
                parts = [f"[Step {step:04d} | Epoch {self.epoch:03d}]"]
                parts.extend(
                    (
                        f"{k}={int(v)}"
                        if isinstance(v, int) or (isinstance(v, float) and v.is_integer())
                        else f"{k}={v:.4f}"
                    )
                    for k, v in metrics.items()
                )
                logger.info(" ".join(parts))

    def _init_logging_backend(self):
        """Initialize logging backend if specified."""
        if self.accelerator.is_main_process:
            self.logger = load_logger(self.config)
        else:
            self.logger = None
        self.accelerator.wait_for_everyone()

    def _init_reward_model(self) -> Tuple[Dict[str, BaseRewardModel], Dict[str, BaseRewardModel]]:
        """Initialize reward model from configuration."""

        # If DeepSpeed ZeRO-3 is enabled, the reward model will be somehow sharded.
        # We need to disable ZeRO-3 init context when loading the model to avoid issues
        # This remains unsupported even with the context manager; do not use ZeRO-3.
        # A possible solution: call DeepSpeed's `zero.GatheredParameters` manually inside the
        # reward model's `forward`.

        # Collect training dataset names so MultiRewardLoader can pre-compute
        # the per-source reward routing used by the runtime reward gate
        # and any future trainer that needs "which rewards apply to source S?"
        # lookups.  Training is the primary path; eval names follow.
        training_dataset_names = (
            [td.name for td in self.config.data_args.training_datasets]
            if self.config.data_args.training_datasets
            else []
        )
        # Collect eval dataset names for per-eval-dataset reward routing
        # (mirror of the training-side bookkeeping).
        eval_dataset_names = (
            [ed.name for ed in self.config.data_args.eval_datasets]
            if self.config.data_args.eval_datasets
            else []
        )

        self.reward_loader = MultiRewardLoader(
            reward_args=self.config.reward_args,
            accelerator=self.accelerator,
            training_dataset_names=training_dataset_names,
            eval_reward_args=self.config.eval_reward_args,
            eval_dataset_names=eval_dataset_names,
            load_context=lambda: self.load_coordinator.load_scope(ComponentRole.REWARD),
        ).load()
        # Get training & eval reward models
        self.reward_models = self.reward_loader.get_training_reward_models()
        self.eval_reward_models = self.reward_loader.get_eval_reward_models()
        train_reward_configs = self.reward_loader.get_reward_configs("train")
        self.reward_processor: Optional[RewardProcessor] = None
        self.reward_buffer: Optional[RewardBuffer] = None
        self.advantage_processor: Optional[AdvantageProcessor] = None
        self.group_coordinator: Optional[GroupCoordinator] = None

        # Only runtime-feedback algorithms own training-side reward groups. Dataset
        # acquisition intentionally leaves sampler_type="auto" because its official
        # DistributedSampler has no reward-group layout to resolve.
        if type(self).execution_contract.feedback is FeedbackMode.RUNTIME_REWARD:
            sampler_layout = get_sampler_layout_contract(self.config.data_args.sampler_type)
            self.group_coordinator = GroupCoordinator(
                self.accelerator,
                sampler_type=self.config.data_args.sampler_type,
                subgroup_size=(
                    self.config.data_args.sampler_subgroup_size
                    if sampler_layout.group_placement == "subgroup_tile"
                    else None
                ),
            )
            group_on_same_rank = sampler_layout.groups_are_rank_local
            async_groupwise_rewards = tuple(
                name
                for name, model in self.reward_models.items()
                if isinstance(model, GroupwiseRewardModel)
                and train_reward_configs[name].async_reward
            )
            if async_groupwise_rewards and not group_on_same_rank:
                raise ValueError(
                    "asynchronous groupwise rewards require a rank-local sampler layout; "
                    f"sampler_type={self.config.data_args.sampler_type!r}, "
                    f"groupwise_rewards={async_groupwise_rewards!r}"
                )
            self.reward_processor = RewardProcessor(
                accelerator=self.accelerator,
                reward_models=self.reward_models,
                reward_configs=train_reward_configs,
                tokenizer=self.adapter.tokenizer,  # For prompt encoding/decoding,
                group_on_same_rank=group_on_same_rank,
                verbose=self.log_args.verbose,
                group_coordinator=self.group_coordinator,
            )
            self.reward_buffer = RewardBuffer(
                self.reward_processor,
                self.training_args.group_size,
            )

            # `cfg.weight` is a Dict[str, float] after `_resolve_reward_weights`,
            # so reward_weights is Dict[reward_name, Dict[dataset_name, float]].
            self.advantage_processor = AdvantageProcessor(
                accelerator=self.accelerator,
                reward_weights={name: cfg.weight for name, cfg in train_reward_configs.items()},
                group_size=self.training_args.group_size,
                global_std=getattr(self.training_args, "global_std", True),
                sampler_type=self.config.data_args.sampler_type,
                verbose=self.log_args.verbose,
                source_id_to_name=self.config.data_args.source_id_to_name,
                group_coordinator=self.group_coordinator,
            )

        # Per-eval-dataset reward processors and buffers.  Eval is now
        # always per-dataset (the legacy single `eval_reward_buffer`
        # was retired with the unified `evaluate()` path); the loop
        # below builds one processor + buffer per eval-eligible entry,
        # which `evaluate()` then iterates.
        self.eval_dataset_reward_processors: Dict[str, RewardProcessor] = {}
        self.eval_dataset_reward_buffers: Dict[str, RewardBuffer] = {}
        self._eval_dataset_configs: Dict[str, "DatasetArguments"] = {}

        if self.config.data_args.eval_datasets:
            self._eval_dataset_configs = {ed.name: ed for ed in self.config.data_args.eval_datasets}
            for ed in self.config.data_args.eval_datasets:
                ds_models = self.reward_loader.get_eval_dataset_reward_models(ed.name)
                ds_configs = self.reward_loader.get_eval_dataset_reward_configs(ed.name)
                if ds_models:
                    ds_processor = RewardProcessor(
                        accelerator=self.accelerator,
                        reward_models=ds_models,
                        reward_configs=ds_configs,
                        tokenizer=self.adapter.tokenizer,
                        # Evaluation currently computes pointwise rewards only and its
                        # ordinary DataLoader is independent of the training sampler.
                        group_on_same_rank=False,
                        verbose=self.log_args.verbose,
                    )
                    self.eval_dataset_reward_processors[ed.name] = ds_processor
                    self.eval_dataset_reward_buffers[ed.name] = RewardBuffer(
                        ds_processor,
                        self.training_args.group_size,
                    )

        return self.reward_models, self.eval_reward_models

    def _init_dataloader(
        self,
    ) -> Tuple[Optional[Union[DataLoader, "MultiSourceTrainDataLoader"]], Dict[str, DataLoader]]:
        """Build train and eval dataloaders.

        Returns:
            Tuple of (train_dataloader, eval_dataloaders_by_name).
        """
        self.load_coordinator.load_components(
            self.adapter.preprocessing_modules,
            device=self.accelerator.device,
        )

        build_train_dataloader = getattr(self, "_build_train_dataloader", None)
        if build_train_dataloader is None:
            # A few lifecycle tests intentionally call this shared method on a
            # lightweight structural host. Preserve that supported boundary while
            # real trainer subclasses continue to override the acquisition seam.
            dataloader, train_dataloaders_by_source = BaseTrainer._build_train_dataloader(self)
        else:
            dataloader, train_dataloaders_by_source = build_train_dataloader()
        self.train_dataloaders_by_source: Dict[str, DataLoader] = train_dataloaders_by_source

        eval_dataloaders = get_eval_dataloaders(
            eval_datasets=self.config.data_args.eval_datasets,
            config=self.config,
            accelerator=self.accelerator,
            preprocess_func=self.adapter.preprocess_func,
        )

        self.adapter.off_load_components(
            components=self.adapter.preprocessing_modules,
        )

        self.accelerator.wait_for_everyone()

        return dataloader, eval_dataloaders

    def _build_train_dataloader(
        self,
    ) -> Tuple[Optional[Union[DataLoader, "MultiSourceTrainDataLoader"]], Dict[str, DataLoader]]:
        """Build the acquisition-specific training dataloader.

        Returns:
            Training loader and its per-source loader mapping.

        Note:
            The default preserves grouped online rollout loading. Dataset-based
            trainers override this hook with the finite offline loader builder; the
            surrounding preprocessing lifecycle remains shared.
        """
        return get_train_dataloader(
            config=self.config,
            accelerator=self.accelerator,
            preprocess_func=self.adapter.preprocess_func,
        )

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Build the single optimizer root, its groups ordered and tagged by role.

        All-AdamW runs get one ``torch.optim.AdamW``. A role that selects Muon
        contributes two groups instead of one, since Muon takes only matrices, and
        the root becomes a ``CompositeOptimizer``.
        """
        registry = self.adapter.component_variant_registry
        trainable_role_names = registry.variant_names
        role_configs = self._role_optimizer_configs()
        configured_role_names = tuple(config.role_name for config in role_configs)
        if configured_role_names != trainable_role_names:
            raise ValueError(
                "expected role optimizer configs to exactly match declared trainable roles "
                f"{trainable_role_names!r}, received {configured_role_names!r}"
            )

        optimizer_args = tuple(
            self._optimizer_args_for_role(config.role_name) for config in role_configs
        )
        self._validate_optimizer_backend(optimizer_args)
        parameters_by_name = {}
        for config in role_configs:
            parameters = registry.parameters(config.role_name)
            if not parameters:
                raise ValueError(
                    f"expected trainable role {config.role_name!r} to own optimizer "
                    "parameters, received none"
                )
            parameters_by_name[config.role_name] = parameters

        self.optimizer = build_optimizer(optimizer_args, parameters_by_name)

        # Muon splits one role across two groups (its matrices and the AdamW
        # remainder), so ownership is recorded per role rather than per group.
        self.optimization_roles = {}
        group_ids_by_role: Dict[str, List[int]] = {}
        for group_id, group in enumerate(self.optimizer.param_groups):
            group_ids_by_role.setdefault(group["role_name"], []).append(group_id)
        for config in role_configs:
            self.optimization_roles[config.role_name] = OptimizationRole(
                config=config,
                parameters=parameters_by_name[config.role_name],
                optimizer_group_ids=tuple(group_ids_by_role[config.role_name]),
            )
        return self.optimizer

    def _optimizer_args_for_role(self, role_name: str) -> OptimizerArguments:
        """Return the optimizer configuration for one trainable role.

        A single-policy run has one configuration and need not name it, which is
        what every existing config file relies on.

        Args:
            role_name: Trainable role to configure.

        Returns:
            The matching optimizer arguments.

        Raises:
            ValueError: If no configuration matches and none can be defaulted.
        """
        configured = self.config.optimizer_args.get_by_name(role_name)
        if configured is not None:
            return configured
        if len(self.config.optimizer_args) == 1:
            # The lone entry configures whichever role this run trains, whatever the
            # file happens to call it. Adopt the role name: `build_optimizer` looks
            # parameters up by `OptimizerArguments.name`, and that lookup is keyed by
            # role, so returning the entry unrenamed finds no parameters at all.
            return replace(self.config.optimizer_args[0], name=role_name)
        available = tuple(config.name for config in self.config.optimizer_args)
        raise ValueError(
            f"expected an optimizer configuration named {role_name!r} under `optimizers`, "
            f"received {available!r}"
        )

    def _role_optimizer_configs(self) -> Tuple[RoleOptimizerConfig, ...]:
        """Build role configs from nested arguments or legacy flat arguments."""
        required_roles = BaseTrainer._required_trainable_roles(self)
        if getattr(self.training_args, "role_update_plan", None) is not None:
            update_plan = BaseTrainer._role_update_plan(self)
            plan_roles = {phase.role_name for phase in update_plan.phases}
            if plan_roles != set(required_roles):
                raise ValueError(
                    "expected role update plan roles to exactly match required trainable roles "
                    f"{required_roles!r}, received {tuple(plan_roles)!r}"
                )

        return tuple(
            BaseTrainer._role_optimizer_config_from_args(
                role_name, self._optimizer_args_for_role(role_name)
            )
            for role_name in required_roles
        )

    @staticmethod
    def _role_optimizer_config_from_args(
        role_name: str, optimizer_args: OptimizerArguments
    ) -> RoleOptimizerConfig:
        """Project one optimizer configuration onto the coordinator's view of a role.

        The coordinator only needs the clip norm and the update cadence; the moment
        parameters are carried along so a role's configuration stays inspectable in
        one place. A Muon role reports its AdamW-half moments, which is what its
        non-matrix parameters actually use.
        """
        betas = getattr(optimizer_args, "betas", None)
        eps = getattr(optimizer_args, "eps", None)
        if betas is None:
            betas = getattr(optimizer_args, "fallback_betas")
            eps = getattr(optimizer_args, "fallback_eps")
        return RoleOptimizerConfig(
            role_name=role_name,
            learning_rate=optimizer_args.learning_rate,
            adam_betas=betas,
            adam_weight_decay=optimizer_args.weight_decay,
            adam_epsilon=eps,
            max_grad_norm=optimizer_args.max_grad_norm,
            update_frequency=optimizer_args.update_frequency,
        )

    def _finish_role_microbatch(self) -> bool:
        """Finish a role microbatch, advancing the public step for the primary role."""
        role_name = self.role_optimization.active_role_name
        stepped = self.role_optimization.finish_microbatch()
        if stepped and role_name == self._primary_role():
            self.step += 1
        return stepped

    def _rebind_prepared_optimization_roles(self) -> None:
        """Rebuild role ownership from prepared optimizer parameter identities."""
        optimizer_groups = self.optimizer.param_groups
        expected_group_ids = tuple(
            group_id
            for role in self.optimization_roles.values()
            for group_id in role.optimizer_group_ids
        )
        if tuple(sorted(expected_group_ids)) != tuple(range(len(optimizer_groups))):
            raise ValueError(
                "expected optimization roles to exhaust prepared optimizer groups "
                f"{tuple(range(len(optimizer_groups)))!r}, received {expected_group_ids!r}"
            )

        rebound_roles = {}
        for role_name, role in self.optimization_roles.items():
            prepared_parameters = []
            for group_id in role.optimizer_group_ids:
                group = optimizer_groups[group_id]
                prepared_role_name = group.get("role_name")
                if prepared_role_name != role_name:
                    raise ValueError(
                        f"prepared optimizer group {group_id} expected role_name "
                        f"{role_name!r}, received {prepared_role_name!r}"
                    )
                group_parameters = tuple(group["params"])
                if not group_parameters:
                    raise ValueError(
                        f"prepared optimizer group {group_id} for role {role_name!r} "
                        "expected at least one parameter, received none"
                    )
                prepared_parameters.extend(group_parameters)
            rebound_roles[role_name] = OptimizationRole(
                config=role.config,
                parameters=tuple(prepared_parameters),
                optimizer_group_ids=role.optimizer_group_ids,
                step=role.step,
                scheduler=role.scheduler,
            )
        self.optimization_roles = rebound_roles

    def _init_prepared_role_optimization(self) -> None:
        """Bind prepared identities and construct the role coordinator."""
        BaseTrainer._rebind_prepared_optimization_roles(self)
        self.role_optimization = RoleOptimizationCoordinator(
            accelerator=self.accelerator,
            model_bundle=self.model_bundle,
            optimizer=self.optimizer,
            roles=self.optimization_roles,
        )

    def _role_update_plan(self) -> RoleUpdatePlan:
        """Return the ordered role plan used for checkpoint compatibility."""
        configured_plan_builder = getattr(self.training_args, "role_update_plan", None)
        if configured_plan_builder is not None:
            configured_plan = configured_plan_builder()
            if not isinstance(configured_plan, RoleUpdatePlan):
                raise TypeError(
                    "expected training_args.role_update_plan() to return RoleUpdatePlan, "
                    f"received {type(configured_plan).__name__}: {configured_plan!r}"
                )
            return configured_plan
        return RoleUpdatePlan(
            phases=tuple(RolePhase(role_name) for role_name in self.optimization_roles)
        )

    def _declare_model_variants(self) -> None:
        """Declare the component variants this algorithm trains, before ``prepare``.

        Roles are the trainer's vocabulary, not the adapter's: an algorithm that
        trains a generator against a fake score names its own variants here and
        keeps the meaning of those names to itself. A single-policy algorithm needs
        only the base variant, which is what this default declares.
        """
        self.adapter.declare_component_variants(self._required_trainable_roles())

        # A weight-only resume already ran while the adapter was being built, when the
        # roles did not exist yet and only the primary artifact could be placed. Now
        # that they do, restore the rest -- a generator resumed beside an untouched fake
        # score trains against the wrong critic and reports nothing wrong.
        resume_path = getattr(self.adapter.model_args, "resume_path", None)
        resume_type = getattr(self.adapter.model_args, "resume_type", None)
        if resume_path and resume_type != "state":
            self.adapter.restore_training_roles(resume_path)
            # PEFT/full checkpoint loaders may materialize restored role weights
            # in fp32. FSDP1 flattens each wrapped block and rejects mixed dtypes,
            # so reapply the adapter's declared trainable dtype before prepare.
            self.adapter.align_component_variant_dtypes()

    def _required_trainable_roles(self) -> Tuple[str, ...]:
        """Return every role this run trains, the one owning the base weights first.

        Once variants are declared the adapter is the source of truth, so a trainer
        that declares them directly is described correctly without also restating
        them in config. Before declaration the algorithm's ``TrainingArguments``
        answer; a single-policy algorithm names none and gets one base variant.
        """
        training_args = getattr(self, "training_args", None)
        declared = getattr(training_args, "required_trainable_roles", None)
        if declared:
            return tuple(declared)
        registry = getattr(getattr(self, "adapter", None), "component_variant_registry", None)
        if isinstance(registry, ComponentVariantRegistry):
            return registry.variant_names
        return (DEFAULT_BASE_VARIANT,)

    def _load_inference_components(self, trainable_module_names: List[str]):
        """
        Load non-trainable components needed at runtime to the accelerator device.

        Trainable modules are already on-device via `accelerator.prepare()`.
        This loads the remaining modules required for inference and,
        when preprocessing is disabled, also loads encoding components
        that would otherwise stay offloaded.
        """
        prepared_names = set(trainable_module_names)

        modules_to_load = list(self.adapter.inference_modules)

        execution_contract = getattr(
            type(self),
            "execution_contract",
            ONLINE_EXECUTION_CONTRACT,
        )
        if execution_contract.acquisition is AcquisitionMode.DATASET:
            modules_to_load.extend(self.adapter.condition_state_encoding_modules)
            modules_to_load.extend(self.adapter.output_state_encoding_modules)

        if not self.config.data_args.enable_preprocess:
            modules_to_load.extend(self.adapter.preprocessing_modules)

        # Resolve group names → concrete names, then deduplicate & exclude prepared
        resolved = self.adapter._resolve_component_names(modules_to_load)
        resolved = [m for m in resolved if m not in prepared_names]

        if resolved:
            self.load_coordinator.load_components(
                resolved,
                device=self.accelerator.device,
            )

    def _validate_paradigm_dynamics(self) -> None:
        """Reject a scheduler whose dynamics the declared paradigm cannot use.

        A coupled algorithm differentiates a stochastic transition, so an ODE
        scheduler leaves it with no transition density and silently wrong policy
        gradients (``constraints.md`` #7). Only the coupled path was guarded, and
        only lazily at the point a transition scale was first needed, which is
        after a run has already started.
        """
        if type(self).paradigm != "coupled":
            return
        scheduler_group = getattr(self.adapter, "scheduler_group", None)
        if scheduler_group is None:
            return
        stochastic = ("Flow-SDE", "Dance-SDE", "CPS")
        for component in scheduler_group.names:
            dynamics_type = scheduler_group[component].dynamics_type
            if dynamics_type not in stochastic:
                raise ValueError(
                    f"coupled algorithm {type(self).__name__} requires stochastic dynamics, "
                    f"received dynamics_type={dynamics_type!r} for component {component!r}; "
                    f"expected one of {stochastic}. Either configure an SDE scheduler or use a "
                    "decoupled algorithm (see constraints #7)."
                )

    def _apply_backend_checkpointing_constraints(self) -> None:
        """Apply the shared owner plan to an adapter that may already be realized."""
        disable_realized_model_checkpointing = configure_checkpointing_backend_plan(
            self.accelerator,
            self.training_args,
        )
        if disable_realized_model_checkpointing:
            self.adapter.disable_gradient_checkpointing()

    def _initialization(self):
        self._validate_paradigm_dynamics()
        configure_deepspeed_micro_batch_size(
            self.accelerator, self.training_args.per_device_batch_size
        )

        self.load_coordinator.bootstrap_targets()

        # Init dataloader, then materialize every live component variant before
        # optimizer and distributed bundle construction.
        self.dataloader, eval_dataloaders = self._init_dataloader()
        self._declare_model_variants()
        self._apply_backend_checkpointing_constraints()
        self.optimizer = self._init_optimizer()

        # Bundle ALL target components (trainable + frozen-but-shardable, e.g.
        # Wan2.2's inactive transformer) into ONE nn.Module so accelerate wraps a
        # single root. DeepSpeed (one engine) and FSDP2 (one root) cannot wrap
        # multiple models, so PPO (policy + critic) and Wan2.2 (shard both, train
        # one) require this. The optimizer/EMA/ref still operate on the
        # requires_grad subset via `get_trainable_parameters()`; frozen members
        # are sharded for memory but never receive gradient.
        canonical_bundle_names = list(self.adapter.target_module_map.keys())
        variant_registry = self.adapter.component_variant_registry
        bundle_members = variant_registry.bundle_members()
        model_bundle = ModelBundle(bundle_members)
        self._unprepared_optimizer_group_roles = tuple(
            group["role_name"] for group in self.optimizer.param_groups
        )

        eval_dataloader_names = list(eval_dataloaders.keys())
        eval_dataloader_list = [eval_dataloaders[n] for n in eval_dataloader_names]

        # One prepare call -> one DDP/FSDP/DeepSpeed root for the whole bundle.
        # (Parameter dtypes -- incl. the FSDP2 uniform-fp32 requirement for sharded trained
        # components -- are already handled in the adapter's `_mix_precision`.)
        prepared = self.load_coordinator.prepare(
            model_bundle,
            self.optimizer,
            *eval_dataloader_list,
        )
        self.model_bundle = prepared[0]
        self.optimizer = prepared[1]
        inner_bundle = self.accelerator.unwrap_model(self.model_bundle)
        # FSDP2 replaces original Parameters with DTensor-backed Parameters while
        # preserving stable module/parameter names. Rebind the variant registry before
        # any PEFT adapter switch attempts to restore trainability.
        variant_registry.rebind_parameters(inner_bundle.members)
        BaseTrainer._init_prepared_role_optimization(self)
        BaseTrainer._validate_multirole_backend(self)
        BaseTrainer._validate_trainable_parameters_survived_prepare(self)
        prepared_eval_dataloaders = prepared[2:]
        self.eval_dataloaders: Dict[str, DataLoader] = dict(
            zip(eval_dataloader_names, prepared_eval_dataloaders)
        )

        # Install routing proxies so adapter forwards (`self.transformer(...)`,
        # `self.transformer_2(...)`, ...) dispatch through the prepared root --
        # required for DDP's reducer / FSDP's gather / the DeepSpeed engine --
        # while attribute access delegates to the inner member.
        for name in canonical_bundle_names:
            self.adapter.set_component(
                name,
                RoutedComponentProxy(
                    self.model_bundle,
                    name,
                    variant_registry,
                    inner_bundle.members,
                ),
            )

        # Load inference modules, excluding all bundle members (already prepared).
        self._load_inference_components(canonical_bundle_names)

        # Build + validate acceleration plugins. Persistent stage='both' accelerators
        # are *applied* later via _apply_shared_acceleration(), after post_init()
        # finishes any state-resume / EMA / reference setup.
        self._init_acceleration()

        # Initialize reward model
        self._init_reward_model()

    def _init_acceleration(self):
        """Build and validate acceleration plugins from ``config.acceleration_args``.

        Two independent slots, each an **ordered list** (both empty by default).
        List order is the application order:

        * ``shared`` — persistent ``stage='both'`` accelerators (e.g.
          ``attention_backend`` then ``torch_compile``) applied to both rollout and
          the training forward. Only built/validated here; they are *applied* later by
          :meth:`_apply_shared_acceleration` (after ``post_init`` finishes
          state-resume / EMA / reference setup), so they transform the final weights.
        * ``rollout`` — accelerators applied per-epoch in :meth:`generate_samples`
          via :meth:`~BaseAccelerator.rollout_context`; may be lossy.

        Each accelerator is validated against this trainer's ``paradigm`` before
        use (fail-fast, ``constraints.md`` #26).
        """
        accel_args = self.config.acceleration_args
        self.shared_accelerators: List[BaseAccelerator] = []
        self.rollout_accelerators: List[BaseAccelerator] = []

        trainer_name = type(self).__name__
        paradigm = type(self).paradigm

        for spec in accel_args.shared:
            accelerator = build_accelerator(spec.name, spec.params)
            validate_accelerator(
                accelerator, slot="shared", paradigm=paradigm, trainer_name=trainer_name
            )
            self.shared_accelerators.append(accelerator)

        for spec in accel_args.rollout:
            accelerator = build_accelerator(spec.name, spec.params)
            validate_accelerator(
                accelerator, slot="rollout", paradigm=paradigm, trainer_name=trainer_name
            )
            self.rollout_accelerators.append(accelerator)
            if self.accelerator.is_main_process:
                logger.info(
                    "Acceleration: rollout accelerator '%s' (safety=%s) enabled.",
                    spec.name,
                    accelerator.safety,
                )

    def _apply_shared_acceleration(self) -> None:
        """Apply persistent ``stage='both'`` accelerators in config order.

        Called from ``__init__`` AFTER ``adapter.post_init()`` so transforms wrap the
        final weights — i.e. after ``accelerator.prepare``, any ``state`` checkpoint
        resume, and EMA / reference-parameter snapshotting.

        Each entry's ``setup`` runs in list order, so a config that lists
        ``attention_backend`` before ``torch_compile`` sets the backend first and
        then compiles the graph capturing it. In-place compilation
        (``nn.Module.compile`` / ``compile_repeated_blocks``) preserves parameter
        identity and ``state_dict`` keys, so checkpointing and the ``copy_``-based
        EMA / ref / named-parameter swaps stay correct.
        """
        for accelerator in self.shared_accelerators:
            accelerator.setup(self.adapter)
            if self.accelerator.is_main_process:
                logger.info(
                    "Acceleration: shared accelerator '%s' (safety=%s) applied to adapter.",
                    type(accelerator).__name__,
                    accelerator.safety,
                )

    @contextmanager
    def _rollout_acceleration(self) -> Iterator[None]:
        """Nest every rollout accelerator's context (first in list = outermost).

        A no-op when no rollout accelerator is configured.
        """
        with ExitStack() as stack:
            for accelerator in self.rollout_accelerators:
                stack.enter_context(accelerator.rollout_context(self.adapter))
            yield

    @staticmethod
    def _patch_deepspeed_autocast(accelerator):
        """Patch DeepSpeed >=0.17.2 to allow external torch.autocast contexts.

        In v0.17.2+, engine.forward() calls validate_nested_autocast() which
        raises AssertionError if torch.autocast is active outside the engine,
        then wraps the forward with torch.autocast(enabled=torch_autocast_enabled).
        When torch_autocast is not configured (the default for bf16 built-in
        mixed-precision), this inner context uses enabled=False, which explicitly
        *disables* any outer autocast and causes dtype mismatches.

        This patch makes the engine transparent to an outer autocast context:
        validate_nested_autocast becomes a no-op, and torch_autocast_enabled /
        torch_autocast_dtype fall through to the active torch.autocast state so
        the engine re-enables (rather than disables) autocast during forward.
        """
        if getattr(accelerator.state, "deepspeed_plugin", None) is None:
            return

        try:
            import deepspeed.runtime.torch_autocast as _ds_ac
            from deepspeed.runtime.engine import DeepSpeedEngine
        except ImportError:
            return

        if getattr(DeepSpeedEngine, "_ff_autocast_patched", False):
            return

        if hasattr(_ds_ac, "validate_nested_autocast"):
            _ds_ac.validate_nested_autocast = lambda engine: None

        if hasattr(DeepSpeedEngine, "torch_autocast_enabled"):
            _orig_enabled = DeepSpeedEngine.torch_autocast_enabled
            _orig_dtype = DeepSpeedEngine.torch_autocast_dtype

            def _patched_enabled(self):
                return _orig_enabled(self) or torch.is_autocast_enabled()

            def _patched_dtype(self):
                if not _orig_enabled(self) and torch.is_autocast_enabled():
                    return torch.get_autocast_gpu_dtype()
                return _orig_dtype(self)

            DeepSpeedEngine.torch_autocast_enabled = _patched_enabled
            DeepSpeedEngine.torch_autocast_dtype = _patched_dtype

        DeepSpeedEngine._ff_autocast_patched = True

    def start(self) -> None:
        """Run the training loop until the configured budget is exhausted.

        Generation acquisition retains the existing pre-rollout checkpoint,
        evaluation, and cycle-level EMA cadence. Dataset acquisition exhausts one
        finite official distributed loader before incrementing ``data_epoch`` and
        publishing post-epoch boundaries. Optimizer progress remains independent.
        """
        contract = type(self).execution_contract
        while self.should_continue_training():
            driver = getattr(self, "acquisition_driver", None)
            if driver is None:
                driver = build_acquisition_driver(contract)
                self.acquisition_driver = driver
            driver.prepare_cycle(
                self,
                self._get_progress(),
                seed=self.training_args.seed,
            )

            if contract.acquisition is AcquisitionMode.GENERATION:
                self._run_periodic_cycle_boundaries()

            self._acquisition_cycle_active = True
            self._acquisition_cycle_incomplete = True
            try:
                driver.run_cycle(self, self._get_progress())
                if contract.acquisition is AcquisitionMode.GENERATION:
                    self.adapter.ema_step(step=self.cycle_index)
                self._after_acquisition_cycle()
                self.progress = self._get_progress().advance_acquisition(
                    contract.acquisition,
                    completed=True,
                )
                self._acquisition_cycle_incomplete = False
            finally:
                self._acquisition_cycle_active = False

            if contract.acquisition is AcquisitionMode.DATASET:
                self._run_periodic_cycle_boundaries()

    def _run_periodic_cycle_boundaries(self) -> None:
        """Run acquisition-specific save/evaluation ordering at a cycle boundary.

        Online training preserves its pre-rollout save-then-evaluate cadence. Offline
        evaluation runs first so an exact checkpoint captures the post-evaluation RNG
        that will precede the next data epoch; model-only saves follow the same visible
        boundary ordering without claiming exact RNG restoration.
        """
        should_save = (
            self.log_args.save_freq > 0
            and self.cycle_index % self.log_args.save_freq == 0
            and self.log_args.save_dir
        )
        save_dir = None
        save_target = None
        if should_save:
            save_dir = os.path.join(
                self.log_args.save_dir,
                str(self.log_args.run_name),
                "checkpoints",
            )
            save_target = os.path.join(save_dir, f"checkpoint-{self.cycle_index}")

        should_evaluate = (
            self.eval_args.eval_freq > 0 and self.cycle_index % self.eval_args.eval_freq == 0
        )

        def save() -> None:
            if save_dir is None:
                return
            if self._should_skip_duplicate_resume_source_checkpoint(save_target):
                return
            self.save_checkpoint(save_dir, epoch=self.cycle_index)

        acquisition = type(self).execution_contract.acquisition
        if acquisition is AcquisitionMode.DATASET:
            if should_evaluate:
                self.evaluate()
            save()
        else:
            save()
            if should_evaluate:
                self.evaluate()
            if getattr(self, "_exact_resume_boundary_pending", False):
                self._exact_resume_boundary_pending = False

    def _should_skip_duplicate_resume_source_checkpoint(
        self,
        save_target: Optional[str],
    ) -> bool:
        """Skip only the first online boundary that resolves to its resume source."""
        if (
            type(self).execution_contract.acquisition is not AcquisitionMode.GENERATION
            or not getattr(self, "_exact_resume_boundary_pending", False)
            or save_target is None
        ):
            return False
        resume_source = getattr(self, "_exact_resume_source_checkpoint", None)
        if resume_source is None:
            return False
        return self._canonical_checkpoint_path(save_target) == resume_source

    def set_trajectory_seed(self, seed: int) -> None:
        """Set the adapter seed for one generated acquisition.

        Args:
            seed: Effective seed for the next generation cycle.
        """
        self.adapter.set_trajectory_seed(seed)

    def run_generation_acquisition(self) -> None:
        """Run one complete generated acquisition and policy update."""
        self._run_training_step()

    def train_on_dataset_batch(self, batch: Any) -> None:
        """Run declared feedback and optimization for one dataset batch.

        Args:
            batch: Collated batch acquired from the finite offline dataloader.
        """
        if type(self).execution_contract.feedback is FeedbackMode.RUNTIME_REWARD:
            self.prepare_feedback(batch)
        self.optimize_batch(batch)

    def _run_training_step(self) -> None:
        """Run one epoch's rollout, feedback and optimization.

        Every trainer supplies ``sample()``; what it stores follows from the
        paradigm, since a coupled algorithm needs the full trajectory and its log
        probabilities while a decoupled one needs only the terminal state.
        Distillation accumulates several dataloader batches before a single
        optimizer step, so the grouping is a hook rather than a fixed sequence.
        """
        cycle_started = time.monotonic()
        rollout_started = cycle_started
        with self.sampling_context():
            samples = self.sample()
        rollout_seconds = time.monotonic() - rollout_started
        feedback_seconds = 0.0
        if type(self).execution_contract.feedback is FeedbackMode.RUNTIME_REWARD:
            if getattr(self.training_args, "reward_optimization_overlap", False):
                self._run_reward_optimization_overlap(
                    samples,
                    cycle_started=cycle_started,
                    rollout_seconds=rollout_seconds,
                )
                return
            feedback_started = time.monotonic()
            self.prepare_feedback(samples)
            feedback_seconds = time.monotonic() - feedback_started
        optimization_started = time.monotonic()
        self.optimize(samples)
        optimization_seconds = time.monotonic() - optimization_started
        metrics = self._critical_path_timing_metrics(
            {
                "timing/rollout_seconds": rollout_seconds,
                "timing/feedback_seconds": feedback_seconds,
                "timing/optimization_seconds": optimization_seconds,
                "timing/cycle_seconds": time.monotonic() - cycle_started,
            }
        )
        self.log_data(metrics, step=self.step)

    def _critical_path_timing_metrics(
        self,
        metrics: Mapping[str, float],
    ) -> Dict[str, float]:
        """Reduce phase durations to rank-wise maxima for critical-path reporting."""
        names = tuple(metrics)
        if not names:
            return {}
        local = torch.tensor(
            [float(metrics[name]) for name in names],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        if self.accelerator.num_processes > 1:
            reduced = self.accelerator.reduce(local, reduction="max")
            if reduced.shape != local.shape:
                raise RuntimeError(
                    "distributed timing reduction returned an invalid shape: "
                    f"expected={tuple(local.shape)}, received={tuple(reduced.shape)}"
                )
            local = reduced
        return {name: float(value) for name, value in zip(names, local.tolist())}

    def _run_reward_optimization_overlap(
        self,
        samples: List[BaseSample],
        *,
        cycle_started: float,
        rollout_seconds: float,
    ) -> None:
        """Consume globally ready reward tiles while the remaining rewards run."""
        try:
            plan = self._build_and_seal_reward_tile_plan(samples)
        except Exception:
            self.reward_buffer.abort_streaming()
            raise

        stream_started = time.monotonic()
        feedback_seconds = 0.0
        optimization_seconds = 0.0
        optimization_started_while_rewards_pending_seconds = 0.0
        reward_wait_seconds = 0.0
        coordination_seconds = 0.0
        first_tile_seconds: Optional[float] = None
        out_of_order_tiles = 0
        poll_count = 0
        readiness_collective_elements = 0

        overlap_context: Any = None
        local_error: Optional[Exception] = None
        preparation_started = time.monotonic()
        try:
            overlap_context = self._prepare_reward_optimization_overlap(samples, plan)
        except Exception as error:
            local_error = error
        preparation_seconds = time.monotonic() - preparation_started
        optimization_seconds += preparation_seconds
        try:
            self._synchronize_reward_overlap_error("optimizer preparation", local_error)
        except Exception:
            self._abort_reward_optimization_overlap(overlap_context)
            self.reward_buffer.abort_streaming()
            raise

        pending = {tile.tile_id: tile for tile in plan.tiles}
        tile_indices = {tile.tile_id: tile.sample_indices for tile in plan.tiles}
        mode = self.training_args.reward_optimization_overlap_mode
        poll_interval = self.training_args.reward_optimization_overlap_poll_interval
        poll_delay = poll_interval
        # Every unsuccessful poll is already a global synchronization point.
        # Back off together while no work becomes runnable, then reset as soon
        # as a tile advances. This bounds readiness latency without issuing a
        # high-frequency all-reduce throughout a slow remote-reward request.
        poll_delay_cap = max(poll_interval, min(1.0, poll_interval * 10.0))

        try:
            while pending:
                coordination_started = time.monotonic()
                poll_count += 1
                local_error: Optional[Exception] = None
                local_ready: Set[int] = set()
                try:
                    local_ready = self.reward_buffer.poll_ready_tiles(
                        {tile_id: tile_indices[tile_id] for tile_id in pending}
                    )
                except Exception as error:
                    local_error = error

                readiness_candidates = self._reward_overlap_readiness_candidates(
                    pending=set(pending),
                    mode=mode,
                )
                if mode == "ordered":
                    ready_values = [
                        int(readiness_candidates[0] in local_ready),
                        int(set(pending).issubset(local_ready)),
                    ]
                else:
                    ready_values = [int(tile_id in local_ready) for tile_id in readiness_candidates]
                ready_flags = torch.tensor(
                    ready_values + [int(local_error is not None)],
                    dtype=torch.int32,
                    device=self.accelerator.device,
                )
                readiness_collective_elements += int(ready_flags.numel())
                ready_counts = self.accelerator.reduce(ready_flags, reduction="sum")
                if int(ready_counts[-1].item()) > 0:
                    self._raise_reward_overlap_errors("reward polling", local_error)
                globally_ready = {
                    tile_id
                    for tile_id, count in zip(
                        readiness_candidates,
                        ready_counts[: len(readiness_candidates)].tolist(),
                    )
                    if count == self.accelerator.num_processes
                }
                all_rewards_ready = (
                    int(ready_counts[1].item()) == self.accelerator.num_processes
                    if mode == "ordered"
                    else len(globally_ready) == len(pending)
                )
                coordination_seconds += time.monotonic() - coordination_started

                next_ordered = min(pending)
                selected = self._select_reward_overlap_tile(
                    pending=set(pending),
                    globally_ready=globally_ready,
                    mode=mode,
                )
                if selected is None:
                    wait_started = time.monotonic()
                    time.sleep(poll_delay)
                    reward_wait_seconds += time.monotonic() - wait_started
                    poll_delay = min(poll_delay * 2.0, poll_delay_cap)
                    continue

                poll_delay = poll_interval
                if first_tile_seconds is None:
                    first_tile_seconds = time.monotonic() - stream_started
                if selected != next_ordered:
                    out_of_order_tiles += 1
                tile = pending.pop(selected)
                tile_samples = plan.samples_for(tile, samples)

                feedback_started = time.monotonic()
                self._resolve_reward_overlap_tile_feedback(
                    tile,
                    tile_samples,
                )
                feedback_seconds += time.monotonic() - feedback_started
                rewards_remain_pending = not all_rewards_ready
                optimization_started = time.monotonic()
                self._optimize_reward_overlap_tile(tile, tile_samples, overlap_context)
                tile_optimization_seconds = time.monotonic() - optimization_started
                optimization_seconds += tile_optimization_seconds
                if rewards_remain_pending:
                    optimization_started_while_rewards_pending_seconds += tile_optimization_seconds

            feedback_started = time.monotonic()
            full_rewards = self._finish_reward_overlap_stream()
            self._compute_synchronized_reward_overlap_advantages(
                samples,
                full_rewards,
                phase="final advantage",
                build_metrics=True,
                collected_layout=self._reward_overlap_acquisition_group_layout,
            )
            feedback_seconds += time.monotonic() - feedback_started
            finalization_started = time.monotonic()
            final_metrics = self._finalize_reward_optimization_overlap(
                overlap_context,
                samples,
            )
            optimization_seconds += time.monotonic() - finalization_started
            stream_seconds = time.monotonic() - stream_started
            timing_metrics = self._critical_path_timing_metrics(
                {
                    "timing/rollout_seconds": rollout_seconds,
                    "timing/feedback_seconds": feedback_seconds,
                    "timing/optimization_seconds": optimization_seconds,
                    "timing/cycle_seconds": time.monotonic() - cycle_started,
                    "timing/reward_overlap/stream_seconds": stream_seconds,
                    "timing/reward_overlap/wait_seconds": reward_wait_seconds,
                    "timing/reward_overlap/coordination_seconds": coordination_seconds,
                    "timing/reward_overlap/preparation_seconds": preparation_seconds,
                    "timing/reward_overlap/first_tile_seconds": first_tile_seconds or 0.0,
                    "timing/reward_overlap/optimization_started_while_rewards_pending_seconds": (
                        optimization_started_while_rewards_pending_seconds
                    ),
                }
            )
            critical_optimization_seconds = timing_metrics["timing/optimization_seconds"]
            pending_at_start_optimization_seconds = timing_metrics[
                "timing/reward_overlap/optimization_started_while_rewards_pending_seconds"
            ]
            timing_metrics.update(
                {
                    "timing/reward_overlap/rollout_seconds": timing_metrics[
                        "timing/rollout_seconds"
                    ],
                    "timing/reward_overlap/optimization_seconds": critical_optimization_seconds,
                    "timing/reward_overlap/optimization_started_after_rewards_ready_seconds": max(
                        0.0,
                        critical_optimization_seconds - pending_at_start_optimization_seconds,
                    ),
                    "timing/reward_overlap/optimization_started_while_rewards_pending_ratio": (
                        pending_at_start_optimization_seconds / critical_optimization_seconds
                        if critical_optimization_seconds > 0
                        else 0.0
                    ),
                }
            )
            metrics = self.advantage_processor.pop_advantage_metrics()
            metrics.update(final_metrics)
            metrics.update(timing_metrics)
            metrics.update(
                {
                    "train/reward_overlap/tile_count": len(plan.tiles),
                    "train/reward_overlap/samples_per_tile": plan.samples_per_tile,
                    "train/reward_overlap/poll_count": poll_count,
                    "train/reward_overlap/readiness_collective_elements": (
                        readiness_collective_elements
                    ),
                    "train/reward_overlap/out_of_order_tiles": out_of_order_tiles,
                    "train/reward_overlap/reused_group_metadata_batches": sum(
                        len(group_infos)
                        for group_infos in self._reward_overlap_group_infos_by_tile.values()
                    ),
                }
            )
            self.log_data(metrics, step=self.step)
        except Exception:
            self._abort_reward_optimization_overlap(overlap_context)
            self.reward_buffer.abort_streaming()
            raise

    def _finish_reward_overlap_stream(self) -> Dict[str, torch.Tensor]:
        """Finish every reward future behind a synchronized rank-local error guard."""
        rewards: Dict[str, torch.Tensor] = {}
        local_error: Optional[Exception] = None
        try:
            rewards = self.reward_buffer.finish_streaming()
        except Exception as error:
            local_error = error
        self._synchronize_reward_overlap_error("reward cycle finish", local_error)
        return rewards

    def _prepare_reward_optimization_overlap(
        self,
        samples: List[BaseSample],
        plan: RewardTilePlan,
    ) -> Any:
        """Prepare objective-owned immutable replay state while rewards run."""
        del samples, plan
        return None

    def _optimize_reward_overlap_tile(
        self,
        tile: RewardTile,
        samples: List[BaseSample],
        context: Any,
    ) -> None:
        """Optimize one globally selected reward-complete tile."""
        del tile, context
        self.optimize(samples)

    def _finalize_reward_optimization_overlap(
        self,
        context: Any,
        samples: List[BaseSample],
    ) -> Mapping[str, Any]:
        """Finish objective-owned state after every reward tile was consumed."""
        del context, samples
        return {}

    def _abort_reward_optimization_overlap(self, context: Any) -> None:
        """Release objective-owned overlap state after a failed cycle."""
        del context

    @staticmethod
    def _reward_overlap_readiness_candidates(
        *,
        pending: Set[int],
        mode: Literal["ordered", "ready"],
    ) -> Tuple[int, ...]:
        """Return tile ids whose individual readiness must cross ranks.

        Ordered execution only needs the head tile. A second scalar in the
        caller records whether *all* pending rewards are ready so overlap timing
        remains exact without communicating one flag per tile.
        """
        if not pending:
            return ()
        if mode == "ordered":
            return (min(pending),)
        if mode == "ready":
            return tuple(sorted(pending))
        raise ValueError(f"unsupported reward overlap mode: {mode!r}")

    @staticmethod
    def _select_reward_overlap_tile(
        *,
        pending: Set[int],
        globally_ready: Set[int],
        mode: Literal["ordered", "ready"],
    ) -> Optional[int]:
        """Select the same reward tile on every rank from global readiness.

        ``ordered`` preserves tile order and therefore lets the oldest pending tile
        gate progress. ``ready`` permits a later tile to bypass that straggler. The
        minimum ready id is still selected to keep the choice deterministic across
        ranks; this does not impose completion order on the reward workers.
        """
        if not pending:
            return None
        globally_ready = pending.intersection(globally_ready)
        if mode == "ordered":
            next_ordered = min(pending)
            return next_ordered if next_ordered in globally_ready else None
        if mode == "ready":
            return min(globally_ready) if globally_ready else None
        raise ValueError(f"unsupported reward overlap mode: {mode!r}")

    def _build_and_seal_reward_tile_plan(
        self,
        samples: List[BaseSample],
    ) -> RewardTilePlan:
        """Build rank-local geometry and fail every rank before streaming starts."""
        plan: Optional[RewardTilePlan] = None
        local_error: Optional[Exception] = None
        try:
            geometry = self.reward_optimization_overlap_geometry(self.config)
            plan = build_reward_tile_plan(
                samples,
                group_size=self.training_args.group_size,
                per_device_batch_size=self.training_args.per_device_batch_size,
                gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
                optimizer_terms_per_batch=geometry.optimizer_terms_per_batch,
                group_layout=geometry.group_layout,
                optimizer_examples_per_group=geometry.optimizer_examples_per_group,
                accumulation_scope=geometry.accumulation_scope,
                group_window_batches=geometry.group_window_batches,
                manifest=self._reward_overlap_acquisition_manifest,
                num_replicas=self.accelerator.num_processes,
            )
            self._validate_reward_overlap_replay_batch_composition(plan, samples)
        except Exception as error:
            local_error = error
        self._synchronize_reward_overlap_error("tile plan construction", local_error)
        if plan is None:  # pragma: no cover - synchronized failure above always raises
            raise RuntimeError("reward tile plan construction returned no plan")

        self._validate_distributed_reward_tile_plan(plan, samples)

        local_error = None
        try:
            if (
                plan.geometry.group_layout != "rank_local"
                and self.reward_buffer.has_async_groupwise_rewards
            ):
                raise ValueError(
                    "cross-rank reward overlap currently supports async pointwise rewards "
                    "only; an async groupwise reward needs a complete group on one rank"
                )
            self.reward_buffer.seal_for_streaming()
        except Exception as error:
            local_error = error
        self._synchronize_reward_overlap_error("reward buffer sealing", local_error)
        return plan

    def _validate_reward_overlap_replay_batch_composition(
        self,
        plan: RewardTilePlan,
        samples: List[BaseSample],
    ) -> None:
        """Keep every pack-dependent rollout micro-batch intact during replay.

        Most adapters are invariant to the other samples in a micro-batch. Some,
        notably Bagel's NaViT-packed forward, are not: changing a pack changes the
        bf16 projection rounding for every member. For those adapters the base
        generation loop records object identities at each rollout boundary. This
        validator proves that tiled replay uses the same samples, order, and
        boundaries before reward streaming or optimization begins.
        """
        if not self.adapter.requires_preserved_replay_batch_composition:
            return

        if (
            plan.geometry.optimizer_examples_per_group is not None
            and plan.geometry.optimizer_examples_per_group != self.training_args.group_size
        ):
            raise ValueError(
                "pack-composition-dependent adapters cannot use reward overlap when the "
                "objective transforms each reward group into a different number of replay "
                "examples"
            )

        manifest = plan.manifest
        if manifest is None or not manifest.rollout_batches:
            raise RuntimeError(
                "reward overlap for a pack-composition-dependent adapter requires rollout "
                "micro-batch metadata; custom generate_samples() implementations must "
                "record the base generation-loop batch boundaries"
            )

        batch_size = self.training_args.per_device_batch_size
        invalid_sizes = tuple(
            len(batch) for batch in manifest.rollout_batches if len(batch) != batch_size
        )
        if invalid_sizes:
            raise ValueError(
                "pack-composition-dependent replay requires every rollout micro-batch to "
                f"match train.per_device_batch_size={batch_size}; invalid_sizes={invalid_sizes!r}"
            )

        manifest.validate_samples(samples)
        boundaries = manifest.rollout_boundaries
        for tile in plan.tiles:
            if tile.start not in boundaries or tile.stop not in boundaries:
                raise ValueError(
                    "reward tile splits a pack-composition-dependent rollout micro-batch: "
                    f"tile_id={tile.tile_id}, range=({tile.start}, {tile.stop}), "
                    f"rollout_boundaries={tuple(sorted(boundaries))!r}"
                )
            replay_batch_count = tile.sample_count // batch_size
            if replay_batch_count != plan.batches_per_tile:
                raise ValueError(
                    "reward tile would change pack-composition-dependent replay batches: "
                    f"tile_id={tile.tile_id}, replay_batches={replay_batch_count}, "
                    f"planned_optimizer_batches={plan.batches_per_tile}"
                )

    def _validate_distributed_reward_tile_plan(
        self,
        plan: RewardTilePlan,
        samples: List[BaseSample],
    ) -> None:
        """Require rank-uniform geometry and complete global cross-rank groups."""
        cross_rank_groups = plan.geometry.group_layout in {
            "cross_rank_sharded",
            "cross_rank_tiled",
        }
        group_layout_code = {
            "rank_local": 0,
            "cross_rank_sharded": 1,
            "cross_rank_tiled": 2,
        }[plan.geometry.group_layout]
        local_group_identities: Optional[torch.Tensor] = None
        local_identity_error: Optional[Exception] = None
        if cross_rank_groups:
            try:
                local_group_identities = torch.as_tensor(
                    group_identity_rows(samples),
                    dtype=torch.int64,
                    device=self.accelerator.device,
                )
            except Exception as error:
                local_identity_error = error
        local_header = torch.tensor(
            [
                len(plan.tiles),
                plan.samples_per_tile,
                plan.batches_per_tile,
                plan.sample_count,
                len(samples),
                group_layout_code,
                int(local_identity_error is not None),
            ],
            dtype=torch.int64,
            device=self.accelerator.device,
        )
        self._reward_overlap_group_infos_by_tile: Dict[int, Tuple[_RewardOverlapGroupInfo, ...]] = (
            {}
        )
        self._reward_overlap_group_layouts_by_tile: Dict[int, CollectedGroupLayout] = {}
        self._reward_overlap_acquisition_group_layout: Optional[CollectedGroupLayout] = None

        if self.accelerator.num_processes > 1:
            gathered_headers = self.accelerator.gather(local_header)
        else:
            gathered_headers = local_header
        expected_header_values = self.accelerator.num_processes * int(local_header.numel())
        if int(gathered_headers.numel()) != expected_header_values:
            raise RuntimeError(
                "distributed reward tile header validation returned an invalid cardinality: "
                f"expected={expected_header_values}, received={gathered_headers.numel()}"
            )
        gathered_headers = gathered_headers.reshape(self.accelerator.num_processes, -1).cpu()
        if not torch.equal(
            gathered_headers[:, :6],
            gathered_headers[0, :6].expand_as(gathered_headers[:, :6]),
        ):
            raise RuntimeError(
                "reward tile geometry differs across ranks: "
                "(tile_count, samples_per_tile, batches_per_tile, plan_sample_count, "
                "local_sample_count, group_layout_code)="
                f"{gathered_headers.tolist()!r}"
            )
        if not torch.equal(gathered_headers[:, 3], gathered_headers[:, 4]):
            raise RuntimeError(
                "reward tile plan sample count differs from the local sample count: "
                f"headers={gathered_headers.tolist()!r}"
            )
        if torch.any(gathered_headers[:, 6]):
            self._raise_reward_overlap_errors("tile UID preparation", local_identity_error)
        if not cross_rank_groups:
            return

        if local_group_identities is None:  # pragma: no cover - header failure above
            raise RuntimeError("reward tile UID preparation returned no payload")
        group_coordinator = getattr(self, "group_coordinator", None)
        group_world_size = (
            group_coordinator.group_world_size
            if group_coordinator is not None
            else self.accelerator.num_processes
        )
        group_rank = (
            group_coordinator.group_rank
            if group_coordinator is not None
            else self.accelerator.process_index
        )
        if group_world_size > 1:
            gathered_identities = (
                group_coordinator.gather(local_group_identities)
                if group_coordinator is not None
                else self.accelerator.gather(local_group_identities)
            )
        else:
            gathered_identities = local_group_identities
        expected_identity_values = group_world_size * plan.sample_count * 2
        if int(gathered_identities.numel()) != expected_identity_values:
            raise RuntimeError(
                "distributed reward identity validation returned an invalid cardinality: "
                f"expected={expected_identity_values}, received={gathered_identities.numel()}"
            )
        gathered_identities = gathered_identities.reshape(
            group_world_size,
            plan.sample_count,
            2,
        )

        def collected_layout(start: int, stop: int) -> CollectedGroupLayout:
            identities = gathered_identities[:, start:stop].reshape(-1, 2)
            _unique, inverse = torch.unique(
                identities,
                dim=0,
                sorted=True,
                return_inverse=True,
            )
            return CollectedGroupLayout(
                group_indices=inverse.cpu().numpy(),
                source_ids=identities[:, 0].cpu().numpy(),
                local_sample_count=stop - start,
                num_processes=group_world_size,
            )

        self._reward_overlap_acquisition_group_layout = collected_layout(
            0,
            plan.sample_count,
        )
        self._reward_overlap_group_layouts_by_tile = {
            tile.tile_id: collected_layout(tile.start, tile.stop) for tile in plan.tiles
        }
        seen_groups: Set[Tuple[int, int]] = set()
        batch_size = self.training_args.per_device_batch_size
        for tile in plan.tiles:
            tile_group_infos: List[_RewardOverlapGroupInfo] = []
            if plan.geometry.group_layout == "cross_rank_tiled":
                tile_identities = gathered_identities[:, tile.start : tile.stop].reshape(-1, 2)
                unique_identities, counts = torch.unique(
                    tile_identities,
                    dim=0,
                    sorted=True,
                    return_counts=True,
                )
                expected_counts = torch.full_like(counts, self.training_args.group_size)
                if not torch.equal(counts, expected_counts):
                    raise ValueError(
                        "cross-rank tiled reward work unit does not contain complete "
                        f"global groups: tile_id={tile.tile_id}, "
                        f"identities={unique_identities.tolist()}, counts={counts.tolist()}, "
                        f"group_size={self.training_args.group_size}"
                    )
                tile_groups = {tuple(identity) for identity in unique_identities.tolist()}
                repeated = seen_groups.intersection(tile_groups)
                if repeated:
                    raise ValueError(
                        "cross-rank tiled reward groups must belong to exactly one work "
                        f"unit; tile_id={tile.tile_id}, "
                        f"repeated_group_identities={sorted(repeated)!r}"
                    )
                seen_groups.update(tile_groups)
                self._reward_overlap_group_infos_by_tile[tile.tile_id] = ()
                continue
            for batch_start in range(tile.start, tile.stop, batch_size):
                batch_stop = min(batch_start + batch_size, tile.stop)
                batch_identities = gathered_identities[:, batch_start:batch_stop].reshape(-1, 2)
                unique_identities, inverse, counts = torch.unique(
                    batch_identities,
                    dim=0,
                    sorted=True,
                    return_inverse=True,
                    return_counts=True,
                )
                expected_counts = torch.full_like(counts, self.training_args.group_size)
                if not torch.equal(counts, expected_counts):
                    raise ValueError(
                        "cross-rank reward optimizer batch does not contain complete "
                        f"global groups: tile_id={tile.tile_id}, "
                        f"batch_start={batch_start}, identities={unique_identities.tolist()}, "
                        f"counts={counts.tolist()}, group_size={self.training_args.group_size}"
                    )
                batch_groups = {tuple(identity) for identity in unique_identities.tolist()}
                repeated = seen_groups.intersection(batch_groups)
                if repeated:
                    raise ValueError(
                        "cross-rank reward groups must belong to exactly one optimizer "
                        f"batch; tile_id={tile.tile_id}, batch_start={batch_start}, "
                        f"repeated_group_identities={sorted(repeated)!r}"
                    )
                seen_groups.update(batch_groups)
                local_batch_identities = gathered_identities[
                    group_rank,
                    batch_start:batch_stop,
                ]
                local_batch_size = batch_stop - batch_start
                local_group_indices = inverse.reshape(
                    group_world_size,
                    local_batch_size,
                )[group_rank]
                tile_group_infos.append(
                    _RewardOverlapGroupInfo(
                        local_group_identities=local_batch_identities.clone(),
                        local_group_indices=local_group_indices.clone(),
                        num_groups=int(unique_identities.shape[0]),
                    )
                )
            self._reward_overlap_group_infos_by_tile[tile.tile_id] = tuple(tile_group_infos)

    def _reward_overlap_group_infos_for_tile(
        self,
        tile_id: int,
    ) -> Tuple[_RewardOverlapGroupInfo, ...]:
        """Return cached cross-rank group mappings for one overlap tile."""
        group_infos = self._reward_overlap_group_infos_by_tile.get(tile_id)
        if group_infos is None:
            raise RuntimeError(
                "reward overlap cross-rank group metadata is unavailable for " f"tile_id={tile_id}"
            )
        return group_infos

    def _resolve_reward_overlap_tile_feedback(
        self,
        tile: RewardTile,
        samples: List[BaseSample],
    ) -> None:
        """Resolve and validate one tile before entering advantage collectives.

        Reward resolution and payload preparation are both rank-local, so one
        error reduction guards them together. Keeping separate guards would add
        one redundant global collective to every optimizer work unit.
        """
        rewards: Dict[str, torch.Tensor] = {}
        prepared_collection = None
        local_error: Optional[Exception] = None
        try:
            rewards = self.reward_buffer.resolve_streaming_tile(tile.sample_indices)
            prepared_collection = self.advantage_processor.prepare_group_reward_collection(
                samples,
                rewards,
                require_all_rewards=True,
                collected_layout=self._reward_overlap_group_layouts_by_tile.get(tile.tile_id),
            )
        except Exception as error:
            local_error = error
        self._synchronize_reward_overlap_error("tile feedback preparation", local_error)

        local_error = None
        try:
            self._compute_reward_overlap_advantages(
                samples,
                rewards,
                build_metrics=False,
                prepared_collection=prepared_collection,
            )
        except Exception as error:
            local_error = error
        self._synchronize_reward_overlap_error("tile advantage computation", local_error)

    def _compute_synchronized_reward_overlap_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        *,
        phase: str,
        build_metrics: bool,
        collected_layout: Optional[CollectedGroupLayout] = None,
    ) -> None:
        """Guard local packing before cross-rank advantage collectives."""
        prepared_collection = None
        local_error: Optional[Exception] = None
        try:
            prepared_collection = self.advantage_processor.prepare_group_reward_collection(
                samples,
                rewards,
                require_all_rewards=True,
                collected_layout=collected_layout,
            )
        except Exception as error:
            local_error = error
        self._synchronize_reward_overlap_error(f"{phase} preparation", local_error)

        local_error = None
        try:
            self._compute_reward_overlap_advantages(
                samples,
                rewards,
                build_metrics=build_metrics,
                prepared_collection=prepared_collection,
            )
        except Exception as error:
            local_error = error
        self._synchronize_reward_overlap_error(f"{phase} computation", local_error)

    def _synchronize_reward_overlap_error(
        self,
        phase: str,
        error: Optional[Exception],
    ) -> None:
        """Propagate a rank-local reward error before peers enter optimizer collectives."""
        if self.accelerator.num_processes <= 1:
            if error is not None:
                raise error
            return
        local_failure = torch.tensor(
            [int(error is not None)],
            dtype=torch.int32,
            device=self.accelerator.device,
        )
        failure_count = self.accelerator.reduce(local_failure, reduction="sum")
        if int(failure_count.item()) == 0:
            return
        self._raise_reward_overlap_errors(phase, error)

    def _raise_reward_overlap_errors(
        self,
        phase: str,
        error: Optional[Exception],
    ) -> None:
        """Gather details after a distributed failure flag is already known."""
        payload = (
            None
            if error is None
            else {
                "rank": self.accelerator.process_index,
                "type": type(error).__name__,
                "message": str(error),
            }
        )
        if self.accelerator.num_processes <= 1:
            if error is not None:
                raise error
            raise RuntimeError(f"reward optimization overlap {phase} failed")
        failures = tuple(item for item in gather_object([payload]) if item is not None)
        message = f"reward optimization overlap {phase} failed across ranks: {failures!r}"
        if error is not None:
            raise RuntimeError(message) from error
        raise RuntimeError(message)

    @contextmanager
    def sampling_context(self) -> Iterator[None]:
        """Parameter scope for rollout generation.

        On-policy sampling needs no swap; algorithms that roll out under EMA, a
        reference snapshot, or a separate sampling model override this.
        """
        yield

    def _after_acquisition_cycle(self) -> None:
        """Update algorithm-owned state after one complete acquisition cycle.

        Generated acquisition calls this once per rollout iteration; dataset
        acquisition calls it once per complete dataloader epoch. Per-update state
        belongs in :meth:`_after_gradient_step` instead.
        """

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Stages 4--5: finalize rewards, compute advantages, and log metrics.

        No policy gradients here. Distillation has no reward signal and overrides
        this with a no-op; algorithms that need extra batching before the loss
        (DPO's chosen/rejected pairing) do that work in :meth:`optimize`, after
        advantages are on each sample.
        """
        rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
        self.compute_advantages(samples, rewards, store_to_samples=True)
        adv_metrics = self.advantage_processor.pop_advantage_metrics()
        if adv_metrics:
            self.log_data(adv_metrics, step=self.step)

    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func: Optional[Union[Literal["sum", "gdpo"], Callable]] = None,
    ) -> torch.Tensor:
        """Turn per-sample rewards into advantages via the advantage processor.

        Args:
            samples: Samples this epoch's rewards belong to.
            rewards: Reward tensors by reward name, aligned with ``samples``.
            store_to_samples: Whether to write advantages back onto each sample.
            aggregation_func: Within-group aggregation, defaulting to the
                configured ``advantage_aggregation``.
        Returns:
            One advantage per sample.
        """
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )

    def _compute_reward_overlap_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        *,
        build_metrics: bool,
        prepared_collection: Any = None,
    ) -> torch.Tensor:
        """Compute tile or acquisition advantages without changing the public API."""
        return self.advantage_processor._compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=True,
            aggregation_func=self.training_args.advantage_aggregation,
            build_metrics=build_metrics,
            prepared_collection=prepared_collection,
        )

    def optimize(self, *args: Any, **kwargs: Any) -> None:
        """Update a policy from generated examples.

        Args:
            *args: Algorithm-specific generated-acquisition inputs.
            **kwargs: Algorithm-specific optimization options.
        """
        raise NotImplementedError(
            f"generation trainer {type(self).__name__} must implement optimize(samples)"
        )

    def optimize_batch(self, batch: Any) -> None:
        """Update a policy from one acquired dataset batch.

        Args:
            batch: Collated offline training batch.
        """
        raise NotImplementedError(
            f"dataset trainer {type(self).__name__} must implement optimize_batch(batch)"
        )

    def _sample_timesteps(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample scheduler-scale training timesteps in ``[0, 1000]``.

        Decoupled algorithms draw a training coordinate rather than replaying a
        stored one, and the strategy is a configuration choice rather than an
        algorithmic one, so it lives here. An algorithm whose draw is part of its
        objective (DPO shares one draw across preference arms) overrides this.

        Args:
            batch_size: Size of the broadcast batch dimension.
            generator: Optional ``torch.Generator``. When supplied, the draw is
                deterministic and cross-rank-reproducible for any strategy, which
                is how a group-based algorithm shares one coordinate across ranks.

        Returns:
            Tensor of shape ``(num_train_timesteps, batch_size)``.

        Raises:
            ValueError: If the configured strategy is not recognized.
        """
        device = self.accelerator.device
        strategy = self.time_sampling_strategy.lower()
        available = [
            "logit_normal",
            "uniform",
            "discrete",
            "discrete_with_init",
            "discrete_wo_init",
        ]

        if strategy == "logit_normal":
            return TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
                stratified=True,
                generator=generator,
            )
        if strategy == "uniform":
            return TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
                generator=generator,
            )
        if strategy.startswith("discrete"):
            discrete_config = {
                "discrete": (True, False),
                "discrete_with_init": (True, True),
                "discrete_wo_init": (False, False),
            }
            if strategy not in discrete_config:
                raise ValueError(
                    f"Unknown time_sampling_strategy: {strategy!r}. Available: {available}"
                )
            include_init, force_init = discrete_config[strategy]
            return TimeSampler.discrete(
                batch_size=batch_size,
                num_train_timesteps=self.num_train_timesteps,
                scheduler_timesteps=self.adapter.scheduler.timesteps,
                timestep_range=self.timestep_range,
                include_init=include_init,
                force_init=force_init,
                generator=generator,
            )

        raise ValueError(f"Unknown time_sampling_strategy: {strategy!r}. Available: {available}")

    def _apply_optimizer_step(
        self,
        loss_info: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, List[torch.Tensor]]:
        """Clip, step, log the accumulated losses and start a fresh accumulation.

        Call this once ``accelerator.sync_gradients`` is true. Only the loss that
        reached ``backward`` is algorithm-specific; clipping, stepping and metric
        reduction are the same for every algorithm.

        Args:
            loss_info: Per-metric values accumulated since the last optimizer step.

        Returns:
            An empty accumulator for the next optimizer step.
        """
        # Hand over the prepared root's full parameter list, not just the trainable
        # subset. Accelerate only delegates to FSDP's collective `clip_grad_norm_`
        # when the list it receives is exactly `model.parameters()`; given a subset it
        # falls through to the plain utility, which under FSDP1 computes the norm from
        # this rank's shard alone and clips inconsistently across ranks. Frozen
        # parameters carry no gradient, so including them changes nothing for the
        # backends that do not shard.
        grad_norm = self.accelerator.clip_grad_norm_(
            self.model_bundle.parameters(),
            self.training_args.max_grad_norm,
        )
        self.optimizer.step()
        if type(self).execution_contract.acquisition is AcquisitionMode.DATASET:
            self.adapter.ema_step(step=self.step)
        self.optimizer.zero_grad()
        self._after_gradient_step()

        reduced = reduce_loss_info(self.accelerator, loss_info)
        reduced["grad_norm"] = grad_norm
        self.log_data({f"train/{k}": v for k, v in reduced.items()}, step=self.step)
        self.step += 1
        return defaultdict(list)

    def _after_gradient_step(self) -> None:
        """Update per-optimizer-step auxiliary weights before metrics are logged.

        Distinct from :meth:`_after_acquisition_cycle`, which runs after a complete
        rollout iteration or dataloader epoch; this runs on every optimizer step,
        which is the cadence DGPO's fast reference EMA needs.
        """

    def _velocity_kl(
        self,
        velocity: LatentState,
        other_velocity: LatentState,
        noised: NoisedState,
    ) -> torch.Tensor:
        """Compute the per-sample squared velocity gap against another policy.

        Under a fixed forward process the KL between two Gaussian transition
        kernels reduces to the squared gap between their velocity predictions, so
        every decoupled algorithm that regularizes towards a reference, an EMA, or
        an older snapshot needs exactly this quantity. Which policy supplies
        ``other_velocity`` is the algorithm's choice; the reduction is not.

        Args:
            velocity: Current-policy velocity per component.
            other_velocity: Reference, EMA or old-snapshot velocity per component.
            noised: Forward-noised state supplying per-sample reduction context.

        Returns:
            Per-sample KL surrogate of shape ``(B,)``.
        """
        errors = {
            name: (velocity.components[name] - other_velocity.components[name]) ** 2
            for name in self.adapter.trajectory_component_order
        }
        return self.adapter.reduce_latent_values(errors, state=noised.state)

    def _order_samples_for_optimize(
        self, samples: List[BaseSample], inner_epoch: int
    ) -> List[BaseSample]:
        """Return the per-inner-epoch sample ordering for the optimize loop.

        When ``training_args.shuffle_samples`` is False, the rollout-pack order is
        preserved so each training micro-batch packs exactly the samples of its
        corresponding rollout ``inference`` pack. For adapters whose batched forward
        is pack-composition-dependent (e.g. Bagel/NaViT packing), this keeps the
        bf16 forward bit-identical between rollout and training (on-policy ratio==1).
        """
        if not self.training_args.shuffle_samples:
            return samples
        perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
        perm = torch.randperm(len(samples), generator=perm_gen)
        return [samples[i] for i in perm]

    def _maybe_offload_samples_to_cpu(self, samples: List[BaseSample]) -> None:
        """Offload each sample's tensors to pinned CPU when offload is enabled.

        Producer half of the CPU-offload pipeline; keeps the rollout buffer's GPU
        peak bounded. Must run BEFORE ``reward_buffer.add_samples`` so the recorded
        ``sync_event`` captures "D2H complete + data on CPU" for async reward
        workers. Uses pinned CPU + blocking D2H so the later per-micro-batch H2D
        reload (``_iter_prefetched_batches``) can be issued asynchronously. No-op
        when ``training_args.offload_samples_to_cpu`` is False (default).
        """
        if not self.training_args.offload_samples_to_cpu:
            return
        for sample in samples:
            sample.to("cpu", pin_memory=True)

    def _iter_prefetched_batches(
        self,
        samples: List[BaseSample],
        per_device_batch_size: int,
    ) -> Iterator[StackedSampleBatch]:
        """Yield device-resident stacked micro-batches for the optimize loop.

        Each yielded :class:`StackedSampleBatch` also exposes the moved per-sample
        objects it was stacked from via ``batch.samples`` -- callers that need
        per-sample access (e.g. OPD teacher routing / ``mu_teacher`` write-back)
        read that, with no second move or a redundant side index.

        When samples are CPU-offloaded (pinned), the next micro-batch's H2D copy
        runs on a dedicated copy stream to overlap the current batch's compute;
        ``wait_stream`` ensures the batch is fully copied before use and
        ``record_stream`` keeps it alive until the default stream is done.
        Otherwise (offload off, no CUDA, or a single batch) it is a plain blocking
        stack. Numerically equivalent either way; only data-movement timing changes.

        Yields:
            StackedSampleBatch: a stacked micro-batch (its source samples are at
            ``batch.samples``).
        """
        yield from iter_prefetched_batches(
            samples,
            per_device_batch_size,
            device=self.accelerator.device,
            offload_samples_to_cpu=self.training_args.offload_samples_to_cpu,
        )

    def sample_batch(
        self,
        batch: Dict[str, Any],
        reward_buffer: Optional[RewardBuffer] = None,
        **extra_inference_kwargs,
    ) -> List[BaseSample]:
        """Unified single-batch sampling pipeline.

        Encapsulates the standard post-inference steps that every trainer
        repeats in its sampling loop:

            1. Merge training/eval args + batch + extra kwargs
            2. ``filter_kwargs`` → ``adapter.inference()``
            3. Inject dataset metadata into samples
            4. Optionally offload samples to CPU
            5. Optionally feed samples into a ``RewardBuffer``

        Subclasses may override this method to customize the per-batch
        pipeline (e.g. adding custom post-processing or using a different
        inference call). The default implementation is sufficient for most
        algorithms.

        Args:
            batch: DataLoader batch dict (contains prompt, metadata, etc.)
            reward_buffer: If provided, ``add_samples()`` is called automatically.
            **extra_inference_kwargs: Passed to ``adapter.inference()`` after
                filtering. Common keys: ``compute_log_prob``,
                ``trajectory_indices``, ``generator``.

        Returns:
            List of generated ``BaseSample`` instances with metadata injected.
        """
        model_batch = {
            key: value for key, value in batch.items() if key not in SAMPLING_IDENTITY_COLUMNS
        }
        sample_kwargs = {**self.training_args, **extra_inference_kwargs, **model_batch}
        sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
        sample_batch = self.adapter.inference(**sample_kwargs)

        # Defensively reset applicable_rewards on every newly produced sample.
        # The factory default is an empty set, but if any future trainer
        # reuses sample objects across epochs (e.g. a sample buffer), stale
        # bookkeeping from prior epochs would corrupt aggregation.  Cheap
        # to do unconditionally; makes the contract explicit.
        for s in sample_batch:
            s.applicable_rewards = set()

        # Inject dataset metadata (e.g. geneval_metadata) into samples' extra_kwargs
        self._inject_batch_metadata(sample_batch, batch)

        # Offload to CPU before reward buffer sees them
        self._maybe_offload_samples_to_cpu(sample_batch)

        # Feed into reward buffer for async/sync reward computation
        if reward_buffer is not None:
            reward_buffer.add_samples(sample_batch)

        return sample_batch

    @staticmethod
    def _augment_batch_with_source(
        batch: Dict[str, Any],
        source_name: str,
        source_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Stamp source routing keys onto a batch dict for downstream propagation.

        Plain DataLoaders (eval, future standalone sampling) lack the
        automatic ``__source__`` / ``__source_id__`` injection that
        ``MultiSourceTrainDataLoader`` provides.  Call this before
        ``sample_batch`` so ``_inject_batch_metadata`` can propagate
        source onto every generated sample via its existing K-repeat
        broadcast logic.
        """
        batch = dict(batch)
        B = len(batch["prompt"])
        batch["__source__"] = [source_name] * B
        if source_id is not None:
            batch["__source_id__"] = [source_id] * B
        return batch

    @staticmethod
    def _inject_batch_metadata(
        samples: List[BaseSample],
        batch: Dict[str, Any],
    ) -> None:
        """Inject dataset metadata into generated samples' extra_kwargs.

        Bridges the gap between dataset JSONL fields and reward model kwargs:
        non-preprocess fields from the dataloader batch are copied into each
        sample's ``extra_kwargs``, making them accessible to reward models via
        ``filter_kwargs(model.__call__, **sample)``.

        Convention: complex metadata values are stored as JSON strings in the
        JSONL for Arrow serialization safety. Reward models parse them with
        ``json.loads()`` as needed.

        Also propagates the per-batch ``__source__`` / ``__source_id__``
        (multi-source training only — populated by
        ``MultiSourceTrainDataLoader`` in ``data_utils/loader.py``) onto
        the typed ``BaseSample.source`` / ``BaseSample.source_id`` fields.
        Drives both the ``RewardProcessor`` gate and the
        ``AdvantageProcessor`` applicability mask.

        Planned training batches additionally carry three reserved sampling
        identity columns. These are attached one-to-one to generated samples
        and never forwarded to the adapter API.

        Args:
            samples: Generated samples from ``adapter.inference()``.
            batch: The dataloader batch dict (may contain ``metadata`` /
                ``__source__`` / ``__source_id__`` keys).
        """
        # Per-prompt ratio used for both metadata and __source__ broadcasting.
        # Some adapters generate K replicates per prompt (group_size > 1) so
        # one batch row maps to several samples.
        sources = batch.get("__source__")
        source_ids = batch.get("__source_id__")
        metadata_list = batch.get(METADATA_COLUMN)
        group_ids = batch.get(SAMPLING_GROUP_ID_COLUMN)
        group_member_ids = batch.get(SAMPLING_GROUP_MEMBER_ID_COLUMN)
        sample_ids = batch.get(SAMPLING_SAMPLE_ID_COLUMN)
        identity_values = (group_ids, group_member_ids, sample_ids)
        has_sampling_identity = any(value is not None for value in identity_values)
        if has_sampling_identity and not all(value is not None for value in identity_values):
            raise ValueError(
                "planned batches must carry group, member, and sample identity together"
            )
        if (
            metadata_list is None
            and sources is None
            and source_ids is None
            and not has_sampling_identity
        ):
            return
        if not samples:
            return

        columns = [
            value
            for value in (
                metadata_list,
                sources,
                source_ids,
                group_ids,
                group_member_ids,
                sample_ids,
            )
            if value is not None
        ]
        B = len(columns[0])
        if any(len(value) != B for value in columns):
            raise ValueError("batch metadata and sampling identity columns must have equal length")
        if has_sampling_identity and len(samples) != B:
            raise ValueError(
                "planned sampling identity requires one adapter output per dataloader row: "
                f"rows={B}, outputs={len(samples)}"
            )
        samples_per_prompt = 1 if has_sampling_identity else len(samples) // B
        if samples_per_prompt < 1:
            return

        for i, sample in enumerate(samples):
            batch_idx = i if has_sampling_identity else i // samples_per_prompt
            if batch_idx >= B:
                continue
            if metadata_list:
                meta = metadata_list[batch_idx]
                if isinstance(meta, dict):
                    sample.extra_kwargs[METADATA_COLUMN] = json.dumps(meta, default=json_default)
            if sources:
                # Homogeneous within a batch in this PR; per-sample shape
                # leaves room for future PRs that may interleave within a
                # batch without a code change.
                sample.source = sources[batch_idx]
            if source_ids:
                sample.source_id = source_ids[batch_idx]
            if has_sampling_identity:
                sample.assign_sampling_identity(
                    group_id=int(group_ids[batch_idx]),
                    group_member_id=int(group_member_ids[batch_idx]),
                    sample_id=int(sample_ids[batch_idx]),
                )

    # ============================ Public Sampling API ============================

    def generate_samples(
        self,
        reward_buffer: Optional[RewardBuffer] = None,
        compute_log_prob: bool = False,
        trajectory_indices: Optional[List[int]] = None,
        **extra_inference_kwargs,
    ) -> List[BaseSample]:
        """Complete one epoch of sample generation.

        Standard pipeline::

            adapter.rollout() → clear buffer → loop(dataloader) {
                sample_batch() → extend samples
            }

        Subclasses call this from their ``sample()`` method with
        algorithm-specific parameters. For fully custom sampling logic
        (e.g. paired generation), override this method directly.

        Args:
            reward_buffer: Buffer for reward computation. Cleared at start
                and fed after each batch automatically.
            compute_log_prob: Whether to store log-probabilities during inference.
            trajectory_indices: Which timestep positions to store in each sample.
                ``[-1]`` = final latent only (default for most algorithms).
                Full list = store all (GRPO needs this for PPO ratio).
                ``None`` = no trajectory recording (used during evaluation).
            **extra_inference_kwargs: Forwarded to ``adapter.inference()``
                after ``filter_kwargs``. Common keys: ``generator``.

        Returns:
            All generated samples for this epoch.

        Note:
            Trainers that override ``generate_samples`` instead of just
            ``sample()`` must still call :meth:`sample_batch` per batch
            so :meth:`_inject_batch_metadata` propagates ``__source__``
            onto every sample.  An end-of-loop runtime check verifies
            this in multi-source mode.
        """
        if self.dataloader is None:
            raise RuntimeError(
                "generate_samples() called but no training dataloader exists. "
                "`data.datasets` has no entry with `train: enabled` (eval-only "
                "config); a trainer should not enter the sampling loop here."
            )

        self.adapter.rollout()
        capture_overlap_manifest = bool(
            reward_buffer is not None
            and getattr(self.training_args, "reward_optimization_overlap", False)
        )
        recorded_rollout_batches: List[Tuple[int, ...]] = []
        self._reward_overlap_acquisition_manifest = None
        if reward_buffer is not None:
            reward_buffer.clear()

        # Multi-source: reseed the per-source schedule + every per-source
        # sampler so replays of the same epoch are reproducible. No-op
        # for the bare DataLoader (no `set_epoch`).
        if hasattr(self.dataloader, "set_epoch"):
            self.dataloader.set_epoch(self.epoch)
        else:
            sampler = getattr(self.dataloader, "batch_sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)

        samples: List[BaseSample] = []
        data_iter = iter(self.dataloader)

        # Stage-3-only acceleration (e.g. feature caching) is scoped to this loop
        # so its state never leaks into the Stage-6 training forward.
        # The outer path stays no_grad; a compile accelerator re-enables gradients
        # only inside the transformer call to match the training compiled graph.
        with self._rollout_acceleration(), torch.no_grad(), self.autocast():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f"Epoch {self.epoch} Sampling",
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_batch = self.sample_batch(
                    batch,
                    reward_buffer=reward_buffer,
                    compute_log_prob=compute_log_prob,
                    trajectory_indices=trajectory_indices,
                    **extra_inference_kwargs,
                )
                if capture_overlap_manifest:
                    recorded_rollout_batches.append(tuple(id(sample) for sample in sample_batch))
                samples.extend(sample_batch)

        if capture_overlap_manifest:
            self._reward_overlap_acquisition_manifest = AcquisitionManifest.from_samples(
                samples,
                rollout_batch_object_ids=recorded_rollout_batches,
            )

        # Multi-source invariant: when more than one training source is
        # active, batches flow through `MultiSourceTrainDataLoader`, which
        # injects `__source__` so every sample carries `source`. Single-source
        # configs use a bare DataLoader (no injection) and the reward gate
        # treats `source is None` as "applies to all" — so the check must NOT
        # fire there. This catches a trainer that overrode generate_samples
        # but bypassed sample_batch / _inject_batch_metadata.
        if len(self.train_dataloaders_by_source) > 1 and samples:
            missing = [i for i, s in enumerate(samples) if s.source is None]
            if missing:
                raise RuntimeError(
                    f"Multi-source training: {len(missing)} sample(s) at indices "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''} are missing "
                    "`source`. Did a trainer override "
                    "`generate_samples` without going through `sample_batch` "
                    "(which calls `_inject_batch_metadata`)?"
                )

        return samples

    def evaluate(self) -> None:
        """Evaluation loop: a single, unified per-dataset path.

        For every eval-eligible entry in ``data.datasets`` (which now
        includes the canonicalized legacy ``data.dataset_dir`` when a
        ``test.jsonl`` exists):

        1. Generate samples using the dataset's DataLoader with per-dataset
           eval overrides (resolution, guidance_scale, num_inference_steps).
        2. Optionally compute and gather configured eval rewards.
        3. Always log generated media; add reward metrics when available.

        Logs are flushed per-dataset to avoid holding all generated samples
        in memory simultaneously.  Uses EMA parameters (if available) and
        eval-specific config (resolution, inference steps, guidance scale).

        No-op when ``self.eval_dataloaders`` is empty.
        """
        if not self.eval_dataloaders:
            return

        self.adapter.eval()

        with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
            for dataset_name, dataloader in self.eval_dataloaders.items():
                buffer = self.eval_dataset_reward_buffers.get(dataset_name)
                if buffer is not None:
                    buffer.clear()
                all_samples: List[BaseSample] = []

                # Merge per-dataset eval overrides with shared eval_args
                ed_config = self._eval_dataset_configs[dataset_name]
                eval_kwargs = (
                    ed_config.eval.get_merged_eval_kwargs(self.eval_args)
                    if ed_config.eval
                    else dict(self.eval_args)
                )

                for batch in tqdm(
                    dataloader,
                    desc=f"Eval/{dataset_name}",
                    disable=not self.show_progress_bar,
                ):
                    batch = self._augment_batch_with_source(
                        batch, dataset_name, ed_config.source_id
                    )
                    generator = create_generator_by_prompt(batch["prompt"], self.training_args.seed)
                    samples = self.sample_batch(
                        batch,
                        reward_buffer=buffer,
                        compute_log_prob=False,
                        generator=generator,
                        trajectory_indices=None,
                        **eval_kwargs,
                    )
                    all_samples.extend(samples)

                gathered_rewards: Dict[str, np.ndarray] = {}
                if buffer is not None:
                    rewards = buffer.finalize(store_to_samples=True, split="pointwise")
                    # Pack all reward columns so evaluation pays for one gather per
                    # dataset rather than one gather per reward model.
                    rewards_tensors = {
                        key: torch.as_tensor(value).to(self.accelerator.device)
                        for key, value in rewards.items()
                    }
                    gathered_rewards = {
                        key: value.cpu().numpy()
                        for key, value in gather_aligned_floating_tensors(
                            self.accelerator,
                            rewards_tensors,
                        ).items()
                    }

                # Log per-dataset immediately to avoid accumulating all samples in memory
                if self.accelerator.is_main_process:
                    log_data: Dict[str, Any] = {}
                    for k, v in gathered_rewards.items():
                        log_data[f"eval/{dataset_name}/reward_{k}_mean"] = np.mean(v)
                        log_data[f"eval/{dataset_name}/reward_{k}_std"] = np.std(v)
                    log_data[f"eval/{dataset_name}/samples"] = all_samples
                    self.log_data(log_data, step=self.step)

        self.accelerator.wait_for_everyone()

    @staticmethod
    def _state_checkpoint_staging_directory(save_directory: str) -> str:
        """Return the deterministic sibling used for atomic state publication."""
        normalized = os.path.normpath(save_directory)
        parent, basename = os.path.split(normalized)
        if not basename or basename in (".", ".."):
            raise ValueError(
                "state checkpoint destination must name a concrete directory, "
                f"received {save_directory!r}"
            )
        return os.path.join(parent, f".{basename}.flow-factory-staging")

    @staticmethod
    def _state_checkpoint_publish_claim(save_directory: str) -> str:
        """Return the sibling lock coordinating local-main publishers."""
        normalized = os.path.normpath(save_directory)
        parent, basename = os.path.split(normalized)
        if not basename or basename in (".", ".."):
            raise ValueError(
                "state checkpoint destination must name a concrete directory, "
                f"received {save_directory!r}"
            )
        return os.path.join(parent, f".{basename}.flow-factory-publish-claim")

    def _synchronize_checkpoint_phase_error(
        self,
        phase: str,
        error: Exception | None,
    ) -> None:
        """Make every rank leave a failed checkpoint phase without a barrier hang."""
        process_index = getattr(self.accelerator, "process_index", 0)
        payload = (
            None
            if error is None
            else {
                "rank": process_index,
                "type": type(error).__name__,
                "message": str(error),
            }
        )
        if getattr(self.accelerator, "num_processes", 1) <= 1:
            if error is not None:
                raise error
            return
        gathered = gather_object([payload])
        failures = tuple(item for item in gathered if item is not None)
        if not failures:
            return
        message = f"exact state checkpoint {phase} failed across ranks: {failures!r}"
        if error is not None:
            raise RuntimeError(message) from error
        raise RuntimeError(message)

    def _claim_state_checkpoint_publication(self, claim_path: str) -> bool:
        """Elect one publisher per visible filesystem using an atomic claim file."""
        os.makedirs(os.path.dirname(claim_path) or ".", exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(claim_path, flags, 0o600)
        except FileExistsError:
            return False
        try:
            owner = f"rank={getattr(self.accelerator, 'process_index', 0)}\n".encode("utf-8")
            os.write(descriptor, owner)
        finally:
            os.close(descriptor)
        return True

    def _save_exact_training_state(self, save_directory: str) -> None:
        """Write Accelerator artifacts, commit a manifest, then publish atomically."""
        staging_directory = ""
        publish_claim = ""
        preflight_error = None
        try:
            if getattr(getattr(self.accelerator, "device", None), "type", None) == "mps":
                raise RuntimeError(
                    "exact state checkpoints are unsupported on MPS because Accelerate "
                    "does not serialize the MPS RNG state required for exact resume; "
                    "set log.save_model_only=true to save resumable model weights"
                )
            if getattr(self, "_acquisition_cycle_active", False) or getattr(
                self,
                "_acquisition_cycle_incomplete",
                False,
            ):
                raise RuntimeError(
                    "exact state checkpoints require a complete acquisition boundary; "
                    "the current rollout iteration or data epoch is active or ended partially"
                )
            self._validate_distributed_runtime_children()
            self._validate_runtime_child_coverage()
            staging_directory = self._state_checkpoint_staging_directory(save_directory)
            publish_claim = self._state_checkpoint_publish_claim(save_directory)
            if os.path.lexists(save_directory):
                raise FileExistsError(
                    "exact state checkpoints are immutable and cannot overwrite an existing "
                    f"destination: {save_directory!r}"
                )
            if os.path.lexists(staging_directory):
                raise FileExistsError(
                    "exact state checkpoint staging already exists, likely from an interrupted "
                    f"save; inspect or remove it explicitly before retrying: {staging_directory!r}"
                )
            if os.path.lexists(publish_claim):
                raise FileExistsError(
                    "exact state checkpoint publication claim already exists, likely from an "
                    f"interrupted save; inspect or remove it explicitly: {publish_claim!r}"
                )
        except Exception as error:
            preflight_error = error
        self._synchronize_checkpoint_phase_error("preflight", preflight_error)
        self.accelerator.wait_for_everyone()

        save_on_each_node = self.accelerator.project_configuration.save_on_each_node
        should_publish = False
        global_claim_error = None
        if self.accelerator.is_main_process:
            try:
                should_publish = self._claim_state_checkpoint_publication(publish_claim)
                if not should_publish:
                    raise FileExistsError(
                        "exact state checkpoint publication was claimed by a concurrent "
                        f"writer after preflight: {publish_claim!r}"
                    )
            except Exception as error:
                global_claim_error = error
        self._synchronize_checkpoint_phase_error("global publisher election", global_claim_error)

        local_claim_error = None
        if (
            save_on_each_node
            and self.accelerator.is_local_main_process
            and not self.accelerator.is_main_process
        ):
            try:
                should_publish = self._claim_state_checkpoint_publication(publish_claim)
            except Exception as error:
                local_claim_error = error
        self._synchronize_checkpoint_phase_error("node publisher election", local_claim_error)

        # On a shared path, the global-main claim is visible to every node and the
        # other local mains lose election. Disable their generic Accelerate writes
        # while preserving per-node writes when each node sees its own filesystem.
        override_node_save = save_on_each_node and self.accelerator.is_local_main_process
        if override_node_save:
            self.accelerator.project_configuration.save_on_each_node = should_publish
        core_save_error = None
        try:
            self.adapter.save_checkpoint(
                save_directory=staging_directory,
                model_only=False,
                include_training_roles=True,
            )
        except Exception as error:
            core_save_error = error
        finally:
            if override_node_save:
                self.accelerator.project_configuration.save_on_each_node = save_on_each_node
        self._synchronize_checkpoint_phase_error("Accelerator artifact save", core_save_error)
        # FSDP/DeepSpeed may finish rank-local shard writes after the main process
        # returns from its own save call. Hash only after every rank has arrived.
        self.accelerator.wait_for_everyone()

        manifest_error = None
        if should_publish:
            try:
                self.runtime_state.prepare_save(staging_directory)
            except Exception as error:
                manifest_error = error
        self._synchronize_checkpoint_phase_error("runtime manifest", manifest_error)
        self.accelerator.wait_for_everyone()

        publication_error = None
        if should_publish:
            try:
                os.replace(staging_directory, save_directory)
            except Exception as error:
                publication_error = error
        self._synchronize_checkpoint_phase_error("atomic publication", publication_error)

        # Keep every filesystem's claim until every publisher has installed its
        # final directory. If one node fails, successful nodes retain an explicit
        # claim beside their final checkpoint instead of looking independently
        # retryable while another node still has only staging artifacts.
        claim_cleanup_error = None
        if should_publish:
            try:
                os.unlink(publish_claim)
            except Exception as error:
                claim_cleanup_error = error
        self._synchronize_checkpoint_phase_error(
            "publication claim cleanup",
            claim_cleanup_error,
        )
        self.accelerator.wait_for_everyone()

    def save_checkpoint(self, save_directory: str, epoch: Optional[int] = None) -> None:
        """Save trainer state to a specific path.

        A periodic checkpoint exists to be resumed from, so it carries the
        training-only roles too. A multi-role run that saved just its generator would
        come back with a freshly initialized fake score and keep training happily
        against the wrong critic.
        """
        if epoch is not None:
            save_directory = os.path.join(save_directory, f"checkpoint-{epoch}")

        if self.log_args.save_model_only:
            self.adapter.save_checkpoint(
                save_directory=save_directory,
                model_only=True,
                include_training_roles=True,
            )
        else:
            self._save_exact_training_state(save_directory)

        self.accelerator.wait_for_everyone()

    def load_checkpoint(
        self,
        path: str,
        resume_type: Optional[Literal["lora", "full", "state"]] = None,
    ) -> None:
        """Load trainer state from a specific path."""
        if resume_type == "state":
            raise RuntimeError(
                "exact training-state resume must be configured through model.resume_path "
                "and model.resume_type='state' before trainer construction, so runtime "
                "identity and child state can be validated before any prepared state mutates"
            )
        self.adapter.load_checkpoint(
            path=path,
            strict=True,
            resume_type=resume_type,
        )
        self.accelerator.wait_for_everyone()

    def cleanup(self) -> None:
        """Initiate non-blocking shutdown of async reward workers.

        Called on KeyboardInterrupt to cancel pending futures and signal
        executor threads to stop. This does NOT wait for threads to finish;
        the caller is expected to follow with os._exit() which will forcefully
        reclaim all resources including GPU memory.
        """
        # Training-side reward buffer.
        train_buf = getattr(self, "reward_buffer", None)
        if train_buf is not None:
            train_buf.shutdown(wait=False, cancel_futures=True)

        # Per-eval-dataset reward buffers.
        for buf in getattr(self, "eval_dataset_reward_buffers", {}).values():
            if buf is not None:
                buf.shutdown(wait=False, cancel_futures=True)
