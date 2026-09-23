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

import glob
import hashlib
import json
import logging

# src/flow_factory/models/abc.py
import os
import re
import shutil
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, fields
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator, DistributedType
from accelerate.state import PartialState
from accelerate.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_PATTERN_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    WEIGHTS_PATTERN_NAME,
    clean_state_dict_for_safetensors,
    has_offloaded_params,
)
from accelerate.utils.modeling import (
    get_state_dict_offloaded_model,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.outputs import BaseOutput
from huggingface_hub import split_torch_state_dict_into_shards
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from safetensors.torch import load_file, save_file

from ..contracts import PipelineIOContract
from ..ema import EMAModuleWrapper
from ..hparams import *
from ..hparams.gradient_checkpointing import GradientCheckpointingSpec
from ..samples import (
    BaseSample,
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    NoisedState,
    ReplayStep,
    StackedSampleBatch,
)
from ..scheduler import (
    SchedulerGroup,
    SDESchedulerMixin,
    SDESchedulerOutput,
)
from ..scheduler import load_scheduler as _load_scheduler
from ..utils.audio import MultiAudioBatch
from ..utils.base import filter_kwargs, is_tensor_list
from ..utils.checkpoint import (
    HF_PATH_PREFIX,
    download_hf_checkpoint,
    infer_lora_config,
    infer_target_modules,
    mapping_lora_state_dict,
    parse_hf_checkpoint_path,
)
from ..utils.image import MultiImageBatch
from ..utils.logger_utils import setup_logger
from ..utils.video import MultiVideoBatch
from . import trajectory_bridge as bridge
from .checkpointing import (
    CheckpointUnit,
    discover_gradient_checkpointing_units,
    select_gradient_checkpointing_units,
    selective_gradient_checkpointing_function,
)
from .condition_state import (
    ConditionStatePreparer,
    PreparedConditionState,
    validate_condition_preparer_required_components,
)
from .latent_geometry import LatentAxes, infer_latent_axes
from .model_bundle import RoutedComponentProxy
from .output_state import (
    DecodedMediaBatch,
    EncodedOutputState,
    OutputStateCodec,
    validate_codec_required_components,
    validate_encoded_output_state,
    validate_output_candidate_batch,
)
from .precision import (
    build_component_load_dtype_kwargs,
    cast_module_role_dtypes,
    component_dtype_mapping,
    parameter_dtype_inventory,
    resolve_component_dtype,
    validate_dtype_policy_selectors,
)
from .runtime import ClassicPipelineRuntime, ComponentRuntime
from .variants import DEFAULT_BASE_VARIANT, ComponentVariantRegistry, ComponentVariantSpec

# Constants
CONFIG_NAME = "config.json"
DIFFUSION_WEIGHTS_NAME = "diffusion_pytorch_model.bin"
DIFFUSION_WEIGHTS_PATTERN_NAME = "diffusion_pytorch_model{suffix}.bin"
DIFFUSION_WEIGHTS_INDEX_NAME = f"{DIFFUSION_WEIGHTS_NAME}.index.json"
SAFE_DIFFUSION_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
SAFE_DIFFUSION_WEIGHTS_PATTERN_NAME = "diffusion_pytorch_model{suffix}.safetensors"
SAFE_DIFFUSION_WEIGHTS_INDEX_NAME = f"{SAFE_DIFFUSION_WEIGHTS_NAME}.index.json"
LORA_ADAPTER_CONFIG_NAME = "adapter_config.json"
LORA_ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
CHECKPOINT_MANIFEST_NAME = "manifest.json"
CHECKPOINT_MANIFEST_VERSION = 1
CHECKPOINT_ROLES_DIRNAME = "roles"

logger = setup_logger(__name__)


@dataclass
class NamedParametersInfo:
    """Metadata for named parameters snapshot."""

    target_components: List[str]
    ema_wrapper: EMAModuleWrapper


@dataclass(frozen=True)
class CheckpointEntry:
    """One writable/readable artifact inside a checkpoint directory.

    A checkpoint holds one artifact per (component, role) pair. A base-only run has
    one; a multi-role run also writes its non-base variants. Save and load paths both
    derive from these entries.

    Attributes:
        component: Target component name, e.g. ``"transformer"``.
        role: Component variant that owns the weights, e.g. ``"base"`` or ``"fake"``.
        relative_path: Location inside the checkpoint, POSIX-style, ``"."`` for root.
    """

    component: str
    role: str
    relative_path: str

    def directory(self, checkpoint_path: str) -> str:
        """Return the absolute directory this entry occupies.

        Args:
            checkpoint_path: Root of the checkpoint.

        Returns:
            Absolute path to the entry's directory.
        """
        if self.relative_path == ".":
            return checkpoint_path
        return os.path.join(checkpoint_path, *self.relative_path.split("/"))


class BaseAdapter(ABC):
    """
    Abstract Base Class for Flow-Factory models.
    """

    _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    component_load_dtype_defaults: ClassVar[Any] = None

    lora_keys: List[str] = [
        "lora_A",
        "lora_B",
        "lora_magnitude_vector",  # DoRA
        "lora_embedding_A",
        "lora_embedding_B",  # Embedding LoRA
        "modules_to_save",  # Additional modules marked for saving
    ]

    # Names of ``preprocess_func`` output columns that must be surfaced in the HF
    # "python" format (returned as PIL, never tensorized) instead of the torch
    # format. They are persisted via the HuggingFace ``Image`` feature (PNG bytes)
    # rather than raw tensors, which lets the dataset store variable-size /
    # variable-count images (e.g. multi-reference I2I) that Arrow cannot serialize
    # as ragged tensors, and read them back as PIL -- see
    # ``data_utils.dataset._apply_torch_format``.
    #
    # MUST contain only genuine RGB image columns: each entry is run through PIL
    # canonicalization (``_to_pil_image_list``), so non-image data would break.
    # Empty by default and OPT-IN per adapter: only declare an output here when it
    # is a genuine RGB image that survives a PIL round-trip. Do NOT declare
    # preprocessed/non-RGB tensors (e.g. VAE-ready video tensors, latents) -- PIL
    # conversion would be lossy and break tensor consumers; non-image columns that
    # must stay python belong in ``dataset.EXTRA_PYTHON_FORMAT_COLUMNS``, not here.
    # The raw modality column ``images`` is always handled as images by the dataset
    # itself, independent of this declaration.
    python_format_columns: ClassVar[frozenset[str]] = frozenset()

    # Opt in only when every transformer forward branch runs inside a diffusers
    # ``cache_context`` while caching is enabled. The rollout cache accelerator
    # rejects the default. ``None`` preserves the existing all-policy behavior for
    # cache-ready adapters; models with narrower upstream support declare the exact
    # user-facing policy ids they accept.
    supports_diffusers_cache: ClassVar[bool] = False
    supported_diffusers_cache_policies: ClassVar[Optional[frozenset[str]]] = None
    supports_fsdp2_cpu_efficient_loading: ClassVar[bool] = False
    # Opt in when replaying a sample under a different micro-batch composition
    # changes its numerical result (for example Bagel's NaViT sequence packing).
    # Reward/optimization overlap then records rollout packs and refuses to tile
    # or replay them with different boundaries.
    requires_preserved_replay_batch_composition: ClassVar[bool] = False
    # Opt in only when FSDP2 communication overlap exceeds the model's activation headroom.
    fsdp2_use_default_stream_unshard: ClassVar[bool] = False
    fsdp2_additional_wrap_module_names: ClassVar[Tuple[str, ...]] = ()
    # The adapter may place one checkpoint inside each FSDP-wrapped block forward.
    fsdp2_use_in_forward_activation_checkpointing: ClassVar[bool] = False
    # Opt in when backward all-gather overlap exceeds the model's peak headroom.
    fsdp2_disable_backward_prefetch: ClassVar[bool] = False
    supports_ordered_references: ClassVar[bool] = False
    preprocess_cache_fields: ClassVar[frozenset[str]] = frozenset()
    preprocess_cache_version: ClassVar[str] = ""
    trajectory_component_order: ClassVar[Tuple[str, ...]] = ("latent",)
    pipeline_io_contract: ClassVar[Optional[PipelineIOContract]] = None
    # A non-empty explanation means that this adapter is intentionally online-only
    # for now. Offline trainer loading surfaces it before model weights are loaded,
    # while online algorithms may continue to construct and use the adapter.
    output_state_codec_unavailable_reason: ClassVar[Optional[str]] = None
    flow_velocity_direction: ClassVar[Literal["noise", "data"]] = "noise"
    # Model conditioning for finite-data velocity matching. These arguments are
    # deliberately independent of sampling configuration and take precedence over
    # both training arguments and dataset conditions. Conventional CFG adapters
    # inherit the neutral scale; adapters with learned guidance embeddings or
    # additional CFG branches replace the complete immutable mapping.
    offline_training_forward_overrides: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {"guidance_scale": 1.0}
    )

    # Resolution-invariant latent axis roles for the model-agnostic latent state
    # API (see `latent_geometry.py`). ``None`` means "infer from latent ndim" via
    # `resolve_latent_axes`, which covers every standard 3D/4D/5D layout. Adapters
    # with a non-standard layout may override this with an explicit `LatentAxes`.
    LATENT_AXES: ClassVar[Optional[LatentAxes]] = None

    # DDP-only knob (ignored under FSDP/DeepSpeed). Set True only for adapters
    # whose training step can leave some trainable parameters without a gradient
    # in a given iteration (e.g. Qwen-Image: guidance=None leaves the guidance
    # embedder unused). The default False lets DDP use static buckets and overlap
    # gradient all-reduce with backward; if an adapter actually needs True but
    # leaves it False, DDP fails fast at the first backward with an explicit
    # "didn't receive grad" error rather than silently producing wrong results.
    ddp_find_unused_parameters: ClassVar[bool] = False

    # Public wrappers that own a shared contract (argument ownership, validation)
    # and delegate the component-specific part to the protected hook of the same
    # name. Overriding one would silently bypass that contract, so subclasses are
    # rejected at class creation instead of at training time.
    _BOUNDARY_OWNING_METHODS: ClassVar[Tuple[str, ...]] = (
        "prepare_condition_state",
        "encode_output_state",
        "decode_output_state",
        "forward_state",
        "reduce_component_latent_values",
        "reduce_flow_matching_objective_values",
        "reduce_latent_values",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in BaseAdapter._BOUNDARY_OWNING_METHODS:
            if name in cls.__dict__:
                if name == "encode_output_state":
                    override_hint = (
                        "Provide build_output_state_codec() and "
                        "_validate_encoded_output_geometry() instead."
                    )
                elif name == "prepare_condition_state":
                    override_hint = "Provide build_condition_state_preparer() instead."
                elif name == "decode_output_state":
                    override_hint = "Override the protected hook _decode_output_state instead."
                else:
                    override_hint = f"Override the protected hook _{name} instead."
                raise TypeError(
                    f"adapter {cls.__name__} must not override BaseAdapter.{name}: it owns a "
                    f"shared contract that an override would bypass ({name} validates its "
                    f"arguments and its result on behalf of every caller). {override_hint}"
                )

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__()
        self.config = config
        self.accelerator = accelerator
        self.model_args = config.model_args
        self.training_args = config.training_args
        self.eval_args = config.eval_args
        self._mode: str = "train"  # ['train', 'eval', 'rollout']
        self._named_parameters: Dict[str, NamedParametersInfo] = {}
        self._component_load_dtype_manifest = self.component_load_dtype_defaults
        self._component_load_dtype_overrides = getattr(
            self.model_args,
            "component_load_dtypes",
            None,
        )

        # Build the component runtime while preserving the public pipeline alias.
        self.component_runtime = self.build_component_runtime()
        self.pipeline = self.component_runtime.pipeline
        if (
            getattr(self.pipeline, "scheduler", None) is None
            and "scheduler" in self.component_runtime.declared_component_names
        ):
            self.component_runtime.materialize_components(["scheduler"])
        self.pipeline.scheduler = self.load_scheduler()
        self.scheduler_group = self.build_scheduler_group()
        if self.scheduler_group.names != self.trajectory_component_order:
            raise ValueError(
                "expected scheduler group names to match trajectory_component_order "
                f"{self.trajectory_component_order}, received {self.scheduler_group.names}"
            )
        if self.scheduler_group.primary is not self.pipeline.scheduler:
            raise ValueError(
                "expected SchedulerGroup.primary to be the canonical pipeline scheduler, "
                f"received primary component {self.scheduler_group.primary_name!r}"
            )
        self._effective_pipeline_io_contract = self._resolve_pipeline_io_contract()
        if self._effective_pipeline_io_contract is not None and not isinstance(
            self._effective_pipeline_io_contract,
            PipelineIOContract,
        ):
            raise TypeError(
                f"adapter {type(self).__name__} expected _resolve_pipeline_io_contract() "
                "to return PipelineIOContract or None, received "
                f"{type(self._effective_pipeline_io_contract).__name__}: "
                f"{self._effective_pipeline_io_contract!r}"
            )

        # Compatibility alias: the runtime override mapping is the sole authoritative cache.
        self._components: Dict[str, torch.nn.Module] = cast(
            Dict[str, torch.nn.Module], self.component_runtime.override_components
        )
        self.model_args.target_components = self.component_runtime.resolve_component_names(
            self.model_args.target_components
        )

        # Build per-request input-condition realization after the component runtime
        # exists, but before any codec may consume its declaration. Like the output
        # codec, the preparer declares lifecycle metadata only.
        self._condition_state_preparer = self._build_condition_state_preparer_declaration()
        self._condition_state_encoding_modules = self._validate_condition_state_preparer_lifecycle()

        # Build target-media encoding only after load-dtype policy, component runtime,
        # scheduler group, and target-name canonicalization are established. The codec
        # declaration is immutable lifecycle metadata; it must not materialize, load,
        # move, or mutate component dtypes.
        self._output_state_codec = self._build_output_state_codec_declaration()
        self._output_state_encoding_modules = self._validate_output_state_codec_lifecycle()

        # Cache target module mapping
        self.target_module_map = self._init_target_module_map()

        # Load checkpoint.
        # 'lora'/'full' load into the unwrapped pipeline modules here (before prepare()).
        # 'state' is deferred to post_init(): accelerator.load_state() only restores into
        # modules/optimizer registered by accelerator.prepare(), which the trainer runs later.
        if self.model_args.resume_path and self.model_args.resume_type != "state":
            self.load_checkpoint(
                self.model_args.resume_path, resume_type=self.model_args.resume_type
            )

        # Merge LoRA adapters into base model when transitioning to full fine-tuning
        if self.model_args.resume_path and self.model_args.finetune_type != "lora":
            self._merge_lora_if_needed()

        # Freeze non-trainable components
        self._freeze_components()

        # Apply LoRA if needed
        if self.model_args.finetune_type == "lora":
            self.apply_lora(
                target_modules=self.model_args.target_modules,
                components=self.model_args.target_components,
                overwrite=False,  # Do not overwrite existing adapters
            )

        # Set precision
        self._mix_precision()

        # NOTE: attention-backend selection is applied later by the trainer's
        # acceleration step (AttentionBackendAccelerator, configured as a `shared`
        # entry in the acceleration block), after accelerator.prepare()/post_init()
        # and before torch.compile — see BaseTrainer._apply_shared_acceleration().
        # It is intentionally NOT set here.

        # Enable gradient checkpointing if needed
        checkpointing_enabled = getattr(
            self.training_args,
            "gradient_checkpointing_enabled",
            bool(getattr(self.training_args, "enable_gradient_checkpointing", False)),
        )
        if checkpointing_enabled:
            self.enable_gradient_checkpointing()

    # ================================== Post Init =================================
    def post_init(self):
        """Hook for additional initialization after main trainer's `accelerator.prepare`."""
        # Full training-state resume must happen here: accelerator.prepare() has now
        # registered the trainable modules and optimizer, so accelerator.load_state()
        # can actually restore model + optimizer + RNG (and any other prepared objects).
        if self.model_args.resume_path and self.model_args.resume_type == "state":
            self.load_checkpoint(self.model_args.resume_path, resume_type="state")
        self._init_ema()
        self._init_ref_parameters()

    # ============================== Latent Casting =================================
    @property
    def latent_storage_dtype(self) -> Optional[torch.dtype]:
        val = getattr(self.training_args, "latent_storage_dtype", None)
        return self._DTYPE_MAP.get(val) if val else None

    def cast_latents(
        self, latents: torch.Tensor, default_dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Cast latents to storage dtype with float16 overflow protection."""
        target = self.latent_storage_dtype or default_dtype
        if target is None or latents.dtype == target:
            return latents
        if target == torch.float16:
            abs_max = latents.abs().max().item()
            if abs_max > 65504.0:
                logger.warning(f"float16 overflow: abs_max={abs_max:.1f} > 65504, clamping.")
                latents = latents.clamp(-65504.0, 65504.0)
        return latents.to(target)

    def cast_latent_state(
        self, state: LatentState, default_dtype: Optional[torch.dtype] = None
    ) -> LatentState:
        """Cast every component latent of a state to the storage dtype.

        Multi-component adapters store and replay a whole ``LatentState`` rather than
        one tensor, and the storage dtype has to reach every component or the stored
        trajectory and the replayed one disagree on precision. Active masks are
        boolean selectors rather than latents, so they are carried over unchanged.

        Args:
            state: Structured multi-component latent state.
            default_dtype: Fallback dtype when no storage dtype is configured.

        Returns:
            State whose component latents use the storage dtype.

        Raises:
            TypeError: If ``state`` is not a ``LatentState``.
        """
        if not isinstance(state, LatentState):
            raise TypeError(
                f"expected LatentState for cast_latent_state, got {type(state).__name__}: {state!r}"
            )
        if self.latent_storage_dtype is None and default_dtype is None:
            return state
        components = {
            name: self.cast_latents(latents, default_dtype=default_dtype)
            for name, latents in state.components.items()
        }
        if all(components[name] is latents for name, latents in state.components.items()):
            return state
        return LatentState(components, active_masks=state.active_masks)

    # =========================== Condition-State Preparation =======================
    @property
    def effective_pipeline_io_contract(self) -> Optional[PipelineIOContract]:
        """Return the checkpoint-realized pipeline input/output contract."""
        return getattr(
            self,
            "_effective_pipeline_io_contract",
            type(self).pipeline_io_contract,
        )

    def _resolve_pipeline_io_contract(self) -> Optional[PipelineIOContract]:
        """Resolve checkpoint-specific capabilities from the class declaration.

        Most adapters use one immutable class-level contract. An adapter wrapping
        checkpoint variants with narrower input semantics may override this hook
        and return a validated specialization after its pipeline config is known.

        Returns:
            Effective contract for this adapter instance, or ``None``.
        """
        return type(self).pipeline_io_contract

    def _build_condition_state_preparer_declaration(
        self,
    ) -> Optional[ConditionStatePreparer]:
        """Build preparer metadata without changing component runtime state."""
        materialized_before = tuple(self.component_runtime.materialized_component_names)
        overrides_before = tuple(self.component_runtime.override_components)
        preparer = self.build_condition_state_preparer()
        materialized_after = tuple(self.component_runtime.materialized_component_names)
        overrides_after = tuple(self.component_runtime.override_components)
        if materialized_after != materialized_before or overrides_after != overrides_before:
            raise RuntimeError(
                f"adapter {type(self).__name__}.build_condition_state_preparer() must be "
                "declaration-only and cannot materialize or replace components: "
                f"materialized_before={materialized_before}, "
                f"materialized_after={materialized_after}, "
                f"overrides_before={overrides_before}, overrides_after={overrides_after}"
            )
        return preparer

    @property
    def condition_state_preparer(self) -> Optional[ConditionStatePreparer]:
        """Return the immutable preparer selected during adapter construction."""
        return getattr(self, "_condition_state_preparer", None)

    @property
    def condition_state_encoding_modules(self) -> Tuple[str, ...]:
        """Return validated component names required for condition realization."""
        return self._condition_state_encoding_modules

    def build_condition_state_preparer(self) -> Optional[ConditionStatePreparer]:
        """Build the adapter-owned runtime condition preparer, if required.

        The default identity path needs no declaration. A model whose input
        condition depends on runtime geometry, stochastic augmentation, or an
        on-device encoder returns a declaration-only preparer here.

        Returns:
            Adapter-owned preparer, or ``None`` for identity preparation.
        """
        return None

    def _validate_condition_state_preparer_lifecycle(self) -> Tuple[str, ...]:
        """Validate condition-preparer lifecycle metadata."""
        preparer = self.condition_state_preparer
        if preparer is None:
            return ()
        return validate_condition_preparer_required_components(
            preparer,
            tuple(self.component_runtime.declared_component_names),
        )

    def prepare_condition_state(
        self,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> PreparedConditionState:
        """Realize one input-owned condition state for a request or batch.

        Args:
            condition: Cached input-only model condition.
            generator: Optional generator for adapter-owned stochastic realization.

        Returns:
            Validated prepared condition reused by every candidate and forward
            derived from this request.
        """
        if not isinstance(condition, Mapping):
            raise TypeError(
                "expected condition-state input to be Mapping[str, Any], "
                f"received {type(condition).__name__}: {condition!r}"
            )
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError(
                "expected condition-state generator to be torch.Generator or None, "
                f"received {type(generator).__name__}: {generator!r}"
            )
        preparer = self.condition_state_preparer
        if preparer is None:
            return PreparedConditionState.identity(condition)
        with torch.no_grad():
            prepared = preparer.prepare_condition_state(condition, generator)
        if not isinstance(prepared, PreparedConditionState):
            raise TypeError(
                "condition-state preparer must return PreparedConditionState, "
                f"received {type(prepared).__name__}"
            )
        return prepared

    # ============================ Output-State Encoding ============================
    def _build_output_state_codec_declaration(self) -> Optional[OutputStateCodec]:
        """Build codec metadata without changing component materialization or overrides."""
        materialized_before = tuple(self.component_runtime.materialized_component_names)
        overrides_before = tuple(self.component_runtime.override_components)
        codec = self.build_output_state_codec()
        materialized_after = tuple(self.component_runtime.materialized_component_names)
        overrides_after = tuple(self.component_runtime.override_components)
        if materialized_after != materialized_before or overrides_after != overrides_before:
            raise RuntimeError(
                f"adapter {type(self).__name__}.build_output_state_codec() must be "
                "declaration-only and cannot materialize or replace components: "
                f"materialized_before={materialized_before}, "
                f"materialized_after={materialized_after}, "
                f"overrides_before={overrides_before}, overrides_after={overrides_after}"
            )
        return codec

    @property
    def output_state_codec(self) -> Optional[OutputStateCodec]:
        """Return the immutable codec selected during adapter construction.

        Returns:
            Adapter-owned output codec, or ``None`` for online-only adapters.
        """
        return self._output_state_codec

    @property
    def output_state_encoding_modules(self) -> Tuple[str, ...]:
        """Return validated component names required for target-media encoding.

        The caller owns component device staging. Keeping this declaration separate
        from :meth:`encode_output_state` prevents a per-batch encode from implicitly
        moving or offloading modules behind the trainer's back.

        Returns:
            Ordered runtime component names required for target encoding.
        """
        return self._output_state_encoding_modules

    def build_output_state_codec(self) -> Optional[OutputStateCodec]:
        """Build the adapter-owned target-media codec, if offline training is supported.

        The component runtime, canonical scheduler, and scheduler group are available
        before this hook runs. This hook declares lifecycle metadata only: it must not
        materialize, load, move, replace, or mutate the dtype of any model component.
        Online-only adapters retain the default ``None``.

        Returns:
            Adapter-owned output codec, or ``None`` when offline output is unsupported.
        """
        return None

    @classmethod
    def _validated_output_state_codec_unavailable_reason(cls) -> Optional[str]:
        """Return a normalized offline-codec blocker declared by the adapter."""
        reason = cls.output_state_codec_unavailable_reason
        if reason is None:
            return None
        if not isinstance(reason, str) or not reason.strip():
            raise TypeError(
                f"adapter {cls.__name__}.output_state_codec_unavailable_reason must be "
                f"a non-empty string or None, received {type(reason).__name__}: {reason!r}"
            )
        return reason.strip()

    @classmethod
    def validate_offline_output_capability(cls) -> None:
        """Fail before model loading unless this adapter can encode offline targets.

        A concrete codec still validates its realized components during adapter
        construction. This class-level check covers declarations that can be proven
        without downloading weights or allocating accelerator memory.

        Returns:
            None after successful static capability validation.

        Raises:
            NotImplementedError: If the adapter declares an actionable offline blocker.
            TypeError: If the contract, codec builder, or geometry hook is missing.
        """
        reason = cls._validated_output_state_codec_unavailable_reason()
        if reason is not None:
            raise NotImplementedError(
                f"offline output-state encoding is unavailable for adapter "
                f"{cls.__name__}: {reason}"
            )
        contract = cls.pipeline_io_contract
        if not isinstance(contract, PipelineIOContract):
            raise TypeError(
                f"offline training requires adapter {cls.__name__} to declare a "
                f"PipelineIOContract, received {type(contract).__name__}: {contract!r}"
            )
        if cls.build_output_state_codec is BaseAdapter.build_output_state_codec:
            raise TypeError(
                f"offline training requires adapter {cls.__name__} to provide an "
                "output-state codec through build_output_state_codec()"
            )
        if cls._validate_encoded_output_geometry is BaseAdapter._validate_encoded_output_geometry:
            raise TypeError(
                f"offline training requires adapter {cls.__name__} to override "
                "_validate_encoded_output_geometry()"
            )

    def _validate_output_state_codec_lifecycle(self) -> Tuple[str, ...]:
        """Validate the adapter's pipeline contract and codec declaration."""
        unavailable_reason = type(self)._validated_output_state_codec_unavailable_reason()
        contract = self.effective_pipeline_io_contract
        if contract is not None and not isinstance(contract, PipelineIOContract):
            raise TypeError(
                f"adapter {type(self).__name__} expected pipeline_io_contract to be "
                f"PipelineIOContract or None, received {type(contract).__name__}: {contract!r}"
            )

        codec = self.output_state_codec
        if codec is None:
            return ()
        if unavailable_reason is not None:
            raise ValueError(
                f"adapter {type(self).__name__} built an output-state codec while declaring "
                "output_state_codec_unavailable_reason; remove the stale blocker declaration"
            )
        if contract is None:
            raise ValueError(
                f"adapter {type(self).__name__} built an output-state codec without declaring "
                "pipeline_io_contract"
            )
        return validate_codec_required_components(
            codec,
            tuple(self.component_runtime.declared_component_names),
        )

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Union[Mapping[str, Any], PreparedConditionState],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        """Encode decoded targets through the adapter-owned validated boundary.

        Args:
            media_batch: Exact output-media sequence for every batch sample.
            condition: Cached or already-prepared model-input condition for the
                same batch.
            generator: Optional deterministic generator used by stochastic encoders.

        Returns:
            Detached clean output state using the adapter's latent-storage policy.

        Raises:
            NotImplementedError: If the adapter declares a known codec blocker.
            RuntimeError: If the adapter does not expose the complete offline codec seam.
            TypeError: If condition or generator has the wrong boundary type.
        """
        unavailable_reason = type(self)._validated_output_state_codec_unavailable_reason()
        if unavailable_reason is not None:
            raise NotImplementedError(
                f"offline output-state encoding is unavailable for adapter "
                f"{type(self).__name__}: {unavailable_reason}"
            )
        contract = self.effective_pipeline_io_contract
        if contract is None:
            raise RuntimeError(
                f"adapter {type(self).__name__} cannot encode output state because it does not "
                "declare pipeline_io_contract"
            )
        codec = self.output_state_codec
        if codec is None:
            raise RuntimeError(
                f"adapter {type(self).__name__} declares pipeline_io_contract but does not "
                "provide an output-state codec through build_output_state_codec()"
            )
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError(
                "expected output-state generator to be torch.Generator or None, "
                f"received {type(generator).__name__}: {generator!r}"
            )

        validated_media = validate_output_candidate_batch(media_batch, contract)
        if isinstance(condition, PreparedConditionState):
            prepared_condition = condition
        elif isinstance(condition, Mapping):
            prepared_condition = self.prepare_condition_state(condition, generator)
        else:
            raise TypeError(
                "expected output-state condition to be Mapping[str, Any] or "
                "PreparedConditionState, "
                f"received {type(condition).__name__}: {condition!r}"
            )
        codec_condition = prepared_condition.output_codec_condition()
        with torch.no_grad():
            encoded = codec.encode_output_state(
                validated_media,
                codec_condition,
                generator,
            )

        encoded = validate_encoded_output_state(
            encoded,
            contract=contract,
            expected_component_order=self.trajectory_component_order,
            expected_batch_size=len(validated_media),
            device=self.device,
        )

        # Offline targets are trajectory states too. Apply the same storage boundary
        # as online rollout after first proving that the codec returned detached state;
        # casting before validation could accidentally hide an attached source tensor.
        clean_state = self.cast_latent_state(encoded.clean_state)
        if clean_state is not encoded.clean_state:
            encoded = EncodedOutputState(
                clean_state=clean_state,
                forward_context=encoded.forward_context,
                decode_context=encoded.decode_context,
                geometry_signatures=encoded.geometry_signatures,
            )
            encoded = validate_encoded_output_state(
                encoded,
                contract=contract,
                expected_component_order=self.trajectory_component_order,
                expected_batch_size=len(validated_media),
                device=self.device,
            )

        self._validate_encoded_output_geometry(validated_media, codec_condition, encoded)
        return encoded

    def decode_output_state(
        self,
        encoded: EncodedOutputState,
        *,
        output_type: Literal["pil", "pt", "np"] = "pil",
    ) -> Any:
        """Decode one encoded offline state through the adapter's existing decoder.

        ``decode_context`` may contain geometry retained only for validation as well as
        kwargs required by a particular decoder. This wrapper forwards only names accepted
        by ``decode_latents`` and supplies the requested output type when that decoder exposes
        the standard ``output_type`` argument.

        Args:
            encoded: Validated single-component output state produced by this adapter.
            output_type: Existing decoder output representation.

        Returns:
            Model-specific decoded image or video batch.

        Raises:
            TypeError: If ``encoded`` or ``output_type`` has the wrong boundary type.
            ValueError: If the state cannot be represented by the legacy single-latent decoder.
        """
        if not isinstance(encoded, EncodedOutputState):
            raise TypeError(
                "expected encoded output state to be EncodedOutputState, "
                f"received {type(encoded).__name__}: {encoded!r}"
            )
        if type(output_type) is not str:
            raise TypeError(
                "expected output_type to be str, "
                f"received {type(output_type).__name__}: {output_type!r}"
            )
        if output_type not in ("pil", "pt", "np"):
            raise ValueError(
                "expected output_type in ('pil', 'pt', 'np'), " f"received {output_type!r}"
            )
        return self._decode_output_state(encoded, output_type=output_type)

    def _decode_output_state(
        self,
        encoded: EncodedOutputState,
        *,
        output_type: Literal["pil", "pt", "np"],
    ) -> Any:
        """Route the default single-component state through ``decode_latents``."""
        if encoded.clean_state.component_names != ("latent",):
            raise ValueError(
                "default _decode_output_state requires exactly one 'latent' component; "
                "multi-component adapters must override the protected hook, received "
                f"{encoded.clean_state.component_names}"
            )
        decode_kwargs = filter_kwargs(
            self.decode_latents,
            **dict(encoded.decode_context),
            output_type=output_type,
        )
        return self.decode_latents(
            encoded.clean_state.components["latent"],
            **decode_kwargs,
        )

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Validate codec geometry against adapter-owned input/configuration facts.

        Generic validation can prove that signatures are internally coherent, but it
        cannot prove that self-reported dimensions agree with configured geometry or
        input-media-derived constraints. Every adapter that supplies a codec must own
        that model-specific comparison explicitly.
        """
        raise NotImplementedError(
            f"adapter {type(self).__name__} provides an output-state codec but must override "
            "_validate_encoded_output_geometry() to validate geometry signatures against "
            "geometry_source="
            f"{self.effective_pipeline_io_contract.geometry_source.value!r}"
        )

    # ============================== Loading Components ==============================
    @abstractmethod
    def load_pipeline(self) -> DiffusionPipeline:
        """Load and return the diffusion pipeline. Must be implemented by subclasses."""
        pass

    def build_component_runtime(self) -> ComponentRuntime:
        """Build the component runtime used by this adapter.

        Returns:
            Classic runtime wrapping the subclass's existing ``load_pipeline`` result.
        """
        return ClassicPipelineRuntime(self.load_pipeline())

    def _load_diffusers_pipeline(
        self,
        pipeline_class: type,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> DiffusionPipeline:
        """Load an eager Diffusers pipeline with the resolved component dtype policy."""
        manifest_policy = self._component_load_dtype_manifest
        user_policy = self._component_load_dtype_overrides

        if isinstance(user_policy, torch.dtype) or (
            user_policy is None and isinstance(manifest_policy, torch.dtype)
        ):
            kwargs.update(
                {
                    key: value
                    for key, value in build_component_load_dtype_kwargs(
                        user_policy=user_policy,
                        manifest_policy=manifest_policy,
                        component_names=(),
                        transformer_names=(),
                        text_encoder_names=(),
                    ).items()
                    if key not in kwargs
                }
            )
        elif isinstance(user_policy, Mapping) or isinstance(manifest_policy, Mapping):
            load_config = getattr(pipeline_class, "load_config", None)
            if not callable(load_config):
                raise TypeError(
                    f"adapter {type(self).__name__} expected {pipeline_class.__name__}.load_config "
                    "to resolve component_load_dtypes mapping"
                )
            config_keys = {
                "cache_dir",
                "force_download",
                "proxies",
                "token",
                "local_files_only",
                "revision",
            }
            config_kwargs = {key: value for key, value in kwargs.items() if key in config_keys}
            pipeline_config = load_config(pretrained_model_name_or_path, **config_kwargs)
            component_names = [
                name
                for name, value in pipeline_config.items()
                if isinstance(value, (list, tuple)) and len(value) >= 2
            ]
            # Adapter defaults may cover several checkpoint variants of one pipeline
            # class. Keep absent class-declared optional components valid for manifest
            # validation, but resolve the actual dtype mapping only for this checkpoint.
            manifest_declared_names = list(
                dict.fromkeys(
                    [
                        *component_names,
                        *getattr(pipeline_class, "_optional_components", ()),
                    ]
                )
            )
            load_dtype_kwargs = build_component_load_dtype_kwargs(
                user_policy=user_policy,
                manifest_policy=manifest_policy,
                component_names=component_names,
                transformer_names=[name for name in component_names if "transformer" in name],
                text_encoder_names=[name for name in component_names if "text_encoder" in name],
                manifest_declared_names=manifest_declared_names,
                preserve_unselected=True,
            )
            kwargs.update(
                {key: value for key, value in load_dtype_kwargs.items() if key not in kwargs}
            )

        return pipeline_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

    def _resolve_component_load_dtype_mapping(
        self,
        *,
        component_names: Sequence[str],
        transformer_names: Sequence[str],
        text_encoder_names: Sequence[str],
    ) -> Dict[str, torch.dtype]:
        """Resolve adapter defaults and user overrides for explicit loader components."""
        return component_dtype_mapping(
            user_policy=getattr(self, "_component_load_dtype_overrides", None),
            manifest_policy=getattr(
                self,
                "_component_load_dtype_manifest",
                getattr(self, "component_load_dtype_defaults", None),
            ),
            component_names=component_names,
            transformer_names=transformer_names,
            text_encoder_names=text_encoder_names,
        )

    def load_scheduler(self) -> SDESchedulerMixin:
        """Load and return the scheduler."""
        scheduler = _load_scheduler(
            pipeline_scheduler=self.pipeline.scheduler,
            scheduler_args=self.config.scheduler_args,
        )
        return scheduler

    def build_scheduler_group(self) -> SchedulerGroup:
        """Build the default single-component scheduler group.

        Initialization calls this only after the canonical scheduler is installed.
        The scheduler setter therefore updates only the pipeline until this group exists.

        Returns:
            Scheduler group exposing the canonical scheduler as ``"latent"``.
        """
        return SchedulerGroup({"latent": self.scheduler}, primary_name="latent")

    # ============================== Component Accessors ==============================
    # ---------------------------------- Wrappers ----------------------------------
    def _unwrap(self, model: Union[torch.nn.Module, RoutedComponentProxy]) -> torch.nn.Module:
        """Get the unwrapped model.

        Peels a `RoutedComponentProxy` to its inner module first (the proxy is
        installed as a component after `accelerator.prepare`), then strips the
        accelerator wrapper (DDP/FSDP/DeepSpeed).
        """
        if isinstance(model, RoutedComponentProxy):
            model = model.inner
        return self.accelerator.unwrap_model(model)

    def set_component(self, name: str, module: Union[torch.nn.Module, RoutedComponentProxy]):
        """Install a component override for this name.

        Accepts either a real ``nn.Module`` (a LoRA wrapper, checkpoint replacement,
        accelerator-prepared module, etc.) or a transparent ``RoutedComponentProxy``
        installed after ``accelerator.prepare``. Every override is excluded from
        runtime device management. The proxy is stored as an ``nn.Module`` stand-in
        (it duck-types one), so the cast is the single, encapsulated acknowledgement
        of that contract; readers via ``get_component`` see a plain ``nn.Module``.
        """
        self.component_runtime.set_component_override(name, cast(torch.nn.Module, module))

    def get_component(self, name: str) -> torch.nn.Module:
        """Get a component, preferring the runtime override if available.

        Overrides also include LoRA/checkpoint replacement modules. After
        `accelerator.prepare`, trainable/bundled components resolve to a transparent
        `RoutedComponentProxy` (a duck-typed nn.Module stand-in) that routes forwards
        through the prepared root; use ``_unwrap`` to recover the real module.
        """
        return cast(torch.nn.Module, self.component_runtime.get_component(name))

    def has_component(self, name: str) -> bool:
        """Return whether the runtime declares a component under this canonical name.

        Component membership is owned by ``component_runtime``, not by Python
        attribute existence. ``hasattr(self, name)`` only agrees for backends whose
        components happen to be adapter properties; a modular or lazy runtime
        declares components that are never attributes, and probing with ``hasattr``
        silently drops them from lifecycle loops.
        """
        return name in self.component_runtime.declared_component_names

    def _require_component(self, name: str) -> torch.nn.Module:
        """Return a declared component, rejecting one the runtime cannot provide.

        ``get_component`` returns ``None`` for a declared-optional component that the
        backend left unset (a Wan 2.1 checkpoint has no ``transformer_2``, an
        image-only pipeline has no ``audio_vae``). Dereferencing that yields a bare
        ``AttributeError`` naming neither the component nor the loop, so callers that
        need a real module ask for it here instead.
        """
        component = self.get_component(name)
        if component is None:
            raise ValueError(
                f"expected component {name!r} of {type(self).__name__} to be available, "
                f"received None from {type(self.component_runtime).__name__}; it is "
                f"declared but unset on this checkpoint. Declared components: "
                f"{self.component_runtime.declared_component_names}"
            )
        return component

    def get_component_unwrapped(self, name: str) -> torch.nn.Module:
        """Get the original unwrapped component."""
        return cast(torch.nn.Module, self.component_runtime.get_canonical_component(name))

    def prepare_diffusers_cache(
        self,
        policy: str,
        component_name: str,
        transformer: torch.nn.Module,
    ) -> None:
        """Prepare model-specific compatibility required by a cache policy.

        Args:
            policy: User-facing diffusers cache policy identifier.
            component_name: Canonical transformer component name.
            transformer: Prepared transformer route that will receive the policy.
        """
        return None

    def get_component_config(self, name: str):
        """Get the config of a component."""
        return self.component_runtime.get_canonical_component(name).config

    def prepare_components(self, accelerator: Accelerator, component_names: List[str]):
        """Prepare specified components with the accelerator."""
        components = [
            self.component_runtime.get_canonical_component(name) for name in component_names
        ]
        prepared = accelerator.prepare(*components)
        for name, module in zip(component_names, prepared):
            self.set_component(name, module)
        return prepared

    # ------------------------------ Text Encoders & Tokenizers ------------------------------
    @property
    def text_encoder_names(self) -> List[str]:
        """Get all text encoder component names from pipeline."""
        return self.component_runtime.text_encoder_names

    @property
    def text_encoders(self) -> List[torch.nn.Module]:
        """Collect all text encoders, preferring prepared versions."""
        return [self.get_component(name) for name in self.text_encoder_names]

    @property
    def text_encoder(self) -> torch.nn.Module:
        """Get the primary text encoder."""
        return self.get_component("text_encoder")

    @text_encoder.setter
    def text_encoder(self, module: torch.nn.Module):
        self.set_component("text_encoder", module)

    @property
    def tokenizer_names(self) -> List[str]:
        """Get all tokenizer names from pipeline."""
        names = [
            name
            for name, value in vars(self.pipeline).items()
            if "tokenizer" in name and not name.startswith("_")
        ]
        return sorted(names)

    @property
    def tokenizers(self) -> List[Any]:
        """Collect all tokenizers from pipeline."""
        return [getattr(self.pipeline, name) for name in self.tokenizer_names]

    @property
    def tokenizer(self) -> Any:
        """Get the primary tokenizer."""
        tokenizers = self.tokenizers
        if not tokenizers:
            raise ValueError("No tokenizer found in the pipeline.")
        return tokenizers[0]

    # -------------------------------------- VAE --------------------------------------
    @property
    def vae(self) -> torch.nn.Module:
        """Get VAE, preferring prepared version."""
        return self.get_component("vae")

    @vae.setter
    def vae(self, module: torch.nn.Module):
        self.set_component("vae", module)

    # ------------------------------------ Audio VAE ------------------------------------
    @property
    def audio_vae(self) -> Optional[torch.nn.Module]:
        """Get audio VAE if available in pipeline, preferring prepared version."""
        if "audio_vae" not in self.component_runtime.declared_component_names:
            return None
        return self.get_component("audio_vae")

    @audio_vae.setter
    def audio_vae(self, module: torch.nn.Module):
        self.set_component("audio_vae", module)

    # ---------------------------------- Transformers ----------------------------------
    @property
    def transformer_names(self) -> List[str]:
        """Get all transformer component names."""
        return self.component_runtime.transformer_names

    @property
    def transformers(self) -> List[torch.nn.Module]:
        """Collect all transformers, preferring prepared versions."""
        return [self.get_component(name) for name in self.transformer_names]

    @property
    def transformer(self) -> torch.nn.Module:
        return self.get_component("transformer")

    @transformer.setter
    def transformer(self, module: torch.nn.Module):
        self.set_component("transformer", module)

    @property
    def transformer_config(self):
        return self.get_component_config("transformer")

    # ------------------------------------ Scheduler ------------------------------------
    @property
    def scheduler(self) -> SDESchedulerMixin:
        return self.pipeline.scheduler

    @scheduler.setter
    def scheduler(self, scheduler: Union[SDESchedulerMixin, SchedulerMixin]):
        """Set the canonical scheduler and refresh an initialized scheduler group."""
        self.pipeline.scheduler = scheduler
        if hasattr(self, "scheduler_group"):
            schedulers = dict(self.scheduler_group)
            schedulers[self.scheduler_group.primary_name] = scheduler
            self.scheduler_group = SchedulerGroup(
                schedulers,
                primary_name=self.scheduler_group.primary_name,
            )

    # ---------------------------------- Device & Dtype ----------------------------------
    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def _inference_dtype(self) -> torch.dtype:
        """Get inference dtype based on mixed precision setting."""
        if self.config.mixed_precision == "fp16":
            return torch.float16
        elif self.config.mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32

    # ============================== Mode Management ==============================

    @property
    def mode(self) -> str:
        """Get current mode."""
        return self._mode

    def eval(self):
        """Set all target components to evaluation mode."""
        self._mode = "eval"
        for name in self.trainable_component_names:
            self.get_component(name).eval()
        self.scheduler_group.eval()

    def rollout(self, *args, **kwargs):
        """Set model to rollout mode."""
        self._mode = "rollout"
        for name in self.trainable_component_names:
            self.get_component(name).eval()
        self.scheduler_group.rollout(*args, **kwargs)

    def train(self, mode: bool = True):
        """Set trainable components to training mode."""
        self._mode = "train" if mode else "eval"
        for name in self.trainable_component_names:
            self.get_component(name).train(mode)
        self.scheduler_group.train(mode=mode)

    def set_trajectory_seed(self, seed: int) -> None:
        """Set scheduler seeds in authoritative component order.

        Args:
            seed: Seed passed unchanged to every component scheduler.
        """
        self.scheduler_group.set_seed(seed)

    # ============================== Target Modules ==============================
    @property
    def default_target_modules(self) -> List[str]:
        """Default target modules for training."""
        return ["to_q", "to_k", "to_v", "to_out.0"]

    @property
    def preprocessing_modules(self) -> List[str]:
        """Modules that are requires for preprocessing"""
        return ["text_encoders", "vae"]

    @property
    def inference_modules(self) -> List[str]:
        """Modules that are required for inference and forward"""
        return ["transformer", "vae"]

    @property
    def trainable_component_names(self) -> List[str]:
        """Names of components with trainable parameters."""
        return [comp for comp, mods in self.target_module_map.items() if mods]

    @property
    def trainable_components(self) -> List[torch.nn.Module]:
        """Prepared model objects with trainable parameters."""
        return [self.get_component(name) for name in self.trainable_component_names]

    # ============================== Component Variants ==============================
    """
        A base-only run trains one copy of each target component. Multi-role runs keep
        several copies live at once, each with its own optimizer groups. The
        named-parameter snapshots below are a *temporal* mechanism (one set of weights
        installed at a time); variants are the *spatial* one.

        This adapter supplies the mechanism and holds no opinion about what a variant
        means. The caller chooses every name, decides which variant is trained when,
        and decides which one an export writes. Under `lora` each variant is a PEFT
        adapter on one shared base; under `full` each gets its own component copy.
    """

    def declare_component_variants(self, trainable_variants: Sequence[str]) -> None:
        """Create live copies of the trainable components, one per named variant.

        Must run before ``accelerator.prepare`` so the bundle sees every member,
        hence the refusal to reconfigure an existing registry.

        A frozen reference is not a variant: it is the same weights at another
        point in time, so callers use
        ``use_ref_parameters()`` (the pre-finetune weights) or its own named
        snapshot instead of declaring a copy that would cost a bundle member.

        Args:
            trainable_variants: Every trainable variant name, base variant first.
                The base owns the canonical components and every later variant is
                layered on it; a base-only run passes one name.

        Raises:
            RuntimeError: If variants are already declared or frozen.
            ValueError: If the finetune type has no variant storage mode.
        """
        existing_registry = getattr(self, "component_variant_registry", None)
        if existing_registry is not None:
            state = "frozen" if existing_registry.is_frozen else "already declared"
            raise RuntimeError(
                f"cannot declare component variants: registry is {state}; "
                f"existing variants are {existing_registry.variant_names!r}"
            )

        finetune_type = self.model_args.finetune_type
        if finetune_type not in ("lora", "full"):
            raise ValueError(
                "expected model finetune_type to be 'lora' or 'full' for component variants, "
                f"received {finetune_type!r}"
            )
        trainable_variants = ComponentVariantRegistry._validate_required_variants(
            trainable_variants
        )
        base_variant = trainable_variants[0]
        registry = ComponentVariantRegistry(self)
        for variant_name in trainable_variants:
            is_base = variant_name == base_variant
            if finetune_type == "lora":
                component_routes = {
                    component_name: component_name
                    for component_name in self.trainable_component_names
                }
                adapter_name = "default" if is_base else variant_name
            else:
                component_routes = {
                    component_name: (
                        component_name if is_base else f"{variant_name}__{component_name}"
                    )
                    for component_name in self.trainable_component_names
                }
                adapter_name = None
            registry.declare(
                ComponentVariantSpec(
                    name=variant_name,
                    storage_mode=finetune_type,
                    component_routes=component_routes,
                    adapter_name=adapter_name,
                )
            )
        registry.materialize(trainable_variants)
        self.component_variant_registry = registry
        self._apply_trainable_dtype_to_variants(
            registry,
            tuple(name for name in registry.variant_names if name != base_variant),
        )

    def _apply_trainable_dtype_to_variants(
        self,
        registry: ComponentVariantRegistry,
        variant_names: Sequence[str],
    ) -> None:
        """Give selected variants the configured trainable parameter dtype.

        ``_mix_precision`` runs while the adapter is built and only sees the base
        components; these variants are materialized later, from PEFT defaults, so
        without this a non-base variant could train in fp32 beside a bf16 base variant:
        twice the memory, a different numerical path, and every RMSNorm it feeds
        falling off the fused kernel because its weight no longer matches the
        activations.

        Args:
            registry: Registry holding the materialized variants.
            variant_names: Variants to align after creation or checkpoint load.
        """
        train_dtype = self.model_args.trainable_parameters_dtype
        if train_dtype is None:
            return
        target_dtype = self._DTYPE_MAP.get(train_dtype, train_dtype)
        if not isinstance(target_dtype, torch.dtype):
            raise TypeError(
                "expected model trainable_parameters_dtype to name a torch dtype, received "
                f"{type(train_dtype).__name__}: {train_dtype!r}"
            )
        for variant_name in variant_names:
            for parameter in registry.parameters(variant_name):
                if parameter.is_floating_point() and parameter.dtype != target_dtype:
                    parameter.data = parameter.data.to(target_dtype)

    def align_component_variant_dtypes(self) -> None:
        """Reapply dtype policy after checkpoint loaders mutate adapter weights."""
        registry = self._require_variant_registry("dtype alignment")
        self._apply_trainable_dtype_to_variants(registry, registry.variant_names)

    def _require_variant_registry(self, purpose: str) -> ComponentVariantRegistry:
        """Return the variant registry, naming the caller that needs it."""
        registry = getattr(self, "component_variant_registry", None)
        if registry is None:
            raise RuntimeError(
                f"adapter={type(self).__name__!r} has no component variant registry for "
                f"{purpose}; call declare_component_variants() before accelerator.prepare"
            )
        return registry

    @contextmanager
    def use_component_variant(self, variant_name: str) -> Iterator[None]:
        """Temporarily route component lookups and forwards to one variant.

        Args:
            variant_name: Declared variant to activate.

        Yields:
            Control while the requested variant is active.

        Raises:
            RuntimeError: If no component variants were declared.
        """
        registry = self._require_variant_registry(f"variant {variant_name!r}")
        with registry.use(variant_name):
            yield

    def variant_parameters(self, variant_name: str) -> Tuple[torch.nn.Parameter, ...]:
        """Return the parameters one variant owns, for a caller-built optimizer group.

        Args:
            variant_name: Declared variant.

        Returns:
            The variant's parameters, in registration order.

        Raises:
            RuntimeError: If no component variants were declared.
        """
        registry = self._require_variant_registry(f"variant {variant_name!r} parameters")
        return registry.parameters(variant_name)

    def _merge_module_pattern(
        self, current_pattern: Union[str, List[str], Set[str]], new_pattern: str
    ) -> Union[str, Set[str]]:
        """
        Resolve pattern and merge into current modules.

        Args:
            current: Current state ('all' or list of modules)
            pattern: New pattern to merge ('all', 'default', or module name)

        Returns:
            'all' or updated module list
        """
        # 'all' is absorbing - once set, stays 'all'
        if current_pattern == "all" or new_pattern == "all":
            return "all"

        # Resolve pattern to module list
        new_modules = self.default_target_modules if new_pattern == "default" else [new_pattern]
        new_pattern_set = set(current_pattern) | set(new_modules)
        return new_pattern_set

    def _parse_target_modules(
        self, target_modules: Union[str, List[str]], components: Union[str, List[str]]
    ) -> Dict[str, Union[List[str], None]]:
        """
        Parse target_modules config into component-specific mapping.

        Args:
            target_modules:
                - 'default': Use self.default_target_modules
                - 'all': Unfreeze all parameters
                - str: Single module pattern
                - List[str]: Module patterns with optional component prefix
            components: Union[str, List[str]]
                - Component(s) to apply target_modules to.

        Returns:
            Dict mapping component names to their target modules.
            Example: {
                'transformer': ['attn.to_q', 'attn.to_k'],
                'transformer_2': 'all',
                'transformer_3': None
            }
        """
        # Normalize components to list
        if isinstance(components, str):
            components = [components]
        if isinstance(target_modules, str):
            target_modules = [target_modules]

        component_map = {comp: set() for comp in components}

        for module in target_modules:
            parts = module.split(".", 1)
            if len(parts) == 2 and parts[0] in components:
                component_map[parts[0]] = self._merge_module_pattern(
                    component_map[parts[0]], parts[1]
                )
            else:
                for comp in components:
                    component_map[comp] = self._merge_module_pattern(component_map[comp], module)

        # Remove duplicates and handle empty lists
        component_map = {
            comp: (
                "all" if mods == "all" else sorted(mods) if mods else None
            )  # Keep None here, to enable `accelerator.prepare` for non-trainable module to save mem.
            for comp, mods in component_map.items()
        }

        return component_map

    def _init_target_module_map(self) -> Dict[str, Union[List[str], None]]:
        """
        Initialize and cache target module mapping from config.

        Returns:
            Dict mapping component names to their target modules.
        """
        component_map = self._parse_target_modules(
            target_modules=self.model_args.target_modules,
            components=self.model_args.target_components,
        )

        return component_map

    # ============================== EMA Management ==============================
    def _ema_tracked_parameters(self) -> List[torch.nn.Parameter]:
        """Return live prepared parameters owned by the base trainable variant.

        FSDP2 replaces module parameters with DTensors during ``prepare``. The
        variant registry is explicitly rebound to those identities, while an
        adapter component lookup can still expose a pre-prepare inner module.
        EMA/reference swaps must therefore use registry ownership whenever it is
        available or they mix plain tensors with live DTensors.
        """
        registry = getattr(self, "component_variant_registry", None)
        if isinstance(registry, ComponentVariantRegistry):
            parameters = list(registry.parameters(registry.base_variant))
        else:
            parameters = self.get_trainable_parameters()
        if not parameters:
            raise RuntimeError(
                "expected at least one live trainable parameter for EMA/reference tracking, "
                f"received none for adapter={type(self).__name__!r}"
            )
        return parameters

    def _init_ema(self):
        """Initialize EMA wrapper for the transformer."""
        if self.training_args.ema_decay > 0:
            ema_device = (
                self.accelerator.device
                if self.training_args.ema_device == "cuda"
                else torch.device("cpu")
            )
            self.ema_wrapper = EMAModuleWrapper(
                parameters=self._ema_tracked_parameters(),
                decay=self.training_args.ema_decay,
                update_step_interval=self.training_args.ema_update_interval,
                device=ema_device,
                decay_schedule=self.training_args.ema_decay_schedule,
                # Pass decay schedule params from training_args
                **self.training_args,
            )
        else:
            self.ema_wrapper = None

    def ema_step(self, step: int):
        """Update EMA parameters."""
        if hasattr(self, "ema_wrapper") and self.ema_wrapper is not None:
            self.ema_wrapper.step(self._ema_tracked_parameters(), optimization_step=step)

    @contextmanager
    def use_ema_parameters(self):
        if hasattr(self, "ema_wrapper") and self.ema_wrapper is not None:
            trainable_params = self._ema_tracked_parameters()
            with self.ema_wrapper.use_ema_parameters(trainable_params):
                yield
        else:
            yield

    @contextmanager
    def use_variant_snapshot(self, snapshot_name: str) -> Iterator[None]:
        """Temporarily install a variant-local parameter EMA snapshot.

        The caller names the snapshot; this adapter attaches no meaning to the
        name. The caller decides when an EMA snapshot should be installed or exported.

        Args:
            snapshot_name: Snapshot registered through the variant registry.

        Yields:
            Control while the snapshot's parameters are installed.

        Raises:
            RuntimeError: If no component variants were declared.
        """
        registry = self._require_variant_registry("parameter EMA")
        with registry.use_snapshot(snapshot_name):
            yield

    def declare_variant_snapshot(self, variant_name: str, snapshot_name: str) -> None:
        """Register a parameter EMA that tracks one trainable variant.

        Args:
            variant_name: Trainable variant the snapshot follows.
            snapshot_name: Unique identifier the caller will read it back by.

        Raises:
            RuntimeError: If no component variants were declared.
        """
        registry = self._require_variant_registry("parameter EMA")
        registry.add_snapshot(variant_name, snapshot_name)

    def has_variant_snapshot(self, snapshot_name: str) -> bool:
        """Report whether a parameter EMA snapshot exists.

        Args:
            snapshot_name: Identifier to look up.

        Returns:
            Whether the snapshot has been declared.
        """
        registry = self._require_variant_registry("parameter EMA")
        return registry.has_snapshot(snapshot_name)

    def update_variant_snapshot(self, snapshot_name: str, decay: float) -> None:
        """Advance a parameter EMA toward its variant's live parameters.

        Args:
            snapshot_name: Existing snapshot identifier.
            decay: Weight kept on the existing snapshot, in ``[0, 1]``.

        Raises:
            RuntimeError: If no component variants were declared.
        """
        registry = self._require_variant_registry("parameter EMA")
        registry.update_snapshot(snapshot_name, decay)

    def get_variant_snapshot(self, snapshot_name: str) -> Tuple[torch.Tensor, ...]:
        """Return the raw tensors of a variant-local parameter EMA snapshot.

        Args:
            snapshot_name: Snapshot registered through the variant registry.

        Returns:
            The snapshot's parameter tensors, for a caller that exports them.

        Raises:
            RuntimeError: If no component variants were declared.
        """
        registry = self._require_variant_registry("parameter EMA")
        return registry.snapshot_tensors(snapshot_name)

    # ============================== Reference Parameters ==============================
    def _init_ref_parameters(self):
        """
        Initialize reference parameters for target components.
        Used for KL regularization during training.
        """
        if self.training_args.requires_ref_model and self.model_args.finetune_type in ["full"]:
            ref_param_device = (
                self.accelerator.device
                if self.training_args.ref_param_device == "cuda"
                else torch.device("cpu")
            )
            self._ref_ema = EMAModuleWrapper(
                parameters=self._ema_tracked_parameters(),
                decay=0.0,  # No decay,
                update_step_interval=0,  # No updates, just store original weights
                device=ref_param_device,
            )
        else:
            self._ref_ema = None

    @contextmanager
    def use_ref_parameters(self):
        """Context manager to use reference parameters."""
        if self.model_args.finetune_type == "lora":
            # The restoration below has to happen after PEFT's disable_adapter contexts
            # have unwound, since leaving one re-marks only the active adapter, so the
            # ExitStack is closed before it runs rather than around it.
            try:
                with ExitStack() as stack:
                    enabled_any = False
                    for comp_name in self.target_module_map.keys():
                        component = (
                            self.get_component(comp_name) if self.has_component(comp_name) else None
                        )
                        if component is not None:
                            unwrapped = self._unwrap(component)

                            # Handle Compiled Models (torch.compile)
                            if hasattr(unwrapped, "_orig_mod"):
                                unwrapped = unwrapped._orig_mod

                            if isinstance(unwrapped, PeftModel):
                                # Enter disable_adapter context for each component
                                stack.enter_context(unwrapped.disable_adapter())
                                enabled_any = True
                    if not enabled_any:
                        logger.warning("No LoRA adapters found to disable in use_ref_parameters")

                    yield
            finally:
                # A multi-role objective that queries its frozen reference between a
                # role's forward and its backward would otherwise lose that role's
                # gradients with no error: both non-base roles can become frozen across
                # this context, leaving their optimizers with nothing to step.
                self._restore_variant_trainability()

        elif self._ref_ema is not None:
            trainable_params = self._ema_tracked_parameters()
            # If ref_ema is on CPU, this line will be very slow!
            with self._ref_ema.use_ema_parameters(trainable_params):
                yield
        else:
            yield

    def _restore_variant_trainability(self) -> None:
        """Reassert active routing and mark every variant parameter trainable.

        A no-op for a base-only run, which declares no variants.
        """
        registry = getattr(self, "component_variant_registry", None)
        if registry is None:
            return
        # PEFT's reference context mutates both requires_grad flags and adapter
        # routing. Merely restoring flags leaves activation-checkpoint
        # recomputation free to run with the disabled/default adapter.
        registry.activate(registry.active_variant)

    # ============================== Named Parameters Snapshot ==============================
    """
        These utilities help to snapshot and restore named parameters for target components.
        NOTE: `use_ref_parameters` always refers to the original model weights before any fine-tuning.

        A caller that refreshes its frozen reference during training, or that keeps more
        than one such copy, needs something more flexible than that single fixed reference.
        The functions below store, use, update, and remove named parameter snapshots, and
        attach no meaning to the names.
    """

    def _get_component_parameters(self, component_names: List[str]) -> List[torch.nn.Parameter]:
        """Get trainable parameters from specified components."""
        params = []
        for comp_name in component_names:
            if self.has_component(comp_name):
                component = self._require_component(comp_name)
                params.extend(p for p in component.parameters() if p.requires_grad)
            else:
                logger.warning(f"Component '{comp_name}' not found in the model. Skipping.")
        return params

    def add_named_parameters(
        self,
        name: str,
        target_components: Optional[Union[str, List[str]]] = None,
        device: Optional[Union[torch.device, str]] = None,
        overwrite: bool = True,
    ) -> None:
        """
        Store current trainable parameters snapshot under a name.

        Args:
            name: Identifier for this parameter snapshot
            target_components: Component names to store. Defaults to components with trainable params.
            device: Storage device (defaults to 'cpu')
            overwrite: Whether to overwrite existing snapshot
        """
        if name in self._named_parameters and not overwrite:
            raise KeyError(f"Named parameters '{name}' exists. Use overwrite=True.")

        # Normalize target components - keep only those with trainable params
        if target_components is None:
            target_components = [k for k, v in self.target_module_map.items() if v]
        elif isinstance(target_components, str):
            target_components = [target_components]

        # Validate
        invalid = set(target_components) - set(self.target_module_map.keys())
        if invalid:
            raise ValueError(
                f"expected target_components to name declared components "
                f"{list(self.target_module_map.keys())}, received unknown {sorted(invalid)}"
            )

        device = torch.device(device) if device else torch.device("cpu")
        params = self._get_component_parameters(target_components)

        if not params:
            raise ValueError(f"No trainable parameters found in {target_components}")

        self._named_parameters[name] = NamedParametersInfo(
            target_components=target_components,
            ema_wrapper=EMAModuleWrapper(
                parameters=params,
                decay=0.0,
                update_step_interval=0,
                device=device,
            ),
        )
        logger.info(f"Stored named parameters '{name}' for {target_components} on {device}")

    @contextmanager
    def use_named_parameters(self, name: str):
        """
        Context manager to temporarily use named parameters.

        Args:
            name: Name of stored parameters snapshot

        Usage:
            adapter.add_named_parameters('init')
            # ... training ...
            with adapter.use_named_parameters('init'):
                evaluate(model)  # Uses stored weights
            # Current weights restored
        """
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found. Available: {self.list_named_parameters()}")

        info = self._named_parameters[name]
        params = self._get_component_parameters(info.target_components)

        with info.ema_wrapper.use_ema_parameters(params):
            yield

    def update_named_parameters(
        self,
        name: str,
        target_components: Optional[Union[str, List[str]]] = None,
        new_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
    ) -> None:
        """
        Update existing named parameters with specified or current values.

        Args:
            name: Name of snapshot to update
            target_components: Components to update. Defaults to originally stored components.
            new_parameters: Parameters to copy from. Defaults to current model parameters.
        """
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found.")

        info = self._named_parameters[name]

        # Resolve target components
        if target_components is None:
            target_components = info.target_components
        elif isinstance(target_components, str):
            target_components = [target_components]

        if not set(target_components).issubset(set(info.target_components)):
            raise ValueError(
                f"expected target_components for snapshot {name!r} to be a subset of the "
                f"components it stored {info.target_components!r}, received "
                f"{sorted(set(target_components) - set(info.target_components))!r}"
            )

        # Resolve parameters
        if new_parameters is None:
            new_parameters = self._get_component_parameters(target_components)
        else:
            new_parameters = list(new_parameters)

        # Validate param count
        if len(new_parameters) != len(info.ema_wrapper.ema_parameters):
            raise ValueError(
                f"Parameter count mismatch: got {len(new_parameters)}, "
                f"expected {len(info.ema_wrapper.ema_parameters)}"
            )

        # Update
        with torch.no_grad():
            for ema_param, param in zip(
                info.ema_wrapper.ema_parameters, new_parameters, strict=True
            ):
                ema_param.data.copy_(param.detach().to(ema_param.device))

        logger.info(f"Updated named parameters '{name}'")

    def remove_named_parameters(self, name: str) -> None:
        """Remove named parameters."""
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found.")
        del self._named_parameters[name]
        logger.info(f"Removed named parameters '{name}'")

    def list_named_parameters(self) -> List[str]:
        """List all stored parameter names."""
        return list(self._named_parameters.keys())

    def get_named_parameters_info(self, name: str) -> Dict[str, Any]:
        """Get info about a named parameter snapshot."""
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found.")
        info = self._named_parameters[name]
        return {
            "name": name,
            "target_components": info.target_components,
            "num_params": len(info.ema_wrapper.ema_parameters),
            "device": str(info.ema_wrapper.device),
        }

    def get_named_parameters(self, name: str) -> List[torch.nn.Parameter]:
        """
        Get the stored parameter tensors for a named snapshot.

        Args:
            name: Identifier of the stored snapshot.

        Returns:
            List[torch.nn.Parameter]: The stored parameter tensors.
        """
        if name not in self._named_parameters:
            raise KeyError(f"'{name}' not found. Available: {self.list_named_parameters()}")
        return self._named_parameters[name].ema_wrapper.ema_parameters

    # ============================== Gradient Checkpointing ==============================
    def _gradient_checkpointing_root(self, component: torch.nn.Module) -> torch.nn.Module:
        """Peel distributed and PEFT wrappers before configuring checkpoint units."""
        root = self._unwrap(component)
        get_base_model = getattr(root, "get_base_model", None)
        if callable(get_base_model):
            root = get_base_model()
        if not isinstance(root, torch.nn.Module):
            raise TypeError(
                "expected gradient checkpointing root as nn.Module, "
                f"received {type(root).__name__}: {root!r}"
            )
        return root

    def _gradient_checkpointing_units(
        self,
        component_name: str,
        component: torch.nn.Module,
    ) -> List[CheckpointUnit]:
        """Return checkpointable blocks in their registered execution order."""
        del component_name
        return discover_gradient_checkpointing_units(component)

    @staticmethod
    def _enable_full_gradient_checkpointing(
        component_name: str,
        component: torch.nn.Module,
    ) -> None:
        """Bridge Diffusers and Transformers full-checkpointing APIs."""
        enable = getattr(component, "enable_gradient_checkpointing", None)
        if callable(enable):
            enable()
            return
        enable = getattr(component, "gradient_checkpointing_enable", None)
        if callable(enable):
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            return
        raise TypeError(
            f"component {component_name!r} ({type(component).__name__}) does not expose "
            "enable_gradient_checkpointing() or gradient_checkpointing_enable()"
        )

    def enable_gradient_checkpointing(self) -> None:
        """Apply the normalized full or selective policy to target components."""
        policy = self.training_args.enable_gradient_checkpointing
        for comp_name in self.model_args.target_components:
            component = self.get_component(comp_name)
            root = self._gradient_checkpointing_root(component)
            if isinstance(policy, bool) or policy.mode == "full":
                self._enable_full_gradient_checkpointing(comp_name, root)
                logger.info("Enabled full gradient checkpointing for %s", comp_name)
                continue
            if policy.mode == "none":
                continue

            enable = getattr(root, "enable_gradient_checkpointing", None)
            if not callable(enable):
                raise TypeError(
                    f"selective gradient checkpointing for component {comp_name!r} "
                    f"requires enable_gradient_checkpointing(custom_func), received "
                    f"{type(root).__name__}"
                )
            units = self._gradient_checkpointing_units(comp_name, root)
            selected = select_gradient_checkpointing_units(policy, units)
            enable(selective_gradient_checkpointing_function(selected))
            logger.info(
                "Enabled selective gradient checkpointing for %s: mode=%s, selected=%d/%d",
                comp_name,
                policy.mode,
                len(selected),
                len(units),
            )
            logger.debug(
                "Gradient checkpoint units for %s -> %s",
                comp_name,
                [name for name, _ in selected],
            )

    def disable_gradient_checkpointing(self) -> None:
        """Disable checkpointing on every materialized trainable variant."""
        registry = getattr(self, "component_variant_registry", None)
        if isinstance(registry, ComponentVariantRegistry):
            components = tuple(registry.bundle_members().values())
        else:
            components = tuple(
                self.get_component(name)
                for name in self.model_args.target_components
                if self.has_component(name)
            )
        seen = set()
        for component in components:
            if id(component) in seen:
                continue
            seen.add(id(component))
            root = self._gradient_checkpointing_root(component)
            disable = getattr(root, "disable_gradient_checkpointing", None)
            if callable(disable):
                disable()
                logger.info("Disabled gradient checkpointing for %s", type(component).__name__)
                continue
            disable = getattr(root, "gradient_checkpointing_disable", None)
            if callable(disable):
                disable()
                logger.info("Disabled gradient checkpointing for %s", type(component).__name__)
                continue
            logger.warning(
                "%s does not support disabling gradient checkpointing",
                type(component).__name__,
            )

    # ============================== Precision Management ==============================
    def _cast_module_mixed_precision(
        self,
        name: str,
        component: torch.nn.Module,
        train_dtype: torch.dtype,
        frozen_dtype: Optional[torch.dtype],
        *,
        force_uniform_dtype: Optional[torch.dtype] = None,
    ) -> int:
        """
        Set floating-point parameters/buffers without a trainable round-trip through frozen_dtype.

        Trainable parameters use ``train_dtype``. Frozen parameters and floating-point buffers use
        ``frozen_dtype`` when it is set, or are left at their loaded dtype when ``frozen_dtype`` is
        ``None`` (no post-load mutation). Integer/bool buffers are left unchanged.
        """
        result = cast_module_role_dtypes(
            component,
            component_name=name,
            trainable_dtype=train_dtype,
            frozen_dtype=frozen_dtype,
            force_uniform_dtype=force_uniform_dtype,
            is_adapter_parameter=lambda parameter_name: any(
                key in parameter_name for key in self.lora_keys
            ),
        )
        if result.protected:
            logger.info(
                "Preserved %d model-protected FP32 parameter/buffer entries in component %r",
                result.protected,
                name,
            )
        return result.trainable

    def _log_component_precision_inventory(
        self,
        name: str,
        component: torch.nn.Module,
        *,
        stage: str,
    ) -> None:
        """Log effective trainable/frozen parameter storage after dtype policy."""
        if not self.accelerator.is_main_process:
            return
        inventory = parameter_dtype_inventory(component)
        rendered = {
            role: {str(dtype): count for dtype, count in sorted(values.items(), key=str)}
            for role, values in inventory.items()
        }
        logger.info(
            "Precision inventory stage=%s component=%s %s",
            stage,
            name,
            rendered,
        )

    def _apply_component_precision_policy(
        self,
        name: str,
        component: torch.nn.Module,
    ) -> None:
        """Apply the configured original-dtype policy to one materialized module."""
        train_dtype = self.model_args.trainable_parameters_dtype
        frozen_dtype = self._frozen_dtype_for_component(name)
        if name in self.model_args.target_components:
            if self._is_fsdp2() and self.accelerator.mixed_precision != "no":
                self._cast_module_mixed_precision(
                    name,
                    component,
                    train_dtype,
                    frozen_dtype,
                    force_uniform_dtype=torch.float32,
                )
            else:
                self._cast_module_mixed_precision(
                    name,
                    component,
                    train_dtype,
                    frozen_dtype,
                )
        elif frozen_dtype is not None:
            self._cast_module_mixed_precision(
                name,
                component,
                train_dtype,
                frozen_dtype,
                force_uniform_dtype=frozen_dtype,
            )
        self._log_component_precision_inventory(name, component, stage="materialize")

    def _frozen_dtype_for_component(self, name: str) -> Optional[torch.dtype]:
        """Return one component's frozen dtype or ``None`` for no mutation."""
        policy = getattr(self.model_args, "frozen_parameters_dtype", None)
        if not getattr(self, "_frozen_dtype_policy_validated", False):
            validate_dtype_policy_selectors(
                policy,
                declared_names=self.component_runtime.declared_component_names,
            )
            self._frozen_dtype_policy_validated = True
        return resolve_component_dtype(
            name,
            user_policy=policy,
            manifest_policy=None,
            transformer_names=self.transformer_names,
            text_encoder_names=self.text_encoder_names,
        )

    def _mix_precision(self):
        """Set trainable params to ``trainable_parameters_dtype``; by default leave frozen params
        and floating-point buffers at their loaded (``from_pretrained``) dtype.

        This is the single place that decides every parameter's *original* dtype before
        ``accelerator.prepare``; the trainer only bundles + prepares.

        Frozen-dtype policy: a scalar ``frozen_parameters_dtype`` applies to every
        component. A mapping resolves concrete component, component-group, then
        ``default`` values in descending priority. A null resolved value performs
        no post-load dtype mutation.

        FSDP2 caveat: FSDP2 shards each unit with ONE original dtype, and accelerate upcasts the
        trainable params to an fp32 master when ``mixed_precision != 'no'``. So a trained component
        that also bundles frozen members (e.g. Wan2.2 trains ``transformer`` while ``transformer_2``
        is frozen-but-sharded) would otherwise mix fp32/low-precision within a unit and trip FSDP2's
        uniform-dtype assert. We therefore force the TRAINED components to a uniform fp32 original
        dtype here (compute stays low-precision via accelerate's ``MixedPrecisionPolicy``); untrained
        components use their resolved frozen dtype when set, else preserve the loaded
        checkpoint dtype.
        """
        train_dtype = self.model_args.trainable_parameters_dtype

        target_set = frozenset(self.model_args.target_components)
        component_names = self._resolve_component_names(None)
        merged_names = list(dict.fromkeys([*component_names, *self.model_args.target_components]))

        # FSDP2: trained (sharded) components need a uniform fp32 original dtype.
        if self._is_fsdp2() and self.accelerator.mixed_precision != "no":
            for name in merged_names:
                if name in target_set:
                    self._cast_module_mixed_precision(
                        name,
                        self.get_component(name),
                        train_dtype,
                        self._frozen_dtype_for_component(name),
                        force_uniform_dtype=torch.float32,
                    )
                else:
                    frozen_dtype = self._frozen_dtype_for_component(name)
                    if frozen_dtype is not None:
                        self._cast_module_mixed_precision(
                            name,
                            self.get_component(name),
                            train_dtype,
                            frozen_dtype,
                            force_uniform_dtype=frozen_dtype,
                        )
                # else: preserve the untrained component's loaded dtype
            frozen_policy = {
                name: self._frozen_dtype_for_component(name)
                for name in merged_names
                if name not in target_set
            }
            logger.info(
                "FSDP2 precision: configured trainable storage=%s, "
                "effective original/master=torch.float32, compute=%s; "
                "frozen component policy -> %s",
                train_dtype,
                self.accelerator.mixed_precision,
                frozen_policy,
            )
            for name in merged_names:
                self._log_component_precision_inventory(
                    name,
                    self.get_component(name),
                    stage="initialize",
                )
            return

        # Split: trainable -> train_dtype; frozen -> frozen_dtype, or preserved when None.
        trainable_count = 0
        frozen_policy = {}
        for name in merged_names:
            component = self.get_component(name)
            frozen_dtype = self._frozen_dtype_for_component(name)
            frozen_policy[name] = frozen_dtype
            if name in target_set:
                trainable_count += self._cast_module_mixed_precision(
                    name,
                    component,
                    train_dtype,
                    frozen_dtype,
                )
            elif frozen_dtype is not None:
                self._cast_module_mixed_precision(
                    name,
                    component,
                    train_dtype,
                    frozen_dtype,
                    force_uniform_dtype=frozen_dtype,
                )
            # else: preserve the fully-frozen component's loaded dtype

        if trainable_count > 0:
            logger.info(
                f"Set {trainable_count} trainable parameters to {train_dtype}; "
                f"frozen component policy -> {frozen_policy}"
            )
        for name in merged_names:
            self._log_component_precision_inventory(
                name,
                self.get_component(name),
                stage="initialize",
            )

    # ============================== LoRA Management ==============================
    def apply_lora(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]] = "transformer",
        overwrite: bool = False,
    ) -> Union[PeftModel, Dict[str, PeftModel]]:
        """
        Apply LoRA adapters to specified components with prefix-based module targeting.

        Args:
            target_modules: Module patterns with optional component prefix
                - 'to_q': Apply to all components in `components`
                - 'transformer.to_q': Apply only to transformer
                - 'transformer_2.to_v': Apply only to transformer_2
                - ['to_q', 'transformer.to_k']: Mixed specification
            components: Component(s) to apply LoRA
            overwrite: When applying LoRA to a component that already has LoRA adapters:
                If True, delete existing 'default' adapter and create new one.
                If False, skip components that already have LoRA adapters.
        """
        # Normalize components to list
        if isinstance(components, str):
            components = [components]

        # Parse with explicit target_modules
        component_modules = self._parse_target_modules(target_modules, components)
        # Apply LoRA to each component
        results = {}
        for comp in components:
            modules = component_modules.get(comp)

            # Handle special cases
            if modules == "default":
                modules = self.default_target_modules
            elif modules == "all":
                modules = "all"  # Keep as 'all' for PEFT
            elif not modules:
                logger.warning(f"No target modules for {comp}, skipping LoRA")
                continue

            lora_config = LoraConfig(
                r=self.model_args.lora_rank,
                lora_alpha=self.model_args.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=modules,
            )

            model_component = self.get_component(comp)

            if isinstance(model_component, PeftModel):
                # Already a PeftModel, check for existing adapter
                has_default = "default" in model_component.peft_config
                if has_default and not overwrite:
                    logger.info(
                        f"Component {comp} already has 'default' adapter. Skipping initialization but enabling gradients."
                    )
                    # We must unfreeze the lora parameters because `_freeze_components` might have frozen them!
                    for name, param in model_component.named_parameters():
                        if any(k in name for k in self.lora_keys):
                            param.requires_grad = True
                    results[comp] = model_component
                    continue

                if has_default and overwrite:
                    # Overwrite: delete existing adapter and reinitialize
                    logger.info(f"Overwriting existing 'default' adapter for {comp}")
                    model_component.delete_adapter("default")

                # Add `default` adapter to existing PeftModel
                model_component.add_adapter("default", lora_config)
            else:
                # Not a PeftModel, initialize directly
                lora_config = LoraConfig(
                    r=self.model_args.lora_rank,
                    lora_alpha=self.model_args.lora_alpha,
                    init_lora_weights="gaussian",
                    target_modules=modules,
                )
                model_component = get_peft_model(model_component, lora_config)
                # Set back to attribute
                self.set_component(comp, model_component)

            # Activate the adapter
            model_component.set_adapter("default")
            results[comp] = model_component

            logger.info(f"Applied LoRA to {comp} with modules: {modules}")

        if not results:
            logger.warning("No LoRA adapters were applied")
            return {}

        return next(iter(results.values())) if len(results) == 1 else results

    # ============================== Distributed Utils ==================================

    # ------------------------------ Dist Types -----------------------------------------
    @property
    def _distributed_type(self) -> DistributedType:
        """Get current distributed type."""
        return self.accelerator.distributed_type

    def _is_deepspeed(self) -> bool:
        """Check if DeepSpeed is enabled."""
        return self._distributed_type == DistributedType.DEEPSPEED

    def _is_fsdp(self) -> bool:
        """Check if FSDP (v1) is enabled."""
        return self._distributed_type == DistributedType.FSDP

    def _is_fsdp2(self) -> bool:
        """Check if FSDP2 is enabled."""
        return getattr(self.accelerator, "is_fsdp2", False)

    # ------------------------------ Shard Strategies ---------------------------------
    def _is_zero3(self) -> bool:
        """Check if DeepSpeed ZeRO Stage 3 (parameter sharding) is enabled."""
        if not self._is_deepspeed():
            return False
        ds_plugin = self.accelerator.state.deepspeed_plugin
        return ds_plugin is not None and ds_plugin.zero_stage == 3

    def _is_fsdp_param_sharded(self) -> bool:
        """Check if FSDP shards parameters across ranks (FULL_SHARD or HYBRID)."""
        if not self._is_fsdp():
            return False
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        if fsdp_plugin is None:
            return False
        from torch.distributed.fsdp import ShardingStrategy

        return fsdp_plugin.sharding_strategy in (
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.HYBRID_SHARD,
            ShardingStrategy._HYBRID_SHARD_ZERO2,
        )

    # ------------------------------ FSDP Views ----------------------------------------
    def _fsdp_root_and_member_prefix(self, model) -> Tuple[Any, str]:
        """Return the prepared FSDP root holding ``model`` and ``model``'s key prefix.

        FSDP bookkeeping lives on the root that ``prepare`` produced, and every collective
        state-dict call has to go through it. The bundle keeps its members in an
        ``nn.ModuleDict``, so a member's parameters appear under ``members.<name>.`` in the
        root's state dict and the caller strips that back off.

        Falls back to ``(model, "")`` when ``model`` is itself the prepared root or is not a
        bundle member, which is what a single-component adapter looks like.
        """
        target = self._unwrap(model)
        for prepared in getattr(self.accelerator, "_models", ()):
            inner = self.accelerator.unwrap_model(prepared)
            members = getattr(inner, "members", None)
            if members is None:
                continue
            if inner is target or prepared is model:
                return prepared, ""
            for name, member in members.items():
                if member is target or self._unwrap(member) is target:
                    return prepared, f"members.{name}."
        return model, ""

    def _fsdp_state_dict_type(self):
        """Get FSDP state_dict_type, returns None if not FSDP."""
        if not self._is_fsdp():
            return None
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        return fsdp_plugin.state_dict_type if fsdp_plugin else None

    def _is_fsdp_collective_state_dict(self) -> bool:
        """Check if FSDP state_dict_type requires collective operations."""
        from torch.distributed.fsdp import StateDictType

        state_dict_type = self._fsdp_state_dict_type()
        if state_dict_type is None:
            return False
        # LOCAL_STATE_DICT does not requires communication while others do
        return state_dict_type != StateDictType.LOCAL_STATE_DICT

    def _is_param_sharded(self) -> bool:
        """Check if parameters are sharded across ranks."""
        return self._is_zero3() or self._is_fsdp2() or self._is_fsdp_param_sharded()

    def _requires_collective_state_dict(self) -> bool:
        """
        Check if state_dict gathering requires all ranks to participate.

        This is True when:
        - DeepSpeed ZeRO-3 (parameters sharded)
        - FSDP2 (always uses collective ops)
        - FSDP with FULL/SHARDED_STATE_DICT (collective save)
        - FSDP with FULL_SHARD (parameters sharded, must gather)
        """
        if self._is_zero3():
            return True
        if self._is_fsdp2():
            return True
        if self._is_fsdp() and (
            self._is_fsdp_param_sharded() or self._is_fsdp_collective_state_dict()
        ):
            return True
        return False

    # ============================== Checkpoint Management ==============================

    # ------------------------------ State Dict ------------------------------------------

    def get_state_dict(
        self,
        model,
        unwrap=True,
        state_dict_keys: Optional[Iterable[str]] = None,
        ignore_frozen_params: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        **Modified from `Accelerator.get_state_dict`**
        Returns the state dictionary of a model sent through [`Accelerator.prepare`] potentially without full
        precision.

        Args:
            model (`torch.nn.Module`):
                A PyTorch model sent through [`Accelerator.prepare`]
            unwrap (`bool`, *optional*, defaults to `True`):
                Whether to return the original underlying state_dict of `model` or to return the wrapped state_dict
                (e.g. for DeepSpeed or FSDP models).
            state_dict_keys (`List[str]`, *optional*):
                If provided, only return the parameters with these keys in the state dict. This is useful for saving with FSDP
                when you only want to save the trainable parameters.
            ignore_frozen_params (`bool`, *optional*, defaults to `False`):
                For FSDP2 only. If `True`, frozen parameters (i.e., those with `requires_grad=False`) will be ignored when saving the state dict.

        Returns:
            `dict`: The state dictionary of the model potentially without full precision.
        ```
        """

        def is_param_match_key(name, keys, strict=True):
            if keys is None:
                return not strict  # strict: no keys → no match; non-strict: no keys → match all
            if strict:
                return name in keys
            return any(k in name for k in keys)

        state_dict_keys = set(state_dict_keys) if state_dict_keys is not None else None

        from accelerate.utils import compare_versions

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            zero3_sharding = self.accelerator.deepspeed_config["zero_optimization"]["stage"] == 3
            tp_sharding = (
                self.accelerator.deepspeed_config.get("tensor_parallel", {}).get("autotp_size", 0)
                > 1
            )
            if zero3_sharding or tp_sharding:
                if model.zero_gather_16bit_weights_on_model_save():
                    ver_min_required = "0.16.4"
                    if tp_sharding and not compare_versions("deepspeed", ">=", ver_min_required):
                        raise ImportError(
                            f"Deepspeed TP requires deepspeed>={ver_min_required}. Please update DeepSpeed via `pip install deepspeed -U`."
                        )
                    state_dict = (
                        model._consolidated_16bit_state_dict()
                        if tp_sharding
                        else model._zero3_consolidated_16bit_state_dict()
                    )
                else:
                    raise ValueError(
                        "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                        "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                        "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                        "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
                    )
            else:
                from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

                state_dict = clone_tensors_for_torch_save(self._unwrap(model).state_dict())
        elif self.accelerator.is_fsdp2:
            # FSDP2: gather the full (unsharded) params to rank0 via the DTensor-aware API.
            # NOTE: the previous `state_dict_keys` path toggled `requires_grad` at runtime to
            # sub-select params, but FSDP2's `ignore_frozen_params` is keyed off the trainability
            # captured at `fully_shard` time -- the runtime toggle is a no-op, so it yielded an
            # EMPTY adapter. Gather straight through (LoRA params are exactly the trainable subset,
            # so `ignore_frozen_params=True` returns them) and let the shared key-filter below
            # narrow to `state_dict_keys` when provided.
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )

            options = StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                cpu_offload=True,
                ignore_frozen_params=ignore_frozen_params,
            )
            state_dict = get_model_state_dict(model, options=options)
        elif self.accelerator.distributed_type == DistributedType.FSDP:
            # FSDP1 through the same DTensor-aware API as FSDP2. The older
            # `FSDP.state_dict_type(model, ...)` is not only deprecated, it mutates FSDP
            # bookkeeping on whichever instance it is handed: components live inside the
            # prepared bundle, so `model` here is a NON-root FSDP instance, and entering
            # that context left `_is_root` set on it. The next rollout forward then died
            # in the real root's `_root_pre_forward` with "Non-root FSDP instance's
            # `_is_root` should not have been set yet", one full epoch after the save.
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )

            # Gather through the FSDP ROOT, never through the component. Components sit
            # inside the prepared bundle, so a component is a non-root FSDP instance (or
            # holds them); handing one to any state-dict API lazy-initializes it as a root
            # and the next rollout forward dies in the real root's `_root_pre_forward` with
            # "Non-root FSDP instance's `_is_root` should not have been set yet" -- a full
            # epoch after the save that caused it. The same applied to the deprecated
            # `FSDP.state_dict_type(component, ...)` this replaced.
            #
            # `ignore_frozen_params` is likewise not forwarded: torch drops frozen entries
            # with `state_dict.pop(fqn)` and no default, and under FSDP1 the FQN it rebuilds
            # for a PEFT parameter wrapped in both `_fsdp_wrapped_module` and
            # `_checkpoint_wrapped_module` is absent from the gathered dict, raising KeyError
            # on a frozen base weight. The `state_dict_keys` filter below already narrows to
            # the adapter.
            root, prefix = self._fsdp_root_and_member_prefix(model)
            options = StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                cpu_offload=True,
            )
            state_dict = get_model_state_dict(root, options=options)
            if prefix:
                state_dict = {
                    key[len(prefix) :]: value
                    for key, value in state_dict.items()
                    if key.startswith(prefix)
                }
        else:
            if unwrap:
                model = self._unwrap(model)
            state_dict = model.state_dict()

        # Filter by keys.
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if is_param_match_key(k, state_dict_keys, strict=False)
        }

        return state_dict

    @classmethod
    def _filter_lora_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        adapter_name: str = "default",
    ) -> Dict[str, torch.Tensor]:
        """
        Filter state dict to only include LoRA parameters.

        Args:
            state_dict: Full model state dict
            adapter_name: Name of the LoRA adapter (default: "default")

        Returns:
            State dict containing only LoRA-related weights
        """
        return {k: v for k, v in state_dict.items() if any(lk in k for lk in cls.lora_keys)}

    # -------------------------------------------- Save ------------------------------------
    def _save_lora(
        self,
        model: torch.nn.Module,
        save_directory: str,
    ) -> None:
        """Save LoRA adapter with distributed training support."""
        unwrapped = self._unwrap(model)

        if not isinstance(unwrapped, PeftModel):
            logger.warning(f"Model is not a PeftModel, falling back to full save.")
            self._save_full_model(
                model,
                save_directory,
                safe_serialization=True,
            )
            return

        # With variants declared, several adapters live on one shared base and
        # save_pretrained would write every one of them, so the save is scoped to the
        # adapter of the variant the caller asked for.
        registry = getattr(self, "component_variant_registry", None)
        selected_adapters = None
        if registry is not None:
            adapter_name = registry.get_spec(registry.active_variant).adapter_name
            if adapter_name is None:
                raise ValueError(
                    f"expected LoRA variant {registry.active_variant!r} to declare an "
                    "adapter_name, received None"
                )
            selected_adapters = [adapter_name]

        # If not sharded save, use standard save_pretrained
        if self._requires_collective_state_dict():
            # Handle sharded save
            # Gather all params before saving
            state_dict = self.get_state_dict(
                model,
                unwrap=True,
                state_dict_keys=self.lora_keys,
                ignore_frozen_params=True,
            )
            if self.accelerator.is_main_process:
                unwrapped.save_pretrained(
                    save_directory,
                    state_dict=state_dict,
                    selected_adapters=selected_adapters,
                )
        else:
            if self.accelerator.is_main_process:
                unwrapped.save_pretrained(
                    save_directory,
                    selected_adapters=selected_adapters,
                )

        if self.accelerator.is_main_process and selected_adapters is not None:
            self._flatten_peft_adapter_subdirectory(save_directory, selected_adapters[0])

        self.accelerator.wait_for_everyone()

    @staticmethod
    def _flatten_peft_adapter_subdirectory(save_directory: str, adapter_name: str) -> None:
        """Lift a named adapter's files out of the subfolder PEFT nests them in.

        ``PeftModel.save_pretrained`` writes any adapter other than ``"default"`` into
        ``<save_directory>/<adapter_name>/``. The role already names the directory it
        was given, so the extra level would make one role's artifact a different shape
        from another's and stop ``PeftModel.from_pretrained`` from reading it directly.

        Args:
            save_directory: Directory the adapter was asked to write to.
            adapter_name: Adapter that was written.
        """
        nested = os.path.join(save_directory, adapter_name)
        if adapter_name == "default" or not os.path.isdir(nested):
            return
        for filename in os.listdir(nested):
            destination = os.path.join(save_directory, filename)
            if os.path.exists(destination):
                os.remove(destination)
            shutil.move(os.path.join(nested, filename), destination)
        os.rmdir(nested)

    def _save_full_model(
        self,
        model: torch.nn.Module,
        save_directory: str,
        max_shard_size: str = "10GB",
        safe_serialization: bool = True,
        dtype: Optional[Union[torch.dtype, str]] = None,
    ) -> None:
        """
        **Modified from `Accelerator.save_model`**
        Save full model weights with distributed training support.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        # Normalize dtype
        if isinstance(dtype, str):
            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }.get(dtype.lower(), torch.bfloat16)

        unwrapped = self._unwrap(model)

        # Check if casting is needed
        cast_needed = False
        if dtype is not None:
            # Try to get model dtype, falling back to parameter inspection
            model_dtype = getattr(unwrapped, "dtype", None)
            if model_dtype is None:
                try:
                    model_dtype = next(unwrapped.parameters()).dtype
                except StopIteration:
                    # Empty model, assume no cast needed
                    model_dtype = dtype

            if model_dtype != dtype:
                cast_needed = True

        # Check offload
        is_offloaded = any(has_offloaded_params(module) for module in unwrapped.modules())

        # No shard, no casting, no offload, save directyly
        if not self._requires_collective_state_dict() and not cast_needed and not is_offloaded:
            # Standard save
            if self.accelerator.is_main_process:
                unwrapped.save_pretrained(
                    save_directory,
                    max_shard_size=max_shard_size,
                    safe_serialization=safe_serialization,
                )
            self.accelerator.wait_for_everyone()
            return

        # Get the state_dict of the model
        if is_offloaded:
            state_dict = get_state_dict_offloaded_model(model)
        else:
            if any(param.device == torch.device("meta") for param in model.parameters()):
                raise RuntimeError(
                    "You can't save the model since some parameters are on the meta device."
                )
            state_dict = self.get_state_dict(model, unwrap=True, ignore_frozen_params=False)

        # Case: DeepSpeed zero3 gets gathered and `state_dict` is empty
        if state_dict is None:
            return

        # Dtype casting
        if dtype is not None:
            for k in state_dict.keys():
                state_dict[k] = state_dict[k].to(device="cpu", dtype=dtype)

        os.makedirs(save_directory, exist_ok=True)

        if safe_serialization:
            state_dict = clean_state_dict_for_safetensors(state_dict)

        weights_name = SAFE_DIFFUSION_WEIGHTS_NAME if safe_serialization else DIFFUSION_WEIGHTS_NAME
        filename_pattern = (
            SAFE_DIFFUSION_WEIGHTS_PATTERN_NAME
            if safe_serialization
            else DIFFUSION_WEIGHTS_PATTERN_NAME
        )

        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
        )

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in state_dict_split.filename_to_tensors.keys()
                and reg.fullmatch(filename_no_suffix) is not None
                and PartialState().is_main_process
            ):
                os.remove(full_filename)

        # Save the model
        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor] for tensor in tensors}
            self.accelerator.save(
                shard, os.path.join(save_directory, filename), safe_serialization=safe_serialization
            )

        # Save the config file
        if hasattr(unwrapped, "config") and unwrapped.config is not None:
            config_save_file = os.path.join(save_directory, CONFIG_NAME)
            if hasattr(unwrapped.config, "save_pretrained"):
                unwrapped.config.save_pretrained(save_directory)
            else:
                # Handle dict-like configs (e.g., FrozenDict from diffusers)
                with open(config_save_file, "w", encoding="utf-8") as f:
                    json.dump(dict(unwrapped.config), f, indent=2, sort_keys=True)

            if self.accelerator.is_main_process:
                logger.info(f"Model config saved in {config_save_file}")

        # Save index if sharded
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = (
                SAFE_DIFFUSION_WEIGHTS_INDEX_NAME
                if safe_serialization
                else DIFFUSION_WEIGHTS_INDEX_NAME
            )
            save_index_file = os.path.join(save_directory, save_index_file)
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            if self.accelerator.is_main_process:
                logger.info(
                    f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                    f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                    f"index located at {save_index_file}."
                )
        else:
            path_to_weights = os.path.join(save_directory, weights_name)
            if self.accelerator.is_main_process:
                logger.info(f"Model weights saved in {path_to_weights}")

    # ------------------------------------- Checkpoint layout -------------------------------------
    def _checkpoint_role_names(self) -> Tuple[str, ...]:
        """Return the variants a checkpoint carries, base first.

        Returns:
            Declared variant names, or just the base name when this adapter
            declares no variants at all.
        """
        registry = getattr(self, "component_variant_registry", None)
        if registry is None:
            return (DEFAULT_BASE_VARIANT,)
        return registry.variant_names

    def _checkpoint_base_role(self) -> str:
        """Return the variant whose weights are the checkpoint's primary artifact."""
        registry = getattr(self, "component_variant_registry", None)
        if registry is None:
            return DEFAULT_BASE_VARIANT
        return registry.base_variant

    def _checkpoint_entries_from_config(self) -> List[CheckpointEntry]:
        """Derive the entries this run would write, from the live configuration.

        The base role of each component keeps the layout every released checkpoint
        already uses -- root for a single target component, ``<component>/`` when
        there are several -- so an existing consumer keeps working. Extra roles
        nest under ``roles/`` beside their component.

        Returns:
            One entry per (component, role) pair that owns weights.
        """
        nests_by_component = len(self.model_args.target_components) > 1
        base_role = self._checkpoint_base_role()
        entries: List[CheckpointEntry] = []
        for component_name, target_modules in self.target_module_map.items():
            if not target_modules or not self.has_component(component_name):
                continue
            component_prefix = f"{component_name}/" if nests_by_component else ""
            for role in self._checkpoint_role_names():
                if role == base_role:
                    relative = component_name if nests_by_component else "."
                else:
                    relative = f"{component_prefix}{CHECKPOINT_ROLES_DIRNAME}/{role}"
                entries.append(CheckpointEntry(component_name, role, relative))
        return entries

    def _read_checkpoint_manifest(self, path: str) -> Optional[Dict[str, Any]]:
        """Return a checkpoint's manifest, or ``None`` when it predates manifests.

        Args:
            path: Checkpoint directory.

        Returns:
            Parsed manifest, or ``None`` when the file is absent.

        Raises:
            ValueError: If the manifest exists but cannot be used.
        """
        manifest_path = os.path.join(path, CHECKPOINT_MANIFEST_NAME)
        if not os.path.isfile(manifest_path):
            return None
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        version = manifest.get("format_version")
        if version != CHECKPOINT_MANIFEST_VERSION:
            raise ValueError(
                f"expected checkpoint manifest format_version "
                f"{CHECKPOINT_MANIFEST_VERSION} at {manifest_path!r}, received {version!r}"
            )
        return manifest

    def _checkpoint_entries(self, path: Optional[str] = None) -> List[CheckpointEntry]:
        """Return every (component, role, directory) a checkpoint holds.

        This is the single source of truth for checkpoint layout. ``save_checkpoint``
        uses it to decide where to write, and every load path uses it to decide where
        to read, so the two cannot drift apart.

        Args:
            path: Checkpoint to inspect. When it carries a manifest the manifest is
                authoritative. When omitted, entries come from the live configuration,
                which is what a save needs.

        Returns:
            Entries in write order, base role first.

        Raises:
            ValueError: If a manifest entry is malformed, or a manifest-free checkpoint
                is asked to supply roles it cannot possibly contain.
        """
        if path is None:
            return self._checkpoint_entries_from_config()

        manifest = self._read_checkpoint_manifest(path)
        if manifest is not None:
            entries = []
            for raw_entry in manifest.get("entries", []):
                missing_keys = {"component", "role", "path"} - set(raw_entry)
                if missing_keys:
                    raise ValueError(
                        f"checkpoint manifest entry in {path!r} is missing "
                        f"{sorted(missing_keys)}, received {raw_entry!r}"
                    )
                entries.append(
                    CheckpointEntry(raw_entry["component"], raw_entry["role"], raw_entry["path"])
                )
        else:
            # A checkpoint written before manifests existed carries the base role at the
            # legacy paths and nothing else.
            base_role = self._checkpoint_base_role()
            entries = [
                entry for entry in self._checkpoint_entries_from_config() if entry.role == base_role
            ]

        if getattr(self, "component_variant_registry", None) is None:
            # The trainer declares variants after the adapter is built, so a weight-only
            # resume runs before the roles exist. Loading a second role now would route
            # it to the one live adapter and overwrite the first, so keep the primary
            # artifact and let `restore_training_roles` place the rest once they exist.
            entries = self._primary_entries(entries)

        self._warn_about_roles_the_checkpoint_omits(path, entries)
        return entries

    def _primary_entries(self, entries: Sequence[CheckpointEntry]) -> List[CheckpointEntry]:
        """Return one entry per component: the one an export would ship.

        Args:
            entries: Every entry a checkpoint provides.

        Returns:
            The first entry seen for each component, in order.
        """
        primary: Dict[str, CheckpointEntry] = {}
        for entry in entries:
            primary.setdefault(entry.component, entry)
        return list(primary.values())

    def restore_training_roles(self, path: str) -> None:
        """Place a checkpoint's non-primary roles now that the variants exist.

        A weight-only resume runs while the adapter is being built, before the trainer
        declares the roles it trains, so only the primary artifact can be placed then.
        This finishes the job so a multi-role resume cannot restore the base variant
        while leaving non-base variants freshly initialized.

        Args:
            path: Checkpoint directory to restore from.
        """
        registry = getattr(self, "component_variant_registry", None)
        if registry is None:
            raise RuntimeError(
                "restore_training_roles() expected declared component variants, received none; "
                "call declare_component_variants() first"
            )
        path = self._resolve_checkpoint_path(path)
        entries = self._checkpoint_entries(path)
        deferred = [entry for entry in entries if entry not in self._primary_entries(entries)]
        if not deferred:
            return

        resume_type = self._detect_checkpoint_type(path)
        for entry in deferred:
            if not self.has_component(entry.component):
                continue
            with registry.use(entry.role):
                if resume_type == "lora":
                    self._load_lora_entry(entry, path)
                else:
                    self._load_full_model_entry(entry, path)
        self.accelerator.wait_for_everyone()

    def _warn_about_roles_the_checkpoint_omits(
        self, path: str, entries: Sequence[CheckpointEntry]
    ) -> None:
        """Say plainly which trainable roles this checkpoint cannot restore.

        Loading an export (base weights only) into a multi-role run is a legitimate
        way to initialize a generator, so this is not an error. It is silent damage
        only if nobody says it happened: the omitted roles keep their initial weights.

        Args:
            path: Checkpoint being loaded.
            entries: Entries the checkpoint actually provides.
        """
        if getattr(self, "component_variant_registry", None) is None:
            # Variants are declared by the trainer after the adapter is built, so before
            # that there is no role vocabulary to compare against and nothing to report.
            return
        omitted = [
            role for role in self._checkpoint_role_names() if role not in {e.role for e in entries}
        ]
        if not omitted or not self.accelerator.is_main_process:
            return
        logger.warning(
            f"checkpoint {path} carries roles {sorted({e.role for e in entries})} but this run "
            f"trains {list(self._checkpoint_role_names())}; roles {omitted} keep their initial "
            "weights. That is expected when initializing from an export, and wrong when resuming "
            "-- resume from a checkpoint saved with include_training_roles, or from a full "
            "training state (`save_model_only: false`)."
        )

    # Artifacts a save authors. A shard index is the dangerous one: it is trusted ahead of
    # a single-file save, so one left behind by a bigger model sends the loader looking for
    # shards that no longer exist.
    _CHECKPOINT_ARTIFACT_GLOBS: ClassVar[Tuple[str, ...]] = (
        "*.safetensors",
        "*.safetensors.index.json",
        "*.bin",
        "*.bin.index.json",
        CONFIG_NAME,
        LORA_ADAPTER_CONFIG_NAME,
        "README.md",
    )

    def _clear_stale_checkpoint_artifacts(self, directory: str) -> None:
        """Remove what a previous save left in this entry's directory.

        Re-running an experiment under the same ``run_name`` writes into a directory that
        already holds another model's files, and nothing overwrites what the new save does
        not happen to produce. The result loads the stale file in preference to the fresh
        one, or fails looking for shards that were never written.

        Only this directory's own artifacts are removed, never its subdirectories, so the
        roles and components nested inside survive to be written in their own turn.

        Args:
            directory: Entry directory about to be written.
        """
        for pattern in self._CHECKPOINT_ARTIFACT_GLOBS:
            for stale in glob.glob(os.path.join(directory, pattern)):
                if os.path.isfile(stale):
                    os.remove(stale)

    def _write_checkpoint_manifest(self, path: str, entries: Sequence[CheckpointEntry]) -> None:
        """Record what was written so a loader never has to guess.

        Args:
            path: Checkpoint directory.
            entries: Entries that were written.
        """
        base_role = self._checkpoint_base_role()
        primary = next((entry for entry in entries if entry.role == base_role), None)
        manifest = {
            "format_version": CHECKPOINT_MANIFEST_VERSION,
            "finetune_type": self.model_args.finetune_type,
            "base_role": base_role,
            "primary": (
                None
                if primary is None
                else {"component": primary.component, "path": primary.relative_path}
            ),
            "entries": [
                {"component": e.component, "role": e.role, "path": e.relative_path} for e in entries
            ],
        }
        with open(os.path.join(path, CHECKPOINT_MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")

    def save_checkpoint(
        self,
        save_directory: str,
        max_shard_size: str = "10GB",
        dtype: Union[torch.dtype, str] = torch.bfloat16,
        save_ema: bool = True,
        model_only: bool = True,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        include_training_roles: bool = False,
        **kwargs,
    ):
        """Save a checkpoint for the target components.

        Args:
            save_directory: Destination directory.
            max_shard_size: Shard size for a sharded full-weight save.
            dtype: Storage dtype for full weights.
            save_ema: Whether to install the EMA parameters while saving.
            model_only: Whether to write model weights rather than training state.
            safe_serialization: Whether to write safetensors.
            variant: Component variant to write. ``None`` writes the base components,
                which is what an export ships; naming one restricts the write to it.
            include_training_roles: Whether to also write training-only variants.
                Exports omit them; resumable checkpoints include them so all roles
                restore together.
            **kwargs: Forwarded to ``accelerator.save_state`` for a training state.

        Raises:
            ValueError: If ``variant`` names a role that owns no savable component.
        """
        # Normalize dtype
        if isinstance(dtype, str):
            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }.get(dtype.lower(), torch.bfloat16)

        # 1. Save the training state if not model_only
        if not model_only:
            if self.accelerator.is_main_process:
                logger.info(f"Saving training state (resume-ready) to {save_directory}...")

            # Variant metadata must be written before accelerate mutates optimizer
            # state, or a resume cannot tell which optimizer group owns which variant.
            multirole_checkpoint_state = getattr(self, "_multirole_checkpoint_state", None)
            if multirole_checkpoint_state is not None:
                multirole_checkpoint_state.prepare_save(save_directory)
            self.accelerator.save_state(
                save_directory, safe_serialization=safe_serialization, **kwargs
            )

            if self.accelerator.is_main_process:
                logger.info(f"Training state saved.")
            return

        # 2. Save the model only. The scope is the caller's: an export ships the base
        # components, while a checkpoint a run will resume from also carries the
        # training-only roles.
        save_context = self.use_ema_parameters if save_ema else nullcontext
        registry = getattr(self, "component_variant_registry", None)

        entries = self._checkpoint_entries()
        if variant is not None:
            entries = [entry for entry in entries if entry.role == variant]
            if not entries:
                raise ValueError(
                    f"expected variant {variant!r} to own savable components, received none; "
                    f"declared variants are {list(self._checkpoint_role_names())}"
                )
        elif not include_training_roles:
            base_role = self._checkpoint_base_role()
            entries = [entry for entry in entries if entry.role == base_role]

        with save_context():
            for entry in entries:
                role_context = nullcontext() if registry is None else registry.use(entry.role)
                with role_context:
                    # Peel the RoutedComponentProxy to the inner module so the save
                    # path operates on the real diffusers/PeftModel (the prepared root
                    # is the ModelBundle; FSDP/DeepSpeed gathering happens inside
                    # get_state_dict on this member's params).
                    component = self._unwrap(self._require_component(entry.component))
                    comp_path = entry.directory(save_directory)
                    os.makedirs(comp_path, exist_ok=True)
                    if self.accelerator.is_main_process:
                        self._clear_stale_checkpoint_artifacts(comp_path)

                    if self.model_args.finetune_type == "lora":
                        if self.accelerator.is_main_process:
                            logger.info(
                                f"Saving LoRA weights for {entry.component} "
                                f"role={entry.role} to {comp_path}"
                            )
                        self._save_lora(component, comp_path)
                    else:
                        if self.accelerator.is_main_process:
                            logger.info(
                                f"Saving full weights for {entry.component} "
                                f"role={entry.role} to {comp_path}"
                            )
                        self._save_full_model(
                            component,
                            comp_path,
                            max_shard_size=max_shard_size,
                            safe_serialization=safe_serialization,
                            dtype=dtype,
                        )

            # Sync after saving
            self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self._write_checkpoint_manifest(save_directory, entries)
            logger.info(f"Checkpoint saved successfully to {save_directory}")

    # -------------------------------------------- Load -------------------------------------------
    def _resolve_checkpoint_path(self, path: str) -> str:
        """
        Resolve `path` to a local directory, downloading from Hugging Face Hub when needed.

        Resolution order:
            1. If `path` starts with ``hf://``, strip the prefix and force HF download
               (lets users override a colliding local directory).
            2. Otherwise, if `path` exists locally, return it as-is.
            3. Otherwise, parse as ``owner/repo[/subfolder][@revision]`` and download
               via Hugging Face Hub.

        Multi-node-safe: all ranks call ``snapshot_download`` directly. Hugging
        Face Hub's per-blob ``WeakFileLock`` serializes concurrent calls within
        each filesystem domain (cross-node on POSIX-locking shared FS, per-node
        on non-shared FS), so exactly one rank per filesystem domain actually
        transfers bytes. Un-gated (rather than ``is_local_main_process`` plus a
        barrier) so a failed download raises uniformly on every affected rank
        instead of leaving siblings deadlocked at a barrier the failing rank
        never reaches. Residual hazard: a rare single-rank transient failure
        (e.g. one node's network blip) can produce asymmetric progress, in
        which case the surviving ranks will eventually trip the NCCL watchdog
        on the final barrier below.

        Args:
            path: Local filesystem path or HF spec (with or without ``hf://`` prefix).

        Returns:
            Absolute local directory path ready for the existing checkpoint loaders.

        Raises:
            FileNotFoundError: When the spec is neither a local path nor a reachable HF repo.
        """
        # Normalize leading ``~`` for local-path inputs; no-op for HF specs since
        # ``expanduser`` only acts on a leading ``~``.
        path = os.path.expanduser(path)
        force_hf = path.startswith(HF_PATH_PREFIX)

        # Local path wins unless an explicit ``hf://`` prefix forces remote.
        if not force_hf and os.path.exists(path):
            return path

        # ``parse_hf_checkpoint_path`` handles the ``hf://`` prefix internally.
        repo_id, subfolder, revision = parse_hf_checkpoint_path(path)

        try:
            local_path = download_hf_checkpoint(repo_id, subfolder, revision)
        except (RepositoryNotFoundError, HfHubHTTPError) as e:
            raise FileNotFoundError(
                f"Checkpoint {path!r} not found locally and could not be fetched "
                f"from Hugging Face Hub (repo={repo_id!r}, subfolder={subfolder!r}, "
                f"revision={revision!r}). For private repos, ensure HF_TOKEN is set "
                f"on ALL nodes."
            ) from e

        # Sync after download so downstream loaders enter the lockstep dispatch
        # together. On symmetric failure every rank raises above before this
        # barrier is reached, so no deadlock; the residual asymmetric-failure
        # case is documented in the docstring.
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_local_main_process:
            logger.info(
                f"[local rank 0 / global rank {self.accelerator.process_index}] "
                f"resolved checkpoint '{path}' -> {local_path}"
            )

        return local_path

    @staticmethod
    def load_sharded_checkpoint(checkpoint_dir: str, index_file: str) -> Dict[str, torch.Tensor]:
        """Load sharded safetensors checkpoint."""
        with open(index_file, "r") as f:
            index = json.load(f)

        state_dict = {}
        loaded_files = set()

        for param_name, filename in index["weight_map"].items():
            if filename not in loaded_files:
                shard_path = os.path.join(checkpoint_dir, filename)
                shard = load_file(shard_path)
                state_dict.update(shard)
                loaded_files.add(filename)

        return state_dict

    def _load_lora(self, path: str) -> None:
        """Load LoRA adapters for every (component, role) the checkpoint holds.

        Paths come from :meth:`_checkpoint_entries`, the same resolver
        ``save_checkpoint`` writes through, and each entry is loaded inside its own
        role context so a role's weights land on that role's variant instead of
        whichever adapter happens to be active.

        Args:
            path: Checkpoint directory.
        """
        registry = getattr(self, "component_variant_registry", None)
        for entry in self._checkpoint_entries(path):
            comp_name = entry.component
            if not self.has_component(comp_name):
                logger.warning(f"Component {comp_name} not found, skipping")
                continue

            role_context = nullcontext() if registry is None else registry.use(entry.role)
            with role_context:
                self._load_lora_entry(entry, path)

    def _load_lora_entry(self, entry: CheckpointEntry, path: str) -> None:
        """Load one checkpoint entry into the currently active variant.

        Args:
            entry: Entry describing which component and role to restore.
            path: Checkpoint directory the entry belongs to.
        """
        comp_name = entry.component
        component = self._require_component(comp_name)
        comp_path = entry.directory(path)

        unwrapped = self._unwrap(component)

        # Auto-detect checkpoint format
        adapter_config_path = os.path.join(comp_path, LORA_ADAPTER_CONFIG_NAME)
        has_config_file = os.path.exists(adapter_config_path)

        if has_config_file:
            # Standard PeftModel format
            if not isinstance(unwrapped, PeftModel):
                unwrapped = PeftModel.from_pretrained(unwrapped, comp_path, is_trainable=True)
                unwrapped.set_adapter("default")
                self.set_component(comp_name, unwrapped)
            else:
                unwrapped.load_adapter(comp_path, unwrapped.active_adapter)
        else:
            # No config file found, manual `state_dict` loading with key mapping
            # Detect `safetensors` or `bin` format with `safetensors` preferred
            safetensors_files = glob.glob(os.path.join(comp_path, "*.safetensors"))
            if safetensors_files:
                state_dict_path = sorted(safetensors_files)[0]
                state_dict = load_file(state_dict_path)
            else:
                bin_files = glob.glob(os.path.join(comp_path, "*.bin"))
                if bin_files:
                    state_dict_path = sorted(bin_files)[0]
                    state_dict = torch.load(state_dict_path, map_location="cpu")
                else:
                    logger.error(f"No checkpoint file (.safetensors or .bin) found at {comp_path}")
                    return

            if self.accelerator.is_main_process:
                logger.info(
                    f"Loaded LoRA `state_dict` from: {state_dict_path}. "
                    f"If this is not wanted, please make sure the directory contains only single checkpoint file. "
                )

            # Apply key mapping for legacy format
            state_dict = mapping_lora_state_dict(state_dict)

            # Infer LoRA configuration from state_dict
            lora_rank, lora_alpha = infer_lora_config(state_dict)
            lora_alpha = self.model_args.lora_alpha or lora_alpha  # Use model arg if given
            if self.model_args.target_modules in [None, "default"]:
                # If default, infer target modules
                target_modules = infer_target_modules(state_dict)
            else:
                target_modules = self.model_args.target_modules

            if self.accelerator.is_main_process:
                logger.info(
                    f"Inferred LoRA config for {comp_name}: "
                    f"rank={lora_rank}, alpha={lora_alpha}, target_modules={target_modules[:5]}..."
                )

            # Create PeftModel if not already
            if not isinstance(unwrapped, PeftModel):
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    init_lora_weights="gaussian",
                    target_modules=target_modules,
                )

                unwrapped = get_peft_model(unwrapped, lora_config)
                unwrapped.set_adapter("default")

            # Load mapped state_dict
            missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)

            # Filter missing keys to LoRA only
            missing = [k for k in missing if any(lk in k for lk in self.lora_keys)]

            if self.accelerator.is_main_process:
                if missing:
                    logger.warning(f"Missing keys: {missing[:5]}...")
                if unexpected:
                    logger.warning(f"Unexpected keys: {unexpected[:5]}...")

            self.set_component(comp_name, unwrapped)

        if self.accelerator.is_main_process:
            logger.info(f"LoRA adapter loaded for {comp_name} role={entry.role} from {comp_path}")

    def _load_full_model(self, path: str, strict: bool = True) -> None:
        """Load full model weights for every (component, role) the checkpoint holds.

        Args:
            path: Checkpoint directory.
            strict: Whether to enforce exact ``state_dict`` key matching.
        """
        registry = getattr(self, "component_variant_registry", None)
        for entry in self._checkpoint_entries(path):
            comp_name = entry.component
            if not self.has_component(comp_name):
                logger.warning(f"Component {comp_name} not found, skipping")
                continue

            role_context = nullcontext() if registry is None else registry.use(entry.role)
            with role_context:
                self._load_full_model_entry(entry, path, strict=strict)

    def _load_full_model_entry(
        self, entry: CheckpointEntry, path: str, strict: bool = True
    ) -> None:
        """Load one full-weight entry into the currently active variant.

        Args:
            entry: Entry describing which component and role to restore.
            path: Checkpoint directory the entry belongs to.
            strict: Whether to enforce exact ``state_dict`` key matching.
        """
        comp_name = entry.component
        component = self._require_component(comp_name)
        comp_path = entry.directory(path)

        unwrapped = self._unwrap(component)
        component_class = unwrapped.__class__

        # `from_pretrained` reads Diffusers-format checkpoints that the manual loader
        # below cannot; a checkpoint written by the manual saver has no model index, so
        # only that lookup is guarded. Its weights are copied into the live module
        # rather than replacing it: the module is a member of the prepared root and a
        # variant of the registry, and swapping the object would silently detach both,
        # leaving the run training a module nobody holds a reference to.
        state_dict = None
        try:
            state_dict = component_class.from_pretrained(comp_path).state_dict()
        except (OSError, ValueError) as e:
            if self.accelerator.is_main_process:
                logger.debug(f"from_pretrained failed for {comp_name}: {e}, trying manual load...")

        if state_dict is None:
            index_file = os.path.join(comp_path, SAFE_DIFFUSION_WEIGHTS_INDEX_NAME)
            weights_file = os.path.join(comp_path, SAFE_DIFFUSION_WEIGHTS_NAME)

            if os.path.exists(index_file):
                state_dict = self.load_sharded_checkpoint(comp_path, index_file)
            elif os.path.exists(weights_file):
                state_dict = load_file(weights_file)
            else:
                logger.error(f"No valid checkpoint found for {comp_name} at {comp_path}")
                return

        # Load state_dict
        missing, unexpected = unwrapped.load_state_dict(state_dict, strict=strict)

        if self.accelerator.is_main_process:
            if missing:
                logger.warning(f"Missing keys for {comp_name}: {missing[:5]}...")
            if unexpected:
                logger.warning(f"Unexpected keys for {comp_name}: {unexpected[:5]}...")
            logger.info(
                f"Full model weights loaded for {comp_name} role={entry.role} from {comp_path}"
            )

    def _load_training_state(self, path: str) -> None:
        """Load full training state for resuming training."""
        if self.accelerator.is_main_process:
            logger.info(f"Loading training state from {path}...")

        # Reject a role layout that disagrees with this run before accelerate
        # restores optimizer state onto the wrong groups.
        multirole_checkpoint_state = getattr(self, "_multirole_checkpoint_state", None)
        if multirole_checkpoint_state is not None:
            multirole_checkpoint_state.validate_load(path)
        self.accelerator.load_state(path)

        if self.accelerator.is_main_process:
            logger.info("Training state loaded successfully.")

    def _detect_checkpoint_type(self, path: str) -> Literal["lora", "full"]:
        """
        Auto-detect checkpoint format by inspecting directory contents.

        A manifest states the format outright. Without one, the entry directories
        are probed for LoRA adapter files (adapter_config.json), falling back to
        'full' when no LoRA signature is found.
        """
        manifest = self._read_checkpoint_manifest(path)
        if manifest is not None and manifest.get("finetune_type") in ("lora", "full"):
            finetune_type = cast(Literal["lora", "full"], manifest["finetune_type"])
            if self.accelerator.is_main_process:
                logger.info(f"Checkpoint manifest at {path} declares a {finetune_type} checkpoint")
            return finetune_type

        paths_to_check = [entry.directory(path) for entry in self._checkpoint_entries(path)]
        for check_path in paths_to_check:
            if os.path.exists(os.path.join(check_path, LORA_ADAPTER_CONFIG_NAME)):
                if self.accelerator.is_main_process:
                    logger.info(f"Auto-detected LoRA checkpoint at {check_path}")
                return "lora"

        if self.accelerator.is_main_process:
            logger.info(f"Auto-detected full model checkpoint at {path}")
        return "full"

    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
        resume_type: Optional[Literal["lora", "full", "state"]] = None,
    ) -> None:
        """
        Load checkpoint for target components.

        Args:
            path: Checkpoint directory path.
            strict: Whether to strictly enforce state_dict key matching (only for full model).
            resume_type: Type of checkpoint to load.
                - 'lora': Load LoRA adapters only
                - 'full': Load full model weights
                - 'state': Load full training state (model + optimizer + RNG)
                - None: Auto-detect based on checkpoint directory contents
        """
        path = self._resolve_checkpoint_path(path)

        # Auto-detect if not specified
        if resume_type is None:
            resume_type = self._detect_checkpoint_type(path)

        if resume_type == "state":
            self._load_training_state(path)
        elif resume_type == "lora":
            self._load_lora(path)
        elif resume_type == "full":
            self._load_full_model(path, strict=strict)
        else:
            raise ValueError(
                f"Invalid resume_type: {resume_type}. Available: ['lora', 'full', 'state']."
            )

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            logger.info(f"Checkpoint loaded successfully from {path} (type={resume_type})")

    def _merge_lora_if_needed(self) -> None:
        """
        Merge LoRA adapters into base model weights when transitioning from
        LoRA checkpoint to full fine-tuning.

        Ensures the model is a plain nn.Module (not PeftModel) before entering
        the full training pipeline. The LoRA weights are permanently fused into
        the base model via merge_and_unload().
        """
        for comp_name in self.model_args.target_components:
            component = self.get_component(comp_name)
            unwrapped = self._unwrap(component)

            if isinstance(unwrapped, PeftModel):
                merged = unwrapped.merge_and_unload()
                self.set_component(comp_name, merged)
                if hasattr(self.pipeline, comp_name):
                    setattr(self.pipeline, comp_name, merged)

                if self.accelerator.is_main_process:
                    logger.info(f"Merged LoRA adapter into base model for {comp_name}")

    # ============================== Freezing Components ==============================
    def _freeze_components(self):
        """Freeze each materialized physical root, then reopen logical targets."""
        seen_roots = set()
        for root_name in self.component_runtime.materialized_component_names:
            component = self.component_runtime.get_canonical_component(root_name)
            if not isinstance(component, nn.Module) or id(component) in seen_roots:
                continue
            seen_roots.add(id(component))
            component.requires_grad_(False)
            component.eval()

        # Selectively unfreeze target components
        for comp_name in self.model_args.target_components:
            if not self.has_component(comp_name):
                logger.warning(f"Component {comp_name} not found, skipping freeze")
                continue

            trainable_modules = self.target_module_map.get(comp_name)

            if self.model_args.finetune_type == "lora":
                trainable_modules = None

            self._freeze_component(comp_name, trainable_modules=trainable_modules)

            # Restore train mode for components that have trainable parameters
            if trainable_modules:
                component = self._require_component(comp_name)
                component.train()

    def _freeze_component(
        self, component_name: str, trainable_modules: Optional[Union[str, List[str]]] = None
    ):
        """Freeze a specific component with optional selective unfreezing."""
        component = self._require_component(component_name)

        if trainable_modules == "all":
            logger.info(f"Unfreezing ALL {component_name} parameters")
            component.requires_grad_(True)
            return

        if isinstance(trainable_modules, str):
            if trainable_modules == "default":
                trainable_modules = self.default_target_modules
            else:
                trainable_modules = [trainable_modules]

        # Freeze all first
        component.requires_grad_(False)

        if not trainable_modules:
            logger.info(f"Froze ALL {component_name} parameters")
            return

        # Selectively unfreeze
        trainable_count = 0
        for name, param in component.named_parameters():
            if any(target in name for target in trainable_modules):
                param.requires_grad = True
                trainable_count += 1

        if trainable_count == 0:
            logger.warning(f"No parameters in {component_name} matched: {trainable_modules}")
        else:
            logger.info(f"Unfroze {trainable_count} parameters in {component_name}")

    # ============================== Trainable Parameters ==============================
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable parameters from all target components."""
        params = []
        for comp_name in self.model_args.target_components:
            if self.has_component(comp_name):
                component = self._require_component(comp_name)
                params.extend(filter(lambda p: p.requires_grad, component.parameters()))
        return params

    def log_trainable_parameters(self):
        """Log trainable parameter statistics for all target components."""
        for comp_name in self.model_args.target_components:
            if not self.has_component(comp_name):
                continue

            component = self.get_component(comp_name)
            if component is None:
                continue
            total_params = 0
            trainable_params = 0
            total_size_bytes = 0
            trainable_size_bytes = 0

            for param in component.parameters():
                param_count = param.numel()
                param_size = param.element_size() * param_count

                total_params += param_count
                total_size_bytes += param_size

                if param.requires_grad:
                    trainable_params += param_count
                    trainable_size_bytes += param_size

            total_size_gb = total_size_bytes / (1024**3)
            trainable_size_gb = trainable_size_bytes / (1024**3)
            trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0

            # Under FSDP these are this rank's shard, and a shard can legitimately hold
            # none of a small adapter: FSDP splits a flattened unit by byte range, so with
            # LoRA the frozen base fills the early shards and the whole adapter can land on
            # one rank. Say so, or "Trainable: 0" on rank 0 reads as a broken run.
            sharded = self.accelerator.distributed_type == DistributedType.FSDP
            scope = " (this rank's shard)" if sharded else ""
            logger.info("=" * 70)
            logger.info(f"{comp_name.capitalize()} Trainable Parameters:{scope}")
            logger.info(f"  Total:      {total_params:>15,d} ({total_size_gb:>6.2f} GB)")
            logger.info(f"  Trainable:  {trainable_params:>15,d} ({trainable_size_gb:>6.2f} GB)")
            logger.info(f"  Percentage: {trainable_percentage:>14.2f}%")
            logger.info("=" * 70)

    # ============================== Device Management ==============================

    def _should_manage_device(self, name: str) -> bool:
        """
        Check if a component's device should be manually managed.
        Runtime overrides (prepared/proxied, LoRA, or checkpoint replacements)
        are never manually moved.
        """
        return not self.component_runtime.has_component_override(name)

    def _resolve_component_names(
        self, components: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        """
        Resolve component specifiers into concrete pipeline attribute names. `None` means all components.

        Handles group names ('text_encoders', 'transformers') by expanding them,
        and passes through concrete names ('text_encoder', 'vae', 'transformer_2') as-is.
        """
        return self.component_runtime.resolve_component_names(components)

    def on_load_components(
        self,
        components: Optional[Union[str, List[str]]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Load specified components to device, skipping prepared (accelerator-managed) ones.

        Args:
            components: Component name(s) or group names ('text_encoders', 'transformers').
                        None loads all components.
            device: Target device. Defaults to accelerator device.
        """
        names = self._resolve_component_names(components)
        materialized_before = set(self.component_runtime.materialized_component_names)
        self.component_runtime.materialize_components(components)
        materialized_after = set(self.component_runtime.materialized_component_names)
        for name in names:
            if name in materialized_after and name not in materialized_before:
                component = self.component_runtime.get_canonical_component(name)
                if isinstance(component, torch.nn.Module):
                    if name not in self.model_args.target_components:
                        component.requires_grad_(False)
                        component.eval()
                    self._apply_component_precision_policy(name, component)
        self.component_runtime.load_stage_components(components, device=device or self.device)

    def off_load_components(self, components: Optional[Union[str, List[str]]] = None):
        """
        Off-load specified components to CPU, skipping prepared (accelerator-managed) ones.

        Args:
            components: Component name(s) or group names ('text_encoders', 'transformers').
                        None off-loads all components.
        """
        self.component_runtime.unload_stage_components(components)

    def on_load(self, device: Optional[Union[torch.device, str]] = None):
        """Load all components to device."""
        self.on_load_components(components=None, device=device)

    def off_load(self):
        """Off-load all components to CPU."""
        self.off_load_components(components=None)

    # Keep convenience aliases for backward compat, all delegate to unified methods
    def on_load_text_encoders(self, device: Optional[Union[torch.device, str]] = None):
        self.on_load_components("text_encoders", device)

    def off_load_text_encoders(self):
        self.off_load_components("text_encoders")

    def on_load_vae(self, device: Optional[Union[torch.device, str]] = None):
        self.on_load_components("vae", device)

    def off_load_vae(self):
        self.off_load_components("vae")

    def on_load_transformers(self, device: Optional[Union[torch.device, str]] = None):
        self.on_load_components("transformers", device)

    def off_load_transformers(self):
        self.off_load_components("transformers")

    # ============================== Preprocessing ==============================
    def preprocess_func(
        self,
        prompt: Optional[List[str]] = None,
        images: Optional[List[Union[Image.Image, List[Image.Image]]]] = None,
        videos: Optional[List[Union[List[Image.Image], List[List[Image.Image]]]]] = None,
        audios: Optional[List[Union[torch.Tensor, List[torch.Tensor]]]] = None,
        **kwargs,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """
        Preprocess input prompt, image, video, and audio into model-compatible embeddings/tensors.
        Always process a batch of inputs.
        Args:
            prompt: List of text prompts. A batch of text inputs.
            images:
                - None: no image input.
                - List[Image.Image]: list of images (a batch of single images)
                - List[List[Image.Image]]: list of list of images (a batch of a list images, each image list can be empty)
            videos:
                - None: no video input.
                - List[Video]: list of videos (a batch of single videos)
                - List[List[Video]]: list of list of videos (a batch of a list videos, each video list can be empty)
            audios:
                - None: no audio input.
                - List[torch.Tensor]: list of audio waveforms (a batch of single audios)
                - List[List[torch.Tensor]]: list of list of audio waveforms (a batch of a list audios, each audio list can be empty)
            **kwargs: Additional keyword arguments for encoder methods.

        """
        results = {}

        for input, encoder_method in [
            (prompt, self.encode_prompt),
            (images, self.encode_image),
            (videos, self.encode_video),
            (audios, self.encode_audio),
        ]:
            if input is not None:
                res = encoder_method(input, **(filter_kwargs(encoder_method, **kwargs)))

                if res is None:
                    # No preprocess needed
                    continue

                if (
                    isinstance(res, dict)
                    and res
                    and all(isinstance(v, (list, torch.Tensor, np.ndarray)) for v in res.values())
                ):
                    results.update(res)
                else:
                    raise ValueError(
                        f"Encoder method {encoder_method.__name__} should return a non-empty dict and each key maps to a list or tensor, "
                        f"but got {type(res)} with values types {[type(v) for v in res.values()]}"
                    )

        return results

    def encode_prompt(
        self,
        prompt: List[str],
        **kwargs,
    ) -> Optional[Dict[str, Union[List[Any], torch.Tensor]]]:
        """Encode a batch of text prompts into model-compatible embeddings.

        Default implementation is a no-op (returns ``None``). Subclasses
        override this when the model needs text conditioning.
        ``preprocess_func`` skips integration when the return value is
        ``None``, so adapters that don't need text encoding can simply
        inherit this default.

        Args:
            prompt: Batch of text prompts produced by
                ``dataset.py._preprocess_batch``.
            **kwargs: Adapter-specific encoding kwargs.

        Returns:
            Mapping from output key to encoded tensor/list, or ``None`` when
            the adapter does not perform prompt encoding.
        """
        pass

    def encode_image(
        self,
        images: MultiImageBatch,
        **kwargs,
    ) -> Optional[Dict[str, Union[List[Any], torch.Tensor]]]:
        """Encode a batch of (multi-)image inputs into latent representations.

        Default implementation is a no-op (returns ``None``). Subclasses
        override this when the model uses image conditioning.
        ``preprocess_func`` skips integration when the return value is
        ``None``, so adapters that don't need image encoding can simply
        inherit this default.

        Args:
            images: ``MultiImageBatch`` produced by
                ``dataset.py._preprocess_batch`` — a ``List[ImageBatch]``
                (ragged) or a uniform-shape tensor/array. Each batch slot
                is itself a list of images (``[]`` for empty samples).
            **kwargs: Adapter-specific encoding kwargs.

        Returns:
            Mapping from output key to encoded tensor/list (e.g.,
            ``condition_images``), or ``None`` when the adapter does not
            perform image encoding.
        """
        pass

    def encode_video(
        self,
        videos: MultiVideoBatch,
        **kwargs,
    ) -> Optional[Dict[str, Union[List[Any], torch.Tensor]]]:
        """Encode a batch of (multi-)video inputs into latent representations.

        Default implementation is a no-op (returns ``None``). Subclasses
        override this when the model uses video conditioning.
        ``preprocess_func`` skips integration when the return value is
        ``None``, so adapters that don't need video encoding can simply
        inherit this default.

        Args:
            videos: ``MultiVideoBatch`` produced by
                ``dataset.py._preprocess_batch`` — a ``List[VideoBatch]``
                (ragged) or a uniform-shape tensor/array. Each batch slot
                is itself a list of videos (``[]`` for empty samples).
            **kwargs: Adapter-specific encoding kwargs.

        Returns:
            Mapping from output key to encoded tensor/list (e.g.,
            ``condition_videos``), or ``None`` when the adapter does not
            perform video encoding.
        """
        pass

    def encode_audio(
        self,
        audios: MultiAudioBatch,
        **kwargs,
    ) -> Optional[Dict[str, Union[List[Any], torch.Tensor]]]:
        """Encode a batch of (multi-)audio inputs into latent representations.

        Default implementation is a no-op (returns ``None``). Subclasses
        override this when the model uses audio conditioning.
        ``preprocess_func`` skips integration when the return value is
        ``None``, so adapters that don't need audio encoding can simply
        inherit this default.

        Args:
            audios: ``MultiAudioBatch`` produced by
                ``dataset.py._preprocess_batch`` — a ``List[AudioBatch]``
                (ragged) or a uniform-shape tensor/array. Each batch slot
                is itself a list of audio waveforms (``[]`` for empty
                samples).
            **kwargs: Adapter-specific encoding kwargs.

        Returns:
            Mapping from output key to encoded tensor/list (e.g.,
            ``condition_audios``), or ``None`` when the adapter does not
            perform audio encoding.
        """
        pass

    # ======================================= Postprocessing =======================================
    @abstractmethod
    def decode_latents(
        self,
        latents: torch.Tensor,
        **kwargs,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Decodes latent representations back into images/videos if applicable.
        """
        pass

    def empty_decoded_media(self, batch_size: int) -> Any:
        """Return model-shaped empty media when rollout decoding is disabled.

        Distillation needs trajectories and conditioning metadata but not decoded
        images/video/audio. The adapter owns the decode return structure, so trainers
        must not infer it from model type or inspect conditioning fields.

        Args:
            batch_size: Number of generated samples.

        Returns:
            Empty media matching this adapter's ``decode_latents`` return structure.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(
                f"expected positive int batch_size for empty decoded media, got {batch_size!r}"
            )
        empty_batch = [None] * batch_size
        if len(self.trajectory_component_order) == 1:
            return empty_batch
        return tuple(list(empty_batch) for _ in self.trajectory_component_order)

    # ======================================= Latent Geometry =======================================
    def resolve_latent_axes(self, latents: torch.Tensor) -> LatentAxes:
        """Resolve the :class:`LatentAxes` (axis roles) for ``latents``.

        Returns the adapter's static ``LATENT_AXES`` override when set, otherwise
        infers from the latent ndim (3=packed, 4=conv, 5=video). Resolution-invariant.

        Args:
            latents: A batched latent tensor.

        Returns:
            The :class:`LatentAxes` describing ``latents``.
        """
        if self.LATENT_AXES is not None:
            return self.LATENT_AXES
        return infer_latent_axes(latents.ndim)

    def resolve_component_latent_axes(self, component: str, latents: torch.Tensor) -> LatentAxes:
        """Resolve latent axes for a trajectory component.

        Args:
            component: Component identifier.
            latents: Batched component latent tensor.

        Returns:
            Latent axes resolved by the legacy single-latent adapter API.
        """
        return bridge.resolve_component_latent_axes(self, component, latents)

    def get_terminal_state(self, batch: StackedSampleBatch) -> LatentState:
        """Read each trajectory component's terminal stored state.

        Args:
            batch: Collated sample batch containing structured or legacy trajectory data.

        Returns:
            Terminal latent state keyed by component.
        """
        return bridge.get_terminal_state(self, batch)

    def get_replay_step(self, batch: StackedSampleBatch, step_index: int) -> ReplayStep:
        """Read one replay transition without coupling component schedules.

        Args:
            batch: Collated sample batch containing structured or legacy trajectory data.
            step_index: Global rollout transition index.

        Returns:
            Current/next states, component times, and optional stored log probability.
        """
        return bridge.get_replay_step(self, batch, step_index)

    def get_replay_callback(
        self, batch: StackedSampleBatch, step_index: int, field: str
    ) -> LatentState:
        """Read one stored rollout callback trajectory for every component.

        Args:
            batch: Collated sample batch containing structured or legacy callback data.
            step_index: Global rollout transition index.
            field: Callback field name, e.g. ``"velocity"`` or ``"next_latents_mean"``.

        Returns:
            Stored callback state keyed by component.
        """
        return bridge.get_replay_callback(self, batch, step_index, field)

    def reference_guidance_kwargs(self, guidance_scale: float) -> Dict[str, object]:
        """Map the canonical distillation reference guidance onto this adapter's forward.

        Adapters whose forward uses a model-specific guidance name may override this
        method without exposing that name to trainer code.

        Args:
            guidance_scale: Guidance strength for the frozen reference score.

        Returns:
            Forward keyword arguments that apply reference-only guidance.
        """
        return {"guidance_scale": guidance_scale}

    def get_state_active_numel(self, state: LatentState) -> Mapping[str, int]:
        """Count each component's active stochastic degrees of freedom.

        The default counts every non-batch element. Adapters whose components carry
        masked or conditioning positions may override this.

        Args:
            state: Batched latent state in ``trajectory_component_order``.

        Returns:
            Positive per-component element counts in component order.
        """
        return bridge.get_state_active_numel(self, state)

    def get_train_step_indices(self) -> torch.Tensor:
        """Return the primary scheduler's positional training step indices.

        Every scheduler-group member must declare the same positional indices;
        their numeric timestep values may differ.

        Returns:
            One-dimensional tensor of rollout positions to train on.
        """
        return bridge.get_train_step_indices(self)

    def get_state_active_numel_per_sample(
        self,
        state: LatentState,
    ) -> Mapping[str, torch.Tensor]:
        """Count each sample's active stochastic degrees of freedom by component.

        This validated wrapper supports variable geometry across a batch. Adapters
        with custom packing or reduction semantics override
        :meth:`_get_state_active_numel_per_sample`, not this method.

        Args:
            state: Batched latent state in ``trajectory_component_order``.

        Returns:
            One positive signed-integer ``(B,)`` tensor per component, on that
            component's device and in component order.
        """
        bridge.validate_state_active_numel_per_sample_input(self, state)
        active_numel = self._get_state_active_numel_per_sample(state)
        return bridge.validate_state_active_numel_per_sample(self, state, active_numel)

    def _get_state_active_numel_per_sample(
        self,
        state: LatentState,
    ) -> Mapping[str, torch.Tensor]:
        """Return adapter-owned per-sample active counts before validation.

        A custom packing adapter must override this hook together with its latent
        reduction hook so both describe exactly the same active elements.
        """
        return bridge.get_state_active_numel_per_sample(self, state)

    def replay_generator_boundary(
        self,
        batch: StackedSampleBatch,
        boundary_index: int,
        *,
        return_fields: Tuple[str, ...] = ("velocity", "next_latents"),
        rtol: float,
        atol: float,
        **forward_kwargs: Any,
    ) -> MultiModalStepOutput:
        """Recompute and validate the transition preceding a generated boundary.

        Args:
            batch: Collated batch owning trajectory states and conditioning.
            boundary_index: Generated state boundary to recompute; at least one.
            return_fields: Scheduler output fields requested from ``forward_state``.
            rtol: Explicit relative tolerance for stored-transition validation.
            atol: Explicit absolute tolerance for stored-transition validation.
            **forward_kwargs: Explicit conditioning not already owned by ``batch``.

        Returns:
            Recomputed one-step output after validating its next state.
        """
        return bridge.replay_generator_boundary(
            self,
            batch,
            boundary_index,
            return_fields=return_fields,
            rtol=rtol,
            atol=atol,
            forward_kwargs=forward_kwargs,
        )

    def build_training_component_times(
        self,
        primary_timesteps: torch.Tensor,
        *,
        batch: Optional[Mapping[str, Any]] = None,
    ) -> ComponentTimes:
        """Map one primary scheduler coordinate onto every component's times.

        Trainers sample a single shared coordinate per sample; this hook turns it
        into component timesteps and sigmas without consuming randomness, so
        heterogeneous components can run on their own schedules. The default maps
        the legacy single ``"latent"`` component with a zero next coordinate.

        Args:
            primary_timesteps: Primary scheduler coordinates of shape ``(B,)``.
            batch: Optional online or offline mapping supplying per-component geometry.

        Returns:
            Component times whose sigma follows the flow-matching schedule.
        """
        return bridge.build_training_component_times(self, primary_timesteps, batch=batch)

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> NoisedState:
        """Draw forward-process noise, then apply it to the clean state.

        This is the RNG-owning hook: it draws once per component in
        ``trajectory_component_order`` and delegates the noise application to
        :meth:`apply_forward_process_noise`. The default draws the legacy single
        ``"latent"`` tensor with diffusers ``randn_tensor``; heterogeneous adapters
        override only the ordered draw.

        Args:
            clean_state: Clean latent state containing exactly ``"latent"``.
            times: Component times including the current ``"latent"`` sigma.
            generator: Optional generator used for the single random draw.

        Returns:
            Noised state, target velocity, and sampled noise.
        """
        return bridge.add_forward_process_noise(
            self,
            clean_state,
            times,
            generator=generator,
        )

    def apply_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        noise: LatentState,
    ) -> NoisedState:
        """Apply already-drawn noise to every clean component.

        Consumes no randomness, so the same noise can be reused across preference
        arms, precomputed passes, and reference passes. Each component is
        interpolated with its own sigma as ``(1 - sigma) * x0 + sigma * noise``.

        Args:
            clean_state: Clean latent state in ``trajectory_component_order``.
            times: Component times including each component's current sigma.
            noise: Noise state matching the clean order, shapes, dtypes, devices.

        Returns:
            Noised state, target velocity, and the supplied noise state.
        """
        return bridge.apply_forward_process_noise(self, clean_state, times, noise)

    def project_velocity_to_clean_state(
        self,
        state: LatentState,
        times: ComponentTimes,
        velocity: LatentState,
    ) -> LatentState:
        """Project a noised state to ``x0`` under this adapter's flow convention.

        ``flow_velocity_direction="noise"`` is the standard
        ``velocity = noise - clean`` convention and uses ``x0 = xt - sigma * velocity``.
        ``"data"`` is the MiniMax H3 convention ``velocity = clean - noise`` and uses
        ``x0 = xt + sigma * velocity``.
        """
        return bridge.project_velocity_to_clean_state(self, state, times, velocity)

    def project_velocity_to_score_state(
        self,
        state: LatentState,
        times: ComponentTimes,
        velocity: LatentState,
    ) -> LatentState:
        """Project adapter-directed velocity through clean state to diffusion score.

        Distillation objectives are written against the score function, while every
        adapter predicts a velocity under its own direction convention. This
        non-overridable wrapper first applies that convention, then delegates the
        schedule-specific clean-to-score conversion to
        :meth:`_project_clean_to_score_state`.

        Args:
            state: Current noised state in ``trajectory_component_order``.
            times: Component times including each current sigma.
            velocity: Adapter-directed velocity prediction matching ``state``.

        Returns:
            Score state in component order with the input active masks preserved.
        """
        bridge.validate_score_projection_inputs(self, state, times, velocity)
        clean_state = self.project_velocity_to_clean_state(state, times, velocity)
        bridge.validate_score_projection_state(self, clean_state, field="clean_state")
        score_state = self._project_clean_to_score_state(state, times, clean_state)
        return bridge.validate_projected_score_state(self, state, score_state)

    def _project_clean_to_score_state(
        self,
        state: LatentState,
        times: ComponentTimes,
        clean_state: LatentState,
    ) -> LatentState:
        """Convert clean prediction to score under the single-latent flow schedule."""
        return bridge.project_clean_to_score_state(self, state, times, clean_state)

    def _project_flow_match_clean_to_score_state(
        self,
        state: LatentState,
        times: ComponentTimes,
        clean_state: LatentState,
    ) -> LatentState:
        """Convert clean prediction with each declared flow-match component schedule."""
        return bridge.project_flow_match_clean_to_score_state(self, state, times, clean_state)

    def forward_state(
        self,
        *,
        batch: StackedSampleBatch,
        state: LatentState,
        times: ComponentTimes,
        next_state: Optional[LatentState] = None,
        compute_log_prob: bool = False,
        return_fields: Tuple[str, ...] = (),
        noise_level: Optional[float] = None,
        **kwargs: Any,
    ) -> MultiModalStepOutput:
        """Run one state through the model, owning the forward-argument boundary.

        This wrapper resolves the arguments the model may see and then dispatches
        to :meth:`_forward_state`. Override the hook, not this method, so a
        component-specific implementation cannot bypass the shared boundary:

        - explicit kwargs naming a state-owned argument (``t``, ``latents``, ...)
          or a bridge-owned one (trajectory storage, trainer metadata) raise;
        - batch-level state-owned keys are dropped in favor of the bridge-owned
          state, times, return fields and noise level;
        - trajectory storage and trainer metadata (e.g. ``advantage``) are read
          from the batch by the bridge and never forwarded to the model.

        Args:
            batch: Collated sample batch supplying conditioning arguments.
            state: Current state in ``trajectory_component_order``.
            times: Current and next times in ``trajectory_component_order``.
            next_state: Optional stored next state in ``trajectory_component_order``.
            compute_log_prob: Whether to compute transition log probability.
            return_fields: Existing scheduler output fields requested from ``forward``.
            noise_level: Existing scheduler noise-level override.
            **kwargs: Explicit adapter/training arguments that do not own state.

        Returns:
            Multi-modal wrapper around the unchanged legacy scheduler output.
        """
        return self._forward_state(
            batch=batch,
            state=state,
            times=times,
            next_state=next_state,
            compute_log_prob=compute_log_prob,
            return_fields=return_fields,
            noise_level=noise_level,
            forward_kwargs=bridge.build_forward_state_kwargs(self, batch, kwargs),
        )

    def _forward_state(
        self,
        *,
        batch: StackedSampleBatch,
        state: LatentState,
        times: ComponentTimes,
        next_state: Optional[LatentState],
        compute_log_prob: bool,
        return_fields: Tuple[str, ...],
        noise_level: Optional[float],
        forward_kwargs: Mapping[str, Any],
    ) -> MultiModalStepOutput:
        """Pack one state for this adapter's ``forward`` and unpack the output.

        Default single-component behavior: require exactly ``"latent"``, map the
        state and times onto the legacy ``forward`` arguments, and wrap the
        scheduler output. Heterogeneous adapters override this hook to pack and
        unpack their own components; ``forward_kwargs`` is already stripped of
        state-owned, trajectory-storage and trainer-metadata keys, so an override
        may forward it as-is (after the usual signature filtering).

        Args:
            batch: Collated sample batch, for conditioning an override needs to
                derive per component; storage fields must not be forwarded.
            state: Current state in ``trajectory_component_order``.
            times: Current and next times in ``trajectory_component_order``.
            next_state: Optional stored next state in ``trajectory_component_order``.
            compute_log_prob: Whether to compute transition log probability.
            return_fields: Existing scheduler output fields requested from ``forward``.
            noise_level: Existing scheduler noise-level override.
            forward_kwargs: Model-conditioning arguments resolved by the wrapper.

        Returns:
            Multi-modal wrapper around this adapter's scheduler output.
        """
        return bridge.default_forward_state(
            self,
            batch=batch,
            state=state,
            times=times,
            next_state=next_state,
            compute_log_prob=compute_log_prob,
            return_fields=return_fields,
            noise_level=noise_level,
            forward_kwargs=forward_kwargs,
        )

    def reduce_component_latent_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        state: Optional[LatentState] = None,
    ) -> Mapping[str, torch.Tensor]:
        """Reduce each component separately to one scalar per sample.

        Validated wrapper around :meth:`_reduce_component_latent_values`. Override
        the hook, not this method, so an override cannot bypass input or output
        validation.

        Args:
            values: Raw per-element component tensors in ``trajectory_component_order``.
            state: Optional replay state supplying component masks to an override.

        Returns:
            One ``(B,)`` tensor per component, in ``trajectory_component_order``.
        """
        batch_size = bridge.validate_reduction_inputs(self, values, state)
        reduced = self._reduce_component_latent_values(values, state=state)
        return bridge.validate_reduced_component_values(self, reduced, batch_size)

    def _reduce_component_latent_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        state: Optional[LatentState] = None,
    ) -> Mapping[str, torch.Tensor]:
        """Per-component reduction hook; averages every non-batch element.

        Adapters whose components carry masked or conditioning positions override
        this to average only the active positions, which a global element sum
        cannot recover. ``state`` carries per-sample context such as a dynamic mask.

        Args:
            values: Raw per-element component tensors in ``trajectory_component_order``.
            state: Optional replay state supplying component masks.

        Returns:
            One ``(B,)`` tensor per component, in ``trajectory_component_order``.
        """
        return bridge.default_reduce_component_latent_values(self, values, state=state)

    def reduce_latent_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        active_numel: Optional[Mapping[str, int]] = None,
        state: Optional[LatentState] = None,
    ) -> torch.Tensor:
        """Reduce component values to one globally element-weighted scalar per sample.

        Validated wrapper around :meth:`_reduce_latent_values`. Override the hook,
        not this method, so an override cannot bypass input or output validation.

        Args:
            values: Component tensors in ``trajectory_component_order``.
            active_numel: Optional partial mapping of positive weights for
                already-reduced ``(B,)`` component scalars.
            state: Optional replay state supplying component masks to an override.

        Returns:
            One globally element-weighted scalar per batch item.
        """
        batch_size = bridge.validate_reduction_inputs(self, values, state)
        reduced = self._reduce_latent_values(values, active_numel=active_numel, state=state)
        return bridge.validate_reduced_latent_values(self, reduced, batch_size)

    def _reduce_latent_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        active_numel: Optional[Mapping[str, int]] = None,
        state: Optional[LatentState] = None,
    ) -> torch.Tensor:
        """Global reduction hook; weights every element of every component once.

        Prefer passing raw per-element tensors whenever only a global result is
        needed. ``active_numel`` is for values already reduced per component, which
        a global sum cannot recover. A single-component ``(B,)`` value is returned
        unchanged to preserve the legacy scalar scale and tensor identity. The
        default ignores ``state``; dynamic mask adapters override this to weight
        only the elements the state marks active.

        Args:
            values: Component tensors in ``trajectory_component_order``.
            active_numel: Optional partial mapping of positive weights for
                already-reduced ``(B,)`` component scalars.
            state: Optional replay state supplying component masks.

        Returns:
            One globally element-weighted scalar per batch item.
        """
        return bridge.default_reduce_latent_values(
            self,
            values,
            active_numel=active_numel,
            state=state,
        )

    def reduce_flow_matching_objective_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        state: Optional[LatentState] = None,
    ) -> torch.Tensor:
        """Reduce offline flow-matching errors to one scalar per sample.

        This objective-specific boundary is deliberately separate from the
        trajectory-wide element-weighted reducer used by online policy and
        distillation algorithms. Most adapters inherit the existing global
        reduction unchanged; multi-modal training recipes may override the
        protected hook without changing rollout likelihood semantics.

        Args:
            values: Per-element squared errors in component order.
            state: Noised state supplying active masks.

        Returns:
            One flow-matching objective value per batch sample.
        """
        batch_size = bridge.validate_reduction_inputs(self, values, state)
        reduced = self._reduce_flow_matching_objective_values(values, state=state)
        return bridge.validate_reduced_latent_values(self, reduced, batch_size)

    def _reduce_flow_matching_objective_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        state: Optional[LatentState] = None,
    ) -> torch.Tensor:
        """Use the existing globally element-weighted reduction by default."""
        return self.reduce_latent_values(values, state=state)

    # ======================================= Sampling & Training =======================================
    @abstractmethod
    def forward(
        self,
        *args,
        **kwargs,
    ) -> SDESchedulerOutput:
        """
        Calculates the log-probability of the action (image/latent) given inputs.
        """
        pass

    @abstractmethod
    def inference(
        self,
        *args,
        **kwargs,
    ) -> List[BaseSample]:
        """
        Execute the generation process (Integration/Sampling).
        Returns a list of BaseSample instances.
        """
        pass
