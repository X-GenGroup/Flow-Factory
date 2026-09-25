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

"""Define the three public MiniMax H3 workflow adapters."""

from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import torch

from ...contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaBinding,
    InputMediaOrder,
    InputMediaRule,
    MediaFormat,
    MediaType,
    NegativePromptPolicy,
    RateRequirement,
)
from ...samples import (
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    NoisedState,
    StackedSampleBatch,
)
from ...scheduler import MiniMaxH3SDEScheduler, SchedulerGroup
from ...utils.logger_utils import setup_logger
from ..abc import BaseAdapter
from ..checkpointing import CheckpointUnit
from ..output_state import DecodedMediaBatch, EncodedOutputState, OutputStateCodec
from ..pipeline_contracts import (
    IMAGE_FORMAT,
    VIDEO_FORMAT_OPTIONAL_FPS,
    audio_video_output_contract,
)
from ..runtime import ModularPipelineRuntime
from ._chunking import (
    H3_MAX_LORA_PROJECTION_TOKENS,
    H3_MAX_ROTARY_TOKENS,
    install_h3_in_forward_block_checkpointing,
    install_h3_lora_projection_chunking,
    install_h3_rotary_chunking,
)
from ._common import apply_forward_process_noise, draw_forward_process_noise
from ._condition import MiniMaxH3ConditionStatePreparer
from ._diffusers_cache import H3_DIFFUSERS_CACHE_POLICIES, prepare_h3_diffusers_cache
from ._output import MiniMaxH3AVOutputCodec, validate_h3_encoded_output_geometry
from .workflow import (
    build_h3_component_runtime,
    build_h3_replay_forward_kwargs,
    build_h3_scheduler,
    build_h3_scheduler_group,
    decode_h3_adapter_latents,
    forward_h3_adapter,
    infer_h3_workflow,
    init_h3_target_module_map,
    load_h3_workflow_pipeline,
    map_h3_training_component_times,
    preprocess_h3_workflow,
)

_H3_PREPROCESS_CACHE_FIELDS = frozenset({"height", "width", "num_frames"})
_H3_PREPROCESS_CACHE_VERSION = "minimax-h3-v3"
logger = setup_logger(__name__)
_H3_OPTIONAL_AUDIO_REFERENCE_FORMAT = MediaFormat(
    type=MediaType.AUDIO,
    fps=RateRequirement.NOT_APPLICABLE,
    sample_rate=RateRequirement.OPTIONAL,
)


class _MiniMaxH3WorkflowAdapter:
    """Share the workflow-invariant behavior of the MiniMax H3 adapters.

    Concrete adapters remain direct ``BaseAdapter`` subclasses and differ only in
    workflow identity, canonical transformer name, and component lists.
    """

    trajectory_component_order: ClassVar[Tuple[str, ...]] = ("video", "audio")
    flow_velocity_direction: ClassVar[Literal["data"]] = "data"
    preprocess_cache_fields: ClassVar[frozenset[str]] = _H3_PREPROCESS_CACHE_FIELDS
    preprocess_cache_version: ClassVar[str] = _H3_PREPROCESS_CACHE_VERSION
    supports_diffusers_cache: ClassVar[bool] = True
    supported_diffusers_cache_policies: ClassVar[frozenset[str]] = H3_DIFFUSERS_CACHE_POLICIES
    supports_fsdp2_cpu_efficient_loading: ClassVar[bool] = True
    requires_pairwise_policy_activation_offload: ClassVar[bool] = True
    # Official Diffusers recipe: load at BF16 and let each ModelMixin preserve
    # its declared FP32 islands (including both H3 autoencoders).
    component_load_dtype_defaults: ClassVar[torch.dtype] = torch.bfloat16

    def prepare_diffusers_cache(
        self,
        policy: str,
        component_name: str,
        transformer: torch.nn.Module,
    ) -> None:
        """Register H3 block metadata required by FirstBlockCache.

        Args:
            policy: User-facing diffusers cache policy identifier.
            component_name: Canonical transformer component name.
            transformer: Prepared transformer route that will receive the policy.

        Raises:
            ValueError: If the accelerator targets another workflow component.
        """
        if component_name != self.transformer_component_name:
            raise ValueError(
                f"MiniMax H3 workflow={self.workflow!r} diffusers cache expected component "
                f"{self.transformer_component_name!r}, received {component_name!r}"
            )
        prepare_h3_diffusers_cache(policy, transformer)

    def _gradient_checkpointing_units(
        self,
        component_name: str,
        component: torch.nn.Module,
    ) -> List[CheckpointUnit]:
        """Return H3's token-refiner then joint-transformer execution order."""
        expected_name = self.transformer_component_name
        if component_name != expected_name:
            raise ValueError(
                f"expected H3 checkpoint component {expected_name!r}, "
                f"received {component_name!r}"
            )
        token_refiner = getattr(component, "token_refiner", None)
        refiner_blocks = getattr(token_refiner, "refiner_blocks", None)
        transformer_blocks = getattr(component, "transformer_blocks", None)
        if not isinstance(refiner_blocks, torch.nn.ModuleList) or not isinstance(
            transformer_blocks,
            torch.nn.ModuleList,
        ):
            raise TypeError(
                "expected MiniMax H3 checkpoint stacks as ModuleList, received "
                f"token_refiner.refiner_blocks={type(refiner_blocks).__name__}, "
                f"transformer_blocks={type(transformer_blocks).__name__}"
            )
        return [
            *[
                (f"token_refiner.refiner_blocks.{index}", block)
                for index, block in enumerate(refiner_blocks)
            ],
            *[
                (f"transformer_blocks.{index}", block)
                for index, block in enumerate(transformer_blocks)
            ],
        ]

    def load_pipeline(self) -> Any:
        """Load this modular workflow from a local directory or Hugging Face repo."""
        return load_h3_workflow_pipeline(
            self.model_args.model_name_or_path,
            workflow=self.workflow,
        )

    def build_component_runtime(self) -> ModularPipelineRuntime:
        """Build the workflow-pruned modular runtime."""
        runtime = build_h3_component_runtime(self)
        if self.workflow != "ref2va" or not self._is_fsdp2():
            return runtime
        configured = install_h3_rotary_chunking(
            runtime.get_component(self.transformer_component_name)
        )
        logger.info(
            "Enabled token-chunked rotary embedding for MiniMax H3 Ref2VA FSDP2: "
            "configured=%d, max_tokens=%d",
            configured,
            H3_MAX_ROTARY_TOKENS,
        )
        return runtime

    def configure_fsdp2_in_forward_activation_checkpointing(
        self,
        model_root: torch.nn.Module,
    ) -> int:
        """Install one inner checkpoint on every materialized H3 block variant."""
        members = getattr(model_root, "members", None)
        if not isinstance(members, torch.nn.ModuleDict):
            raise TypeError(
                "MiniMax H3 FSDP2 checkpointing expected a ModelBundle ModuleDict, "
                f"received {type(members).__name__}"
            )
        configured = 0
        seen = set()
        for component in members.values():
            if id(component) in seen:
                continue
            seen.add(id(component))
            configured += install_h3_in_forward_block_checkpointing(component)
        return configured

    def load_scheduler(self) -> MiniMaxH3SDEScheduler:
        """Build the canonical shift-12 video scheduler."""
        return build_h3_scheduler(self.config.scheduler_args, shift=12.0)

    def build_scheduler_group(self) -> SchedulerGroup:
        """Build ordered fresh video/audio schedulers."""
        return build_h3_scheduler_group(self)

    def _init_target_module_map(self) -> Dict[str, Union[List[str], None]]:
        return init_h3_target_module_map(self)

    def preprocess_func(self, **kwargs: Any) -> Dict[str, Any]:
        return preprocess_h3_workflow(self, **kwargs)

    def build_condition_state_preparer(
        self,
    ) -> Optional[MiniMaxH3ConditionStatePreparer]:
        """Declare runtime prefix preparation only for conditioned workflows."""
        if self.workflow == "t2va":
            return None
        return MiniMaxH3ConditionStatePreparer(self)

    def build_output_state_codec(self) -> OutputStateCodec:
        """Declare the shared audiovisual target codec without loading components."""
        return MiniMaxH3AVOutputCodec(self)

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Require encoded AV rows to match this workflow's cached layout."""
        validate_h3_encoded_output_geometry(self, media_batch, condition, encoded)

    def _decode_output_state(
        self,
        encoded: EncodedOutputState,
        *,
        output_type: Literal["pil", "pt", "np"],
    ) -> Any:
        """Decode both H3 target components through the existing decoder."""
        geometry = encoded.decode_context.get("geometry")
        if not isinstance(geometry, Mapping):
            raise TypeError(
                "MiniMax H3 decode_context requires a geometry mapping, "
                f"received {type(geometry).__name__}: {geometry!r}"
            )
        return self.decode_latents(
            encoded.clean_state,
            geometry=geometry,
            output_type=output_type,
        )

    def build_training_component_times(
        self,
        primary_timesteps: torch.Tensor,
        *,
        batch: Optional[StackedSampleBatch] = None,
    ) -> ComponentTimes:
        return map_h3_training_component_times(self, primary_timesteps)

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> NoisedState:
        return draw_forward_process_noise(clean_state, times, generator=generator)

    def apply_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        noise: LatentState,
    ) -> NoisedState:
        return apply_forward_process_noise(clean_state, times, noise)

    def _reduce_flow_matching_objective_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        state: Optional[LatentState] = None,
    ) -> torch.Tensor:
        """Sum the official video and audio means for offline flow matching.

        H3's audiovisual objective gives each modality its own mean-squared-error
        term.  This objective-specific hook intentionally leaves the globally
        element-weighted reducer used by online likelihood objectives unchanged.
        """
        component_means = self.reduce_component_latent_values(values, state=state)
        return component_means["video"] + component_means["audio"]

    def decode_latents(self, latents: Any, **kwargs: Any) -> Any:
        return decode_h3_adapter_latents(self, latents, **kwargs)

    def empty_decoded_media(self, batch_size: int) -> Any:
        """Preserve H3's video/audio/sample-rate decode structure without decoding."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(
                f"MiniMax H3 expected positive int batch_size, received {batch_size!r}"
            )
        return ([None] * batch_size, [None] * batch_size, None)

    def inference(self, **kwargs: Any) -> List[Any]:
        return infer_h3_workflow(self, **kwargs)

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
        """Replay one stored transition through the public rollout entry point."""
        return self.forward(
            state=state,
            times=times,
            next_state=next_state,
            compute_log_prob=compute_log_prob,
            return_fields=return_fields,
            noise_level=noise_level,
            **build_h3_replay_forward_kwargs(
                forward_kwargs,
                state=state,
                workflow=self.workflow,
            ),
        )

    def forward(self, **kwargs: Any) -> MultiModalStepOutput:
        return forward_h3_adapter(self, **kwargs)


class MiniMaxH3T2VAAdapter(_MiniMaxH3WorkflowAdapter, BaseAdapter):
    """Load the workflow-pruned MiniMax H3 text-to-video-audio partition."""

    pipeline_io_contract = audio_video_output_contract(
        negative_prompt=NegativePromptPolicy.UNSUPPORTED,
        geometry_source=GeometrySource.CONFIGURED,
        batch_capability=BatchCapability.SINGLE_SAMPLE,
    )

    workflow: ClassVar[str] = "t2va"
    transformer_component_name: ClassVar[str] = "transformer"
    preprocessing_modules: ClassVar[List[str]] = ["text_encoder", "tokenizer", "processor"]
    inference_modules: ClassVar[List[str]] = [
        "transformer",
        "vae",
        "video_processor",
        "audio_vae",
    ]


class MiniMaxH3FL2VAAdapter(_MiniMaxH3WorkflowAdapter, BaseAdapter):
    """Load the workflow-pruned MiniMax H3 first/last-frame partition."""

    pipeline_io_contract = audio_video_output_contract(
        negative_prompt=NegativePromptPolicy.UNSUPPORTED,
        input_rules=(
            InputMediaRule(
                format=IMAGE_FORMAT,
                min_count=1,
                max_count=2,
                slots=("first_frame", "last_frame"),
            ),
        ),
        input_binding=InputMediaBinding.GROUPED_BY_TYPE,
        input_order=InputMediaOrder.WITHIN_TYPE,
        geometry_source=GeometrySource.CONFIGURED,
        batch_capability=BatchCapability.SINGLE_SAMPLE,
    )

    workflow: ClassVar[str] = "fl2va"
    transformer_component_name: ClassVar[str] = "transformer"
    preprocessing_modules: ClassVar[List[str]] = [
        "image_processor",
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
    ]
    inference_modules: ClassVar[List[str]] = [
        "transformer",
        "vae",
        "video_processor",
        "audio_vae",
    ]


class MiniMaxH3Ref2VAAdapter(_MiniMaxH3WorkflowAdapter, BaseAdapter):
    """Load the workflow-pruned MiniMax H3 omni-reference partition."""

    pipeline_io_contract = audio_video_output_contract(
        negative_prompt=NegativePromptPolicy.UNSUPPORTED,
        input_rules=(
            InputMediaRule(format=IMAGE_FORMAT, min_count=0, max_count=9),
            InputMediaRule(format=VIDEO_FORMAT_OPTIONAL_FPS, min_count=0, max_count=3),
            InputMediaRule(
                format=_H3_OPTIONAL_AUDIO_REFERENCE_FORMAT,
                min_count=0,
                max_count=3,
            ),
        ),
        input_binding=InputMediaBinding.ORDERED_REFERENCES,
        input_order=InputMediaOrder.GLOBAL,
        min_input_media_count=1,
        max_input_media_count=12,
        required_any_input_types=(MediaType.IMAGE, MediaType.VIDEO),
        geometry_source=GeometrySource.CONFIGURED,
        batch_capability=BatchCapability.SINGLE_SAMPLE,
    )

    workflow: ClassVar[str] = "ref2va"
    transformer_component_name: ClassVar[str] = "transformer_ref"
    fsdp2_use_default_stream_unshard: ClassVar[bool] = True
    fsdp2_use_in_forward_activation_checkpointing: ClassVar[bool] = True
    fsdp2_disable_backward_prefetch: ClassVar[bool] = True
    fsdp2_additional_wrap_module_names: ClassVar[Tuple[str, ...]] = (
        "_ChunkedFeedForward",
        "MiniMaxH3AdaLayerNormModulation",
        "MiniMaxH3Attention",
    )
    supports_ordered_references: ClassVar[bool] = True
    preprocessing_modules: ClassVar[List[str]] = [
        "image_processor",
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "audio_vae",
    ]
    inference_modules: ClassVar[List[str]] = [
        "transformer_ref",
        "vae",
        "video_processor",
        "audio_vae",
    ]

    def apply_lora(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]] = "transformer",
        overwrite: bool = False,
    ) -> Any:
        """Apply PEFT and bound Ref2VA projection memory under FSDP2.

        Args:
            target_modules: PEFT target-module patterns forwarded to ``BaseAdapter.apply_lora``.
            components: Canonical components that receive LoRA.
            overwrite: Whether to replace an existing default adapter.

        Returns:
            The PEFT model, per-component PEFT mapping, or empty mapping returned by the base
            method.

        Raises:
            TypeError: If Ref2VA FSDP2 projection chunking receives a nonstandard PEFT component,
                a non-``nn.Module`` base model, or incompatible projection structure.
            ValueError: If projection chunking conflicts with an earlier installation.
        """
        component_names = (components,) if isinstance(components, str) else tuple(components)
        result = super().apply_lora(
            target_modules=target_modules,
            components=components,
            overwrite=overwrite,
        )
        if (
            not result
            or not self._is_fsdp2()
            or self.transformer_component_name not in component_names
        ):
            return result

        component = self.get_component(self.transformer_component_name)
        get_base_model = getattr(component, "get_base_model", None)
        if not callable(get_base_model):
            raise TypeError(
                "MiniMax H3 Ref2VA FSDP2 LoRA chunking expected a PEFT component "
                f"with get_base_model(), received {type(component).__name__}"
            )
        transformer = get_base_model()
        if not isinstance(transformer, torch.nn.Module):
            raise TypeError(
                "MiniMax H3 Ref2VA FSDP2 LoRA chunking expected an nn.Module base, "
                f"received {type(transformer).__name__}"
            )
        configured = install_h3_lora_projection_chunking(transformer)
        logger.info(
            "Enabled token-chunked PEFT projections for MiniMax H3 Ref2VA FSDP2: "
            "configured=%d, max_tokens=%d",
            configured,
            H3_MAX_LORA_PROJECTION_TOKENS,
        )
        return result
