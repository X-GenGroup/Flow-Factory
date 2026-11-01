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
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)

from ...contracts import (
    DECODED_AUDIO_REPRESENTATION,
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
from ..tensor_parallel import (
    ComposedParallelStyle,
    SequenceBroadcastOutput,
    SequenceBroadcastParameter,
    ShardedInputReplicatedWeight,
)
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
    representation=DECODED_AUDIO_REPRESENTATION,
)


class _MiniMaxH3WorkflowAdapter:
    """Workflow-invariant behavior shared by the MiniMax H3 adapters.

    Concrete adapters remain direct ``BaseAdapter`` subclasses that declare their workflow
    identity, I/O contract, canonical transformer name, and component lists. Ref2VA also
    opts into ordered references, sets FSDP2 memory options, and overrides ``apply_lora``
    to chunk LoRA projections under FSDP2.
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

    @classmethod
    def tensor_parallel_plan(cls) -> Dict[str, Any]:
        """Shard each block's attention; leave the modulation and the fused FFN replicated.

        Only the attention projections are sharded, which is 24% of a block's
        parameters. The other two are not omissions:

        * ``adaln_proj.linear`` (40% of a block) emits 18 modulation vectors of
          the full hidden width, applied to activations that stay replicated.
          Sharding its output would need those activations sharded on hidden
          too, which is sequence parallelism rather than this plan.
        * ``ff.net.0.proj`` (36%) fuses SwiGLU's ``[hidden, gate]`` into one
          weight. Column parallelism has to split *each* half so a rank's local
          output is ``[hidden_r, gate_r]`` and the downstream ``chunk(2)`` still
          separates them -- naive ``Shard(0)`` hands rank 0 the whole ``hidden``.
          Measured on real process groups, no placement describes that packed
          layout while keeping ``full_tensor()`` equal to the original weight
          (``Shard(0)`` reconstructs a permutation; ``_StridedShard`` asserts),
          so sharding it would corrupt every saved checkpoint. diffusers takes
          the permutation and refuses to save instead; this framework saves.

        The parameter saving is therefore modest -- about 21% at ``tp_size=8``.
        The reason to use it here is activation memory: attention activations
        are per-head and do shard, which is what long video sequences spend.

        The two ``token_refiner`` blocks stay replicated: they are 2 of 52 and
        each sharded module costs collectives per step.
        """
        return {
            "transformer_blocks.*.attn.to_q": ColwiseParallel(),
            "transformer_blocks.*.attn.to_k": ColwiseParallel(),
            "transformer_blocks.*.attn.to_v": ColwiseParallel(),
            "transformer_blocks.*.attn.to_out.0": RowwiseParallel(),
            # Per-head RMSNorm over attention_head_dim: replicated weight fed a
            # head-sharded input, so its gradient is a partial sum.
            "transformer_blocks.*.attn.norm_q": ShardedInputReplicatedWeight(),
            "transformer_blocks.*.attn.norm_k": ShardedInputReplicatedWeight(),
        }

    @classmethod
    def sequence_parallel_plan(cls) -> Dict[str, Any]:
        """Shard the token axis over everything the head split could not reach.

        The two plans are complementary here, which is unusual: tensor
        parallelism covers 23.9% of a block -- attention, split by head -- and
        every entry below covers the other 76.1%, split by token. Nothing in a
        block is left replicated on both axes afterwards.

        What the other 76.1% is matters, because it is also where the memory
        goes. ``adaln_proj.linear`` is 40.3% of a block and emits 18 full-width
        modulation vectors (six parameters for each of three modalities) that
        every token selects a row from; ``ff`` is 35.8% and its packed SwiGLU
        has no correct column placement, so tensor parallelism must leave it
        replicated. Both are applied *per token* to activations the head split
        never touched, and the SwiGLU intermediate is ``2 * ffn_dim`` wide --
        the largest activation this architecture holds.

        That gives H3's plan a different shape from Wan's: **only attention
        gathers.** A position-wise layer with a replicated weight is already
        correct on a token slice, so the feed-forward and the whole modulation
        chain run on ``1 / tp_size`` of the tokens with no collective at all.
        The module tensor parallelism could not shard is the one sequence
        parallelism shards best.

        The entries, and what each is for:

        * ``transformer_blocks.0`` scatters the packed sequence on the way in.
          Everything before it -- the three input projections and the token
          refiner -- stays at full length, and the scatter's backward
          all-gathers their gradient, so none of them needs an entry.
        * **every** block, not only the first, re-cuts ``adaln_indices``. This
          is the entry that is specific to H3 and the one a plan written from
          Wan's would be missing: ``adaln_indices`` is a ``(seq_len,)`` vector
          holding one AdaLN table row *per token*, handed to each block from the
          root, and each block applies it as
          ``scale.index_select(0, adaln_indices)``. A block given a token slice
          and the whole index vector broadcasts a full-length modulation against
          a sliced hidden state. ``Replicate() -> Shard(0)`` is a local chunk
          with no communication, and it cuts on the same rule ``Shard(1)`` uses
          for the hidden states, so the two slices always agree.
          ``norm_out`` takes the same treatment for
          ``timestep_indices``. diffusers' own context-parallel plan declares
          exactly these two inputs, which is independent confirmation that they
          are the per-token inputs of this architecture.
        * ``attn`` gathers back to the whole sequence, so self-attention over
          the packed document is unchanged and the ``(seq_len, rotary_dim)``
          rotary tables need no slicing; ``attn.to_out.0`` reduce-scatters
          instead of all-reducing. That is the same two collectives an
          all-reduce already was, so sequence parallelism adds no traffic.
        * ``norm1``, ``norm2`` and ``norm_out.norm`` are ``nn.RMSNorm`` with
          ``elementwise_affine`` left at its default, so all three own a weight
          applied once per token -- unlike Wan, whose ``norm1`` and ``norm3``
          have none. Each rank accumulates its share of their gradient, so each
          takes ``SequenceBroadcastParameter``, as do ``ff.net.0.proj``,
          ``ff.net.2``, ``adaln_proj.linear`` and ``norm_out.linear``. Replacing
          all seven with the plainly replicated style is silent in the loss, to
          the last bit, and leaves 19 of 52 gradients wrong -- the modulation
          projections at 28% of their true value, and ``audio_proj_out`` at
          exactly **zero** on rank 0, because every audio row of the packed
          sequence happens to land on the other rank. A weight that never
          receives a gradient is the shape this failure can take, not just a
          clean ``1 / tp_size``.
        * ``time_embedder`` produces the single ``temb`` every block's
          ``adaln_proj`` and ``norm_out`` read. Its value is broadcast over
          tokens and was never sharded, so no placement records that the
          gradient coming back is a partial sum --
          :class:`~flow_factory.models.tensor_parallel.SequenceBroadcastOutput`
          is what adds them. Dropping it leaves exactly the four
          ``time_embedder`` weights at 40% of their gradient on two ranks and
          moves the global gradient norm by 1.5%, which is the margin a
          tolerance on the norm waves through. H3 feeds that one embedding from
          50 blocks and both output heads.
        * the gather happens at ``proj_out`` and ``audio_proj_out`` rather than
          after the last block, for the reason Wan's does: they narrow 5376 to
          96 and to 32, so the restored tensors are 56x and 168x smaller than
          the hidden states, and ``norm_out``'s modulation stays sharded. It
          cannot happen any later -- the root selects each modality's rows with
          ``index_select(1, video_indices)`` over *global* positions.

        Measured on two 80 GB cards at the production hidden geometry, one
        block, bf16, peak allocation against packed length:

        ===========  =======  =======  =========
        tokens       tp=1     tp=2     tp=2 + SP
        ===========  =======  =======  =========
        16,408       8.5 GB   7.7 GB   5.4 GB
        65,560       27.6 GB  24.8 GB  15.4 GB
        131,096      53.1 GB  47.7 GB  28.9 GB
        196,632      OOM      70.5 GB  42.3 GB
        262,168      OOM      OOM      55.7 GB
        327,704      OOM      OOM      69.2 GB
        393,240      OOM      OOM      OOM
        ===========  =======  =======  =========

        768x1344 at 124 frames packs about 125,000 rows, so the 131,096 row is
        that run: tensor parallelism alone takes 10% off it and the overlay
        takes 46%. The rows below are what that buys -- the longest sequence
        that fits goes from 196,632 tokens to 327,704, and the peak grows at
        0.21 MB per token instead of 0.36 (tp=2) or 0.40 (tp=1). Nothing
        about that is specific to H3's size: the slope is the feed-forward
        intermediate and the modulation chain, both of which scale with the
        token count and neither of which a plan that shards weights can reach.

        Two preconditions, both of which fail loudly rather than silently:

        * **the packed length must divide by ``tp_size``.** ``PrepareModuleInput``
          lifts a plain local tensor with ``DTensor.from_local``, which has no
          global length to read and infers ``local * tp_size``. An odd length
          splits 33/32, the two ranks then infer 66 and 64, and the gather at the
          attention entry issues mismatched all-gathers and ends in
          "the size of tensor a (66) must match the size of tensor b (65)" where
          the rotary tables are applied. ``from_local`` does accept the true
          ``shape``; threading it from the scatter to the gathers would need a
          style ``tensor_parallel.py`` does not have yet.
        * **LoRA must not target a module inside the sequence-parallel region.**
          The adapter's defaults are the attention projections, which is exactly
          what ``derive_lora_plan`` is written for. A target like ``ff.net.2``
          instead takes the non-rowwise branch, which column-shards ``lora_B``
          and leaves its output half the width the base layer produced
          ("the size of tensor a (128) must match the size of tensor b (64)").
        """
        # `MiniMaxH3TransformerBlock.forward` is called positionally with
        # `(hidden_states, temb, adaln_indices, rotary_emb)` and
        # `MiniMaxH3Attention` with `(hidden_states, rotary_emb,
        # attention_mask)`. `PrepareModuleInput` wants one layout per positional
        # argument; `None` leaves that argument alone, which is what the
        # replicated conditioning and the rotary tuple need.
        slice_modulation_rows = PrepareModuleInput(
            input_layouts=(Shard(1), None, Replicate(), None),
            desired_input_layouts=(Shard(1), None, Shard(0), None),
            use_local_output=True,
        )
        return {
            "time_embedder": SequenceBroadcastOutput(),
            "transformer_blocks.*": slice_modulation_rows,
            "transformer_blocks.0": PrepareModuleInput(
                input_layouts=(Replicate(), None, Replicate(), None),
                desired_input_layouts=(Shard(1), None, Shard(0), None),
                use_local_output=True,
            ),
            "transformer_blocks.*.norm1": SequenceBroadcastParameter(),
            "transformer_blocks.*.norm2": SequenceBroadcastParameter(),
            "transformer_blocks.*.adaln_proj.linear": SequenceBroadcastParameter(),
            "transformer_blocks.*.attn": PrepareModuleInput(
                input_layouts=(Shard(1), None, None),
                desired_input_layouts=(Replicate(), None, None),
                use_local_output=True,
            ),
            "transformer_blocks.*.attn.to_out.0": RowwiseParallel(output_layouts=Shard(1)),
            # The packed SwiGLU stays replicated, as the tensor-parallel plan
            # explains, and is position-wise -- so it is correct as it stands on
            # a token slice and needs no gather, only the partial-sum gradient.
            "transformer_blocks.*.ff.net.0.proj": SequenceBroadcastParameter(),
            "transformer_blocks.*.ff.net.2": SequenceBroadcastParameter(),
            "norm_out": PrepareModuleInput(
                input_layouts=(Shard(1), None, Replicate()),
                desired_input_layouts=(Shard(1), None, Shard(0)),
                use_local_output=True,
            ),
            "norm_out.norm": SequenceBroadcastParameter(),
            "norm_out.linear": SequenceBroadcastParameter(),
            "proj_out": ComposedParallelStyle(
                SequenceBroadcastParameter(),
                PrepareModuleOutput(
                    output_layouts=Shard(1),
                    desired_output_layouts=Replicate(),
                    use_local_output=True,
                ),
            ),
            "audio_proj_out": ComposedParallelStyle(
                SequenceBroadcastParameter(),
                PrepareModuleOutput(
                    output_layouts=Shard(1),
                    desired_output_layouts=Replicate(),
                    use_local_output=True,
                ),
            ),
        }

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
