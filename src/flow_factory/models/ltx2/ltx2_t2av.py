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

# src/flow_factory/models/ltx2/ltx2_t2av.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)

from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline, rescale_noise_cfg

from ...contracts import BatchCapability, GeometrySource, NegativePromptPolicy
from ...hparams import *
from ...samples import (
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
    NoisedState,
    StackedSampleBatch,
    T2AVSample,
)
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SchedulerGroup,
    set_scheduler_timesteps,
)
from ...scheduler.flow_match_euler_discrete import calculate_shift
from ...utils.base import filter_kwargs
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import (
    TrajectoryIndicesType,
    create_callback_collector,
    create_trajectory_collector,
)
from ..abc import BaseAdapter
from ..output_state import DecodedMediaBatch, EncodedOutputState
from ..pipeline_contracts import audio_video_output_contract
from ..tensor_parallel import (
    ComposedParallelStyle,
    SequenceBroadcastOutput,
    SequenceBroadcastParameter,
    ShardedAcrossHeadRMSNorm,
)
from ._common import (
    LTX2_COMPONENT_ORDER,
    attach_ltx2_state_masks,
    build_ltx2_component_step_output,
    build_ltx2_full_component_schedule,
    build_ltx2_joint_forward_kwargs,
    build_ltx2_legacy_callback_view,
    build_ltx2_rollout_log_probs,
    build_ltx2_structured_trajectories,
    build_ltx2_training_component_times,
    combine_modality_log_prob,
    draw_ltx2_forward_process_noise,
    split_ltx2_callback_results,
    validate_ltx2_forward_state_inputs,
)
from ._output import (
    LTX2_OFFLINE_FORWARD_OVERRIDES,
    LTX2AVOutputCodec,
    decode_ltx2_output_state,
    reduce_ltx2_flow_matching_objective_values,
    validate_ltx2_encoded_output_geometry,
)

logger = setup_logger(__name__)

# Every attention stack a `LTX2VideoTransformerBlock` owns, in the order its
# forward runs them: video and audio self-attention, video/text and audio/text
# cross-attention, and the audio-to-video / video-to-audio cross-modal pair.
# All six are the same `LTX2Attention` and shard identically.
_ATTENTION_STACKS: Tuple[str, ...] = (
    "attn1",
    "audio_attn1",
    "attn2",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
)

# The block's pre-norms. `norm_elementwise_affine` is False in the LTX-2.0
# config, so these own no parameters; they are named in the sequence-parallel
# plan so a configuration that turns the affine weight on stays correct.
_BLOCK_NORMS: Tuple[str, ...] = (
    "norm1",
    "norm2",
    "norm3",
    "audio_norm1",
    "audio_norm2",
    "audio_norm3",
    "audio_to_video_norm",
    "video_to_audio_norm",
)


# ================================== Sample Dataclass ==================================


@dataclass
class LTX2Sample(T2AVSample):
    """Output class for LTX2 text-to-audio-video adapter."""

    _shared_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "height",
            "width",
            "num_frames",
            "frame_rate",
            "video_seq_len",
            "latent_index_map",
            "log_prob_index_map",
        }
    )

    # Generation shape (explicit fields, consistent with height/width on BaseSample)
    num_frames: Optional[int] = None
    frame_rate: Optional[float] = None

    # Split point for unified latents: latents[:, :video_seq_len] = video, rest = audio
    video_seq_len: Optional[int] = None

    # Connector outputs (video/audio text embeddings, cached from preprocessing)
    connector_prompt_embeds: Optional[torch.Tensor] = None  # (seq, D_video)
    connector_audio_prompt_embeds: Optional[torch.Tensor] = None  # (seq, D_audio)
    connector_attention_mask: Optional[torch.Tensor] = None  # (seq,)

    # Negative prompt connector outputs (for CFG during training forward)
    negative_connector_prompt_embeds: Optional[torch.Tensor] = None
    negative_connector_audio_prompt_embeds: Optional[torch.Tensor] = None
    negative_connector_attention_mask: Optional[torch.Tensor] = None


# ================================== Adapter ==================================


class LTX2_T2AV_Adapter(BaseAdapter):
    """Adapter for LTX2 text-to-audio-video generation with a joint video+audio policy.

    Video and audio are generated jointly through the transformer's cross-modal
    attention and stepped by two twin SDE schedulers, so each denoising transition
    produces a per-modality log_prob. The element-weighted combination of both
    log_probs is the joint policy log_prob that drives policy gradient training.
    """

    component_load_dtype_defaults = {"audio_vae": torch.float32}
    pipeline_io_contract = audio_video_output_contract(
        negative_prompt=NegativePromptPolicy.OPTIONAL,
        geometry_source=GeometrySource.CONFIGURED,
        batch_capability=BatchCapability.UNIFORM,
    )
    offline_training_forward_overrides = LTX2_OFFLINE_FORWARD_OVERRIDES

    supports_diffusers_cache = True
    trajectory_component_order: ClassVar[Tuple[str, ...]] = LTX2_COMPONENT_ORDER

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: LTX2Pipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler
        self.audio_scheduler: FlowMatchEulerDiscreteSDEScheduler

    # ============================== Pipeline Loading ==============================

    def load_pipeline(self) -> LTX2Pipeline:
        return self._load_diffusers_pipeline(
            LTX2Pipeline,
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False,  # Required for FSDP compatibility
        )

    def build_scheduler_group(self) -> SchedulerGroup:
        """Build the ordered video/audio scheduler group and its audio twin.

        Audio is sampled with the same SDE dynamics as video so that both
        modalities form a single joint policy with one per-step log_prob over
        both. A dedicated instance is still required because
        scheduler.step() mutates internal state (step_index), which would
        conflict if shared with the video scheduler. ``load_scheduler`` rebuilds
        an independent scheduler from the same pipeline scheduler + scheduler
        args used for ``self.scheduler`` (which ``BaseAdapter.__init__`` has
        already installed when it calls this hook).

        Returns:
            Scheduler group whose primary ``"video"`` component is the canonical
            pipeline scheduler.
        """
        self.audio_scheduler = self.load_scheduler()
        return SchedulerGroup(
            {"video": self.scheduler, "audio": self.audio_scheduler},
            primary_name="video",
        )

    # ============================== Module Properties ==============================

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for LTX2 transformer.

        Verified against LTX2VideoTransformerBlock.named_modules():
        28 Linear layers per block (6 attention groups x 4 projections + 2 FFN groups x 2 layers).
        """
        return [
            # Video self-attention
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            # Video cross-attention (text)
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            # Audio self-attention
            "audio_attn1.to_q",
            "audio_attn1.to_k",
            "audio_attn1.to_v",
            "audio_attn1.to_out.0",
            # Audio cross-attention (text)
            "audio_attn2.to_q",
            "audio_attn2.to_k",
            "audio_attn2.to_v",
            "audio_attn2.to_out.0",
            # Cross-modal attention (audio->video, video->audio)
            "audio_to_video_attn.to_q",
            "audio_to_video_attn.to_k",
            "audio_to_video_attn.to_v",
            "audio_to_video_attn.to_out.0",
            "video_to_audio_attn.to_q",
            "video_to_audio_attn.to_k",
            "video_to_audio_attn.to_v",
            "video_to_audio_attn.to_out.0",
            # Video FFN
            "ff.net.0.proj",
            "ff.net.2",
            # Audio FFN
            "audio_ff.net.0.proj",
            "audio_ff.net.2",
        ]

    @classmethod
    def tensor_parallel_plan(cls) -> Dict[str, Any]:
        """Shard all six attention stacks and both feed-forwards -- the whole block.

        Both of the things that used to keep attention replicated are expressible
        after all, and the plan went from 43.46% of a block to 99.98% of one --
        42.67% of the model to **98.16%**, measured on the class's own LTX-2.0
        defaults (18.88B as instantiated, 48 layers). Only the four bare
        modulation tables per block are left, which is what
        :meth:`sequence_parallel_plan` covers.

        **Across-head qk-norm.** ``LTX2Attention`` accepts only
        ``qk_norm="rms_norm_across_heads"`` -- it raises on anything else -- so
        ``norm_q``/``norm_k`` are ``RMSNorm(heads * head_dim)``: ``[4096]`` on the
        video stacks and ``[2048]`` on the audio ones, the whole inner dimension
        rather than ``[attention_head_dim]``. Column-parallel q/k split the
        *normalized* axis, which makes the forward wrong rather than merely the
        weight gradient, and it says so: ``Given normalized_shape=[4096] ... got
        input of size [1, 256, 2048]``.
        :class:`~flow_factory.models.tensor_parallel.ShardedAcrossHeadRMSNorm`
        shards the weight alongside the projection and all-reduces the mean
        square, one ``[batch, seq, 1]`` collective per norm. This is the same
        style Wan needs, for the same reason.

        **Full-width rotary tables.** ``LTX2AudioVideoAttnProcessor`` applies RoPE
        to the flat ``[B, S, inner_dim]`` projection, before heads are split, and
        the ``cos``/``sin`` it consumes are full-inner-dim *activations* rather
        than weights. With q/k column-sharded and the tables left whole, that is
        where the plan now fails: ``The size of tensor a (2048) must match the
        size of tensor b (4096) at non-singleton dimension 2``, measured at tp=2.
        The fix is a placement on the tables, not a new mechanism: the four
        ``LTX2AudioVideoRotaryPosEmbed`` modules return ``(cos, sin)`` and own no
        parameters, so ``PrepareModuleOutput`` can declare those outputs
        ``Replicate`` and redistribute them to ``Shard(-1)``, which is a local
        narrow with no collective. Rank *r* then holds exactly the channel range
        ``ColwiseParallel`` gave it in ``to_q``/``to_k``, so the rotation is
        applied to the same channels it was before.

        That slice is only correct because the channel layout is
        shard-aligned. ``repeat_interleave(2, dim=-1)`` pairs channels
        ``(2k, 2k+1)``, the ``inner_dim % (2 * num_pos_dims)`` padding is
        prepended in an even-sized block (4 channels for the 4096-wide video
        tables), and ``inner_dim / tp`` is even for every usable degree -- so a
        contiguous half never splits a rotation pair, and head boundaries land on
        shard boundaries (32 heads of 128 -> 16 of 128 at tp=2).

        **What the previous version of this docstring got wrong**, because
        someone will read both: it said the rotary tables were activations "which
        no ``ParallelStyle`` can slice". They are activations, but they are the
        *return value of a module*, and ``PrepareModuleOutput`` places return
        values. The blocker was also described as needing sequence parallelism to
        clear, and it does not: the mismatch is on the **channel** axis, not the
        token axis, so gathering the sequence back before attention does nothing
        for it. ``Shard(-1)`` on the tables is a tensor-parallel fix and holds
        with or without :meth:`sequence_parallel_plan`.

        ``ShardedAcrossHeadRMSNorm`` sets the parity floor for this plan, and it
        is worth knowing which term it is: it upcasts only the
        ``[batch, seq, 1]`` statistic to float32, so in a float64 audit the plan
        agrees with the unsharded model to 6.3e-7 rather than the 5.5e-15 the
        feed-forward-only entries reach on their own. That is a deliberate
        memory trade inside the shared style, not a sharding error, and in
        float32 -- every real run -- it is below the reduction-order noise floor.

        **This entry assumes ``rope_type="interleaved"``**, which is the class
        default and LTX-2.0's setting. Under ``rope_type="split"`` the same
        modules emit ``[B, heads, S, head_dim/2]`` tables and the correct
        placement is ``Shard(1)``, the head axis; ``Shard(-1)`` there is
        **silently wrong**, because ``apply_split_rotary_emb`` reads its head
        count off ``cos.shape[1]`` and reshapes the query to match whatever it
        finds. Nothing in this repository pins ``rope_type`` today, so a 2.5-era
        config needs this checked before the plan is used.

        ``ff``/``audio_ff`` are plain GELU MLPs, not packed SwiGLUs, so each
        column/row-shards cleanly; sharding ``ff`` also halves the
        ``4 * inner_dim``-wide intermediate, the largest activation per video
        token.

        At 98.16% the per-rank parameter cost over FSDP-only sharding is 1.018x
        at tp=2 and 1.129x at tp=8, against 1.573x and 5.013x for the old
        feed-forward-only plan -- which is the difference between a plan worth
        using and one that costs more memory than it saves. Measured peak at
        16,384 video tokens in bf16 over two layers: 14.70 GB at tp=1 against
        12.41 GB at tp=2.

        What stays replicated is the modulation, 1.84% of the model: the six
        ``LTX2AdaLayerNormSingle.linear`` projections (``time_embed``,
        ``audio_time_embed``, the two ``av_cross_attn_*_scale_shift`` and the two
        ``av_cross_attn_*_gate``) plus the four bare ``*scale_shift_table``
        parameters each block owns. They are applied to hidden states this plan
        leaves at full width, which is what :meth:`sequence_parallel_plan` is
        for.

        Sizes here come from ``LTX2VideoTransformer3DModel``'s own LTX-2.0
        defaults (32 video heads of 128, 32 audio heads of 64, 48 layers); no
        LTX-2 checkpoint was available locally, so the parity evidence behind
        this plan is random-weight only, unlike Wan's.

        ``gated_attn`` adds a per-head ``to_gate_logits`` projection that would
        have to column-shard with the heads. It is off in the LTX-2.0 config and
        is deliberately not named here, because an exact plan entry for a module
        the architecture does not have raises; if it is on, the head-count
        mismatch between the replicated gate and the localized ``attn.heads``
        fails loudly rather than silently.

        ``LTX2_I2AV_Adapter`` repeats this mapping rather than inheriting it: the
        adapter hierarchy is flat by design and the two share only the
        ``LTX2VideoTransformer3DModel`` the paths are written against.
        """
        # One style object per path would work too; these are stateless after
        # construction and each `_apply` hooks a different module.
        slice_rotary_tables = PrepareModuleOutput(
            output_layouts=(Replicate(), Replicate()),
            desired_output_layouts=(Shard(-1), Shard(-1)),
            use_local_output=True,
        )
        plan: Dict[str, Any] = {
            f"transformer_blocks.*.{stack}.{leaf}": style
            for stack in _ATTENTION_STACKS
            for leaf, style in (
                ("to_q", ColwiseParallel()),
                ("to_k", ColwiseParallel()),
                ("to_v", ColwiseParallel()),
                ("to_out.0", RowwiseParallel()),
                ("norm_q", ShardedAcrossHeadRMSNorm()),
                ("norm_k", ShardedAcrossHeadRMSNorm()),
            )
        }
        plan.update(
            {
                "transformer_blocks.*.ff.net.0.proj": ColwiseParallel(),
                "transformer_blocks.*.ff.net.2": RowwiseParallel(),
                "transformer_blocks.*.audio_ff.net.0.proj": ColwiseParallel(),
                "transformer_blocks.*.audio_ff.net.2": RowwiseParallel(),
                "rope": slice_rotary_tables,
                "audio_rope": slice_rotary_tables,
                "cross_attn_rope": slice_rotary_tables,
                "cross_attn_audio_rope": slice_rotary_tables,
            }
        )
        return plan

    @classmethod
    def sequence_parallel_plan(cls) -> Dict[str, Any]:
        """Shard both token axes from the input projections through the output ones.

        **Both streams, not just video.** Video dominates the memory -- 16,384
        tokens of 4096 channels for 121 frames at 1024x1024 against a few hundred
        audio tokens of 2048 -- so sharding audio buys almost nothing on its own.
        It is here because each ``LTX2VideoTransformerBlock`` owns *both* streams'
        modulation tables as bare parameters of one module:
        ``scale_shift_table`` and ``video_a2v_cross_attn_scale_shift_table`` are
        applied to video tokens, ``audio_scale_shift_table`` and
        ``audio_a2v_cross_attn_scale_shift_table`` to audio ones, and
        :class:`~flow_factory.models.tensor_parallel.SequenceBroadcastParameter`
        is per *module*, not per parameter. Sharding only the video axis would
        declare the audio tables' gradients ``Partial`` while every rank had
        already computed them whole, multiplying them by ``tp_size`` -- measured
        at exactly 2.000 on two ranks. Sharding both axes makes one style correct
        for all four.

        The entries are of six kinds.

        * **The boundaries are the input and output projections, not the first
          and last block.** ``proj_in``/``audio_proj_in`` scatter on the way in
          and ``proj_out``/``audio_proj_out`` gather on the way out. Wan scatters
          at ``blocks.0`` instead; LTX-2 cannot, because the root calls each
          block with keyword arguments on the eager path and with 26 positional
          ones under ``_gradient_checkpointing_func``, and ``PrepareModuleInput``
          validates one layout per positional argument -- a plan written for
          either convention raises under the other. The projections are called
          with one positional argument either way. Gathering after ``proj_out``
          also costs far less than gathering after the last block: the projection
          is ``inner_dim -> out_channels``, 4096 -> 128, so the gathered tensor
          is thirty-two times smaller.
        * **Every attention and feed-forward entry gathers the tokens back**, so
          attention sees the whole sequence and the rotary tables, the text
          stream and the attention masks need no slicing at all. Which argument
          carries the sharded stream differs per stack and is read off the
          block's own forward: ``attn1``/``audio_attn1`` are called entirely with
          keywords, ``attn2``/``audio_attn2``/``ff``/``audio_ff`` with one
          positional argument, and the two cross-modal stacks with one positional
          argument *and* a sharded ``encoder_hidden_states`` -- audio-to-video
          takes video queries and audio keys, video-to-audio the reverse, so both
          of their token-carrying inputs have to be gathered.
        * **The out-projections reduce-scatter instead of all-reducing.** That is
          the same traffic split differently, an all-reduce being a
          reduce-scatter followed by an all-gather, so sequence parallelism adds
          no communication. Each lands on the axis of the stream its output
          belongs to: ``audio_to_video_attn.to_out.0`` emits video tokens even
          though its keys are audio, and ``video_to_audio_attn.to_out.0`` emits
          audio tokens.
        * **Every bare modulation table takes
          ``SequenceBroadcastParameter``** -- the four on each block and the two
          on the root. They are read inside a ``forward`` with no module boundary
          to hook, and with the adapter's ``(B,)`` timestep each one broadcasts
          over every token, so each rank accumulates only its own share of the
          gradient. Omitting it on the root alone leaves ``scale_shift_table``
          at **0.5099** and ``audio_scale_shift_table`` at **0.5162** of their
          true gradient magnitude, with a **bit-identical loss** and a global
          gradient norm that moves **0.0083%** -- a tolerance on the norm sees
          nothing. The share is near 1/tp rather than exactly it because the
          per-token contributions are not equal; it is exactly 1/tp only in the
          *over*-reduction direction, which is what the audio tables show at
          2.0000 when only the video axis is sharded.
        * **The conditioning modules take
          ``SequenceBroadcastOutput``.** ``time_embed`` and ``audio_time_embed``
          each return ``(temb, embedded_timestep)`` and both outputs broadcast
          over tokens -- ``temb`` into every block, ``embedded_timestep`` into the
          root's output modulation -- so both indices are reduced. The four
          ``av_cross_attn_*`` heads only use their first output. Without these
          all six train at **0.4999-0.5025** of their true gradient while the
          loss stays bit-identical and the global gradient norm moves 1.29% --
          measured at tp=2 on the real widths.
        * **The block norms take ``SequenceBroadcastParameter`` as a no-op.**
          ``norm_elementwise_affine`` is ``False`` in the LTX-2.0 config, so these
          ``RMSNorm`` modules own no parameters and the style touches nothing;
          normalizing over the last axis is already correct on a token slice. The
          entries exist so a configuration that turns the affine weights on is
          still right, since that weight is applied per token exactly like a
          modulation table. ``norm_out``/``audio_norm_out`` at the root are
          ``LayerNorm(elementwise_affine=False)`` unconditionally and need no
          entry.

        What it buys, measured on two H800s in bf16 over two layers with the
        geometry of 121 frames at 1024x1024 -- 16,384 video tokens and 256 audio
        ones:

        ===============  ==========  ==================================
        configuration    peak        activation + gradient high-water
        ===============  ==========  ==================================
        ``tp=1``         14.70 GB    12.62 GB
        ``tp=2``         12.41 GB    11.05 GB
        ``tp=2`` + this  **9.98 GB**  **8.62 GB**
        ===============  ==========  ==================================

        So the overlay takes a further 19.6% off a peak tensor parallelism alone
        had already cut 15.6%, and 32.1% off ``tp=1``. The saving **grows with
        the sequence**, which is the opposite of what tensor parallelism alone
        does on this architecture: at 2,048 video tokens the same comparison is
        4.31 / 2.98 / 2.87 GB, a 3.9% overlay gain. That is the signature of
        activation sharding -- the residual stream, the norms and the modulated
        copies of it scale with the token count while the weights do not.

        With every entry in place the overlay is exact to the precision the
        tensor-parallel plan already sets: at ``tp=2`` in float64 the loss is
        bit-identical and every one of the 198 gradients agrees with the
        unsharded run to 6.3e-7 element-wise, which is **the same figure the
        tensor-parallel plan alone reaches** and is set by
        ``ShardedAcrossHeadRMSNorm`` holding its mean-square statistic in
        float32 by design. Removing the attention entries and re-running the
        feed-forward-only plan in float64 gives 5.5e-15, which is how that term
        is identified. In float32 the whole thing is below the reduction-order
        noise floor either way.

        **This plan requires the modulation to broadcast over tokens, which is
        what the adapter produces.** ``LTX2_T2AV_Adapter.forward`` passes
        ``timestep`` as ``(B,)``, so ``LTX2AdaLayerNormSingle`` emits
        ``[B, 1, *]`` and every block's ``get_mod_params`` broadcasts it.
        ``LTX2VideoTransformer3DModel`` also accepts a per-token
        ``timestep`` of shape ``(B, S)``, which makes ``temb`` carry the token
        axis -- and then a block multiplies a full-length modulation against a
        token slice. That combination raises ``The size of tensor a (128) must
        match the size of tensor b (256) at non-singleton dimension 1``, measured
        at tp=2; tensor parallelism alone is unaffected. Supporting it would mean
        scattering ``temb`` at each block's input, which cannot be written as the
        same static entry because a ``[B, 1, *]`` tensor has no token axis to
        shard.

        **The token count must be divisible by ``tp_size``, and violating it
        deadlocks rather than raising.** ``PrepareModuleInput`` lifts a plain
        tensor with ``DTensor.from_local``, which infers the global length as
        ``local * tp_size``; an uneven split makes the ranks disagree about it.
        Measured on CPU process groups at tp=2: 9 tokens leaves locals of 5 and
        4 and inferred global lengths of 10 and 8, and 125 leaves 63 / 62 and
        126 / 124 (``.scratch/tp_probe/cpu_uneven_token_axis.py``). On NCCL that
        disagreement means the two ranks post all-gathers of different sizes, so
        a tp=2 run at 125 video tokens **hangs at 100% GPU with no error
        message** -- it does not fail loudly, which is worth knowing before
        debugging one. Video tokens are ``num_frames * height * width`` and audio
        tokens ``audio_num_frames``, both before any padding, so this is a
        geometry constraint on both streams rather than something a pad could
        absorb.

        Two things this plan does not reach, both config-gated and off by
        default. ``use_keyframes_abs_pos_embedding`` adds a root parameter that is
        added to ``proj_in``'s output against a full-length ``video_keyframes_mask``,
        which a sharded output turns into a shape mismatch. ``cross_attn_mod``
        adds ``prompt_adaln``/``audio_prompt_adaln`` and two more block tables;
        the block tables are already covered by the ``transformer_blocks.*``
        entry, but the two conditioning heads would need their own
        ``SequenceBroadcastOutput`` -- which cannot simply be listed here,
        because an exact plan entry naming a module the architecture does not
        have raises when the plan is resolved.

        ``LTX2_I2AV_Adapter`` repeats this mapping for the same reason it repeats
        the tensor-parallel one.
        """
        gather_keyword_stream = PrepareModuleInput(
            input_kwarg_layouts={"hidden_states": Shard(1)},
            desired_input_kwarg_layouts={"hidden_states": Replicate()},
            use_local_output=True,
        )
        gather_positional_stream = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
            use_local_output=True,
        )
        gather_both_streams = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
            input_kwarg_layouts={"encoder_hidden_states": Shard(1)},
            desired_input_kwarg_layouts={"encoder_hidden_states": Replicate()},
            use_local_output=True,
        )
        scatter_tokens = PrepareModuleOutput(
            output_layouts=Replicate(),
            desired_output_layouts=Shard(1),
            use_local_output=True,
        )
        gather_tokens = ComposedParallelStyle(
            SequenceBroadcastParameter(),
            PrepareModuleOutput(
                output_layouts=Shard(1),
                desired_output_layouts=Replicate(),
                use_local_output=True,
            ),
        )
        plan: Dict[str, Any] = {
            # Boundaries.
            "proj_in": scatter_tokens,
            "audio_proj_in": scatter_tokens,
            "proj_out": gather_tokens,
            "audio_proj_out": gather_tokens,
            # Bare modulation tables: two on the root, four on every block.
            "": SequenceBroadcastParameter(),
            "transformer_blocks.*": SequenceBroadcastParameter(),
            # Conditioning values broadcast over every token.
            "time_embed": SequenceBroadcastOutput(output_indices=(0, 1)),
            "audio_time_embed": SequenceBroadcastOutput(output_indices=(0, 1)),
            "av_cross_attn_video_scale_shift": SequenceBroadcastOutput(output_indices=(0,)),
            "av_cross_attn_audio_scale_shift": SequenceBroadcastOutput(output_indices=(0,)),
            "av_cross_attn_video_a2v_gate": SequenceBroadcastOutput(output_indices=(0,)),
            "av_cross_attn_audio_v2a_gate": SequenceBroadcastOutput(output_indices=(0,)),
            # Gathers at each attention and feed-forward entry.
            "transformer_blocks.*.attn1": gather_keyword_stream,
            "transformer_blocks.*.audio_attn1": gather_keyword_stream,
            "transformer_blocks.*.attn2": gather_positional_stream,
            "transformer_blocks.*.audio_attn2": gather_positional_stream,
            "transformer_blocks.*.audio_to_video_attn": gather_both_streams,
            "transformer_blocks.*.video_to_audio_attn": gather_both_streams,
            "transformer_blocks.*.ff": gather_positional_stream,
            "transformer_blocks.*.audio_ff": gather_positional_stream,
            # Out-projections reduce-scatter onto the axis of the stream they emit.
            "transformer_blocks.*.attn1.to_out.0": RowwiseParallel(output_layouts=Shard(1)),
            "transformer_blocks.*.attn2.to_out.0": RowwiseParallel(output_layouts=Shard(1)),
            "transformer_blocks.*.audio_attn1.to_out.0": RowwiseParallel(output_layouts=Shard(1)),
            "transformer_blocks.*.audio_attn2.to_out.0": RowwiseParallel(output_layouts=Shard(1)),
            "transformer_blocks.*.audio_to_video_attn.to_out.0": RowwiseParallel(
                output_layouts=Shard(1)
            ),
            "transformer_blocks.*.video_to_audio_attn.to_out.0": RowwiseParallel(
                output_layouts=Shard(1)
            ),
            "transformer_blocks.*.ff.net.2": RowwiseParallel(output_layouts=Shard(1)),
            "transformer_blocks.*.audio_ff.net.2": RowwiseParallel(output_layouts=Shard(1)),
        }
        plan.update(
            {f"transformer_blocks.*.{norm}": SequenceBroadcastParameter() for norm in _BLOCK_NORMS}
        )
        return plan

    @property
    def preprocessing_modules(self) -> List[str]:
        """Components needed for offline preprocessing (text encoding + connectors)."""
        return ["text_encoders", "connectors"]

    @property
    def inference_modules(self) -> List[str]:
        """Components needed during inference and training forward."""
        return ["transformer", "vae", "audio_vae", "connectors", "vocoder"]

    # ============================== Output-State Encoding ==============================

    def build_output_state_codec(self) -> LTX2AVOutputCodec:
        """Declare deterministic on-the-fly encoding for paired AV targets."""
        return LTX2AVOutputCodec(self)

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Validate target state geometry against the configured LTX2 clocks."""
        validate_ltx2_encoded_output_geometry(
            self,
            media_batch,
            condition,
            encoded,
            conditioned=False,
        )

    def _decode_output_state(
        self,
        encoded: EncodedOutputState,
        *,
        output_type: Literal["pil", "pt", "np"],
    ) -> Any:
        """Decode both components through the joint AV ``decode_latents``."""
        return decode_ltx2_output_state(self, encoded, output_type=output_type)

    def _reduce_flow_matching_objective_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        state: Optional[LatentState] = None,
    ) -> torch.Tensor:
        """Sum official video/audio means without changing online reducers."""
        return reduce_ltx2_flow_matching_objective_values(
            self,
            values,
            state=state,
        )

    # ============================== Input Validation ==============================

    def _check_inputs(
        self,
        height: int,
        width: int,
        num_frames: int,
        prompt: Optional[Union[str, List[str]]] = None,
        connector_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        audio_guidance_scale: Optional[float] = None,
        stg_scale: float = 0.0,
        audio_stg_scale: Optional[float] = None,
        spatio_temporal_guidance_blocks: Optional[List[int]] = None,
    ) -> int:
        """Validate generation parameters and return VAE-compatible num_frames.

        Matches official pipeline check_inputs() validation logic.

        Returns:
            num_frames rounded down to the nearest VAE-temporal-compatible value.
        """
        vae_spatial = self.pipeline.vae_spatial_compression_ratio
        vae_temporal = self.pipeline.vae_temporal_compression_ratio

        if prompt is None and connector_prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `connector_prompt_embeds`. "
                "Cannot leave both undefined."
            )

        do_cfg = guidance_scale > 1.0 or (audio_guidance_scale or guidance_scale) > 1.0
        if (
            do_cfg
            and connector_prompt_embeds is not None
            and negative_connector_prompt_embeds is None
        ):
            raise ValueError(
                "guidance_scale > 1.0 requires negative_connector_prompt_embeds "
                "when using pre-encoded embeddings. Either provide negative "
                "embeddings or set guidance_scale <= 1.0."
            )

        if height % vae_spatial != 0 or width % vae_spatial != 0:
            raise ValueError(
                f"height ({height}) and width ({width}) must be divisible by "
                f"vae_spatial_compression_ratio ({vae_spatial})."
            )
        if (
            (stg_scale > 0.0) or ((audio_stg_scale or 0.0) > 0.0)
        ) and not spatio_temporal_guidance_blocks:
            raise ValueError(
                "Spatio-Temporal Guidance (STG) is enabled (stg_scale > 0) but no "
                "spatio_temporal_guidance_blocks specified. Recommended: [29] for LTX-2."
            )

        if (num_frames - 1) % vae_temporal != 0:
            num_frames = (num_frames - 1) // vae_temporal * vae_temporal + 1
            logger.warning(
                f"num_frames rounded to {num_frames} (must satisfy (num_frames - 1) % {vae_temporal} == 0)."
            )
        return max(num_frames, 1)

    # ============================== Text Encoding ==============================

    def _encode_text(
        self,
        text: List[str],
        max_sequence_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple:
        """Tokenize and encode text via Gemma3, returning (input_ids, embeds, mask).

        Inlines the logic from ``LTX2Pipeline._get_gemma_prompt_embeds`` so that
        ``input_ids`` (used as ``prompt_ids`` / ``negative_prompt_ids`` for reward
        grouping) are obtained from the same tokenization pass as the embeddings.
        """
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        tokenizer = self.pipeline.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text = [t.strip() for t in text]
        tok_out = tokenizer(
            text,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tok_out.input_ids.to(device)
        attention_mask = tok_out.attention_mask.to(device)

        enc_out = self.pipeline.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = torch.stack(enc_out.hidden_states, dim=-1)
        embeds = hidden.flatten(2, 3).to(dtype=dtype)

        return input_ids, embeds, attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 4.0,
        audio_guidance_scale: Optional[float] = None,
        max_sequence_length: int = 1024,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Encode text prompts into connector embeddings for video and audio streams.

        Tokenizes and encodes via Gemma3, then passes through LTX2 connectors to
        produce video/audio text embeddings.  Returns ``prompt_ids`` (and
        ``negative_prompt_ids`` when CFG is enabled) from the same tokenization
        pass -- no redundant tokenizer call.

        Returns:
            Dict with keys: prompt_ids, connector_prompt_embeds,
            connector_audio_prompt_embeds, connector_attention_mask, and their
            negative_* counterparts (including negative_prompt_ids) when CFG is
            enabled.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        batch_size = len(prompt)
        do_classifier_free_guidance = (
            guidance_scale > 1.0 or (audio_guidance_scale or guidance_scale) > 1.0
        )

        # 1. Encode positive prompt
        prompt_ids, prompt_embeds, prompt_attention_mask = self._encode_text(
            prompt,
            max_sequence_length,
            device,
            dtype,
        )

        # 2. Encode negative prompt if CFG
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            if len(negative_prompt) != batch_size:
                raise ValueError(
                    f"negative_prompt batch size {len(negative_prompt)} != "
                    f"prompt batch size {batch_size}"
                )
            negative_prompt_ids, negative_prompt_embeds, negative_prompt_attention_mask = (
                self._encode_text(negative_prompt, max_sequence_length, device, dtype)
            )
            combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            combined_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )
        else:
            negative_prompt_ids = None
            combined_embeds = prompt_embeds
            combined_mask = prompt_attention_mask

        # 3. Connectors: binary mask -> internal additive conversion
        connector_out, connector_audio_out, connector_mask = self.pipeline.connectors(
            combined_embeds,
            combined_mask,
        )

        # 4. Build results, splitting neg/pos if CFG
        if do_classifier_free_guidance:
            neg_conn, pos_conn = connector_out.chunk(2)
            neg_audio_conn, pos_audio_conn = connector_audio_out.chunk(2)
            neg_conn_mask, pos_conn_mask = connector_mask.chunk(2)
            results: Dict[str, Optional[torch.Tensor]] = {
                "prompt_ids": prompt_ids,
                "negative_prompt_ids": negative_prompt_ids,
                "connector_prompt_embeds": pos_conn,
                "connector_audio_prompt_embeds": pos_audio_conn,
                "connector_attention_mask": pos_conn_mask,
                "negative_connector_prompt_embeds": neg_conn,
                "negative_connector_audio_prompt_embeds": neg_audio_conn,
                "negative_connector_attention_mask": neg_conn_mask,
            }
        else:
            results = {
                "prompt_ids": prompt_ids,
                "connector_prompt_embeds": connector_out,
                "connector_audio_prompt_embeds": connector_audio_out,
                "connector_attention_mask": connector_mask,
            }

        return results

    # ============================== Image / Video Encoding ==============================

    def encode_image(self, images, **kwargs):
        """LTX2 T2AV does not use image conditioning."""
        return None

    def encode_video(self, videos, **kwargs):
        """LTX2 T2AV does not use video conditioning."""
        return None

    # ============================== Decoding ==============================

    def decode_latents(
        self,
        video_latents: torch.Tensor,
        audio_latents: Optional[torch.Tensor] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        decode_timestep: float = 0.0,
        decode_noise_scale: Optional[float] = None,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        """Decode packed latents to video frames and audio waveform.

        Video: unpack -> denormalize -> VAE decode.
        Audio: denormalize -> unpack -> audio_vae decode -> vocoder.
        Note the different operation order for video vs audio.
        """
        device = video_latents.device
        batch_size = video_latents.shape[0]
        vae = self.pipeline.vae
        patch_size = self.pipeline.transformer_spatial_patch_size  # 1
        patch_size_t = self.pipeline.transformer_temporal_patch_size  # 1
        vae_spatial = self.pipeline.vae_spatial_compression_ratio  # 32
        vae_temporal = self.pipeline.vae_temporal_compression_ratio  # 8

        # Video decode
        latent_h = height // vae_spatial
        latent_w = width // vae_spatial
        latent_f = (num_frames - 1) // vae_temporal + 1

        # 1. Unpack: (B, seq, C) -> (B, C, F, H, W)
        vid = self.pipeline._unpack_latents(
            video_latents, latent_f, latent_h, latent_w, patch_size, patch_size_t
        )
        # 2. Denormalize: latents * std / scaling_factor + mean
        vid = self.pipeline._denormalize_latents(
            vid, vae.latents_mean, vae.latents_std, vae.config.scaling_factor
        )
        # 3. Timestep conditioning + decode noise injection
        if not vae.config.timestep_conditioning:
            vae_timestep = None
        else:
            noise = torch.randn_like(vid)
            _dt = (
                [decode_timestep] * batch_size
                if not isinstance(decode_timestep, list)
                else decode_timestep
            )
            _dns = (
                _dt
                if decode_noise_scale is None
                else (
                    [decode_noise_scale] * batch_size
                    if not isinstance(decode_noise_scale, list)
                    else decode_noise_scale
                )
            )
            vae_timestep = torch.tensor(_dt, device=device, dtype=vid.dtype)
            _dns_t = torch.tensor(_dns, device=device, dtype=vid.dtype)[:, None, None, None, None]
            vid = (1 - _dns_t) * vid + _dns_t * noise
        # 4. VAE decode
        vid = vid.to(vae.dtype)
        video = vae.decode(vid, vae_timestep, return_dict=False)[0]
        video = self.pipeline.video_processor.postprocess_video(video, output_type=output_type)

        # Audio decode
        audio = None
        if audio_latents is not None:
            audio_vae = self.pipeline.audio_vae
            mel_compression = self.pipeline.audio_vae_mel_compression_ratio
            temporal_compression = self.pipeline.audio_vae_temporal_compression_ratio
            num_mel_bins = (
                audio_vae.config.mel_bins
                if getattr(self.pipeline, "audio_vae", None) is not None
                else 64
            )
            latent_mel_bins = num_mel_bins // mel_compression

            duration_s = num_frames / frame_rate
            sr = self.pipeline.audio_sampling_rate
            hop = self.pipeline.audio_hop_length
            audio_num_frames = round(duration_s * sr / hop / temporal_compression)

            # 1. Denormalize FIRST, then unpack (order differs from video!)
            aud = self.pipeline._denormalize_audio_latents(
                audio_latents, audio_vae.latents_mean, audio_vae.latents_std
            )
            # 2. Unpack: (B, seq, C) -> (B, C, T, mel_bins)
            aud = self.pipeline._unpack_audio_latents(
                aud, audio_num_frames, num_mel_bins=latent_mel_bins
            )
            # 3. Audio VAE decode -> mel spectrogram
            aud = aud.to(audio_vae.dtype)
            mel = audio_vae.decode(aud, return_dict=False)[0]
            # 4. Vocoder -> waveform
            audio = self.pipeline.vocoder(mel)

        return video, audio

    # ============================== x0 / Velocity Conversion ==============================

    def convert_velocity_to_x0(
        self,
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Convert velocity-space prediction to x0 (clean sample) prediction.

        In flow matching: x_t = (1 - sigma) * x_0 + sigma * noise, velocity v = noise - x_0.
        Therefore: x_0 = x_t - sigma * v.

        Args:
            sample: Current noisy sample x_t.
            velocity: Model velocity prediction v.
            sigma: Current noise level sigma = timestep / 1000, in [0, 1].
        """
        return sample - velocity * sigma

    def convert_x0_to_velocity(
        self,
        sample: torch.Tensor,
        x0: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Convert x0 prediction back to velocity space.

        v = (x_t - x_0) / sigma.

        Args:
            sample: Current noisy sample x_t.
            x0: Clean sample prediction.
            sigma: Current noise level sigma = timestep / 1000, in [0, 1].
        """
        return (sample - x0) / sigma

    # ============================== Forward ==============================

    def forward(
        self,
        t: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Split point: latents[:, :video_seq_len] = video, latents[:, video_seq_len:] = audio
        video_seq_len: Optional[int] = None,
        # Text embeddings (from connectors)
        connector_prompt_embeds: Optional[torch.Tensor] = None,
        connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
        connector_attention_mask: Optional[torch.Tensor] = None,
        negative_connector_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_attention_mask: Optional[torch.Tensor] = None,
        # Guidance scales (separate video/audio, matching official pipeline)
        guidance_scale: float = 4.0,
        audio_guidance_scale: Optional[float] = None,
        guidance_rescale: float = 0.0,
        audio_guidance_rescale: Optional[float] = None,
        # STG (Spatio-Temporal Guidance)
        stg_scale: float = 0.0,
        audio_stg_scale: Optional[float] = None,
        spatio_temporal_guidance_blocks: Optional[List[int]] = None,
        # Modality Isolation Guidance
        modality_scale: float = 1.0,
        audio_modality_scale: Optional[float] = None,
        # Generation shape (pixel-space, for computing latent dims + RoPE)
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        audio_num_frames: Optional[int] = None,
        # Positional coords (cached from inference loop)
        video_coords: Optional[torch.Tensor] = None,
        audio_coords: Optional[torch.Tensor] = None,
        # SDE control
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = ["next_latents", "log_prob", "velocity"],
        # LTX-2.3 compatibility
        use_cross_timestep: bool = False,
        # Offline forward: keep the raw velocity when no x0-space guidance runs
        preserve_raw_model_velocity: bool = False,
        # Component-return mode, owned by ``_forward_state``
        _return_components: bool = False,
        **kwargs,
    ) -> Union[FlowMatchEulerDiscreteSDESchedulerOutput, MultiModalStepOutput]:
        """Single denoising step with unified video+audio latents and x0-space multi-guidance.

        Accepts concatenated latents (B, video_seq + audio_seq, C) on the sequence dim.
        Internally splits into video/audio, runs the joint transformer, applies multi-level
        guidance (CFG + STG + Modality Isolation) in x0-space, then steps the video and audio
        SDE schedulers separately (the audio scheduler is a twin of the video one, so both
        modalities form a single joint policy). The outputs are concatenated back so the
        caller sees a single unified SDESchedulerOutput.

        Guidance pipeline (matching official diffusers pipeline_ltx2.py):
            1. Main transformer forward (CFG: [uncond, cond] batch)
            2. Convert velocity predictions to x0-space
            3. CFG delta in x0-space: (scale - 1) * (x0_cond - x0_uncond)
            4. Optional STG: extra forward with perturbed blocks -> stg_delta
            5. Optional Modality Isolation: extra forward with isolate_modalities=True
               -> modality_delta
            6. Combine: x0_guided = x0_cond + cfg_delta + stg_delta + modality_delta
            7. Guidance rescale in x0-space
            8. Convert back to velocity for scheduler step; with
               ``preserve_raw_model_velocity``, keep the raw prediction when no x0-space
               guidance transform ran

        The default return concatenates both modalities into one legacy
        ``SDESchedulerOutput``. ``_return_components`` is reserved for
        :meth:`_forward_state`, which needs the per-modality latents, statistics
        and log-probs captured before that concatenation.

        Note:
            The unified concatenation relies on video and audio packed latents sharing
            the same channel dimension C. For LTX2 this holds because:
              - Video: vae.latent_channels (128) * patch_t (1) * patch_h (1) * patch_w (1) = 128
              - Audio: audio_vae.latent_channels (8) * latent_mel_bins (16) = 128, with
                latent_mel_bins = mel_bins / mel_compression = 64/4
            This is an implicit model constraint (transformer.in_channels ==
            transformer.audio_in_channels == 128). If a future model variant changes the
            audio VAE config, verify that this equality still holds.
        """
        batch_size = latents.shape[0]
        device = latents.device

        # Normalize timestep to a 1-D (B,) tensor.
        # Inference passes a 0-D scalar; training passes (B,) with potentially
        # distinct values per sample (after K-repeat sampling). A singleton (1,)
        # tensor is also accepted and broadcast to (B,).
        if t.ndim == 0 or (t.ndim == 1 and t.shape[0] == 1):
            t = t.expand(batch_size)
        elif t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(
                f"expected `t` to be a scalar, a (1,) tensor, or a 1-D tensor of shape ({batch_size},), "
                f"got shape {tuple(t.shape)}"
            )

        # Compute sigma from timestep: sigma = t / 1000, broadcastable to latent dims
        sigma = (t / 1000).view(-1, 1, 1)  # (B, 1, 1)

        # Default audio guidance to video guidance
        audio_guidance_scale = audio_guidance_scale or guidance_scale
        audio_stg_scale = audio_stg_scale or stg_scale
        audio_modality_scale = audio_modality_scale or modality_scale
        audio_guidance_rescale = audio_guidance_rescale or guidance_rescale

        if (
            guidance_scale > 1.0 or audio_guidance_scale > 1.0
        ) and negative_connector_prompt_embeds is None:
            logger.warning(
                "Passed `guidance_scale` > 1.0, but no `negative_connector_prompt_embeds` provided. "
                "Classifier-free guidance will be disabled."
            )
        do_cfg = (
            guidance_scale > 1.0 or audio_guidance_scale > 1.0
        ) and negative_connector_prompt_embeds is not None
        do_stg = (
            stg_scale > 0.0 or audio_stg_scale > 0.0
        ) and spatio_temporal_guidance_blocks is not None
        do_modality_isolation = modality_scale > 1.0 or audio_modality_scale > 1.0

        # Compute latent dims
        vae_spatial = self.pipeline.vae_spatial_compression_ratio
        vae_temporal = self.pipeline.vae_temporal_compression_ratio
        latent_h = height // vae_spatial
        latent_w = width // vae_spatial
        latent_f = (num_frames - 1) // vae_temporal + 1
        if video_seq_len is None:
            video_seq_len = latent_f * latent_h * latent_w
        if audio_num_frames is None:
            duration_s = num_frames / frame_rate
            sr = self.pipeline.audio_sampling_rate
            hop = self.pipeline.audio_hop_length
            tc = self.pipeline.audio_vae_temporal_compression_ratio
            audio_num_frames = round(duration_s * sr / hop / tc)

        # Split unified latents into video / audio
        video_latents = latents[:, :video_seq_len]
        audio_latents = latents[:, video_seq_len:]
        if next_latents is not None:
            video_next = next_latents[:, :video_seq_len]
            audio_next = next_latents[:, video_seq_len:]
        else:
            video_next = audio_next = None

        # Prepare RoPE coords if not cached
        if video_coords is None:
            video_coords = self.pipeline.transformer.rope.prepare_video_coords(
                batch_size, latent_f, latent_h, latent_w, device, fps=frame_rate
            )
        if audio_coords is None:
            audio_coords = self.pipeline.transformer.audio_rope.prepare_audio_coords(
                batch_size, audio_num_frames, device
            )

        dtype = self.pipeline.transformer.dtype

        # Common transformer kwargs
        transformer_kwargs = dict(
            num_frames=latent_f,
            height=latent_h,
            width=latent_w,
            fps=frame_rate,
            audio_num_frames=audio_num_frames,
            use_cross_timestep=use_cross_timestep,
            attention_kwargs=None,
            return_dict=False,
        )

        # 1. Prepare CFG inputs and run main transformer forward
        if do_cfg:
            lat_in = torch.cat([video_latents, video_latents])
            aud_in = torch.cat([audio_latents, audio_latents])
            text_in = torch.cat([negative_connector_prompt_embeds, connector_prompt_embeds])
            audio_text_in = torch.cat(
                [negative_connector_audio_prompt_embeds, connector_audio_prompt_embeds]
            )
            mask_in = torch.cat([negative_connector_attention_mask, connector_attention_mask])
            vid_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            aud_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))
            # Duplicate timesteps to match torch.cat([lat, lat]) ordering: [t0..tB-1, t0..tB-1]
            ts = torch.cat([t, t])
        else:
            lat_in = video_latents
            aud_in = audio_latents
            text_in = connector_prompt_embeds
            audio_text_in = connector_audio_prompt_embeds
            mask_in = connector_attention_mask
            vid_coords, aud_coords = video_coords, audio_coords
            ts = t

        with self.pipeline.transformer.cache_context("cond_uncond"):
            video_pred, audio_pred = self.transformer(
                hidden_states=lat_in.to(dtype),
                audio_hidden_states=aud_in.to(dtype),
                encoder_hidden_states=text_in,
                audio_encoder_hidden_states=audio_text_in,
                timestep=ts,
                sigma=ts,  # LTX-2.3 compatibility
                encoder_attention_mask=mask_in,
                audio_encoder_attention_mask=mask_in,
                video_coords=vid_coords,
                audio_coords=aud_coords,
                **transformer_kwargs,
            )
        video_pred = video_pred.float()
        audio_pred = audio_pred.float()
        raw_video_pred = video_pred
        raw_audio_pred = audio_pred
        uses_x0_guidance = (
            do_cfg
            or do_stg
            or do_modality_isolation
            or guidance_rescale > 0
            or audio_guidance_rescale > 0
        )

        # 2. Convert to x0-space and compute guidance deltas
        if do_cfg:
            video_uncond, video_cond = video_pred.chunk(2)
            video_x0 = self.convert_velocity_to_x0(video_latents, video_cond, sigma)
            video_x0_uncond = self.convert_velocity_to_x0(video_latents, video_uncond, sigma)
            video_cfg_delta = (guidance_scale - 1) * (video_x0 - video_x0_uncond)

            audio_uncond, audio_cond = audio_pred.chunk(2)
            audio_x0 = self.convert_velocity_to_x0(audio_latents, audio_cond, sigma)
            audio_x0_uncond = self.convert_velocity_to_x0(audio_latents, audio_uncond, sigma)
            audio_cfg_delta = (audio_guidance_scale - 1) * (audio_x0 - audio_x0_uncond)
        else:
            video_x0 = self.convert_velocity_to_x0(video_latents, video_pred, sigma)
            audio_x0 = self.convert_velocity_to_x0(audio_latents, audio_pred, sigma)
            video_cfg_delta = audio_cfg_delta = 0

        pos_text = connector_prompt_embeds
        pos_audio_text = connector_audio_prompt_embeds
        pos_mask = connector_attention_mask
        pos_ts = t

        # 3. STG: extra transformer forward with perturbed blocks
        video_stg_delta = audio_stg_delta = 0
        if do_stg:
            with self.pipeline.transformer.cache_context("uncond_stg"):
                stg_video, stg_audio = self.transformer(
                    hidden_states=video_latents.to(dtype),
                    audio_hidden_states=audio_latents.to(dtype),
                    encoder_hidden_states=pos_text,
                    audio_encoder_hidden_states=pos_audio_text,
                    timestep=pos_ts,
                    sigma=pos_ts,
                    encoder_attention_mask=pos_mask,
                    audio_encoder_attention_mask=pos_mask,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    isolate_modalities=False,
                    spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
                    perturbation_mask=None,
                    **transformer_kwargs,
                )
            stg_video_x0 = self.convert_velocity_to_x0(video_latents, stg_video.float(), sigma)
            stg_audio_x0 = self.convert_velocity_to_x0(audio_latents, stg_audio.float(), sigma)
            video_stg_delta = stg_scale * (video_x0 - stg_video_x0)
            audio_stg_delta = audio_stg_scale * (audio_x0 - stg_audio_x0)

        # 4. Modality Isolation: extra forward with cross-modal attn disabled
        video_modality_delta = audio_modality_delta = 0
        if do_modality_isolation:
            with self.pipeline.transformer.cache_context("uncond_modality"):
                iso_video, iso_audio = self.transformer(
                    hidden_states=video_latents.to(dtype),
                    audio_hidden_states=audio_latents.to(dtype),
                    encoder_hidden_states=pos_text,
                    audio_encoder_hidden_states=pos_audio_text,
                    timestep=pos_ts,
                    sigma=pos_ts,
                    encoder_attention_mask=pos_mask,
                    audio_encoder_attention_mask=pos_mask,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    isolate_modalities=True,
                    spatio_temporal_guidance_blocks=None,
                    perturbation_mask=None,
                    **transformer_kwargs,
                )
            iso_video_x0 = self.convert_velocity_to_x0(video_latents, iso_video.float(), sigma)
            iso_audio_x0 = self.convert_velocity_to_x0(audio_latents, iso_audio.float(), sigma)
            video_modality_delta = (modality_scale - 1) * (video_x0 - iso_video_x0)
            audio_modality_delta = (audio_modality_scale - 1) * (audio_x0 - iso_audio_x0)

        # 5. Combine all guidance deltas in x0-space
        video_x0_guided = video_x0 + video_cfg_delta + video_stg_delta + video_modality_delta
        audio_x0_guided = audio_x0 + audio_cfg_delta + audio_stg_delta + audio_modality_delta

        # 6. Guidance rescale in x0-space
        if guidance_rescale > 0:
            video_x0_guided = rescale_noise_cfg(
                video_x0_guided, video_x0, guidance_rescale=guidance_rescale
            )
        if audio_guidance_rescale > 0:
            audio_x0_guided = rescale_noise_cfg(
                audio_x0_guided, audio_x0, guidance_rescale=audio_guidance_rescale
            )

        # 7. Convert back to velocity; with `preserve_raw_model_velocity`, keep the raw
        # prediction when no x0-space guidance transform ran.
        # The algebraic round trip is numerically unstable near sigma=0 and is
        # unnecessary for SFT/offline-DPO's neutral-guidance forward.
        if uses_x0_guidance or not preserve_raw_model_velocity:
            video_pred = self.convert_x0_to_velocity(
                video_latents,
                video_x0_guided,
                sigma,
            )
            audio_pred = self.convert_x0_to_velocity(
                audio_latents,
                audio_x0_guided,
                sigma,
            )
        else:
            video_pred = raw_video_pred
            audio_pred = raw_audio_pred

        velocity_only = (
            not compute_log_prob and next_latents is None and tuple(return_kwargs) == ("velocity",)
        )
        if velocity_only:
            if _return_components:
                return MultiModalStepOutput(
                    velocity=LatentState({"video": video_pred, "audio": audio_pred})
                )
            return FlowMatchEulerDiscreteSDESchedulerOutput(
                velocity=torch.cat([video_pred, audio_pred], dim=1)
            )

        # 8. Video: SDE scheduler step (with log_prob)
        video_output = self.scheduler.step(
            velocity=video_pred,
            timestep=t,
            latents=video_latents,
            timestep_next=t_next,
            next_latents=video_next,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            return_kwargs=return_kwargs,
            noise_level=noise_level,
        )

        # 9. Audio: SDE scheduler step (twin of video, with log_prob)
        audio_output = self.audio_scheduler.step(
            velocity=audio_pred,
            timestep=t,
            latents=audio_latents,
            timestep_next=t_next,
            next_latents=audio_next,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            return_kwargs=return_kwargs,
            noise_level=noise_level,
        )

        # 10. Component view, captured before the legacy concatenation mutates it
        if _return_components:
            return build_ltx2_component_step_output(
                self,
                video_output=video_output,
                audio_output=audio_output,
                video_velocity=video_pred,
                audio_velocity=audio_pred,
                n_video=video_latents[0].numel(),
                n_audio=audio_latents[0].numel(),
                compute_log_prob=compute_log_prob,
            )

        # 11. Concatenate back into unified latents
        if video_output.next_latents is not None and audio_output.next_latents is not None:
            video_output.next_latents = torch.cat(
                [video_output.next_latents, audio_output.next_latents],
                dim=1,
            )
        if (
            video_output.next_latents_mean is not None
            and getattr(audio_output, "next_latents_mean", None) is not None
        ):
            video_output.next_latents_mean = torch.cat(
                [video_output.next_latents_mean, audio_output.next_latents_mean],
                dim=1,
            )
        if video_output.velocity is not None:
            video_output.velocity = torch.cat(
                [video_output.velocity, audio_pred],
                dim=1,
            )

        # 12. Combine per-step log_prob across modalities
        # Joint transition p(v,a|z_t) = p(v|z_t) p(a|z_t); the element-weighted mean
        # reproduces what a single scheduler over the concatenated [video|audio] latent
        # would return, keeping the log_prob scale consistent with a video-only policy.
        if (
            compute_log_prob
            and video_output.log_prob is not None
            and audio_output.log_prob is not None
        ):
            video_output.log_prob = combine_modality_log_prob(
                video_output.log_prob,
                audio_output.log_prob,
                n_video=video_latents[0].numel(),
                n_audio=audio_latents[0].numel(),
            )

        return video_output

    # ============================== Trajectory Hooks ==============================

    def build_training_component_times(
        self,
        primary_timesteps: torch.Tensor,
        *,
        batch: Optional[StackedSampleBatch] = None,
    ) -> ComponentTimes:
        """Mirror one sampled coordinate onto the twin video/audio schedules.

        Args:
            primary_timesteps: Primary scheduler coordinates of shape ``(B,)``.
            batch: Unused; both components share the primary coordinate.

        Returns:
            Video/audio timesteps and sigmas in authoritative component order.
        """
        return build_ltx2_training_component_times(self, primary_timesteps)

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> NoisedState:
        """Draw video noise then audio noise, then apply both deterministically.

        Args:
            clean_state: Clean latent state in authoritative component order.
            times: Component times including each component's current sigma.
            generator: Optional generator shared by both ordered draws.

        Returns:
            Noised state, target velocity, and the sampled noise.
        """
        return draw_ltx2_forward_process_noise(self, clean_state, times, generator=generator)

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
        """Pack the ordered video/audio state for ``forward`` and unpack components.

        Args:
            batch: Collated batch supplying conditioning and ``video_seq_len``.
            state: Current state in authoritative component order.
            times: Current and next times in authoritative component order.
            next_state: Optional stored next state in authoritative component order.
            compute_log_prob: Whether to compute transition log probabilities.
            return_fields: Scheduler output fields requested from ``forward``.
            noise_level: Scheduler noise-level override.
            forward_kwargs: Model-conditioning arguments resolved by the wrapper.

        Returns:
            Ordered component step output carrying the input active masks.
        """
        validate_ltx2_forward_state_inputs(self, state=state, times=times, next_state=next_state)
        output = self.forward(
            **build_ltx2_joint_forward_kwargs(
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
        )
        return attach_ltx2_state_masks(state, output)

    # ============================== Inference ==============================

    @torch.no_grad()
    def inference(
        self,
        # Raw inputs
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # Generation shape
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        # Scheduling
        num_inference_steps: int = 40,
        sigmas: Optional[List[float]] = None,
        # Guidance (separate video/audio)
        guidance_scale: float = 4.0,
        audio_guidance_scale: Optional[float] = None,
        guidance_rescale: float = 0.0,
        audio_guidance_rescale: Optional[float] = None,
        noise_scale: float = 0.0,
        # STG
        stg_scale: float = 0.0,
        audio_stg_scale: Optional[float] = None,
        spatio_temporal_guidance_blocks: Optional[List[int]] = None,
        # Modality Isolation
        modality_scale: float = 1.0,
        audio_modality_scale: Optional[float] = None,
        # LTX-2.3 compat
        use_cross_timestep: bool = False,
        # Generator
        generator: Optional[torch.Generator] = None,
        # Pre-encoded inputs
        prompt_ids: Optional[torch.Tensor] = None,
        connector_prompt_embeds: Optional[torch.Tensor] = None,
        connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
        connector_attention_mask: Optional[torch.Tensor] = None,
        negative_connector_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_audio_prompt_embeds: Optional[torch.Tensor] = None,
        negative_connector_attention_mask: Optional[torch.Tensor] = None,
        # Decode options
        decode_timestep: float = 0.0,
        decode_noise_scale: Optional[float] = None,
        max_sequence_length: int = 1024,
        # RL-specific
        compute_log_prob: bool = True,
        trajectory_indices: TrajectoryIndicesType = "all",
        extra_call_back_kwargs: List[str] = [],
        **kwargs,
    ) -> List[LTX2Sample]:
        """Full denoising inference loop for LTX2 text-to-audio-video generation.

        Follows official pipeline __call__ with x0-space multi-guidance:
        CFG + STG (Spatio-Temporal Guidance) + Modality Isolation Guidance.
        """
        device = self.device

        # 0. Validate inputs
        num_frames = self._check_inputs(
            height,
            width,
            num_frames,
            prompt=prompt,
            connector_prompt_embeds=connector_prompt_embeds,
            negative_connector_prompt_embeds=negative_connector_prompt_embeds,
            guidance_scale=guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            stg_scale=stg_scale,
            audio_stg_scale=audio_stg_scale,
            spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
        )
        if isinstance(prompt, str):
            prompt = [prompt]

        # 1. Encode prompts
        if connector_prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_ids = encoded["prompt_ids"]
            connector_prompt_embeds = encoded["connector_prompt_embeds"]
            connector_audio_prompt_embeds = encoded["connector_audio_prompt_embeds"]
            connector_attention_mask = encoded["connector_attention_mask"]
            negative_connector_prompt_embeds = encoded.get("negative_connector_prompt_embeds")
            negative_connector_audio_prompt_embeds = encoded.get(
                "negative_connector_audio_prompt_embeds"
            )
            negative_connector_attention_mask = encoded.get("negative_connector_attention_mask")
        else:
            connector_prompt_embeds = connector_prompt_embeds.to(device)
            connector_audio_prompt_embeds = connector_audio_prompt_embeds.to(device)
            connector_attention_mask = connector_attention_mask.to(device)
            if negative_connector_prompt_embeds is not None:
                negative_connector_prompt_embeds = negative_connector_prompt_embeds.to(device)
                negative_connector_audio_prompt_embeds = negative_connector_audio_prompt_embeds.to(
                    device
                )
                negative_connector_attention_mask = negative_connector_attention_mask.to(device)

        batch_size = connector_prompt_embeds.shape[0]

        # 2. Compute dimensions
        vae_spatial = self.pipeline.vae_spatial_compression_ratio
        vae_temporal = self.pipeline.vae_temporal_compression_ratio
        latent_h = height // vae_spatial
        latent_w = width // vae_spatial
        latent_f = (num_frames - 1) // vae_temporal + 1

        duration_s = num_frames / frame_rate
        sr = self.pipeline.audio_sampling_rate
        hop = self.pipeline.audio_hop_length
        audio_temporal_compression = self.pipeline.audio_vae_temporal_compression_ratio
        audio_num_frames = round(duration_s * sr / hop / audio_temporal_compression)
        num_mel_bins = (
            self.pipeline.audio_vae.config.mel_bins
            if getattr(self.pipeline, "audio_vae", None) is not None
            else 64
        )

        # 3. Prepare latents
        video_latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=self.transformer_config.in_channels,
            height=height,
            width=width,
            num_frames=num_frames,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        audio_latents = self.pipeline.prepare_audio_latents(
            batch_size=batch_size,
            num_channels_latents=(
                self.pipeline.audio_vae.config.latent_channels
                if getattr(self.pipeline, "audio_vae", None) is not None
                else 8
            ),
            audio_latent_length=audio_num_frames,
            num_mel_bins=num_mel_bins,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        # 4. Set timesteps
        video_seq_len = latent_f * latent_h * latent_w
        mu = calculate_shift(
            video_seq_len,
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        timesteps = set_scheduler_timesteps(
            self.scheduler,
            num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
        )
        set_scheduler_timesteps(
            self.audio_scheduler,
            num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
        )

        # 5. Prepare positional coords
        video_coords = self.pipeline.transformer.rope.prepare_video_coords(
            batch_size,
            latent_f,
            latent_h,
            latent_w,
            device,
            fps=frame_rate,
        )
        audio_coords = self.pipeline.transformer.audio_rope.prepare_audio_coords(
            batch_size,
            audio_num_frames,
            device,
        )

        # 6. Setup trajectory collectors
        video_seq_len = video_latents.shape[1]
        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        # Concatenate video + audio into unified latents (B, video_seq + audio_seq, C)
        latents = self.cast_latents(torch.cat([video_latents, audio_latents], dim=1))
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(
                trajectory_indices, num_inference_steps
            )
            component_log_prob_collectors = {
                name: create_trajectory_collector(trajectory_indices, num_inference_steps)
                for name in LTX2_COMPONENT_ORDER
            }
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        # 7. Denoising loop
        for i, t in enumerate(timesteps):
            noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kw = list(set(["next_latents", "log_prob", "velocity"] + extra_call_back_kwargs))
            current_compute_log_prob: bool = compute_log_prob and noise_level > 0

            output = self.forward(
                t=t,
                t_next=t_next,
                latents=latents,
                video_seq_len=video_seq_len,
                connector_prompt_embeds=connector_prompt_embeds,
                connector_audio_prompt_embeds=connector_audio_prompt_embeds,
                connector_attention_mask=connector_attention_mask,
                negative_connector_prompt_embeds=negative_connector_prompt_embeds,
                negative_connector_audio_prompt_embeds=negative_connector_audio_prompt_embeds,
                negative_connector_attention_mask=negative_connector_attention_mask,
                guidance_scale=guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                guidance_rescale=guidance_rescale,
                audio_guidance_rescale=audio_guidance_rescale,
                stg_scale=stg_scale,
                audio_stg_scale=audio_stg_scale,
                spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
                modality_scale=modality_scale,
                audio_modality_scale=audio_modality_scale,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                audio_num_frames=audio_num_frames,
                video_coords=video_coords,
                audio_coords=audio_coords,
                noise_level=noise_level,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kw,
                use_cross_timestep=use_cross_timestep,
                _return_components=True,
            )

            # The component output is authoritative; the concatenated view only
            # advances the loop state and feeds the legacy callback collector.
            legacy_view = build_ltx2_legacy_callback_view(self, output)
            latents = self.cast_latents(legacy_view.next_latents)
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob:
                log_prob_collector.collect(output.log_prob, i)
                for name in LTX2_COMPONENT_ORDER:
                    component_log_prob_collectors[name].collect(output.component_log_probs[name], i)
            callback_collector.collect_step(
                step_idx=i,
                output=legacy_view,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": noise_level},
            )

        # 8. Split and Decode
        video_latents = latents[:, :video_seq_len]
        audio_latents = latents[:, video_seq_len:]
        video, audio_waveform = self.decode_latents(
            video_latents,
            audio_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type="pt",
            generator=generator,
        )

        # 9. Split the concatenated rollout into authoritative component trajectories
        collected_states = latent_collector.get_result()
        structured_callbacks, legacy_callbacks = split_ltx2_callback_results(
            self, callback_collector.get_result()
        )
        callback_index_map = callback_collector.get_index_map()
        trajectories = (
            build_ltx2_structured_trajectories(
                self,
                states=torch.stack(collected_states, dim=1),
                state_index_map=latent_collector.get_index_map(),
                video_seq_len=video_seq_len,
                schedule=build_ltx2_full_component_schedule(self, timesteps),
                callbacks=structured_callbacks,
                callback_index_map=callback_index_map if structured_callbacks else None,
                **build_ltx2_rollout_log_probs(
                    self,
                    log_prob_collector=log_prob_collector if compute_log_prob else None,
                    component_log_prob_collectors=(
                        component_log_prob_collectors if compute_log_prob else None
                    ),
                    num_transitions=num_inference_steps,
                ),
            )
            if collected_states
            else None
        )

        # 10. Construct samples (per-batch, NO batch dimension)
        prompt_list = prompt if isinstance(prompt, list) else [prompt] * batch_size

        samples = [
            LTX2Sample(
                # Authoritative video/audio trajectory; legacy fields stay unset.
                trajectory=None if trajectories is None else trajectories[b],
                # Generated media
                video=video[b],
                audio=audio_waveform[b] if audio_waveform is not None else None,
                audio_sample_rate=(
                    int(self.pipeline.vocoder.config.output_sampling_rate)
                    if audio_waveform is not None
                    else None
                ),
                # Metadata
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                video_seq_len=video_seq_len,
                # Prompt
                prompt=prompt_list[b],
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                # Connector embeddings (for training forward)
                connector_prompt_embeds=connector_prompt_embeds[b],
                connector_audio_prompt_embeds=connector_audio_prompt_embeds[b],
                connector_attention_mask=connector_attention_mask[b],
                negative_connector_prompt_embeds=(
                    negative_connector_prompt_embeds[b]
                    if negative_connector_prompt_embeds is not None
                    else None
                ),
                negative_connector_audio_prompt_embeds=(
                    negative_connector_audio_prompt_embeds[b]
                    if negative_connector_audio_prompt_embeds is not None
                    else None
                ),
                negative_connector_attention_mask=(
                    negative_connector_attention_mask[b]
                    if negative_connector_attention_mask is not None
                    else None
                ),
                extra_kwargs={
                    **{k: v[b] for k, v in legacy_callbacks.items()},
                    **({"callback_index_map": callback_index_map} if legacy_callbacks else {}),
                    "duration_s": duration_s,
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()
        return samples
