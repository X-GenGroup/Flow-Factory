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

# src/flow_factory/models/wan/wan2_t2v.py
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from peft import PeftModel
from PIL import Image
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)

from diffusers.pipelines.wan.pipeline_wan import WanPipeline, prompt_clean

from ...contracts import BatchCapability, GeometrySource, NegativePromptPolicy, RateRequirement
from ...hparams import *
from ...samples import T2VSample
from ...scheduler import UniPCMultistepSDEScheduler, UniPCMultistepSDESchedulerOutput
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import (
    TrajectoryIndicesType,
    create_callback_collector,
    create_trajectory_collector,
)
from ..abc import BaseAdapter
from ..output_state import DecodedMediaBatch, EncodedOutputState, OutputStateCodec
from ..pipeline_contracts import video_output_contract
from ..tensor_parallel import (
    ComposedParallelStyle,
    SequenceBroadcastOutput,
    SequenceBroadcastParameter,
    ShardedAcrossHeadRMSNorm,
    ShardedInputReplicatedWeight,
)
from ._output import (
    WanVideoOutputCodec,
    configured_wan_video_output_geometry,
    normalize_wan_video_latents,
    resample_wan_output_video,
    validate_wan_encoded_output_geometry,
)

logger = setup_logger(__name__)


@dataclass
class WanT2VSample(T2VSample):
    #  Class var
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})


class Wan2_T2V_Adapter(BaseAdapter):
    offline_training_forward_overrides = MappingProxyType(
        {"guidance_scale": 1.0, "guidance_scale_2": 1.0}
    )
    # Wan2.2 trains both transformer and transformer_2 but uses only one per
    # timestep (boundary_ratio), so under DDP the other's trainable params get no
    # gradient in a given step. Ignored under DeepSpeed/FSDP.
    ddp_find_unused_parameters = True
    supports_diffusers_cache = True
    component_load_dtype_defaults = {
        "transformers": torch.bfloat16,
        "text_encoders": torch.bfloat16,
        "vae": torch.float32,
    }
    pipeline_io_contract = video_output_contract(
        negative_prompt=NegativePromptPolicy.OPTIONAL,
        output_fps=RateRequirement.REQUIRED,
        geometry_source=GeometrySource.CONFIGURED,
        batch_capability=BatchCapability.SINGLE_SAMPLE,
    )

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: WanPipeline
        self.scheduler: UniPCMultistepSDEScheduler

    def load_pipeline(self) -> WanPipeline:
        return self._load_diffusers_pipeline(
            WanPipeline,
            self.model_args.model_name_or_path,
        )

    def apply_lora(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]] = ["transformer", "transformer_2"],
        **kwargs,
    ) -> Union[PeftModel, Dict[str, PeftModel]]:
        return super().apply_lora(target_modules=target_modules, components=components, **kwargs)

    # ============================ Module Management ============================
    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Wan transformer."""
        return [
            # Self Attention
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            # Cross Attention
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            # Feed Forward Network
            "ffn.net.0.proj",
            "ffn.net.2",
        ]

    @classmethod
    def tensor_parallel_plan(cls) -> Dict[str, Any]:
        """Shard every block's attention and feed-forward.

        Wan's qk-norm is ``rms_norm_across_heads``: ``WanAttention`` builds
        ``RMSNorm(num_heads * attention_head_dim)`` -- ``[5120]`` on the 14B
        config, the whole inner dimension -- and ``WanAttnProcessor`` applies it
        to the flat projection output, before heads are split. Column-parallel
        q/k therefore split the *normalized* axis, which makes the forward wrong
        rather than merely the weight gradient, and it says so:
        ``Given normalized_shape=[5120] ... got input of size [1, 320, 2560]``.
        :class:`~flow_factory.models.tensor_parallel.ShardedAcrossHeadRMSNorm`
        completes it by sharding the weight alongside the projection and
        all-reducing the mean square, which is one ``[batch, seq, 1]`` collective
        per norm.

        ``attn2`` is cross-attention over the text encoder states. Column
        parallelism splits its *output* -- the head axis -- so the text stream is
        still consumed at full width and needs no extra communication; the row
        parallel ``to_out.0`` restores it.

        ``ffn`` is a plain GELU MLP rather than a packed SwiGLU, so both halves
        shard without the ``[hidden, gate]`` interleaving that keeps H3's fused
        projection replicated. Sharding it also halves the ``ffn_dim``-wide
        intermediate, the largest single activation per token.

        What stays replicated is the modulation, and it is where this
        architecture actually spends its memory: ``condition_embedder.time_proj``
        (1.10% of the model) emits six full-width vectors which each block's bare
        ``scale_shift_table`` (0.009%) offsets, applied in fp32 to hidden states
        that stay replicated. Measured on 75,600 video tokens in bf16, that fp32
        chain -- not attention, which SDPA never materializes -- dominates the
        high-water mark, so a feed-forward-only plan saved just 7.8% of peak.
        Sharding attention as well took the same measurement from 7.07 GB to
        4.45 GB. Sharding the modulation would take sequence parallelism.

        The plan now covers every parameter in a block -- 98.4% of the 14B model,
        with only the patch embedder, condition embedder and output projection
        left out. That matters more than the activation saving: per-rank
        parameter bytes are ``P*(c + (1-c)*t)/W``, so an uncovered fraction is
        charged ``tp_size``, and at ``c=0.984`` the excess over FSDP-only
        sharding is 1.6% at tp=2 and 11.5% at tp=8. A plan that covered only the
        feed-forward (``c=0.40``) would have cost 60% more at tp=2 than not
        using tensor parallelism at all.

        The plan is relative to ``WanTransformer3DModel``, which Wan2.2 holds
        twice (``transformer`` and ``transformer_2``); both are sharded with it.
        ``Wan2_I2V_Adapter`` repeats these entries rather than inheriting them,
        because the adapter hierarchy is deliberately flat, and adds three of
        its own: its cross-attention takes a second key/value stream for the
        image conditioning, which has to shard with the query it meets.
        """
        return {
            "blocks.*.attn1.to_q": ColwiseParallel(),
            "blocks.*.attn1.to_k": ColwiseParallel(),
            "blocks.*.attn1.to_v": ColwiseParallel(),
            "blocks.*.attn1.to_out.0": RowwiseParallel(),
            "blocks.*.attn1.norm_q": ShardedAcrossHeadRMSNorm(),
            "blocks.*.attn1.norm_k": ShardedAcrossHeadRMSNorm(),
            "blocks.*.attn2.to_q": ColwiseParallel(),
            "blocks.*.attn2.to_k": ColwiseParallel(),
            "blocks.*.attn2.to_v": ColwiseParallel(),
            "blocks.*.attn2.to_out.0": RowwiseParallel(),
            "blocks.*.attn2.norm_q": ShardedAcrossHeadRMSNorm(),
            "blocks.*.attn2.norm_k": ShardedAcrossHeadRMSNorm(),
            "blocks.*.ffn.net.0.proj": ColwiseParallel(),
            "blocks.*.ffn.net.2": RowwiseParallel(),
        }

    @classmethod
    @classmethod
    def sequence_parallel_plan(cls) -> Dict[str, Any]:
        """Shard the token axis from the patch embedding through the output projection.

        This is where Wan actually spends its memory. Every block wraps each
        sublayer in ``(norm(hidden.float()) * (1 + scale) + shift)`` and the
        root repeats it once more before ``proj_out`` -- full-width fp32
        arithmetic over every token, which tensor parallelism leaves replicated
        because it shards weights, not activations. Measured at 75,600 video
        tokens, that chain holds about 47 GB of the 53 GB high-water mark while
        attention holds almost none, SDPA never materializing the score matrix.

        The entries below are of five kinds:

        * ``blocks.0`` scatters the sequence on the way in; ``proj_out`` gathers
          it on the way out. Gathering there rather than after the last block
          keeps the root's own modulation sharded too, and costs far less: the
          projection is ``inner_dim -> patch * out_channels``, 5120 -> 64 on the
          14B config, so the gathered tensor is eighty times smaller.
        * every attention and feed-forward entry gathers back to the whole
          sequence, so attention sees every token and the rotary embeddings and
          cross-attention text stream need no slicing at all.
        * their out-projections reduce-scatter instead of all-reducing, which is
          the same traffic split differently -- an all-reduce *is* a
          reduce-scatter followed by an all-gather, so sequence parallelism adds
          no communication.
        * ``norm2``'s affine weight and ``proj_out``'s weight are applied to each
          token independently, so each rank accumulates a partial sum of their
          gradient: both take ``SequenceBroadcastParameter``. ``norm1``,
          ``norm3`` and ``norm_out`` have no affine parameters and need no entry
          at all -- normalizing over the last axis is already correct on a token
          slice.
        * ``condition_embedder``'s two timestep outputs are broadcast over every
          token, so the gradient arriving back at them is a partial sum that no
          placement records -- the value itself was never sharded.
          :class:`~flow_factory.models.tensor_parallel.SequenceBroadcastOutput`
          sums it. Omitting this leaves the time embedder training on 27% of
          its gradient, with a bit-identical loss. Its text outputs need nothing:
          they enter cross-attention, where column parallelism already reduces
          the input gradient.
        * the bare ``scale_shift_table`` on every block and on the root is read
          inside a ``forward`` with no module boundary to hook, so it takes
          :class:`~flow_factory.models.tensor_parallel.SequenceBroadcastParameter`.
          Omitting it is silent: the loss stays bit-identical and only that
          table's gradient comes out at this rank's share of the tokens,
          measured at exactly 1/2 on two ranks.

        ``blocks.0`` appears both here and under ``blocks.*``; an exact path
        overrides a wildcard regardless of order.
        """
        # `WanAttention` is called with four positional arguments
        # (`hidden_states, encoder_hidden_states, attention_mask, rotary_emb`),
        # and `PrepareModuleInput` requires one layout per argument. Only the
        # first is token-sharded; the text stream and the rotary tables are
        # whole on every rank already.
        gather_attention = PrepareModuleInput(
            input_layouts=(Shard(1), None, None, None),
            desired_input_layouts=(Replicate(), None, None, None),
            use_local_output=True,
        )
        gather_feed_forward = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
            use_local_output=True,
        )
        return {
            "condition_embedder": SequenceBroadcastOutput(output_indices=(0, 1)),
            "": SequenceBroadcastParameter(),
            "blocks.*": SequenceBroadcastParameter(),
            "blocks.0": ComposedParallelStyle(
                SequenceBroadcastParameter(),
                PrepareModuleInput(
                    input_layouts=(Replicate(), None, None, None),
                    desired_input_layouts=(Shard(1), None, None, None),
                    use_local_output=True,
                ),
            ),
            "blocks.*.norm2": SequenceBroadcastParameter(),
            "blocks.*.attn1": gather_attention,
            "blocks.*.attn2": gather_attention,
            "blocks.*.ffn": gather_feed_forward,
            "blocks.*.attn1.to_out.0": RowwiseParallel(output_layouts=Shard(1)),
            "blocks.*.attn2.to_out.0": RowwiseParallel(output_layouts=Shard(1)),
            "blocks.*.ffn.net.2": RowwiseParallel(output_layouts=Shard(1)),
            "proj_out": ComposedParallelStyle(
                SequenceBroadcastParameter(),
                PrepareModuleOutput(
                    output_layouts=Shard(1),
                    desired_output_layouts=Replicate(),
                    use_local_output=True,
                ),
            ),
        }

    @property
    def inference_modules(self) -> List[str]:
        """Modules that are required for inference and forward."""
        if self.pipeline.config.boundary_ratio is None or self.pipeline.config.boundary_ratio <= 0:
            return ["transformer", "vae"]

        if self.pipeline.config.boundary_ratio >= 1:
            return ["transformer_2", "vae"]

        return ["transformer", "transformer_2", "vae"]

    # ======================== Component Getters & Setters ========================
    @property
    def transformer_2(self) -> torch.nn.Module:
        return self.get_component("transformer_2")

    @transformer_2.setter
    def transformer_2(self, module: torch.nn.Module):
        self.set_component("transformer_2", module)

    # ======================== Encoding & Decoding ========================
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.pipeline.text_encoder(
            text_input_ids.to(device), mask.to(device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        return text_input_ids, prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 3.5,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Encode prompts, and negative prompts when CFG is enabled, with the T5 encoder.

        Args:
            prompt: Prompt or prompts to encode.
            negative_prompt: Negative prompt or prompts, repeated to the prompt batch size.
                Defaults to ``""``; encoded only when CFG is enabled.
            guidance_scale: Classifier-free guidance scale; CFG is enabled when it exceeds 1.0.
            max_sequence_length: Token budget; prompts are truncated and embeddings
                zero-padded to it.
            device: Target device; defaults to the text encoder's device.
            dtype: Target dtype; defaults to the text encoder's dtype.

        Returns:
            Dict with ``prompt_ids`` and ``prompt_embeds``, plus ``negative_prompt_ids`` and
            ``negative_prompt_embeds`` when CFG is enabled.
        """
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_ids, prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        results = {
            "prompt_ids": prompt_ids,
            "prompt_embeds": prompt_embeds,
        }

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            )
            negative_prompt = negative_prompt * (
                len(prompt) // len(negative_prompt)
            )  # Expand to match batch size
            assert len(negative_prompt) == len(
                prompt
            ), "The number of negative prompts must match the number of prompts."

            negative_prompt_ids, negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            results.update(
                {
                    "negative_prompt_ids": negative_prompt_ids,
                    "negative_prompt_embeds": negative_prompt_embeds,
                }
            )

        return results

    def encode_image(self, images: Union[Image.Image, torch.Tensor, List[torch.Tensor]]):
        """Not needed for Wan text-to-video models."""
        return None

    def encode_video(self, videos: Union[torch.Tensor, List[torch.Tensor]]):
        """Not needed for Wan text-to-video models."""
        return None

    def build_output_state_codec(self) -> OutputStateCodec:
        """Declare the on-the-fly target-video codec without loading components."""
        return WanVideoOutputCodec(self)

    def _configured_video_output_geometry(self) -> Tuple[int, int, int, float]:
        """Return configured Wan geometry after exact latent-grid validation."""
        return configured_wan_video_output_geometry(self)

    @staticmethod
    def _resample_output_video(
        video: np.ndarray,
        *,
        source_fps: Optional[float],
        target_frames: int,
        target_fps: float,
    ) -> np.ndarray:
        """Select deterministic nearest-time frames for configured target cadence."""
        return resample_wan_output_video(
            video,
            source_fps=source_fps,
            target_frames=target_frames,
            target_fps=target_fps,
        )

    def _normalize_output_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply the exact inverse of the latent de-normalization in ``decode_latents``."""
        return normalize_wan_video_latents(self, latents)

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Require encoded signatures and decode metadata to match train geometry."""
        validate_wan_encoded_output_geometry(self, media_batch, condition, encoded)

    def decode_latents(
        self, latents: torch.Tensor, output_type: Literal["pt", "pil", "np"] = "pil"
    ) -> torch.Tensor:
        """Decode the latents using the VAE decoder."""
        latents = latents.float()
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(
            1, self.pipeline.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        video = self.pipeline.vae.decode(latents, return_dict=False)[0]

        video = self.pipeline.video_processor.postprocess_video(video, output_type=output_type)
        return video

    # ======================== Inference ========================

    @torch.no_grad()
    def inference(
        self,
        # Ordinary args
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # Prompt encoding args
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        # Other args
        compute_log_prob: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
    ) -> List[WanT2VSample]:
        # 1. Setup args
        device = self.device
        do_classifier_free_guidance = guidance_scale > 1.0

        if self.pipeline.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale
        # Check `num_frames`
        if (num_frames - 1) % self.pipeline.vae_scale_factor_temporal != 0:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.pipeline.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames
                // self.pipeline.vae_scale_factor_temporal
                * self.pipeline.vae_scale_factor_temporal
                + 1
            )
        num_frames = max(num_frames, 1)
        # Check `height` and `width`
        patch_size = (
            self.pipeline.transformer.config.patch_size
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.config.patch_size
        )
        h_multiple_of = self.pipeline.vae_scale_factor_spatial * patch_size[1]
        w_multiple_of = self.pipeline.vae_scale_factor_spatial * patch_size[2]
        calc_height = height // h_multiple_of * h_multiple_of
        calc_width = width // w_multiple_of * w_multiple_of
        if height != calc_height or width != calc_width:
            logger.warning(
                f"`height` and `width` must be multiples of ({h_multiple_of}, {w_multiple_of}) for proper patchification. "
                f"Adjusting ({height}, {width}) -> ({calc_height}, {calc_width})."
            )
            height, width = calc_height, calc_width

        # 2. Encode prompt
        if prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_ids = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
            negative_prompt_ids = encoded.get("negative_prompt_ids", None)
            negative_prompt_embeds = encoded.get("negative_prompt_embeds", None)
        else:
            prompt_embeds = prompt_embeds.to(device)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device)

        batch_size = prompt_embeds.shape[0]
        transformer_dtype = (
            self.pipeline.transformer.dtype
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.dtype
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 3. Set scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare latent variables
        num_channels_latents = (
            self.pipeline.transformer.config.in_channels
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.config.in_channels
        )
        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        # 5. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.pipeline._num_timesteps = len(timesteps)

        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latents = self.cast_latents(latents)
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(
                trajectory_indices, num_inference_steps
            )
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        for i, t in enumerate(timesteps):
            self.pipeline._current_timestep = t
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kwargs = list(
                set(["next_latents", "log_prob", "velocity"] + extra_call_back_kwargs)
            )
            current_compute_log_prob = compute_log_prob and current_noise_level > 0

            output = self.forward(
                t=t,
                t_next=t_next,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                attention_kwargs=attention_kwargs,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=current_noise_level,
            )

            latents = self.cast_latents(output.next_latents)
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob:
                log_prob_collector.collect(output.log_prob, i)

            callback_collector.collect_step(
                step_idx=i,
                output=output,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": current_noise_level},
            )

        self.pipeline._current_timestep = None

        # 6. Decode latents to videos
        decoded_videos = self.decode_latents(latents, output_type="pt")

        # 7. Prepare output samples
        extra_call_back_res = callback_collector.get_result()  # (B, len(trajectory_indices), ...)
        callback_index_map = callback_collector.get_index_map()  # (T,) LongTensor
        all_latents = latent_collector.get_result()  # List[torch.Tensor(B, ...)]
        latent_index_map = latent_collector.get_index_map()  # (T+1,) LongTensor
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
        samples = [
            WanT2VSample(
                # Denoising trajectory
                timesteps=timesteps,
                all_latents=(
                    torch.stack([lat[b] for lat in all_latents], dim=0)
                    if all_latents is not None
                    else None
                ),
                log_probs=(
                    torch.stack([lp[b] for lp in all_log_probs], dim=0)
                    if all_log_probs is not None
                    else None
                ),
                latent_index_map=latent_index_map,
                log_prob_index_map=log_prob_index_map,
                # Generated video & metadata
                video=decoded_videos[b],
                height=height,
                width=width,
                # Prompt info
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b],
                prompt_embeds=prompt_embeds[b],
                # Negative prompt info
                negative_prompt=(
                    negative_prompt[b] if isinstance(negative_prompt, list) else negative_prompt
                ),
                negative_prompt_ids=(
                    negative_prompt_ids[b] if negative_prompt_ids is not None else None
                ),
                negative_prompt_embeds=(
                    negative_prompt_embeds[b] if negative_prompt_embeds is not None else None
                ),
                # Extra kwargs
                extra_kwargs={
                    **{k: v[b] for k, v in extra_call_back_res.items()},
                    "callback_index_map": callback_index_map,
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()

        return samples

    # ======================== Forward (Training) ========================

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        # Optional for CFG
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        noise_level: Optional[float] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "velocity",
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
        ],
        boundary_timestep: Optional[float] = None,
    ) -> UniPCMultistepSDESchedulerOutput:
        """Run one T2V denoising step: transformer forward, optional CFG, and scheduler step.

        Args:
            t: Current timestep; a 1-D tensor uses ``t[0]``, so the batch shares one timestep.
            latents: Current latent representations (B, C, T, H, W).
            prompt_embeds: Text prompt embeddings.
            negative_prompt_embeds: Optional negative prompt embeddings (for CFG).
            guidance_scale: CFG scale for transformer (wan2.1 / wan2.2 high-noise).
            guidance_scale_2: CFG scale for transformer_2 (wan2.2 low-noise); defaults to
                ``guidance_scale``.
            t_next: Next timestep for the scheduler step; reduced like ``t``.
            next_latents: Optional target latents for log-prob computation.
            noise_level: Current noise level for SDE sampling.
            attention_kwargs: Optional kwargs for attention layers.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            boundary_timestep: Timestep threshold for switching transformers (wan2.2);
                defaults to ``boundary_ratio * num_train_timesteps`` when the pipeline sets
                ``boundary_ratio``.

        Returns:
            UniPCMultistepSDESchedulerOutput with the requested fields. The scheduler step is
            skipped when only ``velocity`` is requested, ``compute_log_prob`` is False, and
            ``next_latents`` is None.
        """
        # 1. Prepare variables
        t = t[0] if t.ndim == 1 else t  # A scalar
        if t_next is not None:
            t_next = t_next[0] if t_next.ndim == 1 else t_next

        batch_size = latents.shape[0]
        device = latents.device
        dtype = (
            self.pipeline.transformer.dtype
            if self.pipeline.transformer is not None
            else self.pipeline.transformer_2.dtype
        )

        # Determine boundary timestep
        if boundary_timestep is None and self.pipeline.config.boundary_ratio is not None:
            boundary_timestep = (
                self.pipeline.config.boundary_ratio * self.scheduler.config.num_train_timesteps
            )
        # Determine which transformer to use
        if boundary_timestep is None or t >= boundary_timestep:
            pipeline_transformer = self.pipeline.transformer
            transformer = self.transformer
            current_guidance_scale = guidance_scale
        else:
            pipeline_transformer = self.pipeline.transformer_2
            transformer = self.transformer_2
            current_guidance_scale = (
                guidance_scale_2 if guidance_scale_2 is not None else guidance_scale
            )

        # Auto-detect CFG
        if current_guidance_scale > 1.0 and negative_prompt_embeds is None:
            logger.warning(
                "Passed `guidance_scale` > 1.0, but no `negative_prompt_embeds` provided. "
                "Classifier-free guidance will be disabled."
            )
        do_classifier_free_guidance = (
            negative_prompt_embeds is not None and current_guidance_scale > 1.0
        )

        # 2. Prepare timestep
        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)
        latent_model_input = latents.to(dtype)

        if self.pipeline.config.expand_timesteps:
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            timestep = temp_ts.unsqueeze(0).expand(batch_size, -1)
        else:
            timestep = t.expand(batch_size)

        # 3. Transformer forward pass
        with pipeline_transformer.cache_context("cond"):
            velocity = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

        # 4. Apply CFG
        if do_classifier_free_guidance:
            with pipeline_transformer.cache_context("uncond"):
                velocity_uncond = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            velocity = velocity_uncond + current_guidance_scale * (velocity - velocity_uncond)

        if not compute_log_prob and next_latents is None and tuple(return_kwargs) == ("velocity",):
            return UniPCMultistepSDESchedulerOutput(velocity=velocity)

        # 5. Scheduler step
        output = self.scheduler.step(
            velocity=velocity,
            timestep=t,
            latents=latents,
            timestep_next=t_next,
            next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            return_kwargs=return_kwargs,
            noise_level=noise_level,
        )

        return output
