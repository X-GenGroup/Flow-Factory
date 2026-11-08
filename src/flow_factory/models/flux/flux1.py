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

# src/flow_factory/models/flux/flux1.py
from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from PIL import Image
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, PrepareModuleOutput, RowwiseParallel

from ...contracts import NegativePromptPolicy
from ...hparams import *
from ...samples import T2ISample
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SDESchedulerOutput,
    set_scheduler_timesteps,
)
from ...utils.base import filter_kwargs
from ...utils.dist import context_parallel_geometry
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import (
    CallbackCollector,
    TrajectoryCollector,
    TrajectoryIndicesType,
    create_callback_collector,
    create_trajectory_collector,
)
from ..abc import BaseAdapter
from ..configured_image_output import (
    ConfiguredImageOutputAdapterMixin,
    EncodedImageTensor,
)
from ..parallelism.context import (
    ContextParallelBoundary,
    ContextParallelCapability,
    ContextParallelLayout,
)
from ..parallelism.packed import PackedRowwiseParallel
from ..parallelism.plan import sequence_parallel_enabled
from ..parallelism.sequence import (
    ColumnParallelRegion,
    SequenceBroadcastParameter,
    SequenceParallelRegion,
)
from ..parallelism.tensor import ShardedInputReplicatedWeight, SharedInputColwiseParallel
from ..pipeline_contracts import image_output_contract
from ._output import prepare_flux1_output_latents
from ._sequence_parallel import (
    FluxModulationOutput,
    FluxSequenceParallelInput,
    FluxSingleStreamLayout,
    validate_flux_sequence_boundaries,
)

logger = setup_logger(__name__)


class FluxContextParallelLayout(ContextParallelLayout):
    """Stable identity for the qualified joint image/text CP layout."""

    JOINT_IMAGE_TEXT = "flux_joint_image_text"


@dataclass
class Flux1Sample(T2ISample):
    """Output class for Flux Adapter models."""

    # Class variables
    _shared_fields: ClassVar[frozenset[str]] = frozenset({"img_ids"})
    # Object variables
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    img_ids: Optional[torch.Tensor] = None


class Flux1Adapter(ConfiguredImageOutputAdapterMixin, BaseAdapter):
    """Concrete implementation for Flow Matching models (FLUX.1)."""

    tensor_parallel_head_config = ("transformer", ("num_attention_heads",))
    context_parallel_capability = ContextParallelCapability(
        layout=FluxContextParallelLayout.JOINT_IMAGE_TEXT, tensor_parallel=True
    )

    offline_training_forward_overrides = MappingProxyType({"guidance_scale": 3.5})
    pipeline_io_contract = image_output_contract(
        negative_prompt=NegativePromptPolicy.UNSUPPORTED,
    )

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: FluxPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

    def load_pipeline(self) -> FluxPipeline:
        if self.config.parallelism_args.effective_context_parallel_size > 1:
            from diffusers import FluxTransformer2DModel

            transformer_config = FluxTransformer2DModel.load_config(
                self.model_args.model_name_or_path, subfolder="transformer"
            )
            self._validate_context_parallel_config(
                transformer_config,
                self.config.parallelism_args.effective_context_parallel_size,
                self.config.parallelism_args.effective_tensor_parallel_size,
            )
        return self._load_diffusers_pipeline(
            FluxPipeline, self.model_args.model_name_or_path, low_cpu_mem_usage=False
        )

    @classmethod
    def validate_context_parallel_module(
        cls, module: torch.nn.Module, degree: int, tp_size: int = 1
    ) -> None:
        """Require the native guidance-distilled joint image/text layout."""
        from diffusers.models.transformers.transformer_flux import FluxAttnProcessor

        from diffusers import FluxTransformer2DModel

        if not isinstance(module, FluxTransformer2DModel):
            raise ValueError(
                f"Flux1 CP requires FluxTransformer2DModel, received {type(module).__name__}"
            )
        cls._validate_context_parallel_config(module.config, degree, tp_size)
        if not module.transformer_blocks or not module.single_transformer_blocks:
            raise ValueError("Flux1 CP requires both dual-stream and single-stream blocks")
        for name, child in module.named_modules():
            if hasattr(child, "processor") and not isinstance(child.processor, FluxAttnProcessor):
                raise ValueError(
                    f"Flux1 CP requires native FluxAttnProcessor at {name}; "
                    f"received {type(child.processor).__name__}"
                )

    @staticmethod
    def _validate_context_parallel_config(
        config: Mapping[str, Any], degree: int, tp_size: int = 1
    ) -> None:
        """Validate native metadata before weights and before installing CP hooks."""
        heads = config["num_attention_heads"]
        if heads % (degree * tp_size):
            raise ValueError(
                f"Flux1 CP requires num_attention_heads={heads} divisible by "
                f"cp_size={degree} * tp_size={tp_size}"
            )
        if config.get("guidance_embeds") is not True:
            raise ValueError("Flux1 CP requires guidance_embeds=True")
        if config.get("num_layers", 0) < 1 or config.get("num_single_layers", 0) < 1:
            raise ValueError("Flux1 CP requires both dual-stream and single-stream blocks")
        channels = config["in_channels"]
        output = config.get("out_channels") or channels
        if config.get("patch_size", 1) != 1 or channels % 4 or output != channels:
            raise ValueError(
                "Flux1 CP requires patch_size=1 and equal packed input/output channels divisible "
                f"by four; received in_channels={channels}, out_channels={output}, "
                f"patch_size={config.get('patch_size', 1)}"
            )

    def _validate_context_parallel_inputs(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        img_ids: torch.Tensor,
        joint_attention_kwargs: Optional[Dict[str, Any]],
    ) -> None:
        """Reject unsupported packing and conditioning before native collectives."""
        degree, _ = context_parallel_geometry(self.accelerator)
        if degree == 1:
            return
        if joint_attention_kwargs:
            raise ValueError(
                "Flux1 CP does not support joint_attention_kwargs, IP adapters or special masks"
            )
        config = self.pipeline.transformer.config
        expected = (
            ("latents", latents, 3, config.in_channels),
            ("prompt_embeds", prompt_embeds, 3, config.joint_attention_dim),
            ("pooled_prompt_embeds", pooled_prompt_embeds, 2, config.pooled_projection_dim),
        )
        for name, value, ndim, width in expected:
            if value.ndim != ndim or value.shape[0] != 1 or value.shape[-1] != width:
                raise ValueError(
                    f"Flux1 CP requires {name} with ndim={ndim}, batch=1, width={width}; "
                    f"received shape={tuple(value.shape)}"
                )
        for name, value in (("image", latents), ("text", prompt_embeds)):
            tokens = value.shape[1]
            text_padding = name == "text" and sequence_parallel_enabled(
                self, self.config.parallelism_args.sequence_parallel
            )
            if tokens == 0 or (tokens % degree and not text_padding):
                raise ValueError(
                    f"Flux1 CP {name} token count={tokens} must be positive and divisible by cp_size={degree}"
                )
        if img_ids.shape != (latents.shape[1], 3):
            raise ValueError(
                f"Flux1 CP requires img_ids shape=({latents.shape[1]}, 3); received {tuple(img_ids.shape)}"
            )

    @property
    def default_target_modules(self) -> List[str]:
        """Default Trainable target modules for FLUX.1-dev model."""
        return [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]

    @classmethod
    def tensor_parallel_plan(cls) -> Dict[str, Any]:
        """Shard both native stacks and modulation while preserving canonical weights.

        Single-block ``proj_out`` consumes independently sharded attention and
        MLP outputs in the native 1:4 ratio. PackedRowwiseParallel reversibly
        exchanges those activations before an ordinary contiguous weight shard;
        saved parameters and PEFT factors retain their native two-dimensional
        ordering. The SP overlay additionally partitions residual token rows.

        AdaLN linears shard input columns and reconstruct complete modulation
        vectors, including the final output norm. Every per-head q/k norm sums
        its partial weight gradients. Attention regions share the q/k/v input
        reduction; the single block's independent MLP input owns its own sum.
        """
        return {
            "transformer_blocks.*.norm1.linear": RowwiseParallel(input_layouts=Replicate()),
            "transformer_blocks.*.norm1_context.linear": RowwiseParallel(input_layouts=Replicate()),
            "single_transformer_blocks.*.norm.linear": RowwiseParallel(input_layouts=Replicate()),
            "norm_out.linear": RowwiseParallel(input_layouts=Replicate()),
            # Dual-stream (MMDiT) blocks: no fused projection anywhere. The block
            # calls `attn` by keyword with both streams.
            "transformer_blocks.*.attn": ColumnParallelRegion(
                inputs=("hidden_states", "encoder_hidden_states")
            ),
            "transformer_blocks.*.attn.to_q": SharedInputColwiseParallel(),
            "transformer_blocks.*.attn.to_k": SharedInputColwiseParallel(),
            "transformer_blocks.*.attn.to_v": SharedInputColwiseParallel(),
            "transformer_blocks.*.attn.to_out.0": RowwiseParallel(),
            "transformer_blocks.*.attn.add_q_proj": SharedInputColwiseParallel(),
            "transformer_blocks.*.attn.add_k_proj": SharedInputColwiseParallel(),
            "transformer_blocks.*.attn.add_v_proj": SharedInputColwiseParallel(),
            "transformer_blocks.*.attn.to_add_out": RowwiseParallel(),
            # `gelu-approximate` FeedForward: net.0.proj is one logical block,
            # unlike the GEGLU/SwiGLU packing of the later FLUX generations.
            "transformer_blocks.*.ff": ColumnParallelRegion(inputs=("hidden_states",)),
            "transformer_blocks.*.ff.net.0.proj": SharedInputColwiseParallel(),
            "transformer_blocks.*.ff.net.2": RowwiseParallel(),
            "transformer_blocks.*.ff_context": ColumnParallelRegion(inputs=("hidden_states",)),
            "transformer_blocks.*.ff_context.net.0.proj": SharedInputColwiseParallel(),
            "transformer_blocks.*.ff_context.net.2": RowwiseParallel(),
            # Per-head RMSNorm over attention_head_dim: replicated weight fed a
            # head-sharded input, so its gradient is a partial sum.
            "transformer_blocks.*.attn.norm_q": ShardedInputReplicatedWeight(),
            "transformer_blocks.*.attn.norm_k": ShardedInputReplicatedWeight(),
            "transformer_blocks.*.attn.norm_added_q": ShardedInputReplicatedWeight(),
            "transformer_blocks.*.attn.norm_added_k": ShardedInputReplicatedWeight(),
            "single_transformer_blocks.*.attn": ColumnParallelRegion(inputs=("hidden_states",)),
            "single_transformer_blocks.*.attn.to_q": SharedInputColwiseParallel(),
            "single_transformer_blocks.*.attn.to_k": SharedInputColwiseParallel(),
            "single_transformer_blocks.*.attn.to_v": SharedInputColwiseParallel(),
            "single_transformer_blocks.*.attn.norm_q": ShardedInputReplicatedWeight(),
            "single_transformer_blocks.*.attn.norm_k": ShardedInputReplicatedWeight(),
            "single_transformer_blocks.*.proj_mlp": ColwiseParallel(),
            "single_transformer_blocks.*.proj_out": PackedRowwiseParallel((1, 4)),
        }

    @classmethod
    def sequence_parallel_plan(cls) -> Dict[str, Any]:
        """Keep image/text residual shards through both stacks and the final image head.

        The root pads text to the CP/SP degree and masks padded keys in native
        attention. Both dual streams scatter after embedding and retain their
        shards between blocks. Single blocks concatenate local text/image rows;
        their rotary tables and masks follow the attention gather's rank-major
        order. The independent MLP arm gathers on its complete PEFT wrapper.

        AdaLN outputs sum token-partial gradients after native/PEFT branches
        merge. The final image norm and projection operate on local rows, then
        gather SP before the native CP output gather. Weights and checkpoint
        names stay canonical, and SP remains opt-in.
        """
        return {
            "": FluxSequenceParallelInput(),
            "x_embedder": PrepareModuleOutput(
                output_layouts=Replicate(),
                desired_output_layouts=Shard(1),
                use_local_output=True,
            ),
            "context_embedder": PrepareModuleOutput(
                output_layouts=Replicate(),
                desired_output_layouts=Shard(1),
                use_local_output=True,
            ),
            "transformer_blocks.*.norm1": FluxModulationOutput(),
            "transformer_blocks.*.norm1_context": FluxModulationOutput(),
            "transformer_blocks.*.attn": SequenceParallelRegion(
                gather=("hidden_states", "encoder_hidden_states")
            ),
            "transformer_blocks.*.attn.to_out.0": RowwiseParallel(output_layouts=Shard(1)),
            "transformer_blocks.*.attn.to_add_out": RowwiseParallel(output_layouts=Shard(1)),
            "transformer_blocks.*.ff": SequenceParallelRegion(gather=("hidden_states",)),
            "transformer_blocks.*.ff.net.2": RowwiseParallel(output_layouts=Shard(1)),
            "transformer_blocks.*.ff_context": SequenceParallelRegion(gather=("hidden_states",)),
            "transformer_blocks.*.ff_context.net.2": RowwiseParallel(output_layouts=Shard(1)),
            "single_transformer_blocks.*": FluxSingleStreamLayout(),
            "single_transformer_blocks.*.norm": FluxModulationOutput(),
            "single_transformer_blocks.*.attn": SequenceParallelRegion(gather=("hidden_states",)),
            "single_transformer_blocks.*.proj_out": PackedRowwiseParallel(
                (1, 4), output_layouts=Shard(1)
            ),
            "norm_out": FluxModulationOutput(),
            "proj_out": SequenceBroadcastParameter(),
        }

    @classmethod
    def compose_context_parallel_sequence_boundaries(
        cls, module: torch.nn.Module
    ) -> tuple[ContextParallelBoundary, ...]:
        """Keep native CP input cuts before SP and relocate the output gather after SP."""
        return validate_flux_sequence_boundaries(module)

    # ========================== Tokenizer & Text Encoder ==========================
    @property
    def tokenizer(self) -> Any:
        """Use T5 for longer context length."""
        return self.pipeline.tokenizer_2

    @property
    def text_encoder(self) -> Any:
        """Use T5 text encoder."""
        return self.pipeline.text_encoder_2

    # ======================== Encoding & Decoding ========================

    def encode_prompt(self, prompt: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Encode text prompts using the pipeline's text encoder."""

        execution_device = self.pipeline.text_encoder.device

        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=execution_device,
        )

        prompt_ids = self.pipeline.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(execution_device)

        return {
            "prompt_ids": prompt_ids,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

    def encode_image(self, images: Union[Image.Image, List[Optional[Image.Image]]]) -> None:
        """Not needed for FLUX text-to-image models."""
        pass

    def encode_video(self, videos: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        """Not needed for FLUX text-to-image models."""
        pass

    def _output_geometry_multiple(self) -> int:
        """Require the VAE grid and 2x2 latent packing used by Diffusers."""
        return self.pipeline.vae_scale_factor * 2

    def _encode_output_images(
        self,
        pixel_values: torch.Tensor,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator],
    ) -> EncodedImageTensor:
        """Sample FLUX.1 targets and attach their packed position identifiers."""
        del condition
        packed, img_ids = prepare_flux1_output_latents(self, pixel_values, generator)
        return EncodedImageTensor(
            latents=packed,
            forward_context={"img_ids": img_ids},
            decode_context={},
        )

    def decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: Literal["pil", "pt", "np"] = "pil",
    ) -> Union[List[Image.Image], torch.Tensor, np.ndarray]:
        """Decode latents to images using VAE."""

        latents = self.pipeline._unpack_latents(
            latents, height, width, self.pipeline.vae_scale_factor
        )
        latents = (
            latents / self.pipeline.vae.config.scaling_factor
        ) + self.pipeline.vae.config.shift_factor
        latents = latents.to(dtype=self.pipeline.vae.dtype)

        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type=output_type)

        return images

    # ======================== Inference ========================

    @torch.no_grad()
    def inference(
        self,
        # Ordinary args
        prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        generator: Optional[torch.Generator] = None,
        # Encoded prompt
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # Other args
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
    ) -> List[Flux1Sample]:
        """Execute generation and return ``Flux1Sample`` objects."""

        # 1. Setup
        device = self.device

        # 2. Encode prompts if not provided
        if prompt_embeds is None:
            encoded = self.encode_prompt(prompt)
            prompt_embeds = encoded["prompt_embeds"]
            pooled_prompt_embeds = encoded["pooled_prompt_embeds"]
            prompt_ids = encoded["prompt_ids"]
        else:
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)

        batch_size = len(prompt_embeds)
        dtype = prompt_embeds.dtype

        # 3. Prepare latents
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, latent_image_ids = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # 4. Set timesteps with scheduler
        timesteps = set_scheduler_timesteps(
            scheduler=self.pipeline.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device,
        )

        # 5. Denoising loop
        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latents = self.cast_latents(latents, default_dtype=dtype)
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(
                trajectory_indices, num_inference_steps
            )
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        for i, t in enumerate(timesteps):
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
                pooled_prompt_embeds=pooled_prompt_embeds,
                img_ids=latent_image_ids,
                guidance_scale=guidance_scale,
                compute_log_prob=current_compute_log_prob,
                joint_attention_kwargs=joint_attention_kwargs,
                return_kwargs=return_kwargs,
                noise_level=current_noise_level,
            )

            latents = self.cast_latents(output.next_latents, default_dtype=dtype)
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob:
                log_prob_collector.collect(output.log_prob, i)

            callback_collector.collect_step(
                step_idx=i,
                output=output,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": current_noise_level},
            )

        # 6. Decode images
        images = self.decode_latents(latents, height, width, output_type="pt")

        # 7. Create samples
        extra_call_back_res = callback_collector.get_result()  # (B, len(trajectory_indices), ...)
        callback_index_map = callback_collector.get_index_map()  # (T,) LongTensor
        all_latents = latent_collector.get_result()  # List[torch.Tensor(B, ...)]
        latent_index_map = latent_collector.get_index_map()  # (T+1,) LongTensor
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
        samples = [
            Flux1Sample(
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
                # Prompt
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b],
                pooled_prompt_embeds=pooled_prompt_embeds[b],
                # Image & metadata
                height=height,
                width=width,
                image=images[b],
                img_ids=latent_image_ids,
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
        # Prompt
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        # Image ids
        img_ids: torch.Tensor,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other args
        guidance_scale: Union[float, List[float]] = 3.5,
        noise_level: Optional[float] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "velocity",
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
        ],
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """Forward pass with given timestep, timestep+1 and latents."""
        self._validate_context_parallel_inputs(
            latents, prompt_embeds, pooled_prompt_embeds, img_ids, joint_attention_kwargs
        )
        # 1. Prepare variables
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        guidance = torch.as_tensor(guidance_scale, device=device, dtype=dtype)
        guidance = guidance.expand(batch_size)  # Assume List[float] has len `batch_size`

        # 2. transformer forward
        velocity = self.transformer(
            hidden_states=latents,
            timestep=t.expand(batch_size) / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype),
            img_ids=img_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

        # 3. Scheduler step
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
