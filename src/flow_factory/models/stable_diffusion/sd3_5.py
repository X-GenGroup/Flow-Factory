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

# src/flow_factory/models/stable_diffusion/sd3_5.py
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from PIL import Image
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
)

from ...contracts import NegativePromptPolicy
from ...hparams import *
from ...samples import BaseSample
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SDESchedulerOutput,
    set_scheduler_timesteps,
)
from ...utils.base import filter_kwargs
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
    encode_shift_scale_vae_image,
)
from ..pipeline_contracts import image_output_contract
from ..tensor_parallel import ShardedInputReplicatedWeight

logger = setup_logger(__name__)


@dataclass
class SD3_5Sample(BaseSample):
    # Class var
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    # Obj var
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None
    pooled_prompt_embeds: Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None


class SD3_5Adapter(ConfiguredImageOutputAdapterMixin, BaseAdapter):
    """Concrete implementation for Stable Diffusion 3.5 medium."""

    pipeline_io_contract = image_output_contract(
        negative_prompt=NegativePromptPolicy.OPTIONAL,
    )

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: StableDiffusion3Pipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

    def load_pipeline(self) -> StableDiffusion3Pipeline:
        return self._load_diffusers_pipeline(
            StableDiffusion3Pipeline,
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False,
        )

    # ============================ Modules & Components ============================
    @property
    def default_target_modules(self) -> List[str]:
        """Mirror the adapter surface upstream's SD3 LoRA script trains.

        The ``attn.`` prefix is load-bearing and deliberate. PEFT matches a
        target by suffix, so it reaches ``attn`` and *not* ``attn2`` -- the
        second self-attention the ``dual_attention_layers`` carry (blocks 0-12
        of 24 in SD3.5-medium). Upstream's ``train_dreambooth_lora_sd3.py``
        defaults to exactly this list, and published SD3.5 LoRAs are keyed to
        it, so widening the surface here would silently make every adapter this
        framework writes incompatible with that ecosystem. Train ``attn2`` by
        setting ``model.target_modules`` explicitly, not by changing the default.

        The tensor-parallel plan does cover ``attn2``: sharding is about where a
        weight lives, not about which weights are trained.
        """
        return [
            # Attention modules
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
        ]

    @classmethod
    def tensor_parallel_plan(cls) -> Dict[str, Any]:
        """Shard both MMDiT streams and both FFNs of every joint block.

        SD3.5 medium is 24 heads of 64 over an inner dim of 1536, so tp=2/4/8
        all divide the head count. ``JointTransformerBlock`` keeps one
        ``Attention`` holding *both* streams -- ``to_{q,k,v}``/``to_out.0`` for
        the image tokens and ``add_{q,k,v}_proj``/``to_add_out`` for the text
        tokens -- whose per-head projections are concatenated along the sequence
        axis before one joint ``scaled_dot_product_attention``. Both halves must
        therefore shard together: a rank that owned all heads of the text stream
        and a slice of the image stream could not concatenate them.

        Sharding every attention and both ``gelu-approximate`` FFNs covers
        65.1% of the transformer's 2.24B parameters -- 65.1% of a dual-attention
        block, 66.7% of a plain joint block, 65.2% of the ``context_pre_only``
        tail. What is left replicated is the AdaLN modulation (33% of a plain
        block: ``norm1.linear`` and ``norm1_context.linear``), which emits
        shift/scale/gate vectors of the full hidden width applied to activations
        this plan keeps replicated; sharding its output would require sequence
        parallelism, not this plan. It is also *packed* -- ``1536 x 9`` outputs
        on a dual-attention block, ``1536 x 6`` elsewhere, chunked by the caller
        -- so naive ``Shard(0)`` would hand rank 0 whole leading chunks instead
        of a slice of each, the same objection MiniMax-H3's fused SwiGLU raises.
        ``pos_embed``,
        ``time_text_embed``, ``context_embedder``, ``norm_out`` and ``proj_out``
        sit outside the repeated stack and are left alone for the same reason as
        the other adapters: small, and one collective per step each.

        ``pos_embed.proj`` is also the one ``nn.Conv2d`` in any denoiser this
        framework shards, and torch's sharding rule for
        ``aten.convolution_backward`` marks ``grad_weight`` ``Partial(sum)``
        whatever the input placement, so a replicated convolution's gradient
        materializes ``tp_size`` times too large. It is frozen under the LoRA
        modes this adapter is used in (it is not a target module) and measures
        exactly ``tp_size x`` when unfrozen; see
        ``.scratch/tp_probe/sd3_qwen/probe_conv_replicate.py``. A full finetune
        of SD3.5 under tensor parallelism needs that fixed in shared code first.

        ``attn2`` exists only on the ``dual_attention_layers`` (0-12 of 24 in
        SD3.5 medium); the glob simply matches fewer blocks on the rest, and the
        final block drops ``to_add_out``/``ff_context`` because it is
        ``context_pre_only``.

        ``qk_norm="rms_norm"`` gives per-head ``RMSNorm(64)`` weights shared by
        every head, so all six qk-norm entries need the sharded-input style.
        Their input is laid out ``[batch, heads, seq, head_dim]`` -- the
        processor transposes before normalizing -- so the head axis is 1, not
        the 2 the Qwen processors leave it on. The reduction a replicated weight
        gets wrong is the same either way, but declaring axis 2 here would claim
        a global sequence length ``tp_size`` times the real one.
        """
        return {
            # Joint attention, image stream.
            "transformer_blocks.*.attn.to_q": ColwiseParallel(),
            "transformer_blocks.*.attn.to_k": ColwiseParallel(),
            "transformer_blocks.*.attn.to_v": ColwiseParallel(),
            "transformer_blocks.*.attn.to_out.0": RowwiseParallel(),
            # Joint attention, text stream: same heads, same joint softmax.
            "transformer_blocks.*.attn.add_q_proj": ColwiseParallel(),
            "transformer_blocks.*.attn.add_k_proj": ColwiseParallel(),
            "transformer_blocks.*.attn.add_v_proj": ColwiseParallel(),
            "transformer_blocks.*.attn.to_add_out": RowwiseParallel(),
            # SD3.5's second, image-only attention on the dual-attention layers.
            "transformer_blocks.*.attn2.to_q": ColwiseParallel(),
            "transformer_blocks.*.attn2.to_k": ColwiseParallel(),
            "transformer_blocks.*.attn2.to_v": ColwiseParallel(),
            "transformer_blocks.*.attn2.to_out.0": RowwiseParallel(),
            # `gelu-approximate` feed-forwards: `net.0.proj` is a plain
            # `[inner, dim]` up-projection and `net.2` the down-projection, so
            # neither is packed and both take the textbook styles.
            "transformer_blocks.*.ff.net.0.proj": ColwiseParallel(),
            "transformer_blocks.*.ff.net.2": RowwiseParallel(),
            "transformer_blocks.*.ff_context.net.0.proj": ColwiseParallel(),
            "transformer_blocks.*.ff_context.net.2": RowwiseParallel(),
            # Per-head RMSNorm(attention_head_dim): replicated weight fed a
            # head-sharded input, head axis 1 after the processor's transpose.
            "transformer_blocks.*.attn.norm_q": ShardedInputReplicatedWeight(head_axis=1),
            "transformer_blocks.*.attn.norm_k": ShardedInputReplicatedWeight(head_axis=1),
            "transformer_blocks.*.attn.norm_added_q": ShardedInputReplicatedWeight(head_axis=1),
            "transformer_blocks.*.attn.norm_added_k": ShardedInputReplicatedWeight(head_axis=1),
            "transformer_blocks.*.attn2.norm_q": ShardedInputReplicatedWeight(head_axis=1),
            "transformer_blocks.*.attn2.norm_k": ShardedInputReplicatedWeight(head_axis=1),
        }

    @property
    def tokenizer(self) -> Any:
        return self.pipeline.tokenizer_3

    # ============================ Output Image Encoding ============================
    def _output_geometry_multiple(self) -> int:
        """Match the spatial divisibility enforced by Diffusers SD3 inputs."""
        return self.pipeline.vae_scale_factor * self.pipeline.patch_size

    def _encode_output_images(
        self,
        pixel_values: torch.Tensor,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator],
    ) -> EncodedImageTensor:
        """Sample and normalize SD3.5 target latents on demand."""
        del condition
        latents = encode_shift_scale_vae_image(
            self,
            pixel_values,
            sample_mode="sample",
            generator=generator,
            source="SD3.5 target VAE encode",
        )
        return EncodedImageTensor(
            latents=latents,
            forward_context={},
            decode_context={},
        )

    # ============================ Encoding & Decoding ============================
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 3.5,
        max_sequence_length: Optional[int] = 512,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode the prompt(s) into embeddings using the pipeline's text encoder."""
        device = device if device is not None else self.device
        do_classifier_free_guidance = guidance_scale > 1.0
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
        )
        result = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # Token ids for downstream bookkeeping (used as `prompt_ids` in samples)
        result["prompt_ids"] = text_inputs.input_ids.to(device)

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = (
                    [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                )
                negative_prompt = negative_prompt * (
                    len(prompt) // len(negative_prompt)
                )  # Expand to match batch size
            assert len(prompt) == len(
                negative_prompt
            ), "The number of negative prompts must match the number of prompts."
            result["negative_prompt_embeds"] = negative_prompt_embeds
            result["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

            negative_text_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            result["negative_prompt_ids"] = negative_text_inputs.input_ids.to(device)

        return result

    def encode_image(self, images: Union[Image.Image, torch.Tensor, List[torch.Tensor]]):
        """Not needed for SD3 text-to-image models."""
        pass

    def encode_video(self, videos: Union[torch.Tensor, List[torch.Tensor]]):
        """Not needed for SD3 text-to-image models."""
        pass

    def decode_latents(
        self,
        latents: torch.Tensor,
        output_type: Literal["pil", "pt", "np"] = "pil",
    ) -> torch.Tensor:
        latents = latents.to(self.pipeline.vae.dtype)
        latents = (
            latents / self.pipeline.vae.config.scaling_factor
        ) + self.pipeline.vae.config.shift_factor

        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type=output_type)

        return images

    # ============================ Inference ============================
    @torch.no_grad()
    def inference(
        self,
        # Ordinary args
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        # Encoded Prompt
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # Encoded Negative Prompt
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        # Other args
        compute_log_prob: bool = True,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
    ) -> List[SD3_5Sample]:
        # 1. Setup
        device = self.device
        dtype = self.pipeline.transformer.dtype

        do_classifier_free_guidance = guidance_scale > 1.0
        has_negative_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        if do_classifier_free_guidance and not has_negative_prompt:
            logger.warning(
                "No negative prompt/embeds provided, classifier-free-guidance will be disabled."
            )
            do_classifier_free_guidance = False

        # 2. Encode prompt
        if prompt_embeds is None or pooled_prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt,
                negative_prompt,
                guidance_scale=guidance_scale,
                device=device,
            )
            prompt_embeds = encoded["prompt_embeds"]
            pooled_prompt_embeds = encoded["pooled_prompt_embeds"]
            prompt_ids = encoded["prompt_ids"]
            if do_classifier_free_guidance:
                negative_prompt_embeds = encoded["negative_prompt_embeds"]
                negative_prompt_ids = encoded["negative_prompt_ids"]
                negative_pooled_prompt_embeds = encoded["negative_pooled_prompt_embeds"]
        else:
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            if do_classifier_free_guidance:
                negative_prompt_embeds = negative_prompt_embeds.to(device)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)

        batch_size = len(prompt_embeds)
        num_channels_latents = self.pipeline.transformer.config.in_channels

        # 3. Prepare latent variables
        latents = self.pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
        )
        # latents : torch.Tensor of shape (B, C, H/8, W/8), not packed

        # 4. Prepare noise schedule
        image_seq_len = (latents.shape[2] // self.pipeline.transformer.config.patch_size) * (
            latents.shape[3] // self.pipeline.transformer.config.patch_size
        )
        timesteps = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=image_seq_len,
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
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                joint_attention_kwargs=joint_attention_kwargs,
                compute_log_prob=current_compute_log_prob,
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

        # 6. Decode latents
        images = self.decode_latents(latents=latents, output_type="pt")

        # 7. Create samples
        extra_call_back_res = callback_collector.get_result()  # (B, len(trajectory_indices), ...)
        callback_index_map = callback_collector.get_index_map()  # (T,) LongTensor
        all_latents = latent_collector.get_result()  # List[torch.Tensor(B, ...)]
        latent_index_map = latent_collector.get_index_map()  # (T+1,) LongTensor
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
        samples = [
            SD3_5Sample(
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
                prompt_embeds=prompt_embeds[b] if prompt_embeds is not None else None,
                pooled_prompt_embeds=(
                    pooled_prompt_embeds[b] if pooled_prompt_embeds is not None else None
                ),
                # Negative Prompt
                negative_prompt=negative_prompt[b] if negative_prompt is not None else None,
                negative_prompt_ids=(
                    negative_prompt_ids[b] if negative_prompt_ids is not None else None
                ),
                negative_prompt_embeds=(
                    negative_prompt_embeds[b] if negative_prompt_embeds is not None else None
                ),
                negative_pooled_prompt_embeds=(
                    negative_pooled_prompt_embeds[b]
                    if negative_pooled_prompt_embeds is not None
                    else None
                ),
                # Image & metadata
                height=height,
                width=width,
                image=images[b],
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

    # ============================ Training Forward ============================
    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        # Optional for CFG
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
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
        """Core forward pass for T2I generation.

        Args:
            t: Current timestep tensor.
            t_next: Next timestep tensor.
            latents: Current latent representations (B, C, H, W).
            prompt_embeds: Text prompt embeddings.
            pooled_prompt_embeds: Pooled text prompt embeddings.
            negative_prompt_embeds: Optional negative prompt embeddings (for CFG).
            negative_pooled_prompt_embeds: Optional negative pooled prompt embeddings.
            guidance_scale: CFG scale factor.
            next_latents: Optional target latents for log-prob computation.
            joint_attention_kwargs: Optional kwargs for attention layers.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            noise_level: Current noise level for SDE sampling.

        Returns:
            FlowMatchEulerDiscreteSDESchedulerOutput containing requested outputs.
        """
        # 1. Prepare variables
        batch_size = latents.shape[0]
        timestep = t.expand(batch_size).to(latents.dtype)

        # Auto-detect CFG
        if guidance_scale > 1.0 and (
            negative_prompt_embeds is None or negative_pooled_prompt_embeds is None
        ):
            logger.warning(
                "Passed `guidance_scale` > 1.0, but no `negative_prompt_embeds` or "
                "`negative_pooled_prompt_embeds` provided. Classifier-free guidance will be disabled."
            )
        do_classifier_free_guidance = (
            negative_prompt_embeds is not None
            and negative_pooled_prompt_embeds is not None
            and guidance_scale > 1.0
        )

        # 2. Prepare inputs for CFG
        if do_classifier_free_guidance:
            prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds_input = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
            latents_input = torch.cat([latents, latents], dim=0)
            timestep_input = timestep.repeat(2)
        else:
            prompt_embeds_input = prompt_embeds
            pooled_prompt_embeds_input = pooled_prompt_embeds
            latents_input = latents
            timestep_input = timestep

        # 3. Transformer forward pass
        velocity = self.transformer(
            hidden_states=latents_input,
            timestep=timestep_input,
            encoder_hidden_states=prompt_embeds_input,
            pooled_projections=pooled_prompt_embeds_input,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

        # 4. Apply CFG
        if do_classifier_free_guidance:
            velocity_uncond, velocity_text = velocity.chunk(2)
            velocity = velocity_uncond + guidance_scale * (velocity_text - velocity_uncond)

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
