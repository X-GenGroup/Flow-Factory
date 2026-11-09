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

# src/flow_factory/models/z_image/z_image.py
from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from PIL import Image
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)

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
from ..parallelism.context import ContextParallelCapability, ContextParallelLayout
from ..parallelism.sequence import (
    ColumnParallelRegion,
    SequenceBroadcastOutput,
    SequenceBroadcastParameter,
    SequenceParallelRegion,
)
from ..parallelism.tensor import ShardedInputReplicatedWeight, SharedInputColwiseParallel
from ..pipeline_contracts import image_output_contract

logger = setup_logger(__name__)


class ZImageContextParallelLayout(ContextParallelLayout):
    UNIFIED_IMAGE_TEXT = "z_image_unified_image_text"


@dataclass
class ZImageSample(T2ISample):
    # Class var
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    # Obj var - no extra


class ZImageAdapter(ConfiguredImageOutputAdapterMixin, BaseAdapter):
    """Adapt Z-Image for online generation and offline image targets."""

    supports_sharded_target_materialization = True
    supports_checkpoint_slice_loading = True
    tensor_parallel_head_config = ("transformer", ("n_heads", "n_kv_heads"))

    context_parallel_capability = ContextParallelCapability(
        layout=ZImageContextParallelLayout.UNIFIED_IMAGE_TEXT, tensor_parallel=True
    )

    @classmethod
    def validate_context_parallel_module(cls, module, degree, tp_size=1):
        from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor

        from diffusers import ZImageTransformer2DModel

        if not isinstance(module, ZImageTransformer2DModel):
            raise ValueError(
                f"Z-Image CP requires ZImageTransformer2DModel, received {type(module).__name__}"
            )
        cls._validate_context_parallel_config(module.config, degree, tp_size)
        if not module.layers or not module.noise_refiner or not module.context_refiner:
            raise ValueError("Z-Image CP requires the native image, text and unified stacks")
        for name, child in module.named_modules():
            if (
                hasattr(child, "processor")
                and type(child.processor) is not ZSingleStreamAttnProcessor
            ):
                raise ValueError(f"Z-Image CP requires native attention at {name}")

    @staticmethod
    def _validate_context_parallel_config(config, degree, tp_size=1):
        heads = config["n_heads"]
        if heads != config["n_kv_heads"] or heads % tp_size:
            raise ValueError(
                f"Z-Image CP requires equal Q/KV heads divisible by TP; heads={heads}, kv_heads={config['n_kv_heads']}, tp_size={tp_size}"
            )
        if list(config["all_patch_size"]) != [2] or list(config["all_f_patch_size"]) != [1]:
            raise ValueError(
                "Z-Image CP requires the native T2I patch_size=2, f_patch_size=1 layout"
            )
        if config["n_layers"] < 1 or config["n_refiner_layers"] < 1:
            raise ValueError("Z-Image CP requires nonempty image, text and unified stacks")

    @classmethod
    def configure_context_parallel_module(cls, module):
        from ._context_parallel import ZImageContextAttnProcessor

        for block in module.layers:
            block.attention.set_processor(ZImageContextAttnProcessor())

    @classmethod
    def install_context_parallel_module(cls, module, config):
        from ._context_parallel import install_z_image_context_parallel

        install_z_image_context_parallel(module, config)

    @classmethod
    def compose_context_parallel_sequence_boundaries(cls, module):
        from ._context_parallel import compose_z_image_sequence_boundaries

        return compose_z_image_sequence_boundaries(module)

    offline_training_forward_overrides = MappingProxyType(
        {
            "guidance_scale": 0.0,
            "cfg_normalization": False,
            "cfg_truncation": 1.0,
        }
    )
    pipeline_io_contract = image_output_contract(
        negative_prompt=NegativePromptPolicy.OPTIONAL,
    )

    # Z-Image trains and serves its diffusion transformer in FP32.
    component_load_dtype_defaults = {"transformer": torch.float32}

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: ZImagePipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

    def load_pipeline(self) -> ZImagePipeline:
        if self.config.parallelism_args.effective_context_parallel_size > 1:
            from diffusers import ZImageTransformer2DModel

            config = ZImageTransformer2DModel.load_config(
                self.model_args.model_name_or_path, subfolder="transformer"
            )
            self._validate_context_parallel_config(
                config,
                self.config.parallelism_args.effective_context_parallel_size,
                self.config.parallelism_args.effective_tensor_parallel_size,
            )
        return self._load_diffusers_pipeline(
            ZImagePipeline, self.model_args.model_name_or_path, low_cpu_mem_usage=False
        )

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Z-Image transformer."""
        return [
            "attention.to_k",
            "attention.to_q",
            "attention.to_v",
            "attention.to_out.0",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
        ]

    @classmethod
    def tensor_parallel_plan(cls) -> Dict[str, Any]:
        """Shard attention and feed-forwards in all three native block stacks.

        ``layers``, ``noise_refiner`` and ``context_refiner`` share the same block
        layout. Each attention and separate-projection SwiGLU is a
        ``ColumnParallelRegion``: Q/K/V share one input-gradient reduction, as do
        ``w1`` and ``w3``. ``FeedForward.forward`` names that input ``x``.

        Per-head Q/K RMSNorm weights remain replicated and sum their head-partial
        gradients through ``ShardedInputReplicatedWeight``. Modulation and the
        entry/exit embeddings and projections retain replicated parameters.
        The SP overlay partitions both refiners, the main stack and the final
        image projection, restoring full streams at native assembly boundaries.

        The root's bare ``x_pad_token`` and ``cap_pad_token`` parameters receive
        replicated completion and local views. Their native ``torch.where`` calls
        therefore use plain tensors while checkpoints and optimizers retain the
        original parameter names and DTensor ownership.

        The TP degree must divide the configured Q/KV head counts. For the native
        30-head checkpoint, 2 is the only nontrivial power-of-two divisor; this is
        not a general ``tp_size <= 2`` rule for arbitrary world sizes. CP has its
        own family geometry and transport checks.
        """
        stacks = ("layers", "noise_refiner", "context_refiner")
        leaves: Dict[str, Any] = {
            "attention": ColumnParallelRegion(inputs=("hidden_states",)),
            "attention.to_q": SharedInputColwiseParallel(),
            "attention.to_k": SharedInputColwiseParallel(),
            "attention.to_v": SharedInputColwiseParallel(),
            "attention.to_out.0": RowwiseParallel(),
            "feed_forward": ColumnParallelRegion(inputs=("x",)),
            "feed_forward.w1": SharedInputColwiseParallel(),
            "feed_forward.w3": SharedInputColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(),
            # Per-head RMSNorm over attention_head_dim: replicated weight fed a
            # head-sharded input, so its gradient is a partial sum.
            "attention.norm_q": ShardedInputReplicatedWeight(),
            "attention.norm_k": ShardedInputReplicatedWeight(),
        }
        return {f"{stack}.*.{leaf}": style for stack in stacks for leaf, style in leaves.items()}

    @classmethod
    def sequence_parallel_plan(cls) -> Dict[str, Any]:
        """Shard all three native token streams, gathering before joint assembly.

        Noise and caption refiners have independent native padding, masks and
        rotary tables. Their residual streams stay token-local, while attention
        and SwiGLU gather inputs and reduce-scatter projected outputs. The last
        refiner gathers before ``_build_unified_sequence`` uses global lengths.

        Four affine norms per block sum token-partial weight derivatives. Each
        modulated block reduces its own broadcast conditioning derivative; the
        shared timestep embedder then receives complete contributions exactly
        once. Final image projection gathers before native unpatchification.

        The adapter's flat T2I path is supported. Omni per-token modulation and
        ControlNet residuals require separate token-boundary recipes. Every
        native-padded stream must be divisible by the TP degree.
        """
        from ._sequence_parallel import ZImageRefinerSequenceParallel

        leaves = {
            "attention": SequenceParallelRegion(gather=("hidden_states",)),
            "feed_forward": SequenceParallelRegion(gather=("x",)),
            "attention.to_out.0": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "attention_norm1": SequenceBroadcastParameter(),
            "attention_norm2": SequenceBroadcastParameter(),
            "ffn_norm1": SequenceBroadcastParameter(),
            "ffn_norm2": SequenceBroadcastParameter(),
        }
        return {
            **{
                f"{stack}.*.{leaf}": style
                for stack in ("layers", "noise_refiner", "context_refiner")
                for leaf, style in leaves.items()
            },
            "noise_refiner": ZImageRefinerSequenceParallel(),
            "context_refiner": ZImageRefinerSequenceParallel(),
            "noise_refiner.*.adaLN_modulation": SequenceBroadcastOutput(),
            "layers.0": PrepareModuleInput(
                input_layouts=(Replicate(), None, None, None, None, None, None),
                desired_input_layouts=(Shard(1), None, None, None, None, None, None),
                use_local_output=True,
            ),
            "layers.*.adaLN_modulation": SequenceBroadcastOutput(),
            "all_final_layer.*.adaLN_modulation": SequenceBroadcastOutput(),
            "all_final_layer.*.linear": SequenceBroadcastParameter(),
            "all_final_layer.*": PrepareModuleOutput(
                output_layouts=Shard(1),
                desired_output_layouts=Replicate(),
                use_local_output=True,
            ),
        }

    # ======================== Encoding / Decoding ========================
    # ----------------------- Prompt Encoding -----------------------
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
    ) -> Tuple[List[torch.FloatTensor], torch.Tensor]:
        device = device or self.text_encoder.device

        # Chat formatting is an encoder input, not the caller's reward/replay
        # identity. Never mutate a list that inference will put into samples.
        prompt = [prompt] if isinstance(prompt, str) else list(prompt)
        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.pipeline.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list, text_input_ids

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        max_sequence_length: int = 512,
    ) -> Dict[str, Union[List[torch.FloatTensor], torch.Tensor]]:
        device = device or self.text_encoder.device
        do_classifier_free_guidance = guidance_scale > 0.0
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds, prompt_ids = self._encode_prompt(
            prompt=prompt,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        results = {
            "prompt_embeds": prompt_embeds,
            "prompt_ids": prompt_ids,
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
            negative_prompt_embeds, negative_prompt_ids = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                max_sequence_length=max_sequence_length,
            )
            results.update(
                {
                    "negative_prompt_embeds": negative_prompt_embeds,
                    "negative_prompt_ids": negative_prompt_ids,
                }
            )

        return results

    # ----------------------- Image / Video Encoding & Decoding -----------------------
    def encode_image(
        self,
        images: Union[Image.Image, torch.Tensor, List[torch.Tensor]],
    ):
        """Not needed for Z-Image models."""
        pass

    def encode_video(
        self,
        videos: Union[torch.Tensor, List[torch.Tensor]],
    ):
        """Not needed for Z-Image models."""
        pass

    def _output_geometry_multiple(self) -> int:
        """Require the exact spatial grid enforced by the official pipeline."""
        return self.pipeline.vae_scale_factor * 2

    def _encode_output_images(
        self,
        pixel_values: torch.Tensor,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator],
    ) -> EncodedImageTensor:
        """Sample and normalize Z-Image target latents on demand."""
        del condition
        latents = encode_shift_scale_vae_image(
            self,
            pixel_values,
            sample_mode="sample",
            generator=generator,
            source="Z-Image target VAE encode",
        )
        return EncodedImageTensor(
            latents=latents,
            forward_context={},
            decode_context={},
        )

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

    # ======================== Inference ========================

    @torch.no_grad()
    def inference(
        self,
        # Generation parameters
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        height: int = 1024,
        width: int = 1024,
        # Prompt
        prompt: Union[str, List[str]] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        # Negative prompt
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        # CFG options
        cfg_normalization: Union[bool, float] = False,
        cfg_truncation: Optional[float] = 1.0,
        # Other parameters
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        max_sequence_length: int = 512,
        compute_log_prob: bool = True,
        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
    ):
        """Generate images from text prompts using the Z-Image model."""

        # 1. Setup
        device = self.device
        dtype = self.pipeline.transformer.dtype
        do_classifier_free_guidance = guidance_scale > 0.0

        # 2. Encode prompts if not provided
        if prompt_embeds is None:
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                guidance_scale=guidance_scale,
                device=device,
            )
            prompt_ids = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
            negative_prompt_ids = (
                encoded["negative_prompt_ids"] if do_classifier_free_guidance else None
            )
            negative_prompt_embeds = (
                encoded["negative_prompt_embeds"] if do_classifier_free_guidance else None
            )
        else:
            prompt_embeds = [pe.to(device) for pe in prompt_embeds]
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = [npe.to(device) for npe in negative_prompt_embeds]

        batch_size = len(prompt_embeds)
        num_channels_latents = self.pipeline.transformer.in_channels

        # 3. Prepare latents
        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        # 4. Set scheduler timesteps
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        timesteps = set_scheduler_timesteps(
            self.scheduler,
            num_inference_steps,
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
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                cfg_normalization=cfg_normalization,
                cfg_truncation=cfg_truncation,
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

        # 6. Decode latents to images
        images = self.decode_latents(latents, output_type="pt")

        # 7. Create samples
        extra_call_back_res = callback_collector.get_result()  # (B, len(trajectory_indices), ...)
        callback_index_map = callback_collector.get_index_map()  # (T,) LongTensor
        all_latents = latent_collector.get_result()  # List[torch.Tensor(B, ...)]
        latent_index_map = latent_collector.get_index_map()  # (T+1,) LongTensor
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
        samples = [
            ZImageSample(
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
                # Generated image & metadata
                height=height,
                width=width,
                image=images[b],
                # Encoded prompt
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b] if prompt_embeds is not None else None,
                # Encoded negative prompt
                negative_prompt=negative_prompt[b] if negative_prompt is not None else None,
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
        prompt_embeds: List[torch.FloatTensor],
        # Optional for CFG
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        guidance_scale: float = 5.0,
        cfg_normalization: Union[bool, float] = False,
        cfg_truncation: Optional[float] = 1.0,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        noise_level: Optional[float] = None,
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
            prompt_embeds: List of text prompt embeddings (ragged).
            negative_prompt_embeds: Optional list of negative prompt embeddings.
            guidance_scale: CFG scale factor.
            cfg_normalization: Cap the guided velocity norm at this multiple of the conditional
                velocity norm; ``False`` or ``0`` disables the cap and ``True`` means ``1.0``.
            cfg_truncation: CFG truncation threshold.
            next_latents: Optional target latents for log-prob computation.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            noise_level: Current noise level for SDE sampling.

        Returns:
            FlowMatchEulerDiscreteSDESchedulerOutput containing requested outputs.
        """
        # 1. Prepare variables
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        # Move each tensor in the prompt_embeds lists to the correct device
        prompt_embeds = [pe.to(device) for pe in prompt_embeds]
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = [npe.to(device) for npe in negative_prompt_embeds]

        # Z-Image uses reversed timesteps
        timestep = t.expand(batch_size).to(latents.dtype)
        t_reversed = (1000 - timestep) / 1000
        t_norm = t_reversed[0].item()

        # Auto-detect CFG
        if guidance_scale > 0.0 and negative_prompt_embeds is None:
            logger.warning(
                "Passed `guidance_scale` > 0.0, but no `negative_prompt_embeds` provided. "
                "Classifier-free guidance will be disabled."
            )
        do_classifier_free_guidance = negative_prompt_embeds is not None and guidance_scale > 0.0

        # 2. Determine if CFG should be applied at this timestep
        if (
            do_classifier_free_guidance
            and cfg_truncation is not None
            and float(cfg_truncation) <= 1
            and t_norm > cfg_truncation
        ):
            current_guidance_scale = 0.0
        else:
            current_guidance_scale = guidance_scale

        apply_cfg = do_classifier_free_guidance and current_guidance_scale > 0

        # 3. Prepare inputs
        if apply_cfg:
            latents_typed = latents.to(self.pipeline.transformer.dtype)
            latent_model_input = latents_typed.repeat(2, 1, 1, 1)
            prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds  # List concatenation
            timestep_model_input = t_reversed.repeat(2)
        else:
            latent_model_input = latents.to(self.pipeline.transformer.dtype)
            prompt_embeds_model_input = prompt_embeds
            timestep_model_input = t_reversed

        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        # 4. Transformer forward pass
        model_out_list = self.transformer(
            latent_model_input_list,
            timestep_model_input,
            prompt_embeds_model_input,
            return_dict=False,
        )[0]

        # 5. Apply CFG
        if apply_cfg:
            pos_out = model_out_list[:batch_size]
            neg_out = model_out_list[batch_size:]
            velocity = []

            for j in range(batch_size):
                pos = pos_out[j].float()
                neg = neg_out[j].float()
                pred = pos + current_guidance_scale * (pos - neg)

                # CFG normalization
                if cfg_normalization and float(cfg_normalization) > 0.0:
                    ori_pos_norm = torch.linalg.vector_norm(pos)
                    new_pos_norm = torch.linalg.vector_norm(pred)
                    max_new_norm = ori_pos_norm * float(cfg_normalization)
                    if new_pos_norm > max_new_norm:
                        pred = pred * (max_new_norm / new_pos_norm)

                velocity.append(pred)

            velocity = torch.stack(velocity, dim=0)
        else:
            velocity = torch.stack([out.float() for out in model_out_list], dim=0)

        velocity = velocity.squeeze(2)
        velocity = -velocity  # Z-Image specific: negate velocity prediction

        # 6. Scheduler step
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
