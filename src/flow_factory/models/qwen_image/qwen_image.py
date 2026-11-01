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

# src/flow_factory/models/qwen_image/qwen_image.py
from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from PIL import Image
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.nn.utils.rnn import pad_sequence

import diffusers
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

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
from ...utils.imports import is_version_at_least
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
from ..pipeline_contracts import image_output_contract
from ..tensor_parallel import ShardedInputReplicatedWeight
from ._output import encode_qwen_output_images
from ._utils import _pad_seq_dim

logger = setup_logger(__name__)


@dataclass
class QwenImageSample(T2ISample):
    """Output class for Qwen-Image models."""

    # Class var
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    # Obj var
    prompt_embeds_mask: Optional[torch.FloatTensor] = None
    negative_prompt_embeds_mask: Optional[torch.FloatTensor] = None
    img_shapes: Optional[List[Tuple[int, int, int]]] = None


class QwenImageAdapter(ConfiguredImageOutputAdapterMixin, BaseAdapter):
    """Adapter for Qwen-Image text-to-image models."""

    # Qwen-Image runs with guidance=None, so the transformer's guidance embedder
    # receives no gradient and DDP must scan for unused parameters.
    ddp_find_unused_parameters = True
    supports_diffusers_cache = True
    pipeline_io_contract = image_output_contract(
        negative_prompt=NegativePromptPolicy.OPTIONAL,
    )

    def __init__(self, config: Arguments, accelerator: Accelerator):
        if not is_version_at_least("diffusers", "0.37.0"):
            raise ImportError(
                "QwenImageAdapter requires diffusers>=0.37.0 (the Qwen-Image "
                "transformer derives the text sequence length from "
                "encoder_hidden_states_mask; txt_seq_lens is no longer passed). "
                f"Found diffusers {diffusers.__version__}. "
                "Upgrade with `pip install -U 'diffusers>=0.37.0'`."
            )
        super().__init__(config, accelerator)
        self.pipeline: QwenImagePipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

        self._warned_cfg_no_neg_prompt = False
        self._warned_no_cfg = False

    def load_pipeline(self) -> QwenImagePipeline:
        return self._load_diffusers_pipeline(
            QwenImagePipeline, self.model_args.model_name_or_path, low_cpu_mem_usage=False
        )

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA target modules for Qwen-Image transformer.

        Upstream's training script trains attention only; the feed-forwards are
        Flow-Factory's addition. Both halves of each one have to be named:
        ``net.0.proj`` is the ``GELU`` wrapper's projection, but ``net.2`` *is*
        the down-projection ``Linear`` and owns no ``.proj`` child -- the
        ``net.2.proj`` this replaced matched nothing, so the adapter silently
        covered the up-projections only. PEFT reports no error for a target that
        reaches zero modules. ``QwenImageEditPlusAdapter`` already named it
        correctly.
        """
        return [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",  # Image Stream / Main Stream
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",  # Text Stream
            # FFNs
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
            "txt_mlp.net.0.proj",
            "txt_mlp.net.2",
        ]

    @classmethod
    def tensor_parallel_plan(cls) -> Dict[str, Any]:
        """Shard both streams of every double-stream block; keep AdaLN replicated.

        Mirrors the ``_tp_plan`` upstream diffusers declares for
        ``QwenImageTransformer2DModel`` and adds the four qk-norms upstream
        omits because inference never takes their gradient. Nothing this plan
        names is fused: Q/K/V are three Linears per stream and the
        ``FeedForward`` is ``gelu-approximate`` (``net.0.proj`` up, ``net.2``
        down, no gate half packed in), so every entry is a textbook
        colwise/rowwise.

        The image and text streams share one ``Attention`` and one joint softmax
        over the concatenated ``[text, image]`` sequence, so ``to_{q,k,v}`` and
        ``add_{q,k,v}_proj`` must shard on the same head split -- the
        concatenation is along sequence, not heads.

        That covers 66.7% of a block and 66.5% of the 20.4B-parameter
        transformer. The remaining third of a block is the two AdaLN modulators
        ``img_mod.1``/``txt_mod.1``, 16.7% each: both emit six full-width
        shift/scale/gate vectors applied to activations this plan leaves
        replicated, which would need sequence parallelism to shard -- and both
        are packed ``[3072 x 6]`` weights the block then ``chunk``s, so naive
        ``Shard(0)`` would hand rank 0 whole leading chunks rather than a slice
        of each. ``img_in``/``txt_in``/``txt_norm``, ``time_text_embed``,
        ``norm_out`` and ``proj_out`` sit outside the repeated stack and cost a
        collective per step for parameters that are a rounding error at 60
        layers.

        The final block's text stream is dead -- the model carries only
        ``hidden_states`` past the stack -- so its ``to_add_out`` and
        ``txt_mlp`` are sharded but never receive a gradient. That is already
        why this adapter sets ``ddp_find_unused_parameters``; it is noted here
        because it means a sharded module having no ``.grad`` is expected rather
        than a sign the plan missed.

        Qwen-Image is 24 heads of 128, so tp=2/4/8 all divide the head count.
        ``qk_norm="rms_norm"`` makes all four qk-norms per-head
        ``RMSNorm(128)``: one replicated weight shared across heads, fed an input
        the column-parallel projections have already head-sharded, which is
        exactly the case :class:`ShardedInputReplicatedWeight` exists for. The
        processor reshapes to ``[batch, seq, heads, head_dim]`` without a
        transpose, so the head axis is the style's default 2.

        One caveat the plan cannot express on its own:
        ``QwenDoubleStreamAttnProcessor2_0`` infers the head *count* from the
        local projection width (``unflatten(-1, (-1, head_dim))``) but reads
        ``head_dim`` as ``attn.inner_dim // attn.heads``, and
        ``localize_attention_head_counts`` rewrites ``heads`` while leaving
        ``inner_dim`` whole -- so ``head_dim`` comes out ``tp_size`` times too
        large. It does not corrupt silently: the per-head qk-norm's declared
        head axis makes the DTensor multiply raise ("expected shape should be
        broadcastable to [1, 1024, 12, 256]" against a ``[128]`` weight).
        Localizing ``inner_dim`` alongside ``heads`` restores exact parity; see
        ``.scratch/tp_probe/sd3_qwen/gpu_qwen_image_check.py``.
        """
        return {
            # Image stream.
            "transformer_blocks.*.attn.to_q": ColwiseParallel(),
            "transformer_blocks.*.attn.to_k": ColwiseParallel(),
            "transformer_blocks.*.attn.to_v": ColwiseParallel(),
            "transformer_blocks.*.attn.to_out.0": RowwiseParallel(),
            # Text stream: same heads, same joint softmax.
            "transformer_blocks.*.attn.add_q_proj": ColwiseParallel(),
            "transformer_blocks.*.attn.add_k_proj": ColwiseParallel(),
            "transformer_blocks.*.attn.add_v_proj": ColwiseParallel(),
            "transformer_blocks.*.attn.to_add_out": RowwiseParallel(),
            # Unfused `gelu-approximate` feed-forwards, one per stream.
            "transformer_blocks.*.img_mlp.net.0.proj": ColwiseParallel(),
            "transformer_blocks.*.img_mlp.net.2": RowwiseParallel(),
            "transformer_blocks.*.txt_mlp.net.0.proj": ColwiseParallel(),
            "transformer_blocks.*.txt_mlp.net.2": RowwiseParallel(),
            # Per-head RMSNorm(attention_head_dim): replicated weight, sharded
            # input, so a plainly replicated style would leave a partial sum.
            "transformer_blocks.*.attn.norm_q": ShardedInputReplicatedWeight(),
            "transformer_blocks.*.attn.norm_k": ShardedInputReplicatedWeight(),
            "transformer_blocks.*.attn.norm_added_q": ShardedInputReplicatedWeight(),
            "transformer_blocks.*.attn.norm_added_k": ShardedInputReplicatedWeight(),
        }

    # ======================== Encoding & Decoding ========================

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype

        max_sequence_length = max_sequence_length or self.pipeline.tokenizer_max_length

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.pipeline.prompt_template_encode
        drop_idx = self.pipeline.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.pipeline.tokenizer(
            txt,
            max_length=max_sequence_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        input_ids = txt_tokens.input_ids
        encoder_hidden_states = self.text_encoder(
            input_ids=input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self.pipeline._extract_masked_hidden(
            hidden_states, txt_tokens.attention_mask
        )
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states
        ]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
                for u in split_hidden_states
            ]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        input_ids = input_ids[:, drop_idx:]  # Extract only user input ids
        return input_ids, prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 4.0,
        max_sequence_length: int = 1024,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, torch.Tensor]]:
        """Encode text prompts using the pipeline's text encoder."""

        device = device or self.pipeline.text_encoder.device
        dtype = dtype or self.pipeline.text_encoder.dtype
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Encode positive prompt
        prompt_ids, prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
            prompt=prompt, device=device, dtype=dtype, max_sequence_length=max_sequence_length
        )
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        results = {
            "prompt_ids": prompt_ids,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
        }
        # Encode negative prompt
        if do_classifier_free_guidance:
            negative_prompt = "" if negative_prompt is None else negative_prompt
            negative_prompt = (
                [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            )
            negative_prompt = negative_prompt * (
                len(prompt) // len(negative_prompt)
            )  # Expand to match batch size
            assert len(negative_prompt) == len(
                prompt
            ), "The number of negative prompts must match the number of prompts."
            negative_prompt_ids, negative_prompt_embeds, negative_prompt_embeds_mask = (
                self._get_qwen_prompt_embeds(
                    prompt=negative_prompt,
                    device=device,
                    dtype=dtype,
                    max_sequence_length=max_sequence_length,
                )
            )
            results.update(
                {
                    "negative_prompt_ids": negative_prompt_ids,
                    "negative_prompt_embeds": negative_prompt_embeds[:, :max_sequence_length],
                    "negative_prompt_embeds_mask": negative_prompt_embeds_mask[
                        :, :max_sequence_length
                    ],
                }
            )

        return results

    def encode_image(
        self,
        image: Union[Image.Image, torch.Tensor, List[torch.Tensor]],
    ) -> None:
        """Not needed for Qwen-Image text-to-image models."""
        return None

    def encode_video(self, video: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        """Not needed for Qwen-Image text-to-image models."""
        return None

    def _output_geometry_multiple(self) -> int:
        """Require the VAE grid and 2x2 latent packing used by Diffusers."""
        return self.pipeline.vae_scale_factor * 2

    def _encode_output_images(
        self,
        pixel_values: torch.Tensor,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator],
    ) -> EncodedImageTensor:
        """Sample target images through Qwen's five-dimensional VAE path."""
        return encode_qwen_output_images(self, pixel_values, condition, generator)

    def decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: Literal["pil", "pt", "np"] = "pil",
    ) -> List[Image.Image]:
        """Decode latents to images using VAE."""

        latents = self.pipeline._unpack_latents(
            latents, height, width, self.pipeline.vae_scale_factor
        )
        latents = latents.to(self.pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(
            1, self.pipeline.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        images = self.pipeline.vae.decode(latents, return_dict=False)[0][:, :, 0]
        images = self.pipeline.image_processor.postprocess(images, output_type=output_type)

        return images

    # ======================== Padding Utilities ========================
    def _standardize_data(
        self,
        data: Union[None, torch.Tensor, List[torch.Tensor]],
        padding_value: Union[int, float],
        device: Optional[torch.device] = None,
        max_len: Optional[int] = None,
    ):
        if data is None:
            return None

        # If data is a list (ragged), pad it into a batch tensor first
        if isinstance(data, list):
            # Ensure data is on the correct device before padding
            if len(data) > 0 and data[0].device != device:
                data = [t.to(device) for t in data]
            data = pad_sequence(data, batch_first=True, padding_value=padding_value)
        else:
            data = data.to(device)

        return data[:, :max_len] if data.shape[1] > max_len else data

    def _pad_batch_prompt(
        self,
        prompt_embeds_mask: Union[List[torch.Tensor], torch.Tensor],
        prompt_embeds: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        prompt_ids: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ):
        if isinstance(prompt_embeds_mask, list):
            device = device or prompt_embeds_mask[0].device
            max_pos_len = max(1, int(max(mask.sum() for mask in prompt_embeds_mask)))
        else:
            device = device or prompt_embeds_mask.device
            max_pos_len = max(1, int(prompt_embeds_mask.sum(dim=1).max()))

        if prompt_ids is not None:
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            padded_prompt_ids = self._standardize_data(
                prompt_ids,
                padding_value=pad_token_id,
                device=device,
                max_len=max_pos_len,
            )
        else:
            padded_prompt_ids = None

        if prompt_embeds is not None:
            padded_prompt_embeds = self._standardize_data(
                prompt_embeds,
                padding_value=0.0,
                device=device,
                max_len=max_pos_len,
            )
        else:
            padded_prompt_embeds = None

        padded_prompt_embeds_mask = self._standardize_data(
            prompt_embeds_mask,
            padding_value=0,
            device=device,
            max_len=max_pos_len,
        )
        return padded_prompt_embeds_mask, padded_prompt_embeds, padded_prompt_ids

    # ======================== Inference ========================
    @torch.no_grad()
    def inference(
        self,
        # Ordinary arguments
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,  # Same as `true_cfg_scale` in QwenImagePipeline.
        height: int = 1024,
        width: int = 1024,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # Prompt encoding arguments
        prompt_ids: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        prompt_embeds: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        prompt_embeds_mask: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        negative_prompt_ids: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        negative_prompt_embeds: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        negative_prompt_embeds_mask: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        # Other arguments
        attention_kwargs: Optional[Dict[str, Any]] = {},
        max_sequence_length: int = 1024,
        compute_log_prob: bool = False,
        # Extra callback arguments
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
    ):
        # 1. Prepare inputs
        device = self.device
        dtype = self.pipeline.transformer.dtype
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        if guidance_scale > 1 and not has_neg_prompt and not self._warned_cfg_no_neg_prompt:
            self._warned_cfg_no_neg_prompt = True
            logger.warning(
                f"guidance_scale is passed as {guidance_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided. Warning will only be shown once."
            )
        elif guidance_scale <= 1 and has_neg_prompt and not self._warned_no_cfg:
            self._warned_no_cfg = True
            logger.warning(
                " negative_prompt is passed but classifier-free guidance is not enabled since guidance_scale <= 1. Warning will only be shown once."
            )

        # 2. Get prompt embeddings
        if (prompt is not None and (prompt_embeds is None or prompt_embeds_mask is None)) or (
            negative_prompt is not None
            and (negative_prompt_embeds is None or negative_prompt_embeds_mask is None)
        ):
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            prompt_ids = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
            prompt_embeds_mask = encoded["prompt_embeds_mask"]
            negative_prompt_ids = encoded.get("negative_prompt_ids", None)
            negative_prompt_embeds = encoded.get("negative_prompt_embeds", None)
            negative_prompt_embeds_mask = encoded.get("negative_prompt_embeds_mask", None)
        else:
            prompt_embeds = (
                [p.to(device) for p in prompt_embeds]
                if isinstance(prompt_embeds, list)
                else prompt_embeds.to(device)
            )
            prompt_embeds_mask = (
                [p.to(device) for p in prompt_embeds_mask]
                if isinstance(prompt_embeds_mask, list)
                else prompt_embeds_mask.to(device)
            )
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = (
                    [p.to(device) for p in negative_prompt_embeds]
                    if isinstance(negative_prompt_embeds, list)
                    else negative_prompt_embeds.to(device)
                )
            if negative_prompt_embeds_mask is not None:
                negative_prompt_embeds_mask = (
                    [p.to(device) for p in negative_prompt_embeds_mask]
                    if isinstance(negative_prompt_embeds_mask, list)
                    else negative_prompt_embeds_mask.to(device)
                )

        # 3. Prepare latents
        batch_size = len(prompt_embeds)

        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        img_shapes = [
            [
                (
                    1,
                    height // self.pipeline.vae_scale_factor // 2,
                    width // self.pipeline.vae_scale_factor // 2,
                )
            ]
        ] * batch_size

        # 4. Set scheduler timesteps
        timesteps = set_scheduler_timesteps(
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            seq_len=latents.shape[1],
            device=device,
        )

        guidance = None  # Always None for Qwen-Image

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
                prompt_embeds_mask=prompt_embeds_mask,
                img_shapes=img_shapes,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                guidance_scale=guidance_scale,
                attention_kwargs=attention_kwargs,
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
        decoded_images = self.decode_latents(latents, height, width, output_type="pt")

        # 7. Prepare output samples
        extra_call_back_res = callback_collector.get_result()  # (B, len(trajectory_indices), ...)
        callback_index_map = callback_collector.get_index_map()  # (T,) LongTensor
        all_latents = latent_collector.get_result()  # List[torch.Tensor(B, ...)]
        latent_index_map = latent_collector.get_index_map()  # (T+1,) LongTensor
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
        samples = [
            QwenImageSample(
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
                image=decoded_images[b],
                img_shapes=img_shapes[b],
                # Prompt info
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b],
                prompt_embeds=prompt_embeds[b],
                prompt_embeds_mask=prompt_embeds_mask[b],
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
                negative_prompt_embeds_mask=(
                    negative_prompt_embeds_mask[b]
                    if negative_prompt_embeds_mask is not None
                    else None
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
        prompt_embeds_mask: torch.Tensor,
        img_shapes: List[List[Tuple[int, int, int]]],
        # Optional for CFG
        negative_prompt_embeds: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        negative_prompt_embeds_mask: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        guidance_scale: float = 4.0,
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
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """Core forward pass for T2I generation.

        Args:
            t: Current timestep tensor.
            t_next: Next timestep tensor.
            latents: Current latent representations (B, seq_len, C).
            prompt_embeds: Text prompt embeddings.
            prompt_embeds_mask: Attention mask for prompt embeddings.
            img_shapes: List of image shapes per sample.
            negative_prompt_embeds: Optional negative prompt embeddings (for CFG).
            negative_prompt_embeds_mask: Optional negative prompt attention mask.
            guidance_scale: CFG scale factor.
            next_latents: Optional target latents for log-prob computation.
            attention_kwargs: Optional kwargs for attention layers.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            noise_level: Current noise level for SDE sampling.

        Returns:
            FlowMatchEulerDiscreteSDESchedulerOutput containing requested outputs.
        """
        # 1. Prepare variables
        device = latents.device
        batch_size = latents.shape[0]
        timestep = t.expand(batch_size).to(latents.dtype)
        guidance = None  # Always None for Qwen-Image
        has_negative_prompt = (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        if guidance_scale > 1.0 and not has_negative_prompt:
            logger.warning(
                "Passed `guidance_scale` > 1.0, but no `negative_prompt_embeds` provided. "
                "Classifier-free guidance will be disabled."
            )
        do_true_cfg = guidance_scale > 1.0 and has_negative_prompt

        # Truncate prompt embeddings and masks to the max valid length in the
        # batch. diffusers (>=0.37) derives the per-sample text length from
        # encoder_hidden_states_mask, so the deprecated txt_seq_lens is not passed.
        prompt_embeds_mask, prompt_embeds, _ = self._pad_batch_prompt(
            prompt_embeds_mask=prompt_embeds_mask, prompt_embeds=prompt_embeds, device=device
        )

        if do_true_cfg:
            negative_prompt_embeds_mask, negative_prompt_embeds, _ = self._pad_batch_prompt(
                prompt_embeds_mask=negative_prompt_embeds_mask,
                prompt_embeds=negative_prompt_embeds,
                device=device,
            )

        # 2. Transformer forward pass
        if do_true_cfg:
            # Merge cond/uncond into one batched forward (halves transformer
            # calls). Pad both text streams to a common length; the
            # encoder_hidden_states_mask masks the extra positions and diffusers
            # derives each sample's length from it, so valid outputs match two
            # separate forwards. The merged cond/uncond batch shares one
            # cache_context. Tradeoff: ~2x peak activation memory vs two serial
            # forwards (lower batch/resolution if it OOMs).
            seq_len = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
            prompt_embeds = _pad_seq_dim(prompt_embeds, seq_len, 0.0)
            prompt_embeds_mask = _pad_seq_dim(prompt_embeds_mask, seq_len, 0)
            negative_prompt_embeds = _pad_seq_dim(negative_prompt_embeds, seq_len, 0.0)
            negative_prompt_embeds_mask = _pad_seq_dim(negative_prompt_embeds_mask, seq_len, 0)

            with self.pipeline.transformer.cache_context("cond_uncond"):
                both_pred = self.transformer(
                    hidden_states=torch.cat([latents, latents], dim=0),
                    timestep=torch.cat([timestep, timestep], dim=0) / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=torch.cat(
                        [prompt_embeds_mask, negative_prompt_embeds_mask], dim=0
                    ),
                    encoder_hidden_states=torch.cat([prompt_embeds, negative_prompt_embeds], dim=0),
                    img_shapes=img_shapes * 2,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            velocity, neg_velocity = both_pred.chunk(2, dim=0)

            comb_pred = neg_velocity + guidance_scale * (velocity - neg_velocity)

            # Rescale norm (Qwen-Image specific)
            cond_norm = torch.norm(velocity, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            velocity = comb_pred * (cond_norm / noise_norm)
        else:
            # Single conditional forward pass (no CFG).
            with self.pipeline.transformer.cache_context("cond"):
                velocity = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    attention_kwargs=attention_kwargs,
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
