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

"""Pre-Task-4B LTX2 rollout oracle.

An independent transcription of the legacy inference loop, kept apart from the
adapters so the parity test compares the new collectors against the algorithm
they replaced rather than against themselves. It only calls the public
``forward``/``decode_latents`` surface plus the same collectors the legacy loop
used, so the dispatch order, the numerics and the RNG stream are the legacy ones.
"""

from typing import Any, Dict, List, Optional

import torch

from flow_factory.scheduler import set_scheduler_timesteps
from flow_factory.scheduler.flow_match_euler_discrete import calculate_shift
from flow_factory.utils.trajectory_collector import (
    create_callback_collector,
    create_trajectory_collector,
)


@torch.no_grad()
def run_legacy_rollout(
    adapter: Any,
    *,
    conditioned: bool,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    guidance_scale: float,
    prompt: List[str],
    prompt_ids: torch.Tensor,
    connector_prompt_embeds: torch.Tensor,
    connector_audio_prompt_embeds: torch.Tensor,
    connector_attention_mask: torch.Tensor,
    compute_log_prob: bool,
    trajectory_indices: Any,
    extra_call_back_kwargs: List[str],
    generator: Optional[torch.Generator] = None,
    condition_images: Optional[torch.Tensor] = None,
    noise_scale: float = 0.0,
    decode_timestep: float = 0.0,
    decode_noise_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the legacy LTX2 rollout and return every observable it produced."""
    device = adapter.device
    check_kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "connector_prompt_embeds": connector_prompt_embeds,
        "guidance_scale": guidance_scale,
    }
    if conditioned:
        check_kwargs["condition_images"] = condition_images
    num_frames = adapter._check_inputs(height, width, num_frames, **check_kwargs)

    batch_size = connector_prompt_embeds.shape[0]
    pipeline = adapter.pipeline
    latent_h = height // pipeline.vae_spatial_compression_ratio
    latent_w = width // pipeline.vae_spatial_compression_ratio
    latent_f = (num_frames - 1) // pipeline.vae_temporal_compression_ratio + 1

    duration_s = num_frames / frame_rate
    audio_num_frames = round(
        duration_s
        * pipeline.audio_sampling_rate
        / pipeline.audio_hop_length
        / pipeline.audio_vae_temporal_compression_ratio
    )
    num_mel_bins = pipeline.audio_vae.config.mel_bins

    conditioning_mask = None
    prepare_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_channels_latents": adapter.transformer_config.in_channels,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "noise_scale": noise_scale,
        "dtype": torch.float32,
        "device": device,
        "generator": generator,
    }
    if conditioned:
        video_latents, conditioning_mask = pipeline.prepare_latents(
            image=2.0 * condition_images.to(device=device, dtype=torch.float32) - 1.0,
            **prepare_kwargs,
        )
    else:
        video_latents = pipeline.prepare_latents(**prepare_kwargs)
    audio_latents = pipeline.prepare_audio_latents(
        batch_size=batch_size,
        num_channels_latents=pipeline.audio_vae.config.latent_channels,
        audio_latent_length=audio_num_frames,
        num_mel_bins=num_mel_bins,
        noise_scale=noise_scale,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    mu = calculate_shift(
        latent_f * latent_h * latent_w,
        adapter.scheduler.config.get("base_image_seq_len", 1024),
        adapter.scheduler.config.get("max_image_seq_len", 4096),
        adapter.scheduler.config.get("base_shift", 0.95),
        adapter.scheduler.config.get("max_shift", 2.05),
    )
    timesteps = set_scheduler_timesteps(
        adapter.scheduler, num_inference_steps, device=device, sigmas=None, mu=mu
    )
    set_scheduler_timesteps(
        adapter.audio_scheduler, num_inference_steps, device=device, sigmas=None, mu=mu
    )

    video_coords = pipeline.transformer.rope.prepare_video_coords(
        batch_size, latent_f, latent_h, latent_w, device, fps=frame_rate
    )
    audio_coords = pipeline.transformer.audio_rope.prepare_audio_coords(
        batch_size, audio_num_frames, device
    )

    video_seq_len = video_latents.shape[1]
    latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
    latents = adapter.cast_latents(torch.cat([video_latents, audio_latents], dim=1))
    latent_collector.collect(latents, step_idx=0)
    log_prob_collector = (
        create_trajectory_collector(trajectory_indices, num_inference_steps)
        if compute_log_prob
        else None
    )
    callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

    for step_index, timestep in enumerate(timesteps):
        noise_level = adapter.scheduler.get_noise_level_for_timestep(timestep)
        next_timestep = (
            timesteps[step_index + 1]
            if step_index + 1 < len(timesteps)
            else torch.tensor(0, device=device)
        )
        step_log_prob = compute_log_prob and noise_level > 0
        forward_kwargs: Dict[str, Any] = {
            "t": timestep,
            "t_next": next_timestep,
            "latents": latents,
            "video_seq_len": video_seq_len,
            "connector_prompt_embeds": connector_prompt_embeds,
            "connector_audio_prompt_embeds": connector_audio_prompt_embeds,
            "connector_attention_mask": connector_attention_mask,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "audio_num_frames": audio_num_frames,
            "video_coords": video_coords,
            "audio_coords": audio_coords,
            "noise_level": noise_level,
            "compute_log_prob": step_log_prob,
            "return_kwargs": list(
                set(["next_latents", "log_prob", "velocity"] + extra_call_back_kwargs)
            ),
        }
        if conditioned:
            forward_kwargs["conditioning_mask"] = conditioning_mask
        output = adapter.forward(**forward_kwargs)

        latents = adapter.cast_latents(output.next_latents)
        latent_collector.collect(latents, step_index + 1)
        if step_log_prob:
            log_prob_collector.collect(output.log_prob, step_index)
        callback_collector.collect_step(
            step_idx=step_index,
            output=output,
            keys=extra_call_back_kwargs,
            capturable={"noise_level": noise_level},
        )

    video, audio = adapter.decode_latents(
        latents[:, :video_seq_len],
        latents[:, video_seq_len:],
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        decode_timestep=decode_timestep,
        decode_noise_scale=decode_noise_scale,
        output_type="pt",
        generator=generator,
    )

    return {
        "final_latents": latents,
        "video": video,
        "audio": audio,
        "timesteps": timesteps,
        "video_seq_len": video_seq_len,
        "conditioning_mask": conditioning_mask,
        "duration_s": duration_s,
        "collected_states": latent_collector.get_result(),
        "state_index_map": latent_collector.get_index_map(),
        "log_probs": None if log_prob_collector is None else log_prob_collector.get_result(),
        "log_prob_index_map": (
            None if log_prob_collector is None else log_prob_collector.get_index_map()
        ),
        "callbacks": callback_collector.get_result(),
        "callback_index_map": callback_collector.get_index_map(),
    }
