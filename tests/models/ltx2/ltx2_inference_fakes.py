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

"""Inference-capable LTX2 fakes: real timestep schedules, latent preparation, decode.

Kept separate from ``ltx2_fakes`` because the pre-Task-4A golden generator imports
that module against an older source tree and must not see any newer names. Every
fake here stays deterministic and CPU-only; the only randomness comes from the
latent preparation draws, so a test can pin the RNG stream of a whole rollout.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
from ltx2_fakes import (
    AUDIO_SCHEDULER_OFFSET,
    AUDIO_SEQ_LEN,
    BATCH_SIZE,
    CHANNELS,
    FRAME_RATE,
    HEIGHT,
    NUM_FRAMES,
    TEXT_DIM,
    TEXT_SEQ_LEN,
    VIDEO_SCHEDULER_OFFSET,
    VIDEO_SEQ_LEN,
    WIDTH,
    PipelineFake,
    SchedulerFake,
    TransformerFake,
    conditioning_mask,
)

NUM_INFERENCE_STEPS = 3
NOISE_LEVEL = 0.7
SEED = 20260811
DECODED_HEIGHT = 4
DECODED_WIDTH = 6
DECODED_CHANNELS = 3
AUDIO_SAMPLE_RATE = 16384
MEL_BINS = 64
MEL_COMPRESSION = 4
AUDIO_LATENT_CHANNELS = 8


def _time_shift(mu: float, sigma: torch.Tensor) -> torch.Tensor:
    """Apply the flow-matching resolution shift diffusers uses for LTX2."""
    exp_mu = torch.exp(torch.tensor(mu, dtype=torch.float32))
    return exp_mu / (exp_mu + (1.0 / sigma - 1.0))


class InferenceSchedulerFake(SchedulerFake):
    """Scheduler twin with a real ``set_timesteps`` schedule and noise-level window.

    ``sigma == timestep / 1000`` holds exactly, as it does for the flow-matching
    scheduler, and the final transition leaves the SDE window so its noise level is
    zero. That makes the rollout store log probabilities sparsely, which the
    structured trajectory has to represent with the collector ``-1`` sentinel.
    """

    dynamics_type = "Flow-SDE"

    def __init__(
        self,
        offset: float,
        log: List[Tuple[str, Any]],
        *,
        noise_level: float = NOISE_LEVEL,
    ) -> None:
        super().__init__(offset, log)
        self.noise_level = noise_level
        self.config = {
            "base_image_seq_len": 1024,
            "max_image_seq_len": 4096,
            "base_shift": 0.95,
            "max_shift": 2.05,
        }
        self.timesteps = torch.zeros(0)
        self.sigmas = torch.zeros(1)
        self.set_timesteps_calls: List[Dict[str, Any]] = []

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
        sigmas: Optional[Any] = None,
        mu: Optional[float] = None,
        timesteps: Optional[Any] = None,
    ) -> None:
        """Install a shifted flow-matching schedule and record the dispatch."""
        self.log.append(("set_timesteps", self.offset))
        self.set_timesteps_calls.append({"num_inference_steps": num_inference_steps, "mu": mu})
        if sigmas is None:
            steps = num_inference_steps or NUM_INFERENCE_STEPS
            sigmas = torch.linspace(1.0, 1.0 / steps, steps)
        base = torch.as_tensor(sigmas, dtype=torch.float32)
        shifted = base if mu is None else _time_shift(mu, base)
        self.sigmas = torch.cat([shifted, torch.zeros(1)]).to(device)
        self.timesteps = (shifted * 1000.0).to(device)

    @property
    def train_timesteps(self) -> torch.Tensor:
        """Return the rollout positions inside the SDE window."""
        return torch.arange(0, max(len(self.timesteps) - 1, 0), dtype=torch.int64)

    def get_kl_divergence_denominator(
        self,
        std_dev_t: Optional[torch.Tensor],
        dt: Optional[torch.Tensor],
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Return the Euler-Maruyama transition variance of this twin's window."""
        if not isinstance(std_dev_t, torch.Tensor) or not isinstance(dt, torch.Tensor):
            raise ValueError(
                f"expected std_dev_t and dt tensors for dynamics_type={self.dynamics_type!r}, "
                f"received {type(std_dev_t).__name__} and {type(dt).__name__}"
            )
        return (std_dev_t.float() ** 2 * (-dt.float())).clamp_min(eps)

    def get_noise_level_for_timestep(self, timestep: torch.Tensor) -> float:
        """Return the SDE noise level, or zero for the final deterministic step."""
        matches = (self.timesteps == timestep).nonzero().flatten()
        if matches.numel() == 0:
            raise ValueError(
                f"expected timestep {float(timestep)} in the installed schedule "
                f"{self.timesteps.tolist()}"
            )
        index = int(matches[0].item())
        return self.noise_level if index in set(self.train_timesteps.tolist()) else 0.0


class VAEFake:
    """Video VAE stand-in returning a deterministic decoded pixel tensor."""

    dtype = torch.float32

    def __init__(self) -> None:
        # Timestep conditioning is on so the decode draws from the global RNG,
        # which lets a parity test pin the whole rollout + decode RNG stream.
        self.config = SimpleNamespace(scaling_factor=2.0, timestep_conditioning=True)
        self.latents_mean = torch.tensor(0.5)
        self.latents_std = torch.tensor(1.5)
        self.decode_calls: List[Tuple[int, ...]] = []

    def decode(
        self,
        latents: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Tuple[torch.Tensor]:
        """Return one deterministic ``[-1, 1]`` frame stack per sample."""
        self.decode_calls.append(tuple(latents.shape))
        batch_size = latents.shape[0]
        scale = latents.reshape(batch_size, -1).mean(dim=1).reshape(batch_size, 1, 1, 1, 1)
        frames = torch.arange(
            DECODED_CHANNELS * latents.shape[2] * DECODED_HEIGHT * DECODED_WIDTH,
            dtype=torch.float32,
        ).reshape(1, DECODED_CHANNELS, latents.shape[2], DECODED_HEIGHT, DECODED_WIDTH)
        return (torch.tanh(frames / frames.numel() * scale),)


class AudioVAEFake:
    """Audio VAE stand-in returning a deterministic mel spectrogram."""

    dtype = torch.float32

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            mel_bins=MEL_BINS,
            latent_channels=AUDIO_LATENT_CHANNELS,
        )
        self.latents_mean = torch.tensor(-0.25)
        self.latents_std = torch.tensor(2.0)

    def decode(self, latents: torch.Tensor, return_dict: bool = False) -> Tuple[torch.Tensor]:
        """Return a mel spectrogram proportional to the audio latents."""
        return (latents * 3.0,)


class VocoderFake:
    """Vocoder stand-in mapping a mel spectrogram to a deterministic waveform."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(output_sampling_rate=AUDIO_SAMPLE_RATE)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Return one waveform per sample."""
        return mel.reshape(mel.shape[0], -1) * 0.5


class VideoProcessorFake:
    """Video post-processor returning the ``(B, F, C, H, W)`` layout LTX2 expects."""

    def postprocess_video(self, video: torch.Tensor, output_type: str = "pt") -> torch.Tensor:
        """Denormalize decoded ``(B, C, F, H, W)`` pixels to frame-major ``[0, 1]``."""
        return (video * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 1, 3, 4)


class InferencePipelineFake(PipelineFake):
    """LTX2 pipeline stand-in covering latent preparation, geometry, and decode."""

    audio_vae_mel_compression_ratio = MEL_COMPRESSION
    audio_sampling_rate = AUDIO_SAMPLE_RATE

    def __init__(
        self,
        scheduler: InferenceSchedulerFake,
        transformer: TransformerFake,
        *,
        conditioned: bool,
    ) -> None:
        super().__init__(scheduler, transformer)
        self.conditioned = conditioned
        self.vae = VAEFake()
        self.audio_vae = AudioVAEFake()
        self.vocoder = VocoderFake()
        self.video_processor = VideoProcessorFake()
        self.freed_hooks = 0

    def prepare_latents(
        self,
        *,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        noise_scale: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        image: Optional[torch.Tensor] = None,
    ) -> Any:
        """Draw packed video latents, plus the I2AV conditioning mask when conditioned."""
        latents = torch.randn(
            (batch_size, VIDEO_SEQ_LEN, num_channels_latents),
            generator=generator,
            dtype=dtype,
            device=device,
        )
        if not self.conditioned:
            return latents
        if image is None:
            raise ValueError("expected a conditioning image for the conditioned pipeline fake")
        return latents, conditioning_mask().to(device=device, dtype=dtype)

    def prepare_audio_latents(
        self,
        *,
        batch_size: int,
        num_channels_latents: int,
        audio_latent_length: int,
        num_mel_bins: int,
        noise_scale: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw packed audio latents sharing the video channel count."""
        return torch.randn(
            (batch_size, audio_latent_length, CHANNELS),
            generator=generator,
            dtype=dtype,
            device=device,
        )

    def _denormalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        scaling_factor: float,
    ) -> torch.Tensor:
        """Undo the video latent normalization."""
        return latents * latents_std / scaling_factor + latents_mean

    def _denormalize_audio_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        """Undo the audio latent normalization."""
        return latents * latents_std + latents_mean

    def _unpack_audio_latents(
        self, latents: torch.Tensor, audio_num_frames: int, num_mel_bins: int
    ) -> torch.Tensor:
        """Reshape packed audio latents to a mel-major layout."""
        batch_size = latents.shape[0]
        return latents.reshape(batch_size, audio_num_frames, -1)

    def maybe_free_model_hooks(self) -> None:
        """Record the pipeline teardown the inference loop performs."""
        self.freed_hooks += 1


def condition_images() -> torch.Tensor:
    """Return the deterministic I2AV conditioning image batch."""
    return torch.full((BATCH_SIZE, DECODED_CHANNELS, DECODED_HEIGHT, DECODED_WIDTH), 0.25)


def inference_adapter(cls: type) -> Tuple[Any, List[Tuple[str, Any]]]:
    """Build an LTX2 adapter wired to the inference fakes, plus its dispatch log."""
    log: List[Tuple[str, Any]] = []
    transformer = TransformerFake()
    adapter = object.__new__(cls)
    adapter.pipeline = InferencePipelineFake(
        InferenceSchedulerFake(VIDEO_SCHEDULER_OFFSET, log),
        transformer,
        conditioned=cls.__name__ == "LTX2_I2AV_Adapter",
    )
    adapter.component_runtime = SimpleNamespace(
        get_component=lambda name: transformer,
        get_canonical_component=lambda name: SimpleNamespace(
            config=SimpleNamespace(in_channels=CHANNELS)
        ),
    )
    adapter.load_scheduler = lambda: InferenceSchedulerFake(AUDIO_SCHEDULER_OFFSET, log)
    adapter.scheduler_group = adapter.build_scheduler_group()
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.training_args = SimpleNamespace(latent_storage_dtype=None)
    return adapter, log


def inference_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Return the shared pre-encoded inference arguments for both LTX2 adapters."""
    kwargs: Dict[str, Any] = {
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "frame_rate": FRAME_RATE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": 1.0,
        "prompt": ["a", "b"],
        "prompt_ids": torch.arange(BATCH_SIZE * 2).reshape(BATCH_SIZE, 2),
        "connector_prompt_embeds": torch.zeros(BATCH_SIZE, TEXT_SEQ_LEN, TEXT_DIM),
        "connector_audio_prompt_embeds": torch.zeros(BATCH_SIZE, TEXT_SEQ_LEN, TEXT_DIM),
        "connector_attention_mask": torch.ones(BATCH_SIZE, TEXT_SEQ_LEN),
        "compute_log_prob": True,
        "trajectory_indices": "all",
        "extra_call_back_kwargs": ["next_latents_mean", "std_dev_t", "noise_level", "velocity"],
    }
    kwargs.update(overrides)
    return kwargs


def expected_audio_seq_len() -> int:
    """Return the packed audio sequence length the fake geometry produces."""
    return AUDIO_SEQ_LEN
