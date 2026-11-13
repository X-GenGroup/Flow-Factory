# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decode target-only MiniMax H3 packed rows with pinned frozen blocks."""

from contextlib import contextmanager, nullcontext
from typing import Any, Mapping, Tuple

import torch

from ...samples import LatentState
from ._common import validate_target_state
from .blocks import run_h3_blocks
from .dependency import require_minimax_h3_support


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration as error:
            raise ValueError(
                f"expected decoder module {type(module).__name__} to expose a parameter "
                "or buffer device"
            ) from error


@contextmanager
def _decoder_execution_device(pipeline: Any):
    vae = getattr(pipeline, "vae", None)
    audio_vae = getattr(pipeline, "audio_vae", None)
    if not isinstance(vae, torch.nn.Module) or not isinstance(audio_vae, torch.nn.Module):
        # Lightweight block-contract fakes own no actual decoder modules.
        with nullcontext():
            yield
        return
    device = _module_device(vae)
    audio_device = _module_device(audio_vae)
    if audio_device != device:
        raise ValueError(
            f"MiniMax H3 decoder devices must match, received video={device}, "
            f"audio={audio_device}"
        )
    components = getattr(pipeline, "components", None)
    if not isinstance(components, Mapping):
        raise TypeError(
            f"MiniMax H3 decoder expected pipeline.components mapping, "
            f"received {type(components).__name__}"
        )
    hidden = {}
    for name, component in components.items():
        if isinstance(component, torch.nn.Module) and _module_device(component) != device:
            hidden[name] = component
            setattr(pipeline, name, None)
    try:
        execution_device = pipeline._execution_device
        if execution_device != device:
            raise ValueError(
                f"MiniMax H3 decoder expected execution device {device}, "
                f"received {execution_device}"
            )
        yield
    finally:
        for name, component in hidden.items():
            setattr(pipeline, name, component)


def decode_h3_targets(
    pipeline: Any,
    target_state: LatentState,
    geometry: Mapping[str, int],
    *,
    output_type: str = "pil",
    workflow: str = "t2va",
    input_gradients: bool = False,
) -> Tuple[Any, torch.Tensor, int]:
    """Decode B=1 target rows into video and stereo audio.

    Args:
        pipeline: Adapter-owned pipeline with materialized frozen VAEs.
        target_state: Generated target-only packed rows.
        geometry: Cached latent geometry.
        output_type: Upstream video output type.
        workflow: Workflow identifier used in diagnostics.
        input_gradients: Retain decoder input derivatives for reward optimization.

    Returns:
        Video output, stereo audio tensor, and sample rate.
    """
    validate_target_state(target_state)
    batch_size = target_state.components["video"].shape[0]
    if batch_size != 1:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} decode requires B=1, received B={batch_size}"
        )
    required = ("num_latent_frames", "latent_height", "latent_width", "num_audio_latents")
    missing = tuple(field for field in required if field not in geometry)
    if missing:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} decode geometry missing fields={missing}"
        )
    symbols = require_minimax_h3_support()
    values = {
        **geometry,
        "latents": target_state.components["video"].squeeze(0),
        "audio_latents": target_state.components["audio"].squeeze(0),
        "num_condition_video_rows": 0,
        "num_condition_audio_rows": 0,
        "output_type": output_type,
    }
    with _decoder_execution_device(pipeline):
        outputs = run_h3_blocks(
            pipeline,
            [
                symbols.AfterDenoiseStep(),
                symbols.VideoDecodeStep(),
                symbols.AudioDecodeStep(),
            ],
            values,
            requested_outputs=("videos", "audio", "sampling_rate"),
            workflow=workflow,
            input_gradients=input_gradients,
        )
    audio = outputs["audio"]
    sampling_rate = outputs["sampling_rate"]
    if not isinstance(audio, torch.Tensor) or audio.ndim != 3 or audio.shape[:2] != (1, 2):
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} decoded audio expected shape (1,2,samples), "
            f"received {getattr(audio, 'shape', None)}"
        )
    if not isinstance(sampling_rate, int) or isinstance(sampling_rate, bool) or sampling_rate <= 0:
        raise ValueError(
            f"MiniMax H3 workflow={workflow!r} sampling_rate expected positive int, "
            f"received {sampling_rate!r}"
        )
    return outputs["videos"], audio, sampling_rate
