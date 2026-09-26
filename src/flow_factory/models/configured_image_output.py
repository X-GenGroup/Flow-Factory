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

"""Shared boundary mechanics for configured-resolution image output codecs.

This module deliberately does not implement one universal VAE encoding recipe.
Diffusers image families use different posterior normalization, latent ranks,
packing layouts, and position metadata.  The codec owns only the common decoded
media and geometry boundary; each adapter still implements its exact official
tensor conversion in ``_encode_output_images``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar, Literal, Optional, Tuple

import torch
from PIL import Image

from ..contracts import GeometrySource, MediaType
from ..samples import LatentState
from ..utils.image import require_decoded_rgb_image, require_finite_bchw_image
from .output_state import (
    DecodedMediaBatch,
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
    OutputStateCodec,
)


@dataclass(frozen=True, slots=True)
class EncodedImageTensor:
    """Return one adapter-specific target tensor and its model/decode context."""

    latents: torch.Tensor
    forward_context: Mapping[str, Any]
    decode_context: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ConfiguredImageOutputCodec:
    """Encode one configured-resolution target image per sample on demand."""

    adapter: Any
    required_components: ClassVar[Tuple[str, ...]] = ("vae",)

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        """Validate, preprocess, and delegate family-specific target encoding.

        Args:
            media_batch: Decoded single-image candidate for every sample.
            condition: Model condition associated with the same samples.
            generator: Optional generator accepted by adapter-specific encoders.

        Returns:
            Detached clean latent state and its forward/decode context.
        """
        height, width = self.adapter._configured_output_geometry()
        images = self._extract_images(media_batch)
        pixel_values = self.adapter._preprocess_output_images(images, height, width)
        self._validate_pixel_values(pixel_values, len(images), height, width)

        vae = self.adapter.vae
        vae_dtype = getattr(vae, "dtype", None)
        if not isinstance(vae_dtype, torch.dtype) or not vae_dtype.is_floating_point:
            raise TypeError(
                f"{type(self.adapter).__name__} output codec expected VAE to expose a "
                f"floating dtype, received {vae_dtype!r}"
            )
        pixel_values = pixel_values.to(device=self.adapter.device, dtype=vae_dtype)
        encoded = self.adapter._encode_output_images(
            pixel_values,
            condition,
            generator,
        )
        if not isinstance(encoded, EncodedImageTensor):
            raise TypeError(
                f"{type(self.adapter).__name__}._encode_output_images must return "
                f"EncodedImageTensor, received {type(encoded).__name__}"
            )
        if not isinstance(encoded.latents, torch.Tensor):
            raise TypeError(
                f"{type(self.adapter).__name__} output image latents must be torch.Tensor, "
                f"received {type(encoded.latents).__name__}"
            )

        decode_context = dict(encoded.decode_context)
        for name, value in (("height", height), ("width", width)):
            existing = decode_context.get(name, value)
            if type(existing) is not int or existing != value:
                raise ValueError(
                    f"{type(self.adapter).__name__} output codec decode_context {name!r} "
                    f"must equal configured value {value}, received {existing!r}"
                )
            decode_context[name] = value

        signature = GeometrySignature(
            media=(
                MediaGeometrySignature(
                    type=MediaType.IMAGE,
                    height=height,
                    width=width,
                ),
            )
        )
        return EncodedOutputState(
            clean_state=LatentState({"latent": encoded.latents}),
            forward_context=encoded.forward_context,
            decode_context=decode_context,
            geometry_signatures=tuple(signature for _ in images),
        )

    @staticmethod
    def _extract_images(media_batch: DecodedMediaBatch) -> list[Image.Image]:
        """Extract the exact single PIL image owned by every output sample."""
        images: list[Image.Image] = []
        for sample_index, candidate in enumerate(media_batch):
            if len(candidate) != 1:
                raise ValueError(
                    "configured image output codec expected one image per sample, "
                    f"received {len(candidate)} for sample {sample_index}"
                )
            payload = candidate[0].payload
            images.append(
                require_decoded_rgb_image(
                    payload,
                    source=f"configured image output codec sample {sample_index}",
                )
            )
        return images

    @staticmethod
    def _validate_pixel_values(
        pixel_values: object,
        batch_size: int,
        height: int,
        width: int,
    ) -> None:
        """Require the common image processor tensor boundary."""
        require_finite_bchw_image(
            pixel_values,
            source="image_processor.preprocess",
            batch_size=batch_size,
            height=height,
            width=width,
        )


class ConfiguredImageOutputAdapterMixin:
    """Supply the common codec lifecycle for configured-resolution image models."""

    def build_output_state_codec(self) -> OutputStateCodec:
        """Build an on-the-fly image codec after validating static declarations.

        Returns:
            Configured-resolution output codec bound to this adapter.

        Raises:
            TypeError: If no pipeline contract is declared.
            ValueError: If output media or geometry ownership is incompatible.
        """
        contract = self.effective_pipeline_io_contract
        if contract is None:
            raise TypeError(
                f"adapter {type(self).__name__} must declare pipeline_io_contract before "
                "building a configured image output codec"
            )
        if contract.geometry_source is not GeometrySource.CONFIGURED:
            raise ValueError(
                f"adapter {type(self).__name__} configured image codec requires "
                f"geometry_source='configured', received {contract.geometry_source.value!r}"
            )
        output_types = tuple(item.type for item in contract.output_media.items)
        if output_types != (MediaType.IMAGE,):
            raise ValueError(
                f"adapter {type(self).__name__} configured image codec requires exactly one "
                f"image output, received {output_types}"
            )
        self._configured_output_geometry()
        return ConfiguredImageOutputCodec(self)

    def _configured_output_geometry(self) -> Tuple[int, int]:
        """Return positive configured H/W aligned to the adapter's latent grid."""
        geometry = []
        for name in ("height", "width"):
            value = getattr(self.training_args, name, None)
            if type(value) is not int:
                raise TypeError(
                    f"{type(self).__name__} output geometry expected training_args.{name} "
                    f"to be int, received {type(value).__name__}: {value!r}"
                )
            if value <= 0:
                raise ValueError(
                    f"{type(self).__name__} output geometry expected training_args.{name} > 0, "
                    f"received {value}"
                )
            geometry.append(value)

        multiple = self._output_geometry_multiple()
        if type(multiple) is not int or multiple <= 0:
            raise ValueError(
                f"{type(self).__name__}._output_geometry_multiple must return a positive int, "
                f"received {multiple!r}"
            )
        for name, value in zip(("height", "width"), geometry):
            if value % multiple:
                raise ValueError(
                    f"{type(self).__name__} output geometry expected training_args.{name} "
                    f"to be divisible by {multiple}, received {value}"
                )
        return geometry[0], geometry[1]

    def _output_geometry_multiple(self) -> int:
        """Return the exact pixel-grid multiple imposed by VAE packing."""
        return 1

    def _preprocess_output_images(
        self,
        images: list[Image.Image],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Run the pipeline-owned image processor at configured geometry."""
        return self.pipeline.image_processor.preprocess(
            images,
            height=height,
            width=width,
        )

    def _validate_encoded_output_geometry(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Prove codec signatures and decode geometry match configured H/W."""
        height, width = self._configured_output_geometry()
        expected_signature = GeometrySignature(
            media=(
                MediaGeometrySignature(
                    type=MediaType.IMAGE,
                    height=height,
                    width=width,
                ),
            )
        )
        if len(encoded.geometry_signatures) != len(media_batch):
            raise ValueError(
                f"{type(self).__name__} expected one output geometry signature per target "
                f"sample, received {len(encoded.geometry_signatures)} for {len(media_batch)}"
            )
        for sample_index, signature in enumerate(encoded.geometry_signatures):
            if signature != expected_signature:
                raise ValueError(
                    f"{type(self).__name__} encoded output geometry disagrees with configured "
                    f"height/width {(height, width)} for sample {sample_index}: {signature!r}"
                )
        for name, value in (("height", height), ("width", width)):
            if encoded.decode_context.get(name) != value:
                raise ValueError(
                    f"{type(self).__name__} decode_context {name!r} must equal configured "
                    f"value {value}, received {encoded.decode_context.get(name)!r}"
                )
        self._validate_output_image_context(media_batch, condition, encoded)

    def _validate_output_image_context(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        encoded: EncodedOutputState,
    ) -> None:
        """Allow an adapter to validate family-specific IDs or shape metadata."""
        del media_batch, condition, encoded


VAELatentSampleMode = Literal["sample", "argmax"]


def retrieve_vae_latents(
    encoder_output: Any,
    *,
    sample_mode: VAELatentSampleMode,
    generator: Optional[torch.Generator] = None,
    source: str,
) -> torch.Tensor:
    """Select sampled or argmax latents from one VAE encoder output.

    Args:
        encoder_output: VAE output exposing direct latents or a posterior distribution.
        sample_mode: Explicit posterior selection matching the official pipeline role.
        generator: Generator forwarded unchanged to posterior sampling.
        source: Model-specific identifier included in validation errors.

    Returns:
        Selected latent tensor.

    Raises:
        TypeError: If the selection or encoder surface is invalid.
    """
    if type(sample_mode) is not str:
        raise TypeError(
            f"{source} expected sample_mode to be str, "
            f"received {type(sample_mode).__name__}: {sample_mode!r}"
        )
    if sample_mode not in ("sample", "argmax"):
        raise ValueError(
            f"{source} expected sample_mode in ('sample', 'argmax'), " f"received {sample_mode!r}"
        )
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError(
            f"{source} expected generator to be torch.Generator or None, "
            f"received {type(generator).__name__}: {generator!r}"
        )

    direct_latents = getattr(encoder_output, "latents", None)
    if isinstance(direct_latents, torch.Tensor):
        return direct_latents
    latent_dist = getattr(encoder_output, "latent_dist", None)
    if latent_dist is None and isinstance(encoder_output, (tuple, list)):
        if len(encoder_output) != 1:
            raise TypeError(
                f"{source} expected a single VAE encoder output, received {len(encoder_output)}"
            )
        first_output = encoder_output[0]
        if isinstance(first_output, torch.Tensor):
            return first_output
        direct_latents = getattr(first_output, "latents", None)
        if isinstance(direct_latents, torch.Tensor):
            return direct_latents
        latent_dist = getattr(first_output, "latent_dist", first_output)
    if latent_dist is None and (
        getattr(encoder_output, "sample", None) is not None
        or getattr(encoder_output, "mode", None) is not None
    ):
        latent_dist = encoder_output

    if sample_mode == "sample":
        sample = getattr(latent_dist, "sample", None)
        if not callable(sample):
            raise TypeError(f"{source} expected VAE posterior with callable sample()")
        latents = sample(generator=generator)
    else:
        mode = getattr(latent_dist, "mode", None)
        latents = mode() if callable(mode) else mode
    if not isinstance(latents, torch.Tensor):
        raise TypeError(
            f"{source} expected VAE posterior {sample_mode!r} result to be torch.Tensor, "
            f"received {type(latents).__name__}"
        )
    return latents


def encode_shift_scale_vae_image(
    adapter: Any,
    pixel_values: torch.Tensor,
    *,
    sample_mode: VAELatentSampleMode,
    generator: Optional[torch.Generator] = None,
    source: Optional[str] = None,
) -> torch.Tensor:
    """Apply explicit posterior selection and Diffusers shift/scale normalization.

    This is deliberately role-neutral: an adapter may call it for an input
    condition or for an offline target. The orchestration layer still owns cache
    policy, geometry, packing, and model-specific forward metadata.

    Args:
        adapter: Adapter exposing the canonical VAE.
        pixel_values: Preprocessed BCHW pixels on the VAE device and dtype.
        sample_mode: Explicit posterior selection for the calling pipeline role.
        generator: Generator forwarded unchanged when ``sample_mode='sample'``.
        source: Optional model-specific identifier included in validation errors.

    Returns:
        Shift/scale-normalized VAE latents.
    """
    source = source or f"{type(adapter).__name__} VAE encode"
    vae = adapter.vae
    latents = retrieve_vae_latents(
        vae.encode(pixel_values),
        sample_mode=sample_mode,
        generator=generator,
        source=source,
    )
    shift_factor, scaling_factor = _shift_scale_factors(vae, source=source)
    return (latents - shift_factor) * scaling_factor


def _shift_scale_factors(vae: Any, *, source: str) -> Tuple[float, float]:
    """Validate the scalar latent normalization declared by a Diffusers VAE."""
    config = getattr(vae, "config", None)
    shift_factor = getattr(config, "shift_factor", None)
    scaling_factor = getattr(config, "scaling_factor", None)
    for name, value in (
        ("shift_factor", shift_factor),
        ("scaling_factor", scaling_factor),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"{source} expected numeric VAE config {name}, "
                f"received {type(value).__name__}: {value!r}"
            )
        if not math.isfinite(float(value)):
            raise ValueError(f"{source} expected finite VAE config {name}, received {value!r}")
    if scaling_factor <= 0:
        raise ValueError(
            f"{source} expected VAE config scaling_factor > 0, received {scaling_factor!r}"
        )
    return float(shift_factor), float(scaling_factor)


__all__ = [
    "ConfiguredImageOutputAdapterMixin",
    "ConfiguredImageOutputCodec",
    "EncodedImageTensor",
    "VAELatentSampleMode",
    "encode_shift_scale_vae_image",
    "retrieve_vae_latents",
]
