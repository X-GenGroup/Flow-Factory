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

"""Bagel target-image encoding without optional attention dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar, Literal, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...contracts import MediaType
from ...samples import LatentState
from ...utils.image import require_decoded_rgb_image
from ..output_state import (
    DecodedMediaBatch,
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
)
from .data.data_utils import pil_img2rgb

BagelPosteriorMode = Literal["sample", "argmax"]


@dataclass(frozen=True, slots=True)
class BagelOutputStateCodec:
    """Encode target images through Bagel's custom VAE and packed token layout."""

    adapter: Any
    required_components: ClassVar[Tuple[str, ...]] = ("bagel", "vae")

    def encode_output_state(
        self,
        media_batch: DecodedMediaBatch,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> EncodedOutputState:
        """Transform, sample, normalize, and patchify target images on demand.

        Args:
            media_batch: One decoded target image per sample.
            condition: Input condition paired with the targets. Bagel output encoding
                does not derive target state from the condition.
            generator: Optional generator forwarded to posterior sampling.

        Returns:
            Packed clean target state with output-derived image geometry.
        """
        del condition
        transformed_images, image_shape = _transform_target_images(
            self.adapter,
            media_batch,
        )
        vae = self.adapter.vae
        pixel_values = torch.stack(transformed_images).to(
            device=self.adapter.device,
            dtype=_module_dtype(vae),
        )
        latents = encode_bagel_vae_image(
            vae,
            pixel_values,
            posterior_mode="sample",
            generator=generator,
        )

        bagel = self.adapter.get_component("bagel")
        layout = resolve_bagel_latent_layout(bagel)
        packed_latents = pack_bagel_latents(
            latents,
            image_shape=image_shape,
            layout=layout,
        )

        signature = GeometrySignature(
            media=(
                MediaGeometrySignature(
                    type=MediaType.IMAGE,
                    height=image_shape[0],
                    width=image_shape[1],
                ),
            )
        )
        context = {"image_shape": image_shape}
        return EncodedOutputState(
            clean_state=LatentState({"latent": packed_latents}),
            forward_context=context,
            decode_context=context,
            geometry_signatures=tuple(signature for _ in media_batch),
        )


@dataclass(frozen=True, slots=True)
class BagelLatentLayout:
    """Describe Bagel's VAE-to-token geometry."""

    patch_size: int
    channels: int
    downsample: int


def encode_bagel_vae_image(
    vae: Any,
    pixel_values: torch.Tensor,
    *,
    posterior_mode: BagelPosteriorMode,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply explicit posterior selection and Bagel latent normalization.

    This is the role-neutral numerical primitive for Bagel's custom VAE. Existing
    condition encoding selects the posterior mean through ``vae.reg.sample=False``;
    target encoding calls this primitive with ``posterior_mode='sample'`` so it does
    not mutate that global condition policy.

    Args:
        vae: Bagel custom VAE exposing ``encoder``, ``reg``, ``scale_factor``, and
            ``shift_factor``.
        pixel_values: Floating BCHW images in Bagel's official input range.
        posterior_mode: Explicit posterior selection for the calling role.
        generator: Optional generator used only when sampling.

    Returns:
        Shifted and scaled BCHW clean latents.
    """
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
        shape = tuple(pixel_values.shape) if isinstance(pixel_values, torch.Tensor) else None
        raise ValueError(
            "Bagel VAE target encoding expected a rank-4 BCHW tensor, "
            f"received {type(pixel_values).__name__} with shape {shape}"
        )
    if not pixel_values.is_floating_point():
        raise TypeError(
            "Bagel VAE target encoding expected floating pixel values, "
            f"received {pixel_values.dtype}"
        )
    if type(posterior_mode) is not str:
        raise TypeError(
            "Bagel VAE posterior_mode must be str, "
            f"received {type(posterior_mode).__name__}: {posterior_mode!r}"
        )
    if posterior_mode not in ("sample", "argmax"):
        raise ValueError(
            "Bagel VAE posterior_mode must be 'sample' or 'argmax', " f"received {posterior_mode!r}"
        )
    encoder = getattr(vae, "encoder", None)
    if not callable(encoder):
        raise TypeError("Bagel custom VAE must expose a callable encoder")
    moments = encoder(pixel_values)
    if not isinstance(moments, torch.Tensor) or moments.ndim != 4:
        shape = tuple(moments.shape) if isinstance(moments, torch.Tensor) else None
        raise TypeError(
            "Bagel custom VAE encoder expected a rank-4 tensor of mean/logvar moments, "
            f"received {type(moments).__name__} with shape {shape}"
        )

    reg = getattr(vae, "reg", None)
    chunk_dim = getattr(reg, "chunk_dim", None)
    if not isinstance(chunk_dim, Integral) or isinstance(chunk_dim, bool):
        raise TypeError(
            "Bagel custom VAE reg.chunk_dim must be an integer, " f"received {chunk_dim!r}"
        )
    chunk_dim = int(chunk_dim)
    if chunk_dim < -moments.ndim or chunk_dim >= moments.ndim:
        raise ValueError(
            f"Bagel custom VAE reg.chunk_dim {chunk_dim} is invalid for rank {moments.ndim}"
        )
    normalized_chunk_dim = chunk_dim % moments.ndim
    if moments.shape[normalized_chunk_dim] % 2:
        raise ValueError(
            "Bagel custom VAE encoder mean/logvar dimension must be even, "
            f"received shape {tuple(moments.shape)} at dim {chunk_dim}"
        )
    if not moments.is_floating_point():
        raise TypeError(
            "Bagel custom VAE encoder moments must be floating, " f"received {moments.dtype}"
        )

    selector = getattr(reg, "select", None)
    if not callable(selector):
        raise TypeError(
            "Bagel custom VAE reg must expose callable select() for explicit " "posterior policy"
        )
    latents = selector(
        moments,
        sample=posterior_mode == "sample",
        generator=generator,
    )
    if not isinstance(latents, torch.Tensor) or latents.ndim != moments.ndim:
        shape = tuple(latents.shape) if isinstance(latents, torch.Tensor) else None
        raise TypeError(
            "Bagel custom VAE posterior selection expected a rank-4 tensor, "
            f"received {type(latents).__name__} with shape {shape}"
        )

    shift = _finite_real(getattr(vae, "shift_factor", None), "vae.shift_factor")
    scale = _finite_real(getattr(vae, "scale_factor", None), "vae.scale_factor")
    if scale <= 0:
        raise ValueError(f"Bagel vae.scale_factor must be positive, received {scale!r}")
    normalize_latents = getattr(vae, "normalize_latents", None)
    if not callable(normalize_latents):
        raise TypeError(
            "Bagel custom VAE must expose callable normalize_latents() so condition "
            "and target roles share shift/scale math"
        )
    normalized = normalize_latents(latents)
    if not isinstance(normalized, torch.Tensor) or normalized.shape != latents.shape:
        shape = tuple(normalized.shape) if isinstance(normalized, torch.Tensor) else None
        raise TypeError(
            "Bagel custom VAE normalize_latents expected a shape-preserving tensor, "
            f"received {type(normalized).__name__} with shape {shape}"
        )
    return normalized


def resolve_bagel_latent_layout(bagel: Any) -> BagelLatentLayout:
    """Validate the Bagel module's official two-by-two latent token layout.

    Args:
        bagel: Logical Bagel component exposing latent-layout attributes.

    Returns:
        Validated latent layout.
    """
    patch_size = _positive_int(
        getattr(bagel, "latent_patch_size", None),
        "bagel.latent_patch_size",
    )
    if patch_size != 2:
        raise ValueError(
            "Bagel offline output encoding supports latent_patch_size=2, " f"received {patch_size}"
        )
    return BagelLatentLayout(
        patch_size=patch_size,
        channels=_positive_int(
            getattr(bagel, "latent_channel", None),
            "bagel.latent_channel",
        ),
        downsample=_positive_int(
            getattr(bagel, "latent_downsample", None),
            "bagel.latent_downsample",
        ),
    )


def pack_bagel_latents(
    latents: torch.Tensor,
    *,
    image_shape: Tuple[int, int],
    layout: BagelLatentLayout,
) -> torch.Tensor:
    """Crop and pack BCHW latents into Bagel's ``B,N,patch^2*C`` order.

    Args:
        latents: Shifted and scaled Bagel VAE latents.
        image_shape: Post-transform target height and width.
        layout: Validated Bagel latent layout.

    Returns:
        Packed clean target tokens.
    """
    if not isinstance(latents, torch.Tensor) or latents.ndim != 4:
        shape = tuple(latents.shape) if isinstance(latents, torch.Tensor) else None
        raise ValueError(
            "Bagel target packing expected rank-4 BCHW latents, "
            f"received {type(latents).__name__} with shape {shape}"
        )
    if latents.shape[1] != layout.channels:
        raise ValueError(
            "Bagel custom VAE channel count disagrees with bagel.latent_channel: "
            f"expected {layout.channels}, received shape {tuple(latents.shape)}"
        )
    height, width = image_shape
    if height % layout.downsample or width % layout.downsample:
        raise ValueError(
            "Bagel target geometry must be divisible by bagel.latent_downsample "
            f"{layout.downsample}, received {image_shape}"
        )
    token_height = height // layout.downsample
    token_width = width // layout.downsample
    required_height = token_height * layout.patch_size
    required_width = token_width * layout.patch_size
    if latents.shape[-2] < required_height or latents.shape[-1] < required_width:
        raise ValueError(
            "Bagel custom VAE output is too small for the official crop/patchify path: "
            f"required at least {(required_height, required_width)}, received "
            f"{tuple(latents.shape[-2:])}"
        )

    cropped = latents[:, :, :required_height, :required_width]
    batch_size, channels = cropped.shape[:2]
    patch_size = layout.patch_size
    return (
        cropped.reshape(
            batch_size,
            channels,
            token_height,
            patch_size,
            token_width,
            patch_size,
        )
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(
            batch_size,
            token_height * token_width,
            patch_size * patch_size * channels,
        )
    )


def validate_bagel_encoded_output_geometry(
    adapter: Any,
    media_batch: DecodedMediaBatch,
    encoded: EncodedOutputState,
) -> None:
    """Verify output-derived geometry against Bagel's transform and token grid.

    Args:
        adapter: Bagel adapter exposing transform and logical components.
        media_batch: Decoded target image batch.
        encoded: Codec result after generic boundary validation.
    """
    expected_shapes = tuple(
        _resized_image_shape(adapter.vae_transform, candidate[0].payload, sample_index)
        for sample_index, candidate in enumerate(media_batch)
    )
    image_shape = expected_shapes[0]
    if any(shape != image_shape for shape in expected_shapes[1:]):
        raise ValueError(
            "Bagel output geometry validation expected a uniform transformed batch, "
            f"received {expected_shapes}"
        )

    expected_signature = GeometrySignature(
        media=(
            MediaGeometrySignature(
                type=MediaType.IMAGE,
                height=image_shape[0],
                width=image_shape[1],
            ),
        )
    )
    expected_signatures = tuple(expected_signature for _ in media_batch)
    if encoded.geometry_signatures != expected_signatures:
        raise ValueError(
            "Bagel encoded output signatures must match output-media-derived geometry "
            f"{image_shape}, received {encoded.geometry_signatures!r}"
        )

    expected_context = {"image_shape": image_shape}
    if dict(encoded.forward_context) != expected_context:
        raise ValueError(
            "Bagel forward_context must contain only the output-derived image_shape "
            f"{image_shape}, received {dict(encoded.forward_context)!r}"
        )
    if dict(encoded.decode_context) != expected_context:
        raise ValueError(
            "Bagel decode_context must contain only the output-derived image_shape "
            f"{image_shape}, received {dict(encoded.decode_context)!r}"
        )

    layout = resolve_bagel_latent_layout(adapter.get_component("bagel"))
    if image_shape[0] % layout.downsample or image_shape[1] % layout.downsample:
        raise ValueError(
            "Bagel encoded output geometry must be divisible by bagel.latent_downsample "
            f"{layout.downsample}, received {image_shape}"
        )
    expected_shape = (
        len(media_batch),
        (image_shape[0] // layout.downsample) * (image_shape[1] // layout.downsample),
        layout.patch_size * layout.patch_size * layout.channels,
    )
    actual_shape = tuple(encoded.clean_state.components["latent"].shape)
    if actual_shape != expected_shape:
        raise ValueError(
            "Bagel clean target state disagrees with the output-media token grid: "
            f"expected {expected_shape}, received {actual_shape}"
        )


def _transform_target_images(
    adapter: Any,
    media_batch: DecodedMediaBatch,
) -> tuple[list[torch.Tensor], Tuple[int, int]]:
    transformed_images = []
    image_shapes = []
    for sample_index, candidate in enumerate(media_batch):
        if len(candidate) != 1:
            raise ValueError(
                "Bagel output codec expected one image per sample, "
                f"received {len(candidate)} for sample {sample_index}"
            )
        image = require_decoded_rgb_image(
            candidate[0].payload,
            source=f"Bagel output codec sample {sample_index}",
        )
        transformed = adapter.vae_transform(pil_img2rgb(image))
        if not isinstance(transformed, torch.Tensor):
            raise TypeError(
                "Bagel vae_transform expected torch.Tensor output, "
                f"received {type(transformed).__name__} for sample {sample_index}"
            )
        if transformed.ndim != 3 or transformed.shape[0] != 3:
            raise ValueError(
                "Bagel vae_transform expected CHW RGB output, "
                f"received shape {tuple(transformed.shape)} for sample {sample_index}"
            )
        if not transformed.is_floating_point():
            raise TypeError(
                "Bagel vae_transform expected floating output, "
                f"received dtype {transformed.dtype} for sample {sample_index}"
            )
        if transformed.shape[-2] <= 0 or transformed.shape[-1] <= 0:
            raise ValueError(
                "Bagel vae_transform produced non-positive target geometry "
                f"{tuple(transformed.shape[-2:])} for sample {sample_index}"
            )
        image_shapes.append((transformed.shape[-2], transformed.shape[-1]))
        transformed_images.append(transformed)

    image_shape = image_shapes[0]
    if any(shape != image_shape for shape in image_shapes[1:]):
        raise ValueError(
            "Bagel offline target batches require identical post-transform image "
            f"geometry, received {tuple(image_shapes)}. Use batch size 1 or batch "
            "targets by the geometry produced by Bagel's official vae_transform."
        )
    return transformed_images, image_shape


def _resized_image_shape(
    transform: Any,
    image: Any,
    sample_index: int,
) -> Tuple[int, int]:
    image = require_decoded_rgb_image(
        image,
        source=f"Bagel output geometry sample {sample_index}",
    )
    resize_transform = getattr(transform, "resize_transform", None)
    if not callable(resize_transform):
        raise TypeError("Bagel output geometry validation requires vae_transform.resize_transform")
    resized = resize_transform(pil_img2rgb(image))
    if isinstance(resized, Image.Image):
        return resized.height, resized.width
    if isinstance(resized, torch.Tensor) and resized.ndim >= 2:
        return resized.shape[-2], resized.shape[-1]
    raise TypeError(
        "Bagel resize_transform expected PIL.Image or Tensor output, "
        f"received {type(resized).__name__} for sample {sample_index}"
    )


def _module_dtype(module: nn.Module) -> torch.dtype:
    dtype = getattr(module, "dtype", None)
    if not isinstance(dtype, torch.dtype):
        try:
            dtype = next(module.parameters()).dtype
        except (AttributeError, StopIteration) as exc:
            raise TypeError(
                "Bagel output codec could not resolve the custom VAE parameter dtype"
            ) from exc
    if not dtype.is_floating_point:
        raise TypeError(
            "Bagel output codec expected a floating custom VAE dtype, " f"received {dtype}"
        )
    return dtype


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(
            f"Bagel output codec expected integer {name}, "
            f"received {type(value).__name__}: {value!r}"
        )
    result = int(value)
    if result <= 0:
        raise ValueError(
            f"Bagel output codec expected positive integer {name}, received {result!r}"
        )
    return result


def _finite_real(value: Any, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(
            f"Bagel output codec expected real {name}, "
            f"received {type(value).__name__}: {value!r}"
        )
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"Bagel output codec expected finite {name}, received {result!r}")
    return result


__all__ = [
    "BagelLatentLayout",
    "BagelOutputStateCodec",
    "BagelPosteriorMode",
    "encode_bagel_vae_image",
    "pack_bagel_latents",
    "resolve_bagel_latent_layout",
    "validate_bagel_encoded_output_geometry",
]
