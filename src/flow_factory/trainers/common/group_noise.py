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

"""Group-shared noise for rollout initialization and replay re-noising.

The two call sites answer different questions and must not share one flag:

* ``init_same_noise`` (Flow-GRPO, Flow-DPPO, GRPO-Guard) decides whether members
  of one sampling group start their rollout from the same initial latent. The
  per-step SDE noise stays per-sample. This matches UniRL ``init_same_noise``:
  the group id seeds ``x_T``, while transition noise stays on its own stream.
* ``use_shared_noise`` (DGPO, and opt-in for DiffusionNFT / AWM) decides whether
  members of one group reuse one forward-process noise tensor when a clean
  rollout is re-noised at one training timestep. The next timestep draws a new
  tensor, matching official DGPO. It is not the rollout's initial latent.

Both are keyed by the planned group id so the draw is identical across ranks
and micro-batches without a broadcast.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from diffusers.utils.torch_utils import randn_tensor

from ...data_utils.sampling_plan import SAMPLING_GROUP_ID_COLUMN
from ...hparams.training_args._base import require_config_bool
from ...samples import LEGACY_SOURCE_ID, BaseSample, LatentState, sample_group_key
from ...utils.base import create_generator
from .state_validation import require_latent_state, state_batch_size

# Appended to ``create_generator`` integer keys so these streams never collide
# with each other. A missing generator still uses the process RNG.
SEED_TAG_SHARED_TIMESTEPS = 1
SEED_TAG_SHARED_NOISE = 2
SEED_TAG_SOURCE_NAMESPACE = 3
SEED_TAG_INIT_SAME_NOISE = 4
SEED_TAG_INDEPENDENT_NOISE = 6

# Separate counters so a timestep draw does not shift the following noise draw.
STREAM_FORWARD_NOISE = 1
STREAM_QUERY_TIME = 2
STREAM_CONDITIONAL_FRESH = 3

_SOURCE_ID_COLUMN = "__source_id__"


def _as_row_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int for {name}, got {type(value).__name__}: {value!r}")
    return value


def stable_text_id(text: str) -> int:
    """Map a record id to a stable integer. String ``hash`` depends on ``PYTHONHASHSEED``."""
    if not isinstance(text, str) or not text:
        raise TypeError(f"expected a non-empty str record id, got {type(text).__name__}: {text!r}")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def sample_noise_ids(samples: Sequence[Any]) -> tuple[int, ...]:
    """Return planned sample ids, one per row, for independent seeded draws."""
    if not samples:
        raise ValueError("expected at least one sample for a seeded noise draw")
    ids: List[int] = []
    for index, sample in enumerate(samples):
        sample_id = getattr(sample, "sampling_sample_id", None)
        if isinstance(sample_id, bool) or not isinstance(sample_id, int):
            raise TypeError(
                f"expected int sampling_sample_id for sample[{index}], "
                f"got {type(sample_id).__name__}: {sample_id!r}"
            )
        ids.append(sample_id)
    return tuple(ids)


def record_noise_ids(record_ids: Sequence[str]) -> tuple[int, ...]:
    """Return stable integer ids for an offline batch, in row order."""
    if not record_ids:
        raise ValueError("expected at least one offline record id for a seeded noise draw")
    return tuple(stable_text_id(record_id) for record_id in record_ids)


def _seed_parts(trainer: Any) -> tuple[int, int]:
    return int(trainer.training_args.seed), int(trainer.epoch)


def identity_timestep_generator(
    trainer: Any,
    identity: Sequence[int],
    inner_epoch: int = 0,
) -> torch.Generator:
    """Seed one batch's training coordinates from the sample identity.

    ``TimeSampler`` still spreads the coordinates inside this draw. A different
    identity, or a different inner epoch, builds a different generator. Ranks
    that hold different samples therefore do not reuse each other's coordinates.
    """
    if isinstance(inner_epoch, bool) or not isinstance(inner_epoch, int) or inner_epoch < 0:
        raise TypeError(
            "expected a non-negative int for inner_epoch, "
            f"got {type(inner_epoch).__name__}: {inner_epoch!r}"
        )
    if not identity:
        raise ValueError("expected a non-empty sample identity for training timesteps")
    seed, epoch = _seed_parts(trainer)
    return create_generator(
        seed,
        epoch,
        inner_epoch,
        *[int(sample_id) for sample_id in identity],
        SEED_TAG_SHARED_TIMESTEPS,
    )


def independent_batch_generator(
    trainer: Any,
    identity: Sequence[int],
    timestep_index: int,
    stream: int,
) -> torch.Generator:
    """Seed one forward-process draw for one batch at one timestep.

    The identity is part of the key, so different samples do not share the
    tensor. ``timestep_index`` makes the next training step a new draw. ``stream``
    separates forward noise from query-time and conditional-fresh draws.
    """
    if (
        isinstance(timestep_index, bool)
        or not isinstance(timestep_index, int)
        or timestep_index < 0
    ):
        raise TypeError(
            "expected a non-negative int for timestep_index, "
            f"got {type(timestep_index).__name__}: {timestep_index!r}"
        )
    if isinstance(stream, bool) or not isinstance(stream, int) or stream < 0:
        raise TypeError(
            f"expected a non-negative int for stream, got {type(stream).__name__}: {stream!r}"
        )
    if not identity:
        raise ValueError("expected a non-empty sample identity for independent noise")
    seed, epoch = _seed_parts(trainer)
    return create_generator(
        seed,
        epoch,
        stream,
        timestep_index,
        *[int(sample_id) for sample_id in identity],
        SEED_TAG_INDEPENDENT_NOISE,
    )


def sequenced_draw_generator(
    trainer: Any,
    stream: int,
    identity: Sequence[int],
    *extra: int,
) -> torch.Generator:
    """Seed one draw in a deterministic sequence.

    Repeated calls on the same stream advance a counter, so a second Monte Carlo
    draw is a new tensor. The counter resets when ``seed`` or ``epoch`` changes.
    """
    if isinstance(stream, bool) or not isinstance(stream, int) or stream < 0:
        raise TypeError(
            f"expected a non-negative int for stream, got {type(stream).__name__}: {stream!r}"
        )
    for index, value in enumerate(extra):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(
                f"expected a non-negative int for extra[{index}], "
                f"got {type(value).__name__}: {value!r}"
            )
    seed, epoch = _seed_parts(trainer)
    key = (seed, epoch)
    counts = getattr(trainer, "_independent_draw_counts", None)
    if not isinstance(counts, dict) or getattr(trainer, "_independent_draw_key", None) != key:
        counts = {}
        trainer._independent_draw_counts = counts
        trainer._independent_draw_key = key
    draw_index = int(counts.get(stream, 0))
    counts[stream] = draw_index + 1
    return create_generator(
        seed,
        epoch,
        stream,
        draw_index,
        *extra,
        *[int(sample_id) for sample_id in identity],
        SEED_TAG_INDEPENDENT_NOISE,
    )


def training_timestep_generator(
    trainer: Any,
    inner_epoch: int,
    batch_index: int,
) -> torch.Generator:
    """Seed one microbatch's training-time draw.

    The key is ``(seed, epoch, inner_epoch, batch_index)``. Each call builds a
    new generator, and ``TimeSampler`` spreads ``num_train_timesteps``
    coordinates inside that draw. The next microbatch index draws a different
    set. Ranks that share these integers draw the same coordinates locally.
    """
    for name, value in (("inner_epoch", inner_epoch), ("batch_index", batch_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(
                f"expected a non-negative int for {name}, got {type(value).__name__}: {value!r}"
            )
    return create_generator(
        int(trainer.training_args.seed),
        int(trainer.epoch),
        inner_epoch,
        batch_index,
        SEED_TAG_SHARED_TIMESTEPS,
    )


def _prompt_count(batch: Dict[str, Any]) -> int:
    prompts = batch.get("prompt")
    if isinstance(prompts, str):
        return 1
    if isinstance(prompts, (list, tuple)) and prompts:
        return len(prompts)
    raise ValueError(
        "init_same_noise requires batch['prompt'] to be a non-empty list of strings "
        f"or one string, received {type(prompts).__name__}"
    )


def init_same_noise_generators(
    trainer: Any,
    batch: Dict[str, Any],
) -> Optional[List[torch.Generator]]:
    """Build one CPU generator per row, seeded by the row's sampling group.

    Rows that share ``(seed, epoch, group_id, source_id)`` receive generators
    at the same initial state, so the initial latent draw matches. Adapters
    must not keep consuming these generators for SDE transitions.

    Returns ``None`` when the trainer has no ``init_same_noise`` flag or the
    flag is false. An explicit ``generator`` passed by the caller (evaluation)
    is left untouched by ``sample_batch``.

    Raises:
        TypeError: If the flag or an id is not the expected type.
        ValueError: If the flag is set but the planned group ids are missing
            or do not line up with the prompt batch.
    """
    flag = getattr(trainer.training_args, "init_same_noise", False)
    if not require_config_bool(flag, "train.init_same_noise"):
        return None
    count = _prompt_count(batch)
    group_ids = batch.get(SAMPLING_GROUP_ID_COLUMN)
    if not isinstance(group_ids, (list, tuple)) or len(group_ids) != count:
        received = None if not isinstance(group_ids, (list, tuple)) else len(group_ids)
        raise ValueError(
            "init_same_noise requires one planned sampling group id per prompt row "
            f"({SAMPLING_GROUP_ID_COLUMN}); prompts={count}, received {received}"
        )
    source_ids = batch.get(_SOURCE_ID_COLUMN)
    if source_ids is None:
        source_ids = [LEGACY_SOURCE_ID] * count
    elif not isinstance(source_ids, (list, tuple)) or len(source_ids) != count:
        received = None if not isinstance(source_ids, (list, tuple)) else len(source_ids)
        raise ValueError(
            "init_same_noise requires source ids aligned with the prompt batch; "
            f"prompts={count}, received {received}"
        )
    generators: List[torch.Generator] = []
    for index, (group_id, source_id) in enumerate(zip(group_ids, source_ids)):
        generators.append(
            create_generator(
                int(trainer.training_args.seed),
                int(trainer.epoch),
                _as_row_int(group_id, f"{SAMPLING_GROUP_ID_COLUMN}[{index}]"),
                _as_row_int(source_id, f"{_SOURCE_ID_COLUMN}[{index}]"),
                SEED_TAG_INIT_SAME_NOISE,
            )
        )
    return generators


def require_planned_group_ids(samples: Sequence[BaseSample], *, context: str) -> None:
    """Raise when shared replay noise would fall back to a content fingerprint."""
    missing = [index for index, sample in enumerate(samples) if sample.sampling_group_id is None]
    if missing:
        shown = missing[:5]
        suffix = "..." if len(missing) > len(shown) else ""
        raise ValueError(
            f"{context} requires sampling_group_id on every sample so members of one "
            f"group share noise; missing at indices {shown}{suffix}"
        )


def draw_group_component_noise(
    trainer: Any,
    *,
    unique_id: int,
    source_id: int = LEGACY_SOURCE_ID,
    component_name: str,
    component_index: int,
    shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    inner_epoch: int,
    timestep_index: int,
) -> torch.Tensor:
    """Draw one group's noise for one component from an explicit seed tuple.

    The primary component keeps the legacy DGPO namespace when ``source_id`` is
    unavailable; source-tagged samples append a source namespace so equal prompt
    hashes from different datasets cannot share noise. Further components extend
    the key by their position in ``trajectory_component_order``.
    """
    if not isinstance(component_name, str) or not component_name:
        raise TypeError(
            "expected a non-empty str for component_name, "
            f"got {type(component_name).__name__}: {component_name!r}"
        )
    if (
        isinstance(timestep_index, bool)
        or not isinstance(timestep_index, int)
        or timestep_index < 0
    ):
        raise TypeError(
            "expected a non-negative int for timestep_index, "
            f"got {type(timestep_index).__name__}: {timestep_index!r}"
        )
    namespace = () if component_index == 0 else (component_index,)
    source_namespace = (
        () if source_id == LEGACY_SOURCE_ID else (SEED_TAG_SOURCE_NAMESPACE, source_id)
    )
    generator = create_generator(
        trainer.training_args.seed,
        trainer.epoch,
        inner_epoch,
        int(unique_id),
        SEED_TAG_SHARED_NOISE,
        timestep_index,
        *source_namespace,
        *namespace,
        device=device,
    )
    return randn_tensor(shape, generator=generator, device=device, dtype=dtype)


def shared_group_noise(
    trainer: Any,
    clean_state: LatentState,
    samples: List[BaseSample],
    inner_epoch: int,
    *,
    timestep_index: int,
    draw: Optional[Callable[..., torch.Tensor]] = None,
) -> LatentState:
    """Build source-aware per-group noise for every trajectory component.

    Every sample of a group receives the same component noise at this training
    timestep. ``timestep_index`` is part of the seed, so the next timestep draws
    a new tensor, matching official DGPO's per-step ``randn``. Ranks that share
    the integer key draw the same tensor locally. ``draw`` defaults to
    :func:`draw_group_component_noise`; DGPO passes its instance method so
    tests can replace that method.
    """

    def default_draw(**kwargs: Any) -> torch.Tensor:
        return draw_group_component_noise(trainer, **kwargs)

    if draw is None:
        draw = default_draw
    component_names = trainer.adapter.trajectory_component_order
    context = (
        f"inner_epoch={inner_epoch}, timestep_index={timestep_index}, "
        f"component order {component_names}"
    )
    batch_size = state_batch_size(trainer, clean_state, "terminal clean state")
    if len(samples) != batch_size:
        raise ValueError(
            f"expected {type(trainer).__name__} shared noise ({context}) to receive one "
            f"sample per terminal state row, i.e. {batch_size} samples, received {len(samples)}"
        )
    references: Dict[str, torch.Tensor] = {}
    for name in component_names:
        component = clean_state.components[name]
        if component.ndim < 2 or component.shape[0] != batch_size:
            raise ValueError(
                f"expected {type(trainer).__name__} terminal clean component {name!r} "
                f"({context}) to be batched with shape (B, ...) and batch size "
                f"{batch_size}, received {tuple(component.shape)}"
            )
        references[name] = component

    group_cache: Dict[Any, Dict[str, torch.Tensor]] = {}
    rows: Dict[str, List[torch.Tensor]] = {name: [] for name in component_names}
    for sample in samples:
        group_key = sample_group_key(sample)
        group_noise = group_cache.get(group_key)
        if group_noise is None:
            group_noise = {
                name: draw(
                    unique_id=group_key.unique_id,
                    source_id=group_key.source_id,
                    component_name=name,
                    component_index=component_index,
                    shape=references[name].shape[1:],
                    dtype=references[name].dtype,
                    device=references[name].device,
                    inner_epoch=inner_epoch,
                    timestep_index=timestep_index,
                )
                for component_index, name in enumerate(component_names)
            }
            group_cache[group_key] = group_noise
        for name in component_names:
            noise = group_noise[name]
            reference = references[name]
            if (
                noise.shape != reference.shape[1:]
                or noise.dtype != reference.dtype
                or noise.device != reference.device
            ):
                raise ValueError(
                    f"expected {type(trainer).__name__} shared noise for unique_id="
                    f"{group_key.unique_id}, source_id={group_key.source_id} ({context}) "
                    f"component {name!r} to match the clean "
                    f"per-sample shape/dtype/device ({tuple(reference.shape[1:])}, "
                    f"{reference.dtype}, {reference.device}), received "
                    f"({tuple(noise.shape)}, {noise.dtype}, {noise.device})"
                )
            rows[name].append(noise)
    return require_latent_state(
        trainer,
        LatentState({name: torch.stack(rows[name], dim=0) for name in component_names}),
        f"shared noise ({context})",
    )
