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
  With the flag off, each member still gets an explicit generator, keyed by
  that member's sample id, so ``x_T`` does not follow the process RNG.
* ``use_shared_noise`` (DGPO, and opt-in for DiffusionNFT / AWM) decides whether
  members of one group reuse one forward-process noise tensor when a clean
  rollout is re-noised at one training timestep. The next timestep draws a new
  tensor, matching official DGPO. It is not the rollout's initial latent.

Both are keyed by the planned group id so the draw is identical across ranks
and micro-batches without a broadcast.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from diffusers.utils.torch_utils import randn_tensor

from ...data_utils.sampling_plan import SAMPLING_GROUP_ID_COLUMN, SAMPLING_SAMPLE_ID_COLUMN
from ...hparams.training_args._base import require_config_bool
from ...samples import LEGACY_SOURCE_ID, BaseSample, LatentState, sample_group_key
from ...utils.base import create_generator, stable_text_id
from .state_validation import require_latent_state, state_batch_size

# Appended to ``create_generator`` integer keys so these streams never collide.
SEED_TAG_SHARED_TIMESTEPS = 1
SEED_TAG_SHARED_NOISE = 2
SEED_TAG_SOURCE_NAMESPACE = 3
SEED_TAG_INIT_SAME_NOISE = 4
SEED_TAG_INDEPENDENT_NOISE = 6
# Independent ``x_T``. Distinct from tag 4 so a shared group key cannot land on
# a per-sample key, and from tag 5 (SDE steps) so ``x_T`` is not a transition.
SEED_TAG_INIT_LATENT = 7

# Separate counters so a timestep draw does not shift the following noise draw.
STREAM_FORWARD_NOISE = 1
STREAM_QUERY_TIME = 2
STREAM_CONDITIONAL_FRESH = 3
# Offline target encoding: a stochastic output codec (a VAE posterior
# `sample()`) must draw from the record identity, not the process RNG, or the
# supervised target depends on which rank encoded it -- and on how many ranks
# were launched, since `set_seed(device_specific=True)` keys the process RNG
# per rank. Tensor-parallel peers would then train toward different targets.
STREAM_OUTPUT_ENCODING = 4
# DMD2's forward-process perturbation coordinate. A distinct stream because it
# is drawn twice per replay unit (once for the fake critic, once for the
# generator) and must not correlate with the forward noise added at those
# coordinates.
STREAM_PERTURBATION_TIME = 5

_SOURCE_ID_COLUMN = "__source_id__"


def _as_row_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int for {name}, got {type(value).__name__}: {value!r}")
    return value


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


def _prompt_rows(batch: Dict[str, Any]) -> List[str]:
    prompts = batch.get("prompt")
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise ValueError(
            "rollout initial noise requires batch['prompt'] to be a non-empty list of "
            f"strings or one string, received {type(prompts).__name__}"
        )
    rows: List[str] = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt:
            raise TypeError(
                "expected a non-empty str prompt to seed rollout initial noise at "
                f"row {index}, got {type(prompt).__name__}: {prompt!r}"
            )
        rows.append(prompt)
    return rows


def _optional_id_column(batch: Dict[str, Any], column: str, count: int) -> Optional[List[int]]:
    values = batch.get(column)
    if values is None:
        return None
    if not isinstance(values, (list, tuple)) or len(values) != count:
        received = None if not isinstance(values, (list, tuple)) else len(values)
        raise ValueError(
            f"rollout initial noise requires {column} to align with the prompt batch; "
            f"prompts={count}, received {received}"
        )
    return [_as_row_int(value, f"{column}[{index}]") for index, value in enumerate(values)]


def _source_ids(batch: Dict[str, Any], count: int) -> List[int]:
    values = batch.get(_SOURCE_ID_COLUMN)
    if values is None:
        return [LEGACY_SOURCE_ID] * count
    if not isinstance(values, (list, tuple)) or len(values) != count:
        received = None if not isinstance(values, (list, tuple)) else len(values)
        raise ValueError(
            "rollout initial noise requires source ids aligned with the prompt batch; "
            f"prompts={count}, received {received}"
        )
    return [
        _as_row_int(value, f"{_SOURCE_ID_COLUMN}[{index}]") for index, value in enumerate(values)
    ]


def _rollout_seed(trainer: Any) -> tuple[int, int]:
    args = trainer.training_args
    seed = args["seed"] if isinstance(args, dict) else getattr(args, "seed", None)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"expected int for train.seed, got {type(seed).__name__}: {seed!r}")
    epoch = getattr(trainer, "epoch", None)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise TypeError(
            f"expected a non-negative int for trainer.epoch, got {type(epoch).__name__}: {epoch!r}"
        )
    return seed, epoch


def rollout_initial_latent_generators(
    trainer: Any,
    batch: Dict[str, Any],
) -> List[torch.Generator]:
    """Build one CPU generator per row for the rollout initial latent.

    ``init_same_noise`` selects the key. Both keys include ``train.seed``,
    ``epoch``, and ``source_id``, and neither includes rank or batch position.

    * Flag true: ``(seed, epoch, group_id, source_id)``. Members of one group
      start from the same generator state, so they share ``x_T``. Different
      groups, epochs, and sources do not.
    * Flag false or absent: ``(seed, epoch, group_id, sample_id, source_id)``.
      ``sample_id`` keeps members of one group apart. Swapping batch rows keeps
      each row's noise on that id.

    Planned training batches carry both id columns. A batch with neither column
    (evaluation is expected to pass its own generator and not reach this fallback)
    uses ``(seed, epoch, prompt_hash, row_index, source_id)`` so repeated prompt
    text in one unplanned batch still draws different noise. Adapters must not
    keep consuming these generators for SDE transitions.

    Raises:
        TypeError: If the flag, seed, epoch, or an id has the wrong type.
        ValueError: If the planned ids are only partly present or do not line
            up with the prompt batch.
    """
    flag = getattr(trainer.training_args, "init_same_noise", False)
    shared = require_config_bool(flag, "train.init_same_noise")
    prompts = _prompt_rows(batch)
    count = len(prompts)
    group_ids = _optional_id_column(batch, SAMPLING_GROUP_ID_COLUMN, count)
    sample_ids = _optional_id_column(batch, SAMPLING_SAMPLE_ID_COLUMN, count)
    source_ids = _source_ids(batch, count)
    seed, epoch = _rollout_seed(trainer)
    if shared and group_ids is None:
        raise ValueError(
            "init_same_noise requires one planned sampling group id per prompt row "
            f"({SAMPLING_GROUP_ID_COLUMN}); prompts={count}, received None"
        )
    if not shared and (group_ids is None) != (sample_ids is None):
        missing = SAMPLING_SAMPLE_ID_COLUMN if sample_ids is None else SAMPLING_GROUP_ID_COLUMN
        raise ValueError(
            "independent rollout initial noise requires both planned group ids and "
            f"sample ids; missing {missing}"
        )
    generators: List[torch.Generator] = []
    for index, source_id in enumerate(source_ids):
        if shared:
            assert group_ids is not None
            key = (
                seed,
                epoch,
                group_ids[index],
                source_id,
                SEED_TAG_INIT_SAME_NOISE,
            )
        elif group_ids is None:
            key = (
                seed,
                epoch,
                stable_text_id(prompts[index]),
                index,
                source_id,
                SEED_TAG_INIT_LATENT,
            )
        else:
            assert sample_ids is not None
            key = (
                seed,
                epoch,
                group_ids[index],
                sample_ids[index],
                source_id,
                SEED_TAG_INIT_LATENT,
            )
        generators.append(create_generator(*key))
    return generators


def init_same_noise_generators(
    trainer: Any,
    batch: Dict[str, Any],
) -> List[torch.Generator]:
    """Return per-row ``x_T`` generators. The flag only selects the seed key."""
    return rollout_initial_latent_generators(trainer, batch)


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
