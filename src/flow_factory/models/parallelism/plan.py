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


"""Resolve model declarations into total parameter plans without installing them."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Mapping, NamedTuple, Optional, Tuple

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
)

from .peft import derive_lora_plan
from .sequence import SequenceBroadcastOutput, _RegionEntry
from .tensor import ReplicateParallel

logger = logging.getLogger("flow_factory.models.tensor_parallel")


# Styles that transform activations without distributing any parameter.
_ACTIVATION_ONLY_STYLES: tuple = (
    PrepareModuleInput,
    PrepareModuleOutput,
    PrepareModuleInputOutput,
)


class ComposedParallelStyle(ParallelStyle):
    """Apply several styles to one module, in order.

    ``parallelize_module`` takes one style per path, but a module can need two
    unrelated treatments. ``PrepareModuleInput`` and ``PrepareModuleOutput``
    transform activations and distribute nothing, so a module that owns
    parameters *and* sits on a sharding boundary -- which every transformer
    block holding a modulation table does -- would otherwise have those
    parameters left undistributed, and ``apply_tensor_parallel`` rejects the
    plan. Styles are applied in the supplied order, which also determines the
    registration order of their hooks; callers own that composition.
    """

    def __init__(self, *styles: ParallelStyle) -> None:
        super().__init__()
        if not styles:
            raise ValueError("ComposedParallelStyle needs at least one style")
        self.styles = styles

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        for style in self.styles:
            module = style._apply(module, device_mesh)
        return module


def _modules_owning_parameters(root: nn.Module) -> Iterable[str]:
    """Yield the name of every module that directly owns a parameter."""
    for module_name, module in root.named_modules():
        if any(True for _ in module.named_parameters(recurse=False)):
            yield module_name


class ConditionalEntry(NamedTuple):
    """A plan entry whose modules only some checkpoints of a family have.

    One adapter serves a whole family, and the plan is a ``classmethod`` chosen
    before any weight is read, so it cannot branch on the loaded config. Wan's
    I2V adapter covers both Wan2.1-I2V, whose attention carries a second
    key/value stream for the CLIP image conditioning
    (``added_kv_proj_dim=5120``), and Wan2.2-I2V, which has none
    (``added_kv_proj_dim=None``) -- naming those three modules unconditionally
    makes the glob expansion reject the plan *after* a 14B load.

    Entries sharing a ``group`` are one condition and must agree: either every
    pattern in the group matches or none does. That keeps a typo loud, which is
    the whole point of rejecting a pattern that matches nothing.
    """

    style: ParallelStyle
    group: str


def _adapter_name(adapter: object) -> str:
    return getattr(adapter, "__name__", None) or type(adapter).__name__


def sequence_parallel_enabled(adapter: object, sequence_parallel: Optional[bool]) -> bool:
    """Resolve the configured ``parallelism.sequence_parallel`` against an adapter.

    ``True`` and ``False`` decide outright. Unset defers to the adapter's
    ``sequence_parallel_by_default`` policy. This helper resolves the flag; it
    does not validate plan coverage or realized token geometry. Plan resolution
    and model-owned input boundaries perform those checks separately.

    Args:
        adapter: The adapter class or instance.
        sequence_parallel: The configured tri-state.

    Returns:
        Whether the adapter's sequence-parallel overlay applies.
    """
    if sequence_parallel is not None:
        if not isinstance(sequence_parallel, bool):
            raise TypeError(
                "expected parallelism.sequence_parallel to be a bool or None, received "
                f"{type(sequence_parallel).__name__}: {sequence_parallel!r}"
            )
        return sequence_parallel
    by_default = getattr(adapter, "sequence_parallel_by_default", None)
    if not isinstance(by_default, bool):
        raise TypeError(
            f"expected {_adapter_name(adapter)}.sequence_parallel_by_default to be a bool, "
            f"received {type(by_default).__name__}: {by_default!r}"
        )
    return by_default


def resolve_parallel_plan(
    adapter: object, *, sequence_parallel: Optional[bool]
) -> Dict[str, ParallelStyle]:
    """Merge an adapter's tensor- and sequence-parallel plans under one policy.

    One implementation for two callers that need different things from the same
    rule: the pre-load validator calls it for its *effect*, to reject a bad
    combination before a 14B download, and the applier calls it for its *value*.

    Override semantics, not union: sequence parallelism *refines* the
    tensor-parallel plan, so an entry here replaces the one it restates -- an
    out-projection that reduce-scatters instead of all-reducing is the same
    module, not a second one.

    Whether the overlay applies is :func:`sequence_parallel_enabled`. Asking for
    it by name on an adapter without one is refused, and so is an adapter that
    enables it by default without declaring one, which is the adapter's bug.

    Args:
        adapter: The adapter class or instance owning the plans.
        sequence_parallel: The configured ``parallelism.sequence_parallel``.

    Returns:
        The merged plan.

    Raises:
        ValueError: If the overlay applies and the adapter declares none.
    """
    plan = dict(adapter.tensor_parallel_plan())
    name = _adapter_name(adapter)
    overlay = adapter.sequence_parallel_plan()
    if not sequence_parallel_enabled(adapter, sequence_parallel):
        if sequence_parallel is None and overlay:
            logger.info(
                "%s declares a sequence-parallel plan but does not enable it by default; "
                "set parallelism.sequence_parallel: true to shard the token axis as well.",
                name,
            )
        return plan

    if not overlay:
        if sequence_parallel:
            raise ValueError(
                f"parallelism.sequence_parallel=true was requested but {name} declares no "
                "sequence-parallel plan. Sharding the token axis needs per-architecture "
                "entries -- which norms live in the sharded region, where attention gathers "
                "back, and which parameters broadcast over tokens -- so there is no safe "
                "default. Unset the flag to run with tensor parallelism alone."
            )
        raise ValueError(
            f"{name}.sequence_parallel_by_default is True but {name} declares no "
            "sequence-parallel plan; declare one or turn the default off."
        )
    plan.update(overlay)
    logger.info(
        "Sequence parallelism overlays %d entries on the tensor-parallel plan", len(overlay)
    )
    return plan


def _resolve_patterns(
    root: nn.Module, styles: Mapping[str, ParallelStyle]
) -> Tuple[Dict[str, ParallelStyle], Dict[str, bool]]:
    """Resolve ``*`` wildcards in plan keys against the live module tree.

    Plans are written as ``transformer_blocks.*.attn.to_q``, mirroring the
    notation diffusers uses; ``parallelize_module`` wants concrete paths.

    Matching runs over every module, not only parameter-owning ones: once PEFT
    has injected LoRA, a planned ``Linear`` becomes a ``lora.Linear`` wrapper
    that owns no parameter directly, so restricting the search to parameter
    owners would make a valid plan look like it matched nothing.

    An exact path overrides a wildcard that also reaches it, whatever order the
    plan listed them in. Sequence parallelism needs that: every block takes the
    same treatment through ``blocks.*`` while the first one additionally
    scatters the token axis, and leaving the winner to dict ordering would make
    a correct plan depend on how it was written.

    Knows nothing about which entries are allowed to match nothing -- it reports
    what each pattern found and lets :func:`_reject_unmatched` decide. That
    split is why neither half has to be read to change the other.

    Returns:
        The resolved path-to-style mapping, and whether each original pattern
        matched at least one module.
    """
    # `named_modules()` names the root `""`, which is dropped here so a wildcard
    # cannot match it -- but `""` is a legitimate exact key: Wan and LTX-2 keep
    # an AdaLN table as a bare parameter on the transformer root, and
    # `build_total_plan` completes it under that name.
    candidates = [name for name, _ in root.named_modules() if name]
    reachable = set(candidates) | {""}
    exact = {path: style for path, style in styles.items() if "*" not in path}
    resolved: Dict[str, ParallelStyle] = {}
    matched: Dict[str, bool] = {}

    for pattern, style in styles.items():
        if "*" not in pattern:
            matched[pattern] = pattern in reachable
            continue
        prefix, _, suffix = pattern.partition("*")
        suffix = suffix.lstrip(".")
        hits = [
            name
            for name in candidates
            if name.startswith(prefix) and name.endswith(suffix)
            # `*` stands for exactly one path segment.
            and (middle := name[len(prefix) : len(name) - len(suffix)].strip("."))
            and "." not in middle
        ]
        matched[pattern] = bool(hits)
        for name in hits:
            # An exact entry for the same path wins, but the wildcard still
            # counts as matched so it is not reported unused.
            if name not in exact:
                resolved[name] = style

    resolved.update(exact)
    return resolved, matched


def _reject_unmatched(
    plan: Mapping[str, ParallelStyle], matched: Mapping[str, bool]
) -> Iterable[str]:
    """Rule on the patterns that found nothing, and yield the paths to drop.

    A pattern matching no module normally means the plan names a layer this
    architecture does not have -- a typo, or a renamed layer -- and is an error.
    A :class:`ConditionalEntry` is the exception: its modules exist only in some
    checkpoints of the family its adapter serves. Its ``group`` carries the
    condition, and the members of a group must agree, so a misspelling inside an
    otherwise-present group is still rejected.

    Exact paths and wildcard patterns follow the same rule. This prevents
    an unmatched exact entry from being silently dropped by the applier.

    Raises:
        ValueError: If an unconditional pattern matched nothing, or one
            conditional group matched only partly.
    """
    groups: Dict[str, Dict[str, bool]] = {}
    for path, entry in plan.items():
        if isinstance(entry, ConditionalEntry):
            groups.setdefault(entry.group, {})[path] = matched[path]
        elif not matched[path]:
            raise ValueError(
                f"tensor-parallel plan pattern {path!r} matched no module. The plan "
                f"must name modules this architecture has, or the layer it was written "
                f"for has been renamed."
            )

    for group, outcomes in groups.items():
        if all(outcomes.values()):
            continue
        if any(outcomes.values()):
            # Mixed means a typo, not a different checkpoint: the condition the
            # group names holds for the architecture, so every member of it
            # should have been found.
            raise ValueError(
                f"conditional tensor-parallel group {group!r} matched only partly, which "
                f"means one of its paths is misspelled rather than absent from this "
                f"checkpoint: "
                + ", ".join(
                    f"{path!r}={'matched' if hit else 'missing'}" for path, hit in outcomes.items()
                )
            )
        logger.info(
            "Conditional tensor-parallel group %r names no module in this checkpoint and "
            "was skipped: %s",
            group,
            ", ".join(repr(path) for path in outcomes),
        )
        yield from outcomes


def _expand_globs(root: nn.Module, plan: Mapping[str, ParallelStyle]) -> Dict[str, ParallelStyle]:
    """Turn a written plan into concrete paths, rejecting the ones that name nothing.

    Composition of :func:`_resolve_patterns`, which does not know that an entry
    may be allowed to match nothing, and :func:`_reject_unmatched`, which does
    not know how a pattern is matched.
    """
    styles = {
        path: entry.style if isinstance(entry, ConditionalEntry) else entry
        for path, entry in plan.items()
    }
    resolved, matched = _resolve_patterns(root, styles)
    for path in _reject_unmatched(plan, matched):
        resolved.pop(path, None)
    return resolved


def build_total_plan(
    root: nn.Module, shard_plan: Mapping[str, ParallelStyle]
) -> Dict[str, ParallelStyle]:
    """Complete a sharding plan so that it covers every parameter-owning module.

    Args:
        root: The module that will be parallelized.
        shard_plan: Architecture-specific styles, optionally using ``*`` to stand
            for one path segment.

    Returns:
        A plan naming every parameter-owning module in ``root``, with
        :class:`ReplicateParallel` filled in wherever ``shard_plan`` is silent.
    """
    resolved = _expand_globs(root, shard_plan)
    total = derive_lora_plan(root, resolved)
    for module_name in _modules_owning_parameters(root):
        existing = total.get(module_name)
        if existing is None:
            # Including `""`, the root: Wan and LTX-2 keep an AdaLN table there
            # as a bare parameter, and skipping it left that parameter a plain
            # tensor, which `apply_tensor_parallel` then rejects.
            total[module_name] = ReplicateParallel()
        elif isinstance(
            existing, (*_ACTIVATION_ONLY_STYLES, SequenceBroadcastOutput, _RegionEntry)
        ):
            # An activation transform distributes nothing, so a boundary module
            # that owns parameters still needs a parameter style beside it.
            total[module_name] = ComposedParallelStyle(ReplicateParallel(), existing)
    return total
