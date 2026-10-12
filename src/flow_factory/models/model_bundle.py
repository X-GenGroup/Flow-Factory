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

# src/flow_factory/models/model_bundle.py
"""Single-root module bundle for distributed training.

`ModelBundle` wraps several trainable / frozen-shardable modules (e.g. one or
more transformers) as ONE ``nn.Module`` so that
``accelerator.prepare`` wraps a single root. This is required by DeepSpeed (one
engine per ``prepare``) and FSDP2 (one root) — they cannot wrap multiple models.
It also lets a frozen member (e.g. Wan2.2's inactive transformer) be FSDP-sharded
for memory while only the ``requires_grad`` subset is trained.

`RoutedComponentProxy` is the transparent proxy installed as an adapter component
after ``prepare``: calling it routes the forward through the (wrapped) bundle
root — which is what drives DDP's gradient reducer / FSDP's param all-gather /
the DeepSpeed engine — while attribute access (``.config``, ``.dtype``,
``.parameters()``, ``cache_context``, ``disable_adapter`` ...) delegates to the
inner module so existing adapter code is unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

from .variants import ComponentVariantRegistry


class ModelBundle(nn.Module):
    """Bundle of named submodules dispatched through a single ``forward``.

    The single ``nn.Module`` passed to ``accelerator.prepare`` so the whole
    distributed wrapper (DDP / FSDP / DeepSpeed) has exactly one root. Members
    may be trainable or frozen-but-sharded; only the ``requires_grad`` subset is
    optimized (the trainer builds optimizer groups from the variant registry's parameters).

    Args:
        members: Mapping from component name (e.g. ``"transformer"``,
            ``"transformer_2"``) to its ``nn.Module``. Must be non-empty.
    """

    def __init__(self, members: Dict[str, nn.Module]):
        super().__init__()
        if not members:
            raise ValueError(
                "ModelBundle requires at least one member module, got an empty mapping."
            )
        for name, module in members.items():
            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"ModelBundle member '{name}' must be an nn.Module, "
                    f"got {type(module).__name__}."
                )
        self.members = nn.ModuleDict(members)

    @property
    def _no_split_modules(self):
        """Aggregate member block metadata for Accelerate's FSDP wrap policy.

        accelerate's ``set_auto_wrap_policy`` reads ``getattr(root, "_no_split_modules")``
        off the single root passed to ``prepare`` (this bundle). Without this the
        list is empty, ``transformer_layer_cls`` becomes an empty set, and the whole
        bundle is wrapped as ONE FSDP unit -> one monolithic flat param (the full
        unsharded model materialized on every rank) -> OOM at init for large models
        (e.g. Wan2.2 A14B: ~53GB flat param). We surface the block-class *names* that
        underlying model classes already declare. Older models use
        ``_no_split_modules`` (for example ``WanTransformerBlock``), while newer
        Diffusers models may expose only ``_repeated_blocks`` (for example
        ``LTX2VideoTransformerBlock``). Accelerate then resolves each name to its
        class over this root's submodule tree and wraps per block. Returns ``None``
        when no member declares either form of metadata.
        """
        names: list[str] = []
        # Walk members' submodule tree (NOT self, to avoid recursing on this property);
        # preserve the legacy no-split declaration when present and otherwise use
        # Diffusers' repeated-block declaration.
        for module in self.members.modules():
            nsm = getattr(module, "_no_split_modules", None)
            if not nsm:
                nsm = getattr(module, "_repeated_blocks", None)
            if nsm:
                names.extend(nsm)
        deduped = list(dict.fromkeys(names))
        return deduped or None

    def forward(self, component_name: str, /, *args, **kwargs):
        """Dispatch the call to ``members[component_name]``.

        ``component_name`` is positional-only so it never collides with a
        member's own keyword arguments.

        Args:
            component_name: Name of the member to invoke.
            *args: Positional args forwarded to the member.
            **kwargs: Keyword args forwarded to the member.

        Returns:
            The member's forward output.

        Raises:
            KeyError: If ``component_name`` is not a member of this bundle.
        """
        if component_name not in self.members:
            raise KeyError(
                f"ModelBundle has no member '{component_name}'. "
                f"Available members: {list(self.members.keys())}."
            )
        return self.members[component_name](*args, **kwargs)


class RoutedComponentProxy:
    """Callable proxy that routes a component's forward through a `ModelBundle`.

    Installed as an adapter component runtime override after ``prepare``. Calling it
    invokes the prepared bundle root (driving grad-sync / param-gather), while every
    attribute access falls through to the inner module, so adapter code that does
    ``self.transformer(...)`` / ``self.transformer.config`` keeps working unchanged.

    Not an ``nn.Module`` on purpose: it must not register the inner module as a
    submodule (that would double-count parameters under the bundle root). Use
    ``BaseAdapter._unwrap`` to recover the inner module.

    Args:
        bundle: Prepared single-root model bundle.
        canonical_name: Canonical adapter component name.
        variant_registry: Registry resolving active role routes. A legacy static inner
            module remains accepted for existing direct proxy construction.
        inner_members: Unwrapped bundle members keyed by route name.
    """

    def __init__(
        self,
        bundle: nn.Module,
        canonical_name: str,
        variant_registry: ComponentVariantRegistry | nn.Module,
        inner_members: Mapping[str, nn.Module] | None = None,
    ) -> None:
        # Bypass __setattr__/__getattr__ for the proxy's own fields.
        object.__setattr__(self, "_bundle", bundle)
        object.__setattr__(self, "_canonical_name", canonical_name)
        if isinstance(variant_registry, ComponentVariantRegistry):
            if inner_members is None:
                raise TypeError(
                    "expected inner_members mapping for role-aware RoutedComponentProxy "
                    f"component {canonical_name!r}, received None"
                )
            object.__setattr__(self, "_variant_registry", variant_registry)
            object.__setattr__(self, "_inner_members", dict(inner_members))
            return
        if not isinstance(variant_registry, nn.Module):
            raise TypeError(
                "expected ComponentVariantRegistry or legacy torch.nn.Module inner component for "
                f"RoutedComponentProxy component {canonical_name!r}, received "
                f"{type(variant_registry).__name__}: {variant_registry!r}"
            )
        if inner_members is not None:
            raise TypeError(
                "expected inner_members=None with legacy RoutedComponentProxy inner module "
                f"for component {canonical_name!r}, received {inner_members!r}"
            )
        object.__setattr__(self, "_variant_registry", None)
        object.__setattr__(self, "_inner_members", {canonical_name: variant_registry})

    def _active_route(self) -> str:
        """Resolve the active role's bundle route for this canonical component."""
        registry = object.__getattribute__(self, "_variant_registry")
        canonical_name = object.__getattribute__(self, "_canonical_name")
        if registry is None:
            return canonical_name
        active_variant = registry.active_variant
        variant_spec = registry.get_spec(active_variant)
        if canonical_name not in variant_spec.component_routes:
            return canonical_name
        return registry.resolve_route(active_variant, canonical_name)

    @property
    def inner(self) -> nn.Module:
        """Return the active role's unwrapped inner module."""
        route = self._active_route()
        inner_members = object.__getattribute__(self, "_inner_members")
        if route not in inner_members:
            raise KeyError(
                f"RoutedComponentProxy route {route!r} for canonical component "
                f"{object.__getattribute__(self, '_canonical_name')!r} is missing from "
                f"inner bundle members {tuple(inner_members)!r}"
            )
        return inner_members[route]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        bundle = object.__getattribute__(self, "_bundle")
        return bundle(self._active_route(), *args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        # __getattr__ only fires when normal lookup misses, so the proxy's own
        # fields never reach here.
        return getattr(self.inner, attr)

    def __repr__(self) -> str:
        canonical_name = object.__getattribute__(self, "_canonical_name")
        return (
            f"RoutedComponentProxy(canonical_name={canonical_name!r}, "
            f"route={self._active_route()!r}, inner={type(self.inner).__name__})"
        )
