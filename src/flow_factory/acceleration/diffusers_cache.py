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

# src/flow_factory/acceleration/diffusers_cache.py
"""Lossy rollout-only feature caching via diffusers' native ``CacheMixin``.

Zero extra dependency: diffusers transformers already inherit ``enable_cache`` /
``disable_cache`` (they are ``CacheMixin`` models — the same machinery behind the
``cache_context(...)`` calls Flow-Factory adapters already use). This accelerator
enables a cache policy for the duration of one rollout epoch and tears it down on
exit so the Stage-6 training forward stays exact.

Only valid in the rollout slot of a decoupled / distillation trainer — the
paradigm validator enforces this (``constraints.md`` #7).
"""

import math
from contextlib import contextmanager
from numbers import Real
from typing import TYPE_CHECKING, Iterator, List

from diffusers.hooks import (
    FasterCacheConfig,
    FirstBlockCacheConfig,
    MagCacheConfig,
    PyramidAttentionBroadcastConfig,
    TaylorSeerCacheConfig,
)

from ..utils.logger_utils import setup_logger
from .abc import BaseAccelerator

if TYPE_CHECKING:
    import torch

    from ..models.abc import BaseAdapter

logger = setup_logger(__name__)

# Map the user-facing ``policy`` string to its diffusers config class.
_POLICY_CONFIGS = {
    "first_block": FirstBlockCacheConfig,
    "faster": FasterCacheConfig,
    "pyramid": PyramidAttentionBroadcastConfig,
    "taylorseer": TaylorSeerCacheConfig,
    "magcache": MagCacheConfig,
}


class DiffusersCacheAccelerator(BaseAccelerator):
    """Enable a diffusers cache policy on every transformer during rollout.

    Lossy and rollout-scoped. The default policy is ``first_block`` (FirstBlockCache,
    aka FBCache) which is robust across models and needs only a single ``threshold``.

    Args:
        policy: One of ``first_block`` / ``faster`` / ``pyramid`` / ``taylorseer`` /
            ``magcache``. Defaults to ``first_block``.
        **params: All remaining entry ``params``, forwarded verbatim to the selected
            diffusers config class (e.g. ``threshold`` for FirstBlockCache).

    Note:
        Caching reuses block outputs across denoising steps, so the per-step
        ``cache_context`` the adapter opens around its transformer call must persist
        cache state across the loop. Adapters opt in with ``supports_diffusers_cache``
        once every transformer forward branch runs inside ``transformer.cache_context(...)``
        (Qwen-Image, Wan2, LTX2, FLUX.2-Klein). MiniMax H3 supplies a policy-specific
        compatibility hook and accepts FirstBlockCache only. Verify the reward distribution
        before/after enabling any lossy cache policy.
    """

    safety = "lossy"
    stage = "rollout"

    def _resolve_policy(self) -> str:
        policy = self.params.get("policy", "first_block")
        if not isinstance(policy, str) or policy not in _POLICY_CONFIGS:
            expected = sorted(_POLICY_CONFIGS)
            raise ValueError(
                f"DiffusersCacheAccelerator: unknown policy={policy!r}; expected one of "
                f"{expected}."
            )
        return policy

    def _build_config(self, policy: str):
        params = dict(self.params)
        params.pop("policy", None)
        if policy == "first_block":
            threshold = params.get("threshold", 0.05)
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, Real)
                or not math.isfinite(float(threshold))
                or threshold < 0
            ):
                raise ValueError(
                    "DiffusersCacheAccelerator: policy='first_block' requires threshold "
                    "to be a finite non-negative real number, received "
                    f"{type(threshold).__name__}: {threshold!r}."
                )
        config_cls = _POLICY_CONFIGS[policy]
        try:
            return config_cls(**params)
        except TypeError as e:
            raise ValueError(
                f"DiffusersCacheAccelerator: invalid parameters {sorted(params)} for policy "
                f"'{policy}' ({config_cls.__name__}): {e}"
            ) from e

    @contextmanager
    def rollout_context(self, adapter: "BaseAdapter") -> Iterator[None]:
        policy = self._resolve_policy()
        if not adapter.supports_diffusers_cache:
            raise ValueError(
                f"DiffusersCacheAccelerator: adapter {type(adapter).__name__} does not support "
                "diffusers feature caching because not every transformer forward branch runs "
                "inside `cache_context`. Remove the accelerator or add complete adapter support."
            )
        supported_policies = adapter.supported_diffusers_cache_policies
        if supported_policies is not None and policy not in supported_policies:
            raise ValueError(
                f"DiffusersCacheAccelerator: adapter {type(adapter).__name__} does not support "
                f"policy={policy!r}; supported policies: {sorted(supported_policies)}."
            )

        transformer_names = adapter.transformer_names
        if not transformer_names:
            raise ValueError("DiffusersCacheAccelerator: adapter exposes no transformer to cache.")
        configs = [self._build_config(policy) for _ in transformer_names]

        transformers = []
        for name, config in zip(transformer_names, configs):
            transformer = adapter.get_component(name)
            if not callable(getattr(transformer, "enable_cache", None)):
                raise ValueError(
                    f"DiffusersCacheAccelerator: component '{name}' is not a diffusers "
                    "CacheMixin (no callable `enable_cache`); use a different accelerator."
                )
            transformers.append((name, transformer, config))
        for name, transformer, _ in transformers:
            adapter.prepare_diffusers_cache(policy, name, transformer)

        enabled: List["torch.nn.Module"] = []
        try:
            for name, transformer, config in transformers:
                # Defensive: clear any stale cache left enabled by a prior epoch.
                if getattr(transformer, "is_cache_enabled", False):
                    transformer.disable_cache()
                transformer.enable_cache(config)
                enabled.append(transformer)
                # diffusers' HookRegistry caches its child-registry list the first time
                # a `cache_context` sets a context. If a `cache_context` ran while the
                # cache was DISABLED (e.g. during eval, where the adapter still opens
                # `transformer.cache_context(...)`), that cache was populated EMPTY --
                # before the per-block cache hooks above existed -- so a later
                # `_set_context` never reaches the freshly added block hooks and the
                # block forward raises "No context is set". Invalidate the stale cache on
                # the unwrapped module (the exact object the adapter's `cache_context`
                # targets) so the next context build rediscovers the new hooks. Safe/no-op
                # when no such registry exists yet (the common no-eval path).
                unwrapped = adapter.get_component_unwrapped(name)
                cache_hook = getattr(unwrapped, "_diffusers_hook", None)
                if cache_hook is not None:
                    cache_hook._child_registries_cache = None
                if adapter.accelerator.is_main_process:
                    logger.info(
                        "DiffusersCacheAccelerator: policy='%s' enabled for '%s'.",
                        policy,
                        name,
                    )
            yield
        finally:
            for transformer in enabled:
                transformer.disable_cache()
