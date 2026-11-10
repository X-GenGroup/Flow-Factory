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

import os
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from huggingface_hub import snapshot_download
from torch import nn

from ...loading.checkpoint_slices import construct_from_checkpoint, requests_component
from ...utils.logger_utils import setup_logger
from ..precision import DTypePolicy, build_component_load_dtype_kwargs
from .abc import ComponentRuntime

logger = setup_logger(__name__)


class ModularPipelineRuntime(ComponentRuntime):
    """Manage a lazy modular pipeline without importing model-specific classes."""

    def __init__(
        self,
        pipeline: Any,
        *,
        component_load_dtype_manifest: DTypePolicy = None,
        component_load_dtype_overrides: DTypePolicy = None,
        checkpoint_backed_request: Sequence[str] = (),
    ) -> None:
        super().__init__(pipeline)
        self.component_load_dtype_manifest = component_load_dtype_manifest
        self.component_load_dtype_overrides = component_load_dtype_overrides
        self.checkpoint_backed_request = tuple(checkpoint_backed_request)

    @classmethod
    def from_adapter(cls, adapter: Any, pipeline: Any) -> "ModularPipelineRuntime":
        """Build a modular runtime from the adapter's resolved load policy."""
        return cls(
            pipeline,
            component_load_dtype_manifest=adapter._component_load_dtype_manifest,
            component_load_dtype_overrides=adapter._component_load_dtype_overrides,
            checkpoint_backed_request=adapter._checkpoint_backed_request(),
        )

    @property
    def canonical_components(self) -> Mapping[str, Any]:
        """Return specs declared through pinned ModularPipeline's public API.

        Raises:
            TypeError: If the public spec API has an incompatible shape.
        """
        pretrained_component_names = getattr(self.pipeline, "pretrained_component_names", None)
        config_component_names = getattr(self.pipeline, "config_component_names", None)
        if not isinstance(pretrained_component_names, list) or not all(
            isinstance(name, str) and name for name in pretrained_component_names
        ):
            raise TypeError(
                "Modular pipeline pretrained_component_names must be a list of non-empty strings, "
                f"got {type(pretrained_component_names).__name__}: "
                f"{pretrained_component_names!r}."
            )
        if not isinstance(config_component_names, list) or not all(
            isinstance(name, str) and name for name in config_component_names
        ):
            raise TypeError(
                "Modular pipeline config_component_names must be a list of non-empty strings, "
                f"got {type(config_component_names).__name__}: {config_component_names!r}."
            )
        get_component_spec = getattr(self.pipeline, "get_component_spec", None)
        if not callable(get_component_spec):
            raise TypeError(
                "Modular pipeline must provide callable get_component_spec(name), "
                f"got {get_component_spec!r}."
            )

        declared_names = list(dict.fromkeys([*pretrained_component_names, *config_component_names]))
        return {name: get_component_spec(name) for name in declared_names}

    def _get_materialized_component(self, name: str) -> Any:
        return getattr(self.pipeline, name, None)

    def _materialize_components(self, names: List[str]) -> None:
        unloaded = [name for name in names if self._get_materialized_component(name) is None]
        if unloaded:
            load_components = getattr(self.pipeline, "load_components", None)
            if not callable(load_components):
                raise RuntimeError(
                    "Modular pipeline cannot materialize required components because "
                    f"`load_components(names=...)` is unavailable; expected={unloaded}, "
                    f"received={self._materialized_component_names()}."
                )
            manifest_policy = self.component_load_dtype_manifest
            user_policy = self.component_load_dtype_overrides
            load_kwargs = build_component_load_dtype_kwargs(
                user_policy=user_policy,
                manifest_policy=manifest_policy,
                component_names=self.declared_component_names,
                transformer_names=self.transformer_names,
                text_encoder_names=self.text_encoder_names,
                requested_names=unloaded,
            )
            for name in unloaded:
                module = self._checkpoint_backed_component(name, load_kwargs.get("dtype"))
                if module is not None:
                    self.pipeline.update_components(**{name: module})
            unloaded = [name for name in unloaded if self._get_materialized_component(name) is None]
            if unloaded:
                load_components(names=unloaded, **load_kwargs)

        missing = [name for name in names if self._get_materialized_component(name) is None]
        if missing:
            raise RuntimeError(
                "Modular component materialization failed; "
                f"expected={missing}, received_specs={self.declared_component_names}, "
                f"materialized={self._materialized_component_names()}."
            )

    def _checkpoint_backed_component(self, name: str, dtype: Any) -> Optional[nn.Module]:
        """Build one requested target from its spec without reading weights, or ``None``."""
        if not requests_component(self.checkpoint_backed_request, name):
            return None
        spec = self.pipeline.get_component_spec(name)
        type_hint = getattr(spec, "type_hint", None)
        repository = getattr(spec, "pretrained_model_name_or_path", None)
        if type_hint is None or repository is None or not issubclass(type_hint, nn.Module):
            return None
        subfolder = getattr(spec, "subfolder", None) or ""
        root = (
            Path(repository)
            if os.path.isdir(repository)
            else Path(
                snapshot_download(
                    repository,
                    revision=getattr(spec, "revision", None),
                    allow_patterns=[f"{subfolder}/*" if subfolder else "*"],
                )
            )
        )
        component_dtype = (
            dtype.get(name, dtype.get("default")) if isinstance(dtype, Mapping) else dtype
        )
        module, reason = construct_from_checkpoint(
            type_hint, root / subfolder, component_dtype, variant=getattr(spec, "variant", None)
        )
        if module is None:
            logger.warning("Loading target component %s natively: %s.", name, reason)
            return None
        module._diffusers_load_id = spec.load_id
        logger.info(
            "Target component %s stays in its checkpoint until sharded materialization", name
        )
        return module
